#include "prodigy.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
Prodigy::Prodigy(double lr_, double b1, double b2, double eps,
                 double wd, double d0_, double d_coef_,
                 double growth, bool decouple_,
                 bool use_bc, bool safeguard_warmup_,
                 double beta3_)
    : lr(lr_),
      beta1(b1),
      beta2(b2),
      beta3(beta3_ < 0.0 ? std::sqrt(b2) : beta3_),
      epsilon(eps),
      weight_decay(wd),
      d0(d0_),
      d_coef(d_coef_),
      growth_rate(growth),
      decouple(decouple_),
      use_bias_correction(use_bc),
      safeguard_warmup(safeguard_warmup_),
      k(1),
      d_(d0_),
      d_max_(d0_),
      d_numerator_(0.0),
      d_hat_(0.0) {
    validate(lr_, b1, b2, eps, d0_, d_coef_, growth);
}

// ----------------------------------------------------------------------------
// Validation
// ----------------------------------------------------------------------------
void Prodigy::validate(double lr_, double b1, double b2, double eps,
                       double d0_, double d_coef_, double growth) {
    if (lr_ <= 0.0)
        throw std::invalid_argument("Prodigy: lr must be > 0");
    if (eps <= 0.0)
        throw std::invalid_argument("Prodigy: eps must be > 0");
    if (b1 < 0.0 || b1 >= 1.0)
        throw std::invalid_argument("Prodigy: beta1 must be in [0, 1)");
    if (b2 < 0.0 || b2 >= 1.0)
        throw std::invalid_argument("Prodigy: beta2 must be in [0, 1)");
    if (d0_ <= 0.0)
        throw std::invalid_argument("Prodigy: d0 must be > 0");
    if (d_coef_ <= 0.0)
        throw std::invalid_argument("Prodigy: d_coef must be > 0");
    if (growth < 1.0)
        throw std::invalid_argument("Prodigy: growth_rate must be >= 1");
}

// ----------------------------------------------------------------------------
// Validated setters
// ----------------------------------------------------------------------------
void Prodigy::set_lr(double v) {
    if (v <= 0.0) throw std::invalid_argument("Prodigy: lr must be > 0");
    lr = v;
}

void Prodigy::set_beta1(double v) {
    if (v < 0.0 || v >= 1.0)
        throw std::invalid_argument("Prodigy: beta1 must be in [0, 1)");
    beta1 = v;
}

void Prodigy::set_beta2(double v) {
    if (v < 0.0 || v >= 1.0)
        throw std::invalid_argument("Prodigy: beta2 must be in [0, 1)");
    beta2 = v;
    // If beta3 was derived from beta2, refresh it.
    if (beta3 == std::sqrt(v + 1e-300) || beta3 == 0.0) {
        beta3 = std::sqrt(v);
    }
}

void Prodigy::set_beta3(double v) {
    if (v < 0.0 || v >= 1.0)
        throw std::invalid_argument("Prodigy: beta3 must be in [0, 1)");
    beta3 = v;
}

void Prodigy::set_epsilon(double v) {
    if (v <= 0.0) throw std::invalid_argument("Prodigy: eps must be > 0");
    epsilon = v;
}

void Prodigy::set_weight_decay(double v) {
    if (v < 0.0) throw std::invalid_argument("Prodigy: weight_decay must be >= 0");
    weight_decay = v;
}

void Prodigy::set_d0(double v) {
    if (v <= 0.0) throw std::invalid_argument("Prodigy: d0 must be > 0");
    d0 = v;
}

void Prodigy::set_d_coef(double v) {
    if (v <= 0.0) throw std::invalid_argument("Prodigy: d_coef must be > 0");
    d_coef = v;
}

void Prodigy::set_growth_rate(double v) {
    if (v < 1.0) throw std::invalid_argument("Prodigy: growth_rate must be >= 1");
    growth_rate = v;
}

void Prodigy::set_decouple(bool v) { decouple = v; }
void Prodigy::set_use_bias_correction(bool v) { use_bias_correction = v; }
void Prodigy::set_safeguard_warmup(bool v) { safeguard_warmup = v; }

// ----------------------------------------------------------------------------
// State management
// ----------------------------------------------------------------------------
void Prodigy::ensure_state(void* layer_ptr,
                            const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<ParamState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        ParamState st;
        st.m = Tensor::zeros(p.rows, p.cols);
        st.v = Tensor::zeros(p.rows, p.cols);
        st.s = Tensor::zeros(p.rows, p.cols);
        st.p0 = p.clone();   // snapshot of initial weights
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

size_t Prodigy::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return 0;
    return it->second.size();
}

const Tensor& Prodigy::get_m(void* layer_ptr, size_t idx) const {
    static const Tensor empty_;
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || idx >= it->second.size()) return empty_;
    return it->second[idx].m;
}

const Tensor& Prodigy::get_v(void* layer_ptr, size_t idx) const {
    static const Tensor empty_;
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || idx >= it->second.size()) return empty_;
    return it->second[idx].v;
}

const Tensor& Prodigy::get_s(void* layer_ptr, size_t idx) const {
    static const Tensor empty_;
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || idx >= it->second.size()) return empty_;
    return it->second[idx].s;
}

const Tensor& Prodigy::get_p0(void* layer_ptr, size_t idx) const {
    static const Tensor empty_;
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || idx >= it->second.size()) return empty_;
    return it->second[idx].p0;
}

// ----------------------------------------------------------------------------
// Pass 1: update m, v, s, accumulate delta_numerator and d_denom.
// ----------------------------------------------------------------------------
Prodigy::PassAccumulators Prodigy::update_param_pass1(
    ParamState& st, const Tensor* param, const Tensor* grad,
    double dlr, double bc_factor) {
    PassAccumulators acc{0.0, 0.0};
    const size_t rows = param->rows;
    const size_t cols = param->cols;

    // Factor for the L1 tracker and the D-estimate numerator contribution.
    // (d/d0) * dlr  (or (d/d0) * d if safeguard_warmup).
    const double d = d_;
    const double d_over_d0 = d / d0;
    const double s_factor = safeguard_warmup ? (d_over_d0 * d)
                                             : (d_over_d0 * dlr);
    const double num_factor = (d_over_d0) * dlr;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double g_ij = (*grad)[i][j];
            const double p_ij = (*param)[i][j];
            const double p0_ij = st.p0[i][j];

            // m_t = β1 * m_{t-1} + d * (1-β1) * g
            st.m[i][j] = beta1 * st.m[i][j] + d * (1.0 - beta1) * g_ij;

            // v_t = β2 * v_{t-1} + d² * (1-β2) * g²
            // (bc_factor compensates for the missing 1/(1-β2^k) bias correction.)
            // Note: Prodigy's PyTorch reference uses `d * d` (no bias correction)
            // for v. We follow the reference exactly; bias_correction only affects
            // dlr (the effective step size), not the v EMA itself.
            st.v[i][j] = beta2 * st.v[i][j] + d * d * (1.0 - beta2) * g_ij * g_ij;

            // s_t = β3 * s_{t-1} + s_factor * g
            st.s[i][j] = beta3 * st.s[i][j] + s_factor * g_ij;

            // D-estimate numerator contribution:
            //   delta_numerator += (d/d0) * dlr * <g, p0 - param>
            acc.delta_numerator += num_factor * g_ij * (p0_ij - p_ij);

            // D-estimate denominator contribution:
            //   d_denom += Σ_ij |s_ij|
            acc.d_denom += std::fabs(st.s[i][j]);
        }
    }
    (void)bc_factor;  // unused (we follow PyTorch reference)
    return acc;
}

// ----------------------------------------------------------------------------
// Pass 2: take Adam step with d*lr pre-factor.
// ----------------------------------------------------------------------------
void Prodigy::update_param_pass2(
    ParamState& st, Tensor* param, const Tensor* grad,
    double dlr) {
    const size_t rows = param->rows;
    const size_t cols = param->cols;
    const double d = d_;
    const double eps_d = d * epsilon;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double std_v = std::sqrt(st.v[i][j] + eps_d);
            const double update = st.m[i][j] / std_v;
            // decoupled weight decay: param *= (1 - dlr * wd)
            if (decouple && weight_decay > 0.0) {
                (*param)[i][j] *= (1.0 - dlr * weight_decay);
            }
            (*param)[i][j] -= dlr * update;
            (void)grad;  // grad isn't needed in pass 2 (m, v already updated)
        }
    }
}

// ----------------------------------------------------------------------------
// step()
// ----------------------------------------------------------------------------
void Prodigy::step(Model& model) {
    // Bias correction factor (only affects dlr when use_bias_correction=true).
    //   bc = sqrt(1 - β2^k) / (1 - β1^k)
    double bias_correction = 1.0;
    if (use_bias_correction) {
        const double bc1 = 1.0 - std::pow(beta1, k);
        const double bc2 = std::sqrt(1.0 - std::pow(beta2, k));
        bias_correction = bc2 / bc1;
    }

    // dlr = d * lr * bias_correction (using the CURRENT d before the update)
    const double dlr = d_ * lr * bias_correction;

    // === Pass 1: per-parameter m, v, s EMAs + delta_numerator / d_denom ===
    double total_delta_numerator = 0.0;
    double total_d_denom = 0.0;
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad  = grads[i];

            // Coupled weight decay (decouple=false): modify grad before EMA.
            if (!decouple && weight_decay > 0.0) {
                for (size_t r = 0; r < param->rows; ++r) {
                    for (size_t c = 0; c < param->cols; ++c) {
                        (*grad)[r][c] += weight_decay * (*param)[r][c];
                    }
                }
            }

            auto acc = update_param_pass1(st_vec[i], param, grad, dlr, bias_correction);
            total_delta_numerator += acc.delta_numerator;
            total_d_denom         += acc.d_denom;
        }
    }

    // === Update global D-estimate ===
    // d_numerator *= β3 (carry over EMA)
    d_numerator_ *= beta3;

    // If d_denom == 0, no parameter moved → no step.
    if (total_d_denom > 0.0) {
        const double global_d_numerator = d_numerator_ + total_delta_numerator;
        d_hat_ = d_coef * global_d_numerator / total_d_denom;

        // Initial bootstrap: at exactly d == d0, allow d to grow to d_hat.
        if (d_ == d0) {
            d_ = std::max(d_, d_hat_);
        }
        d_max_ = std::max(d_max_, d_hat_);
        // growth_rate cap: monotonic but bounded per step.
        const double d_growth_cap = d_ * growth_rate;
        d_ = std::min(d_max_, d_growth_cap);

        // Carry the global numerator forward for the next step's EMA decay.
        d_numerator_ = global_d_numerator;
    }

    // === Pass 2: take Adam step with the NEW d ===
    // Recompute dlr with the new d (bias_correction stays the same).
    const double dlr_new = d_ * lr * bias_correction;
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        if (params.empty()) continue;
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param_pass2(st_vec[i], params[i], grads[i], dlr_new);
        }
        layer->zero_grad();
    }

    ++k;
}
