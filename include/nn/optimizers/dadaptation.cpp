#include "dadaptation.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

DAdaptAdam::DAdaptAdam(double lr_, double b1, double b2, double eps, double wd,
                       double d0_, double growth, bool decouple_, bool use_bc)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      weight_decay(wd), d0(d0_), growth_rate(growth),
      decouple(decouple_), use_bias_correction(use_bc),
      k(1), d_(d0_), numerator_weighted_(0.0) {
    // Use setters to validate; they will throw on invalid inputs.
    set_lr(lr_);
    set_beta1(b1);
    set_beta2(b2);
    set_epsilon(eps);
    set_weight_decay(wd);
    set_d0(d0_);
    set_growth_rate(growth);
}

void DAdaptAdam::set_lr(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("DAdaptAdam: lr must be > 0");
    lr = v;
    Optimizer::lr = v;
}

void DAdaptAdam::set_beta1(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("DAdaptAdam: beta1 must be in [0,1)");
    beta1 = v;
}

void DAdaptAdam::set_beta2(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("DAdaptAdam: beta2 must be in [0,1)");
    beta2 = v;
}

void DAdaptAdam::set_epsilon(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("DAdaptAdam: epsilon must be > 0");
    epsilon = v;
}

void DAdaptAdam::set_weight_decay(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("DAdaptAdam: weight_decay must be >= 0");
    weight_decay = v;
}

void DAdaptAdam::set_d0(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("DAdaptAdam: d0 must be > 0");
    d0 = v;
}

void DAdaptAdam::set_growth_rate(double v) {
    if (!(v >= 1.0)) throw std::invalid_argument("DAdaptAdam: growth_rate must be >= 1 (1.0 = no growth, inf = unrestricted)");
    growth_rate = v;
}

void DAdaptAdam::set_decouple(bool v) {
    decouple = v;
}

void DAdaptAdam::set_use_bias_correction(bool v) {
    use_bias_correction = v;
}

void DAdaptAdam::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<ParamState> vec;
    vec.reserve(params.size());
    for (const Tensor* p : params) {
        ParamState st;
        st.m = Tensor(p->rows, p->cols); st.m.fill(0.0);
        st.v = Tensor(p->rows, p->cols); st.v.fill(0.0);
        st.s = Tensor(p->rows, p->cols); st.s.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

void DAdaptAdam::step(Model& model) {
    // Bias-correction factor (PyTorch convention)
    // bc = sqrt(1 - beta2^k) / (1 - beta1^k); if disabled, bc = 1.
    double bias_correction = 1.0;
    if (use_bias_correction) {
        double bc_num = std::sqrt(1.0 - std::pow(beta2, k));
        double bc_den = 1.0 - std::pow(beta1, k);
        bias_correction = bc_num / bc_den;
    }
    // Effective scale: d_effective = d_ * lr * bias_correction
    const double dlr = d_ * lr * bias_correction;

    // sqrt(beta2) is the EMA coefficient for s and r (NOT beta2 itself)
    const double sqrt_beta2 = std::sqrt(beta2);
    const double one_minus_sqrt_beta2 = 1.0 - sqrt_beta2;

    // =====================================================================
    // PHASE 1: Accumulator loop (PyTorch: first pass over params)
    //   - Decoupled WD applied first (mutates param directly)
    //   - denom = sqrt(OLD v) + eps   (use v as it was BEFORE this step's update)
    //   - numerator_acum += dlr * <g_eff, OLD_s / denom>
    //   - update m, v, s (in this order — matches PyTorch in-place mutations)
    //   - sk_l1 += ||NEW s||_1         (uses s AFTER this step's update)
    // =====================================================================
    double numerator_acum = 0.0;
    double sk_l1 = 0.0;

    // We need to remember each param's OLD v value for the accumulator
    // calculation (since v is mutated before we read denom again).
    // PyTorch achieves this with two separate passes; we mirror it by
    // capturing the OLD v in a separate snapshot per param.

    struct ParamSnapshot {
        Tensor* param;
        Tensor* grad;
        ParamState* st;
        Tensor old_v;  // snapshot of v BEFORE the update
        Tensor new_v_after_update;  // for the apply-pass denom
    };

    std::vector<ParamSnapshot> snapshots;

    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::logic_error("DAdaptAdam: param/grad count mismatch");
        }
        ensure_state(layer_ptr, params);
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            if (grad->rows != param->rows || grad->cols != param->cols) {
                throw std::logic_error("DAdaptAdam: param/grad shape mismatch");
            }
            ParamState& st = st_vec[i];

            // Decoupled weight decay: applied BEFORE the gradient step, scales by d*lr*wd
            if (decouple && weight_decay > 0.0) {
                double scale = 1.0 - lr * d_ * weight_decay;
                for (size_t r = 0; r < param->rows; ++r)
                    for (size_t c = 0; c < param->cols; ++c)
                        (*param)[r][c] *= scale;
            }

            // Snapshot OLD v for the accumulator-loop denom
            Tensor old_v_snapshot = st.v;

            // Compute g_eff (coupled WD adds wd*param to gradient — matches PyTorch in-place)
            // First pass: numerator_acum using OLD v and OLD s
            double local_num_acum = 0.0;
            double local_sk_l1 = 0.0;

            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g_raw = (*grad)[r][c];
                    double g_eff = g_raw;
                    if (!decouple && weight_decay > 0.0) {
                        g_eff = g_raw + weight_decay * (*param)[r][c];
                    }
                    // OLD denom (matches PyTorch first loop)
                    double denom_old = std::sqrt(old_v_snapshot[r][c]) + epsilon;
                    double s_old = st.s[r][c];
                    // numerator accumulation: dlr * <g_eff, OLD_s / OLD_denom>
                    local_num_acum += dlr * g_eff * (s_old / denom_old);

                    // Now update m, v, s in-place (matches PyTorch first loop)
                    st.m[r][c] = beta1 * st.m[r][c] + (1.0 - beta1) * dlr * g_eff;
                    st.v[r][c] = beta2 * st.v[r][c] + (1.0 - beta2) * g_eff * g_eff;
                    double s_new = sqrt_beta2 * s_old + one_minus_sqrt_beta2 * dlr * g_eff;
                    st.s[r][c] = s_new;

                    // sk_l1 contribution from NEW s
                    local_sk_l1 += std::abs(s_new);
                }
            }

            numerator_acum += local_num_acum;
            sk_l1 += local_sk_l1;

            // Save snapshot for the apply loop (use NEW v for the apply-pass denom)
            snapshots.push_back({param, grad, &st, old_v_snapshot, st.v});
        }
    }

    // If nothing moved (all gradients zero), return early.
    if (sk_l1 == 0.0) {
        ++k;
        return;
    }

    // =====================================================================
    // PHASE 2: Global D update
    //   r = sqrt(beta2) * r_prev + (1-sqrt(beta2)) * numerator_acum
    //   d_hat = r / ((1 - sqrt(beta2)) * sk_l1)
    //   d = max(d, min(d_hat, d * growth_rate))
    // =====================================================================
    numerator_weighted_ = sqrt_beta2 * numerator_weighted_ + one_minus_sqrt_beta2 * numerator_acum;
    double d_hat = numerator_weighted_ / (one_minus_sqrt_beta2 * sk_l1);
    double d_capped = std::min(d_hat, d_ * growth_rate);
    d_ = std::max(d_, d_capped);

    // =====================================================================
    // PHASE 3: Apply loop (PyTorch: second pass over params)
    //   - denom = sqrt(NEW v) + eps   (use v AFTER the update)
    //   - param -= m / denom
    // =====================================================================
    for (auto& snap : snapshots) {
        Tensor* param = snap.param;
        ParamState& st = *snap.st;
        Tensor& new_v = snap.new_v_after_update;
        for (size_t r = 0; r < param->rows; ++r) {
            for (size_t c = 0; c < param->cols; ++c) {
                double denom_new = std::sqrt(new_v[r][c]) + epsilon;
                (*param)[r][c] -= st.m[r][c] / denom_new;
            }
        }
    }

    // Zero gradients
    for (auto& layer : model.layers) {
        layer->zero_grad();
    }

    ++k;
}

bool DAdaptAdam::has_state(void* layer_ptr) const {
    return state_.find(layer_ptr) != state_.end();
}

size_t DAdaptAdam::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    return it == state_.end() ? 0 : it->second.size();
}

const Tensor& DAdaptAdam::get_m(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).m;
}

const Tensor& DAdaptAdam::get_v(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).v;
}

const Tensor& DAdaptAdam::get_s(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).s;
}