#include "adafactor.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// =====================================================================
// Adafactor
// =====================================================================
//
// Algorithm per 2-D parameter (W ∈ R^{d1×d2}, d1 ≥ 1, d2 ≥ 1):
//   g_t = grad_t
//   R_t = β2_t * R_{t-1} + (1 - β2_t) * row_mean(g_t ⊙ g_t)  (R ∈ R^{d1×1})
//   C_t = β2_t * C_{t-1} + (1 - β2_t) * col_mean(g_t ⊙ g_t)  (C ∈ R^{1×d2})
//   v_t_ij = R_t[i, 0] * C_t[0, j] / max(ε2, mean(R_t))     (v ∈ R^{d1×d2})
//   v̂_t  = v_t / (1 − ∏_{i=1..t} β2_i)                       (bias correction)
//   u_t  = g_t / sqrt(v̂_t + ε1)
//   if relative_step:  lr_t = max(ε2, RMS(W_{t-1})) / RMS(u_t)
//   else:              lr_t = lr
//   W_t  = W_{t-1} − lr_t * (u_t + wd * W_{t-1})
//
// For 1-D parameters (e.g. (1, n) biases), Adafactor falls back to a
// per-element EMA (Adam-style) with the same β2 schedule. The
// relative-step and weight-decay logic apply uniformly.
//
// Memory: per 2-D parameter, two tensors of sizes d1 and d2 (the
// accumulators) instead of a full d1·d2 second-moment matrix. The
// savings are dramatic for large matrices.

Adafactor::Adafactor(double lr_, double beta2_fixed, double eps1, double eps2,
                     double wd, bool rel_step, bool beta2_sched, double dmax_)
    : lr(lr_), beta2(beta2_fixed), epsilon1(eps1), epsilon2(eps2),
      weight_decay(wd), relative_step(rel_step),
      use_beta2_schedule(beta2_sched), dmax(dmax_), t(1), B_prev_(0.0) {
    this->Optimizer::lr = lr_;

    // ---- Defensive validation ----
    if (epsilon1 <= 0.0) {
        throw std::invalid_argument("Adafactor: epsilon1 must be > 0");
    }
    if (epsilon2 <= 0.0) {
        throw std::invalid_argument("Adafactor: epsilon2 must be > 0");
    }
    if (dmax_ <= 0.0) {
        throw std::invalid_argument("Adafactor: dmax must be > 0");
    }
    // β2 is only used when !use_beta2_schedule; clamp into (0, 1) for safety.
    if (!use_beta2_schedule) {
        if (beta2 <= 0.0 || beta2 >= 1.0) {
            // Defensive: rather than throwing (matches the project's optimizer
            // convention), clamp to a sane range.
            if (beta2 < 1e-12) beta2 = 1e-12;
            if (beta2 > 1.0 - 1e-12) beta2 = 1.0 - 1e-12;
        }
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Adafactor: weight_decay must be >= 0");
    }
}

double Adafactor::rms(const Tensor& t_) {
    // RMS = sqrt(mean(t^2))
    if (t_.rows == 0 || t_.cols == 0) return 0.0;
    double sumsq = 0.0;
    size_t n = t_.rows * t_.cols;
    for (size_t i = 0; i < t_.rows; ++i) {
        for (size_t j = 0; j < t_.cols; ++j) {
            double v = t_[i][j];
            sumsq += v * v;
        }
    }
    return std::sqrt(sumsq / static_cast<double>(n));
}

void Adafactor::ensure_state(void* layer_ptr,
                              const std::vector<Tensor*>& params) {
    // Avoid re-init if state already present.
    if (R_state_.find(layer_ptr) != R_state_.end()
        || v1d_state_.find(layer_ptr) != v1d_state_.end()) {
        return;
    }

    std::vector<Tensor> R_vec, C_vec, v1d_vec;
    R_vec.reserve(params.size());
    C_vec.reserve(params.size());
    v1d_vec.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        if (p.rows == 1 || p.cols == 1) {
            // 1-D parameter: per-element EMA with the same shape as p.
            Tensor v(p.rows, p.cols);
            v.fill(0.0);
            v1d_vec.push_back(std::move(v));
            // Placeholder rows for R and C — never used.
            R_vec.emplace_back(0, 0);
            C_vec.emplace_back(0, 0);
        } else {
            // 2-D parameter: R ∈ R^{d1×1}, C ∈ R^{1×d2}.
            Tensor R(p.rows, 1);
            Tensor C(1, p.cols);
            R.fill(0.0);
            C.fill(0.0);
            R_vec.push_back(std::move(R));
            C_vec.push_back(std::move(C));
            // Placeholder for 1-D slot.
            v1d_vec.emplace_back(0, 0);
        }
    }
    R_state_[layer_ptr] = std::move(R_vec);
    C_state_[layer_ptr] = std::move(C_vec);
    v1d_state_[layer_ptr] = std::move(v1d_vec);
}

bool Adafactor::get_R(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = R_state_.find(layer_ptr);
    if (it == R_state_.end()) return false;
    if (param_idx >= it->second.size()) return false;
    out = it->second[param_idx];  // deep copy
    return true;
}

bool Adafactor::get_C(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = C_state_.find(layer_ptr);
    if (it == C_state_.end()) return false;
    if (param_idx >= it->second.size()) return false;
    out = it->second[param_idx];
    return true;
}

bool Adafactor::get_v1d(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = v1d_state_.find(layer_ptr);
    if (it == v1d_state_.end()) return false;
    if (param_idx >= it->second.size()) return false;
    out = it->second[param_idx];
    return true;
}

void Adafactor::update_param_2d(Tensor* param, Tensor* grad,
                                  Tensor& R, Tensor& C,
                                  double b2, double bias_corr) {
    const size_t d1 = param->rows;
    const size_t d2 = param->cols;

    // ---- Step 1: compute per-row and per-col sums of g^2 (and store R, C) ----
    // We do this in a single pass: g^2 → accumulate per-row and per-col sums.
    std::vector<double> row_sumsq(d1, 0.0);
    std::vector<double> col_sumsq(d2, 0.0);
    for (size_t i = 0; i < d1; ++i) {
        for (size_t j = 0; j < d2; ++j) {
            double g = (*grad)[i][j];
            double g2 = g * g;
            row_sumsq[i] += g2;
            col_sumsq[j] += g2;
        }
    }
    // Normalize to row/col means and update R and C (EMA).
    for (size_t i = 0; i < d1; ++i) {
        double row_mean = row_sumsq[i] / static_cast<double>(d2);
        R[i][0] = b2 * R[i][0] + (1.0 - b2) * row_mean;
    }
    for (size_t j = 0; j < d2; ++j) {
        double col_mean = col_sumsq[j] / static_cast<double>(d1);
        C[0][j] = b2 * C[0][j] + (1.0 - b2) * col_mean;
    }

    // ---- Step 2: compute mean(R) for the v_t reconstruction denominator ----
    double R_mean = 0.0;
    for (size_t i = 0; i < d1; ++i) R_mean += R[i][0];
    R_mean /= static_cast<double>(d1);
    double R_mean_safe = std::max(R_mean, epsilon2);

    // ---- Step 3: compute u_t (the per-coord update direction) ----
    // We need u_t's RMS for the relative-step computation. We accumulate
    // u_t in place to also write the parameter update.
    double u_sumsq = 0.0;
    // First pass: compute u_t and accumulate its squared sum.
    // We store u_t in a scratch local to avoid recomputing the sqrt.
    std::vector<std::vector<double>> u(d1, std::vector<double>(d2, 0.0));
    for (size_t i = 0; i < d1; ++i) {
        for (size_t j = 0; j < d2; ++j) {
            double v_ij = (R[i][0] * C[0][j]) / R_mean_safe / bias_corr;
            double denom = std::sqrt(v_ij + epsilon1);
            u[i][j] = (*grad)[i][j] / denom;
            u_sumsq += u[i][j] * u[i][j];
        }
    }
    double u_rms = std::sqrt(u_sumsq / static_cast<double>(d1 * d2));

    // ---- Step 4: compute effective lr ----
    double lr_eff;
    if (relative_step) {
        double w_rms = rms(*param);
        // Paper: lr_t = max(eps2, RMS(W_{t-1})) / RMS(u_t) (with optional dmax scaling)
        // but the "no schedule" reference also scales by 1 / dmax^(0.5) for some
        // implementations. We use the simple form: max(eps2, RMS(W))/RMS(u).
        // dmax is applied as an additional overall scale (lr *= 1/dmax^(0.5))
        // when the parameter has many rows (vocab-sized projections). For
        // ordinary dense layers dmax=1 makes this a no-op.
        lr_eff = std::max(epsilon2, w_rms) / std::max(epsilon2, u_rms);
        if (dmax != 1.0) {
            lr_eff /= std::sqrt(dmax);
        }
    } else {
        lr_eff = lr;
    }

    // ---- Step 5: apply the update ----
    for (size_t i = 0; i < d1; ++i) {
        for (size_t j = 0; j < d2; ++j) {
            double step = u[i][j] + weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr_eff * step;
        }
    }
}

void Adafactor::update_param_1d(Tensor* param, Tensor* grad,
                                  Tensor& v,
                                  double b2, double bias_corr) {
    const size_t n = param->rows * param->cols;
    // Per-element EMA (Adam-style v_t), bias-corrected.
    for (size_t i = 0; i < param->rows; ++i) {
        for (size_t j = 0; j < param->cols; ++j) {
            double g = (*grad)[i][j];
            v[i][j] = b2 * v[i][j] + (1.0 - b2) * g * g;
            double v_hat = v[i][j] / bias_corr;
            double denom = std::sqrt(v_hat + epsilon1);
            double u_ij = g / denom;
            // Step scalar (same per-element for 1-D)
            double step = u_ij + weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * step;  // for 1-D, use constant lr (no relative_step)
        }
    }
    (void)n;  // n is implicit in the loops
}

void Adafactor::step(Model& model) {
    // Compute β2_t for this step and the running bias-correction factor.
    double b2 = current_beta2();
    // Bias correction: B_t = 1 - ∏_{i=1..t} β2_i.
    // We track this as a single value rather than recomputing the product.
    // Initialize on first step: B_1 = 1 - β2_1.
    // The pattern is B_t = 1 - β2_t * (1 - B_{t-1}) because
    //   1 - ∏_{i=1..t} β2_i = 1 - β2_t · ∏_{i=1..t-1} β2_i
    //                         = 1 - β2_t · (1 - (1 - ∏_{i=1..t-1} β2_i))
    //                         = 1 - β2_t · (1 - B_{t-1})
    // We store B_{t-1} in a static-ish local; in practice this is just
    // `(1 - std::pow(β2_schedule_constant, t))`-like but exact for the
    // schedule β2_t = 1 - t^(-0.8). For the fixed-β2 path it's
    //   1 - β2^t
    // For the scheduled path the closed form doesn't simplify cleanly,
    // so we maintain it incrementally.
    double bias_corr;
    if (use_beta2_schedule) {
        // The schedule is β2_i = 1 - i^(-0.8). The cumulative product has no
        // simple closed form, so we track B_t = 1 - ∏ β2_i incrementally.
        bias_corr = 1.0 - b2 * (1.0 - B_prev_);
        if (bias_corr < 1e-12) bias_corr = 1e-12;  // defensive
    } else {
        // Fixed β2: closed form 1 - β2^t.
        bias_corr = 1.0 - std::pow(beta2, t);
        if (bias_corr < 1e-12) bias_corr = 1e-12;
    }

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& R_vec = R_state_[layer_ptr];
        auto& C_vec = C_state_[layer_ptr];
        auto& v1d_vec = v1d_state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            if (p->rows == 1 || p->cols == 1) {
                update_param_1d(p, g, v1d_vec[i], b2, bias_corr);
            } else {
                update_param_2d(p, g, R_vec[i], C_vec[i], b2, bias_corr);
            }
        }

        layer->zero_grad();
    }

    // Persist bias-correction for the next step (scheduled-β2 path).
    if (use_beta2_schedule) {
        B_prev_ = bias_corr;
    }
    ++t;
}
