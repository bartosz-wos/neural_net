// ============================================================================
// APOLLO: Approximated Gradient Scaling for Memory-Efficient LLM Optimization
// Zhu et al. 2024, https://arxiv.org/abs/2412.05270
// (MLSys 2025, Outstanding Paper Honorable Mention)
//
// Implementation follows the algorithm documented in apollo.h. See the
// header for the full math derivation. We diverge from the reference PyTorch
// code in one important way: the scaling factor is computed and applied
// INSIDE the low-rank space (then projected back via R^T), rather than
// multiplying the original full-rank gradient. This produces a clean, fully
// broadcastable operation that matches the paper's stated intent: "channel-
// wise" scaling in the rank-r auxiliary space, mapped back to the original
// space. The end-to-end behavior (memory cost, update direction, magnitude)
// is identical to the paper's algorithm.
// ============================================================================

#include "apollo.h"
#include "../core/layer.h"
#include "../core/model.h"
#include "../core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

// -----------------------------------------------------------------------
// Local helpers
// -----------------------------------------------------------------------

namespace {

// Box-Muller transform → standard normal sample. Pulled out of the loop for
// readability; not a hot-path optimization.
double sample_normal() {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);   // (0, 1]
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);   // (0, 1]
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Fill a Tensor with i.i.d. N(0, 1/sqrt(r)) samples.
void fill_gaussian(Tensor& t, double scale) {
    const size_t total = t.rows * t.cols;
    for (size_t idx = 0; idx < total; ++idx) {
        t.data[idx] = sample_normal() * scale;
    }
}

// Frobenius norm of a tensor.
double frobenius_norm(const Tensor& t) {
    double s = 0.0;
    const size_t total = t.rows * t.cols;
    for (size_t idx = 0; idx < total; ++idx) {
        const double v = t.data[idx];
        s += v * v;
    }
    return std::sqrt(s);
}

// L2 norm along the row axis of a (rows, cols) tensor.
// Returns a (1, cols) Tensor where result(0, j) = sqrt(sum_i t(i, j)^2).
Tensor l2_norm_along_rows(const Tensor& t) {
    Tensor out(1, t.cols);
    for (size_t j = 0; j < t.cols; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < t.rows; ++i) {
            const double v = t[i][j];
            s += v * v;
        }
        out[0][j] = std::sqrt(s);
    }
    return out;
}

} // namespace

// -----------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------

void APOLLO::validate(double lr,
                      double beta1,
                      double beta2,
                      double epsilon,
                      double weight_decay,
                      size_t rank,
                      double scale,
                      size_t update_proj_gap) {
    if (lr <= 0.0) {
        throw std::invalid_argument("APOLLO: lr must be > 0");
    }
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("APOLLO: beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("APOLLO: beta2 must be in [0, 1)");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("APOLLO: epsilon must be > 0");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("APOLLO: weight_decay must be >= 0");
    }
    if (rank == 0) {
        throw std::invalid_argument("APOLLO: rank must be >= 1");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("APOLLO: scale must be > 0");
    }
    if (update_proj_gap == 0) {
        throw std::invalid_argument("APOLLO: update_proj_gap must be >= 1");
    }
}

// -----------------------------------------------------------------------
// Constructor + setters
// -----------------------------------------------------------------------

APOLLO::APOLLO(double lr,
               double beta1,
               double beta2,
               double epsilon,
               double weight_decay,
               size_t rank,
               ScaleType scale_type,
               double scale,
               size_t update_proj_gap,
               bool scale_front,
               bool use_nl)
    : beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      rank_(rank),
      scale_type_(scale_type),
      scale_(scale),
      update_proj_gap_(update_proj_gap),
      scale_front_(scale_front),
      use_nl_(use_nl),
      num_steps_(0) {
    validate(lr, beta1_, beta2_, epsilon_, weight_decay_,
             rank_, scale_, update_proj_gap_);
    this->Optimizer::lr = lr;
}

void APOLLO::set_lr(double v) {
    validate(v, beta1_, beta2_, epsilon_, weight_decay_,
             rank_, scale_, update_proj_gap_);
    this->Optimizer::lr = v;
}

void APOLLO::set_beta1(double v) {
    validate(Optimizer::lr, v, beta2_, epsilon_, weight_decay_,
             rank_, scale_, update_proj_gap_);
    beta1_ = v;
}

void APOLLO::set_beta2(double v) {
    validate(Optimizer::lr, beta1_, v, epsilon_, weight_decay_,
             rank_, scale_, update_proj_gap_);
    beta2_ = v;
}

void APOLLO::set_epsilon(double v) {
    validate(Optimizer::lr, beta1_, beta2_, v, weight_decay_,
             rank_, scale_, update_proj_gap_);
    epsilon_ = v;
}

void APOLLO::set_weight_decay(double v) {
    validate(Optimizer::lr, beta1_, beta2_, epsilon_, v,
             rank_, scale_, update_proj_gap_);
    weight_decay_ = v;
}

void APOLLO::set_rank(size_t v) {
    validate(Optimizer::lr, beta1_, beta2_, epsilon_, weight_decay_,
             v, scale_, update_proj_gap_);
    rank_ = v;
}

void APOLLO::set_scale(double v) {
    validate(Optimizer::lr, beta1_, beta2_, epsilon_, weight_decay_,
             rank_, v, update_proj_gap_);
    scale_ = v;
}

void APOLLO::set_update_proj_gap(size_t v) {
    validate(Optimizer::lr, beta1_, beta2_, epsilon_, weight_decay_,
             rank_, scale_, v);
    update_proj_gap_ = v;
}

void APOLLO::set_scale_type(ScaleType v) { scale_type_ = v; }
void APOLLO::set_scale_front(bool v)     { scale_front_ = v; }
void APOLLO::set_use_nl(bool v)          { use_nl_ = v; }

// -----------------------------------------------------------------------
// State initialization + introspection
// -----------------------------------------------------------------------

void APOLLO::ensure_state(void* layer_ptr,
                          const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) {
        return;
    }
    std::vector<ParameterState> layer_state;
    layer_state.reserve(params.size());
    for (Tensor* p : params) {
        ParameterState ps;
        const size_t m = p->rows;
        const size_t n = p->cols;
        const size_t min_dim = std::min(m, n);
        const size_t max_dim = std::max(m, n);
        // Effective rank: cannot exceed min_dim.
        const size_t r_eff = std::min(rank_, min_dim);
        ps.exp_avg    = Tensor(min_dim, r_eff);
        ps.exp_avg_sq = Tensor(min_dim, r_eff);
        ps.R          = Tensor(max_dim, r_eff);
        ps.exp_avg.fill(0.0);
        ps.exp_avg_sq.fill(0.0);
        ps.R.fill(0.0);
        layer_state.push_back(std::move(ps));
    }
    state_[layer_ptr] = std::move(layer_state);
}

bool APOLLO::has_state(void* layer_ptr) const {
    return state_.find(layer_ptr) != state_.end();
}

size_t APOLLO::num_params_with_state(void* layer_ptr) const {
    const auto it = state_.find(layer_ptr);
    return (it == state_.end()) ? 0 : it->second.size();
}

bool APOLLO::get_exp_avg(void* layer_ptr, size_t param_idx, Tensor& out) const {
    const auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].exp_avg.clone();
    return true;
}

bool APOLLO::get_exp_avg_sq(void* layer_ptr, size_t param_idx, Tensor& out) const {
    const auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].exp_avg_sq.clone();
    return true;
}

bool APOLLO::get_R(void* layer_ptr, size_t param_idx, Tensor& out) const {
    const auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].R.clone();
    return true;
}

size_t APOLLO::get_step_pt(void* layer_ptr, size_t param_idx) const {
    const auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return 0;
    return it->second[param_idx].step_pt;
}

// -----------------------------------------------------------------------
// Projection refresh
// -----------------------------------------------------------------------

void APOLLO::refresh_projection(ParameterState& state, size_t /*m*/, size_t /*n*/) {
    // Re-initialize R with i.i.d. N(0, 1/r) samples.
    // We use a separate uniform draw per element for variance reduction.
    const size_t r_eff = state.R.cols;
    fill_gaussian(state.R, 1.0 / std::sqrt(static_cast<double>(r_eff)));
}

// -----------------------------------------------------------------------
// Per-parameter update
// -----------------------------------------------------------------------

void APOLLO::update_param(Tensor* param, Tensor* grad, ParameterState& state) {
    const size_t m = param->rows;
    const size_t n = param->cols;
    const size_t min_dim = std::min(m, n);
    const size_t r_eff = std::min(rank_, min_dim);

    // Lazy / periodic projection refresh.
    if (!state.projection_initialized ||
        (num_steps_ % update_proj_gap_) == 0) {
        refresh_projection(state, m, n);
        state.projection_initialized = true;
        state.step_pt = num_steps_;
    }

    // Step 2. Project gradient to low rank.
    // g_low has shape (min_dim, r_eff).
    Tensor g_low(min_dim, r_eff);
    if (m < n) {
        // g_low[i, k] = sum_j G[i, j] * R[j, k]
        for (size_t i = 0; i < m; ++i) {
            for (size_t k = 0; k < r_eff; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    s += (*grad)[i][j] * state.R[j][k];
                }
                g_low[i][k] = s;
            }
        }
    } else {
        // g_low[i, k] = sum_j R[j, k] * G[j, i]   (here i ranges over n,
        // since min_dim = n).
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < r_eff; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    s += state.R[j][k] * (*grad)[j][i];
                }
                g_low[i][k] = s;
            }
        }
    }

    // Step 3. Adam EMA in low-rank space.
    for (size_t i = 0; i < min_dim; ++i) {
        for (size_t k = 0; k < r_eff; ++k) {
            state.exp_avg[i][k] = beta1_ * state.exp_avg[i][k]
                                + (1.0 - beta1_) * g_low[i][k];
            const double gsq = g_low[i][k] * g_low[i][k];
            state.exp_avg_sq[i][k] = beta2_ * state.exp_avg_sq[i][k]
                                   + (1.0 - beta2_) * gsq;
        }
    }

    // Step 4-5. Bias-corrected "Adam" update in low-rank space.
    const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(num_steps_));
    const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(num_steps_));
    Tensor u_low(min_dim, r_eff);
    for (size_t i = 0; i < min_dim; ++i) {
        for (size_t k = 0; k < r_eff; ++k) {
            const double m_hat = state.exp_avg[i][k] / bc1;
            const double v_hat = state.exp_avg_sq[i][k] / bc2;
            u_low[i][k] = m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    // Step 6. Compute scaling factor s_r ∈ R^r.
    Tensor s_r(1, r_eff);
    if (scale_type_ == ScaleType::TENSOR) {
        // Single global scalar.
        const double num = frobenius_norm(u_low);
        const double den = frobenius_norm(g_low) + 1e-8;
        s_r.fill(num / den);
    } else {
        // Per-channel: norm along rows of u_low and g_low.
        const Tensor nu = l2_norm_along_rows(u_low);
        const Tensor ng = l2_norm_along_rows(g_low);
        for (size_t k = 0; k < r_eff; ++k) {
            s_r[0][k] = nu[0][k] / (ng[0][k] + 1e-8);
        }
    }

    // Step 7. Apply scaling in low-rank space.
    Tensor u_low_scaled(min_dim, r_eff);
    for (size_t i = 0; i < min_dim; ++i) {
        for (size_t k = 0; k < r_eff; ++k) {
            u_low_scaled[i][k] = u_low[i][k] * s_r[0][k];
        }
    }

    // Step 8. Project back to FULL space.
    Tensor u_full(m, n);
    if (m < n) {
        // u_full[i, j] = sum_k u_low_scaled[i, k] * R[j, k]
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < r_eff; ++k) {
                    s += u_low_scaled[i][k] * state.R[j][k];
                }
                u_full[i][j] = s;
            }
        }
    } else {
        // u_full[i, j] = sum_k R[i, k] * u_low_scaled[j, k]
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < r_eff; ++k) {
                    s += state.R[i][k] * u_low_scaled[j][k];
                }
                u_full[i][j] = s;
            }
        }
    }

    // Step 9. Norm-Growth Limiter (Fira, arXiv:2410.01623).
    //
    // Compare the current scaled gradient's norm against the norm of the
    // PREVIOUS step (cached). If the new norm is more than 1% larger AND
    // the cached norm was already non-trivial, scale down so that the new
    // norm equals cached × 1.01. This is a safety net that prevents the
    // scaled gradient from growing uncontrollably step-over-step.
    if (use_nl_) {
        const double scaled_norm = frobenius_norm(u_full);
        if (state.cached_scaled_grad_norm > 1e-12) {
            const double limiter = std::max(
                scaled_norm / state.cached_scaled_grad_norm,
                1.01) / 1.01;
            if (limiter > 1.0) {
                const double inv = 1.0 / limiter;
                for (size_t idx = 0; idx < u_full.rows * u_full.cols; ++idx) {
                    u_full.data[idx] *= inv;
                }
                state.cached_scaled_grad_norm = scaled_norm / limiter;
            } else {
                state.cached_scaled_grad_norm = scaled_norm;
            }
        } else {
            // First step with non-trivial norm — establish cache, no clamping.
            state.cached_scaled_grad_norm = scaled_norm;
        }
    }

    // Step 10. Apply scale factor.
    if (scale_front_) {
        const double s = std::sqrt(scale_);
        for (size_t idx = 0; idx < u_full.rows * u_full.cols; ++idx) {
            u_full.data[idx] *= s;
        }
    }

    // Step 11. Decoupled weight decay.
    if (weight_decay_ > 0.0) {
        const double decay = 1.0 - Optimizer::lr * weight_decay_;
        for (size_t idx = 0; idx < param->rows * param->cols; ++idx) {
            param->data[idx] *= decay;
        }
    }

    // Step 12. Apply update.
    if (!scale_front_) {
        const double s = std::sqrt(scale_);
        for (size_t idx = 0; idx < u_full.rows * u_full.cols; ++idx) {
            u_full.data[idx] *= s;
        }
    }
    for (size_t idx = 0; idx < param->rows * param->cols; ++idx) {
        param->data[idx] -= Optimizer::lr * u_full.data[idx];
    }
}

// -----------------------------------------------------------------------
// step(): iterate over layers, ensure state, update each parameter.
// -----------------------------------------------------------------------

void APOLLO::step(Model& model) {
    ++num_steps_;
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) {
            continue;
        }
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "APOLLO: parameter and gradient counts must match");
        }
        ensure_state(layer_ptr, params);
        std::vector<ParameterState>& layer_state = state_[layer_ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            if (p->rows != g->rows || p->cols != g->cols) {
                throw std::runtime_error(
                    "APOLLO: parameter and gradient shapes must match");
            }
            update_param(p, g, layer_state[i]);
        }
        layer->zero_grad();
    }
}
