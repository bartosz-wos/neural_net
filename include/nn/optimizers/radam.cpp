#include "radam.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction + validation
// ---------------------------------------------------------------------------
void RAdam::validate(double b1, double b2, double eps) {
    if (!(b1 > 0.0 && b1 < 1.0)) {
        throw std::invalid_argument("RAdam: beta1 must lie in (0, 1)");
    }
    if (!(b2 > 0.0 && b2 < 1.0)) {
        throw std::invalid_argument("RAdam: beta2 must lie in (0, 1)");
    }
    if (!(eps > 0.0)) {
        throw std::invalid_argument("RAdam: epsilon must be > 0");
    }
}

RAdam::RAdam(double lr_, double beta1_, double beta2_, double eps, double wd)
    : lr(lr_), beta1(beta1_), beta2(beta2_),
      epsilon(eps), weight_decay(wd), t(1)
{
    validate(beta1, beta2, epsilon);
}

void RAdam::set_lr(double new_lr)            { lr = new_lr; }
void RAdam::set_beta1(double new_b1)         { validate(new_b1, beta2, epsilon); beta1 = new_b1; }
void RAdam::set_beta2(double new_b2)         { validate(beta1, new_b2, epsilon); beta2 = new_b2; }
void RAdam::set_epsilon(double new_eps)      { validate(beta1, beta2, new_eps); epsilon = new_eps; }
void RAdam::set_weight_decay(double new_wd)  { weight_decay = new_wd; }

// ---------------------------------------------------------------------------
// ρ_t / ρ_∞ / l_t closed-form accessors (used by tests + step())
//
//   ρ_∞ = 2/(1-β2) − 1
//   ρ_t = ρ_∞ − 2·t·β2^t / (1 − β2^t)
//   l_t = sqrt(((ρ_t-4)(ρ_t-2) ρ_∞) / ((ρ_∞-4)(ρ_∞-2) ρ_t))    [only valid if ρ_t > 4]
// ---------------------------------------------------------------------------
double RAdam::get_rho_infty() const {
    return 2.0 / (1.0 - beta2) - 1.0;
}

double RAdam::get_rho_t(int t_step) const {
    double rho_inf = get_rho_infty();
    double b2_t = std::pow(beta2, t_step);
    double denom = 1.0 - b2_t;
    if (denom < 1e-30) {
        // β2^t ≈ 1 means we're at the very start. The 2·t·β2^t/(1-β2^t) term
        // is huge in this regime — ρ_t is effectively -∞. Return a large
        // negative sentinel; callers check the > 4 threshold so they take the
        // SGD-like branch correctly.
        return -1e30;
    }
    return rho_inf - 2.0 * t_step * b2_t / denom;
}

double RAdam::get_l_t(int t_step) const {
    double rho_t   = get_rho_t(t_step);
    double rho_inf = get_rho_infty();
    // The l_t formula has the (ρ_t-4) and (ρ_t-2) factors. If ρ_t ≤ 4, l_t
    // is not used by the step() update — but the formula still evaluates to
    // some real number (which may be 0 if ρ_t == 4 or complex if ρ_t < 2).
    // We return the closed-form evaluation regardless.
    double num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
    double den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
    if (den <= 0.0) {
        // ρ_t out of the [4, ∞) range — return 0 (defensive)
        return 0.0;
    }
    if (num <= 0.0) {
        return 0.0;  // ρ_t < 4: numerator non-positive
    }
    return std::sqrt(num / den);
}

// ---------------------------------------------------------------------------
// State plumbing
// ---------------------------------------------------------------------------
void RAdam::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<RAdamState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        RAdamState st;
        st.m = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.v = Tensor(p.rows, p.cols);
        st.v.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

Tensor RAdam::get_m(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].m;
}

Tensor RAdam::get_v(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].v;
}

// ---------------------------------------------------------------------------
// Per-parameter update (the RAdam signature)
//
//   m_t = β1·m_{t-1} + (1-β1)·g_t
//   v_t = β2·v_{t-1} + (1-β2)·g_t²
//   m̂_t = m_t / (1-β1^t)
//
//   ρ_t = ρ_∞ − 2·t·β2^t / (1-β2^t)
//
//   if ρ_t > 4:
//       l_t = sqrt(((ρ_t-4)(ρ_t-2)ρ_∞) / ((ρ_∞-4)(ρ_∞-2)ρ_t))
//       param = param − lr·(l_t·m̂_t / (sqrt(v̂_t) + ε) + wd·param)
//   else:
//       param = param − lr·(m̂_t + wd·param)
//
// Note v̂_t is only needed in the adaptive branch (ρ_t > 4).
// ---------------------------------------------------------------------------
void RAdam::update_param(Tensor* param, Tensor* grad, RAdamState& st,
                         double lr_, double b1, double b2,
                         double eps, double wd,
                         double b1_c, double b2_c,
                         int t_step) const
{
    // Compute ρ_t once per (β2, t_step) — constant across all (i, j) entries.
    double b2_t     = std::pow(b2, t_step);
    double denom_b2 = 1.0 - b2_t;
    // If denom_b2 is essentially zero (β2^t ≈ 1), ρ_t is very negative —
    // take the SGD-like branch.
    double rho_t;
    if (denom_b2 < 1e-30) {
        rho_t = -1e30;
    } else {
        double rho_inf = 2.0 / (1.0 - b2) - 1.0;
        rho_t = rho_inf - 2.0 * t_step * b2_t / denom_b2;
    }

    // Decide branch once for the whole tensor (uniform across (i, j))
    bool adaptive_branch = (rho_t > 4.0);

    double l_t = 0.0;
    double v_hat_sqrt_plus_eps = 0.0;  // only used in adaptive branch
    if (adaptive_branch) {
        double rho_inf = 2.0 / (1.0 - b2) - 1.0;
        double num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
        double den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
        l_t = (num > 0.0 && den > 0.0) ? std::sqrt(num / den) : 0.0;
        // v̂_sqrt_plus_eps depends on each (i, j) entry's v — compute per-entry.
    }

    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // --- First-moment EMA (same as Adam) ---
            st.m[i][j] = b1 * st.m[i][j] + (1.0 - b1) * g;
            double m_hat = st.m[i][j] / b1_c;

            if (adaptive_branch) {
                // --- Second-moment EMA (same as Adam) ---
                st.v[i][j] = b2 * st.v[i][j] + (1.0 - b2) * g * g;
                double v_hat = st.v[i][j] / b2_c;
                v_hat_sqrt_plus_eps = std::sqrt(v_hat) + eps;

                // --- Adaptive branch: l_t · m̂ / (sqrt(v̂) + ε) ---
                double step = l_t * m_hat / v_hat_sqrt_plus_eps;
                // --- Decoupled weight decay ---
                if (wd > 0.0) {
                    (*param)[i][j] -= lr_ * (step + wd * (*param)[i][j]);
                } else {
                    (*param)[i][j] -= lr_ * step;
                }
            } else {
                // --- SGD-like branch: no v denominator, no l_t ---
                // v is still updated (it informs the next ρ_t computation
                // when we cross the boundary), but we don't use it in the
                // current step.
                st.v[i][j] = b2 * st.v[i][j] + (1.0 - b2) * g * g;

                // --- SGD-like step: lr·m̂ ---
                if (wd > 0.0) {
                    (*param)[i][j] -= lr_ * (m_hat + wd * (*param)[i][j]);
                } else {
                    (*param)[i][j] -= lr_ * m_hat;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------
void RAdam::step(Model& model) {
    // Bias-correction terms for this timestep.
    // At t=1: b1_c = 1 - b1, b2_c = 1 - b2.
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer     = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params     = layer->parameters();
        auto grads      = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i],
                         state_vec[i],
                         lr, beta1, beta2,
                         epsilon, weight_decay,
                         b1_c, b2_c,
                         t);
        }

        layer->zero_grad();
    }

    ++t;
}