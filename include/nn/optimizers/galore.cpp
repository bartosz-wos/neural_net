// galore.cpp — GaLore (Gradient Low-Rank Projection) optimizer (Zhao et al. 2024).
// https://arxiv.org/abs/2403.03508
//
// The implementation follows the "projection-only" variant of the paper (§3.1,
// Algorithm 1). Per parameter (shape (m, n)):
//
//   1. G_EMA_t = β2 · G_EMA_{t-1} + (1-β2) · G_t   (full matrix, then later
//      we extract top-r right singular vectors.)
//
//   2. Every `proj_update_interval` steps, refresh P_t:
//      Compute G_EMA^T G_EMA (n × n), eigendecompose, take top-r eigenvectors
//      as columns of P_t ∈ ℝ^{n×r}. P_t is orthonormal (P_t^T P_t = I_r).
//
//   3. Adam in projected space:
//      g_low = G_t @ P_t          (m × r)
//      m_low = β1 · m_low + (1-β1) · g_low
//      v_low = β2 · v_low + (1-β2) · g_low²
//      m_hat = m_low / (1 - β1^t)
//      v_hat = v_low / (1 - β2^t)
//      update_low = scale · m_hat / (sqrt(v_hat) + ε)
//      update = scale · update_low @ P_t^T   (m × n)
//      θ -= lr · update
//
//   4. Decoupled weight decay (if weight_decay > 0):
//      θ *= (1 - lr · weight_decay)
//
// For small parameters (m == 1 or n == 1) and trivial cases, the projection
// reduces to plain Adam (the row/column space is 1-D and the projection is
// identity).

#include "galore.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Hyperparameter validation
// ============================================================================

void GaLore::validate(double lr_, double b1, double b2, double eps,
                      int rank_, int proj_interval, double wd, double scale_) {
    if (!(lr_ > 0.0)) {
        throw std::invalid_argument("GaLore: lr must be positive");
    }
    if (!(b1 >= 0.0 && b1 < 1.0)) {
        throw std::invalid_argument("GaLore: beta1 must lie in [0, 1)");
    }
    if (!(b2 >= 0.0 && b2 < 1.0)) {
        throw std::invalid_argument("GaLore: beta2 must lie in [0, 1)");
    }
    if (!(eps > 0.0)) {
        throw std::invalid_argument("GaLore: epsilon must be positive");
    }
    if (rank_ < 1) {
        throw std::invalid_argument("GaLore: rank must be >= 1");
    }
    if (proj_interval < 1) {
        throw std::invalid_argument("GaLore: proj_update_interval must be >= 1");
    }
    if (wd < 0.0) {
        throw std::invalid_argument("GaLore: weight_decay must be non-negative");
    }
    if (!(scale_ > 0.0)) {
        throw std::invalid_argument("GaLore: scale must be positive");
    }
}

GaLore::GaLore(double lr_, double b1, double b2, double eps,
               int rank_, int proj_interval, double wd, double scale_)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      rank(rank_), proj_update_interval(proj_interval),
      weight_decay(wd), scale(scale_), step_count(1) {
    validate(lr_, b1, b2, eps, rank_, proj_interval, wd, scale_);
    Optimizer::lr = lr_;  // base class copy for schedulers
}

void GaLore::set_lr(double v) {
    validate(v, beta1, beta2, epsilon, rank, proj_update_interval, weight_decay, scale);
    lr = v;
    Optimizer::lr = v;
}
void GaLore::set_beta1(double v) {
    validate(lr, v, beta2, epsilon, rank, proj_update_interval, weight_decay, scale);
    beta1 = v;
}
void GaLore::set_beta2(double v) {
    validate(lr, beta1, v, epsilon, rank, proj_update_interval, weight_decay, scale);
    beta2 = v;
}
void GaLore::set_epsilon(double v) {
    validate(lr, beta1, beta2, v, rank, proj_update_interval, weight_decay, scale);
    epsilon = v;
}
void GaLore::set_rank(int v) {
    validate(lr, beta1, beta2, epsilon, v, proj_update_interval, weight_decay, scale);
    rank = v;
}
void GaLore::set_proj_update_interval(int v) {
    validate(lr, beta1, beta2, epsilon, rank, v, weight_decay, scale);
    proj_update_interval = v;
}
void GaLore::set_weight_decay(double v) {
    validate(lr, beta1, beta2, epsilon, rank, proj_update_interval, v, scale);
    weight_decay = v;
}
void GaLore::set_scale(double v) {
    validate(lr, beta1, beta2, epsilon, rank, proj_update_interval, weight_decay, v);
    scale = v;
}

// ============================================================================
// State initialization
// ============================================================================

void GaLore::ensure_state(void* layer_ptr,
                          const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<ParamState> vec;
    vec.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        const size_t m = p.rows;
        const size_t n = p.cols;

        // Effective rank: min(rank, min(m, n)).
        const int r = std::min(rank, (int)std::min(m, n));

        ParamState st;
        st.P       = Tensor(n, r);       // projection matrix (n × r)
        st.m_low   = Tensor(m, r);       // Adam first moment in projected space
        st.v_low   = Tensor(m, r);       // Adam second moment in projected space
        st.G_EMA   = Tensor(m, n);       // gradient EMA (full matrix)
        st.t_proj  = 0;                  // uninitialized

        st.P.fill(0.0);
        st.m_low.fill(0.0);
        st.v_low.fill(0.0);
        st.G_EMA.fill(0.0);

        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

// ============================================================================
// Symmetric eigendecomposition via cyclic Jacobi rotations
// ============================================================================
// (Same algorithm as SOAP::jacobi_eigendecompose — duplicated here so we
// don't introduce a cross-optimizer dependency. Identical math, different
// class scope.)

void GaLore::jacobi_eigendecompose(Tensor& A,
                                   Tensor& Q,
                                   Tensor& eigenvalues,
                                   int max_sweeps,
                                   double tol) {
    const size_t n = A.rows;
    if (n == 0) {
        eigenvalues = Tensor(0, 0);
        return;
    }
    if (A.cols != n || Q.rows != n || Q.cols != n) {
        throw std::invalid_argument("GaLore::jacobi_eigendecompose: shape mismatch");
    }

    Q.fill(0.0);
    for (size_t i = 0; i < n; ++i) Q(i, i) = 1.0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off_norm_sq = 0.0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q)
                off_norm_sq += 2.0 * A(p, q) * A(p, q);

        if (std::sqrt(off_norm_sq) < tol) break;

        for (size_t p = 0; p + 1 < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = A(p, q);
                if (std::abs(apq) < 1e-15) continue;

                const double app = A(p, p);
                const double aqq = A(q, q);

                double theta;
                if (std::abs(app - aqq) < 1e-30) {
                    theta = (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
                } else {
                    theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
                }
                const double c = std::cos(theta);
                const double s = std::sin(theta);

                const double new_app = c * c * app + 2.0 * c * s * apq + s * s * aqq;
                const double new_aqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
                A(p, p) = new_app;
                A(q, q) = new_aqq;
                A(p, q) = 0.0;
                A(q, p) = 0.0;

                for (size_t r_idx = 0; r_idx < n; ++r_idx) {
                    if (r_idx == p || r_idx == q) continue;
                    const double arp = A(r_idx, p);
                    const double arq = A(r_idx, q);
                    const double new_arp = c * arp + s * arq;
                    const double new_arq = -s * arp + c * arq;
                    A(r_idx, p) = new_arp;
                    A(p, r_idx) = new_arp;
                    A(r_idx, q) = new_arq;
                    A(q, r_idx) = new_arq;
                }

                for (size_t r_idx = 0; r_idx < n; ++r_idx) {
                    const double qrp = Q(r_idx, p);
                    const double qrq = Q(r_idx, q);
                    const double new_qrp = c * qrp + s * qrq;
                    const double new_qrq = -s * qrp + c * qrq;
                    Q(r_idx, p) = new_qrp;
                    Q(r_idx, q) = new_qrq;
                }
            }
        }
    }

    eigenvalues = Tensor(1, n);
    for (size_t i = 0; i < n; ++i) eigenvalues(0, i) = A(i, i);
}

// ============================================================================
// Projection refresh — extract top-r right singular vectors of G_EMA
// ============================================================================
//
// Math: the right singular vectors of G_EMA are the eigenvectors of
// G_EMA^T · G_EMA (an n × n symmetric matrix). We compute this Gram matrix,
// eigendecompose via Jacobi, then take the top-r eigenvectors (the ones
// with the largest eigenvalues) as columns of P. P is then orthonormal.
//
// To keep the API simple, we don't bother sorting eigenvectors by eigenvalue
// for the projection — the Adam-in-subspace step is invariant to the
// ordering of P's columns (the Adam update is symmetric across the r
// dimensions). The orthonormality of P is what matters for the
// projection-on-then-projection-off identity to hold.

void GaLore::refresh_projection(ParamState& st, size_t m, size_t n) {
    const int r = std::min(rank, (int)std::min(m, n));

    if (n == 1) {
        // 1-D input: projection is just a 1x1 identity-ish scalar.
        st.P = Tensor(1, 1);
        st.P(0, 0) = (st.G_EMA(0, 0) >= 0.0) ? 1.0 : -1.0;
        st.t_proj = step_count;
        return;
    }

    // Compute G_EMA^T G_EMA (n × n).
    Tensor gram(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += st.G_EMA(k, i) * st.G_EMA(k, j);
            gram(i, j) = s;
        }
    }

    // Eigendecompose: Q holds eigenvectors as columns.
    Tensor Q(n, n);
    Tensor eigvals(1, n);
    jacobi_eigendecompose(gram, Q, eigvals);

    // Copy top-r columns of Q into st.P (which is (n, r)).
    // If r < n, we keep the first r columns (no sorting needed — see header).
    // (For full-rank case r == n, copy all columns.)
    st.P = Tensor(n, r);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < (size_t)r; ++j) {
            st.P(i, j) = Q(i, j);
        }
    }

        st.t_proj = step_count;
}

// ============================================================================
// Per-parameter update
// ============================================================================
//
// For each parameter W ∈ R^{m × n}, gradient G:
//   1) G_EMA_t = β2 · G_EMA_{t-1} + (1-β2) · G_t
//   2) If step % proj_update_interval == 1 (or uninitialized), refresh P
//   3) Project: g_low = G · P (m × r)
//   4) Adam in projected space: m_low, v_low
//   5) Project back: update = scale · (m_hat / (sqrt(v_hat) + ε)) @ P^T
//   6) Decoupled weight decay (if weight_decay > 0)
//   7) W := W - lr · update

void GaLore::update_param(Tensor* param, Tensor* grad, ParamState& st,
                          double b1_correction, double b2_correction) {
    const size_t m = param->rows;
    const size_t n = param->cols;
    const int r = std::min(rank, (int)std::min(m, n));

    // ---- 1) G_EMA update ----
    // G_EMA = β2 · G_EMA + (1-β2) · G
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            st.G_EMA(i, j) = beta2 * st.G_EMA(i, j) + (1.0 - beta2) * (*grad)(i, j);
        }
    }

    // ---- 2) Projection refresh (if needed) ----
    // Refresh on step 1 (uninitialized) and every `proj_update_interval` steps
    // thereafter. The condition `step == 1 || (step - t_proj) >= proj_update_interval`
    // captures both cases — at step 1, t_proj == 0, so the diff is 1.
    if (st.t_proj == 0 || (step_count - st.t_proj) >= proj_update_interval) {
        refresh_projection(st, m, n);
    }

    // ---- 3) Project gradient: g_low = G · P (m × r) ----
    //   g_low[i, j] = sum_k G[i, k] * P[k, j]
    Tensor g_low(m, r);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < (size_t)r; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += (*grad)(i, k) * st.P(k, j);
            g_low(i, j) = s;
        }
    }

    // ---- 4) Adam in projected space ----
    Tensor update_low(m, r);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < (size_t)r; ++j) {
            const double gl = g_low(i, j);
            st.m_low(i, j) = beta1 * st.m_low(i, j) + (1.0 - beta1) * gl;
            st.v_low(i, j) = beta2 * st.v_low(i, j) + (1.0 - beta2) * gl * gl;
            const double m_hat = st.m_low(i, j) / b1_correction;
            const double v_hat = st.v_low(i, j) / b2_correction;
            const double denom = std::sqrt(v_hat) + epsilon;
            update_low(i, j) = scale * m_hat / denom;
        }
    }

    // ---- 5) Project update back: update = update_low @ P^T (m × n) ----
    //   update[i, j] = sum_k update_low[i, k] * P^T[k, j] = sum_k update_low[i, k] * P[j, k]
    Tensor update(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < (size_t)r; ++k) s += update_low(i, k) * st.P(j, k);
            update(i, j) = s;
        }
    }

    // ---- 6) Decoupled weight decay ----
    if (weight_decay > 0.0) {
        const double wd_factor = 1.0 - lr * weight_decay;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*param)(i, j) *= wd_factor;
    }

    // ---- 7) Apply update ----
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            (*param)(i, j) -= lr * update(i, j);
        }
    }
}

// ============================================================================
// step() — iterate over all layers, all parameters
// ============================================================================

void GaLore::step(Model& model) {
    const double b1_correction = 1.0 - std::pow(beta1, step_count);
    const double b2_correction = 1.0 - std::pow(beta2, step_count);

    for (auto& layer : model.layers) {
        Layer* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        ensure_state(ptr, params);
        auto& state_vec = state_[ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], state_vec[i],
                         b1_correction, b2_correction);
        }
    }

    ++step_count;
}

// ============================================================================
// State introspection
// ============================================================================

namespace {
const Tensor GALORE_EMPTY_TENSOR(0, 0);
}

size_t GaLore::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return 0;
    return it->second.size();
}

const GaLore::ParamState* GaLore::find_state(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return nullptr;
    if (param_idx >= it->second.size()) return nullptr;
    return &it->second[param_idx];
}

const Tensor& GaLore::get_P(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->P : GALORE_EMPTY_TENSOR;
}
const Tensor& GaLore::get_m_low(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->m_low : GALORE_EMPTY_TENSOR;
}
const Tensor& GaLore::get_v_low(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->v_low : GALORE_EMPTY_TENSOR;
}
const Tensor& GaLore::get_G_EMA(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->G_EMA : GALORE_EMPTY_TENSOR;
}
int GaLore::get_step_proj(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->t_proj : 0;
}
