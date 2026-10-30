// soap.cpp — SOAP (ShampoO with Adam in the Preconditioner's eigenbasis)
// implementation. Vyas et al. 2024 (https://arxiv.org/abs/2409.11321,
// NeurIPS 2024). The algorithm eigendecomposes the Shampoo left/right
// preconditioners every `precondition_frequency` steps, rotates the gradient
// into the eigenbasis, runs Adam in that basis, and rotates the update back.
//
// For 1-D parameters (m == 1 or n == 1) one of the preconditioners is 1x1 and
// we skip its eigendecomposition (use identity). For scalar (m == n == 1)
// parameters we run plain Adam (no rotation needed).

#include "soap.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include "../utils/linear_algebra.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Hyperparameter validation
// ============================================================================

void SOAP::validate(double lr,
                    double beta1,
                    double beta2,
                    double epsilon,
                    int    precondition_frequency,
                    double weight_decay) {
    if (lr < 0.0) {
        throw std::invalid_argument("SOAP: lr must be non-negative");
    }
    if (!(beta1 >= 0.0 && beta1 < 1.0)) {
        throw std::invalid_argument("SOAP: beta1 must lie in [0, 1)");
    }
    if (!(beta2 >= 0.0 && beta2 < 1.0)) {
        throw std::invalid_argument("SOAP: beta2 must lie in [0, 1)");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("SOAP: epsilon must be positive");
    }
    if (precondition_frequency < 1) {
        throw std::invalid_argument("SOAP: precondition_frequency must be >= 1");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("SOAP: weight_decay must be non-negative");
    }
}

SOAP::SOAP(double lr, double b1, double b2, double eps,
           int precond_freq, double wd)
    : beta1(b1), beta2(b2), epsilon(eps),
      precondition_frequency(precond_freq),
      weight_decay(wd), t(1) {
    validate(lr, b1, b2, eps, precond_freq, wd);
    Optimizer::lr = lr;
}

void SOAP::set_lr(double new_lr) {
    validate(new_lr, beta1, beta2, epsilon, precondition_frequency, weight_decay);
    Optimizer::lr = new_lr;
}

void SOAP::set_beta1(double new_beta1) {
    validate(Optimizer::lr, new_beta1, beta2, epsilon, precondition_frequency, weight_decay);
    beta1 = new_beta1;
}

void SOAP::set_beta2(double new_beta2) {
    validate(Optimizer::lr, beta1, new_beta2, epsilon, precondition_frequency, weight_decay);
    beta2 = new_beta2;
}

void SOAP::set_epsilon(double new_eps) {
    validate(Optimizer::lr, beta1, beta2, new_eps, precondition_frequency, weight_decay);
    epsilon = new_eps;
}

void SOAP::set_precondition_frequency(int new_freq) {
    validate(Optimizer::lr, beta1, beta2, epsilon, new_freq, weight_decay);
    precondition_frequency = new_freq;
}

void SOAP::set_weight_decay(double new_wd) {
    validate(Optimizer::lr, beta1, beta2, epsilon, precondition_frequency, new_wd);
    weight_decay = new_wd;
}

// ============================================================================
// State initialization
// ============================================================================

void SOAP::ensure_state(void* layer_ptr,
                        const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<ParameterState> vec;
    vec.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        const size_t m = p.rows;
        const size_t n = p.cols;

        ParameterState st;
        st.L   = (m >= 1) ? Tensor(m, m) : Tensor(1, 1);
        st.R   = (n >= 1) ? Tensor(n, n) : Tensor(1, 1);
        st.Q_L = (m >= 1) ? Tensor(m, m) : Tensor(1, 1);
        st.Q_R = (n >= 1) ? Tensor(n, n) : Tensor(1, 1);
        st.M   = Tensor(m, n);
        st.V   = Tensor(m, n);

        st.L.fill(0.0);
        st.R.fill(0.0);
        // Q_L and Q_R initialized to identity in update_param if/when needed.
        st.M.fill(0.0);
        st.V.fill(0.0);

        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

// ============================================================================
// Per-parameter update
// ============================================================================
//
// For each parameter W ∈ R^{m × n}, gradient G:
//   1) Preconditioner update (every precondition_frequency steps):
//        L_t = β2 · L_{t-1} + (1-β2) · G · G^T
//        R_t = β2 · R_{t-1} + (1-β2) · G^T · G
//        Q_L, λ_L = eigh(L_t)
//        Q_R, λ_R = eigh(R_t)
//   2) Rotate: G_rot = Q_L^T · G · Q_R
//   3) Adam in rotated space:
//        M = β1·M + (1-β1)·G_rot
//        V = β2·V + (1-β2)·G_rot²
//        update_rot = lr · (M / (1-β1^t)) / (sqrt(V / (1-β2^t)) + ε)
//   4) Rotate back: update = Q_L · update_rot · Q_R^T
//   5) W := W - update  (with optional decoupled weight decay)
//
// For 1-D (m == 1 or n == 1) or scalar (m == n == 1) parameters, the
// eigendecomposition is trivial and we collapse to plain Adam.

void SOAP::update_param(Tensor* param,
                        Tensor* grad,
                        ParameterState& st,
                        double b1_correction,
                        double b2_correction) {
    const size_t m = param->rows;
    const size_t n = param->cols;

    // Scalar (1x1): plain Adam.
    if (m == 1 && n == 1) {
        const double g = (*grad)(0, 0);
        st.M(0, 0) = beta1 * st.M(0, 0) + (1.0 - beta1) * g;
        st.V(0, 0) = beta2 * st.V(0, 0) + (1.0 - beta2) * g * g;
        const double m_hat = st.M(0, 0) / b1_correction;
        const double v_hat = st.V(0, 0) / b2_correction;
        const double denom = std::sqrt(v_hat) + epsilon;
        const double update = Optimizer::lr * m_hat / denom;

        if (weight_decay > 0.0) {
            (*param)(0, 0) *= (1.0 - Optimizer::lr * weight_decay);
        }
        (*param)(0, 0) -= update;
        return;
    }

    // ---- 1) Preconditioner update (every precondition_frequency steps) ----
    if (should_precondition(t)) {
        // Update L: L_t = β2·L_{t-1} + (1-β2)·G·G^T
        if (m >= 2) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < n; ++k) s += (*grad)(i, k) * (*grad)(j, k);
                    st.L(i, j) = beta2 * st.L(i, j) + (1.0 - beta2) * s;
                }
            }
        } else {
            // m == 1: L is 1×1. L = sum_k G[0,k]².
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += (*grad)(0, k) * (*grad)(0, k);
            st.L(0, 0) = beta2 * st.L(0, 0) + (1.0 - beta2) * s;
        }

        // Update R: R_t = β2·R_{t-1} + (1-β2)·G^T·G
        if (n >= 2) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < m; ++k) s += (*grad)(k, i) * (*grad)(k, j);
                    st.R(i, j) = beta2 * st.R(i, j) + (1.0 - beta2) * s;
                }
            }
        } else {
            // n == 1: R is 1×1. R = sum_k G[k,0]².
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += (*grad)(k, 0) * (*grad)(k, 0);
            st.R(0, 0) = beta2 * st.R(0, 0) + (1.0 - beta2) * s;
        }

        // Eigendecompose L (if m >= 2).
        //
        // The shared routine takes a const Tensor& and does NOT mutate it,
        // so no working copy is needed. SOAP does not use the eigenvalues at
        // all -- it rotates into the eigenbasis and runs Adam there -- so
        // only the eigenvector matrix is bound.
        if (m >= 2) {
            st.Q_L = nn::utils::symmetric_eigendecompose(st.L).eigenvectors;
        } else {
            // m == 1: Q_L is 1x1 identity (no rotation needed).
            st.Q_L(0, 0) = 1.0;
        }

        // Eigendecompose R (if n >= 2).
        if (n >= 2) {
            st.Q_R = nn::utils::symmetric_eigendecompose(st.R).eigenvectors;
        } else {
            // n == 1: Q_R is 1x1 identity.
            st.Q_R(0, 0) = 1.0;
        }
    }

    // ---- 2) Rotate gradient: G_rot = Q_L^T · G · Q_R ----
    // First: temp = Q_L^T · G, shape (m, n).
    //   temp[i, j] = sum_k Q_L[k, i] * G[k, j]
    Tensor temp(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += st.Q_L(k, i) * (*grad)(k, j);
            temp(i, j) = s;
        }
    }
    // Then: G_rot = temp · Q_R, shape (m, n).
    //   G_rot[i, j] = sum_k temp[i, k] * Q_R[k, j]
    Tensor G_rot(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += temp(i, k) * st.Q_R(k, j);
            G_rot(i, j) = s;
        }
    }

    // ---- 3) Adam in rotated space ----
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double gr = G_rot(i, j);
            st.M(i, j) = beta1 * st.M(i, j) + (1.0 - beta1) * gr;
            st.V(i, j) = beta2 * st.V(i, j) + (1.0 - beta2) * gr * gr;
        }
    }

    // ---- 4) Build update_rot = lr · M̂ / (sqrt(V̂) + ε) ----
    Tensor update_rot(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double m_hat = st.M(i, j) / b1_correction;
            const double v_hat = st.V(i, j) / b2_correction;
            const double denom = std::sqrt(v_hat) + epsilon;
            update_rot(i, j) = Optimizer::lr * m_hat / denom;
        }
    }

    // ---- 5) Rotate update back: update = Q_L · update_rot · Q_R^T ----
    // First: temp2 = update_rot · Q_R^T, shape (m, n).
    //   temp2[i, j] = sum_k update_rot[i, k] * Q_R[j, k]
    Tensor temp2(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += update_rot(i, k) * st.Q_R(j, k);
            temp2(i, j) = s;
        }
    }
    // Then: update = Q_L · temp2, shape (m, n).
    //   update[i, j] = sum_k Q_L[i, k] * temp2[k, j]
    Tensor update(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += st.Q_L(i, k) * temp2(k, j);
            update(i, j) = s;
        }
    }

    // ---- 6) Apply update ----
    if (weight_decay > 0.0) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*param)(i, j) *= (1.0 - Optimizer::lr * weight_decay);
    }
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            (*param)(i, j) -= update(i, j);
}

// ============================================================================
// step() — iterate over all layers, all parameters
// ============================================================================

void SOAP::step(Model& model) {
    // Bias correction denominators.
    const double b1_correction = 1.0 - std::pow(beta1, t);
    const double b2_correction = 1.0 - std::pow(beta2, t);

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

    ++t;
}

// ============================================================================
// State introspection
// ============================================================================

namespace {
const Tensor SOAP_EMPTY_TENSOR(0, 0);
}  // namespace

const SOAP::ParameterState* SOAP::find_state(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return nullptr;
    if (param_idx >= it->second.size()) return nullptr;
    return &it->second[param_idx];
}

Tensor SOAP::get_L(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->L : SOAP_EMPTY_TENSOR;
}
Tensor SOAP::get_R(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->R : SOAP_EMPTY_TENSOR;
}
Tensor SOAP::get_Q_L(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->Q_L : SOAP_EMPTY_TENSOR;
}
Tensor SOAP::get_Q_R(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->Q_R : SOAP_EMPTY_TENSOR;
}
Tensor SOAP::get_M(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->M : SOAP_EMPTY_TENSOR;
}
Tensor SOAP::get_V(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->V : SOAP_EMPTY_TENSOR;
}