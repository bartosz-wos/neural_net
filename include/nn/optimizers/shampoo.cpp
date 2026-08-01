// shampoo.cpp — Shampoo (Gupta, Koren, Singer 2018,
// https://arxiv.org/abs/1802.03668) implementation. Kronecker-factored
// preconditioned SGD that maintains two symmetric covariance matrices per
// 2-D parameter, eigendecomposes them once per step, and applies the
// preconditioner `L^{-1/4} G R^{-1/4}` to the gradient in the update.
//
// For 1-D parameters (m == 1 or n == 1) one of the preconditioners is
// 1×1 and we skip its eigendecomposition (use identity rotation). For
// scalar (m == n == 1) parameters we run plain SGD (no rotation needed).

#include "shampoo.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Hyperparameter validation
// ============================================================================

void Shampoo::validate(double lr,
                       double beta,
                       double eps,
                       double weight_decay) {
    if (lr < 0.0) {
        throw std::invalid_argument("Shampoo: lr must be non-negative");
    }
    if (!(beta >= 0.0 && beta < 1.0)) {
        throw std::invalid_argument("Shampoo: beta must lie in [0, 1)");
    }
    if (eps <= 0.0) {
        throw std::invalid_argument("Shampoo: eps must be positive");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Shampoo: weight_decay must be non-negative");
    }
}

Shampoo::Shampoo(double lr, double b, double p, double wd)
    : beta(b), eps(p), weight_decay(wd), t(1) {
    validate(lr, b, p, wd);
    Optimizer::lr = lr;
}

void Shampoo::set_lr(double new_lr) {
    validate(new_lr, beta, eps, weight_decay);
    Optimizer::lr = new_lr;
}

void Shampoo::set_beta(double new_beta) {
    validate(Optimizer::lr, new_beta, eps, weight_decay);
    beta = new_beta;
}

void Shampoo::set_eps(double new_eps) {
    validate(Optimizer::lr, beta, new_eps, weight_decay);
    eps = new_eps;
}

void Shampoo::set_weight_decay(double new_wd) {
    validate(Optimizer::lr, beta, eps, new_wd);
    weight_decay = new_wd;
}

// ============================================================================
// State initialization
// ============================================================================

void Shampoo::ensure_state(void* layer_ptr,
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
        st.U_L = (m >= 1) ? Tensor(m, m) : Tensor(1, 1);
        st.U_R = (n >= 1) ? Tensor(n, n) : Tensor(1, 1);

        st.L.fill(0.0);
        st.R.fill(0.0);

        // U_L and U_R are initialized to identity at the start of update_param
        // (lazy after the first step where they become the eigh output).
        st.U_L.fill(0.0);
        for (size_t k = 0; k < m; ++k) st.U_L(k, k) = 1.0;
        st.U_R.fill(0.0);
        for (size_t k = 0; k < n; ++k) st.U_R(k, k) = 1.0;

        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

// ============================================================================
// Symmetric eigendecomposition via cyclic Jacobi rotations
// ============================================================================
//
// Jacobi rotations are an O(n^3) iterative method for symmetric eigendecomposition.
// For small matrices (m, n <= ~16, typical Dense layer preconditioner size)
// it converges in a handful of sweeps and is numerically stable. We use
// cyclic sweeps: at each sweep, we visit every off-diagonal pair (p, q)
// with p < q and apply a rotation that zeros A(p, q) (and A(q, p) by symmetry).
//
// A sweep on an (n × n) matrix is O(n^3) — n*(n-1)/2 rotations, each O(n).
// We cap at max_sweeps iterations and break early when the off-diagonal norm
// falls below tol.
//
// On entry, A is a symmetric (n × n) tensor. On exit, A's diagonal contains
// the eigenvalues (in arbitrary order) and Q holds the orthogonal matrix of
// eigenvectors (Q(:, k) is the eigenvector for eigenvalue A(k, k)).
// The eigenvalues Tensor is filled with the diagonal of A on return.

void Shampoo::jacobi_eigendecompose(Tensor& A,
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
        throw std::invalid_argument("Shampoo::jacobi_eigendecompose: shape mismatch");
    }

    // Initialize Q to identity.
    Q.fill(0.0);
    for (size_t i = 0; i < n; ++i) Q(i, i) = 1.0;

    // Working copy of A (we will mutate A in place).
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off_norm_sq = 0.0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q)
                off_norm_sq += 2.0 * A(p, q) * A(p, q);

        if (std::sqrt(off_norm_sq) < tol) break;

        for (size_t p = 0; p < n - 1; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = A(p, q);
                if (std::abs(apq) < 1e-15) continue;

                const double app = A(p, p);
                const double aqq = A(q, q);

                // Compute the rotation angle.
                // Standard Jacobi formula: theta = (aqq - app) / (2 * apq).
                double theta;
                if (std::abs(app - aqq) < 1e-30) {
                    theta = (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
                } else {
                    theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
                }
                const double c = std::cos(theta);
                const double s = std::sin(theta);

                // Apply the rotation to A: A := J^T · A · J.
                // (rows p, q and cols p, q; symmetric so we touch both sides.)
                // New diagonal entries:
                const double new_app = c * c * app + 2.0 * c * s * apq + s * s * aqq;
                const double new_aqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
                A(p, p) = new_app;
                A(q, q) = new_aqq;
                A(p, q) = 0.0;
                A(q, p) = 0.0;

                // Off-diagonal entries: A(r, p) and A(r, q) for r != p, q.
                for (size_t r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = A(r, p);
                    const double arq = A(r, q);
                    const double new_arp = c * arp + s * arq;
                    const double new_arq = -s * arp + c * arq;
                    A(r, p) = new_arp;
                    A(p, r) = new_arp;
                    A(r, q) = new_arq;
                    A(q, r) = new_arq;
                }

                // Apply the same rotation to Q: Q := Q · J.
                for (size_t r = 0; r < n; ++r) {
                    const double qrp = Q(r, p);
                    const double qrq = Q(r, q);
                    const double new_qrp = c * qrp + s * qrq;
                    const double new_qrq = -s * qrp + c * qrq;
                    Q(r, p) = new_qrp;
                    Q(r, q) = new_qrq;
                }
            }
        }
    }

    // Copy diagonal into eigenvalues output.
    eigenvalues = Tensor(1, n);
    for (size_t i = 0; i < n; ++i) eigenvalues(0, i) = A(i, i);
}

// ============================================================================
// Per-parameter update
// ============================================================================
//
// For each parameter W ∈ R^{m × n}, gradient G:
//   1) Preconditioner update (every step — Shampoo is not Adam-like):
//        L_t = β · L_{t-1} + (1-β) · G · G^T
//        R_t = β · R_{t-1} + (1-β) · G^T · G
//        U_L, λ_L = eigh(L_t)
//        U_R, λ_R = eigh(R_t)
//   2) Preconditioner construction:
//        L^{-1/4} = U_L · diag(max(λ_L, eps)^{-1/4}) · U_L^T
//        R^{-1/4} = U_R · diag(max(λ_R, eps)^{-1/4}) · U_R^T
//   3) Apply preconditioner to gradient (rotate, scale by eigvals,
//      rotate back) producing `update`:
//        update = L^{-1/4} · G · R^{-1/4}
//   4) W := W - lr · update  (with optional decoupled weight decay)
//
// For 1-D (m == 1 or n == 1) or scalar (m == n == 1) parameters, the
// eigendecomposition is trivial and we collapse to a degenerate form.

void Shampoo::update_param(Tensor* param,
                           Tensor* grad,
                           ParameterState& st) {
    const size_t m = param->rows;
    const size_t n = param->cols;

    // Scalar (1x1): plain SGD.
    if (m == 1 && n == 1) {
        const double g = (*grad)(0, 0);
        const double update = Optimizer::lr * g;

        if (weight_decay > 0.0) {
            (*param)(0, 0) *= (1.0 - Optimizer::lr * weight_decay);
        }
        (*param)(0, 0) -= update;
        return;
    }

    // ---- 1) Preconditioner update ----
    // Update L: L_t = β·L_{t-1} + (1-β)·G·G^T
    if (m >= 2) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) s += (*grad)(i, k) * (*grad)(j, k);
                st.L(i, j) = beta * st.L(i, j) + (1.0 - beta) * s;
            }
        }
    } else {
        // m == 1: L is 1×1. L = sum_k G[0,k]².
        double s = 0.0;
        for (size_t k = 0; k < n; ++k) s += (*grad)(0, k) * (*grad)(0, k);
        st.L(0, 0) = beta * st.L(0, 0) + (1.0 - beta) * s;
    }

    // Update R: R_t = β·R_{t-1} + (1-β)·G^T·G
    if (n >= 2) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < m; ++k) s += (*grad)(k, i) * (*grad)(k, j);
                st.R(i, j) = beta * st.R(i, j) + (1.0 - beta) * s;
            }
        }
    } else {
        // n == 1: R is 1×1. R = sum_k G[k,0]².
        double s = 0.0;
        for (size_t k = 0; k < m; ++k) s += (*grad)(k, 0) * (*grad)(k, 0);
        st.R(0, 0) = beta * st.R(0, 0) + (1.0 - beta) * s;
    }

    // Eigendecompose L (if m >= 2). Capture eigenvalues for the
    // 1/4-power scaling below.
    Tensor eigvals_L;
    if (m >= 2) {
        // Make a working copy because jacobi_eigendecompose mutates A.
        Tensor L_copy = st.L;
        jacobi_eigendecompose(L_copy, st.U_L, eigvals_L);
        // Note: L_copy is now diagonal; st.U_L holds the eigenvectors.
    } else {
        // m == 1: U_L is 1x1 identity (no rotation needed).
        st.U_L(0, 0) = 1.0;
        eigvals_L = Tensor(1, 1);
        eigvals_L(0, 0) = st.L(0, 0);
    }

    // Eigendecompose R (if n >= 2).
    Tensor eigvals_R;
    if (n >= 2) {
        Tensor R_copy = st.R;
        jacobi_eigendecompose(R_copy, st.U_R, eigvals_R);
    } else {
        // n == 1: U_R is 1x1 identity.
        st.U_R(0, 0) = 1.0;
        eigvals_R = Tensor(1, 1);
        eigvals_R(0, 0) = st.R(0, 0);
    }

    // ---- 2) Build L^{-1/4} G R^{-1/4} ----
    //
    // Compute L^{-1/4} · G first:
    //   (L^{-1/4} G)[i, j] = U_L[i, k] * (max(λ_L[k], eps)^{-1/4}) * U_L[k, p] * G[p, j]
    //                       (sum over k, p)
    // But this is equivalent to: rotate G into L's eigenbasis, scale, rotate back.
    //   step a: tmp1 = U_L^T · G       (rotate into eigenbasis)
    //   step b: tmp2 = tmp1 .* diag(λ_L^{-1/4})    (scale by 1/4 power)
    //   step c: L_inv4_G = U_L · tmp2  (rotate back)
    //
    // Then do the same with R on the right:
    //   step d: tmp3 = L_inv4_G · U_R
    //   step e: tmp4 = tmp3 .* diag(λ_R^{-1/4})    (broadcast across rows)
    //   step f: update = tmp4 · U_R^T
    //
    // For 1-D (one side = 1), the corresponding identity rotation collapses
    // the multiplications on that side.

    // Step a: tmp1 = U_L^T · G   (shape (m, n))
    Tensor tmp1(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += st.U_L(k, i) * (*grad)(k, j);
            tmp1(i, j) = s;
        }
    }

    // Scale tmp1 by λ_L^{-1/4}
    Tensor tmp2(m, n);
    for (size_t i = 0; i < m; ++i) {
        // λ_L corresponds to column i of identity in eigenbasis (diagonal of L_copy)
        const double lam = eigvals_L(0, i);
        const double inv4 = 1.0 / std::pow(std::max(lam, eps), 0.25);
        for (size_t j = 0; j < n; ++j) {
            tmp2(i, j) = tmp1(i, j) * inv4;
        }
    }

    // Step c: L_inv4_G = U_L · tmp2   (shape (m, n))
    Tensor L_inv4_G(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += st.U_L(i, k) * tmp2(k, j);
            L_inv4_G(i, j) = s;
        }
    }

    // Step d: tmp3 = L_inv4_G · U_R   (shape (m, n))
    Tensor tmp3(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += L_inv4_G(i, k) * st.U_R(k, j);
            tmp3(i, j) = s;
        }
    }

    // Step e: scale by λ_R^{-1/4} (broadcast across rows)
    Tensor tmp4(m, n);
    for (size_t j = 0; j < n; ++j) {
        const double lam = eigvals_R(0, j);
        const double inv4 = 1.0 / std::pow(std::max(lam, eps), 0.25);
        for (size_t i = 0; i < m; ++i) {
            tmp4(i, j) = tmp3(i, j) * inv4;
        }
    }

    // Step f: update = tmp4 · U_R^T   (shape (m, n))
    //   update[i, j] = sum_k tmp4[i, k] * U_R[j, k]
    Tensor update(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += tmp4(i, k) * st.U_R(j, k);
            update(i, j) = s;
        }
    }

    // ---- 3) Apply update ----
    if (weight_decay > 0.0) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*param)(i, j) *= (1.0 - Optimizer::lr * weight_decay);
    }
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            (*param)(i, j) -= Optimizer::lr * update(i, j);
}

// ============================================================================
// step() — iterate over all layers, all parameters
// ============================================================================

void Shampoo::step(Model& model) {
    for (auto& layer : model.layers) {
        Layer* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        ensure_state(ptr, params);
        auto& state_vec = state_[ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], state_vec[i]);
        }
    }

    ++t;
}

// ============================================================================
// State introspection
// ============================================================================

namespace {
const Tensor SHAMPOO_EMPTY_TENSOR(0, 0);
}  // namespace

const Shampoo::ParameterState* Shampoo::find_state(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return nullptr;
    if (param_idx >= it->second.size()) return nullptr;
    return &it->second[param_idx];
}

Tensor Shampoo::get_L(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->L : SHAMPOO_EMPTY_TENSOR;
}
Tensor Shampoo::get_R(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->R : SHAMPOO_EMPTY_TENSOR;
}
Tensor Shampoo::get_U_L(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->U_L : SHAMPOO_EMPTY_TENSOR;
}
Tensor Shampoo::get_U_R(void* layer_ptr, size_t param_idx) const {
    auto* st = find_state(layer_ptr, param_idx);
    return st ? st->U_R : SHAMPOO_EMPTY_TENSOR;
}
