#ifndef SHAMPOO_H
#define SHAMPOO_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// Shampoo: Preconditioned Stochastic Gradient Descent
// =========================================================================
//
// Gupta, Koren, Singer 2018,
// "Shampoo: Preconditioned Stochastic Tensor Optimization"
// https://arxiv.org/abs/1802.03668
//
// The core idea: per 2-D parameter W ∈ R^{m×n}, maintain two symmetric
// covariance matrices of the gradient,
//   L_t = β · L_{t-1} + (1-β) · G · G^T        (m×m)
//   R_t = β · R_{t-1} + (1-β) · G^T · G        (n×n)
// and apply the pre-conditioner to the gradient in the update
//   update = L^{-1/4} · G · R^{-1/4}
//   W := W - lr · update
//
// Because L and R are symmetric PSD, we eigendecompose once per step
// (L = U_L · diag(λ_L) · U_L^T, R = U_R · diag(λ_R) · U_R^T), then form
//   L^{-1/4} = U_L · diag(λ_L^{-1/4}) · U_L^T
//   R^{-1/4} = U_R · diag(λ_R^{-1/4}) · U_R^T
//   update = U_L · diag(λ_L^{-1/4}) · U_L^T · G · U_R · diag(λ_R^{-1/4}) · U_R^T
// which is exactly L^{-1/4} G R^{-1/4}.
//
// Shampoo is the FOUNDATIONAL preconditioned SGD algorithm that SOAP and
// PSGD build on. SOAP adds Adam in the rotated basis; PSGD replaces the
// eigendecomposition with a different preconditioner shape. Shampoo itself
// is the classical "Kronecker-factored Approximate Curvature" algorithm.
//
// Per step t (Gupta et al. 2018 §3 Algorithm 1):
//   L_t = β · L_{t-1} + (1-β) · G · G^T
//   R_t = β · R_{t-1} + (1-β) · G^T · G
//   U_L, λ_L = eigh(L_t)
//   U_R, λ_R = eigh(R_t)
//   L_inv4 = U_L · diag(max(λ_L, eps)^{-1/4}) · U_L^T
//   R_inv4 = U_R · diag(max(λ_R, eps)^{-1/4}) · U_R^T
//   update = L_inv4 · G · R_inv4
//   W := W - lr · update
//
// Defaults (Gupta et al. 2018 §5.1):
//   lr = 1e-3, β = 0.9, eps = 1e-12, weight_decay = 0
//
// Edge cases:
//   - m == 1 OR n == 1: skip eigendecomp for that side, use identity rotation
//   - m == 1 AND n == 1: scalar — plain SGD (no preconditioner)
//   - Invalid hyperparameters throw std::invalid_argument
//
// State per parameter:
//   L (m×m), R (n×n)  — symmetric covariance EMA
//   U_L (m×m), U_R (n×n) — cached eigenbases (set after first step)
// =========================================================================

class Shampoo : public Optimizer {
public:
    // Public diagnostics.
    double beta;
    double eps;
    double weight_decay;
    int    t;  // next timestep, starts at 1

    explicit Shampoo(double lr = 1e-3,
                     double beta = 0.9,
                     double eps = 1e-12,
                     double weight_decay = 0.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated mutators.
    void set_lr(double new_lr);
    void set_beta(double new_beta);
    void set_eps(double new_eps);
    void set_weight_decay(double new_wd);

    // Accessors.
    double get_lr() const { return Optimizer::lr; }
    double get_beta() const { return beta; }
    double get_eps() const { return eps; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t() const { return t; }

    // State introspection (for tests / debugging).
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_L(void* layer_ptr, size_t param_idx) const;
    Tensor get_R(void* layer_ptr, size_t param_idx) const;
    Tensor get_U_L(void* layer_ptr, size_t param_idx) const;
    Tensor get_U_R(void* layer_ptr, size_t param_idx) const;

private:
    // Per-parameter state (private but accessed by friend helpers below).
    struct ParameterState {
        Tensor L;     // (m × m) symmetric covariance
        Tensor R;     // (n × n) symmetric covariance
        Tensor U_L;   // (m × m) orthogonal eigenbasis of L (cached)
        Tensor U_R;   // (n × n) orthogonal eigenbasis of R (cached)
    };

    std::map<void*, std::vector<ParameterState>> state_;

    // Helper for the public get_* accessors — must be a member (or friend)
    // because ParameterState is private.
    const ParameterState* find_state(void* layer_ptr, size_t param_idx) const;

    static void validate(double lr,
                         double beta,
                         double eps,
                         double weight_decay);

    // Lazy-init state for a layer on first encounter.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update.
    void update_param(Tensor* param,
                      Tensor* grad,
                      ParameterState& st);

    // Symmetric eigendecomposition via cyclic Jacobi rotations.
    // On entry, A is a symmetric (n × n) Tensor. On exit, Q holds the
    // orthogonal matrix of eigenvectors (columns) and eigenvalues is
    // filled with the eigenvalues of A. Q is initialized to identity.
    //
    // **THE ROTATION ANGLE'S SIGN IS LOAD-BEARING.** The general branch
    // must be
    //
    //     theta = 0.5 * atan2(2 * apq, app - aqq)
    //
    // and NOT `aqq - app`. The second argument's sign decides whether the
    // rotation ANNIHILATES the off-diagonal (correct) or roughly DOUBLES
    // it (wrong). With the wrong sign the routine still returns an
    // ORTHOGONAL matrix and still drives A to a diagonal matrix -- it
    // converges to some orthogonal similarity -- but the reported
    // eigenvalues are NOT the eigenvalues of A and Q's columns are NOT
    // eigenvectors. Measured on a 6x6 symmetric matrix: the wrong sign
    // gave an eigenvalue error of up to 2.95 and an eigenvector residual
    // ‖A·Q − Q·Λ‖ of order 1; the correct sign gives 5.3e-15 and 2.2e-16.
    //
    // **THE `app == aqq` DEGENERATE BRANCH IS CORRECT AND MUST STAY.** It
    // computes theta = ±π/4 directly (the `atan2` would be 0/0 there).
    // It is the proof that the general branch was the wrong one: the same
    // function is right on the degenerate branch and wrong on the general
    // one, and a probe that computes the post-rotation off-diagonal
    // directly measures 1.6e-16 for the degenerate branch against −1.269
    // for the general branch.
    //
    // **THE INVARIANT IS PINNED BY tests/test_jacobi_eig.cpp**, which
    // asserts A·Q == Q·Λ through the PUBLIC accessors. Orthonormality and
    // Frobenius-norm preservation are NOT sufficient: both hold for a
    // product of rotations that is not an eigenbasis, which is why the
    // defect survived every pre-existing test.
    static void jacobi_eigendecompose(Tensor& A,
                                      Tensor& Q,
                                      Tensor& eigenvalues,
                                      int max_sweeps = 100,
                                      double tol = 1e-12);
};

#endif
