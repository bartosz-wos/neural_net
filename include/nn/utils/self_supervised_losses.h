#ifndef SELF_SUPERVISED_LOSSES_H
#define SELF_SUPERVISED_LOSSES_H

#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// Self-Supervised Learning (SSL) Losses
// =============================================================================
// The five canonical representation-learning losses that complement the
// already-shipped InfoNCE/NT-Xent in `contrastive_losses.h`. Each takes L2-
// normalised projection vectors z ∈ R^{2N × D} as input (rows 2i and 2i+1 are
// the two augmented views of sample i) and returns a 1×1 scalar loss.
//
// Forward:  z ∈ R^{2N × D}   →   loss ∈ R^{1 × 1}      (mean over the 2N anchors)
// Backward: z ∈ R^{2N × D}   →   grad ∈ R^{2N × D}     (gradient w.r.t. z)
//
// Reference papers:
//   - DINO       — Caron et al. 2021, https://arxiv.org/abs/2104.14294
//   - BYOL       — Grill et al. 2020, https://arxiv.org/abs/2006.07733
//   - VICReg     — Bardes & Weston 2022, https://arxiv.org/abs/2105.04906
//   - BarlowTwins — Zbontar et al. 2021, https://arxiv.org/abs/2103.03230
//   - W-MSE      — Ermolov et al. 2021, https://arxiv.org/abs/2102.04619
// =============================================================================


// =============================================================================
// BYOL — Bootstrap Your Own Latent (Grill et al. 2020)
// =============================================================================
// L = (1/(2N·D)) Σ_a ||z_a − z_target_{a⊕1}||²
//
// The "target" projection is supplied by the caller as `z_target` of shape
// (2N, D). In a real training loop the caller maintains z_target as an
// exponential moving average of z and updates it after each step. The loss
// itself treats z_target as a fixed constant → backward w.r.t. z_target is
// zero (no gradient through the EMA branch).
//
// dL/dz_a = (2/(2N·D)) · (z_a − z_target_{a⊕1})

class BYOL {
public:
    explicit BYOL() {}

    // Forward: returns scalar loss in a 1×1 tensor.
    Tensor forward(const Tensor& z, const Tensor& z_target);
    // Backward: gradient w.r.t. z (the student projection).
    // `z_target` is the same as passed to forward.
    Tensor backward(const Tensor& z, const Tensor& z_target);
};


// =============================================================================
// VICReg — Variance-Invariance-Covariance Regularization (Bardes 2022)
// =============================================================================
// Invariance:  L_inv = (1/(2N))  · Σ_a ||z_a − z_{a⊕1}||²
// Variance:    L_var = (1/D)     · Σ_d max(0, γ − std_d(z))
//                        where std_d is the per-dim std across the 2N batch
// Covariance:  L_cov = (1/D)     · Σ_{i≠j} C_{ij}²
//                        where C is the (D×D) cross-corr matrix:
//                          C_{ij} = (1/(2N-1)) · Σ_a (z_a_i − μ_i)(z_a_j − μ_j)
// Final:       L = λ · L_inv + μ · L_var + ν · L_cov
//
// Backward is hand-derived: invariance is a direct chain; variance and
// covariance use the mean-subtraction chain (the ∂μ/∂z term) plus the
// off-diagonal C_{ij} chain.

class VICReg {
public:
    explicit VICReg(double sim_w = 25.0,         // λ (paper §3.1, default 25)
                     double std_w = 25.0,         // μ
                     double cov_w = 1.0,          // ν
                     double gamma = 1.0)          // γ (variance hinge)
        : sim_w_(sim_w), std_w_(std_w), cov_w_(cov_w), gamma_(gamma) {
        if (sim_w_ < 0 || std_w_ < 0 || cov_w_ < 0 || gamma_ <= 0) {
            sim_w_ = 25.0; std_w_ = 25.0; cov_w_ = 1.0; gamma_ = 1.0;
        }
    }

    Tensor forward(const Tensor& z);
    Tensor backward(const Tensor& z);

    double get_sim_w() const { return sim_w_; }
    double get_std_w() const { return std_w_; }
    double get_cov_w() const { return cov_w_; }
    double get_gamma()  const { return gamma_; }

private:
    double sim_w_, std_w_, cov_w_, gamma_;
    // Caches for backward (populated by forward, consumed by backward).
    Tensor mean_;       // (1, D) — per-dim mean across 2N
    Tensor centered_;   // (2N, D) — z − mean_  (cached to avoid re-centering in backward)
    Tensor C_;          // (D, D) — cross-correlation matrix
};


// =============================================================================
// BarlowTwins — Redundancy-Reduction (Zbontar et al. 2021)
// =============================================================================
// 1) Center z by subtracting per-column mean: z_c = z − μ.
// 2) Cross-correlation matrix:
//       C_{ij} = (1/(2N)) · Σ_a z_c_a_i · z_c_a_j
// 3) Loss:
//       L = Σ_i (1 − C_{ii})²  +  λ · Σ_{i≠j} C_{ij}²
//
// Backward chains through mean-subtraction and outer-product.

class BarlowTwins {
public:
    explicit BarlowTwins(double lambda = 5e-3)  // paper default
        : lambda_(lambda) {
        if (lambda_ < 0) lambda_ = 5e-3;
    }

    Tensor forward(const Tensor& z);
    Tensor backward(const Tensor& z);

    double get_lambda() const { return lambda_; }

private:
    double lambda_;
    // Caches for backward
    Tensor centered_;
    Tensor C_;
    Tensor mean_;
};


// =============================================================================
// DINO — Self-Distillation with No Labels (Caron et al. 2021)
// =============================================================================
// Inputs:
//   student  : z_s ∈ R^{2N × K}  (K prototypes)
//   teacher  : z_t ∈ R^{2N × K}  (the EMA teacher projection)
//
// Student softmax with temperature τ_s:
//   P_s_a_k = exp(z_s_a_k / τ_s) / Σ_k' exp(z_s_a_k' / τ_s)
// Teacher softmax centered across batch then with temperature τ_t:
//   m_t_k   = mean_a z_t_a_k
//   z_t_c   = z_t − m_t                              (centering)
//   P_t_a_k = exp(z_t_c_a_k / τ_t) / Σ_k' exp(z_t_c_a_k' / τ_t)
//
// Cross-entropy (skip the self-pair a vs a⊕1):
//   L_a = − Σ_k P_t_{a⊕1, k} · log(P_s_a_k)
// Final: L = (1/(2N)) · Σ_a L_a
//
// dL/dz_s_a_k = (P_s_a_k − P_t_{a⊕1, k}) / (2N · τ_s)

class DINO {
public:
    explicit DINO(double student_temp = 0.1,    // τ_s (paper §3.1)
                   double teacher_temp = 0.07)   // τ_t
        : tau_s_(student_temp), tau_t_(teacher_temp) {
        if (tau_s_ <= 0) tau_s_ = 0.1;
        if (tau_t_ <= 0) tau_t_ = 0.07;
    }

    Tensor forward(const Tensor& z_student, const Tensor& z_teacher);
    Tensor backward(const Tensor& z_student, const Tensor& z_teacher);

    double get_student_temp() const { return tau_s_; }
    double get_teacher_temp() const { return tau_t_; }

private:
    double tau_s_, tau_t_;
    // Caches for backward (filled by forward).
    Tensor student_probs_;  // P_s (2N, K)
    Tensor teacher_probs_;  // P_t (2N, K) — already centered+sharpened
};


// =============================================================================
// W-MSE — Whitening-based MSE (Ermolov et al. 2021)
// =============================================================================
// Input: L2-normalised z ∈ R^{2N × D} (caller is responsible for the L2-norm
// step). The whitening transform uses the Cholesky factorisation of the
// 2N×D feature covariance. Following the paper §3.1, the whitening matrix is
// W = (z_cᵀ z_c / N + ε I)^{-1/2} where z_c is the centred z. We use the
// standard Cholesky-based inverse square root: if A = L Lᵀ then A^{-1/2} =
// L^{-1} (after symmetrising) and the centred whitened vector is W · z_c_a.
//
// Loss:
//   L = (1/(2N · D)) · Σ_a ||W · z_c_a − W · z_c_{a⊕1}||²
//
// dL/dz is the chain-rule through the centring + whitening + MSE. The
// whitening-backward is the standard Cholesky-grad (Pinelis–Molay–
// Townsend 2018 form):
//   dL/dA = −0.5 · W · dL/dW · W
//   dL/dz_c_a = Wᵀ · dL/dW_z_c_a
//   dL/dz_a = dL/dz_c_a − (1/(2N)) Σ_b dL/dz_c_b

class WMSE {
public:
    explicit WMSE(double eps = 1e-5)  // ε regularisation on A
        : eps_(eps) {
        if (eps_ <= 0) eps_ = 1e-5;
    }

    Tensor forward(const Tensor& z);
    Tensor backward(const Tensor& z);

    // Test helper: returns the whitening matrix computed during the most
    // recent forward() call. Used to verify the gradient w.r.t. z when W
    // is held fixed (the canonical W-MSE convention per Ermolov et al.
    // 2021 §3.1, where W is recomputed from batch stats but treated as a
    // constant within a backward step).
    Tensor W_for_test() const { return W_; }
    // Forward with an explicit whitening matrix (skips Cholesky / A
    // computation). Only meant for tests.
    Tensor forward_with_W(const Tensor& z, const Tensor& W);

    double get_eps() const { return eps_; }

private:
    double eps_;
    // Caches for backward
    Tensor mean_;
    Tensor centered_;
    Tensor A_;        // (D, D) — feature cov + eps·I  (was L Lᵀ pre-solve)
    Tensor W_;        // (D, D) — whitening matrix A^{-1/2}
};


#endif
