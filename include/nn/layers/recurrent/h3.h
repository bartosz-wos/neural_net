#ifndef H3_H
#define H3_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// H3 (Hungry Hungry Hippos) — Fu, Dao, Saab, Thomas, Rudra, Ré 2023
//   "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
//   https://arxiv.org/abs/2212.14052  (ICLR 2023)
//
// ============================================================================
//
// H3 is a sequence layer that stacks two state-space models with multiplicative
// interactions between their outputs and input projections. The architecture
// was designed to give state-space models the two capabilities that attention
// has but plain SSMs (S4D, GSS) lack:
//
//   (i)  Recall of tokens that appeared after a particular event in the
//        sequence (solved by a Shift-SSM: a sliding window of past K values).
//   (ii) Comparison of tokens across the sequence (solved by multiplicative
//        interactions between SSM outputs and V/Q projections).
//
// ============================================================================
//
// Mathematical formulation (Algorithm 1 in the paper, with the single-head
// simplifications described below):
//
//   Input:        u ∈ R^{N×d}                       (sequence length N, hidden dim d)
//   Parameters:   W_Q, W_K, W_V, W_O ∈ R^{d×d}      (Dense projections)
//                 λ_log ∈ R^{d×d}                   (diagonal-SSM decay, λ = sigmoid(λ_log))
//
//   Step 1 — Input projections:
//     Q = u · W_Q,   K = u · W_K,   V = u · W_V      (each ∈ R^{N×d})
//
//   Step 2 — Shift SSM on K:
//     x_t ∈ R^{d}  (state of length d — holds the last d K values)
//     x_t[0] = K_t
//     x_t[i] = x_{t-1}[i-1]   for i ≥ 1
//     Equivalently:  x_t = [K_t, K_{t-1}, ..., K_{t-d+1}]
//     K̄_t = x_t     (output; we use C = identity, B = e_1)
//
//   Step 3 — Per-token outer product + Diagonal SSM:
//     z_t = K̄_t ⊗ V_t  ∈ R^{d×d}        (outer product per row)
//     Z_t[i,j] = λ[i,j] · Z_{t-1}[i,j] + z_t[i,j]   (per-channel decay-and-add)
//     (initial state Z_0 = 0)
//
//   Step 4 — Recall (multiplicative interaction with Q):
//     O_t = Q_t · Z_t  ∈ R^{d}             (matrix-vector product per row)
//
//   Step 5 — Output projection:
//     y = O · W_O  ∈ R^{N×d}
//
// ============================================================================
//
// Implementation choices (single-head, simplified per the paper's "for
// simplicity" notes in Section 3.2):
//
//   * Single head (H = 1). The multi-head version is a reshape/permute; we
//     start with H = 1 to keep the math tractable and the gradient check
//     surgical. Future work: extend to multi-head.
//   * State dim = d. The shift-SSM state has d slots, holding the last d K
//     values. (No need to decouple state dim from head dim.)
//   * B = e_1, C = I. Fixed per the paper. This makes the shift SSM a pure
//     sliding window with no learnable projection inside the SSM itself.
//   * λ via sigmoid(λ_log). Same idea as Mamba's "A = -exp(A_log)" — keep
//     λ ∈ (0, 1) without constraint violation. λ_log is the unconstrained
//     parameter; we store and update λ_log.
//   * No D skip in the SSM step. The paper has D but its contribution is
//     minor; we omit it for clarity. (D_skip can be added later.)
//
// ----------------------------------------------------------------------------
//
// Backward pass:
//
//   Reverse order:
//
//   5. Output projection: standard Dense backward (dO → grad_y through W_O).
//   4. Recall: O_t = Q_t · Z_t
//        dQ_t = dO_t · Z_t^T           (matrix product)
//        dZ_t = Q_t^T · dO_t           (outer product Q_t ⊗ dO_t)
//   3. Diagonal SSM: Z_t = λ·Z_{t-1} + z_t
//      The forward recurrence accumulates a state per (i,j) channel pair.
//      The backward pass is a *reverse* recurrence over the same scalar
//      decayer: let G_t = dZ_t (the loss gradient at Z_t). Then the gradient
//      at z_t is the sum of all future grad-Z contributions, each weighted
//      by the appropriate λ-power, propagated backward from t = N-1:
//          dz_t = G_t + λ · G_{t+1} + λ^2 · G_{t+2} + ... + λ^{N-1-t} · G_{N-1}
//      This is equivalent to running a *forward* recurrence on the reversed
//      gradient sequence with the same scalar λ per channel. We implement it
//      as a backward sweep: dz_{N-1} = G_{N-1}; then for t = N-2 ... 0,
//      dz_t = G_t + λ * dz_{t+1}.
//      Then dλ[i,j] = sum_t G_{t+1}[i,j] * Z_t[i,j] (the cross-time term
//      from the dZ_t+1 = λ Z_t variation).
//      The gradient w.r.t. K̄ and V comes from the outer-product backward:
//          dK̄_t[i] = sum_j dZ_t_adjusted[i,j] · V_t[j]
//          dV_t[j]  = sum_i dZ_t_adjusted[i,j] · K̄_t[i]
//      where dZ_t_adjusted = dz_t (the gradient at z_t, which is what feeds
//      back into the outer product).
//   2. Shift SSM: K̄_t = [K_t, K_{t-1}, ..., K_{t-d+1}] (sliding window).
//      The backward pass is the *transpose* of the sliding-window structure:
//      dK_t = dK̄_t[0] + dK̄_{t+1}[1] + dK̄_{t+2}[2] + ... + dK̄_{t+d-1}[d-1]
//      (i.e., dK_t is the sum of dK̄ entries that "remembered" K_t at
//      positions [0, 1, 2, ..., d-1] of K̄_t, K̄_{t+1}, ..., K̄_{t+d-1}).
//   1. Q/K/V projections: standard Dense backward.
//   0. d_u = dQ + dK + dV (gradient w.r.t. the original input).
//
// All caches above are written in forward and read in backward.
// ============================================================================

class H3Block : public Layer {
public:
    // d_model:    input/output feature dim (= head dim in single-head mode)
    H3Block(size_t d_model);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_Q.weights; }
    Tensor get_gradients() const override { return W_Q.grad_weights; }
    std::string name() const override { return "H3Block"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }

    // Parameter tensors (public so tests can find them)
    Dense W_Q;             // (d_model -> d_model)
    Dense W_K;             // (d_model -> d_model)
    Dense W_V;             // (d_model -> d_model)
    Dense W_O;             // (d_model -> d_model)
    Tensor lambda_log;     // (d_model, d_model) — unconstrained; λ = sigmoid(λ_log)

    // Hidden gradient buffer for λ_log (set in backward, used in update_weights)
    Tensor grad_lambda_log_;

    // Forward caches (public so tests can inspect them).
    // Note: Tensor is 2D in this codebase, so the (T, d, d) diag-SSM state
    // sequence is flattened to (T*d, d), indexed as `last_Z_(t*d + i, j)`.
    Tensor last_input_;        // (T, d_model)
    Tensor last_Q_;            // (T, d_model)
    Tensor last_K_;            // (T, d_model)
    Tensor last_V_;            // (T, d_model)
    Tensor last_K_bar_;        // (T, d_model) — shift-SSM output
    Tensor last_lambda_;       // (d_model, d_model) — sigmoid(λ_log)
    Tensor last_Z_;            // (T*d_model, d_model) — diag-SSM state sequence
    size_t last_T_ = 0;        // last sequence length, for indexing last_Z_

private:
    size_t d_model_;

    // Helpers
    static double sigmoid(double x);
};

#endif // H3_H
