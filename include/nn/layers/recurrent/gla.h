#ifndef GLA_H
#define GLA_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Gated Linear Attention (GLA) — Yang et al. 2023/2024
//   "Gated Linear Attention Transformers with Hardware-Efficient Training"
//   https://arxiv.org/abs/2312.06635  (NeurIPS 2024)
//
// Standard linear attention (Katharopoulos et al. 2020) replaces softmax
// attention with a linear-time recurrence:
//
//   S_t = S_{t-1} + outer(k_t, v_t)
//   o_t = S_t · q_t
//
// This is fast (O(d) per step) but cannot selectively forget stale
// information — the state grows unboundedly. GLA fixes this by adding an
// INPUT-DEPENDENT per-head forget gate α_t ∈ (0, 1):
//
//   S_t = α_t · S_{t-1} + outer(k_t, v_t)
//   o_t = S_t · q_t
//
// with α_t = sigmoid(W_gate · x_t), a small per-head scalar projection.
// The gate is the key ingredient closing the gap to softmax attention on
// language modeling while preserving O(T·d²) recurrence.
//
// ----------------------------------------------------------------------------
// Per-head, per-time recurrence (single-step BPTT):
//
// Input:  X in R^{T x d_model}
//
// Step 1 — Projections (Dense: y = x·W^T + b):
//   q_t = W_Q · x_t                    in R^{d_inner}
//   k_t = W_K · x_t                    in R^{d_inner}
//   v_t = W_V · x_t                    in R^{d_inner}
//   α_t = sigmoid(W_gate · x_t)        in R^{H}        (scalar per head)
//
// Step 2 — Per-head gated linear-attention recurrence:
//   S_0^(h) = 0 in R^{d x d}
//   for t = 0..T-1:
//     S_t^(h) = α_t[h] · S_{t-1}^(h) + outer(k_t[h], v_t[h])
//     o_t[h] = S_t^(h) · q_t[h]                              (vector in R^d)
//
// Step 3 — Concat heads + output projection:
//   o_t = concat([o_t[0]; ...; o_t[H-1]])               in R^{H*d = d_inner}
//   out_t = W_O · o_t                                  in R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   X:               (T, d_model)
//   Q, K, V:         (T, d_inner = H * d)
//   α:               (T, H)
//   S (per head):    (d, d) — single matrix state per head per time step
//   o:               (T, d_inner)
//   W_O output:      (T, d_model)
//
// We use head_dim = d_inner / n_heads (must divide evenly). For numerical
// gradient checks we use d_model=4, n_heads=2, head_dim=2, T=3-4.
//
// ----------------------------------------------------------------------------
// Learnable parameters: W_Q, W_K, W_V, W_O, W_gate. All stored as Dense
// layers so they get the standard forward/backward/zero_grad hook.
// ============================================================================

class GatedLinearAttention : public Layer {
public:
    GatedLinearAttention(size_t d_model, size_t n_heads, size_t head_dim = 0);

    // Input: (T, d_model) tensor. Output: (T, d_model) tensor.
    Tensor forward(const Tensor& input) override;

    // Standard backward: receives grad_output (T, d_model) and returns
    // grad_input (T, d_model). Internal Dense projections accumulate their
    // own gradients via the standard Layer convention.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    // Parameter accessors (for optimizer discovery)
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_Q_.weights; }
    Tensor get_gradients() const override { return W_Q_.grad_weights; }
    std::string name() const override { return "GatedLinearAttention"; }

    // Test introspection: per-head state at time T (last call).
    // Returns S_T for each head, stacked shape (n_heads, head_dim * head_dim).
    Tensor last_state() const;

    // Per-head, per-token gate (post-sigmoid) from the most recent forward
    // call. Shape (T, n_heads). Empty Tensor when forward has not been called.
    Tensor last_gates() const;

    // Setters / accessors for inspectors
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t d_inner() const { return d_inner_; }

    public:
    // Public Dense accessors (for testing)
    Dense W_Q_, W_K_, W_V_, W_O_, W_gate_;

private:
    // Cache for backward (all per-token, per-head)
    Tensor cache_x_;           // (T, d_model)
    Tensor cache_q_;           // (T, d_inner)
    Tensor cache_k_;           // (T, d_inner)
    Tensor cache_v_;           // (T, d_inner)
    Tensor cache_gate_pre_;    // (T, n_heads) — pre-sigmoid
    Tensor cache_gate_;        // (T, n_heads) — sigmoid (the α_t scalar per head)
    Tensor cache_concat_o_;    // (T, d_inner) — what W_O sees

    // State cache: per-head, per-time. cache_S_[t] is (n_heads, head_dim * head_dim).
    // cache_S_[0] = zero state (before any input).
    std::vector<Tensor> cache_S_;

    // Local gradient buffers (for the recurrence)
    Tensor grad_q_;            // (T, d_inner)
    Tensor grad_k_;            // (T, d_inner)
    Tensor grad_v_;            // (T, d_inner)
    Tensor grad_x_;            // (T, d_model) — returned by backward()

    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    size_t d_inner_;
};

#endif // GLA_H