#ifndef GATED_DELTANET_H
#define GATED_DELTANET_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Gated DeltaNet — Yang, Kautz, Hatamizadeh (ICLR 2025)
//   "Gated Delta Networks: Improving Mamba2 with Delta Rule"
//   https://arxiv.org/abs/2412.06464
//
// Combines two complementary mechanisms for linear-time sequence modelling:
//
//   (1) Mamba2/GLA-style per-head decay gate  α_t ∈ (0,1)  (Yang et al. 2024a,
//       Dao & Gu 2024): uniform memory erasure.
//
//   (2) DeltaNet-style per-head write strength β_t ∈ (0,1) with k-magnitude
//       normalization (Schlag et al. 2021, Yang et al. 2024b): targeted
//       updates of one key-value association at a time.
//
// Final per-head h, per-time t recurrence:
//
//   S_t[h] = α_t[h] · S_{t-1}[h] · (I − β_t[h] · outer(k_t[h], k_t[h]))
//          + β_t[h] · outer(k_t[h], v_t[h])
//   o_t[h] = S_t[h] · q_t[h]
//
// where k_t[h] is the per-head β-scaled k_t, and I is the head_dim×head_dim
// identity matrix. The state S_t is per-head (head_dim × head_dim), and the
// gate α_t and write-strength β_t are scalar per head.
//
// ----------------------------------------------------------------------------
// Architecture (single Gated DeltaNet layer, multi-head):
//
// Input:  X in R^{T x d_model}      (T sequence positions)
//
// Step 1 — Projections (Dense: y = x·W^T + b):
//   q_t   = W_Q · x_t                                  in R^{d_inner}
//   k_t   = W_K · x_t                                  in R^{d_inner}
//   v_t   = W_V · x_t                                  in R^{d_inner}
//   β_t_pre  = W_β · x_t;    β_t = sigmoid(β_t_pre)    in R^H        (scalar per head)
//   gate_pre = W_gate · x_t; gate = sigmoid(gate_pre)  in R^H        (scalar per head)
//
// Step 2 — Per-head k-magnitude normalization (DeltaNet §3.3):
//   for each head h:
//     k_t[h] *= (β_t[h] / ||k_t[h]||)                  (||k|| = per-head L2 norm)
//
// Step 3 — Per-head gated-delta recurrence:
//   S_0^(h) = 0 in R^{head_dim × head_dim}
//   for t = 0..T-1:
//     S_t^(h) = gate_t[h] · S_{t-1}^(h) · (I − β_t[h] · outer(k_t[h], k_t[h]))
//             + β_t[h] · outer(k_t[h], v_t[h])
//     o_t[h]  = S_t^(h) · q_t[h]                                    in R^{head_dim}
//
// Step 4 — Concat heads + output projection:
//   o_t = concat([o_t[0]; ...; o_t[H-1]])               in R^{H*head_dim = d_inner}
//   out_t = W_O · o_t                                   in R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   X:               (T, d_model)
//   Q, K, V:         (T, d_inner = H * head_dim)
//   β, gate:         (T, H)        (scalar per head)
//   S (per head):    (head_dim, head_dim)
//   o:               (T, d_inner)
//   W_O output:      (T, d_model)
//
// We use head_dim = d_inner / n_heads. The default d_inner = d_model (so
// head_dim = d_model / n_heads). Requires d_model % n_heads == 0.
//
// ----------------------------------------------------------------------------
// Learnable parameters (6 Dense layers × 2 = 12 tensors):
//   W_Q, W_K, W_V, W_O  — standard 4-projection setup
//   W_β                  — write strength (delta-rule)
//   W_gate               — per-head decay gate (Mamba2-style)
// ============================================================================

class GatedDeltaNet : public Layer {
public:
    GatedDeltaNet(size_t d_model, size_t n_heads, size_t head_dim = 0);

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
    std::string name() const override { return "GatedDeltaNet"; }

    // Test introspection: per-head state at time T (last call).
    // Returns S_T for each head, stacked shape (n_heads, head_dim * head_dim).
    Tensor last_state() const;

    // Setters / accessors for inspectors
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t d_inner() const { return d_inner_; }

    // Public Dense accessors (for testing)
    Dense W_Q_, W_K_, W_V_, W_O_, W_beta_, W_gate_;

private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    size_t d_inner_;

    // Cache for backward
    Tensor cache_x_;             // (T, d_model)
    Tensor cache_q_;             // (T, d_inner)
    Tensor cache_k_;             // (T, d_inner) — RAW k (before scaling)
    Tensor cache_v_;             // (T, d_inner)
    Tensor cache_k_scaled_;      // (T, d_inner) — k_t after β/|k| rescaling
    Tensor cache_beta_pre_;      // (T, n_heads)
    Tensor cache_beta_;          // (T, n_heads)
    Tensor cache_gate_pre_;      // (T, n_heads)
    Tensor cache_gate_;          // (T, n_heads)
    Tensor cache_concat_o_;      // (T, d_inner) — what W_O sees

    // State cache: per-head, per-time. cache_S_[t] is (n_heads, head_dim * head_dim).
    // cache_S_[0] = zero state (before any input).
    std::vector<Tensor> cache_S_;

    // Local gradient buffers (for the recurrence)
    Tensor grad_q_;              // (T, d_inner)
    Tensor grad_k_;              // (T, d_inner) — RAW k
    Tensor grad_v_;              // (T, d_inner)
    Tensor grad_k_scaled_;       // (T, d_inner)
    Tensor grad_x_;              // (T, d_model) — returned by backward()
};

#endif // GATED_DELTANET_H