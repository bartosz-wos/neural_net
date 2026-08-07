#ifndef DELTANET_H
#define DELTANET_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// DeltaNet — Yang et al. 2024 (NeurIPS 2024)
//   "Linear Attention with the Delta Rule"
//   https://arxiv.org/abs/2406.06484
//
// Linear attention (Katharopoulos et al. 2020) replaces the quadratic softmax
// attention with a linear-time recurrence:
//
//   S_t = S_{t-1} + outer(k_t, v_t)
//   o_t = S_t · q_t
//
// Linear attention has the same expressive bottleneck as any linear function:
// it cannot perform "in-context copying" — once a value is stored, it cannot
// be updated later based on what the model has retrieved. The delta rule
// (Schlag et al. 2021) fixes this by replacing the value with the *residual*
// (the "new information" the state hasn't seen yet):
//
//   S_t = S_{t-1} + α_t · outer(k_t,  v_t - S_{t-1}·k_t)
//   o_t = S_t · q_t
//
// where α_t = 1 / (1 + k_t · S_{t-1} · k_t) is a normalization derived from
// solving the delta-rule optimality condition (delta-update ≡ fast weight
// programming "SGD on the retrieved pattern with learning rate 1/L").
//
// Yang et al. 2024 add a learned k-projection magnitude (the "beta" parameter,
// with |k_t| = sqrt(k_t · k_t) per-coordinate normalization) to stabilize the
// recurrence without losing the delta-rule benefits:
//
//   k_t ← (β / |k_t|) · k_t
//
// This bounds |k_t| away from 0 and ∞, keeping α_t numerical-safe.
//
// ----------------------------------------------------------------------------
// Architecture (single DeltaNet layer, multi-head):
//
// Input:  X in R^{T x d_model}      (T sequence positions)
//
// Step 1 — Projections (Dense: y = x·W^T + b):
//   q_t = W_Q · x_t                                  in R^{d_inner}
//   k_t = W_K · x_t                                  in R^{d_inner}
//   v_t = W_V · x_t                                  in R^{d_inner}
//   β_t = sigmoid(W_β · x_t)                         in R^{H}        (scalar per head)
//
// Step 2 — Per-head k-magnitude normalization (delta-rule paper §3.3):
//   for each head h:
//     k_t[h] *= (β_t[h] / |k_t[h]|)                 (|k| = per-head L2 norm)
//
// Step 3 — Per-head delta-rule recurrence (single-step, parallelisable):
//   S_0^(h) = 0 in R^{d x d}
//   for t = 0..T-1:
//     α_t[h] = 1 / (1 + k_t[h] · S_{t-1}^(h) · k_t[h])             (scalar per head)
//     S_t^(h) = S_{t-1}^(h) + α_t[h] · outer(k_t[h], v_t[h] - S_{t-1}^(h)·k_t[h])
//     o_t[h] = S_t^(h) · q_t[h]                                       (vector in R^d)
//
// Step 4 — Concat heads + output projection:
//   o_t = concat([o_t[0]; ...; o_t[H-1]])               in R^{H*d = d_inner}
//   out_t = W_O · o_t                                  in R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   X:               (T, d_model)
//   Q, K, V:         (T, d_inner = H * d)
//   β:               (T, H)
//   S (per head):    (d, d) — single matrix state per head per time step
//   o:               (T, d_inner)
//   W_O output:      (T, d_model)
//
// We use head_dim = d_inner / n_heads. The default d_inner = d_model (so
// head_dim = d_model / n_heads). For numerical gradient checks we use
// d_model=4, n_heads=2, head_dim=2, T=3-4.
//
// ----------------------------------------------------------------------------
// Learnable parameters: W_Q, W_K, W_V, W_O, W_β. All stored as Dense layers
// (in_features, out_features) so they get the standard forward/backward/zero_grad
// hook.
// ============================================================================

class DeltaNet : public Layer {
public:
    DeltaNet(size_t d_model, size_t n_heads, size_t head_dim = 0);

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
    std::string name() const override { return "DeltaNet"; }

    // Test introspection: per-head state at time T (last call).
    // Returns S_T for each head, stacked shape (n_heads, head_dim * head_dim).
    Tensor last_state() const;

    // Setters / accessors for inspectors
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t d_inner() const { return d_inner_; }

    // Public Dense accessors (for testing)
    Dense W_Q_, W_K_, W_V_, W_O_, W_beta_;

private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    size_t d_inner_;

    // Cache for backward (all per-token, per-head or concatenated)
    Tensor cache_x_;           // (T, d_model)
    Tensor cache_q_;           // (T, d_inner)
    Tensor cache_k_;           // (T, d_inner)  — RAW k (before scaling)
    Tensor cache_v_;           // (T, d_inner)
    Tensor cache_k_scaled_;    // (T, d_inner)  — k_t after β/|k_t| rescaling
    Tensor cache_beta_pre_;    // (T, n_heads)  — pre-sigmoid
    Tensor cache_beta_;        // (T, n_heads)  — sigmoid
    Tensor cache_alpha_;       // (T, n_heads)  — α_t scalar per head
    Tensor cache_concat_o_;    // (T, d_inner) — what W_O sees

    // State cache: per-head, per-time. cache_S_[t] is (n_heads, head_dim*head_dim).
    // cache_S_[0] = zero state (before any input).
    std::vector<Tensor> cache_S_;

    // Local gradient buffers (for the recurrence)
    Tensor grad_q_;            // (T, d_inner)
    Tensor grad_k_;            // (T, d_inner)  — RAW k
    Tensor grad_v_;            // (T, d_inner)
    Tensor grad_k_scaled_;     // (T, d_inner)
    Tensor grad_x_;            // (T, d_model) — returned by backward()
};

#endif // DELTANET_H
