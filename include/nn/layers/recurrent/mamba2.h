#ifndef MAMBA2_H
#define MAMBA2_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Mamba-2 / SSD — Dao & Gu 2024
//   "Transformers are SSMs: Structured State Space Duality"
//
// Mamba-2 is the "structured state space duality" (SSD) layer that unifies
// linear attention and state-space models. The key insight is that, with a
// specific *structured* mask (causal + exponential scalar decay), the SSM
// recurrence is *equivalent* to linear attention with a structured mask:
//
//   SSM:        H_t = a_t * H_{t-1} + b_t k_t^T,    o_t = q_t · H_t
//   Attention:  O = (Q K^T ⊙ M) V     with M_{t,s} = a_t * a_{t-1} * ... * a_{s+1}  for s<t
//
// where a_t is a *scalar* decay (same across all state columns for H_t).
//
// This gives 3 equivalent views:
//   1. Recurrence (sequential, O(1) state per step — efficient inference)
//   2. Linear attention with structured mask (parallel training)
//   3. 1D convolution (chunked SSD — the practical training form)
//
// We implement the recurrence form (1) since it's the most pedagogical and
// the gradient check is tractable. This is the "Mamba-2 single head
// recurrence" view, which is what most Mamba-2 codebases ship as the inner
// loop of the chunked algorithm.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (per sequence position t, multi-head):
//
//   Input:  x_t in R^{d_model}      (t = 0..T-1)
//
//   Step 1 — Input projection (gate path included, like Mamba):
//     p_t = in_proj(x_t)                 in R^{2 * d_inner}
//     x_ssm_t = p_t[:d_inner]            (raw SSM path input, no activation here)
//     g_t     = p_t[d_inner:]            (gate path input)
//
//   Step 2 — Per-head decay, value, key, query (all input-dependent — the
//            "selective" innovation carried over from Mamba-1):
//     a_t   = sigmoid(dt_bias + a_proj(x_t))     in R^{n_heads}     (per-head scalar decay)
//     b_t   = b_proj(x_t)                         in R^{d_inner}     (per-channel value)
//     k_t   = k_proj(x_t)                         in R^{n_heads * head_dim}    (keys, head_dim = d_inner / n_heads)
//     q_t   = q_proj(x_t)                         in R^{n_heads * head_dim}    (queries)
//
//   Step 3 — Selective SSD recurrence (the matrix-state variant):
//     H_0 = 0
//     For t = 0..T-1:
//       H_t = diag(a_t) @ H_{t-1} + outer(b_t, k_t)       (H_t is (d_inner, head_dim) — but with scalar a per head, this is per head block)
//       o_t = matmul(q_t, H_t)                              (output per head, then concat)
//
//   Because a_t is per-head scalar, H_t's structure is block-diagonal across heads.
//   For head h, the (d_inner_h, head_dim) block of H evolves as:
//     H_t[h] = a_t[h] * H_{t-1}[h] + outer(b_t[h], k_t[h])
//     o_t[h] = q_t[h] · H_t[h]    (sum over head_dim)
//
//   Step 4 — Skip + gating + output projection:
//     y_t = o_t + D ⊙ x_ssm_t                              (skip-connection, like Mamba)
//     gated_t  = silu(g_t) ⊙ y_t                            (gated like Mamba)
//     out_t    = out_proj(gated_t)                          in R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   * x:               (T, d_model)
//   * in_proj output:  (T, 2*d_inner)         weights: (2*d_inner, d_model)
//   * a_proj output:   (T, n_heads)           weights: (n_heads, d_model)
//   * b_proj output:   (T, d_inner)           weights: (d_inner, d_model)
//   * k_proj output:   (T, d_inner)           weights: (d_inner, d_model)   [flatten (T, n_heads*head_dim)]
//   * q_proj output:   (T, d_inner)           weights: (d_inner, d_model)   [same flatten]
//   * out_proj weights:(d_model, d_inner)
//
// We use head_dim = d_inner / n_heads (must divide evenly). For numerical
// gradient checks we use small d_inner, n_heads (e.g. d_inner=4, n_heads=2,
// head_dim=2).
//
// ----------------------------------------------------------------------------
// Initialization convention (following Mamba-2 paper):
//   * a_proj bias (dt_bias) is initialized to small positive values (log of
//     initial decay). We use 0.0 init for dt_bias and let the projected
//     a values start near 0.5 after sigmoid — this is a "soft" open gate.
//   * All projection biases initialized to 0 except dt_bias (1.0 for decay
//     bias — convention is to start with decay slightly away from 0 so the
//     exponential recurrence has nontrivial history).
//   * D (skip) initialized to 1.0 (matching Mamba-1 convention).
// ============================================================================

class Mamba2Block : public Layer {
public:
    // d_model:     input/output feature dim
    // n_heads:     number of attention heads (also the SSD "head" structure)
    // d_inner:     inner feature dim (default = 2 * d_model, matching Mamba-1)
    //              must be divisible by n_heads
    Mamba2Block(size_t d_model, size_t n_heads, size_t d_inner = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "Mamba2Block"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t d_inner() const { return d_inner_; }
    size_t head_dim() const { return head_dim_; }

    // Parameter tensors (public for gradient tests)
    Dense in_proj;          // (d_model -> 2*d_inner)
    Dense out_proj;         // (d_inner -> d_model)
    Dense a_proj;           // (d_model -> n_heads)        — pre-sigmoid log-decay
    Dense b_proj;           // (d_model -> d_inner)         — value
    Dense k_proj;           // (d_model -> d_inner)         — key (flattened across heads)
    Dense q_proj;           // (d_model -> d_inner)         — query (flattened across heads)
    Tensor D_skip;          // (1, d_inner)                 — learnable skip connection
    Tensor dt_bias;         // (1, n_heads)                 — bias added before sigmoid (a_proj-bias for clarity)

    // Hidden gradient buffers (for D_skip and dt_bias which are not Dense)
    Tensor grad_D_skip_;    // (1, d_inner)
    Tensor grad_dt_bias_;   // (1, n_heads)

private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_inner_;
    size_t head_dim_;       // = d_inner / n_heads

    // Caches for forward (filled in forward, used in backward)
    Tensor last_input_;         // (T, d_model)
    Tensor last_p_;             // (T, 2*d_inner)   in_proj output
    Tensor last_x_ssm_;         // (T, d_inner)     raw ssm path input (p[:, :d_inner])
    Tensor last_gate_;          // (T, d_inner)     gate path (p[:, d_inner:])
    Tensor last_a_pre_;         // (T, n_heads)     pre-sigmoid decay (a_proj output)
    Tensor last_a_;             // (T, n_heads)     sigmoid(a_pre) — scalar per-head decay
    Tensor last_b_;             // (T, d_inner)     b_proj output
    Tensor last_k_;             // (T, d_inner)     k_proj output (head_dim * n_heads flattened)
    Tensor last_q_;             // (T, d_inner)     q_proj output (head_dim * n_heads flattened)

    // H cache: per-head state, stored as a single (T+1, d_inner, head_dim) tensor
    // flattened to ((T+1)*d_inner, head_dim) — same convention as Mamba-1.
    // Index (t, i, j) at row t*d_inner + i, col j.
    Tensor last_H_;             // ((T+1)*d_inner, head_dim)
    Tensor last_o_;             // (T, d_inner)     SSM output per token
    Tensor last_gated_;         // (T, d_inner)     silu(gate) ⊙ o

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double silu(double x) {
        return x * sigmoid(x);
    }
};

#endif // MAMBA2_H
