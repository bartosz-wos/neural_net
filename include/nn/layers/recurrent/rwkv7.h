#ifndef RWKV7_H
#define RWKV7_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RWKV-7 "Goose" — Peng et al. 2025
//   "RWKV-7 'Goose' with Expressive Dynamic State Evolution"
//   https://arxiv.org/abs/2503.14456
//
// RWKV-7 generalizes the delta rule with vector-valued state gating and
// per-channel in-context learning rates. The state update is:
//
//   wkv_t = wkv_{t-1} · (diag(w_t) − κ̂_t^T (a_t ⊙ κ̂_t))  +  v_t^T · k̃_t
//
// where:
//   - w_t ∈ (0,1)^d    data-dependent per-channel decay (vector-valued)
//   - a_t ∈ (0,1)^d    per-channel in-context learning rate (NEW in RWKV-7;
//                       was scalar in DeltaNet)
//   - κ_t = k_t ⊙ ξ    removal key (ξ is a learned per-channel multiplier)
//   - κ̂_t = κ_t / ||κ_t||_2   per-head L2 normalization (for numerical stability)
//   - k̃_t = k_t ⊙ lerp(1, a_t, α)   replacement key (α is a learned scalar)
//                          — decoupled from removal key κ_t (NEW in RWKV-7)
//
// Per-head: wkv_t[h] ∈ R^{m × m} where m = d / num_heads.
//
// ------------------------------------------------------------------------
// Mathematical formulation (v1 — single time-mixing block, no channel-mix):
//
//   Inputs:
//     x_t  in R^d     for t = 0..T-1
//
//   Step 1 — Token shift (Peng Eq. 2; we keep μ_□ as learnable per-channel):
//     x_t^□ = lerp(x_t, x_{t-1}, μ_□) = μ_□ ⊙ x_t + (1-μ_□) ⊙ x_{t-1}
//                                          for □ ∈ {r, k, v, d, a}
//
//   Step 2 — Projections (Dense; each is y = W · x + b with W shape (d,d)):
//     r_t = W_r · x_t^r + b_r
//     k_t = W_k · x_t^k + b_k
//     v_t = W_v · x_t^v + b_v
//     d_pre_t = W_d · x_t^d + b_d
//     a_pre_t = W_a · x_t^a + b_a
//
//   Step 3 — Decay, learning rate, and key derivation:
//     d_t       = tanh(d_pre_t)                                 in (-1, 1)
//     w_t       = exp(-exp(-0.5) · sigmoid(d_t))                in (0.687, 1)
//     a_t       = sigmoid(a_pre_t)                              in (0, 1)
//     κ_t       = k_t ⊙ ξ                                       (removal key)
//     κ̂_t      = κ_t / ||κ_t||_2                               (per-head L2 norm)
//     k̃_t       = k_t ⊙ lerp(1, a_t, α)
//                 = k_t ⊙ (1 + α · (a_t - 1))
//
//   Step 4 — Generalized delta-rule wkv update (per head):
//     G_t       = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)                (m × m)
//     wkv_t     = wkv_{t-1} · G_t  +  v_t^T · k̃_t               (m × m)
//     wkv_0     = 0
//
//   Step 5 — Output (per head; no LayerNorm / W_o / bonus in v1):
//     o_t[h, j] = sum_i wkv_t[h, i, j] · r_t[h, i]              (the receptance
//                                                                reads wkv_t)
//
//   State: wkv_t ∈ R^{num_heads × m × m}, the generalized-delta-rule fast
//   weights. No hidden cell state. This is the O(1)-per-step state that
//   gives RWKV-7 its constant-memory inference cost.
//
// ------------------------------------------------------------------------
// Initialization convention (matching RWKV-7 paper Appendix L "ξ in [-5.3,9.4]"):
//   * W_r, W_k, W_v, W_d, W_a: xavier-uniform (Dense default).
//   * b_r, b_k, b_v: zero.
//   * b_d: small positive (so initial w_t < 1, allowing state to evolve)
//   * b_a: zero (initial a_t = 0.5, neutral learning rate)
//   * xi: 1.0 (initial κ_t = k_t; standard RWKV-7 init per paper appendix)
//   * alpha: 0.0 (lerp(1, a_t, 0) = 1 → k̃_t = k_t, matching DeltaNet-like init)
//   * μ_r, μ_k, μ_v, μ_d, μ_a: 0.5 (50/50 mix of current and previous token)
//
// ------------------------------------------------------------------------
// Shape convention: forward takes a 2D input (T, d) and returns (T, d).
// wkv state is cached as (T+1, num_heads * m * m) where row t holds
// wkv_{t-1} flattened as [head_0_mat, head_1_mat, ...] (each head_h_mat
// is m×m stored row-major). BPTT traverses this cache backward.
// ============================================================================

class RWKV7TimeMix : public Layer {
public:
    // d: input/output feature dim (must be > 0 and divisible by num_heads)
    // num_heads: number of heads (default 1); head_dim = d / num_heads
    explicit RWKV7TimeMix(size_t d, size_t num_heads = 1);

    // Private validating helper used to throw before member init.
    RWKV7TimeMix(size_t d, size_t num_heads, bool validate_tag);

    // Forward pass on a full sequence.
    // input: (T, d)  ->  output: (T, d)
    Tensor forward(const Tensor& input) override;
    // Backward pass — grad_output: (T, d), returns grad_input: (T, d)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_r.weights; }
    Tensor get_gradients() const override { return W_r.grad_weights; }
    std::string name() const override { return "RWKV7TimeMix"; }

    // Accessors
    size_t d() const { return d_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

    // ---- Public parameters (for tests) ----
    size_t d_;
    size_t num_heads_;
    size_t head_dim_;
    Dense W_r;              // (d, d)  receptance projection
    Dense W_k;              // (d, d)  key projection
    Dense W_v;              // (d, d)  value projection
    Dense W_d;              // (d, d)  decay logit projection
    Dense W_a;              // (d, d)  learning-rate logit projection
    Tensor xi;              // (1, d)  removal-key multiplier
    Tensor alpha;           // (1, 1)  replacement-key interpolation
    Tensor mu_r;            // (1, d)  token-shift mix for r
    Tensor mu_k;            // (1, d)  token-shift mix for k
    Tensor mu_v;            // (1, d)  token-shift mix for v
    Tensor mu_d;            // (1, d)  token-shift mix for d
    Tensor mu_a;            // (1, d)  token-shift mix for a

    // Hidden gradient buffers
    Tensor grad_xi_;        // (1, d)
    Tensor grad_alpha_;     // (1, 1)
    Tensor grad_mu_r_;      // (1, d)
    Tensor grad_mu_k_;      // (1, d)
    Tensor grad_mu_v_;      // (1, d)
    Tensor grad_mu_d_;      // (1, d)
    Tensor grad_mu_a_;      // (1, d)

    // BPTT cache (public for tests; populated by forward, used by backward)
    Tensor last_input_;         // (T, d)
    Tensor last_x_shift_;       // (T, d)   x_{t-1}
    Tensor last_r_in_;          // (T, d)
    Tensor last_k_in_;          // (T, d)
    Tensor last_v_in_;          // (T, d)
    Tensor last_d_in_;          // (T, d)
    Tensor last_a_in_;          // (T, d)
    Tensor last_r_;             // (T, d)   r_pre — receptance projection
    Tensor last_k_;             // (T, d)   k_pre — key projection
    Tensor last_v_;             // (T, d)   v_pre — value projection
    Tensor last_d_pre_;         // (T, d)   d_pre — decay logit (before tanh)
    Tensor last_a_pre_;         // (T, d)   a_pre — learning-rate logit (before sigmoid)
    Tensor last_d_;             // (T, d)   tanh(d_pre)
    Tensor last_w_;             // (T, d)   exp(-exp(-0.5) · sigmoid(d_pre))
    Tensor last_a_;             // (T, d)   sigmoid(a_pre) — in-context learning rate
    Tensor last_kappa_;         // (T, d)   κ_t = k_t ⊙ ξ
    Tensor last_kappa_hat_;     // (T, d)   κ̂_t = κ_t / ||κ_t||_2 (per-head L2-normalized)
    Tensor last_kappa_norm_;    // (T, num_heads)   per-head ||κ_t||_2 (for backward)
    Tensor last_k_tilde_;       // (T, d)   k̃_t = k_t ⊙ lerp(1, a_t, α)
    Tensor last_wkv_;           // (T+1, num_heads * head_dim * head_dim)
                                //   row t holds wkv_{t-1} flattened

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
};

#endif // RWKV7_H