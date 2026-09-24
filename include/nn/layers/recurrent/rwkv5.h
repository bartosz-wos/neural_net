#ifndef RWKV5_H
#define RWKV5_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RWKV-5 "Eagle" — Peng et al. 2024
//   "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
//   https://arxiv.org/abs/2404.05892
//
// Eagle is RWKV-4 (Peng 2023) extended with:
//   (a) MATRIX-VALUED per-head states s_t ∈ R^{m × m} (head_dim m = d/H).
//       The kv contribution is the OUTER PRODUCT k_t[h, i] · v_t[h, j] (NOT
//       element-wise like RWKV-4).
//   (b) A vector-valued LEARNED DECAY `w ∈ R^d` (the vector is shared across
//       the sequence — NOT time-varying; that time-varying input-driven decay
//       is the Finch innovation, omitted here).
//
// The per-head WKV recurrence (paper §3.1, Eagle part):
//
//   kv_t[h, i, j]   = k_t[h, i] · v_t[h, j]                           ∈ R^{m×m}
//   a[h, i]         = exp(-exp(w[h * m + i]))                         ∈ (0, 1)
//   s_t[h, i, j]    = a[h, i] · s_{t−1}[h, i, j] + kv_t[h, i, j]      ∈ R^{m×m}
//   wkv_t[h, i, j]  = s_t[h, i, j] + u[h * m + i] · kv_t[h, i, j]    ∈ R^{m×m}
//   out_pre_t[h*m+i]= Σ_j r_t[h*m+i] · wkv_t[h, i, j]                  ∈ R^d     (before W_o)
//
// Output projection:
//   out_t = out_pre_t · W_o^T + b_o                                              (Dense)
//
// Token-shift (RWKV-4 style, NOT DD-Lerp):
//   r_in_t = μ_r ⊙ x_t + (1 − μ_r) ⊙ x_{t−1}
//   k_in_t = μ_k ⊙ x_t + (1 − μ_k) ⊙ x_{t−1}
//   v_in_t = μ_v ⊙ x_t + (1 − μ_v) ⊙ x_{t−1}
//   with x_{−1} := 0.
//
// ----------------------------------------------------------------------------
// Initialization convention (Eagle, Appendix I):
//   * W_r, W_k, W_v, W_o: xavier-uniform (Dense default).
//   * b_r, b_k, b_v, b_o: zero.
//   * log_w: −0.5 init (Eagle's Appendix I value — a moderate decay, not the
//     very slow RWKV-4 log_w=−5).
//   * u: 0.0 init (no current-token bonus at start).
//   * μ_r, μ_k, μ_v: 0.5 init (50/50 mix of current and previous).
//
// ----------------------------------------------------------------------------
// State shape: s_t ∈ R^{H × m × m} cached flat as (T+1, H·m·m). Row 0 is the
// initial zero state. The state-after-step-t lives in row t+1.
//
// BPTT iterates backward in `backward` (state carrier grad_s_prev * a).
//
// v1 scope: time-mixing only. Channel-mixing is left for a follow-up, matching
// the rwkv4 / rwkv6 / rwkv7 time-mix-only convention in this repo.
// ============================================================================

class RWKV5TimeMix : public Layer {
public:
    // d: input/output feature dim (must be > 0 and divisible by num_heads).
    // num_heads: number of heads (default 1); head_dim m = d / num_heads.
    explicit RWKV5TimeMix(size_t d, size_t num_heads = 1);
    ~RWKV5TimeMix() override;

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
    std::string name() const override { return "RWKV5TimeMix"; }

    // Accessors
    size_t d() const { return d_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

    // ---- Public parameters (for tests) ----
    size_t d_;
    size_t num_heads_;
    size_t head_dim_;

    // Projections: each W_□ has shape (d, d), bias (1, d).
    Dense W_r;
    Dense W_k;
    Dense W_v;
    Dense W_o;

    // Per-channel decay parameter (unconstrained; a[i] = exp(-exp(log_w[i])) ∈ (0, 1))
    Tensor log_w;           // (1, d)
    // Per-channel "current-token bonus" weight (u[h * m + i] multiplies kv[h, i, j] in wkv)
    Tensor u;               // (1, d)
    // Token-shift mix coefficients
    Tensor mu_r;            // (1, d)
    Tensor mu_k;            // (1, d)
    Tensor mu_v;            // (1, d)

    // Hidden gradient buffers
    Tensor grad_log_w_;     // (1, d)
    Tensor grad_u_;         // (1, d)
    Tensor grad_mu_r_;      // (1, d)
    Tensor grad_mu_k_;      // (1, d)
    Tensor grad_mu_v_;      // (1, d)

    // ---- BPTT cache (public for tests; populated by forward, used by backward) ----
    Tensor last_input_;          // (T, d)
    Tensor last_x_shift_;        // (T, d)  x_{t-1} (row 0 = zeros)

    // Token-shift output
    Tensor last_r_in_;           // (T, d)
    Tensor last_k_in_;           // (T, d)
    Tensor last_v_in_;           // (T, d)

    // Projections
    Tensor last_r_;              // (T, d)
    Tensor last_k_;              // (T, d)
    Tensor last_v_;              // (T, d)

    // kv outer product cache, flat (i, j) per head: (T, num_heads * m * m)
    Tensor last_kv_;             // (T, num_heads * m * m)

    // State cache: row t is s_{t-1} flat (head * i * m + j)
    Tensor last_s_;              // (T + 1, num_heads * m * m)

    // wkv cache, flat (i, j) per head: (T, num_heads * m * m)
    Tensor last_wkv_;            // (T, num_heads * m * m)

    // Per-token r·wkv (output before W_o): (T, d)
    Tensor last_out_pre_;        // (T, d)

    // Per-channel decay a[h*m+i] = exp(-exp(log_w[i]))
    Tensor last_a_;              // (1, d)

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
};

#endif // RWKV5_H