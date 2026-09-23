#ifndef RWKV6_H
#define RWKV6_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RWKV-6 "Finch" — Peng et al. 2024
//   "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
//   https://arxiv.org/abs/2404.05892
//
// Finch = RWKV-4 with (a) matrix-valued states like Eagle/RWKV-5, and
// (b) **time-varying per-channel decay** `w_t ∈ (0, 1)^d` driven by the
// input (vs RWKV-5's static learned `w` vector). The decay path uses a
// LoRA-style augmentation: `w_t = exp(-exp(ddlerpd(x_t, x_{t-1})))`, and
// the four `r, k, v, g` projections also use DD-Lerp (data-dependent
// linear interpolation) which is `□_t = ddlerp_□(x_t, x_{t-1}) · W_□`.
//
// DD-Lerp (Eq. 14-15):
//   lora_□(x) = λ_□ + tanh(x · A_□) · B_□
//   ddlerp_□(a, b) = a + (b − a) ⊙ lora_□(a + (b − a) ⊙ µ_x)
//
// Per-head matrix-valued state s_t ∈ R^{m × m} (m = d / num_heads):
//   kv_t   = k_t^T · v_t                                       ∈ R^{m × m}
//   s_t    = diag(w_t[h]) · s_{t−1} + kv_t                    ∈ R^{m × m}
//   wkv_t  = s_t + diag(u[h]) · kv_t                          ∈ R^{m × m}    (bonus via u)
//   o_t[h, j] = Σ_i r_t[h, i] · wkv_t[h, i, j]                ∈ R^m
//
// Output (Eq. 20):
//   out_t = concat_h(SiLU(g_t[h]) ⊙ LayerNorm_h(r_t · wkv_t)) · W_o
//
// Per Finch Appendix I:
//   * W_r, W_k, W_v, W_g, W_d: xavier-uniform init (Dense default).
//   * W_o: zero init (output is exactly 0 at the start — training warmup).
//   * b_r, b_k, b_v, b_g, b_d: zero.
//   * u ∈ R^d: r0 · (1 − i / (D − 1)) + 0.1·((i + 1) mod 3), r0 = 0.5 for v1.
//   * µ_x ∈ R^d: 1 − (i / D)^r1, r1 = 1.0 (Eq. 14 implicit per Appendix I).
//   * λ_□ ∈ R^d: 0 (LoRA path is inactive at init → Finch reduces to RWKV-4/5
//     style token-shift at start, then deviates as λ_□ receives gradients).
//   * A_□ and B_□: U(−10^{-4}, 10^{-4}) (Appendix I).
//
// State shape: s_t ∈ R^{num_heads × m × m}, cached as (T+1, num_heads·m·m).
// BPTT traverses this cache backward.
//
// v1 scope: time-mixing only. Channel-mixing (the standard 3.5D FFN with
// token-shift) is left for a follow-up — matches the existing rwkv4 / rwkv7
// time-mix-only convention in this repo.
//
// v1 also OMITS the per-head LayerNorm in the output (Eq. 20): we pass `r·wkv`
// directly through W_o. This keeps the gradient check tractable at small
// (d, T) and matches the convention used in rwkv4 / rwkv7 layers where no
// internal LayerNorm is added. Adding the per-head LN later is a clean
// extension.
// ============================================================================

class RWKV6TimeMix : public Layer {
public:
    // d: input/output feature dim (must be > 0 and divisible by num_heads).
    // num_heads: number of heads (default 1); head_dim = d / num_heads.
    // num_lora_ranks: LoRA rank R used by r/k/v/g/x (default 32 per Appendix I).
    //   For the d-direction LoRA, the rank is 2·num_lora_ranks (Appendix I).
    explicit RWKV6TimeMix(size_t d, size_t num_heads = 1, size_t num_lora_ranks = 32);
    ~RWKV6TimeMix() override;

    // Private validating helper used to throw before member init.
    RWKV6TimeMix(size_t d, size_t num_heads, size_t num_lora_ranks, bool validate_tag);

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
    std::string name() const override { return "RWKV6TimeMix"; }

    // Accessors
    size_t d() const { return d_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t num_lora_ranks() const { return num_lora_ranks_; }

    // ---- Public parameters (for tests) ----
    size_t d_;
    size_t num_heads_;
    size_t head_dim_;
    size_t num_lora_ranks_;

    // Projections: each W_□ has shape (d, d), bias (1, d).
    // Note: there is NO W_d — per Eq. 17, `d_t = lorad(ddlerpd(x_t, x_{t-1}))`
    // directly (the d-direction uses only the LoRA-d path, no Dense projection).
    Dense W_r;
    Dense W_k;
    Dense W_v;
    Dense W_g;
    Dense W_o;

    // Token-shift mix (Eagle convention; Finch uses same name, slightly different init).
    Tensor u;             // (1, d)  time-first bonus weight (per channel, used in diag form)
    Tensor mu_x;          // (1, d)  shared DD-Lerp base mix for r/k/v/g/x
    Tensor mu_w;          // (1, d)  Finch-specific DD-Lerp base mix for d-direction (Appendix I)

    // LoRA path: for each □ ∈ {r, k, v, g, x} the rank is R = num_lora_ranks_.
    // For d-direction the rank is 2*R per Appendix I.
    //   A_□ : (d, R_□)    B_□ : (R_□, d)    λ_□ : (1, d)
    Tensor A_r, B_r, lambda_r;
    Tensor A_k, B_k, lambda_k;
    Tensor A_v, B_v, lambda_v;
    Tensor A_g, B_g, lambda_g;
    Tensor A_d, B_d, lambda_d;     // d-direction uses rank = 2*R
    Tensor A_x, B_x, lambda_x;

    // ---- Hidden gradient buffers (for non-Dense params) ----
    Tensor grad_u_;
    Tensor grad_mu_x_;
    Tensor grad_mu_w_;
    Tensor grad_lambda_r_, grad_lambda_k_, grad_lambda_v_, grad_lambda_g_, grad_lambda_d_, grad_lambda_x_;
    Tensor grad_A_r_, grad_B_r_;
    Tensor grad_A_k_, grad_B_k_;
    Tensor grad_A_v_, grad_B_v_;
    Tensor grad_A_g_, grad_B_g_;
    Tensor grad_A_d_, grad_B_d_;
    Tensor grad_A_x_, grad_B_x_;

    // ---- BPTT cache (public for tests; populated by forward, used by backward) ----
    Tensor last_input_;          // (T, d)
    Tensor last_x_prev_;         // (T, d)  x_{t-1} (zero in row 0) — INPUT cache
    Tensor last_x_shift_;        // (T, d)  ddlerp_x(x_t, x_{t-1}) — OUTPUT cache

    // Per-token-shift cache (ddlerp output before the W_□ projection)
    Tensor last_r_shift_;        // (T, d)  ddlerp_r(x_t, x_{t-1})
    Tensor last_k_shift_;        // (T, d)
    Tensor last_v_shift_;        // (T, d)
    Tensor last_g_shift_;        // (T, d)
    Tensor last_d_shift_;        // (T, d)  ddlerpd(x_t, x_{t-1})

    // LoRA intermediate cache (for the LoRA path's backprop)
    Tensor last_lora_r_;         // (T, d)  lora_r(input_to_lora_r)
    Tensor last_lora_k_;         // (T, d)
    Tensor last_lora_v_;         // (T, d)
    Tensor last_lora_g_;         // (T, d)
    Tensor last_lora_d_;         // (T, d)
    Tensor last_lora_x_;         // (T, d)  lora_x(shifted-base)

    // LoRA-input cache: the input to the LoRA function (before tanh(·A))
    Tensor last_lora_r_in_;      // (T, d)
    Tensor last_lora_k_in_;
    Tensor last_lora_v_in_;
    Tensor last_lora_g_in_;
    Tensor last_lora_d_in_;
    Tensor last_lora_x_in_;      // (T, d)  a + (b − a) ⊙ µ_x = x_t + (x_{t-1} − x_t) ⊙ µ_x

    // tanh-cached outputs (for backward)
    Tensor last_tanh_r_;         // (T, d)
    Tensor last_tanh_k_;
    Tensor last_tanh_v_;
    Tensor last_tanh_g_;
    Tensor last_tanh_d_;
    Tensor last_tanh_x_;

    // Projections after token-shift
    Tensor last_r_;              // (T, d)  r_t  = r_shift · W_r^T + b_r
    Tensor last_k_;              // (T, d)
    Tensor last_v_;              // (T, d)
    Tensor last_g_;              // (T, d)
    Tensor last_d_pre_;          // (T, d)  pre-w: d_t = lora_d(d_shift) — the LoRA output
                                 // Note: paper Eq. 17 says d_t = lorad(ddlerpd(...)); we
                                 // store the lora-d output as `last_d_pre` (already
                                 // includes the DD-Lerp + LoRA-d augmentation).

    // Per-token time-varying decay (the Finch innovation!)
    Tensor last_w_;              // (T, d)  w_t = exp(-exp(d_t))

    // Per-token kv outer-product cache (so we don't recompute in backward)
    // Stored per head flattened: (T, num_heads · m · m)
    Tensor last_kv_;             // (T, num_heads · m · m)

    // wkv state cache (T+1 rows; row t is s_{t-1} flattened)
    Tensor last_s_;              // (T + 1, num_heads · m · m)

    // Pre-output intermediate (r · wkv): (T, d)
    Tensor last_rwkv_;           // (T, d)

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double tanh(double x) {
        return std::tanh(x);
    }
};

#endif // RWKV6_H