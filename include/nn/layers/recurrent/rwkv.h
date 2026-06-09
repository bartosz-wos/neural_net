#ifndef RWKV_H
#define RWKV_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RWKV — Peng et al. 2023
//   "RWKV: Reinventing RNNs for the Transformer Era"
//
// RWKV is a linear-attention-style sequence model with a time-decay term.
// The full RWKV block has two parts:
//   (1) Time-mixing (a.k.a. "token shift" + WKV recurrence) — the iconic
//       operation that replaces softmax attention.
//   (2) Channel-mixing (a.k.a. "FFN" with token shift) — a 2-layer feedforward
//       with token shift.
//
// In v1 we implement ONLY the time-mixing block. Channel-mixing is essentially
// a 2-layer FFN with token-shift and doesn't add math novelty; we leave it
// for a follow-up (or a wrapper Model class). The time-mixing block is the
// interesting part — it captures the linear-attention-with-decay recurrence
// that gives RWKV its O(1)-per-step inference cost.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (single block, no LayerNorm to keep the gradient
// check tractable; this matches RWKV-4 "no-LN" ablation):
//
//   Inputs:
//     x_t  in R^{d}     for t = 0..T-1
//
//   Step 1 — Token shift (Peng Eq. 2):
//     r_in_t = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_{t-1}
//     k_in_t = μ_k ⊙ x_t + (1 - μ_k) ⊙ x_{t-1}
//     v_in_t = μ_v ⊙ x_t + (1 - μ_v) ⊙ x_{t-1}
//     with x_{-1} := 0 (no padding — first step uses zero shift).
//
//   Step 2 — Projections (each is a Dense y = x W^T + b, with W of shape (d,d)):
//     r_pre_t = W_r · r_in_t + b_r   in R^d
//     k_pre_t = W_k · k_in_t + b_k   in R^d
//     v_pre_t = W_v · v_in_t + b_v   in R^d
//
//   Step 3 — WKV recurrence (the linear-attention-with-decay core):
//     For each output channel i, the per-step "numerator" p_t[i] is:
//        p_t[i] = a[i] · p_{t-1}[i] + k_pre_t · v_pre_t[i]
//     where a[i] = exp(-exp(log_w[i]))  in (0, 1)  is the per-channel decay.
//
//     The "wkv" output at step t is the p_t vector plus a diagonal bonus term
//     that boosts the j=t contribution by a learnable factor exp(u[i]):
//        wkv_t[i] = p_t[i] + (exp(u[i]) - 1) · k_pre_t · v_pre_t[i]
//
//   Step 4 — Receptance-gated output:
//     o_t = sigmoid(r_pre_t) ⊙ wkv_t
//
//   State: p_t in R^d, the "numerator" of the linear-attention recurrence.
//   No hidden cell state like LSTM. This is the O(1)-per-step state that
//   gives RWKV its recurrent-form inference cost.
//
// ----------------------------------------------------------------------------
// Initialization convention (RWKV-4 codebase):
//   * W_r, W_k, W_v: xavier-uniform (Dense default).
//   * b_r, b_k, b_v: zero.
//   * log_w: -5.0 init → w = -exp(-5) ≈ -0.0067, a = exp(w) ≈ 0.993, slow decay.
//   * u: 0.0 init → bonus = 0, then bonus is small initially (no j=t boost).
//   * μ_r, μ_k, μ_v: 0.5 init → start as a 50/50 mix between current and previous.
//
// ----------------------------------------------------------------------------
// Shape convention: forward takes a 2D input (T, d) and returns (T, d).
// The p-state is cached as (T+1, d) (row 0 = initial state 0) for BPTT.
// BPTT traverses this cache backward in `backward`.
// ============================================================================

class RWKVTimeMix : public Layer {
public:
    // d: input/output feature dim (must satisfy d > 0)
    explicit RWKVTimeMix(size_t d);

    // Forward pass on a full sequence.
    // input: (T, d)  ->  output: (T, d)
    Tensor forward(const Tensor& input) override;
    // Backward pass — grad_output: (T, d), returns grad_input: (T, d)
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_r.weights; }
    Tensor get_gradients() const override { return W_r.grad_weights; }
    std::string name() const override { return "RWKVTimeMix"; }

    // Accessors for tests
    size_t d() const { return d_; }

    // ---- Public parameters (for gradient tests / debugging) ----
    size_t d_;
    Dense W_r;              // (d, d)
    Dense W_k;              // (d, d)
    Dense W_v;              // (d, d)
    Tensor log_w;           // (1, d)  unconstrained; a[i] = exp(-exp(log_w[i]))
    Tensor u;               // (1, d)  bonus; bonus_scalar[i] = exp(u[i]) - 1
    Tensor mu_r;            // (1, d)  learnable mix coefficient for r
    Tensor mu_k;            // (1, d)  learnable mix coefficient for k
    Tensor mu_v;            // (1, d)  learnable mix coefficient for v

    // Hidden gradient buffers
    Tensor grad_log_w_;     // (1, d)
    Tensor grad_u_;         // (1, d)
    Tensor grad_mu_r_;      // (1, d)
    Tensor grad_mu_k_;      // (1, d)
    Tensor grad_mu_v_;      // (1, d)

private:

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;         // (T, d)  — cloned to avoid input corruption
    Tensor last_x_shift_;       // (T, d)  — x_{t-1} (row 0 = zeros)
    Tensor last_r_in_;          // (T, d)  — mixed input fed to W_r
    Tensor last_k_in_;          // (T, d)  — mixed input fed to W_k
    Tensor last_v_in_;          // (T, d)  — mixed input fed to W_v
    Tensor last_r_pre_;         // (T, d)  — W_r @ r_in + b_r
    Tensor last_k_pre_;         // (T, d)  — W_k @ k_in + b_k
    Tensor last_v_pre_;         // (T, d)  — W_v @ v_in + b_v
    Tensor last_r_sig_;         // (T, d)  — sigmoid(r_pre)
    Tensor last_p_;             // (T+1, d)  — p_t numerator (row 0 = 0)
    Tensor last_a_;             // (1, d)  — exp(-exp(log_w))
    Tensor last_bonus_;         // (1, d)  — exp(u) - 1
    Tensor last_wkv_;           // (T, d)  — wkv_t = p_t + bonus * (k ⊙ v) (per channel)
    Tensor last_kv_;            // (T, d)  — k_pre_t ⊙ v_pre_t (per-channel element-wise)

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
};

#endif // RWKV_H
