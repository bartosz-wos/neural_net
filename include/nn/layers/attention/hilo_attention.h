#ifndef HILO_ATTENTION_H
#define HILO_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// HiLo Attention — Pan et al., ICLR 2025
//   "HiLo: A High-Low Frequency-Aware Attention for Long Sequence Modeling"
//   https://arxiv.org/abs/2405.13219
// ============================================================================
//
// Standard multi-head attention applies softmax over the same key set in every
// head, so high-frequency detail (local edges, sharp tokens) and low-frequency
// context (long-range trend) compete in the same head. HiLo splits heads into
// two groups:
//
//   * Hi (High-frequency) heads: the first `num_local_heads` heads apply
//     CAUSAL WINDOW softmax over the last `window_size` keys — captures local
//     detail cheaply.
//   * Lo (Low-frequency) heads: the remaining `num_global_heads` heads apply
//     softmax over the AVG-POOLED key/value sequence (non-overlapping windows
//     of size `window_size`, yielding ⌈N/W⌉ pooled tokens) — captures global
//     context at O(N²/W) cost.
//
// A learnable per-head scalar λ[h] ∈ (0,1) interpolates the two streams:
//
//   out_h_j = (1 − λ[h]) · out_local_h_j + λ[h] · out_global_h_j
//
// with `λ[h] = σ(logit_lambda[h])` (default init: logit_lambda=0 → λ=0.5).
//
// Math (single layer, causal convention: query j attends only to keys i ≤ j):
//
//   per head h, d_h = d_model / num_heads:
//     q_j     = (W_q · x)[h·N + j]                   ∈ R^{d_h}
//     k_i     = (W_k · x)[h·N + i]                   ∈ R^{d_h}
//     v_i     = (W_v · x)[h·N + i]                   ∈ R^{d_h}
//
//   if h < num_local_heads:                          // HIGH-FREQUENCY
//     mask[i,j] = 1 iff (j − window_size < i ≤ j)
//     z[i,j]   = (q_j · k_i) / sqrt(d_h),   −∞ if mask=0
//     attn[i,j] = softmax_i(z[:,j])                  // over the WINDOW
//     out_local_h_j = attn @ V
//     factor = 1 − λ[h]
//
//   else:                                            // LOW-FREQUENCY
//     k_pool[p] = (1/W_p) · Σ_{i ∈ window_p} k_i    // ⌈N/W⌉ pooled keys
//     v_pool[p] = (1/W_p) · Σ_{i ∈ window_p} v_i    // ⌈N/W⌉ pooled values
//     mask_pool[p,j] = 1 iff (p·W ≤ j)               // pool's first token at/before j
//     z_pool[p,j] = (q_j · k_pool[p]) / sqrt(d_h),  −∞ if mask=0
//     attn_pool[p,j] = softmax_p(z_pool[:,j])        // over the POOLED keys
//     out_global_h_j = attn_pool @ V_pool
//     factor = λ[h]
//
//   out_h_j = factor · out_X_h_j + (1 − factor) · out_Y_h_j
//            where X, Y = (local, global) respectively.
//
//   Output: out = concat_h(out_h) · W_o + b_o
//
// ----------------------------------------------------------------------------
// Backward derivation (hand-derived; see plan for the full formulae).
//
// Each head produces a softmax chain (local OR global), with the global chain
// ALSO feeding back through the average-pool into dK_local / dV_local. The
// λ-blend is applied as a per-head multiplicative mask on dO before the per-
// stream chains. logit_lambda receives `σ'(logit_lambda) · (out_g − out_l) ·
// dO` summed over (j, d).
//
// Critical bug to avoid (caught during TDD): dK_local and dV_local receive
// contributions from BOTH streams — direct (local-chain) AND back-prop through
// the average pool (global-chain). Forgetting to add both gives wrong
// magnitudes (off by the global contribution, ~50% error in some cells).
//
// ----------------------------------------------------------------------------
// Conventions (match stick_breaking.h):
//   * Input / output: (N, d_model). N tokens, no explicit batch dim.
//   * W_q, W_k, W_v, W_o are `Dense` (weights shaped (out, in)); heads are
//     contiguous column slices of the flat (N, d_model) projections.
//   * Parameter gradients for the projections are accumulated into raw
//     grad_W_* tensors (NOT via Dense::backward) because the per-head chain
//     needs manual control.
//   * Block: pre-LN -> HiLo attn -> residual -> pre-LN -> GELU FFN -> residual
// ============================================================================

class HiLoAttention : public Layer {
public:
    // d_model:        input/output feature dim; must be divisible by num_heads
    // num_heads:      total number of heads (default 8; first half = local,
    //                  second half = global). Even split if num_heads is even.
    // window_size:    pool / window size for both streams (default 4).
    HiLoAttention(size_t d_model, size_t num_heads = 8, size_t window_size = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "HiLoAttention"; }

    // Accessors
    size_t d_model()      const { return d_model_; }
    size_t num_heads()    const { return num_heads_; }
    size_t num_local()    const { return num_local_; }
    size_t num_global()   const { return num_global_; }
    size_t head_dim()     const { return head_dim_; }
    size_t window_size()  const { return window_size_; }
    double inv_temp()     const { return inv_temp_; }
    size_t num_pool()     const { return num_pool_; }

    // Cached tensors from the last forward, useful for tests / introspection
    const Tensor& last_input()     const { return last_input_; }
    const Tensor& last_Q()         const { return last_Q_; }
    const Tensor& last_K()         const { return last_K_; }
    const Tensor& last_V()         const { return last_V_; }
    // K_pool / V_pool per head, stacked: shape (H * num_pool, d_h)
    const Tensor& last_K_pool()    const { return last_K_pool_; }
    const Tensor& last_V_pool()    const { return last_V_pool_; }
    // Local attn per head: shape (H_local * N, N)  — only local heads have rows
    const Tensor& last_A_local()   const { return last_A_local_; }
    // Global attn per head: shape (H_global * N, num_pool)
    const Tensor& last_A_global()  const { return last_A_global_; }
    // Per-head λ values: shape (num_heads,)
    const Tensor& lambda_values()  const { return lambda_; }

    // Public so tests can perturb for FD checks
    Dense W_q, W_k, W_v, W_o;          // each (d_model, d_model)
    Tensor logit_lambda_;              // (num_heads,) — pre-sigmoid
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_logit_lambda_;
    Tensor lambda_;        // (num_heads,) — sigmoid(logit_lambda_)

private:
    size_t d_model_;
    size_t num_heads_;
    size_t num_local_;     // = num_heads / 2 (rounded down if odd)
    size_t num_global_;    // = num_heads - num_local_
    size_t head_dim_;
    size_t window_size_;
    double inv_temp_;
    size_t N_last_;
    size_t num_pool_;

    // Forward caches
    Tensor last_input_;
    Tensor last_Q_;        // (N, d_model)
    Tensor last_K_;        // (N, d_model)
    Tensor last_V_;        // (N, d_model)
    Tensor last_K_pool_;   // (H * num_pool, d_h) — pooled K per head, flat
    Tensor last_V_pool_;   // (H * num_pool, d_h) — pooled V per head, flat
    Tensor last_A_local_;  // (H_local * N, N)        — only first H_local rows
    Tensor last_A_global_; // (H_global * N, num_pool)
    Tensor last_O_pre_;    // (N, d_model) — pre-W_o concat of per-head outputs

    // Per-pool window length (handles the N % W != 0 boundary case)
    std::vector<size_t> pool_window_sizes_;

    // For input-grad: scratch
    Tensor last_grad_O_pre_;  // (N, d_model) — grad w.r.t. concat output

    // Helpers
    void recompute_lambda();
};


// ----------------------------------------------------------------------------
// HiLoBlock — pre-LN -> HiLo attn -> residual -> pre-LN -> FFN -> residual
// ----------------------------------------------------------------------------
class HiLoBlock : public Layer {
public:
    HiLoBlock(size_t d_model, size_t num_heads = 8,
              size_t window_size = 4, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.grad_W_q; }
    std::string name() const override { return "HiLoBlock"; }

    HiLoAttention attn;
    LayerNorm ln1, ln2;
    Dense ffn_fc1_;   // (ffn_dim, d_model)
    Dense ffn_fc2_;   // (d_model, ffn_dim)

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;
    Tensor last_z1_;        // ln1(x)
    Tensor last_attn_out_;  // attn(z1)
    Tensor last_res1_;      // z1 + attn_out
    Tensor last_z2_;        // ln2(res1)
    Tensor last_h_pre_;     // ffn_fc1(z2)
    Tensor last_h_act_;     // GELU(h_pre)
};

#endif // HILO_ATTENTION_H
