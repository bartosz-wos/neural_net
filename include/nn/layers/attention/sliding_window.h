#ifndef SLIDING_WINDOW_H
#define SLIDING_WINDOW_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Sliding Window Attention — Mistral 7B style (Jiang et al. 2023,
//   https://arxiv.org/abs/2310.06825) + Longformer-style global tokens
//   (Beltagy et al. 2020, https://arxiv.org/abs/2004.05150)
//
// Standard softmax attention has O(N²) compute and memory. For long sequences
// (e.g. 32k context), this is prohibitive. Sliding window attention (SWA)
// restricts each query to attend only to the last W tokens (the "window"),
// giving O(N · W · d) compute and a fixed-size KV cache during inference.
//
// The math is identical to softmax attention — only the additive mask differs:
//
//   causal=true:   M[i, j] = -inf  if j < i - W + 1
//                   M[i, j] = 0     otherwise
//   causal=false:  M[i, j] = -inf  if |i - j| > W/2
//                   M[i, j] = 0     otherwise
//
// Plus optional global tokens (Longformer convention): the first
// `num_global` rows/cols of the attention map have NO window mask applied,
// so global tokens attend to and are attended by the entire sequence.
//
// Multi-head with K/V sharing via GQA convention (num_query_heads,
// num_kv_heads), mirroring the GQA layer in this repo. This naturally
// composes with the recent Llama-3 / Mistral architecture work.
//
//   * Input:  (n, d_model)    — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be evenly divisible by num_query_heads (= head_dim)
//   * num_query_heads must be evenly divisible by num_kv_heads
//   * window_size > 0
//   * num_global in [0, n)
//   * No bias on Q/K/V/O projections (matches Mistral/Llama convention)
//   * Numerical: mask uses -1e9 (not -inf) for safe softmax
//
// Conventions match the existing attention layers (cosformer, performer,
// diff_transformer, gqa):
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual)
//   * Block forward shape == input shape (n, d_model)
//   * Model: stack of blocks + classifier
//
// SlidingWindowAttention — the attention layer itself
// SlidingWindowBlock      — pre-LN transformer block wrapper
// SlidingWindowModel      — stack of blocks + classifier
// ============================================================================

class SlidingWindowAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // num_query_heads: number of Q heads (and total heads in the output)
    // num_kv_heads:   number of distinct K and V heads (must divide num_query_heads)
    // window_size:    number of tokens each query attends to
    // num_global:     number of "global" tokens with full attention (default 0)
    // causal:         if true, mask is causal (j in [i-W+1, i]); else symmetric
    SlidingWindowAttention(size_t d_model,
                           size_t num_query_heads,
                           size_t num_kv_heads,
                           size_t window_size,
                           size_t num_global = 0,
                           bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "SlidingWindowAttention"; }

    // Test accessors
    size_t d_model()      const { return d_model_; }
    size_t num_heads()    const { return num_query_heads_; }
    size_t num_kv_heads() const { return num_kv_heads_; }
    size_t head_dim()     const { return head_dim_; }
    size_t window_size()  const { return window_size_; }
    size_t num_global()   const { return num_global_; }
    bool   causal()       const { return causal_; }
    double scale()        const { return scale_; }

    // Per-head attention cache for tests. h in [0, num_query_heads).
    // Returns a const reference to (n, n) — the softmax output for head h.
    const Tensor& last_attn_head(size_t h) const { return last_attn_by_head_[h]; }

    // Test helper: returns the cached input gradient from the last backward.
    // Shape (n, d_model). Useful for FD comparison.
    Tensor grad_input() const { return last_d_input_; }

    // Test helper: disable / enable the window mask (for mutation testing).
    void set_use_window_mask(bool use) { use_window_mask_ = use; }
    bool use_window_mask() const { return use_window_mask_; }

    // Public for test access (matches the GQA / cosformer convention).
    Tensor W_q, W_k, W_v, W_o;          // (d_model, d_model)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

private:
    size_t d_model_;
    size_t num_query_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t group_size_;        // num_query_heads / num_kv_heads
    size_t window_size_;
    size_t num_global_;        // first num_global_ rows/cols are "global"
    bool   causal_;
    double scale_;             // 1 / sqrt(head_dim)
    bool   use_window_mask_;   // false → no mask applied (mutation test hook)

    // BPTT cache
    Tensor last_input_;        // (n, d_model)
    Tensor last_q_;            // (n, d_model) — post-projection
    Tensor last_k_;            // (n, d_model)
    Tensor last_v_;            // (n, d_model)
    Tensor last_attn_;         // (num_query_heads_ * n, n) — per-head softmax output
    std::vector<Tensor> last_attn_by_head_;   // num_query_heads_ Tensors of (n, n)
    Tensor last_head_out_;     // (n, d_model) — concat heads, pre-W_o
    Tensor last_d_input_;      // (n, d_model) — dL/d(input) for FD comparison
};

// ----------------------------------------------------------------------------
// SlidingWindowBlock: pre-LN → SWA → residual → pre-LN → FFN → residual
// ----------------------------------------------------------------------------

class SlidingWindowBlock : public Layer {
public:
    // ffn_dim: FFN hidden dim. If 0, no FFN sub-layer (pure attention block).
    SlidingWindowBlock(size_t d_model,
                       size_t num_query_heads,
                       size_t num_kv_heads,
                       size_t window_size,
                       size_t num_global = 0,
                       bool causal = true,
                       size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn_.W_q; }
    Tensor get_gradients() const override { return attn_.grad_W_q; }
    std::string name() const override { return "SlidingWindowBlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }

    // Returns cached input gradient from last backward (for FD tests).
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_;
    size_t ffn_dim_;

    SlidingWindowAttention attn_;
    LayerNorm ln1_;
    LayerNorm ln2_;

    // FFN sub-block (only used if ffn_dim > 0)
    Tensor W1_, b1_, W2_, b2_;
    Tensor grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    // BPTT cache
    Tensor last_input_;
    Tensor last_ln1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_ln2_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_out_;
    Tensor last_d_input_;
};

// ----------------------------------------------------------------------------
// SlidingWindowModel: in_proj → N blocks → classifier
// ----------------------------------------------------------------------------

class SlidingWindowModel : public Layer {
public:
    SlidingWindowModel(size_t input_dim,
                       size_t d_model,
                       size_t output_dim,
                       size_t num_blocks,
                       size_t num_query_heads,
                       size_t num_kv_heads,
                       size_t window_size,
                       size_t num_global = 0,
                       bool causal = true,
                       size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "SlidingWindowModel"; }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;

    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;

    std::vector<SlidingWindowBlock> blocks_;

    Tensor last_input_;
    Tensor last_proj_;
    Tensor last_block_out_;
};

#endif
