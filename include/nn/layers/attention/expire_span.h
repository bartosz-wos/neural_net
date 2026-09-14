#ifndef EXPIRE_SPAN_H
#define EXPIRE_SPAN_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Expire-Span Attention — Sukhbaatar, Ju, Poff, Roller, Szlam, Weston, Lewis
//   2021 "Not All Memories are Created Equal: A Learnable, Per-Layer Memory
//   Budget" (https://arxiv.org/abs/2105.11850 / EMNLP 2021)
//
// Idea: every layer attaches a small Dense head
//   z_i = W_span · x_i + b_span
//   s_i = clamp(sigmoid(z_i), s_min, 1.0) ∈ [s_min, 1.0]
// At forward, for query i and key j, the effective age is (i - j). Key j
// survives in the softmax iff (i - j) ≤ s_j * S_max_eff (where
// S_max_eff = S_max if S_max > 0 else n). We use a SOFT (differentiable) mask
// for the gradient signal:
//   M[i, j] = -soft_threshold * sigmoid((i - j - s_j * S_max_eff) / temperature)
// where soft_threshold = 1e9 (effectively saturates to 0 on the surviving side
// and -1e9 on the dropped side) and temperature=0.5 keeps the sigmoid
// transition narrow. This is a differentiable relaxation of the paper's hard
// keep/drop; it provides well-defined gradients (the paper uses STE for the
// hard mask, which we don't implement here).
//
// Conventions:
//   * Input:  (n, d_model)    — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be evenly divisible by num_query_heads
//   * num_query_heads must be evenly divisible by num_kv_heads
//   * Causal mask: j > i is always dropped (-1e9 in the limit)
//   * No bias on Q/K/V/O projections (Llama/Mistral style)
//
// ExpireSpanAttention — the attention layer itself
// ExpireSpanBlock      — pre-LN → expire attn → residual → pre-LN → GELU FFN → residual
// ExpireSpanModel      — input projection → stack of blocks → classifier
// ============================================================================

class ExpireSpanAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // num_query_heads: number of Q heads
    // num_kv_heads:   number of distinct K and V heads (must divide num_query_heads)
    // S_max:          max effective age in tokens (default 0 → use n at forward time)
    // s_min:          floor for the sigmoid span (paper §3.2; default 1e-6)
    ExpireSpanAttention(size_t d_model,
                        size_t num_query_heads,
                        size_t num_kv_heads,
                        size_t S_max = 0,
                        double s_min = 1e-6);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "ExpireSpanAttention"; }

    // Test accessors
    size_t d_model()       const { return d_model_; }
    size_t num_heads()     const { return num_query_heads_; }
    size_t num_kv_heads()  const { return num_kv_heads_; }
    size_t head_dim()      const { return head_dim_; }
    size_t S_max()         const { return S_max_; }
    double s_min()         const { return s_min_; }
    double scale()         const { return scale_; }

    // Cache accessors (for FD and signature tests)
    const Tensor& last_z()         const { return last_z_; }        // (n, 1)
    const Tensor& last_span()      const { return last_span_; }     // (n, 1) — clamped
    const Tensor& last_mask()      const { return last_mask_; }     // (n, n), 0 or -1e9
    const Tensor& last_attn_head(size_t h) const { return last_attn_by_head_[h]; }
    Tensor grad_input()            const { return last_d_input_; }

    // Public for test access (matches GQA / cosformer / sliding_window convention).
    Tensor W_q, W_k, W_v, W_o;          // (d_model, d_model)
    Tensor W_span, b_span;              // (1, d_model), (1, 1)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_W_span, grad_b_span;

private:
    size_t d_model_, num_query_heads_, num_kv_heads_, head_dim_, group_size_;
    size_t S_max_;
    double s_min_, scale_;
    double soft_threshold_;   // 1e9
    double temperature_;      // 0.5

    // BPTT cache
    Tensor last_input_;        // (n, d_model)
    Tensor last_q_, last_k_, last_v_;   // (n, d_model)
    Tensor last_attn_;         // (Hq * n, n) — per-head softmax output
    std::vector<Tensor> last_attn_by_head_;
    Tensor last_head_out_;     // (n, d_model)
    Tensor last_z_, last_span_;          // (n, 1)
    Tensor last_mask_;         // (n, n) — additive mask applied (with -1e9 saturations)
    Tensor last_soft_mask_;    // (n, n) — raw soft-mask values for BPTT
    Tensor last_d_input_;      // (n, d_model)
};

// ============================================================================
// ExpireSpanBlock — pre-LN → ExpireSpanAttention → residual →
//                    pre-LN → GELU FFN → residual
// ============================================================================

class ExpireSpanBlock : public Layer {
public:
    // ffn_dim: FFN hidden dim. If 0, no FFN sub-layer (pure attention block).
    ExpireSpanBlock(size_t d_model,
                    size_t num_query_heads,
                    size_t num_kv_heads,
                    size_t S_max = 0,
                    double s_min = 1e-6,
                    size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn_.W_q; }
    Tensor get_gradients() const override { return attn_.grad_W_q; }
    std::string name() const override { return "ExpireSpanBlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }

    // Returns cached input gradient from last backward (for FD tests).
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_;
    size_t ffn_dim_;

    ExpireSpanAttention attn_;
    LayerNorm ln1_;                 // pre-attn
    LayerNorm ln2_;                 // pre-FFN
    Dense ffn_fc1_;                 // (ffn_dim, d_model)
    Dense ffn_fc2_;                 // (d_model, ffn_dim)

    // BPTT cache
    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_pre_;
    Tensor last_ffn_hidden_;
    Tensor last_ffn_out_;
    Tensor last_d_input_;
};

// ============================================================================
// ExpireSpanModel — input projection → stack of blocks → classifier
// ============================================================================

class ExpireSpanModel : public Layer {
public:
    ExpireSpanModel(size_t d_input,
                    size_t d_model,
                    size_t d_output,
                    size_t num_blocks,
                    size_t num_query_heads,
                    size_t num_kv_heads,
                    size_t S_max = 0,
                    double s_min = 1e-6,
                    size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_.get_weights(); }
    Tensor get_gradients() const override { return W_in_.get_gradients(); }
    std::string name() const override { return "ExpireSpanModel"; }

private:
    size_t d_input_;
    size_t d_model_;
    size_t d_output_;
    Dense W_in_, W_out_;
    std::vector<ExpireSpanBlock> blocks_;

    Tensor last_input_;
    Tensor last_proj_;
};

#endif
