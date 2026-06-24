#ifndef MLA_H
#define MLA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Multi-Head Latent Attention (MLA) — DeepSeek-AI 2024
//   "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
//    Language Model" (https://arxiv.org/abs/2405.04434)
//
// The core idea: instead of computing and storing the full K and V tensors
// of shape (n, d_model) per attention layer, compress them into a SHARED
// low-rank "latent" c_KV of shape (n, d_c), with d_c << d_model. The
// compressed latent is what you actually cache during inference, giving
// O(n * d_c) memory and bandwidth instead of O(n * d_model).
//
// Math (multi-head; per head h with head_dim = d_model / num_heads):
//
//   c_Q  = X @ W_dq              (n, d_c)    # shared down-projection of Q
//   Q_h  = c_Q @ W_uq[h]         (n, head_dim)
//   c_KV = X @ W_dkv             (n, d_c)    # shared down-projection of K, V
//   K_h  = c_KV @ W_uk[h]        (n, head_dim)
//   V_h  = c_KV @ W_uv[h]        (n, head_dim)
//   attn_h   = softmax(Q_h @ K_h^T / sqrt(head_dim))   (n, n)
//   head_out = attn_h @ V_h                              (n, head_dim)
//   out      = concat_h head_out @ W_o                  (n, d_model)
//
// Note that W_dq, W_dkv are SHARED across heads (one down-projection each),
// while W_uq, W_uk, W_uv are per-head. This is the central trick — the
// shared c_KV is what enables the KV cache compression. The backward
// correctly accumulates gradients to W_dkv from BOTH the K and V branches
// (and to W_dq from the Q branch).
//
// Conventions (matches GQA):
//   * Input:  (n, d_model)    — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be evenly divisible by num_heads
//   * head_dim = d_model / num_heads
//   * No bias on any of the 6 projections
//   * No causal mask in v1 — calls can mask by zeroing in a wrapper
//   * Per-head projection matrices are stored as flat (d_model, d_c) or
//     (d_model, d_model) tensors with the head blocks stacked along the
//     first dim (the OUTPUT dim), matching the GQA layout convention
//     and avoiding the (out, in) Dense convention confusion.
//
// Param count breakdown:
//   W_dq:  d * d_c            (shared, Q down)
//   W_uq:  d * d_c            (stacked per-head, Q up)
//   W_dkv: d * d_c            (shared, KV down)
//   W_uk:  d * d_c            (stacked per-head, K up)
//   W_uv:  d * d_c            (stacked per-head, V up)
//   W_o:   d * d              (output projection)
//   Total: 5*d*d_c + d^2
//
// For comparison, plain MHA has 4*d^2 parameters. MLA beats MHA when
//   5*d*d_c + d^2 < 4*d^2  =>  d_c < 3d/5
// which is easily satisfied in DeepSeek-V2 (d_c = 4d/9 ≈ 0.44 d).
// ============================================================================

class MLAAttention : public Layer {
public:
    MLAAttention(size_t d_model, size_t num_heads, size_t d_c);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_dq; }
    Tensor get_gradients() const override { return grad_W_dq; }
    std::string name() const override { return "MLAAttention"; }

    // Accessors for tests
    size_t d_model()  const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t d_c()      const { return d_c_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;   // d_model / num_heads
    size_t d_c_;


    double scale_;   // 1 / sqrt(head_dim)

    // BPTT cache
    Tensor last_input_;        // (n, d_model)         cloned
    Tensor last_c_q_;          // (n, d_c)             shared Q down
    Tensor last_c_kv_;         // (n, d_c)             shared KV down
    Tensor last_q_;            // (n, d_model)         per-head Q stacked
    Tensor last_k_;            // (n, d_model)         per-head K stacked
    Tensor last_v_;            // (n, d_model)         per-head V stacked
    Tensor last_attn_;         // (num_heads, n, n)    softmax probs per Q head
    Tensor last_head_out_;     // (n, d_model)         concatenated head outputs (pre-W_o)

public:
    // Shared down-projections (d_model, d_c) — one per Q and one shared
    // for both K and V.
    Tensor W_dq, W_dkv;
    Tensor grad_W_dq, grad_W_dkv;

    // Per-head up-projections, stacked as (d_model, d_c) (one block per
    // head along the output dim). Layout:
    //   W_uq[h, :, :]   at [h*head_dim : (h+1)*head_dim, 0 : d_c]
    //   W_uk[h, :, :]   at [h*head_dim : (h+1)*head_dim, 0 : d_c]
    //   W_uv[h, :, :]   at [h*head_dim : (h+1)*head_dim, 0 : d_c]
    Tensor W_uq, W_uk, W_uv;
    Tensor grad_W_uq, grad_W_uk, grad_W_uv;

    // Output projection (d_model, d_model)
    Tensor W_o;
    Tensor grad_W_o;
};

// ============================================================================
// MLABlock — pre-LN → MLAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

class MLABlock : public Layer {
public:
    MLABlock(size_t d_model, size_t num_heads, size_t d_c, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MLABlock"; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;             // (d_model)  pre-attn
    MLAAttention attn_;
    LayerNorm ln2_;             // (d_model)  pre-FFN
    Dense ffn_fc1_;             // (ffn_dim, d_model)
    Dense ffn_fc2_;             // (d_model, ffn_dim)

    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_hidden_;
    Tensor last_ffn_out_;
};

// ============================================================================
// MLAModel — stack of MLABlocks + classifier head
// ============================================================================

class MLAModel : public Layer {
public:
    MLAModel(size_t d_model, size_t num_heads, size_t d_c,
             size_t out_features, size_t num_blocks = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MLAModel"; }

private:
    size_t d_model_;
    size_t out_features_;
    std::vector<MLABlock> blocks_;
    Dense classifier_;

    Tensor last_input_;
};

#endif
