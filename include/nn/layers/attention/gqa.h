#ifndef GQA_H
#define GQA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Grouped Query Attention (GQA) — Ainslie et al. 2023
//   "GQA: Training Generalized Multi-Query Transformer Models from
//    Multi-Head Checkpoints"
//
// GQA is an interpolation between standard Multi-Head Attention (MHA) and
// Multi-Query Attention (MQA). It is the attention variant used by
// Llama 2, Llama 3, and Mistral.
//
// The key idea: split the Q heads into `num_kv_groups` (= num_kv_heads) groups,
// and share one K projection and one V projection across each group. The Q
// projections remain per-head. Concretely:
//
//   num_query_heads = G * num_kv_heads
//   group_size      = num_query_heads / num_kv_heads
//   head_dim        = d_model / num_query_heads
//
// For each query head h in [0, num_query_heads):
//     kv_head(h) = h / group_size
//     Q_h = X @ W_q[h]   in R^{n x head_dim}     (per-Q-head projection)
//     K_h = X @ W_k[kv_head(h)]                   (shared across group_size Q-heads)
//     V_h = X @ W_v[kv_head(h)]
//     attn_h = softmax(Q_h @ K_h^T / sqrt(head_dim))  in R^{n x n}
//     head_out_h = attn_h @ V_h                     in R^{n x head_dim}
//
// Final output: concat all head_outs along head_dim, then project through W_o.
//
// Conventions (matches Linformer for ergonomics with gradient checks):
//   * Input:  (n, d_model)    — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be evenly divisible by num_query_heads
//   * num_query_heads must be evenly divisible by num_kv_heads
//   * No bias on Q/K/V/O projections (matches paper convention; Llama style)
//   * No causal mask in v1 — the math is the cleanest without it; callers
//     can mask by zeroing in a wrapper layer if needed.
//   * gqa_head_size(h) = group_size   — how many Q heads share head h's K/V
//
// Special cases:
//   * num_kv_heads == num_query_heads  → standard MHA (every Q head has its
//                                          own K and V).
//   * num_kv_heads == 1                → MQA (single K/V shared by all Q heads;
//                                          this is what PaLM uses).
//   * The backward correctly routes gradients: dK[h] = sum of dK_h from all
//     query heads whose kv_head(h_q) == h (i.e., shared in forward → sum in
//     backward). This is the key BPTT difference from MHA.
//
// Parameter count reduction vs MHA:
//   MHA: 4 * d_model^2
//   GQA: 4 * d_model^2 - 2 * d_model^2 * (1 - 1/group_size)
//   MQA: 2 * d_model^2 + 2 * d_model * head_dim
// So GQA cuts K/V projection parameters by a factor of group_size.
//
// GQABlock: pre-LN → GQAAttention → residual → pre-LN → GELU FFN → residual
//   (Same layout as the Linformer block; lets you drop-in a GQA-equipped
//   Transformer without changing surrounding code.)
//
// GQAModel: stack of `num_blocks` GQABlocks + classifier head.
//   Input: (n, d_model). Output: (n, out_features) per-token logits.
// ============================================================================

class GQAAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // num_query_heads: number of Q heads (and total heads in the output)
    // num_kv_heads:   number of distinct K and V heads (must divide num_query_heads)
    GQAAttention(size_t d_model, size_t num_query_heads, size_t num_kv_heads);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "GQAAttention"; }

    // Accessors for tests
    size_t d_model()         const { return d_model_; }
    size_t num_query_heads() const { return num_query_heads_; }
    size_t num_kv_heads()    const { return num_kv_heads_; }
    size_t head_dim()        const { return head_dim_; }
    size_t group_size()      const { return group_size_; }

private:
    size_t d_model_;
    size_t num_query_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;     // d_model / num_query_heads
    size_t group_size_;   // num_query_heads / num_kv_heads

    // Per-Q-head and per-K/V-head projection matrices, all (head_dim, d_model).
    // Storing them as raw Tensors (not Dense) avoids the layout confusion
    // that the legacy MHA had with y = xW^T + b.
    //
    // Layout in the flat (d_model, d_model) tensor: stacked head blocks.
    //   W_q[h, :, :]   lives at [h*head_dim : (h+1)*head_dim, 0 : d_model]
    //   W_k[k, :, :]   lives at [k*head_dim : (k+1)*head_dim, 0 : d_model]
    //   W_v[k, :, :]   lives at [k*head_dim : (k+1)*head_dim, 0 : d_model]
    //   W_o[h, :, :]   lives at [h*head_dim : (h+1)*head_dim, 0 : d_model]
    //
    // W_q and W_o have num_query_heads head blocks; W_k and W_v have
    // num_kv_heads head blocks.  Note: W_o is the "output" projection —
    // it projects the concatenated head outputs (n, d_model) back to (n, d_model).
public:
    Tensor W_q, W_k, W_v, W_o;       // (d_model, d_model)  (stacked head blocks)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    double scale_;   // 1 / sqrt(head_dim)

    // BPTT cache
    Tensor last_input_;          // (n, d_model)         cloned
    Tensor last_q_;              // (n, d_model)         stacked Q heads
    Tensor last_k_;              // (n, d_model)         stacked K heads (num_kv_heads blocks)
    Tensor last_v_;              // (n, d_model)         stacked V heads (num_kv_heads blocks)
    Tensor last_attn_;           // (num_query_heads, n, n)  softmax probabilities per Q head
    Tensor last_head_out_;       // (n, d_model)         concatenated head outputs (pre-W_o)
};

// ============================================================================
// GQABlock — pre-LN → GQAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

class GQABlock : public Layer {
public:
    GQABlock(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
             size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "GQABlock"; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;                 // (d_model)  pre-attn
    GQAAttention attn_;
    LayerNorm ln2_;                 // (d_model)  pre-FFN
    Dense ffn_fc1_;                 // (ffn_dim, d_model)
    Dense ffn_fc2_;                 // (d_model, ffn_dim)

    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_hidden_;
    Tensor last_ffn_out_;
};

// ============================================================================
// GQAModel — stack of GQABlocks + classifier head
// ============================================================================

class GQAModel : public Layer {
public:
    GQAModel(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
             size_t out_features, size_t num_blocks = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "GQAModel"; }

private:
    size_t d_model_;
    size_t out_features_;
    std::vector<GQABlock> blocks_;
    Dense classifier_;

    Tensor last_input_;
};

#endif
