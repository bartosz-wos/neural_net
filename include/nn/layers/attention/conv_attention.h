#ifndef CONV_ATTENTION_H
#define CONV_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// MultiHead-Conv Attention — Yang et al. 2023, "Convolutional Self-Attention"
//
// Reference:
//   "Convolutional Self-Attention Networks" (Yang et al., 2023)
//   https://arxiv.org/abs/2308.01462
//
// Innovation: replace the dense Q/K/V projections of standard multi-head
// self-attention with 1D convolutions over the sequence axis. This adds a
// strong local-inductive bias to attention (similar in spirit to the
// convolutional bias of CNNs) and lets Q/K/V "see" neighbouring positions
// before computing the attention scores.
//
// Per head h, the math becomes:
//   Q_h = Conv1D_q(X)  ∈ R^{n × head_dim}   (kernel_size = k)
//   K_h = Conv1D_k(X)  ∈ R^{n × head_dim}
//   V_h = Conv1D_v(X)  ∈ R^{n × head_dim}
//   scores_h[t, s]  = Q_h[t, :] · K_h[s, :] / sqrt(head_dim)
//   attn_h[t, s]    = softmax_s(scores_h[t, s])
//   out_h[t, :]     = sum_s attn_h[t, s] * V_h[s, :]
// Final: out = Concat(out_0, out_1, ..., out_{H-1}) @ W_o^T + b_o  ∈ R^{n × d_model}
//
// Implementation notes (matching repo conventions):
//   * Conv1D already in the repo, with in_channels × kernel_size as the
//     input feature dim and weights stored as (out_channels, in_channels * ksz).
//   * The "Dense convention" for W_o: W_o is (d_model, d_model), forward
//     y = x @ W_o^T + b_o. Backward updates W_o using sum_t grad_y[t, k] * x[t, j].
//   * The 1D conv for Q/K/V runs over the full d_model channels at once
//     (out_channels = d_model). This means each output position t already
//     encodes local context of t-(k-1)/2..t+(k-1)/2, which is then attended
//     to via the standard dot-product mechanism.
//   * Default kernel_size = 3 with pad=1, stride=1 ("same" padding) so the
//     sequence length is preserved end-to-end.
//   * Multi-head: d_model = num_heads * head_dim. We split Q/K/V along the
//     channel dim into H slices, run per-head attention, then concat.
//
// We expose:
//   * ConvAttention        — single (multi-head) self-attention block with
//                            conv-style Q/K/V projections.
//   * ConvAttentionBlock   — pre-LN → ConvAttention → residual →
//                            pre-LN → GELU FFN → residual.
//   * ConvAttentionModel   — stack of ConvAttentionBlocks + per-token
//                            classifier head.
// ============================================================================

class ConvAttention : public Layer {
public:
    using Layer::forward;  // bring base forward into scope (1-arg override)

    // d_model:    input/output feature dim (must be divisible by num_heads)
    // seq_len:    sequence length n (fixed at construction)
    // num_heads:  number of attention heads (default 1 — single-head)
    // kernel_size:1D conv kernel for Q/K/V (default 3, "same" pad)
    ConvAttention(size_t d_model, size_t seq_len, size_t num_heads = 1,
                  size_t kernel_size = 3);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_o.weights; }
    Tensor get_gradients() const override { return W_o.grad_weights; }
    std::string name() const override { return "ConvAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t seq_len()   const { return seq_len_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t kernel_size() const { return kernel_size_; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t num_heads_;
    size_t head_dim_;
    size_t kernel_size_;
    size_t pad_;

    // Output projection: y = x @ W_o^T + b_o
    Dense W_o;

    // 1D conv weights for Q, K, V projections.
    // Stored as raw Tensors (out_channels, in_channels * kernel_size) to match
    // the Conv1D layer convention, with separate bias (out_channels, 1).
    Tensor Wq_w_, Wk_w_, Wv_w_;       // (d_model, d_model * kernel_size)
    Tensor Wq_b_, Wk_b_, Wv_b_;       // (d_model, 1)
    Tensor grad_Wq_w_, grad_Wk_w_, grad_Wv_w_;
    Tensor grad_Wq_b_, grad_Wk_b_, grad_Wv_b_;

    // im2col caches: per projection, the unfolded input column matrix
    // shape (in_channels * kernel_size, n * seq_out).  seq_out == n for
    // "same" padding, so we can treat this as (d_model*k, n*n).
    Tensor col_q_, col_k_, col_v_;

    double scale_;  // 1.0 / sqrt(head_dim_)

    // Cached for backward
    Tensor last_input_;       // (n, d_model)
    Tensor last_q_;           // (n, d_model)  after conv
    Tensor last_k_;           // (n, d_model)  after conv
    Tensor last_v_;           // (n, d_model)  after conv
    Tensor last_attn_;        // (n, n) softmax per head — flattened across heads
                              //   layout: head h row t col s  =>  index h*n*n + t*n + s
    Tensor last_output_pre_o_;// (n, d_model) pre-W_o output
};
// ============================================================================
// ConvAttentionBlock — pre-LN → ConvAttention → residual →
//                      pre-LN → GELU FFN → residual
// ============================================================================

class ConvAttentionBlock : public Layer {
public:
    ConvAttentionBlock(size_t d_model, size_t seq_len, size_t num_heads = 1,
                       size_t kernel_size = 3, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "ConvAttentionBlock"; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t ffn_dim_;

    LayerNorm ln1_;                // pre-attn
    ConvAttention attn_;
    LayerNorm ln2_;                // pre-FFN
    Dense ffn_fc1_;                // (ffn_dim, d_model)
    Dense ffn_fc2_;                // (d_model, ffn_dim)

    // Cache
    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_h_pre_;            // (n, ffn_dim)
    Tensor last_h_act_;            // (n, ffn_dim)  GELU(last_h_pre_)
    Tensor last_ffn_out_;
};

// ============================================================================
// ConvAttentionModel — stack of blocks + per-token classifier
// ============================================================================

class ConvAttentionModel : public Layer {
public:
    ConvAttentionModel(size_t d_model, size_t seq_len, size_t out_features,
                       size_t num_blocks = 1, size_t num_heads = 1,
                       size_t kernel_size = 3, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "ConvAttentionModel"; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t out_features_;
    size_t num_blocks_;
    size_t num_heads_;
    std::vector<std::unique_ptr<ConvAttentionBlock>> blocks_;
    Dense classifier_;

    Tensor last_input_;
    Tensor last_block_output_;
};

#endif
