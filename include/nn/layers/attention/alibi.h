#ifndef ALIBI_H
#define ALIBI_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// ALiBi (Attention with Linear Biases) — Press et al. 2022
//   "Train Short, Test Long: Attention with Linear Biases Enables Input
//    Length Extrapolation" (https://arxiv.org/abs/2108.12409)
//
// ALiBi replaces standard positional encodings (sinusoidal, learned, RoPE)
// with a fixed, additive linear distance bias added DIRECTLY to the
// pre-softmax attention scores. There are NO learned position parameters.
//
// For each attention head h, a scalar slope m_h is fixed at construction:
//     m_h = 2^(-8 / n_h * (h + 1))      for h = 0..n_h - 1
// The slopes follow a geometric sequence — different heads focus on different
// distance scales (some near, some far).
//
// For a query at position q and a key at position k, the bias added to the
// pre-softmax score for head h is:
//     bias_h(q, k) = -m_h * (q - k)        (non-causal)
//     bias_h(q, k) = -m_h * |q - k|        (alternative form)
//
// We use the non-causal form (q - k) which gives negative bias for keys
// further back in the sequence. This matches the Mistral convention.
//
// Math summary per single-head attention layer (input x ∈ R^{n × d_model}):
//
//   Q = x W_q^T + b_q  ∈ R^{n × d_k}
//   K = x W_k^T + b_k  ∈ R^{n × d_k}
//   V = x W_v^T + b_v  ∈ R^{n × d_k}
//   scores = Q K^T / sqrt(d_k)            ∈ R^{n × n}
//   scores[q, k] -= m_h * (q - k)         (ALiBi bias — same m for single-head)
//   A      = softmax(scores)              ∈ R^{n × n}    (row-wise)
//   out_pre = A V                         ∈ R^{n × d_k}
//   out     = out_pre W_o^T + b_o         ∈ R^{n × d_model}
//
// Backward: the bias is a CONSTANT w.r.t. softmax output, so the standard
// softmax BPTT applies unchanged. No gradient flows to m_h (it's a
// hyperparameter, not a parameter). Q/K/V/O gradients are identical to
// standard scaled dot-product attention.
//
// Multi-head: in the multi-head variant we repeat this per head with a
// per-head slope (vector of size num_heads). For the first iteration we
// provide a single-head AlibiAttention + AlibiBlock (pre-LN wrapper).
// ALiBiMultiHeadAttention follows the standard multi-head pattern from
// the repo (GQA / MLA) and is implemented in the same file.
//
// Used by Mistral — the leading open-source LLM family.
//
// API conventions (matching Linformer / Performer / Transformer in this repo):
//   * (n, d_model) input/output — row-major
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual)
//   * Q/K/V/O: Dense convention y = x W^T + b, W ∈ R^{d_model × d_model}
// ============================================================================

class AlibiAttention : public Layer {
public:
    // d_model: input/output feature dim
    // seq_len: sequence length n (FIXED at construction)
    // num_heads: number of attention heads (each gets its own slope)
    AlibiAttention(size_t d_model, size_t seq_len, size_t num_heads = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "AlibiAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    const Tensor& slopes() const { return slopes_; }       // (num_heads,)
    const Tensor& alibi_bias() const { return alibi_bias_; } // (num_heads, seq_len*seq_len)
    // Test-only accessors (write-enabled) for last_attn_, grad_W_q, etc.
    Tensor& mutable_last_attn() { return last_attn_; }
    Tensor& mutable_grad_W_q() { return grad_W_q; }

    // Test-only: set the per-head slope values and recompute the bias
    // matrix. Used to test "no-bias" output behaviour.
    void set_slopes(const std::vector<double>& new_slopes) {
        for (size_t h = 0; h < num_heads_; ++h) {
            slopes_(0, h) = new_slopes[h];
        }
        for (size_t h = 0; h < num_heads_; ++h) {
            double m = slopes_(0, h);
            for (size_t q = 0; q < seq_len_; ++q) {
                for (size_t k = 0; k < seq_len_; ++k) {
                    alibi_bias_(h, q * seq_len_ + k) = -m * (static_cast<double>(q) - static_cast<double>(k));
                }
            }
        }
    }

    size_t d_model_;
    size_t seq_len_;
    size_t num_heads_;
    size_t head_dim_;       // d_model / num_heads

    // Learned Q/K/V/O projections — Dense convention: y = x W^T + b
    // W shape: (d_model, d_model)
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    // Fixed (non-learnable) ALiBi per-head slopes and precomputed bias.
    // slopes_:       (num_heads,) — m_h = 2^(-8/n_h * (h+1))
    // alibi_bias_:   (num_heads, seq_len * seq_len) flat — row h*seq_len*seq_len + q*seq_len + k
    //                gives bias_h[q, k] = -m_h * (q - k)
    Tensor slopes_;
    Tensor alibi_bias_;

    // BPTT cache
    Tensor last_input_;     // (n, d_model)
    Tensor last_q_;         // (n, d_model) stacked Q heads
    Tensor last_k_;         // (n, d_model) stacked K heads
    Tensor last_v_;         // (n, d_model) stacked V heads
    Tensor last_scores_;    // (num_heads, n, n) pre-softmax scores with ALiBi
    Tensor last_attn_;      // (num_heads, n, n) post-softmax attention weights
    Tensor last_out_;       // (n, d_model) attention output before W_o projection
};

// ----------------------------------------------------------------------------
// AlibiBlock: pre-LN → AlibiAttention → residual → pre-LN → FFN → residual
//   Mirrors the standard transformer block layout. The FFN is a 2-layer Dense
//   with GELU. Mirrors PerformerBlock / GQABlock / MLABlock.
// ----------------------------------------------------------------------------
class AlibiBlock : public Layer {
public:
    size_t d_model_, num_heads_;
    AlibiAttention attn;
    LayerNorm ln1, ln2;
    // FFN: 2 Dense layers, GELU in between
    Tensor W1, b1, W2, b2;
    Tensor grad_W1, grad_b1, grad_W2, grad_b2;

    // BPTT cache
    Tensor last_x_;
    Tensor last_ln1_out_;
    Tensor last_attn_out_;
    Tensor last_resid1_;
    Tensor last_ln2_out_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_out_;

    AlibiBlock(size_t d_model, size_t seq_len, size_t num_heads = 1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::string name() const override { return "AlibiBlock"; }
};

// ----------------------------------------------------------------------------
// AlibiModel: stack of `num_blocks` AlibiBlocks + classifier head
//   (n, d_model) input → (n, out_features) per-token logits.
// ----------------------------------------------------------------------------
class AlibiModel : public Layer {
public:
    size_t d_model_, num_blocks_, out_features_;
    std::vector<AlibiBlock> blocks_;
    LayerNorm final_ln_;
    Tensor W_out_, b_out_;
    Tensor grad_W_out_, grad_b_out_;

    // BPTT cache for input projection + classifier
    Tensor W_in_, b_in_;
    Tensor grad_W_in_, grad_b_in_;
    Tensor last_input_;
    Tensor last_in_proj_;
    Tensor last_final_ln_;
    Tensor last_logits_;

    AlibiModel(size_t d_model, size_t seq_len, size_t out_features,
               size_t num_blocks = 2, size_t num_heads = 1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "AlibiModel"; }
};

#endif // ALIBI_H