#ifndef FLASH_ATTENTION_V2_H
#define FLASH_ATTENTION_V2_H

#include "../../core/layer.h"

#include <cstddef>
#include <string>
#include <vector>

// FlashAttention-2-style causal self-attention.
//
// Reference: Dao, "FlashAttention-2: Faster Attention with Better Parallelism
// and Work Partitioning" (2023), https://arxiv.org/abs/2307.08691.
//
// This scalar CPU implementation is intentionally pedagogical. It adopts the
// paper's Q-block-outer work partition and online-softmax recurrence, but does
// not reproduce CUDA warp scheduling or claim GPU speedups. Forward stores one
// log-sum-exp value per (head, query) instead of a full attention matrix;
// backward recomputes attention probabilities blockwise from Q, K, and LSE.
//
// Public layout matches the repository's legacy FlashAttentionLayer:
// input/output are feature-major tensors with shape (d_model, seq_len).
class FlashAttentionV2Layer : public Layer {
public:
    FlashAttentionV2Layer(size_t d_model, size_t num_heads,
                          size_t query_block_size = 64,
                          size_t key_block_size = 64,
                          bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "FlashAttentionV2Layer"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t query_block_size() const { return query_block_size_; }
    size_t key_block_size() const { return key_block_size_; }
    bool causal() const { return causal_; }

    const Tensor& last_logsumexp() const { return last_logsumexp_; }

    // Kept public for parity with the legacy FlashAttentionLayer and for
    // deterministic reference/gradient tests.
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t query_block_size_;
    size_t key_block_size_;
    bool causal_;
    double scale_;
    bool has_forward_cache_;

    Tensor last_input_;       // (seq_len, d_model), token-major
    Tensor last_query_;       // (seq_len, d_model)
    Tensor last_key_;         // (seq_len, d_model)
    Tensor last_value_;       // (seq_len, d_model)
    Tensor last_context_;     // (seq_len, d_model), before W_o
    Tensor last_logsumexp_;   // (num_heads, seq_len)
};

#endif
