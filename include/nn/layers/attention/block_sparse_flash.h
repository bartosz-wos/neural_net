#ifndef BLOCK_SPARSE_FLASH_H
#define BLOCK_SPARSE_FLASH_H

#include "../../core/layer.h"
#include <vector>
#include <cstdint>

// ============================================================================
// Block-Sparse Flash Attention
//
// Reference: Dao et al. 2022 "FlashAttention: Fast and Memory-Efficient Exact
// Attention with IO-Awareness" §4 (Block-Sparse FlashAttention).
//   https://arxiv.org/abs/2205.14135
//
// Same online-softmax flash recurrence as FlashAttentionV2, but with a 2D
// block mask M ∈ {0,1}^{n_q_blocks × n_k_blocks} that gates which (Q-block,
// K-block) pairs participate. Masked-out blocks contribute nothing to the
// forward output AND no gradient flows through them in backward.
//
// The mask generalizes:
//   - dense (all 1s)
//   - causal (lower-triangular blocks)
//   - sliding window (lower-triangular + diagonal band)
//   - strided (Longformer-style)
//   - BigBird (window + random + global)
//
// Conventions (matches SHLA / NSA / sliding_window in this repo):
//   * Input/Output: (n, d_model) — token-major
//   * d_model must be evenly divisible by num_heads
//   * Multi-head with optional GQA-style K/V sharing (num_kv_heads; 0 → num_heads)
//   * query_block_size / key_block_size are constructor-time knobs (default 4)
//   * Mask is supplied at forward time via forward_with_mask(input, mask)
//   * Mask shape (n_q_blocks, n_k_blocks) must match
//       (ceil(n / query_block_size), ceil(n / key_block_size))
//     for the given input
// ============================================================================

class BlockSparseFlashAttention : public Layer {
public:
    BlockSparseFlashAttention(size_t d_model,
                              size_t num_heads,
                              size_t num_kv_heads = 0,    // 0 → num_heads (MHA)
                              size_t query_block_size = 4,
                              size_t key_block_size = 4);

    // Layer interface — calls forward_with_mask with the last-used mask
    // (or all-ones if no mask has been set yet). For new code, prefer
    // forward_with_mask directly.
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "BlockSparseFlashAttention"; }

    // Configuration accessors
    size_t d_model()           const { return d_model_; }
    size_t num_heads()         const { return num_heads_; }
    size_t num_kv_heads()      const { return num_kv_heads_; }
    size_t head_dim()          const { return head_dim_; }
    size_t query_block_size()  const { return query_block_size_; }
    size_t key_block_size()    const { return key_block_size_; }
    size_t group_size()        const { return group_size_; }

    // Cache sizes from last forward (computed from n)
    size_t n_q_blocks()        const { return n_q_blocks_; }
    size_t n_k_blocks()        const { return n_k_blocks_; }
    size_t last_n()            const { return last_n_; }

    // Test accessors
    const Tensor& last_mask()  const { return last_mask_; }
    const Tensor& last_input() const { return last_input_; }
    const Tensor& last_query() const { return last_query_; }
    const Tensor& last_key()   const { return last_key_; }
    const Tensor& last_value() const { return last_value_; }
    const Tensor& last_context() const { return last_context_; }
    double scale()             const { return scale_; }

    // Public params (matches FA-2 convention)
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // Helpers for building standard masks. Each returns shape (n_q_blocks,
    // n_k_blocks) with values in {0, 1}. 1 = attend, 0 = skip.
    static Tensor build_dense_mask(size_t n_q_blocks, size_t n_k_blocks);
    static Tensor build_causal_mask(size_t n_q_blocks, size_t n_k_blocks);
    static Tensor build_sliding_window_mask(size_t n_q_blocks, size_t n_k_blocks,
                                            size_t window_n_blocks);
    static Tensor build_strided_mask(size_t n_q_blocks, size_t n_k_blocks,
                                     size_t stride);
    static Tensor build_bigbird_mask(size_t n_q_blocks, size_t n_k_blocks,
                                     size_t window_n_blocks, size_t n_global_blocks,
                                     uint32_t seed);

    // Forward with an explicit mask. Required because the base Layer interface
    // doesn't carry a mask parameter. The mask is (n_q_blocks, n_k_blocks) in
    // {0,1}; mask[i,j] = 1 means Q-block i attends to K-block j.
    Tensor forward_with_mask(const Tensor& input, const Tensor& mask);

private:
    size_t d_model_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t group_size_;       // num_heads / num_kv_heads
    size_t query_block_size_;
    size_t key_block_size_;
    double scale_;
    size_t n_q_blocks_;
    size_t n_k_blocks_;
    size_t last_n_;
    bool has_forward_cache_;

    Tensor last_input_;       // (n, d_model)
    Tensor last_query_;       // (n, d_model)
    Tensor last_key_;         // (n, d_model)
    Tensor last_value_;       // (n, d_model)
    Tensor last_context_;     // (n, d_model)  pre-projection attention output
    Tensor last_mask_;        // (n_q_blocks, n_k_blocks)
};

#endif