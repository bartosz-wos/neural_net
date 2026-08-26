#ifndef BLOCK_SPARSE_FLASH_H
#define BLOCK_SPARSE_FLASH_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
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

    // Per-head running max and running sum (from forward's online softmax).
    // These are needed by the backward pass to recover the GLOBAL softmax
    // probabilities (not just per-block) — critical for correct gradients.
    // Stored as Tensors (using aligned allocator) for test access, AND as
    // plain std::vector<double> for backward access (avoids Tensor copy
    // assignment overhead and resize issues under repeated forwards).
    Tensor last_m_h_;   // shape (num_heads, n)
    Tensor last_L_h_;   // shape (num_heads, n)
    std::vector<double> last_m_h_storage_;   // size: num_heads * n
    std::vector<double> last_L_h_storage_;   // size: num_heads * n
};

// ============================================================================
// BlockSparseFlashBlock — pre-LN → BlockSparseFlash → residual → optional
// pre-LN FFN → residual (matches SHLABlock convention).
// ============================================================================

class BlockSparseFlashBlock : public Layer {
public:
    // ffn_dim = 0 disables the FFN sublayer (attention-only block).
    BlockSparseFlashBlock(size_t d_model, size_t num_heads,
                          size_t num_kv_heads, size_t query_block_size,
                          size_t key_block_size, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "BlockSparseFlashBlock"; }

    // Test accessors — the block rebuilds the same dense mask every forward.
    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    const Tensor& last_mask() const { return last_mask_; }

    // Public forward that takes an explicit mask (for tests that exercise
    // sparse patterns inside the residual block).
    Tensor forward_with_mask(const Tensor& input, const Tensor& mask);

private:
    size_t d_model_;
    size_t ffn_dim_;
    size_t query_block_size_;
    size_t key_block_size_;
    size_t n_q_blocks_;
    size_t n_k_blocks_;

    LayerNorm ln1_;                      // pre-attention
    BlockSparseFlashAttention attn_;
    LayerNorm ln2_;                      // pre-FFN
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    Tensor last_mask_;                   // (n_q_blocks, n_k_blocks) dense
    Tensor last_input_;                  // (n, d_model)
    Tensor last_z1_;                     // post-ln1, pre-attn
    Tensor last_attn_out_;               // post-attn, pre-residual
    Tensor last_res1_;                   // post-residual1 (input + attn)
    Tensor last_z2_;                     // post-ln2 (or last_res1_ if no FFN)
    Tensor last_ffn_hidden_;             // PRE-GELU
    Tensor last_ffn_out_;                // post-FFN, pre-residual2
};

// ============================================================================
// BlockSparseFlashModel — input projection → N blocks → classifier head.
// Uses a dense mask inside each block (block-sparse without an external mask
// is equivalent to standard FlashAttention; explicit mask on the Block is
// available via block.forward_with_mask for experimental configurations).
// ============================================================================

class BlockSparseFlashModel : public Layer {
public:
    BlockSparseFlashModel(size_t input_dim, size_t d_model, size_t output_dim,
                          size_t num_blocks, size_t num_heads,
                          size_t num_kv_heads, size_t query_block_size,
                          size_t key_block_size, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "BlockSparseFlashModel"; }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_blocks_;

    Tensor W_in_;
    Tensor b_in_;
    Tensor W_out_;
    Tensor b_out_;
    Tensor grad_W_in_;
    Tensor grad_b_in_;
    Tensor grad_W_out_;
    Tensor grad_b_out_;

    std::vector<BlockSparseFlashBlock> blocks_;

    Tensor last_input_;       // (n, input_dim) — preserved for input grad
    Tensor last_proj_;        // (n, d_model) — pre-block input
    Tensor last_block_out_;   // (n, d_model) — pre-output projection
};

#endif