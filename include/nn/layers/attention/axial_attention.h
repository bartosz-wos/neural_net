#ifndef AXIAL_ATTENTION_H
#define AXIAL_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Axial Attention — Ho, Kalchbrenner, Weissenborn, Salimans 2019
//   "Axial Attention in Multidimensional Transformers"
//   (https://arxiv.org/abs/1912.12180)
//
// Idea: for a (H, W) grid of tokens flattened to (H*W, d_model), standard
// self-attention is O((H*W)^2 · d). Axial attention factors the attention
// into two passes — one along each axis:
//   * Row-axis pass:    for each H-row, attend over the W tokens of that row
//   * Column-axis pass: for each W-column, attend over the H tokens of that col
// Each axis has O(H·W · (H+W)) cost (much cheaper for large grids).
//
// We fuse the two axis outputs with a residual sum:
//   output = out_row_pre + out_col_pre + input
// matching Ho et al. 2019 §3.1 (no learnable combine). Per-axis causal mask
// is applied: row-axis masks out j' > j within a row, column-axis masks out
// i' > i within a column.
//
// Conventions:
//   * Input:  (H*W, d_model)    — n tokens, d_model features
//   * Output: (H*W, d_model)
//   * d_model must be evenly divisible by num_heads (v1 = 1 head per axis)
//   * Per-axis causal mask (default true); both axes share the flag
//   * Biases on output projections only (Llama/Mistral style); no bias on Q/K/V
//
// Classes:
//   AxialAttention         — the two-axis attention layer
//   AxialAttentionBlock    — pre-LN → axial attn → residual → pre-LN → FFN → residual
//   AxialAttentionModel    — input projection → stack of blocks → per-token head
// ============================================================================

class AxialAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // H, W:           2D grid dimensions; input.rows must equal H*W
    // num_heads:      number of heads per axis (v1 uses MHA-style, 1 head by default)
    // causal:         apply per-axis causal mask (default true)
    AxialAttention(size_t d_model, size_t H, size_t W,
                   size_t num_heads = 1, bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q_row; }
    Tensor get_gradients() const override { return grad_W_q_row; }
    std::string name() const override { return "AxialAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t H()         const { return H_; }
    size_t W()         const { return W_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    bool    causal()   const { return causal_; }

    // Cache accessors (for tests + FD perturbation)
    const Tensor& last_attn_row() const { return last_attn_row_; }
    const Tensor& last_attn_col() const { return last_attn_col_; }
    Tensor grad_input() const { return last_d_input_; }

    // Public parameter tensors (test access + mutation)
    // W_*_row are used by the row-axis pass; W_*_col by the column-axis pass.
    Tensor W_q_row, W_k_row, W_v_row, W_o_row;       // (d_model, d_model) each
    Tensor W_q_col, W_k_col, W_v_col, W_o_col;       // (d_model, d_model) each
    Tensor b_row, b_col;                              // (1, d_model) each
    Tensor grad_W_q_row, grad_W_k_row, grad_W_v_row, grad_W_o_row;
    Tensor grad_W_q_col, grad_W_k_col, grad_W_v_col, grad_W_o_col;
    Tensor grad_b_row, grad_b_col;

private:
    size_t d_model_, H_, W_, num_heads_, head_dim_;
    bool   causal_;
    double scale_;

    // BPTT cache
    Tensor last_input_;                            // (H*W, d_model)
    Tensor last_q_row_, last_k_row_, last_v_row_;    // (H*W, d_model)
    Tensor last_q_col_, last_k_col_, last_v_col_;    // (H*W, d_model)
    Tensor last_attn_row_, last_attn_col_;          // (H*W, H*W) — softmax output per axis
    Tensor last_head_out_row_, last_head_out_col_;  // (H*W, d_model) — A @ V
    Tensor last_out_row_, last_out_col_;            // (H*W, d_model) — head_out @ W_o (+ b)
    Tensor last_d_input_;                           // (H*W, d_model)
};

// ============================================================================
// AxialAttentionBlock — pre-LN → AxialAttention → residual →
//                       pre-LN → GELU FFN → residual (optional FFN)
// ============================================================================

class AxialAttentionBlock : public Layer {
public:
    // ffn_dim: FFN hidden dim. If 0, no FFN sub-layer (pure attention block).
    AxialAttentionBlock(size_t d_model, size_t H, size_t W,
                         size_t num_heads = 1,
                         size_t ffn_dim = 0,
                         bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn_.W_q_row; }
    Tensor get_gradients() const override { return attn_.grad_W_q_row; }
    std::string name() const override { return "AxialAttentionBlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_, ffn_dim_;

    AxialAttention attn_;
    LayerNorm ln1_;
    LayerNorm ln2_;
    Dense ffn_fc1_;
    Dense ffn_fc2_;

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
// AxialAttentionModel — input projection → stack of blocks → per-token head
// ============================================================================

class AxialAttentionModel : public Layer {
public:
    AxialAttentionModel(size_t d_input,
                        size_t d_model,
                        size_t d_output,
                        size_t H,
                        size_t W,
                        size_t num_blocks = 2,
                        size_t num_heads = 1,
                        size_t ffn_dim = 0,
                        bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_.get_weights(); }
    Tensor get_gradients() const override { return W_in_.get_gradients(); }
    std::string name() const override { return "AxialAttentionModel"; }

private:
    size_t d_input_, d_model_, d_output_;
    Dense W_in_, W_out_;
    LayerNorm ln_final_;
    std::vector<AxialAttentionBlock> blocks_;

    Tensor last_input_;
    Tensor last_proj_;
};

#endif