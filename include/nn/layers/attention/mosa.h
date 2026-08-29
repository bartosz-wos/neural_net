#ifndef MOSA_H
#define MOSA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <limits>

// ============================================================================
// MoSA (Mixture of Sparse Attention) — Piękos, Csordás, Schmidhuber, May 2025
//   "Mixture of Sparse Attention: Content-Based Learnable Sparse Attention
//    via Expert-Choice Routing"
//   https://arxiv.org/abs/2505.00315
//
// Per head h, per token t:
//
//   r_h[t]     = sigmoid((X W_r^h)[t])              in (0, 1)
//   I_h, r_topk = TopK(r_h, k)                       k indices + their scores
//   X_s        = X[I_h]                              gathered input   (k, d)
//   Q, K, V    = X_s W_Q^h, X_s W_K^h, X_s W_V^h    per-head proj    (k, head_dim)
//   M[i, j]   = 0   if I_h[i] >= I_h[j]             (causal in ORIGINAL positions)
//             = -inf otherwise
//   A         = softmax(Q K^T * scale + M)           (k, k)
//   A_scaled  = diag(r_topk) * A                     (k, k) — router scaling
//   out_h     = A_scaled @ V @ W_O^h                 (k, d_model)
//   output[t] = sum over h with t in I_h: out_h[rank_of_t_in_I_h]
//
// Each head is an "expert" that picks its own top-k tokens via a learned
// sigmoid router. Standard causal attention runs on the gathered subset
// (causality uses ORIGINAL positions, not gathered positions). The per-head
// output is row-scaled by router scores, projected back to d_model via
// per-head W_O, and scattered back to original positions, summing
// contributions across heads.
//
// Conventions (match FoX / SHLA / GQA in this repo):
//   * Input:  (n, d_model) — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be divisible by num_heads; head_dim = d_model / num_heads
//   * No biases on any projection
//   * Causal masking based on original positions
//
// Per-head parameters:
//   W_q[h] : (d_model, head_dim)
//   W_k[h] : (d_model, head_dim)
//   W_v[h] : (d_model, head_dim)
//   W_o[h] : (head_dim, d_model)
//   W_r[h] : (d_model, 1)            — router
// Total: num_heads * (3 * d_model * head_dim + 2 * head_dim * d_model + d_model)
//      = num_heads * (5 * d_model * head_dim + d_model)    (since head_dim * d_model = d_model² / H)
//      = 5 * d_model^2 + num_heads * d_model
// ============================================================================

class MoSAAttention : public Layer {
public:
    MoSAAttention(size_t d_model, size_t num_heads, size_t top_k);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q[0]; }
    Tensor get_gradients() const override { return grad_W_q[0]; }
    std::string name() const override { return "MoSAAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t top_k()     const { return top_k_; }

    // Cached forward state (test access)
    const Tensor& get_last_router() const { return last_r_; }              // (n, num_heads) in (0,1)
    const std::vector<std::vector<size_t>>& get_last_indices() const { return last_I_; }
    const std::vector<std::vector<double>>& get_last_topk_scores() const { return last_r_topk_; }
    const std::vector<Tensor>& get_last_head_pre_o() const { return last_head_pre_o_; }  // per-head (k, head_dim) = A_scaled @ V

    // Causal mask for (head, gathered_i, gathered_j): 0 if I_h[i] >= I_h[j], -inf otherwise.
    double causal_mask(size_t h, size_t i, size_t j) const {
        return (last_I_[h][i] >= last_I_[h][j]) ? 0.0 : -std::numeric_limits<double>::infinity();
    }

    // Public parameters (test access)
    std::vector<Tensor> W_q, W_k, W_v, W_o, W_r;          // W_q[h], W_k[h], W_v[h] : (d, head_dim)
                                                          // W_o[h] : (head_dim, d)
                                                          // W_r[h] : (d, 1)
    std::vector<Tensor> grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_W_r;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t top_k_;
    double scale_;

    // BPTT cache
    Tensor last_input_;                  // (n, d_model)
    Tensor last_r_;                      // (n, num_heads) sigmoid scores
    std::vector<std::vector<size_t>> last_I_;        // last_I_[h] : (k,) selected indices
    std::vector<std::vector<double>> last_r_topk_;   // last_r_topk_[h] : (k,) top-k scores
    std::vector<Tensor> last_X_s_;                    // last_X_s_[h] : (k, d_model) gathered input
    std::vector<Tensor> last_Q_, last_K_, last_V_;    // per-head (k, head_dim)
    std::vector<Tensor> last_A_;                      // per-head (k, k) causal softmax
    std::vector<Tensor> last_A_scaled_;               // per-head (k, k) row-scaled
    std::vector<Tensor> last_head_pre_o_;             // per-head (k, head_dim) = A_scaled @ V
    Tensor last_head_out_;                            // (n, d_model) scattered head output
    Tensor last_d_input_;
};

// ============================================================================
// MoSABlock — pre-LN → MoSAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

class MoSABlock : public Layer {
public:
    MoSABlock(size_t d_model, size_t num_heads, size_t top_k, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MoSABlock"; }

    MoSAAttention& attn() { return attn_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;
    MoSAAttention attn_;
    LayerNorm ln2_;
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    Tensor last_res1_;
    Tensor last_ffn_hidden_;
    Tensor last_d_input_;
};

// ============================================================================
// MoSAModel — input projection → stack of MoSABlocks → output projection
// ============================================================================

class MoSAModel : public Layer {
public:
    MoSAModel(size_t input_dim, size_t d_model, size_t output_dim,
              size_t num_blocks, size_t num_heads, size_t top_k, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "MoSAModel"; }

    size_t num_blocks() const { return blocks_.size(); }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;
    std::vector<MoSABlock> blocks_;
    Tensor last_input_;
    Tensor last_proj_;
    Tensor last_block_out_;
};

#endif
