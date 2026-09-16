#ifndef FOCAL_MODULATION_H
#define FOCAL_MODULATION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Focal Modulation Networks (FocalNet)
// ============================================================================
//
// Reference: Yang, Li, Zeng, Lu, Wang, Zhang, 2022, NeurIPS.
//   "Focal Modulation Networks" (https://arxiv.org/abs/2203.11945)
//
// Replaces self-attention with a stack of depthwise-sequential "context"
// levels fanned back across spatial positions through a per-level modulator
// that derives from a GAP query. The three-stage recipe from the paper:
//
//   1. Hierarchical context (L levels): z_0 = x; z_l = depthwise_shift(z_{l-1})
//      for l ≥ 1. Each level l carries a learnable per-channel scalar
//      `level_scalars_[l]` that gates its output `g_l = w_l ⊙ z_l`.
//   2. Query via global-average-pool: q = (1/S) * Σ_s x[:, s, :], shape (B, D).
//   3. Per-level modulator MLP: m_l = sigmoid(W2_l · GELU(W1_l · q) + b2_l).
//      m_l is shape (B, D) and is broadcast across S spatial positions.
//   4. Aggregate: y = Σ_l m_l ⊙ g_l  (per-spatial-position broadcast).
//   5. Output projection: out = y @ W_o^T + b_o  via Dense.
//
// Block wrapper (matches paper's pre-LN residual structure):
//     z1 = x + FocalModulation(LN1(x))
//     out = z1 + FFN(LN2(z1))    (FFN optional, defaults to ffn_mult=4)
//
// Model: input_proj → num_blocks FocalModulationBlocks → final LN → mean-pool
// over S → classifier.
//
// Conventions:
//   * Input to FocalModulation/Block/Model is (B, S*d_model) flat, matching
//     the rest of the repo.
//   * All linear projections are Dense-style: weights shape (out, in),
//     `y = x · W^T + b`.
//   * GELU is the standard `0.5x(1 + tanh(√(2/π)(x + 0.044715 x³))` formula.
// ============================================================================

// ============================================================================
// FocalModulation — the token-mixer layer
// ============================================================================

class FocalModulation : public Layer {
public:
    // d_model:     feature/channel dim D.
    // seq_len:     token count S (must be known statically).
    // num_levels:  L (number of hierarchical context levels).
    // mlp_dim:     hidden dim of the per-level modulator MLP; 0 → D (=2D)*2.
    FocalModulation(size_t d_model, size_t seq_len,
                    size_t num_levels = 2,
                    size_t mlp_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return level_scalars_[0]; }
    Tensor get_gradients() const override { return grad_level_scalars_[0]; }
    std::string name() const override { return "FocalModulation"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_levels() const { return num_levels_; }
    size_t mlp_dim() const { return mlp_dim_; }
    size_t param_count() const { return cached_param_count_; }

    // Per-level parameter accessors (for tests)
    Tensor& level_scalars(size_t l) { return level_scalars_[l]; }
    Tensor& W1(size_t l) { return W1_[l]; }
    Tensor& b1(size_t l) { return b1_[l]; }
    Tensor& W2(size_t l) { return W2_[l]; }
    Tensor& b2(size_t l) { return b2_[l]; }
    Tensor& Wo() { return W_o; }
    Tensor& bo() { return b_o; }

    Tensor& grad_level_scalars(size_t l) { return grad_level_scalars_[l]; }
    Tensor& grad_W1(size_t l) { return grad_W1_[l]; }
    Tensor& grad_b1(size_t l) { return grad_b1_[l]; }
    Tensor& grad_W2(size_t l) { return grad_W2_[l]; }
    Tensor& grad_b2(size_t l) { return grad_b2_[l]; }
    Tensor& grad_Wo() { return grad_W_o; }
    Tensor& grad_bo() { return grad_b_o; }
    Tensor& grad_input() { return last_d_input_; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t num_levels_;
    size_t mlp_dim_;
    size_t cached_param_count_;

    // Level-wise per-channel scalar: w_l[d], shape (1, d_model).
    std::vector<Tensor> level_scalars_;
    std::vector<Tensor> grad_level_scalars_;

    // Modulator MLPs: per level: W1_l (mlp_dim, d_model), b1_l (1, mlp_dim),
    //                 W2_l (d_model, mlp_dim), b2_l (1, d_model).
    std::vector<Tensor> W1_, b1_, W2_, b2_;
    std::vector<Tensor> grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    // Output projection: W_o (d_model, d_model), b_o (1, d_model).
    Tensor W_o, b_o;
    Tensor grad_W_o, grad_b_o;

    // Per-level "shift-by-1" sequential input derived during forward.
    // z_l_flat[b, s*D + d] = level-l's sequential feature at position s.
    std::vector<Tensor> last_z_l_;          // size L; each (B, S*D)
    std::vector<Tensor> last_g_l_;          // size L; each (B, S*D) = w_l ⊙ z_l

    // Per-level modulator (broadcast across S) — shape (B, D) per level.
    std::vector<Tensor> last_m_l_;          // size L; each (B, D)
    std::vector<Tensor> last_mlp_h_;        // size L; each (B, mlp_dim) pre-GELU
    std::vector<Tensor> last_mlp_h_act_;    // size L; each (B, mlp_dim) post-GELU

    Tensor last_y_;                          // (B, S*D) post-aggregate, pre-output-proj
    Tensor last_input_;                      // (B, S*D) original flat input
    Tensor last_q_;                          // (B, D) GAP query — cached for backward
    Tensor last_d_input_;                    // (B, S*D) input gradient
};


// ============================================================================
// FocalModulationBlock — pre-LN + FocalModulation + residual + FFN
// ============================================================================

class FocalModulationBlock : public Layer {
public:
    // d_model:    feature dim.
    // seq_len:    token count S.
    // num_levels: number of hierarchical levels in the focal modulation.
    // mlp_dim:    hidden dim of the modulator MLP (0 → D*2 by default).
    // ffn_mult:   multiplier for the dense FFN hidden dim (0 = no FFN).
    FocalModulationBlock(size_t d_model, size_t seq_len,
                         size_t num_levels = 2,
                         size_t mlp_dim = 0,
                         size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "FocalModulationBlock"; }

    // Accessors
    FocalModulation& focal() { return focal_; }
    LayerNorm& ln1() { return ln1_; }
    LayerNorm& ln2() { return ln2_; }
    bool has_ffn() const { return ffn_mult_ > 0; }
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    Tensor& grad_input() { return last_d_input_; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t ffn_mult_;
    bool has_ffn_;

    LayerNorm ln1_;
    LayerNorm ln2_;
    FocalModulation focal_;

    // FFN: W1 (ffn_dim, d_model), b1 (1, ffn_dim), W2 (d_model, ffn_dim), b2 (1, d_model)
    Tensor ffn_W1_, ffn_b1_, ffn_W2_, ffn_b2_;
    Tensor grad_ffn_W1_, grad_ffn_b1_, grad_ffn_W2_, grad_ffn_b2_;
    Tensor last_z1_;                        // (B, S*D) post LN1
    Tensor last_focal_out_;                 // (B, S*D)
    Tensor last_z2_;                        // (B, S*D) post-residual + LN2
    Tensor last_ffn_h_act_;                 // (B*S, ffn_dim) post-GELU
    Tensor last_ffn_out_;                   // (B, S*D)
    Tensor last_d_input_;
};

// ============================================================================
// FocalModulationModel — input projection → block stack → mean-pool → classifier
// ============================================================================

class FocalModulationModel : public Layer {
public:
    FocalModulationModel(size_t input_dim, size_t d_model, size_t output_dim,
                         size_t seq_len,
                         size_t num_blocks,
                         size_t num_levels = 2,
                         size_t mlp_dim = 0,
                         size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj_.weights; }
    Tensor get_gradients() const override { return input_proj_.grad_weights; }
    std::string name() const override { return "FocalModulationModel"; }

    size_t num_blocks() const { return num_blocks_; }
    std::vector<FocalModulationBlock>& blocks() { return blocks_; }
    Dense& input_proj() { return input_proj_; }
    Dense& classifier() { return classifier_; }
    LayerNorm& head_ln() { return head_ln_; }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t seq_len_;
    size_t num_blocks_;

    Dense input_proj_;          // (d_model, input_dim)
    LayerNorm head_ln_;
    Dense classifier_;          // (output_dim, d_model)
    std::vector<FocalModulationBlock> blocks_;

    Tensor last_input_;         // (B, S * input_dim)
    Tensor last_z_proj_;        // (B, S * d_model) post input projection
    Tensor last_blocks_out_;    // (B, S * d_model) post block stack
    Tensor last_head_ln_;       // (B, S * d_model)
    Tensor last_pooled_;        // (B, d_model)
    Tensor last_logits_;        // (B, output_dim)
    Tensor last_d_input_;       // (B, S * input_dim)
};

#endif
