// TabNet: Hierarchical Attention for Tabular Deep Learning
// Arik & Pfister 2019 — https://arxiv.org/abs/1908.07442
#ifndef TABNET_H
#define TABNET_H

#include "../../core/layer.h"
#include <vector>

// TabNet uses sequential attention to select which features to use at each decision step.
// Each step focuses on a subset of features, then contributions are accumulated.

class TabNet : public Layer {
public:
    TabNet(int input_dim, int num_outputs, int num_steps = 3,
           int num_attention_heads = 2, int n_independent = 2,
           double relaxation_factor = 1.5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training_ = t; }

    Tensor getAttentionMask(int step) const;
    std::vector<Tensor> getAttentionMasks() const;
    Tensor get_weights() const override { return step_fcs_.front().w; }
    Tensor get_gradients() const override { return step_fcs_.front().grad_w; }

private:
    int input_dim_;
    int num_outputs_;
    int num_steps_;
    int num_attention_heads_;
    int n_independent_;
    double relaxation_factor_;
    int virtual_dim_;
    bool training_;

    // Shared encoder: batch_norm -> fc -> batch_norm -> fc -> relu
    // Used across all decision steps
    struct EncoderBlock {
        Tensor w1, b1;           // FC1: input_dim -> virtual_dim
        Tensor bn1_gamma, bn1_beta;
        Tensor w2, b2;           // FC2: virtual_dim -> virtual_dim
        Tensor bn2_gamma, bn2_beta;
        // Forward caches (per call)
        Tensor last_bn1_out;            // post-BN1
        Tensor last_bn2_out;            // post-BN2 (pre-ReLU)
        Tensor last_mean_bn1, last_inv_std_bn1;
        Tensor last_x_centered_bn1, last_x_norm_bn1;
        Tensor last_mean_bn2, last_inv_std_bn2;
        Tensor last_x_centered_bn2, last_x_norm_bn2;
        // Gradients
        Tensor grad_bn1_gamma, grad_bn1_beta;
        Tensor grad_bn2_gamma, grad_bn2_beta;
        Tensor grad_w1, grad_b1;
        Tensor grad_w2, grad_b2;
    };
    EncoderBlock shared_encoder_;

    // Per-step independent encoder (not shared across steps)
    // batch_norm -> fc -> batch_norm -> fc -> relu
    struct IndependentBlock {
        Tensor w1, b1;
        Tensor bn1_gamma, bn1_beta;
        Tensor w2, b2;
        Tensor bn2_gamma, bn2_beta;
        // Forward caches
        Tensor last_bn1_out;
        Tensor last_bn2_out;
        Tensor last_mean_bn1, last_inv_std_bn1;
        Tensor last_x_centered_bn1, last_x_norm_bn1;
        Tensor last_mean_bn2, last_inv_std_bn2;
        Tensor last_x_centered_bn2, last_x_norm_bn2;
        // Gradients
        Tensor grad_bn1_gamma, grad_bn1_beta;
        Tensor grad_bn2_gamma, grad_bn2_beta;
        Tensor grad_w1, grad_b1;
        Tensor grad_w2, grad_b2;
    };
    std::vector<IndependentBlock> independent_blocks_;

    // Per-step attention block
    // Computes sparse feature attention mask
    struct AttentionBlock {
        Tensor w;                // (1, virtual_dim) — projects h to attention scores
        Tensor last_scores;      // (batch, virtual_dim) — raw attention scores before softmax
        Tensor last_mask;        // (batch, input_dim) — final relaxed mask
        Tensor grad_w;
    };
    std::vector<AttentionBlock> attention_blocks_;

    // Per-step FC output layer: masked features -> num_outputs
    struct StepFC {
        Tensor w, b;
        Tensor grad_w, grad_b;
        // Forward caches
        Tensor last_h_shared;     // (B, virtual_dim) input to step FC
        Tensor last_dec_pre;      // (B, num_outputs) pre-ReLU activations
    };
    std::vector<StepFC> step_fcs_;

    // Input projection: maps (B, input_dim) -> (B, virtual_dim) at step 0 so that
    // every subsequent step operates in virtual_dim space (with virtual_dim-shaped masks).
    Tensor W_in_;             // (virtual_dim_, input_dim_)
    Tensor b_in_;             // (1, virtual_dim_)
    Tensor grad_W_in_;        // (virtual_dim_, input_dim_)
    Tensor grad_b_in_;        // (1, virtual_dim_)
    Tensor last_input_proj_;  // (B, virtual_dim_) cached for backward
    Tensor last_input_;       // (B, input_dim_) cached original input (needed for W_in grad)

    // Prior scales for relaxation factor
    // Updated after each step: P_s = (gamma - M_{s-1}) / gamma
    std::vector<Tensor> prior_scales_;

    // Cache for backward pass
    std::vector<Tensor> last_h_;           // input at each step
    std::vector<Tensor> last_h_indep_;     // output of independent encoder at each step
    std::vector<Tensor> last_masked_h_;    // masked features at each step
    std::vector<Tensor> last_masks_;       // attention masks at each step

    // All parameters for parameter()
    std::vector<Tensor*> param_list_;
    std::vector<Tensor*> grad_list_;
};

#endif // TABNET_H