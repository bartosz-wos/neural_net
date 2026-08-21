#ifndef HYPER_CONNECTION_H
#define HYPER_CONNECTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>

// ============================================================================
// Hyper-Connection — DeepSeek-AI 2025
//   "Hyper-Connections" (https://arxiv.org/abs/2409.19606)
//
// Core idea: replace the implicit residual strength "1" with two
// learnable per-channel parameters α, β ∈ (0, 1):
//
//     out = α ⊙ x + β ⊙ inner(x)
//
// where α = sigmoid(α_log) and β = sigmoid(β_log). At init (α_log ≈ +10,
// β_log ≈ -10) this recovers the standard residual (out ≈ x). The
// parameters start close to a residual identity so early-training dynamics
// match the standard residual, then α/β deviate as they receive gradients.
//
// Math (n_samples, d_model):
//   α ∈ (0, 1)^d, β ∈ (0, 1)^d  — per-channel (broadcast across batch)
//   sub_out = inner(x)                       (n, d)
//   out[t, j] = α[0, j] · x[t, j] + β[0, j] · sub_out[t, j]
//
// Backward (with d_out:
//   d sub_out[t, j] = β[0, j] · d_out[t, j]                       (n, d)
//   d x_residual[t, j] = α[0, j] · d_out[t, j]                    (n, d)
//   d x_inner = inner.backward(d sub_out, lr)                      (n, d)
//   d x total = α ⊙ d_out + d x_inner                              (n, d)
//
//   d α_log[0, j] = sigmoid'(α_log[0, j]) · Σ_t d_out[t, j] · x[t, j]
//   d β_log[0, j] = sigmoid'(β_log[0, j]) · Σ_t d_out[t, j] · sub_out[t, j]
// ============================================================================

class HyperConnection : public Layer {
public:
    // d_model: input/output feature dim. inner: owned sub-layer.
    HyperConnection(size_t d_model, Layer* inner);
    ~HyperConnection() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return alpha_log_; }
    Tensor get_gradients() const override { return grad_alpha_log_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    Layer* inner_layer() const { return inner_.get(); }

    // Parameter accessors (for tests and direct inspection).
    Tensor& alpha_log() { return alpha_log_; }
    Tensor& beta_log() { return beta_log_; }
    const Tensor& grad_alpha_log() const { return grad_alpha_log_; }
    const Tensor& grad_beta_log() const { return grad_beta_log_; }

private:
    size_t d_model_;
    std::unique_ptr<Layer> inner_;

    Tensor alpha_log_;       // (1, d_model), unconstrained — sigmoid reparam
    Tensor beta_log_;        // (1, d_model), unconstrained — sigmoid reparam
    Tensor grad_alpha_log_;  // (1, d_model)
    Tensor grad_beta_log_;   // (1, d_model)

    // Caches for backward pass
    Tensor last_input_;      // (n, d_model)
    Tensor last_sub_out_;    // (n, d_model)
    Tensor last_alpha_;      // (1, d_model)  sigmoid(alpha_log_)
    Tensor last_beta_;       // (1, d_model)  sigmoid(beta_log_)
};

// ============================================================================
// HyperConnectionBlock: pre-LN → Dense(d, ffn) → GELU → Dense(ffn, d) →
//                       α ⊙ x + β ⊙ sub_out
//
// Owns all its internal sublayers (LayerNorm + 2 Dense + HyperConnection).
// ============================================================================
class HyperConnectionBlock : public Layer {
public:
    HyperConnectionBlock(size_t d_model, size_t ffn_dim);
    ~HyperConnectionBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    // Parameter accessors (for tests and direct inspection).
    Tensor& alpha_log() { return alpha_log_; }
    Tensor& beta_log() { return beta_log_; }
    const Tensor& grad_alpha_log() const { return grad_alpha_log_; }
    const Tensor& grad_beta_log() const { return grad_beta_log_; }

private:
    size_t d_model_, ffn_dim_;
    std::unique_ptr<LayerNorm> ln_;     // pre-norm
    std::unique_ptr<Dense> fc1_;        // d_model -> ffn_dim
    std::unique_ptr<Dense> fc2_;        // ffn_dim -> d_model

    // Hyper-connection parameters (the α, β learnable residual scalers)
    Tensor alpha_log_;       // (1, d_model)
    Tensor beta_log_;        // (1, d_model)
    Tensor grad_alpha_log_;  // (1, d_model)
    Tensor grad_beta_log_;   // (1, d_model)

    Tensor last_input_;                  // (n, d_model)
    Tensor last_ln_out_;                 // (n, d_model)
    Tensor last_ffn_hidden_;             // (n, ffn_dim)
    Tensor last_sub_out_;                // (n, d_model)
    Tensor last_alpha_;                  // (1, d_model)
    Tensor last_beta_;                   // (1, d_model)
};

// ============================================================================
// HyperConnectionModel: input projection → N HyperConnectionBlocks → classifier
// ============================================================================
class HyperConnectionModel : public Layer {
public:
    HyperConnectionModel(size_t input_dim, size_t d_model, size_t output_dim,
                         size_t num_blocks, size_t ffn_dim);
    ~HyperConnectionModel() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    size_t input_dim_, d_model_, output_dim_, num_blocks_, ffn_dim_;
    std::unique_ptr<Dense> input_proj_;
    std::vector<std::unique_ptr<HyperConnectionBlock>> blocks_;
    std::unique_ptr<Dense> classifier_;

    Tensor last_input_;  // (n, input_dim)
    Tensor last_proj_;   // (n, d_model)
    std::vector<Tensor> block_outputs_; // cached for backward
};

#endif // HYPER_CONNECTION_H
