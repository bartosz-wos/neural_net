#ifndef DYNAMIC_TANH_H
#define DYNAMIC_TANH_H

#include "../../core/layer.h"

// Dynamic Tanh (DyT) — Zhu, Chen, He (CVPR 2025, arXiv:2503.10622)
// "Transformers without Normalization"
//
// Drop-in replacement for LayerNorm / RMSNorm. Element-wise:
//   out = gamma * tanh(alpha * x) + beta
// where:
//   alpha  ∈ R                 — learnable scalar parameter (init 0.5 per paper §7.1)
//   gamma  ∈ R^{features}      — per-channel scale, init to ones
//   beta   ∈ R^{features}      — per-channel shift, init to zeros
//
// Unlike LayerNorm / RMSNorm, no batch-statistic computation, no eps, no
// training/inference mode. Squashes extreme values via bounded tanh.
class DynamicTanh : public Layer {
public:
    Tensor alpha;       // shape (1, 1)    — learnable scalar scaler
    Tensor gamma;       // shape (1, features) — per-channel scale
    Tensor beta;        // shape (1, features) — per-channel shift
    double init_alpha;  // recorded for re-init
    size_t features_;

    // Caches for backward
    Tensor last_input;
    Tensor last_tanh;   // tanh(alpha * x) — cached to avoid recomputing tanh in backward

    // Gradients
    Tensor grad_alpha_; // shape (1, 1)
    Tensor grad_gamma_; // shape (1, features)
    Tensor grad_beta_;  // shape (1, features)
    Tensor grad_x;      // shape (batch, features)

    DynamicTanh(size_t features, double init_alpha = 0.5);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    size_t features() const { return features_; }
    std::string name() const override { return "DynamicTanh"; }
};

#endif