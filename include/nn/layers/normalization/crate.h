#ifndef NN_NORMALIZATION_CRATE_H
#define NN_NORMALIZATION_CRATE_H

#include "../../core/layer.h"

// CRATE: Convolutions with Rectified Activations
// Uses channel attention with ReLU gating (no sigmoid).
// Key innovation: ReLU-based channel importance with positive scaling, L1 normalized.
// Architecture: DepthwiseConv -> GAP -> FC1 -> ReLU -> FC2 -> ReLU -> L1 normalize -> channel scale
class CRATE : public Layer {
public:
    // channels: number of input channels
    // reduction: channel reduction ratio for MLP (default 4, like MobileNet)
    CRATE(int channels, int reduction = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return fc1_weight_; }
    Tensor get_gradients() const override { return grad_fc1_weight_; }
    std::string name() const override { return "CRATE"; }

private:
    int channels_;
    int reduction_;
    int spatial_size_;
    int H_, W_;

    // Depthwise conv: C groups, kernel 3x3, stride 1, padding 1
    // Stored as separate per-channel kernels: (C, 9) weight, (C,) bias
    Tensor dw_weight_;   // (C, 9)
    Tensor dw_bias_;     // (C,)
    Tensor grad_dw_weight_;
    Tensor grad_dw_bias_;

    // Channel attention MLP
    Tensor fc1_weight_;  // (C, C/reduction) — weights are transposed in storage
    Tensor fc1_bias_;    // (1, C/reduction)
    Tensor fc2_weight_; // (C/reduction, C)
    Tensor fc2_bias_;    // (1, C)
    Tensor grad_fc1_weight_, grad_fc1_bias_;
    Tensor grad_fc2_weight_, grad_fc2_bias_;

    // Cached
    Tensor last_input_;      // (N, C*H*W)
    Tensor last_after_dw_;   // (N, C*H*W)
    Tensor last_attention_;  // (N, C)
    Tensor last_gap_;        // (N, C)

    double relu(double x) const { return std::max(0.0, x); }
};

#endif