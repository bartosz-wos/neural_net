#ifndef COORD_ATTENTION_H
#define COORD_ATTENTION_H

#include "../../core/layer.h"

// CoordAttention: Coordinate Attention (CVPR 2021)
// Embeds coordinate information into channel attention.
// Two-step:
//   (1) Transform to (H,1) and (1,W) to capture long-range horiz/vert dependencies
//   (2) 1x1 conv mixer, then sigmoid, then multiply with original x
// Input:  (N, C, H, W) stored as (N, C*H*W)
// Output: same shape
class CoordAttention : public Layer {
public:
    // channels: number of input channels
    // reduction: channel reduction ratio (default 32)
    CoordAttention(int channels, int reduction = 32);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    int channels_, reduction_;
    // Horizontal branch: avg pool along W -> (N, C, H, 1) -> conv1x1 -> relu
    // Vertical branch: avg pool along H -> (N, C, 1, W) -> conv1x1 -> relu
    // Pooled channels: C -> C/r for the 1x1 conv
    size_t reduced_channels_;
    Tensor conv1x1_h_weight_, conv1x1_h_bias_; // (C/r, C)
    Tensor conv1x1_v_weight_, conv1x1_v_bias_; // (C/r, C)
    Tensor conv1x1_out_weight_, conv1x1_out_bias_; // (C, 2*C/r)
    Tensor grad_conv1x1_h_weight_, grad_conv1x1_h_bias_;
    Tensor grad_conv1x1_v_weight_, grad_conv1x1_v_bias_;
    Tensor grad_conv1x1_out_weight_, grad_conv1x1_out_bias_;
    // Cached
    Tensor last_input_;    // (N, C, H, W)
    Tensor last_attention_; // (N, C, H, W) attention map
    int H_, W_, spatial_size_;
};

#endif