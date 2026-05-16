#ifndef GHOST_MODULE_H
#define GHOST_MODULE_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include "../../activations/activations.h"

// GhostModule from Huawei "GhostNet" paper
// Generates more feature maps from cheap operations (depthwise conv).
// Split into two parts:
//   (1) Regular conv to generate intrinsic features
//   (2) Cheap depthwise operations to generate ghost features
// Output = concat(primary_features, ghost_features)
class GhostModule : public Layer {
public:
    // in_channels: input channels
    // out_channels: total output channels (primary + ghost)
    // kernel_size: kernel for cheap depthwise operation
    // ratio: #ghost = primary * ratio, so primary_channels = out_channels / ratio
    GhostModule(int in_channels, int out_channels,
                int kernel_size = 3, int ratio = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    int in_channels_, out_channels_, primary_channels_, ghost_channels_;
    int kernel_size_, ratio_;
    // Primary conv: in_ch -> primary_ch
    Conv2D primary_conv_;
    // Cheap depthwise conv: groups = primary_channels (DW per channel)
    Conv2D cheap_conv_;
    // Cached
    Tensor last_primary_;    // (N, primary_ch, H, W) stored as (N, primary_ch*H*W)
    Tensor last_input_;       // (N, in_ch*H*W)
    // Spatial dims
    int H_, W_;
    int H_out_, W_out_;
};

#endif