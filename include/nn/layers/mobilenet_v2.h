#ifndef MOBILENET_V2_H
#define MOBILENET_V2_H

#include "../core/layer.h"
#include "convolutions/conv_layer.h"
#include "pooling/pool_layer.h"

// MobileNetV2: inverted residual blocks with linear bottleneck.
// 1) Expand: 1x1 conv (in -> expansion_factor * in_channels)
// 2) Depthwise: 3x3 conv (expansion_factor * in_channels -> expansion_factor * in_channels)
// 3) Linear: no activation after pointwise conv (linear bottleneck)
// 4) Skip connection if input/output channels match and stride==1
class InvertedResidual : public Layer {
public:
    // input_channels, output_channels, stride (1 or 2), expansion_factor t, H, W
    InvertedResidual(size_t in_channels, size_t out_channels, size_t stride,
                     size_t expansion_factor, size_t H, size_t W);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    size_t in_channels_, out_channels_, stride_, expansion_factor_;
    Conv2D expand_conv_;    // 1x1 expand
    Conv2D depthwise_conv_; // 3x3 depthwise
    Conv2D project_conv_;  // 1x1 linear project (no activation)
    size_t H_out_, W_out_;
    bool skip_connection_;
    Tensor last_output_;
};

class MobileNetV2 : public Layer {
public:
    // num_classes: output classes, width_multiplier: width scaling (0-1), H,W: input spatial
    MobileNetV2(size_t num_classes = 1000, double width_multiplier = 1.0,
                size_t H_in = 224, size_t W_in = 224);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D first_conv_;
    std::vector<InvertedResidual> residual_blocks_;
    Conv2D final_conv_; // 1x1 conv to expand to num_classes
    Dense classifier_;
    Tensor last_output_;
    double width_mult_;
};

#endif