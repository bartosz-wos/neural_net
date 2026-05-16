#ifndef DEPTHWISE_SEPARABLE_CONV_H
#define DEPTHWISE_SEPARABLE_CONV_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include <vector>
#include <stdexcept>

// Depthwise Separable Convolution: two-stage convolution
// Stage 1 — Depthwise: groups = in_channels (one filter per input channel)
// Stage 2 — Pointwise: 1x1 conv to mix channels
class DepthwiseSeparableConv : public Layer {
public:
    DepthwiseSeparableConv(int in_channels, int out_channels, int kernel_size,
                           int stride = 1, int padding = 0, int dilation = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "DepthwiseSeparableConv"; }

private:
    void ensure_initialized();

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    int H_;  // input spatial height
    int W_;  // input spatial width
    int H_out_;
    int W_out_;

    // Depthwise conv: one Conv2D per input channel (in_ch=1, out_ch=1)
    std::vector<Conv2D> depthwise_conv_;
    // Pointwise conv: 1x1 conv, in_channels_ -> out_channels_
    Conv2D pointwise_conv_;

    Tensor last_input_;
    Tensor last_depthwise_output_;
};

#endif