#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include "../core/layer.h"
#include "convolutions/conv_layer.h"
#include "pooling/pool_layer.h"
#include "dense/flatten.h"

// SqueezeNet: Fire module — squeeze 1x1 conv followed by expand 1x1 + 3x3 conv.
// Very parameter-efficient, similar accuracy to AlexNet with 50x fewer params.
class FireModule : public Layer {
public:
    // in_channels: input feature maps
    // squeeze_ch: output channels of squeeze 1x1 conv
    // expand1_ch: output channels of expand 1x1 conv
    // expand3_ch: output channels of expand 3x3 conv (spatial)
    FireModule(size_t in_channels, size_t squeeze_ch,
               size_t expand1_ch, size_t expand3_ch,
               size_t H_in, size_t W_in);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D squeeze_;     // 1x1 conv
    Conv2D expand1x1_;    // 1x1 expand
    Conv2D expand3x3_;     // 3x3 expand (padded to same spatial size)
    size_t expand_ch_;     // expand1 + expand3 total channels
    size_t H_out_, W_out_;
    Tensor last_output_;
};

class SqueezeNet : public Layer {
public:
    SqueezeNet(size_t num_classes = 1000, size_t H_in = 224, size_t W_in = 224);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D conv1_;  // initial conv
    MaxPool2D pool1_;
    std::vector<FireModule> fire_modules_;
    MaxPool2D pool_final_;
    Conv2D conv10_; // final 1x1 conv (1x1 or 1x1 conv to num_classes)
    Flatten flatten_;
    Tensor last_output_;
};

#endif