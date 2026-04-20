#ifndef RESNET_H
#define RESNET_H

#include "../core/layer.h"
#include "conv_layer.h"

// ResNet-style residual block — two conv layers with skip connection.
class ResBlock : public Layer {
public:
    ResBlock(size_t in_channels, size_t out_channels, size_t kernel_size,
             size_t stride, size_t H_in, size_t W_in);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    Conv2D conv1_, conv2_;
    bool needs_projection_;
    size_t out_channels_, H_out_, W_out_;
    Tensor last_input_;
    Tensor last_output_;
};

class ResNet : public Layer {
public:
    ResNet(size_t input_channels, size_t num_classes, size_t H_in, size_t W_in,
           const std::vector<size_t>& channels_per_stage,
           const std::vector<size_t>& strides_per_stage,
           size_t blocks_per_stage);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    Conv2D stem_;
    std::vector<ResBlock> stages_;
    Dense fc_;
    Tensor last_output_;
};

#endif