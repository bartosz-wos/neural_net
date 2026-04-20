#ifndef CNN_MODELS_H
#define CNN_MODELS_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include "../pooling/pool_layer.h"
#include "../dense/flatten.h"
#include <vector>

// === VGG Block ===
// 3x3 conv * n_filters -> batch norm (optional) -> activation -> pool
// Output: (batch, n_filters, H_out, W_out) where H_out,W_out reduced by pool
class VGGBlock : public Layer {
public:
    VGGBlock(size_t in_channels, size_t filters[], size_t n_layers,
             bool use_bn = false, size_t pool_stride = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

private:
    std::vector<Conv2D> convs_;
    std::vector<Tensor> bn_gamma_, bn_beta_;
    std::vector<Tensor> running_mean_, running_var_;
    bool use_bn_;
    MaxPool2D pool_;
    Tensor last_output_;
    size_t n_layers_;
};

// === LeNet-5 ===
// Conv(6,5x5) -> AvPool(2x2) -> Conv(16,5x5) -> AvPool(2x2) -> FC(120) -> FC(84) -> FC(10)
// Input: (batch, 1, 32, 32)
class LeNet5 : public Layer {
public:
    LeNet5(size_t num_classes = 10);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

private:
    Conv2D conv1_, conv2_;
    MaxPool2D pool1_, pool2_;
    Flatten flatten_;
    Dense fc1_, fc2_, fc3_;
    Tensor last_output_;
};

// === AlexNet (simplified) ===
// Conv(96,11x11,s4) -> MaxPool(3x3,s2) -> LRN -> Conv(256,5x5) -> LRN -> MaxPool(3x3,s2)
//      -> Conv(384,3x3) -> Conv(384,3x3) -> Conv(256,3x3) -> MaxPool(3x3,s2)
//      -> FC(4096) -> FC(4096) -> FC(num_classes)
// Input: (batch, 3, 227, 227) or similar
class AlexNet : public Layer {
public:
    AlexNet(size_t num_classes = 1000, bool use_lrbn = true);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

private:
    Conv2D conv1_, conv2_, conv3_, conv4_, conv5_;
    MaxPool2D pool1_, pool2_, pool3_;
    Flatten flatten_;
    Dense fc1_, fc2_, fc3_;
    // LRN is simplified — can apply local response normalization in forward
    bool use_lrbn_;
    Tensor last_output_;
};

#endif