#ifndef SQUEEZE_EXCITATION_H
#define SQUEEZE_EXCITATION_H

#include "../core/layer.h"
#include "pooling/pool_layer.h"
#include "convolutions/conv_layer.h"

// Squeeze-Excitation block (Hu et al. 2018).
// Global average pool → FC → ReLU → FC → Sigmoid (channel attention).
class SEBlock : public Layer {
public:
    SEBlock(size_t in_channels, size_t reduction = 16);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    size_t in_channels() const { return in_channels_; }
    // Scale each channel by its excitation weight (broadcast multiply)
    static Tensor scale_channels(const Tensor& input, const Tensor& excitation);

private:
    Dense fc1_; // squeeze: C → C/r
    Dense fc2_; // excite: C/r → C
    size_t in_channels_, reduction_;
    Tensor last_excitation_;
    Tensor last_input_;
};

// SEResNetBlock: residual block with SE channel attention
class SEResNetBlock : public Layer {
public:
    SEResNetBlock(size_t in_channels, size_t out_channels,
                   size_t reduction = 16, size_t H = 32, size_t W = 32);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D conv1_, conv2_;
    SEBlock se_;
    Conv2D skip_conv_;
    bool has_skip_;
    Tensor last_output_;
    size_t H_, W_;
};

#endif