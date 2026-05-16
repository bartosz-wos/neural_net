#ifndef ECA_H
#define ECA_H

#include "../../core/layer.h"

// ECA: Efficient Channel Attention (ECA-Net)
// Uses 1D conv of kernel size k to compute channel attention,
// avoiding global pooling. Adaptive kernel size: k = ceil(log2(C)/2) * 2 - 1
// Input:  (N, C, H, W) stored as (N, C*H*W)
// Output: same shape, channel-wise scale factor applied
class ECALayer : public Layer {
public:
    // channels: number of input channels
    // kernel_size: 1D conv kernel size (default 3)
    ECALayer(int channels, int kernel_size = 3);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return weight_1d_; }
    Tensor get_gradients() const override { return grad_weight_1d_; }

private:
    int channels_, kernel_size_;
    // 1D conv weights: (1, kernel_size) — 1 input channel, 1 output channel
    Tensor weight_1d_;
    Tensor bias_1d_;
    Tensor grad_weight_1d_;
    Tensor grad_bias_1d_;
    // Cached
    Tensor last_input_;        // (N, C, H, W) stored as (N, C*H*W)
    Tensor last_attention_;    // (N, C)
    int H_, W_, spatial_size_;
};

#endif