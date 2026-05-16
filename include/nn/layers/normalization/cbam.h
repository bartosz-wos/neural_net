#ifndef CBAM_H
#define CBAM_H

#include "../../core/layer.h"

// CBAM: Convolutional Block Attention Module
// Sequential: Channel Attention (CAm) then Spatial Attention (SAm)
// Channel Attention: avg_pool + max_pool -> MLP -> sum -> sigmoid
// Spatial Attention: avg_pool + max_pool along channel -> 7x7 conv -> sigmoid
// Input:  (N, C, H, W) stored as (N, C*H*W)
// Output: same shape, with attention applied
class CBAM : public Layer {
public:
    // channels: number of input channels
    // reduction: channel reduction ratio for MLP in channel attention (default 16)
    // kernel_size: kernel for spatial attention conv (default 7)
    CBAM(int channels, int reduction = 16, int kernel_size = 7);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    int channels_, reduction_, kernel_size_;
    // Channel attention: MLP
    Tensor fc1_weight_, fc1_bias_;  // C -> C/r
    Tensor fc2_weight_, fc2_bias_;  // C/r -> C
    Tensor grad_fc1_weight_, grad_fc1_bias_;
    Tensor grad_fc2_weight_, grad_fc2_bias_;
    // Spatial attention: 7x7 conv 2 -> 1
    Tensor spa_weight_, spa_bias_;  // (1, 2, 7, 7) stored as row-major
    Tensor grad_spa_weight_, grad_spa_bias_;
    // Cached
    Tensor last_input_;         // (N, C, H, W)
    Tensor last_channel_att_;  // (N, C) channel attention weights
    Tensor last_spatial_att_;  // (N, H, W) spatial attention weights (stored as N*H*W)
    Tensor last_spa_avg_;      // (N, H*W) avg pool over channels
    Tensor last_spa_max_;      // (N, H*W) max pool over channels
    int H_, W_, spatial_size_;
    int pad_;
};

#endif