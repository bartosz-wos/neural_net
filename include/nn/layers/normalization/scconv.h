#ifndef SCCONV_H
#define SCCONV_H

#include "../../core/layer.h"

// SCConv: Spatial and Channel Reconstruction (CVPR 2020)
// Reconstructs features to suppress channel and spatial redundancy.
// Two branches: Channel Reconstruction (CR) and Spatial Reconstruction (SR)
// CR: group statistics (mean, std per channel group), reconstruct per group
// SR: spatial attention via L2 norm per spatial location
// Input:  (N, C, H, W) stored as (N, C*H*W)
// Output: same shape
class SCConv : public Layer {
public:
    // channels: number of input channels (must be divisible by groups)
    // groups: number of channel groups for reconstruction (default 4)
    SCConv(int channels, int groups = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    int channels_, groups_;
    size_t channels_per_group_;
    // Channel reconstruction: gamma/beta per group
    // gamma: (groups, channels_per_group), beta: same
    Tensor gamma_cr_, beta_cr_;
    Tensor grad_gamma_cr_, grad_beta_cr_;
    // Spatial reconstruction: theta (HxW x HxW attention) — simplified: L2 norm weight
    // We store per-spatial-location scale: (1, H*W)
    Tensor theta_sr_;
    Tensor grad_theta_sr_;
    // Cached
    Tensor last_input_;       // (N, C, H, W)
    Tensor last_group_mean_; // (N, groups, channels_per_group)
    Tensor last_group_std_;  // (N, groups, channels_per_group)
    Tensor last_spatial_norm_; // (N, H*W)
    int H_, W_, spatial_size_;
};

#endif