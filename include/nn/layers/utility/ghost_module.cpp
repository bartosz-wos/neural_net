#include "ghost_module.h"
#include <cmath>
#include <stdexcept>
#include <random>

GhostModule::GhostModule(int in_channels, int out_channels,
                         int kernel_size, int ratio)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), ratio_(ratio),
      primary_channels_(out_channels / ratio),
      ghost_channels_(out_channels - out_channels / ratio),
      primary_conv_(in_channels, primary_channels_, 1, 1, 0, 0),  // 1x1 conv
      // cheap depthwise conv: groups = primary_channels_, each channel processed independently
      // Input to cheap_conv is (N, primary_channels, H, W), output same shape
      cheap_conv_(primary_channels_, primary_channels_,
                  kernel_size, kernel_size, 0, 0,
                  1, 1, 0, 0, 1, 1)  // depthwise: groups=in_ch=out_ch
{
    // Note: cheap_conv_ uses depthwise via groups=in_channels=out_channels
    // The Conv2D constructor initializes weights for (out_channels, in_channels*kH*kW).
    // For depthwise, we need to reinterpret: each output channel only connects to its
    // corresponding input channel. The weight matrix is (primary_ch, primary_ch * kH * kW).
    // We'll re-init the cheap_conv_ weights to be depthwise.
    // Since cheap_conv_ was initialized as groups=primary_ch (depthwise), we
    // only use the per-channel kernel weights.
    // Re-initialize cheap conv weights: identity per channel group
    int fan_in = kernel_size * kernel_size;
    double scale = std::sqrt(2.0 / fan_in);
    std::mt19937 gen(123);
    std::normal_distribution<> dis(0.0, scale);
    cheap_conv_.grad_weights.fill(0.0);
    cheap_conv_.grad_bias.fill(0.0);
    // Depthwise: each of primary_ch filters has primary_ch * kH * kW weights,
    // but only the weights corresponding to channel c (the kH*kW block starting at c*kH*kW)
    // should be non-zero. We init those with small random values, rest are zero.
    cheap_conv_.weights.fill(0.0);
    for (int c = 0; c < primary_channels_; ++c) {
        for (int i = 0; i < kernel_size * kernel_size; ++i) {
            cheap_conv_.weights[c][c * kernel_size_ * kernel_size_ + i] = dis(gen);
        }
    }
    cheap_conv_.bias.fill(0.0);
}

Tensor GhostModule::forward(const Tensor& input) {
    // input: (N, in_ch, H, W) stored as (N, in_ch * H * W)
    H_ = W_ = 0;
    // Infer spatial dims: input.cols = in_ch * H * W
    // We need to know H and W. We'll search for valid H that divides.
    // Since we don't know H and W separately, we store the flat size.
    // We assume square for simplicity in this implementation, or we store H=W=sqrt.
    // Better: we store H*W = input.cols / in_ch_
    int spatial = input.cols / in_channels_;
    H_ = static_cast<int>(std::sqrt(spatial));
    W_ = spatial / H_;
    if (H_ * W_ != spatial) {
        throw std::runtime_error("GhostModule: input spatial dimensions must factor cleanly");
    }
    last_input_ = input;

    // Primary conv: 1x1 conv to generate primary features
    // primary_conv_ expects (N, in_ch, H, W) as (N, in_ch*H*W)
    last_primary_ = primary_conv_.forward(input);
    int primary_spatial = last_primary_.cols / primary_channels_;
    H_out_ = static_cast<int>(std::sqrt(primary_spatial));
    W_out_ = primary_spatial / H_out_;

    // Cheap depthwise: generate ghost features
    // cheap_conv_ is depthwise: groups=primary_channels_
    // Its forward expects (N, primary_channels_, H_out, W_out)
    // But it stores as (N, primary_channels_ * H_out * W_out) internally
    // So we can just call forward directly
    Tensor ghost_features = cheap_conv_.forward(last_primary_);

    // Concatenate along channel dimension: dim 1
    // Output: (N, (primary_ch + ghost_ch), H_out, W_out) = (N, out_ch, H_out, W_out)
    Tensor output(input.rows, out_channels_ * H_out_ * W_out_);

    int primary_size = primary_channels_ * H_out_ * W_out_;
    int ghost_size = ghost_channels_ * H_out_ * W_out_;

    // Copy primary features (already at correct spatial dims)
    for (size_t n = 0; n < input.rows; ++n) {
        for (int i = 0; i < primary_size; ++i) {
            output[n][i] = last_primary_[n][i];
        }
    }
    // Copy ghost features
    for (size_t n = 0; n < input.rows; ++n) {
        for (int i = 0; i < ghost_size; ++i) {
            output[n][primary_size + i] = ghost_features[n][i];
        }
    }

    return output;
}

Tensor GhostModule::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_ch, H_out, W_out)
    // Split: grad_primary = first primary_size per sample, grad_ghost = rest
    int primary_size = primary_channels_ * H_out_ * W_out_;
    int ghost_size = ghost_channels_ * H_out_ * W_out_;

    Tensor grad_primary(last_input_.rows, primary_size);
    Tensor grad_ghost(last_input_.rows, ghost_size);
    for (size_t n = 0; n < grad_output.rows; ++n) {
        for (int i = 0; i < primary_size; ++i) grad_primary[n][i] = grad_output[n][i];
        for (int i = 0; i < ghost_size; ++i) grad_ghost[n][i] = grad_output[n][primary_size + i];
    }

    // Backward through cheap conv (depthwise)
    Tensor grad_primary_from_cheap = cheap_conv_.backward(grad_ghost, learning_rate);

    // Backward through primary conv
    Tensor grad_input = primary_conv_.backward(grad_primary, learning_rate);

    // Add contributions
    for (size_t n = 0; n < grad_input.rows; ++n)
        for (size_t i = 0; i < grad_input.cols; ++i)
            grad_input[n][i] += grad_primary_from_cheap[n][i];

    return grad_input;
}

void GhostModule::update_weights(double learning_rate) {
    primary_conv_.update_weights(learning_rate);
    cheap_conv_.update_weights(learning_rate);
}

void GhostModule::zero_grad() {
    primary_conv_.zero_grad();
    cheap_conv_.zero_grad();
}

std::vector<Tensor*> GhostModule::parameters() {
    return primary_conv_.parameters();
}

std::vector<Tensor*> GhostModule::gradients() {
    return primary_conv_.gradients();
}