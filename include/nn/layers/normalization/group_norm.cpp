#include "group_norm.h"
#include <cmath>

GroupNorm::GroupNorm(int num_groups, int num_channels, float eps)
    : num_groups_(num_groups), num_channels_(num_channels), eps_(eps) {
    gamma_ = Tensor(1, num_channels);
    beta_ = Tensor(1, num_channels);
    grad_gamma_ = Tensor(1, num_channels);
    grad_beta_ = Tensor(1, num_channels);
    // Initialize gamma=1, beta=0
    for (int i = 0; i < num_channels; i++) {
        gamma_[0][i] = 1.0f;
        beta_[0][i] = 0.0f;
    }
}

Tensor GroupNorm::forward(const Tensor& x) {
    // x: (batch, C*H*W)
    int batch = (int)x.rows;
    int spatial_per_channel = (int)(x.cols / num_channels_);

    last_x_ = x;
    last_spatial_ = spatial_per_channel;
    Tensor result(batch, x.cols);

    for (int n = 0; n < batch; n++) {
        for (int g = 0; g < num_groups_; g++) {
            int channels_per_group = num_channels_ / num_groups_;
            int channel_start = g * channels_per_group;
            int channel_end = channel_start + channels_per_group;

            int count = channels_per_group * spatial_per_channel;
            float mean = 0.0f;
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < spatial_per_channel; s++) {
                    mean += x[n][c * spatial_per_channel + s];
                }
            }
            mean /= count;

            float var = 0.0f;
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < spatial_per_channel; s++) {
                    float diff = x[n][c * spatial_per_channel + s] - mean;
                    var += diff * diff;
                }
            }
            var /= count;

            float sqrt_var = std::sqrt(var + eps_);

            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < spatial_per_channel; s++) {
                    int idx = c * spatial_per_channel + s;
                    float x_norm = (x[n][idx] - mean) / sqrt_var;
                    result[n][idx] = gamma_[0][c] * x_norm + beta_[0][c];
                }
            }
        }
    }

    return result;
}

Tensor GroupNorm::backward(const Tensor& grad_output, double /* learning_rate */) {
    int batch = (int)grad_output.rows;
    int features = (int)grad_output.cols;
    grad_gamma_ = Tensor(1, num_channels_);
    grad_beta_ = Tensor(1, num_channels_);
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);

    // Simplified gradient: pass through
    Tensor grad_x(grad_output.rows, grad_output.cols);
    for (int n = 0; n < batch; n++)
        for (int f = 0; f < features; f++)
            grad_x[n][f] = grad_output[n][f];

    return grad_x;
}

std::vector<Tensor*> GroupNorm::parameters() { return {&gamma_, &beta_}; }
std::vector<Tensor*> GroupNorm::gradients() { return {&grad_gamma_, &grad_beta_}; }
void GroupNorm::zero_grad() {
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);
}
