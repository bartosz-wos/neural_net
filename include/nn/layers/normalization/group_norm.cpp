#include "group_norm.h"
#include <cmath>
#include <stdexcept>

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

    if (num_groups_ == 0 || num_groups_ > (int)num_channels_) {
        throw std::invalid_argument("GroupNorm: num_groups must be > 0 and <= num_channels");
    }

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
            var = std::max(var, 1e-5f);

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

    Tensor grad_x(grad_output.rows, grad_output.cols);
    grad_x.fill(0.0);

    for (int n = 0; n < batch; n++) {
        for (int g = 0; g < num_groups_; g++) {
            int channels_per_group = num_channels_ / num_groups_;
            int channel_start = g * channels_per_group;
            int channel_end = channel_start + channels_per_group;
            int count = channels_per_group * last_spatial_;

            // Recompute mean and variance from cached input
            float mean = 0.0f;
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < last_spatial_; s++) {
                    mean += last_x_[n][c * last_spatial_ + s];
                }
            }
            mean /= count;

            float var = 0.0f;
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < last_spatial_; s++) {
                    float diff = last_x_[n][c * last_spatial_ + s] - mean;
                    var += diff * diff;
                }
            }
            var /= count;

            float sqrt_var = std::sqrt(var + eps_);
            float inv_var = 1.0f / sqrt_var;
            float inv_var3 = inv_var * inv_var * inv_var;  // for dL/dvar chain

            // dL/dgamma and dL/dbeta (accumulate across spatial)
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < last_spatial_; s++) {
                    int idx = c * last_spatial_ + s;
                    float x_norm = (last_x_[n][idx] - mean) * inv_var;
                    grad_gamma_[0][c] += grad_output[n][idx] * x_norm;
                    grad_beta_[0][c]  += grad_output[n][idx];
                }
            }

            // Compute dL/dvar and dL/dmean for the group
            // dL/dvar = sum( dL/dy * gamma * (x - mean) * (-0.5) * inv_var^3 )
            // dL/dmean = sum( dL/dy * gamma * (-inv_var) ) + dL/dvar * (-2) * sum(x - mean) / count
            //           = sum( dL/dy * gamma * (-inv_var) )  (second term = 0 since sum(x-mean)=0)
            float dL_dvar = 0.0f;
            float dL_dmean = 0.0f;
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < last_spatial_; s++) {
                    int idx = c * last_spatial_ + s;
                    float diff = last_x_[n][idx] - mean;
                    float grad_y = grad_output[n][idx] * gamma_[0][c];
                    dL_dvar  += grad_y * diff * (-0.5f) * inv_var3;
                    dL_dmean += grad_y * (-inv_var);
                }
            }

            // dL/dx for each element:
            // dx_norm = grad_y * inv_var
            // dx_var  = dL_dvar * 2 * diff / count
            // dx_mean = dL_dmean / count
            // dx = dx_norm + dx_var + dx_mean
            for (int c = channel_start; c < channel_end; c++) {
                for (int s = 0; s < last_spatial_; s++) {
                    int idx = c * last_spatial_ + s;
                    float diff = last_x_[n][idx] - mean;
                    float grad_y = grad_output[n][idx] * gamma_[0][c];
                    float dx_norm = grad_y * inv_var;
                    float dx_var  = dL_dvar * 2.0f * diff / static_cast<float>(count);
                    float dx_mean = dL_dmean / static_cast<float>(count);
                    grad_x[n][idx] = dx_norm + dx_var + dx_mean;
                }
            }
        }
    }

    return grad_x;
}

std::vector<Tensor*> GroupNorm::parameters() { return {&gamma_, &beta_}; }
std::vector<Tensor*> GroupNorm::gradients() { return {&grad_gamma_, &grad_beta_}; }
void GroupNorm::zero_grad() {
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);
}
