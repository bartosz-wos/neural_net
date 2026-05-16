#include "eca.h"
#include <cmath>
#include <random>
#include <stdexcept>

ECALayer::ECALayer(int channels, int kernel_size)
    : channels_(channels), kernel_size_(kernel_size),
      weight_1d_(1, kernel_size),
      bias_1d_(1, kernel_size),
      grad_weight_1d_(1, kernel_size),
      grad_bias_1d_(1, kernel_size)
{
    // Initialize weights with Xavier for 1D conv
    double scale = std::sqrt(2.0 / (1 + kernel_size));
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);
    for (int i = 0; i < kernel_size; ++i) {
        weight_1d_[0][i] = dis(gen);
        bias_1d_[0][i] = 0.0;
    }
}

Tensor ECALayer::forward(const Tensor& input) {
    // input: (N, C, H, W) stored as (N, C*H*W)
    int N = input.rows;
    spatial_size_ = input.cols / channels_;
    H_ = static_cast<int>(std::sqrt(spatial_size_));
    W_ = spatial_size_ / H_;
    if (H_ * W_ != spatial_size_) {
        throw std::runtime_error("ECALayer: spatial dims must factor cleanly");
    }
    last_input_ = input;

    // Global average pooling over spatial dims -> (N, C)
    last_attention_ = Tensor(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double sum = 0.0;
            for (int h = 0; h < H_; ++h) {
                for (int w = 0; w < W_; ++w) {
                    sum += input[n][c * spatial_size_ + h * W_ + w];
                }
            }
            last_attention_[n][c] = sum / spatial_size_;
        }
    }

    // Apply 1D convolution: (N, C) -> (N, C) with 1x1 in "channel" but k-wide along C
    // Efficient ECA: treat each sample independently, compute 1D conv over channel axis
    // The 1D conv slides a kernel of size k across the channel dimension.
    // Simplified: for each sample, we compute:
    //   attention[c] = sigmoid(sum_{k=0}^{K-1} weight[k] * gap[c + k - K/2])
    // This is a "valid" 1D convolution where we iterate only valid positions.
    // To keep output dimensions the same, we pad implicitly.
    // Standard ECA uses: y[i] = sum_{j=0}^{k-1} w[j] * x[i + j - floor(k/2)]
    // with implicit zero-padding for boundary.

    Tensor output_attention(N, channels_);
    int half_k = kernel_size_ / 2;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double conv_sum = bias_1d_[0][half_k];  // center weight (conceptually)
            // The bias in 1D conv is per-output-channel (1 output), so it's per position.
            // We add bias once per output channel.
            // Since 1D conv is valid-only (1 output channel), bias is a scalar added to all.
            // But for per-channel attention, we add the bias term.
            // ECA: attention = sigmoid(1D_conv(gap, kernel_size) + bias)
            // Actually the standard ECA is: y = sigmoid(conv1d(gap, k) + b)
            // where conv1d has 1 input channel, 1 output channel.
            // So the "bias" is just a single scalar added to the result.
            // Let's compute it properly: for valid positions (which is all of them with padding).
        }
    }

    // Actually, let me implement ECA as per the paper:
    // attention = sigmoid(1D_Conv(gap))
    // 1D_Conv: input (C,), kernel (k,), output (C,)
    // With 'same' padding (implicit zero pad at boundaries).
    // Output size = C.
    // bias is (1,) per output channel of 1D conv.
    // Since 1 output channel: bias is a single scalar.
    // But wait: 1D conv with groups=1, in_channels=1, out_channels=1 -> bias shape (1,)
    // That's added to each position. But with 'same' padding, there are C output positions.
    // So this is wrong. Let me reconsider.

    // Actually in ECA, the 1D conv is applied as:
    // y[i] = sum_j w[j] * x[i+j]
    // where x has length C (for each sample).
    // The convolution uses 'valid' mode (no padding), so output length = C - k + 1.
    // Then ECA uses 'same' padding by zero-padding x to length C + k - 1.
    // With 'same', output length = C.
    // bias is a scalar added to each output position.
    // So: attention = sigmoid(1D_conv(gap, kernel=k, padding=same) + b)
    // But the 1D conv in ECA has 1 input channel and 1 output channel.
    // So it takes (C,) -> (C,) with a k-sized kernel.
    // Implementation: for each position c, compute sum over kernel.
    // boundary handling: zero-pad at edges.

    // Reset and recompute properly
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double val = bias_1d_[0][0];  // single bias scalar
            int half_k = kernel_size_ / 2;
            for (int k = 0; k < kernel_size_; ++k) {
                int src = c - half_k + k;
                double w = weight_1d_[0][k];
                if (src >= 0 && src < channels_) {
                    val += w * last_attention_[n][src];
                }
                // else: implicit zero, no contribution
            }
            // Sigmoid
            output_attention[n][c] = 1.0 / (1.0 + std::exp(-val));
        }
    }
    last_attention_ = output_attention;

    // Apply channel attention: scale each channel across all spatial positions
    Tensor output(N, channels_ * spatial_size_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double scale = output_attention[n][c];
            int spatial_base = c * spatial_size_;
            int out_spatial_base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                output[n][out_spatial_base + s] = input[n][spatial_base + s] * scale;
            }
        }
    }
    return output;
}

Tensor ECALayer::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    Tensor grad_input(N, channels_ * spatial_size_);

    // The gradient splits into two paths:
    // (1) dL/d(attention) * d(attention)/d(gap) * d(gap)/d(input)
    // (2) dL/d(input) directly from scale * input
    //
    // d(output)/d(input) = attention[c] for each spatial position
    // So grad_input = grad_output * attention (hadamard)
    // But we also need grad_attention for the 1D conv backward.

    // First: grad through the scale multiplication
    // grad_attention_from_scale = sum_s grad_output[n][c*S+s] * input[n][c*S+s]
    Tensor grad_attention(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double ga = 0.0;
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                ga += grad_output[n][base + s] * last_input_[n][base + s];
            }
            grad_attention[n][c] = ga;  // dL/d(attention[n][c]) = sum_s dL/do * x
        }
    }

    // Now backward through 1D conv (sigmoid + conv)
    // attention = sigmoid(conv1d(gap) + b)
    // d(attention)/d(gap) = sigmoid' * conv1d weights (as a Toeplitz matrix)
    // For each position c: d_att[c] = attention[c] * (1 - attention[c]) * sum_j w[j] * d_att_conv[j]
    // But we have d_att already. So we need d_att_conv = conv1d_backward(d_att, w)
    // The 1D conv is: conv[c] = sum_j w[j] * gap[c + j - half_k] (same padding)
    // So d_gap[c] = sum_j d_conv[c + j] * w[j] (with proper boundary handling)
    // Actually since the conv is linear: d_conv/d_gap = Toeplitz(w)
    // So d_gap = conv1d_backward(d_conv, w) = cross-correlation with w

    // d_att = sigmoid'(conv) * d_conv = attention * (1 - attention) * d_conv
    // So: d_conv = d_att / (attention * (1 - attention))
    // Then: d_gap[c] = sum_j d_conv[c + j] * w[j] (with boundary checks)
    Tensor grad_gap(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double sig = last_attention_[n][c];
            double sig_deriv = sig * (1.0 - sig);
            if (sig_deriv < 1e-12) sig_deriv = 1e-12;
            double d_conv_center = grad_attention[n][c] / sig_deriv;

            // d_gap[c] = sum over kernel positions: d_conv[valid_pos] * w[kernel_pos]
            // where valid_pos = c + offset, offset = k - half_k for kernel pos k
            int half_k = kernel_size_ / 2;
            double dg = 0.0;
            for (int k = 0; k < kernel_size_; ++k) {
                int src = c - half_k + k;
                double w = weight_1d_[0][k];
                if (src >= 0 && src < channels_) {
                    // d_conv at position src
                    double sig_src = last_attention_[n][src];
                    double sig_deriv_src = sig_src * (1.0 - sig_src);
                    if (sig_deriv_src < 1e-12) sig_deriv_src = 1e-12;
                    double d_conv_src = grad_attention[n][src] / sig_deriv_src;
                    dg += d_conv_src * w;
                }
            }
            grad_gap[n][c] = dg;
        }
    }

    // d_gap/d_input: gap is avg pool, so each input position contributes equally
    // d_input[n][c*S+s] += d_gap[n][c] / spatial_size_
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double dg = grad_gap[n][c] / spatial_size_;
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                grad_input[n][base + s] += dg;
            }
        }
    }

    // Also the direct path: grad_input = grad_output * attention
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double scale = last_attention_[n][c];
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                grad_input[n][base + s] += grad_output[n][base + s] * scale;
            }
        }
    }

    // Update weights: grad through the conv
    // d_weight[k] = sum_n sum_c d_att[n][c] / sigmoid_deriv * gap[n][c + half_k - k]
    // Actually d_conv[c] = d_att[c] / (att * (1-att)), and conv[c] = sum_j w[j] * gap[c+j-half_k]
    // So d_w[k] = sum_c d_conv[c] * gap[c + k - half_k] (with boundary check)
    grad_weight_1d_.fill(0.0);
    grad_bias_1d_.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double sig = last_attention_[n][c];
            double sig_deriv = sig * (1.0 - sig);
            if (sig_deriv < 1e-12) sig_deriv = 1e-12;
            double d_conv_c = grad_attention[n][c] / sig_deriv;
            int half_k = kernel_size_ / 2;
            for (int k = 0; k < kernel_size_; ++k) {
                int src = c - half_k + k;
                if (src >= 0 && src < channels_) {
                    grad_weight_1d_[0][k] += d_conv_c * last_attention_[n][src];
                }
            }
            grad_bias_1d_[0][0] += d_conv_c;
        }
    }

    return grad_input;
}

void ECALayer::update_weights(double learning_rate) {
    for (int k = 0; k < kernel_size_; ++k) {
        weight_1d_[0][k] -= learning_rate * grad_weight_1d_[0][k];
    }
    bias_1d_[0][0] -= learning_rate * grad_bias_1d_[0][0];
}

void ECALayer::zero_grad() {
    grad_weight_1d_.fill(0.0);
    grad_bias_1d_.fill(0.0);
}

std::vector<Tensor*> ECALayer::parameters() {
    return {&weight_1d_, &bias_1d_};
}

std::vector<Tensor*> ECALayer::gradients() {
    return {&grad_weight_1d_, &grad_bias_1d_};
}