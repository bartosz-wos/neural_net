#include "crate.h"
#include <cmath>
#include <random>
#include <stdexcept>

CRATE::CRATE(int channels, int reduction)
    : channels_(channels), reduction_(reduction),
      dw_weight_(channels, 9),           // (C, 9) per-channel 3x3 kernels
      dw_bias_(1, channels),              // (C,)
      grad_dw_weight_(channels, 9),
      grad_dw_bias_(1, channels),
      // Channel attention MLP: GAP -> FC1 -> ReLU -> FC2 -> ReLU -> L1 normalize
      // FC1: (C/reduction, C) weight, (1, C/reduction) bias
      fc1_weight_(reduction, channels),   // transposed for efficient matmul
      fc1_bias_(1, reduction),
      fc2_weight_(channels, reduction),   // (C, C/reduction)
      fc2_bias_(1, channels),
      grad_fc1_weight_(reduction, channels),
      grad_fc1_bias_(1, reduction),
      grad_fc2_weight_(channels, reduction),
      grad_fc2_bias_(1, channels)
{
    std::mt19937 gen(42);

    // Xavier init for depthwise kernels
    double dw_scale = std::sqrt(2.0 / 9.0);  // 3x3 kernel
    std::normal_distribution<> dw_dis(0.0, dw_scale);
    for (int c = 0; c < channels_; ++c) {
        for (int k = 0; k < 9; ++k) {
            dw_weight_[c][k] = dw_dis(gen);
        }
        dw_bias_[0][c] = 0.0;
    }

    // Xavier init for FC1: (C/reduction, C)
    double fc1_scale = std::sqrt(2.0 / (channels_ + reduction_));
    std::normal_distribution<> fc1_dis(0.0, fc1_scale);
    for (int i = 0; i < reduction_; ++i) {
        for (int j = 0; j < channels_; ++j) {
            fc1_weight_[i][j] = fc1_dis(gen);
        }
        fc1_bias_[0][i] = 0.0;
    }

    // Xavier init for FC2: (C, C/reduction)
    double fc2_scale = std::sqrt(2.0 / (channels_ + reduction_));
    std::normal_distribution<> fc2_dis(0.0, fc2_scale);
    for (int i = 0; i < channels_; ++i) {
        for (int j = 0; j < reduction_; ++j) {
            fc2_weight_[i][j] = fc2_dis(gen);
        }
        fc2_bias_[0][i] = 0.0;
    }
}

// relu defined in header

Tensor CRATE::forward(const Tensor& input) {
    // input: (N, C, H, W) stored as (N, C*H*W)
    int N = input.rows;
    spatial_size_ = input.cols / channels_;
    H_ = static_cast<int>(std::sqrt(spatial_size_));
    W_ = spatial_size_ / H_;
    if (H_ * W_ != spatial_size_) {
        throw std::runtime_error("CRATE: spatial dims must factor cleanly");
    }
    last_input_ = input;

    // Step 1: DepthwiseConv (depthwise separable: each channel gets its own 3x3 kernel)
    // Input: (N, C, H, W) as (N, C*H*W)
    // For channel c: apply 3x3 conv to the HxW slice
    // Kernel: dw_weight_[c] = 9 weights for 3x3, applied with padding=1 (same)
    // Depthwise conv with groups=C, channels don't mix.
    last_after_dw_ = Tensor(N, channels_ * spatial_size_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            int ch_base = c * spatial_size_;
            double* out_ptr = &last_after_dw_[n][ch_base];
            // For each output position (h, w), we need a 3x3 input patch with padding
            for (int h = 0; h < H_; ++h) {
                for (int w = 0; w < W_; ++w) {
                    double conv_sum = dw_bias_[0][c];
                    double* kw = &dw_weight_[c][0];
                    int out_idx = h * W_ + w;
                    for (int kr = -1; kr <= 1; ++kr) {
                        for (int kc = -1; kc <= 1; ++kc) {
                            int src_h = h + kr;
                            int src_w = w + kc;
                            if (src_h >= 0 && src_h < H_ && src_w >= 0 && src_w < W_) {
                                int ki = (kr + 1) * 3 + (kc + 1);
                                conv_sum += kw[ki] * input[n][ch_base + src_h * W_ + src_w];
                            }
                        }
                    }
                    out_ptr[out_idx] = conv_sum;
                }
            }
        }
    }

    // Step 2: GAP -> channel attention MLP
    // Global average pooling: (N, C, H, W) -> (N, C)
    last_gap_ = Tensor(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double sum = 0.0;
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                sum += last_after_dw_[n][base + s];
            }
            last_gap_[n][c] = sum / spatial_size_;
        }
    }

    // Step 3: FC1 -> ReLU
    // fc1_weight_: (reduction, C), fc1_bias_: (1, reduction)
    // last_gap_: (N, C)
    // s1 = ReLU(gap @ fc1_weight.T + fc1_bias) = (N, reduction)
    Tensor s1(N, reduction_);
    for (int n = 0; n < N; ++n) {
        for (int r = 0; r < reduction_; ++r) {
            double val = fc1_bias_[0][r];
            for (int c = 0; c < channels_; ++c) {
                val += last_gap_[n][c] * fc1_weight_[r][c];  // row-major: fc1_weight[r][c]
            }
            s1[n][r] = relu(val);
        }
    }

    // Step 4: FC2 -> ReLU -> positive clip (ReLU already zeroes negatives)
    // fc2_weight_: (C, reduction), fc2_bias_: (1, C)
    // s2 = ReLU(s1 @ fc2_weight.T + fc2_bias) = (N, C)
    Tensor s2(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double val = fc2_bias_[0][c];
            for (int r = 0; r < reduction_; ++r) {
                val += s1[n][r] * fc2_weight_[c][r];  // row-major: fc2_weight[c][r]
            }
            s2[n][c] = relu(val);  // clip negative to 0
        }
    }

    // Step 5: L1 normalize per sample to get attention weights
    // attention[c] = s2[c] / sum(s2)
    last_attention_ = Tensor(N, channels_);
    for (int n = 0; n < N; ++n) {
        double sum = 0.0;
        for (int c = 0; c < channels_; ++c) {
            sum += s2[n][c];
        }
        // Avoid division by zero
        if (sum < 1e-12) {
            sum = 1e-12;
        }
        for (int c = 0; c < channels_; ++c) {
            last_attention_[n][c] = s2[n][c] / sum;
        }
    }

    // Step 6: Apply channel attention to dw output
    // output = last_after_dw * attention.view(N, C, 1) broadcasted
    Tensor output(N, channels_ * spatial_size_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double scale = last_attention_[n][c];
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                output[n][base + s] = last_after_dw_[n][base + s] * scale;
            }
        }
    }
    return output;
}

Tensor CRATE::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;

    // Backward pass structure:
    // grad_output -> grad_input through:
    //   1. Scale multiplication: output = dw_out * attention
    //   2. L1 normalization: attention = s2 / sum(s2)
    //   3. FC2: s2 = ReLU(z2), z2 = s1 @ W2.T + b2
    //   4. FC1: s1 = ReLU(z1), z1 = gap @ W1.T + b1
    //   5. GAP: gap = avg_pool(dw_out)
    //   6. DepthwiseConv: dw_out = depthwise(input)

    // ===== Step 1: Scale multiplication backward =====
    // output = dw_out * attention (element-wise across spatial, broadcasted over channels)
    // grad_dw_out_from_scale = grad_output * attention
    // grad_attention = sum_s(grad_output * dw_out)
    Tensor grad_dw_out_from_scale(N, channels_ * spatial_size_);
    Tensor grad_attention_total(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double ga = 0.0;
            int base = c * spatial_size_;
            double att = last_attention_[n][c];
            for (int s = 0; s < spatial_size_; ++s) {
                grad_dw_out_from_scale[n][base + s] = grad_output[n][base + s] * att;
                ga += grad_output[n][base + s] * last_after_dw_[n][base + s];
            }
            grad_attention_total[n][c] = ga;
        }
    }

    // ===== Step 2: L1 normalization backward =====
    // attention = s2 / S, S = sum(s2)
    // da_i/ds2_j = delta_ij/S - s2_i/S^2 = delta_ij/S - a_i/S
    // grad_s2 = grad_attention @ (I/S - outer(a, 1)/S)
    // For each i: grad_s2[i] = (grad_attention[i] - dot) / S
    // where dot = sum_k(grad_attention[k] * a[k]) and S = sum(s2)
    //
    // We need S = sum(s2). Re-compute s1 and s2 during backward.
    Tensor grad_s2(N, channels_);
    for (int n = 0; n < N; ++n) {
        // Re-compute s1
        double s1_local[64];
        for (int r = 0; r < reduction_; ++r) {
            double val = fc1_bias_[0][r];
            for (int c = 0; c < channels_; ++c) {
                val += last_gap_[n][c] * fc1_weight_[r][c];
            }
            s1_local[r] = relu(val);
        }
        // Re-compute s2 and S
        double s2_local[1024];
        double S = 0.0;
        for (int c = 0; c < channels_; ++c) {
            double val = fc2_bias_[0][c];
            for (int r = 0; r < reduction_; ++r) {
                val += s1_local[r] * fc2_weight_[c][r];
            }
            s2_local[c] = relu(val);
            S += s2_local[c];
        }
        if (S < 1e-12) S = 1e-12;

        // grad_s2 = (grad_attention - dot) / S
        double dot = 0.0;
        for (int c = 0; c < channels_; ++c) {
            dot += grad_attention_total[n][c] * last_attention_[n][c];
        }
        for (int c = 0; c < channels_; ++c) {
            grad_s2[n][c] = (grad_attention_total[n][c] - dot) / S;
        }
    }

    // ===== Step 3: FC2 backward =====
    // s2 = ReLU(z2), z2 = s1 @ W2.T + b2
    // grad_z2 = grad_s2 * relu_deriv(z2)
    // grad_s1 = grad_z2 @ W2
    // grad_W2 = s1.T @ grad_z2
    // grad_b2 = sum(grad_z2)
    Tensor grad_z2(N, channels_);
    Tensor grad_s1(N, reduction_);
    grad_fc2_weight_.fill(0.0);
    grad_fc2_bias_.fill(0.0);

    for (int n = 0; n < N; ++n) {
        // Re-compute s1 and z2
        double s1_local[64];
        double z2_local[1024];
        for (int r = 0; r < reduction_; ++r) {
            double val = fc1_bias_[0][r];
            for (int c = 0; c < channels_; ++c) {
                val += last_gap_[n][c] * fc1_weight_[r][c];
            }
            s1_local[r] = relu(val);
        }
        for (int c = 0; c < channels_; ++c) {
            double val = fc2_bias_[0][c];
            for (int r = 0; r < reduction_; ++r) {
                val += s1_local[r] * fc2_weight_[c][r];
            }
            z2_local[c] = val;
            grad_z2[n][c] = grad_s2[n][c] * (val > 0.0 ? 1.0 : 0.0);
        }

        // grad_s1 = grad_z2 @ W2
        for (int r = 0; r < reduction_; ++r) {
            double gs1 = 0.0;
            for (int c = 0; c < channels_; ++c) {
                gs1 += grad_z2[n][c] * fc2_weight_[c][r];
            }
            grad_s1[n][r] = gs1;
        }

        // grad_W2 += outer(s1, grad_z2)
        for (int c = 0; c < channels_; ++c) {
            double gz = grad_z2[n][c];
            for (int r = 0; r < reduction_; ++r) {
                grad_fc2_weight_[c][r] += s1_local[r] * gz;
            }
            grad_fc2_bias_[0][c] += gz;
        }
    }

    // ===== Step 4: FC1 backward =====
    // s1 = ReLU(z1), z1 = gap @ W1.T + b1
    // grad_z1 = grad_s1 * relu_deriv(z1)
    // grad_gap = grad_z1 @ W1
    // grad_W1 = gap.T @ grad_z1
    Tensor grad_z1(N, reduction_);
    grad_fc1_weight_.fill(0.0);
    grad_fc1_bias_.fill(0.0);

    for (int n = 0; n < N; ++n) {
        double z1_local[64];
        for (int r = 0; r < reduction_; ++r) {
            double val = fc1_bias_[0][r];
            for (int c = 0; c < channels_; ++c) {
                val += last_gap_[n][c] * fc1_weight_[r][c];
            }
            z1_local[r] = val;
            grad_z1[n][r] = grad_s1[n][r] * (val > 0.0 ? 1.0 : 0.0);
        }

        // grad_W1: (reduction, C) += outer(grad_z1, gap)
        for (int r = 0; r < reduction_; ++r) {
            double gz = grad_z1[n][r];
            for (int c = 0; c < channels_; ++c) {
                grad_fc1_weight_[r][c] += last_gap_[n][c] * gz;
            }
            grad_fc1_bias_[0][r] += gz;
        }
    }

    // ===== Step 5: GAP backward =====
    // gap[c] = sum_s(dw_out[n][c*S+s]) / S
    // grad_dw_out += grad_gap[c] / S for all spatial positions in channel c
    Tensor grad_gap(N, channels_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double gg = 0.0;
            for (int r = 0; r < reduction_; ++r) {
                gg += grad_z1[n][r] * fc1_weight_[r][c];
            }
            grad_gap[n][c] = gg;
        }
    }

    Tensor grad_dw_out(N, channels_ * spatial_size_);
    grad_dw_out.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            double gg = grad_gap[n][c] / spatial_size_;
            int base = c * spatial_size_;
            for (int s = 0; s < spatial_size_; ++s) {
                grad_dw_out[n][base + s] += gg;
            }
        }
    }

    // Add the scale path gradient to dw_out
    for (int n = 0; n < N; ++n) {
        for (int s = 0; s < channels_ * spatial_size_; ++s) {
            grad_dw_out[n][s] += grad_dw_out_from_scale[n][s];
        }
    }

    // ===== Step 6: DepthwiseConv backward =====
    // Forward: out[h,w] = sum_kr,kc kernel[kr,kc] * input[h+kr, w+kc] with padding
    // grad_kernel[c][ki] += sum_{n,h,w} grad_dw_out[n][c][h,w] * input[n][c][h+kr, w+kc]
    // grad_input[n][c][hi,wi] += sum_{kr,kc} grad_dw_out[n][c][hi-kr, wi-kc] * kernel[kr,kc]
    grad_dw_weight_.fill(0.0);
    grad_dw_bias_.fill(0.0);
    Tensor grad_input_raw(N, channels_ * spatial_size_);
    grad_input_raw.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            int ch_base = c * spatial_size_;
            for (int h = 0; h < H_; ++h) {
                for (int w = 0; w < W_; ++w) {
                    double g = grad_dw_out[n][ch_base + h * W_ + w];
                    // grad bias
                    grad_dw_bias_[0][c] += g;
                    // grad weights and input
                    for (int kr = -1; kr <= 1; ++kr) {
                        for (int kc = -1; kc <= 1; ++kc) {
                            int ki = (kr + 1) * 3 + (kc + 1);
                            int src_h = h - kr;
                            int src_w = w - kc;
                            if (src_h >= 0 && src_h < H_ && src_w >= 0 && src_w < W_) {
                                // grad_kernel[ki] += g * input[src]
                                grad_dw_weight_[c][ki] += g * last_input_[n][ch_base + src_h * W_ + src_w];
                            }
                            int inp_h = h + kr;
                            int inp_w = w + kc;
                            if (inp_h >= 0 && inp_h < H_ && inp_w >= 0 && inp_w < W_) {
                                // grad_input[inp] += g * kernel[ki]
                                grad_input_raw[n][ch_base + inp_h * W_ + inp_w] +=
                                    g * dw_weight_[c][ki];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input_raw;
}

void CRATE::update_weights(double learning_rate) {
    // Depthwise conv
    for (int c = 0; c < channels_; ++c) {
        for (int k = 0; k < 9; ++k) {
            dw_weight_[c][k] -= learning_rate * grad_dw_weight_[c][k];
        }
        dw_bias_[0][c] -= learning_rate * grad_dw_bias_[0][c];
    }
    // FC1
    for (int r = 0; r < reduction_; ++r) {
        for (int c = 0; c < channels_; ++c) {
            fc1_weight_[r][c] -= learning_rate * grad_fc1_weight_[r][c];
        }
        fc1_bias_[0][r] -= learning_rate * grad_fc1_bias_[0][r];
    }
    // FC2
    for (int c = 0; c < channels_; ++c) {
        for (int r = 0; r < reduction_; ++r) {
            fc2_weight_[c][r] -= learning_rate * grad_fc2_weight_[c][r];
        }
        fc2_bias_[0][c] -= learning_rate * grad_fc2_bias_[0][c];
    }
}

void CRATE::zero_grad() {
    grad_dw_weight_.fill(0.0);
    grad_dw_bias_.fill(0.0);
    grad_fc1_weight_.fill(0.0);
    grad_fc1_bias_.fill(0.0);
    grad_fc2_weight_.fill(0.0);
    grad_fc2_bias_.fill(0.0);
}

std::vector<Tensor*> CRATE::parameters() {
    return {&dw_weight_, &dw_bias_,
            &fc1_weight_, &fc1_bias_,
            &fc2_weight_, &fc2_bias_};
}

std::vector<Tensor*> CRATE::gradients() {
    return {&grad_dw_weight_, &grad_dw_bias_,
            &grad_fc1_weight_, &grad_fc1_bias_,
            &grad_fc2_weight_, &grad_fc2_bias_};
}