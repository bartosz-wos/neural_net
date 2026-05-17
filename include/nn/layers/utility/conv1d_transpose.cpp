#include "conv1d_transpose.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

// =============================================================================
// TransposedConv1D
// =============================================================================
TransposedConv1D::TransposedConv1D(int in_ch, int out_ch, int kernel_size, int seq_in,
                                  int stride, int pad, int output_pad)
    : in_channels(in_ch), out_channels(out_ch),
      kernel_size(kernel_size), stride(stride), pad(pad),
      output_pad(output_pad), seq_in(seq_in)
{
    int sout = seq_out();
    if (sout <= 0) {
        throw std::invalid_argument("TransposedConv1D: computed output seq_len non-positive");
    }

    int fan_in  = out_channels * kernel_size;
    int fan_out = in_channels  * sout;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);

    // weight layout: (in_channels, out_channels * kernel_size)
    // This is the transpose of the standard Conv1D weight layout
    weights      = Tensor(in_channels, out_channels * kernel_size);
    bias         = Tensor(out_channels, 1);
    grad_weights = Tensor(in_channels, out_channels * kernel_size);
    grad_bias    = Tensor(out_channels, 1);

    for (int i = 0; i < in_channels; ++i) {
        for (int j = 0; j < out_channels * kernel_size; ++j) {
            weights[i][j] = dis(gen);
        }
        bias[i][0] = 0.0;
    }
}

int TransposedConv1D::seq_out() const {
    return (seq_in - 1) * stride - 2 * pad + kernel_size + output_pad;
}

Tensor TransposedConv1D::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != (size_t)in_channels * seq_in) {
        throw std::invalid_argument("TransposedConv1D: input dimension mismatch");
    }

    int sout = seq_out();
    last_input = input;

    // For each output timestep, we do a weighted sum over input timesteps
    // This is equivalent to: each input channel contributes to multiple output positions
    // Implementation: for each (n, c_in), spread over output timesteps via kernel
    // Then sum contributions from all input channels to produce out_channels outputs

    // col2im style: we'll compute the output directly
    // Output: (N, out_channels * sout)
    Tensor output(N, out_channels * sout);
    output.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int t_out = 0; t_out < sout; ++t_out) {
                double sum = 0.0;
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int k = 0; k < kernel_size; ++k) {
                        // Reverse computation: input timestep that this output depends on
                        int t_in = t_out * stride + k - pad;
                        if (t_in >= 0 && t_in < seq_in) {
                            int col_idx  = c_in * kernel_size + k;
                            int weight_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
                            double w = weights.data[weight_idx];
                            double x = input[n][c_in * seq_in + t_in];
                            sum += w * x;
                        }
                    }
                }
                output[n][c_out * sout + t_out] = sum + bias[c_out][0];
            }
        }
    }

    return output;
}

Tensor TransposedConv1D::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;
    int sout = seq_out();

    // grad_w: accumulate dW from output gradients
    // For transposed conv: dW = sum over output positions of d(out) * input contribution
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            // accumulate bias gradient
            for (int t_out = 0; t_out < sout; ++t_out) {
                grad_bias[c_out][0] += grad_output[n][c_out * sout + t_out];
            }
        }
    }

    // dW: each weight connects output(c_out, t_out) to input(c_in, t_in)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int t_out = 0; t_out < sout; ++t_out) {
                double go = grad_output[n][c_out * sout + t_out];
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int t_in = t_out * stride + k - pad;
                        if (t_in >= 0 && t_in < seq_in) {
                            int w_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
                            double x = last_input[n][c_in * seq_in + t_in];
                            grad_weights.data[w_idx] += go * x;
                        }
                    }
                }
            }
        }
    }

    // dX: gradient w.r.t. input — same as forward but in reverse
    Tensor grad_input(N, in_channels * seq_in);
    grad_input.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int t_in = 0; t_in < seq_in; ++t_in) {
                double sum = 0.0;
                for (int c_out = 0; c_out < out_channels; ++c_out) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int t_out = t_in + pad - k;
                        if (t_out >= 0 && t_out < sout && (t_out * stride + k - pad) == t_in) {
                            int w_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
                            double w = weights.data[w_idx];
                            double go = grad_output[n][c_out * sout + t_out];
                            sum += w * go;
                        }
                    }
                }
                grad_input[n][c_in * seq_in + t_in] = sum;
            }
        }
    }

    (void)learning_rate;
    return grad_input;
}

void TransposedConv1D::update_weights(double learning_rate) {
    for (int i = 0; i < in_channels; ++i) {
        for (int j = 0; j < out_channels * kernel_size; ++j) {
            weights.data[i * out_channels * kernel_size + j] -= learning_rate * grad_weights.data[i * out_channels * kernel_size + j];
        }
        bias[i][0] -= learning_rate * grad_bias[i][0];
    }
}

std::vector<Tensor*> TransposedConv1D::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> TransposedConv1D::gradients() {
    return {&grad_weights, &grad_bias};
}

void TransposedConv1D::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

// =============================================================================
// Upsample1D — nearest-neighbor 2x upsampling
// =============================================================================
Upsample1D::Upsample1D(int channels, int seq_in)
    : channels_(channels), seq_in_(seq_in), seq_out_(seq_in * 2) {
    std::cerr << "[Upsample1D CTOR] channels=" << channels << ", seq_in=" << seq_in << std::endl;
}

Tensor Upsample1D::forward(const Tensor& input) {
    int N = input.rows;
    std::cerr << "[Upsample1D] input.cols=" << input.cols << ", expected=" << (channels_ * seq_in_) << std::endl;
    if (input.cols != (size_t)channels_ * seq_in_) {
        throw std::invalid_argument("Upsample1D: input dimension mismatch");
    }
    last_input_ = input;

    Tensor output(N, channels_ * seq_out_);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            for (int t = 0; t < seq_in_; ++t) {
                double val = input[n][c * seq_in_ + t];
                output[n][c * seq_out_ + 2 * t]     = val;
                output[n][c * seq_out_ + 2 * t + 1] = val;
            }
        }
    }
    return output;
}

Tensor Upsample1D::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;
    (void)learning_rate;
    Tensor grad_input(N, channels_ * seq_in_);
    grad_input.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels_; ++c) {
            for (int t = 0; t < seq_in_; ++t) {
                // Both duplicated positions contribute to same input gradient
                grad_input[n][c * seq_in_ + t] =
                    grad_output[n][c * seq_out_ + 2 * t] +
                    grad_output[n][c * seq_out_ + 2 * t + 1];
            }
        }
    }
    return grad_input;
}