#include "conv1d.h"
#include <cmath>
#include <random>
#include <stdexcept>

Conv1D::Conv1D(int in_ch, int out_ch, int ksz, int seq_len_in,
               int stride, int pad)
    : in_channels(in_ch), out_channels(out_ch),
      kernel_size(ksz), stride(stride), pad(pad), seq_len(seq_len_in)
{
    seq_out = (seq_len + 2 * pad - ksz) / stride + 1;
    if (seq_out <= 0) {
        throw std::invalid_argument("Conv1D: output seq_len non-positive");
    }

    int fan_in = in_channels * ksz;
    int fan_out = out_channels * seq_out;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);

    weights = Tensor(out_channels, in_channels * ksz);
    bias = Tensor(out_channels, 1);
    grad_weights = Tensor(out_channels, in_channels * ksz);
    grad_bias = Tensor(out_channels, 1);

    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * ksz; ++i) {
            weights[o][i] = dis(gen);
        }
        bias[o][0] = 0.0;
    }
}

Tensor Conv1D::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != (size_t)in_channels * seq_len) {
        throw std::invalid_argument("Conv1D: input dimension mismatch");
    }

    // im2col: each output timestep gets a column of kernel samples
    // col shape: (in_channels * kernel_size, N * seq_out)
    col = Tensor(in_channels * kernel_size, N * seq_out);
    col.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int t_out = 0; t_out < seq_out; ++t_out) {
            int col_idx = n * seq_out + t_out;
            for (int c = 0; c < in_channels; ++c) {
                for (int k = 0; k < kernel_size; ++k) {
                    int t = t_out * stride + k - pad;
                    double val = 0.0;
                    if (t >= 0 && t < seq_len) {
                        val = input[n][c * seq_len + t];
                    }
                    int row_idx = c * kernel_size + k;
                    col[row_idx][col_idx] = val;
                }
            }
        }
    }
    last_input = input;

    // matmul: weights (out_channels, in_channels*ksz) * col (in_channels*ksz, N*seq_out)
    Tensor Z = weights * col; // (out_channels, N * seq_out)

    // add bias: broadcast bias[o][0] to all seq_out positions per batch
    for (int o = 0; o < out_channels; ++o) {
        for (int idx = 0; idx < N * seq_out; ++idx) {
            Z[o][idx] += bias[o][0];
        }
    }

    // reshape to (N, out_channels * seq_out)
    Tensor output(N, out_channels * seq_out);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int t = 0; t < seq_out; ++t) {
                output[n][o * seq_out + t] = Z[o][n * seq_out + t];
            }
        }
    }
    return output;
}

Tensor Conv1D::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;

    // reshape grad_output to (out_channels, N*seq_out)
    Tensor grad_out_mat(out_channels, N * seq_out);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int t = 0; t < seq_out; ++t) {
                grad_out_mat[o][n * seq_out + t] = grad_output[n][o * seq_out + t];
            }
        }
    }

    // gradient w.r.t. weights: dW = grad_out_mat * col^T
    Tensor col_T = col.transpose();
    Tensor dW = grad_out_mat * col_T; // (out_channels, in_channels*ksz)
    grad_weights = grad_weights + dW;

    // gradient w.r.t. bias: sum over all positions
    Tensor db(out_channels, 1);
    for (int o = 0; o < out_channels; ++o) {
        double sum = 0.0;
        for (int i = 0; i < N * seq_out; ++i) {
            sum += grad_out_mat[o][i];
        }
        db[o][0] = sum;
    }
    grad_bias = grad_bias + db;

    // gradient w.r.t. input: dX_col = weights^T * grad_out_mat
    Tensor weights_T = weights.transpose();
    Tensor dX_col = weights_T * grad_out_mat; // (in_channels*ksz, N*seq_out)

    // col2im: accumulate gradients back to input
    Tensor grad_input(N, in_channels * seq_len);
    grad_input.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int t_out = 0; t_out < seq_out; ++t_out) {
            int col_idx = n * seq_out + t_out;
            for (int c = 0; c < in_channels; ++c) {
                for (int k = 0; k < kernel_size; ++k) {
                    int t = t_out * stride + k - pad;
                    if (t >= 0 && t < seq_len) {
                        int row_idx = c * kernel_size + k;
                        grad_input[n][c * seq_len + t] += dX_col[row_idx][col_idx];
                    }
                }
            }
        }
    }

    (void)learning_rate; // weight update done via step() call
    return grad_input;
}

void Conv1D::update_weights(double learning_rate) {
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kernel_size; ++i) {
            weights[o][i] -= learning_rate * grad_weights[o][i];
        }
        bias[o][0] -= learning_rate * grad_bias[o][0];
    }
}

std::vector<Tensor*> Conv1D::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Conv1D::gradients() {
    return {&grad_weights, &grad_bias};
}

void Conv1D::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}
