#include "pool_layer.h"
#include <stdexcept>
#include <limits>

MaxPool2D::MaxPool2D(int kH, int kW, int H_in, int W_in, int stride_h, int stride_w)
    : kernel_h(kH), kernel_w(kW), stride_h(stride_h), stride_w(stride_w),
      H(H_in), W(W_in)
{
    H_out = (H - kH) / stride_h + 1;
    W_out = (W - kW) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("MaxPool2D: output dimensions non-positive");
    }
}

Tensor MaxPool2D::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols % (H * W) != 0) {
        throw std::invalid_argument("MaxPool2D: input cols not divisible by H*W");
    }
    int C = input.cols / (H * W);

    Tensor output(N, C * H_out * W_out);
    last_input = input; // cache

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i_out = 0; i_out < H_out; ++i_out) {
                for (int j_out = 0; j_out < W_out; ++j_out) {
                    int h_start = i_out * stride_h;
                    int w_start = j_out * stride_w;
                    double max_val = -std::numeric_limits<double>::infinity();
                    for (int i = 0; i < kernel_h; ++i) {
                        for (int j = 0; j < kernel_w; ++j) {
                            int h = h_start + i;
                            int w = w_start + j;
                            if (h < H && w < W) {
                                double val = input[n][c * H * W + h * W + w];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    output[n][c * H_out * W_out + i_out * W_out + j_out] = max_val;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    int C = last_input.cols / (H * W);
    Tensor grad_input(N, C * H * W);
    grad_input.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i_out = 0; i_out < H_out; ++i_out) {
                for (int j_out = 0; j_out < W_out; ++j_out) {
                    int h_start = i_out * stride_h;
                    int w_start = j_out * stride_w;
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_h = -1, max_w = -1;
                    // recompute max position
                    for (int i = 0; i < kernel_h; ++i) {
                        for (int j = 0; j < kernel_w; ++j) {
                            int h = h_start + i;
                            int w = w_start + j;
                            if (h < H && w < W) {
                                double val = last_input[n][c * H * W + h * W + w];
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = h;
                                    max_w = w;
                                }
                            }
                        }
                    }
                    if (max_h >= 0) {
                        grad_input[n][c * H * W + max_h * W + max_w] += grad_output[n][c * H_out * W_out + i_out * W_out + j_out];
                    }
                }
            }
        }
    }
    return grad_input;
}

// MaxPool1D
MaxPool1D::MaxPool1D(int ksz, int seq_len, int ch, int stride)
    : kernel_size(ksz), stride(stride), channels(ch), seq_len(seq_len)
{
    seq_out = (seq_len + ksz - 1) / stride; // simplified: assumes pad=0
}

Tensor MaxPool1D::forward(const Tensor& input) {
    int N = input.rows;
    last_input = input;
    max_indices_.assign(channels * N, std::vector<int>(seq_out, -1));

    Tensor output(N, channels * seq_out);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int t_out = 0; t_out < seq_out; ++t_out) {
                double max_val = -1e100;
                int max_idx = -1;
                for (int k = 0; k < kernel_size; ++k) {
                    int t = t_out * stride + k;
                    if (t < seq_len) {
                        double val = input[n][c * seq_len + t];
                        if (val > max_val) { max_val = val; max_idx = t; }
                    }
                }
                int idx_2d = c * N + n;
                output[n][c * seq_out + t_out] = max_val;
                max_indices_[idx_2d][t_out] = max_idx;
            }
        }
    }
    return output;
}

Tensor MaxPool1D::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    Tensor grad_input(N, channels * seq_len);
    grad_input.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int t_out = 0; t_out < seq_out; ++t_out) {
                int idx_2d = c * N + n;
                int max_idx = max_indices_[idx_2d][t_out];
                if (max_idx >= 0) {
                    grad_input[n][c * seq_len + max_idx] += grad_output[n][c * seq_out + t_out];
                }
            }
        }
    }
    return grad_input;
}

// AvgPool1D
AvgPool1D::AvgPool1D(int ksz, int seq_len, int ch, int stride)
    : kernel_size(ksz), stride(stride), channels(ch), seq_len(seq_len)
{
    seq_out = (seq_len + ksz - 1) / stride;
}

Tensor AvgPool1D::forward(const Tensor& input) {
    int N = input.rows;
    last_input = input;
    Tensor output(N, channels * seq_out);
    double norm = 1.0 / kernel_size;
    counts_.assign(channels * N, std::vector<int>(seq_out, 1));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int t_out = 0; t_out < seq_out; ++t_out) {
                double sum = 0.0;
                int count = 0;
                for (int k = 0; k < kernel_size; ++k) {
                    int t = t_out * stride + k;
                    if (t < seq_len) {
                        sum += input[n][c * seq_len + t];
                        ++count;
                    }
                }
                int idx_2d = c * N + n;
                output[n][c * seq_out + t_out] = sum / (count > 0 ? count : 1);
                counts_[idx_2d][t_out] = count;
            }
        }
    }
    return output;
}

Tensor AvgPool1D::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    Tensor grad_input(N, channels * seq_len);
    grad_input.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int t_out = 0; t_out < seq_out; ++t_out) {
                double grad_val = grad_output[n][c * seq_out + t_out];
                int idx_2d = c * N + n;
                int count = counts_[idx_2d][t_out];
                for (int k = 0; k < kernel_size; ++k) {
                    int t = t_out * stride + k;
                    if (t < seq_len) {
                        grad_input[n][c * seq_len + t] += grad_val / static_cast<double>(count);
                    }
                }
            }
        }
    }
    return grad_input;
}
