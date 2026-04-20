#include "conv_layer.h"
#include <cmath>
#include <random>
#include <stdexcept>

Conv2D::Conv2D()
    : in_channels(0), out_channels(0),
      kernel_h(1), kernel_w(1),
      stride_h(1), stride_w(1),
      pad_h(0), pad_w(0),
      dilation_h(1), dilation_w(1),
      H(0), W(0), H_out(0), W_out(0)
{
    weights = Tensor(0, 0);
    bias = Tensor(0, 0);
    grad_weights = Tensor(0, 0);
    grad_bias = Tensor(0, 0);
}

Conv2D::Conv2D(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
               int stride_h, int stride_w, int pad_h, int pad_w,
               int dilation_h, int dilation_w)
    : in_channels(in_ch), out_channels(out_ch),
      kernel_h(kH), kernel_w(kW),
      stride_h(stride_h), stride_w(stride_w),
      pad_h(pad_h), pad_w(pad_w),
      dilation_h(dilation_h), dilation_w(dilation_w),
      H(H_in), W(W_in)
{
    // Compute output spatial dimensions (with dilation)
    H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("Conv2D: output dimensions non-positive");
    }

    // Initialize weights with Xavier/Glorot
    int fan_in = in_channels * kH * kW;
    int fan_out = out_channels * kH * kW;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);

    weights = Tensor(out_channels, in_channels * kH * kW);
    bias = Tensor(out_channels, 1);
    grad_weights = Tensor(out_channels, in_channels * kH * kW);
    grad_bias = Tensor(out_channels, 1);

    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kH * kW; ++i) {
            weights[o][i] = dis(gen);
        }
        bias[o][0] = 0.0;
    }
}

Tensor Conv2D::im2col(const Tensor& input, int N, int C, int H, int W,
                     int kH, int kW, int stride_h, int stride_w,
                     int pad_h, int pad_w, int dilation_h, int dilation_w,
                     int& H_out, int& W_out) {
    H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    Tensor col(C * kH * kW, N * H_out * W_out);
    col.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            for (int j_out = 0; j_out < W_out; ++j_out) {
                int col_idx = n * H_out * W_out + i_out * W_out + j_out;
                for (int c = 0; c < C; ++c) {
                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h = i_out * stride_h + i * dilation_h - pad_h;
                            int w = j_out * stride_w + j * dilation_w - pad_w;
                            double val = 0.0;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                val = input[n][c * H * W + h * W + w];
                            }
                            int row_idx = c * kH * kW + i * kW + j;
                            col[row_idx][col_idx] = val;
                        }
                    }
                }
            }
        }
    }
    return col;
}

void Conv2D::col2im(Tensor& grad_input, const Tensor& grad_col,
                    int N, int C, int H, int W,
                    int kH, int kW, int stride_h, int stride_w,
                    int pad_h, int pad_w, int dilation_h, int dilation_w) {
    int H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    // grad_input assumed zeroed
    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            for (int j_out = 0; j_out < W_out; ++j_out) {
                int col_idx = n * H_out * W_out + i_out * W_out + j_out;
                for (int c = 0; c < C; ++c) {
                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h = i_out * stride_h + i * dilation_h - pad_h;
                            int w = j_out * stride_w + j * dilation_w - pad_w;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int row_idx = c * kH * kW + i * kW + j;
                                grad_input[n][c * H * W + h * W + w] += grad_col[row_idx][col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != in_channels * H * W) {
        throw std::invalid_argument("Conv2D: input dimension mismatch");
    }

    // im2col
    col = im2col(input, N, in_channels, H, W,
                 kernel_h, kernel_w, stride_h, stride_w,
                 pad_h, pad_w, dilation_h, dilation_w,
                 H_out, W_out);
    last_input = input;

    int out_spatial = H_out * W_out;
    // Z = weights * col
    Tensor Z = weights * col; // shape (out_channels, N * out_spatial)

    // Add bias
    for (int o = 0; o < out_channels; ++o) {
        for (int idx = 0; idx < N * out_spatial; ++idx) {
            Z[o][idx] += bias[o][0];
        }
    }

    // Reshape to (N, out_channels * out_spatial)
    Tensor output(N, out_channels * out_spatial);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int s = 0; s < out_spatial; ++s) {
                output[n][o * out_spatial + s] = Z[o][n * out_spatial + s];
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;
    int out_spatial = H_out * W_out;

    // Reshape grad_output to matrix (out_channels, N*out_spatial)
    Tensor grad_out_mat(out_channels, N * out_spatial);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int s = 0; s < out_spatial; ++s) {
                grad_out_mat[o][n * out_spatial + s] = grad_output[n][o * out_spatial + s];
            }
        }
    }

    // Gradient w.r.t. weights: dW = grad_out_mat * col^T
    // Accumulate into grad_weights (handles multiple backward calls before zero_grad)
    Tensor col_T = col.transpose(); // (N*out_spatial, in_channels*kH*kW)
    Tensor dW = grad_out_mat * col_T; // (out_channels, in_channels*kH*kW)
    grad_weights = grad_weights + dW;

    // Gradient w.r.t. bias: sum over all spatial positions and batch
    Tensor db(out_channels, 1);
    for (int o = 0; o < out_channels; ++o) {
        double sum = 0.0;
        for (int i = 0; i < N * out_spatial; ++i) {
            sum += grad_out_mat[o][i];
        }
        db[o][0] = sum;
    }
    grad_bias = grad_bias + db;

    // Gradient w.r.t. input: dX_col = weights^T * grad_out_mat
    Tensor weights_T = weights.transpose(); // (in_channels*kH*kW, out_channels)
    Tensor dX_col = weights_T * grad_out_mat; // (in_channels*kH*kW, N*out_spatial)

    // Convert dX_col to grad_input using col2im
    Tensor grad_input(N, in_channels * H * W);
    grad_input.fill(0.0);
    col2im(grad_input, dX_col, N, in_channels, H, W,
           kernel_h, kernel_w, stride_h, stride_w,
           pad_h, pad_w, dilation_h, dilation_w);

    return grad_input;
}

void Conv2D::update_weights(double learning_rate) {
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kernel_h * kernel_w; ++i) {
            weights[o][i] -= learning_rate * grad_weights[o][i];
        }
        bias[o][0] -= learning_rate * grad_bias[o][0];
    }
    grad_weights.fill(0.0);  // Reset for next accumulation cycle
}

std::vector<Tensor*> Conv2D::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Conv2D::gradients() {
    return {&grad_weights, &grad_bias};
}

void Conv2D::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

// END Conv2D
