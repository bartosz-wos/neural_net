// Spatial Transformer Network (STN) — Jaderberg et al. 2015
// "Spatial Transformer Networks" (https://arxiv.org/abs/1506.02025).
//
// See spatial_transformer.h for the full math.

#include "spatial_transformer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ============================================================================
// Construction
// ============================================================================

SpatialTransformer::SpatialTransformer(int batch, int channels, int H_in, int W_in)
    : N(batch), C(channels), H(H_in), W(W_in),
      H_out(H_in), W_out(W_in),
      use_fixed_theta_(false) {
    if (N <= 0 || C <= 0 || H_in <= 0 || W_in <= 0) {
        throw std::invalid_argument("SpatialTransformer: dims must be positive");
    }

    // Localization conv: 8 filters, 3x3 kernel, same padding -> H_pool × W_pool
    // H_pool = H/2 (2x2 stride-2 max pool). For H=4, H_pool=2.
    const int kH = 3, kW = 3;
    const int loc_conv_out = 8;
    const int H_pool = H_in / 2;
    const int W_pool = W_in / 2;

    loc_conv_W = Tensor::random(loc_conv_out, channels * kH * kW, 0.1);
    loc_conv_b = Tensor(loc_conv_out, 1);
    for (int i = 0; i < loc_conv_out; ++i) loc_conv_b(i, 0) = 0.0;

    loc_dense1_W = Tensor::random(32, loc_conv_out * H_pool * W_pool, 0.1);
    loc_dense1_b = Tensor(1, 32);
    for (int i = 0; i < 32; ++i) loc_dense1_b(0, i) = 0.0;

    loc_dense2_W = Tensor::random(6, 32, 0.1);
    loc_dense2_b = Tensor(1, 6);

    // Initialize loc_dense2_W so that the initial theta is close to identity.
    // theta = relu(loc_dense2_W * relu(loc_dense1_W * pool(conv(x)) + b1) + b2)
    // We want theta ≈ [[1, 0, 0], [0, 1, 0]].
    // Heuristic: bias b2 = [1, 0, 0, 0, 1, 0] (so a positive bias drives
    // theta[0,0] and theta[1,1] toward 1 via relu), and shrink W so the
    // pre-relu activations are small (so other entries ≈ 0 after relu).
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 32; ++j) loc_dense2_W(i, j) *= 0.05;
    }
    loc_dense2_b(0, 0) = 1.0;  // theta[0,0]
    loc_dense2_b(0, 2) = 0.0;  // theta[0,2]
    loc_dense2_b(0, 4) = 1.0;  // theta[1,1]
    loc_dense2_b(0, 5) = 0.0;  // theta[1,2] (unused)

    // Allocate gradients
    grad_loc_conv_W   = Tensor::zeros(loc_conv_out, channels * kH * kW);
    grad_loc_conv_b   = Tensor::zeros(loc_conv_out, 1);
    grad_loc_dense1_W = Tensor::zeros(32, loc_conv_out * H_pool * W_pool);
    grad_loc_dense1_b = Tensor::zeros(1, 32);
    grad_loc_dense2_W = Tensor::zeros(6, 32);
    grad_loc_dense2_b = Tensor::zeros(1, 6);

    fixed_theta_ = Tensor::zeros(2, 3);
}

// ============================================================================
// Parameter / gradient plumbing
// ============================================================================

std::vector<Tensor*> SpatialTransformer::parameters() {
    return {&loc_conv_W, &loc_conv_b, &loc_dense1_W, &loc_dense1_b,
            &loc_dense2_W, &loc_dense2_b};
}

std::vector<Tensor*> SpatialTransformer::gradients() {
    return {&grad_loc_conv_W, &grad_loc_conv_b, &grad_loc_dense1_W, &grad_loc_dense1_b,
            &grad_loc_dense2_W, &grad_loc_dense2_b};
}

void SpatialTransformer::zero_grad() {
    for (auto* g : gradients()) {
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j)
                (*g)(i, j) = 0.0;
    }
}

void SpatialTransformer::update_weights(double learning_rate) {
    // Note: in this minimal implementation we use a fixed learning rate
    // step on each parameter. The caller should have zeroed grads after.
    auto params = parameters();
    auto grads = gradients();
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t r = 0; r < params[i]->rows; ++r) {
            for (size_t c = 0; c < params[i]->cols; ++c) {
                (*params[i])(r, c) -= learning_rate * (*grads[i])(r, c);
            }
        }
    }
}

void SpatialTransformer::set_theta(const Tensor& theta) {
    if (theta.rows != 2 || theta.cols != 3) {
        throw std::invalid_argument("SpatialTransformer::set_theta: theta must be (2, 3)");
    }
    fixed_theta_ = theta;
    use_fixed_theta_ = true;
}

// ============================================================================
// Helpers
// ============================================================================

Tensor SpatialTransformer::conv_im2col(const Tensor& input, int in_ch, int kH, int kW) {
    // input: (N, in_ch * H * W) channels-first within each sample
    // output: (in_ch * kH * kW, N * H * W) im2col matrix
    const int H_in = H, W_in = W;
    const int pad_h = (kH - 1) / 2;
    const int pad_w = (kW - 1) / 2;
    (void)pad_h; (void)pad_w;
    Tensor col(in_ch * kH * kW, N * H_in * W_in);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_ch; ++c) {
            for (int ki = 0; ki < kH; ++ki) {
                for (int kj = 0; kj < kW; ++kj) {
                    int row = c * kH * kW + ki * kW + kj;
                    for (int i = 0; i < H_in; ++i) {
                        for (int j = 0; j < W_in; ++j) {
                            int src_i = i + ki - pad_h;
                            int src_j = j + kj - pad_w;
                            double v = 0.0;
                            if (src_i >= 0 && src_i < H_in && src_j >= 0 && src_j < W_in) {
                                v = input(n, c * H_in * W_in + src_i * W_in + src_j);
                            }
                            int col_idx = n * H_in * W_in + i * W_in + j;
                            col(row, col_idx) = v;
                        }
                    }
                }
            }
        }
    }
    return col;
}

void SpatialTransformer::conv_col2im(Tensor& grad_input, const Tensor& grad_col,
                                       int in_ch, int kH, int kW) {
    const int H_in = H, W_in = W;
    const int pad_h = (kH - 1) / 2;
    const int pad_w = (kW - 1) / 2;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_ch; ++c) {
            for (int ki = 0; ki < kH; ++ki) {
                for (int kj = 0; kj < kW; ++kj) {
                    int row = c * kH * kW + ki * kW + kj;
                    for (int i = 0; i < H_in; ++i) {
                        for (int j = 0; j < W_in; ++j) {
                            int src_i = i + ki - pad_h;
                            int src_j = j + kj - pad_w;
                            if (src_i >= 0 && src_i < H_in && src_j >= 0 && src_j < W_in) {
                                int col_idx = n * H_in * W_in + i * W_in + j;
                                grad_input(n, c * H_in * W_in + src_i * W_in + src_j) +=
                                    grad_col(row, col_idx);
                            }
                        }
                    }
                }
            }
        }
    }
}

Tensor SpatialTransformer::conv2d_forward(const Tensor& input, const Tensor& W_param, const Tensor& b,
                                            int in_ch, int out_ch, int kH, int kW) {
    // W_param shape: (out_ch, in_ch * kH * kW) — Dense-style
    // input: (N, in_ch * H * W)
    // output: (N, out_ch * H * W)
    Tensor col = conv_im2col(input, in_ch, kH, kW);  // (in_ch*kH*kW, N*H*W)
    // out = W_param @ col — out is (out_ch, N*H*W), reshape to (N, out_ch*H*W)
    Tensor out_mat = W_param * col;  // (out_ch, N*H*W)
    // Add bias per channel and reshape to (N, out_ch * H * W) channels-first per sample
    Tensor out(N, (size_t)out_ch * H * W);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int p = 0; p < H * W; ++p) {
                int row = n * H * W + p;
                out(n, oc * H * W + p) = out_mat(oc, row) + b(oc, 0);
            }
        }
    }
    return out;
}

Tensor SpatialTransformer::conv2d_backward(const Tensor& grad_output, const Tensor& W_param,
                                            int in_ch, int out_ch, int kH, int kW,
                                            Tensor& dW, Tensor& db) {
    // grad_output: (N, out_ch * H * W)
    // W_param: (out_ch, in_ch * kH * kW)
    // Returns grad_input: (N, in_ch * H * W); updates dW, db
    Tensor grad_col_mat(W_param.cols, (size_t)N * H * W);  // (in_ch*kH*kW, N*H*W)
    // grad_col = W_param^T @ grad_output_reshaped
    // First reshape grad_output to (out_ch, N*H*W)
    Tensor grad_out_mat((size_t)out_ch, (size_t)N * H * W);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int p = 0; p < (int)(H * W); ++p) {
                int row = n * H * W + p;
                grad_out_mat(oc, row) = grad_output(n, oc * H * W + p);
            }
        }
    }
    Tensor Wt = W_param.transpose();  // (in_ch*kH*kW, out_ch)
    grad_col_mat = Wt * grad_out_mat;

    // dW += grad_out_mat @ col^T  -> (out_ch, in_ch*kH*kW)
    Tensor col = conv_im2col(last_input, in_ch, kH, kW);
    Tensor colt = col.transpose();  // (N*H*W, in_ch*kH*kW)
    Tensor dW_inc = grad_out_mat * colt;  // (out_ch, in_ch*kH*kW)
    for (size_t i = 0; i < dW.rows; ++i)
        for (size_t j = 0; j < dW.cols; ++j)
            dW(i, j) += dW_inc(i, j);

    // db += sum over batch and spatial
    for (int oc = 0; oc < out_ch; ++oc) {
        double s = 0.0;
        for (int n = 0; n < N; ++n)
            for (int p = 0; p < (int)(H * W); ++p)
                s += grad_output(n, oc * H * W + p);
        db(oc, 0) += s;
    }

    // d_input via col2im
    Tensor d_input(N, (size_t)in_ch * H * W);
    conv_col2im(d_input, grad_col_mat, in_ch, kH, kW);
    return d_input;
}

Tensor SpatialTransformer::maxpool2d_forward(const Tensor& input) const {
    // 2x2 stride 2 max pool. Input: (N, C*H*W). Output: (N, C*H/2*W/2).
    const int Hp = H / 2, Wp = W / 2;
    Tensor out(N, C * Hp * Wp);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < Hp; ++i) {
                for (int j = 0; j < Wp; ++j) {
                    int i0 = i * 2, j0 = j * 2;
                    double mx = input(n, c * H * W + i0 * W + j0);
                    mx = std::max(mx, input(n, c * H * W + i0 * W + (j0 + 1)));
                    mx = std::max(mx, input(n, c * H * W + (i0 + 1) * W + j0));
                    mx = std::max(mx, input(n, c * H * W + (i0 + 1) * W + (j0 + 1)));
                    out(n, c * Hp * Wp + i * Wp + j) = mx;
                }
            }
        }
    }
    return out;
}

Tensor SpatialTransformer::maxpool2d_backward(const Tensor& grad_output, int Hp, int Wp) {
    // grad_output: (N, C * Hp * Wp). Returns d_input: (N, C * H * W).
    Tensor d_input(N, C * H * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < Hp; ++i) {
                for (int j = 0; j < Wp; ++j) {
                    int i0 = i * 2, j0 = j * 2;
                    // Find argmax of the 2x2 window
                    double best = last_input(n, c * H * W + i0 * W + j0);
                    int best_r = i0, best_c = j0;
                    for (int di = 0; di < 2; ++di)
                        for (int dj = 0; dj < 2; ++dj) {
                            int ri = i0 + di, cj = j0 + dj;
                            double vv = last_input(n, c * H * W + ri * W + cj);
                            if (vv > best) { best = vv; best_r = ri; best_c = cj; }
                        }
                    d_input(n, c * H * W + best_r * W + best_c) +=
                        grad_output(n, c * Hp * Wp + i * Wp + j);
                }
            }
        }
    }
    return d_input;
}

Tensor SpatialTransformer::dense_forward(const Tensor& input, const Tensor& W, const Tensor& b) const {
    // input: (N, in_dim), W: (out_dim, in_dim), b: (1, out_dim)
    // output: (N, out_dim)  — where out(n, j) = sum_k input(n, k) * W(j, k) + b(0, j)
    Tensor out(input.rows, W.rows);
    for (size_t n = 0; n < input.rows; ++n) {
        for (size_t j = 0; j < W.rows; ++j) {
            double s = b(0, j);
            for (size_t k = 0; k < input.cols; ++k) {
                s += input(n, k) * W(j, k);
            }
            out(n, j) = s;
        }
    }
    return out;
}

Tensor SpatialTransformer::dense_backward(const Tensor& grad_output, const Tensor& W,
                                           Tensor& dW, Tensor& db) {
    // grad_output: (N, out_dim)
    // W: (out_dim, in_dim)
    // Returns d_input: (N, in_dim)
    Tensor d_input(last_input.rows, W.cols);
    // d_input(n, k) = sum_j grad_output(n, j) * W(j, k)
    for (size_t n = 0; n < grad_output.rows; ++n) {
        for (size_t k = 0; k < W.cols; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < W.rows; ++j) {
                s += grad_output(n, j) * W(j, k);
            }
            d_input(n, k) = s;
        }
    }
    // dW(j, k) += sum_n grad_output(n, j) * last_input(n, k)
    for (size_t n = 0; n < grad_output.rows; ++n) {
        for (size_t j = 0; j < W.rows; ++j) {
            for (size_t k = 0; k < W.cols; ++k) {
                dW(j, k) += grad_output(n, j) * last_input(n, k);
            }
        }
    }
    // db(0, j) += sum_n grad_output(n, j)
    for (size_t j = 0; j < W.rows; ++j) {
        double s = 0.0;
        for (size_t n = 0; n < grad_output.rows; ++n) s += grad_output(n, j);
        db(0, j) += s;
    }
    return d_input;
}

Tensor SpatialTransformer::relu_forward(const Tensor& input) const {
    Tensor out(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            out(i, j) = std::max(0.0, input(i, j));
    return out;
}

Tensor SpatialTransformer::relu_backward(const Tensor& grad_output, const Tensor& input) const {
    Tensor out(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i)
        for (size_t j = 0; j < grad_output.cols; ++j)
            out(i, j) = input(i, j) > 0.0 ? grad_output(i, j) : 0.0;
    return out;
}

// ============================================================================
// Localization network
// ============================================================================

Tensor SpatialTransformer::localization_forward(const Tensor& input) {
    // input: (N, C*H*W)
    // output: theta (2, 3)
    Tensor h = conv2d_forward(input, loc_conv_W, loc_conv_b, C, 8, 3, 3);
    h = maxpool2d_forward(h);  // (N, 8 * Hp * Wp)
    Tensor h_relu = relu_forward(h);
    Tensor h_dense1 = dense_forward(h_relu, loc_dense1_W, loc_dense1_b);  // (N, 32)
    Tensor h_dense1_relu = relu_forward(h_dense1);
    Tensor h_dense2 = dense_forward(h_dense1_relu, loc_dense2_W, loc_dense2_b);  // (N, 6)
    Tensor h_dense2_relu = relu_forward(h_dense2);
    // Average over batch and reshape to (2, 3)
    Tensor theta(2, 3);
    for (int n = 0; n < N; ++n) {
        theta(0, 0) += h_dense2_relu(n, 0);
        theta(0, 1) += h_dense2_relu(n, 1);
        theta(0, 2) += h_dense2_relu(n, 2);
        theta(1, 0) += h_dense2_relu(n, 3);
        theta(1, 1) += h_dense2_relu(n, 4);
        theta(1, 2) += h_dense2_relu(n, 5);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) theta(i, j) /= N;
    return theta;
}

Tensor SpatialTransformer::localization_backward(const Tensor& d_theta) {
    // d_theta: (2, 3). We need to propagate gradients through the loc net.
    // We re-compute the forward intermediates here so the backward has access
    // to the pre-relu activations.

    // 1. d_theta -> d_h_dense2_relu (broadcast across batch): each of the 6
    //    entries gets the same gradient (since we averaged over batch in fwd).
    Tensor d_h_dense2_relu(N, 6);
    for (int n = 0; n < N; ++n) {
        d_h_dense2_relu(n, 0) = d_theta(0, 0) / N;
        d_h_dense2_relu(n, 1) = d_theta(0, 1) / N;
        d_h_dense2_relu(n, 2) = d_theta(0, 2) / N;
        d_h_dense2_relu(n, 3) = d_theta(1, 0) / N;
        d_h_dense2_relu(n, 4) = d_theta(1, 1) / N;
        d_h_dense2_relu(n, 5) = d_theta(1, 2) / N;
    }

    // Re-compute forward intermediates (cheap, used for relu mask)
    Tensor h_conv = conv2d_forward(last_input, loc_conv_W, loc_conv_b, C, 8, 3, 3);
    const int Hp = H / 2, Wp = W / 2;
    Tensor h_pool = maxpool2d_forward(h_conv);
    Tensor h_pool_relu_pre = h_pool;       // input to relu
    Tensor h_pool_relu = relu_forward(h_pool);
    Tensor h_dense1_pre = dense_forward(h_pool_relu, loc_dense1_W, loc_dense1_b);  // input to relu
    Tensor h_dense1 = h_dense1_pre;
    Tensor h_dense1_relu = relu_forward(h_dense1);
    Tensor h_dense2_pre = dense_forward(h_dense1_relu, loc_dense2_W, loc_dense2_b);  // input to relu
    Tensor h_dense2 = h_dense2_pre;

    // 2. relu backward (dense2_relu input = h_dense2)
    Tensor d_h_dense2 = relu_backward(d_h_dense2_relu, h_dense2);
    // 3. dense2 backward -> d_h_dense1_relu + updates to grad_loc_dense2_W/b
    Tensor d_h_dense1_relu_via_dense2 = dense_backward(
        d_h_dense2, loc_dense2_W, grad_loc_dense2_W, grad_loc_dense2_b);
    // 4. relu backward
    Tensor d_h_dense1_b = relu_backward(d_h_dense1_relu_via_dense2, h_dense1);
    // 5. dense1 backward -> d_h_pool_relu + updates to grad_loc_dense1_W/b
    // For dense_backward to work it needs last_input — set it to h_pool_relu temporarily
    Tensor saved_input = last_input;
    last_input = h_pool_relu;
    Tensor d_h_pool_relu = dense_backward(
        d_h_dense1_b, loc_dense1_W, grad_loc_dense1_W, grad_loc_dense1_b);
    // 6. relu backward
    Tensor d_h_pool = relu_backward(d_h_pool_relu, h_pool_relu_pre);
    // 7. maxpool backward -> d_h_conv
    Tensor d_h_conv = maxpool2d_backward(d_h_pool, Hp, Wp);
    // 8. conv backward -> d_input + updates to grad_loc_conv_W/b
    last_input = saved_input;
    Tensor d_input = conv2d_backward(
        d_h_conv, loc_conv_W, C, 8, 3, 3,
        grad_loc_conv_W, grad_loc_conv_b);
    return d_input;
}

// ============================================================================
// Grid generator
// ============================================================================

Tensor SpatialTransformer::generate_grid(const Tensor& theta) const {
    // theta: (2, 3). Output: (H_out * W_out, 2)
    // For each output pixel (i_out, j_out):
    //   x_norm = 2*j_out / (W_out - 1) - 1   (PyTorch convention)
    //   y_norm = 2*i_out / (H_out - 1) - 1
    //   [x_src, y_src]^T = theta * [x_norm, y_norm, 1]^T
    // With this convention, the identity theta maps output pixel j to
    // exactly input pixel j (since the un-normalize step is symmetric).
    Tensor grid(H_out * W_out, 2);
    for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
            double xn = (W_out > 1) ? (2.0 * j / (W_out - 1) - 1.0) : 0.0;
            double yn = (H_out > 1) ? (2.0 * i / (H_out - 1) - 1.0) : 0.0;
            double xs = theta(0, 0) * xn + theta(0, 1) * yn + theta(0, 2);
            double ys = theta(1, 0) * xn + theta(1, 1) * yn + theta(1, 2);
            int idx = i * W_out + j;
            grid(idx, 0) = xs;
            grid(idx, 1) = ys;
        }
    }
    return grid;
}

Tensor SpatialTransformer::grid_backward(const Tensor& d_grid) const {
    // d_grid: (H_out * W_out, 2). Returns d_theta: (2, 3).
    // d_theta(a, b) = sum over (i, j) of d_grid(i, j)[a] * source_coord(i, j)[b-1 if b>0 else b]
    // Note: the affine is [xs, ys]^T = theta * [xn, yn, 1]^T where theta is (2, 3).
    // d_theta(a, b) = sum_{i,j} d_grid(i, j, a) * coord_b where coord_0 = xn, coord_1 = yn, coord_2 = 1.
    Tensor d_theta(2, 3);
    for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
            int idx = i * W_out + j;
            double xn = (W_out > 1) ? (2.0 * j / (W_out - 1) - 1.0) : 0.0;
            double yn = (H_out > 1) ? (2.0 * i / (H_out - 1) - 1.0) : 0.0;
            // a=0, b=0: xn; a=0, b=1: yn; a=0, b=2: 1
            // a=1, b=0: xn; a=1, b=1: yn; a=1, b=2: 1
            d_theta(0, 0) += d_grid(idx, 0) * xn;
            d_theta(0, 1) += d_grid(idx, 0) * yn;
            d_theta(0, 2) += d_grid(idx, 0) * 1.0;
            d_theta(1, 0) += d_grid(idx, 1) * xn;
            d_theta(1, 1) += d_grid(idx, 1) * yn;
            d_theta(1, 2) += d_grid(idx, 1) * 1.0;
        }
    }
    return d_theta;
}

// ============================================================================
// Bilinear sampler
// ============================================================================

Tensor SpatialTransformer::bilinear_sample_forward(const Tensor& input, const Tensor& grid) {
    // input: (N, C * H * W)
    // grid: (H_out * W_out, 2) — normalized source coords in [-1, 1]^2
    // output: (N, C * H_out * W_out)
    //
    // For each output pixel (i, j), for each channel c, for each batch n:
    //   xs = grid(i * W_out + j, 0), ys = grid(..., 1)
    //   Convert to input pixel coords: sx = (xs + 1) / 2 * (W - 1), sy = (ys + 1) / 2 * (H - 1)
    //   If sx, sy out of [0, W-1] / [0, H-1], out = 0
    //   Else bilinear interp
    Tensor out(N, C * H_out * W_out);
    last_bilinear_w.assign(H_out * W_out, std::vector<double>(4, 0.0));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    int idx = i * W_out + j;
                    double xs = grid(idx, 0);
                    double ys = grid(idx, 1);
                    // [-1, 1] -> [0, W-1] / [0, H-1]
                    double sx = (xs + 1.0) * 0.5 * (W - 1);
                    double sy = (ys + 1.0) * 0.5 * (H - 1);

                    int x0 = (int)std::floor(sx);
                    int y0 = (int)std::floor(sy);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    double wx1 = sx - x0;
                    double wy1 = sy - y0;
                    double wx0 = 1.0 - wx1;
                    double wy0 = 1.0 - wy1;

                    double val = 0.0;
                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
                        val += wx0 * wy0 * input(n, c * H * W + y0 * W + x0);
                        last_bilinear_w[idx][0] = wx0 * wy0;  // weight to (y0, x0)
                    }
                    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
                        val += wx1 * wy0 * input(n, c * H * W + y0 * W + x1);
                        last_bilinear_w[idx][1] = wx1 * wy0;  // weight to (y0, x1)
                    }
                    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
                        val += wx0 * wy1 * input(n, c * H * W + y1 * W + x0);
                        last_bilinear_w[idx][2] = wx0 * wy1;  // weight to (y1, x0)
                    }
                    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        val += wx1 * wy1 * input(n, c * H * W + y1 * W + x1);
                        last_bilinear_w[idx][3] = wx1 * wy1;  // weight to (y1, x1)
                    }
                    out(n, c * H_out * W_out + i * W_out + j) = val;
                }
            }
        }
    }
    return out;
}

Tensor SpatialTransformer::bilinear_sample_backward_input(const Tensor& grad_output) {
    // grad_output: (N, C * H_out * W_out)
    // Returns d_input: (N, C * H * W)
    Tensor d_input(N, C * H * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    int idx = i * W_out + j;
                    double g = grad_output(n, c * H_out * W_out + idx);
                    if (g == 0.0) continue;
                    double xs = last_grid(idx, 0);
                    double ys = last_grid(idx, 1);
                    double sx = (xs + 1.0) * 0.5 * (W - 1);
                    double sy = (ys + 1.0) * 0.5 * (H - 1);

                    int x0 = (int)std::floor(sx);
                    int y0 = (int)std::floor(sy);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    double wx1 = sx - x0;
                    double wy1 = sy - y0;
                    double wx0 = 1.0 - wx1;
                    double wy0 = 1.0 - wy1;

                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
                        d_input(n, c * H * W + y0 * W + x0) += g * wx0 * wy0;
                    }
                    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
                        d_input(n, c * H * W + y0 * W + x1) += g * wx1 * wy0;
                    }
                    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
                        d_input(n, c * H * W + y1 * W + x0) += g * wx0 * wy1;
                    }
                    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        d_input(n, c * H * W + y1 * W + x1) += g * wx1 * wy1;
                    }
                }
            }
        }
    }
    return d_input;
}

Tensor SpatialTransformer::bilinear_sample_backward_grid(const Tensor& grad_output) {
    // grad_output: (N, C * H_out * W_out)
    // Returns d_grid: (H_out * W_out, 2)
    // For each output pixel, sum grad * input_at_corner over channels and batch.
    // d(xs) = sum over (n, c) grad(n, c, i, j) * (∂V/∂sx)
    //   where V = bilinear sample with V depending on sx via wx0, wx1, sy.
    // dV/dsx = (wy0 * (-input(y0,x1) + input(y0,x0)) * ... wait let's do it directly.
    //
    // For one output pixel, V = w00*V(y0,x0) + w01*V(y0,x1) + w10*V(y1,x0) + w11*V(y1,x1)
    // where w00 = wx0*wy0, w01 = wx1*wy0, w10 = wx0*wy1, w11 = wx1*wy1.
    // dV/dsx = (∂w00/dsx)*V(y0,x0) + (∂w01/dsx)*V(y0,x1) + ...
    //          wx0 = 1 - (sx - x0), so ∂wx0/dsx = -1, ∂wx1/dsx = +1
    // dV/dsx = -1*wy0*V(y0,x0) + 1*wy0*V(y0,x1) - 1*wy1*V(y1,x0) + 1*wy1*V(y1,x1)
    // dV/dsy similar: -1*wx0*V(y0,x0) - 1*wx1*V(y0,x1) + 1*wx0*V(y1,x0) + 1*wx1*V(y1,x1)
    Tensor d_grid(H_out * W_out, 2);
    // MUTATION TEST: zero out d_grid to verify grad check catches it
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    int idx = i * W_out + j;
                    double g = grad_output(n, c * H_out * W_out + idx);
                    if (g == 0.0) continue;
                    double xs = last_grid(idx, 0);
                    double ys = last_grid(idx, 1);
                    double sx = (xs + 1.0) * 0.5 * (W - 1);
                    double sy = (ys + 1.0) * 0.5 * (H - 1);
                    int x0 = (int)std::floor(sx);
                    int y0 = (int)std::floor(sy);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    double wx0 = 1.0 - (sx - x0);
                    double wy0 = 1.0 - (sy - y0);
                    double wx1 = sx - x0;
                    double wy1 = sy - y0;

                    double v00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
                                 ? last_input(n, c * H * W + y0 * W + x0) : 0.0;
                    double v01 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H)
                                 ? last_input(n, c * H * W + y0 * W + x1) : 0.0;
                    double v10 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H)
                                 ? last_input(n, c * H * W + y1 * W + x0) : 0.0;
                    double v11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H)
                                 ? last_input(n, c * H * W + y1 * W + x1) : 0.0;

                    double dxs = -wy0 * v00 + wy0 * v01 - wy1 * v10 + wy1 * v11;
                    double dys = -wx0 * v00 - wx1 * v01 + wx0 * v10 + wx1 * v11;
                    // dsx/ds_src_x = (W - 1) / 2 (since sx = (xs+1)/2 * (W-1))
                    // Similarly for sy
                    double chain = (W - 1) * 0.5;
                    d_grid(idx, 0) += g * dxs * chain;
                    chain = (H - 1) * 0.5;
                    d_grid(idx, 1) += g * dys * chain;
                }
            }
        }
    }
    return d_grid;
}

// ============================================================================
// Forward / Backward orchestration
// ============================================================================

Tensor SpatialTransformer::forward(const Tensor& input) {
    if (input.rows != (size_t)N || input.cols != (size_t)(C * H * W)) {
        throw std::invalid_argument("SpatialTransformer::forward: input shape mismatch");
    }
    last_input = input;

    // 1. Get theta
    Tensor theta;
    if (use_fixed_theta_) {
        theta = fixed_theta_;
    } else {
        theta = localization_forward(input);
    }
    last_theta = theta;

    // 2. Generate grid
    Tensor grid = generate_grid(theta);
    last_grid = grid;

    // 3. Bilinear sample
    Tensor out = bilinear_sample_forward(input, grid);
    return out;
}

Tensor SpatialTransformer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != (size_t)N || grad_output.cols != (size_t)(C * H_out * W_out)) {
        throw std::invalid_argument("SpatialTransformer::backward: grad_output shape mismatch");
    }

    // 1. d_grid from bilinear sample
    Tensor d_grid = bilinear_sample_backward_grid(grad_output);

    // 2. d_theta from grid
    Tensor d_theta = grid_backward(d_grid);

    // 3. d_input to locator: from localization backward if not fixed, else
    //    from bilinear sample backward.
    Tensor d_input;
    if (use_fixed_theta_) {
        d_input = bilinear_sample_backward_input(grad_output);
    } else {
        d_input = localization_backward(d_theta);
    }
    return d_input;
}