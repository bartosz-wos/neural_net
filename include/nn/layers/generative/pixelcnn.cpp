// ============================================================================
// PixelCNN / Gated PixelCNN — van den Oord et al. 2016
//
// See pixelcnn.h for the full mathematical formulation and references.
//
// This file implements:
//   * MaskedConv2d        : Conv2D with a fixed binary mask on the kernel.
//   * GatedPixelCNNBlock  : gated activation (tanh * sigmoid) + 1x1 residual.
//   * PixelCNN            : full autoregressive model (first A-conv + gated blocks
//                           + final 1x1 classifier producing per-pixel logits).
//
// Conventions follow the rest of the repo:
//   * Input shape: (1, in_channels * H * W) — single image, channels-first.
//   * Dense convention for the 1x1 / 2D conv: weights (out_channels, in_channels * kH * kW).
//   * Output: y = X @ W^T + b (see im2col / weights * col path below).
// ============================================================================

#include "pixelcnn.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

// ============================================================================
// MaskedConv2d
// ============================================================================

MaskedConv2d::MaskedConv2d()
    : in_channels(0), out_channels(0),
      kernel_h(1), kernel_w(1),
      stride_h(1), stride_w(1),
      pad_h(0), pad_w(0),
      dilation_h(1), dilation_w(1),
      H(0), W(0), H_out(0), W_out(0),
      mask_type('B')
{
    weights = Tensor(0, 0);
    bias = Tensor(0, 0);
    grad_weights = Tensor(0, 0);
    grad_bias = Tensor(0, 0);
    mask = Tensor(0, 0);
    weight_mask = Tensor(0, 0);
}

MaskedConv2d::MaskedConv2d(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
                           char mask_in,
                           int stride_h_in, int stride_w_in,
                           int pad_h_in, int pad_w_in,
                           int dilation_h_in, int dilation_w_in)
    : in_channels(in_ch), out_channels(out_ch),
      kernel_h(kH), kernel_w(kW),
      stride_h(stride_h_in), stride_w(stride_w_in),
      // Default pad = "same" (kH-1)/2, (kW-1)/2. Pass pad_h_in=-1 / pad_w_in=-1
      // explicitly to request the default.
      pad_h(0), pad_w(0),  // placeholder, set below
      dilation_h(dilation_h_in), dilation_w(dilation_w_in),
      H(H_in), W(W_in),
      mask_type(mask_in)
{
    pad_h = (pad_h_in < 0) ? ((kH - 1) / 2) : pad_h_in;
    pad_w = (pad_w_in < 0) ? ((kW - 1) / 2) : pad_w_in;
    if (mask_in != 'A' && mask_in != 'B') {
        throw std::invalid_argument("MaskedConv2d: mask_type must be 'A' or 'B'");
    }
    if (kH <= 0 || kW <= 0) {
        throw std::invalid_argument("MaskedConv2d: kernel size must be > 0");
    }
    if (kH % 2 == 0 || kW % 2 == 0) {
        throw std::invalid_argument("MaskedConv2d: kernel must be odd (so center pixel exists)");
    }

    H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("MaskedConv2d: output dimensions non-positive");
    }

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

    // Build the kernel mask (1, kH*kW)
    mask = build_kernel_mask(kH, kW, mask_type);
    // Broadcast to weight shape (out_channels, in_channels * kH * kW)
    // Same mask applied to every (out_ch, in_ch) pair.
    weight_mask = Tensor(out_channels, in_channels * kH * kW);
    for (int o = 0; o < out_channels; ++o) {
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < kH * kW; ++i) {
                weight_mask(o, c * kH * kW + i) = mask(0, (size_t)i);
            }
        }
    }
    enforce_mask();
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

Tensor MaskedConv2d::build_kernel_mask(int kH, int kW, char mask_type) {
    Tensor m(1, kH * kW);
    int center_i = kH / 2;
    int center_j = kW / 2;
    for (int i = 0; i < kH; ++i) {
        for (int j = 0; j < kW; ++j) {
            // "Causal" 2D mask: blocked if (i, j) is below or to the right of
            // center, OR if it's the center pixel (for type A).
            // Specifically:
            //   * PixelCNN standard: a position is allowed iff (i, j) is
            //     "northwest" of the center, i.e. (i < center_i) OR
            //     (i == center_i AND j <= center_j).
            //   * Type A additionally blocks the center: (i, j) != center.
            //   * Type B allows the center: same as PixelCNN standard.
            bool allowed;
            if (i < center_i) {
                allowed = true;
            } else if (i == center_i) {
                allowed = (j <= center_j);
            } else {
                allowed = false;
            }
            if (mask_type == 'A') {
                if (i == center_i && j == center_j) allowed = false;
            }
            m(0, i * kW + j) = allowed ? 1.0 : 0.0;
        }
    }
    return m;
}

void MaskedConv2d::enforce_mask() {
    if (weight_mask.rows == 0) return;
    for (size_t i = 0; i < weights.rows; ++i) {
        for (size_t j = 0; j < weights.cols; ++j) {
            if (weight_mask(i, j) == 0.0) {
                weights(i, j) = 0.0;
            }
        }
    }
}

// im2col — identical algorithm to Conv2D (we re-implement to keep this layer
// self-contained and avoid cross-class dependency on the static protected
// members of Conv2D which are not part of its public interface).
Tensor MaskedConv2d::im2col(const Tensor& input, int N, int C, int H, int W,
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

void MaskedConv2d::col2im(Tensor& grad_input, const Tensor& grad_col,
                          int N, int C, int H, int W,
                          int kH, int kW, int stride_h, int stride_w,
                          int pad_h, int pad_w, int dilation_h, int dilation_w) {
    int H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
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

Tensor MaskedConv2d::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != in_channels * H * W) {
        throw std::invalid_argument("MaskedConv2d: input dimension mismatch");
    }
    if (N != 1) {
        throw std::invalid_argument("MaskedConv2d: v1 supports only N=1 (single image)");
    }

    // Enforce mask on weights before forward (in case update happened)
    enforce_mask();

    last_input = input;
    col = im2col(input, N, in_channels, H, W,
                 kernel_h, kernel_w, stride_h, stride_w,
                 pad_h, pad_w, dilation_h, dilation_w,
                 H_out, W_out);

    int out_spatial = H_out * W_out;
    Tensor Z = weights * col; // (out_channels, N * out_spatial)

    // Add bias
    for (int o = 0; o < out_channels; ++o) {
        for (int idx = 0; idx < N * out_spatial; ++idx) {
            Z[o][idx] += bias[o][0];
        }
    }

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

Tensor MaskedConv2d::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    int out_spatial = H_out * W_out;

    // Reshape grad_output to matrix (out_channels, N * out_spatial)
    Tensor grad_out_mat(out_channels, N * out_spatial);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int s = 0; s < out_spatial; ++s) {
                grad_out_mat[o][n * out_spatial + s] = grad_output[n][o * out_spatial + s];
            }
        }
    }

    // dW = grad_out_mat * col^T
    Tensor col_T = col.transpose();
    Tensor dW = grad_out_mat * col_T;
    // Apply mask to gradient so blocked positions get zero grad
    for (size_t i = 0; i < dW.rows; ++i) {
        for (size_t j = 0; j < dW.cols; ++j) {
            if (weight_mask(i, j) == 0.0) dW(i, j) = 0.0;
        }
    }
    for (size_t i = 0; i < grad_weights.rows; ++i) {
        for (size_t j = 0; j < grad_weights.cols; ++j) {
            grad_weights(i, j) += dW(i, j);
        }
    }

    // db
    Tensor db(out_channels, 1);
    for (int o = 0; o < out_channels; ++o) {
        double sum = 0.0;
        for (int i = 0; i < N * out_spatial; ++i) sum += grad_out_mat[o][i];
        db[o][0] = sum;
    }
    for (int o = 0; o < out_channels; ++o) {
        grad_bias(o, 0) += db[o][0];
    }

    // dX_col = weights^T * grad_out_mat
    Tensor weights_T = weights.transpose();
    Tensor dX_col = weights_T * grad_out_mat;

    // col2im
    Tensor grad_input(N, in_channels * H * W);
    grad_input.fill(0.0);
    col2im(grad_input, dX_col, N, in_channels, H, W,
           kernel_h, kernel_w, stride_h, stride_w,
           pad_h, pad_w, dilation_h, dilation_w);

    return grad_input;
}

void MaskedConv2d::update_weights(double learning_rate) {
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kernel_h * kernel_w; ++i) {
            weights[o][i] -= learning_rate * grad_weights[o][i];
            // Re-zero blocked positions AFTER update
            if (weight_mask(o, i) == 0.0) weights[o][i] = 0.0;
        }
        bias[o][0] -= learning_rate * grad_bias[o][0];
    }
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

std::vector<Tensor*> MaskedConv2d::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> MaskedConv2d::gradients() {
    return {&grad_weights, &grad_bias};
}

void MaskedConv2d::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

// ============================================================================
// GatedPixelCNNBlock
// ============================================================================

GatedPixelCNNBlock::GatedPixelCNNBlock(int in_ch, int hidden_ch, int kH, int kW,
                                       int H_in, int W_in,
                                       char mask, int cond_dim_in)
    : in_channels(in_ch), hidden_channels(hidden_ch),
      H(H_in), W(W_in), kH(kH), kW(kW), cond_dim(cond_dim_in),
      conv_v_(in_ch, 2 * hidden_ch, kH, kW, H_in, W_in, mask),
      conv_u_(hidden_ch, in_ch, 1, 1, H_in, W_in, 'B'),
      cond_proj_((size_t)(cond_dim_in > 0 ? cond_dim_in : 1),
                 (size_t)(2 * hidden_ch)),
      bias_v_(2 * hidden_ch, 1),
      grad_bias_v_(2 * hidden_ch, 1)
{
    if (in_ch <= 0 || hidden_ch <= 0 || kH <= 0 || kW <= 0) {
        throw std::invalid_argument("GatedPixelCNNBlock: invalid dims");
    }
    if (H_in <= 0 || W_in <= 0) {
        throw std::invalid_argument("GatedPixelCNNBlock: H, W must be > 0");
    }
    if (cond_dim < 0) {
        throw std::invalid_argument("GatedPixelCNNBlock: cond_dim must be >= 0");
    }

    std::mt19937 gen(123);
    std::normal_distribution<> dis(0.0, 0.01);
    for (int i = 0; i < 2 * hidden_ch; ++i) {
        bias_v_(i, 0) = dis(gen);
    }
    grad_bias_v_.fill(0.0);
}

Tensor GatedPixelCNNBlock::forward(const Tensor& input) {
    return forward_with_cond(input, Tensor(0, 0));
}

Tensor GatedPixelCNNBlock::forward_with_cond(const Tensor& input, const Tensor& cond) {
    if (input.rows != 1 || input.cols != (size_t)(in_channels * H * W)) {
        throw std::invalid_argument("GatedPixelCNNBlock: input shape mismatch");
    }
    last_input_ = input.clone();
    bool use_cond = (cond_dim > 0 && cond.rows > 0 && cond.cols > 0);
    used_cond_ = use_cond;
    if (use_cond) {
        if (cond.cols != (size_t)cond_dim) {
            throw std::invalid_argument("GatedPixelCNNBlock: cond dim mismatch");
        }
        last_cond_ = cond.clone();
    }

    // Step 1: masked conv -> 2*hidden channels (pre-activation)
    Tensor v = conv_v_.forward(input); // (1, 2*hidden * H * W)
    // Add per-channel bias
    int spatial = H * W;
    for (int c = 0; c < 2 * hidden_channels; ++c) {
        for (int s = 0; s < spatial; ++s) {
            v(0, c * spatial + s) += bias_v_(c, 0);
        }
    }
    // Conditioning: project cond -> 2*hidden, broadcast over pixels
    if (use_cond) {
        // cond_proj_.weights is (2*hidden, cond_dim); cond is (1, cond_dim)
        // output is (1, 2*hidden). We need to add the same bias to every pixel.
        Tensor cond_proj = cond_proj_.forward(cond); // (1, 2*hidden)
        for (int c = 0; c < 2 * hidden_channels; ++c) {
            double bias_c = cond_proj(0, c);
            for (int s = 0; s < spatial; ++s) {
                v(0, c * spatial + s) += bias_c;
            }
        }
    }
    last_v_ = v.clone();

    // Step 2: split f and g, apply tanh * sigmoid
    Tensor h(1, hidden_channels * spatial);
    for (int c = 0; c < hidden_channels; ++c) {
        for (int s = 0; s < spatial; ++s) {
            double f = v(0, c * spatial + s);
            double g = v(0, (hidden_channels + c) * spatial + s);
            double t = std::tanh(f);
            double sg = 1.0 / (1.0 + std::exp(-g));
            h(0, c * spatial + s) = t * sg;
        }
    }
    last_h_ = h.clone();

    // Step 3: 1x1 conv -> in_channels (residual contribution)
    Tensor u = conv_u_.forward(h);
    last_u_ = u.clone();

    // Step 4: residual sum
    Tensor out(1, in_channels * spatial);
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = input.data[i] + u.data[i];
    }
    last_output_ = out.clone();
    return out;
}

Tensor GatedPixelCNNBlock::backward(const Tensor& grad_output, double lr) {
    Tensor dummy_cond(0, 0);
    return backward_with_cond(grad_output, lr, dummy_cond);
}

Tensor GatedPixelCNNBlock::backward_with_cond(const Tensor& grad_output, double /* lr */,
                                              const Tensor& cond) {
    if (grad_output.rows != 1 || grad_output.cols != (size_t)(in_channels * H * W)) {
        throw std::invalid_argument("GatedPixelCNNBlock: grad_output shape mismatch");
    }
    int spatial = H * W;
    bool use_cond = used_cond_;

    conv_v_.zero_grad();
    conv_u_.zero_grad();
    cond_proj_.zero_grad();
    grad_bias_v_.fill(0.0);

    // Step 1: residual split. out = input + u, so:
    //   d_input += grad_output
    //   d_u = grad_output
    Tensor d_u = grad_output.clone();
    Tensor d_input_from_residual = grad_output.clone();

    // Step 2: backward through conv_u_ (1x1)
    //   d_h = conv_u_.backward(d_u)
    Tensor d_h = conv_u_.backward(d_u, 0.0);

    // Step 3: backward through the gated activation h = tanh(f) * sigmoid(g)
    //   v layout: [f_0..f_{H-1}, g_0..g_{H-1}] per pixel
    //   h_c = tanh(f_c) * sigmoid(g_c)
    //   d_f_c = d_h_c * (1 - tanh(f_c)^2) * sigmoid(g_c)
    //   d_g_c = d_h_c * tanh(f_c) * sigmoid(g_c) * (1 - sigmoid(g_c))
    Tensor d_v(1, 2 * hidden_channels * spatial);
    d_v.fill(0.0);
    for (int c = 0; c < hidden_channels; ++c) {
        for (int s = 0; s < spatial; ++s) {
            double f = last_v_(0, c * spatial + s);
            double g = last_v_(0, (hidden_channels + c) * spatial + s);
            double t = std::tanh(f);
            double sg = 1.0 / (1.0 + std::exp(-g));
            double dh = d_h(0, c * spatial + s);
            double df = dh * (1.0 - t * t) * sg;
            double dg = dh * t * sg * (1.0 - sg);
            d_v(0, c * spatial + s) = df;
            d_v(0, (hidden_channels + c) * spatial + s) = dg;
        }
    }

    // Step 3b: bias_v_ gradient (sum over spatial for each channel)
    for (int c = 0; c < 2 * hidden_channels; ++c) {
        double acc = 0.0;
        for (int s = 0; s < spatial; ++s) acc += d_v(0, c * spatial + s);
        grad_bias_v_(c, 0) += acc;
    }

    // Step 3c: if conditioning was used, gradient through cond_proj_
    // The conditioning adds cond_proj(cond) to v (broadcast over pixels),
    // so d_cond_proj = d_v (summed over spatial per channel), and
    // d_cond = cond_proj_.backward(d_cond_proj, 0.0).
    if (use_cond) {
        Tensor d_cond_proj(1, 2 * hidden_channels);
        for (int c = 0; c < 2 * hidden_channels; ++c) {
            double acc = 0.0;
            for (int s = 0; s < spatial; ++s) acc += d_v(0, c * spatial + s);
            d_cond_proj(0, c) = acc;
        }
        // Backward through cond_proj_ (Dense: y = X @ W^T + b)
        // For Dense: dX = dY @ W, dW += dY^T @ X, db += sum over batch of dY
        // dY is (1, 2*hidden), X is (1, cond_dim), W is (2*hidden, cond_dim)
        // dX = dY @ W -> (1, cond_dim) — gradient w.r.t. cond (we discard it;
        //                   the caller has the cond tensor if they want it)
        // We need to populate cond_proj_.grad_weights and cond_proj_.grad_bias.
        // The Dense::backward returns dX, which is what we want.
        cond_proj_.backward(d_cond_proj, 0.0);
    }

    // Step 4: backward through conv_v_ (masked conv with d_v as grad_output)
    Tensor d_input_from_v = conv_v_.backward(d_v, 0.0);

    // Step 5: combine d_input contributions
    Tensor d_input(1, in_channels * spatial);
    for (size_t i = 0; i < d_input.data.size(); ++i) {
        d_input.data[i] = d_input_from_residual.data[i] + d_input_from_v.data[i];
    }
    return d_input;
}

void GatedPixelCNNBlock::update_weights(double learning_rate) {
    conv_v_.update_weights(learning_rate);
    conv_u_.update_weights(learning_rate);
    if (cond_dim > 0) {
        cond_proj_.update_weights(learning_rate);
    }
    // Update bias_v_
    for (int c = 0; c < 2 * hidden_channels; ++c) {
        bias_v_(c, 0) -= learning_rate * grad_bias_v_(c, 0);
    }
    grad_bias_v_.fill(0.0);
}

std::vector<Tensor*> GatedPixelCNNBlock::parameters() {
    if (cond_dim > 0) {
        return {&conv_v_.weights, &conv_v_.bias, &bias_v_,
                &conv_u_.weights, &conv_u_.bias,
                &cond_proj_.weights, &cond_proj_.bias};
    } else {
        return {&conv_v_.weights, &conv_v_.bias, &bias_v_,
                &conv_u_.weights, &conv_u_.bias};
    }
}

std::vector<Tensor*> GatedPixelCNNBlock::gradients() {
    if (cond_dim > 0) {
        return {&conv_v_.grad_weights, &conv_v_.grad_bias, &grad_bias_v_,
                &conv_u_.grad_weights, &conv_u_.grad_bias,
                &cond_proj_.grad_weights, &cond_proj_.grad_bias};
    } else {
        return {&conv_v_.grad_weights, &conv_v_.grad_bias, &grad_bias_v_,
                &conv_u_.grad_weights, &conv_u_.grad_bias};
    }
}

void GatedPixelCNNBlock::zero_grad() {
    conv_v_.zero_grad();
    conv_u_.zero_grad();
    if (cond_dim > 0) cond_proj_.zero_grad();
    grad_bias_v_.fill(0.0);
}

// ============================================================================
// PixelCNN
// ============================================================================

PixelCNN::PixelCNN(int in_ch, int hidden_ch, int kH, int kW, int H_in, int W_in,
                   int n_blocks_in, int n_vals_in,
                   char first_mask_in, int cond_dim_in)
    : in_channels(in_ch),
      hidden_channels(hidden_ch),
      n_blocks(n_blocks_in),
      n_vals(n_vals_in),
      H(H_in), W(W_in),
      kH(kH), kW(kW),
      first_mask(first_mask_in),
      cond_dim(cond_dim_in),
      first_conv_(in_ch, hidden_ch, kH, kW, H_in, W_in, first_mask_in),
      classifier_(hidden_ch, in_ch * n_vals_in, 1, 1, H_in, W_in, 'B')
{
    if (in_ch <= 0 || hidden_ch <= 0 || n_blocks_in <= 0 || n_vals_in <= 0) {
        throw std::invalid_argument("PixelCNN: invalid dims");
    }
    if (kH <= 0 || kW <= 0 || H_in <= 0 || W_in <= 0) {
        throw std::invalid_argument("PixelCNN: invalid spatial dims");
    }
    if (cond_dim_in < 0) {
        throw std::invalid_argument("PixelCNN: cond_dim must be >= 0");
    }
    // Build gated blocks. Each block takes `in_channels` as input (the channel
    // width flowing through the network) and outputs `in_channels` for residual
    // compatibility, but its internal hidden_channels can be wider.
    // The first block receives `hidden_channels` from the first conv, so we
    // need a pre-block projection to map hidden_channels -> in_channels.
    // For simplicity in v1: the first conv outputs `in_channels` (not hidden).
    // Adjust: if hidden != in, use a 1x1 projection before the first block.
    // To keep things tractable, we just enforce hidden_channels == in_channels
    // for v1. (Tests use hidden=2, in_ch=1 — but the architecture should
    // support hidden != in via a projection layer.)
    //
    // Actually, the simpler architecture: the first conv outputs
    // `hidden_channels`, and each gated block reduces back to in_channels.
    // Then the NEXT block expects in_channels as input. We just keep a single
    // block width = in_channels, with hidden_channels being the INTERNAL width
    // of the gated conv. For multiple blocks, the residual path is the same
    // width (in_channels), so blocks are stackable without extra projections.
    // The first conv is then in_channels -> hidden_channels, and the first
    // block's residual is hidden -> in_channels via conv_u_. After that, all
    // blocks are in -> in.
    //
    // For v1 we restrict: first conv outputs in_channels, and hidden_channels
    // is the internal width of the gated convs (does not affect residual
    // width). All blocks operate on in_channels input and output.
    //
    // RECONCILE: Set first_conv_ output = in_channels (overwriting the
    // argument we passed). This is a slight deviation from the constructor
    // signature, but matches the simpler v1 architecture and is what the
    // tests expect (see test 15: hidden=2, but we only need the OUT shape to
    // be C_in * n_vals * H * W).
    //
    // Wait — looking again, the first conv was constructed as
    // MaskedConv2d(in_ch, hidden_ch, ...). If hidden_ch != in_ch, this won't
    // match the block contract. We need to rebuild the first conv to output
    // in_ch instead.
    //
    // Simpler: rebuild first_conv_ as in_ch -> in_ch, and use hidden_ch
    // purely for the gated conv internal width. To do this without breaking
    // the existing construction, we'll do it after the constructor body
    // by reassigning. But first_conv_ is a MaskedConv2d, not a pointer, so
    // we can use placement-new or simply construct in-place.

    // Approach: re-construct first_conv_ in-place with the corrected out_ch.
    // Since C++ doesn't allow that cleanly, we use std::launder trick or
    // just override the weights/bias with a new mask.
    // For simplicity and clarity, we re-build:
    // (The earlier construction is wasted but doesn't matter functionally.)
    // We rebuild first_conv_ with the corrected dims.
    // To do this, we need to call the assignment operator or use a member
    // function that rebuilds. MaskedConv2d doesn't have one, so we use
    // placement-new on a fresh in-place buffer.
    // Simpler: we just leave first_conv_ at hidden_ch output. The first
    // block then needs to accept hidden_ch as input. We make the first
    // block accept hidden_ch (not in_ch) and output in_ch via residual.
    // This means each block has different input dim. To keep things uniform,
    // we use a uniform "channel" width = hidden_channels throughout, and
    // the final classifier is hidden_channels -> in_ch * n_vals.
    // But that requires the first conv to output hidden_channels. ✓ already.
    // And the classifier's input is hidden_channels. ✓ already.
    // The blocks all have in_ch = hidden_channels. So we just pass
    // hidden_channels to the block constructor.
    //
    // Let's NOT rebuild. Use hidden_channels as the uniform internal width.
    // Each block: hidden_ch -> hidden_ch (residual preserves width).
    // The first conv: in_ch -> hidden_ch. ✓
    // The classifier: hidden_ch -> in_ch * n_vals. ✓ (already done)
    for (int b = 0; b < n_blocks_in; ++b) {
        blocks_.emplace_back(new GatedPixelCNNBlock(
            hidden_ch, hidden_ch, kH, kW, H_in, W_in, 'B', cond_dim_in));
    }
    last_block_outputs_.reserve(n_blocks_in);
}

Tensor PixelCNN::forward(const Tensor& input) {
    return forward_with_cond(input, Tensor(0, 0));
}

Tensor PixelCNN::forward_with_cond(const Tensor& input, const Tensor& cond) {
    if (input.rows != 1 || input.cols != (size_t)(in_channels * H * W)) {
        throw std::invalid_argument("PixelCNN: input shape mismatch");
    }
    last_input_ = input.clone();
    bool use_cond = (cond_dim > 0 && cond.rows > 0 && cond.cols > 0);
    used_cond_ = use_cond;
    if (use_cond) {
        if (cond.cols != (size_t)cond_dim) {
            throw std::invalid_argument("PixelCNN: cond dim mismatch");
        }
        last_cond_ = cond.clone();
    }

    // Step 1: first masked conv (type A)
    Tensor h = first_conv_.forward(input);  // (1, hidden * H * W)
    last_h0_ = h.clone();

    // Step 2: stack of gated blocks
    last_block_outputs_.clear();
    for (auto& block : blocks_) {
        if (use_cond) {
            h = block->forward_with_cond(h, cond);
        } else {
            h = block->forward(h);
        }
        last_block_outputs_.push_back(h.clone());
    }

    // Step 3: classifier (1x1 conv): hidden -> in_ch * n_vals
    Tensor logits = classifier_.forward(h);
    return logits;
}

Tensor PixelCNN::backward(const Tensor& grad_output, double lr) {
    Tensor dummy_cond(0, 0);
    return backward_with_cond(grad_output, lr, dummy_cond);
}

Tensor PixelCNN::backward_with_cond(const Tensor& grad_output, double /* lr */,
                                    const Tensor& cond) {
    if (grad_output.rows != 1 || grad_output.cols != (size_t)(in_channels * n_vals * H * W)) {
        throw std::invalid_argument("PixelCNN: grad_output shape mismatch");
    }
    // Step 1: backward through classifier
    Tensor d_h = classifier_.backward(grad_output, 0.0);

    // Step 2: backward through blocks in reverse order
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        if (used_cond_) {
            d_h = blocks_[i]->backward_with_cond(d_h, 0.0, last_cond_);
        } else {
            d_h = blocks_[i]->backward(d_h, 0.0);
        }
    }

    // Step 3: backward through first conv
    Tensor d_input = first_conv_.backward(d_h, 0.0);
    return d_input;
}

void PixelCNN::update_weights(double learning_rate) {
    first_conv_.update_weights(learning_rate);
    for (auto& block : blocks_) {
        block->update_weights(learning_rate);
    }
    classifier_.update_weights(learning_rate);
}

std::vector<Tensor*> PixelCNN::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&first_conv_.weights);
    p.push_back(&first_conv_.bias);
    for (auto& block : blocks_) {
        auto bp = block->parameters();
        for (auto* t : bp) p.push_back(t);
    }
    p.push_back(&classifier_.weights);
    p.push_back(&classifier_.bias);
    return p;
}

std::vector<Tensor*> PixelCNN::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&first_conv_.grad_weights);
    g.push_back(&first_conv_.grad_bias);
    for (auto& block : blocks_) {
        auto bg = block->gradients();
        for (auto* t : bg) g.push_back(t);
    }
    g.push_back(&classifier_.grad_weights);
    g.push_back(&classifier_.grad_bias);
    return g;
}

void PixelCNN::zero_grad() {
    first_conv_.zero_grad();
    for (auto& block : blocks_) {
        block->zero_grad();
    }
    classifier_.zero_grad();
}
