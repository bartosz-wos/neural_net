#include "depthwise_separable_conv.h"
#include <cmath>
#include <stdexcept>

DepthwiseSeparableConv::DepthwiseSeparableConv(int in_channels, int out_channels,
                                               int kernel_size, int stride,
                                               int padding, int dilation)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), H_(0), W_(0), H_out_(0), W_out_(0)
{
    if (in_channels <= 0 || out_channels <= 0) {
        throw std::invalid_argument("DepthwiseSeparableConv: channels must be positive");
    }
    if (kernel_size <= 0) {
        throw std::invalid_argument("DepthwiseSeparableConv: kernel_size must be positive");
    }
    // Depthwise convs and pointwise conv initialized lazily on first forward
}

void DepthwiseSeparableConv::ensure_initialized() {
    if (!depthwise_conv_.empty()) return;

    if (H_ <= 0 || W_ <= 0) {
        throw std::runtime_error("DepthwiseSeparableConv: spatial dimensions not set");
    }

    H_out_ = (H_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    W_out_ = (W_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    if (H_out_ <= 0 || W_out_ <= 0) {
        throw std::invalid_argument("DepthwiseSeparableConv: output dimensions non-positive");
    }

    // Create one Conv2D per channel for depthwise: each has in_ch=1, out_ch=1
    depthwise_conv_.clear();
    depthwise_conv_.reserve(in_channels_);
    for (int c = 0; c < in_channels_; ++c) {
        depthwise_conv_.emplace_back(
            1, 1,
            kernel_size_, kernel_size_,
            H_, W_,
            stride_, stride_,
            padding_, padding_,
            dilation_, dilation_
        );
    }

    // Pointwise conv: 1x1 conv from in_channels_ -> out_channels_
    pointwise_conv_ = Conv2D(
        in_channels_, out_channels_,
        1, 1,
        H_out_, W_out_,
        1, 1,
        0, 0,
        1, 1
    );
}

Tensor DepthwiseSeparableConv::forward(const Tensor& input) {
    int N = input.rows;
    int spatial_size = input.cols / in_channels_;
    if (input.cols % in_channels_ != 0) {
        throw std::invalid_argument("DepthwiseSeparableConv: input cols must be in_channels * H * W");
    }

    H_ = static_cast<int>(std::sqrt(spatial_size));
    W_ = spatial_size / H_;
    if (H_ * W_ != spatial_size) {
        H_ = 1;
        W_ = spatial_size;
    }

    last_input_ = input;
    ensure_initialized();

    // Stage 1: Depthwise — per-channel spatial convolution
    last_depthwise_output_ = Tensor(N, in_channels_ * H_out_ * W_out_);

    for (int c = 0; c < in_channels_; ++c) {
        // Extract channel c: (N, H_*W_)
        Tensor ch_input(N, H_ * W_);
        int col_base = c * H_ * W_;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < H_ * W_; ++s) {
                ch_input[n][s] = input[n][col_base + s];
            }
        }

        // Forward depthwise conv: (N, H_*W_) -> (N, H_out_*W_out_)
        Tensor ch_output = depthwise_conv_[c].forward(ch_input);

        // Copy to depthwise output
        int dst_col = c * H_out_ * W_out_;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < H_out_ * W_out_; ++s) {
                last_depthwise_output_[n][dst_col + s] = ch_output[n][s];
            }
        }
    }

    // Stage 2: Pointwise — 1x1 conv to mix channels
    Tensor output = pointwise_conv_.forward(last_depthwise_output_);
    return output;
}

Tensor DepthwiseSeparableConv::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;

    // Backward through pointwise conv (lr=0 — update later)
    Tensor grad_depthwise_out = pointwise_conv_.backward(grad_output, 0.0);

    // Backward through depthwise stage: get per-channel grad_inputs
    Tensor grad_input(N, in_channels_ * H_ * W_);
    grad_input.fill(0.0);

    for (int c = 0; c < in_channels_; ++c) {
        int col_base = c * H_out_ * W_out_;
        Tensor grad_ch(N, H_out_ * W_out_);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < H_out_ * W_out_; ++s) {
                grad_ch[n][s] = grad_depthwise_out[n][col_base + s];
            }
        }

        // Backward through depthwise conv c — captures grad_input, accumulates weight grads
        Tensor grad_ch_input = depthwise_conv_[c].backward(grad_ch, 0.0);

        // Copy to grad_input
        int dst_col = c * H_ * W_;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < H_ * W_; ++s) {
                grad_input[n][dst_col + s] = grad_ch_input[n][s];
            }
        }
    }

    // Update all weights
    pointwise_conv_.update_weights(learning_rate);
    for (int c = 0; c < in_channels_; ++c) {
        depthwise_conv_[c].update_weights(learning_rate);
    }

    return grad_input;
}

void DepthwiseSeparableConv::update_weights(double learning_rate) {
    pointwise_conv_.update_weights(learning_rate);
    for (int c = 0; c < in_channels_; ++c) {
        depthwise_conv_[c].update_weights(learning_rate);
    }
}

void DepthwiseSeparableConv::zero_grad() {
    pointwise_conv_.zero_grad();
    for (int c = 0; c < in_channels_; ++c) {
        depthwise_conv_[c].zero_grad();
    }
}

Tensor DepthwiseSeparableConv::get_weights() const {
    if (in_channels_ == 0) return Tensor(0, 0);
    // Depthwise: stack per-channel kernels (in_channels, kernel_size^2)
    Tensor dw(in_channels_, kernel_size_ * kernel_size_);
    for (int c = 0; c < in_channels_; ++c) {
        Tensor w = depthwise_conv_[c].get_weights();
        for (size_t i = 0; i < w.cols; ++i) {
            dw[c][i] = w[0][i];
        }
    }
    // Pointwise: (out_channels, in_channels)
    Tensor pw = pointwise_conv_.get_weights();
    int max_cols = std::max(dw.cols, pw.cols);
    Tensor result(in_channels_ + out_channels_, max_cols);
    result.fill(0.0);
    for (int i = 0; i < dw.rows; ++i)
        for (size_t j = 0; j < dw.cols; ++j)
            result[i][j] = dw[i][j];
    for (int i = 0; i < pw.rows; ++i)
        for (size_t j = 0; j < pw.cols; ++j)
            result[dw.rows + i][j] = pw[i][j];
    return result;
}

Tensor DepthwiseSeparableConv::get_gradients() const {
    if (in_channels_ == 0) return Tensor(0, 0);
    Tensor dw(in_channels_, kernel_size_ * kernel_size_);
    for (int c = 0; c < in_channels_; ++c) {
        Tensor g = depthwise_conv_[c].get_gradients();
        for (size_t i = 0; i < g.cols; ++i) {
            dw[c][i] = g[0][i];
        }
    }
    Tensor pw = pointwise_conv_.get_gradients();
    int max_cols = std::max(dw.cols, pw.cols);
    Tensor result(in_channels_ + out_channels_, max_cols);
    result.fill(0.0);
    for (int i = 0; i < dw.rows; ++i)
        for (size_t j = 0; j < dw.cols; ++j)
            result[i][j] = dw[i][j];
    for (int i = 0; i < pw.rows; ++i)
        for (size_t j = 0; j < pw.cols; ++j)
            result[dw.rows + i][j] = pw[i][j];
    return result;
}

std::vector<Tensor*> DepthwiseSeparableConv::parameters() {
    std::vector<Tensor*> params;
    for (int c = 0; c < in_channels_; ++c) {
        auto p = depthwise_conv_[c].parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    auto pp = pointwise_conv_.parameters();
    params.insert(params.end(), pp.begin(), pp.end());
    return params;
}

std::vector<Tensor*> DepthwiseSeparableConv::gradients() {
    std::vector<Tensor*> grads;
    for (int c = 0; c < in_channels_; ++c) {
        auto g = depthwise_conv_[c].gradients();
        grads.insert(grads.end(), g.begin(), g.end());
    }
    auto pg = pointwise_conv_.gradients();
    grads.insert(grads.end(), pg.begin(), pg.end());
    return grads;
}