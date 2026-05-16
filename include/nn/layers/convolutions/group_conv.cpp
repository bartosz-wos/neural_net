#include "group_conv.h"
#include <cmath>
#include <stdexcept>

GroupConv::GroupConv(int in_channels, int out_channels, int kernel_size,
                     int num_groups, int stride, int padding, int dilation)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size),
      num_groups_(num_groups),
      stride_(stride), padding_(padding), dilation_(dilation),
      H_(0), W_(0), H_out_(0), W_out_(0)
{
    if (num_groups_ <= 0) {
        throw std::invalid_argument("GroupConv: num_groups must be positive");
    }
    if (in_channels_ % num_groups_ != 0) {
        throw std::invalid_argument("GroupConv: in_channels must be divisible by num_groups");
    }
    if (out_channels_ % num_groups_ != 0) {
        throw std::invalid_argument("GroupConv: out_channels must be divisible by num_groups");
    }
    if (kernel_size <= 0) {
        throw std::invalid_argument("GroupConv: kernel_size must be positive");
    }

    channels_per_group_ = in_channels_ / num_groups_;
    out_channels_per_group_ = out_channels_ / num_groups_;

    // Pre-create group Conv2D objects with placeholder H,W=1.
    // Actual dimensions are set on first forward.
    group_convs_.reserve(num_groups_);
    for (int g = 0; g < num_groups_; ++g) {
        group_convs_.emplace_back(
            channels_per_group_,
            out_channels_per_group_,
            kernel_size_, kernel_size_,
            1, 1,
            stride_, stride_,
            padding_, padding_,
            dilation, dilation
        );
    }
}

Tensor GroupConv::forward(const Tensor& input) {
    int N = input.rows;
    size_t total_cols = static_cast<size_t>(in_channels_) * static_cast<size_t>(H_) * static_cast<size_t>(W_);
    int spatial_size = input.cols / in_channels_;
    if (input.cols % in_channels_ != 0) {
        throw std::invalid_argument("GroupConv: input cols must be in_channels * H * W");
    }
    if (input.cols != total_cols) {
        // Infer H, W from spatial_size
        H_ = static_cast<int>(std::sqrt(spatial_size));
        W_ = spatial_size / H_;
        if (H_ * W_ != spatial_size) {
            H_ = 1;
            W_ = spatial_size;
        }
    }

    H_out_ = (H_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    W_out_ = (W_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    if (H_out_ <= 0 || W_out_ <= 0) {
        throw std::invalid_argument("GroupConv: output dimensions non-positive");
    }

    // Re-init group convs if H_, W_ were just set (i.e., first forward or new input size)
    if (group_convs_[0].H == 0) {
        for (int g = 0; g < num_groups_; ++g) {
            group_convs_[g] = Conv2D(
                channels_per_group_, out_channels_per_group_,
                kernel_size_, kernel_size_,
                H_, W_,
                stride_, stride_,
                padding_, padding_,
                dilation_, dilation_
            );
        }
    }

    last_input_ = input;
    last_group_outputs_.clear();
    last_group_outputs_.reserve(num_groups_);

    int spatial_per_group = H_ * W_;
    int group_input_size = channels_per_group_ * spatial_per_group;
    int group_output_size = out_channels_per_group_ * H_out_ * W_out_;

    Tensor output(N, out_channels_ * H_out_ * W_out_);

    for (int g = 0; g < num_groups_; ++g) {
        // Extract group's input channels
        Tensor group_input(N, group_input_size);
        int src_col = g * group_input_size;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < group_input_size; ++c) {
                group_input[n][c] = input[n][src_col + c];
            }
        }

        // Forward through this group's conv
        Tensor group_output = group_convs_[g].forward(group_input);
        last_group_outputs_.push_back(group_output);

        // Copy to output
        int dst_col_base = g * group_output_size;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < group_output_size; ++s) {
                output[n][dst_col_base + s] = group_output[n][s];
            }
        }
    }

    return output;
}

Tensor GroupConv::backward(const Tensor& grad_output, double learning_rate) {
    int N = grad_output.rows;
    int spatial_per_group = H_ * W_;
    int group_input_size = channels_per_group_ * spatial_per_group;
    int group_output_size = out_channels_per_group_ * H_out_ * W_out_;

    // grad_output: (N, out_channels * H_out * W_out)
    // Single backward pass: for each group, split grad, run backward, collect grad_input
    Tensor grad_input(N, in_channels_ * H_ * W_);
    grad_input.fill(0.0);

    for (int g = 0; g < num_groups_; ++g) {
        int src_col = g * group_output_size;
        Tensor grad_group(N, group_output_size);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < group_output_size; ++s) {
                grad_group[n][s] = grad_output[n][src_col + s];
            }
        }

        // Backward returns grad_wrt_input for this group
        // Also accumulates grad_weights/bias in the group's Conv2D
        Tensor grad_group_input = group_convs_[g].backward(grad_group, 0.0);

        // Copy grad_group_input to correct channel range in grad_input
        int dst_col_base = g * group_input_size;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < group_input_size; ++c) {
                grad_input[n][dst_col_base + c] = grad_group_input[n][c];
            }
        }
    }

    // Update all group weights
    for (int g = 0; g < num_groups_; ++g) {
        group_convs_[g].update_weights(learning_rate);
    }

    return grad_input;
}

void GroupConv::update_weights(double learning_rate) {
    for (int g = 0; g < num_groups_; ++g) {
        group_convs_[g].update_weights(learning_rate);
    }
}

void GroupConv::zero_grad() {
    for (int g = 0; g < num_groups_; ++g) {
        group_convs_[g].zero_grad();
    }
}

Tensor GroupConv::get_weights() const {
    if (num_groups_ == 0) return Tensor(0, 0);
    Tensor first = group_convs_[0].get_weights();
    int rows = first.rows;
    int cols = first.cols * num_groups_;
    Tensor result(rows, cols);
    result.fill(0.0);
    for (int g = 0; g < num_groups_; ++g) {
        Tensor w = group_convs_[g].get_weights();
        for (int i = 0; i < rows; ++i) {
            for (size_t j = 0; j < w.cols; ++j) {
                result[i][g * w.cols + j] = w[i][j];
            }
        }
    }
    return result;
}

Tensor GroupConv::get_gradients() const {
    if (num_groups_ == 0) return Tensor(0, 0);
    Tensor first = group_convs_[0].get_gradients();
    int rows = first.rows;
    int cols = first.cols * num_groups_;
    Tensor result(rows, cols);
    result.fill(0.0);
    for (int g = 0; g < num_groups_; ++g) {
        Tensor g_grad = group_convs_[g].get_gradients();
        for (int i = 0; i < rows; ++i) {
            for (size_t j = 0; j < g_grad.cols; ++j) {
                result[i][g * g_grad.cols + j] = g_grad[i][j];
            }
        }
    }
    return result;
}

std::vector<Tensor*> GroupConv::parameters() {
    std::vector<Tensor*> params;
    for (int g = 0; g < num_groups_; ++g) {
        auto p = group_convs_[g].parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

std::vector<Tensor*> GroupConv::gradients() {
    std::vector<Tensor*> grads;
    for (int g = 0; g < num_groups_; ++g) {
        auto g_grads = group_convs_[g].gradients();
        grads.insert(grads.end(), g_grads.begin(), g_grads.end());
    }
    return grads;
}