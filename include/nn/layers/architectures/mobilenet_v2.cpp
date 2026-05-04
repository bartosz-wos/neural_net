#include "mobilenet_v2.h"
#include <cmath>

InvertedResidual::InvertedResidual(size_t in_channels, size_t out_channels, size_t stride,
                                    size_t expansion_factor, size_t H, size_t W)
    : in_channels_(in_channels), out_channels_(out_channels), stride_(stride),
      expansion_factor_(expansion_factor),
      expand_conv_(in_channels, in_channels * expansion_factor, 1, 1, H, W, 1, 1, 0, 0),
      depthwise_conv_(in_channels * expansion_factor, in_channels * expansion_factor,
                     3, 3, H, W, stride, stride, 1, 1),
      project_conv_(in_channels * expansion_factor, out_channels, 1, 1,
                    (H - 1) / stride + 1, (W - 1) / stride + 1, 1, 1, 0, 0),
      skip_connection_(in_channels == out_channels && stride == 1),
      H_out_((H - 1) / stride + 1), W_out_((H - 1) / stride + 1),
      last_output_(1, out_channels * H_out_ * W_out_) {}

Tensor InvertedResidual::forward(const Tensor& input) {
    // Expansion: 1x1 conv (in -> t*in)
    Tensor x = expand_conv_.forward(input);
    // ReLU6
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0); // ReLU6
    last_expand_relu_ = x;

    // Depthwise conv
    x = depthwise_conv_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0); // ReLU6
    last_depthwise_relu_ = x;

    // Project: 1x1 conv (t*in -> out), linear (no activation)
    last_output_ = project_conv_.forward(x);

    // Skip connection
    if (skip_connection_) {
        for (size_t i = 0; i < last_output_.rows; ++i)
            for (size_t j = 0; j < last_output_.cols; ++j)
                last_output_[i][j] += input[i][j];
    }
    return last_output_;
}

Tensor InvertedResidual::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, out_channels * H_out * W_out)
    // Forward: input → expand(ReLU6) → depthwise(ReLU6) → project → [+ input if skip]
    // Backward: grad_output → project back → depthwise back → expand back (+ identity if skip)

    if (skip_connection_) {
        // Skip: grad_output = grad_project + grad_identity
        // dL/d(project_input) from project backward
        // dL/d(input) from identity = grad_output directly

        // 1) Backprop project_conv (linear, no activation)
        Tensor grad_depthwise = project_conv_.backward(grad_output, learning_rate);

        // 2) Backprop depthwise ReLU6
        // dL/d(depthwise_raw) = dL/d(depthwise_relu) * relu6_mask
        // relu6_mask = 1 if 0 < raw < 6, else 0
        // raw output was last_depthwise_relu_ (after ReLU6), but we need pre-ReLU6 raw.
        // depthwise_conv_.last_input_ has raw output before ReLU6
        Tensor grad_depthwise_raw(depthwise_conv_.last_input.rows, depthwise_conv_.last_input.cols);
        for (size_t i = 0; i < depthwise_conv_.last_input.rows; ++i)
            for (size_t j = 0; j < depthwise_conv_.last_input.cols; ++j) {
                double raw = depthwise_conv_.last_input[i][j];
                grad_depthwise_raw[i][j] = (raw > 0.0 && raw < 6.0) ? grad_depthwise[i][j] : 0.0;
            }

        // 3) Backprop depthwise conv
        Tensor grad_expanded = depthwise_conv_.backward(grad_depthwise_raw, learning_rate);

        // 4) Backprop expand ReLU6
        // raw output was expand_conv_ output before ReLU6. Use expand_conv_.last_input_
        Tensor grad_expand_raw(expand_conv_.last_input.rows, expand_conv_.last_input.cols);
        for (size_t i = 0; i < expand_conv_.last_input.rows; ++i)
            for (size_t j = 0; j < expand_conv_.last_input.cols; ++j) {
                double raw = expand_conv_.last_input[i][j];
                grad_expand_raw[i][j] = (raw > 0.0 && raw < 6.0) ? grad_expanded[i][j] : 0.0;
            }

        // 5) Backprop expand_conv
        Tensor grad_input = expand_conv_.backward(grad_expand_raw, learning_rate);

        // 6) Add identity gradient (skip connection backprop)
        for (size_t i = 0; i < grad_input.rows; ++i)
            for (size_t j = 0; j < grad_input.cols; ++j)
                grad_input[i][j] += grad_output[i][j];

        return grad_input;
    } else {
        // No skip: straightforward backprop through project → depthwise → expand

        // 1) Backprop project_conv (linear)
        Tensor grad_depthwise = project_conv_.backward(grad_output, learning_rate);

        // 2) Apply depthwise ReLU6 mask
        Tensor grad_depthwise_raw(depthwise_conv_.last_input.rows, depthwise_conv_.last_input.cols);
        for (size_t i = 0; i < depthwise_conv_.last_input.rows; ++i)
            for (size_t j = 0; j < depthwise_conv_.last_input.cols; ++j) {
                double raw = depthwise_conv_.last_input[i][j];
                grad_depthwise_raw[i][j] = (raw > 0.0 && raw < 6.0) ? grad_depthwise[i][j] : 0.0;
            }

        // 3) Backprop depthwise conv
        Tensor grad_expanded = depthwise_conv_.backward(grad_depthwise_raw, learning_rate);

        // 4) Apply expand ReLU6 mask
        Tensor grad_expand_raw(expand_conv_.last_input.rows, expand_conv_.last_input.cols);
        for (size_t i = 0; i < expand_conv_.last_input.rows; ++i)
            for (size_t j = 0; j < expand_conv_.last_input.cols; ++j) {
                double raw = expand_conv_.last_input[i][j];
                grad_expand_raw[i][j] = (raw > 0.0 && raw < 6.0) ? grad_expanded[i][j] : 0.0;
            }

        // 5) Backprop expand_conv
        Tensor grad_input = expand_conv_.backward(grad_expand_raw, learning_rate);

        return grad_input;
    }
}

void InvertedResidual::update_weights(double learning_rate) {
    expand_conv_.update_weights(learning_rate);
    depthwise_conv_.update_weights(learning_rate);
    project_conv_.update_weights(learning_rate);
}

void InvertedResidual::zero_grad() {
    expand_conv_.zero_grad();
    depthwise_conv_.zero_grad();
    project_conv_.zero_grad();
}

std::vector<Tensor*> InvertedResidual::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : expand_conv_.parameters()) result.push_back(p);
    for (Tensor* p : depthwise_conv_.parameters()) result.push_back(p);
    for (Tensor* p : project_conv_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> InvertedResidual::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : expand_conv_.gradients()) result.push_back(g);
    for (Tensor* g : depthwise_conv_.gradients()) result.push_back(g);
    for (Tensor* g : project_conv_.gradients()) result.push_back(g);
    return result;
}

// === MobileNetV2 ===

MobileNetV2::MobileNetV2(size_t num_classes, double width_multiplier, size_t H_in, size_t W_in)
    : width_mult_(width_multiplier),
      first_conv_(3, (size_t)(32 * width_multiplier), 3, 3, H_in, W_in, 2, 2, 1, 1),
      final_conv_((size_t)(1280 * width_multiplier), num_classes, 1, 1, 7, 7, 1, 1, 0, 0),
      classifier_(1280, num_classes),
      last_output_(1, num_classes) {

    size_t H = (H_in - 3) / 2 + 1;
    size_t W = (W_in - 3) / 2 + 1;

    // Inverted residual config: [t, c, n, s]
    // t=expansion factor, c=output channels, n=num blocks, s=stride
    struct { size_t t, c, n, s; } config[] = {
        {1, 16, 1, 1},   // conv2
        {6, 24, 2, 2},   // conv3
        {6, 32, 3, 2},   // conv4
        {6, 64, 4, 2},   // conv5
        {6, 96, 3, 1},   // conv6
        {6, 160, 3, 2},  // conv7
        {6, 320, 1, 1},   // conv8
    };

    size_t in_ch = (size_t)(32 * width_multiplier);
    for (auto& cfg : config) {
        size_t out_ch = (size_t)(cfg.c * width_multiplier);
        for (size_t i = 0; i < cfg.n; ++i) {
            size_t stride = (i == 0) ? cfg.s : 1;
            residual_blocks_.emplace_back(in_ch, out_ch, stride, cfg.t, H, W);
            in_ch = out_ch;
            if (cfg.s == 2 && i == 0) {
                H = (H - 3) / 2 + 1;
                W = (W - 3) / 2 + 1;
            }
        }
    }
}

Tensor MobileNetV2::forward(const Tensor& input) {
    Tensor x = first_conv_.forward(input);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0);

    for (auto& blk : residual_blocks_)
        x = blk.forward(x);

    // Final conv + ReLU6
    x = final_conv_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0);
    last_final_relu_ = x;

    // Flatten
    size_t batch = x.rows;
    last_flat_ = Tensor(batch, x.cols);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            last_flat_[i][j] = x[i][j];

    last_output_ = classifier_.forward(last_flat_);
    return last_output_;
}

Tensor MobileNetV2::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, num_classes)
    // Backward: classifier → flatten → final_conv → residual_blocks → first_conv
    // Forward: input → first_conv(ReLU6) → residual_blocks → final_conv(ReLU6) → flatten → classifier

    // 1) Classifier backward
    // classifier_.forward took last_flat_ as input
    // classifier_.backward computes grad w.r.t. last_flat_ (flattened features)
    Tensor grad_flat = classifier_.backward(grad_output, learning_rate);

    // 2) Flatten is pass-through (identity), grad_flat is same shape as last_flat_
    size_t batch = grad_output.rows;

    // 3) ReLU6 backward for final conv output
    // dL/d(final_conv_out) = grad_flat
    // d(out)/d(x) = 1 if 0 < x < 6, else 0
    // first_conv_.last_input_ stores raw output before ReLU6
    Tensor grad_final_conv(batch, grad_flat.cols);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < grad_flat.cols; ++j) {
            double v = last_final_relu_[i][j];
            grad_final_conv[i][j] = (v > 0.0 && v < 6.0) ? grad_flat[i][j] : 0.0;
        }

    // 4) Final conv backward
    Tensor grad_after_blocks = final_conv_.backward(grad_final_conv, learning_rate);

    // 5) Residual blocks backward (in reverse order)
    for (auto it = residual_blocks_.rbegin(); it != residual_blocks_.rend(); ++it)
        grad_after_blocks = it->backward(grad_after_blocks, learning_rate);

    // 6) First conv ReLU6 backward
    // first_conv_.last_input_ is raw conv output (before ReLU6)
    // Apply mask: gradient only flows where raw > 0
    const Tensor& raw_first_conv = first_conv_.last_input;
    Tensor grad_first_conv(raw_first_conv.rows, raw_first_conv.cols);
    for (size_t i = 0; i < raw_first_conv.rows; ++i)
        for (size_t j = 0; j < raw_first_conv.cols; ++j)
            grad_first_conv[i][j] = raw_first_conv[i][j] > 0.0 ? grad_after_blocks[i][j] : 0.0;

    // 7) First conv backward (returns gradient w.r.t. input)
    Tensor grad_input = first_conv_.backward(grad_first_conv, learning_rate);

    return grad_input;
}

void MobileNetV2::update_weights(double learning_rate) {
    first_conv_.update_weights(learning_rate);
    final_conv_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
    for (auto& blk : residual_blocks_)
        blk.update_weights(learning_rate);
}

void MobileNetV2::zero_grad() {
    first_conv_.zero_grad();
    final_conv_.zero_grad();
    classifier_.zero_grad();
    for (auto& blk : residual_blocks_)
        blk.zero_grad();
}

std::vector<Tensor*> MobileNetV2::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : first_conv_.parameters()) result.push_back(p);
    for (Tensor* p : final_conv_.parameters()) result.push_back(p);
    for (Tensor* p : classifier_.parameters()) result.push_back(p);
    for (auto& blk : residual_blocks_)
        for (Tensor* p : blk.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> MobileNetV2::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : first_conv_.gradients()) result.push_back(g);
    for (Tensor* g : final_conv_.gradients()) result.push_back(g);
    for (Tensor* g : classifier_.gradients()) result.push_back(g);
    for (auto& blk : residual_blocks_)
        for (Tensor* g : blk.gradients())
            result.push_back(g);
    return result;
}
