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
      H_out_((H - 1) / stride + 1), W_out_((W - 1) / stride + 1),
      last_output_(1, out_channels * H_out_ * W_out_) {}

Tensor InvertedResidual::forward(const Tensor& input) {
    // Expansion: 1x1 conv (in -> t*in)
    Tensor x = expand_conv_.forward(input);
    // ReLU6
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0); // ReLU6

    // Depthwise conv
    x = depthwise_conv_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0);

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
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
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

    // Final conv + avg pool + classifier
    x = final_conv_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::min(std::max(0.0, x[i][j]), 6.0);

    // Flatten
    size_t batch = x.rows;
    Tensor flat(batch, x.cols);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            flat[i][j] = x[i][j];

    last_output_ = classifier_.forward(flat);
    return last_output_;
}

Tensor MobileNetV2::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
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