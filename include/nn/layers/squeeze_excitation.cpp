#include "squeeze_excitation.h"
#include <cmath>

SEBlock::SEBlock(size_t in_channels, size_t reduction)
    : fc1_(in_channels, std::max(in_channels / reduction, size_t(1))),
      fc2_(std::max(in_channels / reduction, size_t(1)), in_channels),
      in_channels_(in_channels), reduction_(reduction),
      last_excitation_(1, in_channels) {}

Tensor SEBlock::forward(const Tensor& input) {
    last_input_ = input;
    size_t batch = input.rows;
    size_t channels = in_channels_;
    size_t spatial = input.cols / channels;

    // Global average pooling: squeeze each channel to a scalar
    Tensor gap(batch, channels);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t s = 0; s < spatial; ++s)
                sum += input[b][c * spatial + s];
            gap[b][c] = sum / spatial;
        }
    }

    // FC → ReLU → FC → Sigmoid
    Tensor z = fc1_.forward(gap);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t j = 0; j < z.cols; ++j)
            z[i][j] = std::max(0.0, z[i][j]);

    z = fc2_.forward(z);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t j = 0; j < z.cols; ++j)
            z[i][j] = 1.0 / (1.0 + std::exp(-z[i][j])); // sigmoid

    last_excitation_ = z; // (batch, channels)
    return last_excitation_;
}

Tensor SEBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void SEBlock::update_weights(double learning_rate) {
    fc1_.update_weights(learning_rate);
    fc2_.update_weights(learning_rate);
}

void SEBlock::zero_grad() {
    fc1_.zero_grad();
    fc2_.zero_grad();
}

std::vector<Tensor*> SEBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> SEBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    return result;
}

Tensor SEBlock::scale_channels(const Tensor& input, const Tensor& excitation) {
    size_t batch = input.rows;
    size_t channels = excitation.cols;
    size_t spatial = input.cols / channels;

    Tensor output(batch, input.cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double w = excitation[b][c];
            for (size_t s = 0; s < spatial; ++s)
                output[b][c * spatial + s] = input[b][c * spatial + s] * w;
        }
    }
    return output;
}

// === SEResNetBlock ===

SEResNetBlock::SEResNetBlock(size_t in_channels, size_t out_channels,
                               size_t reduction, size_t H, size_t W)
    : conv1_(in_channels, out_channels, 3, 3, H, W, 1, 1, 1, 1),
      conv2_(out_channels, out_channels, 3, 3, H, W, 1, 1, 1, 1),
      se_(out_channels, reduction),
      skip_conv_(in_channels, out_channels, 1, 1, H, W, 1, 1, 0, 0),
      has_skip_(in_channels != out_channels),
      H_(H), W_(W),
      last_output_(1, out_channels * H * W) {}

Tensor SEResNetBlock::forward(const Tensor& input) {
    Tensor x = conv1_.forward(input);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);

    x = conv2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);

    // SE excitation + channel scaling
    x = SEBlock::scale_channels(x, se_.forward(x));

    // Skip connection
    if (has_skip_) {
        Tensor skip = skip_conv_.forward(input);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j)
                x[i][j] += skip[i][j];
    } else {
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j)
                x[i][j] += input[i][j];
    }

    last_output_ = x;
    return last_output_;
}

Tensor SEResNetBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void SEResNetBlock::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv2_.update_weights(learning_rate);
    se_.update_weights(learning_rate);
    skip_conv_.update_weights(learning_rate);
}

void SEResNetBlock::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
    se_.zero_grad();
    skip_conv_.zero_grad();
}

std::vector<Tensor*> SEResNetBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    for (Tensor* p : se_.parameters()) result.push_back(p);
    for (Tensor* p : skip_conv_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> SEResNetBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv2_.gradients()) result.push_back(g);
    for (Tensor* g : se_.gradients()) result.push_back(g);
    for (Tensor* g : skip_conv_.gradients()) result.push_back(g);
    return result;
}