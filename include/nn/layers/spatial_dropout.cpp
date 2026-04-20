#include "spatial_dropout.h"
#include <random>

static std::mt19937 rng_(42);

void Dropout1D::apply_mask(Tensor& input) {
    if (!training_ || p_ == 0.0) return;
    // Input: (batch, seq_len * channels) or (batch, features)
    // We treat features as channels of size seq_len each
    // Simplified: treat entire input vector as channels, drop with probability p
    size_t total = input.cols;
    mask_.resize(input.rows);
    for (size_t i = 0; i < input.rows; ++i) {
        mask_[i].resize(total);
        for (size_t j = 0; j < total; ++j) {
            mask_[i][j] = (rng_() / (double)RAND_MAX < p_);
            if (mask_[i][j]) input[i][j] = 0.0;
        }
    }
}

Tensor Dropout1D::forward(const Tensor& input) {
    last_output_ = input;
    if (!training_ || p_ == 0.0) return last_output_;
    last_output_ = input; // copy
    apply_mask(last_output_);
    // Scale during training so inference is identity
    double scale = 1.0 / (1.0 - p_);
    for (size_t i = 0; i < last_output_.rows; ++i)
        for (size_t j = 0; j < last_output_.cols; ++j)
            last_output_[i][j] *= scale;
    return last_output_;
}

Tensor Dropout1D::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    Tensor grad = grad_output;
    if (!training_ || p_ == 0.0) return grad;
    double scale = 1.0 / (1.0 - p_);
    for (size_t i = 0; i < grad.rows; ++i)
        for (size_t j = 0; j < grad.cols; ++j)
            if (mask_[i][j]) grad[i][j] = 0.0;
            else grad[i][j] *= scale;
    return grad;
}

void Dropout2D::build_mask(size_t channels) {
    n_channels_ = channels;
    channel_mask_.resize(channels);
    for (size_t c = 0; c < channels; ++c)
        channel_mask_[c] = (rng_() / (double)RAND_MAX < p_);
}

Tensor Dropout2D::forward(const Tensor& input) {
    last_output_ = input;
    if (!training_ || p_ == 0.0) return last_output_;
    last_output_ = input;
    // Infer (channels, spatial_size) — try to detect square spatial dims
    // Stored as (batch, channels * spatial)
    // We need to know spatial size to reconstruct 2D structure
    // Heuristic: assume channels=1, spatial=cols (flattened)
    size_t channels = 1;
    size_t spatial = input.cols;
    build_mask(channels);
    double scale = 1.0 / (1.0 - p_);
    for (size_t i = 0; i < last_output_.rows; ++i)
        for (size_t c = 0; c < channels; ++c)
            if (channel_mask_[c])
                for (size_t s = 0; s < spatial; ++s)
                    last_output_[i][c * spatial + s] = 0.0;
            else
                for (size_t s = 0; s < spatial; ++s)
                    last_output_[i][c * spatial + s] *= scale;
    return last_output_;
}

Tensor Dropout2D::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    Tensor grad = grad_output;
    if (!training_ || p_ == 0.0) return grad;
    size_t channels = 1, spatial = grad.cols;
    double scale = 1.0 / (1.0 - p_);
    for (size_t i = 0; i < grad.rows; ++i)
        for (size_t c = 0; c < channels; ++c)
            if (channel_mask_[c])
                for (size_t s = 0; s < spatial; ++s)
                    grad[i][c * spatial + s] = 0.0;
            else
                for (size_t s = 0; s < spatial; ++s)
                    grad[i][c * spatial + s] *= scale;
    return grad;
}