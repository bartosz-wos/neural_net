#include "weight_norm.h"
#include <cmath>

WeightNorm::WeightNorm(Layer* wrapped, bool learnable_scale)
    : wrapped_(wrapped), learnable_scale_(learnable_scale) {
    std::vector<Tensor*> params = wrapped_->parameters();
    if (params.empty()) return;
    v_ = *params[0]; // snapshot of original weights

    if (learnable_scale_) {
        size_t out_ch = v_.rows;
        g_ = Tensor(out_ch, 1);
        for (size_t i = 0; i < out_ch; ++i) g_[i][0] = 1.0;
    }
}

void WeightNorm::normalize_weights() {
    std::vector<Tensor*> params = wrapped_->parameters();
    if (params.empty()) return;
    Tensor& w = *params[0];

    size_t out_ch = v_.rows;
    for (size_t i = 0; i < out_ch; ++i) {
        double norm = 0.0;
        for (size_t j = 0; j < v_.cols; ++j)
            norm += v_[i][j] * v_[i][j];
        norm = std::sqrt(norm) + 1e-8;
        double g_i = learnable_scale_ ? g_[i][0] : 1.0;
        for (size_t j = 0; j < w.cols; ++j)
            w[i][j] = g_i * v_[i][j] / norm;
    }
}

Tensor WeightNorm::forward(const Tensor& input) {
    normalize_weights();
    return wrapped_->forward(input);
}

Tensor WeightNorm::backward(const Tensor& grad_output, double learning_rate) {
    return wrapped_->backward(grad_output, learning_rate);
}

void WeightNorm::update_weights(double learning_rate) {
    wrapped_->update_weights(learning_rate);
    std::vector<Tensor*> params = wrapped_->parameters();
    if (!params.empty()) v_ = *params[0]; // sync v after weight update
}

void WeightNorm::zero_grad() {
    wrapped_->zero_grad();
}