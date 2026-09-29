#include "layer_scale.h"
#include <stdexcept>

// LayerScale (Cai et al., ICCV 2022)
//
// Element-wise per-channel multiplicative gate:
//   out[i, j] = lambda[j] * x[i, j]
//
// Backward:
//   grad_x[i, j]     = grad_out[i, j] * lambda[j]
//   grad_lambda[j]   = Σ_i grad_out[i, j] * x[i, j]
//
// The init_value parameter lets callers pick a small constant (paper default
// 1e-4) so the residual branch starts near-identity.

LayerScale::LayerScale(size_t features, double init_value)
    : lambda_(Tensor(1, features))
    , features_(features)
    , init_value_(init_value)
{
    if (features == 0) {
        throw std::invalid_argument("LayerScale: features must be > 0");
    }
    if (init_value < 0.0) {
        throw std::invalid_argument("LayerScale: init_value must be >= 0");
    }

    lambda_.fill(init_value);

    // Init grad buffer
    grad_lambda_ = Tensor(1, features);
    grad_lambda_.fill(0.0);
}

Tensor LayerScale::forward(const Tensor& input) {
    const size_t batch    = input.rows;
    const size_t features = input.cols;
    if (features != features_) {
        throw std::invalid_argument("LayerScale::forward: feature dim mismatch (input "
                                    + std::to_string(features) + " vs "
                                    + std::to_string(features_) + ")");
    }

    last_input = input;
    Tensor output(batch, features);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < features; ++j) {
            output[i][j] = lambda_[0][j] * input[i][j];
        }
    }
    return output;
}

Tensor LayerScale::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t batch    = grad_output.rows;
    const size_t features = grad_output.cols;

    // Lazy init (matches LayerNorm / DynamicTanh convention)
    if (grad_lambda_.rows == 0) {
        grad_lambda_ = Tensor(1, features);
        grad_lambda_.fill(0.0);
    }

    grad_x = Tensor(batch, features);

    // Per-channel gradient: grad_lambda[j] = Σ_i grad_out[i, j] * x[i, j]
    // Reset accumulator on this backward call (LayerScale's grad_lambda is a
    // per-call overwrite, not accumulate — matching the DynamicTanh alpha
    // scalar behavior. zero_grad() must be called between backward+update cycles).
    for (size_t j = 0; j < features; ++j) {
        grad_lambda_[0][j] = 0.0;
    }

    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < features; ++j) {
            const double go = grad_output[i][j];
            grad_x[i][j]    = go * lambda_[0][j];
            grad_lambda_[0][j] += go * last_input[i][j];
        }
    }

    return grad_x;
}

void LayerScale::update_weights(double learning_rate) {
    for (size_t j = 0; j < features_; ++j) {
        lambda_[0][j] -= learning_rate * grad_lambda_[0][j];
    }
}

std::vector<Tensor*> LayerScale::parameters() {
    return {&lambda_};
}

std::vector<Tensor*> LayerScale::gradients() {
    return {&grad_lambda_};
}

void LayerScale::zero_grad() {
    grad_lambda_.fill(0.0);
}