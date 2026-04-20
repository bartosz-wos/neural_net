#include "layer.h"
#include <random>
#include <cmath>

Dense::Dense(size_t in_features, size_t out_features)
    : weights(out_features, in_features), bias(1, out_features),
      grad_weights(out_features, in_features), grad_bias(1, out_features),
      last_input(0, 0)  // default-constructed empty tensor
{
    init_weights("xavier");
}

void Dense::init_weights(const std::string& scheme) {
    size_t fan_in = weights.cols;
    size_t fan_out = weights.rows;
    std::mt19937 gen(42);

    if (scheme == "he") {
        // He (Kaiming) initialization — optimal for ReLU networks
        double std = std::sqrt(2.0 / fan_in);
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < fan_out; ++i)
            for (size_t j = 0; j < fan_in; ++j)
                weights[i][j] = dis(gen);
    } else if (scheme == "uniform") {
        // Uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)]
        double bound = 1.0 / std::sqrt(fan_in);
        std::uniform_real_distribution<> dis(-bound, bound);
        for (size_t i = 0; i < fan_out; ++i)
            for (size_t j = 0; j < fan_in; ++j)
                weights[i][j] = dis(gen);
    } else if (scheme == "zeros") {
        weights.fill(0.0);
    } else {
        // Default: Xavier/Glorot
        double std = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < fan_out; ++i)
            for (size_t j = 0; j < fan_in; ++j)
                weights[i][j] = dis(gen);
    }
    // Bias always zero-initialized
    bias.fill(0.0);
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

Tensor Dense::forward(const Tensor& input) {
    last_input = input;  // cache for backward
    Tensor result = input * weights.transpose();
    // Manually broadcast bias (1, out_features) to each batch row
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result[i][j] += bias[0][j];
        }
    }
    return result;
}

Tensor Dense::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (batch, out_features)
    // Compute gradient w.r.t. weights: grad_output^T * input  -> (out, in)
    Tensor grad_w = grad_output.transpose() * last_input;  // (out, batch) * (batch, in) = (out, in)

    // Accumulate gradients
    grad_weights = grad_weights + grad_w;

    // Compute gradient w.r.t. bias: sum over batch
    Tensor grad_b = Tensor(1, grad_output.cols);
    for (size_t j = 0; j < grad_output.cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_output.rows; ++i) {
            sum += grad_output[i][j];
        }
        grad_b[0][j] = sum;
    }
    grad_bias = grad_bias + grad_b;

    // Compute gradient w.r.t. input for next layer: grad_output * weights
    Tensor grad_input = grad_output * weights;
    return grad_input;
}

void Dense::update_weights(double learning_rate) {
    size_t in_f = weights.cols;
    size_t out_f = weights.rows;
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            weights[i][j] -= learning_rate * grad_weights[i][j];
        }
        bias[0][i] -= learning_rate * grad_bias[0][i];
    }
    // NOTE: gradients NOT zeroed here; optimizer or caller must zero via zero_grad()
}

std::vector<Tensor*> Dense::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Dense::gradients() {
    return {&grad_weights, &grad_bias};
}

void Dense::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}
