#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <vector>
#include <memory>
#include <string>

// Abstract base layer
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output, double learning_rate) = 0;
    virtual void update_weights(double learning_rate) = 0;
    virtual Tensor get_weights() const = 0;
    virtual Tensor get_gradients() const = 0;
    virtual std::vector<Tensor*> parameters() = 0;
    virtual std::vector<Tensor*> gradients() = 0;
    virtual void zero_grad() = 0;
    // Optional: name for debugging/logging. Default returns "Layer"
    virtual std::string name() const { return "Layer"; }
};

// Dense (fully connected) layer: y = xW^T + b
class Dense : public Layer {
public:
    Tensor weights;  // shape: (out_features, in_features)
    Tensor bias;     // shape: (1, out_features)
    Tensor grad_weights;
    Tensor grad_bias;
    Tensor last_input;  // cached for backward pass

    Dense(size_t in_features, size_t out_features);
    // Reinitialize weights with scheme: "xavier" (default), "he" (Kaiming/He for ReLU), "uniform", "zeros"
    void init_weights(const std::string& scheme = "xavier");
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

// Templated Activation layer: F must provide operator()(double) and derivative(double)
template<typename F>
class Activation : public Layer {
    F func;
    Tensor last_input;
public:
    explicit Activation(F f) : func(f) {}
    Tensor forward(const Tensor& input) override {
        last_input = input;
        return input.apply(func);
    }
    Tensor backward(const Tensor& grad_output, double /* learning_rate */) override {
        Tensor result(grad_output.rows, grad_output.cols);
        for (size_t i = 0; i < grad_output.rows; ++i) {
            for (size_t j = 0; j < grad_output.cols; ++j) {
                result[i][j] = grad_output[i][j] * func.derivative(last_input[i][j]);
            }
        }
        return result;
    }
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
};

#endif
