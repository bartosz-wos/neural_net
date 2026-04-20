#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

#include "../core/model.h"
#include <cmath>
#include <iostream>

// Numerical gradient checker for verifying layer implementations.
// Computes numerical gradient using finite differences and compares to analytical gradient.
// Returns max absolute difference — should be < tolerance for correct implementations.
struct GradientChecker {
    double epsilon = 1e-3;
    double tolerance = 1e-3;

    // Check a single layer in isolation (as if it were the only layer).
    // Provide a forward function that takes a Tensor and returns output.
    // Provide an analytical gradient function.
    double check_layer(
        const Tensor& input,
        const Tensor& weights,
        std::function<Tensor(const Tensor&, const Tensor&)> forward_fn,
        std::function<Tensor(const Tensor&, const Tensor&, const Tensor&)> grad_fn,
        const Tensor& grad_output
    );

    // Check a model's gradient end-to-end using MSE loss.
    // Perturbs each parameter individually, compares numerical vs analytical gradient.
    double check_model(
        Model& model,
        const Tensor& X,
        const Tensor& y,
        double eps = 1e-3
    );
};

double GradientChecker::check_layer(
    const Tensor& input,
    const Tensor& weights,
    std::function<Tensor(const Tensor&, const Tensor&)> forward_fn,
    std::function<Tensor(const Tensor&, const Tensor&, const Tensor&)> grad_fn,
    const Tensor& grad_output
) {
    // Compute analytical gradient
    Tensor grad_analytical = grad_fn(input, weights, grad_output);

    // Compute numerical gradient for each weight
    double max_diff = 0.0;
    for (size_t i = 0; i < weights.rows; ++i) {
        for (size_t j = 0; j < weights.cols; ++j) {
            double orig = weights[i][j];

            // f(w + eps)
            const_cast<Tensor&>(weights)[i][j] = orig + epsilon;
            Tensor out_plus = forward_fn(input, weights);

            // f(w - eps)
            const_cast<Tensor&>(weights)[i][j] = orig - epsilon;
            Tensor out_minus = forward_fn(input, weights);

            // restore
            const_cast<Tensor&>(weights)[i][j] = orig;

            // Numerical gradient
            double grad_num = 0.0;
            for (size_t n = 0; n < out_plus.rows; ++n) {
                for (size_t c = 0; c < out_plus.cols; ++c) {
                    grad_num += (out_plus[n][c] - out_minus[n][c]) / (2 * epsilon);
                }
            }

            double diff = std::abs(grad_num - grad_analytical[i][j]);
            max_diff = std::max(max_diff, diff);
        }
    }
    return max_diff;
}

double GradientChecker::check_model(
    Model& model,
    const Tensor& X,
    const Tensor& y,
    double eps
) {
    // Forward
    Tensor pred = model.forward(X);
    // MSE loss gradient w.r.t. prediction: 2/N * (pred - y)
    size_t N = X.rows;
    Tensor grad_loss = pred - y;
    grad_loss = grad_loss * (2.0 / static_cast<double>(N));

    // Backward
    model.backward(grad_loss, 0.0); // lr=0 so weights unchanged

    // Numerical check on first Dense layer
    Dense* dense = nullptr;
    for (auto& layer : model.layers) {
        dense = dynamic_cast<Dense*>(layer.get());
        if (dense) break;
    }
    if (!dense) return 0.0;

    double max_diff = 0.0;
    Tensor& W = dense->weights;
    for (size_t i = 0; i < W.rows; ++i) {
        for (size_t j = 0; j < W.cols; ++j) {
            double orig = W[i][j];

            // f(W + eps)
            W[i][j] = orig + eps;
            Tensor pred_plus = model.forward(X);
            double loss_plus = 0.0;
            for (size_t n = 0; n < N; ++n)
                for (size_t c = 0; c < pred_plus.cols; ++c)
                    loss_plus += (pred_plus[n][c] - y[n][c]) * (pred_plus[n][c] - y[n][c]);
            loss_plus /= static_cast<double>(N);

            // f(W - eps)
            W[i][j] = orig - eps;
            Tensor pred_minus = model.forward(X);
            double loss_minus = 0.0;
            for (size_t n = 0; n < N; ++n)
                for (size_t c = 0; c < pred_minus.cols; ++c)
                    loss_minus += (pred_minus[n][c] - y[n][c]) * (pred_minus[n][c] - y[n][c]);
            loss_minus /= static_cast<double>(N);

            // restore
            W[i][j] = orig;

            // Numerical gradient of loss w.r.t. weight
            double grad_num = (loss_plus - loss_minus) / (2 * eps);
            double grad_analytical = dense->grad_weights[i][j];
            double diff = std::abs(grad_num - grad_analytical);
            max_diff = std::max(max_diff, diff);
        }
    }
    return max_diff;
}

#endif
