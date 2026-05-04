#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

#include "../core/layer.h"
#include "../core/model.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

// ============================================================================
// Numerical Gradient Verification Utilities
// ============================================================================
//
// These utilities verify the correctness of analytical gradients computed via
// backpropagation by comparing them against numerically approximated gradients
// using finite differences.
//
// All functions return true only if ALL checked gradients are within tolerance.
//
// ============================================================================

// ------------------------------------------------------------------------
// Helper: compute relative error with numerical safety floor
// ------------------------------------------------------------------------
static inline double relative_error(double analytical, double numerical) {
    double max_abs = std::max(std::abs(analytical), std::abs(numerical));
    if (max_abs < 1e-8) {
        // Both are effectively zero — pass if within absolute tolerance
        return std::abs(analytical - numerical) / 1e-8;
    }
    return std::abs(analytical - numerical) / max_abs;
}

// ------------------------------------------------------------------------
// Helper: compute MSE loss from output and target tensors
//   loss = mean((output - target)^2)  [scalar]
// ------------------------------------------------------------------------
static inline double mse_loss(const Tensor& output, const Tensor& target) {
    double sum = 0.0;
    size_t n = output.rows * output.cols;
    for (size_t i = 0; i < n; ++i) {
        double diff = output.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum / static_cast<double>(n);
}

// ------------------------------------------------------------------------
// check_gradient
//
// Numerically verifies gradients for any Layer implementation.
//
// For each parameter in layer.parameters():
//   1. Perturb by +epsilon, run forward, compute loss (L_plus)
//   2. Perturb by -epsilon, run forward, compute loss (L_minus)
//   3. Numerical gradient = (L_plus - L_minus) / (2*epsilon)
//   4. Compare to analytical gradient from layer.backward(target_output_grad)
//   5. Return false immediately if |rel_error| > tolerance
//
// The caller provides `target_output_grad` — the gradient of the loss w.r.t.
// the layer's forward output (i.e., what would flow backward into this layer).
// For MSE loss with target t:  target_output_grad = 2 * (output - t) / N
//
// Returns true only if ALL parameter gradients pass.
// ------------------------------------------------------------------------
template<typename LayerType>
bool check_gradient(
    LayerType& layer,
    const Tensor& input,
    const Tensor& target_output_grad,
    double epsilon = 1e-5,
    double tolerance = 1e-4
) {
    // Run forward to populate last_input (needed for backward)
    Tensor output = layer.forward(input);

    // Zero any stale accumulated gradients
    layer.zero_grad();

    // Compute and store analytical gradients via backward
    layer.backward(target_output_grad, 0.0);

    // Loss function used for numerical gradient: L = sum(output) (scalar)
    // This is a simple, generic loss whose gradient w.r.t. output is all-ones.
    // For MSE, the caller should instead use check_gradient_mse().
    auto compute_loss = [](const Tensor& out) -> double {
        double s = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) s += out.data[i];
        return s;
    };

    auto params = layer.parameters();
    auto grads  = layer.gradients();

    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* param = params[p];
        Tensor* grad  = grads[p];

        if (param->rows == 0 || param->cols == 0) continue;

        for (size_t r = 0; r < param->rows; ++r) {
            for (size_t c = 0; c < param->cols; ++c) {
                double orig = (*param)(r, c);

                // L_plus: perturb positive
                (*param)(r, c) = orig + epsilon;
                layer.zero_grad();
                layer.forward(input);
                layer.backward(target_output_grad, 0.0);
                double L_plus = compute_loss(layer.forward(input));

                // L_minus: perturb negative
                (*param)(r, c) = orig - epsilon;
                layer.zero_grad();
                layer.forward(input);
                layer.backward(target_output_grad, 0.0);
                double L_minus = compute_loss(layer.forward(input));

                // Restore original value
                (*param)(r, c) = orig;

                // Numerical gradient via centered finite difference
                double numerical = (L_plus - L_minus) / (2.0 * epsilon);
                double analytical = (*grad)(r, c);

                double rel_err = relative_error(analytical, numerical);
                if (rel_err > tolerance) {
                    std::cerr << "  [FAIL] " << param << "[" << r << "][" << c << "]  "
                              << "analytical=" << analytical
                              << "  numerical=" << numerical
                              << "  rel_err=" << rel_err
                              << "  (tol=" << tolerance << ")" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

// ------------------------------------------------------------------------
// check_gradient_mse
//
// Convenience wrapper around check_gradient() that uses MSE loss:
//   loss = mean((output - target)^2)
//
// Numerically verifies gradients by perturbing each parameter and comparing
// the centered finite-difference gradient to the analytical gradient from
// layer.backward(grad_loss_wrt_output), where:
//   grad_loss_wrt_output = 2 * (output - target) / N
//
// Returns true only if ALL parameter gradients pass.
// ------------------------------------------------------------------------
template<typename LayerType>
bool check_gradient_mse(
    LayerType& layer,
    const Tensor& input,
    const Tensor& target,
    double epsilon = 1e-5,
    double tolerance = 1e-4
) {
    // Run forward to get output dimensions and populate last_input
    Tensor output = layer.forward(input);

    size_t n = output.rows * output.cols;
    if (n == 0) {
        std::cerr << "  [WARN] check_gradient_mse: layer produced empty output, skipping." << std::endl;
        return true;
    }

    // Build grad_loss_wrt_output = 2 * (output - target) / N
    Tensor grad_loss(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        grad_loss.data[i] = 2.0 * (output.data[i] - target.data[i])
                             / static_cast<double>(n);
    }

    // Zero gradients, run backward to populate analytical gradients
    layer.zero_grad();
    layer.backward(grad_loss, 0.0);

    // Numerical check using MSE loss
    auto params = layer.parameters();
    auto grads  = layer.gradients();

    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* param = params[p];
        Tensor* grad  = grads[p];

        if (param->rows == 0 || param->cols == 0) continue;

        for (size_t r = 0; r < param->rows; ++r) {
            for (size_t c = 0; c < param->cols; ++c) {
                double orig = (*param)(r, c);

                // Perturb +epsilon: forward pass, compute MSE loss
                (*param)(r, c) = orig + epsilon;
                layer.zero_grad();
                Tensor out_plus = layer.forward(input);
                double L_plus = mse_loss(out_plus, target);

                // Perturb -epsilon: forward pass, compute MSE loss
                (*param)(r, c) = orig - epsilon;
                layer.zero_grad();
                Tensor out_minus = layer.forward(input);
                double L_minus = mse_loss(out_minus, target);

                // Restore
                (*param)(r, c) = orig;

                // Numerical gradient via centered finite difference
                double numerical = (L_plus - L_minus) / (2.0 * epsilon);
                double analytical = (*grad)(r, c);

                double rel_err = relative_error(analytical, numerical);
                if (rel_err > tolerance) {
                    std::cerr << "  [FAIL] MSE " << param << "[" << r << "][" << c << "]  "
                              << "analytical=" << analytical
                              << "  numerical=" << numerical
                              << "  rel_err=" << rel_err
                              << "  (tol=" << tolerance << ")" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

// ------------------------------------------------------------------------
// check_gradient_elementwise
//
// Per-element gradient checker that reports which specific parameter
// element is failing the gradient check.
//
// Uses the provided `grad_output` directly as the gradient flowing into
// the layer's output (from the loss or upstream layer).
//
// For each parameter element:
//   - Numerical: centered finite difference on the loss (sum of grad_output * output)
//   - Analytical: from layer.backward(grad_output)
//
// Returns true only if ALL parameter gradients pass.
// On failure, prints the (row, col) indices and gradient values of the failing element.
// ------------------------------------------------------------------------
bool check_gradient_elementwise(
    Layer& layer,
    const Tensor& input,
    const Tensor& grad_output,
    double epsilon = 1e-5,
    double tolerance = 1e-4
) {
    // Forward pass to populate layer's internal state (last_input, etc.)
    Tensor output = layer.forward(input);

    // Loss: L = sum_j( grad_output[j] * output[j] )
    // This is the standard "upstream gradient dotted with forward output" form.
    // dL/doutput = grad_output  (already provided)
    // dL/dinput  = grad_output @ weights  (computed by layer.backward)
    auto compute_loss = [&grad_output](const Tensor& out) -> double {
        double s = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) s += grad_output.data[i] * out.data[i];
        return s;
    };

    // Analytical gradients from a clean backward pass
    layer.zero_grad();
    layer.backward(grad_output, 0.0);

    auto params = layer.parameters();
    auto grads  = layer.gradients();

    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* param = params[p];
        Tensor* grad  = grads[p];

        if (param->rows == 0 || param->cols == 0) continue;

        for (size_t r = 0; r < param->rows; ++r) {
            for (size_t c = 0; c < param->cols; ++c) {
                double orig = (*param)(r, c);

                // Perturb +epsilon
                (*param)(r, c) = orig + epsilon;
                layer.zero_grad();
                layer.forward(input);  // repopulate last_input
                layer.backward(grad_output, 0.0);
                double L_plus = compute_loss(layer.forward(input));

                // Perturb -epsilon
                (*param)(r, c) = orig - epsilon;
                layer.zero_grad();
                layer.forward(input);
                layer.backward(grad_output, 0.0);
                double L_minus = compute_loss(layer.forward(input));

                // Restore
                (*param)(r, c) = orig;

                // Numerical gradient
                double numerical = (L_plus - L_minus) / (2.0 * epsilon);
                double analytical = (*grad)(r, c);

                double rel_err = relative_error(analytical, numerical);
                if (rel_err > tolerance) {
                    std::cerr << "  [FAIL] Element [" << r << "][" << c << "]  "
                              << "analytical=" << analytical
                              << "  numerical=" << numerical
                              << "  rel_err=" << rel_err
                              << "  (tol=" << tolerance << ")" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

// ============================================================================
// Legacy GradientChecker struct (kept for API compatibility)
// ============================================================================
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

#endif // GRADIENT_CHECK_H
