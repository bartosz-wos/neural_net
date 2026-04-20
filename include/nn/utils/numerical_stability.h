#ifndef NUMERICAL_STABILITY_H
#define NUMERICAL_STABILITY_H

#include "../core/tensor.h"
#include <string>
#include <vector>

// Numerical stability test suite for neural network components.
// Tests activation functions, loss functions, and gradients for NaN/Inf issues.
class NumericalStabilityTest {
public:
    struct TestResult {
        std::string name;
        bool passed;
        std::string message;
        double max_abs_value;
        bool has_nan;
        bool has_inf;
    };

    // Run all stability tests
    static std::vector<TestResult> run_all();

    // Test activation functions with extreme inputs
    static TestResult test_activations();

    // Test loss functions (softmax, cross-entropy, MSE)
    static TestResult test_loss_functions();

    // Test gradient flow with near-zero and large values
    static TestResult test_gradient_flow();

    // Test softmax with large logits (overflow stress)
    static TestResult test_softmax_overflow();

    // Test element-wise ops with boundary values
    static TestResult test_boundary_ops();

    // Verify gradient clipping works correctly
    static bool test_gradient_clipping(double max_norm);

private:
    static bool contains_nan(const Tensor& t);
    static bool contains_inf(const Tensor& t);
    static double max_abs(const Tensor& t);
};

#endif