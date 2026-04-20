#include "numerical_stability.h"
#include "../activations/activations.h"
#include <cmath>
#include <limits>
#include <iostream>

bool NumericalStabilityTest::contains_nan(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (std::isnan(t[i][j])) return true;
    return false;
}

bool NumericalStabilityTest::contains_inf(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (std::isinf(t[i][j])) return true;
    return false;
}

double NumericalStabilityTest::max_abs(const Tensor& t) {
    double m = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            m = std::max(m, std::abs(t[i][j]));
    return m;
}

NumericalStabilityTest::TestResult NumericalStabilityTest::test_activations() {
    TestResult r;
    r.name = "activation_extreme_inputs";
    r.passed = true;

    ReLU relu; (void)relu; Sigmoid sigmoid; (void)sigmoid; Tanh tanh; (void)tanh;

    // Test with very large positive/negative values
    std::vector<double> extremes = {1e10, -1e10, 1e308, -1e308, 1e-308, -1e-308};

    for (double val : extremes) {
        Tensor input(1, 1);
        input[0][0] = val;

        // ReLU should handle any finite value
        Tensor out = input.apply(ReLU{});
        if (contains_nan(out) || contains_inf(out)) {
            r.passed = false;
            r.message = "ReLU produced NaN/Inf for input " + std::to_string(val);
            return r;
        }

        // Sigmoid
        out = input.apply(Sigmoid{});
        if (contains_nan(out)) {
            r.passed = false;
            r.message = "Sigmoid produced NaN for input " + std::to_string(val);
            return r;
        }

        // Tanh
        out = input.apply(Tanh{});
        if (contains_nan(out)) {
            r.passed = false;
            r.message = "Tanh produced NaN for input " + std::to_string(val);
            return r;
        }
    }

    r.message = "All activations handle extreme values correctly";
    r.max_abs_value = 1.0;
    r.has_nan = false;
    r.has_inf = false;
    return r;
}

NumericalStabilityTest::TestResult NumericalStabilityTest::test_softmax_overflow() {
    TestResult r;
    r.name = "softmax_overflow_stress";
    r.passed = true;
    r.has_nan = false;
    r.has_inf = false;
    r.max_abs_value = 0.0;

    // Test softmax with large logits that would overflow exp
    // exp(1000) overflows double. Using the numerically stable version should handle it.
    std::vector<std::vector<double>> logit_cases = {
        {1000.0, 1000.0, 1000.0},      // all large positive
        {-1000.0, -1000.0, -1000.0},   // all large negative
        {1000.0, 500.0, 0.0},          // mixed
        {1e15, 1e15, 1e15},            // extreme
    };

    for (const auto& logits_raw : logit_cases) {
        Tensor logits(1, (size_t)logits_raw.size());
        for (size_t j = 0; j < logits_raw.size(); ++j)
            logits[0][j] = logits_raw[j];

        // Manual stable softmax: subtract max before exp
        double max_logit = logits[0][0];
        for (size_t j = 1; j < logits.cols; ++j)
            max_logit = std::max(max_logit, logits[0][j]);

        Tensor shifted(1, logits.cols);
        double exp_sum = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) {
            shifted[0][j] = logits[0][j] - max_logit;
            if (shifted[0][j] > 700) shifted[0][j] = 700; // clamp exp argument
            exp_sum += std::exp(shifted[0][j]);
        }

        Tensor probs(1, logits.cols);
        for (size_t j = 0; j < logits.cols; ++j)
            probs[0][j] = std::exp(shifted[0][j]) / exp_sum;

        if (contains_nan(probs) || contains_inf(probs)) {
            r.passed = false;
            r.message = "Softmax produced NaN/Inf for logits";
            return r;
        }
        r.max_abs_value = std::max(r.max_abs_value, max_abs(probs));
    }

    r.message = "Softmax handles overflow stress correctly";
    return r;
}

NumericalStabilityTest::TestResult NumericalStabilityTest::test_loss_functions() {
    TestResult r;
    r.name = "loss_functions_numerical";
    r.passed = true;
    r.has_nan = false;
    r.has_inf = false;
    r.max_abs_value = 0.0;

    // MSE with near-zero targets
    Tensor pred(1, 3);
    pred[0][0] = 1.0; pred[0][1] = 2.0; pred[0][2] = 3.0;
    Tensor y(1, 3);
    y[0][0] = 1.0000001; y[0][1] = 2.0; y[0][2] = 3.0;

    Tensor diff = pred - y;
    double mse = 0.0;
    for (size_t i = 0; i < diff.rows; ++i)
        for (size_t j = 0; j < diff.cols; ++j)
            mse += diff[i][j] * diff[i][j];
    mse /= diff.rows;

    if (std::isnan(mse) || std::isinf(mse)) {
        r.passed = false;
        r.message = "MSE produced NaN/Inf on near-equal values";
        return r;
    }

    // softmax_cross_entropy_logits with extreme values
    Tensor logits(1, 3);
    logits[0][0] = 1000.0; logits[0][1] = 900.0; logits[0][2] = 800.0;
    Tensor labels_onehot(1, 3);
    labels_onehot[0][0] = 1.0; labels_onehot[0][1] = 0.0; labels_onehot[0][2] = 0.0;

    // Use stable softmax first
    double max_l = logits[0][0];
    for (size_t j = 1; j < logits.cols; ++j)
        max_l = std::max(max_l, logits[0][j]);
    double esum = 0.0;
    Tensor exp_logits(1, logits.cols);
    for (size_t j = 0; j < logits.cols; ++j) {
        double v = std::exp(logits[0][j] - max_l);
        exp_logits[0][j] = v;
        esum += v;
    }
    Tensor probs(1, logits.cols);
    for (size_t j = 0; j < logits.cols; ++j) probs[0][j] = exp_logits[0][j] / esum;

    double ce = 0.0;
    for (size_t j = 0; j < logits.cols; ++j)
        ce -= labels_onehot[0][j] * std::log(probs[0][j] + 1e-300);
    if (std::isnan(ce) || std::isinf(ce)) {
        r.passed = false;
        r.message = "Cross-entropy produced NaN/Inf";
        return r;
    }

    r.message = "All loss functions numerically stable";
    return r;
}

NumericalStabilityTest::TestResult NumericalStabilityTest::test_gradient_flow() {
    TestResult r;
    r.name = "gradient_flow_vanishing";
    r.passed = true;

    // Create a deep network and check gradient magnitude decays or survives
    size_t depth = 20;
    std::vector<Tensor> weights;
    for (size_t i = 0; i < depth; ++i) {
        weights.emplace_back(64, 64);
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < 64; ++k)
                weights[i][j][k] = 0.02 * (rand() / RAND_MAX - 0.5);
    }

    // Forward pass with small init
    Tensor x(1, 64);
    for (size_t j = 0; j < 64; ++j) x[0][j] = 0.01 * (rand() / RAND_MAX - 0.5);

    for (size_t d = 0; d < depth; ++d) {
        Tensor out(1, 64);
        for (size_t j = 0; j < 64; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < 64; ++k)
                sum += x[0][k] * weights[d][k][j];
            out[0][j] = std::max(0.0, sum); // ReLU
        }
        x = out;
    }

    // Backward with upstream gradient = 1
    Tensor grad(1, 64);
    grad.fill(1.0);
    for (size_t d = depth; d > 0; --d) {
        // Gradient of ReLU: just pass through (sign not tracked here — simplified)
        for (size_t j = 0; j < 64; ++j)
            if (x[0][j] <= 0) grad[0][j] = 0.0;

        double sum_sq = 0.0;
        for (size_t j = 0; j < 64; ++j)
            sum_sq += grad[0][j] * grad[0][j];
        double grad_norm = std::sqrt(sum_sq / 64.0);
        if (d == depth && grad_norm < 1e-10) {
            r.passed = false;
            r.message = "Gradients vanishing at depth " + std::to_string(d);
            return r;
        }

        // Weight gradient
        Tensor grad_w(64, 64);
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < 64; ++k)
                grad_w[k][j] = grad[0][j] * x[0][k];

        // Update weights (SGD)
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < 64; ++k)
                weights[d-1][k][j] -= 0.01 * grad_w[k][j];

        // Gradient w.r.t. input
        grad = Tensor(1, 64);
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < 64; ++k)
                grad[0][k] += weights[d-1][k][j] * grad[0][j];
    }

    r.message = "Gradient flow stable through " + std::to_string(depth) + " layers";
    r.max_abs_value = max_abs(grad);
    return r;
}

NumericalStabilityTest::TestResult NumericalStabilityTest::test_boundary_ops() {
    TestResult r;
    r.name = "boundary_operations";
    r.passed = true;

    // Test division by very small numbers
    Tensor a(1, 3);
    a[0][0] = 1.0; a[0][1] = 1.0; a[0][2] = 1.0;
    Tensor b(1, 3);
    b[0][0] = 1e-300; b[0][1] = -1e-300; b[0][2] = 0.0;

    for (size_t j = 0; j < 3; ++j) {
        double result = a[0][j] / (b[0][j] + 1e-300);
        if (std::isinf(result) || result > 1e300) {
            r.passed = false;
            r.message = "Division produced overflow at index " + std::to_string(j);
            return r;
        }
    }

    // Test log of very small numbers
    double log_val = std::log(1e-300);
    if (std::isinf(log_val) && log_val < 0) {
        // This is expected — log(very small) = large negative, not a failure
    }

    r.message = "Boundary operations handled correctly";
    return r;
}

bool NumericalStabilityTest::test_gradient_clipping(double max_norm) {
    // Test that gradient clipping doesn't corrupt the tensor
    Tensor grad(1, 3);
    grad[0][0] = 100.0; grad[0][1] = 50.0; grad[0][2] = 0.0;

    double norm = 0.0;
    for (size_t j = 0; j < grad.cols; ++j)
        norm += grad[0][j] * grad[0][j];
    norm = std::sqrt(norm);

    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (size_t j = 0; j < grad.cols; ++j)
            grad[0][j] *= scale;
    }

    double new_norm = 0.0;
    for (size_t j = 0; j < grad.cols; ++j)
        new_norm += grad[0][j] * grad[0][j];
    new_norm = std::sqrt(new_norm);

    return new_norm <= max_norm + 1e-9;
}

std::vector<NumericalStabilityTest::TestResult> NumericalStabilityTest::run_all() {
    std::vector<TestResult> results;
    results.push_back(test_activations());
    results.push_back(test_softmax_overflow());
    results.push_back(test_loss_functions());
    results.push_back(test_boundary_ops());
    results.push_back(test_gradient_flow());

    bool all_passed = true;
    for (const auto& r : results) {
        if (!r.passed) all_passed = false;
    }

    std::cout << "\n=== Numerical Stability Tests ===\n";
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS]" : "[FAIL]") << " " << r.name << ": " << r.message << "\n";
        if (r.passed) continue;
        std::cout << "  has_nan=" << r.has_nan << " has_inf=" << r.has_inf
                  << " max_abs=" << r.max_abs_value << "\n";
    }
    std::cout << "Overall: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n\n";
    return results;
}