// test_rmsnorm.cpp — Gradient correctness tests for RMSNorm
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/normalization/rms_norm.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8);
}

static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t[i][j] * t[i][j];
    return std::sqrt(s);
}

// =====================================================================
// Test 1: RMSNorm forward pass basic sanity
// =====================================================================
static void test_rmsnorm_forward() {
    cout << endl << "-- Test 1: RMSNorm forward pass --" << endl;

    RMSNorm norm(4, 1e-5);
    // Manually set gamma to all ones to simplify checks
    for (size_t f = 0; f < 4; ++f) norm.gamma[0][f] = 1.0;

    Tensor input(2, 4);
    input[0][0] = 1.0; input[0][1] = 0.0; input[0][2] = 0.0; input[0][3] = 0.0;
    input[1][0] = 0.0; input[1][1] = 2.0; input[1][2] = 0.0; input[1][3] = 0.0;

    Tensor out = norm.forward(input);

    // For sample 0: x = [1,0,0,0], rms = sqrt(1/4 + eps) ≈ 0.5
    // For sample 1: x = [0,2,0,0], rms = sqrt(4/4 + eps) = sqrt(1+eps) ≈ 1
    double rms0 = std::sqrt(1.0 / 4.0 + 1e-5);
    double rms1 = std::sqrt(4.0 / 4.0 + 1e-5);
    check("RMSNorm sample 0 output matches", std::abs(out[0][0] - 1.0/rms0) < 1e-6);
    check("RMSNorm sample 1 output matches", std::abs(out[1][1] - 2.0/rms1) < 1e-6);
    check("RMSNorm output shape (batch=2, features=4)", out.rows == 2 && out.cols == 4);
}

// =====================================================================
// Test 2: RMSNorm gradient on gamma (analytical vs numerical)
// =====================================================================
static void test_rmsnorm_gamma_gradient() {
    cout << endl << "-- Test 2: RMSNorm gamma gradient (analytical vs numerical) --" << endl;

    size_t batch = 4;
    size_t features = 8;
    RMSNorm norm(features, 1e-5);

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = (b * features + f + 1) * 0.1;

    Tensor out = norm.forward(input);

    // Upstream gradient: ones
    Tensor grad_out(batch, features);
    grad_out.fill(1.0);

    norm.zero_grad();
    Tensor grad_x = norm.backward(grad_out, 0.0);
    double ana_grad0 = norm.grad_gamma_[0][0];  // capture BEFORE perturbing weights
    double ana_grad1 = norm.grad_gamma_[0][1];  // capture BEFORE perturbing weights

    // Numerical gradient on gamma[0][0]: eps-based finite difference
    double eps = 1e-4;
    double orig = norm.gamma[0][0];

    // loss = sum(out) — since grad_out is ones, loss = sum(out)
    // Forward with gamma + eps
    norm.gamma[0][0] = orig + eps;
    norm.zero_grad();
    Tensor out_plus = norm.forward(input);
    double loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_plus += out_plus[b][f];

    // Forward with gamma - eps
    norm.gamma[0][0] = orig - eps;
    norm.zero_grad();
    Tensor out_minus = norm.forward(input);
    double loss_minus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_minus += out_minus[b][f];

    norm.gamma[0][0] = orig;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double err = rel_error(num_grad, ana_grad0);

    check("RMSNorm gamma[0][0] numerical vs analytical gradient", err < 1e-2);

    // Second channel
    eps = 1e-4;
    orig = norm.gamma[0][1];

    norm.gamma[0][1] = orig + eps;
    norm.zero_grad();
    out_plus = norm.forward(input);
    loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_plus += out_plus[b][f];

    norm.gamma[0][1] = orig - eps;
    norm.zero_grad();
    out_minus = norm.forward(input);
    loss_minus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_minus += out_minus[b][f];

    norm.gamma[0][1] = orig;

    double num_grad2 = (loss_plus - loss_minus) / (2.0 * eps);
    double err2 = rel_error(num_grad2, ana_grad1);  // ana_grad1 captured before perturbations
    cout << "    [DEBUG] gamma[0][1]: num_grad2=" << num_grad2 << " ana_grad1=" << ana_grad1 << " err2=" << err2 << endl;

    check("RMSNorm gamma[0][1] numerical vs analytical gradient", err2 < 1e-2);
}

// =====================================================================
// Test 3: RMSNorm gradient on input (analytical vs numerical)
// =====================================================================
static void test_rmsnorm_input_gradient() {
    cout << endl << "-- Test 3: RMSNorm input gradient (analytical vs numerical) --" << endl;

    size_t batch = 3;
    size_t features = 6;
    RMSNorm norm(features, 1e-5);

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = (b * features + f + 1) * 0.2 - 0.5;

    Tensor out = norm.forward(input);

    Tensor grad_out(batch, features);
    grad_out.fill(1.0);

    norm.zero_grad();
    Tensor grad_x = norm.backward(grad_out, 0.0);

    // Numerical gradient on input[1][2]
    double eps = 1e-4;
    double orig = input[1][2];

    norm.zero_grad();
    Tensor inp_plus = input;
    inp_plus[1][2] = orig + eps;
    norm.last_x = inp_plus;  // force the cached input
    // We need to re-run forward with the modified input; simpler to use norm directly
    // Actually we need to re-run forward; forward uses last_x internally
    // So we need to call forward with the modified input
    Tensor out_plus = norm.forward(inp_plus);
    double loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_plus += out_plus[b][f];

    inp_plus[1][2] = orig - eps;
    norm.zero_grad();
    // Note: last_x needs to be set properly for each call
    // This is tricky since forward() caches last_x = input.
    // We'll set last_x and also need to recompute RMS. Simpler: just re-run with modified input.
    // Actually the forward pass computes RMS from last_x = input.
    // So calling forward with modified inp_plus will give us the new output.

    inp_plus[1][2] = orig - eps;
    norm.zero_grad();
    out_plus = norm.forward(inp_plus);
    loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            loss_plus += out_plus[b][f];
    // Wait I'm mixing variable names. Let me restart.
    {
        Tensor inp_p = input; inp_p[1][2] = orig + eps;
        norm.zero_grad();
        Tensor out_p = norm.forward(inp_p);
        double lp = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < features; ++f)
                lp += out_p[b][f];

        Tensor inp_m = input; inp_m[1][2] = orig - eps;
        norm.zero_grad();
        Tensor out_m = norm.forward(inp_m);
        double lm = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < features; ++f)
                lm += out_m[b][f];

        double num_grad = (lp - lm) / (2.0 * eps);
        double ana_grad = grad_x[1][2];
        double err = rel_error(num_grad, ana_grad);

        check("RMSNorm input[1][2] numerical vs analytical gradient", err < 1e-2);
    }

    // Also check input[0][0]
    {
        double orig00 = input[0][0];
        double eps2 = 1e-4;

        Tensor inp_p = input; inp_p[0][0] = orig00 + eps2;
        norm.zero_grad();
        Tensor out_p = norm.forward(inp_p);
        double lp = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < features; ++f)
                lp += out_p[b][f];

        Tensor inp_m = input; inp_m[0][0] = orig00 - eps2;
        norm.zero_grad();
        Tensor out_m = norm.forward(inp_m);
        double lm = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < features; ++f)
                lm += out_m[b][f];

        double num_grad = (lp - lm) / (2.0 * eps2);
        double ana_grad = grad_x[0][0];
        double err = rel_error(num_grad, ana_grad);

        check("RMSNorm input[0][0] numerical vs analytical gradient", err < 1e-2);
    }
}

// =====================================================================
// Test 4: RMSNorm grad_gamma is non-zero after backward
// =====================================================================
static void test_rmsnorm_grad_gamma_nonzero() {
    cout << endl << "-- Test 4: RMSNorm grad_gamma non-zero --" << endl;

    RMSNorm norm(8, 1e-5);
    Tensor input(5, 8);
    input.fill(0.5);

    norm.zero_grad();
    Tensor out = norm.forward(input);
    Tensor grad_out(5, 8);
    grad_out.fill(1.0);
    norm.backward(grad_out, 0.0);

    double g_norm = tensor_l2norm(norm.grad_gamma_);
    check("RMSNorm grad_gamma_ is non-zero after backward", g_norm > 1e-10);
}

// =====================================================================
// Test 5: RMSNorm grad_x is non-zero after backward
// =====================================================================
static void test_rmsnorm_grad_x_nonzero() {
    cout << endl << "-- Test 5: RMSNorm grad_x non-zero --" << endl;

    RMSNorm norm(8, 1e-5);
    Tensor input(5, 8);
    input.fill(0.3);

    norm.zero_grad();
    Tensor out = norm.forward(input);
    Tensor grad_out(5, 8);
    grad_out.fill(1.0);
    Tensor grad_x = norm.backward(grad_out, 0.0);

    double gx_norm = tensor_l2norm(grad_x);
    check("RMSNorm grad_x is non-zero after backward", gx_norm > 1e-10);
}

// =====================================================================
// Test 6: RMSNorm update_weights changes gamma
// =====================================================================
static void test_rmsnorm_update_weights() {
    cout << endl << "-- Test 6: RMSNorm update_weights changes gamma --" << endl;

    RMSNorm norm(4, 1e-5);
    for (size_t f = 0; f < 4; ++f) norm.gamma[0][f] = 1.0;

    Tensor input(2, 4);
    input.fill(0.1);

    norm.zero_grad();
    Tensor out = norm.forward(input);
    Tensor grad_out(2, 4);
    grad_out.fill(1.0);
    norm.backward(grad_out, 0.0);

    // Copy old gamma
    Tensor old_gamma = norm.gamma;

    norm.update_weights(0.01);

    bool changed = false;
    for (size_t f = 0; f < 4; ++f)
        if (std::abs(norm.gamma[0][f] - old_gamma[0][f]) > 1e-12)
            changed = true;

    check("RMSNorm update_weights actually changes gamma", changed);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== RMSNorm Gradient Correctness Tests ===" << endl;
    cout << setprecision(8);

    test_rmsnorm_forward();
    test_rmsnorm_gamma_gradient();
    test_rmsnorm_input_gradient();
    test_rmsnorm_grad_gamma_nonzero();
    test_rmsnorm_grad_x_nonzero();
    test_rmsnorm_update_weights();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}