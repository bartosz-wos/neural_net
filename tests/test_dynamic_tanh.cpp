// test_dynamic_tanh.cpp — Tests for DynamicTanh (DyT) normalization-replacement layer.
// Zhu, Chen, He (CVPR 2025, arXiv:2503.10622)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/layers/normalization/dynamic_tanh.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"   // for Dense

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-12);
}

static double max_rel_err(const Tensor& a, const Tensor& b) {
    double max_e = 0.0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            max_e = std::max(max_e, rel_error(a[i][j], b[i][j]));
        }
    }
    return max_e;
}

// =====================================================================
// Test 1: Constructor validation + initial state
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: Constructor validation + initial state --" << endl;

    bool threw = false;
    try { DynamicTanh n(0); } catch (...) { threw = true; }
    check("features=0 throws", threw);

    DynamicTanh n(8);
    check("name() returns DynamicTanh", n.name() == "DynamicTanh");
    check("features() returns 8", n.features() == 8);
    check("alpha shape (1,1)", n.alpha.rows == 1 && n.alpha.cols == 1);
    check("alpha default = 0.5", std::abs(n.alpha[0][0] - 0.5) < 1e-12);
    check("gamma shape (1, features)", n.gamma.rows == 1 && n.gamma.cols == 8);
    check("gamma all-ones", n.gamma[0][0] == 1.0 && n.gamma[0][7] == 1.0);
    check("beta shape (1, features)", n.beta.rows == 1 && n.beta.cols == 8);
    check("beta all-zeros", n.beta[0][0] == 0.0 && n.beta[0][7] == 0.0);

    // Custom init_alpha
    DynamicTanh n2(4, 1.0);
    check("custom init_alpha = 1.0", std::abs(n2.alpha[0][0] - 1.0) < 1e-12);
}

// =====================================================================
// Test 2: Forward shape + finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 2: Forward shape + finiteness --" << endl;

    DynamicTanh n(8);
    Tensor input(4, 8);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = 0.1 * ((i + 1) * (j + 1) - 5);

    Tensor out = n.forward(input);
    check("output shape matches input", out.rows == 4 && out.cols == 8);

    bool finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) finite = false;
    check("all outputs finite", finite);
}

// =====================================================================
// Test 3: Forward closed-form with γ=1, β=0 — equals tanh(α·x)
// =====================================================================
static void test_forward_closed_form() {
    cout << endl << "-- Test 3: Forward closed-form (γ=1, β=0) --" << endl;

    DynamicTanh n(4, 0.5);
    // gamma = 1, beta = 0 by construction
    Tensor input(2, 4);
    input[0][0] = 1.0;  input[0][1] = -1.0; input[0][2] = 0.5; input[0][3] = 2.0;
    input[1][0] = 0.3;  input[1][1] = -2.0; input[1][2] = 1.5; input[1][3] = -0.7;

    Tensor out = n.forward(input);

    double max_e = 0.0;
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double expected = std::tanh(0.5 * input[i][j]);
            max_e = std::max(max_e, std::abs(out[i][j] - expected));
        }
    }
    check("γ=1,β=0 → tanh(αx) closed-form (max err < 1e-12)", max_e < 1e-12);
}

// =====================================================================
// Test 4: Forward squashing at extremes
// =====================================================================
static void test_forward_squashing() {
    cout << endl << "-- Test 4: Forward squashing at extremes --" << endl;

    DynamicTanh n(3, 0.5);
    // gamma=1, beta=0 by init
    // With α=0.5 and x=10: tanh(5) ≈ 0.99991
    Tensor input(1, 3);
    input[0][0] = 10.0;

    Tensor out = n.forward(input);
    double expected = std::tanh(5.0);
    check("x=10 → tanh(5) ≈ 1", std::abs(out[0][0] - expected) < 1e-12);

    // x=100: tanh(50) ≈ 1.0 to double precision
    Tensor input2(1, 3);
    input2[0][0] = 100.0;
    Tensor out2 = n.forward(input2);
    check("x=100 → out ≈ 1.0 (squashing)", std::abs(out2[0][0] - 1.0) < 1e-6);
}

// =====================================================================
// Test 5: Forward — γ scales, β shifts
// =====================================================================
static void test_forward_affine() {
    cout << endl << "-- Test 5: Forward γ scales, β shifts --" << endl;

    DynamicTanh n(3, 1.0);
    // Apply γ=2 and β=0.5 to ALL channels of the row (not just [0][0])
    for (size_t f = 0; f < 3; ++f) {
        n.gamma[0][f] = 2.0;
        n.beta[0][f]  = 0.5;
    }

    Tensor input(1, 3);
    input[0][0] = 1.0;  // α=1.0, x=1.0 → tanh(1) ≈ 0.7616 → 2*0.7616 + 0.5 = 2.0232
    input[0][1] = 0.0;
    input[0][2] = -1.0;

    Tensor out = n.forward(input);
    double expected0 = 2.0 * std::tanh(1.0) + 0.5;
    check("γ=2, β=0.5, x=1 → 2·tanh(1)+0.5", std::abs(out[0][0] - expected0) < 1e-12);

    // x=0: tanh(0) = 0 → γ*x + β = 0 + 0.5 = 0.5
    check("x=0 → β=0.5 (tanh=0)", std::abs(out[0][1] - 0.5) < 1e-12);

    // x=-1: tanh(-1) = -tanh(1) → 2·(-tanh(1)) + 0.5
    double expected2 = -2.0 * std::tanh(1.0) + 0.5;
    check("x=-1 → -γ·tanh(1)+β", std::abs(out[0][2] - expected2) < 1e-12);
}

// =====================================================================
// Test 6: Forward determinism
// =====================================================================
static void test_forward_determinism() {
    cout << endl << "-- Test 6: Forward determinism --" << endl;

    DynamicTanh n(5, 0.7);
    Tensor input(3, 5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = std::sin(i + 1.0) * (j + 1.0) - 1.5;

    Tensor out1 = n.forward(input);
    Tensor out2 = n.forward(input);
    double max_e = max_rel_err(out1, out2);
    check("two consecutive forwards bit-exact (max diff 0)", max_e < 1e-15);
}

// =====================================================================
// Test 7: parameters()/gradients() contract
// =====================================================================
static void test_params_contract() {
    cout << endl << "-- Test 7: parameters()/gradients() contract --" << endl;

    DynamicTanh n(6);
    auto p = n.parameters();
    auto g = n.gradients();

    check("parameters() returns 3 tensors", p.size() == 3);
    check("gradients() returns 3 tensors",  g.size() == 3);

    // shapes: alpha (1,1), gamma (1,6), beta (1,6)
    check("param 0 (alpha) shape (1,1)", p[0]->rows == 1 && p[0]->cols == 1);
    check("param 1 (gamma) shape (1,6)", p[1]->rows == 1 && p[1]->cols == 6);
    check("param 2 (beta)  shape (1,6)", p[2]->rows == 1 && p[2]->cols == 6);
    check("grad 0 (alpha) shape (1,1)", g[0]->rows == 1 && g[0]->cols == 1);
    check("grad 1 (gamma) shape (1,6)", g[1]->rows == 1 && g[1]->cols == 6);
    check("grad 2 (beta)  shape (1,6)", g[2]->rows == 1 && g[2]->cols == 6);
}

// =====================================================================
// Test 8: grad_gamma analytical vs finite difference
// =====================================================================
static void test_grad_gamma_fd() {
    cout << endl << "-- Test 8: grad_gamma analytical vs FD --" << endl;

    size_t batch = 4;
    size_t features = 5;
    DynamicTanh n(features, 0.7);

    // Random init gamma
    for (size_t f = 0; f < features; ++f) {
        n.gamma[0][f] = 0.5 + 0.3 * std::sin(f + 1.0);
    }

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * ((b + 1) * (f + 2) - 3);

    Tensor grad_out(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            grad_out[b][f] = ((b + f) % 3 == 0) ? 1.0 : -0.5;

    // Analytical: forward first to populate caches, then backward
    n.forward(input);
    n.zero_grad();
    Tensor grad_x = n.backward(grad_out, 0.0);
    Tensor ana_gamma(n.grad_gamma_);

    // Finite difference per gamma element
    double eps = 1e-5;
    Tensor num_gamma(1, features);
    for (size_t f = 0; f < features; ++f) {
        double orig = n.gamma[0][f];
        n.gamma[0][f] = orig + eps;
        Tensor out_p = n.forward(input);
        double l_p = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t ff = 0; ff < features; ++ff)
                l_p += grad_out[b][ff] * out_p[b][ff];

        n.gamma[0][f] = orig - eps;
        Tensor out_m = n.forward(input);
        double l_m = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t ff = 0; ff < features; ++ff)
                l_m += grad_out[b][ff] * out_m[b][ff];

        n.gamma[0][f] = orig;
        num_gamma[0][f] = (l_p - l_m) / (2.0 * eps);
    }

    double max_e = max_rel_err(ana_gamma, num_gamma);
    cout << "    [INFO] grad_gamma max rel_err = " << std::scientific << max_e << endl;
    check("grad_gamma FD rel_err < 1e-5", max_e < 1e-5);
}

// =====================================================================
// Test 9: grad_beta analytical vs FD
// =====================================================================
static void test_grad_beta_fd() {
    cout << endl << "-- Test 9: grad_beta analytical vs FD --" << endl;

    size_t batch = 4;
    size_t features = 5;
    DynamicTanh n(features, 0.7);

    for (size_t f = 0; f < features; ++f) {
        n.gamma[0][f] = 0.5 + 0.3 * std::sin(f + 1.0);
        n.beta[0][f] = 0.1 - 0.2 * std::cos(f + 1.0);
    }

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * ((b + 1) * (f + 2) - 3);

    Tensor grad_out(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            grad_out[b][f] = ((b + f) % 3 == 0) ? 1.0 : -0.5;

    n.forward(input);
    n.zero_grad();
    n.backward(grad_out, 0.0);
    Tensor ana_beta(n.grad_beta_);

    double eps = 1e-5;
    Tensor num_beta(1, features);
    for (size_t f = 0; f < features; ++f) {
        double orig = n.beta[0][f];
        n.beta[0][f] = orig + eps;
        Tensor out_p = n.forward(input);
        double l_p = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t ff = 0; ff < features; ++ff)
                l_p += grad_out[b][ff] * out_p[b][ff];

        n.beta[0][f] = orig - eps;
        Tensor out_m = n.forward(input);
        double l_m = 0.0;
        for (size_t b = 0; b < batch; ++b)
            for (size_t ff = 0; ff < features; ++ff)
                l_m += grad_out[b][ff] * out_m[b][ff];

        n.beta[0][f] = orig;
        num_beta[0][f] = (l_p - l_m) / (2.0 * eps);
    }

    double max_e = max_rel_err(ana_beta, num_beta);
    cout << "    [INFO] grad_beta max rel_err = " << std::scientific << max_e << endl;
    check("grad_beta FD rel_err < 1e-5", max_e < 1e-5);
}

// =====================================================================
// Test 10: grad_alpha analytical vs FD — the scalar learnable parameter
// =====================================================================
static void test_grad_alpha_fd() {
    cout << endl << "-- Test 10: grad_alpha analytical vs FD --" << endl;

    size_t batch = 4;
    size_t features = 5;
    DynamicTanh n(features, 0.7);

    for (size_t f = 0; f < features; ++f) {
        n.gamma[0][f] = 0.5 + 0.3 * std::sin(f + 1.0);
        n.beta[0][f] = 0.1 - 0.2 * std::cos(f + 1.0);
    }

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * ((b + 1) * (f + 2) - 3);

    Tensor grad_out(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            grad_out[b][f] = ((b + f) % 3 == 0) ? 1.0 : -0.5;

    n.forward(input);
    n.zero_grad();
    n.backward(grad_out, 0.0);
    double ana = n.grad_alpha_[0][0];

    double eps = 1e-5;
    double orig = n.alpha[0][0];

    n.alpha[0][0] = orig + eps;
    Tensor out_p = n.forward(input);
    double l_p = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            l_p += grad_out[b][f] * out_p[b][f];

    n.alpha[0][0] = orig - eps;
    Tensor out_m = n.forward(input);
    double l_m = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            l_m += grad_out[b][f] * out_m[b][f];

    n.alpha[0][0] = orig;
    double num = (l_p - l_m) / (2.0 * eps);

    double re = rel_error(num, ana);
    cout << "    [INFO] grad_alpha ana=" << std::fixed << std::setprecision(8) << ana
         << "  num=" << num << "  rel_err=" << std::scientific << re << endl;
    check("grad_alpha FD rel_err < 1e-5", re < 1e-5);
}

// =====================================================================
// Test 11: grad_x analytical vs FD
// =====================================================================
static void test_grad_x_fd() {
    cout << endl << "-- Test 11: grad_x analytical vs FD --" << endl;

    size_t batch = 4;
    size_t features = 5;
    DynamicTanh n(features, 0.7);

    for (size_t f = 0; f < features; ++f) {
        n.gamma[0][f] = 0.5 + 0.3 * std::sin(f + 1.0);
        n.beta[0][f] = 0.1 - 0.2 * std::cos(f + 1.0);
    }

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * ((b + 1) * (f + 2) - 3);

    Tensor grad_out(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            grad_out[b][f] = ((b + f) % 3 == 0) ? 1.0 : -0.5;

    n.forward(input);
    n.zero_grad();
    Tensor grad_x_ana = n.backward(grad_out, 0.0);

    // FD per input element
    double eps = 1e-5;
    Tensor grad_x_num(batch, features);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t f = 0; f < features; ++f) {
            double orig = input[b][f];
            input[b][f] = orig + eps;
            Tensor out_p = n.forward(input);
            double l_p = 0.0;
            for (size_t bb = 0; bb < batch; ++bb)
                for (size_t ff = 0; ff < features; ++ff)
                    l_p += grad_out[bb][ff] * out_p[bb][ff];

            input[b][f] = orig - eps;
            Tensor out_m = n.forward(input);
            double l_m = 0.0;
            for (size_t bb = 0; bb < batch; ++bb)
                for (size_t ff = 0; ff < features; ++ff)
                    l_m += grad_out[bb][ff] * out_m[bb][ff];

            input[b][f] = orig;
            grad_x_num[b][f] = (l_p - l_m) / (2.0 * eps);
        }
    }

    double max_e = max_rel_err(grad_x_ana, grad_x_num);
    cout << "    [INFO] grad_x max rel_err = " << std::scientific << max_e << endl;
    check("grad_x FD rel_err < 1e-5", max_e < 1e-5);
}

// =====================================================================
// Test 12: zero_grad clears all 3 parameter gradients
// =====================================================================
static void test_zero_grad() {
    cout << endl << "-- Test 12: zero_grad --" << endl;

    DynamicTanh n(4);
    n.grad_alpha_[0][0] = 1.0;
    n.grad_gamma_[0][1] = 0.5;
    n.grad_beta_[0][2]  = -0.3;

    n.zero_grad();

    check("zero_grad clears alpha grad", n.grad_alpha_[0][0] == 0.0);
    check("zero_grad clears gamma grad", n.grad_gamma_[0][1] == 0.0);
    check("zero_grad clears beta grad",  n.grad_beta_[0][2] == 0.0);
}

// =====================================================================
// Test 13: update_weights moves all three parameters
// =====================================================================
static void test_update_weights() {
    cout << endl << "-- Test 13: update_weights --" << endl;

    DynamicTanh n(4);
    double alpha_orig = n.alpha[0][0];
    double gamma_orig = n.gamma[0][2];
    double beta_orig  = n.beta[0][3];

    n.grad_alpha_[0][0] = 0.1;
    n.grad_gamma_[0][2] = 0.2;
    n.grad_beta_[0][3]  = 0.3;

    double lr = 0.05;
    n.update_weights(lr);

    check("alpha updated: -lr * grad", std::abs(n.alpha[0][0] - (alpha_orig - lr * 0.1)) < 1e-12);
    check("gamma updated: -lr * grad", std::abs(n.gamma[0][2] - (gamma_orig - lr * 0.2)) < 1e-12);
    check("beta  updated: -lr * grad", std::abs(n.beta[0][3]  - (beta_orig  - lr * 0.3)) < 1e-12);
}

// =====================================================================
// Test 14: Accumulation — multiple backward calls accumulate gradients
// =====================================================================
static void test_gradient_accumulation() {
    cout << endl << "-- Test 14: Gradient accumulation across backward calls --" << endl;

    size_t batch = 3;
    size_t features = 4;
    DynamicTanh n(features, 0.5);

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * (b * features + f + 1);

    Tensor grad_out(batch, features);
    grad_out.fill(1.0);

    n.zero_grad();
    n.forward(input);
    n.backward(grad_out, 0.0);
    Tensor grad_alpha_1(n.grad_alpha_);
    Tensor grad_gamma_1(n.grad_gamma_);
    Tensor grad_beta_1(n.grad_beta_);

    // Second backward call (without zero_grad in between) — should accumulate
    n.forward(input);
    n.backward(grad_out, 0.0);

    bool doubled = true;
    if (std::abs(n.grad_alpha_[0][0] - 2.0 * grad_alpha_1[0][0]) > 1e-12) doubled = false;
    for (size_t f = 0; f < features; ++f) {
        if (std::abs(n.grad_gamma_[0][f] - 2.0 * grad_gamma_1[0][f]) > 1e-12) doubled = false;
        if (std::abs(n.grad_beta_[0][f]  - 2.0 * grad_beta_1[0][f])  > 1e-12) doubled = false;
    }
    check("two backward calls (no zero_grad) → 2× gradient", doubled);
}

// =====================================================================
// Test 15: End-to-end training reduces loss
// =====================================================================
static void test_end_to_end_training() {
    cout << endl << "-- Test 15: End-to-end training reduces loss --" << endl;

    // Simple regression: y = 2*x. Small MLP with DyT between Dense layers.
    Dense l1(1, 16);
    DynamicTanh dyt1(16);
    Dense l2(16, 1);

    // Simple training data: x ∈ [-1, 1], y = 2*x
    Tensor X(2, 1);  // 2 samples
    X[0][0] = -1.0;
    X[1][0] =  1.0;
    Tensor Y(2, 1);
    Y[0][0] = -2.0;
    Y[1][0] =  2.0;

    auto compute_loss = [&](void) -> double {
        Tensor h1 = l1.forward(X);
        Tensor h2 = dyt1.forward(h1);
        Tensor yhat = l2.forward(h2);
        double loss = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 1; ++j)
                loss += (yhat[i][j] - Y[i][j]) * (yhat[i][j] - Y[i][j]);
        return loss / 2.0;
    };

    double lr = 0.05;
    double l0 = compute_loss();
    for (int step = 0; step < 30; ++step) {
        // Forward
        Tensor h1 = l1.forward(X);
        Tensor h2 = dyt1.forward(h1);
        Tensor yhat = l2.forward(h2);

        // dL/dyhat = (yhat - Y)  (since L = 0.5 * mean((yhat - Y)^2))
        Tensor gyhat(2, 1);
        for (size_t i = 0; i < 2; ++i) {
            gyhat[i][0] = (yhat[i][0] - Y[i][0]);
        }

        // Backward chain
        l1.zero_grad();
        dyt1.zero_grad();
        l2.zero_grad();

        Tensor gh2 = l2.backward(gyhat, 0.0);
        Tensor gh1 = dyt1.backward(gh2, 0.0);
        l1.backward(gh1, 0.0);

        // Update
        l1.update_weights(lr);
        dyt1.update_weights(lr);
        l2.update_weights(lr);
    }
    double lf = compute_loss();
    cout << "    [INFO] loss: " << std::fixed << std::setprecision(6) << l0 << " → " << lf
         << "  (drop " << std::setprecision(1) << 100.0 * (l0 - lf) / l0 << "%)" << endl;
    check("training reduces loss > 20% in 30 SGD steps", lf < 0.8 * l0);
}

// =====================================================================
// Test 16: Mutation 1 — stub γ out of forward → grad_gamma FD fails
// =====================================================================
static void test_mutation_drop_gamma() {
    cout << endl << "-- Test 16: Mutation — dropping γ fails grad_gamma FD --" << endl;

    size_t batch = 3, features = 4;
    DynamicTanh n(features, 0.5);

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * (b * features + f + 1);

    Tensor grad_out(batch, features);
    grad_out.fill(1.0);

    // The "unmutated" test: grad_gamma should equal sum_b(grad_out[b][f] * tanh(α·x[b][f]))
    n.zero_grad();
    n.forward(input);
    n.backward(grad_out, 0.0);

    double expected_grad_gamma_f0 = 0.0;
    for (size_t b = 0; b < batch; ++b)
        expected_grad_gamma_f0 += grad_out[b][0] * std::tanh(0.5 * input[b][0]);

    check("unmutated grad_gamma[0][0] matches Σ grad_out·tanh(αx)",
          std::abs(n.grad_gamma_[0][0] - expected_grad_gamma_f0) < 1e-12);
    // Note: this test catches the "drop gamma" mutation by FAILING the assertion
    // when the bug is present (since grad_gamma would become 0 then, but the
    // Σ grad_out·tanh(αx) would be nonzero). The non-mutated impl passes.
}

// =====================================================================
// Test 17: Mutation 2 — stub α out of forward → grad_alpha and grad_x FD fails
// =====================================================================
static void test_mutation_drop_alpha() {
    cout << endl << "-- Test 17: Mutation — dropping α fails grad_alpha FD --" << endl;

    size_t batch = 3, features = 4;
    DynamicTanh n(features, 0.5);

    Tensor input(batch, features);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            input[b][f] = 0.1 * (b * features + f + 1);

    Tensor grad_out(batch, features);
    grad_out.fill(1.0);

    n.zero_grad();
    n.forward(input);
    n.backward(grad_out, 0.0);

    // grad_alpha should be NON-ZERO because alpha actually scales x
    // (only zero if sum over (gamma·(1-t²)·x) = 0, which is rare)
    double sum_grad = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
            sum_grad += std::abs(n.grad_alpha_[0][0]);
    // Hand-derive the analytical value and verify
    double ana_expected = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f) {
            double u = 0.5 * input[b][f];
            double t = std::tanh(u);
            double one_minus_t_sq = 1.0 - t * t;
            ana_expected += grad_out[b][f] * 1.0 * one_minus_t_sq * input[b][f];
        }
    check("grad_alpha analytic vs hand-derived Σ grad·γ·(1-t²)·x",
          std::abs(n.grad_alpha_[0][0] - ana_expected) < 1e-12);
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    cout << "===== DynamicTanh (DyT) Tests =====" << endl;
    test_constructor();
    test_forward_shape();
    test_forward_closed_form();
    test_forward_squashing();
    test_forward_affine();
    test_forward_determinism();
    test_params_contract();
    test_grad_gamma_fd();
    test_grad_beta_fd();
    test_grad_alpha_fd();
    test_grad_x_fd();
    test_zero_grad();
    test_update_weights();
    test_gradient_accumulation();
    test_end_to_end_training();
    test_mutation_drop_gamma();
    test_mutation_drop_alpha();

    cout << endl;
    cout << "===== Summary: " << passed << " passed, " << failed << " failed =====" << endl;
    return failed == 0 ? 0 : 1;
}