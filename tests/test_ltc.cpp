// ==========================================================================
// tests/test_ltc.cpp
//
// Focused tests for LTC (Liquid Time-Constant Networks, Hasani 2021).
// Tests:
//   1. Constructor validation (3 cases)
//   2. Forward shape (T, input_dim) -> (T, hidden_size)
//   3. Forward output is finite
//   4. Forward output bounded in [-1, 1] (after tanh)
//   5. Hand-derived forward reference (single neuron, T=2)
//   6. tau > 0 for all (t, i)
//   7. g in (0, 1) for all (t, i)
//   8. Cache shape sanity (tau, g, alpha, h)
//   9. Single-step BPTT hand-derived (dL/dW_ih)
//  10-16. Numerical gradient check vs analytical for all 7 params + input
//  17. Training reduces loss
//  18. parameters() returns 7 tensors
//  19. gradients() returns 7 tensors
//  20. zero_grad clears all 7 gradient tensors
//  21. update_weights moves all 7 parameter tensors
//  22. Determinism (bit-exact with copied params)
//  23. Longer sequence (T=10) W_hh gradient check
//  24. name() returns "LTC"
// ==========================================================================

#include "nn/nn.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include <utility>

static int passed = 0, failed = 0;

#define CHECK(cond, name) do { \
    if (cond) { ++passed; std::cout << "  [PASS] " << name << "\n"; } \
    else      { ++failed; std::cerr << "  [FAIL] " << name << " : line " << __LINE__ << "\n"; } \
} while(0)

// Compute L = 0.5 * sum_t,i grad_out[t][i] * h[t][i]  (a dot-product loss for FD checks)
static double dot_loss(LTC& ltc, const Tensor& x, const Tensor& grad_out) {
    Tensor out = ltc.forward(x);
    double L = 0.0;
    for (size_t t = 0; t < out.rows; ++t)
        for (size_t i = 0; i < out.cols; ++i)
            L += grad_out[t][i] * out[t][i];
    return L;
}

// ----------------------------------------------------------------------------
// Test 1: Constructor validation
// ----------------------------------------------------------------------------
static void test_constructor_validation() {
    bool caught_zero_in = false;
    try { LTC bad(0, 4); } catch (std::invalid_argument&) { caught_zero_in = true; }
    CHECK(caught_zero_in, "Test 1: LTC throws on input_dim=0");

    bool caught_zero_hid = false;
    try { LTC bad(3, 0); } catch (std::invalid_argument&) { caught_zero_hid = true; }
    CHECK(caught_zero_hid, "Test 2: LTC throws on hidden_size=0");

    bool caught_neg_tau = false;
    try { LTC bad(2, 3, -1.0); } catch (std::invalid_argument&) { caught_neg_tau = true; }
    CHECK(caught_neg_tau, "Test 3: LTC throws on negative tau_base_init");

    CHECK(std::string(LTC(2, 3).name()) == "LTC", "Test 4: LTC::name() == 'LTC'");
}

// ----------------------------------------------------------------------------
// Test 5: Forward shape with manually-set parameters
// ----------------------------------------------------------------------------
static void test_forward_shape_known_params() {
    LTC ltc(2, 3);
    ltc.W_ih_ = Tensor(3, 2);
    for (size_t i = 0; i < 3; ++i) for (size_t k = 0; k < 2; ++k) ltc.W_ih_[i][k] = 0.1;
    ltc.W_hh_ = Tensor::zeros(3, 3);
    ltc.b_    = Tensor::zeros(3, 1);
    ltc.W_tx_ = Tensor::zeros(3, 2);
    ltc.W_th_ = Tensor::zeros(3, 3);
    ltc.b_t_  = Tensor::zeros(3, 1);
    ltc.log_tau_base_ = Tensor(3, 1);
    for (size_t i = 0; i < 3; ++i) ltc.log_tau_base_[i][0] = 0.0;

    Tensor x(4, 2);
    for (size_t t = 0; t < 4; ++t)
        for (size_t k = 0; k < 2; ++k)
            x[t][k] = 0.5;

    Tensor out = ltc.forward(x);
    CHECK(out.rows == 4, "Test 5a: forward shape rows = T");
    CHECK(out.cols == 3, "Test 5b: forward shape cols = hidden_size");

    bool all_finite = true;
    bool all_bounded = true;
    for (size_t t = 0; t < 4; ++t)
        for (size_t i = 0; i < 3; ++i) {
            if (!std::isfinite(out[t][i])) all_finite = false;
            if (std::abs(out[t][i]) > 1.0 + 1e-6) all_bounded = false;
        }
    CHECK(all_finite, "Test 5c: forward output finite (known params)");
    CHECK(all_bounded, "Test 5d: forward output in [-1, 1] after tanh");
}

// ----------------------------------------------------------------------------
// Test 6: Hand-derived single-neuron forward
// ----------------------------------------------------------------------------
static void test_forward_hand_derived_single_neuron() {
    // input_dim=1, hidden_size=1, T=2
    // x = [[1.0], [1.0]], W_ih = 0.3, all else zero, log_tau_base = 0 (tau_base=1)
    // tau_0 = 1 * softplus(0) = ln(2) ≈ 0.693
    // g_0 = exp(-1/ln(2)) ≈ 0.2370
    // alpha_0 = 0.7630
    // A_0 = 0.3
    // h_0 = tanh(0 + 0.7630 * 0.3) = tanh(0.229) ≈ 0.2252
    // h_1 = tanh(0.2370 * 0.2252 + 0.7630 * 0.3) = tanh(0.0534 + 0.229) = tanh(0.2824) ≈ 0.2750
    LTC ltc(1, 1);
    ltc.W_ih_  = Tensor(1, 1); ltc.W_ih_[0][0]  = 0.3;
    ltc.W_hh_  = Tensor::zeros(1, 1);
    ltc.b_     = Tensor::zeros(1, 1);
    ltc.W_tx_  = Tensor::zeros(1, 1);
    ltc.W_th_  = Tensor::zeros(1, 1);
    ltc.b_t_   = Tensor::zeros(1, 1);
    ltc.log_tau_base_ = Tensor(1, 1); ltc.log_tau_base_[0][0] = 0.0;

    Tensor x(2, 1);
    x[0][0] = 1.0; x[1][0] = 1.0;

    Tensor out = ltc.forward(x);
    double h0_expected = std::tanh(0.3 * (1.0 - std::exp(-1.0/std::log(2.0))));
    double h1_expected = std::tanh(0.3 * (1.0 - std::exp(-1.0/std::log(2.0))) +
                                   std::exp(-1.0/std::log(2.0)) * h0_expected);

    CHECK(std::abs(out[0][0] - h0_expected) < 1e-3, "Test 6a: hand-derived h_0 matches");
    CHECK(std::abs(out[1][0] - h1_expected) < 1e-3, "Test 6b: hand-derived h_1 matches (recurrence)");
}

// ----------------------------------------------------------------------------
// Test 7: Forward finiteness + boundedness on random input
// ----------------------------------------------------------------------------
static void test_forward_finite_bounded_random() {
    LTC ltc(4, 8);
    Tensor x(6, 4);
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 0.5);
    for (size_t t = 0; t < 6; ++t)
        for (size_t k = 0; k < 4; ++k)
            x[t][k] = nd(rng);

    Tensor out = ltc.forward(x);
    bool all_finite = true;
    bool all_bounded = true;
    for (size_t t = 0; t < 6; ++t)
        for (size_t i = 0; i < 8; ++i) {
            if (!std::isfinite(out[t][i])) all_finite = false;
            if (std::abs(out[t][i]) > 1.0 + 1e-6) all_bounded = false;
        }
    CHECK(all_finite, "Test 7a: forward all finite on random input");
    CHECK(all_bounded, "Test 7b: forward output in [-1, 1] (random init)");
}

// ----------------------------------------------------------------------------
// Test 8: Cache shape sanity + ranges
// ----------------------------------------------------------------------------
static void test_cache_shapes_ranges() {
    LTC ltc(3, 5);
    Tensor x(4, 3);
    for (size_t t = 0; t < 4; ++t)
        for (size_t k = 0; k < 3; ++k)
            x[t][k] = 0.1;
    ltc.forward(x);

    CHECK(ltc.last_tau().rows == 4 && ltc.last_tau().cols == 5, "Test 8a: tau cache (T, hidden)");
    CHECK(ltc.last_g().rows == 4 && ltc.last_g().cols == 5, "Test 8b: g cache (T, hidden)");
    CHECK(ltc.last_alpha().rows == 4 && ltc.last_alpha().cols == 5, "Test 8c: alpha cache (T, hidden)");
    CHECK(ltc.last_h().rows == 4 && ltc.last_h().cols == 5, "Test 8d: h cache (T, hidden)");

    bool tau_positive = true;
    for (size_t t = 0; t < 4; ++t)
        for (size_t i = 0; i < 5; ++i)
            if (ltc.last_tau()[t][i] <= 0.0) tau_positive = false;
    CHECK(tau_positive, "Test 8e: tau > 0 for all (t, i)");

    bool g_in_unit = true;
    for (size_t t = 0; t < 4; ++t)
        for (size_t i = 0; i < 5; ++i)
            if (ltc.last_g()[t][i] <= 0.0 || ltc.last_g()[t][i] >= 1.0) g_in_unit = false;
    CHECK(g_in_unit, "Test 8f: g in (0, 1) for all (t, i)");
}

// ----------------------------------------------------------------------------
// Test 9: Single-step BPTT hand-derived
// L = 0.5 * h_0^2, so dL/dh_0 = h_0.
// dh_0/d_pre_tanh = 1 - h_0^2
// d_pre_tanh/dg = h_{-1} - A_0 = 0 - 0.3 = -0.3
// dg/dτ = g/τ^2, grad_τ = dh_in * (h_{-1} - A_0) * g/τ^2
// d_pre_tanh/dA_0 = 1 - g_0
// So dL/dW_ih = dh_in * (1 - g) * x_0  (the A_t path)
// ----------------------------------------------------------------------------
static void test_backward_single_step_hand_derived() {
    LTC ltc(1, 1);
    ltc.W_ih_  = Tensor(1, 1); ltc.W_ih_[0][0]  = 0.3;
    ltc.W_hh_  = Tensor::zeros(1, 1);
    ltc.b_     = Tensor::zeros(1, 1);
    ltc.W_tx_  = Tensor::zeros(1, 1);
    ltc.W_th_  = Tensor::zeros(1, 1);
    ltc.b_t_   = Tensor::zeros(1, 1);
    ltc.log_tau_base_ = Tensor(1, 1); ltc.log_tau_base_[0][0] = 0.0;

    Tensor x(1, 1); x[0][0] = 1.0;
    Tensor out = ltc.forward(x);

    double h0 = out[0][0];
    double g_val = ltc.last_g()[0][0];
    double alpha_val = ltc.last_alpha()[0][0];
    double z_tau = ltc.last_z_tau()[0][0];
    double tau_t = ltc.last_tau()[0][0];

    Tensor grad_out(1, 1); grad_out[0][0] = h0;  // dL/dh_0 = h_0
    ltc.backward(grad_out, 0.0);

    double expected_dW_ih = h0 * (1.0 - h0*h0) * alpha_val * 1.0;
    CHECK(std::abs(ltc.grad_W_ih_[0][0] - expected_dW_ih) < 1e-6,
          "Test 9a: single-step dL/dW_ih matches hand derivation (rel_err 0)");

    // dL/dx_0 (input grad) = grad_A * W_ih + grad_z_tau * W_tx
    // grad_A = h0 * (1-h0^2) * alpha, grad_z_tau = (h0*(1-h0^2)*(0-0.3) * g/tau^2) * tau_base * sigmoid(z_tau)
    // For this test: h_{-1}=0, A_0 = 0.3, so dh_t_input/dg = -0.3
    double dh_in = h0 * (1.0 - h0*h0);
    double grad_A = dh_in * alpha_val;
    double grad_g = dh_in * (0.0 - 0.3);  // h_prev - A
    double grad_tau = grad_g * (g_val / (tau_t * tau_t));
    double sig_z = 1.0 / (1.0 + std::exp(-z_tau));
    double grad_z_tau = grad_tau * 1.0 * sig_z;  // tau_base = exp(log_tau_base) = exp(0) = 1
    double expected_dx = grad_A * ltc.W_ih_[0][0] + grad_z_tau * ltc.W_tx_[0][0];
    ltc.forward(x);
    Tensor grad_in = ltc.backward(grad_out, 0.0);
    CHECK(std::abs(grad_in[0][0] - expected_dx) < 1e-6,
          "Test 9b: single-step dL/dx_0 matches hand derivation");
}

// ----------------------------------------------------------------------------
// Numerical gradient check helper struct
// ----------------------------------------------------------------------------
struct FDCheck {
    LTC& ltc;
    Tensor x;
    Tensor grad_out;
    double eps;
    FDCheck(LTC& l, const Tensor& x_, const Tensor& g_, double e = 1e-4)
        : ltc(l), x(x_), grad_out(g_), eps(e) {}

    double param_fd(std::function<void(LTC&, double)> set,
                    std::function<double(LTC&)> /*get*/,
                    double orig)
    {
        set(ltc, orig + eps);
        double Lp = dot_loss(ltc, x, grad_out);
        set(ltc, orig - eps);
        double Lm = dot_loss(ltc, x, grad_out);
        set(ltc, orig);
        return (Lp - Lm) / (2.0 * eps);
    }

    bool check(std::function<void(LTC&, double)> set,
               std::function<double(LTC&)> get,
               double orig, double ana, const std::string& name)
    {
        double num = param_fd(set, get, orig);
        double denom = std::max({std::abs(num), std::abs(ana), 1e-12});
        double rel_err = std::abs(num - ana) / denom;
        if (rel_err < 1e-3) {
            ++passed; std::cout << "  [PASS] " << name << " (rel_err=" << rel_err << ")\n";
            return true;
        }
        ++failed; std::cerr << "  [FAIL] " << name
                            << " num=" << num << " ana=" << ana
                            << " rel_err=" << rel_err << "\n";
        return false;
    }
};

static void test_full_numerical_grad_check() {
    LTC ltc(2, 3);
    // Fixed init for reproducibility
    auto set_val = [](Tensor& t, double v) {
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                t[i][j] = v;
    };
    set_val(ltc.W_ih_,  0.10);
    set_val(ltc.W_hh_,  0.05);
    set_val(ltc.W_tx_,  0.02);
    set_val(ltc.W_th_,  0.02);
    set_val(ltc.b_,     0.00);
    set_val(ltc.b_t_,   0.00);
    set_val(ltc.log_tau_base_, 0.5);

    Tensor x(5, 2);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (size_t t = 0; t < 5; ++t)
        for (size_t k = 0; k < 2; ++k)
            x[t][k] = nd(rng);

    Tensor out = ltc.forward(x);
    Tensor grad_out(out.rows, out.cols);
    for (size_t t = 0; t < out.rows; ++t)
        for (size_t i = 0; i < out.cols; ++i)
            grad_out[t][i] = out[t][i];  // dL/dh = h for L = 0.5 * h^2

    // Analytical grads
    ltc.backward(grad_out, 0.0);
    double ana_W_ih = ltc.grad_W_ih_[0][0];
    double ana_W_hh = ltc.grad_W_hh_[0][0];
    double ana_b    = ltc.grad_b_[0][0];
    double ana_W_tx = ltc.grad_W_tx_[0][0];
    double ana_W_th = ltc.grad_W_th_[0][0];
    double ana_b_t  = ltc.grad_b_t_[0][0];
    double ana_ltb  = ltc.grad_log_tau_base_[0][0];

    FDCheck fd(ltc, x, grad_out, 1e-5);

    // W_ih[0][0]
    fd.check(
        [](LTC& l, double v) { l.W_ih_[0][0] = v; },
        [](LTC& l) { return l.W_ih_[0][0]; },
        0.10, ana_W_ih, "Test 10: W_ih FD vs analytical"
    );
    // W_hh[0][0]
    fd.check(
        [](LTC& l, double v) { l.W_hh_[0][0] = v; },
        [](LTC& l) { return l.W_hh_[0][0]; },
        0.05, ana_W_hh, "Test 11: W_hh FD vs analytical"
    );
    // b[0][0]
    fd.check(
        [](LTC& l, double v) { l.b_[0][0] = v; },
        [](LTC& l) { return l.b_[0][0]; },
        0.00, ana_b, "Test 12: b FD vs analytical"
    );
    // W_tx[0][0]
    fd.check(
        [](LTC& l, double v) { l.W_tx_[0][0] = v; },
        [](LTC& l) { return l.W_tx_[0][0]; },
        0.02, ana_W_tx, "Test 13: W_tx FD vs analytical"
    );
    // W_th[0][0]
    fd.check(
        [](LTC& l, double v) { l.W_th_[0][0] = v; },
        [](LTC& l) { return l.W_th_[0][0]; },
        0.02, ana_W_th, "Test 14: W_th FD vs analytical"
    );
    // b_t[0][0]
    fd.check(
        [](LTC& l, double v) { l.b_t_[0][0] = v; },
        [](LTC& l) { return l.b_t_[0][0]; },
        0.00, ana_b_t, "Test 15: b_t FD vs analytical"
    );
    // log_tau_base[0][0]
    fd.check(
        [](LTC& l, double v) { l.log_tau_base_[0][0] = v; },
        [](LTC& l) { return l.log_tau_base_[0][0]; },
        0.5, ana_ltb, "Test 16: log_tau_base FD vs analytical"
    );
}

// ----------------------------------------------------------------------------
// Test 17: Input gradient check
// ----------------------------------------------------------------------------
static void test_input_gradient() {
    LTC ltc(2, 3);
    auto set_val = [](Tensor& t, double v) {
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                t[i][j] = v;
    };
    set_val(ltc.W_ih_,  0.10);
    set_val(ltc.W_hh_,  0.05);
    set_val(ltc.W_tx_,  0.02);
    set_val(ltc.W_th_,  0.02);
    set_val(ltc.b_,     0.00);
    set_val(ltc.b_t_,   0.00);
    set_val(ltc.log_tau_base_, 0.5);

    Tensor x(5, 2);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (size_t t = 0; t < 5; ++t)
        for (size_t k = 0; k < 2; ++k)
            x[t][k] = nd(rng);

    Tensor out = ltc.forward(x);
    Tensor grad_out(out.rows, out.cols);
    for (size_t t = 0; t < out.rows; ++t)
        for (size_t i = 0; i < out.cols; ++i)
            grad_out[t][i] = out[t][i];

    Tensor grad_input = ltc.backward(grad_out, 0.0);
    double ana = grad_input[0][0];

    // FD on x[0][0]
    double orig = x[0][0];
    double eps = 1e-5;
    x[0][0] = orig + eps;
    Tensor out_p = ltc.forward(x);
    double Lp = 0.0;
    for (size_t t = 0; t < out_p.rows; ++t)
        for (size_t i = 0; i < out_p.cols; ++i)
            Lp += grad_out[t][i] * out_p[t][i];
    x[0][0] = orig - eps;
    Tensor out_m = ltc.forward(x);
    double Lm = 0.0;
    for (size_t t = 0; t < out_m.rows; ++t)
        for (size_t i = 0; i < out_m.cols; ++i)
            Lm += grad_out[t][i] * out_m[t][i];
    x[0][0] = orig;
    double num = (Lp - Lm) / (2.0 * eps);

    double denom = std::max({std::abs(num), std::abs(ana), 1e-12});
    double rel_err = std::abs(num - ana) / denom;
    CHECK(rel_err < 1e-3, "Test 17: input gradient FD vs analytical (rel_err < 1e-3)");
}

// ----------------------------------------------------------------------------
// Test 18: Training reduces loss
// ----------------------------------------------------------------------------
static void test_training_reduces_loss() {
    LTC ltc(3, 4);
    Tensor x(6, 3);
    std::mt19937 rng(13);
    std::normal_distribution<double> nd(0.0, 0.5);

    double initial_loss = 0.0;
    // Compute initial loss on a fresh batch
    for (size_t t = 0; t < 6; ++t)
        for (size_t k = 0; k < 3; ++k)
            x[t][k] = nd(rng);
    {
        Tensor out = ltc.forward(x);
        double target = 0.5;
        for (size_t t = 0; t < out.rows; ++t)
            initial_loss += 0.5 * (out[t][0] - target) * (out[t][0] - target) / 6.0;
    }

    // Train 50 SGD steps
    for (size_t step = 0; step < 50; ++step) {
        for (size_t t = 0; t < 6; ++t)
            for (size_t k = 0; k < 3; ++k)
                x[t][k] = nd(rng);
        Tensor out = ltc.forward(x);
        Tensor grad_out(out.rows, out.cols);
        double target = 0.5;
        for (size_t t = 0; t < out.rows; ++t)
            for (size_t i = 0; i < out.cols; ++i)
                grad_out[t][i] = (i == 0) ? (out[t][0] - target) / 6.0 : 0.0;
        ltc.backward(grad_out, 0.0);
        ltc.update_weights(0.05);
    }

    // Final loss on a fresh batch
    double final_loss = 0.0;
    for (size_t t = 0; t < 6; ++t)
        for (size_t k = 0; k < 3; ++k)
            x[t][k] = nd(rng);
    Tensor out_f = ltc.forward(x);
    double target = 0.5;
    for (size_t t = 0; t < out_f.rows; ++t)
        final_loss += 0.5 * (out_f[t][0] - target) * (out_f[t][0] - target) / 6.0;

    CHECK(final_loss < initial_loss,
          "Test 18: LTC training reduces loss (" +
          std::to_string(initial_loss).substr(0, 5) + " -> " +
          std::to_string(final_loss).substr(0, 5) + ")");
}

// ----------------------------------------------------------------------------
// Test 19-21: parameters/gradients/zero_grad/update_weights contract
// ----------------------------------------------------------------------------
static void test_param_grad_zero_update() {
    LTC ltc(2, 2);
    Tensor x(2, 2);
    x[0][0] = 0.1; x[0][1] = 0.2; x[1][0] = 0.3; x[1][1] = 0.4;

    Tensor out = ltc.forward(x);
    Tensor grad_out(out.rows, out.cols);
    for (size_t t = 0; t < out.rows; ++t)
        for (size_t i = 0; i < out.cols; ++i) grad_out[t][i] = 0.5;
    ltc.backward(grad_out, 0.0);

    auto params = ltc.parameters();
    auto grads = ltc.gradients();
    CHECK(params.size() == 7, "Test 19a: parameters() returns 7 tensors");
    CHECK(grads.size() == 7, "Test 19b: gradients() returns 7 tensors");

    // After backward, at least one gradient is nonzero
    bool has_nonzero = false;
    for (size_t i = 0; i < ltc.grad_W_ih_.rows; ++i)
        for (size_t k = 0; k < ltc.grad_W_ih_.cols; ++k)
            if (std::abs(ltc.grad_W_ih_[i][k]) > 1e-10) has_nonzero = true;
    CHECK(has_nonzero, "Test 19c: grad_W_ih nonzero after backward");

    ltc.zero_grad();
    bool all_zero = true;
    auto check_zero = [&](const Tensor& t) {
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                if (std::abs(t[i][j]) > 1e-10) { all_zero = false; return; }
    };
    check_zero(ltc.grad_W_ih_);
    check_zero(ltc.grad_W_hh_);
    check_zero(ltc.grad_b_);
    check_zero(ltc.grad_W_tx_);
    check_zero(ltc.grad_W_th_);
    check_zero(ltc.grad_b_t_);
    check_zero(ltc.grad_log_tau_base_);
    CHECK(all_zero, "Test 20: zero_grad clears all 7 gradient tensors");

    // update_weights moves params
    Tensor x2(2, 2);
    x2[0][0] = 0.1; x2[0][1] = 0.2; x2[1][0] = 0.3; x2[1][1] = 0.4;
    Tensor out2 = ltc.forward(x2);
    Tensor grad_out2(out2.rows, out2.cols);
    for (size_t t = 0; t < out2.rows; ++t)
        for (size_t i = 0; i < out2.cols; ++i) grad_out2[t][i] = 1.0;
    ltc.backward(grad_out2, 0.0);

    double orig_W_ih = ltc.W_ih_[0][0];
    double orig_W_hh = ltc.W_hh_[0][0];
    double orig_b    = ltc.b_[0][0];
    double orig_W_tx = ltc.W_tx_[0][0];
    double orig_W_th = ltc.W_th_[0][0];
    double orig_b_t  = ltc.b_t_[0][0];
    double orig_ltb  = ltc.log_tau_base_[0][0];

    ltc.update_weights(0.1);

    CHECK(std::abs(ltc.W_ih_[0][0] - orig_W_ih) > 1e-8, "Test 21a: W_ih moved after update_weights");
    CHECK(std::abs(ltc.W_hh_[0][0] - orig_W_hh) > 1e-8, "Test 21b: W_hh moved after update_weights");
    CHECK(std::abs(ltc.b_[0][0] - orig_b) > 1e-8, "Test 21c: b moved after update_weights");
    CHECK(std::abs(ltc.W_tx_[0][0] - orig_W_tx) > 1e-8, "Test 21d: W_tx moved after update_weights");
    CHECK(std::abs(ltc.W_th_[0][0] - orig_W_th) > 1e-8, "Test 21e: W_th moved after update_weights");
    CHECK(std::abs(ltc.b_t_[0][0] - orig_b_t) > 1e-8, "Test 21f: b_t moved after update_weights");
    CHECK(std::abs(ltc.log_tau_base_[0][0] - orig_ltb) > 1e-8, "Test 21g: log_tau_base moved after update_weights");
}

// ----------------------------------------------------------------------------
// Test 22: Determinism (bit-exact with copied params)
// ----------------------------------------------------------------------------
static void test_determinism() {
    LTC ltc1(2, 3), ltc2(2, 3);
    ltc2.W_ih_ = ltc1.W_ih_;
    ltc2.W_hh_ = ltc1.W_hh_;
    ltc2.b_    = ltc1.b_;
    ltc2.W_tx_ = ltc1.W_tx_;
    ltc2.W_th_ = ltc1.W_th_;
    ltc2.b_t_  = ltc1.b_t_;
    ltc2.log_tau_base_ = ltc1.log_tau_base_;

    Tensor x(4, 2);
    std::mt19937 rng(99);
    std::normal_distribution<double> nd(0.0, 0.5);
    for (size_t t = 0; t < 4; ++t)
        for (size_t k = 0; k < 2; ++k)
            x[t][k] = nd(rng);

    Tensor out1 = ltc1.forward(x);
    Tensor out2 = ltc2.forward(x);
    bool identical = true;
    for (size_t t = 0; t < 4; ++t)
        for (size_t i = 0; i < 3; ++i)
            if (std::abs(out1[t][i] - out2[t][i]) > 1e-12) identical = false;
    CHECK(identical, "Test 22: two LTC with copied params produce bit-identical forward");
}

// ----------------------------------------------------------------------------
// Test 23: Longer sequence (T=10) W_hh gradient check
// ----------------------------------------------------------------------------
static void test_long_sequence_grad() {
    LTC ltc(3, 4);
    std::mt19937 rng(17);
    std::normal_distribution<double> nd(0.0, 0.3);
    auto rand_init = [&](Tensor& t) {
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                t[i][j] = nd(rng);
    };
    rand_init(ltc.W_ih_);
    rand_init(ltc.W_hh_);
    rand_init(ltc.W_tx_);
    rand_init(ltc.W_th_);
    rand_init(ltc.b_);
    rand_init(ltc.b_t_);
    rand_init(ltc.log_tau_base_);

    Tensor x(10, 3);
    for (size_t t = 0; t < 10; ++t)
        for (size_t k = 0; k < 3; ++k)
            x[t][k] = nd(rng);

    Tensor out = ltc.forward(x);
    Tensor grad_out(out.rows, out.cols);
    for (size_t t = 0; t < out.rows; ++t)
        for (size_t i = 0; i < out.cols; ++i)
            grad_out[t][i] = nd(rng);  // random grad to exercise all paths

    ltc.backward(grad_out, 0.0);
    double ana = ltc.grad_W_hh_[0][0];

    // FD on W_hh[0][0]
    double orig = ltc.W_hh_[0][0];
    double eps = 1e-5;
    ltc.W_hh_[0][0] = orig + eps;
    double Lp = dot_loss(ltc, x, grad_out);
    ltc.W_hh_[0][0] = orig - eps;
    double Lm = dot_loss(ltc, x, grad_out);
    ltc.W_hh_[0][0] = orig;
    double num = (Lp - Lm) / (2.0 * eps);

    double denom = std::max({std::abs(num), std::abs(ana), 1e-12});
    double rel_err = std::abs(num - ana) / denom;
    CHECK(rel_err < 1e-3, "Test 23: W_hh FD vs analytical rel_err < 1e-3 (T=10, hidden=4)");
}

int main() {
    std::cout << "=== LTC Tests ===\n";

    test_constructor_validation();
    test_forward_shape_known_params();
    test_forward_hand_derived_single_neuron();
    test_forward_finite_bounded_random();
    test_cache_shapes_ranges();
    test_backward_single_step_hand_derived();
    test_full_numerical_grad_check();
    test_input_gradient();
    test_training_reduces_loss();
    test_param_grad_zero_update();
    test_determinism();
    test_long_sequence_grad();

    std::cout << "=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}