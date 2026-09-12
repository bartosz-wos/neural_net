// tests/test_s5.cpp
// S5 (Simplified State Space Layers) — Smith et al. ICLR 2022
//
// Focused test plan (8 tests). Mutation-tested non-vacuous where applicable.

#include "nn/layers/recurrent/s5.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

// Note: the Layer / Tensor / Dense classes are in the global namespace,
// not inside `nn::`. The headers live under include/nn/ but the classes
// are at top level (a quirk of this repo's structure).

static int n_pass = 0;
static int n_fail = 0;
static int test_idx = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  [FAIL] " << msg << std::endl; \
        n_fail++; \
    } else { \
        n_pass++; \
    } \
} while(0)

#define TEST_START(name) do { \
    test_idx++; \
    std::cout << "--- Test " << test_idx << ": " << name << " ---" << std::endl; \
} while(0)

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// Finite-difference gradient checker. Returns max relative error.
static double fd_check_input(S5Layer& l, const Tensor& input, double eps = 1e-5) {
    Tensor y = l.forward(input);
    // dummy loss = sum(y); grad = 1 everywhere
    Tensor grad(y.rows, y.cols);
    grad.fill(1.0);
    l.zero_grad();
    Tensor analytical = l.backward(grad, 0.0);

    Tensor input_plus = input.clone();
    Tensor input_minus = input.clone();
    double max_rel = 0.0;
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            input_plus(i, j) = input(i, j) + eps;
            input_minus(i, j) = input(i, j) - eps;
            Tensor y_plus = l.forward(input_plus);
            Tensor y_minus = l.forward(input_minus);
            double f_plus = 0.0, f_minus = 0.0;
            for (size_t r = 0; r < y.rows; ++r)
                for (size_t c = 0; c < y.cols; ++c) {
                    f_plus += y_plus(r, c);
                    f_minus += y_minus(r, c);
                }
            double num = (f_plus - f_minus) / (2.0 * eps);
            double ana = analytical(i, j);
            double denom = std::max(std::abs(num), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double rel = std::abs(num - ana) / denom;
            if (rel > max_rel) max_rel = rel;
            input_plus(i, j) = input(i, j);
            input_minus(i, j) = input(i, j);
        }
    }
    return max_rel;
}

static double fd_check_param(S5Layer& l, Tensor& param, const Tensor& grad_param,
                             const Tensor& input, double eps = 1e-5) {
    Tensor y = l.forward(input);
    Tensor grad(y.rows, y.cols);
    grad.fill(1.0);
    l.zero_grad();
    l.backward(grad, 0.0);

    double max_rel = 0.0;
    Tensor saved = param.clone();
    for (size_t i = 0; i < param.rows; ++i) {
        for (size_t j = 0; j < param.cols; ++j) {
            double orig = param(i, j);
            param(i, j) = orig + eps;
            Tensor y_plus = l.forward(input);
            double f_plus = 0.0;
            for (size_t r = 0; r < y.rows; ++r)
                for (size_t c = 0; c < y.cols; ++c) f_plus += y_plus(r, c);

            param(i, j) = orig - eps;
            Tensor y_minus = l.forward(input);
            double f_minus = 0.0;
            for (size_t r = 0; r < y.rows; ++r)
                for (size_t c = 0; c < y.cols; ++c) f_minus += y_minus(r, c);

            param(i, j) = orig;
            double num = (f_plus - f_minus) / (2.0 * eps);
            double ana = grad_param(i, j);
            double denom = std::max(std::abs(num), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double rel = std::abs(num - ana) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    return max_rel;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

static void test_constructor_validation() {
    TEST_START("constructor validation");
    bool threw_dm0 = false, threw_ds0 = false, threw_dtneg = false;
    try { S5Layer l(0, 4); } catch (std::invalid_argument&) { threw_dm0 = true; }
    try { S5Layer l(4, 0); } catch (std::invalid_argument&) { threw_ds0 = true; }
    try { S5Layer l(4, 4, true, -1.0); } catch (std::invalid_argument&) { threw_dtneg = true; }
    CHECK(threw_dm0, "d_model=0 throws");
    CHECK(threw_ds0, "d_state=0 throws");
    CHECK(threw_dtneg, "dt<0 throws");

    // smallest valid
    bool ok_small = true;
    try { S5Layer l(1, 1); } catch (...) { ok_small = false; }
    CHECK(ok_small, "smallest valid constructs (d_model=1, d_state=1)");

    bool ok_uni = true;
    try { S5Layer l(4, 4, /*bidir=*/false); } catch (...) { ok_uni = false; }
    CHECK(ok_uni, "unidirectional constructs");
}

static void test_hippo_eigenvalues() {
    TEST_START("HiPPO-LegS eigenvalues");
    S5Layer l(4, 4, /*bidir=*/false);
    const Tensor& A_re = l.A_re();
    // λ_n = −sqrt((2n+1)(2n+2))
    double expect0 = -std::sqrt(1.0 * 2.0);
    double expect1 = -std::sqrt(3.0 * 4.0);
    double expect2 = -std::sqrt(5.0 * 6.0);
    double expect3 = -std::sqrt(7.0 * 8.0);
    double rel_err = 0.0;
    rel_err = std::max(rel_err, std::abs(A_re(0, 0) - expect0) / std::abs(expect0));
    rel_err = std::max(rel_err, std::abs(A_re(1, 0) - expect1) / std::abs(expect1));
    rel_err = std::max(rel_err, std::abs(A_re(2, 0) - expect2) / std::abs(expect2));
    rel_err = std::max(rel_err, std::abs(A_re(3, 0) - expect3) / std::abs(expect3));
    CHECK(rel_err < 1e-12, "A_n = -sqrt((2n+1)(2n+2)) matches HiPPO-LegS closed form");

    // B_n = sqrt(2n+1)
    const Tensor& B_re = l.B_re();
    double be_rel_err = 0.0;
    be_rel_err = std::max(be_rel_err, std::abs(B_re(0, 0) - std::sqrt(1.0)));
    be_rel_err = std::max(be_rel_err, std::abs(B_re(1, 0) - std::sqrt(3.0)));
    be_rel_err = std::max(be_rel_err, std::abs(B_re(2, 0) - std::sqrt(5.0)));
    CHECK(be_rel_err < 1e-12, "B_n = sqrt(2n+1) matches HiPPO-LegS");

    // A_im and B_im are zero (real eigenvalues)
    CHECK(std::abs(l.A_im()(0, 0)) < 1e-12, "A_im = 0 (real eigenvalues)");
    CHECK(std::abs(l.B_im()(0, 0)) < 1e-12, "B_im = 0 (real B)");
}

static void test_forward_shape() {
    TEST_START("forward shape & finite");
    S5Layer l(4, 8);
    Tensor x(8, 4);
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = dist(gen);
    Tensor y = l.forward(x);
    CHECK(y.rows == 8, "output rows match input rows");
    CHECK(y.cols == 4, "output cols match d_model");
    bool finite = true, nonzero = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y(i, j))) finite = false;
            if (std::abs(y(i, j)) > 1e-6) nonzero = true;
        }
    CHECK(finite, "output is finite");
    CHECK(nonzero, "output is non-trivially nonzero");
}

static void test_hand_derived_forward() {
    TEST_START("hand-derived T=1 forward reference");
    // Single channel, single timestep: x_0 = 1, A = -sqrt(2), B = 1, C = 1, D = 0.
    // dt = 0.1, half_dt = 0.05.
    // I_minus = 1 - half_dt * A_n = 1 - 0.05 * (-sqrt(2)) = 1 + 0.05 * sqrt(2).
    // B̄ = B / I_minus = 1 / (1 + 0.05 * sqrt(2)).
    // h_0 = B̄ * x_0 = B̄.
    // y_0 = Re(C * h_0) = B̄.
    S5Layer l(1, 1, /*bidir=*/false, /*dt=*/0.1);
    // Override the random init
    l.A_re_(0, 0) = -std::sqrt(2.0);
    l.A_im_(0, 0) = 0.0;
    l.B_re_(0, 0) = 1.0;
    l.B_im_(0, 0) = 0.0;
    l.C_re_(0, 0) = 1.0;
    l.C_im_(0, 0) = 0.0;
    l.D_skip_(0, 0) = 0.0;

    Tensor x(1, 1); x(0, 0) = 1.0;
    Tensor y = l.forward(x);
    double expect = 1.0 / (1.0 + 0.05 * std::sqrt(2.0));
    double rel = std::abs(y(0, 0) - expect) / std::abs(expect);
    std::cout << "  y(0,0) = " << y(0, 0) << "  expect = " << expect
              << "  rel_err = " << rel << std::endl;
    CHECK(rel < 1e-12, "T=1 hand-derived forward matches (rel_err < 1e-12)");
}

static void test_fd_input_unidirectional() {
    TEST_START("FD input gradient (unidirectional)");
    S5Layer l(4, 4, /*bidir=*/false);
    Tensor x(4, 4);
    std::mt19937 gen(7);
    std::normal_distribution<double> dist(0.0, 0.3);  // non-trivial scale
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = dist(gen);
    double rel = fd_check_input(l, x);
    std::cout << "  input FD rel_err = " << rel << std::endl;
    CHECK(rel < 1e-3, "FD input gradient at rel_err < 1e-3 (target < 1e-5)");
}

static void test_fd_params_unidirectional() {
    TEST_START("FD parameter gradients (unidirectional)");
    S5Layer l(3, 3, /*bidir=*/false);
    Tensor x(4, 3);
    std::mt19937 gen(11);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = dist(gen);

    // FD for each parameter
    double rel_A_re = fd_check_param(l, l.A_re_, l.grad_A_re_, x);
    double rel_B_re = fd_check_param(l, l.B_re_, l.grad_B_re_, x);
    double rel_C_re = fd_check_param(l, l.C_re_, l.grad_C_re_, x);
    double rel_D    = fd_check_param(l, l.D_skip_, l.grad_D_skip_, x);
    std::cout << "  FD rel_err: A_re=" << rel_A_re << "  B_re=" << rel_B_re
              << "  C_re=" << rel_C_re << "  D=" << rel_D << std::endl;
    CHECK(rel_A_re < 1e-3, "A_re FD < 1e-3");
    CHECK(rel_B_re < 1e-3, "B_re FD < 1e-3");
    CHECK(rel_C_re < 1e-3, "C_re FD < 1e-3");
    CHECK(rel_D    < 1e-3, "D_skip FD < 1e-3");
}

static void test_bidirectional_forward() {
    TEST_START("bidirectional forward shape + finiteness");
    S5Layer l(3, 4, /*bidir=*/true);
    Tensor x(5, 3);
    std::mt19937 gen(99);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) x(i, j) = dist(gen);
    Tensor y = l.forward(x);
    CHECK(y.rows == 5 && y.cols == 3, "bidirectional output shape");
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y(i, j))) finite = false;
    CHECK(finite, "bidirectional output finite");
}

static void test_s5block_training() {
    TEST_START("S5Block end-to-end training (synthetic)");
    // Random target mapping: y[t, m] = sin(0.5 * x[t, m]) + small noise.
    S5Block block(4, 6);
    Tensor x(16, 4);
    Tensor y_target(16, 4);
    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) {
            x(i, j) = dist(gen);
            y_target(i, j) = std::sin(0.5 * x(i, j));
        }

    // Initial loss
    Tensor y_pred = block.forward(x);
    double loss0 = 0.0;
    for (size_t i = 0; i < y_pred.rows; ++i)
        for (size_t j = 0; j < y_pred.cols; ++j) {
            double d = y_pred(i, j) - y_target(i, j);
            loss0 += d * d;
        }
    loss0 /= (y_pred.rows * y_pred.cols);

    // Train 30 SGD steps
    double lr = 0.01;
    for (int step = 0; step < 30; ++step) {
        y_pred = block.forward(x);
        Tensor grad(y_pred.rows, y_pred.cols);
        for (size_t i = 0; i < grad.rows; ++i)
            for (size_t j = 0; j < grad.cols; ++j)
                grad(i, j) = 2.0 * (y_pred(i, j) - y_target(i, j)) / (grad.rows * grad.cols);
        block.zero_grad();
        block.backward(grad, lr);
        block.update_weights(lr);
    }
    y_pred = block.forward(x);
    double loss1 = 0.0;
    for (size_t i = 0; i < y_pred.rows; ++i)
        for (size_t j = 0; j < y_pred.cols; ++j) {
            double d = y_pred(i, j) - y_target(i, j);
            loss1 += d * d;
        }
    loss1 /= (y_pred.rows * y_pred.cols);
    std::cout << "  loss: " << loss0 << " -> " << loss1 << std::endl;
    CHECK(loss1 < loss0, "S5Block training reduces loss");
    CHECK(std::isfinite(loss1), "loss is finite after training");
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main() {
    std::cout << std::fixed << std::setprecision(6);
    test_constructor_validation();
    test_hippo_eigenvalues();
    test_forward_shape();
    test_hand_derived_forward();
    test_fd_input_unidirectional();
    test_fd_params_unidirectional();
    test_bidirectional_forward();
    test_s5block_training();

    std::cout << "\n=== Summary: " << n_pass << " passed, " << n_fail << " failed ==="
              << std::endl;
    return n_fail == 0 ? 0 : 1;
}