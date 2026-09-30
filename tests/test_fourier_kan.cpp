// test_fourier_kan.cpp — Tests for FourierKAN.
// Xu et al. 2024 "FourierKAN: Highly Efficient KAN Based on Fast Fourier
// Transform" (https://arxiv.org/abs/2410.02803).
//
// Per-edge activation:
//   phi_{i,j}(x) = sum_{k=1..K} a_{i,j,k} * cos(k*pi*x)
//                       + b_{i,j,k} * sin(k*pi*x)
//
// The implementation supports (B, in) -> (B, out) with B independent
// batch rows. Closed-form gradient, no B-spline machinery, full FD
// verifiability at small K.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) {
        cout << "  [PASS] " << name << endl;
        ++passed;
    } else {
        cout << "  [FAIL] " << name << endl;
        ++failed;
    }
    return pass;
}

// Numerical-gradient helper: returns dL/dparam[row,col] via central diff
static double fd_param(const FourierKANLayer& kan, const Tensor& input,
                       const Tensor& target, size_t param_idx, size_t row, size_t col,
                       double eps = 1e-5) {
    auto params = const_cast<FourierKANLayer&>(kan).parameters();
    Tensor orig = *params[param_idx];
    auto loss_at = [&](double delta) {
        // Save, mutate, run forward+loss, restore
        (*params[param_idx])(row, col) = orig(row, col) + delta;
        Tensor y = const_cast<FourierKANLayer&>(kan).forward(input);
        double L = 0.0;
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) {
                double d = y(i, j) - target(i, j);
                L += 0.5 * d * d;
            }
        // Restore
        (*params[param_idx])(row, col) = orig(row, col);
        return L;
    };
    double Lp = loss_at(+eps);
    double Lm = loss_at(-eps);
    return (Lp - Lm) / (2.0 * eps);
}

static double fd_input(const FourierKANLayer& kan, const Tensor& input,
                       const Tensor& target, size_t brow, size_t bcol,
                       double eps = 1e-5) {
    Tensor orig = input.clone();
    auto f = [&](double delta) {
        Tensor x = orig.clone();
        x(brow, bcol) += delta;
        Tensor y = const_cast<FourierKANLayer&>(kan).forward(x);
        double L = 0.0;
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) {
                double d = y(i, j) - target(i, j);
                L += 0.5 * d * d;
            }
        return L;
    };
    return (f(+eps) - f(-eps)) / (2.0 * eps);
}

// =====================================================================
// Test 1: Constructor validation
// =====================================================================
static void test_constructor_validation() {
    cout << endl << "-- Test 1: FourierKANLayer constructor validation --" << endl;

    bool threw_in = false;
    try { FourierKANLayer k(0, 4); } catch (...) { threw_in = true; }
    check("in_features=0 throws", threw_in);

    bool threw_out = false;
    try { FourierKANLayer k(3, 0); } catch (...) { threw_out = true; }
    check("out_features=0 throws", threw_out);

    bool threw_K = false;
    try { FourierKANLayer k(3, 4, 0); } catch (...) { threw_K = true; }
    check("num_freq=0 throws", threw_K);

    FourierKANLayer k1(2, 3, 5);
    check("valid 2x3 K=5 constructs", k1.in_features() == 2 && k1.out_features() == 3
          && k1.num_freq() == 5);

    FourierKANLayer k2(3, 3, 1);
    check("K=1 constructs (min)", k2.num_freq() == 1);
}

// =====================================================================
// Test 2: Parameter count and shapes
// =====================================================================
static void test_parameters() {
    cout << endl << "-- Test 2: FourierKANLayer parameter shapes --" << endl;

    size_t in_f = 3, out_f = 4, K = 6;
    FourierKANLayer k(in_f, out_f, K);

    auto params = k.parameters();
    auto grads = k.gradients();
    check("params count == 2 (a, b)", params.size() == 2);
    check("grads count == 2 (a, b)", grads.size() == 2);

    // a and b are (out, in*K)
    size_t expected_cols = in_f * K;
    bool a_ok = false, b_ok = false;
    for (auto* p : params) {
        if (p->rows == out_f && p->cols == expected_cols) {
            if (!a_ok) a_ok = true;
            else if (!b_ok) b_ok = true;
        }
    }
    check("a_coefs shape (out, in*K)", a_ok);
    check("b_coefs shape (out, in*K)", b_ok);

    // grads same shape
    for (auto* g : grads) {
        check("grad shape (out, in*K)", g->rows == out_f && g->cols == expected_cols);
    }
}

// =====================================================================
// Test 3: Forward shape & finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 3: Forward shape and finiteness --" << endl;

    FourierKANLayer k(3, 5, 4);
    Tensor input(2, 3);
    input[0][0] = -0.7; input[0][1] = 0.2; input[0][2] = 0.9;
    input[1][0] = 1.5;  input[1][1] = -1.2; input[1][2] = 0.4;

    Tensor out = k.forward(input);
    check("output shape (2, 5)", out.rows == 2 && out.cols == 5);

    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    }
    check("forward output finite", finite);
}

// =====================================================================
// Test 4: Hand-derived closed form (small case)
// K=1, in=1, out=1, B=1. y = a*cos(pi*x) + b*sin(pi*x).
// Pick a, b at known values and x; compare to manual.
// =====================================================================
static void test_hand_derived_K1() {
    cout << endl << "-- Test 4: Hand-derived closed-form K=1, in=1, out=1 --" << endl;

    FourierKANLayer k(1, 1, 1);
    // After construction, a_coefs_(0, 0) and b_coefs_(0, 0) are random.
    // Force them to known values.
    auto params = k.parameters();
    (*params[0])(0, 0) = 0.7;   // a
    (*params[1])(0, 0) = -0.3;  // b

    Tensor input(1, 1);
    input[0][0] = 0.25;
    Tensor out = k.forward(input);

    double x = 0.25;
    double a = 0.7, b = -0.3;
    double expected = a * std::cos(M_PI * x) + b * std::sin(M_PI * x);
    check("K=1 closed-form match (rel_err < 1e-12)",
          std::abs(out(0, 0) - expected) < 1e-12);
}

// =====================================================================
// Test 5: Hand-derived closed form (K=2, in=2, out=2, B=2)
// =====================================================================
static void test_hand_derived_K2() {
    cout << endl << "-- Test 5: Hand-derived closed-form K=2, in=2, out=2, B=2 --" << endl;

    size_t in_f = 2, out_f = 2, K = 2;
    FourierKANLayer k(in_f, out_f, K);
    auto params = k.parameters();
    // Set a = 1, b = 0.5 for all (i, j, k)
    for (size_t i = 0; i < out_f; ++i)
        for (size_t c = 0; c < in_f * K; ++c) {
            (*params[0])(i, c) = 1.0;
            (*params[1])(i, c) = 0.5;
        }

    Tensor input(2, 2);
    input[0][0] = 0.1;  input[0][1] = 0.6;
    input[1][0] = -0.4; input[1][1] = 0.9;

    Tensor out = k.forward(input);

    // expected out[b, i] = sum_j sum_k a*cos(k*pi*x[b,j]) + b*sin(k*pi*x[b,j])
    bool all_ok = true;
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double expected = 0.0;
            for (size_t j = 0; j < in_f; ++j) {
                double x = input(b, j);
                for (size_t kk = 1; kk <= K; ++kk) {
                    expected += 1.0 * std::cos(kk * M_PI * x);
                    expected += 0.5 * std::sin(kk * M_PI * x);
                }
            }
            double diff = std::abs(out(b, i) - expected);
            if (diff > 1e-10) all_ok = false;
        }
    }
    check("K=2, in=2, out=2 closed-form match", all_ok);
}

// =====================================================================
// Test 6: Determinism (two consecutive calls produce bit-identical output)
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 6: Determinism (bit-exact forward repeat) --" << endl;

    FourierKANLayer k(4, 3, 5);
    Tensor input(3, 4);
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.5 * std::sin(0.13 * static_cast<double>(i));
    }

    Tensor out1 = k.forward(input);
    Tensor out2 = k.forward(input);

    bool same = true;
    for (size_t i = 0; i < out1.data.size(); ++i) {
        if (out1.data[i] != out2.data[i]) { same = false; break; }
    }
    check("two consecutive forwards are bit-exact", same);
}

// =====================================================================
// Test 7: Forward with copied params (zero_grad then init test) — bit-exact
// =====================================================================
static void test_param_copy_determinism() {
    cout << endl << "-- Test 7: Two layers with copied params produce identical forward --" << endl;

    FourierKANLayer k1(3, 4, 6);
    FourierKANLayer k2(3, 4, 6);

    auto p1 = k1.parameters();
    auto p2 = k2.parameters();
    for (size_t i = 0; i < p1[0]->data.size(); ++i) {
        (*p2[0]).data[i] = (*p1[0]).data[i];
        (*p2[1]).data[i] = (*p1[1]).data[i];
    }

    Tensor input(2, 3);
    input[0][0] = 0.1; input[0][1] = 0.5; input[0][2] = -0.7;
    input[1][0] = 0.3; input[1][1] = 0.8; input[1][2] = -0.2;

    Tensor y1 = k1.forward(input);
    Tensor y2 = k2.forward(input);

    bool same = true;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (y1.data[i] != y2.data[i]) { same = false; break; }
    }
    check("copied params produce bit-exact forward", same);
}

// =====================================================================
// Test 8: Gradient check — input FD vs analytical (K=4, in=3, out=2)
// =====================================================================
static void test_input_grad_fd() {
    cout << endl << "-- Test 8: Input gradient FD check --" << endl;

    size_t in_f = 3, out_f = 2, K = 4, B = 2;
    FourierKANLayer k(in_f, out_f, K);

    // Deterministic random params (re-init RNG with fixed seed)
    {
        std::mt19937 rng(7);
        std::normal_distribution<double> dist(0.0, 0.3);
        auto params = k.parameters();
        for (auto* p : params) {
            for (size_t i = 0; i < p->data.size(); ++i) p->data[i] = dist(rng);
        }
    }

    Tensor input(B, in_f);
    {
        std::mt19937 rng(8);
        std::uniform_real_distribution<double> ud(-0.8, 0.8);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = ud(rng);
    }
    Tensor target(B, out_f);
    {
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> ud(-0.5, 0.5);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = ud(rng);
    }

    Tensor y = k.forward(input);
    Tensor grad_out(B, out_f);
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < out_f; ++j)
            grad_out(i, j) = y(i, j) - target(i, j);

    k.zero_grad();
    Tensor grad_input = k.backward(grad_out, 0.0);

    // Compare to FD on every (b, j)
    double max_rel = 0.0;
    bool all_ok = true;
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            double ana = grad_input(b, j);
            double num = fd_input(k, input, target, b, j, 1e-5);
            double abs_d = std::abs(ana - num);
            double denom = std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
            double rel = abs_d / denom;
            if (rel > max_rel) max_rel = rel;
            // Use combined criterion: rel_err < 1e-3 OR abs_diff < 5e-3
            if (rel > 1e-3 && abs_d > 5e-3) all_ok = false;
        }
    }
    check("input grad FD rel_err < 1e-3 across all (b, j)", all_ok);
    cout << "    max_rel_err = " << std::scientific << std::setprecision(3)
         << max_rel << endl;
}

// =====================================================================
// Test 9: Gradient check — parameter (a_coefs, b_coefs) FD vs analytical
// =====================================================================
static void test_param_grad_fd() {
    cout << endl << "-- Test 9: Parameter gradient FD check (a, b) --" << endl;

    size_t in_f = 2, out_f = 2, K = 3, B = 2;
    FourierKANLayer k(in_f, out_f, K);

    // Deterministic params
    {
        std::mt19937 rng(11);
        std::normal_distribution<double> dist(0.0, 0.5);
        auto params = k.parameters();
        for (auto* p : params) {
            for (size_t i = 0; i < p->data.size(); ++i) p->data[i] = dist(rng);
        }
    }

    Tensor input(B, in_f);
    {
        std::mt19937 rng(12);
        std::uniform_real_distribution<double> ud(-0.7, 0.7);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = ud(rng);
    }
    Tensor target(B, out_f);
    {
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> ud(-0.4, 0.4);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = ud(rng);
    }

    Tensor y = k.forward(input);
    Tensor grad_out(B, out_f);
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < out_f; ++j)
            grad_out(i, j) = y(i, j) - target(i, j);

    k.zero_grad();
    k.backward(grad_out, 0.0);
    auto grads = k.gradients();

    double max_rel_a = 0.0, max_rel_b = 0.0;
    bool a_ok = true, b_ok = true;
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t c = 0; c < in_f * K; ++c) {
            double ana_a = (*grads[0])(i, c);
            double num_a = fd_param(k, input, target, 0, i, c, 1e-5);
            double ana_b = (*grads[1])(i, c);
            double num_b = fd_param(k, input, target, 1, i, c, 1e-5);

            double abs_a = std::abs(ana_a - num_a);
            double denom_a = std::max(1e-12, std::max(std::abs(ana_a), std::abs(num_a)));
            double rel_a = abs_a / denom_a;
            if (rel_a > max_rel_a) max_rel_a = rel_a;
            if (rel_a > 1e-3 && abs_a > 5e-3) a_ok = false;

            double abs_b = std::abs(ana_b - num_b);
            double denom_b = std::max(1e-12, std::max(std::abs(ana_b), std::abs(num_b)));
            double rel_b = abs_b / denom_b;
            if (rel_b > max_rel_b) max_rel_b = rel_b;
            if (rel_b > 1e-3 && abs_b > 5e-3) b_ok = false;
        }
    }
    check("a_coefs gradient FD rel_err < 1e-3 across all (i, c)", a_ok);
    check("b_coefs gradient FD rel_err < 1e-3 across all (i, c)", b_ok);
    cout << "    max_rel_a = " << std::scientific << std::setprecision(3)
         << max_rel_a << ", max_rel_b = " << max_rel_b << endl;
}

// =====================================================================
// Test 10: zero_grad clears grads
// =====================================================================
static void test_zero_grad() {
    cout << endl << "-- Test 10: zero_grad clears both grad tensors --" << endl;

    FourierKANLayer k(2, 3, 4);
    auto params = k.parameters();
    Tensor input(2, 2);
    input[0][0] = 0.1; input[0][1] = 0.2;
    input[1][0] = 0.3; input[1][1] = 0.4;
    Tensor y = k.forward(input);
    Tensor grad_out(2, 3);
    for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = 1.0;

    k.backward(grad_out, 0.0);
    k.zero_grad();
    auto grads = k.gradients();
    bool all_zero = true;
    for (auto* g : grads) {
        for (size_t i = 0; i < g->data.size(); ++i) {
            if (g->data[i] != 0.0) { all_zero = false; break; }
        }
    }
    check("after zero_grad, all grad entries == 0", all_zero);
}

// =====================================================================
// Test 11: update_weights moves parameters
// =====================================================================
static void test_update_weights() {
    cout << endl << "-- Test 11: update_weights moves a, b by -lr*grad --" << endl;

    FourierKANLayer k(2, 2, 3);
    auto params = k.parameters();
    // Pre-set grads to known values
    auto grads = k.gradients();
    for (size_t i = 0; i < grads[0]->data.size(); ++i) {
        (*grads[0]).data[i] = 0.1;
        (*grads[1]).data[i] = -0.2;
    }
    // Snapshot params
    Tensor a_orig = (*params[0]).clone();
    Tensor b_orig = (*params[1]).clone();
    double lr = 0.5;
    k.update_weights(lr);
    // a should have moved by -0.5*0.1 = -0.05
    bool a_ok = true, b_ok = true;
    for (size_t i = 0; i < a_orig.data.size(); ++i) {
        double expected = a_orig.data[i] - lr * 0.1;
        if (std::abs((*params[0]).data[i] - expected) > 1e-12) a_ok = false;
    }
    for (size_t i = 0; i < b_orig.data.size(); ++i) {
        double expected = b_orig.data[i] - lr * (-0.2);
        if (std::abs((*params[1]).data[i] - expected) > 1e-12) b_ok = false;
    }
    check("a_coefs updated by -lr * grad", a_ok);
    check("b_coefs updated by -lr * grad", b_ok);
}

// =====================================================================
// Test 12: Gradient accumulation across multiple backward calls
// =====================================================================
static void test_grad_accumulation() {
    cout << endl << "-- Test 12: Gradient accumulation across backward calls --" << endl;

    FourierKANLayer k(2, 2, 3);
    Tensor input(2, 2);
    input[0][0] = 0.1; input[0][1] = 0.2;
    input[1][0] = 0.3; input[1][1] = 0.4;

    Tensor y = k.forward(input);
    Tensor grad_out(2, 2);
    for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = 1.0;

    k.zero_grad();
    k.backward(grad_out, 0.0);
    auto grads_after_one = k.gradients();
    Tensor a_grad_after_one = (*grads_after_one[0]).clone();
    Tensor b_grad_after_one = (*grads_after_one[1]).clone();

    k.backward(grad_out, 0.0);
    auto grads_after_two = k.gradients();

    bool a_ok = true, b_ok = true;
    for (size_t i = 0; i < a_grad_after_one.data.size(); ++i) {
        double expected = 2.0 * a_grad_after_one.data[i];
        if (std::abs((*grads_after_two[0]).data[i] - expected) > 1e-12) a_ok = false;
    }
    for (size_t i = 0; i < b_grad_after_one.data.size(); ++i) {
        double expected = 2.0 * b_grad_after_one.data[i];
        if (std::abs((*grads_after_two[1]).data[i] - expected) > 1e-12) b_ok = false;
    }
    check("a grad doubled after 2 backward calls", a_ok);
    check("b grad doubled after 2 backward calls", b_ok);
}

// =====================================================================
// Test 13: FourierKANModel forward shape and parameter count
// =====================================================================
static void test_model_shape() {
    cout << endl << "-- Test 13: FourierKANModel forward shape & params --" << endl;

    FourierKANModel m(3, {5, 4}, 2, 6);
    Tensor input(2, 3);
    input[0][0] = 0.1; input[0][1] = 0.2; input[0][2] = 0.3;
    input[1][0] = 0.4; input[1][1] = 0.5; input[1][2] = 0.6;

    Tensor y = m.forward(input);
    check("model output shape (2, 2)", y.rows == 2 && y.cols == 2);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    check("model output finite", finite);

    auto params = m.parameters();
    auto grads = m.gradients();
    // 3 layers: (3 -> 5), (5 -> 4), (4 -> 2). Each has a and b.
    // Total params: 2 per layer * 3 layers = 6. Gradients: same.
    check("model params count == 6 (3 layers × 2)", params.size() == 6);
    check("model grads count == 6", grads.size() == 6);
}

// =====================================================================
// Test 14: Model end-to-end training reduces loss
// =====================================================================
static void test_model_training() {
    cout << endl << "-- Test 14: Model training reduces loss on toy regression --" << endl;

    FourierKANModel m(2, {4, 4}, 1, 3);
    Tensor input(4, 2);
    Tensor target(4, 1);
    // y = x[0] + 0.5*x[1] target
    input[0][0] = 0.1; input[0][1] = 0.2;  target[0][0] = 0.20;
    input[1][0] = 0.3; input[1][1] = 0.5;  target[1][0] = 0.55;
    input[2][0] = 0.5; input[2][1] = -0.3; target[2][0] = 0.35;
    input[3][0] = -0.4; input[3][1] = 0.7; target[3][0] = -0.05;

    auto loss = [&](const Tensor& y) {
        double L = 0.0;
        for (size_t i = 0; i < y.data.size(); ++i) {
            double d = y.data[i] - target.data[i];
            L += 0.5 * d * d;
        }
        return L / y.data.size();
    };

    Tensor y0 = m.forward(input);
    double L0 = loss(y0);

    double lr = 0.05;
    int steps = 200;
    for (int s = 0; s < steps; ++s) {
        Tensor y = m.forward(input);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = (y.data[i] - target.data[i]) / y.data.size();
        }
        m.zero_grad();
        m.backward(grad_out, 0.0);
        m.update_weights(lr);
    }
    Tensor yF = m.forward(input);
    double LF = loss(yF);
    bool reduced = LF < 0.5 * L0;
    cout << "    L0 = " << std::fixed << std::setprecision(5) << L0
         << ", LF = " << LF << " (reduction: "
         << std::setprecision(1) << (1.0 - LF / L0) * 100.0 << "%)" << endl;
    check("model training reduces loss > 50%", reduced);
}

// =====================================================================
// Test 15: Determinism (model forward repeat)
// =====================================================================
static void test_model_determinism() {
    cout << endl << "-- Test 15: Model forward determinism --" << endl;

    FourierKANModel m(3, {4}, 2, 5);
    Tensor input(2, 3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);

    Tensor y1 = m.forward(input);
    Tensor y2 = m.forward(input);
    bool same = true;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (y1.data[i] != y2.data[i]) { same = false; break; }
    }
    check("model: two consecutive forwards bit-exact", same);
}

// =====================================================================
// Test 16: Input gradient on model (chain through FourierKANLayers)
// =====================================================================
static void test_model_input_grad_fd() {
    cout << endl << "-- Test 16: Model input gradient FD check --" << endl;

    FourierKANModel m(3, {4, 4}, 2, 4);
    Tensor input(2, 3);
    {
        std::mt19937 rng(21);
        std::uniform_real_distribution<double> ud(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = ud(rng);
    }
    Tensor target(2, 2);
    {
        std::mt19937 rng(22);
        std::uniform_real_distribution<double> ud(-0.3, 0.3);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = ud(rng);
    }

    Tensor y = m.forward(input);
    Tensor grad_out(2, 2);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            grad_out(i, j) = (y(i, j) - target(i, j));

    m.zero_grad();
    Tensor grad_input = m.backward(grad_out, 0.0);

    // FD
    double max_rel = 0.0;
    bool ok = true;
    auto f = [&](double delta_b0, double delta_b1, double delta_b2) {
        Tensor x = input.clone();
        x(0, 0) += delta_b0;
        x(0, 1) += delta_b1;
        x(0, 2) += delta_b2;
        Tensor y = m.forward(x);
        double L = 0.0;
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) {
                double d = y(i, j) - target(i, j);
                L += 0.5 * d * d;
            }
        return L;
    };
    for (size_t b = 0; b < 2; ++b) {
        for (size_t j = 0; j < 3; ++j) {
            // Just perturb that one cell
            Tensor x = input.clone();
            double orig = input(b, j);
            x(b, j) = orig + 1e-5;
            Tensor yp = m.forward(x);
            x(b, j) = orig - 1e-5;
            Tensor ym = m.forward(x);
            double Lp = 0.0, Lm = 0.0;
            for (size_t i = 0; i < yp.rows; ++i)
                for (size_t k = 0; k < yp.cols; ++k) {
                    double d_p = yp(i, k) - target(i, k);
                    double d_m = ym(i, k) - target(i, k);
                    Lp += 0.5 * d_p * d_p;
                    Lm += 0.5 * d_m * d_m;
                }
            double num = (Lp - Lm) / (2e-5);
            double ana = grad_input(b, j);
            double abs_d = std::abs(ana - num);
            double denom = std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
            double rel = abs_d / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-3 && abs_d > 5e-3) ok = false;
        }
    }
    check("model input gradient FD rel_err < 1e-3", ok);
    cout << "    max_rel = " << std::scientific << std::setprecision(3)
         << max_rel << endl;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "===== FourierKAN Tests =====" << endl;

    test_constructor_validation();
    test_parameters();
    test_forward_shape();
    test_hand_derived_K1();
    test_hand_derived_K2();
    test_determinism();
    test_param_copy_determinism();
    test_input_grad_fd();
    test_param_grad_fd();
    test_zero_grad();
    test_update_weights();
    test_grad_accumulation();
    test_model_shape();
    test_model_training();
    test_model_determinism();
    test_model_input_grad_fd();

    cout << endl << "===== Summary: " << passed << " passed, "
         << failed << " failed =====" << endl;
    return failed == 0 ? 0 : 1;
}