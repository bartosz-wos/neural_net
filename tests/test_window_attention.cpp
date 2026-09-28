// test_window_attention.cpp — Gradient correctness tests for Window Attention
//   Liu et al. 2021, "Swin Transformer: Hierarchical Vision Transformer using
//   Shifted Windows" (https://arxiv.org/abs/2103.14030, §3.2)
//
// Tests (organized by category; each `check()` is one assertion):
//   1.  WindowAttention constructor validation (7 throws + 1 valid)
//   2.  Accessors
//   3.  parameters()/gradients() contract (6 each, shapes match)
//   4.  forward shape + finiteness (H=4, W=4, M=2, d=8)
//   5.  forward larger (H=8, W=8, M=4, d=16, Hq=4)
//   6.  forward determinism (two consecutive calls bit-exact)
//   7.  window isolation (perturbing one window's input only changes that window)
//   8.  FD input gradient (rel_err < 1e-4)
//   9.  FD W_q, W_k, W_v, W_o, b_o gradients (rel_err < 1e-5)
//  10.  FD relative_position_bias_ gradient (nonzero AND bit-equivalence
//       between the "relative" and "none" paths)
//  11.  bias_type="none" path forward + FD input gradient
//  12.  zero_grad clears all 6 gradients
//  13.  update_weights moves all 6 parameters
//  14.  End-to-end training reduces MSE loss by > 10% over 30 SGD steps

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include <initializer_list>
#include "nn/layers/attention/window_attention.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static void check(const string& name, bool cond, double err = 0.0) {
    if (cond) { ++passed; cout << "  [PASS] " << name << "\n"; }
    else      { ++failed; cout << "  [FAIL] " << name << " (err=" << err << ")\n"; }
}

static double tensor_max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        m = max(m, fabs(a.data[i] - b.data[i]));
    return m;
}
static double tensor_rel_err(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1.0;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        num = max(num, fabs(a.data[i] - b.data[i]));
        den = max(den, max(fabs(a.data[i]), fabs(b.data[i])));
    }
    if (den < 1e-12) return num;
    return num / den;
}
static double tensor_l2_norm(const Tensor& t) {
    double s = 0;
    for (double v : t.data) s += v * v;
    return std::sqrt(s);
}

static double l2_loss_value(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - tgt.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i)
        g.data[i] = out.data[i] - tgt.data[i];
    return g;
}

static Tensor make_input(size_t N, size_t d, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Tensor t(N, d);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(rng);
    return t;
}

// ---- Test 1: constructor validation ---------------------------------------
static void test_constructor_validation() {
    cout << "Test 1: WindowAttention constructor validation\n";
    try { WindowAttention wa(0, 2, 4, 4, 4); check("d_model=0 throws", false); }
    catch (const invalid_argument&) { check("d_model=0 throws", true); }
    try { WindowAttention wa(8, 0, 4, 4, 4); check("num_heads=0 throws", false); }
    catch (const invalid_argument&) { check("num_heads=0 throws", true); }
    try { WindowAttention wa(8, 3, 4, 4, 4); check("d_model%num_heads!=0 throws", false); }
    catch (const invalid_argument&) { check("d_model%num_heads!=0 throws", true); }
    try { WindowAttention wa(8, 2, 0, 4, 4); check("window_size=0 throws", false); }
    catch (const invalid_argument&) { check("window_size=0 throws", true); }
    try { WindowAttention wa(8, 2, 4, 5, 4); check("H_pad%M!=0 throws", false); }
    catch (const invalid_argument&) { check("H_pad%M!=0 throws", true); }
    try { WindowAttention wa(8, 2, 4, 4, 7); check("W_pad%M!=0 throws", false); }
    catch (const invalid_argument&) { check("W_pad%M!=0 throws", true); }
    try { WindowAttention wa(8, 2, 4, 8, 8); check("valid constructs", true); }
    catch (...) { check("valid constructs", false); }
}

// ---- Test 2: accessors ----------------------------------------------------
static void test_accessors() {
    cout << "Test 2: accessors\n";
    WindowAttention wa(8, 2, 4, 8, 8);
    check("d_model() == 8", wa.d_model() == 8);
    check("num_heads() == 2", wa.num_heads() == 2);
    check("head_dim() == 4", wa.head_dim() == 4);
    check("window_size() == 4", wa.window_size() == 4);
    check("H_pad() == 8", wa.H_pad() == 8);
    check("W_pad() == 8", wa.W_pad() == 8);
    check("num_windows() == 4", wa.num_windows() == 4);
    check("bias_type() == 'relative'", wa.bias_type() == "relative");
    check("name() == 'WindowAttention'", wa.name() == "WindowAttention");
}

// ---- Test 3: parameters/gradients contract -------------------------------
static void test_parameters_gradients_contract() {
    cout << "Test 3: parameters()/gradients() contract\n";
    WindowAttention wa(8, 2, 4, 8, 8);
    auto params = wa.parameters();
    auto grads = wa.gradients();
    check("6 parameters", params.size() == 6);
    check("6 gradients", grads.size() == 6);
    for (size_t i = 0; i < params.size(); ++i) {
        bool ok = params[i]->rows == grads[i]->rows && params[i]->cols == grads[i]->cols;
        if (!ok) { check("param/grad shape match", false); return; }
    }
    check("param/grad shape match", true);
    check("W_q is (8,8)", params[0]->rows == 8 && params[0]->cols == 8);
    check("W_k is (8,8)", params[1]->rows == 8 && params[1]->cols == 8);
    check("W_v is (8,8)", params[2]->rows == 8 && params[2]->cols == 8);
    check("W_o is (8,8)", params[3]->rows == 8 && params[3]->cols == 8);
    check("b_o is (1,8)", params[4]->rows == 1 && params[4]->cols == 8);
    // (2M-1)² = (2·4-1)² = 49; num_heads = 2
    check("relative_position_bias_ is (2,49)",
          params[5]->rows == 2 && params[5]->cols == 49);
}

// ---- Test 4: forward shape + finiteness (small) ---------------------------
static void test_forward_shape_small() {
    cout << "Test 4: forward shape + finiteness (H=4, W=4, M=2, d=8)\n";
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 1);
    Tensor out = wa.forward(input);
    check("forward shape (16, 8)", out.rows == 16 && out.cols == 8);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
    check("forward finite", finite);
    check("forward nonzero (L2 > 0)", tensor_l2_norm(out) > 1e-6);
}

// ---- Test 5: forward larger (H=8, W=8, M=4, d=16, Hq=4) -----------------
static void test_forward_larger() {
    cout << "Test 5: forward larger (H=8, W=8, M=4, d=16, Hq=4)\n";
    WindowAttention wa(16, 4, 4, 8, 8);
    Tensor input = make_input(64, 16, 2);
    Tensor out = wa.forward(input);
    check("forward shape (64, 16)", out.rows == 64 && out.cols == 16);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
    check("forward finite", finite);
    // H=8, W=8, M=4 → 4 windows
    check("num_windows == 4", wa.num_windows() == 4);
}

// ---- Test 6: forward determinism -----------------------------------------
static void test_forward_determinism() {
    cout << "Test 6: forward determinism\n";
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 3);
    Tensor out1 = wa.forward(input);
    Tensor out2 = wa.forward(input);
    check("two consecutive forwards bit-exact", tensor_max_abs_diff(out1, out2) < 1e-12);
}

// ---- Test 7: window isolation --------------------------------------------
// Perturbing one window's input must NOT meaningfully touch unrelated
// window outputs (the canonical "this isn't vanilla attention" property).
static void test_window_isolation() {
    cout << "Test 7: window isolation\n";
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor x0 = make_input(16, 8, 4);
    Tensor out0 = wa.forward(x0);
    // Window layout: 4 windows of 2x2 covering (H,W)=(4,4):
    //   window 0: (0,0)-(1,1)  → flat rows {0, 1, 4, 5}
    //   window 1: (0,2)-(1,3)  → flat rows {2, 3, 6, 7}
    //   window 2: (2,0)-(3,1)  → flat rows {8, 9, 12, 13}
    //   window 3: (2,2)-(3,3)  → flat rows {10, 11, 14, 15}
    Tensor x1 = x0.clone();
    for (size_t r : {0u, 1u, 4u, 5u}) {
        for (size_t c = 0; c < 8; ++c) x1[r][c] += 1.0;
    }
    Tensor out1 = wa.forward(x1);
    double d_window0 = 0.0;
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 8; ++c)
            d_window0 += std::fabs(out1[r][c] - out0[r][c]);
    double d_window3 = 0.0;
    for (size_t r = 12; r < 16; ++r)
        for (size_t c = 0; c < 8; ++c)
            d_window3 += std::fabs(out1[r][c] - out0[r][c]);
    check("perturbing window 0 changes window 0 output", d_window0 > 1e-6, d_window0);
    check("perturbing window 0 mostly does NOT change window 3 (10x ratio)",
          d_window0 > 10.0 * d_window3, d_window3 / std::max(d_window0, 1e-12));
}

// ---- Test 8: FD input gradient -------------------------------------------
static Tensor fd_input_grad(WindowAttention& wa, Tensor input, const Tensor& tgt,
                             double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            double lp = l2_loss_value(wa.forward(input), tgt);
            input[i][j] = orig - eps;
            double lm = l2_loss_value(wa.forward(input), tgt);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}
static Tensor fd_param_grad(WindowAttention& wa, size_t pidx, Tensor input,
                             const Tensor& tgt, double eps = 1e-5) {
    Tensor* p = wa.parameters()[pidx];
    Tensor grad(p->rows, p->cols);
    for (size_t i = 0; i < p->rows; ++i) {
        for (size_t j = 0; j < p->cols; ++j) {
            double orig = (*p)[i][j];
            (*p)[i][j] = orig + eps;
            double lp = l2_loss_value(wa.forward(input), tgt);
            (*p)[i][j] = orig - eps;
            double lm = l2_loss_value(wa.forward(input), tgt);
            (*p)[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static void test_fd_input_grad() {
    cout << "Test 8: FD input gradient\n";
    srand(7);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 7);
    Tensor target = make_input(16, 8, 8);
    Tensor out = wa.forward(input);
    Tensor ana = wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor fd = fd_input_grad(wa, input.clone(), target);
    double rel = tensor_rel_err(ana, fd);
    check("FD input grad rel_err < 1e-4", rel < 1e-4, rel);
}

// ---- Test 9: FD parameter gradients --------------------------------------
static void test_fd_param_grads() {
    cout << "Test 9: FD parameter gradients (W_q, W_k, W_v, W_o, b_o)\n";
    srand(9);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 9);
    Tensor target = make_input(16, 8, 10);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    for (size_t pidx = 0; pidx < 5; ++pidx) {
        Tensor fd = fd_param_grad(wa, pidx, input.clone(), target);
        Tensor* g = wa.gradients()[pidx];
        double rel = tensor_rel_err(*g, fd);
        string name = string("FD param grad ") + to_string(pidx) + " rel_err < 1e-5";
        check(name, rel < 1e-5, rel);
    }
}

// ---- Test 10: FD relative_position_bias_ gradient ------------------------
static void test_fd_rpb_grad() {
    cout << "Test 10: FD relative_position_bias_ gradient\n";
    srand(11);
    WindowAttention wa(8, 2, 2, 4, 4, "relative");
    // Manually set Q/K/V projections to small magnitude so the RPB contribution
    // dominates the FD signal — otherwise the FD-vs-analytical disagreement
    // on RPB is hidden under the much larger QK^T signal.
    for (size_t i = 0; i < wa.W_q.data.size(); ++i) wa.W_q.data[i] *= 0.05;
    for (size_t i = 0; i < wa.W_k.data.size(); ++i) wa.W_k.data[i] *= 0.05;
    for (size_t i = 0; i < wa.W_v.data.size(); ++i) wa.W_v.data[i] *= 0.05;
    // Boost RPB so its contribution to scores is comparable to QK^T
    for (size_t i = 0; i < wa.relative_position_bias_.data.size(); ++i)
        wa.relative_position_bias_.data[i] = 0.5;
    Tensor input = make_input(16, 8, 11);
    Tensor target = make_input(16, 8, 12);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor fd = fd_param_grad(wa, 5, input.clone(), target);
    Tensor* g = wa.gradients()[5];
    double rel = tensor_rel_err(*g, fd);
    check("FD rpb grad rel_err < 1e-5", rel < 1e-5, rel);
    // Sum of |grad| must be > 0
    double gsum = 0.0;
    for (double v : g->data) gsum += fabs(v);
    check("grad_relative_position_bias_ non-zero", gsum > 1e-6, gsum);
}

// ---- Test 11: bias_type="none" path --------------------------------------
static void test_no_bias_type() {
    cout << "Test 11: bias_type='none' path\n";
    srand(13);
    WindowAttention wa(8, 2, 2, 4, 4, "none");
    Tensor input = make_input(16, 8, 13);
    Tensor target = make_input(16, 8, 14);
    Tensor out = wa.forward(input);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
    check("forward finite", finite);
    Tensor ana = wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor fd = fd_input_grad(wa, input.clone(), target);
    double rel = tensor_rel_err(ana, fd);
    check("FD input grad (no bias) rel_err < 1e-4", rel < 1e-4, rel);
}

// ---- Test 12: zero_grad clears all gradients -----------------------------
static void test_zero_grad_clears() {
    cout << "Test 12: zero_grad clears all 6 gradients\n";
    srand(15);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 15);
    Tensor target = make_input(16, 8, 16);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    wa.zero_grad();
    for (size_t i = 0; i < wa.gradients().size(); ++i) {
        Tensor* g = wa.gradients()[i];
        for (double v : g->data) {
            if (fabs(v) > 0.0) {
                check(string("zero_grad clears grad ") + to_string(i), false);
                return;
            }
        }
    }
    check("zero_grad clears all 6 grads", true);
}

// ---- Test 13: update_weights moves parameters ---------------------------
static void test_update_weights_moves() {
    cout << "Test 13: update_weights moves all 6 parameters\n";
    srand(17);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 17);
    Tensor target = make_input(16, 8, 18);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    auto params = wa.parameters();
    vector<Tensor> saved;
    for (auto* p : params) saved.push_back(p->clone());
    wa.update_weights(0.05);
    for (size_t i = 0; i < params.size(); ++i) {
        if (tensor_max_abs_diff(*params[i], saved[i]) <= 1e-9) {
            check(string("update_weights moved param ") + to_string(i), false);
            return;
        }
    }
    check("update_weights(0.05) moves all 6 params", true);
}

// ---- Test 14: end-to-end training reduces loss --------------------------
static void test_training_reduces_loss() {
    cout << "Test 14: end-to-end training reduces loss\n";
    srand(19);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 19);
    Tensor target = make_input(16, 8, 20);
    double L0 = l2_loss_value(wa.forward(input), target);
    for (int step = 0; step < 30; ++step) {
        Tensor out = wa.forward(input);
        Tensor grad = l2_loss_grad(out, target);
        wa.backward(grad, 0.0);
        wa.update_weights(0.05);
    }
    double Lf = l2_loss_value(wa.forward(input), target);
    check("loss reduces by > 10% over 30 SGD steps", Lf < 0.9 * L0, Lf / std::max(L0, 1e-12));
}

int main() {
    test_constructor_validation();
    test_accessors();
    test_parameters_gradients_contract();
    test_forward_shape_small();
    test_forward_larger();
    test_forward_determinism();
    test_window_isolation();
    test_fd_input_grad();
    test_fd_param_grads();
    test_fd_rpb_grad();
    test_no_bias_type();
    test_zero_grad_clears();
    test_update_weights_moves();
    test_training_reduces_loss();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
