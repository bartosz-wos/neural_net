// test_based.cpp — Gradient correctness tests for Based Linear Attention
//   (Arora et al. 2024, https://arxiv.org/abs/2402.18668)
//
// 2nd-order Taylor-approximated linear attention with causal mask.
//
// Tests:
//   1.  Constructor validation (d_model=0, seq_len=0, feature_dim=0 throw)
//   2.  Forward shape (n, d_model) -> (n, d_model)
//   3.  Forward output is finite
//   4.  Taylor feature map identity: phi(x)^T phi(y) == 1 + (x·y)/√d + (x·y)²/(2d)
//   5.  Causal mask: A[i, j] = 0 for j > i
//   6.  Hand-computed forward (n=1, d=2, fd=1)
//   7.  Input gradient FD check (n=3, d=3, fd=2)
//   8.  W_q gradient FD check
//   9.  W_k gradient FD check
//  10.  W_v gradient FD check
//  11.  W_o gradient FD check
//  12.  zero_grad clears all 4 gradients
//  13.  update_weights moves all 4 parameters
//  14.  BasedBlock forward shape (n, d) -> (n, d)
//  15.  BasedBlock input gradient FD check
//  16.  BasedModel training step reduces loss
//  17.  Longer sequence (T=6) input gradient FD check
//  18.  parameters()/gradients() contract (4 tensors, shapes matched)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/layers/attention/based.h"

using namespace std;

static int passed = 0, failed = 0;
static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}
static double rel_err(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
}
static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i)
        g.data[i] = output.data[i] - target.data[i];
    return g;
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor() {
    cout << endl << "--- Test 1: constructor validation ---" << endl;
    bool threw_d = false, threw_n = false, threw_f = false, valid_ok = true;
    try { BasedAttention bad(0, 4, 2); } catch (const std::exception&) { threw_d = true; }
    try { BasedAttention bad(4, 0, 2); } catch (const std::exception&) { threw_n = true; }
    try { BasedAttention bad(4, 4, 0); } catch (const std::exception&) { threw_f = true; }
    try { BasedAttention good(4, 4, 2); } catch (const std::exception&) { valid_ok = false; }
    cout << "  d_model=0 threw: " << threw_d
         << "  seq_len=0 threw: " << threw_n
         << "  feature_dim=0 threw: " << threw_f
         << "  valid ok: " << valid_ok << endl;
    check("constructor validates correctly", threw_d && threw_n && threw_f && valid_ok);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape
// ---------------------------------------------------------------------------
static void test_forward_shape() {
    cout << endl << "--- Test 2: forward shape (n=4, d=4, fd=3) ---" << endl;
    size_t n = 4, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    BasedAttention attn(d, n, 3);
    Tensor out = attn.forward(input);
    cout << "  Input: " << input.rows << "x" << input.cols
         << "  Output: " << out.rows << "x" << out.cols << endl;
    check("forward shape (n, d) -> (n, d)", out.rows == n && out.cols == d);
}

// ---------------------------------------------------------------------------
// Test 3: forward output is finite
// ---------------------------------------------------------------------------
static void test_forward_finite() {
    cout << endl << "--- Test 3: forward output is finite (n=8, d=4, fd=2) ---" << endl;
    size_t n = 8, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.3 * sin(0.1 * i) + 0.05 * j;
    BasedAttention attn(d, n, 2);
    Tensor out = attn.forward(input);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    check("output is finite", finite);
}

// ---------------------------------------------------------------------------
// Test 4: Taylor feature map identity
//   phi(x)^T phi(y) = 1 + (x·y)/sqrt(d) + (x·y)^2 / (2d)
//   We check this directly by computing phi(x) and phi(y) and taking the dot.
// ---------------------------------------------------------------------------
static void test_taylor_identity() {
    cout << endl << "--- Test 4: Taylor feature map identity phi(x)^T phi(y) = 1 + s/√d + s²/(2d) ---" << endl;
    size_t d = 4;
    Tensor x(1, d), y(1, d);
    for (size_t j = 0; j < d; ++j) { x(0, j) = 0.1 * j - 0.3; y(0, j) = -0.2 + 0.07 * j; }
    // Compute phi(x), phi(y) by hand for the reference.
    const double rd = std::sqrt((double)d);
    const double r2 = std::sqrt(2.0);
    const double rrd = std::sqrt(rd);
    const double inv_r2_rd = 1.0 / (r2 * rd);
    const size_t phi_dim = 1 + d + d * d;
    Tensor phi_x(1, phi_dim), phi_y(1, phi_dim);
    phi_x(0, 0) = 1.0; phi_y(0, 0) = 1.0;
    for (size_t k = 0; k < d; ++k) {
        phi_x(0, 1 + k) = x(0, k) / rrd;
        phi_y(0, 1 + k) = y(0, k) / rrd;
    }
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            phi_x(0, 1 + d + i * d + j) = x(0, i) * x(0, j) * inv_r2_rd;
            phi_y(0, 1 + d + i * d + j) = y(0, i) * y(0, j) * inv_r2_rd;
        }
    }
    // Hand-compute the dot product.
    double dot = 0.0;
    for (size_t k = 0; k < phi_dim; ++k) dot += phi_x(0, k) * phi_y(0, k);
    // Reference: 1 + s/√d + s²/(2d) where s = x·y
    double s = 0.0;
    for (size_t j = 0; j < d; ++j) s += x(0, j) * y(0, j);
    double ref = 1.0 + s / rd + s * s / (2.0 * d);
    double err = rel_err(dot, ref);
    cout << "  phi(x)^T phi(y) = " << dot << "  ref = " << ref
         << "  rel_err = " << err << endl;
    check("Taylor feature map identity", err < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 5: causal mask: A[i, j] = 0 for j > i
// ---------------------------------------------------------------------------
static void test_causal_mask() {
    cout << endl << "--- Test 5: causal mask A[i, j] = 0 for j > i ---" << endl;
    size_t n = 5, d = 3;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    BasedAttention attn(d, n, 2);
    attn.forward(input);
    const Tensor& A = attn.last_A();
    bool causal = true;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (A(i, j) != 0.0) causal = false;
        }
    }
    cout << "  A shape: " << A.rows << "x" << A.cols << "  causal ok: " << causal << endl;
    check("causal mask enforced", causal);
}

// ---------------------------------------------------------------------------
// Test 6: hand-computed forward (n=1, fd=1, all-ones input)
//   At n=1, the linear-attention output is:
//     A[0,0] = phi_q[0]·phi_k[0] = 1 + q0²/√1 + q0²/2 = 1 + q0² + q0²/2 (for d=1)
//     y[0] = A[0,0] * V[0] = A * V
//     z[0] = phi_q[0] · phi_k[0] + eps = A + eps
//     out[0] = y[0] / z[0] = V[0] (since A and z both contain the same phi_q·phi_k)
//     Then out_pre is multiplied by W_o.
// ---------------------------------------------------------------------------
static void test_hand_forward() {
    cout << endl << "--- Test 6: hand-computed forward (n=1, d=2, fd=1) ---" << endl;
    BasedAttention attn(2, 1, 1);
    // Zero out all weights and biases, then put specific values.
    attn.W_q.fill(0.0); attn.W_k.fill(0.0);
    attn.W_v.fill(0.0); attn.W_o.fill(0.0);
    // W_q is (d_model=2, feature_dim=1) with column 0 = [1, 0]
    //   → q_pre[0] = x[0,0]*1 + x[0,1]*0 = 0.5
    // W_k is (2, 1) with column 0 = [1, 0]
    //   → k_pre[0] = 0.5 (so Q == K, A[0,0] = phi_q[0]·phi_k[0])
    // W_v is (2, 2); v[i, j] = sum_d x[i, d] * W_v[d, j]
    //   → v[0, 0] = 0.5*1 + 0.3*1 = 0.8 ; v[0, 1] = 0
    // W_o is (2, 2); out[i, j'] = sum_j out_pre[i, j] * W_o[j, j']
    //   → out[0, 0] = out_pre[0, 0] * 1 ; out[0, 1] = 0
    attn.W_q(0, 0) = 1.0;
    attn.W_q(1, 0) = 0.0;
    attn.W_k(0, 0) = 1.0;
    attn.W_k(1, 0) = 0.0;
    attn.W_v(0, 0) = 1.0; attn.W_v(0, 1) = 0.0;
    attn.W_v(1, 0) = 1.0; attn.W_v(1, 1) = 0.0;
    attn.W_o(0, 0) = 1.0; attn.W_o(0, 1) = 0.0;
    attn.W_o(1, 0) = 0.0; attn.W_o(1, 1) = 0.0;

    Tensor input(1, 2);
    input(0, 0) = 0.5;
    input(0, 1) = 0.3;
    Tensor out = attn.forward(input);
    // At n=1: A[0,0] = phi_q[0]·phi_k[0] (since tril is just [A[0,0]])
    //         z[0] = phi_q[0]·phi_k[0] + eps  (same)
    //         out_pre[0, 0] = A[0,0] * V[0, 0] / z[0] = V[0, 0] = 0.8
    //         out[0, 0] = out_pre[0, 0] * W_o[0, 0] + out_pre[0, 1] * W_o[1, 0] = 0.8
    //         out[0, 1] = 0 (W_o[*, 1] are all 0)
    cout << "  out: [" << out(0, 0) << ", " << out(0, 1) << "]" << endl;
    double err0 = rel_err(out(0, 0), 0.8);
    double err1 = rel_err(out(0, 1), 0.0);
    cout << "  rel_err[0]: " << err0 << "  rel_err[1]: " << err1 << endl;
    check("hand-computed forward (n=1)", err0 < 1e-9 && err1 < 1e-9);
}

// ---------------------------------------------------------------------------
// Test 7: input gradient FD check
// ---------------------------------------------------------------------------
static void test_input_grad() {
    cout << endl << "--- Test 7: input gradient FD check (n=3, d=3, fd=2) ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    BasedAttention attn(d, n, fd);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    Tensor grad_x = attn.backward(grad_loss, 0.0);

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            input(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = grad_x(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            if (err > 0.1) {
                cout << "  x[" << i << "][" << j << "]: ana=" << ana
                     << " num=" << num << " err=" << err << endl;
            }
        }
    }
    cout << "  max rel_err: " << max_err << endl;
    check("input gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 8: W_q gradient FD check
// ---------------------------------------------------------------------------
static void test_wq_grad() {
    cout << endl << "--- Test 8: W_q gradient FD check ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    BasedAttention attn(d, n, fd);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_q;
    Tensor& G = attn.grad_W_q;
    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < W.rows && n_check < 6; ++i) {
        for (size_t j = 0; j < W.cols && n_check < 6; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            W(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            W(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = G(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            ++n_check;
        }
    }
    cout << "  W_q max rel_err: " << max_err << endl;
    check("W_q gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 9: W_k gradient FD check
// ---------------------------------------------------------------------------
static void test_wk_grad() {
    cout << endl << "--- Test 9: W_k gradient FD check ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    BasedAttention attn(d, n, fd);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_k;
    Tensor& G = attn.grad_W_k;
    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < W.rows && n_check < 6; ++i) {
        for (size_t j = 0; j < W.cols && n_check < 6; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            W(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            W(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = G(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            ++n_check;
        }
    }
    cout << "  W_k max rel_err: " << max_err << endl;
    check("W_k gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 10: W_v gradient FD check
// ---------------------------------------------------------------------------
static void test_wv_grad() {
    cout << endl << "--- Test 10: W_v gradient FD check ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    BasedAttention attn(d, n, fd);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_v;
    Tensor& G = attn.grad_W_v;
    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < W.rows && n_check < 6; ++i) {
        for (size_t j = 0; j < W.cols && n_check < 6; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            W(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            W(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = G(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            ++n_check;
        }
    }
    cout << "  W_v max rel_err: " << max_err << endl;
    check("W_v gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 11: W_o gradient FD check
// ---------------------------------------------------------------------------
static void test_wo_grad() {
    cout << endl << "--- Test 11: W_o gradient FD check ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    BasedAttention attn(d, n, fd);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_o;
    Tensor& G = attn.grad_W_o;
    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < W.rows && n_check < 6; ++i) {
        for (size_t j = 0; j < W.cols && n_check < 6; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            W(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            W(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = G(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            ++n_check;
        }
    }
    cout << "  W_o max rel_err: " << max_err << endl;
    check("W_o gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 12: zero_grad clears all 4 gradients
// ---------------------------------------------------------------------------
static void test_zero_grad() {
    cout << endl << "--- Test 12: zero_grad clears all 4 gradients ---" << endl;
    BasedAttention attn(4, 3, 2);
    Tensor input(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    Tensor target(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    Tensor out = attn.forward(input);
    attn.backward(l2_loss_grad(out, target), 0.0);
    double before = 0.0;
    for (Tensor* g : attn.gradients())
        for (double v : g->data) before += fabs(v);
    attn.zero_grad();
    double after = 0.0;
    for (Tensor* g : attn.gradients())
        for (double v : g->data) after += fabs(v);
    cout << "  |g| before: " << before << "  after: " << after << endl;
    check("zero_grad clears all 4 gradients", before > 1e-8 && after == 0.0);
}

// ---------------------------------------------------------------------------
// Test 13: update_weights moves all 4 parameters
// ---------------------------------------------------------------------------
static void test_update_weights() {
    cout << endl << "--- Test 13: update_weights moves all 4 parameters ---" << endl;
    BasedAttention attn(4, 3, 2);
    Tensor input(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    Tensor target(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    attn.zero_grad();
    Tensor out = attn.forward(input);
    attn.backward(l2_loss_grad(out, target), 0.0);
    vector<Tensor> snap;
    for (Tensor* p : attn.parameters()) snap.push_back(*p);
    attn.update_weights(0.5);
    size_t moved = 0;
    auto params = attn.parameters();
    for (size_t k = 0; k < params.size(); ++k) {
        double d = 0.0;
        for (size_t m = 0; m < params[k]->data.size(); ++m)
            d += fabs(params[k]->data[m] - snap[k].data[m]);
        if (d > 1e-12) ++moved;
    }
    cout << "  moved: " << moved << " / " << params.size() << endl;
    check("update_weights moves all 4 parameters", moved == params.size());
}

// ---------------------------------------------------------------------------
// Test 14: BasedBlock forward shape
// ---------------------------------------------------------------------------
static void test_block_forward() {
    cout << endl << "--- Test 14: BasedBlock forward shape (n=4, d=4, fd=2) ---" << endl;
    size_t n = 4, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    BasedBlock block(d, n, 2);
    Tensor out = block.forward(input);
    cout << "  in: " << input.rows << "x" << input.cols
         << "  out: " << out.rows << "x" << out.cols << endl;
    check("block shape (n, d) -> (n, d)", out.rows == n && out.cols == d);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    check("block output finite", finite);
}

// ---------------------------------------------------------------------------
// Test 15: BasedBlock input gradient FD check
// ---------------------------------------------------------------------------
static void test_block_input_grad() {
    cout << endl << "--- Test 15: BasedBlock input gradient FD check (n=3, d=3, fd=2) ---" << endl;
    size_t n = 3, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    BasedBlock block(d, n, fd);
    Tensor out = block.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    block.zero_grad();
    Tensor grad_x = block.backward(grad_loss, 0.0);
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = block.forward(input);
            double Lp = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = block.forward(input);
            double Lm = l2_loss_value(out_m, target);
            input(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = grad_x(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
        }
    }
    cout << "  max rel_err: " << max_err << endl;
    check("block input gradient FD (rel_err < 1e-2)", max_err < 1e-2);
}

// ---------------------------------------------------------------------------
// Test 17: longer sequence (T=6) input gradient FD check
// ---------------------------------------------------------------------------
static void test_long_input_grad() {
    cout << endl << "--- Test 17: longer sequence (T=6) input gradient FD check ---" << endl;
    size_t n = 6, d = 3, fd = 2;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 + 0.04 * i + 0.03 * j;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.4 + 0.05 * j;

    BasedAttention attn(d, n, fd);
    attn.zero_grad();
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    Tensor grad_x = attn.backward(grad_loss, 0.0);

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            double Lp = l2_loss_value(attn.forward(input), target);
            input(i, j) = orig - eps;
            double Lm = l2_loss_value(attn.forward(input), target);
            input(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = grad_x(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
        }
    }
    cout << "  max rel_err: " << max_err << endl;
    check("long-sequence input gradient FD (rel_err < 1e-2)", max_err < 1e-2);
}

// ---------------------------------------------------------------------------
// Test 18: parameters/gradients contract
// ---------------------------------------------------------------------------
static void test_param_grad_contract() {
    cout << endl << "--- Test 18: parameters()/gradients() contract ---" << endl;
    BasedAttention attn(4, 3, 2);
    auto ps = attn.parameters();
    auto gs = attn.gradients();
    bool ok = (ps.size() == 4 && gs.size() == 4);
    if (ok)
        for (size_t k = 0; k < ps.size(); ++k)
            if (ps[k]->rows != gs[k]->rows || ps[k]->cols != gs[k]->cols) ok = false;
    bool shapes_ok = ok &&
        attn.W_q.rows == 4 && attn.W_q.cols == 2 &&
        attn.W_k.rows == 4 && attn.W_k.cols == 2 &&
        attn.W_v.rows == 4 && attn.W_v.cols == 4 &&
        attn.W_o.rows == 4 && attn.W_o.cols == 4;
    cout << "  params: " << ps.size() << "  grads: " << gs.size()
         << "  shapes_ok: " << shapes_ok << "  name: " << attn.name() << endl;
    check("parameters/gradients contract", ok && shapes_ok);
}

// ---------------------------------------------------------------------------
// Test 16: BasedModel training step reduces loss
// ---------------------------------------------------------------------------
static void test_model_train() {
    cout << endl << "--- Test 16: BasedModel training step reduces loss ---" << endl;
    size_t n = 4, in_dim = 4, d = 4, of = 3, num_blocks = 2;
    BasedModel model(in_dim, d, n, of, num_blocks, 2, 0);
    Tensor input(n, in_dim);
    Tensor target(n, of);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < in_dim; ++j) input(i, j) = 0.1 * i + 0.05 * j;
        for (size_t j = 0; j < of; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    }
    double L0 = l2_loss_value(model.forward(input), target);
    double Lf = L0;
    double lr = 0.01;
    for (int step = 0; step < 30; ++step) {
        model.zero_grad();
        Tensor out = model.forward(input);
        Lf = l2_loss_value(out, target);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
    }
    model.zero_grad();
    Lf = l2_loss_value(model.forward(input), target);
    cout << "  L0: " << L0 << "  Lf: " << Lf << endl;
    check("BasedModel training reduces loss (Lf < L0 * 0.9)", Lf < L0 * 0.9);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cout << "=== Based Linear Attention Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_constructor();
    test_forward_shape();
    test_forward_finite();
    test_taylor_identity();
    test_causal_mask();
    test_hand_forward();
    test_input_grad();
    test_wq_grad();
    test_wk_grad();
    test_wv_grad();
    test_wo_grad();
    test_zero_grad();
    test_update_weights();
    test_block_forward();
    test_block_input_grad();
    test_model_train();
    test_long_input_grad();
    test_param_grad_contract();

    cout << endl << "=== Summary: " << passed << "/" << (passed + failed)
         << " tests passed ===" << endl;
    return failed == 0 ? 0 : 1;
}
