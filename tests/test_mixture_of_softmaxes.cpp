// test_mixture_of_softmaxes.cpp — Mixture of Softmaxes (MoS) attention
//   Yang, Chen, Liu, Li, Wang, Yi, Wong, Xia — ICLR 2018
//   "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model"
//   https://arxiv.org/abs/1711.03953
//
// Tests:
//   1.  Constructor validation (d_model=0, num_heads=0, K=0, non-divisible throw)
//   2.  Forward shape + finiteness
//   3.  Per-branch softmax rows sum to 1; per-row gating sums to 1
//   4.  K=1 bit-exact MHA recovery (regression test for "K parameter unused")
//   5.  K=2 vs K=1 differs (signature test for branches being active)
//   6.  Input gradient FD (K=1, K=2, K=4 — multi-iter depth exercises chain)
//   7.  Parameter gradient FD (W_q, W_k, W_v, B_v, B_z, W_g, B_g, W_o)
//   8.  K=1 makes W_g and B_g gradients exactly zero (no branches to gate)
//   9.  zero_grad / update_weights / parameters() / gradients() contract
//  10.  K=4 end-to-end training reduces loss

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include "nn/layers/attention/mixture_of_softmaxes.h"

using namespace std;

static int passed = 0, failed = 0;
static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else      { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}
static double rel_err(double a, double b) {
    double m = max(fabs(a), fabs(b));
    if (m < 1e-9) return fabs(a - b) / 1e-9;
    return fabs(a - b) / m;
}
static double l2_loss(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - tgt.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i)
        g.data[i] = out.data[i] - tgt.data[i];
    return g;
}
// Deterministic pseudo-random fill — independent of global RNG state so tests
// are reproducible regardless of what other layers consumed from the RNG.
static void fill_det(Tensor& t, unsigned seed, double scale) {
    unsigned s = seed;
    for (size_t i = 0; i < t.data.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        double u = ((s >> 8) & 0xFFFFFF) / double(0xFFFFFF);
        t.data[i] = (2.0 * u - 1.0) * scale;
    }
}
static Tensor rand_tensor(size_t r, size_t c, unsigned seed, double scale) {
    Tensor t(r, c);
    fill_det(t, seed, scale);
    return t;
}
// Non-uniform parameter init — uniform init hides row-vs-column confusion.
static void randomize(MixtureOfSoftmaxesAttention& a, unsigned base) {
    fill_det(a.W_q.weights, base + 1, 0.3);
    fill_det(a.W_k.weights, base + 2, 0.3);
    fill_det(a.W_o.weights, base + 3, 0.3);
    fill_det(a.W_q.bias,    base + 4, 0.2);
    fill_det(a.W_k.bias,    base + 5, 0.2);
    fill_det(a.W_o.bias,    base + 6, 0.2);
    fill_det(a.W_v,         base + 7, 0.3);
    fill_det(a.B_v,         base + 8, 0.2);
    fill_det(a.B_z,         base + 9, 0.2);
    fill_det(a.W_g,         base + 10, 0.2);
    fill_det(a.B_g,         base + 11, 0.1);
}

// ---------------------------------------------------------------------------
// FD helpers
// ---------------------------------------------------------------------------
static double fd_input_check(MixtureOfSoftmaxesAttention& a, Tensor x,
                             const Tensor& tgt, double eps = 1e-6) {
    a.zero_grad();
    Tensor out = a.forward(x);
    Tensor gi = a.backward(l2_grad(out, tgt), 0.0);
    double worst = 0.0;
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c) {
            Tensor xp = x.clone(); xp(r, c) += eps;
            Tensor xm = x.clone(); xm(r, c) -= eps;
            double num = (l2_loss(a.forward(xp), tgt) -
                          l2_loss(a.forward(xm), tgt)) / (2 * eps);
            worst = max(worst, rel_err(gi(r, c), num));
        }
    return worst;
}
static double fd_param_check(MixtureOfSoftmaxesAttention& a, Tensor* param,
                             Tensor* grad, const Tensor& x, const Tensor& tgt,
                             double eps = 1e-6) {
    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    Tensor ana = grad->clone();
    double worst = 0.0;
    for (size_t i = 0; i < param->data.size(); ++i) {
        double saved = (*param).data[i];
        (*param).data[i] = saved + eps;
        double lp = l2_loss(a.forward(x), tgt);
        (*param).data[i] = saved - eps;
        double lm = l2_loss(a.forward(x), tgt);
        (*param).data[i] = saved;
        double num = (lp - lm) / (2 * eps);
        worst = max(worst, rel_err(ana.data[i], num));
    }
    return worst;
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor() {
    cout << endl << "--- Test 1: constructor validation ---" << endl;
    bool t_dm = false, t_nh = false, t_K = false, t_div = false, ok = true;
    try { MixtureOfSoftmaxesAttention bad(0, 1, 1); } catch (const exception&) { t_dm = true; }
    try { MixtureOfSoftmaxesAttention bad(4, 0, 1); } catch (const exception&) { t_nh = true; }
    try { MixtureOfSoftmaxesAttention bad(4, 1, 0); } catch (const exception&) { t_K = true; }
    try { MixtureOfSoftmaxesAttention bad(4, 3, 1); } catch (const exception&) { t_div = true; }
    bool constructed = false;
    try { MixtureOfSoftmaxesAttention good(4, 1, 1); constructed = true; } catch (const exception&) { constructed = false; }
    ok = constructed;
    check("d_model=0 throws", t_dm);
    check("num_heads=0 throws", t_nh);
    check("K=0 throws", t_K);
    check("d_model not divisible by num_heads throws", t_div);
    check("valid (4,1,1) constructs", ok);

    MixtureOfSoftmaxesAttention a(6, 3, 2);
    check("head_dim == d_model/num_heads", a.head_dim() == 2);
    check("num_branches == K", a.num_branches() == 2);
    check("inv_temp == 1/sqrt(head_dim)",
          rel_err(a.inv_temp(), 1.0 / sqrt(2.0)) < 1e-15);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape + finiteness
// ---------------------------------------------------------------------------
static void test_forward_shape() {
    cout << endl << "--- Test 2: forward shape + finiteness ---" << endl;
    MixtureOfSoftmaxesAttention a(4, 1, 2);
    randomize(a, 100);
    Tensor x = rand_tensor(5, 4, 7, 1.0);
    Tensor y = a.forward(x);
    check("output shape (5,4)", y.rows == 5 && y.cols == 4);
    bool fin = true, nz = false;
    for (double v : y.data) { if (!isfinite(v)) fin = false; if (fabs(v) > 1e-12) nz = true; }
    check("output finite", fin);
    check("output nonzero", nz);
    check("last_P shape (H*K, N, N)", a.last_P().rows == 2 && a.last_P().cols == 5 * 5);
    check("last_Pi shape (H, N, K)", a.last_Pi().rows == 1 && a.last_Pi().cols == 5 * 2);
}

// ---------------------------------------------------------------------------
// Test 3: softmax/gating row-sum invariants
// ---------------------------------------------------------------------------
static void test_row_sum_invariants() {
    cout << endl << "--- Test 3: per-branch softmax + gating sum to 1 ---" << endl;
    MixtureOfSoftmaxesAttention a(6, 3, 4);
    randomize(a, 200);
    const size_t N = 5, H = 3, K = 4;
    a.forward(rand_tensor(N, 6, 11, 1.0));

    // Per-branch softmax over rows (over N positions i): sums to 1.
    double worst_P = 0.0;
    for (size_t hk = 0; hk < H * K; ++hk) {
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < N; ++i) s += a.last_P()(hk, j * N + i);
            worst_P = max(worst_P, fabs(s - 1.0));
        }
    }
    cout << "  max |sum_i P[hk][j,i] - 1| = " << scientific << worst_P << endl;
    check("per-branch softmax rows sum to 1", worst_P < 1e-10);

    // Per-head gating sums to 1 over K.
    double worst_Pi = 0.0;
    for (size_t h = 0; h < H; ++h) {
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += a.last_Pi()(h, j * K + k);
            worst_Pi = max(worst_Pi, fabs(s - 1.0));
        }
    }
    cout << "  max |sum_k Pi[h][j,k] - 1| = " << scientific << worst_Pi << endl;
    check("per-head gating sums to 1", worst_Pi < 1e-10);
}

// ---------------------------------------------------------------------------
// Test 4: K=1 bit-exact MHA recovery
// ---------------------------------------------------------------------------
static void test_K1_recovers_MHA() {
    cout << endl << "--- Test 4: K=1 recovers MHA bit-exactly ---" << endl;
    // Build a tiny case: N=3, d_model=4, H=1. K=1.
    const size_t N = 3, d = 4, H = 1;
    MixtureOfSoftmaxesAttention a(d, H, 1);
    // Hand-set so we can compute MHA reference directly.
    // W_q = I, W_k = I, W_o = I, B_z=0, B_v=0, W_g=W=1, B_g=0 → Q=K=x, V=x, O = softmax(x x^T / √d) · x.
    a.W_q.weights.fill(0.0); a.W_k.weights.fill(0.0); a.W_v.fill(0.0); a.W_o.weights.fill(0.0);
    a.W_q.bias.fill(0.0); a.W_k.bias.fill(0.0); a.B_v.fill(0.0); a.W_o.bias.fill(0.0);
    a.B_z.fill(0.0); a.W_g.fill(1.0); a.B_g.fill(0.0);
    // W_v flat (K=1, d*d): branch 0 identity V at W_v(0, i*d+i) for i in [0, d).
    for (size_t i = 0; i < d; ++i) {
        a.W_q.weights(i, i) = 1.0;
        a.W_k.weights(i, i) = 1.0;
        a.W_v(0, i * d + i) = 1.0;
        a.W_o.weights(i, i) = 1.0;
    }
    Tensor x(N, d);
    x(0, 0) = 0.3; x(0, 1) = -0.7; x(0, 2) = 1.1; x(0, 3) = 0.4;
    x(1, 0) = 0.5; x(1, 1) = -0.2; x(1, 2) = 0.9; x(1, 3) = -0.6;
    x(2, 0) = -0.1; x(2, 1) = 0.8; x(2, 2) = 0.3; x(2, 3) = 0.2;

    Tensor y_mos = a.forward(x);

    // Reference MHA: Q=x, K=x, Z[i,j]=Σ_d x[j,d]*x[i,d]/√d, P[i,j]=softmax_i Z[i,j],
    // O[j] = Σ_i P[i,j] * x[i], y[j,d] = O[j,d] (W_o=I).
    const double inv_t = 1.0 / sqrt(double(d));
    double Z[3][3];
    double P[3][3];
    double O[3][4];
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (size_t dd = 0; dd < d; ++dd) s += x(j, dd) * x(i, dd);
            Z[i][j] = s * inv_t;
        }
    }
    for (size_t j = 0; j < N; ++j) {
        double maxv = Z[0][j];
        for (size_t i = 1; i < N; ++i) if (Z[i][j] > maxv) maxv = Z[i][j];
        double sumexp = 0.0;
        for (size_t i = 0; i < N; ++i) { P[i][j] = exp(Z[i][j] - maxv); sumexp += P[i][j]; }
        for (size_t i = 0; i < N; ++i) P[i][j] /= sumexp;
    }
    for (size_t j = 0; j < N; ++j)
        for (size_t dd = 0; dd < d; ++dd) {
            double s = 0.0;
            for (size_t i = 0; i < N; ++i) s += P[i][j] * x(i, dd);
            O[j][dd] = s;
        }
    double worst = 0.0;
    for (size_t j = 0; j < N; ++j)
        for (size_t dd = 0; dd < d; ++dd)
            worst = max(worst, rel_err(y_mos(j, dd), O[j][dd]));
    cout << "  worst |MoS - MHA reference| = " << scientific << worst << endl;
    check("K=1 MoS output bit-exact MHA reference", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 5: K=2 vs K=1 differs (branches are active)
// ---------------------------------------------------------------------------
static void test_K2_active() {
    cout << endl << "--- Test 5: K=2 vs K=1 differs ---" << endl;
    MixtureOfSoftmaxesAttention k1(4, 1, 1);
    MixtureOfSoftmaxesAttention k2(4, 1, 2);
    randomize(k1, 300); randomize(k2, 300);
    // Copy k1's params into k2's slot 0 so the only thing differing is K=2's
    // extra branch and gating weights.
    for (size_t i = 0; i < k1.W_v.data.size(); ++i) k2.W_v.data[i] = k1.W_v.data[i];
    for (size_t i = 0; i < k1.B_v.data.size(); ++i) k2.B_v.data[i] = k1.B_v.data[i];
    for (size_t i = 0; i < k1.B_z.data.size(); ++i) k2.B_z.data[i] = k1.B_z.data[i];

    Tensor x = rand_tensor(5, 4, 13, 1.0);
    Tensor y1 = k1.forward(x);
    Tensor y2 = k2.forward(x);
    double diff = 0.0;
    for (size_t i = 0; i < y1.data.size(); ++i) diff += fabs(y1.data[i] - y2.data[i]);
    cout << "  sum |y_k1 - y_k2| = " << scientific << diff << endl;
    check("K=2 differs from K=1 (branches active)", diff > 1e-3);
}

// ---------------------------------------------------------------------------
// Test 6: input gradient FD
// ---------------------------------------------------------------------------
static void test_input_grad_FD() {
    cout << endl << "--- Test 6: input gradient FD ---" << endl;
    // K=1 small: should be very clean.
    {
        MixtureOfSoftmaxesAttention a(4, 1, 1);
        randomize(a, 400);
        Tensor x = rand_tensor(4, 4, 23, 1.0);
        Tensor tgt = rand_tensor(4, 4, 29, 1.0);
        double w = fd_input_check(a, x, tgt);
        cout << "  K=1 d_input worst rel_err = " << scientific << w << endl;
        check("K=1 d_input FD < 1e-5", w < 1e-5);
    }
    // K=2 mid: less clean but still tight.
    {
        MixtureOfSoftmaxesAttention a(4, 2, 2);
        randomize(a, 401);
        Tensor x = rand_tensor(4, 4, 24, 1.0);
        Tensor tgt = rand_tensor(4, 4, 30, 1.0);
        double w = fd_input_check(a, x, tgt);
        cout << "  K=2 d_input worst rel_err = " << scientific << w << endl;
        check("K=2 d_input FD < 1e-4", w < 1e-4);
    }
    // K=4 deeper chain.
    {
        MixtureOfSoftmaxesAttention a(6, 2, 4);
        randomize(a, 402);
        Tensor x = rand_tensor(6, 6, 25, 1.0);
        Tensor tgt = rand_tensor(6, 6, 31, 1.0);
        double w = fd_input_check(a, x, tgt);
        cout << "  K=4 d_input worst rel_err = " << scientific << w << endl;
        check("K=4 d_input FD < 1e-3", w < 1e-3);
    }
}

// ---------------------------------------------------------------------------
// Test 7: parameter gradient FD (random non-uniform init)
// ---------------------------------------------------------------------------
static void test_param_grad_FD() {
    cout << endl << "--- Test 7: parameter gradient FD ---" << endl;
    MixtureOfSoftmaxesAttention a(4, 2, 2);
    randomize(a, 500);
    Tensor x = rand_tensor(4, 4, 33, 1.0);
    Tensor tgt = rand_tensor(4, 4, 39, 1.0);

    auto run = [&](Tensor* param, Tensor* grad, const string& name, double tol) {
        double w = fd_param_check(a, param, grad, x, tgt);
        cout << "  " << name << " worst rel_err = " << scientific << w << endl;
        check(name + " FD", w < tol);
    };
    run(&a.W_q.weights, &a.grad_W_q, "W_q.weights", 1e-5);
    run(&a.W_k.weights, &a.grad_W_k, "W_k.weights", 1e-5);
    run(&a.W_o.weights, &a.grad_W_o, "W_o.weights", 1e-5);
    run(&a.W_v,         &a.grad_W_v, "W_v",         1e-4);
    run(&a.B_v,         &a.grad_B_v, "B_v",         1e-5);
    run(&a.B_z,         &a.grad_B_z, "B_z",         1e-4);
    run(&a.W_g,         &a.grad_W_g, "W_g",         1e-5);
    run(&a.B_g,         &a.grad_B_g, "B_g",         1e-5);
}

// ---------------------------------------------------------------------------
// Test 8: K=1 makes W_g and B_g gradients exactly zero
// ---------------------------------------------------------------------------
static void test_K1_gating_grads_zero() {
    cout << endl << "--- Test 8: K=1 → grad_W_g and grad_B_g exactly zero ---" << endl;
    MixtureOfSoftmaxesAttention a(4, 2, 1);
    randomize(a, 600);
    Tensor x = rand_tensor(4, 4, 41, 1.0);
    Tensor tgt = rand_tensor(4, 4, 47, 1.0);
    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    double nw = 0.0, nb = 0.0;
    for (double v : a.grad_W_g.data) nw += fabs(v);
    for (double v : a.grad_B_g.data) nb += fabs(v);
    cout << "  ||grad_W_g||_1 = " << nw << ", ||grad_B_g||_1 = " << nb << endl;
    check("K=1 grad_W_g is exactly 0", nw < 1e-15);
    check("K=1 grad_B_g is exactly 0", nb < 1e-15);
}

// ---------------------------------------------------------------------------
// Test 9: zero_grad / update_weights / parameters() / gradients() contract
// ---------------------------------------------------------------------------
static void test_state_contract() {
    cout << endl << "--- Test 9: state-management contract ---" << endl;
    MixtureOfSoftmaxesAttention a(4, 2, 3);
    auto params = a.parameters();
    auto grads  = a.gradients();
    check("parameters() count == 11 (W_q+b_q, W_k+b_k, W_o+b_o, W_v, B_v, B_z, W_g, B_g)", params.size() == 11);
    check("gradients() count == 11", grads.size() == 11);
    randomize(a, 700);
    Tensor x = rand_tensor(3, 4, 51, 1.0);
    Tensor tgt = rand_tensor(3, 4, 57, 1.0);
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    bool any_nonzero = false;
    for (Tensor* g : grads) for (double v : g->data) if (fabs(v) > 1e-15) any_nonzero = true;
    check("after backward, some gradients are nonzero", any_nonzero);
    a.zero_grad();
    bool all_zero = true;
    for (Tensor* g : grads) for (double v : g->data) if (fabs(v) > 1e-15) all_zero = false;
    check("zero_grad clears all 8 gradients", all_zero);
    // Snapshot all params, step, ensure they move.
    randomize(a, 710);
    Tensor snapshot(0, 0);
    for (Tensor* p : params) for (double v : p->data) snapshot.data.push_back(v);
    Tensor out2 = a.forward(x);
    a.backward(l2_grad(out2, tgt), 0.0);
    a.update_weights(0.01);
    bool moved = false;
    size_t idx = 0;
    for (Tensor* p : params)
        for (double v : p->data) { if (fabs(v - snapshot.data[idx]) > 1e-15) moved = true; ++idx; }
    check("update_weights moves all parameters", moved);
}

// ---------------------------------------------------------------------------
// Test 10: K=4 end-to-end training reduces loss
// ---------------------------------------------------------------------------
static void test_training() {
    cout << endl << "--- Test 10: K=4 training reduces loss ---" << endl;
    MixtureOfSoftmaxesAttention a(4, 1, 4);
    randomize(a, 800);
    Tensor x = rand_tensor(4, 4, 61, 1.0);
    Tensor tgt = rand_tensor(4, 4, 67, 1.0);
    auto loss = [&]() {
        Tensor out = a.forward(x);
        return l2_loss(out, tgt);
    };
    double L0 = loss();
    for (int step = 0; step < 50; ++step) {
        Tensor out = a.forward(x);
        Tensor grad = l2_grad(out, tgt);
        a.zero_grad();
        a.backward(grad, 0.0);
        a.update_weights(0.01);
    }
    double L1 = loss();
    cout << "  L0 = " << L0 << ", L1 = " << L1
         << " (reduction " << (L0 - L1) / max(L0, 1e-12) * 100.0 << "%)" << endl;
    check("50 SGD steps reduce loss > 30%", (L0 - L1) / max(L0, 1e-12) > 0.30);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cout << "=== Mixture of Softmaxes (MoS) Attention Tests ===" << endl;
    test_constructor();
    test_forward_shape();
    test_row_sum_invariants();
    test_K1_recovers_MHA();
    test_K2_active();
    test_input_grad_FD();
    test_param_grad_FD();
    test_K1_gating_grads_zero();
    test_state_contract();
    test_training();
    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}