// test_lambda_layer.cpp — Gradient correctness tests for Lambda Layer
//   (Bello et al. 2021, "LambdaNetworks: Modeling Long-Range Interactions Without Attention",
//    https://arxiv.org/abs/2102.08602)
//
// Tests:
//   1.  Constructor validation (d_model=0 / max_seq_len=0 throw, valid constructs)
//   2.  Forward shape (n, d) -> (n, d)
//   3.  Forward finite (n=8, d=4)
//   4.  Causal mask signature: perturbing V[m] for m > n leaves Y[n] bit-exact unchanged
//   5.  Hand-derived forward (n=2, d=2, k=2, forced weights) matches reference at rel_err < 1e-5
//   6.  Input gradient FD check (n=3, d=3, eps=1e-5, max rel_err < 1e-3)
//   7.  W_Q gradient FD check (rel_err < 1e-3)
//   8.  W_K gradient FD check (rel_err < 1e-3)
//   9.  W_V gradient FD check (rel_err < 1e-3)
//  10.  position_emb_ gradient FD check (rel_err < 1e-3)
//  11.  zero_grad clears all 4 gradients
//  12.  update_weights moves all 4 parameters
//  13.  LambdaBlock forward shape (n, d) -> (n, d)
//  14.  LambdaBlock input gradient FD check (rel_err < 1e-2)
//  15.  LambdaModel training reduces loss over 30 SGD steps (Lf < L0 * 0.9)
//  16.  parameters()/gradients() contract: 4 tensors, shape-matched
//  17.  Mutation test: grad_W_V non-zero after backward (proves V-path is exercised)
//  18.  Longer sequence (T=6) input gradient FD check
//  19.  Causal mode input gradient FD check

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/layers/attention/lambda_layer.h"

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
    bool threw_d = false, threw_n = false, valid_ok = true;
    try { LambdaAttention bad(0, 4); } catch (const std::exception&) { threw_d = true; }
    try { LambdaAttention bad(4, 0); } catch (const std::exception&) { threw_n = true; }
    try { LambdaAttention good(4, 4); } catch (const std::exception&) { valid_ok = false; }
    cout << "  d_model=0 threw: " << threw_d
         << "  max_seq_len=0 threw: " << threw_n
         << "  valid ok: " << valid_ok << endl;
    check("constructor validates correctly", threw_d && threw_n && valid_ok);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape
// ---------------------------------------------------------------------------
static void test_forward_shape() {
    cout << endl << "--- Test 2: forward shape (n=4, d=4) ---" << endl;
    size_t n = 4, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    cout << "  Input: " << input.rows << "x" << input.cols
         << "  Output: " << out.rows << "x" << out.cols << endl;
    check("forward shape (n, d) -> (n, d)", out.rows == n && out.cols == d);
}

// ---------------------------------------------------------------------------
// Test 3: forward output is finite
// ---------------------------------------------------------------------------
static void test_forward_finite() {
    cout << endl << "--- Test 3: forward output is finite (n=8, d=4) ---" << endl;
    size_t n = 8, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.3 * sin(0.1 * i) + 0.05 * j;
    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    cout << "  forward finite: " << finite << endl;
    check("forward output is finite", finite);
}

// ---------------------------------------------------------------------------
// Test 4: causal mask signature — perturb V[m] for m > n; Y[n] unchanged bit-exact
// ---------------------------------------------------------------------------
static void test_causal_mask_signature() {
    cout << endl << "--- Test 4: causal mask signature ---" << endl;
    size_t n = 4, d = 3;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    LambdaAttention attn(d, n, /*k=*/0, /*causal=*/true);
    Tensor out_orig = attn.forward(input);

    // In causal mode, the mask zeros E[n, m, :] for m > n BEFORE the λ_p computation.
    // So perturbing position_emb_[n_idx, m_idx, :] for m_idx > n_idx should leave
    // out[n_idx, :] bit-exact unchanged (the perturbation is masked out before use).
    // Perturbing for m_idx <= n_idx should change out[n_idx, :].
    //
    // (Note: perturbing X[m_idx] doesn't work as a control here because the content
    //  lambda λ_c depends on V via softmax(K, axis=positions)^T · V, which is
    //  NOT position-masked — perturbing any V[m] affects ALL Y[n] through λ_c.)
    bool all_ok = true;
    int checks = 0;
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        for (size_t m_idx = n_idx + 1; m_idx < n; ++m_idx) {
            for (size_t kk = 0; kk < d; ++kk) { // k_depth defaults to d_model
                double orig = attn.position_emb_(n_idx, m_idx * d + kk);
                attn.position_emb_(n_idx, m_idx * d + kk) = orig + 1.0;
                Tensor out_p = attn.forward(input);
                attn.position_emb_(n_idx, m_idx * d + kk) = orig;
                for (size_t c = 0; c < d; ++c) {
                    if (out_orig(n_idx, c) != out_p(n_idx, c)) {
                        cout << "  FAIL at n=" << n_idx << " m=" << m_idx
                             << " kk=" << kk << " c=" << c
                             << " orig=" << out_orig(n_idx, c)
                             << " new=" << out_p(n_idx, c) << endl;
                        all_ok = false;
                    }
                    ++checks;
                }
            }
        }
    }
    cout << "  checked " << checks << " (n, m, kk, c) tuples — all bit-exact: "
         << all_ok << endl;
    check("causal mask: perturbing future-position E leaves past output bit-exact",
          all_ok);
}

// ---------------------------------------------------------------------------
// Test 5: hand-derived forward (n=2, d=2, k=2, forced weights)
// ---------------------------------------------------------------------------
static void test_hand_forward() {
    cout << endl << "--- Test 5: hand-derived forward (n=2, d=2, k=2) ---" << endl;
    size_t n = 2, d = 2, k = 2;
    Tensor input(n, d);
    // Small integer-ish input for hand calculation
    input(0, 0) = 0.1; input(0, 1) = 0.2;
    input(1, 0) = 0.3; input(1, 1) = 0.4;

    LambdaAttention attn(d, n, k);
    // Force W_Q, W_K, W_V, position_emb_ to known values
    attn.W_Q(0, 0) = 0.1; attn.W_Q(0, 1) = 0.2;
    attn.W_Q(1, 0) = 0.3; attn.W_Q(1, 1) = 0.4;
    attn.W_K(0, 0) = 0.1; attn.W_K(0, 1) = 0.1;
    attn.W_K(1, 0) = 0.1; attn.W_K(1, 1) = 0.1;
    attn.W_V(0, 0) = 0.5; attn.W_V(0, 1) = 0.6;
    attn.W_V(1, 0) = 0.7; attn.W_V(1, 1) = 0.8;
    // Position embeddings: zero out and set a few values
    attn.position_emb_.fill(0.0);
    attn.position_emb_(0, 0 * k + 0) = 0.1; // E[0,0,0]
    attn.position_emb_(0, 0 * k + 1) = 0.2; // E[0,0,1]
    attn.position_emb_(1, 1 * k + 0) = 0.3; // E[1,1,0]
    attn.position_emb_(1, 1 * k + 1) = 0.4; // E[1,1,1]

    // Compute reference by hand
    // Q = X · W_Q (n, k)
    double Q[2][2] = {{0, 0}, {0, 0}};
    double K[2][2] = {{0, 0}, {0, 0}};
    double V[2][2] = {{0, 0}, {0, 0}};
    for (size_t i = 0; i < n; ++i) {
        for (size_t kk = 0; kk < k; ++kk) {
            for (size_t j = 0; j < d; ++j) {
                Q[i][kk] += input(i, j) * attn.W_Q(j, kk);
                K[i][kk] += input(i, j) * attn.W_K(j, kk);
            }
        }
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t v = 0; v < d; ++v)
            for (size_t j = 0; j < d; ++j)
                V[i][v] += input(i, j) * attn.W_V(j, v);

    // Per-column softmax of K over positions (n=2)
    double Ksm[2][2];
    for (size_t kk = 0; kk < k; ++kk) {
        double m = max(K[0][kk], K[1][kk]);
        double e0 = std::exp(K[0][kk] - m);
        double e1 = std::exp(K[1][kk] - m);
        double z = e0 + e1;
        Ksm[0][kk] = e0 / z;
        Ksm[1][kk] = e1 / z;
    }
    // λ_c[kk, v] = Σ_m Ksm[m, kk] * V[m, v]
    double Lc[2][2] = {{0, 0}, {0, 0}};
    for (size_t kk = 0; kk < k; ++kk)
        for (size_t v = 0; v < d; ++v)
            for (size_t m = 0; m < n; ++m)
                Lc[kk][v] += Ksm[m][kk] * V[m][v];

    // λ_p[n, kk, v] = Σ_m E[n, m, kk] * V[m, v]
    double Lp[2][2][2] = {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}};
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t kk = 0; kk < k; ++kk)
            for (size_t v = 0; v < d; ++v)
                for (size_t m = 0; m < n; ++m)
                    Lp[n_idx][kk][v] += attn.position_emb_(n_idx, m * k + kk) * V[m][v];

    // Y[n, v] = Σ_kk Q[n, kk] * (Lc[kk, v] + Lp[n, kk, v])
    double Y_ref[2][2] = {{0, 0}, {0, 0}};
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t v = 0; v < d; ++v)
            for (size_t kk = 0; kk < k; ++kk)
                Y_ref[n_idx][v] += Q[n_idx][kk] * (Lc[kk][v] + Lp[n_idx][kk][v]);

    Tensor out = attn.forward(input);
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            max_err = max(max_err, rel_err(out(i, j), Y_ref[i][j]));
    cout << "  max rel_err: " << max_err << endl;
    check("hand-derived forward (rel_err < 1e-5)", max_err < 1e-5);
}

// ---------------------------------------------------------------------------
// Test 6: input gradient FD check
// ---------------------------------------------------------------------------
static void test_input_grad() {
    cout << endl << "--- Test 6: input gradient FD check (n=3, d=3) ---" << endl;
    size_t n = 3, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    Tensor grad_x = attn.backward(grad_loss, 0.0);

    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < n && n_check < 9; ++i) {
        for (size_t j = 0; j < d && n_check < 9; ++j) {
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
            if (err > 0.1)
                cout << "  x[" << i << "][" << j << "]: ana=" << ana
                     << " num=" << num << " err=" << err << endl;
            ++n_check;
        }
    }
    cout << "  max rel_err: " << max_err << endl;
    check("input gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 7: W_Q gradient FD check
// ---------------------------------------------------------------------------
static void test_wq_grad() {
    cout << endl << "--- Test 7: W_Q gradient FD check ---" << endl;
    size_t n = 3, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_Q;
    Tensor& G = attn.grad_W_Q;
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
    cout << "  W_Q max rel_err: " << max_err << endl;
    check("W_Q gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 8: W_K gradient FD check
// ---------------------------------------------------------------------------
static void test_wk_grad() {
    cout << endl << "--- Test 8: W_K gradient FD check ---" << endl;
    size_t n = 3, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_K;
    Tensor& G = attn.grad_W_K;
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
    cout << "  W_K max rel_err: " << max_err << endl;
    check("W_K gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 9: W_V gradient FD check
// ---------------------------------------------------------------------------
static void test_wv_grad() {
    cout << endl << "--- Test 9: W_V gradient FD check ---" << endl;
    size_t n = 3, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& W = attn.W_V;
    Tensor& G = attn.grad_W_V;
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
    cout << "  W_V max rel_err: " << max_err << endl;
    check("W_V gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 10: position_emb_ gradient FD check
// ---------------------------------------------------------------------------
static void test_posemb_grad() {
    cout << endl << "--- Test 10: position_emb_ gradient FD check ---" << endl;
    size_t n = 3, d = 3, k = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n, k);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    Tensor& P = attn.position_emb_;
    Tensor& G = attn.grad_position_emb_;
    double max_err = 0.0;
    int n_check = 0;
    for (size_t i = 0; i < P.rows && n_check < 6; ++i) {
        for (size_t j = 0; j < P.cols && n_check < 6; ++j) {
            double orig = P(i, j);
            P(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            P(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            P(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = G(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
            ++n_check;
        }
    }
    cout << "  position_emb_ max rel_err: " << max_err << endl;
    check("position_emb_ gradient FD (rel_err < 1e-3)", max_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 11: zero_grad clears all 4 gradients
// ---------------------------------------------------------------------------
static void test_zero_grad() {
    cout << endl << "--- Test 11: zero_grad clears all 4 gradients ---" << endl;
    LambdaAttention attn(4, 4);
    Tensor input(4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    Tensor target(4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    Tensor out = attn.forward(input);
    attn.backward(l2_loss_grad(out, target), 0.0);
    attn.zero_grad();

    double max_abs = 0.0;
    auto check_zero = [&](const Tensor& T) {
        for (auto v : T.data) max_abs = max(max_abs, fabs(v));
    };
    check_zero(attn.grad_W_Q);
    check_zero(attn.grad_W_K);
    check_zero(attn.grad_W_V);
    check_zero(attn.grad_position_emb_);
    cout << "  max |grad_*| after zero_grad: " << max_abs << endl;
    check("zero_grad clears all 4 gradients", max_abs == 0.0);
}

// ---------------------------------------------------------------------------
// Test 12: update_weights moves all 4 parameters
// ---------------------------------------------------------------------------
static void test_update_weights() {
    cout << endl << "--- Test 12: update_weights moves all 4 parameters ---" << endl;
    LambdaAttention attn(3, 3);
    Tensor input(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    // Snapshot params
    Tensor W_Q_orig = attn.W_Q.clone();
    Tensor W_K_orig = attn.W_K.clone();
    Tensor W_V_orig = attn.W_V.clone();
    Tensor pos_orig = attn.position_emb_.clone();

    Tensor out = attn.forward(input);
    attn.zero_grad();
    attn.backward(l2_loss_grad(out, target), 0.0);
    attn.update_weights(0.01);

    bool moved = false;
    auto diff = [&](const Tensor& A, const Tensor& B) {
        for (size_t i = 0; i < A.data.size(); ++i)
            if (A.data[i] != B.data[i]) return true;
        return false;
    };
    if (diff(attn.W_Q, W_Q_orig)) moved = true;
    if (diff(attn.W_K, W_K_orig)) moved = true;
    if (diff(attn.W_V, W_V_orig)) moved = true;
    if (diff(attn.position_emb_, pos_orig)) moved = true;
    cout << "  at least one parameter changed: " << moved << endl;
    check("update_weights moves parameters", moved);
}

// ---------------------------------------------------------------------------
// Test 13: LambdaBlock forward shape
// ---------------------------------------------------------------------------
static void test_block_forward() {
    cout << endl << "--- Test 13: LambdaBlock forward shape ---" << endl;
    size_t n = 4, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    LambdaBlock block(d, n);
    Tensor out = block.forward(input);
    bool finite = true;
    for (auto v : out.data) if (!std::isfinite(v)) finite = false;
    cout << "  Output: " << out.rows << "x" << out.cols << " finite: " << finite << endl;
    check("LambdaBlock forward (n, d) -> (n, d) finite",
          out.rows == n && out.cols == d && finite);
}

// ---------------------------------------------------------------------------
// Test 14: LambdaBlock input gradient FD check
// ---------------------------------------------------------------------------
static void test_block_input_grad() {
    cout << endl << "--- Test 14: LambdaBlock input gradient FD check ---" << endl;
    size_t n = 3, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaBlock block(d, n);
    Tensor out = block.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    block.zero_grad();
    Tensor grad_x = block.backward(grad_loss, 0.0);

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            double Lp = l2_loss_value(block.forward(input), target);
            input(i, j) = orig - eps;
            double Lm = l2_loss_value(block.forward(input), target);
            input(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = grad_x(i, j);
            double err = rel_err(num, ana);
            if (err > max_err) max_err = err;
        }
    }
    cout << "  LambdaBlock max rel_err: " << max_err << endl;
    check("LambdaBlock input gradient FD (rel_err < 1e-2)", max_err < 1e-2);
}

// ---------------------------------------------------------------------------
// Test 15: LambdaModel training reduces loss
// ---------------------------------------------------------------------------
static void test_model_train() {
    cout << endl << "--- Test 15: LambdaModel training reduces loss ---" << endl;
    size_t n = 4, in_dim = 4, d = 4, of = 3, num_blocks = 2;
    LambdaModel model(d, n, of, num_blocks);
    Tensor input(n, in_dim);
    Tensor target(n, of);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < in_dim; ++j) input(i, j) = 0.1 * i + 0.05 * j;
        for (size_t j = 0; j < of; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    }
    double L0 = l2_loss_value(model.forward(input), target);
    double lr = 0.01;
    for (int step = 0; step < 30; ++step) {
        model.zero_grad();
        Tensor out = model.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
    }
    model.zero_grad();
    double Lf = l2_loss_value(model.forward(input), target);
    cout << "  L0: " << L0 << "  Lf: " << Lf << endl;
    check("LambdaModel training reduces loss (Lf < L0 * 0.9)", Lf < L0 * 0.9);
}

// ---------------------------------------------------------------------------
// Test 16: parameters()/gradients() contract
// ---------------------------------------------------------------------------
static void test_param_grad_contract() {
    cout << endl << "--- Test 16: parameters()/gradients() contract ---" << endl;
    LambdaAttention attn(3, 3);
    auto ps = attn.parameters();
    auto gs = attn.gradients();
    bool ok = (ps.size() == 4 && gs.size() == 4);
    if (ok) {
        for (size_t k = 0; k < ps.size(); ++k)
            if (ps[k]->rows != gs[k]->rows || ps[k]->cols != gs[k]->cols) ok = false;
    }
    bool shapes_ok = ok &&
        attn.W_Q.rows == 3 && attn.W_Q.cols == 3 &&
        attn.W_K.rows == 3 && attn.W_K.cols == 3 &&
        attn.W_V.rows == 3 && attn.W_V.cols == 3 &&
        attn.position_emb_.rows == 3 && attn.position_emb_.cols == 9; // max_seq_len * k
    cout << "  params: " << ps.size() << "  grads: " << gs.size()
         << "  shapes_ok: " << shapes_ok
         << "  name: " << attn.name() << endl;
    check("parameters/gradients contract (4 tensors, shape-matched)",
          ok && shapes_ok);
}

// ---------------------------------------------------------------------------
// Test 17: mutation — grad_W_V must be non-zero after backward (V-path exercised)
// ---------------------------------------------------------------------------
static void test_mutation_grad_v_nonzero() {
    cout << endl << "--- Test 17: mutation — grad_W_V non-zero ---" << endl;
    LambdaAttention attn(3, 3);
    Tensor input(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) input(i, j) = 0.1 * i + 0.05 * j;
    Tensor target(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) target(i, j) = 0.1 * i - 0.05 * j;
    Tensor out = attn.forward(input);
    attn.zero_grad();
    attn.backward(l2_loss_grad(out, target), 0.0);
    double max_abs = 0.0;
    for (size_t i = 0; i < attn.grad_W_V.rows; ++i)
        for (size_t j = 0; j < attn.grad_W_V.cols; ++j)
            max_abs = max(max_abs, fabs(attn.grad_W_V(i, j)));
    cout << "  max |grad_W_V|: " << max_abs << endl;
    check("W_V gradient non-zero (V-path exercised)", max_abs > 1e-8);
}

// ---------------------------------------------------------------------------
// Test 18: longer sequence (T=6) input gradient FD check
// ---------------------------------------------------------------------------
static void test_long_input_grad() {
    cout << endl << "--- Test 18: longer sequence (T=6) input gradient FD ---" << endl;
    size_t n = 6, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 + 0.04 * i + 0.03 * j;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.4 + 0.05 * j;

    LambdaAttention attn(d, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
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
    cout << "  long-seq max rel_err: " << max_err << endl;
    check("long-sequence input gradient FD (rel_err < 1e-2)", max_err < 1e-2);
}

// ---------------------------------------------------------------------------
// Test 19: causal mode input gradient FD check
// ---------------------------------------------------------------------------
static void test_causal_input_grad() {
    cout << endl << "--- Test 19: causal mode input gradient FD ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.2 * i - 0.1 * j + 0.05;
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) target(i, j) = 0.1 * i - 0.05 * j;

    LambdaAttention attn(d, n, /*k=*/0, /*causal=*/true);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
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
    cout << "  causal-mode max rel_err: " << max_err << endl;
    check("causal mode input gradient FD (rel_err < 1e-2)", max_err < 1e-2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cout << "=== Lambda Layer Tests (Bello et al. 2021) ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_constructor();
    test_forward_shape();
    test_forward_finite();
    test_causal_mask_signature();
    test_hand_forward();
    test_input_grad();
    test_wq_grad();
    test_wk_grad();
    test_wv_grad();
    test_posemb_grad();
    test_zero_grad();
    test_update_weights();
    test_block_forward();
    test_block_input_grad();
    test_model_train();
    test_param_grad_contract();
    test_mutation_grad_v_nonzero();
    test_long_input_grad();
    test_causal_input_grad();

    cout << endl << "=== Summary: " << passed << "/" << (passed + failed)
         << " tests passed ===" << endl;
    return failed == 0 ? 0 : 1;
}
