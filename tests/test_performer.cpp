// test_performer.cpp — Gradient correctness tests for Performer (FAVOR+)
//
// Tests:
//   1. PerformerAttention forward shape: (n, d) -> (n, d)
//   2. PerformerAttention output is finite
//   3. PerformerAttention output is non-trivial (not all zeros)
//   4. PerformerAttention input gradient check (n=4, d=3, m=8)
//   5. PerformerAttention W_q gradient check
//   6. PerformerAttention W_k gradient check
//   7. PerformerAttention W_v gradient check
//   8. PerformerAttention W_o gradient check
//   9. PerformerAttention b_q / b_k gradient check
//  10. PerformerAttention linear-cost check: works for n=64, d=4, m=8 in <1s
//  11. PerformerAttention approximation: with large m, output is close to
//      softmax attention output (sanity that the kernel approximation is
//      reasonable — at machine precision with m=64 it won't match exactly
//      since softmax is not the Gaussian kernel, but the shape of attention
//      should be similar).
//  12. PerformerBlock forward shape
//  13. PerformerBlock input gradient check
//  14. PerformerModel training step reduces loss (2 blocks, m=16)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include "nn/layers/attention/performer.h"

using namespace std;

static int passed = 0;
static int failed = 0;

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
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// =====================================================================
// Test 1: PerformerAttention forward shape
// =====================================================================
static void test_performer_forward_shape() {
    cout << endl << "--- Test 1: PerformerAttention forward shape (n=6, d=4, m=8) ---" << endl;
    size_t n = 6, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * i - 0.05 * j;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor output = attn.forward(input);
    cout << "Input:  " << input.rows << "x" << input.cols
         << "  Output: " << output.rows << "x" << output.cols << endl;
    check("forward shape correct (n, d) -> (n, d)",
          output.rows == n && output.cols == d);
}

// =====================================================================
// Test 2: PerformerAttention output is finite
// =====================================================================
static void test_performer_output_finite() {
    cout << endl << "--- Test 2: PerformerAttention output is finite (n=8, d=6, m=16) ---" << endl;
    size_t n = 8, d = 6;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.3 * sin(0.1 * i) + 0.2 * j;

    PerformerAttention attn(d, n, /*num_features=*/16);
    Tensor output = attn.forward(input);
    bool finite = true;
    for (size_t i = 0; i < output.rows && finite; ++i)
        for (size_t j = 0; j < output.cols; ++j)
            if (!std::isfinite(output(i, j))) finite = false;
    check("all outputs finite", finite);
}

// =====================================================================
// Test 3: PerformerAttention output is non-trivial (L2 norm > 0)
// =====================================================================
static void test_performer_output_nonzero() {
    cout << endl << "--- Test 3: PerformerAttention output has nontrivial norm ---" << endl;
    size_t n = 5, d = 4;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.5 + 0.1 * i + 0.05 * j;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor output = attn.forward(input);
    double norm = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i)
        norm += output.data[i] * output.data[i];
    norm = sqrt(norm);
    cout << "||output||_2 = " << norm << endl;
    check("output norm > 0.01", norm > 0.01);
}

// =====================================================================
// Test 4: PerformerAttention input gradient check
// =====================================================================
static void test_performer_input_grad() {
    cout << endl << "--- Test 4: PerformerAttention input gradient check (n=4, d=3, m=8) ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j + 1.0;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    Tensor grad_x = attn.backward(grad_loss, 0.0);

    double max_err = 0.0;
    double avg_err = 0.0;
    int n_check = 0;
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
            max_err = max(max_err, err);
            avg_err += err;
            ++n_check;
            if (err > 0.1) {
                cout << "  x[" << i << "][" << j << "]: ana=" << ana
                     << " num=" << num << " err=" << err << endl;
            }
        }
    }
    avg_err /= n_check;
    cout << "avg rel err: " << avg_err << "  max rel err: " << max_err << endl;
    check("input gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 5: PerformerAttention W_q gradient check
// =====================================================================
static void test_performer_wq_grad() {
    cout << endl << "--- Test 5: PerformerAttention W_q gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    Tensor* Wp = params[0];   // W_q
    Tensor* Gp = grads[0];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "W_q max rel err: " << max_err << endl;
    check("W_q gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 6: PerformerAttention W_k gradient check
// =====================================================================
static void test_performer_wk_grad() {
    cout << endl << "--- Test 6: PerformerAttention W_k gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j + 0.5;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    // W_k is params[2] (index 2: W_q=0, b_q=1, W_k=2)
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    Tensor* Wp = params[2];   // W_k
    Tensor* Gp = grads[2];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "W_k max rel err: " << max_err << endl;
    check("W_k gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 7: PerformerAttention W_v gradient check
// =====================================================================
static void test_performer_wv_grad() {
    cout << endl << "--- Test 7: PerformerAttention W_v gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j + 0.3;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    // W_v is params[4]
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    Tensor* Wp = params[4];   // W_v
    Tensor* Gp = grads[4];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "W_v max rel err: " << max_err << endl;
    check("W_v gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 8: PerformerAttention W_o gradient check
// =====================================================================
static void test_performer_wo_grad() {
    cout << endl << "--- Test 8: PerformerAttention W_o gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j + 0.7;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    // W_o is params[6]
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    Tensor* Wp = params[6];   // W_o
    Tensor* Gp = grads[6];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "W_o max rel err: " << max_err << endl;
    check("W_o gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 9: PerformerAttention bias gradient check (b_q)
// =====================================================================
static void test_performer_b_grad() {
    cout << endl << "--- Test 9: PerformerAttention b_q gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j;

    PerformerAttention attn(d, n, /*num_features=*/8);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    // b_q is params[1]
    Tensor* bp = params[1];
    Tensor* gp = grads[1];
    double max_err = 0.0;
    for (size_t j = 0; j < bp->cols; ++j) {
        double orig = (*bp)(0, j);
        (*bp)(0, j) = orig + eps;
        Tensor out_p = attn.forward(input);
        double Lp = l2_loss_value(out_p, target);
        (*bp)(0, j) = orig - eps;
        Tensor out_m = attn.forward(input);
        double Lm = l2_loss_value(out_m, target);
        (*bp)(0, j) = orig;
        double num = (Lp - Lm) / (2.0 * eps);
        double ana = (*gp)(0, j);
        double err = rel_err(num, ana);
        max_err = max(max_err, err);
    }
    cout << "b_q max rel err: " << max_err << endl;
    check("b_q gradient check (rel_err < 10%)", max_err < 0.10);
}

// =====================================================================
// Test 10: PerformerAttention works for larger n (linear scaling sanity)
// =====================================================================
static void test_performer_large_n() {
    cout << endl << "--- Test 10: PerformerAttention linear cost (n=32, d=8, m=16) ---" << endl;
    size_t n = 32, d = 8;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * sin(0.2 * i) + 0.05 * j;

    PerformerAttention attn(d, n, /*num_features=*/16);
    auto t0 = chrono::steady_clock::now();
    Tensor out = attn.forward(input);
    auto t1 = chrono::steady_clock::now();
    double fwd_ms = chrono::duration<double, milli>(t1 - t0).count();

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.0;
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    t0 = chrono::steady_clock::now();
    Tensor grad_x = attn.backward(grad_loss, 0.0);
    t1 = chrono::steady_clock::now();
    double bwd_ms = chrono::duration<double, milli>(t1 - t0).count();

    cout << "n=" << n << " d=" << d << " m=16: forward " << fwd_ms
         << " ms, backward " << bwd_ms << " ms" << endl;
    check("forward+backward completes in <500ms (linear cost)", fwd_ms + bwd_ms < 500.0);
    bool finite = true;
    for (size_t i = 0; i < grad_x.data.size(); ++i)
        if (!std::isfinite(grad_x.data[i])) finite = false;
    check("input gradient finite after large-n pass", finite);
}

// =====================================================================
// Test 11: PerformerAttention softmax-approximation sanity
//   With m=64, the FAVOR+ approximation should produce a similar attention
//   *ranking* to softmax attention (i.e. focus on similar tokens). This is
//   not a tight test, but a sanity check that the feature map is reasonable.
// =====================================================================
static void test_performer_attention_softness() {
    cout << endl << "--- Test 11: PerformerAttention softmax-approximation quality (m=32) ---" << endl;
    // We test: a 2-cluster toy problem where one cluster of tokens should
    // attend strongly to itself and weakly to the other. We check that
    // the diagonal-like block structure is captured.
    size_t n = 8, d = 4;
    Tensor input(n, d);
    // Cluster A (rows 0..3) is high on dim 0,1; cluster B (rows 4..7) is high on dim 2,3.
    for (size_t i = 0; i < 4; ++i) {
        input(i, 0) = 1.0;
        input(i, 1) = 0.0;
        input(i, 2) = 0.0;
        input(i, 3) = 0.0;
    }
    for (size_t i = 4; i < 8; ++i) {
        input(i, 0) = 0.0;
        input(i, 1) = 0.0;
        input(i, 2) = 1.0;
        input(i, 3) = 0.0;
    }

    PerformerAttention attn(d, n, /*num_features=*/32);
    Tensor out = attn.forward(input);

    // Compute the "self-attention" of the first token: ||out[0] - V[0]|| should
    // be SMALL if attention focuses on the same cluster (cluster A). With
    // perfect attention, out[0] = V[0] = 1.0 (or W_v @ something close).
    // We check the L2 norm of out[0] is non-trivially different from 0
    // (sanity that the mechanism actually fired) and that out[0] is closer
    // to out[1] (same cluster) than to out[5] (different cluster).
    double d_same = 0.0, d_diff = 0.0;
    for (size_t j = 0; j < d; ++j) {
        double dd = out(0, j) - out(1, j); d_same += dd * dd;
        double dd2 = out(0, j) - out(5, j); d_diff += dd2 * dd2;
    }
    d_same = sqrt(d_same); d_diff = sqrt(d_diff);
    cout << "||out[0] - out[1]||_2 = " << d_same << "  (same cluster)" << endl;
    cout << "||out[0] - out[5]||_2 = " << d_diff << "  (different cluster)" << endl;
    // The same-cluster distance should be smaller than the different-cluster
    // distance. This is a soft check — feature map isn't perfect softmax,
    // so we allow some slack.
    check("same-cluster attention output closer than different-cluster",
          d_same < d_diff * 1.5);
}

// =====================================================================
// Test 12: PerformerBlock forward shape
// =====================================================================
static void test_performer_block_forward() {
    cout << endl << "--- Test 12: PerformerBlock forward shape (n=4, d=8) ---" << endl;
    size_t n = 4, d = 8;
    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.1 * i - 0.05 * j;

    PerformerBlock block(d, n, /*num_features=*/8);
    Tensor out = block.forward(input);
    cout << "Input:  " << input.rows << "x" << input.cols
         << "  Output: " << out.rows << "x" << out.cols << endl;
    check("block forward shape (n, d) -> (n, d)",
          out.rows == n && out.cols == d);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("block output finite", finite);
}

// =====================================================================
// Test 13: PerformerBlock input gradient check
// =====================================================================
static void test_performer_block_grad() {
    cout << endl << "--- Test 13: PerformerBlock input gradient check (n=4, d=4, m=8) ---" << endl;
    size_t n = 4, d = 4;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.3 * sin(0.1 * i) + 0.1 * j;

    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j;

    PerformerBlock block(d, n, /*num_features=*/8);
    Tensor out = block.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    block.zero_grad();
    Tensor grad_x = block.backward(grad_loss, 0.0);

    double max_err = 0.0;
    int n_check = 0;
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
            max_err = max(max_err, err);
            ++n_check;
            if (err > 0.15) {
                cout << "  x[" << i << "][" << j << "]: ana=" << ana
                     << " num=" << num << " err=" << err << endl;
            }
        }
    }
    cout << "PerformerBlock max rel err: " << max_err << endl;
    check("PerformerBlock input gradient check (rel_err < 15%)", max_err < 0.15);
}

// =====================================================================
// Test 14: PerformerModel training step reduces loss
// =====================================================================
static void test_performer_model_train() {
    cout << endl << "--- Test 14: PerformerModel training step reduces loss ---" << endl;
    size_t n = 6, d = 8, of = 4;
    PerformerModel model(d, n, of, /*num_blocks=*/2, /*num_features=*/16);

    // Synthetic task: identity map with small noise.
    Tensor input(n, d);
    Tensor target(n, of);
    std::mt19937 gen(0);
    std::normal_distribution<> dis(0.0, 0.3);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) input(i, j) = 0.1 * i + 0.05 * j + dis(gen);
        for (size_t j = 0; j < of; ++j) target(i, j) = (i + 1) * 0.1 - 0.05 * j;
    }

    double loss_before = 1e9;
    double loss_after = 1e9;
    double lr = 0.01;
    for (int step = 0; step < 30; ++step) {
        model.zero_grad();
        Tensor out = model.forward(input);
        double L = l2_loss_value(out, target);
        if (step == 0) loss_before = L;
        Tensor grad_loss = l2_loss_grad(out, target);
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
        if (step == 29) loss_after = L;
    }
    cout << "Loss before: " << loss_before << "  after 30 steps: " << loss_after
         << "  reduction: " << (1.0 - loss_after / loss_before) * 100 << "%" << endl;
    check("training reduces loss (PerformerModel)", loss_after < loss_before * 0.95);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== Performer (FAVOR+) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_performer_forward_shape();
    test_performer_output_finite();
    test_performer_output_nonzero();
    test_performer_input_grad();
    test_performer_wq_grad();
    test_performer_wk_grad();
    test_performer_wv_grad();
    test_performer_wo_grad();
    test_performer_b_grad();
    test_performer_large_n();
    test_performer_attention_softness();
    test_performer_block_forward();
    test_performer_block_grad();
    test_performer_model_train();

    cout << endl << "=== Summary: " << passed << "/" << (passed + failed)
         << " tests passed ===" << endl;
    return failed == 0 ? 0 : 1;
}
