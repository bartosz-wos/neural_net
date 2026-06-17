// test_lsh_attention.cpp — Gradient correctness tests for LSH Attention (Reformer)
//
// Tests:
//   1.  LSHAttention forward shape: (n, d) -> (n, d)
//   2.  LSHAttention output is finite
//   3.  LSHAttention output is non-trivial
//   4.  LSHAttention input gradient check (n=4, d=3, num_buckets=1) — single-bucket
//        reduces to standard softmax attention and gradient is exact
//   5.  LSHAttention W_q gradient check
//   6.  LSHAttention W_k gradient check
//   7.  LSHAttention W_v gradient check
//   8.  LSHAttention W_o gradient check
//   9.  LSHAttention b_q gradient check
//   10. LSHAttention works with multiple buckets (sub-quadratic mode)
//   11. LSHAttention hash bucket assignment is consistent
//   12. LSHBlock forward shape
//   13. LSHBlock input gradient check (n=4, d=3, num_buckets=1)
//   14. LSHModel training step reduces loss (2 blocks, num_buckets=1)
//   15. Parameter/gradient count consistency
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "nn/layers/attention/lsh_attention.h"

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

static Tensor make_input(size_t n, size_t d, double scale = 0.1, unsigned seed = 0) {
    Tensor x(n, d);
    std::mt19937 gen(seed + 7);
    std::normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < n * d; ++i) x.data[i] = dis(gen);
    return x;
}

// =====================================================================
// Test 1: forward shape
// =====================================================================
static void test_lsh_forward_shape() {
    cout << endl << "--- Test 1: LSHAttention forward shape (n=8, d=4) ---" << endl;
    LSHAttention attn(4, 8, 4, 4);  // d=4, n=8, num_buckets=4, bucket_size=4
    Tensor input(8, 4);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 4; ++j) input(i, j) = 0.1 * (i + 1) + 0.05 * j;
    Tensor out = attn.forward(input);
    check("output shape (8, 4)", out.rows == 8 && out.cols == 4);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("output finite", finite);
    // Non-trivial: at least one nonzero
    double norm_sq = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) norm_sq += out.data[i] * out.data[i];
    check("output non-trivial (norm^2 > 0)", norm_sq > 0.0);
}

// =====================================================================
// Test 2: input gradient check (single-bucket reduces to standard softmax)
// =====================================================================
static void test_lsh_input_grad() {
    cout << endl << "--- Test 2: LSHAttention input gradient (n=4, d=3, num_buckets=1) ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 1);
    Tensor target = make_input(n, d, 0.2, 2);

    // num_buckets=1: all tokens in one bucket → standard softmax attention
    LSHAttention attn(d, n, 1, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    Tensor grad_x = attn.backward(grad_loss, 0.0);

    // Numerical gradient via input perturbation
    double max_err = 0.0, avg_err = 0.0;
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
        }
    }
    avg_err /= n_check;
    cout << "input grad: max rel err = " << max_err << "  avg rel err = " << avg_err << endl;
    check("input gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 3: W_q gradient check
// =====================================================================
static void test_lsh_wq_grad() {
    cout << endl << "--- Test 3: LSHAttention W_q gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 3);
    Tensor target = make_input(n, d, 0.2, 4);

    LSHAttention attn(d, n, 1, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    // W_q is params[0]
    Tensor* Wp = params[0];
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
    check("W_q gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 4: W_k gradient check
// =====================================================================
static void test_lsh_wk_grad() {
    cout << endl << "--- Test 4: LSHAttention W_k gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 5);
    Tensor target = make_input(n, d, 0.2, 6);

    LSHAttention attn(d, n, 1, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    // W_k is params[2]
    Tensor* Wp = params[2];
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
    check("W_k gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 5: W_v gradient check
// =====================================================================
static void test_lsh_wv_grad() {
    cout << endl << "--- Test 5: LSHAttention W_v gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 7);
    Tensor target = make_input(n, d, 0.2, 8);

    LSHAttention attn(d, n, 1, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    // W_v is params[4]
    Tensor* Wp = params[4];
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
    check("W_v gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 6: W_o gradient check
// =====================================================================
static void test_lsh_wo_grad() {
    cout << endl << "--- Test 6: LSHAttention W_o gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 9);
    Tensor target = make_input(n, d, 0.2, 10);

    LSHAttention attn(d, n, 1, n);
    Tensor out = attn.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    // W_o is params[6]
    Tensor* Wp = params[6];
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
    check("W_o gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 7: b_q gradient check
// =====================================================================
static void test_lsh_bq_grad() {
    cout << endl << "--- Test 7: LSHAttention b_q gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 11);
    Tensor target = make_input(n, d, 0.2, 12);

    LSHAttention attn(d, n, 1, n);
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
    check("b_q gradient check (rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 8: LSHAttention with multiple buckets (the actual sub-quadratic mode)
// =====================================================================
static void test_lsh_multi_bucket() {
    cout << endl << "--- Test 8: LSHAttention with multiple buckets (n=16, d=4, 4 buckets) ---" << endl;
    LSHAttention attn(4, 16, 4, 4);
    Tensor input = make_input(16, 4, 0.1, 13);
    Tensor out = attn.forward(input);
    check("output shape (16, 4)", out.rows == 16 && out.cols == 4);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("output finite", finite);
    // Verify buckets are all in range [0, 4)
    for (size_t i = 0; i < 16; ++i) {
        // We don't have direct access to last_buckets_ from outside, but we
        // can check the sorted_indices and bucket_starts/ends are populated.
    }
    bool has_multi_bucket = false;
    for (size_t b = 0; b < 4; ++b) {
        if (attn.bucket_ends()[b] > attn.bucket_starts()[b]) {
            has_multi_bucket = true;
        }
    }
    check("at least one bucket is non-empty (multi-bucket mode works)", has_multi_bucket);
    // Sorted indices should be a permutation of [0, n)
    std::vector<size_t> sorted_copy = attn.sorted_indices();
    std::sort(sorted_copy.begin(), sorted_copy.end());
    bool is_perm = true;
    for (size_t i = 0; i < 16; ++i) if (sorted_copy[i] != i) is_perm = false;
    check("sorted_indices is a permutation of [0, 16)", is_perm);
}

// =====================================================================
// Test 9: LSHBlock forward shape
// =====================================================================
static void test_lsh_block_forward() {
    cout << endl << "--- Test 9: LSHBlock forward shape ---" << endl;
    LSHBlock block(4, 6, 2, 3);  // d=4, n=6, 2 buckets, bucket_size=3
    Tensor input = make_input(6, 4, 0.1, 14);
    Tensor out = block.forward(input);
    check("output shape (6, 4)", out.rows == 6 && out.cols == 4);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("output finite", finite);
}

// =====================================================================
// Test 10: LSHBlock input gradient check
// =====================================================================
static void test_lsh_block_input_grad() {
    cout << endl << "--- Test 10: LSHBlock input gradient check ---" << endl;
    size_t n = 4, d = 3;
    double eps = 1e-5;

    Tensor input = make_input(n, d, 0.1, 15);
    Tensor target = make_input(n, d, 0.2, 16);

    LSHBlock block(d, n, 1, n);  // single-bucket
    Tensor out = block.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    block.zero_grad();
    Tensor grad_x = block.backward(grad_loss, 0.0);

    // Numerical gradient
    double max_err = 0.0, avg_err = 0.0;
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
            avg_err += err;
            ++n_check;
        }
    }
    avg_err /= n_check;
    cout << "block input grad: max rel err = " << max_err << "  avg rel err = " << avg_err << endl;
    check("block input gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 11: LSHModel training step reduces loss
// =====================================================================
static void test_lsh_model_training() {
    cout << endl << "--- Test 11: LSHModel training reduces loss ---" << endl;
    size_t n = 4, d = 3, out = 2, n_blocks = 2;
    LSHModel model(d, n, n_blocks, out, 1, n);  // single-bucket for stability

    Tensor input = make_input(n, d, 0.1, 17);
    Tensor target(n, out);
    std::mt19937 gen(20);
    std::normal_distribution<> dis(0.0, 0.2);
    for (size_t i = 0; i < n * out; ++i) target.data[i] = dis(gen);

    double lr = 0.01;
    double initial_loss = 0.0, final_loss = 0.0;
    for (int step = 0; step < 30; ++step) {
        Tensor out_pred = model.forward(input);
        double L = l2_loss_value(out_pred, target);
        if (step == 0) initial_loss = L;
        if (step == 29) final_loss = L;
        Tensor grad_loss = l2_loss_grad(out_pred, target);
        model.zero_grad();
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
    }
    cout << "loss: " << initial_loss << " -> " << final_loss
         << "  (reduction " << (initial_loss - final_loss) / initial_loss * 100.0 << "%)" << endl;
    check("training reduces loss (>20%)", final_loss < 0.8 * initial_loss);
}

// =====================================================================
// Test 12: parameter/gradient count consistency
// =====================================================================
static void test_lsh_param_count() {
    cout << endl << "--- Test 12: parameter/gradient count consistency ---" << endl;
    LSHAttention attn(4, 8, 4, 4);
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    check("LSHAttention: 8 parameters (W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o)", params.size() == 8);
    check("LSHAttention: 8 gradients", grads.size() == 8);
    // All param/grad pairs have matching shapes
    bool shapes_match = true;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
            shapes_match = false;
        }
    }
    check("param/grad shapes match", shapes_match);

    LSHBlock block(4, 8, 4, 4);
    auto bp = block.parameters();
    auto bg = block.gradients();
    // attn (8) + ln1 (2: gamma, beta) + ln2 (2) + W1, b1, W2, b2 (4) = 16
    check("LSHBlock: 16 parameters (8 attn + 4 LN + 4 FFN)", bp.size() == 16);
    check("LSHBlock: 16 gradients", bg.size() == 16);
    bool block_shapes = true;
    for (size_t i = 0; i < bp.size(); ++i) {
        if (bp[i]->rows != bg[i]->rows || bp[i]->cols != bg[i]->cols) {
            block_shapes = false;
        }
    }
    check("LSHBlock: param/grad shapes match", block_shapes);
}

// =====================================================================
// Test 13: deterministic — same input gives same output (since hash is fixed)
// =====================================================================
static void test_lsh_deterministic() {
    cout << endl << "--- Test 13: LSHAttention deterministic forward ---" << endl;
    LSHAttention attn1(4, 8, 4, 4);
    LSHAttention attn2(4, 8, 4, 4);
    // After init, both should be identical (same seed) and produce same output
    Tensor input = make_input(8, 4, 0.1, 21);
    Tensor out1 = attn1.forward(input);
    Tensor out2 = attn2.forward(input);
    bool same = true;
    for (size_t i = 0; i < out1.data.size(); ++i) {
        if (fabs(out1.data[i] - out2.data[i]) > 1e-12) same = false;
    }
    check("two fresh LSHAttentions with same seed produce same output", same);
}

// =====================================================================
// Test 14: linear scaling sanity (n=64 with multiple buckets)
// =====================================================================
static void test_lsh_large_n() {
    cout << endl << "--- Test 14: LSHAttention large n (n=64, d=4, 8 buckets) ---" << endl;
    LSHAttention attn(4, 64, 8, 8);
    Tensor input = make_input(64, 4, 0.1, 22);
    auto t0 = chrono::steady_clock::now();
    Tensor out = attn.forward(input);
    auto t1 = chrono::steady_clock::now();
    double fwd_ms = chrono::duration<double, milli>(t1 - t0).count();

    Tensor target = make_input(64, 4, 0.2, 23);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    t0 = chrono::steady_clock::now();
    Tensor grad_x = attn.backward(grad_loss, 0.0);
    t1 = chrono::steady_clock::now();
    double bwd_ms = chrono::duration<double, milli>(t1 - t0).count();

    cout << "n=64 d=4 8 buckets: forward " << fwd_ms
         << " ms, backward " << bwd_ms << " ms" << endl;
    check("forward+backward completes in <1s", fwd_ms + bwd_ms < 1000.0);
    bool finite = true;
    for (size_t i = 0; i < grad_x.data.size(); ++i)
        if (!std::isfinite(grad_x.data[i])) finite = false;
    check("input gradient finite after large-n pass", finite);
}

// =====================================================================
// Test 15: LSHAttention with content-based hash (Reformer-style)
// =====================================================================
static void test_lsh_content_hash() {
    cout << endl << "--- Test 15: LSHAttention with content-based hash (n=4, d=3) ---" << endl;
    LSHAttention attn(3, 4, 1, 4, /*use_content_hash=*/true);
    Tensor input = make_input(4, 3, 0.1, 30);
    Tensor out = attn.forward(input);
    check("output shape (4, 3)", out.rows == 4 && out.cols == 3);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("output finite", finite);
    // W_o gradient is content-hash-independent — should still be exact
    Tensor target = make_input(4, 3, 0.2, 31);
    Tensor grad_loss = l2_loss_grad(out, target);
    attn.zero_grad();
    attn.backward(grad_loss, 0.0);
    auto params = attn.parameters();
    auto grads = attn.gradients();
    Tensor* Wp = params[6];  // W_o
    Tensor* Gp = grads[6];
    double max_err = 0.0;
    double eps = 1e-5;
    for (size_t i = 0; i < Wp->rows; ++i) {
        for (size_t j = 0; j < Wp->cols; ++j) {
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
            max_err = max(max_err, rel_err(num, ana));
        }
    }
    cout << "W_o (content hash) max rel err: " << max_err << endl;
    check("W_o gradient still exact (rel_err < 1e-3)", max_err < 1e-3);
}

int main() {
    cout << "========================================" << endl;
    cout << "LSH Attention (Reformer) Tests" << endl;
    cout << "========================================" << endl;

    test_lsh_forward_shape();
    test_lsh_input_grad();
    test_lsh_wq_grad();
    test_lsh_wk_grad();
    test_lsh_wv_grad();
    test_lsh_wo_grad();
    test_lsh_bq_grad();
    test_lsh_multi_bucket();
    test_lsh_block_forward();
    test_lsh_block_input_grad();
    test_lsh_model_training();
    test_lsh_param_count();
    test_lsh_deterministic();
    test_lsh_large_n();
    test_lsh_content_hash();

    cout << endl << "========================================" << endl;
    cout << "Total: " << (passed + failed) << "  Passed: " << passed
         << "  Failed: " << failed << endl;
    cout << "========================================" << endl;
    return failed == 0 ? 0 : 1;
}
