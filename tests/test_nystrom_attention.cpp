// test_nystrom_attention.cpp — Gradient correctness tests for NystromAttention
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/attention/nystrom_attention.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t[i][j] * t[i][j];
    return std::sqrt(s);
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8);
}

// =====================================================================
// Test 1: NystromAttention forward pass basic sanity
// =====================================================================
static void test_nystrom_forward() {
    cout << endl << "-- Test 1: NystromAttention forward pass --" << endl;

    NystromAttention layer(64, 4);  // embed_dim=64, num_heads=4

    size_t batch = 2, seq_len = 16, embed_dim = 64;
    // Shape: (batch, seq_len * embed_dim) flat storage
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.5);

    Tensor out = layer.forward(query, key, value);

    check("NystromAttention output shape (batch, seq_len*embed_dim)",
          out.rows == batch && out.cols == seq_len * embed_dim);

    bool all_finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) all_finite = false;
    check("NystromAttention output values are finite", all_finite);
}

// =====================================================================
// Test 2: NystromAttention backward pass produces non-zero gradients
// =====================================================================
static void test_nystrom_backward_nonzero() {
    cout << endl << "-- Test 2: NystromAttention backward — non-zero gradients --" << endl;

    size_t embed_dim = 64, num_heads = 4;
    NystromAttention layer(embed_dim, num_heads);

    size_t batch = 2, seq_len = 16;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.5);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);

    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);

    Tensor grad_x = layer.backward(grad_out, 0.0);

    double gq_norm = tensor_l2norm(layer.grad_W_q);
    double gk_norm = tensor_l2norm(layer.grad_W_k);
    double gv_norm = tensor_l2norm(layer.grad_W_v);
    double go_norm = tensor_l2norm(layer.grad_W_o);

    check("NystromAttention grad_W_q is non-zero", gq_norm > 1e-10);
    check("NystromAttention grad_W_k is non-zero", gk_norm > 1e-10);
    check("NystromAttention grad_W_v is non-zero", gv_norm > 1e-10);
    check("NystromAttention grad_W_o is non-zero", go_norm > 1e-10);
    check("NystromAttention grad_x shape matches input",
          grad_x.rows == batch && grad_x.cols == seq_len * embed_dim);
    double gx_norm = tensor_l2norm(grad_x);
    check("NystromAttention grad_x is non-zero", gx_norm > 1e-10);
}

// =====================================================================
// Test 3: NystromAttention numerical gradient on W_o
// =====================================================================
static void test_nystrom_numerical_Wo() {
    cout << endl << "-- Test 3: NystromAttention numerical gradient on W_o --" << endl;

    size_t embed_dim = 32, num_heads = 2;
    NystromAttention layer(embed_dim, num_heads);

    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.3);

    // Forward, backward to get analytical gradient
    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    // Save analytical gradient BEFORE numerical check disturbs state
    double ana_grad = layer.grad_W_o[0][0];

    double orig_wo00 = layer.W_o[0][0];
    double eps = 1e-3;

    // Plus perturbation
    layer.W_o[0][0] = orig_wo00 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(query, key, value);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    // Minus perturbation
    layer.W_o[0][0] = orig_wo00 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(query, key, value);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_o[0][0] = orig_wo00;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double err = rel_error(num_grad, ana_grad);

    check("NystromAttention W_o[0][0] numerical vs analytical gradient", err < 1e-5);
}

static double compute_l2_loss(const Tensor& out) {
    double loss = 0.0;
    for (size_t r = 0; r < out.rows; ++r)
        for (size_t c = 0; c < out.cols; ++c) {
            double d = out[r][c] - 1.0;
            loss += d * d;
        }
    return loss;
}

// =====================================================================
// Test 4: NystromAttention numerical gradient on W_q
// =====================================================================
static void test_nystrom_numerical_Wq() {
    cout << endl << "-- Test 4: NystromAttention numerical gradient on W_q --" << endl;

    size_t embed_dim = 32, num_heads = 2;
    NystromAttention layer(embed_dim, num_heads);

    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double ana_grad = layer.grad_W_q[1][0];

    double orig_wq10 = layer.W_q[1][0];
    double eps = 1e-3;

    layer.W_q[1][0] = orig_wq10 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(query, key, value);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_q[1][0] = orig_wq10 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(query, key, value);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_q[1][0] = orig_wq10;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double err = rel_error(num_grad, ana_grad);

    check("NystromAttention W_q[1][0] numerical vs analytical gradient", err < 1e-5);
}

// =====================================================================
// Test 5: NystromAttention numerical gradient on W_k
// =====================================================================
static void test_nystrom_numerical_Wk() {
    cout << endl << "-- Test 5: NystromAttention numerical gradient on W_k --" << endl;

    size_t embed_dim = 32, num_heads = 2;
    NystromAttention layer(embed_dim, num_heads);

    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double ana_grad = layer.grad_W_k[0][2];

    double orig_wk02 = layer.W_k[0][2];
    double eps = 1e-3;

    layer.W_k[0][2] = orig_wk02 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(query, key, value);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_k[0][2] = orig_wk02 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(query, key, value);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_k[0][2] = orig_wk02;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double err = rel_error(num_grad, ana_grad);

    check("NystromAttention W_k[0][2] numerical vs analytical gradient", err < 1e-5);
}

// =====================================================================
// Test 6: NystromAttention numerical gradient on W_v
// =====================================================================
static void test_nystrom_numerical_Wv() {
    cout << endl << "-- Test 6: NystromAttention numerical gradient on W_v --" << endl;

    size_t embed_dim = 32, num_heads = 2;
    NystromAttention layer(embed_dim, num_heads);

    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.3);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double ana_grad = layer.grad_W_v[1][1];

    double orig_wv11 = layer.W_v[1][1];
    double eps = 1e-3;

    layer.W_v[1][1] = orig_wv11 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(query, key, value);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_v[1][1] = orig_wv11 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(query, key, value);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_v[1][1] = orig_wv11;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double err = rel_error(num_grad, ana_grad);

    check("NystromAttention W_v[1][1] numerical vs analytical gradient", err < 1e-5);
}

// =====================================================================
// Test 7: NystromAttention update_weights changes W matrices
// =====================================================================
static void test_nystrom_update_weights() {
    cout << endl << "-- Test 7: NystromAttention update_weights changes W --" << endl;

    NystromAttention layer(16, 2);
    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * 16, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * 16, 0.3);
    Tensor value = Tensor::random(batch, seq_len * 16, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * 16);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    Tensor old_W_q = layer.W_q;
    Tensor old_W_k = layer.W_k;
    Tensor old_W_v = layer.W_v;
    Tensor old_W_o = layer.W_o;

    layer.update_weights(0.01);

    bool changed = false;
    for (size_t i = 0; i < layer.W_q.rows; ++i)
        for (size_t j = 0; j < layer.W_q.cols; ++j)
            if (std::abs(layer.W_q[i][j] - old_W_q[i][j]) > 1e-12) changed = true;
    if (!changed)
        for (size_t i = 0; i < layer.W_k.rows; ++i)
            for (size_t j = 0; j < layer.W_k.cols; ++j)
                if (std::abs(layer.W_k[i][j] - old_W_k[i][j]) > 1e-12) changed = true;

    check("NystromAttention update_weights changes weights", changed);
}

// =====================================================================
// Test 8: NystromAttention zero_grad clears gradients
// =====================================================================
static void test_nystrom_zero_grad() {
    cout << endl << "-- Test 8: NystromAttention zero_grad clears gradients --" << endl;

    NystromAttention layer(16, 2);
    size_t batch = 2, seq_len = 8;
    Tensor query = Tensor::random(batch, seq_len * 16, 0.3);
    Tensor key   = Tensor::random(batch, seq_len * 16, 0.3);
    Tensor value = Tensor::random(batch, seq_len * 16, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);
    Tensor grad_out(batch, seq_len * 16);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double norm_before = tensor_l2norm(layer.grad_W_q);
    check("NystromAttention grad_W_q non-zero before zero_grad", norm_before > 1e-10);

    layer.zero_grad();
    double norm_after = tensor_l2norm(layer.grad_W_q);
    check("NystromAttention grad_W_q zero after zero_grad", norm_after < 1e-10);
}

// =====================================================================
// Test 9: NystromAttention forward with short sequence (fallback path)
// =====================================================================
static void test_nystrom_short_sequence() {
    cout << endl << "-- Test 9: NystromAttention short seq (fallback path) --" << endl;

    NystromAttention layer(32, 2);

    size_t batch = 1, seq_len = 4, embed_dim = 32;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.5);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);

    check("NystromAttention short seq output shape",
          out.rows == batch && out.cols == seq_len * embed_dim);

    bool all_finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) all_finite = false;
    check("NystromAttention short seq output is finite", all_finite);

    // Backward on short seq
    Tensor grad_out(batch, seq_len * embed_dim);
    grad_out.fill(1.0);
    Tensor grad_x = layer.backward(grad_out, 0.0);
    double gq_norm = tensor_l2norm(layer.grad_W_q);
    check("NystromAttention short seq backward produces grad_W_q",
          gq_norm > 1e-10);
}

// =====================================================================
// Test 10: NystromAttention with custom num_landmarks
// =====================================================================
static void test_nystrom_custom_landmarks() {
    cout << endl << "-- Test 10: NystromAttention custom num_landmarks --" << endl;

    // embed_dim=32, num_heads=2, num_landmarks=2
    NystromAttention layer(32, 2, 2);

    size_t batch = 1, seq_len = 8, embed_dim = 32;
    Tensor query = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor key   = Tensor::random(batch, seq_len * embed_dim, 0.5);
    Tensor value = Tensor::random(batch, seq_len * embed_dim, 0.5);

    layer.zero_grad();
    Tensor out = layer.forward(query, key, value);

    check("NystromAttention custom landmarks output shape",
          out.rows == batch && out.cols == seq_len * embed_dim);

    bool all_finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) all_finite = false;
    check("NystromAttention custom landmarks output is finite", all_finite);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== NystromAttention Gradient Correctness Tests ===" << endl;
    cout << setprecision(8);

    test_nystrom_forward();
    test_nystrom_backward_nonzero();
    test_nystrom_numerical_Wo();
    test_nystrom_numerical_Wq();
    test_nystrom_numerical_Wk();
    test_nystrom_numerical_Wv();
    test_nystrom_update_weights();
    test_nystrom_zero_grad();
    test_nystrom_short_sequence();
    test_nystrom_custom_landmarks();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}