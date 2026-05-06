// test_flash_attention.cpp — Gradient correctness tests for FlashAttention
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/attention/flash_attention.h"
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
// Test 1: FlashAttention forward pass basic sanity
// =====================================================================
static void test_flash_attention_forward() {
    cout << endl << "-- Test 1: FlashAttention forward pass --" << endl;

    FlashAttentionLayer layer(64, 4);  // d_model=64, num_heads=4

    size_t seq_len = 16;
    Tensor input = Tensor::random(64, seq_len, 0.5);

    Tensor out = layer.forward(input);

    check("FlashAttention output shape (d_model, seq_len)",
          out.rows == 64 && out.cols == seq_len);
    bool all_finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) all_finite = false;
    check("FlashAttention output values are finite", all_finite);
}

// =====================================================================
// Test 2: FlashAttention backward pass produces non-zero gradients
// =====================================================================
static void test_flash_attention_backward_nonzero() {
    cout << endl << "-- Test 2: FlashAttention backward pass — non-zero gradients --" << endl;

    size_t d_model = 64;
    size_t num_heads = 4;
    FlashAttentionLayer layer(d_model, num_heads);

    size_t seq_len = 16;
    Tensor input = Tensor::random(d_model, seq_len, 0.5);

    Tensor out = layer.forward(input);

    Tensor grad_out(d_model, seq_len);
    grad_out.fill(1.0);

    layer.zero_grad();
    Tensor grad_x = layer.backward(grad_out, 0.0);

    double gq_norm = tensor_l2norm(layer.grad_W_q);
    double gk_norm = tensor_l2norm(layer.grad_W_k);
    double gv_norm = tensor_l2norm(layer.grad_W_v);
    double go_norm = tensor_l2norm(layer.grad_W_o);

    check("FlashAttention grad_W_q is non-zero", gq_norm > 1e-10);
    check("FlashAttention grad_W_k is non-zero", gk_norm > 1e-10);
    check("FlashAttention grad_W_v is non-zero", gv_norm > 1e-10);
    check("FlashAttention grad_W_o is non-zero", go_norm > 1e-10);
    check("FlashAttention grad_x shape matches input", grad_x.rows == d_model && grad_x.cols == seq_len);
    double gx_norm = tensor_l2norm(grad_x);
    check("FlashAttention grad_x is non-zero", gx_norm > 1e-10);
}

// =====================================================================
// Test 3: FlashAttention numerical gradient on W_o
// =====================================================================
static void test_flash_attention_numerical_Wo() {
    cout << endl << "-- Test 3: FlashAttention numerical gradient on W_o --" << endl;

    size_t d_model = 32;
    size_t num_heads = 2;
    FlashAttentionLayer layer(d_model, num_heads);

    size_t seq_len = 8;
    Tensor input = Tensor::random(d_model, seq_len, 0.3);

    // Forward, backward, reset, then numerical check
    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(d_model, seq_len);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double orig_wo00 = layer.W_o[0][0];
    double eps = 1e-3;

    // loss = sum(outputs)
    layer.zero_grad();
    out = layer.forward(input);
    double loss_base = 0.0;
    for (size_t r = 0; r < out.rows; ++r)
        for (size_t c = 0; c < out.cols; ++c)
            loss_base += out[r][c];

    layer.W_o[0][0] = orig_wo00 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(input);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_o[0][0] = orig_wo00 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(input);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_o[0][0] = orig_wo00;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = layer.grad_W_o[0][0];
    double err = rel_error(num_grad, ana_grad);

    check("FlashAttention W_o[0][0] numerical vs analytical gradient", err < 1e-1);
}

// =====================================================================
// Test 4: FlashAttention numerical gradient on W_q
// =====================================================================
static void test_flash_attention_numerical_Wq() {
    cout << endl << "-- Test 4: FlashAttention numerical gradient on W_q --" << endl;

    size_t d_model = 32;
    size_t num_heads = 2;
    FlashAttentionLayer layer(d_model, num_heads);

    size_t seq_len = 8;
    Tensor input = Tensor::random(d_model, seq_len, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(d_model, seq_len);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double orig_wq10 = layer.W_q[1][0];
    double eps = 1e-3;

    layer.W_q[1][0] = orig_wq10 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(input);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_q[1][0] = orig_wq10 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(input);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_q[1][0] = orig_wq10;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = layer.grad_W_q[1][0];
    double err = rel_error(num_grad, ana_grad);

    check("FlashAttention W_q[1][0] numerical vs analytical gradient", err < 1e-1);
}

// =====================================================================
// Test 5: FlashAttention numerical gradient on W_k
// =====================================================================
static void test_flash_attention_numerical_Wk() {
    cout << endl << "-- Test 5: FlashAttention numerical gradient on W_k --" << endl;

    size_t d_model = 32;
    size_t num_heads = 2;
    FlashAttentionLayer layer(d_model, num_heads);

    size_t seq_len = 8;
    Tensor input = Tensor::random(d_model, seq_len, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(d_model, seq_len);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double orig_wk02 = layer.W_k[0][2];
    double eps = 1e-3;

    layer.W_k[0][2] = orig_wk02 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(input);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_k[0][2] = orig_wk02 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(input);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_k[0][2] = orig_wk02;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = layer.grad_W_k[0][2];
    double err = rel_error(num_grad, ana_grad);

    check("FlashAttention W_k[0][2] numerical vs analytical gradient", err < 1e-1);
}

// =====================================================================
// Test 6: FlashAttention numerical gradient on W_v
// =====================================================================
static void test_flash_attention_numerical_Wv() {
    cout << endl << "-- Test 6: FlashAttention numerical gradient on W_v --" << endl;

    size_t d_model = 32;
    size_t num_heads = 2;
    FlashAttentionLayer layer(d_model, num_heads);

    size_t seq_len = 8;
    Tensor input = Tensor::random(d_model, seq_len, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(d_model, seq_len);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double orig_wv11 = layer.W_v[1][1];
    double eps = 1e-3;

    layer.W_v[1][1] = orig_wv11 + eps;
    layer.zero_grad();
    Tensor out_plus = layer.forward(input);
    double loss_plus = 0.0;
    for (size_t r = 0; r < out_plus.rows; ++r)
        for (size_t c = 0; c < out_plus.cols; ++c)
            loss_plus += out_plus[r][c];

    layer.W_v[1][1] = orig_wv11 - eps;
    layer.zero_grad();
    Tensor out_minus = layer.forward(input);
    double loss_minus = 0.0;
    for (size_t r = 0; r < out_minus.rows; ++r)
        for (size_t c = 0; c < out_minus.cols; ++c)
            loss_minus += out_minus[r][c];

    layer.W_v[1][1] = orig_wv11;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = layer.grad_W_v[1][1];
    double err = rel_error(num_grad, ana_grad);

    check("FlashAttention W_v[1][1] numerical vs analytical gradient", err < 1e-1);
}

// =====================================================================
// Test 7: FlashAttention update_weights changes W matrices
// =====================================================================
static void test_flash_attention_update_weights() {
    cout << endl << "-- Test 7: FlashAttention update_weights changes W matrices --" << endl;

    FlashAttentionLayer layer(16, 2);
    Tensor input = Tensor::random(16, 8, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(16, 8);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    // Copy old weights
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

    check("FlashAttention update_weights actually changes W_q or W_k", changed);
}

// =====================================================================
// Test 8: FlashAttention zero_grad clears gradients
// =====================================================================
static void test_flash_attention_zero_grad() {
    cout << endl << "-- Test 8: FlashAttention zero_grad clears gradients --" << endl;

    FlashAttentionLayer layer(16, 2);
    Tensor input = Tensor::random(16, 8, 0.3);

    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor grad_out(16, 8);
    grad_out.fill(1.0);
    layer.backward(grad_out, 0.0);

    double norm_before = tensor_l2norm(layer.grad_W_q);
    check("FlashAttention grad_W_q is non-zero before zero_grad", norm_before > 1e-10);

    layer.zero_grad();
    double norm_after = tensor_l2norm(layer.grad_W_q);
    check("FlashAttention grad_W_q is zero after zero_grad", norm_after < 1e-10);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== FlashAttention Gradient Correctness Tests ===" << endl;
    cout << setprecision(8);

    test_flash_attention_forward();
    test_flash_attention_backward_nonzero();
    test_flash_attention_numerical_Wo();
    test_flash_attention_numerical_Wq();
    test_flash_attention_numerical_Wk();
    test_flash_attention_numerical_Wv();
    test_flash_attention_update_weights();
    test_flash_attention_zero_grad();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}