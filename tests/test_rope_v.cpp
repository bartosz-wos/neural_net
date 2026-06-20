// test_rope_v.cpp — Gradient correctness tests for RoPE-with-V (rotates Q/K/V).
//
// RoPE-on-K-and-V: extension of RoPE that also rotates the V tensor with the
// same per-position rotation. The Q·K^T inner product still depends only on
// relative position (because both Q and K get the same rotation per position),
// and V carries the same positional information so all three tensors share
// a consistent positional encoding.
//
// Tests:
//  1. Forward shape, output values differ from input
//  2. Orthogonality (L2 norm per token preserved for Q, K, and V)
//  3. Numerical gradient check on Q (input gradient)
//  4. Numerical gradient check on K (input gradient)
//  5. Numerical gradient check on V (input gradient)
//  6. Round-trip forward + backward gradient flow (Q, K, V all non-zero)
//  7. Different seq_len / dim / base values
//  8. Cache gradient accumulates correctly across the three tensors
//  9. Layer interface (single-input forward) still rotates V on a copy
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include "nn/layers/attention/rope_v.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}



static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8);
}

// Per-token L2 norm: returns a vector of (rows*seq_len) norms.
static vector<double> per_token_l2(const Tensor& t, int seq_len, int dim) {
    vector<double> norms;
    for (size_t b = 0; b < t.rows; ++b)
        for (int pos = 0; pos < seq_len; ++pos) {
            double s = 0.0;
            for (int d = 0; d < dim; ++d)
                s += t[b][pos * dim + d] * t[b][pos * dim + d];
            norms.push_back(std::sqrt(s));
        }
    return norms;
}

// =====================================================================
// Test 1: RoPEWithV forward — output differs from input (rotation applied)
// =====================================================================
static void test_rope_v_forward_basic() {
    cout << endl << "-- Test 1: RoPEWithV forward pass applies rotation --" << endl;

    int dim = 8;
    int max_seq = 32;
    RoPEWithV rope(dim, max_seq, 10000.0f);
    rope.precompute_theta_freqs(8);

    size_t batch = 2;
    size_t seq = 4;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq; ++s)
            for (int d = 0; d < dim; ++d) {
                q[b][s * dim + d] = (b * seq + s) * dim + d + 1;
                k[b][s * dim + d] = (b * seq + s) * dim + d + 1;
                v[b][s * dim + d] = (b * seq + s) * dim + d + 1;
            }

    Tensor q_orig = q;
    Tensor k_orig = k;
    Tensor v_orig = v;

    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);

    check("RoPEWithV output q shape matches input", q_rot.rows == batch && q_rot.cols == seq * dim);
    check("RoPEWithV output k shape matches input", k_rot.rows == batch && k_rot.cols == seq * dim);
    check("RoPEWithV output v shape matches input", v_rot.rows == batch && v_rot.cols == seq * dim);

    double max_diff_q = 0.0, max_diff_k = 0.0, max_diff_v = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            max_diff_q = std::max(max_diff_q, std::abs(q_rot[b][c] - q_orig[b][c]));
            max_diff_k = std::max(max_diff_k, std::abs(k_rot[b][c] - k_orig[b][c]));
            max_diff_v = std::max(max_diff_v, std::abs(v_rot[b][c] - v_orig[b][c]));
        }
    check("RoPEWithV Q output differs from input", max_diff_q > 1e-6);
    check("RoPEWithV K output differs from input", max_diff_k > 1e-6);
    check("RoPEWithV V output differs from input", max_diff_v > 1e-6);

    bool q_finite = true, k_finite = true, v_finite = true;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q_rot.cols; ++c) {
            if (!std::isfinite(q_rot[b][c])) q_finite = false;
            if (!std::isfinite(k_rot[b][c])) k_finite = false;
            if (!std::isfinite(v_rot[b][c])) v_finite = false;
        }
    check("RoPEWithV Q output values finite", q_finite);
    check("RoPEWithV K output values finite", k_finite);
    check("RoPEWithV V output values finite", v_finite);
}

// =====================================================================
// Test 2: RoPEWithV preserves L2 norm per token (orthogonal rotation)
// =====================================================================
static void test_rope_v_orthogonality() {
    cout << endl << "-- Test 2: RoPEWithV rotation preserves L2 norm (orthogonal) --" << endl;

    int dim = 8;
    int seq = 8;
    RoPEWithV rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 4;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    srand(42);
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq; ++s)
            for (int d = 0; d < dim; ++d) {
                q[b][s * dim + d] = (rand() % 1000) / 500.0 - 1.0;
                k[b][s * dim + d] = (rand() % 1000) / 500.0 - 1.0;
                v[b][s * dim + d] = (rand() % 1000) / 500.0 - 1.0;
            }

    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);

    auto q_norms_in  = per_token_l2(q, seq, dim);
    auto k_norms_in  = per_token_l2(k, seq, dim);
    auto v_norms_in  = per_token_l2(v, seq, dim);
    auto q_norms_out = per_token_l2(q_rot, seq, dim);
    auto k_norms_out = per_token_l2(k_rot, seq, dim);
    auto v_norms_out = per_token_l2(v_rot, seq, dim);

    bool all_ok = true;
    double max_err_q = 0.0, max_err_k = 0.0, max_err_v = 0.0;
    for (size_t i = 0; i < q_norms_in.size(); ++i) {
        max_err_q = std::max(max_err_q, std::abs(q_norms_in[i] - q_norms_out[i]));
        max_err_k = std::max(max_err_k, std::abs(k_norms_in[i] - k_norms_out[i]));
        max_err_v = std::max(max_err_v, std::abs(v_norms_in[i] - v_norms_out[i]));
        if (std::abs(q_norms_in[i] - q_norms_out[i]) > 1e-8) all_ok = false;
        if (std::abs(k_norms_in[i] - k_norms_out[i]) > 1e-8) all_ok = false;
        if (std::abs(v_norms_in[i] - v_norms_out[i]) > 1e-8) all_ok = false;
    }
    check("RoPEWithV L2 norm preserved for Q, K, V per token", all_ok);
    cout << "    [DEBUG] max L2-norm errors Q=" << max_err_q
         << " K=" << max_err_k << " V=" << max_err_v << endl;
}

// =====================================================================
// Test 3: Numerical gradient check on Q (input gradient)
// =====================================================================
static void test_rope_v_numerical_gradient_q() {
    cout << endl << "-- Test 3: RoPEWithV numerical gradient on Q (input) --" << endl;

    int dim = 8;
    int seq = 4;
    RoPEWithV rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    srand(7);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            q[b][c] = (rand() % 1000) / 300.0 - 0.5;
            k[b][c] = (rand() % 1000) / 300.0 - 0.5;
            v[b][c] = (rand() % 1000) / 300.0 - 0.5;
        }

    rope.zero_grad();
    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);
    Tensor gq(batch, seq * dim); gq.fill(1.0);
    Tensor gk(batch, seq * dim); gk.fill(0.0);
    Tensor gv(batch, seq * dim); gv.fill(0.0);
    Tensor grad_q = rope.backward_qkv(gq, gk, gv);

    double eps = 1e-4;
    double orig = q[0][2];

    Tensor q_plus = q; q_plus[0][2] = orig + eps;
    rope.zero_grad();
    auto [qp, kp, vp] = rope.forward(q_plus, k, v);
    double loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qp.cols; ++c)
            loss_plus += qp[b][c];

    Tensor q_minus = q; q_minus[0][2] = orig - eps;
    rope.zero_grad();
    auto [qm, km, vm] = rope.forward(q_minus, k, v);
    double loss_minus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qm.cols; ++c)
            loss_minus += qm[b][c];

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = grad_q[0][2];
    double err = rel_error(num_grad, ana_grad);
    check("RoPEWithV Q[0][2] numerical vs analytical gradient", err < 1e-1);
    cout << "    [DEBUG] Q[0][2]: num=" << num_grad << " ana=" << ana_grad << " rel_err=" << err << endl;
}

// =====================================================================
// Test 4: Numerical gradient check on K (input gradient)
// =====================================================================
static void test_rope_v_numerical_gradient_k() {
    cout << endl << "-- Test 4: RoPEWithV numerical gradient on K (input) --" << endl;

    int dim = 8;
    int seq = 4;
    RoPEWithV rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    srand(11);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            q[b][c] = (rand() % 1000) / 300.0 - 0.5;
            k[b][c] = (rand() % 1000) / 300.0 - 0.5;
            v[b][c] = (rand() % 1000) / 300.0 - 0.5;
        }

    rope.zero_grad();
    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);
    Tensor gq(batch, seq * dim); gq.fill(0.0);
    Tensor gk(batch, seq * dim); gk.fill(1.0);
    Tensor gv(batch, seq * dim); gv.fill(0.0);
    Tensor grad_q = rope.backward_qkv(gq, gk, gv);
    const Tensor& grad_k = rope.backward_k();
    const Tensor& grad_v = rope.backward_v();
    (void)grad_q;
    (void)grad_v;

    double eps = 1e-4;
    size_t idx = 1 * seq * dim + 3;  // batch=1, token=0, dim=3
    double orig = k[idx / q.cols][idx % q.cols];

    Tensor k_plus = k; k_plus[idx / q.cols][idx % q.cols] = orig + eps;
    rope.zero_grad();
    auto [qp, kp, vp] = rope.forward(q, k_plus, v);
    double loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < kp.cols; ++c)
            loss_plus += kp[b][c];

    Tensor k_minus = k; k_minus[idx / q.cols][idx % q.cols] = orig - eps;
    rope.zero_grad();
    auto [qm, km, vm] = rope.forward(q, k_minus, v);
    double loss_minus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < km.cols; ++c)
            loss_minus += km[b][c];

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = grad_k[idx / q.cols][idx % q.cols];
    double err = rel_error(num_grad, ana_grad);
    check("RoPEWithV K[b=1,t=0,d=3] numerical vs analytical gradient", err < 1e-1);
    cout << "    [DEBUG] K[b=1,t=0,d=3]: num=" << num_grad << " ana=" << ana_grad << " rel_err=" << err << endl;
}

// =====================================================================
// Test 5: Numerical gradient check on V (input gradient)
// =====================================================================
static void test_rope_v_numerical_gradient_v() {
    cout << endl << "-- Test 5: RoPEWithV numerical gradient on V (input) --" << endl;

    int dim = 8;
    int seq = 4;
    RoPEWithV rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    srand(13);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            q[b][c] = (rand() % 1000) / 300.0 - 0.5;
            k[b][c] = (rand() % 1000) / 300.0 - 0.5;
            v[b][c] = (rand() % 1000) / 300.0 - 0.5;
        }

    rope.zero_grad();
    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);
    Tensor gq(batch, seq * dim); gq.fill(0.0);
    Tensor gk(batch, seq * dim); gk.fill(0.0);
    Tensor gv(batch, seq * dim); gv.fill(1.0);
    Tensor grad_q = rope.backward_qkv(gq, gk, gv);
    const Tensor& grad_v = rope.backward_v();
    (void)grad_q;

    double eps = 1e-4;
    size_t b_idx = 0;
    size_t col_idx = seq * dim - 1;  // last element
    double orig = v[b_idx][col_idx];

    Tensor v_plus = v; v_plus[b_idx][col_idx] = orig + eps;
    rope.zero_grad();
    auto [qp, kp, vp] = rope.forward(q, k, v_plus);
    double loss_plus = 0.0;
    for (size_t bb = 0; bb < batch; ++bb)
        for (size_t c = 0; c < vp.cols; ++c)
            loss_plus += vp[bb][c];

    Tensor v_minus = v; v_minus[b_idx][col_idx] = orig - eps;
    rope.zero_grad();
    auto [qm, km, vm] = rope.forward(q, k, v_minus);
    double loss_minus = 0.0;
    for (size_t bb = 0; bb < batch; ++bb)
        for (size_t c = 0; c < vm.cols; ++c)
            loss_minus += vm[bb][c];

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = grad_v[b_idx][col_idx];
    double err = rel_error(num_grad, ana_grad);
    check("RoPEWithV V[0][last] numerical vs analytical gradient", err < 1e-1);
    cout << "    [DEBUG] V[0][last]: num=" << num_grad << " ana=" << ana_grad << " rel_err=" << err << endl;
}

// =====================================================================
// Test 6: Round-trip forward + backward gradient flow (all three tensors non-zero)
// =====================================================================
static void test_rope_v_backward_nonzero() {
    cout << endl << "-- Test 6: RoPEWithV backward produces non-zero gradients for Q, K, V --" << endl;

    int dim = 8;
    int seq = 8;
    RoPEWithV rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    srand(17);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            q[b][c] = (rand() % 1000) / 500.0 - 1.0;
            k[b][c] = (rand() % 1000) / 500.0 - 1.0;
            v[b][c] = (rand() % 1000) / 500.0 - 1.0;
        }

    rope.zero_grad();
    auto [q_rot, k_rot, v_rot] = rope.forward(q, k, v);
    Tensor gq(batch, seq * dim); gq.fill(1.0);
    Tensor gk(batch, seq * dim); gk.fill(0.5);
    Tensor gv(batch, seq * dim); gv.fill(-0.5);
    Tensor grad_q = rope.backward_qkv(gq, gk, gv);
    const Tensor& grad_k = rope.backward_k();
    const Tensor& grad_v = rope.backward_v();

    bool q_nonzero = false, k_nonzero = false, v_nonzero = false;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < grad_q.cols; ++c) {
            if (std::abs(grad_q[b][c]) > 1e-9) q_nonzero = true;
            if (std::abs(grad_k[b][c]) > 1e-9) k_nonzero = true;
            if (std::abs(grad_v[b][c]) > 1e-9) v_nonzero = true;
        }
    check("RoPEWithV grad_q is non-zero", q_nonzero);
    check("RoPEWithV grad_k is non-zero", k_nonzero);
    check("RoPEWithV grad_v is non-zero", v_nonzero);

    // Cache gradient should also be non-zero after summing across Q, K, V
    bool cache_nonzero = false;
    for (size_t r = 0; r < rope.grad_cos_cache.rows; ++r)
        for (size_t c = 0; c < rope.grad_cos_cache.cols; ++c)
            if (std::abs(rope.grad_cos_cache[r][c]) > 1e-9) cache_nonzero = true;
    check("RoPEWithV grad_cos_cache is non-zero", cache_nonzero);

    bool sin_nonzero = false;
    for (size_t r = 0; r < rope.grad_sin_cache.rows; ++r)
        for (size_t c = 0; c < rope.grad_sin_cache.cols; ++c)
            if (std::abs(rope.grad_sin_cache[r][c]) > 1e-9) sin_nonzero = true;
    check("RoPEWithV grad_sin_cache is non-zero", sin_nonzero);
}

// =====================================================================
// Test 7: Different seq_len / dim / base values
// =====================================================================
static void test_rope_v_different_shapes() {
    cout << endl << "-- Test 7: RoPEWithV with different seq_len, dim, and base --" << endl;

    // dim=16, seq=4, base=10000
    {
        RoPEWithV rope(16, 32, 10000.0f);
        rope.precompute_theta_freqs(4);
        Tensor q(1, 4 * 16);
        Tensor k(1, 4 * 16);
        Tensor v(1, 4 * 16);
        for (size_t c = 0; c < q.cols; ++c) { q[0][c] = c * 0.1; k[0][c] = c * 0.1; v[0][c] = c * 0.1; }
        auto [qr, kr, vr] = rope.forward(q, k, v);
        check("dim=16 forward shape", qr.rows == 1 && qr.cols == 4 * 16);
        check("dim=16 K forward shape", kr.rows == 1 && kr.cols == 4 * 16);
        check("dim=16 V forward shape", vr.rows == 1 && vr.cols == 4 * 16);
    }
    // dim=4, seq=8
    {
        RoPEWithV rope(4, 32, 10000.0f);
        rope.precompute_theta_freqs(8);
        Tensor q(2, 8 * 4);
        Tensor k(2, 8 * 4);
        Tensor v(2, 8 * 4);
        srand(31);
        for (size_t c = 0; c < q.cols; ++c) { q[0][c] = (rand()%100)/50.0-1.0; k[0][c] = (rand()%100)/50.0-1.0; v[0][c] = (rand()%100)/50.0-1.0; }
        auto [qr, kr, vr] = rope.forward(q, k, v);
        // Orthogonality
        auto q_in  = per_token_l2(q, 8, 4);
        auto q_out = per_token_l2(qr, 8, 4);
        auto v_in  = per_token_l2(v, 8, 4);
        auto v_out = per_token_l2(vr, 8, 4);
        bool ok = true;
        for (size_t i = 0; i < q_in.size(); ++i) {
            if (std::abs(q_in[i] - q_out[i]) > 1e-8) ok = false;
            if (std::abs(v_in[i] - v_out[i]) > 1e-8) ok = false;
        }
        check("dim=4,seq=8 L2-norm preserved for Q and V", ok);
    }
    // base=5000
    {
        RoPEWithV rope(8, 32, 5000.0f);
        rope.precompute_theta_freqs(4);
        Tensor q(1, 4 * 8);
        Tensor k(1, 4 * 8);
        Tensor v(1, 4 * 8);
        for (size_t c = 0; c < q.cols; ++c) { q[0][c] = 0.1 * c; k[0][c] = 0.1 * c; v[0][c] = 0.1 * c; }
        auto [qr, kr, vr] = rope.forward(q, k, v);
        bool finite = true;
        for (size_t c = 0; c < qr.cols; ++c)
            if (!std::isfinite(qr[0][c]) || !std::isfinite(vr[0][c])) finite = false;
        check("base=5000 forward finite (Q and V)", finite);
    }
}

// =====================================================================
// Test 8: Cache gradient accumulates correctly across Q, K, V
// (the dL/dcos_qkv should equal dL/dcos_q + dL/dcos_k + dL/dcos_v)
// =====================================================================
static void test_rope_v_cache_gradient_accumulation() {
    cout << endl << "-- Test 8: RoPEWithV cache gradient accumulates across Q, K, V --" << endl;

    int dim = 4;
    int seq = 2;
    RoPEWithV rope(dim, 16, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 1;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    Tensor v(batch, seq * dim);
    for (size_t c = 0; c < q.cols; ++c) {
        q[0][c] = 0.1 * c + 0.5;
        k[0][c] = 0.2 * c - 0.3;
        v[0][c] = -0.1 * c + 0.7;
    }

    // Build a one-hot gradient
    Tensor gq(batch, seq * dim); gq.fill(0.0);
    Tensor gk(batch, seq * dim); gk.fill(0.0);
    Tensor gv(batch, seq * dim); gv.fill(0.0);
    gq[0][0] = 1.0;
    gk[0][0] = 1.0;
    gv[0][0] = 1.0;

    rope.zero_grad();
    auto [qr, kr, vr] = rope.forward(q, k, v);
    rope.backward_qkv(gq, gk, gv);

    // Hand-compute expected cache gradient for (pos=0, dim=0):
    // grad_cos(0,0) = sum_b [ -dr_q[0] * x_q[dim/2] + dr_q[dim/2] * x_q[0]
    //                    - dr_k[0] * x_k[dim/2] + dr_k[dim/2] * x_k[0]
    //                    - dr_v[0] * x_v[dim/2] + dr_v[dim/2] * x_v[0] ]
    // With our setup: x_q[0]=0.5, x_q[2]=0.7; x_k[0]=-0.3, x_k[2]=0.1; x_v[0]=0.7, x_v[2]=0.5
    // dr_q[0]=1, dr_q[2]=0 (and similarly k/v), so:
    // grad_cos(0,0) = -(1)*0.7 + 0*0.5  +  -(1)*0.1 + 0*(-0.3)  +  -(1)*0.5 + 0*0.7
    //               = -0.7 - 0.1 - 0.5 = -1.3
    double expected_grad_cos = -0.7 - 0.1 - 0.5;
    double actual_grad_cos = rope.grad_cos_cache(0, 0);
    double err = rel_error(expected_grad_cos, actual_grad_cos);
    check("RoPEWithV grad_cos_cache(0,0) hand-computed matches", err < 1e-2);
    cout << "    [DEBUG] grad_cos(0,0): expected=" << expected_grad_cos
         << " actual=" << actual_grad_cos << " rel_err=" << err << endl;

    // sin cache: grad_sin = sum_b [ -dr_q[0] * x_q[0] - dr_q[dim/2] * x_q[dim/2] + ... ]
    // = -1*0.5 - 0*0.7  +  -1*(-0.3) - 0*0.1  +  -1*0.7 - 0*0.5
    // = -0.5 + 0.3 - 0.7 = -0.9
    double expected_grad_sin = -0.5 + 0.3 - 0.7;
    double actual_grad_sin = rope.grad_sin_cache(0, 0);
    double err2 = rel_error(expected_grad_sin, actual_grad_sin);
    check("RoPEWithV grad_sin_cache(0,0) hand-computed matches", err2 < 1e-2);
    cout << "    [DEBUG] grad_sin(0,0): expected=" << expected_grad_sin
         << " actual=" << actual_grad_sin << " rel_err=" << err2 << endl;
}

// =====================================================================
// Test 9: Layer interface (single-input forward) — rotates V on a copy too
// =====================================================================
static void test_rope_v_layer_interface() {
    cout << endl << "-- Test 9: RoPEWithV Layer interface (single-input) --" << endl;

    int dim = 4;
    int seq = 2;
    RoPEWithV rope(dim, 16, 10000.0f);
    rope.precompute_theta_freqs(seq);

    Tensor input(1, seq * dim);
    for (size_t c = 0; c < input.cols; ++c) input[0][c] = 0.5 + 0.1 * c;
    Tensor input_orig = input;

    Tensor out = rope.forward(input);

    check("Layer forward shape", out.rows == input.rows && out.cols == input.cols);

    bool differs = false;
    for (size_t c = 0; c < out.cols; ++c)
        if (std::abs(out[0][c] - input_orig[0][c]) > 1e-6) differs = true;
    check("Layer forward applies rotation", differs);

    // Parameters empty (no learnable)
    bool no_params = rope.parameters().empty();
    check("RoPEWithV has no learnable parameters", no_params);

    // zero_grad works
    rope.zero_grad();
    for (size_t r = 0; r < rope.grad_cos_cache.rows; ++r)
        for (size_t c = 0; c < rope.grad_cos_cache.cols; ++c) {
            if (std::abs(rope.grad_cos_cache[r][c]) > 1e-9 ||
                std::abs(rope.grad_sin_cache[r][c]) > 1e-9) {
                check("zero_grad leaves caches at zero", false);
            }
        }
    check("zero_grad clears both cache gradients", true);
}

int main() {
    cout << "==========================================" << endl;
    cout << " RoPEWithV (Rotary Position Embedding + V)" << endl;
    cout << "==========================================" << endl;

    test_rope_v_forward_basic();
    test_rope_v_orthogonality();
    test_rope_v_numerical_gradient_q();
    test_rope_v_numerical_gradient_k();
    test_rope_v_numerical_gradient_v();
    test_rope_v_backward_nonzero();
    test_rope_v_different_shapes();
    test_rope_v_cache_gradient_accumulation();
    test_rope_v_layer_interface();

    cout << endl << "==========================================" << endl;
    cout << " Total: " << (passed + failed) << "  Passed: " << passed
         << "  Failed: " << failed << endl;
    cout << "==========================================" << endl;

    return failed == 0 ? 0 : 1;
}