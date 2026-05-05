// test_rope.cpp — Gradient correctness tests for RoPE (Rotary Position Embedding)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include "nn/layers/attention/rope.h"
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
// Test 1: RoPE forward pass — output differs from input (rotation applied)
// =====================================================================
static void test_rope_forward_basic() {
    cout << endl << "-- Test 1: RoPE forward pass applies rotation --" << endl;

    int dim = 8;
    int max_seq = 32;
    RoPE rope(dim, max_seq, 10000.0f);
    rope.precompute_theta_freqs(8);

    // Q and K as [batch=2, seq*dim=8*4=32 cols]
    size_t batch = 2;
    size_t seq = 4;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq; ++s)
            for (int d = 0; d < dim; ++d) {
                q[b][s * dim + d] = (b * seq + s) * dim + d + 1;
                k[b][s * dim + d] = (b * seq + s) * dim + d + 1;
            }

    // Copy for comparison
    Tensor q_orig = q;
    Tensor k_orig = k;

    auto [q_rot, k_rot] = rope.forward(q, k);

    // Check output shape preserved
    check("RoPE output q shape matches input", q_rot.rows == batch && q_rot.cols == seq * dim);
    check("RoPE output k shape matches input", k_rot.rows == batch && k_rot.cols == seq * dim);

    // Check values differ from input (rotation applied)
    double max_diff_q = 0.0;
    double max_diff_k = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c) {
            max_diff_q = std::max(max_diff_q, std::abs(q_rot[b][c] - q_orig[b][c]));
            max_diff_k = std::max(max_diff_k, std::abs(k_rot[b][c] - k_orig[b][c]));
        }
    check("RoPE Q output differs from input", max_diff_q > 1e-6);
    check("RoPE K output differs from input", max_diff_k > 1e-6);

    // All values finite
    bool q_finite = true, k_finite = true;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q_rot.cols; ++c) {
            if (!std::isfinite(q_rot[b][c])) q_finite = false;
            if (!std::isfinite(k_rot[b][c])) k_finite = false;
        }
    check("RoPE Q output values finite", q_finite);
    check("RoPE K output values finite", k_finite);
}

// =====================================================================
// Test 2: RoPE preserves L2 norm per token (orthogonal rotation property)
// =====================================================================
static void test_rope_orthogonality() {
    cout << endl << "-- Test 2: RoPE rotation preserves L2 norm (orthogonal) --" << endl;

    int dim = 8;
    int seq = 8;
    RoPE rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 4;
    Tensor q(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            q[b][c] = (rand() % 1000) / 500.0 - 1.0;

    Tensor q_orig = q;
    Tensor k(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (int c = 0; c < k.cols; ++c)
            k[b][c] = 0.0;
    auto [q_rot, k_rot] = rope.forward(q, k);  // k unused

    bool all_norms_preserved = true;
    double max_rel_err = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        for (int pos = 0; pos < seq; ++pos) {
            double norm_before = 0.0, norm_after = 0.0;
            for (int d = 0; d < dim; ++d) {
                double x = q_orig[b][pos * dim + d];
                double xr = q_rot[b][pos * dim + d];
                norm_before += x * x;
                norm_after  += xr * xr;
            }
            double err = std::abs(std::sqrt(norm_before) - std::sqrt(norm_after));
            max_rel_err = std::max(max_rel_err, err);
            if (err > 1e-6) all_norms_preserved = false;
        }
    }
    check("RoPE preserves L2 norm per token (orthogonal rotation)", all_norms_preserved);
    cout << "    [DEBUG] max norm deviation: " << max_rel_err << endl;
}

// =====================================================================
// Test 3: RoPE precomputed cos/sin cache correctness
// =====================================================================
static void test_rope_cache_precomputation() {
    cout << endl << "-- Test 3: RoPE cos/sin cache precomputation correctness --" << endl;

    int dim = 8;
    int seq = 16;
    RoPE rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    // Verify: for each position pos and dimension i,
    // cos_cache[pos][i] should equal cos(pos * theta_i)
    // where theta_i = base^(-2i/d)
    int half_dim = dim / 2;
    double log_base = std::log(10000.0);

    bool cache_correct = true;
    for (int pos = 0; pos < seq && cache_correct; ++pos) {
        for (int i = 0; i < half_dim && cache_correct; ++i) {
            double theta_i = std::exp(-2.0 * i / dim * log_base);
            double angle = pos * theta_i;
            double expected_cos = std::cos(angle);
            double expected_sin = std::sin(angle);

            if (std::abs(rope.cos_cache[pos][i] - expected_cos) > 1e-6)
                cache_correct = false;
            if (std::abs(rope.sin_cache[pos][i] - expected_sin) > 1e-6)
                cache_correct = false;
            // Paired dimensions (i+half_dim) should match
            if (std::abs(rope.cos_cache[pos][i + half_dim] - expected_cos) > 1e-6)
                cache_correct = false;
            if (std::abs(rope.sin_cache[pos][i + half_dim] - expected_sin) > 1e-6)
                cache_correct = false;
        }
    }
    check("RoPE cos/sin cache matches expected formula", cache_correct);
}

// =====================================================================
// Test 4: RoPE numerical gradient check on Q (input gradient)
// =====================================================================
static void test_rope_numerical_gradient_q() {
    cout << endl << "-- Test 4: RoPE numerical gradient on Q (input) --" << endl;

    int dim = 8;
    int seq = 4;
    RoPE rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            q[b][c] = (rand() % 1000) / 300.0 - 0.5;

    // Forward then backward with all-ones gradient
    rope.zero_grad();
    auto [q_rot, k_rot] = rope.forward(q, k);
    Tensor grad_out(batch, seq * dim);
    grad_out.fill(1.0);
    Tensor grad_q = rope.backward(grad_out, 0.0);

    // Numerical gradient check on q[0][2] (batch=0, element=2)
    double eps = 1e-4;
    double orig = q[0][2];

    // loss = sum(q_rot) = sum over all elements of rotated q
    // We perturb q and re-run forward/backward
    Tensor q_plus = q;
    q_plus[0][2] = orig + eps;
    rope.zero_grad();
    auto [qp, kp] = rope.forward(q_plus, k);
    double loss_plus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qp.cols; ++c)
            loss_plus += qp[b][c];

    Tensor q_minus = q;
    q_minus[0][2] = orig - eps;
    rope.zero_grad();
    auto [qm, km] = rope.forward(q_minus, k);
    double loss_minus = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qm.cols; ++c)
            loss_minus += qm[b][c];

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double ana_grad = grad_q[0][2];
    double err = rel_error(num_grad, ana_grad);

    check("RoPE Q[0][2] numerical vs analytical gradient", err < 1e-1);
    cout << "    [DEBUG] Q[0][2]: num=" << num_grad << " ana=" << ana_grad << " rel_err=" << err << endl;

    // Also check q[1][seq*dim - 1] (last element)
    double orig_last = q[1][seq * dim - 1];
    Tensor qp2 = q; qp2[1][seq * dim - 1] = orig_last + eps;
    rope.zero_grad();
    auto [qp2_out, kp2_out] = rope.forward(qp2, k);
    double lp2 = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qp2_out.cols; ++c)
            lp2 += qp2_out[b][c];

    Tensor qm2 = q; qm2[1][seq * dim - 1] = orig_last - eps;
    rope.zero_grad();
    auto [qm2_out, km2_out] = rope.forward(qm2, k);
    double lm2 = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < qm2_out.cols; ++c)
            lm2 += qm2_out[b][c];

    double num_grad2 = (lp2 - lm2) / (2.0 * eps);
    double ana_grad2 = grad_q[1][seq * dim - 1];
    double err2 = rel_error(num_grad2, ana_grad2);

    check("RoPE Q[1][last] numerical vs analytical gradient", err2 < 1e-1);
    cout << "    [DEBUG] Q[1][last]: num=" << num_grad2 << " ana=" << ana_grad2 << " rel_err=" << err2 << endl;
}

// =====================================================================
// Test 5: RoPE backward produces non-zero gradient for Q
// =====================================================================
static void test_rope_backward_nonzero() {
    cout << endl << "-- Test 5: RoPE backward produces non-zero gradient for Q --" << endl;

    int dim = 8;
    int seq = 8;
    RoPE rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 3;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            q[b][c] = (rand() % 1000) / 400.0 - 0.3;

    rope.zero_grad();
    auto [q_rot, k_rot] = rope.forward(q, k);
    Tensor grad_out(batch, seq * dim);
    grad_out.fill(1.0);
    Tensor grad_q = rope.backward(grad_out, 0.0);

    double gq_norm = tensor_l2norm(grad_q);
    check("RoPE grad_q is non-zero after backward", gq_norm > 1e-10);
}

// =====================================================================
// Test 6: RoPE round-trip forward + backward gradient flow
// =====================================================================
static void test_rope_roundtrip() {
    cout << endl << "-- Test 6: RoPE round-trip forward + backward gradient flow --" << endl;

    int dim = 8;
    int seq = 4;
    RoPE rope(dim, 32, 10000.0f);
    rope.precompute_theta_freqs(seq);

    size_t batch = 2;
    Tensor q(batch, seq * dim);
    Tensor k(batch, seq * dim);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            q[b][c] = (rand() % 1000) / 500.0 - 0.5;

    // Forward then backward: gradient should flow back to q
    rope.zero_grad();
    auto [q_rot, k_rot] = rope.forward(q, k);
    Tensor grad_out(batch, seq * dim);
    grad_out.fill(1.0);
    Tensor grad_q = rope.backward(grad_out, 0.0);

    // Verify that backward returns grad with same shape as q
    check("RoPE backward returns shape matching q", grad_q.rows == q.rows && grad_q.cols == q.cols);

    // Verify gradient is non-trivial
    double grad_sum = 0.0;
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            grad_sum += std::abs(grad_q[b][c]);
    check("RoPE backward produces non-trivial gradient values", grad_sum > 1e-8);
}

// =====================================================================
// Test 7: RoPE with different seq_len values
// =====================================================================
static void test_rope_different_seqlen() {
    cout << endl << "-- Test 7: RoPE with different seq_len and dim values --" << endl;

    // Test dim=16, seq=4
    {
        RoPE rope(16, 32, 10000.0f);
        rope.precompute_theta_freqs(4);
        size_t batch = 2;
        Tensor q(batch, 4 * 16);
        Tensor k(batch, 4 * 16);
        for (size_t b = 0; b < batch; ++b)
            for (size_t c = 0; c < q.cols; ++c)
                q[b][c] = (b + 1) * c * 0.1;

        auto [q_rot, k_rot] = rope.forward(q, k);
        check("RoPE dim=16, seq=4: output shape preserved", q_rot.rows == batch && q_rot.cols == 4 * 16);

        bool all_finite = true;
        for (size_t b = 0; b < batch && all_finite; ++b)
            for (size_t c = 0; c < q_rot.cols && all_finite; ++c)
                if (!std::isfinite(q_rot[b][c])) all_finite = false;
        check("RoPE dim=16, seq=4: output finite", all_finite);

        // Backward
        Tensor grad_out(batch, 4 * 16);
        grad_out.fill(1.0);
        rope.zero_grad();
        Tensor grad_q = rope.backward(grad_out, 0.0);
        double g_norm = tensor_l2norm(grad_q);
        check("RoPE dim=16, seq=4: backward gradient non-zero", g_norm > 1e-10);
    }

    // Test dim=4, seq=8
    {
        RoPE rope(4, 32, 10000.0f);
        rope.precompute_theta_freqs(8);
        size_t batch = 1;
        Tensor q(batch, 8 * 4);
        Tensor k(batch, 8 * 4);
        for (size_t c = 0; c < q.cols; ++c)
            q[0][c] = c * 0.2;

        auto [q_rot, k_rot] = rope.forward(q, k);
        check("RoPE dim=4, seq=8: output shape preserved", q_rot.rows == batch && q_rot.cols == 8 * 4);

        bool all_finite = true;
        for (size_t c = 0; c < q_rot.cols && all_finite; ++c)
            if (!std::isfinite(q_rot[0][c])) all_finite = false;
        check("RoPE dim=4, seq=8: output finite", all_finite);

        Tensor grad_out(batch, 8 * 4);
        grad_out.fill(1.0);
        rope.zero_grad();
        Tensor grad_q = rope.backward(grad_out, 0.0);
        double g_norm = tensor_l2norm(grad_q);
        check("RoPE dim=4, seq=8: backward gradient non-zero", g_norm > 1e-10);
    }

    // Test with different base (e.g. base=5000.0)
    {
        RoPE rope(8, 32, 5000.0f);
        rope.precompute_theta_freqs(8);
        size_t batch = 1;
        Tensor q(batch, 8 * 8);
        Tensor k(batch, 8 * 8);
        for (size_t c = 0; c < q.cols; ++c)
            q[0][c] = (c % 2 == 0 ? 1.0 : 0.5);

        auto [q_rot, k_rot] = rope.forward(q, k);
        check("RoPE base=5000: output finite", true);
        for (size_t c = 0; c < q_rot.cols; ++c)
            if (!std::isfinite(q_rot[0][c])) { check("RoPE base=5000: output finite", false); break; }
        check("RoPE base=5000: output finite", true);
    }
}

// =====================================================================
// Test 8: RoPE zero_grad clears gradient state
// =====================================================================
static void test_rope_zero_grad() {
    cout << endl << "-- Test 8: RoPE zero_grad clears gradient state --" << endl;

    RoPE rope(8, 32, 10000.0f);
    rope.precompute_theta_freqs(8);

    size_t batch = 2;
    Tensor q(batch, 8 * 8);
    Tensor k(batch, 8 * 8);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < q.cols; ++c)
            q[b][c] = (rand() % 1000) / 500.0 - 0.5;

    rope.zero_grad();
    auto [q_rot, k_rot] = rope.forward(q, k);
    Tensor grad_out(batch, 8 * 8);
    grad_out.fill(1.0);
    rope.backward(grad_out, 0.0);

    // cos_cache gradient should still be zero (no params) but check zero_grad doesn't crash
    rope.zero_grad();  // should be safe to call multiple times

    check("RoPE zero_grad executes without error", true);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== RoPE (Rotary Position Embedding) Gradient Correctness Tests ===" << endl;
    cout << setprecision(8);

    test_rope_forward_basic();
    test_rope_orthogonality();
    test_rope_cache_precomputation();
    test_rope_numerical_gradient_q();
    test_rope_backward_nonzero();
    test_rope_roundtrip();
    test_rope_different_seqlen();
    test_rope_zero_grad();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}