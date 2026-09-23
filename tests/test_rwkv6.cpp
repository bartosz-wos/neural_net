// Tests for RWKV-6 "Finch" (Peng et al. 2024)
// "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
// https://arxiv.org/abs/2404.05892
// We implement Finch Time-Mixing only (channel-mixing is left as a future
// extension, matching the existing RWKV-4 / RWKV-7 time-mixing-only convention
// in this repo).
#include "nn/nn.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace nn;

static int g_pass = 0;
static int g_fail = 0;

struct CoutFlusher {
    CoutFlusher() { std::cout.setf(std::ios::unitbuf); }
} _flusher;

#define CHECK(cond, name)                                                    \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else { ++g_fail; std::cout << "FAIL: " << name                        \
                                   << " (line " << __LINE__ << ")\n"; }      \
    } while (0)

#define CHECK_NEAR(a, b, tol, name)                                         \
    do {                                                                     \
        double aa = (a), bb = (b);                                           \
        if (std::abs(aa - bb) <= (tol)) { ++g_pass; }                        \
        else { ++g_fail; std::cout << "FAIL: " << name                       \
                                   << " expected " << bb << " got " << aa    \
                                   << " (line " << __LINE__ << ")\n"; }      \
    } while (0)

// ============================================================================
// Centered finite-difference utilities
// ============================================================================
// Finite-difference gradient of (sum_t y_t · grad_output_t) w.r.t. p(i, j).
// IMPORTANT: caller must snapshot m's analytical gradient field before iterating
// over many fd_param calls — fd_param does NOT zero grad (it would erase the
// analytical values the test wants to compare against).
static double fd_param(Tensor& p, RWKV6TimeMix& m, Tensor& x, const Tensor& grad_output,
                       size_t i, size_t j, double eps = 1e-5) {
    double orig = p(i, j);
    p(i, j) = orig + eps;
    Tensor yp = m.forward(x);
    double fp = 0.0;
    for (size_t r = 0; r < yp.rows; ++r)
        for (size_t c = 0; c < yp.cols; ++c)
            fp += yp(r, c) * grad_output(r, c);
    p(i, j) = orig - eps;
    Tensor ym = m.forward(x);
    double fm = 0.0;
    for (size_t r = 0; r < ym.rows; ++r)
        for (size_t c = 0; c < ym.cols; ++c)
            fm += ym(r, c) * grad_output(r, c);
    p(i, j) = orig;
    return (fp - fm) / (2 * eps);
}

static double fd_input_elem(RWKV6TimeMix& m, Tensor& x, size_t i, size_t j,
                            const Tensor& grad_output, double eps = 1e-5) {
    double orig = x(i, j);
    x(i, j) = orig + eps;
    Tensor yp = m.forward(x);
    double fp = 0.0;
    for (size_t r = 0; r < yp.rows; ++r)
        for (size_t c = 0; c < yp.cols; ++c)
            fp += yp(r, c) * grad_output(r, c);
    x(i, j) = orig - eps;
    Tensor ym = m.forward(x);
    double fm = 0.0;
    for (size_t r = 0; r < ym.rows; ++r)
        for (size_t c = 0; c < ym.cols; ++c)
            fm += ym(r, c) * grad_output(r, c);
    x(i, j) = orig;
    return (fp - fm) / (2 * eps);
}

// Compute relative error vs FD for an entire tensor field of param grads.
static double max_rel_err(const Tensor& ana, const Tensor& fd) {
    double max_e = 0.0;
    for (size_t i = 0; i < ana.rows; ++i) {
        for (size_t j = 0; j < ana.cols; ++j) {
            double a = ana(i, j);
            double f = fd(i, j);
            double denom = std::max(std::abs(a), std::abs(f));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(a - f) / denom;
            if (e > max_e) max_e = e;
        }
    }
    return max_e;
}

// ----------------------------------------------------------------------------
// Test 1: Constructor validation + accessors
// ----------------------------------------------------------------------------
static void test_1_constructor() {
    std::cout << "Test 1: RWKV6TimeMix constructor validation + accessors\n";

    // d=0 throws
    bool threw_d0 = false;
    try { RWKV6TimeMix m(0, 1); } catch (const std::invalid_argument&) { threw_d0 = true; }
    CHECK(threw_d0, "d=0 throws");

    // num_heads=0 throws
    bool threw_h0 = false;
    try { RWKV6TimeMix m(4, 0); } catch (const std::invalid_argument&) { threw_h0 = true; }
    CHECK(threw_h0, "num_heads=0 throws");

    // d % num_heads != 0 throws
    bool threw_div = false;
    try { RWKV6TimeMix m(5, 2); } catch (const std::invalid_argument&) { threw_div = true; }
    CHECK(threw_div, "d % num_heads != 0 throws");

    // valid construction
    {
        RWKV6TimeMix m(8, 1);
        CHECK(m.d() == 8, "d() == 8");
        CHECK(m.num_heads() == 1, "num_heads() == 1");
        CHECK(m.head_dim() == 8, "head_dim() == 8");
    }
    {
        RWKV6TimeMix m(12, 3);
        CHECK(m.d() == 12, "d() == 12 (3-head)");
        CHECK(m.num_heads() == 3, "num_heads() == 3");
        CHECK(m.head_dim() == 4, "head_dim() == 4");
    }
    {
        RWKV6TimeMix m(8, 1, 8);   // smaller LoRA rank
        CHECK(m.num_lora_ranks() == 8, "num_lora_ranks() == 8");
    }

    // name
    {
        RWKV6TimeMix m(4, 1);
        CHECK(m.name() == "RWKV6TimeMix", "name() == RWKV6TimeMix");
    }
}

// ----------------------------------------------------------------------------
// Test 2: At init, W_o = 0 → output is zero (per Appendix I init spec)
// ----------------------------------------------------------------------------
static void test_2_init_zero_output() {
    std::cout << "Test 2: W_o=0 at init → output is zero\n";
    // Use a small random input; with W_o=0 the output must be exactly zero
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    Tensor y = m.forward(x);
    bool all_zero = true;
    double max_abs = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            max_abs = std::max(max_abs, std::abs(y(i, j)));
            if (std::abs(y(i, j)) > 1e-12) all_zero = false;
        }
    CHECK(all_zero, "init: all output entries == 0 (max_abs < 1e-12)");
    CHECK(max_abs == 0.0, "init: max_abs == 0");
}

// ----------------------------------------------------------------------------
// Test 3: Forward shape + finite + nonzero (with non-zero W_o)
// ----------------------------------------------------------------------------
static void test_3_forward_shape() {
    std::cout << "Test 3: Forward shape (T=4, d=8) finite + nonzero\n";
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(8, 1);
    // Randomize W_o to a small non-zero value so output isn't trivially zero
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor y = m.forward(x);
    CHECK(y.rows == 4, "y.rows == 4");
    CHECK(y.cols == 8, "y.cols == 8");
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y(i, j))) finite = false;
            if (std::abs(y(i, j)) > 1e-6) nonzero = true;
        }
    }
    CHECK(finite, "all forward entries finite");
    CHECK(nonzero, "at least one forward entry nonzero");

    // Multi-head version
    RWKV6TimeMix mh(8, 2);  // head_dim = 4
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            mh.W_o.weights(i, j) = uni(rng);
    Tensor yh = mh.forward(x);
    CHECK(yh.rows == 4 && yh.cols == 8, "multi-head forward shape (4, 8)");
    finite = true; nonzero = false;
    for (size_t i = 0; i < yh.rows; ++i)
        for (size_t j = 0; j < yh.cols; ++j) {
            if (!std::isfinite(yh(i, j))) finite = false;
            if (std::abs(yh(i, j)) > 1e-6) nonzero = true;
        }
    CHECK(finite, "multi-head finite");
    CHECK(nonzero, "multi-head nonzero");
}

// ----------------------------------------------------------------------------
// Test 4: T=1 hand-derived reference
// ----------------------------------------------------------------------------
// At T=1, s_0 = 0, wkv_0 = diag(u)·k_0^T·v_0, o_0 = r_0 · wkv_0
// (state never evolved). We override a fresh Finch so W_o = I, u = 1, all
// random tensors initialized to a known configuration; verify exact match.
static void test_4_t1_reference() {
    std::cout << "Test 4: T=1 hand-derived reference\n";

    // Build a Finch with all known values; we test forward-only here
    RWKV6TimeMix m(4, 1);
    // Disable LoRA path: set all LoRA A to 0 and λ = 0 so ddlerp reduces to
    //   ddlerp(a, b) = a + (b − a) ⊙ 0 = a
    // meaning the projection input is just x_t (no shift), and the
    // shift input `x_{t-1}` is irrelevant for T=1.
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_d, &m.A_x}) {
        A->fill(0.0);
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_d, &m.B_x}) {
        B->fill(0.0);
    }
    for (auto* L : {&m.lambda_r, &m.lambda_k, &m.lambda_v, &m.lambda_g, &m.lambda_d, &m.lambda_x}) {
        L->fill(0.0);
    }

    // Set W_r/W_k/W_v/W_g to known xavier-style values, and biases to 0.
    // We set W_o = I so output = r · wkv (with no LN/SiLU since we omitted those).
    m.W_o.weights.fill(0.0);
    for (size_t i = 0; i < 4; ++i) m.W_o.weights(i, i) = 1.0;
    m.W_o.bias.fill(0.0);

    // Set u to a known vector
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = (i + 1) * 0.1;

    // Set W_k, W_v to identity-ish: k = x_t, v = x_t (so that kv_t = x_t^T·x_t)
    m.W_k.weights.fill(0.0); for (size_t i = 0; i < 4; ++i) m.W_k.weights(i, i) = 1.0;
    m.W_v.weights.fill(0.0); for (size_t i = 0; i < 4; ++i) m.W_v.weights(i, i) = 1.0;
    m.W_k.bias.fill(0.0); m.W_v.bias.fill(0.0);

    // Set W_r to a specific matrix to make r_t a known function of x_t
    m.W_r.weights.fill(0.0);
    m.W_r.weights(0, 0) = 0.7; m.W_r.weights(1, 1) = 0.5;
    m.W_r.weights(2, 2) = 0.3; m.W_r.weights(3, 3) = 0.1;
    m.W_r.bias.fill(0.0);

    // Disable the LoRA-d path so d_t = x_t directly: set A_d = 0 and λ_d = 0
    // so lora_d ≡ 0, then ddlerpd(a, b) = a + (b − a) ⊙ 0 = a. With T=1, x_0 is a
    // and x_{-1} = 0 = b, so d_t = x_0.
    m.A_d.fill(0.0); m.B_d.fill(0.0); m.lambda_d.fill(0.0);
    // (mu_w doesn't matter when lora_d = 0)
    for (size_t i = 0; i < 4; ++i) m.mu_w(0, i) = 0.5;

    // Set W_g such that g_t = 0 (we'll ignore g in this T=1 test, just need it non-NaN)
    m.W_g.weights.fill(0.0);
    m.W_g.bias.fill(0.0);

    // A hand-crafted input
    Tensor x(1, 4);
    x(0, 0) = 0.1; x(0, 1) = 0.2; x(0, 2) = 0.3; x(0, 3) = 0.4;

    Tensor y = m.forward(x);

    // Hand-derive:
    // x_t = [0.1, 0.2, 0.3, 0.4]
    // k_t = x_t (W_k = I, b_k = 0)
    // v_t = x_t (W_v = I, b_v = 0)
    // r_t = [0.07, 0.1, 0.09, 0.04]
    // d_pre_t = x_t; d_t = exp(-exp(d_t))
    //   d_t[0] = exp(-exp(0.1)) = exp(-1.10517) = 0.3314
    //   d_t[1] = exp(-exp(0.2)) = exp(-1.2214) = 0.2947
    //   d_t[2] = exp(-exp(0.3)) = exp(-1.3499) = 0.2592
    //   d_t[3] = exp(-exp(0.4)) = exp(-1.4918) = 0.2247
    // w_t = w_t  (per-channel)
    // s_0 = 0;  wkv_0 = diag(u)·k_0^T·v_0 + 0 = diag(u)·outer(x_0, x_0)
    //   wkv_0[i, j] = u[i] · x_i · x_j  (with u = [0.1, 0.2, 0.3, 0.4])
    //   wkv_0[0,0] = 0.1*0.1*0.1 = 0.001
    //   wkv_0[0,1] = 0.1*0.1*0.2 = 0.002
    //   etc.
    // o_0[i] = sum_j wkv_0[j, i] · r_0[j]
    //   o_0[0] = 0.001*0.07 + 0.002*0.1 + 0.003*0.09 + 0.004*0.04
    //          = 0.00007 + 0.0002 + 0.00027 + 0.00016 = 0.0007
    //   o_0[1] = sum_j (u[j]*x_j*x_1) * r_j = x_1 * sum_j u[j]*x_j*r_j
    // etc. Just compute below.
    Tensor wkv_ref(4, 4);
    double u_arr[4] = {0.1, 0.2, 0.3, 0.4};
    double x_arr[4] = {0.1, 0.2, 0.3, 0.4};
    // s_0 = kv_0 (since s_{-1} = 0). wkv_0 = s_0 + diag(u)·kv_0 = (1 + u[i])·kv_0[i, j].
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            wkv_ref(i, j) = (1.0 + u_arr[i]) * x_arr[i] * x_arr[j];
    Tensor o_ref(1, 4);
    double r_arr[4] = {0.07, 0.1, 0.09, 0.04};
    for (size_t i = 0; i < 4; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < 4; ++j) {
            s += wkv_ref(j, i) * r_arr[j];
        }
        o_ref(0, i) = s;
    }
    // (Note: at T=1, since s_0=0 and w_t multiplies s_{t-1}=0, the per-step
    //  decay chain contributes 0; the output is exactly r_0·diag(u)·k_0^T·v_0.)

    double max_diff = 0.0;
    for (size_t i = 0; i < 4; ++i)
        max_diff = std::max(max_diff, std::abs(y(0, i) - o_ref(0, i)));
    CHECK(max_diff < 1e-12, "T=1 output matches hand-derived reference (max_diff < 1e-12)");
    std::cout << "  hand-derived o_ref[0] = " << o_ref(0, 0) << "  actual y(0,0) = " << y(0, 0) << "\n";
}

// ----------------------------------------------------------------------------
// Test 5: Input gradient FD vs analytical (T=1)
// ----------------------------------------------------------------------------
static void test_5_input_grad_t1() {
    std::cout << "Test 5: Input gradient FD vs analytical (T=1)\n";
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(1, 4);
    for (size_t i = 0; i < 4; ++i) x(0, i) = uni(rng);

    RWKV6TimeMix m(4, 1);
    // Randomize all parameters with non-degenerate scale
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    m.W_r.bias.fill(0.0); m.W_k.bias.fill(0.0); m.W_v.bias.fill(0.0);
    // u, lambda, mu_x, A, B non-degenerate
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = uni(rng) * 0.5 + 0.5;
    for (size_t i = 0; i < 4; ++i) {
        m.lambda_r(0, i) = uni(rng) * 0.3;
        m.lambda_k(0, i) = uni(rng) * 0.3;
        m.lambda_v(0, i) = uni(rng) * 0.3;
        m.lambda_g(0, i) = uni(rng) * 0.3;
        m.lambda_d(0, i) = uni(rng) * 0.3;
        m.lambda_x(0, i) = uni(rng) * 0.3;
        m.mu_x(0, i) = 0.5;
        m.mu_w(0, i) = 0.5;
    }
    // LoRA A/B: small non-zero
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_x}) {
        for (size_t i = 0; i < A->rows; ++i)
            for (size_t j = 0; j < A->cols; ++j)
                (*A)(i, j) = uni(rng) * 1e-2;
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_x}) {
        for (size_t i = 0; i < B->rows; ++i)
            for (size_t j = 0; j < B->cols; ++j)
                (*B)(i, j) = uni(rng) * 1e-2;
    }
    // lorad: rank = 2*rank, so A_d/B_d shapes differ
    for (size_t i = 0; i < m.A_d.rows; ++i)
        for (size_t j = 0; j < m.A_d.cols; ++j)
            m.A_d(i, j) = uni(rng) * 1e-2;
    for (size_t i = 0; i < m.B_d.rows; ++i)
        for (size_t j = 0; j < m.B_d.cols; ++j)
            m.B_d(i, j) = uni(rng) * 1e-2;

    Tensor grad_output(1, 4);
    for (size_t i = 0; i < 4; ++i) grad_output(0, i) = uni(rng);

    Tensor y = m.forward(x);
    Tensor grad_x = m.backward(grad_output, 0.0);

    // FD on each input element
    double max_rel_e = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        double fd = fd_input_elem(m, x, 0, i, grad_output);
        double ana = grad_x(0, i);
        double denom = std::max(std::abs(ana), std::abs(fd));
        if (denom < 1e-12) denom = 1e-12;
        double e = std::abs(ana - fd) / denom;
        if (e > max_rel_e) max_rel_e = e;
    }
    std::cout << "  T=1 input grad max_rel_err = " << max_rel_e << "\n";
    CHECK(max_rel_e < 1e-4, "T=1 input grad FD vs analytical < 1e-4");
}

// ----------------------------------------------------------------------------
// Test 6: Input gradient FD vs analytical (T=3) — exercises BPTT
// ----------------------------------------------------------------------------
static void test_6_input_grad_t3() {
    std::cout << "Test 6: Input gradient FD vs analytical (T=3)\n";
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    m.W_r.bias.fill(0.0); m.W_k.bias.fill(0.0); m.W_v.bias.fill(0.0);
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = uni(rng) * 0.5 + 0.5;
    for (size_t i = 0; i < 4; ++i) {
        m.lambda_r(0, i) = uni(rng) * 0.3;
        m.lambda_k(0, i) = uni(rng) * 0.3;
        m.lambda_v(0, i) = uni(rng) * 0.3;
        m.lambda_g(0, i) = uni(rng) * 0.3;
        m.lambda_d(0, i) = uni(rng) * 0.3;
        m.lambda_x(0, i) = uni(rng) * 0.3;
        m.mu_x(0, i) = 0.5;
        m.mu_w(0, i) = 0.5;
    }
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_x}) {
        for (size_t i = 0; i < A->rows; ++i)
            for (size_t j = 0; j < A->cols; ++j)
                (*A)(i, j) = uni(rng) * 1e-2;
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_x}) {
        for (size_t i = 0; i < B->rows; ++i)
            for (size_t j = 0; j < B->cols; ++j)
                (*B)(i, j) = uni(rng) * 1e-2;
    }
    for (size_t i = 0; i < m.A_d.rows; ++i)
        for (size_t j = 0; j < m.A_d.cols; ++j)
            m.A_d(i, j) = uni(rng) * 1e-2;
    for (size_t i = 0; i < m.B_d.rows; ++i)
        for (size_t j = 0; j < m.B_d.cols; ++j)
            m.B_d(i, j) = uni(rng) * 1e-2;

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y = m.forward(x);
    Tensor grad_x = m.backward(grad_output, 0.0);

    double max_rel_e = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_input_elem(m, x, i, j, grad_output);
            double ana = grad_x(i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(ana - fd) / denom;
            if (e > max_rel_e) max_rel_e = e;
        }
    }
    std::cout << "  T=3 input grad max_rel_err = " << max_rel_e << "\n";
    CHECK(max_rel_e < 1e-3, "T=3 input grad FD vs analytical < 1e-3");
}

// ----------------------------------------------------------------------------
// Test 7: Parameter gradients — finite vs analytical
// ----------------------------------------------------------------------------
static void test_7_parameter_grads() {
    std::cout << "Test 7: Parameter gradients FD vs analytical\n";
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    // Use small non-zero init for everything
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    m.W_r.bias.fill(0.0); m.W_k.bias.fill(0.0); m.W_v.bias.fill(0.0);
    m.W_g.bias.fill(0.0); m.W_o.bias.fill(0.0);
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = uni(rng) * 0.3 + 0.5;
    for (size_t i = 0; i < 4; ++i) {
        m.lambda_r(0, i) = uni(rng) * 0.3;
        m.lambda_k(0, i) = uni(rng) * 0.3;
        m.lambda_v(0, i) = uni(rng) * 0.3;
        m.lambda_g(0, i) = uni(rng) * 0.3;
        m.lambda_d(0, i) = uni(rng) * 0.3;
        m.lambda_x(0, i) = uni(rng) * 0.3;
        m.mu_x(0, i) = 0.5;
        m.mu_w(0, i) = 0.5;
    }
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_x}) {
        for (size_t i = 0; i < A->rows; ++i)
            for (size_t j = 0; j < A->cols; ++j)
                (*A)(i, j) = uni(rng) * 1e-2;
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_x}) {
        for (size_t i = 0; i < B->rows; ++i)
            for (size_t j = 0; j < B->cols; ++j)
                (*B)(i, j) = uni(rng) * 1e-2;
    }
    for (size_t i = 0; i < m.A_d.rows; ++i)
        for (size_t j = 0; j < m.A_d.cols; ++j)
            m.A_d(i, j) = uni(rng) * 1e-2;
    for (size_t i = 0; i < m.B_d.rows; ++i)
        for (size_t j = 0; j < m.B_d.cols; ++j)
            m.B_d(i, j) = uni(rng) * 1e-2;

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y = m.forward(x);
    m.backward(grad_output, 0.0);

    // FD on each parameter type — pick representative cells
    // W_r
    {
        Tensor fd(4, 4); for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j)
            fd(i, j) = fd_param(m.W_r.weights, m, x, grad_output, i, j);
        double e = max_rel_err(m.W_r.grad_weights, fd);
        std::cout << "  W_r grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "W_r grad rel_err < 1e-3");
    }
    // W_k
    {
        Tensor fd(4, 4); for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j)
            fd(i, j) = fd_param(m.W_k.weights, m, x, grad_output, i, j);
        double e = max_rel_err(m.W_k.grad_weights, fd);
        std::cout << "  W_k grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "W_k grad rel_err < 1e-3");
    }
    // W_v
    {
        Tensor fd(4, 4); for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j)
            fd(i, j) = fd_param(m.W_v.weights, m, x, grad_output, i, j);
        double e = max_rel_err(m.W_v.grad_weights, fd);
        std::cout << "  W_v grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "W_v grad rel_err < 1e-3");
    }
    // W_g
    {
        Tensor fd(4, 4); for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j)
            fd(i, j) = fd_param(m.W_g.weights, m, x, grad_output, i, j);
        double e = max_rel_err(m.W_g.grad_weights, fd);
        std::cout << "  W_g grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "W_g grad rel_err < 1e-3");
    }
    // W_o
    {
        Tensor fd(4, 4); for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j)
            fd(i, j) = fd_param(m.W_o.weights, m, x, grad_output, i, j);
        double e = max_rel_err(m.W_o.grad_weights, fd);
        std::cout << "  W_o grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "W_o grad rel_err < 1e-3");
    }
    // u
    {
        Tensor fd(1, 4); for (size_t i = 0; i < 4; ++i)
            fd(0, i) = fd_param(m.u, m, x, grad_output, 0, i);
        double e = max_rel_err(m.grad_u_, fd);
        std::cout << "  u grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "u grad rel_err < 1e-3");
    }
    // mu_x
    {
        Tensor fd(1, 4); for (size_t i = 0; i < 4; ++i)
            fd(0, i) = fd_param(m.mu_x, m, x, grad_output, 0, i);
        double e = max_rel_err(m.grad_mu_x_, fd);
        std::cout << "  mu_x grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "mu_x grad rel_err < 1e-3");
    }
    // mu_w
    {
        Tensor fd(1, 4); for (size_t i = 0; i < 4; ++i)
            fd(0, i) = fd_param(m.mu_w, m, x, grad_output, 0, i);
        double e = max_rel_err(m.grad_mu_w_, fd);
        std::cout << "  mu_w grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "mu_w grad rel_err < 1e-3");
    }
    // lambda_r
    {
        Tensor fd(1, 4); for (size_t i = 0; i < 4; ++i)
            fd(0, i) = fd_param(m.lambda_r, m, x, grad_output, 0, i);
        double e = max_rel_err(m.grad_lambda_r_, fd);
        std::cout << "  lambda_r grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "lambda_r grad rel_err < 1e-3");
    }
    // A_r (LoRA-r input projection, shape (4, R))
    {
        size_t R = m.A_r.cols;
        Tensor fd(m.A_r.rows, R);
        for (size_t i = 0; i < m.A_r.rows; ++i)
            for (size_t j = 0; j < R; ++j)
                fd(i, j) = fd_param(m.A_r, m, x, grad_output, i, j);
        double e = max_rel_err(m.grad_A_r_, fd);
        std::cout << "  A_r grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "A_r grad rel_err < 1e-3");
    }
    // B_r (LoRA-r output projection, shape (R, 4))
    {
        size_t R = m.B_r.rows;
        Tensor fd(R, m.B_r.cols);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < m.B_r.cols; ++j)
                fd(i, j) = fd_param(m.B_r, m, x, grad_output, i, j);
        double e = max_rel_err(m.grad_B_r_, fd);
        std::cout << "  B_r grad max_rel_err = " << e << "\n";
        CHECK(e < 1e-3, "B_r grad rel_err < 1e-3");
    }
}

// ----------------------------------------------------------------------------
// Test 8: zero_grad + update_weights
// ----------------------------------------------------------------------------
static void test_8_zero_grad_update() {
    std::cout << "Test 8: zero_grad clears all gradients; update_weights moves all params\n";
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y = m.forward(x);
    m.backward(grad_output, 0.0);

    // Capture all params
    std::vector<Tensor> before;
    for (auto* p : m.parameters()) before.push_back(p->clone());

    // Verify at least one grad is nonzero (otherwise test is vacuous)
    bool any_nonzero = false;
    for (auto* g : m.gradients()) {
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j)
                if (std::abs((*g)(i, j)) > 1e-12) any_nonzero = true;
    }
    CHECK(any_nonzero, "at least one gradient is nonzero after backward");

    // update_weights
    m.update_weights(0.01);
    // Verify at least one parameter moved
    bool any_moved = false;
    auto params = m.parameters();
    for (size_t k = 0; k < params.size(); ++k) {
        Tensor diff(params[k]->rows, params[k]->cols);
        for (size_t i = 0; i < params[k]->rows; ++i)
            for (size_t j = 0; j < params[k]->cols; ++j)
                diff(i, j) = (*params[k])(i, j) - before[k](i, j);
        double max_d = 0.0;
        for (size_t i = 0; i < diff.rows; ++i)
            for (size_t j = 0; j < diff.cols; ++j)
                max_d = std::max(max_d, std::abs(diff(i, j)));
        if (max_d > 1e-9) any_moved = true;
    }
    CHECK(any_moved, "update_weights moves at least one parameter");

    // zero_grad
    m.zero_grad();
    bool all_zero = true;
    for (auto* g : m.gradients()) {
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j)
                if (std::abs((*g)(i, j)) > 1e-15) all_zero = false;
    }
    CHECK(all_zero, "zero_grad clears all gradients");
}

// ----------------------------------------------------------------------------
// Test 9: Determinism — two consecutive calls produce identical output
// ----------------------------------------------------------------------------
static void test_9_determinism() {
    std::cout << "Test 9: Forward determinism (two consecutive calls bit-exact)\n";
    std::mt19937 rng(23);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    Tensor y1 = m.forward(x);
    Tensor y2 = m.forward(x);
    double max_d = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j)
            max_d = std::max(max_d, std::abs(y1(i, j) - y2(i, j)));
    CHECK(max_d == 0.0, "deterministic: y1 == y2 bit-exact");
}

// ----------------------------------------------------------------------------
// Test 10: Training reduces loss
// ----------------------------------------------------------------------------
static void test_10_training() {
    std::cout << "Test 10: Training reduces loss (50 SGD steps)\n";
    std::mt19937 rng(29);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);
    Tensor target(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            target(i, j) = uni(rng) * 0.5;

    RWKV6TimeMix m(8, 1);
    double lr = 0.01;
    double prev_loss = 1e9;
    bool decreasing = true;
    for (int step = 0; step < 50; ++step) {
        Tensor y = m.forward(x);
        double loss = 0.0;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 8; ++j)
                loss += (y(i, j) - target(i, j)) * (y(i, j) - target(i, j));
        loss /= 32.0;
        if (loss > prev_loss + 1e-6) decreasing = false;
        prev_loss = loss;

        Tensor grad_output(4, 8);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 8; ++j)
                grad_output(i, j) = 2.0 * (y(i, j) - target(i, j)) / 32.0;
        m.zero_grad();
        m.backward(grad_output, 0.0);
        m.update_weights(lr);
    }
    std::cout << "  final loss = " << prev_loss << "\n";
    CHECK(prev_loss < 0.1, "training reduced loss below 0.1");
    CHECK(decreasing, "loss monotonically non-increasing across 50 steps");
}

// ----------------------------------------------------------------------------
// Test 11: Mutation — zeroing LoRA-A_r weights changes output
// ----------------------------------------------------------------------------
static void test_11_mutation() {
    std::cout << "Test 11: Mutation tests (LoRA-A_r, mu_w, u all affect output)\n";
    std::mt19937 rng(31);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    // Init LoRA to non-zero so they actually contribute
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_d, &m.A_x}) {
        for (size_t i = 0; i < A->rows; ++i)
            for (size_t j = 0; j < A->cols; ++j)
                (*A)(i, j) = uni(rng) * 1e-1;
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_d, &m.B_x}) {
        for (size_t i = 0; i < B->rows; ++i)
            for (size_t j = 0; j < B->cols; ++j)
                (*B)(i, j) = uni(rng) * 1e-1;
    }
    for (size_t i = 0; i < 4; ++i) {
        m.lambda_r(0, i) = uni(rng) * 0.3;
        m.lambda_k(0, i) = uni(rng) * 0.3;
        m.lambda_v(0, i) = uni(rng) * 0.3;
        m.lambda_g(0, i) = uni(rng) * 0.3;
        m.lambda_d(0, i) = uni(rng) * 0.3;
        m.lambda_x(0, i) = uni(rng) * 0.3;
        m.mu_x(0, i) = 0.5;
        m.mu_w(0, i) = 0.5;
        m.u(0, i) = uni(rng) * 0.3 + 0.5;
    }

    Tensor y_base = m.forward(x);
    double max_y = 0.0;
    for (size_t i = 0; i < y_base.rows; ++i)
        for (size_t j = 0; j < y_base.cols; ++j)
            max_y = std::max(max_y, std::abs(y_base(i, j)));

    // Zero LoRA-A_r
    m.A_r.fill(0.0);
    Tensor y_after = m.forward(x);
    double diff_A = 0.0;
    for (size_t i = 0; i < y_after.rows; ++i)
        for (size_t j = 0; j < y_after.cols; ++j)
            diff_A = std::max(diff_A, std::abs(y_base(i, j) - y_after(i, j)));
    std::cout << "  [diag] A_r zeroed: diff_A=" << diff_A << " (threshold 1e-7)\n";
    CHECK(diff_A > 1e-7, "zeroing A_r changes output (LoRA path is exercised)");
    // Reset A_r to its initial random state (so we can test other mutations cleanly)
    m.A_r.fill(0.0);
    for (size_t i = 0; i < m.A_r.rows; ++i)
        for (size_t j = 0; j < m.A_r.cols; ++j)
            m.A_r(i, j) = uni(rng) * 1e-1;

    // Reset and verify mu_w matters (currently mu_w = 0.5)
    for (size_t i = 0; i < 4; ++i) m.mu_w(0, i) = 0.9;
    Tensor y_d = m.forward(x);
    double diff_d = 0.0;
    for (size_t i = 0; i < y_d.rows; ++i)
        for (size_t j = 0; j < y_d.cols; ++j)
            diff_d = std::max(diff_d, std::abs(y_base(i, j) - y_d(i, j)));
    std::cout << "  [diag] mu_w 0.5->0.9: diff_d=" << diff_d << " (threshold 1e-7)\n";
    CHECK(diff_d > 1e-7, "changing mu_w changes output (decay chain is exercised)");

    // Reset and verify u matters
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = 0.0;
    Tensor y_u = m.forward(x);
    double diff_u = 0.0;
    for (size_t i = 0; i < y_u.rows; ++i)
        for (size_t j = 0; j < y_u.cols; ++j)
            diff_u = std::max(diff_u, std::abs(y_base(i, j) - y_u(i, j)));
    CHECK(diff_u > 1e-4, "zeroing u changes output (bonus chain is exercised)");
}

// ----------------------------------------------------------------------------
// Test 12: T=6 input gradient FD (deeper BPTT, multi-head)
// ----------------------------------------------------------------------------
static void test_12_input_grad_t6_multihead() {
    std::cout << "Test 12: Input gradient FD vs analytical (T=6, multi-head)\n";
    std::mt19937 rng(37);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(6, 8);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);

    RWKV6TimeMix m(8, 2);  // head_dim = 4
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j) {
            m.W_r.weights(i, j) = uni(rng);
            m.W_k.weights(i, j) = uni(rng);
            m.W_v.weights(i, j) = uni(rng);
            m.W_g.weights(i, j) = uni(rng);
            m.W_o.weights(i, j) = uni(rng);
        }
    m.W_r.bias.fill(0.0); m.W_k.bias.fill(0.0); m.W_v.bias.fill(0.0);
    for (size_t i = 0; i < 8; ++i) m.u(0, i) = uni(rng) * 0.5 + 0.5;
    for (size_t i = 0; i < 8; ++i) {
        m.lambda_r(0, i) = uni(rng) * 0.3;
        m.lambda_k(0, i) = uni(rng) * 0.3;
        m.lambda_v(0, i) = uni(rng) * 0.3;
        m.lambda_g(0, i) = uni(rng) * 0.3;
        m.lambda_d(0, i) = uni(rng) * 0.3;
        m.lambda_x(0, i) = uni(rng) * 0.3;
        m.mu_x(0, i) = 0.5;
        m.mu_w(0, i) = 0.5;
    }
    for (auto* A : {&m.A_r, &m.A_k, &m.A_v, &m.A_g, &m.A_x}) {
        for (size_t i = 0; i < A->rows; ++i)
            for (size_t j = 0; j < A->cols; ++j)
                (*A)(i, j) = uni(rng) * 1e-2;
    }
    for (auto* B : {&m.B_r, &m.B_k, &m.B_v, &m.B_g, &m.B_x}) {
        for (size_t i = 0; i < B->rows; ++i)
            for (size_t j = 0; j < B->cols; ++j)
                (*B)(i, j) = uni(rng) * 1e-2;
    }
    for (size_t i = 0; i < m.A_d.rows; ++i)
        for (size_t j = 0; j < m.A_d.cols; ++j)
            m.A_d(i, j) = uni(rng) * 1e-2;
    for (size_t i = 0; i < m.B_d.rows; ++i)
        for (size_t j = 0; j < m.B_d.cols; ++j)
            m.B_d(i, j) = uni(rng) * 1e-2;

    Tensor grad_output(6, 8);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 8; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y = m.forward(x);
    Tensor grad_x = m.backward(grad_output, 0.0);

    double max_rel_e = 0.0;
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            double fd = fd_input_elem(m, x, i, j, grad_output);
            double ana = grad_x(i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(ana - fd) / denom;
            if (e > max_rel_e) max_rel_e = e;
        }
    }
    std::cout << "  T=6 multi-head input grad max_rel_err = " << max_rel_e << "\n";
    CHECK(max_rel_e < 1e-3, "T=6 multi-head input grad FD vs analytical < 1e-3");
}

// ============================================================================
int main() {
    test_1_constructor();
    test_2_init_zero_output();
    test_3_forward_shape();
    test_4_t1_reference();
    test_5_input_grad_t1();
    test_6_input_grad_t3();
    test_7_parameter_grads();
    test_8_zero_grad_update();
    test_9_determinism();
    test_10_training();
    test_11_mutation();
    test_12_input_grad_t6_multihead();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}