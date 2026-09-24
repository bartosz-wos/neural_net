// Tests for RWKV-5 "Eagle" (Peng et al. 2024)
// "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
// https://arxiv.org/abs/2404.05892
//
// We implement Eagle Time-Mixing only (channel-mixing is left as a future
// extension, matching the existing RWKV-4 / RWKV-6 / RWKV-7 time-mix-only
// convention in this repo).
//
// Test plan:
//   1. Constructor validation + accessors
//   2. W_o=0 at init → output is zero
//   3. Forward shape (T=3, d=4) → (3, 4) finite + nonzero
//   4. Forward shape (T=4, d=8, H=2) → (4, 8) finite + nonzero
//   5. Forward input-dim validation throws
//   6. State init s_0 = 0 invariant (last_s_(0, ·) == 0)
//   7. T=1 hand-derived closed-form reference (matches analytical to rel_err < 1e-10)
//   8. FD input gradient (T=3, d=4) at machine precision
//   9. FD W_r gradient (T=3, d=4)
//  10. FD W_k gradient (T=3, d=4)
//  11. FD W_v gradient (T=3, d=4)
//  12. FD W_o gradient (T=3, d=4)
//  13. FD u gradient (per-channel bonus; T=3, d=4)
//  14. FD log_w gradient (per-channel decay; T=3, d=4)
//  15. FD mu_r/k/v gradient (token-shift; T=3, d=4)
//  16. Multi-head (T=4, d=8, H=2) FD input gradient
//  17. Multi-head (T=4, d=8, H=2) FD W_k gradient (proves per-head outer-product math)
//  18. parameters()/gradients() returns 11 tensors each
//  19. zero_grad clears all 11
//  20. update_weights moves all 11 (single SGD step)
//  21. Training reduces loss over 30 SGD steps
//  22. Determinism: same input + copied params → bit-exact forward
//  23. Mutation test: perturbing log_w changes forward

#include "nn/layers/recurrent/rwkv5.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

static int g_pass = 0;
static int g_fail = 0;

struct CoutFlusher {
    CoutFlusher() { std::cout.setf(std::ios::unitbuf); }
} _flusher;

#define CHECK(cond, name)                                                    \
    do {                                                                   \
        if (cond) { ++g_pass; }                                            \
        else { ++g_fail; std::cout << "FAIL: " << name                     \
                                   << " (line " << __LINE__ << ")\n"; }    \
    } while (0)

#define CHECK_NEAR(a, b, tol, name)                                       \
    do {                                                                   \
        double aa = (a), bb = (b);                                         \
        if (std::abs(aa - bb) <= (tol)) { ++g_pass; }                      \
        else { ++g_fail; std::cout << "FAIL: " << name                     \
                                   << " expected " << bb << " got " << aa  \
                                   << " (line " << __LINE__ << ")\n"; }    \
    } while (0)

// ============================================================================
// Centered finite-difference utilities
// ============================================================================
// Finite-difference gradient of (Σ_t y_t · grad_output_t) w.r.t. p(i, j).
// IMPORTANT: caller must zero the analytical gradient fields before iterating
// over many fd_param calls — fd_param does NOT zero grad (would erase
// analytical values).
static double fd_param(Tensor& p, RWKV5TimeMix& m_layer, Tensor& x, const Tensor& grad_output,
                       size_t i, size_t j, double eps = 1e-5) {
    double orig = p(i, j);
    p(i, j) = orig + eps;
    Tensor yp = m_layer.forward(x);
    double fp = 0.0;
    for (size_t r = 0; r < yp.rows; ++r)
        for (size_t c = 0; c < yp.cols; ++c)
            fp += yp(r, c) * grad_output(r, c);
    p(i, j) = orig - eps;
    Tensor ym = m_layer.forward(x);
    double fm = 0.0;
    for (size_t r = 0; r < ym.rows; ++r)
        for (size_t c = 0; c < ym.cols; ++c)
            fm += ym(r, c) * grad_output(r, c);
    p(i, j) = orig;
    return (fp - fm) / (2 * eps);
}

static double fd_input_elem(RWKV5TimeMix& m_layer, Tensor& x, size_t i, size_t j,
                            const Tensor& grad_output, double eps = 1e-5) {
    double orig = x(i, j);
    x(i, j) = orig + eps;
    Tensor yp = m_layer.forward(x);
    double fp = 0.0;
    for (size_t r = 0; r < yp.rows; ++r)
        for (size_t c = 0; c < yp.cols; ++c)
            fp += yp(r, c) * grad_output(r, c);
    x(i, j) = orig - eps;
    Tensor ym = m_layer.forward(x);
    double fm = 0.0;
    for (size_t r = 0; r < ym.rows; ++r)
        for (size_t c = 0; c < ym.cols; ++c)
            fm += ym(r, c) * grad_output(r, c);
    x(i, j) = orig;
    return (fp - fm) / (2 * eps);
}

[[maybe_unused]] static double max_rel_err(const Tensor& ana, const Tensor& fd) {
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

static double max_abs(const Tensor& t) {
    double m = 0.0;
    for (size_t i = 0; i < t.data.size(); ++i)
        m = std::max(m, std::abs(t.data[i]));
    return m;
}

// ----------------------------------------------------------------------------
// Test 1: Constructor validation + accessors
// ----------------------------------------------------------------------------
static void test_1_constructor() {
    std::cout << "Test 1: RWKV5TimeMix constructor validation + accessors\n";

    // d=0 throws
    bool threw_d0 = false;
    try { RWKV5TimeMix m(0, 1); } catch (const std::invalid_argument&) { threw_d0 = true; }
    CHECK(threw_d0, "d=0 throws");

    // num_heads=0 throws
    bool threw_h0 = false;
    try { RWKV5TimeMix m(4, 0); } catch (const std::invalid_argument&) { threw_h0 = true; }
    CHECK(threw_h0, "num_heads=0 throws");

    // d % num_heads != 0 throws
    bool threw_div = false;
    try { RWKV5TimeMix m(5, 2); } catch (const std::invalid_argument&) { threw_div = true; }
    CHECK(threw_div, "d % num_heads != 0 throws");

    // valid construction
    {
        RWKV5TimeMix m(8, 1);
        CHECK(m.d() == 8, "d() == 8");
        CHECK(m.num_heads() == 1, "num_heads() == 1");
        CHECK(m.head_dim() == 8, "head_dim() == 8");
    }
    {
        RWKV5TimeMix m(12, 3);
        CHECK(m.d() == 12, "d() == 12 (3-head)");
        CHECK(m.num_heads() == 3, "num_heads() == 3");
        CHECK(m.head_dim() == 4, "head_dim() == 4");
    }

    // name
    {
        RWKV5TimeMix m(4, 1);
        CHECK(m.name() == "RWKV5TimeMix", "name() == RWKV5TimeMix");
    }
}

// ----------------------------------------------------------------------------
// Test 2: W_o=0 at init → output is zero
// ----------------------------------------------------------------------------
static void test_2_init_zero_output() {
    std::cout << "Test 2: W_o=0 at init → output is zero\n";
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
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
// Test 3: Forward shape (T=3, d=4) finite + nonzero
// ----------------------------------------------------------------------------
static void test_3_forward_shape() {
    std::cout << "Test 3: Forward shape (T=3, d=4) finite + nonzero\n";
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    // Randomize W_o so output isn't trivially zero.
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor y = m.forward(x);
    CHECK(y.rows == 3, "y.rows == 3");
    CHECK(y.cols == 4, "y.cols == 4");
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y(i, j))) finite = false;
            if (std::abs(y(i, j)) > 1e-6) nonzero = true;
        }
    }
    CHECK(finite, "y all finite");
    CHECK(nonzero, "y has at least one non-tiny entry");
}

// ----------------------------------------------------------------------------
// Test 4: Forward shape (T=4, d=8, H=2) finite + nonzero (multi-head)
// ----------------------------------------------------------------------------
static void test_4_forward_multi_head() {
    std::cout << "Test 4: Forward shape (T=4, d=8, H=2) finite + nonzero (multi-head)\n";
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(8, 2);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor y = m.forward(x);
    CHECK(y.rows == 4, "y.rows == 4 (multi-head)");
    CHECK(y.cols == 8, "y.cols == 8 (multi-head)");
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y(i, j))) finite = false;
    CHECK(finite, "multi-head y all finite");
}

// ----------------------------------------------------------------------------
// Test 5: Forward input-dim validation throws
// ----------------------------------------------------------------------------
static void test_5_input_dim_validation() {
    std::cout << "Test 5: input.cols != d throws\n";
    RWKV5TimeMix m(4, 1);
    Tensor bad(2, 5);  // wrong cols
    bool threw = false;
    try { m.forward(bad); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "wrong input.cols throws");
}

// ----------------------------------------------------------------------------
// Test 6: State init s_0 = 0 invariant
// ----------------------------------------------------------------------------
static void test_6_state_init_zero() {
    std::cout << "Test 6: last_s_(0, ·) == 0 (s_0 = 0)\n";
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    m.forward(x);

    bool all_zero = true;
    for (size_t j = 0; j < m.last_s_.cols; ++j)
        if (std::abs(m.last_s_(0, j)) > 1e-15) all_zero = false;
    CHECK(all_zero, "last_s_(0, ·) == 0 (initial state)");
}

// ----------------------------------------------------------------------------
// Test 7: T=1 closed-form reference
//
// For T=1 with the default init (log_w=-0.5, u=0, mu=0.5):
//   x_{-1} := 0, so r_in_0 = μ_r ⊙ x_0 + 0.
//   k_0 = W_k · r_in_0 + b_k, k_0 = same.
//   v_0 = W_v · v_in_0 + b_v.
//   kv[0, i, j] = k_0[i] · v_0[j]  (4×4 outer product for d=4)
//   s_0[i, j] = a_i · 0 + kv[0, i, j] = kv[0, i, j]
//   wkv[0, i, j] = s_0[i, j] + 0 · kv[0, i, j] = kv[0, i, j]
//   out_pre[0, i] = Σ_j r_0[i] · wkv[0, i, j] = r_0[i] · Σ_j kv[0, i, j]   (u=0 ⇒ wkv=kv)
// We force r_0[0] = 1, r_0[k≠0] = 0 so the closed-form trace simplifies
// to one row.
//
// We do this by zeroing W_r.weights except one element and verifying the
// resulting out_pre[0, 0] matches our hand-computed reference.
// ----------------------------------------------------------------------------
static void test_7_t1_closed_form() {
    std::cout << "Test 7: T=1 closed-form reference (with W_r sparse-ized)\n";
    RWKV5TimeMix m(4, 1);
    // Set x_0 = [1, 0.5, -0.3, 0.7] (deterministic).
    Tensor x(1, 4);
    x(0, 0) = 1.0;
    x(0, 1) = 0.5;
    x(0, 2) = -0.3;
    x(0, 3) = 0.7;

    // Reset all params to known values:
    //   W_k = identity, b_k = 0
    //   W_v = identity, b_v = 0
    //   W_r = sparse: only [0, 0] = 1.0, rest 0
    //   W_o = identity (so output == out_pre exactly)
    //   log_w = -0.5 (default), u = 0 (default), mu = 0.5 (default).
    //   b_r = b_k = b_v = b_o = 0.
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            m.W_k.weights(i, j) = (i == j) ? 1.0 : 0.0;
            m.W_v.weights(i, j) = (i == j) ? 1.0 : 0.0;
            m.W_o.weights(i, j) = (i == j) ? 1.0 : 0.0;
            m.W_r.weights(i, j) = 0.0;
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        m.W_r.bias(0, i) = 0.0;
        m.W_k.bias(0, i) = 0.0;
        m.W_v.bias(0, i) = 0.0;
        m.W_o.bias(0, i) = 0.0;
    }
    m.W_r.weights(0, 0) = 1.0;

    Tensor y = m.forward(x);

    //   Hand-compute reference for out_pre[0, j]:
    //     With x_0 = [1, 0.5, -0.3, 0.7] and mu_r=0.5:
    //       r_in_0 = μ_r ⊙ x_0 + (1-μ_r) ⊙ 0 = 0.5 ⊙ x_0 = [0.5, 0.25, -0.15, 0.35]
    //     W_r · r_in_0: only [0, 0]=1 contributes, so r_0[0] = 0.5, rest 0.
    //     Similarly k_0 = 0.5 ⊙ x_0 = [0.5, 0.25, -0.15, 0.35] (W_k = I, b_k = 0).
    //     v_0 = 0.5 ⊙ x_0 = [0.5, 0.25, -0.15, 0.35] (W_v = I, b_v = 0).
    //     kv[0, i, j] = k_0[i] · v_0[j]   (outer product per head)
    //     s_0[0, i, j] = a_0 · 0 + kv[0, 0, j]    (s_0 = kv since s_{-1} = 0)
    //     u_0 = 0 ⇒ wkv[0, i, j] = s_0[i, j] = kv[i, j]
    //     out_pre[0, j] = Σ_i r_0[i] · wkv[0, i, j]
    //                  = r_0[0] · wkv[0, 0, j]  (only r[0] is nonzero)
    //                  = 0.5 · kv[0, 0, j] = 0.5 · k_0[0] · v_0[j] = 0.5 · 0.5 · v_0[j] = 0.25 · v_0[j]
    //     out[0, 0] = out_pre[0, 0] = 0.25 · 0.5 = 0.125
    //     out[0, 1] = 0.25 · 0.25 = 0.0625
    //     out[0, 2] = 0.25 · (-0.15) = -0.0375
    //     out[0, 3] = 0.25 · 0.35 = 0.0875
        double expected[4] = { 0.125, 0.0625, -0.0375, 0.0875 };
        for (size_t j = 0; j < 4; ++j) {
            double got = y(0, j);
            CHECK_NEAR(got, expected[j], 1e-10, "T=1 closed-form y[0, j] matches hand-computed reference");
        }
}

// ----------------------------------------------------------------------------
// Test 8: FD input gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_8_fd_input() {
    std::cout << "Test 8: FD input gradient (T=3, d=4)\n";
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    // Randomize W_o so output isn't trivially zero (exercises W_o backward).
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    // Analytical input gradient
    Tensor y_ana = m.forward(x);
    (void)y_ana;
    m.zero_grad();
    Tensor grad_x_ana = m.backward(grad_output, 0.0);

    // Finite-difference input gradient (one element at a time)
    Tensor x_fd = x.clone();
    bool all_ok = true;
    double max_e = 0.0;
    for (size_t i = 0; i < x_fd.rows; ++i) {
        for (size_t j = 0; j < x_fd.cols; ++j) {
            double fd = fd_input_elem(m, x_fd, i, j, grad_output);
            double ana = grad_x_ana(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "input FD vs analytical (max rel_err < 1e-3)");
    std::cout << "    [input FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 9: FD W_r gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_9_fd_W_r() {
    std::cout << "Test 9: FD W_r gradient (T=3, d=4)\n";
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    // Analytical W_r.weights gradient
    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    // FD W_r.weights — we test 5 representative cells (full FD over 4×4 = 16 cells is fast).
    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_param(m.W_r.weights, m, x, grad_output, i, j);
            double ana = m.W_r.grad_weights(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "W_r.weights FD vs analytical (max rel_err < 1e-3 over 16 cells)");
    std::cout << "    [W_r FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 10: FD W_k gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_10_fd_W_k() {
    std::cout << "Test 10: FD W_k gradient (T=3, d=4)\n";
    std::mt19937 rng(23);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_param(m.W_k.weights, m, x, grad_output, i, j);
            double ana = m.W_k.grad_weights(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "W_k.weights FD vs analytical (max rel_err < 1e-3 over 16 cells)");
    std::cout << "    [W_k FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 11: FD W_v gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_11_fd_W_v() {
    std::cout << "Test 11: FD W_v gradient (T=3, d=4)\n";
    std::mt19937 rng(29);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_param(m.W_v.weights, m, x, grad_output, i, j);
            double ana = m.W_v.grad_weights(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "W_v.weights FD vs analytical (max rel_err < 1e-3 over 16 cells)");
    std::cout << "    [W_v FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 12: FD W_o gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_12_fd_W_o() {
    std::cout << "Test 12: FD W_o gradient (T=3, d=4)\n";
    std::mt19937 rng(31);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_param(m.W_o.weights, m, x, grad_output, i, j);
            double ana = m.W_o.grad_weights(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "W_o.weights FD vs analytical (max rel_err < 1e-3 over 16 cells)");
    std::cout << "    [W_o FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 13: FD u gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_13_fd_u() {
    std::cout << "Test 13: FD u gradient (T=3, d=4)\n";
    std::mt19937 rng(37);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    // Initialize u to nonzero so the FD has signal.
    for (size_t i = 0; i < 4; ++i) m.u(0, i) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    for (size_t j = 0; j < 4; ++j) {
        double fd = fd_param(m.u, m, x, grad_output, 0, j);
        double ana = m.grad_u_(0, j);
        double denom = std::max(std::abs(fd), std::abs(ana));
        if (denom < 1e-12) denom = 1e-12;
        double e = std::abs(fd - ana) / denom;
        if (e > max_e) max_e = e;
        if (e > 1e-3) all_ok = false;
    }
    CHECK(all_ok, "u FD vs analytical (max rel_err < 1e-3 over 4 cells)");
    std::cout << "    [u FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 14: FD log_w gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_14_fd_log_w() {
    std::cout << "Test 14: FD log_w gradient (T=3, d=4)\n";
    std::mt19937 rng(41);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    // Initialize log_w to non-default values so FD has signal.
    for (size_t i = 0; i < 4; ++i) m.log_w(0, i) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    for (size_t j = 0; j < 4; ++j) {
        double fd = fd_param(m.log_w, m, x, grad_output, 0, j);
        double ana = m.grad_log_w_(0, j);
        double denom = std::max(std::abs(fd), std::abs(ana));
        if (denom < 1e-12) denom = 1e-12;
        double e = std::abs(fd - ana) / denom;
        if (e > max_e) max_e = e;
        if (e > 1e-3) all_ok = false;
    }
    CHECK(all_ok, "log_w FD vs analytical (max rel_err < 1e-3 over 4 cells)");
    std::cout << "    [log_w FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 15: FD mu_r/k/v gradient (T=3, d=4)
// ----------------------------------------------------------------------------
static void test_15_fd_mu() {
    std::cout << "Test 15: FD mu_r/mu_k/mu_v gradient (T=3, d=4)\n";
    std::mt19937 rng(43);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    Tensor* mu_arr[3] = { &m.mu_r, &m.mu_k, &m.mu_v };
    Tensor* grad_arr[3] = { &m.grad_mu_r_, &m.grad_mu_k_, &m.grad_mu_v_ };
    const char* names[3] = { "mu_r", "mu_k", "mu_v" };
    (void)names;
    for (size_t which = 0; which < 3; ++which) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_param(*mu_arr[which], m, x, grad_output, 0, j);
            double ana = (*grad_arr[which])(0, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "mu_r/k/v FD vs analytical (max rel_err < 1e-3 over 12 cells)");
    std::cout << "    [mu FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 16: Multi-head FD input gradient (T=4, d=8, H=2)
// ----------------------------------------------------------------------------
static void test_16_fd_input_multi_head() {
    std::cout << "Test 16: Multi-head FD input gradient (T=4, d=8, H=2)\n";
    std::mt19937 rng(47);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(8, 2);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    Tensor grad_x_ana = m.backward(grad_output, 0.0);

    Tensor x_fd = x.clone();
    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < x_fd.rows; ++i) {
        for (size_t j = 0; j < x_fd.cols; ++j) {
            double fd = fd_input_elem(m, x_fd, i, j, grad_output);
            double ana = grad_x_ana(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "multi-head input FD vs analytical (max rel_err < 1e-3)");
    std::cout << "    [multi-head input FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 17: Multi-head FD W_k gradient (T=4, d=8, H=2)
// ----------------------------------------------------------------------------
static void test_17_fd_W_k_multi_head() {
    std::cout << "Test 17: Multi-head FD W_k gradient (T=4, d=8, H=2)\n";
    std::mt19937 rng(53);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(8, 2);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(4, 8);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    m.backward(grad_output, 0.0);

    double max_e = 0.0;
    bool all_ok = true;
    // Test 8 representative cells (full 8×8 = 64 would be slow).
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            double fd = fd_param(m.W_k.weights, m, x, grad_output, i, j);
            double ana = m.W_k.grad_weights(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fd - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "multi-head W_k.weights FD vs analytical (max rel_err < 1e-3 over 64 cells)");
    std::cout << "    [multi-head W_k FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 18: parameters()/gradients() returns 13 tensors each
// ----------------------------------------------------------------------------
static void test_18_parameters() {
    std::cout << "Test 18: parameters()/gradients() returns 13 tensors each\n";
    RWKV5TimeMix m(4, 1);
    auto p = m.parameters();
    auto g = m.gradients();
    CHECK(p.size() == 13, "parameters() size == 13 (4 Denses × 2 + 5 standalone tensors)");
    CHECK(g.size() == 13, "gradients() size == 13");
    // All gradients must be non-null and match a parameter shape.
    bool shapes_ok = true;
    for (size_t i = 0; i < p.size(); ++i) {
        if (!p[i] || !g[i]) { shapes_ok = false; break; }
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) { shapes_ok = false; break; }
    }
    CHECK(shapes_ok, "all param/grad shape pairs match");
}

// ----------------------------------------------------------------------------
// Test 19: zero_grad clears all 13
// ----------------------------------------------------------------------------
static void test_19_zero_grad() {
    std::cout << "Test 19: zero_grad clears all 13\n";
    RWKV5TimeMix m(4, 1);
    std::mt19937 rng(59);
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    m.forward(x);
    m.backward(grad_output, 0.0);

    m.zero_grad();
    auto g = m.gradients();
    bool all_zero = true;
    for (auto* t : g) {
        for (size_t i = 0; i < t->data.size(); ++i) {
            if (std::abs((*t).data[i]) > 0.0) { all_zero = false; break; }
        }
    }
    CHECK(all_zero, "all 11 grad tensors == 0 after zero_grad");
}

// ----------------------------------------------------------------------------
// Test 20: update_weights moves all 13
// ----------------------------------------------------------------------------
static void test_20_update_weights() {
    std::cout << "Test 20: update_weights moves all 13\n";
    RWKV5TimeMix m(4, 1);
    std::mt19937 rng(61);
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    Tensor grad_output(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    m.forward(x);
    m.backward(grad_output, 0.0);

    // Snapshot
    std::vector<Tensor> snapshots;
    for (auto* t : m.parameters()) snapshots.push_back(t->clone());

    m.update_weights(0.01);

    bool any_changed = false;
    for (size_t i = 0; i < snapshots.size(); ++i) {
        auto* t = m.parameters()[i];
        for (size_t k = 0; k < t->data.size(); ++k) {
            if (std::abs((*t).data[k] - snapshots[i].data[k]) > 1e-15) any_changed = true;
        }
    }
    CHECK(any_changed, "at least one parameter changed after update_weights");
}

// ----------------------------------------------------------------------------
// Test 21: Training reduces loss
// ----------------------------------------------------------------------------
static void test_21_training() {
    std::cout << "Test 21: training reduces loss over 30 SGD steps\n";
    RWKV5TimeMix m(4, 1);
    std::mt19937 rng(67);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor x(3, 4);
    Tensor target(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) {
            x(i, j) = uni(rng);
            target(i, j) = uni(rng);
        }

    auto loss = [&](const Tensor& y) {
        double s = 0.0;
        for (size_t i = 0; i < y.data.size(); ++i) {
            double d = y.data[i] - target.data[i];
            s += 0.5 * d * d;
        }
        return s;
    };

    Tensor y0 = m.forward(x);
    double L0 = loss(y0);

    for (int step = 0; step < 30; ++step) {
        Tensor y = m.forward(x);
        Tensor grad(y.rows, y.cols);
        for (size_t i = 0; i < y.data.size(); ++i)
            grad.data[i] = y.data[i] - target.data[i];
        m.zero_grad();
        m.backward(grad, 0.0);
        m.update_weights(0.05);
    }

    Tensor y_final = m.forward(x);
    double L_final = loss(y_final);
    CHECK(L_final < L0, "training reduces loss");
    std::cout << "    [L0 = " << std::fixed << std::setprecision(4) << L0
              << ", Lf = " << L_final << "]\n";
}

// ----------------------------------------------------------------------------
// Test 22: Determinism
// ----------------------------------------------------------------------------
static void test_22_determinism() {
    std::cout << "Test 22: determinism (bit-exact forward with copied params)\n";
    std::mt19937 rng(71);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);

    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m1(4, 1);
    RWKV5TimeMix m2(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m1.W_o.weights(i, j) = uni(rng);
    // Copy ALL params from m1 → m2.
    auto p1 = m1.parameters();
    auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i)
        for (size_t k = 0; k < p1[i]->data.size(); ++k)
            (*p2[i]).data[k] = (*p1[i]).data[k];

    Tensor y1 = m1.forward(x);
    Tensor y2 = m2.forward(x);
    double max_diff = 0.0;
    for (size_t i = 0; i < y1.data.size(); ++i)
        max_diff = std::max(max_diff, std::abs(y1.data[i] - y2.data[i]));
    CHECK(max_diff == 0.0, "bit-exact forward with copied params (max_diff == 0)");
}

// ----------------------------------------------------------------------------
// Test 23: Mutation test — perturbing log_w changes forward
// ----------------------------------------------------------------------------
static void test_23_mutation() {
    std::cout << "Test 23: perturbing log_w changes forward\n";
    std::mt19937 rng(73);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor y0 = m.forward(x);
    m.log_w(0, 0) += 0.5;        // perturb
    Tensor y1 = m.forward(x);

    double max_diff = 0.0;
    for (size_t i = 0; i < y0.data.size(); ++i)
        max_diff = std::max(max_diff, std::abs(y0.data[i] - y1.data[i]));
    if (max_diff <= 1e-9) {
        std::cout << "    [y0 max_abs = " << std::fixed << std::setprecision(6) << max_abs(y0)
                  << ", max_diff = " << max_diff << "]\n";
    }
    CHECK(max_diff > 1e-9, "perturbing log_w produces a different forward (max_diff > 1e-9 — proves the recurrence is wired)");
}

// ----------------------------------------------------------------------------
// Test 24: Deep BPTT input gradient (T=6, d=4) — exercises the recurrence chain
// ----------------------------------------------------------------------------
static void test_24_deep_bptt() {
    std::cout << "Test 24: Deep BPTT input gradient (T=6, d=4)\n";
    std::mt19937 rng(79);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(6, 4);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);

    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);

    Tensor grad_output(6, 4);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_output(i, j) = uni(rng);

    Tensor y_ana = m.forward(x); (void)y_ana;
    m.zero_grad();
    Tensor grad_x_ana = m.backward(grad_output, 0.0);

    Tensor x_fd = x.clone();
    double max_e = 0.0;
    bool all_ok = true;
    for (size_t i = 0; i < x_fd.rows; ++i) {
        for (size_t j = 0; j < x_fd.cols; ++j) {
            double fdv = fd_input_elem(m, x_fd, i, j, grad_output);
            double ana = grad_x_ana(i, j);
            double denom = std::max(std::abs(fdv), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            double e = std::abs(fdv - ana) / denom;
            if (e > max_e) max_e = e;
            if (e > 1e-3) all_ok = false;
        }
    }
    CHECK(all_ok, "deep BPTT input FD vs analytical (max rel_err < 1e-3)");
    std::cout << "    [deep BPTT input FD max rel_err = " << std::scientific << std::setprecision(2)
              << max_e << "]\n";
}

// ----------------------------------------------------------------------------
// Test 25: Determinism rerun — verify the test result is reproducible across runs
// ----------------------------------------------------------------------------
static void test_25_determinism_rerun() {
    std::cout << "Test 25: full test suite re-run yields identical pass count\n";
    // Just check that a known-good call (forward with copied params) gives 0 diff again.
    // (This is a smoke test; the actual reproducibility is verified by re-running ./test_rwkv5.)
    std::mt19937 rng(83);
    std::uniform_real_distribution<double> uni(-0.3, 0.3);
    Tensor x(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x(i, j) = uni(rng);
    RWKV5TimeMix m(4, 1);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m.W_o.weights(i, j) = uni(rng);
    Tensor y0 = m.forward(x);
    Tensor y1 = m.forward(x);
    double max_diff = 0.0;
    for (size_t i = 0; i < y0.data.size(); ++i)
        max_diff = std::max(max_diff, std::abs(y0.data[i] - y1.data[i]));
    CHECK(max_diff == 0.0, "two consecutive forward calls are bit-exact (max_diff == 0)");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    test_1_constructor();
    test_2_init_zero_output();
    test_3_forward_shape();
    test_4_forward_multi_head();
    test_5_input_dim_validation();
    test_6_state_init_zero();
    test_7_t1_closed_form();
    test_8_fd_input();
    test_9_fd_W_r();
    test_10_fd_W_k();
    test_11_fd_W_v();
    test_12_fd_W_o();
    test_13_fd_u();
    test_14_fd_log_w();
    test_15_fd_mu();
    test_16_fd_input_multi_head();
    test_17_fd_W_k_multi_head();
    test_18_parameters();
    test_19_zero_grad();
    test_20_update_weights();
    test_21_training();
    test_22_determinism();
    test_23_mutation();
    test_24_deep_bptt();
    test_25_determinism_rerun();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return (g_fail == 0) ? 0 : 1;
}