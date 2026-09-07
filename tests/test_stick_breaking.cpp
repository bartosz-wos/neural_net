// test_stick_breaking.cpp — Stick-Breaking Attention (Tan et al., ICLR 2025)
//   https://arxiv.org/abs/2410.17980
//
// Tests:
//   1.  Constructor validation (d_model=0, num_heads=0, non-divisible throw)
//   2.  Forward shape (N, d_model) -> (N, d_model), finite
//   3.  Row j=0 attends to nothing -> o_0 == r_head exactly (remainder on)
//   4.  Strict causality: perturbing v_i for i >= j leaves o_j bit-exact
//   5.  Stick-breaking invariant: sum_i A[i,j] + rem_j == 1, 0 <= A <= 1
//   6.  Recency bias: equal logits => A[i,j] decreasing in (j - i)
//   7.  Independent O(L^3) direct-product reference for A matches log-space
//   8.  Hand-derived N=2 forward reference
//   9.  Input gradient FD (N=4, H=2)
//  10.  W_q / W_k / W_v / W_o / remainder_ param gradient FD
//  11.  Deeper N=6 input gradient FD
//  12.  Multi-head equals two independent single-head layers
//  13.  use_remainder=false path: forward + input FD
//  14.  zero_grad clears all 9 gradients
//  15.  update_weights moves all 9 parameters
//  16.  StickBreakingBlock forward shape + input gradient FD
//  17.  StickBreakingModel training reduces loss over 40 SGD steps
//  18.  parameters()/gradients() contract (shape-matched, 9 tensors)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include "nn/layers/attention/stick_breaking.h"

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
// Deterministic pseudo-random fill (independent of global RNG state, so tests
// are reproducible regardless of what other layers consumed from the RNG).
static void fill_det(Tensor& t, unsigned seed, double scale) {
    unsigned s = seed;
    for (size_t i = 0; i < t.data.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        double u = ((s >> 8) & 0xFFFFFF) / double(0xFFFFFF);  // [0,1)
        t.data[i] = (2.0 * u - 1.0) * scale;
    }
}
static Tensor rand_tensor(size_t r, size_t c, unsigned seed, double scale) {
    Tensor t(r, c);
    fill_det(t, seed, scale);
    return t;
}
// Give the layer a non-degenerate, asymmetric parameter set. Uniform or
// symmetric init can hide row-vs-column transposition bugs (TDD skill).
static void randomize(StickBreakingAttention& a, unsigned base) {
    fill_det(a.W_q.weights, base + 1, 0.5);
    fill_det(a.W_k.weights, base + 2, 0.5);
    fill_det(a.W_v.weights, base + 3, 0.5);
    fill_det(a.W_o.weights, base + 4, 0.5);
    fill_det(a.W_q.bias,    base + 5, 0.2);
    fill_det(a.W_k.bias,    base + 6, 0.2);
    fill_det(a.W_v.bias,    base + 7, 0.2);
    fill_det(a.W_o.bias,    base + 8, 0.2);
    fill_det(a.remainder_,  base + 9, 0.4);
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor() {
    cout << endl << "--- Test 1: constructor validation ---" << endl;
    bool t_dm = false, t_nh = false, t_div = false, ok = true;
    try { StickBreakingAttention bad(0, 1); } catch (const exception&) { t_dm = true; }
    try { StickBreakingAttention bad(4, 0); } catch (const exception&) { t_nh = true; }
    try { StickBreakingAttention bad(4, 3); } catch (const exception&) { t_div = true; }
    try { StickBreakingAttention good(4, 2); } catch (const exception&) { ok = false; }
    check("d_model=0 throws", t_dm);
    check("num_heads=0 throws", t_nh);
    check("d_model not divisible by num_heads throws", t_div);
    check("valid (4,2) constructs", ok);

    StickBreakingAttention a(6, 3);
    check("head_dim == d_model/num_heads", a.head_dim() == 2);
    check("inv_temp == 1/sqrt(head_dim)",
          rel_err(a.inv_temp(), 1.0 / sqrt(2.0)) < 1e-15);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape + finiteness
// ---------------------------------------------------------------------------
static void test_forward_shape() {
    cout << endl << "--- Test 2: forward shape + finiteness ---" << endl;
    StickBreakingAttention a(4, 2);
    randomize(a, 100);
    Tensor x = rand_tensor(5, 4, 7, 1.0);
    Tensor y = a.forward(x);
    check("output shape (5,4)", y.rows == 5 && y.cols == 4);
    bool fin = true, nz = false;
    for (double v : y.data) { if (!isfinite(v)) fin = false; if (fabs(v) > 1e-12) nz = true; }
    check("output finite", fin);
    check("output nonzero", nz);
    check("last_A shape (H*N, N)", a.last_A().rows == 2 * 5 && a.last_A().cols == 5);
    check("last_rem shape (H, N)", a.last_rem().rows == 2 && a.last_rem().cols == 5);
}

// ---------------------------------------------------------------------------
// Test 3: query 0 attends to nothing -> o_0 == remainder (pre-W_o)
// ---------------------------------------------------------------------------
static void test_first_row_is_remainder() {
    cout << endl << "--- Test 3: row j=0 attends to nothing ---" << endl;
    StickBreakingAttention a(4, 2);
    randomize(a, 200);
    Tensor x = rand_tensor(4, 4, 11, 1.0);
    a.forward(x);
    // Attention row 0 must be all zero for every head.
    bool zero_row = true;
    for (size_t h = 0; h < a.num_heads(); ++h)
        for (size_t i = 0; i < 4; ++i)
            if (a.last_A()(h * 4 + 0, i) != 0.0) zero_row = false;
    check("A[:,j=0] all zero (nothing to attend to)", zero_row);
    bool rem_one = true;
    for (size_t h = 0; h < a.num_heads(); ++h)
        if (rel_err(a.last_rem()(h, 0), 1.0) > 1e-15) rem_one = false;
    check("rem_0 == 1 (whole stick left over)", rem_one);
}

// ---------------------------------------------------------------------------
// Test 4: strict causality — perturbing token i >= j cannot change o_j
// ---------------------------------------------------------------------------
static void test_causality() {
    cout << endl << "--- Test 4: strict causality signature ---" << endl;
    StickBreakingAttention a(4, 2);
    randomize(a, 300);
    const size_t N = 5;
    Tensor x = rand_tensor(N, 4, 13, 1.0);
    Tensor y1 = a.forward(x);
    // Perturb the LAST token: rows 0..N-2 of the output must be bit-exact.
    Tensor x2 = x.clone();
    for (size_t d = 0; d < 4; ++d) x2(N - 1, d) += 3.7;
    Tensor y2 = a.forward(x2);
    bool unchanged = true;
    for (size_t t = 0; t + 1 < N; ++t)
        for (size_t d = 0; d < 4; ++d)
            if (y1(t, d) != y2(t, d)) unchanged = false;
    check("perturbing token N-1 leaves rows 0..N-2 bit-exact", unchanged);
    bool last_changed = false;
    for (size_t d = 0; d < 4; ++d)
        if (y1(N - 1, d) != y2(N - 1, d)) last_changed = true;
    check("last row DID change (test is non-vacuous)", last_changed);
}

// ---------------------------------------------------------------------------
// Test 5: stick-breaking invariant sum_i A[i,j] + rem_j == 1
// ---------------------------------------------------------------------------
static void test_stick_invariant() {
    cout << endl << "--- Test 5: stick-breaking invariant ---" << endl;
    StickBreakingAttention a(6, 3);
    randomize(a, 400);
    const size_t N = 6;
    a.forward(rand_tensor(N, 6, 17, 1.5));
    double worst = 0.0;
    bool in_range = true;
    for (size_t h = 0; h < a.num_heads(); ++h)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double v = a.last_A()(h * N + j, i);
                if (v < 0.0 || v > 1.0) in_range = false;
                s += v;
            }
            worst = max(worst, fabs(s + a.last_rem()(h, j) - 1.0));
            // Sum of allocated weights must never exceed the whole stick.
            if (s > 1.0 + 1e-12) in_range = false;
        }
    cout << "  max |sum_i A[i,j] + rem_j - 1| = " << scientific << worst << endl;
    check("stick fully accounted for (sum A + rem == 1)", worst < 1e-12);
    check("0 <= A[i,j] <= 1 and sum A <= 1", in_range);
}

// ---------------------------------------------------------------------------
// Test 6: recency bias — with equal logits, A decreases with distance
// ---------------------------------------------------------------------------
static void test_recency_bias() {
    cout << endl << "--- Test 6: recency bias with equal logits ---" << endl;
    // Force all q,k identical so every z[i,j] is the same value; then
    // A[i,j] = beta * (1-beta)^(j-1-i) must be strictly decreasing in (j-i).
    StickBreakingAttention a(2, 1, /*use_remainder=*/false);
    a.W_q.weights.fill(0.0); a.W_k.weights.fill(0.0); a.W_v.weights.fill(0.0);
    a.W_o.weights.fill(0.0);
    a.W_q.bias.fill(0.0); a.W_k.bias.fill(0.0);
    a.W_v.bias.fill(0.0); a.W_o.bias.fill(0.0);
    // Constant q = k = (1, 0) via bias only -> every z[i,j] = 1/sqrt(2).
    a.W_q.bias(0, 0) = 1.0;
    a.W_k.bias(0, 0) = 1.0;
    const size_t N = 5;
    Tensor x(N, 2); x.fill(0.0);
    a.forward(x);
    const size_t j = N - 1;
    bool decreasing = true;
    for (size_t i = 0; i + 1 < j; ++i)
        if (!(a.last_A()(j, i) < a.last_A()(j, i + 1))) decreasing = false;
    cout << "  A[i, j=4] = ";
    for (size_t i = 0; i < j; ++i) cout << fixed << setprecision(5) << a.last_A()(j, i) << " ";
    cout << endl;
    check("A[i,j] strictly increasing in i (i.e. recency-biased)", decreasing);
    // Geometric-decay closed form: A[i,j] = beta * (1-beta)^(j-1-i)
    const double beta = 1.0 / (1.0 + exp(-1.0 / sqrt(2.0)));
    double worst = 0.0;
    for (size_t i = 0; i < j; ++i) {
        double expect = beta * pow(1.0 - beta, double(j - 1 - i));
        worst = max(worst, rel_err(a.last_A()(j, i), expect));
    }
    cout << "  worst rel_err vs beta*(1-beta)^(j-1-i) = " << scientific << worst << endl;
    check("matches geometric closed form", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 7: independent O(L^3) direct-product reference for A
// ---------------------------------------------------------------------------
static void test_direct_product_reference() {
    cout << endl << "--- Test 7: direct-product reference for A ---" << endl;
    // Production uses the log-space form (Eq. 13). Here we recompute A with
    // the LITERAL product form of Eq. 1 to catch log-space algebra errors.
    StickBreakingAttention a(4, 2);
    randomize(a, 500);
    const size_t N = 5, H = 2, dh = 2;
    Tensor x = rand_tensor(N, 4, 19, 1.0);
    a.forward(x);
    Tensor Q = a.W_q.forward(x), K = a.W_k.forward(x);
    double worst = 0.0;
    for (size_t h = 0; h < H; ++h) {
        size_t off = h * dh;
        for (size_t j = 0; j < N; ++j) {
            // beta[i] = sigmoid(inv_temp * q_j . k_i)
            vector<double> beta(j, 0.0);
            for (size_t i = 0; i < j; ++i) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d) dot += Q(j, off + d) * K(i, off + d);
                beta[i] = 1.0 / (1.0 + exp(-dot * a.inv_temp()));
            }
            for (size_t i = 0; i < j; ++i) {
                double prod = beta[i];
                for (size_t k = i + 1; k < j; ++k) prod *= (1.0 - beta[k]);
                worst = max(worst, rel_err(a.last_A()(h * N + j, i), prod));
            }
        }
    }
    cout << "  worst rel_err (log-space vs direct product) = " << scientific << worst << endl;
    check("log-space A matches direct product form", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 8: hand-derived N=2 forward
// ---------------------------------------------------------------------------
static void test_hand_derived() {
    cout << endl << "--- Test 8: hand-derived N=2 reference ---" << endl;
    StickBreakingAttention a(2, 1, /*use_remainder=*/true);
    // Identity projections so q_j = x_j, k_i = x_i, v_i = x_i.
    a.W_q.weights.fill(0.0); a.W_k.weights.fill(0.0); a.W_v.weights.fill(0.0);
    a.W_o.weights.fill(0.0);
    for (size_t i = 0; i < 2; ++i) {
        a.W_q.weights(i, i) = 1.0;
        a.W_k.weights(i, i) = 1.0;
        a.W_v.weights(i, i) = 1.0;
        a.W_o.weights(i, i) = 1.0;
    }
    a.W_q.bias.fill(0.0); a.W_k.bias.fill(0.0);
    a.W_v.bias.fill(0.0); a.W_o.bias.fill(0.0);
    a.remainder_(0, 0) = 0.5; a.remainder_(0, 1) = -0.25;

    Tensor x(2, 2);
    x(0, 0) = 0.3; x(0, 1) = -0.7;
    x(1, 0) = 1.1; x(1, 1) =  0.4;
    Tensor y = a.forward(x);

    // Expected, computed from the formula (not a hand-typed decimal):
    const double inv_t = 1.0 / sqrt(2.0);
    const double z = (x(1, 0) * x(0, 0) + x(1, 1) * x(0, 1)) * inv_t;
    const double beta = 1.0 / (1.0 + exp(-z));
    const double rem = 1.0 - beta;
    // Row 0: nothing to attend to -> o_0 = 1 * r
    double e00 = a.remainder_(0, 0), e01 = a.remainder_(0, 1);
    // Row 1: o_1 = beta * v_0 + rem * r
    double e10 = beta * x(0, 0) + rem * a.remainder_(0, 0);
    double e11 = beta * x(0, 1) + rem * a.remainder_(0, 1);
    double worst = max(max(rel_err(y(0, 0), e00), rel_err(y(0, 1), e01)),
                       max(rel_err(y(1, 0), e10), rel_err(y(1, 1), e11)));
    cout << "  beta = " << fixed << setprecision(10) << beta
         << ", worst rel_err = " << scientific << worst << endl;
    check("hand-derived N=2 forward matches", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// FD helpers
// ---------------------------------------------------------------------------
// Central-difference check of the input gradient.
static double fd_input_check(StickBreakingAttention& a, Tensor x,
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
// Central-difference check of a parameter tensor. `param` and `grad` are
// matched by POINTER, never by shape — shape matching silently reads the wrong
// gradient when two parameters share a shape.
static double fd_param_check(StickBreakingAttention& a, Tensor* param,
                             Tensor* grad, const Tensor& x, const Tensor& tgt,
                             double eps = 1e-6) {
    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    Tensor ana = grad->clone();
    double worst = 0.0;
    for (size_t i = 0; i < param->data.size(); ++i) {
        double orig = param->data[i];
        param->data[i] = orig + eps;
        double lp = l2_loss(a.forward(x), tgt);
        param->data[i] = orig - eps;
        double lm = l2_loss(a.forward(x), tgt);
        param->data[i] = orig;
        worst = max(worst, rel_err(ana.data[i], (lp - lm) / (2 * eps)));
    }
    return worst;
}

// ---------------------------------------------------------------------------
// Test 9 / 10 / 11: gradient checks
// ---------------------------------------------------------------------------
static void test_gradients() {
    cout << endl << "--- Test 9-11: gradient FD checks ---" << endl;
    StickBreakingAttention a(4, 2);
    randomize(a, 600);
    Tensor x = rand_tensor(4, 4, 23, 1.0);
    Tensor tgt = rand_tensor(4, 4, 29, 1.0);

    double e_in = fd_input_check(a, x, tgt);
    cout << "  input grad rel_err (N=4, H=2) = " << scientific << e_in << endl;
    check("input gradient FD", e_in < 1e-4);

    struct { const char* n; Tensor* p; Tensor* g; } ps[] = {
        {"W_q.weights", &a.W_q.weights, &a.grad_W_q},
        {"W_k.weights", &a.W_k.weights, &a.grad_W_k},
        {"W_v.weights", &a.W_v.weights, &a.grad_W_v},
        {"W_o.weights", &a.W_o.weights, &a.grad_W_o},
        {"W_q.bias",    &a.W_q.bias,    &a.W_q.grad_bias},
        {"W_k.bias",    &a.W_k.bias,    &a.W_k.grad_bias},
        {"W_v.bias",    &a.W_v.bias,    &a.W_v.grad_bias},
        {"W_o.bias",    &a.W_o.bias,    &a.W_o.grad_bias},
        {"remainder_",  &a.remainder_,  &a.grad_remainder_},
    };
    for (auto& e : ps) {
        double err = fd_param_check(a, e.p, e.g, x, tgt);
        cout << "  " << setw(12) << e.n << " grad rel_err = " << scientific << err << endl;
        check(string(e.n) + " gradient FD", err < 1e-4);
    }

    // Deeper sequence — exercises longer stick-breaking chains.
    StickBreakingAttention b(4, 2);
    randomize(b, 700);
    Tensor x6 = rand_tensor(6, 4, 31, 1.0);
    Tensor t6 = rand_tensor(6, 4, 37, 1.0);
    double e6 = fd_input_check(b, x6, t6);
    cout << "  input grad rel_err (N=6) = " << scientific << e6 << endl;
    check("deeper N=6 input gradient FD", e6 < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 12: multi-head == two independent single-head layers
// ---------------------------------------------------------------------------
static void test_multihead_split() {
    cout << endl << "--- Test 12: multi-head equals independent heads ---" << endl;
    const size_t N = 4, dm = 4, dh = 2;
    StickBreakingAttention mh(dm, 2, /*use_remainder=*/true);
    randomize(mh, 800);
    // Make W_o the identity so the pre-projection per-head outputs are visible.
    mh.W_o.weights.fill(0.0);
    for (size_t i = 0; i < dm; ++i) mh.W_o.weights(i, i) = 1.0;
    mh.W_o.bias.fill(0.0);

    Tensor x = rand_tensor(N, dm, 41, 1.0);
    Tensor y = mh.forward(x);

    // Each head must use ONLY its own dh-wide column slice of Q/K/V. Recompute
    // every head independently from the projections and compare — this is what
    // catches per-head offset / stride bugs (a head reading a neighbour's
    // columns would still produce finite, plausible output).
    double worst = 0.0;
    Tensor Q = mh.W_q.forward(x), K = mh.W_k.forward(x), V = mh.W_v.forward(x);
    for (size_t h = 0; h < 2; ++h) {
        for (size_t j = 0; j < N; ++j) {
            vector<double> beta(j, 0.0);
            for (size_t i = 0; i < j; ++i) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += Q(j, h * dh + d) * K(i, h * dh + d);
                beta[i] = 1.0 / (1.0 + exp(-dot * mh.inv_temp()));
            }
            double asum = 0.0;
            vector<double> A(j, 0.0);
            for (size_t i = 0; i < j; ++i) {
                double p = beta[i];
                for (size_t k = i + 1; k < j; ++k) p *= (1.0 - beta[k]);
                A[i] = p; asum += p;
            }
            for (size_t d = 0; d < dh; ++d) {
                double acc = (1.0 - asum) * mh.remainder_(h, d);
                for (size_t i = 0; i < j; ++i) acc += A[i] * V(i, h * dh + d);
                worst = max(worst, rel_err(y(j, h * dh + d), acc));
            }
        }
    }
    cout << "  worst rel_err (per-head independent recompute) = "
         << scientific << worst << endl;
    check("heads are computed independently on their own slices", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 13: use_remainder=false
// ---------------------------------------------------------------------------
static void test_no_remainder() {
    cout << endl << "--- Test 13: use_remainder=false ---" << endl;
    StickBreakingAttention a(4, 2, /*use_remainder=*/false);
    randomize(a, 900);
    Tensor x = rand_tensor(4, 4, 43, 1.0);
    Tensor tgt = rand_tensor(4, 4, 47, 1.0);
    Tensor y = a.forward(x);
    // Row 0 attends to nothing and there is no sink -> pre-W_o output is 0,
    // so y row 0 == W_o bias exactly.
    bool row0_is_bias = true;
    for (size_t d = 0; d < 4; ++d)
        if (rel_err(y(0, d), a.W_o.bias(0, d)) > 1e-14) row0_is_bias = false;
    check("no-remainder: row 0 == W_o bias", row0_is_bias);
    double e = fd_input_check(a, x, tgt);
    cout << "  input grad rel_err (no remainder) = " << scientific << e << endl;
    check("no-remainder input gradient FD", e < 1e-4);
    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    bool rem_grad_zero = true;
    for (double v : a.grad_remainder_.data) if (v != 0.0) rem_grad_zero = false;
    check("no-remainder: grad_remainder_ stays zero", rem_grad_zero);
}

// ---------------------------------------------------------------------------
// Test 14 / 15 / 18: bookkeeping
// ---------------------------------------------------------------------------
static void test_bookkeeping() {
    cout << endl << "--- Test 14/15/18: zero_grad, update_weights, contract ---" << endl;
    StickBreakingAttention a(4, 2);
    randomize(a, 1000);
    Tensor x = rand_tensor(4, 4, 53, 1.0);
    Tensor tgt = rand_tensor(4, 4, 59, 1.0);

    auto ps = a.parameters();
    auto gs = a.gradients();
    check("parameters() returns 9 tensors", ps.size() == 9);
    check("gradients() returns 9 tensors", gs.size() == 9);
    bool shapes_ok = ps.size() == gs.size();
    for (size_t i = 0; i < min(ps.size(), gs.size()); ++i)
        if (ps[i]->rows != gs[i]->rows || ps[i]->cols != gs[i]->cols) shapes_ok = false;
    check("parameter/gradient shapes match pairwise", shapes_ok);

    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    size_t nonzero = 0;
    for (Tensor* g : gs) {
        for (double v : g->data) if (fabs(v) > 1e-14) { ++nonzero; break; }
    }
    cout << "  gradients with a nonzero entry: " << nonzero << "/9" << endl;
    check("all 9 gradients receive signal", nonzero == 9);

    a.zero_grad();
    bool all_zero = true;
    for (Tensor* g : gs) for (double v : g->data) if (v != 0.0) all_zero = false;
    check("zero_grad clears all 9 gradients", all_zero);

    // update_weights must move every parameter.
    a.zero_grad();
    out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    vector<Tensor> before;
    for (Tensor* p : ps) before.push_back(p->clone());
    a.update_weights(0.1);
    size_t moved = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
        bool diff = false;
        for (size_t k = 0; k < ps[i]->data.size(); ++k)
            if (ps[i]->data[k] != before[i].data[k]) diff = true;
        if (diff) ++moved;
    }
    cout << "  parameters moved by update_weights: " << moved << "/9" << endl;
    check("update_weights moves all 9 parameters", moved == 9);
}

// ---------------------------------------------------------------------------
// Test 16: Block
// ---------------------------------------------------------------------------
static void test_block() {
    cout << endl << "--- Test 16: StickBreakingBlock ---" << endl;
    StickBreakingBlock blk(4, 2, 8);
    randomize(blk.attn, 1100);
    fill_det(blk.ffn_fc1_.weights, 1200, 0.4);
    fill_det(blk.ffn_fc2_.weights, 1300, 0.4);
    fill_det(blk.ffn_fc1_.bias, 1400, 0.1);
    fill_det(blk.ffn_fc2_.bias, 1500, 0.1);

    Tensor x = rand_tensor(4, 4, 61, 1.0);
    Tensor tgt = rand_tensor(4, 4, 67, 1.0);
    Tensor y = blk.forward(x);
    check("block output shape (4,4)", y.rows == 4 && y.cols == 4);
    bool fin = true;
    for (double v : y.data) if (!isfinite(v)) fin = false;
    check("block output finite", fin);

    // Input-gradient FD through the whole block (catches the ln2 residual trap).
    blk.zero_grad();
    Tensor out = blk.forward(x);
    Tensor gi = blk.backward(l2_grad(out, tgt), 0.0);
    double eps = 1e-6, worst = 0.0;
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c) {
            Tensor xp = x.clone(); xp(r, c) += eps;
            Tensor xm = x.clone(); xm(r, c) -= eps;
            double num = (l2_loss(blk.forward(xp), tgt) -
                          l2_loss(blk.forward(xm), tgt)) / (2 * eps);
            worst = max(worst, rel_err(gi(r, c), num));
        }
    cout << "  block input grad rel_err = " << scientific << worst << endl;
    check("block input gradient FD", worst < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 17: Model training reduces loss
// ---------------------------------------------------------------------------
static void test_model_training() {
    cout << endl << "--- Test 17: StickBreakingModel training ---" << endl;
    bool threw = false;
    try { StickBreakingModel bad(3, 4, 2, 0); } catch (const exception&) { threw = true; }
    check("num_blocks=0 throws", threw);

    StickBreakingModel m(3, 4, 2, 2, 2);
    Tensor x = rand_tensor(5, 3, 71, 1.0);
    Tensor tgt = rand_tensor(5, 2, 73, 0.5);

    Tensor y0 = m.forward(x);
    check("model output shape (5,2)", y0.rows == 5 && y0.cols == 2);
    double l0 = l2_loss(y0, tgt), lf = l0;
    for (int step = 0; step < 40; ++step) {
        m.zero_grad();
        Tensor out = m.forward(x);
        lf = l2_loss(out, tgt);
        m.backward(l2_grad(out, tgt), 0.0);
        m.update_weights(0.02);
    }
    cout << "  loss " << fixed << setprecision(6) << l0 << " -> " << lf
         << "  (" << setprecision(1) << (100.0 * (l0 - lf) / l0) << "% reduction)" << endl;
    check("training reduces loss", lf < l0 * 0.9);
    bool fin = true;
    for (double v : m.forward(x).data) if (!isfinite(v)) fin = false;
    check("model output still finite after training", fin);
}

int main() {
    cout << "=== Stick-Breaking Attention Tests (Tan et al., ICLR 2025) ===" << endl;
    try {
        test_constructor();
        test_forward_shape();
        test_first_row_is_remainder();
        test_causality();
        test_stick_invariant();
        test_recency_bias();
        test_direct_product_reference();
        test_hand_derived();
        test_gradients();
        test_multihead_split();
        test_no_remainder();
        test_bookkeeping();
        test_block();
        test_model_training();
    } catch (const exception& e) {
        cout << "  [EXCEPTION] " << e.what() << endl;
        ++failed;
    }
    cout << endl << "=== Summary: " << passed << " passed, " << failed
         << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
