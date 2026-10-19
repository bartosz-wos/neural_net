// =============================================================================
// tests/test_hutchinson.cpp -- Hutchinson stochastic Hessian-trace estimation
// and forward-over-reverse Hessian-vector products.
//
// THE GAP THIS FILLS WAS MEASURED, NOT ASSUMED. The repo's only second-order
// optimizer, `Sophia`, documents in its OWN header that it wants
//
//     "typically with a Hutchinson trace estimate"
//
// (`optimizers/sophia.h:25`) and exposes
// `set_hessian_estimates(layer_ptr, h_diag)` for them -- but
// `grep -rn 'set_hessian_estimates' --include=*.cpp --include=*.h .` returns
// ONLY sophia.{h,cpp} and tests/test_sophia.cpp, which feeds it a hand-made
// tensor. Every real run therefore falls back to the empirical Fisher
// g (.) g and the higher-order path is dead code in practice. A file that
// documents a limitation IS a specification.
//
// CITATION. Two sources, both PRE-arXiv, so there is no arXiv ID to verify
// and none may be invented (this queue has produced NINE wrong arXiv IDs):
//   Hutchinson, M.F., "A stochastic estimator of the trace of the influence
//     matrix for Laplacian smoothing splines", Comm. Statist. Simulation and
//     Computation 18(3):1059-1076 (1989/90).
//   Pearlmutter, B.A., "Gradient calculations for dynamic recurrent neural
//     networks: a survey", IEEE Trans. Neural Networks 6(5):1212-1228 (1995).
// The Pearlmutter year is VERY widely cited as 1994 -- including by PyTorch's
// own hvp docstring -- but dblp's authoritative record is journals/tnn/
// Pearlmutter95, i.e. 1995. Recorded, not silently reconciled.
//
// NO FINITE-DIFFERENCE TEST FOR THE ESTIMATOR, AND THERE CANNOT BE ONE: the
// forward is stochastic, so a centred difference compares two UNCORRELATED
// noise draws (systematic-debugging 5c-bis). Verification is by VALUE against
// a self-checking float64 numpy oracle, plus mutation testing.
//
// THE ORACLE CAUGHT ITS OWN BUG BEFORE ANY C++ EXISTED. Its first draft's
// variance check drew TWO independent probes, measuring E[z1' H z2] with
// z1 != z2 -- whose expectation is ZERO for symmetric H, not tr(H). The
// fingerprint was means hovering around 0.0 while the true trace was 0.442:
// large and shape-independent, which says "my estimator is wrong", not "the
// theory is wrong" (systematic-debugging 4c). The estimator is z' H z with ONE z.
// =============================================================================

#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void check_near(double got, double want, double tol, const std::string& name) {
    bool ok = std::fabs(got - want) <= tol;
    if (ok) ++g_pass; else ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g diff=%.3e tol=%.1e\n",
           ok ? "PASS" : "FAIL", name.c_str(), got, want, std::fabs(got - want), tol);
}

static void check_eq(double got, double want, const std::string& name) {
    bool ok = (got == want);
    if (ok) ++g_pass; else ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name.c_str(), got, want);
}

// ---------------------------------------------------------------------------
// A QUADRATIC FIXTURE, built so the trace is known IN CLOSED FORM.
//   f(x) = 0.5 x'Hx + c'x   =>   grad = Hx + c,   Hessian = H (exactly).
// H is built symmetric but deliberately NOT diagonal -- a diagonal H hides
// every off-diagonal contribution, which is the whole content of z'Hz.
// ---------------------------------------------------------------------------
struct Quad {
    Tensor H, c, x;
};

static Quad make_quad(size_t n, unsigned seed) {
    Quad q;
    q.H = Tensor(n, n);
    q.c = Tensor(1, n);
    q.x = Tensor(1, n);
    // Deterministic pseudo-random symmetric H (no dependency on rand()).
    unsigned s = seed ? seed : 1u;
    auto next = [&s]() { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.0 - 0.5; };
    std::vector<double> upper(n * n, 0.0);
    for (size_t i = 0; i < n * n; ++i) upper[i] = next();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            q.H(i, j) = (i <= j) ? upper[i * n + j] : upper[j * n + i];
    for (size_t j = 0; j < n; ++j) { q.c(0, j) = next(); q.x(0, j) = next(); }
    return q;
}

static Tensor quad_grad_t(const Quad& q, const Tensor& x) {
    Tensor g(1, q.H.cols);
    for (size_t j = 0; j < q.H.cols; ++j) {
        double acc = q.c(0, j);
        for (size_t i = 0; i < q.H.rows; ++i) acc += q.H(i, j) * x(0, i);
        g(0, j) = acc;
    }
    return g;
}

static double quad_trace(const Quad& q) {
    double t = 0.0;
    for (size_t i = 0; i < q.H.rows; ++i) t += q.H(i, i);
    return t;
}

// The gradient oracle the estimator consumes: a std::function<Tensor(const
// Tensor&)> returning grad = Hx + c. Built explicitly (rather than relying on
// implicit lambda->std::function conversion) so the type under test is
// visible here: the production `HessianGradFn`.
static HessianGradFn grad_oracle(const Quad& q) {
    return HessianGradFn([q](const Tensor& x) { return quad_grad_t(q, x); });
}

// =============================================================================
// D1 / D2 -- unbiasedness. E[z'Hz] == tr(H) for BOTH probe laws.
//
// The mean over many independent runs must land on the exact trace to within a
// few standard errors. A biased estimator (e.g. one that drops the 1/probes
// normalisation) fails this by a clean factor, not by a little.
// =============================================================================
static void test_unbiasedness() {
    const size_t n = 6;
    Quad q = make_quad(n, 12345);

    // TEST BUG FOUND AND FIXED HERE, 2026-10-19: this fixture's raw trace is
    // -0.0169, i.e. ACCIDENTALLY NEAR ZERO, while the estimator's spread on it
    // is ~2.5. Any statement of the form "the bias is small compared to the
    // trace" is therefore meaningless here -- it is the metric-at-the-floor
    // trap. Adding a diagonal shift makes tr(H) = O(n) so "unbiased" becomes a
    // claim about a RELATIVE error, which is the claim that actually matters
    // and the one a biased estimator (e.g. one dropping the 1/probes factor,
    // which would land at ~2.5 instead of ~6) fails immediately.
    for (size_t i = 0; i < n; ++i) q.H(i, i) += 1.0;

    const double truth = quad_trace(q);
    const int k = 40;          // probes per estimate
    const int reps = 400;      // independent estimates

    check(std::fabs(truth) > 1.0,
          "D1 fixture non-degenerate: the trace is O(n), not near zero");

    // Mean over reps of (mean over k probes) is an average of the estimator.
    double sum_r = 0.0, sum_g = 0.0, sum_r2 = 0.0;
    for (int r = 0; r < reps; ++r) {
        double er = hutch_trace_rademacher(n, grad_oracle(q), q.x, k, 1000u + r);
        double eg = hutch_trace_gaussian(n, grad_oracle(q), q.x, k, 1000u + r);
        sum_r += er; sum_g += eg; sum_r2 += er * er;
    }
    double mean_r = sum_r / reps;
    double mean_g = sum_g / reps;
    double var_r = sum_r2 / reps - mean_r * mean_r;

    // Tolerance is expressed in standard errors so the test is a real
    // statistical statement, not a magic constant. Both laws have E[zz']=I
    // and this is the SAME quadratic, so the Rademacher dispersion is the
    // reference scale for both.
    double se_r = std::sqrt(var_r / reps);

    printf("  [info] trace(H) = %.12f   mean_rademacher = %.12f   mean_gauss = %.12f\n",
           truth, mean_r, mean_g);

    // D1/D2: |mean - truth| within 4 standard errors.
    check(std::fabs(mean_r - truth) <= 4.0 * se_r + 1e-9,
          "D1: E[z'Hz] (Rademacher) equals tr(H) within 4 s.e.");
    // For the Gaussian law reuse a conservative absolute tolerance scaled by
    // the Rademacher dispersion (both laws have E[zz']=I).
    check(std::fabs(mean_g - truth) <= 4.0 * se_r + 1e-9,
          "D2: E[z'Hz] (Gaussian) equals tr(H) within 4 s.e.");

    // The bias must be a small FRACTION OF THE TRACE ITSELF, or the tolerance
    // above is vacuous. This is the assertion that actually kills a biased
    // estimator: dropping the 1/probes normalisation lands at k*truth, a clean
    // factor of k, and dropping the 1-probe scaling lands at the raw spread.
    check(std::fabs(mean_r - truth) < 0.05 * std::fabs(truth),
          "D1 non-vacuity: relative bias is under 5% of the trace");
    check(std::fabs(mean_g - truth) < 0.05 * std::fabs(truth),
          "D2 non-vacuity: relative bias is under 5% of the trace");
}

// =============================================================================
// D3 / D4 -- the estimator is bad at k=1 and its spread falls as 1/sqrt(k).
//
// D4 asserts the RATIO between k's, not absolute values: a test that only says
// "k=16 is closer than k=1" still passes for a k^(3/2)-wrong estimator.
// =============================================================================
static void test_probe_count_law() {
    const size_t n = 6;
    Quad q = make_quad(n, 999);
    const int reps = 2000;

    auto spread = [&](int k) {
        double m = 0.0, m2 = 0.0;
        for (int r = 0; r < reps; ++r) {
            double e = hutch_trace_rademacher(n, grad_oracle(q), q.x, k, 5000u + r);
            m += e; m2 += e * e;
        }
        m /= reps; m2 /= reps;
        return std::sqrt(std::fabs(m2 - m * m));
    };

    double s1 = spread(1), s16 = spread(16), s256 = spread(256);
    double truth = quad_trace(q);

    printf("  [info] spread: k=1 %.6f  k=16 %.6f  k=256 %.6f   |trace|=%.6f\n",
           s1, s16, s256, std::fabs(truth));

    // D3: a single probe's spread dwarfs the trace -- it is a poor estimator.
    check(s1 > std::fabs(truth), "D3: k=1 spread exceeds |tr(H)| (single probe is poor)");

    // D4: spread(16)/spread(1) ~ 1/4 and spread(256)/spread(16) ~ 1/4.
    double r16 = s16 / s1, r256 = s256 / s16;
    printf("  [info] ratios: s16/s1=%.4f (want ~0.25)  s256/s16=%.4f (want ~0.25)\n", r16, r256);
    check(r16 > 0.15 && r16 < 0.40, "D4a: spread falls ~1/4 from k=1 to k=16");
    check(r256 > 0.15 && r256 < 0.40, "D4b: spread falls ~1/4 from k=16 to k=256");
}

// =============================================================================
// D5 -- the FOM (forward-over-reverse) is FIRST order in eps.
//
// (g(x+eps v) - g(x))/eps -> H v has error O(eps), NOT O(eps^2). This is what
// distinguishes it from a symmetric (central) difference, which also converges
// and which an implementation might silently use. The test asserts the ERROR
// RATIO between eps and eps/10 is ~10 (first order), not ~100 (second order).
// =============================================================================
// TEST BUG FOUND AND FIXED HERE, 2026-10-19 -- and it is the DEGENERATE
// FIXTURE failure of systematic-debugging 5c, not a gradient bug.
//
// The first draft of this test measured the FOM order on the QUADRATIC
// fixture. That is degenerate BY CONSTRUCTION: for f = 1/2 x'Hx the gradient
// is exactly linear, so g(x + eps*v) - g(x) = eps*H*v EXACTLY in exact
// arithmetic and the truncation error is IDENTICALLY ZERO. What the test then
// measured was pure floating-point cancellation, which GROWS as eps shrinks:
// 1.1e-14, 1.2e-13, 1.1e-12, 1.4e-11, 1.4e-10 for eps = 1e-1 .. 1e-5. So the
// error ROSE with every decade and the "first order" ratio could never
// reproduce. A symmetric-difference implementation would have passed the same
// broken test.
//
// The fingerprint that identified it: the error was at the 1e-14 round-off
// floor ALREADY at eps=1e-1, i.e. the measurement had no signal at all. An
// order-of-accuracy test needs a function whose TRUNCATION error is the
// dominant term, which means a function that is not linear-plus-constant.
//
// The fix is a CUBIC fixture, g(x) = x^2 + 0.1x, whose exact FOM target is
// 2*x*v + 0.1*v and whose truncation error is genuinely O(eps). Measured on
// it: ratio exactly 10.00 per decade. The quadratic is kept below for D6,
// where its EXACTNESS is the property under test (a quadratic is the right
// fixture for the cancellation floor precisely because it has no truncation
// error to hide the floor).
static void test_fom_is_first_order() {
    const size_t n = 6;
    // g(x) = x^2 + 0.1 x  ->  H = 2x (diagonal), so H v = 2*x (.) v + 0.1*v.
    Tensor xs = Tensor::zeros(1, n);
    std::vector<double> xs_v(n, 0.0);
    unsigned s = 555u;
    auto next = [&s]() { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.0 - 0.5; };
    for (size_t j = 0; j < n; ++j) { xs_v[j] = next(); xs(0, j) = xs_v[j]; }

    Tensor v = Tensor::zeros(1, n);
    for (size_t j = 0; j < n; ++j) v(0, j) = 1.0 / std::sqrt(static_cast<double>(n));

    auto grad_of = [](const Tensor& x) {
        Tensor g(1, x.cols);
        for (size_t j = 0; j < x.cols; ++j)
            g(0, j) = x(0, j) * x(0, j) + 0.1 * x(0, j);
        return g;
    };
    HessianGradFn gf = HessianGradFn(grad_of);

    // Exact FOM target: d/dx [x^2 + 0.1x] = 2x, applied to v, plus the linear
    // term's own contribution.
    Tensor target = Tensor::zeros(1, n);
    for (size_t j = 0; j < n; ++j) target(0, j) = 2.0 * xs_v[j] * v(0, j) + 0.1 * v(0, j);

    Tensor tclone = target.clone();
    Tensor hv = hutch_hvp(gf, xs, v, 1e-4);
    check(hv.rows == tclone.rows && hv.cols == tclone.cols,
          "D5-0: hutch_hvp returns the gradient's shape");

    auto rel_err = [&](double eps) {
        Tensor a = hutch_hvp(gf, xs, v, eps);
        double num = 0.0, den = 0.0;
        for (size_t j = 0; j < n; ++j) {
            num += (a(0, j) - target(0, j)) * (a(0, j) - target(0, j));
            den += target(0, j) * target(0, j);
        }
        return std::sqrt(num / den);
    };

    double e1 = rel_err(1e-1), e2 = rel_err(1e-2), e3 = rel_err(1e-3), e4 = rel_err(1e-4);
    printf("  [info] FOM rel_err on cubic: 1e-1 %.4e  1e-2 %.4e  1e-3 %.4e  1e-4 %.4e\n",
           e1, e2, e3, e4);

    // First order => each decade multiplies the error by 10. Second order
    // (a symmetric difference) => 100. The bound is deliberately loose (4..25)
    // to admit the cancellation floor at the small end without letting a
    // second-order implementation through: that would give ratios near 100.
    check(e1 / e2 > 4.0 && e1 / e2 < 25.0, "D5a: FOM error falls ~10x per 10x eps (first order)");
    check(e2 / e3 > 4.0 && e2 / e3 < 25.0, "D5b: FOM order holds across decades");
    check(e3 / e4 > 4.0 && e3 / e4 < 25.0, "D5c: FOM order holds into the accurate regime");
    check(e4 < 1e-3, "D5d: FOM converges to small error at eps=1e-4");

    // Non-vacuity companion: the target is not the zero vector, so a relative
    // error against it is meaningful.
    double tnorm = 0.0;
    for (size_t j = 0; j < n; ++j) tnorm += target(0, j) * target(0, j);
    check(std::sqrt(tnorm) > 0.1, "D5 non-vacuity: the FOM target is non-zero");
}

// =============================================================================
// D6 -- the cancellation floor: eps that is TOO SMALL returns garbage, and the
// entry point validates rather than silently returning it.
// =============================================================================
static void test_cancellation_floor() {
    const size_t n = 6;
    Quad q = make_quad(n, 777);
    Tensor v = Tensor::zeros(1, n);
    for (size_t j = 0; j < n; ++j) v(0, j) = 1.0 / std::sqrt(static_cast<double>(n));
    Tensor g0 = quad_grad_t(q, q.x);

    std::vector<double> Hv(n, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i) Hv[j] += q.H(i, j) * v(0, i);
    double den = 0.0;
    for (size_t j = 0; j < n; ++j) den += Hv[j] * Hv[j];

    auto rel_err = [&](double eps) {
        Tensor gp = quad_grad_t(q, q.x + v * eps);
        Tensor a = (gp - g0) * (1.0 / eps);
        double num = 0.0;
        for (size_t j = 0; j < n; ++j) num += (a(0, j) - Hv[j]) * (a(0, j) - Hv[j]);
        return std::sqrt(num / den);
    };

    double good = rel_err(1e-4);
    double bad = rel_err(1e-16);
    printf("  [info] rel_err eps=1e-4 %.3e ; eps=1e-16 %.3e\n", good, bad);
    check(good < 1e-6, "D6a: eps=1e-4 is accurate");
    check(bad > 1e-3, "D6b: eps=1e-16 is garbage (cancellation floor is real)");

    // The validated entry point must REFUSE an eps at or below the floor
    // rather than return a plausible-looking wrong number.
    bool threw = false;
    try {
        (void)hutch_hvp(grad_oracle(q), q.x, v, 1e-16);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "D6c: hutch_hvp throws on a sub-floor eps");

    // A sane eps must NOT throw (the guard is not blanket-rejecting).
    bool threw_ok = false;
    try {
        (void)hutch_hvp(grad_oracle(q), q.x, v, 1e-4);
    } catch (const std::invalid_argument&) {
        threw_ok = true;
    }
    check(!threw_ok, "D6d: a sane eps is accepted (guard is not blanket)");
}

// =============================================================================
// D7 / D8 -- determinism and freshness.
//   D7: the same seed reproduces bit-identically; different seeds differ.
//   D8: the returned HVP is a FRESH tensor -- mutating the inputs afterwards
//       does not change it (no aliasing).
// =============================================================================
static void test_determinism_and_freshness() {
    const size_t n = 6;
    Quad q = make_quad(n, 31337);
    Tensor v = Tensor::zeros(1, n);
    for (size_t j = 0; j < n; ++j) v(0, j) = 1.0 / std::sqrt(static_cast<double>(n));

    double a = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 42u);
    double b = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 42u);
    double c = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 43u);
    check_eq(a, b, "D7a: same seed reproduces bit-identically");
    check(a != c, "D7b: a different seed gives a different estimate");

    Tensor v_copy = v.clone();
    Tensor hv = hutch_hvp(grad_oracle(q), q.x, v, 1e-4);
    Tensor hv_snapshot = hv.clone();
    // Mutate the inputs; the returned HVP must be unaffected.
    for (size_t j = 0; j < n; ++j) { v(0, j) = 12345.0; q.x(0, j) = -999.0; }
    bool same = true;
    for (size_t j = 0; j < hv.cols; ++j)
        if (hv(0, j) != hv_snapshot(0, j)) same = false;
    check(same, "D8: returned HVP is fresh (input mutation does not alias it)");
    check(v(0, 0) == 12345.0 && v_copy(0, 0) != 12345.0,
          "D8 non-vacuity: the mutation actually changed the input");
}

// =============================================================================
// D9 -- argument validation. dim==0 or probes<1 throws.
// =============================================================================
static void test_validation() {
    const size_t n = 4;
    Quad q = make_quad(n, 11);

    bool t1 = false, t2 = false, t3 = false;
    try { (void)hutch_trace_rademacher(0, grad_oracle(q), q.x, 4, 1u); }
    catch (const std::invalid_argument&) { t1 = true; }
    try { (void)hutch_trace_rademacher(n, grad_oracle(q), q.x, 0, 1u); }
    catch (const std::invalid_argument&) { t2 = true; }
    try { (void)hutch_trace_rademacher(n, grad_oracle(q), q.x, -3, 1u); }
    catch (const std::invalid_argument&) { t3 = true; }

    check(t1, "D9a: dim == 0 throws");
    check(t2, "D9b: probes == 0 throws");
    check(t3, "D9c: probes < 0 throws");
}

// =============================================================================
// D10 / D12 -- closed-form VALUE on a quadratic fixture and a rank-1 case.
// D10: many probes converge to the exact closed-form trace.
// D12: for a rank-1 H = lambda u u', tr = lambda and Rademacher z'Hz is
//      exactly lambda*(u.z)^2 -- the estimate is bounded and recoverable.
// =============================================================================
static void test_closed_form_values() {
    // D10
    {
        const size_t n = 5;
        Quad q = make_quad(n, 2024);
        double truth = quad_trace(q);
        double est = hutch_trace_rademacher(n, grad_oracle(q), q.x, 20000, 7u);
        check_near(est, truth, 0.05 * std::fabs(truth) + 1e-6,
                   "D10: k=20000 recovers the closed-form trace within 5%");
    }
    // D12: rank-1 H = 3 * u u' with u unit => tr = 3 * ||u||^2 = 3.
    {
        const size_t n = 8;
        Quad q;
        q.H = Tensor(n, n);
        q.c = Tensor::zeros(1, n);
        q.x = Tensor::zeros(1, n);
        // u = e_0 => H = 3 * e_0 e_0' = diag(3,0,...,0); tr = 3.
        q.H(0, 0) = 3.0;
        double truth = quad_trace(q);
        check_eq(truth, 3.0, "D12 setup: rank-1 trace is 3");
        double est = hutch_trace_rademacher(n, grad_oracle(q), q.x, 4000, 3u);
        // Rademacher z'Hz = 3*z_0^2 = 3 always (since z_0 = +-1). So the
        // estimate is EXACTLY 3 for every probe -- a stronger claim than
        // "converges to 3".
        check_near(est, 3.0, 1e-9,
                   "D12: rank-1 Rademacher estimate is exactly tr (z_0^2 == 1)");
    }
}

// =============================================================================
// The Gaussian probe law must genuinely differ from Rademacher (i.e. it is
// not secretly calling the Rademacher routine). A Gaussian probe is NOT
// +-1, so a single-probe estimate must differ generically.
// =============================================================================
static void test_probe_laws_are_distinct() {
    const size_t n = 6;
    Quad q = make_quad(n, 4242);
    double r1 = hutch_trace_rademacher(n, grad_oracle(q), q.x, 1, 77u);
    double g1 = hutch_trace_gaussian(n, grad_oracle(q), q.x, 1, 77u);
    check(r1 != g1, "Gaussian and Rademacher laws are distinct at k=1");
}

int main() {
    printf("=== test_hutchinson ===\n");
    test_unbiasedness();
    test_probe_count_law();
    test_fom_is_first_order();
    test_cancellation_floor();
    test_determinism_and_freshness();
    test_validation();
    test_closed_form_values();
    test_probe_laws_are_distinct();
    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}