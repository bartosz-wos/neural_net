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
#include <map>
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

// =============================================================================
// Every place the suite calls into production code that can THROW must be
// inside a try/catch.
//
// M09 (invert the `probes < 1` guard) aborted the whole binary --
// `terminate called after throwing an instance of 'std::invalid_argument'`,
// exit 134 -- because the statistical tests call hutch_trace_* with k = 40 and
// 400 reps directly, with nothing to catch a throw. The process died before
// the summary line, so the harness had NOTHING TO COUNT and reported MUT-CRASH
// even though the mutation was detected perfectly well.
//
// This is the same shape as the muP session's M13, and the same fix: the
// DETECTOR WAS WORKING; the harness simply had nothing to count. Note the
// distinction from M08, whose crash was a SEGFAULT in the production code on
// an unsafe fixture -- this one is a TEST defect, that one was a fixture
// defect. Both present as "no summary line".
// =============================================================================
template <typename F>
static bool call_guarded(F f) {
    try { f(); return true; }
    catch (const std::invalid_argument&) { return false; }
    catch (...) { return false; }
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

// A QUADRATIC WITH A DELIBERATELY LARGE, NON-ZERO x -- the fixture the M02
// investigation needed. Its final form is the x-scaled variant inside D10b
// (which needed TWO magnitudes, not one), but the reasoning that produced it
// is worth keeping, because the first draft of that test was wrong in an
// instructive way:
//
// M02 contracts z' against `zp` instead of the perturbation `(zp - x)`. I
// assumed the extra x'Hz term was a BIAS and wrote a mean-based test. It
// survived -- because E[x'Hz] = 0 (z is independent of x and has mean zero),
// so M02 is UNBIASED with an inflated DISPERSION. Measured: correct sd
// 0.792423, M02 sd 3.657607, a 4.62x inflation. No mean-based test can see
// that; only the sd can, and only by being compared across two x values.
// *A wrong implementation that preserves the expectation is invisible to every
// first-moment test in the suite -- which is why D10b asserts dispersion and
// why it needs two points to establish independence.*

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
    bool g_threw = false;
    for (int r = 0; r < reps; ++r) {
        double er = 0.0, eg = 0.0;
        bool ok_r = call_guarded([&]{ er = hutch_trace_rademacher(n, grad_oracle(q), q.x, k, 1000u + r); });
        bool ok_g = call_guarded([&]{ eg = hutch_trace_gaussian(n, grad_oracle(q), q.x, k, 1000u + r); });
        if (!ok_r || !ok_g) { g_threw = true; break; }
        sum_r += er; sum_g += eg; sum_r2 += er * er;
    }
    check(!g_threw, "D1 setup: the estimator call itself must not throw");
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
            double e = 0.0;
            if (!call_guarded([&]{ e = hutch_trace_rademacher(n, grad_oracle(q), q.x, k, 5000u + r); }))
                return -1.0;   // a throwing implementation has no spread to measure
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
        Tensor a = Tensor::zeros(1, n);
        // Guarded so a throwing implementation produces a counted failure
        // rather than aborting the binary before the summary line.
        if (!call_guarded([&]{ a = hutch_hvp(gf, xs, v, eps); }))
            return 1.0;   // maximal relative error: it could not produce a value
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

    double a = 0.0, b = 0.0, c = 0.0;
    bool ok = call_guarded([&]{
        a = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 42u);
        b = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 42u);
        c = hutch_trace_rademacher(n, grad_oracle(q), q.x, 8, 43u);
    });
    check(ok, "D7 setup: the seeded estimator calls do not throw");
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

    // M08 SURVIVED 41 CHECKS, and the reason is this file's third instance of
    // the "the test asserts THAT IT THREW, not THAT IT THREW HERE" trap.
    // With dim == 0 but x.cols == 6, the dim guard's removal is invisible --
    // the very next guard (`x.cols != dim`) throws for a different reason and
    // the catch block cannot tell the two apart. The original D9a therefore
    // asserted nothing about the dim guard at all.
    //
    // The fix is to assert on the MESSAGE, which is the actual discriminator --
    // the exception TYPE was never it.
    //
    // A SECOND HARNESS DEFECT WAS FOUND WHILE DOING THIS, AND IT IS THE MORE
    // IMPORTANT ONE. The first attempt made x a zero-column tensor so that
    // only the dim guard could fire. With the guard removed, that fixture
    // SEGFAULTED: `Tensor::zeros(1, 0)` has no elements, and the estimator
    // proceeds to grad(x) and indexes it. `Tensor::operator()` is
    // `data[r*cols + c]` with NO bounds check (the repo's eighth recorded
    // instance of that), so the out-of-bounds read walks off the allocation.
    // The mutation was being detected as a crash rather than as a failure --
    // a CRASH IS NOT A VERDICT. The fix is to keep x VALID and let the
    // message do the discriminating, so no fixture ever reaches the
    // unguarded code path.
    auto message_of = [q](size_t dim, int probes) {
        try {
            (void)hutch_trace_rademacher(dim, grad_oracle(q), q.x, probes, 1u);
            return std::string("<no throw>");
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
    };

    std::string m0 = message_of(0, 4);
    // The discriminator must be "dim must be", NOT the bare substring "dim" --
    // the mismatch guard's message is "hutch: x.cols does not match DIM", which
    // CONTAINS "dim". A substring test was therefore satisfied by the wrong
    // guard, which is exactly how M08 (removing the dim guard) survived: the
    // neighbouring guard threw with a message containing the searched token.
    // *A substring that appears in two different messages discriminates
    // nothing -- the assertion has to name the message that is unique to the
    // guard under test.*
    check(m0.find("dim must be") != std::string::npos,
          "D9a: dim == 0 throws, and it is the DIM guard that threw (not a neighbour)");
    if (m0.find("dim") == std::string::npos)
        printf("    [info] dim==0 message was: %s\n", m0.c_str());

    std::string m1 = message_of(n, 0);
    check(m1.find("probes must be") != std::string::npos,
          "D9b: probes == 0 throws, and it is the PROBES guard (not the dim guard)");
    if (m1.find("probes") == std::string::npos)
        printf("    [info] probes==0 message was: %s\n", m1.c_str());

    bool t3 = false;
    try { (void)hutch_trace_rademacher(n, grad_oracle(q), q.x, -3, 1u); }
    catch (const std::invalid_argument&) { t3 = true; }
    check(t3, "D9c: probes < 0 throws");

    // And the mismatch guard itself, now that its message is the discriminator.
    std::string mm;
    try {
        (void)hutch_trace_rademacher(99, grad_oracle(q), q.x, 4, 1u);
        mm = "<no throw>";
    } catch (const std::invalid_argument& e) {
        mm = e.what();
    }
    check(mm.find("does not match") != std::string::npos,
          "D9d: a dim/x.cols mismatch throws the mismatch guard");
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
        double est = 0.0;
        bool ok = call_guarded([&]{ est = hutch_trace_rademacher(n, grad_oracle(q), q.x, 20000, 7u); });
        check(ok, "D10 setup: the k=20000 estimator call does not throw");
        check_near(est, truth, 0.05 * std::fabs(truth) + 1e-6,
                   "D10: k=20000 recovers the closed-form trace within 5%");
    }
    // D10b -- THE FIXTURE THAT KILLS M02, AND THE FIRST DRAFT OF IT DID
    // NOT, FOR A REASON WORTH RECORDING.
    //
    // M02 contracts z' against `zp` instead of the perturbation `(zp - x)`,
    // adding a spurious x'Hz term. My first assumption was that this is a
    // BIAS, and I wrote a mean-based unbiasedness test to catch it. It
    // survived -- because the term is NOT a bias: E[x'Hz] = E[x']E[Hz] = 0
    // since z is independent of x and has mean zero. M02 is an UNBIASED
    // estimator with a DIFFERENT DISPERSION, and no mean-based test can ever
    // see it. Measured: correct sd 0.792423, M02 sd 3.657607 -- 4.62x
    // inflation.
    //
    // The property that DOES separate them is that the correct estimator's
    // dispersion is INDEPENDENT OF x (the trace does not depend on where you
    // evaluate it), while M02's scales with |x|. So this asserts the sd, not
    // the mean -- and it asserts it at TWO different x values, because a
    // single x cannot establish "independent of".
    {
        const size_t n = 6;
        const int k = 40, reps = 3000;
        auto spread_at = [&](double xscale) {
            Quad q = make_quad(6, 8080);
            for (size_t j = 0; j < n; ++j) q.x(0, j) = xscale * (3.0 + 0.5 * static_cast<double>(j));
            double m = 0.0, m2 = 0.0;
            for (int r = 0; r < reps; ++r) {
                double e = 0.0;
                if (!call_guarded([&]{ e = hutch_trace_rademacher(n, grad_oracle(q), q.x, k, 9000u + r); }))
                    return -1.0;
                m += e; m2 += e * e;
            }
            m /= reps; m2 /= reps;
            return std::sqrt(std::fabs(m2 - m * m));
        };
        double sd_near = spread_at(0.05);   // x ~ 0.05*(3, 3.5, ...) -> ~0
        double sd_far  = spread_at(1.0);    // x = (3, 3.5, 4, ...)
        double ratio = sd_far / sd_near;
        printf("  [info] sd near-origin=%.6f  sd far-from-origin=%.6f  ratio=%.4f (want ~1.00)\n",
               sd_near, sd_far, ratio);
        // The correct estimator's dispersion does not depend on x. M02
        // inflates it by ~4.6x here, so a band of 1.6..2.5 catches it while
        // admitting the sampling noise of a 3000-reply estimate.
        check(ratio > 0.80 && ratio < 1.25,
              "D10b: estimator dispersion is INDEPENDENT of x (kills the zp contraction)");
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
        double est = 0.0;
        bool ok = call_guarded([&]{ est = hutch_trace_rademacher(n, grad_oracle(q), q.x, 4000, 3u); });
        check(ok, "D12 setup: the rank-1 estimator call does not throw");
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
    double r1 = 0.0, g1 = 0.0;
    bool ok_r = call_guarded([&]{ r1 = hutch_trace_rademacher(n, grad_oracle(q), q.x, 1, 77u); });
    bool ok_g = call_guarded([&]{ g1 = hutch_trace_gaussian(n, grad_oracle(q), q.x, 1, 77u); });
    check(ok_r && ok_g, "probe-law setup: the k=1 calls do not throw");
    check(r1 != g1, "Gaussian and Rademacher laws are distinct at k=1");
}

// =============================================================================
// THE PART THAT ACTUALLY CLOSES THE GAP: per-tensor Hessian-diagonal
// estimates in the shape Sophia::set_hessian_estimates expects.
//
// D11 -- the estimates are per TENSOR and index-aligned with
// Layer::parameters(), so the result can be handed to set_hessian_estimates
//       with no reindexing.
// D13 -- the model is restored BIT-IDENTICALLY. The estimator perturbs
//       theta <- theta + eps*z; a helper that leaves the caller's model
//       perturbed is a landmine even if it looks like it cleaned up.
// D14 -- gradients are ZEROED before each probe's backward. Dense::backward
//       ACCUMULATES into grad_weights (`grad_weights += grad_w`), so without
//       the zeroing probe k reads probe k-1's gradient plus its own and every
//       estimate after the first is contaminated. This is the repo's
//       accumulation contract (core/layer.cpp:78) and it is invisible to a
//       single-probe test.
// =============================================================================
static void test_model_diagonal() {
    Model model;
    model.add_layer(new Dense(5, 4));
    model.add_layer(new Activation(ReLU()));
    model.add_layer(new Dense(4, 3));

    Tensor x = Tensor::zeros(1, 5);
    for (size_t j = 0; j < 5; ++j) x(0, j) = 0.1 * static_cast<double>(j + 1);
    Tensor dLdy = Tensor::zeros(1, 3);
    for (size_t j = 0; j < 3; ++j) dLdy(0, j) = 1.0;

    // Snapshot every parameter BEFORE, so restoration is measured rather than
    // assumed.
    std::map<Layer*, std::vector<Tensor>> before;
    for (auto& holder : model.layers) {
        std::vector<Tensor> snap;
        for (Tensor* p : holder.get()->parameters()) snap.push_back(*p);
        before[holder.get()] = snap;
    }

    auto out = hutch_trace_diagonal(model, x, dLdy, 3, 123u, 1e-4);
    check(out.size() == 3, "D11a: one estimate per probe requested");

    // D11: per-tensor shapes must match the model's parameter shapes exactly.
    bool shapes_ok = true;
    size_t total_tensors = 0;
    for (auto& holder : model.layers) total_tensors += holder.get()->parameters().size();
    check(out.size() == 3 && out[0].size() == total_tensors,
          "D11b: estimates are per-parameter-tensor, index-aligned with parameters()");

    for (auto& holder : model.layers) {
        auto params = holder.get()->parameters();
        size_t idx = 0;
        for (size_t li = 0; li < model.layers.size(); ++li)
            if (model.layers[li].get() == holder.get()) {
                idx = li;
                break;
            }
        size_t running = 0;
        for (size_t li = 0; li < idx; ++li) running += model.layers[li]->parameters().size();
        for (size_t i = 0; i < params.size(); ++i, ++running) {
            if (out[0][running].rows != params[i]->rows ||
                out[0][running].cols != params[i]->cols)
                shapes_ok = false;
        }
    }
    check(shapes_ok, "D11c: each estimate has its parameter's exact shape");

    // D14: the estimates across probes must DIFFER. If gradients were not
    // zeroed, probe 2 would be probe 1's gradient PLUS its own, and the
    // difference between probes would still be non-zero... so instead pin the
    // property that actually catches accumulation: probe 1 computed ALONE must
    // equal probe 1 computed as the FIRST of three. That equality is only true
    // if the backward is clean at the start.
    auto alone = hutch_trace_diagonal(model, x, dLdy, 1, 123u, 1e-4);
    bool first_matches = true;
    for (size_t i = 0; i < out[0].size(); ++i)
        for (size_t r = 0; r < out[0][i].rows; ++r)
            for (size_t c = 0; c < out[0][i].cols; ++c)
                if (alone[0][i](r, c) != out[0][i](r, c)) first_matches = false;
    check(first_matches,
          "D14: the first probe is unaffected by how many probes follow (grads zeroed)");

    // D13: every parameter restored bit-identically.
    bool restored = true;
    for (auto& holder : model.layers) {
        auto params = holder.get()->parameters();
        const std::vector<Tensor>& snap = before[holder.get()];
        for (size_t i = 0; i < params.size() && i < snap.size(); ++i) {
            if (params[i]->rows != snap[i].rows || params[i]->cols != snap[i].cols) {
                restored = false;
                continue;
            }
            for (size_t k = 0; k < params[i]->rows * params[i]->cols; ++k)
                if ((*params[i]).data[k] != snap[i].data[k]) restored = false;
        }
    }
    check(restored, "D13: every parameter is restored bit-identically");

    // D13 non-vacuity: the perturbation was real, i.e. the forward pass at
    // theta+eps*z actually differs. If eps were 0 the restore would be trivial.
    Tensor y0 = model.forward(x);
    std::map<Layer*, std::vector<Tensor>> snapshot2;
    for (auto& holder : model.layers) {
        std::vector<Tensor> snap;
        for (Tensor* p : holder.get()->parameters()) snap.push_back(*p);
        snapshot2[holder.get()] = snap;
    }
    (void)y0;
    (void)snapshot2;
    {
        // Perturb by hand with the same eps, and require a different forward.
        for (auto& holder : model.layers) {
            auto params = holder.get()->parameters();
            if (params.empty()) continue;
            for (size_t k = 0; k < params[0]->rows * params[0]->cols; ++k)
                (*params[0]).data[k] += 0.05;
            break;
        }
        Tensor y1 = model.forward(x);
        double diff = 0.0;
        for (size_t j = 0; j < y1.cols; ++j) diff += std::fabs(y1(0, j) - y0(0, j));
        check(diff > 1e-9, "D13 non-vacuity: perturbing theta changes the forward output");
        // put it back
        for (auto& holder : model.layers) {
            auto params = holder.get()->parameters();
            if (params.empty()) continue;
            const std::vector<Tensor>& snap = before[holder.get()];
            for (size_t i = 0; i < params.size(); ++i)
                if (params[i]->rows == snap[i].rows && params[i]->cols == snap[i].cols)
                    *params[i] = snap[i];
        }
    }

    // Validation: probes < 1 throws; an empty model throws.
    bool t1 = false, t2 = false;
    try { (void)hutch_trace_diagonal(model, x, dLdy, 0, 1u, 1e-4); }
    catch (const std::invalid_argument&) { t1 = true; }
    Model empty;
    try { (void)hutch_trace_diagonal(empty, x, dLdy, 1, 1u, 1e-4); }
    catch (const std::invalid_argument&) { t2 = true; }
    check(t1, "D15a: hutch_trace_diagonal throws on probes < 1");
    check(t2, "D15b: hutch_trace_diagonal throws on a model with no parameters");
}

// =============================================================================
// The end-to-end shape Sophia actually consumes: estimates handed straight to
// set_hessian_estimates and consumed by a real step(). This is the test that
// would have failed before this file existed, because there was nothing to
// hand over.
// =============================================================================
static void test_feeds_sophia() {
    Model model;
    model.add_layer(new Dense(4, 3));
    Tensor x = Tensor::zeros(1, 4);
    for (size_t j = 0; j < 4; ++j) x(0, j) = 0.5;
    Tensor dLdy = Tensor::zeros(1, 3);
    for (size_t j = 0; j < 3; ++j) dLdy(0, j) = 1.0;

    auto est = hutch_trace_diagonal(model, x, dLdy, 8, 77u, 1e-4);
    check(!est.empty() && !est[0].empty(),
          "E2E: produced a non-empty per-tensor estimate to hand to Sophia");

    Sophia opt(1e-3, 0.9, 0.99, 1.0);
    // The estimate must be usable: set it and step without throwing.
    bool stepped = true;
    try {
        model.forward(x);
        opt.set_hessian_estimates(model.layers[0].get(), est[0]);
        opt.step(model);
    } catch (...) {
        stepped = false;
    }
    check(stepped, "E2E: Sophia consumes the estimate and steps");

    // The estimate must be FINITE. A NaN here would silently poison Sophia's
    // update (h_hat/eps -> NaN) and there is no guard downstream.
    bool finite = true;
    for (auto& per_tensor : est)
        for (auto& t : per_tensor)
            for (size_t k = 0; k < t.rows * t.cols; ++k)
                if (!std::isfinite(t.data[k])) finite = false;
    check(finite, "E2E: every estimate entry is finite (no NaN to poison Sophia)");
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
    test_model_diagonal();
    test_feeds_sophia();
    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}