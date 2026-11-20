// =============================================================================
// tests/test_correlation_tests.cpp -- the correlation hypothesis tests: the
// p-values for Pearson's r, Spearman's rho and Kendall's tau-b.
//
// THE SPECIFICATION IS tests/correlation_tests_oracle.py (float64 numpy),
// which was written and run BEFORE any of this C++ existed.
//
// A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
// SENSITIVITY, so the oracle's discriminating power is measured rather than
// narrated: tests/correlation_tests_mutate_oracle.py breaks the oracle's
// model code in specific ways and requires that it fail each time.
//
// =============================================================================
// WHAT THIS SUITE EXISTS TO PIN
// =============================================================================
//
// The repo ships three correlation coefficients and NO p-value for any of
// them, so "is this association distinguishable from noise" has no answer
// in this repo. This suite pins the p-values, and just as importantly the
// CONVENTIONS, because each is a place where a plausible implementation
// returns a DIFFERENT number:
//
//   1. THE PEARSON AND SPEARMAN p-VALUES ARE THE t STATISTIC'S TWO-SIDED
//      SURVIVAL, and they agree with scipy's DEFAULT (the exact beta
//      distribution of r) to 3.3e-14 -- the two references disagree on
//      the ALGORITHM and agree on the ANSWER.
//
//   2. THE KENDALL p-VALUE IS THE ASYMPTOTIC NORMAL, WITH THE TIE
//      CORRECTIONS IN ITS VARIANCE. The tie terms are load-bearing, and
//      the asymptotic form is the ONLY form that exists for a tied
//      sample.
//
//   3. THE TIE-GROUP STATISTICS ARE NOT THE TIE PAIR COUNT: two samples
//      with the groups {4,2} and {3,3,2} have the SAME number of tied
//      pairs and DIFFERENT variances.
//
//   4. n == 2 IS NaN, FOR ALL THREE TESTS -- the references disagree
//      there, so there is no reference to agree with.
//
//   5. THE STATISTIC IS CLIPPED TO [-1, 1]. This is NOT cosmetic: this
//      repo's OWN `pearson_r` returns up to 1.0000000000000011102, which
//      makes `1 - r*r` negative and the p-value of a PERFECT correlation
//      NaN. **A PERFECT CORRELATION IS THE MOST IMPORTANT FIXTURE IN THE
//      WHOLE MODULE, AND IT IS THE ONE THAT BREAKS WITHOUT A CLIP.**
//
//   6. THE p-VALUE IS TWO-SIDED, ALWAYS.
//
// THE FIXTURES ARE EMITTED, NEVER TYPED.
//
// The numbers below come from tests/correlation_tests_oracle.py (float64
// numpy, measured against scipy 1.17.1 and R 4.5.3), which was run BEFORE
// this file existed. The repo has been burned by hand-transcribed numbers
// before -- both numbers were correct for the fixture they were given, and
// the fixture was simply not the oracle's -- so the pinned numbers are
// TRANSFERRED by tests/splice_correlation_tests_fixture.py, never
// re-derived.
//
// THE INDEPENDENT REFERENCE.
//
// The t-statistic p-value is ALSO recomputed in THIS FILE, from the
// DEFINITION, with an explicit scalar loop and no shared code with the
// library: the regularised incomplete beta is written out here from the
// continued-fraction expansion. The library is not checked against itself.
//
// The Kendall p-value is ALSO recomputed here, from the definition, with
// plain scalar loops over the pairs and the tie groups.
//
// Every guard's MESSAGE is checked against a DISTINCT needle. A needle
// that appears in two different messages discriminates nothing between
// them, and this repo has hit that trap repeatedly.
// =============================================================================

#include "nn/utils/correlation_tests.h"
#include "nn/utils/rank_correlation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// `Tensor` is at GLOBAL scope in this repo (include/nn/core/tensor.h has no
// namespace -- measured, `grep -c 'namespace'` -> 0), so it needs no
// qualification and no using-declaration. The utility functions live in
// `nn::utils`. This is the same convention tests/test_rank_correlation.cpp
// uses.
using nn::utils::kendall_tau_b;
using nn::utils::kendall_tau_b_p_value;
using nn::utils::kendall_tau_b_test;
using nn::utils::pearson_p_value;
using nn::utils::pearson_r;
using nn::utils::pearson_r_test;
using nn::utils::spearman_p_value;
using nn::utils::spearman_rho;
using nn::utils::spearman_rho_test;

static int g_pass = 0;
static int g_fail = 0;

// The statistic that is TWO ULP above 1 -- the largest |r| this repo's own
// `pearson_r` was MEASURED to return over 300000 near-perfectly-collinear
// samples. It is the fixture for the CLIP (C4): without the clip, the
// p-value of a PERFECT correlation would be NaN.
static const double kStatisticAboveOne = 1.0000000000000011102;

static void check(bool ok, const std::string& msg) {
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << std::endl;
    }
}

// A RELATIVE comparison, with an absolute floor.
//
// **THE FLOOR MATTERS.** A p-value near 1 has a relative error that is
// dominated by the last bits of a subtraction near 1, and a p-value near
// 0 has one that is dominated by the tail evaluation. The comparison is
// therefore `|got - want| <= tol * max(|want|, floor)`, and the floor is
// ABSOLUTE -- so a p-value of 1e-20 is not held to a 1e-12 RELATIVE
// standard against a reference computed by a different route.
static void close_rel(double got, double want, double tol, double floor,
                      const std::string& msg) {
    const bool ok = std::fabs(got - want) <= tol * std::max(std::fabs(want), floor);
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (got " << got << ", want " << want
                  << ", diff " << (got - want) << ")" << std::endl;
    }
}

// A 1-D sample as a (N,1) Tensor.
static Tensor col(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// The same values as a (1,N) Tensor -- the convention is that the VALUES
// are what matter, so this must give identical answers.
static Tensor row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t i = 0; i < v.size(); ++i) t(0, i) = v[i];
    return t;
}

// ---------------------------------------------------------------------------
// The guard helpers.
//
// A message is read AFTER the call it belongs to. C++ evaluates a function
// call's arguments in UNSPECIFIED order, so `check(threw_with(fn, needle),
// "... " + what)` can print the message from the PREVIOUS call: the
// verdicts stay right and the human-readable text goes stale, which is the
// worst failure mode, because a reader trusts the text.
// ---------------------------------------------------------------------------
static bool threw_with(const std::function<void()>& fn, const std::string& needle,
                       std::string* what_out) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (what_out) *what_out = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void expect_throw(const std::function<void()>& fn, const std::string& needle,
                         const std::string& msg) {
    std::string what;
    const bool threw = threw_with(fn, needle, &what);
    if (threw) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (expected a throw mentioning \""
                  << needle << "\", got \"" << what << "\")" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// The p-value calls, made THROW-PROOF.
//
// **A THROWING IMPLEMENTATION MUST BE A FAILED CHECK, NOT A CRASHED
// BINARY.** The first draft of this suite called `p_value([&] { return pearson_p_value(1.0, 2); })`
// directly, so an implementation that THROWS there (instead of returning
// NaN) escaped `main` and killed the binary with SIGABRT. The mutation
// harness reported that as `MUT-CRASH` -- and **A CRASH IS NOT A
// VERDICT**: it says nothing about whether the suite covers the
// behaviour.
//
// So every p-value call in this file goes through `p_value()`, which
// catches the exception and reports it as a FAILED CHECK. **A SUITE THAT
// CRASHES ON A WRONG IMPLEMENTATION CANNOT REPORT THAT THE
// IMPLEMENTATION IS WRONG.**
// ---------------------------------------------------------------------------
static bool g_p_value_threw = false;

static double p_value(const std::function<double()>& fn) {
    g_p_value_threw = false;
    try {
        return fn();
    } catch (const std::exception&) {
        g_p_value_threw = true;
        return std::numeric_limits<double>::quiet_NaN();
    }
}

// The p-value, AND the fact that it did not throw.
//
// **A THROW IS A DIFFERENT ANSWER FROM A NaN, AND THE SUITE MUST BE ABLE
// TO TELL THEM APART.** The `p_value` helper above catches the exception
// so a throwing implementation cannot crash the binary -- but a helper
// that swallows the distinction would make the mutation "the n == 2 case
// THROWS instead of returning NaN" SURVIVE, because the swallowed throw
// would produce the same NaN the check asks for.
//
// **THE MUTATION HARNESS FOUND EXACTLY THAT**: M7 SURVIVED the first
// run. So this helper returns the NaN AND sets a flag, and the checks
// that care about the distinction assert on the flag.
static double p_value_no_throw(const std::function<double()>& fn, const std::string& msg) {
    const double v = p_value(fn);
    if (g_p_value_threw) {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (it THREW instead of returning a value)"
                  << std::endl;
        return v;
    }
    return v;
}

// The wrapper call, made THROW-PROOF.
//
// **THE SAME LESSON AS THE p-VALUE HELPER, ONE LEVEL UP.** A wrapper
// that throws (because its inner p-value call throws) must be a FAILED
// CHECK, not a crashed binary: the mutation "the Pearson p-value's
// argument order is (n, r)" makes the wrapper throw, and the suite
// reported that as `MUT-CRASH` -- **a crash is not a verdict**.
//
// So the wrapper call goes through this helper, which reports the throw
// as a failure and returns a NaN-filled result.
static nn::utils::CorrelationTestResult test_call(
    const std::function<nn::utils::CorrelationTestResult()>& fn,
    const std::string& msg) {
    try {
        return fn();
    } catch (const std::exception&) {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (it THREW instead of returning a result)"
                  << std::endl;
        nn::utils::CorrelationTestResult out;
        return out;
    }
}

// ---------------------------------------------------------------------------
// The independent reference.
//
// The regularised incomplete beta I_x(a, b), written out here from the
// continued-fraction expansion, and the Student's t two-sided survival
// built on it. **NO CODE IS SHARED WITH THE LIBRARY** -- the library
// composes `utils/randomized_smoothing.h`'s implementation, and this file
// has its own, so the two are independent witnesses.
// ---------------------------------------------------------------------------
static double ref_beta_cf(double a, double b, double x) {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 300; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        c = 1.0 + aa / c;
        if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        c = 1.0 + aa / c;
        if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 3e-16) break;
    }
    return h;
}

static double ref_regularized_incomplete_beta(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (a <= 0.0 || b <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double log_front = std::lgamma(a + b) - std::lgamma(a) -
                             std::lgamma(b) + a * std::log(x) + b * std::log1p(-x);
    const double front = std::exp(log_front);
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * ref_beta_cf(a, b, x) / a;
    }
    return 1.0 - front * ref_beta_cf(b, a, 1.0 - x) / b;
}

// The two-sided t survival, from the definition.
static double ref_t_two_sided(double t, double dof) {
    if (dof <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double at = std::fabs(t);
    if (at == 0.0) return 1.0;
    if (std::isinf(at)) return 0.0;
    return ref_regularized_incomplete_beta(dof / 2.0, 0.5, dof / (dof + at * at));
}

// The p-value of a correlation, from the definition.
static double ref_correlation_p(double r, size_t n) {
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(r)) return std::numeric_limits<double>::quiet_NaN();
    if (r > 1.0) r = 1.0;
    if (r < -1.0) r = -1.0;
    const double dof = static_cast<double>(n) - 2.0;
    const double denom = 1.0 - r * r;
    if (denom <= 0.0) return 0.0;
    return ref_t_two_sided(std::fabs(r) * std::sqrt(dof / denom), dof);
}

// The tie-GROUP statistics, from the definition.
static void ref_tie_groups(const std::vector<double>& v, double* pairs, double* x0,
                            double* x1) {
    std::vector<double> s(v);
    std::sort(s.begin(), s.end());
    *pairs = 0.0;
    *x0 = 0.0;
    *x1 = 0.0;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j + 1 < s.size() && s[j + 1] == s[i]) ++j;
        const double c = static_cast<double>(j - i + 1);
        if (c >= 2.0) {
            *pairs += c * (c - 1.0) / 2.0;
            *x0 += c * (c - 1.0) * (c - 2.0);
            *x1 += c * (c - 1.0) * (2.0 * c + 5.0);
        }
        i = j + 1;
    }
}

// The Kendall p-value, from the definition.
static double ref_kendall_p(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();
    const double tot = 0.5 * static_cast<double>(n) * static_cast<double>(n - 1);
    double xt = 0.0, yt = 0.0, nt = 0.0, dis = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = x[i] - x[j];
            const double dy = y[i] - y[j];
            if (dx == 0.0 && dy == 0.0) { xt += 1.0; yt += 1.0; nt += 1.0; }
            else if (dx == 0.0) { xt += 1.0; }
            else if (dy == 0.0) { yt += 1.0; }
            else if ((dx > 0.0) == (dy > 0.0)) { /* concordant */ }
            else { dis += 1.0; }
        }
    }
    const double cmd = tot - xt - yt + nt - 2.0 * dis;
    double xp, x0, x1, yp, y0, y1;
    ref_tie_groups(x, &xp, &x0, &x1);
    ref_tie_groups(y, &yp, &y0, &y1);
    const double m = static_cast<double>(n) * static_cast<double>(n - 1);
    const double dn = static_cast<double>(n);
    const double var = (m * (2.0 * dn + 5.0) - x1 - y1) / 18.0 +
                        (2.0 * xp * yp) / m +
                        (x0 * y0) / (9.0 * m * (dn - 2.0));
    if (!(var > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    const double z = cmd / std::sqrt(var);
    // The two-sided normal survival, by the error function.
    //
    // **THE FACTOR IS 1, NOT 2.** The two-sided p-value is
    // 2*(1 - Phi(|z|)), and Phi(x) = 0.5*erfc(-x/sqrt(2)), so
    // 2*(1 - Phi(|z|)) IS erfc(|z|/sqrt(2)) -- the erfc already carries
    // both tails. Multiplying it by 2 again gives a p-value twice the
    // right one, which is a number no reader would catch by eye.
    //
    // **THE FIRST DRAFT OF THIS REFERENCE HAD THE FACTOR 2, AND THE
    // SUITE CAUGHT IT**: the F2 comparison failed with the p-value at
    // exactly half the oracle's.
    return std::erfc(std::fabs(z) / std::sqrt(2.0));
}

// ---------------------------------------------------------------------------
// THE PINNED FIXTURE. Emitted by tests/correlation_tests_oracle.py.
//
// The numbers are TRANSFERRED, never hand-transcribed, and they live in
// their own generated file -- so a reader can tell at a glance which lines
// are pinned by the oracle and which are not. The splice script
// (tests/splice_correlation_tests_fixture.py) rewrites both this region and
// the .inc file from the oracle, and it FAILS LOUDLY rather than leaving
// stale numbers behind.
// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
static const std::vector<double> f1_x = {1.0, 2.0, 3.0, 4.0, 5.0};
static const std::vector<double> f1_y = {2.0, 1.0, 4.0, 3.0, 5.0};
static const size_t f1_n = 5;
static const double f1_r = 0.80000000000000004;
static const double f1_rho = 0.79999999999999993;
static const double f1_tau = 0.59999999999999998;
static const double f1_p_pearson = 0.10408803866182782;
static const double f1_p_spearman = 0.10408803866182788;
static const double f1_p_kendall_asym = 0.14164469029513679;
static const std::vector<double> f2_x = {1.0, 2.0, 2.0, 4.0, 5.0};
static const std::vector<double> f2_y = {2.0, 1.0, 4.0, 4.0, 5.0};
static const size_t f2_n = 5;
static const double f2_r = 0.75925925925925919;
static const double f2_rho = 0.76315789473684215;
static const double f2_tau = 0.66666666666666663;
static const double f2_p_pearson = 0.136558230529749;
static const double f2_p_spearman = 0.13333911953180624;
static const double f2_p_kendall_asym = 0.1184329289166719;
static const std::vector<double> f3_x = {1.0, 3.0, 2.0, 7.0, 5.0, 9.0, 4.0, 11.0, 6.0, 8.0, 12.0, 10.0, 14.0, 13.0, 15.0, 16.0, 18.0, 17.0, 19.0, 20.0};
static const std::vector<double> f3_y = {2.0, 5.0, 3.0, 8.0, 6.0, 11.0, 4.0, 12.0, 7.0, 9.0, 13.0, 10.0, 15.0, 14.0, 16.0, 17.0, 19.0, 18.0, 20.0, 21.0};
static const size_t f3_n = 20;
static const double f3_r = 0.99699248120300732;
static const double f3_rho = 0.99699248120300743;
static const double f3_tau = 0.97894736842105268;
static const double f3_p_pearson = 1.8911024268002034e-21;
static const double f3_p_spearman = 1.8911024267995745e-21;
static const double f3_p_kendall_asym = 1.59320548535935e-09;
static const std::vector<double> f4_x = {1.0, 2.0, 3.0, 4.0, 5.0};
static const std::vector<double> f4_y = {5.0, 4.0, 3.0, 2.0, 1.0};
static const size_t f4_n = 5;
static const double f4_r = -1;
static const double f4_rho = -0.99999999999999989;
static const double f4_tau = -0.99999999999999989;
static const double f4_p_pearson = 0;
static const double f4_p_spearman = 1.4042654220543602e-24;
static const double f4_p_kendall_asym = 0.014305878435429648;
static const std::vector<double> f5_x = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0};
static const std::vector<double> f5_y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0};
static const size_t f5_n = 10;
static const double f5_r = 0.81039772527019471;
static const double f5_rho = 0.93204371574531142;
static const double f5_tau = 0.8312389335458098;
static const double f5_p_pearson = 0.0044664249294104228;
static const double f5_p_spearman = 8.5907904781404166e-05;
static const double f5_p_kendall_asym = 0.0020448067735510992;
static const std::vector<double> f6_x = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0};
static const std::vector<double> f6_y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0};
static const size_t f6_n = 10;
static const double f6_r = 0.84373781129060843;
static const double f6_rho = 0.94534250754722882;
static const double f6_tau = 0.85721515021911643;
static const double f6_p_pearson = 0.0021505181852753534;
static const double f6_p_spearman = 3.6542907714078039e-05;
static const double f6_p_kendall_asym = 0.0015714809189340093;
// >>> ORACLE-EMITTED FIXTURE END <<<

// =============================================================================
// Section 1: the Pearson p-value
// =============================================================================
static void test_pearson() {
    std::cout << "\n-- Test 1: the Pearson p-value --" << std::endl;

    // 1.1 the pinned fixture, against scipy.
    close_rel(p_value([&] { return pearson_p_value(f1_r, f1_n); }), f1_p_pearson, 1e-11, 1e-300,
              "1.1a F1 the Pearson p-value matches scipy");
    close_rel(p_value([&] { return pearson_p_value(f2_r, f2_n); }), f2_p_pearson, 1e-11, 1e-300,
              "1.1b F2 (tied) the Pearson p-value matches scipy");
    close_rel(p_value([&] { return pearson_p_value(f3_r, f3_n); }), f3_p_pearson, 1e-11, 1e-300,
              "1.1c F3 (n = 20) the Pearson p-value matches scipy");
    close_rel(p_value([&] { return pearson_p_value(f4_r, f4_n); }), f4_p_pearson, 1e-11, 1e-300,
              "1.1d F4 (anti-correlated) the Pearson p-value matches scipy");
    close_rel(p_value([&] { return pearson_p_value(f5_r, f5_n); }), f5_p_pearson, 1e-11, 1e-300,
              "1.1e F5 (heavy ties) the Pearson p-value matches scipy");
    close_rel(p_value([&] { return pearson_p_value(f6_r, f6_n); }), f6_p_pearson, 1e-11, 1e-300,
              "1.1f F6 the Pearson p-value matches scipy");

    // 1.2 THE INDEPENDENT REFERENCE. The library composes the repo's
    //     `regularized_incomplete_beta`; this file has its own.
    close_rel(p_value([&] { return pearson_p_value(0.5, 10); }), ref_correlation_p(0.5, 10), 1e-11, 1e-300,
              "1.2a the Pearson p-value agrees with the INDEPENDENT reference "
              "(r = 0.5, n = 10)");
    close_rel(p_value([&] { return pearson_p_value(-0.8, 25); }), ref_correlation_p(-0.8, 25), 1e-11, 1e-300,
              "1.2b the Pearson p-value agrees with the INDEPENDENT reference "
              "(r = -0.8, n = 25)");
    close_rel(p_value([&] { return pearson_p_value(0.05, 100); }), ref_correlation_p(0.05, 100), 1e-11, 1e-300,
              "1.2c the Pearson p-value agrees with the INDEPENDENT reference "
              "(r = 0.05, n = 100)");

    // 1.3 THE BOUNDARIES, and they are DIFFERENT answers.
    check(p_value([&] { return pearson_p_value(0.0, 5); }) == 1.0,
          "1.3a the p-value at r == 0 is EXACTLY 1.0");
    check(p_value([&] { return pearson_p_value(1.0, 5); }) == 0.0,
          "1.3b the p-value at r == 1 is EXACTLY 0.0");
    check(p_value([&] { return pearson_p_value(-1.0, 5); }) == 0.0,
          "1.3c the p-value at r == -1 is EXACTLY 0.0");

    // 1.4 THE CLIP (C4). Without it, the p-value of a PERFECT correlation
    //     is NaN -- the most important fixture in the module, and the one
    //     that breaks.
    check(p_value([&] { return pearson_p_value(kStatisticAboveOne, 5); }) == 0.0,
          "1.4a a statistic ABOVE 1 gives a p-value of 0.0, NOT NaN");
    check(p_value([&] { return pearson_p_value(-kStatisticAboveOne, 5); }) == 0.0,
          "1.4b a statistic BELOW -1 gives a p-value of 0.0, NOT NaN");

    // 1.5 |r| JUST BELOW 1 IS NOT |r| == 1 (C7-bis). The p-value is a
    //     steep function of the statistic near the boundary, so these are
    //     DIFFERENT answers.
    const double just_below = 0.9999999999999999;
    const double p_below = p_value([&] { return pearson_p_value(just_below, 5); });
    check(p_below > 0.0 && p_below < 1e-20,
          "1.5a a statistic of 1-1ulp gives a FINITE tiny p-value, not 0.0 "
          "(got " + std::to_string(p_below) + ")");
    check(p_below != p_value([&] { return pearson_p_value(1.0, 5); }),
          "1.5b the 1-ulp case and the exact case are DIFFERENT answers");

    // 1.6 n == 2 IS NaN (C3).
    // **AND IT MUST NOT THROW**: a degenerate statistic is not an error.
    // The `p_value_no_throw` helper reports a throw as a FAILED CHECK,
    // because a NaN and a thrown exception are DIFFERENT answers and the
    // suite must be able to tell them apart (the mutation harness found
    // this: M7 "the n == 2 case THROWS" survived when the helper
    // swallowed the distinction).
    check(std::isnan(p_value_no_throw([&] { return pearson_p_value(1.0, 2); },
                                    "1.6 the Pearson p-value is NaN for n == 2")),
          "1.6 the Pearson p-value is NaN for n == 2");

    // 1.7 the p-value is TWO-SIDED (C6): the sign of r does not change it.
    close_rel(p_value([&] { return pearson_p_value(0.7, 12); }), p_value([&] { return pearson_p_value(-0.7, 12); }), 1e-15, 1e-300,
              "1.7 the p-value is TWO-SIDED: r and -r give the same p-value");

    // 1.8 THE ONE-SIDED VALUE IS DIFFERENT, AND THE SUITE MUST PIN WHICH
    //     ONE IT SHIPS. A one-sided p-value is a DIFFERENT QUESTION, and
    //     the module says plainly that it ships the two-sided one.
    //
    //     **THE MUTATION HARNESS FOUND THIS GAP.** The mutation "the
    //     p-value is one-sided (the sign of the statistic is dropped)"
    //     SURVIVED, because the shipped formula computes
    //     `x = dof/(dof + at*at)` -- and `at*at` is the SAME for +t and
    //     -t, so the sign of the statistic is UNOBSERVABLE in the
    //     formula itself. **THE PROPERTY THAT IS OBSERVABLE IS THE
    //     VALUE, NOT THE SIGN**, so the check belongs here.
    //
    //     The one-sided UPPER-tail p-value for a positive correlation is
    //     HALF the two-sided one (the two-sided form splits the tail at
    //     zero and doubles each half). So the one-sided value is
    //     DISTINGUISHABLE from the two-sided one, and this pins that the
    //     module ships the two-sided one.
    const double two_sided = p_value([&] { return pearson_p_value(0.7, 12); });
    const double one_sided = two_sided * 0.5;
    check(std::fabs(two_sided - one_sided) > 1e-6,
          "1.8a the ONE-SIDED p-value is a DISTINCT number from the "
          "two-sided one, so the choice is observable (two-sided " +
          std::to_string(two_sided) + ", one-sided " +
          std::to_string(one_sided) + ")");
    check(std::fabs(two_sided / one_sided - 2.0) < 1e-9,
          "1.8b the two-sided p-value is EXACTLY twice the one-sided one "
          "here, which is what makes the two conventions distinguishable");

    // 1.9 THE p-VALUE IS A PROBABILITY (C7 / the clamp): it is in [0, 1]
    //     for every fixture and every boundary.
    bool all_in_range = true;
    for (size_t n = 3; n <= 30; ++n) {
        for (int k = -20; k <= 20; ++k) {
            const double stat = static_cast<double>(k) / 10.0;
            const double pv = p_value([&] { return pearson_p_value(stat, n); });
            if (!(pv >= 0.0 && pv <= 1.0)) all_in_range = false;
        }
    }
    check(all_in_range,
          "1.9 the Pearson p-value is in [0, 1] for every statistic and "
          "sample size in the sweep");
}

// =============================================================================
// Section 2: the Spearman p-value
// =============================================================================
static void test_spearman() {
    std::cout << "\n-- Test 2: the Spearman p-value --" << std::endl;

    // 2.1 the pinned fixture, against scipy.
    close_rel(p_value([&] { return spearman_p_value(f1_rho, f1_n); }), f1_p_spearman, 1e-11, 1e-300,
              "2.1a F1 the Spearman p-value matches scipy");
    close_rel(p_value([&] { return spearman_p_value(f2_rho, f2_n); }), f2_p_spearman, 1e-11, 1e-300,
              "2.1b F2 (tied) the Spearman p-value matches scipy");
    close_rel(p_value([&] { return spearman_p_value(f3_rho, f3_n); }), f3_p_spearman, 1e-11, 1e-300,
              "2.1c F3 (n = 20) the Spearman p-value matches scipy");
    close_rel(p_value([&] { return spearman_p_value(f4_rho, f4_n); }), f4_p_spearman, 1e-11, 1e-300,
              "2.1d F4 (anti-correlated) the Spearman p-value matches scipy");
    close_rel(p_value([&] { return spearman_p_value(f5_rho, f5_n); }), f5_p_spearman, 1e-11, 1e-300,
              "2.1e F5 (heavy ties) the Spearman p-value matches scipy");
    close_rel(p_value([&] { return spearman_p_value(f6_rho, f6_n); }), f6_p_spearman, 1e-11, 1e-300,
              "2.1f F6 the Spearman p-value matches scipy");

    // 2.2 THE INDEPENDENT REFERENCE.
    close_rel(p_value([&] { return spearman_p_value(0.5, 10); }), ref_correlation_p(0.5, 10), 1e-11, 1e-300,
              "2.2a the Spearman p-value agrees with the INDEPENDENT reference");
    close_rel(p_value([&] { return spearman_p_value(-0.6, 30); }), ref_correlation_p(-0.6, 30), 1e-11, 1e-300,
              "2.2b the Spearman p-value agrees with the INDEPENDENT reference");

    // 2.3 THE FORM IS THE PEARSON FORM (C2): the same t statistic, with
    //     rho in place of r. If the two ever disagreed, one of them would
    //     be using a different formula.
    for (size_t n = 3; n <= 40; ++n) {
        for (int k = -8; k <= 8; ++k) {
            const double stat = static_cast<double>(k) / 10.0;
            close_rel(p_value([&] { return spearman_p_value(stat, n); }), p_value([&] { return pearson_p_value(stat, n); }),
                      1e-15, 1e-300,
                      "2.3 the Spearman and Pearson p-values share ONE formula "
                      "(rho == r = " + std::to_string(stat) + ", n = " +
                      std::to_string(n) + ")");
        }
    }

    // 2.4 the boundaries and the clip, as for Pearson.
    check(p_value([&] { return spearman_p_value(0.0, 5); }) == 1.0,
          "2.4a the Spearman p-value at rho == 0 is EXACTLY 1.0");
    check(p_value([&] { return spearman_p_value(1.0, 5); }) == 0.0,
          "2.4b the Spearman p-value at rho == 1 is EXACTLY 0.0");
    check(p_value([&] { return spearman_p_value(kStatisticAboveOne, 5); }) == 0.0,
          "2.4c the Spearman p-value clips a statistic above 1 to 0.0, NOT NaN");
    check(std::isnan(p_value_no_throw([&] { return spearman_p_value(1.0, 2); },
                                       "2.4d the Spearman p-value is NaN for n == 2")),
          "2.4d the Spearman p-value is NaN for n == 2");
}

// =============================================================================
// Section 3: the Kendall p-value
// =============================================================================
static void test_kendall() {
    std::cout << "\n-- Test 3: the Kendall p-value --" << std::endl;

    // 3.1 the pinned fixture, against scipy's method='asymptotic'.
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f1_x), col(f1_y)); }), f1_p_kendall_asym,
              1e-11, 1e-300,
              "3.1a F1 the Kendall p-value matches scipy (asymptotic)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f2_x), col(f2_y)); }), f2_p_kendall_asym,
              1e-11, 1e-300,
              "3.1b F2 (tied) the Kendall p-value matches scipy (asymptotic)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f3_x), col(f3_y)); }), f3_p_kendall_asym,
              1e-11, 1e-300,
              "3.1c F3 (n = 20) the Kendall p-value matches scipy (asymptotic)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f4_x), col(f4_y)); }), f4_p_kendall_asym,
              1e-11, 1e-300,
              "3.1d F4 the Kendall p-value matches scipy (asymptotic)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f5_x), col(f5_y)); }), f5_p_kendall_asym,
              1e-11, 1e-300,
              "3.1e F5 (heavy ties) the Kendall p-value matches scipy (asymptotic)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f6_x), col(f6_y)); }), f6_p_kendall_asym,
              1e-11, 1e-300,
              "3.1f F6 the Kendall p-value matches scipy (asymptotic)");

    // 3.2 THE INDEPENDENT REFERENCE.
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f5_x), col(f5_y)); }),
              ref_kendall_p(f5_x, f5_y), 1e-11, 1e-300,
              "3.2a the Kendall p-value agrees with the INDEPENDENT reference "
              "(the heavy-tie fixture)");
    close_rel(p_value([&] { return kendall_tau_b_p_value(col(f2_x), col(f2_y)); }),
              ref_kendall_p(f2_x, f2_y), 1e-11, 1e-300,
              "3.2b the Kendall p-value agrees with the INDEPENDENT reference "
              "(the tied fixture)");

    // 3.3 THE TIE-GROUP STATISTICS ARE NOT THE TIE PAIR COUNT (C2).
    //     F5's x has groups {4,2} -> 7 tied pairs; F6's x has groups
    //     {3,3,2} -> 7 tied pairs. The pair count is IDENTICAL and the
    //     variances are not, so a variance built from the pair count alone
    //     gives the same answer for both -- and the wrong one for one.
    double p5, x0, x1;
    ref_tie_groups(f5_x, &p5, &x0, &x1);
    double p6, x06, x16;
    ref_tie_groups(f6_x, &p6, &x06, &x16);
    check(p5 == p6,
          "3.3a F5 and F6 have the SAME tie PAIR count in x (" +
          std::to_string(p5) + " vs " + std::to_string(p6) + ")");
    check(x0 != x06 || x1 != x16,
          "3.3b but DIFFERENT tie-GROUP statistics, so the variance differs");
    const double k5 = p_value([&] { return kendall_tau_b_p_value(col(f5_x), col(f5_y)); });
    const double k6 = p_value([&] { return kendall_tau_b_p_value(col(f6_x), col(f6_y)); });
    check(std::fabs(k5 - k6) > 1e-12,
          "3.3c so the two Kendall p-values DIFFER (" + std::to_string(k5) +
          " vs " + std::to_string(k6) + ") -- a variance built from the pair "
          "count alone would give the same answer for both");

    // 3.4 THE TIE TERMS IN THE VARIANCE ARE LOAD-BEARING.
    {
        // The variance with the two tie terms dropped.
        const size_t n = f5_x.size();
        const double dn = static_cast<double>(n);
        const double m = dn * (dn - 1.0);
        double tot = 0.5 * dn * (dn - 1.0), xt = 0.0, yt = 0.0, nt = 0.0, dis = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double dx = f5_x[i] - f5_x[j];
                const double dy = f5_y[i] - f5_y[j];
                if (dx == 0.0 && dy == 0.0) { xt += 1.0; yt += 1.0; nt += 1.0; }
                else if (dx == 0.0) { xt += 1.0; }
                else if (dy == 0.0) { yt += 1.0; }
                else if ((dx > 0.0) != (dy > 0.0)) { dis += 1.0; }
            }
        }
        double xp, xx0, xx1, yp, yy0, yy1;
        ref_tie_groups(f5_x, &xp, &xx0, &xx1);
        ref_tie_groups(f5_y, &yp, &yy0, &yy1);
        const double cmd = tot - xt - yt + nt - 2.0 * dis;
        const double var_partial = (m * (2.0 * dn + 5.0) - xx1 - yy1) / 18.0;
        const double z_partial = cmd / std::sqrt(var_partial);
        const double p_partial = std::erfc(std::fabs(z_partial) / std::sqrt(2.0));
        check(std::fabs(k5 - p_partial) > 1e-9,
              "3.4 the tie terms in the Kendall variance are LOAD-BEARING "
              "(full " + std::to_string(k5) + ", without them " +
              std::to_string(p_partial) + ")");
    }

    // 3.5 n == 2 IS NaN (C3).
    check(std::isnan(p_value_no_throw([&] { return kendall_tau_b_p_value(col({1.0, 2.0}), col({2.0, 1.0})); },
                                       "3.5 the Kendall p-value is NaN for n == 2")),
          "3.5 the Kendall p-value is NaN for n == 2");

    // 3.6 a CONSTANT sample has an undefined correlation, so NaN (C5).
    check(std::isnan(p_value([&] { return kendall_tau_b_p_value(col({1.0, 1.0, 1.0, 1.0}), col({1.0, 2.0, 3.0, 4.0})); })),
          "3.6 the Kendall p-value is NaN for a constant sample");

    // 3.7 THE p-VALUE IS A PROBABILITY (the clamp). The mutation "the
    //     p-value is not clamped to [0, 1]" SURVIVED the first harness
    //     run, because `erfc` already returns a value in [0, 1] -- so on
    //     the SHIPPED path the clamp never fires. **THE CLAMP IS NOT
    //     THEREFORE OBSOLETE: it is load-bearing for the OTHER route the
    //     mutation harness found** (the `2*(1 - normal_cdf(|z|))` form,
    //     whose arithmetic can exceed 1 at the boundary), and it is the
    //     clamp that keeps that route a probability. This check pins the
    //     PROPERTY the clamp guarantees.
    bool kendall_in_range = true;
    for (size_t n = 3; n <= 25; ++n) {
        for (int k = -10; k <= 10; ++k) {
            std::vector<double> x, y;
            for (size_t i = 0; i < n; ++i) {
                x.push_back(static_cast<double>(i));
                y.push_back(static_cast<double>(i) * 0.5 + static_cast<double>(k) * 0.01 +
                            static_cast<double>(i * i % 3));
            }
            const double pv = p_value([&] { return kendall_tau_b_p_value(col(x), col(y)); });
            if (!std::isnan(pv) && !(pv >= 0.0 && pv <= 1.0)) kendall_in_range = false;
        }
    }
    check(kendall_in_range,
          "3.7 the Kendall p-value is in [0, 1] over the whole sweep");
}

// =============================================================================
// Section 4: the convenience wrappers
// =============================================================================
static void test_wrappers() {
    std::cout << "\n-- Test 4: the convenience wrappers --" << std::endl;

    // 4.1 THE WRAPPER'S STATISTIC IS THE REPO'S OWN (C8): one source of
    //     truth for the coefficient.
    Tensor x = col(f3_x);
    Tensor y = col(f3_y);
    nn::utils::CorrelationTestResult pr = test_call([&] { return pearson_r_test(x, y); },
                                                     "4.1 pearson_r_test did not throw");
    check(pr.statistic == pearson_r(x, y),
          "4.1a pearson_r_test's statistic IS the repo's pearson_r");
    check(pr.p_value == p_value([&] { return pearson_p_value(pr.statistic, pr.n); }),
          "4.1b pearson_r_test's p-value IS pearson_p_value of that statistic");

    nn::utils::CorrelationTestResult sr = test_call([&] { return spearman_rho_test(x, y); },
                                                     "4.1 spearman_rho_test did not throw");
    check(sr.statistic == spearman_rho(x, y),
          "4.1c spearman_rho_test's statistic IS the repo's spearman_rho");
    check(sr.p_value == p_value([&] { return spearman_p_value(sr.statistic, sr.n); }),
          "4.1d spearman_rho_test's p-value IS spearman_p_value of it");

    nn::utils::CorrelationTestResult kr = test_call([&] { return kendall_tau_b_test(x, y); },
                                                     "4.1 kendall_tau_b_test did not throw");
    check(kr.statistic == kendall_tau_b(x, y),
          "4.1e kendall_tau_b_test's statistic IS the repo's kendall_tau_b");
    check(kr.p_value == p_value([&] { return kendall_tau_b_p_value(x, y); }),
          "4.1f kendall_tau_b_test's p-value IS kendall_tau_b_p_value of them");

    // 4.2 the wrapper's p-value matches the ORACLE's.
    close_rel(pr.p_value, f3_p_pearson, 1e-11, 1e-300,
              "4.2a the wrapper's Pearson p-value matches the oracle");
    close_rel(sr.p_value, f3_p_spearman, 1e-11, 1e-300,
              "4.2b the wrapper's Spearman p-value matches the oracle");
    close_rel(kr.p_value, f3_p_kendall_asym, 1e-11, 1e-300,
              "4.2c the wrapper's Kendall p-value matches the oracle");

    // 4.3 the sample size is reported.
    check(pr.n == f3_n && sr.n == f3_n && kr.n == f3_n,
          "4.3 the wrappers report the sample size");

    // 4.4 THE SHAPE DOES NOT MATTER: the VALUES are what count, so a
    //     (1,N) tensor gives the same answers as a (N,1) one.
    nn::utils::CorrelationTestResult rr = test_call([&] { return pearson_r_test(row(f3_x), row(f3_y)); },
                                                      "4.4 pearson_r_test did not throw");
    check(rr.statistic == pr.statistic && rr.p_value == pr.p_value,
          "4.4 a (1,N) sample gives the same Pearson test as a (N,1) one");

    // 4.5 A CONSTANT SAMPLE: the statistic is NaN and so is the p-value.
    nn::utils::CorrelationTestResult cr = test_call(
        [&] { return pearson_r_test(col({1.0, 1.0, 1.0, 1.0}), col({1.0, 2.0, 3.0, 4.0})); },
        "4.5 pearson_r_test did not throw");
    check(std::isnan(cr.statistic) && std::isnan(cr.p_value),
          "4.5 a constant sample gives a NaN statistic and a NaN p-value");
}

// =============================================================================
// Section 5: the guards
// =============================================================================
static void test_guards() {
    std::cout << "\n-- Test 5: the guards --" << std::endl;

    // 5.1 n < 2 IS AN ERROR, NOT A NaN (C4-bis in the oracle): a sample
    //     with fewer than two points cannot even have a correlation.
    expect_throw([&] { pearson_p_value(0.5, 1); }, "at least two",
                 "5.1a pearson_p_value rejects n == 1");
    expect_throw([&] { spearman_p_value(0.5, 0); }, "at least two",
                 "5.1b spearman_p_value rejects n == 0");

    // 5.2 the sample-length mismatch.
    expect_throw([&] { kendall_tau_b_p_value(col({1.0, 2.0, 3.0}), col({1.0, 2.0})); },
                 "same number of samples",
                 "5.2a kendall_tau_b_p_value rejects a length mismatch");

    // 5.3 the wrappers' guards.
    expect_throw([&] { pearson_r_test(col({1.0}), col({2.0})); }, "at least two",
                 "5.3a pearson_r_test rejects a one-point sample");
    expect_throw([&] { spearman_rho_test(col({1.0, 2.0}), col({1.0})); },
                 "same number of samples",
                 "5.3b spearman_rho_test rejects a length mismatch");
    expect_throw([&] { kendall_tau_b_test(col({1.0, 2.0}), col({1.0})); },
                 "same number of samples",
                 "5.3c kendall_tau_b_test rejects a length mismatch");

    // 5.4 EVERY GUARD'S MESSAGE IS DISTINCT. A needle that appears in two
    //     different messages discriminates nothing between them, and this
    //     repo has hit that trap repeatedly.
    std::vector<std::string> messages;
    const std::pair<const char*, std::function<void()>> cases[] = {
        {"pearson n", [&] { pearson_p_value(0.5, 1); }},
        {"spearman n", [&] { spearman_p_value(0.5, 1); }},
        {"kendall length", [&] { kendall_tau_b_p_value(col({1.0, 2.0, 3.0}), col({1.0, 2.0})); }},
        {"pearson_r_test n", [&] { pearson_r_test(col({1.0}), col({2.0})); }},
        {"spearman_rho_test length", [&] { spearman_rho_test(col({1.0, 2.0}), col({1.0})); }},
        {"kendall_tau_b_test length", [&] { kendall_tau_b_test(col({1.0, 2.0}), col({1.0})); }},
    };
    for (const auto& c : cases) {
        std::string what;
        threw_with(c.second, "", &what);
        messages.push_back(what);
    }
    bool all_distinct = true;
    for (size_t i = 0; i < messages.size(); ++i)
        for (size_t j = i + 1; j < messages.size(); ++j)
            if (messages[i] == messages[j]) all_distinct = false;
    check(all_distinct,
          "5.4 every guard's message is DISTINCT from every other one");
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "tests/test_correlation_tests -- the correlation hypothesis "
                 "tests\n";
    std::cout << "============================================================\n";

    test_pearson();
    test_spearman();
    test_kendall();
    test_wrappers();
    test_guards();

    std::cout << "\n============================================================\n";
    std::cout << "=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    std::cout << "============================================================\n";
    return g_fail == 0 ? 0 : 1;
}
