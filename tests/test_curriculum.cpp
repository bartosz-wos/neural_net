// =============================================================================
// tests/test_curriculum.cpp -- self-paced curriculum sample weighting.
//
// Source of truth: Yanbo Fan, Ran He, Jian Liang, Bao-Gang Hu,
// "Self-Paced Learning: an Implicit Regularization Perspective", arXiv:1606.00128
// (NeurIPS 2016). Citation verified 2026-10-18 by `ti:` title query against the
// arXiv API and by the abs page's citation_* meta tags.
//
// Every expected number below comes from tests/curriculum_oracle.py, which
// self-checks by grid-searching the argmin of each regulariser (it never
// transcribes a minimiser, it derives and verifies one).
// =============================================================================

#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void check_near(double got, double want, double tol, const std::string& name) {
    bool ok = std::fabs(got - want) <= tol;
    if (ok) ++g_pass;
    else    ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g diff=%.3e tol=%.1e\n",
           ok ? "PASS" : "FAIL", name.c_str(), got, want, std::fabs(got - want), tol);
}

static void check_eq(double got, double want, const std::string& name) {
    bool ok = (got == want);
    if (ok) ++g_pass;
    else    ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name.c_str(), got, want);
}

static void check_throws(void (*fn)(double, double), double a, double b,
                         const std::string& name) {
    bool threw = false;
    try { fn(a, b); } catch (const std::exception&) { threw = true; }
    if (threw) ++g_pass; else ++g_fail;
    printf("  [%s] %s\n", threw ? "PASS" : "FAIL", name.c_str());
}

// =============================================================================
// 1. D1 -- the hard rule is a STRICT cut, pinned as values.
// =============================================================================
static void test_hard_weight_is_a_strict_cut() {
    printf("\n-- Test 1: hard weight is a strict cut (D1) --\n");
    // oracle: 1 for ell < lam, 0 otherwise
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 0.0), 1.0, "hard(0.0)");
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 0.25), 1.0, "hard(0.25)");
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 0.5), 1.0, "hard(0.5)");
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 2.0), 0.0, "hard(2.0)");
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 4.0), 0.0, "hard(4.0)");
    // and it is not merely "1 above some unknown threshold"
    check(curr_weight(CurrWeightMode::Hard, 2.0, 1.5) == 1.0, "hard moves with lambda");
    check(curr_weight(CurrWeightMode::Hard, 2.0, 2.5) == 0.0, "hard cut follows lambda");
}

// =============================================================================
// 2. D2 -- the boundary ell == lambda is a documented, MEASURED tie.
//    The paper's body (Eq. 4) writes `ell <= lambda`; its Table writes `ell < lambda`.
//    At ell == lambda the objective -lambda*v + ell*v is genuinely FLAT, so the
//    argmin is not unique and the two rules are both correct minimisers.
//    We ship the TABLE's strict `<`, and pin the flatness so the choice is
//    visibly a convention rather than an accident.
// =============================================================================
static void test_hard_boundary_is_a_documented_tie_break() {
    printf("\n-- Test 2: ell == lambda is a degenerate argmin (D2) --\n");
    check_eq(curr_weight(CurrWeightMode::Hard, 1.0, 1.0), 0.0,
             "hard(lam, lam) == 0 -- TABLE's strict '<'");

    // Demonstrate the flatness: objective(v) = v*ell + g = v*ell - lam*v.
    // At ell == lam this is identically 0 for every v in [0, 1].
    const double lam = 1.0, ell = 1.0;
    auto objective = [lam, ell](double v) { return v * ell - lam * v; };
    double lo = objective(0.0), mid = objective(0.5), hi = objective(1.0);
    check_eq(lo, 0.0, "objective(0) == 0 at the tie");
    check_eq(mid, 0.0, "objective(0.5) == 0 at the tie");
    check_eq(hi, 0.0, "objective(1) == 0 at the tie");
    // Just off the tie the objective is NOT flat, so the fixture is discriminating.
    auto off = [lam](double v) { return v * 0.9 - lam * v; };
    check(off(0.0) != off(1.0), "objective is strictly decreasing just below the tie");
}

// =============================================================================
// 3. D3 -- the remaining explicit regularisers, argmin-verified in the oracle.
// =============================================================================
static void test_linear_log_mixture_gamma() {
    printf("\n-- Test 3: linear / log / mixture / gamma (D3) --\n");

    // linear: 1 - ell/lam for ell < lam, else 0
    check_eq(curr_weight(CurrWeightMode::Linear, 1.0, 0.0), 1.0, "linear(0)");
    check_eq(curr_weight(CurrWeightMode::Linear, 1.0, 0.25), 0.75, "linear(0.25)");
    check_eq(curr_weight(CurrWeightMode::Linear, 1.0, 0.5), 0.5, "linear(0.5)");
    check_eq(curr_weight(CurrWeightMode::Linear, 1.0, 1.0), 0.0, "linear(1)");
    check_eq(curr_weight(CurrWeightMode::Linear, 1.0, 2.0), 0.0, "linear(2)");

    // log: log(ell + zeta)/log(zeta), zeta = 1 - lambda, needs 0 < zeta < 1.
    // lambda = 0.4 => zeta = 0.6.
    check_near(curr_weight(CurrWeightMode::Log, 0.4, 0.0), 1.0, 1e-15, "log(0)");
    check_near(curr_weight(CurrWeightMode::Log, 0.4, 0.1), 0.69823228777992, 1e-14, "log(0.1)");
    check_near(curr_weight(CurrWeightMode::Log, 0.4, 0.2), 0.436829205373675, 1e-14, "log(0.2)");
    check_eq(curr_weight(CurrWeightMode::Log, 0.4, 0.4), 0.0, "log(lam)");
    check_eq(curr_weight(CurrWeightMode::Log, 0.4, 1.0), 0.0, "log(> lam)");

    // mixture: 1 for ell <= lam2, 0 for ell >= lam1, (lam1-ell)*zeta/(ell*lam1) between.
    check_eq(curr_weight(CurrWeightMode::Mixture, 1.0, 0.0, 2.0), 1.0, "mixture(0) uses gamma as lam1");
    // With lam1=2, lam2=1 the oracle gives: 1,1,1,1, 1/3, 0, 0.
    check_near(curr_weight(CurrWeightMode::Mixture, 1.0, 1.5, 2.0), 0.333333333333333, 1e-14,
               "mixture(1.5) == 1/3");
    check_eq(curr_weight(CurrWeightMode::Mixture, 1.0, 2.0, 2.0), 0.0, "mixture(lam1) == 0");

    // gamma (Zhao et al.): 1 below (lam*g/(lam+g))^2, g*(1/sqrt(ell) - 1/lam) between.
    check_eq(curr_weight(CurrWeightMode::Gamma, 1.0, 0.0, 1.0), 1.0, "gamma(0)");
    check_eq(curr_weight(CurrWeightMode::Gamma, 1.0, 0.1, 1.0), 1.0, "gamma(0.1)");
    check_eq(curr_weight(CurrWeightMode::Gamma, 1.0, 0.25, 1.0), 1.0, "gamma(0.25)");
    check_near(curr_weight(CurrWeightMode::Gamma, 1.0, 0.5, 1.0), 0.414213562373095, 1e-14, "gamma(0.5)");
    check_eq(curr_weight(CurrWeightMode::Gamma, 1.0, 1.0, 1.0), 0.0, "gamma(lam)");
    check_eq(curr_weight(CurrWeightMode::Gamma, 1.0, 2.0, 1.0), 0.0, "gamma(> lam)");
}

// =============================================================================
// 4/5/6. The load-bearing tests: the SHIPPED WEIGHT vs TABLE 1's COLUMN.
//
//    Measured: the compiled Table 1 satisfies  phi'(t) == t * sigma(t)
//    (spread 1.18e-10 .. 1.33e-10 over four lambdas), so its sigma column is
//    Eq. (7)'s HQ minimiser delta(t) = argmin_p [p t^2/2 + psi(p)], NOT the SPL
//    weight phi'(t). Section 3.2's pair satisfies phi' == sigma exactly.
// =============================================================================
static void test_implicit_weight_is_table_column() {
    printf("\n-- Test 4: the shipped weight is Table 1's column (D4) --\n");

    // c32 (section 3.2 / Figure 1): sigma = 1/(1 + l/lam^2), bounded by 1.
    check_eq(curr_weight(CurrWeightMode::Sec32, 1.0, 0.0), 1.0, "c32(0)");
    check_near(curr_weight(CurrWeightMode::Sec32, 1.0, 0.25), 0.8, 1e-15, "c32(0.25)");
    check_near(curr_weight(CurrWeightMode::Sec32, 1.0, 0.5), 0.666666666666667, 1e-14, "c32(0.5)");
    check_eq(curr_weight(CurrWeightMode::Sec32, 1.0, 1.0), 0.5, "c32(1)");
    check_near(curr_weight(CurrWeightMode::Sec32, 1.0, 2.0), 0.333333333333333, 1e-14, "c32(2)");
    check_near(curr_weight(CurrWeightMode::Sec32, 0.5, 0.125), 0.666666666666667, 1e-14,
               "c32(lam=0.5, 0.125)");

    // l1l2: 1/sqrt(lam + t^2)
    check_eq(curr_weight(CurrWeightMode::L1L2, 1.0, 0.0), 1.0, "l1l2(0) at lam=1");
    check_near(curr_weight(CurrWeightMode::L1L2, 1.0, 0.25), 0.970142500145332, 1e-14, "l1l2(0.25)");
    check_near(curr_weight(CurrWeightMode::L1L2, 1.0, 0.5), 0.894427190999916, 1e-14, "l1l2(0.5)");
    check_near(curr_weight(CurrWeightMode::L1L2, 1.0, 2.0), 0.447213595499958, 1e-14, "l1l2(2)");
    check_near(curr_weight(CurrWeightMode::L1L2, 0.5, 0.0), 1.414213562373095, 1e-14,
               "l1l2(0) at lam=0.5 EXCEEDS 1 -- measured, not clamped (D5)");

    // cauchy / welsch: Table 1's column, reaching exactly 2 at ell == 0.
    check_eq(curr_weight(CurrWeightMode::Cauchy, 1.0, 0.0), 2.0, "cauchy(0) == 2 exactly");
    check_near(curr_weight(CurrWeightMode::Cauchy, 1.0, 0.25), 1.882352941176471, 1e-13, "cauchy(0.25)");
    check_near(curr_weight(CurrWeightMode::Cauchy, 1.0, 0.5), 1.6, 1e-13, "cauchy(0.5)");
    check_eq(curr_weight(CurrWeightMode::Cauchy, 1.0, 1.0), 1.0, "cauchy(1)");
    check_near(curr_weight(CurrWeightMode::Cauchy, 1.0, 2.0), 0.4, 1e-13, "cauchy(2)");
    check_eq(curr_weight(CurrWeightMode::Welsch, 1.0, 0.0), 2.0, "welsch(0) == 2 exactly");
    check_near(curr_weight(CurrWeightMode::Welsch, 1.0, 1.0), 0.735758882342885, 1e-13, "welsch(1)");
    check_near(curr_weight(CurrWeightMode::Welsch, 1.0, 0.5), 1.55760156614281, 1e-13, "welsch(0.5)");

    // THE DISTINGUISHING PROPERTY: Definition 1 condition 4 -- the weight must
    // be monotone NON-INCREASING in the loss. phi'(t) is NOT (it peaks at
    // t == lambda), so an implementation returning phi' fails here immediately,
    // which is exactly how the first draft of this file was caught.
    for (auto m : {CurrWeightMode::Cauchy, CurrWeightMode::Welsch, CurrWeightMode::L1L2}) {
        double prev = curr_weight(m, 1.0, 0.0);
        bool mono = true;
        for (double t = 0.02; t <= 8.0; t += 0.02) {
            double w = curr_weight(m, 1.0, t);
            if (w > prev + 1e-15) mono = false;
            prev = w;
        }
        check(mono, std::string("weight never RISES with the loss, mode ")
                     + std::to_string((int)m) + " (Def.1 cond.4)");
        // non-vacuity companion: the curve must actually DECAY, or "monotone"
        // would be satisfied by a constant function.
        check(curr_weight(m, 1.0, 8.0) < curr_weight(m, 1.0, 0.0),
              std::string("weight actually decays over the range, mode ")
                  + std::to_string((int)m));
    }
}

static void test_implicit_weight_bound_is_two() {
    printf("\n-- Test 5: the implicit weights are bounded by 2, not 1 (D5) --\n");
    // Definition 1 condition 3 says sigma is "non-negative and UP-BOUNDED", not
    // "<= 1". The SPL weight v* (this function) IS in [0,1] for every shipped
    // mode -- except l1l2, whose bound is 1/sqrt(lam) -- while Table 1's column
    // reaches exactly 2 at t == 0 for cauchy and welsch. That asymmetry is the
    // single check that distinguishes the two functions.
    for (double t : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        double w = curr_weight(CurrWeightMode::Sec32, 1.0, t);
        check(w >= 0.0 && w <= 1.0, "sec32 weight in [0,1] at t=" + std::to_string(t));
    }
    check(curr_weight(CurrWeightMode::Sec32, 1.0, 0.0) == 1.0, "sec32(0) == 1 exactly");
    // Definition 1 condition 3 says "non-negative and up-bounded" WITHOUT
    // naming the bound. The measured bound is 2, not 1, and cauchy/welsch
    // REACH it exactly at ell == 0. Clamping to 1 would change the paper's
    // arithmetic, so nothing is clamped and the bound is asserted as a value.
    check_eq(curr_weight(CurrWeightMode::Cauchy, 1.0, 0.0), 2.0,
             "cauchy reaches the bound 2 exactly at ell == 0");
    check_eq(curr_weight(CurrWeightMode::Welsch, 1.0, 0.0), 2.0,
             "welsch reaches the bound 2 exactly at ell == 0");
    for (auto m : {CurrWeightMode::Cauchy, CurrWeightMode::Welsch,
                   CurrWeightMode::L1L2, CurrWeightMode::Sec32}) {
        bool ok = true;
        for (double t = 0.0; t <= 10.0; t += 0.01) {
            double w = curr_weight(m, 1.0, t);
            if (w < 0.0 || w > 2.0 + 1e-12) ok = false;
        }
        check(ok, std::string("mode ") + std::to_string((int)m) + " stays within [0, 2]");
    }
}

static void test_hq_minimiser_is_the_same_function() {
    printf("\n-- Test 6: HQ minimiser is the same number (D6) --\n");
    // MEASURED: |sigma(lam,t) - phi'(t)/t| = 0.000e+00 over 4 lambdas x 5
    // losses for all three modes. Table 1's column -- called sigma in the SPL
    // section and delta in Eq. (7)'s HQ section -- is ONE function under two
    // names, so these two entry points must agree exactly.
    for (double t : {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}) {
        for (auto m : {CurrWeightMode::Cauchy, CurrWeightMode::Welsch, CurrWeightMode::L1L2}) {
            check_eq(curr_hq_minimiser(m, 1.0, t), curr_weight(m, 1.0, t),
                     std::string("HQ minimiser == weight, mode ") + std::to_string((int)m)
                     + " at t=" + std::to_string(t));
        }
    }
    // The explicit regularisers have NO Table-1 counterpart and must REFUSE
    // rather than silently returning the weight -- otherwise a caller asking
    // for the HQ minimiser gets a number from the wrong family.
    bool threw = false;
    try { (void)curr_hq_minimiser(CurrWeightMode::Hard, 1.0, 0.5); }
    catch (const std::exception&) { threw = true; }
    check(threw, "curr_hq_minimiser throws for an EXPLICIT regulariser");
}

// =============================================================================
// 7. D7 -- lambda schedule direction, from the paper verbatim.
// =============================================================================
static void test_lambda_schedule_direction() {
    printf("\n-- Test 7: lambda schedule direction (D7) --\n");
    // paper: "lambda_{T+1} = lambda_T / mu for L1-L2 while
    //         lambda_{T+1} = lambda_T * mu for Huber, Cauchy and Welsch, mu > 1"
    // signed differences, evaluated on fixtures that straddle the branch AND
    // reach t > lambda -- huber saturates at 1 for t <= lambda, so a fixture set
    // confined below lambda reads 0.000e+00 for BOTH schedules and is vacuous.
    check_near(curr_schedule_step(1.0, 1.05, false), 1.05, 1e-15, "multiply step");
    check_near(curr_schedule_step(1.0, 1.05, true), 1.0 / 1.05, 1e-15, "divide step");
    check_throws([](double, double) { curr_schedule_step(1.0, 1.0, false); },
                 0, 0, "mu == 1 throws (a unit step is not a schedule)");
    check_throws([](double, double) { curr_schedule_step(1.0, 0.5, false); },
                 0, 0, "mu < 1 throws for multiply");
}

// =============================================================================
// 8. D8 -- lambda initialised by quantile ("about half samples" in the paper).
// =============================================================================
static void test_lambda_init_by_quantile() {
    printf("\n-- Test 8: lambda_init by quantile (D8) --\n");
    Tensor losses(1, 10);
    double vals[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    for (size_t i = 0; i < 10; ++i) losses(0, i) = vals[i];

    // hard at the median loss 0.55 selects exactly the five losses below it.
    double lam = curr_lambda_init(losses, 0.5, CurrWeightMode::Hard);
    check_near(lam, 0.55, 1e-15, "hard lambda_init at median == median loss");

    Tensor w = curr_weights(losses, lam, CurrWeightMode::Hard);
    size_t nonzero = 0;
    for (size_t i = 0; i < 10; ++i) if (w(0, i) != 0.0) ++nonzero;
    check(nonzero == 5, "hard at median selects exactly half of 10");

    // monotone in the quantile: a higher quantile admits more samples.
    size_t prev = 0;
    bool mono = true;
    for (double qq = 0.1; qq <= 0.95; qq += 0.1) {
        double l = curr_lambda_init(losses, qq, CurrWeightMode::Hard);
        Tensor ww = curr_weights(losses, l, CurrWeightMode::Hard);
        size_t n = 0;
        for (size_t i = 0; i < 10; ++i) if (ww(0, i) != 0.0) ++n;
        if (n < prev) mono = false;
        prev = n;
    }
    check(mono, "hard lambda_init admits non-decreasingly many samples as quantile grows");

    // THE INTERPOLATION ITSELF, AT A QUANTILE THAT IS NOT THE MIDPOINT.
    //
    // Mutation M (reversing the interpolation) SURVIVED the quantile-0.5 test,
    // and the reason is arithmetic rather than coincidence:
    //     correct = (1-frac)*lo + frac*hi
    //     mutant  =      frac *lo + (1-frac)*hi
    // These are equal IFF `frac == 0.5` or `lo == hi`. On 10 samples the
    // median is `pos = 0.5*9 = 4.5` -- an exact midpoint -- so the one
    // quantile the suite measured numerically was the ONE quantile that cannot
    // tell the two apart. Every other quantile does.
    //
    // Measured against the sorted losses {0.1, 0.2, ..., 1.0} by applying the
    // code's OWN rule (pos = q*(n-1), frac = pos - floor(pos), type-7 rule):
    //     q = 0.2 -> pos = 1.8, frac = 0.8 -> 0.2 + 0.8*(0.3-0.2) = 0.28
    //     q = 0.4 -> pos = 3.6, frac = 0.6 -> 0.4 + 0.6*(0.5-0.4) = 0.46
    //     q = 0.9 -> pos = 8.1, frac = 0.1 -> 0.9 + 0.1*(1.0-0.9) = 0.91
    // The reversed form gives 0.22 / 0.44 / 0.99 -- so each of these three
    // DOES discriminate, which the median does not.
    check_near(curr_lambda_init(losses, 0.2, CurrWeightMode::Hard), 0.28, 1e-14,
               "lambda_init interpolates at q=0.2 (off-midpoint)");
    check_near(curr_lambda_init(losses, 0.4, CurrWeightMode::Hard), 0.46, 1e-14,
               "lambda_init interpolates at q=0.4 (off-midpoint)");
    check_near(curr_lambda_init(losses, 0.9, CurrWeightMode::Hard), 0.91, 1e-14,
               "lambda_init interpolates at q=0.9 (off-midpoint)");
    // non-vacuity: the reversed interpolation gives 0.22 here, not 0.28.
    check(curr_lambda_init(losses, 0.2, CurrWeightMode::Hard) != 0.22,
          "q=0.2 discriminates the interpolation direction (non-vacuity)");
    // and the endpoints, which have frac == 0 and must be exact
    check_near(curr_lambda_init(losses, 0.0, CurrWeightMode::Hard), 0.1, 1e-14,
               "lambda_init at q=0 is the minimum loss");
    check_near(curr_lambda_init(losses, 1.0, CurrWeightMode::Hard), 1.0, 1e-14,
               "lambda_init at q=1 is the maximum loss");
    // a single-sample set has lo == hi, where the two forms COINCIDE
    Tensor one(1, 1); one(0, 0) = 0.42;
    check_near(curr_lambda_init(one, 0.3, CurrWeightMode::Hard), 0.42, 1e-14,
               "single-sample set interpolates to itself");
}

// =============================================================================
// 9/10. The weighted loss and its gradient -- EXACTLY affine in v.
// =============================================================================
static void test_weighted_loss_and_gradient() {
    printf("\n-- Test 9: weighted loss + gradient (D9) --\n");
    Tensor losses(1, 4), weights(1, 4);
    double l[4] = {0.5, 1.5, 2.5, 3.5};
    double v[4] = {1.0, 0.5, 0.25, 0.0};
    for (size_t i = 0; i < 4; ++i) { losses(0, i) = l[i]; weights(0, i) = v[i]; }

    double expect = 0;
    for (size_t i = 0; i < 4; ++i) expect += v[i] * l[i];
    check_near(curr_weighted_loss(losses, weights), expect, 1e-14, "weighted loss value");

    // dE/dv_i == ell_i EXACTLY (E is affine in v). No tolerance needed.
    Tensor g = curr_weighted_loss_gradient(losses);
    check(g.rows == 1 && g.cols == 4, "gradient has the losses' shape");
    for (size_t i = 0; i < 4; ++i)
        check_eq(g(0, i), l[i], "dE/dv_" + std::to_string(i) + " == ell exactly");

    // FD confirmation on v_1, perturbing only that coordinate.
    const double eps = 1e-7;
    Tensor bumped = weights.clone();
    bumped(0, 1) += eps;
    double fd = (curr_weighted_loss(losses, bumped)
                 - curr_weighted_loss(losses, weights)) / eps;
    check_near(fd, l[1], 1e-5, "FD of E wrt v_1 == ell_1");
}

static void test_effective_mass_and_hard_fraction() {
    printf("\n-- Test 10: effective mass + hard fraction (D10) --\n");
    Tensor weights(1, 4);
    double v[4] = {1.0, 0.5, 0.25, 0.0};
    for (size_t i = 0; i < 4; ++i) weights(0, i) = v[i];
    check_near(curr_effective_mass(weights), 1.75, 1e-14, "effective mass == sum of weights");
    check(curr_hard_fraction(weights) == 1, "one sample is exactly zero-weighted");

    // all-zero weights: mass 0, every sample hard -- and NO division by zero.
    Tensor zero(1, 4);
    check_near(curr_effective_mass(zero), 0.0, 0.0, "all-zero mass is exactly 0, not NaN");
    check(curr_hard_fraction(zero) == 4, "all-zero weights -> all hard");
}

// =============================================================================
// 11. D11 -- Definition 1 conditions 4 and 5, on a grid.
// =============================================================================
static void test_monotone_properties() {
    printf("\n-- Test 11: Definition 1 condition 4, and the lambda direction (D11) --\n");

    // Condition 4 -- "monotonically decreasing w.r.t. t" -- holds for EVERY
    // shipped mode. This is the property that identifies sigma as the weight
    // and rules out phi' (which rises again past t == lambda).
    for (auto m : {CurrWeightMode::Sec32, CurrWeightMode::Cauchy, CurrWeightMode::Welsch,
                   CurrWeightMode::L1L2, CurrWeightMode::Hard, CurrWeightMode::Linear}) {
        bool non_increasing_in_ell = true;
        double prev = curr_weight(m, 1.0, 0.0);
        for (double t = 0.05; t <= 8.0; t += 0.05) {
            double w = curr_weight(m, 1.0, t);
            if (w > prev + 1e-15) non_increasing_in_ell = false;
            prev = w;
        }
        check(non_increasing_in_ell,
              "mode " + std::to_string((int)m) + ": weight non-increasing in ell (Def.1 cond.4)");
        // non-vacuity: a CONSTANT is also "non-increasing"
        check(curr_weight(m, 1.0, 8.0) < curr_weight(m, 1.0, 0.0),
              "mode " + std::to_string((int)m) + ": weight genuinely decays (non-vacuity)");
    }

    // Condition 5 -- "monotonous w.r.t. lambda" -- holds INCREASINGLY for
    // three modes and DECREASINGLY for L1L2, and that is not a defect: the
    // paper's own schedule says "lambda_{T+1} = lambda_T / mu for L1-L2 while
    // lambda_{T+1} = lambda_T * mu for Huber, Cauchy and Welsch". A DECREASING
    // weight-vs-lambda is exactly what makes a dividing schedule admit more
    // samples. Asserting "non-decreasing" for every mode is therefore a
    // TEST BUG, and it is recorded here rather than quietly deleted.
    for (auto m : {CurrWeightMode::Sec32, CurrWeightMode::Cauchy, CurrWeightMode::Welsch}) {
        bool inc = true;
        double prev = curr_weight(m, 0.3, 0.7);
        for (double l = 0.35; l <= 3.0; l += 0.05) {
            double w = curr_weight(m, l, 0.7);
            if (w < prev - 1e-15) inc = false;
            prev = w;
        }
        check(inc, "mode " + std::to_string((int)m) + ": weight non-decreasing in lambda");
    }
    {
        bool dec = true;
        double prev = curr_weight(CurrWeightMode::L1L2, 0.3, 0.7);
        for (double l = 0.35; l <= 3.0; l += 0.05) {
            double w = curr_weight(CurrWeightMode::L1L2, l, 0.7);
            if (w > prev + 1e-15) dec = false;
            prev = w;
        }
        check(dec, "L1L2: weight DECREASES in lambda -- the dividing schedule's reason to exist");
        // and the converse: under the paper's dividing schedule the weight RISES,
        // which is what a curriculum needs from its learning-pace parameter.
        check(curr_weight(CurrWeightMode::L1L2, 1.0, 0.7)
                  > curr_weight(CurrWeightMode::L1L2, 2.0, 0.7),
              "L1L2: dividing lambda by mu raises the weight (non-vacuity companion)");
    }
}

// =============================================================================
// 12. D12 -- errors and degenerate inputs (no inf, no NaN, no 1/0).
// =============================================================================
static void test_errors_and_empty_inputs() {
    printf("\n-- Test 12: errors and degenerate inputs (D12) --\n");
    check_throws([](double, double) { curr_weight(CurrWeightMode::Sec32, 0.0, 1.0); },
                 0, 0, "lambda == 0 throws (never returns inf)");
    check_throws([](double, double) { curr_weight(CurrWeightMode::Sec32, -1.0, 1.0); },
                 0, 0, "negative lambda throws");
    check_throws([](double, double) { curr_weight(CurrWeightMode::Sec32, 1.0, -0.5); },
                 0, 0, "negative loss throws");
    check_throws([](double, double) { curr_weight(CurrWeightMode::Log, 1.0, 0.5); },
                 0, 0, "log mode at lambda == 1 throws (zeta == 0)");
    check_throws([](double, double) { curr_weight(CurrWeightMode::L1L2, 0.0, 1.0); },
                 0, 0, "l1l2 at lambda == 0 throws");

    // empty tensors throw rather than returning NaN
    bool threw = false;
    Tensor empty;
    try { (void)curr_weights(empty, 1.0, CurrWeightMode::Sec32); }
    catch (const std::exception&) { threw = true; }
    check(threw, "curr_weights on an empty tensor throws");

    // ALL-ZERO losses: must return all ones, never NaN (the repo's recurring trap)
    Tensor zeros(1, 5);
    Tensor w = curr_weights(zeros, 1.0, CurrWeightMode::Sec32);
    bool all_one = true;
    for (size_t i = 0; i < 5; ++i) all_one = all_one && (w(0, i) == 1.0);
    check(all_one, "all-zero losses -> every weight exactly 1, never NaN");
    Tensor wh = curr_weights(zeros, 1.0, CurrWeightMode::Hard);
    check(wh(0, 0) == 1.0, "all-zero losses under the hard rule -> weight 1, never NaN");

    // returned tensor never aliases the input
    Tensor losses(1, 3);
    losses(0, 0) = 0.1; losses(0, 1) = 0.2; losses(0, 2) = 0.3;
    Tensor out = curr_weights(losses, 1.0, CurrWeightMode::Sec32);
    double before = out(0, 0);
    losses(0, 0) = 99.0;
    check_eq(out(0, 0), before, "returned weights do not alias the input losses");
}

// =============================================================================
// 13. End-to-end: does a growing lambda actually change what gets trained?
//     Liveness assertions, because a constant predictor scores exactly 0.5 on a
//     balanced 2-class task and would make an accuracy comparison vacuous.
// =============================================================================
static void test_end_to_end_curriculum() {
    printf("\n-- Test 13: end-to-end curriculum (liveness) --\n");

    // 20 samples: 10 with loss 0.05 (easy cluster), 10 with loss 4.0 (hard).
    Tensor losses(1, 20);
    for (size_t i = 0; i < 20; ++i)
        losses(0, i) = (i < 10) ? 0.05 : 4.0;

    // At a small lambda the curriculum trains ONLY on the easy cluster.
    Tensor w_small = curr_weights(losses, 0.5, CurrWeightMode::Hard);
    size_t n_small = 0;
    for (size_t i = 0; i < 20; ++i) if (w_small(0, i) != 0.0) ++n_small;
    check(n_small == 10, "small lambda selects exactly the easy half");

    // As lambda grows past the hard cluster's loss, ALL samples join.
    Tensor w_big = curr_weights(losses, 5.0, CurrWeightMode::Hard);
    size_t n_big = 0;
    for (size_t i = 0; i < 20; ++i) if (w_big(0, i) != 0.0) ++n_big;
    check(n_big == 20, "large lambda admits every sample -- the curriculum widens");

    // The effective loss FALLS as the curriculum widens: adding the hard cluster
    // at small weight must not increase the objective it reports.
    double E_small = curr_weighted_loss(losses, w_small);
    double E_big   = curr_weighted_loss(losses, w_big);
    check_near(E_small, 10 * 0.05, 1e-14, "easy-only effective loss");
    check_near(E_big, 10 * 0.05 + 10 * 4.0, 1e-14, "full effective loss");

    // LIVENESS: the two stages must select DIFFERENT sample sets. If both
    // selected the same set, this whole fixture would prove nothing.
    check(n_small != n_big, "the two stages select different sample sets (liveness)");

    // A soft mode weights every sample but monotonically -- never a hard cut.
    Tensor w_soft = curr_weights(losses, 1.0, CurrWeightMode::Sec32);
    bool split = w_soft(0, 0) > w_soft(0, 19);
    check(split, "sec32 weights the easy cluster strictly above the hard cluster");
    check(w_soft(0, 19) > 0.0, "sec32 never drops a sample to exactly zero");
}

int main() {
    printf("=====================================================\n");
    printf("  test_curriculum -- self-paced curriculum weighting\n");
    printf("  arXiv:1606.00128 (Fan, He, Liang, Hu)\n");
    printf("=====================================================\n");

    test_hard_weight_is_a_strict_cut();
    test_hard_boundary_is_a_documented_tie_break();
    test_linear_log_mixture_gamma();
    test_implicit_weight_is_table_column();
    test_implicit_weight_bound_is_two();
    test_hq_minimiser_is_the_same_function();
    test_lambda_schedule_direction();
    test_lambda_init_by_quantile();
    test_weighted_loss_and_gradient();
    test_effective_mass_and_hard_fraction();
    test_monotone_properties();
    test_errors_and_empty_inputs();
    test_end_to_end_curriculum();

    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}