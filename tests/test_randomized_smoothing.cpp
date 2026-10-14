// test_randomized_smoothing.cpp — TDD tests for certified adversarial
// robustness via randomized smoothing.
//
//   Jeremy M. Cohen, Elan Rosenfeld, J. Zico Kolter — "Certified Adversarial
//   Robustness via Randomized Smoothing", ICML 2019, arXiv:1902.02918.
//
// EVERY numeric expectation below is ORACLE-DERIVED from scipy 1.18.1 /
// statsmodels 0.15.0 (never hand-derived by me) via
// tests/randomized_smoothing_oracle.py, which is a verbatim copy of the script
// run during the session. The reason is recorded in the knowledge-distillation
// session in this repo: an oracle transcribed by the same author as the
// implementation is not independent, and the KD oracle had three of its own
// bugs before a line of C++ existed.
//
// The HEADLINE test needs no oracle at all — Appendix B Prop. 3/4 gives a
// closed form: for a linear base classifier f(x) = sign(w.x + b), the smoothed
// classifier g is IDENTICAL to f and the certified radius converges to the true
// robust radius |w.x + b| / ||w||.
//
// THE SEMANTICS EACH FIXTURE EXISTS TO PIN (see the header's D1..D9):
//
//   D3 PREDICT's hypothesis test uses ONLY THE TOP TWO COUNTS, with n = nA + nB
//      -- not the full sample count. Using all n changes the p-value at the
//      same alpha and silently abstains on fixtures that should not abstain.
//      Oracle: binomtest(nA, nA+nB, 0.5). Two-sided at p=0.5 (D4).
//
//   D5 CERTIFY DISCARDS the n0 samples used to guess the class. The
//      Clopper-Pearson bound is formed from the SECOND round only. Forming it
//      from the pooled n0+n samples is a different (and unsound) statistic.
//
//   D6 CP lower bound = Beta^-1(alpha; k, n-k+1). The paper's appendix cites
//      `proportion_confint(k, n, alpha=2*alpha, method="beta")`, which is
//      correct ONLY because statsmodels halves alpha internally for the
//      two-sided default (proportion.py:208-210) and the "beta" branch is
//      literally beta.ppf(alpha, k, n-k+1) (:271). Using 2*alpha directly
//      gives a bound whose defining property P(X>=k) == alpha evaluates to
//      0.1000000000 at alpha=0.05 -- a 2x-too-optimistic certification.
//
//   D7 k == 0 gives EXACTLY 0.0, not NaN. scipy returns NaN because Beta(0, .)
//      is degenerate; the Clopper-Pearson lower bound at zero successes is
//      exactly zero and must certify a NEGATIVE-then-clamped radius.
//
//   D8 The general Theorem 1 form is sigma/2 * (Phi^-1(pA) - Phi^-1(pB)). It
//      collapses to sigma * Phi^-1(pA) ONLY because pB = 1 - pA is assumed.
//      Pinned on a fixture where that identity is exact, plus one where the
//      general form is needed to see them apart.
//
//   D2 The radius requires p_lower > 1/2 STRICTLY. p_lower == 1/2 exactly
//      certifies radius 0 and is NOT a certification -- abstain.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstddef>
#include <functional>
#include <limits>
#include "nn/utils/randomized_smoothing.h"

static int passed = 0;
static int failed = 0;

static void check(const std::string& name, bool pass) {
    if (pass) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else       { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
}

// Relative-or-absolute closeness, scaled by magnitude -- the discipline the
// temperature-scaling session established: a fixed tolerance on a value of
// order 1e-16 is meaningless, and a fixed tolerance on 7.0 is too loose.
static bool close(double a, double b, double rel = 1e-9, double abs_floor = 1e-12) {
    const double d = std::fabs(a - b);
    return d <= abs_floor || d <= rel * std::fabs(b);
}

static bool throws_invalid(const std::function<void()>& fn) {
    try { fn(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

// A base classifier that always returns `c`, used to force exact counts.
struct ConstClassifier {
    int c;
    int operator()(const Tensor&) const { return c; }
};

// f(x) = sign(w.x + b) -- the linear base classifier of Appendix B. Returns 0 or 1.
struct LinearClassifier {
    std::vector<double> w;
    double b;
    int operator()(const Tensor& x) const {
        double acc = b;
        for (size_t j = 0; j < w.size(); ++j) acc += w[j] * x[0][j];
        // sign(0) is undefined in the paper; classify it as 0 deterministically.
        return acc > 0.0 ? 1 : 0;
    }
};

// A classifier whose answer depends on how many entries exceed a threshold,
// so its behaviour under noise is analytically predictable.
struct ThresholdClassifier {
    double thresh;
    int operator()(const Tensor& x) const { return x[0][0] > thresh ? 1 : 0; }
};

// THREE classes, with analytically known probabilities under the noise.
// At x = 0, sigma = 1 the added noise is Z ~ N(0,1), so thresholding Z at the
// two oracle quantiles below yields exactly P(class 2)=0.10, P(class 1)=0.20,
// P(class 0)=0.70 (scipy: norm.ppf(0.30) and norm.ppf(0.10)).
struct ThreeClassClassifier {
    double t0;   // above this -> class 0
    double t1;   // above this (but <= t0) -> class 1
    int operator()(const Tensor& x) const {
        const double z = x[0][0];
        if (z > t0) return 0;
        if (z > t1) return 1;
        return 2;
    }
};

// A classifier that NEVER returns class 0, so `counts[0] == 0`.
// This exists for a reason found by mutation, not by inspection: M9 (storing
// the winning class INDEX into the runner-up COUNT slot) only misbehaves when
// the leading count is 0, so every classifier in the suite that always returns
// class 0 or a three-class classifier that returns class 0 some of the time
// cannot reach it. This one does.
struct NeverZeroClassifier {
    int operator()(const Tensor&) const { return 2; }   // always class 2
};
// Two classes where class 1 WINS and class 0 is the runner-up, with class 0
// holding real mass. P(class 1) = 0.70 at threshold norm.ppf(0.30).
struct MinorityClassClassifier {
    double t;
    int operator()(const Tensor& x) const { return x[0][0] > t ? 1 : 0; }
};
struct TieClassifier {
    int operator()(const Tensor& x) const { return x[0][0] > 0.0 ? 1 : 0; }
};

// ---------------------------------------------------------------------------
// 1. normal_cdf -- Phi(x) = 0.5 * erfc(-x / sqrt 2)
// ---------------------------------------------------------------------------
// TOLERANCES ARE CALIBRATED TO THE MEASURED NUMERICAL FLOOR, not chosen by
// intuition. Measured against scipy 1.18.1:
//   normal_cdf     : worst rel 1.268e-14, at the DEEP TAIL x = -8 (erfc's own
//                    accuracy vs scipy's survival function, not a defect of
//                    ours). So the floor here is 2e-14, not 1e-15.
//   normal_quantile: worst rel 3.8e-16 for p < 0.5, and 2.36e-12 at
//                    p = 1-1e-6. Double precision cannot resolve Phi^-1 beyond
//                    ~1e-6 in the far tail -- inverting a CDF whose own value is
//                    1-1e-12 -- so the p = 1-1e-12 fixture gets an ABSOLUTE
//                    tolerance of 1e-5, which is honest about what the
//                    representation can carry.
// Neither tolerance is a cover for an implementation bug: the core regime that
// certified_radius actually uses (p_lower in [0.5, 1)) is machine-exact, and
// the Prop-3/4 headline test below pins the whole pipeline against a closed
// form with no oracle at all.
static void test_normal_cdf() {
    std::cout << "-- Test 1: normal_cdf against scipy.stats.norm.cdf" << std::endl;
    // Oracle (rs_oracle_final.out):
    struct F { double x; double want; };
    const F fx[] = {
        {-8.0, 6.2209605742717405e-16},
        {-3.5, 0.00023262907903552502},
        {-2.0, 0.022750131948179195},
        {-1.0, 0.15865525393145707},
        {-0.5, 0.30853753872598688},
        { 0.0, 0.5},
        { 0.5, 0.69146246127401312},
        { 1.0, 0.84134474606854293},
        { 2.0, 0.97724986805182079},
        { 3.5, 0.99976737092096446},
        { 8.0, 0.99999999999999933},
    };
    for (const F& f : fx)
        check("Phi(" + std::to_string(f.x) + ")", close(normal_cdf(f.x), f.want, 2e-14, 1e-300));

    check("Phi is symmetric: Phi(x) + Phi(-x) == 1",
          close(normal_cdf(1.7) + normal_cdf(-1.7), 1.0, 1e-14));
    check("Phi(0) is exactly 0.5", normal_cdf(0.0) == 0.5);
}

// ---------------------------------------------------------------------------
// 2. normal_quantile -- Phi^-1 by bisection on erfc.
//    Oracle: scipy.stats.norm.ppf. My kernel was measured to rel <= 4.5e-16.
// ---------------------------------------------------------------------------
static void test_normal_quantile() {
    std::cout << "-- Test 2: normal_quantile against scipy.stats.norm.ppf" << std::endl;
    struct F { double p; double want; };
    const F fx[] = {
        {1e-12, -7.0344838253011313},
        {1e-06, -4.7534243088228987},
        {0.0001, -3.7190164854556804},
        {0.001, -3.0902323061678132},
        {0.005, -2.5758293035489008},
        {0.01, -2.3263478740408408},
        {0.05, -1.6448536269514729},
        {0.1, -1.2815515655446004},
        {0.25, -0.67448975019608171},
        {0.5, 0.0},
        {0.75, 0.67448975019608171},
        {0.9, 1.2815515655446004},
        {0.95, 1.6448536269514722},
        {0.99, 2.3263478740408408},
        {0.995, 2.5758293035489004},
        {0.999, 3.0902323061678132},
        {0.999999, 4.7534243088170873},
        {1.0 - 1e-12, 7.0344869100478356},
    };
    for (const F& f : fx) {
        // The far-tail fixture (p = 1-1e-12) carries an ABSOLUTE tolerance of
        // 1e-5, documented above: Phi^-1 there is limited by the double
        // representation of a CDF whose own value is 1-1e-12, not by the
        // algorithm. Everything else is checked at 1e-11 relative.
        const double abs_tol = (f.p > 1.0 - 1e-9) ? 1e-5 : 1e-11;
        check("Phi^-1(" + std::to_string(f.p) + ")",
              close(normal_quantile(f.p), f.want, 1e-11, abs_tol));
    }

    check("Phi^-1(0.5) == 0", normal_quantile(0.5) == 0.0);
    check("round trip Phi(Phi^-1(p)) == p at p=0.001",
          close(normal_cdf(normal_quantile(0.001)), 0.001, 1e-12));
    check("round trip Phi(Phi^-1(p)) == p at p=0.9",
          close(normal_cdf(normal_quantile(0.9)), 0.9, 1e-12));
    // A sign flip is the fingerprint of a mixed-up bracket.
    check("Phi^-1 is sign-symmetric: Phi^-1(p) == -Phi^-1(1-p)",
          close(normal_quantile(0.01), -normal_quantile(0.99), 1e-11, 1e-11));
    // Out-of-domain inputs must be rejected rather than silently clamped.
    check("Phi^-1(0) throws", throws_invalid([]{ normal_quantile(0.0); }));
    check("Phi^-1(1) throws", throws_invalid([]{ normal_quantile(1.0); }));
    check("Phi^-1(1.5) throws", throws_invalid([]{ normal_quantile(1.5); }));
    check("Phi^-1(-0.1) throws", throws_invalid([]{ normal_quantile(-0.1); }));
}

// ---------------------------------------------------------------------------
// 3. clopper_pearson_lower -- Beta^-1(alpha; k, n-k+1), with k=0 -> 0 (D6, D7).
// ---------------------------------------------------------------------------
static void test_clopper_pearson() {
    std::cout << "-- Test 3: Clopper-Pearson lower bound vs scipy.stats.beta.ppf" << std::endl;
    struct F { int k; int n; double a; double want; };
    const F fx[] = {
        // k == 0 is EXACTLY 0 (D7): scipy returns nan here.
        {  0,  10, 0.05,  0.0},
        {  0, 100, 0.001, 0.0},
        {  1,  10, 0.05,  0.0051161968918237008},
        {  1, 100, 0.001, 1.0004953285956376e-05},
        {  3, 100, 0.05,  0.0082258291076526876},
        { 50, 100, 0.05,  0.41362171463091174},
        { 70, 100, 0.05,  0.61577938720579839},
        { 95, 100, 0.05,  0.89774662235672553},
        { 99, 100, 0.01,  0.93545726795141837},
        {100, 100, 0.001, 0.93325430079699101},
        {100,1000, 0.001, 0.072930496291540997},
        {999,1000, 0.001, 0.99080450236461237},
        {927,1000, 0.001, 0.89813460002130241},
        {500,1000, 0.001, 0.45077105398478234},
    };
    for (const F& f : fx)
        check("CP_lower(k=" + std::to_string(f.k) + ",n=" + std::to_string(f.n) +
              ",a=" + std::to_string(f.a) + ")",
              close(clopper_pearson_lower(f.k, f.n, f.a), f.want, 1e-9));

    // THE DEFINING PROPERTY, measured on the oracle: at p = Beta^-1(a; k, n-k+1)
    // the tail P(X >= k) is EXACTLY a. The 2*alpha form gives 0.1000000000 at
    // a=0.05, so this assertion is what distinguishes the correct bound from
    // the paper's own mis-scaled snippet (D6).
    //
    // DIRECTION -- AND THIS WAS WRONG TWICE, IN MY TESTS, NOT THE CODE. The
    // first version asserted the bound DECREASES in alpha (on the reasoning
    // "more confidence must shrink the bound"); the second version asserted it
    // INCREASES with the same wrong intuition. The measured answer, which also
    // settles it:
    //     alpha=0.100 -> 0.431416825594
    //     alpha=0.050 -> 0.413621714631
    //     alpha=0.025 -> 0.398321129503
    // so the bound INCREASES in alpha, because alpha is the failure probability
    // you ACCEPT, not the confidence you demand. A 90%-confidence bound is
    // allowed to sit higher than a 99.9%-confidence one.
    //
    // The interpretation was confirmed by simulation rather than by argument:
    // drawing k ~ Binomial(100, 0.41362171463091174) and forming the bound at
    // each alpha, the rate at which the true p falls BELOW the bound is
    // 0.07457 / 0.03300 / 0.02030 at alpha = 0.10 / 0.05 / 0.025 -- each within
    // its own alpha, which is exactly the coverage guarantee a valid
    // (1-alpha) lower bound must have.
    const double a05_k50 = clopper_pearson_lower(50, 100, 0.05);
    const double a10_k50 = clopper_pearson_lower(50, 100, 0.10);
    const double a025_k50 = clopper_pearson_lower(50, 100, 0.025);
    check("CP bound is strictly INCREASING in alpha (alpha is the ACCEPTED "
          "failure rate: a weaker requirement sits higher)",
          a025_k50 < a05_k50 && a05_k50 < a10_k50);
    // D2 BOUNDARY -- AND THIS IS THE MUTATION THAT PROVED THE FIXTURE WAS MISSING.
    // M2 relaxed `p_lower > 1/2` to `p_lower >= 1/2` and the suite stayed
    // green, which is the TDD skill's "a mutation that survives is a question
    // about the test". The probe answered it: there is an EXACT boundary
    // reachable. CP_lower(k=n, n, alpha) = alpha^(1/n) exactly, so choosing
    // alpha = 0.5^n makes the bound land on 0.5 to the last bit. Measured:
    //     n=2, alpha=0.25  -> CP_lower = 0.5 exactly
    //     n=3, alpha=0.125 -> CP_lower = 0.5 exactly
    //     n=6, alpha=0.015625 -> CP_lower = 0.5 exactly
    // At that point the radius is 0, which certifies nothing, so `certify`
    // MUST abstain. Without this fixture the strict comparison is untested.
    check("D2 BOUNDARY: CP_lower(n, n, 0.5^n) is EXACTLY 0.5",
          clopper_pearson_lower(2, 2, 0.25) == 0.5 &&
          clopper_pearson_lower(6, 6, 0.015625) == 0.5);
    check("D2 BOUNDARY: certified_radius at p_lower == 0.5 is exactly 0",
          certified_radius(1.0, clopper_pearson_lower(2, 2, 0.25)) == 0.0);
    {
        // A classifier that wins exactly n of n samples cannot happen with a
        // fixed seed, so drive the boundary through the STATISTICS: at
        // p_lower == 0.5 exactly the radius is 0 and the rule must abstain.
        // certify() computes p_lower internally, so this is asserted on the
        // primitive it uses, which is the part that can be exact.
        const double half = clopper_pearson_lower(2, 2, 0.25);
        check("D2 BOUNDARY: a p_lower of exactly 1/2 certifies radius 0 "
              "(so the strict > is load-bearing)",
              half == 0.5 && certified_radius(2.0, half) == 0.0);
    }

    // D7 BOUNDARY -- k = 0 IS REACHABLE AND MUST BE EXACTLY 0.0, NEVER NaN.
    // M7 (delete the `if (k == 0) return 0.0;` guard) survived the first
    // mutation run, so this is a fixture the suite did not have. scipy returns
    // NaN from `beta.ppf(a, 0, n+1)` because Beta(0, .) is degenerate, so an
    // implementation without the guard produces a NaN bound -- which then
    // poisons a whole batch through the radius, and `abstained = !(NaN > 0.5)`
    // evaluates to TRUE by accident, hiding the corruption behind a plausible
    // answer. The direct assertion below is what pins it.
    check("D7: clopper_pearson_lower(0, n, alpha) is EXACTLY 0.0, not NaN",
          clopper_pearson_lower(0, 10, 0.05) == 0.0 &&
          clopper_pearson_lower(0, 1000, 0.001) == 0.0);
    check("D7: the k=0 bound is not merely small, it is exactly zero",
          !std::isnan(clopper_pearson_lower(0, 100, 0.05)));
    // A classifier that loses every sample of the second round: the guessed
    // class vanishes, so certify must report count 0 and abstain with radius 0
    // rather than returning a NaN radius.
    SmoothingConfig kcfg;
    kcfg.sigma = 1.0; kcfg.num_guess_samples = 100; kcfg.num_samples = 1000;
    kcfg.alpha = 0.001; kcfg.seed = 5;
    Tensor kx(1, 1);
    kx[0][0] = 0.0;
    ThreeClassClassifier weak40{0.25334710313579972, -0.38532046640756773};
    const CertifiedPrediction pk = certify(weak40, kx, kcfg);
    check("D7: an abstaining certification never returns a NaN radius",
          !std::isnan(pk.radius) && pk.radius == 0.0);

    check("CP bound at k==n is > 0 (a real radius is certifiable)",
          clopper_pearson_lower(100, 100, 0.001) > 0.9);

    // Invalid arguments.
    check("CP(n=0) throws", throws_invalid([]{ clopper_pearson_lower(0, 0, 0.05); }));
    check("CP(k>n) throws", throws_invalid([]{ clopper_pearson_lower(10, 5, 0.05); }));
    check("CP(k<0) throws", throws_invalid([]{ clopper_pearson_lower(-1, 5, 0.05); }));
    check("CP(alpha=0) throws", throws_invalid([]{ clopper_pearson_lower(5, 10, 0.0); }));
    check("CP(alpha=1) throws", throws_invalid([]{ clopper_pearson_lower(5, 10, 1.0); }));
    check("CP(alpha>1) throws", throws_invalid([]{ clopper_pearson_lower(5, 10, 2.0); }));
}

// ---------------------------------------------------------------------------
// 4. binom_pvalue_half -- two-sided test at p = 0.5 (D3, D4).
//    Oracle: scipy.stats.binomtest(nA, nA+nB, 0.5).pvalue
// ---------------------------------------------------------------------------
static void test_binom_pvalue() {
    std::cout << "-- Test 4: two-sided binom p-value at p=0.5 vs scipy binomtest" << std::endl;
    struct F { int n_a; int n_b; double want; };
    const F fx[] = {
        {  1,  0, 1.0},
        {  2,  1, 1.0},
        {  3,  0, 0.25},
        { 10,  9, 1.0},
        { 10,  5, 0.30175781249999994},
        {  6,  4, 0.75390625},
        {  5,  5, 1.0},
        { 20, 11, 0.14961278438568115},
        {100,  1, 8.0463812332543204e-29},
        {  3, 10, 0.09228515625},
        {  9,  6, 0.60723876953124989},
    };
    for (const F& f : fx)
        check("binomtest(nA=" + std::to_string(f.n_a) + ",nB=" + std::to_string(f.n_b) + ")",
              close(binom_pvalue_half(f.n_a, f.n_b), f.want, 1e-12, 1e-300));

    // The p-value must be symmetric in its arguments' roles only up to the
    // two-sided symmetry -- a one-sided implementation returns the tail only.
    check("binom p-value is symmetric: p(nA,nB) == p(nB,nA)  [two-sided, not one]",
          close(binom_pvalue_half(6, 4), binom_pvalue_half(4, 6), 1e-14));

    check("binom p-value at 0 successes over 0 trials is 1",
          binom_pvalue_half(0, 0) == 1.0);
    check("binom p-value with n_a<0 throws",
          throws_invalid([]{ binom_pvalue_half(-1, 5); }));
    check("binom p-value with n_b<0 throws",
          throws_invalid([]{ binom_pvalue_half(5, -1); }));
}

// ---------------------------------------------------------------------------
// 5. Theorem 1 -- the certified radius (D2, D8).
// ---------------------------------------------------------------------------
static void test_certified_radius() {
    std::cout << "-- Test 5: Theorem 1 radius vs oracle" << std::endl;
    // General form sigma/2 * (Phi^-1(pA) - Phi^-1(pB)):
    struct G { double p_a; double p_b; double s; double want; };
    const G g[] = {
        {0.9, 0.1, 1.0, 1.2815515655446004},
        {0.7, 0.3, 2.0, 1.0488010254160816},
        {0.99, 0.01, 0.5, 1.1631739370204204},
        {0.6, 0.4, 1.0, 0.25334710313579972},
    };
    for (const G& t : g)
        check("R(pA=" + std::to_string(t.p_a) + ",pB=" + std::to_string(t.p_b) +
              ",sigma=" + std::to_string(t.s) + ")",
              close(certified_radius_from_probs(t.s, t.p_a, t.p_b), t.want, 1e-10));

    // Symmetric form sigma * Phi^-1(pA), with pB = 1 - pA:
    struct S { double p_a; double s; double want; };
    const S sx[] = {
        {0.9, 1.0, 1.2815515655446004},
        {0.7, 2.0, 1.0488010254160813},
        {0.99, 0.5, 1.1631739370204204},
        {0.6, 1.0, 0.25334710313579972},
    };
    for (const S& t : sx)
        check("R_sym(pA=" + std::to_string(t.p_a) + ",sigma=" + std::to_string(t.s) + ")",
              close(certified_radius(t.s, t.p_a), t.want, 1e-10));

    // D8 GUARD: the two forms AGREE exactly when pB = 1 - pA, and DISAGREE
    // otherwise. A test that only checked the symmetric form could not tell
    // them apart, so this is the assertion that makes the general form real.
    const double sym = certified_radius(1.0, 0.9);
    check("D8: with pB = 1-pA the general and symmetric forms AGREE",
          close(certified_radius_from_probs(1.0, 0.9, 0.1), sym, 1e-10));
    check("D8: with pB != 1-pA the two forms MUST DISAGREE",
          !close(certified_radius_from_probs(1.0, 0.9, 0.4), sym, 1e-6));
    // pB larger => runner-up more likely => SMALLER radius. A sign flip here is
    // the classic bug.
    check("radius is non-increasing in pB",
          certified_radius_from_probs(1.0, 0.9, 0.05) >=
          certified_radius_from_probs(1.0, 0.9, 0.30));
    check("radius is non-decreasing in pA",
          certified_radius_from_probs(1.0, 0.95, 0.05) >=
          certified_radius_from_probs(1.0, 0.60, 0.05));
    // sigma is linear in the radius.
    check("radius is linear in sigma",
          close(certified_radius_from_probs(2.0, 0.9, 0.1), 2.0 * sym, 1e-10));

    check("certified_radius(pA=0.5) is 0", close(certified_radius(1.0, 0.5), 0.0, 1e-12));
    check("certified_radius(pA=1.0) throws", throws_invalid([]{ certified_radius(1.0, 1.0); }));
    check("certified_radius(pA<=0) throws", throws_invalid([]{ certified_radius(1.0, 0.0); }));
    check("certified_radius(sigma=0) throws", throws_invalid([]{ certified_radius(0.0, 0.9); }));
    check("certified_radius_from_probs(pB>pA) throws",
          throws_invalid([]{ certified_radius_from_probs(1.0, 0.4, 0.6); }));
}

// ---------------------------------------------------------------------------
// 6. Theorem 6 -- certified test-set accuracy lower bound.
//    Algebra taken from the PROOF (p. 24), since the displayed Eq. 16 is
//    mangled by the PDF's fraction layout.
// ---------------------------------------------------------------------------
static void test_certified_accuracy_bound() {
    std::cout << "-- Test 6: Theorem 6 certified-accuracy bound" << std::endl;
    struct F { int y; int m; double a; double rho; double want; };
    const F fx[] = {
        {  0, 100, 0.05,  0.05, -0.10779276725225534},
        {  0, 1000, 0.001, 0.001, -0.0043954291881490156},
        {100, 100,  0.001, 0.001, 1.0000000000000002},
        {900, 1000, 0.001, 0.001, 0.89650547171275208},
        {500, 1000, 0.001, 0.001, 0.49610507131235154},
        {950, 1000, 0.001, 0.001, 0.94655552176280211},
    };
    for (const F& t : fx)
        check("Thm6(y=" + std::to_string(t.y) + ",m=" + std::to_string(t.m) + ")",
              close(certified_accuracy_lower_bound(t.y, t.m, t.a, t.rho), t.want, 1e-10));

    // The bound can be NEGATIVE (no certified examples) -- it must be returned
    // as computed, and the CALLER clamps. An implementation that clamps inside
    // the statistic would silently inflate every small-sample result.
    check("Thm6 with y=0 returns the raw negative value (caller clamps)",
          certified_accuracy_lower_bound(0, 100, 0.05, 0.05) < 0.0);
    check("Thm6 is monotone in y",
          certified_accuracy_lower_bound(900, 1000, 0.001, 0.001) >
          certified_accuracy_lower_bound(500, 1000, 0.001, 0.001));
    // A perfect certification can exceed 1 slightly; do NOT clamp in here.
    check("Thm6 at y=m slightly exceeds 1 and is returned unclamped",
          certified_accuracy_lower_bound(100, 100, 0.001, 0.001) > 1.0);

    check("Thm6(m=0) returns 0", certified_accuracy_lower_bound(0, 0, 0.05, 0.05) == 0.0);
    check("Thm6(y>m) throws", throws_invalid([]{ certified_accuracy_lower_bound(10, 5, 0.05, 0.05); }));
    check("Thm6(alpha=0) throws", throws_invalid([]{ certified_accuracy_lower_bound(5, 10, 0.0, 0.05); }));
    check("Thm6(alpha=1) throws", throws_invalid([]{ certified_accuracy_lower_bound(5, 10, 1.0, 0.05); }));
    check("Thm6(rho=0) throws", throws_invalid([]{ certified_accuracy_lower_bound(5, 10, 0.05, 0.0); }));
    check("Thm6(rho=1) throws", throws_invalid([]{ certified_accuracy_lower_bound(5, 10, 0.05, 1.0); }));
}

// ---------------------------------------------------------------------------
// 7. sample_under_noise -- D1 (additive N(0, sigma^2 I)) and D9 (determinism).
// ---------------------------------------------------------------------------
static void test_sample_under_noise() {
    std::cout << "-- Test 7: sample_under_noise (D1 additive noise, D9 determinism)" << std::endl;
    Tensor x(1, 3);
    x[0][0] = 0.5; x[0][1] = -0.25; x[0][2] = 1.75;
    ConstClassifier c0{0};

    // D9: the same seed gives bit-identical counts.
        // NOTE the classifier here must DEPEND ON THE INPUT, or every seed gives
        // the same all-ones vector and the "different seed" check below is
        // vacuous -- it would pass for an implementation that ignored `seed`
        // entirely. ThresholdClassifier splits the noise, so the counts actually
        // record the draw.
        ThresholdClassifier tseed{0.5};
        const std::vector<int> a1 = sample_under_noise(tseed, x, 1.0, 200, 1234);
        const std::vector<int> a2 = sample_under_noise(tseed, x, 1.0, 200, 1234);
        check("D9: identical seed gives identical counts", a1 == a2);
        const std::vector<int> a3 = sample_under_noise(tseed, x, 1.0, 200, 4321);
        check("D9: a different seed changes the draw", a1 != a3);
        // And the counts must be non-degenerate, i.e. both classes actually appear,
        // so the equality above is a real comparison rather than two empty vectors.
        check("D9 GUARD: both classes occur, so the seed check is not vacuous",
              a1.size() == 2 && a1[0] > 0 && a1[1] > 0);

    // Counts sum to num_samples -- the invariant that catches a dropped sample.
    const std::vector<int> c = sample_under_noise(c0, x, 1.0, 137, 7);
    int total = 0; for (int v : c) total += v;
    check("counts sum to num_samples", total == 137);
    check("counts has one entry per class present", c.size() >= 1);

    // D1: noise is ADDITIVE and centred. A classifier thresholding at exactly
    // x[0] should flip on roughly half the samples at any sigma (P(x+sigma*Z > x)
    // = Phi(0) = 1/2), NOT on ~half of the samples of a MULTIPLICATIVE noise.
    ThresholdClassifier thr{0.5};   // x[0]=0.5: P(0.5 + sigma*Z > 0.5) = 0.5
    const std::vector<int> flips = sample_under_noise(thr, x, 1.0, 20000, 99);
    const double rate = static_cast<double>(flips[1]) / 20000.0;
    check("D1: additive zero-mean noise flips P=1/2 of the samples (" +
          std::to_string(rate) + ")", close(rate, 0.5, 0.03));

    // The empirical rate must follow the analytic Phi(x0/sigma) from the
    // oracle, which is the sharpest available check on the noise law itself.
    // Oracle: x0=1.5, sigma=1.0 -> 0.93319279873114191
    Tensor x15(1, 1); x15[0][0] = 1.5;
    ThresholdClassifier thr15{0.0};
    const std::vector<int> f15 = sample_under_noise(thr15, x15, 1.0, 40000, 555);
    check("D1: empirical P(x+sigma*Z > 0) matches Phi(1.5/1.0)=" +
          std::to_string(static_cast<double>(f15[1]) / 40000.0),
          close(static_cast<double>(f15[1]) / 40000.0, 0.93319279873114191, 0.01));

    check("sample_under_noise(num_samples=0) returns no counts",
          sample_under_noise(c0, x, 1.0, 0, 1).empty());
    check("sample_under_noise(num_samples<0) throws",
          throws_invalid([&]{ sample_under_noise(c0, x, 1.0, -1, 1); }));
    check("sample_under_noise(sigma<0) throws",
          throws_invalid([&]{ sample_under_noise(c0, x, -1.0, 10, 1); }));
}

// ---------------------------------------------------------------------------
// 8. smoothed_predict -- D3 (top-two only) and D4 (abstain iff p > alpha).
// ---------------------------------------------------------------------------
static void test_smoothed_predict() {
    std::cout << "-- Test 8: smoothed_predict abstention rule (D3, D4)" << std::endl;
    Tensor x(1, 2);
    x[0][0] = 0.0; x[0][1] = 0.0;

    // All samples vote class 0: nA = n, nB = 0, so the two-sided binom test on
    // (n, 0) gives p = 2*(1/2)^n, which is <= alpha for n >= 11 at a=0.001 --
    // so it must NOT abstain. A one-sided p-value would also pass this, so the
    // discriminating fixture is below.
    ConstClassifier c0{0};
    const SmoothedPrediction p0 = smoothed_predict(c0, x, 1.0, 100, 0.001, 11);
    check("unanimous class 0 at n=100 does not abstain", !p0.abstained);
    check("unanimous prediction returns class 0", p0.class_index == 0);
    check("unanimous: n_a = 100", p0.n_a == 100);
    check("unanimous: n_b = 0 (D3 top-two only)", p0.n_b == 0);

    // TIED votes must ABSTAIN at any sane alpha: there is no evidence for a
    // top class. p = 1 here, and 1 > alpha, so abstain.
    // A THREE-CLASS classifier with analytically KNOWN probabilities. At x=0,
    // sigma=1 the noise is Z ~ N(0,1), so thresholding on Z at Phi^-1(0.30)
    // and Phi^-1(0.10) gives
    //     P(class 2) = 0.10, P(class 1) = 0.20, P(class 0) = 0.70
    // (oracle thresholds: -0.5244005127080409 and -1.2815515655446004).
    //
    // This fixture exists for D3, and it is the ONLY shape that can show it:
    // the third class must hold real mass so that n_a + n_b < num_samples. A
    // two-class fixture can never distinguish "the test used the top two
    // counts" from "the test used the whole sample".
    ThreeClassClassifier thr3{-0.5244005127080409, -1.2815515655446004};
    const SmoothedPrediction pb = smoothed_predict(thr3, x, 1.0, 2000, 0.001, 2024);
    check("three-class fixture: class 0 is the plurality", pb.class_index == 0);
    check("three-class fixture: n_a > n_b (0.70 vs 0.20 of the mass)",
          pb.n_a > pb.n_b);
    check("D3: n_a + n_b is STRICTLY LESS than num_samples because class 2 "
          "holds mass (" + std::to_string(pb.n_a + pb.n_b) + " < 2000)",
          pb.n_a + pb.n_b < 2000);
    check("D3: n_a/n_b tracks the true 0.70/0.20 ratio",
          close(static_cast<double>(pb.n_a) / static_cast<double>(pb.n_b), 3.5, 0.25));
    check("three-class fixture does not abstain (0.70 vs 0.20 is decisive)",
          !pb.abstained);

    // D3 GUARD -- THE DISCRIMINATING ASSERTION. The returned p-value must be
    // binomtest(n_a, n_a+n_b) computed from the RETURNED counts. At n_a=700,
    // n_b=200 (sum 900 of 2000) the oracle gives 1.1691765767235576e-65, while
    // a full-n implementation would compute binomtest(700, 2000) =
    // 1.7665678007949568e-37. Any implementation that feeds the whole sample
    // into the test cannot produce the first number.
    check("returned p_value == binom_pvalue_half(n_a, n_b) exactly",
          pb.p_value == binom_pvalue_half(pb.n_a, pb.n_b));
    check("returned n_a >= n_b (top-two, sorted)", pb.n_a >= pb.n_b);

    // A TIE must abstain at any alpha: there is no evidence for a top class.
    // A fair coin almost never ties EXACTLY in a finite sample (the first
    // version of this fixture asserted n_a == n_b on 200 draws, which failed
    // with 105/95 -- the classifier was right and the assertion was asking for
    // a coincidence). What is actually true, and what the rule depends on, is
    // that a near-even split gives a LARGE p-value and therefore abstains.
    // The exact-tie case is pinned directly on binom_pvalue_half below.
    TieClassifier tie;
    const SmoothedPrediction pt = smoothed_predict(tie, x, 1.0, 200, 0.5, 31);
    check("fair-coin split is near even (|n_a - n_b| small relative to n)",
          std::abs(pt.n_a - pt.n_b) < 20);
    check("fair-coin split gives a LARGE p-value (> 0.2)",
          pt.p_value > 0.2);
    check("fair-coin split abstains at alpha=0.5", pt.abstained);
    // The exact tie IS the p-value's worst case, and it is exact:
    // binomtest(n, n) = 1 for any n, so a tie always abstains.
    check("an exact tie has p-value exactly 1.0 (no evidence either way)",
          binom_pvalue_half(100, 100) == 1.0 &&
          binom_pvalue_half(1, 1) == 1.0 &&
          binom_pvalue_half(7, 7) == 1.0);

    // A classifier that always returns a class OTHER than 0 makes `counts[0] == 0`,
    // which is the ONLY configuration in which the index/count bug in `top_two`
    // can fire (M9's first run survived because every other fixture had a
    // non-zero leading count). Under the fix the runner-up of a unanimous
    // class-2 sample is 0, so the test must report n_b == 0 and still answer.
    NeverZeroClassifier nz;
    const SmoothedPrediction pn = smoothed_predict(nz, x, 1.0, 100, 0.5, 1);
    check("M9 GUARD: a classifier that never emits class 0 returns class 2",
          pn.class_index == 2);
    check("M9 GUARD: n_a == num_samples", pn.n_a == 100);
    check("M9 GUARD: n_b == 0 (no other class was ever emitted)", pn.n_b == 0);
    check("M9 GUARD: such a classifier is decisive and does not abstain",
          !pn.abstained);
    check("M9 GUARD: the p-value is the decisive (n, 0) value, not a spurious "
          "n_b index", pn.p_value == binom_pvalue_half(100, 0));
    // The same shape through CERTIFY, where the class is guessed from counts.
    SmoothingConfig ncfg;
    ncfg.sigma = 1.0; ncfg.num_guess_samples = 10; ncfg.num_samples = 100;
    ncfg.alpha = 0.5; ncfg.seed = 1;
    const CertifiedPrediction qn = certify(nz, x, ncfg);
    check("M9 GUARD: certify returns class 2 and certifies it",
          !qn.abstained && qn.class_index == 2 && qn.count == 100);

    // Two classes where the RUNNER-UP IS CLASS 0 and class 1 wins. This is the
    // shape that exposes the `top_two` index/count bug (M9), and it is
    // surprisingly narrow: the corruption happens only when the winning class
    // is NOT index 0, because the buggy line stores the *old best's index* (0)
    // into the runner-up count slot -- and when class 0 is itself the winner,
    // the stored 0 happens to equal the correct runner-up count only if class 0
    // never emitted anything.
    //
    // Measured over the exhaustive space (counts vectors of length 1..4 with
    // entries 0..4, 780 vectors): 188 of them DIFFER between the fixed and the
    // buggy function, and every two-class vector (a, b) with a < b and a != 0
    // is one of them. So the fixture must have class 1 win with class 0 still
    // holding real mass -- P(class 1) = 0.70, P(class 0) = 0.30 does it.
    //
    // At n = 2000 the oracle gives n_a ~ 600, n_b ~ 1400 and
    // binomtest(600, 2000, 0.5) = 2.3039745559914301e-73, whereas the buggy
    // version computes binomtest(n_a, 0), which is astronomically small and
    // would abstain NEVER. So this fixture separates the two on the abstention
    // decision itself, not merely on a returned integer.
    // P(class 1) = 1 - Phi(t) = 0.70 requires t = norm.ppf(0.30) = -0.5244005127080409.
    MinorityClassClassifier minority{-0.5244005127080409};
    const SmoothedPrediction pm9 = smoothed_predict(minority, x, 1.0, 2000, 0.001, 2024);
    check("M9 FIXTURE: class 1 is the winner and class 0 is the runner-up",
          pm9.class_index == 1 && pm9.n_a > pm9.n_b);
    check("M9 FIXTURE: the runner-up count is class 0's REAL count, not 0",
          pm9.n_b > 500);
    check("M9 FIXTURE: p_value uses the real runner-up count",
          pm9.p_value == binom_pvalue_half(pm9.n_a, pm9.n_b) &&
          pm9.p_value != binom_pvalue_half(pm9.n_a, 0));
    check("M9 FIXTURE: n_a + n_b covers both emitted classes",
          pm9.n_a + pm9.n_b == 2000);
    // Same shape through CERTIFY: the guess round also calls top_two, so a
    // corrupted runner-up there would change the guessed class.
    SmoothingConfig m9cfg;
    m9cfg.sigma = 1.0; m9cfg.num_guess_samples = 100; m9cfg.num_samples = 1000;
    m9cfg.alpha = 0.001; m9cfg.seed = 2024;
    const CertifiedPrediction q9 = certify(minority, x, m9cfg);
    check("M9 FIXTURE: certify guesses class 1 and certifies it",
          !q9.abstained && q9.class_index == 1);

    // A large alpha must make PREDICT more willing to answer.
    ConstClassifier c0b{0};
    const SmoothedPrediction loose = smoothed_predict(c0b, x, 1.0, 10, 0.9, 5);
    check("large alpha answers more readily", !loose.abstained);

    check("smoothed_predict(num_samples<1) throws",
          throws_invalid([&]{ smoothed_predict(c0, x, 1.0, 0, 0.05, 1); }));
    check("smoothed_predict(alpha=0) throws",
          throws_invalid([&]{ smoothed_predict(c0, x, 1.0, 10, 0.0, 1); }));
    check("smoothed_predict(alpha=1) throws",
          throws_invalid([&]{ smoothed_predict(c0, x, 1.0, 10, 1.0, 1); }));
    check("smoothed_predict(sigma<0) throws",
          throws_invalid([&]{ smoothed_predict(c0, x, -1.0, 10, 0.05, 1); }));
}

// ---------------------------------------------------------------------------
// 9. certify -- D5 (the n0 round is DISCARDED) and D2 (p_lower > 1/2 strictly).
// ---------------------------------------------------------------------------
static void test_certify() {
    std::cout << "-- Test 9: certify (D5 discarded n0 round, D2 p_lower > 1/2)" << std::endl;
    Tensor x(1, 1);
    x[0][0] = 0.0;

    // Unanimous class 0 => count = n, so the Clopper-Pearson bound is the
    // k = n endpoint: CP_lower(1000, 1000, 0.001) = 0.99311604842093382
    // (= alpha^(1/n), measured), comfortably above 1/2, so it CERTIFIES.
    // NOTE: this is the k == n row. The 0.072930496291540997 figure in the
    // CP table is the k = 100, n = 1000 row -- transcribing the two as one is
    // exactly the hand-calc mistake the temperature-scaling session recorded.
    ConstClassifier c0{0};
    SmoothingConfig cfg;
    cfg.sigma = 1.0;
    cfg.num_guess_samples = 100;
    cfg.num_samples = 1000;
    cfg.alpha = 0.001;
    cfg.seed = 31337;
    const CertifiedPrediction cert = certify(c0, x, cfg);
    check("unanimous classifier does not abstain", !cert.abstained);
    check("unanimous classifier certifies class 0", cert.class_index == 0);
    check("count == num_samples exactly", cert.count == 1000);
    check("unanimous: p_lower is the k=n endpoint Beta^-1(0.001; 1000, 1)",
          close(cert.p_lower, 0.99311604842093382, 1e-9));
    check("unanimous: radius == sigma*Phi^-1(p_lower) = 2.4632626147808114",
          close(cert.radius, 2.4632626147808114, 1e-8));
    // The radius must be FINITE and positive -- a NaN here would come from a
    // degenerate Beta shape at k == n, which is D7's mirror image.
    check("unanimous: radius is finite", std::isfinite(cert.radius));

    // D2 THE REAL ABSTENTION FIXTURE. A 40/25/35 split has a plurality of only
    // 0.40, and no sample size at alpha = 0.001 can lift its Clopper-Pearson
    // bound above 1/2 -- CP_lower(400, 1000, 0.001) = 0.3524307119. So this
    // MUST abstain. This is the rule that makes `certify` refuse to certify
    // rather than report a negative or meaningless radius, and it is a genuine
    // model property, not an edge case: the paper's Proposition 6 constructs
    // base classifiers for which g is robust at no radius at all.
    ThreeClassClassifier weak{0.25334710313579972, -0.38532046640756773};
    SmoothingConfig wcfg = cfg;
    wcfg.seed = 4242;
    const CertifiedPrediction pw = certify(weak, x, wcfg);
    check("D2: a 40/25/35 plurality ABSTAINS (p_lower < 1/2)", pw.abstained);
    check("D2: abstaining result reports radius 0", pw.radius == 0.0);
    check("D2: abstaining result reports a sub-1/2 p_lower",
          pw.p_lower > 0.0 && pw.p_lower < 0.5);
    // Same n, same alpha, same sigma -- only the CLASSIFIER differs. So the
    // abstention is caused by the bound and not by the configuration.
    ThreeClassClassifier strong{-0.5244005127080409, -1.2815515655446004};
    const CertifiedPrediction ps = certify(strong, x, wcfg);
    check("D2 GUARD: the SAME config certifies a 70/20/10 classifier",
          !ps.abstained && ps.radius > 0.0);

    // At a looser alpha the SAME unanimous sample certifies a real radius.
    SmoothingConfig loose = cfg;
    loose.alpha = 0.05;
    const CertifiedPrediction cert2 = certify(c0, x, loose);
    check("looser alpha lets the unanimous sample certify", !cert2.abstained);
    check("looser alpha yields a positive radius", cert2.radius > 0.0);

    // D2 BOUNDARY THROUGH certify() ITSELF -- THIS IS THE FIXTURE THAT CATCHES M2.
    // The primitive-level fixture above proves the STATISTIC hits 0.5 exactly,
    // but M2 mutates the comparison inside `certify`, so the suite needs a case
    // that drives the boundary through the real entry point. It is reachable:
    // with num_samples = 2 and alpha = 0.25 = 0.5^2, an unanimous classifier
    // gives count == 2, and CP_lower(2, 2, 0.25) is EXACTLY 0.5 (measured).
    // Under the correct strict `>` this abstains with radius 0; under the
    // mutant's `>=` it would "certify" a radius of exactly 0 -- i.e. report a
    // robustness guarantee of zero, which is precisely the failure the paper's
    // `if p_A > 1/2` guards against.
    ConstClassifier bc0{0};
    SmoothingConfig bcfg;
    bcfg.sigma = 1.0;
    bcfg.num_guess_samples = 2;
    bcfg.num_samples = 2;
    bcfg.alpha = 0.25;          // 0.5^2
    bcfg.seed = 1;
    const CertifiedPrediction pb2 = certify(bc0, x, bcfg);
    check("D2 BOUNDARY (end-to-end): p_lower is EXACTLY 1/2 here",
          pb2.p_lower == 0.5);
    check("D2 BOUNDARY (end-to-end): a radius of zero is NOT a certification, "
          "so it ABSTAINS", pb2.abstained);
    check("D2 BOUNDARY (end-to-end): and reports radius 0",
          pb2.radius == 0.0);

    // D5 GUARD: the bound must come from the SECOND round only. With a
    // classifier whose answer does not depend on the input, the second round
    // has count == n; if an implementation pooled the n0 samples it would
    // report count == n + n0 == 1100.
    check("D5: count excludes the n0 guess samples", cert2.count == 1000);

    // D5 GUARD THAT THE UNANIMOUS FIXTURE CANNOT SEE. M5 (pool the n0 round
    // into the estimate) survived the first mutation run for a structural
    // reason: every D5 fixture used a classifier that returns the same class
    // for every sample, where count == n either way and CP_lower(n, n, a) ==
    // CP_lower(n, n+n0, a) is NOT the same... except that the k=n endpoint is
    // a^(1/n), which DOES depend on n. So the mutant does change the bound --
    // but only for a classifier whose count is below n, and every D5 fixture
    // had count == n. This fixture has count strictly below n, so the pooled
    // and unpooled bounds are genuinely different numbers.
    ThreeClassClassifier mixed{0.25334710313579972, -0.38532046640756773};
    SmoothingConfig mcfg = cfg;
    mcfg.alpha = 0.05;
    mcfg.seed = 777;
    const CertifiedPrediction pm = certify(mixed, x, mcfg);
    check("D5 GUARD: a non-unanimous classifier has count < num_samples",
          pm.count < mcfg.num_samples);
    check("D5 GUARD: p_lower uses the SECOND round's n, not n0 + n "
          "(" + std::to_string(pm.p_lower) + ")",
          close(pm.p_lower,
                clopper_pearson_lower(pm.count, mcfg.num_samples, mcfg.alpha),
                1e-12));
    // And the pooled value it must NOT equal, computed from the same count:
    const double pooled =
        clopper_pearson_lower(pm.count, mcfg.num_samples + mcfg.num_guess_samples,
                              mcfg.alpha);
    check("D5 GUARD: the pooled (n0+n) bound is a DIFFERENT number, so this "
          "fixture can distinguish them", !close(pm.p_lower, pooled, 1e-6));

    // D5 GUARD vs THE MUTATED HELPER -- THIS IS WHAT ACTUALLY CATCHES M5.
    // The two assertions above both compute their reference by CALLING
    // clopper_pearson_lower, which M5 mutates as well, so they compare the
    // mutant against itself and stay green. The independent check is a
    // hard-coded ORACLE value for this exact (count, n, alpha):
    //     CP_lower(375, 1000, 0.05) = 0.34961543093405756   (correct)
    //     CP_lower(375, 1100, 0.05) = 0.31728184736950937   (M5, pooled)
    // (measured with scipy 1.18.1; the fixture's seed fixes count at 375.)
    check("D5 GUARD (oracle): p_lower is the scipy value 0.34961543093405756",
          close(pm.p_lower, 0.34961543093405756, 1e-12));
    check("D5 GUARD (oracle): and is NOT the pooled value 0.31728184736950937",
          !close(pm.p_lower, 0.31728184736950937, 1e-6));
    // The count itself is unchanged by the mutation, so pinning it too makes the
    // failure localise to the denominator rather than to the sampling.
    check("D5 GUARD (oracle): count is exactly 375 at this seed",
          pm.count == 375);
    // And THAT LAST ASSERTION is the one that actually catches M5, for a
    // non-obvious reason worth recording. M5 changes how many samples the
    // SECOND round DRAWS (num_samples -> num_samples + num_guess_samples)
    // without changing the denominator `certify` divides by, so it moves BOTH
    // the count and the bound: measured on this fixture, count goes 375 -> 415
    // and p_lower goes 0.34961543093405756 -> 0.38908144392140764. That is why
    // comparing p_lower against the oracle still passed while the mutant ran --
    // the mutant is a different ESTIMATOR (more samples, wrong denominator),
    // not a different way of computing the same estimator. Pinning the count
    // itself, which the mutation cannot leave alone, is what localises it.
    check("D5 GUARD (oracle): the second round draws exactly num_samples "
          "samples (M5 would draw 1100 and report a different count)",
          pm.count == 375 && pm.count != 415);

    // The class must be the argmax of the FIRST (discarded) round. With a
    // unanimous classifier the two rounds agree, so this only pins the
    // unanimous case; the discriminating fixture is the linear classifier in
    // test 10, where the two rounds can genuinely differ.
    check("certified class matches the unanimous classifier",
          cert2.class_index == 0);

    // Determinism (D9).
    const CertifiedPrediction cert3 = certify(c0, x, loose);
    check("D9: certify is deterministic for a fixed seed",
          cert2.count == cert3.count && cert2.radius == cert3.radius &&
          cert2.p_lower == cert3.p_lower);

    // A classifier that ALWAYS says class 1 must certify class 1, not 0.
    ConstClassifier c1{1};
    const CertifiedPrediction cert4 = certify(c1, x, loose);
    check("always-class-1 classifier certifies class 1", cert4.class_index == 1);

    check("certify(sigma<0) throws",
          throws_invalid([&]{ SmoothingConfig b = loose; b.sigma = -1.0; certify(c0, x, b); }));
    check("certify(num_samples<1) throws",
          throws_invalid([&]{ SmoothingConfig b = loose; b.num_samples = 0; certify(c0, x, b); }));
    check("certify(num_guess_samples<1) throws",
          throws_invalid([&]{ SmoothingConfig b = loose; b.num_guess_samples = 0; certify(c0, x, b); }));
    check("certify(alpha out of range) throws",
          throws_invalid([&]{ SmoothingConfig b = loose; b.alpha = 0.0; certify(c0, x, b); }));
    check("certify(num_guess_samples<0) throws",
          throws_invalid([&]{ SmoothingConfig b = loose; b.num_guess_samples = -5; certify(c0, x, b); }));
}

// ---------------------------------------------------------------------------
// 10. HEADLINE -- Appendix B Prop. 3/4 closed form, NO ORACLE NEEDED.
//     For f(x) = sign(w.x+b):  g == f identically, and the certified radius
//     converges from BELOW to the true robust radius |w.x + b| / ||w||.
//     Oracle cross-check: ||w|| = 1 exactly, true radius = 0.73 exactly,
//     exact p_A = Phi(0.73/0.5) = 0.92785496303410619.
// ---------------------------------------------------------------------------
static void test_linear_classifier_closed_form() {
    std::cout << "-- Test 10: HEADLINE -- Appendix B Prop 3/4 closed form" << std::endl;
    LinearClassifier lin{{0.6, -0.8, 0.0}, 0.35};
    Tensor x(1, 3);
    x[0][0] = 0.9; x[0][1] = 0.2; x[0][2] = -0.5;

    // ||w|| == 1 and w.x + b == 0.91, so the true robust radius is exactly 0.73.
    check("fixture: f(x) = 1 (w.x+b = 0.91 > 0)", lin(x) == 1);
    const double true_radius = 0.73;

    SmoothingConfig cfg;
    cfg.sigma = 0.5;
    cfg.alpha = 0.001;
    cfg.seed = 20241014;

    // Prop. 3: g == f at every sigma. CERTIFY must return class 1 always.
    for (double sig : {0.25, 0.5, 2.0}) {
        SmoothingConfig c = cfg; c.sigma = sig; c.num_samples = 20000;
        const CertifiedPrediction p = certify(lin, x, c);
        check("Prop3: g(x) == f(x) = 1 at sigma=" + std::to_string(sig),
              !p.abstained && p.class_index == 1);
    }

    // Prop. 4/5: the certified radius is a VALID bound -- it never exceeds the
    // true one -- and it CONVERGES to it as n grows. Each doubling of n must
    // move the certified radius closer to 0.73.
    double prev_err = std::numeric_limits<double>::infinity();
    double r_small = 0.0, r_big = 0.0;
    for (int n : {2000, 20000, 200000}) {
        SmoothingConfig c = cfg; c.num_samples = n;
        const CertifiedPrediction p = certify(lin, x, c);
        check("Prop4: certified radius < true radius at n=" + std::to_string(n),
              p.radius > 0.0 && p.radius < true_radius);
        const double err = std::fabs(true_radius - p.radius);
        check("Prop4: error shrinks from n=" + std::to_string(n - 1) +
              " to n=" + std::to_string(n) + " (" + std::to_string(prev_err) +
              " -> " + std::to_string(err) + ")", err < prev_err);
        prev_err = err;
        if (n == 2000) r_small = p.radius;
        if (n == 200000) r_big = p.radius;
    }
    // A bound that OVER-shoots the true radius is unsound -- this is the single
    // most important assertion in the file.
    check("Prop5: bound never OVERSHOOTS the true robust radius", r_big < true_radius);
    // And it gets materially close at large n.
    check("Prop4: at n=200000 the radius is within 5% of 0.73 (" +
          std::to_string(r_big) + ")", std::fabs(true_radius - r_big) < 0.05 * true_radius);
    check("larger n gives a LARGER radius", r_big > r_small);
}

// ---------------------------------------------------------------------------
// 11. certified_accuracy -- the end-to-end aggregation the paper reports.
// ---------------------------------------------------------------------------
static void test_certified_accuracy_aggregation() {
    std::cout << "-- Test 11: certified accuracy aggregation over a test set" << std::endl;
    // A perfectly smooth classifier: every point certifies at a huge radius.
    // The aggregate must be 1.0 (clamped) and must NOT be vacuous.
    LinearClassifier lin{{1.0}, 5.0};   // always class 1, far from the boundary
    Tensor x(1, 1);
    x[0][0] = 5.0;
    SmoothingConfig cfg;
    cfg.sigma = 1.0; cfg.alpha = 0.05; cfg.num_samples = 100; cfg.seed = 8;

    int certified_at_r = 0;
    const int m = 20;
    for (int i = 0; i < m; ++i) {
        const CertifiedPrediction p = certify(lin, x, cfg);
        if (!p.abstained && p.class_index == 1 && p.radius > 0.0) ++certified_at_r;
    }
    check("all m points certify at radius > 0", certified_at_r == m);
    // LIVENESS GUARD (the adversarial session's lesson): an aggregate of 1.0 is
    // only meaningful if the certification actually returned a POSITIVE radius
    // for real points. Assert the radius itself, not just the count.
    const CertifiedPrediction one = certify(lin, x, cfg);
    check("LIVENESS: radius is strictly positive, not a clipped zero",
          one.radius > 0.0);
    // The Theorem 6 bound at y = m. At m = 20 with alpha = rho = 0.05 the bound
    // is 0.88597864447444363 -- NOT 1.0. The slack term
    // sqrt(2*m*alpha*(1-alpha)*ln(1/rho) - ln(1/rho)/3) scales with sqrt(m),
    // so a small sample cannot certify itself perfectly even when every point
    // certified. This is the correct statistical statement (an oracle value,
    // not a wish), and asserting `>= 1.0` here would be asserting something
    // the bound was never designed to deliver.
    const double lb = certified_accuracy_lower_bound(m, m, cfg.alpha, cfg.alpha);
    check("Thm6 at y=m=20, alpha=rho=0.05 is 0.88597864447444363 (not 1.0)",
          close(lb, 0.88597864447444363, 1e-9));
    check("Thm6 at y=m is BELOW 1 for a small sample", lb < 1.0);
    // ... and it approaches 1 as m grows at fixed alpha, which is the property
    // that makes it a bound rather than an arbitrary discount. Oracle:
    // Thm6(1000, 1000, 0.001, 0.001) = 0.99660557181285203.
    check("Thm6 at y=m grows toward 1 as m grows",
          certified_accuracy_lower_bound(1000, 1000, 0.001, 0.001) >
          certified_accuracy_lower_bound(20, 20, 0.05, 0.05));
    // LIVENESS: the bound must still be a bound -- it cannot exceed the
    // observed fraction, and it must be positive here (every point certified).
    check("Thm6 at y=m is positive when every point certified", lb > 0.0);
}

// ---------------------------------------------------------------------------
// 12. API guard -- the suite must not be satisfiable by a stub.
// ---------------------------------------------------------------------------
static void test_api_guard() {
    std::cout << "-- Test 12: API guard" << std::endl;
    // A fake classifier returning an out-of-range class must not crash the
    // counting loop; it is the caller's contract, but negative classes must at
    // least not be indexed out of bounds. We assert on the well-formed path.
    check("SmoothingConfig defaults are the paper's CIFAR-10 setting "
          "(sigma=0.25, alpha=0.001)",
          SmoothingConfig().sigma == 0.25 && SmoothingConfig().alpha == 0.001);
    check("SmoothingConfig defaults: n0=100, n=1000",
          SmoothingConfig().num_guess_samples == 100 &&
          SmoothingConfig().num_samples == 1000);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== test_randomized_smoothing (Cohen, Rosenfeld, Kolter "
                 "arXiv:1902.02918) ===" << std::endl;
    test_normal_cdf();
    test_normal_quantile();
    test_clopper_pearson();
    test_binom_pvalue();
    test_certified_radius();
    test_certified_accuracy_bound();
    test_sample_under_noise();
    test_smoothed_predict();
    test_certify();
    test_linear_classifier_closed_form();
    test_certified_accuracy_aggregation();
    test_api_guard();
    std::cout << "\n=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}