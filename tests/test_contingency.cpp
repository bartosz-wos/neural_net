// =============================================================================
// tests/test_contingency.cpp -- the CATEGORICAL-AGREEMENT and
// CONTINGENCY-TABLE-INFERENCE utilities: Cohen's kappa (unweighted / linear
// / quadratic disagreement weights), the Matthews correlation coefficient,
// Pearson's chi-square test of independence, the G-test (the
// log-likelihood-ratio sibling on the same table), McNemar's test for paired
// binary outcomes, and Fisher's exact test for a 2x2 table -- plus the
// general chi-square survival function (the regularized upper incomplete
// gamma) that all three table tests need.
//
// WHAT THIS SUITE EXISTS TO PIN
// =============================================================================
//
// The repo shipped ~250 layers and NO categorical agreement or categorical
// association machinery at all. This suite pins the estimators and, just as
// importantly, the CONVENTIONS, because every one of them is a place where a
// plausible implementation silently returns a DIFFERENT number:
//
//   1. THE df == 1 CHI-SQUARE PATH IS THE REPO'S EXISTING erfc CLOSED FORM,
//      BIT FOR BIT. The suite compares with `==`, NOT with a tolerance, and
//      it checks the general incomplete-gamma route against that closed form
//      to 1e-14 -- two independent routes to the same number, which is what
//      makes the incomplete gamma trustworthy.
//
//   2. THE WEIGHTED KAPPA'S WEIGHT MATRIX IS BUILT OVER THE EXPLICIT
//      CLASS-INDEX SPACE, so an ABSENT intermediate class does NOT compress
//      the distances. The F2 fixture has class 2 absent, and the suite
//      asserts the weighted answers DIFFER from the unweighted one, and
//      that the compressed reading of the SAME fixture is a different
//      number -- otherwise the fixture would not discriminate.
//
//   3. THE DEGENERATE KAPPA IS NaN, NOT 0.0. The two degenerate cases in
//      this library are OPPOSITE to one another, and the fixtures pin both.
//
//   4. THE KAPPA'S EXPECTED MATRIX MAY BE IN EITHER ORIENTATION: every
//      weight matrix this module supports is SYMMETRIC, so for symmetric w
//      the sums over w_ij E_ij and over w_ij E_ji are equal by relabelling
//      and the transposition cancels exactly. **THIS IS NOT DIRECTLY
//      OBSERVABLE IN THE C++ API** -- the expected matrix is not exposed, and
//      the diagonal of E is symmetric under the transposition too -- so
//      there is NO test for it here, and inventing one would be a lie. See
//      the C4 comment in the kappa section.
//
//   5. THE DEGENERATE MATTHEWS COEFFICIENT IS 0.0, NOT NaN -- the OPPOSITE
//      of the kappa's degenerate case. F5 pins it; F4 is a MULTICLASS
//      fixture whose |MCC| is large, so a binary-only shortcut is visibly
//      wrong.
//
//   6. YATES' CONTINUITY CORRECTION APPLIES ONLY TO A 2x2 TABLE, and its
//      magnitude is CLAMPED to each cell's OWN deviation. The F8 fixture
//      is the clamp discriminator: every cell has |O - E| = 0.2, so the
//      clamped statistic is 0.0 while the textbook (unclamped) one would be
//      0.3125.
//
//   7. THE DEGREES OF FREEDOM ARE (R-1)(C-1), NOT THE CELL COUNT MINUS ONE
//      (which is what scipy's power_divergence uses, and which is wrong for
//      a contingency table). The F7 fixture is a 2x3 table with df == 2, and
//      its p-value is what proves the general incomplete gamma is needed at
//      all.
//
//   8. AN EMPTY ROW OR COLUMN (a zero expected count) THROWS
//      std::invalid_argument -- for the chi-square test AND for the G-test.
//
//   9. A ZERO OBSERVED CELL CONTRIBUTES EXACTLY 0 TO THE G-TEST AND IS
//      SKIPPED. The naive expression is 0 * log(0 / E) = 0 * (-inf) = NaN
//      and it destroys the whole statistic.
//
//  10. McNEMAR'S CONTINUITY CORRECTION IS APPLIED IFF n01 != n10; the
//      numerator is (|n01 - n10| - 1)^2 WITH the absolute value; and when
//      n01 == n10 the statistic is 0 -- NOT 1 / (2 * n01). The F11 fixture
//      is the discriminating one, F12 makes the absolute value bite, and
//      F13 has no discordant pairs, where the statistic is NaN.
//
//  11. FISHER'S TWO-SIDED p-VALUE USES THE PROBABILITY ORDERING with
//      scipy's 1 + 1e-14 RELATIVE tie tolerance, NOT a doubled one-sided
//      tail. The F14 and F15 fixtures separate the two rules.
//
//  12. FISHER'S ODDS RATIO IS THE CROSS-PRODUCT (a*d)/(b*c), NOT R's
//      conditional MLE, which is a different estimator.
//
//  13. THE EXACT McNEMAR p-VALUE IS THE REPO'S EXISTING two-sided binomial
//      tail at p = 1/2 on the discordant pairs.
//
// THE FIXTURES ARE EMITTED, NEVER TYPED.
//
// The numbers below come from tests/contingency_oracle.py (float64 numpy,
// measured against scipy 1.17.1, scikit-learn 1.9.0 and R 4.5.3), which was
// run BEFORE this file existed. The repo has been burned by hand-transcribed
// numbers before -- both numbers were correct for the fixture they were
// given, and the fixture was simply not the oracle's -- so the pinned
// numbers are TRANSFERRED by tests/splice_contingency_fixture.py, never
// re-derived.
//
// THE INDEPENDENT REFERENCE.
//
// The weighted kappa and the Matthews coefficient are ALSO recomputed in
// THIS FILE, from the DEFINITION, with explicit scalar loops and no shared
// code with the library: a plain nested loop over a contingency table, no
// library call of any kind. The library is not checked against itself.
// The kappa's reference is a SECOND implementation, and it is deliberately
// a SEQUENTIAL sum where the library uses numpy's pairwise reduction -- so
// the two agree to round-off on the fixtures, and a mutation that moved
// the sums would show up as a disagreement rather than as agreement.
//
// The suite also recomputes, from the definitions and with plain scalar
// loops, the regularized upper incomplete gamma, the hypergeometric pmf
// with the probability-ordering sweep, and the two-sided binomial tail --
// so the library's chi-square survival function, its Fisher p-value and its
// exact McNemar p-value each have a second witness that shares no code with
// the implementation.
//
// THE GUARDS.
//
// Every guard's MESSAGE is checked against a DISTINCT needle. A needle that
// appears in two different messages discriminates nothing between them, and
// this repo has hit that trap repeatedly.
// =============================================================================

#include "nn/utils/contingency.h"
#include "nn/core/tensor.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
    // FLUSH ON EVERY CHECK. A buffered stdout loses its buffer when the
    // process dies, and a SEGFAULT then looks like a SILENT crash -- which
    // the mutation harness cannot distinguish from a detected defect.
    fflush(stdout);
}

static void check_near(double got, double want, double rel_tol,
                       const std::string& name) {
    if (!std::isfinite(want) && !std::isfinite(got)) {
        // Both infinite (or both NaN): compare exactly.
        check(got == want || (std::isnan(got) && std::isnan(want)), name);
        return;
    }
    const double denom = std::max(std::fabs(want), 1e-12);
    const double rel = std::fabs(got - want) / denom;
    if (rel <= rel_tol) {
        ++g_pass;
        printf("  [PASS] %s (got %.15g, want %.15g, rel %.2e)\n",
               name.c_str(), got, want, rel);
    } else {
        ++g_fail;
        printf("  [FAIL] %s (got %.15g, want %.15g, rel %.2e > %.2e)\n",
               name.c_str(), got, want, rel, rel_tol);
    }
    fflush(stdout);
}

// A throwing call, converted into a REPORTED failure. An exception that
// escapes main() aborts the process and loses every result after it, so
// a clean failure turns into a crash -- which the mutation harness cannot
// score.
//
// THE MESSAGE IS READ AFTER THE CALL, never before: C++ evaluates a
// function call's arguments in UNSPECIFIED order, so a helper that writes
// the message and a caller that reads it in the same expression can print
// the message from the PREVIOUS call.
static bool throws_with(const std::function<void()>& fn, const std::string& needle,
                        std::string* what_out = nullptr) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (what_out) *what_out = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helpers to turn the oracle's flat vectors into the library's Tensors.
//
// A LABEL VECTOR IS A COLUMN, (N,1). The module rejects a non-single-column
// input rather than silently reinterpreting it, so a (1,N) tensor holding
// the same values is NOT the same sample here.
// ---------------------------------------------------------------------------
static Tensor col(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

static Tensor mat(const std::vector<std::vector<double>>& rows) {
    Tensor t(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            t(i, j) = rows[i][j];
    return t;
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for Cohen's kappa, computed from the DEFINITION
// with explicit scalar loops.
//
// It shares NO code with the library. This is what makes the kappa checks
// meaningful: a check against the library's own output is not a second
// witness, and a finite-difference style check compares the implementation
// against ITSELF.
//
// The formula (the header's C2/C3):
//
//   p_o = (1/N) * sum_i C_ii
//   p_e = sum_i (r_i/N) * (c_i/N)
//   kappa_w = 1 - (sum_ij w_ij C_ij) / (sum_ij w_ij E_ij),  E_ij = r_i c_j / N
//
// and the weight matrix is built over the EXPLICIT class-index space
// 0 .. num_classes-1, so an absent intermediate class does NOT compress the
// distances.
// ---------------------------------------------------------------------------
struct ReferenceKappa {
    double kappa = std::numeric_limits<double>::quiet_NaN();
    double observed_agreement = 0.0;
    double expected_agreement = 0.0;
    size_t num_agreements = 0;
};

static ReferenceKappa reference_cohen_kappa(const std::vector<double>& labels_a,
                                            const std::vector<double>& labels_b,
                                            size_t num_classes,
                                            KappaWeights weights) {
    ReferenceKappa out;
    const size_t n = labels_a.size();
    if (n == 0) return out;

    // The contingency table, built here, over the explicit class-index space.
    std::vector<double> table(num_classes * num_classes, 0.0);
    for (size_t k = 0; k < n; ++k) {
        const size_t i = static_cast<size_t>(labels_a[k]);
        const size_t j = static_cast<size_t>(labels_b[k]);
        table[i * num_classes + j] += 1.0;
    }

    std::vector<double> row(num_classes, 0.0), col_sums(num_classes, 0.0);
    double trace = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            row[i] += table[i * num_classes + j];
            col_sums[j] += table[i * num_classes + j];
        }
        trace += table[i * num_classes + i];
    }

    const double total = static_cast<double>(n);
    out.observed_agreement = trace / total;
    out.num_agreements = static_cast<size_t>(trace);

    double expected_agreement = 0.0;
    for (size_t i = 0; i < num_classes; ++i)
        expected_agreement += (row[i] / total) * (col_sums[i] / total);
    out.expected_agreement = expected_agreement;

    double numerator = 0.0, denominator = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            const double diff = static_cast<double>(i) - static_cast<double>(j);
            double w = 0.0;
            switch (weights) {
                case KappaWeights::Unweighted: w = (i == j) ? 0.0 : 1.0; break;
                case KappaWeights::Linear:     w = std::fabs(diff);        break;
                case KappaWeights::Quadratic:  w = diff * diff;            break;
            }
            const double e = row[i] * col_sums[j] / total;
            numerator += w * table[i * num_classes + j];
            denominator += w * e;
        }
    }
    out.kappa = (denominator == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                      : 1.0 - numerator / denominator;
    return out;
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for the Matthews correlation coefficient,
// computed from the DEFINITION with explicit scalar loops.
//
//   cov_ytyp = c*N - sum_k p_k t_k
//   cov_ypyp = N^2 - sum_k p_k^2
//   cov_ytyt = N^2 - sum_k t_k^2
//   MCC      = cov_ytyp / sqrt(cov_ytyt * cov_ypyp)
//
// and the DEGENERATE case (a vanishing denominator) is 0.0, NOT NaN -- the
// OPPOSITE of the kappa's degenerate case.
// ---------------------------------------------------------------------------
static double reference_matthews(const std::vector<double>& truth,
                                 const std::vector<double>& prediction,
                                 size_t num_classes) {
    const size_t n = truth.size();
    if (n == 0) return 0.0;
    const double total = static_cast<double>(n);

    std::vector<double> t_sum(num_classes, 0.0), p_sum(num_classes, 0.0);
    double correct = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const size_t t = static_cast<size_t>(truth[k]);
        const size_t p = static_cast<size_t>(prediction[k]);
        t_sum[t] += 1.0;
        p_sum[p] += 1.0;
        if (t == p) correct += 1.0;
    }

    double dot_tp = 0.0, dot_pp = 0.0, dot_tt = 0.0;
    for (size_t k = 0; k < num_classes; ++k) {
        dot_tp += t_sum[k] * p_sum[k];
        dot_pp += p_sum[k] * p_sum[k];
        dot_tt += t_sum[k] * t_sum[k];
    }

    const double cov_ytyp = correct * total - dot_tp;
    const double cov_ypyp = total * total - dot_pp;
    const double cov_ytyt = total * total - dot_tt;
    const double denominator = cov_ytyt * cov_ypyp;
    if (denominator == 0.0) return 0.0;
    return cov_ytyp / std::sqrt(denominator);
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for the regularized upper incomplete gamma
// Q(a, x) = 1 - P(a, x): the SERIES for P(a, x) when x < a + 1 and the
// CONTINUED FRACTION for Q(a, x) otherwise, with exp(-x + a log x - lgamma a)
// factored out of both.
//
// This is the numerical engine behind the general chi-square survival
// function. It is written here from the definition so that the library's
// engine has a second witness.
// ---------------------------------------------------------------------------
static double reference_gamma_series(double a, double x) {
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int i = 0; i < 1000; ++i) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) <= std::fabs(sum) * 3e-16) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

static double reference_gamma_continued_fraction(double a, double x) {
    const double fpmin = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / fpmin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 1000; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 3e-16) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

static double reference_gamma_q(double a, double x) {
    if (x < a + 1.0) return 1.0 - reference_gamma_series(a, x);
    return reference_gamma_continued_fraction(a, x);
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for the hypergeometric pmf and Fisher's
// two-sided probability-ordering sweep, from the DEFINITION.
//
// The pmf is evaluated in LOG SPACE, because a direct factorial computation
// overflows for the table sizes a caller will actually pass.
//
// `use_tolerance` selects scipy's 1 + 1e-14 RELATIVE tie tolerance. Two
// tables can share the observed pmf EXACTLY IN EXACT ARITHMETIC while their
// float64 evaluations differ in the last bits; the tolerance is what
// includes the second one, and a bare comparison excludes it and HALVES the
// p-value.
// ---------------------------------------------------------------------------
static double reference_hypergeom_pmf(double x, double M, double n, double N) {
    if (x < 0.0 || x > n || x > M || (N - x) > (M - n)) return 0.0;
    const double log_p =
        (std::lgamma(n + 1.0) - std::lgamma(x + 1.0) - std::lgamma(n - x + 1.0) +
         std::lgamma(M - n + 1.0) - std::lgamma(N - x + 1.0) - std::lgamma(M - n - N + x + 1.0) -
         std::lgamma(M + 1.0) + std::lgamma(N + 1.0) + std::lgamma(M - N + 1.0));
    return std::exp(log_p);
}

static double reference_fisher_p(const std::vector<std::vector<double>>& table,
                                  bool use_tolerance) {
    const double a = table[0][0], b = table[0][1], c = table[1][0], d = table[1][1];
    const double row1 = a + b, col1 = a + c, total = a + b + c + d;
    const double pexact = reference_hypergeom_pmf(a, total, row1, col1);
    const double gamma = use_tolerance ? (1.0 + 1e-14) : 1.0;
    const double lo = std::max(0.0, col1 - (total - row1));
    const double hi = std::min(row1, col1);
    double sum = 0.0;
    for (double x = lo; x <= hi; x += 1.0) {
        const double p = reference_hypergeom_pmf(x, total, row1, col1);
        if (p <= pexact * gamma) sum += p;
    }
    return std::min(sum, 1.0);
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for the two-sided binomial tail at p = 1/2,
// from the DEFINITION:  min(1, 2 * min(P(X <= k), P(X >= k))).
//
// This is the repo's EXISTING `binom_pvalue_half`, recomputed here with
// plain scalar loops so that the exact McNemar p-value has a second witness
// that shares no code with the implementation.
// ---------------------------------------------------------------------------
static double reference_binom_pvalue_half(int n_a, int n_b) {
    const int n = n_a + n_b;
    if (n == 0) return 1.0;  // a vacuous test is not evidence of anything
    double lower = 0.0, upper = 0.0;
    double choose = 1.0;
    for (int i = 0; i <= n; ++i) {
        if (i > 0) choose = choose * (n - i + 1) / i;
        const double p = choose * std::pow(0.5, static_cast<double>(n));
        if (i <= n_a) lower += p;
        if (i >= n_a) upper += p;
    }
    return std::min(1.0, 2.0 * std::min(lower, upper));
}

// =============================================================================
// THE FIXTURES (emitted by the oracle -- never hand-transcribed).
// =============================================================================

// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
// Emitted by tests/contingency_oracle.py --emit-cpp. float64 numpy.
// The numbers are TRANSFERRED, never re-derived by hand.
// --- F1: the chi-square survival function -------------------------
// df = 1 is the repo's EXISTING erfc closed form; df = 2 and 10
// EXERCISE THE GENERAL INCOMPLETE GAMMA, which the repo lacks.
const std::vector<double> f1_chi2_x = {0.5, 1.0, 3.841458820694124, 10.0, 23.02585092994046, 100.0};
const std::vector<double> f1_chi2_df = {1.0, 2.0, 1.0, 10.0, 2.0, 50.0};
const std::vector<double> f1_chi2_sf = {0.4795001221869535, 0.6065306597126334, 0.05000000000000008, 0.44049328506521224, 9.999999999999977e-06, 3.4549313829848695e-05};
// --- F2: Cohen's kappa, the DISCRIMINATING label-space fixture ----
// Class 2 is ABSENT, so the weighted kappa differs between the
// compressed space and the explicit class-index space.
const std::vector<double> f2_k_a = {0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0};
const std::vector<double> f2_k_b = {0.0, 3.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0};
const size_t f2_k_classes = 4;
const double f2_k_unweighted_kappa = 0.36842105263157887;
const double f2_k_unweighted_po = 0.5555555555555556;
const double f2_k_unweighted_pe = 0.2962962962962963;
const double f2_k_linear_kappa = 0.4424778761061947;
const double f2_k_linear_po = 0.5555555555555556;
const double f2_k_linear_pe = 0.2962962962962963;
const double f2_k_quadratic_kappa = 0.48275862068965514;
const double f2_k_quadratic_po = 0.5555555555555556;
const double f2_k_quadratic_pe = 0.2962962962962963;
// --- F3: Cohen's kappa, the DEGENERATE case -----------------------
// The answer is NaN, NOT 0.
const std::vector<double> f3_k_a = {0.0, 0.0, 0.0, 0.0};
const std::vector<double> f3_k_b = {0.0, 0.0, 0.0, 0.0};
const size_t f3_k_classes = 2;
// --- F4: MCC, an ASYMMETRIC multiclass fixture --------------------
const std::vector<double> f4_t = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0};
const std::vector<double> f4_p = {1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 1.0};
const size_t f4_classes = 3;
const double f4_mcc = -0.5057694217035739;
// --- F5: MCC, the DEGENERATE case ---------------------------------
// The answer is 0.0, NOT NaN -- the OPPOSITE of the kappa case.
const std::vector<double> f5_t = {0.0, 1.0, 2.0, 1.0};
const std::vector<double> f5_p = {1.0, 1.0, 1.0, 1.0};
const size_t f5_classes = 3;
const double f5_mcc = 0.0;
// --- F6/F7/F8: the chi-square tests -------------------------------
const std::vector<std::vector<double>> f6_table = {
    {10.0, 20.0},
    {30.0, 15.0}};
const double f6_corrected_stat = 6.752232142857143;
const double f6_corrected_p = 0.009363047637533945;
const double f6_uncorrected_stat = 8.035714285714285;
const double f6_uncorrected_p = 0.004586392080253493;
const size_t f6_dof = 1;
const std::vector<std::vector<double>> f7_table = {
    {10.0, 20.0, 5.0},
    {30.0, 15.0, 7.0}};
const double f7_chi2_stat = 8.03247645211931;
const double f7_chi2_p = 0.018020627120288134;
const size_t f7_dof = 2;
// The YATES CLAMP: every cell has |O-E| = 0.2 < 0.5, so the
// clamped statistic is 0.0 and the unclamped one is 0.3125.
const std::vector<std::vector<double>> f8_table = {
    {1.0, 1.0},
    {1.0, 2.0}};
const double f8_corrected_stat = 0.0;
const double f8_corrected_p = 1.0;
const double f8_uncorrected_stat = 0.1388888888888888;
const double f8_uncorrected_p = 0.7093881150142265;
// --- F9: the G-test, with a ZERO observed cell ---------------------
const std::vector<std::vector<double>> f9_table = {
    {0.0, 5.0},
    {10.0, 20.0}};
const double f9_g_stat = 3.6880211032792385;
const double f9_g_p = 0.05480460223095297;
const size_t f9_dof = 1;
const std::vector<std::vector<double>> f9_g_2x3_table = {
    {10.0, 20.0, 5.0},
    {30.0, 15.0, 7.0}};
const double f9_g_2x3_stat = 8.173284326835873;
const double f9_g_2x3_p = 0.016795535643078025;
const size_t f9_g_2x3_dof = 2;
// --- F10-F13: McNemar's test --------------------------------------
// F10: n01 != n10 -> the correction IS applied.
const std::vector<double> f10_a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> f10_b = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const double f10_mcnemar_stat = 0.125;
const double f10_mcnemar_p = 0.7236736098317631;
const double f10_mcnemar_exact_p = 0.7265625;
const double f10_mcnemar_uncorr_stat = 0.5;
const double f10_n01 = 5.0;
const double f10_n10 = 3.0;
// F11: n01 == n10 -> the correction is SKIPPED and the statistic is 0.
const std::vector<double> f11_a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> f11_b = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const double f11_mcnemar_stat = 0.0;
const double f11_mcnemar_p = 1.0;
// F12: n01 < n10 -> the ABSOLUTE value matters.
const std::vector<double> f12_a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> f12_b = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const double f12_mcnemar_stat = 0.125;
const double f12_mcnemar_p = 0.7236736098317631;
// F13: no discordant pairs -> NaN, not 0.
const std::vector<double> f13_a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> f13_b = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const double f13_mcnemar_stat = std::numeric_limits<double>::quiet_NaN();
const double f13_mcnemar_p = std::numeric_limits<double>::quiet_NaN();
const double f13_mcnemar_exact_p = 1.0;
// --- F14/F15: Fisher's exact test ---------------------------------
const std::vector<std::vector<double>> f14_table = {
    {10.0, 20.0},
    {30.0, 15.0}};
const double f14_fisher_p = 0.008824760272676115;
const double f14_fisher_or = 0.25;
const std::vector<std::vector<double>> f15_table = {
    {8.0, 2.0},
    {1.0, 5.0}};
const double f15_fisher_p = 0.03496503496503495;
const double f15_fisher_or = 20.0;
// The TIE-TOLERANCE fixture: two tables share the observed pmf
// EXACTLY, so scipy's relative tolerance (1 + 1e-14) is what
// includes the second one. Without it the p-value HALVES.
const std::vector<std::vector<double>> f16_table = {
    {0.0, 2.0},
    {5.0, 3.0}};
const double f16_fisher_p = 0.4444444444444446;
const double f16_fisher_or = 0.0;
// >>> ORACLE-EMITTED FIXTURE END <<<

// =============================================================================
// THE TESTS
// =============================================================================

// ---------------------------------------------------------------------------
// C1. THE CHI-SQUARE SURVIVAL FUNCTION.
//
// The df == 1 path is the repo's EXISTING closed form,
// std::erfc(std::sqrt(statistic / 2.0)), BIT FOR BIT -- the comparison
// below is `==`, NOT a tolerance, because "the same expression" is the
// convention and a tolerance would hide a drift.
//
// The GENERAL path is a separate implementation (the regularized upper
// incomplete gamma), and it is checked against the df = 1 closed form to
// 1e-14: two independent routes to the same number.
//
// The general path is ALSO checked against the EXACT closed forms that exist
// for integer degrees of freedom -- Q(1,u) = e^-u, Q(2,u) = e^-u (1+u),
// Q(3,u) = e^-u (1 + u + u^2/2) -- which is a witness that does not share
// any code with the library at all.
// ---------------------------------------------------------------------------
static void test_c1_chi_square_survival() {
    printf("\n--- C1: chi-square survival ---\n");

    // The fixture values, for every (x, df) pair the oracle pinned.
    for (size_t i = 0; i < f1_chi2_x.size(); ++i) {
        const double x = f1_chi2_x[i];
        const double df = f1_chi2_df[i];
        char name[160];
        snprintf(name, sizeof(name), "F1 chi2_sf(x=%.10g, df=%.10g)", x, df);
        check_near(chi_square_survival(x, df), f1_chi2_sf[i], 1e-13, name);
    }

    // C1: THE df == 1 PATH IS BIT-IDENTICAL TO THE erfc CLOSED FORM.
    // Compared with `==`, not with a tolerance.
    for (size_t i = 0; i < f1_chi2_x.size(); ++i) {
        if (f1_chi2_df[i] != 1.0) continue;
        const double x = f1_chi2_x[i];
        char name[160];
        snprintf(name, sizeof(name),
                 "F1 chi2_sf(x=%.10g, df=1) is BIT-IDENTICAL to erfc(sqrt(x/2))", x);
        check(chi_square_survival(x, 1.0) == std::erfc(std::sqrt(x / 2.0)), name);
    }
    // The same, on values that are NOT in the fixture: the convention is a
    // property of the function, not of the fixture.
    for (double x : {1e-8, 0.25, 2.5, 7.0, 13.5, 40.0, 1e4}) {
        char name[160];
        snprintf(name, sizeof(name),
                 "C1 chi2_sf(x=%.10g, df=1) is BIT-IDENTICAL to erfc(sqrt(x/2))", x);
        check(chi_square_survival(x, 1.0) == std::erfc(std::sqrt(x / 2.0)), name);
    }

    // C1: THE GENERAL PATH AGREES WITH THE df = 1 CLOSED FORM TO 1e-14.
    // The reference gamma here is written in THIS FILE from the definition,
    // so this is a comparison of two independent routes, not of the library
    // against itself.
    for (double x : {1e-8, 0.01, 0.5, 1.0, 2.5, 7.0, 13.5, 40.0, 1e4}) {
        const double general = reference_gamma_q(0.5, x / 2.0);
        const double closed = std::erfc(std::sqrt(x / 2.0));
        char name[160];
        snprintf(name, sizeof(name),
                 "C1 the incomplete gamma agrees with the df=1 erfc closed form at x=%.10g", x);
        check_near(general, closed, 1e-14, name);
    }
    // And the library's GENERAL path (df != 1) is that same gamma, so it
    // agrees with the df = 1 closed form wherever the two are comparable.
    for (double x : {0.5, 3.841458820694124, 23.02585092994046}) {
        const double general_path = reference_gamma_q(0.5, x / 2.0);
        const double closed = std::erfc(std::sqrt(x / 2.0));
        char name[160];
        snprintf(name, sizeof(name),
                 "C1 the general-path gamma and the df=1 closed form agree at x=%.10g", x);
        check_near(general_path, closed, 1e-14, name);
    }

    // C1: THE GENERAL PATH AGAINST THE EXACT CLOSED FORMS. These share no
    // code with the library whatsoever.
    for (double x : {0.25, 1.0, 5.0, 12.0}) {
        const double u = x / 2.0;
        char name[160];
        snprintf(name, sizeof(name), "C1 chi2_sf(x=%.10g, df=2) == exp(-x/2)", x);
        check_near(chi_square_survival(x, 2.0), std::exp(-u), 1e-13, name);
        snprintf(name, sizeof(name), "C1 chi2_sf(x=%.10g, df=4) == exp(-u)(1+u)", x);
        check_near(chi_square_survival(x, 4.0), std::exp(-u) * (1.0 + u), 1e-13, name);
        snprintf(name, sizeof(name), "C1 chi2_sf(x=%.10g, df=6) == exp(-u)(1+u+u^2/2)", x);
        check_near(chi_square_survival(x, 6.0),
                   std::exp(-u) * (1.0 + u + 0.5 * u * u), 1e-13, name);
    }

    // C1: A STATISTIC OF EXACTLY ZERO IS 1.0, EXACTLY. The guard is not
    // cosmetic: the general path evaluates log(x) with x = statistic/2, so a
    // statistic of 0 is log(0) = -inf and the whole expression is NaN.
    check(chi_square_survival(0.0, 1.0) == 1.0, "C1 chi2_sf(0, df=1) == 1.0 EXACTLY");
    check(chi_square_survival(0.0, 2.0) == 1.0, "C1 chi2_sf(0, df=2) == 1.0 EXACTLY");
    check(chi_square_survival(0.0, 3.0) == 1.0, "C1 chi2_sf(0, df=3) == 1.0 EXACTLY");
    check(chi_square_survival(0.0, 50.0) == 1.0, "C1 chi2_sf(0, df=50) == 1.0 EXACTLY");

    // The GUARDS: a negative statistic and a non-positive df.
    check(throws_with([&] { chi_square_survival(-1.0, 2.0); }, "must not be negative"),
          "C1 chi2_survival rejects a NEGATIVE statistic");
    check(throws_with([&] { chi_square_survival(1.0, 0.0); }, "degrees of freedom must be positive"),
          "C1 chi2_survival rejects df == 0");
    check(throws_with([&] { chi_square_survival(1.0, -2.0); }, "degrees of freedom must be positive"),
          "C1 chi2_survival rejects a NEGATIVE df");
}

// ---------------------------------------------------------------------------
// C2/C3/C4. COHEN'S KAPPA.
//
// C4 (the orientation of the expected matrix) IS NOT TESTED HERE, AND THAT
// IS DELIBERATE. scikit-learn builds expected = outer(col_sums, row_sums) /
// N, the transpose of the conventional E_ij = r_i c_j / N. Every weight
// matrix this module supports (the off-diagonal indicator, |i-j|, (i-j)^2)
// is SYMMETRIC, and for symmetric w the sums over w_ij E_ij and over
// w_ij E_ji are equal by relabelling -- so the transposition cancels
// EXACTLY. It is also unobservable from the C++ API: the expected matrix is
// not exposed, and the only trace-like quantity the API reports,
// expected_agreement = sum_i E_ii / N, is symmetric under the transposition
// as well. A test that "checked" it would be checking nothing.
// ---------------------------------------------------------------------------
static void test_c2_cohen_kappa() {
    printf("\n--- C2/C3: Cohen's kappa ---\n");

    const std::vector<double> a = f2_k_a;
    const std::vector<double> b = f2_k_b;

    // C2: THE THREE KAPPA VALUES.
    KappaResult ku = cohen_kappa(col(a), col(b), f2_k_classes, KappaWeights::Unweighted);
    KappaResult kl = cohen_kappa(col(a), col(b), f2_k_classes, KappaWeights::Linear);
    KappaResult kq = cohen_kappa(col(a), col(b), f2_k_classes, KappaWeights::Quadratic);
    check_near(ku.kappa, f2_k_unweighted_kappa, 1e-13, "F2 the unweighted kappa");
    check_near(kl.kappa, f2_k_linear_kappa, 1e-13, "F2 the LINEAR kappa");
    check_near(kq.kappa, f2_k_quadratic_kappa, 1e-13, "F2 the QUADRATIC kappa");

    // C2: THE WEIGHTED ANSWERS MUST DIFFER FROM THE UNWEIGHTED ONE,
    // otherwise the fixture would not discriminate the label space.
    check(std::fabs(kl.kappa - ku.kappa) > 1e-3,
          "F2 the LINEAR kappa DIFFERS from the unweighted one (the fixture discriminates)");
    check(std::fabs(kq.kappa - ku.kappa) > 1e-3,
          "F2 the QUADRATIC kappa DIFFERS from the unweighted one (the fixture discriminates)");
    check(std::fabs(kq.kappa - kl.kappa) > 1e-3,
          "F2 the QUADRATIC kappa DIFFERS from the LINEAR one (the weight schemes are not the same)");

    // C2: THE FIXTURE REALLY HAS AN ABSENT INTERMEDIATE CLASS. Derived from
    // the fixture's own vectors, not typed: class 2 never appears.
    bool class2_absent = true, class3_present = false;
    for (double v : a) { if (v == 2.0) class2_absent = false; if (v == 3.0) class3_present = true; }
    for (double v : b) { if (v == 2.0) class2_absent = false; if (v == 3.0) class3_present = true; }
    check(class2_absent, "F2 class 2 is ABSENT from the fixture (so the compressed reading differs)");
    check(class3_present, "F2 class 3 IS present (so the absent class is INTERMEDIATE)");

    // C2: THE COMPRESSED READING IS A DIFFERENT NUMBER. The same data,
    // relabelled onto the present classes only, gives a different linear
    // kappa -- and the module must NOT report that one.
    {
        std::vector<double> ca, cb;
        for (double v : a) ca.push_back(v == 3.0 ? 2.0 : v);
        for (double v : b) cb.push_back(v == 3.0 ? 2.0 : v);
        const KappaResult compressed =
            cohen_kappa(col(ca), col(cb), 3, KappaWeights::Linear);
        check(std::fabs(compressed.kappa - kl.kappa) > 1e-3,
              "F2 the COMPRESSED (present-labels) linear kappa is a DIFFERENT number");
        check_near(kl.kappa, f2_k_linear_kappa, 1e-13,
                   "F2 the module reports the EXPLICIT class-index space kappa");
    }

    // THE AGREEMENT COMPONENTS, against the oracle's values. These are the
    // cross-check that the module follows the oracle's p_o / p_e
    // conventions and not merely its kappa.
    check_near(ku.observed_agreement, f2_k_unweighted_po, 1e-13, "F2 the observed agreement p_o");
    check_near(ku.expected_agreement, f2_k_unweighted_pe, 1e-13, "F2 the expected agreement p_e");
    check_near(kl.observed_agreement, f2_k_linear_po, 1e-13, "F2 the observed agreement (linear)");
    check_near(kl.expected_agreement, f2_k_linear_pe, 1e-13, "F2 the expected agreement (linear)");
    check_near(kq.observed_agreement, f2_k_quadratic_po, 1e-13, "F2 the observed agreement (quadratic)");
    check_near(kq.expected_agreement, f2_k_quadratic_pe, 1e-13, "F2 the expected agreement (quadratic)");

    // The counts.
    check(ku.num_total == a.size(), "F2 num_total is the number of items");
    size_t agreements = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] == b[i]) ++agreements;
    check(ku.num_agreements == agreements, "F2 num_agreements is the number of agreeing items");

    // The unweighted kappa must equal the textbook (p_o - p_e) / (1 - p_e).
    check_near(ku.kappa,
               (ku.observed_agreement - ku.expected_agreement) /
                   (1.0 - ku.expected_agreement),
               1e-12, "F2 the unweighted kappa == (p_o - p_e) / (1 - p_e)");

    // ---- THE INDEPENDENT SCALAR-LOOP REFERENCE ------------------------
    // This is what a comparison against the library's own output cannot do:
    // it is a SECOND implementation, written from the definition.
    {
        const ReferenceKappa ru = reference_cohen_kappa(a, b, f2_k_classes, KappaWeights::Unweighted);
        const ReferenceKappa rl = reference_cohen_kappa(a, b, f2_k_classes, KappaWeights::Linear);
        const ReferenceKappa rq = reference_cohen_kappa(a, b, f2_k_classes, KappaWeights::Quadratic);
        check_near(ku.kappa, ru.kappa, 1e-12, "F2 the unweighted kappa matches the INDEPENDENT reference");
        check_near(kl.kappa, rl.kappa, 1e-12, "F2 the linear kappa matches the INDEPENDENT reference");
        check_near(kq.kappa, rq.kappa, 1e-12, "F2 the quadratic kappa matches the INDEPENDENT reference");
        check_near(ku.observed_agreement, ru.observed_agreement, 1e-13,
                   "F2 p_o matches the INDEPENDENT reference");
        check_near(ku.expected_agreement, ru.expected_agreement, 1e-13,
                   "F2 p_e matches the INDEPENDENT reference");
        check(ku.num_agreements == ru.num_agreements,
              "F2 num_agreements matches the INDEPENDENT reference");
    }

    // ---- C3: THE DEGENERATE KAPPA IS NaN, NOT 0.0 --------------------
    {
        const std::vector<double> one(4, 0.0);
        const KappaResult du = cohen_kappa(col(one), col(one), f3_k_classes, KappaWeights::Unweighted);
        const KappaResult dl = cohen_kappa(col(one), col(one), f3_k_classes, KappaWeights::Linear);
        const KappaResult dq = cohen_kappa(col(one), col(one), f3_k_classes, KappaWeights::Quadratic);
        check(std::isnan(du.kappa), "F3 the degenerate UNWEIGHTED kappa is NaN");
        check(std::isnan(dl.kappa), "F3 the degenerate LINEAR kappa is NaN");
        check(std::isnan(dq.kappa), "F3 the degenerate QUADRATIC kappa is NaN");
        check(!std::isnan(du.observed_agreement) && !std::isnan(du.expected_agreement),
              "F3 the degenerate kappa's COMPONENTS are still finite numbers");
        check_near(du.observed_agreement, 1.0, 1e-15, "F3 the degenerate observed agreement is 1");
        check_near(du.expected_agreement, 1.0, 1e-15, "F3 the degenerate expected agreement is 1");
        // The reference agrees that the answer is undefined.
        const ReferenceKappa ru = reference_cohen_kappa(one, one, f3_k_classes, KappaWeights::Unweighted);
        check(std::isnan(ru.kappa), "F3 the INDEPENDENT reference also reports NaN");
    }
}

// ---------------------------------------------------------------------------
// C5. THE MATTHEWS CORRELATION COEFFICIENT.
//
// The DEGENERATE case is 0.0, NOT NaN -- the OPPOSITE of the kappa's
// degenerate case, and the two are not interchangeable.
//
// The F4 fixture is MULTICLASS and its |MCC| is large, so a binary-only
// shortcut (which scikit-learn itself used to take) is visibly wrong.
// ---------------------------------------------------------------------------
static void test_c5_matthews() {
    printf("\n--- C5: the Matthews correlation coefficient ---\n");

    const std::vector<double> t = f4_t;
    const std::vector<double> p = f4_p;

    const MatthewsResult m = matthews_correlation(col(t), col(p), f4_classes);
    check_near(m.mcc, f4_mcc, 1e-13, "F4 the multiclass MCC");
    check(std::fabs(m.mcc) > 0.4,
          "F4 |MCC| is LARGE, so a wrong formula (or a binary-only shortcut) is visibly wrong");
    check(m.num_total == t.size(), "F4 num_total is the number of items");

    size_t correct = 0;
    for (size_t i = 0; i < t.size(); ++i) if (t[i] == p[i]) ++correct;
    check(m.num_correct == correct, "F4 num_correct is the number of correct items");

    // THE INDEPENDENT SCALAR-LOOP REFERENCE.
    check_near(m.mcc, reference_matthews(t, p, f4_classes), 1e-13,
               "F4 the MCC matches the INDEPENDENT reference");

    // The MCC is TRANSPOSE-INVARIANT (a measured property, not a hope):
    // cov_ytyp uses dot(p_sum, t_sum), which is symmetric, and the
    // denominator is a PRODUCT of the two marginal variances. So the F4
    // fixture CANNOT be used to catch a row/column swap -- the suite says so
    // rather than pretending otherwise.
    {
        const MatthewsResult swapped = matthews_correlation(col(p), col(t), f4_classes);
        check(swapped.mcc == m.mcc, "F4 the MCC is TRANSPOSE-INVARIANT (measured, not hoped for)");
    }

    // The MCC is INVARIANT TO THE LABEL SPACE: the formula only uses the
    // marginal counts. The KAPPA is not -- the two are different statistics.
    {
        const MatthewsResult wider = matthews_correlation(col(t), col(p), f4_classes + 2);
        check_near(wider.mcc, f4_mcc, 1e-13, "F4 the MCC is invariant to the label space");
    }

    // ---- C5: THE DEGENERATE MCC IS 0.0, NOT NaN ----------------------
    {
        const std::vector<double> dt = f5_t;
        const std::vector<double> dp = f5_p;
        const MatthewsResult dm = matthews_correlation(col(dt), col(dp), f5_classes);
        check(dm.mcc == 0.0, "F5 the degenerate MCC is 0.0 EXACTLY");
        check(!std::isnan(dm.mcc), "F5 the degenerate MCC is NOT NaN (the OPPOSITE of the kappa)");
        check_near(dm.mcc, f5_mcc, 1e-15, "F5 the degenerate MCC matches the oracle's value");
        check_near(dm.mcc, reference_matthews(dt, dp, f5_classes), 1e-15,
                   "F5 the degenerate MCC matches the INDEPENDENT reference");
        check(dm.num_total == dt.size(), "F5 num_total is the number of items");
    }
}

// ---------------------------------------------------------------------------
// C6, C6-bis, C6-ter. PEARSON'S CHI-SQUARE TEST OF INDEPENDENCE.
//
// C6       the correction is applied ONLY to a 2x2 table.
// C6-bis   the correction magnitude is CLAMPED to the cell's OWN deviation.
// C6-ter   the degrees of freedom are (R-1)(C-1), NOT the cell count - 1.
// ---------------------------------------------------------------------------
static void test_c6_chi_square() {
    printf("\n--- C6/C6-bis/C6-ter: the chi-square test of independence ---\n");

    // ---- F6: THE ORDINARY 2x2, CORRECTED AND UNCORRECTED --------------
    {
        const Tensor table = mat(f6_table);
        const ChiSquareTestResult corrected = chi_square_test_of_independence(table, true);
        const ChiSquareTestResult uncorrected = chi_square_test_of_independence(table, false);
        check_near(corrected.statistic, f6_corrected_stat, 1e-13, "F6 the 2x2 CORRECTED statistic");
        check_near(uncorrected.statistic, f6_uncorrected_stat, 1e-13, "F6 the 2x2 UNCORRECTED statistic");
        check_near(corrected.p_value, f6_corrected_p, 1e-12, "F6 the 2x2 CORRECTED p-value");
        check_near(uncorrected.p_value, f6_uncorrected_p, 1e-12, "F6 the 2x2 UNCORRECTED p-value");
        check(corrected.degrees_of_freedom == f6_dof, "F6 the 2x2 degrees of freedom");
        check(corrected.statistic != uncorrected.statistic,
              "F6 the corrected and uncorrected statistics DIFFER (the two forms are different answers)");
        check(corrected.statistic < uncorrected.statistic,
              "F6 the corrected statistic is the SMALLER one (the correction shrinks the deviation)");
        check(corrected.min_expected_count > 0.0, "F6 the minimum expected count is reported and positive");
    }

    // ---- F8: THE YATES CLAMP DISCRIMINATOR ---------------------------
    // Every cell of [[1,1],[1,2]] has |O - E| = 0.2 < 0.5, so the CLAMPED
    // statistic is 0.0 and the UNCLAMPED (textbook) one would be 0.3125.
    {
        const Tensor table = mat(f8_table);
        const ChiSquareTestResult corrected = chi_square_test_of_independence(table, true);
        const ChiSquareTestResult uncorrected = chi_square_test_of_independence(table, false);
        check(corrected.statistic == 0.0, "F8 the CLAMPED Yates statistic is 0.0 on [[1,1],[1,2]]");
        check_near(corrected.statistic, f8_corrected_stat, 1e-15, "F8 the CLAMPED statistic matches the oracle");
        check_near(uncorrected.statistic, f8_uncorrected_stat, 1e-13, "F8 the UNCORRECTED statistic is the plain Pearson");
        check_near(corrected.p_value, f8_corrected_p, 1e-15, "F8 the CLAMPED corrected p-value is 1");
        check(corrected.statistic != uncorrected.statistic,
              "F8 the corrected and uncorrected statistics DIFFER (the two forms are different answers)");
        // The textbook (UNCLAMPED) form would be 0.3125 here. Assert the
        // module is NOT that: the clamp is the whole point.
        double textbook = 0.0;
        {
            // E_ij for the 2x2 [[1,1],[1,2]]: N = 5.
            const double N = 5.0;
            const double e[4] = {2.0 * 2.0 / N, 2.0 * 3.0 / N, 3.0 * 2.0 / N, 3.0 * 3.0 / N};
            const double o[4] = {1.0, 1.0, 1.0, 2.0};
            for (int i = 0; i < 4; ++i) {
                const double d = std::fabs(o[i] - e[i]) - 0.5;
                textbook += d * d / e[i];
            }
        }
        check(std::fabs(corrected.statistic - textbook) > 1e-3,
              "F8 the statistic is NOT the UNCLAMPED textbook form (the clamp bites)");
    }

    // ---- F7: THE 2x3 TABLE, df == 2 (C6-ter) --------------------------
    {
        const Tensor table = mat(f7_table);
        const ChiSquareTestResult res = chi_square_test_of_independence(table, true);
        check(res.degrees_of_freedom == 2, "F7 the 2x3 degrees of freedom are 2");
        check(res.degrees_of_freedom == f7_dof, "F7 the 2x3 degrees of freedom match the oracle");
        check(res.degrees_of_freedom != f7_table.size() * f7_table[0].size() - 1,
              "F7 the degrees of freedom are NOT the cell count minus one");
        check_near(res.statistic, f7_chi2_stat, 1e-13, "F7 the 2x3 statistic");
        check_near(res.p_value, f7_chi2_p, 1e-12, "F7 the 2x3 p-value");

        // C6: THE CORRECTION IS NOT APPLIED TO A 2x3 TABLE. The corrected
        // and the uncorrected runs are the SAME number here.
        const ChiSquareTestResult uncorrected = chi_square_test_of_independence(table, false);
        check(res.statistic == uncorrected.statistic,
              "F7 the 2x3 statistic is the SAME with and without the correction flag");

        // C6-ter: df == 2 REQUIRES THE GENERAL INCOMPLETE GAMMA. The df = 1
        // closed form is a different number, so the p-value pins the general
        // path.
        const double df1_closed = std::erfc(std::sqrt(res.statistic / 2.0));
        check(std::fabs(res.p_value - df1_closed) > 1e-3,
              "F7 the p-value is NOT the df=1 closed form (the general incomplete gamma is needed)");
        check_near(res.p_value, reference_gamma_q(1.0, res.statistic / 2.0), 1e-13,
                   "F7 the p-value matches the INDEPENDENT incomplete-gamma reference");
    }

    // The p-value of a table is the chi-square survival function at the
    // reported statistic and the reported degrees of freedom.
    {
        const ChiSquareTestResult res = chi_square_test_of_independence(mat(f6_table), false);
        check_near(res.p_value, chi_square_survival(res.statistic,
                                                    static_cast<double>(res.degrees_of_freedom)),
                   1e-15, "F6 the p-value is chi2_sf(statistic, df)");
    }
}

// ---------------------------------------------------------------------------
// C7/C8. THE EMPTY-ROW GUARD, AND THE ZERO OBSERVED CELL.
//
// C7: an empty row or column makes an expected count zero and the statistic
// UNDEFINED. scipy raises, R returns NaN, and this module THROWS
// std::invalid_argument.
//
// C8: a zero observed cell contributes EXACTLY 0 to the G-test and is
// SKIPPED. The naive expression 0 * log(0 / E) is 0 * (-inf) = NaN and it
// destroys the whole statistic.
// ---------------------------------------------------------------------------
static void test_c7_c8_empty_rows_and_zero_cells() {
    printf("\n--- C7/C8: empty rows, empty columns, zero cells ---\n");

    // ---- C7: AN EMPTY ROW (a zero expected count) THROWS --------------
    {
        const Tensor empty_row = mat(std::vector<std::vector<double>>{{0.0, 0.0}, {3.0, 3.0}});
        check(throws_with([&] { chi_square_test_of_independence(empty_row, true); },
                          "empty row or column"),
              "C7 the chi-square test rejects a table with an EMPTY ROW");
        check(throws_with([&] { chi_square_test_of_independence(empty_row, false); },
                          "empty row or column"),
              "C7 the chi-square test rejects an empty row with the correction OFF too");
        check(throws_with([&] { g_test_of_independence(empty_row); }, "empty row or column"),
              "C7 the G-test rejects a table with an EMPTY ROW");
    }
    // ---- C7: AN EMPTY COLUMN THROWS TOO -------------------------------
    {
        const Tensor empty_col = mat(std::vector<std::vector<double>>{{3.0, 0.0}, {3.0, 0.0}});
        check(throws_with([&] { chi_square_test_of_independence(empty_col, true); },
                          "empty row or column"),
              "C7 the chi-square test rejects a table with an EMPTY COLUMN");
        check(throws_with([&] { g_test_of_independence(empty_col); }, "empty row or column"),
              "C7 the G-test rejects a table with an EMPTY COLUMN");
    }

    // ---- C8: THE ZERO OBSERVED CELL ----------------------------------
    {
        const Tensor table = mat(f9_table);
        const GTestResult res = g_test_of_independence(table);
        check(std::isfinite(res.statistic), "F9 the G statistic is FINITE on a table with a ZERO CELL");
        check(std::isfinite(res.p_value), "F9 the G p-value is FINITE on a table with a ZERO CELL");
        check(!std::isnan(res.statistic), "F9 the G statistic is NOT NaN (the zero cell contributed exactly 0)");
        check_near(res.statistic, f9_g_stat, 1e-13, "F9 the G statistic with a zero cell");
        check_near(res.p_value, f9_g_p, 1e-12, "F9 the G p-value with a zero cell");
        check(res.degrees_of_freedom == f9_dof, "F9 the G degrees of freedom");

        // The zero cell contributes EXACTLY 0 -- the same as the naive
        // "skip the zero cells" sweep the oracle's reference performs.
        double expected = 0.0;
        {
            const size_t R = f9_table.size(), C = f9_table[0].size();
            double N = 0.0;
            std::vector<double> row_sums(R, 0.0), col_sums(C, 0.0);
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    N += f9_table[i][j];
                    row_sums[i] += f9_table[i][j];
                    col_sums[j] += f9_table[i][j];
                }
            }
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    const double o = f9_table[i][j];
                    if (o <= 0.0) continue;  // the zero cell contributes 0
                    const double e = row_sums[i] * col_sums[j] / N;
                    expected += o * std::log(o / e);
                }
            }
            expected *= 2.0;
        }
        check_near(res.statistic, expected, 1e-13,
                   "F9 the G statistic matches the zero-cell-skipping sweep");
        check(res.statistic > 0.0, "F9 the G statistic is positive (the table is not independent)");
    }

    // ---- THE 2x3 G-TEST (the general path again) ----------------------
    {
        const GTestResult res = g_test_of_independence(mat(f9_g_2x3_table));
        check_near(res.statistic, f9_g_2x3_stat, 1e-13, "F9 the 2x3 G statistic");
        check_near(res.p_value, f9_g_2x3_p, 1e-12, "F9 the 2x3 G p-value");
        check(res.degrees_of_freedom == f9_g_2x3_dof, "F9 the 2x3 G degrees of freedom are 2");
    }
}

// ---------------------------------------------------------------------------
// C9, C9-bis, C9-ter, C9-quater, C10, C11. McNEMAR'S TEST.
//
// The statistic follows R's stats::mcnemar.test EXACTLY:
//   * the continuity correction is applied IFF n01 != n10;
//   * the corrected numerator is (|n01 - n10| - 1)^2 -- WITH the absolute
//     value;
//   * when n01 == n10 the statistic is 0, NOT 1 / (2 * n01);
//   * when n01 + n10 == 0 the statistic is NaN;
//   * the exact p-value is the repo's existing two-sided binomial tail.
// ---------------------------------------------------------------------------
static void test_c9_mcnemar() {
    printf("\n--- C9/C10/C11: McNemar's test ---\n");

    // ---- F10: ASYMMETRIC -- THE CORRECTION IS APPLIED ----------------
    {
        const McNemarResult res = mcnemar_test(col(f10_a), col(f10_b), true);
        check_near(res.statistic, f10_mcnemar_stat, 1e-13, "F10 the ASYMMETRIC statistic (the correction IS applied)");
        check_near(res.p_value, f10_mcnemar_p, 1e-12, "F10 the ASYMMETRIC p-value");
        check_near(res.exact_p_value, f10_mcnemar_exact_p, 1e-13, "F10 the EXACT p-value");
        check(res.n01 == static_cast<size_t>(f10_n01), "F10 n01");
        check(res.n10 == static_cast<size_t>(f10_n10), "F10 n10");
        check(res.n_discordant == res.n01 + res.n10, "F10 n_discordant == n01 + n10");

        // The corrected and the uncorrected statistics are DIFFERENT
        // numbers on an asymmetric table.
        const McNemarResult uncorrected = mcnemar_test(col(f10_a), col(f10_b), false);
        check_near(uncorrected.statistic, f10_mcnemar_uncorr_stat, 1e-13,
                   "F10 the UNCORRECTED statistic");
        check(res.statistic != uncorrected.statistic,
              "F10 the corrected and uncorrected statistics DIFFER");
        check(res.statistic < uncorrected.statistic,
              "F10 the corrected statistic is the SMALLER one");

        // The counts, recomputed here from the fixture's own vectors.
        size_t n01 = 0, n10 = 0;
        for (size_t i = 0; i < f10_a.size(); ++i) {
            if (f10_a[i] == 0.0 && f10_b[i] == 1.0) ++n01;
            if (f10_a[i] == 1.0 && f10_b[i] == 0.0) ++n10;
        }
        check(res.n01 == n01, "F10 n01 matches the count recomputed from the fixture");
        check(res.n10 == n10, "F10 n10 matches the count recomputed from the fixture");

        // The exact p-value against the INDEPENDENT binomial reference.
        check_near(res.exact_p_value,
                   reference_binom_pvalue_half(static_cast<int>(std::min(n01, n10)),
                                               static_cast<int>(std::max(n01, n10))),
                   1e-13, "F10 the exact p-value matches the INDEPENDENT binomial reference");
    }

    // ---- F11: SYMMETRIC -- THE CORRECTION IS SKIPPED (THE FIXTURE) ----
    {
        const McNemarResult res = mcnemar_test(col(f11_a), col(f11_b), true);
        check(res.statistic == 0.0, "F11 the SYMMETRIC statistic is EXACTLY 0 (the correction is SKIPPED)");
        check_near(res.statistic, f11_mcnemar_stat, 1e-15, "F11 the SYMMETRIC statistic matches the oracle");
        check_near(res.p_value, f11_mcnemar_p, 1e-15, "F11 the SYMMETRIC p-value is 1");
        check(res.n01 == res.n10, "F11 the SYMMETRIC fixture really has n01 == n10");
        // The naive implementation that ALWAYS subtracts 1 would report
        // 1 / (2 * n01) here. Assert the module is NOT that.
        const double naive = 1.0 / (2.0 * static_cast<double>(res.n01));
        check(res.statistic != naive,
              "F11 the SYMMETRIC statistic is NOT 1/(2*n01) (the fixture discriminates)");
        check(std::fabs(res.statistic - naive) > 1e-6,
              "F11 the SYMMETRIC statistic differs from 1/(2*n01) by a wide margin");
        // And the corrected and the uncorrected runs agree here: on a
        // symmetric table the flag changes NOTHING.
        const McNemarResult uncorrected = mcnemar_test(col(f11_a), col(f11_b), false);
        check(res.statistic == uncorrected.statistic,
              "F11 the SYMMETRIC statistic is the SAME with and without the correction flag");
    }

    // ---- F12: n01 < n10 -- THE ABSOLUTE VALUE MATTERS ------------------
    {
        const McNemarResult res = mcnemar_test(col(f12_a), col(f12_b), true);
        check(res.n01 < res.n10, "F12 the fixture really has n01 < n10");
        check_near(res.statistic, f12_mcnemar_stat, 1e-13, "F12 the statistic uses the ABSOLUTE value");
        check_near(res.p_value, f12_mcnemar_p, 1e-12, "F12 the p-value");
        // The SIGNED form (n01 - n10 - 1)^2 would give (3-5-1)^2/8 = 1.125.
        const double signed_form = std::pow(static_cast<double>(res.n01) -
                                                static_cast<double>(res.n10) - 1.0,
                                            2.0) / static_cast<double>(res.n_discordant);
        check(std::fabs(res.statistic - signed_form) > 1e-3,
              "F12 the statistic is NOT the SIGNED form (the absolute value matters)");
        // And the absolute form is exactly right.
        const double absolute_form = std::pow(std::fabs(static_cast<double>(res.n01) -
                                                          static_cast<double>(res.n10)) - 1.0,
                                              2.0) / static_cast<double>(res.n_discordant);
        check_near(res.statistic, absolute_form, 1e-15,
                   "F12 the statistic is (|n01-n10| - 1)^2 / (n01+n10)");
    }

    // ---- F13: NO DISCORDANT PAIRS -> NaN ------------------------------
    {
        const McNemarResult res = mcnemar_test(col(f13_a), col(f13_b), true);
        check(std::isnan(res.statistic), "F13 the statistic is NaN with no discordant pairs");
        check(std::isnan(res.p_value), "F13 the p-value is NaN with no discordant pairs");
        check(res.statistic != 0.0, "F13 the statistic is NOT 0 (it is 0/0)");
        check(res.n_discordant == 0, "F13 n_discordant is 0");
        check(res.n01 == 0 && res.n10 == 0, "F13 n01 and n10 are 0");
        check_near(res.exact_p_value, f13_mcnemar_exact_p, 1e-15,
                   "F13 the exact p-value is 1 (a vacuous test is not evidence)");
        check_near(res.exact_p_value, reference_binom_pvalue_half(0, 0), 1e-15,
                   "F13 the exact p-value matches the INDEPENDENT binomial reference");
    }
}

// ---------------------------------------------------------------------------
// C12, C12-bis, C13. FISHER'S EXACT TEST.
//
// The two-sided p-value uses the PROBABILITY ORDERING rule with scipy's
// 1 + 1e-14 RELATIVE tie tolerance, NOT a doubled one-sided tail. The
// odds ratio is the UNCONDITIONAL cross-product (a*d)/(b*c), NOT R's
// conditional MLE.
// ---------------------------------------------------------------------------
static void test_c12_c13_fisher() {
    printf("\n--- C12/C13: Fisher's exact test ---\n");

    // ---- F14: A 2x2 WITH NO TIES --------------------------------------
    {
        const FisherExactResult res = fisher_exact_test(mat(f14_table));
        check_near(res.p_value, f14_fisher_p, 1e-12, "F14 the p-value (the probability ordering)");
        check_near(res.odds_ratio, f14_fisher_or, 1e-14,
                   "F14 the odds ratio is the CROSS-PRODUCT (not R's conditional MLE)");
        // The cross-product, computed here from the table.
        const double a = f14_table[0][0], b = f14_table[0][1];
        const double c = f14_table[1][0], d = f14_table[1][1];
        check_near(res.odds_ratio, (a * d) / (c * b), 1e-14,
                   "F14 the odds ratio == (a*d)/(b*c), computed from the table");
        // The INDEPENDENT reference sweep.
        check_near(res.p_value, reference_fisher_p(f14_table, true), 1e-12,
                   "F14 the p-value matches the INDEPENDENT probability-ordering sweep");
    }

    // ---- F15: A 2x2 WITH TIES -----------------------------------------
    {
        const FisherExactResult res = fisher_exact_test(mat(f15_table));
        check_near(res.p_value, f15_fisher_p, 1e-12, "F15 the p-value with ties (the probability ordering)");
        check_near(res.odds_ratio, f15_fisher_or, 1e-13, "F15 the odds ratio with ties");
        check_near(res.p_value, reference_fisher_p(f15_table, true), 1e-12,
                   "F15 the p-value matches the INDEPENDENT probability-ordering sweep");
        // C12: THE DOUBLED ONE-SIDED TAIL IS NOT THE TWO-SIDED RULE. The
        // fixture must be able to tell the two apart.
        {
            const double a = f15_table[0][0], b = f15_table[0][1];
            const double c = f15_table[1][0], d = f15_table[1][1];
            const double row1 = a + b, col1 = a + c, total = a + b + c + d;
            double lower = 0.0, upper = 0.0;
            for (double x = 0.0; x <= std::min(row1, col1); x += 1.0) {
                const double p = reference_hypergeom_pmf(x, total, row1, col1);
                if (x <= a) lower += p;
                if (x >= a) upper += p;
            }
            const double doubled = std::min(1.0, 2.0 * std::min(lower, upper));
            check(std::fabs(doubled - res.p_value) > 1e-9,
                  "F15 the p-value is NOT a doubled one-sided tail (the rules differ)");
        }
    }

    // ---- F16: THE TIE-TOLERANCE FIXTURE -------------------------------
    {
        const FisherExactResult res = fisher_exact_test(mat(f16_table));
        check_near(res.p_value, f16_fisher_p, 1e-12,
                   "F16 the p-value with an EXACT pmf tie (the 1 + 1e-14 tolerance)");
        check_near(res.odds_ratio, f16_fisher_or, 1e-15, "F16 the odds ratio with an exact pmf tie");
        check_near(res.p_value, reference_fisher_p(f16_table, true), 1e-12,
                   "F16 the p-value matches the INDEPENDENT sweep WITH the tie tolerance");
        // The tie: two support points share the observed pmf, so the p-value
        // is twice the observed pmf -- and the fixture's p-value is a
        // fraction of 1, not a small number.
        check(res.p_value > 0.4 && res.p_value < 0.5,
              "F16 the p-value is the SUM over the tied support points (both are included)");
    }

    // ---- C12-bis: THE TIE TOLERANCE IS OBSERVABLE IN THIS BUILD --------
    // The F16 table's tie is BIT-EXACT under glibc's std::lgamma, so on
    // this host the tolerance happens to change nothing for IT. The table
    // [[0,2],[15,13]] -- named in the module header's C14-bis section --
    // has a tie that is INEXACT under both libms, and there the tolerance
    // is the difference between including the second support point and not:
    //
    //     with the tolerance   0.48275862068965414
    //     without it           0.24137931034482621  (the p-value HALVES)
    //
    // Both numbers are COMPUTED here by the INDEPENDENT reference sweep, so
    // nothing is hand-typed, and the module must agree with the
    // with-tolerance one.
    {
        const std::vector<std::vector<double>> tie_table = {{0.0, 2.0}, {15.0, 13.0}};
        const double with_tolerance = reference_fisher_p(tie_table, true);
        const double without_tolerance = reference_fisher_p(tie_table, false);
        check(std::fabs(with_tolerance - without_tolerance) > 1e-6,
              "C12-bis the tie table's p-value really does depend on the tolerance");
        const FisherExactResult res = fisher_exact_test(mat(tie_table));
        check_near(res.p_value, with_tolerance, 1e-12,
                   "C12-bis the module's p-value is the WITH-tolerance sweep");
        check(std::fabs(res.p_value - without_tolerance) > 1e-6,
              "C12-bis the module's p-value is NOT the sweep WITHOUT the tolerance");
    }
}

// ---------------------------------------------------------------------------
// THE GUARDS.
//
// Every one of these must throw std::invalid_argument, with a message that
// names the specific problem. A needle that appears in two different
// messages discriminates nothing between them.
// ---------------------------------------------------------------------------
static void test_guards() {
    printf("\n--- guards ---\n");

    const std::vector<double> good = f2_k_a;
    const std::vector<double> good_b = f2_k_b;

    // ---- MISMATCHED ROW COUNTS ----------------------------------------
    check(throws_with([&] { cohen_kappa(col(good), col(f3_k_b), f2_k_classes); },
                      "same number of rows"),
          "cohen_kappa rejects a row-count mismatch");
    check(throws_with([&] { matthews_correlation(col(good), col(f3_k_b), f2_k_classes); },
                      "same number of rows"),
          "matthews_correlation rejects a row-count mismatch");
    {
        // **THE FIXTURE MUST ISOLATE THE GUARD.** f10_a and f11_a are BOTH
        // 40 items long, so pairing them would NOT be a row-count mismatch
        // at all. The shorter vector is built here.
        std::vector<double> shorter(f10_a.begin(), f10_a.end() - 1);
        check(throws_with([&] { mcnemar_test(col(f10_a), col(shorter), true); },
                          "same number of rows"),
              "mcnemar_test rejects a row-count mismatch");
    }

    // ---- A NON-INTEGER CLASS INDEX ------------------------------------
    {
        std::vector<double> fractional = good;
        fractional[2] = 0.5;
        check(throws_with([&] { cohen_kappa(col(fractional), col(good_b), f2_k_classes); },
                          "non-integer class index"),
              "cohen_kappa rejects a NON-INTEGER class index");
        check(throws_with([&] { matthews_correlation(col(fractional), col(good_b), f2_k_classes); },
                          "non-integer class index"),
              "matthews_correlation rejects a NON-INTEGER class index");
    }

    // ---- AN OUT-OF-RANGE CLASS INDEX ----------------------------------
    {
        std::vector<double> out_of_range = good;
        out_of_range[2] = static_cast<double>(f2_k_classes);  // == num_classes
        check(throws_with([&] { cohen_kappa(col(out_of_range), col(good_b), f2_k_classes); },
                          "class index out of range"),
              "cohen_kappa rejects an OUT-OF-RANGE class index");
        check(throws_with([&] { matthews_correlation(col(out_of_range), col(good_b), f2_k_classes); },
                          "class index out of range"),
              "matthews_correlation rejects an OUT-OF-RANGE class index");
        std::vector<double> negative = good;
        negative[2] = -1.0;
        check(throws_with([&] { cohen_kappa(col(negative), col(good_b), f2_k_classes); },
                          "class index out of range"),
              "cohen_kappa rejects a NEGATIVE class index");
    }

    // ---- num_classes == 0 ---------------------------------------------
    check(throws_with([&] { cohen_kappa(col(good), col(good_b), 0); }, "num_classes must be positive"),
          "cohen_kappa rejects num_classes == 0");
    check(throws_with([&] { matthews_correlation(col(good), col(good_b), 0); },
                      "num_classes must be positive"),
          "matthews_correlation rejects num_classes == 0");

    // ---- A LABEL VECTOR THAT IS NOT A SINGLE COLUMN -------------------
    // **THE FIXTURE MUST ISOLATE THE GUARD.** A multi-column tensor whose
    // row count disagrees with its partner would trip the ROW-COUNT guard
    // first, and the single-column guard would never be reached. So the
    // wide tensors below have the SAME row count as their partners.
    {
        Tensor wide(good_b.size(), 2);
        for (size_t i = 0; i < good_b.size(); ++i) { wide(i, 0) = 0.0; wide(i, 1) = 1.0; }
        check(throws_with([&] { cohen_kappa(wide, col(good_b), f2_k_classes); }, "single column"),
              "cohen_kappa rejects a multi-column label tensor");
        check(throws_with([&] { matthews_correlation(wide, col(good_b), f2_k_classes); },
                          "single column"),
              "matthews_correlation rejects a multi-column label tensor");
        Tensor wide_b(f10_b.size(), 2);
        for (size_t i = 0; i < f10_b.size(); ++i) { wide_b(i, 0) = 0.0; wide_b(i, 1) = 1.0; }
        check(throws_with([&] { mcnemar_test(wide_b, col(f10_b), true); }, "single column"),
              "mcnemar_test rejects a multi-column outcome tensor");
    }

    // ---- AN EMPTY LABEL VECTOR ----------------------------------------
    check(throws_with([&] { cohen_kappa(Tensor(0, 1), Tensor(0, 1), f2_k_classes); }, "at least one row"),
          "cohen_kappa rejects an empty label vector");

    // ---- FISHER: A TABLE THAT IS NOT 2x2 ------------------------------
    {
        const Tensor not_2x2 = mat(std::vector<std::vector<double>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
        check(throws_with([&] { fisher_exact_test(not_2x2); }, "must be 2x2"),
              "fisher_exact_test rejects a table that is NOT 2x2");
        const Tensor two_by_three = mat(std::vector<std::vector<double>>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        check(throws_with([&] { fisher_exact_test(two_by_three); }, "must be 2x2"),
              "fisher_exact_test rejects a 3x2 table");
    }

    // ---- FISHER: A NEGATIVE COUNT -------------------------------------
    {
        const Tensor negative = mat(std::vector<std::vector<double>>{{-1.0, 2.0}, {3.0, 4.0}});
        check(throws_with([&] { fisher_exact_test(negative); }, "negative count"),
              "fisher_exact_test rejects a NEGATIVE count");
    }

    // ---- FISHER: A COUNT THAT IS NOT A WHOLE NUMBER -------------------
    {
        const Tensor fractional = mat(std::vector<std::vector<double>>{{1.5, 2.0}, {3.0, 4.0}});
        check(throws_with([&] { fisher_exact_test(fractional); }, "whole number"),
              "fisher_exact_test rejects a count that is NOT a whole number");
    }

    // ---- McNEMAR: AN OUTCOME THAT IS NEITHER 0 NOR 1 ------------------
    {
        std::vector<double> bad = f10_a;
        bad[3] = 0.5;
        check(throws_with([&] { mcnemar_test(col(bad), col(f10_b), true); }, "neither 0 nor 1"),
              "mcnemar_test rejects an outcome that is NEITHER 0 NOR 1");
    }

    // ---- THE CHI-SQUARE TEST: A TABLE WITH TOO FEW ROWS OR COLUMNS ----
    {
        const Tensor one_by_two = mat(std::vector<std::vector<double>>{{1.0, 2.0}});
        check(throws_with([&] { chi_square_test_of_independence(one_by_two, true); },
                          "at least two rows and two columns"),
              "the chi-square test rejects a table with fewer than two rows");
        const Tensor two_by_one = mat(std::vector<std::vector<double>>{{1.0}, {2.0}});
        check(throws_with([&] { chi_square_test_of_independence(two_by_one, true); },
                          "at least two rows and two columns"),
              "the chi-square test rejects a table with fewer than two columns");
        check(throws_with([&] { g_test_of_independence(two_by_one); },
                          "at least two rows and two columns"),
              "the G-test rejects a table with fewer than two columns");
    }

    // ---- THE CHI-SQUARE TEST: A NEGATIVE COUNT ------------------------
    {
        const Tensor negative = mat(std::vector<std::vector<double>>{{-1.0, 2.0}, {3.0, 4.0}});
        check(throws_with([&] { chi_square_test_of_independence(negative, true); },
                          "negative count"),
              "the chi-square test rejects a NEGATIVE count");
        check(throws_with([&] { g_test_of_independence(negative); }, "negative count"),
              "the G-test rejects a NEGATIVE count");
    }

    // ---- EVERY GUARD MESSAGE MUST BE DISTINCT -------------------------
    // A needle that appears in two different messages discriminates nothing
    // between them, and this repo has hit that trap repeatedly.
    std::string m_row, m_nonint, m_range, m_2x2, m_negative;
    throws_with([&] { cohen_kappa(col(good), col(f3_k_b), f2_k_classes); }, "", &m_row);
    throws_with([&] {
        std::vector<double> fractional = good;
        fractional[2] = 0.5;
        cohen_kappa(col(fractional), col(good_b), f2_k_classes);
    }, "", &m_nonint);
    throws_with([&] {
        std::vector<double> out_of_range = good;
        out_of_range[2] = static_cast<double>(f2_k_classes);
        cohen_kappa(col(out_of_range), col(good_b), f2_k_classes);
    }, "", &m_range);
    throws_with([&] {
        fisher_exact_test(mat(std::vector<std::vector<double>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    }, "", &m_2x2);
    throws_with([&] {
        fisher_exact_test(mat(std::vector<std::vector<double>>{{-1.0, 2.0}, {3.0, 4.0}}));
    }, "", &m_negative);
    check(!m_row.empty() && !m_nonint.empty() && !m_range.empty() && !m_2x2.empty() &&
              !m_negative.empty(),
          "every guard throws a message");
    check(m_row != m_nonint && m_row != m_range && m_row != m_2x2 && m_row != m_negative &&
              m_nonint != m_range && m_nonint != m_2x2 && m_nonint != m_negative &&
              m_range != m_2x2 && m_range != m_negative && m_2x2 != m_negative,
          "the guard messages are pairwise distinct");
}

int main() {
    printf("============================================================\n");
    printf("tests/test_contingency.cpp -- the contingency statistics\n");
    printf("============================================================\n");

    test_c1_chi_square_survival();
    test_c2_cohen_kappa();
    test_c5_matthews();
    test_c6_chi_square();
    test_c7_c8_empty_rows_and_zero_cells();
    test_c9_mcnemar();
    test_c12_c13_fisher();
    test_guards();

    printf("\n============================================================\n");
    printf("=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    printf("============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
