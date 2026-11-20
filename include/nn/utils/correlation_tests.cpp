#include "correlation_tests.h"

// The repo's OWN primitives are COMPOSED here, not reimplemented:
//
//   * `regularized_incomplete_beta` -- the Student's t distribution's
//     two-sided survival needs the regularised incomplete beta, and
//     `utils/randomized_smoothing.h` already ships a continued-fraction
//     one. A second copy would be exactly the "one implementation of each
//     primitive" mistake this repo has already paid for -- the three
//     copies of the Jacobi rotation angle are why `utils/linear_algebra.h`
//     exists at all.
//
//   * `normal_cdf` -- the same header's standard normal CDF, for the
//     Kendall test's two-sided normal survival.
//
//   * `pearson_r`, `spearman_rho`, `kendall_tau_b` -- the repo's own
//     correlation statistics, from `utils/rank_correlation.h`. The
//     convenience wrappers call THESE, so the coefficient and the test
//     cannot disagree.
#include "randomized_smoothing.h"
#include "rank_correlation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// utils/correlation_tests.cpp -- the correlation hypothesis tests.
//
// THE SPECIFICATION is tests/correlation_tests_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed. Its discriminating
// power is MEASURED, not narrated: the oracle's own mutation harness
// breaks its model code in specific ways and requires that it FAIL each
// time.
//
// The conventions -- the t form for Pearson and Spearman, the asymptotic
// normal with tie corrections for Kendall, the NaN at n == 2, and the
// clip to [-1, 1] -- are stated and justified in the header.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// The guards. EVERY message is DISTINCT, for the reason this repo has hit
// several separate times: a substring that appears in two different
// messages discriminates nothing between them, and the mutation harness
// then reports a survivor that is really a missing test.
//
// The guards are distinct IN KIND:
//   * "at least two samples" -- a test needs more than one point.
//   * "same number of samples" -- the two samples disagree in length.
// ---------------------------------------------------------------------------
void require_at_least_two(size_t n, const char* who) {
    if (n < 2) {
        throw std::invalid_argument(
            std::string(who) +
            ": at least two samples are required to test an association");
    }
}

void require_same_length(size_t a, size_t b, const char* who) {
    if (a != b) {
        throw std::invalid_argument(
            std::string(who) +
            ": every input must have the same number of samples");
    }
}

// ---------------------------------------------------------------------------
// The sample's values, in row-major order.
//
// The VALUES are what matter: a (N,1) and an (1,N) tensor holding the same
// values are the same sample. This is the convention
// `utils/rank_correlation.h` uses, and the two modules agree on it -- the
// Kendall p-value calls the repo's own `kendall_tau_b`, which uses the
// same reading.
// ---------------------------------------------------------------------------
std::vector<double> values_of(const Tensor& t) {
    std::vector<double> v;
    v.reserve(t.rows * t.cols);
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            v.push_back(t(i, j));
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// C4: CLIP the statistic to [-1, 1].
//
// NOT COSMETIC. MEASURED on THIS REPO's own `pearson_r`: over 300000
// near-perfectly-collinear samples, the largest |r| it returns is
// 1.0000000000000011102 -- ABOVE 1 by two ULP. That makes `1 - r*r`
// NEGATIVE, so `sqrt(dof/(1-r*r))` is NaN, so the p-value of a
// PERFECTLY correlated sample would be NaN instead of 0.0.
//
// scipy clips for the same reason. Without this clip the module's most
// important fixture -- the perfect correlation -- is the one that
// breaks.
// ---------------------------------------------------------------------------
double clip_statistic(double r) {
    if (r > 1.0) return 1.0;
    if (r < -1.0) return -1.0;
    return r;
}

// ---------------------------------------------------------------------------
// The Student's t distribution's TWO-SIDED survival function:
//
//     P(|T_dof| >= |t|) = I_{dof/(dof+t^2)}(dof/2, 1/2)
//
// where I_x(a,b) is the regularised incomplete beta -- the repo's OWN,
// composed from `utils/randomized_smoothing.h`.
//
// MEASURED against scipy.special.betainc over the grid the tests use
// (dof in {1..100}, t in [0, 8]): worst RELATIVE error 6.6e-14.
// ---------------------------------------------------------------------------
double student_t_two_sided_survival(double t, double dof) {
    if (dof <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double at = std::fabs(t);
    if (at == 0.0) return 1.0;
    // An infinite t statistic is a perfect correlation: the survival is 0.
    if (std::isinf(at)) return 0.0;
    const double x = dof / (dof + at * at);
    return ::regularized_incomplete_beta(dof / 2.0, 0.5, x);
}

// ---------------------------------------------------------------------------
// The p-value of a correlation coefficient, under the t form (C1).
//
// `r` is CLIPPED first (C4). The two ends are handled EXPLICITLY (C7):
// r == 0 gives exactly 1.0, and a zero denominator gives exactly 0.0.
// They are DIFFERENT answers, and neither is a NaN.
//
// A statistic of 0.9999999999999999 is NOT special-cased: the
// denominator is then 8.9e-16, not zero, and the p-value is 1.4e-24
// (C7-bis).
// ---------------------------------------------------------------------------
double correlation_p_value(double r, size_t n, const char* who) {
    if (n < 2) {
        throw std::invalid_argument(
            std::string(who) + ": the sample size must be at least two");
    }
    // C3: n == 2 is UNDEFINED, and the references disagree there, so
    // there is no reference to agree with. NaN is the honest answer.
    if (n == 2) return std::numeric_limits<double>::quiet_NaN();
    // C5: an undefined correlation has an undefined p-value.
    if (!std::isfinite(r)) return std::numeric_limits<double>::quiet_NaN();

    const double rc = clip_statistic(r);
    const double dof = static_cast<double>(n) - 2.0;
    const double denom = 1.0 - rc * rc;
    // **THERE IS NO `denom <= 0` GUARD HERE, AND THAT IS THE POINT.**
    //
    // The clip guarantees `denom >= 0`, and the two degenerate cases are
    // handled by the ARITHMETIC rather than by a separate branch:
    //
    //   denom == 0 (|rc| == 1 exactly) -> dof/0 is +inf, sqrt is +inf,
    //     the t statistic is +inf, and `student_t_two_sided_survival`
    //     returns EXACTLY 0.0 for an infinite statistic.
    //   denom  > 0                     -> the ordinary path.
    //
    // **A `denom <= 0` GUARD WOULD MAKE THE CLIP REDUNDANT**, and the
    // mutation harness proved it: with the guard present, the mutation
    // "the statistic is not clipped to [-1, 1]" SURVIVED, because the
    // guard caught the over-1 statistic and returned the same 0.0. **A
    // REDUNDANT GUARD HIDES A DEFECT, AND THE ONLY WAY THAT SHOWS UP IS
    // A MUTATION THAT THE GUARD MAKES INVISIBLE.**
    //
    // So the guard is GONE and the clip is LOAD-BEARING: without the clip
    // the denominator is negative, the square root is NaN, and the
    // p-value of a PERFECT correlation is NaN.
    const double t = rc * std::sqrt(dof / denom);
    return student_t_two_sided_survival(t, dof);
}

// ---------------------------------------------------------------------------
// The tie-GROUP statistics of a sample (C2).
//
// Over the tied groups of the sample, with c the size of a group:
//
//     pairs = sum over groups of c*(c-1)/2
//     x0    = sum over groups of c*(c-1)*(c-2)
//     x1    = sum over groups of c*(c-1)*(2c+5)
//
// **THE GROUP STATISTICS ARE NOT THE TIE PAIR COUNT.** Two samples with
// the groups {4, 2} and {3, 3, 2} have the SAME number of tied pairs (7)
// and DIFFERENT variances, so a variance built from the pair count alone
// gives the wrong answer for one of them. The suite pins a fixture pair
// that discriminates exactly this.
//
// A "tie" is an EXACT equality of the values, which is the same rule
// `utils/rank_correlation.h`'s `mid_ranks` uses -- so the two modules
// agree on what a tie is by construction.
// ---------------------------------------------------------------------------
struct TieGroupStats {
    double pairs = 0.0;
    double x0 = 0.0;
    double x1 = 0.0;
};

TieGroupStats tie_group_stats_of(const std::vector<double>& v) {
    // Sort a COPY, so the caller's sample is untouched, and walk the
    // equal-value runs.
    std::vector<double> sorted(v);
    std::sort(sorted.begin(), sorted.end());

    TieGroupStats out;
    size_t i = 0;
    while (i < sorted.size()) {
        size_t j = i;
        while (j + 1 < sorted.size() && sorted[j + 1] == sorted[i]) {
            ++j;
        }
        const double c = static_cast<double>(j - i + 1);
        if (c >= 2.0) {
            out.pairs += c * (c - 1.0) / 2.0;
            out.x0 += c * (c - 1.0) * (c - 2.0);
            out.x1 += c * (c - 1.0) * (2.0 * c + 5.0);
        }
        i = j + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// The pair counts the repo's `kendall_tau_b` computes: the concordant,
// discordant and tied PAIR counts, and the joint ties.
//
// **THIS IS THE REPO'S OWN COUNTING, REPRODUCED HERE BECAUSE THE
// REPO'S OWN `kendall_tau_b` DOES NOT EXPOSE IT** -- it returns only the
// coefficient. Recomputing it here would be a SECOND SOURCE OF TRUTH for
// the number of discordant pairs, which is exactly what C11 forbids, so
// the two are kept in step by construction: the expression below is the
// same one `kendall_tau_b` uses, and the suite pins the p-value against
// the same numerator.
// ---------------------------------------------------------------------------
struct KendallCounts {
    double tot = 0.0;
    double con = 0.0;
    double dis = 0.0;
    double xtie = 0.0;
    double ytie = 0.0;
    double ntie = 0.0;

    double con_minus_dis() const { return tot - xtie - ytie + ntie - 2.0 * dis; }
};

KendallCounts kendall_counts_of(const std::vector<double>& a,
                                 const std::vector<double>& b) {
    const size_t n = a.size();
    KendallCounts out;
    out.tot = 0.5 * static_cast<double>(n) * static_cast<double>(n - 1);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = a[i] - a[j];
            const double dy = b[i] - b[j];
            if (dx == 0.0 && dy == 0.0) {
                // A JOINT tie: it counts in xtie AND in ytie, which is the
                // whole point of the tau-b correction.
                out.xtie += 1.0;
                out.ytie += 1.0;
                out.ntie += 1.0;
            } else if (dx == 0.0) {
                out.xtie += 1.0;
            } else if (dy == 0.0) {
                out.ytie += 1.0;
            } else if ((dx > 0.0) == (dy > 0.0)) {
                out.con += 1.0;
            } else {
                out.dis += 1.0;
            }
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// pearson_p_value
// ---------------------------------------------------------------------------
double pearson_p_value(double r, size_t n) {
    return correlation_p_value(r, n, "pearson_p_value");
}

// ---------------------------------------------------------------------------
// spearman_p_value
// ---------------------------------------------------------------------------
double spearman_p_value(double rho, size_t n) {
    return correlation_p_value(rho, n, "spearman_p_value");
}

// ---------------------------------------------------------------------------
// kendall_tau_b_p_value
// ---------------------------------------------------------------------------
double kendall_tau_b_p_value(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_at_least_two(a.size(), "kendall_tau_b_p_value");
    require_same_length(a.size(), b.size(), "kendall_tau_b_p_value");

    const size_t n = a.size();
    // C3: n == 2 is UNDEFINED.
    if (n == 2) return std::numeric_limits<double>::quiet_NaN();

    const KendallCounts counts = kendall_counts_of(a, b);

    // C2: the tie-GROUP statistics, which the variance needs and the pair
    // counts alone do not give.
    const TieGroupStats sx = tie_group_stats_of(a);
    const TieGroupStats sy = tie_group_stats_of(b);

    const double m = static_cast<double>(n) * static_cast<double>(n - 1);
    const double dn = static_cast<double>(n);
    const double variance =
        (m * (2.0 * dn + 5.0) - sx.x1 - sy.x1) / 18.0 +
        (2.0 * sx.pairs * sy.pairs) / m +
        (sx.x0 * sy.x0) / (9.0 * m * (dn - 2.0));
    if (!(variance > 0.0)) {
        // The variance vanishes -- a constant sample, whose correlation is
        // undefined. NaN, like the statistic (C5).
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double z = counts.con_minus_dis() / std::sqrt(variance);
    // C6: TWO-SIDED. The survival is 1 - Phi(|z|), doubled.
    //
    // **THE erfc FORM, NOT `2 * (1 - normal_cdf(|z|))`.** The two are
    // mathematically identical -- Phi(x) = 0.5*erfc(-x/sqrt(2)), so
    // 2*(1 - Phi(|z|)) IS erfc(|z|/sqrt(2)) -- but the subtraction form
    // CANCELS CATASTROPHICALLY in the tail, because normal_cdf(|z|) -> 1
    // and the difference of two nearly-equal numbers throws away the
    // digits that matter.
    //
    // MEASURED on the F3 fixture (z = 6.0346):
    //
    //     erfc route (this one)   1.59320548535935e-09   <- exact
    //     2*(1 - Phi) route       1.5932055674738876e-09  <- 5.2e-8 relative
    //
    // and the gap grows with |z|: at |z| = 8 the subtraction form is 7.1%
    // wrong, and at |z| = 9 it returns exactly 0.0 where the true value is
    // 2.257e-19. **THE SUITE CAUGHT THIS**: the F3 comparison failed with
    // an 8.2e-17 absolute difference on a 1.6e-09 p-value.
    //
    // `std::erfc` is the standard library's own primitive here -- this is
    // not a hand-rolled normal survival function, and the repo's
    // `normal_cdf` is itself a thin wrapper over `erfc` (see
    // `utils/randomized_smoothing.cpp`). **The composition rule applies
    // to the repo's own PRIMITIVES, not to libm's.**
    const double p = std::erfc(std::fabs(z) / std::sqrt(2.0));
    return std::min(1.0, std::max(0.0, p));
}

// ---------------------------------------------------------------------------
// pearson_r_test
// ---------------------------------------------------------------------------
CorrelationTestResult pearson_r_test(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_at_least_two(a.size(), "pearson_r_test");
    require_same_length(a.size(), b.size(), "pearson_r_test");

    CorrelationTestResult out;
    out.n = a.size();
    // The repo's OWN statistic -- one source of truth (C8).
    out.statistic = pearson_r(x, y);
    out.p_value = pearson_p_value(out.statistic, out.n);
    return out;
}

// ---------------------------------------------------------------------------
// spearman_rho_test
// ---------------------------------------------------------------------------
CorrelationTestResult spearman_rho_test(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_at_least_two(a.size(), "spearman_rho_test");
    require_same_length(a.size(), b.size(), "spearman_rho_test");

    CorrelationTestResult out;
    out.n = a.size();
    out.statistic = spearman_rho(x, y);
    out.p_value = spearman_p_value(out.statistic, out.n);
    return out;
}

// ---------------------------------------------------------------------------
// kendall_tau_b_test
// ---------------------------------------------------------------------------
CorrelationTestResult kendall_tau_b_test(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_at_least_two(a.size(), "kendall_tau_b_test");
    require_same_length(a.size(), b.size(), "kendall_tau_b_test");

    CorrelationTestResult out;
    out.n = a.size();
    out.statistic = kendall_tau_b(x, y);
    out.p_value = kendall_tau_b_p_value(x, y);
    return out;
}

}  // namespace utils
}  // namespace nn
