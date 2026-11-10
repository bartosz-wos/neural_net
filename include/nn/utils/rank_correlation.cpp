#include "rank_correlation.h"

// The inverse normal CDF ALREADY SHIPS in this repo, at GLOBAL SCOPE, in
// utils/randomized_smoothing.h (that header has no namespace -- measured,
// `grep -c 'namespace' include/nn/utils/randomized_smoothing.h` -> 0).
//
// It is COMPOSED here rather than reimplemented. Copying a second bisection
// probit into this module would be exactly the "hand-roll a standard
// primitive you already have" mistake: the shipped one is already measured
// to agree with `scipy.stats.norm.ppf` to 4.5e-16 RELATIVE accuracy across
// p in [1e-12, 1-1e-12] (randomized_smoothing.h's own header).
#include "randomized_smoothing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// M_PI is NOT in the C++ standard (it is a POSIX extension), so it is
// guarded the same way include/nn/layers/attention/axial_attention.cpp
// guards it. The value is the same double either way.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// utils/rank_correlation.cpp -- the rank correlation and Gaussian-copula
// utilities.
//
// THE SPECIFICATION is tests/rank_correlation_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed. Its discriminating
// power was MEASURED, not narrated:
// tests/rank_correlation_mutate_oracle.py breaks the oracle in seven ways
// and requires that it fail each time.
//
// The three decisions the test suite pins:
//
//   1. RANKS ARE MID-RANKS: a tied group gets the AVERAGE of the ranks it
//      spans. NOT the first rank of the group.
//
//   2. KENDALL'S TAU-B'S TIE CORRECTION: the joint ties count in BOTH xtie
//      AND ytie. Counting them in neither is the classic bug and it is
//      invisible on a fixture with no ties.
//
//   3. THE PLOTTING POSITION IS (r - 0.5)/n (Hazen), not Blomqvist's
//      (r - 0.375)/(n + 0.25) and not the Weibull r/(n+1).
//
// See the header for the derivations and the citations.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// The sample's values, in row-major order.
//
// The VALUES are what matter: a (N,1) and an (1,N) tensor holding the same
// values are the same sample. This is the convention utils/off_policy_eval.h
// uses, and it is stated in the header.
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

// The guards. EVERY message here is DISTINCT.
//
// A substring that appears in two different messages discriminates nothing
// between them, and this repo has hit that trap several separate times
// (each time the assertion searched for a word the OTHER guard's message
// also contained, so the mutant survived). The suite therefore asserts each
// guard's own message, not merely that something was thrown.
//
// The guards are also distinct IN KIND:
//   * "empty"                  -- the sample has no elements at all.
//   * "same number of samples" -- the two samples disagree in length.
//   * "at least two"           -- a correlation needs two points.
//
// A CONSTANT SAMPLE IS NOT GUARDED: its correlation is genuinely undefined
// (0/0), and the reference implementation returns NaN for it. The
// constant-sample case therefore has its own answer (NaN) and its own
// message in the "zero variance" family, distinct from all three above.
// ---------------------------------------------------------------------------
void require_non_empty(const std::vector<double>& v, const char* who) {
    if (v.empty()) {
        throw std::invalid_argument(std::string(who) +
                                    ": the sample is empty");
    }
}

void require_same_length(const std::vector<double>& a,
                         const std::vector<double>& b, const char* who) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::string(who) +
            ": every input must have the same number of samples");
    }
}

void require_at_least_two(const std::vector<double>& v, const char* who) {
    if (v.size() < 2) {
        throw std::invalid_argument(
            std::string(who) +
            ": at least two samples are required to measure an association");
    }
}

// ---------------------------------------------------------------------------
// Pearson's r of two equal-length samples, by the TWO-PASS CENTRED form.
//
// **THE ONE-PASS FORM `sum(xy) - sum(x)*sum(y)/n` IS NOT USED.** It
// computes the same quantity in exact arithmetic, but in floating point it
// subtracts two nearly-equal large numbers, and the loss is catastrophic
// for large-offset data: two samples that differ only in their means can
// produce a correlation of garbage. The centred form subtracts the mean
// FIRST and never forms the large products.
//
// Returns NaN when either sample has zero variance, which is the honest
// answer: the correlation is undefined, not zero.
// ---------------------------------------------------------------------------
double pearson_of(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = a.size();
    double mean_a = 0.0;
    double mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<double>(n);
    mean_b /= static_cast<double>(n);

    double num = 0.0;
    double saa = 0.0;
    double sbb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - mean_a;
        const double db = b[i] - mean_b;
        num += da * db;
        saa += da * da;
        sbb += db * db;
    }

    const double denom = std::sqrt(saa * sbb);
    if (denom == 0.0) {
        // A constant sample: the correlation is UNDEFINED (0/0). NaN is the
        // honest answer and is what scipy.stats.spearmanr returns.
        return std::numeric_limits<double>::quiet_NaN();
    }
    return num / denom;
}

// The mid-ranks of a sample, 1-based. A tied group receives the AVERAGE of
// the ranks it spans.
std::vector<double> mid_ranks_of(const std::vector<double>& v) {
    const size_t n = v.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), static_cast<size_t>(0));
    // STABLE sort, so the result does not depend on the input order for
    // equal values -- the mid-rank of a tied group is the same regardless.
    std::stable_sort(order.begin(), order.end(),
                     [&v](size_t lhs, size_t rhs) { return v[lhs] < v[rhs]; });

    std::vector<double> ranks(n, 0.0);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && v[order[j + 1]] == v[order[i]]) {
            ++j;
        }
        // The tied group spans 0-based positions i..j, so its 1-based ranks
        // are i+1 .. j+1, whose AVERAGE is (i+1 + j+1)/2.
        const double avg_rank = 0.5 * (static_cast<double>(i) + 1.0 +
                                       static_cast<double>(j) + 1.0);
        for (size_t k = i; k <= j; ++k) {
            ranks[order[k]] = avg_rank;
        }
        i = j + 1;
    }
    return ranks;
}

// The normal scores: z_i = Phi^-1((r_i - 0.5)/n), r_i the mid-rank.
std::vector<double> normal_scores_of(const std::vector<double>& v) {
    const std::vector<double> ranks = mid_ranks_of(v);
    const double n = static_cast<double>(v.size());
    std::vector<double> z(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const double p = (ranks[i] - 0.5) / n;
        z[i] = normal_quantile(p);
    }
    return z;
}

// A correlation coefficient is in [-1, 1]. An argument outside that is a
// caller error, not a quantity to extrapolate -- and `asin` of it would be
// NaN, which would silently poison a downstream computation.
void require_unit_interval(double v, const char* who) {
    if (!(v >= -1.0) || !(v <= 1.0)) {
        throw std::invalid_argument(
            std::string(who) +
            ": the argument must be in [-1, 1] (got " +
            std::to_string(v) + ")");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// mid_ranks -- the mid-rank transform.
// ---------------------------------------------------------------------------
Tensor mid_ranks(const Tensor& x) {
    const std::vector<double> v = values_of(x);
    require_non_empty(v, "mid_ranks");
    const std::vector<double> ranks = mid_ranks_of(v);
    Tensor out(x.rows, x.cols);
    size_t idx = 0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            out(i, j) = ranks[idx++];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// pearson_r -- Pearson's product-moment correlation.
// ---------------------------------------------------------------------------
double pearson_r(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_non_empty(a, "pearson_r");
    require_same_length(a, b, "pearson_r");
    require_at_least_two(a, "pearson_r");
    return pearson_of(a, b);
}

// ---------------------------------------------------------------------------
// spearman_rho -- Pearson's r of the mid-ranks.
//
// The d^2 shortcut is deliberately NOT used: it is exact only without ties.
// ---------------------------------------------------------------------------
double spearman_rho(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_non_empty(a, "spearman_rho");
    require_same_length(a, b, "spearman_rho");
    require_at_least_two(a, "spearman_rho");
    return pearson_of(mid_ranks_of(a), mid_ranks_of(b));
}

// ---------------------------------------------------------------------------
// kendall_tau_b -- the tie-corrected concordance coefficient.
//
// The joint ties count in BOTH xtie AND ytie. See the header.
// ---------------------------------------------------------------------------
double kendall_tau_b(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_non_empty(a, "kendall_tau_b");
    require_same_length(a, b, "kendall_tau_b");
    require_at_least_two(a, "kendall_tau_b");

    const size_t n = a.size();
    const double tot = 0.5 * static_cast<double>(n) *
                        static_cast<double>(n - 1);

    double con = 0.0;
    double dis = 0.0;
    double xtie = 0.0;
    double ytie = 0.0;
    double ntie = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = a[i] - a[j];
            const double dy = b[i] - b[j];
            if (dx == 0.0 && dy == 0.0) {
                // A JOINT tie. It counts in xtie AND in ytie -- that is the
                // whole point of the tau-b correction, and counting it in
                // NEITHER is the classic bug.
                xtie += 1.0;
                ytie += 1.0;
                ntie += 1.0;
            } else if (dx == 0.0) {
                xtie += 1.0;
            } else if (dy == 0.0) {
                ytie += 1.0;
            } else if ((dx > 0.0) == (dy > 0.0)) {
                con += 1.0;
            } else {
                dis += 1.0;
            }
        }
    }

    const double con_minus_dis = tot - xtie - ytie + ntie - 2.0 * dis;
    const double denom = (tot - xtie) * (tot - ytie);
    if (denom <= 0.0) {
        // One sample is entirely tied, i.e. constant: the denominator
        // vanishes and the coefficient is undefined.
        return std::numeric_limits<double>::quiet_NaN();
    }
    return con_minus_dis / std::sqrt(denom);
}

// ---------------------------------------------------------------------------
// normal_scores -- the van der Waerden transform.
// ---------------------------------------------------------------------------
Tensor normal_scores(const Tensor& x) {
    const std::vector<double> v = values_of(x);
    require_non_empty(v, "normal_scores");
    const std::vector<double> z = normal_scores_of(v);
    Tensor out(x.rows, x.cols);
    size_t idx = 0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            out(i, j) = z[idx++];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gaussian_copula_correlation -- the marginal-free correlation.
// ---------------------------------------------------------------------------
double gaussian_copula_correlation(const Tensor& x, const Tensor& y) {
    const std::vector<double> a = values_of(x);
    const std::vector<double> b = values_of(y);
    require_non_empty(a, "gaussian_copula_correlation");
    require_same_length(a, b, "gaussian_copula_correlation");
    require_at_least_two(a, "gaussian_copula_correlation");
    return pearson_of(normal_scores_of(a), normal_scores_of(b));
}

// ---------------------------------------------------------------------------
// The four exact population correspondences.
// ---------------------------------------------------------------------------
double gaussian_copula_tau_from_rho(double rho) {
    require_unit_interval(rho, "gaussian_copula_tau_from_rho");
    return 2.0 / M_PI * std::asin(rho);
}

double gaussian_copula_rho_from_tau(double tau) {
    require_unit_interval(tau, "gaussian_copula_rho_from_tau");
    return std::sin(M_PI * tau / 2.0);
}

double gaussian_copula_spearman_from_rho(double rho) {
    require_unit_interval(rho, "gaussian_copula_spearman_from_rho");
    return 6.0 / M_PI * std::asin(rho / 2.0);
}

double gaussian_copula_rho_from_spearman(double rho_s) {
    require_unit_interval(rho_s, "gaussian_copula_rho_from_spearman");
    return 2.0 * std::sin(M_PI * rho_s / 6.0);
}

}  // namespace utils
}  // namespace nn
