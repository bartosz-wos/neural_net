#include "contingency.h"

// The EXACT McNemar p-value is the repo's EXISTING binomial tail.
// utils/randomized_smoothing.h declares `double binom_pvalue_half(int, int)`
// at GLOBAL SCOPE (that header has no namespace -- measured, `grep -c
// 'namespace' include/nn/utils/randomized_smoothing.h` -> 0), and its own
// oracle pins it to scipy.stats.binomtest.
//
// It is COMPOSED here rather than reimplemented (C11): hand-rolling a
// second binomial tail in this module would be exactly the "copy a standard
// primitive you already have" mistake. The composition is checked against
// scipy in tests/contingency_oracle.py.
//
// NOTE: randomized_smoothing.h also pulls in core/model.h. That is the price
// of composing, and it is the price utils/rank_correlation.cpp already pays
// for the same reason (it composes the same header's `normal_quantile`).
// The PUBLIC header of this module does NOT include it -- see the header's
// "THE DEPENDENCIES" section.
#include "randomized_smoothing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// utils/contingency.cpp -- the categorical-agreement and
// contingency-table-inference utilities.
//
// THE SPECIFICATION is tests/contingency_oracle.py (float64 numpy), which was
// run BEFORE this file existed. Its discriminating power is MEASURED, not
// narrated: tests/contingency_mutate_oracle.py breaks the oracle's model code
// in specific ways and requires that it FAIL each time.
//
// The conventions -- the explicit class-index space, the NaN-vs-0.0
// degenerate cases, the CLAMPED Yates correction, the zero-cell guard in the
// G-test, McNemar's symmetric-table rule, the probability-ordering rule in
// Fisher's exact test, and the cross-product odds ratio -- are stated and
// justified in the header.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// Guards. EVERY message is DISTINCT, for the reason this repo has hit
// several separate times: a needle that appears in two different messages
// discriminates nothing between them, and the mutation harness then reports
// a survivor that is really a missing test.
// ---------------------------------------------------------------------------

// The class indices of a (N,1) label tensor, validated against the EXPLICIT
// class-index space 0..num_classes-1.
//
// `who` is a prefix that identifies BOTH the function and the tensor, so
// every message is distinct from every other message in this file.
std::vector<size_t> extract_class_indices(const Tensor& labels, size_t num_classes,
                                           const char* who) {
    if (labels.rows == 0)
        throw std::invalid_argument(std::string(who) + " must have at least one row");
    if (labels.cols != 1)
        throw std::invalid_argument(std::string(who) + " must be a single column");
    std::vector<size_t> out(labels.rows);
    for (size_t i = 0; i < labels.rows; ++i) {
        double v = labels(i, 0);
        if (!std::isfinite(v))
            throw std::invalid_argument(std::string(who) +
                                        " contains a non-finite class index at row " +
                                        std::to_string(i));
        double rounded = std::round(v);
        // The same 1e-12 integrality tolerance utils/metrics.cpp uses, so a
        // class index that is a whole number up to float64 noise is accepted
        // and one that is not (0.5, 2.25) is rejected.
        if (std::fabs(v - rounded) > 1e-12)
            throw std::invalid_argument(std::string(who) +
                                        " contains a non-integer class index at row " +
                                        std::to_string(i));
        if (rounded < 0.0 || rounded >= static_cast<double>(num_classes))
            throw std::invalid_argument(std::string(who) +
                                        " class index out of range at row " +
                                        std::to_string(i));
        out[i] = static_cast<size_t>(rounded);
    }
    return out;
}

// The counts of a contingency table, built LOCALLY.
//
// utils/metrics.h has a `confusion_matrix`, but it lives at GLOBAL scope and
// drags core/model.h into every consumer of it; this module would rather
// depend on core/tensor.h alone. Building the table here ALSO buys the
// thing the metric cannot give: the EXPLICIT class-index space (C2). A
// confusion matrix built from the labels that happen to appear compresses
// the class indices, and the WEIGHTED kappa's answer changes when it does.
//
// `table` is row-major: table(i * num_cols + j) is the count of (i, j).
std::vector<double> build_contingency_table(const std::vector<size_t>& rows,
                                             const std::vector<size_t>& cols,
                                             size_t num_rows, size_t num_cols) {
    std::vector<double> table(num_rows * num_cols, 0.0);
    for (size_t k = 0; k < rows.size(); ++k)
        table[rows[k] * num_cols + cols[k]] += 1.0;
    return table;
}

// The row sums and the column sums of a row-major table.
void table_margins(const std::vector<double>& table, size_t num_rows, size_t num_cols,
                   std::vector<double>& row_sums, std::vector<double>& col_sums) {
    row_sums.assign(num_rows, 0.0);
    col_sums.assign(num_cols, 0.0);
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            double v = table[i * num_cols + j];
            row_sums[i] += v;
            col_sums[j] += v;
        }
    }
}

// The expected counts E_ij = r_i * c_j / N, in the CONVENTIONAL orientation
// (rows are the first factor, columns the second).
//
// C4: scikit-learn builds the TRANSPOSE of this matrix, and it does not
// matter -- every weight matrix this module supports is symmetric, so the
// two orientations give the same sums. This module uses the conventional one.
std::vector<double> expected_counts(size_t num_rows, size_t num_cols,
                                     const std::vector<double>& row_sums,
                                     const std::vector<double>& col_sums, double total) {
    std::vector<double> expected(num_rows * num_cols, 0.0);
    for (size_t i = 0; i < num_rows; ++i)
        for (size_t j = 0; j < num_cols; ++j)
            expected[i * num_cols + j] = row_sums[i] * col_sums[j] / total;
    return expected;
}

// The guard the chi-square and G-test share: a table with fewer than two
// rows or two columns, a non-finite count, a negative count, or an EMPTY ROW
// OR COLUMN (which makes an expected count zero and the statistic
// undefined -- C7).
//
// `who` distinguishes the two callers so the messages differ.
void validate_contingency_table(const Tensor& table, const char* who) {
    if (table.rows < 2 || table.cols < 2)
        throw std::invalid_argument(std::string(who) +
                                    ": the table must have at least two rows and two columns");
    for (size_t i = 0; i < table.rows; ++i)
        for (size_t j = 0; j < table.cols; ++j) {
            double v = table(i, j);
            if (!std::isfinite(v))
                throw std::invalid_argument(std::string(who) +
                                            ": the table contains a non-finite count");
            if (v < 0.0)
                throw std::invalid_argument(std::string(who) +
                                            ": the table contains a negative count");
        }
}

// ---------------------------------------------------------------------------
// numpy's PAIRWISE summation, reproduced exactly.
//
// numpy's `ndarray.sum()` is NOT a naive left-to-right loop. It accumulates
// in blocks of EIGHT with EIGHT partial sums, combines them pairwise, and
// splits the array in half above 128 elements. The oracle's kappa model sums
// with numpy, so a C++ loop that accumulates SEQUENTIALLY disagrees with it
// in the last bits -- and in the near-zero kappa regime, where
// `1 - numerator/denominator` cancels catastrophically, that last bit is
// worth 5.6e-04 of RELATIVE difference.
//
// MEASURED: with this reduction the module reproduces the oracle's kappa
// model BIT FOR BIT on 1200/1200 random cases (all three weight schemes);
// a sequential accumulation reproduces it on 732/1200 only, and the two
// differ by up to 5.6e-04 in RELATIVE terms on the cases where the kappa is
// near zero -- far outside the repo's own 1e-12 relative tolerance.
//
// This is numpy/core/src/umath/loops_arithm_fp.dispatch.c.src's
// `pairwise_sum_DOUBLE`, transcribed.
// ---------------------------------------------------------------------------
double numpy_pairwise_sum_at(const double* values, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) res += values[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (size_t j = 0; j < 8; ++j) r[j] = values[j];
        size_t i = 8;
        const size_t end = n - (n % 8);
        for (; i < end; i += 8)
            for (size_t j = 0; j < 8; ++j) r[j] += values[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += values[i];
        return res;
    }
    const size_t half = n / 2;
    const size_t split = half - (half % 8);
    return numpy_pairwise_sum_at(values, split) + numpy_pairwise_sum_at(values + split, n - split);
}

// The same reduction over a vector.
double numpy_pairwise_sum(const std::vector<double>& values) {
    return numpy_pairwise_sum_at(values.data(), values.size());
}

// ---------------------------------------------------------------------------
// The regularized incomplete gamma, the numerical engine of the general
// chi-square survival function.
//
// The series for P(a,x) and the continued fraction for Q(a,x), the
// classical pair, with `exp(-x + a*log(x) - lgamma(a))` factored out of
// both. The tolerances and the iteration cap are the ones the oracle uses.
// ---------------------------------------------------------------------------
constexpr int kGammaItMax = 1000;
constexpr double kGammaEps = 3e-16;
constexpr double kGammaFpMin = 1e-300;

// The series for the regularized LOWER incomplete gamma P(a, x).
double gamma_series(double a, double x) {
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int i = 0; i < kGammaItMax; ++i) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * kGammaEps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// The continued fraction for the regularized UPPER incomplete gamma Q(a, x).
double gamma_continued_fraction(double a, double x) {
    double b = x + 1.0 - a;
    double c = 1.0 / kGammaFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kGammaItMax; ++i) {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kGammaFpMin) d = kGammaFpMin;
        c = b + an / c;
        if (std::fabs(c) < kGammaFpMin) c = kGammaFpMin;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kGammaEps) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

// The regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
double gamma_upper(double a, double x) {
    if (x < a + 1.0) return 1.0 - gamma_series(a, x);
    return gamma_continued_fraction(a, x);
}

// ---------------------------------------------------------------------------
// The hypergeometric pmf P(X = x) for X ~ Hypergeometric(M, n, N), in LOG
// SPACE.
//
// The parameter order is scipy.stats.hypergeom.pmf(k, M, n, N): M is the
// population size, n the number of "successes" in the population, N the
// number of draws. In the 2x2 Fisher case: M is the table total, n is the
// first row's total, N is the first column's total.
//
// std::lgamma is used because the pmf has to be evaluated for values that
// would overflow a direct factorial computation.
// ---------------------------------------------------------------------------
double hypergeometric_pmf(double x, double M, double n, double N) {
    // Outside the support the pmf is 0. The oracle's guard, verbatim.
    if (x < 0.0 || x > n || x > M || (N - x) > (M - n)) return 0.0;
    double log_p = (std::lgamma(n + 1.0) - std::lgamma(x + 1.0) - std::lgamma(n - x + 1.0) +
                    std::lgamma(M - n + 1.0) - std::lgamma(N - x + 1.0) -
                    std::lgamma(M - n - N + x + 1.0) - std::lgamma(M + 1.0) +
                    std::lgamma(N + 1.0) + std::lgamma(M - N + 1.0));
    return std::exp(log_p);
}

}  // namespace

// ---------------------------------------------------------------------------
// chi_square_survival
//
// C1: the df == 1 path is the repo's EXISTING closed form, bit for bit --
// the same expression utils/survival.cpp uses for the log-rank test's
// p-value, so the two cannot drift.
//
// The general path (df != 1) is the regularized upper incomplete gamma.
// ---------------------------------------------------------------------------
double chi_square_survival(double statistic, double degrees_of_freedom) {
    if (!std::isfinite(degrees_of_freedom))
        throw std::invalid_argument("chi_square_survival: the degrees of freedom must be finite");
    if (!std::isfinite(statistic))
        throw std::invalid_argument("chi_square_survival: the statistic must be finite");
    if (degrees_of_freedom <= 0.0)
        throw std::invalid_argument("chi_square_survival: the degrees of freedom must be positive");
    if (statistic < 0.0)
        throw std::invalid_argument("chi_square_survival: the statistic must not be negative");
    // P(X >= 0) = 1 for every df. The guard is not cosmetic: the general
    // path evaluates log(x) with x = statistic / 2, so a statistic of 0 is
    // log(0) = -inf and the whole expression is NaN.
    if (statistic == 0.0) return 1.0;
    // C1: bit for bit with utils/survival.cpp.
    if (degrees_of_freedom == 1.0) return std::erfc(std::sqrt(statistic / 2.0));
    return gamma_upper(degrees_of_freedom / 2.0, statistic / 2.0);
}

// ---------------------------------------------------------------------------
// cohen_kappa
// ---------------------------------------------------------------------------
KappaResult cohen_kappa(const Tensor& labels_a, const Tensor& labels_b, size_t num_classes,
                        KappaWeights weights) {
    if (num_classes == 0)
        throw std::invalid_argument("cohen_kappa: num_classes must be positive");
    if (labels_a.rows != labels_b.rows)
        throw std::invalid_argument(
            "cohen_kappa: the two label vectors must have the same number of rows");
    std::vector<size_t> a = extract_class_indices(labels_a, num_classes, "cohen_kappa: labels_a");
    std::vector<size_t> b = extract_class_indices(labels_b, num_classes, "cohen_kappa: labels_b");

    // The table lives in the EXPLICIT class-index space (C2): an absent
    // intermediate class does NOT compress the distances.
    std::vector<double> table = build_contingency_table(a, b, num_classes, num_classes);
    std::vector<double> row_sums, col_sums;
    table_margins(table, num_classes, num_classes, row_sums, col_sums);

    double total = static_cast<double>(labels_a.rows);
    // C14: the two reductions below are numpy's PAIRWISE sums, not naive
    // left-to-right accumulations, because the oracle's model is numpy's.
    // `np.trace` reduces a DIAGONAL with the same pairwise routine.
    std::vector<double> diagonal;
    diagonal.reserve(num_classes);
    for (size_t i = 0; i < num_classes; ++i) diagonal.push_back(table[i * num_classes + i]);
    double trace = numpy_pairwise_sum(diagonal);
    std::vector<double> expected =
        expected_counts(num_classes, num_classes, row_sums, col_sums, total);

    KappaResult res;
    res.num_agreements = static_cast<size_t>(trace);
    res.num_total = labels_a.rows;
    res.observed_agreement = trace / total;
    std::vector<double> expected_diagonal;
    expected_diagonal.reserve(num_classes);
    for (size_t i = 0; i < num_classes; ++i)
        expected_diagonal.push_back(expected[i * num_classes + i]);
    res.expected_agreement = numpy_pairwise_sum(expected_diagonal) / total;

    // The weighted form, for all three weight matrices:
    //     kappa_w = 1 - (sum_ij w_ij C_ij) / (sum_ij w_ij E_ij)
    // For the UNWEIGHTED matrix (1 off the diagonal) this is algebraically
    // (p_o - p_e) / (1 - p_e), and the oracle computes it this way too.
    //
    // C14: the PRODUCTS are rounded to float64 first (exactly as numpy's
    // `w * table` materialises them) and the terms are then summed with
    // numpy's pairwise reduction. Both details are load-bearing: the
    // numerator's terms are large and the denominator's are small, and
    // `1 - numerator/denominator` cancels catastrophically when the kappa
    // is near zero.
    std::vector<double> numerator_terms;
    std::vector<double> denominator_terms;
    numerator_terms.reserve(num_classes * num_classes);
    denominator_terms.reserve(num_classes * num_classes);
    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            double diff = static_cast<double>(i) - static_cast<double>(j);
            double w;
            switch (weights) {
                case KappaWeights::Unweighted:
                    w = (i == j) ? 0.0 : 1.0;
                    break;
                case KappaWeights::Linear:
                    w = std::fabs(diff);
                    break;
                case KappaWeights::Quadratic:
                default:
                    w = diff * diff;
                    break;
            }
            numerator_terms.push_back(w * table[i * num_classes + j]);
            denominator_terms.push_back(w * expected[i * num_classes + j]);
        }
    }
    double numerator = numpy_pairwise_sum(numerator_terms);
    double denominator = numpy_pairwise_sum(denominator_terms);

    // C3: the undefined case is NaN, not 0. It happens exactly when the
    // denominator vanishes -- which is when the two raters share only one
    // label, so the chance agreement is 1.
    res.kappa = (denominator == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                     : 1.0 - numerator / denominator;
    return res;
}

// ---------------------------------------------------------------------------
// matthews_correlation
// ---------------------------------------------------------------------------
MatthewsResult matthews_correlation(const Tensor& truth, const Tensor& prediction,
                                    size_t num_classes) {
    if (num_classes == 0)
        throw std::invalid_argument("matthews_correlation: num_classes must be positive");
    if (truth.rows != prediction.rows)
        throw std::invalid_argument(
            "matthews_correlation: the truth and prediction vectors must have the same number of rows");
    std::vector<size_t> t =
        extract_class_indices(truth, num_classes, "matthews_correlation: truth");
    std::vector<size_t> p =
        extract_class_indices(prediction, num_classes, "matthews_correlation: prediction");

    std::vector<double> table = build_contingency_table(t, p, num_classes, num_classes);
    std::vector<double> t_sum, p_sum;
    table_margins(table, num_classes, num_classes, t_sum, p_sum);

    double n = static_cast<double>(truth.rows);
    double correct = 0.0;
    for (size_t k = 0; k < num_classes; ++k) correct += table[k * num_classes + k];

    double dot_tp = 0.0, dot_pp = 0.0, dot_tt = 0.0;
    for (size_t k = 0; k < num_classes; ++k) {
        dot_tp += t_sum[k] * p_sum[k];
        dot_pp += p_sum[k] * p_sum[k];
        dot_tt += t_sum[k] * t_sum[k];
    }

    double cov_ytyp = correct * n - dot_tp;
    double cov_ypyp = n * n - dot_pp;
    double cov_ytyt = n * n - dot_tt;
    double denom = cov_ytyt * cov_ypyp;

    MatthewsResult res;
    res.num_correct = static_cast<size_t>(correct);
    res.num_total = truth.rows;
    // C5: the DEGENERATE case is 0.0, NOT NaN. This is the OPPOSITE of the
    // kappa case (C3), and the two are not interchangeable.
    res.mcc = (denom == 0.0) ? 0.0 : cov_ytyp / std::sqrt(denom);
    return res;
}

// ---------------------------------------------------------------------------
// chi_square_test_of_independence
// ---------------------------------------------------------------------------
ChiSquareTestResult chi_square_test_of_independence(const Tensor& table, bool correct) {
    validate_contingency_table(table, "chi_square_test_of_independence");
    const size_t num_rows = table.rows;
    const size_t num_cols = table.cols;

    std::vector<double> observed(num_rows * num_cols, 0.0);
    for (size_t i = 0; i < num_rows; ++i)
        for (size_t j = 0; j < num_cols; ++j) observed[i * num_cols + j] = table(i, j);

    std::vector<double> row_sums, col_sums;
    table_margins(observed, num_rows, num_cols, row_sums, col_sums);
    double total = 0.0;
    for (size_t i = 0; i < num_rows; ++i) total += row_sums[i];

    // C7: an empty row or column makes an expected count zero and the
    // statistic undefined. scipy raises, R returns NaN, and this module
    // throws -- the repo's established convention for invalid input.
    for (size_t i = 0; i < num_rows; ++i)
        if (row_sums[i] == 0.0)
            throw std::invalid_argument(
                "chi_square_test_of_independence: the table has an empty row or column");
    for (size_t j = 0; j < num_cols; ++j)
        if (col_sums[j] == 0.0)
            throw std::invalid_argument(
                "chi_square_test_of_independence: the table has an empty row or column");

    std::vector<double> expected =
        expected_counts(num_rows, num_cols, row_sums, col_sums, total);

    // C6: the continuity correction is applied ONLY to a 2x2 table, and the
    // flag is honoured ONLY there.
    const bool apply_correction = correct && num_rows == 2 && num_cols == 2;

    double statistic = 0.0;
    double min_expected = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            double e = expected[i * num_cols + j];
            double o = observed[i * num_cols + j];
            min_expected = std::min(min_expected, e);
            double adjusted = o;
            if (apply_correction) {
                // C6-bis: the CLAMP. The correction magnitude is min(0.5, the
                // cell's OWN deviation), and it is applied in the direction of
                // the deviation -- exactly scipy's and R's route, which is
                // algebraically the clamped form
                //     ( |O_ij - E_ij| - min(0.5, |O_ij - E_ij|) )^2 / E_ij
                // but reproduces their float64 arithmetic bit for bit.
                double diff = e - o;
                double magnitude = std::min(0.5, std::fabs(diff));
                double direction = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
                adjusted = o + magnitude * direction;
            }
            double delta = adjusted - e;
            statistic += delta * delta / e;
        }
    }

    ChiSquareTestResult res;
    res.statistic = statistic;
    // C6-ter: the contingency degrees of freedom, NOT scipy's
    // power_divergence "cells - 1".
    res.degrees_of_freedom = (num_rows - 1) * (num_cols - 1);
    res.p_value = chi_square_survival(res.statistic, static_cast<double>(res.degrees_of_freedom));
    res.min_expected_count = min_expected;
    return res;
}

// ---------------------------------------------------------------------------
// g_test_of_independence
// ---------------------------------------------------------------------------
GTestResult g_test_of_independence(const Tensor& table) {
    validate_contingency_table(table, "g_test_of_independence");
    const size_t num_rows = table.rows;
    const size_t num_cols = table.cols;

    std::vector<double> observed(num_rows * num_cols, 0.0);
    for (size_t i = 0; i < num_rows; ++i)
        for (size_t j = 0; j < num_cols; ++j) observed[i * num_cols + j] = table(i, j);

    std::vector<double> row_sums, col_sums;
    table_margins(observed, num_rows, num_cols, row_sums, col_sums);
    double total = 0.0;
    for (size_t i = 0; i < num_rows; ++i) total += row_sums[i];

    for (size_t i = 0; i < num_rows; ++i)
        if (row_sums[i] == 0.0)
            throw std::invalid_argument(
                "g_test_of_independence: the table has an empty row or column");
    for (size_t j = 0; j < num_cols; ++j)
        if (col_sums[j] == 0.0)
            throw std::invalid_argument(
                "g_test_of_independence: the table has an empty row or column");

    std::vector<double> expected =
        expected_counts(num_rows, num_cols, row_sums, col_sums, total);

    double log_likelihood = 0.0;
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            double o = observed[i * num_cols + j];
            // C8: a ZERO observed cell contributes EXACTLY 0 and is SKIPPED.
            // The naive expression 0 * log(0 / E) is 0 * (-inf) = NaN and
            // destroys the whole statistic.
            if (o <= 0.0) continue;
            log_likelihood += o * std::log(o / expected[i * num_cols + j]);
        }
    }

    GTestResult res;
    res.statistic = 2.0 * log_likelihood;
    res.degrees_of_freedom = (num_rows - 1) * (num_cols - 1);
    res.p_value = chi_square_survival(res.statistic, static_cast<double>(res.degrees_of_freedom));
    return res;
}

// ---------------------------------------------------------------------------
// mcnemar_test
// ---------------------------------------------------------------------------
McNemarResult mcnemar_test(const Tensor& a, const Tensor& b, bool correct) {
    if (a.rows != b.rows)
        throw std::invalid_argument(
            "mcnemar_test: the two outcome vectors must have the same number of rows");
    if (a.rows == 0)
        throw std::invalid_argument("mcnemar_test: the outcome vectors must have at least one row");
    if (a.cols != 1)
        throw std::invalid_argument("mcnemar_test: the first outcome vector must be a single column");
    if (b.cols != 1)
        throw std::invalid_argument("mcnemar_test: the second outcome vector must be a single column");

    size_t n01 = 0;
    size_t n10 = 0;
    for (size_t i = 0; i < a.rows; ++i) {
        double x = a(i, 0);
        double y = b(i, 0);
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::invalid_argument("mcnemar_test: the outcomes contain a non-finite value");
        // The outcomes are BINARY: exactly 0 or exactly 1. A tolerance here
        // would silently accept 0.5 as "half an outcome", which is not a
        // thing a paired binary dataset can express.
        if (x != 0.0 && x != 1.0)
            throw std::invalid_argument("mcnemar_test: the first outcome is neither 0 nor 1");
        if (y != 0.0 && y != 1.0)
            throw std::invalid_argument("mcnemar_test: the second outcome is neither 0 nor 1");
        if (x == 0.0 && y == 1.0) ++n01;
        if (x == 1.0 && y == 0.0) ++n10;
    }

    McNemarResult res;
    res.n01 = n01;
    res.n10 = n10;
    res.n_discordant = n01 + n10;

    if (res.n_discordant == 0) {
        // C10: with no discordant pairs the statistic is 0/0, which is NaN,
        // not 0. R reports NaN for both the statistic and the p-value.
        res.statistic = std::numeric_limits<double>::quiet_NaN();
        res.p_value = std::numeric_limits<double>::quiet_NaN();
        // A vacuous test is not evidence of anything, so it does not reject.
        res.exact_p_value = 1.0;
        return res;
    }

    double diff = static_cast<double>(n01) - static_cast<double>(n10);
    double n_disc = static_cast<double>(res.n_discordant);
    if (correct && n01 != n10) {
        // C9-ter: the ABSOLUTE value. R writes abs(x - t(x)) - 1, so the
        // numerator is |n01 - n10| - 1 and NOT n01 - n10 - 1; the two differ
        // whenever n01 < n10.
        double numerator = std::fabs(diff) - 1.0;
        res.statistic = numerator * numerator / n_disc;
    } else {
        // C9-bis/C9-quater: the correction is SKIPPED when n01 == n10, so
        // the statistic is the plain squared difference -- 0 there, NOT
        // 1 / (2 * n01).
        res.statistic = diff * diff / n_disc;
    }
    res.p_value = chi_square_survival(res.statistic, 1.0);
    // C11: the repo's EXISTING binomial tail, composed rather than
    // reimplemented.
    res.exact_p_value = ::binom_pvalue_half(static_cast<int>(std::min(n01, n10)),
                                             static_cast<int>(std::max(n01, n10)));
    return res;
}

// ---------------------------------------------------------------------------
// fisher_exact_test
// ---------------------------------------------------------------------------
FisherExactResult fisher_exact_test(const Tensor& table) {
    if (table.rows != 2 || table.cols != 2)
        throw std::invalid_argument("fisher_exact_test: the table must be 2x2");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double v = table(i, j);
            if (!std::isfinite(v))
                throw std::invalid_argument(
                    "fisher_exact_test: the table contains a non-finite count");
            if (v < 0.0)
                throw std::invalid_argument("fisher_exact_test: the table contains a negative count");
            if (std::fabs(v - std::round(v)) > 1e-12)
                throw std::invalid_argument(
                    "fisher_exact_test: the table contains a count that is not a whole number");
        }
    }

    double a = std::round(table(0, 0));
    double b = std::round(table(0, 1));
    double c = std::round(table(1, 0));
    double d = std::round(table(1, 1));

    double row1 = a + b;
    double row2 = c + d;
    double col1 = a + c;
    double col2 = b + d;
    double total = row1 + row2;

    FisherExactResult res;
    // An empty row or column: p_value 1.0 and an odds ratio of NaN, which is
    // what scipy returns.
    if (row1 == 0.0 || row2 == 0.0 || col1 == 0.0 || col2 == 0.0) {
        res.p_value = 1.0;
        res.odds_ratio = std::numeric_limits<double>::quiet_NaN();
        return res;
    }

    // C13: the UNCONDITIONAL cross-product odds ratio. R's fisher.test
    // reports the CONDITIONAL MLE, which is a different estimator.
    if (c > 0.0 && b > 0.0)
        res.odds_ratio = a * d / (c * b);
    else
        res.odds_ratio = std::numeric_limits<double>::infinity();

    // C12: the PROBABILITY ORDERING rule -- the sum of the pmf over every
    // x in the support whose pmf is at most the observed table's pmf.
    //
    // C12-bis: the 1 + 1e-14 RELATIVE TOLERANCE is load-bearing. Two tables
    // can share the observed pmf exactly in exact arithmetic while their
    // float64 evaluations differ in the last bits; the tolerance includes
    // the second one and a bare comparison EXCLUDES it, which HALVES the
    // p-value on the tie fixture.
    double pexact = hypergeometric_pmf(a, total, row1, col1);
    const double gamma = 1.0 + 1e-14;
    double lo = std::max(0.0, col1 - (total - row1));
    double hi = std::min(row1, col1);
    double sum = 0.0;
    for (double x = lo; x <= hi; x += 1.0) {
        double p = hypergeometric_pmf(x, total, row1, col1);
        if (p <= pexact * gamma) sum += p;
    }
    res.p_value = std::min(sum, 1.0);
    return res;
}

}  // namespace utils
}  // namespace nn
