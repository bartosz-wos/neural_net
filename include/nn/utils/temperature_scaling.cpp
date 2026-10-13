#include "temperature_scaling.h"

#include "losses.h"   // log_softmax — public, already max-subtracted

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// ============================================================================
// Temperature scaling — Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern
// Neural Networks", arXiv:1706.04599 (ICML 2017), Sec. 4.2.
//
// Gradient form and its derivation are in the header; the formula here is
//
//   dL/dT = (1 / (n T^2)) * sum_i [ z_{i,y_i} - sum_j p_j(i) * z_j(i) ]
//
// (z_{i,y_i} MINUS sum_j p_j z_j — the bracket order is load-bearing).
// ============================================================================

namespace {

// Golden-section ratio. The constant is named so the mutation harness can
// assert the baseline file is pristine by grepping for it.
const double kGoldenRatio = 0.6180339887498949;

void validate_temperature(double T) {
    if (!(T > 0.0)) {
        throw std::invalid_argument("temperature must be > 0, got " + std::to_string(T));
    }
}

void validate_logits(const Tensor& logits) {
    if (logits.rows == 0 || logits.cols == 0) {
        throw std::invalid_argument("logits must have at least one row and one column");
    }
}

// Class index of row i, validated the same way metrics.cpp validates labels:
// must be finite, integral, and in [0, cols).
size_t label_index(const Tensor& labels, size_t row, size_t num_classes) {
    const double raw = labels(row, 0);
    if (!std::isfinite(raw)) {
        throw std::invalid_argument("labels contains a non-finite class index at row " +
                                    std::to_string(row));
    }
    const double rounded = std::round(raw);
    if (std::fabs(raw - rounded) > 1e-12) {
        throw std::invalid_argument("labels contains a non-integer class index at row " +
                                    std::to_string(row));
    }
    if (rounded < 0.0 || rounded >= static_cast<double>(num_classes)) {
        throw std::invalid_argument("labels class index " + std::to_string(rounded) +
                                    " out of range for " + std::to_string(num_classes) +
                                    " classes at row " + std::to_string(row));
    }
    return static_cast<size_t>(rounded);
}

void validate_pair(const Tensor& logits, const Tensor& labels) {
    validate_logits(logits);
    if (labels.rows == 0 || labels.cols != 1) {
        throw std::invalid_argument("labels must be a non-empty (N, 1) column of class indices");
    }
    if (labels.rows != logits.rows) {
        throw std::invalid_argument("labels and logits must have the same number of rows");
    }
}

// Softmax(logits / T), computed through the SHIPPED log_softmax rather than a
// second copy of the max-subtraction: a duplicated numerically-stable softmax
// is a drift risk, and log_softmax is public for exactly this reason.
Tensor scaled_probs(const Tensor& logits, double T) {
    Tensor scaled(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i)
        for (size_t j = 0; j < logits.cols; ++j)
            scaled(i, j) = logits(i, j) / T;

    const Tensor log_probs = log_softmax(scaled);
    Tensor probs(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i)
        for (size_t j = 0; j < logits.cols; ++j)
            probs(i, j) = std::exp(log_probs(i, j));
    return probs;
}

}  // namespace

Tensor temperature_scale(const Tensor& logits, double T) {
    validate_logits(logits);
    validate_temperature(T);
    return scaled_probs(logits, T);
}

double temperature_nll(const Tensor& logits, const Tensor& labels, double T) {
    validate_pair(logits, labels);
    validate_temperature(T);

    const Tensor probs = scaled_probs(logits, T);
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        const size_t y = label_index(labels, i, logits.cols);
        // Sum rather than log: log(0) is -inf, and underflowing p to exactly 0
        // at a very small T is expected behaviour, not an error. -log(eps) with
        // eps = DBL_MIN is the largest finite loss a double can represent here,
        // and it keeps the objective finite so the line search stays well-behaved
        // at the T -> 0 limit.
        total -= std::log(std::max(probs(i, y), std::numeric_limits<double>::min()));
    }
    // MEAN over rows, per the header's note (5). A sum would be a different
    // objective whose optimum depends on the batch size.
    return total / static_cast<double>(logits.rows);
}

double temperature_nll_grad(const Tensor& logits, const Tensor& labels, double T) {
    validate_pair(logits, labels);
    validate_temperature(T);

    const Tensor probs = scaled_probs(logits, T);
    const double inv_T2 = 1.0 / (T * T);
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        const size_t y = label_index(labels, i, logits.cols);
        // p . z : the softmax-weighted mean of the logit vector.
        double pz = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) pz += probs(i, j) * logits(i, j);
        total += (logits(i, y) - pz) * inv_T2;
    }
    return total / static_cast<double>(logits.rows);
}

double fit_temperature(const Tensor& logits, const Tensor& labels,
                       double t_min, double t_max, int max_iter) {
    validate_pair(logits, labels);
    if (!(t_min > 0.0)) {
        throw std::invalid_argument("t_min must be > 0, got " + std::to_string(t_min));
    }
    if (!(t_max > t_min)) {
        throw std::invalid_argument("t_max must exceed t_min");
    }
    if (max_iter < 0) {
        throw std::invalid_argument("max_iter must be non-negative");
    }

    // Search in log T so the bracket stays two-sided over the whole positive
    // line. A symmetric bracket in T cannot represent both T* = 0.03 and
    // T* = 11.7, and both occur on ordinary fixtures.
    //
    // The objective is UNIMODAL but NOT CONVEX in T, in log T, or in 1/T — the
    // paper calls it convex (Sec. 4.2) and that was measured false on 3 of 5
    // fixtures. A line search needs only unimodality, so golden-section is the
    // correct tool; a Newton/gradient step would have no convexity guarantee to
    // stand on. Do not "optimise" this into a gradient descent.
    double lo = std::log(t_min);
    double hi = std::log(t_max);

    // Evaluate the BRACKET ENDPOINTS FIRST, and only let a strictly better
    // interior point displace them. This is not cosmetic: on data whose NLL is
    // monotone toward a boundary — a perfectly separable validation set, where
    // the optimum is the T -> 0 limit and NLL reaches exactly 0 at t_min — a
    // search that only tracks interior points never evaluates t_min itself and
    // returns a strictly worse temperature. It was caught by
    // test_fit_boundary_cases asserting the clamp, and the golden-section
    // midpoint returned 5.8e-2 where t_min = 1e-2 was the true minimiser.
    // Ties resolve to the endpoint because in the degenerate case the endpoint
    // IS the minimiser; a tie is then not a coin flip, it is the answer.
    double best_T = t_min;
    double best_L = temperature_nll(logits, labels, t_min);

    const double upper_L = temperature_nll(logits, labels, t_max);
    if (upper_L < best_L) {
        best_L = upper_L;
        best_T = t_max;
    }

    double a = hi - kGoldenRatio * (hi - lo);
    double b = lo + kGoldenRatio * (hi - lo);
    double fa = temperature_nll(logits, labels, std::exp(a));
    double fb = temperature_nll(logits, labels, std::exp(b));

    for (int it = 0; it < max_iter && (hi - lo) > 1e-12; ++it) {
        if (fa < fb) {
            hi = b;
            b = a;
            fb = fa;
            a = hi - kGoldenRatio * (hi - lo);
            fa = temperature_nll(logits, labels, std::exp(a));
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + kGoldenRatio * (hi - lo);
            fb = temperature_nll(logits, labels, std::exp(b));
        }

        // Track the best point SEEN, not the bracket midpoint. On a separable
        // set the objective is monotone towards the T -> 0 limit and the
        // optimum is the boundary; returning the midpoint would return an
        // interior point that is strictly worse than the best one found.
        const double mid = std::exp((lo + hi) / 2.0);
        const double Lm = temperature_nll(logits, labels, mid);
        if (Lm < best_L) {
            best_L = Lm;
            best_T = mid;
        }
    }

    // Always respect the requested bracket.
    if (best_T < t_min) best_T = t_min;
    if (best_T > t_max) best_T = t_max;
    return best_T;
}

Tensor calibrate_logits(const Tensor& logits, const Tensor& labels,
                        double t_min, double t_max) {
    const double T = fit_temperature(logits, labels, t_min, t_max);
    return temperature_scale(logits, T);
}