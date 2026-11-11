#include "isotonic_regression.h"

#include "metrics.h"   // expected_calibration_error, brier_score_loss

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Isotonic regression -- Best & Chakravarti's active-set framework.
//
// The algorithm, and the four semantics that are easy to get wrong, are
// spelled out in the header. This file is the implementation of that
// specification; the specification itself is tests/isotonic_oracle.py.
//
// The fit is:
//   1. collapse tied x to a WEIGHTED MEAN of y, summing the weights;
//   2. run the Pool-Adjacent-Violators algorithm on the collapsed points;
//   3. trim the interior knots whose fitted value equals both neighbours';
//   4. evaluate by LINEAR INTERPOLATION through the retained knots.
// ============================================================================

namespace {

// ---------------------------------------------------------------------------
// A PAVA block: the accumulated weighted sum of y, the accumulated weight,
// and the number of POINTS it covers.
//
// The block's value is sum_wy / sum_w, recomputed from the ACCUMULATED
// SUMS. It is never the mean of two block means -- that is wrong whenever
// the blocks differ in weight or in size.
// ---------------------------------------------------------------------------
struct Block {
    double sum_wy = 0.0;
    double sum_w = 0.0;
    size_t count = 0;   // the number of POINTS in the block
    size_t first = 0;   // the index of the first point in the block
};

double block_value(const Block& b) { return b.sum_wy / b.sum_w; }

// A point on the collapsed grid: one distinct x, with the weighted mean of
// the y's at that x and the summed weight there.
struct CollapsedPoint {
    double x = 0.0;
    double y = 0.0;      // the WEIGHTED MEAN of y at this x
    double weight = 0.0;  // the SUMMED weight at this x
};

void require_column(const Tensor& t, size_t rows, const char* name,
                    const char* fn) {
    if (t.cols != 1) {
        throw std::invalid_argument(std::string(fn) + " requires " + name +
                                    " to be a (N,1) column");
    }
    if (t.rows != rows) {
        throw std::invalid_argument(std::string(fn) + " requires " + name +
                                    " to have the same number of rows");
    }
}

void require_finite_column(const Tensor& t, const char* name, const char* fn) {
    for (size_t i = 0; i < t.rows; ++i) {
        if (!std::isfinite(t(i, 0))) {
            throw std::invalid_argument(std::string(fn) + " requires finite " +
                                        name + " values");
        }
    }
}

}  // namespace

namespace nn {
namespace utils {

IsotonicFit fit_isotonic(const Tensor& x, const Tensor& y, const Tensor& weights,
                          IsotonicDirection direction) {
    const char* fn = "fit_isotonic";

    if (x.rows == 0 || y.rows == 0) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires at least one row");
    }
    if (x.rows != y.rows) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires x and y to have the same number of rows");
    }
    require_column(x, x.rows, "x", fn);
    require_column(y, y.rows, "y", fn);
    require_finite_column(x, "x", fn);
    require_finite_column(y, "y", fn);

    // The weights are optional: an empty tensor means uniform weights.
    const bool has_weights = weights.rows > 0;
    if (has_weights) {
        if (weights.cols != 1) {
            throw std::invalid_argument(std::string(fn) +
                                        " requires weights to be a (N,1) column");
        }
        if (weights.rows != x.rows) {
            throw std::invalid_argument(
                std::string(fn) +
                " requires weights to have the same number of rows as x");
        }
        require_finite_column(weights, "weights", fn);
        for (size_t i = 0; i < weights.rows; ++i) {
            if (weights(i, 0) < 0.0) {
                throw std::invalid_argument(std::string(fn) +
                                            " requires non-negative weights");
            }
        }
    }

    // ---- 1. Drop the zero-weight points, and collect the rest.
    //
    // A zero-weight point must not contribute to the tie-collapse, must
    // not shift the fitted range, and must not appear as a knot.
    std::vector<double> xs, ys, ws;
    xs.reserve(x.rows);
    ys.reserve(y.rows);
    ws.reserve(x.rows);
    size_t positive = 0;
    for (size_t i = 0; i < x.rows; ++i) {
        const double w = has_weights ? weights(i, 0) : 1.0;
        if (w <= 0.0) continue;   // zero-weight points are DROPPED
        ++positive;
        xs.push_back(x(i, 0));
        ys.push_back(y(i, 0));
        ws.push_back(w);
    }
    if (positive == 0) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires at least one positive weight");
    }

    // ---- 2. Sort by x.
    //
    // std::stable_sort keeps the original order within a tie, which keeps
    // the accumulation order -- and therefore the floating-point sum --
    // deterministic for a given input.
    std::vector<size_t> order(xs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return xs[a] < xs[b]; });

    // ---- 3. Collapse the tied x to a WEIGHTED MEAN of y, summing the
    //         weights.
    std::vector<CollapsedPoint> grid;
    grid.reserve(xs.size());
    for (size_t k = 0; k < order.size();) {
        const double xk = xs[order[k]];
        double sum_wy = 0.0;
        double sum_w = 0.0;
        size_t j = k;
        while (j < order.size() && xs[order[j]] == xk) {
            sum_wy += ws[order[j]] * ys[order[j]];
            sum_w += ws[order[j]];
            ++j;
        }
        CollapsedPoint p;
        p.x = xk;
        p.y = sum_wy / sum_w;
        p.weight = sum_w;
        grid.push_back(p);
        k = j;
    }

    // ---- 4. The Pool-Adjacent-Violators algorithm.
    //
    // A stack of blocks; whenever the second-to-last block's value
    // VIOLATES the monotonicity against the last block's, the two are
    // POOLED. The pooled block's value is the weighted mean of ALL the
    // points it contains, recomputed from the accumulated sums.
    const bool increasing = (direction == IsotonicDirection::Increasing);
    std::vector<Block> stack;
    stack.reserve(grid.size());
    std::vector<double> fitted(grid.size(), 0.0);

    for (size_t g = 0; g < grid.size(); ++g) {
        Block b;
        b.sum_wy = grid[g].weight * grid[g].y;
        b.sum_w = grid[g].weight;
        b.count = 1;
        b.first = g;
        stack.push_back(b);

        while (stack.size() >= 2) {
            const Block& prev = stack[stack.size() - 2];
            const Block& last = stack[stack.size() - 1];
            const double vp = block_value(prev);
            const double vl = block_value(last);
            const bool violation = increasing ? (vp > vl) : (vp < vl);
            if (!violation) break;

            // Pool the two ADJACENT violators into the earlier block.
            Block& a = stack[stack.size() - 2];
            a.sum_wy += stack.back().sum_wy;
            a.sum_w += stack.back().sum_w;
            a.count += stack.back().count;
            stack.pop_back();
        }
    }

    // Expand the blocks back over the grid.
    for (const Block& b : stack) {
        const double v = block_value(b);
        for (size_t g = b.first; g < b.first + b.count; ++g) fitted[g] = v;
    }

    // ---- 5. Trim the redundant knots.
    //
    // An interior knot whose fitted value equals BOTH neighbours' is
    // dropped. The removed point lies on the segment between its
    // neighbours, so the interpolated function is unchanged -- but the knot
    // LIST is part of the public output, so the trimming is observable.
    std::vector<double> kx, ky;
    kx.reserve(grid.size());
    ky.reserve(grid.size());
    for (size_t g = 0; g < grid.size(); ++g) {
        const bool interior = (g > 0 && g + 1 < grid.size());
        const bool redundant =
            interior && fitted[g] == fitted[g - 1] && fitted[g] == fitted[g + 1];
        if (redundant) continue;
        kx.push_back(grid[g].x);
        ky.push_back(fitted[g]);
    }

    IsotonicFit fit;
    fit.x_knots = std::move(kx);
    fit.y_knots = std::move(ky);
    fit.direction = direction;
    return fit;
}

Tensor isotonic_predict(const IsotonicFit& fit, const Tensor& x,
                         IsotonicOutOfBounds oob) {
    const char* fn = "isotonic_predict";

    if (fit.x_knots.empty() || fit.y_knots.empty()) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires a fitted map with at least one knot");
    }
    if (x.cols != 1) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires x to be a (N,1) column");
    }
    require_finite_column(x, "x", fn);

    const size_t n = fit.x_knots.size();
    const double x_min = fit.x_knots.front();
    const double x_max = fit.x_knots.back();

    Tensor out(x.rows, 1);
    for (size_t i = 0; i < x.rows; ++i) {
        const double q = x(i, 0);

        if (q < x_min || q > x_max) {
            switch (oob) {
                case IsotonicOutOfBounds::Clip:
                    // The nearer END VALUE -- not the mean of the ends.
                    out(i, 0) = (q < x_min) ? fit.y_knots.front() : fit.y_knots.back();
                    continue;
                case IsotonicOutOfBounds::Nan:
                    out(i, 0) = std::numeric_limits<double>::quiet_NaN();
                    continue;
                case IsotonicOutOfBounds::Raise:
                    throw std::out_of_range(std::string(fn) +
                                            ": score is outside the fitted range");
            }
        }

        if (n == 1) {
            // A single knot: the map is constant.
            out(i, 0) = fit.y_knots[0];
            continue;
        }

        // The LINEAR INTERPOLATION through the retained knots. The step
        // function is NOT what is returned here -- the distinction is
        // observable and the suite pins it.
        size_t hi = 1;
        while (hi < n && fit.x_knots[hi] < q) ++hi;
        if (hi >= n) hi = n - 1;
        const size_t lo = hi - 1;
        const double x0 = fit.x_knots[lo];
        const double x1 = fit.x_knots[hi];
        const double y0 = fit.y_knots[lo];
        const double y1 = fit.y_knots[hi];
        if (x1 == x0) {
            out(i, 0) = y0;
        } else {
            const double t = (q - x0) / (x1 - x0);
            out(i, 0) = y0 + t * (y1 - y0);
        }
    }
    return out;
}

IsotonicFit fit_isotonic_calibration(const Tensor& scores, const Tensor& labels,
                                      const Tensor& weights) {
    const char* fn = "fit_isotonic_calibration";

    if (labels.cols != 1) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires labels to be a (N,1) column");
    }
    for (size_t i = 0; i < labels.rows; ++i) {
        const double v = labels(i, 0);
        if (!(v == 0.0 || v == 1.0)) {
            throw std::invalid_argument(
                std::string(fn) + " requires binary labels in {0, 1}");
        }
    }

    // The calibration map is non-decreasing by definition, and the targets
    // are 0/1, so every fitted value is a convex combination of them and
    // lies in [0, 1] BY CONSTRUCTION.
    return fit_isotonic(scores, labels, weights, IsotonicDirection::Increasing);
}

Tensor isotonic_calibrate(const IsotonicFit& fit, const Tensor& scores,
                           IsotonicOutOfBounds oob) {
    return isotonic_predict(fit, scores, oob);
}

IsotonicCalibrationReport evaluate_isotonic_calibration(
    const Tensor& calibration_scores, const Tensor& calibration_labels,
    const Tensor& eval_scores, const Tensor& eval_labels, size_t n_bins,
    IsotonicOutOfBounds oob) {
    const char* fn = "evaluate_isotonic_calibration";

    if (eval_scores.rows != eval_labels.rows) {
        throw std::invalid_argument(
            std::string(fn) +
            " requires eval_scores and eval_labels to have the same number of rows");
    }
    if (eval_scores.rows == 0) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires at least one evaluation row");
    }
    if (n_bins == 0) {
        throw std::invalid_argument(std::string(fn) + " requires n_bins >= 1");
    }

    // The map is fitted on the CALIBRATION set.
    const IsotonicFit fit = fit_isotonic_calibration(calibration_scores,
                                                     calibration_labels);
    // The report is measured on the EVALUATION set. Keeping the two apart
    // is the whole point: a map fitted by PAVA interpolates its fitting
    // set, so an in-sample report measures nothing (see the header).
    const Tensor calibrated = isotonic_calibrate(fit, eval_scores, oob);

    IsotonicCalibrationReport rep;
    rep.n_knots = fit.n_knots();
    rep.n_samples = eval_scores.rows;

    // The repo's OWN metrics -- not a re-implementation of ECE or Brier.
    rep.ece_before = expected_calibration_error(eval_labels, eval_scores, n_bins,
                                                 1.0);
    rep.ece_after = expected_calibration_error(eval_labels, calibrated, n_bins,
                                                1.0);
    rep.brier_before = brier_score_loss(eval_labels, eval_scores);
    rep.brier_after = brier_score_loss(eval_labels, calibrated);
    return rep;
}

}  // namespace utils
}  // namespace nn
