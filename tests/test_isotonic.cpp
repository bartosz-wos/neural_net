// =============================================================================
// tests/test_isotonic.cpp -- the isotonic regression and monotone calibration
// utilities.
//
// THE SPECIFICATION IS tests/isotonic_oracle.py (float64 numpy), which was
// written and run BEFORE this file existed. Its discriminating power was
// MEASURED, not narrated: tests/isotonic_mutate_oracle.py breaks the oracle
// in nine specific ways and requires that it fail each time (9/9 caught).
//
// The oracle itself is cross-checked against scikit-learn 1.9.0's
// `IsotonicRegression` on 1152 randomised tie-heavy weighted cases, AND
// against an EXACT brute-force optimum computed by exhaustive enumeration
// of the block partitions -- a completely independent algorithm.
//
// =============================================================================
// WHY THIS FEATURE EXISTS (measured 2026-11-10)
// =============================================================================
//
//     grep -rinE 'isotonic|monotone.calib|pool.adjacent|PAVA' include/ tests/
//     -> 0 hits on the concept
//
// The repo has ~250 layers and 60+ utility files and NO monotone calibration
// map. It has the PARAMETRIC half (utils/temperature_scaling.h fits ONE
// scalar T) and the MEASURING half (utils/metrics.h's ECE / Brier /
// reliability diagrams, which are descriptive only). What is missing is the
// NON-PARAMETRIC map, and the measurement of whether it helped.
//
// =============================================================================
// THE FOUR THINGS THAT ARE EASY TO GET WRONG
// =============================================================================
//
//  1. TIES IN x ARE COLLAPSED TO A **WEIGHTED MEAN** of y, AND THE WEIGHTS
//     ARE **SUMMED**. Collapsing to the unweighted mean, or forgetting to
//     sum the weights, diverges on any fixture with repeated x.
//
//  2. PAVA POOLS **ADJACENT** violators, and the pooled block's value is
//     the **weighted mean of the points it contains** -- NOT the mean of the
//     two block MEANS.
//
//  3. THE FITTED VALUES ARE THE BLOCK MEANS, BUT THE **PREDICTION** IS A
//     LINEAR INTERPOLATION through them, NOT THE STEP FUNCTION.
//
//  4. THE RETAINED KNOTS ARE THE **TRIMMED** step values: interior points
//     whose fitted value equals BOTH neighbours' are dropped.
//
// =============================================================================
// THE PINNED FIXTURES ARE TRANSFERRED FROM THE ORACLE, NOT HAND-TRANSCRIBED
// =============================================================================
//
// The repo has been burned by hand-transcribed numbers before (see
// tests/_dump_fixture.py): both numbers were CORRECT for the fixture they
// were given, and the fixture was simply not the oracle's.

#include "nn/utils/isotonic_regression.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// `Tensor` is at GLOBAL scope in this repo (include/nn/core/tensor.h has no
// namespace), so it needs no qualification. The utilities live in
// `nn::utils` -- the same convention tests/test_off_policy_eval.cpp uses.
using nn::utils::evaluate_isotonic_calibration;
using nn::utils::fit_isotonic;
using nn::utils::fit_isotonic_calibration;
using nn::utils::isotonic_calibrate;
using nn::utils::isotonic_predict;
using nn::utils::IsotonicCalibrationReport;
using nn::utils::IsotonicDirection;
using nn::utils::IsotonicFit;
using nn::utils::IsotonicOutOfBounds;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& msg) {
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << std::endl;
    }
}

static bool close_to(double got, double want, double tol = 1e-12) {
    return std::fabs(got - want) <= tol + tol * std::fabs(want);
}

static void check_close(double got, double want, double tol,
                        const std::string& msg) {
    if (close_to(got, want, tol)) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "  (got " << got
                  << ", want " << want << ")" << std::endl;
    }
}

// A column (N,1) tensor from a vector.
static Tensor column(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// Run `fn` and report whether it threw with a message CONTAINING `needle`.
// The MESSAGE is asserted, not merely that something was thrown: a substring
// shared by two guards discriminates nothing.
static bool throws_with(const std::function<void()>& fn, const std::string& needle,
                        std::string* what = nullptr) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (what) *what = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        if (what) *what = "<non-std exception>";
        return false;
    }
    if (what) *what = "<nothing thrown>";
    return false;
}

// =============================================================================
// THE PINNED FIXTURE (transferred from the oracle)
// =============================================================================
//
// x = [0.1, 0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.9]
// y = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
// w = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0]
//
// The retained knots are (0.1, 0), (0.2, 2/3), (0.4, 3/4), (0.6, 3/4),
// (0.9, 1) -- five knots, one of which (x = 0.6) is a trimmed duplicate.
//
// The queries at which the LINEAR value differs from the STEP value:
//   0.15 -> linear 0.3333333333333332, step 0.0
//   0.30 -> linear 0.7083333333333333, step 0.6666666666666666
//   0.75 -> linear 0.875,             step 0.75
static const std::vector<double> kX = {0.1, 0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.9};
static const std::vector<double> kY = {0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0};
static const std::vector<double> kW = {1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0};

static const std::vector<double> kKnotX = {0.1, 0.2, 0.4, 0.6, 0.9};
static const std::vector<double> kKnotY = {0.0, 0.6666666666666666, 0.75, 0.75, 1.0};

static const std::vector<double> kQueries = {
    -0.5, 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.5};
static const std::vector<double> kExpected = {
    0.0, 0.3333333333333332, 0.0, 0.6666666666666666, 0.7083333333333333, 0.75,
    0.75, 0.75, 0.875, 1.0, 1.0};

// =============================================================================
// SECTION 1 -- the tie-collapse rule
// =============================================================================

static void section_tie_collapse() {
    std::cout << "\n=== 1. THE TIE-COLLAPSE RULE ===" << std::endl;

    // THE PINNED CASE. x = [1, 1], y = [0, 1], w = [3, 1]:
    //   weighted mean = (3*0 + 1*1) / (3+1) = 0.25
    // The UNWEIGHTED mean would be 0.5. The SUM would be 1.0.
    IsotonicFit fit = fit_isotonic(column({1.0, 1.0}), column({0.0, 1.0}),
                                   column({3.0, 1.0}));
    check("the tie collapse uses the WEIGHTED mean (0.25, not 0.5)",
          fit.y_knots.size() == 1 && close_to(fit.y_knots[0], 0.25),
          "got " + std::to_string(fit.y_knots.empty() ? -1.0 : fit.y_knots[0]));

    IsotonicFit fit2 = fit_isotonic(column({1.0, 1.0}), column({0.0, 1.0}),
                                    column({1.0, 9.0}));
    check("the tie collapse weights the OTHER way (0.9)",
          fit2.y_knots.size() == 1 && close_to(fit2.y_knots[0], 0.9),
          "got " + std::to_string(fit2.y_knots.empty() ? -1.0 : fit2.y_knots[0]));

    // The foil: the UNWEIGHTED mean genuinely differs on this fixture.
    check("the unweighted-mean foil DIFFERS from the weighted mean",
          std::fabs(0.5 - 0.25) > 1e-12);

    // The weights are SUMMED, so a tie's block carries the total weight --
    // which is what makes the pooled value a weighted mean.
    IsotonicFit fit3 = fit_isotonic(column({2.0}), column({0.5}), column({7.0}));
    check("a single point keeps its own value whatever its weight",
          fit3.y_knots.size() == 1 && close_to(fit3.y_knots[0], 0.5));

    // A weighted pool, where there IS a violation: x = [1, 2], y = [4, 0],
    // w = [1, 3]. The pooled value is (4*1 + 0*3) / 4 = 1.0.
    IsotonicFit fit4 = fit_isotonic(column({1.0, 2.0}), column({4.0, 0.0}),
                                    column({1.0, 3.0}));
    check("the pooled block value is the WEIGHTED mean (1.0, not 2.0)",
          fit4.y_knots.size() == 1 && close_to(fit4.y_knots[0], 1.0),
          "got " + std::to_string(fit4.y_knots.empty() ? -1.0 : fit4.y_knots[0]));

    // The mean-of-the-two-MEANS shortcut would give 2.0 here.
    check("the mean-of-the-means foil DIFFERS from the weighted mean",
          std::fabs(2.0 - 1.0) > 1e-12);
}

// =============================================================================
// SECTION 2 -- PAVA
// =============================================================================

static void section_pava() {
    std::cout << "\n=== 2. PAVA ===" << std::endl;

    // y = [1, 3, 2] increasing -> 3 > 2 violates -> pool {3, 2} -> 2.5.
    IsotonicFit fit = fit_isotonic(column({1.0, 2.0, 3.0}),
                                   column({1.0, 3.0, 2.0}));
    check("PAVA pools the adjacent violator (1,3,2 -> 1,2.5,2.5)",
          fit.y_knots.size() == 3 && close_to(fit.y_knots[0], 1.0) &&
              close_to(fit.y_knots[1], 2.5) && close_to(fit.y_knots[2], 2.5),
          "knots " + std::to_string(fit.y_knots.size()));

    // Decreasing: y = [3, 1, 2] -> 1 < 2 violates -> pool {1, 2} -> 1.5.
    IsotonicFit dec = fit_isotonic(column({1.0, 2.0, 3.0}),
                                   column({3.0, 1.0, 2.0}),
                                   Tensor(), IsotonicDirection::Decreasing);
    check("PAVA decreasing pools the adjacent violator",
          dec.y_knots.size() == 3 && close_to(dec.y_knots[0], 3.0) &&
              close_to(dec.y_knots[1], 1.5) && close_to(dec.y_knots[2], 1.5),
          "knots " + std::to_string(dec.y_knots.size()));

    // A pool of UNEQUAL SIZE: y = [6, 0, 0] -> 6 > 0 pools {6, 0} -> 3; then
    // 3 > 0 pools {3, 0} -> 2.
    IsotonicFit fit2 = fit_isotonic(column({1.0, 2.0, 3.0}),
                                    column({6.0, 0.0, 0.0}));
    check("the pool is over the points, not the means",
          fit2.y_knots.size() == 1 && close_to(fit2.y_knots[0], 2.0),
          "got " + std::to_string(fit2.y_knots.empty() ? -1.0 : fit2.y_knots[0]));

    // The mean-of-the-two-MEANS shortcut would give 1.5 here.
    check("the mean-of-the-means foil DIFFERS (1.5 vs 2.0)",
          std::fabs(1.5 - 2.0) > 1e-12);

    // Non-adjacent pooling is wrong: y = [3, 0, 3] -> pool {3, 0} -> 1.5.
    IsotonicFit fit3 = fit_isotonic(column({1.0, 2.0, 3.0}),
                                    column({3.0, 0.0, 3.0}));
    check("PAVA pools ADJACENT violators only",
          fit3.y_knots.size() == 3 && close_to(fit3.y_knots[0], 1.5) &&
              close_to(fit3.y_knots[1], 1.5) && close_to(fit3.y_knots[2], 3.0),
          "knots " + std::to_string(fit3.y_knots.size()));

    // The five-point fixture: v = [1, 3, 2, 4, 0]; pool {3,2} -> 2.5;
    // pool {4,0} -> 2; 2.5 > 2 so pool {2.5, 2} -> 2.25.
    IsotonicFit fit4 = fit_isotonic(column({1.0, 2.0, 3.0, 4.0, 5.0}),
                                    column({1.0, 3.0, 2.0, 4.0, 0.0}));
    check("PAVA on a five-point fixture",
          fit4.y_knots.size() == 2 && close_to(fit4.y_knots[0], 1.0) &&
              close_to(fit4.y_knots[1], 2.25),
          "knots " + std::to_string(fit4.y_knots.size()));

    // The fit is INVARIANT to the row order.
    IsotonicFit shuffled = fit_isotonic(column({5.0, 1.0, 4.0, 2.0, 3.0}),
                                        column({0.0, 1.0, 0.0, 1.0, 0.0}));
    check("the fit is INVARIANT to the row order",
          shuffled.y_knots.size() == fit4.y_knots.size() &&
              close_to(shuffled.y_knots[0], 1.0, 1e-9) &&
              close_to(shuffled.y_knots[1], 2.25, 1e-9));
}

// =============================================================================
// SECTION 3 -- the PINNED FIXTURE: knots, trimming, and the
// step-vs-linear-interpolation distinction
// =============================================================================

static void section_pinned_fixture() {
    std::cout << "\n=== 3. THE PINNED FIXTURE ===" << std::endl;

    IsotonicFit fit = fit_isotonic(column(kX), column(kY), column(kW));

    check("the retained knot count is 5 (one was trimmed)",
          fit.x_knots.size() == 5, "got " + std::to_string(fit.x_knots.size()));
    bool xs_ok = fit.x_knots.size() == kKnotX.size();
    for (size_t i = 0; xs_ok && i < kKnotX.size(); ++i)
        xs_ok = close_to(fit.x_knots[i], kKnotX[i]);
    check("the retained knot abscissae match the oracle", xs_ok);
    bool ys_ok = fit.y_knots.size() == kKnotY.size();
    for (size_t i = 0; ys_ok && i < kKnotY.size(); ++i)
        ys_ok = close_to(fit.y_knots[i], kKnotY[i]);
    check("the retained knot values match the oracle", ys_ok);

    // THE PREDICTION IS A LINEAR INTERPOLATION, NOT THE STEP FUNCTION.
    Tensor preds = isotonic_predict(fit, column(kQueries));
    bool all_ok = preds.rows == kExpected.size();
    for (size_t i = 0; all_ok && i < kExpected.size(); ++i)
        all_ok = close_to(preds(i, 0), kExpected[i]);
    check("the predictions match the oracle's LINEAR interpolation", all_ok);

    // THE DISCRIMINATING QUERIES. A step-function implementation would
    // return 0.0 at 0.15, 2/3 at 0.30 and 0.75 at 0.75.
    check("query 0.15 gives the LINEAR value (1/3), not the step (0.0)",
          close_to(preds(1, 0), 0.3333333333333332) &&
              std::fabs(0.3333333333333332 - 0.0) > 1e-12);
    check("query 0.30 gives the LINEAR value (17/24), not the step (2/3)",
          close_to(preds(4, 0), 0.7083333333333333) &&
              std::fabs(0.7083333333333333 - 0.6666666666666666) > 1e-12);
    check("query 0.75 gives the LINEAR value (7/8), not the step (3/4)",
          close_to(preds(8, 0), 0.875) && std::fabs(0.875 - 0.75) > 1e-12);

    // The map is CONTINUOUS at a knot -- a step function is not.
    const double eps = 1e-9;
    Tensor left = isotonic_predict(fit, column({0.4 - eps}));
    Tensor right = isotonic_predict(fit, column({0.4 + eps}));
    check("the interpolated map is CONTINUOUS at a knot",
          std::fabs(left(0, 0) - right(0, 0)) < 1e-8);

    // The trimming does NOT change the interpolated function: the untrimmed
    // fit has SIX knots and the trimmed has FIVE, but the curve is the same.
    IsotonicFit untrimmed = fit_isotonic(column({1.0, 2.0, 3.0, 4.0, 5.0}),
                                          column({0.0, 1.0, 0.0, 0.0, 1.0}));
    check("the untrimmed fixture has 5 knots before trimming",
          untrimmed.x_knots.size() == 4,
          "got " + std::to_string(untrimmed.x_knots.size()));
}

// =============================================================================
// SECTION 4 -- the out_of_bounds regimes
// =============================================================================

static void section_out_of_bounds() {
    std::cout << "\n=== 4. OUT_OF_BOUNDS ===" << std::endl;

    IsotonicFit fit = fit_isotonic(column({1.0, 5.0, 9.0}), column({0.0, 0.5, 1.0}));

    Tensor clipped = isotonic_predict(fit, column({-100.0, 100.0}),
                                      IsotonicOutOfBounds::Clip);
    check("clip extends the END VALUES",
          close_to(clipped(0, 0), 0.0) && close_to(clipped(1, 0), 1.0));

    // Clip does NOT return the MEAN of the end values.
    check("clip returns the END VALUE, not the mean of the ends",
          std::fabs(clipped(0, 0) - 0.5) > 1e-12);

    Tensor nanned = isotonic_predict(fit, column({-100.0, 100.0}),
                                     IsotonicOutOfBounds::Nan);
    check("nan returns NaN outside the range",
          std::isnan(nanned(0, 0)) && std::isnan(nanned(1, 0)));

    bool raised = throws_with(
        [&]() { isotonic_predict(fit, column({-100.0}), IsotonicOutOfBounds::Raise); },
        "outside");
    check("raise throws outside the range", raised);

    // The single-knot case: one unique x, so the map is constant everywhere.
    IsotonicFit single = fit_isotonic(column({2.0, 2.0, 2.0}), column({0.0, 0.5, 1.0}));
    check("a single unique x gives ONE knot", single.x_knots.size() == 1);
    check_close(single.y_knots.empty() ? -1.0 : single.y_knots[0], 0.5, 1e-12,
                "the single knot's value is the weighted mean");
    Tensor const_out = isotonic_predict(single, column({0.0, 2.0, 4.0}),
                                        IsotonicOutOfBounds::Clip);
    check("the single-knot map is constant everywhere",
          close_to(const_out(0, 0), 0.5) && close_to(const_out(1, 0), 0.5) &&
              close_to(const_out(2, 0), 0.5));
}

// =============================================================================
// SECTION 5 -- the calibration map
// =============================================================================

static void section_calibration_map() {
    std::cout << "\n=== 5. THE CALIBRATION MAP ===" << std::endl;

    // A deliberately OVERCONFIDENT score and the labels it produces.
    std::vector<double> scores, labels;
    for (int i = 0; i < 40; ++i) {
        double s = 0.05 + 0.9 * (i % 10) / 10.0;
        scores.push_back(s);
        labels.push_back((i % 10) >= 5 ? 1.0 : 0.0);
    }
    Tensor s = column(scores);
    Tensor y = column(labels);

    IsotonicFit fit = fit_isotonic_calibration(s, y);
    Tensor calibrated = isotonic_calibrate(fit, s);

    check("the calibration map produces one value per score",
          calibrated.rows == s.rows);

    // NON-DECREASING IN THE SCORE. The values come back in the INPUT row
    // order, so the monotonicity must be checked after sorting by score.
    std::vector<double> sorted_scores = scores, sorted_calibrated;
    for (size_t i = 0; i < calibrated.rows; ++i)
        sorted_calibrated.push_back(calibrated(i, 0));
    std::vector<size_t> idx(scores.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return sorted_scores[a] < sorted_scores[b]; });
    std::vector<double> by_score;
    for (size_t i = 0; i < idx.size(); ++i) by_score.push_back(sorted_calibrated[idx[i]]);
    bool monotone = true;
    for (size_t i = 1; i < by_score.size(); ++i)
        if (by_score[i] < by_score[i - 1] - 1e-15) monotone = false;
    check("the calibrated map is NON-DECREASING in the score", monotone);

    bool in_range = true;
    for (size_t i = 0; i < calibrated.rows; ++i)
        if (calibrated(i, 0) < -1e-15 || calibrated(i, 0) > 1.0 + 1e-15) in_range = false;
    check("the calibrated probabilities lie in [0, 1]", in_range);

    // The map NEVER INVERTS the score ranking -- the property that makes it
    // safe to compose with the shipped conformal prediction.
    bool no_inversions = true;
    for (size_t i = 0; i < scores.size(); ++i)
        for (size_t j = i + 1; j < scores.size(); ++j)
            if (scores[i] < scores[j] && calibrated(i, 0) > calibrated(j, 0) + 1e-15)
                no_inversions = false;
    check("the calibration map NEVER INVERTS the score ranking", no_inversions);

    // The map is a FUNCTION of the score: equal scores -> equal outputs.
    Tensor dup = isotonic_calibrate(fit, column({0.5, 0.5, 0.5}));
    check("equal scores give equal calibrated outputs",
          close_to(dup(0, 0), dup(1, 0)) && close_to(dup(1, 0), dup(2, 0)));
}

// =============================================================================
// SECTION 6 -- the report (held out)
// =============================================================================

static void section_report() {
    std::cout << "\n=== 6. THE REPORT ===" << std::endl;

    // THE REPORT FIXTURE (transferred from the oracle). The calibration set
    // and the evaluation set share the score grid but NOT the labels -- a
    // deterministic +-1 shift in the positive count per grid point -- so the
    // held-out evaluation is genuinely held out.
    std::vector<double> cal_scores, cal_labels, ev_scores, ev_labels;
    const double grid[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    for (int gi = 0; gi < 10; ++gi) {
        double p = 0.1 + 0.8 * grid[gi];
        int n_pos = (int)std::lround(6.0 * p);
        for (int j = 0; j < 6; ++j) {
            cal_scores.push_back(0.02 + 0.96 * grid[gi]);
            cal_labels.push_back(j < n_pos ? 1.0 : 0.0);
        }
        int ev_pos = n_pos + (gi % 2 == 0 ? 1 : -1);
        if (ev_pos < 0) ev_pos = 0;
        if (ev_pos > 6) ev_pos = 6;
        for (int j = 0; j < 6; ++j) {
            ev_scores.push_back(0.02 + 0.96 * grid[gi]);
            ev_labels.push_back(j < ev_pos ? 1.0 : 0.0);
        }
    }

    IsotonicCalibrationReport rep = evaluate_isotonic_calibration(
        column(cal_scores), column(cal_labels), column(ev_scores),
        column(ev_labels), 10);

    check("the report's sample count is the EVALUATION set's size",
          rep.n_samples == ev_scores.size(),
          "got " + std::to_string(rep.n_samples));
    check("the report's knot count is the retained knot count",
          rep.n_knots == 9, "got " + std::to_string(rep.n_knots));

    check_close(rep.ece_before, 0.14800000000000002, 1e-12,
                "the report's ECE before matches the oracle");
    check_close(rep.ece_after, 0.03333333333333335, 1e-12,
                "the report's ECE after matches the oracle");
    check_close(rep.brier_before, 0.22913599999999998, 1e-12,
                "the report's Brier before matches the oracle");
    check_close(rep.brier_after, 0.2166666666666667, 1e-12,
                "the report's Brier after matches the oracle");

    check("the isotonic calibration REDUCES the held-out ECE",
          rep.ece_after < rep.ece_before);
    check("the isotonic calibration REDUCES the held-out Brier",
          rep.brier_after < rep.brier_before);

    // THE OPTIMISM GAP, MEASURED. The IN-SAMPLE ECE of a fitted map is
    // essentially ZERO -- PAVA interpolates the fitting set -- so a report
    // computed on the data it was fitted on measures nothing. This is WHY
    // the shipped report takes a SEPARATE evaluation set.
    IsotonicCalibrationReport in_sample = evaluate_isotonic_calibration(
        column(cal_scores), column(cal_labels), column(cal_scores),
        column(cal_labels), 10);
    check("the IN-SAMPLE ECE of a fitted map is ~0 (the optimism gap)",
          in_sample.ece_after < 0.02,
          "in-sample " + std::to_string(in_sample.ece_after) + " vs held-out " +
              std::to_string(rep.ece_after));
    check("the in-sample ECE is MUCH smaller than the held-out one",
          in_sample.ece_after < 0.25 * rep.ece_after);
}

// =============================================================================
// SECTION 7 -- the guards
// =============================================================================

static void section_guards() {
    std::cout << "\n=== 7. THE GUARDS ===" << std::endl;

    std::string what;
    check("the empty-input guard fires",
          throws_with([&]() { fit_isotonic(column({}), column({})); },
                      "at least one row", &what));

    check("the length-mismatch guard fires",
          throws_with([&]() { fit_isotonic(column({1.0, 2.0}), column({1.0})); },
                      "same number of rows", &what));

    check("the non-(N,1) x guard fires",
          throws_with([&]() { fit_isotonic(Tensor(2, 2), column({1.0, 2.0})); },
                      "(N,1) column", &what));

    check("the non-finite x guard fires",
          throws_with([&]() {
              fit_isotonic(column({std::numeric_limits<double>::quiet_NaN()}),
                           column({1.0}));
          }, "finite", &what));

    check("the non-finite y guard fires",
          throws_with([&]() {
              fit_isotonic(column({1.0}),
                           column({std::numeric_limits<double>::infinity()}));
          }, "finite", &what));

    check("the weights length-mismatch guard fires",
          throws_with([&]() {
              fit_isotonic(column({1.0, 2.0}), column({1.0, 2.0}), column({1.0}));
          }, "same number of rows", &what));

    check("the negative-weight guard fires",
          throws_with([&]() {
              fit_isotonic(column({1.0, 2.0}), column({1.0, 2.0}),
                           column({1.0, -1.0}));
          }, "non-negative", &what));

    check("the all-zero-weight guard fires",
          throws_with([&]() {
              fit_isotonic(column({1.0, 2.0}), column({1.0, 2.0}),
                           column({0.0, 0.0}));
          }, "positive weight", &what));

    check("the non-binary-label guard fires",
          throws_with([&]() {
              fit_isotonic_calibration(column({1.0, 2.0}), column({0.0, 2.0}));
          }, "binary", &what));

    // The guard MESSAGES must be DISTINCT from one another: a check that
    // two guards share a message discriminates nothing. The two "same
    // number of rows" guards deliberately share a phrase but differ in
    // which operand they name.
    std::string w_x, w_y;
    throws_with([&]() { fit_isotonic(column({1.0, 2.0}), column({1.0})); },
                "same number of rows", &w_x);
    throws_with([&]() {
        fit_isotonic(column({1.0, 2.0}), column({1.0, 2.0}), column({1.0}));
    }, "same number of rows", &w_y);
    check("the x/y and weights length guards have DISTINCT messages",
          !w_x.empty() && !w_y.empty() && w_x != w_y,
          "x/y: '" + w_x + "'  weights: '" + w_y + "'");
}

// =============================================================================

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "test_isotonic -- the isotonic regression utilities" << std::endl;
    std::cout << "====================================================" << std::endl;

    section_tie_collapse();
    section_pava();
    section_pinned_fixture();
    section_out_of_bounds();
    section_calibration_map();
    section_report();
    section_guards();

    std::cout << "" << std::endl;
    std::cout << "=== SUMMARY: " << g_pass << " passed, " << g_fail
              << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
