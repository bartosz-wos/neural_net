#ifndef NN_UTILS_ISOTONIC_REGRESSION_H
#define NN_UTILS_ISOTONIC_REGRESSION_H

#include "../core/tensor.h"

#include <cstddef>
#include <vector>

// ===========================================================================
// Isotonic regression and the monotone calibration map
//
//   Best, M. J. and Chakravarti, N. -- "Active set algorithms for isotonic
//   regression; A unifying framework", Mathematical Programming 47 (1990),
//   425-439.
//
//   Chakravarti, N. -- "Isotonic Median Regression: A Linear Programming
//   Approach", Mathematics of Operations Research 14(2) (May 1989), 303-308.
//
//   Zadrozny, B. and Elkan, C. -- "Transforming Classifier Scores into
//   Accurate Multiclass Probability Estimates", KDD 2001, 204-213.
//
//   Niculescu-Mizil, A. and Caruana, R. -- "Predicting Good Probabilities
//   with Supervised Learning", ICML 2005.
//
// CITATIONS. Verified 2026-11-10 by a title search against the publishers'
// records. NO arXiv ID IS ASSERTED: all five references are pre-arXiv or
// have no arXiv record, and this repo's expansion queue has produced
// fifteen wrong arXiv identifiers -- every one a plausible-looking
// neighbour of the right paper. The reference implementation used for the
// cross-check is scikit-learn 1.9.0's `sklearn.isotonic.IsotonicRegression`.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-10)
// ---------------------------------------------------------------------------
//
//   grep -rinE 'isotonic|monotone.calib|pool.adjacent|PAVA' include/ tests/
//   -> 0 hits on the concept
//
// The repo has ~250 layers and 60+ utility files and NO monotone calibration
// map. It has the PARAMETRIC half of the same problem and the MEASURING
// half, but not the map:
//
//   * utils/temperature_scaling.h -- fits ONE scalar T by minimising NLL on
//     softmax(z / T). A one-parameter family: it cannot represent a step,
//     a plateau, or any non-logistic shape.
//   * utils/metrics.h -- ECE / ACE / TACE / classwise-ECE / Brier /
//     reliability diagrams. DESCRIPTIVE ONLY: it reports how far the
//     probabilities are from the observed frequencies and changes nothing.
//
// What was missing is the NON-PARAMETRIC map, and the measurement of
// whether it helped. `evaluate_isotonic_calibration` supplies the second,
// and it composes with the repo's OWN metrics.h rather than re-implementing
// ECE or Brier.
//
// IS IT REDUNDANT WITH THE CONFORMAL PREDICTION? NO, AND THE TWO ARE
// COMPLEMENTARY:
//
//   * Conformal prediction answers "is the true label in the SET?" and
//     guarantees MARGINAL COVERAGE for ANY score distribution. It says
//     nothing about whether a member probability is 0.9 or 0.1.
//   * Isotonic calibration answers "given this score, how likely is the
//     positive class?" and guarantees MONOTONICITY. It says nothing about
//     coverage.
//
// So the natural pipeline is: isotonic-calibrate the scores, then feed the
// calibrated probabilities into the conformal machinery. The map NEVER
// INVERTS the score ranking, which is what makes that composition valid.
//
// ---------------------------------------------------------------------------
// WHAT IT IS MODEL-FREE AND WHY THAT MATTERS
// ---------------------------------------------------------------------------
//
// Every core below is a free function over `Tensor`. The whole file is
// therefore testable with no Model, no optimizer and no network -- the same
// shape `utils/sinkhorn.h`, `utils/crf.h`, `utils/conformal.h` and
// `utils/off_policy_eval.h` were built with. Nothing here is random.
//
// ---------------------------------------------------------------------------
// THE FOUR SEMANTICS THAT ARE EASY TO GET WRONG
// ---------------------------------------------------------------------------
//
// (1) TIES IN x ARE COLLAPSED TO A *WEIGHTED MEAN* OF y, AND THE WEIGHTS
//     ARE *SUMMED*. This is the rule in scikit-learn's `_make_unique`, and
//     it is the most common way an independent implementation diverges: an
//     implementation that collapses tied x to the UNWEIGHTED mean of their
//     y's, or that forgets to sum the weights, gets a different answer on
//     any fixture with repeated x. Measured: x = [1, 1], y = [0, 1],
//     w = [3, 1] gives 0.25 -- not 0.5 and not 1.0.
//
// (2) PAVA POOLS *ADJACENT* VIOLATORS, AND THE POOLED BLOCK'S VALUE IS THE
//     WEIGHTED MEAN OF THE POINTS IT CONTAINS -- not the mean of the two
//     block MEANS, which is wrong whenever the two blocks differ in weight
//     or in size. The block's value is recomputed from the ACCUMULATED
//     SUMS, never averaged.
//
// (3) THE FITTED VALUES ARE THE BLOCK MEANS, BUT THE *PREDICTION* IS A
//     LINEAR INTERPOLATION THROUGH THEM. The isotonic regression itself is a
//     STEP function; scikit-learn's `predict` is a piecewise-LINEAR
//     interpolant through the retained knots. Measured: with x = [1,2,3,4,5]
//     and y = [0,1,0,0,1] the retained knots are (1,0), (2,1/3), (4,1/3),
//     (5,1) and predict(4.5) = 2/3, where the pure step function gives 1.0.
//     THE DISTINCTION IS OBSERVABLE, and this file pins it.
//
// (4) THE RETAINED KNOTS ARE THE *TRIMMED* STEP VALUES: interior points
//     whose fitted value equals BOTH neighbours' are dropped ("remove
//     unnecessary points for faster prediction", scikit-learn's own
//     comment). The trimming does not change the interpolated function --
//     the removed point lies on the segment between its neighbours -- but it
//     DOES change the knot list, which is part of the public output.
//
// ---------------------------------------------------------------------------
// TWO MORE THAT NOBODY EXPECTS
// ---------------------------------------------------------------------------
//
// (5) THE DECREASING CASE IS NOT "NEGATE AND FIT INCREASING". It is a
//     separate PAVA run whose violation test is reversed. The two agree
//     algebraically on the FIT, but not on the knot list, because the
//     tie-collapsing and the trimming are directional.
//
// (6) ZERO-WEIGHT POINTS ARE DROPPED, NOT TREATED AS ZERO-WEIGHT POINTS.
//     scikit-learn masks `sample_weight > 0` before the fit. A zero-weight
//     point must not contribute to the tie-collapse, must not shift the
//     fitted range, and must not appear as a knot.
//
// ---------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, STATED SO IT CANNOT CREEP IN
// ---------------------------------------------------------------------------
//
//   * MULTICLASS ONE-VS-REST CALIBRATION. scikit-learn's
//     `CalibratedClassifierCV` fits one isotonic map per class and
//     renormalises the row. A different composition rule, and the
//     renormalisation is where the interesting bugs live.
//   * ANY `Layer`, ANY RANDOMNESS, ANY MODEL.
//   * QUANTILE REGRESSION / CQR-style monotone quantile fitting. A
//     different objective (pinball loss, not squared loss).
//   * CROSS-VALIDATED CALIBRATION. The repo ships `utils/cross_validation.h`
//     for it; the report below takes an explicit held-out evaluation set
//     instead, which is the same idea stated at the call site.
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// The direction of the monotone fit.
//
// Increasing -- the fitted map is non-DECREASING (a higher score never maps
// to a lower value). This is the calibration direction.
//
// Decreasing -- the fitted map is non-INCREASING. The violation test in
// PAVA is reversed; see note (5) in the header.
// ---------------------------------------------------------------------------
enum class IsotonicDirection { Increasing, Decreasing };

// ---------------------------------------------------------------------------
// What to do with a query that falls OUTSIDE the fitted score range.
//
// Clip -- return the nearer END VALUE. This is what scikit-learn's own
// calibration code uses (`IsotonicRegression(out_of_bounds="clip")`), and
// it is the default here because a calibration map must return a
// probability for EVERY score.
//
// Nan -- return NaN. Honest about the fact that the map is undefined
// there, and it propagates into any downstream metric rather than being
// silently smoothed over.
//
// Raise -- throw std::out_of_range.
//
// THE CHOICE IS THE CALLER'S. A map that silently clips hides a
// distribution shift; a map that silently returns NaN poisons a metric
// that does not check for it. Both are defensible, so neither is assumed.
// ---------------------------------------------------------------------------
enum class IsotonicOutOfBounds { Clip, Nan, Raise };

// ---------------------------------------------------------------------------
// A fitted monotone map.
//
// The map is the PIECEWISE-LINEAR interpolant through (x_knots, y_knots).
// `x_knots` is strictly increasing; `y_knots` is monotone in the fitted
// direction.
//
// The knots are the RETAINED ones (note 4): an interior point whose
// fitted value equals both neighbours' is dropped, so the knot list can be
// shorter than the number of distinct scores.
// ---------------------------------------------------------------------------
struct IsotonicFit {
    std::vector<double> x_knots;
    std::vector<double> y_knots;
    IsotonicDirection direction = IsotonicDirection::Increasing;

    size_t n_knots() const { return x_knots.size(); }

    // The fitted score range -- the bounds against which the
    // IsotonicOutOfBounds policy is applied. These are the FIRST and LAST
    // knot, which are always retained (note 4).
    double x_min() const { return x_knots.empty() ? 0.0 : x_knots.front(); }
    double x_max() const { return x_knots.empty() ? 0.0 : x_knots.back(); }
};

// ---------------------------------------------------------------------------
// The WEIGHTED L2 isotonic regression of `y` on `x`, fitted by the
// Pool-Adjacent-Violators algorithm.
//
//   x        (N,1) column of scores. Must be finite.
//   y        (N,1) column of targets. Must be finite.
//   weights  (N,1) column of NON-NEGATIVE weights, or an EMPTY tensor for
//            uniform weights. Points with weight ZERO ARE DROPPED (note 6).
//   direction  Increasing (the default) or Decreasing.
//
// The fit is: collapse tied x to a weighted mean (note 1), run PAVA
// (note 2), and trim the redundant knots (note 4).
//
// Throws std::invalid_argument if the shapes disagree, if any value is not
// finite, if a weight is negative, or if every weight is zero (which would
// leave nothing to fit).
// ---------------------------------------------------------------------------
IsotonicFit fit_isotonic(const Tensor& x, const Tensor& y,
                          const Tensor& weights = Tensor(),
                          IsotonicDirection direction = IsotonicDirection::Increasing);

// ---------------------------------------------------------------------------
// Evaluate a fitted map at a column of scores.
//
// The result is the LINEAR INTERPOLATION through the retained knots (note
// 3), with the IsotonicOutOfBounds policy applied against the fitted
// range.
//
// `x` is (N,1) and the result is (N,1).
//
// Throws std::out_of_range when the policy is Raise and a query falls
// outside the fitted range, and std::invalid_argument when `fit` has no
// knots.
// ---------------------------------------------------------------------------
Tensor isotonic_predict(const IsotonicFit& fit, const Tensor& x,
                         IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

// ---------------------------------------------------------------------------
// The CALIBRATION MAP: a monotone non-decreasing map from an arbitrary
// score to P(y = 1 | score), fitted by the isotonic regression of the 0/1
// labels on the scores.
//
//   scores  (N,1) column of scores. Any finite real values -- a
//           probability, a logit, a decision function.
//   labels  (N,1) column of BINARY labels in {0, 1}.
//   weights (N,1) column of non-negative weights, or empty for uniform.
//
// The result is an IsotonicFit with direction Increasing. Because the
// targets are 0/1, every fitted value is a convex combination of them and
// therefore lies in [0, 1] -- the map is a probability map by
// CONSTRUCTION, not by clamping.
//
// Throws std::invalid_argument if a label is not 0 or 1, and the same
// guards as fit_isotonic.
// ---------------------------------------------------------------------------
IsotonicFit fit_isotonic_calibration(const Tensor& scores, const Tensor& labels,
                                      const Tensor& weights = Tensor());

// ---------------------------------------------------------------------------
// Apply a calibration map to a column of scores.
//
// `scores` is (N,1) and the result is (N,1), each entry a calibrated
// probability in [0, 1].
// ---------------------------------------------------------------------------
Tensor isotonic_calibrate(const IsotonicFit& fit, const Tensor& scores,
                           IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

// ---------------------------------------------------------------------------
// THE REPORT: does the calibration actually help?
//
// Measured with the repo's OWN `utils/metrics.h` -- `expected_calibration_error`
// and `brier_score_loss` -- not with a re-implementation of either.
//
//   calibration_scores / calibration_labels  the set the MAP IS FITTED on.
//   eval_scores / eval_labels                the set the report is
//                                            MEASURED on.
//
// THE TWO SETS MUST BE DISTINCT, AND THAT IS THE WHOLE POINT.
//
// A map fitted by PAVA INTERPOLATES its fitting set, so the IN-SAMPLE
// calibration error of a fitted map is essentially ZERO -- measured at
// 0.00000 on the fixture in tests/test_isotonic.cpp, against a held-out
// value of 0.03333 on the same map. **A report computed on the data the
// map was fitted on measures nothing at all**, and it reports a number that
// looks wonderful. That is why this function takes a separate evaluation
// set: so that the failure mode cannot happen by accident. Passing the
// same tensor twice is legal and is exactly the in-sample report.
//
// The before/after numbers are measured on the EVALUATION set, so they
// compare like with like.
//
//   ece_before    the ECE of the RAW scores.
//   ece_after     the ECE of the CALIBRATED scores.
//   brier_before  the Brier score of the RAW scores.
//   brier_after   the Brier score of the CALIBRATED scores.
//   n_knots       the number of retained knots in the fitted map.
//   n_samples     the number of rows in the EVALUATION set.
//
// The scores must be probabilities in [0, 1] for the ECE to be meaningful
// (the repo's metrics.h enforces that), and the labels must be binary.
// ---------------------------------------------------------------------------
struct IsotonicCalibrationReport {
    double ece_before = 0.0;
    double ece_after = 0.0;
    double brier_before = 0.0;
    double brier_after = 0.0;
    size_t n_knots = 0;
    size_t n_samples = 0;
};

IsotonicCalibrationReport evaluate_isotonic_calibration(
    const Tensor& calibration_scores, const Tensor& calibration_labels,
    const Tensor& eval_scores, const Tensor& eval_labels,
    size_t n_bins = 10,
    IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_ISOTONIC_REGRESSION_H
