#ifndef NN_UTILS_METRICS_H
#define NN_UTILS_METRICS_H

#include "../core/tensor.h"
#include <cstddef>
#include <vector>

// Classification metrics for model evaluation.
//
// Conventions:
//   * Predictions may be either (N, C) score/probability tensors (argmax per row)
//     or (N, 1) integer predicted-label tensors.
//   * Labels may be either (N, 1) integer class-index tensors or (N, C)
//     one-hot / probability tensors (argmax per row).
//   * confusion_matrix rows are TRUE labels and columns are PREDICTED labels,
//     matching scikit-learn's standard orientation.
//
// All functions throw std::invalid_argument on shape mismatches, non-integer
// class-index tensors, or out-of-range labels.

struct ClassificationMetrics {
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1;
    std::vector<size_t> support;

    double accuracy = 0.0;
    double macro_precision = 0.0;
    double macro_recall = 0.0;
    double macro_f1 = 0.0;
    double weighted_precision = 0.0;
    double weighted_recall = 0.0;
    double weighted_f1 = 0.0;
};

double accuracy_score(const Tensor& predictions, const Tensor& labels);
double top_k_accuracy_score(const Tensor& scores, const Tensor& labels, size_t k);
Tensor confusion_matrix(const Tensor& predictions, const Tensor& labels, size_t num_classes);
ClassificationMetrics classification_report(const Tensor& predictions,
                                            const Tensor& labels,
                                            size_t num_classes);

// ===========================================================================
// Ranking and calibration metrics
//
// The four functions above are all ARGMAX-based: they collapse a score vector
// to one integer per row before measuring anything, which throws away exactly
// the information a ranking metric needs. This section measures the two
// properties a production classifier is actually judged on:
//
//   1. RANKING QUALITY  -- does a higher score mean "more likely positive"?
//      roc_auc_score / average_precision_score.
//
//   2. PROBABILITY QUALITY -- when the model says 0.9, is it right 90% of the
//      time? brier_score_loss / log_loss / calibration_curve /
//      expected_calibration_error.
//
// Semantics follow scikit-learn (fetched and read 2026-10-11: metrics/
// _ranking.py, metrics/_classification.py, calibration/calibration.py). The
// rules a reimplementation gets wrong are each noted at their declaration and
// pinned by a named test in tests/test_metrics.cpp.
// ===========================================================================

// ---------------------------------------------------------------------------
// Averaging mode for the per-class multiclass metrics. Mirrors scikit-learn's
// `average=` argument.
//   Macro    -- unweighted mean over classes (ignores class imbalance)
//   Weighted -- mean weighted by per-class support (#true instances)
// ---------------------------------------------------------------------------
#ifndef NN_METRICS_AVERAGING_ENUM_DEFINED
#define NN_METRICS_AVERAGING_ENUM_DEFINED
enum class MulticlassAveraging { Macro, Weighted };
#endif

// `scale_by_half` for brier_score_loss. scikit-learn's default is "auto",
// which halves the score when the probability matrix has FEWER THAN 3 COLUMNS
// (metrics/_classification.py:3849-3852) -- a rule about column COUNT, not
// about whether the data is binary. A 2-column matrix IS halved; a 3-column
// matrix is NOT, even when the data inside it is binary-shaped. The (N,1)
// binary form is also halved: it is scored as the equivalent two-column
// (1-p, p) form first, so the factor cancels exactly. All three verified by
// measurement against scikit-learn 1.9.1 (0.0375 / 0.09 / 0.1466667) and
// pinned by test_brier_score_loss.
#ifndef NN_METRICS_BRIER_SCALE_ENUM_DEFINED
#define NN_METRICS_BRIER_SCALE_ENUM_DEFINED
enum class BrierScale { Auto, Half, NoHalf };
#endif

// Bin-edge strategy for calibration_curve.
#ifndef NN_METRICS_CALIBRATION_STRATEGY_ENUM_DEFINED
#define NN_METRICS_CALIBRATION_STRATEGY_ENUM_DEFINED
enum class CalibrationStrategy { Uniform, Quantile };
#endif

// Receiver operating characteristic. thresholds[0] is +inf (the all-negative
// operating point) and the thresholds then DECREASE, so fpr/tpr are ascending
// and fpr is non-decreasing -- which is what makes the trapezoid rule valid.
struct ROCCurve {
    std::vector<double> fpr;
    std::vector<double> tpr;
    std::vector<double> thresholds;
};

// Precision-recall curve, returned RECALL-DECREASING with a (1.0, 0.0)
// sentinel appended at the end (calibration/_ranking.py:1202-1210). The
// sentinel is what makes average_precision_score's step integral well-defined.
struct PrecisionRecallCurve {
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> thresholds;
};

// Reliability diagram. EMPTY BINS ARE DROPPED, so every vector has the same
// length, which is the count of non-empty bins and is <= n_bins.
struct CalibrationCurve {
    std::vector<double> prob_true;   // positive fraction per reported bin
    std::vector<double> prob_pred;   // mean predicted probability per bin
    std::vector<size_t> counts;      // samples per reported bin
};

// Trapezoidal area under an arbitrary curve. Direction-aware: an x that is
// non-increasing yields the positive area, matching sklearn's auc().
double auc_trapezoid(const std::vector<double>& x, const std::vector<double>& y);

// Binary ROC-AUC. y_true and y_score are (N, 1) columns; y_score need not be a
// probability (it may be a decision function / any monotone score).
// Returns NaN when only one class is present -- AUC is undefined there, and
// returning 0.5 or 1.0 would both be wrong answers to an ill-posed question.
double roc_auc_score(const Tensor& y_true, const Tensor& y_score, double pos_label = 1.0);

// Area under the PR curve in the STEP sense: -sum(d_recall * precision[:-1]),
// clamped at 0. This is NOT auc_trapezoid over the same curve and the two
// disagree on ordinary data (see test_ap_is_not_trapezoid_area_under_pr).
double average_precision_score(const Tensor& y_true, const Tensor& y_score,
                               double pos_label = 1.0);

ROCCurve roc_curve(const Tensor& y_true, const Tensor& y_score, double pos_label = 1.0);
PrecisionRecallCurve precision_recall_curve(const Tensor& y_true, const Tensor& y_score,
                                            double pos_label = 1.0);

// Mean squared error between the predicted probability and the one-hot label,
// optionally halved. y_proba may be (N, 1) (binary probability of class 1) or
// (N, C) (class probabilities). BrierScale::Auto halves when cols < 3.
double brier_score_loss(const Tensor& y_true_index, const Tensor& y_proba,
                        BrierScale scale = BrierScale::Auto);

// Mean negative log-likelihood of the labels under y_proba. Probabilities are
// clipped to [DBL_EPSILON, 1 - DBL_EPSILON] BEFORE the log, so a hard
// prediction of 1.0 on a true label yields the finite -log(DBL_EPSILON)
// (~36.04) rather than -inf. Only the TRUE class of each row contributes
// (sklearn's _xlogy), which is why the 1-column and 2-column forms of the same
// data give different answers.
double log_loss(const Tensor& y_true_index, const Tensor& y_proba);

// Reliability diagram. n_bins equal-width intervals over [0,1] (Uniform) or
// n_bins equal-mass intervals (Quantile). Bin assignment is LEFT-closed, so a
// probability exactly on an interior edge falls in the LOWER bin.
CalibrationCurve calibration_curve(const Tensor& y_true, const Tensor& y_prob,
                                   size_t n_bins, double pos_label,
                                   CalibrationStrategy strategy = CalibrationStrategy::Uniform);

// Sum over reported bins of (bin_count / n) * |prob_true - prob_pred|.
double expected_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                  size_t n_bins, double pos_label,
                                  CalibrationStrategy strategy = CalibrationStrategy::Uniform);

// One-vs-rest multiclass ROC-AUC. y_proba is (N, C) with each row summing to
// 1; the macro/weighted difference is real whenever class counts differ.
double roc_auc_score(const Tensor& y_true_index, const Tensor& y_proba,
                     MulticlassAveraging averaging);

// One-vs-rest multiclass average precision, same one-hot argument convention.
double average_precision_score(const Tensor& y_true_index, const Tensor& y_proba,
                               MulticlassAveraging averaging);

#endif
