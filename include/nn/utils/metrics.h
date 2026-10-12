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

// ---------------------------------------------------------------------------
// Calibration diagnostics beyond ECE
//
// Three further measures, each a variation on the bin-weighting rule that
// expected_calibration_error already implements. Definitions read from the
// primary sources (fetched 2026-10-12, NOT from memory) and each pinned by a
// named test in tests/test_calibration_diagnostics.cpp.
// ---------------------------------------------------------------------------

// ADAPTIVE calibration error (Nixon et al., "Measuring Calibration in Deep
// Learning", arXiv:1904.01685 Sec. 5.2-5.3).
//
// Two rules make it different from expected_calibration_error, and BOTH are
// load-bearing:
//
//  1. BINS ARE EQUAL-COUNT, NOT EQUAL-WIDTH. Nixon Sec. 5.2: "We chose an
//     adaptive scheme which spaces the bin intervals so that each contains an
//     equal number of predictions", and Sec. 5.3 defines range r "by the
//     floor(N/R)th index of the sorted and thresholded predictions". Because
//     floor(N/R) need not divide N, the ranges are NOT equal in size -- so the
//     edges are quantiles OF THE DATA, not fixed multiples of 1/n_bins.
//  2. THE WEIGHT IS UNIFORM 1/R, not proportional to bin population. Sec. 5.3
//     writes the average as 1/(K*R). This is the point of the metric: estimate
//     the error where predictions are actually made, instead of letting a
//     sparsely-populated bin be silently down-weighted.
//
// y_true/y_prob are (N,1) binary columns; n_bins is the requested range count
// R; threshold drops predictions at or below it before binning (see
// thresholded_calibration_error). A sample whose predicted probability equals
// the threshold is KEPT. Returns 0.0 when every sample is filtered out, rather
// than NaN.
//
// A single-class fixture is NOT an error, unlike roc_auc_score: with no
// positives every bin has accuracy 0, so the metric reduces to the mean
// confidence -- a well-defined and useful measure of overconfidence.
double adaptive_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                  double pos_label, size_t n_bins,
                                  double threshold = 0.0);

// THRESHOLDED adaptive calibration error (Nixon Sec. 5.3, "TACE").
//
// The same measure, but evaluating ONLY predictions above `threshold`. The
// motivation is that "the vast majority of softmax predictions become
// infinitesimal" and a bin is otherwise wasted on a region the model never
// meaningfully occupies. Nixon states ACE is the special case of TACE at
// threshold 0 -- that identity is asserted, not assumed.
//
// threshold must lie in [0, 1]; 0.0 reproduces adaptive_calibration_error
// exactly, 1.0 keeps nothing (0.0).
double thresholded_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                     double pos_label, size_t n_bins,
                                     double threshold);

// CLASSWISE (multiclass) calibration error (Kull et al., "Beyond temperature
// scaling", NeurIPS 2019, arXiv:1910.12656, Eq. 4):
//
//   classwise-ECE = (1/k) sum_j sum_i (|B_ij| / n) |y_j(B_ij) - p_j(B_ij)|
//
// EVERY class is scored, not just the predicted one: class j pairs p_j(x)
// against the indicator Y = j. This is the metric's entire reason to exist --
// a model can be perfectly confidence-calibrated on its top prediction and
// still be badly miscalibrated on the classes it did not pick, which is
// exactly the failure temperature scaling cannot fix (the paper's Fig. 1c).
//
// Note the per-class inner weight IS the sampled |B|/n, NOT ACE's uniform
// 1/R: the two disagree whenever bins are uneven, which test 6b pins.
//
// y_true_index is an (N,1) column of class indices; y_proba is (N,C) with
// each row summing to 1. Classes with zero support still receive a (trivially
// zero) per-class score in classwise_ece_per_class, and are EXCLUDED from the
// weighted average -- an undefined estimate should not drag the mean down.
std::vector<double> classwise_ece_per_class(const Tensor& y_true_index,
                                            const Tensor& y_proba, size_t n_bins);

// Unweighted mean over the k classes (the 1/k of Kull's Eq. 4).
double classwise_ece(const Tensor& y_true_index, const Tensor& y_proba,
                     size_t n_bins);

// Macro is the plain 1/k mean. Weighted averages by class support, matching
// the w_k weighting of the paper's marginal calibration error (Eq. 3); it
// differs from Macro exactly when class counts are unequal.
double classwise_ece(const Tensor& y_true_index, const Tensor& y_proba,
                     size_t n_bins, MulticlassAveraging averaging);

// ---------------------------------------------------------------------------
// Ranking metrics beyond binary AUC: DCG / NDCG / MAP / MRR
//
// The ROC-AUC / average_precision_score functions above take a (N,1) BINARY
// label column -- one score per EXAMPLE, and the label is 0 or 1. Everything
// in this section takes a genuinely different input convention: a
// (N_queries, N_docs) matrix of RELEVANCE GRADES in [0, inf) alongside a
// matrix of scores. That is the retrieval / ranking setup, not the
// single-example setup, and no amount of reshaping turns one into the other.
//
// Definitions read from the primary sources on 2026-10-13 -- scikit-learn
// 1.9.1 `sklearn/metrics/_ranking.py` (_dcg_sample_scores :1655,
// _tie_averaged_dcg :1713, _ndcg_sample_scores :1898) and trec_eval via
// ir_measures 0.4.3 (AP, RR) as an independent second implementation. The
// rules a reimplementation gets wrong are each pinned by a named test in
// tests/test_ranking_metrics.cpp.
// ---------------------------------------------------------------------------

// DISCOUNTED CUMULATIVE GAIN (Järvelin & Kekäläinen, SIGIR 2002).
//
//   DCG@k(p) = sum_{rank=1..k} rel_p(rank) / log_base(rank + 1)
//
// The discount is log_base(rank+1) with rank starting at 1, so the TOP rank is
// discounted by exactly 1.0. Note the indexing: sklearn writes
// `discount = 1 / (log(arange(n) + 2) / log(log_base))` (_ranking.py:1698),
// i.e. the array index j maps to rank j+1 and the log argument is j+2. The
// classic `1/log2(rank)` starting at 0.6309 is an OFF-BY-ONE that gives a
// plausible but wrong number.
//
// k truncates by ZEROING THE DISCOUNT beyond k, which zeroes the gain and the
// normalizing ideal gain alike. That is why ndcg@k is normalised by the ideal
// TRUNCATED AT k and not by the full-length ideal. k == 0 is the SENTINEL for
// "no truncation" (the default), not an error; k > N_docs is allowed and
// behaves as the full length.
//
// TIES: gains are TIE-AVERAGED, not sorted pessimistically. Within a group of
// equal scores each document's gain is replaced by the group's MEAN gain and
// the group contributes mean_gain * (sum of the group's discounts) -- the
// estimator of McSherry & Najork (ECIR 2008, _tie_averaged_dcg :1713), which
// is what sklearn computes. This makes DCG independent of the order in which
// tied documents happen to sit in the row. NOTE the contrast with
// mean_reciprocal_rank below, which resolves ties the other way; the
// disagreement is a genuine inconsistency in the IR literature and both rules
// are pinned by tests.
//
// rel is (N_queries, N_docs); score is (N_queries, N_docs) and may be any
// monotone score (a probability, a logit, a decision-function value). Returns
// the unweighted MEAN over queries.
double dcg_score(const Tensor& rel, const Tensor& score,
                 size_t k = 0, double log_base = 2.0);

// NORMALIZED DCG -- DCG divided by the ideal DCG, in [0, 1].
//
// The ideal gain is computed by ranking with `rel` ITSELF as the score
// (_ranking.py:1942), which is what makes it the maximum. A query whose ideal
// gain is 0 (NOTHING is relevant) scores 0.0 -- the division is skipped, not
// guarded, and the 0.0 IS INCLUDED in the mean over queries rather than
// dropped. Dropping it would let a retrieval system buy a higher score by
// refusing to return anything, which is precisely the degenerate behaviour a
// ranking metric exists to penalise.
//
// Requires N_docs > 1 (sklearn raises otherwise: "Computing NDCG is only
// meaningful when there is more than 1 document") and rel >= 0.
double ndcg_score(const Tensor& rel, const Tensor& score,
                  size_t k = 0, double log_base = 2.0);

// MEAN AVERAGE PRECISION (the TREC / Manning et al. IR book definition):
// the mean, over the relevant documents, of the precision at that document's
// rank. Verified against trec_eval's AP on five fixtures to the last digit.
//
// This is a DIFFERENT function from the `average_precision_score` shipped
// above, which is scikit-learn's STEP integral over the PR curve. The two
// agree on the fixtures measured here, but they are different estimators and
// conflating them would silently redefine an already-shipped symbol -- hence a
// separate name. Pinned by test_map_is_mean_precision_at_relevant_ranks.
//
// GRADES ARE THRESHOLDED, NOT WEIGHTED: relevance enters as the binary
// indicator `rel > 0` and the grade magnitude is ignored. This matches trec_eval,
// which defines AP over a binary qrel view, and it is a real behavioural
// difference -- on graded relevance {3,2,1,0} the binary view gives 0.75 while a
// relevance-weighted variant gives 0.4166666666666667.
//
// k restricts the ranking window: only hits at rank <= k count. A window with
// no hits scores 0.0.
double mean_average_precision(const Tensor& rel, const Tensor& score,
                              size_t k = 0);

// MEAN RECIPROCAL RANK: the mean, over queries, of 1/rank of the FIRST
// relevant document. A query with no relevant document scores 0.0 and is
// INCLUDED in the mean (same reason as ndcg_score).
//
// TIES ARE RESOLVED PESSIMISTICALLY: within a group of equal scores, relevant
// documents are placed as late as possible (irrelevant first; relevant keep
// their relative document order). Verified against scikit-learn's 1-D
// `average_precision_score`, which is position-INDEPENDENT and returns 1/3 both
// when every score is tied with one relevant document and when the relevant
// document sits inside a two-way tie below a higher-scoring irrelevant one.
// That separates this rule from the docid-ascending one (which would give 1.0
// and 0.5) and from the tie-averaged one (0.6111 and 0.4167).
//
// A NOTE ON WHY trec_eval IS NOT CITED FOR THIS: trec_eval sorts tied
// documents by DESCENDING docid, so its RR returns 1.0 / 0.5 / 0.3333 for the
// same three tied documents depending only on where the relevant document sits
// in the run file. Its tie handling is an artifact of file bookkeeping, not
// semantics, so it is deliberately not the oracle here.
//
// Note the contrast with dcg_score above, which tie-AVERAGES instead; the
// disagreement is a genuine inconsistency in the IR literature, not in this
// file.
double mean_reciprocal_rank(const Tensor& rel, const Tensor& score, size_t k = 0);

// Per-query forms. Each returns N_queries values in row order, and the
// aggregate above is exactly their unweighted mean -- asserted by
// test_aggregate_is_mean_over_queries. Exposed because a per-query score is
// what a retrieval system actually reports, and because the difference between
// a good query and a hopeless one is invisible in the mean.
std::vector<double> dcg_per_query(const Tensor& rel, const Tensor& score,
                                  size_t k = 0, double log_base = 2.0);
std::vector<double> ndcg_per_query(const Tensor& rel, const Tensor& score,
                                   size_t k = 0, double log_base = 2.0);
std::vector<double> average_precision_per_query(const Tensor& rel,
                                                const Tensor& score,
                                                size_t k = 0);
std::vector<double> reciprocal_rank_per_query(const Tensor& rel,
                                              const Tensor& score,
                                              size_t k = 0);

// ---------------------------------------------------------------------------
// NOT IMPLEMENTED, DELIBERATELY: the "D2" score.
// ---------------------------------------------------------------------------
// The expansion queue listed a "D2 score" alongside adaptive and classwise
// ECE. D1/D2 are NOT named anywhere in the paper this metric family comes
// from -- Kumar, Liang & Ma, "Verified Uncertainty Calibration" (NeurIPS 2019,
// arXiv:1909.10155) defines the top-label CE (Def. 2.3) and the marginal CE
// (Def. 2.4) in both its v1 and v2, and its public library
// (github.com/AnanyaKumar/verified_calibration) exposes no d1/d2 symbol.
//
// Rather than ship a plausible-looking function under a name that does not
// mean what the caller will assume it means, D2 is left out until a primary
// source for its definition is available. If it turns out to be the total
// variation distance between the predicted probability vector and the one-hot
// label -- the natural "how far is this distribution from the truth" reading
// -- that is a two-line addition on top of the (N,C) loop already present in
// classwise_ece_per_class. Recorded here so the gap is deliberate rather than
// forgotten.

#endif
