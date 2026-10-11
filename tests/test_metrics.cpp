// Metrics utility tests — classification accuracy / top-k / confusion matrix / F1
//
// Tests:
//   1. accuracy_score with score matrix + integer labels
//   2. accuracy_score with one-hot labels
//   3. top_k_accuracy_score (top-1/top-2/top-C)
//   4. confusion_matrix uses rows=true labels, cols=predicted labels
//   5. classification_report per-class precision/recall/F1 + macro/weighted
//   6. predicted-label vectors are supported directly
//   7. invalid shapes / labels throw clear exceptions
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "nn/utils/metrics.h"

using std::cout;
using std::endl;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, msg)                                  \
    do {                                                                       \
        double a_ = (actual);                                                   \
        double e_ = (expected);                                                 \
        if (std::fabs(a_ - e_) <= (tol)) {                                      \
            cout << "[PASS] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << endl;                                \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << " diff=" << std::fabs(a_ - e_)       \
                 << endl;                                                      \
        }                                                                      \
        ++total;                                                               \
    } while (0)

static Tensor make_index_labels(const std::vector<size_t>& labels) {
    Tensor y(labels.size(), 1);
    for (size_t i = 0; i < labels.size(); ++i) y(i, 0) = static_cast<double>(labels[i]);
    return y;
}

static Tensor make_one_hot_labels(const std::vector<size_t>& labels, size_t classes) {
    Tensor y(labels.size(), classes);
    for (size_t i = 0; i < labels.size(); ++i) y(i, labels[i]) = 1.0;
    return y;
}

static void test_accuracy_with_integer_labels() {
    cout << "-- Test 1: accuracy_score with integer labels --" << endl;
    Tensor scores({
        {2.0, 1.0, 0.0},   // pred 0, label 0
        {0.1, 0.2, 3.0},   // pred 2, label 2
        {5.0, 4.0, 3.0},   // pred 0, label 1 (wrong)
        {0.0, 6.0, 1.0},   // pred 1, label 1
    });
    Tensor labels = make_index_labels({0, 2, 1, 1});
    CHECK_NEAR(accuracy_score(scores, labels), 0.75, 1e-12,
               "accuracy_score = 3/4 for score matrix + index labels");
}

static void test_accuracy_with_one_hot_labels() {
    cout << "-- Test 2: accuracy_score with one-hot labels --" << endl;
    Tensor scores({
        {1.0, 0.0, 0.0},   // pred 0, label 0
        {0.0, 2.0, 1.0},   // pred 1, label 2 (wrong)
        {0.0, 0.1, 4.0},   // pred 2, label 2
    });
    Tensor labels = make_one_hot_labels({0, 2, 2}, 3);
    CHECK_NEAR(accuracy_score(scores, labels), 2.0 / 3.0, 1e-12,
               "accuracy_score accepts one-hot/probability labels via argmax");
}

static void test_top_k_accuracy() {
    cout << "-- Test 3: top_k_accuracy_score --" << endl;
    Tensor scores({
        {0.9, 0.8, 0.1, 0.0},   // label 0: top-1 hit
        {0.6, 0.5, 0.4, 0.3},   // label 1: top-2 hit, top-1 miss
        {0.4, 0.5, 0.6, 0.7},   // label 1: only top-4 hit
        {0.0, 0.2, 0.3, 0.4},   // label 2: top-2 hit, top-1 miss
    });
    Tensor labels = make_index_labels({0, 1, 1, 2});
    CHECK_NEAR(top_k_accuracy_score(scores, labels, 1), 0.25, 1e-12,
               "top-1 accuracy = 1/4");
    CHECK_NEAR(top_k_accuracy_score(scores, labels, 2), 0.75, 1e-12,
               "top-2 accuracy = 3/4");
    CHECK_NEAR(top_k_accuracy_score(scores, labels, 99), 1.0, 1e-12,
               "top-k clamps k to class count, so k>=C gives 100% for valid labels");
}

static void test_confusion_matrix_orientation() {
    cout << "-- Test 4: confusion_matrix orientation --" << endl;
    // labels: [0, 0, 0, 1, 1, 2]
    // preds:  [0, 1, 2, 1, 2, 2]
    // Rows are TRUE labels; columns are PREDICTED labels:
    //   true 0: pred 0, pred 1, pred 2 once each -> [1,1,1]
    //   true 1: pred 1 once, pred 2 once         -> [0,1,1]
    //   true 2: pred 2 once                      -> [0,0,1]
    // This fixture is intentionally asymmetric so a transposed implementation
    // fails the test (e.g. cm[0,1]=1 but cm[1,0]=0).
    Tensor preds = make_index_labels({0, 1, 2, 1, 2, 2});
    Tensor labels = make_index_labels({0, 0, 0, 1, 1, 2});
    Tensor cm = confusion_matrix(preds, labels, 3);

    CHECK(cm.rows == 3 && cm.cols == 3, "confusion matrix shape is C x C");
    CHECK_NEAR(cm(0, 0), 1.0, 1e-12, "cm[true=0,pred=0] = 1");
    CHECK_NEAR(cm(0, 1), 1.0, 1e-12, "cm[true=0,pred=1] = 1");
    CHECK_NEAR(cm(0, 2), 1.0, 1e-12, "cm[true=0,pred=2] = 1");
    CHECK_NEAR(cm(1, 0), 0.0, 1e-12, "cm[true=1,pred=0] = 0");
    CHECK_NEAR(cm(1, 1), 1.0, 1e-12, "cm[true=1,pred=1] = 1");
    CHECK_NEAR(cm(1, 2), 1.0, 1e-12, "cm[true=1,pred=2] = 1");
    CHECK_NEAR(cm(2, 0), 0.0, 1e-12, "cm[true=2,pred=0] = 0");
    CHECK_NEAR(cm(2, 1), 0.0, 1e-12, "cm[true=2,pred=1] = 0");
    CHECK_NEAR(cm(2, 2), 1.0, 1e-12, "cm[true=2,pred=2] = 1");
}

static void test_classification_report_values() {
    cout << "-- Test 5: classification_report values --" << endl;
    Tensor preds = make_index_labels({0, 1, 2, 1, 2, 2});
    Tensor labels = make_index_labels({0, 0, 0, 1, 1, 2});
    ClassificationMetrics m = classification_report(preds, labels, 3);

    CHECK(m.precision.size() == 3 && m.recall.size() == 3 &&
          m.f1.size() == 3 && m.support.size() == 3,
          "classification_report returns per-class vectors of length C");
    CHECK_NEAR(m.accuracy, 0.5, 1e-12, "accuracy = trace(cm)/N = 3/6");

    CHECK_NEAR(m.precision[0], 1.0, 1e-12, "class 0 precision = 1/1");
    CHECK_NEAR(m.recall[0],    1.0 / 3.0, 1e-12, "class 0 recall = 1/3");
    CHECK_NEAR(m.f1[0],        0.5, 1e-12, "class 0 F1 = 0.5");
    CHECK(m.support[0] == 3, "class 0 support = 3");

    CHECK_NEAR(m.precision[1], 0.5, 1e-12, "class 1 precision = 1/2");
    CHECK_NEAR(m.recall[1],    0.5, 1e-12, "class 1 recall = 1/2");
    CHECK_NEAR(m.f1[1],        0.5, 1e-12, "class 1 F1 = 0.5");
    CHECK(m.support[1] == 2, "class 1 support = 2");

    CHECK_NEAR(m.precision[2], 1.0 / 3.0, 1e-12, "class 2 precision = 1/3");
    CHECK_NEAR(m.recall[2],    1.0, 1e-12, "class 2 recall = 1/1");
    CHECK_NEAR(m.f1[2],        0.5, 1e-12, "class 2 F1 = 0.5");
    CHECK(m.support[2] == 1, "class 2 support = 1");

    CHECK_NEAR(m.macro_precision, 11.0 / 18.0, 1e-12, "macro precision averages classes equally");
    CHECK_NEAR(m.macro_recall,    11.0 / 18.0, 1e-12, "macro recall averages classes equally");
    CHECK_NEAR(m.macro_f1,        0.5, 1e-12, "macro F1 averages classes equally");
    CHECK_NEAR(m.weighted_precision, 13.0 / 18.0, 1e-12, "weighted precision weights by support");
    CHECK_NEAR(m.weighted_recall,    0.5, 1e-12, "weighted recall weights by support");
    CHECK_NEAR(m.weighted_f1,        0.5, 1e-12, "weighted F1 weights by support");
}

static void test_predicted_label_vector_support() {
    cout << "-- Test 6: predicted-label vectors --" << endl;
    Tensor preds = make_index_labels({2, 0, 1, 1});
    Tensor labels = make_index_labels({2, 1, 1, 0});
    CHECK_NEAR(accuracy_score(preds, labels), 0.5, 1e-12,
               "accuracy_score accepts N x 1 predicted-label tensors");

    Tensor cm = confusion_matrix(preds, labels, 3);
    CHECK_NEAR(cm(2, 2), 1.0, 1e-12, "predicted-label vector contributes true2/pred2 count");
    CHECK_NEAR(cm(1, 0), 1.0, 1e-12, "predicted-label vector contributes true1/pred0 count");
    CHECK_NEAR(cm(1, 1), 1.0, 1e-12, "predicted-label vector contributes true1/pred1 count");
    CHECK_NEAR(cm(0, 1), 1.0, 1e-12, "predicted-label vector contributes true0/pred1 count");
}

static void test_invalid_inputs_throw() {
    cout << "-- Test 7: invalid inputs throw --" << endl;
    Tensor scores({{1.0, 0.0}, {0.0, 1.0}});
    Tensor one_label = make_index_labels({0});
    bool threw_rows = false;
    try { (void)accuracy_score(scores, one_label); }
    catch (const std::invalid_argument&) { threw_rows = true; }
    CHECK(threw_rows, "row mismatch throws invalid_argument");

    Tensor labels_bad = make_index_labels({0, 2});  // class 2 invalid for C=2
    bool threw_label = false;
    try { (void)top_k_accuracy_score(scores, labels_bad, 1); }
    catch (const std::invalid_argument&) { threw_label = true; }
    CHECK(threw_label, "out-of-range label throws invalid_argument");

    bool threw_k0 = false;
    try { (void)top_k_accuracy_score(scores, make_index_labels({0, 1}), 0); }
    catch (const std::invalid_argument&) { threw_k0 = true; }
    CHECK(threw_k0, "top_k_accuracy_score rejects k=0");

    Tensor fractional(2, 1);
    fractional(0, 0) = 0.2;
    fractional(1, 0) = 1.0;
    bool threw_fractional = false;
    try { (void)accuracy_score(make_index_labels({0, 1}), fractional); }
    catch (const std::invalid_argument&) { threw_fractional = true; }
    CHECK(threw_fractional, "fractional class index labels throw invalid_argument");

    bool threw_classes = false;
    try { (void)confusion_matrix(scores, make_index_labels({0, 1}), 0); }
    catch (const std::invalid_argument&) { threw_classes = true; }
    CHECK(threw_classes, "confusion_matrix rejects num_classes=0");
}

// ---------------------------------------------------------------------------
// Ranking / calibration metrics (ROC-AUC, average precision, Brier, log-loss,
// calibration curve, ECE).
//
// Every expected value in this section was produced by RUNNING scikit-learn
// 1.9.1 on the exact fixture below (uv venv in the scratch dir, 2026-10-11),
// not by hand-arithmetic in the test. Two of them are scikit-learn's own
// doctest outputs (marked), so they additionally pin the impl to the
// reference's published contract.
// ---------------------------------------------------------------------------

// Build a (N,1) score/probability column from a plain vector.
static Tensor make_column(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// T1: ROC curve on FIX_A = sklearn's own docstring fixture (ranking.py:83-89,
// pos_label=2 form). Ties in the score MUST collapse into a single threshold.
static void test_roc_curve_matches_reference() {
    cout << "-- Test 8: roc_curve shape and values --" << endl;
    // y_true = [0,0,1,1], y_score = [0.1, 0.4, 0.35, 0.8]
    Tensor y = make_column({0, 0, 1, 1});
    Tensor s = make_column({0.1, 0.4, 0.35, 0.8});
    ROCCurve c = roc_curve(y, s);
    CHECK(c.fpr.size() == 5, "roc_curve returns 5 points (one per distinct score + origin)");
    CHECK(c.tpr.size() == c.fpr.size(), "fpr and tpr have equal length");
    CHECK(c.thresholds.size() == c.fpr.size(), "thresholds have equal length");
    CHECK(std::isinf(c.thresholds[0]) && c.thresholds[0] > 0,
          "first threshold is +inf (sklearn prepends the all-negative point)");
    // Scores 0.1, 0.35, 0.4, 0.8 are all distinct -> 4 thresholds + origin.
    // 1 positive, 3 negatives.
    CHECK_NEAR(max_abs_diff(c.fpr, {0.0, 0.0, 0.5, 0.5, 1.0}), 0.0, 1e-12,
               "fpr == [0, 0, 0.5, 0.5, 1]");
    CHECK_NEAR(max_abs_diff(c.tpr, {0.0, 0.5, 0.5, 1.0, 1.0}), 0.0, 1e-12,
               "tpr == [0, 0.5, 0.5, 1, 1]");
    CHECK_NEAR(c.thresholds[1], 0.8, 1e-12, "thresholds are in DESCENDING score order");
    CHECK_NEAR(c.thresholds[4], 0.1, 1e-12, "last threshold is the smallest score");
    CHECK_NEAR(c.fpr[0], 0.0, 1e-12, "curve starts at fpr=0");
    CHECK_NEAR(c.tpr[0], 0.0, 1e-12, "curve starts at tpr=0");
    CHECK_NEAR(c.fpr.back(), 1.0, 1e-12, "curve ends at fpr=1");
    CHECK_NEAR(c.tpr.back(), 1.0, 1e-12, "curve ends at tpr=1");
}

// T2: ROC-AUC on the same fixture = 0.75, which is exactly the value printed in
// scikit-learn's own docstring (ranking.py:88-89).
static void test_roc_auc_score_value() {
    cout << "-- Test 9: roc_auc_score on the reference fixture --" << endl;
    Tensor y = make_column({0, 0, 1, 1});
    Tensor s = make_column({0.1, 0.4, 0.35, 0.8});
    CHECK_NEAR(roc_auc_score(y, s), 0.75, 1e-12,
               "roc_auc_score = 0.75 (scikit-learn docstring value)");
    // AUC must be the trapezoid area UNDER THE RETURNED CURVE, not under some
    // re-derived sweep: the two agreeing is the wiring proof.
    ROCCurve c = roc_curve(y, s);
    CHECK_NEAR(auc_trapezoid(c.fpr, c.tpr), roc_auc_score(y, s), 1e-12,
               "roc_auc_score == auc_trapezoid(roc_curve(...))");
}

// T3: average precision on FIX_A.
static void test_average_precision_value() {
    cout << "-- Test 10: average_precision_score value --" << endl;
    Tensor y = make_column({0, 0, 1, 1});
    Tensor s = make_column({0.1, 0.4, 0.35, 0.8});
    CHECK_NEAR(average_precision_score(y, s), 0.8333333333333333, 1e-12,
               "average_precision_score = 5/6 on FIX_A");
}

// T4: the PR curve is returned recall-DECREASING, with a (1.0, 0.0) sentinel.
static void test_precision_recall_curve_values() {
    cout << "-- Test 11: precision_recall_curve values --" << endl;
    Tensor y = make_column({0, 0, 1, 1});
    Tensor s = make_column({0.1, 0.4, 0.35, 0.8});
    PrecisionRecallCurve c = precision_recall_curve(y, s);
    CHECK_NEAR(max_abs_diff(c.precision, {0.5, 0.6666666666666666, 0.5, 1.0, 1.0}), 0.0, 1e-12,
               "pr precision == [0.5, 0.667, 0.5, 1, 1]");
    CHECK_NEAR(max_abs_diff(c.recall, {1.0, 1.0, 0.5, 0.5, 0.0}), 0.0, 1e-12,
               "pr recall == [1, 1, 0.5, 0.5, 0]  (DEcreasing)");
    CHECK_NEAR(max_abs_diff(c.thresholds, {0.1, 0.35, 0.4, 0.8}), 0.0, 1e-12,
               "pr thresholds == [0.1, 0.35, 0.4, 0.8]");
    for (size_t i = 1; i < c.recall.size(); ++i) {
        if (c.recall[i] > c.recall[i - 1] + 1e-15) {
            CHECK(false, "recall must be non-increasing throughout");
            return;
        }
    }
    CHECK(true, "recall is non-increasing throughout (monotonicity)");
    CHECK_NEAR(c.precision.back(), 1.0, 1e-12, "precision sentinel is exactly 1.0");
    CHECK_NEAR(c.recall.back(), 0.0, 1e-12, "recall sentinel is exactly 0.0");
}

// T5: THE most-pinned test in the suite. Average precision is the STEP
// integral of the PR curve; the trapezoid area under the same curve is a
// DIFFERENT and equally plausible number. On FIX_X they are 0.5428571 and
// 0.4380952. A reimplementation that integrates the curve instead of stepping
// over it returns a value that looks right and is wrong.
static void test_ap_is_not_trapezoid_area_under_pr() {
    cout << "-- Test 12: average_precision != trapezoid(PR)  [non-specificity] --" << endl;
    Tensor y = make_column({0, 1, 0, 1, 1, 0, 1, 0});
    Tensor s = make_column({0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1});
    double ap = average_precision_score(y, s);
    PrecisionRecallCurve c = precision_recall_curve(y, s);
    double trap = auc_trapezoid(c.recall, c.precision);
    CHECK_NEAR(ap, 0.5428571428571428, 1e-12, "AP(FIX_X) = 0.5428571428571428");
    CHECK_NEAR(trap, 0.4380952380952381, 1e-12, "trapezoid area under PR = 0.4380952380952381");
    CHECK(std::fabs(ap - trap) > 0.05,
          "AP and trapezoid(PR) MUST disagree on this fixture (else it cannot test)");
}

// T6: tied scores must collapse to one threshold. All-tied data has AUC 0.5
// and AP 0.25 (AP is not 0.5: the step integral of a flat PR curve is 1/4).
static void test_tied_scores_collapse_to_one_threshold() {
    cout << "-- Test 13: all-tied scores --" << endl;
    Tensor y = make_column({0, 0, 0, 1});
    Tensor s = make_column({0.5, 0.5, 0.5, 0.5});
    CHECK_NEAR(roc_auc_score(y, s), 0.5, 1e-12, "all-tied scores -> AUC exactly 0.5");
    CHECK_NEAR(average_precision_score(y, s), 0.25, 1e-12, "all-tied scores -> AP exactly 0.25");
    ROCCurve c = roc_curve(y, s);
    CHECK(c.fpr.size() == 2, "4 tied samples give exactly 2 curve points (origin + 1 threshold)");
}

// T7: perfect and inverted separation.
static void test_perfect_and_inverted_separation() {
    cout << "-- Test 14: perfect / inverted separation --" << endl;
    Tensor y = make_column({0, 0, 0, 0, 1, 1, 1, 1});
    Tensor good = make_column({0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9});
    Tensor bad = make_column({0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1});
    CHECK_NEAR(roc_auc_score(y, good), 1.0, 1e-12, "perfectly separated -> AUC exactly 1.0");
    CHECK_NEAR(average_precision_score(y, good), 1.0, 1e-12, "perfectly separated -> AP exactly 1.0");
    CHECK_NEAR(roc_auc_score(y, bad), 0.0, 1e-12, "inverted ranking -> AUC exactly 0.0");
}

// T8: a single positive among negatives.
static void test_single_positive() {
    cout << "-- Test 15: single positive sample --" << endl;
    Tensor y = make_column({0, 0, 1});
    Tensor s = make_column({0.5, 0.4, 0.9});
    CHECK_NEAR(roc_auc_score(y, s), 1.0, 1e-12, "one positive ranked top -> AUC 1.0");
    ROCCurve c = roc_curve(y, s);
    CHECK_NEAR(max_abs_diff(c.fpr, {0.0, 0.0, 1.0}), 0.0, 1e-12, "single-pos fpr == [0,0,1]");
    CHECK_NEAR(max_abs_diff(c.tpr, {0.0, 1.0, 1.0}), 0.0, 1e-12, "single-pos tpr == [0,1,1]");
}

// T9: the minimal two-sample fixture.
static void test_two_sample_minimum() {
    cout << "-- Test 16: two-sample minimum fixture --" << endl;
    Tensor y = make_column({0, 1});
    Tensor s = make_column({0.1, 0.9});
    CHECK_NEAR(roc_auc_score(y, s), 1.0, 1e-12, "two-sample AUC = 1.0");
    CHECK_NEAR(average_precision_score(y, s), 1.0, 1e-12, "two-sample AP = 1.0");
    Tensor y2 = make_column({0, 1});
    Tensor s2 = make_column({0.9, 0.1});
    CHECK_NEAR(roc_auc_score(y2, s2), 0.0, 1e-12, "two-sample inverted AUC = 0.0");
    PrecisionRecallCurve c = precision_recall_curve(y, s);
    CHECK_NEAR(max_abs_diff(c.precision, {0.5, 1.0, 1.0}), 0.0, 1e-12,
               "two-sample PR precision == [0.5, 1, 1]");
}

// T10: pos_label semantics. y uses 1 as the positive class by default; here
// the labels are 0/1 and the scores are inverted, so the curve must show the
// "everything positive first" shape rather than the trivial one.
static void test_pos_label_semantics() {
    cout << "-- Test 17: pos_label semantics --" << endl;
    Tensor y = make_column({1, 0, 1, 0});
    Tensor s = make_column({0.9, 0.1, 0.8, 0.2});
    CHECK_NEAR(roc_auc_score(y, s), 1.0, 1e-12, "pos_label defaults to 1.0 -> AUC 1.0");
    ROCCurve c = roc_curve(y, s);
    CHECK_NEAR(max_abs_diff(c.fpr, {0.0, 0.0, 0.0, 1.0}), 0.0, 1e-12,
               "pos_label fpr == [0,0,0,1] (negatives never scored above 0.2)");
    // Now the same scores with the OPPOSITE class declared positive. The one
    // thing verified against scikit-learn is that AUC FLIPS 1.0 -> 0.0: the
    // samples that were positives become negatives, so a perfectly separating
    // score becomes perfectly anti-separating. The curves themselves are NOT
    // literal mirrors of each other (sklearn gives fpr=[0,0.5,1,1],
    // tpr=[0,0,0,1] for pos_label=0 vs fpr=[0,0,0,1], tpr=[0,0.5,1,1] for
    // pos_label=1), so each curve is asserted against its own measured value
    // rather than against a relationship between the two. The AUC flip is the
    // assertion that actually proves pos_label is honoured: an impl that
    // ignored the argument would return 1.0 here.
    ROCCurve c0 = roc_curve(y, s, 0.0);
    CHECK_NEAR(roc_auc_score(y, s, 0.0), 0.0, 1e-12,
               "pos_label=0 flips AUC to 0.0 (the separated class is now negative)");
    CHECK_NEAR(max_abs_diff(c0.fpr, {0.0, 0.5, 1.0, 1.0}), 0.0, 1e-12,
               "pos_label=0 fpr == [0, 0.5, 1, 1]  [sklearn-verified]");
    CHECK_NEAR(max_abs_diff(c0.tpr, {0.0, 0.0, 0.0, 1.0}), 0.0, 1e-12,
               "pos_label=0 tpr == [0, 0, 0, 1]  [sklearn-verified]");
    CHECK(max_abs_diff(c0.fpr, c.fpr) > 1e-6 && max_abs_diff(c0.tpr, c.tpr) > 1e-6,
          "pos_label=0 produces genuinely different curves from pos_label=1");
}

// T11: invalid inputs throw.
static void test_ranking_metric_invalid_inputs() {
    cout << "-- Test 18: ranking metric invalid inputs throw --" << endl;
    Tensor two = make_column({0, 1});
    Tensor one = make_column({0});
    bool threw_rows = false;
    try { (void)roc_auc_score(two, one); }
    catch (const std::invalid_argument&) { threw_rows = true; }
    CHECK(threw_rows, "row-count mismatch throws");

    Tensor empty(0, 1);
    bool threw_empty = false;
    try { (void)roc_auc_score(empty, empty); }
    catch (const std::invalid_argument&) { threw_empty = true; }
    CHECK(threw_empty, "empty input throws");

    Tensor nan_score = make_column({0.1, std::nan(""), 0.9});
    bool threw_nan = false;
    try { (void)roc_auc_score(two, nan_score); }
    catch (const std::invalid_argument&) { threw_nan = true; }
    CHECK(threw_nan, "non-finite score throws");

    // No positives at all -> AUC is undefined, not 0.5 and not 1.0.
    Tensor all_neg = make_column({0, 0, 0});
    double single_class = roc_auc_score(all_neg, make_column({0.1, 0.2, 0.3}));
    CHECK(single_class != single_class, "single-class input yields NaN (undefined), not a number");
}

// T12-T15: Brier score. The `scale_by_half` auto rule is the trap: scikit-learn
// halves when the probability matrix has FEWER THAN 3 COLUMNS, which means a
// 2-COLUMN multiclass input IS halved while a 1-COLUMN binary input is not
// halved by the same rule.
static void test_brier_score_loss() {
    cout << "-- Test 19: brier_score_loss --" << endl;
    Tensor y = make_column({0, 1, 1, 0});
    Tensor p1 = make_column({0.1, 0.9, 0.8, 0.3});
    // (N,1) is the BINARY form and is NOT halved by "auto" -- verified
    // (0.0375 = the raw MSE). This is the sklearn doctest value.
    CHECK_NEAR(brier_score_loss(y, p1, BrierScale::Auto), 0.0375, 1e-12,
               "brier binary (N,1) auto = raw MSE = 0.0375  [sklearn doctest]");
    CHECK_NEAR(brier_score_loss(y, p1, BrierScale::NoHalf), 0.075, 1e-12,
               "brier scale_by_half=false = 0.075  [sklearn doctest]");
    CHECK_NEAR(brier_score_loss(y, p1, BrierScale::Half), 0.0375, 1e-12,
               "brier explicit scale_by_half=true = 0.0375");

    Tensor hard = make_column({0.0, 1.0, 1.0, 0.0});
    CHECK_NEAR(brier_score_loss(y, hard, BrierScale::NoHalf), 0.0, 1e-15,
               "brier of perfect hard predictions is exactly 0.0");

    // THE AUTO-RULE TEST. The "auto" halving fires on a 2-COLUMN matrix -- the
    // C < 3 column-count rule applies only from 2 columns up, because a
    // 1-column matrix is binary and the reference expands it to (n,2) before
    // consulting the rule. So: (N,1) not halved, (N,2) halved, (N,3) not.
    Tensor y2 = make_column({0, 1});
    Tensor p2({{0.7, 0.3}, {0.3, 0.7}});
    CHECK_NEAR(brier_score_loss(y2, p2, BrierScale::Auto), 0.09, 1e-12,
               "2-column brier auto-halves -> 0.09");
    CHECK_NEAR(brier_score_loss(y2, p2, BrierScale::NoHalf), 0.18, 1e-12,
               "2-column brier scale_by_half=false -> 0.18");
    CHECK_NEAR(brier_score_loss(y2, p2, BrierScale::Half), 0.09, 1e-12,
               "2-column brier explicit scale_by_half=true = same as auto = 0.09");

    // 3 columns: no halving.
    Tensor y3 = make_column({0, 1, 2});
    Tensor p3({{0.8, 0.1, 0.1}, {0.2, 0.7, 0.1}, {0.2, 0.2, 0.6}});
    CHECK_NEAR(brier_score_loss(y3, p3, BrierScale::Auto), 0.1466666666666667, 1e-12,
               "3-column brier auto does NOT halve -> 0.1466667  [sklearn doctest]");
}

// T16-T19: log-loss. Two distinct subtleties, each a separate test.
static void test_log_loss() {
    cout << "-- Test 20: log_loss --" << endl;
    Tensor y = make_column({0, 1, 1, 0});
    Tensor p1 = make_column({0.1, 0.9, 0.8, 0.3});
    CHECK_NEAR(log_loss(y, p1), 0.19763488164214868, 1e-12,
               "log_loss 1-column = 0.19763488164214868");

    // Clip. In the (N,1) binary form the column is P(class 1), so a row with
        // label 1 reads the column and label 0 reads (1 - column). The clip to
        // [DBL_EPSILON, 1 - DBL_EPSILON] is what keeps -log() finite: a label-0
        // row whose column is 1.0 means P(class 0) = 0, which clips to
        // DBL_EPSILON and yields the finite -log(DBL_EPSILON) = 36.04365338911715.
        // Without the clip this is -inf. Both directions are asserted because the
        // label selects which of the two clips applies.
        Tensor yzero = make_column({0});
        Tensor pone = make_column({1.0});
        CHECK_NEAR(log_loss(yzero, pone), 36.04365338911715, 1e-9,
                   "log_loss clips a label-0 row at p=1.0 to -log(DBL_EPSILON) = 36.04365338911715 (NOT -inf)");
        Tensor yone = make_column({1});
        CHECK_NEAR(log_loss(yone, pone), 2.2204460492503136e-16, 1e-24,
                   "log_loss label-1 row at p=1.0 -> -log(1-DBL_EPSILON) = DBL_EPSILON");
        Tensor yzero2 = make_column({0});
        Tensor pzero = make_column({0.0});
        CHECK_NEAR(log_loss(yzero2, pzero), 2.2204460492503136e-16, 1e-24,
                   "log_loss label-0 row at p=0.0 -> -log(1) = 0 (clipped to 1-eps)");
        Tensor yone2 = make_column({1});
        CHECK_NEAR(log_loss(yone2, pzero), 36.04365338911715, 1e-9,
                   "log_loss label-1 row at p=0.0 -> -log(DBL_EPSILON) = 36.04365338911715");
        CHECK_NEAR(std::numeric_limits<double>::epsilon(), 2.2204460492503136e-16, 1e-30,
                   "the clip constant is DBL_EPSILON");

    // xlogy: a 2-column row contributes ONLY the true class's -log(p). Summing
    // -log(p_c) over all columns gives a different (wrong) number.
    Tensor y2 = make_column({0, 1, 1, 0});
    Tensor p2({{0.1, 0.9}, {0.9, 0.1}, {0.8, 0.2}, {0.35, 0.65}});
    CHECK_NEAR(log_loss(y2, p2), 1.8161075557302173, 1e-12,
               "log_loss 2-column counts ONLY the true class (xlogy)");
    CHECK(std::fabs(log_loss(y2, p2) - log_loss(y, p1)) > 0.1,
          "1-column and 2-column log_loss MUST differ on this fixture");

    bool threw = false;
    try { (void)log_loss(y, make_column({0.1, 1.5, 0.8, 0.3})); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "log_loss rejects a probability outside [0,1]");
}

// T20-T23: calibration curve.
static void test_calibration_curve() {
    cout << "-- Test 21: calibration_curve --" << endl;
    Tensor y = make_column({0, 0, 0, 0, 1, 1, 1, 1, 1});
    Tensor p = make_column({0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0});

    CalibrationCurve c = calibration_curve(y, p, 3, 1.0);
    // scikit-learn docstring (calibration.py:1409-1414).
    CHECK_NEAR(max_abs_diff(c.prob_true, {0.0, 0.5, 1.0}), 0.0, 1e-12,
               "uniform prob_true == [0, 0.5, 1]  [sklearn doctest]");
    CHECK_NEAR(max_abs_diff(c.prob_pred, {0.2, 0.525, 0.85}), 0.0, 1e-12,
               "uniform prob_pred == [0.2, 0.525, 0.85]  [sklearn doctest]");
    CHECK(c.counts.size() == 3, "counts has one entry per reported bin");
    CHECK(c.counts[0] == 3 && c.counts[1] == 2 && c.counts[2] == 4,
          "uniform bin counts == [3, 2, 4]");
    CHECK(c.prob_true.size() == c.prob_pred.size(), "prob_true/prob_pred lengths agree");

    CalibrationCurve q = calibration_curve(y, p, 3, 1.0, CalibrationStrategy::Quantile);
    CHECK_NEAR(max_abs_diff(q.prob_true, {0.0, 0.6666666666666666, 1.0}), 0.0, 1e-12,
               "quantile prob_true == [0, 0.667, 1]");
    CHECK_NEAR(max_abs_diff(q.prob_pred, {0.2, 0.5833333333333334, 0.9}), 0.0, 1e-12,
               "quantile prob_pred == [0.2, 0.5833333333333334, 0.9]");
    CHECK(q.counts[0] == 3 && q.counts[1] == 3 && q.counts[2] == 3,
          "quantile bin counts == [3, 3, 3] (equal-mass by construction)");
    CHECK(std::fabs(c.prob_pred[1] - q.prob_pred[1]) > 1e-6,
          "uniform and quantile strategies MUST give different bin means");

    // Empty bins are DROPPED: 4 samples all inside [0,0.2) with n_bins=5.
    Tensor ysmall = make_column({0, 1, 1, 0});
    Tensor psmall = make_column({0.05, 0.06, 0.07, 0.08});
    CalibrationCurve csmall = calibration_curve(ysmall, psmall, 5, 1.0);
    CHECK(csmall.prob_true.size() == 1, "empty bins are DROPPED (4 samples, 5 bins -> 1 entry)");
    CHECK(csmall.counts[0] == 4, "the single surviving bin holds all 4 samples");
    CHECK_NEAR(csmall.prob_pred[0], 0.065, 1e-12, "surviving bin's mean probability = 0.065");
    CHECK_NEAR(csmall.prob_true[0], 0.5, 1e-12, "surviving bin's positive fraction = 0.5");
}

// T24: bin-edge assignment. np.searchsorted's default is side='left', so a
// probability landing exactly on an interior edge goes to the LOWER bin.
static void test_calibration_bin_edge_is_left_closed() {
    cout << "-- Test 22: calibration bin edges are LEFT-closed --" << endl;
    // n_bins=2 -> interior edge at 0.5. Samples at 0.4, 0.5, 0.6.
    Tensor y = make_column({0, 1, 1});
    Tensor p = make_column({0.4, 0.5, 0.6});
    CalibrationCurve c = calibration_curve(y, p, 2, 1.0);
    CHECK(c.counts.size() == 2, "both bins are non-empty");
    CHECK(c.counts[0] == 2 && c.counts[1] == 1,
          "p=0.5 lands in the LOWER bin (counts [2,1], not [1,2])");
    CHECK_NEAR(c.prob_pred[0], 0.45, 1e-12, "lower bin mean = (0.4+0.5)/2");
    CHECK_NEAR(c.prob_pred[1], 0.6, 1e-12, "upper bin mean = 0.6");
}

// T25-T26: ECE.
static void test_expected_calibration_error() {
    cout << "-- Test 23: expected_calibration_error --" << endl;
    Tensor y = make_column({0, 0, 0, 0, 1, 1, 1, 1, 1});
    Tensor p = make_column({0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0});
    CHECK_NEAR(expected_calibration_error(y, p, 3, 1.0), 0.13888888888888892, 1e-12,
               "ECE uniform n_bins=3 = 0.13888888888888892");
    CHECK_NEAR(expected_calibration_error(y, p, 3, 1.0, CalibrationStrategy::Quantile),
               0.12777777777777777, 1e-12, "ECE quantile n_bins=3 = 0.12777777777777777");

    // ECE == 0 on data constructed to be perfectly calibrated.
    Tensor cy, cp;
    {
        std::vector<double> yy, pp;
        const double ps[] = {0.1, 0.3, 0.5, 0.7, 0.9};
        for (double pv : ps) {
            size_t npos = static_cast<size_t>(std::lround(pv * 10));
            for (size_t i = 0; i < 10; ++i) {
                yy.push_back(i < npos ? 1.0 : 0.0);
                pp.push_back(pv);
            }
        }
        cy = make_column(yy);
        cp = make_column(pp);
    }
    CalibrationCurve cc = calibration_curve(cy, cp, 5, 1.0);
    CHECK(cc.prob_true.size() == 5, "calibrated-by-construction fixture fills all 5 bins");
    double ece = expected_calibration_error(cy, cp, 5, 1.0);
    CHECK_NEAR(ece, 0.0, 1e-12, "ECE == 0 on a perfectly calibrated fixture");

    bool threw = false;
    try { (void)calibration_curve(y, make_column({0.1, 1.4, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0}), 5, 1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "calibration_curve rejects a probability outside [0,1]");

    bool threw_bins = false;
    try { (void)calibration_curve(y, p, 0, 1.0); }
    catch (const std::invalid_argument&) { threw_bins = true; }
    CHECK(threw_bins, "calibration_curve rejects n_bins=0");
}

// T27-T31: one-vs-rest multiclass averaging. The searched fixture is the one
// sklearn reports as macro=0.388889 / weighted=0.541667, i.e. the two averages
// genuinely differ, so a reimplementation that always returns one of them is
// caught.
static void test_multiclass_ovr_averaging() {
    cout << "-- Test 24: multiclass one-vs-rest averaging --" << endl;
    Tensor y = make_column({2, 2, 1, 1, 1, 1, 0, 1});
    Tensor p({
        {0.8309305373525558, 0.08125819134993448, 0.08781127129750985},
        {0.29395218002812934, 0.22784810126582278, 0.4781997187060478},
        {0.25980392156862747, 0.0, 0.7401960784313726},
        {0.20570264765784116, 0.7413441955193483, 0.05295315682281059},
        {0.5339034819792303, 0.3750763591936469, 0.09102015882712278},
        {0.26168224299065423, 0.36033229491173413, 0.37798546209761164},
        {0.06259541984732825, 0.4320610687022901, 0.5053435114503817},
        {0.4498069498069498, 0.46718146718146714, 0.083011583011583},
    });
    CHECK_NEAR(roc_auc_score(y, p, MulticlassAveraging::Macro), 0.3888888888888889, 1e-12,
               "3-class macro OVR ROC-AUC = 0.3888888888888889");
    CHECK_NEAR(roc_auc_score(y, p, MulticlassAveraging::Weighted), 0.5416666666666667, 1e-12,
               "3-class weighted OVR ROC-AUC = 0.5416666666666667");
    CHECK(std::fabs(roc_auc_score(y, p, MulticlassAveraging::Macro) -
                    roc_auc_score(y, p, MulticlassAveraging::Weighted)) > 0.05,
          "macro and weighted MUST differ on this fixture");

    // Perfectly separable 3-class: both averages collapse to 1.0.
    Tensor yp = make_column({0, 1, 2, 0, 1, 2});
    Tensor pp({
        {0.7, 0.2, 0.1}, {0.2, 0.7, 0.1}, {0.1, 0.2, 0.7},
        {0.6, 0.3, 0.1}, {0.1, 0.6, 0.3}, {0.2, 0.2, 0.6},
    });
    CHECK_NEAR(roc_auc_score(yp, pp, MulticlassAveraging::Macro), 1.0, 1e-12,
               "separable 3-class macro OVR AUC = 1.0");
    CHECK_NEAR(roc_auc_score(yp, pp, MulticlassAveraging::Weighted), 1.0, 1e-12,
               "separable 3-class weighted OVR AUC = 1.0");

    // Rows that do not sum to 1 are rejected -- sklearn checks allclose(1, sum).
    Tensor bad({{0.5, 0.2}, {0.3, 0.3}});
    bool threw = false;
    try { (void)roc_auc_score(make_column({0, 1}), bad, MulticlassAveraging::Macro); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "multiclass ROC-AUC rejects rows that do not sum to 1");
}

// T32: determinism.
static void test_ranking_metrics_deterministic() {
    cout << "-- Test 25: ranking metrics are deterministic --" << endl;
    Tensor y = make_column({0, 1, 0, 1, 1, 0, 1, 0});
    Tensor s = make_column({0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1});
    double a1 = roc_auc_score(y, s);
    double a2 = roc_auc_score(y, s);
    double p1 = average_precision_score(y, s);
    double p2 = average_precision_score(y, s);
    CHECK(a1 == a2, "roc_auc_score is bit-identical across calls");
    CHECK(p1 == p2, "average_precision_score is bit-identical across calls");
    ROCCurve r1 = roc_curve(y, s);
    ROCCurve r2 = roc_curve(y, s);
    CHECK(max_abs_diff(r1.fpr, r2.fpr) == 0.0 && max_abs_diff(r1.tpr, r2.tpr) == 0.0,
          "roc_curve is bit-identical across calls");
}

int main() {
    cout << "=== Metrics Utility Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_accuracy_with_integer_labels();
    test_accuracy_with_one_hot_labels();
    test_top_k_accuracy();
    test_confusion_matrix_orientation();
    test_classification_report_values();
    test_predicted_label_vector_support();
    test_invalid_inputs_throw();

    test_roc_curve_matches_reference();
    test_roc_auc_score_value();
    test_average_precision_value();
    test_precision_recall_curve_values();
    test_ap_is_not_trapezoid_area_under_pr();
    test_tied_scores_collapse_to_one_threshold();
    test_perfect_and_inverted_separation();
    test_single_positive();
    test_two_sample_minimum();
    test_pos_label_semantics();
    test_ranking_metric_invalid_inputs();

    test_brier_score_loss();
    test_log_loss();

    test_calibration_curve();
    test_calibration_bin_edge_is_left_closed();
    test_expected_calibration_error();

    test_multiclass_ovr_averaging();
    test_ranking_metrics_deterministic();

    cout << "=== Result: " << passed << "/" << total << " passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
