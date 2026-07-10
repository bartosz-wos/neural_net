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
#include <cmath>
#include <iostream>
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

    cout << "=== Result: " << passed << "/" << total << " passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
