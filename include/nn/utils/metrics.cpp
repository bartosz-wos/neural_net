#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require_nonempty_rows(const Tensor& t, const char* name) {
    if (t.rows == 0) {
        throw std::invalid_argument(std::string(name) + " must have at least one row");
    }
}

size_t checked_index_value(double value, size_t num_classes, const char* name, size_t row) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " contains non-finite class index at row " +
                                    std::to_string(row));
    }
    double rounded = std::round(value);
    if (std::fabs(value - rounded) > 1e-12) {
        throw std::invalid_argument(std::string(name) + " contains non-integer class index at row " +
                                    std::to_string(row));
    }
    if (rounded < 0.0 || rounded > static_cast<double>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(std::string(name) + " contains invalid class index at row " +
                                    std::to_string(row));
    }
    size_t idx = static_cast<size_t>(rounded);
    if (idx >= num_classes) {
        throw std::invalid_argument(std::string(name) + " class index out of range at row " +
                                    std::to_string(row));
    }
    return idx;
}

size_t argmax_row(const Tensor& t, size_t row) {
    if (t.cols == 0) {
        throw std::invalid_argument("argmax requires at least one column");
    }
    size_t best = 0;
    double best_val = t(row, 0);
    for (size_t j = 1; j < t.cols; ++j) {
        if (t(row, j) > best_val) {
            best_val = t(row, j);
            best = j;
        }
    }
    return best;
}

std::vector<size_t> extract_classes(const Tensor& t, size_t num_classes, const char* name) {
    require_nonempty_rows(t, name);
    if (num_classes == 0) {
        throw std::invalid_argument("num_classes must be positive");
    }
    if (t.cols == 0) {
        throw std::invalid_argument(std::string(name) + " must have at least one column");
    }

    std::vector<size_t> out(t.rows);
    if (t.cols == 1) {
        for (size_t i = 0; i < t.rows; ++i) {
            out[i] = checked_index_value(t(i, 0), num_classes, name, i);
        }
    } else {
        if (t.cols != num_classes) {
            throw std::invalid_argument(std::string(name) + " score/probability columns must match num_classes");
        }
        for (size_t i = 0; i < t.rows; ++i) {
            out[i] = argmax_row(t, i);
        }
    }
    return out;
}

size_t inferred_num_classes(const Tensor& predictions, const Tensor& labels) {
    size_t inferred = 0;
    if (predictions.cols > 1) inferred = std::max(inferred, predictions.cols);
    if (labels.cols > 1) inferred = std::max(inferred, labels.cols);
    if (inferred == 0) {
        for (size_t i = 0; i < predictions.rows; ++i) {
            double v = predictions(i, 0);
            if (!std::isfinite(v) || std::fabs(v - std::round(v)) > 1e-12 || v < 0.0) {
                throw std::invalid_argument("predictions contain invalid class index");
            }
            inferred = std::max(inferred, static_cast<size_t>(std::round(v)) + 1);
        }
        for (size_t i = 0; i < labels.rows; ++i) {
            double v = labels(i, 0);
            if (!std::isfinite(v) || std::fabs(v - std::round(v)) > 1e-12 || v < 0.0) {
                throw std::invalid_argument("labels contain invalid class index");
            }
            inferred = std::max(inferred, static_cast<size_t>(std::round(v)) + 1);
        }
    }
    if (inferred == 0) {
        throw std::invalid_argument("could not infer num_classes from empty inputs");
    }
    return inferred;
}

void require_same_rows(const Tensor& predictions, const Tensor& labels) {
    require_nonempty_rows(predictions, "predictions");
    require_nonempty_rows(labels, "labels");
    if (predictions.rows != labels.rows) {
        throw std::invalid_argument("predictions and labels must have the same number of rows");
    }
}

}  // namespace

double accuracy_score(const Tensor& predictions, const Tensor& labels) {
    require_same_rows(predictions, labels);
    size_t num_classes = inferred_num_classes(predictions, labels);
    std::vector<size_t> pred = extract_classes(predictions, num_classes, "predictions");
    std::vector<size_t> truth = extract_classes(labels, num_classes, "labels");

    size_t correct = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] == truth[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(pred.size());
}

double top_k_accuracy_score(const Tensor& scores, const Tensor& labels, size_t k) {
    require_same_rows(scores, labels);
    if (k == 0) {
        throw std::invalid_argument("top_k_accuracy_score requires k >= 1");
    }
    if (scores.cols <= 1) {
        if (k != 1) {
            throw std::invalid_argument("top_k_accuracy_score requires a score matrix for k > 1");
        }
        return accuracy_score(scores, labels);
    }

    const size_t num_classes = scores.cols;
    std::vector<size_t> truth = extract_classes(labels, num_classes, "labels");
    const size_t kk = std::min(k, num_classes);

    size_t hits = 0;
    std::vector<size_t> order(num_classes);
    for (size_t i = 0; i < scores.rows; ++i) {
        for (size_t j = 0; j < num_classes; ++j) order[j] = j;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) {
                             if (scores(i, a) == scores(i, b)) return a < b;
                             return scores(i, a) > scores(i, b);
                         });
        bool hit = false;
        for (size_t r = 0; r < kk; ++r) {
            if (order[r] == truth[i]) {
                hit = true;
                break;
            }
        }
        if (hit) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(scores.rows);
}

Tensor confusion_matrix(const Tensor& predictions, const Tensor& labels, size_t num_classes) {
    require_same_rows(predictions, labels);
    if (num_classes == 0) {
        throw std::invalid_argument("confusion_matrix requires num_classes > 0");
    }
    std::vector<size_t> pred = extract_classes(predictions, num_classes, "predictions");
    std::vector<size_t> truth = extract_classes(labels, num_classes, "labels");

    Tensor cm = Tensor::zeros(num_classes, num_classes);
    for (size_t i = 0; i < pred.size(); ++i) {
        cm(truth[i], pred[i]) += 1.0;
    }
    return cm;
}

ClassificationMetrics classification_report(const Tensor& predictions,
                                            const Tensor& labels,
                                            size_t num_classes) {
    Tensor cm = confusion_matrix(predictions, labels, num_classes);
    ClassificationMetrics m;
    m.precision.assign(num_classes, 0.0);
    m.recall.assign(num_classes, 0.0);
    m.f1.assign(num_classes, 0.0);
    m.support.assign(num_classes, 0);

    double trace = 0.0;
    double total = 0.0;
    for (size_t c = 0; c < num_classes; ++c) {
        trace += cm(c, c);
        for (size_t p = 0; p < num_classes; ++p) {
            total += cm(c, p);
        }
    }
    m.accuracy = (total > 0.0) ? trace / total : 0.0;

    size_t total_support = 0;
    for (size_t c = 0; c < num_classes; ++c) {
        double tp = cm(c, c);
        double predicted_as_c = 0.0;
        double actual_c = 0.0;
        for (size_t r = 0; r < num_classes; ++r) predicted_as_c += cm(r, c);
        for (size_t p = 0; p < num_classes; ++p) actual_c += cm(c, p);

        m.support[c] = static_cast<size_t>(actual_c);
        total_support += m.support[c];
        m.precision[c] = (predicted_as_c > 0.0) ? tp / predicted_as_c : 0.0;
        m.recall[c] = (actual_c > 0.0) ? tp / actual_c : 0.0;
        double denom = m.precision[c] + m.recall[c];
        m.f1[c] = (denom > 0.0) ? (2.0 * m.precision[c] * m.recall[c] / denom) : 0.0;

        m.macro_precision += m.precision[c];
        m.macro_recall += m.recall[c];
        m.macro_f1 += m.f1[c];
    }

    if (num_classes > 0) {
        m.macro_precision /= static_cast<double>(num_classes);
        m.macro_recall /= static_cast<double>(num_classes);
        m.macro_f1 /= static_cast<double>(num_classes);
    }

    if (total_support > 0) {
        for (size_t c = 0; c < num_classes; ++c) {
            double w = static_cast<double>(m.support[c]) / static_cast<double>(total_support);
            m.weighted_precision += w * m.precision[c];
            m.weighted_recall += w * m.recall[c];
            m.weighted_f1 += w * m.f1[c];
        }
    }

    return m;
}

// ===========================================================================
// Ranking and calibration metrics
//
// Mirrors scikit-learn's internal decomposition: one `binary_clf_curve`
// helper feeds both roc_curve and precision_recall_curve, so the two cannot
// drift apart. Source read 2026-10-11 (metrics/_ranking.py,
// metrics/_classification.py, calibration/calibration.py).
// ===========================================================================
namespace {

struct BinaryClfCurve {
    std::vector<double> fps;
    std::vector<double> tps;
    std::vector<double> thresholds;
};

// tps/fps accumulated over samples sorted by DESCENDING score, emitted once
// per DISTINCT score value (_ranking.py:914-918). One point per sample instead
// of one per distinct score is the single easiest way to get this wrong: it
// makes tied data produce a curve with redundant collinear points, which
// changes the trapezoid area.
BinaryClfCurve binary_clf_curve(const std::vector<double>& y_true_bin,
                                const std::vector<double>& y_score) {
    const size_t n = y_true_bin.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    // Stable so that ties keep their input order -- makes the result
    // deterministic and independent of the sort implementation.
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return y_score[a] > y_score[b]; });

    std::vector<double> sorted_scores(n);
    std::vector<double> cum_tp(n + 1, 0.0);
    std::vector<double> cum_fp(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        sorted_scores[i] = y_score[order[i]];
        cum_tp[i + 1] = cum_tp[i] + y_true_bin[order[i]];
        cum_fp[i + 1] = cum_fp[i] + (1.0 - y_true_bin[order[i]]);
    }

    BinaryClfCurve out;
    for (size_t i = 0; i < n; ++i) {
        const bool last = (i + 1 == n);
        if (last || sorted_scores[i] != sorted_scores[i + 1]) {
            out.tps.push_back(cum_tp[i + 1]);
            out.fps.push_back(cum_fp[i + 1]);
            out.thresholds.push_back(sorted_scores[i]);
        }
    }
    return out;
}

// Validate + flatten a (N,1) label column and a (N,C) score/probability block.
void require_binary_columns(const Tensor& y_true, const Tensor& y_score,
                            const char* fn) {
    if (y_true.rows == 0 || y_score.rows == 0) {
        throw std::invalid_argument(std::string(fn) + " requires at least one row");
    }
    if (y_true.rows != y_score.rows) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires y_true and y_score to have the same number of rows");
    }
    if (y_true.cols != 1) {
        throw std::invalid_argument(std::string(fn) + " requires y_true to be a (N,1) column");
    }
}

std::vector<double> label_column(const Tensor& y_true, const char* fn) {
    std::vector<double> v(y_true.rows);
    for (size_t i = 0; i < y_true.rows; ++i) v[i] = y_true(i, 0);
    (void)fn;
    return v;
}

void require_finite(const std::vector<double>& v, const char* what, const char* fn) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            throw std::invalid_argument(std::string(fn) + " requires finite " + what +
                                        " (non-finite at row " + std::to_string(i) + ")");
        }
    }
}

void require_probabilities(const Tensor& y_proba, const char* fn) {
    for (size_t i = 0; i < y_proba.rows; ++i) {
        for (size_t j = 0; j < y_proba.cols; ++j) {
            double p = y_proba(i, j);
            if (!std::isfinite(p)) {
                throw std::invalid_argument(std::string(fn) + " requires finite probabilities");
            }
            if (p < 0.0 || p > 1.0) {
                throw std::invalid_argument(std::string(fn) +
                                            " requires probabilities in [0,1] (out of range at (" +
                                            std::to_string(i) + "," + std::to_string(j) + "))");
            }
        }
    }
}

size_t checked_class_index(double value, size_t num_classes, const char* fn, size_t row) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(fn) + " contains non-finite class index at row " +
                                    std::to_string(row));
    }
    double rounded = std::round(value);
    if (std::fabs(value - rounded) > 1e-12) {
        throw std::invalid_argument(std::string(fn) + " contains non-integer class index at row " +
                                    std::to_string(row));
    }
    if (rounded < 0.0) {
        throw std::invalid_argument(std::string(fn) + " contains negative class index at row " +
                                    std::to_string(row));
    }
    size_t idx = static_cast<size_t>(rounded);
    if (idx >= num_classes) {
        throw std::invalid_argument(std::string(fn) + " class index " + std::to_string(idx) +
                                    " out of range for " + std::to_string(num_classes) +
                                    " classes at row " + std::to_string(row));
    }
    return idx;
}

// A binary label must be exactly 0 or 1. Unlike a multiclass index there is no
// column to bound it against, so it is validated on its own terms.
double checked_binary_label(double value, const char* fn, size_t row) {
    if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires binary labels in {0,1} (got " +
                                    std::to_string(value) + " at row " + std::to_string(row) + ")");
    }
    return value;
}

// One-vs-rest expansion: class c's problem is "is the label c?" scored by
// column c. Returns the per-class binary vectors plus each class's support.
struct OvrDecomposition {
    std::vector<std::vector<double>> binary_labels;
    std::vector<std::vector<double>> binary_scores;
    std::vector<size_t> supports;
    size_t n_classes = 0;
};

OvrDecomposition make_ovr(const Tensor& y_true_index, const Tensor& y_proba,
                          const char* fn) {
    if (y_proba.cols < 2) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires a multiclass probability matrix with >= 2 columns");
    }
    if (y_true_index.rows != y_proba.rows) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires y_true and y_proba to have the same number of rows");
    }
    if (y_true_index.cols != 1) {
        throw std::invalid_argument(std::string(fn) + " requires y_true to be a (N,1) column");
    }
    require_probabilities(y_proba, fn);
    // sklearn checks allclose(1, y_score.sum(axis=1)) for multiclass ROC-AUC
    // (_ranking.py:801-805) -- probabilities that do not sum to 1 are rejected.
    for (size_t i = 0; i < y_proba.rows; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < y_proba.cols; ++j) s += y_proba(i, j);
        if (std::fabs(s - 1.0) > 1e-8) {
            throw std::invalid_argument(std::string(fn) +
                                        " requires each probability row to sum to 1 (row " +
                                        std::to_string(i) + " sums to " + std::to_string(s) + ")");
        }
    }

    OvrDecomposition ovr;
    ovr.n_classes = y_proba.cols;
    ovr.binary_labels.assign(ovr.n_classes, {});
    ovr.binary_scores.assign(ovr.n_classes, {});
    ovr.supports.assign(ovr.n_classes, 0);
    for (size_t c = 0; c < ovr.n_classes; ++c) {
        ovr.binary_labels[c].reserve(y_proba.rows);
        ovr.binary_scores[c].reserve(y_proba.rows);
    }
    for (size_t i = 0; i < y_proba.rows; ++i) {
        size_t c = checked_class_index(y_true_index(i, 0), ovr.n_classes, fn, i);
        ++ovr.supports[c];
        for (size_t k = 0; k < ovr.n_classes; ++k) {
            ovr.binary_labels[k].push_back(c == k ? 1.0 : 0.0);
            ovr.binary_scores[k].push_back(y_proba(i, k));
        }
    }
    return ovr;
}

double average_ovr(const std::vector<double>& per_class, const std::vector<size_t>& supports,
                   MulticlassAveraging averaging, const char* fn) {
    double total = 0.0;
    for (double v : per_class) {
        if (!std::isfinite(v)) {
            // A class absent from y_true has no defined one-vs-rest AUC. The
            // reference averages over the classes that ARE present
            // (_average_binary_score skips empty columns); do the same rather
            // than propagating NaN into a number the caller will act on.
            continue;
        }
        total += v;
    }
    size_t counted = 0;
    for (double v : per_class) {
        if (std::isfinite(v)) ++counted;
    }
    if (counted == 0) {
        throw std::invalid_argument(std::string(fn) + " is undefined: no class is present");
    }
    if (averaging == MulticlassAveraging::Macro) {
        return total / static_cast<double>(counted);
    }
    double support_total = 0.0;
    double weighted = 0.0;
    for (size_t c = 0; c < per_class.size(); ++c) {
        if (!std::isfinite(per_class[c])) continue;
        weighted += static_cast<double>(supports[c]) * per_class[c];
        support_total += static_cast<double>(supports[c]);
    }
    return weighted / support_total;
}

// sklearn's `drop_intermediate` filter for roc_curve (_ranking.py:1335-1352),
// which is ON by default for ROC (and OFF for the PR curve). It keeps the
// first and last point, plus any point that is NOT on a straight line between
// its neighbours -- the test is numpy's `diff(x, n=2)`, i.e. the literal
// SECOND DERIVATIVE x[i] - 2*x[i+1] + x[i+2], NOT x[i+2] - x[i]. Both fps and
// tps are tested. Getting this wrong changes the CURVE's length but never its
// area (the dropped points are collinear), so only a test that asserts the
// point count or the threshold values can catch it.
//
// Applied to the RAW sweep, before the (0,0)/+inf origin is prepended -- the
// origin is not part of the filtered array in the reference, and filtering it
// in would misjudge the first real threshold (whose left neighbour would be
// the origin's zeros rather than the previous threshold).
void drop_collinear(BinaryClfCurve& c) {
    const size_t n = c.fps.size();
    if (n <= 2) return;
    std::vector<bool> keep(n, false);
    keep[0] = true;
    keep[n - 1] = true;
    for (size_t i = 1; i + 1 < n; ++i) {
        // np.diff(x, 2)[j] == x[j] - 2*x[j+1] + x[j+2], and the reference
        // concatenates [True, diff(x,2), True] -- so the second derivative
        // governing index i is centred on i: x[i-1] - 2*x[i] + x[i+1].
        const double d2_fps = c.fps[i - 1] - 2.0 * c.fps[i] + c.fps[i + 1];
        const double d2_tps = c.tps[i - 1] - 2.0 * c.tps[i] + c.tps[i + 1];
        if (d2_fps != 0.0 || d2_tps != 0.0) keep[i] = true;
    }
    BinaryClfCurve out;
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        out.fps.push_back(c.fps[i]);
        out.tps.push_back(c.tps[i]);
        out.thresholds.push_back(c.thresholds[i]);
    }
    c = out;
}

// The library's shared threshold-sweep, exposed so roc_curve / AP / the OVR
// overloads all consume one code path.
BinaryClfCurve sweep(const Tensor& y_true, const Tensor& y_score, double pos_label,
                     const char* fn) {
    require_binary_columns(y_true, y_score, fn);
    std::vector<double> labels = label_column(y_true, fn);
    std::vector<double> scores(y_score.rows);
    for (size_t i = 0; i < y_score.rows; ++i) scores[i] = y_score(i, 0);
    require_finite(labels, "labels", fn);
    require_finite(scores, "scores", fn);
    std::vector<double> binary(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        binary[i] = (labels[i] == pos_label) ? 1.0 : 0.0;
    }
    return binary_clf_curve(binary, scores);
}

// Same two metrics on plain vectors, for the one-vs-rest expansion (which
// builds std::vector columns, not Tensors). Returns NaN for an undefined
// one-vs-rest problem rather than throwing, so a class absent from y_true can
// be skipped by average_ovr.
double binary_roc_auc(const std::vector<double>& binary_labels,
                      const std::vector<double>& scores) {
    BinaryClfCurve c = binary_clf_curve(binary_labels, scores);
    if (c.fps.back() <= 0.0 || c.tps.back() <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Prepend the (0,0) origin, matching roc_curve, before integrating.
    double area = 0.0;
    double prev_x = 0.0, prev_y = 0.0;
    for (size_t i = 0; i < c.fps.size(); ++i) {
        const double x = c.fps[i] / c.fps.back();
        const double y = c.tps[i] / c.tps.back();
        area += (x - prev_x) * (y + prev_y) / 2.0;
        prev_x = x;
        prev_y = y;
    }
    return area;
}

double binary_average_precision(const std::vector<double>& binary_labels,
                                const std::vector<double>& scores) {
    BinaryClfCurve c = binary_clf_curve(binary_labels, scores);
    const double n_pos = c.tps.back();
    if (n_pos <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> precision(c.fps.size());
    std::vector<double> recall(c.fps.size());
    for (size_t i = 0; i < c.fps.size(); ++i) {
        const double denom = c.tps[i] + c.fps[i];
        precision[i] = denom != 0.0 ? c.tps[i] / denom : 0.0;
        recall[i] = c.tps[i] / n_pos;
    }
    double sum = 0.0;
    for (size_t i = 0; i + 1 < recall.size(); ++i) {
        sum += (recall[i + 1] - recall[i]) * precision[i];
    }
    return std::max(0.0, -sum);
}

}  // namespace

double auc_trapezoid(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("auc_trapezoid requires x and y of equal length");
    }
    if (x.size() < 2) {
        throw std::invalid_argument("auc_trapezoid requires at least 2 points");
    }
    // sklearn's auc() (_ranking.py:101-109): an entirely non-increasing x is
    // integrated with direction = -1; a MIXED-direction x is an error.
    bool any_decreasing = false;
    bool any_increasing = false;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] < x[i - 1]) any_decreasing = true;
        if (x[i] > x[i - 1]) any_increasing = true;
    }
    if (any_decreasing && any_increasing) {
        throw std::invalid_argument("auc_trapezoid requires x to be monotonic");
    }
    const double direction = any_decreasing ? -1.0 : 1.0;
    double area = 0.0;
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        area += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0;
    }
    return direction * area;
}

ROCCurve roc_curve(const Tensor& y_true, const Tensor& y_score, double pos_label) {
    const char* fn = "roc_curve";
    BinaryClfCurve c = sweep(y_true, y_score, pos_label, fn);
    // ROC drops collinear points by default; the PR curve does NOT. Applied to
    // the raw sweep, before the origin is prepended below.
    drop_collinear(c);

    ROCCurve out;
    // The leading (0,0) point at threshold +inf: the operating point that
    // predicts the negative class for every sample (_ranking.py:1354-1358).
    out.fpr.push_back(0.0);
    out.tpr.push_back(0.0);
    out.thresholds.push_back(std::numeric_limits<double>::infinity());
    const double n_neg = c.fps.back();
    const double n_pos = c.tps.back();
    for (size_t i = 0; i < c.fps.size(); ++i) {
        // A single-class input makes one of these ratios meaningless; the
        // reference fills the whole array with NaN (_ranking.py:1360-1372).
        out.fpr.push_back(n_neg > 0.0 ? c.fps[i] / n_neg
                                      : std::numeric_limits<double>::quiet_NaN());
        out.tpr.push_back(n_pos > 0.0 ? c.tps[i] / n_pos
                                      : std::numeric_limits<double>::quiet_NaN());
        out.thresholds.push_back(c.thresholds[i]);
    }
    return out;
}

double roc_auc_score(const Tensor& y_true, const Tensor& y_score, double pos_label) {
    ROCCurve c = roc_curve(y_true, y_score, pos_label);
    bool all_finite = true;
    for (size_t i = 0; i < c.fpr.size(); ++i) {
        if (!std::isfinite(c.fpr[i]) || !std::isfinite(c.tpr[i])) all_finite = false;
    }
    // Undefined rather than 0.5 or 1.0 -- see the header.
    if (!all_finite) return std::numeric_limits<double>::quiet_NaN();
    return auc_trapezoid(c.fpr, c.tpr);
}

PrecisionRecallCurve precision_recall_curve(const Tensor& y_true, const Tensor& y_score,
                                            double pos_label) {
    const char* fn = "precision_recall_curve";
    BinaryClfCurve c = sweep(y_true, y_score, pos_label, fn);

    std::vector<double> precision(c.fps.size());
    std::vector<double> recall(c.fps.size());
    for (size_t i = 0; i < c.fps.size(); ++i) {
        const double denom = c.tps[i] + c.fps[i];
        // 0, not NaN, when nothing has been predicted positive at this
        // threshold (_ranking.py:1189-1190).
        precision[i] = denom != 0.0 ? c.tps[i] / denom : 0.0;
        // With no positives at all the reference sets recall to 1 everywhere.
        recall[i] = c.tps.back() > 0.0 ? c.tps[i] / c.tps.back() : 1.0;
    }

    PrecisionRecallCurve out;
    // Returned recall-DECREASING, with the (1.0, 0.0) sentinel appended
    // (_ranking.py:1202-1210).
    for (size_t k = c.fps.size(); k-- > 0;) {
        out.precision.push_back(precision[k]);
        out.recall.push_back(recall[k]);
        out.thresholds.push_back(c.thresholds[k]);
    }
    out.precision.push_back(1.0);
    out.recall.push_back(0.0);
    return out;
}

double average_precision_score(const Tensor& y_true, const Tensor& y_score,
                               double pos_label) {
    PrecisionRecallCurve c = precision_recall_curve(y_true, y_score, pos_label);
    // STEP integral, not the trapezoid (_ranking.py:241-257):
    //   -sum(d_recall * precision[:-1])
    // On ordinary data these differ -- see
    // test_ap_is_not_trapezoid_area_under_pr.
    double sum = 0.0;
    for (size_t i = 0; i + 1 < c.recall.size(); ++i) {
        sum += (c.recall[i + 1] - c.recall[i]) * c.precision[i];
    }
    // The reference clamps because the sum can come out as -0.0.
    return std::max(0.0, -sum);
}

double brier_score_loss(const Tensor& y_true_index, const Tensor& y_proba, BrierScale scale) {
    const char* fn = "brier_score_loss";
    if (y_true_index.rows != y_proba.rows) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires y_true and y_proba to have the same number of rows");
    }
    if (y_true_index.cols != 1 || y_proba.cols == 0) {
        throw std::invalid_argument(std::string(fn) + " requires (N,1) labels and a (N,C) matrix");
    }
    require_probabilities(y_proba, fn);

    double sum = 0.0;
    if (y_proba.cols == 1) {
        // (N,1) is the BINARY form: the single column holds P(class 1), and
        // the reference expands it to the two-column form (1-p, p) before
        // scoring (_classification.py:3621-3660). The squared error therefore
        // runs over BOTH columns: a label-0 row contributes
        // (0 - (1-p))^2 + (1 - p)^2 = 2*p^2, and a label-1 row contributes
        // 2*(1-p)^2. Summing only the target column gives the right answer up
        // to a factor of 2 -- and "auto" then halves it back, which is exactly
        // why the halving rule must be applied to the two-column sum.
        for (size_t i = 0; i < y_proba.rows; ++i) {
            const double target = checked_binary_label(y_true_index(i, 0), fn, i);
            const double p = y_proba(i, 0);
            // probs are (1-p, p); one-hot is (1,0) or (0,1)
            const double d0 = (1.0 - target) - (1.0 - p);
            const double d1 = target - p;
            sum += d0 * d0 + d1 * d1;
        }
    } else {
        for (size_t i = 0; i < y_proba.rows; ++i) {
            size_t c = checked_class_index(y_true_index(i, 0), y_proba.cols, fn, i);
            for (size_t j = 0; j < y_proba.cols; ++j) {
                const double target = (j == c) ? 1.0 : 0.0;
                const double d = target - y_proba(i, j);
                sum += d * d;
            }
        }
    }
    double value = sum / static_cast<double>(y_proba.rows);

    // "auto" halves when the probability matrix has FEWER THAN 3 COLUMNS
    // (_classification.py:3849-3852). Both the (N,1) and (N,2) forms count as
    // < 3: the (N,1) binary case above already accumulates the equivalent
    // two-column sum, so its halved result (0.0375) matches the reference
    // exactly. A 3-column matrix is not halved.
    bool halve;
    switch (scale) {
        case BrierScale::Half: halve = true; break;
        case BrierScale::NoHalf: halve = false; break;
        case BrierScale::Auto:
        default: halve = (y_proba.cols < 3); break;
    }
    return halve ? value * 0.5 : value;
}

double log_loss(const Tensor& y_true_index, const Tensor& y_proba) {
    const char* fn = "log_loss";
    if (y_true_index.rows != y_proba.rows) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires y_true and y_proba to have the same number of rows");
    }
    if (y_true_index.cols != 1 || y_proba.cols == 0) {
        throw std::invalid_argument(std::string(fn) + " requires (N,1) labels and a (N,C) matrix");
    }
    require_probabilities(y_proba, fn);

    // eps is DBL_EPSILON (_classification.py:3449-3450), not 1e-15 and not 1e-7:
    // the clip is what makes a hard prediction of 1.0 return -log(DBL_EPSILON)
    // instead of -inf.
    const double eps = std::numeric_limits<double>::epsilon();
    double sum = 0.0;
    if (y_proba.cols == 1) {
        // Binary (N,1) form: the column is P(class 1), and only the true label
        // contributes (_classification.py:3452's _xlogy). A label of 1 therefore
        // reads the column, and a label of 0 reads (1 - column) -- NOT
        // log(1 - column) of a second, absent column.
        for (size_t i = 0; i < y_proba.rows; ++i) {
            const double target = checked_binary_label(y_true_index(i, 0), fn, i);
            double p = y_proba(i, 0);
            if (target == 0.0) p = 1.0 - p;
            p = std::min(std::max(p, eps), 1.0 - eps);
            sum += -std::log(p);
        }
    } else {
        for (size_t i = 0; i < y_proba.rows; ++i) {
            size_t c = checked_class_index(y_true_index(i, 0), y_proba.cols, fn, i);
            double p = y_proba(i, c);
            p = std::min(std::max(p, eps), 1.0 - eps);
            sum += -std::log(p);
        }
    }
    return sum / static_cast<double>(y_proba.rows);
}

CalibrationCurve calibration_curve(const Tensor& y_true, const Tensor& y_prob, size_t n_bins,
                                   double pos_label, CalibrationStrategy strategy) {
    const char* fn = "calibration_curve";
    require_binary_columns(y_true, y_prob, fn);
    if (n_bins == 0) {
        throw std::invalid_argument(std::string(fn) + " requires n_bins >= 1");
    }
    std::vector<double> labels = label_column(y_true, fn);
    std::vector<double> probs(y_prob.rows);
    for (size_t i = 0; i < y_prob.rows; ++i) probs[i] = y_prob(i, 0);
    require_finite(probs, "probabilities", fn);
    require_probabilities(y_prob, fn);

    // Bin edges. Uniform = equal width over [0,1]; Quantile = equal mass, with
    // the quantile linearly interpolated (calibration.py:1442-1443).
    std::vector<double> edges;
    if (strategy == CalibrationStrategy::Uniform) {
        edges.resize(n_bins + 1);
        for (size_t i = 0; i <= n_bins; ++i) {
            edges[i] = static_cast<double>(i) / static_cast<double>(n_bins);
        }
    } else {
        std::vector<double> sorted(probs);
        std::sort(sorted.begin(), sorted.end());
        edges.resize(n_bins + 1);
        for (size_t i = 0; i <= n_bins; ++i) {
            const double q = static_cast<double>(i) / static_cast<double>(n_bins);
            const double pos = q * static_cast<double>(sorted.size() - 1);
            const size_t lo = static_cast<size_t>(std::floor(pos));
            const size_t hi = std::min(lo + 1, sorted.size() - 1);
            edges[i] = sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
        }
    }

    // Interior edges only. Assignment is LEFT-closed: a value exactly on an
    // interior edge goes into the LOWER bin, matching np.searchsorted's default
    // side='left' (calibration.py:1455).
    const std::vector<double> interior(edges.begin() + 1, edges.end() - 1);
    std::vector<double> bin_sum(n_bins, 0.0);
    std::vector<double> bin_true(n_bins, 0.0);
    std::vector<size_t> bin_count(n_bins, 0);
    for (size_t i = 0; i < probs.size(); ++i) {
        size_t lo = 0, hi = interior.size();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (interior[mid] < probs[i]) lo = mid + 1;
            else hi = mid;
        }
        bin_count[lo] += 1;
        bin_sum[lo] += probs[i];
        bin_true[lo] += (labels[i] == pos_label) ? 1.0 : 0.0;
    }

    // EMPTY BINS ARE DROPPED (calibration.py:1458-1462) -- a caller can then
    // tell "no samples here" from "this bin's mean is 0" by the output length.
    CalibrationCurve out;
    for (size_t k = 0; k < n_bins; ++k) {
        if (bin_count[k] == 0) continue;
        out.prob_true.push_back(bin_true[k] / static_cast<double>(bin_count[k]));
        out.prob_pred.push_back(bin_sum[k] / static_cast<double>(bin_count[k]));
        out.counts.push_back(bin_count[k]);
    }
    return out;
}

double expected_calibration_error(const Tensor& y_true, const Tensor& y_prob, size_t n_bins,
                                  double pos_label, CalibrationStrategy strategy) {
    CalibrationCurve c = calibration_curve(y_true, y_prob, n_bins, pos_label, strategy);
    const double n = static_cast<double>(y_true.rows);
    double ece = 0.0;
    for (size_t k = 0; k < c.counts.size(); ++k) {
        const double w = static_cast<double>(c.counts[k]) / n;
        ece += w * std::fabs(c.prob_true[k] - c.prob_pred[k]);
    }
    return ece;
}

double roc_auc_score(const Tensor& y_true_index, const Tensor& y_proba,
                     MulticlassAveraging averaging) {
    const char* fn = "roc_auc_score";
    OvrDecomposition ovr = make_ovr(y_true_index, y_proba, fn);
    std::vector<double> per_class(ovr.n_classes);
    for (size_t c = 0; c < ovr.n_classes; ++c) {
        per_class[c] = binary_roc_auc(ovr.binary_labels[c], ovr.binary_scores[c]);
    }
    return average_ovr(per_class, ovr.supports, averaging, fn);
}

double average_precision_score(const Tensor& y_true_index, const Tensor& y_proba,
                               MulticlassAveraging averaging) {
    const char* fn = "average_precision_score";
    OvrDecomposition ovr = make_ovr(y_true_index, y_proba, fn);
    std::vector<double> per_class(ovr.n_classes);
    for (size_t c = 0; c < ovr.n_classes; ++c) {
        per_class[c] = binary_average_precision(ovr.binary_labels[c], ovr.binary_scores[c]);
    }
    return average_ovr(per_class, ovr.supports, averaging, fn);
}

// ===========================================================================
// Calibration diagnostics beyond ECE
// ===========================================================================

namespace {

// Split `probs` into at most n_bins EQUAL-COUNT ranges ordered ascending by
// probability (Nixon et al., arXiv:1904.01685 Sec. 5.2-5.3).
//
// Sec. 5.2: "We chose an adaptive scheme which spaces the bin intervals so
// that each contains an equal number of predictions." Sec. 5.3: "Calibration
// range r defined by the floor(N/R)th index of the sorted and thresholded
// predictions."
//
// floor(N/R) does not necessarily divide N, so the ranges are NOT equal in
// size. The boundary index is therefore rounded UP: with N=3, R=2 the
// `ceil` reading gives ranges of sizes 2 and 1, matching Sec. 5.2's stated
// goal ("each contains an equal number of predictions"), while a literal
// `floor` gives sizes 1 and 2 -- the same multiset, assigned to the opposite
// ends of the confidence range, which silently changes the metric (0.175 vs
// 0.200 on the fixture in test 4). Rounding up is the reading that keeps the
// paper's stated property; the literal floor would put the smallest
// confidence in its own range and leave the largest confidence crowded
// together, which is the opposite of the motivation.
//
// This is the one place this implementation knowingly generalises past the
// paper's literal `floor(N/R)` wording, and it does so to satisfy the
// property the same section states. test_ace_differs_from_count_weighted
// pins the resulting value.
//
// std::stable_sort keeps equal-probability samples in input order, so the
// assignment cannot vary with sort implementation.
std::vector<size_t> equal_count_bins(const std::vector<double>& probs, size_t n_bins,
                                     std::vector<size_t>& bin_size) {
    const size_t n = probs.size();
    bin_size.assign(n_bins, 0);
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&probs](size_t a, size_t b) { return probs[a] < probs[b]; });

    const size_t per_bin = (n + n_bins - 1) / n_bins;  // ceil(N/R), min 1
    std::vector<size_t> bin_of(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t b = (per_bin == 0) ? 0 : i / per_bin;
        if (b >= n_bins) b = n_bins - 1;  // the remainder lands in the last range
        bin_of[order[i]] = b;
        ++bin_size[b];
    }
    return bin_of;
}

// Shared engine for ACE and TACE. `threshold` drops predictions at or below it
// BEFORE binning. y_true/y_prob are (N,1) binary columns.
double adaptive_calibration_error_impl(const Tensor& y_true, const Tensor& y_prob,
                                       double pos_label, size_t n_bins,
                                       double threshold, const char* fn) {
    require_binary_columns(y_true, y_prob, fn);
    if (n_bins == 0) {
        throw std::invalid_argument(std::string(fn) + " requires n_bins >= 1");
    }
    if (!(threshold >= 0.0) || !(threshold <= 1.0)) {
        throw std::invalid_argument(std::string(fn) + " requires threshold within [0, 1]");
    }
    const std::vector<double> labels = label_column(y_true, fn);
    std::vector<double> probs(y_prob.rows);
    for (size_t i = 0; i < y_prob.rows; ++i) probs[i] = y_prob(i, 0);
    require_finite(probs, "probabilities", fn);
    require_probabilities(y_prob, fn);

    // Thresholding (Nixon Sec. 5.3): keep only predictions AT OR ABOVE the
    // threshold. The comparison is `>=`, NOT `>` -- with a strict `>` a
    // prediction of exactly 1.0 is dropped by a threshold of 1.0, and the
    // single most confident prediction the model can make would silently
    // vanish from its own calibration score. `>=` makes threshold 0 keep
    // every prediction (identity) and threshold 1 keep only the certain ones,
    // which is what both edges mean. Test 5 pins the boundary explicitly.
    std::vector<size_t> keep;
    keep.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] >= threshold) keep.push_back(i);
    }
    if (keep.empty()) return 0.0;

    std::vector<double> kp;
    kp.reserve(keep.size());
    for (size_t i : keep) kp.push_back(probs[i]);
    std::vector<size_t> bin_size;
    const std::vector<size_t> bin_of = equal_count_bins(kp, n_bins, bin_size);

    // UNIFORM 1/R average over the ranges that actually exist. Nixon writes
    // 1/(K*R); a range holding no kept sample has no acc/conf pair to
    // contribute, so the mean is taken over the non-empty ones rather than
    // silently charging the metric for an empty bin.
    double total = 0.0;
    size_t nonempty = 0;
    for (size_t r = 0; r < n_bins; ++r) {
        if (bin_size[r] == 0) continue;
        ++nonempty;
        double sum_p = 0.0, hits = 0.0;
        for (size_t s = 0; s < keep.size(); ++s) {
            if (bin_of[s] != r) continue;
            sum_p += kp[s];
            hits += (labels[keep[s]] == pos_label) ? 1.0 : 0.0;
        }
        const double count = static_cast<double>(bin_size[r]);
        total += std::fabs(hits / count - sum_p / count);
    }
    return total / static_cast<double>(nonempty);
}

}  // namespace

double adaptive_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                  double pos_label, size_t n_bins, double threshold) {
    return adaptive_calibration_error_impl(y_true, y_prob, pos_label, n_bins, threshold,
                                           "adaptive_calibration_error");
}

double thresholded_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                     double pos_label, size_t n_bins, double threshold) {
    return adaptive_calibration_error_impl(y_true, y_prob, pos_label, n_bins, threshold,
                                           "thresholded_calibration_error");
}

std::vector<double> classwise_ece_per_class(const Tensor& y_true_index,
                                            const Tensor& y_proba, size_t n_bins) {
    const char* fn = "classwise_ece_per_class";
    if (n_bins == 0) {
        throw std::invalid_argument(std::string(fn) + " requires n_bins >= 1");
    }
    const OvrDecomposition ovr = make_ovr(y_true_index, y_proba, fn);
    const double n = static_cast<double>(y_proba.rows);

    std::vector<double> per_class(ovr.n_classes, 0.0);
    for (size_t k = 0; k < ovr.n_classes; ++k) {
        // Class k's problem is binary: score p_k(x), label = (Y == k). The
        // binning reuses the SHIPPED equal-width calibration_curve, and the
        // weight is the SAMPLED |B|/n -- NOT ACE's uniform 1/R. That
        // asymmetry is the difference between this metric and ACE.
        const std::vector<double>& scores = ovr.binary_scores[k];
        const std::vector<double>& labels = ovr.binary_labels[k];
        Tensor s(static_cast<size_t>(scores.size()), 1);
        Tensor y(static_cast<size_t>(labels.size()), 1);
        for (size_t i = 0; i < scores.size(); ++i) {
            s(i, 0) = scores[i];
            y(i, 0) = labels[i];
        }
        const CalibrationCurve c =
            calibration_curve(y, s, n_bins, 1.0, CalibrationStrategy::Uniform);
        double acc = 0.0;
        for (size_t b = 0; b < c.counts.size(); ++b) {
            const double w = static_cast<double>(c.counts[b]) / n;
            acc += w * std::fabs(c.prob_true[b] - c.prob_pred[b]);
        }
        per_class[k] = acc;
    }
    return per_class;
}

double classwise_ece(const Tensor& y_true_index, const Tensor& y_proba, size_t n_bins) {
    const std::vector<double> per = classwise_ece_per_class(y_true_index, y_proba, n_bins);
    double sum = 0.0;
    for (double v : per) sum += v;
    return sum / static_cast<double>(per.size());
}

double classwise_ece(const Tensor& y_true_index, const Tensor& y_proba,
                     size_t n_bins, MulticlassAveraging averaging) {
    const char* fn = "classwise_ece";
    if (n_bins == 0) {
        throw std::invalid_argument(std::string(fn) + " requires n_bins >= 1");
    }
    const std::vector<double> per = classwise_ece_per_class(y_true_index, y_proba, n_bins);
    if (averaging == MulticlassAveraging::Macro) {
        double sum = 0.0;
        for (double v : per) sum += v;
        return sum / static_cast<double>(per.size());
    }
    // Support-weighted (the w_k of Kull's marginal CE, Eq. 3). A class with
    // NO instances has an undefined score -- identically 0, which would drag
    // a mean down without meaning anything -- so it is excluded from both
    // the sum and the weight.
    const OvrDecomposition ovr = make_ovr(y_true_index, y_proba, fn);
    double sum = 0.0, weight = 0.0;
    for (size_t k = 0; k < per.size(); ++k) {
        if (ovr.supports[k] == 0) continue;
        sum += static_cast<double>(ovr.supports[k]) * per[k];
        weight += static_cast<double>(ovr.supports[k]);
    }
    if (weight == 0.0) return 0.0;
    return sum / weight;
}
