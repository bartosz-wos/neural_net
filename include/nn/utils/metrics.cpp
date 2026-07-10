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
