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

#endif
