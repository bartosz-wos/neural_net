#ifndef LIGHTGBM_STYLE_H
#define LIGHTGBM_STYLE_H

#include "../core/tensor.h"
#include <vector>

class HistogramBoosting {
public:
    HistogramBoosting(size_t n_estimators = 100,
                       double learning_rate = 0.1,
                       size_t max_bins = 255,
                       size_t max_depth = 0,
                       size_t num_leaves = 31,
                       size_t min_child_samples = 20,
                       bool regression = false);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

private:
    struct HistogramBin {
        double sum_grad = 0.0;
        double sum_hess = 0.0;
        size_t count = 0;
    };

    struct LeafNode {
        double leaf_value = 0.0;
        std::vector<HistogramBin> hist;
    };

    struct SplitCandidate {
        size_t feature;
        double threshold;
        double gain;
        double left_grad_sum, right_grad_sum;
        double left_hess_sum, right_hess_sum;
        size_t left_count, right_count;
    };

    size_t n_estimators_, max_bins_, max_depth_, num_leaves_, min_child_samples_;
    double learning_rate_;
    bool regression_;
    std::vector<double> base_predictions_;
    std::vector<LeafNode> leaves_;

    void build_histogram(const Tensor& X, const Tensor& grad,
                           std::vector<HistogramBin>& out) const;
    double compute_leaf_value(double grad_sum, double hess_sum) const;
};

#endif