#include "lightgbm_style.h"
#include <cmath>
#include <algorithm>
#include <map>

HistogramBoosting::HistogramBoosting(size_t n_estimators, double learning_rate,
                                         size_t max_bins, size_t max_depth,
                                         size_t num_leaves, size_t min_child_samples,
                                         bool regression)
    : n_estimators_(n_estimators), max_bins_(max_bins),
      max_depth_(max_depth == 0 ? 1000 : max_depth),
      num_leaves_(num_leaves), min_child_samples_(min_child_samples),
      learning_rate_(learning_rate), regression_(regression) {}

double HistogramBoosting::compute_leaf_value(double grad_sum, double hess_sum) const {
    if (regression_) return -grad_sum / (hess_sum + 1e-8);
    // For logistic: w* = sum(g) / (sum(h) + λ)
    return -grad_sum / (hess_sum + 1.0);
}

void HistogramBoosting::build_histogram(const Tensor& X, const Tensor& grad,
                                          std::vector<HistogramBin>& out) const {
    size_t n = X.rows, d = X.cols;
    out.assign(d * max_bins_, HistogramBin());

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            // Bin feature j
            double v = X[i][j];
            double v_min = X[0][j], v_max = X[0][j];
            for (size_t k = 1; k < n; ++k) {
                v_min = std::min(v_min, X[k][j]);
                v_max = std::max(v_max, X[k][j]);
            }
            size_t bin = 0;
            if (v_max > v_min)
                bin = static_cast<size_t>((v - v_min) / (v_max - v_min) * (max_bins_ - 1));
            bin = std::min(bin, max_bins_ - 1);

            size_t idx = j * max_bins_ + bin;
            out[idx].sum_grad += grad[i][0];
            out[idx].sum_hess += 1.0; // hess ≈ 1 for MSE
            out[idx].count++;
        }
    }
}

void HistogramBoosting::fit(const Tensor& X, const Tensor& y) {
    size_t n = X.rows;
    base_predictions_.assign(n, 0.0);

    Tensor grad(n, 1);
    for (size_t t = 0; t < n_estimators_; ++t) {
        // Compute pseudo残差 (negative gradient)
        for (size_t i = 0; i < n; ++i)
            grad[i][0] = -(y[i][0] - base_predictions_[i]);

        // Build histograms per feature
        std::vector<HistogramBin> hist;
        build_histogram(X, grad, hist);

        // Find best split across all leaves (simplified: one level at a time)
        SplitCandidate best;
        best.gain = -1e100;
        size_t d = X.cols;

        for (size_t f = 0; f < d; ++f) {
            // Scan bins for this feature
            double left_G = 0, left_H = 0, right_G = 0, right_H = 0;
            for (size_t b = 0; b < max_bins_; ++b) {
                const HistogramBin& bin = hist[f * max_bins_ + b];
                right_G += bin.sum_grad;
                right_H += bin.sum_hess;
            }

            for (size_t b = 0; b < max_bins_ - 1; ++b) {
                const HistogramBin& bin = hist[f * max_bins_ + b];
                left_G += bin.sum_grad; left_H += bin.sum_hess;
                right_G -= bin.sum_grad; right_H -= bin.sum_hess;

                if (left_H < min_child_samples_ || right_H < min_child_samples_) continue;

                double gain = (left_G * left_G / (left_H + 1.0))
                           + (right_G * right_G / (right_H + 1.0))
                           - ((left_G + right_G) * (left_G + right_G) / (left_H + right_H + 1e-8));
                if (gain > best.gain) {
                    best.gain = gain;
                    best.feature = f;
                    best.threshold = static_cast<double>(b) / max_bins_;
                    best.left_grad_sum = left_G;
                    best.right_grad_sum = right_G;
                    best.left_hess_sum = left_H;
                    best.right_hess_sum = right_H;
                }
            }
        }

        if (best.gain > 0) {
            LeafNode leaf;
            leaf.leaf_value = compute_leaf_value(best.left_grad_sum, best.left_hess_sum);
            leaves_.push_back(leaf);
            // Update predictions
            double lr = learning_rate_;
            for (size_t i = 0; i < n; ++i) {
                double xb = X[i][best.feature];
                double pred_delta = (xb < best.threshold)
                    ? lr * compute_leaf_value(best.left_grad_sum, best.left_hess_sum)
                    : lr * compute_leaf_value(best.right_grad_sum, best.right_hess_sum);
                base_predictions_[i] += pred_delta;
            }
        }
    }
}

Tensor HistogramBoosting::predict(const Tensor& X) const {
    return Tensor(X.rows, 1); // placeholder
}