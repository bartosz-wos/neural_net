#ifndef TABULAR_ENSEMBLE_H
#define TABULAR_ENSEMBLE_H

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <fstream>

// TabularDataset: loads CSV, normalizes features, train/test split
class TabularDataset {
public:
    struct Split {
        Tensor X_train, X_test, y_train, y_test;
    };

    TabularDataset(const std::string& csv_path,
                    bool has_header = true,
                    bool normalize = true,
                    double train_ratio = 0.8,
                    size_t label_col = 0,
                    char sep = ',');
    Split split() const;
    const Tensor& X() const { return X_; }
    const Tensor& y() const { return y_; }
    size_t n_features() const { return n_features_; }
    size_t n_samples() const { return n_samples_; }

private:
    Tensor X_, y_;
    size_t n_features_, n_samples_;
    double train_ratio_;
};

// Decision stump: 1-level decision tree as weak learner for boosting
class DecisionStump {
public:
    DecisionStump() : feature_idx_(0), threshold_(0.0), polarity_(1.0), alpha_(0.0) {}

    void fit(const Tensor& X, const Tensor& y, const std::vector<double>& weights = {});
    Tensor predict(const Tensor& X) const;
    double error(const Tensor& X, const Tensor& y, const std::vector<double>& weights = {}) const;
    double alpha() const { return alpha_; }
    size_t feature() const { return feature_idx_; }
    double threshold() const { return threshold_; }

private:
    size_t feature_idx_;
    double threshold_;
    double polarity_; // +1: predicts left for x < t, right otherwise; -1: flipped
    double alpha_;
    std::vector<double> predictions_;
};

#endif