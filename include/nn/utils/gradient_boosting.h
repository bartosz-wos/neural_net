#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "../core/tensor.h"
#include "tabular_ensemble.h"
#include <vector>

// Generic Gradient Boosting — fits sequential learners against negative gradient of loss.
class GradientBoosting {
public:
    enum class Loss { MSE, CROSS_ENTROPY };

    GradientBoosting(size_t n_estimators = 100,
                     double learning_rate = 0.1,
                     size_t max_depth = 3,
                     Loss loss = Loss::MSE);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    double score(const Tensor& X, const Tensor& y) const;

private:
    struct TreeNode {
        size_t feature_idx;
        double threshold;
        double value; // leaf value or internal split value
        std::vector<TreeNode> left, right;
        bool is_leaf;
    };

    struct RegressionTree {
        std::vector<TreeNode> nodes;
        double fit(const Tensor& X, const Tensor& residuals);
        double predict(const Tensor& X) const;
        double compute_split_gini(const Tensor& X, const Tensor& y,
                                    size_t feature, double threshold,
                                    double& left_val, double& right_val) const;
    };

    size_t n_estimators_;
    double learning_rate_;
    size_t max_depth_;
    Loss loss_;
    std::vector<double> F0_; // initial predictions (global mean)
    std::vector<RegressionTree> trees_;
};

// XGBoost-style tree: gradient statistics with L1/L2 regularization
class XGBoostTree {
public:
    XGBoostTree(size_t max_depth = 3, double lambda = 1.0, double gamma = 0.0,
                  double min_child_weight = 1.0, double subsample = 1.0,
                  double colsample_bytree = 1.0);

    void fit(const Tensor& X, const Tensor& g, const Tensor& h);
    Tensor predict(const Tensor& X) const;

private:
    struct XGBNode {
        size_t feature;
        double threshold;
        double leaf_weight;
        double gain;
        bool is_leaf;
        XGBNode *left, *right;
    };

    void build(XGBNode& node, const Tensor& X,
               const std::vector<size_t>& indices,
               size_t depth);
    double compute_score(const std::vector<double>& G, const std::vector<double>& H) const;
    double compute_weight(const std::vector<double>& G, const std::vector<double>& H) const;

    size_t max_depth_;
    double lambda_, gamma_, min_child_weight_, subsample_, colsample_bytree_;
    XGBNode root_;
};

class XGBoostClassifier {
public:
    XGBoostClassifier(size_t n_estimators = 50,
                       size_t max_depth = 3,
                       double learning_rate = 0.1,
                       double lambda = 1.0,
                       double gamma = 0.0,
                       double subsample = 0.8,
                       double colsample_bytree = 0.8);

    void fit(const Tensor& X, const Tensor& y, size_t n_classes);
    Tensor predict_proba(const Tensor& X) const;
    Tensor predict(const Tensor& X) const;

private:
    size_t n_estimators_, max_depth_, n_classes_;
    double learning_rate_, lambda_, gamma_, subsample_, colsample_bytree_;
    std::vector<std::vector<XGBoostTree>> trees_per_class_;
    std::vector<double> F0_;
};

#endif