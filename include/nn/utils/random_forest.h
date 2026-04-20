#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "../core/tensor.h"
#include <vector>
#include <random>

// Random Forest — bagging + random feature subset per split.
// Each tree is trained on a bootstrap sample + random feature selection.
// Prediction: majority vote (classification) or mean (regression).
class RandomForest {
public:
    RandomForest(size_t n_trees = 50,
                  size_t max_depth = 10,
                  size_t min_samples_leaf = 5,
                  size_t max_features = 0, // 0 = sqrt(n_features)
                  bool regression = false);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    double score(const Tensor& X, const Tensor& y) const;

private:
    struct TreeNode {
        size_t feature = 0;
        double threshold = 0.0;
        double value = 0.0; // leaf prediction
        bool is_leaf = false;
        TreeNode *left = nullptr, *right = nullptr;
    };

    struct Tree {
        TreeNode root;
        bool regression;
        double fit_node(TreeNode& node, const Tensor& X, const Tensor& y,
                         const std::vector<size_t>& indices,
                         size_t depth, size_t max_depth,
                         size_t min_samples_leaf, size_t max_features,
                         std::mt19937& rng);
        double predict_node(const TreeNode& node, const Tensor& X, size_t row) const;
        void free_node(TreeNode& node);
    };

    size_t n_trees_, max_depth_, min_samples_leaf_, max_features_;
    bool regression_;
    std::vector<Tree> trees_;
};

#endif