#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H

#include "../core/tensor.h"
#include <vector>
#include <random>

// Isolation Forest — anomaly detection via random recursive partitions.
// Anomaly score = 2^{-(avg_path_depth / c(n))}
// where c(n) = 2*ln(n-1) + γ - 2*(n-1)/n (harmonic number approximation).
class IsolationForest {
public:
    // n_trees: number of trees, max_depth: max tree depth,
    // contamination: expected fraction of anomalies (0-1), seed: RNG seed
    IsolationForest(size_t n_trees = 100,
                      size_t max_depth = 8,
                      double contamination = 0.0,
                      unsigned int seed = 42);

    void fit(const Tensor& X);
    // Returns anomaly scores (higher = more anomalous), or binary predictions
    Tensor anomaly_score(const Tensor& X) const;
    Tensor predict(const Tensor& X) const; // -1=normal, +1=anomaly

private:
    struct ITreeNode {
        size_t feature = 0;
        double threshold = 0.0;
        bool is_leaf = false;
        double size = 0.0; // number of data points at this node
        ITreeNode *left = nullptr, *right = nullptr;
    };

    struct ITree {
        ITreeNode root;
        size_t max_depth;
        double fit_node(ITreeNode& node, const Tensor& X,
                        const std::vector<size_t>& indices,
                        size_t depth, size_t max_depth,
                        std::mt19937& rng);
        double path_length(const ITreeNode& node, const Tensor& x,
                             double e) const; // e = current path length
        void free_node(ITreeNode& node);
    };

    size_t n_trees_, max_depth_;
    double contamination_;
    double c_n_; // average path length adjustment constant
    std::vector<ITree> trees_;
    Tensor X_train_;
    double anomaly_threshold_;
};

#endif