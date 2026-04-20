#include "isolation_forest.h"
#include <algorithm>
#include <cmath>

static std::mt19937 iforest_rng(42);

double c_n(size_t n) {
    if (n <= 1) return 0.0;
    if (n == 2) return 1.0;
    return 2.0 * (std::log(static_cast<double>(n - 1)) + 0.5772156649) - 2.0 * (static_cast<double>(n - 1)) / n;
}

double IsolationForest::ITree::fit_node(ITreeNode& node,
                                          const Tensor& X,
                                          const std::vector<size_t>& indices,
                                          size_t depth, size_t max_depth,
                                          std::mt19937& rng) {
    node.size = indices.size();
    if (depth >= max_depth || indices.size() <= 1) {
        node.is_leaf = true;
        return path_length(node, X.get_row(0), 0.0); // dummy
    }

    size_t m = X.cols;
    size_t f = static_cast<size_t>(rng()) % m;
    double min_v = 1e100, max_v = -1e100;
    for (size_t i : indices) {
        min_v = std::min(min_v, X[i][f]);
        max_v = std::max(max_v, X[i][f]);
    }

    if (std::abs(max_v - min_v) < 1e-9) {
        node.is_leaf = true;
        return 0.0;
    }

    std::uniform_real_distribution<double> dist(min_v, max_v);
    node.feature = f;
    node.threshold = dist(rng);
    node.is_leaf = false;
    node.left = new ITreeNode();
    node.right = new ITreeNode();
    node.left->is_leaf = node.right->is_leaf = false;

    std::vector<size_t> left_idx, right_idx;
    for (size_t idx : indices)
        (X[idx][f] < node.threshold ? left_idx : right_idx).push_back(idx);

    fit_node(*node.left, X, left_idx, depth + 1, max_depth, rng);
    fit_node(*node.right, X, right_idx, depth + 1, max_depth, rng);
    return 0.0;
}

double IsolationForest::ITree::path_length(const ITreeNode& node,
                                            const Tensor& x,
                                            double e) const {
    if (node.is_leaf) {
        // c(size of leaf node's original sample)
        double c = c_n(static_cast<size_t>(node.size));
        return e + c;
    }
    if (x[0][node.feature] < node.threshold)
        return path_length(*node.left, x, e + 1.0);
    else
        return path_length(*node.right, x, e + 1.0);
}

void IsolationForest::ITree::free_node(ITreeNode& node) {
    if (!node.is_leaf) {
        if (node.left) { free_node(*node.left); delete node.left; }
        if (node.right) { free_node(*node.right); delete node.right; }
    }
}

IsolationForest::IsolationForest(size_t n_trees, size_t max_depth,
                                    double contamination, unsigned int seed)
    : n_trees_(n_trees), max_depth_(max_depth),
      contamination_(contamination), c_n_(0.0), anomaly_threshold_(0.0) {
    iforest_rng.seed(seed);
}

void IsolationForest::fit(const Tensor& X) {
    X_train_ = X;
    size_t n = X.rows;
    c_n_ = c_n(n);

    trees_.clear();
    for (size_t t = 0; t < n_trees_; ++t) {
        ITree tree;
        tree.max_depth = max_depth_;

        // Bootstrap sample
        std::vector<size_t> boot(n);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (size_t i = 0; i < n; ++i) boot[i] = dist(iforest_rng);

        tree.root.is_leaf = false;
        tree.fit_node(tree.root, X, boot, 0, max_depth_, iforest_rng);
        trees_.push_back(tree);
    }

    // Compute anomaly threshold from training scores if contamination > 0
    if (contamination_ > 0) {
        Tensor scores = anomaly_score(X);
        std::vector<double> sorted_scores(n);
        for (size_t i = 0; i < n; ++i) sorted_scores[i] = scores[i][0];
        std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<double>());
        size_t idx = static_cast<size_t>(contamination_ * n);
        anomaly_threshold_ = sorted_scores[std::min(idx, n - 1)];
    }
}

Tensor IsolationForest::anomaly_score(const Tensor& X) const {
    size_t n = X.rows;
    Tensor scores(n, 1);
    double denom = c_n_ + 1e-10;

    for (size_t i = 0; i < n; ++i) {
        double sum_pl = 0.0;
        for (const auto& tree : trees_) {
            sum_pl += tree.path_length(tree.root, X.get_row(i), 0.0);
        }
        double avg_pl = sum_pl / n_trees_;
        scores[i][0] = std::pow(2.0, -avg_pl / denom);
    }
    return scores;
}

Tensor IsolationForest::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor scores = anomaly_score(X);
    Tensor out(n, 1);
    double thresh = anomaly_threshold_ > 0 ? anomaly_threshold_ : 0.5;
    for (size_t i = 0; i < n; ++i)
        out[i][0] = (scores[i][0] >= thresh) ? 1.0 : -1.0;
    return out;
}