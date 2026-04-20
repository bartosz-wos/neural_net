#include "random_forest.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <map>

static std::mt19937 rng_(42);

double RandomForest::Tree::fit_node(TreeNode& node,
                                     const Tensor& X, const Tensor& y,
                                     const std::vector<size_t>& indices,
                                     size_t depth, size_t max_depth,
                                     size_t min_samples_leaf,
                                     size_t max_features,
                                     std::mt19937& rng) {
    if (indices.size() < min_samples_leaf || depth >= max_depth) {
        node.is_leaf = true;
        if (regression) {
            double sum = 0.0;
            for (size_t idx : indices) sum += y[idx][0];
            node.value = sum / indices.size();
        } else {
            // Majority vote
            std::map<double, size_t> counts;
            for (size_t idx : indices) {
                double label = y[idx][0];
                counts[label]++;
            }
            double best = 0;
            size_t best_c = 0;
            for (auto& p : counts)
                if (p.second > best_c) { best_c = p.second; best = p.first; }
            node.value = best;
        }
        return node.value;
    }

    size_t n = indices.size();
    size_t m = X.cols;
    size_t feat_to_try = (max_features == 0) ? static_cast<size_t>(std::sqrt(m)) : max_features;
    feat_to_try = std::min(feat_to_try, m);

    // Random feature subset
    std::vector<size_t> feat_pool(m);
    for (size_t i = 0; i < m; ++i) feat_pool[i] = i;
    std::shuffle(feat_pool.begin(), feat_pool.end(), rng);
    std::vector<size_t> feat_idx(feat_to_try);
    for (size_t i = 0; i < feat_to_try; ++i) feat_idx[i] = feat_pool[i];

    double best_score = regression ? 1e100 : 0.0;
    size_t best_f = 0;
    double best_t = 0.0;
    double best_left_val = 0.0, best_right_val = 0.0;

    for (size_t fi = 0; fi < feat_to_try; ++fi) {
        size_t f = feat_idx[fi];
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t idx : indices)
            sorted.emplace_back(X[idx][f], idx);
        std::sort(sorted.begin(), sorted.end());

        for (size_t t = 0; t + 1 < sorted.size(); ++t) {
            double thr = (sorted[t].first + sorted[t + 1].first) / 2.0;
            std::vector<size_t> left, right;
            for (size_t idx : indices)
                (X[idx][f] < thr ? left : right).push_back(idx);

            if (left.empty() || right.empty()) continue;

            if (regression) {
                double lv = 0, rv = 0;
                for (size_t i : left) lv += y[i][0];
                for (size_t i : right) rv += y[i][0];
                lv /= left.size(); rv /= right.size();
                double mse = 0.0;
                for (size_t i : left) { double r = y[i][0] - lv; mse += r*r; }
                for (size_t i : right) { double r = y[i][0] - rv; mse += r*r; }
                if (mse < best_score) {
                    best_score = mse;
                    best_f = f; best_t = thr;
                    best_left_val = lv; best_right_val = rv;
                }
            } else {
                // Information gain (classification)
                auto gini = [&](const std::vector<size_t>& idxs) {
                    std::map<double, size_t> c;
                    for (size_t i : idxs) c[y[i][0]]++;
                    double g = 1.0;
                    for (auto& p : c) {
                        double frac = static_cast<double>(p.second) / idxs.size();
                        g -= frac * frac;
                    }
                    return g;
                };
                double ig = 1.0;
                double w_left = left.size(), w_right = right.size();
                ig -= (w_left / n) * gini(left) + (w_right / n) * gini(right);
                if (ig > best_score) {
                    best_score = ig;
                    best_f = f; best_t = thr;
                    // compute majority labels
                    std::map<double, size_t> cl, cr;
                    for (size_t i : left) cl[y[i][0]]++;
                    for (size_t i : right) cr[y[i][0]]++;
                    double bl = 0, br = 0, bestl = 0, bestr = 0;
                    for (auto& p : cl) if (p.second > bestl) { bestl = p.second; bl = p.first; }
                    for (auto& p : cr) if (p.second > bestr) { bestr = p.second; br = p.first; }
                    best_left_val = bl; best_right_val = br;
                }
            }
        }
    }

    node.feature = best_f;
    node.threshold = best_t;
    node.is_leaf = false;
    node.left = new TreeNode();
    node.right = new TreeNode();
    node.left->is_leaf = node.right->is_leaf = false;

    std::vector<size_t> left_idx, right_idx;
    for (size_t idx : indices)
        (X[idx][best_f] < best_t ? left_idx : right_idx).push_back(idx);

    fit_node(*node.left, X, y, left_idx, depth + 1, max_depth,
              min_samples_leaf, max_features, rng);
    fit_node(*node.right, X, y, right_idx, depth + 1, max_depth,
              min_samples_leaf, max_features, rng);

    return node.value;
}

double RandomForest::Tree::predict_node(const TreeNode& node,
                                          const Tensor& X, size_t row) const {
    if (node.is_leaf) return node.value;
    return predict_node(X[row][node.feature] < node.threshold ? *node.left : *node.right,
                         X, row);
}

void RandomForest::Tree::free_node(TreeNode& node) {
    if (!node.is_leaf) {
        if (node.left) { free_node(*node.left); delete node.left; }
        if (node.right) { free_node(*node.right); delete node.right; }
    }
}

RandomForest::RandomForest(size_t n_trees, size_t max_depth,
                               size_t min_samples_leaf, size_t max_features, bool regression)
    : n_trees_(n_trees), max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf), max_features_(max_features),
      regression_(regression) {}

void RandomForest::fit(const Tensor& X, const Tensor& y) {
    trees_.clear();
    size_t n = X.rows;

    for (size_t t = 0; t < n_trees_; ++t) {
        Tree tree;
        tree.regression = regression_;

        // Bootstrap sample
        std::vector<size_t> boot(n);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (size_t i = 0; i < n; ++i) boot[i] = dist(rng_);

        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = boot[i];

        tree.root.is_leaf = false;
        tree.fit_node(tree.root, X, y, indices,
                      0, max_depth_, min_samples_leaf_,
                      max_features_, rng_);
        trees_.push_back(tree);
    }
}

Tensor RandomForest::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> preds;
        for (const auto& tree : trees_)
            preds.push_back(tree.predict_node(tree.root, X, i));

        if (regression_) {
            double sum = 0.0;
            for (double p : preds) sum += p;
            out[i][0] = sum / preds.size();
        } else {
            std::map<double, size_t> counts;
            for (double p : preds) counts[p]++;
            size_t best_c = 0;
            double best = 0;
            for (auto& c : counts)
                if (c.second > best_c) { best_c = c.second; best = c.first; }
            out[i][0] = best;
        }
    }
    return out;
}

double RandomForest::score(const Tensor& X, const Tensor& y) const {
    Tensor pred = predict(X);
    size_t n = y.rows;
    double correct = 0.0;
    // Scale-adaptive tolerance for regression: 1% of y range, floor at 0.5
    double y_min = 1e100, y_max = -1e100;
    for (size_t i = 0; i < n; ++i) {
        double v = y[i][0];
        if (v < y_min) y_min = v;
        if (v > y_max) y_max = v;
    }
    double tol = regression_ ? std::max(0.5, (y_max - y_min) * 0.01) : 0.5;
    for (size_t i = 0; i < n; ++i)
        if ((regression_ && std::abs(pred[i][0] - y[i][0]) < tol)
            || (!regression_ && pred[i][0] == y[i][0]))
            correct += 1.0;
    return correct / n;
}