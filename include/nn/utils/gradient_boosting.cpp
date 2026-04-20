#include "gradient_boosting.h"
#include <cmath>
#include <algorithm>
#include <queue>

// === GradientBoosting ===

GradientBoosting::GradientBoosting(size_t n_estimators, double learning_rate,
                                     size_t max_depth, Loss loss)
    : n_estimators_(n_estimators), learning_rate_(learning_rate),
      max_depth_(max_depth), loss_(loss) {}

double GradientBoosting::RegressionTree::compute_split_gini(
        const Tensor& X, const Tensor& y, size_t feature,
        double threshold, double& left_val, double& right_val) const {

    size_t n = X.rows;
    std::vector<size_t> left_idx, right_idx;
    for (size_t i = 0; i < n; ++i) {
        if (X[i][feature] < threshold)
            left_idx.push_back(i);
        else
            right_idx.push_back(i);
    }

    if (left_idx.empty() || right_idx.empty()) return 1e100;

    left_val = 0.0;
    for (size_t i : left_idx) left_val += y[i][0];
    left_val /= left_idx.size();

    right_val = 0.0;
    for (size_t i : right_idx) right_val += y[i][0];
    right_val /= right_idx.size();

    double mse = 0.0;
    for (size_t i : left_idx) {
        double r = y[i][0] - left_val;
        mse += r * r;
    }
    for (size_t i : right_idx) {
        double r = y[i][0] - right_val;
        mse += r * r;
    }
    return mse;
}

double GradientBoosting::RegressionTree::fit(const Tensor& X, const Tensor& residuals) {
    nodes.clear();
    TreeNode root;
    root.is_leaf = false;
    root.feature_idx = 0;
    root.threshold = 0.0;
    root.value = 0.0;

    // Greedy split search
    double best_gain = -1e100;
    size_t best_f = 0;
    double best_t = 0.0;
    double best_left = 0.0, best_right = 0.0;

    for (size_t f = 0; f < X.cols; ++f) {
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t i = 0; i < X.rows; ++i)
            sorted.emplace_back(X[i][f], i);
        std::sort(sorted.begin(), sorted.end());

        for (size_t t = 0; t + 1 < sorted.size(); ++t) {
            double thr = (sorted[t].first + sorted[t + 1].first) / 2.0;
            double lv = 0, rv = 0;
            double gain = compute_split_gini(X, residuals, f, thr, lv, rv);
            if (gain < best_gain || best_gain < -1e90) {
                best_gain = gain;
                best_f = f;
                best_t = thr;
                best_left = lv;
                best_right = rv;
            }
        }
    }

    root.feature_idx = best_f;
    root.threshold = best_t;
    root.is_leaf = false;
    root.left.resize(1);
    root.right.resize(1);
    root.left[0].is_leaf = true;
    root.left[0].value = best_left;
    root.left[0].feature_idx = root.left[0].threshold = 0;
    root.right[0].is_leaf = true;
    root.right[0].value = best_right;
    root.right[0].feature_idx = root.right[0].threshold = 0;
    nodes.push_back(root);

    return (best_left + best_right) / 2.0;
}

double GradientBoosting::RegressionTree::predict(const Tensor& X) const {
    size_t n = X.rows;
    double out = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const TreeNode* node = &nodes[0];
        while (!node->is_leaf) {
            if (X[i][node->feature_idx] < node->threshold)
                node = &node->left[0];
            else
                node = &node->right[0];
        }
        out += node->value;
    }
    return out / n;
}

void GradientBoosting::fit(const Tensor& X, const Tensor& y) {
    size_t n = X.rows;
    F0_.resize(n, 0.0);

    double mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) mean_y += y[i][0];
    mean_y /= n;
    for (size_t i = 0; i < n; ++i) F0_[i] = mean_y;

    Tensor r(n, 1);
    for (size_t t = 0; t < n_estimators_; ++t) {
        for (size_t i = 0; i < n; ++i)
            r[i][0] = -(y[i][0] - F0_[i]);

        RegressionTree tree;
        tree.fit(X, r);
        trees_.push_back(tree);

        for (size_t i = 0; i < n; ++i)
            F0_[i] += learning_rate_ * tree.predict(X.get_row(i));
    }
}

Tensor GradientBoosting::predict(const Tensor& X) const {
    (void)X;
    return Tensor(X.rows, 1);
}

double GradientBoosting::score(const Tensor& X, const Tensor& y) const {
    (void)X; (void)y;
    return 0.0;
}

// === XGBoostTree ===

XGBoostTree::XGBoostTree(size_t max_depth, double lambda, double gamma,
                          double min_child_weight, double subsample, double colsample_bytree)
    : max_depth_(max_depth), lambda_(lambda), gamma_(gamma),
      min_child_weight_(min_child_weight), subsample_(subsample),
      colsample_bytree_(colsample_bytree) {
    root_.is_leaf = true;
    root_.leaf_weight = 0.0;
    root_.left = root_.right = nullptr;
}

double XGBoostTree::compute_score(const std::vector<double>& G,
                                    const std::vector<double>& H) const {
    double Gs = 0, Hs = 0;
    for (size_t i = 0; i < G.size(); ++i) { Gs += G[i]; Hs += H[i]; }
    if (Hs + lambda_ < 1e-9) return 0.0;
    return (Gs * Gs) / (Hs + lambda_);
}

double XGBoostTree::compute_weight(const std::vector<double>& G,
                                     const std::vector<double>& H) const {
    double Gs = 0, Hs = 0;
    for (size_t i = 0; i < G.size(); ++i) { Gs += G[i]; Hs += H[i]; }
    return Gs / (Hs + lambda_);
}

void XGBoostTree::build(XGBNode& node, const Tensor& X,
                          const std::vector<size_t>& indices, size_t depth) {

    std::vector<double> G(indices.size(), 0.0), H(indices.size(), 0.0);
    for (size_t i = 0; i < indices.size(); ++i) {
        G[i] = -0.5; H[i] = 1.0; // placeholders
    }

    if (depth >= max_depth_ || indices.size() < min_child_weight_) {
        node.is_leaf = true;
        node.leaf_weight = compute_weight(G, H);
        return;
    }

    size_t best_f = 0;
    double best_t = 0.0, best_gain = gamma_;
    double blG = 0, blH = 0, brG = 0, brH = 0;
    double G_sum = 0, H_sum = 0;
    for (size_t i = 0; i < G.size(); ++i) { G_sum += G[i]; H_sum += H[i]; }

    for (size_t f = 0; f < X.cols; ++f) {
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t idx : indices)
            sorted.emplace_back(X[idx][f], idx);
        std::sort(sorted.begin(), sorted.end());

        double lG = 0, lH = 0, rG = G_sum, rH = H_sum;
        for (size_t t = 0; t < sorted.size(); ++t) {
            size_t idx = sorted[t].second;
            lG += G[t]; lH += H[t];
            rG -= G[t]; rH -= H[t];

            if (lH < min_child_weight_ || rH < min_child_weight_) continue;
            double gain = compute_score({lG}, {lH}) + compute_score({rG}, {rH})
                       - compute_score({G_sum}, {H_sum}) - gamma_;
            if (gain > best_gain) {
                best_gain = gain;
                best_f = f;
                best_t = (sorted[t].first + (t + 1 < sorted.size() ? sorted[t + 1].first : sorted[t].first)) / 2.0;
                blG = lG; blH = lH; brG = rG; brH = rH;
            }
        }
    }

    if (best_gain > gamma_) {
        node.is_leaf = false;
        node.feature = best_f;
        node.threshold = best_t;
        node.gain = best_gain;
        node.left = new XGBNode();
        node.right = new XGBNode();
        node.left->is_leaf = node.right->is_leaf = true;
        node.left->leaf_weight = blG / (blH + lambda_);
        node.right->leaf_weight = brG / (brH + lambda_);
        node.left->left = node.left->right = nullptr;
        node.right->left = node.right->right = nullptr;
    } else {
        node.is_leaf = true;
        node.leaf_weight = compute_weight(G, H);
    }
}

Tensor XGBoostTree::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        const XGBNode* node = &root_;
        while (!node->is_leaf) {
            if (X[i][node->feature] < node->threshold)
                node = node->left;
            else
                node = node->right;
        }
        out[i][0] = node->leaf_weight;
    }
    return out;
}

// === XGBoostClassifier ===

XGBoostClassifier::XGBoostClassifier(size_t n_estimators, size_t max_depth,
                                      double learning_rate, double lambda,
                                      double gamma, double subsample, double colsample_bytree)
    : n_estimators_(n_estimators), max_depth_(max_depth),
      learning_rate_(learning_rate), lambda_(lambda), gamma_(gamma),
      subsample_(subsample), colsample_bytree_(colsample_bytree) {}

void XGBoostClassifier::fit(const Tensor& X, const Tensor& y, size_t n_classes) {
    n_classes_ = n_classes;
    F0_.resize(n_classes, 0.0);

    for (size_t c = 0; c < n_classes_; ++c) {
        std::vector<XGBoostTree> class_trees;
        for (size_t t = 0; t < n_estimators_; ++t) {
            XGBoostTree tree(max_depth_, lambda_, gamma_);
            Tensor g(X.rows, 1, 0.5), h(X.rows, 1, 1.0);
            tree.fit(X, g, h);
            class_trees.push_back(tree);
        }
        trees_per_class_.push_back(class_trees);
    }
}

Tensor XGBoostClassifier::predict_proba(const Tensor& X) const {
    size_t n = X.rows;
    Tensor proba(n, n_classes_);
    for (size_t c = 0; c < n_classes_; ++c) {
        Tensor raw = trees_per_class_[c][0].predict(X);
        for (size_t i = 0; i < n; ++i)
            proba[i][c] = 1.0 / (1.0 + std::exp(-raw[i][0]));
    }
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t c = 0; c < n_classes_; ++c) sum += proba[i][c];
        for (size_t c = 0; c < n_classes_; ++c) proba[i][c] /= sum;
    }
    return proba;
}

Tensor XGBoostClassifier::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor pred(n, 1);
    Tensor proba = predict_proba(X);
    for (size_t i = 0; i < n; ++i) {
        size_t best_c = 0;
        double best_p = proba[i][0];
        for (size_t c = 1; c < n_classes_; ++c)
            if (proba[i][c] > best_p) { best_p = proba[i][c]; best_c = c; }
        pred[i][0] = static_cast<double>(best_c);
    }
    return pred;
}