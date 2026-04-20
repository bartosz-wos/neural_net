#include "tabular_ensemble.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>

// === TabularDataset ===

TabularDataset::TabularDataset(const std::string& csv_path,
                               bool has_header, bool normalize,
                               double train_ratio, size_t label_col, char sep)
    : n_features_(0), n_samples_(0), train_ratio_(train_ratio) {

    std::ifstream f(csv_path);
    if (!f.is_open()) return;

    std::vector<std::vector<double>> rows;
    std::string line;
    if (has_header) std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;
        while (std::getline(ss, cell, sep)) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) rows.push_back(row);
    }
    f.close();

    n_samples_ = rows.size();
    if (n_samples_ == 0) return;

    size_t n_cols = rows[0].size();
    n_features_ = n_cols - 1;

    X_ = Tensor(n_samples_, n_features_);
    y_ = Tensor(n_samples_, 1);

    for (size_t i = 0; i < n_samples_; ++i) {
        size_t col = 0;
        for (size_t j = 0; j < n_cols; ++j) {
            if (j == label_col) {
                y_[i][0] = rows[i][j];
            } else {
                X_[i][col++] = rows[i][j];
            }
        }
    }

    if (normalize) {
        std::vector<double> mins(n_features_, 1e100), maxs(n_features_, -1e100);
        for (size_t j = 0; j < n_features_; ++j) {
            for (size_t i = 0; i < n_samples_; ++i) {
                mins[j] = std::min(mins[j], X_[i][j]);
                maxs[j] = std::max(maxs[j], X_[i][j]);
            }
        }
        for (size_t j = 0; j < n_features_; ++j) {
            double range = maxs[j] - mins[j];
            if (range < 1e-9) range = 1.0;
            for (size_t i = 0; i < n_samples_; ++i)
                X_[i][j] = (X_[i][j] - mins[j]) / range;
        }
    }
}

TabularDataset::Split TabularDataset::split() const {
    Split s;
    size_t n = n_samples_;
    size_t train_n = static_cast<size_t>(n * train_ratio_);

    s.X_train = Tensor(train_n, n_features_);
    s.y_train = Tensor(train_n, 1);
    s.X_test = Tensor(n - train_n, n_features_);
    s.y_test = Tensor(n - train_n, 1);

    for (size_t i = 0; i < train_n; ++i) {
        for (size_t j = 0; j < n_features_; ++j)
            s.X_train[i][j] = X_[i][j];
        s.y_train[i][0] = y_[i][0];
    }
    for (size_t i = train_n; i < n; ++i) {
        size_t ti = i - train_n;
        for (size_t j = 0; j < n_features_; ++j)
            s.X_test[ti][j] = X_[i][j];
        s.y_test[ti][0] = y_[i][0];
    }

    return s;
}

// === DecisionStump ===

void DecisionStump::fit(const Tensor& X, const Tensor& y, const std::vector<double>& weights) {
    size_t n = X.rows;
    size_t m = X.cols;
    if (weights.empty()) {
        std::vector<double> w(n, 1.0 / n);
        return fit(X, y, w);
    }

    double best_gini = 1e100;
    feature_idx_ = 0;
    threshold_ = 0.0;
    polarity_ = 1.0;

    for (size_t f = 0; f < m; ++f) {
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t i = 0; i < n; ++i)
            sorted.emplace_back(X[i][f], i);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t t = 0; t < n - 1; ++t) {
            double thr = (sorted[t].first + sorted[t + 1].first) / 2.0;

            // Gini for left/right split
            double left_pos = 0, left_neg = 0, right_pos = 0, right_neg = 0;
            double left_w = 0, right_w = 0;
            for (size_t i = 0; i < n; ++i) {
                double w = weights[i];
                bool left = (X[i][f] < thr);
                double label = y[i][0];
                if (left) {
                    left_w += w;
                    if (label > 0) left_pos += w;
                    else left_neg += w;
                } else {
                    right_w += w;
                    if (label > 0) right_pos += w;
                    else right_neg += w;
                }
            }

            double gini_left = 1.0 - (left_pos / (left_w + 1e-10)) * (left_pos / (left_w + 1e-10))
                            - (left_neg / (left_w + 1e-10)) * (left_neg / (left_w + 1e-10));
            double gini_right = 1.0 - (right_pos / (right_w + 1e-10)) * (right_pos / (right_w + 1e-10))
                              - (right_neg / (right_w + 1e-10)) * (right_neg / (right_w + 1e-10));
            double weighted_gini = (left_w / n) * gini_left + (right_w / n) * gini_right;

            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                feature_idx_ = f;
                threshold_ = thr;
                polarity_ = 1.0;
            }

            // Flipped polarity
            double gini_left_f = 1.0 - (right_pos / (right_w + 1e-10)) * (right_pos / (right_w + 1e-10))
                               - (right_neg / (right_w + 1e-10)) * (right_neg / (right_w + 1e-10));
            double gini_right_f = 1.0 - (left_pos / (left_w + 1e-10)) * (left_pos / (left_w + 1e-10))
                                - (left_neg / (left_w + 1e-10)) * (left_neg / (left_w + 1e-10));
            double weighted_gini_f = (right_w / n) * gini_left_f + (left_w / n) * gini_right_f;

            if (weighted_gini_f < best_gini) {
                best_gini = weighted_gini_f;
                feature_idx_ = f;
                threshold_ = thr;
                polarity_ = -1.0;
            }
        }
    }
}

Tensor DecisionStump::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        bool lt = (X[i][feature_idx_] < threshold_);
        out[i][0] = (polarity_ == 1.0) ? (lt ? 1.0 : -1.0) : (lt ? -1.0 : 1.0);
    }
    return out;
}

double DecisionStump::error(const Tensor& X, const Tensor& y,
                             const std::vector<double>& weights) const {
    size_t n = X.rows;
    double err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double pred = predict(X.get_row(i))[0][0];
        if (pred != y[i][0]) err += (weights.empty() ? 1.0 : weights[i]);
    }
    return err / n;
}