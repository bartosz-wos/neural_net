#include "adaboost.h"
#include <cmath>
#include <algorithm>

AdaBoost::AdaBoost(size_t n_learners) : n_learners_(n_learners) {}

void AdaBoost::fit(const Tensor& X, const Tensor& y) {
    size_t n = X.rows;
    std::vector<double> w(n, 1.0 / n);
    alphas_.clear();
    stumps_.clear();

    for (size_t t = 0; t < n_learners_; ++t) {
        DecisionStump stump;
        stump.fit(X, y, w);
        double err = stump.error(X, y, w);

        // Clamp error to avoid division by zero or log(0)
        err = std::max(1e-10, std::min(err, 1.0 - 1e-10));
        double alpha = std::log((1.0 - err) / err);

        stumps_.push_back(stump);
        alphas_.push_back(alpha);

        // Reweight samples: w *= exp(-alpha * y * H_t(x))
        Tensor preds = stump.predict(X);
        double Z = 0.0; // normalization
        for (size_t i = 0; i < n; ++i) {
            double margin = y[i][0] * preds[i][0];
            w[i] *= std::exp(-alpha * margin);
            Z += w[i];
        }
        // Normalize
        if (Z > 0)
            for (size_t i = 0; i < n; ++i) w[i] /= Z;
    }
}

Tensor AdaBoost::predict(const Tensor& X) const {
    size_t n = X.rows;
    Tensor out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t t = 0; t < stumps_.size(); ++t) {
            Tensor pred = stumps_[t].predict(X.get_row(i));
            sum += alphas_[t] * pred[i][0];
        }
        out[i][0] = (sum >= 0) ? 1.0 : -1.0;
    }
    return out;
}

double AdaBoost::score(const Tensor& X, const Tensor& y) const {
    Tensor pred = predict(X);
    size_t n = y.rows;
    double correct = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (pred[i][0] == y[i][0]) correct += 1.0;
    return correct / n;
}