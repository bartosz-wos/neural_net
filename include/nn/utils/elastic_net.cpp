#include "elastic_net.h"
#include <cmath>
#include <algorithm>

ElasticNet::ElasticNet(double alpha, double l1_ratio)
    : alpha_(alpha), l1_ratio_(l1_ratio) {}

double ElasticNet::penalty(const Tensor& w) const {
    double L1 = 0.0, L2_sq = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j) {
            double v = w[i][j];
            L1 += std::abs(v);
            L2_sq += v * v;
        }
    return alpha_ * (l1_ratio_ * L1 + (1.0 - l1_ratio_) * L2_sq / 2.0);
}

double ElasticNet::objective(const Tensor& X, const Tensor& y,
                               const Tensor& w, const Tensor& w0) const {
    size_t n = X.rows;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double pred = w0.rows > 0 ? w0[0][0] : 0.0;
        for (size_t j = 0; j < w.cols; ++j)
            pred += X[i][j] * w[0][j];
        double err = pred - y[i][0];
        mse += err * err;
    }
    mse /= n;
    return mse + penalty(w);
}

Tensor ElasticNet::fit(const Tensor& X, const Tensor& y, const Tensor& weights_init) {
    // Simplified: analytical solution for low-dim case + soft thresholding
    size_t n = X.rows, d = X.cols;
    Tensor w = weights_init.rows > 0 ? weights_init : Tensor(1, d);
    if (weights_init.rows == 0) w.fill(0.0);

    // Ridge solution as warm start: w = (X^TX + α(1-λ)I)^{-1} X^Ty
    // Approximate inverse via diagonal dominance
    Tensor XtX(d, d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            for (size_t k = 0; k < n; ++k)
                XtX[i][j] += X[k][i] * X[k][j];

    for (size_t i = 0; i < d; ++i)
        XtX[i][i] += alpha_ * (1.0 - l1_ratio_);

    // Compute X^Ty
    Tensor Xty(d, 1);
    for (size_t i = 0; i < d; ++i)
        for (size_t k = 0; k < n; ++k)
            Xty[i][0] += X[k][i] * y[k][0];

    // Gauss-Seidel-style iterative soft thresholding
    for (size_t iter = 0; iter < 500; ++iter) {
        for (size_t j = 0; j < d; ++j) {
            double residual = Xty[j][0];
            for (size_t k = 0; k < d; ++k)
                if (k != j) residual -= XtX[j][k] * w[0][k];
            residual /= XtX[j][j];

            // Soft thresholding: S_α(x) = sign(x) * max(|x| - α, 0)
            double L1_pen = alpha_ * l1_ratio_;
            if (residual > L1_pen) w[0][j] = (residual - L1_pen) / (XtX[j][j] + alpha_ * (1.0 - l1_ratio_));
            else if (residual < -L1_pen) w[0][j] = (residual + L1_pen) / (XtX[j][j] + alpha_ * (1.0 - l1_ratio_));
            else w[0][j] = 0.0;
        }
    }
    return w;
}

// === ElasticNetCD ===

ElasticNetCD::ElasticNetCD(double alpha, double l1_ratio,
                              size_t max_iter, double tol)
    : alpha_(alpha), l1_ratio_(l1_ratio), max_iter_(max_iter), tol_(tol) {}

Tensor ElasticNetCD::fit(const Tensor& X, const Tensor& y, const Tensor& weights_init) {
    size_t n = X.rows, d = X.cols;
    Tensor w = weights_init.rows > 0 ? weights_init : Tensor(1, d);
    if (weights_init.rows == 0) w.fill(0.0);

    // Precompute column norms for normalization
    std::vector<double> col_norm(d, 0.0);
    for (size_t j = 0; j < d; ++j)
        for (size_t i = 0; i < n; ++i)
            col_norm[j] += X[i][j] * X[i][j];

    for (size_t iter = 0; iter < max_iter_; ++iter) {
        double max_change = 0.0;
        for (size_t j = 0; j < d; ++j) {
            if (col_norm[j] < 1e-9) continue;

            double old_wj = w[0][j];
            double residual = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double pred = y[i][0];
                for (size_t k = 0; k < d; ++k)
                    if (k != j) pred -= X[i][k] * w[0][k];
                residual += X[i][j] * (pred - w[0][j] * X[i][j]);
            }
            residual /= n;

            double L1_pen = alpha_ * l1_ratio_ / (col_norm[j] + 1e-10);
            double denom = col_norm[j] / n + alpha_ * (1.0 - l1_ratio_);
            double raw = residual / denom;

            if (raw > L1_pen) w[0][j] = (raw - L1_pen) / denom;
            else if (raw < -L1_pen) w[0][j] = (raw + L1_pen) / denom;
            else w[0][j] = 0.0;

            max_change = std::max(max_change, std::abs(w[0][j] - old_wj));
        }
        if (max_change < tol_) break;
    }
    return w;
}