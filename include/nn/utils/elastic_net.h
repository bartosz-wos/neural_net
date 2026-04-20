#ifndef ELASTIC_NET_H
#define ELASTIC_NET_H

#include "../core/tensor.h"
#include <vector>

// ElasticNet utility: combined L1 (Lasso) + L2 (Ridge) regularization.
// Loss = MSE + α * λ * ||w||_1 + α * (1-λ) * ||w||_2²/2
// where α = alpha (penalty strength), λ = l1_ratio (0=L2 only, 1=L1 only).
class ElasticNet {
public:
    ElasticNet(double alpha = 1.0, double l1_ratio = 0.5);
    Tensor fit(const Tensor& X, const Tensor& y,
               const Tensor& weights_init = Tensor());
    double penalty(const Tensor& weights) const;
    double objective(const Tensor& X, const Tensor& y,
                      const Tensor& weights, const Tensor& weights_init = Tensor()) const;

private:
    double alpha_;   // penalty strength
    double l1_ratio_; // L1/L2 mix (0=pure L2, 1=pure L1)
};

// Coordinate Descent solver for ElasticNet (closed-form soft-thresholding per coordinate)
class ElasticNetCD {
public:
    ElasticNetCD(double alpha = 1.0, double l1_ratio = 0.5,
                  size_t max_iter = 1000, double tol = 1e-6);
    Tensor fit(const Tensor& X, const Tensor& y,
               const Tensor& weights_init = Tensor());

private:
    double alpha_, l1_ratio_;
    size_t max_iter_;
    double tol_;
};

#endif