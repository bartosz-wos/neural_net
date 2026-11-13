#include "quantile_regression.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// ===========================================================================
// The numerically careful primitives.
// ===========================================================================

namespace {

// The pinball loss, rho_tau(e), with the SHIPPED tie convention.
//
//   e >= 0  ->  e * tau          (under-prediction)
//   e <  0  ->  e * (tau - 1)    (over-prediction)
//
// THE TIE AT e == 0 IS THE UNDER-PREDICTION BRANCH, and that is not a free
// choice: `utils/distribution_losses.h` `QuantileLoss` does the same, and
// the K = 1 layer is required to be BIT-FOR-BIT equal to it. The value at
// e == 0 is 0 either way, so the tie only affects the GRADIENT.
inline double pinball(double e, double tau) {
    if (e >= 0.0) return e * tau;
    return e * (tau - 1.0);
}

// d(rho_tau)/d(pred), with the same tie convention.
//
//   e >= 0  ->  -tau        (d/de is +tau, and de/dpred is -1)
//   e <  0  ->  (1 - tau)   (d/de is tau - 1)
inline double pinball_grad_pred(double e, double tau) {
    if (e >= 0.0) return -tau;
    return 1.0 - tau;
}

}  // namespace

// ===========================================================================
// CONSTRUCTION
// ===========================================================================

QuantileRegression::QuantileRegression(size_t in_features,
                                         std::vector<double> quantiles)
    : in_features_(in_features),
      quantiles_(std::move(quantiles)),
      weights_(quantiles_.size(), in_features),
      bias_(1, quantiles_.size()),
      grad_weights_(quantiles_.size(), in_features),
      grad_bias_(1, quantiles_.size()),
      last_input_(0, 0) {
    if (in_features == 0) {
        throw std::invalid_argument(
            "QuantileRegression: in_features must be at least 1");
    }
    if (quantiles_.empty()) {
        throw std::invalid_argument(
            "QuantileRegression: at least one quantile level is required");
    }
    for (size_t k = 0; k < quantiles_.size(); ++k) {
        const double t = quantiles_[k];
        if (!(t > 0.0) || !(t < 1.0)) {
            throw std::invalid_argument(
                "QuantileRegression: quantile level " + std::to_string(k) +
                " is " + std::to_string(t) +
                ", which is outside the open interval (0, 1)");
        }
        if (k > 0 && !(t > quantiles_[k - 1])) {
            throw std::invalid_argument(
                "QuantileRegression: quantile levels must be strictly "
                "increasing, but level " + std::to_string(k) + " (" +
                std::to_string(t) + ") does not exceed level " +
                std::to_string(k - 1) + " (" +
                std::to_string(quantiles_[k - 1]) + ")");
        }
    }

    // The slopes start at ZERO and the intercepts at ZERO.
    //
    // **ZERO, NOT A RANDOM INITIALISATION.** A quantile model's columns are
    // ORDERED by level, and a random initialisation would give the levels
    // arbitrary, mutually-inconsistent slopes -- so the model would start
    // life with its quantile functions crossing, and the crossing diagnostic
    // would be measuring the INITIALISATION rather than the fit. Starting
    // every level at the same flat predictor (all K predictions equal, so no
    // crossings) means a crossing, if one appears, is attributable to the
    // fit.
    weights_.fill(0.0);
    bias_.fill(0.0);
    grad_weights_.fill(0.0);
    grad_bias_.fill(0.0);
}

void QuantileRegression::zero_grad() {
    grad_weights_.fill(0.0);
    grad_bias_.fill(0.0);
}

// ===========================================================================
// THE FORWARD PASS
//
//   pred_ik = x_i . w_k + b_k
//
// which is the same product the shipped Dense computes, with K outputs
// instead of one. The matrix product is (N, D) * (D, K) -> (N, K).
// ===========================================================================

Tensor QuantileRegression::forward(const Tensor& input) {
    if (input.cols != in_features_) {
        throw std::invalid_argument(
            "QuantileRegression: input width " + std::to_string(input.cols) +
            " does not match in_features " + std::to_string(in_features_));
    }

    const size_t K = quantiles_.size();
    const size_t N = input.rows;

    // weights_ is (K, D); its transpose is (D, K), so
    // input * weights_.transpose() is (N, D) * (D, K) = (N, K).
    Tensor out = input * weights_.transpose();

    // Manually broadcast the intercept row (1, K) across the batch.
    for (size_t k = 0; k < K; ++k) {
        const double bk = bias_[0][k];
        for (size_t i = 0; i < N; ++i) {
            out[i][k] += bk;
        }
    }

    // DEEP copy: the cached input must not alias the caller's tensor.
    last_input_ = input.clone();
    return out;
}

// ===========================================================================
// THE PINBALL OBJECTIVE AND ITS GRADIENT
//
//   L = (1/N) * sum_i sum_k rho_{tau_k}( y_i - pred_ik )
//
// MEAN OVER THE BATCH, SUM OVER THE LEVELS -- the shipped QuantileLoss's
// reduction, so that the K = 1 case is bit-for-bit equal to it.
//
// `y` is (N, 1): ONE response per row, compared against ALL K levels. This
// is DIFFERENT from the shipped QuantileLoss, which takes an elementwise
// (N, K) target, and the difference is checked explicitly rather than
// inferred from the shapes.
Tensor QuantileRegression::pinball_loss(const Tensor& pred,
                                        const Tensor& y) const {
    const size_t K = quantiles_.size();
    const size_t N = pred.rows;

    if (pred.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_loss: pred has " +
            std::to_string(pred.cols) + " columns, expected " +
            std::to_string(K));
    }
    if (y.cols != 1) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_loss: y must be (N, 1) -- ONE "
            "response per row, compared against all " + std::to_string(K) +
            " levels -- but it has " + std::to_string(y.cols) + " columns");
    }
    if (y.rows != N) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_loss: y has " +
            std::to_string(y.rows) + " rows, expected " + std::to_string(N));
    }
    if (N == 0) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_loss: pred must have at least one row");
    }

    double total = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double yi = y[i][0];
        for (size_t k = 0; k < K; ++k) {
            total += pinball(yi - pred[i][k], quantiles_[k]);
        }
    }
    // Mean over the BATCH. The sum over the levels is NOT divided by K: the
    // shipped QuantileLoss sums its columns the same way, and this is what
    // makes the K = 1 equality exact. The consequence -- the loss scales
    // with K -- is documented in the header.
    Tensor out(1, 1);
    out[0][0] = total / static_cast<double>(N);
    return out;
}

Tensor QuantileRegression::pinball_gradient(const Tensor& pred,
                                            const Tensor& y) const {
    const size_t K = quantiles_.size();
    const size_t N = pred.rows;

    if (pred.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_gradient: pred has " +
            std::to_string(pred.cols) + " columns, expected " +
            std::to_string(K));
    }
    if (y.cols != 1) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_gradient: y must be (N, 1) -- ONE "
            "response per row, compared against all " + std::to_string(K) +
            " levels -- but it has " + std::to_string(y.cols) + " columns");
    }
    if (y.rows != N) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_gradient: y has " +
            std::to_string(y.rows) + " rows, expected " + std::to_string(N));
    }
    if (N == 0) {
        throw std::invalid_argument(
            "QuantileRegression::pinball_gradient: pred must have at least one "
            "row");
    }

    const double inv_N = 1.0 / static_cast<double>(N);
    Tensor grad(N, K);
    for (size_t i = 0; i < N; ++i) {
        const double yi = y[i][0];
        for (size_t k = 0; k < K; ++k) {
            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k]) * inv_N;
        }
    }
    return grad;
}

// ===========================================================================
// THE BACKWARD PASS
//
// `grad_output` is dL/d(pred), shape (N, K) -- exactly what
// `pinball_gradient` produces.
//
//   dL/dw_k    = sum_i grad_output_ik * x_i        (K, D)
//   dL/db_k    = sum_i grad_output_ik               (1, K)
//   dL/dx_i    = sum_k grad_output_ik * w_k         (N, D)
//
// THE GRADIENTS ACCUMULATE (+=), like every other layer in the repo: the
// shipped `Dense::backward` does the same, and a caller that calls backward
// twice before update_weights must see the SUM.
// ===========================================================================

Tensor QuantileRegression::backward(const Tensor& grad_output,
                                    double /* learning_rate */) {
    const size_t K = quantiles_.size();
    const size_t D = in_features_;
    const size_t N = last_input_.rows;

    if (grad_output.rows != N || grad_output.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::backward: grad_output must be (N, K) = (" +
            std::to_string(N) + ", " + std::to_string(K) + "), got (" +
            std::to_string(grad_output.rows) + ", " +
            std::to_string(grad_output.cols) + ")");
    }
    if (N == 0) {
        throw std::invalid_argument(
            "QuantileRegression::backward: no forward pass has been cached");
    }

    // dL/dw_k = sum_i grad_ik * x_i
    // The slope matrix is (K, D) and grad_output is (N, K), so the
    // transpose is (K, N) and the product is (K, N) * (N, D) = (K, D).
    Tensor grad_w = grad_output.transpose() * last_input_;
    for (size_t r = 0; r < K; ++r) {
        for (size_t c = 0; c < D; ++c) {
            grad_weights_[r][c] += grad_w[r][c];
        }
    }

    // dL/db_k = sum_i grad_ik
    for (size_t k = 0; k < K; ++k) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += grad_output[i][k];
        }
        grad_bias_[0][k] += sum;
    }

    // dL/dx_i = sum_k grad_ik * w_k
    // grad_output is (N, K) and weights_ is (K, D), so the product is
    // (N, K) * (K, D) = (N, D).
    Tensor grad_input = grad_output * weights_;
    return grad_input;
}

// ===========================================================================
// THE UPDATE
// ===========================================================================

void QuantileRegression::update_weights(double learning_rate) {
    const size_t K = quantiles_.size();
    const size_t D = in_features_;
    for (size_t k = 0; k < K; ++k) {
        for (size_t d = 0; d < D; ++d) {
            weights_[k][d] -= learning_rate * grad_weights_[k][d];
        }
        bias_[0][k] -= learning_rate * grad_bias_[0][k];
    }
    // NOTE: gradients NOT zeroed here; the caller must zero via zero_grad().
}

// ===========================================================================
// THE CROSSING DIAGNOSTIC AND THE MONOTONE PROJECTION
// ===========================================================================

size_t QuantileRegression::crossing_count(const Tensor& pred) const {
    const size_t K = quantiles_.size();
    if (pred.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::crossing_count: pred has " +
            std::to_string(pred.cols) + " columns, expected " +
            std::to_string(K));
    }
    const size_t N = pred.rows;

    // The number of ADJACENT inverted pairs, summed over the rows.
    //
    // ADJACENT, not all pairs: the sequence is non-decreasing if and only if
    // every adjacent pair is ordered, so the adjacent count is zero exactly
    // when the row is monotone. Counting all pairs would report the same
    // zero/non-zero distinction but a different magnitude.
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k + 1 < K; ++k) {
            if (pred[i][k] > pred[i][k + 1]) ++count;
        }
    }
    return count;
}

Tensor QuantileRegression::monotone(const Tensor& pred) const {
    const size_t K = quantiles_.size();
    if (pred.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::monotone: pred has " +
            std::to_string(pred.cols) + " columns, expected " +
            std::to_string(K));
    }
    const size_t N = pred.rows;

    // A PROJECTION, not the model's output. Each row is sorted ascending.
    //
    // THE SORT IS PER ROW, NOT PER COLUMN: the rows are independent
    // observations, and sorting a column would mix observations. Each row's
    // MULTISET is preserved -- only the order within the row changes -- so
    // the projection cannot invent or destroy a value.
    Tensor out = pred.clone();
    std::vector<double> row(K);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) row[k] = pred[i][k];
        std::sort(row.begin(), row.end());
        for (size_t k = 0; k < K; ++k) out[i][k] = row[k];
    }
    return out;
}

Tensor QuantileRegression::level_column(const Tensor& pred, size_t k) const {
    const size_t K = quantiles_.size();
    if (pred.cols != K) {
        throw std::invalid_argument(
            "QuantileRegression::level_column: pred has " +
            std::to_string(pred.cols) + " columns, expected " +
            std::to_string(K));
    }
    if (k >= K) {
        throw std::out_of_range(
            "QuantileRegression::level_column: level index " +
            std::to_string(k) + " is out of range (there are " +
            std::to_string(K) + " levels)");
    }
    Tensor out(pred.rows, 1);
    for (size_t i = 0; i < pred.rows; ++i) {
        out[i][0] = pred[i][k];
    }
    return out;
}

// ===========================================================================
// THE PARAMETER ACCESSORS
// ===========================================================================

double QuantileRegression::weight_at(size_t k, size_t d) const {
    if (k >= quantiles_.size() || d >= in_features_) {
        throw std::out_of_range(
            "QuantileRegression::weight_at: index out of range");
    }
    return weights_[k][d];
}

void QuantileRegression::set_weight_at(size_t k, size_t d, double value) {
    if (k >= quantiles_.size() || d >= in_features_) {
        throw std::out_of_range(
            "QuantileRegression::set_weight_at: index out of range");
    }
    weights_[k][d] = value;
}

double QuantileRegression::bias_at(size_t k) const {
    if (k >= quantiles_.size()) {
        throw std::out_of_range(
            "QuantileRegression::bias_at: index out of range");
    }
    return bias_[0][k];
}

void QuantileRegression::set_bias_at(size_t k, double value) {
    if (k >= quantiles_.size()) {
        throw std::out_of_range(
            "QuantileRegression::set_bias_at: index out of range");
    }
    bias_[0][k] = value;
}
