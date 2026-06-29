#include "gumbel_softmax.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

GumbelSoftmax::GumbelSoftmax(double temperature, bool hard, unsigned seed)
    : temperature_(temperature > 0.0 ? temperature : 1.0),
      hard_(hard),
      training_(true),
      seed_(seed),
      rng_(seed == 0
               ? std::mt19937(std::random_device{}())
               : std::mt19937(seed)) {}

Tensor GumbelSoftmax::softmax_with_temperature(const Tensor& x, double temperature) {
    // y_i = exp(x_i / tau) / sum_j exp(x_j / tau). Row-wise, numerically stable.
    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        // subtract max for stability (doesn't change result)
        double m = -1e300;
        for (size_t j = 0; j < x.cols; ++j) {
            double v = x[i][j] / temperature;
            if (v > m) m = v;
        }
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] / temperature - m);
            out[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum > 0.0 ? sum : 1e-300);
        for (size_t j = 0; j < x.cols; ++j) {
            out[i][j] *= inv;
        }
    }
    return out;
}

Tensor GumbelSoftmax::sample_gumbel(size_t rows, size_t cols) {
    // g = -log(-log(u)), u ~ Uniform(0, 1).  We avoid log(0) by clamping
    // u into [eps, 1 - eps] before taking logs.
    constexpr double eps = 1e-12;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    Tensor g(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double u = uni(rng_);
            if (u < eps) u = eps;
            if (u > 1.0 - eps) u = 1.0 - eps;
            double neg_log_u = -std::log(u);
            // -log(0) overflow: clamp
            if (neg_log_u < eps) neg_log_u = eps;
            g[i][j] = -std::log(neg_log_u);
        }
    }
    return g;
}

Tensor GumbelSoftmax::forward(const Tensor& logits) {
    if (logits.rows == 0 || logits.cols == 0) {
        throw std::invalid_argument("GumbelSoftmax::forward: empty input");
    }
    if (temperature_ <= 0.0) {
        throw std::invalid_argument("GumbelSoftmax::forward: temperature must be > 0");
    }

    last_logits_ = logits;

    if (training_) {
        // y_soft = softmax((logits + g) / tau)
        Tensor g = sample_gumbel(logits.rows, logits.cols);
        Tensor shifted(logits.rows, logits.cols);
        for (size_t i = 0; i < logits.rows; ++i) {
            for (size_t j = 0; j < logits.cols; ++j) {
                shifted[i][j] = (logits[i][j] + g[i][j]) / temperature_;
            }
        }
        last_y_soft_ = softmax_with_temperature(shifted, 1.0);  // already divided by tau
    } else {
        // Eval: deterministic temperature-scaled softmax, no Gumbel noise.
        last_y_soft_ = softmax_with_temperature(logits, temperature_);
    }

    if (hard_) {
        // Straight-through: return one-hot(argmax(y_soft)).
        Tensor y_hard(logits.rows, logits.cols);
        for (size_t i = 0; i < logits.rows; ++i) {
            size_t best = 0;
            double best_v = last_y_soft_[i][0];
            for (size_t j = 1; j < logits.cols; ++j) {
                if (last_y_soft_[i][j] > best_v) {
                    best_v = last_y_soft_[i][j];
                    best = j;
                }
            }
            for (size_t j = 0; j < logits.cols; ++j) {
                y_hard[i][j] = (j == best) ? 1.0 : 0.0;
            }
        }
        return y_hard;
    }

    return last_y_soft_;
}

Tensor GumbelSoftmax::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (last_y_soft_.rows == 0 || last_y_soft_.cols == 0) {
        throw std::runtime_error("GumbelSoftmax::backward called before forward");
    }
    if (grad_output.rows != last_y_soft_.rows || grad_output.cols != last_y_soft_.cols) {
        throw std::invalid_argument("GumbelSoftmax::backward: grad_output shape mismatch");
    }

    // d_logits[b,k] = (1/tau) * (grad_output[b,k] * y[b,k] - y[b,k] * sum_i grad_output[b,i] * y[b,i])
    // Derived from softmax with shifted logits z = (l + g) / tau:
    //   dy_k / dl_j = (1/tau) * (delta_{kj} * y_k - y_k * y_j)
    //   d_logits_{b,k} = sum_j grad_output_{b,j} * dy_j / dl_k
    Tensor grad_logits(last_y_soft_.rows, last_y_soft_.cols);
    for (size_t b = 0; b < last_y_soft_.rows; ++b) {
        // sum_i grad_output[b,i] * y[b,i]  — shared per-row scalar
        double row_dot = 0.0;
        for (size_t i = 0; i < last_y_soft_.cols; ++i) {
            row_dot += grad_output[b][i] * last_y_soft_[b][i];
        }
        for (size_t k = 0; k < last_y_soft_.cols; ++k) {
            double yk = last_y_soft_[b][k];
            grad_logits[b][k] = (1.0 / temperature_) * yk * (grad_output[b][k] - row_dot);
        }
    }
    return grad_logits;
}