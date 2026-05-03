#include "activations.h"
#include <cmath>
#include <algorithm>

Tensor Softmax::operator()(const Tensor& t) const {
    Tensor result(t.rows, t.cols);
    for (size_t i = 0; i < t.rows; ++i) {
        double max_val = t[i][0];
        for (size_t j = 1; j < t.cols; ++j) {
            if (t[i][j] > max_val) max_val = t[i][j];
        }
        double sum_exp = 0.0;
        for (size_t j = 0; j < t.cols; ++j) {
            result[i][j] = std::exp(t[i][j] - max_val);
            sum_exp += result[i][j];
        }
        sum_exp = std::max(sum_exp, 1e-300);  // guard: all exp underflow → 0
        for (size_t j = 0; j < t.cols; ++j) {
            result[i][j] /= sum_exp;
        }
    }
    return result;
}

double Softmax::cross_entropy_loss(const Tensor& logits, const Tensor& labels) {
    Tensor probs = Softmax()(logits);
    double loss = 0.0;
    size_t N = logits.rows;
    for (size_t i = 0; i < probs.rows; ++i) {
        for (size_t j = 0; j < probs.cols; ++j) {
            if (labels[i][j] > 0) {
                loss -= std::log(probs[i][j] + 1e-7);
                break; // one-hot, so break after the 1
            }
        }
    }
    return loss / N;
}

Tensor Softmax::cross_entropy_grad(const Tensor& logits, const Tensor& labels) {
    Tensor probs = Softmax()(logits);
    Tensor grad = probs - labels;  // dL/dz = softmax(z) - y (for one-hot)
    grad = grad * (1.0 / static_cast<double>(logits.rows));
    return grad;
}

Tensor LogSoftmax::operator()(const Tensor& t) const {
    Tensor result(t.rows, t.cols);
    for (size_t i = 0; i < t.rows; ++i) {
        double max_logit = t[i][0];
        for (size_t j = 1; j < t.cols; ++j)
            if (t[i][j] > max_logit) max_logit = t[i][j];
        double sum_exp = 0.0;
        for (size_t j = 0; j < t.cols; ++j)
            sum_exp += std::exp(t[i][j] - max_logit);
        sum_exp = std::max(sum_exp, 1e-300);  // guard: all exp underflow → 0
        double log_sum_exp = std::log(sum_exp);
        for (size_t j = 0; j < t.cols; ++j)
            result[i][j] = t[i][j] - max_logit - log_sum_exp;
    }
    return result;
}


// Numerically stable combined softmax + cross-entropy loss
// Handles both one-hot labels and class-index labels
double softmax_cross_entropy(const Tensor& logits, const Tensor& labels) {
    size_t N = logits.rows;
    size_t C = logits.cols;
    double total_loss = 0.0;
    for (size_t i = 0; i < N; ++i) {
        // Find max logit for numerical stability
        double max_logit = logits[i][0];
        for (size_t j = 1; j < C; ++j)
            if (logits[i][j] > max_logit) max_logit = logits[i][j];
        // Compute softmax stably: exp(logit - max) / sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < C; ++j)
            sum_exp += std::exp(logits[i][j] - max_logit);
        // Find target class and accumulate loss
        int target = -1;
        for (size_t j = 0; j < C; ++j) {
            if (labels[i][j] > 0.5) { target = static_cast<int>(j); break; }
        }
        if (target < 0) { // try class index format
            target = static_cast<int>(labels[i][0]);
        }
        double log_prob = logits[i][target] - max_logit - std::log(std::max(sum_exp, 1e-300));
        total_loss -= log_prob;
    }
    return total_loss / static_cast<double>(N);
}

// Gradient of the stable softmax-cross-entropy: (softmax(logits) - labels) / N
// where softmax is computed stably (subtract max before exp)
Tensor softmax_cross_entropy_grad(const Tensor& logits, const Tensor& labels) {
    size_t N = logits.rows;
    size_t C = logits.cols;
    Tensor grad(N, C);
    for (size_t i = 0; i < N; ++i) {
        // Stable softmax for this row
        double max_logit = logits[i][0];
        for (size_t j = 1; j < C; ++j)
            if (logits[i][j] > max_logit) max_logit = logits[i][j];
        std::vector<double> exp_logits(C);
        double sum_exp = 0.0;
        for (size_t j = 0; j < C; ++j) {
            exp_logits[j] = std::exp(logits[i][j] - max_logit);
            sum_exp += exp_logits[j];
        }
        // Determine target
        int target = -1;
        for (size_t j = 0; j < C; ++j) {
            if (labels[i][j] > 0.5) { target = static_cast<int>(j); break; }
        }
        if (target < 0)
            target = static_cast<int>(labels[i][0]);
        // dL/dz_k = softmax_k - y_k
        for (size_t j = 0; j < C; ++j) {
            double prob = exp_logits[j] / sum_exp;
            double y = (j == static_cast<size_t>(target)) ? 1.0 : 0.0;
            grad[i][j] = (prob - y) / static_cast<double>(N);
        }
    }
    return grad;
}

Tensor PReLU::operator()(const Tensor& t) const {
    return t.apply([this](double x) {
        // Clamp for numerical stability
        double x_clamped = std::max(-100.0, std::min(100.0, x));
        return x_clamped >= 0 ? x_clamped : alpha * x_clamped;
    });
}

Tensor LeakyReLU::operator()(const Tensor& t) const {
    return t.apply([this](double x) { return x >= 0 ? x : slope * x; });
}

Tensor ELU::operator()(const Tensor& t) const {
    return t.apply([this](double x) { return x >= 0 ? x : alpha * (std::exp(x) - 1.0); });
}

Tensor Softplus::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        // Clamp large positive x to avoid exp overflow
        if (x > 700) x = 700;
        return std::log(1.0 + std::exp(x));
    });
}

static const double GELU_A = std::sqrt(2.0 / std::acos(-1.0));

Tensor GELU::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        // Clamp input to [-4, 4] for numerical stability
        double x_clamped = std::max(-4.0, std::min(4.0, x));
        double cdf = 0.5 * (1.0 + std::tanh(GELU_A * (x_clamped + 0.044715 * x_clamped * x_clamped * x_clamped)));
        return x * cdf;
    });
}

double GELU::derivative(double x) const {
    // Clamp input to [-4, 4] for numerical stability
    double x_clamped = std::max(-4.0, std::min(4.0, x));
    double arg = GELU_A * (x_clamped + 0.044715 * x_clamped * x_clamped * x_clamped);
    double tanh_val = std::tanh(arg);
    double tanh_sq = tanh_val * tanh_val;  // sech²(arg) = 1 - tanh²
    double cdf = 0.5 * (1.0 + tanh_val);
    double pdf = 0.5 * GELU_A * (1.0 - tanh_sq) * (1.0 + 3.0 * 0.044715 * x_clamped * x_clamped);
    // FIX (Bug 8): use x_clamped instead of unclamped x in the x*pdf term
    return cdf + x_clamped * pdf;
}

Tensor Swish::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        double sig = 1.0 / (1.0 + std::exp(-x));
        return x * sig;
    });
}

double Swish::derivative(double x) const {
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig + x * sig * (1.0 - sig);
}

Tensor Mish::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        // Clamp to [-700, 700]: exp(700) ~ 1e304, log overflow past that
        if (x > 700)      x = 700;
        else if (x < -700) x = -700;
        double sp = std::log(1.0 + std::exp(x)); // softplus, numerically stable both sides
        return x * std::tanh(sp);
    });
}

Tensor SELU::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        if (x >= 0) return scale * x;
        double x_clamped = std::max(x, -700.0);
        return scale * alpha * (std::exp(x_clamped) - 1.0);
    });
}

double SELU::derivative(double x) const {
    if (x >= 0) return scale;
    double x_clamped = std::max(x, -700.0);
    return scale * alpha * std::exp(x_clamped);
}

Tensor HardSigmoid::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        if (x <= -3.0) return 0.0;
        if (x >= 3.0)  return 1.0;
        return (x + 3.0) / 6.0;
    });
}

Tensor HardSwish::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        if (x <= -3.0) return 0.0;
        if (x >= 3.0)  return x;
        return x * (x + 3.0) / 6.0;
    });
}
