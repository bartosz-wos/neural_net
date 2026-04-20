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
                loss -= std::log(probs[i][j] + 1e-12);
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
        double log_prob = logits[i][target] - max_logit - std::log(sum_exp);
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

Tensor LeakyReLU::operator()(const Tensor& t) const {
    return t.apply([this](double x) { return x >= 0 ? x : slope * x; });
}

Tensor ELU::operator()(const Tensor& t) const {
    return t.apply([this](double x) { return x >= 0 ? x : alpha * (std::exp(x) - 1.0); });
}

Tensor Softplus::operator()(const Tensor& t) const {
    return t.apply([](double x) { return std::log(1.0 + std::exp(x)); });
}

static const double GELU_A = std::sqrt(2.0 / std::acos(-1.0));

Tensor GELU::operator()(const Tensor& t) const {
    return t.apply([](double x) {
        double cdf = 0.5 * (1.0 + std::tanh(GELU_A * (x + 0.044715 * x * x * x)));
        return x * cdf;
    });
}

double GELU::derivative(double x) const {
    double arg = GELU_A * (x + 0.044715 * x * x * x);
    double tanh_val = std::tanh(arg);
    double tanh_sq = tanh_val * tanh_val;  // sech²(arg) = 1 - tanh²
    double cdf = 0.5 * (1.0 + tanh_val);
    double pdf = 0.5 * GELU_A * (1.0 - tanh_sq) * (1.0 + 3.0 * 0.044715 * x * x);
    return cdf + x * pdf;
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
        double sp = std::log(1.0 + std::exp(x)); // softplus
        return x * std::tanh(sp);
    });
}
