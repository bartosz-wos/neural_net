#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../core/tensor.h"
#include <cmath>
#include <functional>

// Activation functions as function objects for use with Tensor::apply
struct ReLU {
    double operator()(double x) const { return x > 0 ? x : 0; }
    double derivative(double x) const { return x > 0 ? 1.0 : 0.0; }
};

struct Sigmoid {
    double operator()(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
    double derivative(double x) const {
        double s = this->operator()(x);
        return s * (1.0 - s);
    }
};

struct Tanh {
    double operator()(double x) const { return std::tanh(x); }
    double derivative(double x) const {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
};

struct Softmax {
    // Applies softmax row-wise to a 2D tensor (treats each row as a sample)
    Tensor operator()(const Tensor& t) const;
    // Cross-entropy loss with softmax (expects logits and one-hot labels)
    static double cross_entropy_loss(const Tensor& logits, const Tensor& labels);
    // Gradient of cross-entropy w.r.t. logits: (softmax - one_hot) / N
    static Tensor cross_entropy_grad(const Tensor& logits, const Tensor& labels);
};

struct LogSoftmax {
    // Applies log-softmax row-wise: log(softmax(x)) = x - max - log(sum(exp(x - max)))
    // Numerically stable: subtracts max logit before exp to prevent overflow
    Tensor operator()(const Tensor& t) const;
};

// PReLU: Parametric ReLU — learnable negative slope (default 0.01)
// From "Delving Deep into Rectifiers" (He et al., 2015)
struct PReLU {
    double alpha;
    explicit PReLU(double init_alpha = 0.01) : alpha(init_alpha) {}
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const {
        // Clamp for numerical stability
        double x_clamped = std::max(-100.0, std::min(100.0, x));
        return x_clamped >= 0 ? x_clamped : alpha * x_clamped;
    }
    double derivative(double x) const { return x >= 0 ? 1.0 : alpha; }
};

// LeakyReLU: slope for negative side (default 0.01)
struct LeakyReLU {
    double slope;
    explicit LeakyReLU(double slope = 0.01) : slope(slope) {}
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const { return x >= 0 ? x : slope * x; }
    double derivative(double x) const { return x >= 0 ? 1.0 : slope; }
};

// ELU: exponential linear unit
struct ELU {
    double alpha;
    explicit ELU(double alpha = 1.0) : alpha(alpha) {}
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const { return x >= 0 ? x : alpha * (std::exp(x) - 1.0); }
    double derivative(double x) const { return x >= 0 ? 1.0 : alpha * std::exp(x); }
};

// Softplus: smooth ReLU
struct Softplus {
    Tensor operator()(const Tensor& t) const;
    double derivative(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
};

// GELU: Gaussian Error Linear Unit (approx from original paper)
struct GELU {
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const {
        // Clamp input for numerical stability (matches Tensor version)
        double x_clamped = std::max(-4.0, std::min(4.0, x));
        double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * (x_clamped + 0.044715 * x_clamped * x_clamped * x_clamped)));
        return x * cdf;
    }
    double derivative(double x) const;
};

// Swish: x * sigmoid(x)
struct Swish {
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const { return x / (1.0 + std::exp(-x)); }
    double derivative(double x) const;
};

// SELU: Scaled Exponential Linear Unit — self-normalizing (Klambauer et al., 2017)
struct SELU {
    static constexpr double alpha = 1.6732632423543772848470426433812;
    static constexpr double scale  = 1.0507009873554804934193349852946;
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const {
        return x >= 0 ? scale * x : scale * alpha * (std::exp(x) - 1.0);
    }
    double derivative(double x) const;
};

// Mish: x * tanh(softplus(x)) — smooth activation, better than ReLU on ImageNet
struct Mish {
    Tensor operator()(const Tensor& t) const;
    double operator()(double x) const {
        double sp = std::log(1.0 + std::exp(x));  // softplus
        return x * std::tanh(sp);
    }
    double derivative(double x) const {
        double sp = std::log(1.0 + std::exp(x));
        double tanh_sp = std::tanh(sp);
        return tanh_sp + x * (1.0 - tanh_sp * tanh_sp) * (1.0 / (1.0 + std::exp(-x)));
    }
};

// Helper: apply activation and its derivative elementwise
// Numerically stable softmax + cross-entropy combined loss
// (subtracts max logit before exp to avoid overflow)
double softmax_cross_entropy(const Tensor& logits, const Tensor& labels);

// Gradient of the combined softmax-cross-entropy w.r.t. logits
// (softmax - labels) / N  but computed stably
Tensor softmax_cross_entropy_grad(const Tensor& logits, const Tensor& labels);

inline Tensor activate(const Tensor& t, const std::function<double(double)>& f) {
    return t.apply(f);
}

#endif
