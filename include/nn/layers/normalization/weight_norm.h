#ifndef WEIGHT_NORM_H
#define WEIGHT_NORM_H

#include "../../core/layer.h"

// WeightNorm wrapper: normalizes layer weights by their L2 norm.
// Optionally includes a learned per-output-channel scaling factor.
// Currently supports wrapping Dense layers.
class WeightNorm {
public:
    WeightNorm(Layer* wrapped, bool learnable_scale = true);
    virtual ~WeightNorm() = default;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output, double learning_rate);
    void update_weights(double learning_rate);
    void zero_grad();
    std::vector<Tensor*> parameters() { return wrapped_->parameters(); }
    std::vector<Tensor*> gradients() { return {}; }

    Layer* inner() const { return wrapped_; }

private:
    Layer* wrapped_;
    bool learnable_scale_;
    Tensor g_;        // per-output-channel scaling (rows x 1)
    Tensor v_;        // original (unnormalized) weights reference

    void normalize_weights();
};

#endif