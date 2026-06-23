#include "stochastic_depth.h"
#include <random>
#include <algorithm>

namespace {
// Single global PRNG used for stochastic-depth block sampling.
// Mirrors the convention in spatial_dropout.cpp (Dropout1D/Dropout2D share a
// similar mt19937 with fixed seed for reproducibility).
static std::mt19937 stoch_rng_(1729);

// Sample a single Bernoulli for block survival: returns true if the block
// is KEPT (probability 1 - p_drop). Returns false if the block is DROPPED.
static bool sample_kept(double p_drop) {
    if (p_drop <= 0.0) return true;
    if (p_drop >= 1.0) return false;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    return uni(stoch_rng_) >= p_drop;
}
} // namespace

StochasticDepth::StochasticDepth(Layer* inner, double base_p, bool use_linear_schedule)
    : inner_(inner),
      base_p_(std::max(0.0, std::min(1.0, base_p))),
      p_drop_(std::max(0.0, std::min(1.0, base_p))),
      use_linear_schedule_(use_linear_schedule),
      dropped_(false),
      last_scale_(1.0) {}

Tensor StochasticDepth::forward(const Tensor& input) {
    // Decide whether to drop the block on this forward call.
    // sample_kept returns true with probability (1 - p_drop_); that's the
    // "block kept" path. With probability p_drop_ we drop the residual
    // branch entirely and return zeros (the downstream `+x` then makes the
    // full residual connection the identity).
    //
    // Inverted-dropout scaling: the kept-path output is multiplied by
    // 1/(1 - p_drop_) so the expected value at inference (when p_drop = 0
    // and there's no scaling) matches the un-scaled inner output.
    if (sample_kept(p_drop_)) {
        // KEPT
        dropped_ = false;
        double scale = (p_drop_ < 1.0) ? (1.0 / (1.0 - p_drop_)) : 1.0;
        last_scale_ = scale;
        Tensor inner_out = inner_->forward(input);
        for (size_t i = 0; i < inner_out.rows; ++i)
            for (size_t j = 0; j < inner_out.cols; ++j)
                inner_out[i][j] *= scale;
        return inner_out;
    } else {
        // DROPPED — output is zero, the downstream `+x` makes the layer identity.
        dropped_ = true;
        last_scale_ = 0.0;
        Tensor z(input.rows, input.cols);
        z.fill(0.0);
        return z;
    }
}

Tensor StochasticDepth::backward(const Tensor& grad_output, double learning_rate) {
    if (dropped_) {
        // Block was dropped in forward → no gradient flows to inner; the
        // residual `+x` path handles identity gradient outside this layer.
        Tensor z(grad_output.rows, grad_output.cols);
        z.fill(0.0);
        return z;
    }
    // Block was kept → dL/d_inner_out = grad_output * scale
    // The inner layer handles its own chain rule; we just scale.
    Tensor scaled = grad_output;
    for (size_t i = 0; i < scaled.rows; ++i)
        for (size_t j = 0; j < scaled.cols; ++j)
            scaled[i][j] *= last_scale_;
    Tensor grad_in = inner_->backward(scaled, learning_rate);
    return grad_in;
}

void StochasticDepth::update_weights(double learning_rate) {
    inner_->update_weights(learning_rate);
}

std::vector<Tensor*> StochasticDepth::parameters() {
    return inner_->parameters();
}

std::vector<Tensor*> StochasticDepth::gradients() {
    return inner_->gradients();
}

void StochasticDepth::zero_grad() {
    inner_->zero_grad();
}

Tensor StochasticDepth::get_weights() const {
    // No learnable weights in this wrapper. Return an empty tensor to satisfy
    // the interface. (The inner layer's weights are reachable via
    // inner_layer()->get_weights().)
    return Tensor(0, 0);
}

Tensor StochasticDepth::get_gradients() const {
    return Tensor(0, 0);
}

void StochasticDepth::set_epoch_progress(double progress) {
    if (!use_linear_schedule_) return;
    double p = std::max(0.0, std::min(1.0, progress));
    p_drop_ = std::max(0.0, std::min(1.0, base_p_ * p));
}
