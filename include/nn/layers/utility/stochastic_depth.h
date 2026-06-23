#ifndef STOCHASTIC_DEPTH_H
#define STOCHASTIC_DEPTH_H

#include "../../core/layer.h"
#include <memory>
#include <string>

// StochasticDepth: residual-block drop-the-whole-block wrapper.
// Huang et al. 2016, "Deep Networks with Stochastic Depth"
// (https://arxiv.org/abs/1603.09382).
//
// During training, the inner sub-layer is *skipped entirely* with probability
// `p_drop_` — the residual branch is dropped as a single unit (NOT element-wise
// dropout). When the block is kept, its output is scaled by 1/(1-p_drop_) so
// that the expected output at inference time is identical to the un-scaled
// inner layer (inverted-dropout convention).
//
// During eval mode, the inner sub-layer is always applied and no scaling is
// applied — i.e. the layer is the identity wrapper.
//
// Typical usage in a deep residual network: each block is wrapped as
//   out = identity + StochasticDepth(inner_block)(x)
// so when the block is dropped, out = identity + x = x (the skip path
// naturally "takes over"). The 1/(1-p) scaling makes the running statistics
// of the block outputs match the deterministic network.
//
// Training schedule: `set_epoch_progress(progress in [0, 1])` updates
// p_drop_ according to a linear rule (similar to the original paper):
//   p_drop = base_p * progress
// Call from the training loop:
//   layer.set_epoch_progress(static_cast<double>(epoch) / max_epochs);
class StochasticDepth : public Layer {
public:
    // base_p: maximum drop probability (reached at progress=1).
    // Defaults to 0.5; the original paper uses up to 0.5 for the deepest
    // blocks in a 152-layer ResNet.
    explicit StochasticDepth(Layer* inner, double base_p = 0.5,
                              bool use_linear_schedule = true);

    // Drop the inner block with probability p_drop_ (when training).
    // When the block is kept, output = inner(x) * 1/(1 - p_drop_)
    // When the block is dropped, output = inner(x) * 0 (i.e. zero out the
    // residual branch — the downstream `+ x` then makes the layer identity).
    Tensor forward(const Tensor& input) override;

    // Inverse of forward: only the kept-path contributes to inner gradient.
    // dL/dx is passed through as identity (since y = scale * inner(x) — the
    // residual identity path is NOT part of this layer; it's a separate
    // addition outside).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "StochasticDepth"; }

    // StochasticDepth always samples a Bernoulli at every forward pass. To
    // make the layer deterministic (eval mode), call set_p_drop(0.0) before
    // forward — the wrapper then always keeps the inner block with scale=1.

    // Update the drop probability.
    // progress is clamped to [0, 1]. If use_linear_schedule_ is true:
    //   p_drop_ = base_p * progress
    // Otherwise p_drop_ is left unchanged.
    void set_epoch_progress(double progress);

    // Direct setter for fixed (non-scheduled) p_drop. Sets base_p_ = p.
    void set_p_drop(double p) { base_p_ = p; p_drop_ = p; }

    // Inspectors
    double p_drop() const { return p_drop_; }
    double base_p() const { return base_p_; }
    bool use_linear_schedule() const { return use_linear_schedule_; }
    bool was_dropped() const { return dropped_; }   // last forward: was block dropped?
    double last_scale() const { return last_scale_; } // last forward: applied scale
    Layer* inner_layer() const { return inner_.get(); }

private:
    std::unique_ptr<Layer> inner_;
    double base_p_;                  // max drop probability (final value at progress=1)
    double p_drop_;                  // current drop probability (may be updated by schedule)
    bool use_linear_schedule_;
    bool dropped_;                   // was the last forward call's block dropped?
    double last_scale_;              // scale applied to inner(x) on the last forward call
};

#endif
