#ifndef NN_UTILS_LR_MULTIPLIER_H
#define NN_UTILS_LR_MULTIPLIER_H

#include "../core/tensor.h"
#include "../optimizers/optimizer.h"   // full definition: the wrapper subclasses Optimizer
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Model;
class Optimizer;
class Layer;

namespace lr_multiplier_detail {
    // Parameter identity used to look a scale up. `layer_name` is the owning
    // layer's name() and `param_index` its position in parameters().
    struct ParamKey {
        std::string layer_name;
        size_t param_index;
        bool operator<(const ParamKey& other) const {
            if (layer_name != other.layer_name) return layer_name < other.layer_name;
            return param_index < other.param_index;
        }
    };
}

// ===========================================================================
// LrMultiplier -- per-tensor / per-layer learning-rate multipliers.
//
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//   grep -rinE 'param_group|parameter_group|param_groups|layer.*lr.*decay
//          |llrd|depth.*scal.*lr' include/         -> 0 hits
//
// The repo has 39 optimizers and NOT ONE of them accepts a per-tensor
// learning rate: `Optimizer` exposes a single scalar `lr`, and
// `Layer::update_weights(double)` takes a single scalar. Worse, the
// per-tensor-LR plumbing is NOT a one-line addition to the base class:
// of 39 optimizer .cpp files, exactly ONE routes its update through
// `update_weights(lr)`; the other 38 read `lr` directly inside their update
// loop. So adding a hook would mean editing every optimizer in the repo.
//
// `utils/mup.h` records the same gap from the other side: its
// `mup_adam_lr_tensor` returns per-tensor LR multipliers and its header says
// the repo's optimizers "have no per-tensor-LR hook, and adding one is a
// repo-wide change this header is not entitled to make unilaterally".
// This file fills that gap WITHOUT making that change.
//
// ---------------------------------------------------------------------------
// WHY THE MULTIPLIER IS APPLIED TO THE DELTA, NOT TO THE GRADIENT
// ---------------------------------------------------------------------------
// The obvious implementation is to scale the gradient `g <- s*g` before
// calling `step()`. It is exact for SGD and is SILENTLY ANNIHILATED BY ADAM.
//
// Measured (before this file was written), with the identity
//     step(s*g) - step(0) == s * (step(g) - step(0)):
//
//     SGD : |step(3g)-step(0) - 3*(step(g)-step(0))| / max|3*Delta| = 2.64e-16
//     Adam:                                          ...            = 6.67e-01
//
// Adam's update is lr*m_hat/(sqrt(v_hat)+eps); after one step m_hat ~ g and
// v_hat ~ g^2, so the update carries g/sqrt(g^2) = sign(g) and multiplying g
// by 3 changes NOTHING. A gradient-scaling feature would be a silent no-op on
// the single most-used optimizer family in this repo.
//
// So the multiplier instead rescales the REALIZED UPDATE: the wrapper
// snapshots every parameter before `step()`, lets the optimizer do whatever
// it does, then sets
//
//     w  <-  w_before + s * (w_after - w_before)
//
// This is optimizer-agnostic -- it works for all 39 optimizers, including the
// scale-invariant ones, because it never assumes anything about how the
// update was computed.
//
// ---------------------------------------------------------------------------
// PITFALL THIS DESIGN AVOIDS, AND WHY IT IS ASSUMED AWAY
// ---------------------------------------------------------------------------
// Several optimizers do model-global work inside `step()` (Prodigy, Muon,
// AdaBelief, ...). Stepping the model PIECEWISE to give different layers
// different learning rates would split that global state across two partial
// steps. `lr_multiplier_apply` therefore takes NO optimizer and performs NO
// stepping: the caller steps once, and the multiplier corrects the result.
// A single `step()` is always performed, so global state advances exactly as
// it would have. The price is stated honestly in the header's D6: the
// correction is applied to a delta, and any optimizer whose update depends
// on MORE than its own gradient and state is not exactly reproduced -- which
// is precisely why no such optimizer claims to be.
// ===========================================================================

class LrMultiplier {
public:
    LrMultiplier() : default_scale_(1.0) {}

    // Default scale for any parameter with no explicit entry (D1: 1.0).
    double default_scale() const { return default_scale_; }
    void set_default(double s) { default_scale_ = s; }

    // Named scale; overrides the default (D3).
    void set(const std::string& layer_name, double scale) {
        scales_[lr_multiplier_detail::ParamKey{layer_name, 0}] = scale;
    }

    // Per-parameter entry, indexed by parameters() position (D3). Taking the
    // index explicitly is what makes a single Dense layer expressible as two
    // independently-scaled groups (weights vs bias) -- which a name-only map
    // cannot, and which is exactly what a muP-style bundle needs.
    void set_param(const std::string& layer_name, size_t param_index, double scale) {
        scales_[lr_multiplier_detail::ParamKey{layer_name, param_index}] = scale;
    }

    // Resolve a scale. Unnamed parameters take the default (D2/D4): an
    // unmatched name is NOT an error, because a wrapper that throws on a
    // layer nobody named is unusable across a 250-layer repo.
    double resolve(const std::string& layer_name, size_t param_index = 0) const {
        auto it = scales_.find(lr_multiplier_detail::ParamKey{layer_name, param_index});
        if (it != scales_.end()) return it->second;
        return default_scale_;
    }

    // True if this parameter has an explicit entry (not the default).
    bool has(const std::string& layer_name, size_t param_index = 0) const {
        return scales_.count(lr_multiplier_detail::ParamKey{layer_name, param_index}) > 0;
    }

    size_t size() const { return scales_.size(); }

    // Scales are unordered by contract: this is a LOOKUP table, not a list.
    // Deterministic iteration (for serialization / tests) is available via
    // entries().
    std::vector<std::pair<lr_multiplier_detail::ParamKey, double>> entries() const {
        return std::vector<std::pair<lr_multiplier_detail::ParamKey, double>>(
            scales_.begin(), scales_.end());
    }

private:
    double default_scale_;
    std::map<lr_multiplier_detail::ParamKey, double> scales_;
};

// Snapshot every parameter of `model`, so a later delta rescale can recover
// what the optimizer changed. Take this BEFORE `step()`.
void lr_multiplier_snapshot(Model& model, std::map<Layer*, std::vector<Tensor>>& out);

// Rescale each parameter's realized delta:  w <- w_before + s*(w - w_before).
// This is the correction half of a step the caller has already performed.
void lr_multiplier_apply_snapshot(Model& model, const LrMultiplier& mult,
                                  std::map<Layer*, std::vector<Tensor>>& before);

// Convenience form for a caller that HAS a snapshot but wants to pass it
// optionally. With no snapshot the delta is undefined, so this takes one at
// the current weights -- which makes the correction a zero delta, i.e. a
// no-op. That is the SAFE behaviour rather than a silent wrong answer, but it
// is also not useful: prefer `lr_multiplier_apply_snapshot`, or the wrapper.
void lr_multiplier_apply(Model& model, const LrMultiplier& mult,
                         std::map<Layer*, std::vector<Tensor>>* before = nullptr);

// A drop-in Optimizer wrapper: steps the wrapped optimizer once, then applies
// the multiplier to the realized delta. Use this rather than calling
// `step()` and `lr_multiplier_apply()` by hand.
class LrMultiplierOptimizer : public Optimizer {
public:
    LrMultiplierOptimizer(Optimizer* inner, LrMultiplier mult)
        : inner_(inner), mult_(std::move(mult)) {
        lr = inner_ ? inner_->lr : 0.001;
    }

    // The wrapped optimizer's step(), followed by the delta rescale.
    void step(Model& model) override {
        if (!inner_) return;
        std::map<Layer*, std::vector<Tensor>> before;
        lr_multiplier_snapshot(model, before);
        inner_->step(model);
        lr_multiplier_apply_snapshot(model, mult_, before);
        lr = inner_->lr;
    }

    // Deliberately NOT delegated: a wrapper that claims to handle weight decay
    // would double-apply it. The wrapped optimizer decides.
    bool handles_weight_decay() const override {
        return inner_ ? inner_->handles_weight_decay() : false;
    }

    Optimizer* inner() const { return inner_; }
    LrMultiplier& multiplier() { return mult_; }

private:
    Optimizer* inner_;   // NOT owned; the caller keeps the lifetime
    LrMultiplier mult_;
};

#endif