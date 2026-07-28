#ifndef GRADIENT_CENTRALIZATION_H
#define GRADIENT_CENTRALIZATION_H

#include "optimizer.h"
#include "../core/model.h"
#include "../core/tensor.h"
#include <memory>
#include <stdexcept>

// Gradient Centralization (GC) — Yu et al. 2020
// "Gradient Centralization: A New Optimization Strategy for Deep Neural Networks"
// https://arxiv.org/abs/2004.01461
//
// Gradient Centralization is a state-free per-parameter optimizer wrapper that
// optimizes the network's learning efficiency by centering each gradient tensor
// (subtracting its mean) before passing it to the inner optimizer. The paper
// shows that this smooths the loss landscape and accelerates convergence, with
// negligible per-step overhead and zero extra hyperparameters.
//
// Algorithm (per parameter tensor G, applied in-place before inner.step):
//   - For 2-D grad of shape (R, C) with ROW mode:
//       mean_G[i] = (1/C) * sum_j G[i][j]
//       G[i][j] -= mean_G[i]
//   - For 2-D grad of shape (R, C) with COLUMN mode (default for Dense):
//       mean_G[j] = (1/R) * sum_i G[i][j]
//       G[i][j] -= mean_G[j]
//   - For 1-D bias tensor of shape (1, C):
//       treated as COLUMN mode (single row)
//   - For scalar (1, 1) parameter:
//       mean is the value itself, so G -= mean → 0 → no-op
//
// COLUMN mode is the default and matches the convention used in the Yu et al.
// paper for Dense layers (per-output-feature centering). ROW mode is the
// convention for Conv2D (per-filter centering) where each row is a filter.
//
// Usage:
//   SGD sgd(0.1);
//   GradientCentralization gc(&sgd);  // default COLUMN
//   gc.step(model);
//
// Handles weight decay delegation:
//   gc.handles_weight_decay() == inner.handles_weight_decay()
//   So a GC(AdamW) chain correctly lets AdamW apply its own decoupled WD.
class GradientCentralization : public Optimizer {
public:
    enum class CenterMode { ROW, COLUMN };

    // Construct wrapper around an inner optimizer. Throws on null inner.
    explicit GradientCentralization(Optimizer* inner,
                                     CenterMode mode = CenterMode::COLUMN);

    // Apply centering to every layer's gradient, then call inner.step(model).
    void step(Model& model) override;

    // Delegate handles_weight_decay to the inner optimizer so that wrapping
    // chains (like WeightDecay outer) correctly skip re-application.
    bool handles_weight_decay() const override;

    // Accessors
    Optimizer* inner() const { return inner_.get(); }
    CenterMode mode() const { return mode_; }
    void set_mode(CenterMode mode) { mode_ = mode; }

private:
    // Apply per-parameter centering in-place to `grad`.
    // 2-D tensors use the mode_ selection; 1-D/scalar tensors use column semantics.
    void center_gradient(Tensor& grad) const;

    // Inner optimizer (owns).
    std::unique_ptr<Optimizer> inner_;
    CenterMode mode_;
};

#endif
