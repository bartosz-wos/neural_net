#ifndef GROKFAST_H
#define GROKFAST_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <memory>

class Model;
class Tensor;

// GrokFast Optimizer — Lee, Sim, Ye, Choi 2024 (arXiv:2405.20233,
// "Grokfast: Accelerated Grokking by Amplifying Slow Gradients",
// NeurIPS 2024 Spotlight).
//
// Reference implementation: https://github.com/ironjr/grokfast (Python/PyTorch).
//
// Idea: in many training regimes (especially "grokking" tasks where
// generalization comes after extended overfitting), the gradient signal that
// drives generalization is HIDDEN under fast-oscillating noise. GrokFast
// maintains an EMA-filtered "slow" gradient buffer per parameter and adds
// it (amplified by lambda) to the raw gradient before passing to the inner
// optimizer. Mathematically:
//
//     buf_t     = alpha * buf_{t-1} + (1 - alpha) * grad_t           # EMA
//     grad_filtered_t = grad_t + lambda * buf_t                    # amplify
//     theta_t   = inner.step(grad_filtered_t)                       # delegate
//
// The wrapper is a thin shim: per-parameter, it (a) maintains one EMA-filter
// buffer (same shape as the parameter, lazily initialized on first step),
// (b) overwrites the stored gradient with the filtered value in-place, then
// (c) calls inner.step(model). The inner optimizer does not need to know
// about GrokFast — it just sees a filtered gradient.
//
// Defaults (paper §3.2 / Table 3):
//   - lambda = 2.0   (amplification factor; paper main experiments use 2.0)
//   - alpha  = 0.98  (EMA momentum; default for the EMA-filter variant)
//   - inner  = Adam  (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
//
// State: one Tensor (the EMA-filter buffer) per parameter, same shape as
// the parameter, lazy-initialized on first encounter. No second-moment
// statistics are stored by GrokFast itself.
//
// Edge cases:
//   - lambda = 0  → filtered gradient = raw gradient (the EMA filter still
//                   updates, but its contribution is multiplied by 0).
//                   Equivalent to inner.step on the raw gradient.
//   - alpha = 0   → buf_t = grad_t exactly (no filtering). Equivalent to
//                   (1 + lambda) global scaling on the gradient.
//   - alpha = 1   → buf_t = buf_{t-1}; the filter is frozen (degenerate).
//                   Allowed by validation (boundary).
//
// `handles_weight_decay()` delegates to the inner optimizer so wrapping
// chains (e.g. `WeightDecay(GrokFast(AdamW))`) compose correctly.
class GrokFast : public Optimizer {
public:
    // Construct GrokFast wrapping `inner`. The wrapper takes ownership of
    // `inner` via unique_ptr.
    //   - lambda must be >= 0 (paper uses 2.0)
    //   - alpha  must be in [0, 1] (paper uses 0.98)
    // Throws std::invalid_argument on bad inputs or null inner.
    explicit GrokFast(std::unique_ptr<Optimizer> inner,
                      double lambda = 2.0,
                      double alpha = 0.98);

    // Apply the EMA filter to all parameter gradients, then delegate to
    // inner.step(model).
    void step(Model& model) override;

    // Weight decay is delegated to the inner optimizer (no double-application).
    bool handles_weight_decay() const override;

    // Setters (validate input).
    void set_lambda(double v);
    void set_alpha(double v);
    void set_lr(double v);   // forwards to inner (sets Optimizer::lr too)

    // Accessors.
    double get_lambda() const { return lambda_; }
    double get_alpha()  const { return alpha_; }
    double get_lr()     const;   // returns inner's lr (with type-cast)

    Optimizer* inner() const { return inner_.get(); }

    // State diagnostics.
    bool   has_state(void* layer_ptr, size_t param_idx) const;
    size_t last_num_params_filtered() const { return last_num_params_filtered_; }

private:
    std::unique_ptr<Optimizer> inner_;
    double lambda_;
    double alpha_;
    size_t last_num_params_filtered_;
    // Per-layer state: for each Layer* we keep one EMA buffer per parameter.
    std::map<void*, std::vector<Tensor>> buf_state_;
};

#endif