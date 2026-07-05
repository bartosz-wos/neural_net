#ifndef SAM_H
#define SAM_H

#include "optimizer.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <map>
#include <vector>
#include <memory>
#include <functional>

// =========================================================================
// SAM: Sharpness-Aware Minimization
// =========================================================================
//
// Foret, Klein, Mozafari, Lopez-Paz, Netanyahu 2021
// "Sharpness-Aware Minimization for Efficiently Improving Generalization"
// (https://arxiv.org/abs/2102.11600)
//
// SAM is a meta-optimizer that wraps any inner optimizer (typically SGD or
// Adam). It searches for parameters that lie in neighborhoods having
// uniformly low loss, which empirically improves generalization.
//
// Algorithm (per step):
//   1. Compute the gradient g at the current weights w.
//   2. Compute the "ascent" perturbation:
//        eps_w = rho * g / max(||g||, eps_global)
//      (broadcast the same scalar to every parameter coordinate)
//   3. Move to the perturbed weights w_pert = w + eps_w.
//   4. Compute the gradient g_pert at the perturbed weights.
//   5. Restore the original weights w.
//   6. Apply the inner optimizer's step using g_pert (NOT g).
//
// Because SAM requires TWO forward+backward passes per step (one at w, one
// at w_perturbed), it cannot fit cleanly into Optimizer::step(Model&)
// which is invoked once per step. We expose the two-phase API:
//
//   model.forward(X);
//   model.backward(grad, 0.0);              // populates grad at w
//   sam.first_step(model);                 // save w, perturb w -> w+eps
//   model.forward(X);
//   model.backward(grad, 0.0);              // populates grad at w+eps
//   sam.second_step(model);                // restore w, run inner.step()
//
// The base Optimizer::step(Model&) is overridden to throw — callers must
// use the two-phase API.
//
// Hyperparameters:
//   - rho: neighborhood size (paper default 0.05; smaller for fine-tuning).
//     The maximum L2 distance the perturbed weights can move from w.
//   - eps_global: numerical stabilizer for the gradient norm (default 1e-12).
//   - adaptive (Foret §4 appendix variant): per-parameter perturbation
//     scaling `eps_w[i] = rho * grad[i] / (||grad_w||_i + eps_param)` where
//     ||grad_w||_i is the per-parameter (not global) L2 norm. Improves
//     stability for layers with very different gradient magnitudes.
//     Disabled by default — use the standard formulation.
//
// State:
//   - For each (layer, parameter) we store one Tensor copy of the weights
//     (used to restore after the perturbation). This is the SAME memory
//     footprint as the inner optimizer's state (each copy is one Tensor).
//   - A bool flag indicating whether the optimizer is currently in the
//     perturbed state (between first_step and second_step).
//
// SAM takes ownership of the inner optimizer (via unique_ptr), same
// pattern as Lookahead.
// =========================================================================
class SAM : public Optimizer {
public:
    explicit SAM(Optimizer* inner, double rho = 0.05, double eps_global = 1e-12, bool adaptive = false);

    // Override the base class step() to require the two-phase API.
    // Calling this directly is a programming error — it means the caller
    // didn't perform the second forward+backward at the perturbed position.
    void step(Model& model) override;

    // Phase 1: compute the perturbation and apply it.
    //   - Snapshots current weights into `saved_weights_`.
    //   - Computes the global gradient L2 norm across ALL parameters of ALL layers.
    //   - For each parameter, sets `param += rho * grad / max(global_norm, eps_global)`.
    //   - Marks the optimizer as "perturbed".
    //   - Does NOT clear gradients (the caller will run another backward).
    void first_step(Model& model);

    // Phase 2: restore weights, then run the inner optimizer's step.
    //   - Verifies we are in the perturbed state.
    //   - Restores saved weights into the model parameters.
    //   - Calls inner_->step(model) — this uses the gradient that the caller
    //     just computed at the perturbed position.
    //   - Clears `saved_weights_` and marks the optimizer as not perturbed.
    void second_step(Model& model);

    // ----- Accessors for testing -----
    double get_rho() const { return rho_; }
    double get_eps_global() const { return eps_global_; }
    bool get_adaptive() const { return adaptive_; }
    Optimizer* inner() const { return inner_.get(); }
    bool is_perturbed() const { return perturbed_; }
    double last_global_grad_norm() const { return last_global_norm_; }
    // For testing: per-parameter perturbation magnitude used on the most
    // recent first_step (the maximum element-wise |epsilon| across all
    // parameters, in the adaptive case; or rho * grad/max(||g||, eps),
    // which is constant across parameters in the standard case).
    double last_perturbation_norm() const { return last_eps_scalar_; }

    // SAM does not apply weight decay directly — the inner optimizer does.
    bool handles_weight_decay() const override {
        return inner_ ? inner_->handles_weight_decay() : false;
    }

private:
    // For each (layer_ptr, param_index), save the original weight Tensor.
    std::map<void*, std::vector<Tensor>> saved_weights_;

    // Configuration
    std::unique_ptr<Optimizer> inner_;
    double rho_;
    double eps_global_;
    bool adaptive_;

    // State
    bool perturbed_ = false;
    double last_global_norm_ = 0.0;  // global ||grad|| at first_step
    double last_eps_scalar_ = 0.0;   // scalar perturbation magnitude used at first_step

    // Compute the L2 norm of all gradients across all parameters of all layers.
    double compute_global_grad_norm(const Model& model) const;

    // Compute the per-parameter L2 norm of its gradient (for adaptive SAM).
    double compute_param_grad_norm(const Tensor& grad) const;
};

#endif