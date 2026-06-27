#ifndef SOPHIA_H
#define SOPHIA_H

#include "optimizer.h"
#include <map>
#include <vector>

// Sophia: "Sophia: A Scalable Stochastic Second-order Optimizer"
// Liu et al. 2023 (https://arxiv.org/abs/2305.11242)
//
// Update rule (per parameter θ):
//   g_t    = grad_t                                            (gradient at step t)
//   m_t    = β1 * m_{t-1} + (1 - β1) * g_t                     (first-moment EMA)
//   m̂_t   = m_t / (1 - β1^t)                                  (bias correction)
//   h_t    = β2 * h_{t-1} + (1 - β2) * h_diag_t                (diagonal Hessian EMA,
//                                                              re-estimated every k steps)
//   update = clip(m̂_t / max(h_t, ε), -ρ, ρ)                    (clipped update direction)
//   θ     -= lr * (update + wd * θ)                            (decoupled weight decay)
//
// Where:
//   - h_diag_t is the per-coordinate diagonal of the loss Hessian
//     (or Gauss-Newton / empirical Fisher surrogate).
//   - For testability this implementation supports TWO sources of h_diag:
//       (1) External: the user calls `set_hessian_estimates(layer_ptr, h_diag)`
//           before step() — typically with a Hutchinson trace estimate.
//       (2) Default (Sophia-G style): h_diag_t = g_t ⊙ g_t — the empirical Fisher.
//   - ρ is the clipping bound (paper default ρ = 1).
//   - update_period k (default 10): how often to re-estimate the Hessian.
//     Sophia estimates h_t every k steps; intermediate steps reuse the most
//     recent estimate. This implementation reuses the running EMA on every
//     step (the EMA naturally smooths over k-step windows when fed the same
//     h_diag value for k consecutive steps).
//
// Recommended hyperparameters from the paper:
//   - lr ~ 1e-3 to 3e-4 (similar to Adam)
//   - beta1 = 0.9 (momentum), beta2 = 0.99 (Hessian EMA)
//   - rho = 1.0 (update clipping bound)
//   - epsilon = 1e-12 (numerical stabilizer for h_t)
//   - update_period k = 10
//   - weight_decay: AdamW-style decoupled
class Sophia : public Optimizer {
public:
    double lr;
    double beta1;          // momentum EMA coefficient
    double beta2;          // diagonal-Hessian EMA coefficient
    double epsilon;        // numerical stabilizer in the divisor (1e-12 in the paper)
    double rho;            // update clipping bound (1.0 in the paper)
    int update_period;     // k: re-estimate Hessian every k steps
    int t;                 // timestep counter (used for bias correction)
    double weight_decay;   // AdamW-style decoupled weight decay (default 0)

    explicit Sophia(double lr = 1e-3,
                    double b1 = 0.9,
                    double b2 = 0.99,
                    double eps = 1e-12,
                    double rho_ = 1.0,
                    int k = 10,
                    double wd = 0.0);

    void step(Model& model) override;

    // Sophia already applies weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // Set externally-computed Hessian diagonal estimates for a specific layer.
    // One Tensor per parameter; each Tensor must match the corresponding param's shape.
    // These estimates override the default empirical-Fisher (g ⊙ g) source for step().
    // Called by the user before step() (e.g. when using Hutchinson trace estimation).
    void set_hessian_estimates(void* layer_ptr, const std::vector<Tensor>& h_diag);

    // Test accessors: read out the stored first-moment / Hessian EMA values
    // for a specific (layer, param_index, row, col) coordinate.
    double last_m_value(void* layer_ptr, size_t param_idx, size_t r, size_t c) const;
    double last_h_value(void* layer_ptr, size_t param_idx, size_t r, size_t c) const;

private:
    // Per-layer state
    std::map<void*, std::vector<Tensor>> m_state_;   // first-moment EMA
    std::map<void*, std::vector<Tensor>> h_state_;   // diagonal Hessian EMA

    // Externally-supplied Hessian estimates (consumed by step(); cleared after).
    std::map<void*, std::vector<Tensor>> hessian_input_;
    std::map<void*, bool> hessian_input_set_;

    // Initialize state for a layer if not already done.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update. h_diag is the source for h_t at this step
    // (either empirical-Fisher default or externally-supplied).
    void update_param(Tensor* param, Tensor* grad, Tensor& m, Tensor& h,
                      const Tensor* h_diag_src, double b1_corr);

    // Compute the empirical-Fisher h_diag for a single parameter (g ⊙ g).
    void empirical_fisher(const Tensor& grad, Tensor& out) const;
};

#endif
