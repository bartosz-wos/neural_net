#ifndef ADABELEIF_H
#define ADABELEIF_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

class Tensor;

// AdaBelief: Adapting Stepsizes by the Belief in Gradient Direction
// Paper: https://arxiv.org/abs/2010.07468
//
// Key innovation: replaces second moment v_t with variance of gradient residuals:
//   m_t = beta1 * m_{t-1} + (1-beta1) * grad_t              (first moment)
//   s_t = beta2 * s_{t-1} + (1-beta2) * (grad_t - m_{t-1})^2  (belief variance)
//
// Interpretation: (grad - m_prev)^2 measures how "surprising" the gradient is
// relative to the exponential moving average. If grad ≈ m (low surprise),
// we have high belief and take a large step. If grad is far from m (high surprise),
// we have low belief and take a small step.
//
// AdaBelief properties:
//   - Uses bias-corrected first moment m_hat for direction
//   - Uses bias-corrected belief variance s_hat for step size denominator
//   - Better generalization than Adam in image classification, language modeling
//   - Implements AdamW-style weight decay
class AdaBelief : public Optimizer {
public:
    double lr;
    double beta1;      // first moment decay
    double beta2;      // second moment (belief variance) decay
    double epsilon;    // numerical stability (for sqrt)
    int t;             // timestep (starts at 1)
    double weight_decay; // L2 regularization (AdamW style)

    explicit AdaBelief(double lr = 0.001,
                       double b1 = 0.9,
                       double b2 = 0.999,
                       double eps = 1e-8,
                       double wd = 0.0);

    void step(Model& model) override;

private:
    // Belief state per parameter: (m, s) where m=first moment, s=belief variance
    struct BeliefState {
        Tensor m;  // first moment
        Tensor s;  // belief variance: E[(grad - m_prev)^2]
    };

    // Per-layer state: maps Layer* -> vector of BeliefState per parameter
    std::map<void*, std::vector<BeliefState>> state_;

    // Initialize state for a layer on first encounter
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update
    void update_param(Tensor* param, Tensor* grad,
                      BeliefState& st,
                      double lr, double b1, double b2,
                      double eps, double wd,
                      double b1_c, double b2_c);
};

#endif