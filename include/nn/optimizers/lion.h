#ifndef LION_H
#define LION_H

#include "optimizer.h"
#include <map>
#include <vector>

// Lion: "Symbolic Discovery of Optimization Algorithms"
// Chen et al. 2023 (https://arxiv.org/abs/2302.06675)
//
// Update rule (per parameter):
//   c_t = beta1 * m_{t-1} + (1 - beta1) * grad_t   // interpolation
//   m_t = beta2 * m_{t-1} + (1 - beta2) * grad_t   // EMA momentum
//   param -= lr * (sign(c_t) + wd * param)
//
// Where:
//   - sign(x) = +1 if x > 0, -1 if x < 0, 0 if x == 0
//   - lr is the learning rate (typically 3-10x smaller than Adam's lr)
//   - wd is decoupled weight decay (AdamW-style)
//
// Key properties:
//   - Memory-efficient: ONE state tensor per parameter (vs Adam's two)
//   - Uses sign-of-momentum direction (constant magnitude updates)
//   - Different beta1 (interpolation) vs beta2 (EMA momentum) — Lion's signature
//   - No bias correction needed (sign() is invariant to scale)
//   - Asymmetric: c_t uses a fast-decaying interpolation while m_t uses slow-decaying EMA
//
// Recommended hyperparameters from the paper:
//   - lr ~ 3-10x smaller than Adam's lr (e.g., Adam 1e-3 -> Lion 3e-4 to 1e-4)
//   - beta1 = 0.9 (interpolation), beta2 = 0.99 (EMA) — defaults
class Lion : public Optimizer {
public:
    double lr;
    double beta1;       // interpolation coefficient for sign direction
    double beta2;       // EMA coefficient for momentum state
    double weight_decay;  // decoupled weight decay (AdamW-style)

    explicit Lion(double lr = 1e-4,
                  double b1 = 0.9,
                  double b2 = 0.99,
                  double wd = 0.0);

    void step(Model& model) override;

    // Lion already applies weight decay internally
    bool handles_weight_decay() const override { return true; }

private:
    // Per-layer state: maps Layer* -> vector of m (one Tensor per parameter)
    std::map<void*, std::vector<Tensor>> state_;

    // Initialize state for a layer if not already done
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update
    void update_param(Tensor* param, Tensor* grad, Tensor& m);
};

#endif
