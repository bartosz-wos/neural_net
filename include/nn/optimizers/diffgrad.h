#ifndef DIFFGRAD_H
#define DIFFGRAD_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

class Tensor;

// DiffGrad: "diffGrad: An Optimization Method for Convolutional Neural Networks"
// Dubey, Chakraborty, Roy, Mukherjee, Singh, Chaudhuri (2019)
// Paper: https://arxiv.org/abs/1909.11015
//
// Key idea: Adam's second-moment denominator adapts to the magnitude of the
// gradient but is INSENSITIVE to how rapidly the gradient is changing. A
// parameter with a stable, slowly-changing gradient (already near optimum) and
// a parameter with a rapidly-flipping gradient (still in a high-curvature
// region) get the same per-step scale. diffGrad introduces a per-element
// "friction coefficient" DFC that MULTIPLIES the first-moment step:
//
//     dfc_t   = sigmoid(|g_{t-1} − g_t|) = 1 / (1 + exp(−|g_{t-1} − g_t|))
//             ∈ (0, 1)  (in practice ≈ 0.5 when |g_{t-1} − g_t| ≈ 0
//                         and ≈ 1.0 when the change is large)
//
// When the gradient is barely moving (dfc ≈ 0.5), the friction is large and
// the effective step shrinks — exactly the regime where Adam overshoots.
// When the gradient is changing rapidly (dfc → 1), there is no friction and
// the step matches Adam. This trades the cheap "use the first moment
// intactly" assumption of Adam for a cost of one extra stored tensor per
// parameter: g_{t-1}.
//
// Algorithm (per parameter):
//     g_t  ← grad
//     m_t  ← β1 · m_{t-1} + (1−β1) · g_t
//     v_t  ← β2 · v_{t-1} + (1−β2) · g_t²
//     dfc  ← 1 / (1 + exp(−|g_{t-1} − g_t|))
//     m_eff← dfc ⊙ m_t
//     step_size = lr · √(1−β2^t) / (1−β1^t)
//     param ← param − step_size · m_eff / (√v_t + ε)
//     g_{t-1} ← g_t           (cache for the next step)
//
// Follows shivram1987/diffGrad reference (PyTorch diffGrad_v2.py).
//
// Weight decay: decoupled AdamW-style `param *= (1 − lr · wd)` so that
// `WeightDecay` wrappers in the repo correctly skip re-application.
// `handles_weight_decay()` returns true.
//
// Default hyperparameters (paper §III, PyTorch reference defaults):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8, wd = 0

class DiffGrad : public Optimizer {
public:
    double lr;
    double beta1;        // first moment decay
    double beta2;        // second moment decay
    double epsilon;      // numerical stability constant (added to sqrt(v))
    int t;               // step counter (starts at 1)
    double weight_decay; // decoupled (AdamW) weight decay

    explicit DiffGrad(double lr = 0.001,
                      double b1 = 0.9,
                      double b2 = 0.999,
                      double eps = 1e-8,
                      double wd = 0.0);

    void step(Model& model) override;

    // Decoupled weight decay — WeightDecay wrapper should skip re-application.
    bool handles_weight_decay() const override { return true; }

    // Convenience setters (validate inputs).
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);

    // Test diagnostics.
    bool has_state(void* layer_ptr, size_t param_idx) const;
    int  get_step() const { return t; }

private:
    struct State {
        Tensor m;       // first moment (β1-EMA of grad)
        Tensor v;       // second moment (β2-EMA of grad²)
        Tensor g_prev;  // cached gradient from the previous step
    };

    std::map<void*, std::vector<State>> state_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, State& st,
                      double lr, double b1, double b2,
                      double eps, double wd,
                      double b1_c, double b2_c);
};

#endif
