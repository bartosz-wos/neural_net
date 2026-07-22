#ifndef SIGNUM_H
#define SIGNUM_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// Signum: signSGD with exponential-moving-average momentum
//
// Bernstein, Wang, Azizzadenesheli, Anandkumar (ICML 2018),
// "signSGD: Compressed Optimisation for Non-Convex Problems"
// https://proceedings.mlr.press/v80/bernstein18a.html
//
// Per parameter coordinate:
//   m_t = beta * m_{t-1} + (1-beta) * g_t
//   s_t = sign(m_t)
//   w_t = w_{t-1} - lr * (s_t + weight_decay * w_{t-1})
//
// beta=0 reduces exactly to signSGD. Bias correction is unnecessary because
// multiplying m_t by the positive scalar 1/(1-beta^t) cannot change its sign.
// One full-size momentum tensor is stored per parameter, matching the paper.
// =========================================================================
class Signum : public Optimizer {
public:
    explicit Signum(double lr = 0.01,
                    double beta = 0.9,
                    double weight_decay = 0.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    void set_lr(double new_lr);
    void set_beta(double new_beta);
    void set_weight_decay(double new_weight_decay);

    double get_lr() const { return Optimizer::lr; }
    double get_beta() const { return beta_; }
    double get_weight_decay() const { return weight_decay_; }
    size_t num_steps() const { return num_steps_; }

    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_momentum(void* layer_ptr, size_t param_idx) const;

private:
    double beta_;
    double weight_decay_;
    size_t num_steps_;
    std::map<void*, std::vector<Tensor>> state_;

    static void validate(double lr, double beta, double weight_decay);
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, Tensor& momentum);
};

#endif
