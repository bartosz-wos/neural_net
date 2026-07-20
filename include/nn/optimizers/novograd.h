#ifndef NOVOGRAD_H
#define NOVOGRAD_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// NovoGrad: tensor-wise adaptive second moments
//
// Ginsburg et al. (2020),
// "Training Deep Networks with Stochastic Gradient Normalized by Layerwise
// Adaptive Second Moments"
// https://arxiv.org/abs/1905.11286
//
// For every parameter tensor, NovoGrad keeps one scalar EMA of the full
// squared-gradient norm and one full-size momentum tensor:
//
//   s_t = ||g_t||_2^2
//   v_1 = s_1
//   v_t = beta2 * v_{t-1} + (1-beta2) * s_t       (t > 1)
//   u_t = g_t / (sqrt(v_t) + epsilon) + weight_decay * w_t
//   m_t = beta1 * m_{t-1} + alpha * u_t
//   w_{t+1} = w_t - lr * m_t
//
// where alpha is 1 by default (Polyak-style momentum, as in the paper) or
// 1-beta1 when grad_averaging is enabled (Adam-style EMA variant). AMSGrad
// optionally replaces v_t in the denominator with max(v_1, ..., v_t).
//
// Because v_t is scalar rather than parameter-shaped, optimizer state costs
// one full tensor plus two scalars per parameter, versus Adam's two tensors.
// Weight decay is added after gradient normalization and before momentum,
// matching Algorithm 1 and the NVIDIA OpenSeq2Seq reference implementation.
//
// NovoGrad deliberately uses inherited Optimizer::lr so every repository LR
// scheduler updates the learning rate actually consumed by step().
// =========================================================================
class NovoGrad : public Optimizer {
public:
    explicit NovoGrad(double lr = 1e-3,
                      double beta1 = 0.95,
                      double beta2 = 0.98,
                      double epsilon = 1e-8,
                      double weight_decay = 0.0,
                      bool grad_averaging = false,
                      bool amsgrad = false);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated mutators.
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_epsilon(double new_epsilon);
    void set_weight_decay(double new_weight_decay);
    void set_grad_averaging(bool enabled) { grad_averaging_ = enabled; }
    void set_amsgrad(bool enabled) { amsgrad_ = enabled; }

    // Hyperparameter accessors.
    double get_lr() const { return Optimizer::lr; }
    double get_beta1() const { return beta1_; }
    double get_beta2() const { return beta2_; }
    double get_epsilon() const { return epsilon_; }
    double get_weight_decay() const { return weight_decay_; }
    bool get_grad_averaging() const { return grad_averaging_; }
    bool get_amsgrad() const { return amsgrad_; }
    size_t num_steps() const { return num_steps_; }

    // State introspection. Missing tensor state returns Tensor(0, 0); scalar
    // accessors return false for unseen layers or out-of-range parameters.
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_momentum(void* layer_ptr, size_t param_idx) const;
    bool get_second_moment(void* layer_ptr,
                           size_t param_idx,
                           double& out) const;
    bool get_max_second_moment(void* layer_ptr,
                               size_t param_idx,
                               double& out) const;

private:
    struct ParameterState {
        Tensor momentum;
        double second_moment = 0.0;
        double max_second_moment = 0.0;
        bool initialized = false;
    };

    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    bool grad_averaging_;
    bool amsgrad_;
    size_t num_steps_;

    std::map<void*, std::vector<ParameterState>> state_;

    static void validate(double lr,
                         double beta1,
                         double beta2,
                         double epsilon,
                         double weight_decay);
    static double squared_l2_norm(const Tensor& tensor);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, ParameterState& state);
};

#endif
