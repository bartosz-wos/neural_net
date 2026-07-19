#ifndef LARS_H
#define LARS_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// LARS: Layer-wise Adaptive Rate Scaling
//
// You, Gitman, Ginsburg (2017),
// "Large Batch Training of Convolutional Networks"
// https://arxiv.org/abs/1708.03888
//
// LARS stabilizes large-batch SGD by selecting a local step scale for each
// parameter tensor from the ratio between the parameter norm and its update
// norm. The update implemented here follows the canonical modern form used by
// self-supervised systems such as Barlow Twins:
//
//   u_t = g_t + lambda * w_t
//   q_t = eta * ||w_t||_2 / (||u_t||_2 + epsilon)
//   m_t = momentum * m_{t-1} + q_t * u_t
//   w_{t+1} = w_t - lr * m_t
//
// If either norm is zero, q_t falls back to 1 so the update stays finite and
// zero-initialized parameters can still move. Vector-shaped parameters
// (rows == 1 or cols == 1) are excluded from adaptation and weight decay by
// default, matching the common bias/normalization filter in Barlow Twins and
// SimCLR recipes. Both filters are independently configurable.
//
// State per parameter: one momentum tensor with the same shape as the
// parameter. The most recent trust ratio is also cached for diagnostics.
//
// Unlike several legacy optimizers in this repository, LARS deliberately does
// NOT redeclare `lr`: it consumes Optimizer::lr directly, so LRScheduler writes
// affect the learning rate used by step().
// =========================================================================
class LARS : public Optimizer {
public:
    explicit LARS(double lr = 0.1,
                  double momentum = 0.9,
                  double weight_decay = 1e-4,
                  double trust_coefficient = 1e-3,
                  double epsilon = 1e-8,
                  bool exclude_1d_from_adaptation = true,
                  bool exclude_1d_from_weight_decay = true);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated mutators.
    void set_lr(double new_lr);
    void set_momentum(double new_momentum);
    void set_weight_decay(double new_weight_decay);
    void set_trust_coefficient(double new_trust_coefficient);
    void set_epsilon(double new_epsilon);
    void set_exclude_1d_from_adaptation(bool exclude);
    void set_exclude_1d_from_weight_decay(bool exclude);

    // Hyperparameter accessors.
    double get_lr() const { return Optimizer::lr; }
    double get_momentum() const { return momentum_; }
    double get_weight_decay() const { return weight_decay_; }
    double get_trust_coefficient() const { return trust_coefficient_; }
    double get_epsilon() const { return epsilon_; }
    bool get_exclude_1d_from_adaptation() const {
        return exclude_1d_from_adaptation_;
    }
    bool get_exclude_1d_from_weight_decay() const {
        return exclude_1d_from_weight_decay_;
    }
    size_t num_steps() const { return num_steps_; }

    // State / diagnostic accessors. A missing momentum buffer returns an empty
    // Tensor; get_last_trust_ratio returns false when no such entry exists.
    bool has_state(void* layer_ptr) const {
        return momentum_state_.find(layer_ptr) != momentum_state_.end();
    }
    Tensor get_momentum_buffer(void* layer_ptr, size_t param_idx) const;
    bool get_last_trust_ratio(void* layer_ptr,
                              size_t param_idx,
                              double& out) const;

private:
    double momentum_;
    double weight_decay_;
    double trust_coefficient_;
    double epsilon_;
    bool exclude_1d_from_adaptation_;
    bool exclude_1d_from_weight_decay_;
    size_t num_steps_;

    std::map<void*, std::vector<Tensor>> momentum_state_;
    std::map<void*, std::vector<double>> last_trust_ratios_;

    static void validate(double lr,
                         double momentum,
                         double weight_decay,
                         double trust_coefficient,
                         double epsilon);
    static double l2_norm(const Tensor& tensor);
    static bool is_vector_shaped(const Tensor& tensor);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    double update_param(Tensor* param, Tensor* grad, Tensor& momentum_buffer);
};

#endif
