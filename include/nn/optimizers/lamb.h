#ifndef LAMB_H
#define LAMB_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// LAMB: Layer-wise Adaptive Moments optimizer for Batch training
//
// You et al. (2019),
// "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
// https://arxiv.org/abs/1904.00962
//
// Algorithm 2 applies Adam's coordinate-wise normalization, then rescales the
// complete update independently for every parameter tensor:
//
//   m_t = beta1*m_{t-1} + (1-beta1)*g_t
//   v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
//   r_t = m_hat/(sqrt(v_hat)+epsilon)
//   u_t = r_t + weight_decay*w_t
//   q_t = ||w_t||_2/||u_t||_2
//   w_{t+1} = w_t - lr*q_t*u_t
//
// A zero parameter or update norm uses q_t=1. The repository's historical
// API exposes trust_ratio_gamma; it is retained as a symmetric clamp
// q_t in [1/gamma, gamma]. Setting gamma=1 forces the Adam update, while a
// large value closely follows the unclamped paper recurrence.
//
// State per parameter: two tensors (m, v), both matching parameter shape.
// LAMB uses inherited Optimizer::lr so repository LR schedulers update the
// learning rate actually consumed by step().
// =========================================================================
class LAMB : public Optimizer {
public:
    // Existing public diagnostics retained for source compatibility.
    double beta1;
    double beta2;
    double epsilon;
    double beta1_corr;
    double beta2_corr;
    int t;  // next timestep, starts at 1
    double trust_ratio_gamma;
    double weight_decay;

    explicit LAMB(double lr = 1e-3,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-6,
                  double trust_ratio_gamma = 10.0,
                  double weight_decay = 0.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated mutators.
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_epsilon(double new_epsilon);
    void set_trust_ratio_gamma(double new_gamma);
    void set_weight_decay(double new_weight_decay);

    // Hyperparameter and state accessors.
    double get_lr() const { return Optimizer::lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_epsilon() const { return epsilon; }
    double get_trust_ratio_gamma() const { return trust_ratio_gamma; }
    double get_weight_decay() const { return weight_decay; }
    int get_step() const { return t; }

    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    bool get_m(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_v(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_last_trust_ratio(void* layer_ptr,
                              size_t param_idx,
                              double& out) const;

private:
    struct ParameterState {
        Tensor m;
        Tensor v;
    };

    std::map<void*, std::vector<ParameterState>> state_;
    std::map<void*, std::vector<double>> last_trust_ratios_;

    static void validate(double lr,
                         double beta1,
                         double beta2,
                         double epsilon,
                         double trust_ratio_gamma,
                         double weight_decay);
    static double l2_norm(const Tensor& tensor);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    double update_param(Tensor* param,
                        Tensor* grad,
                        ParameterState& state,
                        double beta1_correction,
                        double beta2_correction);
};

#endif
