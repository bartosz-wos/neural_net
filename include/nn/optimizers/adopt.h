#ifndef ADOPT_H
#define ADOPT_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

class Model;

// ADOPT — Taniguchi et al., NeurIPS 2024
// "ADOPT: Modified Adam Can Converge with Any beta2 with the Optimal Rate"
// https://arxiv.org/abs/2411.02853
// Official reference: https://github.com/iShohei220/adopt
//
// Unlike Adam, ADOPT normalizes the current gradient with the PREVIOUS second
// moment before updating momentum. The first step only initializes v = g^2.
// For t >= 2:
//   g_norm = clip(g / max(sqrt(v_{t-1}), eps), +/-(t-1)^clip_exp)
//   m_t = beta1*m_{t-1} + (1-beta1)*g_norm
//   theta_t = theta_{t-1} - lr*m_t
//   v_t = beta2*v_{t-1} + (1-beta2)*g^2
class ADOPT : public Optimizer {
public:
    explicit ADOPT(double lr = 1e-3,
                   double beta1 = 0.9,
                   double beta2 = 0.9999,
                   double epsilon = 1e-6,
                   double clip_exp = 0.25,
                   double weight_decay = 0.0,
                   bool decoupled = false);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    void set_lr(double value);
    void set_beta1(double value);
    void set_beta2(double value);
    void set_epsilon(double value);
    void set_clip_exp(double value);
    void set_weight_decay(double value);
    void set_decoupled(bool value) { decoupled_ = value; }

    double get_lr() const { return Optimizer::lr; }
    double get_beta1() const { return beta1_; }
    double get_beta2() const { return beta2_; }
    double get_epsilon() const { return epsilon_; }
    double get_clip_exp() const { return clip_exp_; }
    double get_weight_decay() const { return weight_decay_; }
    bool get_decoupled() const { return decoupled_; }
    size_t get_t() const { return t_; }

    bool has_state(void* layer_ptr) const;
    size_t num_params_with_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_v(void* layer_ptr, size_t param_idx) const;

private:
    struct ParamState {
        Tensor m;
        Tensor v;
    };

    double beta1_;
    double beta2_;
    double epsilon_;
    double clip_exp_;
    double weight_decay_;
    bool decoupled_;
    size_t t_;
    std::map<void*, std::vector<ParamState>> state_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

#endif
