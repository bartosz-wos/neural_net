#ifndef SCHEDULE_FREE_ADAMW_H
#define SCHEDULE_FREE_ADAMW_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// Schedule-Free AdamW
//
// Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024,
// "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682)
// NeurIPS 2024 best-paper nominee.
//
// Canonical reference implementation:
//   https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
//
// The "schedule-free" property: train for any number of steps with NO warmup,
// NO decay, NO lr schedule — and still match or beat tuned-schedule AdamW.
// The key idea is to maintain three coupled sequences per parameter:
//
//   x_k : the "eval point" — where the next forward/backward runs.
//   z_k : the "iterate" — the parameter that gets Adam-updated.
//   y_k : the "averaged" parameter — the convex combo returned to the user.
//         y_k = (1 - beta1) * z_k + beta1 * x_k
//
// In training mode the parameter `p` holds `y_k`; the user calls
// `optimizer.eval()` to swap to `x_k` for evaluation/inference, and
// `optimizer.train()` to swap back to `y_k`.
//
// Per-step update (in train mode, with parameter holding y_k = p):
//
//   bias_corr2 = 1 - beta2^(k+1)                       (k is the step count)
//   exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * g²
//   denom      = sqrt(exp_avg_sq / bias_corr2) + eps
//   u          = g / denom                              (with optional wd·y added)
//   z_{k+1}    = z_k - lr * u                          (the iterate step)
//   y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k + lr * (beta1 * (1-ckp1) - 1) * u
//                where ckp1 = 1/(k+1) when r=0, weight_lr_power=0
//                                (= (k+1)^r * lr_max^weight_lr_power / weight_sum)
//
// After the in-place updates above, `p` already equals y_{k+1}; no separate
// `x` storage is needed in `p` (we keep `x` only implicitly via `y` and `z`).
//
// At eval() time we swap p from y back to x using the closed-form
// relation y = (1-beta1)*z + beta1*x  ⇒  x = (y - (1-beta1)*z)/beta1.
// Implemented in-place as p.lerp_(z, weight=1-1/beta1) which gives the
// same value from the relationship y + w*(z-y) = x with w = 1-1/beta1.
//
// At train() time we swap p from x back to y via
// p.lerp_(z, weight=1-beta1) which gives (1-(1-beta1))*x + (1-beta1)*z = y.
//
// Defaults (per the reference implementation):
//   lr = 1.0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0,
//   warmup_steps = 0, r = 0, weight_lr_power = 2.0
//
// State per parameter: two tensors (z, exp_avg_sq), same shape as the parameter.
// =========================================================================
class ScheduleFreeAdamW : public Optimizer {
public:
    // --- Hyperparameters (public for inspection / test access) ---
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    int    warmup_steps;
    double r;               // polynomial weighting power in the average (default 0)
    double weight_lr_power; // warmup-time lr weighting power (default 2.0)

    explicit ScheduleFreeAdamW(double lr = 1.0,
                               double beta1 = 0.9,
                               double beta2 = 0.999,
                               double eps = 1e-8,
                               double weight_decay = 0.0,
                               int warmup_steps = 0,
                               double r = 0.0,
                               double weight_lr_power = 2.0);

    void step(Model& model) override;

    // Schedule-Free AdamW applies weight decay internally (coupled form
    // grad_normalized += weight_decay * y, matching PyTorch reference).
    bool handles_weight_decay() const override { return true; }

    // --- Mode switching (canonical reference API) ---
    // eval(Model&) : swap parameter from y to x for evaluation / inference.
    // train(Model&): swap parameter from x back to y for training.
    // We take the Model explicitly because in our codebase the optimizer
    // does not store a persistent reference to the model.
    void eval(Model& model);
    void train(Model& model);

    bool is_train_mode() const { return train_mode_; }

    // --- Validated mutators ---
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_eps(double new_eps);
    void set_weight_decay(double new_wd);
    void set_warmup_steps(int new_warmup);
    void set_r(double new_r);
    void set_weight_lr_power(double new_p);

    // --- Accessors ---
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_eps() const { return eps; }
    double get_weight_decay() const { return weight_decay; }
    int get_warmup_steps() const { return warmup_steps; }
    double get_r() const { return r; }
    double get_weight_lr_power() const { return weight_lr_power; }
    int get_k() const { return k_; }
    double get_lr_max() const { return lr_max_; }
    double get_weight_sum() const { return weight_sum_; }

    // --- State introspection ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    // Return a clone of the per-parameter `z` iterate. Empty tensor if no state.
    Tensor get_z(void* layer_ptr, size_t param_idx) const;
    // Return a clone of the per-parameter `exp_avg_sq` second moment.
    Tensor get_exp_avg_sq(void* layer_ptr, size_t param_idx) const;

private:
    struct ParameterState {
        Tensor z;            // iterate, same shape as parameter
        Tensor exp_avg_sq;   // Adam-style second moment, same shape as parameter
    };

    std::map<void*, std::vector<ParameterState>> state_;

    int k_ = 0;             // step counter (1-indexed when read)
    bool train_mode_ = false;
    double lr_max_ = -1.0;
    double weight_sum_ = 0.0;

    // Static validator for the constructor.
    static void validate(double lr, double beta1, double beta2,
                         double eps, double weight_decay,
                         int warmup_steps, double r, double weight_lr_power);

    // Lazy-init state for a layer's parameters.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update step (in-place on `param`).
    void update_param(Tensor* param, Tensor* grad, ParameterState& st,
                      double scheduled_lr, double ckp1);

    // Mode-swap helpers — operate on every (layer, param) in the model.
    // Called from the public eval()/train() overloads which flip the flag.

    // Compute ckp1 = weight / weight_sum (the eval-point mixing fraction).
    double compute_ckp1(double sched);
};

#endif
