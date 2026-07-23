#ifndef SCHEDULE_FREE_SGD_H
#define SCHEDULE_FREE_SGD_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// Schedule-Free SGD
//
// Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024,
// "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682)
// NeurIPS 2024 best-paper nominee.
//
// Canonical reference implementation:
//   https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/sgd_schedulefree.py
//
// Schedule-Free SGD is the SGD companion to Schedule-Free AdamW (already in
// this repo). It shares the same three-sequence coupling
//
//   x_k : the "eval point"  — where the next forward/backward runs.
//   z_k : the "iterate"     — the parameter that gets gradient-updated.
//   y_k : the "averaged" parameter — the convex combo returned to the user.
//         y_k = (1 - beta1) * z_k + beta1 * x_k
//
// and the same mode-swap API (`optimizer.train(model)` / `optimizer.eval(model)`).
//
// The ONLY difference from Schedule-Free AdamW: Schedule-Free SGD has NO
// second moment. There is no β2 and no `exp_avg_sq` EMA. The per-step update
// is the bare (coupled-decay) gradient step:
//
//   u          = g + weight_decay * y           (the "update direction")
//   z_{k+1}    = z_k - lr * u                   (the iterate step)
//   y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k + lr * (beta1 * (1-ckp1) - 1) * u
//                where ckp1 = weight / weight_sum with the same recurrence
//                as Schedule-Free AdamW (see schedule_free_adamw.h)
//
// State per parameter: ONE tensor (z) — same shape as the parameter.
// This makes Schedule-Free SGD the lightest-weight "schedule-free" optimizer
// in the repo — useful as the inner optimizer inside Lookahead, and the
// natural baseline to compare against Schedule-Free AdamW on the same
// warmup/warmup_steps/r/weight_lr_power schedule.
//
// Defaults (matching the canonical reference implementation):
//   lr = 1.0, beta1 = 0.9, weight_decay = 0,
//   warmup_steps = 0, r = 0, weight_lr_power = 2.0
// =========================================================================
class ScheduleFreeSGD : public Optimizer {
public:
    // --- Hyperparameters (public for inspection / test access) ---
    double lr;
    double beta1;
    double weight_decay;
    int    warmup_steps;
    double r;               // polynomial weighting power in the average (default 0)
    double weight_lr_power; // warmup-time lr weighting power (default 2.0)

    explicit ScheduleFreeSGD(double lr = 1.0,
                             double beta1 = 0.9,
                             double weight_decay = 0.0,
                             int warmup_steps = 0,
                             double r = 0.0,
                             double weight_lr_power = 2.0);

    void step(Model& model) override;

    // Schedule-Free SGD applies weight decay internally (the coupled form
    // `u = g + weight_decay * y`, matching the PyTorch reference). This means
    // WeightDecay wrappers should NOT re-apply decay when the inner optimizer
    // is Schedule-Free SGD — matches ScheduleFreeAdamW's behavior.
    bool handles_weight_decay() const override { return true; }

    // --- Mode switching (canonical reference API) ---
    void eval(Model& model);
    void train(Model& model);

    bool is_train_mode() const { return train_mode_; }

    // --- Validated mutators ---
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_weight_decay(double new_wd);
    void set_warmup_steps(int new_warmup);
    void set_r(double new_r);
    void set_weight_lr_power(double new_p);

    // --- Accessors ---
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
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

private:
    struct ParameterState {
        Tensor z;            // iterate, same shape as parameter
    };

    std::map<void*, std::vector<ParameterState>> state_;

    int k_ = 0;             // step counter (1-indexed when read)
    bool train_mode_ = false;
    double lr_max_ = -1.0;
    double weight_sum_ = 0.0;

    // Static validator for the constructor.
    static void validate(double lr, double beta1, double weight_decay,
                         int warmup_steps, double r, double weight_lr_power);

    // Lazy-init state for a layer's parameters.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update step (in-place on `param`).
    void update_param(Tensor* param, Tensor* grad, ParameterState& st,
                      double scheduled_lr, double ckp1);

    // Compute ckp1 = weight / weight_sum (the eval-point mixing fraction).
    double compute_ckp1(double sched);
};

#endif