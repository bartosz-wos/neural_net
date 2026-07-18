#ifndef RADAM_H
#define RADAM_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cmath>
#include <map>
#include <vector>

class Tensor;

// =========================================================================
// RAdam: "On the Variance of the Adaptive Learning Rate and Beyond"
// Liu, Jiang, He, Chen, Liu, Gao, Han (2020).
//   https://arxiv.org/abs/1908.03265
//
// RAdam is an Adam variant that fixes Adam's high-variance early-training
// step. Adam's adaptive denominator `1/sqrt(v̂)` blows up when v̂ is tiny
// (which it always is on the first few steps), causing a huge step + high
// variance. RAdam introduces a `ρ_t` term that tracks the "effective SMA
// length" of v_t and gates the adaptive LR:
//
//   m_t = β1 · m_{t-1} + (1 − β1) · g_t
//   v_t = β2 · v_{t-1} + (1 − β2) · g_t²              (same as Adam)
//   m̂_t = m_t / (1 − β1^t)
//
//   ρ_∞ = 2 / (1 − β2) − 1                           (max SMA length)
//   ρ_t = ρ_∞ − 2·t·β2^t / (1 − β2^t)                (current effective SMA length)
//
//   if ρ_t > 4:                                       # adaptive branch (warmup done)
//       l_t = sqrt(((ρ_t-4)(ρ_t-2) ρ_∞) / ((ρ_∞-4)(ρ_∞-2) ρ_t))
//       param = param − lr · (l_t · m̂_t / (sqrt(v̂_t) + ε) + wd · param)
//   else:                                             # SGD-like branch (warmup phase)
//       param = param − lr · (m̂_t + wd · param)
//
// Note `v̂_t = v_t / (1 − β2^t)` is only needed in the adaptive branch.
//
// Key insight: during the warmup phase (typically the first ~5 steps for
// β2=0.999), RAdam does NOT use the adaptive denominator — it falls back
// to pure SGD-momentum (with bias-corrected m̂). Only after ρ_t crosses 4
// does the adaptive LR activate, and even then it gets a `l_t < 1` damping
// factor that monotonically grows to 1 as t → ∞.
//
// State per parameter (lazy-initialized on first step()):
//   m: first-moment EMA, same shape as parameter
//   v: second-moment EMA, same shape as parameter
//
// Default hyperparameters (paper defaults):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8, wd = 0
//
// Validation:
//   β1 ∈ (0, 1), β2 ∈ (0, 1), ε > 0
// =========================================================================
class RAdam : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;

    explicit RAdam(double lr          = 1e-3,
                   double beta1_in    = 0.9,
                   double beta2_in    = 0.999,
                   double eps         = 1e-8,
                   double wd          = 0.0);

    void set_lr(double new_lr);
    void set_beta1(double new_b1);
    void set_beta2(double new_b2);
    void set_epsilon(double new_eps);
    void set_weight_decay(double new_wd);

    double get_lr()           const { return lr; }
    double get_beta1()        const { return beta1; }
    double get_beta2()        const { return beta2; }
    double get_epsilon()      const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t()            const { return t; }

    // Computed ρ_∞ from current β2. Useful for tests.
    double get_rho_infty() const;

    // Computed ρ_t for a given t (1-indexed). Useful for tests.
    // At t=1, ρ_1 < 4 (SGD-like branch).
    double get_rho_t(int t_step) const;

    // Computed l_t for a given t (1-indexed). Only meaningful when ρ_t > 4.
    double get_l_t(int t_step) const;

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_m(void* layer_ptr, size_t param_idx) const;
    Tensor get_v(void* layer_ptr, size_t param_idx) const;

private:
    struct RAdamState {
        Tensor m;
        Tensor v;
    };
    std::map<void*, std::vector<RAdamState>> state_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    void update_param(Tensor* param,
                      Tensor* grad,
                      RAdamState& st,
                      double lr_,
                      double b1,
                      double b2,
                      double eps,
                      double wd,
                      double b1_c,
                      double b2_c,
                      int t_step) const;

    static void validate(double b1, double b2, double eps);
};

#endif