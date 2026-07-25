#ifndef DADAPTATION_H
#define DADAPTATION_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <limits>

// DAdaptation: "Learning-Rate-Free Learning by D-Adaptation"
// Defazio & Mishchenko 2023 (https://arxiv.org/abs/2301.07733), ICML 2023
//
// KEY IDEA: instead of asking the user to tune a learning rate, automatically
// learn the distance `D = ||x0 - x*||` from the initial weights to the optimal
// weights. The effective step size is `d * lr` where `d` is a maintained lower
// bound on `D` that grows whenever the empirical estimate exceeds the current
// bound. This is provably optimal (paper Theorem 1: `f(x_hat) - f* = O(DG/sqrt(n))`).
//
// We implement Algorithm 5 (Adam with D-Adaptation) from the paper. Per step:
//
//   PER PARAMETER (with current `d`):
//     m_{k+1} = beta1 * m_k + (1-beta1) * d * lr * g_k      (first moment, weighted by d*lr)
//     v_{k+1} = beta2 * v_k + (1-beta2) * g_k^2             (second moment, unweighted)
//     s_{k+1} = sqrt(beta2) * s_k + (1-sqrt(beta2)) * d * lr * g_k   (L1-norm tracking)
//     denominator = sqrt(v_{k+1}) + eps
//     numerator_acum += d*lr * dot(g, s / denominator)      (inner product for D estimate)
//     sk_l1 += ||s_{k+1}||_1                                  (L1 norm, summed globally)
//     x_{k+1} = x_k - m_{k+1} / denominator                  (Adam step)
//
//   GLOBAL D UPDATE (after all parameters processed):
//     r = sqrt(beta2) * r_prev + (1-sqrt(beta2)) * numerator_acum
//     d_hat = r / ((1-sqrt(beta2)) * sk_l1)
//     d = max(d, min(d_hat, d * growth_rate))               (monotone non-decreasing)
//
// State per parameter: 4 tensors (m, v, z, s) same shape as the parameter.
// Plus 1 global scalar `d`, 1 global scalar `r` (a.k.a. numerator_weighted),
// 1 step counter k.
//
// "z" is the unweighted g accumulator (used for the optional layer-scale
// tracking — see Appendix F). It is not actually needed for the DAdaptation
// algorithm itself; we omit it from state (the paper's algorithm 5 has only m,
// v, s). Note: PyTorch reference keeps `s` only, no `z` — we match that.
//
// Bias correction (Adam-style):
//   use_bias_correction=false  → d_effective = d * lr * 1             (paper default)
//   use_bias_correction=true   → d_effective = d * lr * sqrt(1-beta2^k) / (1-beta1^k)
//
// Weight decay:
//   decouple=false (paper default) → grad += wd * param               (coupled, Adam-style)
//   decouple=true                 → param *= (1 - d * lr * wd)       (decoupled, AdamW-style)
//
// "Leave LR set to 1 unless you encounter instability" — the algorithm does
// the scaling via `d`. Set `growth_rate=1.02` to enforce a soft warmup cap.
//
// Recommended from the paper:
//   lr = 1.0 (paper default)
//   d0 = 1e-6 (paper default — algorithm grows it to true D)
//   beta1 = 0.9, beta2 = 0.999, eps = 1e-8 (Adam defaults)
class DAdaptAdam : public Optimizer {
public:
    double lr;                    // base learning rate (user-tunable; paper default 1.0)
    double beta1;                 // first-moment EMA coefficient
    double beta2;                 // second-moment EMA coefficient
    double epsilon;               // denominator stabilizer
    double weight_decay;          // coupled or decoupled weight decay
    double d0;                    // initial lower bound on D (paper default 1e-6)
    double growth_rate;           // multiplicative cap on d growth per step (paper default inf)
    bool   decouple;              // decoupled (AdamW) vs coupled (Adam) weight decay
    bool   use_bias_correction;   // Adam bias correction on/off (paper default off)
    int    k;                     // step counter (starts at 1)

    // Constructor with sensible defaults matching the paper / PyTorch reference
    DAdaptAdam(double lr_ = 1.0,
               double b1 = 0.9,
               double b2 = 0.999,
               double eps = 1e-8,
               double wd = 0.0,
               double d0_ = 1e-6,
               double growth = std::numeric_limits<double>::infinity(),
               bool decouple_ = false,
               bool use_bc = false);

    // Validated setters
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);
    void set_d0(double v);
    void set_growth_rate(double v);
    void set_decouple(bool v);
    void set_use_bias_correction(bool v);

    void step(Model& model) override;

    // If decouple=true we apply decoupled WD internally; coupled WD is NOT
    // internal (it modifies the gradient). PyTorch reference convention.
    bool handles_weight_decay() const override { return decouple; }

    // State accessors
    bool        has_state(void* layer_ptr) const;
    size_t      num_params_with_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_v(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_s(void* layer_ptr, size_t param_idx) const;

    double get_d() const { return d_; }
    double get_numerator_weighted() const { return numerator_weighted_; }
    int    get_step() const { return k; }

private:
    struct ParamState {
        Tensor m;  // first moment
        Tensor v;  // second moment
        Tensor s;  // L1-norm tracker (sqrt-beta2 EMA of d*lr*g)
    };
    std::map<void*, std::vector<ParamState>> state_;

    double d_;                       // current lower bound on D (running scalar)
    double numerator_weighted_;      // `r` from the paper

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

#endif