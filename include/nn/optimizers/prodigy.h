#ifndef PRODIGY_H
#define PRODIGY_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <limits>

class Model;

// ============================================================================
// Prodigy: "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
// Konstantin Mishchenko (2024), ICML 2024 (https://arxiv.org/abs/2306.06101).
//
// Prodigy is the learning-rate-free follow-up to D-Adaptation (Defazio &
// Mishchenko 2023). Like DAdapt-Adam, it maintains a running scalar `d`
// (an online lower bound on `D = ||x0 - x*||`, the distance from the initial
// weights to the optimum) and uses `d * lr` as the effective step size. The
// contribution of Prodigy over DAdaptation is twofold:
//
//   1. EMA scaling: the per-parameter m_t and v_t updates are SCALED by `d`.
//      D-Adaptation uses `d*lr` ONLY on the final step (pass 2). Prodigy
//      scales the first/second moments by `d` directly so the EMA inherits
//      the scale of the learning rate. Concretely:
//        m_t = β1 * m_{t-1} + d * (1 - β1) * g_t
//        v_t = β2 * v_{t-1} + d² * (1 - β2) * g_t²
//      (vs D-Adaptation which keeps m, v unweighted and adds d*lr in the
//      parameter step).
//
//   2. Same-rank inner product in the D-estimate numerator: instead of
//      using the per-parameter L1 tracker `s` for the denominator only,
//      Prodigy adds a (d/d0) * dlr * <g, p0 - param> term to the
//      numerator. This inner product is large when the gradient is
//      consistent with the historical direction "back toward p0", which
//      means we should grow `d` more aggressively.
//
// The result is faster convergence on a wider range of practical problems
// (the Prodigy paper reports 13-26% improvement over D-Adaptation on
// standard benchmarks without any tuning).
//
// === Algorithm (per Mishchenko 2024, Algorithm 1) ===
//
// PER PARAMETER (using current `d`):
//   m_t = β1 * m_{t-1} + d * (1 - β1) * g_t
//   v_t = β2 * v_{t-1} + d² * (1 - β2) * g_t²
//   s_t = β3 * s_{t-1} + (d/d0) * dlr * g_t        [safeguard_warmup=false]
//   s_t = β3 * s_{t-1} + (d/d0) * d    * g_t       [safeguard_warmup=true]
//
//   delta_numerator += (d/d0) * dlr * <g, p0 - param>
//
//   d_denom += Σ_ij |s_ij|
//
// (also clamps the per-parameter pre-factor (d/d0) so the L1 tracker
// doesn't explode when d >> d0.)
//
// GLOBAL D ESTIMATE (after all parameters processed):
//   d_numerator *= β3
//   global_d_numerator = d_numerator + delta_numerator
//   d_hat = d_coef * global_d_numerator / d_denom           (if d_denom > 0)
//   if d == d0: d = max(d, d_hat)                            (initial bootstrap)
//   d_max = max(d_max, d_hat)
//   d = min(d_max, d * growth_rate)                          (monotone)
//
// PARAMETER UPDATE (pass 2):
//   dlr = d * lr * bias_correction  (or just d * lr if use_bias_correction=false)
//   denom = sqrt(v_t) + d * eps
//   if decouple: param *= (1 - dlr * weight_decay)
//   param -= dlr * m_t / denom
//
// === Defaults (Mishchenko 2024 / PyTorch reference) ===
//   lr = 1.0                       — "Leave LR set to 1 unless you encounter instability"
//   beta1 = 0.9, beta2 = 0.999
//   beta3 = sqrt(beta2) ≈ 0.9995
//   eps = 1e-8
//   weight_decay = 0
//   d0 = 1e-6                      — initial D-estimate (algorithm grows it)
//   d_coef = 1.0                   — preferred tuning knob
//   growth_rate = inf               — unbounded growth (no warmup cap)
//   decouple = true                — AdamW-style decoupled weight decay
//   use_bias_correction = false    — no Adam bias correction (matches paper)
//   safeguard_warmup = false
//
// === Validation (from PyTorch reference constructor) ===
//   lr > 0, eps > 0, d0 > 0, d_coef > 0, growth_rate >= 1.0
//   0 ≤ beta1 < 1, 0 ≤ beta2 < 1
//   weight_decay >= 0
//
// === State per parameter (lazy on first step) ===
//   m, v, s — same shape as param
//   p0      — initial weights snapshot (cloned at first sight)
//
// === Global state ===
//   d_             — current D-estimate (starts at d0)
//   d_max_         — monotone ceiling for `d`
//   d_numerator_   — running numerator EMA (carries β3 multiplier across steps)
//   d_hat_         — last candidate D (for introspection)
//   k              — step counter, starts at 1
//
// === Public API ===
//   Prodigy(...)                  — constructor with defaults
//   set_lr / set_beta1 / set_beta2 / set_beta3 / set_epsilon / set_weight_decay
//   set_d0 / set_d_coef / set_growth_rate
//   set_decouple / set_use_bias_correction / set_safeguard_warmup
//   get_* (l/r/beta1/beta2/beta3/epsilon/wd/d0/d_coef/growth/decouple/...)
//   get_d / get_d_max / get_d_numerator / get_d_hat   — global state
//   get_k                                           — step counter
//   step(model)
//   handles_weight_decay() → true iff decouple=true
//   has_state(&layer_ptr) / get_m / get_v / get_s / get_p0   — state accessors
//
// === Reference ===
//   Mishchenko, K. "Prodigy: An Expeditiously Adaptive Parameter-Free
//   Learner." ICML 2024. https://arxiv.org/abs/2306.06101
//   PyTorch reference: https://github.com/konstmish/prodigy
// ============================================================================
class Prodigy : public Optimizer {
public:
    // --- Public hyperparameters (settable via constructor or mutators) ---
    double lr;                       // base learning rate (default 1.0)
    double beta1;                    // first-moment EMA coefficient (default 0.9)
    double beta2;                    // second-moment EMA coefficient (default 0.999)
    double beta3;                    // L1-tracker EMA coefficient (default sqrt(beta2))
    double epsilon;                  // numerical stabilizer in denom (default 1e-8)
    double weight_decay;             // AdamW-style decoupled WD (default 0)
    double d0;                       // initial D-estimate (default 1e-6)
    double d_coef;                   // D-update scaling factor (default 1.0)
    double growth_rate;              // multiplicative cap on `d` growth per step (default inf)
    bool   decouple;                 // AdamW-style decoupled WD (default true)
    bool   use_bias_correction;      // Adam bias correction on/off (default false)
    bool   safeguard_warmup;         // use `d` instead of `dlr` in L1 tracker (default false)

    // --- Step counter (paper convention: starts at 1) ---
    int k;

    // --- Constructor with paper defaults ---
    explicit Prodigy(
        double lr_             = 1.0,
        double b1              = 0.9,
        double b2              = 0.999,
        double eps             = 1e-8,
        double wd              = 0.0,
        double d0_             = 1e-6,
        double d_coef_         = 1.0,
        double growth          = std::numeric_limits<double>::infinity(),
        bool   decouple_       = true,
        bool   use_bc          = false,
        bool   safeguard_warmup_ = false,
        double beta3_          = -1.0); // -1 → derive from sqrt(beta2) at construction

    // --- Validated setters (throw std::invalid_argument on invalid input) ---
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_beta3(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);
    void set_d0(double v);
    void set_d_coef(double v);
    void set_growth_rate(double v);
    void set_decouple(bool v);
    void set_use_bias_correction(bool v);
    void set_safeguard_warmup(bool v);

    // --- Accessors ---
    double get_lr()                  const { return lr; }
    double get_beta1()               const { return beta1; }
    double get_beta2()               const { return beta2; }
    double get_beta3()               const { return beta3; }
    double get_epsilon()             const { return epsilon; }
    double get_weight_decay()        const { return weight_decay; }
    double get_d0()                  const { return d0; }
    double get_d_coef()              const { return d_coef; }
    double get_growth_rate()         const { return growth_rate; }
    bool   get_decouple()            const { return decouple; }
    bool   get_use_bias_correction() const { return use_bias_correction; }
    bool   get_safeguard_warmup()    const { return safeguard_warmup; }
    int    get_k()                   const { return k; }

    // --- Global D-estimate state accessors ---
    double get_d()           const { return d_; }
    double get_d_max()       const { return d_max_; }
    double get_d_numerator() const { return d_numerator_; }
    double get_d_hat()       const { return d_hat_; }

    // --- Optimizer interface ---
    void step(Model& model) override;

    // If decouple=true we apply decoupled weight decay internally.
    // Coupled weight decay is NOT internal (it modifies the gradient).
    bool handles_weight_decay() const override { return decouple; }

    // --- State introspection (for tests + debugging) ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    size_t num_params_with_state(void* layer_ptr) const;

    // Retrieve the m, v, s, or p0 tensor for (layer, param_idx).
    // Returns an empty (0,0) Tensor if (layer, param_idx) has not been seen.
    const Tensor& get_m(void* layer_ptr, size_t param_idx)  const;
    const Tensor& get_v(void* layer_ptr, size_t param_idx)  const;
    const Tensor& get_s(void* layer_ptr, size_t param_idx)  const;
    const Tensor& get_p0(void* layer_ptr, size_t param_idx) const;

private:
    // --- Per-parameter state ---
    struct ParamState {
        Tensor m;   // first moment, shape (rows, cols)
        Tensor v;   // second moment, shape (rows, cols)
        Tensor s;   // L1 tracker, shape (rows, cols)
        Tensor p0;  // initial weights snapshot, shape (rows, cols)
    };
    std::map<void*, std::vector<ParamState>> state_;

    // --- Global D-estimate state ---
    double d_;             // current D-estimate (lower bound on ||x0 - x*||)
    double d_max_;         // monotone ceiling for `d`
    double d_numerator_;   // running numerator EMA
    double d_hat_;         // last candidate D (for introspection)

    // --- Helpers ---
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    static void validate(double lr_, double b1, double b2, double eps,
                        double d0_, double d_coef_, double growth);

    // Per-parameter update. Pass 1 (m, v, s, delta_numerator); pass 2 (denom + step).
    struct PassAccumulators {
        double delta_numerator;  // per-step D-estimate numerator contribution
        double d_denom;          // per-step D-estimate denominator contribution
    };
    PassAccumulators update_param_pass1(
        ParamState& st, const Tensor* param, const Tensor* grad,
        double dlr, double bc_factor);
    void update_param_pass2(
        ParamState& st, Tensor* param, const Tensor* grad,
        double dlr);
};

#endif
