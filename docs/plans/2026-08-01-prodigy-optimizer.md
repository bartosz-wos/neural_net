# Prodigy Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a Prodigy (Mishchenko 2024, https://arxiv.org/abs/2306.06101, ICML 2024) optimizer to the repo — the learning-rate-free Adam variant that builds on D-Adaptation by adding a coordinate-wise rescaling.

**Architecture:** A two-pass optimizer that maintains per-parameter `m` (first moment), `v` (second moment), `s` (L1 tracker), and `p0` (initial weights) plus a global `d` (D-estimate) that adapts throughout training. The first pass accumulates `delta_numerator += (d/d0) * dlr * <g, p0 - param>` and per-parameter EMAs of m, v, s. The second pass updates `d` from the accumulated numerator/denominator, then takes the Adam step with the adaptive `d*lr` pre-factor.

**Tech Stack:** C++17, Tensor class, follows the DAdaptation pattern (`include/nn/optimizers/dadaptation.{h,cpp}`).

**Location:** `include/nn/optimizers/prodigy.{h,cpp}` + `tests/test_prodigy.cpp` + Makefile + umbrella header.

---

## Algorithm reference (Mishchenko 2024, Algorithm 1, "Prodigy")

### Per-step, two-pass

**Pass 1 — accumulate denominators and update EMAs (per parameter, with current `d`):**
```
g = grad                                          (cached on parameter)
# State init (lazy, on first encounter):
#   m, v, s: zero tensors of param shape
#   p0: param at first sight (clone)

# EMA updates (Prodigy scales by d, not by d^2):
m_t = β1 * m_{t-1} + d * (1-β1) * g
v_t = β2 * v_{t-1} + d² * (1-β2) * g²

# L1-tracker (β3, default sqrt(β2))
s_t = β3 * s_{t-1} + (d/d0) * dlr * g           (safeguard_warmup=false)
# alternatively: s_t = β3 * s_{t-1} + (d/d0) * d * g  (safeguard_warmup=true)

# D-estimate numerator (per-parameter contribution)
delta_numerator += (d/d0) * dlr * <g, p0 - param>
# D-estimate denominator (per-parameter L1 contribution)
d_denom += sum(|s|)
```

**Update `d` (global, after all parameters processed):**
```
d_numerator *= β3                                         (carry over EMA)
global_d_numerator = d_numerator + delta_numerator

if d_denom > 0:
    d_hat = d_coef * global_d_numerator / d_denom
    if d == d0: d = max(d, d_hat)                        # initial bootstrap
    d_max = max(d_max, d_hat)                            # monotone ceiling
    d = min(d_max, d * growth_rate)                      # monotone, with growth cap
```

**Pass 2 — take Adam step with the new `d` (per parameter):**
```
dlr = d * lr * bias_correction (or simply d*lr if bias_correction off)
denom = sqrt(v) + d * eps
# Coupled WD (decouple=false): grad += wd * param
# Decoupled WD (decouple=true):  param *= (1 - dlr * wd)
param -= dlr * m / denom
```

---

## Defaults (paper / PyTorch reference)

- `lr = 1.0` — "Leave LR set to 1 unless you encounter instability"
- `β1 = 0.9`, `β2 = 0.999`, `β3 = sqrt(β2) ≈ 0.9995` (paper default)
- `eps = 1e-8`
- `weight_decay = 0`
- `d0 = 1e-6` (initial D-estimate; algorithm grows it)
- `d_coef = 1.0` (preferred tuning knob)
- `growth_rate = inf` (unbounded growth)
- `decouple = true` (AdamW-style decoupled WD — PyTorch default)
- `use_bias_correction = false` (paper default)
- `safeguard_warmup = false` (default)

---

## Validation (mirrors PyTorch reference)

- `d0 > 0`
- `lr > 0`
- `eps > 0`
- `0 ≤ β1 < 1`, `0 ≤ β2 < 1`
- `weight_decay ≥ 0`
- `d_coef > 0`
- `growth_rate ≥ 1.0`

---

## State per parameter (lazy-initialized on first step)

- `m` — first moment, same shape as param
- `v` — second moment, same shape as param
- `s` — L1 tracker, same shape as param
- `p0` — initial weights snapshot, same shape as param (cloned at first sight)

Plus global state:
- `d` — current D-estimate (initial `d0`)
- `d_max` — monotone ceiling for `d`
- `d_numerator` — running numerator_β3-EMA
- `d_hat` — last D candidate (for debugging)
- `k` — step counter, starts at 1

---

## TDD test plan

### Test groups (40+ focused checks)

**G1. Construction & defaults (8 checks)**
- `lr=1.0`, `β1=0.9`, `β2=0.999`, `β3=sqrt(0.999)`, `eps=1e-8`, `wd=0`, `d0=1e-6`, `d_coef=1.0`, `growth_rate=inf`, `k=1`, `decouple=true`, `use_bc=false`, `safeguard_warmup=false`
- Non-default constructor stores values
- `d` initial value equals `d0`
- `d_max` initial value equals `d0`
- `d_numerator` initial value is 0
- `d_hat` initial value is 0
- `get_*` accessors round-trip
- `handles_weight_decay() == true` (since decouple=true by default)

**G2. Validated setters (8 checks)**
- `set_lr` rejects lr ≤ 0
- `set_eps` rejects eps ≤ 0
- `set_beta1` rejects β1 ≥ 1 or β1 < 0
- `set_beta2` rejects β2 ≥ 1 or β2 < 0
- `set_d0` rejects d0 ≤ 0
- `set_d_coef` rejects d_coef ≤ 0
- `set_growth_rate` rejects growth_rate < 1.0
- `set_*` accept valid values

**G3. State initialization (3 checks)**
- `has_state(&layer, 0)` → false before step
- State initialized after first step (m, v, s, p0)
- `p0` after first step equals the weight at first sight (cloned)

**G4. D-estimate growth (5 checks)**
- D-estimate `d` monotonically non-decreasing (after step 1)
- `d_max` monotonically non-decreasing
- D-estimate grows when gradient is consistent (10 steps on a simple loss, d > d0 after step ≥ 1)
- `d_numerator` evolves per step (not constant)
- `d_hat` reported correctly (matches `d_coef * global_d_numerator / d_denom`)

**G5. Closed-form first step (5 checks)**
- Tiny model: Dense(2,1) with all-zero weights, grad=[1,1], lr=1, d0=1 → d=1 after bootstrap since d==d0
- After step 1: delta_numerator = sum(grad * (p0 - param)) = 0 (since p0 == param == 0), so d_hat = 0 → d stays at 1
- m_t = β1*0 + d*(1-β1)*g = d*0.1*g
- v_t = β2*0 + d²*(1-β2)*g² = d²*0.001*g²
- Check Adam-style step matches reference formula

**G6. Decoupled weight decay (3 checks)**
- With wd and decouple=true, param *= (1 - dlr * wd) at the end of step
- Coupled WD (decouple=false) modifies grad before the EMA
- Zero-grad step with wd shrinks params

**G7. End-to-end training (3 checks)**
- Linear regression `y=2x` with lr=1: loss decreases ≥ 50% over 200 steps
- Multi-layer model: each layer's state tracked independently
- D-estimate adapts upward on a converging training run

**G8. Multi-layer, multi-parameter (3 checks)**
- 2 Dense layers, each with weight + bias: 4 sets of state (m, v, s, p0 each)
- Independent `step` counter
- Determinism (two fresh instances produce bit-identical updates)

**G9. Bias correction (2 checks)**
- use_bias_correction=true: dlr = d * lr * sqrt(1-β2^k) / (1-β1^k)
- use_bias_correction=false: dlr = d * lr (default)

**G10. Edge cases (3 checks)**
- All-zero grad: d_denom == 0 → early return (no step)
- Single-parameter model (one layer, one param) works
- Empty model returns without crashing

**G11. Forward pass through D-Adaptation comparison (2 checks)**
- Tiger vs DAdaptation: compare against DAdaptation step for the same hyperparameters (signatures differ but both reduce loss)
- Prodigy (lr=1.0) vs DAdaptation (lr=1.0, d0=1e-6) follow same trajectory when d is clamped to d0

**G12. Non-vacuousness (3 mutation tests)**
- Stub out the `d_hat` bootstrap (`if d == d0, d = max(d, d_hat)`) → fails D-estimate growth test
- Drop the `d²` factor in v update → v differs from expected
- Drop the L1 tracker `s` update → d_denom stays 0
- Drop the `<g, p0 - param>` inner product → d_hat stays 0
- Drop the `(d/d0)` factor on the gradient scaling → m differs

---

## Implementation outline

### File 1: `include/nn/optimizers/prodigy.h`

```cpp
#ifndef PRODIGY_H
#define PRODIGY_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <limits>

class Prodigy : public Optimizer {
public:
    // hyperparameters
    double lr, beta1, beta2, beta3, epsilon, weight_decay;
    double d0, d_coef, growth_rate;
    bool decouple, use_bias_correction, safeguard_warmup;
    int k;

    // Constructor + validated setters + accessors
    Prodigy(double lr=1.0, double b1=0.9, double b2=0.999,
            double eps=1e-8, double wd=0.0,
            double d0_=1e-6, double d_coef_=1.0,
            double growth=std::numeric_limits<double>::infinity(),
            bool decouple_=true, bool use_bc=false,
            bool safeguard_warmup_=false,
            double beta3_=std::numeric_limits<double>::quiet_NaN());

    // Constructors implicitly set beta3 = sqrt(beta2) if NaN is passed.

    void set_lr(double);
    void set_beta1(double);
    void set_beta2(double);
    void set_beta3(double);
    void set_epsilon(double);
    void set_weight_decay(double);
    void set_d0(double);
    void set_d_coef(double);
    void set_growth_rate(double);
    void set_decouple(bool);
    void set_use_bias_correction(bool);
    void set_safeguard_warmup(bool);

    // Accessors
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_beta3() const { return beta3; }
    double get_epsilon() const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    double get_d0() const { return d0; }
    double get_d_coef() const { return d_coef; }
    double get_growth_rate() const { return growth_rate; }
    bool get_decouple() const { return decouple; }
    bool get_use_bias_correction() const { return use_bias_correction; }
    bool get_safeguard_warmup() const { return safeguard_warmup; }
    int get_k() const { return k; }
    double get_d() const { return d_; }
    double get_d_max() const { return d_max_; }
    double get_d_numerator() const { return d_numerator_; }
    double get_d_hat() const { return d_hat_; }

    void step(Model& model) override;

    bool handles_weight_decay() const override { return decouple; }

    // State accessors for testing
    bool has_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t idx) const;
    const Tensor& get_v(void* layer_ptr, size_t idx) const;
    const Tensor& get_s(void* layer_ptr, size_t idx) const;
    const Tensor& get_p0(void* layer_ptr, size_t idx) const;

private:
    struct ParamState {
        Tensor m, v, s, p0;
    };
    std::map<void*, std::vector<ParamState>> state_;

    double d_;             // current D-estimate
    double d_max_;         // monotone ceiling
    double d_numerator_;   // running numerator EMA
    double d_hat_;         // last candidate D

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    static void validate(double b1, double b2, double eps_in,
                        double d0_in, double d_coef_in, double growth_in);
};

#endif
```

### File 2: `include/nn/optimizers/prodigy.cpp`

Two-pass implementation following the algorithm. Bias correction uses `sqrt(1-β2^k) / (1-β1^k)` when `use_bias_correction=true`. Coupled WD is applied to the gradient at the start of pass 1; decoupled WD is applied to the parameter at the end of pass 2.

NaN handling: skip the step if `d_denom == 0` (no gradient flow), matching PyTorch reference.

### File 3: `tests/test_prodigy.cpp`

Run the G1–G12 test groups via the standard harness pattern. Use `gradient_check.h` style helper for the closed-form math. Include a small Linear regression smoke test (Dense(2,1) on y=2x, lr=1.0, d0=1e-6, 100 steps).

### File 4: Makefile + umbrella header

- Add `build/test_prodigy` compile rule
- Add to `tests:` deps line
- Add `echo && ./build/test_prodigy` to `run_tests:`
- Add `#include "nn/optimizers/prodigy.h"` to `include/nn/nn.h`

---

## Verification

1. `make tests` — compiles all 121+ test binaries
2. `make run_tests` — runs all stable suites; expect the new `Prodigy` suite to pass all 40+ checks
3. Aggregate suite continues to pass
