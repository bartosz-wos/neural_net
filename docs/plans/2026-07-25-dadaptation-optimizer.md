# DAdaptation Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add DAdaptation (Defazio & Mishchenko 2023, ICML 2023, https://arxiv.org/abs/2301.07733) — a learning-rate-free Adam variant that automatically determines its own step size. The user passes `lr=1.0` (or any constant) and the algorithm discovers the optimal distance-to-solution `D` and adapts `D · lr` as the effective step size.

**Architecture:** A `DAdaptAdam` optimizer implementing **Algorithm 5** of the paper (Adam with D-Adaptation). Per-parameter Adam state (`m`, `v`) plus a GLOBAL **dual-averaging state** (`z`, `s` scalars per parameter) used to estimate `D` from the empirical inner-product lower bound `d_hat = r / ((1 - √β2) * ||s||_1)`. State per parameter: 4 tensors — `m`, `v` (Adam), `z` (Adam-like accumulator weighted by `d·lr`), `s` (√β2-EMA of `d·lr·g`). Plus a single global scalar `D` maintained across all parameters.

**Tech Stack:** C++17, existing `Tensor`, `Optimizer`, `Model`, `Layer` infrastructure. Header in `include/nn/optimizers/dadaptation.{h,cpp}`; tests in `tests/test_dadaptation.cpp`. Plan: bite-sized, each task ≤5 min focused work.

---

## Source / Reference

- Paper: Defazio & Mishchenko 2023, *Learning-Rate-Free Learning by D-Adaptation*, ICML 2023, https://arxiv.org/abs/2301.07733
- Official PyTorch reference: https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adam.py
- Algorithm 5 (Adam with D-Adaptation), page 10 of the paper.

## Paper Algorithm (per parameter θ)

Per-step update for Adam with D-Adaptation:
```
m_{k+1} = β1 * m_k + (1 - β1) * d * lr * g_k
v_{k+1} = β2 * v_k + (1 - β2) * g_k²
denom = √v_{k+1} + ε
x_{k+1} = x_k - m_{k+1} / denom
```

**Learning-rate update (computed across ALL parameters, then fed back as `d`)**:
```
s_{k+1} = √β2 * s_k + (1 - √β2) * d * lr * g_k
r_{k+1} = √β2 * r_k + (1 - √β2) * d * lr * <g_k, s_k> / denom
d_hat = r_{k+1} / ((1 - √β2) * ||s_{k+1}||_1)
d_{k+1} = max(d_k, d_hat)
```

Where:
- `d` is a global scalar maintained across all parameters (initialized to `d0 > 0`, default `1e-6`)
- `lr` is the user-supplied base learning rate (paper default `1.0`, since the algorithm does the scaling)
- `<g, s>` is the inner product (computed element-wise: `Σ_i g_i * s_i / denom_i`)
- `||s||_1` is the L1 norm summed across all parameters' `s` tensors
- `r` and `s` use a √β2 EMA coefficient (different from β2)
- Decoupled weight decay (AdamW style): `θ *= (1 - lr * d * wd)` (the paper's `decouple` flag)

The key trick: `m` and `s` are integrated using the running estimate of `d · lr` (so the EMA contributes a running estimate of the optimum), and `d` is updated from the empirical inner-product bound which is provably a lower bound on `D` (the distance from initial to optimal params).

## Memory layout

Per parameter: 4 tensors same shape as the parameter (`m`, `v`, `z`, `s`). Plus 1 global scalar `d`, 1 global scalar `numerator_weighted = r`, 1 global scalar `step counter k`.

The four per-parameter tensors make this **2x Adam's state size** but enable the `D` estimation.

## Design choices

1. **No bias correction**: PyTorch reference implementation has `use_bias_correction` flag off by default. Match the paper (bias correction off by default — turn on via setter).
2. **Decoupled weight decay**: Default off (matching the paper, where `decouple=False` is the default in the PyTorch implementation). Configurable via `set_decouple(true)`. When off, weight decay is applied as `grad += decay * param` (coupled).
3. **`d0 > 0`**: Paper default `1e-6`. Validated > 0.
4. **`growth_rate`**: Optional upper bound on multiplicative growth per step (default `inf`). Used to mimic warmup when set to e.g. `1.02`.
5. **`layer_scale`**: Per-parameter multiplier on the effective LR (default `1.0` per param). Lets users scale LR per layer without breaking the global `D` estimate.
6. **API**: `DAdaptAdam(lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, d0=1e-6, growth_rate=inf, decouple=false, use_bias_correction=false)`. User can pass `lr=1.0` and the algorithm adapts.

## Why DAdaptation is well-tested

- The paper proves convergence `O(DG/√n)` with `D` learned automatically — no other tuning needed.
- Algorithm 5's `d_hat` formula is `r / ((1 - √β2) * ||s||_1)`. Both numerator (`r`) and denominator (`||s||_1`) come from the same √β2-EMA structure, so the division is well-conditioned (numerator and denominator both → 0 at the same rate).
- The `D` estimate is monotonically non-decreasing (`d_{k+1} = max(d_k, d_hat)`).

## File-by-File Plan

### Files to create

- `include/nn/optimizers/dadaptation.h`
- `include/nn/optimizers/dadaptation.cpp`
- `tests/test_dadaptation.cpp`

### Files to modify

- `include/nn/nn.h` — add `#include "optimizers/dadaptation.h"` after the signum/adam_mini block
- `Makefile` — add `build/test_dadaptation` build rule, add to `tests:` and `run_tests:` deps

---

## Tasks

### Task 1: Header skeleton with class declaration

Create `include/nn/optimizers/dadaptation.h` declaring the `DAdaptAdam` class with:
- All public configuration fields and validated setters
- Internal state structure (4 tensors per parameter + globals)
- Forward declaration of `Model` and `Tensor`
- `handles_weight_decay()` returning `true` when `decouple=true`, else `false`
- Documented algorithm summary in comments

### Task 2: RED test — defaults & validated setters

Add `tests/test_dadaptation.cpp` with tests T1-T2:
- T1: default constructor gives `lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, wd=0, d0=1e-6, decouple=false, use_bias_correction=false, growth_rate=inf, k=1`
- T2: validated setters throw on invalid inputs (`d0 ≤ 0`, `lr ≤ 0`, `eps ≤ 0`, `betas ∉ [0,1)`)

Run: must fail to compile (no header yet) → expected failure.

### Task 3: GREEN — constructor + setters (no step logic yet)

Implement `DAdaptAdam::DAdaptAdam(...)` and the validated setters in `dadaptation.cpp`. The `step()` method can be a stub that does nothing for now.

Run tests T1-T2 → should pass. Run full suite → ensure no regressions.

### Task 4: RED test — closed-form first step

Add T3: closed-form first step on a single Dense(2,2) layer with all-zero init, all-one gradients, lr=1.0, d0=1e-6.
Expected (by hand):
- `m_1 = (1-β1) · 1 · 1e-6 · 1 = (1-0.9) · 1e-6 = 1e-7` per element (shape (2,2))
- `v_1 = (1-β2) · 1² = 1e-3` per element
- `s_1 = (1-√β2) · 1 · 1e-6 · 1 = (1-√0.999) · 1e-6 ≈ 5.013e-10` per element
- `denom = √1e-3 + 1e-8 ≈ 0.031622776 + 1e-8 ≈ 0.031622786`
- `param_1 = 0 - 1e-7 / 0.031622786 ≈ -3.16228e-6` per element

Run: must fail (step is a stub) → expected failure.

### Task 5: GREEN — `step()` implementation

Implement `DAdaptAdam::step()` in two phases:
- **Phase 1** (per-parameter): compute `m`, `v`, `s`, accumulate `numerator_acum = Σ_r d*lr*<g, s>/denom>`, accumulate `sk_l1 = Σ_i ||s_i||_1`.
- **Phase 2** (global): compute `r = √β2 * r + (1-√β2) * numerator_acum`, compute `d_hat = r / ((1-√β2) * sk_l1)`, update `d = max(d, min(d_hat, d * growth_rate))`, save `r` to `numerator_weighted`.
- **Phase 3** (per-parameter): apply parameter update `param -= m / denom`. If `decouple && wd > 0`: `param *= (1 - lr * d * wd)`. Else if `wd > 0`: add `wd * param` to gradient BEFORE the step.

Run tests T3 → should pass.

### Task 6: RED test — `D` estimate is monotonically non-decreasing

Add T4: Run 5 steps on Dense(2,2) with constant gradient = 1.0. After each step, check `d` is non-decreasing (with `use_bias_correction=true` so the `D` estimate grows predictably).

Run: should fail (D not yet exposed) → expected failure.

### Task 7: GREEN — `get_d()` accessor

Add `get_d()` and `get_numerator_weighted()` accessors; add `get_z()` and `get_s()` per-parameter accessors.

Run T4 → should pass.

### Task 8: RED test — `step()` throws when gradients are zero (sk_l1 == 0)

Add T5: when all gradients are 0, the algorithm returns without updating (matches PyTorch's `if sk_l1 == 0: return loss` behavior). When at least one gradient is non-zero, parameter changes.

Run: should pass once step is implemented (since 0 grad → no update is the natural behavior of `s = √β2 * s + (1-√β2) · 0 = √β2 * s` which decays `s` toward 0). May need a guard in implementation.

### Task 9: RED test — decoupled weight decay

Add T6: with `decouple=true, wd=0.1, lr=1.0, d=d0=1e-6`, one step on Dense(2,2) with all-zero gradient, all-1 init params: `param *= (1 - lr * d * wd) = (1 - 1e-7) ≈ 0.9999999`.

Run: should fail (decoupled wd not yet implemented) → expected failure.

### Task 10: GREEN — decoupled weight decay

In step(), before computing the gradient accumulators, if `decouple && wd > 0`: `param *= (1 - lr * d * wd)`.

Run T6 → should pass.

### Task 11: RED test — coupled weight decay

Add T7: with `decouple=false, wd=0.1, lr=1.0, d=d0=1e-6`, one step on Dense(2,2) with all-zero gradient, all-1 init params: gradient becomes `0 + 0.1 * 1 = 0.1` per element. Then `m = (1-β1) * 0.1 * 1e-6 = 1e-8`. With `v = 0`, `denom = eps = 1e-8`. `param = 1 - 1e-8 / 1e-8 = 0` per element.

Run: should fail (coupled wd not yet implemented) → expected failure.

### Task 12: GREEN — coupled weight decay

In step(), if `!decouple && wd > 0`: add `wd * param` to gradient BEFORE computing `m`, `v`, `s`.

Run T7 → should pass.

### Task 13: RED test — parameter/gradient count mismatch

Add T8: manually construct a model with malformed param/grad arrays (or just verify by checking that mismatched shapes throw `std::logic_error`).

Run: should fail (no shape guards yet) → expected failure.

### Task 14: GREEN — shape guards

In step(), before the per-parameter loop, check `params.size() == grads.size()` and per-element `grad->rows == param->rows && grad->cols == param->cols`. Throw `std::logic_error` with descriptive message.

Run T8 → should pass.

### Task 15: RED test — determinism

Add T9: two fresh `DAdaptAdam` instances with same config, run 5 steps on identical Dense(2,2) with identical random gradients, verify all params and `d` end up bit-exact.

Run: should pass once step is implemented. (Determinism comes from no RNG in the algorithm itself.)

### Task 16: RED test — end-to-end training reduces loss

Add T10: train a Dense(1,1) model on `y = 2x` (synthetic) for 50 steps using `DAdaptAdam(lr=1.0)`. Initial MSE should drop significantly (>50% reduction).

Run: should pass.

### Task 17: RED test — signature vs Adam

Add T11: run both Adam(lr=0.001) and DAdaptAdam(lr=1.0) on identical gradient sequence for 10 steps. The two trajectories should differ (different effective LR scales). Verify L2 distance between updated params > some threshold.

Run: should pass.

### Task 18: RED test — bias correction toggle

Add T12: with `use_bias_correction=true`, after 1 step the parameter update is `m / (sqrt(v * (1-β2^2)) / (1-β1^2) + eps)` (the bias-corrected Adam form). Verify `d` grows faster than the non-corrected case.

Run: should pass.

### Task 19: RED test — `growth_rate` cap

Add T13: with `growth_rate=1.02`, verify `d_{k+1} ≤ 1.02 * d_k` per step.

Run: should pass.

### Task 20: Mutation testing

Run the following 4 mutations and verify at least 2 dedicated failures each:
1. Skip `m` update (keep `m` zero) → T3 fails
2. Skip `d_hat = max(d, d_hat)` → T4 fails
3. Use β2 instead of √β2 in `s` EMA → T3 fails
4. Skip weight decay entirely → T7 fails

### Task 21: Wire into nn.h and Makefile

- Add `#include "optimizers/dadaptation.h"` to `include/nn/nn.h` after `adam_mini.h`
- Add `$(BUILD_DIR)/test_dadaptation: $(LIB_OBJS) $(BUILD_DIR)/test_dadaptation.o` rule
- Add `$(BUILD_DIR)/test_dadaptation` to `tests:` deps
- Add `$(BUILD_DIR)/test_dadaptation` to `run_tests:` deps + echo + run command

### Task 22: Final verification

Run `make tests` (compile everything) and `make run_tests` (run everything). Confirm all 96+ tests pass.

---

## Pitfalls

1. **The `d · lr` weighting**: `m`, `s`, `r` all use `d · lr` as a scaling factor (not just `lr`). Easy to forget — it's the entire point of the algorithm.

2. **√β2 vs β2**: `s` and `r` use EMA coefficient `√β2`, NOT `β2`. Confusing these gives a wrong scaling for `d_hat` (the `(1-√β2)` correction term in the denominator breaks).

3. **Two-phase algorithm**: `d_hat` must be computed AFTER all per-parameter accumulators but BEFORE the parameter update. Otherwise `s` and `r` are corrupted by the `d` update. (Actually, the parameter update uses `m/denom`, not `s`, so the order between `d_hat` and `param -= m/denom` doesn't matter — but `d_hat` MUST come AFTER the per-param accumulator loop. Easy mistake.)

4. **L1 norm `||s||_1`**: sum across ALL parameters (not per-parameter). Compute across the whole model, then use the global sum in `d_hat`.

5. **`d_hat` can go negative**: paper explicitly says "Negative values of `d_hat` were seen in most of the experiments" — the `max(d, d_hat)` prevents it from corrupting `d`. Don't `abs(d_hat)` or assert `d_hat > 0`.

6. **Decoupled vs coupled WD interaction with `d`**: decoupled WD uses `lr * d` (not `lr`) as the effective scale. The `d` update happens BEFORE the decoupled WD on the next step. This is the same convention as PyTorch reference.

7. **`growth_rate=inf`**: this is the default. Implement as `growth_rate = std::numeric_limits<double>::infinity()` with `min(d_hat, d * growth_rate) = min(d_hat, inf) = d_hat` so the cap doesn't bind.

8. **bias_correction `d = d * lr * bias_correction`**: PyTorch applies this at the start of each step. With `use_bias_correction=false` (default), `bias_correction = 1` so `d * lr * 1 = d * lr` — matches the paper's formula.

## Non-vacuousness check

Each test will be mutation-tested:
- T3 (closed-form first step) — proves Adam updates are correct
- T4 (D monotonicity) — proves `max(d, d_hat)` is applied
- T6 (decoupled WD) — proves decoupled WD scaling by `d*lr`
- T7 (coupled WD) — proves coupled WD adds to gradient
- T9 (determinism) — proves no RNG in the algorithm
- T10 (training reduces loss) — proves end-to-end correctness
- T11 (signature vs Adam) — proves DAdaptation differs from Adam
- T12 (bias correction) — proves bias correction path works
- T13 (growth_rate cap) — proves cap is applied

## Verification

After implementation:
```bash
cd /home/stefan/neural_net
make tests                # compiles all 100+ test binaries
make run_tests            # executes them all
```

The focused suite `build/test_dadaptation` should report N/N passes with all checks at machine precision.