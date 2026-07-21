# Schedule-Free AdamW Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the Schedule-Free AdamW optimizer (Defazio et al. 2024, NeurIPS best-paper nominee) — a "no learning rate schedule required" AdamW variant — in `include/nn/optimizers/schedule_free_adamw.{h,cpp}`, with full TDD coverage and mutation testing.

**Architecture:** Schedule-Free maintains three coupled sequences per parameter (`x`, `y`, `z`) where `y` is the averaged parameter held in `param` during training, `z` is the iterate that gets Adam-updated, and `x` is the eval point (held in `param` during evaluation). The state per parameter is `(z, exp_avg_sq)` — two tensors, similar to AdamW's footprint. The mode-swap API `optimizer.train(model)` / `optimizer.eval(model)` flips which sequence is in `param`. No LR schedule is required; convergence is guaranteed for any number of steps via the convex mixing coefficient `ckp1 = 1/(k+1)`.

**Tech Stack:** C++, custom assertion harness, repo Makefile build system, TDD with mutation testing per the systematic-debugging skill.

---

## Reference

- Paper: Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024, "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682), NeurIPS 2024 best-paper nominee
- Reference implementation: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py

## Algorithm

### State per parameter

- `z` — iterate (same shape as parameter)
- `exp_avg_sq` — Adam-style second moment EMA (same shape as parameter)
- `y_k = (1 - beta1) * z_k + beta1 * x_k` — averaged parameter (held in `param` during training)
- `x_k` — eval point (held in `param` during evaluation)

### State init (at first step)

- `z = clone(param)` (preserves initial value)
- `exp_avg_sq = 0`

### Per-step update (in train mode; `param` holds `y`)

At step k (k = 0-indexed; current step is k+1):

```
sched        = (k < warmup_steps) ? (k+1)/warmup_steps : 1
scheduled_lr = lr * sched
lr_max       = max(scheduled_lr, lr_max)
weight       = (k+1)^r * lr_max^weight_lr_power
weight_sum  += weight
ckp1         = weight / weight_sum
bias_corr2   = 1 - beta2^(k+1)
exp_avg_sq   = beta2 * exp_avg_sq + (1-beta2) * g^2
denom        = sqrt(exp_avg_sq / bias_corr2) + eps
u            = g / denom + (weight_decay > 0 ? weight_decay * y : 0)
z_{k+1}      = z_k - scheduled_lr * u
y_{k+1}      = ckp1 * z_{k+1} + (1-ckp1) * y_k + scheduled_lr * (beta1 * (1-ckp1) - 1) * u
```

### Mode swaps (canonical reference)

- `eval(model)`: p currently holds `y`. Compute `x` via the closed-form:
  `y = (1-beta1)*z + beta1*x  ⇒  x = (y - (1-beta1)*z)/beta1`
  Implemented in-place as `p.lerp_(z, weight=1 - 1/beta1)`.
- `train(model)`: p currently holds `x`. Compute `y`:
  `y = beta1*x + (1-beta1)*z`
  Implemented in-place as `p.lerp_(z, weight=1-beta1)`.

### Public API

```cpp
class ScheduleFreeAdamW : public Optimizer {
public:
    double lr, beta1, beta2, eps, weight_decay;
    int    warmup_steps;
    double r, weight_lr_power;

    explicit ScheduleFreeAdamW(double lr = 1.0, double beta1 = 0.9,
                               double beta2 = 0.999, double eps = 1e-8,
                               double weight_decay = 0.0,
                               int warmup_steps = 0,
                               double r = 0.0,
                               double weight_lr_power = 2.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    void eval(Model& model);
    void train(Model& model);
    bool is_train_mode() const { return train_mode_; }

    void set_lr(double); void set_beta1(double); void set_beta2(double);
    void set_eps(double); void set_weight_decay(double);
    void set_warmup_steps(int); void set_r(double); void set_weight_lr_power(double);

    double get_lr() const; double get_beta1() const; double get_beta2() const;
    double get_eps() const; double get_weight_decay() const;
    int get_warmup_steps() const; double get_r() const; double get_weight_lr_power() const;
    int get_k() const; double get_lr_max() const; double get_weight_sum() const;

    bool has_state(void* layer_ptr) const;
    Tensor get_z(void* layer_ptr, size_t param_idx) const;
    Tensor get_exp_avg_sq(void* layer_ptr, size_t param_idx) const;
};
```

---

## Implementation Tasks

### Task 1: Header file with public API

**Files:** Create `include/nn/optimizers/schedule_free_adamw.h`

**Step 1:** Write the header. ~165 lines. Document each public method with a one-line comment.

**Step 2:** Run `g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c include/nn/optimizers/schedule_free_adamw.cpp -o /tmp/x.o` (the .cpp doesn't exist yet — expect a "file not found" — that confirms the path is correct).

### Task 2: Implementation skeleton with validation + setters

**Files:** Create `include/nn/optimizers/schedule_free_adamw.cpp`

**Step 1:** Implement constructor, `validate()`, and 8 setters (lr, beta1, beta2, eps, wd, warmup_steps, r, weight_lr_power).

**Step 2:** Compile: `g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c include/nn/optimizers/schedule_free_adamw.cpp -o /tmp/x.o`. Expected: no errors.

### Task 3: ensure_state() and accessors (state init)

**Files:** Modify `include/nn/optimizers/schedule_free_adamw.cpp`

**Step 1:** Implement `ensure_state()` that initializes `z = clone(param)`, `exp_avg_sq = 0`. Plus `get_z()`, `get_exp_avg_sq()`, `has_state()`.

**Step 2:** Compile clean.

### Task 4: Per-parameter update step (`update_param`)

**Files:** Modify `include/nn/optimizers/schedule_free_adamw.cpp`

**Step 1:** Implement `update_param` exactly as documented in the algorithm above. Cache `z_old` for the y-update step.

**Step 2:** Compile clean.

### Task 5: Public `step()` method

**Files:** Modify `include/nn/optimizers/schedule_free_adamw.cpp`

**Step 1:** Implement `step()` that computes `sched`, `scheduled_lr`, `lr_max`, `weight_sum`, `ckp1`, walks model layers, calls `ensure_state()` and `update_param()` per parameter, zeros grads, increments `k_`. Throws if not in train mode.

**Step 2:** Compile clean.

### Task 6: Mode-swap methods (`eval`, `train`)

**Files:** Modify `include/nn/optimizers/schedule_free_adamw.cpp`

**Step 1:** Implement `eval(Model&)` and `train(Model&)` with the lerp formulas.

**Step 2:** Compile clean.

### Task 7: Register in umbrella header

**Files:** Modify `include/nn/nn.h`

**Step 1:** Add `#include "optimizers/schedule_free_adamw.h"` after `came.h`.

### Task 8: Test file

**Files:** Create `tests/test_schedule_free_adamw.cpp`

**Step 1:** Write 21 test sections (T1-T21) covering: defaults, validation throws, state init, closed-form first step, EMA recurrence, bias correction in denom, y vs z distinctness, eval/train mode swap (closed-form verification), coupled weight decay, malformed layer guard, determinism, end-to-end linear regression, signature test vs vanilla AdamW, ckp1 progression, warmup schedule, independent state across layers, independent state across parameters in same layer, gradient clearing, dense layer integration, r/weight_lr_power effects, state accessors before step, empty layer skip.

**Step 2:** Compile and run. Expected: 110/110 pass.

### Task 9: Mutation testing (5 mutations)

**Step 1:** Apply each mutation, run tests, count failures, restore. Mutations:
1. Drop y update → T4 + T6 + T7 + T8 + T11 + T18 + T21 should fail (≥ 7).
2. Skip bias correction in denom → T5b should fail (3 tests).
3. Wrong lerp weight in eval (use 1-beta1 instead of 1-1/beta1) → T7 should fail (2 tests).
4. Wrong lerp weight in train (use 1-1/beta1 instead of 1-beta1) → T7 should fail (2 tests).
5. Drop exp_avg_sq EMA (use raw g²) → T3 + T5 + T5b + T18 should fail (≥ 6 tests).

**Step 2:** Restore all mutations; final test run shows 110/110 pass.

### Task 10: Register in Makefile

**Files:** Modify `Makefile`

**Step 1:** Add `$(BUILD_DIR)/test_schedule_free_adamw: $(LIB_OBJS) $(BUILD_DIR)/test_schedule_free_adamw.o` rule.
**Step 2:** Add to `tests:` dependency list.
**Step 3:** Add `@echo "=== Running Schedule-Free AdamW Tests ===" && ./$(BUILD_DIR)/test_schedule_free_adamw` to `run_tests:` target.

### Task 11: Commit and push

```bash
git add include/nn/optimizers/schedule_free_adamw.h
git add include/nn/optimizers/schedule_free_adamw.cpp
git add tests/test_schedule_free_adamw.cpp
git add include/nn/nn.h
git add Makefile
git add EXPANSION_QUEUE.md  # move entry to Done
git commit -m "feat(optimizers): add Schedule-Free AdamW (Defazio 2024, NeurIPS), 110/110 tests pass"
git push origin master
```

---

## Test design notes

### Why closed-form tests are important

The optimizer is a chain of small operations. Naive "loss goes down" tests would pass even if several operations were subtly broken (e.g., wrong lerp weight, missing bias correction). Closed-form tests at each step verify the operation is *exactly* right.

### Why mode-swap is non-trivial

The schedule-free trick is that `y` (held in `param` during training) is a convex combination of `z` and `x`. The mode swap does NOT save/restore — it computes one sequence from the others via closed-form. If the lerp weight is wrong, the test catches it.

### Why bias correction matters

`bc2 = 1 - beta2^(k+1)` increases the effective `denom` by `1/sqrt(bc2)` early in training. Without it the optimizer is identical to plain AdamW at the gradient-scaling level. Test 5b verifies the bias correction is actually applied by checking step-by-step z values against the closed-form.

---

## Verification checklist

- [ ] `make build/test_schedule_free_adamw` succeeds
- [ ] `./build/test_schedule_free_adamw` returns 110/110
- [ ] All 5 mutations caught by ≥ 2 tests each
- [ ] `make tests` succeeds (no regressions in other 70+ test suites)
- [ ] `make run_tests` runs the new test and all others pass
- [ ] Git commit + push succeeds
