# CAME Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the CAME optimizer (Luo et al. 2023, ACL) — a confidence-guided adaptive optimizer with sublinear memory footprint — in `include/nn/optimizers/came.{h,cpp}`, with full TDD coverage and mutation testing.

**Architecture:** Following the canonical reference implementation (https://github.com/yangluo7/CAME/blob/master/came_pytorch/CAME.py), CAME uses a factored row/column EMA of `g²` to approximate the Adam adaptive denominator, plus a confidence-guided residual correction. For 2-D parameters the state is `(exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_res_row, exp_avg_res_col, RMS)` (similar memory to Adafactor). For 1-D parameters the residual correction is identity.

**Tech Stack:** C++, GoogleTest-style custom assertion harness, repo Makefile build system, TDD with mutation testing per the systematic-debugging skill.

---

## Reference

- Paper: Luo, Ren, Zheng, Jiang, Jiang, You 2023, "CAME: Confidence-guided Adaptive Memory Efficient Optimization" (https://arxiv.org/abs/2307.02047), ACL 2023 Long Papers, pages 4442-4453
- Reference implementation: https://github.com/yangluo7/CAME/blob/master/came_pytorch/CAME.py

## Algorithm (canonical reference implementation)

For each parameter `p`, gradient `g`, hyperparameters `β = (β1, β2, β3)`, `ε = (ε1, ε2)`, `clip_threshold`, `weight_decay`:

### Standard Adam-like adaptive update (step 1-3)

```
1. raw = g² + ε1                                     # pre-modulated raw squared gradient
   if 2-D (factored):
       row_EMA = β2 · row_EMA + (1-β2) · mean(raw, axis=cols)    # R^{d1×1}
       col_EMA = β2 · col_EMA + (1-β2) · mean(raw, axis=rows)    # R^{1×d2}
       update = g / sqrt(row_EMA · col_EMA / mean(row_EMA))      # reconstructed v_t
   else (1-D, non-factored):
       exp_avg_sq = β2 · exp_avg_sq + (1-β2) · raw
       update = g / sqrt(exp_avg_sq)

2. update /= max(1, RMS(update) / clip_threshold)    # RMS-clip stability

3. exp_avg = β1 · exp_avg + (1-β1) · update          # Adam-style momentum
```

### Confidence-guided residual correction (step 4-5)

```
4. res = (update - exp_avg)² + ε2                    # per-element residual "instability"

   if 2-D (factored):
       res_row = β3 · res_row + (1-β3) · mean(res, axis=cols)   # R^{d1×1}
       res_col = β3 · res_col + (1-β3) · mean(res, axis=rows)   # R^{1×d2}
       update = exp_avg · sqrt(res_row · res_col / mean(res_row))  # confidence scaling
   else (1-D, non-factored):
       update = exp_avg                              # residual is identity

5. p *= (1 − lr · wd)                                # decoupled weight decay (subtractive)
   p -= lr · update
```

### Default hyperparameters

```
lr = 2e-3, β = (0.9, 0.999, 0.9999), ε = (1e-30, 1e-16),
clip_threshold = 1.0, weight_decay = 0.0
```

(These match the official `came_pytorch` defaults. The paper's BERT-large experiments use lr=2e-4, but the canonical code defaults to 2e-3.)

---

## Phase 1: TDD Foundation

### Task 1: Header file skeleton (RED — no impl)

**Files:**
- Create: `include/nn/optimizers/came.h`

**Content:** Declare the class skeleton with public API mirroring the Adafactor pattern:

```cpp
class CAME : public Optimizer {
public:
    // Hyperparameters (public for inspection / test access).
    double lr;
    double beta1, beta2, beta3;
    double eps1, eps2;
    double clip_threshold;
    double weight_decay;
    int t;

    explicit CAME(double lr = 2e-3,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double beta3 = 0.9999,
                  double eps1 = 1e-30,
                  double eps2 = 1e-16,
                  double clip_threshold = 1.0,
                  double weight_decay = 0.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated setters.
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_beta3(double new_beta3);
    void set_eps1(double new_eps1);
    void set_eps2(double new_eps2);
    void set_clip_threshold(double new_clip);
    void set_weight_decay(double new_wd);

    // Accessors.
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_beta3() const { return beta3; }
    double get_eps1() const { return eps1; }
    double get_eps2() const { return eps2; }
    double get_clip_threshold() const { return clip_threshold; }
    double get_weight_decay() const { return weight_decay; }
    int get_t() const { return t; }

    // State introspection.
    bool has_state(void* layer_ptr) const;
    Tensor get_exp_avg(void* layer_ptr, size_t param_idx) const;
    bool get_exp_avg_sq_row(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_sq_col(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_res_row(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_res_col(void* layer_ptr, size_t param_idx, Tensor& out) const;
    double get_rms(void* layer_ptr, size_t param_idx) const;

private:
    struct ParameterState {
        Tensor exp_avg;             // (rows, cols)
        Tensor exp_avg_sq_row;      // (rows, 1) — 2-D only
        Tensor exp_avg_sq_col;      // (1, cols) — 2-D only
        Tensor exp_avg_res_row;     // (rows, 1) — 2-D only
        Tensor exp_avg_res_col;     // (1, cols) — 2-D only
        Tensor exp_avg_sq;          // (rows, cols) — 1-D only
        double rms = 0.0;
        bool is_1d = false;
    };

    std::map<void*, std::vector<ParameterState>> state_;

    static void validate(double lr,
                         double beta1, double beta2, double beta3,
                         double eps1, double eps2,
                         double clip_threshold, double weight_decay);

    static double rms(const Tensor& t);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    void update_param_2d(Tensor* param, Tensor* grad, ParameterState& state);
    void update_param_1d(Tensor* param, Tensor* grad, ParameterState& state);
};
```

**Step 1:** Write the header file.
**Step 2:** Verify it compiles by running `make -B build/test_came 2>&1 | head -20` — should fail with "undefined reference to CAME::..." or similar because there's no .cpp yet.

### Task 2: Test file skeleton — public behavior tests (RED)

**Files:**
- Create: `tests/test_came.cpp`

Write ~30 focused tests covering:
- Constructor defaults (T1)
- Constructor non-default args (T2)
- Validation throws (T3): lr ≤ 0, β1/β2/β3 ∉ [0,1], eps1/eps2 ≤ 0, clip_threshold ≤ 0, negative wd
- Mutator validation (T4)
- State shape correctness (T5): 2-D Dense(3,2) → exp_avg(3,2), row_ema(3,1), col_ema(1,2), res_row(3,1), res_col(1,2); 1-D bias (1,2) → exp_avg(1,2), exp_avg_sq(1,2)
- State has_state() before/after step (T6)
- t increments correctly (T7)
- First-step closed-form: with β1=0 and clip=∞ (very large), update matches `g / sqrt(g² + ε1)` exactly (T8)
- First-step closed-form with β1=0 and factored 2-D: update = `g / sqrt(row_EMA · col_EMA / mean(row_EMA))` (T9)
- Row EMA recurrence at step 2 (T10): `row_EMA_2 = β2·row_EMA_1 + (1-β2)·mean(g_2²+ε1, axis=cols)`
- Momentum recurrence (T11)
- Residual `res = (update - exp_avg)² + ε2` (T12)
- Res row/col EMA recurrence (T13)
- Confidence scaling: 2-D update = exp_avg · sqrt(res_row · res_col / mean(res_row)) (T14)
- 1-D identity path: `update = exp_avg` after residual step (T15)
- RMS-clip boundary: update with very large norm gets clipped to clip_threshold (T16)
- RMS-clip identity: update with small norm unchanged (T17)
- Weight decay shrinks params at zero gradient (T18): `param *= (1 − lr·wd)` exactly
- Two fresh CAME instances with identical grad sequences → bit-exact params (determinism) (T19)
- End-to-end loss reduction on linear regression (T20)
- Independent state across layers (T21)
- Independent state across parameters (T22)
- Gradient clearing (T23)
- Parameter/gradient count mismatch throws (T24)
- Parameter/gradient shape mismatch throws (T25)

**Step 1:** Write the test file with all the assertions.
**Step 2:** Compile test → should fail (header exists but no impl).

### Task 3: Minimal impl — validation + defaults + state shape

**Files:**
- Create: `include/nn/optimizers/came.cpp`

Implement:
- Constructor with `validate()`
- All accessors and setters (validated)
- `ensure_state()` for both 2-D and 1-D parameter shapes
- `rms()` static helper
- State accessors (has_state, get_exp_avg, get_exp_avg_sq_row, etc.)
- `step()` skeleton that iterates layers, ensures state, and calls `update_param_2d` or `update_param_1d`

**Step 1:** Implement.
**Step 2:** Build: `make tests` — verify compile success.
**Step 3:** Run T1-T7 from the test suite — should now pass.

### Task 4: Implement update_param_1d (GREEN for 1-D path)

Implement the 1-D update following the canonical reference:
```
1. raw = g² + ε1
2. exp_avg_sq = β2·exp_avg_sq + (1-β2)·raw
3. update = g / sqrt(exp_avg_sq)
4. update /= max(1, RMS(update) / clip_threshold)
5. exp_avg = β1·exp_avg + (1-β1)·update
6. (residual is identity for 1-D: update = exp_avg)
7. param *= (1 − lr·wd)
8. param −= lr·update
```

Verify T8 (β1=0 closed-form), T11 (momentum recurrence), T15 (1-D identity), T18 (weight decay), and T20 (end-to-end) pass.

### Task 5: Implement update_param_2d (GREEN for 2-D path)

Implement the 2-D factored update following the canonical reference:
```
1. raw = g² + ε1  (per-element)
2. row_EMA = β2·row_EMA + (1-β2)·mean(raw, axis=cols)  # R^{d1×1}
3. col_EMA = β2·col_EMA + (1-β2)·mean(raw, axis=rows)  # R^{1×d2}
4. update = g / sqrt(row_EMA · col_EMA / mean(row_EMA))  # reconstructed v_t
5. update /= max(1, RMS(update) / clip_threshold)
6. exp_avg = β1·exp_avg + (1-β1)·update
7. res = (update - exp_avg)² + ε2  (per-element)
8. res_row = β3·res_row + (1-β3)·mean(res, axis=cols)  # R^{d1×1}
9. res_col = β3·res_col + (1-β3)·mean(res, axis=rows)  # R^{1×d2}
10. update = exp_avg · sqrt(res_row · res_col / mean(res_row))  # confidence scaling
11. param *= (1 − lr·wd)
12. param −= lr·update
```

Verify T9 (β1=0 2-D closed-form), T10 (row EMA recurrence), T12 (residual), T13 (res row EMA), T14 (confidence scaling), and T20 pass.

### Task 6: Mutation testing

Run the mutation tests to confirm non-vacuousness:

1. **Drop residual correction** in update_param_2d (replace `update = exp_avg · sqrt(...)` with `update = exp_avg`): should cause T13 and T14 to fail.
2. **Drop RMS clip** (replace `update /= max(1, RMS(update)/clip)` with `update = update`): should cause T16 to fail.
3. **Drop weight decay** (remove the `param *= (1 − lr·wd)` line): should cause T18 to fail.
4. **Drop row/col factoring** in step 4 (compute `update = g / sqrt(exp_avg_sq_full)` with full-tensor v): should cause T9 to fail (closed-form mismatch).

Each mutation: revert after confirming failure → re-run full suite → confirm all green.

### Task 7: Wire up and final integration

- Add `#include "optimizers/came.h"` to `include/nn/nn.h`.
- Register `build/test_came` in `Makefile` as a build target.
- Add `$(BUILD_DIR)/test_came` to the `tests:` dependency list.
- Add `@echo "=== Running CAME Tests ===" && ./$(BUILD_DIR)/test_came` to `run_tests`.
- Commit with conventional commit message `feat(optimizers): add CAME confidence-guided adaptive memory-efficient optimization`.

---

## Verification checklist

- [ ] All ~30 focused tests pass at machine precision
- [ ] At least 3 mutation tests confirmed non-vacuous
- [ ] `make tests` compiles all targets cleanly
- [ ] `make run_tests` reaches the new "=== Running CAME Tests ===" line
- [ ] Build/test suite is green for the existing 80+ test targets (no regressions)
- [ ] Added to `include/nn/nn.h` umbrella
- [ ] Registered in Makefile (`build/test_came`, `make tests` / `make run_tests`)
- [ ] Plan file created at `docs/plans/2026-07-21-came-optimizer.md`
- [ ] Commit and push to `master`
