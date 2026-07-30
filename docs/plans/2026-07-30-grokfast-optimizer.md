# GrokFast Optimizer — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a `GrokFast` optimizer wrapper that amplifies slow gradients via an EMA filter, accelerating grokking on tasks where the model needs to find rare generalization signal. Reference: Lee et al. 2024, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" (https://arxiv.org/abs/2405.20233, NeurIPS 2024 Spotlight), official code: https://github.com/ironjr/grokfast.

**Architecture:** A standalone `GrokFast` class that wraps any existing optimizer (default: AdamW). The wrapper sits BETWEEN the backward pass (which populates gradients) and the inner optimizer's `step()` call. Per parameter, per step:
1. Maintain an EMA filter `buf` of the gradient: `buf = alpha * buf + (1 - alpha) * grad`
2. Compute the amplified "slow" gradient: `grad_filtered = grad + lambda * buf`
3. OVERWRITE the parameter's stored gradient with `grad_filtered`
4. Call `inner.step(model)` (the inner optimizer sees the filtered gradient)
5. `handles_weight_decay()` delegates to the inner optimizer (no double-application)

The wrapper owns the EMA filter state per parameter (one Tensor per parameter, same shape as the parameter). The inner optimizer is owned via `unique_ptr`.

**Tech Stack:** C++17, the existing `Optimizer` / `Model` / `Tensor` / `Layer` core. No new dependencies.

---

## Reference math (verified against Lee et al. 2024 §3.2 and ironjr/grokfast `grokfast.py` reference)

```
Per parameter θ, per step t:
    buf_t     = α · buf_{t-1} + (1 - α) · grad_t           # EMA filter
    grad_filtered_t = grad_t + λ · buf_t                    # amplify slow component
    θ ← inner_optimizer.step(grad_filtered_t)
```

Defaults (paper §3.2 / Table 1):
- `lambda = 2.0`  (amplification factor — paper uses 2.0 in main experiments)
- `alpha  = 0.98` (EMA momentum — paper Table 3)
- inner = Adam (lr=1e-3, β1=0.9, β2=0.999, ε=1e-8)

When `lambda = 0`, GrokFast is a no-op (the filtered gradient equals the raw gradient → EMA filter still updates but its contribution is multiplied by 0).

State:
- One `Tensor buf` per parameter (same shape as the parameter, initialized lazily on first step)
- No second-moment statistics (unlike Adam-style optimizers) — only the slow-gradient EMA

Edge cases:
- `lambda = 0`: filtered gradient = raw gradient → step is bit-identical to inner.step (after one extra multiplication by 0). The `buf` is still maintained but doesn't influence the update.
- `alpha = 0`: `buf_t = grad_t` exactly → filtered = `(1 + λ) · grad` (a global scale, then inner.step). This is the "no-filtering" limit.
- `alpha = 1`: `buf_t = buf_{t-1}` → gradient-free filter, updates stall (the buf is frozen at zero → no slow-grad contribution).
- Parameter shape change between steps (unusual): the lazy-init guards by comparing `rows,cols` and re-initializing `buf` if shape changed.

---

## Files to touch

| Operation | Path |
|-----------|------|
| Create | `include/nn/optimizers/grokfast.h` |
| Create | `include/nn/optimizers/grokfast.cpp` |
| Modify | `include/nn/nn.h` (add umbrella include) |
| Modify | `Makefile` (build rule, tests dep, run_tests echo) |
| Create | `tests/test_grokfast.cpp` |
| Create | `docs/plans/2026-07-30-grokfast-optimizer.md` (this file) |

---

## Task 1: Write the failing test scaffold

**Objective:** Compile a test file that exercises the GrokFast optimizer wrapper and proves the implementation is missing.

**Files:**
- Create: `tests/test_grokfast.cpp`

**Step 1:** Write the test file outline that:
- Includes `nn/optimizers/grokfast.h`
- Constructs a `GrokFast` wrapping a real `Adam` inner
- Asserts defaults (`lambda=2.0`, `alpha=0.98`)
- Asserts validation throws on bad `lambda` (negative) or bad `alpha` (not in [0, 1))
- Asserts closed-form first-step math on Dense(2,2) with all-ones grad:
  - `buf_1 = (1 - 0.98) · 1 = 0.02`
  - `grad_filtered_1 = 1 + 2.0 · 0.02 = 1.04`
  - `m_1 = (1 - 0.9) · 1.04 = 0.104`
  - For ε=1e-8, `v_1 = (1 - 0.999) · 1.04² = 0.00108032`
  - The update should match this formula to ~1e-6
- Asserts `lambda=0` reduces to inner Adam bit-exactly (after accounting for inner's step counter)
- Asserts `alpha=0` (no-filtering limit) → `buf_t = grad_t` exactly
- Asserts `handles_weight_decay()` delegates to inner (true for AdamW, false for Adam)
- Asserts state-init: `has_state` returns false before step, true after step
- Asserts end-to-end linear regression loss decreases over 50 steps
- Asserts signature: GrokFast(Adam) trajectory differs from plain Adam over 3+ steps with constant grad

**Step 2:** Run `make build/test_grokfast` — expected to fail (header missing).

---

## Task 2: GrokFast header

**Objective:** Public API for the wrapper.

**Files:**
- Create: `include/nn/optimizers/grokfast.h`

**Step 1:** Define `class GrokFast : public Optimizer` with:
- ctor `GrokFast(std::unique_ptr<Optimizer> inner, double lambda = 2.0, double alpha = 0.98)`
- `void step(Model& model) override` — main entry point
- `bool handles_weight_decay() const override` — delegates to inner
- Setters: `set_lambda(double)`, `set_alpha(double)`, `set_lr(double)` (forwards to inner)
- Test diagnostics: `bool has_state(void* layer_ptr, size_t param_idx) const`, `size_t last_num_params_filtered() const`, `Optimizer* inner() const`, `double get_lambda() const`, `double get_alpha() const`, `double get_lr() const` (forwards to inner)
- Private: `std::unique_ptr<Optimizer> inner_`, `double lambda_`, `double alpha_`, `size_t last_num_params_filtered_`, `std::map<void*, std::vector<Tensor>> buf_state_` (one `buf` per parameter)

**Step 2:** Compile header-only by including it (header guard).

---

## Task 3: GrokFast implementation

**Objective:** Minimum correct implementation.

**Files:**
- Create: `include/nn/optimizers/grokfast.cpp`

**Algorithm (per step):**
1. Walk every layer in `model.layers`.
2. For each parameter `p` of the layer:
   a. Get the parameter's stored gradient `g`.
   b. If `buf_state[layer_ptr][i]` is uninitialized OR its shape doesn't match `p`: create `buf = Tensor(p->rows, p->cols)` and `fill(0.0)`.
   c. For each element `(r, c)`:
      - `buf[r][c] = alpha * buf[r][c] + (1 - alpha) * g[r][c]`
      - `g_filtered[r][c] = g[r][c] + lambda * buf[r][c]`
      - WRITE BACK `g[r][c] = g_filtered[r][c]` (so the inner optimizer sees the filtered gradient).
3. Record `last_num_params_filtered_`.
4. Call `inner_->step(model)`.

Constructor validation:
- `inner` must not be null → throw `std::invalid_argument`
- `lambda` must be >= 0 → throw `std::invalid_argument`
- `alpha` must be in `[0, 1)` → throw `std::invalid_argument`

Setter validation (same as constructor).

---

## Task 4: Verify GREEN

**Step 1:** Run `make build/test_grokfast && ./build/test_grokfast` — expected to PASS at machine precision.

**Step 2:** Run `make run_tests` to verify no regressions.

**Step 3:** Mutation test: zero out the `buf[r][c]` write (i.e., buf never gets updated) → tests should fail. Verify the test catches it. Restore the line.

---

## Task 5: Register and document

**Files:**
- Modify: `include/nn/nn.h` — add `#include "optimizers/grokfast.h"` in alphabetical order near `gradient_centralization.h`.
- Modify: `Makefile` — add `build/test_grokfast` rule, dep in `tests:`, and `=== Running GrokFast Tests ===` echo in `run_tests:`.

---

## Verification

`make tests && make run_tests` — all 118+1=119 stable suites pass; GrokFast's 40+ focused checks pass at machine precision.