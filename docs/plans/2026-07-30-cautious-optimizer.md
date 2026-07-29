# Cautious Optimizer — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a Cautious wrapper optimizer that masks momentum updates to only allow sign-aligned steps, with density compensation to preserve the average step magnitude. Reference: Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code" (https://arxiv.org/abs/2411.16085), official code: https://github.com/kyleliang919/C-Optim.

**Architecture:** A standalone `Cautious` class that wraps any existing momentum-based optimizer (default: AdamW). The wrapper intercepts the parameter update direction by masking entries where the inner optimizer's first-moment direction disagrees with the raw gradient, then re-normalizes the masked update by the mask's mean to compensate for the lost step mass. State is keyed by (layer_ptr, param_idx) and stores the masked-update normalization factor for transparency. Pre/post hooks via intercepting the parameter walk happen inside `step()`: after the inner optimizer has computed its `m_t` and is about to apply `param -= lr * m_hat / (sqrt(v) + eps)`, we apply the mask first.

**Tech Stack:** C++17, the existing `Optimizer` / `Model` / `Tensor` / `Layer` core. No new dependencies.

---

## Reference math (verified against paper §2 and `c_adamw.py` reference)

Per parameter, per step:
```
# inner optimizer computes its standard update direction u_t:
#   AdamW: u_t = m_hat_t / (sqrt(v_hat_t) + eps)
#   Lion: u_t = sign(beta1 * m_t + (1 - beta1) * g_t)
#   SGD+M: u_t = m_t
#   Muon: u_t = ...
# Cautious only modifies HOW that direction is applied:
mask_t = (u_t * grad > 0).to(u_t.dtype)         # 1 where sign agrees, 0 otherwise
mask_mean_t = mask_t.mean().clamp(min=eps_mask)  # density compensation, floor 1e-3
masked_u_t = u_t * mask_t / mask_mean_t
param -= lr * masked_u_t
```

The mask is APPLIED to the inner optimizer's update direction (NOT just `m_in ugly-aware ways). The masking preserves the AVERAGE step magnitude when signs mostly agree (mean ≈ 1) and reduces the effective LR when more than half the entries disagree (mean < 1 → compensation factor > 1 → per-element step grows). This is the whole "one line of code" trick.

Edge cases:
- All mask entries zero (mean → 0): clamp to `eps_mask` (default 1e-3) → mask becomes zero → zero step. This is the "do nothing" fallback.
- Tensor of size 1: mean is the single mask entry → compensation collapses to identity if mask=1, zero step if mask=0.

---

## Files to touch

| Operation | Path |
|-----------|------|
| Create | `include/nn/optimizers/cautious.h` |
| Create | `include/nn/optimizers/cautious.cpp` |
| Modify | `include/nn/nn.h` (add umbrella include) |
| Modify | `Makefile` (build rule, tests dep, run_tests echo) |
| Create | `tests/test_cautious.cpp` |
| Create | `docs/plans/2026-07-30-cautious-optimizer.md` (this file) |

---

## Task 1: Write the failing test scaffold

**Objective:** Compile a test file that exercises the Cautious optimizer wrapper stub and proves the implementation is missing.

**Files:**
- Create: `tests/test_cautious.cpp`

**Step 1:** Write the test file outline that:
- Includes `nn/optimizers/cautious.h`
- Constructs a `Cautious` wrapping a real `Adam` inner
- Asserts defaults (lr=1e-3, eps_mask=1e-3, mask_mode=UPDATE)
- Asserts validation throws on bad eps_mask
- Asserts closed-form first-step math on Dense(2,2) with all-positive grad (mask=1, mean=1, no change vs AdamW)
- Asserts closed-form first-step with mixed-sign gradient (mask=0 for disagreeing entries, half-entries get 2× compensation)
- Asserts `handles_weight_decay()` delegates to inner
- Asserts state-init diagnostics
- Asserts end-to-end linear regression loss decreases over 80 steps
- Asserts signature vs Adam: trajectory differs when gradients are mixed-sign

**Step 2:** Run `make build/test_cautious` — expected to fail (header missing).

---

## Task 2: Cautious header

**Objective:** Public API for the wrapper.

**Files:**
- Create: `include/nn/optimizers/cautious.h`

**Step 1:** Define `class Cautious : public Optimizer` with:
- ctor `Cautious(std::unique_ptr<Optimizer> inner, double eps_mask = 1e-3)`
- `void step(Model& model) override` — main entry point
- `bool handles_weight_decay() const override` — delegates to inner
- Setters: `set_eps_mask(double)`, `set_lr(double)` (forwards to inner)
- Test diagnostics: `MaskStats last_stats() const`, `size_t num_params_updated() const`, `bool has_state(void* layer_ptr, size_t param_idx) const`
- Private struct `Entry { double mask_sum; double n_entries; }` and `std::map<void*, std::vector<Entry>> stats_`
- Private `void mask_and_scale(Tensor& update, const Tensor& grad, Entry& entry)` helper

**Step 2:** Compile header-only by including it (header guard).

---

## Task 3: Cautious implementation

**Objective:** Minimum correct implementation.

**Files:**
- Create: `include/nn/optimizers/cautious.cpp`

**Algorithm (per step, per parameter):**
1. Iterate layers, then params, like every other optimizer.
2. For each parameter, snapshot the parameter and its gradient.
3. Apply the inner optimizer's `step` to capture the post-update difference (param_after - param_before) → that's `lr * u_t` semantically.
   - **Problem:** this is hard to extract cleanly because existing optimizers mutate `param` in place.
   - **Solution:** snapshot param BEFORE inner.step, then for each param apply the cautious mask by reconstructing the update direction from the difference.
4. **Reversible approach:** actually use a per-parameter apply mask. Snapshot param BEFORE inner.step. Snapshot param AFTER inner.step. Compute `u_t = (param_after - param_before) / lr`. Then set `param = param_before`, then `param -= lr * (u_t * mask_t / mask_mean_t)`. This produces the same end state as AdamW when mask=1, and the cautious state when mask ≠ 1.
5. Aggregate mask stats and store.

**Step 1:** Write the impl. Edge cases:
- Null `inner_` throws `std::invalid_argument` at construction.
- `eps_mask <= 0` throws.
- `lr < 0` throws.
- All-zero mask → use unclamped mask (0/0 → 0, no step).
- Param has no gradient → skip.
- `handles_weight_decay()` → `inner_->handles_weight_decay()`.

**Step 2:** Run the test from Task 1 — expected: all matched signatures pass.

---

## Task 4: Mutation tests

**Objective:** Verify the tests catch missing mask, missing density compensation, and disabled cautious mode.

**Tests:**
1. Stub out the mask application line (comment `param -= lr * u_t * mask / mask_mean`) → expect tests fail.
2. Skip the `mask_mean` normalization (use `mask / 1.0`) → expect closed-form tests fail.
3. Replace mask with all-ones (no-op) → expect signature-vs-Adam test passes vacuously → that's expected behavior; document it.

---

## Task 5: Wire into umbrella + Makefile

**Files:**
- Modify: `include/nn/nn.h` — add `#include "optimizers/cautious.h"` after `gradient_centralization.h`.
- Modify: `Makefile` — add compile rule, tests dep, run_tests echo.

**Step 1:** Add the entries.
**Step 2:** Run `make tests` and confirm full suite builds.

---

## Task 6: Commit

Atomic commits:
1. `feat(optimizers): add Cautious wrapper (Liang 2024)`
2. `test(optimizers): add Cautious test coverage`
3. `chore: register Cautious in umbrella + Makefile`

---

## Risks & gotchas

- **Mutation in place:** Inner optimizers mutate `param` directly. We need to snapshot BEFORE/AFTER to reconstruct `u_t`. Order-of-operations matters: inner.step() must NOT zero gradients before we extract the gradient for masking. The existing optimizers in this repo zero_grad at the END of step(), so snapshot before step is safe.
- **Density compensation floor:** `eps_mask = 1e-3` matches the reference impl. Smaller values risk mask amplification when most entries disagree. Larger values weaken the compensation.
- **Re-wrap safety:** Storing inner via `unique_ptr` follows the SAM/WeightDecay/Lookahead pattern. Avoid double ownership.
- **Backward compat:** Don't break existing optimizer contracts. `Cautious` only ADDS a wrapper; no existing optimizer is modified.
- **Closed-form test:** For mixed-sign grad, choose `[+1, -1, +1, -1]` so mask = `[1, 0, 1, 0]`, mean = 0.5, compensation = 2.0. Each agreeing entry gets `2 × AdamW`'s step.

---

## Verification checklist

- [ ] `make tests` builds all 110+ test binaries
- [ ] `./build/test_cautious` reports `=== Summary: N passed, 0 failed ===` with N ~ 40-60
- [ ] Mutation tests (mask removed, mean removed) cause the suite to fail
- [ ] No new compiler warnings
- [ ] `git push origin master` succeeds
