# MARS Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the MARS optimizer (Yuan et al., 2024) — a variance-reduced Adam variant that adds a gradient-shift correction `g_t − g_{t-1}` to reduce gradient variance and improve convergence. Targets the gap left by recent adaptive methods (Adan/RAdam/Yogi) which the repo already covers.

**Architecture:** MARS wraps a base Adam update with a MARSM (Multi-Adam Reweighted Smoothing, "shifted momentum") step that injects the previous gradient into the update direction. Lives as `include/nn/optimizers/mars.{h,cpp}`. Public API matches Adam / Lion / RAdam pattern (accessors, validating setters, lazy-init state per layer). State per parameter: `(m, v)` — same memory footprint as AdamW, plus one extra scalar `grad_prev_norm_sq` per parameter OR stored `g_prev` tensor. We use the latter (`g_prev` tensor) for full reproducibility.

**Tech Stack:** C++17, single-header Tensor ops already in repo, follows `Lion`/`Adan` template style.

---

## Background — MARS algorithm

Paper: "MARS: Unleashing the Power of Variance Reduction for Training Large Language Models", Yuan, Zhang, Chen, Yu, Liu, Sun, Wang (2024) — https://arxiv.org/abs/2401.11615.

**Full MARS (MARSM variant) update, per parameter `θ`:**
1. `g̃_t = g_t + γ · (g_t − g_{t-1})`     (MARSM shift correction; γ ∈ [0,1])
2. `m_t = β1 · m_{t-1} + (1 − β1) · g̃_t`  (first moment on corrected gradient)
3. `v_t = β2 · v_{t-1} + (1 − β2) · g̃_t²`
4. `m̂_t = m_t / (1 − β1^t)`,  `v̂_t = v_t / (1 − β2^t)`
5. `θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε) + lr · wd · θ_t`   (AdamW decoupled wd)

γ is the only MARS-specific hyperparameter. Defaults from paper: γ = 0.025 (small — the role of (g − g_prev) is a small correction, not a giant shift). β1 = 0.9, β2 = 0.999, ε = 1e-8, wd = 0, lr = 3e-4 (but expository; we default to 1e-3).

**MARSE (epsilon clip) variant** clips `g̃` to `sign(g̃) · clip(|g̃| / |g_{t-1}|, ε_max, 1)` after the shift — keeps the shift ratio bounded. We expose this as `clip=true` constructor flag with `eps_max=1.0` default.

**Properties worth testing:**
- At γ=0: MARS reduces exactly to Adam (soul test).
- The shift direction: under constant gradient `g`, `g̃ = g`, and MARS = Adam again (correct: no info in `g − g_prev = 0`).
- State size: (m, v, g_prev) — three tensors per parameter (Adam is two).
- First step: `g_prev = 0`, so `g̃ = g`, `m_1 = (1-β1)·g`, `v_1 = (1-β2)·g²` — same as Adam step 1.

---

## Plan Structure (Template A)

The plan is broken into numbered TDD tasks. Each task adds one piece of behavior with a test that fails before the implementation and passes after. Commit at the end of each task.

### Task 1: Skeleton — header, defaults, validation, no-op `step()`
- Create `include/nn/optimizers/mars.h` with the class definition following the `Lion`/`RAdam` style. Public members: `lr, beta1, beta2, gamma, epsilon, weight_decay, clip, eps_max, t`. Constructor with defaults. Validation throws on invalid hyperparameters.
- Create `include/nn/optimizers/mars.cpp` with empty `step()` that iterates layers and doesn't yet touch state.
- Add `#include "optimizers/mars.h"` to `include/nn/nn.h` umbrella.
- Wire up build target `build/test_mars` in Makefile.
- Tests that should pass after this task: constructors (defaults + non-defaults), validation throws (β1 ∉ [0,1), β2 ∉ [0,1), γ ∉ [0,1], ε ≤ 0, negative wd, eps_max ≤ 0, invalid clip flag), `t == 1` after construction, `step()` doesn't crash on empty model and on a model with a single Dense layer (no gradients yet → no-op updates).

### Task 2: State shape + lazy-init
- Implement `ensure_state(layer_ptr, params)` mirroring the Lion pattern. Per-parameter state is a `(m, v, g_prev)` triple, all same shape as the parameter.
- Tests: `has_state()` returns false before step, true after; R, C (or m, v, g_prev) shapes match the parameter; bias has (1, out) state.

### Task 3: First-step closed form (Adam-equivalent at γ=0 OR first step)
- Implement `update_param()` following MARSM update.
- First step: g_prev = 0 by lazy-init. So `g̃ = g + γ·(g − 0) = (1+γ)·g`. Then m = (1-β1)·(1+γ)·g, v = (1-β2)·(1+γ)²·g². Final: `θ_1 − lr · m / (√v + ε) + lr·wd·θ_1`.
- At γ=0: that gives standard Adam step-1 closed form. **This is the soul test** — MARS at γ=0 equals Adam.
- Tests: (a) at γ=0 and zero second moment prior, the first-step param update matches hand-derived Adam step-1 formula; (b) first step works with non-zero γ.

### Task 4: γ-shift closed form (MARS signature)
- Two successive steps with gradient g then h; g_prev = g after step 1. Step 2: g̃ = h + γ·(h − g); m, v update on g̃. Verify against hand-derived closed form.
- Test: shift direction is correct for both positive and negative corrections.

### Task 5: Adam equivalence test
- Run a Dense(2,1) model with a few gradient steps under both MARS(γ=0) and Adam. The final parameters should match bit-for-bit (or to rel_err ~1e-12 due to float summation).
- This is the strongest non-vacuous signature — distinguishes MARS from a generic Adam wrapper.

### Task 6: Decoupled weight decay
- At zero gradient with wd>0, param shrinks by `lr · wd · param` per step (AdamW-style).
- Test: weight decay at zero gradient shrinks params exactly; non-zero gradient also includes wd term.

### Task 7: CLIP variant (MARSE)
- When `clip=true`, `g̃` is clipped per element: `g̃_clip = sign(g̃) · min(max(|g̃| / max(|g_prev|, 1e-8), eps_max), 1.0)`.
- Test: when |g̃| / |g_prev| > 1, g̃ is clipped to ±g_prev (or +eps_max multiplier); when ratio < eps_max, g̃ is scaled up to eps_max.

### Task 8: Determinism
- Two fresh MARS instances, identical gradient sequences → bit-identical params. (rel_err < 1e-12)

### Task 9: End-to-end training
- Train a 2-input, 1-output Dense model on `y = 2·x1 + 3·x2 − 1`. 100 steps with MARS(γ=0.025) reduces loss to < 50% of initial. Bigger loss reduction is even better.

### Task 10: Sign vs Adam signature test
- Under oscillating gradient: MARS handles g_prev shift so the *direction* of motion differs from Adam. Specifically, at end of 5 steps with gradient sequence +1, −1, +1, −1, +1 (β1=0, no EMA from m), Adam's effective gradient is average of g (≈ +1) so weights decrease; MARS sees g_prev (e.g. +1 last) → shift on next g affects update direction. Test that final weights after a controlled gradient sequence differ from Adam.

### Task 11: Mutation tests
- (a) Drop the γ·(g − g_prev) shift (γ=0 effectively always) → test 5 fails (Adam-equivalence), test 4 fails (shift closed form).
- (b) Drop g_prev lazy init (use 0 vector every step) → test 4 fails (computed shift uses 0 not first g), test 10 signature differs.
- Both mutations should produce multiple test failures.

### Task 12: Wire into `tests:` and `run_tests:` in Makefile
- Add `$(BUILD_DIR)/test_mars` to the `tests:` deps and `make run_tests` invocation.

### Task 13: Cleanup
- Remove any debug `std::cerr` left from instrumentation.
- Verify there are no stray .bak, ref_*, or debug_* files.

---

## File paths (exact)

- `include/nn/optimizers/mars.h`
- `include/nn/optimizers/mars.cpp`
- `tests/test_mars.cpp`
- `include/nn/nn.h` — add `#include "optimizers/mars.h"` near the other optimizer includes
- `Makefile` — add `$(BUILD_DIR)/test_mars: $(LIB_OBJS) $(BUILD_DIR)/test_mars.o` rule, add to `tests:` deps, add to `run_tests` invocation

## Verification

After all tasks:
- Run `make tests` and `make run_tests`. New tests should report something like `=== Summary: NN/NN checks passed ===`.
- All old tests continue to pass.
- Run mutation tests to verify the new tests are non-vacuous.

## Constraints

- TDD: every behavior is added via a failing test first.
- Mutate the implementation in 2+ representative ways and ensure the new tests catch at least one of them.
- Conventional commit messages: e.g. `feat(optimizers): MARS — variance-reduced Adam, N/M tests pass`.
- One atomic commit per task.
