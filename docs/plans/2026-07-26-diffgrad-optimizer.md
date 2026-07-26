# DiffGrad Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Dubey et al. 2019's DiffGrad optimizer (https://arxiv.org/abs/1909.11015) as a fully-tested first-class member of the repo's optimizer collection.

**Architecture:** Stand-alone `DiffGrad` class (1 header + 1 cpp) following the exact proven pattern of the repo's existing optimizers (`AdaBelief`, `Lion`, etc.): per-parameter state map keyed by `Layer*`, an `ensure_state` constructor that lazily allocates state buffers sized to the parameter shape, and a `step(Model&)` loop that walks `model.layers` and pushes updates through a small helper `update_param` that takes `(param, grad, state)`. State per parameter: 3 Tensors (`m`, `v`, `g_prev`) plus a single timestep `t`. No external RNG — pure deterministic state machine.

**Tech Stack:** C++17. Reuses `Tensor` (only `Tensor::rows`, `Tensor::cols`, `Tensor::operator[][]`), `Model`, `Layer::parameters/gradients`. No new dependencies.

**Source-of-truth for the algorithm**: arxiv:1909.11015 §III (Eqs. 17-22) and the official shivram1987/diffGrad `diffGrad_v2.py` reference implementation (preferred over the buggy `diffGrad.py` v1). The math is:

```
g_t  ← grad   (already populated by the caller)
m_t  ← β1 · m_{t-1} + (1-β1) · g_t
v_t  ← β2 · v_{t-1} + (1-β2) · g_t²
dfc  ← 1 / (1 + exp( − |g_{t-1} − g_t| ))         ∈ (0, 1)
m_eff← dfc ⊙ m_t
denom ← √(v_t) + ε
update← (lr · √(1-β2^t) / (1-β1^t)) · m_eff / denom
param ← param − update
g_{t-1} ← g_t           (cache for the next step)

AdamW-style weight decay (decoupled): if wd > 0, param ← param · (1 − lr · wd)
```

When the optimizer is constructed with `wd == 0`, the optimisation reduces to plain diffGrad. When `β1=β2=0.5, ε=0.5` and the gradient is identically zero, no state changes and `param` is untouched.

---

### Task 1: Write the failing test file

**Files:**
- Create: `tests/test_diffgrad.cpp`
- Create: `tests/test_diffgrad_make.log` (expected: compile-fails first)

**Step 1:** Write `tests/test_diffgrad.cpp` with 30+ checks across the following invariant buckets — *no production code yet*:

- (a) **Defaults round-trip**: `DiffGrad()` (or `DiffGrad(0.001)`) exposes `lr=0.001, β1=0.9, β2=0.999, ε=1e-8, wd=0, t=1`, `handles_weight_decay()=true`.
- (b) **Constructor validation throws** (each): `lr<0`, `β1∉[0,1)`, `β2∉[0,1)`, `ε≤0`, `wd<0`.
- (c) **Setters throw**: negative lr, β1≥1, β2≥1, ε≤0, wd<0.
- (d) **Zero gradient does not change params**: dense(2,2) zero-grad step → weights & bias unchanged bit-exact, but `t` advances 1→2.
- (e) **Single-step closed-form analytic check** (hand-derived):
  - Dense(2,2), all-zero init, all-one gradient, lr=1, β1=β2=0.5, ε=0.5, wd=0.
  - Predicted values per (i,j):
    `m_1 = 0.5 · 1 = 0.5`
    `v_1 = 0.5 · 1 = 0.5`, `denom = √0.5 + 0.5 ≈ 1.2071`
    `g_prev = 0` initial → `dfc = 1/(1+exp(-1)) = sigmoid(1) ≈ 0.7311`
    `m_eff = dfc · m_1 ≈ 0.3655`
    `step_size = lr · √(1−0.5) / (1−0.5) = 1 · 0.7071/0.5 = 1.4142`
    `update = 1.4142 · 0.3655 / 1.2071 ≈ 0.4280`
    `param_1 = 0 − 0.4280 = −0.4280`.
  - Assert `|param_1 − (−0.4280)| < 1e−6` bit-exact vs the same closed-form expression.
- (f) **DFC varies with gradient change**:
  - On step 1, `g_prev = 0`, gradient `[1, 1]` → `dfc = sigmoid(1) ≈ 0.7311` everywhere.
  - On step 2, same gradient `[1, 1]` → `g_prev = [1,1]`, so `dfc = sigmoid(0) = 0.5` (smaller step).
  - On step 2, switch gradient to `[2, 2]` → `dfc = sigmoid(1) ≈ 0.7311`, a larger step than with constant `[1, 1]`.
- (g) **Differentiates from Adam**: when gradient oscillates `[1, -1, 1, -1, 1, -1]`, diffGrad's DFC reduction around sign-flip yields a measurably different trajectory than Adam from `tests/test_lookahead` style fixture.
- (h) **State shape correctness** on `Dense(2,3)`: `m` is (2,3), `v` is (2,3), `g_prev` is (2,3), all zero-initialised.
- (i) **State lazy-init**: `has_state(layer, idx)` returns false before any step, true after; different layer → its own state.
- (j) **Decoupled weight decay** (wd > 0): with `[1.0]` param, no gradient (synthetic fake-positive zero-grad), `param *= (1 − lr·wd)` at every step → after 5 steps at lr=0.1, wd=0.1: `1.0 · (1−0.01)^5 = 0.9509`.
- (k) **Multi-layer independence**: two `Dense` layers in the same model get separate state, separate `t` advancement, step on each updates only that layer.
- (l) **Determinism** (signature): two freshly-constructed `DiffGrad` instances with identical hyper-parameters produce bit-exact trajectories over 10 random-grad steps; same with `set_step` reset.
- (m) **End-to-end training reduction**: y=2x linear regression (Dense(1,1) over `x ∈ [-1, 1]`, `y = 2x`), 60 steps with `lr=0.1`, β=defaults, ε=1e-8 → MSE drops >99% from initial value.
- (n) **`handles_weight_decay()` is true** (decoupled form returns true so the `WeightDecay` wrapper correctly skips re-applying decay).
- (o) **Mutation-tested coverage** (4 mutations for non-vacuousness):
  1. *remove DFC*: set `dfc := 1` → diffGrad reduces to Adam; closed-form and DFC-variation tests in (e) and (f) both fail. Expect ≥3 assertion failures.
  2. *swap `m_eff` for raw `m_t`*: closed-form changes → expect test (e) fails.
  3. *forget g_prev update*: leave g_prev = 0 forever → DFC always sigmoid(1) ≈ 0.7311 → sign-flip test (g) fails.
  4. *use bias-corrected `b1_c` in m_eff* (incorrect scaling) → closed-form fails.

**Step 2: Run the test to verify it fails**

Run: `make build/test_diffgrad`
Expected: build fails because `nn/optimizers/diffgrad.h` does not exist yet (linker error or "no such file").

### Task 2: Write the production header `diffgrad.h`

**Files:**
- Create: `include/nn/optimizers/diffgrad.h`

Header contains: a `DiffGrad : public Optimizer` class with public members `lr, beta1, beta2, epsilon, weight_decay, t`, the public constructor with defaults matching the paper, `step(Model&)`, `handles_weight_decay()` returning true (decoupled-WD form), and a `DiffGradState` struct holding `Tensor m, v, g_prev` (private). The header is 70-90 lines, mirrors `adabelief.h` style.

### Task 3: Write the production cpp `diffgrad.cpp`

**Files:**
- Create: `include/nn/optimizers/diffgrad.cpp`

Contains:
- The constructor (no validation in body — validation is in setters).
- `ensure_state(layer_ptr, params)`: lazy-init state for each parameter to its shape with `fill(0.0)`.
- `update_param(param, grad, st, lr, b1, b2, eps, wd, b1_c, b2_c)`: per-element loop matching `adabelief.cpp` style but with the DFC switch (`dfc = 1.0/(1+exp(-|g_prev−grad|))`, `m_eff = dfc*m`, `denom = sqrt(v)+eps`, `step_size = lr*sqrt(b2_c)/b1_c`, `param -= step_size*m_eff/denom`, weight-decay path `param *= (1 - lr*wd)` if wd>0; g_prev←grad at end of inner loop).
- `step(model)`: compute `b1_c = 1 - β1^t, b2_c = 1 - β2^t`, walk `model.layers`, skip params.empty(), dispatch.

### Task 4: Compile and run the test

**Files:**
- Add `$(BUILD_DIR)/test_diffgrad` rule to the `tests:` block in `Makefile` (alongside `test_adabelief`).
- Add `#include "nn/optimizers/diffgrad.h"` to `include/nn/nn.h` umbrella.
- Add `$(BUILD_DIR)/test_diffgrad: $(LIB_OBJS) $(BUILD_DIR)/test_diffgrad.o` build rule in `Makefile`.
- Add `=== Running DiffGrad Tests ===` line in `run_tests` after `=== Running AdaBelief Tests ===`.

**Step 1:** Makefile changes (3 patches), then `make tests` should compile the new binary.

**Step 2:** `./build/test_diffgrad` — expect all focused checks to pass, ideally 30-40 of them at machine precision.

**Step 3:** Run the full suite: `make run_tests` — make sure the rest of the 95+ stable suites still pass and the 4 deferred suites still defer.

### Task 5: Mutation-testing the production code

Apply each of the 4 mutations listed above to a temporary copy, rebuild, and confirm the focused test suite reports the expected failures. Revert.

### Task 6: Commit & push

- `git add` SPECIFIC files: `tests/test_diffgrad.cpp`, `include/nn/optimizers/diffgrad.{h,cpp}`, `Makefile`, `include/nn/nn.h`, `EXPANSION_QUEUE.md`, `docs/plans/2026-07-26-diffgrad-optimizer.md`.
- Commit: `feat(optimizers): add DiffGrad (Dubey 2019) gradient-friction coefficient`.
- Push: `git push origin master`.

### Task 7: Move queue entry to Done

Append a Done entry to `EXPANSION_QUEUE.md` summarising: file path, paper reference, math formula, test count, mutation count, and the `make tests`/`make run_tests` final status. The Idea entry is removed from `## Ideas`.

---

## Verification

- [x] `make tests` builds the new binary without warnings
- [x] `./build/test_diffgrad` reports ≥30 focused checks, all PASS, ideally at machine precision (rel_err < 1e-10)
- [x] `make run_tests` still executes the 95+ stable suites cleanly (4 deferred suites still defer per `NOT_FIXED.md`)
- [x] All 4 mutation tests yield ≥1 focused failure each
- [x] DECL-form `lr < 0`, `β1 ≥ 1`, `β2 ≥ 1`, `ε ≤ 0`, `wd < 0` all throw `std::invalid_argument`
- [x] `git log --oneline -n 1` shows the new commit, `git push` returns clean
