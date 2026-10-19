# Per-tensor / per-layer Learning-Rate Multipliers — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Give every optimizer in the repo a per-tensor learning-rate multiplier — without touching any of the 39 optimizers.

**Architecture:** A new `LrMultiplier` value type holds a `(name → scale)` map plus an optional default, resolves a scale for a named parameter, and applies it by **post-hoc rescaling of the realized update delta** (`w ← w + s·(w_after − w_before)`). The mechanism was chosen by measurement, not intuition — see "Why not gradient pre-scaling" below.

**Tech Stack:** C++17, existing `Tensor`/`Model`/`Optimizer` core, hand-rolled test harness matching `tests/test_curriculum.cpp` shape.

---

## Why not gradient pre-scaling

The obvious mechanism — scale the gradient `g ← s·g` before `optimizer.step()` — was the first
draft's design and is **wrong for this repo**. Measured (`tests/probe_lrscaling.cpp`, since deleted):

| optimizer | `\|step(3g) − step(0) − 3·(step(g) − step(0))\| / max\|3·Δ\|` |
|---|---|
| SGD  | `2.64e-16` (exact) |
| Adam | `6.67e-01` (**annihilated**) |

Adam is *scale-invariant* in `g`: the update is `lr·m̂/(√v̂+ε)`, and with one step `m̂ ∝ g`, `v̂ ∝ g²`,
so `g/√(g²) = sign(g)` — multiplying `g` by 3 changes the update by nothing at all
(`step(1g) == step(3g) == 0.4`). Any feature that scaled gradients would be a **silent no-op on the
repo's most-used optimizer family**. This is systematic-debugging §5d (the function under test was
reached but had no effect) arriving at design time.

Post-hoc delta rescaling is instead **optimizer-agnostic**: whatever `step()` does, the wrapper
observes the realized delta and rescales it, so it works for all 39 optimizers including the
scale-invariant ones.

---

## Semantics to pin (each gets a named test)

- **D1** A scale of `1.0` is an **exact** no-op (bit-identical weights), not "close to".
- **D2** Unnamed parameters take the **default** scale.
- **D3** A named parameter takes its own scale, overriding the default.
- **D4** Unknown layer names are **not** an error — they fall through to the default. (A wrapper
  that throws on an unmatched name is unusable across a 250-layer repo.)
- **D5** Scale `0` freezes the parameter exactly; negative scales are legal (gradient reversal).
- **D6** The multiplier applies to the **delta only** — `w_before` is never mutated, so the
  wrapper is idempotent under repeated application and does not double-count.
- **D7** An optimizer that changes *nothing* for a layer produces a zero delta, so any scale leaves
  it unchanged (no NaN, no drift).

---

## Tasks

### Task 1: `LrMultiplier` core (TDD)

**Files:**
- Create: `include/nn/utils/lr_multiplier.h`, `.cpp`
- Test: `tests/test_lr_multiplier.cpp`

**Step 1 — write the failing test.** Assert `LrMultiplier` resolve semantics D1–D3 first.

**Step 2 — run it, watch it fail** (undefined reference to `LrMultiplier`).

**Step 3 — minimal implementation:** the value type + `resolve()`.

**Step 4 — run, verify pass. Commit.**

### Task 2: Model-level application (TDD)

**Objective:** `lr_multiplier_apply(Model&, const LrMultiplier&)` rescales each parameter's delta.

**Step 1** failing test — a `Dense` layer with an overridden scale must move by exactly `s·Δ`.
**Step 2** verify RED. **Step 3** implement. **Step 4** verify GREEN. Commit.

### Task 3: The `LrMultiplierOptimizer` wrapper

**Objective:** a `Optimizer` subclass that steps a wrapped optimizer then applies the delta rescale.

**Step 1** failing test on a **real `Adam`** — this is the test that would have caught the
gradient-pre-scaling design error. **Step 2** RED. **Step 3** implement. **Step 4** GREEN. Commit.

### Task 4: Wire into the umbrella header + Makefile

**Files:** `include/nn/nn.h`, `Makefile` (link rule, `tests:` prerequisite, `run_tests` line).

**Verification:** `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` compiles
standalone, and `make run_tests` builds and runs the new suite.

### Task 5: Mutation testing

Per the TDD skill, a green suite proves nothing until a mutation is caught. Mutate the delta-rescale
line and confirm the suite fails; also verify the pristine tree is restored.

---

## Out of scope (deliberate)

- **Not** modifying any of the 39 optimizers. That is the repo-wide change this file is not
  entitled to make, for the same reason `mup.h` declined to add a per-tensor-LR hook.
- **Not** integrating with `mup_*_lr_tensor`, which returns per-tensor multipliers as a `Tensor`;
  bridging `Tensor → name → scale` needs a role-naming convention this repo has not adopted.