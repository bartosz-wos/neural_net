# ADOPT Optimizer Implementation Plan

> **For Hermes:** Implement task-by-task with strict RED→GREEN TDD and mutation verification.

**Goal:** Add the NeurIPS 2024 ADOPT optimizer, which normalizes each current gradient by the *previous* second moment before updating momentum.

**Architecture:** Follow the official `iShohei220/adopt` implementation: lazy `(m, v)` state per parameter; the first call initializes `v = g²` without moving parameters; subsequent calls compute `g_norm = clip(g / max(sqrt(v_prev), eps), ±(t-1)^clip_exp)`, then update `m`, parameters, and finally `v`. Support coupled (paper default) and decoupled weight decay while retaining scheduler-compatible inherited learning rate.

**Tech Stack:** C++17, existing `Optimizer` / `Model` / `Tensor` APIs, Makefile test binaries.

**References:** Taniguchi et al., “ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate,” NeurIPS 2024, arXiv:2411.02853; official implementation `github.com/iShohei220/adopt/src/adopt/adopt.py` (inspected 2026-08-02).

---

### Task 1: Specify the public contract with failing tests

**Objective:** Lock defaults, validation, state shape, first-step initialization, and the defining previous-second-moment recurrence.

**Files:**
- Create: `tests/test_adopt.cpp`
- Modify: `Makefile`

**Steps:**
1. Add tests for defaults `lr=1e-3`, `beta1=0.9`, `beta2=0.9999`, `eps=1e-6`, `clip_exp=0.25`, `weight_decay=0`, `decoupled=false`, `t=1`.
2. Add constructor/setter validation and lazy state shape tests.
3. Add a first-step test proving `v=g²`, `m=0`, parameters unchanged, and `t` advances.
4. Add a two-step hand-derived signature with `beta1=beta2=0`, `eps=1`, no clipping: first gradient 2 initializes `v=4`; second gradient 1 uses denominator 2, producing normalized gradient/momentum `0.5`, parameter `-0.5`, then updates `v` to 1. This distinguishes ADOPT from Adam/current-v normalization.
5. Register `build/test_adopt` and `run_tests` entries.
6. Run `make build/test_adopt`; expected RED: missing `nn/optimizers/adopt.h`.

### Task 2: Implement the minimal optimizer

**Objective:** Make the focused behavioral contract pass without unrelated refactors.

**Files:**
- Create: `include/nn/optimizers/adopt.h`
- Create: `include/nn/optimizers/adopt.cpp`
- Modify: `include/nn/nn.h`

**Steps:**
1. Add validated constructor/setters/accessors and lazy `(m,v)` state.
2. Add parameter/gradient count and shape guards.
3. Implement first-step `v += g²` and no parameter update.
4. On later steps, apply coupled decay to the gradient when configured, normalize with *old* `v`, clip at `(t-1)^clip_exp`, update momentum, optionally apply decoupled decay, update parameters, then update `v`.
5. Clear gradients once per layer and increment global `t` once per optimizer call.
6. Add the umbrella-header include.
7. Run `make build/test_adopt && ./build/test_adopt`; expected GREEN with no failures.

### Task 3: Harden behavior and prove non-vacuousness

**Objective:** Cover clipping, both decay modes, malformed layers, isolation, determinism, and useful training behavior.

**Files:**
- Modify: `tests/test_adopt.cpp`
- Modify only if a confirmed defect appears: `include/nn/optimizers/adopt.{h,cpp}`

**Steps:**
1. Add tests for dynamic clipping, coupled versus decoupled weight decay, state isolation, malformed count/shape guards, gradient clearing, and deterministic trajectories.
2. Add a small linear-regression test showing loss reduction.
3. Mutate the denominator from old `v` to current-gradient/current-`v` behavior; the hand-derived signature must fail. Restore it.
4. Mutate away first-step no-op; its dedicated test must fail. Restore it.
5. Run focused test and umbrella syntax check:
   - `./build/test_adopt`
   - `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`

### Task 4: Integrate, document, and verify the repository

**Objective:** Ship ADOPT as the next expansion while preserving all stable behavior.

**Files:**
- Modify: `EXPANSION_QUEUE.md`
- Existing integration files above

**Steps:**
1. Move the queue from empty to a Done entry documenting the verified ADOPT implementation and focused test count.
2. Run `make tests` to compile every registered test binary.
3. Run `make run_tests`; expected: all registered stable suites pass. If a known deferred flaky suite manifests, verify it matches `NOT_FIXED.md` and report exactly rather than masking it.
4. Inspect `git diff --check`, `git diff --stat`, `git status`, and the exact diff.
5. Commit only the plan and ADOPT-specific files using conventional commits, then push `master`.
