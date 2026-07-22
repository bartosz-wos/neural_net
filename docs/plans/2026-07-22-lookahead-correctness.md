# Lookahead Optimizer Correctness Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Correct the existing Lookahead meta-optimizer to match Zhang et al. 2019 Algorithm 1 and expose professional validation/state diagnostics.

**Architecture:** Keep Lookahead as an owning wrapper around any `Optimizer`. Snapshot slow weights before the first fast update; call the inner optimizer on every step; every `k` steps update `slow <- slow + alpha * (fast - slow)` and copy slow back to the model parameters. Preserve per-layer/per-parameter tensor state and reject malformed model or optimizer configurations before mutation.

**Tech Stack:** C++17, existing `Optimizer`/`Model`/`Layer`/`Tensor` APIs, Make-based test harness.

**References:** Zhang, Lucas, Hinton, Ba, “Lookahead Optimizer: k steps forward, 1 step back,” NeurIPS 2019, Algorithm 1; official `michaelrzhang/lookahead` implementation.

---

### Task 1: Add paper-signature and validation tests

**Objective:** Make the current reversed interpolation and missing validation fail observably before production changes.

**Files:**
- Modify: `tests/test_lookahead.cpp`

**Step 1: Write failing tests**

Add asymmetric `alpha=0.25` assertions for `slow = slow + alpha*(fast-slow)`, boundary tests proving `alpha=0` restores the old slow point and `alpha=1` accepts the fast point, constructor validation for null inner optimizer, `k <= 0`, and `alpha` outside `[0,1]`, plus step/state accessors.

**Step 2: Run test to verify failure**

Run: `make build/test_lookahead && ./build/test_lookahead`

Expected: FAIL in the `alpha=0.25` signature and validation/accessor checks because the current implementation uses the complementary interpolation and does not validate configuration.

**Step 3: Commit**

Do not commit RED independently; proceed directly to Task 2 and commit the complete green change atomically.

### Task 2: Implement the paper-correct slow update

**Objective:** Make Lookahead follow Algorithm 1 with one unambiguous synchronization path.

**Files:**
- Modify: `include/nn/optimizers/lookahead.h`
- Modify: `include/nn/optimizers/lookahead.cpp`

**Step 1: Write minimal implementation**

Validate `inner != nullptr`, `k > 0`, and `alpha in [0,1]`. Initialize slow tensors lazily from current parameters. On every call, execute the inner step and increment the counter. At each `k` boundary, compute each element as `slow += alpha*(fast-slow)` and assign `fast = slow`. Add `num_steps()`, `has_state()`, and cloned `get_slow_weight()` diagnostics. Guard parameter-count and shape drift.

**Step 2: Run focused test to verify pass**

Run: `make build/test_lookahead && ./build/test_lookahead`

Expected: all focused checks pass with no warnings.

### Task 3: Mutation-check and integrate

**Objective:** Prove the signature test distinguishes Lookahead from the existing reversed formula and register the verified behavior in project documentation.

**Files:**
- Modify temporarily then restore: `include/nn/optimizers/lookahead.cpp`
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Mutation test**

Temporarily replace `slow += alpha*(fast-slow)` with `fast += alpha*(slow-fast)` and rerun `./build/test_lookahead`.

Expected: the asymmetric interpolation and alpha-boundary tests fail.

Restore the correct implementation and rerun the focused test.

**Step 2: Run aggregate verification**

Run: `make tests`

Run: `make run_tests`

Mechanically compare binaries compiled by `tests:` with binaries invoked by `run_tests`; run any omissions individually.

Expected: all stable suites complete without failure markers; known deferred flaky suites remain documented in `NOT_FIXED.md`.

**Step 3: Update queue and commit**

Move the completed improvement into `EXPANSION_QUEUE.md` `## Done` with the corrected recurrence, API, focused check count, mutation result, and aggregate test result.

```bash
git add docs/plans/2026-07-22-lookahead-correctness.md \
  tests/test_lookahead.cpp \
  include/nn/optimizers/lookahead.h \
  include/nn/optimizers/lookahead.cpp \
  EXPANSION_QUEUE.md
git commit -m "fix(optimizers): correct Lookahead slow-weight update"
git push origin master
```
