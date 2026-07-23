# Early Stopping Utility Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a callback-friendly `InMemoryEarlyStopping` utility that stops after a configured number of non-improving validation metrics and can restore the model parameters from the best observation.

**Architecture:** `InMemoryEarlyStopping` remains independent of `Trainer`: callers pass each epoch's metric and model to `step(metric, model)`, then break when it returns `true`. The utility snapshots every parameter returned by each layer's `parameters()` using deep `Tensor::clone()` copies; restore validates parameter count and shapes before writing values back. Minimize and maximize semantics share one improvement predicate.

**Tech Stack:** C++17, existing `Model`/`Layer`/`Tensor` APIs, self-contained Makefile test executable.

---

### Task 1: Specify metric tracking and patience semantics

**Objective:** Establish the public API and exact stop boundary with behavior-first tests.

**Files:**
- Create: `tests/test_early_stopping.cpp`
- Modify: `Makefile:136-145,370-395,397-490`

**Step 1: Write failing tests**

Create a test fixture using a one-layer `Dense` model and assertions for:

```cpp
InMemoryEarlyStopping stop(2, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, false);
check("first observation continues", !stop.step(1.0, model));
check("first miss continues", !stop.step(1.1, model));
check("second consecutive miss stops", stop.step(1.2, model));
check("bad epoch count equals patience", stop.num_bad_epochs() == 2);
```

Also cover constructor defaults, invalid patience/min_delta, first-observation state, a genuine improvement resetting the bad-epoch count, and `min_delta` equality counting as no improvement.

**Step 2: Run test to verify failure**

Run: `make build/test_early_stopping`
Expected: FAIL because `nn/utils/early_stopping.h` does not exist.

**Step 3: Write minimal implementation**

Create the class declaration and metric-only implementation:

```cpp
enum class InMemoryEarlyStoppingMode { MINIMIZE, MAXIMIZE };

class InMemoryEarlyStopping {
public:
    explicit InMemoryEarlyStopping(size_t patience = 5,
                           double min_delta = 0.0,
                           InMemoryEarlyStoppingMode mode = InMemoryEarlyStoppingMode::MINIMIZE,
                           bool restore_best_weights = true);
    bool step(double metric, Model& model);
    void reset();
    // accessors
};
```

Improvement is strict and delta-aware:

```cpp
return mode_ == InMemoryEarlyStoppingMode::MINIMIZE
    ? metric < best_metric_ - min_delta_
    : metric > best_metric_ + min_delta_;
```

The first finite metric is always accepted as the baseline. Stop exactly when `num_bad_epochs_ >= patience_`. Reject non-finite metrics.

**Step 4: Run test to verify pass**

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: all Task 1 checks PASS.

### Task 2: Add deep best-weight snapshots and restoration

**Objective:** Preserve and restore the exact best model parameters without aliasing.

**Files:**
- Create: `include/nn/utils/early_stopping.h`
- Create: `include/nn/utils/early_stopping.cpp`
- Modify: `tests/test_early_stopping.cpp`

**Step 1: Write failing tests**

Add tests that set a known Dense weight/bias at the best metric, mutate them on later non-improvements, then verify automatic restoration at the stop boundary:

```cpp
InMemoryEarlyStopping stop(2, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, true);
set_params(model, 1.0);
stop.step(0.5, model);
set_params(model, 9.0);
stop.step(0.6, model);
set_params(model, 10.0);
check("stop reached", stop.step(0.7, model));
check("best weights restored", all_params_equal(model, 1.0));
```

Cover explicit `restore_best(model)`, `restore_best_weights=false`, deep-copy isolation, empty models, and calling restore before any observation.

**Step 2: Run test to verify failure**

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: FAIL in restoration assertions because snapshot support is missing.

**Step 3: Write minimal implementation**

Flatten parameters in stable layer/parameter order. On each improvement:

```cpp
best_parameters_.clear();
for (auto& layer : model.layers)
    for (Tensor* parameter : layer->parameters())
        best_parameters_.push_back(parameter->clone());
```

On restore, collect current pointers, require the same count and shapes, then copy each scalar from the snapshot. If automatic restoration is enabled, restore once at the stop transition.

**Step 4: Run test to verify pass**

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: all restoration checks PASS.

### Task 3: Cover maximize, reset, topology guards, and idempotence

**Objective:** Harden edge cases and prove the state machine does not drift after stopping.

**Files:**
- Modify: `tests/test_early_stopping.cpp`
- Modify: `include/nn/utils/early_stopping.{h,cpp}`

**Step 1: Write failing tests**

Add tests for:
- `MAXIMIZE` accepting larger metrics and rejecting smaller ones.
- `reset()` clearing stopped/best/bad-count/step-count/snapshot state while preserving configuration.
- Parameter-count drift and parameter-shape drift throwing during restore.
- `step()` after stopping returning true without changing counters or overwriting restored parameters.
- `NaN`, `+inf`, and `-inf` metrics throwing without mutating state.
- Best-step index and observation count.

**Step 2: Run test to verify failure**

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: FAIL only in the newly added behavior assertions.

**Step 3: Write minimal implementation**

Add the missing state transitions and guards. Keep `step()` idempotent after `stopped_` becomes true, and validate finiteness before any state mutation.

**Step 4: Run test to verify pass**

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: focused suite PASS with no warnings.

### Task 4: Mutation-test the stop boundary and snapshot

**Objective:** Prove tests distinguish plausible broken implementations.

**Files:**
- Temporarily modify and restore: `include/nn/utils/early_stopping.cpp`

**Step 1: Mutate patience boundary**

Temporarily replace `num_bad_epochs_ >= patience_` with `num_bad_epochs_ > patience_`.

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: the exact-boundary tests FAIL.

Restore the production condition.

**Step 2: Mutate snapshot ownership**

Temporarily skip copying improved parameters into `best_parameters_` (or overwrite the snapshot on a non-improvement).

Run: `make build/test_early_stopping && ./build/test_early_stopping`
Expected: restoration/deep-copy tests FAIL.

Restore production code and rerun focused tests; expected: PASS.

### Task 5: Publish the utility and verify the repository

**Objective:** Wire the finished utility into the public API, aggregate build, queue ledger, and full test run.

**Files:**
- Modify: `include/nn/nn.h:53-61`
- Modify: `Makefile`
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Add public/build wiring**

Add `#include "utils/early_stopping.h"`, a `build/test_early_stopping` rule, the target under `tests:`, and execution under `run_tests`.

**Step 2: Verify compiled-vs-executed coverage**

Mechanically compare test prerequisites and `run_tests` commands. Expected: no newly compiled test omitted from execution.

**Step 3: Run focused and aggregate gates**

Run:

```bash
make build/test_early_stopping
./build/test_early_stopping
make tests
make run_tests
```

Expected: focused suite passes; aggregate build succeeds; all stable suites execute without failure markers.

**Step 4: Update queue**

Move InMemoryEarlyStopping from `## Ideas` to `## Done` with API, safety guarantees, focused check count, mutation results, and aggregate verification.

**Step 5: Commit**

```bash
git add docs/plans/2026-07-23-early-stopping.md \
  tests/test_early_stopping.cpp \
  include/nn/utils/early_stopping.h include/nn/utils/early_stopping.cpp \
  include/nn/nn.h Makefile EXPANSION_QUEUE.md
git commit -m "feat(training): add early stopping with best-weight restore"
```
