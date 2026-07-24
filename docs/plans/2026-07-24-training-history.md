# Training History Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a callback-friendly in-memory training history with deterministic CSV export for epoch metrics.

**Architecture:** `TrainingHistory` owns an ordered `std::vector<TrainingEpochRecord>`. Callers record one finite observation per epoch or bind `callback()` directly to `Trainer::set_epoch_callback`; the utility provides indexed access, best-epoch queries, reset, and fixed-schema CSV output without changing `Trainer` or `Model`.

**Tech Stack:** C++17 standard library, existing self-contained test harness, Makefile.

---

### Task 1: Define behavior with a failing focused test

**Objective:** Specify configuration-free recording, validation, queries, callback integration, reset, and CSV behavior before implementation.

**Files:**
- Create: `tests/test_training_history.cpp`
- Modify: `Makefile`

**Step 1: Write failing test**

Cover empty state, ordered records, duplicate/out-of-order epoch rejection, transactional non-finite rejection, minimize/maximize best queries, callback recording, reset, deterministic precision-preserving CSV, empty CSV, and unwritable-path errors.

**Step 2: Run test to verify failure**

Run: `make build/test_training_history`
Expected: FAIL because `nn/utils/training_history.h` does not exist.

**Step 3: Commit**

Do not commit RED separately; continue immediately to Task 2.

### Task 2: Implement the minimal training-history utility

**Objective:** Make the focused behavioral contract pass without changing training-loop infrastructure.

**Files:**
- Create: `include/nn/utils/training_history.h`
- Create: `include/nn/utils/training_history.cpp`
- Test: `tests/test_training_history.cpp`

**Step 1: Write minimal implementation**

Define `TrainingEpochRecord { size_t epoch; double train_loss; double val_loss; double learning_rate; }` and `TrainingHistory` with `record`, `callback`, `size/empty/records/at/latest`, `best_epoch(metric, mode)`, `to_csv`, `save_csv`, and `clear`. Reject non-finite metrics and non-increasing epoch indices before mutation. CSV schema is fixed as `epoch,train_loss,val_loss,learning_rate` and uses `max_digits10` precision.

**Step 2: Run focused test to verify pass**

Run: `make build/test_training_history && ./build/test_training_history`
Expected: all focused checks pass.

**Step 3: Mutation-test**

Temporarily reverse the best-epoch comparison and confirm dedicated minimize/maximize assertions fail; restore it and rerun.

### Task 3: Integrate and verify the public surface

**Objective:** Expose the utility and ensure aggregate build/run coverage.

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Add umbrella include and aggregate test wiring**

Add `utils/training_history.h`, the focused binary to `tests:`, and its invocation to `run_tests`.

**Step 2: Compile public umbrella header**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
Expected: exit 0, no diagnostics.

**Step 3: Run aggregate gates**

Run: `make tests` then `make run_tests`.
Expected: all 99 registered binaries compile; all 95 stable suites execute successfully, with the four deferred unstable suites remaining omitted per `NOT_FIXED.md`.

**Step 4: Update queue and commit**

Move Training History to `## Done` with the exact focused/aggregate results, stage only the listed files, and commit `feat(training): add training history and CSV export`.
