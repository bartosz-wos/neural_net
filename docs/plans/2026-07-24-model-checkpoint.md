# Model Checkpoint Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add generic, callback-friendly model parameter checkpoints that save the latest or best monitored epoch and restore into an existing model safely.

**Architecture:** `ModelCheckpoint` serializes every tensor exposed by `Layer::parameters()` into a versioned binary file, independent of concrete layer type. Loading targets an already-constructed model and validates tensor count and every shape before mutating any parameter. Best-only mode tracks a finite metric with strict `min_delta`; latest mode writes every observation. File replacement uses a sibling temporary file followed by rename so a failed write does not destroy the previous checkpoint.

**Tech Stack:** C++17 standard library (`fstream`, `cstdio`, fixed-width integers), existing `Model`/`Layer`/`Tensor` APIs, self-contained C++ test harness, Makefile.

---

### Task 1: Define the checkpoint contract with a failing test

**Objective:** Specify generic save/load, monitoring, validation, atomicity-facing behavior, callback integration, and reset before implementation.

**Files:**
- Create: `tests/test_model_checkpoint.cpp`
- Modify: `Makefile`

**Step 1: Write failing test**

Cover constructor defaults/non-defaults; empty path and invalid `min_delta`; manual `save` and `load`; deep value restoration across multiple layers/tensors; best-only minimize/maximize with strict-delta and earliest-tie behavior; latest mode overwriting each epoch; non-finite and negative-epoch rejection before state/file mutation; callback binding to a `Model`; reset preserving configuration; missing/bad-magic/truncated file rejection; parameter-count and shape mismatch rejection transactionally; empty-model round trip.

**Step 2: Run test to verify failure**

Run: `make build/test_model_checkpoint`
Expected: FAIL because `nn/utils/model_checkpoint.h` does not exist.

### Task 2: Implement the minimal generic checkpoint utility

**Objective:** Make the focused behavioral contract pass without relying on incomplete concrete-layer architecture serialization.

**Files:**
- Create: `include/nn/utils/model_checkpoint.h`
- Create: `include/nn/utils/model_checkpoint.cpp`
- Test: `tests/test_model_checkpoint.cpp`

**Step 1: Implement the public API**

Define `ModelCheckpointMode { MINIMIZE, MAXIMIZE }` and `ModelCheckpoint(path, save_best_only=true, mode=MINIMIZE, min_delta=0)`. Provide `step(epoch, metric, model)`, `callback(model)`, `save(model)`, `load(model)`, `reset()`, configuration/state accessors, and `num_saved()`.

**Step 2: Implement transactional serialization**

Write magic/version/tensor count, then `(rows, cols, data)` for each parameter to `<path>.tmp`; check every write and rename only after close succeeds. On load, parse the entire file into temporary tensors, reject malformed/trailing data, validate count/shapes against the model, then copy values in one mutation phase.

**Step 3: Run focused test**

Run: `make build/test_model_checkpoint && ./build/test_model_checkpoint`
Expected: all focused checks pass.

**Step 4: Mutation-test**

Temporarily invert the minimize comparator and confirm the best-only trajectory assertions fail; restore and rerun. Temporarily remove the final parameter copy and confirm restoration assertions fail; restore and rerun.

### Task 3: Integrate and verify the public surface

**Objective:** Expose checkpointing from the umbrella header and keep aggregate test registration honest.

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Wire public and aggregate surfaces**

Add `utils/model_checkpoint.h`, add `build/test_model_checkpoint` to `tests:`, and invoke it from `run_tests`.

**Step 2: Compile the public umbrella header**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
Expected: exit 0 with no diagnostics.

**Step 3: Verify compiled-vs-executed coverage**

Mechanically compare `tests:` prerequisites with `run_tests` commands. Expected omissions are exactly the four deferred suites in `NOT_FIXED.md`; run all newly added focused tests.

**Step 4: Run aggregate gates**

Run: `make tests` then `make run_tests` in the foreground.
Expected: every registered binary compiles and every stable registered suite succeeds.

**Step 5: Update queue, commit, and push**

Add Model Checkpoint to `## Done` with exact focused/mutation/aggregate evidence; stage only listed files; commit `feat(training): add generic model checkpoints`; push `master` to `origin`.
