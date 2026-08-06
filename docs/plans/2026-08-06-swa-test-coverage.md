# SWA (Stochastic Weight Averaging) Test Coverage

> **For Hermes:** Single-task delivery: add focused test coverage for the existing `SWAOptimizer` (Izmailov et al. 2018) at `include/nn/optimizers/swa.{h,cpp}`, which was previously shipped but had no dedicated tests.

**Goal:** Add `tests/test_swa.cpp` with focused coverage of SWA's averaging logic, the user-initiated `record()` path, the `update_bn_stats()` BN forward-pass trigger, and the `SWALRScheduler` LR transitions. Register in Makefile and add to `include/nn/nn.h` umbrella for parity with the other optimizers.

**Why this addition:** The repo has 32 optimizer classes (full list at `include/nn/optimizers/`) and a `make tests` / `make run_tests` aggregate that should cover every shipped component. SWA was the most prominent one without a test file — a significant gap because SWA is a wrapper optimizer (not a vanilla SGD/Adam variant) and its averaged-weights contract is what production users actually rely on.

**Coverage (42 focused checks across 15 test functions):**

1. **Pre-warmup no-op** — averaging doesn't start until `start_after_` steps
2. **`averaged_count()` accuracy** — reflects `step_count_ - start_after_` post-start
3. **Arithmetic mean closed form** — after 5 increments of 1.0, average == 3.0
4. **`swap_to_averaged` overwrites live weights** with the running mean
5. **`record()` updates average without calling inner.step**
6. **`update_bn_stats()` runs forward passes and BN state updates**
7. **`SWALRScheduler` warmup + swa_lr transitions** — full state-machine
8. **`SWALRScheduler` zero-warmup immediate-swa edge case**
9. **`SWALRScheduler` warmup-only mode (no swa transition)**
10. **`SWALRScheduler.reset()` restores initial state**
11. **Determinism** — two fresh instances with same gradient sequence produce bit-identical averaged weights
12. **End-to-end** — averaged weights differ from live weights
13. **Inner optimizer still runs during warmup** — averaging is the only thing gated
14. **Multi-layer independence** — every layer's parameters averaged separately
15. **Param shape preservation** — `Dense(3, 5)` averaged weights still shape `(5, 3)`, bias `(1, 5)`

**Mutation-tested:** Two mutations caught 8 and 13 test failures respectively, confirming the tests are non-vacuous:
- Replacing the averaging formula with `averaged[i] = w_t` (no running mean) → 8 failures
- Commenting out `inner_->step(model)` → 13 failures

**Implementation note:** No changes to `include/nn/optimizers/swa.{h,cpp}` were needed — the existing implementation is correct, just untested. The SWALRScheduler constructor intentionally doesn't call `update_lr()` — `current_lr_` is initialized to `start_lr_` and recomputed only on `step()`. This matches the SWA-paper convention (no warmup math applied before any training step has occurred). Tests are written against this behavior.

**File layout:**
- Tests: `tests/test_swa.cpp` (NEW, 42 checks, no test fails on first run)
- Makefile: `build/test_swa` compile rule, `tests:` deps line, `run_tests:` echo
- Umbrella: `#include "optimizers/swa.h"` added to `include/nn/nn.h`
