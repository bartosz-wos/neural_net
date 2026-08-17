# xLSTM Block (Beck et al. 2024, §5) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the canonical xLSTM architecture block and stackable model — composing `sLSTM` (scalar cell) and `mLSTM` (matrix cell) sublayers with shared pre-norm and a GELU FFN into a single residual stream — with full analytical backward and machine-precision gradient checks against centered FD.

**Architecture:** Three classes:
1. `XLSTMBlock(d_model, slstm_hidden, mlstm_hidden, ffn_mult, cell_type, slstm_per_block)` — A single Beck et al. 2024 §5 xLSTM block. Two pre-norm residual sublayers composed sequentially: (a) **mixer sublayer** — one `sLSTM` cell + (optionally) one `mLSTM` cell after it (controlled by `cell_type ∈ {SLSTM_ONLY, MLSTM_AFTER, BOTH_PARALLEL}`) under shared pre-norm `LN_1`; (b) **FFN sublayer** — 2-layer GELU MLP under pre-norm `LN_2`. Residual `out = x + mixer(LN_1(x)) + ffn(LN_2(x))`. Layout is `(T, d_model)` end-to-end (sLSTM/mLSTM cells use `(T, hidden)` and a Dense projects `hidden → d_model`).
2. `XLSTMModel(input_dim, d_model, output_dim, num_layers, slstm_hidden, mlstm_hidden, ffn_mult, cell_type)` — Embed → stack of N `XLSTMBlock` (each block has independent state) → final LayerNorm → classifier.
3. State is per-block (each `XLSTMBlock` has its own sLSTM/mLSTM cells — these hold per-cell recurrence state which is owned by the cell itself for `forward`/`backward`; blocks stack without sharing state across blocks).

**Tech Stack:** Existing infra — `Layer`/`Tensor`/`Dense` from core, reuses `SLSTMCell` (`include/nn/layers/recurrent/xlstm.h`), `MLSTMCell` (`include/nn/layers/recurrent/mlstm.h`), `LayerNorm` (`include/nn/layers/normalization/layer_norm.h`), GELU activation via raw tensor math (matches Griffin style). SGD optimizer.

**Reference:** Beck et al. 2024, "xLSTM: Extended Long Short-Term Memory" (https://arxiv.org/abs/2404.05704), §5 (xLSTM architecture: post-LN blocks with sLSTM/mLSTM mixer sublayer + dense FFN sublayer, residual connection). The paper uses post-LN; we follow **pre-norm** to match the convention used by all other blocks in this repo (Griffin, Jamba, Conformer, Mamba-2). Residual structure (per §5): `out = x + mixer(LN(x)) + ffn(LN(x))` — the mixer is itself a sum when both cells are active.

---

## Background

The xLSTM paper (Beck et al. 2024) introduces two cell variants — sLSTM (scalar memory) and mLSTM (matrix memory) — and a canonical **block architecture** (§5) that composes these cells with normalization and a feed-forward sublayer. The repo already ships:

- `SLSTMCell` (`include/nn/layers/recurrent/xlstm.{h,cpp}`) — sLSTM with exponential gating + log-domain stabilizer. 10/10 tests pass.
- `MLSTMCell` (`include/nn/layers/recurrent/mlstm.{h,cpp}`) — mLSTM with matrix cell + covariance normalizer. 11/11 tests pass.

What's missing: the **block-level composition** that turns either or both cells into a reusable sequence block. The paper supports heterogeneous block stacks (some blocks use sLSTM, others mLSTM, others both). We support three composition modes:

| `cell_type` | Mixer sublayer |
|---|---|
| `SLSTM_ONLY` | `out_mixer = sLSTM(LN(x)) → d_model` |
| `MLSTM_AFTER` | `out_mixer = sLSTM(LN(x)) → d_model; + mLSTM(LN(x)) → d_model` |
| `BOTH_PARALLEL` | (same as MLSTM_AFTER — paper §5 actually stacks them, but the parallel variant is a clean drop-in and avoids the sequential gradient-chain complexity) |

`SLSTM_ONLY` is the default and matches the sLSTM-only variant in §A.1 of the paper.

## Why this is interesting

- Closes out the xLSTM paper: the cells are shipped but the architecture unit is missing
- Provides the first block in the repo that composes sLSTM + mLSTM into a single forward pass
- Tests will exercise the residual+pre-norm gradient chain through two heterogeneous cells (one scalar-state, one matrix-state)
- Adds a stackable `XLSTMModel` that reuses the existing `Layer` interface

## Conventions

- `Layer` interface: `forward(T, d) → (T, d)`, `backward(grad_out, lr) → grad_in`
- (T, d_model) layout end-to-end; sLSTM/mLSTM cells operate on `(T, hidden)` and the block wraps them with a Dense `hidden → d_model` projector
- Pre-norm residuals (matches Griffin/Jamba/Conformer convention in this repo)
- Public members following the Griffin/Jamba style — `forward`/`backward` plus accessors for tests

## Files to create

- `include/nn/layers/architectures/xlstm_block.h` — header for `XLSTMBlock`, `XLSTMModel`
- `include/nn/layers/architectures/xlstm_block.cpp` — implementation
- `tests/test_xlstm_block.cpp` — focused test suite
- `docs/plans/2026-08-17-xlstm-block.md` — this plan

## Files to modify

- `include/nn/nn.h` — add `#include "layers/architectures/xlstm_block.h"` after the Griffin include
- `Makefile` — add `build/test_xlstm_block` compile rule, add `$(BUILD_DIR)/test_xlstm_block` to the `tests:` target deps, and add `@echo "=== Running xLSTM Block Tests ===" && ./$(BUILD_DIR)/test_xlstm_block` to the `run_tests` recipe

---

## Implementation tasks

Each task is sized for one TDD cycle: write failing test, watch fail, write minimal code, watch pass.

### Task 1: Header skeleton + test scaffold

**Step 1:** Write `tests/test_xlstm_block.cpp` with the empty main + a `test_constructor_validates_dims` that fails (no header → compile error).

**Step 2:** Create `include/nn/layers/architectures/xlstm_block.h` with class declarations but throwing `std::logic_error("not implemented")` for `forward/backward`. Confirm compile and the test reports "feature missing".

**Step 3:** Add `XLSTMBlock` to `nn.h` umbrella and add Makefile rule. Confirm `make build/test_xlstm_block` succeeds (compiles, links).

### Task 2: Forward shape + finiteness (SLSTM_ONLY mode)

**Test:** `XLSTMBlock(d_model=4, slstm_hidden=4, ..., cell_type=SLSTM_ONLY).forward((T=3, d=4))` returns `(3, 4)`, all finite, nonzero.

**Implement:** Forward path:
```
ln1_out = LN_1(x)                            // (T, d)
slstm_out = SLSTMCell(d, slstm_hidden).fwd(ln1_out)  // (T, slstm_hidden)
projected = Dense(slstm_hidden, d).fwd(slstm_out)   // (T, d)
ln2_out = LN_2(x + projected)
ffn_out = Dense(d, mult*d) → GELU → Dense(mult*d, d) of ln2_out
out = x + projected + ffn_out
```
Cache: `last_input`, `last_ln1_out`, `last_slstm_h`, `last_proj_out`, `last_ln2_in`, `last_ln2_out`, `last_ffn_hidden`, `last_ffn_out`, `last_residual`.

### Task 3: MLSTM_AFTER mode (forward + finiteness)

**Test:** `XLSTMBlock(..., cell_type=MLSTM_AFTER)` forward shape `(T=2, d=4)` → `(2, 4)`, finite, nonzero. Both `slstm_out` and `mlstm_out` contribute.

**Implement:** When `cell_type == MLSTM_AFTER`, add the mLSTM path:
```
mlstm_out = MLSTMCell(d, mlstm_hidden).fwd(ln1_out)  // (T, mlstm_hidden)
mlstm_proj = Dense(mlstm_hidden, d).fwd(mlstm_out)    // (T, d)
mixer = projected + mlstm_proj                          // (T, d)
```
Backwards chain for MLSTM_AFTER: gradient of mixer flows back to BOTH the sLSTM and mLSTM projectors and their underlying cells.

### Task 4: Backward — input gradient (SLSTM_ONLY)

**Test:** centered FD vs analytical on input. rel_err < 1e-2 at eps=1e-4.

**Implement:** `backward(grad_out, lr)` returns `grad_in`:
```
grad_residual = grad_out
grad_x_residual = grad_residual                       // residual term
grad_x_residual += ffn_proj2.backward(grad_residual)  // ffn out → ffn hidden
grad_x_residual += gelu_backward(grad_x_residual)     // ffn hidden → ffn pre-act
grad_x_residual += ffn_proj1.backward(grad_x_residual) // ffn pre-act → ln2 input
grad_ln2_in = grad_x_residual + grad_residual          // ln2 out → ln2 in
grad_ln2_x = ln_2.backward(grad_ln2_in, lr)           // ln2 in → ln2 input (= x + projected)
grad_x_via_ln2 = grad_ln2_x                           // residual to x
grad_proj_out = grad_ln2_x                            // gradient through projected
grad_slstm_h = proj_dense.backward(grad_proj_out, lr)  // (T, slstm_hidden)
grad_ln1_out = slstm.backward(grad_slstm_h, lr)        // (T, d)
grad_ln1_x = ln_1.backward(grad_ln1_out, lr)          // (T, d)
grad_x = grad_x_residual + grad_x_via_ln2 + grad_ln1_x
```

### Task 5: Backward — sLSTM cell parameter gradient

**Test:** centered FD vs analytical on `SLSTMCell.W` (the combined `[x;h]` projection) inside the block. rel_err < 5e-2 (sLSTM has known ~5e-2 noise floor due to exp/log_sigmoid stabilizer).

**Implement:** Already covered by Task 4's chain — sLSTM's own backward computes `grad_W` and `grad_b`. Verify FD agrees.

### Task 6: Backward — Dense projector + LayerNorm + FFN parameter gradient

**Test:** centered FD vs analytical on (a) `proj_dense.W`, (b) `LN_1.gamma`, (c) `ffn_proj1.W`, (d) `ffn_proj2.W`. All rel_err < 1e-3 (Dense and LayerNorm have clean gradient chains).

**Implement:** Already covered by Task 4's chain. Verify each path independently.

### Task 7: Backward — mLSTM path parameter gradient (MLSTM_AFTER)

**Test:** centered FD vs analytical on `MLSTMCell.W` inside the block. rel_err < 5e-2 (similar to sLSTM — mLSTM has its own numerical noise from the max(1, q^T N q) normalizer).

**Implement:** Already covered by Task 3's chain — mLSTM's own backward computes `grad_W` and `grad_b`.

### Task 8: Training reduces loss

**Test:** 50 SGD steps at lr=1e-3 on a synthetic regression task (target = small linear projection of `embed(x)`). Loss decreases > 5%.

**Implement:** Not new — just verify the full chain works for end-to-end training.

### Task 9: Determinism — bit-identical forward with copied params

**Test:** Two fresh `XLSTMBlock` instances with identical config; copy `slstm.W/b`, `mlstm.W/b`, all `Dense.weights/bias`, `LN.gamma/beta` from one to the other. Forward on the same input produces bit-identical output (max abs diff = 0).

**Implement:** Add `copy_params_from(other)` helper that does the deep copy.

### Task 10: Parameter count contract

**Test:** `XLSTMBlock(d=4, slstm_hidden=4, mlstm_hidden=4, ffn_mult=2, SLSTM_ONLY)` has expected param count: SLSTMCell(4,4) = 4*4*4 + 4*4 = 80, proj_dense(4,4) = 16+4 = 20, ln1(4)+ln2(4) = 8+8 = 16, ffn = 4*(4*2) + (4*2)*4 + 4*2 + 4 = 32+32+8+4 = 76 → total = 80+20+16+76 = 192.

`MLSTM_AFTER`: + MLSTMCell(4,4) = 6*4*(4+4) + 6*4 = 6*4*8 + 24 = 192+24 = 216 → total = 192 + 216 + 4*4+4 = 428.

**Implement:** Add `count_parameters()` helper that sums all `parameters()` tensors' `data.size()`.

### Task 11: `XLSTMModel` forward + finiteness

**Test:** `XLSTMModel(input_dim=3, d_model=4, output_dim=2, num_layers=2, slstm_hidden=4, ffn_mult=2, cell_type=SLSTM_ONLY).forward((T=2, input_dim=3))` returns `(T, output_dim)` = `(2, 2)`, finite, nonzero.

**Implement:**
```
embed_out = embed.forward(input)             // (T, d_model)
x = embed_out
for each block:
    x = block.forward(x)                      // (T, d_model)
x = final_ln.forward(x)
out = classifier.forward(x)                   // (T, output_dim)
```

### Task 12: `XLSTMModel` training reduces loss

**Test:** 80 SGD steps at lr=5e-3, loss decreases > 50%.

**Implement:** Standard training loop.

### Task 13: `XLSTMModel` parameter count scales linearly

**Test:** num_layers=2 and num_layers=4; expected params scale linearly.

---

## Test plan summary

**At least 15 focused checks** (one per task plus the constructor validation):

1. Constructor validation: d_model=0 throws, slstm_hidden=0 throws, cell_type ∈ {SLSTM_ONLY, MLSTM_AFTER, BOTH_PARALLEL} only
2. Forward shape + finiteness + nonzero (SLSTM_ONLY, T=3, d=4)
3. Forward shape + finiteness + nonzero (MLSTM_AFTER, T=2, d=4)
4. Forward shape + finiteness (T=8, d=4) — longer sequence exercises BPTT
5. Input gradient FD check (SLSTM_ONLY, rel_err < 1e-2)
6. sLSTM W parameter gradient FD check (rel_err < 5e-2)
7. mLSTM W parameter gradient FD check (MLSTM_AFTER, rel_err < 5e-2)
8. proj_dense.W gradient FD check (rel_err < 1e-3)
9. LN_1.gamma gradient FD check (rel_err < 1e-3)
10. LN_2.gamma gradient FD check (rel_err < 1e-3)
11. ffn_proj1.W gradient FD check (rel_err < 1e-3)
12. ffn_proj2.W gradient FD check (rel_err < 1e-3)
13. Training reduces loss 50 SGD steps (loss decreases > 5%)
14. Determinism — bit-identical forward with copied params (max abs diff = 0)
15. Parameter count contract (SLSTM_ONLY = 192, MLSTM_AFTER = 428)
16. `XLSTMModel` forward shape + finiteness
17. `XLSTMModel` training reduces loss 80 steps (loss decreases > 50%)
18. `XLSTMModel` parameter count scales linearly with num_layers

Plus the mutation tests (run after the suite is green):
- Stub out sLSTM backward (return zeros for grad_W/grad_b) → sLSTM W grad check fails (non-vacuity)
- Stub out proj_dense backward → proj_dense W grad check fails (non-vacuity)

---

## Conventions

- Pre-norm residual (matches Griffin/Jamba/Conformer convention)
- (T, d_model) input/output layout (matches sLSTM/mLSTM expectation)
- Forward returns `out = x + mixer + ffn` (residual + mixer + ffn summed into a single stream; the mixer itself can be `proj(slstm(LN(x)))` or `proj(slstm(LN(x))) + proj(mlstm(LN(x)))`)
- Parameter tensors are public for test introspection (Dense.W, Dense.b, LN.gamma, LN.beta, plus sLSTM.W/b and mLSTM.W/b via the embedded cells)
- Single SGD-step optimizer for training tests (no Adam — keep it simple)

## Bug-fix patterns to apply from prior sessions

1. **Sequential-residual backward chain in Griffin/Jamba** — both proved the gradient chain through three pre-norm residual branches works at machine precision when each sublayer caches its forward intermediates correctly. Reuse the pattern.
2. **Public cells for test introspection** — sLSTM/mLSTM cells hold their own gradients (`grad_W`, `grad_b`). Verify FD against the block-level chain rather than re-running the cell-level chain.
3. **TDD mutation testing** — stub out one backward call (e.g. set `grad_W` to zero manually after forward), confirm at least one test fails.
4. **Hand-derived magnitude assertions** — at least one test asserts a specific known value, not just agreement (catches half-scale bugs).
5. **`copy_params_from` helper** — necessary for determinism tests on Layer instances that own Dense sublayers.

---

## End-to-end smoke test

After all 15+ checks pass, do a final end-to-end smoke:
- `XLSTMModel(input_dim=4, d_model=8, output_dim=2, num_layers=2, slstm_hidden=8, ffn_mult=2, SLSTM_ONLY)` on a synthetic regression task
- 80 SGD steps, lr=5e-3
- Loss should decrease > 50%
- Verify all parameter updates are finite

---

## Remember

Bite-sized tasks (2-5 min each), exact file paths, complete code, exact commands with expected output, verification steps. DRY. YAGNI. TDD. Frequent commits.