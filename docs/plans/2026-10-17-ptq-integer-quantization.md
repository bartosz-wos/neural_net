# Post-Training Integer Quantization — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `include/nn/utils/ptq.{h,cpp}` — symmetric int8 post-training
quantization with per-tensor and per-channel scales, round-tripping through a
real `Model`, plus a straight-through estimator for quantization-aware training.

**Architecture:** Model-free by construction — every core is a free function over
`Tensor`s, so the whole file is testable with no `Model`, no optimizer and no
network. This is the same shape `utils/vat.h`, `utils/vqcodebook.h` and
`utils/ortho_penalty.h` were built with, and it is what made those sessions'
suites possible without a network fixture.

**Tech Stack:** C++17, the repo's `Tensor` (strictly 2-D, `operator()(r, c)`
with **no bounds check**), g++ at the project's real `-O2 -march=native
-Wall -Wextra`, Makefile with the `print-lib-objs` link-set target.

---

## 0. The gap, measured rather than assumed

```
$ grep -rilE 'int8|uint8|zero_point|dequant|per.channel.*scale' include/ \
    | grep -viE 'vq_vae|vqcodebook|bitnet|log_linear'
<empty>
```

Every hit for a low-precision scheme in this repo is one of three things:
`vq_vae.{h,cpp}` / `vqcodebook.{h,cpp}` (VQ-VAE's *vector* quantization — the
indices are integers, which is a **different object**: there is no fixed grid,
no scale, and no dequantization), `layers/utility/bitnet.h` (1-bit *training*),
and per-channel `gamma` **scales** in normalization layers (a learned affine,
not a quantization grid).

So: the repo can *train* in low precision (`bitnet.h`), can *index* with integer
codes (`vq_vae.h`), and can *compress* structurally (`magnitude_pruning.h`), and
**has no way to take a trained float model and store its weights as int8**. That
is the deployment step between "training finished" and "inference on device",
and it is the one every one of those three files is missing.

**Citation, verified 2026-10-17 by TITLE query against the arXiv API** (the rule
that has caught nine wrong IDs in this queue — never trust an ID from memory):

```
$ curl -G 'http://export.arxiv.org/api/query' \
    --data-urlencode 'search_query=ti:"Integer-Arithmetic-Only Inference"'
['http://arxiv.org/abs/1712.05877v1']
['Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference']
```

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew
Howard, Hartwig Adam, Dmitry Kalenichenko. **arXiv:1712.05877.** A title query
for two *other* int8 papers this session expected (`ti:"Data Quantization for Low
Bit Width..."`, `ti:"8-bit approximation for INT8-based..."`) returned **zero
results**, which is the wrong-ID problem catching itself before it cost a
session.

## 1. The scheme

Paper Sec. 3, symmetric quantization (their default):

```
scale = max|w| / 127
q_k   = clamp(round(w_k / scale), -127, 127)
w_hat = q_k * scale
```

## 2. Measured findings that shape the plan

All from `tests/ptq_oracle.py`, which self-checks before emitting a number.

**(F1) A centred-FD gradient check CANNOT verify this forward.** The
quantize→dequantize map is piecewise-constant in `w`. Measured: perturbing
`w[0]` by `1e-7` changes the output by **exactly 0.0**, for a correct
straight-through estimator and an arbitrarily wrong one alike. So FD reads 0
for both and cannot discriminate. *Verification must be by VALUE, plus
mutation testing on the backward.* (This is systematic-debugging §5c-zeroth:
a hard/piecewise-constant forward.)

**(F2) The one exception is the max entry, and that is a real property.**
Measured per-entry FD on `[0.37, -1.21, 2.05, 0.0]` → `[0.0, 0.0, 1.0, 0.0]`.
Entry 2 is the tensor's own maximum, and `scale = max|w|/127` **moves with it**,
dragging the whole grid. So FD is vacuous on 3 of 4 entries and *not* on the
fourth. A clean constant ratio across every entry would have meant a rounding
bug; exactly one non-zero entry means the max-derived scale.

**(F3) The rounding mode is a real semantic no tolerance test can see.** numpy's
`rint` is half-to-EVEN; C's `std::round` is half-AWAY-FROM-ZERO. On a fixture
built to hit half-grid points (`w/scale = [63.5, 126.5, 127.0, -63.5, -126.5,
0.5]`) they disagree on **3 of 6** entries. The C++ must use `std::round`, and
the suite pins it with the measured codes `[64, 127, 127, -64, -127, 1]`.

**(F4) PER-CHANNEL NEVER LOSES — a theorem, not a failed search.** A 3×3 sweep
over `big ∈ {1e2,1e4,1e6} × small ∈ {1e-3,1e-6,1e-9}` found **no** shape where
per-channel has worse L2 error. The proof: per-channel error on channel `c` is
bounded by `max|w_c|/(2·127)`, per-tensor by `max|w_all|/(2·127)`, and
`max|w_c| ≤ max|w_all|`. Finer granularity **dominates pointwise**. *The plan's
first draft asserted a "per-channel loses" fixture; that assertion would have
been false, so it is not in the suite.*

**(F5) A zero tensor must return zeros, not NaN.** `max|w|/127` is `0/127 = 0`
here, but a `scale = 0` dequantization is the trap the VQ-VAE session already
hit once (the reference's `decay=1` returned an all-NaN codebook). Guarded and
pinned.

## 3. Why activation checkpointing was REJECTED (recorded so it is not re-tried)

The other measured absence this session, refuted for *this* repo:

- `grep -rilE 'activation_checkpoint|gradient.checkpoint|recompute' include/` →
  **0 hits** (every `recompute_` is a scheduler's internal method).
- **152 headers already cache `last_input` for their own `backward`.**
  Activation checkpointing trades *stored activations* for *recomputation*. With
  152 layers caching their inputs, there is nothing left to trade: the feature
  would save **zero memory** here.
- Every stochastic layer draws from the **global `rand()`**. Replaying a
  checkpointed segment mid-backward would draw a *different* mask — measured
  `sum|fwd − fwd| = 3.10` across two forwards of the same model through a
  training `Dropout`, versus `0.0` for the deterministic control.
- Making the replay exact needs a capturable RNG. **Eight idioms measured, none
  worked**: `setstate(h)` points the global generator *at* the buffer that
  `rand()` then advances in place, so a second restore rewinds to `P+n`, not
  `P`; and copying 256+ bytes back into glibc's 128-byte state buffer
  **corrupted the heap** (`munmap_chunk(): invalid pointer`). A `Model`-level
  solution would need the repo to *own* an RNG and re-seed ~152 layers — a
  repo-wide change that this queue is not entitled to make unilaterally.

**Refuted by measurement, not by taste. Do not re-pop it without first changing
the `last_input` convention across 152 headers.**

---

## 4. Tasks

### Task 1: RED — the failing test

**Files:** Create `tests/test_ptq.cpp`

Write the full suite against `nn/utils/ptq.h` before the header exists. Every
expected value is transcribed from `tests/ptq_oracle.py` output, not from
memory. Verify the RED:

Run: `make build/test_ptq` → expect **undefined references** to `ptq_*`
symbols only (a RED that fails for a typo proves nothing — check the error text).

### Task 2: GREEN — the header and core

**Files:** Create `include/nn/utils/ptq.h`, `include/nn/utils/ptq.cpp`

Minimal code to pass Tasks 1's first block only. Then re-run.

### Task 3: Add the next behaviour, one TDD cycle at a time

Each cycle: one new assertion → run → **watch it fail** → minimal code → run →
commit. Order: per-channel scales, round-trip error metrics, the Model-level
`PtqConfig` / `quantize_model`, the straight-through estimator.

### Task 4: Wire into `nn.h` and the Makefile

Umbrella header + `tests:` prerequisite + `run_tests` line + link rule.
**Verify the umbrella compiles standalone:**
`g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`

### Task 5: Mutation harness

Follow the repo's accumulated defences (this is its **tenth** harness):
- Get the link set from `make print-lib-objs` (measured 302 objects) — never
  regex the Makefile.
- Verify each `sed` **applied** via `diff -q`; a no-op edit is `MUT-NOOP`, not
  a survivor.
- Snapshot **both** `.cpp` and header if either is mutated; restore atomically
  (`cp base tmp && mv -f tmp src`).
- `MUT-BROKEN` means the build failed — never a verdict.
- Count crashes (`MUT-CRASH`) in their own column.
- Assert the object count `>= 10` or abort before reporting anything.

### Task 6: Verify and commit

`make run_tests` → no new failures. ASan+UBSan clean. Zero warnings at the
project's real flags. Commit, push, update `EXPANSION_QUEUE.md`.

## 5. Anti-patterns this repo has paid for

- Never index a `Tensor` outside `(rows, cols)` — `operator()` is
  `data[r*cols + c]` with **no bounds check** (has produced heap corruption
  three times).
- A clean constant ratio across every cell is the signature of *the harness
  differentiating the wrong function*, not of an implementation bug.
- A test that passes on the first run tests existing behaviour.
- Assert the **magnitude**, not just agreement: FD is structurally blind to
  constant-factor bugs.