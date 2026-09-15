# 3D Axial Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a 3D / volumetric extension of the existing `AxialAttention` layer (Ho et al. 2019) that adds a third depth-axis (T) pass on top of the existing row/column axes. This handles video, 3D medical imaging, and any `(H, W, T)` grid data with O((H+W+T)·H·W·T·d) attention cost instead of the naive O((H·W·T)²·d).

**Architecture:** Mirror the existing 2D `AxialAttention` exactly — per-axis Q/K/V/W_o projections, scaled dot-product attention with per-axis causal mask, additive residual fuse — but with three axes (H, W, T) instead of two (row, col). The row/col definitions from the 2D class are generalized: for an `(H, W, T)` volume flattened in row-major order to `(H·W·T, d_model)`, an "axis pass" is one slice of axis-coordinates holding the other two fixed.

**Tech Stack:** C++17, hand-rolled matmul-style loops on `Tensor`, `Dense` / `LayerNorm` for the block wrappers (matching the 2D class).

## References

- Ho, Kalchbrenner, Weissenborn, Salimans 2019, "Axial Attention in Multidimensional Transformers" (https://arxiv.org/abs/1912.12180) — original 2D axial; §3 (video) introduces the third depth-axis pass this file extends.
- Wang, Zhu, Wang, Lu 2021, "Axial Attention in 3D Segmentation" — applies 3D axial attention to volumetric medical image segmentation (key reference for the H/W/T formulation).

## Three-axis design (H, W, T)

For a tensor `X ∈ R^{H × W × T × d_model}` flattened row-major to `(H·W·T, d_model)`:
- `idx(i, j, t) = i·W·T + j·T + t`

Three independent axis passes share Q/K/V/W_o projections (one set per axis, like the 2D class):

```
H-axis (intra-temporal-column-row): for each query at (i, j, t) attend over i' ∈ [0, H)
   mask: only keys (i', j, t) contribute (i.e., j' = j, t' = t)
   causal on i' > i if causal_h=true

W-axis (intra-temporal-column-row): for each query at (i, j, t) attend over j' ∈ [0, W)
   mask: only keys (i, j', t) contribute (i.e., i' = i, t' = t)
   causal on j' > j if causal_w=true

T-axis (depth):                     for each query at (i, j, t) attend over t' ∈ [0, T)
   mask: only keys (i, j, t') contribute (i.e., i' = i, j' = j)
   causal on t' > t if causal_t=true   ← standard video causal convention

Fuse: out = out_h + out_w + out_t + input   (parameter-free, residual)
```

Each axis pass is O((axis length) · n_other tokens · d_model) total. Total cost is
`O((H + W + T) · H·W·T · d_model)`. For H=W=T=N the savings vs full self-attention
are `O(N²) → O(N³ · 3) → O(N³)`, i.e., quadratic→cubic in cost vs cubic→quartic in
the volume — really quadratic→linear-in-each-axis.

Per axis: num_heads multi-head split. Scores per head are independent; softmax per
head per row (i.e., per query). The full attention output `A @ V` is concatenated
back to `(H·W·T, d_model)`.

## Why per-axis causal flags (causal_h / causal_w / causal_t)

- causal_t (depth) is the *standard video convention*: previous frames cannot
  attend to future frames. Default true.
- causal_w (column) is the natural choice for left-to-right / right-to-left
  language-image grids (default true to match 2D axial column).
- causal_h (row) is sometimes non-causal: for symmetric rows, no causal
  directional bias is needed (default true to match 2D axial row).
  Future iterations could let users override per axis.

## Code style: mirror 2D axial_attention.{h,cpp}

- Same parameter layout convention (W_q etc., `output-feature × input-feature`).
- Same backward order (W_o backward first, then per-head softmax, then Q/K/V backward).
- Same duplicate-code-per-axis style (3 explicit axes instead of 2; ~3x larger than
  the 2D file, which is already 783 lines for 2 axes).
- Per-axis causal flag instead of one shared `causal` flag.
- Tensor `last_attn_h_`, `last_q_h_`, etc. cache naming convention follows
  the 2D `last_attn_row_`, `last_q_row_` naming.
- Public API style matches: `forward(input)` returns Tensor; `backward(grad, lr)`
  returns the gradient w.r.t. input; constructor takes positional args; constructor
  validates positive dims and `d_model % num_heads == 0`.

## File layout

- `include/nn/layers/attention/three_d_axial_attention.h` — class declaration + inline doc.
- `include/nn/layers/attention/three_d_axial_attention.cpp` — implementation.
- `tests/test_three_d_axial_attention.cpp` — gradient correctness tests.
- Makefile: `$(BUILD_DIR)/test_three_d_axial_attention`, tests/run_tests lists.
- Umbrella header `include/nn/nn.h`: add `#include "layers/attention/three_d_axial_attention.h"`.

---

## Task 1: Class skeletons and accessor conventions

**Files:** Create `include/nn/layers/attention/three_d_axial_attention.h`.

**Step 1:** Create the file with header guards, doc block, three class
skeletons (`ThreeDAxialAttention`, `ThreeDAxialAttentionBlock`,
`ThreeDAxialAttentionModel`) with constructor signatures, public
parameters/gradients accessor overrides, name override, and inline doc
explaining per-axis mask / fuse convention. Match the 2D axial_attention.h
style exactly (these are 3-axis extensions, no surprises in API).

**Step 2:** Run `make build/include/nn/layers/attention/three_d_axial_attention.o`
to verify the header parses.

(For now we'll ship the implementation in the .cpp immediately — see Task 2.)

## Task 2: `ThreeDAxialAttention` implementation

**Files:** Create `include/nn/layers/attention/three_d_axial_attention.cpp`.

**Step 1:** Mirror the 2D axial_attention.cpp structure exactly:
- Local `row_softmax` and GELU helpers (same as 2D)
- Constructor: random-init all 15 weights (12 W's, 3 biases), validate
  d_model > 0, H/W/T > 0, num_heads > 0, d_model % num_heads == 0.
- `parameters()`, `gradients()` returning 15 + 15 entries.
- `zero_grad()` zeroes all 15 grad tensors.
- `update_weights(lr)` applies SGD-style update to all 15 weights.
- `forward(input)`:
  - Validate `input.rows == H·W·T` and `input.cols == d_model`.
  - Project Q/K/V for all 3 axes (15 matmuls — one Q/K/V triple per axis).
  - For each axis: build per-query scores masked to keys on the same
    (other-2-coords) hyperplane, optionally with causal mask on the running
    axis coordinate, row-softmax, A @ V, head_out @ W_o + b, cache.
  - Fuse: `out = out_h + out_w + out_t + input`.
- `backward(grad_output, lr)`:
  - Clone grad_output into 3 axis grads (`d_out_h = d_out_w = d_out_t = grad_output`)
    (each pre-fuse axis receives the full grad; this matches the 2D convention).
  - Per axis: d_head_out = d_out_axis @ W_o_axis^T, accumulator for grad_W_o,
    accumulator for grad_b_axis, per-head d_V/d_A/softmax-grad/d_Q/d_K chain,
    accumulate per-head into d_Q_axis/d_K_axis/d_V_axis, then accumulate
    grad_W_q/k/v_axis and d_input contribution.
  - Add residual contribution to d_input (`+= grad_output`).
- `last_attn_h_, last_attn_w_, last_attn_t_` for FD tests / inspection.

**Step 2:** Run `make build/include/nn/layers/attention/three_d_axial_attention.o`
to verify the .cpp compiles cleanly (-Wall -Wextra, no warnings).

## Task 3: `ThreeDAxialAttentionBlock` and `ThreeDAxialAttentionModel`

**Files:** Continue in `three_d_axial_attention.cpp`.

**Step 1:** `ThreeDAxialAttentionBlock(d_model, H, W, T, num_heads=1, ffn_dim=0,
causal_h=true, causal_w=true, causal_t=true)`:
- pre-LN → 3D axial attention → residual → pre-LN → GELU FFN → residual
- if `ffn_dim == 0` skip the FFN (pure attention block, matches 2D convention)
- forward/backward mirrors 2D `AxialAttentionBlock` exactly, just plugs in 3D axial.

**Step 2:** `ThreeDAxialAttentionModel(d_input, d_model, d_output, H, W, T,
num_blocks=2, num_heads=1, ffn_dim=0, causal_h=true, causal_w=true, causal_t=true)`:
- input projection → N blocks → final LN → per-token classifier
- mirrors 2D `AxialAttentionModel`.

**Step 3:** Compile + verify.

## Task 4: Test file skeleton

**Files:** Create `tests/test_three_d_axial_attention.cpp`.

**Step 1:** Mirror test_axial_attention.cpp structure:
- `passed`/`failed` counters, `check(name, cond, err)` helper, max_abs_diff / rel_err
  helpers, l2_loss_value / l2_loss_grad helpers, FD input-gradient helper.
- FD gradient helpers: `fd_param_grad_W_q_h(...)`, etc. — a single helper that
  perturbs a weight tensor entry, calls forward, computes FD, returns a Tensor.
- First test: `test_three_d_axial_constructor_validation` (4 invalid + 1 valid).
  Run it standalone. Then move on.

**Step 2:** Run the new binary — expect 1 test passes.

## Task 5: Forward shape + finiteness + degenerate-dim tests

**Tests:** Add to `tests/test_three_d_axial_attention.cpp`.

- Forward shape (H=2, W=3, T=2, d_model=4, num_heads=2) input → (12, 4).
- Output finite (no NaN/Inf) for random seed.
- Output non-trivial (L2 norm > 1e-6 → not all zeros).
- H=1, W=1, T=1 degenerate forward (should still be finite).
- W=1 only-axis: only W and T axes are non-degenerate.

## Task 6: Mask correctness tests

**Tests:** Add to `tests/test_three_d_axial_attention.cpp`.

- H-axis mask: only same (j, t) coords get nonzero scores (the relevant subset of
  `last_attn_h_[q_idx][k_idx]` sums to 1; all others are 0).
- W-axis mask: only same (i, t) coords get nonzero.
- T-axis mask: only same (i, j) coords get nonzero.
- Causal mask: per-axis causal blocks the future-direction.
- Non-causal on a single axis allows the future tokens to attend.

## Task 7: FD gradient correctness (input + all 15 weights + 3 biases)

**Tests:** Add to `tests/test_three_d_axial_attention.cpp`.

For each gradient target W, perturb W[i][j] by ±eps, recompute forward, take
central FD. Compare to analytical `grad_W[i][j]`:
- input grad (random non-uniform init), rel_err < 1e-4
- W_q_h, W_k_h, W_v_h, W_o_h (4 FD per axis), rel_err < 1e-5 each
- W_q_w, W_k_w, W_v_w, W_o_w (4)
- W_q_t, W_k_t, W_v_t, W_o_t (4)
- b_h, b_w, b_t (3 biases), rel_err < 1e-5 each

If a gradient check fails, the rules from systematic-debugging apply — *is the
test config degenerate first?* Random non-uniform init at scale 0.05 or so is
the standard non-degenerate config; central FD with eps=1e-5. If the impl uses
the same convention as 2D axial_attention.cpp, this should match to ~1e-5
without surprises.

## Task 8: Block + Model tests

**Tests:** Add to `tests/test_three_d_axial_attention.cpp`.

- Block constructor validation (3 invalid + 1 valid).
- Block forward shape with and without FFN.
- Block FD input gradient (rel_err < 1e-3 — LN+FFN chain).
- Model constructor validation (3 invalid + 1 valid).
- Model forward shape (H=2, W=3, T=2, num_blocks=2).
- End-to-end training: random 3D regression target, MSE loss must decrease
  > 30 % over ~100 SGD steps at lr=0.01.
- update_weights moves all 15 parameter groups.

## Task 9: Makefile + umbrella + final wiring

**Files:** Edit `Makefile`, `include/nn/nn.h`.

- Add `$(BUILD_DIR)/test_three_d_axial_attention: $(LIB_OBJS) $(BUILD_DIR)/test_three_d_axial_attention.o`
  followed by the link rule.
- Add the test binary to the `$(BUILD_DIR)/test_...` aggregate dep in the
  `tests:` target.
- Add `@echo "=== Running 3D Axial Attention Tests ===" && ./$(BUILD_DIR)/test_three_d_axial_attention`
  to `run_tests`.
- Add `#include "layers/attention/three_d_axial_attention.h"` to `include/nn/nn.h`
  near the existing `axial_attention.h` line.
- Verify the umbrella header still compiles standalone
  (`g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`).

## Task 10: Move EXPANSION_QUEUE entry, commit, push

- Move the `## Ideas` bullet "3D Attention / Axial Attention" to `## Done`,
  one-line summary describing what was built and test results.
- `git add` specific files, commit with the conventional
  `feat(attention): 3D Axial Attention (H, W, T) — three-axis attention + blocks + model`
  message style. Push to master.

---

## Failure-mode handbook (axis duplication trap)

The 2D file duplicates the forward/backward body for row and column. The 3D
file duplicates 3x instead of 2x. Most gradient bugs in this codebase come
from one of:

1. **Two distinct quantities in one scratch buffer.** Each axis MUST have its
   own d_Q/d_K/d_V/d_attn/d_head_out/d_pre_soft scratch variables; a shared
   buffer that you "reuse across axes" will collapse them.
2. **Mask score-table reuse.** The `(n, n)` scores tensor for one axis must
   not be carried over to another axis.
3. **Causal condition swapped.** the "running axis" of an axis pass is the
   one whose coordinate CHANGES. H-axis: i. W-axis: j. T-axis: t. Easy to
   write `kj > qj` for the W-axis but accidentally also for the H-axis
   (would mask out other rows entirely).
4. **Per-head softmax using axis-local row instead of per-head row.** With
   num_heads > 1, the head split uses `h_off = h * head_dim_;` see 2D axial
   for the canonical pattern.

To defend against these, the test suite includes (a) mask correctness tests
that catch a wrong axis-of-mask, (b) cross-axis isolation tests (gradient
through axis A doesn't leak into axis B's weights in a degenerate way),
and (c) FD gradient checks at machine precision per axis.

## Reference files

- `include/nn/layers/attention/axial_attention.{h,cpp}` — the 2D class to mirror.
- `tests/test_axial_attention.cpp` — test style to mirror (19 tests, 31 checks).
