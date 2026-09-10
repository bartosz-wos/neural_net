# Multi-Scale Deformable 1D Attention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).
> **Skills loaded:** systematic-debugging (numerical-gradient trap awareness), test-driven-development (mutation-tested non-vacuous checks), writing-plans (this template).

**Goal:** Add `MultiScaleDeformable1DAttention` — the multi-resolution generalization of the already-shipped `Deformable1DAttention` (Xia et al. 2023). Instead of one sampling grid, maintain `L` feature levels (strided pools of the input); each query predicts `L × K` offsets AND `L × K` attention weights directly from the query, then samples each level and aggregates with a softmax-normalized attention.

**Paper:** Zhu et al., ICLR 2021, "Deformable DETR: Deformable Transformers for End-to-End Object Detection", https://arxiv.org/abs/2010.04159 — Eq. 3 (multi-scale deformable attention). 1D adaptation as in Deformable DETR §A.

**Architecture:** Two classes in `include/nn/layers/attention/multi_scale_deformable_attention.{h,cpp}`:

1. `MultiScaleDeformable1DAttention(d_model, num_heads, num_levels, num_points)` — the
   attention layer itself.
2. `MultiScaleDeformable1DBlock(d_model, num_heads, num_levels, num_points)` —
   pre-LN → multi-scale deformable attn → residual → pre-LN → GELU FFN → residual.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `LayerNorm`, `Dense`, `Deformable1DAttention`
(reused only for its bilinear-sampling math style; the multi-scale version is a separate class).
No new deps.

---

## The math (paper Eq. 3, 1D-adapted; do NOT deviate)

```
Input X ∈ R^{n × d_model}
L levels, K points per level, H heads (head_dim = d_model / H)

Per head h:
  Sampling logits A_raw = X · W_attn                         in R^{n × (L·K)}
  Offsets         Δ_raw = X · W_offsets                     in R^{n × (L·K)}
  Query                 Q = X · W_q                          in R^{n × d_model}
  Value (per level)    V^l = level_projection_l(X · W_v)    in R^{n_l × d_model}

Levels:
  level 0 length = n
  level l length = ⌈n / 2^l⌉   (strided pooling of X · W_v by 2)
  level_l projection is a learnable Dense(d_model, d_model) per level (L projections).

Attention weights (paper's key efficiency trick — softmax over L*K jointly, no Q·K dot product):
  A[t, l, k] = softmax_{l,k}(A_raw[t, l·K + k])

Sampling positions (paper Eq. 3 — signed bounded, per-level scaled):
  ref_l[t] = (t / (n - 1)) · ((n_l - 1) / 1)   for query t, level l
                                # the canonical DETR convention: each query's reference
                                # is its own row index, scaled to that level's length.
  pos_{t,l,k} = clamp( ref_l[t] + s_l · tanh(Δ_raw[t, l·K + k]), 0, n_l - 1 )
  where s_l = max(1, (n_l - 1) / 2) — the offset scale at level l

Bilinear sample V^l at pos_{t,l,k} → V_sampled[t, l, k, d] ∈ R^{head_dim}

Output per (t, d):
  out[t, h, d] = Σ_l Σ_k A[t, l, k] · V_sampled[t, l, k, h*head_dim + d]

Concat heads → (n, d_model) → W_o → (n, d_model)

Backward (hand-derived — verify each step):
  d_V_sampled[t, l, k, d] = A[t, l, k] · d_concat[t, h*head_dim + d]
  d_A[t, l, k]            = Σ_d d_concat[t, h*head_dim + d] · V_sampled[t, l, k, d]
  d_A_raw[t, l, k]        = softmax_backward(A, d_A)[t, l, k]
  d_W_attn                = X^T · d_A_raw                  ; d_X += d_A_raw · W_attn^T
  d_pos[t, l, k]          = Σ_d (V^l[j, d] - V^l[i, d]) · d_V_sampled[t, l, k, d]
  d_pos *= (1 - clamp_saturated_mask)                       # no gradient past clamp
  d_Δ_raw[t, l, k]        = d_pos · s_l · (1 - tanh²(Δ_raw))
  d_W_offsets             = X^T · d_Δ_raw                   ; d_X += d_Δ_raw · W_offsets^T
  d_V^l                   = bilinear_backward(V^l, d_V_sampled, pos)   per (t, l, k)
  d_X (via V path)        = d_V_l · W_v_l^T                   per level, plus the
                                                                   strided-pool backward
                                                                   that "scatters" d_V^l
                                                                   back to the n rows it
                                                                   came from.
```

**Distinct from single-scale `Deformable1DAttention`**:
- Multi-level pooling/scatter (each level has its own K, V projection + length)
- No Q·K dot-product attention — `A` is predicted directly from X (the paper's efficiency move)
- `reference_points` is NOT a learnable parameter; it's the query-row-index, level-scaled
  (per DETR §3.2, the canonical convention)

---

## Task breakdown

### Task 1: Header — class declarations

**Files:** create `include/nn/layers/attention/multi_scale_deformable_attention.h`

**Contents:**
- License + paper header comment (40 lines math)
- `class MultiScaleDeformable1DAttention : public Layer` with:
  - `W_q: Dense(d_model, d_model)`, `W_attn: Dense(d_model, H*L*K)`, `W_offsets: Dense(d_model, H*L*K)`
  - `W_o: Dense(d_model, d_model)`, plus `std::vector<Dense> level_v_projs` (L of them, each `(d_model, d_model)`)
  - caches: `last_input_`, `last_q_`, `last_offsets_`, `last_attn_`, `last_positions_`, `last_V_sampled_`, `last_levels_` (the pooled X·W_v per level — needed for the bilinear backward), `last_concat_`
  - forward/backward/update_weights/zero_grad/parameters/gradients/name
  - accessors: `d_model()`, `num_heads()`, `num_levels()`, `num_points()`, `head_dim()`
- `class MultiScaleDeformable1DBlock : public Layer` (pre-LN attn + residual + pre-LN GELU FFN + residual)

### Task 2: Constructor + accessors

```cpp
MultiScaleDeformable1DAttention(size_t d_model, size_t num_heads,
                               size_t num_levels, size_t num_points);
```

Validate: d_model>0, num_heads>0, num_levels>0, num_points>0; d_model % num_heads == 0.
Init `level_v_projs_` to L `Dense(d_model, d_model)` instances (no biases for V projections, matching the DETR convention; but repo's `Dense` constructor adds bias by default — leave bias on, the test verifies it stays small).

### Task 3: Forward — sampling positions

```cpp
Tensor forward(const Tensor& input);
```

Steps:
1. Validate shape `(N, d_model)`, N>0.
2. Cache `last_input_`. Compute `Q = X·W_q + b_q` (L×K rows for Q? no — just H heads for Q, then split).
3. Per level `l`, compute `V^l = level_v_projs_[l].forward(X)` → (n_l, d_model). Pool by striding: `n_l = ⌈n / 2^l⌉`, take rows `[0, 2^l, 4^l, ...]`.
4. Compute `A_raw = X·W_attn + b_attn` → (n, H·L·K). Compute `Δ_raw = X·W_offsets + b_offsets` → (n, H·L·K).
5. Per (t, h, l, k): `pos = ref_l[t] + s_l · tanh(Δ_raw[t, h·L·K + l·K + k])`, clamp to `[0, n_l-1]`.
   `ref_l[t] = (double)t * (n_l - 1) / max(1, n - 1)`.
6. Bilinear-sample `V^l` at `pos` per head h slice → `V_sampled[t, h, l, k, d]`.
7. Softmax `A_raw` over the `L·K` axis for each (t, h) to get `A`.
8. Compute head output: `out[t, h, d] = Σ_{l,k} A[t, h, l, k] · V_sampled[t, h, l, k, d]`.
9. Concat heads → `last_concat_` (n, d_model). Apply `W_o`. Return.

**Verify RED:** write `test_multi_scale_deformable_attention.cpp` test 1 (constructor), test 2 (forward shape + finite + nonzero).

### Task 4: Bilinear sampling correctness (signature test)

**Write test 3:** `num_levels=1` reduces to the multi-scale-deformable with single-level pooling,
which is NOT the single-scale `Deformable1DAttention` (because the single-scale version uses
Q·K dot-product attention). Verify: with `num_levels=1, num_points=1` and all W_offsets=0,
`reference_points` = identity (`ref[t] = t * (n-1)/(n-1) = t`), the output equals
`softmax(A_raw) · V_sampled` where V_sampled is `V[t, :]` (no interpolation).

This is the "no-clamp, no-interpolation" reduction test.

### Task 5: Backward — softmax + bilinear + clamp mask

```cpp
Tensor backward(const Tensor& grad_output, double lr);
```

Steps:
1. Backward through `W_o` (mirror `Deformable1DAttention::backward`).
2. Per (t, h): d_V_sampled = A · d_concat (broadcasted), d_A = Σ_d d_concat · V_sampled.
3. Softmax backward (over L·K axis): `d_A_raw[t, h, l, k] = A · (d_A − Σ A·d_A)`.
4. Backward through `W_attn`: d_W_attn += X^T · d_A_raw; d_X += d_A_raw · W_attn^T.
5. Bilinear backward: for each (t, h, l, k):
   - `d_pos[t,h,l,k] = Σ_d (V^l[j, d] − V^l[i, d]) · d_V_sampled[t, h, l, k, d]` (where j,i are the floor/ceil of pos at level l).
   - Mask d_pos = 0 if `pos` was on the clamp boundary.
6. Backward through offsets: `d_Δ_raw = d_pos · s_l · (1 − tanh²(Δ_raw))`. Then `d_W_offsets += X^T · d_Δ_raw; d_X += d_Δ_raw · W_offsets^T`.
7. Backward through V^l sampling: for each (t, h, l, k), accumulate `d_V^l[i, d] += β·d_V_sampled`, `d_V^l[j, d] += α·d_V_sampled`. (No pos dependency in V path since bilinear weights are pos-only.)
8. Backward through level_v_projs: for each level l, `d_X_pooled = bilinear_scatter(d_V^l, n_l, l)` then `d_X += level_v_projs_[l].backward(d_X_pooled, lr)`. The scatter puts each pooled row's gradient back onto the original rows it was pooled FROM (the row `r` of level `l` came from row `r · 2^l` of the input).

**Verify RED:** test 4 — gradient FD for `d_X` (input), `d_W_q`, `d_W_attn`, `d_W_offsets`, `d_W_o`, plus a level-V-projection gradient. All at rel_err ≤ 1e-3 with random non-uniform init (random init is mandatory — uniform init passes vacuously).

### Task 6: Clamp-saturation zero-gradient check (mutation test for the clamp mask)

**Write test 5:** With a query whose offsets are saturated at `pos = n_l - 1` for ALL k,l:
- For THAT query's output, d_pos must be exactly 0 (clamp mask).
- For a neighboring query whose offsets push pos into the interior, d_pos ≠ 0.

This catches a missed clamp mask (which would leak gradient and silently pass FD — same trap as Deformable1DAttention test 10). **Mutation-verify**: reverting the clamp mask must break this test.

### Task 7: Signature — single query's offsets all on level 0

**Write test 6:** With `num_levels=2`, force query 0's offsets to all point into level 0 (small enough that they don't reach into level 1's strided-pool region). Perturbing level-1 features should leave query 0's output bit-exact unchanged.

This catches the bug where the d_V^l scatter is wrong (level-1 gradient leaks into query 0's output).

### Task 8: zero_grad / update_weights / parameters / gradients contract

**Write test 7:** All standard.

### Task 9: end-to-end training step

**Write test 8:** MultiScaleDeformable1DBlock forward shape + input gradient FD + training reduces loss > 5% over 30 SGD steps (small target — this layer has many parameters and may converge slowly with random init).

### Task 10: Register in umbrella header + Makefile

- Add `#include "layers/attention/multi_scale_deformable_attention.h"` to `include/nn/nn.h` after `deformable_attention.h`.
- Add `$(BUILD_DIR)/test_multi_scale_deformable_attention: $(LIB_OBJS) $(BUILD_DIR)/test_multi_scale_deformable_attention.o` rule.
- Add to `tests:` deps line.
- Add to `run_tests` echo: `=== Running Multi-Scale Deformable 1D Attention Tests ===`.

### Task 11: Update EXPANSION_QUEUE.md

Move the entry from `## Ideas` to `## Done` with a one-line summary.

---

## Verification gate

After all tasks:
- `make build/test_multi_scale_deformable_attention -j` compiles clean under `-Wall -Wextra` with no warnings.
- `./build/test_multi_scale_deformable_attention` reports all checks pass.
- Umbrella header compiles standalone: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`.
- `make tests` runs the whole suite, no regressions.

---

## Pitfalls (pre-loaded)

1. **Test-config degeneracy** (systematic-debugging §5c): uniform init makes `Σ W[k, j] = Σ W[j, k]` so a row-vs-column gradient confusion passes vacuously. ALWAYS use random init (Tensor::random(0.3)) for FD tests. The plan above already specifies random non-uniform.

2. **Hand-derived expected value typo** (TDD skill caveat): every "matches FD to 1e-3" assertion is correct, but if we hand-compute any closed-form reference (test 3 in particular), re-derive it on paper BEFORE writing the assertion, and verify the impl's actual output matches our derivation before locking the expected value.

3. **Clamp mask missing** (from Deformable1DAttention's bug history): without the clamp mask, the bilinear backward reports a non-zero gradient for saturated points. FD correctly reports 0 (the forward is locally constant there). Analytical and FD agree at ~0 — test passes vacuously. Test 6 (signature) and Test 5 (clamp saturation) catch this.

4. **Level-pooling scatter error**: each row `r` of level `l` was pooled from row `r·2^l` of the (already-projected) input. The backward must SCATTER d_V^l back onto those exact rows. If the scatter goes to a different row, level-1 gradient leaks into the wrong query's input gradient. Test 7 catches this.

5. **Softmax axis**: softmax is over the joint (l, k) axis for each (t, h), not over t. Get this wrong and the attention weights don't sum to 1 over (l, k). Add a forward test that asserts `Σ_{l,k} A[t, h, l, k] = 1` for every (t, h).

6. **Single-iteration loops**: every backward loop here runs N times (or N·H·L·K times), so mutation testing works without needing a multi-iter config.

7. **No new dependencies**: this layer uses `Dense`, `Layer`, `LayerNorm`, `Tensor` — all already in the repo. Don't introduce a new helper class for the bilinear sampler; inline it (4 lines, identical to Deformable1DAttention's inline version).

---

## Plan length

11 tasks. Each is a single concrete change. Expected session: ~3 hours including FD test tuning.
