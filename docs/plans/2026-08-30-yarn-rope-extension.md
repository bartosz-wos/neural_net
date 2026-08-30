# YaRN (Yet another RoPE extensioN) Implementation Plan

**Goal:** Add `YaRNRoPE` — Peng, Quesnelle, Fan, Shippole 2023,
https://arxiv.org/abs/2309.00071. Long-context extension of rotary
position embeddings via per-dimension NTK-aware scaling plus an optional
attention temperature term.

**Why this matters:** Vanilla RoPE was trained for a fixed max context
length and degrades on longer sequences because the angle
`θ_i = base^(-2i/d)` is independent of position. YaRN extends RoPE to
arbitrary `scale = target_len / original_len` by:

1. **NTK-aware per-dim frequency scaling**: for each rotation pair index
   `i ∈ [0, d/2)`, replace `θ_i` with `θ_i / freq_scale_by_dim[i]` where
   `freq_scale_by_dim[i] = 1 / (α · (i - i_min) / (i_max - i_min) + 1)`.
   High-frequency (small-i) dims are barely affected; low-frequency
   (large-i) dims are scaled down, which stretches their effective
   wavelength without extrapolation outside the trained range. Default
   `α = 0.1` (the "fast beta" ramp from §3.2).

2. **Attention temperature `sqrt(1/t)`** — multiplies `Q·K^T` scores by
   `sqrt(1/t)` where `t = 1 - ramp · (1 - cur_step/total_steps)`. At
   training step 0, `t = 1 - ramp` (cold); after the ramp period, `t = 1`
   (full temperature restore). Default `ramp = 32` tokens (paper §3.3).

YaRN produces **bit-exact equivalence to vanilla RoPE** at `scale = 1`
(no scaling, no temperature), and **gradual NTK-style extension** for
`scale > 1`.

**Architecture (drop-in on top of `RoPE` / `RoPEWithV`):**

```
For position pos in [0, seq_len), rotation pair index i in [0, d/2):
  angle = pos · theta_i_scaled
where
  theta_i_scaled = base^(-2i/d) / freq_scale_by_dim[i]
and
  freq_scale_by_dim[i] = 1 / (α · (i - 0) / ((d/2) - 1) + 1)
```

The forward rotation, backward transpose, and gradient accumulation are
identical to `RoPE` — only the **cache precomputation** differs.

**Files:**
- Create `include/nn/layers/attention/yarn_rope.{h,cpp}`
- Create `tests/test_yarn_rope.cpp`
- Modify `include/nn/nn.h` (umbrella include), `Makefile` (build rule
  + `tests:` deps line + `run_tests:` echo)

**Tech Stack:** C++17, repo `Tensor`, `Layer`. No external deps.

---

## Tasks (TDD, one behavior per cycle)

1. Write `tests/test_yarn_rope.cpp` with constructor-validation tests.
   Run → RED (no header).
2. Implement `YaRNRoPE` constructor + validation + scale=1 equivalence
   to vanilla `RoPE`. Run → GREEN on equivalence tests.
3. Add **per-dim freq scaling invariants test**: `freq_scale_by_dim[i]`
   monotonically increases from `1 / (α + 1)` to `1`, the
   `scale = 1` config leaves all caches equal to vanilla `RoPE`, the
   `scale > 1` config produces cos/sin caches that differ from vanilla
   by a measurable amount.
4. Add **attention temperature test**: at `ramp_factor = 0`, `t = 1`,
   temperature `sqrt(1/t) = 1` (no scaling); at `ramp_factor = 1`,
   `t = 0`, `sqrt(1/t) = inf` (theoretical; clamp at large but
   finite). Verify the `temperature()` accessor returns the documented
   value.
5. Add **input gradient FD check** vs analytical — must be at machine
   precision (the rotation math is identical to RoPE; only the cache
   differs).
6. Add **mutation tests**: zeroing the freq_scale_by_dim[i] factor
   changes the output; varying `alpha` from 0.1 → 0 changes the output
   measurably.
7. Add **forward shape** test: `(batch, seq*dim) → (batch, seq*dim)`
   for `forward(q, k)` and `backward_qkv(q_grad, k_grad, v_grad)`
   returns the same shape.
8. Add **end-to-end YaRNRoPE on a synthetic position-encoding task**:
   construct Q/K such that the Q·K^T score depends on relative
   position, run forward + MSE loss against a known target that
   requires long-context; with `scale=2, alpha=0.1`, the loss must
   decrease over 50 SGD steps.
9. Wire umbrella + Makefile.

---

## Implementation Notes

The `YaRNRoPE` class mirrors `RoPEWithV` in layout (q/k/v rotation,
`backward_qkv` for per-tensor gradients) but adds three new public
parameters: `scale`, `alpha`, `ramp_factor`. The freq_scale_by_dim[i]
table is precomputed once in `precompute_theta_freqs(seq_len)` and
cached internally as a `Tensor` of shape `(dim/2,)` for test access.

The temperature is returned via `attention_temperature()` accessor
(documentation: "multiply attention scores by this value to apply
YaRN's temperature correction"). It does NOT modify the rotation
itself — that would corrupt the relative-position property of
Q·K^T — it lives outside the RoPE layer and is applied by the
attention compute path. For tests we just verify the formula.

**Backward correctness:** Because the forward rotation is identical
to `RoPE` (just with different cos/sin cache values), the analytical
backward is identical to `RoPE`'s. The gradient check is essentially
a regression test confirming the math was transcribed correctly
rather than a fresh derivation.

**Bug guard during TDD:** YaRN's freq_scale_by_dim[i] is per-pair
index (0 .. d/2-1). It's NOT per-dim (0 .. d-1). The cos/sin cache
mirrors RoPE's pairing convention: cache[*, i] == cache[*, i+d/2] for
each i. Make sure to copy into both paired slots when filling the
cache, identical to RoPE's `precompute_theta_freqs`.
