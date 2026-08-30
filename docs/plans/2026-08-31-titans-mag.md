# Titans MAG (Memory as a Gate) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a clean, testable `TitansMAG` recurrent layer — the middle of the three Titans variants (MAC < MAG < MAL, Behrouz et al. 2025, https://arxiv.org/abs/2501.00663). Unlike MAC where the memory outputs `y_t = M_{t-1} · q_t`, in MAG the memory is **gated by the input**: `y_t = (M_{t-1} · x_t) ⊙ x_t` (element-wise product). The output is a *gating* of the input by what the memory produces from x_t.

**Architecture:** Same update rule as `TitansMAC` (the test-time surprise-weighted momentum recurrence on M) but with a different output equation:
1. `q_t, k_t, v_t` projected from `x_t` via shared `W_qkv : d_model → 3·d_model`.
2. **Surprise** `η_t = ‖x_t − M_{t-1} · k_t‖₂ / (‖x_t‖₂ + eps)`.
3. **Per-token gate** `α_t = sigmoid(W_α · [x_t ; ‖v_t‖]) · η_t`.
4. **Memory update** `M_t = (1 − α_t) · M_{t-1} + α_t · (v_t ⊗ k_t)`.
5. **Output (MAG)** `y_t = (M_t · x_t) ⊙ x_t` — element-wise product of memory's transformation of x with x itself.

This is a natural extension: identical update recurrence to MAC (so we can share the surprise/gate/update logic structure), but the output equation adds a chain `x → y_t` through both the (M·x) gate path AND the ⊙x path (which means `dL/dx_t` has an extra `(M·x)` factor compared to MAC).

**Tech Stack:** Hand-rolled C++17, mirrors the existing `TitansMAC` interface (Dense projections, Tensor math, gradient-checked backward).

---

## Task Structure

### Task 1: Header skeleton + constructor

**Files:**
- Create: `include/nn/layers/recurrent/titans_mag.h`
- Create: `include/nn/layers/recurrent/titans_mag.cpp` (constructor only)

**Step 1: Write failing test for constructor**

```cpp
TEST: TitansMAG(d_model, 0, 0) constructs
TEST: TitansMAG(d_model=0) throws
TEST: TitansMAG(d_model, d_inner=2) throws (must equal d_model in v1)
TEST: parameters() returns 5 tensors (W_qkv.weights, W_qkv.bias, W_alpha.weights, W_alpha.bias, M)
TEST: name() == "TitansMAG"
TEST: d_model() / d_inner() / seg_len() accessors return the construction values
```

**Step 2: Implement constructor** — minimal, throws on invalid args, allocates `M_ = 0`, projections with small random init (same mt19937(0xCAFE0002u) seed-distinct from MAC for independence).

**Step 3: Verify pass.**

### Task 2: Forward pass

**Step 1: Write failing test for forward shape + finiteness + nonzero**

```cpp
TEST: forward(T=3, d=4) → output (3, 4), finite, nonzero for random input
TEST: forward shape preserved for T=6, d=4
TEST: after one forward, M_ is non-zero (memory was updated)
TEST: forward produces identical output with copied params (max_diff = 0)
```

**Step 2: Implement forward.** Mirror MAC's structure for steps 1-4 (q/k/v projection, surprise η, gate α, M update), but the **output** step (5) is now `y_t = (M_t · x_t) ⊙ x_t`. Cache `last_mx_ = (M_t · x_t)` (the gate values before ⊙) for backward.

**Step 3: Verify pass.**

### Task 3: Backward — input gradient

**Step 1: Write failing FD gradient check**

```cpp
TEST: input grad FD check rel_err < 1e-4 (small d=3, T=3)
```

The MAG-specific chain through `y_t = (M_t · x_t) ⊙ x_t` adds:
- `dL/dx_t[j] += (M_t·x_t)[j] · dy_t[j] + (M_t^T · (dy_t ⊙ x_t))[j] · 1` (the second factor is 1 because �x_t is the identity on x_t in the second slot)

**Step 2: Implement backward.** Reuse MAC's surprise/gate/update backward structure, plus the extra MAG-specific chain through (M·x) and ⊙x.

**Step 3: Verify pass.**

### Task 4: Backward — parameter gradients

**Step 1: Write failing FD checks for all 5 params**

```cpp
TEST: W_qkv.weights grad FD check rel_err < 1e-4
TEST: W_qkv.bias grad FD check rel_err < 1e-4
TEST: W_alpha.weights grad FD check rel_err < 1e-4
TEST: W_alpha.bias grad FD check rel_err < 1e-4
TEST: M grad FD check rel_err < 1e-4 (per-element)
```

**Step 2: Extend backward.**

**Step 3: Verify pass.**

### Task 5: Training + contracts

```cpp
TEST: zero_grad clears all 5 gradients
TEST: update_weights moves all 5 parameters
TEST: parameters()/gradients() return 5 tensors with correct shapes
TEST: training reduces loss over 50 SGD steps (lr=1e-3)
TEST: longer sequence (T=6) input grad FD check rel_err < 1e-4
TEST: mutation — perturbing W_alpha.bias changes output measurably
```

### Task 6: MAG-specific property — gating reduces magnitude

```cpp
TEST: with M=0 (memory blank), y_t = (0 ⊙ x_t) = 0 (gating by zero memory → zero output)
TEST: with x_t = 0 (zero input), y_t = (M_t·0) ⊙ 0 = 0 (zero input → zero output)
TEST: y_t sign respects x_t sign in each component (sign of y_t[j] == sign of x_t[j] when (M·x)[j] > 0, sign of y_t[j] == -sign of x_t[j] when (M·x)[j] < 0)
TEST: mutation — zeroing the M update leaves y_t = 0 forever (catches "forgot to update M")
```

### Task 7: Umbrella + Makefile + Run

- Add `#include "layers/recurrent/titans_mag.h"` to `include/nn/nn.h` (after `titans_mac.h`)
- Add `build/test_titans_mag` rule, `tests:` deps, `=== Running Titans MAG Tests ===` echo
- `make tests && ./build/test_titans_mag`

### Task 8: Mutation testing

For each task's GREEN pass, before declaring done:
- Stub out a critical line (e.g. drop the `⊙ x_t` element-wise product — y becomes just M·x; drop the surprise weight from `α_t`; zero out the (1-α) M-update carrier)
- Confirm the test suite catches the mutation
- Strengthen if it passes vacuously

---

## Bug-class watch list (from systematic-debugging skill)

- **`M=0` at init degeneracy** (same as MAC). First forward: `M_0 = 0`, so surprise `η_t = ‖x_t‖/‖x_t‖ = 1`, gate `α_t ≈ 0.5`. After the first update `M_1 ≈ 0.5 · v_1 ⊗ k_1`, so `y_0 = (M_1 · x_0) ⊙ x_0` is nonzero starting at t=0. The "y_t = 0 at init" property **only holds at t=0 BEFORE the update**, but our layer always updates M then outputs, so y_0 ≠ 0 even at init. Test the property `y_t` is finite+nonzero, not `y_t == 0`.

- **`⊙ x_t` chain — easy to miss the indirect path**. `y_t = (M·x) ⊙ x_t` has TWO ways x_t contributes to y_t: (a) the explicit `⊙ x_t` (dL/dx_t includes `dy_t ⊙ x_t` term from the ⊙), and (b) the implicit `M·x` where x appears on both sides (dL/dx_t includes `M^T · (dy_t ⊙ x_t)` term). Forgetting (b) means input grad FD fails on the second column by a factor of `(M·x)_j` ≈ 0.5 — clear fingerprint of the missing path.

- **FD vacuity from uniform M_init**. If M_ is initialized to all zeros (it is), the first token's update is `M_1 = α_0 · v_0 ⊗ k_0`, which has rank 1. The forward `y_0 = (M_1·x_0) ⊙ x_0` lives in a 1-D subspace of x_0. The gradient w.r.t. `M_0` is zero (since `M_0=0` contributes nothing to `y_0`), but the gradient w.r.t. `M_1` is real. After the first token, M is full-rank from many updates. So the FD test on `M` should perturb entries of `M_1` (i.e., post-update memory) which DOES have real gradient. The standard "perturb `M_` initial" FD won't catch M-init bugs because M_ is read-only once updated; the gradient flows to the persistent `M_` only via the M_{t-1} → M_t recurrence carrier. Test the **post-update** perturbation by perturbing `M_` itself (which is read at t=0 as M_{0}) — perturbing M_(i,j) by ε changes y_0 → that's the right test.

- **Determinism / copy_params_from**. Use the `srand(seed)` + `Tensor::random(scale)` pattern. The persistent memory `M` is part of layer state — copy_params_from must copy it. Use a different RNG seed than `TitansMAC` (0xCAFE0002u) so test fixtures that construct both can stay independent.

---

## Done criteria

- [ ] All test cases pass at the documented rel_err tolerances (target 30+ checks, FD rel_err < 1e-4)
- [ ] Mutation tests catch at least 3 distinct implementation bugs (no ⊙x, missing (M·x) chain, missing surprise→M_{t-1} chain)
- [ ] Plan reflects the actual final design (post-implementation amendments noted)
- [ ] Commit message follows `feat(recurrent): Titans MAG — ...` style
- [ ] EXPANSION_QUEUE.md updated with Done entry
- [ ] NOT_FIXED.md untouched (no new bugs introduced)
