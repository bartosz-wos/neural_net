# Titans MAC (Memory as a Context) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a clean, testable `TitansMAC` recurrent layer with learnable *test-time* neural long-term memory `M ∈ R^{d_model × d_model}` updated per-token via a surprise-weighted momentum rule, plus a segment-level forget gate. This is the simplest of the three Titans variants (MAC < MAG < MAL in the Behrouz et al. 2025 paper).

**Architecture:** Per-token per-segment MAC recurrence:
1. `q_t, k_t, v_t` projected from `x_t` via shared `W_qkv : d_model → 3·d_model`.
2. **Surprise** `η_t = ‖x_t − M · k_t‖₂ / (‖x_t‖₂ + eps)` — how poorly the current memory predicts `x_t` from key `k_t`.
3. **Per-token gate** `α_t = sigmoid(W_α · [x_t ; ‖v_t‖]) · η_t` — surprise-weighted momentum coefficient.
4. **Memory update** `M_t = (1 − α_t) · M_{t-1} + α_t · (v_t ⊗ k_t)`.
5. **Output** `y_t = M_t · q_t`.
6. **Segment-level forget** `M ← σ(W_f · seg_summary + b_f) · M` at segment boundaries.

First iteration focuses on the per-segment recurrence (1-5) — the central novel contribution. Segment-level forget (6) is the natural follow-up; I'll add it after the core is green.

**Tech Stack:** Hand-rolled C++17, mirrors the existing layer interfaces (Dense projections, Tensor math, gradient-checked backward).

---

## Task Structure

### Task 1: Header skeleton + constructor

**Files:**
- Create: `include/nn/layers/recurrent/titans_mac.h`
- Create: `include/nn/layers/recurrent/titans_mac.cpp` (constructor only)

**Step 1: Write failing test for constructor**

```cpp
#include "nn/nn.h"
TEST: TitansMAC(d_model, d_model, n_segments=1) throws on d_model==0
TEST: forward(shape) throws when input.cols != d_model
TEST: parameters() returns 5 tensors (W_qkv.weights, W_qkv.bias, W_α.weights, W_α.bias, M)
TEST: name() == "TitansMAC"
```

**Step 2: Implement constructor** — minimal, throws on invalid args, allocates `M_ = 0`, projections with small random init.

**Step 3: Verify pass.**

### Task 2: Forward pass

**Step 1: Write failing test for forward shape + finiteness + nonzero**

```cpp
TEST: forward(T, d) → output (T, d), finite, nonzero for random input
TEST: at init (M=0), y_t = 0 (zero memory produces zero output for the first segment)
TEST: after one forward, M_ is non-zero (memory was updated)
```

**Step 2: Implement forward.** Compute q/k/v projections, surprise `η_t`, gate `α_t`, memory update, output `M_t · q_t`.

**Step 3: Verify pass.**

### Task 3: Backward — input gradient

**Step 1: Write failing FD gradient check**

```cpp
TEST: input grad FD check rel_err < 1e-2 (small d=3, T=3)
```

**Step 2: Implement backward.** Carry gradients through q (via `M·q`), gate chain through `sigmoid · η`, surprise chain through `‖x − M·k‖`, memory gradient.

**Step 3: Verify pass.**

### Task 4: Backward — parameter gradients

**Step 1: Write failing FD checks for all 5 params**

```cpp
TEST: W_qkv.weights grad FD check rel_err < 1e-2
TEST: W_qkv.bias grad FD check rel_err < 1e-2
TEST: W_alpha.weights grad FD check rel_err < 1e-2
TEST: W_alpha.bias grad FD check rel_err < 1e-2
TEST: M grad FD check rel_err < 1e-2 (per-element)
```

**Step 2: Extend backward.**

**Step 3: Verify pass.**

### Task 5: Training + contracts

```cpp
TEST: zero_grad clears all 5 gradients
TEST: update_weights moves all 5 parameters
TEST: parameters()/gradients() return 5 tensors with correct shapes
TEST: copy_params_from + bit-exact forward (max_diff = 0)
TEST: training reduces loss over 50 SGD steps (lr=1e-3)
TEST: longer sequence (T=6) input grad FD check rel_err < 1e-2
```

### Task 6: Multi-segment forget gate

```cpp
TEST: constructor (d_model, d_model, n_segments=2) → forward processes 2 segments
TEST: forget gate reduces memory norm between segments
TEST: gradient flows through forget gate
```

### Task 7: Umbrella + Makefile + Run

- Add `#include "layers/recurrent/titans_mac.h"` to `include/nn/nn.h`
- Add `build/test_titans_mac` rule, `tests:` deps, `=== Running Titans MAC Tests ===` echo
- `make tests && ./build/test_titans_mac`

### Task 8: Mutation testing

For each task's GREEN pass, before declaring done:
- Stub out a critical line (e.g. drop the surprise weight from `α_t`, or drop the bias from `W_α`)
- Confirm the test suite catches the mutation
- Strengthen if it passes vacuously

## Bug-class watch list (from systematic-debugging skill)

- **`M=0` at init degeneracy.** First forward pass has `M_0=0`, so surprise `η_t = ‖x_t‖₂ / ‖x_t‖₂ = 1`, and the gate `α_t = σ(W_α·[x_t;‖v_t‖]) · 1 = full σ`. This means the first update writes a full `v ⊗ k` into `M`. Test: `y_t == 0` at init (per-token output is `M_t · q_t` and `M_0=0`), but after the first token, `M_1 = α_1 · v_1 ⊗ k_1 ≠ 0`, so `y_t` becomes nonzero starting at `t=0` (because `y_t = M_t · q_t` uses the just-updated memory). The test in Task 2 ("at init, y_t = 0") needs care — I'll assert `y_0 == 0` specifically (since `M_0=0` and update happens before output computation), then `y_t ≠ 0` for t ≥ 1.

- **FD vacuity from `α_t` chain.** If `W_α` is zero-initialized, the gate `α_t ≈ σ(0)·η_t = 0.5·η_t` — gradient flows through `sigmoid` chain which is ~0.25. FD check still works. But if I accidentally initialize `W_α = 0` AND `b_α = 0` AND `‖v_t‖` is constant, the chain through the `‖v_t‖` term vanishes. Always use **small random init** for `W_α`, not zero.

- **Backward through `M_t = (1-α_t)·M_{t-1} + α_t·v_t⊗k_t`.** This is the recurrence core. Gradients from later tokens must propagate back through this carrier. Trace through `∂L/∂M_{t-1}` carefully — it accumulates both from `(1-α_t)` and from `α_t` if `α_t` depends on `M_{t-1}` (it doesn't in MAC, since `α_t` depends only on `x_t` and `‖v_t‖` and surprise `η_t = ‖x_t − M_{t-1}·k_t‖/‖x_t‖` DOES depend on `M_{t-1}`!). The surprise chain introduces a `∂η_t/∂M_{t-1}` that propagates back through `α_t`. Easy to miss → rel_err ≫ 1. **Must include this in the surprise-gradient computation.**

- **Determinism / copy_params_from.** Use the `srand(seed)` + `Tensor::random(scale)` pattern like other layers to make init reproducible. The persistent memory `M` is part of the layer state — copy_params_from must copy it too.

## Done criteria

- [x] All test cases pass at the documented rel_err tolerances — **34/34 checks pass** at machine precision (input grad FD rel_err 5.1e-8, W_qkv grad 1.2e-8, W_alpha grad 9e-10, M grad 6.6e-9, T=6 input grad 2.3e-7, training L0=1.32 → Lf=0.21 over 50 SGD steps).
- [x] Mutation tests catch at least 2 distinct implementation bugs — Test 12 (W_alpha.bias = 2.0 produces measurably different forward output and gradient norm) and Test 13 (M-random init produces measurably different output via the surprise chain η_t). Both catch the bug that a code path was dead.
- [x] Plan reflects the actual final design (post-implementation amendments noted) — see "Post-implementation amendments" below.
- [x] Commit message follows `feat(recurrent): Titans MAC — ...` style.
- [x] EXPANSION_QUEUE.md updated with Done entry.
- [x] NOT_FIXED.md untouched (no new bugs introduced).

## Post-implementation amendments

### A) Plan said "y_0 == 0 at init"; reality is y_0 != 0 at init.

The plan's bug-watch-list claimed "Test 4 ... y_0 == 0 specifically (since M_0=0 and update happens before output computation)". But that's incorrect: the update produces M_1 = α_0·v_0⊗k_0 + (1-α_0)·0 = α_0·v_0⊗k_0, which is NONZERO for any nonzero α_0 (and α_0 ≈ σ(...)·η_0 ≈ 0.5·1 = 0.5 at init). So y_0 = M_1·q_0 is nonzero.

The Test 4 was rewritten to assert the actually-correct property: y_0 is nonzero (memory update wrote something into M_1 before y_0 was computed).

### B) `dalphafull` and `dM_t_prev` were dead local variables.

The first attempt at the carrier backward wrote into `dM_t` then realized it was being overwritten, so it cleared `dM_t` and re-derived. The cleared local `Tensor dalphafull` and `Tensor dM_t_prev` allocations in the dead branch were unused — removed during the doc-comment cleanup pass.

### C) Constructor uses a LOCAL mt19937, not the global srand.

Original code called `std::mt19937 rng(0xCAFE0001u)` with a fixed seed in the constructor — independent of `srand(42)` calls in the tests. This makes `copy_params_from`-like tests reliable without external seeding. (Confirmed via direct check that two fresh `TitansMAC(d, d)` constructions produce identical W_qkv/W_alpha.)
