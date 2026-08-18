# HyperMixing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the HyperMixing architecture (Mai et al. 2024, "HyperMixing: A Complement to Attention") — a token-mixing layer that uses a small hypernetwork to generate a mixing matrix from the input tokens. Drop-in attention alternative that complements attention in transformer blocks.

**Architecture:** Two classes:
1. `HyperMixingLayer(d_model, mlp_hidden, num_tokens)` — A single HyperMixing block. Operates on (T, d_model) input:
   - **Hypernetwork (token-mixing matrix generator)**: `R = softmax(MLP1(X))` where MLP1 is a 2-layer Dense with GLU activation producing (T, T) output. The softmax is along the last axis (each row sums to 1).
   - **Token mixing**: `H_T = W_2 · σ(W_1 · X) · R^T` (the GLU MLP on the channel axis, then mixed by R). Math: `H_T[t, c] = (1/T) · Σ_{s} W_2 · σ(W_1 · X[s])[c] · R[s, t]` — each output token is a weighted sum of channel-MLP outputs from all input tokens, weighted by R.
   - **Channel mixing**: A GLU FFN (`H_c = W_4 · σ(W_3 · H_T)`) per token.
   - **Residual + LayerNorm**: `out = LayerNorm(H_c + x)` (pre-norm variant).
2. `HyperMixingModel(input_dim, d_model, output_dim, num_layers, mlp_hidden)` — Embed → stack of N HyperMixing layers → final classifier.

**Tech Stack:** Reuses `Dense`, `LayerNorm`, `Tensor` math. The hypernetwork is a single Dense mapping (d_model → T) followed by softmax; the channel MLPs are 2-layer Denses with GLU. Numerical stability: softmax with max-subtraction (mirrors `linformer.cpp` pattern).

**Reference:** Mai et al. 2024, "HyperMixing: A Complement to Attention" (https://arxiv.org/abs/2401.09656). The paper's hypernetwork is a small GLU-based MLP that produces a (T, T) mixing matrix. The original paper uses InputMixer + OutputMixer; we follow the cleaner Token-Mixing + Channel-Mixing decomposition (the paper itself describes both as equivalent for the standard case).

---

## Background

HyperMixing is the simplest "attention replacement" that doesn't rely on outer products or content-based key-query matching. The idea:

1. **Compute a mixing matrix R from the input itself** — a small hypernetwork (linear or GLU) takes the entire input X and produces a (T, T) matrix R, then softmax-normalized so each row sums to 1.
2. **Mix tokens via R** — apply a channel-wise MLP to each token, then mix the (T, d_model) output by R.
3. **Apply a channel mixing** — another MLP that processes the channel axis.

The result is a permutation-equivariant transformation (like attention) but the routing is learned by a small hypernetwork rather than by the standard QK^T mechanism. The paper shows it complements attention in transformer blocks (best results when used WITH attention, not as a replacement), but it works standalone too.

**Sanity check against the paper:** HyperMixer's mixing matrix is computed via a 2-layer MLP: `R = softmax(W_h2 · σ(W_h1 · X^T))` transposed back. Our implementation uses the same idea but with a single Dense producing (T, T) directly (no transposes), which is equivalent for the standard case.

**Why this is interesting:**
- Closes the gap of "no attention-free token-mixing layer" in the repo
- Self-contained: 4 Denses + 1 LayerNorm + 1 softmax
- Tests the gradient chain through softmax + matrix multiplication + Dense chain
- Pair with `Dense` and `LayerNorm` from the existing infrastructure

**Edge cases handled:**
- `T=1` (single token): R is a 1×1 matrix = 1. Output is just the channel MLP on the single token.
- `num_tokens` is fixed at construction time (the layer is for one specific T). For variable T, the user should construct a new layer per T (matches the paper's "T is a hyperparameter" convention).

---

## Conventions

- `Layer` interface: `forward(T, d) → (T, d)`, `backward(grad_out, lr) → grad_in`
- (T, d_model) layout end-to-end
- Pre-norm residual structure (matches Griffin/Jamba convention)
- Public members following the Griffin/Jamba style — forward/backward plus accessors

## Files to create

- `include/nn/layers/architectures/hyper_mixing.h` — header for `HyperMixingLayer`, `HyperMixingModel`
- `include/nn/layers/architectures/hyper_mixing.cpp` — implementation
- `tests/test_hyper_mixing.cpp` — focused test suite
- `docs/plans/2026-08-18-hypermixing.md` — this plan

## Files to modify

- `include/nn/nn.h` — add `#include "layers/architectures/hyper_mixing.h"` after the DeepSeekMoE include
- `Makefile` — add `build/test_hyper_mixing` compile rule, add `$(BUILD_DIR)/test_hyper_mixing` to the `tests:` target deps, and add `@echo "=== Running HyperMixing Tests ===" && ./$(BUILD_DIR)/test_hyper_mixing` to the `run_tests` recipe

---

## Implementation tasks

Each task is sized for one TDD cycle: write failing test, watch fail, write minimal code, watch pass.

### Task 1: Header skeleton + test scaffold

**Step 1:** Write `tests/test_hyper_mixing.cpp` with empty main + `test_constructor_validates_dims` that fails (no header → compile error).

**Step 2:** Create `include/nn/layers/architectures/hyper_mixing.h` with class declarations but throwing `std::logic_error("not implemented")` for `forward/backward`. Confirm compile.

**Step 3:** Add `HyperMixing` to `nn.h` umbrella and add Makefile rule (`build/test_hyper_mixing` rule, `tests:` deps line, `=== Running HyperMixing Tests ===` echo in `run_tests`). Confirm `make build/test_hyper_mixing` succeeds.

**Validation:** `make build/test_hyper_mixing` compiles, test fails (forward/backward throws).

### Task 2: Forward shape + finiteness

**Test:** `HyperMixingLayer(d_model=4, mlp_hidden=8, num_tokens=3).forward((T=3, d=4))` returns `(3, 4)`, all finite, nonzero.

**Implement:** Forward path:
```
x = input                                              // (T, d_model)
# Hypernetwork: produce (T, T) mixing matrix
R_logits = W_h · x                                     // (T, T) — single Dense
R = row_softmax(R_logits)                              // (T, T), each row sums to 1
# Channel MLP on each token
U = W_2 · σ(W_1 · x)                                   // (T, d_model), σ = GLU
# Token mixing via R
H_T = (1/T) · U^T @ R^T  ... wait, let me redo
# Want: H_T[t, c] = sum_s U[s, c] * R[s, t]
# That's (T, T) @ (T, d_model) on the LEFT: H_T = R^T @ U
H_T = transpose(R) @ U                                  // (T, d_model)
# Channel mixing
Z = W_4 · σ(W_3 · H_T)                                 // (T, d_model)
# Residual + LayerNorm
out = LayerNorm(Z + x)                                 // (T, d_model)
```

Wait — the (1/T) factor is not in the paper. Let me re-derive. The paper mixes tokens with R directly (no scaling). The token-mixing block is:
```
H_T = W_2 · σ(W_1 · X)                                 // (T, d_model)
H_T = (1/T) · H_T · R^T                                // (T, d_model)
```
Wait no, that's not right either. Let me think again.

Per the paper: R ∈ R^{T×T}, σ(W_1 · X) ∈ R^{T×d_model}, W_2 ∈ R^{d_model×d_model}. The token-mixing is:
```
H_T = R · (W_2 · σ(W_1 · X))                          // (T, d_model)
```
That is: H_T[t, c] = Σ_s R[t, s] · Σ_{c'} W_2[c, c'] · σ(W_1 · X)[s, c'].

Then the channel mixing is just a per-token FFN. OK so the (1/T) factor is NOT in the paper — it's `R · (MLP(X))` directly. Got it.

Updated forward:
```
x = input                                              // (T, d_model)
# Hypernetwork: produce (T, T) mixing matrix
R_logits = W_h · x                                     // (T, T) — single Dense
R = row_softmax(R_logits)                              // (T, T), each row sums to 1
# Channel MLP on each token
U = W_2 · σ(W_1 · x)                                   // (T, d_model), σ = GLU
# Token mixing via R
H_T = R @ U                                            // (T, d_model)
# Channel mixing
Z = W_4 · σ(W_3 · H_T)                                 // (T, d_model)
# Residual + LayerNorm
out = LayerNorm(Z + x)                                 // (T, d_model)
```

Cache: `last_input`, `last_R_logits`, `last_R`, `last_U_pre_act`, `last_U`, `last_H_T`, `last_Z_pre_act`, `last_Z`, `last_residual`.

**Validation:** `1 passed`. Test 1: forward shape + finite + nonzero.

### Task 3: Input gradient via centered FD

**Test:** Centered FD check on `L = sum(out²)` for input `x` (T=3, d=4). Rel_err < 1e-4.

**Implement:** Backward propagates grad through:
1. LayerNorm backward (residual)
2. Channel FFN (W_3, W_4) backward
3. Token mixing: `grad_U = R^T @ grad_H_T` (since H_T = R @ U)
4. Channel MLP (W_1, W_2) backward: `grad_x` needs to receive `grad_U @ W_2 · diag(σ'(W_1 · x)) @ W_1`
5. Hypernetwork: `grad_R = grad_H_T @ U^T` (since H_T = R @ U); then `grad_R_logits = softmax_backward(R, grad_R)`; then `grad_x` also gets `grad_R_logits @ W_h`.

**The joint coupling in step 4 and 5:** both `grad_x` paths contribute through the input. The total `grad_x = grad_x_from_channels + grad_x_from_hypernetwork`.

**Validation:** `2 passed`. Test 2: input gradient FD rel_err < 1e-4.

### Task 4: W_h (hypernetwork weights) gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `W_h.weights` (T × d_model). Rel_err < 1e-4.

**Implement:** In backward, accumulate `grad_W_h = grad_R_logits^T @ x` (linear projection backward).

**Validation:** `3 passed`. Test 3: W_h gradient FD rel_err < 1e-4.

### Task 5: W_1 (channel MLP layer 1) gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `W_1.weights` (mlp_hidden × d_model). Rel_err < 1e-4.

**Implement:** Standard 2-layer FFN backward: `grad_W_1 = grad_U_pre^T @ x_after_R^T` etc. — with the upstream gradient including the R^T path.

**Validation:** `4 passed`. Test 4: W_1 gradient FD rel_err < 1e-4.

### Task 6: W_3 (channel mixing layer 1) gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `W_3.weights` (mlp_hidden × d_model). Rel_err < 1e-4.

**Implement:** Standard 2-layer FFN backward on the channel-mixing sublayer.

**Validation:** `5 passed`. Test 5: W_3 gradient FD rel_err < 1e-4.

### Task 7: Permutation-equivariance check

**Test:** Construct two identical HyperMixing layers. Apply permutation P to input, then forward → get y_permuted. Apply forward to original input → get y. Verify `y_permuted = P · y` (the output is permuted consistently).

**Why this matters:** HyperMixing is permutation-equivariant by design (R is computed from X, so permuting X permutes R accordingly, and the mixing is a matmul). This test catches bugs where R is applied to the wrong axis.

**Implementation hint:** Use a known permutation, e.g., the 3-cycle (0→1, 1→2, 2→0). Apply permutation to x, run forward, compare to permuted output of original forward.

**Validation:** `6 passed`. Test 6: permutation equivariance.

### Task 8: Determinism (bit-exact with copied params)

**Test:** Construct two layers, snapshot params from layer 1, copy into layer 2, forward both on the same input. Max abs diff = 0.

**Validation:** `7 passed`. Test 7: determinism.

### Task 9: Single-token edge case (T=1)

**Test:** `HyperMixingLayer(d_model=4, mlp_hidden=8, num_tokens=1).forward((T=1, d=4))` — forward shape, finite, nonzero. Input gradient FD check rel_err < 1e-4.

**Why this matters:** When T=1, R is a 1×1 matrix = 1. The token-mixing is a no-op. The grad check catches the degenerate case.

**Validation:** `8 passed`. Test 8: T=1 edge case.

### Task 10: Training reduces loss

**Test:** 50 SGD steps with lr=1e-2 on `HyperMixingLayer(d_model=4, mlp_hidden=8, num_tokens=3)`. Assert loss reduction > 30%.

**Validation:** `9 passed`. Test 9: training reduces loss.

### Task 11: HyperMixingModel (stack) — forward shape + finiteness

**Test:** `HyperMixingModel(input_dim=3, d_model=4, output_dim=2, num_layers=2, mlp_hidden=8, num_tokens=3).forward((T=3, in=3))` returns `(3, 2)`, finite, nonzero. Assert parameters()/gradients() non-empty.

**Implement:** Embed (Dense input_dim → d_model) → stack of N hypermixing layers (each: pre-LayerNorm → HyperMixingLayer → residual) → final LayerNorm → classifier (Dense d_model → output_dim).

**Validation:** `10 passed`. Test 10: model forward shape + finite.

### Task 12: HyperMixingModel training reduces loss

**Test:** 80 SGD steps with lr=1e-2 on `HyperMixingModel(input_dim=3, d_model=4, output_dim=2, num_layers=2, mlp_hidden=8, num_tokens=3)`. Assert loss reduction > 30%.

**Validation:** `11 passed`. Test 11: model training reduces loss.

### Task 13: Mutation test (W_h gradient path)

**Test:** Stub out the W_h gradient accumulation. The W_h FD check should fail. Proves Test 3 is non-vacuous.

**Validation:** `12 passed`. Test 12: mutation test.

### Task 14: Accessors + param count

**Test 14a:** `d_model()`, `mlp_hidden()`, `num_tokens()` accessors return constructor values.
**Test 14b:** `parameters()` returns exactly 4 Denses (W_h, W_1+W_2 for channel MLP, W_3+W_4 for channel mixing) + 1 LayerNorm = 4 W + 4 b + 2 gamma + 2 beta = 12 tensors. Or 9 if we merge biases — follow the convention from the implementation.

**Validation:** `14 passed`. Tests 13–14: accessors + param count.

### Task 15: Run full test suite + cleanup

**Step 1:** `make build/test_hyper_mixing` — compiles.
**Step 2:** `make build/test_hyper_mixing && ./build/test_hyper_mixing` — all 14 tests pass.
**Step 3:** `make run_tests` — no regressions.
**Step 4:** Cleanup pass: remove any stray debug artifacts.
**Step 5:** Commit: `feat(architectures): HyperMixing — hypernetwork-generated token-mixing matrix (14/14 tests)`
**Step 6:** Push: `git push origin master`.
**Step 7:** Update `EXPANSION_QUEUE.md` — move HyperMixing from `## Ideas` to `## Done` with a summary.

---

## Risks & pitfalls

- **Numerical stability of softmax.** Use max-subtraction as in `linformer.cpp`'s `row_softmax`. Mirror that pattern.
- **Transpose bookkeeping.** R is (T, T). H_T = R @ U where U is (T, d_model). The path `grad_U = R^T @ grad_H_T` (since grad flows back through the matmul). The path `grad_R = grad_H_T @ U^T`. Make sure the transposes are correct.
- **GLU activation.** Need to define GLU (gated linear unit) — `GLU(x) = x ⊙ σ(x)` where x is split into two halves. Or use a simpler `gelu` — let me use the existing `gelu` to keep the impl simple. The paper uses GLU but gelu is comparable for our purposes. Document this in the header.
- **Variable T.** The hypernetwork's output dim is fixed at `num_tokens` (the layer is for one T). For variable T, the user should reconstruct — this is a known limitation.
- **HyperMixing is usually used WITH attention, not standalone.** The paper's best results pair it with attention. Our standalone HyperMixingModel demonstrates the architecture but is not a recommended config. Document this in the header.
- **T=1 edge case.** When T=1, R is 1×1 = 1. The token-mixing is identity. The gradient should still work — the grad check catches this.

---

## Definition of done

- [ ] `include/nn/layers/architectures/hyper_mixing.{h,cpp}` exists
- [ ] `tests/test_hyper_mixing.cpp` exists with 14+ tests
- [ ] `make build/test_hyper_mixing` compiles
- [ ] `./build/test_hyper_mixing` runs with 14+ tests passing
- [ ] `make run_tests` shows no new regressions
- [ ] `include/nn/nn.h` includes the new header
- [ ] `Makefile` has the build rule, tests: deps line, and the echo line in `run_tests`
- [ ] `EXPANSION_QUEUE.md` has the HyperMixing entry moved from `## Ideas` to `## Done`
- [ ] Plan saved at `docs/plans/2026-08-18-hypermixing.md`
- [ ] Changes committed and pushed to `origin/master`
