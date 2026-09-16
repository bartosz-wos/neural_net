# Focal Modulation Networks (FocalNet) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Focal Modulation Networks (Yang et al., NeurIPS 2022, https://arxiv.org/abs/2203.11945) for the `neural_net` C++ ML repo — a token-mixer that **replaces self-attention** with a stack of depthwise "context" layers + global-average-pool query + per-level GELU-then-sigmoid modulator MLP, fanned back across spatial positions.

**Architecture:** v1 simplification of §3.2 of the paper. For input (B, S, D) with B batch and S tokens:

1. **Hierarchical context (L levels)** — each level l ∈ {0, …, L-1} applies a depthwise-sequential mixing `z_l[b, s, d] = w_l[d] * x_l[b, s, d]`. v1 uses independent per-channel scalars (a trivial "depthwise 1D conv" with kernel_size=1 in the channel-projection direction and a sequence-direction shift implemented as permuted access — we'll use **separate flat weights per level** over the sequence direction for v1 to keep a real sequential mixing). Simpler, fully-testable, gradient-tractable.
2. **Query** — `q[b, d] = (1/S) * sum_s x[b, s, d]` — global-average-pool (GAP) to a single per-batch token.
3. **Modulator MLP per level** — `m_l[b, d] = sigmoid(W2_l · GELU(W1_l · q + b1_l) + b2_l)` where W1_l, W2_l are (mlp_dim × D) Dense matrices (with a small hidden dim, e.g. 2*D). The modulator is then **broadcast across S spatial positions**: `m_l[b, s, d] = m_l[b, d]` for all s.
4. **Aggregate** — `y[b, s, d] = sum_l m_l[b, s, d] * z_l[b, s, d]`.
5. **Output projection** — `out[b, s, d'] = sum_d y[b, s, d] * W_o[d', d] + b_o[d']`.

The block wrapper follows the paper's pre-LN residual structure: `out = x + focal(LN1(x)) + FFN(LN2(x + focal(LN1(x))))`. The model is: `input_proj → num_blocks blocks → final LN → mean-pool over S → classifier`.

**Tech Stack:** C++17, hand-rolled matmul-style loops on `Tensor`, `Dense` for all linear projections, `LayerNorm` for pre-LN, `GELU` activation.

## References

- Yang, Li, Zeng, Lu, Wang, Zhang 2022, "Focal Modulation Networks" (https://arxiv.org/abs/2203.11945), NeurIPS 2022.
- Reference PyTorch implementation: https://github.com/microsoft/FocalNet

## Design Decisions

- **v1 simplicity**: depthwise context = per-channel scalar per level over the chain `x_l = cumulative-input at level l`. Concretely for level 0 it's `z_0 = x * w_0` (elementwise), for level ℓ>0 the input to level ℓ is shifted by 1 position along S. This gives a real sequential context (not just plain GAP-twice) while staying analytically tractable and gradient-friendly. The reference uses depthwise conv with kernel=3, dilated, gated ReLU; ours uses a kernel-2 dilated-on-the-prior-level scheme that's equivalent in compute at small scale.
- **Modulator gating**: the paper uses **sigmoid** in the original Eq. 5. The official codebase applies GELU-then-Linear for the inner activation but ends with **a learned gate** (an additive learnable scalar per level). For v1 we use **sigmoid(W2 · GELU(W1 · q))** which is the standard "gating MLP" reading of §3.2 and gives well-conditioned gradients.
- **Parameter-flattening convention**: input to a layer is flat (B, S*d_model) — matches the repo's `Dense` linear-projection convention and what `MlpMixerBlock` does. The FocalModulation layer reshapes internally to (B, S, d_model) for the S-aware chain.

## Phase 0 — Setup files

### Task 0.1: Create header `include/nn/layers/architectures/focal_modulation.h`

Three classes:
- `FocalModulation(d_model, seq_len, num_levels=2, mlp_dim=0)` — the layer.
- `FocalModulationBlock(d_model, seq_len, num_levels=2, mlp_dim=0, ffn_mult=4)` — pre-LN + FocalModulation + residual + pre-LN + GELU FFN + residual.
- `FocalModulationModel(input_dim, d_model, output_dim, seq_len, num_blocks, num_levels=2, mlp_dim=0, ffn_mult=4)` — full model.

Parameter count (per FocalModulation layer): `(L * d_model)` per-channel scalars + `W1_l (mlp_dim, d_model) + b1_l (1, mlp_dim)` for each l + `W2_l (d_model, mlp_dim) + b2_l (1, d_model)` for each l + `W_o (d_model, d_model) + b_o (1, d_model)`. With L=2 and d_model=8 and mlp_dim=16: 2·8 + 2·(16·8+16 + 8·16+8) + (8·8+8) = 16 + 2·(144+144) + 72 = 16 + 576 + 72 = **664**.

Memory layout: `(B, S*D)` flat input → reshape to (B, S, D) for the focal chain → reshape back to (B, S*D) before returning.

---

## Phase 1 — TDD per class (each task is one full red-green cycle)

### Task 1: FocalModulation constructor + forward + finiteness

**File:** `tests/test_focal_modulation.cpp`, `include/nn/layers/architectures/focal_modulation.{h,cpp}`.

Test that:
- (a) Construction throws on `d_model=0`, `seq_len=0`, `num_levels=0`, `num_levels>seq_len` (degenerate), `mlp_dim=0`.
- (b) Construction with valid args succeeds; accessors return correct values; parameter count matches the formula above (L=2, d=8, mlp=16 → 664).
- (c) Forward shape (B=2, S=4, D=8 → (2, 32) flat) is finite and nonzero.
- (d) Per-token zero-out test: zeroing a single weight_l[d]=0 makes that level's contribution vanish (verified by setting W1_l=W2_l=0 and verifying output = 0 + W_o contribution... actually we'll test that zeroing ALL level scalars gives output = 0 except W_o math).

### Task 2: Forward end-to-end hand-derived reference

For an extremely small case (B=1, S=2, D=2, L=1), with hand-set simple values (e.g., W_o=I, W1_l=ones, W2_l=ones, b's = 0, level scalar w_l=1), we can hand-compute the output and verify it matches to machine precision. This catches layout bugs immediately.

### Task 3: FD input gradient

For random non-uniform init (B=2, S=4, D=8, L=2, mlp=16) verify `(analytical_d_input - fd_d_input) / max(|a|, |n|, 1e-12) < 1e-4`.

### Task 4: FD parameter gradients (representative subset)

- A level scalar `level_scalars_[0][d=3]`: FD vs analytical, rel_err < 1e-5.
- A W1_l, W2_l, W_o, b_o: FD vs analytical, rel_err < 1e-5.

### Task 5: Modulator-broadcast sanity

Setting all `W1_l, W2_l` to identity + bias 0 + using only one level makes the modulator a constant channel-wise gain. Verify the output pattern matches `W_o · (w_0 ⊙ x)` elementwise with the modulator broadcast.

### Task 6: zero_grad clears all gradients; update_weights moves all parameters

After `backward(...) + zero_grad()` all gradient tensors have zero everywhere. After `backward() + update_weights(0.1)`, no parameter equals its pre-update value (because lr·grad is nonzero somewhere).

### Task 7: FocalModulationBlock forward + finiteness

Pre-LN + focal + residual + pre-LN + GELU FFN + residual. Forward at B=2, S=4, D=8 → (2, 32) finite, nonzero. Without FFN → same shape, finite.

### Task 8: FocalModulationBlock FD input grad

Same as Task 3, but for the block (rel_err < 1e-3 — LN+FFN chain).

### Task 9: FocalModulationModel forward shape

B=2, S=4, D_in=3, D=8, D_out=2, num_blocks=2 → forward shape (2, 2). Mean-pool over S is part of the model.

### Task 10: FocalModulationModel end-to-end training

y = sum(input_proj then blocks then pool then classifier) on a simple synthetic regression. 50 SGD steps at lr=0.01 should reduce MSE > 30%.

### Task 11: update_weights moves all parameter groups

For the model, every parameter (input_proj, classifier, per-block LN, focal params, FFN) updates by a different amount after one backward + update_weights.

### Task 12: zero_grad clears across the full model

After zero_grad, all parameter gradients are zero.

---

## Phase 2 — Registration

### Task 13: Register in umbrella header `include/nn/nn.h` and Makefile

Add `#include "layers/architectures/focal_modulation.h"` after `include/nn/layers/architectures/sparse_mixer.h`. Verify standalone compile with `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`. Add build rule `$(BUILD_DIR)/test_focal_modulation: $(LIB_OBJS) $(BUILD_DIR)/test_focal_modulation.o`, add to `tests:` deps, add `=== Running Focal Modulation Tests ===` echo in `run_tests`.

### Task 14: Confirm no regressions

Run a representative test set (e.g. `make run_tests`). Confirm the previously-deferred failures in NOT_FIXED.md remain deferred and not in our way.

### Task 15: Commit and update EXPANSION_QUEUE

`docs(plans)` for the plan, `feat(architectures)` for the implementation, `docs: mark FocalNet as Done` to move the entry. Push to master.
