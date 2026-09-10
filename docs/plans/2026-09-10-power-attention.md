# Power Attention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).
> **Skills loaded:** systematic-debugging (numerical-gradient trap awareness), test-driven-development (mutation-tested non-vacuous checks), writing-plans (this template).

**Goal:** Add `PowerAttention` — a learnable per-head power-parameter generalization of softmax attention. Each head learns its own temperature (via `p_log → p = softplus(p_log) + p_eps`) so the model can mix sharpened (closer to argmax) and softened (closer to uniform) attention patterns across heads.

**Paper:** Poli et al., 2024, "Power Attention: Transformers Need Less Compute Than You Think" (https://arxiv.org/abs/2403.14248).

**Architecture:** Two classes in `include/nn/layers/attention/power_attention.{h,cpp}`:
1. `PowerAttention(d_model, num_heads=1, p_eps=0.1)` — the attention layer.
2. `PowerBlock(d_model, num_heads=1, ffn_dim=0, p_eps=0.1)` — pre-LN → power attn → residual → pre-LN → GELU FFN → residual.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `LayerNorm`, `Dense`. No new deps.

---

## The math (paper §, paraphrased)

```
Per head h, per token j:
  z[i,j]   = (q_j · k_i) / sqrt(d_h)
  p[h]     = softplus(p_log[h]) + p_eps        (per-head learnable power; init p_log=0 → p ≈ 1.24)
  s[i,j]   = p · z[i,j]                         (raised logits)
  A[i,j]   = softmax_i(s[i,j])                  (numerically stable via max-shift)
  out_h_j  = Σ_i A[i,j] · v_i
```

**Limit intuition** (with `T = 1/p` as the softmax temperature):
- `p → ∞ (T → 0)`: A → argmax   (sharp, sparse)
- `p = 1` (T = 1): standard softmax
- `p → 0 (T → ∞)`: A → uniform  (diffuse)

The paper shows trained models learn *heterogeneous* per-head `p`, giving the model a mix of sharp and diffuse heads for free.

**Backward derivation** (hand-derived; the formulae in plain form):

```
Forward caches (per head, per query):
  s_k = p · z_k − p · z_max              (shifted, scaled logits)
  A_k = exp(s_k) / Σ_i exp(s_i)

Given dA_k from dO, with sumA_w = Σ_i A_i · dA_i (the A-WEIGHTED sum, NOT raw Σ dA_i):
  dS_k = A_k · (dA_k − sumA_w)         (standard softmax-backward)
  dZ_k = p · dS_k                       (chain through p)
  dQ_j += inv_temp · Σ_i dZ[i,j] · K[i]
  dK_i += inv_temp · Σ_j dZ[i,j] · Q[j]
  dp_log[h] += softplus'(p_log[h]) · Σ_ij z[i,j] · dS[i,j]
```

**Critical bug to avoid (caught during TDD)**: the standard softmax-backward sum is `Σ_i A_i · dA_i`, **not** `Σ_i dA_i`. Forgetting the A-weighting produces a `~3-10x` scale error and sign-flips in some entries of dW_q / dW_k. The corrected formulation is what makes every parameter FD land at machine precision.

**Translation-invariance signature** for W_k.bias: a uniform shift in every key's projection is a constant added to every logit in the row of softmax → softmax cancels it → FD and analytical gradients are *both* exactly 0 (within FP noise). W_q.bias does NOT share this property because the shift varies per query (since K[i] is not constant across i).

---

## Tasks completed (TDD)

| # | Task | Test count | Status |
|---|------|-----------|--------|
| 1 | Constructor validation (d_model=0, num_heads=0, non-divisible, p_eps<0 throw; p_eps=0 valid) | 8 | PASS |
| 2 | Forward shape, finiteness, nonzero, cache shape | 5 | PASS |
| 3 | p=1 matches standard softmax (machine precision — proved by setting p_log such that softplus(p_log) = 1 and comparing element-by-element) | 1 | PASS |
| 4 | Sum-to-1 + positivity invariants | 2 | PASS |
| 5 | FD gradients on W_q/W_k/W_v/W_o (weights + biases), p_log, input — all at machine precision | 12 | PASS |
| 5b | Translation-invariance signature for W_k.bias (both FD and analytical are 0) | 2 | PASS |
| 6 | Deeper N=5 input FD | 1 | PASS |
| 7 | Multi-head = two independent single-head recompute | 1 | PASS |
| 8 | p_log gradient is non-zero on a non-degenerate task | 1 | PASS |
| 9 | parameters() returns 9 tensors; gradients() returns 9; shapes matched | 3 | PASS |
| 10 | zero_grad clears all 9 gradients | 1 | PASS |
| 11 | update_weights moves all 9 parameters (with W_k.bias = 0 expected) | 1 | PASS |
| 12 | PowerBlock forward + input FD | 3 | PASS |
| 13 | PowerBlock training reduces loss | 2 | PASS |
| 14 | Mutation tests (non-vacuousness: p factor chain exercised; W_q FD sensitive) | 3 | PASS |

**Total: 45/45 focused checks pass.**

---

## Bugs caught and fixed during TDD

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | `PowerAttention::backward` returned empty `Tensor()` instead of `dX` | FD input check segfaulted (empty input grad → max() over zero-size data → SIGSEGV) | Computed and returned `dX = Σ_h (dQ · W_qᵀ + dK · W_kᵀ + dV · W_vᵀ)` |
| 2 | `softmax-backward` used `Σ_i dA_i` instead of `Σ_i A_i · dA_i` | W_q / W_k weights and biases FD at rel_err 1.0–2.0 (sign-flips and 3-10x magnitude errors) | Changed `sumA = Σ_i dA_local(i, j)` to `sumA_w = Σ_i A[i,j] · dA_local(i, j)` |
| 3 | `PowerBlock::backward` summed `d_attn_in` and `d_x_from_ln1` directly — wrong layer | Block input FD at rel_err 1.32 | Restructured to mirror stick_breaking: `d_res1 += ln2.backward(d_z2); d_z1 = d_res1 + attn.backward(d_res1); return ln1.backward(d_z1)` |
| 4 | `PowerAttention::update_weights` called `W_q.update_weights(lr)` which uses `Dense::grad_weights` (always zero) | W_q/W_k/W_v/W_o parameters did not move after backward | Manual SGD step on each projection using `grad_W_*` + `*_grad_bias` tensors |
| 5 | `PowerBlock` constructor had `ffn_fc1_(ffn_dim, d_model)` / `ffn_fc2_(d_model, ffn_dim)` — arguments swapped (Dense is `(in, out)`, not `(out, in)`) | Forward threw "Tensor multiplication dimension mismatch" | Swapped to `ffn_fc1_(d_model, ffn_dim)` / `ffn_fc2_(ffn_dim, d_model)` |
| 6 | `p_eps=0` was rejected in constructor | Test 3 (p=1 reference) needs p_eps=0 to set `softplus(p_log) = 1` exactly | Relaxed to `p_eps >= 0` (p = softplus(p_log) > 0 anyway) |
| 7 | Test "all 9 gradients receive signal" / "all 9 params move" assumed W_k.bias has nonzero grad | W_k.bias is *correctly* 0 (translation invariance) | Adjusted assertion to "at least 8/9" |
| 8 | Removed redundant bias accumulation pass that was double-counting grad_W_q / grad_W_k in backward | (would have produced wrong gradients if left in place) | Consolidated bias accumulation into the same per-head loop |

---

## Conventions (match attention/stick_breaking.h)
* Input / output: (N, d_model). N tokens, no explicit batch dim.
* W_q, W_k, W_v, W_o are `Dense` (weights shaped (out, in)).
* `p_log` is a learnable per-head vector of length num_heads.
* `grad_W_q` / `grad_W_k` / `grad_W_v` / `grad_W_o` are raw tensors (separate from `Dense::grad_weights`); `grad_p_log_` is the gradient through `softplus(p_log)`.
* Block: pre-LN → Power attn → residual → pre-LN → GELU FFN → residual
