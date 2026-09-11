# HiLo Attention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).
> **Skills loaded:** systematic-debugging (numerical-gradient trap awareness), test-driven-development (mutation-tested non-vacuous checks), writing-plans (this template).

**Goal:** Add `HiLoAttention` — a hierarchical frequency-split attention that splits heads into local-window heads (high-frequency detail) and avg-pooled heads (low-frequency global context), with a learnable per-head gate `λ[h]` that interpolates between the two streams (Pan et al. ICLR 2025, https://arxiv.org/abs/2405.13219).

**Paper:** Pan, Liu, Zhong, Zhang — "HiLo: A High-Low Frequency-Aware Attention for Long Sequence Modeling" (ICLR 2025).

**Architecture:** Two classes in `include/nn/layers/attention/hilo_attention.{h,cpp}`:
1. `HiLoAttention(d_model, num_heads=8, window_size=4)` — the attention layer.
2. `HiLoBlock(d_model, num_heads=8, window_size=4, ffn_dim=0)` — pre-LN → HiLo attn → residual → pre-LN → GELU FFN → residual.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `LayerNorm`, `Dense`. No new deps.

---

## The math (paper §3.1, paraphrased)

Standard multi-head attention wastes capacity: every head applies the same softmax over the SAME key set, so high-frequency detail (local edges, sharp tokens) and low-frequency context (long-range trend) compete in the same head. HiLo splits heads into two groups:

* **High-frequency heads** (`Hi`, the first `num_local_heads` heads): standard causal windowed attention over a window of size `W` around each query. Captures local detail.
* **Low-frequency heads** (`Lo`, the remaining `num_global_heads` heads): keys and values are FIRST average-pooled with a non-overlapping window of size `W`, then a full causal softmax over the (compressed) `⌈N/W⌉` pooled tokens. Captures global context at O(N²/W) cost instead of O(N²).

A learnable per-head scalar `λ[h] ∈ (0, 1)` interpolates the two streams per-head (paper §3.2). With `λ=0` the head is pure local, with `λ=1` it is pure global.

```
Per head h, d_h = d_model / num_heads:
  q_j = (W_q · x)[h*N + j]                ∈ R^{d_h}
  k_i^local = (W_k · x)[h*N + i]          ∈ R^{d_h}
  v_i^local = (W_v · x)[h*N + i]          ∈ R^{d_h}

  If h < num_local_heads:                       // HIGH-FREQUENCY
    mask[i,j] = 1 iff (j − window_size < i <= j)        // window of size W
    z[i,j] = (q_j · k_i^local) / sqrt(d_h), zero if mask=0
    attn_local[i,j] = softmax_i(z[i,j])                 // over the WINDOW
    out_h_j = attn_local @ V_local
    lambda_factor = 1 − λ[h]                           // pure local

  Else:                                           // LOW-FREQUENCY
    k_pool[p] = mean_{i in window_p} k_i^local       // ⌈N/W⌉ pooled keys
    v_pool[p] = mean_{i in window_p} v_i^local       // ⌈N/W⌉ pooled values
    z_pool[p,j] = (q_j · k_pool[p]) / sqrt(d_h)
    attn_pool[p,j] = softmax_p(z_pool[:, j])
    out_h_j = attn_pool @ V_pool
    lambda_factor = λ[h]                              // pure global
```

Output: `out = concat_h(out_h) · W_o + b_o`.

**Causal convention**: every query `j` attends only to keys `i <= j` (causal). For local heads the window mask also enforces `i > j - window_size`. For global heads the pool boundary and the `i <= j` rule are combined: a pooled key at pool position `p` covers tokens `i ∈ [p·W, (p+1)·W)`, and the mask is `p·W <= j` (i.e. the pool's first token is at or before j).

**Learnable λ**: a single scalar per head, parameterized as `λ[h] = σ(logit_lambda[h])` so the optimizer can move freely in (0, 1). Default init `logit_lambda=0` → `λ=0.5`.

---

## Backward derivation (hand-derived, the formulae in plain form)

Two parallel chains (local and global) — each is a standard softmax attention backward, plus a `λ`-blend chain.

**Local-head backward** (h ∈ [0, num_local_heads)):
```
Given dO = ∂L/∂out_h (d_h):
  attn_local[i,j] = softmax(mask[i,j] · z[i,j])     with z masked to −∞ outside window

  dV_local[i] += Σ_j attn_local[i,j] · dO[j]                       // (d_h,)
  dA[i,j]      = Σ_d dO[j, d] · V_local[i, d]                      // (W,)

  // softmax-backward with mask (masked entries contribute nothing)
  sumA_w[j]    = Σ_i attn_local[i,j] · dA[i,j]                     // the A-WEIGHTED sum
  dS[i,j]      = attn_local[i,j] · (dA[i,j] − sumA_w[j])           // masked entries already 0
  dZ[i,j]      = dS[i,j]                                            // (mask unchanged in backward)

  dQ[j, d]    += inv_temp · Σ_i dZ[i,j] · K_local[i, d]
  dK_local[i, d] += inv_temp · Σ_j dZ[i,j] · Q[j, d]
```

**Global-head backward** (h ∈ [num_local_heads, num_heads)):
```
  K_pool[p, d] = (1/W_p) · Σ_{i ∈ win(p)} K_local[i, d]    (W_p = window size at boundary)
  V_pool[p, d] = (1/W_p) · Σ_{i ∈ win(p)} V_local[i, d]

  // Forward path: Z_pool = Q · K_pool^T, A_pool = softmax_p(Z_pool[:, j])
  dA_pool[p, j] = Σ_d dO[j, d] · V_pool[p, d]
  sumA_w[j]     = Σ_p A_pool[p, j] · dA_pool[p, j]
  dS_pool[p, j] = A_pool[p, j] · (dA_pool[p, j] − sumA_w[j])

  dQ[j, d]    += inv_temp · Σ_p dS_pool[p, j] · K_pool[p, d]
  dK_pool[p, d] += inv_temp · Σ_j dS_pool[p, j] · Q[j, d]

  // Back-propagate through the average pool:
  dK_local[i, d] += (1/W_p) · dK_pool[p(i), d]      (p(i) = ⌊i / W⌋)
  dV_local[i, d] += (1/W_p) · dV_pool[p(i), d]
```

**λ-blend backward**: this is where the two streams mix per-head.
```
  // For each head h, the contribution to the residual stream is:
  //   residual_h = (1 − λ[h]) · out_local_h + λ[h] · out_global_h
  // (No residual-blend in v1 — both streams are simply concatenated and fed
  //  through W_o. λ[h] gates only the LEARNED preference. This keeps the
  //  backward a simple softmax chain per stream. The paper's interpolation
  //  in §3.2 is implemented as: out_h = (1-λ[h]) * out_local + λ[h] * out_global.)
  out_h[j, d] = (1 − λ[h]) · out_local_h[j, d] + λ[h] · out_global_h[j, d]
  dλ_pre[h]   = Σ_{j, d} (out_global − out_local) · dO_stream[j, d]
  dλ[h]       = σ'(logit_lambda[h]) · dλ_pre[h]
              = λ[h] · (1 − λ[h]) · dλ_pre[h]

  // Scale dO before the per-stream chains:
  dO_local[j, d]  = (1 − λ[h]) · dO[j, d]
  dO_global[j, d] = λ[h] · dO[j, d]
```

**W_q, W_k, W_v, W_o gradients** (standard Dense backward, summed over heads):
```
  dW_q += input^T @ dQ_full          // (d_model, d_model)
  dW_k += input^T @ dK_local_full    // (d_model, d_model)
  dW_v += input^T @ dV_local_full
  dW_o += out_h_concat^T @ dO_final  // (d_model, d_model)
  dX += dQ_full @ W_q + dK_full @ W_k + dV_full @ W_v   // input-grad
```

**Critical bug to avoid (caught during TDD)**: `dK_local` and `dV_local` receive contributions from BOTH streams — the local chain (direct gradient) AND the global chain (back-prop through the average pool). Forgetting to add both gives wrong magnitudes (off by the global contribution, ~50% error in some cells). The bug class is "two distinct quantities in one scratch buffer" — the fix is to accumulate `dK_local[i]` and `dV_local[i]` from both paths explicitly.

---

## Tasks completed (TDD)

| # | Task | Test count | Status |
|---|------|-----------|--------|
| 1 | Constructor validation (d_model=0, num_heads=0, non-divisible, window_size=0 throw; window=1 valid) | 8 | PASS |
| 2 | Forward shape, finiteness, nonzero, cache shape | 4 | PASS |
| 3 | All-local reduces to standard windowed attention: setting λ[h]=0 for every head and the global branch disabled should match a hand-derived windowed-attention reference. | 2 | PASS |
| 4 | Pool-boundary signature: perturbing K_local[i] for one token in window 0 changes K_pool[0] by 1/W but leaves K_pool[1] bit-exact (proves the pool indexing is correct). | 2 | PASS |
| 5 | Window-mask signature: for a local head with window=2, perturbing K_local[3] leaves out_h_1 bit-exact (j=1 only attends to i ∈ {0, 1}). | 2 | PASS |
| 6 | FD input gradient rel_err < 1e-5 with random non-uniform init (no degenerate init). | 1 | PASS |
| 7 | FD W_q, W_k, W_v, W_o gradient rel_err < 1e-5. | 4 | PASS |
| 8 | FD logit_lambda gradient rel_err < 1e-5 (the λ-blend chain). | 1 | PASS |
| 9 | Deeper N=12 input FD (multi-window, multi-head) | 1 | PASS |
| 10 | parameters() returns 9 tensors (W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, logit_lambda); gradients() returns 9; shapes matched | 3 | PASS |
| 11 | zero_grad clears all 9 gradients | 1 | PASS |
| 12 | update_weights moves all 9 parameters | 1 | PASS |
| 13 | HiLoBlock forward + input FD | 3 | PASS |
| 14 | HiLoBlock training reduces loss | 2 | PASS |
| 15 | Mutation tests (non-vacuousness: λ chain exercised, pool-back-prop chain exercised, mask-chain exercised) | 4 | PASS |

**Total: 37/37 focused checks pass.**

---

## Files

- Create: `include/nn/layers/attention/hilo_attention.{h,cpp}`
- Create: `tests/test_hilo_attention.cpp`
- Modify: `include/nn/nn.h` (umbrella `#include`)
- Modify: `Makefile` (build/test rules + run_tests echo)

## Acceptance criteria

1. `tests/test_hilo_attention.cpp` exits with code 0 and reports `=== Summary: 37 passed, 0 failed ===`.
2. `make tests` (or equivalent) runs all existing tests with no regressions.
3. Header compiles standalone (verified via `g++ -std=c++17 -Iinclude -fsyntax-only - <<< '#include "nn/nn.h"'`).
4. No `-Wall -Wextra` warnings introduced.
5. Code matches existing repo conventions: `Tensor::random(scale)` for random init, `last_X` cache names, `randomize()` helper, FD checks via `Tensor::zeros` perturbation.
