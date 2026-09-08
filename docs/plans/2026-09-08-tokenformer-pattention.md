# Tokenformer / Pattention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `Pattention` — a "tokenized-parameter" attention projection that
replaces a `Dense` layer with two parameter-token tensors `K_p, V_p`, so a model
can grow its parameter count by *appending parameter tokens* without retraining
the existing ones. Plus a `TokenformerBlock` that drops Pattention into the FFN
slot of a pre-LN transformer block.

**Paper:** Wang et al., 2024, "Tokenformer: Rethinking Transformer Scaling with
Tokenized Model Parameters", https://arxiv.org/abs/2410.23168
(No official public reference impl at the time of writing; we follow the
formulation in §3 of the paper and validate against the *defining algebraic
property* — appending zero parameter tokens must leave the output bit-exact.)

**Architecture:** Two classes in `include/nn/layers/attention/tokenformer.{h,cpp}`:
1. `Pattention(d_in, d_out, num_param_tokens)` — drop-in replacement for `Dense`
   in the FFN slot. Parameters are `K_p: (P, d_in)` and `V_p: (P, d_out)` plus
   optional bias `b: (1, d_out)`. Forward: `out = softmax(x · K_pᵀ) · V_p + b`.
2. `TokenformerBlock(d_model, num_heads, num_param_tokens)` — pre-LN causal
   self-attention (re-using `Dense` projections W_q/W_k/W_v/W_o, mirroring the
   repo's `StickBreakingBlock` style) + residual + pre-LN Pattention FFN +
   residual. Demonstrates the swap-in replacement and exercises the bias-free
   FFN path end-to-end.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `LayerNorm`, `Dense`. No new deps.

---

## Verified formulation (from the paper — do NOT deviate)

### Pattention (tokenized-parameter projection)

Input `X ∈ R^{n × d_in}`, parameter keys `K_p ∈ R^{P × d_in}`, parameter values
`V_p ∈ R^{P × d_out}`, optional bias `b ∈ R^{1 × d_out}`.

```
S      = X · K_pᵀ                          in R^{n × P}      (input × param-keys)
A      = row_softmax(S)                     in R^{n × P}      (mixing weights)
out    = A · V_p + b                        in R^{n × d_out}  (linear combination of param-values)
```

This collapses to a Dense layer `(d_in → d_out)` exactly when `P = d_out` and
`K_p = W`, `V_p = I`, `b = 0` — but the power of Pattention is that `P` is
*independent* of `d_out`: you can have **more** parameter tokens than output
dimensions (expressivity win) or **fewer** (compression). The number of effective
parameters per projection is `(P · (d_in + d_out) + d_out)`, all linearly
addressable from the input, and you can grow `P` post-training without
disturbing the existing parameters.

### Defining algebraic property — must be bit-exact

If `K_p' = [K_p ; 0]` (append zero rows) and `V_p' = [V_p ; 0]` (append zero rows),
then for any input `X`:
```
out' = softmax(X · K_p'ᵀ) · V_p'  ==  out = softmax(X · K_pᵀ) · V_p   bit-exact
```
because (a) the zero rows of `K_p'` contribute `−∞` to softmax exponent sums,
making their attention weights exactly zero, and (b) those zero-weight rows
contribute zero to the output sum. This is the paper's *defining claim* (the
reason you can scale a model by appending parameter tokens), so it gets a
dedicated signature test.

### Backward (hand-derived, the part that is easy to get wrong)

Per row `t`, with `dO[t, :]` (d_out-dim):

```
dV_p[p, :] += A[t, p] * dO[t, :]
dA[t, p]    = dO[t, :] · V_p[p, :]ᵀ                        (chain through A·V_p)
db[:]      += dO[t, :]
```

Then `dA = dL/dA` and we need `dS[t, p] = dA[t, p] − A[t, p] · Σ_{p'} dA[t, p']`
(softmax backward; the subtraction is per-row, the sum is over the `P` dim).

```
dK_p[p, :] += dS[t, p] · X[t, :]
dX[t, :]  += dS[t, p] · K_p[p, :]                (sum over p)
```

All five paths are independently tested with FD.

---

## Task 1: header + skeleton, constructor validation (RED → GREEN)

**Files:**
- Create: `include/nn/layers/attention/tokenformer.h`
- Create: `include/nn/layers/attention/tokenformer.cpp`
- Create: `tests/test_tokenformer.cpp`
- Modify: `Makefile` (build rule, `tests:` deps, `run_tests` echo)
- Modify: `include/nn/nn.h` (umbrella include)

**Step 1: write failing tests for constructor validation**
- d_in = 0 → throw
- d_out = 0 → throw
- num_param_tokens = 0 → throw
- valid (3, 4, 5) → constructable

**Step 2: write the header + minimal constructor-only .cpp, run tests, watch green.**

**Step 3: commit**

---

## Task 2: forward shape + bias (RED → GREEN)

**Files:** same.

**Step 1: failing test for forward shape `(n, d_in) → (n, d_out)`, finite, nonzero**
**Step 2: implement forward with row-softmax (stable: max-subtract per row)**
**Step 3: run, watch green**
**Step 4: commit**

---

## Task 3: hand-derived N=1 single-token forward reference (RED → GREEN)

**Files:** same.

**Step 1: failing test**
- n=1, d_in=2, d_out=2, P=2. Set X=[[1,2]], K_p=[[1,0],[0,1]], V_p=[[3,4],[5,6]],
  b=0. Hand-compute S=[[1,2]], softmax→ [(e/(e+e²), e²/(e+e²))], then out = A·V_p.
- Assert exact match to 1e-12.

**Step 2: already passes from Task 2's forward; this test pins the exact value**
**Step 3: commit**

---

## Task 4: gradient FD checks — input, K_p, V_p, b (RED → GREEN)

**Files:** same.

**Step 1: failing tests**
- input gradient FD (n=4, d_in=3, d_out=4, P=5) — rel_err ≤ 1e-5
- K_p gradient FD (every entry) — rel_err ≤ 1e-5
- V_p gradient FD (every entry) — rel_err ≤ 1e-5
- bias gradient FD (every entry) — rel_err ≤ 1e-5
- Use random non-uniform init + random input so row-vs-column confusion cannot pass.

**Step 2: implement backward per the formulas above**
**Step 3: run, watch green**
**Step 4: commit**

---

## Task 5: token-append invariance — the SIGNATURE test (RED → GREEN)

**Files:** same.

**Step 1: failing test (must pass bit-exact from Task 2 onward)**
- Construct Pattention with P=3, d_in=2, d_out=3
- Compute `out_a` for some random input X
- Construct Pattention' with P=5, d_in=2, d_out=3; copy K_p[0..2], V_p[0..2] from
  original; set K_p'[3..4] and V_p'[3..4] to **all zeros**
- Compute `out_b` for the SAME X
- Assert `max_abs_diff(out_a, out_b) == 0.0` exactly

This is the paper's defining claim; bit-exact is the success criterion, not a threshold.

**Step 2: already passes; this test pins the property.**
**Step 3: commit**

---

## Task 6: reduction-to-Dense sanity (RED → GREEN)

**Files:** same.

**Step 1: failing test**
- Set `V_p = I_{d_out}` (identity matrix, requires `P = d_out`).
- Set `K_p = W` for a Dense layer.
- Set Pattention's bias = 0.
- For any random input X, assert `Pattention(X) == Dense(X)` to 1e-12.

This pins the *algebraic equivalence* — Pattention with `P = d_out`, identity
V_p, zero bias is just a repackaged Dense layer.

**Step 2: already passes; this test pins the property.**
**Step 3: commit**

---

## Task 7: update_weights moves parameters + zero_grad clears + training (RED → GREEN)

**Files:** same.

**Step 1: failing tests**
- `zero_grad()` clears every grad tensor to 0.0 (max abs = 0)
- `update_weights(0.1)` actually moves K_p, V_p, b (max abs diff > 1e-10)
- `parameters()` / `gradients()` count = 3 (K_p, V_p, b) and shape-matched
- **Training reduces loss**: tiny linear regression (y = 2x + 1) wrapped in a
  Pattention(d_in=1, d_out=1, P=4) + identity output. 100 SGD steps at lr=0.05.
  Loss should drop by > 50%.

**Step 2: implement update_weights (SGD inline, mirroring FoXAttention style)**
**Step 3: run, watch green**
**Step 4: commit**

---

## Task 8: TokenformerBlock — pre-LN causal self-attn + Pattention FFN + residuals

**Files:** same.

**Step 1: failing tests**
- Forward shape `(n, d_model) → (n, d_model)`, finite, nonzero.
- Construction: d_model must be divisible by num_heads (else throw).
- Construction: num_param_tokens > 0 (else throw).
- **Single-token causality** — perturbing token `n-1` does not affect rows `0..n-2`
  of the self-attention output (the Pattention FFN does see the perturbation
  through the residual, so we test self-attention output separately by short-
  circuiting the FFN — instead, we test that rows 0..n-2 of the output differ
  strictly less than row n-1 when perturbing the input; this catches the
  "self-attention forgot the mask" bug class without needing a hook).
- Block input gradient FD (n=4, d_model=4, H=2, P=6) — rel_err ≤ 1e-4
- All projection param FD (W_q, W_k, W_v, W_o, K_p, V_p, b) — rel_err ≤ 1e-5

**Step 2: implement TokenformerBlock**
**Step 3: run, watch green**
**Step 4: commit**

---

## Task 9: TokenformerModel — N blocks + classifier, end-to-end training

**Files:** same.

**Step 1: failing test**
- TokenformerModel(d_input, d_model, d_output, num_blocks, num_heads, num_param_tokens)
- Forward shape (n=4, d_input=3) → (n, d_output=2), finite
- Training reduces loss over 30 SGD steps at lr=0.05 on a tiny regression
- Loss decreases by > 20%

**Step 2: implement model wrapper**
**Step 3: run, watch green**
**Step 4: commit**

---

## Task 10: register in umbrella header + Makefile + final suite run

**Files:**
- Modify: `include/nn/nn.h` (add `#include "layers/attention/tokenformer.h"`)
- Modify: `Makefile` (`build/test_tokenformer` rule, `tests:` deps, `run_tests` echo)
- Verify umbrella compiles standalone: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
- Run full test suite (`make tests`).

**Mutation testing (after all tests green):**
- Drop the row-max subtraction in stable softmax → Tests 3 (hand-derived) and 4 (FD K_p) catch it (the softmax saturates).
- Drop `db += dO` → Test 4's bias FD catches it.
- Drop `−A · ΣdA` term in softmax backward → Test 4's K_p FD catches it.
- Forget the parameter-tokens-append invariance property in the header comment (cosmetic only — test still passes).

---

## Files touched (final)

- Create: `include/nn/layers/attention/tokenformer.h`
- Create: `include/nn/layers/attention/tokenformer.cpp`
- Create: `tests/test_tokenformer.cpp`
- Create: `docs/plans/2026-09-08-tokenformer-pattention.md` (this file)
- Modify: `include/nn/nn.h` (1-line include)
- Modify: `Makefile` (3-line changes)

## Out of scope (v1)

- Multi-head Pattention (paper mentions it; the FFN slot is single-head by convention here).
- Pattention variants from the paper (linear attention, sigmoid attention) — softmax is the canonical case.
- Sparse / top-k parameter-token routing — the paper mentions these but they add complexity orthogonal to the core property.
