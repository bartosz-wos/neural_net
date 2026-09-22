# FastFormer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Wu et al. AAAI 2021 "FastFormer: Additive Attention Can Be All You Need" (https://arxiv.org/abs/2108.09084) as `FastFormerAttention` + `FastFormerBlock` + `FastFormerModel` in `include/nn/layers/attention/fastformer.{h,cpp}` with a focused test suite and full registration in the umbrella `nn.h` and Makefile.

**Architecture:** Three classes that mirror the existing attention layers in this repo (cosFormer, linformer, etc.):
- `FastFormerAttention(d_model, num_heads, causal=false)` — single multi-head FastFormer layer.
- `FastFormerBlock(d_model, num_heads, ffn_hidden=0, causal=false)` — pre-LN → attn → residual → pre-LN → GELU FFN → residual.
- `FastFormerModel(input_dim, d_model, output_dim, seq_len, num_blocks=2, num_heads=4, ffn_hidden=0, causal=false)` — input projection → stack of blocks → mean pool → classifier.

**Tech Stack:** C++17, existing Tensor (rows × cols), Layer base class, LayerNorm, raw Tensors for projections (single-head implementation collapsed into multi-head via reshape; matches repo convention).

---

## Math (single head, repeated across heads)

Given input `X ∈ R^{n × d}`, project to Q, K, V all in `R^{n × d}` (single head):

1. **Query global pooling.**
   `q_global = (1/n) * Σ_i Q_i ∈ R^d` (column vector).

2. **Modulate K by sigmoid gate from (Q_i, q_global).**
   For each token i:
     `p_i = sigmoid(W_p · [Q_i ; q_global] + b_p) ∈ R^d`  (W_p: d × 2d, b_p: d)
     `K_mod_i = p_i ⊙ K_i ∈ R^d`
   `k_global = (1/n) * Σ_i K_mod_i ∈ R^d`.

3. **Modulate Q by sigmoid gate from (K_i, k_global).**
   For each token i:
     `b_i = sigmoid(W_b · [K_i ; k_global] + b_b) ∈ R^d`  (W_b: d × 2d, b_b: d)
     `Q_mod_i = b_i ⊙ Q_i ∈ R^d`

4. **Linear-attention token-wise.**
   For each token i, compute attention over the seq dim:
     `α_i = softmax(Q_mod_i · K^T / sqrt(d)) ∈ R^n`     — a row of (n, n) attention
     `c_i = α_i · V ∈ R^d`                              — a single token's value-mix
   This is O(n · n · d) per token, O(n² · d) total. For the multi-head case this is
   O(n² · d_per_head) per head = O(n² · d_model), same complexity as standard softmax
   but with NO attention-mask overhead and a much smaller constant (no QK^T stored as
   (n,n); computed row by row). For a tractable reference impl we keep it explicit O(n²)
   for now — the O(n) savings come from a separate additive linear-attn variant.

   For the v1 we follow the *literal* FastFormer math from the paper §3.2:

     `α_i = softmax(Q_mod_i · K^T / sqrt(d))`
     `c_i = α_i · V`

   Total output: `out_i = b_i ⊙ V_i + γ ⊙ c_i` where `γ = sigmoid(w_γ^T c_i + b_γ)`
   is a scalar gate per token (W_γ: d → 1). The paper §3.2 uses both additive and
   multiplicative mixing; we implement the additive form (Eq. 7 in the paper) plus
   the b_i ⊙ V_i residual mixing.

5. **Multi-head:** split Q/K/V per head (d_per_head = d_model / num_heads), apply
   the above per head, concatenate back, project with W_o ∈ R^{d × d}.

The "FastFormer" name refers to the gating making most operations O(n); the per-token
softmax over K^T IS still O(n), but the additive pooling over Q and K_mod is O(n·d).
So the full complexity per layer is O(n·d²) (the dominant n²·d in the per-token softmax
can be replaced with the linear-attention trick: `c_i = (Q_mod_i · K^T) · V` =
`Q_mod_i · (K^T · V)`, computing `K^T · V ∈ R^{d × d}` once → O(n · d²)). For v1 we
keep the explicit form for clarity and gradient-check tractability.

## Layout convention (matches repo)

- Input/output shape: `(n, d_model)` where n = sequence length.
- All projections as raw Tensors (NOT Dense), keeping `(rows, cols)` = `(out, in)` convention.
- Forward returns `Tensor(n, d_model)`.

## Parameter count (single head, d_model=d)

- W_q: d × d, b_q: d
- W_k: d × d, b_k: d
- W_v: d × d, b_v: d
- W_p: d × 2d, b_p: d       (gating for K from (Q, q_global))
- W_b: d × 2d, b_b: d       (gating for Q from (K, k_global))
- w_γ: 1 × d, b_γ: 1       (scalar gate for the c_i mixing)
- W_o: d × d, b_o: d       (multi-head output projection)

Total per head: 7d² + 7d + 1 ≈ 7d². For multi-head, W_o is shared across heads.

---

## File Layout

```
include/nn/layers/attention/fastformer.h        # Header (declarations + doc)
include/nn/layers/attention/fastformer.cpp      # Implementation
tests/test_fastformer.cpp                       # Test suite
docs/plans/2026-09-22-fastformer.md             # This plan
```

---

## TDD Cycles

### Task 1: Header + skeleton `FastFormerAttention`

Write `fastformer.h` with the class declaration, `last_*` cache fields, and the
constructor signature. **Implement just enough to compile.**

### Task 2: Forward path

Write the full forward for `FastFormerAttention`:
- QKV projections
- q_global, k_global pooling
- p_i and b_i sigmoid gates
- α_i softmax over K^T (causal mask supported)
- c_i = α_i · V
- out_i = b_i ⊙ V_i + sigmoid(w_γ · c_i + b_γ) · c_i
- multi-head reshape + concat + W_o projection

**Test #1 (forward):** shape and finiteness of output for n=4, d=8, H=2.

### Task 3: Backward path

Write the full backward chain:
- d(out_i) = grad_output[i]
- d(sigmoid gate γ): standard sigmoid backward
- d(c_i) = γ · (dout_i - γ · dot(dout_i, c_i) / 1) (the sigmoid·c contribution) + b_i ⊙ V path back to V
  — actually: out_i = b_i ⊙ V_i + γ_i · c_i where γ_i = σ(w_γ·c_i + b_γ), so:
     dγ_i = dout_i · c_i,  d(w_γ·c_i + b_γ) = γ_i · (1 - γ_i) · dγ_i
     dc_i += γ_i · dout_i + (w_γ^T scaled) · dγ_i_back
     dV_i += b_i · dout_i + b_i · dout_i (the b_i ⊙ V_i contribution) — careful, see actual formula
- d(V_i) from both the V_i ⊙ b_i path and the α_i · V path
- d(α_i) = d(c_i) · V^T (row-wise)
- d(Q_mod_i · K^T / sqrt(d)) via softmax backward: d_scores[i][j] = α_i[j] · (dα_i[j] - Σ_k α_i[k] · dα_i[k])
- d(K) from Q_mod · K^T path (scatter to K_j = Σ_i Q_mod_i · d_scores[i][j] / sqrt(d))
- d(Q_mod) from Q_mod · K^T path (dQ_mod_i = (1/sqrt(d)) · Σ_j d_scores[i][j] · K_j)
- d(b_i) and d(K_i) from the Q-modulation step: Q_mod_i = b_i ⊙ Q_i, so dQ_i += b_i · dQ_mod_i, db_i += Q_i · dQ_mod_i
- d(K_i) from the K-modulation step: K_mod_i = p_i ⊙ K_i, dK_i += p_i · dK_mod_i
- d(K_mod) from the q_global mean: dK_mod_i = (1/n) · d(k_global) (broadcast across i)
- d(p_i) from K_mod = p_i ⊙ K_i: dp_i = K_i · dK_mod_i
- d(Q_i) from q_global mean: dq_global = (1/n) · Σ d_p_i back to (Q_i, q_global)
- d(W_p, b_p) from p_i = σ(W_p · [Q_i; q_global] + b_p): standard Linear backward with input (n, 2d)
- d(W_b, b_b) from b_i = σ(W_b · [K_i; k_global] + b_b): same pattern
- d(W_q, b_q), d(W_k, b_k), d(W_v, b_v): standard Dense-style backward

### Task 4: Tests

- Constructor validation (d_model=0, num_heads=0, non-divisible)
- Forward shape & finiteness
- Forward changes output (different inputs produce different outputs)
- Causal mask signature (causal=true masks future tokens)
- Non-causal signature (no mask)
- FD gradient check on input (rel_err < 1e-4)
- FD gradient checks on W_q, W_k, W_v, W_o, W_p, W_b, w_γ (rel_err < 1e-4)
- FD gradient checks on biases (rel_err < 1e-3)
- End-to-end training on small regression reduces loss
- parameters() / gradients() contracts
- zero_grad() clears all gradients
- Bit-exact equality test on a deterministic config (rerun twice produces identical output)

### Task 5: Block + Model wrappers

- `FastFormerBlock`: pre-LN → FastFormerAttn → residual → pre-LN → GELU FFN → residual
  (FFN optional; when ffn_hidden=0 it's an identity). Use the existing GELU + Dense for FFN.
- `FastFormerModel`: input projection (Dense) → stack of blocks → mean pool → classifier (Dense).
  Tests: forward shape, end-to-end training reduces loss, gradient shape consistency.

### Task 6: Register everywhere

- Add to `include/nn/nn.h` umbrella.
- Add to Makefile: `$(BUILD_DIR)/test_fastformer` build rule, dependency in `tests:`, echo in `run_tests`.
- Update `EXPANSION_QUEUE.md` (move from Ideas → Done).

---

## Verification

- All tests pass at machine precision (rel_err < 1e-4 typically, < 1e-3 for biases).
- Run: `make build/test_fastformer && ./build/test_fastformer` → expect ≥ 30 checks pass.
- Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` → exit 0 (umbrella compiles).

---

## Style Notes

- Match the existing attention/cosformer.h header style: top comment block, layout
  convention notes, class declaration with raw Tensors (not Dense), explicit constructors.
- Use `static Tensor row_softmax` helper in the .cpp (matches cosformer/linformer convention).
- Use `Tensor::zeros(r, c)` and `Tensor::random(r, c, scale)` for init.
- Use Xavier-uniform style init (random(0.1) is fine for d=8; we'll use 0.05 for stability).
- Single implementation file, no sub-helpers needed.
