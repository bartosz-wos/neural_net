# Deformable 1D Attention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `Deformable1DAttention` — a sparse attention layer where each query
attends to only K bilinearly-sampled key/value pairs (K ≪ N), with sampling
offsets LEARNED from the query. Plus a `Deformable1DBlock` and `Deformable1DModel`
wrapper that demonstrates end-to-end training.

**Paper:** Xia et al., ICLR 2023, "Vision Transformers with Deformable Attention",
https://arxiv.org/abs/2201.00520 (1D adaptation per §A.2; the paper's main result
is for 2D image patches but the 1D formulation is a strict subset and is used in
sequence models).

**Architecture:** Two classes in `include/nn/layers/attention/deformable_attention.{h,cpp}`:

1. `Deformable1DAttention(d_model, num_heads, num_points, num_ref_points)`
   - learnable `W_q: (d_model, d_model)`, `W_k: (d_model, d_model)`, `W_v: (d_model, d_model)`
   - learnable `W_offsets: (d_model, num_heads × num_points)` — predicts per-query, per-point offsets
   - learnable `reference_points: (1, num_ref_points)` — fixed reference grid (normalized in [0, 1])
   - learnable `W_o: (d_model, d_model)` (output projection)
   - Forward: `Q = X·W_q`, `K = X·W_k`, `V = X·W_v`. Per head h, per query t:
     `pos_{t,k,h} = reference_points[r] + σ(W_offsets[X_t]) · 2^k` (multi-scale optional — we'll use 1 scale for simplicity, paper §A.2 single-level case)
     Sample K, V at pos via bilinear interpolation; softmax over the K points.

2. `Deformable1DBlock(d_model, num_heads, num_points, num_ref_points)` — pre-LN →
   deformable attn → residual → pre-LN → GELU FFN → residual (Dense FFN).

3. `Deformable1DModel(d_input, d_model, d_output, num_blocks, num_heads, num_points, num_ref_points)`
   — input proj → N blocks → final LN → classifier.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `LayerNorm`, `Dense`. No new deps.

---

## Verified formulation (from the paper — do NOT deviate)

### Deformable1DAttention

```
Input X ∈ R^{n × d_model}

Per head h (head_dim = d_model / num_heads):
  Q_h = (X · W_q)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}
  K_h = (X · W_k)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}
  V_h = (X · W_v)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}

Offsets (paper Eq. 2 — SIGNED and BOUNDED):
  raw_h     = (X · W_offsets)[:, h·num_points : (h+1)·num_points]      in R^{n × num_points}
  ΔQ_h      = s · tanh(raw_h)                                          values in (-s, s)
  where s = offset_scale = max(1, (n-1)/2)

  tanh — NOT sigmoid. An unsigned σ(raw)·(n-1) offset can only push a sample
  RIGHT, so most points saturate against the n-1 clamp and the early part of
  the sequence becomes unreachable. tanh lets a query look backward as well as
  forward, bounded to ±s around its reference point.

Reference points (paper §3.2, cell-centre uniform grid in [0,1], scaled to [0, n-1]):
  reference_points[0, r] init = (r + 0.5) / num_points     (cell centres, strictly interior)
  ref_r = reference_points[0, r] · (n - 1)    for r in [0, num_ref_points)
  Sampling positions per query t and per point k:
  pos_{t,k} = clamp(ref_r[k] + ΔQ_h[t, k], 0, n - 1)

  Cell centres rather than the endpoint grid r/(P-1): the endpoint grid pins the
  first and last reference points exactly on the clamp boundary, where the
  position gradient is zero, wasting part of the sampling budget from step 0.

Bilinear sampling of K and V at positions pos:
  Let i = floor(pos), j = i + 1, α = pos - i, β = 1 - α
  K_sampled[t, k] = β · K[i] + α · K[j]
  V_sampled[t, k] = β · V[i] + α · V[j]
  (with j clamped to n-1 if i = n-1; if pos is exactly integer, β=1, α=0)

Attention scores (paper Eq. 5):
  score_{t, k} = (Q_h[t] · K_sampled[t, k]) / sqrt(head_dim)
  attn_{t, k} = softmax_k(score_{t, k})                  in R^{n × num_points}
  head_out[t] = Σ_k attn_{t, k} · V_sampled[t, k]        in R^{head_dim}

Output:
  concat heads → (n × d_model)
  out = concat · W_o                                     in R^{n × d_model}
```

### Backward (hand-derived, see task plan)

The bilinear sampling makes `K_sampled`, `V_sampled` differentiable in K, V,
ΔQ (offset) and reference_points (via positions). The softmax+matmul is
standard. All gradients flow through the bilinear chain: e.g.

```
dK[i]      += β · dK_sampled[t, k]
dK[j]      += α · dK_sampled[t, k]
dα         += (K[j] - K[i]) · dK_sampled[t, k]
dpos       += dα                                  (α = pos - floor(pos))
dpos       = 0   IF the clamp is active at this point   <-- REQUIRED
dΔQ[t, k]  += dpos
draw[t, k] += dΔQ · s · (1 - tanh(raw)²)
dW_offsets += X[t]^T · draw
dref[k]    += dpos · (n - 1)
```

The clamp mask is not optional. Where `pos` is clamped the forward is locally
constant in `pos`, so the true derivative is 0; without the mask the analytical
gradient is non-zero where FD correctly reports 0.

Note the FD suite alone does NOT catch either the sigmoid/tanh bug or a missing
clamp mask: the saturated region is genuinely flat, so analytical and numerical
agree at ~0 (test-config degeneracy). Dedicated invariant tests are required —
see Tests 9 and 10 in `tests/test_deformable_attention.cpp`. Test 10 must probe
the LEFT boundary: at the right boundary `i == j == n-1`, so `K[j] - K[i] == 0`
and the check passes vacuously whether or not the mask is present.

(Plus the standard softmax + matmul + W_q/k/v/o chain.)

---

## Out of scope for v1

- 2D deformable attention (paper's main contribution)
- Multi-scale offset prediction (paper §3.3 with 2^k ladder)
- Modulated attention (paper §3.4 with per-point attention weights from a learned sigmoid)
- Causal mask (deformable attention in the paper is non-causal; users wrap if needed)