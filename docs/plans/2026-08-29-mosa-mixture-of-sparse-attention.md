# MoSA (Mixture of Sparse Attention) Implementation Plan

**Goal:** Add `MoSAAttention` / `MoSABlock` / `MoSAModel` — content-based
learnable sparse attention via Expert-Choice routing (Piękos, Csordás,
Schmidhuber, May 2025, "Mixture of Sparse Attention: Content-Based Learnable
Sparse Attention via Expert-Choice Routing", https://arxiv.org/abs/2505.00315).

**Architecture:**

For each attention head `h` independently:

```
r_h[t]     = sigmoid((X W_r^h)[t])              in (0, 1)
I_h, r_topk = TopK(r_h, k)                       k indices + their scores
X_s        = X[I_h]                              gathered input   (k, d)
Q, K, V    = X_s W_Q^h, X_s W_K^h, X_s W_V^h    per-head proj    (k, head_dim)
M[i, j]   = 0   if I_h[i] >= I_h[j]    (causal in original positions)
          = -inf otherwise
A         = softmax(Q K^T * scale + M)           (k, k)
A_scaled  = diag(r_topk) * A                     (k, k) — router scaling
out_h     = A_scaled @ V @ W_O^h                 (k, d)
output[t] = sum over h with t in I_h: out_h[rank_of_t_in_I_h]
```

In words: each head is an "expert" that picks its own top-k tokens via a
learned sigmoid router; standard causal attention runs on the gathered subset
(causality uses *original* positions, not gathered positions); the per-head
output is then row-scaled by the router scores, projected back to d_model via
per-head W_O, and scattered back to the original positions, summing
contributions across heads.

**Backward chain:**

For each head `h`:
```
d_out_h[i]    = d_output[I_h[i]]                              (gather)
dA_scaled     = d_out_h @ (W_O^h)^T                           -> contribution to dA
d_W_O         = (A_scaled V)^T @ d_out_h                      (standard)
dV            = A_scaled^T @ d_out_h
dA            = dA_scaled * r_topk[ :, None ]                  (r per row)
d_logits      = standard softmax backward through Q K^T * scale + M
dQ, dK, dV_qk = from d_logits
d_X_s         = dQ (W_Q^h)^T + dK (W_K^h)^T + dV_qk (W_V^h)^T
scatter_add   : d_input[I_h[i]] += d_X_s[i]
d_W_Q         = X_s^T dQ         (etc for K, V, O)

# router backward:
d_r_topk[i]   = sum_j dA_scaled[i, j] * A[i, j]
d_r[j]        = d_r_topk[rank_of_j]   if j in I_h else 0
d_W_r^h[c, h] = sum_t d_r[t] * dσ(z_t)/dz_t * X[t, c]
                = sum_t d_r[t] * r_h[t] * (1 - r_h[t]) * X[t, c]
```

**Files:**
- Create `include/nn/layers/attention/mosa.h` / `.cpp`
- Create `tests/test_mosa.cpp`
- Modify `include/nn/nn.h` (umbrella include), `Makefile` (build rule + `tests:` + `run_tests:`)

**Tech Stack:** C++17, repo `Tensor`, `Layer`, `LayerNorm`, `Dense`; no external deps.

## Tasks (TDD, one behavior per cycle)

1. Write `tests/test_mosa.cpp` with constructor-validation + forward-shape tests. Run → RED (no header).
2. Implement `MoSAAttention` constructor + validation + forward. Run → GREEN.
3. Add router invariant test (`r ∈ (0, 1)`); selected-token count = k; mask causality test.
4. Add input-gradient FD check + parameter FD checks (`W_q/W_k/W_v/W_o` for one head).
5. Add W_r gradient FD check (the MoSA signature parameter).
6. Add **degeneracy test**: `k = T` (top-k = all tokens) ⇒ MoSA reduces to standard MHA.
7. Add `MoSABlock` (pre-LN + residual + GELU FFN) forward + FD check.
8. Add `MoSAModel` forward shape + training-reduces-loss.
9. Mutation test: stub router scoring ⇒ training must fail.
10. Register in umbrella + Makefile; run full `make run_tests`; commit.

**Verification:** `make build/test_mosa && ./build/test_mosa` → all checks pass; FD rel_err < 1e-4.
