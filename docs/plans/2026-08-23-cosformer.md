# cosFormer Implementation Plan

> Implemented from `EXPANSION_QUEUE.md` "Ideas" entry (cosFormer, Qin et al. ICLR 2022).

## Goal

Add `cosFormer` — a linear-time attention variant that replaces softmax attention with a kernel that is (i) non-negative and (ii) re-weighted by a cosine distance factor, capturing both key properties of softmax attention that prior kernel-method approximations (Linear Transformer, Performer) failed to match.

## Architecture

`CosFormerAttention` is a drop-in replacement for `PerformerAttention` / `LinformerAttention`. The trick is the Ptolemy identity `cos(a-b) = cos(a)·cos(b) + sin(a)·sin(b)`, which decomposes the cos re-weighting into two outer-product-able terms. Forward becomes O(N·d²) instead of O(N²·d):

```
Q_relu = ReLU(Q)        (n, d)
K_relu = ReLU(K)        (n, d)
Q_cos[t, j] = Q_relu[t, j] · cos(π t / (2M))
Q_sin[t, j] = Q_relu[t, j] · sin(π t / (2M))
K_cos[t, j] = K_relu[t, j] · cos(π t / (2M))
K_sin[t, j] = K_relu[t, j] · sin(π t / (2M))

KV_cos = K_cos^T @ V    (d, d)    ← one (d×d) accumulator, O(N·d²) total
KV_sin = K_sin^T @ V    (d, d)    ← another (d×d) accumulator
Ksum_cos = Σ_t K_cos[t, :]  (d,)  ← row-sum of K_cos
Ksum_sin = Σ_t K_sin[t, :]  (d,)

num[t, k] = Q_cos[t] @ KV_cos[:, k] + Q_sin[t] @ KV_sin[:, k]    (n, d)
den[t]    = Q_cos[t] @ Ksum_cos + Q_sin[t] @ Ksum_sin              (n,)
out[t, k] = num[t, k] / (den[t] + ε)                               (n, d)
output    = out @ W_o^T                                             (n, d)
```

For each new query the work is O(d²). Across N queries it's O(N·d²) — `O(N)` for fixed d, vs softmax's `O(N²)`.

## Tech Stack

- C++17, hand-rolled forward/backward (matches repo conventions for Linformer/Performer/Hyena)
- Single-head (matches Linformer/Performer convention; multi-head via concatenation)
- Bidirectional (non-causal) v1 — paper does both, this matches existing repo conventions
- `eps=1e-6` denominator floor
- M default = seq_len (paper requires M ≥ N)

## Files

- **Create**: `include/nn/layers/attention/cosformer.h`
- **Create**: `include/nn/layers/attention/cosformer.cpp`
- **Create**: `tests/test_cosformer.cpp`
- **Modify**: `include/nn/nn.h` — add `#include "layers/attention/cosformer.h"`
- **Modify**: `Makefile` — add `build/test_cosformer` rule, deps, echo

## Implementation

### 1. Header `cosformer.h`

Three classes:
- `CosFormerAttention` — the attention layer itself (forward, backward, params, gradients, accessors)
- `CosFormerBlock` — pre-LN → attn → residual → pre-LN → FFN(GELU) → residual
- `CosFormerModel` — stack of blocks + classifier

Public accessors expose `W_q, W_k, W_v, W_o` (raw tensors) for direct gradient testing.
Test helper `zero_pos_for_test()` lets mutation tests verify the cos/sin position vectors are actually used.

### 2. Implementation `cosformer.cpp`

#### Forward
1. Compute Q, K, V via linear projections
2. Apply ReLU to Q, K (non-negativity per paper)
3. Per-row scaling: `Q_cos[t, j] = Q_relu[t, j] · cos_pos[t]`, etc.
4. Two (d×d) KV accumulators and two (d,) row-sums
5. Compute num, den, out_pre = num / (den + ε)
6. Apply W_o projection

#### Backward
1. **Output projection**: `d_out_pre = grad_output @ W_o`, `grad_W_o += grad_output^T @ out_pre`
2. **Through division by den**: `d_num = grad_out_pre / z`, `d_den = -Σ_k grad_out_pre · num / z²`
3. **Through num/den decomposition**:
   - `d_KV_cos = Q_cos^T @ d_num`, `d_KV_sin = Q_sin^T @ d_num`
   - `d_Ksum_cos = Q_cos^T · d_den`, `d_Ksum_sin = Q_sin^T · d_den`
   - `d_Q_cos = d_num @ KV_cos^T + d_den · Ksum_cos`, similar for sin
4. **Through KV = K^T @ V**: `d_V = K_cos · dKV_cos + K_sin · dKV_sin`, `d_K = dKV @ V^T + d_Ksum broadcast`
5. **Through per-row cos/sin scaling + ReLU**: `d_Q_relu = d_Q_cos · cos_pos + d_Q_sin · sin_pos`, then ReLU backward
7. **Through projections**: `d_input = d_Q @ W_q + d_K @ W_k + d_V @ W_v`, `grad_W_q = d_Q^T @ X`, etc.

#### Block
Standard pre-LN block. Key implementation detail: the GELU backward needs the **pre-activation** cached separately, since the post-GELU output gets overwritten in-place during forward. Also, the residual backward requires adding `d_res1` (NOT just `grad_output`) to `d_input` — the residual from `res1 = input + attn_out` sends the full `dL/d(res1)` to input.

### 3. Tests `test_cosformer.cpp`

12 focused checks, all passing at machine precision:

1. **forward shape** — `(n=6, d=4) → (6, 4)`
2. **output finite + nonzero**
3. **input gradient FD check** — rel_err 1.1e-9
4. **parameter gradient FD checks** (W_q, W_k, W_v, W_o) — rel_err 6.7e-7, 6.7e-7, 1.8e-9, 3.4e-10
5. **position vector properties** — `cos(0)=1`, `sin(0)=0`, `cos²+sin²=1`
6. **cos re-weighting symmetry** — `cos(i,j) == cos(j,i)` since cos is even
7. **recency bias** — cos weights decrease monotonically with `|i-j|`
8. **block forward shape**
9. **block input gradient FD check** — rel_err 1.5e-9
10. **model training reduces loss** — 57% reduction over 30 SGD steps at lr=0.005
11. **M > seq_len** — longer cos period still produces finite output
12. **mutation test** — zeroing cos_pos alters output (proves the cos re-weighting path is actually exercised)

## Verification

```bash
make build/test_cosformer -j4
./build/test_cosformer
# Expected: 20/20 checks pass

make run_tests
# Expected: all suites pass (the ODERNN grad-check failure is the documented
# pre-existing deferred entry in NOT_FIXED.md; cosFormer adds no regressions)
```

## Bugs caught during TDD

1. **size_t underflow in test** — `(double)(i - j)` where `i, j` are `size_t` produced `1.8e19` for `i < j`, leading to nonsense `cos` values. Fixed by casting through `int` first.

2. **Shape mismatch in block backward** — I was passing `(n, ffn_dim)` gradient to `ln2_.backward()` which expects `(n, d_model)`. Fixed by routing through `ffn_fc1_.backward()` first to get the correct shape, then through LN.

3. **Wrong residual bypass** — `d_input = d_ln1 + grad_output` only accounted for the outer residual (`out = res1 + ffn_out`). Missing the inner residual bypass `res1 = input + attn_out` which sends `dL/d(res1)` (the FULL `grad_output + d_ln2`) to input. Fixed to `d_input = d_ln1 + d_res1`.

4. **Test asserted wrong invariant** — Initial test 6 asserted `s(i,j) == s(j,i)` symmetry of the **full kernel** `s = ReLU(Q[i])·ReLU(K[j]) · cos(π(i-j)/2M)`. But `ReLU(Q[i])·ReLU(K[j])` is NOT symmetric (Q ≠ K). Fixed to test only the **cos factor** symmetry, which is what cosFormer structurally contributes.

## What cosFormer captures vs. softmax

- **Non-negativity**: enforced by `ReLU(Q), ReLU(K)`. The cos/sin scaling preserves sign.
- **Recency bias / locality**: `cos(π(i-j)/2M)` peaks at `i=j` and falls off with `|i-j|`. For `M=N`, the kernel value reaches 0 at `|i-j|=N` (the edges of the attention window).
- **Bidirectional**: non-causal (matches Linformer/Performer in this repo). Causal masking would be a wrapper layer.

## Reuse opportunities for future work

The `ReLU(x) ⊙ cos/sin` feature-map decomposition in cosFormer is structurally similar to other linear-attention variants in the repo. If a future paper proposes "ReLU(Q)·learnable_position", the same per-row scaling infrastructure could be reused. The `K_cos^T @ V` + `K_sin^T @ V` pattern is also the foundation of any kernel-based linear attention with a factorized feature map.