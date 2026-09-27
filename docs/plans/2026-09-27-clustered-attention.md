# Clustered Attention Implementation Plan

> **For Hermes:** TDD task-by-task; minimal scope, machine-precision FD checks.

**Goal:** Add Clustered Attention (Vyas, Katharopoulos, Fleuret — NeurIPS 2020, https://arxiv.org/abs/2007.04825), a fast O(N·C·D) approximation to softmax attention via query clustering. This is the missing classic sparse-attention variant in the repo's attention library; it complements the already-shipped Linformer/Performer/LSH/BigBird which all approximate attention differently.

**Architecture:** A single attention primitive `ClusteredAttention(d_model, num_heads, num_clusters, top_k=0, iters=5)` that takes `X ∈ R^{n × d_model}`, clusters queries via K-Means (initialized via LSH-style random-hyperplane bucket seeding to match the paper's recipe), computes standard scaled-dot-product attention on the cluster centroids, and broadcasts cluster outputs to all members of the cluster. With `top_k > 0` the layer activates the "improved clustered attention" variant: per cluster, the mass assigned by the clustered attention to its top-k keys is *redistributed* via exact per-query softmax over those keys. As `num_clusters → n` and `top_k = 0` the output converges to vanilla softmax attention bit-exactly (the special case we use to write FD checks).

**Tech Stack:** Hand-rolled C++17, `Tensor`/`Layer` base, build with the existing Makefile.

---

## Algorithm (paper §3.2 + §3.3)

For `Q, K, V ∈ R^{n × d_k}` (per head; we run per-head), `C ∈ [1, n]` clusters, optional top-k refinement `k ≥ 0`:

### 3.2 — Vanilla clustered attention

1. **Cluster the queries.** K-Means on `Q` (rows are queries), `iters` iterations of Lloyd's algorithm (random init + paper §3.2.2 "hamming-distance LSH" seeding alternative is overkill for our v1; we use Xavier-uniform random init, fixed seed). Returns `S ∈ {0,1}^{n × C}` (one-hot cluster assignment) and centroids `Q_c ∈ R^{C × d_k}`.
2. **Clustered attention matrix.** `A_c = softmax(Q_c · K^T / √d_k) ∈ R^{C × n}`. Softmax row-wise (one row per centroid).
3. **Clustered values.** `V̂_c = A_c · V ∈ R^{C × d_v}`.
4. **Broadcast.** `V̂_i = V̂_c[cluster_assign[i], :] ∈ R^{d_v}` — output for query `i` is the value of its centroid.

### 3.3 — Improved (top-k redistribution)

After step 3, for each cluster `j ∈ [0, C)`:
- **Top-k mask `T ∈ {0,1}^{C × n}`** where `T[j, l] = 1` iff `l` is among the top-k keys for cluster `j` (by `A_c[j, :]`).
- **Mass:** `m̂_j = Σ_{l: T[j,l]=1} A_c[j, l]` — the total probability mass assigned to those keys.
- For each query `i` with `cluster_assign[i] = j`:
  - `exp_logits_l = exp(Q_i · K_l / √d_k)` for `l` with `T[j, l]=1`.
  - `local_softmax_l = exp_logits_l / Σ_{l' with T[j,l']=1} exp_logits_{l'}`
  - For the local positions: `attn_l = m̂_j · local_softmax_l`; for non-local: `attn_l = A_c[j, l]`.
- `V̂_i = Σ_l attn_l · V_l`. Two contributions are summed naturally via the formula.

### Backward (clustered, no top-k for FD-check tractability)

The cluster assignment is **fixed** during backward (the standard practice; matches the paper's analysis). The chain:

`grad_Q[i]` (sum per-query) ← flows through the per-query attention `V̂_i = V̂_c[cluster_assign[i], :]`:
- `dL/d(V̂_c[c, :]) += grad_V̂[i]` for all queries `i` with `cluster_assign[i] = c`.
- `dL/d(A_c[c, l]) = Σ_d V[l, d] · grad_V̂_c[c, d]` ← standard matmul backward.
- `dL/d(A_c_pre[c, l]) = A_c[c, l] · (dL/d(A_c[c, l]) − Σ_{l'} A_c[c, l'] · dL/d(A_c[c, l']))` ← softmax backward.
- `dL/d(Q_c[c, d]) = (1/√d_k) · Σ_l K[l, d] · dL/d(A_c_pre[c, l])` ← matmul backward.
- `dL/d(Q[i, d]) += dL/d(Q_c[c, d]) / |cluster c|` for all `i` with `cluster_assign[i] = c` (K-Means centroid update rule, "fixed centroid" backward — matches the paper's analysis assuming the centroid doesn't move).
- `dL/d(K[l, d]) += (1/√d_k) · Σ_c Q_c[c, d] · dL/d(A_c_pre[c, l])`.

For the broadcast: `dL/d(V[l, d]) += dL/d(A_c[c, l])_row_aggregated` flows through the `V̂_c = A_c · V` matmul as above.

### Key testable invariants

1. **C=N recovers vanilla attention** bit-exactly (each query is its own cluster, no approximation).
2. **Cluster broadcast property**: queries with `cluster_assign[i] = cluster_assign[i']` get identical outputs.
3. **Forward determinism**: same seed → same clusters → identical output across calls.

## Files

- **Create:** `include/nn/layers/attention/clustered_attention.h`
- **Create:** `include/nn/layers/attention/clustered_attention.cpp`
- **Create:** `tests/test_clustered_attention.cpp`
- **Modify:** `include/nn/nn.h` — add `#include "layers/attention/clustered_attention.h"` after `fastformer.h`.
- **Modify:** `Makefile` — add `build/test_clustered_attention` rule, `tests:` deps entry, `=== Running Clustered Attention Tests ===` echo in `run_tests`.

## Task 1 — Header + constructor + accessors (RED → GREEN)

Include `Tensor`/`Layer`, define `ClusteredAttention(d_model, num_heads, num_clusters, top_k=0, iters=5, causal=false)` with constructor validation:
- `d_model = 0` throws
- `num_heads = 0` or `d_model % num_heads != 0` throws
- `num_clusters < 1` throws
- `top_k < 0` throws

Tests: 4 throw cases + 4 accessors (`d_model`, `num_heads`, `num_clusters`, `top_k`, `iters`, `causal`, `name() == "ClusteredAttention"`).

## Task 2 — Forward shape and vanilla-attention equivalence (RED → GREEN)

Test with `num_clusters = n` (no clustering) — must match vanilla softmax attention bit-exactly (within FP noise). Tests:
- `n=4, d=4, H=2, C=4 → forward shape (4, 4)` finite, nonzero.
- `n=4, d=4, H=2, C=4 → matches vanilla softmax attention for the same Q/K/V` to rel_err < 1e-10.

## Task 3 — Cluster broadcast property (RED → GREEN)

Test: queries with identical Q get identical outputs (they share a cluster). With `n=6, d=4, H=1, C=2`, set `Q[0] = Q[1]`, verify `output[0] == output[1]`.

## Task 4 — Forward determinism (RED → GREEN)

Two consecutive forwards (with the same seed for K-Means) produce bit-exact output (no global RNG state drift).

## Task 5 — Input FD check (RED → GREEN)

For `n=6, d=3, H=1, C=6` (clustering recovery), center-difference FD for the input gradient Q chain. Expected `rel_err < 1e-6` (FD with eps=1e-5; the clustering-recovery path makes this exact).

## Task 6 — Clustered broadcast gradient (RED → GREEN)

For `n=6, d=3, H=1, C=2` (real clustering), verify that perturbing `Q[0]` does not change `output[1]` and vice-versa when `0` and `1` are in different clusters (proof the broadcast is per-cluster).

## Task 7 — top-k path bit-exact equivalence (RED → GREEN)

With `top_k = n` (mask is full), the improved-clustered math must reduce to vanilla softmax attention. Test on `n=4, d=3, H=1, C=2, top_k=4` against vanilla attention to rel_err < 1e-10.

## Task 8 — parameters/gradients/update_weights/zero_grad contracts (RED → GREEN)

The layer has 4 parameters (W_q, W_k, W_v, W_o, all `(d_model, d_model)`). Verify:
- `parameters()` returns 4 tensors of shape `(d_model, d_model)`.
- `gradients()` returns 4 tensors of matching shape.
- `update_weights(lr)` moves parameters when gradients are non-zero.
- `zero_grad()` clears all gradients.

## Task 9 — Multi-head + training reduces loss (RED → GREEN)

Full block with `n=4, d_model=8, H=2`: forward shape, training on a synthetic permutation-invariant regression reduces loss > 50% over 30 SGD steps.

## Task 10 — Mutation test (RED → GREEN)

Stub out the cluster-broadcast (replace `output[i] = V_c[cluster_assign[i]]` with `output[i] = 0.0`). At least 3 tests fail. Revert mutation.

---

## Total

10 tasks, each RED → GREEN. No FD checks expecting sub-1e-4 precision on the BROADCASTED (clustered, C<N) path — that's an inherently lossy approximation. FD checks are targeted at the C=N (full recovery) and top_k=N paths where the layer is mathematically exact.

Conventions match recent additions (tome, sagpool, gatv2):
- Style: header docstring with paper info + algorithm + use case + why it matters.
- Makefile: `build/test_clustered_attention` rule added at the end of the per-test build rules, `tests:` deps list, `=== Running Clustered Attention Tests ===` echo inserted in `run_tests`.
- nn.h umbrella: appended after `fastformer.h` (alphabetical ordering).
