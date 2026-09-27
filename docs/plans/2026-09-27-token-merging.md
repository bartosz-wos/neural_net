# Token Merging (ToMe) Implementation Plan

> **For Hermes:** TDD task-by-task; minimal scope, machine-precision FD checks.

**Goal:** Add a Token Merging layer for fast ViT-style inference via bipartite soft token matching (Bolya et al. ICCV 2023).

**Architecture:** Standalone `TokenMerging(reduce_ratio=0.5)` layer that takes `X ∈ R^{N × d}` and returns `X' ∈ R^{N' × d}` (N' = ⌊N · r⌋) plus an internal unmerge map. No learnable parameters in v1 (matching is a fixed function of X); the algorithm is bipartite soft matching: split N into A (even) and B (odd), compute cosine-style similarity for every (a, b) pair, select each `a`'s argmax-b as its match, merge via average.

**Tech Stack:** Hand-rolled C++17, `Tensor`/`Layer` base, build with the existing Makefile.

---

## Algorithm (paper §3.1, simplified for v1)

Given `X ∈ R^{N × d}` and `r ∈ (0, 1)`:

1. Let `n_keep = ⌊N · r⌋` (number of output tokens).
2. Partition indices `{0, ..., N-1}` into A (even) and B (odd). A has ⌈N/2⌉, B has ⌊N/2⌋.
3. For each `a ∈ A`:
   - For each `b ∈ B`: `s_{ab} = (X_a · X_b) / (2·||X_a||·||X_b|| + eps)`  (normalized cosine, ½ scale so cos ≤ 1 maps to s ≤ 1).
   - Pick `b* = argmax_b s_{ab}` (with deterministic tie-break by smallest b-index for reproducibility).
4. Merge: for each matched (a, b*) pair, `X'_k = (X_a + X_{b*}) / 2`. With `|A|` matches and `n_keep ≤ |A|`, take the first `n_keep` matches by a-index (so reduce is monotonic in the original ordering — paper §3.1 "we iteratively take the most similar pairs and merge them").

This is the standard ToMe bipartite match; the v1 reference implementation in `facebookresearch/ToMe` (https://github.com/facebookresearch/ToMe) uses the same `score = X_a·X_b / (||X_a||·||X_b||)` and matches even→odd indices.

### Backward (v1, fixed-match)

The match is a fixed function of X but we treat the merge step as differentiable via the chain rule on `X_a` and `X_{b*}` (and the cos-similarity score for the *weights* only — for v1 we are non-trainable, so we only need the chain from the merge's output to its inputs). For each output row `X'_k = (X_{a_k} + X_{b*_k}) / 2`:

- `dL/d(X_{a_k}) += (1/2) · grad_X'[k]`
- `dL/d(X_{b*_k}) += (1/2) · grad_X'[k]`

Sum across all `(a, b*)` pairs that touch the same input index. The "fixed match" assumption is the standard ToMe v1 convention (matches are computed at forward but the gradient doesn't propagate back through the argmax; this matches the paper's STE convention without explicitly using STE since v1 is non-trainable).

### Unmerge

`forward_with_unmerge(x)` returns the unmerge map `(a_dst, b_dst)` so callers can build symmetric encoders. `unmerge(grad_out)` reproduces a `(N, d)` gradient by scattering each output row's grad to BOTH of its source indices (each gets the full grad — this is the "unmerge" convention where you effectively duplicated the grad to both contributors). For v1 we use the conservative unmerge: each source index receives the grad contribution from the (single) output row it contributed to, divided by 2 (matching the forward ½ factor). This makes the round-trip `unmerge(forward(x)) ≈ x` only approximately equal (off by a constant per-token scale) — sufficient for the test that exercises the API.

## Files

- **Create:** `include/nn/layers/attention/tome.h`
- **Create:** `include/nn/layers/attention/tome.cpp`
- **Create:** `tests/test_tome.cpp`
- **Modify:** `include/nn/nn.h` — add `#include "layers/attention/tome.h"` after `sagpool.h` (line 161).
- **Modify:** `Makefile` — add `$(BUILD_DIR)/test_tome` rule (after test_sagpool at line 451-452), add `$(BUILD_DIR)/test_tome` to `tests:` deps (after test_sagpool line 728), add `=== Running Token Merging Tests ===` echo after SAGPool (line 943).

## Task 1 — Constructor validation (RED → GREEN)

**Files:** `include/nn/layers/attention/tome.h`, `tests/test_tome.cpp`

**Step 1:** Write the test (3 cases: `r=0` throws, `r>1` throws, `r=0.5` constructs with `r() == 0.5`).

**Step 2:** Run `make build/test_tome && ./build/test_tome` → expect FAIL (header not found).

**Step 3:** Write the header with the throwing constructor and the empty impl cpp.

**Step 4:** Run test → expect PASS for the 3 cases.

**Step 5:** Commit.

## Task 2 — Forward shape + bipartite partitioning (RED → GREEN)

**Step 1:** Test (N=8, d=4, r=0.5) → forward returns shape (4, 4). Assert output is finite + nonzero.

**Step 2:** Run → FAIL.

**Step 3:** Implement forward with the algorithm above.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 3 — Match-edge signature + monotonicity (RED → GREEN)

**Step 1:** Test `last_match_edges()` returns the (a_idx, b_idx) pairs, with the expected shape and the property that `a_idx < N/2 + (N%2)` and `b_idx < N/2`.

**Step 2:** Run → FAIL.

**Step 3:** Cache match edges in `forward`; expose via `last_match_edges()` returning `std::vector<std::pair<size_t, size_t>>`.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 4 — Merged output equals (X_a + X_b*)/2 on a closed-form fixture (RED → GREEN)

**Step 1:** Hand-craft a small (N=4, d=2) input where the bipartite match is unambiguous. Pre-compute the expected merge by hand and assert max |X' - X'_ref| < 1e-12 for every cell.

**Step 2:** Run → FAIL.

**Step 3:** Tighten the merge implementation to match the closed-form exactly.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 5 — Backward: FD input gradient (RED → GREEN)

**Step 1:** FD test on (N=6, d=3, r=0.5 → N'=3) with non-uniform random init and a known target. Assert `max |grad_X_fd - grad_X_ana| / max(|grad_X_fd|, 1e-12) < 1e-4`.

**Step 2:** Run → FAIL.

**Step 3:** Implement `backward(grad_out, lr)` with the (1/2)-scatter.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 6 — Forward + unmerge round-trip (RED → GREEN)

**Step 1:** Test `forward_with_unmerge(x)` returns the pair `(X', unmerge_map)` where `unmerge_map` has the right shape. Then call `unmerge(grad_X')` and verify the result has shape `(N, d)` and is finite.

**Step 2:** Run → FAIL.

**Step 3:** Implement `forward_with_unmerge` (returns X' + caches unmerge map) and `unmerge(grad_out)` returning a `(N, d)` Tensor.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 7 — parameters()/gradients()/update_weights()/zero_grad() contracts (RED → GREEN)

**Step 1:** Test all four methods return empty vectors (v1 is non-trainable) and don't throw; `name() == "TokenMerging"`.

**Step 2:** Run → FAIL.

**Step 3:** Add the methods.

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 8 — Determinism + edge cases (RED → GREEN)

**Step 1:** Tests: (a) two consecutive forward calls bit-exact; (b) r=0.5 with N=5 → N'=2 (⌊2.5⌋=2); (c) r=0.5 with N=2 → N'=1 (trivial merge); (d) forward + backward when X has zero-rows doesn't NaN.

**Step 2:** Run → FAIL.

**Step 3:** Tighten implementation to handle edge cases (zero-norm eps protection, integer floor).

**Step 4:** Run → PASS.

**Step 5:** Commit.

## Task 9 — Mutation test (catch false-positives)

**Step 1:** Manually mutate the impl to skip the merge averaging (`X'_k = X_{a_k}` instead of `(X_{a_k} + X_{b*_k})/2`). Confirm the FD input gradient test fails (rel_err jumps from 0 → 1.0).

**Step 2:** Restore. Confirm tests pass again.

## Task 10 — Register + final test pass + commit

**Step 1:** Add header to umbrella, Makefile rules + tests deps + run_tests echo.

**Step 2:** Run `./build/test_tome` 3 times → all pass.

**Step 3:** Run an adjacent test (`./build/test_sagpool` or `./build/test_vit`) → no regressions.

**Step 4:** Update EXPANSION_QUEUE.md: move ToMe from Ideas → Done, append one-line summary.

**Step 5:** Commit + push.

## Edge cases / known v1 limitations (commented in header)

- **Match-fixed (no STE).** The match edges are computed in forward but the gradient doesn't propagate back through the argmax. This is the standard ToMe v1 inference-only behavior; production training use would need STE on the matching, deferred to v2.
- **No learnable params.** Matching is purely a function of X.
- **Monotonic reduction.** When |A| > n_keep (which happens when r·N < |A|), we keep only the first n_keep matches by a-index. This matches the paper's "iterative" framing.
- **Cosine-eps.** Zero-norm rows are guarded with a tiny `eps = 1e-8` so `0/0` → `0` rather than NaN.
