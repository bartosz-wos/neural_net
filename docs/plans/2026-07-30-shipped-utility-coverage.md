# Test Coverage for 4 Shipped-but-Untested Utilities — Plan

> **For Hermes:** Use subagent-driven-development (or direct work) to add four test files following the project conventions.

**Goal:** Add test coverage for `label_smoothing`, `clip_grad_norm`, `mixup_cutmix`, and `elastic_net` — all already shipped in `include/nn/nn.h` but with no test file. Each is a heavily-used utility and every refactor currently risks silently breaking it.

**Architecture:** Four independent test files (no cross-file deps), each in the project's standard `tests/test_<name>.cpp` shape with `g_pass`/`g_fail` counters and `ASSERT`/`ASSERT_NEAR` macros. The label_smoothing and elastic_net suites do real numerical-vs-analytical gradient checks against the library's `gradient_check.h` helper. The clip_grad_norm and mixup_cutmix suites do shape/value/edge-case tests (no gradient chain).

**Tech Stack:** C++17, existing project headers, Makefile registration in the `tests:` deps line and the `run_tests:` block.

---

## Reference math (verified against source)

### LabelSmoothingCrossEntropy
- Smooth targets: `y_smooth = y * (1 - ε) + ε / K`
- Forward: `L = (1/B) * Σ_b Σ_j y_smooth[b][j] * (-log softmax(logits)[b][j])`
- Backward: `∂L/∂logits[b][j] = (softmax(logits)[b][j] - y_smooth[b][j]) / B`
- Hard labels (target rows = (B, 1) with integer class indices): converted to one-hot then smoothed
- Soft labels (target rows = (B, K)): used as-is

### clip_grad_norm_
- Compute `total_norm = sqrt(Σ_p Σ_ij g_p[i][j]²)`
- If `total_norm > max_norm`: scale every gradient by `max_norm / total_norm` (in-place)
- Otherwise: leave gradients unchanged
- Returns `total_norm` (or `max_norm` when clipping happened — since post-clip norm IS `max_norm`)

### Mixup
- λ ~ Beta(α, α) sampled via two Gamma(α,1) draws and normalized
- For each sample n: pick a random partner index; `x_mixed[n] = λ*x[n] + (1-λ)*x[partner]`
- y_a = original y, y_b = same y (the impl reuses y for both — single-label assumption)

### CutMix
- λ ~ Beta(α, α) same as Mixup
- Box dims: `target_h = max(1, min(H, √λ*H))`, `target_w = max(1, min(W, √λ*W))`
- Sample top-left `(y_start, x_start)` uniformly in valid range
- Replace pixels in the box with the partner's pixels, leave the rest with the original

### ElasticNet
- Penalty: `α * (l1_ratio * ||w||_1 + (1-l1_ratio) * ||w||_2² / 2)`
- MSE: `(1/n) * Σ (pred_i - y_i)²`
- `fit()` uses ridge warm-start (closed-form `(X^TX + α(1-λ)I)^-1 X^Ty`); CD solver uses coordinate descent soft-thresholding

---

## Task 1: Label Smoothing tests

**Files:**
- Create: `tests/test_label_smoothing.cpp`

Tests (16+ checks):
1. Default constructor `smoothing=0.1`; `get_smoothing()` round-trip.
2. Non-default `smoothing=0.2` round-trip.
3. Forward with `smoothing=0` ≡ standard cross-entropy — known analytical value on `[2,2]` logits with target=0.
4. Forward with `smoothing=0.1` matches hand-derived formula on `[1,2]` logits with target=0.
5. Smoothed targets with `smoothing=0.1, K=4`: target=2 → `[0.025, 0.025, 0.925, 0.025]`.
6. Soft-label input: `(B, K)` targets used as-is (skip smoothing conversion).
7. Numerical gradient check (per-feature) vs analytical backward: rel_err < 1e-5.
8. Backward gradient sums to ≈0 per row (the `softmax - y_smooth` rows sum to 0, property of softmax+target-consistent loss).
9. Multi-batch (B=3): loss normalized by batch.
10. Numerical gradient on multi-batch (B=3, K=3): rel_err < 1e-5.
11. Larger `smoothing=0.5`: forward still finite and non-negative; gradient still sums-to-zero per row.
12. `smoothing=1.0` (uniform target): gradient = `(softmax - 1/K) / B` exactly.

## Task 2: clip_grad_norm_ tests

**Files:**
- Create: `tests/test_clip_grad_norm.cpp`

Tests (12+ checks):
1. Single small tensor: norm below max → returns norm unchanged; tensor unchanged.
2. Single large tensor: norm exceeds max → returns `max_norm`; tensor scaled by `max_norm/orig_norm` exactly.
3. Multiple tensors: total norm accumulated correctly; clip applied to all.
4. Zero gradients: norm=0; no scaling; return 0.
5. Mixed signs (positive/negative gradients): L2 norm squares them properly.
6. Null pointer in the list: skipped, doesn't crash.
7. Single element vs max_norm=0: returns the element's norm (no division-by-zero); if > 0, scaling factor 0/max_norm = 0 → tensor zeros out.
8. Returns the *pre-clip* norm when no clipping needed; returns `max_norm` when clipping occurred.
9. Single tensor with float-valued grad values (1.0, 2.0, 2.0): norm = 3.0 exactly.
10. Cross-tensor scaling consistency: all tensors scaled by the same factor.
11. Tensor of size (1,1) vs (4,4) shape normalization works for both.

## Task 3: Mixup/CutMix tests

**Files:**
- Create: `tests/test_mixup_cutmix.cpp`

Tests (18+ checks):
1. Mixup result shape: same as input X; y_a/y_b same as input y; lambda in [0, 1].
2. Mixup interpolation: `x_mixed[n] == λ*x[n] + (1-λ)*x[partner]` — verify for one row with deterministic λ.
3. Mixup determinism with fixed `seed_` (would need to expose; if not, use multiple runs to verify finite output).
4. Mixup preserves mean when λ=0.5: `||mean(x_mixed) - mean(X)||_1 < eps` (alpha=large forces λ→0.5).
5. Mixup extreme λ=1 (alpha very large): `x_mixed == x` approximately.
6. Mixup extreme λ=0 (alpha very small): x_mixed tends to use partner pixels only — at least one row should be unchanged if partner is itself.
7. CutMix result shape: same as input X; y_a/y_b same; lambda in [0, 1].
8. CutMix with λ=1 (alpha large): box covers whole image → x_mixed == partner X entirely.
9. CutMix with λ=0 (alpha small): box size = 1×1 → 1 pixel replaced.
10. CutMix box boundaries: top-left (y_start, x_start) in valid range [0, H-target_h], [0, W-target_w].
11. CutMix preserves area ratio: number of replaced pixels ≈ λ * H * W.
12. AutoAugment: returns MIXUP or CUTMIX result depending on RNG.
13. AutoAugment both branches reachable over many runs.
14. Mixup with batch=1: should still produce a result (1 row mixed with a partner = itself if batch=1).
15. Mixup output finite (no NaN/Inf) over many runs.
16. CutMix with H=1 or W=1: degenerate case, should still be finite.
17. Mixup/CutMix alpha=0: lambda samples behave reasonably (gamma(0) is well-defined: Exp(1/rate) with rate=∞... actually gamma(0) is degenerate — skip if it crashes; document).
18. Mixup uses 1D layout (each row is one sample): output shape equals input.

## Task 4: ElasticNet tests

**Files:**
- Create: `tests/test_elastic_net.cpp`

Tests (14+ checks):
1. Penalty pure L1 (l1_ratio=1.0): equals α * L1 norm.
2. Penalty pure L2 (l1_ratio=0.0): equals α * L2²/2.
3. Penalty mix (l1_ratio=0.5): equals α * (0.5 * L1 + 0.5 * L2²/2).
4. Penalty zero weights = 0.
5. Penalty scaled by α (l1_ratio=0.5, α=2.0): equals 2x the α=1.0 case.
6. Penalty known hand-derived on [[1,2,3]] with α=1, l1_ratio=0.5: L1=6, L2²/2=14/2=7 → 0.5*6+0.5*7=6.5.
7. Objective: MSE + penalty; on trivial all-zero pred target=0: MSE=0 → objective=penalty.
8. Objective: pure L2 (l1_ratio=0): ridge-style penalty only.
9. Fit returns weight tensor of correct shape (1, d) for input (n, d).
10. Fit with closed-form warm-start: on separable data (X=I, y=given), recovers y exactly (within tol) when L1 is small.
11. Fit converges (decreases objective) over multiple iterations.
12. ElasticNetCD constructor defaults (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6).
13. ElasticNetCD fit on tiny dataset: returns finite weights of correct shape.
14. Penalty pure L1 soft-thresholds correctly: with α=0.5, l1_ratio=1.0, on w=[0.5, -0.5]: penalty = 0.5 * 1.0 = 0.5.
15. Penalty pure L2: w=[0.5, 0.5] with α=0.5, l1_ratio=0.0: penalty = 0.5 * 0.5 = 0.25.
16. Penalty symmetry: penalty(w) == penalty(-w) (L1 and L2 are sign-symmetric).

## Task 5: Makefile registration

**Files:**
- Modify: `Makefile`

For each of the 4 test files:
- Add compile rule: `$(BUILD_DIR)/test_<name>: $(LIB_OBJS) $(BUILD_DIR)/test_<name>.o\n\t$(CXX) $^ -o $@`
- Add to `tests:` deps list (one line, space-separated, alphabetically near similar tests)
- Add to `run_tests:` block: `@echo "=== Running <Name> Tests ===" && ./$(BUILD_DIR)/test_<name>`

Verify with `make tests` (compile), then `make run_tests` (run all stable suites) — all should pass.

## Task 6: Document and commit

- Update `EXPANSION_QUEUE.md` — move the entry from `## Ideas` to `## Done` with summary + test counts.
- Single commit per file: `test(utils): add <Name> test coverage (N/N checks)`.
- Or single combined commit `test(utils): add coverage for 4 shipped utilities (N+N+N+N checks)` — match recent style.
- Push to `origin/master`.

---

## Verification commands

```bash
make tests 2>&1 | tail -5     # should compile all 4 new tests
./build/test_label_smoothing   # should print "=== Summary: N passed, 0 failed ==="
./build/test_clip_grad_norm    # ditto
./build/test_mixup_cutmix      # ditto
./build/test_elastic_net       # ditto
make run_tests | tail -5       # should pass through to the end
```

## Pitfalls to avoid

- **clip_grad_norm_** mutates the *gradient tensors in place*. Tests must use their own tensors (don't share with other tests that expect pre-mutation values).
- **Mixup** uses `std::random_device{}()` in its constructor — non-deterministic. Tests that need determinism must either expose a seed (look at the header; if no seed exposed, only test finiteness/shape, not specific values) OR rely on alpha-extreme tests where the random draw converges to a fixed point.
- **Label smoothing with `targets.cols != 1` AND `targets.cols != K`**: the impl's `hard_labels` detection uses `targets.cols == 1 || (targets.rows == batch && targets.cols != K)` — the second clause is true when `targets.cols != K`. So if targets come in as (batch, K) with K != expected K, it's treated as soft and used as-is. Tests must use either `(batch, 1)` integer labels or `(batch, K)` soft labels with matching K.
- **ElasticNet `objective()`** uses `w0.rows > 0` to add an intercept — if no `weights_init` is passed, the intercept is 0. Tests should verify this behavior explicitly.
- **LabelSmoothing hard-label smoothing `smoothing=1.0`** → uniform target → loss = -log(1/K) for any logits → gradient = (softmax - 1/K)/B. Test this exact relationship.