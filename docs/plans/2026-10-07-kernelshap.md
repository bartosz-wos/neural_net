# KernelSHAP Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Shapley values and KernelSHAP (Lundberg & Lee, arXiv:1705.07874) for model-agnostic feature attribution, completing the interpretability category alongside the just-shipped Integrated Gradients.

**Architecture:** New header `include/nn/interpretability/shapley.{h,cpp}` exposing four stateless functions. (1) `exact_shapley` enumerates all `2^M` coalitions for small feature counts — the ground truth any approximate method must match. (2) `kernel_weights` builds the KernelSHAP least-squares weights. (3) `kernel_shap` samples coalitions, evaluates the model on masked inputs, and solves the weighted least squares via a self-contained Gaussian elimination with ridge fallback. (4) `path_dependent_shap` is the permutation (Owen) estimator. All read from a `Model` and never mutate training state.

**Tech Stack:** C++17, repo `Tensor` (real-only 2-D), `Model`, existing `Dense`. No new dependencies.

---

## Why this feature, this feature now

The `## Ideas` queue is fully drained (every bullet is commented as "popped"), so this run picks new work. The audit: `interpretability/` holds **1 header out of 294**. It is the thinnest category in the repo by an order of magnitude, and Integrated Gradients shipped on 2026-10-07 leaving it with exactly one method. IG's theoretical parent — Shapley values — is confirmed absent by content grep:

```
grep -ril 'shapley\|coalition\|kernelshap\|feature_mask' include/ tests/   # → nothing
```

KernelSHAP is the natural next item: it is the only method in the attribution family that satisfies the **efficiency** axiom (`sum(phi) == f(x) - f(baseline)`) *exactly* rather than approximately, and it is the method a practitioner is most likely to reach for next. It composes with `attribution.h` (shares `input_gradient`'s zero-grad discipline) rather than duplicating it.

## Authoritative source verification (per writing-plans §3b)

The formulas below were read from the reference implementation, **not from memory**. This matters: a prior session found that captum's trapezoidal rule under-integrates (weights sum to `(n-1)/n`, not 1) — a deviation that only surfaced by reading source. Reading `shap/explainers/_coalition.py` and `_kernel.py` on 2026-10-07:

- **Kernel weight** (`_coalition.py:373-374`):
  ```python
  def _compute_weight(total, selected):
      return 1 / (total * math.comb(total - 1, selected))
  ```
  This is the SHAP kernel `w(d) = (M-1) / (pi·d²·(M-2)·2^(M-1))` in its *sampling* form — the two are algebraically identical up to the constant absorbed into the least-squares scale. The `1/(M·C(M-1,d))` form is what the reference actually implements, so it is what this repo implements.
  **Guards:** `d=0` → `C(M-1,0)=1` → `1/M`. `d=M` → `C(M-1,M)` is **0** in Python (returns 0, not an error) — the full coalition gets weight `1/M` by the formula's own convention here, but see Decision D2, where the solver drops the `d=M` row anyway.
- **Efficiency constraint** (`_kernel.py:737-786`): solve the least squares on `M-1` free features, then recover the last as
  `phi[last] = (fx - fnull) - sum(w)`.
  This is not an optimization — it is what makes efficiency hold exactly instead of approximately. Implemented as specified.
- **Normal equations** (`_kernel.py:761-763`): `(Xᵀ W X) φ = Xᵀ W y` with `WX = kernelWeights[:,None] * X`.
- **Singular fallback** (`_kernel.py:762-775`): reference catches `LinAlgError` and re-solves as `lstsq(sqrt(W)·X, sqrt(W)·y)`. This repo has **no** lstsq, so v1 uses a tiny ridge (`λ=1e-10·trace`) instead — documented deviation D3.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Expose **both** `exact_shapley` (exhaustive, `M ≤ 12`) and `kernel_shap` (sampled) | Exhaustive is the ground truth; testing an approximation against the approximation's own reference is how silent bugs survive. `exact_shapley` is the oracle the FD-style tests compare against. |
| D2 | Efficiency enforced **by construction** via the last-feature residual, for both exact and kernel | The efficiency axiom is SHAP's defining property. Enforcing it structurally (rather than hoping the quadrature hits it) makes `sum(phi) == fx - fnull` exact in float for any coalition set. |
| D3 | Ridge fallback `λ = 1e-10 · trace(A)` instead of lstsq | No lstsq exists in-repo. At `1e-10·trace` the ridge only engages on a genuinely rank-deficient system (e.g. `M` > samples sampled) and perturbs the solution at the 1e-10 relative level — below the noise floor of every downstream assertion. |
| D4 | Masking is **full-feature** (`M = cols`); row-wise masking for batched rows | The repo's `Tensor` is 2-D and the model is feed-forward, so a column mask is the only mask that is simultaneously well-defined and invertible. Per-row masking would need a per-row model call and is out of scope for v1. |
| D5 | Attributes the **logit**, same convention as `attribution.h` | Consistency with the shipped IG. Softmax probabilities saturate; the gradient that attribution methods integrate vanishes. |
| D6 | Every entry point zeroes parameter grads before *and* after | `Dense::backward` accumulates into `grad_weights` (layer.cpp:79). An attribution pass must leave no residue. Mirrors `input_gradient`. |

## Out of scope (v1)
- Interventional vs observational marginal expectation with a background dataset (`shap.maskers`) — v1 uses the baseline-row mask, which is the `SimpleImputer`-style masker.
- TreeSHAP / LinearSHAP (path-dependent exact, model-specific).
- Interaction values (SHAP-IQ).
- Batch grouping (feature grouping / hierarchical partitioning) — `_coalition.py`'s tree building is for `shap.TreeExplainer` partitioning, not KernelSHAP itself.
- Per-row attribution for multi-row inputs beyond row 0's channel.

## File map

- **Create:** `include/nn/interpretability/shapley.h`, `include/nn/interpretability/shapley.cpp`
- **Modify:** `include/nn/nn.h` (umbrella include, after line 316)
- **Create:** `tests/test_shapley.cpp`
- **Modify:** `Makefile` (3 sites: link rule ~line 227, `tests:` list, `run_tests` line ~945)
- **Modify:** `EXPANSION_QUEUE.md` (add to `## Done`)

---

### Task 1: Header + skeleton that compiles

**Objective:** Create `shapley.h` declaring the full public API, and a `shapley.cpp` with definitions stubbed to throw, so the umbrella header compiles standalone.

**Files:** Create `include/nn/interpretability/shapley.h`, `include/nn/interpretability/shapley.cpp`; Modify `include/nn/nn.h`.

**Step 1: Write the header** — exact declarations:

```cpp
struct ShapleyResult {
    Tensor attributions;   // (cols,) — one per input feature
    double base_value;     // f(baseline) for the attributed target
    double completeness_delta;  // sum(attributions) - (f(x) - f(baseline)); 0 by construction
    size_t n_features;
    size_t n_coalitions;   // coalitions actually evaluated
    std::string method;    // "exact" | "kernel" | "permutation"
};
```

Functions: `model_output(Model&, const Tensor&, size_t target)`; `kernel_weights(size_t M, Tensor& weights)`; `exact_shapley(Model&, const Tensor& x, const Tensor& baseline, size_t target)`; `kernel_shap(Model&, const Tensor& x, const Tensor& baseline, size_t target, size_t n_samples, unsigned seed)`; `path_dependent_shap(Model&, const Tensor& x, const Tensor& baseline, size_t target, size_t n_permutations, unsigned seed)`.

**Step 2: Compile the umbrella header standalone (MANDATORY per writing-plans §3b):**
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: exit 0. A focused test including only `shapley.h` cannot catch cross-header collisions.

---

### Task 2: `model_output` + `kernel_weights`

**Objective:** RED tests for output targeting and the weight table, then GREEN implementation.

**Step 1: Failing tests** in `tests/test_shapley.cpp`:
- `model_output` on a hand-built linear model matches `W·x` at each target column.
- `kernel_weights(M, w)` produces `w[d] == 1/(M·C(M-1,d))` for `M=5`, all `d` in `0..M` — **computed independently** via a Pascal-triangle binomial, not by re-calling the impl.
- `kernel_weights(0, ...)` throws; `kernel_weights(1, ...)` gives exactly `[1.0]`.
- Weight symmetry: `w[0] == w[M]` (both `1/M`).

**Step 2: Verify RED** — run, confirm it fails because the functions are absent, not a typo.

**Step 3: Implement.** `kernel_weights` builds `C(M-1, d)` with a Pascal row (integer-exact, no overflow for `M ≤ 20`), then `1.0/(M * comb)`. **Handle `d = M`: `C(M-1, M)` is 0** → the formula would give infinity; guard it to `1/M` (matching the convention above) rather than emitting inf.

**Step 4: Verify GREEN**, then **mutation-test**: flip `1/(M*comb)` to `1/(M*comb)` with `comb(M,d)` instead of `comb(M-1,d)` → the weight test must fail. Confirm the mutation was written to disk (`grep`) before trusting a pass.

**Step 5: Commit.**

---

### Task 3: `exact_shapley` — the ground-truth oracle

**Objective:** Exhaustive Shapley values over all `2^(M-1)` coalitions per feature (fixing one feature as present reduces the work to half the full lattice).

**Files:** Modify `include/nn/interpretability/shapley.cpp`.

**Step 1: Failing tests:**
- **Hand-derived M=2 closed form.** With `f(x) = w·x` linear and features `(x0, x1)`, the Shapley values are exactly `(x0-b0)·w0` and `(x1-b1)·w1` — the cooperative-game linearity property. This is an *independent* reference derived from the definition, not from the code.
- **Efficiency:** `sum(phi) == f(x) - f(baseline)` to 1e-12 on a nonlinear MLP.
- **Dummy/null feature:** appending a feature the model never reads gives `phi == 0` exactly.
- **Symmetry:** with two identical features, `phi_0 == phi_1` exactly.
- **Efficiency at `M=1`**: a single feature gets `phi_0 == f(x) - f(baseline)` exactly.
- Non-regression: a `2^M`-scale call leaves parameter grads untouched.

**Step 2: Verify RED.**

**Step 3: Implement.** For each feature `i`, enumerate subsets `S ⊆ {0..M-1} \ {i}`; accumulate
`phi_i = Σ_S |S|!(M-|S|-1)!/M! · [ f(mask(x,b,S ∪ {i})) − f(mask(x,b,S)) ]`
with the factorial ratio computed in a numerically-stable way (compute the *coefficient* per `|S|` once per `|S|`, and guard `M ≤ 12` by throwing with a clear message above that — `2^12 = 4096` per feature is already 49k model calls at `M=12`).
Then apply **D2**: force `phi[last] = (fx − fnull) − Σ_{i<last} phi_i` so efficiency is exact by construction.

**Step 4: Verify GREEN** + full-suite `make run_tests` for regressions.

**Step 5: Mutation-test (2 mutations, both must be caught):**
- Remove the `|S|!(M−|S|−1)!/M!` weight (uniform averaging) → hand-derived M=2 and dummy-feature tests must fail.
- Drop the D2 efficiency-residual line → the `completeness_delta` test must fail.

**Step 6: Commit.**

---

### Task 4: `kernel_shap` — sampled WLS

**Objective:** KernelSHAP via the weighted-least-squares solve from `_kernel.py`.

**Step 1: Failing tests:**
- **Approximation converges to exact.** For `M ≤ 6` with a large sample budget, `kernel_shap` matches `exact_shapley` to a stated tolerance (say `1e-3` relative). This is the headline test and it is a real oracle comparison, not self-consistency.
- **Efficiency holds exactly** for any sample count including `n_samples` too small to identify `M` features — the D2 residual makes this hold regardless.
- `kernel_shap` with `n_samples = 0` throws (no path to identify).
- Determinism: same seed → bit-exact identical attributions across two calls.
- Different seeds → *close* but not required equal (guards against accidentally ignoring the seed).

**Step 2: Verify RED.**

**Step 3: Implement.** Sample coalitions by enumerating the `d=1..M-1` lattice (shap's efficient approach) rather than independent bitmask draws, because independent uniform draws over `2^M` under-sample the informative mid-size coalitions and badly degrade the WLS conditioning. Build design matrix `X` (rows = coalition masks, cols = features present) and `y = f(masked) − f(baseline)`, weight row `k` by `kernel_weights(M)[|S_k|]`.

Solve `(XᵀWX)φ = XᵀWy` for `M-1` free features via **Gaussian elimination with partial pivoting** written inline (~40 lines), then D2-residual the last. On a singular pivot, add ridge `λ = 1e-10·trace(A)` to the diagonal and retry (D3).

**Step 4: Verify GREEN.**

**Step 5: Mutation-test (3 mutations):**
- Drop the kernel weights entirely (uniform `w=1`) → the convergence-to-exact test must fail.
- Transpose `XᵀWX` → must fail.
- Remove the D2 residual → the exact-efficiency test must fail.

**Step 6: Commit.**

---

### Task 5: `path_dependent_shap` — permutation estimator

**Objective:** Owen's permutation estimator; the third standard SHAP method.

**Step 1: Failing tests:**
- Converges toward `exact_shapley` with enough permutations.
- Efficiency holds exactly (D2 applies here too).
- Deterministic under a fixed seed.

**Step 2: Verify RED.**

**Step 3: Implement.** Deterministic-LCG permutation sampling (the repo already uses LCG-style deterministic RNG elsewhere — reuse that convention, do **not** call `rand()`; `adaln_zero` has a known intermittent-NaN history and shared global RNG streams cause cross-suite flakiness).

> **CORRECTION 2026-10-08 — the Owen weight in this step is WRONG. Do not implement it.**
> The original text said: credit `phi_i += (f(prefix ∪ {i}) − f(prefix)) · (M−1−|prefix|)/(M−1)`. That per−position weight must be dropped: the marginals are averaged **unweighted**.
> Proof it is wrong, independent of any reference implementation: on a linear game `f(x) = Σ c_i x_i`, the marginal of feature `i` is `c_i·(x_i − b_i)` for **every** coalition it can join — it does not depend on the prefix. Averaging over orderings therefore recovers `phi_i` **only** if the weights are unweighted (or sum to exactly the normaliser). Three candidate weight families were tested against the analytic linear-game oracle and all fail: `1/(M·C(M−1,k))` sums to 2/3 at M=4 (biased 1.5×), `C(M−1,k)/2^(M−1)` sums to 3 (biased 3×), `k!(M−1−k)!/(M−1)!` sums to 8/3 (biased 1.5×).
> Authority: `shap/explainers/_permutation.py:209-216` accumulates `row_values[ind] += outputs[i+1] − outputs[i]` with no weight, walks each ordering forward **and** backward (antithetic), and divides by `2·npermutations` (line 248).
> Note the plan's mutation step 5 is correspondingly obsolete: dropping a factor that does not exist is not a mutation. The real mutation is on the loop bound — see below.

**Antithetic pass — loop bound (add 2026-10-08, after a real bug).** The forward pass walks `order[0..M-1]`, crediting `order[k]` with `v({order[0..k}) - v({order[0..k-1})`. The backward pass must walk DOWN from the full coalition so that `order[k]` joins the complementary **suffix** `{order[k+1..M-1})` — that pairing is the antithetic variance reduction. It must visit **k = M-1 down to k = 0 inclusive**. Writing it as `for (size_t k = m; k-- > 1;)` skips `k = 0`, which credits `order[0]` once and every other position twice while still dividing by `2*n_permutations`, giving a systematic bias of exactly `1 - 1/(2m)` that does **not** shrink with budget. The correct loop is `for (size_t k = m; k-- > 0;)`.

**Step 4: Verify GREEN.** **Step 5: Mutation-test** — mutate the backward loop bound from `k-- > 0` back to `k-- > 1` and confirm the suite fails (measured: 25 failures). **Also required: a linear-game unbiasedness test** — on `f(x) = x` the unweighted permutation estimator is exact for *any* budget, so it is a bias detector that pure variance cannot pass. The budget-comparison convergence test alone is not sufficient, because variance shrinks the error at both budgets too. **Step 6: Commit.**

---

### Task 6: Register + full verification

**Objective:** Wire into the build and prove nothing regressed.

**Step 1:** Add the 3 Makefile sites. Match existing formatting exactly (the `tests:` list is one long continuation; the `run_tests` line wraps in `.run_tests_failed`).
**Step 2:** `make build/test_shapley && ./build/test_shapley` → expect all green, deterministic across 3 reruns, warning-clean under `-Wall -Wextra`.
**Step 3:** `make run_tests` → confirm no new failures vs the 1 known (`test_neural_ode`).
**Step 4:** Umbrella standalone compile (re-verify after integration).
**Step 5:** Move the entry to `## Done` in `EXPANSION_QUEUE.md`; add the new idea bullet to `## Ideas` so the queue is not left drained.
**Step 6:** Commit + push.

---

## Self-contradiction scan

- Ordering of forward declarations in the header matches the order of definition in the `.cpp` — single consistent order (Task 1 order preserved through Tasks 2-5).
- D2 is stated once, applies identically to all three estimators, and each task's test list re-asserts it. No task contradicts it.
- The "out of scope" list (batch grouping, TreeSHAP, SHAP-IQ) is consistent with D4's full-feature-only masking; D4 does not silently promise per-row masking.
- Task 3's `M ≤ 12` guard and Task 4's arbitrary `M` are both consistent with the `exact`/`kernel` distinction stated in D1.