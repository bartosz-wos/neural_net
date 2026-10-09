# Partial Dependence / ICE Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship one-way Partial Dependence curves and Individual Conditional Expectation
curves (Greenwell, *Partial Dependence Plots*, R Journal 2017) as the fourth
interpretability method, verified in closed form.

**Architecture:** `include/nn/interpretability/partial_dependence.{h,cpp}`. Three public
surface pieces: a model-free `pd_grid` (percentile/quantile grid builder), a
model-free `ice_curves` (the per-sample response curves, which PD averages), and
`partial_dependence(model, X, feature, cfg, target)` returning both in one pass.
Stateless free functions + plain config/result structs, matching `shapley.h` / `lime.h`.

**Tech Stack:** C++17, repo `Tensor` (2-D double), `Model::forward`, no new deps.

---

## CITATION CORRECTION (verified, not assumed)

`EXPANSION_QUEUE.md` attributes PD/ICE to **"Greenwell arXiv:1409.7002"**. That is
**wrong**. Queried the arXiv API on 2026-10-09:

```
id_list=1409.7002 → "Entropy and Optimization of Portfolios"
```

a Markowitz/portfolio-optimization paper with no relationship to partial dependence.
I could not find the queue's ID attributed to Greenwell anywhere, so this plan cites
what is actually verifiable:

- **Greenwell, Brandon M. "Partial Dependence Plots."** *R Journal* 9(1):421–436,
  2017. <https://journal.r-project.org/articles/RJ-2017-016/RJ-2017-016.pdf> — the
  canonical statement of the method.
- **Greenwell & Boehmke, "pdp: An R Package for Constructing Partial Dependence
  Plots"**, *R Journal* 9(1):421–436 (companion), and
  <https://digitalcommons.unl.edu/r-journal/438/>.
- Note: Greenwell's thesis (Univ. of Missouri, 2017, "Exploring Some Axis
  Anomalies in Machine Learning Models") is the predecessor but is not a
  citable standalone paper for this feature.

This correction goes back into `EXPANSION_QUEUE.md` with the Done entry so a future
session does not re-copy the bad ID.

## SEMANTICS (read from reference source, 2026-10-09)

Read from `sklearn/inspection/_partial_dependence.py` and
`sklearn/inspection/_plot/partial_dependence.py` on GitHub `main`, not from memory:

| Rule | Source line | Value / behavior |
|---|---|---|
| One-way PD definition | `_partial_dependence_brute` docstring, :224-231 | "average of the predictions of an estimator over a grid of feature values"; every sample of `X` has its variables of interest replaced by the grid value |
| Grid endpoints | `partial_dependence(percentiles=...)` :377 | **`percentiles=(0.05, 0.95)`** — NOT `(0,1)` and NOT min/max |
| Grid resolution | `grid_resolution=100` :378 | equally spaced, `np.linspace(lo, hi, num=grid_resolution, endpoint=True)` :149-153 |
| Quantile rule | `_nanquantile` (`sklearn/utils/stats.py:13-38`) | `np.nanquantile(column, quantiles, axis=0)` = **linear interpolation** between order statistics |
| Low-resolution fallback | :138-142 | if the feature has `< grid_resolution` **unique** values, use the sorted uniques instead of a linspace — a 3-valued integer column must not get 100 grid points |
| Degenerate column | :144-148 | if the two empirical percentiles are `allclose`, the reference **raises**; v1 must not silently return a flat curve from a zero-width grid |
| ICE centering | `_plot/partial_dependence.py:1249,1254` | `preds = preds - preds[..., 0, None]` — subtract the **grid[0]** (left endpoint) value, per instance for ICE and on the average for PD |
| `subsample` | :256, 492-499 | `1000` default, and **the full dataset still computes the average** when `kind='both'` — subsample limits ICE *curves drawn*, not the PD average |

Two of these are exactly the kind of detail that produces a plausible-but-wrong
implementation:

1. **Percentiles, not min/max.** `(0.05, 0.95)` is the reference default and the
   reason PD curves do not blow up at isolated outliers. `percentiles=(0,1)` is a
   legal config, but it must be *chosen*, not inherited from a wrong default.
2. **ICE centering subtracts grid[0], not the per-curve mean or the curve's own
   value at that instance's actual feature value.** The subtract is
   `preds[..., 0]` — the FIRST grid point — for every curve, and the PD average is
   centered by the average's OWN grid[0], not by the mean of the per-curve
   grid[0]s. Those coincide here (averaging commutes with the subtraction) but the
   test should assert the per-curve form anyway, since that is where a
   reimplementation slips.

## CONVENTIONS (inherited, do not invent)

- Feature = one input **column** of `X`, matching `shapley.h` / `lime.h`.
- `target` selects the model output column being explained; attribution is on the
  **logit**, never a softmax probability (same reason as `lime.h`: probabilities
  saturate).
- Deterministic for a given `seed`; local xorshift64*, never `std::rand()`
  (the `test_adaln_zero` cross-suite RNG entry in `NOT_FIXED.md`).
- Leaves no gradient residue: zero before and after, like `lime_explain`.
- No new learnable parameters → no FD gradient checks are possible or needed. This
  method is inference-time only, which is what the queue means by "trivially
  correct" — the testable content is the *grid* and the *averaging*, both of which
  have closed forms.

## CLOSED-FORM ORACLES (why this is unusually testable)

For a **linear** black box `f(x) = Σ_k c_k x_k` (no bias), PD has an exact answer
that does not depend on `X` at all:

```
PD(z) = (1/N) Σ_i f(x_i with x_j := z) = f(x̄ with x_j := z)
      = c_j·z + Σ_{k≠j} c_k·x̄_k
```

So the PD curve is an **exact line of slope `c_j`**, whose intercept is the mean
response of the dataset with column `j` removed. That single identity pins: the
grid, the replacement, the averaging, and the (absence of) normalization. It is the
same class of oracle as LIME's closed-form `β_j = c_j·σ_j`, which is the reason
this entry was ordered as the cheapest.

ICE adds the per-instance leg: `ICE[i][k] = c_j·grid[k] + Σ_{l≠j} c_l·x[i][l]`,
again exact, and **independently checkable per instance** — an implementation that
averages ICE back to PD before returning cannot pass the `ICE[i]` test unless it
returns the individual curves.

Centered forms, from the reference rule: `PD_c(z) = PD(z) − PD(grid[0])` gives a
curve through the origin, and `ICE_c[i](z) = ICE[i](z) − ICE[i][grid[0]]`.

---

## Tasks

### Task 1: `pd_grid` — percentile grid + low-resolution fallback

**Objective:** Public model-free grid builder.

**Files:**
- Create: `include/nn/interpretability/partial_dependence.h`
- Create: `include/nn/interpretability/partial_dependence.cpp`
- Create: `tests/test_partial_dependence.cpp`

**Step 1: Write failing tests** (`test_pd_grid_*`):
- hand fixture `x = [0,1,2,3,4,5,6,7,8,9]` (N=10, one column),
  `p = (0.05, 0.95)`, `grid_resolution = 3`. numpy's linear-interpolating
  quantile: `lo = 0.05*(9) = 0.45`, `hi = 0.95*9 = 8.55`. So the grid is
  `[0.45, 4.5, 8.55]` — computed INDEPENDENTLY in the test as
  `lo + (hi-lo)*k/(g-1)`, not by calling the impl.
- `grid_resolution = 11` on the same column: uniques = 10 < 11 → grid is the
  **sorted uniques** `[0..9]`, length 10. This is the fallback rule.
- `percentiles = (0,1)` → grid endpoints are `0.0` and `9.0` exactly.
- validation: `X.rows == 0` throws; `feature >= X.cols` throws;
  `p_lo >= p_hi` throws; `p` outside `[0,1]` throws; `grid_resolution <= 1` throws;
  `grid_resolution == 1` throws (reference: `grid_resolution <= 1` is an error).
- degenerate column (all identical values) throws rather than returning a
  zero-width grid — this is the `allclose` branch at :144-148.

**Step 2: Run** `make build/test_partial_dependence` → link/compile error
(`pd_grid` undeclared). Confirm the RED is "symbol missing", not a typo.

**Step 3: Implement `pd_grid`.** Signature:

```cpp
std::vector<double> pd_grid(const Tensor& X, size_t feature,
                            double percentile_lo = 0.05,
                            double percentile_hi = 0.95,
                            size_t grid_resolution = 100);
```

Linear-interpolating quantile (numpy `linear` rule), which for sorted `v` with
`h = q*(n-1)`, `i = floor(h)`, `frac = h-i` is
`v[i]*(1-frac) + v[i+1]*frac` (clamped at the top). Use `std::sort` on a copy —
do not require the caller to pass a sorted column, and do not mutate `X`.

**Step 4: Run** → PASS.

**Step 5: Commit** `feat(interpretability): pd_grid percentile grid + low-resolution fallback`.

### Task 2: `ice_curves` — per-instance responses (model-free)

**Objective:** The per-sample response matrix PD averages over. Model-free so it is
reachable with a caller-supplied response table.

**Files:** modify `partial_dependence.{h,cpp}`, `tests/test_partial_dependence.cpp`

**API:**
```cpp
// Average the ICE matrix over instances -> the PD curve.
// `ice` is (n_instances, n_points). Returns length n_points.
std::vector<double> average_ice(const std::vector<std::vector<double>>& ice);

// Centered variants, matching the reference's `centered=True`.
std::vector<double> center_pd(const std::vector<double>& pd);
void center_ice(std::vector<std::vector<double>>& ice);
```

`average_ice` uses a **uniform** mean (the reference's `np.average(pred, axis=0,
weights=None)` at :305). `center_*` subtract element `[0]`.

**Step 1: Failing tests:**
- hand ICE matrix, `average_ice` == hand-computed column means to 1e-15.
- `center_pd`: `[5,7,9] → [0,2,4]`; `center_ice` on a 2×3 matrix subtracts each
  ROW's own first element (this is the per-instance form).
- **mutation guard:** `center_*` subtracting the MEAN instead of `[0]` must fail —
  asserted by a fixture where `pd[0] != mean(pd)` (e.g. `[1, 100, 100]`).
- validation: empty matrix throws; a ragged matrix throws.

**Step 2-4:** RED → implement → GREEN.

**Step 5: Commit** `feat(interpretability): ICE averaging + reference centering rule`.

### Task 3: `partial_dependence` — the model-facing pass

**Objective:** One call producing grid, PD curve, and ICE curves.

**Files:** modify `partial_dependence.{h,cpp}`, `tests/test_partial_dependence.cpp`

**API:**
```cpp
struct PartialDependenceConfig {
    double percentile_lo = 0.05;
    double percentile_hi = 0.95;
    size_t grid_resolution = 100;
    size_t subsample = 1000;   // ICE curves returned; PD average uses ALL of X
    unsigned seed = 42;
};

struct PartialDependenceResult {
    std::vector<double> grid;                 // length n_points
    std::vector<double> average;              // PD curve, length n_points
    std::vector<std::vector<double>> individual;  // ICE, n_curves x n_points
    size_t feature = 0, target = 0;
    size_t n_points = 0, n_instances = 0, n_curves = 0;
    bool centered = false;
};

PartialDependenceResult partial_dependence(Model& model, const Tensor& X,
                                           size_t feature,
                                           const PartialDependenceConfig& cfg,
                                           size_t target = 0);
```

**The averaging is over ALL of `X`, not the subsample** (`_partial_dependence.py:498-499`).
`subsample` only caps how many ICE curves are stored. This is a real distinction
and gets its own test: with `subsample < X.rows`, `average` must still equal the
mean over all `X` rows.

**Step 1: Failing tests** (the closed-form oracle):
- **Linear-model oracle.** `f(x) = 3x_0 − 2x_1`, `X` = 6 rows with varying column
  1. Assert `pd[k] ≈ 3·grid[k] − 2·mean(X[:,1])` at 1e-9 for every k, and
  `ICE[i][k] ≈ 3·grid[k] − 2·X[i][1]` at 1e-9 for every i, k. Both computed in the
  test from `grid` and `X` directly.
- **Exact-affine slope test** (distinguishes PD from a single-instance curve):
  assert `pd[k+1] − pd[k] == c_j·(grid[k+1] − grid[k])` — the true partial
  dependence of a linear model, not the effect at one instance.
- **Constant-column model:** all-zero weights, bias `b` → `pd[k] == b` for all k
  (proves the grid is actually applied and averaged, not bypassed).
- `centered = true` → `average[0] == 0.0` exactly.
- `subsample` semantics test (above), using a nonlinear model where dropping rows
  would move the average.
- determinism: two calls, same seed → bit-identical `individual`; different seed
  → different subset (when `subsample < X.rows`), same `average`.
- gradient residue: `zero_grad` before/after (mirror `lime_explain`).
- validation: `X.rows == 0`, `feature >= X.cols`, `target >= out.cols`
  (`std::out_of_range`, checked BEFORE any forward so a bad target cannot leave
  gradients half-touched).

**Step 2-4:** RED → implement → GREEN.

**Step 5: Commit** `feat(interpretability): partial_dependence + ICE pass`.

### Task 4: Wire the build, umbrella header, mutation, full suite

**Objective:** registered in `nn.h` + `Makefile`, mutation-verified, no regressions.

**Files:** `include/nn/nn.h`, `Makefile`, `tests/test_partial_dependence.cpp`

- `nn.h`: add `#include "interpretability/partial_dependence.h"` after the LIME
  block, with a one-line comment in the existing style.
- Verify the umbrella compiles STANDALONE (writing-plans §3b):
  `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
- `Makefile`: add the link rule next to `test_lime`'s, add
  `$(BUILD_DIR)/test_partial_dependence` to the `tests:` prerequisite list (the
  `test_mixture_of_softmaxes` bug — missing from the list means never built), and
  add a `run_tests` line after `test_lime`.
- Warning-clean under `-Wall -Wextra`.
- Determinism: 3 reruns identical.
- **Mutation testing** (mandatory, 3 mutations, each must be caught):
  1. `average_ice` divided by `n-1` → the PD oracle must fail (a clean
     `n/(n-1)` factor, not a subtle drift).
  2. `center_*` subtracting the mean instead of element `[0]` → the
     `center_pd([1,100,100])` fixture must fail.
  3. PD averaging over the subsample instead of all of `X` → the subsample test
     must fail.
  Confirm each mutation was actually written to disk (`grep` the mutated string)
  before believing a "0 failures" result — see the SpikingLIF session's silent
  no-op `sed`.
- Full `make run_tests`: compare against the known-failing set in
  `NOT_FIXED.md` (`test_neural_ode`, `test_wgan_gp`, `test_vit`,
  `test_coord_network`, and intermittent `test_adaln_zero`). No NEW failures.

**Step 5: Commit** `chore(build): register test_partial_dependence in tests:, run_tests, link rule`.

### Task 5: Update `EXPANSION_QUEUE.md`

Move the PD/ICE bullet from `## Ideas` to `## Done` with: the corrected citation
(the queue's `arXiv:1409.7002` is a portfolio-optimization paper — say so
explicitly), the closed-form oracle, the test count, and the two semantics read
from source (`percentiles=(0.05,0.95)`, centering by `grid[0]`).

Next to pop then becomes **DeepLIFT** (arXiv:1703.01385), which shares
`input_gradient` from `attribution.h` — the remaining Grad-CAM bullet needs the
repo-wide intermediate-activation-gradient plumbing and stays last.

---

## Risks / things that could bite

- **N=1 batch vacuity.** PD with `X.rows == 1` makes `average == ICE[0]`, so a
  buggy implementation that collapses ICE to PD passes. The oracle tests use
  `X.rows >= 6` and assert the per-instance ICE leg independently.
- **`grid_resolution` vs uniques interaction.** Testing only `grid_resolution = 3`
  on a 10-unique column never exercises the fallback; testing only `grid_resolution
  = 11` never exercises linspace. Both are asserted.
- **Quantile interpolation method.** numpy's default is *linear*; `lower`/`higher`/
  `nearest` give different grids and all look reasonable. The fixture uses
  `N = 10` so that `q=0.05` → `h = 0.45` lands strictly between order statistics
  and the interpolation rule is actually load-bearing.
- **Assuming ICE ≠ PD for a linear model.** For a linear model `PD` and every
  `ICE[i]` are lines with the SAME slope and different intercepts. A test that only
  checks the slope passes an implementation that returned PD for everything.