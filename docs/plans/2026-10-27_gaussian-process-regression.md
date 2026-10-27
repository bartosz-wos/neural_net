# Gaussian Process Regression Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/gaussian_process.{h,cpp}` — exact GP regression: the SE and Matérn kernel matrices, the Cholesky-based posterior mean and predictive variance, the log marginal likelihood, the analytic hyperparameter gradients, and the leave-one-out predictive quantities — plus a numpy oracle, a C++ test suite, and a mutation harness.

**Architecture:** Model-free, exactly like `utils/sinkhorn.h` and `utils/crf.h`: every core is a free function over `Tensor`. Inputs `X` (n, d) and `y` (n, 1); the covariance matrix is built in-process. The numerics are the Cholesky path of GPML Algorithm 2.1 — never a matrix inverse. Hyperparameter gradients use GPML eq. (5.9); LOO quantities use eq. (5.12).

**Tech Stack:** C++17, the repo's `Tensor` class, numpy float64 oracle (`tests/gp_oracle.py`, run with `/usr/bin/python3` which has numpy 2.3.5), a bash mutation harness (`tests/mutate_gp.sh`).

---

## THE PICK, AND WHY IT IS NOT REDUNDANT

**The queue's Ideas list is DRAINED** (the CRF was the fifteenth pop and the list was emptied). The repo's own methodology then applies: verify absence **BY CONTENT**, not by filename.

```
grep -rinE 'gaussian.?process|kernel.?regression|rbf.?kernel|matern|squaredexp|log.?marginal|gp_regress|gp_fit|gp_predict' \
     include/ tests/ --include=*.h --include=*.cpp
-> 0 hits. The only matches in the whole tree are this queue and this plan
   naming the gap.
```

**Redundancy measured, not argued.** The repo ships ~250 layers and 60+ utility files. Nothing computes a *posterior over a function*:

| Shipped file | What it does | Why it is not a GP |
|---|---|---|
| `utils/mmd_loss.h` | RBF/polynomial/IMQ kernel matrices | computes a *discrepancy between two point sets*; no posterior, no predictive distribution, no hyperparameters |
| `utils/gmm.h` | Gaussian mixture model | a *finite* mixture over latent components; a GP is an infinite mixture with a non-diagonal covariance |
| `utils/knn_classifier.h` (layers/architectures) | nearest-neighbour classifier | non-parametric but *deterministic*, no predictive variance, no marginal likelihood |
| `utils/numerical_stability.h` | numerically-safe log-softmax / log-sum-exp | the *building block* for the log-domain marginal likelihood, but not the model |
| `utils/grid_search.h`, `utils/lr_finder.h` | hyperparameter search over an *existing* model's training loop | there is no model to search over — GPs are fitted by marginal likelihood, not by SGD |

**The gap is not "a missing layer" — it is a missing MODEL FAMILY.** The repo has no Bayes. Every shipped uncertainty method (`utils/conformal.h`, `utils/temperature_scaling.h`, `utils/randomized_smoothing.h`, `utils/metrics.h`) is *post-hoc*: it calibrates or bounds a prediction that was already made. A GP is the only one of these that produces the predictive distribution *from the data*, with a fitted noise level and a fitted length-scale.

---

## CITATION: NO arXiv ID MAY BE ASSERTED

The source is:

> C. E. Rasmussen, C. K. I. Williams — **Gaussian Processes for Machine Learning**, the MIT Press, Cambridge, 2006. ISBN 026218253X.

**Verified 2026-10-27 by TITLE search against the arXiv API** (the API itself sanity-checked in the same run: `id_list=1706.04599` returned "On Calibration of Modern Neural Networks", so the endpoint was live):

```
GET http://export.arxiv.org/api/query?search_query=ti:"Gaussian Processes for Machine Learning"
-> totalResults = 2, and BOTH are unrelated papers
   ("Quantum multi-output Gaussian Processes...", 2411.09123;
    "MMGP: a Mesh Morphing Gaussian Process...", 2305.12871)

GET .../api/query?search_query=ti:"Gaussian Processes for Machine Learning" AND au:Williams_C
-> totalResults = 0

https://gaussianprocess.org/gpml/  (the OFFICIAL book page)
-> zero occurrences of the string "arxiv"
```

So: **the GPML book is NOT on arXiv.** The ID `0605074` that circulates for it is **WRONG** — `https://arxiv.org/abs/0605074` returns **HTTP 404**, and `id_list=cs/0605074` resolves to *"SAT Solving for Argument Filterings"*. This is exactly the failure class this repo's queue has hit thirteen times: **a wrong arXiv ID that is a plausible-looking neighbour of the right paper.** The header will assert the MIT Press citation and explicitly state that no arXiv ID exists.

The formulas are read from the primary source PDF (`https://gaussianprocess.org/gpml/chapters/RW.pdf`, 266 pages), extracted to `/tmp/gpml.txt` with pymupdf, and the equation numbers below are the book's own.

---

## THE SPECIFICATION, FROM THE PRIMARY SOURCE

All equation numbers are GPML's own.

### The covariance functions

eq. (2.16), SE with the eq. (2.31) parameterisation:

```
k(xp, xq) = σf² · exp( -½ |xp - xq|² / ℓ² )      (noise-free kernel k_f)
k_y(xp, xq) = k_f(xp, xq) + σn² δpq               (covariance of the NOISY targets)
```

The **length-scale is ℓ, not ℓ²**, and the exponent divides by `ℓ²` — a `ℓ` vs `ℓ²` slip is invisible on a single fixture with ℓ = 1 and is caught immediately by ℓ = 2.

### The posterior (Algorithm 2.1, eqs 2.25 / 2.26 / 2.30)

```
K_y   = K_f + σn² I                       (n, n)
L     = chol(K_y)                         (lower triangular)
α     = L⁻ᵀ L⁻¹ y                         (solve, do NOT invert)
k_*   = (n, 1) vector of k_f(x*, x_i)

mean(x*)    = k_*ᵀ α                                    eq. (2.25)
var(x*)     = k(x*,x*) - vᵀ v,  v = L⁻¹ k_*             eq. (2.26)
log p(y|X)  = -½ yᵀ α - Σ_i log L_ii - (n/2) log 2π    eq. (2.30)
```

**Two sign/coefficient facts that a reimplementation gets wrong:**

1. `α = L⁻ᵀ L⁻¹ y` — the solve is done in **two triangular solves**, and the log-determinant is `2 Σ log L_ii`, so the `-½ log|K|` term is `- Σ log L_ii`. Using `-½ Σ log L_ii` halves the complexity penalty.
2. The predictive **variance** subtracts `vᵀv` from `k(x*,x*)`. For a NOISY test target `y*` the noise variance `σn²` is ADDED to the variance (Algorithm 2.1's note: "to compute the predictive distribution for noisy test data y*, simply add the noise variance σn² to the predictive variance of f*").

### The hyperparameter gradient (eq. 5.9)

```
∂/∂θⱼ log p(y|X, θ) = ½ tr( (ααᵀ - K_y⁻¹) ∂K_y/∂θⱼ )
```

**The `½` and the `tr` are the whole formula.** GPML's own footnote 5 states the computation must be organised as vector-by-matrix first and **only the diagonal terms of the product** — computing the full `(ααᵀ - K⁻¹) ∂K` matrix is O(n³) per hyperparameter and is what makes the naive implementation unusable.

### LOO (eq. 5.12)

```
μ_i  = y_i - [K⁻¹y]_i / [K⁻¹]_ii
σ_i² = 1 / [K⁻¹]_ii
```

**`K` here is the NOISE-FREE `K_f`, not `K_y`** — eq. (5.12) is stated for the noise-free covariance. GPML's own text: "careful inspection reveals that the mean µ_i is in fact independent of y_i as indeed it should be" — a property a suite can pin, because it is an *observable* consequence of the formula.

---

## NUMERICS: WHY CHOLESKY AND NOT AN INVERSE

GPML §A.4 and Algorithm 2.1's own note: the Cholesky factorisation "is faster and numerically more stable". Every solve in this file is a **triangular solve against `L`**; the inverse matrix is never formed. This is not a micro-optimisation:

- `K_y` is **symmetric positive definite** whenever `σn² > 0`, and **singular** when `σn² = 0` and two training points coincide. The Cholesky path *detects* this (a non-positive pivot) and can report it; an explicit inverse would produce `inf`/`nan` silently.
- The log-determinant `log|K_y| = 2 Σ log L_ii` is exact and never overflows, whereas `det(K_y)` underflows for large `n`.

**The failure mode this guards against:** an implementation that forms `K_y⁻¹` explicitly and computes `mean = k_*ᵀ K_y⁻¹ y` is *numerically wrong in a way that a small fixture will not reveal*, and the O(n³) inverse is also 3× the work of two triangular solves.

---

## WHAT THE SUITE MUST PIN, AND WHY EACH ITEM IS NON-VACUOUS

Every fixture below is chosen so that the property it tests **cannot** be satisfied by a wrong implementation. The repo's recurring lesson: *a fixture that does not vary in the dimension the test is about cannot test it.*

| # | Property | Why a naive implementation fails it |
|---|---|---|
| P1 | `mean(x*)` at a **training point** with `σn² = 0` reproduces `y` exactly | a solver that forgets the noise term in `K_y` does not interpolate |
| P2 | The **interpolation** property: `σn² = 0` ⇒ mean passes through every training target | an off-by-one in the noise (`σn` vs `σn²`) breaks it |
| P3 | The **noise-free posterior variance at a training point is ~0** (not `σf²`) | the classic mistake is to return `k(x*,x*)` without the `vᵀv` subtraction |
| P4 | The **predictive variance at a far-away test point** approaches `σf²` | a solver that returns the variance of `y` instead of `f` saturates at `σf² + σn²` |
| P5 | **Adding `σn²` to the predictive variance** for a noisy target changes the answer by exactly `σn²` | pins the "noisy vs noise-free" distinction of Algorithm 2.1's note |
| P6 | The **log marginal likelihood matches a direct multivariate-normal density** computed by `numpy.linalg.slogdet` | an independent route to the same number — a hand-rolled `log|K|` via `det` is caught |
| P7 | The **hyperparameter gradient matches a central finite difference** of the log marginal likelihood | the gradient is the part most likely to be wrong by a factor or a sign |
| P8 | The **hyperparameter gradient matches `0.5*trace((ααᵀ - K⁻¹) dK/dθ)`** computed independently | pins the `½` and the `tr` |
| P9 | The **length-scale derivative has the right SIGN**: a larger ℓ fits smoother data | a sign flip in `∂k/∂ℓ` is invisible to a magnitude check |
| P10 | The **LOO mean is independent of `y_i`** (eq. 5.12) | a LOO implementation that refits on the full data fails this; it is an observable consequence of the formula |
| P11 | The **LOO variance is `1/[K_f⁻¹]_ii`** and is independent of `σn` | pins that LOO uses `K_f`, not `K_y` |
| P12 | The **marginal likelihood is maximised at the true hyperparameters** on a fixture generated from known hyperparameters | the end-to-end statement that the whole thing is a *fit*, not a formula |
| P13 | The **noise variance → 0 limit** reproduces the noise-free case | the `σn` vs `σn²` boundary |
| P14 | **Singular `K_y`** (duplicate training points, `σn = 0`) is **detected and reported**, not silently `nan` | the reason for the Cholesky path |
| P15 | **`σn = 0` with distinct points** is legal and interpolates | guards against a solver that requires `σn > 0` |

**P7 is the one that needs care.** A central difference of the marginal likelihood in `ℓ` is fine — the forward is **deterministic** — but the perturbation must be large enough to clear the double-precision noise floor. The repo's own rule (systematic-debugging §5c): *a fixture whose discriminating value is only correct to within a rounding error does not discriminate anything.* The oracle measures the usable ε.

**P7 does NOT need an FD check on the predictive mean.** The mean is *linear* in `y` and its dependence on `ℓ` is smooth, so FD works there too — but the **hyperparameter gradient of the marginal likelihood** is the one that matters for fitting, and that is what P7/P8 pin.

---

## MUTATION TESTING: THE PLAN

The harness is `tests/mutate_gp.sh`, following `tests/mutate_crf.sh` / `tests/mutate_conformal.sh`. Its rules, from the repo's own hard-won lessons:

1. `restore()` **NEVER deletes the backup**; cleanup happens exactly once, on EXIT.
2. The link set is obtained from **`make print-lib-objs`** (already in the Makefile for exactly this reason) — never from a regex over the Makefile, which yields zero objects and turns every mutation into a false verdict. **Assert the object count ≥ 10 before reporting any verdict.**
3. The mutated object is **excluded from the link set** — a link error is not a test failure.
4. Every mutation is **checksummed before and after** the edit; a no-op edit is a harness defect, not a verdict.
5. Every anchor is **counted** before editing; an ambiguous anchor (count > 1) is a refusal, not a coin flip. Re-check **all** existing anchors after adding a feature.
6. A mutation that makes the binary **crash** (`rc >= 128`) is a THIRD verdict class — not "CAUGHT".
7. `MUT-BROKEN` (build failure) is neither "caught" nor "bad"; it is re-run in isolation before being believed.
8. **No backticks, no `$(...)`, no `!`** in double-quoted harness strings.
9. The harness **must not run concurrently** with a `make` — they share `build/*.o`.

**Planned mutations (each with the property it is supposed to kill):**

| ID | Mutation | Kills |
|---|---|---|
| M1 | `log|K_y|` term `-Σ log L_ii` → `-½ Σ log L_ii` | the factor-2 in the complexity penalty (P6) |
| M2 | `α = L⁻ᵀL⁻¹y` → `L⁻¹y` (drop one triangular solve) | the predictive mean (P1, P2) |
| M3 | predictive variance: drop the `- vᵀv` term | P3, P4 |
| M4 | `K_y = K_f + σn²I` → `K_f + σnI` | P1, P2, P13 |
| M5 | `K_y = K_f + σn²I` → `K_f` (no noise) | P1, P2, P15 |
| M6 | SE exponent `/(ℓ²)` → `/ℓ` | P7, P9 |
| M7 | `σf²` → `σf` in the kernel amplitude | P7, P8 |
| M8 | hyperparameter gradient: drop the `½` | P8 |
| M9 | hyperparameter gradient: `tr` → `sum of all entries` | P8 |
| M10 | hyperparameter gradient: `ααᵀ - K⁻¹` → `ααᵀ` (drop the `-K⁻¹` term) | P8 |
| M11 | LOO: use `K_y` instead of `K_f` | P11 |
| M12 | LOO: `σ_i² = 1/[K⁻¹]_ii` → `1/[K⁻¹]_ii` without the `K_f` swap, or drop the `y_i` term | P10, P11 |
| M13 | the `½ log 2π` constant term dropped | P6 |
| M14 | the singular-Cholesky detection removed (return `nan` silently) | P14 |
| M15 | the `σn²` addition to the noisy-target predictive variance dropped | P5 |

---

## TASKS

### Task 1: Write the numpy oracle

**Files:** Create `tests/gp_oracle.py`

**Step 1: Write the oracle**

It must:
- build the SE and Matérn kernel matrices;
- compute the posterior mean, predictive variance and log marginal likelihood via `numpy.linalg.cholesky` / `slogdet` / `solve`;
- compute the hyperparameter gradient by (a) central finite differences and (b) the closed form `0.5*trace((ααᵀ - K⁻¹) ∂K/∂θ)`;
- compute the LOO quantities from eq. (5.12);
- **self-check every section** (assert, not just print) — a self-check that prints a verdict it never tested is worse than none;
- print the numbers the C++ suite will assert against.

**Step 2: Run it**

Run: `/usr/bin/python3 tests/gp_oracle.py`
Expected: exit 0, every section green.

**Step 3: Commit**

```bash
git add tests/gp_oracle.py
git commit -m "test(utils): numpy oracle for Gaussian process regression"
```

### Task 2: Write the plan (this file) — DONE

**Files:** Create `docs/plans/2026-10-27_gaussian-process-regression.md`

### Task 3: Write the failing C++ tests (RED)

**Files:** Create `tests/test_gp.cpp`

Write the 15 properties P1–P15 above, plus the argument-validation tests (empty `X`, mismatched `y` rows, `ℓ <= 0`, `σf <= 0`, `σn < 0`, non-finite inputs) — each with a **distinct** message, because a substring shared by two messages discriminates nothing.

**Step 1: Build and verify RED**

Run: `make build/test_gp 2>&1 | tail -5`
Expected: **compile error** — `utils/gaussian_process.h` does not exist. This is the RED state: the test cannot even build.

**Step 2: Commit**

```bash
git add tests/test_gp.cpp
git commit -m "test(utils): Gaussian process regression tests (RED)"
```

### Task 4: Implement the header and source (GREEN)

**Files:** Create `include/nn/utils/gaussian_process.h`, `include/nn/utils/gaussian_process.cpp`; Modify `include/nn/nn.h`

The header states the semantics, the preconditions, the throws, and the out-of-scope list, in the style of `utils/crf.h` / `utils/sinkhorn.h`. The implementation is Cholesky-only.

**Step 1: Add the umbrella include**

`include/nn/nn.h`, next to the other utils includes.

**Step 2: Build**

Run: `make build/test_gp`
Expected: compiles clean at `-O2 -march=native -Wall -Wextra`, **zero warnings**.

**Step 3: Run the suite**

Run: `./build/test_gp`
Expected: **all pass**.

**Step 4: Verify the umbrella header compiles standalone**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: exit 0, no output.

**Step 5: Commit**

```bash
git add include/nn/utils/gaussian_process.h include/nn/utils/gaussian_process.cpp include/nn/nn.h
git commit -m "feat(utils): Gaussian process regression (exact posterior, marginal likelihood, hyperparameter gradients, LOO)"
```

### Task 5: Register the suite in the build

**Files:** Modify `Makefile`

Three edits, all of them:
- the link rule `$(BUILD_DIR)/test_gp: $(LIB_OBJS) $(BUILD_DIR)/test_gp.o`;
- `$(BUILD_DIR)/test_gp` in the `tests:` prerequisite list;
- the `run_tests` line.

**Step 1: Build and run the full suite**

Run: `make -j$(nproc) run_tests`
Expected: 265 suites; the aggregate run's only failure is the pre-existing `test_neural_ode` (the ODERNN entry already deferred in `NOT_FIXED.md`).

**Step 2: Commit**

```bash
git add Makefile
git commit -m "chore: register the Gaussian process suite in the build"
```

### Task 6: Mutation harness

**Files:** Create `tests/mutate_gp.sh`

**Step 1: Write the harness** following the nine rules above.

**Step 2: Run it**

Run: `bash tests/mutate_gp.sh`
Expected: every mutation is CAUGHT, with zero SURVIVED / zero BROKEN / zero NOOP.

**Step 3: If a mutation SURVIVES**, it is a **diagnostic about the fixture**, not a pass: ask what dimension the mutation targets and whether any fixture VARIES in that dimension. Do not silently drop it.

**Step 4: Commit**

```bash
git add tests/mutate_gp.sh
git commit -m "test(utils): mutation coverage for Gaussian process regression"
```

### Task 7: Sanitizers and the full-suite re-baseline

**Step 1: ASan + UBSan**

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
    include/nn/utils/gaussian_process.cpp tests/test_gp.cpp -o /tmp/test_gp_san
/tmp/test_gp_san
```
Expected: clean, no sanitizer output.

**Step 2: Full suite**

Run: `make -j$(nproc) run_tests`
Expected: no new failures beyond the pre-existing `test_neural_ode`.

### Task 8: Update the queue

**Files:** Modify `EXPANSION_QUEUE.md`

Move the entry from `## Ideas` to `## Done` with a one-line summary and the test results, in the style of the CRF entry.

---

## OUT OF SCOPE FOR v1, stated so it cannot creep in

- **Sparse / variational GPs** (Titsias, Hensman; Wilson et al. "GPyTorch", arXiv:1810.00303) — a different algorithm with a different cost.
- **Multi-output / multi-task GPs** — a different model.
- **Non-Gaussian likelihoods** (Bernoulli / Bernoulli-probit GPs) — a different model.
- **Matern kernels** are IN scope; the rational-exponential and other kernels are not.
- **GP classification** (Margossian et al., arXiv:1710.09430) — a different deliverable.
- GPU / blocked computation.
- Any change to `utils/mmd_loss.h` (it has its own RBF; the two must not be conflated — the MMD kernel has no `σf²` amplitude and no length-scale in the GP parameterisation).
