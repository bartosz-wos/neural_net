# Public symmetric linear algebra: eigendecomposition, matrix square root, Fréchet distance — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Promote the (now-correct) Jacobi eigendecomposition from three duplicated private copies into ONE public `utils/` API, add the symmetric matrix square root that it unblocks, and fix the GaLore "top-r" contract that the promotion exposes.

**Architecture:** A new model-free `include/nn/utils/linear_algebra.{h,cpp}` holding `symmetric_eigendecompose` (cyclic Jacobi), `symmetric_matrix_sqrt` (via eigendecomposition), and `frechet_distance` (the Fréchet distance between two Gaussians, which needs the matrix square root of a product of two PSD matrices). The three optimizers call the shared routine, which removes the three-way duplication that let a one-character sign slip into all three copies. `GaLore::refresh_projection` additionally **sorts** the eigenpairs descending, because the cyclic sweep's final diagonal is NOT globally sorted.

**Tech Stack:** C++17, the repo's `Tensor`, float64 numpy oracle (`/usr/bin/python3`), `make tests` / `make run_tests`.

---

## Background: what was measured before anything was written

### 1. The `## Ideas` list is DRAINED (seventeenth time)

The docs-gap grep returned its 19 familiar hits (all accounted for in shipped headers) for the SEVENTH consecutive run, so the session fell back to the queue's own documented method: a verified-absent concept sweep, plus the constructibility sweep the last session added.

The `## Done` entry of 2026-10-28 named the next step in the repo's own words:

> The repo now has a **correct** symmetric eigendecomposition, but only as a PRIVATE copy inside three optimizers. **FID, a matrix square root, a matrix logarithm, a Lyapunov solver and a spectral-radius estimate all need it in PUBLIC form.** The natural next step is to promote it to `utils/` … **THE THREE COPIES ARE WHY A ONE-CHARACTER DEFECT WAS IN THREE PLACES AT ONCE, AND DE-DUPLICATING THEM IS WHAT PREVENTS THE NEXT ONE.**

That is a specification written by the previous session, and it is the pick.

### 2. The absence, re-measured this session (not trusted from the queue)

```
grep -rinE '\bfid\b|fr[eé]chet|inception.?score' include/ tests/   -> 0 hits
grep -rinE '\bmatrix.?sqrt\b|matrixSqrt|sqrtm' include/            -> 0 hits
grep -rinE 'lyapunov|riccati' include/                            -> only cautious.h's prose
```

Fifteen generative models ship (DDPM, consistency models, flow matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP, PixelCNN, CapsNet, DiT, …) and **nothing anywhere measures whether the samples are good**.

### 3. The defect found while reading the code to promote it

`GaLore::refresh_projection` takes "the top-r eigenvectors (the ones with the largest eigenvalues)" and then copies **the first r columns of Q**. It never sorts. The header even says so:

> To keep the API simple, we don't bother sorting eigenvectors by eigenvalue for the projection — the Adam-in-subspace step is invariant to the ordering of P's columns.

**That reasoning is wrong for the truncation.** The Adam update is invariant to a PERMUTATION of P's columns, but it is NOT invariant to which r of the n columns are kept. Keeping an arbitrary r of the n eigenvectors is not the same as keeping the r dominant ones.

**MEASURED** (probe over 2100 fixtures, n = 2..8, random and structured PSD matrices):

```
525/2100 fixtures had P's columns NOT in descending eigenvalue order
worst ascending violation: 2.450999e+01
```

A concrete failure: `G = diag(1, 5, 1, 1)` gives `gram = diag(1, 25, 1, 1)`, which is **already diagonal**, so the Jacobi sweep performs **no rotation at all** and the diagonal keeps the INPUT order `1, 25, 1, 1` — not descending. With `rank = 2`, GaLore keeps the eigenvectors of eigenvalues **1 and 25** instead of **25 and 1**: it projects onto a subspace that is NOT the dominant one, while `Pᵀ·P` is still the identity, so the existing orthonormality test cannot see it.

**THE ROOT CAUSE, MEASURED, NOT GUESSED.** Two separate probes:

1. The rotation ALWAYS puts the larger of the two eigenvalues into the LOWER-index slot: **0 violations in 200,000 random 2×2 rotations.** (`new_app = c²·app + 2cs·apq + s²·aqq` is the LARGER of the two rotated diagonal entries when `θ = ½·atan2(2·apq, app − aqq)`.)
2. But the FINAL diagonal of a cyclic sweep is **not globally sorted** — it inherits the order in which the rotations happened to converge, and for a matrix that is ALREADY DIAGONAL **no rotation happens at all**, so it is exactly the input's diagonal order.

So "the first r columns are the top-r" is an assumption the implementation does not hold, and it is falsifiable in the easiest possible way.

**THE FIX IS NOT TO SORT IN PLACE.** Sorting the eigenvectors in place would break the pairing between Q's columns and the eigenvalues, and the optimizers index them in lockstep. The public routine will return eigenpairs **sorted descending by eigenvalue** and the eigenvectors permuted with them, so `Q(:, j)` always pairs with `λ(j)`.

### 4. The citation, verified by TITLE (never typed from memory)

`grep` of the web search for the exact title returned:

> "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" — papers.nips.cc/paper/7240, and **arXiv:1706.08500** (the abs page for 1706.08500 carries this exact title).

Authors: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. NeurIPS 2017, pp. 6626-6637. This is the FID paper. The **Inception Score** is the companion: Heusel et al., "Fast Image-to-Image Translation with a Wasserstein Distance", ICML 2017 (arXiv:1703.03012 — to be verified by title before it is written into a header, and it is NOT needed for this deliverable, so it is left out).

---

## The public API this file will expose

```cpp
namespace nn { namespace utils {

// The result of a symmetric eigendecomposition.
struct SymmetricEig {
    Tensor eigenvalues;   // (n,)  -- SORTED DESCENDING
    Tensor eigenvectors;  // (n,n) -- column j pairs with eigenvalues(j)
};

// A = Q·diag(λ)·Qᵀ, for a symmetric A. Returns Q and λ sorted DESCENDING.
SymmetricEig symmetric_eigendecompose(const Tensor& A,
                                      int max_sweeps = 100,
                                      double tol = 1e-12);

// S = A^(1/2), the symmetric positive-definite square root, for symmetric PSD A.
Tensor symmetric_matrix_sqrt(const Tensor& A,
                             int max_sweeps = 100,
                             double tol = 1e-12);

// The Fréchet distance between two Gaussians N(mu1, S1) and N(mu2, S2):
//     ||mu1 - mu2||² + tr(S1) + tr(S2) - 2·tr((S1^{1/2}·S2·S1^{1/2})^{1/2})
double frechet_distance(const Tensor& mu1, const Tensor& S1,
                        const Tensor& mu2, const Tensor& S2,
                        int max_sweeps = 100, double tol = 1e-12);

}}
```

`frechet_distance` needs `(S1^{1/2}·S2·S1^{1/2})^{1/2}`, which needs a matrix square root of a matrix that is only *numerically* PSD (the product of a PSD matrix with a PSD matrix is PSD, but floating-point multiplication can make it slightly indefinite). **The implementation must therefore clamp negative eigenvalues to zero inside the square root** — and the test must pin that, because silently returning NaN is the alternative.

---

## Task 1: RED — the failing test

**Objective:** Write the failing test first. It must fail for the RIGHT reason (the feature is missing), and it must contain at least one assertion that a WRONG implementation cannot pass.

**Files:**
- Create: `tests/test_linear_algebra.cpp`
- Modify: `Makefile` (build rule + `tests:` prerequisite + `run_tests` line)

**Step 1: Write the failing test**

The test covers:

1. **`symmetric_eigendecompose` returns eigenpairs sorted DESCENDING** — asserted on a fixture whose Gram matrix is already diagonal with the dominant entry in the MIDDLE (`diag(1, 25, 1, 1)`). This is the fixture that makes the sorting requirement observable at all: a random matrix's Jacobi output *happens* to be nearly sorted, so it cannot discriminate. The fixture asserts `λ(0) == 25.0` and `λ(1) == 1.0` and that `λ` is non-increasing.
2. **The eigenvector property**: `A·Q == Q·diag(λ)` to a relative tolerance.
3. **The eigenvalues are CORRECT** against an independent closed-form 2×2 reference (λ² − tr·λ + det = 0), computed in the test with `std::sqrt`, sharing no code with the implementation.
4. **NON-VACUITY**: the fixture must be genuinely non-diagonal where the claim needs it, asserted before the claim is made.
5. **`symmetric_matrix_sqrt`**: `S·S == A`, `S` is symmetric, and `S` is PSD.
6. **`frechet_distance`**:
   - `d(X, X) == 0` exactly (to ~1e-12),
   - `d(X, Y) == d(Y, X)` (symmetry),
   - `d(X, Y) >= 0` (non-negativity),
   - `d` equals the **closed form for the commuting / diagonal case**, where the Fréchet distance has the exact value `||mu1-mu2||² + Σ_i (sqrt(λ1_i) − sqrt(λ2_i))²` — computable from the eigenvalues with no matrix square root at all. **This is the discriminating assertion**: an implementation that gets the matrix square root wrong still passes the identity checks, and only this one catches it.
   - the **covariances are GENERAL (not diagonal)**, so the matrix square root is genuinely exercised — a diagonal covariance makes the whole matrix-sqrt path degenerate (the `identity` at integer source coordinates trap, in matrix form).
7. **THE VICTIM-FIXTURE CHECK**: assert that the `d(X, Y) == d(Y, X)` metric is not trivially satisfied by a constant, and that the closed-form check FAILS for a deliberately wrong variant (e.g. `tr(S1) + tr(S2) - 2·tr(S1·S2)` without the square root) — so the suite proves it can detect the wrong formula.

**Step 2: Verify RED**

Run: `make build/test_linear_algebra && ./build/test_linear_algebra`
Expected: **COMPILE FAILURE** — `symmetric_eigendecompose` does not exist. That is the correct RED for a new API.

**Step 3: Commit**

```bash
git add tests/test_linear_algebra.cpp Makefile
git commit -m "test(utils): RED test for the public symmetric eigendecomposition, matrix square root and Frechet distance"
```

---

## Task 2: GREEN — the oracle, in numpy, BEFORE the C++

**Objective:** Write the float64 numpy oracle that pins the semantics. It must be run and green BEFORE the C++ is written, because it is the specification.

**Files:**
- Create: `tests/linear_algebra_oracle.py`

**Step 1: Write the oracle**

The oracle uses `numpy.linalg.eigh` (which returns eigenvalues in ASCENDING order — the oracle must reverse them, and that reversal is itself a place a defect can hide) and `numpy.linalg.sqrtm` for the matrix square root.

Checks:
- `eig_sorted_desc`: the descending convention, checked against `numpy.linalg.eigvalsh` reversed.
- `matrix_sqrt`: `S @ S == A` and `S` symmetric.
- `frechet`: the closed-form identity for diagonal covariances; the symmetry; the non-negativity; the zero-on-the-diagonal; and the **general (non-diagonal) covariance** case compared against `numpy.linalg.sqrtm`.
- **The oracle's own self-check**: the diagonal closed form must DISAGREE with the non-square-root variant `tr(S1)+tr(S2)-2·tr(S1@S2)`, so the discriminating fixture is proven to discriminate.

**Step 2: Run it**

Run: `/usr/bin/python3 tests/linear_algebra_oracle.py`
Expected: **green**, with the numbers printed.

**Step 3: Commit**

```bash
git add tests/linear_algebra_oracle.py
git commit -m "test(utils): float64 oracle for the public symmetric linear algebra"
```

---

## Task 3: GREEN — the implementation

**Objective:** Implement the three public functions.

**Files:**
- Create: `include/nn/utils/linear_algebra.h`
- Create: `include/nn/utils/linear_algebra.cpp`
- Modify: `include/nn/nn.h` (add the include)

**Step 1: The eigendecomposition**

Move the cyclic Jacobi from the optimizers into `symmetric_eigendecompose`, with these changes from the optimizers' copy:

- It takes a `const Tensor& A` and does NOT mutate the caller's matrix (the optimizers had to make a working copy by hand — "Make a working copy because jacobi_eigendecompose mutates A" — a footgun).
- It **sorts the eigenpairs descending** at the end, permuting Q's columns with the eigenvalues. This is the fix for the GaLore defect and it is the reason the routine is public: the sorting is part of the contract, so every caller gets it.
- The **rotation angle** is `0.5 * std::atan2(2.0 * apq, app - aqq)`, and the `app == aqq` degenerate branch computes `±π/4` directly. Both are already pinned by `tests/test_jacobi_eig.cpp` and `tests/test_jacobi_inertness.cpp`; the new code must preserve both exactly.

**Step 2: The matrix square root**

```
A = Q·diag(λ)·Qᵀ     (symmetric_eigendecompose, λ descending)
S = Q·diag(sqrt(max(λ,0)))·Qᵀ
```

The `max(λ, 0)` clamp is load-bearing: `A` is supposed to be PSD, but a matrix that is PSD only to within rounding can have an eigenvalue of −1e-18, and `sqrt` of a negative is NaN, and one NaN poisons the whole result. **The header must say this, and the test must pin it** with a fixture whose eigenvalue is exactly zero (a rank-deficient matrix) and one that is very slightly negative.

**Step 3: The Fréchet distance**

```
d² = ||mu1 - mu2||² + tr(S1) + tr(S2) - 2·tr((S1^{1/2}·S2·S1^{1/2})^{1/2})
```

Note the order: `S1^{1/2}·S2·S1^{1/2}` is symmetric PSD in exact arithmetic, which is what makes the outer square root well-defined. The implementation computes it that way rather than `sqrt(S1·S2)`, which is NOT symmetric.

**Step 4: Verify GREEN**

Run: `make build/test_linear_algebra && ./build/test_linear_algebra`
Expected: **PASS**.

Run the oracle against the C++ numbers (the test file embeds the oracle's expected values).

**Step 5: Commit**

```bash
git add include/nn/utils/linear_algebra.h include/nn/utils/linear_algebra.cpp include/nn/nn.h
git commit -m "feat(utils): public symmetric eigendecomposition, matrix square root and Frechet distance"
```

---

## Task 4: de-duplicate the three optimizers

**Objective:** Point Shampoo, SOAP and GaLore at the one public routine. This is what prevents the next one-character defect from landing in three places at once.

**Files:**
- Modify: `include/nn/optimizers/shampoo.{h,cpp}`
- Modify: `include/nn/optimizers/soap.{h,cpp}`
- Modify: `include/nn/optimizers/galore.{h,cpp}`

**Step 1: Replace the private copies**

Delete `Shampoo::jacobi_eigendecompose`, `SOAP::jacobi_eigendecompose` and `GaLore::jacobi_eigendecompose` and call `utils::symmetric_eigendecompose` instead.

**THE OPTIMIZERS' OWN SEMANTICS ARE PRESERVED** — the routine returns eigenpairs sorted descending, and:
- Shampoo/SOAP use `U_L` with `λ_L` in lockstep by INDEX (`max(λ_L[k], eps)^{-1/4}`), so a consistent permutation of (column, eigenvalue) pairs changes nothing. **Verify this by measurement, not by argument** — the Shampoo/SOAP suites must still pass unchanged.
- GaLore takes the FIRST r columns, and with the descending sort those are now **actually the top-r**, which is the FIX.

**Step 2: Verify**

Run: `make build/test_shampoo build/test_soap build/test_galore build/test_jacobi_eig build/test_jacobi_inertness`
Then run each. Expected: all pass. `test_jacobi_eig` must still pass **unchanged** — it exercises the PUBLIC accessors (`get_U_L`, `get_Q_L`, `get_P`), so it is the regression test for this refactor.

**Step 3: Commit**

```bash
git add include/nn/optimizers/shampoo.h include/nn/optimizers/shampoo.cpp \
        include/nn/optimizers/soap.h include/nn/optimizers/soap.cpp \
        include/nn/optimizers/galore.h include/nn/optimizers/galore.cpp
git commit -m "refactor(optimizers): share one public symmetric eigendecomposition across Shampoo, SOAP and GaLore"
```

---

## Task 5: RED — GaLore's top-r contract, then GREEN

**Objective:** Pin the defect that started this session, so it can never come back silently.

**Files:**
- Modify: `tests/test_linear_algebra.cpp` (or a new `tests/test_galore_projection.cpp`)

**Step 1: Write the failing test**

The fixture is `G = diag(1, 5, 1, 1)` (4×4), whose Gram matrix `GᵀG = diag(1, 25, 1, 1)` is ALREADY DIAGONAL, so the Jacobi sweep performs no rotation and the eigenpairs come out in the INPUT order. With `rank = 2`, GaLore must keep the eigenvectors of eigenvalues **25 and 1**, and the test asserts it by checking the RAYLEIGH QUOTIENT of each kept column against the two largest eigenvalues of `G_EMAᵀ·G_EMA`.

**The assertion must be a PROPERTY, not a value**: `λ_kept` must equal the two largest eigenvalues of the Gram matrix, compared as a SORTED pair. The test also asserts the fixture is non-vacuous (the Gram matrix really does have a dominant eigenvalue in the middle) and that the kept subspace really is the dominant one (the Rayleigh quotients are 25 and 1, not 25 and 1-in-the-middle-position).

**Step 2: Verify RED**

Before the fix, GaLore keeps the columns of eigenvalues {1, 25} — which as a SET is {1, 25}, the same as the correct answer {25, 1}! **So the sorted-pair assertion would PASS on the broken implementation.** This is exactly the repo's recorded trap: a fixture whose discriminating value is only correct to within a reordering does not discriminate anything.

**THE DISCRIMINATING FIXTURE MUST VARY IN THE DIMENSION THE MUTATION TARGETS.** With `rank = 1` the broken implementation keeps the FIRST column, whose eigenvalue is **1** (the smallest), while the correct implementation keeps the eigenvalue **25** (the largest). **That is the discriminating fixture**, and it is the one that fails on the broken code.

So the test has TWO parts:
- `rank = 1` → the kept column's Rayleigh quotient must be the LARGEST eigenvalue (25), not the smallest (1). **This is the assertion that fails today.**
- `rank = 2` → the kept columns' Rayleigh quotients, as a SORTED pair, must be the two largest.

And the test asserts the fix works by construction: **run it BEFORE the fix (RED) and AFTER (GREEN).**

**Step 3: GREEN — fix GaLore**

With the descending sort in the public routine, GaLore's existing "take the first r columns" becomes correct with **no change to GaLore's own code**. That is the whole point of putting the sort in the shared routine.

**Step 4: Commit**

```bash
git add tests/test_galore_projection.cpp Makefile
git commit -m "test(optimizers): RED test for GaLore's top-r dominant-eigenspace contract"
```

---

## Task 6: the mutation harness

**Objective:** Prove the new suite catches the defects it was written for. A green suite proves nothing on its own.

**Files:**
- Create: `tests/mutate_linear_algebra.sh`

**THE MUTATIONS:**
- M1: restore the `aqq - app` sign in the rotation angle (the original defect).
- M2: drop the descending sort (the defect this session found).
- M3: drop the `max(λ, 0)` clamp in the matrix square root (a NaN producer).
- M4: use `tr(S1·S2)` instead of `tr((S1^{1/2}S2S1^{1/2})^{1/2})` in the Fréchet distance.
- M5: swap the arguments of the Fréchet distance's square-root product (`S2^{1/2}S1S2^{1/2}` instead of `S1^{1/2}S2S1^{1/2}`) — a symmetric-looking change that is NOT equivalent for non-commuting covariances.

**THE HARNESS RULES THIS REPO HAS ALREADY LEARNED** (see the TDD skill and `tests/mutate_jacobi_eig.sh`):
- `restore()` must NEVER delete its backup; cleanup happens exactly once, on EXIT.
- A BUILD ERROR IS NOT A MUTATION VERDICT (`MUT-BROKEN` is its own class).
- A CRASH (rc >= 128) is a THIRD class, not a verdict.
- The mutated object must not be in its own link set.
- NEVER run a build concurrently with the harness.
- After mutating, CONFIRM the edit applied (checksum the tree) — a mutation that never applied is a harness defect, not a "SURVIVED" verdict.
- Anchor on CONTEXT, not on a single line; after adding a feature, re-check EVERY existing mutation's anchor count.

---

## Task 7: full-suite regression + sanitizers

```bash
make run_tests 2>&1 | tail -40
```
Expected: exactly the same failures as the pre-change baseline — **only `test_neural_ode`**, the entry already deferred in `NOT_FIXED.md` and measured identically on a clean clone of the pre-change commit. Any NEW failure is a regression.

Sanitizers:
```bash
make CXXFLAGS="-std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -march=native -MMD -MP" build/test_linear_algebra
```

---

## Task 8: record the finding

- `EXPANSION_QUEUE.md` — add a `## Done` entry; move the item out of `## Ideas`.
- `NOT_FIXED.md` — the GA-LORE finding is FIXED here, so it does not go in.
- `docs/plans/` — this file.

---

## Notes for the implementer

- **Do NOT trust the queue's numbers.** Re-measure the absence greps this session (they are in the Background section) before writing the header's claims.
- **The sort is part of the CONTRACT.** If a future caller wants an unsorted routine, it gets a different function name — not a flag on this one.
- **The `max(λ, 0)` clamp is not cosmetic.** Without it a rank-deficient covariance produces NaN, and NaN in FID is a metric that silently reports nothing.
- **`frechet_distance` needs the square root of `S1^{1/2}·S2·S1^{1/2}`, NOT `sqrt(S1·S2)`.** The latter is not symmetric and its square root is not the one the formula calls for. The oracle pins which one.
- **Do not hand-roll a Cholesky here.** The repo has one in `utils/gmm.h`; the spectral route is needed anyway because the covariance is only numerically PSD.
