# Sample-Quality Metrics (FID + Inception Score) — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship the two canonical sample-quality metrics for generative models — the **Fréchet Inception Distance** and the **Inception Score** — as model-free utilities over `Tensor`, together with the **sample Gaussian estimator** they are built on (sample mean + sample covariance), which the repo does not have at all.

**Architecture:** a new `include/nn/utils/sample_quality.{h,cpp}` in the same shape as `utils/sliced_wasserstein.h`, `utils/rank_correlation.h` and `utils/linear_algebra.h` — every core is a free function over `Tensor`, no `Layer`, no `Model`, no randomness, no optimizer. The FID path **composes** the already-shipped `utils::frechet_distance` rather than re-implementing it, so there is exactly one copy of the Gaussian-to-Gaussian distance in the repo.

**Tech Stack:** C++17, the repo's `Tensor`, the project's `-O2 -march=native -Wall -Wextra` build, a float64 numpy+scipy oracle in `tests/`, an oracle-mutation harness in `tests/`, a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

The `## Ideas` list was drained for the 27th time (every bullet is marked `[DONE]`), so the session fell back to the queue's own documented method: a **verified-absent concept sweep**.

```
grep -rinE '\bfid\b|inception' include/ tests/ --include=*.h --include=*.cpp
-> include/nn/utils/linear_algebra.h:16,114,116,124,125   (all PROSE)
-> include/nn/utils/sliced_wasserstein.h:28,29,30          (all PROSE)
-> include/nn/utils/linear_algebra.cpp:11,259,274,278      (all PROSE)
-> include/nn/utils/rank_correlation.h:31, include/nn/utils/dtw.h:28  (PROSE)
-> tests/test_sliced_wasserstein.cpp:28,29                  (PROSE)
-> tests/test_linear_algebra.cpp:37,51                      (PROSE)
```

**ZERO IMPLEMENTATION.** And the absence is not merely a gap — **two shipped headers name it as the missing piece in their own words**, which is the strongest possible evidence because it is *about this repo*:

| shipped | what it says |
|---|---|
| `utils/linear_algebra.h:22-26` | "FIFTEEN generative models ship in this repo -- DDPM, consistency models, flow matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP, PixelCNN, CapsNet, DiT and more -- and **NOTHING ANYWHERE MEASURES WHETHER THE SAMPLES ARE GOOD**." |
| `utils/linear_algebra.h:123-128` | "It does NOT implement FID itself: FID additionally needs an Inception network's pool3 features, which this repo does not ship and which is a model, not a utility. **What this file provides is the piece that was genuinely missing and genuinely reusable.**" |
| `utils/linear_algebra.h:116-121` | "The companion Inception Score paper … is deliberately NOT cited here: **this file does not implement the Inception Score**." |
| `utils/sliced_wasserstein.h:28-30` | "`utils/linear_algebra.h` -- the Frechet distance, the FID building block, which additionally needs an Inception network's pool3 features. **This repo ships no Inception network**, so FID cannot be [computed]." |

### It is not redundant with the shipped neighbours

- `utils/linear_algebra.h` `frechet_distance(mu1, S1, mu2, S2)` takes **two Gaussians already in hand**. It cannot take features. **The step from features to a Gaussian is the missing piece**, and no shipped file performs it.
- `utils/gmm.h` stores a Gaussian with a **GIVEN** covariance and a **Cholesky** factorisation of it. It never *estimates* one from data — and a Cholesky of a sample covariance from `n` samples in `d > n` dimensions is **singular** (rank ≤ n−1), so the Cholesky route cannot be reused here at all.
- `utils/mmd_loss.h` is a kernel two-sample statistic with no Gaussian fit and no covariance; `utils/sliced_wasserstein.h` is a sliced transport distance with no Gaussian fit and no covariance. Neither produces a mean/covariance pair.
- `utils/metrics.h` is entirely **classification/ranking** metrics on labels and scores.

### What is shipped vs. what is NOT, stated plainly

**The Inception network is NOT shipped, and this plan does not pretend otherwise.** FID is a metric **on feature vectors**; the Inception network is one *choice* of feature extractor (the paper's default). This repo ships the **metric**; the caller supplies features from whatever extractor it has (Inception, a classifier, a CLIP-style encoder, or the model's own penultimate layer). That is the same model-free shape as `utils/sliced_wasserstein.h`, which measures two point sets and likewise ships no network.

The reusable core — **estimating a Gaussian from a sample** — is valuable on its own, independently of any metric, and it is the thing that is genuinely absent.

---

## The mathematics, and the places it is easy to get wrong

### FID (Heusel et al. 2017, NeurIPS)

For feature sets `X` (real) and `Y` (fake), with sample means `mu_X, mu_Y` and sample covariances `S_X, S_Y`:

```
FID = ||mu_X - mu_Y||^2 + tr(S_X) + tr(S_Y) - 2 * tr( (S_X^(1/2) . S_Y . S_X^(1/2))^(1/2) )
```

**FID IS THE SQUARED FRÉCHET DISTANCE, NOT THE DISTANCE.** This is the single most common implementation error, and the repo's shipped `frechet_distance` returns the **distance** (`sqrt` of the radicand). Measured on the oracle's fixture:

```
radicand (d^2) = 5.990601114713      sqrt = 2.447570451430
```

A FID that returns `d` instead of `d^2` is off by a factor of `sqrt(5.99) = 2.4476` — a clean, large, easily-detected error.

**THE REPO'S `frechet_distance` IS REUSED, NOT REIMPLEMENTED.** Its `S1^(1/2) . S2 . S1^(1/2)` form differs from the reference implementation's `sqrtm(S1 @ S2)` form, and the two were measured to agree:

```
tr((S1h S2 S1h)^h)  = 10.852053157499698
tr(sqrtm(S1 @ S2))  = 10.852053157499698     (agreement 5.3e-15)
```

The repo's form is the numerically better one (the product is symmetric PSD, so the outer square root is real; `S1 @ S2` is NOT symmetric — measured asymmetry `0.0739` — and its square root is complex in general). So the shipped function is correct and **a second copy of the same maths would be a second source of truth to keep in sync**, which is exactly what the `TransformerStack` session (2026-11-18) recorded as the reason that stack was deliberately THIN.

**FID IS SQUARED *AFTER* THE CLAMP.** The shipped `frechet_distance` clamps its radicand at zero from below, so `frechet_distance(...) >= 0` always and `FID = d*d >= 0` always. The reference implementation computes the radicand directly and can return a tiny negative number in pathological cases. Squaring the shipped distance is both simpler and better behaved; the suite pins `FID(X, X) == 0` **exactly**, which only holds because of the clamp.

**THE COVARIANCE ESTIMATOR IS THE UNBIASED ONE (divide by n−1).** `np.cov`'s default (`ddof=1`) and `tfp.stats.covariance`'s default (`sample_stats=True`) are both the unbiased estimator, and pytorch-fid's `np.cov(act, rowvar=False)` inherits it. The **biased** estimator (divide by n) is what `tf.nn.moments` computes, so both readings exist in the wild. Measured on the oracle's fixture, they differ:

```
unbiased (n-1): FID = 5.990601114713
biased   (n)  : FID = 5.918540015989
```

The suite pins the unbiased value **and** asserts the biased value differs, so the convention is a *measured* choice rather than an accident.

**THE RANK-DEFICIENT CASE IS THE REALISTIC ONE.** With `n` samples of `d` features and `d > n`, the sample covariance has rank ≤ n−1 and is **singular** — measured `rank(Sa) = 9` for `n = 10, d = 64`, with a minimum eigenvalue of `−5.9e-15` from rounding. The shipped `symmetric_matrix_sqrt` clamps eigenvalues at zero, so this works; but a Cholesky-based implementation would **throw**. A 2048-dimensional Inception feature space with a few hundred samples is exactly this case, so the suite must exercise it.

### Inception Score (Salimans et al. 2016)

For class-probability vectors `p_1..p_N` (each summing to 1) over `K` classes, split into `S` contiguous groups. For each group `g` with marginal `py_g = mean of the group's rows`:

```
IS_g = exp( mean_{x in g} KL(p_x || py_g) )
KL(p || q) = sum_k p_k * log(p_k / q_k)
IS = mean over groups of IS_g
```

**THE `0 * log(0/q) = 0` CONVENTION IS LOAD-BEARING.** The naive expression `p * (log(p) - log(q))` evaluates `0 * (-inf)` on a zero probability and produces **NaN** — measured on a one-hot fixture. A metric that reports NaN reports nothing.

**THE SPLIT COUNT CHANGES THE VALUE**, so it is a pinned convention, not an implementation detail. Measured on the same one-hot fixture:

```
IS with 1 split  = 9.86297313554185
IS with 10 splits = 7.8751657630923475
```

**IS OF AN IDENTICAL SET IS EXACTLY 1.0** — every row equals the marginal, so every KL is 0 and `exp(0) = 1`. This is the sharpest non-vacuity property of the metric and it is exact.

---

## The oracle comes first, and its own sensitivity is measured

`tests/sample_quality_oracle.py` is a float64 numpy+scipy specification written **before any C++ exists**. It pins:

1. `np.cov` (an independent, maintained implementation) against the hand-rolled sample covariance — agreement measured at `1.11e-16`.
2. `scipy.linalg.sqrtm(S1 @ S2)` (the reference implementation's own product order) against the repo's symmetrised form — agreement measured at `5.3e-15`.
3. The **discriminating foils**: the biased estimator, the `d`-instead-of-`d²` variant, the naive product order, the NaN-producing KL. **A foil that produces NaN is a VACUOUS foil** (it discriminates trivially), so the finite wrong variants are the ones used as foils, and the NaN case is asserted separately as a *load-bearing guard*.
4. The rank-deficient fixture (`n < d`) and the near-identical fixture (the covariance floor).

`tests/sample_quality_mutate_oracle.py` breaks the oracle's model code in specific ways and requires it to **fail** each time. **A green oracle on its first run proves nothing about its own sensitivity** (the `off_policy_eval` session, 2026-10-30) — the harness is what turns "the oracle agrees with itself" into "the oracle has teeth".

The C++ fixtures are **emitted by the oracle** (`--emit-cpp`) and spliced into the suite by `tests/splice_sample_quality_fixture.py`. The repo has been burned by hand-transcribed numbers before: both numbers were correct for the fixture they were given, and the fixture simply was not the oracle's. **Pinned numbers transfer only across an identical fixture, so they are transferred, never re-derived.**

---

## Task list

### Task 1: Write the oracle and run it

**Objective:** produce the float64 specification, and make it catch its own defects.

**Files:**
- Create: `tests/sample_quality_oracle.py`

**Step 1:** write the oracle with the sections above; every section prints its numbers AND asserts on them.

**Step 2:** run it.

```bash
/usr/bin/python3 tests/sample_quality_oracle.py
```
Expected: `=== SUMMARY: ORACLE GREEN ===` (a non-zero count means the oracle rejected something — investigate before writing C++).

**Step 3:** commit.

```bash
git add tests/sample_quality_oracle.py
git commit -m "test(utils): the float64 oracle for the sample-quality metrics"
```

### Task 2: The oracle's own mutation harness

**Objective:** prove the oracle catches the defects it was written for.

**Files:**
- Create: `tests/sample_quality_mutate_oracle.py`

**Step 1:** write the harness — a mutation list, an apply step that refuses an ambiguous anchor, a checksum so a no-op edit is reported as a harness defect rather than a verdict, and a pristine-green precondition.

**Step 2:** run it.

```bash
/usr/bin/python3 tests/sample_quality_mutate_oracle.py
```
Expected: every mutation `CAUGHT`; any `SURVIVED` is a REAL coverage gap in the oracle and must be fixed there, not explained away.

**Step 3:** commit.

```bash
git add tests/sample_quality_mutate_oracle.py
git commit -m "test(utils): the oracle's own mutation harness"
```

### Task 3: The implementation

**Objective:** ship `utils/sample_quality.{h,cpp}`.

**Files:**
- Create: `include/nn/utils/sample_quality.h`
- Create: `include/nn/utils/sample_quality.cpp`
- Modify: `include/nn/nn.h` (add the include)

**Step 1:** write the header, documenting every convention above and every guard's message.

**Step 2:** write the implementation.

```cpp
// sample_gaussian_parameters -- the empirical mean and covariance of a
// feature matrix.  features is (n, d): n samples of d-dimensional
// feature vectors.  The covariance is the UNBIASED estimator (n-1).
GaussianParams sample_gaussian_parameters(const Tensor& features);

// frechet_inception_distance -- the FID of two feature sets.  This is the
// SQUARED Frechet distance, and it REUSES utils::frechet_distance.
double frechet_inception_distance(const Tensor& real_features,
                                  const Tensor& fake_features);

// inception_score -- the IS of a class-probability matrix.
double inception_score(const Tensor& probabilities, int splits = 10);
```

**Step 3:** register in `include/nn/nn.h`.

**Step 4:** build.

```bash
make -j$(nproc) setup
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: no diagnostics. The umbrella must compile **standalone** — a focused test that includes only the new header cannot catch a cross-header redefinition.

**Step 5:** commit.

```bash
git add include/nn/utils/sample_quality.h include/nn/utils/sample_quality.cpp include/nn/nn.h
git commit -m "feat(utils): the sample-quality metrics (FID, Inception Score, sample Gaussian)"
```

### Task 4: The C++ suite

**Objective:** pin every convention, against the oracle's emitted fixtures.

**Files:**
- Create: `tests/test_sample_quality.cpp`
- Create: `tests/splice_sample_quality_fixture.py`
- Modify: `Makefile` (build rule, `tests:` prerequisite, `run_tests` line)

**Step 1:** write the suite with the fixture markers:

```cpp
// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
// >>> ORACLE-EMITTED FIXTURE END <<<
```

**Step 2:** splice.

```bash
/usr/bin/python3 tests/splice_sample_quality_fixture.py
```

**Step 3:** build and run.

```bash
make -j$(nproc) build/test_sample_quality && ./build/test_sample_quality
```
Expected: every check `[PASS]`, `=== SUMMARY: N passed, 0 failed ===`.

**Step 4:** commit.

```bash
git add tests/test_sample_quality.cpp tests/splice_sample_quality_fixture.py Makefile
git commit -m "test(utils): the suite for the sample-quality metrics"
```

### Task 5: The C++ mutation harness

**Objective:** find the coverage gaps the suite cannot report by construction.

**Files:**
- Create: `tests/mutate_sample_quality.sh`

**Step 1:** write the harness, honouring every rule this repo has already paid for:

- `restore()` **never** deletes its backup, and is atomic (`cp tmp && mv -f`); cleanup happens exactly once, on EXIT.
- The link set comes from `make print-lib-objs`, **never** from a regex over the Makefile; the object count is asserted `>= 10`.
- The mutated TU is compiled **separately** and swapped in, so it is never in its own link set.
- `MUT-BROKEN` (build failed) and `MUT-CRASH` (`rc >= 128`) are **separate verdict classes** from `CAUGHT`.
- A mutation that did not apply is a **harness defect**, reported as such — the tree is checksummed before and after the edit.
- The anchor must occur **exactly once**, counted by the same engine that performs the edit.
- `rm -f` the intermediates before every build; never run a build concurrently with the harness.
- The pristine tree must be **green** before any mutation runs.
- The tree is rebuilt and re-verified **on EXIT** before any verdict is reported.

**Step 2:** run it.

```bash
bash tests/mutate_sample_quality.sh
```
Expected: `0 survived`. A survivor is a REAL coverage gap — strengthen the suite, do not drop the mutation.

**Step 3:** commit.

```bash
git add tests/mutate_sample_quality.sh
git commit -m "test(utils): the mutation harness for the sample-quality metrics"
```

### Task 6: Sanitizers, the full suite, and the queue record

**Objective:** prove no regression, and record the work.

**Files:**
- Modify: `EXPANSION_QUEUE.md`
- Modify: `NOT_FIXED.md` (only if a new entry is warranted)

**Step 1:** ASan + UBSan.

```bash
make BUILD_DIR=build_asan -j$(nproc) build_asan/test_sample_quality
./build_asan/test_sample_quality
```
Expected: clean. A silent out-of-bounds read is invisible to the suite that contains it — **only a sanitizer finds it**.

**Step 2:** the full suite, and compare against the **pre-work baseline** (measured before any change: exactly one failing suite, `test_neural_ode`, the long-standing entry in `NOT_FIXED.md`).

```bash
make -j$(nproc) tests && make run_tests
```
Expected: the same single failure, and no new one.

**Step 3:** move the item to `## Done` in `EXPANSION_QUEUE.md` with a one-line summary and the test results.

**Step 4:** commit and push.

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: record the sample-quality metrics in the queue"
git push origin master
```

---

## Scope, stated plainly

- **No Inception network ships, and none is faked.** The metrics are defined on feature vectors and class probabilities; the caller supplies them. A from-scratch "Inception" with random weights would produce numbers that are meaningless *as FID*, which is worse than not shipping it.
- **No `Layer`, no `Model`, no randomness, no optimizer.** Every core is a free function over `Tensor`.
- **The FID path composes `utils::frechet_distance`.** There is exactly one copy of the Gaussian-to-Gaussian distance in the repo.
- **The Inception Score takes probabilities, not a network.** The `splits` convention is pinned and documented, not left implicit.
