# Isotonic Regression and Monotone Calibration Map — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship a model-free isotonic-regression utility module (`include/nn/utils/isotonic_regression.{h,cpp}`) that fits a monotone (non-decreasing or non-increasing) map from a score to a calibrated probability by the Pool-Adjacent-Violators algorithm, plus the weighted-L2 isotonic regression primitive it is built on.

**Architecture:** free functions over `Tensor` in `nn::utils`, matching the shape of `utils/sinkhorn.h`, `utils/crf.h`, `utils/conformal.h`, `utils/off_policy_eval.h` and `utils/rank_correlation.h` — no classes, no `Layer`, no randomness. The one piece of machinery it composes rather than reimplements is the calibration diagnostics already shipped in `utils/metrics.h` (`expected_calibration_error`, `brier_score_loss`), which is how the "did the calibration help" question is answered.

**Tech Stack:** C++17, the repo's `Tensor` (row-major, `AlignedAllocator<double>`), the project's own `-O2 -march=native -Wall -Wextra` build, a float64 numpy oracle in `tests/`, an oracle-mutation harness in `tests/`, and a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

```
grep -rinE 'isotonic|monotone.calib|pool.adjacent|PAVA' include/ tests/
-> 0 hits on the concept. The only "monotone" hits are unrelated prose
   ("monotone non-decreasing" in dadaptation.h / prodigy.h, "monotone
   spline" in neural_spline_flow.h, "monotone" in the temperature_scaling
   and curriculum comments).
```

The repo has ~250 layers and 60+ utility files and **no monotone calibration map**. What it has instead is the *parametric* half of the same problem:

| shipped | what it is | what it cannot do |
|---|---|---|
| `utils/temperature_scaling.h` | ONE scalar `T` fitted by minimising NLL on `softmax(z / T)` | a one-parameter family; it cannot represent a step, a plateau, or any non-logistic shape |
| `utils/metrics.h` | ECE / ACE / TACE / classwise-ECE / Brier / reliability diagrams | **descriptive only** — it reports how far the probabilities are from the observed frequencies and changes nothing |
| `utils/conformal.h` | distribution-free *coverage* of a prediction SET | coverage is not calibration: a set can cover perfectly while every member probability is wrong |

**THE COMPOSITION IS THE POINT, AND IT IS MEASURABLE.** The repo can already *measure* calibration error and it can already *fit* a one-scalar calibration. What it cannot do is fit a **non-parametric monotone** calibration map and then measure it with the diagnostics that already exist. Isotonic regression is the standard non-parametric alternative to Platt scaling and to temperature scaling, and it is the method the calibration literature compares against (Zadrozny & Elkan; Niculescu-Mizil & Caruana).

### WHY "COMPOSES WITH THE SHIPPED CONFORMAL PREDICTION" IS A REAL CLAIM AND NOT A SLOGAN

The two are **complementary, not alternatives**, and the repo ships both:

- **Conformal prediction** answers *"is the true label in the set?"* and guarantees **marginal coverage** — the set covers with probability ≥ 1-α, for ANY score distribution. It says nothing about whether a member probability is 0.9 or 0.1.
- **Isotonic calibration** answers *"given this score, how likely is the positive class?"* and guarantees only **monotonicity** and (for a calibration set drawn from the same distribution) **calibration in expectation**. It says nothing about coverage.

So the natural pipeline is: **isotonic-calibrate the scores → feed the calibrated probabilities into the conformal machinery.** The score's *ordering* is preserved by an increasing isotonic map, so a conformal set built on calibrated scores still covers; and the calibrated probabilities are now meaningful numbers. The suite pins both halves: the map is monotone non-decreasing (which is what makes the conformal composition valid), and a conformal prediction set built on calibrated scores still covers.

### SCOPE BOUNDARY (stated so it cannot creep in)

**v1 is the BINARY, MODEL-FREE case.** One score in, one calibrated probability out.

**OUT OF SCOPE FOR v1, stated so it cannot creep in:**
- **Multiclass one-vs-rest calibration** (scikit-learn's `_fit_calibrator` fits one isotonic map per class and renormalises the row). A different composition rule, and the renormalisation is where the interesting bugs live — a v2 candidate, not v1.
- **Any `Layer`, any randomness, any model.** Nothing here touches a network.
- **Quantile regression / CQR-style monotone quantile fitting.** A different objective (pinball loss, not squared loss).
- **Cross-validated calibration** (fitting the map out-of-fold). The repo ships `utils/cross_validation.h` for exactly this, but the natural v1 is the primitive plus the report.

## The mathematics, and the four places it is easy to get wrong

**1. PAVA IS THE ALGORITHM, AND IT POOLS *ADJACENT* VIOLATORS.** The input is sorted by `x`; a stack of blocks is maintained; whenever the second-to-last block's mean exceeds the last block's mean (for an increasing fit), the two are **merged** and the merged block's value is the **weighted** mean of everything it contains. This is the algorithm of Best & Chakravarti, and it is the algorithm scikit-learn's `IsotonicRegression` runs.

**2. TIES IN `x` ARE COLLAPSED FIRST, TO A *WEIGHTED MEAN*, AND THE WEIGHTS ARE SUMMED.** This is the rule in scikit-learn's `_make_unique`, and it is the single most common way an independent implementation diverges: an implementation that collapses tied `x` to the *unweighted* mean of their `y`s, or that forgets to sum the weights, gets a different answer on any fixture with repeated `x`. **Measured**: `x=[1,1], y=[0,1], w=[3,1]` gives `0.25` (the weighted mean), not `0.5` (the plain mean) and not `1.0`.

**3. THE FITTED VALUES ARE THE BLOCK MEANS; THE *PREDICTION* IS A LINEAR INTERPOLATION THROUGH THEM.** The isotonic regression itself is a **step function** on the training `x`s. scikit-learn's `predict` is not that step function — it is `interp1d(X, y, kind="linear")` through the retained knots, i.e. a **piecewise-linear** function. **Measured**: with `x=[1,2,3,4,5], y=[0,1,0,0,1]`, the fitted knots are `(1,0), (2,1/3), (4,1/3), (5,1)` and `predict(4.5) = 2/3`, where the pure step function would give `1.0`. **The distinction is observable and the suite pins it.**

**4. THE RETAINED KNOTS ARE THE *TRIMMED* STEP VALUES.** After PAVA, interior points whose fitted value equals **both** neighbours' are dropped ("remove unnecessary points for faster prediction", scikit-learn's own comment). The trimming does not change the interpolated function — the removed point lies on the segment between its neighbours — but it changes the *knot list*, and the knot list is part of the public output. **Measured**: `x=[1,2,3,4,5], y=[0,1,0,0,1]` keeps four knots, not five.

### THE THREE FITTED-KNOT SEMANTICS THAT NOBODY EXPECTS

**S1. `increasing=False` IS NOT "NEGATE AND FIT INCREASING".** scikit-learn implements the decreasing case by *reversing the input order*, running the increasing algorithm, and reversing back. That is algebraically the same fit but it produces a **different knot list** (the knots come out in reverse order in the internal representation), and the difference is observable in the tie-collapsing step. The suite pins the decreasing case against scikit-learn directly.

**S2. ZERO WEIGHTS ARE DROPPED, NOT TREATED AS ZERO-WEIGHT POINTS.** scikit-learn masks out `sample_weight > 0` before the fit. A zero-weight point must not contribute to the tie-collapse, must not shift the `x` range, and must not appear as a knot.

**S3. `out_of_bounds` HAS THREE REGIMES, AND "CLIP" IS NOT THE ONLY ONE.** scikit-learn's `IsotonicRegression` defaults to `out_of_bounds="nan"`, but scikit-learn's own **calibration** code uses `out_of_bounds="clip"`. The distinction matters: `"clip"` extends the end values, `"nan"` returns NaN outside the fitted range, and `"raise"` throws. The module exposes all three explicitly, because a calibration map that silently returns NaN for an out-of-range score is a trap and a calibration map that silently clips hides a distribution shift — **the caller must choose.**

## The shipped API

```cpp
namespace nn { namespace utils {

enum class IsotonicDirection { Increasing, Decreasing };
enum class IsotonicOutOfBounds { Clip, Nan, Raise };

// The fitted monotone map: the retained knots and the direction.
struct IsotonicFit {
    std::vector<double> x_knots;   // strictly increasing
    std::vector<double> y_knots;   // monotone in the fitted direction
    IsotonicDirection direction = IsotonicDirection::Increasing;
    // x_min() / x_max() are the fitted range, NOT the knots' extremes
    // (which are the same, but the range is what out-of-bounds uses).
};

// Fit the WEIGHTED L2 isotonic regression of y on x by PAVA.
IsotonicFit fit_isotonic(const Tensor& x, const Tensor& y,
                          const Tensor& weights,
                          IsotonicDirection direction = IsotonicDirection::Increasing);

// The weighted-L2 isotonic regression evaluated by LINEAR INTERPOLATION
// through the retained knots.
Tensor isotonic_predict(const IsotonicFit& fit, const Tensor& x,
                         IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

// The CALIBRATION MAP: a monotone non-decreasing map from an arbitrary
// score to P(y = 1 | score), fitted by the PAVA on the 0/1 labels.
Tensor isotonic_calibrate(const Tensor& scores, const Tensor& labels,
                           const Tensor& weights = Tensor(),
                           IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

// The REPORT: does the calibration actually help? Measured with the
// repo's OWN metrics.h, not a re-implementation.
struct IsotonicCalibrationReport {
    double ece_before = 0.0, ece_after = 0.0;
    double brier_before = 0.0, brier_after = 0.0;
    size_t n_knots = 0;
    size_t n_samples = 0;
};
IsotonicCalibrationReport evaluate_isotonic_calibration(
    const Tensor& scores, const Tensor& labels,
    const Tensor& weights = Tensor(),
    size_t n_bins = 10,
    IsotonicOutOfBounds oob = IsotonicOutOfBounds::Clip);

}}
```

**WHY THE REPORT EXISTS.** A calibration map that is fitted and never measured is a liability: isotonic regression on a small calibration set **overfits** and can make the held-out calibration *worse*. The report is therefore computed on the **same** data by default and the header says so loudly — the honest comparison is out-of-fold, and the repo's `utils/cross_validation.h` is the tool for that. **The report's contract is stated, not assumed.**

## Tasks

### Task 1: The float64 numpy oracle — the specification

**Objective:** Write `tests/isotonic_oracle.py`, the float64 numpy specification, and run it green.

**Files:**
- Create: `tests/isotonic_oracle.py`

**Content:** The oracle implements PAVA **from the mathematics**, and is cross-checked against **scikit-learn 1.9.0's `IsotonicRegression`** (a MAINTAINED reference implementation, available at `/usr/bin/python3`) on randomised fixtures — the same discipline the n-gram session used against NLTK's `WittenBellProbDist`.

The cross-check is the point: **a PAVA implementation written by the same author as the C++ is not independent.** The oracle's PAVA is checked against scikit-learn's on ≥ 500 randomised cases with **heavy ties** in `x` and **non-uniform weights**, in both directions, plus the three `out_of_bounds` regimes and the tie-collapse/trim rules.

Sections:
| § | content |
|---|---|
| 1 | the tie-collapse rule — weighted mean of `y`, summed weights; the `x=[1,1], y=[0,1], w=[3,1]` → `0.25` case pinned |
| 2 | PAVA itself — hand fixture with a known pool, and the increasing/decreasing cases |
| 3 | the **step vs. linear-interpolation** distinction — the `x=[1,2,3,4,5], y=[0,1,0,0,1]` fixture, where `predict(4.5)` is `2/3` under interpolation and `1.0` under the step function |
| 4 | the **trimming** rule — the retained knot list, pinned |
| 5 | the three `out_of_bounds` regimes, including the single-knot case |
| 6 | the **calibration map** — monotone non-decreasing, output in `[0,1]`, and the invariance properties |
| 7 | the **report** — ECE and Brier before/after, computed with scikit-learn's own `calibration_curve` / `brier_score_loss` so the numbers are not self-referential |
| 8 | the pinned fixture, emitted as C++ literals |

**Step 2: Run the oracle**

Run: `/usr/bin/python3 tests/isotonic_oracle.py`
Expected: `=== SUMMARY: N passed, 0 failed ===`, exit 0.

**NOTE:** `/usr/bin/python3` has numpy 2.3.5 + scipy 1.17.1 + sklearn 1.9.0. The `python3` on `PATH` is the Hermes venv, which has **no numpy** — use the absolute path.

**Step 3: Commit**

```bash
git add tests/isotonic_oracle.py
git commit -m "test(utils): float64 oracle for the isotonic regression utilities"
```

### Task 2: Measure the oracle's own sensitivity

**Objective:** Write `tests/isotonic_mutate_oracle.py` and require that it catches every mutation. **A green oracle on its first run proves nothing about its own sensitivity** — this repo has now paid for that lesson twice (the off-policy-evaluation oracle's "two defects in its first draft" turned out to be zero).

**Files:**
- Create: `tests/isotonic_mutate_oracle.py`

**Mutations:** each breaks the specification in one specific way and the harness requires the oracle to FAIL:

| id | breaks |
|---|---|
| OM1 | the tie-collapse uses the UNWEIGHTED mean of `y` |
| OM2 | the tie-collapse does not SUM the weights |
| OM3 | PAVA merges non-adjacent violators (a global-minimum shortcut) |
| OM4 | the merged block's value is the mean of the two block MEANS, not of the block's points |
| OM5 | the prediction is the STEP function, not the linear interpolation |
| OM6 | the trimming is not applied (all knots retained) |
| OM7 | zero-weight points are kept instead of dropped |
| OM8 | the decreasing direction is implemented as "negate and fit increasing" |
| OM9 | `out_of_bounds=Clip` returns the mean of the end values instead of the end values |

**The harness rules this repo has already learned:**
1. The anchor must occur **exactly once** — a twice-occurring anchor makes the edit a coin flip, and a coin flip is not a verdict.
2. The mutated source is compared to the pristine source — a mutation that did not apply is a **harness defect**, not a verdict.
3. A non-zero exit with **zero reported failures** is a crash, not a verdict → `MUT-BROKEN`.
4. A **survivor is a hole in the SPECIFICATION**, never a pass.

**Step 2: Run the harness**

Run: `/usr/bin/python3 tests/isotonic_mutate_oracle.py`
Expected: `=== SUMMARY: 9 caught, 0 survived, 0 broken ===`, exit 0.

**Step 3: Commit**

```bash
git add tests/isotonic_mutate_oracle.py
git commit -m "test(utils): mutation harness for the isotonic regression oracle"
```

### Task 3: The RED C++ test

**Objective:** Write `tests/test_isotonic.cpp` and watch it fail to compile because the header does not exist yet.

**Files:**
- Create: `tests/test_isotonic.cpp`

**The test mirrors the oracle's sections**, plus:
- the **guard** checks (empty input, length mismatch, non-finite values, negative weights, all-zero weights, an unknown `out_of_bounds`) — every guard's **message** is asserted, not merely that something was thrown, because a substring shared by two guards discriminates nothing;
- the **foil** checks — the wrong variants must be *shown to differ* from the right answer, so a suite that cannot tell them apart cannot pass;
- the pinned fixture's exact values, transferred from the oracle's emitted C++ literals (**not** hand-transcribed — the repo has been burned by hand-transcribed numbers before, see `tests/_dump_fixture.py`).

**Step 2: Run it and watch it fail**

Run: `make build/test_isotonic`
Expected: compile error — `nn/utils/isotonic_regression.h: No such file or directory`.

**Step 3: Commit**

```bash
git add tests/test_isotonic.cpp
git commit -m "test(utils): RED test for the isotonic regression utilities"
```

### Task 4: The implementation

**Objective:** Write `include/nn/utils/isotonic_regression.h` and `.cpp` so the RED test goes green.

**Files:**
- Create: `include/nn/utils/isotonic_regression.h`
- Create: `include/nn/utils/isotonic_regression.cpp`

**Implementation notes:**
- The tie-collapse sorts by `x` and groups equal `x`, accumulating `sum(w * y)` and `sum(w)`. **The sort must be stable in the sense that equal `x` groups are contiguous** — a `std::sort` on `(x, y)` pairs is fine, but the accumulation must be per-group.
- PAVA keeps a stack of `(sum_wy, sum_w, count, first_x_index)` blocks. Pooling merges the last two.
- **The block's value is `sum_wy / sum_w`**, recomputed from the accumulated sums — never the average of the two block means (that is OM4, and it is wrong whenever the two blocks have different weights or different sizes).
- The prediction is a linear interpolation between the retained knots, with the three `out_of_bounds` regimes applied against the fitted `x` range.
- The report calls the repo's OWN `metrics.h` functions (`expected_calibration_error`, `brier_score_loss`) — **no re-implementation** of ECE or Brier in this file.

**Step 2: Run it and watch it pass**

Run: `make build/test_isotonic && ./build/test_isotonic`
Expected: all checks pass, exit 0.

**Step 3: Commit**

```bash
git add include/nn/utils/isotonic_regression.h include/nn/utils/isotonic_regression.cpp
git commit -m "feat(utils): the isotonic regression and monotone calibration utilities"
```

### Task 5: Wire it into the public surface and the test target

**Objective:** Make the module reachable from `nn/nn.h` and runnable from `make run_tests`.

**Files:**
- Modify: `include/nn/nn.h` (add the include next to the other utils)
- Modify: `Makefile` (add the `$(BUILD_DIR)/test_isotonic` link rule, add it to the `tests:` prerequisite list, add the `run_tests` line)

**Step 2: Verify the umbrella header compiles standalone**

Run:
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: no output, exit 0.

**Step 3: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "build: wire the isotonic regression suite into the test target"
```

### Task 6: The C++ mutation harness

**Objective:** Write `tests/mutate_isotonic.sh` and require that it catches every mutation.

**Files:**
- Create: `tests/mutate_isotonic.sh`
- Modify: `tests/test_isotonic.cpp` (add the checks the survivors demand)

**Mutations** (each targets one of the four "places it is easy to get wrong" plus the guards):

| id | breaks |
|---|---|
| M1 | the tie-collapse uses the UNWEIGHTED mean |
| M2 | the tie-collapse does not sum the weights |
| M3 | PAVA pools non-adjacent violators |
| M4 | the pooled block value is the mean of the two block MEANS |
| M5 | the prediction is the step function, not the interpolation |
| M6 | the trimming is not applied |
| M7 | zero-weight points are kept |
| M8 | the decreasing direction is "negate and fit increasing" |
| M9 | `out_of_bounds=Clip` returns the mean of the end values |
| M10 | the length-mismatch guard is removed |
| M11 | the non-finite guard is removed |
| M12 | the report's "after" is computed from the BEFORE probabilities |

**The harness rules this repo has already learned** (copied from `tests/mutate_rank_correlation.sh`):
1. `restore()` must NEVER delete its backup; cleanup happens exactly once, on EXIT. A self-cleaning restore works exactly once and leaves a MUTANT IN THE TREE.
2. The link set comes from `make print-lib-objs`, NEVER from a regex over the Makefile (the variable is recursive, so a regex yields an empty set and every mutation scores a false `MUT-BROKEN`). The count is asserted `>= 10`.
3. The mutated TU is compiled separately and swapped in for the library's copy.
4. `MUT-BROKEN` means "the build failed" — a SEPARATE verdict class, never counted as a pass.
5. `rc >= 128` is a CRASH — a THIRD verdict class, not a verdict about the suite.
6. `rm -f` the intermediates before every build.
7. NEVER run a build while this harness is running.
8. The anchor must be UNIQUE; the count is asserted to be exactly 1 before the edit.

**Step 2: Run the harness**

Run: `bash tests/mutate_isotonic.sh`
Expected: `=== SUMMARY: 12 caught, 0 survived, 0 broken, 0 crashed, 0 noop, 0 anchor ===`, exit 0.

**Step 3: Commit**

```bash
git add tests/mutate_isotonic.sh tests/test_isotonic.cpp
git commit -m "test(utils): mutation harness for the isotonic regression utilities"
```

### Task 7: The aggregate gates

**Objective:** Prove nothing regressed.

**Step 1: Run the full suite**

Run: `make run_tests`
Expected: the only failure is the `test_neural_ode` (ODERNN) entry already deferred in `NOT_FIXED.md`.

**Step 2: Run the sanitizers on the new binary**

Run:
```bash
make BUILD_DIR=build_asan build_asan/test_isotonic
./build_asan/test_isotonic
```
Expected: clean, no ASan/UBSan reports.

**Step 3: Check the diff**

Run: `git diff --check && git status --porcelain`
Expected: no whitespace errors, no stray files.

### Task 8: Record the work

**Objective:** Update `EXPANSION_QUEUE.md` — move the bullet from `## Ideas` to `## Done` with a one-line summary and the test results, and note the verified-absent list for the next session.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 2: Commit**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: record the isotonic regression utilities in the expansion queue"
```

**Step 3: Push**

```bash
git push origin master
```

---

## References

- **Best, M. J. and Chakravarti, N.** "Active set algorithms for isotonic regression; A unifying framework." *Mathematical Programming* **47** (1990), 425–439. DOI `10.1007/BF01580873`. — the active-set / PAVA framework the fit is built on. **Verified 2026-11-10 by a title search**; this is the reference scikit-learn's own `isotonic_regression` docstring cites ("section 3").
- **Chakravarti, N.** "Isotonic Median Regression: A Linear Programming Approach." *Mathematics of Operations Research* **14**(2) (May 1989), 303–308. DOI `10.1287/moor.14.2.303`. — the PAVA-as-dual-of-an-LP result. **Verified 2026-11-10 by a title search.**
- **Zadrozny, B. and Elkan, C.** "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD 2001*, 204–213. DOI `10.1145/775047.775151`. — the isotonic-calibration paper. **Verified 2026-11-10 by a title search.**
- **Zadrozny, B. and Elkan, C.** "Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers." *ICML 2001*, 609–616. — the binary isotonic calibration. **Verified 2026-11-10 by a title search** (the record at `cseweb.ucsd.edu/~elkan/calibrated.pdf`).
- **Niculescu-Mizil, A. and Caruana, R.** "Predicting Good Probabilities with Supervised Learning." *ICML 2005*. DOI `10.1145/1102351.1102430`. — the sigmoid-vs-isotonic comparison this repo's `temperature_scaling.h` is the sigmoid half of. **Verified 2026-11-10 by a title search.**

**NO arXiv ID IS ASSERTED.** All five references are pre-arXiv or have no arXiv record, and this repo's expansion queue has produced fifteen wrong arXiv identifiers — every one a plausible-looking neighbour of the right paper. The reference implementation used for the cross-check is **scikit-learn 1.9.0**, installed locally at `/usr/bin/python3`.
