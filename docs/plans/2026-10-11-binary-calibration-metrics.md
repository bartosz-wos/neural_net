# Binary & Multiclass Ranking / Calibration Metrics — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Extend `include/nn/utils/metrics.{h,cpp}` with the scoring-rule family the
repo does not have — ROC-AUC, PR-AUC (average precision), ROC/PR curves,
Brier score, log-loss, calibration curve and ECE — so a classifier can be
evaluated on *ranking quality* and *probability quality*, not just accuracy.

**Architecture:** New public free functions in the existing `metrics.{h,cpp}`
(no new files, no new category — `metrics` is already the evaluation category
and already registered in `nn.h`). Two new public structs that own curve data
(`ROCCurve`, `PrecisionRecallCurve`, `CalibrationCurve`) plus one enum
(`MulticlassAveraging`). Everything builds on one shared private
`BinaryClfCurve` helper, mirroring scikit-learn's internal decomposition, so
ROC and PR cannot drift apart.

**Tech Stack:** C++17, the repo's 2-D `Tensor`, `std::vector`, hand-rolled
tests in the existing `tests/test_metrics.cpp` style (`CHECK` / `CHECK_NEAR`
macros, `=== Summary ===`).

---

## The gap, stated precisely

`metrics.{h,cpp}` today exposes exactly four functions:

```
accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report
```

All four are **argmax-based**: they collapse a score vector to one integer per
row before measuring anything. Verified absent by content grep on 2026-10-11:

```
grep -ril 'brier\|calibration\|average_precision\|roc_auc\|ece' include/   → 0
```

That is not a niche gap. It is the reason the metrics file cannot evaluate the
two things a production classifier is actually judged on:

1. **Ranking quality** — does a higher score mean "more likely positive"?
   ROC-AUC and PR-AUC measure exactly that and need the *scores*, which the
   argmax throws away. This is the metric for imbalanced data where accuracy
   is uninformative (99% accuracy is the trivial model).
2. **Probability quality** — when the model says 0.9, is it right 90% of the
   time? Brier, log-loss, calibration curve and ECE all measure that.

It is also the exact complement of yesterday's `losses.{h,cpp}`: those are the
*training* objectives (CE / NLL / BCE), these are their *evaluation* cousins.
Every one of them is a strictly proper scoring rule or a ranking statistic.

## Why this and not another layer

`## Ideas` in `EXPANSION_QUEUE.md` is fully drained — all 20+ bullets are
marked popped and the last one (canonical losses) shipped yesterday. So the
queue offers no next item and a new layer would be invented rather than
requested. `metrics.{h,cpp}` is the one place with an obvious, verifiable,
user-facing hole in a file the repo already owns.

---

## Semantics read from source, not from memory

Every rule below was read out of scikit-learn's source, fetched 2026-10-11.
**This step found four bugs in my own hand-written oracle before a single line
of C++ existed**, which is exactly why it is written into the plan.

| # | Rule | Source | Why a reimplementation gets it wrong |
|---|---|---|---|
| D1 | ROC curve prepends a `+inf` threshold giving `(0,0)`; `fpr = fps/fps[-1]`, `tpr = tps/tps[-1]` | `_ranking.py:1310-1375` | You must group **equal scores into one threshold**, not one point per sample. `threshold_idxs = nonzero(diff(y_score)) ∪ {n-1}` (`:914-918`). Sampling per point makes the trapezoid wrong for any tied data. |
| D2 | `roc_auc_score = auc(fpr, tpr)`, trapezoid, direction-aware | `auc` `:91-115` | Naive `Σ (x[i+1]-x[i])(y[i]+y[i+1])/2` is only correct because fpr is non-decreasing. The reference guards the decreasing case. |
| D3 | **AP ≠ area under PR (trapezoid).** AP is the step-function integral `-Σ Δrecall · precision[i]`, clamped at 0 | `:241-257` | `trapezoid(PR) ≠ AP` — the two orders differ on the repo's own fixture X: **0.542857 vs 0.438095**. A reimplementation that integrates the curve gets a *different plausible number*. This is the single most-pinned test in the suite. |
| D4 | `precision[ps==0] = 0`, not NaN | `:1189-1190` | Uninitialised/NaN precision at empty thresholds. |
| D5 | PR curve is returned **recall-decreasing** (flipped, plus a `(1.0, 0.0)` sentinel) | `:1202-1210` | Returning it ascending silently mis-associates precision with recall. |
| D6 | **Brier halves when `cols < 3`** (`scale_by_half="auto"`), NOT when binary | `_classification.py:3849-3852` | The natural guess "binary → halve" is wrong: a **2-column** multiclass input *does* get halved. Pinned by `B2cls = 0.09` vs `B2cls no-half = 0.18`. |
| D7 | log-loss clips probabilities to `[eps, 1-eps]` with `eps = DBL_EPSILON` | `:3449-3450` | Without the clip, `p=1.0` on a true label returns `-inf`. With it, you get the **finite** `36.04365338911715` = `-log(DBL_EPSILON)`. Both the clip AND its exact value are asserted. |
| D8 | `_xlogy` is 0 where the label indicator is 0, so log-loss sums only the true class | `:3452` | Summing `-log(p_c)` over *all* classes is a different metric (`log(p)` of the full row, off by `log Σp`). Pinned: 2-col fixture gives **1.8161075557302173**, not the single-column form's 0.1976. |
| D9 | Calibration bins use `np.searchsorted(edges[1:-1], p)` — **side-left**, so a value exactly on an interior edge falls in the **lower** bin | `calibration.py:1455` | `side='right'` puts exact-edge values in the upper bin. Only distinguishable on a fixture with a probability exactly on an edge — hence the deliberate `p=0.5` case. |
| D10 | Calibration **drops empty bins** — the output is shorter than `n_bins` | `calibration.py:1458-1462` | Allocating `n_bins` slots and filling them means the caller cannot tell "empty" from "real". Output length must equal the non-zero bin count. |
| D11 | `quantile` strategy interpolates the quantile linearly | `calibration.py:1442-1443` | `np.percentile`'s default linear interpolation. A nearest-rank quantile is an equally plausible-looking and wrong axis. |
| D12 | Multiclass ROC-AUC is **one-vs-rest**, then averaged; `weighted` divides by total support | `:795-855` | One-vs-one is a different number. And macro ≠ weighted whenever class counts differ — pinned on a fixture searched for exactly that (0.388889 vs 0.541667). |

---

## Files

- Modify: `include/nn/utils/metrics.h` (add declarations + 3 structs + 1 enum)
- Modify: `include/nn/utils/metrics.cpp` (implement; extend the anonymous namespace)
- Modify: `tests/test_metrics.cpp` (extend the RED suite)
- Modify: `Makefile` (already builds and runs `test_metrics` — no change needed)

`nn.h` already includes `utils/metrics.h`? **To verify**; if not, one line.

---

## Task 1: RED suite for the ranking metrics

**Objective:** Tests for the curve helpers + ROC-AUC + AP, written and
observed failing before any implementation exists.

**Files:**
- Modify: `tests/test_metrics.cpp`

**Step 1: append tests to `tests/test_metrics.cpp`**

Test cases (each with its expected value taken from the ground-truth run above,
not from my own arithmetic):

```
T1  ROC curve on FIX_A: fpr=[0,0,0.5,0.5,1], tpr=[0,0.5,0.5,1,1], thr[0]==+inf
T2  ROC-AUC FIX_A = 0.75                      (equals sklearn's own doctest)
T3  AP FIX_A = 0.8333333333333333
T4  PR curve FIX_A: precision=[0.5,0.6666666666666666,0.5,1,1]
                     recall=[1,1,0.5,0.5,0]  thr=[0.1,0.35,0.4,0.8]
T5  NON-SPECIFICITY: AP(X) = 0.5428571428571428 != trapezoid(PR)(X) = 0.4380952380952381
T6  all-tied scores -> AUC 0.5 exactly, AP 0.25 exactly
T7  perfect separation -> AUC 1.0, AP 1.0;  inverted -> AUC 0.0
T8  single positive among negatives -> AUC 1.0, fpr=[0,0,1], tpr=[0,1,1]
T9  two rows only: AUC([0,1],[0.1,0.9]) = 1.0, PR precision=[0.5,1,1]
T10 pos_label semantics: y=[1,0,1,0] s=[0.9,0.1,0.8,0.2] -> fpr=[0,0,0,1]
T11 edge cases throw: mismatched rows, empty input, non-finite scores
```

**Step 2: build + run, verify RED**

```
make $(grep -o 'build/test_metrics' Makefile | head -1)   # or: make build/test_metrics
./build/test_metrics
```
Expected: **compile failure** — `roc_auc_score` undeclared. That is the correct
RED: the symbol does not exist, which is the honest failure mode for a new API.

**Step 3: commit**

```bash
git add tests/test_metrics.cpp
git commit -m "test(metrics): RED suite for ROC/PR curves, ROC-AUC, average precision"
```

## Task 2: GREEN — curves + ROC-AUC + AP

**Objective:** Minimal implementation to make Task 1 pass.

**Step 1: add to `metrics.h`**

```cpp
#ifndef NN_METRICS_AVERAGING_ENUM_DEFINED
#define NN_METRICS_AVERAGING_ENUM_DEFINED
// Averaging mode for the per-class multiclass metrics (mirrors scikit-learn's
// `average=` argument on roc_auc_score / average_precision_score).
enum class MulticlassAveraging { Macro, Weighted };
#endif

struct ROCCurve {
    std::vector<double> fpr;
    std::vector<double> tpr;
    std::vector<double> thresholds;   // thresholds[0] == +inf, then decreasing
};

struct PrecisionRecallCurve {
    std::vector<double> precision;
    std::vector<double> recall;       // recall is DECREASING
    std::vector<double> thresholds;
};

double roc_auc_score(const Tensor& y_true, const Tensor& y_score,
                     double pos_label = 1.0);
double average_precision_score(const Tensor& y_true, const Tensor& y_score,
                               double pos_label = 1.0);
ROCCurve roc_curve(const Tensor& y_true, const Tensor& y_score,
                   double pos_label = 1.0);
PrecisionRecallCurve precision_recall_curve(const Tensor& y_true, const Tensor& y_score,
                                            double pos_label = 1.0);
double auc_trapezoid(const std::vector<double>& x, const std::vector<double>& y);
```

**Step 2: implement in `metrics.cpp`**

One private helper mirroring `_binary_clf_curve`:

```cpp
struct BinaryClfCurve {
    std::vector<double> fps, tps, thresholds;   // at distinct descending thresholds
};
BinaryClfCurve binary_clf_curve(const std::vector<bool>& y_true,   // == pos_label
                                const std::vector<double>& y_score);
```

Sort by descending score **stably**, accumulate tps/fps, and emit one point per
*distinct* score value — D1. Then `roc_curve` prepends the `(0,0)/+inf` point and
divides, `precision_recall_curve` computes `precision = tps/(tps+fps)` with 0
where the denominator is 0 (D4) and flips (D5), and
`average_precision_score` takes the step integral (D3).

**Step 3: build + run, verify GREEN**

Expected: `=== Summary: N passed, 0 failed ===`, exit 0.

**Step 4: commit**

```bash
git add include/nn/utils/metrics.h include/nn/utils/metrics.cpp tests/test_metrics.cpp
git commit -m "feat(metrics): ROC/PR curves, roc_auc_score, average_precision_score"
```

## Task 3: RED → GREEN for the proper scoring rules

**Objective:** Brier score + log-loss, with the `scale_by_half` and clip rules
pinned.

```
T12 Brier binary FIX_C = 0.0375  (sklearn doctest value)
T13 Brier scale_by_half=false = 0.075
T14 Brier 3-class = 0.1466666666666667
T15 Brier hard predictions -> exactly 0.0
T16 Brier auto-halves at cols<3: 2-col = 0.09, no-half = 0.18
T17 log-loss FIX_C = 0.19763488164214868
T18 log-loss clip: label=1, p=[[1.0,0.0]] -> exactly -log(DBL_EPSILON)=36.04365338911715
T19 log-loss xlogy: 2-col FIX_C row -> 1.8161075557302173 (NOT the 1-col form)
T20 reject probabilities outside [0,1]
```

**API:**

```cpp
double brier_score_loss(const Tensor& y_true_index, const Tensor& y_proba,
                        bool scale_by_half = true /* -1 sentinel for "auto" */);
double log_loss(const Tensor& y_true_index, const Tensor& y_proba);
```

Use an explicit `BrierScale` sentinel rather than `bool` so `"auto"` can be
expressed without a magic third value on the default argument.

**Commit:** `feat(metrics): brier_score_loss, log_loss`

## Task 4: RED → GREEN for calibration

```
T21 calibration_curve FIX_B uniform n_bins=3: prob_true=[0,0.5,1], prob_pred=[0.2,0.525,0.85]
T22 quantile: prob_true=[0,0.6666666666666666,1], prob_pred=[0.2,0.5833333333333334,0.9]
T23 bin counts [3,2,4] uniform, [3,3,3] quantile
T24 empty bins are DROPPED: 4 samples all in bin 0, n_bins=5 -> output length 1
T25 side-left searchsorted: p exactly 0.5 falls in the LOWER bin
T26 ECE FIX_B uniform = 0.13888888888888892, quantile = 0.12777777777777777
T27 ECE == 0 on a fixture constructed to be calibrated by definition
T28 reject p outside [0,1]
```

**API:**

```cpp
#ifndef NN_METRICS_CALIBRATION_STRATEGY_DEFINED
#define NN_METRICS_CALIBRATION_STRATEGY_DEFINED
enum class CalibrationStrategy { Uniform, Quantile };
#endif

struct CalibrationCurve {
    std::vector<double> prob_true;
    std::vector<double> prob_pred;
    std::vector<size_t> counts;       // same length as prob_true / prob_pred
};

CalibrationCurve calibration_curve(const Tensor& y_true, const Tensor& y_prob,
                                   size_t n_bins, double pos_label,
                                   CalibrationStrategy strategy = CalibrationStrategy::Uniform);
double expected_calibration_error(const Tensor& y_true, const Tensor& y_prob,
                                  size_t n_bins, double pos_label,
                                  CalibrationStrategy strategy = CalibrationStrategy::Uniform);
```

**Commit:** `feat(metrics): calibration_curve, expected_calibration_error`

## Task 5: Multiclass averaging

```
T29 OVR per-class AUCs on the searched fixture:
    AUC[0]=0.0 (support 1), AUC[1]=0.6666666666666667 (support 5), AUC[2]=0.5 (support 2)
T30 macro = 0.3888888888888889, weighted = 0.5416666666666667  (they MUST differ)
T31 perfectly-separable 3-class -> macro == weighted == 1.0
T32 probabilities must sum to 1 per row (allclose), else throw
```

**API:** overloads taking `(y_true_index, y_proba, MulticlassAveraging)`.

**Commit:** `feat(metrics): one-vs-rest multiclass ROC-AUC and average precision`

## Task 6: verify

1. Umbrella header compiles standalone:
   `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
2. Full `make run_tests` — compare the failure list against the baseline
   (the 4 known pre-existing failures in `NOT_FIXED.md`).
3. Determinism: run `test_metrics` 3×, expect identical output.

## Mutation testing (mandatory)

Because FD cannot verify a scoring rule (they are not gradient-carrying
operators), prove the tests are non-vacuous by mutation. Baseline snapshot
taken ONCE **before** any mutation, asserted pristine, each `sed` verified with
`diff -q` before building, restored between mutations. Compile failure is
reported as `MUT-BROKEN`, never as `MUT-CAUGHT`.

| # | Mutation | Must be caught by |
|---|---|---|
| M1 | drop the `+inf`/`(0,0)` prepend in `roc_curve` | T1, T2 |
| M2 | make `binary_clf_curve` emit one point per SAMPLE instead of per distinct score | T6 (tied scores), T1 |
| M3 | replace the AP step integral with `auc_trapezoid(recall, precision)` | **T5 only** |
| M4 | change `scale_by_half` auto rule from `cols<3` to `cols<2` | T16 only |
| M5 | remove the `[eps, 1-eps]` clip in log-loss | T18 (returns −inf) |
| M6 | change `searchsorted` side from left to right | T25 only |
| M7 | stop dropping empty calibration bins | T24 only |
| M8 | change weighted averaging to divide by `num_classes` | T30 only |

M3, M4, M6, M7 and M8 are each pinned by exactly one test; that is intentional
and is the reason those tests exist.