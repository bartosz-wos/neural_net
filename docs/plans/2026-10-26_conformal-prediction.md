# Conformal Prediction Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add **split conformal prediction** — distribution-free coverage guarantees for this repo's classifiers and regressors — as `include/nn/utils/conformal.{h,cpp}`, with the Adaptive Prediction Set (APS) for classification and Conformalized Quantile Regression (CQR) for regression.

**Architecture:** A new free-function utility file in `nn::utils`, model-free by construction: every core is a free function over `Tensor`, so the whole transform is testable with no `Model`, no optimizer and no network — the same shape `utils/sinkhorn.h`, `utils/hutchinson.h`, `utils/mup.h` and `utils/weight_std.h` were built with. Two score functions, one calibration rule, two prediction-set builders.

**Tech Stack:** C++17, the repo's `Tensor` class, the existing `utils/logit_processing.h` `softmax`, a numpy float64 oracle, and the repo's mutation harness.

---

## 0. The gap, measured (not assumed)

The queue's `## Ideas` list was drained a **sixteenth** time. Its standing **docs-gap grep** ran first:

```
grep -rinE 'not entitled|out of scope|deliberately not|is not exposed|for now we do not|this repo does not|no hook' include/nn --include=*.h
-> 19 hits, all already accounted for in shipped headers
```

**Zero new hits — the sixth consecutive run.** The queue itself predicts this decay: "each of the previous three was a file THIS QUEUE had just written. Its yield decays as the file it reads gets fixed."

The fallback was a **concept sweep** over families this repo does not ship, each measured BY CONTENT (matching lines printed, not filenames):

```
grep -rinE 'conformal|coverage.guarantee|exchangeab' include/ tests/
-> 2 hits, BOTH incidental:
   include/nn/utils/temperature_scaling.h:27  // "metrics SCORES a fixed prediction set, whereas ..."
   tests/test_randomized_smoothing.cpp:285    // a comment about "the coverage guarantee a valid"
```

**ZERO IMPLEMENTATION.** Not a header, not a class, not a function.

### Why it is not redundant

The repo already ships a rich **calibration** suite and the queue's rule is that an absence proves a feature is missing, not that it belongs — so redundancy was *measured*, not argued:

| Shipped | What it computes | Why it is NOT conformal prediction |
|---|---|---|
| `utils/metrics.h` — ECE / ACE / TACE / classwise-ECE | how close probabilities are to frequencies | **descriptive**, not a guarantee; no calibration set, no quantile, no set |
| `utils/temperature_scaling.h` | one scalar `T` fitted on held-out data | **parametric**; assumes the model is already right and only mis-scaled |
| `utils/randomized_smoothing.h` | a **certified L2 radius** around the prediction | a different object entirely (an L2 ball in input space, not a set of labels) |
| `utils/distribution_losses.h` — `QuantileLoss` (pinball) | the **loss** CQR trains with | the loss is shipped; **the calibration that uses it is not** |

**`QuantileLoss` is the strongest evidence that CQR belongs here**: the repo already ships the exact training loss CQR requires (`utils/distribution_losses.h:118`) and nothing consumes it for uncertainty quantification. That is the "a file names its own missing piece" shape — a shipped primitive with no consumer.

Conformal prediction is also **the** distribution-free uncertainty primitive, and this repo's whole `utils/` directory is model-free post-processing utilities. It composes with the shipped `ngram_lm` (which has a perplexity but no coverage guarantee) and with `utils/logit_processing.h` (which reshapes logits into a distribution — the exact input APS needs).

### Citations, verified by TITLE and by abs-page meta scrape (2026-10-26)

This repo's expansion queue has produced **thirteen wrong arXiv IDs**. Every citation below was verified by scraping `arxiv.org/abs/<id>` for its `citation_title` / `citation_author` meta tags:

| id | title (verbatim from meta) | authors (verbatim, in order) |
|---|---|---|
| `1905.03222` | "Conformalized Quantile Regression" | Romano, Yaniv; Patterson, Evan; Candès, Emmanuel J. |
| `1604.04173` | "Distribution-Free Predictive Inference For Regression" | Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry |
| `2006.02544` | "Classification with Valid and Adaptive Coverage" | Romano, Yaniv; Sesia, Matteo; Candès, Emmanuel J. |
| `2107.07511` | "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" | Angelopoulos, Anastasios N.; Bates, Stephen |

The **algorithms were transcribed from the papers' own HTML**, not from memory:
- `https://arxiv.org/html/2107.07511v3` (593 KB, HTTP 200) — the APS score (Eq. 4), the APS prediction set (Eq. 5), the CQR score (Eq. 6), the CQR interval (Eq. 7) and the coverage theorem (Eq. 64).
- `https://arxiv.org/abs/1905.03222` for the CQR paper's identity, and `https://arxiv.org/abs/2006.02544` for the APS paper's identity.

**NO arXiv ID is asserted for anything else.** In particular the "Conformal Prediction" original paper (Vovk, Gammerman, Shafer) and the "Classification with Valid and Adaptive Coverage" journal venue are not claimed beyond what the meta tags verified.

---

## 1. The semantics, and every one of them is a place a reimplementation goes wrong

### 1.1 The calibration quantile — the `+1` is load-bearing

The theorem (Eq. 64, read from the tutorial's HTML) is

```
q̂ = inf { q : |{i : s(X_i, Y_i) ≤ q}| / n ≥ ⌈(n+1)(1-α)⌉ / n }
```

and for **distinct** calibration scores, `q̂ = s_{⌈(n+1)(1-α)⌉}` — the `⌈(n+1)(1-α)⌉`-th smallest score.

**Traps:**

- **T1. The `+1` is not optional.** `⌈(n+1)(1-α)⌉` and `⌈n(1-α)⌉` differ whenever the correction crosses an integer, and the difference is a **full calibration point** — i.e. the coverage guarantee changes.
- **T2. `⌈(n+1)(1-α)⌉` CAN EXCEED `n`.** With `n = 9, α = 0.1`: `⌈10 · 0.9⌉ = 9` — OK. But with `n = 4, α = 0.1`: `⌈5 · 0.9⌉ = 5 > n = 4`. Then the quantile is **`+∞`**, and the prediction set is the **entire label space**. This is not a bug; it is the honest consequence of "not enough calibration data to certify 90% coverage". A implementation that silently clamps the index to `n` returns a finite threshold and **breaks the guarantee**.
- **T3. The quantile is an ORDER STATISTIC, not an interpolation.** With distinct scores the answer is exactly the `k`-th smallest. An implementation that interpolates (numpy's default `linear`) returns a value between two scores and is a **different rule**.
- **T4. Ties are real and the `inf` form matters.** With ties, `q̂` is the smallest `q` such that the fraction of scores `≤ q` reaches the threshold — i.e. the **`k`-th smallest score, taken as a value, with ties broken by VALUE not by index**. So `q̂ = k-th smallest` holds even with ties (it is the same number), and this is what the implementation must return.
- **T5. The `inf` form is the same as the `k`-th smallest** *provided* `k = ⌈(n+1)(1-α)⌉ ≤ n`. With `k > n` the `inf` form gives `+∞` and the `k`-th smallest is undefined. The implementation must produce `+∞` in that case.

### 1.2 APS — the score is a CUMULATIVE SUM, and the set is built by CUMULATIVE MASS

From the tutorial's Eq. 4–5, with `π` the permutation of `{1..K}` sorting the softmax output from most to least likely:

```
score(x, y) = Σ_{j=1}^{k} f̂(x)_{π_j}   where y = π_k
T(x)       = { π_1, ..., π_k }   where k = inf{ k : Σ_{j=1}^{k} f̂(x)_{π_j} ≥ q̂ }
```

**Traps:**

- **T6. The score is a CUMULATIVE sum, not a probability.** `score(x,y) = 1 - f̂(x)_{y}` (the naive score) is a **different rule** that produces **larger** sets on the same data. The suite must pin the difference.
- **T7. The score for the `k`-th ranked class is `cumsum[k]`, the SUM THROUGH that class.** So the score of the most likely class is `f̂(x)_{π_1}`, not `0` and not `f̂(x)_{π_1} - something`.
- **T8. The score is `1.0` for the LEAST likely class** (the full cumulative sum), so `q̂ = 1.0` is achievable and `q̂ = 1.0` means "include every class".
- **T9. The prediction set is built by the SAME cumulative sum.** `k = inf{ k : cumsum[k] ≥ q̂ }`, and the set is `{π_1..π_k}`. **The crossing token is KEPT** (the `≥`, not `>`). A `>` implementation drops a class.
- **T10. Ties in the SORT are broken by class index.** The suite's fixture must have a tie to pin this, and the tie-break must be deterministic.

### 1.3 CQR — the score is a MAX of two one-sided gaps

From the tutorial's Eq. 6–7, with `t̂_{α/2}` and `t̂_{1-α/2}` the lower and upper quantile predictions:

```
score(x, y) = max{ t̂_{α/2}(x) - y ,  y - t̂_{1-α/2}(x) }
T(x)       = [ t̂_{α/2}(x) - q̂ ,  t̂_{1-α/2}(x) + q̂ ]
```

**Traps:**

- **T11. The score is a MAX, not a sum and not an absolute value.** `|y - mid|` is a different rule.
- **T12. The interval is `lower - q̂` to `upper + q̂`** — the calibration error is added to BOTH sides. An implementation that adds it to only one side is wrong, and an implementation that makes the interval `[lower - q̂, upper + q̂]` where the halves are *not* the same quantiles is wrong.
- **T13. `α/2` and `1-α/2` are the two quantile LEVELS**, so the conformal level `α` is split across the two tails. A user asking for `α = 0.1` gets the 5% and 95% quantiles.

### 1.4 The coverage guarantee is a DISTRIBUTION-FREE statement about the CALIBRATION SET

- **T14. `α` is the MISCoverage rate**, so the guarantee is `P[Y_test ∈ T(X_test)] ≥ 1-α`. An implementation that flips the sign is wrong.
- **T15. Calibration and test must be SEPARATE.** The guarantee is exchangeability-based; reusing training points as calibration invalidates it. The API must make the split explicit (calibrate on one set, predict on another) — never a single "fit and score" call on the same data.

---

## 2. API sketch

```cpp
namespace nn { namespace utils {

// ---- the calibration rule -------------------------------------------------
// k = ceil((n+1)*(1-alpha));  +inf when k > n.
size_t conformal_quantile_index(size_t n, double alpha);
bool  conformal_quantile_is_infinite(size_t n, double alpha);

// The order statistic itself. `scores` is (n,1) or (n,) of calibration scores.
// Returns +inf when k > n.
double conformal_quantile(const Tensor& scores, double alpha);

// ---- score functions ------------------------------------------------------
// APS: cumulative softmax mass through the true class.
// `probs` is (n,K) softmax output; `labels` is (n,1) class indices.
Tensor aps_score(const Tensor& probs, const Tensor& labels);
// CQR: max{ lower - y, y - upper }; `lower`/`upper` are (n,1), `y` is (n,1).
Tensor cqr_score(const Tensor& lower, const Tensor& upper, const Tensor& y);

// ---- prediction sets ------------------------------------------------------
// APS: cumulative-mass threshold. Returns the set as a sorted vector of class ids.
std::vector<int> aps_prediction_set(const Tensor& probs_row, double qhat);
// CQR: the calibrated interval.
Tensor cqr_interval(double lower, double upper, double qhat);   // (1,2)

}}
```

`Tensor` has no 1-D support (it is `(rows, cols)` with `data` flat), so `conformal_quantile` accepts an `(n,1)` column and the helpers take `(n,K)` matrices.

---

## 3. Tasks

### Task 1: The numpy oracle (RED for the C++)

**Objective:** Write `tests/conformal_oracle.py` — the specification. It must run green BEFORE any C++ exists, and it must contain **foils** that prove the C++ suite's assertions are not vacuous.

**Steps:**
1. Write the oracle with sections: (1) the calibration rule, (2) APS scores, (3) APS sets, (4) CQR scores and intervals, (5) **coverage by Monte-Carlo**, (6) **foils**.
2. **The foils are the load-bearing part.** For each wrong variant, assert that it DIFFERS from the correct one on the fixture:
   - naive score `1 - f̂_y` vs APS cumulative score → must differ (T6)
   - `k = ⌈n(1-α)⌉` vs `k = ⌈(n+1)(1-α)⌉` → must differ on some `(n, α)` (T1)
   - clamping `k` to `n` vs `+∞` → must differ when `k > n` (T2)
   - interpolated quantile vs order statistic → must differ (T3)
   - `>` vs `≥` on the crossing → must differ (T9)
   - sum vs max for the CQR score → must differ (T11)
   - one-sided vs two-sided CQR interval → must differ (T12)
3. Run it; it must be green.

### Task 2: The header

**Objective:** `include/nn/utils/conformal.h` with the semantics above written out, in the repo's house style (the `weight_std.h` / `sinkhorn.h` header-comment discipline: the gap, the citations, and every trap).

### Task 3: The implementation

**Objective:** `include/nn/utils/conformal.cpp`, green against the oracle's numbers.

### Task 4: The C++ test suite

**Objective:** `tests/test_conformal.cpp`, asserting the oracle's numbers, with the foils asserted as *differences* so the checks are not vacuous.

### Task 5: The mutation harness

**Objective:** `tests/mutate_conformal.sh`, copied in structure from `tests/mutate_weight_std.sh` (which already encodes this repo's twenty-plus harness defects).

### Task 6: Build wiring + full suite + docs

**Objective:** Makefile entries, `nn.h` include, `EXPANSION_QUEUE.md` move to Done, commit + push.

---

## 4. Verification plan

- Oracle green before C++ exists.
- C++ suite green, and **every foil asserted as a difference** (not just "the right answer is right").
- Mutation harness: 0 survived, 0 broken, 0 noop. Any survivor that is provably inert gets an **inverted-verdict** harness function.
- ASan + UBSan clean.
- Zero warnings at the project's real `-O2 -march=native -Wall -Wextra`.
- Umbrella header compiles standalone: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`.
- Full `make run_tests`: the ONLY failure must be the `test_neural_ode` entry already deferred in `NOT_FIXED.md`, measured identically on the pre-change baseline.
