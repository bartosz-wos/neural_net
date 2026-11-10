# Rank Correlation and Gaussian-Copula Utilities — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship a model-free rank-correlation and Gaussian-copula utility module (`include/nn/utils/rank_correlation.{h,cpp}`) that measures the agreement between two rankings (Spearman's rho, Kendall's tau-b), provides the marginal-free copula correlation, and exposes the four exact population correspondences that tie a rank correlation to the underlying Gaussian copula parameter.

**Architecture:** free functions over `Tensor` in `nn::utils`, matching the shape of `utils/sinkhorn.h`, `utils/crf.h`, `utils/conformal.h`, `utils/gaussian_process.h` and `utils/off_policy_eval.h` — no classes, no `Layer`, no randomness. The caller supplies two samples; the module returns a scalar statistic. The one piece of machinery it composes rather than reimplements is the inverse normal CDF, which **already ships** in `utils/randomized_smoothing.h` as `normal_quantile` (measured there to agree with `scipy.stats.norm.ppf` to 4.5e-16 relative across `p ∈ [1e-12, 1-1e-12]`).

**Tech Stack:** C++17, the repo's `Tensor` (row-major, `AlignedAllocator<double>`), the project's own `-O2 -march=native -Wall -Wextra` build, a float64 numpy/scipy oracle in `tests/`, an oracle-mutation harness in `tests/`, and a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

```
grep -rinE 'copula|rank.?correl|spearman|kendall' include/ tests/
-> 0 content hits

grep -rinE '\bpearson\b|\bcorrelation\b' include/nn/utils/*.h
-> only "Clopper-Pearson" (a binomial bound, a DIFFERENT object)
```

The repo has ~250 layers and 60+ utility files and **no correlation machinery of any kind** — not Pearson's r, not Spearman's rho, not Kendall's tau, and nothing copula-shaped.

What it has is adjacent but different:

| shipped | what it measures | why it is not this feature |
|---|---|---|
| `utils/metrics.h` | ROC-AUC, AP, NDCG, MAP, MRR | the **quality** of a model's ranking against labels, not the **agreement between two rankings** |
| `utils/self_supervised_losses.h` | a cross-correlation **matrix** | a Gram matrix, not a rank statistic |
| `utils/linear_algebra.h` | the Frechet distance | a distance between two **Gaussians**, not a rank statistic |

**The `## Ideas` list was drained a NINETEENTH time** (the docs-gap grep returned its 23 familiar hits for the NINTH consecutive run, all accounted for in shipped headers), so the session fell back to the queue's own documented method — a verified-absent concept sweep — and popped the **LAST** bullet of that sweep.

## Scope boundary (stated so it cannot creep in)

**v1 is the BIVARIATE, MODEL-FREE case.** Two samples in, one scalar out.

**OUT OF SCOPE FOR v1, stated so it cannot creep in:**
- **Multivariate** rank correlation (Kendall's W, Spearman's footrule as a multivariate functional). A different data layout.
- **Hypothesis tests / p-values.** The asymptotic normal approximations have their own tie corrections, and the exact permutation test is a different order of complexity. The *statistics* ship; the *tests* do not.
- **Any `Layer`, any randomness, any model.** Nothing here touches a network.
- **Blomqvist's beta.** See "The one thing this module deliberately does NOT ship" below.

## The mathematics, and the three places it is easy to get wrong

**1. RANKS ARE MID-RANKS.** A tied group receives the **average** of the ranks it spans (R's `ties.method="average"`, scipy's `rankdata(method="average")`). Not the first rank of the group, not the last, not the dense rank.

**2. THE CLASSIC SPEARMAN SHORTCUT IS WRONG WITH TIES.** `rho = 1 - 6*sum(d^2)/(n*(n^2-1))` is exact only when there are **no ties at all**. With ties the definition is *Pearson's r of the mid-ranks*, and the two differ. The suite pins both: the shortcut **agrees** on a tie-free fixture and **differs** on a tied one.

**3. KENDALL'S TAU-B'S TIE CORRECTION, AND THE JOINT TIES.** With `tot = n(n-1)/2`, `xtie` = pairs tied in x, `ytie` = pairs tied in y, `ntie` = pairs tied in **both**, and `dis` = discordant pairs:

```
con_minus_dis = tot - xtie - ytie + ntie - 2*dis
tau_b         = con_minus_dis / sqrt((tot - xtie) * (tot - ytie))
```

**The joint ties count in BOTH `xtie` AND `ytie`.** This is measured, not asserted: the oracle's first draft counted them in neither, and the discrepancy against `scipy.stats.kendalltau(variant="b")` was immediate and systematic (e.g. `-0.045454545454545456` against `-0.04761904761904762`).

**THE FOUR EXACT POPULATION CORRESPONDENCES.** For a bivariate **Gaussian** with correlation `rho` (equivalently, the Gaussian copula with parameter `rho`):

```
Kendall's tau   tau   = (2/pi) * arcsin(rho)      rho = sin(pi * tau / 2)
Spearman's rho   rho_S = (6/pi) * arcsin(rho/2)    rho = 2 * sin(pi * rho_S / 6)
```

**Each closed form is verified in the oracle against an INDEPENDENT numerical computation that does not contain it:**
- `tau` against `E[sign(U)sign(V)]` by **1-D conditional quadrature** of the bivariate normal concordance probability.
- `rho_S` against `12 * E[Phi(X)Phi(Y)] - 3` by **Gauss-Hermite quadrature**, with the 2-D and 1-D forms cross-checked against each other and a self-check at `rho = 0` (where independence makes the answer exactly 0).

A closed form checked against a quadrature that does not contain it is evidence. A closed form checked against itself is not.

## The one thing this module deliberately does NOT ship, and why

**Blomqvist's beta is AMBIGUOUS IN THE LITERATURE and is not shipped under that name.**

- The `wdm` R package (Thomas Nagler, MIT) — which the maintained `easystats/correlation` package **delegates to** for its `"blomqvist"` method (`wdm::wdm(x, y, method = "blomqvist")`) — implements it as a **median-quadrant** statistic:
  `beta = 2 * (mass in the lower-left + upper-right quadrants about the medians) / (total mass) - 1`.
  Read from `wdm/inst/include/wdm/bbeta.hpp` in CRAN `wdm_0.2.6`.
- Other sources describe a **normal-scores** ("medial") correlation.

These are **different functions of the data**. Shipping one of them under the name "Blomqvist" would ship a definition this repo cannot justify.

**The normal-scores correlation IS shipped — under the name `gaussian_copula_correlation`, which says exactly what it computes** — and the median-quadrant variant is not shipped at all, because nothing in the repo needs it.

## The property that makes the copula half worth building

**A copula statistic is INVARIANT under any strictly increasing transformation of either marginal.** That is checkable to machine precision, and it is exactly what separates a copula statistic from a linear one:

- Pearson's r of the **raw** values is NOT invariant (a monotone squashing of one marginal changes it).
- Pearson's r of the **normal scores** IS invariant, because the normal-scores transform is a monotone function of the marginal.

The suite asserts the invariance AND asserts that the raw-Pearson foil **fails** it, so the property is not vacuous.

---

## Tasks

### Task 1: The float64 numpy oracle — the specification

**Objective:** Write `tests/rank_correlation_oracle.py`, the float64 numpy specification, and run it green.

**Files:**
- Create: `tests/rank_correlation_oracle.py`

**Step 1: Write the oracle**

Seven sections, in this order:

| § | content |
|---|---|
| 1 | mid-ranks — hand fixture, `1..n` for tie-free, cross-check vs `scipy.stats.rankdata(method="average")`, the `min`-tie foil, a 4-way tie |
| 2 | Spearman — cross-check vs `scipy.stats.spearmanr`, the d² shortcut agreeing (tie-free) and **differing** (tied), monotone invariance, symmetry, `rho(x,x)=1`, `rho(x,-x)=-1`, range |
| 3 | tau-b — cross-check vs `scipy.stats.kendalltau(variant="b")` on 2000 tie-heavy cases, the joint-tie foil, the no-tie reduction to `(C-D)/tot`, invariance, symmetry, extremes, range |
| 4 | normal scores — the plotting-position **round-trip** `Phi(z_i) == (r_i - 0.5)/n`, the Hazen/Blomqvist/Weibull position foils, order preservation, symmetry, asymptotic mean/variance |
| 5 | the copula correlation — **marginal-freeness** + the raw-Pearson foil that fails it, recovery of `rho` on a large bivariate normal, convergence of the rank statistics to their population values |
| 6 | the four population correspondences — quadrature verification, the inverse round-trips, the endpoints, monotonicity, the factor-2 and dropped-`/2` foils |
| 7 | the pinned fixture, emitted as C++ literals |

**Step 2: Run the oracle**

Run: `/usr/bin/python3 tests/rank_correlation_oracle.py`
Expected: `=== SUMMARY: 96 passed, 0 failed ===`, exit 0.

**NOTE:** `/usr/bin/python3` has numpy 2.3.5 + scipy 1.17.1. The `python3` on `PATH` is the Hermes venv, which has numpy but **no scipy** — use the absolute path.

**Step 3: Commit**

```bash
git add tests/rank_correlation_oracle.py
git commit -m "test(utils): float64 oracle for the rank correlation and Gaussian copula utilities"
```

### Task 2: Measure the oracle's own sensitivity

**Objective:** Write `tests/rank_correlation_mutate_oracle.py` and require that it catches all seven mutations. A green oracle on its first run proves nothing.

**Files:**
- Create: `tests/rank_correlation_mutate_oracle.py`

**Step 1: Write the harness**

Seven mutations, each breaking the specification in one specific way:

| id | breaks |
|---|---|
| OM1 | mid-ranks use the first rank of a tied group |
| OM2 | tau-b counts joint ties in neither `xtie` nor `ytie` |
| OM3 | the plotting position is Blomqvist's, not Hazen's |
| OM4 | the population tau is `(1/pi)asin(rho)` |
| OM5 | the population Spearman drops the `/2` |
| OM6 | the copula correlation uses the raw values |
| OM7 | Spearman uses the d² shortcut |

**The harness rules this repo has already learned:**
1. The anchor must occur **exactly once** — a twice-occurring anchor makes the edit a coin flip, and a coin flip is not a verdict. The harness asserts the count and reports `MUT-HARNESS-BROKEN` otherwise.
2. The mutated source is compared to the pristine source — a mutation that did not apply is a **harness defect**, not a verdict.
3. A non-zero exit with **zero reported failures** is a crash, not a verdict → `MUT-BROKEN`.
4. A **survivor is a hole in the SPECIFICATION**, never a pass.

**Step 2: Run the harness**

Run: `/usr/bin/python3 tests/rank_correlation_mutate_oracle.py`
Expected: `=== SUMMARY: 7 caught, 0 survived, 0 broken ===`, exit 0.

**Step 3: Commit**

```bash
git add tests/rank_correlation_mutate_oracle.py
git commit -m "test(utils): mutation harness for the rank correlation oracle"
```

### Task 3: The RED C++ test

**Objective:** Write `tests/test_rank_correlation.cpp` and watch it fail to compile because the header does not exist yet.

**Files:**
- Create: `tests/test_rank_correlation.cpp`

**Step 1: Write the test**

Sections mirroring the oracle, plus:
- the **guard** checks (empty input, length mismatch, `n < 2`, a constant input) — every guard's **message** is asserted, not merely that something was thrown, because a substring shared by two guards discriminates nothing;
- the **foil** checks — the wrong variants must be *shown to differ* from the right answer, so a suite that cannot tell them apart cannot pass;
- the pinned fixture's exact values, transferred from the oracle's emitted C++ literals (**not** hand-transcribed — the repo has been burned by hand-transcribed numbers before, see `tests/_dump_fixture.py`).

**Step 2: Run it and watch it fail**

Run: `make build/test_rank_correlation`
Expected: compile error — `nn/utils/rank_correlation.h: No such file or directory`.

**Step 3: Commit**

```bash
git add tests/test_rank_correlation.cpp
git commit -m "test(utils): RED test for the rank correlation and Gaussian copula utilities"
```

### Task 4: The implementation

**Objective:** Write `include/nn/utils/rank_correlation.{h,cpp}` and make the suite green.

**Files:**
- Create: `include/nn/utils/rank_correlation.h`
- Create: `include/nn/utils/rank_correlation.cpp`
- Modify: `include/nn/nn.h` (add the `#include` to the utils block)

**Step 1: The header**

Public API in `namespace nn { namespace utils { ... } }`:

```cpp
// The mid-ranks of a sample: a tied group receives the AVERAGE of the
// ranks it spans. Shape-preserving.
Tensor mid_ranks(const Tensor& x);

// Pearson's r. Returns NaN if either sample is constant.
double pearson_r(const Tensor& x, const Tensor& y);

// Spearman's rho: Pearson's r of the mid-ranks.
double spearman_rho(const Tensor& x, const Tensor& y);

// Kendall's tau-b, tie-corrected.
double kendall_tau_b(const Tensor& x, const Tensor& y);

// The normal scores: z_i = Phi^-1((r_i - 0.5)/n).
Tensor normal_scores(const Tensor& x);

// The Gaussian copula correlation: Pearson's r of the normal scores.
// MARGINAL-FREE.
double gaussian_copula_correlation(const Tensor& x, const Tensor& y);

// The four exact population correspondences.
double gaussian_copula_tau_from_rho(double rho);
double gaussian_copula_rho_from_tau(double tau);
double gaussian_copula_spearman_from_rho(double rho);
double gaussian_copula_rho_from_spearman(double rho_s);
```

**Shape convention (stated, because it is a decision):** every statistic takes two **1-D samples** given as `(N, 1)` or `(N,)`-shaped tensors of the **same number of elements**. The *values* are what matter; a `(N,1)` and an `(1,N)` tensor holding the same values are the same sample — the same convention `utils/off_policy_eval.h` uses.

**Guards (every message DISTINCT — a shared substring discriminates nothing):**
- empty sample;
- the two samples have different lengths;
- fewer than 2 elements (a correlation needs 2 points);
- a **constant** sample. This one is NOT an error: Pearson's r of a constant sample is genuinely **undefined** (0/0), and `scipy.stats.spearmanr` returns NaN for it. The module returns NaN, matching the reference, and says so in the header. **A NaN is the honest answer here; throwing would be a lie about the mathematics.**

**Step 2: The implementation**

- `mid_ranks`: sort indices, walk equal-value runs, assign the average of the spanned ranks.
- `pearson_r`: two-pass centred form (subtract the mean first) — the one-pass `sum(xy) - sum(x)sum(y)/n` form loses catastrophic precision on large-offset data.
- `spearman_rho`: `pearson_r(mid_ranks(x), mid_ranks(y))`.
- `kendall_tau_b`: the direct O(n²) pair loop, accumulating `con`, `dis`, `xtie`, `ytie`, `ntie` per the formula above.
- `normal_scores`: `normal_quantile((r - 0.5)/n)` per element, **composing** `normal_quantile` from `utils/randomized_smoothing.h` rather than hand-rolling a probit.
- `gaussian_copula_correlation`: `pearson_r(normal_scores(x), normal_scores(y))`.
- the four correspondences: `asin`/`sin` with the `2/pi`, `6/pi`, `/2` factors as above.

**The composition of `normal_quantile`:** it is declared at **global scope** in `utils/randomized_smoothing.h` (that header has no namespace — measured, `grep -c 'namespace'` → 0). So `rank_correlation.cpp` includes it and calls `normal_quantile` unqualified. **The alternative — copying the bisection into this module — would be exactly the "hand-roll a standard primitive" mistake the repo's own debugging notes warn about.**

**Step 3: Build and run**

Run: `make build/test_rank_correlation && ./build/test_rank_correlation`
Expected: `=== SUMMARY: N passed, 0 failed ===`, exit 0.

**Step 4: Verify the umbrella header compiles alone**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
Expected: no output, exit 0. **A focused test that includes only the new header proves the new TU works in isolation; it does NOT prove the library's public include surface composes.**

**Step 5: Commit**

```bash
git add include/nn/utils/rank_correlation.h include/nn/utils/rank_correlation.cpp include/nn/nn.h
git commit -m "feat(utils): the rank correlation and Gaussian copula utilities"
```

### Task 5: The mutation harness for the C++ suite

**Objective:** Prove `tests/test_rank_correlation.cpp` actually catches the defects it was written for.

**Files:**
- Create: `tests/mutate_rank_correlation.sh`

**Step 1: Write the harness**

Mutations targeting the decisions above:

| id | breaks | caught by |
|---|---|---|
| M1 | the mid-ranks use the first rank of a tied group | §1 |
| M2 | tau-b counts joint ties in neither `xtie` nor `ytie` | §3 |
| M3 | the normal-scores plotting position is Blomqvist's | §4 |
| M4 | the population tau is `(1/pi)asin(rho)` | §6 |
| M5 | the population Spearman drops the `/2` | §6 |
| M6 | the copula correlation uses the raw values | §5 |
| M7 | Spearman uses the d² shortcut | §2 |
| M8 | the length-mismatch guard is removed | the guards |
| M9 | the constant-sample guard returns 0 instead of NaN | the guards |
| M10 | the `pearson_r` one-pass form is used | §2 |

**The harness rules this repo has already learned** (all of them are real, documented failures in this repo's history):
1. `restore()` must **never** delete its backup — it is called once per mutation and once on EXIT, so a self-cleaning restore works exactly once and leaves a **mutant in the tree**. Cleanup happens exactly once, on EXIT, and restore is **atomic** (`cp tmp && mv -f`).
2. The link set comes from `make print-lib-objs`, **never** from regex-parsing the Makefile. `LIB_OBJS` is a recursive make variable whose literal text contains no `.o` filenames, so a regex yields an empty set and every mutation scores a false "did not link". The count is asserted `>= 10`.
3. **The mutated object must not be in its own link set.** It legitimately IS in the library's list (it is the TU being mutated), so the mutated TU is compiled **separately** and swapped in for the library's copy.
4. `MUT-BROKEN` means "the build failed". It is a **separate class**, never counted as a pass.
5. `rc >= 128` is a **crash**, which is a **third** verdict class — not evidence about the suite.
6. `rm -f` the intermediates before every build (stale `.o` files produce phantom failures).
7. **Never run a build while the harness is running** — two harnesses that mutate the same source file race on `build/*.o`.

**Step 2: Run the harness**

Run: `bash tests/mutate_rank_correlation.sh`
Expected: every mutation CAUGHT; 0 survived; 0 broken; 0 crashed.

**Step 3: Commit**

```bash
git add tests/mutate_rank_correlation.sh
git commit -m "test(utils): mutation harness for the rank correlation utilities"
```

### Task 6: Wire it into the build and run the full suite

**Objective:** Make the suite part of `make run_tests`, prove no regressions, and check it under the sanitizers.

**Files:**
- Modify: `Makefile` (the `$(BUILD_DIR)/test_rank_correlation` link rule, the `tests:` prerequisite list, and the `run_tests` line)

**Step 1: Makefile** — add the link rule, add the binary to `tests:`, add the `run_tests` line.

**Step 2: Verify the Makefile edit landed in the RIGHT place**

Run: `git diff Makefile`
Expected: exactly three hunks, all about `test_rank_correlation`. **A fuzzy patch on a Makefile can silently destroy a sibling target** — the `all:` and `tests:` continuations both end with the same `\` line, and this repo has been bitten by exactly that.

**Step 3: ASan + UBSan**

Run: `make BUILD_DIR=build_asan build_asan/test_rank_correlation && ./build_asan/test_rank_correlation`
Expected: clean, exit 0, no sanitizer output. **A sanitizer needs its OWN object tree** — linking an ASan-instrumented `.o` against a non-sanitizer link set fails with a wall of `undefined reference to __asan_report_*`, which looks like a source bug and is not one.

**Step 4: The full suite**

Run: `make run_tests`
Expected: **exactly one failing suite, `test_neural_ode`** — the pre-existing, deferred ODERNN gradient failure recorded in `NOT_FIXED.md` and re-confirmed on clean master. **Zero NEW failures.**

**Step 5: Commit**

```bash
git add Makefile
git commit -m "build: wire the rank correlation suite into the test target"
```

### Task 7: Record it

**Objective:** Move the bullet to `## Done` and note what was shipped.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Update the queue**

Move the "RANK CORRELATION / COPULA UTILITIES" bullet out of the verified-absent sweep into `## Done`, with the one-line summary and the test results, and record:
- the queue was drained a **NINETEENTH** time;
- the docs-gap grep returned its 23 familiar hits for the **NINTH** consecutive run;
- the **Blomqvist ambiguity** finding, so a future session does not "discover" it and ship a guess;
- the **joint-tie convention** finding, which cost the oracle its first draft.

**Step 2: Commit**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: record the rank correlation and Gaussian copula utilities"
```

---

## Verification checklist (all of it, before the work is called done)

- [ ] The oracle is green: `96 passed, 0 failed`.
- [ ] The oracle-mutation harness reports `7 caught, 0 survived, 0 broken`.
- [ ] The C++ suite is green.
- [ ] The C++ mutation harness reports every mutation CAUGHT.
- [ ] The umbrella header compiles alone.
- [ ] ASan + UBSan clean.
- [ ] `make run_tests` shows **zero new failures**.
- [ ] Every claim above was checked with a real tool output in this session.

## The single most likely way to get this wrong

**Shipping the wrong tie convention.** Three separate ones are in play — mid-ranks, tau-b's tie correction, and the plotting position — and each is individually invisible on a fixture that does not exercise it. The suite's job is to have a fixture that exercises **all three at once**, which is why the pinned fixture carries **joint ties** and the oracle's §3 cross-checks 2000 tie-heavy cases against scipy.
