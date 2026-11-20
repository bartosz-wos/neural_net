# Categorical Agreement and Contingency-Table Inference — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship the categorical-agreement and categorical-association statistics as model-free utilities over `Tensor` — **Cohen's kappa** (chance-corrected agreement, with unweighted / linear / quadratic disagreement weights), the **Matthews correlation coefficient** (the imbalance-robust multiclass metric), the **Pearson chi-square test of independence** on an R×C contingency table, the **G-test** (the likelihood-ratio sibling on the same table), **McNemar's test** for paired binary outcomes, and **Fisher's exact test** for a 2×2 table — for a repo that has none of them.

**Architecture:** a new `include/nn/utils/contingency.{h,cpp}` in the same shape as `utils/survival.{h,cpp}`, `utils/rank_correlation.h` and `utils/sample_quality.h` — every core is a free function over `Tensor`, no `Layer`, no `Model`, no randomness, no optimizer. The module's numerical engine is a **general chi-square survival function** (the regularized upper incomplete gamma), which the repo has never shipped: `utils/survival.h` says so in as many words.

**Tech Stack:** C++17, the repo's `Tensor`, the project's `-O2 -march=native -Wall -Wextra` build, a float64 numpy+scipy+sklearn oracle in `tests/`, an oracle-mutation harness in `tests/`, a bash mutation harness in `tests/`, and a fixture-splice script.

---

## Why this feature, measured not assumed

The `## Ideas` list was drained for the 30th time (every bullet is marked `[DONE]`), so the session fell back to the queue's own documented method: a **verified-absent concept sweep**, run against the whole neighbourhood rather than by keyword alone.

```
grep -rinE 'cohen|kappa'            include/  -> 6 hits, ALL FALSE POSITIVES
                                            (megalodon's kappa_q / kappa_k are DiT
                                             modulation parameters; randomized_smoothing
                                             cites "Jeremy M. Cohen" the author)
grep -rinE 'matthews|\bmcc\b'        include/  -> 0
grep -rinE 'mcnemar'                 include/  -> 0
grep -rinE 'fisher.?exact'           include/  -> 0
grep -rinE 'contingency'             include/  -> 0
grep -rinE 'cramer|phi.?coefficient' include/  -> 0
grep -rinE 'likelihood.?ratio'       include/  -> 0
grep -rinE 'chi.?square|chisq|chi2'  include/  -> 3 hits, ALL the df=1 closed form
                                            in utils/survival.{h,cpp}
grep -rinE 'incomplete.?gamma'       include/  -> 3 hits, ALL comments saying the repo
                                            does NOT have one
```

**ZERO IMPLEMENTATION.** The repo ships ~250 layers and 60+ utility files and no categorical-agreement or categorical-association machinery at all.

### It is not redundant with the shipped neighbours

- `utils/metrics.h` — accuracy, F1, ROC-AUC, ECE, NDCG, MRR. Every one is a **score-vs-label** statistic. None of them compares two **labelings of the same items**, which is what kappa and McNemar do, and none of them corrects for the agreement two raters would reach **by chance**.
- `utils/rank_correlation.h` — Pearson / Spearman / Kendall on **continuous** quantities. There is no categorical analogue of association anywhere in the repo.
- `utils/survival.h` — the log-rank test's p-value is the **chi-square-1** survival function `erfc(sqrt(x/2))`, which is a closed form that exists **only** for one degree of freedom. A 2×2 contingency table has df = 1 and fits; a 2×3 table has df = 2 and does **not**. The repo cannot currently produce the p-value for any table bigger than 2×2.
- `utils/conformal.h` — the coverage of a **set**. No notion of two raters, of a contingency table, or of chance agreement.

### The chi-square p-value for df > 1 is the repo's own NAMED missing piece

`utils/survival.h` states it explicitly:

> *"k > 2 GROUPS IS OUT OF SCOPE ... the statistic generalises to k-1 degrees of freedom, but its p-value does not have a closed form in erfc, and shipping a general incomplete gamma for one extra function is a scope decision this file does not make silently."*

That is the queue's highest-signal kind of absence: **a file that names its own missing piece**. One extra function was not worth an incomplete gamma; a whole family of contingency-table tests is. The general chi-square survival function ships here, and the df = 1 case keeps the repo's existing exact closed form so the two code paths agree **bit for bit**.

---

## The mathematics, and the places it is easy to get wrong

Every convention below was MEASURED against `scikit-learn 1.9.0`, `scipy 1.17.1` and `R 4.5.3` on this host, not recalled. The discriminating fixture for each trap is named.

### The general chi-square survival function

    P(X >= x)  for  X ~ chi2(k)   =   Q(k/2, x/2)

`Q(a, x)` is the regularized upper incomplete gamma. The implementation is the
classical series / continued-fraction pair: the **series** for `P(a, x)` when
`x < a + 1`, the **continued fraction** for `Q(a, x)` otherwise, with
`exp(-x + a*log(x) - lgamma(a))` factored out of both.

    df = 1  ->  the exact closed form erfc(sqrt(x/2))

**C1. THE df = 1 PATH IS THE REPO'S EXISTING CLOSED FORM, BIT FOR BIT.** The
shipped `utils/survival.cpp` computes `std::erfc(std::sqrt(statistic / 2.0))`.
The new function returns exactly that for `df == 1`, so the two code paths
cannot drift. The **general** path is a *separate* implementation and is
checked against the closed form at df = 1 to **1e-14** — two independent routes
to the same number, which is what makes the incomplete gamma trustworthy.

Measured against `scipy.stats.chi2.sf` over `df ∈ {1..100}` and
`x ∈ [1e-8, 1e5]`: **worst relative error 2.9e-14**.

### Cohen's kappa

    p_o = (1/N) * sum_i C_ii                    observed agreement
    p_e = sum_i (r_i/N) * (c_i/N)                agreement expected by chance
    kappa = (p_o - p_e) / (1 - p_e)

The **weighted** form replaces the diagonal weight with a **disagreement**
weight `w_ij = |i - j|` (linear) or `(i - j)^2` (quadratic):

    kappa_w = 1 - (sum_ij w_ij C_ij) / (sum_ij w_ij E_ij)

**C2. THE WEIGHT MATRIX IS BUILT OVER THE EXPLICIT CLASS-INDEX SPACE, NOT OVER
THE PRESENT LABELS.** This is a real fork and it changes the answer. Measured
on a fixture whose present classes are `{0, 1, 3}` (class 2 absent):

| weights | over present labels (compressed) | over the explicit index space | differ |
|---|---|---|---|
| unweighted | 0.304347826087 | 0.304347826087 | no |
| linear    | 0.304347826087 | 0.360000000000 | **YES** |
| quadratic | 0.360000000000 | 0.421686746988 | **YES** |

`scikit-learn`'s `labels=None` path compresses; its `labels=[...]` path does
not. The repo's own convention is explicit — `utils/metrics.h`'s
`confusion_matrix` takes an explicit `num_classes` — so **this module uses the
explicit class-index space**, and the fixture above is the one that pins it.
(The unweighted case is space-independent, which is why only the weighted rows
differ; the table above is the evidence, not a guess.)

**C3. THE UNDEFINED CASE IS NaN.** When the two raters share only one label
the expected agreement is 1 and the denominator vanishes. `scikit-learn`
returns `NaN` and raises an `UndefinedMetricWarning`; `utils/metrics.h` uses
`NaN` for its undefined metrics too. The kappa result therefore carries
`NaN` for `kappa` plus the `p_o` / `p_e` components so a caller can see
*why*.

**C4. scikit-learn's `expected` MATRIX IS TRANSPOSED RELATIVE TO THE
CONVENTIONAL ONE — AND IT DOES NOT MATTER, FOR A PROVABLE REASON.**
`sklearn` builds `expected = outer(col_sums, row_sums) / N`, i.e. the
transpose of the textbook `E_ij = r_i c_j / N`. Every weight matrix this
module supports (the off-diagonal indicator, `|i-j|`, `(i-j)^2`) is
**symmetric**, and for symmetric `w` the sums `sum_ij w_ij E_ij` and
`sum_ij w_ij E_ji` are equal by relabelling. So the transposition cancels
exactly. The module uses the conventional orientation, and the fixture set
would catch a mistake in either direction.

### Matthews correlation coefficient

    t_k = # true class k        p_k = # predicted class k
    c   = sum_k C_kk              N = total
    cov_ytyp = c*N - sum_k p_k t_k
    cov_ypyp = N^2 - sum_k p_k^2
    cov_ytyt = N^2 - sum_k t_k^2
    MCC = cov_ytyp / sqrt(cov_ytyt * cov_ypyp)

**C5. THE DEGENERATE CASE IS 0.0, NOT NaN — AND scikit-learn's KAPPA
DEGENERATE CASE IS NaN, NOT 0.0.** Two different conventions in one library,
measured:

| | `sklearn.metrics.cohen_kappa_score` | `sklearn.metrics.matthews_corrcoef` |
|---|---|---|
| degenerate | `NaN` (+ `UndefinedMetricWarning`) | **`0.0`** |

The MCC denominator vanishes when either marginal is concentrated (all
predictions in one class, or all truths in one class); `sklearn` returns
`0.0` there, and so does this module. The kappa undefined case returns
`NaN`, and so does this module. **These are not interchangeable and the
fixtures pin both.**

The MCC fixture also has to be built so the **label space does not matter**
(class count invariance is a property of the formula), and the *paired*
class agreement fixture has to be asymmetric, because a symmetric confusion
matrix has a symmetric MCC that a transposed implementation would also
produce.

### Pearson's chi-square test of independence

    E_ij = r_i * c_j / N
    X2   = sum_ij (O_ij - E_ij)^2 / E_ij
    df   = (R - 1) * (C - 1)
    p    = P(chi2_df >= X2)

**C6. YATES' CONTINUITY CORRECTION IS APPLIED ONLY TO A 2×2 TABLE, AND THE
FLAG IS HONOURED ONLY THERE.** `scipy.stats.chi2_contingency` and
`R::chisq.test` agree exactly on this: the correction is applied when
`correct && nrow == 2 && ncol == 2`, and ignored for every larger table.
Measured on the 2×2 `[[10,20],[30,15]]`: corrected `6.752232142857143`,
uncorrected `8.035714285714286`; `scipy` and `R` both report the corrected
value by default and the uncorrected value on request. The module follows
both references exactly: a `correct` flag, defaulting to on, honoured only
for 2×2.

**C6-bis. THE CORRECTION IS **CLAMPED** TO THE CELL'S OWN DEVIATION, AND THE
UNCORRECTED FORM IS WRONG.** This is a correction to the plan's first
draft, found by reading `scipy.stats.chi2_contingency`'s source and
`stats:::chisq.test`'s source rather than by recalling the textbook:

    scipy:  diff = expected - observed
            direction = sign(diff)
            magnitude = minimum(0.5, abs(diff))        <-- the CLAMP
            observed += magnitude * direction
            X2 = sum((observed - expected)^2 / expected)

    R:      YATES <- min(0.5, abs(x - E))               <-- elementwise clamp
            X2 = sum((abs(x - E) - YATES)^2 / E)

Both references therefore compute

    X2_yates = sum_ij ( |O_ij - E_ij| - min(0.5, |O_ij - E_ij|) )^2 / E_ij

which is **NOT** `sum_ij (|O_ij - E_ij| - 0.5)^2 / E_ij` whenever some cell
has `|O - E| < 0.5`. The discriminating fixture is the 2×2
`[[1,1],[1,2]]`, where every cell has `|O - E| = 0.2`:

| formula | value | `scipy` | `R` |
|---|---|---|---|
| `sum((|d| - 0.5)^2/E)` — the UNCLAMPED form | `0.31250000000000006` | — | — |
| `sum((|d| - min(0.5,|d|))^2/E)` — the CLAMPED form | `0.0` | **`0.0`** | **`0.0`** |

Measured over **every** 2×2 table with entries in `0..8` and no empty
row or column (**6272 tables**), `scipy` and `R` agree to **1.8e-15** on
the corrected statistic. The module implements the CLAMPED form.

The clamp matters because in a 2×2 table all four cells share the same
deviation magnitude, `|O - E| = |ad - bc| / N`, so the clamp is either
uniform or not at all — but a table like `[[1,1],[1,2]]` has a deviation
of `0.2`, well under `0.5`, and the unclamped form would report a
statistic of `0.3125` where the references report `0`.

**C6-ter. THE p-VALUE USES `df = (R-1)(C-1)`, NOT `scipy`'s
`power_divergence` DEGREES OF FREEDOM.** `scipy.stats.power_divergence`
computes its own p-value with `df = number_of_cells - 1`, which is
**wrong for a contingency table** (for a 4×3 table it uses 11 where the
contingency df is 6). `scipy.stats.chi2_contingency` uses the correct
`(R-1)(C-1)`; measured over 300 random R×C tables, its p-value agrees
with `chi2.sf(statistic, (R-1)*(C-1))` to **0.0**. The module uses
`(R-1)(C-1)` for the chi-square test, for the G-test, and for
`power_divergence`-style G p-values.

**C6-quater. `scipy.stats.power_divergence`'s `lambda_` IS INVERTED FROM
THE OBVIOUS READING, AND `lambda_=1` IS PEARSON WHILE `lambda_=0` IS THE
G-TEST.** The docstring's parameter order does not match the branch
order in the body. Measured: `power_divergence(obs, exp, lambda_=1)`
equals `sum((O-E)^2/E)` to **0.0**, and `lambda_=0` equals
`2 * sum(O log(O/E))` to **5.6e-16** over 600 random R×C tables. The G
statistic is validated against **two** independent references (below).

**C7. AN EMPTY ROW OR COLUMN MAKES THE EXPECTED COUNT ZERO AND THE STATISTIC
UNDEFINED.** `scipy` raises `ValueError` ("The internally computed table of
expected frequencies has a zero element") and `R` returns `NaN`. The module
**throws** `std::invalid_argument` with its own message, which is the repo's
established convention for invalid input (and `utils/metrics.h`'s).

### The G-test

    G = 2 * sum_ij O_ij * log(O_ij / E_ij)          df = (R-1)(C-1)

**C8. A ZERO OBSERVED CELL CONTRIBUTES 0, NOT `0 * log(0)`.** The term
`O log(O/E)` has limit 0 as `O -> 0`, so a zero cell contributes exactly 0.
This is the classic trap: the naive expression is `0 * (-inf) = NaN` and the
whole statistic is destroyed. The fixture **must** contain a zero cell, or
the mutation that deletes the guard is invisible.

### McNemar's test (paired binary outcomes)

    n01 = #(a = 0, b = 1)        n10 = #(a = 1, b = 0)
    X2  = (|n01 - n10| - 1)^2 / (n01 + n10)          df = 1, corrected
    X2  = (n01 - n10)^2 / (n01 + n10)                df = 1, uncorrected
    p_exact = two-sided binomial test on (n01 + n10) trials at p = 1/2

**C9. THE CONTINUITY CORRECTION IS **NOT** APPLIED WHEN THE TABLE IS
SYMMETRIC — AND THAT IS THE NON-OBVIOUS PART.** `R::mcnemar.test`'s
condition is `correct && (r == 2) && any(x - t(x) != 0)`. So when
`n01 == n10` the correction is **skipped** and the statistic is the
uncorrected `0`, **not** `(0 - 1)^2 / (2*n01)`. Measured on
`[[10,10],[10,10]]`: `R` reports statistic `0`, p-value `1`. A naive
implementation that always subtracts 1 reports `1/20` there. **That is the
discriminating fixture.**

**C9-bis. BUT "SYMMETRIC" IS ABOUT THE **WHOLE** TABLE, NOT JUST THE
OFF-DIAGONAL CELLS, AND `R`'s FORMULA IS NOT THE TEXTBOOK
`|b - c|` FORM.** This is a correction to the plan's first draft, found
by reading `stats:::mcnemar.test`'s source rather than by recalling the
textbook. The source computes, for a 2×2 table `x`:

    if (correct && r == 2 && any(x - t(x) != 0)) y <- abs(x - t(x)) - 1
    else                                         y <- x - t(x)
    x <- x + t(x)
    STATISTIC <- sum(y[upper.tri(x)]^2 / x[upper.tri(x)])

For a 2×2 table `[[a,b],[c,d]]` with `x = [[n00,n01],[n10,n11]]`:
`x - t(x)` has entries `[[0, n01-n10],[n10-n01, 0]]`, so `any(x - t(x) != 0)`
is `n01 != n10` — the correction depends on the off-diagonal cells alone,
and the diagonal cells never enter the statistic. `upper.tri` of a 2×2
picks the single element `(1,2)`, so the statistic is

    corrected:   (|n01 - n10| - 1)^2 / (n01 + n10)
    uncorrected: (n01 - n10)^2       / (n01 + n10)

**The `n01 == n10` case is the one that discriminates**: `R` reports
statistic **0** (the correction was skipped, so the plain difference
squared is 0), whereas the textbook "always subtract 1" form gives
`(-1)^2 / (2*n01) = 1/(2*n01)`, a **nonzero** statistic and a p-value
well below 1. Measured on `[[30,5],[5,25]]` (`n01 == n10 == 5`): `R`
reports statistic `0.00000000`, p-value `1.0`; the textbook form would
report `0.1` and `p = 0.7514`.

**C9-ter. `R`'s CORRECTION IS APPLIED TO THE **OFF-DIAGONAL DIFFERENCE
ONLY**, SO THE RESULT IS `( |n01 - n10| - 1 )^2 / (n01 + n10)` — WITH THE
ABSOLUTE VALUE.** `R` writes `abs(x - t(x)) - 1`, so the numerator is
`|n01 - n10| - 1`, **not** `n01 - n10 - 1`. The two differ whenever
`n01 < n10`: the textbook form would give `(-2)^2 = 4` where `R` gives
`(|n01-n10| - 1)^2`. Measured on `[[0,5],[3,0]]` (`n01 = 5, n10 = 3`): `R`
reports statistic `0.12500000` and p-value `0.723673609832`, and
`(5-3-1)^2/(5+3) = 0.125` — the same, because the absolute value does
not change the **squared** numerator when the difference is positive.
The discriminating case is `n01 < n10`: on `[[0,3],[5,0]]` (`n01 = 3,
n10 = 5`) `R` reports `(|3-5|-1)^2/8 = 0.125`, whereas the signed form
`(3-5-1)^2/8 = 0.5625` is wrong. **The module uses the absolute value.**

**C9-quater. THE `R` SYMMETRY CHECK IS `any(x - t(x) != 0)` — WHICH FOR A
2×2 IS EXACTLY `n01 != n10`.** So the rule is precisely: apply the
continuity correction **iff `n01 != n10`**; otherwise the statistic is
`0`. Measured on four tables that discriminate this:
`[[10,10],[10,10]]` → `0` (symmetric, skipped);
`[[30,5],[5,25]]` → `0` (symmetric off-diagonal, skipped);
`[[30,10],[5,25]]` → `1.06666667` (asymmetric, corrected);
`[[5,20],[10,15]]` → `2.70000000` (asymmetric, corrected). All four match
`R` exactly.

**C10. `n01 + n10 == 0` IS `NaN`, NOT 0.** With no discordant pairs the
statistic is `0/0`. `R` reports `NaN` for both the statistic and the
p-value. Measured on `[[30,0],[0,30]]`.

**C11. THE EXACT p-VALUE IS THE REPO'S **EXISTING** `binom_pvalue_half`.**
The exact McNemar test is a two-sided binomial test at `p = 1/2` on the
`n01 + n10` discordant pairs. `utils/randomized_smoothing.h` already ships
that function, and its own oracle pins it to `scipy.stats.binomtest`. This
module **composes** it rather than hand-rolling a second binomial tail
(`utils/rank_correlation.cpp` sets the precedent for composing across the
utils). The composition is checked against `scipy` in the oracle.

### Fisher's exact test (2×2)

    p_two_sided = sum over the support of  hypergeom.pmf(x)  for every x whose
                   pmf is at most the observed table's pmf (probability ordering)
    odds_ratio  = (a*d) / (b*c)

**C12. THE TWO-SIDED p-VALUE USES THE **PROBABILITY ORDERING** RULE, NOT A
DOUBLED ONE-SIDED TAIL.** Measured against `scipy.stats.fisher_exact` on
eight tables including three with ties: the probability-ordering rule agrees
to **0.0** on every one, while the doubled-tail rule does not. The
probability-ordering rule is also `R::fisher.test`'s.

**C13. THE ODDS RATIO IS THE UNCONDITIONAL (CROSS-PRODUCT) ONE, AND `R`'s IS
DIFFERENT.** `scipy` returns `0.25` for `[[10,20],[30,15]]` — the
cross-product `(a*d)/(b*c)`. `R` reports `0.2550704` — the **conditional
MLE**, which is a different estimator. The module ships the cross-product
ratio, and the header says so, because the cross-product is the quantity the
contingency table actually determines.

---

## The oracle caught four defects, and two of them were conventions

The oracle was written and run **before** any C++ existed, and it is written
against the *definitions* with the references used to **confirm** them. Two
of the four things it disagreed with were conventions rather than bugs, and
both are now pinned explicitly:

1. the **label space** of the weighted kappa (C2) — the oracle's first
   implementation compressed the present labels and disagreed with the
   explicit-index answer on the weighted fixtures;
2. the **symmetric-table** rule of McNemar's correction (C9) — the oracle
   first applied the correction unconditionally and disagreed with `R` on the
   `[[10,10],[10,10]]` fixture;
3. the **zero-cell** contribution in the G-test (C8) — the oracle's first
   version produced `NaN` on a table with a zero cell;
4. the **transposed** `expected` matrix (C4) — the oracle's first version
   matched `sklearn`'s transpose, and the reason the answer is unaffected
   (symmetric weights) had to be derived rather than assumed.

---

## The oracle-mutation harness must find real coverage gaps

`tests/contingency_mutate_oracle.py` breaks the oracle's model code in
specific ways and requires that it **FAIL** each time. An oracle that cannot
fail certifies nothing. The harness asserts the **pristine** oracle passes
before any verdict is reported, refuses to edit an anchor that occurs twice
(a coin flip is not a verdict), and reports a source that did not change as
`MUT-NOOP` rather than as a coverage gap.

The mutation list is chosen so that each one lands on a **different
convention**: flip the Yates condition, apply McNemar's correction to a
symmetric table, drop the zero-cell guard in the G-test, compress the kappa
label space, transpose the expected matrix, use the doubled tail in Fisher,
swap the McNemar discordant cells, and so on.

---

## The tasks

### Task 1: the oracle

**Files:** Create `tests/contingency_oracle.py`.

Run it under `/usr/bin/python3` (the interpreter with numpy, scipy and
sklearn on this host — the same one the sibling oracles hardcode, and the
same reason: a bare `python3` resolves to whatever is first on `PATH`).

    /usr/bin/python3 tests/contingency_oracle.py
    /usr/bin/python3 tests/contingency_oracle.py --emit-cpp

It must:
- compute every quantity from the **definition**;
- cross-check against `scikit-learn` (kappa, MCC), `scipy` (chi-square,
  G-test, Fisher, binomial) and `R` (McNemar, via `Rscript`) and **print the
  measured agreement**;
- emit the C++ fixture block between the two marker comments.

### Task 2: the oracle-mutation harness

**Files:** Create `tests/contingency_mutate_oracle.py`.

    /usr/bin/python3 tests/contingency_mutate_oracle.py

Requirements: assert the pristine oracle passes first; refuse ambiguous
anchors; report `MUT-NOOP` for an edit that did not apply; score
`MUT-BROKEN` separately from a real failure.

### Task 3: the header and the implementation

**Files:** Create `include/nn/utils/contingency.h` and
`include/nn/utils/contingency.cpp`.

The header states every convention (C1-C13) in the repo's house style: the
convention, the measurement that settled it, and what the alternative
reading would have returned. The implementation follows the plan exactly.

### Task 4: the test suite

**Files:** Create `tests/test_contingency.cpp` and
`tests/splice_contingency_fixture.py`.

The fixture numbers are **transferred**, never hand-transcribed. The suite
pins every convention, both degenerate cases, the guard messages, and an
**independent scalar-loop reference** for the kappa and MCC formulas so a
finite-difference-free agreement check still has a second witness.

### Task 5: the C++ mutation harness

**Files:** Create `tests/mutate_contingency.sh`.

Same shape as the sibling harnesses: a pristine baseline that must pass, one
mutation at a time, each verdict reported, and the three verdict classes that
must not be confused (`MUT-BROKEN` is a build failure, `MUT-CRASH` is a
runtime crash, `MUT-NOOP` is an edit that did not apply).

### Task 6: wire it in

**Files:** Modify `include/nn/nn.h` and `Makefile`.

- `nn.h`: add `#include "utils/contingency.h"` next to the other utils.
- `Makefile`: the `build/test_contingency` link rule, the `tests:`
  prerequisite, and the `run_tests` line.

### Task 7: verify

    make -j"$(nproc)" build/test_contingency && ./build/test_contingency
    /usr/bin/python3 tests/contingency_oracle.py
    /usr/bin/python3 tests/contingency_mutate_oracle.py
    bash tests/mutate_contingency.sh

    # ASan + UBSan. A sanitizer needs its OWN object tree, and the repo's
    # link rules do NOT put $(CXXFLAGS) on the link line, so the sanitizer
    # runtime has to be added there EXPLICITLY or the link fails with
    # `undefined reference to __ubsan_handle_nonnull_arg`.
    make BUILD_DIR=build_asan setup
    make BUILD_DIR=build_asan \
         CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer" \
         -j"$(nproc)" build_asan/test_contingency
    g++ -fsanitize=address,undefined -fno-omit-frame-pointer \
        $(find build_asan -name '*.o') -o build_asan/test_contingency
    ./build_asan/test_contingency

    make run_tests

The suite sweep must show **zero new failures**. The only failing suite
before this work is `test_neural_ode` (the long-standing entry in
`NOT_FIXED.md`), which must be re-measured **identically**.

---

## Out of scope for v1, stated so it cannot creep in

- **The k-group log-rank test.** `utils/survival.h` keeps its df = 1 scope.
  This module ships the general chi-square survival function, which is what
  the k-group log-rank would need, but the log-rank's own API is not changed
  here — a change to a shipped module's public surface belongs in its own
  commit with its own oracle.
- **Yates' correction for tables larger than 2×2.** No reference defines it.
- **The conditional-MLE odds ratio** (`R`'s). A different estimator, named
  differently, for a different question.
- **Mid-p, Barnard and Boschloo exact tests.** Different tests.
- **Weighted Fisher's exact.** `scipy` does not implement it.
- **A `chi2_contingency` for R×C tables with unequal margins.** The
  hypergeometric support is handled exactly for the 2×2 case only.
- **Cohen's kappa for `K > 2` with non-nominal (ordinal) scales** beyond the
  linear and quadratic disagreement weights.
