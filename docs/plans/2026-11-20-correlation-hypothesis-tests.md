# Correlation Hypothesis Tests Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** ship the **p-values** for Pearson's r, Spearman's rho and Kendall's tau-b — the hypothesis tests for the repo's three shipped correlation coefficients, of which it currently ships **none**.

**Architecture:** a new `include/nn/utils/correlation_tests.{h,cpp}` in the same model-free shape as `utils/survival.{h,cpp}`, `utils/rank_correlation.{h,cpp}` and `utils/contingency.{h,cpp}` — every core is a free function over `Tensor` or over plain scalars, no `Layer`, no `Model`, no randomness, no optimizer. The numerical engine is the **Student's t distribution's two-sided survival function**, computed through the **regularised incomplete beta** the repo already ships (`utils/randomized_smoothing.h`'s `regularized_incomplete_beta`), which becomes public for this purpose.

**Tech Stack:** C++17 (repo's `-std=c++17 -O2 -Wall -Wextra -march=native`), float64 numpy+scipy oracle, a bash mutation harness, an oracle-mutation harness, a fixture-splice script, ASan+UBSan.

---

## Why this feature, measured not assumed

The `## Ideas` list was **drained a thirtieth time** on 2026-11-20, so the session fell back to the queue's own documented method: a verified-absent concept sweep. The sweep's highest-signal result was a **shipped header naming its own missing piece**:

`include/nn/utils/rank_correlation.h`, under "WHAT IS DELIBERATELY NOT SHIPPED, AND WHY":

> * Hypothesis tests and p-values. The asymptotic normal approximations carry their own tie corrections and the exact permutation test is a different order of complexity. The STATISTICS ship; the TESTS do not.

**A file that names its own missing piece is a specification** — the pattern the queue's own history records as its strongest signal. The measured absence:

```
grep -rn 'p_value' include/nn/utils/*.h
  -> only `utils/randomized_smoothing.h` (its abstention test)
```

**ZERO p-values for any correlation in a repo with ~250 layers and 60+ utility files.**

### It is not redundant with the shipped neighbours

- `utils/rank_correlation.h` — the **STATISTICS**. Its own header names the tests as its missing piece, and this module fills it.
- `utils/metrics.h` — ROC-AUC, AP, NDCG. Ranking quality against labels, not the significance of an association.
- `utils/conformal.h` — the coverage of a **SET**. "Does this set contain the right things", not "is this association distinguishable from noise".
- `utils/survival.h` — the log-rank test's p-value, a chi-square on a **risk set**. A different test on a different data layout.
- `utils/contingency.h` — the chi-square test of **INDEPENDENCE on a contingency table**. Counts, not two labelings of the same items.

### And the previous session's work unblocked it

The contingency-statistics module shipped **2026-11-20** brought the **general chi-square survival function** (the regularised upper incomplete gamma) with it. Before it, this repo could produce a chi-square p-value only for **df == 1**. The asymptotic p-value for a rank correlation is now one function call away.

---

## The conventions, and the places where the textbook answer is not the answer

Every convention below was **MEASURED** against scipy 1.17.1 and R 4.5.3 on this host, not recalled.

### C1. The Pearson and Spearman p-values are the t statistic's two-sided survival

```
t = |r| * sqrt( (n-2) / (1 - r^2) )
p = P(|T_{n-2}| >= |t|) = I_{dof/(dof+t^2)}(dof/2, 1/2)
```

`I_x(a,b)` is the regularised incomplete beta — **the repo's own**, composed from `utils/randomized_smoothing.h` rather than reimplemented.

**MEASURED**: the t form agrees with scipy's `pearsonr` DEFAULT (which uses the **exact beta distribution** of r) to **3.3e-14** worst relative over every n in {3…100} × 400 random fixtures, and with scipy's `spearmanr` to **9.4e-14** over 3000 fixtures. **The two references disagree on the ALGORITHM and agree on the ANSWER.**

The t form ships because it is R's convention, it is the form scipy itself uses for Spearman (so **one** routine serves both tests), and it degrades gracefully where the beta form's shape parameter `n/2-1` goes to zero.

### C1-bis. The two forms are DISTINGUISHABLE, and the harness refuted an "inert" claim

The oracle's first draft declared the mutation from the t form to the exact-beta form **INERT**, on the reasoning that "they agree to 14 digits". **The mutation harness caught it**: the oracle's own R-agreement check has a **1e-14** tolerance and the measured difference is **1.65e-13 — above it**.

> *A convention that is observable at the oracle's own tolerance is not inert, no matter how small the relative difference looks.*

### C2. The Kendall p-value is the asymptotic normal, with the tie corrections in its variance

```
var = (m*(2n+5) - x1 - y1)/18 + (2*xtie*ytie)/m + (x0*y0)/(9*m*(n-2))
z   = con_minus_dis / sqrt(var)
p   = 2 * (1 - Phi(|z|))
```

with, over the tie **groups** of x (and separately y), `c` the group size:
`xtie = Σ c(c-1)/2`, `x0 = Σ c(c-1)(c-2)`, `x1 = Σ c(c-1)(2c+5)`.

**The tie terms are load-bearing**: MEASURED on the heavy-tie fixture, the full variance gives **2.044807e-03** and dropping the two tie terms gives **1.945774e-03**.

**And the tie-group statistics are not the tie PAIR COUNT**: F5's x has the groups {4,2} → 7 tied pairs; F6's x has {3,3,2} → 7 tied pairs. Same pair count, **different variances**.

**And the asymptotic form is the only form that exists for a tied sample**: scipy's `kendalltau(method='exact')` **raises** on ties, and R's `cor.test` warns and falls back.

### C3. n == 2 is NaN, for all three tests

The references **disagree** there: MEASURED, scipy's `pearsonr` returns **1.0**, scipy's `spearmanr` returns **NaN**, and R's `cor.test` refuses outright. There is therefore no reference to agree with, and **NaN is the honest answer** — the test is undefined, not "no evidence" and not an error.

### C4. The statistic is clipped to [-1, 1] — and the clip is load-bearing only because there is no second guard

MEASURED on **this repo's own** `pearson_r`: over 300000 near-perfectly-collinear samples, the largest `|r|` it returns is **1.0000000000000011102** — above 1, by two ULP. That makes `1 - r*r` **negative**, so the p-value of a **perfect** correlation would be **NaN**.

scipy clips for the same reason. **The clip is load-bearing only because there is no second guard** — the first draft had both a clip and a `denom <= 0` branch, and the mutation harness proved the branch made the clip **invisible** (the mutation "the statistic is not clipped" **survived**). **A redundant guard hides a defect.**

### C5. A constant sample is NaN, not a throw.
### C6. The p-value is two-sided, always. Every reference is two-sided.
### C7. The p-value is 1.0 at r == 0 and 0.0 at |r| == 1 exactly.

### C7-bis. |r| just below 1 is not |r| == 1

A statistic of 0.9999999999999999 gives **1.4e-24**, not 0.0: the p-value is an extremely steep function of the statistic near the boundary. The module does not special-case `|r| == 1`; it divides by `1 - r^2` and lets the result be what it is.

### C8. The statistic is an argument, not a recomputation

The p-value functions take the statistic and the sample size. The repo's `pearson_r`, `spearman_rho` and `kendall_tau_b` each have their own tie conventions and their own floating-point details, and recomputing them would create a **second source of truth** for each. The convenience wrappers call the repo's own statistics.

---

## Task 1: the oracle

**Files:** Create `tests/correlation_tests_oracle.py`.

Written and run **before any C++ exists**. Float64 numpy, measured against scipy 1.17.1 and R 4.5.3.

Its discriminating power is **measured, not narrated**: `tests/correlation_tests_mutate_oracle.py` breaks the oracle's model code in 11 ways and requires that it fail each time. **The oracle's first draft declared one of them INERT and the harness refuted that claim** (C1-bis).

## Task 2: the implementation

**Files:** Create `include/nn/utils/correlation_tests.{h,cpp}`.

Composes the repo's own `regularized_incomplete_beta` (which becomes public for this purpose), `normal_cdf` and the three correlation statistics.

**The `regularized_incomplete_beta` export** is a separate commit: it moves the function out of `utils/randomized_smoothing.cpp`'s anonymous namespace and declares it in the header, because a **second** continued-fraction incomplete beta would be the "one implementation of each primitive" mistake this repo has already paid for.

## Task 3: the suite

**Files:** Create `tests/test_correlation_tests.cpp`, `tests/splice_correlation_tests_fixture.py`.

The pinned numbers are **transferred** from the oracle's emitted C++ literals, never hand-transcribed.

The suite carries an **independent reference** — its own incomplete beta, its own t survival and its own Kendall variance, written from the definitions with plain scalar loops and **no shared code with the library**.

**Every p-value call goes through a throw-proof helper.** A throwing implementation must be a **failed check**, not a crashed binary: the mutation harness reported the first draft's `MUT-CRASH` and **a crash is not a verdict**.

## Task 4: the mutation harness

**Files:** Create `tests/mutate_correlation_tests.sh`.

17 mutations. Same shape as the sibling harnesses, with every guard this repo has already paid for: a `restore()` that never deletes its backup, the link set from `make print-lib-objs` with a count assertion, the mutated TU compiled separately and swapped into the link set, and `MUT-BROKEN` / `MUT-CRASH` / `MUT-NOOP` as three separate classes.

## Task 5: wire it in

**Files:** Modify `include/nn/nn.h` and `Makefile`.

## Task 6: verify

```
make -j"$(nproc)" build/test_correlation_tests && ./build/test_correlation_tests
/usr/bin/python3 tests/correlation_tests_oracle.py
/usr/bin/python3 tests/correlation_tests_mutate_oracle.py
bash tests/mutate_correlation_tests.sh

# ASan + UBSan. A sanitizer needs its OWN object tree, and the repo's
# link rules do NOT put $(CXXFLAGS) on the link line, so the sanitizer
# runtime has to be added there EXPLICITLY.
make BUILD_DIR=build_asan setup
make BUILD_DIR=build_asan \
     CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer" \
     -j"$(nproc)" build_asan/test_correlation_tests
g++ -fsanitize=address,undefined -fno-omit-frame-pointer \
    $(make print-lib-objs | sed 's|^build/|build_asan/|' | grep -v 'utils/correlation_tests.o') \
    build_asan/test_correlation_tests.o build_asan/utils/correlation_tests.o \
    -o build_asan/test_correlation_tests

make run_tests
```

The suite sweep must show **zero new failures**. The only failing suite before this work is `test_neural_ode` (the long-standing entry in `NOT_FIXED.md`), which must be re-measured **identically**.

---

## Out of scope for v1, stated so it cannot creep in

- **The exact permutation test.** A different algorithm (an enumeration over orderings), and `utils/rank_correlation.h` names it as such.
- **scipy's `kendalltau(method='exact')`.** It **raises** on ties, so it is not a usable default for the common case.
- **One-sided p-values.** A one-sided p-value needs a **direction**, and choosing one silently is the kind of default this repo refuses to make.
- **A `Layer`, any randomness, any model.** This module is model-free.
- **Changing `utils/rank_correlation.h`'s statistics.** The p-values are a new module; the coefficients are untouched.
