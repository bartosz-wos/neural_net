# Survival Analysis (Kaplan-Meier, Nelson-Aalen, log-rank, Harrell's C, Cox PH) — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship the core estimators of survival analysis as model-free utilities over `Tensor` — the **Kaplan-Meier estimator** with Greenwood's variance, the **Nelson-Aalen cumulative hazard**, the **two-sample log-rank test**, **Harrell's concordance index**, and the **Cox proportional-hazards partial log-likelihood with its analytic gradient** — for a repo that has none of it.

**Architecture:** a new `include/nn/utils/survival.{h,cpp}` in the same shape as `utils/sliced_wasserstein.h`, `utils/rank_correlation.h`, `utils/isotonic_regression.h` and `utils/sample_quality.h` — every core is a free function over `Tensor`, no `Layer`, no `Model`, no randomness, no optimizer. The Cox partial log-likelihood ships with a closed-form gradient so it is directly usable by the repo's existing optimizers.

**Tech Stack:** C++17, the repo's `Tensor`, the project's `-O2 -march=native -Wall -Wextra` build, a float64 numpy+scipy oracle in `tests/`, an oracle-mutation harness in `tests/`, a bash mutation harness in `tests/`, and a fixture-splice script.

---

## Why this feature, measured not assumed

The `## Ideas` list was drained for the 28th time (every bullet is marked `[DONE]`), so the session fell back to the queue's own documented method: a **verified-absent concept sweep**.

```
grep -rinE 'survival|kaplan|cox.proportional|hazard|time.to.event|censoring'
        include/ tests/ --include=*.h --include=*.cpp
-> include/nn/layers/utility/stochastic_depth.cpp:11
     ("Sample a single Bernoulli for block SURVIVAL")      <- FALSE POSITIVE
-> include/nn/utils/sinkhorn.cpp:359
     ("an argument-evaluation-order HAZARD")                 <- FALSE POSITIVE
```

**ZERO IMPLEMENTATION.** The repo ships ~250 layers and 60+ utility files and no survival analysis at all.

A wider sweep measured the surrounding territory, so the pick was made against the whole neighbourhood rather than by keyword alone:

| sweep | hits | verdict |
|---|---|---|
| `causal\|backdoor\|front.door\|do.calculus\|confound\|propensity` | 284 | all "causal ATTENTION" — false positives |
| `fisher.information\|empirical.fisher\|kullback` | 11 | shipped (`utils/distribution_losses.h`) |
| `bayesian.opt\|tpe\|acquisition.function` | 6 | all "GradientPenaltyLoss" — false positives |
| `bandit\|thompson.sampl\|upper.confidence.bound` | 1 | a comment in `off_policy_eval.h` — false positive |
| `survival\|kaplan\|cox.proportional\|hazard\|time.to.event\|censoring` | 2 | **both false positives** — the pick |

### It is not redundant with the shipped neighbours

- `utils/conformal.h` measures a SET's coverage. It has no notion of a survival time, of censoring, or of a hazard.
- `utils/rank_correlation.h` measures the association between two quantities observed **without** censoring. A censored observation breaks exactly the assumption a rank statistic rests on.
- `utils/gaussian_process.h` regresses a value on covariates. The subject here is a **time to event**, observed only up to a censoring time, and the estimator that handles that is a different algorithm — a risk set, not a design matrix.
- `utils/isotonic_regression.h` fits a monotone map on a single response. It is post-hoc calibration, not a survival model.

All the shipped neighbours are distribution-free or non-parametric in the same sense, so this is the same family of tool and not a new kind of thing.

---

## The mathematics, and the places it is easy to get wrong

### The Kaplan-Meier estimator

```
S(t) = prod over event times u <= t of (1 - d(u) / n_risk(u))
```

**C1. THE RISK SET AT t IS `{j : t_j >= t}`.** A subject **censored at exactly t** is still at risk at t — it is known to have survived to t — and is removed only for times strictly after t. The opposite convention drops a censored-at-t subject from the risk set at the very moment it is known to be alive.

Measured, and the fixture **discriminates** the two conventions: on the three-subject fixture (times `[1,1,2]`, events `[1,0,1]`) the at-risk count at t=1 is **3** and `S(1) = 2/3`; the opposite convention gives **2** and `S(1) = 1/2`. A fixture that cannot tell the two apart pins nothing.

Cross-checked against lifelines' `KaplanMeierFitter`: the at-risk counts and the survival curve agree to **< 1e-12**.

### The log-rank test

```
statistic = (O1 - E1)^2 / V
E1 = sum over event times of d(t) * n1(t) / n(t)
V  = sum over event times of d(t) * (n1/n) * (1 - n1/n) * ((n - d)/(n - 1))
p  = erfc(sqrt(statistic / 2))
```

**The chi-square-1 survival function has an exact closed form in `erfc`** — no incomplete gamma, no special-function library.

### Harrell's concordance index — the SIGN is the trap

**C7. THIS FILE SHIPS THE RISK CONVENTION: HIGHER SCORE = HIGHER RISK = SHORTER SURVIVAL.** The pair is concordant when the subject who failed **first** has the **higher** risk score.

**The two readings exist in the wild, and they are COMPLEMENTS of each other.** lifelines' `concordance_index(event_times, predicted_scores, ...)` treats the score as a **survival-time-like** quantity — HIGHER SCORE = LONGER SURVIVAL — and its own Cox example therefore passes `-cph.predict_partial_hazard(...)`, **NEGATED**.

Measured on a four-subject fixture with times `[1,2,3,4]` and all events:

| score vector | meaning | lifelines |
|---|---|---|
| `[4,3,2,1]` | high = early failure (a RISK score) | **0.0** |
| `[1,2,3,4]` | high = late failure | **1.0** |

So an implementation that silently picks one reports **1 − C** of the other. This file ships the risk convention, because a Cox model's own output is a risk score and every published C-index for a Cox model uses that orientation — and the oracle pins it from **both** sides via the complementarity identity `C_risk(s) + C_risk(-s) == 1`, which holds exactly for every score vector including one with tied scores.

### The Cox partial log-likelihood

**C9. THE TIE HANDLING IS AN EXPLICIT PARAMETER, not a silent default.**

```
BRESLOW: l = sum over t of [ sum over i in D(t) of x_i.beta - d(t) * log(S0(t)) ]
EFRON:   l = sum over t of [ sum over i in D(t) of x_i.beta
                            - sum over l = 0..d(t)-1 of log(S0(t) - (l/d(t)) * S0_D(t)) ]
```

Breslow and Efron **coincide when no two events share a time**, which is exactly why a test that never produces a tie cannot tell them apart. The fixtures include one **with** tied event times, and the suite asserts the two methods **differ** there.

**C11. THE PARTIAL LOG-LIKELIHOOD IS REPORTED AS A MEAN OVER THE NUMBER OF EVENTS, NOT A RAW SUM.**

Measured: lifelines reports the sum (`-28.552063357741332` on the 24-subject fixture) and the mean is exactly that divided by the event count (17), giving `-1.6795331386906664`.

The reason is the repo's own, twice-established convention: `utils/losses.h` documents that the mean reduction divides by `numel`, **not** by the row count (D7), and `NOT_FIXED.md`'s VAE entry records the bug that happens when two halves of one loss are scaled differently — *"a loss that scales with the batch size is not a per-sample quantity"*. The partial likelihood is a sum over **event times**, so the analogous per-event quantity divides by the number of events. The gradient is a mean too, so the step size does not change with the event count — which is what makes the objective usable by the repo's optimizers.

---

## The oracle caught four defects, and two of them were conventions

`tests/survival_oracle.py` was run **before** any C++ existed. Its first run reported **18 failures**, and every one was a finding:

**Two were CONVENTION questions that only measurement could settle:**

1. **The concordance sign.** The oracle's first draft followed lifelines' convention; the cross-check disagreed by exactly `1 − C` (ours `0.5238`, lifelines `0.4762`). Resolving it needed a *second* measurement — a four-subject fixture with a known-perfect risk score — not an argument about which is "right".

2. **The mean-vs-sum.** The oracle's first draft reported lifelines' raw sum; the cross-check then disagreed by exactly the event count. That is the `NOT_FIXED.md` VAE convention biting again, and it was resolved by measurement, not by taste.

**Two were the oracle's own hand-written expectations being wrong** — the classic *"the fixture, not the code"* family, and they are why the fixture was rebuilt rather than the expectations edited:

3. The F1 fixture's hand-computed at-risk counts did not match the fixture that was actually written (6 expected, 12 present).

4. The all-censored median comparison was `abs(inf - inf)`, which is `NaN`, and `NaN <= 0` is false — so the check could never pass.

---

## The oracle-mutation harness found three real coverage gaps

`tests/survival_mutate_oracle.py` breaks the oracle's model code in twenty specific ways. Its first run reported **17 caught / 3 survived**, and **every survivor was a real gap** — the skill's rule is that a survivor is a specification for a better check, not a nudge to drop the mutation.

| survivor | why it survived | the fix |
|---|---|---|
| "the median is INTERPOLATED" | F1's survival curve landed **exactly** on 0.5, where interpolation returns the same point | **F11**: a curve that **crosses** 0.5 strictly between two event times, so the two rules disagree |
| "the log-rank variance drops the finite-population factor" | every event time carried `d = 1`, and `(n-d)/(n-1)` is then **exactly 1** — the factor is arithmetically inert | F5's two groups now **share event times**, so the factor is 13/14 and 10/12 |
| "the Cox Breslow denominator keeps the tied deaths' contribution" | the mutation moves the value **and** the gradient **consistently**, so the finite-difference check — which compares the implementation **against itself** — still passes | an **independent scalar-loop reference** in the oracle, compared absolutely |

**The third one is the most reusable finding in this file.** *A finite-difference check compares the implementation against itself, and therefore cannot see a defect that moves the value and the gradient together.* The only thing that can see it is a second implementation to compare against — the same pattern `utils/sample_quality.h` uses for its `inception_score_reference`.

After the three fixes: **20 caught / 0 survived / 0 broken / 0 harness defects**.

---

## The C++ mutation harness

`tests/mutate_survival.sh` breaks `utils/survival.cpp` in twenty-three specific ways. Its first run reported **22 caught / 0 survived / 1 crashed**, and the crash was a defect in the **mutation**, not in the suite:

**M8 CRASHED (rc=139), AND A CRASH IS NOT A VERDICT.** Its first form replaced the Nelson-Aalen's grid with all the distinct times while leaving the hazard vector at the event-grid length, so the step lookup indexed past the end of a shorter vector. A mutation that makes the binary crash says **nothing** about whether the suite covers the behaviour. It was rewritten to change the grid at its **source** (the shared builder), so every vector stays the same length as its grid and the mutation changes the **observable** instead of the memory safety.

After the fix: **23 caught / 0 survived / 0 broken / 0 crashed / 0 noop**.

---

## The tasks

### Task 1: the oracle

**Objective:** specify every estimator and every convention in float64 numpy, before any C++ exists.

**Files:**
- Create: `tests/survival_oracle.py`

**Step 1: write the model code and the checks**

The checks assert ABSOLUTE values from the fixtures (not only relative properties), the analytic gradient against finite differences, and the model against an independent scalar-loop reference.

**Step 2: run it**

Run: `python3 tests/survival_oracle.py`
Expected: `oracle: 93 passed, 0 failed`

**Step 3: cross-check against lifelines**

The oracle cross-checks against lifelines 0.30.3 — a maintained implementation — for the Kaplan-Meier curve, the at-risk counts, the log-rank statistic and p-value, the concordance index, and the Cox partial log-likelihood (at lifelines' own MLE, so the check is of the **definition** and not of the optimiser).

### Task 2: the oracle-mutation harness

**Objective:** prove the oracle has teeth.

**Files:**
- Create: `tests/survival_mutate_oracle.py`

Run: `python3 tests/survival_mutate_oracle.py`
Expected: `oracle-mutation: 20 caught / 0 survived / 0 broken / 0 harness defects`

**The three survivors of the first run are the file's most valuable content**, and each one names a fixture that was added in response.

### Task 3: the header and the implementation

**Objective:** ship the five estimators.

**Files:**
- Create: `include/nn/utils/survival.h`
- Create: `include/nn/utils/survival.cpp`

**The header carries the conventions and the measurements that justify them** — the risk set, the right-continuity, the median rule, Greenwood's limit, the log-rank's hypergeometric variance, the concordance sign, the Cox tie handling and the mean-over-events.

### Task 4: the test suite

**Objective:** pin every convention and every guard.

**Files:**
- Create: `tests/test_survival.cpp`
- Create: `tests/splice_survival_fixture.py`

The fixtures are **emitted** by the oracle and spliced in by the script, never hand-transcribed.

**THE FIXTURES MUST ISOLATE THE GUARD THEY TEST.** Four guard assertions failed on their first run because their fixtures violated an *earlier* precondition — the row-count guard fired first, so the guard under test was never reached and the assertion would have passed for the wrong reason. The guard **order** is correct; the **fixtures** were wrong.

### Task 5: the C++ mutation harness

**Objective:** find the coverage gaps the suite cannot report.

**Files:**
- Create: `tests/mutate_survival.sh`

### Task 6: wire it in

**Files:**
- Modify: `include/nn/nn.h` — add `#include "utils/survival.h"`
- Modify: `Makefile` — the link rule, the `tests:` prerequisite, the `run_tests` line

### Task 7: verify

```bash
make -j6 build/test_survival && ./build/test_survival     # 170/170
bash tests/mutate_survival.sh                               # 23 caught / 0 survived
/usr/bin/python3 tests/survival_oracle.py                  # 93/93
/usr/bin/python3 tests/survival_mutate_oracle.py           # 20 caught / 0 survived
make run_tests                                              # no new failures
```

**THE ORACLES RUN UNDER `/usr/bin/python3`, NOT A BARE `python3`.** That is
the interpreter with numpy and scipy on this host, and it is what the
sibling oracles in this repo already hardcode (`tests/dtw_mutate_oracle.py`,
`tests/block_cached_decode_mutate_oracle.py`,
`tests/edit_distance_mutate_oracle.py`). A bare `python3` resolves to
whatever is first on PATH — which, inside an agent's own virtual
environment, is **not** the one with numpy — and the oracle then fails to
import, which looks like a broken oracle rather than a wrong interpreter.

**`lifelines` IS OPTIONAL.** The oracle's cross-check against it is wrapped
in a try/except that prints `[SKIP]` and continues, so the oracle's own
93 checks run without it. It is installed user-side here
(`/usr/bin/python3 -m pip install --user lifelines`) so the cross-check
runs too.

**ASan+UBSan:** the mutated TU and the test TU are compiled with `-fsanitize=address,undefined` and linked against the plain library, which is the repo's per-TU sanitizer convention (`tests/mutate_logit_processing.sh`, `tests/mutate_lr_multiplier.sh`). Expected: **clean**.

---

## Out of scope for v1, stated so it cannot creep in

- **k > 2 GROUPS in the log-rank test.** The statistic generalises to k−1 degrees of freedom, but its p-value does not have a closed form in `erfc`, and shipping a general incomplete gamma for one extra function is a scope decision this file does not make silently.
- **A `Layer`.** The Cox partial log-likelihood is an objective, not a forward transform, and it is shipped as a free function over `Tensor` so it composes with the repo's existing optimizers directly.
- **A baseline cumulative hazard.** The partial log-likelihood has no baseline term by construction (C8); adding one is a different estimator.
- **Stratified or time-varying Cox.**
- **Parametric survival models** (Weibull, exponential, Gompertz–Makehan). A different family of estimators with a different objective.
- **Any change to the shipped `utils/losses.h`, `utils/conformal.h` or `utils/metrics.h`.**
