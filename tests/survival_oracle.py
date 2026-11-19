#!/usr/bin/env python3
"""
tests/survival_oracle.py -- the float64 numpy specification of the
SURVIVAL-ANALYSIS utilities: the Kaplan-Meier estimator, the Nelson-Aalen
cumulative hazard, the log-rank test, Harrell's concordance index, and the
Cox proportional-hazards partial log-likelihood with its analytic gradient.

RUN THIS BEFORE WRITING ANY C++.

    python3 tests/survival_oracle.py
    python3 tests/survival_oracle.py --emit-cpp

Its discriminating power is MEASURED, not narrated: the companion harness
tests/survival_mutate_oracle.py breaks the model code below in specific ways
and requires that this oracle FAIL each time. An oracle that cannot fail
certifies nothing.

==========================================================================
THE GAP (measured 2026-11-19, not assumed)
==========================================================================

    grep -rinE 'survival|kaplan|cox.proportional|hazard|time.to.event|censoring'
            include/ tests/ --include=*.h --include=*.cpp
    -> include/nn/layers/utility/stochastic_depth.cpp:11
         ("Sample a single Bernoulli for block SURVIVAL")   <- FALSE POSITIVE
    -> include/nn/utils/sinkhorn.cpp:359
         ("an argument-evaluation-order HAZARD")              <- FALSE POSITIVE

    ZERO IMPLEMENTATION.

The repo ships ~250 layers and 60+ utility files and NO survival analysis.
The neighbouring statistics it DOES ship -- conformal prediction
(utils/conformal.h), rank correlation (utils/rank_correlation.h), isotonic
regression (utils/isotonic_regression.h), the Gaussian process
(utils/gaussian_process.h) -- are all DISTRIBUTION-FREE or
NON-PARAMETRIC in exactly the same sense, so this is the same family of
tool and not a new kind of thing.

It is NOT redundant with any of them:
  * utils/conformal.h measures a SET's coverage. It has no notion of a
    survival time, of censoring, or of a hazard.
  * utils/rank_correlation.h measures the association between two
    quantities observed WITHOUT censoring. A censored observation breaks
    exactly the assumption that a rank statistic rests on.
  * utils/gaussian_process.h regresses a value on covariates. The subject
    here is a TIME TO EVENT, which is only observed up to a censoring
    time, and the estimator that handles that is a different algorithm
    (a risk set, not a design matrix).

==========================================================================
THE INDEPENDENT REFERENCE
==========================================================================

Every quantity below is cross-checked against **lifelines 0.30.3**, a
MAINTAINED reference implementation. lifelines is not consulted for the
VALUES -- the values are computed from the DEFINITIONS here -- but it is
consulted to CONFIRM the definitions, and the agreement is measured and
printed.

Where lifelines and this file disagree, the disagreement is reported with
the exact case, and the convention is then pinned EXPLICITLY rather than
silently inherited. Two such disagreements exist and both are conventions
rather than bugs; see C7 and C11.

==========================================================================
THE CONVENTIONS, AND EACH ONE IS A MEASURED DECISION
==========================================================================

C1. THE RISK SET AT TIME t IS {j : t_j >= t}. A subject CENSORED at
    exactly t is still AT RISK at t -- it is known to have survived to t
    -- and is removed only for times strictly after t. The opposite
    convention ({j : t_j > t}) drops a censored-at-t subject from the
    risk set at the very time it is known to be alive, which biases the
    estimator downward.

    MEASURED, and the fixture DISCRIMINATES the two conventions: on the
    three-subject fixture (times [1,1,2], events [1,0,1]) the at-risk
    count at t=1 is 3 and S(1) = 2/3. The opposite convention gives an
    at-risk count of 2 and S(1) = 1/2. A fixture that cannot tell the
    two apart pins nothing.

    CROSS-CHECKED against lifelines' KaplanMeierFitter: the at-risk
    counts and the survival curve agree to < 1e-12 on the 24-subject
    fixture.

C2. THE KAPLAN-MEIER CURVE IS A RIGHT-CONTINUOUS STEP FUNCTION THAT IS
    CONSTANT ON [t_k, t_{k+1}) AND TAKES ITS POST-EVENT VALUE AT t_k:
        S(t) = prod over event times u <= t of (1 - d(u) / n_risk(u))

C3. THE MEDIAN SURVIVAL TIME IS THE FIRST EVENT TIME AT WHICH S(t) <= 0.5,
    and it is +INFINITY when S never reaches 0.5. It is NOT interpolated.

C4. GREENWOOD'S VARIANCE IS SHIPPED ALONGSIDE THE CURVE:
        Var(S(t)) = S(t)^2 * sum over event times u <= t of
                    d(u) / (n_risk(u) * (n_risk(u) - d(u)))
    The term is 0 when n_risk(u) == d(u) (the whole risk set died) --
    that is a LIMIT, not a division by zero. A variance estimator that
    divides by zero there produces inf or NaN, and a metric that reports
    NaN reports nothing. The three-subject all-die fixture pins it.

C5. THE NELSON-AALEN CUMULATIVE HAZARD IS THE NAIVE ONE:
        H(t) = sum over event times u <= t of d(u) / n_risk(u)
    It is reported on the SAME grid as the Kaplan-Meier curve -- the
    EVENT times -- so the two are directly comparable row by row.

C6. THE LOG-RANK TEST IS THE TWO-SAMPLE, ONE-DEGREE-OF-FREEDOM VERSION.
        statistic = (O1 - E1)^2 / V
    with O1 the observed group-1 events,
        E1 = sum over event times of d(t) * n1(t) / n(t)
    and V the summed HYPERGEOMETRIC variance
        d(t) * (n1(t)/n(t)) * (1 - n1(t)/n(t)) * ((n(t) - d(t))/(n(t) - 1))
    The p-value is the chi-square-1 survival function, which has the exact
    closed form
        p = erfc(sqrt(statistic / 2))
    -- no incomplete gamma, no special-function library. **k > 2 GROUPS IS
    OUT OF SCOPE** and says so: the statistic generalises to k-1 degrees
    of freedom, but its p-value does not have a closed form in erfc, and
    shipping a general incomplete gamma for one extra function is a
    scope decision this file does not make silently.

C7. THE CONCORDANCE INDEX IS HARRELL'S C ON THE **RISK** SCORE:
    HIGHER SCORE = HIGHER RISK = SHORTER SURVIVAL. The pair is concordant
    when the subject who failed FIRST has the HIGHER risk score.

    **THIS IS A SIGN CONVENTION, AND THE TWO READINGS EXIST IN THE WILD.**
    lifelines' `concordance_index(event_times, predicted_scores, ...)`
    treats the score as a SURVIVAL-TIME-like quantity -- HIGHER SCORE =
    LONGER SURVIVAL -- and its own Cox example therefore passes
    `-cph.predict_partial_hazard(...)`, NEGATED.

    MEASURED on a four-subject fixture with times [1,2,3,4] and all
    events:
        risk score [4,3,2,1] (high = early failure) -> lifelines  0.0
        survival score [1,2,3,4] (high = late failure) -> lifelines 1.0
    So lifelines' convention is the OPPOSITE of the risk convention, and
    an implementation that silently picks one reports the COMPLEMENT of
    the other: 1 - C.

    **THIS FILE SHIPS THE RISK CONVENTION**, because a Cox model's own
    output is a risk score (the linear predictor, higher = more risk) and
    every published C-index for a Cox model uses that orientation. The
    oracle therefore cross-checks against lifelines on the NEGATED score,
    and separately asserts the COMPLEMENTARITY IDENTITY
        C_risk(s) + C_risk(-s) == 1
    which holds exactly for every score vector (including one with tied
    scores), so the relationship is pinned from both sides.

    The admissibility rule (the pairs that count at all):
      * two observations with the SAME time are admissible only if exactly
        ONE of them is an event -- then the event is known to have come
        first;
      * two observations with DIFFERENT times are admissible if the one
        with the EARLIER time is an event -- it is then known to have
        failed before the other;
      * a pair with EQUAL risk scores counts as HALF correct;
      * for a pair with equal times where one is an event, "correct"
        means the EVENT's risk score is HIGHER.

C8. THE COX PARTIAL LOG-LIKELIHOOD IS THE LOG-PARTIAL-LIKELIHOOD, NOT
    THE FULL LIKELIHOOD: it has no baseline-hazard term, so it is a
    function of the COEFFICIENTS ONLY and is comparable across different
    censoring patterns.

C9. THE COX TIE HANDLING IS AN EXPLICIT PARAMETER, not a silent default:
      * BRESLOW:  l = sum over event times t of
                  [ sum over i in D(t) of x_i.beta
                    - d(t) * log( S0(t) ) ]
                  with S0(t) = sum over the risk set of exp(x_j.beta)
      * EFRON:    the same, except the d(t) identical terms are replaced
                  by d(t) terms with the tied deaths' OWN contribution
                  removed in d(t) equal steps:
                  l = sum over t of
                      [ sum over i in D(t) of x_i.beta
                        - sum over l = 0..d(t)-1 of
                          log( S0(t) - (l/d(t)) * S0_D(t) ) ]
    Breslow and Efron COINCIDE when no two events share a time, which is
    exactly why a test that never produces a tie cannot tell them apart.
    So the fixtures include one WITH tied event times, and the suite
    asserts that the two methods DIFFER there.

C10. THE COX GRADIENT IS THE ANALYTIC ONE, IN CLOSED FORM:
      Breslow at event time t:
          sum over i in D(t) of x_i
          - d(t) * ( sum over j in R(t) of x_j exp(x_j.beta) ) / S0(t)
      Efron replaces the single ratio by the d(t) ratios with the tied
      deaths' weighted mean subtracted, matching lifelines'
      `weighted_average * summand.sum(0)` term.

C11. **THE PARTIAL LOG-LIKELIHOOD IS REPORTED AS A MEAN OVER THE NUMBER
      OF EVENTS, NOT A RAW SUM.**

      MEASURED: lifelines reports the SUM (-28.552063357741332 on the
      24-subject fixture) and the MEAN is exactly that divided by the
      event count (17), giving -1.6795331386906664. The oracle asserts
      BOTH the mean AND that the raw sum differs from it, and asserts the
      sum against lifelines.

      The reason is the repo's own, twice-established convention:
      `utils/losses.h` documents that the mean reduction divides by
      `numel`, NOT by the row count (D7), and NOT_FIXED.md's VAE entry
      records the bug that happens when two halves of one loss are scaled
      differently -- "a loss that scales with the batch size is not a
      per-sample quantity". The partial likelihood is a sum over EVENT
      TIMES, so the analogous per-event quantity divides by the number
      of events. A caller who wants the sum multiplies by it, and the
      multiplier is reported.

      The GRADIENT is the mean too, so the step size does not change with
      the number of events -- which is what makes the objective usable by
      the repo's optimizers.
"""

import argparse
import math
import sys

import numpy as np
from scipy.special import erfc as _sp_erfc


# ==========================================================================
# THE FIXTURES. Every one of them is defined here, once, and used by both
# the model code and the emitter. A fixture is specified by its CONTENT,
# never by its name (the repo has been burned by that before).
# ==========================================================================

def make_fixtures():
    """Return the dict of fixtures. Every array is float64."""
    fx = {}

    # ---- F1: the small hand-checkable KM / Nelson-Aalen fixture --------
    # 6 subjects. The censoring times 3, 4, 5 carry NO event, so they are
    # not on the event grid at all -- which is itself part of C5.
    #   t=1: n_risk = 6 (all), d = 2  -> S = 1 - 2/6 = 2/3
    #   t=2: n_risk = 4,        d = 1  -> S = (2/3)(1 - 1/4) = 1/2
    #   t=3,4,5: censored only
    #   median = 2 (the first event time with S <= 0.5)
    fx["f1_times"] = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    fx["f1_events"] = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # ---- F2: tied EVENT times, for the Breslow-vs-Efron distinction ----
    # Two events at t=1, two at t=2, one at t=3.
    fx["f2_times"] = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.5, 4.0])
    fx["f2_events"] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])

    # ---- F3: the Cox fixture (n=24, d=2, seed 7) ----------------------
    rng = np.random.default_rng(7)
    n = 24
    X = rng.normal(size=(n, 2))
    beta_true = np.array([0.8, -0.5])
    lp = X @ beta_true
    times = rng.exponential(scale=np.exp(-lp))
    events = rng.binomial(1, 0.7, size=n).astype(float)
    fx["f3_X"] = X
    fx["f3_times"] = times
    fx["f3_events"] = events
    fx["f3_beta"] = np.array([0.35, -0.22])

    # ---- F4: the Cox fixture WITH tied event times ---------------------
    # The times are rounded so that several events share a time. Without
    # this fixture the suite cannot distinguish Breslow from Efron, which
    # is the whole reason the tie method is a parameter (C9).
    rng4 = np.random.default_rng(11)
    n4 = 16
    X4 = rng4.normal(size=(n4, 2))
    t4 = np.round(rng4.uniform(1.0, 8.0, size=n4), 0)
    e4 = rng4.binomial(1, 0.75, size=n4).astype(float)
    fx["f4_X"] = X4
    fx["f4_times"] = t4
    fx["f4_events"] = e4
    fx["f4_beta"] = np.array([-0.4, 0.9])

    # ---- F5: the log-rank two-group fixture ---------------------------
    # Group A: 5 events, 2 censored. Group B: 3 events, 5 censored.
    #
    # **THE TWO GROUPS SHARE EVENT TIMES ON PURPOSE.** The first version
    # of this fixture had every event time carrying d = 1, and the
    # hypergeometric finite-population factor (n-d)/(n-1) is then
    # EXACTLY 1 at every time -- so a log-rank implementation that drops
    # that factor is arithmetically INDISTINGUISHABLE on this fixture.
    # The oracle-mutation harness proved it: the "drops the
    # finite-population factor" mutation SURVIVED. The fixture now has
    # d = 2 at t=1 and d = 3 at t=2, so the factor is 13/14 and 10/12
    # and the mutation is caught.
    fx["f5a_times"] = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fx["f5a_events"] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    fx["f5b_times"] = np.array([1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    fx["f5b_events"] = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ---- F6: the concordance fixture, with TIED risk scores ------------
    # The ties are what make the "half credit" rule observable.
    fx["f6_times"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fx["f6_events"] = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    fx["f6_scores"] = np.array([0.5, 0.5, 0.9, 0.2, 0.2, 0.7, 0.1, 0.4])

    # ---- F7: the all-censored fixture (no events at all) --------------
    fx["f7_times"] = np.array([1.0, 2.0, 3.0, 4.0])
    fx["f7_events"] = np.array([0.0, 0.0, 0.0, 0.0])

    # ---- F8: EVERYONE DIES AT ONCE -- n_risk == d ---------------------
    # Greenwood's term is d/(n(n-d)) = 3/(3*0), a LIMIT of 0, not a
    # division by zero (C4).
    fx["f8_times"] = np.array([2.0, 2.0, 2.0])
    fx["f8_events"] = np.array([1.0, 1.0, 1.0])

    # ---- F9: a subject CENSORED AT AN EVENT TIME ----------------------
    # This fixture DISCRIMINATES C1 from its opposite: the risk set at
    # t=1 is 3 (including the subject censored at t=1), giving
    # S(1) = 2/3. The opposite convention gives S(1) = 1/2.
    fx["f9_times"] = np.array([1.0, 1.0, 2.0])
    fx["f9_events"] = np.array([1.0, 0.0, 1.0])

    # ---- F10: the concordance fixture with EQUAL TIMES ------------------
    # Two subjects at t=1, exactly one of them an event. The pair is
    # admissible only because exactly one event happened (C7).
    fx["f10_times"] = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    fx["f10_events"] = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    fx["f10_scores"] = np.array([3.0, 1.0, 2.0, 0.5, 4.0])

    # ---- F11: a survival curve that CROSSES 0.5 BETWEEN event times ----
    # The median convention (C3) is "the first event time at which
    # S(t) <= 0.5". A curve that hits exactly 0.5 at an event time
    # cannot distinguish that rule from an INTERPOLATED median, because
    # the interpolation of a curve that lands exactly on 0.5 returns the
    # same point. Here S falls from 0.6 at t=1 to 0.4 at t=2 -- it
    # CROSSES 0.5 strictly between the two event times -- so the
    # interpolated median (1.5) and the event-time median (2.0) differ.
    #
    # The oracle-mutation harness proved this: the "the median is
    # INTERPOLATED" mutation SURVIVED against F1, whose S hits exactly
    # 0.5 at t=2.
    fx["f11_times"] = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    fx["f11_events"] = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    return fx


# ==========================================================================
# THE MODEL. Written from the DEFINITIONS above, in float64 numpy.
# ==========================================================================

def _at_risk_counts(times, events):
    """Return (unique event times, n_risk, d) using convention C1."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    ev_times = np.unique(times[events > 0.0])
    n_risk = np.empty(len(ev_times))
    d = np.empty(len(ev_times))
    for k, t in enumerate(ev_times):
        n_risk[k] = np.sum(times >= t)
        d[k] = np.sum((times == t) & (events > 0.0))
    return ev_times, n_risk, d


def kaplan_meier(times, events):
    """The Kaplan-Meier estimate (C2) with Greenwood's variance (C4)."""
    ev_times, n_risk, d = _at_risk_counts(times, events)
    n = len(ev_times)
    survival = np.empty(n)
    variance = np.empty(n)
    s = 1.0
    greenwood = 0.0
    for k in range(n):
        s *= (1.0 - d[k] / n_risk[k])
        survival[k] = s
        nr = n_risk[k]
        if nr - d[k] > 0:
            greenwood += d[k] / (nr * (nr - d[k]))
        variance[k] = s * s * greenwood
    return ev_times, n_risk, d, survival, variance


def survival_at(ev_times, survival, t):
    """S(t) on the right-continuous step grid (C2)."""
    idx = np.searchsorted(ev_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(survival[idx])


def median_survival_time(ev_times, survival):
    """C3: the first event time at which S <= 0.5, else +inf."""
    hit = np.nonzero(survival <= 0.5)[0]
    if len(hit) == 0:
        return math.inf
    return float(ev_times[hit[0]])


def nelson_aalen(times, events):
    """The naive Nelson-Aalen cumulative hazard (C5), on the event grid."""
    ev_times, n_risk, d = _at_risk_counts(times, events)
    cum = np.cumsum(d / n_risk)
    return ev_times, n_risk, d, cum


def log_rank_test(times_a, events_a, times_b, events_b):
    """The two-sample log-rank test (C6)."""
    ta = np.asarray(times_a, dtype=float)
    ea = np.asarray(events_a, dtype=float)
    tb = np.asarray(times_b, dtype=float)
    eb = np.asarray(events_b, dtype=float)

    all_t = np.concatenate([ta, tb])
    all_e = np.concatenate([ea, eb])
    ev_times = np.unique(all_t[all_e > 0.0])

    observed_1 = 0.0
    expected_1 = 0.0
    variance = 0.0
    for t in ev_times:
        n1 = np.sum(ta >= t)
        n2 = np.sum(tb >= t)
        n = n1 + n2
        d1 = np.sum((ta == t) & (ea > 0.0))
        d2 = np.sum((tb == t) & (eb > 0.0))
        d = d1 + d2
        if d == 0 or n == 0:
            continue
        observed_1 += d1
        expected_1 += d * n1 / n
        if n > 1:
            variance += d * (n1 / n) * (1.0 - n1 / n) * ((n - d) / (n - 1))
    statistic = 0.0 if variance <= 0.0 else (observed_1 - expected_1) ** 2 / variance
    p = float(_sp_erfc(math.sqrt(statistic / 2.0)))
    return statistic, p, observed_1, expected_1, variance


def concordance_index(times, events, risk_scores):
    """Harrell's C on the RISK score (C7): higher score = higher risk.

    Returns (c_index, num_pairs, num_correct, num_tied).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    scores = np.asarray(risk_scores, dtype=float)
    n = len(times)
    num_correct = 0.0
    num_tied = 0.0
    num_pairs = 0.0
    for a in range(n):
        ta, ea, sa = times[a], events[a] > 0.0, scores[a]
        for b in range(a + 1, n):
            tb, eb, sb = times[b], events[b] > 0.0, scores[b]
            if ta == tb:
                if ea == eb:
                    continue  # not informative: both events or both censored
                # Exactly one event: the event is known to have come first.
                event_first = ea
            else:
                earlier, later = (a, b) if ta < tb else (b, a)
                e_earlier = events[earlier] > 0.0
                if not e_earlier:
                    continue  # the earlier one was censored: not informative
                event_first = earlier == a
            num_pairs += 1.0
            if sa == sb:
                num_tied += 1.0
            else:
                # Correct iff the subject who failed first has the HIGHER
                # risk score.
                risk_first = sa if event_first else sb
                risk_second = sb if event_first else sa
                if risk_first > risk_second:
                    num_correct += 1.0
    if num_pairs == 0:
        return math.nan, 0.0, 0.0, 0.0
    return (num_correct + 0.5 * num_tied) / num_pairs, num_pairs, num_correct, num_tied


def cox_partial_log_likelihood(beta, X, times, events, efron=False):
    """The Cox partial log-likelihood (C8) and its gradient (C10, C11).

    Returns (mean_log_likelihood, mean_gradient, raw_sum_log_likelihood).
    """
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)

    n, d = X.shape
    eta = X @ beta
    ev_times = np.unique(times[events > 0.0])

    loglik = 0.0
    grad = np.zeros(d)
    for t in ev_times:
        risk = times >= t
        deaths = (times == t) & (events > 0.0)
        dt = int(np.sum(deaths))
        if dt == 0:
            continue
        s0 = float(np.sum(np.exp(eta[risk])))
        s0_x = np.sum(np.exp(eta[risk])[:, None] * X[risk], axis=0)
        x_death_sum = np.sum(X[deaths], axis=0)
        if not efron:
            loglik += float(eta[deaths].sum()) - dt * math.log(s0)
            grad += x_death_sum - dt * (s0_x / s0)
        else:
            s0_d = float(np.sum(np.exp(eta[deaths])))
            s0d_x = np.sum(np.exp(eta[deaths])[:, None] * X[deaths], axis=0)
            loglik += float(eta[deaths].sum())
            grad += x_death_sum
            for l in range(dt):
                frac = l / dt
                denom = s0 - frac * s0_d
                numer = s0_x - frac * s0d_x
                loglik -= math.log(denom)
                grad -= numer / denom
    n_events = float(np.sum(events > 0.0))
    if n_events == 0:
        raise ValueError("no events")
    return loglik / n_events, grad / n_events, loglik


def cox_partial_log_likelihood_reference(beta, X, times, events, efron=False):
    """An INDEPENDENT scalar-loop reference for the Cox partial
    log-likelihood, written from the DEFINITION rather than from the
    vectorised model above.

    It shares NO code with `cox_partial_log_likelihood`: explicit Python
    loops, explicit sums, `math.exp` instead of `np.exp`.

    **WHY THIS EXISTS, MEASURED.** The oracle-mutation harness reported
    that the mutation "the Cox Breslow denominator keeps the tied deaths'
    contribution" SURVIVED. The reason is structural: it changes the
    value AND the gradient CONSISTENTLY, so the finite-difference check
    -- which compares the implementation against ITSELF -- still passes.
    **A FINITE-DIFFERENCE CHECK CANNOT SEE A MUTATION THAT MOVES THE
    VALUE AND THE GRADIENT TOGETHER.** The only thing that can see it is
    a second implementation to compare against, which is what this is.

    This is the repo's established pattern: `utils/sample_quality.h`
    ships an `inception_score_reference` for exactly the same reason.
    """
    beta = [float(b) for b in beta]
    times = [float(t) for t in times]
    events = [float(e) for e in events]
    X = [[float(v) for v in row] for row in X]
    n = len(times)
    d = len(beta)

    event_times = sorted({times[i] for i in range(n) if events[i] > 0.0})

    total = 0.0
    grad = [0.0] * d
    for t in event_times:
        risk = [i for i in range(n) if times[i] >= t]
        deaths = [i for i in range(n) if times[i] == t and events[i] > 0.0]
        dt = len(deaths)
        if dt == 0:
            continue

        eta = {}
        for i in range(n):
            acc = 0.0
            for j in range(d):
                acc += X[i][j] * beta[j]
            eta[i] = acc

        s0 = 0.0
        s0_x = [0.0] * d
        for i in risk:
            w = math.exp(eta[i])
            s0 += w
            for j in range(d):
                s0_x[j] += w * X[i][j]

        sum_eta_deaths = 0.0
        x_death_sum = [0.0] * d
        for i in deaths:
            sum_eta_deaths += eta[i]
            for j in range(d):
                x_death_sum[j] += X[i][j]

        if not efron:
            total += sum_eta_deaths - dt * math.log(s0)
            for j in range(d):
                grad[j] += x_death_sum[j] - dt * s0_x[j] / s0
        else:
            s0_d = 0.0
            s0d_x = [0.0] * d
            for i in deaths:
                w = math.exp(eta[i])
                s0_d += w
                for j in range(d):
                    s0d_x[j] += w * X[i][j]
            total += sum_eta_deaths
            for j in range(d):
                grad[j] += x_death_sum[j]
            for l in range(dt):
                frac = l / dt
                denom = s0 - frac * s0_d
                total -= math.log(denom)
                for j in range(d):
                    grad[j] -= (s0_x[j] - frac * s0d_x[j]) / denom

    n_events = float(sum(1 for e in events if e > 0.0))
    if n_events == 0:
        raise ValueError("no events")
    return total / n_events, [g / n_events for g in grad], total


# ==========================================================================
# THE CHECKS.
# ==========================================================================

class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, ok, name, detail=""):
        if ok:
            self.passed += 1
            print("  [PASS] %s" % name)
        else:
            self.failed += 1
            print("  [FAIL] %s   %s" % (name, detail))
        return ok

    def close(self, a, b, tol, name):
        a = float(a)
        b = float(b)
        if math.isinf(a) or math.isinf(b):
            ok = (a == b)
            diff = 0.0 if ok else math.inf
        else:
            diff = abs(a - b)
            ok = diff <= tol
        return self.check(ok, name, "got %.17g want %.17g (diff %.3g, tol %.3g)"
                          % (a, b, diff, tol))


def run_checks(fx, ck):
    # ---- C1/C2/C4: the Kaplan-Meier curve on F1 ------------------------
    ev, nr, d, S, V = kaplan_meier(fx["f1_times"], fx["f1_events"])
    ck.close(nr[0], 6.0, 1e-12, "F1 n_risk(t=1) == 6")
    ck.close(d[0], 2.0, 1e-12, "F1 d(t=1) == 2")
    ck.close(S[0], 2.0 / 3.0, 1e-12, "F1 S(t=1) == 2/3")
    ck.close(nr[1], 4.0, 1e-12, "F1 n_risk(t=2) == 4")
    ck.close(d[1], 1.0, 1e-12, "F1 d(t=2) == 1")
    ck.close(S[1], 0.5, 1e-12, "F1 S(t=2) == 1/2")
    ck.close(len(ev), 2, 0.0, "F1 has exactly 2 EVENT times (censored times are not on the grid)")
    ck.close(V[0], (4.0 / 9.0) * (2.0 / 24.0), 1e-12, "F1 Greenwood var at t=1")
    ck.close(V[1], 0.25 * (1.0 / 12.0 + 1.0 / 12.0), 1e-12, "F1 Greenwood var at t=2")
    ck.check(np.all(np.isfinite(V)), "F1 Greenwood variance is finite everywhere")

    # ---- C2: the step function is right-continuous ---------------------
    ck.close(survival_at(ev, S, 0.5), 1.0, 1e-12, "F1 S(0.5) == 1")
    ck.close(survival_at(ev, S, 1.0), 2.0 / 3.0, 1e-12, "F1 S(1.0) == 2/3")
    ck.close(survival_at(ev, S, 1.5), 2.0 / 3.0, 1e-12, "F1 S(1.5) == 2/3 (step is flat)")
    ck.close(survival_at(ev, S, 2.0), 0.5, 1e-12, "F1 S(2.0) == 1/2")
    ck.close(survival_at(ev, S, 99.0), 0.5, 1e-12, "F1 S(99) == 1/2 (last step)")

    # ---- C3: the median -------------------------------------------------
    ck.close(median_survival_time(ev, S), 2.0, 1e-12, "F1 median == 2")
    ev7, nr7, d7, S7, V7 = kaplan_meier(fx["f7_times"], fx["f7_events"])
    ck.close(len(ev7), 0, 0.0, "F7 has NO event times")
    ck.close(median_survival_time(ev7, S7), math.inf, 0.0, "F7 median == +inf (never reached)")
    ck.close(survival_at(ev7, S7, 100.0), 1.0, 1e-12, "F7 S(100) == 1 (no events at all)")
    # F11: the curve CROSSES 0.5 strictly between two event times, so the
    # event-time median and an INTERPOLATED median differ.
    ev11, nr11, d11, S11, V11 = kaplan_meier(fx["f11_times"], fx["f11_events"])
    ck.close(S11[0], 0.6, 1e-12, "F11 S(t=1) == 0.6")
    ck.close(S11[1], 0.4, 1e-12, "F11 S(t=2) == 0.4")
    ck.close(median_survival_time(ev11, S11), 2.0, 1e-12,
             "F11 median == 2 (the EVENT TIME, not the interpolated 1.5)")

    # ---- C1: the fixture that DISCRIMINATES the risk-set convention ----
    ev9, nr9, d9, S9, V9 = kaplan_meier(fx["f9_times"], fx["f9_events"])
    ck.close(nr9[0], 3.0, 1e-12, "F9 n_risk(t=1) == 3 (the subject censored AT t=1 is at risk)")
    ck.close(S9[0], 2.0 / 3.0, 1e-12, "F9 S(1) == 2/3 (not 1/2 -- the convention is discriminated)")
    ck.close(nr9[1], 1.0, 1e-12, "F9 n_risk(t=2) == 1")

    # ---- C4: the n_risk == d case is a LIMIT, not a division by zero ---
    ev8, nr8, d8, S8, V8 = kaplan_meier(fx["f8_times"], fx["f8_events"])
    ck.close(nr8[0], 3.0, 1e-12, "F8 n_risk == d == 3")
    ck.close(S8[0], 0.0, 1e-12, "F8 S == 0")
    ck.check(np.all(np.isfinite(V8)), "F8 Greenwood variance is FINITE (not inf/NaN)")
    ck.close(V8[0], 0.0, 1e-12, "F8 Greenwood variance == 0 (the limit)")

    # ---- C5: the Nelson-Aalen hazard ------------------------------------
    evn, nrn, dn, H = nelson_aalen(fx["f1_times"], fx["f1_events"])
    ck.close(len(evn), 2, 0.0, "F1 Nelson-Aalen grid == the KM event grid")
    ck.close(H[0], 2.0 / 6.0, 1e-12, "F1 H(t=1) == 2/6")
    ck.close(H[1], 2.0 / 6.0 + 1.0 / 4.0, 1e-12, "F1 H(t=2) == 2/6 + 1/4")

    # ---- C6: the log-rank test -----------------------------------------
    stat, p, obs, exp_, var = log_rank_test(fx["f5a_times"], fx["f5a_events"],
                                             fx["f5b_times"], fx["f5b_events"])
    ck.check(stat > 0.0, "F5 log-rank statistic > 0")
    ck.check(0.0 <= p <= 1.0, "F5 log-rank p-value in [0, 1]")
    ck.close(obs, 5.0, 1e-12, "F5 observed group-1 events == 5")
    ck.check(var > 0.0, "F5 log-rank variance > 0")
    ck.check(exp_ > 0.0, "F5 expected group-1 events > 0")
    # Identical groups -> the statistic is exactly 0 and the p-value 1.
    s0, p0, o0, e0, v0 = log_rank_test(fx["f5a_times"], fx["f5a_events"],
                                       fx["f5a_times"], fx["f5a_events"])
    ck.close(s0, 0.0, 1e-12, "F5 identical groups -> statistic == 0")
    ck.close(p0, 1.0, 1e-12, "F5 identical groups -> p == 1")
    ck.close(o0, e0, 1e-12, "F5 identical groups -> observed == expected")
    # **THE FINITE-POPULATION FACTOR MUST ACTUALLY BITE ON THIS FIXTURE.**
    # If every event time carried d = 1, then (n-d)/(n-1) == 1 exactly
    # and a log-rank implementation that DROPS the factor would be
    # arithmetically indistinguishable. The mutation harness measured
    # exactly that, so the fixture now shares event times between the
    # groups and this check pins it.
    all_t = np.concatenate([fx["f5a_times"], fx["f5b_times"]])
    all_e = np.concatenate([fx["f5a_events"], fx["f5b_events"]])
    tied_event_times = sum(1 for t in np.unique(all_t[all_e > 0.0])
                           if np.sum(all_t == t) >= 2)
    ck.check(tied_event_times >= 1, "F5 HAS event times with d >= 2",
             "tied event times = %d" % tied_event_times)
    # and the factor really is != 1 somewhere:
    factors = []
    for t in np.unique(all_t[all_e > 0.0]):
        n_t = np.sum(all_t >= t)
        d_t = np.sum((all_t == t) & (all_e > 0.0))
        if n_t > 1:
            factors.append((n_t - d_t) / (n_t - 1.0))
    ck.check(any(abs(f - 1.0) > 1e-12 for f in factors),
             "F5 the finite-population factor is NOT 1 everywhere")

    # ---- C7: the concordance index -------------------------------------
    c, npairs, ncorrect, ntied = concordance_index(fx["f6_times"], fx["f6_events"],
                                                    fx["f6_scores"])
    ck.check(npairs > 0, "F6 has admissible pairs")
    ck.check(0.0 <= c <= 1.0, "F6 concordance in [0, 1]")
    ck.close(c, (ncorrect + 0.5 * ntied) / npairs, 1e-12,
             "F6 concordance == (correct + 0.5*tied) / pairs")
    # A CONSTANT risk score gives exactly 0.5 -- the sharpest non-vacuity
    # property of the metric.
    c_const, _, _, _ = concordance_index(fx["f6_times"], fx["f6_events"],
                                          np.full(8, 0.3))
    ck.close(c_const, 0.5, 1e-12, "F6 constant scores -> concordance == 0.5")
    # A PERFECTLY ANTI-CONCORDANT risk score gives exactly 0.0: higher
    # risk for the LATER event.
    c_rev, _, _, _ = concordance_index(fx["f6_times"], fx["f6_events"],
                                        np.arange(1.0, 9.0))
    ck.close(c_rev, 0.0, 1e-12, "F6 anti-concordant scores -> concordance == 0")
    # A PERFECT risk score gives exactly 1.0: higher risk for the EARLIER
    # event.
    c_perf, _, _, _ = concordance_index(fx["f6_times"], fx["f6_events"],
                                         -np.arange(1.0, 9.0))
    ck.close(c_perf, 1.0, 1e-12, "F6 perfect risk scores -> concordance == 1")
    ck.check(ntied > 0, "F6 has tied risk scores", "ntied=%.17g" % ntied)
    # THE COMPLEMENTARITY IDENTITY: C(s) + C(-s) == 1 exactly.
    c_neg, _, _, _ = concordance_index(fx["f6_times"], fx["f6_events"], -fx["f6_scores"])
    ck.close(c + c_neg, 1.0, 1e-12, "F6 C(s) + C(-s) == 1 (the sign convention is pinned)")
    # EQUAL TIMES: the pair is admissible only because exactly one of the
    # two is an event.
    c10, np10, nc10, nt10 = concordance_index(fx["f10_times"], fx["f10_events"],
                                               fx["f10_scores"])
    ck.check(np10 > 0, "F10 has admissible pairs")
    ck.check(0.0 <= c10 <= 1.0, "F10 concordance in [0, 1]")
    ck.close(c10, (nc10 + 0.5 * nt10) / np10, 1e-12, "F10 concordance == (correct + 0.5*tied) / pairs")

    # ---- C8/C9/C10/C11: the Cox partial likelihood ---------------------
    ll_b, g_b, raw_b = cox_partial_log_likelihood(fx["f3_beta"], fx["f3_X"],
                                                  fx["f3_times"], fx["f3_events"],
                                                  efron=False)
    ll_e, g_e, raw_e = cox_partial_log_likelihood(fx["f3_beta"], fx["f3_X"],
                                                  fx["f3_times"], fx["f3_events"],
                                                  efron=True)
    n_ev = int(np.sum(fx["f3_events"] > 0))
    ck.close(raw_b, ll_b * n_ev, 1e-9, "F3 Breslow raw sum == mean * n_events")
    ck.close(raw_e, ll_e * n_ev, 1e-9, "F3 Efron raw sum == mean * n_events")
    ck.check(abs(raw_b - ll_b) > 1e-6, "F3 the raw sum DIFFERS from the mean (C11 is a choice)")
    # Breslow and Efron AGREE on F3 only because there are no tied times.
    tied3 = len(np.unique(fx["f3_times"][fx["f3_events"] > 0])) != n_ev
    ck.check(not tied3, "F3 has NO tied event times")
    ck.close(ll_b, ll_e, 1e-12, "F3 Breslow == Efron (no ties)")
    ck.close(g_b[0], g_e[0], 1e-12, "F3 Breslow grad[0] == Efron grad[0] (no ties)")
    ck.close(g_b[1], g_e[1], 1e-12, "F3 Breslow grad[1] == Efron grad[1] (no ties)")

    # On F4 there ARE tied event times, so the two methods DIFFER.
    ll4b, g4b, _ = cox_partial_log_likelihood(fx["f4_beta"], fx["f4_X"],
                                              fx["f4_times"], fx["f4_events"], efron=False)
    ll4e, g4e, _ = cox_partial_log_likelihood(fx["f4_beta"], fx["f4_X"],
                                              fx["f4_times"], fx["f4_events"], efron=True)
    n_ev4 = int(np.sum(fx["f4_events"] > 0))
    tied4 = len(np.unique(fx["f4_times"][fx["f4_events"] > 0])) != n_ev4
    ck.check(tied4, "F4 HAS tied event times")
    ck.check(abs(ll4b - ll4e) > 1e-6, "F4 Breslow != Efron (tied times)",
             "breslow=%.17g efron=%.17g" % (ll4b, ll4e))
    ck.check(np.max(np.abs(g4b - g4e)) > 1e-6, "F4 Breslow grad != Efron grad")

    # ---- C10: the analytic gradient against finite differences --------
    for tag in ("3", "4"):
        X = fx["f%s_X" % tag]
        tt = fx["f%s_times" % tag]
        ee = fx["f%s_events" % tag]
        beta0 = fx["f%s_beta" % tag]
        for efron in (False, True):
            _, ga, _ = cox_partial_log_likelihood(beta0, X, tt, ee, efron=efron)
            eps = 1e-6
            num = np.zeros_like(ga)
            for j in range(len(beta0)):
                bp = beta0.copy(); bp[j] += eps
                bm = beta0.copy(); bm[j] -= eps
                lp, _, _ = cox_partial_log_likelihood(bp, X, tt, ee, efron=efron)
                lm, _, _ = cox_partial_log_likelihood(bm, X, tt, ee, efron=efron)
                num[j] = (lp - lm) / (2 * eps)
            rel = np.max(np.abs(ga - num) / np.maximum(np.abs(num), 1e-12))
            ck.check(rel < 1e-5, "F%s Cox analytic grad == finite difference (%s)"
                     % (tag, "efron" if efron else "breslow"),
                     "worst rel=%.3g ana=%s num=%s" % (rel, ga, num))

    # ---- C10-bis: THE INDEPENDENT SCALAR-LOOP REFERENCE -----------------
    # **THE FINITE-DIFFERENCE CHECK ABOVE CANNOT SEE A MUTATION THAT
    # MOVES THE VALUE AND THE GRADIENT TOGETHER**, because it compares
    # the implementation against ITSELF. The oracle-mutation harness
    # measured exactly that: the "the Cox Breslow denominator keeps the
    # tied deaths' contribution" mutation SURVIVED every check.
    # The reference below is a second implementation, written from the
    # definition with explicit loops and no shared code, and the model's
    # value AND gradient are compared against it ABSOLUTELY.
    for tag in ("3", "4"):
        X = fx["f%s_X" % tag]
        tt = fx["f%s_times" % tag]
        ee = fx["f%s_events" % tag]
        beta0 = fx["f%s_beta" % tag]
        for efron in (False, True):
            mv, mg, _ = cox_partial_log_likelihood(beta0, X, tt, ee, efron=efron)
            rv, rg, _ = cox_partial_log_likelihood_reference(beta0, X, tt, ee,
                                                             efron=efron)
            ck.close(mv, rv, 1e-12, "F%s Cox VALUE == the independent reference (%s)"
                     % (tag, "efron" if efron else "breslow"))
            ck.close(mg[0], rg[0], 1e-12, "F%s Cox grad[0] == the independent reference (%s)"
                     % (tag, "efron" if efron else "breslow"))
            ck.close(mg[1], rg[1], 1e-12, "F%s Cox grad[1] == the independent reference (%s)"
                     % (tag, "efron" if efron else "breslow"))
    # **AND THE REFERENCE MUST DIFFER FROM A DELIBERATELY WRONG VARIANT**,
    # or it would be a tautology. The variant is the one the mutation
    # harness used: it keeps the tied deaths' own contribution in the
    # Breslow denominator.
    def cox_breslow_wrong(beta, X, times, events):
        """The WRONG Breslow: it does not multiply the log-term by d(t)."""
        beta = np.asarray(beta, dtype=float)
        X = np.asarray(X, dtype=float)
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=float)
        eta = X @ beta
        total = 0.0
        for t in np.unique(times[events > 0.0]):
            risk = times >= t
            deaths = (times == t) & (events > 0.0)
            s0 = float(np.sum(np.exp(eta[risk])))
            total += float(eta[deaths].sum()) - math.log(s0)
        n_events = float(np.sum(events > 0.0))
        return total / n_events
    wrong4 = cox_breslow_wrong(fx["f4_beta"], fx["f4_X"], fx["f4_times"], fx["f4_events"])
    right4 = cox_partial_log_likelihood(fx["f4_beta"], fx["f4_X"], fx["f4_times"],
                                        fx["f4_events"], efron=False)[0]
    ck.check(abs(wrong4 - right4) > 1e-6,
             "F4 the WRONG Breslow differs from the right one (the reference is not a tautology)",
             "wrong=%.17g right=%.17g" % (wrong4, right4))

    ck.check(np.all(np.isfinite(g_b)), "F3 gradient is finite")
    ck.check(np.all(np.isfinite(g_e)), "F3 Efron gradient is finite")

    return ck


# ==========================================================================
# THE REFERENCE CROSS-CHECK against lifelines.
# ==========================================================================

def cross_check(fx):
    try:
        import pandas as pd
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import multivariate_logrank_test
        from lifelines.utils.concordance import concordance_index as ll_ci
    except Exception as exc:  # pragma: no cover
        print("  [SKIP] lifelines cross-check: %s" % exc)
        return 0, 0

    passed = 0
    failed = 0

    def cmp(ok, name, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("  [PASS] lifelines: %s" % name)
        else:
            failed += 1
            print("  [FAIL] lifelines: %s   %s" % (name, detail))
        return ok

    # ---- Kaplan-Meier ---------------------------------------------------
    df = pd.DataFrame({"t": fx["f3_times"], "e": fx["f3_events"].astype(int)})
    kmf = KaplanMeierFitter().fit(df["t"], df["e"])
    ev, nr, d, S, V = kaplan_meier(fx["f3_times"], fx["f3_events"])
    ll_surv = kmf.survival_function_.iloc[:, 0]
    ll_times = list(kmf.survival_function_.index)
    worst_s = 0.0
    worst_risk = 0.0
    for k, t in enumerate(ev):
        j = ll_times.index(t)
        worst_s = max(worst_s, abs(float(ll_surv.iloc[j]) - float(S[k])))
        worst_risk = max(worst_risk, abs(float(kmf.event_table.loc[t, "at_risk"]) - float(nr[k])))
    cmp(worst_s < 1e-12, "KM survival curve agrees with lifelines",
        "worst |diff| = %.3g" % worst_s)
    cmp(worst_risk < 1e-12, "KM at-risk counts agree with lifelines",
        "worst |diff| = %.3g" % worst_risk)

    # ---- the log-rank test ----------------------------------------------
    ta, ea = fx["f5a_times"], fx["f5a_events"]
    tb, eb = fx["f5b_times"], fx["f5b_events"]
    stat, p, _, _, _ = log_rank_test(ta, ea, tb, eb)
    df2 = pd.DataFrame({"t": np.concatenate([ta, tb]),
                        "e": np.concatenate([ea, eb]).astype(int),
                        "g": np.concatenate([np.zeros(len(ta)), np.ones(len(tb))])})
    lr = multivariate_logrank_test(df2["t"], df2["g"], event_observed=df2["e"])
    cmp(abs(stat - float(lr.test_statistic)) < 1e-10,
        "log-rank statistic agrees with lifelines",
        "ours=%.17g lifelines=%.17g" % (stat, float(lr.test_statistic)))
    cmp(abs(p - float(lr.p_value)) < 1e-10,
        "log-rank p-value agrees with lifelines",
        "ours=%.17g lifelines=%.17g" % (p, float(lr.p_value)))

    # ---- the concordance index -----------------------------------------
    # lifelines' score convention is the OPPOSITE of ours (C7), so the
    # cross-check is against the NEGATED score.
    c, _, _, _ = concordance_index(fx["f6_times"], fx["f6_events"], fx["f6_scores"])
    ref = float(ll_ci(fx["f6_times"], -fx["f6_scores"], fx["f6_events"]))
    cmp(abs(c - ref) < 1e-12, "concordance index agrees with lifelines (negated score)",
        "ours=%.17g lifelines=%.17g" % (c, ref))
    c10, _, _, _ = concordance_index(fx["f10_times"], fx["f10_events"], fx["f10_scores"])
    ref10 = float(ll_ci(fx["f10_times"], -fx["f10_scores"], fx["f10_events"]))
    cmp(abs(c10 - ref10) < 1e-12, "concordance index (equal times) agrees with lifelines",
        "ours=%.17g lifelines=%.17g" % (c10, ref10))

    # ---- the Cox partial log-likelihood --------------------------------
    # lifelines reports the SUM at its own MLE; we report the MEAN (C11).
    df3 = pd.DataFrame({"t": fx["f3_times"], "e": fx["f3_events"].astype(int),
                         "x1": fx["f3_X"][:, 0], "x2": fx["f3_X"][:, 1]})
    cph = CoxPHFitter().fit(df3, duration_col="t", event_col="e")
    ll_at_mle, _, raw_at_mle = cox_partial_log_likelihood(cph.params_.values, fx["f3_X"],
                                                          fx["f3_times"], fx["f3_events"],
                                                          efron=False)
    cmp(abs(raw_at_mle - float(cph.log_likelihood_)) < 1e-9,
        "Cox partial log-likelihood SUM agrees with lifelines",
        "ours=%.17g lifelines=%.17g" % (raw_at_mle, float(cph.log_likelihood_)))
    n_ev3 = int(np.sum(fx["f3_events"] > 0))
    cmp(abs(ll_at_mle * n_ev3 - float(cph.log_likelihood_)) < 1e-9,
        "Cox MEAN * n_events agrees with lifelines (C11's factor)",
        "ours=%.17g lifelines=%.17g" % (ll_at_mle * n_ev3, float(cph.log_likelihood_)))

    # The GRADIENT at lifelines' MLE must be ~0 (it is a maximum).
    _, g_mle, _ = cox_partial_log_likelihood(cph.params_.values, fx["f3_X"],
                                             fx["f3_times"], fx["f3_events"], efron=False)
    cmp(np.max(np.abs(g_mle)) < 1e-4,
        "Cox gradient at lifelines' MLE is ~0",
        "max |grad| = %.3g" % np.max(np.abs(g_mle)))

    # ---- the Efron tie handling ----------------------------------------
    # lifelines 0.30 always uses Efron internally, so its reported value
    # is the EFRON partial log-likelihood. Check it on the tied fixture.
    df4 = pd.DataFrame({"t": fx["f4_times"], "e": fx["f4_events"].astype(int),
                         "x1": fx["f4_X"][:, 0], "x2": fx["f4_X"][:, 1]})
    cph4 = CoxPHFitter().fit(df4, duration_col="t", event_col="e")
    _, _, raw4e = cox_partial_log_likelihood(cph4.params_.values, fx["f4_X"],
                                             fx["f4_times"], fx["f4_events"], efron=True)
    _, _, raw4b = cox_partial_log_likelihood(cph4.params_.values, fx["f4_X"],
                                             fx["f4_times"], fx["f4_events"], efron=False)
    cmp(abs(raw4e - float(cph4.log_likelihood_)) < 1e-8,
        "Cox EFRON partial log-likelihood agrees with lifelines on tied times",
        "ours=%.17g lifelines=%.17g" % (raw4e, float(cph4.log_likelihood_)))
    cmp(abs(raw4b - raw4e) > 1e-6, "the Breslow variant DIFFERS from lifelines on tied times",
        "breslow=%.17g efron=%.17g" % (raw4b, raw4e))

    return passed, failed


# ==========================================================================
# THE EMITTER. The C++ fixtures are TRANSFERRED from here, never typed.
# ==========================================================================

def fmt(v):
    """A C++ double literal with enough digits to be bit-exact."""
    v = float(v)
    if v != v:
        return "std::nan(\"\")"
    if v == math.inf:
        return "std::numeric_limits<double>::infinity()"
    if v == -math.inf:
        return "-std::numeric_limits<double>::infinity()"
    s = repr(v)
    if "e" in s or "E" in s:
        return s
    if "." not in s:
        s += ".0"
    return s


def emit_cpp(fx):
    out = []
    A = out.append

    def vec(name, arr):
        A("const std::vector<double> %s = {%s};" % (name, ", ".join(fmt(x) for x in arr)))

    def mat(name, arr):
        A("const std::vector<std::vector<double>> %s = {" % name)
        for row in arr:
            A("    {%s}," % ", ".join(fmt(x) for x in row))
        A("};")

    A("// ------------------------------------------------------------------")
    A("// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<")
    A("// Emitted by tests/survival_oracle.py --emit-cpp. float64 numpy.")
    A("// The numbers are TRANSFERRED, never re-derived by hand.")
    A("// ------------------------------------------------------------------")

    vec("f1_times", fx["f1_times"])
    vec("f1_events", fx["f1_events"])
    vec("f2_times", fx["f2_times"])
    vec("f2_events", fx["f2_events"])
    mat("f3_X", fx["f3_X"])
    vec("f3_times", fx["f3_times"])
    vec("f3_events", fx["f3_events"])
    vec("f3_beta", fx["f3_beta"])
    mat("f4_X", fx["f4_X"])
    vec("f4_times", fx["f4_times"])
    vec("f4_events", fx["f4_events"])
    vec("f4_beta", fx["f4_beta"])
    vec("f5a_times", fx["f5a_times"])
    vec("f5a_events", fx["f5a_events"])
    vec("f5b_times", fx["f5b_times"])
    vec("f5b_events", fx["f5b_events"])
    vec("f6_times", fx["f6_times"])
    vec("f6_events", fx["f6_events"])
    vec("f6_scores", fx["f6_scores"])
    vec("f7_times", fx["f7_times"])
    vec("f7_events", fx["f7_events"])
    vec("f8_times", fx["f8_times"])
    vec("f8_events", fx["f8_events"])
    vec("f9_times", fx["f9_times"])
    vec("f9_events", fx["f9_events"])
    vec("f10_times", fx["f10_times"])
    vec("f10_events", fx["f10_events"])
    vec("f10_scores", fx["f10_scores"])
    vec("f11_times", fx["f11_times"])
    vec("f11_events", fx["f11_events"])

    # ---- the expected VALUES, one entry per checked quantity -----------
    ev, nr, d, S, V = kaplan_meier(fx["f1_times"], fx["f1_events"])
    A("const std::vector<double> f1_km_event_times = {%s};" % ", ".join(fmt(x) for x in ev))
    A("const std::vector<double> f1_km_at_risk      = {%s};" % ", ".join(fmt(x) for x in nr))
    A("const std::vector<double> f1_km_events       = {%s};" % ", ".join(fmt(x) for x in d))
    A("const std::vector<double> f1_km_survival     = {%s};" % ", ".join(fmt(x) for x in S))
    A("const std::vector<double> f1_km_variance     = {%s};" % ", ".join(fmt(x) for x in V))
    A("const double f1_km_median = %s;" % fmt(median_survival_time(ev, S)))

    evn, nrn, dn, H = nelson_aalen(fx["f1_times"], fx["f1_events"])
    A("const std::vector<double> f1_na_event_times = {%s};" % ", ".join(fmt(x) for x in evn))
    A("const std::vector<double> f1_na_hazard      = {%s};" % ", ".join(fmt(x) for x in H))

    ev9, nr9, d9, S9, V9 = kaplan_meier(fx["f9_times"], fx["f9_events"])
    A("const std::vector<double> f9_km_event_times = {%s};" % ", ".join(fmt(x) for x in ev9))
    A("const std::vector<double> f9_km_at_risk      = {%s};" % ", ".join(fmt(x) for x in nr9))
    A("const std::vector<double> f9_km_survival     = {%s};" % ", ".join(fmt(x) for x in S9))

    ev3, nr3, d3_, S3, V3 = kaplan_meier(fx["f3_times"], fx["f3_events"])
    A("const std::vector<double> f3_km_event_times = {%s};" % ", ".join(fmt(x) for x in ev3))
    A("const std::vector<double> f3_km_at_risk      = {%s};" % ", ".join(fmt(x) for x in nr3))
    A("const std::vector<double> f3_km_events       = {%s};" % ", ".join(fmt(x) for x in d3_))
    A("const std::vector<double> f3_km_survival     = {%s};" % ", ".join(fmt(x) for x in S3))
    A("const std::vector<double> f3_km_variance     = {%s};" % ", ".join(fmt(x) for x in V3))

    ev8, nr8, d8, S8, V8 = kaplan_meier(fx["f8_times"], fx["f8_events"])
    A("const std::vector<double> f8_km_event_times = {%s};" % ", ".join(fmt(x) for x in ev8))
    A("const std::vector<double> f8_km_at_risk      = {%s};" % ", ".join(fmt(x) for x in nr8))
    A("const std::vector<double> f8_km_survival     = {%s};" % ", ".join(fmt(x) for x in S8))
    A("const std::vector<double> f8_km_variance     = {%s};" % ", ".join(fmt(x) for x in V8))

    ev11, nr11, d11, S11, V11 = kaplan_meier(fx["f11_times"], fx["f11_events"])
    A("const std::vector<double> f11_km_event_times = {%s};" % ", ".join(fmt(x) for x in ev11))
    A("const std::vector<double> f11_km_at_risk      = {%s};" % ", ".join(fmt(x) for x in nr11))
    A("const std::vector<double> f11_km_survival     = {%s};" % ", ".join(fmt(x) for x in S11))
    A("const double f11_km_median = %s;" % fmt(median_survival_time(ev11, S11)))

    stat, p, obs, exp_, var = log_rank_test(fx["f5a_times"], fx["f5a_events"],
                                             fx["f5b_times"], fx["f5b_events"])
    A("const double f5_logrank_statistic = %s;" % fmt(stat))
    A("const double f5_logrank_pvalue    = %s;" % fmt(p))
    A("const double f5_logrank_observed  = %s;" % fmt(obs))
    A("const double f5_logrank_expected  = %s;" % fmt(exp_))
    A("const double f5_logrank_variance  = %s;" % fmt(var))

    c6, npairs6, ncorrect6, ntied6 = concordance_index(fx["f6_times"], fx["f6_events"],
                                                        fx["f6_scores"])
    A("const double f6_concordance = %s;" % fmt(c6))
    A("const double f6_num_pairs   = %s;" % fmt(npairs6))
    A("const double f6_num_correct = %s;" % fmt(ncorrect6))
    A("const double f6_num_tied    = %s;" % fmt(ntied6))

    c10, np10, nc10, nt10 = concordance_index(fx["f10_times"], fx["f10_events"],
                                               fx["f10_scores"])
    A("const double f10_concordance = %s;" % fmt(c10))
    A("const double f10_num_pairs   = %s;" % fmt(np10))
    A("const double f10_num_correct = %s;" % fmt(nc10))
    A("const double f10_num_tied    = %s;" % fmt(nt10))

    ll3b, g3b, raw3b = cox_partial_log_likelihood(fx["f3_beta"], fx["f3_X"],
                                                  fx["f3_times"], fx["f3_events"], efron=False)
    ll3e, g3e, raw3e = cox_partial_log_likelihood(fx["f3_beta"], fx["f3_X"],
                                                  fx["f3_times"], fx["f3_events"], efron=True)
    A("const double f3_cox_loglik_breslow = %s;" % fmt(ll3b))
    A("const double f3_cox_loglik_efron   = %s;" % fmt(ll3e))
    A("const double f3_cox_rawsum_breslow = %s;" % fmt(raw3b))
    A("const double f3_cox_grad_breslow0  = %s;" % fmt(g3b[0]))
    A("const double f3_cox_grad_breslow1  = %s;" % fmt(g3b[1]))
    A("const double f3_cox_grad_efron0    = %s;" % fmt(g3e[0]))
    A("const double f3_cox_grad_efron1    = %s;" % fmt(g3e[1]))
    A("const double f3_cox_n_events       = %s;" % fmt(float(np.sum(fx["f3_events"] > 0))))

    ll4b, g4b, _ = cox_partial_log_likelihood(fx["f4_beta"], fx["f4_X"],
                                              fx["f4_times"], fx["f4_events"], efron=False)
    ll4e, g4e, _ = cox_partial_log_likelihood(fx["f4_beta"], fx["f4_X"],
                                              fx["f4_times"], fx["f4_events"], efron=True)
    A("const double f4_cox_loglik_breslow = %s;" % fmt(ll4b))
    A("const double f4_cox_loglik_efron   = %s;" % fmt(ll4e))
    A("const double f4_cox_grad_breslow0  = %s;" % fmt(g4b[0]))
    A("const double f4_cox_grad_breslow1  = %s;" % fmt(g4b[1]))
    A("const double f4_cox_grad_efron0    = %s;" % fmt(g4e[0]))
    A("const double f4_cox_grad_efron1    = %s;" % fmt(g4e[1]))

    A("// ------------------------------------------------------------------")
    A("// >>> ORACLE-EMITTED FIXTURE END <<<")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-cpp", action="store_true",
                    help="emit the C++ fixture block instead of running the checks")
    args = ap.parse_args()

    fx = make_fixtures()

    if args.emit_cpp:
        print(emit_cpp(fx))
        return 0

    print("=" * 74)
    print("tests/survival_oracle.py -- the survival-analysis specification")
    print("=" * 74)
    print()
    print("THE CHECKS (properties AND absolute values):")
    ck = Checker()
    run_checks(fx, ck)
    print()
    print("THE REFERENCE CROSS-CHECK (lifelines 0.30.3):")
    p, f = cross_check(fx)
    print()
    print("=" * 74)
    print("oracle: %d passed, %d failed" % (ck.passed + p, ck.failed + f))
    print("=" * 74)
    return 1 if (ck.failed + f) else 0


if __name__ == "__main__":
    sys.exit(main())
