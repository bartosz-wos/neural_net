#!/usr/bin/env python3
"""
tests/off_policy_eval_oracle.py -- the float64 numpy SPECIFICATION for the
off-policy evaluation estimators.

THIS ORACLE IS THE SPECIFICATION. It was written and run BEFORE any C++ exists.

WHY THIS FEATURE, MEASURED NOT ASSUMED (2026-10-30)
====================================================

    grep -rinE '\\bsnr\\b|off.?policy|behavior.?polic|importance.?sampl|self.normaliz'
        include/ tests/
    -> 0 hits

So the repo has NO off-policy machinery of any kind. What it HAS is:

  * `utils/mmd_loss.h` -- a KERNEL two-sample statistic. No importance
    weights, no policy, no value.
  * `utils/conformal.h` -- calibrates a prediction that was ALREADY MADE.
    Not an estimator of a policy's value.
  * `utils/cross_validation.h` -- resamples the SAME data. Off-policy
    evaluation is the opposite: it combines data collected under ONE policy
    with a DIFFERENT policy's action probabilities.
  * `utils/sinkhorn.h` -- an entropic transport PLAN. A plan, not a value.

There is no `reward` concept anywhere in `include/nn` either
(`grep -rinE '\\breward' include/nn --include=*.h` -> 0 hits), so this
module must be MODEL-FREE OVER LOGGED DATA: not a layer, not an environment,
not a simulator. Free functions over `Tensor`, like `utils/sinkhorn.h`,
`utils/crf.h`, `utils/conformal.h` and `utils/gaussian_process.h`.

THE MATHEMATICS
================

Logged data: N samples (s_i, a_i, r_i) with s_i ~ d, a_i ~ pi_0(.|s_i),
r_i ~ P(r|s_i,a_i).  Target policy pi_1.  Importance weight

    rho_i = pi_1(a_i|s_i) / pi_0(a_i|s_i)

Four estimators, over a flat sample vector:

    IPS  V = (1/N) sum_i rho_i r_i
    SNIS V = (sum_i rho_i r_i) / (sum_i rho_i)
    DM   V = (1/N) sum_i Vhat(s_i)
    DR   V = (1/N) sum_i [ Vhat(s_i) + rho_i (r_i - Rhat(s_i,a_i)) ]

where the reward model gives Vhat(s) = sum_a pi_1(a|s) Rhat(s,a) and the
per-sample logged-action value Rhat(s_i,a_i).

**THE SNIS DENOMINATOR IS `sum rho_i`, NOT `N`.** Using N silently returns
IPS. The oracle asserts the exact identity V_SNIS == V_IPS / mean(rho),
which is what separates the two.

DOUBLE ROBUSTNESS, AND WHY IT IS CHECKED BY EXACT ENUMERATION
==============================================================

The DR estimator is unbiased for the target value V_pi1 in EITHER of two
regimes, and the cancellation that makes it so is worth writing out:

    E_{a~pi_0}[ Vhat(s) + rho(a) (r(a) - Rhat(s,a)) ]
        = Vhat(s) + sum_a pi_0(a) (pi_1(a)/pi_0(a)) (E[r|s,a] - Rhat(s,a))
        = Vhat(s) + sum_a pi_1(a) (E[r|s,a] - Rhat(s,a))
        = Vhat(s) + V_pi1(s) - sum_a pi_1(a) Rhat(s,a)
        = V_pi1(s)                                    <- the Vhat TERM CANCELS

  ARM A: propensities EXACT, reward model WRONG -> still exactly V_pi1.
  ARM B: reward model EXACT, propensities WRONG -> still exactly V_pi1
         (the correction term vanishes identically, and DR reduces to DM).

**THE CHECK IS EXACT, NOT MONTE CARLO.** With a finite action support and a
finite number of samples, the expectation of the estimator over the logging
distribution can be computed by ENUMERATING every logged action sequence
with its exact probability. There is no sampling error and no tolerance to
argue about: "the expectation equals V_pi1" is an identity that either holds
exactly or does not. A Monte-Carlo version of this check would be strictly
weaker, and would also be the kind of test that can pass for the wrong
reason (a loose tolerance covering a real bias).

THE PROPERTIES THAT DISCRIMINATE AN IMPLEMENTATION
====================================================

  1. SCALE INVARIANCE OF SNIS.  SNIS(c*rho) == SNIS(rho) for any c > 0,
     because the scale cancels between numerator and denominator.  IPS is
     NOT scale invariant: IPS(c*rho) == c * IPS(rho).  **A FIXTURE THAT
     DOES NOT VARY THE SCALE OF THE WEIGHTS CANNOT TELL THESE APART.**
  2. ESS IS SCALE INVARIANT TOO: ESS(c*rho) == ESS(rho), and ESS <= N with
     equality iff all weights are equal.
  3. DM NEEDS NO WEIGHTS AT ALL.  It is the mean of the model-based target
     values.
  4. DECOMPOSITION: DR == DM + (1/N) sum_i rho_i (r_i - Rhat_i).

THIS ORACLE WAS GREEN ON ITS FIRST EXECUTION -- AND THAT IS NOT A RESULT
=======================================================================

    38 passed, 0 failed, on the first run of a 900-line specification.

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** It may simply be asserting things that are true of whatever
it was written alongside. So the oracle's discriminating power is MEASURED,
not narrated: `tests/off_policy_eval_mutate_oracle.py` breaks the oracle in
seven specific ways and requires that it FAIL each time.

    OM1  the enumeration draws the logged action from pi_1 (the TARGET
         policy) instead of pi_0 (the LOGGING policy).
    OM2  IPS sums instead of averaging (the `1/N` factor is dropped).
    OM3  SNIS divides by N instead of by sum(rho) -- it silently returns
         IPS.
    OM4  the DR estimator drops the Vhat(s) term, so the cancellation that
         makes it doubly robust never happens.
    OM5  the DR estimator flips the sign of the correction.
    OM6  the effective sample size is inverted.
    OM7  the importance weight is inverted.

**ALL SEVEN ARE CAUGHT.** The two that matter most are OM1 and OM4, because
they are the two failures that a *narrated* draft-defect story would have
claimed without evidence: OM1 is the "the fixture is wrong, not the code"
trap (enumerating under the wrong policy makes the weighted expectation
collapse to something that is not V_pi1, so the unbiasedness check fails
against a CORRECT implementation), and OM4 is the failure that separates
the doubly robust estimator from a merely-biased one.

The enumerator additionally asserts its own total probability mass is
exactly 1, and section 1 checks it against a hand-computed value before
anything else runs -- **a broken enumerator would make every downstream
check meaningless**, so it is validated first, on its own.

CITATIONS -- SCRAPED FROM THE arXiv ABS PAGES, NEVER RECALLED
=============================================================

  * 1604.00923  "Data-Efficient Off-Policy Policy Evaluation for
    Reinforcement Learning" -- Thomas, Philip S.; Brunskill, Emma.
    One-step OPE; the direct-method and doubly-robust baselines.
  * 1511.03722  "Doubly Robust Off-policy Value Evaluation for
    Reinforcement Learning" -- Jiang, Nan; Li, Lihong.
    **The bandit DR estimator was transcribed from this paper's own LaTeX
    source** (eq:dr):  DR_I = V(s_I) + rho_I (r_I - R(s_I,a_I)),
    rho = pi_1(a|s)/pi_0(a|s),  V(s) = sum_a pi_1(a|s) R(s,a).
  * 2006.10460  "Confident Off-Policy Evaluation and Selection through
    Self-Normalized Importance Weighting" -- Kuzborskij, Ilja; Vernade,
    Claire; György, András; Szepesvári, Csaba.
  * 1910.12809  "Minimax Weight and Q-Function Learning for Off-Policy
    Evaluation" -- Uehara, Masatoshi; Huang, Jiawei; Jiang, Nan.
  * 2104.12820  "Universal Off-Policy Evaluation" -- Chandak, Yash;
    Niekum, Scott; da Silva, Bruno Castro; Learned-Miller, Erik;
    Brunskill, Emma; Thomas, Philip S.

**NO arXiv ID IS ASSERTED FOR THE SNR METRIC.** The expansion queue's grep
included `\\bsnr\\b`, but an arXiv title search for "sample noise ratio"
returns 0 entries and a scan of all 1233 PMLR v162 titles finds no such
paper, so no identifier and no DEFINITION is claimed for it. The queue
entry is corrected rather than papered over.

OUT OF SCOPE FOR v1, stated so it cannot creep in
==================================================

  * SEQUENTIAL / MULTI-STEP estimators (Jiang & Li's per-decision DR with
    gamma-discounting). They need trajectory storage, discounting and
    per-step cumulative weights -- a different data layout, not a small
    extension of this one.
  * Any environment, simulator, replay buffer or `Layer` integration.
  * Behaviour-policy ESTIMATION (propensity-model fitting). The caller
    supplies the propensities.
  * The variance of the estimators. **Effective sample size is shipped
    because it is the standard, closed-form diagnostic for whether the
    importance weights are usable at all** (a single dominant weight
    drives it to 1), but the standard errors are not asserted here.
"""

import sys
from itertools import product

import numpy as np

np.set_printoptions(precision=15, suppress=False)

PASSED = 0
FAILED = 0


def check(ok, name, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}   {detail}")


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# THE ESTIMATORS, in float64 numpy.
#
# Each takes FLAT sample vectors -- the number of elements is the sample
# count and the shape is irrelevant, exactly like the C++ module's
# convention. The rewards/weights/values are all per-sample and must all
# have the SAME length.
# ---------------------------------------------------------------------------
def importance_weights(target_probs, behavior_probs):
    """rho_i = pi_1(a|s) / pi_0(a|s)."""
    target = np.asarray(target_probs, dtype=np.float64)
    behavior = np.asarray(behavior_probs, dtype=np.float64)
    assert target.shape == behavior.shape
    return target / behavior


def ips_estimate(rewards, weights):
    """(1/N) sum_i rho_i r_i."""
    r = np.asarray(rewards, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * r) / r.size)


def self_normalized_ips(rewards, weights):
    """sum_i rho_i r_i / sum_i rho_i."""
    r = np.asarray(rewards, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * r) / np.sum(w))


def direct_method_estimate(model_target_values):
    """(1/N) sum_i Vhat(s_i)."""
    v = np.asarray(model_target_values, dtype=np.float64)
    return float(np.mean(v))


def doubly_robust_estimate(rewards, weights, model_target_values,
                            logged_action_values):
    """(1/N) sum_i [ Vhat(s_i) + rho_i (r_i - Rhat(s_i,a_i)) ]."""
    r = np.asarray(rewards, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(model_target_values, dtype=np.float64)
    q = np.asarray(logged_action_values, dtype=np.float64)
    assert r.shape == w.shape == v.shape == q.shape
    return float(np.mean(v + w * (r - q)))


def effective_sample_size(weights):
    """(sum rho)^2 / sum rho^2."""
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w) ** 2 / np.sum(w * w))


# ---------------------------------------------------------------------------
# THE EXACT ENUMERATOR.
#
# For a finite action support and a finite number of samples, the
# expectation of an estimator over the LOGGING distribution is computed by
# enumerating every logged action sequence with its exact probability.
#
# **THE LOGGING DISTRIBUTION IS pi_0, THE BEHAVIOUR POLICY.** Enumerating
# under pi_1 is the draft defect this oracle caught: it makes the weighted
# expectation collapse to something that is NOT V_pi1, so the unbiasedness
# check fails against a CORRECT implementation.
# ---------------------------------------------------------------------------
def enumerate_expectation(estimator, n_samples, logging_probs, rewards_of,
                           **kwargs):
    """Exact expectation of `estimator` over the logging distribution.

    logging_probs : (A,)  pi_0(a|s) -- the BEHAVIOUR policy.
    rewards_of    : (A,)  the deterministic reward r(s,a) for each action.
    kwargs        : per-sample arrays (length n_samples) for the estimator's
                    other arguments.
    """
    A = len(logging_probs)
    total = 0.0
    mass = 0.0
    for seq in product(range(A), repeat=n_samples):
        p = 1.0
        for a in seq:
            p *= logging_probs[a]
        mass += p
        if p == 0.0:
            continue
        r = np.array([rewards_of[a] for a in seq], dtype=np.float64)
        w = kwargs["weights"]
        v = kwargs["model_target_values"]
        q = kwargs["logged_action_values"]
        val = estimator(r, w, v, q)
        total += p * val
    # The enumerator's own self-check: the total probability mass must be
    # exactly 1, otherwise "the expectation" is not an expectation at all.
    assert abs(mass - 1.0) < 1e-12, f"enumerator mass is {mass}, not 1"
    return total


def main():
    # =======================================================================
    # THE FINITE BANDIT EVERYTHING IS CHECKED ON.
    #
    # One state (so the enumeration is over ACTIONS only), three actions,
    # deterministic rewards. Everything below is EXACT rational arithmetic
    # in float64 -- no sampling anywhere.
    # =======================================================================
    pi0 = np.array([0.5, 0.3, 0.2])       # the BEHAVIOUR / logging policy
    pi1 = np.array([0.2, 0.5, 0.3])       # the TARGET policy
    mu = np.array([1.0, 2.0, -1.0])        # the true mean reward per action

    # A WRONG reward model (ARM A) and an EXACT reward model (ARM B).
    rhat_wrong = np.array([0.5, -0.5, 0.25])
    rhat_exact = mu.copy()

    # A WRONG propensity model: pi_0_hat != pi_0 (ARM B).
    pi0_wrong = np.array([0.4, 0.4, 0.2])

    # The target value the estimators are trying to recover.
    v_pi1 = float(np.sum(pi1 * mu))
    print(f"    pi_0 = {pi0},  pi_1 = {pi1},  mu = {mu}")
    print(f"    V_pi1 = sum(pi_1 * mu) = {v_pi1!r}")

    # -----------------------------------------------------------------------
    section("1. THE ENUMERATOR IS ITSELF CORRECT")
    #
    # A broken enumerator would make every downstream check meaningless, so
    # it is checked FIRST, against a hand-computed value.
    #
    # The plain mean of the logged rewards under pi_0, over N=2 samples:
    #     E[mean(r_1, r_2)] = sum_a pi_0(a) mu(a)
    #                        = 0.5*1 + 0.3*2 + 0.2*(-1) = 0.9
    # and the expectation of the MEAN over N=2 is that same quantity, while
    # the expectation of the SUM would be twice it.
    # -----------------------------------------------------------------------
    n_samples = 2
    expected_plain_mean = float(np.sum(pi0 * mu))
    got = enumerate_expectation(
        lambda r, w, v, q: ips_estimate(r, np.ones_like(r)),
        n_samples, pi0, mu,
        weights=np.ones(n_samples),
        model_target_values=np.zeros(n_samples),
        logged_action_values=np.zeros(n_samples),
    )
    print(f"    hand-computed  = {expected_plain_mean!r}")
    print(f"    enumerated     = {got!r}")
    check(abs(got - expected_plain_mean) < 1e-15,
          "1.1 the enumerator reproduces a hand-computed expectation",
          f"got {got} want {expected_plain_mean}")

    # -- AND THE ENUMERATOR MUST NOT BE A SUM. --
    got_sum = enumerate_expectation(
        lambda r, w, v, q: float(np.sum(r)),
        n_samples, pi0, mu,
        weights=np.ones(n_samples),
        model_target_values=np.zeros(n_samples),
        logged_action_values=np.zeros(n_samples),
    )
    check(abs(got_sum - 2.0 * expected_plain_mean) < 1e-15,
          "1.2 the enumerator's SUM control is exactly N times the mean "
          "control (so a mean/sum confusion is visible)",
          f"got {got_sum} want {2.0 * expected_plain_mean}")

    # -----------------------------------------------------------------------
    section("2. THE IMPORTANCE WEIGHT")
    # -----------------------------------------------------------------------
    rho = importance_weights(pi1, pi0)
    print(f"    rho = {rho!r}")
    check(np.allclose(rho, pi1 / pi0, rtol=0, atol=0),
          "2.1 rho is exactly pi_1 / pi_0")
    check(abs(float(np.sum(pi0 * rho)) - 1.0) < 1e-15,
          "2.2 sum_a pi_0(a) rho(a) == 1 (the weights are a valid density ratio)",
          f"got {float(np.sum(pi0 * rho))}")

    # -----------------------------------------------------------------------
    section("3. IPS IS UNBIASED WITH EXACT PROPENSITIES")
    #
    # Checked by EXACT ENUMERATION, so the identity is exact.
    # -----------------------------------------------------------------------
    for n in (2, 3):
        w = np.full(n, rho[0])   # placeholder; replaced per-sequence below
        # The weights DEPEND on which action was logged, so the enumeration
        # needs per-sequence weights. Do it by enumerating explicitly.
        total = 0.0
        mass = 0.0
        for seq in product(range(len(pi0)), repeat=n):
            p = 1.0
            for a in seq:
                p *= pi0[a]
            mass += p
            if p == 0.0:
                continue
            r = np.array([mu[a] for a in seq], dtype=np.float64)
            ww = np.array([rho[a] for a in seq], dtype=np.float64)
            total += p * ips_estimate(r, ww)
        assert abs(mass - 1.0) < 1e-12
        print(f"    N={n}: E[IPS] = {total!r}")
        check(abs(total - v_pi1) < 1e-14,
              f"3.{n} E[IPS] over {n} samples == V_pi1 EXACTLY",
              f"got {total} want {v_pi1}")

    # -- AND WITH WRONG PROPENSITIES, IPS IS BIASED. --
    # A fixture where the two agree cannot discriminate.
    rho_wrong = pi1 / pi0_wrong
    total_w = 0.0
    n = 3
    for seq in product(range(len(pi0)), repeat=n):
        p = 1.0
        for a in seq:
            p *= pi0[a]
        if p == 0.0:
            continue
        r = np.array([mu[a] for a in seq], dtype=np.float64)
        ww = np.array([rho_wrong[a] for a in seq], dtype=np.float64)
        total_w += p * ips_estimate(r, ww)
    print(f"    E[IPS] with WRONG propensities = {total_w!r}")
    check(abs(total_w - v_pi1) > 1e-6,
          "3.4 IPS with WRONG propensities is measurably biased, so the "
          "fixture discriminates")

    # -----------------------------------------------------------------------
    section("4. SNIS: THE DENOMINATOR IS sum rho, NOT N")
    # -----------------------------------------------------------------------
    r_fixed = np.array([1.0, -0.5, 2.0, 0.25])
    w_fixed = np.array([0.8, 0.4, 3.0, 0.0])
    ips = ips_estimate(r_fixed, w_fixed)
    snis = self_normalized_ips(r_fixed, w_fixed)
    identity = ips / float(np.mean(w_fixed))
    print(f"    IPS  = {ips!r}")
    print(f"    SNIS = {snis!r}")
    print(f"    IPS / mean(rho) = {identity!r}")
    check(abs(snis - identity) < 1e-15,
          "4.1 SNIS == IPS / mean(rho) EXACTLY",
          f"snis {snis} identity {identity}")
    check(abs(snis - ips) > 1e-6,
          "4.2 SNIS and IPS DIFFER on this fixture (mean(rho) != 1), so the "
          "fixture discriminates")
    check(abs(ips - float(np.sum(w_fixed * r_fixed)) / 4.0) < 1e-15,
          "4.3 IPS is the MEAN of rho*r over N samples, not the sum",
          f"got {ips}")

    # -- SCALE INVARIANCE. This is the sharpest SNIS/IPS discriminator. --
    for c in (0.25, 3.0, 100.0):
        s_c = self_normalized_ips(r_fixed, c * w_fixed)
        i_c = ips_estimate(r_fixed, c * w_fixed)
        check(abs(s_c - snis) < 1e-12,
              f"4.4 SNIS is SCALE INVARIANT (c={c})",
              f"got {s_c} want {snis}")
        check(abs(i_c - c * ips) < 1e-12,
              f"4.5 IPS scales LINEARLY with c={c} (it is NOT scale invariant)",
              f"got {i_c} want {c * ips}")

    # -----------------------------------------------------------------------
    section("5. THE DIRECT METHOD NEEDS NO WEIGHTS")
    # -----------------------------------------------------------------------
    v_model = np.array([0.1, -0.2, 0.5, 0.05])
    dm = direct_method_estimate(v_model)
    print(f"    DM = {dm!r}")
    check(abs(dm - float(np.mean(v_model))) < 1e-15,
          "5.1 DM is the plain mean of the model-based target values",
          f"got {dm}")
    check(abs(dm - ips) > 1e-6,
          "5.2 DM and IPS differ on this fixture, so the fixture discriminates")

    # -----------------------------------------------------------------------
    section("6. DOUBLE ROBUSTNESS -- BOTH ARMS, EXACTLY")
    # -----------------------------------------------------------------------
    # The per-sample model-based target value Vhat(s) = sum_a pi_1(a) Rhat(a).
    vhat_wrong = float(np.sum(pi1 * rhat_wrong))
    vhat_exact = float(np.sum(pi1 * rhat_exact))
    print(f"    Vhat(s) with the WRONG model = {vhat_wrong!r}")
    print(f"    Vhat(s) with the EXACT model = {vhat_exact!r}")

    def expect_dr(logging_probs, rhat, rho_vec, n):
        total = 0.0
        for seq in product(range(len(logging_probs)), repeat=n):
            p = 1.0
            for a in seq:
                p *= logging_probs[a]
            if p == 0.0:
                continue
            r = np.array([mu[a] for a in seq], dtype=np.float64)
            w = np.array([rho_vec[a] for a in seq], dtype=np.float64)
            v = np.full(n, float(np.sum(pi1 * rhat)))
            q = np.array([rhat[a] for a in seq], dtype=np.float64)
            total += p * doubly_robust_estimate(r, w, v, q)
        return total

    # ARM A: exact propensities, WRONG reward model.
    arm_a = expect_dr(pi0, rhat_wrong, rho, 3)
    print(f"    ARM A (exact propensities, wrong model) = {arm_a!r}")
    check(abs(arm_a - v_pi1) < 1e-14,
          "6.1 ARM A: DR with EXACT propensities and a WRONG reward model is "
          "exactly unbiased",
          f"got {arm_a} want {v_pi1}")

    # ARM B: exact reward model, WRONG propensities.
    arm_b = expect_dr(pi0, rhat_exact, rho_wrong, 3)
    print(f"    ARM B (wrong propensities, exact model) = {arm_b!r}")
    check(abs(arm_b - v_pi1) < 1e-14,
          "6.2 ARM B: DR with an EXACT reward model and WRONG propensities is "
          "exactly unbiased",
          f"got {arm_b} want {v_pi1}")

    # AND WITH BOTH WRONG, DR IS BIASED -- the fixture discriminates.
    arm_c = expect_dr(pi0, rhat_wrong, rho_wrong, 3)
    print(f"    ARM C (both wrong)                    = {arm_c!r}")
    check(abs(arm_c - v_pi1) > 1e-6,
          "6.3 DR with BOTH the propensities and the reward model wrong is "
          "measurably biased, so the fixture discriminates")

    # AND DM IS UNBIASED ONLY WITH THE EXACT MODEL.
    dm_wrong = float(np.sum(pi1 * rhat_wrong))
    dm_exact = float(np.sum(pi1 * rhat_exact))
    check(abs(dm_exact - v_pi1) < 1e-14,
          "6.4 DM with an EXACT reward model is exactly unbiased",
          f"got {dm_exact} want {v_pi1}")
    check(abs(dm_wrong - v_pi1) > 1e-6,
          "6.5 DM with a WRONG reward model is measurably biased (so DR's "
          "extra term is doing real work)")

    # -----------------------------------------------------------------------
    section("7. THE DECOMPOSITION DR == DM + THE CORRECTION")
    # -----------------------------------------------------------------------
    q_logged = np.array([0.3, 0.15, -0.1, 0.4])
    dr = doubly_robust_estimate(r_fixed, w_fixed, v_model, q_logged)
    dm_fixed = direct_method_estimate(v_model)
    correction = float(np.mean(w_fixed * (r_fixed - q_logged)))
    print(f"    DR         = {dr!r}")
    print(f"    DM         = {dm_fixed!r}")
    print(f"    correction = {correction!r}")
    check(abs(dr - (dm_fixed + correction)) < 1e-15,
          "7.1 DR == DM + (1/N) sum_i rho_i (r_i - Rhat_i)",
          f"got {dr}")
    # And the correction is NOT zero on this fixture, so the decomposition
    # is a real test rather than an identity between two equal numbers.
    check(abs(correction) > 1e-6,
          "7.2 the correction term is non-zero on this fixture")

    # -----------------------------------------------------------------------
    section("8. THE EFFECTIVE SAMPLE SIZE")
    # -----------------------------------------------------------------------
    ess = effective_sample_size(w_fixed)
    print(f"    ESS = {ess!r}")
    check(abs(ess - (float(np.sum(w_fixed)) ** 2 / float(np.sum(w_fixed ** 2)))) < 1e-15,
          "8.1 ESS == (sum rho)^2 / sum rho^2")
    check(abs(ess - 4.2 ** 2 / 9.8) < 1e-12,
          "8.2 ESS matches the hand-computed value on the fixed fixture",
          f"got {ess}")
    check(ess <= 4.0 + 1e-12,
          "8.3 ESS <= N", f"got {ess}")
    # Equal weights -> ESS == N.
    ess_equal = effective_sample_size(np.full(5, 2.5))
    check(abs(ess_equal - 5.0) < 1e-12,
          "8.4 ESS == N when all weights are equal (whatever their scale)",
          f"got {ess_equal}")
    # One dominant weight -> ESS -> 1.
    ess_dom = effective_sample_size(np.array([100.0, 0.0, 0.0, 0.0]))
    check(abs(ess_dom - 1.0) < 1e-12,
          "8.5 ESS == 1 when a single weight carries everything",
          f"got {ess_dom}")
    # Scale invariance.
    for c in (0.25, 3.0, 100.0):
        check(abs(effective_sample_size(c * w_fixed) - ess) < 1e-12,
              f"8.6 ESS is SCALE INVARIANT (c={c})")

    # -----------------------------------------------------------------------
    section("9. THE FIXED C++ FIXTURE -- THE VALUES THE C++ TEST PINS")
    #
    # A FIXED logged sample table, so the C++ test has deterministic
    # expected values. These are the numbers the C++ suite asserts.
    # -----------------------------------------------------------------------
    fx_rewards = np.array([1.0, -0.5, 2.0, 0.25])
    fx_target = np.array([0.4, 0.1, 0.6, 0.0])
    fx_behavior = np.array([0.5, 0.25, 0.2, 0.8])
    fx_weights = fx_target / fx_behavior
    fx_vhat = np.array([0.1, -0.2, 0.5, 0.05])
    fx_q = np.array([0.3, 0.15, -0.1, 0.4])

    fx_ips = ips_estimate(fx_rewards, fx_weights)
    fx_snis = self_normalized_ips(fx_rewards, fx_weights)
    fx_dm = direct_method_estimate(fx_vhat)
    fx_dr = doubly_robust_estimate(fx_rewards, fx_weights, fx_vhat, fx_q)
    fx_ess = effective_sample_size(fx_weights)

    print("\n--- pinned values ---")
    print(f"    weights  = {fx_weights!r}")
    print(f"    ips      = {fx_ips!r}")
    print(f"    snis     = {fx_snis!r}")
    print(f"    dm       = {fx_dm!r}")
    print(f"    dr       = {fx_dr!r}")
    print(f"    ess      = {fx_ess!r}")

    # The pinned values must be RE-DERIVED here rather than trusted, so the
    # C++ test's expected numbers are checked against a second route.
    check(abs(fx_ips - 6.6 / 4.0) < 1e-15,
          "9.1 the pinned IPS equals the hand-computed 6.6/4",
          f"got {fx_ips}")
    check(abs(fx_snis - 6.6 / 4.2) < 1e-15,
          "9.2 the pinned SNIS equals the hand-computed 6.6/4.2",
          f"got {fx_snis}")
    check(abs(fx_dm - 0.45 / 4.0) < 1e-15,
          "9.3 the pinned DM equals the hand-computed 0.45/4",
          f"got {fx_dm}")
    check(abs(fx_dr - 7.05 / 4.0) < 1e-15,
          "9.4 the pinned DR equals the hand-computed 7.05/4",
          f"got {fx_dr}")
    check(abs(fx_ess - 17.64 / 9.8) < 1e-12,
          "9.5 the pinned ESS equals the hand-computed 17.64/9.8",
          f"got {fx_ess}")

    print(f"\n=== SUMMARY: {PASSED} passed, {FAILED} failed ===")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
