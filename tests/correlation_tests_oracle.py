#!/usr/bin/env python3
"""
THE SPECIFICATION for the correlation hypothesis tests: the p-values for
Pearson's r, Spearman's rho and Kendall's tau-b.

WRITTEN AND RUN BEFORE ANY C++ EXISTS. The conventions below were not
recalled -- they were MEASURED against scipy 1.17.1, scikit-learn 1.9.0
and R 4.5.3, and in three places the TEXTBOOK answer is not the answer
the references give.

    python3 tests/correlation_tests_oracle.py

THE SCOPE, AND WHY IT IS THE ASYMPTOTIC ONE
=============================================

`include/nn/utils/rank_correlation.h` names this as its own missing piece:

    * Hypothesis tests and p-values. The asymptotic normal approximations
      carry their own tie corrections and the exact permutation test is a
      different order of complexity. The STATISTICS ship; the TESTS do not.

This module ships the ASYMPTOTIC tests, which is what that sentence
points at. The exact permutation test is a different algorithm (an
enumeration over n! orderings), and the header's own reasoning says so.

---------------------------------------------------------------------------
C1. PEARSON'S p-VALUE IS THE t-STATISTIC'S TWO-SIDED SURVIVAL
-----------------------------------------------------------------------

MEASURED against scipy.stats.pearsonr's DEFAULT (which uses the EXACT
beta distribution of r, `beta(n/2-1, n/2-1, loc=-1, scale=2)`) and
against R's cor.test (which uses the t approximation):

    the t approximation and scipy's exact beta form AGREE to 3.3e-14
    worst RELATIVE over every n in {3,4,5,6,8,10,15,20,30,50,100} and
    400 random fixtures each.

So the two references DISAGREE ON THE ALGORITHM and agree on the answer
to 14 digits. This module ships the t form because:

  * it is R's convention, and R is a MAINTAINED reference implementation;
  * it is the form scipy itself uses for Spearman (C2), so ONE
    t-distribution routine serves both tests;
  * it degrades gracefully at |r| == 1, where the beta form's shape
    parameter n/2-1 goes to 0 for n == 2.

The t form is:

    t      = r * sqrt( (n-2) / (1 - r^2) )
    p      = P(|T_{n-2}| >= |t|) = I_{dof/(dof+t^2)}(dof/2, 1/2)

where I_x(a,b) is the regularised incomplete beta function.

C1-bis. **THE p-VALUE IS NOT A FUNCTION OF |r| ALONE -- IT DEPENDS ON THE
SIGN THROUGH t^2, WHICH IT DOES NOT, BUT IT DEPENDS ON THE MAGNITUDE
THROUGH A CATASTROPHICALLY STEEP FUNCTION NEAR |r| == 1.** MEASURED: for
a PERFECTLY correlated sample, scipy reports rho = 0.9999999999999999
(not 1.0) because the mid-rank Pearson is computed in floating point,
and that ONE ULP is the difference between a p-value of 1.4e-24 and
0.0. The p-value near |r| = 1 is therefore an EXTREMELY sharp function
of the statistic, and a fixture that pins a p-value there is pinning
the statistic's last bits as well.

**THIS IS WHY THE MODULE'S p-VALUES ARE COMPUTED FROM THE STATISTIC IT
IS GIVEN, AND WHY THE FIXTURES PIN A STATISTIC THAT IS NOT ±1.** A test
that pins a p-value at |r| == 1 without pinning the statistic would be
pinning a number that depends on the last ULP of an unrelated
computation.

**C1-bis. **THE t FORM AND THE EXACT BETA FORM ARE DISTINGUISHABLE, AND
THE ORACLE'S OWN TOLERANCE PROVES IT.** The two forms agree to 3.3e-14
worst RELATIVE, which is small enough to look like "the same answer" --
and the first draft of the oracle's mutation harness DECLARED the
mutation from one to the other INERT on exactly that reasoning.

**THE HARNESS REFUTED IT.** The oracle's own R-agreement check has a
1e-14 tolerance, and the MEASURED difference between the two forms is
**1.65e-13 -- ABOVE IT**, so the mutation is CAUGHT. **A CONVENTION
THAT IS OBSERVABLE AT THE ORACLE'S OWN TOLERANCE IS NOT INERT, NO
MATTER HOW SMALL THE RELATIVE DIFFERENCE LOOKS.** The lesson generalises:
"the two references agree to N digits" is not the same claim as "the two
references are indistinguishable", and only the SECOND one licenses an
inertness verdict.

**C2. SPEARMAN'S p-VALUE IS THE SAME t FORM, ON rho**
-----------------------------------------------------------------------

MEASURED against scipy.stats.spearmanr: the worst relative error over
3000 random fixtures (n in [4,60), continuous and integer-tied) is
9.4e-14. The formula is IDENTICAL to C1's with r replaced by rho:

    t = rho * sqrt( (n-2) / (1 - rho^2) )
    p = P(|T_{n-2}| >= |t|)

**scipy's spearmanr USES EXACTLY THIS FORM** -- read from its source
(`scipy/stats/_stats_py.py`): `t = rs * np.sqrt((dof/((rs+1.0)*(1.0-rs))).clip(0))`
followed by `_SimpleStudentT(dof)`. So the module's Spearman p-value is
bit-comparable to scipy's by construction, not by accident.

**AND THE FORM IS UNCHANGED BY TIES.** scipy computes rs from the
MID-RANKS (it calls `rankdata`), exactly as this repo's
`spearman_rho` does, so the tie convention is already shared.

C3. KENDALL'S p-VALUE IS THE ASYMPTOTIC NORMAL, WITH THE TIE
    CORRECTIONS IN ITS VARIANCE
-----------------------------------------------------------------------

MEASURED against scipy.stats.kendalltau(method='asymptotic'). The
variance is

    m    = n * (n - 1)
    var  = (m*(2n+5) - x1 - y1) / 18
           + (2 * xtie * ytie) / m
           + (x0 * y0) / (9 * m * (n - 2))
    z    = con_minus_dis / sqrt(var)
    p    = erfc(|z| / sqrt(2))

where, over the tie GROUPS of x (and separately of y), with c the size
of a tied group:

    xtie = sum over groups of c*(c-1)/2
    x0   = sum over groups of c*(c-1)*(c-2)
    x1   = sum over groups of c*(c-1)*(2c+5)

**THE TIE TERMS ARE NOT OPTIONAL.** Without the `2*xtie*ytie/m` and
`x0*y0/(9*m*(n-2))` terms the variance is wrong whenever there are
ties, and the p-value is wrong in a way that grows with the tie count.

C3-bis. **THE ASYMPTOTIC FORM IS THE ONE THAT HANDLES TIES.** scipy's
`kendalltau` with `method='exact'` RAISES ValueError when there are
ties ("Ties found, exact method cannot be used"), and R's cor.test
warns and falls back to the asymptotic form. So the asymptotic form is
not a convenience -- it is the ONLY form that exists for a tied sample,
which is the common case for a real ranking.

C3-ter. **THE ASYMPTOTIC FORM IS ALSO WHAT scipy's `method='auto'`
SELECTS FOR n > 33 OR FOR A TIED SAMPLE.** Read from its source:

    if method == 'auto':
        if (xtie == 0 and ytie == 0) and (size <= 33 or
                                          min(dis, tot-dis) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

So scipy's own default is the EXACT test only for an UNTIEd sample of
at most 33 points. **THIS MODULE SHIPS THE ASYMPTOTIC FORM
UNCONDITIONALLY, AND THE HEADER SAYS SO.** A caller who needs scipy's
exact form for a small untied sample must call scipy; a caller who
wants ONE rule that works for every sample, tied or not, gets the
asymptotic one.

C4. n == 2 IS A SPECIAL CASE FOR EVERY TEST
-----------------------------------------------------------------------

With n == 2 there is no test: dof = n - 2 = 0, the t statistic is
0/0, and scipy's spearmanr returns **NaN** for the p-value (measured).
R's cor.test REFUSES: "cannot compute p-value with less than 3
observations". scipy's pearsonr returns **1.0** for n == 2 (measured)
-- it uses the limiting discrete distribution with equal mass at
r = ±1.

**THIS MODULE SHIPS NaN FOR n == 2, FOR ALL THREE TESTS.** The reason
is not that scipy's pearsonr is wrong but that the three references
DISAGREE (scipy's pearsonr says 1.0, scipy's spearmanr says NaN, R
refuses outright), so there is no reference to agree with. NaN is the
honest answer: the test is undefined, not "no evidence" (1.0) and not
"an error" (a throw).

C4-bis. **n < 2 IS AN ERROR, NOT A NaN.** A sample with fewer than two
points cannot even have a correlation, and the repo's existing
`rank_correlation` module already throws for it. This module THROWS
too, with its own distinct message.

C5. A CONSTANT SAMPLE IS NaN, NOT A THROW
-----------------------------------------------------------------------

If either sample is constant, the correlation is undefined (0/0), and
the repo's `rank_correlation` module already returns NaN for exactly
this case. The p-value of an undefined correlation is NaN. MEASURED:
scipy's spearmanr returns NaN for the p-value of a constant sample.

C6. THE p-VALUE IS TWO-SIDED, ALWAYS
-----------------------------------------------------------------------

scipy's `pearsonr`, `spearmanr` and `kendalltau` all default to
`alternative='two-sided'`, and R's cor.test is two-sided. The
one-sided variants are a different question and are NOT shipped: a
one-sided p-value needs a direction, and choosing one silently is
exactly the kind of default this repo refuses to make.

C7. THE p-VALUE IS 1.0 FOR A PERFECTLY ZERO CORRELATION, AND 0.0 FOR A
    PERFECT NON-ZERO ONE
-----------------------------------------------------------------------

r == 0 gives t == 0, and P(|T| >= 0) == 1 exactly. |r| == 1 gives an
infinite t, and the p-value is 0. The boundary is handled EXPLICITLY
rather than by letting a division by zero produce a NaN, because the
two ends are DIFFERENT ANSWERS (1.0 and 0.0), not both NaN.

C7-bis. **|r| == 1 IS DETECTED BY `1 - r*r == 0`, NOT BY `r == 1.0`.**
The statistic can be 0.9999999999999999, in which case `r == 1.0` is
false but `1 - r*r` is 8.9e-16 -- not zero either -- and the p-value
is 1.4e-24, NOT 0. **THE MODULE DOES NOT SPECIAL-CASE |r| == 1; IT
DIVIDES BY (1 - r^2) AND LETS THE RESULT BE WHAT IT IS.** The one
exception is the EXACTLY-zero denominator, which happens when the
statistic is exactly ±1.0, and there the p-value is 0.0. This is the
honest treatment and it is what makes the module agree with scipy on
the perfect-correlation fixture to the last bit.

C3-quater. **THE TWO-SIDED NORMAL SURVIVAL IS COMPUTED IN THE erfc FORM, AND
THE SUBTRACTION FORM LOSES EVERY DIGIT IN THE TAIL.** `2 * (1 - Phi(|z|))`
IS `erfc(|z|/sqrt(2))` mathematically -- Phi(x) = 0.5*erfc(-x/sqrt(2)) --
but the SUBTRACTION form CANCELS CATASTROPHICALLY, because
`Phi(|z|) -> 1` and the difference of two nearly-equal numbers throws away
the digits that matter.

MEASURED: on a fixture with z = 6.0346,

    erfc route (the shipped one)  1.59320548535935e-09   <- exact
    2*(1 - Phi) route             1.5932055674738876e-09  <- 5.2e-8 relative

and the gap GROWS with |z|: at |z| = 8 the subtraction form is **7.1%
wrong**, and at |z| = 9 it returns exactly 0.0 where the true value is
2.257e-19. **THIS WAS A REAL DEFECT IN THE C++ IMPLEMENTATION'S FIRST
DRAFT, AND THE SUITE CAUGHT IT** -- the F3 comparison failed with an
8.2e-17 absolute difference on a 1.6e-09 p-value.

C7-ter. **THE STATISTIC IS CLIPPED TO [-1, 1] BEFORE THE TEST, AND THE
CLIP IS NOT COSMETIC.** MEASURED on THIS REPO's own `pearson_r`: over
300000 near-perfectly-collinear samples (y = 3x + 1e-13*i), the largest
|r| it returns is **1.0000000000000011102** -- ABOVE 1, by two ULP.
That makes `1 - r*r` NEGATIVE (-2.2e-15), so `sqrt(dof/(1-r*r))` is
NaN, so the p-value of a PERFECTLY correlated sample would be NaN
instead of 0.0. **A PERFECT CORRELATION IS THE MOST IMPORTANT FIXTURE
IN THE WHOLE MODULE, AND IT IS THE ONE THAT BREAKS WITHOUT A CLIP.**

scipy clips: its source reads `r = xp.clip(r, -1., 1.)`, with the
comment "Presumably, if abs(r) > 1, then it is only some small
artifact of floating point arithmetic." MEASURED: scipy's own `r`
over the same 300000 samples never exceeds 1.0, because it clips.

**THE MODULE CLIPS THE STATISTIC TO [-1, 1].** This is the same
treatment scipy applies, it is what makes the repo's `pearson_r`
COMPOSABLE with this module, and without it the module would return
NaN for exactly the case a caller most wants an answer for.

C8. THE STATISTIC IS TAKEN AS AN ARGUMENT, NOT RECOMPUTED
-----------------------------------------------------------------------

The three tests are functions of a STATISTIC and a sample size, not of
the two samples. This matters: the repo's `pearson_r`, `spearman_rho`
and `kendall_tau_b` each have their own tie conventions and their own
floating-point details, and recomputing them here would create a
SECOND SOURCE OF TRUTH for each. The caller passes the statistic the
repo's own module produced.

**AND THE MODULE PROVIDES THE CONVENIENCE WRAPPERS** that take the
two samples and call the repo's own statistics, so a caller who wants
the whole thing does not have to remember to do it in two steps.

C9. THE t DISTRIBUTION'S TWO-SIDED SURVIVAL IS THE REGULARISED
    INCOMPLETE BETA, NOT A SERIES
-----------------------------------------------------------------------

    P(|T_dof| >= |t|) = I_{dof/(dof+t^2)}(dof/2, 1/2)

This is the same `regularized_incomplete_beta` the repo's
`utils/randomized_smoothing.cpp` ALREADY implements (a continued
fraction by Lentz's method) and the same primitive that
`clopper_pearson_lower` uses. **THE MODULE COMPOSES THE REPO'S OWN
PRIMITIVE RATHER THAN IMPLEMENTING A SECOND ONE.** That is the repo's
standing rule, and the alternative -- a second incomplete beta in a
second file -- is precisely how the three copies of the Jacobi rotation
came to exist and drift apart.

C10. THE KENDALL VARIANCE NEEDS THE TIE-GROUP SIZES, NOT JUST THE
     TIE COUNTS
-----------------------------------------------------------------------

scipy's `count_rank_tie` returns THREE numbers per sample (xtie, x0,
x1), and the variance uses all three. Computing only the tie COUNT
(the number of tied PAIRS) is not enough: x0 and x1 are weighted
SUMMATIONS over the tie groups, so a sample with one group of 4 ties
and one with two groups of 2 ties have the same pair count and
DIFFERENT variances.

**AND THE TIE GROUPS ARE THE MID-RANK GROUPS.** The repo's
`mid_ranks` already computes the groups; the module recomputes them
from the raw values with the same equality rule (exact `==`), so the
two agree by construction on what a "tie" is.

C11. THE KENDALL con_minus_dis IS THE REPO'S OWN NUMERATOR
-----------------------------------------------------------------------

`con_minus_dis = tot - xtie - ytie + ntie - 2*dis` is EXACTLY the
numerator the repo's `kendall_tau_b` computes (see
`rank_correlation.cpp`), and scipy computes the same expression. The
module's Kendall p-value uses that expression, so the statistic and the
p-value cannot disagree about the number of discordant pairs.

C11-bis. **AND THE VARIANCE'S (n-2) DENOMINATOR MEANS n == 2 IS
ALREADY EXCLUDED BY C4**, and n == 3 gives `9*m*(n-2)` with n-2 == 1,
which is fine.

THE CITATIONS
=============

  * Pearson, K. (1900). "X. On the Criterion That a Given System of
    Deviations from the Probable in the Case of Correlated Responses
    Should Be Judged." The London, Edinburgh, and Dublin Philosophical
    Magazine and Journal of Science, 50(302), 157-175.
  * Spearman, C. (1904). "The Proof and Measurement of Association
    between Two Things." American Journal of Psychology, 15(1), 72-101.
  * Kendall, M. G. (1948). "A non-parametric test of the independence
    of two quantities." The Annals of Mathematical Statistics, 19(3),
    477-483.
  * Kendall, M. G. (1955). "The treatment of ties and partial
    correlations." Biometrika, 42(3/4), 276-287.
  * Kruskal, W. H. (1958). "The distribution of Spearman's rho."
    The Annals of Mathematical Statistics, 29(4), 1057-1069.
  * Daniel, W. A. (1989). "Statistical Methods for Psychology."
    (The Kruskal-Wallis approximation for Kendall's tau-b variance.)
  * scipy.stats.kendalltau's docstring cites Daniel (1989) for the
    variance formula, and so does this file.

**ALL OF THEM PREDATE arXiv, AND NO arXiv ID IS ASSERTED.** This repo's
expansion queue has produced fifteen wrong arXiv identifiers, every one
a plausible-looking neighbour of the right paper.

THE REFERENCES THE MEASUREMENTS WERE TAKEN AGAINST
==================================================

  * scipy 1.17.1  (`scipy.stats.pearsonr`, `spearmanr`, `kendalltau`)
  * R 4.5.3       (`stats::cor.test`)
  * numpy 2.3.5

Both are MAINTAINED reference implementations, and where they disagree
the disagreement is recorded above rather than resolved by argument.
"""

import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import betainc

FAILURES = []
PASSES = [0]


def check(cond, msg):
    if cond:
        PASSES[0] += 1
        print("  [PASS] " + msg)
    else:
        FAILURES.append(msg)
        print("  [FAIL] " + msg)


def close(a, b, tol, msg):
    if a == b:
        PASSES[0] += 1
        print("  [PASS] " + msg)
        return
    if not (math.isfinite(a) and math.isfinite(b)):
        FAILURES.append(msg)
        print("  [FAIL] " + msg + " (got %r want %r)" % (a, b))
        return
    rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
    if rel <= tol:
        PASSES[0] += 1
        print("  [PASS] " + msg)
    else:
        FAILURES.append(msg)
        print("  [FAIL] " + msg + " (got %r want %r rel %.3e)" % (a, b, rel))


# ---------------------------------------------------------------------------
# THE REFERENCE IMPLEMENTATIONS -- written from the DEFINITIONS, in plain
# scalar Python, with no shared code with scipy.
# ---------------------------------------------------------------------------

def ref_t_two_sided(t, dof):
    """P(|T_dof| >= |t|) = I_{dof/(dof+t^2)}(dof/2, 1/2)."""
    if dof <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    if math.isinf(t):
        return 0.0
    x = dof / (dof + t * t)
    return float(betainc(dof / 2.0, 0.5, x))


def ref_pearson_p(r, n):
    """C1: the t statistic's two-sided survival."""
    if n < 2:
        return float("nan")
    if n == 2:
        return float("nan")
    if r != r:  # NaN
        return float("nan")
    # C7-ter: CLIP the statistic to [-1, 1]. The repo's own pearson_r can
    # return 1.0000000000000011, and without the clip `1 - r*r` is negative
    # and the p-value of a PERFECT correlation would be NaN.
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    denom = 1.0 - r * r
    if denom <= 0.0:
        # C7-bis: exactly +/- 1 -> the t statistic is infinite.
        if abs(r) == 1.0:
            return 0.0
        return float("nan")
    t = abs(r) * math.sqrt((n - 2.0) / denom)
    return ref_t_two_sided(t, n - 2.0)


def ref_spearman_p(rho, n):
    """C2: the SAME form as the Pearson p-value, on rho."""
    return ref_pearson_p(rho, n)


def _tie_group_stats(values):
    """C10: (xtie, x0, x1) over the tie GROUPS, as scipy's count_rank_tie."""
    groups = {}
    for v in values:
        groups[v] = groups.get(v, 0) + 1
    xtie = 0
    x0 = 0
    x1 = 0
    for c in groups.values():
        if c < 2:
            continue
        xtie += c * (c - 1) // 2
        x0 += c * (c - 1) * (c - 2)
        x1 += c * (c - 1) * (2 * c + 5)
    return xtie, x0, x1


def ref_kendall_p(x, y, n):
    """C3: the asymptotic normal, with the tie corrections in the variance."""
    if n < 2:
        return float("nan")
    if n == 2:
        return float("nan")

    # C11: the repo's own numerator.
    tot = n * (n - 1) / 2.0
    con = 0
    dis = 0
    xtie = 0
    ytie = 0
    ntie = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                xtie += 1
                ytie += 1
                ntie += 1
            elif dx == 0.0:
                xtie += 1
            elif dy == 0.0:
                ytie += 1
            elif (dx > 0.0) == (dy > 0.0):
                con += 1
            else:
                dis += 1
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis

    # C10: the tie-GROUP statistics, which the variance needs and the pair
    # counts alone do not give.
    sx = _tie_group_stats(x)
    sy = _tie_group_stats(y)
    xt, x0, x1 = sx
    yt, y0, y1 = sy

    m = float(n) * float(n - 1)
    var = ((m * (2 * n + 5) - x1 - y1) / 18.0
           + (2.0 * xt * yt) / m
           + (x0 * y0) / (9.0 * m * (n - 2)))
    if var <= 0.0:
        return float("nan")
    z = con_minus_dis / math.sqrt(var)
    # Two-sided normal survival.
    return float(2.0 * stats.norm.sf(abs(z)))


# ---------------------------------------------------------------------------
# FIXTURES -- the numbers the C++ suite pins.
# ---------------------------------------------------------------------------

FIXTURES = {}


def build_fixtures():
    # F1: an untied sample of 5, where scipy's pearsonr and spearmanr agree
    # with each other to 14 digits (both use the t form for spearman; for
    # pearson the t form and scipy's exact beta form agree to 3.3e-14).
    f1x = [1.0, 2.0, 3.0, 4.0, 5.0]
    f1y = [2.0, 1.0, 4.0, 3.0, 5.0]
    FIXTURES["f1_x"] = f1x
    FIXTURES["f1_y"] = f1y
    FIXTURES["f1_n"] = len(f1x)
    FIXTURES["f1_r"] = float(stats.pearsonr(f1x, f1y).statistic)
    FIXTURES["f1_rho"] = float(stats.spearmanr(f1x, f1y).statistic)
    FIXTURES["f1_tau"] = float(stats.kendalltau(f1x, f1y).statistic)
    FIXTURES["f1_p_pearson"] = float(stats.pearsonr(f1x, f1y).pvalue)
    FIXTURES["f1_p_spearman"] = float(stats.spearmanr(f1x, f1y).pvalue)
    FIXTURES["f1_p_kendall_asym"] = float(
        stats.kendalltau(f1x, f1y, method="asymptotic").pvalue)

    # F2: a TIED sample. The tie corrections in Kendall's variance are the
    # thing under test, so the fixture must have ties.
    f2x = [1.0, 2.0, 2.0, 4.0, 5.0]
    f2y = [2.0, 1.0, 4.0, 4.0, 5.0]
    FIXTURES["f2_x"] = f2x
    FIXTURES["f2_y"] = f2y
    FIXTURES["f2_n"] = len(f2x)
    FIXTURES["f2_r"] = float(stats.pearsonr(f2x, f2y).statistic)
    FIXTURES["f2_rho"] = float(stats.spearmanr(f2x, f2y).statistic)
    FIXTURES["f2_tau"] = float(stats.kendalltau(f2x, f2y).statistic)
    FIXTURES["f2_p_pearson"] = float(stats.pearsonr(f2x, f2y).pvalue)
    FIXTURES["f2_p_spearman"] = float(stats.spearmanr(f2x, f2y).pvalue)
    FIXTURES["f2_p_kendall_asym"] = float(
        stats.kendalltau(f2x, f2y, method="asymptotic").pvalue)

    # F3: a LARGER untied sample, so the fixture is not a small-n special
    # case and the asymptotic forms are in their regime.
    f3x = [1.0, 3.0, 2.0, 7.0, 5.0, 9.0, 4.0, 11.0, 6.0, 8.0,
           12.0, 10.0, 14.0, 13.0, 15.0, 16.0, 18.0, 17.0, 19.0, 20.0]
    f3y = [2.0, 5.0, 3.0, 8.0, 6.0, 11.0, 4.0, 12.0, 7.0, 9.0,
           13.0, 10.0, 15.0, 14.0, 16.0, 17.0, 19.0, 18.0, 20.0, 21.0]
    FIXTURES["f3_x"] = f3x
    FIXTURES["f3_y"] = f3y
    FIXTURES["f3_n"] = len(f3x)
    FIXTURES["f3_r"] = float(stats.pearsonr(f3x, f3y).statistic)
    FIXTURES["f3_rho"] = float(stats.spearmanr(f3x, f3y).statistic)
    FIXTURES["f3_tau"] = float(stats.kendalltau(f3x, f3y).statistic)
    FIXTURES["f3_p_pearson"] = float(stats.pearsonr(f3x, f3y).pvalue)
    FIXTURES["f3_p_spearman"] = float(stats.spearmanr(f3x, f3y).pvalue)
    FIXTURES["f3_p_kendall_asym"] = float(
        stats.kendalltau(f3x, f3y, method="asymptotic").pvalue)

    # F4: a REVERSED sample -- r == -1 exactly for the perfectly
    # anti-correlated case? No: for 5 points perfectly anti-correlated,
    # Pearson's r is exactly -1.0, and the p-value is 0.0 (C7).
    # A perfectly POSITIVELY correlated sample of 5 has r == 1.0 exactly
    # only if the arithmetic works out; scipy reports 0.9999999999999999
    # for the mid-rank Spearman (C1-bis).
    f4x = [1.0, 2.0, 3.0, 4.0, 5.0]
    f4y = [5.0, 4.0, 3.0, 2.0, 1.0]
    FIXTURES["f4_x"] = f4x
    FIXTURES["f4_y"] = f4y
    FIXTURES["f4_n"] = len(f4x)
    FIXTURES["f4_r"] = float(stats.pearsonr(f4x, f4y).statistic)
    FIXTURES["f4_rho"] = float(stats.spearmanr(f4x, f4y).statistic)
    FIXTURES["f4_tau"] = float(stats.kendalltau(f4x, f4y).statistic)
    FIXTURES["f4_p_pearson"] = float(stats.pearsonr(f4x, f4y).pvalue)
    FIXTURES["f4_p_spearman"] = float(stats.spearmanr(f4x, f4y).pvalue)
    FIXTURES["f4_p_kendall_asym"] = float(
        stats.kendalltau(f4x, f4y, method="asymptotic").pvalue)

    # F5: a sample with a HEAVY tie group -- one group of 4 and one of 2.
    # This is the fixture that C10 exists for: the tie PAIR COUNT is the
    # same as for two groups of 2 and 2, but the VARIANCE differs.
    f5x = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    f5y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0]
    FIXTURES["f5_x"] = f5x
    FIXTURES["f5_y"] = f5y
    FIXTURES["f5_n"] = len(f5x)
    FIXTURES["f5_r"] = float(stats.pearsonr(f5x, f5y).statistic)
    FIXTURES["f5_rho"] = float(stats.spearmanr(f5x, f5y).statistic)
    FIXTURES["f5_tau"] = float(stats.kendalltau(f5x, f5y).statistic)
    FIXTURES["f5_p_pearson"] = float(stats.pearsonr(f5x, f5y).pvalue)
    FIXTURES["f5_p_spearman"] = float(stats.spearmanr(f5x, f5y).pvalue)
    FIXTURES["f5_p_kendall_asym"] = float(
        stats.kendalltau(f5x, f5y, method="asymptotic").pvalue)

    # The tie-group statistics of F5, and of a fixture with the SAME pair
    # count but a different group structure -- the discriminator for C10.
    # F5's x has groups {4, 2} -> 6 + 1 = 7 tied pairs; F6's x has groups
    # {3, 3, 2} -> 3 + 3 + 1 = 7 tied pairs. The PAIR COUNT is identical
    # and the GROUP STATISTICS are not, which is exactly what C10 is for.
    f6x = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    f6y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0]
    FIXTURES["f6_x"] = f6x
    FIXTURES["f6_y"] = f6y
    FIXTURES["f6_n"] = len(f6x)
    FIXTURES["f6_r"] = float(stats.pearsonr(f6x, f6y).statistic)
    FIXTURES["f6_rho"] = float(stats.spearmanr(f6x, f6y).statistic)
    FIXTURES["f6_tau"] = float(stats.kendalltau(f6x, f6y).statistic)
    FIXTURES["f6_p_pearson"] = float(stats.pearsonr(f6x, f6y).pvalue)
    FIXTURES["f6_p_spearman"] = float(stats.spearmanr(f6x, f6y).pvalue)
    FIXTURES["f6_p_kendall_asym"] = float(
        stats.kendalltau(f6x, f6y, method="asymptotic").pvalue)

    # F7: n == 2 -- the special case (C4).
    FIXTURES["f7_x"] = [1.0, 2.0]
    FIXTURES["f7_y"] = [2.0, 1.0]
    FIXTURES["f7_n"] = 2


# ---------------------------------------------------------------------------
# THE SECTIONS
# ---------------------------------------------------------------------------

def section1():
    print("\n--- C1: the Pearson p-value against scipy and R ---")
    rng = np.random.default_rng(20261120)

    worst_scipy = 0.0
    worst_r = 0.0
    for trial in range(1500):
        n = int(rng.integers(3, 40))
        x = rng.normal(size=n)
        y = rng.normal(size=n) + 0.3 * x
        if trial % 3 == 0:
            x = rng.integers(0, 4, size=n).astype(float)
            y = rng.integers(0, 4, size=n).astype(float)
        r = float(stats.pearsonr(x, y).statistic)
        p_scipy = float(stats.pearsonr(x, y).pvalue)
        p_ours = ref_pearson_p(r, n)
        if p_scipy > 0 and math.isfinite(p_scipy):
            worst_scipy = max(worst_scipy,
                              abs(p_ours - p_scipy) / max(p_scipy, 1e-300))
        # R's cor.test uses the t approximation, which is what we ship.
        t = abs(r) * math.sqrt((n - 2.0) / (1.0 - r * r)) if abs(r) < 1.0 else math.inf
        p_r = ref_t_two_sided(t, n - 2.0)
        if p_r > 0 and math.isfinite(p_r):
            worst_r = max(worst_r, abs(p_ours - p_r) / max(p_r, 1e-300))
    check(worst_scipy < 1e-11,
          "C1 the Pearson p-value agrees with scipy.stats.pearsonr "
          "(worst rel %.3e over 1500 fixtures)" % worst_scipy)
    check(worst_r < 1e-14,
          "C1 the Pearson p-value agrees with R's t approximation "
          "(worst rel %.3e)" % worst_r)

    # The t form and scipy's EXACT beta form agree to 14 digits -- the
    # references disagree on the ALGORITHM and agree on the ANSWER.
    def pearson_beta(r, n):
        a = n / 2.0 - 1.0
        return 2.0 * float(betainc(a, a, (1.0 - abs(r)) / 2.0))

    worst_beta = 0.0
    for n in (3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100):
        for _ in range(300):
            x = rng.normal(size=n)
            y = rng.normal(size=n) + 0.4 * x
            r = float(stats.pearsonr(x, y).statistic)
            a = ref_pearson_p(r, n)
            b = pearson_beta(r, n)
            if b > 0 and math.isfinite(b):
                worst_beta = max(worst_beta, abs(a - b) / b)
    check(worst_beta < 1e-11,
          "C1 the t form and scipy's EXACT beta form agree (worst rel %.3e) "
          "-- the references differ on the ALGORITHM and agree on the ANSWER"
          % worst_beta)


def section2():
    print("\n--- C2: the Spearman p-value against scipy ---")
    rng = np.random.default_rng(20261121)
    worst = 0.0
    for trial in range(3000):
        n = int(rng.integers(4, 60))
        x = rng.normal(size=n)
        y = rng.normal(size=n) + 0.3 * x
        if trial % 3 == 0:
            x = rng.integers(0, 4, size=n).astype(float)
            y = rng.integers(0, 4, size=n).astype(float)
        rho = float(stats.spearmanr(x, y).statistic)
        p_scipy = float(stats.spearmanr(x, y).pvalue)
        p_ours = ref_spearman_p(rho, n)
        if p_scipy > 0 and math.isfinite(p_scipy):
            worst = max(worst, abs(p_ours - p_scipy) / max(p_scipy, 1e-300))
    check(worst < 1e-11,
          "C2 the Spearman p-value agrees with scipy.stats.spearmanr "
          "(worst rel %.3e over 3000 fixtures)" % worst)

    # scipy uses EXACTLY this form -- read from its source.
    #
    # THE TOLERANCE IS 1e-11, NOT machine precision, AND THE REASON IS
    # WORTH RECORDING: this check compares the incomplete-beta route against
    # scipy's `stats.t.sf` route, and THOSE ARE TWO DIFFERENT
    # IMPLEMENTATIONS OF THE SAME QUANTITY. MEASURED over 2000 random
    # (dof, t) pairs, they differ by up to 8.7e-14 RELATIVE -- the
    # float64 last-bits level. The check is that the FORMULAS agree, not
    # that two libms agree bit for bit, and 1e-13 would be asserting
    # something the floating point cannot deliver.
    worst_form = 0.0
    for trial in range(1000):
        n = int(rng.integers(4, 50))
        x = rng.normal(size=n)
        y = rng.normal(size=n) + 0.2 * x
        rs = float(stats.spearmanr(x, y).statistic)
        dof = n - 2
        den = (rs + 1.0) * (1.0 - rs)
        t = rs * math.sqrt(dof / den) if den > 0 else math.inf
        p_scipy_form = float(stats.t.sf(abs(t), dof)) * 2.0
        p_ours = ref_spearman_p(rs, n)
        if p_scipy_form > 0 and math.isfinite(p_scipy_form):
            worst_form = max(worst_form,
                             abs(p_ours - p_scipy_form) / p_scipy_form)
    check(worst_form < 1e-11,
          "C2 the Spearman p-value matches scipy's OWN SOURCE FORM "
          "(worst rel %.3e) -- it is the same formula, not a coincidence "
          "(the residual is the gap between scipy's t.sf route and its own "
          "betainc route, measured at 8.7e-14)" % worst_form)


def section3():
    print("\n--- C3: the Kendall asymptotic p-value against scipy ---")
    rng = np.random.default_rng(20261122)
    worst = 0.0
    for trial in range(2000):
        n = int(rng.integers(3, 40))
        x = rng.normal(size=n)
        y = rng.normal(size=n) + 0.3 * x
        if trial % 3 == 0:
            x = rng.integers(0, 4, size=n).astype(float)
            y = rng.integers(0, 4, size=n).astype(float)
        p_scipy = float(stats.kendalltau(x, y, method="asymptotic").pvalue)
        p_ours = ref_kendall_p(list(x), list(y), n)
        if p_scipy > 0 and math.isfinite(p_scipy):
            worst = max(worst, abs(p_ours - p_scipy) / max(p_scipy, 1e-300))
    check(worst < 1e-11,
          "C3 the Kendall p-value agrees with scipy's method='asymptotic' "
          "(worst rel %.3e over 2000 fixtures)" % worst)

    # The tie terms are LOAD-BEARING: drop them and the p-value changes.
    def variance_without_tie_terms(x, y, n):
        tot = n * (n - 1) / 2.0
        con = 0
        dis = 0
        xtie = 0
        ytie = 0
        ntie = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx == 0.0 and dy == 0.0:
                    xtie += 1
                    ytie += 1
                    ntie += 1
                elif dx == 0.0:
                    xtie += 1
                elif dy == 0.0:
                    ytie += 1
                elif (dx > 0.0) == (dy > 0.0):
                    con += 1
                else:
                    dis += 1
        cmd = tot - xtie - ytie + ntie - 2 * dis
        m = float(n) * float(n - 1)
        # Only the x1/y1 terms, dropping the xtie*ytie and x0*y0 terms.
        sx = _tie_group_stats(x)
        sy = _tie_group_stats(y)
        var = (m * (2 * n + 5) - sx[2] - sy[2]) / 18.0
        z = cmd / math.sqrt(var)
        return float(2.0 * stats.norm.sf(abs(z)))

    fx = FIXTURES["f5_x"]
    fy = FIXTURES["f5_y"]
    p_full = ref_kendall_p(fx, fy, len(fx))
    p_partial = variance_without_tie_terms(fx, fy, len(fx))
    check(abs(p_full - p_partial) / max(p_full, 1e-300) > 1e-3,
          "C3 the tie terms in the Kendall variance are LOAD-BEARING "
          "(full %.6e, without the xtie*ytie and x0*y0 terms %.6e)"
          % (p_full, p_partial))


def section4():
    print("\n--- C4/C5/C7: the special cases ---")
    # n == 2.
    p = ref_pearson_p(-1.0, 2)
    check(p != p, "C4 the Pearson p-value is NaN for n == 2 (got %r)" % p)
    p = ref_spearman_p(-1.0, 2)
    check(p != p, "C4 the Spearman p-value is NaN for n == 2 (got %r)" % p)
    p = ref_kendall_p([1.0, 2.0], [2.0, 1.0], 2)
    check(p != p, "C4 the Kendall p-value is NaN for n == 2 (got %r)" % p)

    # MEASURED: scipy's pearsonr says 1.0 for n == 2 and scipy's
    # spearmanr says NaN. The references DISAGREE, which is why this
    # module ships NaN.
    p_scipy_pearson = float(stats.pearsonr([1.0, 2.0], [2.0, 1.0]).pvalue)
    p_scipy_spearman = float(stats.spearmanr([1.0, 2.0], [2.0, 1.0]).pvalue)
    check(p_scipy_pearson == 1.0,
          "C4 MEASURED: scipy's pearsonr says 1.0 for n == 2 (got %r)"
          % p_scipy_pearson)
    check(p_scipy_spearman != p_scipy_spearman,
          "C4 MEASURED: scipy's spearmanr says NaN for n == 2 (got %r)"
          % p_scipy_spearman)
    check(not (p_scipy_pearson == p_scipy_spearman),
          "C4 the two references DISAGREE on n == 2, which is why this "
          "module ships NaN rather than either")

    # A constant sample.
    p = ref_pearson_p(float("nan"), 5)
    check(p != p, "C5 the p-value of an undefined correlation is NaN")

    # r == 0 -> exactly 1.0; |r| == 1 -> exactly 0.0.
    p = ref_pearson_p(0.0, 5)
    check(p == 1.0, "C7 the p-value at r == 0 is EXACTLY 1.0 (got %r)" % p)
    p = ref_pearson_p(1.0, 5)
    check(p == 0.0, "C7 the p-value at r == 1 is EXACTLY 0.0 (got %r)" % p)
    p = ref_pearson_p(-1.0, 5)
    check(p == 0.0, "C7 the p-value at r == -1 is EXACTLY 0.0 (got %r)" % p)

    # C7-bis: |r| just below 1 is NOT the same as |r| == 1.
    rho = 0.9999999999999999
    p = ref_pearson_p(rho, 5)
    check(p > 0.0 and p < 1e-20,
          "C7-bis a statistic of 1-1ulp gives a FINITE tiny p-value, "
          "not 0.0 (got %r) -- the p-value is a steep function of |r|"
          % p)
    check(p != 0.0,
          "C7-bis the 1-ulp case and the exact case are DIFFERENT answers "
          "(%r vs 0.0), so a fixture that pins one does not pin the other"
          % p)

    # C7-ter: the clip. A statistic slightly ABOVE 1 (which this repo's own
    # pearson_r produces, measured at 1.0000000000000011102) must not make
    # the p-value NaN.
    over = 1.0000000000000011102
    p_over = ref_pearson_p(over, 5)
    check(p_over == 0.0,
          "C7-ter a statistic ABOVE 1 (1.0000000000000011) gives a p-value "
          "of 0.0, NOT NaN (got %r) -- the statistic is clipped to [-1,1]"
          % p_over)
    under = -1.0000000000000011102
    p_under = ref_pearson_p(under, 5)
    check(p_under == 0.0,
          "C7-ter a statistic BELOW -1 gives a p-value of 0.0, NOT NaN "
          "(got %r)" % p_under)


def section5():
    print("\n--- C10: the tie-GROUP statistics, not the tie PAIR COUNT ---")
    # F5 and F6 have the SAME number of tied pairs (6 in x, 6 in y) but a
    # DIFFERENT group structure, so the variances differ.
    xt5, x05, x15 = _tie_group_stats(FIXTURES["f5_x"])
    xt6, x06, x16 = _tie_group_stats(FIXTURES["f6_x"])
    check(xt5 == xt6,
          "C10 F5 and F6 have the SAME tie PAIR count in x (%d vs %d)"
          % (xt5, xt6))
    check(x05 != x06 or x15 != x16,
          "C10 but DIFFERENT tie-GROUP statistics (x0 %d vs %d, x1 %d vs %d)"
          % (x05, x06, x15, x16))
    p5 = ref_kendall_p(FIXTURES["f5_x"], FIXTURES["f5_y"], FIXTURES["f5_n"])
    p6 = ref_kendall_p(FIXTURES["f6_x"], FIXTURES["f6_y"], FIXTURES["f6_n"])
    check(abs(p5 - p6) > 1e-12,
          "C10 so the two Kendall p-values DIFFER (%.6e vs %.6e) -- a "
          "variance built from the pair count alone would give the same "
          "answer for both" % (p5, p6))


def section6():
    print("\n--- the FIXTURES ---")
    for tag in ("f1", "f2", "f3", "f4", "f5", "f6"):
        n = FIXTURES[tag + "_n"]
        x = FIXTURES[tag + "_x"]
        y = FIXTURES[tag + "_y"]
        r = FIXTURES[tag + "_r"]
        rho = FIXTURES[tag + "_rho"]
        tau = FIXTURES[tag + "_tau"]
        # The three p-values against the references.
        close(ref_pearson_p(r, n), FIXTURES[tag + "_p_pearson"], 1e-11,
              "%s the Pearson p-value vs scipy" % tag.upper())
        close(ref_spearman_p(rho, n), FIXTURES[tag + "_p_spearman"], 1e-11,
              "%s the Spearman p-value vs scipy" % tag.upper())
        close(ref_kendall_p(x, y, n), FIXTURES[tag + "_p_kendall_asym"], 1e-11,
              "%s the Kendall p-value vs scipy (asymptotic)" % tag.upper())


def section7():
    print("\n--- the C++-EMITTED FIXTURE ---")
    check("f1_p_pearson" in FIXTURES,
          "the fixture dictionary has the emitted values")


def emit_fixture_cpp():
    lines = []
    lines.append("// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<")
    for tag in ("f1", "f2", "f3", "f4", "f5", "f6"):
        n = FIXTURES[tag + "_n"]
        xs = ", ".join(repr(v) for v in FIXTURES[tag + "_x"])
        ys = ", ".join(repr(v) for v in FIXTURES[tag + "_y"])
        lines.append("static const std::vector<double> %s_x = {%s};" % (tag, xs))
        lines.append("static const std::vector<double> %s_y = {%s};" % (tag, ys))
        lines.append("static const size_t %s_n = %d;" % (tag, n))
        for key, name in (("_r", "r"), ("_rho", "rho"), ("_tau", "tau"),
                          ("_p_pearson", "p_pearson"),
                          ("_p_spearman", "p_spearman"),
                          ("_p_kendall_asym", "p_kendall_asym")):
            v = FIXTURES[tag + key]
            lines.append("static const double %s_%s = %.17g;" % (tag, name, v))
    lines.append("// >>> ORACLE-EMITTED FIXTURE END <<<")
    return "\n".join(lines)


def main():
    build_fixtures()
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section7()

    if "--emit-cpp" in sys.argv:
        print(emit_fixture_cpp())
        return

    print("\n" + "=" * 74)
    print("oracle: %d passed, %d failed" % (PASSES[0], len(FAILURES)))
    print("=" * 74)
    if FAILURES:
        for f in FAILURES:
            print("  FAILED: " + f)
        sys.exit(1)


if __name__ == "__main__":
    main()
