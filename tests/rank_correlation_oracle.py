#!/usr/bin/env python3
"""
tests/rank_correlation_oracle.py -- the float64 numpy SPECIFICATION for the
rank correlation and Gaussian-copula utilities.

THIS ORACLE IS THE SPECIFICATION. It was written and run BEFORE any C++ for
this feature existed. Its discriminating power is MEASURED, not narrated:
tests/rank_correlation_mutate_oracle.py breaks this oracle in seven specific
ways and requires that it FAIL each time (see that file).

WHY THIS FEATURE, MEASURED NOT ASSUMED (2026-11-10)
=====================================================

    grep -rinE 'copula|rank.?correl|spearman|kendall' include/ tests/
    -> 0 content hits

    grep -rinE '\\bpearson\\b|\\bcorrelation\\b' include/nn/utils/*.h
    -> only "Clopper-Pearson" (a binomial bound, a DIFFERENT object)

So the repo has ~250 layers and 60+ utility files and NO correlation
machinery of any kind: not Pearson's r, not Spearman's rho, not Kendall's
tau, and nothing copula-shaped.

What it HAS is adjacent but different:
  * `utils/metrics.h`      -- ROC-AUC, AP, NDCG, MAP, MRR. RANKING QUALITY
                              of a model against labels. Not the AGREEMENT
                              between two rankings, which is what a rank
                              correlation measures.
  * `utils/self_supervised_losses.h` -- a cross-correlation MATRIX for the
                              Barlow-Twins objective. A Gram matrix, not a
                              rank statistic.
  * `utils/linear_algebra.h` -- the Frechet distance. A distance between two
                              GAUSSIANS, again not a rank statistic.

WHY "COPULA" IS IN THE BULLET, AND WHAT THE COPULA PART ACTUALLY IS
===================================================================

A copula is the dependence structure of a joint distribution with the
marginals removed. The practical consequence -- and the reason this is worth
building -- is the MARGINAL-FREENESS property:

    a copula statistic is INVARIANT under any strictly increasing
    transformation of either marginal.

That is checkable to machine precision, and it is exactly what separates a
copula statistic from a linear one. Pearson's r of the RAW values is NOT
invariant (a monotone squashing of one marginal changes it); Pearson's r of
the NORMAL SCORES IS invariant, because the normal-scores transform is a
monotone function of the marginal.

THE FOUR EXACT POPULATION CORRESPONDENCES
==========================================

For a bivariate GAUSSIAN with correlation rho (equivalently: the Gaussian
copula with parameter rho), the POPULATION rank statistics are

    Kendall's tau   tau   = (2/pi) * arcsin(rho)
    Spearman's rho   rho_S = (6/pi) * arcsin(rho/2)

and their inverses are

    rho = sin(pi * tau / 2)
    rho = 2 * sin(pi * rho_S / 6)

**THE QUADRATURE CHECK BELOW IS THE POINT.** Each closed form is verified
against an INDEPENDENT numerical computation that does not use the closed
form at all:

  * tau is verified against  E[sign(U) sign(V)]  for jointly standard normal
    (U, V) with correlation rho, computed by 2-D QUADRATURE of the
    bivariate normal density over the first quadrant. That integral is what
    a concordance probability IS.
  * rho_S is verified against  12 * int int Phi(x) Phi(y) phi_2(x,y;rho) - 3,
    the standard integral form of the population Spearman coefficient,
    again by 2-D quadrature.

A closed form checked against a quadrature that does not contain it is
evidence. A closed form checked against itself is not.

DO NOT USE scipy.stats.spearmanr / kendalltau AS THE ORACLE
============================================================

They are used here as a CROSS-CHECK against a MAINTAINED REFERENCE
IMPLEMENTATION, which is a different and stronger role than being the
specification. The specification below is written from the DEFINITIONS, so a
shared misreading would have to be made independently in two places.

THE TIE CONVENTIONS, WHICH ARE THE WHOLE GAME HERE
===================================================

Three separate tie conventions are in play and they are EASY TO CONFLATE:

 1. RANKS. A tied group receives the AVERAGE of the ranks it spans
    ("mid-ranks"). R's `rank(..., ties.method="average")`, scipy's
    `rankdata(method="average")`. Used by Spearman's rho.

 2. KENDALL'S TAU-B's TIE CORRECTION. With
        tot   = n(n-1)/2            (all pairs)
        xtie  = pairs tied in x
        ytie  = pairs tied in y
        ntie  = pairs tied in BOTH x and y
        dis   = discordant pairs
    the numerator is
        con_minus_dis = tot - xtie - ytie + ntie - 2*dis
    and the denominator is
        sqrt((tot - xtie) * (tot - ytie)).

    **THE JOINT TIES COUNT IN *BOTH* xtie AND ytie.** That is the single
    most common way to get tau-b wrong, and it is invisible on a fixture
    with no ties. Section 3 pins it with a fixture that HAS joint ties.

 3. THE NORMAL-SCORES PLOTTING POSITION. z_i = Phi^-1((r_i - 0.5)/n) with
    r_i the mid-rank. The (r - 0.5)/n position is the van der Waerden /
    Hazen position used by the copula transform. Section 4 pins it with an
    exact ROUND-TRIP: Phi(z_i) must return (r_i - 0.5)/n.

WHAT IS ASSERTED ABOUT "BLOMQVIST'S BETA" -- AND WHY IT IS NOT HERE
==================================================================

The queue bullet named "copula utilities" and rank correlation. It did NOT
name Blomqvist's beta, and there is a good reason to leave it out: THE
NAME IS AMBIGUOUS IN THE LITERATURE, and the two candidate definitions are
NOT the same statistic.

  * The `wdm` R package (Thomas Nagler, MIT), which the maintained
    `easystats/correlation` package DELEGATES to for its "blomqvist" method
    (`wdm::wdm(x, y, method = "blomqvist")`), implements it as a
    MEDIAN-QUADRANT statistic:
        beta = 2 * (mass in the lower-left + upper-right quadrants
                    about the medians) / (total mass) - 1.
    See wdm/inst/include/wdm/bbeta.hpp, read from CRAN wdm_0.2.6.
  * Other sources describe a normal-scores ("medial") correlation.

These are different functions of the data. Shipping one of them under the
name "Blomqvist" would ship a definition this file cannot justify, and the
repo's own rule is that an identifier is claimed only when it is pinned to a
scraped source. **The normal-scores correlation IS shipped -- under the name
`gaussian_copula_correlation`, which says exactly what it computes** -- and
the median-quadrant variant is not shipped at all, because nothing in the
repo needs it and its definition is not what the name is used for here.
"""

import math
import sys

import numpy as np
from scipy import integrate
from scipy.stats import kendalltau, norm, rankdata, spearmanr

FAILURES = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(msg)
        print("  [FAIL] " + msg)
    return bool(cond)


def close(a, b, tol, msg):
    a = float(a)
    b = float(b)
    ok = abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    CHECKS[0] += 1
    if not ok:
        FAILURES.append("%s   (got %.17g, want %.17g, diff %.3e)" % (msg, a, b, a - b))
        print("  [FAIL] %s   (got %.17g, want %.17g, diff %.3e)" % (msg, a, b, a - b))
    return ok


# =============================================================================
# The reference implementations, written FROM THE DEFINITIONS.
# =============================================================================


def ref_mid_ranks(x):
    """Average (mid-) ranks: a tied group gets the mean of the ranks it spans.
    1-based."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # the tied group spans 0-based positions i..j, so 1-based ranks
        # i+1 .. j+1; their average is (i+1 + j+1)/2
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks


def ref_pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float(xm @ xm) * float(ym @ ym))
    if denom == 0.0:
        return float("nan")
    return float(xm @ ym) / denom


def ref_spearman_rho(x, y):
    """Pearson's r of the mid-ranks."""
    return ref_pearson_r(ref_mid_ranks(x), ref_mid_ranks(y))


def ref_kendall_tau_b(x, y):
    """Kendall's tau-b, from the definition.

    tot   = all pairs
    xtie  = pairs tied in x      (INCLUDING the joint ties)
    ytie  = pairs tied in y      (INCLUDING the joint ties)
    ntie  = pairs tied in both
    dis   = discordant pairs

    con_minus_dis = tot - xtie - ytie + ntie - 2*dis
    tau_b         = con_minus_dis / sqrt((tot-xtie)*(tot-ytie))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    tot = n * (n - 1) // 2
    con = dis = xtie = ytie = ntie = 0
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
    denom = (tot - xtie) * (tot - ytie)
    if denom <= 0:
        return float("nan")
    return con_minus_dis / math.sqrt(denom)


def ref_normal_scores(x):
    """z_i = Phi^-1((r_i - 0.5)/n), r_i the mid-rank."""
    r = ref_mid_ranks(x)
    n = r.size
    return norm.ppf((r - 0.5) / n)


def ref_gaussian_copula_correlation(x, y):
    """Pearson's r of the normal scores: the marginal-free correlation."""
    return ref_pearson_r(ref_normal_scores(x), ref_normal_scores(y))


# --- the four exact population correspondences -------------------------------


def pop_tau_from_rho(rho):
    return 2.0 / math.pi * math.asin(rho)


def pop_rho_from_tau(tau):
    return math.sin(math.pi * tau / 2.0)


def pop_spearman_from_rho(rho):
    return 6.0 / math.pi * math.asin(rho / 2.0)


def pop_rho_from_spearman(rho_s):
    return 2.0 * math.sin(math.pi * rho_s / 6.0)


def quad_tau(rho):
    """E[sign(U)sign(V)] for jointly standard normal (U,V) with corr rho,
    by NUMERICAL QUADRATURE (no closed form anywhere below).

    The concordance probability is P(U>0, V>0), and conditioning on U=u
    gives the ONE-DIMENSIONAL integral

        P(U>0, V>0) = int_0^inf  phi(u) * Phi(rho*u/sqrt(1-rho^2)) du

    and E[sign(U)sign(V)] = 4*P(U>0,V>0) - 1 by symmetry. The integrand is
    smooth and bounded, which is why the 1-D form is used instead of a 2-D
    integral of the (singular at |rho| -> 1) bivariate density.

    rho = +-1 IS DEGENERATE: there the bivariate normal is V = +-U exactly,
    so P(U>0,V>0) is 1/2 or 0 and E[sign sign] is +-1. Those endpoints are
    handled by the degenerate value, not by quadrature.
    """
    if rho >= 1.0:
        return 1.0
    if rho <= -1.0:
        return -1.0
    s = math.sqrt(1.0 - rho * rho)

    def integrand(u):
        return norm.pdf(u) * norm.cdf(rho * u / s)

    val = integrate.quad(integrand, 0.0, np.inf, limit=400)[0]
    return 4.0 * val - 1.0


def _gh_expectation(g, n=200):
    """E_{e~N(0,1)}[ g(e) ] by Gauss-Hermite quadrature.

    numpy's hermegauss satisfies
        int f(t) * exp(-t^2/2) dt  ~=  sum_i w_i f(x_i)
    and the standard normal density is exp(-t^2/2)/sqrt(2*pi), so
        E[g(e)] = (1/sqrt(2*pi)) * sum_i w_i g(x_i).
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    total = 0.0
    for i in range(nodes.size):
        total += weights[i] * g(nodes[i])
    return total / math.sqrt(2.0 * math.pi)


def quad_spearman(rho):
    """12 * E[Phi(X) Phi(Y)] - 3, for (X,Y) jointly standard normal with
    correlation rho, by GAUSS-HERMITE QUADRATURE.

    The population Spearman coefficient of a continuous bivariate law with
    CDF F and density f is

        rho_S = 12 * E[ F(X) F(Y) ] - 3.

    WRITING THE INTEGRAL IN 1-D. With X = e1 and Y = rho*e1 + s*e2
    (s = sqrt(1-rho^2), e1 and e2 independent standard normals),

        E_{e2}[ Phi(rho*e1 + s*e2) ] = Phi(rho*e1 / sqrt(1+s^2))
                                    = Phi(rho*e1 / sqrt(2-rho^2)),

    because a + b*e2 ~ N(a, b^2) and hence
    E[Phi(a + b*e2)] = P(Z - b*e2 <= a) = Phi(a/sqrt(1+b^2)). So the
    whole expectation collapses to

        E[Phi(X)Phi(Y)] = E_{e1}[ Phi(e1) * Phi(rho*e1/sqrt(2-rho^2)) ],

    a ONE-DIMENSIONAL integral of a smooth bounded integrand. The 2-D
    Gauss-Hermite form is computed as well and the two are asserted to
    agree, so a normalisation slip in either cannot pass unnoticed.

    SELF-CHECK: at rho = 0 the marginals are INDEPENDENT, so
    E[Phi(X)Phi(Y)] = E[Phi(X)] * E[Phi(Y)] = 0.25 exactly, and the
    population Spearman coefficient must be 12*0.25 - 3 = 0. The routine
    asserts that, so a quadrature that silently returned garbage could not
    be mistaken for a verification.
    """
    s = math.sqrt(1.0 - rho * rho)
    denom = math.sqrt(2.0 - rho * rho)

    def g(e1):
        return norm.cdf(e1) * norm.cdf(rho * e1 / denom)

    e_1d = _gh_expectation(g)

    # the 2-D form, for the cross-check
    nodes, weights = np.polynomial.hermite_e.hermegauss(120)
    total = 0.0
    for i in range(nodes.size):
        e1 = nodes[i]
        for j in range(nodes.size):
            e2 = nodes[j]
            total += weights[i] * weights[j] * norm.cdf(e1) * norm.cdf(rho * e1 + s * e2)
    e_2d = total / (2.0 * math.pi)

    if abs(e_1d - e_2d) > 1e-10:
        raise AssertionError(
            "quad_spearman: the 1-D and 2-D Gauss-Hermite forms disagree "
            "(%.17g vs %.17g) -- the quadrature is wrong, not the formula"
            % (e_1d, e_2d))

    return 12.0 * e_1d - 3.0


# =============================================================================
# SECTION 1 -- mid-ranks
# =============================================================================

def section1():
    print("\n=== Section 1: mid-ranks (average ranks for ties) ===")

    # 1.1 a hand-computed fixture
    x = [10.0, 20.0, 20.0, 30.0]
    want = [1.0, 2.5, 2.5, 4.0]
    got = ref_mid_ranks(x).tolist()
    check(got == want, "1.1 mid-ranks hand fixture %s != %s" % (got, want))

    # 1.2 no ties -> 1..n
    got = ref_mid_ranks([5.0, 1.0, 3.0, 2.0, 4.0]).tolist()
    check(sorted(got) == [1.0, 2.0, 3.0, 4.0, 5.0],
          "1.2 mid-ranks of a tie-free sample are 1..n")

    # 1.3 cross-check vs scipy's maintained rankdata
    rng = np.random.default_rng(20261110)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 12))
        a = rng.integers(0, 4, size=n).astype(float)
        worst = max(worst, float(np.max(np.abs(ref_mid_ranks(a) - rankdata(a, method="average")))))
    close(worst, 0.0, 1e-12, "1.3 mid-ranks match scipy rankdata('average') on 200 tie-heavy cases")

    # 1.4 the "min rank" convention is DIFFERENT and must be rejected
    a = np.array([10.0, 20.0, 20.0, 30.0])
    check(not np.allclose(ref_mid_ranks(a), rankdata(a, method="min")),
          "1.4 mid-ranks differ from the 'min' tie convention (the foil works)")

    # 1.5 a large tie group gets the average of the whole span
    a = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    got = ref_mid_ranks(a).tolist()
    check(got == [2.5, 2.5, 2.5, 2.5, 5.0],
          "1.5 a 4-way tie spans 1..4 and gets 2.5 (got %s)" % got)


# =============================================================================
# SECTION 2 -- Spearman's rho
# =============================================================================

def section2():
    print("\n=== Section 2: Spearman's rho ===")

    rng = np.random.default_rng(20261111)

    # 2.1 cross-check vs scipy (maintained reference implementation)
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 15))
        a = rng.integers(0, 5, size=n).astype(float)
        b = rng.integers(0, 5, size=n).astype(float)
        want = spearmanr(a, b).statistic
        got = ref_spearman_rho(a, b)
        if not math.isnan(want) and not math.isnan(got):
            worst = max(worst, abs(got - want))
    close(worst, 0.0, 1e-12, "2.1 Spearman matches scipy spearmanr on 300 tie-heavy cases")

    # 2.2 the CLASSIC d^2 SHORTCUT IS WRONG WITH TIES.
    #     1 - 6*sum(d^2)/(n*(n^2-1)) is only valid when there are NO ties.
    def d2_shortcut(x, y):
        rx = ref_mid_ranks(x)
        ry = ref_mid_ranks(y)
        n = len(x)
        s = float(np.sum((rx - ry) ** 2))
        return 1.0 - 6.0 * s / (n * (n * n - 1.0))

    tie_free_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tie_free_y = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
    close(d2_shortcut(tie_free_x, tie_free_y), ref_spearman_rho(tie_free_x, tie_free_y),
          1e-12, "2.2a the d^2 shortcut AGREES with the mid-rank definition when there are no ties")

    tied_x = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    tied_y = np.array([2.0, 1.0, 4.0, 4.0, 6.0, 5.0])
    sc = d2_shortcut(tied_x, tied_y)
    right = ref_spearman_rho(tied_x, tied_y)
    check(abs(sc - right) > 1e-6,
          "2.2b the d^2 shortcut DIFFERS from the mid-rank definition when there ARE ties "
          "(shortcut %.12f vs right %.12f)" % (sc, right))

    # 2.3 MONOTONE INVARIANCE -- Spearman depends on the ORDER, not the values
    base_x = rng.normal(size=200)
    base_y = rng.normal(size=200)
    r0 = ref_spearman_rho(base_x, base_y)
    for f in (lambda v: np.exp(v), lambda v: v ** 3, lambda v: 2.0 * v + 7.0):
        close(ref_spearman_rho(f(base_x), f(base_y)), r0, 1e-12,
              "2.3 Spearman is invariant under a strictly increasing transform of either marginal")

    # 2.4 symmetry
    close(ref_spearman_rho(base_x, base_y), ref_spearman_rho(base_y, base_x), 1e-15,
          "2.4 Spearman is symmetric")

    # 2.5 perfect monotone -> exactly 1; reversed -> exactly -1
    v = rng.normal(size=50)
    close(ref_spearman_rho(v, v), 1.0, 1e-15, "2.5a Spearman(x, x) == 1")
    close(ref_spearman_rho(v, -v), -1.0, 1e-15, "2.5b Spearman(x, -x) == -1")

    # 2.6 range
    ok = True
    for _ in range(200):
        n = int(rng.integers(2, 15))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        r = ref_spearman_rho(a, b)
        if not (-1.0 - 1e-12 <= r <= 1.0 + 1e-12):
            ok = False
    check(ok, "2.6 |Spearman| <= 1 on 200 random cases")


# =============================================================================
# SECTION 3 -- Kendall's tau-b
# =============================================================================

def section3():
    print("\n=== Section 3: Kendall's tau-b (the tie correction) ===")

    rng = np.random.default_rng(20261112)

    # 3.1 cross-check vs scipy (maintained reference implementation)
    worst = 0.0
    for _ in range(2000):
        n = int(rng.integers(2, 11))
        a = rng.integers(0, 4, size=n).astype(float)
        b = rng.integers(0, 4, size=n).astype(float)
        want = kendalltau(a, b, variant="b").statistic
        got = ref_kendall_tau_b(a, b)
        if not math.isnan(want) and not math.isnan(got):
            worst = max(worst, abs(got - want))
    close(worst, 0.0, 1e-12, "3.1 tau-b matches scipy kendalltau(variant='b') on 2000 tie-heavy cases")

    # 3.2 THE JOINT-TIE CONVENTION. Joint ties must count in BOTH xtie AND
    #     ytie. The foil counts them in NEITHER.
    def tau_b_joint_ties_counted_once(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        tot = n * (n - 1) // 2
        con = dis = xtie = ytie = ntie = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx == 0.0 and dy == 0.0:
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
        denom = (tot - xtie) * (tot - ytie)
        if denom <= 0:
            return float("nan")
        return con_minus_dis / math.sqrt(denom)

    jt_x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    jt_y = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    right = ref_kendall_tau_b(jt_x, jt_y)
    foil = tau_b_joint_ties_counted_once(jt_x, jt_y)
    check(abs(right - foil) > 1e-9,
          "3.2 the joint-tie foil DIFFERS from the correct tau-b (right %.12f, foil %.12f)"
          % (right, foil))

    # 3.3 a NO-TIE fixture: tau-b reduces to (C-D)/tot
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    n = 5
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if (dx > 0) == (dy > 0):
                con += 1
            else:
                dis += 1
    close(ref_kendall_tau_b(x, y), (con - dis) / (n * (n - 1) // 2), 1e-12,
          "3.3 tau-b reduces to (C-D)/tot when there are no ties")

    # 3.4 monotone invariance, symmetry, extremes
    base_x = rng.normal(size=200)
    base_y = rng.normal(size=200)
    t0 = ref_kendall_tau_b(base_x, base_y)
    for f in (lambda v: np.exp(v), lambda v: v ** 3, lambda v: 2.0 * v + 7.0):
        close(ref_kendall_tau_b(f(base_x), f(base_y)), t0, 1e-12,
              "3.4a tau-b is invariant under a strictly increasing transform")
    close(ref_kendall_tau_b(base_x, base_y), ref_kendall_tau_b(base_y, base_x), 1e-15,
          "3.4b tau-b is symmetric")
    v = rng.normal(size=60)
    close(ref_kendall_tau_b(v, v), 1.0, 1e-15, "3.4c tau-b(x, x) == 1")
    close(ref_kendall_tau_b(v, -v), -1.0, 1e-15, "3.4d tau-b(x, -x) == -1")

    # 3.5 range
    ok = True
    for _ in range(300):
        n = int(rng.integers(2, 15))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        t = ref_kendall_tau_b(a, b)
        if not (-1.0 - 1e-12 <= t <= 1.0 + 1e-12):
            ok = False
    check(ok, "3.5 |tau-b| <= 1 on 300 random cases")

    # 3.6 tau-b of a monotone increasing pair is exactly 1 even WITH ties
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    y = np.array([10.0, 10.0, 20.0, 30.0, 30.0, 40.0])
    close(ref_kendall_tau_b(x, y), 1.0, 1e-15, "3.6 tau-b of a monotone pair with ties is exactly 1")


# =============================================================================
# SECTION 4 -- normal scores (the copula marginal transform)
# =============================================================================

def section4():
    print("\n=== Section 4: normal scores ===")

    rng = np.random.default_rng(20261113)

    # 4.1 the plotting position ROUND-TRIP: Phi(z_i) == (r_i - 0.5)/n
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 20))
        a = rng.integers(0, 5, size=n).astype(float)
        z = ref_normal_scores(a)
        r = ref_mid_ranks(a)
        p = (r - 0.5) / n
        worst = max(worst, float(np.max(np.abs(norm.cdf(z) - p))))
    close(worst, 0.0, 1e-12, "4.1 Phi(normal_scores) reproduces the (r-0.5)/n plotting position")

    # 4.2 the position is the mid-EMI one, NOT the (r-0.375)/(n+0.25) Blomqvist
    #     position and NOT r/(n+1)
    n = 10
    a = np.arange(1.0, n + 1.0)
    r = ref_mid_ranks(a)
    hazen = (r - 0.5) / n
    blom = (r - 0.375) / (n + 0.25)
    weibull = r / (n + 1.0)
    check(not np.allclose(hazen, blom), "4.2a the Hazen position differs from the Blomqvist position")
    check(not np.allclose(hazen, weibull), "4.2b the Hazen position differs from the Weibull position")
    check(abs(hazen[0] - 0.05) < 1e-15 and abs(hazen[-1] - 0.95) < 1e-15,
          "4.2c the Hazen position runs from 0.5/n to 1-0.5/n")

    # 4.3 the scores are ORDER-PRESERVING and their ranks are the mid-ranks
    a = rng.normal(size=100)
    z = ref_normal_scores(a)
    order_a = np.argsort(a)
    order_z = np.argsort(z)
    check(np.array_equal(order_a, order_z), "4.3 the normal scores preserve the order of the data")

    # 4.4 the scores are symmetric about 0 for a symmetric sample
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    z = ref_normal_scores(a)
    close(float(z[0] + z[4]), 0.0, 1e-12, "4.4 the normal scores of a symmetric sample are symmetric")

    # 4.5 the transform is MONOTONE-INVARIANT: the scores of f(x) are the
    #     scores of x for any strictly increasing f
    base = rng.normal(size=100)
    z0 = ref_normal_scores(base)
    for f in (lambda v: np.exp(v), lambda v: v ** 3, lambda v: 2.0 * v + 7.0):
        close(float(np.max(np.abs(ref_normal_scores(f(base)) - z0))), 0.0, 1e-12,
              "4.5 the normal scores are invariant under a strictly increasing transform")

    # 4.6 mean 0 and variance 1 for a LARGE uniform sample (asymptotic)
    a = np.random.default_rng(1).uniform(size=200000)
    z = ref_normal_scores(a)
    close(float(np.mean(z)), 0.0, 0.02, "4.6a the normal scores of a large uniform sample have mean ~0")
    close(float(np.var(z)), 1.0, 0.02, "4.6b the normal scores of a large uniform sample have variance ~1")


# =============================================================================
# SECTION 5 -- the Gaussian copula correlation (marginal-freeness)
# =============================================================================

def section5():
    print("\n=== Section 5: the Gaussian copula correlation ===")

    rng = np.random.default_rng(20261114)

    # 5.1 MARGINAL-FREENESS -- the defining property of a copula statistic.
    base_x = rng.normal(size=300)
    base_y = rng.normal(size=300)
    c0 = ref_gaussian_copula_correlation(base_x, base_y)
    for f in (lambda v: np.exp(v), lambda v: v ** 3, lambda v: 2.0 * v + 7.0,
              lambda v: 1.0 / (1.0 + np.exp(-v))):
        close(ref_gaussian_copula_correlation(f(base_x), f(base_y)), c0, 1e-12,
              "5.1 the copula correlation is invariant under a strictly increasing transform "
              "of either marginal")

    # 5.2 and it is NOT invariant for the RAW Pearson correlation -- the foil
    #     that proves the property is non-trivial
    p0 = ref_pearson_r(base_x, base_y)
    pf = ref_pearson_r(np.exp(base_x), base_y)
    check(abs(p0 - pf) > 1e-6,
          "5.2 the RAW Pearson correlation is NOT marginal-free (so 5.1 is a real property) "
          "(pearson %.12f vs pearson-of-exp %.12f)" % (p0, pf))

    # 5.3 for a bivariate GAUSSIAN sample the copula correlation converges to rho
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    data = rng.multivariate_normal([0.0, 0.0], cov, size=4000)
    close(ref_gaussian_copula_correlation(data[:, 0], data[:, 1]), rho, 0.05,
          "5.3 the copula correlation of a large bivariate normal sample recovers rho")

    # 5.4 and so do the rank statistics, at their population values
    close(ref_spearman_rho(data[:, 0], data[:, 1]), pop_spearman_from_rho(rho), 0.02,
          "5.4a the sample Spearman converges to (6/pi)asin(rho/2)")
    close(ref_kendall_tau_b(data[:, 0], data[:, 1]), pop_tau_from_rho(rho), 0.02,
          "5.4b the sample tau-b converges to (2/pi)asin(rho)")

    # 5.5 the copula correlation is symmetric and in [-1, 1]
    close(ref_gaussian_copula_correlation(base_x, base_y),
          ref_gaussian_copula_correlation(base_y, base_x), 1e-15, "5.5a symmetry")
    ok = True
    for _ in range(200):
        n = int(rng.integers(2, 15))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        c = ref_gaussian_copula_correlation(a, b)
        if not (-1.0 - 1e-12 <= c <= 1.0 + 1e-12):
            ok = False
    check(ok, "5.5b |copula correlation| <= 1 on 200 random cases")


# =============================================================================
# SECTION 6 -- the four exact population correspondences
# =============================================================================

def section6():
    print("\n=== Section 6: the Gaussian-copula population correspondences ===")

    rhos = [0.0, 0.25, -0.5, 0.75, 0.9, -0.95, 1.0, -1.0]

    # 6.0 THE QUADRATURE VALIDATES ITSELF BEFORE IT IS USED AS A REFERENCE.
    #      At rho = 0 the marginals are independent, so the population
    #      Spearman coefficient is exactly 0. A quadrature that silently
    #      returned garbage could not pass this.
    close(quad_spearman(0.0), 0.0, 1e-9,
          "6.0 the Gauss-Hermite quadrature self-check at rho=0 (must be exactly 0)")
    close(quad_tau(0.0), 0.0, 1e-9,
          "6.0b the 1-D concordance quadrature self-check at rho=0 (must be exactly 0)")

    # 6.1 tau = (2/pi)asin(rho)  -- verified against 2-D QUADRATURE
    for rho in rhos:
        close(quad_tau(rho), pop_tau_from_rho(rho), 2e-6,
              "6.1 tau=(2/pi)asin(rho) at rho=%+.2f matches the quadrature" % rho)

    # 6.2 rho_S = (6/pi)asin(rho/2) -- verified against 2-D QUADRATURE
    for rho in rhos:
        close(quad_spearman(rho), pop_spearman_from_rho(rho), 2e-6,
              "6.2 rho_S=(6/pi)asin(rho/2) at rho=%+.2f matches the quadrature" % rho)

    # 6.3 the inverses are exact inverses
    for rho in rhos:
        close(pop_rho_from_tau(pop_tau_from_rho(rho)), rho, 1e-14,
              "6.3a rho_from_tau(tau_from_rho(rho)) == rho at rho=%+.2f" % rho)
        close(pop_rho_from_spearman(pop_spearman_from_rho(rho)), rho, 1e-14,
              "6.3b rho_from_spearman(spearman_from_rho(rho)) == rho at rho=%+.2f" % rho)

    # 6.4 the endpoints
    close(pop_tau_from_rho(1.0), 1.0, 1e-15, "6.4a tau(rho=1) == 1")
    close(pop_tau_from_rho(-1.0), -1.0, 1e-15, "6.4b tau(rho=-1) == -1")
    close(pop_spearman_from_rho(1.0), 1.0, 1e-15, "6.4c rho_S(rho=1) == 1")
    close(pop_spearman_from_rho(-1.0), -1.0, 1e-15, "6.4d rho_S(rho=-1) == -1")
    close(pop_tau_from_rho(0.0), 0.0, 1e-15, "6.4e tau(rho=0) == 0")

    # 6.5 monotonicity
    prev = -2.0
    ok = True
    for i in range(0, 101):
        rho = -1.0 + 2.0 * i / 100.0
        t = pop_tau_from_rho(rho)
        if t < prev - 1e-15:
            ok = False
        prev = t
    check(ok, "6.5 tau(rho) is non-decreasing on [-1, 1]")

    # 6.6 the two population values are ORDERED: for 0 < rho < 1,
    #     tau < rho_S < 1 (rho_S is always the LARGER of the two)
    ok = True
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if not (pop_tau_from_rho(rho) < pop_spearman_from_rho(rho)):
            ok = False
    check(ok, "6.6 for 0 < rho < 1 the population tau is strictly below the population rho_S")

    # 6.7 the WRONG closed forms must be distinguishable
    #     (a) tau = (1/pi)asin(rho)      -- off by a factor 2
    #     (b) rho_S = (6/pi)asin(rho)    -- rho/2 dropped
    for rho in [0.3, 0.6, 0.9]:
        check(abs(pop_tau_from_rho(rho) - (1.0 / math.pi) * math.asin(rho)) > 1e-6,
              "6.7a the factor-2 foil for tau is distinguishable at rho=%+.2f" % rho)
        check(abs(pop_spearman_from_rho(rho) - (6.0 / math.pi) * math.asin(rho)) > 1e-6,
              "6.7b the rho/2-dropped foil for rho_S is distinguishable at rho=%+.2f" % rho)


# =============================================================================
# SECTION 7 -- the pinned fixture the C++ suite asserts against
# =============================================================================

FIXTURES = {}


def section7():
    print("\n=== Section 7: the pinned fixtures ===")

    # A tie-heavy fixture with joint ties, so every convention is exercised.
    fx = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0])
    fy = np.array([2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 4.0])

    FIXTURES["fx"] = fx.tolist()
    FIXTURES["fy"] = fy.tolist()
    FIXTURES["mid_x"] = ref_mid_ranks(fx).tolist()
    FIXTURES["mid_y"] = ref_mid_ranks(fy).tolist()
    FIXTURES["spearman"] = ref_spearman_rho(fx, fy)
    FIXTURES["tau_b"] = ref_kendall_tau_b(fx, fy)
    FIXTURES["copula_corr"] = ref_gaussian_copula_correlation(fx, fy)
    FIXTURES["pearson"] = ref_pearson_r(fx, fy)
    FIXTURES["normal_scores_x"] = ref_normal_scores(fx).tolist()

    print("  fx             = %s" % FIXTURES["fx"])
    print("  fy             = %s" % FIXTURES["fy"])
    print("  mid_ranks(x)   = %s" % FIXTURES["mid_x"])
    print("  mid_ranks(y)   = %s" % FIXTURES["mid_y"])
    print("  pearson_r      = %.17g" % FIXTURES["pearson"])
    print("  spearman_rho   = %.17g" % FIXTURES["spearman"])
    print("  kendall_tau_b  = %.17g" % FIXTURES["tau_b"])
    print("  copula_corr    = %.17g" % FIXTURES["copula_corr"])
    print("  normal_scores(x) = %s" % ["%.17g" % v for v in FIXTURES["normal_scores_x"]])

    # sanity: the fixture really does have joint ties
    joint = 0
    for i in range(fx.size):
        for j in range(i + 1, fx.size):
            if fx[i] == fx[j] and fy[i] == fy[j]:
                joint += 1
    check(joint > 0, "7.1 the pinned fixture contains JOINT ties (%d of them)" % joint)

    # and the tie conventions are all distinguishable ON IT
    sc = 1.0 - 6.0 * float(np.sum((ref_mid_ranks(fx) - ref_mid_ranks(fy)) ** 2)) / (
        fx.size * (fx.size * fx.size - 1.0))
    check(abs(sc - FIXTURES["spearman"]) > 1e-6,
          "7.2 the d^2 shortcut differs from the pinned Spearman on the pinned fixture")


def emit_fixture_cpp():
    """Emit the fixture as C++ literals so the pinned numbers transfer
    EXACTLY. (The repo has been burned by hand-transcribed numbers before:
    see tests/_dump_fixture.py.)"""
    print("\n--- C++ fixture ---")
    print("    // Pinned by tests/rank_correlation_oracle.py (float64 numpy).")
    print("    const std::vector<double> fx = {%s};" %
          ", ".join("%.17g" % v for v in FIXTURES["fx"]))
    print("    const std::vector<double> fy = {%s};" %
          ", ".join("%.17g" % v for v in FIXTURES["fy"]))
    print("    // mid_ranks(x) = {%s}" % ", ".join("%.17g" % v for v in FIXTURES["mid_x"]))
    print("    // mid_ranks(y) = {%s}" % ", ".join("%.17g" % v for v in FIXTURES["mid_y"]))
    print("    // pearson_r      = %.17g" % FIXTURES["pearson"])
    print("    // spearman_rho   = %.17g" % FIXTURES["spearman"])
    print("    // kendall_tau_b  = %.17g" % FIXTURES["tau_b"])
    print("    // copula_corr    = %.17g" % FIXTURES["copula_corr"])
    print("    // normal_scores(x) = {%s}" %
          ", ".join("%.17g" % v for v in FIXTURES["normal_scores_x"]))


def main():
    print("tests/rank_correlation_oracle.py -- the float64 numpy SPECIFICATION")
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section7()
    emit_fixture_cpp()
    print("\n=== SUMMARY: %d passed, %d failed ===" % (CHECKS[0] - len(FAILURES), len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
