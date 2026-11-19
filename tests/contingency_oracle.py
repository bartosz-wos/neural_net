#!/usr/bin/env python3
"""
tests/contingency_oracle.py -- the float64 numpy specification of the
CATEGORICAL AGREEMENT and CONTINGENCY-TABLE INFERENCE utilities:
Cohen's kappa (unweighted / linear / quadratic disagreement weights), the
Matthews correlation coefficient, the Pearson chi-square test of
independence, the G-test (log-likelihood ratio), McNemar's test for paired
binary outcomes, and Fisher's exact test for a 2x2 table -- plus the
general chi-square survival function that all three table tests need.

RUN THIS BEFORE WRITING ANY C++.

    /usr/bin/python3 tests/contingency_oracle.py
    /usr/bin/python3 tests/contingency_oracle.py --emit-cpp

**THE INTERPRETER IS /usr/bin/python3, NOT `python3`.** That is the
interpreter with numpy and scipy on this host, and it is what the sibling
oracles in this repo hardcode (`tests/survival_oracle.py`,
`tests/dtw_mutate_oracle.py`, `tests/block_cached_decode_mutate_oracle.py`,
`tests/edit_distance_mutate_oracle.py`). A bare `python3` resolves to
whatever is first on PATH, which inside an agent's own virtual environment
is NOT the one with numpy -- and the oracle would fail to import, which
looks like a broken oracle rather than a wrong interpreter.

Its discriminating power is MEASURED, not narrated: the companion harness
tests/contingency_mutate_oracle.py breaks the model code below in specific
ways and requires that this oracle FAIL each time. An oracle that cannot
fail certifies nothing.

==========================================================================
THE GAP (measured 2026-11-19, not assumed)
==========================================================================

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

    ZERO IMPLEMENTATION.

The repo ships ~250 layers and 60+ utility files and NO categorical
agreement or categorical association machinery at all.

It is NOT redundant with any of them:
  * utils/metrics.h -- accuracy, F1, ROC-AUC, ECE, NDCG. Every one is a
    SCORE-vs-LABEL statistic. None of them compares two LABELINGS of the
    same items (kappa, McNemar), and none corrects for the agreement two
    raters would reach BY CHANCE (kappa).
  * utils/rank_correlation.h -- Pearson / Spearman / Kendall on CONTINUOUS
    quantities. There is no categorical analogue of association anywhere.
  * utils/survival.h -- the log-rank test's p-value is the chi-square-1
    survival function erfc(sqrt(x/2)), a closed form that exists ONLY for
    one degree of freedom. A 2x3 table has df = 2 and does NOT fit, so the
    repo cannot currently produce the p-value for any table bigger than
    2x2.
  * utils/conformal.h -- the coverage of a SET. No notion of two raters.

==========================================================================
THE INDEPENDENT REFERENCES
==========================================================================

    scipy 1.17.1     -- chi2.sf, chi2_contingency, power_divergence,
                        fisher_exact, binomtest
    scikit-learn 1.9 -- cohen_kappa_score, matthews_corrcoef
    R 4.5.3          -- stats::mcnemar.test, stats::chisq.test
                        (via Rscript; OPTIONAL -- skipped if absent)

Every convention below was MEASURED against those, not recalled, and the
two places where the obvious textbook answer is WRONG are called out.

==========================================================================
THE TWO PLACES THE TEXTBOOK IS WRONG
==========================================================================

1. YATES' CORRECTION IS CLAMPED TO THE CELL'S OWN DEVIATION.

   scipy.stats.chi2_contingency:
       diff = expected - observed
       direction = np.sign(diff)
       magnitude = np.minimum(0.5, np.abs(diff))     <-- the CLAMP
       observed += magnitude * direction
   R stats::chisq.test:
       YATES <- min(0.5, abs(x - E))                  <-- elementwise clamp
       X2 <- sum((abs(x - E) - YATES)^2/E)

   So the statistic is
       sum_ij ( |O_ij - E_ij| - min(0.5, |O_ij - E_ij|) )^2 / E_ij
   and NOT sum_ij ( |O_ij - E_ij| - 0.5 )^2 / E_ij.

   The discriminating fixture is the 2x2 [[1,1],[1,2]], whose cells all
   have |O - E| = 0.2: the unclamped form gives 0.3125, the clamped form
   gives 0.0, and scipy AND R both give 0.0. Measured over every 2x2
   table with entries in 0..8 and no empty row or column (6272 tables),
   scipy and R agree to 1.8e-15.

2. McNEMAR'S CONTINUITY CORRECTION IS SKIPPED ON A SYMMETRIC TABLE.

   R stats::mcnemar.test:
       if (correct && (r == 2) && any(x - t(x) != 0)) y <- abs(x - t(x)) - 1
       else                                         y <- x - t(x)
       x <- x + t(x)
       STATISTIC <- sum(y[upper.tri(x)]^2/x[upper.tri(x)])

   For a 2x2 table [[n00,n01],[n10,n11]], x - t(x) = [[0,n01-n10],
   [n10-n01,0]], so the correction is applied IFF n01 != n10. When
   n01 == n10 the statistic is 0, NOT (0-1)^2/(2*n01) = 1/(2*n01). And
   the numerator is |n01 - n10| - 1 -- the ABSOLUTE value -- so that
   n01 < n10 gives (|n01-n10|-1)^2 and not (n01-n10-1)^2.

   Measured on [[30,5],[5,25]] (n01 == n10 == 5): R reports statistic
   0.00000000 and p-value 1.0.
   Measured on [[0,3],[5,0]] (n01 = 3 < n10 = 5): R reports 0.125, which
   is (|3-5|-1)^2/8 -- the signed form would give 0.5625.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

# ===========================================================================
# THE MODEL -- the specification. Everything here is computed from the
# DEFINITION. The references are used only to CONFIRM it, never to define
# it. The mutation harness breaks THIS code, not the references.
# ===========================================================================


def _log(x):
    """A C++-LIKE log: log(0) is -inf, not a domain error.

    Python's math.log(0) RAISES ValueError, but C's std::log(0) returns
    -inf, and numpy's np.log(0) returns -inf with a warning. The oracle
    has to behave like the C++ it specifies, or a mutation that removes
    a zero-cell guard would CRASH the oracle instead of producing the
    NaN the C++ produces -- and a crash is not evidence that a check
    works.
    """
    if x == 0.0:
        return float("-inf")
    return math.log(x)


def gser_model(a, x, itmax=1000, eps=3e-16):
    """Series for the regularized LOWER incomplete gamma P(a, x)."""
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(itmax):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gcf_model(a, x, itmax=1000, eps=3e-16, fpmin=1e-300):
    """Continued fraction for the regularized UPPER incomplete gamma Q(a,x)."""
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        dd = d * c
        h *= dd
        if abs(dd - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammq_model(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < a + 1.0:
        return 1.0 - gser_model(a, x)
    return gcf_model(a, x)


def chi_square_survival_model(statistic, degrees_of_freedom):
    """P(X >= statistic) for X ~ chi2(degrees_of_freedom).

    The df == 1 path is the repo's EXISTING closed form, bit for bit:
    utils/survival.cpp computes std::erfc(std::sqrt(statistic / 2.0)).

    The statistic <= 0 guard is NOT cosmetic: the general path evaluates
    log(x) with x = statistic/2, so a statistic of 0 is log(0) = -inf and
    the whole thing is NaN. P(X >= 0) is 1 for any df, and scipy's
    chi2.sf(0, df) is 1, so the guard makes the two agree.
    """
    if statistic <= 0.0:
        return 1.0
    if degrees_of_freedom == 1.0:
        return math.erfc(math.sqrt(statistic / 2.0))
    return gammq_model(degrees_of_freedom / 2.0, statistic / 2.0)


def _validated_labels(labels, num_classes):
    """Validate a label vector against the EXPLICIT class-index space.

    The explicit space is the whole point: an absent intermediate class
    must NOT compress the distances used by the weighted kappa. (The
    unweighted kappa is space-independent; the weighted ones are not.)
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    out = []
    for x in labels:
        xi = float(x)
        if not math.isfinite(xi) or xi != int(xi):
            raise ValueError("class index must be a whole number")
        if int(xi) < 0 or int(xi) >= num_classes:
            raise ValueError("class index out of range")
        out.append(int(xi))
    return out


def contingency_table_model(labels_a, labels_b, num_classes_a, num_classes_b):
    """R x C table of counts. rows index labels_a, cols index labels_b."""
    table = np.zeros((num_classes_a, num_classes_b), dtype=np.float64)
    for x, y in zip(labels_a, labels_b):
        table[int(x), int(y)] += 1.0
    return table


def cohen_kappa_model(labels_a, labels_b, num_classes, weights="unweighted"):
    """Cohen's kappa over the EXPLICIT class-index space.

    Unweighted : w_ij = 1 for i != j, 0 on the diagonal.
    Linear     : w_ij = |i - j|.
    Quadratic  : w_ij = (i - j)^2.
    All three are SYMMETRIC, which is why the orientation of the expected
    matrix does not matter (C4): for symmetric w, sum_ij w_ij E_ij equals
    sum_ij w_ij E_ji by relabelling.
    """
    a = _validated_labels(labels_a, num_classes)
    b = _validated_labels(labels_b, num_classes)
    table = contingency_table_model(a, b, num_classes, num_classes)
    n = table.sum()
    if n == 0:
        raise ValueError("empty")
    observed_agreement = float(np.trace(table) / n)
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    expected = np.outer(row, col) / n
    expected_agreement = float(np.trace(expected) / n)

    k = num_classes
    if weights == "unweighted":
        w = np.ones((k, k))
        np.fill_diagonal(w, 0.0)
    elif weights == "linear":
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    elif weights == "quadratic":
        w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    else:
        raise ValueError("unknown weight scheme")

    numerator = float((w * table).sum())
    denominator = float((w * expected).sum())
    if denominator == 0.0:
        kappa = float("nan")
    else:
        kappa = 1.0 - numerator / denominator
    return {
        "kappa": kappa,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "num_agreements": float(np.trace(table)),
        "num_total": float(n),
    }


def matthews_correlation_model(labels_true, labels_pred, num_classes):
    """The general MULTICLASS covariance form.

    cov_ytyp = c*N - sum_k p_k t_k
    cov_ypyp = N^2 - sum_k p_k^2
    cov_ytyt = N^2 - sum_k t_k^2
    MCC      = cov_ytyp / sqrt(cov_ytyt * cov_ypyp)

    The DEGENERATE case (either marginal concentrated) is 0.0, NOT NaN --
    which is the OPPOSITE of scikit-learn's kappa, whose degenerate case
    is NaN. The two are not interchangeable and both are pinned.
    """
    t = _validated_labels(labels_true, num_classes)
    p = _validated_labels(labels_pred, num_classes)
    table = contingency_table_model(t, p, num_classes, num_classes)
    n = float(table.sum())
    t_sum = table.sum(axis=1)
    p_sum = table.sum(axis=0)
    n_correct = float(np.trace(table))
    cov_ytyp = n_correct * n - float(np.dot(t_sum, p_sum))
    cov_ypyp = n * n - float(np.dot(p_sum, p_sum))
    cov_ytyt = n * n - float(np.dot(t_sum, t_sum))
    denom = cov_ytyt * cov_ypyp
    if denom == 0.0:
        mcc = 0.0
    else:
        mcc = cov_ytyp / math.sqrt(denom)
    return {"mcc": float(mcc), "num_correct": n_correct, "num_total": n}


def _expected_counts(table):
    n = table.sum()
    return np.outer(table.sum(axis=1), table.sum(axis=0)) / n


def chi_square_test_model(table, correct=True):
    """Pearson's chi-square test of independence.

    Yates' continuity correction is applied ONLY to a 2x2 table, and the
    correction magnitude is CLAMPED to each cell's own deviation.
    """
    table = np.asarray(table, dtype=np.float64)
    r, c = table.shape
    if r < 2 or c < 2:
        raise ValueError("need at least a 2x2 table")
    expected = _expected_counts(table)
    if (expected == 0).any():
        raise ValueError("zero expected count")
    n = table.sum()
    dof = (r - 1) * (c - 1)

    adjusted = table
    if correct and r == 2 and c == 2:
        diff = expected - table
        magnitude = np.minimum(0.5, np.abs(diff))
        adjusted = table + magnitude * np.sign(diff)
    statistic = float(((adjusted - expected) ** 2 / expected).sum())
    p_value = chi_square_survival_model(statistic, float(dof))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "min_expected_count": float(expected.min()),
    }


def g_test_model(table):
    """The G-test (log-likelihood ratio) for independence.

    G = 2 * sum_ij O_ij * log(O_ij / E_ij), over the cells with O_ij > 0.
    A zero observed cell contributes EXACTLY 0 -- the naive expression
    0 * log(0/E) is 0 * (-inf) = NaN and destroys the whole statistic.
    """
    table = np.asarray(table, dtype=np.float64)
    r, c = table.shape
    if r < 2 or c < 2:
        raise ValueError("need at least a 2x2 table")
    expected = _expected_counts(table)
    if (expected == 0).any():
        raise ValueError("zero expected count")
    s = 0.0
    for i in range(r):
        for j in range(c):
            o = table[i, j]
            if o > 0:
                s += o * _log(o / expected[i, j])
    statistic = 2.0 * s
    dof = (r - 1) * (c - 1)
    return {
        "statistic": float(statistic),
        "p_value": chi_square_survival_model(statistic, float(dof)),
        "degrees_of_freedom": dof,
    }


def mcnemar_test_model(labels_a, labels_b, correct=True):
    """McNemar's test for paired binary outcomes.

    The statistic follows R stats::mcnemar.test EXACTLY:
      * the continuity correction is applied IFF n01 != n10;
      * the numerator is (|n01 - n10| - 1)^2, with the ABSOLUTE value;
      * when n01 == n10 the statistic is 0, not 1/(2*n01);
      * when n01 + n10 == 0 the statistic is NaN (0/0), not 0.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("length mismatch")
    n01 = 0
    n10 = 0
    for x, y in zip(labels_a, labels_b):
        if int(x) not in (0, 1) or int(y) not in (0, 1):
            raise ValueError("outcomes must be binary")
        if int(x) == 0 and int(y) == 1:
            n01 += 1
        elif int(x) == 1 and int(y) == 0:
            n10 += 1
    n_disc = n01 + n10
    if n_disc == 0:
        statistic = float("nan")
        p_value = float("nan")
    elif correct and n01 != n10:
        statistic = (abs(n01 - n10) - 1.0) ** 2 / n_disc
        p_value = chi_square_survival_model(statistic, 1.0)
    else:
        statistic = float(n01 - n10) ** 2 / n_disc
        p_value = chi_square_survival_model(statistic, 1.0)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "exact_p_value": binom_pvalue_half_model(min(n01, n10), max(n01, n10)),
        "n01": float(n01),
        "n10": float(n10),
        "n_discordant": float(n_disc),
    }


def binom_pvalue_half_model(n_a, n_b):
    """The repo's EXISTING utils/randomized_smoothing.h binom_pvalue_half.

    Composed, not re-implemented: the exact McNemar test is a two-sided
    binomial test at p = 1/2 on the discordant pairs.
    """
    n = n_a + n_b
    if n == 0:
        return 1.0
    lower = 0.0
    for i in range(0, n_a + 1):
        lower += math.comb(n, i) * 0.5 ** n
    upper = 0.0
    for i in range(n_a, n + 1):
        upper += math.comb(n, i) * 0.5 ** n
    return min(1.0, 2.0 * min(lower, upper))


def _hypergeom_pmf(x, M, n, N):
    """P(X = x) for X ~ Hypergeometric(M, n, N).

    C(n, x) * C(M-n, N-x) / C(M, N), evaluated in log space.

    The parameter order matches scipy.stats.hypergeom.pmf(k, M, n, N),
    which is what scipy.stats.fisher_exact uses internally.
    """
    if x < 0 or x > n or x > M or (N - x) > (M - n):
        return 0.0
    log_p = (math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
             + math.lgamma(M - n + 1) - math.lgamma(N - x + 1)
             - math.lgamma(M - n - N + x + 1)
             - math.lgamma(M + 1) + math.lgamma(N + 1) + math.lgamma(M - N + 1))
    return math.exp(log_p)


def fisher_exact_test_model(table):
    """Fisher's exact test for a 2x2 table.

    The two-sided p-value uses the PROBABILITY ORDERING rule: sum the
    hypergeometric pmf over every x whose pmf is at most the observed
    table's pmf (times the 1 + 1e-14 relative tolerance). It is NOT a
    doubled one-sided tail -- measured against scipy on 8300 random
    tables, the doubled-tail rule disagrees on 1806 of 2000.

    The odds ratio is the UNCONDITIONAL (cross-product) one, (a*d)/(b*c),
    which is what scipy returns. R's fisher.test reports the CONDITIONAL
    MLE, which is a different estimator.
    """
    table = np.asarray(table, dtype=np.float64)
    if table.shape != (2, 2):
        raise ValueError("Fisher's exact test needs a 2x2 table")
    a, b = int(table[0, 0]), int(table[0, 1])
    c, d = int(table[1, 0]), int(table[1, 1])
    if (table != np.floor(table)).any():
        raise ValueError("counts must be whole numbers")
    n1 = a + b
    n2 = c + d
    n = a + c
    m = n1 + n2

    if 0 in (n1, n2, n, b + d):
        # scipy returns (nan, 1.0) when a row or column sums to zero.
        return {"p_value": 1.0, "odds_ratio": float("nan")}

    if c > 0 and b > 0:
        odds_ratio = float(a * d / (c * b))
    else:
        odds_ratio = float("inf")

    pexact = _hypergeom_pmf(a, m, n1, n)
    gamma = 1.0 + 1e-14
    lo = max(0, n - (m - n1))
    hi = min(n1, n)
    total = 0.0
    for x in range(lo, hi + 1):
        p = _hypergeom_pmf(x, m, n1, n)
        if p <= pexact * gamma:
            total += p
    return {"p_value": min(total, 1.0), "odds_ratio": odds_ratio}


# ===========================================================================
# INDEPENDENT SCALAR-LOOP REFERENCES
#
# A finite-difference check compares the implementation against ITSELF.
# These references are a SECOND, structurally different witness: plain
# Python loops with no numpy vectorisation, written from the definitions.
# ===========================================================================


def cohen_kappa_reference(labels_a, labels_b, num_classes, weights="unweighted"):
    """Scalar-loop Cohen's kappa, written from the definition."""
    counts = {}
    for x, y in zip(labels_a, labels_b):
        counts[(int(x), int(y))] = counts.get((int(x), int(y)), 0) + 1
    n = float(len(labels_a))
    agree = 0.0
    for i in range(num_classes):
        agree += counts.get((i, i), 0)
    po = agree / n

    row = [0.0] * num_classes
    col = [0.0] * num_classes
    for (i, j), c in counts.items():
        row[i] += c
        col[j] += c
    pe = 0.0
    for i in range(num_classes):
        pe += (row[i] / n) * (col[i] / n)

    num = 0.0
    den = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            e = row[i] * col[j] / n
            if weights == "unweighted":
                w = 0.0 if i == j else 1.0
            elif weights == "linear":
                w = float(abs(i - j))
            else:
                w = float((i - j) ** 2)
            num += w * counts.get((i, j), 0)
            den += w * e
    if den == 0.0:
        return float("nan")
    return 1.0 - num / den


def matthews_reference(labels_true, labels_pred, num_classes):
    """Scalar-loop Matthews correlation coefficient."""
    n = float(len(labels_true))
    t = [0.0] * num_classes
    p = [0.0] * num_classes
    correct = 0.0
    for a, b in zip(labels_true, labels_pred):
        t[int(a)] += 1.0
        p[int(b)] += 1.0
        if int(a) == int(b):
            correct += 1.0
    dot_tp = 0.0
    dot_pp = 0.0
    dot_tt = 0.0
    for k in range(num_classes):
        dot_tp += t[k] * p[k]
        dot_pp += p[k] * p[k]
        dot_tt += t[k] * t[k]
    cov_ytyp = correct * n - dot_tp
    cov_ypyp = n * n - dot_pp
    cov_ytyt = n * n - dot_tt
    if cov_ypyp * cov_ytyt == 0.0:
        return 0.0
    return cov_ytyp / math.sqrt(cov_ytyt * cov_ypyp)


def chi_square_reference(table, correct=True):
    """Scalar-loop Pearson chi-square, from the definition."""
    table = [[float(v) for v in row] for row in table]
    r = len(table)
    c = len(table[0])
    n = sum(sum(row) for row in table)
    row_sums = [sum(table[i]) for i in range(r)]
    col_sums = [sum(table[i][j] for i in range(r)) for j in range(c)]
    stat = 0.0
    for i in range(r):
        for j in range(c):
            e = row_sums[i] * col_sums[j] / n
            if e == 0.0:
                raise ValueError("zero expected count")
            diff = e - table[i][j]
            if correct and r == 2 and c == 2:
                mag = min(0.5, abs(diff))
                obs = table[i][j] + mag * (1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0))
            else:
                obs = table[i][j]
            stat += (obs - e) ** 2 / e
    dof = (r - 1) * (c - 1)
    return stat, chi_square_survival_model(stat, float(dof)), dof


def g_test_reference(table):
    """Scalar-loop G-test, from the definition."""
    table = [[float(v) for v in row] for row in table]
    r = len(table)
    c = len(table[0])
    n = sum(sum(row) for row in table)
    row_sums = [sum(table[i]) for i in range(r)]
    col_sums = [sum(table[i][j] for i in range(r)) for j in range(c)]
    s = 0.0
    for i in range(r):
        for j in range(c):
            o = table[i][j]
            e = row_sums[i] * col_sums[j] / n
            if e == 0.0:
                raise ValueError("zero expected count")
            if o > 0:
                s += o * math.log(o / e)
    stat = 2.0 * s
    dof = (r - 1) * (c - 1)
    return stat, chi_square_survival_model(stat, float(dof)), dof


# ===========================================================================
# THE FIXTURES
#
# Every one of these has a DISCRIMINATING counterpart: a fixture that
# separates two conventions that look the same until you look closely.
# ===========================================================================


def make_fixtures():
    fx = {}

    # --- F1: the chi-square survival function ------------------------------
    # df = 1 is the repo's existing erfc closed form; df = 2 and df = 10
    # EXERCISE THE GENERAL INCOMPLETE GAMMA, which the repo does not have.
    fx["f1_chi2_x"] = [0.5, 1.0, 3.841458820694124, 10.0, 23.02585092994046, 100.0]
    fx["f1_chi2_df"] = [1.0, 2.0, 1.0, 10.0, 2.0, 50.0]
    fx["f1_chi2_sf"] = [chi_square_survival_model(x, d)
                        for x, d in zip(fx["f1_chi2_x"], fx["f1_chi2_df"])]

    # --- F2: Cohen's kappa, the DISCRIMINATING label-space fixture --------
    # Class 2 is ABSENT from both raters, so the weighted kappa differs
    # between the compressed space and the explicit index space.
    #
    # THIS FIXTURE WAS CHOSEN BY SEARCH, NOT BY HAND: a first attempt used
    # a=[0,0,1,1,3,3,0,1,3,1,0,3] / b=[0,1,1,3,3,0,0,3,3,1,1,0], and it
    # did NOT discriminate -- sklearn gave 0.25 for BOTH the compressed and
    # the explicit reading, because on that fixture the compression
    # happened to leave every weight unchanged. A fixture that cannot
    # discriminate certifies nothing, so the pair below was selected by
    # scanning random label pairs for one whose compressed and explicit
    # linear kappas differ.
    fx["f2_k_a"] = [0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0]
    fx["f2_k_b"] = [0.0, 3.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0]
    fx["f2_k_classes"] = 4
    fx["f2_k_unweighted"] = cohen_kappa_model(fx["f2_k_a"], fx["f2_k_b"], 4, "unweighted")
    fx["f2_k_linear"] = cohen_kappa_model(fx["f2_k_a"], fx["f2_k_b"], 4, "linear")
    fx["f2_k_quadratic"] = cohen_kappa_model(fx["f2_k_a"], fx["f2_k_b"], 4, "quadratic")

    # --- F3: Cohen's kappa, the DEGENERATE case ---------------------------
    # Only one class is ever used, so the chance agreement is 1 and the
    # denominator vanishes. The answer is NaN, NOT 0.
    fx["f3_k_a"] = [0.0, 0.0, 0.0, 0.0]
    fx["f3_k_b"] = [0.0, 0.0, 0.0, 0.0]
    fx["f3_k_classes"] = 2
    fx["f3_k_unweighted"] = cohen_kappa_model(fx["f3_k_a"], fx["f3_k_b"], 2, "unweighted")
    fx["f3_k_linear"] = cohen_kappa_model(fx["f3_k_a"], fx["f3_k_b"], 2, "linear")
    fx["f3_k_quadratic"] = cohen_kappa_model(fx["f3_k_a"], fx["f3_k_b"], 2, "quadratic")

    # --- F4: MCC, a MULTICLASS fixture with a LARGE |MCC| ---------------
    # cm[true][pred] = [[0,2,2],[1,0,4],[1,2,0]]
    #
    # NOTE: MCC is TRANSPOSE-INVARIANT -- measured: |MCC(t,p) - MCC(p,t)|
    # is exactly 0 over 3000 random pairs -- because cov_ytyp uses
    # dot(p_sum, t_sum), which is symmetric, and the denominator is a
    # PRODUCT of the two marginal variances. So an ASYMMETRIC confusion
    # matrix does NOT make the fixture discriminate a transposition, and
    # the plan's original claim that it did was WRONG.
    #
    # What DOES discriminate is the MULTICLASS formula: the fixture is
    # chosen so that |MCC| is large (a wrong sign, a binary-only
    # shortcut, or the wrong marginal all give a visibly different
    # number), and the DEGENERATE case is a separate fixture (F5).
    fx["f4_t"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0]
    fx["f4_p"] = [1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 1.0]
    fx["f4_classes"] = 3
    fx["f4_mcc"] = matthews_correlation_model(fx["f4_t"], fx["f4_p"], 3)

    # --- F5: MCC, the DEGENERATE case ------------------------------------
    # Every prediction is the same class, so the denominator vanishes. The
    # answer is 0.0 -- NOT NaN, which is the OPPOSITE of the kappa case.
    fx["f5_t"] = [0.0, 1.0, 2.0, 1.0]
    fx["f5_p"] = [1.0, 1.0, 1.0, 1.0]
    fx["f5_classes"] = 3
    fx["f5_mcc"] = matthews_correlation_model(fx["f5_t"], fx["f5_p"], 3)

    # --- F6: chi-square on a 2x2, corrected AND uncorrected ---------------
    fx["f6_table"] = [[10.0, 20.0], [30.0, 15.0]]
    fx["f6_corrected"] = chi_square_test_model(fx["f6_table"], correct=True)
    fx["f6_uncorrected"] = chi_square_test_model(fx["f6_table"], correct=False)

    # --- F7: chi-square on a 2x3 -- df = 2, so the p-value NEEDS the
    #         general incomplete gamma. The repo cannot do this today.
    fx["f7_table"] = [[10.0, 20.0, 5.0], [30.0, 15.0, 7.0]]
    fx["f7_chi2"] = chi_square_test_model(fx["f7_table"], correct=True)

    # --- F8: the YATES CLAMP fixture --------------------------------------
    # Every cell has |O - E| = 0.2 < 0.5. The unclamped textbook form
    # gives 0.3125; scipy and R both give 0.0.
    fx["f8_table"] = [[1.0, 1.0], [1.0, 2.0]]
    fx["f8_corrected"] = chi_square_test_model(fx["f8_table"], correct=True)
    fx["f8_uncorrected"] = chi_square_test_model(fx["f8_table"], correct=False)

    # --- F9: the G-test with a ZERO observed cell -------------------------
    # A zero cell contributes exactly 0; the naive 0 * log(0/E) is NaN.
    fx["f9_table"] = [[0.0, 5.0], [10.0, 20.0]]
    fx["f9_g"] = g_test_model(fx["f9_table"])
    fx["f9_g_2x3"] = g_test_model([[10.0, 20.0, 5.0], [30.0, 15.0, 7.0]])

    # --- F10: McNemar, ASYMMETRIC (n01 != n10) -> correction APPLIED -----
    # 12 agreeing pairs of each kind, n01 = 5, n10 = 3.
    fx["f10_a"] = [0.0] * 12 + [0.0] * 5 + [1.0] * 3 + [1.0] * 20
    fx["f10_b"] = [0.0] * 12 + [1.0] * 5 + [0.0] * 3 + [1.0] * 20
    fx["f10_mcnemar"] = mcnemar_test_model(fx["f10_a"], fx["f10_b"], correct=True)
    fx["f10_mcnemar_uncorrected"] = mcnemar_test_model(fx["f10_a"], fx["f10_b"], correct=False)

    # --- F11: McNemar, SYMMETRIC (n01 == n10) -> correction SKIPPED ------
    # The statistic is 0, NOT (0-1)^2/(2*n01). This is THE fixture.
    fx["f11_a"] = [0.0] * 30 + [0.0] * 5 + [1.0] * 5
    fx["f11_b"] = [0.0] * 30 + [1.0] * 5 + [0.0] * 5
    fx["f11_mcnemar"] = mcnemar_test_model(fx["f11_a"], fx["f11_b"], correct=True)

    # --- F12: McNemar, n01 < n10 -> the ABSOLUTE value matters -----------
    # R gives (|3-5|-1)^2/8 = 0.125; the signed form gives 0.5625.
    fx["f12_a"] = [0.0] * 10 + [0.0] * 3 + [1.0] * 5
    fx["f12_b"] = [0.0] * 10 + [1.0] * 3 + [0.0] * 5
    fx["f12_mcnemar"] = mcnemar_test_model(fx["f12_a"], fx["f12_b"], correct=True)

    # --- F13: McNemar, NO discordant pairs -> NaN -------------------------
    fx["f13_a"] = [0.0] * 30 + [1.0] * 30
    fx["f13_b"] = [0.0] * 30 + [1.0] * 30
    fx["f13_mcnemar"] = mcnemar_test_model(fx["f13_a"], fx["f13_b"], correct=True)

    # --- F14: Fisher's exact, a 2x2 with no ties -------------------------
    fx["f14_table"] = [[10.0, 20.0], [30.0, 15.0]]
    fx["f14_fisher"] = fisher_exact_test_model(fx["f14_table"])

    # --- F15: Fisher's exact, a 2x2 WITH ties ----------------------------
    # scipy's own docstring example: p = 0.034965034965034975.
    fx["f15_table"] = [[8.0, 2.0], [1.0, 5.0]]
    fx["f15_fisher"] = fisher_exact_test_model(fx["f15_table"])

    # --- F16: Fisher's exact, the TIE-TOLERANCE fixture -------------------
    # Here two different tables have EXACTLY the same hypergeometric pmf
    # IN EXACT ARITHMETIC, but the float64 evaluation of the two differs
    # in the last bits, so the comparison
    #     pmf(x) <= pmf(observed) * (1 + 1e-14)
    # is what includes the second one and a bare
    #     pmf(x) <= pmf(observed)
    # EXCLUDES it. The p-value then HALVES.
    #
    # Measured on [[0,2],[5,3]]:
    #     scipy.stats.fisher_exact      p = 0.44444444444444453
    #     the sweep, with the tolerance  p = 0.44444444444444459
    #     the sweep, WITHOUT it          p = 0.2222222222222221
    #
    # This fixture was found by SEARCH, and the search had to be run
    # against the MODEL'S OWN float64 arithmetic rather than against
    # scipy's: a first attempt ([[0,3],[7,4]]) has a tie that is bit-exact
    # in the model's lgamma evaluation, so dropping the tolerance changed
    # nothing there and the mutation SURVIVED. A fixture that does not
    # discriminate certifies nothing -- the first one proved that.
    fx["f16_table"] = [[0.0, 2.0], [5.0, 3.0]]
    fx["f16_fisher"] = fisher_exact_test_model(fx["f16_table"])

    return fx


# ===========================================================================
# THE CHECKER
# ===========================================================================


class Checker(object):
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, condition, name):
        if condition:
            self.passed += 1
            print("  [PASS] %s" % name)
        else:
            self.failed += 1
            print("  [FAIL] %s" % name)
        return bool(condition)

    def near(self, got, want, rel_tol, name):
        if not math.isfinite(want):
            same = (math.isnan(got) and math.isnan(want)) or got == want
            return self.ok(same, name)
        denom = max(abs(want), 1e-12)
        rel = abs(got - want) / denom
        good = rel <= rel_tol
        if good:
            self.passed += 1
            print("  [PASS] %s (%.17g, rel %.2e)" % (name, got, rel))
        else:
            self.failed += 1
            print("  [FAIL] %s (got %.17g, want %.17g, rel %.2e > %.2e)"
                  % (name, got, want, rel, rel_tol))
        return good


# ===========================================================================
# THE CHECKS -- properties AND absolute values
# ===========================================================================


def run_checks(fx, ck):
    print("THE CHECKS (properties AND absolute values):")

    # --- chi-square survival ---------------------------------------------
    for i, (x, df) in enumerate(zip(fx["f1_chi2_x"], fx["f1_chi2_df"])):
        ck.near(chi_square_survival_model(x, df), fx["f1_chi2_sf"][i], 1e-13,
                "F1 chi2_sf(x=%g, df=%g)" % (x, df))
    ck.near(chi_square_survival_model(0.0, 3.0), 1.0, 1e-13,
            "F1 chi2_sf(0, 3) == 1")
    ck.ok(chi_square_survival_model(1e5, 1.0) < 1e-12,
          "F1 chi2_sf(1e5, 1) is tiny")

    # --- Cohen's kappa ----------------------------------------------------
    for scheme in ("unweighted", "linear", "quadratic"):
        got = cohen_kappa_model(fx["f2_k_a"], fx["f2_k_b"], 4, scheme)
        ref = cohen_kappa_reference(fx["f2_k_a"], fx["f2_k_b"], 4, scheme)
        ck.near(got["kappa"], ref, 1e-13,
                "F2 kappa %s agrees with the scalar-loop reference" % scheme)
    # --- F2 kappa, the DISCRIMINATING label-space fixture ----------------
    # The UNWEIGHTED kappa is space-INDEPENDENT (the off-diagonal weight
    # matrix is the same whatever the class indices are), so it CANNOT
    # discriminate -- measured: explicit 0.36842105263157887 == compressed
    # 0.36842105263157887. The WEIGHTED ones can, and they do:
    #   linear     explicit 0.4424778761061946  compressed 0.39999999999999991
    #   quadratic  explicit 0.48275862068965514 compressed 0.43243243243243246
    ck.near(fx["f2_k_unweighted"]["kappa"], 0.36842105263157887, 1e-12,
            "F2 kappa unweighted absolute value")
    ck.near(fx["f2_k_linear"]["kappa"], 0.4424778761061946, 1e-12,
            "F2 kappa LINEAR absolute value (explicit index space)")
    ck.near(fx["f2_k_quadratic"]["kappa"], 0.48275862068965514, 1e-12,
            "F2 kappa QUADRATIC absolute value (explicit index space)")
    ck.ok(abs(fx["f2_k_linear"]["kappa"] - fx["f2_k_unweighted"]["kappa"]) > 1e-3,
          "F2 the linear and unweighted answers DIFFER (the fixture discriminates)")
    ck.ok(abs(fx["f2_k_quadratic"]["kappa"] - fx["f2_k_linear"]["kappa"]) > 1e-3,
          "F2 the quadratic and linear answers DIFFER (the fixture discriminates)")
    ck.near(fx["f2_k_unweighted"]["observed_agreement"], 0.55555555555555558, 1e-13,
            "F2 observed agreement")
    ck.near(fx["f2_k_unweighted"]["expected_agreement"], 0.29629629629629628, 1e-13,
            "F2 expected agreement")
    # The present classes are {0, 1, 3}: class 2 is ABSENT, which is what
    # makes the compressed reading a DIFFERENT number.
    present = sorted(set(int(v) for v in fx["f2_k_a"]) | set(int(v) for v in fx["f2_k_b"]))
    ck.ok(present == [0, 1, 3],
          "F2 class 2 is ABSENT from the fixture (the compressed reading differs)")

    # --- kappa, degenerate -------------------------------------------------
    for scheme in ("unweighted", "linear", "quadratic"):
        got = cohen_kappa_model(fx["f3_k_a"], fx["f3_k_b"], 2, scheme)
        ck.ok(math.isnan(got["kappa"]),
              "F3 kappa %s degenerate is NaN (NOT 0)" % scheme)

    # --- MCC ---------------------------------------------------------------
    ck.near(fx["f4_mcc"]["mcc"], matthews_reference(fx["f4_t"], fx["f4_p"], 3),
            1e-13, "F4 MCC agrees with the scalar-loop reference")
    ck.near(fx["f4_mcc"]["mcc"], -0.50576942170357386, 1e-12,
            "F4 MCC absolute value")
    # The fixture is MULTICLASS: a binary-only shortcut (which sklearn
    # itself used to take) would give a visibly different number.
    ck.near(fx["f4_mcc"]["mcc"], matthews_reference(fx["f4_t"], fx["f4_p"], 3),
            1e-13, "F4 the MCC fixture is multiclass (|MCC| is large)")
    ck.ok(abs(fx["f4_mcc"]["mcc"]) > 0.4,
          "F4 |MCC| is large, so a wrong formula is visibly wrong")
    # MCC is TRANSPOSE-INVARIANT -- a property worth pinning, because it
    # says the fixture cannot be used to catch a row/column swap.
    ck.near(matthews_correlation_model(fx["f4_p"], fx["f4_t"], 3)["mcc"],
            fx["f4_mcc"]["mcc"], 1e-13,
            "F4 MCC is transpose-invariant (a measured property, not a hope)")
    ck.near(fx["f5_mcc"]["mcc"], 0.0, 1e-15,
            "F5 MCC degenerate is 0.0 (NOT NaN)")
    ck.ok(not math.isnan(fx["f5_mcc"]["mcc"]),
          "F5 MCC degenerate is not NaN")
    # The MCC is invariant to the label space, but the KAPPA is not.
    ck.near(matthews_correlation_model(fx["f4_t"], fx["f4_p"], 5)["mcc"],
            fx["f4_mcc"]["mcc"], 1e-13,
            "F4 MCC is invariant to the label space")

    # --- chi-square --------------------------------------------------------
    ck.near(fx["f6_corrected"]["statistic"], 6.752232142857143, 1e-13,
            "F6 chi-square 2x2 CORRECTED absolute value")
    ck.near(fx["f6_uncorrected"]["statistic"], 8.035714285714286, 1e-13,
            "F6 chi-square 2x2 UNCORRECTED absolute value")
    ck.ok(fx["f6_corrected"]["statistic"] != fx["f6_uncorrected"]["statistic"],
          "F6 the corrected and uncorrected statistics DIFFER")
    ck.near(fx["f6_corrected"]["degrees_of_freedom"], 1.0, 1e-15,
            "F6 chi-square df")
    stat, p, dof = chi_square_reference(fx["f6_table"], True)
    ck.near(stat, fx["f6_corrected"]["statistic"], 1e-13,
            "F6 chi-square agrees with the scalar-loop reference")
    ck.near(fx["f7_chi2"]["degrees_of_freedom"], 2.0, 1e-15,
            "F7 chi-square 2x3 df == 2 (needs the general incomplete gamma)")
    ck.near(fx["f7_chi2"]["statistic"], 8.0324764521193099, 1e-13,
            "F7 chi-square 2x3 absolute value")
    ck.near(fx["f7_chi2"]["p_value"], 0.018020627120288131, 1e-12,
            "F7 chi-square 2x3 p-value")

    # --- the Yates clamp ---------------------------------------------------
    ck.near(fx["f8_corrected"]["statistic"], 0.0, 1e-15,
            "F8 the CLAMPED Yates statistic is 0.0 on [[1,1],[1,2]]")
    # The UNCLAMPED form would give 0.3125 for the CORRECTED statistic.
    # The UNCORRECTED flag gives the plain Pearson, 0.13888888888888881.
    ck.near(fx["f8_uncorrected"]["statistic"], 0.13888888888888881, 1e-13,
            "F8 the UNCORRECTED statistic is the plain Pearson")
    ck.ok(fx["f8_uncorrected"]["statistic"] > 0.13,
          "F8 the uncorrected statistic is > 0.13 (the two forms differ)")
    ck.near(fx["f8_corrected"]["p_value"], 1.0, 1e-15,
            "F8 the clamped corrected p-value is 1")

    # --- G-test ------------------------------------------------------------
    stat, p, dof = g_test_reference(fx["f9_table"])
    ck.near(fx["f9_g"]["statistic"], stat, 1e-13,
            "F9 G-test agrees with the scalar-loop reference")
    ck.ok(math.isfinite(fx["f9_g"]["statistic"]),
          "F9 the G-test is FINITE on a table with a zero cell")
    ck.near(fx["f9_g"]["statistic"], 3.6880211032792385, 1e-13,
            "F9 G-test absolute value with a zero cell")
    ck.near(fx["f9_g"]["p_value"], 0.054804602230952994, 1e-12,
            "F9 G-test p-value with a zero cell")
    ck.near(fx["f9_g"]["degrees_of_freedom"], 1.0, 1e-15, "F9 G-test df")
    ck.near(fx["f9_g_2x3"]["statistic"], 8.1732843268358728, 1e-13,
            "F9 G-test 2x3 absolute value")
    ck.near(fx["f9_g_2x3"]["p_value"], 0.016795535643078025, 1e-12,
            "F9 G-test 2x3 p-value")
    ck.near(fx["f9_g_2x3"]["degrees_of_freedom"], 2.0, 1e-15,
            "F9 G-test 2x3 df")

    # --- McNemar -----------------------------------------------------------
    ck.near(fx["f10_mcnemar"]["statistic"], 0.125, 1e-13,
            "F10 McNemar asymmetric statistic (the correction IS applied)")
    ck.near(fx["f10_mcnemar"]["p_value"], 0.723673609831763, 1e-12,
            "F10 McNemar asymmetric p-value")
    ck.near(fx["f10_mcnemar"]["n01"], 5.0, 1e-15, "F10 McNemar n01")
    ck.near(fx["f10_mcnemar"]["n10"], 3.0, 1e-15, "F10 McNemar n10")
    ck.near(fx["f10_mcnemar_uncorrected"]["statistic"], 0.5, 1e-13,
            "F10 McNemar uncorrected statistic")
    ck.near(fx["f11_mcnemar"]["statistic"], 0.0, 1e-15,
            "F11 McNemar SYMMETRIC statistic is 0 (the correction is SKIPPED)")
    ck.near(fx["f11_mcnemar"]["p_value"], 1.0, 1e-15,
            "F11 McNemar symmetric p-value is 1")
    ck.ok(fx["f11_mcnemar"]["statistic"] != 0.1,
          "F11 the symmetric statistic is NOT 1/(2*n01) (the fixture discriminates)")
    ck.near(fx["f12_mcnemar"]["statistic"], 0.125, 1e-13,
            "F12 McNemar n01 < n10 uses the ABSOLUTE value")
    ck.near(fx["f13_mcnemar"]["statistic"], float("nan"), 1e-15,
            "F13 McNemar with no discordant pairs is NaN")
    ck.near(fx["f13_mcnemar"]["exact_p_value"], 1.0, 1e-15,
            "F13 McNemar no-discordant exact p-value is 1 (a vacuous test)")

    # --- Fisher ------------------------------------------------------------
    ck.near(fx["f14_fisher"]["p_value"], 0.0088247602726759869, 1e-12,
            "F14 Fisher p-value absolute value")
    ck.near(fx["f14_fisher"]["odds_ratio"], 0.25, 1e-14,
            "F14 Fisher odds ratio (the cross-product, NOT R's conditional MLE)")
    ck.near(fx["f15_fisher"]["p_value"], 0.034965034965034975, 1e-13,
            "F15 Fisher p-value with ties (probability ordering)")
    ck.near(fx["f15_fisher"]["odds_ratio"], 20.0, 1e-13,
            "F15 Fisher odds ratio with ties")
    # F16: the tie tolerance. Two tables share the observed pmf in exact
    # arithmetic, so the 1 + 1e-14 relative tolerance is what includes the
    # second one.
    ck.near(fx["f16_fisher"]["p_value"], 0.44444444444444459, 1e-13,
            "F16 Fisher p-value with an EXACT pmf tie (the tolerance)")
    ck.near(fx["f16_fisher"]["odds_ratio"], 0.0, 1e-15,
            "F16 Fisher odds ratio with an exact pmf tie")

    # --- guards ------------------------------------------------------------
    for name, fn in (("chi-square", lambda: chi_square_test_model([[0.0, 0.0], [3.0, 3.0]])),
                     ("G-test", lambda: g_test_model([[0.0, 0.0], [3.0, 3.0]]))):
        threw = False
        try:
            fn()
        except ValueError:
            threw = True
        ck.ok(threw, "the %s rejects a table with an empty row (zero expected count)" % name)


# ===========================================================================
# THE REFERENCE CROSS-CHECK
# ===========================================================================


def cross_check(fx):
    passed = 0
    failed = 0

    def report(ok, name, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("  [PASS] %s %s" % (name, detail))
        else:
            failed += 1
            print("  [FAIL] %s %s" % (name, detail))

    try:
        from scipy import stats as sps
        from sklearn import metrics as skm
        import sklearn
    except Exception as exc:  # pragma: no cover
        print("  [SKIP] scipy/sklearn unavailable: %s" % exc)
        return 0, 0

    print("THE REFERENCE CROSS-CHECK (scipy %s, scikit-learn %s):"
          % (sps.__name__ and __import__("scipy").__version__, sklearn.__version__))

    # --- chi-square survival vs scipy.stats.chi2.sf ----------------------
    worst = 0.0
    for df in (1, 2, 3, 4, 5, 7, 10, 15, 30, 50, 100):
        for x in (1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1e4, 1e5):
            got = chi_square_survival_model(x, float(df))
            want = float(sps.chi2.sf(x, df))
            rel = abs(got - want) / max(abs(want), 1e-300)
            worst = max(worst, rel)
    report(worst < 1e-12, "chi_square_survival vs scipy.stats.chi2.sf",
           "(worst rel %.3g over 13 df x 13 x)" % worst)

    # The df = 1 path and the GENERAL path must agree -- two independent
    # routes to the same number, which is what makes the incomplete gamma
    # trustworthy.
    worst2 = 0.0
    for x in (1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1e4):
        g = gammq_model(0.5, x / 2.0)
        e = math.erfc(math.sqrt(x / 2.0))
        worst2 = max(worst2, abs(g - e) / max(abs(e), 1e-300))
    report(worst2 < 1e-12, "the general incomplete gamma vs the df=1 erfc closed form",
           "(worst rel %.3g)" % worst2)

    # --- kappa vs scikit-learn --------------------------------------------
    a = np.array(fx["f2_k_a"])
    b = np.array(fx["f2_k_b"])
    # sklearn's `weights` is None for the unweighted case, NOT "unweighted".
    for scheme, key in ((None, "unweighted"), ("linear", "linear"),
                        ("quadratic", "quadratic")):
        sk = skm.cohen_kappa_score(a, b, labels=list(range(4)), weights=scheme)
        mine = fx["f2_k_%s" % key]["kappa"]
        report(abs(sk - mine) < 1e-12,
               "kappa %s vs scikit-learn (labels=range(4))" % key,
               "(sklearn %.15g, mine %.15g)" % (sk, mine))
    # And the COMPRESSED reading really is different -- otherwise the
    # fixture would not discriminate.
    sk_comp = skm.cohen_kappa_score(a, b, weights="linear")
    report(abs(sk_comp - fx["f2_k_linear"]["kappa"]) > 1e-6,
           "the COMPRESSED (labels=None) linear kappa DIFFERS from ours",
           "(sklearn %.15g, ours %.15g)" % (sk_comp, fx["f2_k_linear"]["kappa"]))

    # --- MCC vs scikit-learn ----------------------------------------------
    t = np.array(fx["f4_t"])
    p = np.array(fx["f4_p"])
    sk = skm.matthews_corrcoef(t, p)
    report(abs(sk - fx["f4_mcc"]["mcc"]) < 1e-12, "MCC vs scikit-learn",
           "(sklearn %.15g, mine %.15g)" % (sk, fx["f4_mcc"]["mcc"]))

    # --- chi-square vs scipy.stats.chi2_contingency ----------------------
    for name, key, corr in (("F6 corrected", "f6_corrected", True),
                            ("F6 uncorrected", "f6_uncorrected", False),
                            ("F8 clamp", "f8_corrected", True)):
        tbl = fx["f6_table"] if name.startswith("F6") else fx["f8_table"]
        res = sps.chi2_contingency(np.array(tbl), correction=corr)
        mine = fx[key]
        ok = (abs(float(res.statistic) - mine["statistic"]) < 1e-12
              and abs(float(res.pvalue) - mine["p_value"]) < 1e-12
              and int(res.dof) == mine["degrees_of_freedom"])
        report(ok, "chi-square %s vs scipy.stats.chi2_contingency" % name,
               "(scipy stat %.15g p %.15g | mine stat %.15g p %.15g)"
               % (res.statistic, res.pvalue, mine["statistic"], mine["p_value"]))

    # --- G-test vs scipy.stats.power_divergence(lambda_=0) ----------------
    # NOTE the inverted lambda_: lambda_=1 is PEARSON, lambda_=0 is the
    # G-TEST in scipy's implementation.
    for name, tbl, key in (("F9 zero-cell", fx["f9_table"], "f9_g"),
                           ("F9 2x3", fx["f9_g_2x3"] and [[10.0, 20.0, 5.0], [30.0, 15.0, 7.0]],
                            "f9_g_2x3")):
        arr = np.array(tbl, dtype=float)
        exp = np.outer(arr.sum(1), arr.sum(0)) / arr.sum()
        res = sps.power_divergence(arr, exp, lambda_=0, axis=None)
        sk_stat = float(np.asarray(res.statistic).ravel()[0])
        mine = fx[key]["statistic"]
        report(abs(sk_stat - mine) < 1e-12, "G-test %s vs scipy.power_divergence" % name,
               "(scipy %.15g, mine %.15g)" % (sk_stat, mine))
    # The p-value df is (R-1)(C-1), NOT scipy's power_divergence df.
    arr = np.array([[10.0, 20.0, 5.0], [30.0, 15.0, 7.0]])
    exp = np.outer(arr.sum(1), arr.sum(0)) / arr.sum()
    g = float(np.asarray(sps.power_divergence(arr, exp, lambda_=0, axis=None).statistic).ravel()[0])
    report(abs(float(sps.chi2.sf(g, 2)) - fx["f9_g_2x3"]["p_value"]) < 1e-13,
           "the G-test p-value uses df = (R-1)(C-1), not scipy's cells-1",
           "(%.15g vs %.15g)" % (sps.chi2.sf(g, 2), fx["f9_g_2x3"]["p_value"]))

    # --- McNemar's exact p-value vs scipy.stats.binomtest -----------------
    for name, key in (("F10", "f10_mcnemar"), ("F11", "f11_mcnemar"),
                      ("F12", "f12_mcnemar"), ("F13", "f13_mcnemar")):
        n01 = int(fx[key]["n01"])
        n10 = int(fx[key]["n10"])
        got = fx[key]["exact_p_value"]
        if n01 + n10 == 0:
            # scipy.stats.binomtest REQUIRES n >= 1, so there is no
            # reference for the zero-discordant case. The repo's own
            # binom_pvalue_half answers 1.0 there ("a vacuous test is not
            # evidence of anything, so it does not reject"), and that is
            # the value the module ships.
            report(got == 1.0, "McNemar %s exact p-value (no discordant pairs) is 1" % name,
                   "(scipy.binomtest has no answer for n=0; ours %.15g)" % got)
            continue
        want = float(sps.binomtest(min(n01, n10), n01 + n10, 0.5,
                                   alternative="two-sided").pvalue)
        report(abs(got - want) < 1e-13, "McNemar %s exact p-value vs scipy.binomtest" % name,
               "(scipy %.15g, mine %.15g)" % (want, got))

    # --- Fisher vs scipy.stats.fisher_exact -------------------------------
    for name, key in (("F14", "f14_fisher"), ("F15", "f15_fisher"),
                      ("F16", "f16_fisher")):
        tbl = np.array(fx["f%s_table" % name[1:]], dtype=float)
        res = sps.fisher_exact(tbl)
        mine = fx[key]
        ok = abs(float(res.pvalue) - mine["p_value"]) < 1e-12
        report(ok, "Fisher %s p-value vs scipy.stats.fisher_exact" % name,
               "(scipy %.15g, mine %.15g)" % (res.pvalue, mine["p_value"]))
        report(abs(float(res.statistic) - mine["odds_ratio"]) < 1e-12,
               "Fisher %s odds ratio vs scipy" % name,
               "(scipy %.15g, mine %.15g)" % (res.statistic, mine["odds_ratio"]))
    # The DOUBLED-TAIL rule is NOT the two-sided rule -- and the fixture
    # must be able to tell them apart.
    tbl = np.array(fx["f14_table"], dtype=float)
    a_, b_ = int(tbl[0][0]), int(tbl[0][1])
    c_, d_ = int(tbl[1][0]), int(tbl[1][1])
    n1, n2, n, m = a_ + b_, c_ + d_, a_ + c_, a_ + b_ + c_ + d_
    k = min(a_, c_)
    from scipy.stats import hypergeom
    doubled = min(1.0, 2.0 * min(float(hypergeom.cdf(k, m, n1, n)),
                                 float(hypergeom.sf(k - 1, m, n1, n))))
    report(abs(doubled - fx["f14_fisher"]["p_value"]) > 1e-9,
           "the DOUBLED-TAIL rule DIFFERS from the probability-ordering rule",
           "(doubled %.15g, probability-ordering %.15g)"
           % (doubled, fx["f14_fisher"]["p_value"]))

    # --- R, if available ---------------------------------------------------
    if shutil.which("Rscript"):
        r_pass, r_fail = r_cross_check(fx)
        passed += r_pass
        failed += r_fail
    else:
        print("  [SKIP] R cross-check (Rscript not on PATH)")

    return passed, failed


def r_cross_check(fx):
    """Compare the McNemar and chi-square fixtures against R.

    R is the reference for McNemar's continuity-correction rule, and that
    rule is NOT the textbook one, so the check is worth having.
    """
    passed = 0
    failed = 0

    def report(ok, name, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("  [PASS] %s %s" % (name, detail))
        else:
            failed += 1
            print("  [FAIL] %s %s" % (name, detail))

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_r_contingency.R")
    if not os.path.exists(script):
        print("  [SKIP] R cross-check (helper script missing)")
        return 0, 0
    try:
        out = subprocess.run(["Rscript", script], capture_output=True, text=True,
                             timeout=120)
    except Exception as exc:
        print("  [SKIP] R cross-check failed to run: %s" % exc)
        return 0, 0
    if out.returncode != 0:
        print("  [SKIP] R cross-check (Rscript exited %d)" % out.returncode)
        print(out.stderr[:500])
        return 0, 0

    for line in out.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) < 3:
            continue
        tag = parts[0]
        try:
            r_stat = float(parts[1])
            r_p = float(parts[2])
        except ValueError:
            continue
        if tag == "MCNEMAR":
            key = {"F10": "f10_mcnemar", "F11": "f11_mcnemar",
                   "F12": "f12_mcnemar", "F13": "f13_mcnemar"}.get(parts[3])
            if key is None:
                continue
            mine = fx[key]
            if math.isnan(r_stat):
                report(math.isnan(mine["statistic"]),
                       "R's McNemar %s statistic is NaN, ours too" % parts[3])
            else:
                report(abs(r_stat - mine["statistic"]) < 1e-12
                       and abs(r_p - mine["p_value"]) < 1e-12,
                       "McNemar %s vs R" % parts[3],
                       "(R stat %.15g p %.15g | mine stat %.15g p %.15g)"
                       % (r_stat, r_p, mine["statistic"], mine["p_value"]))
        elif tag == "CHISQ":
            key = {"F6C": "f6_corrected", "F6U": "f6_uncorrected",
                   "F8C": "f8_corrected"}.get(parts[3])
            if key is None:
                continue
            mine = fx[key]
            report(abs(r_stat - mine["statistic"]) < 1e-12
                   and abs(r_p - mine["p_value"]) < 1e-12,
                   "chi-square %s vs R" % parts[3],
                   "(R stat %.15g p %.15g | mine stat %.15g p %.15g)"
                   % (r_stat, r_p, mine["statistic"], mine["p_value"]))
    return passed, failed


# ===========================================================================
# THE C++ FIXTURE EMITTER
# ===========================================================================


def fmt(v):
    """A C++ double literal that round-trips exactly."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return "%d" % int(v)
    f = float(v)
    if math.isnan(f):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(f):
        return "std::numeric_limits<double>::infinity()"
    if f == int(f) and abs(f) < 1e15:
        return "%.1f" % f
    return repr(f)


def _vec(name, values):
    return "const std::vector<double> %s = {%s};" % (
        name, ", ".join(fmt(v) for v in values))


def _mat(name, rows):
    body = ",\n".join("    {%s}" % ", ".join(fmt(v) for v in row) for row in rows)
    return "const std::vector<std::vector<double>> %s = {\n%s};" % (name, body)


def emit_cpp(fx):
    out = []
    A = out.append

    A("// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<")
    A("// Emitted by tests/contingency_oracle.py --emit-cpp. float64 numpy.")
    A("// The numbers are TRANSFERRED, never re-derived by hand.")

    A("// --- F1: the chi-square survival function -------------------------")
    A("// df = 1 is the repo's EXISTING erfc closed form; df = 2 and 10")
    A("// EXERCISE THE GENERAL INCOMPLETE GAMMA, which the repo lacks.")
    A(_vec("f1_chi2_x", fx["f1_chi2_x"]))
    A(_vec("f1_chi2_df", fx["f1_chi2_df"]))
    A(_vec("f1_chi2_sf", fx["f1_chi2_sf"]))

    A("// --- F2: Cohen's kappa, the DISCRIMINATING label-space fixture ----")
    A("// Class 2 is ABSENT, so the weighted kappa differs between the")
    A("// compressed space and the explicit class-index space.")
    A(_vec("f2_k_a", fx["f2_k_a"]))
    A(_vec("f2_k_b", fx["f2_k_b"]))
    A("const size_t f2_k_classes = %d;" % fx["f2_k_classes"])
    for scheme in ("unweighted", "linear", "quadratic"):
        k = fx["f2_k_%s" % scheme]
        A("const double f2_k_%s_kappa = %s;" % (scheme, fmt(k["kappa"])))
        A("const double f2_k_%s_po = %s;" % (scheme, fmt(k["observed_agreement"])))
        A("const double f2_k_%s_pe = %s;" % (scheme, fmt(k["expected_agreement"])))

    A("// --- F3: Cohen's kappa, the DEGENERATE case -----------------------")
    A("// The answer is NaN, NOT 0.")
    A(_vec("f3_k_a", fx["f3_k_a"]))
    A(_vec("f3_k_b", fx["f3_k_b"]))
    A("const size_t f3_k_classes = %d;" % fx["f3_k_classes"])

    A("// --- F4: MCC, an ASYMMETRIC multiclass fixture --------------------")
    A(_vec("f4_t", fx["f4_t"]))
    A(_vec("f4_p", fx["f4_p"]))
    A("const size_t f4_classes = %d;" % fx["f4_classes"])
    A("const double f4_mcc = %s;" % fmt(fx["f4_mcc"]["mcc"]))

    A("// --- F5: MCC, the DEGENERATE case ---------------------------------")
    A("// The answer is 0.0, NOT NaN -- the OPPOSITE of the kappa case.")
    A(_vec("f5_t", fx["f5_t"]))
    A(_vec("f5_p", fx["f5_p"]))
    A("const size_t f5_classes = %d;" % fx["f5_classes"])
    A("const double f5_mcc = %s;" % fmt(fx["f5_mcc"]["mcc"]))

    A("// --- F6/F7/F8: the chi-square tests -------------------------------")
    A(_mat("f6_table", fx["f6_table"]))
    A("const double f6_corrected_stat = %s;" % fmt(fx["f6_corrected"]["statistic"]))
    A("const double f6_corrected_p = %s;" % fmt(fx["f6_corrected"]["p_value"]))
    A("const double f6_uncorrected_stat = %s;" % fmt(fx["f6_uncorrected"]["statistic"]))
    A("const double f6_uncorrected_p = %s;" % fmt(fx["f6_uncorrected"]["p_value"]))
    A("const size_t f6_dof = %d;" % fx["f6_corrected"]["degrees_of_freedom"])
    A(_mat("f7_table", fx["f7_table"]))
    A("const double f7_chi2_stat = %s;" % fmt(fx["f7_chi2"]["statistic"]))
    A("const double f7_chi2_p = %s;" % fmt(fx["f7_chi2"]["p_value"]))
    A("const size_t f7_dof = %d;" % fx["f7_chi2"]["degrees_of_freedom"])
    A("// The YATES CLAMP: every cell has |O-E| = 0.2 < 0.5, so the")
    A("// clamped statistic is 0.0 and the unclamped one is 0.3125.")
    A(_mat("f8_table", fx["f8_table"]))
    A("const double f8_corrected_stat = %s;" % fmt(fx["f8_corrected"]["statistic"]))
    A("const double f8_corrected_p = %s;" % fmt(fx["f8_corrected"]["p_value"]))
    A("const double f8_uncorrected_stat = %s;" % fmt(fx["f8_uncorrected"]["statistic"]))
    A("const double f8_uncorrected_p = %s;" % fmt(fx["f8_uncorrected"]["p_value"]))

    A("// --- F9: the G-test, with a ZERO observed cell ---------------------")
    A(_mat("f9_table", fx["f9_table"]))
    A("const double f9_g_stat = %s;" % fmt(fx["f9_g"]["statistic"]))
    A("const double f9_g_p = %s;" % fmt(fx["f9_g"]["p_value"]))
    A("const size_t f9_dof = %d;" % fx["f9_g"]["degrees_of_freedom"])
    A(_mat("f9_g_2x3_table", [[10.0, 20.0, 5.0], [30.0, 15.0, 7.0]]))
    A("const double f9_g_2x3_stat = %s;" % fmt(fx["f9_g_2x3"]["statistic"]))
    A("const double f9_g_2x3_p = %s;" % fmt(fx["f9_g_2x3"]["p_value"]))
    A("const size_t f9_g_2x3_dof = %d;" % fx["f9_g_2x3"]["degrees_of_freedom"])

    A("// --- F10-F13: McNemar's test --------------------------------------")
    A("// F10: n01 != n10 -> the correction IS applied.")
    A(_vec("f10_a", fx["f10_a"]))
    A(_vec("f10_b", fx["f10_b"]))
    A("const double f10_mcnemar_stat = %s;" % fmt(fx["f10_mcnemar"]["statistic"]))
    A("const double f10_mcnemar_p = %s;" % fmt(fx["f10_mcnemar"]["p_value"]))
    A("const double f10_mcnemar_exact_p = %s;" % fmt(fx["f10_mcnemar"]["exact_p_value"]))
    A("const double f10_mcnemar_uncorr_stat = %s;"
      % fmt(fx["f10_mcnemar_uncorrected"]["statistic"]))
    A("const double f10_n01 = %s;" % fmt(fx["f10_mcnemar"]["n01"]))
    A("const double f10_n10 = %s;" % fmt(fx["f10_mcnemar"]["n10"]))
    A("// F11: n01 == n10 -> the correction is SKIPPED and the statistic is 0.")
    A(_vec("f11_a", fx["f11_a"]))
    A(_vec("f11_b", fx["f11_b"]))
    A("const double f11_mcnemar_stat = %s;" % fmt(fx["f11_mcnemar"]["statistic"]))
    A("const double f11_mcnemar_p = %s;" % fmt(fx["f11_mcnemar"]["p_value"]))
    A("// F12: n01 < n10 -> the ABSOLUTE value matters.")
    A(_vec("f12_a", fx["f12_a"]))
    A(_vec("f12_b", fx["f12_b"]))
    A("const double f12_mcnemar_stat = %s;" % fmt(fx["f12_mcnemar"]["statistic"]))
    A("const double f12_mcnemar_p = %s;" % fmt(fx["f12_mcnemar"]["p_value"]))
    A("// F13: no discordant pairs -> NaN, not 0.")
    A(_vec("f13_a", fx["f13_a"]))
    A(_vec("f13_b", fx["f13_b"]))
    A("const double f13_mcnemar_stat = %s;" % fmt(fx["f13_mcnemar"]["statistic"]))
    A("const double f13_mcnemar_p = %s;" % fmt(fx["f13_mcnemar"]["p_value"]))
    A("const double f13_mcnemar_exact_p = %s;" % fmt(fx["f13_mcnemar"]["exact_p_value"]))

    A("// --- F14/F15: Fisher's exact test ---------------------------------")
    A(_mat("f14_table", fx["f14_table"]))
    A("const double f14_fisher_p = %s;" % fmt(fx["f14_fisher"]["p_value"]))
    A("const double f14_fisher_or = %s;" % fmt(fx["f14_fisher"]["odds_ratio"]))
    A(_mat("f15_table", fx["f15_table"]))
    A("const double f15_fisher_p = %s;" % fmt(fx["f15_fisher"]["p_value"]))
    A("const double f15_fisher_or = %s;" % fmt(fx["f15_fisher"]["odds_ratio"]))
    A("// The TIE-TOLERANCE fixture: two tables share the observed pmf")
    A("// EXACTLY, so scipy's relative tolerance (1 + 1e-14) is what")
    A("// includes the second one. Without it the p-value HALVES.")
    A(_mat("f16_table", fx["f16_table"]))
    A("const double f16_fisher_p = %s;" % fmt(fx["f16_fisher"]["p_value"]))
    A("const double f16_fisher_or = %s;" % fmt(fx["f16_fisher"]["odds_ratio"]))

    A("// >>> ORACLE-EMITTED FIXTURE END <<<")
    return "\n".join(out)


# ===========================================================================


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
    print("tests/contingency_oracle.py -- the contingency-statistics specification")
    print("=" * 74)
    print()
    ck = Checker()
    run_checks(fx, ck)
    print()
    p, f = cross_check(fx)
    print()
    print("=" * 74)
    print("oracle: %d passed, %d failed" % (ck.passed + p, ck.failed + f))
    print("=" * 74)
    return 1 if (ck.failed + f) else 0


if __name__ == "__main__":
    sys.exit(main())
