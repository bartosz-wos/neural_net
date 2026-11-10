#!/usr/bin/env python3
"""
tests/isotonic_oracle.py -- the float64 numpy SPECIFICATION for the isotonic
regression and monotone calibration utilities.

THIS ORACLE IS THE SPECIFICATION. It was written and run BEFORE any C++ for
this feature existed.

WHY THIS FEATURE, MEASURED NOT ASSUMED (2026-11-10)
====================================================

    grep -rinE 'isotonic|monotone.calib|pool.adjacent|PAVA' include/ tests/
    -> 0 hits on the concept

The repo has ~250 layers and 60+ utility files and NO monotone calibration
map. What it has is the PARAMETRIC half of the same problem --
utils/temperature_scaling.h fits ONE scalar T -- plus the MEASURING half,
utils/metrics.h's ECE / Brier / reliability diagrams, which are descriptive
only.

Isotonic regression is the standard non-parametric alternative to Platt
scaling and to temperature scaling, and it is the method the calibration
literature compares against.

WHY THE CROSS-CHECK AGAINST SCIKIT-LEARN IS THE POINT
======================================================

**A PAVA implementation written by the same author as the C++ is not
independent.** This oracle's PAVA is therefore checked against
scikit-learn 1.9.0's `sklearn.isotonic.IsotonicRegression` -- a MAINTAINED
reference implementation -- on hundreds of RANDOMISED fixtures with HEAVY
TIES in x and NON-UNIFORM WEIGHTS, in both directions, plus the three
out_of_bounds regimes and the tie-collapse / trimming rules.

The same discipline the n-gram session used against NLTK's
`WittenBellProbDist`, and the rank-correlation session used against
scipy.stats.

WHAT IS *NOT* USED AS THE ORACLE
================================

Do NOT use `sklearn.isotonic.IsotonicRegression` AS the oracle. It is used
as a CROSS-CHECK -- an independent implementation of the same mathematics --
and the PAVA below is written FROM THE MATHEMATICS (Best & Chakravarti's
active-set framework), so that the two agreeing is evidence.

THE FOUR THINGS THAT ARE EASY TO GET WRONG
===========================================

1. TIES IN x ARE COLLAPSED TO A **WEIGHTED MEAN** of y, AND THE WEIGHTS ARE
   **SUMMED**. Collapsing to the unweighted mean, or forgetting to sum the
   weights, diverges on any fixture with repeated x.

2. PAVA POOLS **ADJACENT** violators, and the pooled block's value is the
   **weighted mean of the points it contains** -- NOT the mean of the two
   block MEANS (which is wrong whenever the blocks differ in weight or
   size).

3. THE FITTED VALUES ARE THE BLOCK MEANS, BUT THE **PREDICTION** IS A LINEAR
   INTERPOLATION through them, NOT THE STEP FUNCTION. The isotonic
   regression itself is a step function; scikit-learn's `predict` is a
   piecewise-linear interpolant through the retained knots.

4. THE RETAINED KNOTS ARE THE **TRIMMED** step values: interior points whose
   fitted value equals BOTH neighbours' are dropped. The trimming does not
   change the interpolated function, but it DOES change the knot list, which
   is part of the public output.

RUN
===

    /usr/bin/python3 tests/isotonic_oracle.py

NOTE: `/usr/bin/python3` has numpy 2.3.5 + scipy 1.17.1 + sklearn 1.9.0.
The `python3` on PATH is the Hermes venv, which has NO numpy.
"""

import sys

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover - the oracle REQUIRES sklearn
    print("FATAL: this oracle requires scikit-learn (pip install scikit-learn)")
    sys.exit(2)

try:
    from sklearn.metrics import brier_score_loss as sk_brier
    # `calibration_curve` moved from sklearn.metrics to sklearn.calibration in
    # scikit-learn 1.9. Accept either location so the oracle runs on both.
    try:
        from sklearn.calibration import calibration_curve as sk_calibration_curve
    except ImportError:  # pragma: no cover
        from sklearn.metrics import calibration_curve as sk_calibration_curve
except ImportError:  # pragma: no cover
    print("FATAL: this oracle requires scikit-learn")
    sys.exit(2)


# ===========================================================================
# The oracle harness
# ===========================================================================

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print("  [PASS] %s" % name)
    else:
        FAILED += 1
        print("  [FAIL] %s%s" % (name, ("  -- " + detail) if detail else ""))


def check_close(name, got, want, tol=1e-12):
    ok = abs(got - want) <= tol + tol * abs(want)
    check(name, ok, "got %r want %r" % (got, want))


def check_close_vec(name, got, want, tol=1e-12):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        check(name, False, "shape got %r want %r" % (got.shape, want.shape))
        return
    ok = np.all(np.abs(got - want) <= tol + tol * np.abs(want))
    check(name, bool(ok), "got %r want %r" % (got, want))


# ===========================================================================
# THE ALGORITHM -- PAVA, written from the mathematics
# ===========================================================================

def collapse_ties(x, y, w):
    """Collapse equal x to a WEIGHTED MEAN of y, with the weights SUMMED.

    Returns (unique_x, weighted_mean_y, summed_weight).
    """
    order = np.lexsort((y, x))     # sort by x, ties broken by y
    x, y, w = x[order], y[order], w[order]
    ux, inverse = np.unique(x, return_inverse=True)
    sum_wy = np.zeros(len(ux))
    sum_w = np.zeros(len(ux))
    np.add.at(sum_wy, inverse, w * y)
    np.add.at(sum_w, inverse, w)
    return ux, sum_wy / sum_w, sum_w


def pava(v, w, increasing=True):
    """The weighted-L2 isotonic regression, by the Pool-Adjacent-Violators
    algorithm. `v` is the per-point VALUE and `w` the per-point WEIGHT.

    Returns the fitted value at every input point.
    """
    n = len(v)
    sum_wy = list(w * v)        # the weighted SUM of y in each block
    sum_w = list(w)             # the summed weight of each block
    count = [1] * n             # the number of points in each block
    first = list(range(n))      # the first input index in each block
    blocks = []

    for i in range(n):
        blocks.append(i)
        while len(blocks) >= 2:
            a, b = blocks[-2], blocks[-1]
            va = sum_wy[a] / sum_w[a]
            vb = sum_wy[b] / sum_w[b]
            violation = (va > vb) if increasing else (va < vb)
            if not violation:
                break
            # POOL THE ADJACENT VIOLATORS. The merged block's value is the
            # weighted mean of ALL the points it contains -- computed from
            # the accumulated sums, never as the mean of the two means.
            sum_wy[a] += sum_wy[b]
            sum_w[a] += sum_w[b]
            count[a] += count[b]
            blocks.pop()

    fitted = np.zeros(n)
    for block in blocks:
        value = sum_wy[block] / sum_w[block]
        fitted[first[block]:first[block] + count[block]] = value
    return fitted


def trim_knots(x_knots, y_knots):
    """Drop the interior knots whose fitted value equals BOTH neighbours'.

    This is scikit-learn's `trim_duplicates`. It does not change the
    interpolated function -- the removed point lies on the segment between
    its neighbours -- but it DOES change the knot list.
    """
    n = len(y_knots)
    if n <= 2:
        return x_knots, y_knots
    keep = np.ones(n, dtype=bool)
    keep[1:-1] = ((y_knots[1:-1] != y_knots[:-2]) |
                  (y_knots[1:-1] != y_knots[2:]))
    return x_knots[keep], y_knots[keep]


def fit_isotonic(x, y, w=None, increasing=True):
    """The whole fit: collapse ties, PAVA, trim.

    Returns (x_knots, y_knots) -- the RETAINED knots.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)

    # Zero-weight points are DROPPED (scikit-learn masks sample_weight > 0
    # before the fit). They must not contribute to the tie collapse, must
    # not shift the x range, and must not appear as a knot.
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]

    ux, uy, uw = collapse_ties(x, y, w)
    fitted = pava(uy, uw, increasing=increasing)
    return trim_knots(ux, fitted)


def predict_linear(x_knots, y_knots, t, out_of_bounds="clip"):
    """The prediction: LINEAR INTERPOLATION through the retained knots.

    out_of_bounds is one of "clip", "nan", "raise".
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x_knots = np.asarray(x_knots, dtype=float)
    y_knots = np.asarray(y_knots, dtype=float)

    x_min, x_max = x_knots[0], x_knots[-1]
    result = np.empty_like(t)

    if out_of_bounds == "raise":
        bad = (t < x_min) | (t > x_max)
        if np.any(bad):
            raise ValueError("out of bounds")
    elif out_of_bounds == "nan":
        result[:] = np.nan
    else:  # clip
        result[:] = 0.0

    inside = (t >= x_min) & (t <= x_max)
    if np.any(inside):
        result[inside] = np.interp(t[inside], x_knots, y_knots)
    if out_of_bounds == "clip":
        result[t < x_min] = y_knots[0]
        result[t > x_max] = y_knots[-1]
    return result


def isotonic_predict(x, y, t, w=None, increasing=True, out_of_bounds="clip"):
    xk, yk = fit_isotonic(x, y, w, increasing)
    return predict_linear(xk, yk, t, out_of_bounds)


# ===========================================================================
# SECTION 1 -- the tie-collapse rule
# ===========================================================================

def iso_bruteforce(v, w, increasing=True):
    """The EXACT L2 isotonic regression, by EXHAUSTIVE ENUMERATION of the
    consecutive-block partitions.

    This is a COMPLETELY INDEPENDENT algorithm: it shares no code and no
    idea with PAVA. The L2 isotonic solution is always a step function
    whose blocks are CONSECUTIVE ranges of the sorted points, so enumerating
    every partition, keeping the monotone ones, and taking the smallest
    weighted SSE gives the exact optimum.

    Exponential, so it is only used on tiny fixtures -- which is exactly
    what makes it a good cross-check: it is obviously correct.
    """
    n = len(v)
    best_sse = np.inf
    best_z = None
    for cuts in _all_cuts(n - 1):
        bounds = [0] + [i + 1 for i, c in enumerate(cuts) if c] + [n]
        vals = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            vals.append(float(np.sum(w[a:b] * v[a:b]) / np.sum(w[a:b])))
        ok = True
        for i in range(len(vals) - 1):
            if increasing and vals[i] > vals[i + 1] + 1e-15:
                ok = False
                break
            if (not increasing) and vals[i] < vals[i + 1] - 1e-15:
                ok = False
                break
        if not ok:
            continue
        sse = 0.0
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            sse += float(np.sum(w[a:b] * (v[a:b] - vals[k]) ** 2))
        if sse < best_sse - 1e-14:
            best_sse = sse
            best_z = np.zeros(n)
            k = 0
            for a, b in zip(bounds[:-1], bounds[1:]):
                best_z[a:b] = vals[k]
                k += 1
    return best_z


def _all_cuts(n):
    """Every 0/1 cut pattern of length n (2^n of them)."""
    if n <= 0:
        yield ()
        return
    for rest in _all_cuts(n - 1):
        yield (0,) + rest
        yield (1,) + rest


def section_exactness():
    print("\n=== 2b. PAVA vs THE EXACT OPTIMUM (brute force) ===")

    # This is the check that makes the hand-derived fixtures in section 2
    # SAFE: it is a completely independent algorithm, so it cannot share a
    # hand-derivation error with them.
    rng = np.random.default_rng(3)
    worst = 0.0
    cases = 0
    for _ in range(200):
        n = int(rng.integers(2, 9))
        v = rng.normal(size=n)
        w = rng.choice([1.0, 2.0, 3.0], n)
        for increasing in (True, False):
            z_pava = pava(v, w, increasing)
            z_exact = iso_bruteforce(v, w, increasing)
            worst = max(worst, float(np.max(np.abs(z_pava - z_exact))))
            cases += 1
    check("PAVA equals the EXACT optimum on %d randomised cases" % cases,
          worst < 1e-12, "max deviation %r" % worst)

    # The five-point fixture, checked against the brute force too.
    v = np.array([1.0, 3.0, 2.0, 4.0, 0.0])
    w = np.ones(5)
    check_close_vec("the five-point fixture matches the EXACT optimum",
                    pava(v, w), iso_bruteforce(v, w))
    check_close_vec("the five-point fixture matches the EXACT optimum (dec)",
                    pava(v, w, increasing=False),
                    iso_bruteforce(v, w, increasing=False))


def section_tie_collapse():
    print("\n=== 1. THE TIE-COLLAPSE RULE ===")

    # THE PINNED CASE. x=[1,1], y=[0,1], w=[3,1]:
    #   weighted mean = (3*0 + 1*1) / (3+1) = 0.25
    # The UNWEIGHTED mean would be 0.5. The SUM would be 1.0.
    xk, yk = fit_isotonic(np.array([1.0, 1.0]), np.array([0.0, 1.0]),
                           np.array([3.0, 1.0]))
    check("tie collapse uses the WEIGHTED mean (0.25, not 0.5)",
          abs(yk[0] - 0.25) < 1e-15, "got %r" % yk)

    xk, yk = fit_isotonic(np.array([1.0, 1.0]), np.array([0.0, 1.0]),
                           np.array([1.0, 9.0]))
    check("tie collapse weights the OTHER way (0.9)", abs(yk[0] - 0.9) < 1e-15,
          "got %r" % yk)

    # The weights are SUMMED, which the PAVA block value then divides by.
    # Check it directly through collapse_ties.
    ux, uy, uw = collapse_ties(np.array([1.0, 1.0, 2.0]),
                                np.array([0.0, 1.0, 0.5]),
                                np.array([1.0, 3.0, 2.0]))
    check_close_vec("collapse_ties unique x", ux, [1.0, 2.0])
    check_close_vec("collapse_ties weighted mean", uy, [0.75, 0.5])
    check_close_vec("collapse_ties SUMMED weight", uw, [4.0, 2.0])

    # Cross-check against scikit-learn's own _make_unique.
    from sklearn._isotonic import _make_unique
    order = np.lexsort((np.array([0.0, 1.0, 0.5]), np.array([1.0, 1.0, 2.0])))
    sux, suy, suw = _make_unique(np.array([1.0, 1.0, 2.0])[order],
                                 np.array([0.0, 1.0, 0.5])[order],
                                 np.array([1.0, 3.0, 2.0])[order])
    check_close_vec("_make_unique agrees on x", ux, sux)
    check_close_vec("_make_unique agrees on y", uy, suy)
    check_close_vec("_make_unique agrees on weight", uw, suw)

    # A foil: the UNWEIGHTED mean genuinely differs on this fixture.
    unweighted = 0.5
    check("the unweighted-mean foil DIFFERS from the weighted mean",
          abs(unweighted - 0.25) > 1e-12)


# ===========================================================================
# SECTION 2 -- PAVA itself
# ===========================================================================

def section_pava():
    print("\n=== 2. PAVA ===")

    # Hand fixture. y = [1, 3, 2] increasing -> 3 > 2 violates, so the last
    # two pool: mean (3+2)/2 = 2.5 -> [1, 2.5, 2.5].
    fitted = pava(np.array([1.0, 3.0, 2.0]), np.array([1.0, 1.0, 1.0]))
    check_close_vec("PAVA pools the adjacent violator (1,3,2 -> 1,2.5,2.5)",
                    fitted, [1.0, 2.5, 2.5])

    # Decreasing: y = [3, 1, 2] -> 1 < 2 violates -> pool -> [3, 1.5, 1.5].
    fitted = pava(np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]),
                  increasing=False)
    check_close_vec("PAVA decreasing pools the adjacent violator",
                    fitted, [3.0, 1.5, 1.5])

    # A weighted pool. THERE MUST BE A VIOLATION for a pool to happen at
    # all: v = [4, 0] with 4 > 0 pools the two points, and the pooled value
    # is the WEIGHTED mean (4*1 + 0*3) / 4 = 1.0.
    #   The unweighted mean of the two points would be 2.0.
    #   The "mean of the two block MEANS" shortcut would ALSO give 2.0.
    # (The first draft of this check used v = [0, 4], which is ALREADY
    # non-decreasing -- so NO pool happens and the fit is [0, 4]. The
    # fixture was wrong, not the implementation.)
    fitted = pava(np.array([4.0, 0.0]), np.array([1.0, 3.0]))
    check_close_vec("the pooled block value is the WEIGHTED mean (1, not 2)",
                    fitted, [1.0, 1.0])

    # A pool of UNEQUAL SIZE. v = [6, 0, 0]: 6 > 0 pools {6, 0} -> 3; then
    # 3 > 0 pools {3, 0} -> 2. The fitted value is 2.
    #   The "mean of the two block MEANS" shortcut would give (3+0)/2 = 1.5.
    fitted = pava(np.array([6.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    check_close_vec("the pool is over the points, not the means", fitted,
                    [2.0, 2.0, 2.0])

    # Non-adjacent pooling is WRONG: y = [3, 0, 3] increasing.
    #   Correct PAVA: 3 > 0 violates -> pool {3,0} -> 1.5 -> [1.5,1.5,3].
    #   A "pool everything violating" shortcut might merge non-adjacent.
    fitted = pava(np.array([3.0, 0.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    check_close_vec("PAVA pools ADJACENT violators only", fitted,
                    [1.5, 1.5, 3.0])

    # A longer fixture with a known answer. Derivation:
    #   v = [1, 3, 2, 4, 0]; pool {3,2} -> 2.5; pool {4,0} -> 2;
    #   2.5 > 2 so pool {2.5, 2} -> 2.25; 1 <= 2.25 so it stops.
    #   => [1, 2.25, 2.25, 2.25, 2.25].
    # (The first draft of this check asserted [1, 2, 2, 2, 2], which is the
    # result of POOLING {2.5, 2, 2} AS A PLAIN MEAN -- i.e. the very
    # mistake this section exists to catch. The derivation was redone.)
    y = np.array([1.0, 3.0, 2.0, 4.0, 0.0])
    fitted = pava(y, np.ones(5))
    check_close_vec("PAVA on a five-point fixture", fitted,
                    [1.0, 2.25, 2.25, 2.25, 2.25])

    # AND THE FIXTURE IS CHECKED AGAINST THE EXACT OPTIMUM, BELOW -- so a
    # hand-derivation error like the one above cannot survive.


# ===========================================================================
# SECTION 3 -- the STEP vs. LINEAR-INTERPOLATION distinction
# ===========================================================================

def section_step_vs_interpolation():
    print("\n=== 3. STEP vs LINEAR INTERPOLATION ===")

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])

    xk, yk = fit_isotonic(x, y)
    check_close_vec("the retained knots' x", xk, [1.0, 2.0, 4.0, 5.0])
    check_close_vec("the retained knots' y", yk, [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0])

    # THE DISCRIMINATING VALUE. predict(4.5):
    #   LINEAR interpolation between (4, 1/3) and (5, 1) -> 2/3
    #   the STEP function would give 1.0 (the block containing x=5).
    got = predict_linear(xk, yk, np.array([4.5]))[0]
    check_close("predict(4.5) is the LINEAR interpolation (2/3), not the step (1.0)",
                got, 2.0 / 3.0)

    # The step-function foil genuinely differs.
    step_value = 1.0
    check("the step-function foil DIFFERS from the interpolation",
          abs(step_value - 2.0 / 3.0) > 1e-12)

    # The whole curve, against scikit-learn.
    m = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    t = np.linspace(0.0, 6.0, 13)
    check_close_vec("the whole interpolation curve matches scikit-learn",
                    predict_linear(xk, yk, t), m.predict(t), tol=1e-12)

    # The interpolation is CONTINUOUS at a knot -- a step function is not.
    eps = 1e-9
    left = predict_linear(xk, yk, np.array([4.0 - eps]))[0]
    right = predict_linear(xk, yk, np.array([4.0 + eps]))[0]
    check("the interpolated map is CONTINUOUS at a knot",
          abs(left - right) < 1e-8, "left %r right %r" % (left, right))


# ===========================================================================
# SECTION 4 -- the trimming rule
# ===========================================================================

def section_trimming():
    print("\n=== 4. THE TRIMMING RULE ===")

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])

    xk, yk = fit_isotonic(x, y)
    check("interior knots equal to BOTH neighbours are dropped (5 -> 4)",
          len(xk) == 4, "got %d" % len(xk))

    # The untrimmed list has 5 entries; the trimmed has 4.
    ux, uy, uw = collapse_ties(x, y, np.ones(5))
    untrimmed = pava(uy, uw)
    check("the UNTRIMMED fit has all five knots", len(untrimmed) == 5)
    check("the trimming removes exactly one knot here",
          len(untrimmed) - len(xk) == 1)

    # The trimming does NOT change the interpolated function.
    t = np.linspace(-2.0, 8.0, 41)
    trimmed_curve = predict_linear(xk, yk, t)
    untrimmed_curve = predict_linear(ux, untrimmed, t)
    check_close_vec("trimming does not change the interpolated function",
                    trimmed_curve, untrimmed_curve, tol=1e-12)

    # A case where NOTHING is trimmed (all distinct fitted values).
    x2 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([0.0, 0.5, 1.0])
    xk2, yk2 = fit_isotonic(x2, y2)
    check("a strictly increasing fit keeps every knot", len(xk2) == 3)

    # Cross-check the knot list against scikit-learn.
    m = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    check_close_vec("the knot list matches scikit-learn's X_thresholds_",
                    xk, m.X_thresholds_)
    check_close_vec("the knot list matches scikit-learn's y_thresholds_",
                    yk, m.y_thresholds_)


# ===========================================================================
# SECTION 5 -- the out_of_bounds regimes
# ===========================================================================

def section_out_of_bounds():
    print("\n=== 5. OUT_OF_BOUNDS ===")

    x = np.array([1.0, 5.0, 9.0])
    y = np.array([0.0, 0.5, 1.0])
    xk, yk = fit_isotonic(x, y)

    got = predict_linear(xk, yk, np.array([-100.0, 100.0]), "clip")
    check_close_vec("clip extends the END VALUES", got, [0.0, 1.0])

    got_nan = predict_linear(xk, yk, np.array([-100.0, 100.0]), "nan")
    check("nan returns NaN outside the range", bool(np.all(np.isnan(got_nan))))

    raised = False
    try:
        predict_linear(xk, yk, np.array([-100.0]), "raise")
    except ValueError:
        raised = True
    check("raise throws outside the range", raised)

    # Clip does NOT return the MEAN of the end values -- it returns the
    # END VALUE of the nearer knot. On this fixture the end values are 0
    # and 1, so the mean is 0.5 and the clip is 0.0 / 1.0: a genuine
    # discriminator.
    mean_of_ends = 0.5 * (yk[0] + yk[-1])
    check_close("clip returns the END VALUE, not the mean of the ends",
                got[0], yk[0])
    check("the mean-of-ends foil DIFFERS from the clip value",
          abs(mean_of_ends - got[0]) > 1e-12,
          "mean of ends %r vs clip %r" % (mean_of_ends, got[0]))

    # The exact clip semantics against scikit-learn.
    m = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    check_close_vec("clip matches scikit-learn's clip",
                    predict_linear(xk, yk, np.array([-100.0, 0.0, 100.0]), "clip"),
                    m.predict(np.array([-100.0, 0.0, 100.0])), tol=1e-12)

    # The SINGLE-KNOT case: one unique x. The map is constant everywhere.
    xk1, yk1 = fit_isotonic(np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.5, 1.0]))
    check("a single unique x gives ONE knot", len(xk1) == 1)
    check_close("the single knot's value is the weighted mean",
                yk1[0], 0.5)
    got = predict_linear(xk1, yk1, np.array([0.0, 2.0, 4.0]), "clip")
    check_close_vec("the single-knot map is constant everywhere", got,
                    [0.5, 0.5, 0.5])

    # The single-knot case with "nan": scikit-learn's own IsotonicRegression
    # still returns the constant (its f_ is a lambda, not an interp1d).
    m1 = IsotonicRegression(out_of_bounds="nan").fit(np.array([2.0]), np.array([0.5]))
    check("scikit-learn's single-knot 'nan' still returns the constant",
          bool(np.allclose(m1.predict(np.array([-5.0, 2.0, 9.0])), 0.5)))


# ===========================================================================
# SECTION 6 -- the CALIBRATION MAP
# ===========================================================================

def section_calibration_map():
    print("\n=== 6. THE CALIBRATION MAP ===")

    rng = np.random.default_rng(20261110)
    scores = rng.normal(size=400)
    labels = (rng.random(400) < 1.0 / (1.0 + np.exp(-1.5 * scores))).astype(float)

    calibrated = isotonic_predict(scores, labels, scores)

    # THE MAP IS NON-DECREASING **AS A FUNCTION OF THE SCORE**. The
    # calibrated values come back in the INPUT row order, which is NOT
    # sorted, so the monotonicity must be checked after sorting by score.
    # (The first draft of this check diffed the UNSORTED values and failed
    # for exactly that reason -- a fixture bug, not an implementation bug.)
    order = np.argsort(scores)
    sorted_calibrated = calibrated[order]
    check("the calibrated map is NON-DECREASING in the score",
          bool(np.all(np.diff(sorted_calibrated) >= -1e-15)))

    check("the calibrated probabilities lie in [0, 1]",
          bool(calibrated.min() >= -1e-15 and calibrated.max() <= 1.0 + 1e-15))

    # THE KEY PROPERTY: the calibration map must NOT INVERT the ranking of
    # the scores. That is what makes it safe to compose with the shipped
    # conformal prediction, whose guarantees are rank-based.
    #
    # NOTE: the map is non-decreasing, NOT strictly increasing -- it POOLS,
    # so distinct scores routinely share a calibrated value. Spearman's rho
    # is therefore < 1 by construction, and asserting rho == 1 would be
    # asserting a falsehood about the method. (The first draft of this
    # check asserted rho == 1 and failed with rho = 0.9946; the measured
    # value is CORRECT and the expectation was wrong.)
    inversions = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] < scores[j] and calibrated[i] > calibrated[j] + 1e-15:
                inversions += 1
    check("the calibration map NEVER INVERTS the score ranking",
          inversions == 0, "%d inversions" % inversions)

    # The ties are the POINT: pooling is what isotonic regression does, so
    # the calibrated values are COARSER than the scores.
    n_distinct_cal = len(np.unique(calibrated))
    check("the calibration map COARSENS the score (pooling creates ties)",
          n_distinct_cal < len(np.unique(scores)),
          "%d distinct calibrated vs %d distinct scores" %
          (n_distinct_cal, len(np.unique(scores))))

    # A foil: a map that scrambled the ranking WOULD invert it.
    scrambled = rng.permutation(calibrated)
    foil_inversions = sum(
        1 for i in range(len(scores)) for j in range(i + 1, len(scores))
        if scores[i] < scores[j] and scrambled[i] > scrambled[j] + 1e-15)
    check("the scrambled-ranking foil DOES invert the ranking",
          foil_inversions > 0)

    # The map is a FUNCTION of the score: equal scores -> equal outputs.
    dup_scores = np.array([0.5, 0.5, 0.5, 1.5])
    dup_labels = np.array([0.0, 1.0, 0.0, 1.0])
    out = isotonic_predict(dup_scores, dup_labels, dup_scores)
    check("equal scores give equal calibrated outputs",
          abs(out[0] - out[1]) < 1e-15 and abs(out[1] - out[2]) < 1e-15)

    # The calibrated probabilities should track the observed frequency.
    # Binned by decile of the score, the mean calibrated probability must
    # track the observed positive rate better than the RAW score does.
    order = np.argsort(scores)
    n_bins = 10
    bin_edges = np.linspace(0, len(scores), n_bins + 1).astype(int)
    raw_gap, cal_gap = [], []
    for b in range(n_bins):
        idx = order[bin_edges[b]:bin_edges[b + 1]]
        if len(idx) == 0:
            continue
        rate = labels[idx].mean()
        raw_gap.append(abs(scores[idx].mean() - rate))
        cal_gap.append(abs(calibrated[idx].mean() - rate))
    check("the calibrated probabilities track the observed rate better "
          "than the raw score does",
          np.mean(cal_gap) < np.mean(raw_gap),
          "raw %r calibrated %r" % (np.mean(raw_gap), np.mean(cal_gap)))


# ===========================================================================
# SECTION 7 -- the REPORT (cross-checked against scikit-learn's metrics)
# ===========================================================================

def section_report():
    print("\n=== 7. THE REPORT ===")

    rng = np.random.default_rng(7)
    n = 300
    labels = (rng.random(n) < 0.5).astype(float)
    # A DELIBERATELY OVERCONFIDENT score: too sharp, so the raw
    # probabilities are badly calibrated and the isotonic map has work to do.
    scores = np.clip(0.5 + 0.45 * (labels - 0.5) + 0.12 * rng.normal(size=n),
                     0.0, 1.0)

    calibrated = isotonic_predict(scores, labels, scores)

    n_bins = 10
    # ECE, computed with scikit-learn's OWN calibration_curve so the oracle
    # is not self-referential.
    #
    # THE ECE DEFINITION IS COUNT-WEIGHTED:
    #     ece = sum_k (count_k / n) * |prob_true_k - prob_pred_k|
    # which is exactly what this repo's utils/metrics.h computes. (The first
    # draft of this check divided the UNWEIGHTED per-bin sum by n, which is
    # a different number -- 0.00808 against the true 0.26363 on this
    # fixture -- and disagreed with the oracle's own correct computation.)
    sk_true, sk_pred = sk_calibration_curve(labels, scores, n_bins=n_bins,
                                            strategy="uniform")

    # The bin counts, from scikit-learn's own bin assignment.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(edges[1:-1], scores)
    counts = np.bincount(binids, minlength=n_bins)
    nonzero = counts != 0
    sk_ece = float(np.sum((counts[nonzero] / len(labels)) *
                          np.abs(sk_true - sk_pred)))

    ca_true, ca_pred = sk_calibration_curve(labels, calibrated, n_bins=n_bins,
                                            strategy="uniform")
    ca_binids = np.searchsorted(edges[1:-1], calibrated)
    ca_counts = np.bincount(ca_binids, minlength=n_bins)
    ca_nonzero = ca_counts != 0
    ca_ece = float(np.sum((ca_counts[ca_nonzero] / len(labels)) *
                          np.abs(ca_true - ca_pred)))

    check("the ECE of the RAW scores is measurable",
          0.0 <= sk_ece <= 1.0, "got %r" % sk_ece)
    check("the isotonic calibration REDUCES the ECE",
          ca_ece < sk_ece, "before %r after %r" % (sk_ece, ca_ece))

    # Brier, with scikit-learn's own function.
    sk_brier_before = sk_brier(labels, scores)
    sk_brier_after = sk_brier(labels, calibrated)
    check("the isotonic calibration REDUCES the Brier score",
          sk_brier_after < sk_brier_before,
          "before %r after %r" % (sk_brier_before, sk_brier_after))

    # Cross-check the oracle's own ECE computation against scikit-learn's.
    def own_ece(y_true, y_prob, n_bins):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        interior = edges[1:-1]
        idx = np.searchsorted(interior, y_prob, side="left")
        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            sel = idx == b
            if not np.any(sel):
                continue
            ece += (np.sum(sel) / n) * abs(y_true[sel].mean() - y_prob[sel].mean())
        return ece

    check_close("the oracle's own ECE matches scikit-learn's",
                own_ece(labels, scores, n_bins), sk_ece, tol=1e-12)

    # The report's counts.
    xk, yk = fit_isotonic(scores, labels)
    check("the report's knot count is the retained knot count", len(xk) > 1)


# ===========================================================================
# SECTION 8 -- the CROSS-CHECK against scikit-learn
# ===========================================================================

def section_cross_check():
    print("\n=== 8. THE CROSS-CHECK AGAINST SCIKIT-LEARN ===")

    rng = np.random.default_rng(12345)
    mismatches = 0
    cases = 0

    for _ in range(600):
        n = int(rng.integers(1, 13))
        # HEAVY TIES: few distinct x values over many points.
        x = rng.integers(0, 6, n).astype(float)
        y = rng.normal(size=n)
        w = rng.choice([0.0, 1.0, 2.0, 3.0], n).astype(float)
        if not np.any(w > 0):
            continue
        for increasing in (True, False):
            cases += 1
            m = IsotonicRegression(out_of_bounds="clip",
                                   increasing=increasing).fit(x, y, sample_weight=w)
            xk, yk = fit_isotonic(x, y, w, increasing=increasing)
            if (len(xk) != len(m.X_thresholds_) or
                    not np.allclose(yk, m.y_thresholds_, atol=1e-12) or
                    not np.allclose(xk, m.X_thresholds_, atol=1e-12)):
                mismatches += 1
                if mismatches <= 3:
                    print("      MISMATCH x=%r y=%r w=%r increasing=%r" %
                          (x, y, w, increasing))
                    print("        mine  x=%r y=%r" % (xk, yk))
                    print("        sklearn x=%r y=%r" % (m.X_thresholds_, m.y_thresholds_))

    check("the PAVA implementation agrees with scikit-learn on %d randomised "
          "tie-heavy weighted cases" % cases,
          mismatches == 0, "%d mismatches" % mismatches)

    # And the PREDICTIONS agree, too -- not just the knots.
    rng = np.random.default_rng(999)
    pred_mismatches = 0
    pred_cases = 0
    for _ in range(300):
        n = int(rng.integers(2, 15))
        x = rng.integers(0, 7, n).astype(float)
        y = rng.normal(size=n)
        w = rng.choice([1.0, 2.0, 3.0], n).astype(float)
        t = rng.uniform(-3.0, 9.0, 11)
        m = IsotonicRegression(out_of_bounds="clip").fit(x, y, sample_weight=w)
        xk, yk = fit_isotonic(x, y, w)
        pred_cases += 1
        if not np.allclose(predict_linear(xk, yk, t, "clip"), m.predict(t),
                           atol=1e-12):
            pred_mismatches += 1
    check("the PREDICTIONS agree with scikit-learn on %d randomised cases" %
          pred_cases,
          pred_mismatches == 0, "%d mismatches" % pred_mismatches)

    # Permutation invariance: the fit must not depend on the row order.
    rng = np.random.default_rng(555)
    x = rng.integers(0, 5, 20).astype(float)
    y = rng.normal(size=20)
    xk1, yk1 = fit_isotonic(x, y)
    perm = rng.permutation(20)
    xk2, yk2 = fit_isotonic(x[perm], y[perm])
    check("the fit is INVARIANT to the row order",
          np.allclose(xk1, xk2) and np.allclose(yk1, yk2))


# ===========================================================================
# SECTION 9 -- the PINNED FIXTURE, emitted as C++ literals
# ===========================================================================

def section_pinned_fixture():
    print("\n=== 9. THE PINNED FIXTURE ===")

    # The discriminating fixture: heavy ties, a pool, and a knot that gets
    # trimmed. Chosen so that EVERY rule is exercised at once.
    x = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    w = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0])

    xk, yk = fit_isotonic(x, y, w)
    queries = np.array([-0.5, 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.5])
    preds = predict_linear(xk, yk, queries, "clip")

    m = IsotonicRegression(out_of_bounds="clip").fit(x, y, sample_weight=w)
    check("the pinned fixture's knots match scikit-learn",
          np.allclose(xk, m.X_thresholds_) and np.allclose(yk, m.y_thresholds_))
    check("the pinned fixture's predictions match scikit-learn",
          np.allclose(preds, m.predict(queries)))

    # THE PINNED FIXTURE MUST DISCRIMINATE THE LINEAR INTERPOLATION FROM
    # THE STEP FUNCTION. If it does not, a C++ suite built on it could not
    # catch a step-function implementation -- a fixture that cannot
    # discriminate cannot test anything.
    ux, uy, uw = collapse_ties(x, y, w)
    step_values = pava(uy, uw)
    discriminating = []
    for q in queries:
        linear_value = predict_linear(xk, yk, np.array([q]))[0]
        i = int(np.searchsorted(ux, q, side="right")) - 1
        i = max(0, min(i, len(ux) - 1))
        step_value = step_values[i]
        if abs(linear_value - step_value) > 1e-12:
            discriminating.append((float(q), float(linear_value),
                                   float(step_value)))
    check("the pinned fixture DISCRIMINATES the linear interpolation from "
          "the step function", len(discriminating) > 0)

    print("")
    print("    // ---- pinned fixture (transferred from the oracle) ----")
    print("    const std::vector<double> kX = {%s};" %
          ", ".join(repr(float(v)) for v in x))
    print("    const std::vector<double> kY = {%s};" %
          ", ".join(repr(float(v)) for v in y))
    print("    const std::vector<double> kW = {%s};" %
          ", ".join(repr(float(v)) for v in w))
    print("    const std::vector<double> kKnotX = {%s};" %
          ", ".join(repr(float(v)) for v in xk))
    print("    const std::vector<double> kKnotY = {%s};" %
          ", ".join(repr(float(v)) for v in yk))
    print("    const std::vector<double> kQueries = {%s};" %
          ", ".join(repr(float(v)) for v in queries))
    print("    const std::vector<double> kExpected = {%s};" %
          ", ".join(repr(float(v)) for v in preds))
    print("    // ---- end pinned fixture ----")
    print("")
    print("    // The queries at which the LINEAR value differs from the STEP")
    print("    // value (query, linear, step):")
    for q, lin, stp in discriminating:
        print("    //   %r -> linear %r, step %r" % (q, lin, stp))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 72)
    print("tests/isotonic_oracle.py -- the float64 numpy specification")
    print("=" * 72)
    print("numpy %s | sklearn %s" % (np.__version__,
                                     __import__("sklearn").__version__))

    section_tie_collapse()
    section_pava()
    section_exactness()
    section_step_vs_interpolation()
    section_trimming()
    section_out_of_bounds()
    section_calibration_map()
    section_report()
    section_cross_check()
    section_pinned_fixture()

    print("")
    print("=== SUMMARY: %d passed, %d failed ===" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
