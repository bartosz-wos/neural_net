#!/usr/bin/env python3
# =============================================================================
# tests/dtw_oracle.py -- THE SPECIFICATION for include/nn/utils/dtw.{h,cpp}.
#
# THIS FILE IS THE DEFINITION. It was written and run BEFORE the C++
# implementation existed, and it is the only place the expected numbers come
# from. The C++ test consumes the fixtures this file emits; nothing is
# hand-transcribed.
#
# -----------------------------------------------------------------------------
# WHY THE ORACLE IS NOT "THE DP, AGAIN"
# -----------------------------------------------------------------------------
#
# A differential test that compares an implementation against the SAME
# recurrence written twice catches transcription errors and nothing else.
#
# So this oracle has THREE levels, each independent of the implementation:
#
#   LEVEL 1  EXHAUSTIVE MONOTONE-PATH ENUMERATION. Every path from (0,0) to
#            (n-1,m-1) through the warping lattice, with steps (1,0), (0,1)
#            and (1,1), is enumerated and the cheapest is taken. This is the
#            DEFINITION of the warping problem rather than an algorithm for
#            it, and it shares no structure with the dynamic program.
#
#   LEVEL 2  THE BANDED LATTICE, enumerated the same way. The band is part
#            of the enumeration, not a filter applied afterwards.
#
#   LEVEL 3  THE MAINTAINED REFERENCE. `dtw-python` is installed locally and
#            is used as an independent check on the oracle itself. If the
#            oracle and the reference disagree, the ORACLE is the thing under
#            suspicion -- it is code written today.
#
# -----------------------------------------------------------------------------
# THE SEVEN SEMANTICS THAT ARE EASY TO GET WRONG
# -----------------------------------------------------------------------------
#
#  1. THE STEP PATTERN IS symmetric1 -- the THREE-neighbour lattice
#     (diagonal, vertical, horizontal). dtw-python's DEFAULT step pattern is
#     symmetric2, which is a DIFFERENT function: measured to differ on
#     200/200 random pairs. Every reference call therefore passes
#     `step_pattern=dtw.symmetric1` EXPLICITLY.
#
#     And symmetric1 is genuinely three neighbours, not a pattern with
#     two-step jumps: measured, x=[0,100] against y=[0] costs 100.0 under
#     dtw-python's symmetric1 and under the three-neighbour DP -- a
#     jump-capable pattern would have cost 0.
#
#  2. THE LOCAL COST IS EUCLIDEAN, over the ROW VECTOR of each observation.
#     dtw-python's `cdist` is applied to the rows, so a (T,D) tensor is T
#     observations of D dimensions each -- NOT D observations of T
#     dimensions. Measured to agree on 200/200 random multidimensional cases.
#
#  3. THE DISTANCE IS RAW AND CUMULATIVE. It is NOT normalised by the path
#     length and NOT normalised by (n+m). dtw-python's `distance` attribute
#     is the raw cumulative cost and is what this module ships.
#
#  4. THE BAND IS |i - j| <= band ON TIME INDICES. The offset does not
#     matter -- 1-based cell indices and 0-based data indices give the same
#     answer, because both indices shift together.
#
#  5. A BAND THAT ADMITS NO PATH IS AN INFEASIBLE REQUEST, NOT A LARGE
#     DISTANCE. A monotone path exists iff band >= |n - m| -- measured on
#     3000 random cases, 0 violations. dtw-python raises ValueError in that
#     situation; this module returns +inf and says so in its header.
#
#  6. THE WARPPING PATH IS NOT UNIQUE. Ties in the dynamic program admit
#     several optimal paths. The shipped tie-break is measured against
#     dtw-python's own `index1`/`index2` on tie-heavy INTEGER observations
#     (continuous observations almost never tie, so they cannot discriminate).
#
#  7. DTW IS SYMMETRIC BUT NOT A METRIC. Measured: 0/300 asymmetric pairs,
#     but 7/300 triangle-inequality violations. So the module must not call
#     DTW a metric.
#
# -----------------------------------------------------------------------------
# THE ORACLE'S OWN SENSITIVITY IS MEASURED, NOT NARRATED
# -----------------------------------------------------------------------------
#
# tests/dtw_mutate_oracle.py breaks this oracle in specific ways and requires
# that it fail each time. A green oracle on its first run would prove nothing
# at all.
#
# =============================================================================

import itertools
import random
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("FATAL: numpy is required by the dtw oracle")
try:
    import dtw
except ImportError:  # pragma: no cover
    sys.exit("FATAL: dtw-python is required by the dtw oracle")


# =============================================================================
# LEVEL 1 -- EXHAUSTIVE MONOTONE-PATH ENUMERATION
# =============================================================================

def local_cost(x, y):
    """Euclidean distance between two observation vectors."""
    return float(np.linalg.norm(np.asarray(x, dtype=float)
                                - np.asarray(y, dtype=float)))


def enumerate_paths(n, m, band=None):
    """Every monotone path from (0,0) to (n-1,m-1) through the warping
    lattice, as a list of (i,j) pairs.

    The steps are (1,0) -- advance in x only, (0,1) -- advance in y only,
    and (1,1) -- advance in both. These are the three neighbours of the
    symmetric1 lattice.

    When `band` is given, a cell (i,j) is FORBIDDEN if |i-j| > band, and the
    band is part of the enumeration rather than a filter applied afterwards.
    """
    if n == 0 or m == 0:
        return []
    paths = []

    def rec(i, j, acc):
        if i == n - 1 and j == m - 1:
            paths.append(tuple(acc))
            return
        for (di, dj) in ((1, 0), (0, 1), (1, 1)):
            ni, nj = i + di, j + dj
            if ni > n - 1 or nj > m - 1:
                continue
            if band is not None and abs(ni - nj) > band:
                continue
            rec(ni, nj, acc + [(ni, nj)])

    rec(0, 0, [(0, 0)])
    return paths


def brute_force_dtw(X, Y, band=None):
    """The DTW distance by exhaustive enumeration of every monotone path."""
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return float("inf")
    best = None
    for path in enumerate_paths(n, m, band):
        cost = 0.0
        for (i, j) in path:
            cost += local_cost(X[i], Y[j])
        if best is None or cost < best:
            best = cost
    return best if best is not None else float("inf")


def dp_dtw(X, Y, band=None):
    """The dynamic program the C++ ships: symmetric1, Euclidean local cost,
    raw cumulative distance.

    D[0][0] = 0
    D[i][j] = min(D[i-1][j-1], D[i-1][j], D[i][j-1]) + local(x[i-1], y[j-1])
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return float("inf")
    inf = float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i - j) > band:
                continue
            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \
                + local_cost(X[i - 1], Y[j - 1])
    return D[n][m]


def dp_dtw_path(X, Y, band=None, order=("d", "l", "u")):
    """The warping path, by backtracking the dynamic program.

    The predecessor preference is `order`, over the codes
        'd' -- the diagonal predecessor (i-1, j-1),
        'l' -- the horizontal predecessor (i, j-1),
        'u' -- the vertical predecessor (i-1, j).

    The cells are returned as 0-BASED DATA INDICES: the first is (0,0) and
    the last is (n-1, m-1). This is exactly dtw-python's `index1`/`index2`
    convention, so the two can be compared directly.
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return []
    inf = float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i - j) > band:
                continue
            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \
                + local_cost(X[i - 1], Y[j - 1])
    cells = []
    i, j = n, m
    while i > 0 and j > 0:
        c = local_cost(X[i - 1], Y[j - 1])
        moved = False
        for p in order:
            if p == "d" and D[i][j] == D[i - 1][j - 1] + c:
                cells.append((i - 1, j - 1))
                i -= 1
                j -= 1
                moved = True
                break
            if p == "l" and D[i][j] == D[i][j - 1] + c:
                cells.append((i - 1, j - 1))
                j -= 1
                moved = True
                break
            if p == "u" and D[i][j] == D[i - 1][j] + c:
                cells.append((i - 1, j - 1))
                i -= 1
                moved = True
                break
        if not moved:
            # A backtracker that cannot move is BROKEN. It is reported as an
            # empty path -- which every check below then rejects -- rather
            # than raised, because a raised exception CRASHES the oracle and
            # a crash is classified MUT-BROKEN rather than CAUGHT by the
            # mutation harness. The failure must be REPORTED to be a verdict.
            return []
    while i > 0:
        cells.append((i - 1, 0))
        i -= 1
    while j > 0:
        cells.append((0, j - 1))
        j -= 1
    # The origin cell (0,0) is ALREADY in `cells` -- it is the first cell the
    # backtrack appends when it reaches (1,1). Appending it again here would
    # duplicate it, and the duplicated origin is what made an earlier probe
    # report 0/300 for EVERY tie-break order (a harness bug, not a property
    # of dtw-python).
    cells.reverse()
    return cells


def path_cost(X, Y, path):
    """The accumulated local cost along a warping path."""
    return sum(local_cost(X[i], Y[j]) for (i, j) in path)


# =============================================================================
# THE CHECKS
# =============================================================================

def check(label, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        check.failures += 1
    return ok


check.failures = 0


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def ref_dtw(X, Y, band=None):
    """dtw-python with the step pattern pinned EXPLICITLY.

    dtw-python's DEFAULT step pattern is symmetric2, which is a different
    function -- so passing it explicitly is not optional.
    """
    kwargs = {"step_pattern": dtw.symmetric1}
    if band is not None:
        kwargs["window_type"] = dtw.sakoeChibaWindow
        kwargs["window_args"] = {"window_size": band}
    try:
        return float(dtw.dtw(np.asarray(X, dtype=float),
                             np.asarray(Y, dtype=float), **kwargs).distance)
    except ValueError:
        # No warping path compatible with the local constraints.
        return float("inf")


def main():
    random.seed(20261112)

    # -------------------------------------------------------------------------
    section("SECTION 1 -- the exhaustive path enumeration agrees with the DP")

    n_ok = n_tot = 0
    for _ in range(150):
        n = random.randint(1, 6)
        m = random.randint(1, 6)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        n_tot += 1
        if abs(brute_force_dtw(X, Y) - dp_dtw(X, Y)) < 1e-12:
            n_ok += 1
    check("exhaustive monotone-path enumeration == the DP (univariate)",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The same, in TWO dimensions, where the local cost is the Euclidean
    # distance between the ROW VECTORS.
    n_ok = n_tot = 0
    for _ in range(100):
        n = random.randint(1, 5)
        m = random.randint(1, 5)
        d = random.randint(1, 3)
        X = [[random.uniform(-3, 3) for _ in range(d)] for _ in range(n)]
        Y = [[random.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
        n_tot += 1
        if abs(brute_force_dtw(X, Y) - dp_dtw(X, Y)) < 1e-12:
            n_ok += 1
    check("exhaustive monotone-path enumeration == the DP (multivariate)",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # -------------------------------------------------------------------------
    section("SECTION 2 -- the DP agrees with dtw-python, with the step "
            "pattern pinned EXPLICITLY")

    n_ok = n_tot = 0
    for _ in range(300):
        n = random.randint(1, 7)
        m = random.randint(1, 7)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        n_tot += 1
        if abs(ref_dtw(X, Y) - dp_dtw(X, Y)) < 1e-9:
            n_ok += 1
    check("the DP == dtw-python(symmetric1) (univariate)",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    n_ok = n_tot = 0
    for _ in range(200):
        n = random.randint(1, 6)
        m = random.randint(1, 6)
        d = random.randint(1, 4)
        X = [[random.uniform(-3, 3) for _ in range(d)] for _ in range(n)]
        Y = [[random.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
        n_tot += 1
        if abs(ref_dtw(X, Y) - dp_dtw(X, Y)) < 1e-9:
            n_ok += 1
    check("the DP == dtw-python(symmetric1) (multivariate, rows are "
          "observations)", n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The step pattern really is symmetric1 and NOT the default.
    diff = 0
    for _ in range(100):
        n = random.randint(2, 6)
        m = random.randint(2, 6)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        s1 = float(dtw.dtw(np.asarray(X), np.asarray(Y),
                           step_pattern=dtw.symmetric1).distance)
        s2 = float(dtw.dtw(np.asarray(X), np.asarray(Y),
                           step_pattern=dtw.symmetric2).distance)
        if abs(s1 - s2) > 1e-12:
            diff += 1
    check("symmetric1 and symmetric2 genuinely DIFFER (so the default must "
          "be overridden)", diff >= 90, f"{diff}/100 differ")

    # And symmetric1 is the THREE-neighbour lattice, not a jump-capable one.
    check("symmetric1 is the three-neighbour lattice: x=[0,100] vs y=[0] "
          "costs 100",
          abs(ref_dtw([0.0, 100.0], [0.0]) - 100.0) < 1e-12
          and abs(dp_dtw([0.0, 100.0], [0.0]) - 100.0) < 1e-12)

    # -------------------------------------------------------------------------
    section("SECTION 3 -- the band")

    # The band is |i-j| <= band on time indices.
    n_ok = n_tot = 0
    for _ in range(300):
        n = random.randint(1, 8)
        m = random.randint(1, 8)
        band = random.randint(0, 10)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        n_tot += 1
        mine = dp_dtw(X, Y, band)
        theirs = ref_dtw(X, Y, band)
        if (mine == float("inf")) == (theirs == float("inf")) and \
                (mine == float("inf") or abs(mine - theirs) < 1e-9):
            n_ok += 1
    check("the banded DP == dtw-python(banded)", n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The banded DP == the banded exhaustive enumeration.
    n_ok = n_tot = 0
    for _ in range(150):
        n = random.randint(1, 5)
        m = random.randint(1, 5)
        band = random.randint(abs(n - m), abs(n - m) + 4)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        n_tot += 1
        if abs(brute_force_dtw(X, Y, band) - dp_dtw(X, Y, band)) < 1e-12:
            n_ok += 1
    check("the banded DP == the banded exhaustive enumeration",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # Feasibility: a monotone path exists iff band >= |n - m|.
    n_ok = n_tot = 0
    for _ in range(3000):
        n = random.randint(1, 9)
        m = random.randint(1, 9)
        band = random.randint(0, 10)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        feasible = dp_dtw(X, Y, band) != float("inf")
        n_tot += 1
        if feasible == (band >= abs(n - m)):
            n_ok += 1
    check("a monotone path exists iff band >= |n - m|",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The band actually CONSTRAINS -- it must not be a no-op.
    diff = 0
    for _ in range(200):
        n = random.randint(2, 7)
        m = random.randint(2, 7)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        if abs(dp_dtw(X, Y, 0) - dp_dtw(X, Y, None)) > 1e-12:
            diff += 1
    check("the band CONSTRAINS the distance (band=0 differs from no band)",
          diff >= 150, f"{diff}/200 differ")

    # -------------------------------------------------------------------------
    section("SECTION 4 -- the warping path")

    # The path's cost equals the distance.
    n_ok = n_tot = 0
    for _ in range(200):
        n = random.randint(1, 7)
        m = random.randint(1, 7)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        p = dp_dtw_path(X, Y)
        n_tot += 1
        if abs(path_cost(X, Y, p) - dp_dtw(X, Y)) < 1e-12:
            n_ok += 1
    check("the path's accumulated cost == the distance",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The path is a legal monotone path: it starts at (0,0), ends at
    # (n-1,m-1), and every step is (1,0), (0,1) or (1,1).
    n_ok = n_tot = 0
    for _ in range(200):
        n = random.randint(1, 7)
        m = random.randint(1, 7)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        p = dp_dtw_path(X, Y)
        n_tot += 1
        legal = (p[0] == (0, 0) and p[-1] == (n - 1, m - 1)
                 and all((p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                         in ((1, 0), (0, 1), (1, 1)) for k in range(len(p) - 1)))
        if legal:
            n_ok += 1
    check("the path is a legal monotone path (start, end, and steps)",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The path is one of the OPTIMAL paths enumerated exhaustively.
    n_ok = n_tot = 0
    for _ in range(120):
        n = random.randint(1, 5)
        m = random.randint(1, 5)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        p = dp_dtw_path(X, Y)
        best = brute_force_dtw(X, Y)
        n_tot += 1
        if abs(path_cost(X, Y, p) - best) < 1e-12:
            n_ok += 1
    check("the path's cost == the exhaustively-enumerated optimum",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # A BROKEN backtracker must be REPORTED, not allowed to crash the
    # oracle: a crash is not a verdict, and the mutation harness needs to SEE
    # the failure. DM13 removed the horizontal and vertical fallbacks from the
    # backtracker, which made the oracle die with a RuntimeError instead of
    # reporting -- and a crash is classified MUT-BROKEN, not CAUGHT.
    broken = 0
    for _ in range(120):
        n = random.randint(1, 5)
        m = random.randint(1, 5)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        try:
            p = dp_dtw_path(X, Y)
            if abs(path_cost(X, Y, p) - dp_dtw(X, Y)) >= 1e-12:
                broken += 1
        except Exception:
            broken += 1
    check("the path backtracker always produces an optimal path (a broken "
          "backtracker is REPORTED, not raised)", broken == 0,
          f"{broken}/120 broken")

    # The TIE-BREAK. Continuous observations almost never tie, so they
    # cannot discriminate the candidate orders -- the tie-heavy INTEGER
    # sample is the discriminating one.
    print()
    print("  the tie-break, measured on tie-heavy INTEGER observations:")
    for order in (("d", "l", "u"), ("d", "u", "l"), ("u", "d", "l"),
                  ("l", "d", "u"), ("u", "l", "d"), ("l", "u", "d")):
        ok = tot = 0
        for _ in range(300):
            n = random.randint(2, 7)
            m = random.randint(2, 7)
            X = [float(random.randint(0, 4)) for _ in range(n)]
            Y = [float(random.randint(0, 4)) for _ in range(m)]
            lib = dtw.dtw(np.asarray(X), np.asarray(Y),
                          step_pattern=dtw.symmetric1)
            libp = list(zip(lib.index1.tolist(), lib.index2.tolist()))
            tot += 1
            if dp_dtw_path(X, Y, order=order) == libp:
                ok += 1
        print(f"    order={order}: path identical to dtw-python {ok}/{tot}")
        if order == ("d", "l", "u"):
            check("the shipped tie-break ('d','l','u') reproduces "
                  "dtw-python's path on tie-heavy observations",
                  ok == tot, f"{ok}/{tot}")
            shipped_order = order

    # The tie-break must be pinned by the DEFAULT, not only by an explicitly
    # passed order -- otherwise a mutation that changes the default is
    # invisible (DM14 survived for exactly this reason: the tie-break
    # measurement passed `order=` explicitly, so it never exercised the
    # default the C++ actually ships).
    n_ok = n_tot = 0
    for _ in range(300):
        n = random.randint(2, 7)
        m = random.randint(2, 7)
        X = [float(random.randint(0, 4)) for _ in range(n)]
        Y = [float(random.randint(0, 4)) for _ in range(m)]
        lib = dtw.dtw(np.asarray(X), np.asarray(Y),
                      step_pattern=dtw.symmetric1)
        libp = list(zip(lib.index1.tolist(), lib.index2.tolist()))
        n_tot += 1
        if dp_dtw_path(X, Y) == libp:
            n_ok += 1
    check("the DEFAULT tie-break reproduces dtw-python's path on tie-heavy "
          "observations (the default is what the C++ ships)",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # -------------------------------------------------------------------------
    section("SECTION 5 -- the metric properties, measured")

    asym = 0
    for _ in range(300):
        n = random.randint(1, 6)
        m = random.randint(1, 6)
        k = random.randint(1, 6)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        Z = [random.uniform(-3, 3) for _ in range(k)]
        if abs(dp_dtw(X, Y) - dp_dtw(Y, X)) > 1e-9:
            asym += 1
    check("DTW is SYMMETRIC", asym == 0, f"{asym}/300 asymmetric pairs")

    viol = 0
    example = None
    for _ in range(300):
        n = random.randint(1, 6)
        m = random.randint(1, 6)
        k = random.randint(1, 6)
        X = [random.uniform(-3, 3) for _ in range(n)]
        Y = [random.uniform(-3, 3) for _ in range(m)]
        Z = [random.uniform(-3, 3) for _ in range(k)]
        a, b, c = dp_dtw(X, Y), dp_dtw(X, Z), dp_dtw(Z, Y)
        if a > b + c + 1e-9:
            viol += 1
            if example is None:
                example = (a, b, c)
    check("DTW is NOT a metric: the triangle inequality is violated",
          viol > 0, f"{viol}/300 violations, example (dXY,dXZ,dZY)={example}")

    # -------------------------------------------------------------------------
    section("SECTION 6 -- the discriminating cases")

    # EVERY value below was MEASURED by this oracle, not hand-derived. The
    # first draft of this section asserted 2.0 for d([0,1,2],[0,2,4]) and for
    # d([0,1,2],[2,1,0]); the oracle measured 3.0 and 4.0. A hand-derived
    # expected value is exactly the failure mode this repo has hit before --
    # the number was CORRECT for the fixture it was written for, and the
    # fixture was simply not the oracle's.
    check("x=[0,1,2] vs y=[0,1,2]: distance 0",
          abs(dp_dtw([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]])) < 1e-12)
    check("x=[0,1,2] vs y=[0,2,4]: distance 3",
          abs(dp_dtw([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]]) - 3.0) < 1e-12)
    check("x=[0,1,2] vs y=[2,1,0]: distance 4 (the warping CANNOT undo a "
          "full reversal -- the path must stay monotone)",
          abs(dp_dtw([[0.0], [1.0], [2.0]], [[2.0], [1.0], [0.0]]) - 4.0) < 1e-12)
    check("x=[0,10] vs y=[0,0]: distance 10",
          abs(dp_dtw([[0.0], [10.0]], [[0.0], [0.0]]) - 10.0) < 1e-12)
    check("the band=0 distance of x=[0,1,2] vs y=[0,2,4] is 3 (the diagonal "
          "only)",
          abs(dp_dtw([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], band=0) - 3.0) < 1e-12)
    # THE POINT OF DTW: the same trajectory, sampled at different rates.
    # Here the two series are the same path, time-shifted, so the DTW
    # distance is ZERO while the plain L2 distance is 3.16.
    shifted_x = [[0.0], [0.0], [0.0], [1.0], [2.0], [3.0]]
    shifted_y = [[0.0], [1.0], [2.0], [3.0], [3.0], [3.0]]
    l2 = float(np.linalg.norm(np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
                              - np.array([0.0, 1.0, 2.0, 3.0, 3.0, 3.0])))
    check("the same trajectory sampled differently has DTW 0 but L2 3.16",
          abs(dp_dtw(shifted_x, shifted_y)) < 1e-12 and abs(l2 - 3.1622776601683795) < 1e-9,
          f"dtw={dp_dtw(shifted_x, shifted_y)} l2={l2}")

    # -------------------------------------------------------------------------
    section("SECTION 7 -- the pinned fixture")

    def cpp_tensor(name, arr):
        flat = np.asarray(arr, dtype=float).reshape(-1)
        body = ", ".join("{:.17g}".format(v) for v in flat)
        print(f"static const double {name}[] = {{{body}}};")

    print()
    print("// ==== generated by tests/dtw_oracle.py; do not hand-edit ====")
    print("struct DtwCase {")
    print("    int n;              // time steps in x")
    print("    int m;              // time steps in y")
    print("    int dim;            // dimensions per observation")
    print("    double dist;        // the DTW distance")
    print("    int n_path;         // number of cells in the warping path")
    print("    int i0, j0;         // the path's FIRST cell")
    print("    int ie, je;         // the path's LAST cell")
    print("};")
    print("static const DtwCase kDtwCases[] = {")

    # Each fixture is (x, y, dim): x and y are lists of OBSERVATIONS, and an
    # observation is a list of `dim` coordinates. A dim-1 observation is
    # written as a one-element list.
    dtw_fixtures = [
        ([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]], 1),
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], 1),
        ([[0.0], [1.0], [2.0]], [[2.0], [1.0], [0.0]], 1),
        ([[0.0], [10.0]], [[0.0], [0.0]], 1),
        ([[1.0]], [[1.0]], 1),
        ([[1.0]], [[2.0]], 1),
        ([[0.0], [1.0], [2.0], [3.0]], [[0.0], [3.0], [6.0]], 1),
        ([[0.0], [0.0]], [[0.0], [0.0]], 1),
        ([[3.0], [1.0], [2.0]], [[1.0], [2.0], [3.0]], 1),
        ([[0.0], [5.0], [0.0]], [[0.0], [0.0], [5.0]], 1),
        # TWO dimensions: each observation is a 2-D point.
        ([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]], 2),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], 2),
        ([[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]], 2),
        # THREE dimensions.
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
         [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], 3),
    ]
    for (Xr, Yr, dim) in dtw_fixtures:
        assert len(Xr[0]) == dim, (Xr, dim)
        assert len(Yr[0]) == dim, (Yr, dim)
        assert all(len(o) == dim for o in Xr), (Xr, dim)
        assert all(len(o) == dim for o in Yr), (Yr, dim)
        dist = dp_dtw(Xr, Yr)
        path = dp_dtw_path(Xr, Yr)
        if abs(path_cost(Xr, Yr, path) - dist) >= 1e-12:
            check("dtw fixture %r/%r: the path's cost equals the distance"
                  % (Xr, Yr), False)
        # cross-check against dtw-python
        theirs = ref_dtw(Xr, Yr)
        if abs(theirs - dist) > 1e-9:
            check("dtw fixture %r/%r: the DP matches dtw-python" % (Xr, Yr),
                  False, "ours=%r theirs=%r" % (dist, theirs))
        print(f"    {{{len(Xr)}, {len(Yr)}, {dim}, {dist!r}, {len(path)}, "
              f"{path[0][0]}, {path[0][1]}, {path[-1][0]}, {path[-1][1]}}},")
    print("};")
    print(f"static const int kDtwCaseCount = {len(dtw_fixtures)};")

    print()
    print("// ==== generated by tests/dtw_oracle.py; do not hand-edit ====")
    print("struct DtwBandCase {")
    print("    int n;")
    print("    int m;")
    print("    int band;")
    print("    double dist;   // +inf when the band admits no path")
    print("};")
    print("static const DtwBandCase kDtwBandCases[] = {")
    band_fixtures = [
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], 0),
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], 1),
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], 2),
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]], 3),
        ([[0.0], [1.0], [2.0], [3.0]], [[0.0], [3.0], [6.0]], 0),
        ([[0.0], [1.0], [2.0], [3.0]], [[0.0], [3.0], [6.0]], 1),
        ([[0.0], [1.0], [2.0], [3.0]], [[0.0], [3.0], [6.0]], 3),
        ([[0.0], [1.0]], [[0.0], [1.0], [2.0]], 0),
        ([[0.0], [1.0]], [[0.0], [1.0], [2.0]], 1),
        ([[0.0], [1.0], [2.0]], [[0.0], [1.0]], 0),
        ([[0.0], [1.0], [2.0]], [[0.0], [1.0]], 1),
        ([[0.0], [1.0]], [[0.0], [1.0], [2.0], [3.0]], 1),
        ([[0.0], [1.0]], [[0.0], [1.0], [2.0], [3.0]], 2),
        # a band that admits NO path: |n-m| = 3 but band = 1
        ([[0.0], [1.0]], [[0.0], [1.0], [2.0], [3.0], [4.0]], 1),
    ]
    for (Xr, Yr, band) in band_fixtures:
        dist = dp_dtw(Xr, Yr, band)
        # cross-check the banded value against dtw-python
        theirs = ref_dtw(Xr, Yr, band)
        if (dist == float("inf")) != (theirs == float("inf")) or \
                (dist != float("inf") and abs(dist - theirs) > 1e-9):
            check("banded fixture %r/%r band=%d matches dtw-python"
                  % (Xr, Yr, band), False, "ours=%r theirs=%r" % (dist, theirs))
        print(f"    {{{len(Xr)}, {len(Yr)}, {band}, {dist!r}}},")
    print("};")

    print()
    print("// ==== generated by tests/dtw_oracle.py; do not hand-edit ====")
    print("struct DtwPathCase {")
    print("    int n;")
    print("    int m;")
    print("    int n_path;")
    print("    int cells[32];   // flattened (i,j) pairs, n_path of them")
    print("};")
    print("static const DtwPathCase kDtwPathCases[] = {")
    path_fixtures = [
        ([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]]),
        ([[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]]),
        ([[0.0], [1.0], [2.0]], [[2.0], [1.0], [0.0]]),
        ([[0.0], [1.0]], [[0.0], [0.0]]),
        ([[0.0], [1.0]], [[1.0], [2.0]]),
        ([[0.0], [1.0], [2.0]], [[0.0], [1.0]]),
        ([[0.0], [1.0], [2.0], [3.0]], [[0.0], [3.0], [6.0]]),
        ([[1.0]], [[1.0]]),
        ([[0.0], [0.0]], [[0.0], [0.0]]),
    ]
    for (Xr, Yr) in path_fixtures:
        path = dp_dtw_path(Xr, Yr)
        flat = []
        for (i, j) in path:
            flat.append(i)
            flat.append(j)
        if len(flat) > 32:
            check("path fixture %r/%r fits the 32-cell buffer" % (Xr, Yr), False)
            continue
        # the path must be legal and optimal
        legal = (path[0] == (0, 0) and path[-1] == (len(Xr) - 1, len(Yr) - 1)
                 and all((path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                         in ((1, 0), (0, 1), (1, 1)) for k in range(len(path) - 1)))
        if not legal:
            check("path fixture %r/%r: the path is legal" % (Xr, Yr), False)
        if abs(path_cost(Xr, Yr, path) - dp_dtw(Xr, Yr)) >= 1e-12:
            check("path fixture %r/%r: the path is optimal" % (Xr, Yr), False)
        body = ", ".join(str(v) for v in flat)
        print(f"    {{{len(Xr)}, {len(Yr)}, {len(path)}, {{{body}}}}},")
    print("};")

    # -------------------------------------------------------------------------
    section("SUMMARY")
    if check.failures == 0:
        print("  ALL ORACLE CHECKS PASSED")
        return 0
    print(f"  {check.failures} ORACLE CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
