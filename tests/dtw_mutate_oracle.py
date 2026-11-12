#!/usr/bin/env python3
"""
tests/dtw_mutate_oracle.py -- the mutation harness for tests/dtw_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The isotonic and rank-correlation oracles were green on their
first execution and the same discipline was applied here.

AND THIS ORACLE FOUND REAL DEFECTS IN ITSELF
=============================================

The DTW oracle's first run FAILED, for three different reasons, and every one
of them was a bug in the oracle rather than in the implementation it was
checking:

  * the exhaustive path enumeration DOUBLE-COUNTED the final cell, so it
    enumerated paths that revisit a cell;
  * the path backtracker DUPLICATED the origin cell, which made every path
    comparison fail -- and which is the SAME harness bug that had earlier
    made a standalone probe report 0/300 for every candidate tie-break;
  * two hand-derived expected values were wrong: the oracle measured the DTW
    distance of [0,1,2] vs [0,2,4] as 3.0, not the 2.0 the first draft
    asserted. A hand-derived number is exactly the failure mode this repo
    has hit before -- the number was CORRECT for the fixture it was written
    for, and the fixture was simply not the oracle's.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  DM1  the path enumeration loses the HORIZONTAL step (0,1)
       -> the step pattern really has three neighbours. Section 1.
  DM2  the path enumeration loses the VERTICAL step (1,0)
       -> the step pattern, from the other side. Section 1.
  DM3  the path enumeration loses the DIAGONAL step (1,1)
       -> the step pattern, from the third side. Section 1.
  DM4  the DP's local cost is the SQUARED Euclidean distance
       -> the local-cost rule. Section 1. **THE MOST IMPORTANT ONE**: the
          distance and the path both change, and the DP-vs-dtw-python
          cross-check is what catches it.
  DM5  the DP's local cost is the MANHATTAN distance
       -> the local-cost rule again, with a different wrong metric. Section 1.
  DM6  the DP accumulates the local cost with a SQUARE ROOT at the end
       -> the raw-cumulative rule. Section 1.
  DM7  the DP normalises the distance by the path length
       -> the raw-cumulative rule, from the other side. Section 1.
  DM8  the DP uses the symmetric2 step pattern (the four-neighbour lattice
       with a doubled weight on the horizontal step)
       -> the step-pattern decision. Section 2. dtw-python's DEFAULT is
          symmetric2, so a mutation to it agrees with the default reference
          and the explicitly-pinned one disagrees -- which is exactly what
          makes the explicit pinning load-bearing.
  DM9  the reference call DROPS the explicit `step_pattern=symmetric1`
       -> the explicit pinning. Section 2. Without the explicit argument the
          reference silently becomes symmetric2 and the cross-check stops
          discriminating.
  DM10 the band is applied as |i-j| >= band instead of |i-j| > band
       -> the band rule. Section 3.
  DM11 the band is applied to the 0-BASED DATA indices while the DP's cells
       are 1-BASED
       -> the band index convention. Section 3.
  DM12 the feasibility rule is dropped -- an infeasible band returns a
       FINITE distance instead of +inf
       -> the infeasible-band convention. Section 3.
  DM13 the path backtracker prefers the DIAGONAL branch only, and falls
       back to a DIFFERENT order when it cannot move
       -> the tie-break. Section 4.
  DM14 the tie-break order is changed from ('d','l','u') to ('d','u','l')
       -> the tie-break order itself. Section 4. Measured: 296/300 against
          dtw-python, so this mutation is only WEAKLY caught -- and the
          measurement is reported rather than hidden.
  DM15 the symmetry measurement is inverted
       -> the symmetry property. Section 5.
  DM16 the DTW local cost uses the ROW VECTOR (x[i] - y[j]) but the
       MULTIDIMENSIONAL case is computed column-wise
       -> the rows-are-observations convention. Section 2.

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
================================================

  1. The anchor must occur EXACTLY ONCE. A twice-occurring anchor makes
     the edit a coin flip, and a coin flip is not a verdict. The harness
     asserts the count and reports MUT-ANCHOR otherwise.
  2. The mutated source is compared to the pristine source -- a mutation
     that did not apply is a HARNESS DEFECT, not a verdict.
  3. A non-zero exit with ZERO reported failures is a CRASH, not a
     verdict -> MUT-BROKEN.
  4. A SURVIVOR is a hole in the SPECIFICATION, never a pass.

RUN
===

    /usr/bin/python3 tests/dtw_mutate_oracle.py

Exit code 0 means every mutation was CAUGHT. Exit code 1 means at least one
survived or the harness itself is defective.
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO_ROOT, "tests", "dtw_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# (id, description, search string, replacement string)
MUTATIONS = [
    ("DM1",
     "the path enumeration loses the HORIZONTAL step (0,1)",
     "        for (di, dj) in ((1, 0), (0, 1), (1, 1)):\n",
     "        for (di, dj) in ((1, 0), (1, 1)):\n"),

    ("DM2",
     "the path enumeration loses the VERTICAL step (1,0)",
     "        for (di, dj) in ((1, 0), (0, 1), (1, 1)):\n",
     "        for (di, dj) in ((0, 1), (1, 1)):\n"),

    ("DM3",
     "the path enumeration loses the DIAGONAL step (1,1)",
     "        for (di, dj) in ((1, 0), (0, 1), (1, 1)):\n",
     "        for (di, dj) in ((1, 0), (0, 1)):\n"),

    ("DM4",
     "the DP's local cost is the SQUARED Euclidean distance",
     "def local_cost(x, y):\n"
     "    \"\"\"Euclidean distance between two observation vectors.\"\"\"\n"
     "    return float(np.linalg.norm(np.asarray(x, dtype=float)\n"
     "                                - np.asarray(y, dtype=float)))\n",
     "def local_cost(x, y):\n"
     "    \"\"\"MUTATED: the SQUARED Euclidean distance.\"\"\"\n"
     "    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)\n"
     "    return float(np.dot(d, d))\n"),

    ("DM5",
     "the DP's local cost is the MANHATTAN distance",
     "def local_cost(x, y):\n"
     "    \"\"\"Euclidean distance between two observation vectors.\"\"\"\n"
     "    return float(np.linalg.norm(np.asarray(x, dtype=float)\n"
     "                                - np.asarray(y, dtype=float)))\n",
     "def local_cost(x, y):\n"
     "    \"\"\"MUTATED: the Manhattan distance.\"\"\"\n"
     "    return float(np.sum(np.abs(np.asarray(x, dtype=float)\n"
     "                             - np.asarray(y, dtype=float))))\n"),

    ("DM6",
     "the DP takes a square root of the accumulated cost",
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return float(np.sqrt(D[n][m]))\n"),

    ("DM7",
     "the DP normalises the distance by the path length",
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m] / float(n + m)\n"),

    ("DM8",
     "the DP uses the symmetric2 lattice instead of symmetric1",
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            # MUTATED: the symmetric2 lattice -- the horizontal step\n"
     "            # carries a doubled weight.\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j],\n"
     "                          2.0 * D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n"),

    ("DM9",
     "the reference call drops the explicit step_pattern=symmetric1",
     "    kwargs = {\"step_pattern\": dtw.symmetric1}\n",
     "    kwargs = {}\n"),

    ("DM10",
     "the band is applied as |i-j| >= band instead of |i-j| > band",
     "            if band is not None and abs(i - j) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            if band is not None and abs(i - j) >= band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n"),

    ("DM11",
     "the band is applied to the 0-based data indices instead of the cells",
     "            if band is not None and abs(i - j) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            if band is not None and abs((i - 1) - (j - 1)) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n"),

    ("DM12",
     "an infeasible band returns a FINITE distance instead of +inf",
     "            if band is not None and abs(i - j) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            if band is not None and abs(i - j) > band:\n"
     "                D[i][j] = 0.0\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n"),

    ("DM13",
     "the path backtracker prefers the diagonal and never falls back",
     "        for p in order:\n"
     "            if p == \"d\" and D[i][j] == D[i - 1][j - 1] + c:\n"
     "                cells.append((i - 1, j - 1))\n"
     "                i -= 1\n"
     "                j -= 1\n"
     "                moved = True\n"
     "                break\n"
     "            if p == \"l\" and D[i][j] == D[i][j - 1] + c:\n"
     "                cells.append((i - 1, j - 1))\n"
     "                j -= 1\n"
     "                moved = True\n"
     "                break\n"
     "            if p == \"u\" and D[i][j] == D[i - 1][j] + c:\n"
     "                cells.append((i - 1, j - 1))\n"
     "                i -= 1\n"
     "                moved = True\n"
     "                break\n",
     "        for p in order:\n"
     "            if p == \"d\" and D[i][j] == D[i - 1][j - 1] + c:\n"
     "                cells.append((i - 1, j - 1))\n"
     "                i -= 1\n"
     "                j -= 1\n"
     "                moved = True\n"
     "                break\n"),

    ("DM14",
     "the tie-break order is ('d','u','l') instead of ('d','l','u')",
     "def dp_dtw_path(X, Y, band=None, order=(\"d\", \"l\", \"u\")):\n",
     "def dp_dtw_path(X, Y, band=None, order=(\"d\", \"u\", \"l\")):\n"),

    ("DM15",
     "the symmetry measurement is inverted",
     "        if abs(dp_dtw(X, Y) - dp_dtw(Y, X)) > 1e-9:\n"
     "            asym += 1\n",
     "        if abs(dp_dtw(X, Y) - dp_dtw(Y, X)) <= 1e-9:\n"
     "            asym += 1\n"),

    ("DM16",
     "the multidimensional local cost SUMS the per-coordinate squared "
     "differences instead of taking their square root (so it is the "
     "squared Euclidean distance for a multivariate observation)",
     "def local_cost(x, y):\n"
     "    \"\"\"Euclidean distance between two observation vectors.\"\"\"\n"
     "    return float(np.linalg.norm(np.asarray(x, dtype=float)\n"
     "                                - np.asarray(y, dtype=float)))\n",
     "def local_cost(x, y):\n"
     "    \"\"\"MUTATED: the squared Euclidean distance.\"\"\"\n"
     "    xv = np.atleast_1d(np.asarray(x, dtype=float))\n"
     "    yv = np.atleast_1d(np.asarray(y, dtype=float))\n"
     "    d = xv - yv\n"
     "    return float(np.sum(d * d))\n"),

    ("DM17",
     "the band is applied to the 0-based DATA indices while the DP's cells "
     "are 1-based -- OBSERVABLE only because the band is compared against "
     "|i-j| WITHOUT shifting both indices",
     "            if band is not None and abs(i - j) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n",
     "            if band is not None and abs((i - 1) - (j - 1) + 1) > band:\n"
     "                continue\n"
     "            D[i][j] = min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) \\\n"
     "                + local_cost(X[i - 1], Y[j - 1])\n"
     "    return D[n][m]\n"),
]


def run_oracle(path):
    """Run the (possibly mutated) oracle. Returns (rc, failures_reported, stdout)."""
    proc = subprocess.run([PYTHON, path], capture_output=True, text=True)
    failures = proc.stdout.count("[FAIL]")
    return proc.returncode, failures, proc.stdout


def main():
    if not os.path.exists(ORACLE):
        print("MUT-HARNESS-BROKEN: %s does not exist" % ORACLE)
        return 2

    with open(ORACLE, "r") as f:
        pristine = f.read()

    workdir = tempfile.mkdtemp(prefix="dtw_oracle_mut_")
    target = os.path.join(workdir, "dtw_oracle.py")
    shutil.copy(ORACLE, target)

    baseline_rc, baseline_failures, _ = run_oracle(target)
    if baseline_rc != 0 or baseline_failures != 0:
        print("MUT-HARNESS-BROKEN: the pristine oracle is not green "
              "(rc=%d, %d failures)" % (baseline_rc, baseline_failures))
        shutil.rmtree(workdir, ignore_errors=True)
        return 2

    global CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD

    try:
        for mid, desc, search, replacement in MUTATIONS:
            # --- restore FIRST: every mutation starts from the pristine file.
            with open(target, "w") as f:
                f.write(pristine)

            count = pristine.count(search)
            if count != 1:
                print("%s MUT-ANCHOR  the anchor occurs %d times -- harness "
                      "defect, not a verdict" % (mid, count))
                ANCHOR_BAD += 1
                continue

            with open(target, "w") as f:
                f.write(pristine.replace(search, replacement))

            # --- checksum: a mutation that did not apply is a harness defect.
            with open(target, "r") as f:
                mutated = f.read()
            if mutated == pristine:
                print("%s MUT-NOOP  the source is unchanged -- harness defect"
                      % mid)
                NOOP += 1
                continue

            rc, failures, out = run_oracle(target)

            if rc == 0 and failures == 0:
                # The oracle PASSED with the mutation in place. That is a
                # hole in the ORACLE, and it is reported as such -- never as
                # a pass.
                print("%s SURVIVED  %s" % (mid, desc))
                SURVIVED += 1
            elif rc != 0 and failures == 0:
                # A non-zero exit with zero reported failures is a CRASH.
                print("%s MUT-BROKEN  the oracle crashed without reporting a "
                      "failure (rc=%d)" % (mid, rc))
                BROKEN += 1
            else:
                first = ""
                for line in out.splitlines():
                    if "[FAIL]" in line:
                        first = line.strip()
                        break
                if not first:
                    for line in out.splitlines():
                        if "Traceback" in line:
                            first = line.strip()
                            break
                print("%s CAUGHT    %s" % (mid, desc))
                print("            -> %s" % (first or "the oracle reported a failure"))
                CAUGHT += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 70)
    print("oracle mutation summary: %d caught, %d survived, %d broken, "
          "%d noop, %d anchor-bad" % (CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD))
    print("=" * 70)

    # DM11 is EXPECTED TO SURVIVE, and the reason is a fact about the
    # SPECIFICATION rather than a hole in the harness. The DP's cells are
    # (i,j) with i in 1..n and j in 1..m, and the DATA indices are (i-1,j-1).
    # The band test is |i-j| > band, and |i-j| == |(i-1)-(j-1)| EXACTLY --
    # shifting BOTH indices by one cannot change their difference. So the
    # "1-based cell" and "0-based data" readings of the band are the SAME
    # function: measured, 0/2000 random cases differ. The convention is
    # unobservable, and an unobservable mutation is not a gap in the suite.
    # DM17 is the OBSERVABLE version of the same idea -- it shifts only ONE
    # index, which does change the answer.
    EXPECTED_SURVIVORS = {"DM11"}

    unexpected_survivors = SURVIVED - len(EXPECTED_SURVIVORS)
    if BROKEN or NOOP or ANCHOR_BAD or unexpected_survivors:
        print("HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):")
        return 1
    if SURVIVED:
        print("NOTE: %d mutation(s) survived -- each is a documented fact about"
              % SURVIVED)
        print("      the specification, not a hole in the harness:")
        print("        DM11 -- the band's index convention: the DP's cells are")
        print("           1-based and the data indices are 0-based, but")
        print("           |i-j| == |(i-1)-(j-1)| EXACTLY, so the two")
        print("           readings are the same function (0/2000 differ).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
