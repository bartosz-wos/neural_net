#!/usr/bin/env python3
"""
tests/isotonic_mutate_oracle.py -- the mutation harness for
tests/isotonic_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The off-policy-evaluation oracle's first draft CLAIMED it
had caught two defects in itself, and it had caught NEITHER -- the claim
was deleted rather than kept as flavour. The rank-correlation oracle was
green on its first execution too, and the same discipline was applied.

So the oracle's discriminating power is MEASURED here, not narrated.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the tie-collapse uses the UNWEIGHTED mean of y
       -> the tie-collapse rule. Section 1.
  OM2  the tie-collapse does not SUM the weights
       -> the tie-collapse rule, from the other side. Section 1.
  OM3  PAVA pools NON-ADJACENT violators (a global-minimum shortcut)
       -> the adjacency rule. Section 2.
  OM4  the merged block's value is the mean of the two block MEANS,
       not of the block's points
       -> the pooling rule. Section 2. THE MOST IMPORTANT ONE: it is
          invisible on a fixture whose blocks all have the same size.
  OM5  the prediction is the STEP function, not the linear interpolation
       -> the step-vs-interpolation distinction. Section 3.
  OM6  the trimming is not applied (all knots retained)
       -> the trimming rule. Section 4.
  OM7  zero-weight points are kept instead of dropped
       -> the zero-weight rule. Section 1/4.
  OM8  the decreasing direction is implemented as "negate and fit
       increasing"
       -> the decreasing-direction rule. Section 2.
  OM9  out_of_bounds=Clip returns the MEAN of the end values instead of
       the end values
       -> the out-of-bounds rule. Section 5.

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

    /usr/bin/python3 tests/isotonic_mutate_oracle.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "isotonic_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# id | description | search | replacement
MUTATIONS = [
    ("OM1",
     "the tie-collapse uses the UNWEIGHTED mean of y",
     "    np.add.at(sum_wy, inverse, w * y)\n"
     "    np.add.at(sum_w, inverse, w)\n"
     "    return ux, sum_wy / sum_w, sum_w",
     "    np.add.at(sum_wy, inverse, y)\n"
     "    np.add.at(sum_w, inverse, w)\n"
     "    return ux, sum_wy / sum_w, sum_w"),

    ("OM2",
     "the tie-collapse does not SUM the weights",
     "    np.add.at(sum_w, inverse, w)\n"
     "    return ux, sum_wy / sum_w, sum_w",
     "    np.add.at(sum_w, inverse, 1.0)\n"
     "    return ux, sum_wy / sum_w, sum_w"),

    ("OM3",
     "PAVA pools NON-ADJACENT violators",
     "            a, b = blocks[-2], blocks[-1]\n"
     "            va = sum_wy[a] / sum_w[a]\n"
     "            vb = sum_wy[b] / sum_w[b]\n"
     "            violation = (va > vb) if increasing else (va < vb)\n"
     "            if not violation:\n"
     "                break\n"
     "            # POOL THE ADJACENT VIOLATORS. The merged block's value is the\n"
     "            # weighted mean of ALL the points it contains -- computed from\n"
     "            # the accumulated sums, never as the mean of the two means.\n"
     "            sum_wy[a] += sum_wy[b]\n"
     "            sum_w[a] += sum_w[b]\n"
     "            count[a] += count[b]\n"
     "            blocks.pop()",
     "            a, b = blocks[0], blocks[-1]\n"
     "            va = sum_wy[a] / sum_w[a]\n"
     "            vb = sum_wy[b] / sum_w[b]\n"
     "            violation = (va > vb) if increasing else (va < vb)\n"
     "            if not violation:\n"
     "                break\n"
     "            sum_wy[b] += sum_wy[a]\n"
     "            sum_w[b] += sum_w[a]\n"
     "            count[b] += count[a]\n"
     "            blocks.pop(0)"),

    ("OM4",
     "the merged block's value is the mean of the two block MEANS",
     "            sum_wy[a] += sum_wy[b]\n"
     "            sum_w[a] += sum_w[b]\n"
     "            count[a] += count[b]",
     "            sum_wy[a] = 0.5 * (sum_wy[a] + sum_wy[b])\n"
     "            sum_w[a] = 2.0\n"
     "            count[a] += count[b]"),

    ("OM5",
     "the prediction is the STEP function, not the linear interpolation",
     "    if np.any(inside):\n"
     "        result[inside] = np.interp(t[inside], x_knots, y_knots)",
     "    if np.any(inside):\n"
     "        idx = np.searchsorted(x_knots, t[inside], side='right') - 1\n"
     "        idx = np.clip(idx, 0, len(y_knots) - 1)\n"
     "        result[inside] = y_knots[idx]"),

    ("OM6",
     "the trimming is not applied (all knots retained)",
     "    keep[1:-1] = ((y_knots[1:-1] != y_knots[:-2]) |\n"
     "                  (y_knots[1:-1] != y_knots[2:]))",
     "    pass"),

    ("OM7",
     "zero-weight points are kept instead of dropped",
     "    keep = w > 0\n"
     "    x, y, w = x[keep], y[keep], w[keep]",
     "    keep = np.ones(len(w), dtype=bool)\n"
     "    x, y, w = x[keep], y[keep], w[keep]"),

    ("OM8",
     "the decreasing direction is 'negate and fit increasing'",
     "            violation = (va > vb) if increasing else (va < vb)",
     "            violation = (va > vb)"),

    ("OM9",
     "out_of_bounds=Clip returns the MEAN of the end values",
     "        result[t < x_min] = y_knots[0]\n"
     "        result[t > x_max] = y_knots[-1]",
     "        _mid = 0.5 * (y_knots[0] + y_knots[-1])\n"
     "        result[t < x_min] = _mid\n"
     "        result[t > x_max] = _mid"),
]


def run_oracle(path):
    """Run the (possibly mutated) oracle. Returns (rc, failures_reported)."""
    proc = subprocess.run([PYTHON, path], capture_output=True, text=True)
    failures = proc.stdout.count("[FAIL]")
    return proc.returncode, failures, proc.stdout


def main():
    if not os.path.exists(ORACLE):
        print("MUT-HARNESS-BROKEN: %s does not exist" % ORACLE)
        return 2

    # The pristine source, and a scratch copy that is mutated in place.
    with open(ORACLE, "r") as f:
        pristine = f.read()

    workdir = tempfile.mkdtemp(prefix="isotonic_oracle_mut_")
    target = os.path.join(workdir, "isotonic_oracle.py")
    shutil.copy(ORACLE, target)

    # The oracle imports nothing from the repo, so a copy in a scratch
    # directory runs standalone.
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
                for line in out.split("\n"):
                    if "[FAIL]" in line:
                        first = line.strip()
                        break
                print("%s CAUGHT  %s" % (mid, desc))
                print("      %s" % first)
                CAUGHT += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("")
    print("=== SUMMARY: %d caught, %d survived, %d broken, %d noop, "
          "%d anchor ===" % (CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD))
    if SURVIVED > 0:
        print("SURVIVORS ARE HOLES IN THE ORACLE (the SPECIFICATION):")
        return 1
    if BROKEN > 0 or NOOP > 0 or ANCHOR_BAD > 0:
        print("HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
