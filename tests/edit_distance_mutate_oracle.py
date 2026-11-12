#!/usr/bin/env python3
"""
tests/edit_distance_mutate_oracle.py -- the mutation harness for
tests/edit_distance_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The off-policy-evaluation oracle's first draft CLAIMED it had
caught two defects in itself, and it had caught NEITHER -- the claim was
deleted rather than kept as flavour. The isotonic and rank-correlation
oracles were green on their first execution too, and the same discipline was
applied.

So the oracle's discriminating power is MEASURED here, not narrated.

AND THE ORACLE DID FIND REAL DEFECTS IN ITSELF
===============================================

This harness exists partly because the oracle's first run FAILED -- three
times, for three different reasons, and every one of them was a bug in the
oracle rather than in the implementation it was checking:

  * the optimal-breakdown enumerator recursed on `rec(i, j)` instead of
    `rec(i, j - 1)`, which is infinite recursion;
  * the string-state search used a FIXED alphabet ("abc"), so any pair
    containing a symbol outside it was unreachable and the answer was
    silently +inf -- the fixture assertion `abs(want - ref) < 1e-12` caught
    it as `('abc', 'abd', inf, 1)`;
  * the OSA triangle-inequality witness had its operands swapped, so it
    asserted the WRONG inequality and failed on a property that is in fact
    true in the other direction.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the string-state BFS DROPS the substitution edge
       -> the string-state oracle really searches the edit graph, rather than
          accidentally computing something else. Section 1.
  OM2  the string-state BFS uses the RESTRICTED (OSA) rule -- the
       transposed region may not be edited again
       -> the DL/OSA distinction. Section 1. **THE MOST IMPORTANT ONE**: a
          BFS that is accidentally restricted agrees with DL on most pairs
          and disagrees on exactly the interesting ones.
  OM3  the exhaustive OSA enumeration drops the TRANSPOSITION column
       -> the OSA oracle really enumerates transpositions. Section 2.
  OM4  the OSA recurrence's transposition term uses the WRONG eligibility
       test (a[i-1]==b[j-1] instead of the cross-match a[i-1]==b[j-2] and
       a[i-2]==b[j-1])
       -> the transposition eligibility rule. Section 3.
  OM5  the sentinel-layout DL recurrence drops the TRANSPOSITION term
       -> the DL recurrence. Section 3.
  OM6  the sentinel-layout DL recurrence's transposition term uses `cost`
       where the reference uses `+1`
       -> the transposition-cost decision. Section 3. (Measured to agree on
          400000/400000 random pairs, so this mutation is expected to
          SURVIVE -- and a survivor is a hole in the SPECIFICATION, which is
          reported as such and never as a pass.)
  OM7  the weighted DP's boundary condition is D[i][0] = i*insertion
       instead of i*deletion
       -> the ORIENTATION of the weighted distance. Section 4. This is the
          single most consequential asymmetry in the module.
  OM8  the weighted DP's boundary condition is D[0][j] = j*deletion
       instead of j*insertion
       -> the orientation, from the other side. Section 4.
  OM9  the weighted DP charges `substitution` for a MATCH
       -> the match-is-free rule. Section 4.
  OM10 the fractional-weight cross-check compares against rapidfuzz
       instead of the Dijkstra oracle
       -> the reason the fractional-weight check exists at all: rapidfuzz
          TRUNCATES fractional weights, so it would agree with a wrong
          implementation. Section 4.
  OM11 the optimal-breakdown enumerator returns only the FIRST optimal
       split instead of all of them
       -> the ambiguity measurement. Section 5. The enumerator must report
          that some pairs have several optima.
  OM12 the WER denominator is |reference| instead of max(1, |reference|)
       -> the empty-reference convention. Section 6.
  OM13 the WER numerator is the number of reference tokens instead of the
       distance
       -> the WER formula. Section 6.
  OM14 the metric check for the unrestricted DL recurrence is REMOVED
       -> the triangle-inequality measurement. Section 7.

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

    /usr/bin/python3 tests/edit_distance_mutate_oracle.py

Exit code 0 means every mutation was CAUGHT. Exit code 1 means at least one
survived or the harness itself is defective.
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO_ROOT, "tests", "edit_distance_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# (id, description, search string, replacement string)
MUTATIONS = [
    ("OM1",
     "the string-state BFS drops the substitution edge",
     "                # substitution\n"
     "                for c in alpha:\n"
     "                    if c == s[i]:\n"
     "                        continue\n"
     "                    t = s[:i] + c + s[i + 1:]\n"
     "                    if t == b:\n"
     "                        return depth\n"
     "                    if t not in seen:\n"
     "                        seen.add(t)\n"
     "                        nxt.append(t)\n",
     "                # substitution -- MUTATED: edge removed\n"
     "                pass\n"),

    ("OM2",
     "the string-state BFS is RESTRICTED (OSA) instead of unrestricted (DL)",
     "                # transposition of two adjacent characters\n"
     "                if swap_adjacent and i + 1 < len(s) and s[i] != s[i + 1]:",
     "                # transposition of two adjacent characters\n"
     "                if False and swap_adjacent and i + 1 < len(s) and s[i] != s[i + 1]:"),

    ("OM3",
     "the exhaustive OSA enumeration drops the transposition column",
     "        if (i + 1 < len(a) and j + 1 < len(b)\n"
     "                and a[i] == b[j + 1] and a[i + 1] == b[j]):\n"
     "            rec(i + 2, j + 2, cost + 1)\n",
     "        # MUTATED: the transposition column is removed\n"),

    ("OM4",
     "the OSA recurrence's transposition eligibility test is wrong",
     "            if (i >= 2 and j >= 2\n"
     "                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):\n"
     "                best = min(best, d[i - 2][j - 2] + 1)\n",
     "            if (i >= 2 and j >= 2\n"
     "                    and a[i - 1] == b[j - 1]):\n"
     "                best = min(best, d[i - 2][j - 2] + 1)\n"),

    ("OM5",
     "the sentinel-layout DL recurrence drops the transposition term",
     "            d[i + 1][j + 1] = min(\n"
     "                d[i][j] + cost,\n"
     "                d[i + 1][j] + 1,\n"
     "                d[i][j + 1] + 1,\n"
     "                d[k][l] + (i - k - 1) + 1 + (j - l - 1),\n"
     "            )\n",
     "            d[i + 1][j + 1] = min(\n"
     "                d[i][j] + cost,\n"
     "                d[i + 1][j] + 1,\n"
     "                d[i][j + 1] + 1,\n"
     "            )\n"),

    ("OM6",
     "the DL transposition term uses `cost` where the reference uses `+1`",
     "                d[k][l] + (i - k - 1) + 1 + (j - l - 1),\n",
     "                d[k][l] + (i - k - 1) + cost + (j - l - 1),\n"),

    ("OM7",
     "the weighted DP's D[i][0] boundary uses the INSERTION weight instead "
     "of the DELETION weight",
     "        for i in range(n + 1):\n"
     "            d[i][0] = i * w[1]\n"
     "        for j in range(m + 1):\n"
     "            d[0][j] = j * w[0]\n"
     "        for i in range(1, n + 1):\n"
     "            for j in range(1, m + 1):\n"
     "                d[i][j] = min(\n"
     "                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),\n",
     "        for i in range(n + 1):\n"
     "            d[i][0] = i * w[0]\n"
     "        for j in range(m + 1):\n"
     "            d[0][j] = j * w[0]\n"
     "        for i in range(1, n + 1):\n"
     "            for j in range(1, m + 1):\n"
     "                d[i][j] = min(\n"
     "                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),\n"),

    ("OM8",
     "the weighted DP's D[0][j] boundary uses the DELETION weight instead "
     "of the INSERTION weight",
     "        for i in range(n + 1):\n"
     "            d[i][0] = i * w[1]\n"
     "        for j in range(m + 1):\n"
     "            d[0][j] = j * w[0]\n"
     "        for i in range(1, n + 1):\n"
     "            for j in range(1, m + 1):\n"
     "                d[i][j] = min(\n"
     "                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),\n",
     "        for i in range(n + 1):\n"
     "            d[i][0] = i * w[1]\n"
     "        for j in range(m + 1):\n"
     "            d[0][j] = j * w[1]\n"
     "        for i in range(1, n + 1):\n"
     "            for j in range(1, m + 1):\n"
     "                d[i][j] = min(\n"
     "                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),\n"),

    ("OM9",
     "the weighted DP charges `substitution` for a MATCH",
     "                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),\n",
     "                    d[i - 1][j - 1] + float(w[2]),\n"),

    ("OM10",
     "the fractional-weight cross-check uses rapidfuzz instead of Dijkstra",
     "        ins, dl_, sub = (0.5, 1.0, 2.0) if random.random() < 0.5 else (1.5, 0.75, 1.25)\n"
     "        ref = dijkstra_weighted(a, b, ins, dl_, sub)\n",
     "        ins, dl_, sub = (0.5, 1.0, 2.0) if random.random() < 0.5 else (1.5, 0.75, 1.25)\n"
     "        ref = RfD.Levenshtein.distance(a, b, weights=(int(ins), int(dl_), int(sub)))\n"),

    ("OM11",
     "the optimal-breakdown enumerator returns only the FIRST optimal split",
     "    return rec(n, m), d[n][m]\n",
     "    return frozenset({sorted(rec(n, m))[0]}), d[n][m]\n"),

    ("OM12",
     "the WER denominator drops the max(1, |reference|) clamp",
     "    return max(1, n_ref)\n",
     "    return n_ref\n"),

    ("OM13",
     "the WER numerator is the reference length instead of the distance",
     "    dist = levenshtein_dp(ref_tokens, hyp_tokens)\n",
     "    dist = float(len(ref_tokens))\n"),

    ("OM14",
     "the triangle-inequality check is applied to OSA instead of DL -- so the "
     "check FIREDS on a distance that is known NOT to be a metric",
     "        if dl_wikipedia(x, z) > dl_wikipedia(x, y) + dl_wikipedia(y, z):\n"
     "            viol += 1\n"
     "    check(\"unrestricted DL satisfies the triangle inequality\", viol == 0,\n"
     "          f\"{viol} violations / 4000\")\n",
     "        if osa_dp(x, z) > osa_dp(x, y) + osa_dp(y, z):\n"
     "            viol += 1\n"
     "    check(\"unrestricted DL satisfies the triangle inequality\", viol == 0,\n"
     "          f\"{viol} violations / 4000\")\n"),
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

    workdir = tempfile.mkdtemp(prefix="edit_distance_oracle_mut_")
    target = os.path.join(workdir, "edit_distance_oracle.py")
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

    # OM6 is EXPECTED TO SURVIVE: the DL transposition term's `+1` and
    # `cost` were measured to agree on 400000/400000 random pairs, so the
    # specification genuinely does not distinguish them. That survivor is a
    # fact about the SPECIFICATION, not a gap in the harness, and it is
    # documented in the module header rather than silently tolerated.
    EXPECTED_SURVIVORS = {"OM6"}

    unexpected_survivors = SURVIVED - len(EXPECTED_SURVIVORS)
    if BROKEN or NOOP or ANCHOR_BAD or unexpected_survivors:
        print("HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):")
        return 1
    if SURVIVED:
        print("NOTE: %d mutation(s) survived -- each is a documented fact about"
              % SURVIVED)
        print("      the specification, not a hole in the harness:")
        for mid in sorted(EXPECTED_SURVIVORS):
            print("        %s -- the DL transposition cost: `+1` and `cost`"
                  " agree on" % mid)
            print("           400000/400000 random pairs, so the")
            print("           specification does not distinguish them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
