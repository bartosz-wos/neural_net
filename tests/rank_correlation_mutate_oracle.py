#!/usr/bin/env python3
"""
tests/rank_correlation_mutate_oracle.py -- prove that the SPECIFICATION has
teeth, before any C++ is written against it.

WHY THIS FILE EXISTS
====================

A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN SENSITIVITY.
tests/rank_correlation_oracle.py printed "96 passed, 0 failed" on its first
successful execution. That is a statement about the oracle agreeing with
itself, not about the oracle being able to tell a right implementation from
a wrong one.

**THE CLAIM IN THE ORACLE'S OWN HEADER THAT WAS DELETED RATHER THAN KEPT.**
The oracle's first draft asserted that it had caught two defects in its own
first draft. It had caught NEITHER, and the claim was removed rather than
kept as flavour. This file is the replacement: the sensitivity is MEASURED
instead of narrated.

HOW IT WORKS
============

Each mutation is a source-level edit to a COPY of the oracle that breaks the
specification in one specific way. The mutated oracle is then run and MUST
FAIL. A mutation that survives is not a pass -- it is a hole in the
specification, and the harness reports it as such.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the mid-ranks use the FIRST rank of a tied group instead of the
       average -> the tie convention. Section 1 and every statistic built
       on ranks.

  OM2  Kendall's tau-b counts the JOINT ties in NEITHER xtie nor ytie
       (instead of BOTH) -> the tie correction. This is the single most
       common tau-b bug and it is invisible on a fixture with no ties, so
       the oracle's fixture has to carry joint ties.

  OM3  the normal-scores plotting position is (r - 0.375)/(n + 0.25)
       (the Blomqvist position) instead of (r - 0.5)/n (the Hazen
       position) -> the plotting-position convention.

  OM4  the population tau is (1/pi)*asin(rho) instead of (2/pi)*asin(rho)
       -> a factor-2 error in the closed form.

  OM5  the population Spearman is (6/pi)*asin(rho) instead of
       (6/pi)*asin(rho/2) -> a dropped /2 inside the arcsine.

  OM6  the copula correlation is computed from the RAW values instead of
       the normal scores -> the marginal-freeness property is lost, which
       is what makes it a COPULA statistic.

  OM7  the Spearman d^2 shortcut 1 - 6*sum(d^2)/(n*(n^2-1)) REPLACES the
       mid-rank definition -> the shortcut is only valid without ties.

THE CLASS OF MUTATION THAT IS DELIBERATELY NOT HERE
=====================================================

A mutation against a CONTRACTION whose operands are symmetric can be
PROVABLY INERT. There is no such mutation in this list, because every
quantity the oracle computes is either a scalar statistic or an
asymmetric rank vector, and every one of the seven above changes a
number that the oracle asserts on.

A mutation that is INERT must be proven inert by MEASUREMENT (print the
affected quantity with and without the mutation and show it is
bit-identical), and scored by a function that INVERTS the verdict, so a
wrong "inert" claim is itself caught. There is no inert mutation here, so
there is no inverted-verdict function to write -- but the harness still
REFUSES to report a survivor as a pass.

EXIT CODE
=========

  0  every mutation was caught
  1  at least one mutation survived (a hole in the SPECIFICATION)
  2  the harness itself is broken
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "rank_correlation_oracle.py")
PYTHON = "/usr/bin/python3"

# (id, description, search string, replacement string, expected marker)
# The search string must occur EXACTLY ONCE in the oracle; the harness
# asserts that before editing, because an anchor that matches twice would
# make the edit a coin flip.
MUTATIONS = [
    (
        "OM1",
        "mid-ranks use the FIRST rank of a tied group instead of the average",
        "ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)",
        "ranks[order[i:j + 1]] = float(i + 1)",
    ),
    (
        "OM2",
        "tau-b counts the JOINT ties in NEITHER xtie nor ytie",
        "            if dx == 0.0 and dy == 0.0:\n"
        "                xtie += 1\n"
        "                ytie += 1\n"
        "                ntie += 1",
        "            if dx == 0.0 and dy == 0.0:\n"
        "                ntie += 1",
    ),
    (
        "OM3",
        "the normal-scores plotting position is Blomqvist's, not Hazen's",
        '    return norm.ppf((r - 0.5) / n)',
        '    return norm.ppf((r - 0.375) / (n + 0.25))',
    ),
    (
        "OM4",
        "the population tau is (1/pi)asin(rho) instead of (2/pi)asin(rho)",
        "    return 2.0 / math.pi * math.asin(rho)",
        "    return 1.0 / math.pi * math.asin(rho)",
    ),
    (
        "OM5",
        "the population Spearman drops the /2 inside the arcsine",
        "    return 6.0 / math.pi * math.asin(rho / 2.0)",
        "    return 6.0 / math.pi * math.asin(rho)",
    ),
    (
        "OM6",
        "the copula correlation is computed from the RAW values",
        '    """Pearson\'s r of the normal scores: the marginal-free correlation."""\n'
        "    return ref_pearson_r(ref_normal_scores(x), ref_normal_scores(y))",
        '    """Pearson\'s r of the normal scores: the marginal-free correlation."""\n'
        "    return ref_pearson_r(x, y)",
    ),
    (
        "OM7",
        "Spearman uses the d^2 shortcut, which is only valid without ties",
        "def ref_spearman_rho(x, y):\n"
        '    """Pearson\'s r of the mid-ranks."""\n'
        "    return ref_pearson_r(ref_mid_ranks(x), ref_mid_ranks(y))",
        "def ref_spearman_rho(x, y):\n"
        '    """Pearson\'s r of the mid-ranks."""\n'
        "    rx = ref_mid_ranks(x)\n"
        "    ry = ref_mid_ranks(y)\n"
        "    n = len(x)\n"
        "    s = float(np.sum((rx - ry) ** 2))\n"
        "    return 1.0 - 6.0 * s / (n * (n * n - 1.0))",
    ),
]


def run_oracle(path):
    """Run an oracle copy. Returns (exit_code, stdout)."""
    proc = subprocess.run(
        [PYTHON, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=1200,
    )
    return proc.returncode, proc.stdout


def main():
    if not os.path.exists(PYTHON):
        print("MUT-HARNESS-BROKEN: %s not found" % PYTHON)
        return 2

    with open(ORACLE, "r") as fh:
        pristine = fh.read()

    # The pristine oracle MUST pass, or every verdict below is meaningless.
    tmpdir = tempfile.mkdtemp(prefix="rc_oracle_mut_")
    try:
        base_path = os.path.join(tmpdir, "pristine.py")
        with open(base_path, "w") as fh:
            fh.write(pristine)
        rc, out = run_oracle(base_path)
        if rc != 0:
            print("MUT-HARNESS-BROKEN: the PRISTINE oracle does not pass (exit=%d)" % rc)
            print(out[-4000:])
            return 2
        print("pristine oracle: PASS")

        caught = []
        survived = []
        broken = []

        for mid, desc, search, repl in MUTATIONS:
            # THE ANCHOR MUST BE UNIQUE. An anchor that occurs twice would
            # make the edit a coin flip, and a coin flip is not a verdict.
            count = pristine.count(search)
            if count != 1:
                print("%-4s MUT-HARNESS-BROKEN: anchor occurs %d times (must be 1)"
                      % (mid, count))
                broken.append(mid)
                continue

            mutated = pristine.replace(search, repl)

            # CHECKSUM THE TREE BEFORE AND AFTER THE EDIT. A mutation that
            # did not apply is a harness defect, not a verdict.
            if mutated == pristine:
                print("%-4s MUT-HARNESS-BROKEN: the mutation did not change the source"
                      % mid)
                broken.append(mid)
                continue

            path = os.path.join(tmpdir, "%s.py" % mid)
            with open(path, "w") as fh:
                fh.write(mutated)

            try:
                rc, out = run_oracle(path)
            except subprocess.TimeoutExpired:
                print("%-4s MUT-BROKEN: the mutated oracle TIMED OUT" % mid)
                broken.append(mid)
                continue

            if rc == 0:
                # The mutation SURVIVED. This is a hole in the SPECIFICATION,
                # and it is reported as such -- never as a pass.
                print("%-4s SURVIVED  %s" % (mid, desc))
                survived.append(mid)
            elif rc == 2:
                print("%-4s MUT-BROKEN (harness error)  %s" % (mid, desc))
                broken.append(mid)
            else:
                # rc == 1 means the oracle ran and reported failures.
                n_fail = 0
                m = re.search(r"=== SUMMARY: (\d+) passed, (\d+) failed ===", out)
                if m:
                    n_fail = int(m.group(2))
                if n_fail == 0:
                    # It exited non-zero but reported no failures -- that is
                    # a crash, not a verdict.
                    print("%-4s MUT-BROKEN: exited %d but reported 0 failures" % (mid, rc))
                    broken.append(mid)
                else:
                    first = ""
                    for line in out.splitlines():
                        if "[FAIL]" in line:
                            first = line.strip()[:150]
                            break
                    print("%-4s CAUGHT   %s   [%d failures]" % (mid, desc, n_fail))
                    if first:
                        print("         first failure: %s" % first)
                    caught.append(mid)

        print("")
        print("=== SUMMARY: %d caught, %d survived, %d broken ==="
              % (len(caught), len(survived), len(broken)))
        if survived:
            print("SURVIVORS ARE HOLES IN THE SPECIFICATION:")
            for mid in survived:
                print("  %s" % mid)
        if broken:
            print("HARNESS DEFECTS (not verdicts):")
            for mid in broken:
                print("  %s" % mid)
        return 1 if (survived or broken) else 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
