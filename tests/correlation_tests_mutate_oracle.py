#!/usr/bin/env python3
"""
tests/correlation_tests_mutate_oracle.py -- prove that the SPECIFICATION has
teeth, before any C++ is written against it.

WHY THIS FILE EXISTS
====================

A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN SENSITIVITY.
tests/correlation_tests_oracle.py printed "43 passed, 0 failed" on its first
successful execution. That is a statement about the oracle agreeing with
ITSELF, not about the oracle being able to tell a right implementation
from a wrong one.

**A COROLLARY THAT BIT THIS REPO TWICE ALREADY: the interesting question
is not "does the oracle fail when I break it" but "WHAT WOULD IT NOT
NOTICE".** So the mutations below are not random -- each one breaks the
oracle in a way that a plausible IMPLEMENTATION might also break, and the
harness requires that the oracle notice.

HOW IT WORKS
============

Each mutation is a source-level edit to a COPY of the oracle that breaks
the specification in one specific way. The mutated oracle is then run and
MUST FAIL. A mutation that survives is not a pass -- it is a hole in the
specification, and the harness reports it as such.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the t statistic's degrees of freedom are n-2 -> n-1
       -> C1. The p-value depends on the dof through BOTH the t statistic
          and the survival function.

  OM2  the statistic is NOT clipped to [-1, 1]
       -> C4. **THE MOST IMPORTANT ONE.** Without the clip, the p-value of
          a PERFECT correlation is NaN, because the statistic this repo's
          own `pearson_r` produces (measured at 1.0000000000000011102)
          makes `1 - r*r` negative.

  OM3  the n == 2 case returns 1.0 instead of NaN
       -> C3. The references DISAGREE at n == 2, so the oracle's own
          choice has to be pinned by an assertion rather than inherited.

  OM4  the p-value is the ONE-SIDED upper tail instead of the two-sided one
       -> C6. **THE SIGN OF THE STATISTIC IS UNOBSERVABLE**, because the
          formula computes `x = dof/(dof + t*t)` and `t*t` is the same
          for +t and -t. **THE FIRST FORM OF THIS MUTATION DROPPED THE
          ABSOLUTE VALUE, AND IT SURVIVED** -- a real gap in the
          specification, since the oracle's checks could not see it. The
          observable difference between one-sided and two-sided is the
          VALUE (the one-sided upper tail is HALF the two-sided one), so
          the mutation now targets the VALUE.

  OM5  the Kendall variance DROPS the `2*xtie*ytie/m` term
       -> C2. The tie terms are load-bearing.

  OM6  the Kendall variance DROPS the `x0*y0/(9*m*(n-2))` term
       -> C2, the other tie term.

  OM7  the tie-GROUP statistics use the tie PAIR COUNT instead
       -> C2. **THE SECOND MOST IMPORTANT ONE.** F5's x has the groups
          {4,2} and F6's x has {3,3,2}: the SAME number of tied pairs and
          DIFFERENT variances. A variance built from the pair count alone
          gives the same answer for both, so the oracle's F5/F6 pair
          cannot tell the difference -- and neither can the C++.

  OM8  the two-sided normal survival is DOUBLED
       -> C6. The erfc already carries both tails, so the factor-2 form is
          a p-value twice the right one. **THIS WAS A REAL BUG IN THE
          C++ SUITE'S OWN INDEPENDENT REFERENCE**, and the oracle caught
          it.

  OM9  the Kendall p-value uses `2*(1 - norm.sf(|z|))` instead of
       `2*norm.sf(|z|)`
       -> the CANCELLATION defect. `1 - Phi` in the tail loses every
          digit that matters.

  OM10 the incomplete beta's arguments are swapped
       -> C1: the incomplete beta's arguments are (dof/2, 1/2), and
          swapping them gives a different number.

  OM11 the Pearson p-value uses the EXACT BETA form instead of the t form
       -> C1. **THE FIRST FORM OF THIS MUTATION WAS DECLARED INERT, AND
          THAT WAS WRONG.** The two forms agree to 3.3e-14 RELATIVE, so
          the claim "they are the same answer" looked safe -- but the
          oracle's own R-agreement check has a 1e-14 tolerance, and the
          measured difference is 1.65e-13, which is ABOVE it. **A
          CONVENTION THAT IS OBSERVABLE AT THE ORACLE'S OWN TOLERANCE IS
          NOT INERT, NO MATTER HOW SMALL THE RELATIVE DIFFERENCE
          LOOKS.** The harness caught the wrong claim, and the mutation is
          now expected to be CAUGHT.

THE CLASS OF MUTATION THAT IS DELIBERATELY NOT HERE
====================================================

A mutation against a CONTRACTION whose operands are symmetric can be
PROVABLY INERT. Every quantity the oracle computes is either a scalar
statistic or an asymmetric vector, so no mutation in this list is inert by
construction -- and the ONE that was declared inert (OM11) turned out to
be observable, which is why the inertness measurement exists.

A mutation that is INERT must be proven inert by MEASUREMENT (print the
affected quantity with and without the mutation and show it is
bit-identical), and scored by a function that INVERTS the verdict, so a
wrong "inert" claim is itself caught. There is no inert mutation here, so
the inverted-verdict function has nothing to score -- but the harness
still REFUSES to report a survivor as a pass.

EXIT CODE
=========

  0  every mutation was caught (or was measured-inert)
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
ORACLE = os.path.join(HERE, "correlation_tests_oracle.py")
PYTHON = "/usr/bin/python3"

# (id, description, search string, replacement string, expectation)
# The search string must occur EXACTLY ONCE in the oracle; the harness
# asserts that before editing, because an anchor that matches twice would
# make the edit a coin flip.
MUTATIONS = [
    (
        "OM1",
        "the t statistic's degrees of freedom are n-2 -> n-1",
        "    t = abs(r) * math.sqrt((n - 2.0) / denom)\n"
        "    return ref_t_two_sided(t, n - 2.0)",
        "    t = abs(r) * math.sqrt((n - 1.0) / denom)\n"
        "    return ref_t_two_sided(t, n - 1.0)",
        "caught",
    ),
    (
        "OM2",
        "the statistic is not clipped to [-1, 1]",
        "    if r > 1.0:\n"
        "        r = 1.0\n"
        "    elif r < -1.0:\n"
        "        r = -1.0\n",
        "",
        "caught",
    ),
    (
        "OM3",
        "the n == 2 case returns 1.0 instead of NaN",
        "    if n == 2:\n"
        "        return float(\"nan\")\n"
        "    if r != r:  # NaN",
        "    if n == 2:\n"
        "        return 1.0\n"
        "    if r != r:  # NaN",
        "caught",
    ),
    (
        "OM4",
        "the p-value is the ONE-SIDED upper tail instead of the two-sided one",
        "    t = abs(r) * math.sqrt((n - 2.0) / denom)\n"
        "    return ref_t_two_sided(t, n - 2.0)",
        "    t = abs(r) * math.sqrt((n - 2.0) / denom)\n"
        "    return 0.5 * ref_t_two_sided(t, n - 2.0)",
        "caught",
    ),
    (
        "OM5",
        "the Kendall variance drops the 2*xtie*ytie/m term",
        "    var = ((m * (2 * n + 5) - x1 - y1) / 18.0\n"
        "           + (2.0 * xt * yt) / m\n"
        "           + (x0 * y0) / (9.0 * m * (n - 2)))",
        "    var = ((m * (2 * n + 5) - x1 - y1) / 18.0\n"
        "           + (x0 * y0) / (9.0 * m * (n - 2)))",
        "caught",
    ),
    (
        "OM6",
        "the Kendall variance drops the x0*y0/(9*m*(n-2)) term",
        "    var = ((m * (2 * n + 5) - x1 - y1) / 18.0\n"
        "           + (2.0 * xt * yt) / m\n"
        "           + (x0 * y0) / (9.0 * m * (n - 2)))",
        "    var = ((m * (2 * n + 5) - x1 - y1) / 18.0\n"
        "           + (2.0 * xt * yt) / m)",
        "caught",
    ),
    (
        "OM7",
        "the tie-group statistics use the tie PAIR COUNT instead",
        "        xtie += c * (c - 1) // 2\n"
        "        x0 += c * (c - 1) * (c - 2)\n"
        "        x1 += c * (c - 1) * (2 * c + 5)",
        "        xtie += c * (c - 1) // 2\n"
        "        x0 += xtie * (c - 2)\n"
        "        x1 += xtie * (2 * c + 5)",
        "caught",
    ),
    (
        "OM8",
        "the two-sided normal survival is doubled",
        "    z = con_minus_dis / math.sqrt(var)\n"
        "    # Two-sided normal survival.\n"
        "    return float(2.0 * stats.norm.sf(abs(z)))",
        "    z = con_minus_dis / math.sqrt(var)\n"
        "    # Two-sided normal survival.\n"
        "    return float(2.0 * 2.0 * stats.norm.sf(abs(z)))",
        "caught",
    ),
    (
        "OM9",
        "the Kendall p-value uses 2*(1 - norm.sf(|z|)) instead of 2*norm.sf(|z|)",
        "    z = con_minus_dis / math.sqrt(var)\n"
        "    # Two-sided normal survival.\n"
        "    return float(2.0 * stats.norm.sf(abs(z)))",
        "    z = con_minus_dis / math.sqrt(var)\n"
        "    # Two-sided normal survival.\n"
        "    return float(2.0 * (1.0 - stats.norm.sf(abs(z))))",
        "caught",
    ),
    (
        "OM10",
        "the incomplete beta's arguments are swapped",
        "    x = dof / (dof + t * t)\n"
        "    return float(betainc(dof / 2.0, 0.5, x))",
        "    x = dof / (dof + t * t)\n"
        "    return float(betainc(0.5, dof / 2.0, x))",
        "caught",
    ),
    (
        "OM11",
        "the Pearson p-value uses the EXACT BETA form instead of the t form",
        "    t = abs(r) * math.sqrt((n - 2.0) / denom)\n"
        "    return ref_t_two_sided(t, n - 2.0)",
        "    a = n / 2.0 - 1.0\n"
        "    return 2.0 * float(betainc(a, a, (1.0 - abs(r)) / 2.0))",
        "caught",
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


def measure_inert(pristine, mutated):
    """PROVE that an inert mutation is inert, by MEASUREMENT.

    A mutation that is claimed to be inert must be shown to leave the
    quantity it touches BIT-IDENTICAL. This runs BOTH versions of the
    oracle's model over a grid of fixtures and compares the affected
    quantity. If the two differ anywhere, the "inert" claim is WRONG and
    the harness FAILS -- so a wrong inertness claim is itself caught.
    """
    grid = []
    import math
    import random
    rnd = random.Random(20261120)
    for n in (3, 5, 8, 12, 20, 35):
        for _ in range(40):
            x = [rnd.gauss(0.0, 1.0) for _ in range(n)]
            y = [rnd.gauss(0.0, 1.0) + 0.3 * xi for xi in x]
            grid.append((x, y, n))

    def collect(source):
        ns = {}
        exec(compile(source, "<oracle>", "exec"), ns)
        out = []
        for x, y, n in grid:
            r = ns["ref_pearson_r"](x, y)
            out.append(ns["ref_pearson_p"](r, n))
        return out

    a = collect(pristine)
    b = collect(mutated)
    worst = 0.0
    for u, v in zip(a, b):
        if u != u and v != v:
            continue
        if u != v:
            worst = max(worst, abs(u - v) / max(abs(u), 1e-300))
    return worst


def main():
    if not os.path.exists(PYTHON):
        print("MUT-HARNESS-BROKEN: %s not found" % PYTHON)
        return 2

    with open(ORACLE, "r") as fh:
        pristine = fh.read()

    # The pristine oracle MUST pass, or every verdict below is meaningless.
    tmpdir = tempfile.mkdtemp(prefix="ct_oracle_mut_")
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
        inert_ok = []
        inert_broken = []

        for mid, desc, search, repl, expectation in MUTATIONS:
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
                if expectation == "inert":
                    # THE INVERTED VERDICT. A mutation declared inert must
                    # be PROVEN inert by measurement: the affected quantity
                    # must be bit-identical with and without it. If it is
                    # NOT, the "inert" claim is wrong and the harness
                    # FAILS -- so a wrong inertness claim is itself
                    # caught.
                    try:
                        worst = measure_inert(pristine, mutated)
                    except Exception as exc:  # noqa: BLE001
                        print("%-4s MUT-HARNESS-BROKEN: the inertness probe raised: %s"
                              % (mid, exc))
                        broken.append(mid)
                        continue
                    if worst == 0.0:
                        print("%-4s INERT-OK  %s  (MEASURED: the affected "
                              "quantity is bit-identical)" % (mid, desc))
                        inert_ok.append(mid)
                    else:
                        print("%-4s INERT-BROKEN  %s  (the affected quantity "
                              "DIFFERS by %.3e relative -- the inertness "
                              "claim is WRONG)" % (mid, desc, worst))
                        inert_broken.append(mid)
                else:
                    print("%-4s SURVIVED  %s" % (mid, desc))
                    survived.append(mid)
            elif rc == 2:
                print("%-4s MUT-BROKEN (harness error)  %s" % (mid, desc))
                broken.append(mid)
            else:
                # rc == 1 means the oracle ran and reported failures.
                n_fail = 0
                m = re.search(r"oracle: (\d+) passed, (\d+) failed", out)
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
                            first = line.strip()[:160]
                            break
                    print("%-4s CAUGHT   %s   [%d failures]" % (mid, desc, n_fail))
                    if first:
                        print("         first failure: %s" % first)
                    caught.append(mid)

        print("")
        print("mutations: %d total, %d CAUGHT, %d SURVIVED, %d MUT-BROKEN, "
              "%d harness-defect" % (len(MUTATIONS), len(caught), len(survived),
                                     len(broken), len(broken)))
        print("inert (inverted verdict): %d INERT-OK, %d INERT-BROKEN"
              % (len(inert_ok), len(inert_broken)))
        if survived:
            print("SURVIVORS ARE HOLES IN THE SPECIFICATION:")
            for mid in survived:
                print("  %s" % mid)
        if broken:
            print("HARNESS DEFECTS (not verdicts):")
            for mid in broken:
                print("  %s" % mid)
        if inert_broken:
            print("INERTNESS CLAIMS THAT THE MEASUREMENT REFUSED:")
            for mid in inert_broken:
                print("  %s" % mid)
        return 1 if (survived or broken or inert_broken) else 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
