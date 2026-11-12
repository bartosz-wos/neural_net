#!/usr/bin/env python3
"""
tests/ordinal_regression_mutate_oracle.py -- the mutation harness for
tests/ordinal_regression_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The off-policy-evaluation oracle's first draft CLAIMED it
had caught two defects in itself, and it had caught NEITHER -- the claim
was deleted rather than kept as flavour.

The ordinal oracle is in exactly that position: it now reports
"229 checks passed, 0 failed", and that number says only that the oracle
agrees with the code beside it in the same file. What it does NOT say is
whether any of those 229 checks would notice if the MODEL were wrong.

So the discriminating power is MEASURED here, not narrated.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the interior class probability is the NAIVE difference of two
       cumulatives instead of the stable product
       -> the stable-form decision (FINDING 1). MUST be caught by F3, where
          the naive form is EXACTLY 0 and the truth is 2.685e-18.
  OM2  the stable product DROPS the e^{-v} factor
       -> the stable-form identity. This is the bug the oracle's own first
          draft had; it must be caught by the quadrature cross-check.
  OM3  the gradient is computed in PROBABILITY space instead of log space
       -> the log-space gradient decision (FINDING 2). MUST be caught by
          F3, where the probability-space derivative is EXACTLY 0 and the
          truth is about -1.
  OM4  the q term is dropped from the interior class's theta-partial
       -> the gap-dependence of an interior class's probability.
  OM5  the gap is recovered by DIFFERENCING the thetas instead of from the
       primitive
       -> the gap-absorbed decision (section [7]). MUST be caught by F4,
          where differencing gives exactly 0 and the truth is 3.7e-44.
  OM6  the ordering is NOT structural (the raw parameters are used as the
       cut points directly)
       -> the structural-ordering decision.
  OM7  the gap shift is credited to ONE class only (the second draft's
       error)
       -> the three-case d/dgap rule. MUST be caught by F1/F2, whose
          observed classes are not adjacent to the gaps being perturbed.
  OM8  the monotone direction is inverted (a higher eta moves mass towards
       LOWER classes)
       -> the monotonicity decision (section [6]).
  OM9  the "1 - c" factor is computed as (1.0 - c) instead of sigmoid(-a)
       -> the stable evaluation of the 1-c factor. This is the cancellation
          that FINDING 2 is about, applied to the derivative.
  OM10 the e_k factor uses the UNSTABLE 1 - c_k
       -> the same cancellation, on the other side.

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
===============================================

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
/usr/bin/python3 tests/ordinal_regression_mutate_oracle.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "ordinal_regression_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# id | description | search | replacement
MUTATIONS = [
    ("OM1",
     "the interior class probability is the NAIVE difference of two cumulatives",
     "        out[k] = (logsig(a[k]) + logsig(-a[k - 1])\n"
     "                  + _log1mexp_neg(gaps[k - 1]))",
     "        out[k] = logsig(a[k]) - logsig(a[k - 1])"),

    ("OM2",
     "the stable product DROPS the e^{-v} factor",
     "        out[k] = (logsig(a[k]) + logsig(-a[k - 1])\n"
     "                  + _log1mexp_neg(gaps[k - 1]))",
     "        out[k] = (logsig(a[k]) + logsig(a[k - 1])\n"
     "                  + _log1mexp_neg(gaps[k - 1]))"),

    ("OM3",
     "the gradient is computed in PROBABILITY space instead of log space",
     "        acc_eta = 0.0\n"
     "        for k in range(K):\n"
     "            if k == 0:\n"
     "                acc_eta += g[k] * (-e[0])\n"
     "            elif k == K - 1:\n"
     "                acc_eta += g[k] * (c[K - 2])\n"
     "            else:\n"
     "                acc_eta += g[k] * (c[k - 1] - e[k])",
     "        acc_eta = 0.0\n"
     "        for k in range(K):\n"
     "            if k == 0:\n"
     "                acc_eta += g[k] * (-c[0] * (1.0 - c[0]))\n"
     "            elif k == K - 1:\n"
     "                acc_eta += g[k] * (c[K - 2] * (1.0 - c[K - 2]))\n"
     "            else:\n"
     "                acc_eta += g[k] * (c[k - 1] * (1.0 - c[k - 1]) "
     "- c[k] * (1.0 - c[k]))"),

    ("OM4",
     "the q term is dropped from the interior class's gap coefficient",
     "            special[k] += g[k] * (e[k] + q[k - 1])",
     "            special[k] += g[k] * (e[k])"),

    ("OM5",
     "the gap is recovered by DIFFERENCING the thetas instead of the primitive",
     "    return [softplus(float(rk)) for rk in r[1:]]",
     "    th = thresholds_from_raw(r)\n"
     "    return [th[i + 1] - th[i] for i in range(len(th) - 1)]"),

    ("OM6",
     "the ordering is NOT structural: the raw parameters are the cut points",
     "    theta = []\n"
     "    prev = 0.0\n"
     "    for i, ri in enumerate(r):\n"
     "        if i == 0:\n"
     "            prev = float(ri)\n"
     "        else:\n"
     "            prev = prev + softplus(float(ri))\n"
     "        theta.append(prev)\n"
     "    return theta",
     "    return [float(ri) for ri in r]"),

    ("OM7",
     "the gap shift is credited to ONE class only",
     "    for j in range(K - 2):\n"
     "        dL_dgap[j] += special[j + 1]",
     "    for j in range(K - 2):\n"
     "        dL_dgap[j] += special[j + 1]\n"
     "    for j in range(K - 2):\n"
     "        dL_dgap[j] = special[j + 1] if j + 1 < K - 1 else dL_dgap[j]"),

    ("OM8",
     "the monotone direction is inverted",
     "                acc_eta += g[k] * (c[k - 1] - e[k])",
     "                acc_eta += g[k] * (e[k] - c[k - 1])"),

    ("OM9",
     "the e_k factor uses the UNSTABLE (1.0 - c_k)",
     "        e = [sigmoid(-v) for v in a]         # e_j = 1 - c_j, computed STABLY",
     "        e = [1.0 - c[j] for j, v in enumerate(a)]"),

    ("OM10",
     "the bulk coefficient uses the UNSTABLE (1.0 - c_k)",
     "            bulk[k] += g[k] * (e[k] - c[k - 1])",
     "            bulk[k] += g[k] * ((1.0 - c[k]) - c[k - 1])"),
]


def run_oracle(target):
    p = subprocess.run([PYTHON, target], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    out = p.stdout or ""
    failures = out.count("[FAIL]")
    return p.returncode, failures, out


def main():
    global CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD

    with open(ORACLE, "r") as f:
        pristine = f.read()

    rc, failures, out = run_oracle(ORACLE)
    if rc != 0 or failures != 0:
        print("THE PRISTINE ORACLE IS NOT GREEN (rc=%d, failures=%d)." % (rc, failures))
        print(out[-4000:])
        return 1
    print("pristine oracle: green")
    print("")

    workdir = tempfile.mkdtemp(prefix="ordmut_")
    try:
        for mid, desc, search, repl in MUTATIONS:
            count = pristine.count(search)
            if count != 1:
                print("%s MUT-ANCHOR  the anchor occurs %d times (must be 1)" % (mid, count))
                ANCHOR_BAD += 1
                continue

            target = os.path.join(workdir, "mutated_oracle.py")
            with open(target, "w") as f:
                f.write(pristine.replace(search, repl))

            with open(target, "r") as f:
                if f.read() == pristine:
                    print("%s MUT-NOOP  the mutation did not change the file" % mid)
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
