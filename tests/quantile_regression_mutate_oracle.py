#!/usr/bin/env python3
"""
tests/quantile_regression_mutate_oracle.py -- the mutation harness for
tests/quantile_regression_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The off-policy-evaluation oracle's first draft CLAIMED it
had caught two defects in itself, and it had caught NEITHER -- the claim
was deleted rather than kept as flavour.

The quantile-regression oracle reports "3215 checks passed, 0 failed", and
that number says only that the oracle agrees with the code beside it in the
same file. What it does NOT say is whether any of those 3215 checks would
notice if the MODEL were wrong. So the discriminating power is MEASURED
here, not narrated.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the pinball loss uses the OVER-prediction branch at e >= 0
       -> the tie convention. MUST be caught, because the tie convention is
          what makes the K = 1 layer bit-for-bit equal to the shipped
          `QuantileLoss`, and the oracle asserts that equality exactly.
  OM2  the gradient's over-prediction branch has the wrong sign
       -> the direction of the gradient.
  OM3  the loss is reduced by dividing by (N * K) instead of by N
       -> the reduction ("mean over the batch, sum over the levels"). MUST
          be caught, because the K = 1 equality pins it.
  OM4  the gradient is not divided by N
       -> the same reduction, on the gradient side.
  OM5  the gradient's level index is off by one
       -> the per-level association between a column and its level.
  OM6  the gradient mixes the LEVELS (every column uses level 0)
       -> the LEVEL DECOUPLING.
  OM7  the closed form `tau*u - min(u, 0)` is replaced by the two-sided
       piecewise form
       -> the loss's own definition. It MUST be caught by the quadrature
          cross-check, which is the definition-level reference.
  OM8  the quadrature's ORIENTATION is wrong for u < 0 (the sign of the
       integral is flipped)
       -> the definition-level cross-check itself. This is the mutation
          that the oracle's own first draft had; it MUST be caught, and its
          signature is that EVERY negative-u check fails by a factor of
          exactly -1.
  OM9  the inverse-ECDF quantile uses the WRONG index (floor instead of
       ceil)
       -> the closed form of the intercept-only minimiser. MUST be caught
          by the brute-force grid search.
  OM10 the grid comparison is made TWO-SIDED again (the closed form is
       required to EQUAL the grid's minimum)
       -> the one-sided comparison. This is the oracle's own first-draft
          bug: the grid cannot land on a kink, so the grid's minimum is
          strictly ABOVE the true minimum, and a two-sided equality fails
          by one grid step. MUST be caught -- by the ragged sample, where
          the closed form is strictly better.
  OM11 the crossing diagnostic counts ALL pairs instead of the adjacent ones
       -> the adjacency rule.
  OM12 the monotone rearrangement sorts DESCENDING
       -> the direction of the rearrangement.
  OM13 the monotone rearrangement sorts the COLUMNS instead of the ROWS
       -> the per-row rule.
  OM14 the total loss does NOT separate into per-level losses (every level
       uses level 0's tau)
       -> the level decoupling, on the LOSS side. **The first draft of this
          mutation DELETED a check instead of breaking a value, and it
          SURVIVED -- correctly.** A mutation that removes an assertion
          cannot make the oracle fail; it only makes the oracle quieter.
          The first draft of this harness therefore measured nothing about
          the oracle. The mutation now breaks the MODEL, which is what the
          check is there to catch.
  OM15 the training simulation's update is sign-flipped
       -> the derivation of the C++'s update rule. MUST be caught, because
          the simulated endpoint would then not reach the closed-form
          quantile.

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

TWO HARNESS DEFECTS THIS FILE FOUND IN ITS OWN FIRST DRAFT
==========================================================

Both were found by RUNNING it, not by reading it -- which is the point of
running it.

  1. **OM14's first draft DELETED A CHECK INSTEAD OF BREAKING A VALUE, and
     it SURVIVED.** The mutation removed an assertion, so the oracle had one
     fewer thing to fail on -- it could only ever get QUIETER. A mutation
     that removes the thing that would catch it measures nothing. The
     mutation now breaks the MODEL (every level uses level 0's tau, so the
     total loss no longer separates into the per-level losses), which is
     what the check is there to catch. It is CAUGHT with 17 failures.
     **RULE: A MUTATION MUST BREAK A VALUE THE CHECK COMPARES. DELETING A
     CHECK IS NOT A MUTATION.**

  2. **OM10's first draft had a FORMAT-STRING ARITY BUG, and it CRASHED
     instead of failing.** The replacement passed two arguments to a format
     string with one `%.2f` specifier, so the mutated oracle raised a
     TypeError on the very first check that used it -- rc=1 with ZERO
     reported failures. The harness correctly classified that as MUT-BROKEN
     rather than CAUGHT, because a crash says nothing about whether the
     checks have teeth. The replacement now carries both specifiers.
     **RULE: A MUTATION THAT MAKES THE ORACLE CRASH IS A HARNESS DEFECT,
     AND THE VERDICT CLASSES MUST KEEP IT SEPARATE FROM CAUGHT.**

RUN
===
/usr/bin/python3 tests/quantile_regression_mutate_oracle.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "quantile_regression_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# id | description | search | replacement
MUTATIONS = [
    ("OM1",
     "the pinball loss uses the OVER-prediction branch at e >= 0",
     "    return np.where(e >= 0.0, e * tau, e * (tau - 1.0))",
     "    return np.where(e >= 0.0, e * (tau - 1.0), e * tau)"),

    ("OM2",
     "the gradient's over-prediction branch has the wrong sign",
     "    return np.where(e >= 0.0, -tau, 1.0 - tau)",
     "    return np.where(e >= 0.0, -tau, tau - 1.0)"),

    ("OM3",
     "the loss is reduced by dividing by (N * K) instead of by N",
     "    return total / n",
     "    return total / (n * len(taus))"),

    ("OM4",
     "the gradient is not divided by N",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], tau) / n",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], tau)"),

    ("OM5",
     "the gradient's level index is off by one",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], tau) / n",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], taus[max(k - 1, 0)]) / n"),

    ("OM6",
     "the gradient mixes the LEVELS (every column uses level 0)",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], tau) / n",
     "        g[:, k] = rho_tau_grad_pred(e[:, k], taus[0]) / n"),

    ("OM7",
     "the closed form tau*u - min(u, 0) is replaced by a wrong one",
     "        close(tau * u - min(u, 0.0), rho_tau(u, tau), 1e-15,",
     "        close(tau * u + min(u, 0.0), rho_tau(u, tau), 1e-15,"),

    ("OM8",
     "the quadrature's ORIENTATION is wrong for u < 0",
     "            val = -val",
     "            val = val"),

    ("OM9",
     "the inverse-ECDF quantile uses floor instead of ceil",
     "    idx = int(np.ceil(n * tau)) - 1",
     "    idx = int(np.floor(n * tau)) - 1"),

    ("OM10",
     "the grid comparison is TWO-SIDED again",
     "    check(closed_val <= float(vals.min()) + 1e-12,\n"
     "          \"tau=%.2f: the closed-form quantile attains a value no greater than \"\n"
     "          \"the grid's minimum (closed %.17g, grid %.17g)\"\n"
     "          % (tau, closed_val, float(vals.min())))",
     "    check(abs(closed_val - float(vals.min())) <= 1e-12,\n"
     "          \"tau=%.2f: the closed-form quantile equals the grid minimum \"\n"
     "          \"(closed %.17g, grid %.17g)\"\n"
     "          % (tau, closed_val, float(vals.min())))"),

    ("OM11",
     "the crossing diagnostic counts ALL pairs instead of the adjacent ones",
     "        for j in range(k - 1):\n"
     "            if pred[i, j] > pred[i, j + 1]:\n"
     "                total += 1",
     "        for j in range(k - 1):\n"
     "            for l in range(j + 1, k):\n"
     "                if pred[i, j] > pred[i, l]:\n"
     "                    total += 1"),

    ("OM12",
     "the monotone rearrangement sorts DESCENDING",
     "        out[i, :] = np.sort(out[i, :])",
     "        out[i, :] = np.sort(out[i, :])[::-1]"),

    ("OM13",
     "the monotone rearrangement sorts the COLUMNS instead of the ROWS",
     "    for i in range(out.shape[0]):\n"
     "        out[i, :] = np.sort(out[i, :])",
     "    for j in range(out.shape[1]):\n"
     "        out[:, j] = np.sort(out[:, j])"),

    ("OM14",
     "the total loss does NOT separate into per-level losses (every level "
     "uses level 0's tau)",
     "    for k, tau in enumerate(taus):\n"
     "        total += float(np.sum(rho_tau(e[:, k], tau)))",
     "    for k, tau in enumerate(taus):\n"
     "        total += float(np.sum(rho_tau(e[:, k], taus[0])))"),

    ("OM15",
     "the training simulation's update is sign-flipped",
     "            b[k] -= lr * grad",
     "            b[k] += lr * grad"),
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

    workdir = tempfile.mkdtemp(prefix="qrmut_")
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
                print("%s SURVIVED  %s" % (mid, desc))
                print("      -> a hole in the SPECIFICATION, not a pass")
                SURVIVED += 1
            elif rc != 0 and failures == 0:
                print("%s MUT-BROKEN  %s  (rc=%d, no reported failures -- a CRASH, "
                      "not a verdict)" % (mid, desc, rc))
                BROKEN += 1
            else:
                print("%s CAUGHT  %s  [%d failures]" % (mid, desc, failures))
                CAUGHT += 1
    finally:
        try:
            for name in os.listdir(workdir):
                os.remove(os.path.join(workdir, name))
            os.rmdir(workdir)
        except OSError:
            pass

    print("")
    print("=== SUMMARY: %d caught, %d survived, %d broken, %d noop, %d anchor ==="
          % (CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD))
    if SURVIVED > 0:
        print("SURVIVORS ARE HOLES IN THE SPECIFICATION:")
        return 1
    if BROKEN > 0 or NOOP > 0 or ANCHOR_BAD > 0:
        print("HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
