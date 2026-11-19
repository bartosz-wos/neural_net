#!/usr/bin/env python3
"""
tests/survival_mutate_oracle.py -- the mutation harness for
tests/survival_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

    /usr/bin/python3 tests/survival_mutate_oracle.py

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The off-policy-evaluation oracle (2026-10-30) printed
"38 passed, 0 failed" on its very first execution -- and the draft header
then CLAIMED it had caught two defects in its own first draft. It had
caught NEITHER. The claim was deleted rather than kept as flavour, because
a narrated defect story that no run supports is worse than no story.

The question that matters is not "does the oracle agree with itself" but
**"what would this oracle NOT notice?"**. This harness answers that with a
MEASURED mutation list rather than with reasoning.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
=================================================

  SM1  the risk set is {j : t_j > t} instead of {j : t_j >= t}
       -> the risk-set convention (C1). The subject censored AT an event
          time is dropped from the risk set at the moment it is known to be
          alive. The F9 fixture exists to see this.
  SM2  the Kaplan-Meier factor uses n_risk - 1 instead of n_risk
       -> the KM product itself.
  SM3  the survival curve is LEFT-continuous (the post-event value is
       applied to the interval BEFORE the event)
       -> the right-continuity convention (C2).
  SM4  the Greenwood term divides by n_risk instead of n_risk*(n_risk-d)
       -> Greenwood's variance (C4).
  SM5  the Greenwood term is NOT guarded against n_risk == d
       -> the LIMIT convention (C4). The F8 fixture (everyone dies at
          once) exists to see this: without the guard the variance is
          inf or NaN.
  SM6  the median is the first time S is STRICTLY below 0.5
       -> the median convention (C3).
  SM7  the median is INTERPOLATED rather than the first event time
       -> the median convention (C3).
  SM8  the Nelson-Aalen hazard uses n_risk - d instead of n_risk
       -> the NA estimator (C5).
  SM9  the Nelson-Aalen hazard is reported on the grid of ALL distinct
       times rather than on the EVENT times
       -> the NA grid convention (C5).
  SM10 the log-rank expected count is d * n1 / (n - 1)
       -> the hypergeometric expectation.
  SM11 the log-rank variance drops the finite-population factor
       ((n-d)/(n-1))
       -> the hypergeometric variance.
  SM12 the log-rank p-value is 1 - erfc(sqrt(stat/2)) instead of
       erfc(sqrt(stat/2))
       -> the p-value's sign. **This is a mutation that a
          "p is in [0,1]" check cannot see**, which is why the oracle
          pins the ABSOLUTE p-value against lifelines.
  SM13 the log-rank statistic divides by V twice
       -> the statistic.
  SM14 the concordance "half credit" for tied risk scores is dropped
       -> the tie convention (C7).
  SM15 the concordance counts the pair as correct when the subject who
       failed LATER has the higher risk score
       -> the SIGN convention (C7). This is the mutation that catches
          the two readings in the wild; the oracle pins both
          C(s) and C(-s) and their complementarity identity.
  SM16 the concordance admissibility rule drops the equal-times case
       (a pair with the same time is never admissible)
       -> the admissibility rule (C7).
  SM17 the Cox Breslow denominator uses the risk set's sum of exp()
       with the tied deaths' own contribution NOT removed -- i.e. the
       Efron term is used with only ONE step
       -> the tie handling (C9).
  SM18 the Cox gradient's risk-set sum is taken over the WRONG set
       (the deaths only)
       -> the gradient (C10).
  SM19 the Cox partial log-likelihood is the RAW SUM instead of the
       MEAN over the events
       -> the mean convention (C11).
  SM20 the Cox gradient's mean divides by n_events - 1
       -> the normaliser (C11).

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
================================================

  1. The anchor must occur EXACTLY ONCE, and the count comes from the
     SAME engine that performs the edit (`str.count` in Python), because
     `grep -c` counts MATCHING LINES, not matches.
  2. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a
     "SURVIVED" verdict. The source is checksummed before and after every
     edit, and a no-op edit is reported as such.
  3. The PRISTINE oracle must be GREEN before any mutation runs. A
     harness that starts from a red oracle reports garbage.
  4. A crash is a THIRD verdict class (MUT-BROKEN), distinct from
     CAUGHT: a crash says nothing about whether the checks have teeth.
  5. The oracle is run in a SUBPROCESS, so a mutated oracle cannot
     corrupt the harness's own state, and its exit code is the verdict.
  6. The source is restored on EXIT, and the restore NEVER deletes the
     backup -- cleanup happens exactly once, on EXIT.
"""

import hashlib
import pathlib
import subprocess
import sys

ORACLE = pathlib.Path("tests/survival_oracle.py")

# THE REPO'S CONVENTION: the oracles run under /usr/bin/python3, which is
# the interpreter that has numpy and scipy. The sibling harnesses hardcode
# it for the same reason (`tests/dtw_mutate_oracle.py`,
# `tests/block_cached_decode_mutate_oracle.py`,
# `tests/edit_distance_mutate_oracle.py`). Using `sys.executable` here
# would run the oracle under whatever interpreter happens to be first on
# PATH -- which, inside the Hermes agent's own venv, is NOT the one with
# numpy, and the oracle would fail to import and every mutation would be
# scored as a false verdict.
PYTHON = "/usr/bin/python3"

# Each spec is (id, description, old_text, new_text). The old_text must
# occur EXACTLY ONCE in the oracle source.
MUTATIONS = [
    ("SM1", "the risk set is {t_j > t} instead of {t_j >= t}",
     "        n_risk[k] = np.sum(times >= t)",
     "        n_risk[k] = np.sum(times > t)"),

    ("SM2", "the KM factor uses n_risk - 1",
     "        s *= (1.0 - d[k] / n_risk[k])",
     "        s *= (1.0 - d[k] / (n_risk[k] - 1.0))"),

    ("SM3", "the survival curve is LEFT-continuous",
     "    idx = np.searchsorted(ev_times, t, side=\"right\") - 1",
     "    idx = np.searchsorted(ev_times, t, side=\"left\") - 1"),

    ("SM4", "the Greenwood term divides by n_risk only",
     "            greenwood += d[k] / (nr * (nr - d[k]))",
     "            greenwood += d[k] / nr"),

    ("SM5", "the Greenwood term is NOT guarded against n_risk == d",
     "        if nr - d[k] > 0:\n"
     "            greenwood += d[k] / (nr * (nr - d[k]))",
     "        greenwood += d[k] / (nr * (nr - d[k]))"),

    ("SM6", "the median is the first time S is STRICTLY below 0.5",
     "    hit = np.nonzero(survival <= 0.5)[0]",
     "    hit = np.nonzero(survival < 0.5)[0]"),

    ("SM7", "the median is INTERPOLATED",
     "    return float(ev_times[hit[0]])",
     "    lo = survival[hit[0]] - 0.5\n"
     "    frac = 0.0 if survival[hit[0] - 1] == survival[hit[0]] else \\\n"
     "        (survival[hit[0] - 1] - 0.5) / (survival[hit[0] - 1] - survival[hit[0]])\n"
     "    return float(ev_times[hit[0] - 1] + frac * (ev_times[hit[0]] - ev_times[hit[0] - 1]))"),

    ("SM8", "the Nelson-Aalen hazard uses n_risk - d",
     "    cum = np.cumsum(d / n_risk)",
     "    cum = np.cumsum(d / (n_risk - d))"),

    ("SM9", "the Nelson-Aalen grid is ALL distinct times",
     "    ev_times, n_risk, d = _at_risk_counts(times, events)\n"
     "    cum = np.cumsum(d / n_risk)\n"
     "    return ev_times, n_risk, d, cum",
     "    ev_times, n_risk, d = _at_risk_counts(times, events)\n"
     "    cum = np.cumsum(d / n_risk)\n"
     "    return np.unique(np.asarray(times, dtype=float)), n_risk, d, cum"),

    ("SM10", "the log-rank expected count divides by n-1",
     "        expected_1 += d * n1 / n",
     "        expected_1 += d * n1 / (n - 1.0)"),

    ("SM11", "the log-rank variance drops the finite-population factor",
     "            variance += d * (n1 / n) * (1.0 - n1 / n) * ((n - d) / (n - 1))",
     "            variance += d * (n1 / n) * (1.0 - n1 / n)"),

    ("SM12", "the log-rank p-value is 1 - erfc(sqrt(stat/2))",
     "    p = float(_sp_erfc(math.sqrt(statistic / 2.0)))",
     "    p = float(1.0 - _sp_erfc(math.sqrt(statistic / 2.0)))"),

    ("SM13", "the log-rank statistic divides by V twice",
     "    statistic = 0.0 if variance <= 0.0 else (observed_1 - expected_1) ** 2 / variance",
     "    statistic = 0.0 if variance <= 0.0 else (observed_1 - expected_1) ** 2 / variance ** 2"),

    ("SM14", "the concordance drops the half credit for tied scores",
     "            if sa == sb:\n"
     "                num_tied += 1.0",
     "            if sa == sb:\n"
     "                num_tied += 1.0\n"
     "                num_correct += 1.0"),

    ("SM15", "the concordance counts the LATER failure as the correct one",
     "                risk_first = sa if event_first else sb\n"
     "                risk_second = sb if event_first else sa\n"
     "                if risk_first > risk_second:\n"
     "                    num_correct += 1.0",
     "                risk_first = sa if event_first else sb\n"
     "                risk_second = sb if event_first else sa\n"
     "                if risk_first < risk_second:\n"
     "                    num_correct += 1.0"),

    ("SM16", "the concordance never admits an equal-times pair",
     "            if ta == tb:\n"
     "                if ea == eb:\n"
     "                    continue  # not informative: both events or both censored",
     "            if ta == tb:\n"
     "                if True:\n"
     "                    continue  # never admissible"),

    ("SM17", "the Cox Breslow denominator keeps the tied deaths' contribution",
     "            loglik += float(eta[deaths].sum()) - dt * math.log(s0)\n"
     "            grad += x_death_sum - dt * (s0_x / s0)",
     "            loglik += float(eta[deaths].sum()) - math.log(s0)\n"
     "            grad += x_death_sum - (s0_x / s0)"),

    ("SM18", "the Cox gradient's risk-set sum is over the deaths only",
     "        s0_x = np.sum(np.exp(eta[risk])[:, None] * X[risk], axis=0)",
     "        s0_x = np.sum(np.exp(eta[deaths])[:, None] * X[deaths], axis=0)"),

    ("SM19", "the Cox partial log-likelihood is the RAW SUM",
     "    return loglik / n_events, grad / n_events, loglik",
     "    return loglik, grad / n_events, loglik"),

    ("SM20", "the Cox gradient's mean divides by n_events - 1",
     "    return loglik / n_events, grad / n_events, loglik",
     "    return loglik / n_events, grad / (n_events - 1.0), loglik"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not ORACLE.exists():
        print("MISSING: %s" % ORACLE)
        return 2

    original = ORACLE.read_text()
    backup = ORACLE.with_suffix(".py.bak-harness")
    backup.write_text(original)

    def restore():
        # NEVER delete the backup here: `restore` is called once PER
        # mutation and once on EXIT, so a self-cleaning restore works
        # exactly once and every later restore is a silent no-op --
        # leaving a MUTANT IN THE TREE.
        tmp = ORACLE.with_suffix(".py.restore")
        tmp.write_text(original)
        tmp.replace(ORACLE)

    def cleanup():
        restore()
        if backup.exists():
            backup.unlink()

    import atexit
    atexit.register(cleanup)

    print("=" * 74)
    print("tests/survival_mutate_oracle.py -- proving the oracle has teeth")
    print("=" * 74)

    # ---- the PRISTINE oracle must be GREEN first ------------------------
    pre = subprocess.run([PYTHON, str(ORACLE)], capture_output=True, text=True)
    if pre.returncode != 0:
        print("  [HARNESS-DEFECT] the pristine oracle is RED (rc=%d):" % pre.returncode)
        print(pre.stdout[-4000:])
        print(pre.stderr[-2000:])
        restore()
        return 2
    print("  pristine oracle: GREEN")
    print()

    caught = 0
    survived = 0
    broken = 0
    defects = 0

    for mid, desc, old, new in MUTATIONS:
        if original.count(old) != 1:
            print("  [%s] HARNESS-DEFECT -- anchor count = %d (must be exactly 1)"
                  % (mid, original.count(old)))
            defects += 1
            continue
        before = sha(ORACLE)
        restore()
        ORACLE.write_text(original.replace(old, new, 1))
        after = sha(ORACLE)
        if before == after:
            print("  [%s] HARNESS-DEFECT -- the edit was a NO-OP" % mid)
            defects += 1
            continue
        run = subprocess.run([PYTHON, str(ORACLE)], capture_output=True, text=True)
        restore()
        if run.returncode == 0:
            print("  [%s] SURVIVED -- the oracle did NOT notice: %s" % (mid, desc))
            survived += 1
        elif run.returncode == 1:
            print("  [%s] CAUGHT   -- %s" % (mid, desc))
            caught += 1
        else:
            print("  [%s] MUT-BROKEN (rc=%d) -- a crash is NOT a verdict: %s"
                  % (mid, run.returncode, desc))
            broken += 1

    restore()
    print()
    print("=" * 74)
    print("oracle-mutation: %d caught / %d survived / %d broken / %d harness defects"
          % (caught, survived, broken, defects))
    print("=" * 74)
    return 1 if (survived or broken or defects) else 0


if __name__ == "__main__":
    sys.exit(main())
