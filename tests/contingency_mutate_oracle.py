#!/usr/bin/env python3
"""
tests/contingency_mutate_oracle.py -- the mutation harness for
tests/contingency_oracle.py.

WHY THIS FILE EXISTS

An oracle is code you just wrote. A green oracle proves only that the
oracle agrees with the implementation you wrote beside it; it may be
asserting things that are true of BOTH, and a spec that cannot fail
cannot certify anything. So the oracle's DISCRIMINATING POWER is measured
here, the same way the C++ suite's is measured by
tests/mutate_contingency.sh.

THE RULES THIS HARNESS FOLLOWS

  * The PRISTINE oracle must pass first. A verdict reported against a
    broken baseline is meaningless.
  * An anchor that occurs MORE THAN ONCE is REFUSED, not edited: picking
    one of two occurrences at random is a coin flip, not a verdict.
  * An edit that did not apply is reported as MUT-NOOP, which is a
    HARNESS DEFECT -- not a coverage gap.
  * MUT-BROKEN (the oracle failed to run at all) is scored SEPARATELY
    from a real failure. A crash is not evidence that the check works.
  * Every mutation lands on a DIFFERENT CONVENTION, and each one's
    discriminating fixture is named.

RUN IT

    /usr/bin/python3 tests/contingency_mutate_oracle.py

Exit code 0 iff every mutation was CAUGHT.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ORACLE = "tests/contingency_oracle.py"
PYTHON = "/usr/bin/python3"

# ---------------------------------------------------------------------------
# THE MUTATIONS
#
# Each entry is (id, anchor, replacement, description, discriminating
# fixture). The anchor must be UNIQUE in the oracle source -- the harness
# refuses to edit an ambiguous one.
# ---------------------------------------------------------------------------
MUTATIONS = [
    # --- the chi-square survival function ---------------------------------
    ("M1",
     '    if degrees_of_freedom == 1.0:\n'
     '        return math.erfc(math.sqrt(statistic / 2.0))\n'
     '    return gammq_model(degrees_of_freedom / 2.0, statistic / 2.0)',
     '    if degrees_of_freedom == 2.0:\n'
     '        return math.erfc(math.sqrt(statistic / 2.0))\n'
     '    return gammq_model(degrees_of_freedom / 2.0, statistic / 2.0)',
     "the df=1 path is claimed for df=2 (the closed form only exists for df=1)",
     "F1 (df=2 entry)"),

    ("M2",
     '    if statistic <= 0.0:\n        return 1.0\n',
     '    if statistic <= 0.0:\n        return 0.0\n',
     "the statistic<=0 guard returns 0.0 instead of 1.0",
     "the F1 chi2_sf(0, 3) == 1 check"),

    ("M3",
     '    if x < a + 1.0:\n        return 1.0 - gser_model(a, x)\n'
     '    return gcf_model(a, x)',
     '    if x < a + 1.0:\n        return gser_model(a, x)\n'
     '    return gcf_model(a, x)',
     "the series branch returns P(a,x) instead of Q(a,x) -- the "
     "survival function is INVERTED on the series branch",
     "F1 (the general-df entries)"),

    # --- Cohen's kappa -----------------------------------------------------
    ("M4",
     '    elif weights == "linear":\n'
     '        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))\n'
     '    elif weights == "quadratic":\n'
     '        w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2',
     '    elif weights == "linear":\n'
     '        w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2\n'
     '    elif weights == "quadratic":\n'
     '        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))',
     "the linear and quadratic weight matrices are SWAPPED",
     "F2 (the linear and quadratic kappa values)"),

    ("M5",
     '    expected = np.outer(row, col) / n\n'
     '    expected_agreement = float(np.trace(expected) / n)',
     '    expected = np.outer(col, row) / n\n'
     '    expected_agreement = float(np.trace(expected) / n)',
     "the expected matrix is TRANSPOSED (sklearn's own convention; the "
     "weights are symmetric so it should NOT change the answer -- if this "
     "mutation is CAUGHT the symmetry argument is wrong)",
     "F2 (the kappa values -- INERT by construction)"),

    ("M6",
     '    if denominator == 0.0:\n        kappa = float("nan")\n'
     '    else:\n        kappa = 1.0 - numerator / denominator',
     '    if denominator == 0.0:\n        kappa = 0.0\n'
     '    else:\n        kappa = 1.0 - numerator / denominator',
     "the DEGENERATE kappa is 0.0 instead of NaN (sklearn's convention is "
     "NaN, and the MCC's is 0.0 -- they are NOT interchangeable)",
     "F3 (the degenerate kappa)"),

    ("M7",
     '    if weights == "unweighted":\n'
     '        w = np.ones((k, k))\n'
     '        np.fill_diagonal(w, 0.0)',
     '    if weights == "unweighted":\n'
     '        w = np.ones((k, k))\n'
     '        np.fill_diagonal(w, 1.0)',
     "the unweighted weight matrix keeps its diagonal (so the diagonal "
     "counts as agreement AND the off-diagonal counts as disagreement)",
     "F2 (the unweighted kappa)"),

    # --- the Matthews correlation coefficient ------------------------------
    ("M8",
     '    if denom == 0.0:\n        mcc = 0.0\n'
     '    else:\n        mcc = cov_ytyp / math.sqrt(denom)',
     '    if denom == 0.0:\n        mcc = float("nan")\n'
     '    else:\n        mcc = cov_ytyp / math.sqrt(denom)',
     "the DEGENERATE MCC is NaN instead of 0.0 (the OPPOSITE of the kappa "
     "convention -- this is the pair the suite exists to pin)",
     "F5 (the degenerate MCC)"),

    ("M9",
     '    cov_ytyp = n_correct * n - float(np.dot(t_sum, p_sum))',
     '    cov_ytyp = n_correct * n - float(np.dot(p_sum, p_sum))',
     "cov_ytyp uses the PREDICTED marginal twice instead of the "
     "true/predicted dot product",
     "F4 (the MCC value)"),

    # --- the chi-square test -----------------------------------------------
    ("M10",
     '        magnitude = np.minimum(0.5, np.abs(diff))\n'
     '        adjusted = table + magnitude * np.sign(diff)',
     '        magnitude = np.full(diff.shape, 0.5)\n'
     '        adjusted = table + magnitude * np.sign(diff)',
     "the Yates correction is UNCLAMPED (the textbook form; scipy and R "
     "both clamp it to the cell's own deviation)",
     "F8 (the clamp fixture)"),

    ("M11",
     '    if correct and r == 2 and c == 2:\n'
     '        diff = expected - table',
     '    if correct and r >= 2 and c >= 2:\n'
     '        diff = expected - table',
     "the Yates correction is applied to EVERY table size, not just 2x2",
     "F7 (the 2x3 table)"),

    ("M12",
     '    dof = (r - 1) * (c - 1)\n\n    adjusted = table',
     '    dof = r * c - 1\n\n    adjusted = table',
     "the degrees of freedom are the CELL COUNT minus one (scipy's "
     "power_divergence convention) instead of (R-1)(C-1)",
     "F7 (the 2x3 df)"),

    ("M13",
     '    statistic = float(((adjusted - expected) ** 2 / expected).sum())',
     '    statistic = float((np.abs(adjusted - expected) ** 2 / expected).sum())',
     "the chi-square statistic is ABSOLUTED before squaring -- this is "
     "ALGEBRAICALLY IDENTICAL to squaring the signed deviation, so it is "
     "INERT BY CONSTRUCTION and MUST survive",
     "F6 and F7 (INERT -- see INERT_MUTATIONS)"),

    # --- the G-test ---------------------------------------------------------
    ("M14",
     '            if o > 0:\n                s += o * _log(o / expected[i, j])',
     '            s += o * _log(o / expected[i, j])',
     "the ZERO-CELL GUARD is removed, so 0 * log(0/E) = NaN destroys the "
     "whole statistic. The oracle's _log is a C++-like log that returns "
     "-inf for 0, so the mutated oracle produces NaN rather than raising "
     "-- which is exactly what the C++ produces.",
     "F9 (the zero-cell table)"),

    ("M15",
     '    statistic = 2.0 * s\n    dof = (r - 1) * (c - 1)',
     '    statistic = 1.0 * s\n    dof = (r - 1) * (c - 1)',
     "the G-test's factor of 2 is dropped",
     "F9 (the G-test value)"),

    # --- McNemar ------------------------------------------------------------
    ("M16",
     '    elif correct and n01 != n10:\n'
     '        statistic = (abs(n01 - n10) - 1.0) ** 2 / n_disc',
     '    elif correct:\n'
     '        statistic = (abs(n01 - n10) - 1.0) ** 2 / n_disc',
     "the continuity correction is applied EVEN ON A SYMMETRIC TABLE "
     "(R skips it there, so the statistic is 0)",
     "F11 (the symmetric table)"),

    ("M17",
     '        statistic = (abs(n01 - n10) - 1.0) ** 2 / n_disc',
     '        statistic = (n01 - n10 - 1.0) ** 2 / n_disc',
     "the ABSOLUTE VALUE is dropped from the McNemar numerator (R uses "
     "abs, so n01 < n10 gives a different answer)",
     "F12 (n01 < n10)"),

    ("M18",
     '    if n_disc == 0:\n        statistic = float("nan")\n'
     '        p_value = float("nan")',
     '    if n_disc == 0:\n        statistic = 0.0\n'
     '        p_value = 1.0',
     "the no-discordant-pairs case is 0 (not NaN); R reports NaN for both "
     "the statistic and the p-value",
     "F13 (the no-discordant fixture)"),

    # --- Fisher's exact test ------------------------------------------------
    ("M19",
     '        if p <= pexact * gamma:\n            total += p',
     '        if p <= pexact:\n            total += p',
     "the probability ordering drops scipy's 1 + 1e-14 relative tolerance, "
     "so a table whose pmf ties the observed one exactly is EXCLUDED",
     "F16 (the exact-tie fixture)"),

    ("M20",
     '    lo = max(0, n - (m - n1))\n    hi = min(n1, n)\n'
     '    total = 0.0\n'
     '    for x in range(lo, hi + 1):\n'
     '        p = _hypergeom_pmf(x, m, n1, n)\n'
     '        if p <= pexact * gamma:\n'
     '            total += p\n'
     '    return {"p_value": min(total, 1.0), "odds_ratio": odds_ratio}',
     '    lo = max(0, n - (m - n1))\n    hi = min(n1, n)\n'
     '    k = min(a, c)\n'
     '    total = 0.0\n'
     '    for x in range(lo, hi + 1):\n'
     '        p = _hypergeom_pmf(x, m, n1, n)\n'
     '        if p <= pexact * gamma:\n'
     '            total += p\n'
     '    doubled = min(1.0, 2.0 * min(total, 1.0 - total))\n'
     '    return {"p_value": doubled, "odds_ratio": odds_ratio}',
     "the two-sided p-value becomes a DOUBLED one-sided tail instead of the "
     "probability ordering (R and scipy use the ordering)",
     "F14 and F15"),

    ("M21",
     '    if c > 0 and b > 0:\n'
     '        odds_ratio = float(a * d / (c * b))\n'
     '    else:\n'
     '        odds_ratio = float("inf")',
     '    if c > 0 and b > 0:\n'
     '        odds_ratio = float(a * c / (b * d))\n'
     '    else:\n'
     '        odds_ratio = float("inf")',
     "the odds ratio is the WRONG cross-product (a*c/(b*d) instead of "
     "a*d/(b*c))",
     "F14 and F15 (the odds ratio)"),
]

# ---------------------------------------------------------------------------
# THE INERT MUTATIONS
#
# A mutation that PROVABLY cannot change anything is not a coverage gap --
# it is a bad mutation. But it must still be MEASURED, not asserted, and
# the harness verdict is INVERTED for these: the mutation is expected to
# SURVIVE, so surviving is the PASS and being caught is the FAILURE.
#
# That inversion matters: without it, an "inert" mutation that started to
# matter would be scored as a success, and the harness would report a
# clean bill of health for a suite that had just gone blind.
#
# Each entry is (id, why-it-is-inert, how-to-measure-it).
# ---------------------------------------------------------------------------
INERT_MUTATIONS = {
    "M5": (
        "the expected matrix is transposed. Every weight matrix this module "
        "supports (the off-diagonal indicator, |i-j|, (i-j)^2) is SYMMETRIC, "
        "and for symmetric w the sums sum_ij w_ij E_ij and sum_ij w_ij E_ji "
        "are EQUAL by relabelling -- so the transposition cannot change the "
        "kappa. (This is the plan's C4 argument.)",
        "the mutation is caught ONLY IF the kappa values change, so a CAUGHT "
        "verdict means the symmetry argument is WRONG and must be "
        "investigated, not celebrated."),
    "M13": (
        "the chi-square deviation is ABSOLUTED before squaring. |x|^2 == x^2 "
        "for every real x, so this is the same arithmetic written differently.",
        "as above: CAUGHT means the fixture set lost the power to compare "
        "the two forms, which is impossible, so a CAUGHT verdict here means "
        "the harness is mis-scoring."),
}


def run_oracle(path):
    """Run the oracle at `path`. Returns (exit_code, stdout)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    proc = subprocess.run([PYTHON, path], capture_output=True, text=True,
                          env=env, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if not os.path.exists(ORACLE):
        print("the oracle %s does not exist" % ORACLE)
        return 1

    print("=" * 74)
    print("tests/contingency_mutate_oracle.py -- the oracle mutation harness")
    print("=" * 74)
    print()

    # --- the pristine baseline MUST pass -----------------------------------
    print("THE PRISTINE BASELINE (must pass, or every verdict below is void):")
    rc, out = run_oracle(ORACLE)
    if rc != 0:
        print("  [FAIL] the pristine oracle does NOT pass -- aborting")
        print(out[-4000:])
        return 1
    print("  [PASS] the pristine oracle passes")
    print()

    results = []
    src_original = open(ORACLE).read()

    for mut_id, anchor, replacement, description, fixture in MUTATIONS:
        # --- the anchor must be UNIQUE ------------------------------------
        count = src_original.count(anchor)
        if count != 1:
            print("  [HARNESS-DEFECT] %s: the anchor occurs %d times -- "
                  "REFUSING to edit (a coin flip is not a verdict)"
                  % (mut_id, count))
            results.append((mut_id, "HARNESS-DEFECT", description))
            continue

        mutated = src_original.replace(anchor, replacement, 1)
        if mutated == src_original:
            print("  [HARNESS-DEFECT] %s: the edit did not apply" % mut_id)
            results.append((mut_id, "MUT-NOOP", description))
            continue

        tmp_dir = tempfile.mkdtemp(prefix="contingency_oracle_mut_")
        tmp_path = os.path.join(tmp_dir, "contingency_oracle.py")
        try:
            with open(tmp_path, "w") as fh:
                fh.write(mutated)
            rc, out = run_oracle(tmp_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if rc == 0:
            verdict = "SURVIVED"
            detail = "the mutated oracle STILL PASSES"
        elif "Traceback" in out or "Error" in out:
            verdict = "MUT-BROKEN"
            detail = "the mutated oracle failed to RUN (not evidence)"
        else:
            verdict = "CAUGHT"
            detail = "the mutated oracle FAILED a check"

        # --- the INVERTED verdict for a provably-inert mutation -------------
        if mut_id in INERT_MUTATIONS:
            why, how = INERT_MUTATIONS[mut_id]
            if verdict == "SURVIVED":
                verdict = "INERT-OK"
                detail = ("the mutation survived, as it MUST: it is provably "
                          "inert. (inverted verdict: surviving is the PASS)")
            elif verdict == "CAUGHT":
                verdict = "INERT-BROKEN"
                detail = ("the mutation was CAUGHT, but it is provably inert "
                          "-- so the harness is mis-scoring, or the "
                          "inertness argument is wrong. INVESTIGATE.")
            else:
                verdict = "INERT-BROKEN"
                detail = "the mutation did not run cleanly; investigate."
            results.append((mut_id, verdict, description))
            print("  [%-13s] %s -- %s" % (verdict, mut_id, description))
            print("                 fixture: %s" % fixture)
            print("                 INERT: %s" % why)
            print("                 %s" % detail)
            print("                 %s" % how)
            continue

        results.append((mut_id, verdict, description))
        print("  [%-13s] %s -- %s" % (verdict, mut_id, description))
        print("                 fixture: %s" % fixture)
        print("                 %s" % detail)
        if verdict == "CAUGHT":
            for line in out.split("\n"):
                if "[FAIL]" in line:
                    print("                 %s" % line.strip())

    print()
    caught = sum(1 for _, v, _ in results if v == "CAUGHT")
    survived = sum(1 for _, v, _ in results if v == "SURVIVED")
    broken = sum(1 for _, v, _ in results if v == "MUT-BROKEN")
    defect = sum(1 for _, v, _ in results if v in ("MUT-NOOP", "HARNESS-DEFECT"))
    inert_ok = sum(1 for _, v, _ in results if v == "INERT-OK")
    inert_broken = sum(1 for _, v, _ in results if v == "INERT-BROKEN")
    print("=" * 74)
    print("mutations: %d total, %d CAUGHT, %d SURVIVED, %d MUT-BROKEN, "
          "%d harness-defect" % (len(results), caught, survived, broken, defect))
    print("inert (inverted verdict): %d INERT-OK, %d INERT-BROKEN"
          % (inert_ok, inert_broken))
    print("=" * 74)

    if survived:
        print()
        print("A SURVIVED MUTATION IS A DIAGNOSTIC ABOUT THE FIXTURE, NOT A")
        print("VERDICT ABOUT THE ORACLE. Ask: does any fixture VARIE in the")
        print("dimension the mutation targets?")
    if broken:
        print()
        print("MUT-BROKEN means the mutated oracle CRASHED. That is not")
        print("evidence that the check has teeth -- a crash and a wrong")
        print("answer are different things, and only the latter counts.")

    return 1 if (survived or broken or defect or inert_broken) else 0


if __name__ == "__main__":
    sys.exit(main())
