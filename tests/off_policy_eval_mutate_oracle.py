#!/usr/bin/env python3
"""
tests/off_policy_eval_mutate_oracle.py -- MEASURE the oracle's discriminating
power by deliberately breaking the oracle and confirming it FAILS.

WHY THIS FILE EXISTS
====================

`tests/off_policy_eval_oracle.py` printed "38 passed, 0 failed" on its very
first execution. **A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS
OWN SENSITIVITY** -- it may simply be asserting things that are true of the
code it was written alongside.

So the oracle's discriminating power is MEASURED here rather than narrated:
each mutation below breaks the oracle in one specific way, and the check
passes only if the oracle FAILS with that mutation in place. The mutation is
then reverted and the oracle re-run to confirm it is green again.

This is the same discipline the C++ mutation harness applies to the C++
suite (`tests/mutate_off_policy_eval.sh`), applied one level up, to the
SPECIFICATION.

THE MUTATIONS
==============

  OM1  the enumeration draws the logged action from pi_1 (the TARGET
       policy) instead of pi_0 (the LOGGING policy). The weighted
       expectation then collapses to something that is NOT V_pi1, so the
       unbiasedness checks must FAIL.
  OM2  IPS sums instead of averaging (the `1/N` factor is dropped). Every
       expectation is then off by exactly the factor N.
  OM3  SNIS divides by N instead of by sum(rho), i.e. it silently returns
       IPS.
  OM4  the DR estimator drops the Vhat(s) term. The cancellation that makes
       DR doubly robust no longer happens, so ARM A must FAIL.
  OM5  the DR estimator flips the sign of the correction: rho * (Rhat - r).
  OM6  the effective sample size uses sum(rho^2)/sum(rho)^2 -- the
       INVERSE -- instead of (sum rho)^2 / sum rho^2.
  OM7  the importance weight is inverted: pi_0 / pi_1 instead of
       pi_1 / pi_0.

A mutation that the oracle SURVIVES is a hole in the specification, and this
script says so loudly rather than letting the green run stand unqualified.
"""

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tests" / "off_policy_eval_oracle.py"
PY = sys.executable

PASSED = 0
FAILED = 0


def check(ok, name, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}   {detail}")


def run_oracle(source_text):
    """Run the oracle with `source_text` as its body. Returns (rc, output)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source_text)
        path = f.name
    try:
        proc = subprocess.run([PY, path], capture_output=True, text=True,
                              timeout=300)
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


# The mutations: (id, description, old, new). The OLD text must occur
# EXACTLY ONCE in the oracle source -- the anchor count is checked here
# rather than assumed, because an anchor that occurs twice would edit an
# arbitrary one of the two.
MUTATIONS = [
    ("OM1", "the enumeration draws from the TARGET policy, not the logging policy",
     "                p *= pi0[a]\n            mass += p",
     "                p *= pi1[a]\n            mass += p"),
    ("OM2", "IPS sums instead of averaging",
     "    return float(np.sum(w * r) / r.size)",
     "    return float(np.sum(w * r))"),
    ("OM3", "SNIS divides by N instead of by sum(rho)",
     "    return float(np.sum(w * r) / np.sum(w))",
     "    return float(np.sum(w * r) / r.size)"),
    ("OM4", "the DR estimator drops the Vhat(s) term",
     "    return float(np.mean(v + w * (r - q)))",
     "    return float(np.mean(w * (r - q)))"),
    ("OM5", "the DR estimator flips the sign of the correction",
     "    return float(np.mean(v + w * (r - q)))",
     "    return float(np.mean(v + w * (q - r)))"),
    ("OM6", "the effective sample size is inverted",
     "    return float(np.sum(w) ** 2 / np.sum(w * w))",
     "    return float(np.sum(w * w) / np.sum(w) ** 2)"),
    ("OM7", "the importance weight is inverted",
     "    return target / behavior",
     "    return behavior / target"),
]


def main():
    original = ORACLE.read_text()

    # -----------------------------------------------------------------------
    # THE PRISTINE ORACLE MUST BE GREEN.
    # -----------------------------------------------------------------------
    rc, out = run_oracle(original)
    print(f"pristine oracle: rc={rc}")
    check(rc == 0, "the pristine oracle is green",
          out.strip().splitlines()[-1] if out.strip() else "")

    print("\n=== the oracle's discriminating power ===")
    for mid, desc, old, new in MUTATIONS:
        count = original.count(old)
        if count != 1:
            check(False, f"{mid} {desc}",
                  f"ANCHOR COUNT = {count} (must be exactly 1)")
            continue
        mutated = original.replace(old, new, 1)
        mrc, mout = run_oracle(mutated)
        # The mutated oracle MUST fail. A surviving mutation is a hole in
        # the specification.
        failed_lines = [ln for ln in mout.splitlines() if "[FAIL]" in ln]
        check(mrc != 0 and len(failed_lines) > 0,
              f"{mid} {desc}",
              f"the mutated oracle still PASSED (rc={mrc})")
        if failed_lines:
            print(f"        caught by: {failed_lines[0].strip()[:96]}")
        # And the ORIGINAL must still be green -- the mutation was not
        # written into the file on disk.
        prc, pout = run_oracle(original)
        if prc != 0:
            check(False, f"{mid} the pristine oracle is still green afterwards",
                  pout.strip().splitlines()[-1] if pout.strip() else "")
        else:
            print(f"        [PASS] {mid} the pristine oracle is still green "
                  f"afterwards")

    print(f"\n=== SUMMARY: {PASSED} passed, {FAILED} failed ===")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
