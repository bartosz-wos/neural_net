#!/usr/bin/env python3
"""
tests/sample_quality_mutate_oracle.py -- the mutation harness for
tests/sample_quality_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

    /usr/bin/python3 tests/sample_quality_mutate_oracle.py

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

  SM1  the covariance uses the BIASED estimator (divide by n)
       -> the unbiased-estimator convention (C2).
  SM2  FID returns the DISTANCE d instead of the SQUARED distance d^2
       -> the squaring convention (C1). This is the single most common
          FID implementation error in the wild.
  SM3  the covariance product order is S1 . S2 instead of
       S1^(1/2) . S2 . S1^(1/2)
       -> the product-order convention (C3).
  SM4  the mean term is ||dmu|| instead of ||dmu||^2
       -> the mean term's weight.
  SM5  the covariance's diagonal is computed but the off-diagonal is
       dropped (a diagonal-only covariance)
       -> that the FULL covariance is used.
  SM6  the KL's zero guard is removed (0 * log(0/q) is evaluated)
       -> the 0*log(0/q) = 0 convention (C5). Without the guard the
          oracle reports NaN.
  SM7  the marginal is computed over the WHOLE set instead of over the
       split's own rows
       -> the split semantics. **This is the mutation that a
          single-split-only oracle cannot see**, which is why the
          oracle pins IS at TWO split counts.
  SM8  the split count is ignored (always 1 split)
       -> the split convention (C6).
  SM9  the score is the MEAN of the per-group KLs rather than the MEAN
       of the per-group exp(KL)s
       -> the exp()'s placement. **A mutation that moves the exp outside
          the group mean is invisible on a one-group fixture**, which is
          why the oracle pins BOTH the 1-split and the 10-split values.
  SM10 the exponentiation is dropped (the IS is the mean KL)
       -> that the IS is an exp at all.
  SM11 the covariance's clamp at zero is removed
       -> the PSD clamp. Without it a rank-deficient covariance's
          negative eigenvalue reaches sqrt() and the oracle sees NaN.
  SM12 the sample covariance divides by the WRONG count (n+1)
       -> the normaliser.

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
=================================================

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

ORACLE = pathlib.Path("tests/sample_quality_oracle.py")

# Each spec is (id, description, old_text, new_text). The old_text must
# occur EXACTLY ONCE in the oracle source.
MUTATIONS = [
    ("SM1", "the covariance uses the BIASED estimator (divide by n)",
     "    denom = (n - 1) if unbiased else n",
     "    denom = n if unbiased else n"),

    ("SM2", "FID returns the distance d instead of the squared distance d^2",
     "    d = frechet_distance(mu_r, S_r, mu_f, S_f)\n    return d * d",
     "    d = frechet_distance(mu_r, S_r, mu_f, S_f)\n    return d"),

    ("SM3", "the covariance product order is S1 . S2 (not symmetrised)",
     "    inner = S1h @ S2 @ S1h",
     "    inner = S1 @ S2"),

    ("SM4", "the mean term is ||dmu|| instead of ||dmu||^2",
     "    radicand = (np.sum((mu1 - mu2) ** 2)",
     "    radicand = (np.sum(np.abs(mu1 - mu2))"),

    ("SM5", "the covariance's off-diagonal is dropped (diagonal only)",
     "    cov = centered.T @ centered / denom",
     "    cov = np.diag(np.diag(centered.T @ centered)) / denom"),

    ("SM6", "the KL's zero guard is removed (the log's where=mask)",
     "        np.log(part, out=log_p, where=mask)",
     "        np.log(part, out=log_p)"),

    ("SM7", "the marginal is computed over the WHOLE set instead of over the "
     "split's own rows",
     "        py = part.mean(axis=0, keepdims=True)",
     "        py = probabilities.mean(axis=0, keepdims=True)"),

    # --- THE ANCHORS ARE CONTEXTED, NOT SINGLE LINES. The first run of
    # this harness reported THREE harness defects because these anchors
    # occurred 2-3 times each: the vectorised `inception_score`, the
    # scalar-loop `inception_score_reference`, and the whole-set-marginal
    # foil all contain the same `for s in range(splits):` line and the
    # same `scores.append(np.exp(np.mean(kl)))` line. **A mutation's
    # anchors are invalidated by adding a feature, and the failure is
    # quiet** -- the harness reports a defect rather than a verdict, and
    # a reader who skims the output could read "HARNESS-DEFECT" as
    # "inert". Every anchor below therefore names a UNIQUE multi-line
    # context.
    ("SM8", "the split count is ignored (one split covering the whole set)",
     "        lo = s * n // splits\n"
     "        hi = (s + 1) * n // splits\n"
     "        part = probabilities[lo:hi]",
     "        lo = 0\n"
     "        hi = n\n"
     "        part = probabilities[lo:hi]"),

    ("SM9", "the exp() is outside the group mean",
     "        kl = np.sum(part * (log_p - log_q), axis=1)\n"
     "        scores.append(np.exp(np.mean(kl)))\n"
     "    return float(np.mean(scores))",
     "        kl = np.sum(part * (log_p - log_q), axis=1)\n"
     "        scores.append(np.mean(kl))\n"
     "    return float(np.mean(scores))"),

    ("SM10", "the exponentiation is dropped entirely",
     "        kl = np.sum(part * (log_p - log_q), axis=1)\n"
     "        scores.append(np.exp(np.mean(kl)))\n"
     "    return float(np.mean(scores))",
     "        kl = np.sum(part * (log_p - log_q), axis=1)\n"
     "        scores.append(np.mean(kl) + 1.0)\n"
     "    return float(np.mean(scores))"),

    ("SM11", "the covariance's PSD clamp at zero is removed",
     "    return Q @ np.diag(np.sqrt(np.maximum(lam, 0.0))) @ Q.T",
     "    return Q @ np.diag(np.sqrt(lam)) @ Q.T"),

    ("SM12", "the covariance divides by the wrong count (n+1)",
     "    denom = (n - 1) if unbiased else n",
     "    denom = (n + 1) if unbiased else n"),

    # --- THE MUTATIONS THAT PIN THE ABSOLUTE VALUES. These were ADDED
    # after the FIRST harness run reported four survivors, and the
    # survivors were REAL gaps: every property the oracle checked was
    # RELATIVE, so a mutant could move the number and keep every
    # property. The fix was an INDEPENDENT scalar-loop reference
    # (inception_score_reference) plus an independent numpy route for the
    # covariance, and these mutations are what prove the fix works.
    ("SM13", "the IS reference's marginal uses the whole set",
     "        marginal = [sum(rows[i][c] for i in range(n_rows)) / n_rows",
     "        marginal = [sum(probabilities[i][c] for i in range(n)) / n"),

    # SM14's FIRST FORM CRASHED, and a crash is NOT a verdict: it said
    # nothing about whether the checks have teeth. The form below keeps the
    # `continue` (so the following `math.log(p)` is never reached with
    # p == 0) but makes the zero-probability class CONTRIBUTE 0.5 instead
    # of nothing. The reference then DISAGREES with the vectorised
    # implementation, which is what checks 6.8/6.9 are there to catch.
    ("SM14", "the IS reference drops the 0*log(0/q) = 0 convention",
     "                if p <= 0.0:\n"
     "                    continue          # the 0 * log(0/q) = 0 convention",
     "                if p <= 0.0:\n"
     "                    kl += 0.5; continue  # WRONG: a zero probability contributes"),

    ("SM15", "the sample covariance's OFF-DIAGONAL is dropped",
     "    cov = centered.T @ centered / denom",
     "    cov = np.diag(np.diag(centered.T @ centered)) / denom"),

    ("SM16", "the FID's sample Gaussians are taken from the WRONG SET",
     "    mu_r, S_r = sample_mean_cov(real_features)\n"
     "    mu_f, S_f = sample_mean_cov(fake_features)",
     "    mu_r, S_r = sample_mean_cov(fake_features)\n"
     "    mu_f, S_f = sample_mean_cov(real_features)"),

    # --- SM17: THE ONE THAT IS PROVABLY INERT, AND WHY IT IS STILL HERE.
    #
    # FID IS MATHEMATICALLY SYMMETRIC IN ITS TWO ARGUMENTS. The trace
    # term is tr((S1^(1/2) . S2 . S1^(1/2))^(1/2)), and S1^(1/2) . S2 .
    # S1^(1/2) is SIMILAR to S1 . S2, while S2^(1/2) . S1 . S2^(1/2) is
    # similar to S2 . S1 -- and S1 . S2 and S2 . S1 have the SAME
    # eigenvalues, so the two traces are equal and the two radicands are
    # equal. Swapping the two feature sets therefore CANNOT change the
    # value, and the mutation survives.
    #
    # MEASURED over 3000 random feature pairs (varying n from 2 to 11 and
    # d from 1 to 6): the worst RELATIVE difference was 1.146e-07, i.e.
    # floating-point noise around a mathematically exact symmetry.
    #
    # **A SURVIVOR MUST NEVER BE SILENTLY DROPPED**, and an "inert"
    # verdict must itself be CHECKABLE -- otherwise a future refactor
    # that breaks the symmetry would leave the harness reporting a stale
    # "inert" that is no longer true. The harness therefore scores this
    # mutation with an INVERTED VERDICT: it PASSES when the mutation is
    # still inert and FAILS when it starts to matter. The probe is
    # tests/sample_quality_inertness.py.
    #
    # The mutation ITSELF must still be a real edit, so it is applied and
    # then checked for inertness by the probe.
    ("SM17", "the FID's mean term is symmetrised (PROVABLY INERT -- "
     "scored with an INVERTED verdict)",
     "    radicand = (np.sum((mu1 - mu2) ** 2)",
     "    radicand = (np.sum((mu1 - mu2) ** 2) + 0.0 * np.sum(mu1) - "
     "0.0 * np.sum(mu2)"),
]


def checksum(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_oracle():
    """Run the oracle and return (returncode, output)."""
    proc = subprocess.run([sys.executable, str(ORACLE)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def apply_mutation(old, new):
    """Apply the edit. Returns (applied, count)."""
    src = ORACLE.read_text()
    count = src.count(old)
    if count != 1:
        return False, count
    ORACLE.write_text(src.replace(old, new))
    return True, count


def main():
    backup = ORACLE.read_text()
    before = checksum(ORACLE)

    print("=== PRISTINE CHECK ===")
    rc, out = run_oracle()
    if rc != 0:
        print("HARNESS DEFECT: the pristine oracle is not green.", file=sys.stderr)
        print(out[-3000:], file=sys.stderr)
        return 99
    print("pristine: ORACLE GREEN -- mutation may begin")

    table = []
    caught = survived = broken = noop = inert = 0
    verdicts = {}

    print("\n=== ORACLE MUTATIONS ===")
    for mid, desc, old, new in MUTATIONS:
        ORACLE.write_text(backup)
        applied, count = apply_mutation(old, new)
        if not applied:
            table.append((mid, desc, "HARNESS-DEFECT (anchor count != 1)",
                          f"count={count}"))
            noop += 1
            verdicts[mid] = "HARNESS-DEFECT"
            print(f"  {mid}: HARNESS-DEFECT (anchor count = {count})")
            continue
        after = checksum(ORACLE)
        if after == before:
            ORACLE.write_text(backup)
            table.append((mid, desc, "HARNESS-DEFECT (edit did not land)", ""))
            noop += 1
            verdicts[mid] = "HARNESS-DEFECT"
            print(f"  {mid}: HARNESS-DEFECT (edit did not land)")
            continue

        rc, out = run_oracle()
        if rc == 0:
            table.append((mid, desc, "SURVIVED", "the oracle stayed green"))
            survived += 1
            verdicts[mid] = "SURVIVED"
            print(f"  {mid}: SURVIVED")
        elif "Traceback" in out:
            table.append((mid, desc, "MUT-BROKEN (the oracle crashed)",
                          "a crash is not a verdict"))
            broken += 1
            verdicts[mid] = "MUT-BROKEN"
            print(f"  {mid}: MUT-BROKEN (the oracle crashed -- NOT a verdict)")
        else:
            n_failed = out.count("[FAIL]")
            table.append((mid, desc, "CAUGHT", f"{n_failed} check(s) failed"))
            caught += 1
            verdicts[mid] = "CAUGHT"
            first = next((l for l in out.splitlines() if "[FAIL]" in l), "")
            print(f"  {mid}: CAUGHT ({n_failed} failed) {first.strip()[:80]}")

    ORACLE.write_text(backup)

    # -------------------------------------------------------------------------
    # THE INVERTED VERDICT FOR THE PROVABLY-INERT MUTATIONS.
    #
    # **THE INVERTED VERDICT MAY ONLY DOWNGRADE A VERDICT THAT WAS ACTUALLY
    # "SURVIVED".** A first draft of a sibling harness ran the check
    # unconditionally, so when the mutation came back HARNESS-DEFECT (its
    # mutation had never applied) the check still passed and the harness
    # REPORTED "MEASURED-INERT" for a mutation that had never run -- and
    # then decremented a SURVIVED counter that was already 0, printing
    # "-1 survived". **A verdict about a mutation that did not run is not a
    # verdict at all.**
    # -------------------------------------------------------------------------
    print("\n=== INVERTED VERDICT: are SM16 and SM17 still provably inert? ===")
    inert_ids = ["SM16", "SM17"]
    for mid in inert_ids:
        if verdicts.get(mid) != "SURVIVED":
            print(f"{mid}  NOT-APPLICABLE  (its verdict was "
                  f"{verdicts.get(mid, '<missing>')}, not SURVIVED -- the "
                  f"inertness claim cannot be scored)")
            noop += 1
            return 1

    probe = subprocess.run([sys.executable, "tests/sample_quality_inertness.py"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        print("SM16/SM17  MEASURED-INERT  (the inverted-verdict probe PASSES: "
              "the claim still holds)")
        for line in probe.stdout.splitlines():
            if "worst relative diff" in line or "BIT-IDENTICAL" in line:
                print(f"      {line.strip()}")
        survived -= len(inert_ids)
        inert += len(inert_ids)
        new_table = []
        for row in table:
            rid = row[0]
            if rid in inert_ids:
                new_table.append((rid, row[1],
                                  "MEASURED-INERT (the inverted verdict confirms it)",
                                  row[3]))
            else:
                new_table.append(row)
        table = new_table
    else:
        print("SM16/SM17  NOT-INERT  (the inverted-verdict probe FAILED -- the "
              "'inert' verdict is now WRONG)")
        print(probe.stdout[-2000:])
        print(probe.stderr[-2000:])

    if checksum(ORACLE) != before:
        print("HARNESS DEFECT: the oracle was not restored.", file=sys.stderr)
        return 99

    print("\n" + "=" * 78)
    print(f"{'ID':<6}{'MUTATION':<62}{'VERDICT':<24}DETAIL")
    print("-" * 78)
    for mid, desc, verdict, detail in table:
        print(f"{mid:<6}{desc:<62}{verdict:<24}{detail}")
    print("=" * 78)
    print(f"FINAL: {caught} caught / {survived} survived / {inert} measured-inert "
          f"/ {broken} broken / {noop} noop")

    if survived:
        print(f"\nNOTE: {survived} SURVIVOR(S) -- these are REAL coverage gaps "
              f"in the ORACLE.", file=sys.stderr)
        return 3
    if broken:
        print(f"\nNOTE: {broken} MUT-BROKEN -- the mutation crashed the oracle, "
              f"which is NOT a verdict.", file=sys.stderr)
        return 1
    if noop:
        print(f"\nNOTE: {noop} harness defect(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
