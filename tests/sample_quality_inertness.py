#!/usr/bin/env python3
"""
tests/sample_quality_inertness.py -- the INVERTED-VERDICT probe for the
two mutations that PROVABLY cannot change the FID's value.

    /usr/bin/python3 tests/sample_quality_inertness.py

WHAT THIS IS FOR
================

The oracle's mutation harness reported survivors, and a survivor is NOT
automatically a coverage gap: **a mutation that provably changes nothing
is not a vacuous test, it is an INERT mutation, and the correct verdict
is "inert" -- but that verdict must itself be CHECKABLE.**

Two mutations are inert BY CONSTRUCTION:

  SM16  the FID's sample Gaussians are taken from the WRONG SET
        (real and fake swapped).
  SM17  the FID's mean term is symmetrised by a zero-weighted term.

WHY FID IS SYMMETRIC, SO SM16 CANNOT MATTER
===========================================
The FID's trace term is

    tr( (S1^(1/2) . S2 . S1^(1/2))^(1/2) )

`S1^(1/2) . S2 . S1^(1/2)` is SIMILAR to `S1 . S2` (via S1^(1/2)), and
`S2^(1/2) . S1 . S2^(1/2)` is similar to `S2 . S1`. The matrices `S1 . S2`
and `S2 . S1` have the SAME EIGENVALUES (they are similar to each other
when S1 and S2 are invertible, and in general they are both similar to
the same symmetric product). So the two square roots have the same
eigenvalues, the two traces are equal, the two radicands are equal, and
the two FIDs are equal. **Swapping the arguments cannot change the
value.**

MEASURED over 3000 random feature pairs (n from 2 to 11, d from 1 to 6):
the worst RELATIVE difference was 1.146e-07 -- floating-point noise
around a mathematically exact symmetry.

WHY SM17 CANNOT MATTER
======================
`0.0 * np.sum(mu1) - 0.0 * np.sum(mu2)` is exactly zero for every finite
input, so the radicand is unchanged bit-for-bit.

THE RULE THIS ENCODES
=====================
**A SURVIVOR MUST NEVER BE SILENTLY DROPPED.** An unexplained survivor
is indistinguishable from a real coverage gap, and a one-off "inert"
verdict in a harness log is a claim that decays: the next refactor could
break the symmetry and leave the harness reporting a stale "inert" that
is no longer true.

So the inertness is re-derived HERE, INDEPENDENTLY of the harness, from
the first principles, and this probe **FAILS if the symmetry ever stops
holding**. The harness scores the two mutations with the INVERTED
verdict: it PASSES while the mutation is inert and FAILS when it starts
to matter.
"""

import importlib.util
import pathlib
import sys

import numpy as np

ORACLE = pathlib.Path("tests/sample_quality_oracle.py")
_spec = importlib.util.spec_from_file_location("sample_quality_oracle", ORACLE)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def main():
    print("=== THE FID SYMMETRY, RE-DERIVED INDEPENDENTLY ===")

    # -- 1. THE SYMMETRY HOLDS OVER MANY RANDOM FEATURE PAIRS. The
    #    dimensions VARY, because a symmetry that only holds at one shape
    #    is not a symmetry.
    rng = np.random.default_rng(4242)
    trials = 3000
    worst_rel = 0.0
    worst_case = None
    for _ in range(trials):
        n = int(rng.integers(2, 12))
        d = int(rng.integers(1, 7))
        X = rng.normal(size=(n, d))
        Y = rng.normal(size=(n, d)) + rng.normal() * 0.5
        ab = _oracle.frechet_inception_distance(X, Y)
        ba = _oracle.frechet_inception_distance(Y, X)
        denom = max(abs(ab), 1e-300)
        rel = abs(ab - ba) / denom
        if rel > worst_rel:
            worst_rel = rel
            worst_case = (n, d, ab, ba)
    print(f"  pairs tested         : {trials}")
    print(f"  worst relative diff  : {worst_rel:.3e}")
    print(f"  worst case (n, d)    : {worst_case[0]}, {worst_case[1]}")
    ok_sym = worst_rel < 1e-6
    print(f"  [{'PASS' if ok_sym else 'FAIL'}] the FID is symmetric in its "
          f"two arguments")

    # -- 2. THE SYMMETRY IS ALSO EXACT IN THE MEAN TERM AND THE TRACE
    #    TERM SEPARATELY, which is where the proof lives.
    X = rng.normal(size=(20, 5))
    Y = rng.normal(size=(20, 5)) + 0.3
    mu_r, S_r = _oracle.sample_mean_cov(X)
    mu_f, S_f = _oracle.sample_mean_cov(Y)
    mean_term = float(np.sum((mu_r - mu_f) ** 2))
    mean_term_swapped = float(np.sum((mu_f - mu_r) ** 2))
    ok_mean = abs(mean_term - mean_term_swapped) <= 1e-12 * max(1.0, mean_term)
    print(f"  [{'PASS' if ok_mean else 'FAIL'}] the mean term is symmetric "
          f"({mean_term!r} vs {mean_term_swapped!r})")

    S_rh = _oracle.psd_sqrt(S_r)
    S_fh = _oracle.psd_sqrt(S_f)
    trace_ab = float(np.trace(_oracle.psd_sqrt(S_rh @ S_f @ S_rh)))
    trace_ba = float(np.trace(_oracle.psd_sqrt(S_fh @ S_r @ S_fh)))
    ok_trace = abs(trace_ab - trace_ba) <= 1e-9 * max(1.0, abs(trace_ab))
    print(f"  [{'PASS' if ok_trace else 'FAIL'}] the trace term is symmetric "
          f"({trace_ab!r} vs {trace_ba!r})")

    # -- 3. THE ZERO-WEIGHTED SYMMETRISATION TERM (SM17) IS EXACTLY ZERO.
    #    This is the bit-for-bit statement, not the approximate one.
    term = 0.0 * np.sum(mu_r) - 0.0 * np.sum(mu_f)
    ok_zero = (term == 0.0)
    print(f"  [{'PASS' if ok_zero else 'FAIL'}] SM17's zero-weighted term is "
          f"EXACTLY zero ({term!r})")

    # -- 4. AND THE MUTATED FID IS BIT-IDENTICAL TO THE PRISTINE ONE over
    #    the same random pairs -- the STRONGEST form of the inertness
    #    claim, because it is about the whole function and not one term.
    def mutated_fid(real_features, fake_features):
        mu1, S1 = _oracle.sample_mean_cov(real_features)
        mu2, S2 = _oracle.sample_mean_cov(fake_features)
        S1h = _oracle.psd_sqrt(S1)
        inner = S1h @ S2 @ S1h
        inner_sqrt = _oracle.psd_sqrt(inner)
        radicand = (np.sum((mu1 - mu2) ** 2) + 0.0 * np.sum(mu1)
                    - 0.0 * np.sum(mu2)
                    + np.trace(S1) + np.trace(S2) - 2.0 * np.trace(inner_sqrt))
        if radicand < 0.0:
            radicand = 0.0
        d = np.sqrt(radicand)
        return d * d

    rng2 = np.random.default_rng(777)
    worst_abs = 0.0
    for _ in range(500):
        n = int(rng2.integers(2, 12))
        d = int(rng2.integers(1, 7))
        X = rng2.normal(size=(n, d))
        Y = rng2.normal(size=(n, d)) + rng2.normal() * 0.5
        a = _oracle.frechet_inception_distance(X, Y)
        b = mutated_fid(X, Y)
        worst_abs = max(worst_abs, abs(a - b))
    ok_bit = worst_abs == 0.0
    print(f"  [{'PASS' if ok_bit else 'FAIL'}] SM17's mutated FID is "
          f"BIT-IDENTICAL to the pristine one (worst abs diff {worst_abs!r})")

    all_ok = ok_sym and ok_mean and ok_trace and ok_zero and ok_bit
    print(f"\n=== SUMMARY: {'INERTNESS HOLDS' if all_ok else 'INERTNESS BROKEN'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
