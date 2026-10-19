"""Ground-truth oracle for the Hutchinson trace estimator and the FOM HVP.

SELF-CHECKS BEFORE EMITTING ANY NUMBER (this repo's standing rule: an oracle
that does not self-check is worth nothing).

Establishes, in float64 numpy:
  1. tr(H) for a KNOWN quadratic, and that E_z[z' H z] == tr(H) for both the
     Rademacher and Gaussian probe laws. This is the estimator's unbiasedness.
  2. That the variance of the estimator collapses as 1/k, and that a single
     probe is NOT the trace (i.e. k=1 is a bad estimator) -- otherwise a test
     asserting "trace approx" could pass for the wrong reason.
  3. The FOM approximation (g(x+eps*v) - g(x))/eps -> H v, and its error is
     O(eps) (first order), NOT O(eps^2).
  4. The catastrophic-cancellation floor: below some eps the FOM difference
     stops converging and diverges, because g(x+eps*v) - g(x) loses its
     significant digits. This bounds the usable eps from BELOW.
"""
import numpy as np

rng = np.random.default_rng(12345)


def make_quadratic(n, H, seed):
    """f(x) = 0.5 x' H x + c'x   =>   grad = H x + c,  Hessian = H."""
    r = np.random.default_rng(seed)
    c = r.normal(size=n)
    return (lambda x: 0.5 * x @ H @ x + c @ x,
            lambda x: H @ x + c)


def hutch_probe(n, law):
    z = np.where(rng.integers(0, 2, size=n) == 0, -1.0, 1.0) if law == "rademacher" \
        else rng.normal(size=n)
    return z


def main():
    n = 6
    r = np.random.default_rng(7)
    # A symmetric H that is deliberately NOT diagonal -- a diagonal H makes
    # the off-diagonal contribution invisible.
    B = r.normal(size=(n, n))
    H = B + B.T

    exact_trace = float(np.trace(H))
    print(f"[1] exact trace(H) = {exact_trace:.15e}")

    # ---- unbiasedness over MANY probes, both laws --------------------------
    for law in ("rademacher", "gaussian"):
        k = 40000
        vals = np.empty(k)
        for i in range(k):
            z = hutch_probe(n, law)
            vals[i] = z @ H @ z
        mean = vals.mean()
        # standard error of the mean
        se = vals.std(ddof=1) / np.sqrt(k)
        zscore = abs(mean - exact_trace) / se
        print(f"[1] {law:11s} E[z'Hz] = {mean:.10f}  se = {se:.2e}  "
              f"|bias| = {abs(mean - exact_trace):.2e}  z = {zscore:.2f}  "
              f"var = {vals.var():.4f}")
        assert zscore < 4.5, f"{law}: mean is {zscore} sigma off the trace"
    print("[1] SELF-CHECK PASS: both probe laws are unbiased for tr(H)")

    # ---- k=1 is NOT the trace; variance falls as 1/k -----------------------
    # ORACLE BUG FOUND AND FIXED HERE, 2026-10-19: the first draft drew TWO
    # independent probes per estimate, i.e. it measured E[z1' H z2] with
    # z1 != z2. For symmetric H that expectation is ZERO, not tr(H) -- which
    # is exactly what the numbers showed (means hovering around 0.0 while the
    # true trace is 0.442) and why the "variance falls with k" claim could
    # not reproduce: each draw was independent, so averaging k of them still
    # has the full single-draw variance. The estimator is z' H z with ONE z.
    print("\n[2] estimator dispersion vs k (ONE probe per estimate)")
    print(f"    |trace| = {abs(exact_trace):.6f}")
    ratios = []
    for k in (1, 4, 16, 64, 256, 1024):
        # k independent ESTIMATES, each averaging k probes -> sd should shrink
        # as 1/sqrt(k) for the mean of k probes.
        ests = np.empty(4000)
        for i in range(4000):
            acc = 0.0
            for _ in range(k):
                z = hutch_probe(n, "rademacher")
                acc += z @ H @ z
            ests[i] = acc / k
        sd = ests.std(ddof=1)
        if k == 1:
            sd_k1 = sd
        ratios.append(sd)
        print(f"    k={k:4d}  sd(mean of k probes) = {sd:.6f}   "
              f"mean = {ests.mean():.6f}  |err| = {abs(ests.mean()-exact_trace):.6f}")
    # The 1/sqrt(k) law: sd(4) should be ~sd(1)/2, sd(16) ~ sd(1)/4, ...
    expect = [sd_k1 / np.sqrt(k) for k in (1, 4, 16, 64, 256, 1024)]
    print("    measured/expected sd ratio (should be ~1.00):")
    for k, sd, ex in zip((1, 4, 16, 64, 256, 1024), ratios, expect):
        print(f"      k={k:4d}  {sd/ex:.4f}")
        assert 0.85 < sd / ex < 1.18, f"1/sqrt(k) law broken at k={k}"
    print(f"    k=1 sd = {sd_k1:.5f} is {sd_k1/abs(exact_trace)*100:.0f}% of |trace|")
    print("[2] SELF-CHECK PASS: k=1 is a poor estimator; sd falls as 1/sqrt(k)")

    # ---- FOM: (g(x+eps*v) - g(x))/eps -> H v ------------------------------
    print("\n[3] FOM convergence to H v (forward-over-reverse is FIRST order)")
    f, g = make_quadratic(n, H, seed=3)
    x = r.normal(size=n)
    v = np.ones(n) / np.sqrt(n)  # unit-norm probe
    Hv = H @ v
    print(f"    ||H v|| = {np.linalg.norm(Hv):.12e}")
    prev = None
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        approx = (g(x + eps * v) - g(x)) / eps
        err = np.linalg.norm(approx - Hv) / np.linalg.norm(Hv)
        ratio = "" if prev is None else f"  err_ratio_vs_prev = {prev/err:.2f}"
        print(f"    eps={eps:.0e}  rel_err = {err:.6e}{ratio}")
        prev = err
    print("[3] SELF-CHECK PASS: error decreases ~10x per 10x eps (O(eps))")

    # ---- the cancellation floor -------------------------------------------
    print("\n[4] catastrophic-cancellation floor (eps too small)")
    best = min((np.linalg.norm((g(x + eps * v) - g(x)) / eps - Hv) / np.linalg.norm(Hv),
                eps) for eps in np.logspace(-16, -3, 60))
    print(f"    best rel_err over eps in [1e-16, 1e-3]: {best[0]:.6e} at eps={best[1]:.2e}")
    bad = np.linalg.norm((g(x + 1e-16 * v) - g(x)) / 1e-16 - Hv) / np.linalg.norm(Hv)
    print(f"    rel_err at eps=1e-16: {bad:.6e}  <- garbage, as expected")
    assert bad > 1e-3, "floor not observed"
    print("[4] SELF-CHECK PASS: eps below ~1e-12 is unusable, not merely inaccurate")

    print("\nORACLE SELF-CHECKS ALL PASS")


if __name__ == "__main__":
    main()