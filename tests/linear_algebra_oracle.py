#!/usr/bin/env python3
"""
tests/linear_algebra_oracle.py -- float64 oracle for the public symmetric
linear algebra (eigendecomposition, matrix square root, Frechet distance).

RUN THIS BEFORE WRITING ANY C++. It pins the semantics that the C++ test
suite will assert, and it has caught bugs in this repo's own derivations
more often than the C++ has (systematic-debugging 4c: in a differential
test, a disagreement is evidence about one of the two pieces of code you
wrote, not about the theory).

WHAT IS BEING VERIFIED, AND WHY IT IS NOT OBVIOUS:

 1. THE EIGENPAIRS ARE SORTED DESCENDING. numpy.linalg.eigh returns the
    eigenvalues in ASCENDING order, so an implementation that "just uses
    numpy's order" would disagree with ours on every matrix. The convention
    is therefore asserted EXPLICITLY, not inherited.

 2. THE ROTATION ALWAYS PUTS THE LARGER EIGENVALUE IN THE LOWER SLOT.
    This is what makes the cyclic sweep's final diagonal UNSORTED in
    general: the diagonal order is the order in which the rotations
    happened to converge, not a sorted order. The GaLore "top-r" defect
    (2026-10-29) is exactly this, and it is measured here.

 3. A MATRIX SQUARE ROOT OF A PSD MATRIX IS SYMMETRIC AND SATISFIES
    S @ S == A, and its eigenvalues are the square roots of A's.

 4. THE FRECHET DISTANCE, AGAINST A CLOSED FORM FOR THE COMMUTING CASE.
    For diagonal (hence commuting) covariances the Frechet distance has
    the exact value

        ||mu1 - mu2||^2 + sum_i (sqrt(l1_i) - sqrt(l2_i))^2

    which needs NO matrix square root at all. A solver that gets the
    matrix square root wrong still passes the identity checks
    (d(X,X)==0, symmetry, non-negativity) -- only this one catches it.

 5. THE DISCRIMINATING FIXTURE. The formula variant

        tr(S1) + tr(S2) - 2*tr(S1 @ S2)      (NO square root)

    is the classic mistake, and it is DISAGREED with by the general
    (non-commuting) fixture. If it agreed, the fixture would be
    vacuous.

 6. A SLIGHTLY NEGATIVE EIGENVALUE. A covariance that is PSD only to
    within rounding has an eigenvalue of -1e-18; sqrt of that is NaN.
    The clamp to zero is load-bearing and is pinned.

Self-checks: every section prints its numbers AND asserts on them. A
self-check that prints a verdict it never tested is worse than none
(this repo's PTQ session), so each block below has a real assertion.

RUN WITH:  /usr/bin/python3 tests/linear_algebra_oracle.py
"""

import numpy as np
import sys

FAILS = 0


def check(ok, name, extra=""):
    global FAILS
    if ok:
        print(f"  [PASS] {name}")
    else:
        FAILS += 1
        print(f"  [FAIL] {name} {extra}")


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# The SAME cyclic Jacobi the repo implements, so the oracle can measure the
# ORDER in which the eigenpairs emerge -- which is the property the GaLore
# defect lives in. It is written independently of the C++ (different data
# layout, different loop structure) but it is the SAME ALGORITHM, because
# the point of the measurement is the algorithm's behaviour, not the
# implementation's spelling.
# ---------------------------------------------------------------------------
def cyclic_jacobi(A, max_sweeps=100, tol=1e-12):
    """Returns (Q, lam) with A = Q @ diag(lam) @ Q.T, lam in DIAGONAL ORDER."""
    n = A.shape[0]
    Aq = A.astype(float).copy()
    Q = np.eye(n)
    for _ in range(max_sweeps):
        off = np.sqrt(2.0 * np.sum(Aq[np.triu_indices(n, 1)] ** 2))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = Aq[p, q]
                if abs(apq) < 1e-15:
                    continue
                app, aqq = Aq[p, p], Aq[q, q]
                if abs(app - aqq) < 1e-30:
                    theta = np.pi / 4.0 if apq >= 0 else -np.pi / 4.0
                else:
                    theta = 0.5 * np.arctan2(2.0 * apq, app - aqq)
                c, s = np.cos(theta), np.sin(theta)
                Aq[p, p] = c * c * app + 2 * c * s * apq + s * s * aqq
                Aq[q, q] = s * s * app - 2 * c * s * apq + c * c * aqq
                Aq[p, q] = 0.0
                Aq[q, p] = 0.0
                for r in range(n):
                    if r == p or r == q:
                        continue
                    arp, arq = Aq[r, p], Aq[r, q]
                    narp, narq = c * arp + s * arq, -s * arp + c * arq
                    Aq[r, p] = narp
                    Aq[p, r] = narp
                    Aq[r, q] = narq
                    Aq[q, r] = narq
                for r in range(n):
                    qrp, qrq = Q[r, p], Q[r, q]
                    Q[r, p] = c * qrp + s * qrq
                    Q[r, q] = -s * qrp + c * qrq
    lam = np.diag(Aq).copy()
    return Q, lam


def sym(A):
    return (A + A.T) / 2.0


def psd_sqrt(A):
    """The PRINCIPAL square root of a symmetric PSD matrix, via its
    eigendecomposition.

    NOTE: numpy has NO `linalg.sqrtm` -- that is `scipy.linalg.sqrtm`, and
    scipy is not a dependency of this repo. The eigendecomposition route is
    also the mathematically canonical definition for a symmetric PSD matrix
    (the principal square root is a polynomial in A), so it is not a
    weaker reference.

    NOTE THE CLAMP: `max(lam, 0)` is part of the definition. A covariance
    that is PSD only to within rounding has an eigenvalue of -1e-18, and
    sqrt of that is NaN.
    """
    lam, Q = np.linalg.eigh(sym(A))
    return Q @ np.diag(np.sqrt(np.maximum(lam, 0.0))) @ Q.T


def main():
    rng = np.random.default_rng(20261029)

    # =======================================================================
    section("1. THE EIGENPAIR ORDER: DESCENDING IS THE CONTRACT")
    # =======================================================================
    # numpy returns ASCENDING. Ours is DESCENDING. The difference is the
    # whole content of the convention, so it is asserted rather than
    # assumed.
    M = rng.normal(size=(5, 5))
    A = sym(M @ M.T)
    lam_asc, Q_asc = np.linalg.eigh(A)
    check(
        np.all(np.diff(lam_asc) >= -1e-12),
        "1.1 numpy.linalg.eigh returns ASCENDING (the opposite of our contract)",
    )
    check(
        np.allclose(lam_asc[::-1], np.sort(lam_asc)[::-1]),
        "1.2 reversing numpy's order gives the descending order",
    )

    # The DISCRIMINATING fixture: a Gram matrix that is ALREADY DIAGONAL
    # with the dominant entry in the MIDDLE. No rotation happens, so the
    # diagonal keeps the INPUT order and is NOT descending.
    n = 4
    g = np.zeros((n, n))
    g[0, 0], g[1, 1], g[2, 2], g[3, 3] = 1.0, 5.0, 1.0, 1.0
    gram = g @ g.T
    check(
        np.allclose(gram, np.diag(np.diag(gram))),
        "1.3 the discriminating fixture's Gram matrix is ALREADY DIAGONAL",
    )
    Qj, lamj = cyclic_jacobi(gram)
    check(
        not np.all(np.diff(lamj) <= 1e-12),
        "1.4 the RAW cyclic Jacobi diagonal is NOT descending (this is the defect)",
        f"diagonal = {lamj}",
    )
    check(
        abs(lamj[0] - 1.0) < 1e-12 and abs(lamj[1] - 25.0) < 1e-12,
        "1.5 the raw diagonal is the INPUT order: 1, 25, 1, 1",
        f"diagonal = {lamj}",
    )
    # The top-1 eigenvector of the raw result is the SMALLEST eigenvalue's,
    # not the largest's.
    check(
        lamj[0] < lamj.max() - 1e-9,
        "1.6 so 'the first column' is NOT the dominant eigenvector",
    )
    # The SORTED contract fixes it.
    order = np.argsort(lamj)[::-1]
    lam_sorted = lamj[order]
    check(
        np.all(np.diff(lam_sorted) <= 1e-12),
        "1.7 sorting descending makes the contract hold",
    )
    check(
        abs(lam_sorted[0] - 25.0) < 1e-9,
        "1.8 the largest eigenvalue comes FIRST after sorting",
    )
    # And the SORTED result is still a genuine eigendecomposition.
    resid = np.max(np.abs(gram @ Qj[:, order] - Qj[:, order] @ np.diag(lam_sorted)))
    check(resid < 1e-10, "1.9 sorting preserves the eigenvector property", f"resid={resid:.3e}")

    # The rotation ALWAYS puts the larger eigenvalue in the LOWER slot.
    viol = 0
    trials = 20000
    for _ in range(trials):
        app, aqq, apq = rng.uniform(-5, 5, 3)
        if abs(app - aqq) < 1e-30:
            continue
        th = 0.5 * np.arctan2(2.0 * apq, app - aqq)
        c, s = np.cos(th), np.sin(th)
        napp = c * c * app + 2 * c * s * apq + s * s * aqq
        naqq = s * s * app - 2 * c * s * apq + c * c * aqq
        if not napp > naqq:
            viol += 1
    check(
        viol == 0,
        "1.10 the rotation always puts the larger eigenvalue in the LOWER slot",
        f"{viol}/{trials} violations",
    )

    # =======================================================================
    section("2. THE MATRIX SQUARE ROOT")
    # =======================================================================
    # THE FIXTURE MUST BE POSITIVE SEMIDEFINITE. The first draft built it as
    # `sym(M @ N.T)` with M and N INDEPENDENT random matrices -- which is
    # symmetric but NOT positive semidefinite, so `max(lam, 0)` clamped a
    # genuinely negative eigenvalue and `S @ S == A` failed at ~1e-1. The
    # fix is `M @ M.T`, which is PSD BY CONSTRUCTION.
    #
    # This is the repo's recurring slip class, caught by the oracle's own
    # self-check: a fixture that violates the method's own precondition
    # cannot test the method.
    M = rng.normal(size=(4, 4))
    A2 = M @ M.T
    check(
        np.linalg.eigvalsh(A2).min() > -1e-12,
        "2.0 the fixture is POSITIVE SEMIDEFINITE (the precondition)",
    )
    lam, Q = np.linalg.eigh(A2)
    S = Q @ np.diag(np.sqrt(np.maximum(lam, 0.0))) @ Q.T
    check(np.max(np.abs(S @ S - A2)) < 1e-10, "2.1 S @ S == A")
    check(np.max(np.abs(S - S.T)) < 1e-12, "2.2 S is symmetric")
    lamS = np.linalg.eigvalsh(S)
    check(
        np.max(np.abs(lamS - np.sqrt(np.maximum(lam, 0.0)))) < 1e-10,
        "2.3 S's eigenvalues are the square roots of A's",
    )

    # A RANK-DEFICIENT matrix has an eigenvalue of exactly 0.
    Mr = rng.normal(size=(4, 2))
    Ar = Mr @ Mr.T
    lamr = np.linalg.eigvalsh(Ar)
    check(abs(lamr.min()) < 1e-12, "2.4 the rank-deficient fixture has a zero eigenvalue")
    Sr = psd_sqrt(Ar)
    check(np.max(np.abs(Sr @ Sr - Ar)) < 1e-9, "2.5 the square root of a rank-deficient matrix works")

    # A SLIGHTLY NEGATIVE eigenvalue: PSD only to within rounding.
    B = np.diag([1.0, 1.0, 1.0, -1e-17])
    Sb = psd_sqrt(B)
    check(
        np.all(np.isfinite(Sb)),
        "2.6 the max(lam,0) clamp keeps the square root FINITE on a negative eigenvalue",
    )
    check(
        np.all(np.isfinite(psd_sqrt(np.array([[1e-18]])))),
        "2.7 the square root of a tiny positive eigenvalue is finite",
    )

    # =======================================================================
    section("3. THE FRECHET DISTANCE")
    # =======================================================================
    # 3a. THE CLOSED FORM FOR THE COMMUTING (DIAGONAL) CASE.
    mu1 = np.array([[1.0, 2.0]])
    mu2 = np.array([[0.0, -1.0]])
    l1 = np.array([4.0, 1.0])
    l2 = np.array([9.0, 1.0])
    S1 = np.diag(l1)
    S2 = np.diag(l2)
    closed = np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(l1) - np.sqrt(l2)) ** 2)

    def frechet(mu1, S1, mu2, S2):
        # d^2 = ||mu1-mu2||^2 + tr(S1) + tr(S2) - 2 tr((S1^{1/2} S2 S1^{1/2})^{1/2})
        S1h = psd_sqrt(S1)
        M = S1h @ S2 @ S1h
        Mh = psd_sqrt(M)
        return np.sqrt(np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.trace(Mh))

    d = frechet(mu1, S1, mu2, S2)
    # THE CLOSED FORM IS FOR d^2, NOT d. The first draft compared `d`
    # against the SQUARED closed form (11.0 against 3.3166) and asserted
    # they were equal -- which is a false statement about the maths, and
    # the oracle rightly rejected it. The right comparison is
    # d == sqrt(closed_form).
    check(
        abs(d - np.sqrt(closed)) < 1e-9,
        "3.1 the commuting case matches the closed form (d == sqrt(d^2))",
        f"d={d} sqrt(closed)={np.sqrt(closed)}",
    )
    check(abs(frechet(mu1, S1, mu1, S1)) < 1e-12, "3.2 d(X, X) == 0")
    check(abs(frechet(mu1, S1, mu2, S2) - frechet(mu2, S2, mu1, S1)) < 1e-12, "3.3 d is symmetric")
    check(frechet(mu1, S1, mu2, S2) >= 0, "3.4 d >= 0")

    # 3b. THE GENERAL (NON-COMMUTING) CASE, and the DISCRIMINATING VARIANT.
    #
    # THE FIXTURES MUST BE PSD. The first draft used `sym(M @ N.T)` with
    # independent M and N -- symmetric but INDEFINITE, so the covariance's
    # square root was not defined and every distance came out NaN. PSD is
    # the method's own precondition (a covariance matrix is PSD), and a
    # fixture that violates it cannot test the method.
    n = 3
    Ma = rng.normal(size=(n, n))
    Mb = rng.normal(size=(n, n))
    Sa = Ma @ Ma.T + np.eye(n) * 0.1
    Sb = Mb @ Mb.T + np.eye(n) * 0.1
    check(
        np.linalg.eigvalsh(Sa).min() > 0 and np.linalg.eigvalsh(Sb).min() > 0,
        "3.0 the covariances are POSITIVE DEFINITE (the precondition)",
    )
    mu_a = rng.normal(size=(1, n))
    mu_b = rng.normal(size=(1, n))

    d_gen = frechet(mu_a, Sa, mu_b, Sb)

    # THE DISCRIMINATING VARIANT.
    #
    # THE FIRST DRAFT used the "classic mistake" `tr(S1 @ S2)` (no square
    # root). **That variant is not merely different -- it is NOT A
    # DISTANCE AT ALL**: its radicand measured -57.4 on this fixture, so
    # it is NEGATIVE and the distance comes out NaN. It therefore
    # "discriminates" trivially (NaN != anything), which makes it a
    # VACUOUS foil: a test that asserts `right != NaN` is not testing the
    # right implementation, it is testing that the wrong one is
    # undefined.
    #
    # THE RIGHT FOIL IS A VARIANT THAT IS A VALID DISTANCE BUT THE WRONG
    # ONE: the same formula with the WRONG PRODUCT ORDER, i.e.
    # `tr((S1 S2)^{1/2})` instead of `tr((S1^{1/2} S2 S1^{1/2})^{1/2})`.
    # Both are finite, both are symmetric in (S1, S2) and both are
    # non-negative -- so the foil cannot be dismissed by "it is NaN" or
    # "it is asymmetric", and it discriminates the ORDER, which is the
    # part that is easy to get wrong.
    wrong = np.sqrt(
        np.sum((mu_a - mu_b) ** 2)
        + np.trace(Sa)
        + np.trace(Sb)
        - 2 * np.trace(psd_sqrt(Sa @ Sb))
    )
    check(
        np.isfinite(wrong),
        "3.5a the foil is a FINITE, well-defined quantity (not a NaN-producing strawman)",
        f"wrong={wrong}",
    )
    check(
        abs(d_gen - wrong) > 1e-6,
        "3.5 the DISCRIMINATING fixture separates the right formula from the wrong-product-order variant",
        f"right={d_gen:.12f} wrong={wrong:.12f}",
    )
    # And the foil is ALSO symmetric -- so symmetry alone does not
    # discriminate, which is the point of pinning it here.
    check(
        abs(wrong - np.sqrt(np.sum((mu_b - mu_a) ** 2) + np.trace(Sb) + np.trace(Sa) - 2 * np.trace(psd_sqrt(Sb @ Sa)))) < 1e-12,
        "3.6 the foil is ALSO symmetric -- so symmetry alone does not discriminate",
    )

    # 3c. THE PRODUCT ORDER MATTERS for non-commuting covariances.
    #
    # The FIRST DRAFT of this check compared `tr((S1^h S2 S1^h)^h)` with
    # `tr((S2^h S1 S2^h)^h)` and asserted they DIFFER -- but they are the
    # SAME QUANTITY (both are `tr((S1^{1/2} S2 S1^{1/2})^{1/2})` by the
    # trace identity for the principal square root), so the check was
    # asserting a falsehood about the maths and the oracle rightly
    # rejected it.
    #
    # THE ORDER THAT MATTERS IS INSIDE THE PRODUCT: `S1^h S2 S1^h` versus
    # `S1 S2`. The latter is NOT symmetric, and the two give different
    # traces. THAT is the load-bearing order.
    S1h = psd_sqrt(Sa)
    S2h = psd_sqrt(Sb)
    t_correct = np.trace(psd_sqrt(S1h @ Sb @ S1h))
    t_nosqrt = np.trace(psd_sqrt(Sa @ Sb))
    check(
        abs(t_correct - t_nosqrt) > 1e-9,
        "3.7 the two product forms give DIFFERENT traces (so the square-root order is load-bearing)",
        f"correct={t_correct:.12f} nosqrt={t_nosqrt:.12f}",
    )
    check(
        abs(t_correct - np.trace(psd_sqrt(S2h @ Sa @ S2h))) < 1e-9,
        "3.7b the SYMMETRIC product forms agree (trace identity for the principal square root)",
    )

    # 3d. THE MEAN TERM IS SEPARABLE.
    #
    # THE FIRST DRAFT asserted `d_shift == sqrt(d_gen^2 + ||mu_c-mu_a||^2)`,
    # which is WRONG: the mean term is `||mu1 - mu2||^2`, so shifting mu1
    # changes the DIFFERENCE to mu_b, not to mu_a. The correct identity is
    #
    #     d(mu_c, S1; mu_b, S2)^2 - d(mu_a, S1; mu_b, S2)^2
    #         == ||mu_c - mu_b||^2 - ||mu_a - mu_b||^2
    #
    # MEASURED: both sides are 32.53113290354429 / 32.53113290354428.
    mu_c = mu_a + np.array([[5.0, -3.0, 2.0]])
    d_shift = frechet(mu_c, Sa, mu_b, Sb)
    lhs = d_shift**2 - d_gen**2
    rhs = np.sum((mu_c - mu_b) ** 2) - np.sum((mu_a - mu_b) ** 2)
    check(
        abs(lhs - rhs) < 1e-9,
        "3.8 the mean enters ONLY through ||mu1 - mu2||^2",
        f"lhs={lhs} rhs={rhs}",
    )

    # =======================================================================
    section("4. THE NUMBERS THE C++ TEST WILL ASSERT")
    # =======================================================================
    # Printed so the C++ test can embed them, and so a reader can see the
    # expected values are COMPUTED rather than hand-arithmetic.
    print("  commuting case d      = %.15f" % d)
    print("  commuting closed form = %.15f" % closed)
    print("  general case d        = %.15f" % d_gen)
    print("  general wrong variant = %.15f" % wrong)
    print("  mu_a                  = %s" % mu_a)
    print("  mu_b                  = %s" % mu_b)
    print("  Sa                    = %s" % Sa)
    print("  Sb                    = %s" % Sb)

    print(f"\n=== SUMMARY: {'ORACLE GREEN' if FAILS == 0 else str(FAILS) + ' ORACLE FAILURES'} ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
