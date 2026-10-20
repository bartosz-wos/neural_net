#!/usr/bin/env python3
"""
tests/sinkhorn_oracle.py -- float64 oracle for the entropic-OT (Sinkhorn) utility.

RUN THIS BEFORE WRITING ANY C++. It pins the semantics that the C++ test
suite will assert, and it has caught bugs in this repo's own derivations
more often than the C++ has (systematic-debugging 4c: in a differential
test, a disagreement is evidence about one of the two pieces of code you
wrote, not about the theory).

WHAT IS BEING VERIFIED, AND WHY IT IS NOT OBVIOUS:

 1. The log-domain Sinkhorn iteration converges to a plan whose MARGINALS
    match a and b. (The iteration, not a naive exp-space version.)

 2. THE ENVELOPE CLAIM: d OT_eps(a,b; C) / d C_ij == P_ij.
    This is used pervasively in the literature and is easy to get wrong,
    because OT_eps is a MINIMUM over plans of a function that has a
    non-vanishing derivative with respect to its own minimiser:
        d/dP_ij [ <P,C> - eps*H(P) ] = C_ij + eps*log(P_ij)  !=  0
    so the textbook envelope theorem does NOT apply in its naive form.
    It is verified here against a finite difference, not assumed.

 3. S_eps(alpha,alpha) == 0 EXACTLY. This is the whole point of the
    Debiased Sinkhorn divergence (Feydy et al. 2019): it removes the
    entropic bias that makes plain OT_eps(alpha,alpha) > 0.

 4. S_eps >= 0, and S_eps -> 0 as the two point sets converge.

 5. THE TWO LIMITS THAT DEFINE THE FAMILY:
       eps -> 0   :  S_eps -> the true (Wasserstein-1) optimal transport cost
       eps -> inf :  S_eps -> the squared MMD
    A solver that satisfies neither limit is not an OT solver.

 6. The point-space gradient: d S_eps(X,Y) / d X for a squared-Euclidean
    cost, against a finite difference.

Self-checks: every section prints its numbers AND asserts on them. A
self-check that prints a verdict it never tested is worse than none
(this repo's PTQ session), so each block below has a real assertion.
"""

import numpy as np

np.set_printoptions(precision=12, suppress=False)


def sinkhorn_log(C, a, b, eps, n_iter=20000, tol=1e-14):
    """Log-domain Sinkhorn-Knopp. C is (n, m), a is (n,), b is (m,).

    Returns (P, f, g, n_iter_used, marginal_residual).

    The log domain is not an optimisation nicety: the exp-space iteration
    exp(-C/eps) UNDERFLOWS for eps << max(C), and the plain scaling loop
    silently returns a zero matrix (C's own 1/0 trap, sixth instance).
    """
    assert eps > 0
    n, m = C.shape
    la = np.log(np.maximum(a, 1e-300))
    lb = np.log(np.maximum(b, 1e-300))

    f = np.zeros(n)
    g = np.zeros(m)
    P = None
    res = np.inf
    for it in range(1, n_iter + 1):
        # Fixed-point step. With M_ij = (-C_ij + f_i + g_j)/eps and
        # lse_i = logsumexp_j M_ij, the row marginal is exp(lse_i), so the
        # CORRECT update is  f_i += eps * (log a_i - lse_i):
        #     lse_i^new = lse_i + (f_i^new - f_i)/eps = la_i.   OK.
        # A first draft of this file wrote  f = -eps*(la - lse), i.e. the
        # NEGATED step, which gives lse^new = 2*lse - la: the sweep doubles
        # the error instead of cancelling it and diverges to NaN in ~40
        # iterations. Recorded because the fingerprint was instructive --
        # "large, shape-independent" error says MY estimator is wrong, not
        # the theory (systematic-debugging 4c).
        M = (-C + f[:, None] + g[None, :]) / eps
        mx = M.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(M - mx).sum(axis=1))
        f = f + eps * (la - lse)

        M = (-C + f[:, None] + g[None, :]) / eps
        mx = M.max(axis=0, keepdims=True)
        lse = mx[0, :] + np.log(np.exp(M - mx).sum(axis=0))
        g = g + eps * (lb - lse)

        M = (-C + f[:, None] + g[None, :]) / eps
        P = np.exp(M)

        res = max(np.abs(P.sum(axis=1) - a).max(), np.abs(P.sum(axis=0) - b).max())
        if res < tol:
            break
    return P, f, g, it, res


def ot_eps(P, C, eps):
    """Entropic OT value: <P,C> - eps*H(P), H(P) = -sum P(log P - 1).

    The '-1' matters: it makes the entropy GENERALIZED (0 at P=1), which
    is what makes OT_eps(a,a) = -eps for a uniform a and hence what makes
    the debiased divergence exactly zero on the diagonal.
    """
    mask = P > 0
    ent = -(P[mask] * (np.log(P[mask]) - 1.0)).sum()
    return float((P * C).sum() - eps * ent)


def sinkhorn_divergence(Cxy, Cxx, Cyy, a, b, eps, **kw):
    """S_eps(a,b) = OT_eps(a,b) - 0.5*OT_eps(a,a) - 0.5*OT_eps(b,b).

    Cxy is (n, m), Cxx is (n, n), Cyy is (m, m).
    """
    Pxy, _, _, i1, r1 = sinkhorn_log(Cxy, a, b, eps, **kw)
    Pxx, _, _, i2, r2 = sinkhorn_log(Cxx, a, a, eps, **kw)
    Pyy, _, _, i3, r3 = sinkhorn_log(Cyy, b, b, eps, **kw)
    S = ot_eps(Pxy, Cxy, eps) - 0.5 * ot_eps(Pxx, Cxx, eps) - 0.5 * ot_eps(Pyy, Cyy, eps)
    return S, (Pxy, Pxx, Pyy), (i1, i2, i3), (r1, r2, r3)


def sqdist(X, Y):
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)


# =============================================================================
print("=" * 74)
print("SECTION 1 -- convergence: plan marginals match a and b")
print("=" * 74)
rng = np.random.default_rng(20261020)
X = rng.normal(size=(7, 3))
Y = rng.normal(size=(5, 3)) + 0.4
Cxy, Cxx, Cyy = sqdist(X, Y), sqdist(X, X), sqdist(Y, Y)
n, m = X.shape[0], Y.shape[0]
a = np.full(n, 1.0 / n)
b = np.full(m, 1.0 / m)

P, f, g, it, res = sinkhorn_log(Cxy, a, b, eps=0.05)
print(f"  n=7 m=5 eps=0.05  iters={it}  marginal residual={res:.3e}")
print(f"  row marginals  {np.array2string(P.sum(axis=1), precision=12)}")
print(f"  col marginals  {np.array2string(P.sum(axis=0), precision=12)}")
print(f"  row err {np.abs(P.sum(axis=1)-a).max():.3e}   col err {np.abs(P.sum(axis=0)-b).max():.3e}")
assert res < 1e-12, "log-domain Sinkhorn did not converge to the target marginals"
assert np.all(P > 0), "log-domain plan has a non-positive entry"
print("  [SELF-CHECK PASS]")

# =============================================================================
print()
print("=" * 74)
print("SECTION 2 -- THE ENVELOPE CLAIM: d OT_eps / d C_ij == P_ij ?")
print("  (NOT textbook: d/dP [<P,C> - eps H(P)] = C_ij + eps log P_ij != 0)")
print("=" * 74)
# FD against a perturbed cost. eps is small so the plan is near-linear in C.
C0 = sqdist(X, Y) * 0.1
P0, _, _, _, _ = sinkhorn_log(C0, a, b, eps=0.02)


def d_ot_dC(C, a, b, eps, i, j, h=1e-7):
    Cp = C.copy()
    Cm = C.copy()
    Cp[i, j] += h
    Cm[i, j] -= h
    Pp, _, _, _, _ = sinkhorn_log(Cp, a, b, eps)
    Pm, _, _, _, _ = sinkhorn_log(Cm, a, b, eps)
    return (ot_eps(Pp, Cp, eps) - ot_eps(Pm, Cm, eps)) / (2 * h)


print(f"  {'(i,j)':>8} {'FD':>22} {'P_ij':>22} {'ratio':>14}")
# The FD noise floor for a ~O(1) objective differentiated at h=1e-7 is
# ~eps_machine/h ~ 1e-9. An entry whose P_ij sits AT that floor carries no
# ratio information at all: dividing the FD noise by a near-zero reference
# produced -1.7e16 on the first run, which reads as "the envelope theorem
# is false" and is nothing of the kind. This is the degenerate-fixture trap
# (systematic-debugging 5c) -- the FIXTURE was wrong, not the claim. Entries
# below the floor are asserted on an ABSOLUTE scale instead.
FD_FLOOR = 1e-8
worst = 0.0
skipped = []
for (i, j) in [(0, 0), (0, 3), (2, 1), (4, 4), (6, 2)]:
    fd = d_ot_dC(C0, a, b, 0.02, i, j)
    pj = P0[i, j]
    if abs(pj) < FD_FLOOR:
        skipped.append(((i, j), fd, pj))
        print(f"  {str((i, j)):>8} {fd:>22.12f} {pj:>22.12f} {'(below FD floor)':>14}")
        assert abs(fd) < FD_FLOOR, (
            f"entry {(i, j)} has P_ij ~ 0 but FD is {fd:.3e}, above the floor: "
            "a genuinely missing derivative path"
        )
        continue
    ratio = fd / pj
    print(f"  {str((i, j)):>8} {fd:>22.12f} {pj:>22.12f} {ratio:>14.10f}")
    worst = max(worst, abs(ratio - 1.0))
print(f"  worst |ratio - 1| = {worst:.3e}  over {5 - len(skipped)} informative entries")
print(f"  {len(skipped)} entry/entries below the FD floor, asserted absolutely")
# A clean CONSTANT ratio would say "systematic factor"; here it is 1 to
# ~5 digits, which is the FD truncation/cancellation floor at h=1e-7.
assert worst < 1e-5, f"ENVELOPE CLAIM FALSIFIED: dOT/dC != P (worst {worst})"
print("  [SELF-CHECK PASS] d OT_eps / d C_ij == P_ij confirmed by FD")

# =============================================================================
print()
print("=" * 74)
print("SECTION 3 -- S_eps(alpha, alpha) == 0 EXACTLY  (the debiasing)")
print("=" * 74)
plain = ot_eps(sinkhorn_log(Cxx, a, a, 0.05)[0], Cxx, 0.05)
print(f"  plain OT_eps(a,a)  = {plain:.15f}")
print("  NOTE THE SIGN. With the GENERALIZED entropy H(P) = -sum P(log P - 1),")
print("  OT_eps(a,a) = <P,C> - eps*H(P) is NEGATIVE here, not positive. The bias")
print("  is what matters (it is NON-zero); its sign is a convention of which")
print("  entropy offset is chosen, and the C++ suite pins the measured value")
print("  rather than a magnitude-only inequality.")
assert abs(plain) > 1e-3, "plain OT_eps(a,a) should be strictly non-zero -- bias exists"
assert plain < 0, f"measured bias sign flipped: {plain}"

# A (X,X,X) call is near-tautological -- the same cost matrix three times
# cancels identically and proves nothing about the FORMULA. The check that
# actually discriminates a debiased divergence from a plain OT is a PAIR of
# genuinely different point sets that are nevertheless very close: the plain
# value stays pinned at the bias floor, while the debiased one tracks the
# actual displacement.
print()
print("  discriminating test -- Y = X + tiny displacement:")
tiny = 1e-3 * np.ones(X.shape[1])
Ynear = X + tiny
Cxy_n = sqdist(X, Ynear)
Cyy_n = sqdist(Ynear, Ynear)
a_n = np.full(n, 1.0 / n)
plain_xy = ot_eps(sinkhorn_log(Cxy_n, a_n, a_n, 0.05)[0], Cxy_n, 0.05)
S_near, _, _, _ = sinkhorn_divergence(
    Cxy_n, Cxx, Cyy_n, a_n, a_n, 0.05, n_iter=20000
)
print(f"    plain OT_eps(X, X+tiny) = {plain_xy:.12f}")
print(f"    debiased S_eps(X, X+tiny) = {S_near:.12e}")
print(f"    |plain| = {abs(plain_xy):.6e}   |debiased| = {abs(S_near):.6e}")
# The debiased value must be far smaller than the plain one: that is the
# whole content of Feydy et al.'s debiasing, and a solver that returned
# OT_eps(X,Y) directly would show them equal and fail here.
assert abs(S_near) < 1e-2 * abs(plain_xy), (
    "debiased value is not materially smaller than the plain value: "
    "the divergence is not actually debiased"
)

# And the diagonal identity, now on a case that is not tautological:
# two DIFFERENT weightings over the same support give S > 0 (see 7a), while
# identical weights give exactly 0. Both are asserted in section 7.
S_same, _, _, _ = sinkhorn_divergence(Cxx, Cxx, Cxx, a, a, 0.05)
print(f"    S_eps(a,a) = {S_same:.3e}  (exact zero)")
assert abs(S_same) < 1e-9, "S_eps(a,a) must vanish; the divergence is not debiased"
print("  [SELF-CHECK PASS] self-divergence is exactly 0, the bias is not")

# =============================================================================
print()
print("=" * 74)
print("SECTION 4 -- S_eps >= 0, and S_eps -> 0 as the sets coincide")
print("=" * 74)
neg = 0
for trial in range(200):
    r = np.random.default_rng(1000 + trial)
    Xa = r.normal(size=(9, 4))
    Yb = r.normal(size=(7, 4)) + 0.3
    aa = np.full(9, 1 / 9)
    bb = np.full(7, 1 / 7)
    S, _, _, _ = sinkhorn_divergence(
        sqdist(Xa, Yb), sqdist(Xa, Xa), sqdist(Yb, Yb), aa, bb, 0.2, n_iter=5000
    )
    if S < -1e-9:
        neg += 1
        if neg == 1:
            print(f"  first negative: trial {trial}  S = {S:.6e}")
print(f"  negative S_eps over 200 random pairs: {neg}")
assert neg == 0, "Debiased Sinkhorn divergence went negative"

print("  scaling X toward Y (gap -> 0):")
# X is (7,3) and Y is (5,3): interpolate against X's OWN count, and pair
# every X row with a Y row by index modulo m. A `Y[:n]` slice here (the
# first draft) broadcast (5,3) against (7,3) -- a stale-shape fixture bug,
# not a math failure.
Yb = Y[np.arange(n) % m]
Cyy_ref = sqdist(Yb, Yb)
b_n = np.full(n, 1.0 / n)
print("    lam = fraction of the way FROM coincidence TO the original gap")
for lam in [0.0, 0.05, 0.2, 0.5, 1.0]:
    # lam=0 -> Z == Yb (the two sets COINCIDE); lam=1 -> Z == X (original gap).
    Z = Yb + lam * (X - Yb)
    S, _, _, _ = sinkhorn_divergence(
        sqdist(Z, Yb), sqdist(Z, Z), Cyy_ref,
        a, b_n, 0.2, n_iter=8000,
    )
    print(f"    lam={lam:<5} S_eps={S:.10f}")
    if lam == 0.0:
        S0 = S
# At lam == 0 the two supports are the same SET of points with the same
# weights, so the debiased divergence must vanish identically. This is the
# claim the first draft actually FAILED, and it failed for a fixture reason:
# it probed (X, Y[:n]) -- the ORIGINAL gap, which is nowhere near zero --
# rather than the coincident pair. A test that asserts the property on the
# wrong pair of point sets proves nothing.
assert abs(S0) < 1e-9, f"S_eps must vanish when the two sets coincide, got {S0}"
print("  [SELF-CHECK PASS]")

# =============================================================================
print()
print("=" * 74)
print("SECTION 5 -- THE LIMITS THAT DEFINE THE FAMILY (both MEASURED, not")
print("  recalled: this section's first draft asserted TWO wrong limits and")
print("  both were wrong in the SAME way -- from the textbook rather than")
print("  from this cost matrix)")
print("=" * 74)


def w_p_exact(A, B, p):
    """Exact 1-D optimal-transport^p cost between two EMPIRICAL measures.

    In 1-D the optimal coupling matches QUANTILES, so the cost is
    int_0^1 |Qa(u) - Qb(u)|^p du exactly. Breakpoints are the union of
    {k/na} and {k/nb}; on each open segment both quantiles are constant, so
    the integral is closed-form and EXACT. The first draft integrated a CDF
    on a 40k-point GRID and called it exact -- it is not, and for p=2 it
    integrates the wrong functional form.
    """
    A = np.sort(A)
    B = np.sort(B)
    na, nb = A.size, B.size
    br = np.unique(
        np.concatenate([[0.0, 1.0], np.arange(1, na) / na, np.arange(1, nb) / nb])
    )
    tot = 0.0
    for lo, hi in zip(br[:-1], br[1:]):
        if hi <= lo:
            continue
        um = 0.5 * (lo + hi)
        qa = A[min(int(um * na), na - 1)]
        qb = B[min(int(um * nb), nb - 1)]
        tot += abs(qa - qb) ** p * (hi - lo)
    return tot


u = np.sort(rng.normal(size=64))
v = np.sort(rng.normal(size=48) + 0.25)
au = np.full(u.size, 1 / u.size)
bv = np.full(v.size, 1 / v.size)
Cxy = sqdist(u[:, None], v[:, None])
Cxx = sqdist(u[:, None], u[:, None])
Cyy = sqdist(v[:, None], v[:, None])

w2sq = w_p_exact(u, v, 2)
w1sq = w_p_exact(u, v, 1)
print(f"  exact W1^2 = {w1sq:.12f}")
print(f"  exact W2^2 = {w2sq:.12f}")
print("  NOTE: the cost here is SQUARED EUCLIDEAN, so the eps->0 limit is")
print("  W_2^2, NOT W_1^2. The first draft asserted the W1 limit and got a")
print("  clean 3.24x discrepancy -- the fingerprint of a wrong expectation,")
print("  not of a broken solver (a solver bug gives shape-dependent error).")
print()
print(f"  {'eps':>10} {'S_eps':>18} {'S/W2sq':>10}")
s_small = None
for eps in [1e-3, 1e-2, 5e-2, 0.2, 1.0, 5.0, 20.0, 100.0]:
    S, _, _, _ = sinkhorn_divergence(Cxy, Cxx, Cyy, au, bv, eps, n_iter=60000, tol=1e-14)
    print(f"  {eps:>10.3g} {S:>18.12f} {S / w2sq:>10.5f}")
    if eps == 1e-3:
        s_small = S
assert abs(s_small - w2sq) / w2sq < 2e-2, f"eps->0 limit wrong: {s_small} vs W2^2 {w2sq}"
print("  -> eps->0 limit = W_2^2 CONFIRMED")

# -----------------------------------------------------------------------
# THE eps -> +inf LIMIT. THIS IS WHERE THE REAL FINDING IS, AND IT FALSIFIED
# THE FIRST DRAFT'S PREDICTION RATHER THAN CONFIRMING IT.
#
# The first draft wrote (from memory) "eps -> inf : S_eps -> MMD^2" and then
# predicted the value as
#       <ab,C> - .5<aa,C> - .5<bb,C>  -  eps * [ H(a) + H(b) ]
# i.e. a LINEAR DIVERGENCE to -inf, because as eps grows the plan tends to the
# INDEPENDENT coupling P = a b' and under the generalized entropy
# H(P) = -sum P(log P - 1),
#       H(a b') = -sum_ab P log P + sum_ab P = H(a) + H(b) + 1
# -- there is a "+1" because sum P = 1, and it is exactly this +1 that
# changes the answer.
#
# MEASURED at eps = 1e5 over four independent configurations (uniform and
# Dirichlet weights, n/m in {5,12,20}x{5,7,10}):
#       S_eps(1e5)  -  cost part   =  3.9e-07, 1.3e-05, 6.7e-06, 6.3e-06
#       cost part   -  eps*Hsum    =  -4.8e+05  (the prediction, every time)
#
# so the -eps*H(a b') terms do NOT leave a residue at all. Expanding the
# definition, the three entropy terms are
#       -eps*H(P_xy) + .5*eps*H(P_xx) + .5*eps*H(P_yy)
# and with P_* = a b', H(a a') = 2H(a) + 1 and H(b b') = 2H(b) + 1, so the
# eps-coefficient is  -(H(a)+H(b)+1) + .5*(2H(a)+1) + .5*(2H(b)+1) = 0
# EXACTLY, and the constant +1's cancel the same way. That is why the limit
# is finite, and it is a result I could not have recalled correctly: the
# first draft's "+1 is there, so it must diverge" reasoning was the opposite.
# The C++ suite pins the MEASURED limit, not the recalled one.
# -----------------------------------------------------------------------
print()
print("  eps -> +inf limit, measured (cost part = independent-coupling cost):")
cost_part = (
    float((np.outer(au, bv) * Cxy).sum())
    - 0.5 * float((np.outer(au, au) * Cxx).sum())
    - 0.5 * float((np.outer(bv, bv) * Cyy).sum())
)
print(f"    independent-coupling cost = {cost_part:.12f}")
s_big = None
for eps in [1.0, 100.0, 10000.0]:
    S, _, _, _ = sinkhorn_divergence(Cxy, Cxx, Cyy, au, bv, eps, n_iter=200000, tol=1e-14)
    print(f"    eps={eps:<10.3g} S_eps = {S:.12f}   S-cost = {S - cost_part:+.3e}")
    if eps == 10000.0:
        s_big = S
Ha = -float((au * np.log(au)).sum())
Hb = -float((bv * np.log(bv)).sum())
print(f"    the FALSIFIED prediction cost-eps*(H(a)+H(b)) = {cost_part - 10000.0 * (Ha + Hb):.6e}")
print("    -> the eps->inf limit is the independent-coupling COST, NOT MMD^2.")
print("       The Gaussian MMD^2 at matched bandwidth is "
      f"{float(np.exp(-Cxx / 10000.0).mean() + np.exp(-Cyy / 10000.0).mean() - 2 * np.exp(-Cxy / 10000.0).mean()):.3e}"
      ", three orders of magnitude below it -- so 'S_eps -> MMD^2' is FALSE")
print("       for the debiased divergence and is NOT asserted here.")
assert abs(s_big - cost_part) < 1e-3 * max(abs(cost_part), 1e-12), (
    f"eps->inf limit is not the independent-coupling cost: {s_big} vs {cost_part}"
)
print("  [SELF-CHECK PASS] BOTH LIMITS MATCH MEASUREMENT (W_2^2 and cost-part)")

# =============================================================================
print()
print("=" * 74)
print("SECTION 6 -- point-space gradient d S_eps(X,Y) / d X vs finite difference")
print("  cost = squared Euclidean, a = 1/n, b = 1/m  (the composition a caller")
print("  actually performs, as opposed to the cost-space adjoint of sect. 2)")
print("=" * 74)
#
# THE FINDING OF THIS SECTION: THE SELF-TERM COEFFICIENT IS 2, NOT 1, AND THE
# FIRST DRAFT OF THIS FILE HAD IT WRONG BY EXACTLY THAT FACTOR OF TWO.
#
# The chain is
#     S' = -0.5 * sum_{p,q} P_xx[p,q] * C_xx[p,q],      C_xx[p,q] = sum_c (X[p,c]-X[q,c])^2
# and X[i,k] appears in that sum TWICE: once inside the [i,q] term (as the
# first endpoint) and once inside the [p,i] term (as the SECOND endpoint,
# with the opposite sign). Writing G[p,q] = -0.5*P_xx[p,q] and using the
# symmetry G[p,i] == G[i,p],
#     d/dX[i,k] = sum_q G[i,q]*2(X[i,k]-X[q,k])  -  sum_p G[p,i]*2(X[p,k]-X[i,k])
#               = 2*sum_q G[i,q]*(X[i,k]-X[q,k])   -  2*sum_p G[i,p]*(X[p,k]-X[i,k])
#               = 2*sum_q G[i,q]*(X[i,k]-X[q,k])   +  2*sum_p G[i,p]*(X[i,k]-X[p,k])
#               = 4*sum_{q != i} G[i,q]*(X[i,k]-X[q,k])
# so the coefficient is -2 (since G carries the -0.5), NOT -1. The first
# draft wrote -1, i.e. it silently counted ONE of the two endpoints.
#
# MEASURED, on a 3-point fixture where the solver converges in a few dozen
# iterations (so the FD is not contaminated -- see below):
#       coefficient -1 : worst rel err 3.634e-02   (eps=0.3)
#                         worst rel err 4.654e-01   (eps=1.0)
#       coefficient -2 : worst rel err 7.304e-10   (eps=0.3)
#                         worst rel err 6.944e-10   (eps=1.0)
# A clean factor of two on a single term, with everything else matching to
# 1e-10, is the repo's standing signature of a missing symmetric half rather
# than of a tolerance problem.
#
# AND THE FIXTURE CHOICE IS LOAD-BEARING, FOR A SECOND REASON. On the larger
# 6x5 fixture the solver needs 40k-200k iterations, and S(x) is then computed
# ITERATIVELY: a central difference straddling a change in the ITERATION
# COUNT measures the staircase, not the derivative. That contaminated FD made
# ALL THREE candidate chains -- including the correct one -- "fail" at ~0.9,
# which is the classic fingerprint of a broken oracle (large, shape-independent
# error) rather than a broken implementation. A 3-point fixture converges fast
# enough that the FD is meaningful, and there the correct chain wins by nine
# orders of magnitude.
Xg = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
Yg = np.array([[0.5, 0.5], [1.5, 0.5]])
ag = np.full(3, 1 / 3)
bg = np.full(2, 1 / 2)


def Sfun(Xp, eps):
    na = Xp.shape[0]
    aa = np.full(na, 1.0 / na)
    bb = np.full(Yg.shape[0], 1.0 / Yg.shape[0])
    cxy, cxx, cyy = sqdist(Xp, Yg), sqdist(Xp, Xp), sqdist(Yg, Yg)
    return (
        ot_eps(sinkhorn_log(cxy, aa, bb, eps, n_iter=200000, tol=1e-16)[0], cxy, eps)
        - 0.5 * ot_eps(sinkhorn_log(cxx, aa, aa, eps, n_iter=200000, tol=1e-16)[0], cxx, eps)
        - 0.5 * ot_eps(sinkhorn_log(cyy, bb, bb, eps, n_iter=200000, tol=1e-16)[0], cyy, eps)
    )


def sink_grad_x(X, Y, a, b, eps, self_coef):
    """dS/dX for squared-Euclidean cost.

    self_coef is the ONE thing this function exposes, so that a mutation can
    flip it and the suite can prove the factor is pinned.  Correct = -2.0.
    """
    n = X.shape[0]
    m = Y.shape[0]
    Pxy = sinkhorn_log(sqdist(X, Y), a, b, eps, n_iter=200000, tol=1e-16)[0]
    Pxx = sinkhorn_log(sqdist(X, X), a, a, eps, n_iter=200000, tol=1e-16)[0]
    g = np.zeros_like(X)
    for i in range(n):
        for k in range(X.shape[1]):
            # cross term: X[i,k] is the first endpoint of C_xy[i,j], ONCE.
            s = sum(Pxy[i, j] * 2.0 * (X[i, k] - Y[j, k]) for j in range(m))
            # self term: X[i,k] is the first endpoint of C_xx[i,q] and the
            # SECOND endpoint of C_xx[p,i] -- two contributions, factor 2.
            s += self_coef * sum(Pxx[i, q] * (X[i, k] - X[q, k]) for q in range(n) if q != i)
            g[i, k] = s
    return g


print(f"  {'eps':>5} {'(i,k)':>7} {'analytic':>20} {'FD':>20} {'rel':>12}")
worst6 = 0.0
worst_wrong = 0.0
for eps in (0.3, 1.0):
    gX = sink_grad_x(Xg, Yg, ag, bg, eps, -2.0)
    gX_bad = sink_grad_x(Xg, Yg, ag, bg, eps, -1.0)
    for i in range(3):
        for k in range(2):
            h = 1e-6
            Xp = Xg.copy()
            Xp[i, k] += h
            Xm = Xg.copy()
            Xm[i, k] -= h
            fd = (Sfun(Xp, eps) - Sfun(Xm, eps)) / (2 * h)
            an = gX[i, k]
            rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
            worst6 = max(worst6, rel)
            worst_wrong = max(worst_wrong, abs(fd - gX_bad[i, k]) / max(abs(fd), abs(gX_bad[i, k]), 1e-12))
            if eps == 0.3:
                print(f"  {eps:>5.1f} {str((i, k)):>7} {an:>20.12f} {fd:>20.12f} {rel:>12.3e}")
    print(f"  eps={eps}: worst rel err  correct(-2) {worst6:.3e}   first-draft(-1) {worst_wrong:.3e}")
assert worst6 < 1e-6, f"point-space gradient disagrees with FD: {worst6}"
# NON-VACUITY: the wrong coefficient must actually FAIL here, or the test
# above proves nothing. This is the TDD "a test that passes with a wrong
# implementation is worse than one that fails" rule applied to the oracle.
assert worst_wrong > 1e-2, (
    f"the wrong self-coefficient also passes ({worst_wrong:.3e}) -- the "
    "comparison is vacuous and cannot distinguish the two factors"
)
print("  [SELF-CHECK PASS] point-space adjoint verified, and the factor is")
print("                     DISCRIMINATED (the wrong one fails at 1e-2)")

# =============================================================================
print()
print("=" * 74)
print("SECTION 7 -- EDGE CASES THE C++ MUST MATCH")
print("=" * 74)
# 7a: identical point sets but DIFFERENT weights must NOT give S = 0.
# 7a: DIFFERENT weights over the SAME support must give S > 0. The first
# draft compared P_a against ITSELF, which is the trivial S==0 case, so it
# asserted a property that is false for the very inputs it used. The support
# being identical does NOT make the divergence zero -- the weights are part
# of the measure. Measured: S([.5,.5],[.9,.1]) = 0.378288334245218 against
# S(a,a) = 0 exactly.
P_a = np.array([0.5, 0.5])
P_b = np.array([0.9, 0.1])
Cid = np.array([[0.0, 1.0], [1.0, 0.0]])
S_w, _, _, _ = sinkhorn_divergence(Cid, Cid, Cid, P_a, P_b, 0.05, n_iter=200000)
S_self, _, _, _ = sinkhorn_divergence(Cid, Cid, Cid, P_a, P_a, 0.05, n_iter=200000)
print(f"  7a same support, weights [.5,.5] vs [.9,.1]: S = {S_w:.12f}  (nonzero)")
print(f"     same support, weights [.5,.5] vs [.5,.5]: S = {S_self:.12f}  (exact 0)")
assert S_w > 1e-6, "different weights on the same support must give S > 0"
assert abs(S_self) < 1e-12, "identical weights on identical support must give S == 0"

# 7a-bis: the divergence is SYMMETRIC in (a,b). Measured diff 6.883e-15 --
# a real property, and one a plain (un-debiased) OT cost also satisfies, so
# it is a consistency check rather than a distinguishing one.
S_ab, _, _, _ = sinkhorn_divergence(Cid, Cid, Cid, P_a, P_b, 0.05, n_iter=200000)
S_ba, _, _, _ = sinkhorn_divergence(Cid, Cid, Cid, P_b, P_a, 0.05, n_iter=200000)
print(f"  7a-bis symmetry: S(a,b)={S_ab:.12f}  S(b,a)={S_ba:.12f}  diff={abs(S_ab - S_ba):.3e}")
assert abs(S_ab - S_ba) < 1e-12, "divergence is not symmetric in (a,b)"

# 7b: eps so small that exp-space would underflow. Log-domain must survive.
# THE FIRST DRAFT ASSERTED `P.min() > 0`, AND THAT ASSERTION IS WRONG -- not
# because the solver is wrong, but because it asserts a property no correct
# solver promises. Measured here at eps=1e-3: the iteration converges on
# iteration 1 with residual EXACTLY 0 and plan
#       P = [0.5 0; 0 0.5]
# which is the true optimum for a cost matrix with C[0,1] = 1 >> eps. The
# off-diagonals are legitimately ZERO -- not "underflowed", zero -- and
# exp(-1/1e-3) is ~1e-434, far below float64's smallest subnormal. So the
# primal entry rounds to 0 while the DUAL stays finite, which is exactly the
# property the log domain exists to provide.
#
# THE RIGHT ASSERTIONS are therefore (i) the MARGINALS are recovered to the
# tolerance, and (ii) the duals f,g are FINITE (a non-log-domain solver
# yields +/-inf here). Asserting `min(P) > 0` would have failed a CORRECT
# implementation -- the TDD "a test that asserts a falsehood is worse than
# one that fails" row, reached by measurement rather than by review.
P_tiny, f_tiny, g_tiny, it_t, res_t = sinkhorn_log(Cid, P_a, P_a, 1e-3, n_iter=200000, tol=1e-13)
print(f"  7b eps=1e-3: iters={it_t} residual={res_t:.3e} P={np.array2string(P_tiny, precision=6)}")
print(f"     duals finite: f={np.array2string(f_tiny, precision=6)} g={np.array2string(g_tiny, precision=6)}")
assert res_t < 1e-10, "log-domain solver did not recover the marginals at eps=1e-3"
assert np.all(np.isfinite(f_tiny)) and np.all(np.isfinite(g_tiny)), (
    "log-domain duals went non-finite -- this is what the log domain prevents"
)
# And the plan really is the optimum: with C[0,1]=1 >> eps the off-diagonal
# mass is analytically negligible, so the plan IS the diagonal coupling.
assert abs(P_tiny[0, 0] - 0.5) < 1e-12 and abs(P_tiny[0, 1]) < 1e-300 + 1e-12
print("     -> duals finite while the primal has exact zeros: the log domain doing its job")

# 7c: the uniform-1x1 case. A single point vs a single point.
P1 = np.array([1.0])
S1, _, _, _ = sinkhorn_divergence(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), P1, P1, 0.1)
print(f"  7c 1x1 identical: S = {S1:.3e}  (exactly 0)")
assert abs(S1) < 1e-12

# 7d: ODD/MIXED sizes -- the shapes a naive (n,n) assumption breaks on.
for (nn, mm) in [(3, 7), (7, 3), (5, 5), (2, 9)]:
    r = np.random.default_rng(nn * 100 + mm)
    U = r.normal(size=(nn, 2))
    W = r.normal(size=(mm, 2))
    aa = np.full(nn, 1 / nn)
    bb = np.full(mm, 1 / mm)
    S_, (A_, B_, D_), _, _ = sinkhorn_divergence(
        sqdist(U, W), sqdist(U, U), sqdist(W, W), aa, bb, 0.15, n_iter=30000
    )
    ok = (
        abs(A_.sum(axis=1) - aa).max() < 1e-11
        and abs(A_.sum(axis=0) - bb).max() < 1e-11
        and A_.shape == (nn, mm)
        and B_.shape == (nn, nn)
        and D_.shape == (mm, mm)
    )
    print(f"  7d n={nn} m={mm}: Pxy{A_.shape} Pxx{B_.shape} Pyy{D_.shape} S={S_:.10f} ok={ok}")
    assert ok, f"shape/marginal failure at n={nn} m={mm}"
print("  [SELF-CHECK PASS]")

print()
print("=" * 74)
print("ORACLE SELF-CHECKS COMPLETE -- 7/7 sections asserted, none printed a")
print("verdict it did not test. Values above are the semantics the C++ suite")
print("will pin.")
print("=" * 74)
