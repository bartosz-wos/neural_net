#!/usr/bin/env python3
"""
tests/ordinal_regression_oracle.py -- the SPECIFICATION for the cumulative-link
ordinal regression layer (include/nn/layers/dense/ordinal_regression.{h,cpp}).

This file was written and run BEFORE the C++ existed. It is the definition,
not a description of the implementation.

===============================================================================
THE MODEL, STATED FROM THE DEFINITION
===============================================================================

The cumulative link model (McCullagh 1980, "Regression analysis of ordinal
data", JRSS-B 42(2):109-142) says: there is a LATENT variable

    Y* = eta + eps,        eps ~ standard Logistic(0, 1)

and K-1 ORDERED cut points theta_1 < theta_2 < ... < theta_{K-1} (with
theta_0 = -inf and theta_K = +inf) such that

    Y = k   iff   theta_{k-1} < Y* <= theta_k .

Therefore, with f(u) = e^{-u} / (1 + e^{-u})^2 the standard logistic density,

    P(Y = k | x) = INTEGRAL_{theta_{k-1}}^{theta_k} f(z - eta) dz .      (1)

Everything below is checked against (1). Nothing is checked against the
implementation's own formulas.

THE LAYER'S OUTPUT IS log P(Y = k | x), NOT P(Y = k | x). That is a
deliberate, load-bearing choice; see FINDING 2 below.

===============================================================================
THE PARAMETERISATION, AND WHY THE GAPS ARE KEPT SEPARATE
===============================================================================

The cut points are stored as K-1 RAW parameters r, with

    theta_1 = r_1
    theta_k = theta_{k-1} + gap_k,      gap_k := softplus(r_k) > 0

so the ORDERING IS STRUCTURAL: no optimiser step can produce theta_k <=
theta_{k-1}, hence no class probability can ever be negative. The map
r -> theta is a BIJECTION onto the strictly increasing vectors (r_1 free,
r_k = softplus^{-1}(theta_k - theta_{k-1})), so nothing is lost.

The slope is beta in R^D and the latent score is eta = x . beta. There is NO
separate intercept: P(Y <= k) = sigmoid(theta_k - eta) is invariant under
shifting every theta by c and eta by c, so an intercept would be EXACTLY
redundant with a uniform shift of the cut points. Pinning the intercept to
zero removes that degeneracy instead of hiding it.

**THE GAPS MUST NOT BE RECOVERED BY DIFFERENCING THE THETAS.** The
cumulative sum theta_k = theta_{k-1} + gap_k can ABSORB a tiny gap when
theta_1 is large: theta_1 = 100 with gap = softplus(-100) = 3.7e-44 gives
theta_2 == theta_1 EXACTLY in float64, so differencing reports a gap of 0.
The gap enters the class probability AND the gradient, so the
implementation keeps it as its own quantity, computed from r directly.
This oracle measures that (section [7]).

===============================================================================
WHY THERE ARE THREE REFERENCES AND NOT ONE
===============================================================================

(A) scipy.integrate.quad on (1). DEFINITION-LEVEL: it integrates the density
    over the interval, so it never assumes the closed form of a difference
    of two sigmoids. Accurate to ~1e-13 in the moderate regime.

(B) The `decimal` module at 60 significant digits. DEFINITION-LEVEL for the
    extreme regime, where (A) cannot resolve a probability of 1e-18 (quad's
    ABSOLUTE tolerance is 1e-14, so it reports 0.0 there). The closed form
    is evaluated in 60-digit arithmetic, which removes float64 cancellation.
    The closed form itself is validated by (A) in the moderate regime, so
    using it at high precision in the extreme regime measures the FLOAT64
    implementation rather than being circular.

(C) Central finite differences of the NLL, for the gradients. The finite
    difference is taken on the OBJECTIVE, so it validates the ANALYTIC
    gradient without presupposing it.

===============================================================================
THE TWO FINDINGS THAT DROVE THE WHOLE DESIGN
===============================================================================

FINDING 1 -- THE PROBABILITY OF AN INTERIOR CLASS IS A DIFFERENCE OF TWO
CUMULATIVES, AND THE NAIVE DIFFERENCE LOSES ALL OF IT.

    P(Y = k) = sigma(a_k) - sigma(a_{k-1}),   a_k = theta_k - eta.

When eta is large and negative, BOTH a_k and a_{k-1} are large and positive,
so BOTH sigmoids round to exactly 1.0 in float64 and the difference is
EXACTLY ZERO -- the class is silently assigned probability 0 while its true
probability is around 2.7e-18. That is not a rounding curiosity: it is the
difference between a class that is merely very unlikely and a class that is
impossible, and the NLL of the latter is +inf.

The stable identity is

    sigma(u) - sigma(v) = sigma(u) * sigma(-v) * (1 - e^{-(u-v)})     (u > v)

because sigma(u) - sigma(v) = (e^{-v} - e^{-u}) / ((1+e^{-u})(1+e^{-v}))
and e^{-v} - e^{-u} = e^{-v} * (1 - e^{-(u-v)}). Every factor is O(1), so
the product keeps full RELATIVE accuracy where the naive difference loses it
completely. This oracle MEASURES that gap.

FINDING 2 -- THE SAME CANCELLATION SITS INSIDE THE GRADIENT, ONE LEVEL
DEEPER, WHICH IS WHY THE LAYER OUTPUTS LOG PROBABILITIES.

The probability-space gradient carries the factor

    d sigma(a) / d eta = -sigma(a) * (1 - sigma(a))

and in the F3 fixture sigma(a) rounds to exactly 1.0, so that factor is
EXACTLY ZERO. A gradient computed in probability space is therefore ZERO
where the true gradient is about -1 (a 100% relative error), and no amount
of care in the probability computation can rescue it.

In LOG space the derivative factors are

    d log sigma(a) / d a      = 1 - sigma(a) = sigmoid(-a)   [stable]
    d log sigma(-a) / d a   = -sigma(a)                    [stable]

and the interior classes carry ONE extra term, because their probability
depends on the GAP between two cut points and not only on their positions:

    log P_k = log sigma(a_k) + log sigma(-a_{k-1}) + log(1 - e^{-gap_{k-1}})

with gap_{k-1} = theta_k - theta_{k-1} (independent of eta), so

    q_{k-1} := e^{-gap_{k-1}} / (1 - e^{-gap_{k-1}}) = 1 / (e^{gap_{k-1}} - 1)

    d log P_k / d a_k     =  (1 - c_k) + q_{k-1}
    d log P_k / d a_{k-1} = -(c_{k-1} + q_{k-1})

and for the two END classes, which have no gap:

    d log P_1 / d a_1     =  (1 - c_1)
    d log P_K / d a_{K-1} = -c_{K-1}

**THE q TERM IS THE WHOLE POINT.** It is what makes the interior classes'
gradient w.r.t. the cut points correct: the probability of an interior class
depends on the SPREAD of two cut points. An implementation that
differentiates only the positions gets the cut-point gradient wrong.

The layer's backward therefore consumes dL/d(log P) -- which is exactly what
the repo's own `utils/losses.h` NLLLoss produces -- and never forms 1/P nor
c(1-c).

===============================================================================
THE DERIVATIVES IN FULL (the specification the C++ implements)
===============================================================================

With c_k := sigmoid(a_k), e_k := sigmoid(-a_k) = 1 - c_k (computed
STABLY, never as 1 - c_k), and A_k, B_k the per-class partials of
log P_k w.r.t. a_k and a_{k-1}:

    class 1      : A_1 = e_1                     B_1 = 0
    class k      : A_k = e_k + q_{k-1}            B_k = -(c_{k-1} + q_{k-1})
    class K      : A_K = 0                       B_K = -c_{K-1}

    d log P_k / d eta    = -(A_k + B_k)
    d log P_k / d theta_j = A_k at j = k, plus B_k at j = k-1

and the chain from theta to the raw parameters r (theta_1 = r_1,
theta_j = theta_{j-1} + softplus(r_j), so dtheta_j/dr_1 = 1 for every j and
dtheta_j/dr_m = sigmoid(r_m) for 2 <= m <= j):

    dL/dr_1 = sum_j dL/dtheta_j
    dL/dr_m = sigmoid(r_m) * sum_{j >= m} dL/dtheta_j      (m >= 2)

and dL/dbeta = sum_rows (dL/deta_row) * x_row.
===============================================================================
"""

import math
import sys
from decimal import Decimal, getcontext

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

getcontext().prec = 60

G_PASS = 0
G_FAIL = 0
G_FAILURES = []


def check(ok, msg):
    global G_PASS, G_FAIL
    if ok:
        G_PASS += 1
    else:
        G_FAIL += 1
        G_FAILURES.append(msg)
        print("  [FAIL] %s" % msg)


def close(a, b, tol, msg):
    a = float(a)
    b = float(b)
    if math.isnan(a) or math.isnan(b):
        check(False, "%s (got %r, want %r, NaN)" % (msg, a, b))
        return
    denom = max(abs(a), abs(b))
    rel = abs(a - b) / denom if denom > 0 else abs(a - b)
    check(rel <= tol, "%s (got %.17g, want %.17g, rel %.3g)" % (msg, a, b, rel))


# -----------------------------------------------------------------------------
# the model, from the definition
# -----------------------------------------------------------------------------
def logistic_pdf(u):
    """Standard logistic density, overflow-safe (expit(u)*expit(-u))."""
    s = expit(u)
    return s * (1.0 - s)


def softplus(t):
    if t > 0.0:
        return t + math.log1p(math.exp(-t))
    return math.log1p(math.exp(t))


def softplus_inv(y):
    """softplus^{-1}(y) for y > 0."""
    return math.log(math.expm1(y))


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def gap_vector(r):
    """The EXACT gaps, straight from the primitive.

    The gap is softplus(r_k), NOT theta_k - theta_{k-1}: the cumulative sum
    can ABSORB a tiny gap when theta_1 is large (theta_1 = 100 with a gap of
    3.7e-44 gives theta_2 == theta_1 in float64), and the gap enters the
    probability AND the gradient, so it must never be recovered by
    differencing. This is the same reason the C++ keeps the gaps separate.
    """
    return [softplus(float(rk)) for rk in r[1:]]


def thresholds_from_raw(r):
    """Ordered cut points from the raw parameters r (softplus increments)."""
    theta = []
    prev = 0.0
    for i, ri in enumerate(r):
        if i == 0:
            prev = float(ri)
        else:
            prev = prev + softplus(float(ri))
        theta.append(prev)
    return theta


def eta_of(x, beta):
    return float(np.dot(np.asarray(x, dtype=float), np.asarray(beta, dtype=float)))


def _a_vector(x, beta, r):
    """a_k = theta_k - eta, with the gaps carried separately."""
    eta = eta_of(x, beta)
    theta = thresholds_from_raw(r)
    return [t - eta for t in theta]


def _q_vector(r):
    """q_k = 1 / (e^{gap_k} - 1) for the interior classes, k = 1..K-2.

    A non-positive gap is mathematically impossible (a gap is softplus of
    something) but a MUTATED oracle can produce one, and the division would
    raise ZeroDivisionError -- a crash, which is a THIRD verdict class
    rather than a reported failure. It returns +inf instead, so the
    comparison downstream reports a clean [FAIL].
    """
    out = []
    for g in gap_vector(r):
        if g <= 0.0:
            out.append(math.inf)
        else:
            out.append(1.0 / math.expm1(g))
    return out


def _log1mexp_neg(gap):
    """log(1 - e^{-gap}), stable for small gap, -inf for a non-positive gap."""
    if gap <= 0.0:
        return -math.inf
    return math.log(-math.expm1(-gap))


def probs_quadrature(x, beta, r):
    """(A) DEFINITION-LEVEL: integrate the logistic density over the intervals."""
    eta = eta_of(x, beta)
    theta = thresholds_from_raw(r)
    bounds = [-np.inf] + theta + [np.inf]
    out = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        val, _err = quad(lambda z: logistic_pdf(z - eta), lo, hi,
                         epsabs=1e-15, epsrel=1e-12, limit=400)
        out.append(val)
    return np.array(out)


def probs_naive(x, beta, r):
    """The NAIVE difference of two cumulatives, in float64."""
    a = _a_vector(x, beta, r)
    cdf = [sigmoid(v) for v in a]
    out = [cdf[0]]
    for k in range(1, len(cdf)):
        out.append(cdf[k] - cdf[k - 1])
    out.append(1.0 - cdf[-1])
    return np.array(out)


def log_probs_stable(x, beta, r):
    """log P(Y=k), computed stably, in float64. THIS IS THE LAYER'S OUTPUT."""
    a = _a_vector(x, beta, r)
    gaps = gap_vector(r)
    K = len(a) + 1
    out = [0.0] * K

    def logsig(z):
        return -softplus(-z)

    out[0] = logsig(a[0])
    for k in range(1, len(a)):
        out[k] = (logsig(a[k]) + logsig(-a[k - 1])
                  + _log1mexp_neg(gaps[k - 1]))
    out[K - 1] = logsig(-a[-1])
    return out


def probs_stable(x, beta, r):
    """The STABLE probabilities, from the stable log probabilities."""
    return np.array([math.exp(v) for v in log_probs_stable(x, beta, r)])


def _dec_sigmoid(z):
    if z >= 0:
        return Decimal(1) / (Decimal(1) + (-z).exp())
    e = z.exp()
    return e / (Decimal(1) + e)


def _dec_softplus(t):
    """softplus in Decimal, so the gaps are exact to 60 digits too.

    softplus(t) = log(1 + e^t) = t + log(1 + e^{-t}) for t > 0.
    NOTE the PLUS inside (1 + exp(-t)): writing (1 - exp(-t)) here yields a
    NEGATIVE log argument for small t, which is how this function's first
    draft was wrong -- and it produced negative PROBABILITIES, which the
    quadrature cross-check in section [1] caught immediately.
    """
    if t > 0:
        return t + (1 + (-t).exp()).ln()
    return (1 + t.exp()).ln()


def probs_decimal(x, beta, r):
    """(B) 60-digit evaluation of the closed form."""
    eta = Decimal(repr(eta_of(x, beta)))
    theta = []
    prev = Decimal(0)
    for i, ri in enumerate(r):
        if i == 0:
            prev = Decimal(repr(float(ri)))
        else:
            prev = prev + _dec_softplus(Decimal(repr(float(ri))))
        theta.append(prev)
    a = [t - eta for t in theta]
    K = len(a) + 1
    out = [Decimal(0)] * K
    out[0] = _dec_sigmoid(a[0])
    for k in range(1, len(a)):
        gap = theta[k] - theta[k - 1]
        one_minus = Decimal(1) - (-gap).exp()
        out[k] = _dec_sigmoid(a[k]) * _dec_sigmoid(-a[k - 1]) * one_minus
    out[K - 1] = _dec_sigmoid(-a[-1])
    return out


def log_probs_decimal(x, beta, r):
    """60-digit log probabilities."""
    return [v.ln() for v in probs_decimal(x, beta, r)]


def nll_from_log_probs(logp_rows, labels):
    return sum(-logp_rows[row][labels[row]] for row in range(len(labels))) / len(labels)


# -----------------------------------------------------------------------------
# the analytic gradient (the SPECIFICATION of the gradient)
# -----------------------------------------------------------------------------
def analytic_grads(X, beta, r, labels):
    """(dL/dbeta, dL/draw) for the MEAN ordinal NLL, in LOG space.

    The engine is the one the Layer uses: grad_output is dL/d(log P) and the
    layer turns it into dL/deta, dL/du and dL/dgap, then into dL/dbeta and
    dL/draw. For the mean NLL, dL/d(log P) is -1/N at the observed class.

    ------------------------------------------------------------------------
    WHY (u, GAPS) AND NOT (theta)
    ------------------------------------------------------------------------
    The q term -- q_j = 1/(e^{gap_j}-1) -- appears with OPPOSITE signs in
    the two theta-partials of an interior class, so it CANCELS when the two
    partials are summed. With a tiny gap q is astronomically large (2.7e43
    for a 3.7e-44 gap) and the cancellation destroys the answer: the
    oracle's first draft returned EXACTLY 0 where the finite difference
    returned -1.

    Differentiating in (u, gap) instead, the q term never has to cancel.
    Write theta_k = u + g_0 + g_1 + ... + g_{k-1}, so:
      * u = theta_1 shifts EVERY cut point, which is exactly equivalent to
        shifting eta the other way, so dL/du = -dL/deta and it carries no q.
      * d/dg_j has THREE cases per class, and this is the part that is easy
        to get wrong (the oracle's second draft got it wrong and the
        finite differences caught it):
            j <= k-2  : BOTH theta_k and theta_{k-1} shift, so the class
                        contributes (e_k - c_{k-1}), with NO q.
            j == k-1  : ONLY theta_k shifts, and the gap itself changes, so
                        the class contributes (e_k + q_{k-1}).
            j >= k    : nothing.
        A gap shift therefore touches EVERY class at or above it -- in
        particular the LAST class, whose probability is 1 - sigma(a_{K-2})
        and whose a depends on every gap. An implementation that credits the
        gap to a single class gets dL/dr wrong for every fixture whose
        observed class is not the one adjacent to that gap.
      * the chain gap_j = softplus(r_{j+1}) multiplies by sigmoid(r_{j+1}),
        and sigmoid(r_{j+1}) * q_j stays O(1): it is never formed as
        2.7e43 times 3.7e-44.
    """
    n = len(labels)
    K = len(r) + 1
    D = len(beta)
    gaps = gap_vector(r)                 # gaps[j] = theta_{j+1} - theta_j
    q = [math.inf if g <= 0.0 else 1.0 / math.expm1(g) for g in gaps]

    dL_deta_rows = []
    bulk = [0.0] * K                    # the "j <= k-2" coefficient per class
    special = [0.0] * K                 # the "j == k-1" coefficient per class

    for row in range(n):
        a = _a_vector(X[row], beta, r)
        c = [sigmoid(v) for v in a]          # c_j = sigmoid(a_j)
        e = [sigmoid(-v) for v in a]         # e_j = 1 - c_j, computed STABLY

        g = [0.0] * K
        g[labels[row]] = -1.0 / n          # d(mean NLL) / d log P

        # d log P_k / d eta. The gap is INDEPENDENT of eta, so no q here.
        acc_eta = 0.0
        for k in range(K):
            if k == 0:
                acc_eta += g[k] * (-e[0])
            elif k == K - 1:
                acc_eta += g[k] * (c[K - 2])
            else:
                acc_eta += g[k] * (c[k - 1] - e[k])
        dL_deta_rows.append(acc_eta)

        # the gap coefficients
        bulk[0] += g[0] * e[0]             # class 0 has no gap term at all
        for k in range(1, K - 1):
            bulk[k] += g[k] * (e[k] - c[k - 1])
            special[k] += g[k] * (e[k] + q[k - 1])
        bulk[K - 1] += g[K - 1] * (-c[K - 2])

    dL_dbeta = [0.0] * D
    for row in range(n):
        for d_ in range(D):
            dL_dbeta[d_] += dL_deta_rows[row] * float(X[row][d_])

    # dL/dg_j = sum over classes k >= j+2 of bulk[k], plus special[j+1].
    dL_dgap = [0.0] * (K - 2)
    suffix = 0.0
    for k in range(K - 1, 1, -1):
        suffix += bulk[k]
        dL_dgap[k - 2] = suffix
    for j in range(K - 2):
        dL_dgap[j] += special[j + 1]

    # r -> (u, gap): r_1 = u = theta_1, gap_j = softplus(r_{j+1})
    dL_dr = [0.0] * (K - 1)
    for row in range(n):
        dL_dr[0] += -dL_deta_rows[row]
    for j in range(K - 2):
        dL_dr[j + 1] = dL_dgap[j] * sigmoid(float(r[j + 1]))
    return dL_dbeta, dL_dr


def fd_grads(X, beta, r, labels, h=1e-6):
    """(C) central finite differences of the MEAN ordinal NLL."""
    n = len(labels)

    def objective(b, rr):
        tot = 0.0
        for row in range(n):
            lp = log_probs_stable(X[row], b, rr)
            tot += -lp[labels[row]]
        return tot / n

    gb = []
    for d_ in range(len(beta)):
        bp = list(beta); bp[d_] += h
        bm = list(beta); bm[d_] -= h
        gb.append((objective(bp, r) - objective(bm, r)) / (2 * h))
    gr = []
    for j in range(len(r)):
        rp = list(r); rp[j] += h
        rm = list(r); rm[j] -= h
        gr.append((objective(beta, rp) - objective(beta, rm)) / (2 * h))
    return gb, gr


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------
FIXTURES = {
    "F1": dict(K=3, D=2, quad=True,
               beta=[0.7, -1.3],
               r=[0.25, 0.8],
               X=[[1.0, 0.5], [-0.5, 2.0]],
               labels=[0, 2]),
    "F2": dict(K=5, D=3, quad=True,
               beta=[1.1, -0.4, 0.7],
               r=[-0.3, 0.2, 0.9, 0.5],
               X=[[1.0, 0.0, -1.0], [0.5, 1.5, 0.25], [-2.0, 0.75, 1.0]],
               labels=[3, 0, 4]),
    # the extreme one: eta = -40, so both cumulatives round to 1.0 in float64
    "F3": dict(K=3, D=2, quad=False,
               beta=[-30.0, -10.0],
               r=[0.0, math.log(math.e - 1.0)],
               X=[[1.0, 1.0]],
               labels=[1]),
    # the gap-absorption one: theta_1 = 100 and a gap of 3.7e-44, so the
    # cumulative sum CANNOT represent the gap by differencing
    "F4": dict(K=3, D=2, quad=False,
               beta=[0.0, 0.0],
               r=[100.0, softplus_inv(3.7e-44)],
               X=[[1.0, 0.0]],
               labels=[1]),
    # F5 and F6 exist to make the (1 - c) CANCELLATION OBSERVABLE. Without
    # them the mutation harness's OM9 and OM10 SURVIVED, because in every
    # other fixture the e-term is a rounding-level correction to an O(1)
    # quantity and no finite difference can see it. A survivor is a hole in
    # the SPECIFICATION, and the fix is a fixture, not a dropped mutation.
    #
    # F5: the OBSERVED class is the FIRST one, and its a is large positive, so
    # c = sigma(a) rounds to EXACTLY 1.0 and the whole gradient is carried by
    # e = 1 - c. The naive (1.0 - c) is then EXACTLY 0 while the true value
    # is 4.2e-18.
    "F5": dict(K=3, D=2, quad=False,
               beta=[-30.0, -10.0],
               r=[0.0, math.log(math.e - 1.0)],
               X=[[1.0, 1.0]],
               labels=[0]),
    # F6: FOUR classes, so there is an interior class whose gradient is
    # carried by the bulk term (e_k - c_{k-1}), and c_{k-1} is tiny while
    # e_k = 4.2e-18. The naive (1.0 - c_k) is exactly 0 there too.
    "F6": dict(K=4, D=2, quad=False,
               beta=[0.0, 0.0],
               r=[-100.0, softplus_inv(40.0), softplus_inv(100.0)],
               X=[[1.0, 0.0]],
               labels=[2]),
}


def main():
    print("=" * 78)
    print("ordinal regression oracle -- cumulative link model, float64")
    print("=" * 78)

    # ------------------------------------------------------------------
    print("\n[1] the closed form agrees with the DEFINITION (quadrature)")
    for name, f in FIXTURES.items():
        if not f["quad"]:
            continue  # quad cannot resolve 1e-18; section [2] covers those
        beta, r, X = f["beta"], f["r"], f["X"]
        for row in range(len(X)):
            pq = probs_quadrature(X[row], beta, r)
            ps = probs_stable(X[row], beta, r)
            pn = probs_naive(X[row], beta, r)
            for k in range(len(pq)):
                close(ps[k], pq[k], 1e-9,
                      "%s row %d class %d: stable form vs quadrature" % (name, row, k))
                close(pn[k], pq[k], 1e-9,
                      "%s row %d class %d: naive form vs quadrature" % (name, row, k))

    # ------------------------------------------------------------------
    print("\n[2] the 60-digit reference, and the FLOAT64 CANCELLATION")
    for name, f in FIXTURES.items():
        beta, r, X = f["beta"], f["r"], f["X"]
        for row in range(len(X)):
            pd = probs_decimal(X[row], beta, r)
            ps = probs_stable(X[row], beta, r)
            pn = probs_naive(X[row], beta, r)
            for k in range(len(pd)):
                rel = abs(float(pd[k]) - ps[k]) / max(abs(float(pd[k])), 1e-300)
                check(rel <= 1e-12,
                      "%s row %d class %d: stable float64 vs 60-digit "
                      "(rel %.3g, got %.17g want %.17g)"
                      % (name, row, k, rel, ps[k], float(pd[k])))
            if name == "F3":
                print("      %s row %d  60-digit truth : %s"
                      % (name, row, ["%.6e" % float(v) for v in pd]))
                print("      %s row %d  stable float64: %s"
                      % (name, row, ["%.6e" % v for v in ps]))
                print("      %s row %d  NAIVE float64 : %s"
                      % (name, row, ["%.6e" % v for v in pn]))
                check(pn[1] == 0.0,
                      "F3: the naive difference is expected to be EXACTLY 0.0 here "
                      "(got %.17g)" % pn[1])
                check(ps[1] > 0.0,
                      "F3: the stable form must be strictly positive (got %.17g)" % ps[1])
                rel = abs(float(pd[1]) - ps[1]) / float(pd[1])
                check(rel <= 1e-12,
                      "F3: stable float64 matches 60 digits on the cancelled class "
                      "(rel %.3g)" % rel)

    # ------------------------------------------------------------------
    print("\n[2b] the LOG probabilities against the 60-digit reference")
    for name, f in FIXTURES.items():
        beta, r, X = f["beta"], f["r"], f["X"]
        for row in range(len(X)):
            pd = log_probs_decimal(X[row], beta, r)
            lp = log_probs_stable(X[row], beta, r)
            for k in range(len(pd)):
                want = float(pd[k])
                close(lp[k], want, 1e-11,
                      "%s row %d class %d: log-prob vs 60-digit" % (name, row, k))

    # ------------------------------------------------------------------
    print("\n[3] structural invariants")
    for name, f in FIXTURES.items():
        beta, r, X, labels = f["beta"], f["r"], f["X"], f["labels"]
        K = f["K"]
        theta = thresholds_from_raw(r)
        for j in range(len(theta) - 1):
            check(theta[j + 1] >= theta[j],
                  "%s: cut points must be NON-DECREASING (%s)" % (name, theta))
        for row in range(len(X)):
            p = probs_stable(X[row], beta, r)
            check(abs(sum(p) - 1.0) <= 1e-14,
                  "%s row %d: probabilities sum to 1 (got %.17g)" % (name, row, sum(p)))
            for k, pk in enumerate(p):
                check(pk >= 0.0,
                      "%s row %d class %d: probability >= 0 (got %.17g)"
                      % (name, row, k, pk))
            cum = np.cumsum(p)
            for k in range(len(cum) - 1):
                check(cum[k] <= cum[k + 1] + 1e-15,
                      "%s row %d: cumulative must be non-decreasing" % (name, row))
        for y in labels:
            check(0 <= y < K, "%s: label %d in range" % (name, y))

    # ------------------------------------------------------------------
    print("\n[4] the analytic gradient against central finite differences")
    for name, f in FIXTURES.items():
        beta, r, X, labels = f["beta"], f["r"], f["X"], f["labels"]
        gb, gr = analytic_grads(X, beta, r, labels)
        fgb, fgr = fd_grads(X, beta, r, labels)
        for d_ in range(len(beta)):
            close(gb[d_], fgb[d_], 1e-5, "%s: dL/dbeta[%d] analytic vs FD" % (name, d_))
        for j in range(len(r)):
            close(gr[j], fgr[j], 1e-5, "%s: dL/dr[%d] analytic vs FD" % (name, j))

    # ------------------------------------------------------------------
    print("\n[4b] THE PROBABILITY-SPACE GRADIENT IS WRONG HERE (the reason the")
    print("     gradient is specified in log space) -- measured, not asserted")
    beta, r, X, labels = FIXTURES["F3"]["beta"], FIXTURES["F3"]["r"], \
        FIXTURES["F3"]["X"], FIXTURES["F3"]["labels"]
    a = _a_vector(X[0], beta, r)
    c = [sigmoid(v) for v in a]
    naive_dP_deta = -c[1] * (1.0 - c[1]) + c[0] * (1.0 - c[0])
    print("      F3: c = %s   (both round to 1.0 in float64)" % c)
    print("      F3: naive dP_2/deta = %.17g" % naive_dP_deta)
    check(naive_dP_deta == 0.0,
          "F3: the naive probability-space derivative is expected to be EXACTLY 0.0 "
          "(got %.17g)" % naive_dP_deta)
    gb_log, gr_log = analytic_grads(X, beta, r, labels)
    fgb, fgr = fd_grads(X, beta, r, labels)
    print("      F3: log-space dL/dbeta[0] = %.17g  (FD %.17g)" % (gb_log[0], fgb[0]))
    print("      F3: log-space dL/dr[0]   = %.17g  (FD %.17g)" % (gr_log[0], fgr[0]))
    check(abs(gb_log[0] - fgb[0]) <= 1e-5 * max(abs(fgb[0]), 1e-12),
          "F3: the log-space gradient must match FD where the naive one is 0")

    # ------------------------------------------------------------------
    print("\n[4c] THE (1 - c) FACTOR IS COMPUTED STABLY, and the cancellation")
    print("     is MEASURED rather than asserted")
    for name, f in FIXTURES.items():
        beta, r, X = f["beta"], f["r"], f["X"]
        for row in range(len(X)):
            a = _a_vector(X[row], beta, r)
            c = [sigmoid(v) for v in a]
            naive = [1.0 - cv for cv in c]
            stable = [sigmoid(-v) for v in a]
            for k in range(len(c)):
                if c[k] == 1.0:
                    # the cancellation is REAL here: c has rounded to 1.0
                    check(naive[k] == 0.0,
                          "%s row %d: the naive (1.0 - c) must be EXACTLY 0 when "
                          "c == 1.0 (got %.17g)" % (name, row, naive[k]))
                    check(stable[k] > 0.0,
                          "%s row %d: the stable sigmoid(-a) must stay positive "
                          "(got %.17g)" % (name, row, stable[k]))
                    check(stable[k] != naive[k],
                          "%s row %d: the two forms must DIFFER here" % (name, row))
                else:
                    close(stable[k], naive[k], 1e-14,
                          "%s row %d: the two forms must agree when c != 1.0"
                          % (name, row))

    # ------------------------------------------------------------------
    print("\n[5] the NLL value")
    for name, f in FIXTURES.items():
        beta, r, X, labels = f["beta"], f["r"], f["X"], f["labels"]
        lp = [log_probs_stable(X[row], beta, r) for row in range(len(X))]
        nll = nll_from_log_probs(lp, labels)
        print("      %s NLL = %.17g" % (name, nll))
        check(math.isfinite(nll), "%s: the NLL must be FINITE" % name)

    # ------------------------------------------------------------------
    print("\n[6] MONOTONICITY IN THE LATENT SCORE (the defining property)")
    print("     a HIGHER latent score must move mass towards HIGHER classes,")
    print("     so P(Y <= k) must DECREASE in eta and the expected class must RISE")
    beta = [1.0]
    r = [0.0, math.log(math.e - 1.0)]
    prev_cdf = None
    for eta_target in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
        p = probs_stable([eta_target], beta, r)
        cdf = np.cumsum(p)
        if prev_cdf is not None:
            ok = True
            for k in range(len(cdf)):
                if cdf[k] > prev_cdf[k] + 1e-12:
                    ok = False
            check(ok, "P(Y<=k) must DECREASE with eta (eta=%g)" % eta_target)
        prev_cdf = cdf
    means = []
    for eta_target in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        p = probs_stable([eta_target], beta, r)
        means.append(sum(k * pk for k, pk in enumerate(p)))
    for i in range(len(means) - 1):
        check(means[i + 1] > means[i],
              "the expected class must increase with eta (%g -> %g)" % (means[i], means[i + 1]))

    # ------------------------------------------------------------------
    print("\n[7] the ORDERING is structural, and the GAP must not be differenced")
    for r_try in [[0.0, 0.0, 0.0], [-50.0, 50.0, -50.0], [100.0, -100.0, 100.0]]:
        th = thresholds_from_raw(r_try)
        ok = all(th[j + 1] >= th[j] for j in range(len(th) - 1))
        check(ok, "raw %s -> thresholds %s must be non-decreasing" % (r_try, th))
        gaps = gap_vector(r_try)
        ok2 = all(g > 0.0 for g in gaps)
        check(ok2, "raw %s -> gaps %s must be strictly positive" % (r_try, gaps))
    # the differencing trap, measured
    r_trap = [100.0, softplus_inv(3.7e-44)]
    th_trap = thresholds_from_raw(r_trap)
    diff_gap = th_trap[1] - th_trap[0]
    exact_gap = gap_vector(r_trap)[0]
    print("      trap: theta = %s" % th_trap)
    print("      trap: gap by DIFFERENCING = %.17g" % diff_gap)
    print("      trap: gap from the PRIMITIVE = %.17g" % exact_gap)
    check(diff_gap == 0.0,
          "the differencing trap must actually fire (got %.17g)" % diff_gap)
    check(exact_gap > 0.0,
          "the exact gap must be strictly positive (got %.17g)" % exact_gap)

    # ------------------------------------------------------------------
    print("\n[8] the r -> (u, gaps) map is a BIJECTION onto (u, positive gaps)")
    for name, f in FIXTURES.items():
        # u = r_1, and the gaps come from the primitive. The round trip is
        # done in GAP space, NOT by differencing the float64 thetas -- doing
        # it by differencing is exactly the absorption trap of section [7],
        # and it fails for F4 by construction.
        u = f["r"][0]
        gaps = gap_vector(f["r"])
        rebuilt = [u]
        for g in gaps:
            rebuilt.append(softplus_inv(g))
        for j in range(len(rebuilt)):
            close(rebuilt[j], f["r"][j], 1e-12,
                  "%s: r -> (u, gaps) -> r round trip [%d]" % (name, j))
        # and the thetas rebuilt from (u, gaps) match the cumulative sum
        theta = thresholds_from_raw(f["r"])
        rebuilt_theta = [u]
        for g in gaps:
            rebuilt_theta.append(rebuilt_theta[-1] + g)
        for j in range(len(theta)):
            close(rebuilt_theta[j], theta[j], 1e-12,
                  "%s: theta rebuilt from (u, gaps) [%d]" % (name, j))

    # ------------------------------------------------------------------
    print("\n[9] emitting the C++ fixture block")
    print_emit()

    print("\n" + "=" * 78)
    print("oracle: %d checks passed, %d failed" % (G_PASS, G_FAIL))
    if G_FAIL:
        for m in G_FAILURES:
            print("  FAILED: %s" % m)
        return 1
    print("=" * 78)
    return 0


def fmt_list(vals):
    return ", ".join("%.17g" % v for v in vals)


def print_emit():
    print("")
    print("// ==== BEGIN ORACLE-GENERATED (tests/ordinal_regression_oracle.py) ====")
    print("// Regenerate with:  /usr/bin/python3 tests/ordinal_regression_oracle.py")
    print("// Do NOT hand-edit: these numbers are the SPECIFICATION.")
    for name, f in FIXTURES.items():
        beta, r, X, labels = f["beta"], f["r"], f["X"], f["labels"]
        K, D = f["K"], f["D"]
        theta = thresholds_from_raw(r)
        print("static const size_t ORC_%s_K = %d;" % (name, K))
        print("static const size_t ORC_%s_D = %d;" % (name, D))
        print("static const size_t ORC_%s_N = %d;" % (name, len(X)))
        print("static const double ORC_%s_BETA[%d] = {%s};" % (name, D, fmt_list(beta)))
        print("static const double ORC_%s_RAW[%d] = {%s};" % (name, K - 1, fmt_list(r)))
        print("static const double ORC_%s_THETA[%d] = {%s};" % (name, K - 1, fmt_list(theta)))
        print("static const int ORC_%s_LABEL[%d] = {%s};"
              % (name, len(X), ", ".join(str(y) for y in labels)))
        # FLAT arrays: row-major X (N*D) and LOGP (N*K). A flat array cannot
        # be mis-indexed by a row/column mix-up, which is the failure mode
        # this repo has been burned by before.
        xflat = []
        for row in X:
            xflat.extend(float(v) for v in row)
        print("static const double ORC_%s_X[%d] = {%s};" % (name, len(xflat), fmt_list(xflat)))
        prows = []
        for row in range(len(X)):
            pd = probs_decimal(X[row], beta, r)
            prows.extend(float(v) for v in pd)
        print("static const double ORC_%s_P[%d] = {%s};" % (name, len(prows), fmt_list(prows)))
        lprows = []
        for row in range(len(X)):
            lp = log_probs_stable(X[row], beta, r)
            lprows.extend(v for v in lp)
        print("static const double ORC_%s_LOGP[%d] = {%s};"
              % (name, len(lprows), fmt_list(lprows)))
        lp = [log_probs_stable(X[row], beta, r) for row in range(len(X))]
        nll = nll_from_log_probs(lp, labels)
        print("static const double ORC_%s_NLL = %.17g;" % (name, nll))
        gb, gr = analytic_grads(X, beta, r, labels)
        print("static const double ORC_%s_GRAD_BETA[%d] = {%s};" % (name, D, fmt_list(gb)))
        print("static const double ORC_%s_GRAD_RAW[%d] = {%s};" % (name, K - 1, fmt_list(gr)))
    print("// ==== END ORACLE-GENERATED ====")


if __name__ == "__main__":
    sys.exit(main())
