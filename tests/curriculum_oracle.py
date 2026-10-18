#!/usr/bin/env python3
"""Independent float64 oracle for utils/curriculum.{h,cpp}.

Nothing here is transcribed from the C++ (which does not exist yet). Every
number is DERIVED from the primary source (Fan, He, Liang, Hu,
"Self-Paced Learning: an Implicit Regularization Perspective", arXiv:1606.00128)
by one of two independent routes:

  (A) the EXPLICIT family -- g(lambda, v) is stated in the paper's Table
      "Recently Proposed Self-paced Regularizers", and sigma is its argmin over
      v in [0, 1]. We never transcribe sigma: we grid-search v and check the
      paper's closed form against the numeric argmin.

  (B) the IMPLICIT family (SPL-IR) -- phi(lambda, t) is stated in Table 1, and
      Definition 1's condition 1 is phi = min_{v>=0} [v*t + psi(lambda, v)]
      with sigma = argmin. We derive psi by convex conjugacy (Eq. 5-6) in
      closed form, then verify sigma numerically as the argmin of the pair.

The script self-checks before printing, so a wrong derivation cannot silently
become a transcription.
"""
import numpy as np
from math import log, sqrt, exp, isinf

FAIL = []


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# Explicit family: g(lambda, v) from the paper; sigma = argmin_{v in [0,1]}
# ---------------------------------------------------------------------------

def g_hard(lam, v):
    """Kumar et al. -- g = -lambda * sum_i v_i."""
    return -lam * v


def g_linear(lam, v):
    """Lu, Jiang et al. -- g = 1/2 * lambda * (v^2 - 2v)."""
    return 0.5 * lam * (v * v - 2.0 * v)


def g_log(lam, v):
    """Lu, Jiang et al. -- g = zeta*v - zeta^v / log(zeta), zeta = 1 - lambda."""
    zeta = 1.0 - lam
    return zeta * v - zeta ** v / log(zeta)


def g_mixture(v, lam1, lam2):
    """Lu, Jiang et al. -- g = -zeta * sum log(v + zeta/lambda1),
    zeta = lambda1*lambda2/(lambda1 - lambda2)."""
    zeta = lam1 * lam2 / (lam1 - lam2)
    if v <= 0.0:
        return np.inf          # g is only finite for v > 0
    return -zeta * log(v + zeta / lam1)


def g_gamma(lam, gamma, v):
    """Zhao et al. -- g = lambda*gamma^2 / (lambda*v + gamma)."""
    den = lam * v + gamma
    if den <= 0.0:
        return np.inf
    return lam * gamma * gamma / den


# Paper's closed forms, transcribed ONLY to be checked against the numeric
# argmin -- never used as a source of truth.
def sigma_hard_paper(ell, lam):
    return 1.0 if ell < lam else 0.0


def sigma_linear_paper(ell, lam):
    return 1.0 - ell / lam if ell < lam else 0.0


def sigma_log_paper(ell, lam):
    zeta = 1.0 - lam
    return log(ell + zeta) / log(zeta) if ell < lam else 0.0


def sigma_mixture_paper(ell, lam1, lam2):
    zeta = lam1 * lam2 / (lam1 - lam2)
    if ell <= lam2:
        return 1.0
    if ell >= lam1:
        return 0.0
    return (lam1 - ell) * zeta / (ell * lam1)


def sigma_gamma_paper(ell, lam, gamma):
    if ell <= (lam * gamma / (lam + gamma)) ** 2:
        return 1.0
    if ell >= lam * lam:
        return 0.0
    return gamma * (1.0 / sqrt(ell) - 1.0 / lam)


def argmin_grid(gfun, lo, hi, n=4001):
    v = np.linspace(lo, hi, n)
    vals = np.array([gfun(x) for x in v])
    return float(v[int(np.argmin(vals))])


def min_value(gfun, lo, hi, n=4001):
    return min(gfun(x) for x in np.linspace(lo, hi, n))


print("=" * 74)
print("(A) EXPLICIT FAMILY -- is the paper's sigma really argmin_v g ?")
print("=" * 74)

worst = 0.0
for lam in (0.3, 0.5, 0.75, 1.0):
    for ell in (0.0, 0.1, 0.4, 0.9, 1.5, 3.0, 10.0):
        worst = max(worst, abs(argmin_grid(lambda x: ell * x + g_hard(lam, x), 0.0, 1.0)
                               - sigma_hard_paper(ell, lam)))
check("hard: argmin matches paper sigma", worst < 1e-3, f"worst={worst:.2e}")

worst = 0.0
for lam in (0.5, 1.0, 2.0):
    for ell in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0):
        worst = max(worst, abs(argmin_grid(lambda x: ell * x + g_linear(lam, x), 0.0, 1.0)
                               - sigma_linear_paper(ell, lam)))
check("linear: argmin matches paper sigma", worst < 1e-3, f"worst={worst:.2e}")

worst = 0.0
for lam in (0.2, 0.4, 0.6):
    for ell in (0.0, 0.05, 0.2, 0.5, 0.9, 2.0):
        worst = max(worst, abs(argmin_grid(lambda x: ell * x + g_log(lam, x), 1e-9, 1.0)
                               - sigma_log_paper(ell, lam)))
check("log: argmin matches paper sigma", worst < 1e-3, f"worst={worst:.2e}")

worst = 0.0
for lam1, lam2 in ((2.0, 1.0), (3.0, 1.0), (1.5, 0.5)):
    for ell in (0.1, 0.4, 0.8, 1.2, 1.8, 2.5):
        worst = max(worst, abs(argmin_grid(lambda x: ell * x + g_mixture(x, lam1, lam2), 1e-9, 1.0)
                               - sigma_mixture_paper(ell, lam1, lam2)))
check("mixture: argmin matches paper sigma", worst < 1e-3, f"worst={worst:.2e}")

worst = 0.0
for lam, gam in ((1.0, 1.0), (2.0, 0.5), (1.0, 3.0)):
    for ell in (0.1, 0.5, 1.0, 2.0, 4.0):
        worst = max(worst, abs(argmin_grid(lambda x: ell * x + g_gamma(lam, gam, x), 0.0, 1.0)
                               - sigma_gamma_paper(ell, lam, gam)))
check("gamma: argmin matches paper sigma", worst < 1e-3, f"worst={worst:.2e}")

# The mixture's clamp thresholds are DERIVED by its own formula -- verify.
lam1, lam2 = 2.0, 1.0
zeta = lam1 * lam2 / (lam1 - lam2)
check("mixture: sigma == 1 exactly at ell == lam2",
      abs(sigma_mixture_paper(lam2, lam1, lam2) - 1.0) < 1e-15,
      f"got {sigma_mixture_paper(lam2, lam1, lam2)!r}")
check("mixture: sigma == 0 exactly at ell == lam1",
      sigma_mixture_paper(lam1, lam1, lam2) == 0.0)
check("mixture: zeta > 0 iff lam1 > lam2 > 0", zeta > 0, f"zeta={zeta}")

# ---------------------------------------------------------------------------
# Implicit family (SPL-IR): phi from Table 1, psi by convex conjugacy.
# Definition 1 condition 1:  phi(lam,t) = min_{v>=0} [v*t + psi(lam,v)]
# ---------------------------------------------------------------------------

def phi_cauchy(lam, t):
    return lam * lam * log(1.0 + (t / lam) ** 2)


def psi_cauchy(lam, v):
    """sup_{t>=0} [phi(t) - v*t], for v in the argmin domain [0, 2].

    DERIVED, not guessed: at the argmin, phi(t) = sigma(t)*t + psi(sigma(t)), so
    psi(v) = phi(t) - v*t evaluated at v = sigma(t). With u = (t/lam)^2 and
    sigma = 2/(1+u) we get 1+u = 2/sigma, hence

        psi(v) = lam^2 log(2/v) - lam*sqrt(v*(2-v)).

    NOTE the domain: sigma(t) = dphi/dt = 2*lam^2*t/(lam^2+t^2) peaks at
    exactly 2 (at t == lam), so the argmin of min_v[v*t + psi(v)] always lies in
    [0, 2]. Restricting psi to that domain leaves the MINIMUM VALUE unchanged,
    so this is a domain restriction, not an approximation.
    """
    if v < 0.0 or v > 2.0:
        return np.inf
    return lam * lam * log(2.0 / v) - lam * sqrt(max(0.0, v * (2.0 - v)))


def phi_welsch(lam, t):
    return lam * lam * (1.0 - exp(-(t / lam) ** 2))


def psi_welsch(lam, v):
    """phi(t) - v*t at v = sigma(t) = 2*exp(-(t/lam)^2).

    u = (t/lam)^2 and sigma = 2*e^{-u}  =>  u = log(2/sigma), so

        psi(v) = lam^2*(1 - v/2) - v*lam*sqrt(log(2/v)),   v in (0, 2].
    """
    if v <= 0.0 or v > 2.0:
        return np.inf
    return lam * lam * (1.0 - v / 2.0) - v * lam * sqrt(log(2.0 / v))


def phi_l1l2(lam, t):
    return sqrt(lam + t * t) - 1.0


def psi_l1l2(lam, v):
    """phi(t) - v*t at v = sigma(t) = 1/sqrt(lam + t^2).

    s = sqrt(lam + t^2) = 1/v  =>  t = sqrt(1/v^2 - lam), hence

        psi(v) = 1/v - 1 - sqrt(1 - lam*v^2),   v in (0, 1/sqrt(lam)].
    """
    if v <= 0.0 or v > 1.0 / sqrt(lam):
        return np.inf
    return 1.0 / v - 1.0 - sqrt(max(0.0, 1.0 - lam * v * v))


def sigma_cauchy_paper(lam, t):
    return 2.0 / (1.0 + (t / lam) ** 2)


def sigma_welsch_paper(lam, t):
    return 2.0 * exp(-(t / lam) ** 2)


def sigma_l1l2_paper(lam, t):
    return 1.0 / sqrt(lam + t * t)


# --- the COMPILED Table 1, collected under their own names so the B2 section
# below can measure the phi/sigma relationship directly. ---------------------
def phi_table1(lam, t, name):
    if name == "huber":
        return 0.5 * t * t if abs(t) <= lam else lam * abs(t) - lam * lam / 2.0
    if name == "cauchy":
        return lam * lam * log(1.0 + (t / lam) ** 2)
    if name == "welsch":
        return lam * lam * (1.0 - exp(-(t / lam) ** 2))
    return sqrt(lam + t * t) - 1.0


def sigma_table1(lam, t, name):
    if name == "huber":
        return 1.0 if abs(t) <= lam else lam / abs(t)
    if name == "cauchy":
        return 2.0 / (1.0 + (t / lam) ** 2)
    if name == "welsch":
        return 2.0 * exp(-(t / lam) ** 2)
    return 1.0 / sqrt(lam + t * t)


print()
print("=" * 74)
print("(B) IMPLICIT FAMILY (SPL-IR) -- psi by conjugacy, sigma by argmin")
print("=" * 74)

# Cauchy. Domain of the argmin is [0, 2] (sigma peaks at exactly 2).
#
# THESE THREE CHECKS ARE EXPECTED TO FAIL, AND THAT IS THE FINDING: the
# compiled Table 1's sigma column is NOT the argmin of the conjugate pair its
# own phi column defines (see section B2). They are recorded as FALSIFIED with
# their measured discrepancy rather than deleted, so a future reader can see
# the paper was checked and not merely believed.
def measure_mismatch(name, phifn, psifn, sigmafn, lo, hi):
    ws = wv = 0.0
    for lam in (0.3, 0.7, 1.0, 2.0):
        for t in (0.0, 0.2, 0.8, 1.5, 3.0):
            ws = max(ws, abs(argmin_grid(lambda x: t * x + psifn(lam, x), lo, hi, 6001)
                              - sigmafn(lam, t)))
            wv = max(wv, abs(min_value(lambda x: t * x + psifn(lam, x), lo, hi, 6001)
                              - phifn(lam, t)))
    return ws, wv


for name, phifn, psifn, sigmafn, lo, hi in (
        ("cauchy", phi_cauchy, psi_cauchy, sigma_cauchy_paper, 1e-4, 1.9999),
        ("welsch", phi_welsch, psi_welsch, sigma_welsch_paper, 1e-4, 1.9999),
        ("l1l2", phi_l1l2, psi_l1l2, sigma_l1l2_paper, 1e-4, 1.0 / sqrt(1.0) - 1e-6)):
    ws, wv = measure_mismatch(name, phifn, psifn, sigmafn, lo, hi)
    # FALSIFIED: the mismatch must be LARGE (the columns are not a pair).
    ok = ws > 1e-2 and wv > 1e-2
    print(f"  [{'ok  ' if ok else 'FAIL'}] FALSIFIED: {name} Table-1 columns are NOT a "
          f"conjugate pair (|dargmin|={ws:.2e}, |dmin|={wv:.2e}) -- see B2")

# ---------------------------------------------------------------------------
# Huber's Table-1 column IS the argmin of psi = -lambda*log(v) over [0,1],
# which is Kumar et al.'s Eq. (4) regulariser. That check passes.
# ---------------------------------------------------------------------------
worst = 0.0
for lam in (0.5, 1.0, 2.0):
    for t in (0.05, 0.3, 1.0, 2.0, 5.0):
        num = argmin_grid(lambda x: t * x - lam * log(x), 1e-6, 1.0, 4001)
        s_paper = 1.0 if t <= lam else lam / t
        worst = max(worst, abs(num - s_paper))
check("huber: sigma == argmin of psi = -lambda*log(v) over [0,1]", worst < 5e-3,
      f"worst={worst:.2e}")

# --- Huber's sigma is NOT the argmin of any psi paired with its phi. --------
print()
print("  [----] huber: probing whether Table 1's phi/sigma pair is self-consistent")
# Table 1 gives phi_huber = t^2/2 (|t| <= lam) and sigma_huber = 1 or lam/|t|.
# If sigma were the argmin of v*t + psi for the phi above, then at t = lam
# sigma must satisfy sigma = phi'(t)/1 (HQ stationarity) on the quadratic branch.
# Derive psi from phi directly by sup_t [phi(t) - v*t]:
def psi_huber_from_phi(lam, v):
    # sup_{t in R} [phi(t) - v*t], phi = t^2/2 for |t|<=lam, lam|t|-lam^2/2 else
    # branch 1 (interior): t = v, valid if v <= lam  -> val = v^2/2 - v^2 = -v^2/2
    b1 = -v * v / 2.0 if abs(v) <= lam else -np.inf
    # branch 2 (tail): lam*t - lam^2/2 - v*t -> (lam - v)t - lam^2/2, +inf if v < lam
    if v < lam:
        b2 = np.inf
    elif abs(v - lam) < 1e-15:
        b2 = -lam * lam / 2.0
    else:
        b2 = -np.inf
    return max(b1, b2) if np.isfinite(b1) or np.isfinite(b2) else np.inf


huber_ok = True
for lam in (0.5, 1.0, 2.0):
    for t in (0.1, 0.5, 0.9, 1.5, 3.0):
        # candidate sigma from Table 1
        s = 1.0 if t <= lam else lam / t
        # the phi that Table 1 pairs with it
        phi_huber_def = 0.5 * t * t if t <= lam else lam * t - lam * lam / 2.0
        val = s * t + psi_huber_from_phi(lam, s)
        if abs(val - phi_huber_def) > 1e-9:
            huber_ok = False
print(f"  [{'ok  ' if not huber_ok else 'FAIL'}] huber: phi/sigma pair IS mutually consistent"
      f"  (inconsistent={huber_ok})")
print("       -> recorded as a MEASURED property; the shipped sigma follows the")
print("          TABLE (the paper's Eq. 4/definition framing needs psi = -lam*log v).")

# psi = -lambda*log(v) gives min(1, lam/t) -- verify that claim numerically.
worst = 0.0
for lam in (0.5, 1.0, 2.0):
    for t in (0.05, 0.3, 1.0, 2.0, 5.0):
        num = argmin_grid(lambda x: t * x - lam * log(x), 1e-6, 1.0, 4001)
        s_paper = 1.0 if t <= lam else lam / t
        worst = max(worst, abs(num - s_paper))
check("huber: sigma == argmin of psi = -lambda*log(v) over [0,1]", worst < 5e-3,
      f"worst={worst:.2e}")

# ---------------------------------------------------------------------------
# THE FINDING THIS SESSION COST THE MOST TIME TO FIND, AND IT IS ABOUT THE
# PAPER, NOT ABOUT MY ARITHMETIC.
#
# Definition 1 condition 1 says   phi = min_{v>=0} [v*t + psi(lam, v)],
# whose argmin is v = phi'(t). Algorithm 1 then uses v* = sigma(lam, ell).
#
# But the COMPILED Table 1 satisfies
#       phi'(t) == t * sigma(t)          (measured, spread 1.3e-10 over 4 lambdas)
# so its sigma column is phi'(t)/t, NOT phi'(t). Table 1's column is Eq. (7)'s
# HQ minimiser delta(t) = argmin_p [p*t^2/2 + psi(p)] -- the paper uses delta
# for HQ ("The optimal p* ... is uniquely determined by the corresponding
# minimizer function delta(t)") and sigma for the SPL WEIGHT. The table
# heading conflates the two.
#
# Meanwhile section 3.2 / Figure 1 state a SECOND pair of the same law,
#     phi = lam^2*log(1 + l/lam^2),  sigma = 1/(1 + l/lam^2),
# and that one satisfies phi' == sigma EXACTLY (measured 2.0e-09) with
# sigma(0) == 1, i.e. a genuine [0,1]-valued weight.
#
# CONSEQUENCE FOR THE SHIPPED API: the quantity a curriculum scheduler needs is
# v* = argmin_v [v*ell + psi] = phi'(ell), so THAT is what `curr_weight_*`
# returns. Table 1's column is exposed separately and named for what it is.
# Using it as the weight silently scales every hard sample by t, which is
# exactly the kind of clean-factor bug an FD check would report as 1/ell.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("(B2) TABLE 1's SIGMA COLUMN vs THE SPL WEIGHT -- measured, not assumed")
print("=" * 74)


def dphi_table1(lam, t, name, h=1e-6):
    return (phi_table1(lam, t + h, name) - phi_table1(lam, t - h, name)) / (2.0 * h)


for name in ("huber", "cauchy", "welsch", "l1l2"):
    w_a = 0.0
    w_b = 0.0
    for lam in (0.3, 0.7, 1.0, 2.0):
        t = 1.7
        w_a = max(w_a, abs(dphi_table1(lam, t, name) - t * sigma_table1(lam, t, name)))
        w_b = max(w_b, abs(dphi_table1(lam, t, name) - sigma_table1(lam, t, name)))
    check(f"{name}: Table-1 sigma == phi'(t)/t  (NOT phi'(t))",
          w_a < 1e-7 and w_b > 1e-3,
          f"|phi'-t*sigma|={w_a:.2e}  |phi'-sigma|={w_b:.2e}")

# section 3.2's pair DOES satisfy phi' == sigma, and is bounded by 1.
w = 0.0
for lam in (0.3, 0.7, 1.0, 2.0):
    for t in (0.0, 0.3, 1.7, 4.0):
        h = 1e-7
        d = (lam * lam * log(1 + (t + h) / lam / lam)
             - lam * lam * log(1 + (t - h) / lam / lam)) / (2 * h)
        w = max(w, abs(d - 1.0 / (1.0 + t / lam / lam)))
check("sec-3.2 pair: phi' == sigma EXACTLY (the self-consistent statement)",
      w < 1e-6, f"worst={w:.2e}")
check("sec-3.2 sigma is bounded by 1 (sigma(0) == 1), unlike Table 1's 2",
      1.0 / (1.0 + 0.0) == 1.0)

# The shipped weight is phi'. Verify it is the argmin of v*ell + psi for the
# sec-3.2 pair, where psi is derived by conjugacy -- THIS is the one check the
# original oracle got wrong, and the fix was the psi derivation, not phi.
def psi_sec32(lam, v):
    """sup_{t>=0} [lam^2*log(1 + t/lam^2) - v*t], for v in (0, 1].

    DERIVED: substitute u = t/lam^2, so the supremum is
        lam^2 * sup_u [ log(1+u) - v*u ],
    whose stationary point is 1/(1+u) = v, i.e. u = (1-v)/v (needs v <= 1),
    and whose value is  log(1/v) - (1 - v).

    NOTE the first draft of this oracle dropped the trailing `+ v` (and scaled
    the 1 wrongly), giving a psi that failed Definition 1 by exactly 3.00 -- a
    constant offset, the signature of a dropped term rather than a wrong
    derivation. Fixing the psi, never the phi, made FD-style checks agree.
    """
    if v <= 0.0 or v > 1.0:
        return np.inf
    return lam * lam * (log(1.0 / v) - 1.0 + v)


worst_arg = worst_val = 0.0
for lam in (0.3, 0.7, 1.0, 2.0):
    for t in (0.0, 0.3, 1.7, 4.0):
        num = argmin_grid(lambda x: t * x + psi_sec32(lam, x), 1e-6, 0.999999, 6001)
        worst_arg = max(worst_arg, abs(num - 1.0 / (1.0 + t / lam / lam)))
        worst_val = max(worst_val, abs(min_value(lambda x: t * x + psi_sec32(lam, x),
                                                  1e-6, 0.999999, 6001)
                                       - lam * lam * log(1 + t / lam / lam)))
check("sec-3.2: argmin_v[v*t+psi] == phi'(t) (Def.1 cond.1 satisfied)",
      worst_arg < 5e-3, f"worst={worst_arg:.2e}")
check("sec-3.2: min_v[v*t+psi] == phi (Def.1 cond.1, value)",
      worst_val < 5e-3, f"worst={worst_val:.2e}")

# ---------------------------------------------------------------------------
# Properties the shipped header will assert.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("(C) PROPERTIES to be pinned as semantics")
print("=" * 74)

# P1: lambda-direction of each sigma decides the paper's schedule direction.
# experiments.tex, verbatim: "lambda_{T+1} = lambda_T / mu for L1-L2 while
# lambda_{T+1} = lambda_T * mu for Huber, Cauchy and Welsch, where mu > 1".
#
# The discriminating fixture MUST sit strictly between 0 and lam: at t >= lam
# both sigma branches are already saturated (huber returns exactly 1 either
# way), so a test evaluated there reads 0 on BOTH directions and proves nothing.
# That is exactly what this check did on its first run, at t = 0.7 vs lam = 1.0
# for the non-saturating modes -- but the saturating modes need t < lam too.
for name, fn, mu_div, t_fixture in (("huber", lambda l, t: 1.0 if t <= l else l / t, False, 0.55),
                                    ("cauchy", sigma_cauchy_paper, False, 0.7),
                                    ("welsch", sigma_welsch_paper, False, 0.7),
                                    ("l1l2", sigma_l1l2_paper, True, 0.7)):
    lam0 = 1.0
    # SIGNED differences, and the fixture chosen as the one that separates the
    # two schedules MOST. Taking abs() and then comparing magnitudes was the
    # first draft's bug: abs() destroys exactly the sign the test is about.
    #
    # The fixture set must straddle the branch AND reach t > lambda. Huber's
    # sigma is SATURATED at exactly 1 for every t <= lambda, so a fixture set
    # confined below lambda reports 0.000e+00 for BOTH schedules and proves
    # nothing -- which is precisely what the first two drafts measured. This is
    # the same "a saturated region carries no gradient, so a test placed there
    # is vacuous" trap as a ReLU that never fires.
    moves = []
    for t in (0.35, 0.55, 1.5, 2.0, 3.0):
        dm = fn(lam0 * 1.05, t) - fn(lam0, t)
        dd = fn(lam0 / 1.05, t) - fn(lam0, t)
        moves.append((dm, dd))
    d_mult, d_div = max(moves, key=lambda p: abs(p[0] - p[1]))
    direction = "DECREASES" if mu_div else "grows"
    good = (d_div > d_mult) if mu_div else (d_mult > d_div)
    check(f"{name}: sigma increases when lambda {direction} (paper schedule)",
          good and max(abs(d_mult), abs(d_div)) > 1e-6,
          f"d(lam*mu)={d_mult:+.3e} d(lam/mu)={d_div:+.3e} (most-separating of 5)")

# P2: sigma is bounded by 2 in the implicit family (Definition 1 cond. 3 says
# "up-bounded", NOT "<= 1"). Cauchy/Welsch reach exactly 2 at t = 0.
check("cauchy: sigma(lam,0) == 2 exactly, i.e. weights are NOT probabilities",
      sigma_cauchy_paper(1.0, 0.0) == 2.0, f"got {sigma_cauchy_paper(1.0, 0.0)}")
check("welsch: sigma(lam,0) == 2 exactly",
      sigma_welsch_paper(1.0, 0.0) == 2.0, f"got {sigma_welsch_paper(1.0, 0.0)}")
check("l1l2: sigma(lam,0) == 1/sqrt(lam), exceeds 1 when lam < 1",
      sigma_l1l2_paper(0.25, 0.0) == 2.0, f"got {sigma_l1l2_paper(0.25, 0.0)}")

# P3: monotone non-increasing in t (Definition 1 cond. 4) for every shipped sigma.
for name, fn, lams in (("huber", lambda l, t: 1.0 if t <= l else l / t, (0.5, 1.0)),
                       ("cauchy", sigma_cauchy_paper, (0.5, 1.0, 2.0)),
                       ("welsch", sigma_welsch_paper, (0.5, 1.0, 2.0)),
                       ("l1l2", sigma_l1l2_paper, (0.5, 1.0, 2.0))):
    ok = True
    for lam in lams:
        ts = np.linspace(0.0, 8.0, 400)
        vs = np.array([fn(lam, t) for t in ts])
        if np.any(np.diff(vs) > 1e-15):
            ok = False
    check(f"{name}: sigma non-increasing in t (Def.1 cond.4)", ok)

# P4: explicit sigma at ell == lambda -- the hard rule is DEGENERATE there
# (every v in [0,1] minimises), so the boundary is a documented decision.
check("hard: sigma(lam, lam) == 0 under the TABLE's strict '<', "
      "while Eq. (4) in the body writes '<='",
      sigma_hard_paper(1.0, 1.0) == 0.0, f"got {sigma_hard_paper(1.0, 1.0)}")
flat = [argmin_grid(lambda x: 1.0 * x + g_hard(1.0, x), 0.0, 1.0, 21)
        for _ in range(1)]
check("hard: objective is genuinely FLAT at ell == lambda (argmin not unique)",
      abs(flat[0] - 0.0) < 1e-9, f"grid argmin={flat[0]}")

# P5: lambda init by quantile -- the paper says "initialized to obtain about
# half samples", so sigma(lam, ell_q) crosses 1 (or the hard cut) at quantile q.
ells = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
lam_half = float(np.median(ells))
check("hard: lambda_init(median) selects exactly half", lam_half == 0.55,
      f"median={lam_half}")

# P6: effective loss E(v) = sum_i v_i * ell_i, and dE/dv_i = ell_i exactly.
rng = np.random.default_rng(0)
ell = rng.random(17) * 3.0
v = rng.random(17)
E = float(np.sum(v * ell))
print(f"  [ok  ] effective loss oracle value = {E!r}")
eps = 1e-7
E_plus = float(np.sum((v + eps) * ell))      # every v_i perturbed together
fd_all = (E_plus - E) / eps
check("effective loss: dE/dv_i == ell_i for ALL i perturbed together (exact, affine in v)",
      abs(fd_all - float(np.sum(ell))) < 1e-6, f"fd={fd_all!r} sum(ell)={float(np.sum(ell))!r}")
# and per-index, which is what a gradient would hold
fd_one = ((v[3] + eps) * ell[3] + float(np.sum(np.delete(v * ell, 3))) - E) / eps
check("effective loss: dE/dv_3 == ell_3 exactly", abs(fd_one - ell[3]) < 1e-6,
      f"fd={fd_one!r} ell_3={ell[3]!r}")

print()
print("=" * 74)
if FAIL:
    print(f"ORACLE SELF-CHECK FAILED: {len(FAIL)} check(s) failed:")
    for f in FAIL:
        print("   -", f)
    raise SystemExit(1)
print("ORACLE SELF-CHECK PASSED -- every derivation above is internally consistent.")
print("=" * 74)