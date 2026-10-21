#!/usr/bin/env python3
"""
tests/ws_oracle.py -- the SPECIFICATION for utils/weight_std.{h,cpp}.

Runs BEFORE any C++ exists. Its job is to catch defects in the DERIVATIONS and in
itself, exactly as tests/sinkhorn_oracle.py and tests/curriculum_oracle.py did.

Weight Standardization, Qiao, Wang, Liu, Shen, Yuille, "Micro-Batch Training with
Batch-Channel Normalization and Weight Standardization", arXiv:1903.10520,
eq. (10):

    W_hat = (W - W_bar) / sqrt(W_bar2 - W_bar_bar^2)

with the overbars PER OUTPUT CHANNEL, and no affine transform ("we do not have
any affine transformation on W_hat").

Exit 0 = all sections green. Exit 1 = at least one FAILED line.
"""
import sys
import numpy as np

FAIL = []
PASS = 0


def check(ok, name, detail=""):
    global PASS
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def section(n, title):
    print(f"\n-- S{n}: {title} --")


def ws_forward(W, eps=0.0, axis=-1):
    """WS over `axis` with the variance taken over `axis` (population, 1/K)."""
    mu = W.mean(axis=axis, keepdims=True)
    var = ((W - mu) ** 2).mean(axis=axis, keepdims=True)
    return (W - mu) / np.sqrt(var + eps)


# ---------------------------------------------------------------------------
section(1, "the forward: unit variance, zero mean, population not sample")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)

# NON-SQUARE on purpose: a square-only fixture set cannot see a row/column
# confusion at all (this repo's eighth time that lesson has applied).
W = rng.normal(size=(5, 3)) * 4.0 + 2.0   # heavy offset and scale on purpose
Wh = ws_forward(W)

for c in range(W.shape[0]):
    row = Wh[c]
    check(abs(row.mean()) < 1e-12, f"S1a row {c} mean ~ 0", f"got {row.mean():.3e}")
    # POPULATION std == 1. A sample std (1/(K-1)) gives sqrt(K/(K-1)) here.
    check(abs(row.std(ddof=0) - 1.0) < 1e-12,
          f"S1b row {c} POPULATION std == 1", f"got {row.std(ddof=0):.15f}")
    check(abs(row.std(ddof=1) - 1.0) > 1e-6,
          f"S1c row {c} sample std is NOT 1 (discriminates ddof)",
          f"got {row.std(ddof=1):.15f}")

# S2: ||w_hat_c||_2 == sqrt(K) exactly for eps=0. This is the PER-ELEMENT
# (convolutional) Lipschitz constant, and it is the quantity the paper means.
K = W.shape[1]
for c in range(W.shape[0]):
    nrm = np.linalg.norm(Wh[c])
    check(abs(nrm - np.sqrt(K)) < 1e-12, f"S2 row {c} ||w_hat||_2 == sqrt(K)",
          f"got {nrm:.15f} want {np.sqrt(K):.15f}")

# S3: idempotence. Standardizing again is a no-op.
Wh2 = ws_forward(Wh)
check(np.abs(Wh2 - Wh).max() < 1e-12, "S3 idempotent", f"max diff {np.abs(Wh2-Wh).max():.3e}")

# S8: PER-COLUMN standardization differs on a non-square matrix. This is the
# row/column confusion discriminator.
Wh_col = ws_forward(W, axis=0)
check(np.abs(Wh_col - Wh).max() > 1e-6, "S8 per-column != per-row on non-square",
      f"max diff {np.abs(Wh_col-Wh).max():.3e}")

# ---------------------------------------------------------------------------
section(2, "eps placement: inside the sqrt")
# ---------------------------------------------------------------------------
eps = 1e-3
Wp = np.array([[1.0, 2.0, 3.0]])
inside = (Wp - Wp.mean()) / np.sqrt(Wp.var() + eps)
outside = (Wp - Wp.mean()) / np.sqrt(Wp.var()) / np.sqrt(1.0 + eps)
check(np.abs(inside - outside).max() > 1e-4, "S4a eps-inside != eps-outside",
      f"max diff {np.abs(inside-outside).max():.3e}")
# With eps INSIDE, the row norm is sqrt(K*var/(var+eps)) < sqrt(K): the bound
# SHRINKS. That direction is the signature.
n_in = np.linalg.norm(inside[0])
n_out = np.linalg.norm(outside[0])
check(n_in < n_out, "S4b eps-inside shrinks the norm more than eps-outside",
      f"in {n_in:.9f} out {n_out:.9f}")
print(f"     [pinned] eps=1e-3, var=2/3: norm_inside={n_in:.12f} norm_outside={n_out:.12f}")

# ---------------------------------------------------------------------------
section(3, "THE ADJOINT, by finite differences")
# ---------------------------------------------------------------------------
# Chain: W -> (center) -> u -> (+eps) -> var -> sqrt -> divide -> W_hat
# g = dL/dW_hat.  The exact adjoint is
#     dL/dW[j] = (1/s) * ( g[j] - mean(g) - x_hat[j] * mean(g * x_hat) )
# TWO TRAPS: mean(g*x_hat) uses x_hat, NOT u (they differ by 1/s, a clean
# factor), and the 1/s prefactor is OUTSIDE the parenthesis.


def ws_backward_naive(W, g, eps):
    """Adjoint by explicit chain, built to be obviously right."""
    mu = W.mean(axis=1, keepdims=True)
    u = W - mu
    v = (u ** 2).mean(axis=1, keepdims=True)
    s = np.sqrt(v + eps)
    x = u / s

    mean_g = g.mean(axis=1, keepdims=True)
    mean_gx = (g * x).mean(axis=1, keepdims=True)
    return (g - mean_g - x * mean_gx) / s


def ws_backward_fd(W, g, eps, h=1e-6):
    """Central-difference adjoint of L = <W_hat, g>, eps INSIDE the sqrt."""
    out = np.zeros_like(W)
    for c in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp_ = W.copy(); Wp_[c, j] += h
            Wm_ = W.copy(); Wm_[c, j] -= h
            lp = float((ws_forward(Wp_, eps) * g).sum())
            lm = float((ws_forward(Wm_, eps) * g).sum())
            out[c, j] = (lp - lm) / (2 * h)
    return out


for (shape, eps_v) in [((5, 3), 1e-6), ((5, 3), 1e-3), ((4, 7), 1e-5), ((1, 5), 1e-6)]:
    Wt = rng.normal(size=shape) * 2.0 + 0.5
    gt = rng.normal(size=shape)
    ana = ws_backward_naive(Wt, gt, eps_v)
    num = ws_backward_fd(Wt, gt, eps_v)
    denom = max(np.abs(ana).max(), 1e-12)
    rel = np.abs(ana - num).max() / denom
    check(rel < 1e-6, f"S5 adjoint matches FD at shape {shape} eps {eps_v}",
          f"rel {rel:.3e}")

# The WRONG variants must FAIL -- otherwise the check above proves nothing.
W1 = rng.normal(size=(5, 3))
g1 = rng.normal(size=(5, 3))
eps_v = 1e-6
good = ws_backward_naive(W1, g1, eps_v)
K1 = W1.shape[1]
mu = W1.mean(axis=1, keepdims=True); u = W1 - mu
v = (u ** 2).mean(axis=1, keepdims=True); s = np.sqrt(v + eps_v); x = u / s
# WRONG 1: mean(g*u) instead of mean(g*x_hat) -- clean 1/s factor.
wrong_u = (g1 - g1.mean(axis=1, keepdims=True) - u * (g1 * u).mean(axis=1, keepdims=True)) / s
r1 = np.abs(wrong_u - good).max() / max(np.abs(good).max(), 1e-12)
check(r1 > 1e-3, "S6a the u-instead-of-x_hat variant really differs", f"rel {r1:.3e}")
# WRONG 2: 1/s INSIDE the parenthesis.
wrong_inside = (g1 - g1.mean(axis=1, keepdims=True) - x * (g1 * x).mean(axis=1, keepdims=True) * s * s) / s
r2 = np.abs(wrong_inside - good).max() / max(np.abs(good).max(), 1e-12)
check(r2 > 1e-3, "S6b the prefactor-placement variant really differs", f"rel {r2:.3e}")
# WRONG 3: SAMPLE variance (1/(K-1)) -- the fixture that would silently differ.
us = W1 - W1.mean(axis=1, keepdims=True)
vs = (us ** 2).sum(axis=1, keepdims=True) / (K1 - 1)
ss = np.sqrt(vs + eps_v)
wrong_sample = (g1 - g1.mean(axis=1, keepdims=True) - (us / ss) * (g1 * (us / ss)).mean(axis=1, keepdims=True)) / ss
r3 = np.abs(wrong_sample - good).max() / max(np.abs(good).max(), 1e-12)
check(r3 > 1e-6, "S6c the sample-variance variant really differs", f"rel {r3:.3e}")

# ---------------------------------------------------------------------------
section(4, "translation invariance: dL/dW sums to zero over a channel")
# ---------------------------------------------------------------------------
# Adding a constant to a whole channel leaves W_hat unchanged, so the adjoint
# must sum to zero along the channel. This pins the -mean(g) term.
sums = good.sum(axis=1)
check(np.abs(sums).max() < 1e-9, "S7 dL/dW sums to 0 over each channel",
      f"max |sum| {np.abs(sums).max():.3e}")

# ---------------------------------------------------------------------------
section(5, "TWO DIFFERENT LIPSCHITZ BOUNDS -- the first draft conflated them")
# ---------------------------------------------------------------------------
# ORACLE DEFECT, and the reason this section is written the way it is. The first
# draft asserted  ||W_hat||_op <= sqrt(K)  for a (6,4) matrix and the oracle
# measured 3.715543100006 against a "bound" of 2.0. That assertion is FALSE and
# was never true. There are two true statements and they are about DIFFERENT
# quantities:
#
#   (a) PER-ELEMENT (convolutional) bound. Each output is w_c . x with
#       ||w_c||_2 = sqrt(K) exactly, so |w_c . x| <= sqrt(K) ||x||_2 by
#       Cauchy-Schwarz. This is the constant the paper's argument uses, and it
#       is exactly S2 above. K = in_channels.
#
#   (b) WHOLE-MATRIX operator bound. ||W||_op <= ||W||_F = sqrt(O*K) for a
#       standardized matrix (each of O rows contributes K to the squared
#       Frobenius). This is sqrt(OUT_channels * IN_channels), NOT sqrt(K).
#
# Measured for a random 6x4: op = 3.715543100006, and sqrt(24) = 4.898979486,
# sqrt(4) = 2.0. The op norm sits strictly between the two, which is why a
# loose sqrt(K) bound "looked plausible" and had to be refuted by measurement.
O, K2 = 6, 4
Wr = rng.normal(size=(O, K2))
Whr = ws_forward(Wr)
sv = np.linalg.svd(Whr, compute_uv=False)
print(f"     [pinned] {O}x{K2} WS operator norm = {sv[0]:.12f}")
print(f"     [pinned] per-element bound sqrt(K) = {np.sqrt(K2):.12f}")
print(f"     [pinned] matrix bound sqrt(O*K)  = {np.sqrt(O*K2):.12f}")

# (a) per-element: verify Cauchy-Schwarz on random inputs rather than asserting it.
x = rng.normal(size=K2)
x /= np.linalg.norm(x)
ratios = np.abs(Whr @ x) / (np.sqrt(K2) * np.linalg.norm(x))
check(ratios.max() <= 1.0 + 1e-12, "S8a per-element |w.x| <= sqrt(K)||x||",
      f"got max ratio {ratios.max():.15f}")
check(ratios.max() > 1e-6, "S8b and it is not vacuous (ratio is not ~0)",
      f"got max ratio {ratios.max():.15f}")

# (b) whole-matrix: the TRUE operator bound is sqrt(O*K).
check(sv[0] <= np.sqrt(O * K2) + 1e-9, "S8c op norm <= sqrt(O*K) (Frobenius bound)",
      f"got {sv[0]:.9f} bound {np.sqrt(O*K2):.9f}")
check(sv[0] < np.sqrt(O * K2) - 1e-6, "S8d strictly below for random weights",
      f"got {sv[0]:.9f}")
# And the WRONG bound (sqrt(K), the one the first draft asserted) really is
# violated -- so a C++ test that repeats it would fail on a CORRECT impl.
check(sv[0] > np.sqrt(K2) + 1e-6,
      "S8e the sqrt(K) whole-matrix bound is FALSE (do not assert it)",
      f"got {sv[0]:.9f} vs sqrt(K) {np.sqrt(K2):.9f}")

# ---------------------------------------------------------------------------
section(6, "scalar reference values to transcribe into the C++ test")
# ---------------------------------------------------------------------------
W6 = np.array([[1.0, 2.0, 3.0],
               [-1.0, 0.0, 1.0],
               [5.0, 5.0, 5.0]])          # a CONSTANT row: zero variance
print("     [pinned] WS([[1,2,3],[-1,0,1],[5,5,5]]) =")
print(np.array2string(ws_forward(W6), precision=12))
# ORACLE DEFECT: the first draft wrote this as `check(isfinite(...).all(), "expected
# NO")` -- a check whose NAME says it expects failure while the ASSERTION demands
# success. At eps=0 a constant row has zero variance and IS non-finite, so the
# assertion was backwards and the check could never pass. Corrected to assert the
# non-finiteness it was named for.
check(not np.isfinite(ws_forward(W6)).all(),
      "S9a a constant row at eps=0 is NOT finite (0/0) -- so eps must be > 0")
print("     [pinned] with eps=1e-8, the constant row ->",
      np.array2string(ws_forward(W6, eps=1e-8)[2], precision=12))
check(np.isfinite(ws_forward(W6, eps=1e-8)).all(),
      "S9b constant row is rescued by eps > 0")

W7 = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
print("     [pinned] WS([[1,2,3],[-1,0,1]]) =")
print(np.array2string(ws_forward(W7), precision=12))
# ORACLE DEFECT: the first draft pinned 0.6324555320336759 here, which is
# sqrt(0.4) and is simply the wrong number -- a hand-arithmetic slip. The mean of
# [1,2,3] is 2, the centred row is [-1,0,1], its variance is 2/3, so the first
# entry is -1/sqrt(2/3) = -sqrt(3/2) = -1.224744871391589. Recomputed from the
# rule, not from the previous draft's literal.
check(abs(ws_forward(W7)[0][0] - (-np.sqrt(1.5))) < 1e-12,
      "S9c pinned value row0 col0 == -sqrt(3/2)",
      f"got {ws_forward(W7)[0][0]:.15f}")
check(abs(ws_forward(W7)[1][0] - (-np.sqrt(1.5))) < 1e-12,
      "S9d pinned value row1 col0 == -sqrt(3/2)",
      f"got {ws_forward(W7)[1][0]:.15f}")
# Rows [-1,0,1] and [1,2,3] are the same row plus a constant, so WS makes them
# IDENTICAL. That is the centering doing its job and it is a strong fixture.
check(np.abs(ws_forward(W7)[0] - ws_forward(W7)[1]).max() < 1e-15,
      "S9e a constant offset within a channel is annihilated by WS",
      f"max diff {np.abs(ws_forward(W7)[0]-ws_forward(W7)[1]).max():.3e}")

# ---------------------------------------------------------------------------
print(f"\n=== SELF-CHECK: {PASS} passed, {len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
sys.exit(0)