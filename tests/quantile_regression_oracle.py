#!/usr/bin/env python3
"""
tests/quantile_regression_oracle.py -- the float64 numpy SPECIFICATION for
the linear quantile-regression layer
(include/nn/layers/dense/quantile_regression.{h,cpp}).

THIS ORACLE IS THE SPECIFICATION. It was written and run BEFORE any C++ for
this feature existed. Its discriminating power is MEASURED, not narrated:
tests/quantile_regression_mutate_oracle.py breaks this oracle in specific
ways and requires that it FAIL each time.

WHY THIS FEATURE, MEASURED NOT ASSUMED (2026-11-13)
====================================================

    grep -rinE 'quantile.regress|pinball|quantile.?head|non.?crossing' \
            include/ tests/ --include=*.h --include=*.cpp
    -> include/nn/utils/conformal.h:11,25,74
         (the CQR paper, and: "utils/distribution_losses.h ships QuantileLoss
          (the pinball loss). That is the TRAINING LOSS CQR needs -- and it is
          the strongest evidence that CQR belongs here: a shipped primitive
          with NO CONSUMER.")
    -> include/nn/utils/isotonic_regression.h:134
         ("QUANTILE REGRESSION / CQR-style monotone quantile fitting. A
          different objective (pinball loss, not squared loss).")
    -> include/nn/utils/distribution_losses.h:106  (the QuantileLoss class)
    -> tests/test_distribution_losses.cpp          (its own unit tests)

ZERO IMPLEMENTATION of a quantile REGRESSION. The repo ships the conformal
machinery that CONSUMES quantile predictions:

    Tensor cqr_score(const Tensor& lower, const Tensor& upper, const Tensor& y);

and nothing anywhere in ~250 layers PRODUCES them.

THE MODEL
=========

For a response y and features x in R^D, and K quantile levels

    0 < tau_1 < tau_2 < ... < tau_K < 1,

the linear quantile regression (Koenker and Bassett, "Regression Quantiles",
Econometrica 46(1):33-50, 1978 -- verified 2026-11-13 by title search against
the Econometric Society's own index page; NO arXiv ID is asserted, the paper
predates arXiv by three decades) has one linear predictor PER LEVEL:

    q_k(x) = x . w_k + b_k,     k = 1..K.

The levels are fitted INDEPENDENTLY: the k-th level's pinball loss depends
only on the k-th column of the predictions and only on row k of the weight
matrix. This is the classical estimator, and it has the classical pathology:
the fitted quantile functions CAN CROSS (q_{tau_1}(x) > q_{tau_2}(x) at some
x), which is exactly what breaks a conformalised-quantile-regression
interval. So the layer ships a crossing DIAGNOSTIC and an explicit pointwise
monotone REARRANGEMENT, and says which one is the model's output.

THE PINBALL (QUANTILE) LOSS
============================

For level tau and residual e = y - pred,

    rho_tau(e) = e * tau          if e >= 0    (under-prediction: pred <= y)
               = e * (tau - 1)    if e <  0    (over-prediction:  pred >  y)

which is the INTEGRAL OF ITS OWN SUBGRADIENT -- the definition-level form,

    rho_tau(u) = INTEGRAL_0^u ( tau - 1{v < 0} ) dv,                   (1)

which this oracle checks against scipy.integrate.quad.

The subgradient of rho_tau at e = 0 is the SET [tau - 1, tau]; the repo's
shipped utils/distribution_losses.h QuantileLoss picks the RIGHT end (tau),
i.e. the "e >= 0" branch, and the layer MUST pick the same one so that the
K = 1 layer and the shipped loss agree exactly.

THE REDUCTION
==============

    L = (1/N) * sum_i sum_k rho_{tau_k}( y_i - pred_ik )

MEAN OVER THE BATCH, SUM OVER THE LEVELS -- the same reduction the shipped
QuantileLoss uses (it divides by N, the row count, and sums the columns). The
consequence is that the total loss SCALES WITH K, and so does the effective
learning rate. This is deliberate: it is what makes the K = 1 layer
BIT-FOR-BIT equal to the shipped QuantileLoss on the same data, which is the
strongest available composition check, and it pins the loss, the gradient, the
tie convention and the reduction simultaneously.

WHAT IS CHECKED HERE
====================

  A. rho_tau against (1) by quadrature -- definition level.
  B. rho_tau against the closed form tau*u - min(u, 0).
  C. The gradient against central finite differences, away from the kinks.
  D. The kink value exactly, against the shipped convention.
  E. The intercept-only minimiser against the closed-form inverse-ECDF sample
     quantile AND against a brute-force grid search AND against scipy.
  F. The layer's forward against the direct formula.
  G. The layer's analytic gradients against finite differences of the TOTAL
     loss. The pinball objective is PIECEWISE LINEAR in the parameters, so a
     central difference that does not cross a kink is EXACT up to roundoff --
     the check is tight to ~1e-10, not merely ~1e-6.
  H. Level decoupling: the total-loss gradient w.r.t. level k's parameters
     equals the SINGLE-level loss gradient for that level.
  I. Crossing: the diagnostic and the monotone rearrangement, on constructed
     and on HUNTED fixtures.
  J. The C++ fixture constants.

Python: numpy + scipy. No repository code is imported: the oracle is
independent of the implementation by construction.
"""

import sys
import numpy as np
from scipy import integrate
from scipy import optimize

PASS = 0
FAIL = 0
FAILURES = []


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(msg)
        print("  [FAIL] " + msg)


def close(a, b, tol=1e-12, msg=""):
    a = float(a)
    b = float(b)
    d = abs(a - b)
    scale = max(abs(a), abs(b))
    ok = d <= tol if scale == 0.0 else d / scale <= tol
    check(ok, "%s (got %.17g, want %.17g, absdiff %.3g, rel %.3g)"
          % (msg, a, b, d, 0.0 if scale == 0.0 else d / scale))


def section(name):
    print("-- " + name)


# =============================================================================
# THE PINBALL LOSS
# =============================================================================

def rho_tau(e, tau):
    """The pinball loss, exactly as the shipped QuantileLoss defines it."""
    e = np.asarray(e, dtype=np.float64)
    return np.where(e >= 0.0, e * tau, e * (tau - 1.0))


def rho_tau_grad_pred(e, tau):
    """dL/d(pred) for the pinball loss, with the shipped tie convention."""
    e = np.asarray(e, dtype=np.float64)
    return np.where(e >= 0.0, -tau, 1.0 - tau)


# =============================================================================
# A. rho_tau AGAINST THE QUADRATURE OF ITS OWN SUBGRADIENT
# =============================================================================

section("A. rho_tau against the integral of its own subgradient (definition level)")
for tau in (0.05, 0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 0.95):
    for u in (-3.7, -1.0, -0.25, -1e-9, 0.0, 1e-9, 0.25, 1.0, 3.7):
        # rho_tau(u) = integral_0^u ( tau - 1{v < 0} ) dv
        # Integrate in two pieces so the kink at v = 0 is never straddled.
        #
        # THE ORIENTATION MATTERS: scipy.integrate.quad(a, b) returns
        # INTEGRAL_a^b, so for u < 0 the call quad(f, u, 0) is the NEGATIVE
        # of integral_0^u and has to be negated back. The first draft of this
        # oracle got that backwards and every negative-u check failed by a
        # factor of exactly -1 -- which is the fingerprint of an orientation
        # error rather than of a wrong definition.
        if u >= 0.0:
            val, err = integrate.quad(
                lambda v: tau - (1.0 if v < 0.0 else 0.0), 0.0, u,
                limit=200, points=[])
        else:
            val, err = integrate.quad(
                lambda v: tau - (1.0 if v < 0.0 else 0.0), u, 0.0,
                limit=200, points=[])
            val = -val
        close(val, rho_tau(u, tau), 1e-12,
              "tau=%.2f u=%.17g: quadrature matches the closed form" % (tau, u))

# =============================================================================
# B. rho_tau AGAINST tau*u - min(u, 0)
# =============================================================================

section("B. rho_tau against the closed form tau*u - min(u, 0)")
for tau in (0.05, 0.25, 0.5, 0.75, 0.95):
    for u in (-2.5, -0.125, 0.0, 0.125, 2.5):
        close(tau * u - min(u, 0.0), rho_tau(u, tau), 1e-15,
              "tau=%.2f u=%.17g: tau*u - min(u,0)" % (tau, u))

# =============================================================================
# C. THE GRADIENT AGAINST CENTRAL FINITE DIFFERENCES
# =============================================================================

section("C. the gradient against central finite differences")
rng = np.random.default_rng(20261113)
for tau in (0.05, 0.3, 0.5, 0.8, 0.95):
    for u in (-2.3, -0.4, 0.4, 2.3):
        eps = 1e-6
        fd = (rho_tau(u + eps, tau) - rho_tau(u - eps, tau)) / (2.0 * eps)
        ana = -rho_tau_grad_pred(u, tau)   # d/de, then negate for d/dpred
        close(ana, fd, 1e-6,
              "tau=%.2f u=%.17g: d/de matches the finite difference" % (tau, u))

# A random sweep, to catch a branch that is right only on the axis.
for _ in range(400):
    tau = float(rng.uniform(0.01, 0.99))
    u = float(rng.uniform(-5.0, 5.0))
    eps = 1e-6
    fd = (rho_tau(u + eps, tau) - rho_tau(u - eps, tau)) / (2.0 * eps)
    ana = -rho_tau_grad_pred(u, tau)
    if abs(ana - fd) > 1e-6:
        check(False, "random sweep: tau=%.17g u=%.17g ana=%.17g fd=%.17g"
              % (tau, u, ana, fd))
        break
else:
    check(True, "random sweep: 400 random (tau, u) gradient checks agree")

# =============================================================================
# D. THE KINK, EXACTLY
# =============================================================================

section("D. the kink: the shipped convention picks the under-prediction branch")
close(rho_tau(0.0, 0.3), 0.0, 0.0, "rho_tau(0) = 0 exactly")
close(rho_tau_grad_pred(0.0, 0.3), -0.3, 0.0,
      "dL/dpred at e == 0 is -tau (the under-prediction branch, as shipped)")
close(rho_tau_grad_pred(0.0, 0.7), -0.7, 0.0,
      "dL/dpred at e == 0 is -tau for tau = 0.7 too")

# =============================================================================
# E. THE INTERCEPT-ONLY MINIMISER
# =============================================================================

def inverse_ecdf_quantile(sample, tau):
    """inf{ b : #{y_i <= b}/N >= tau } -- the inverse ECDF quantile."""
    s = np.sort(np.asarray(sample, dtype=np.float64))
    n = s.size
    idx = int(np.ceil(n * tau)) - 1
    if idx < 0:
        idx = 0
    if idx > n - 1:
        idx = n - 1
    return float(s[idx])


def pinball_objective_intercept(sample, tau, b):
    """(1/N) * sum_i rho_tau(y_i - b) -- the N = 1, D = 1, X = 0 objective."""
    s = np.asarray(sample, dtype=np.float64)
    return float(np.sum(rho_tau(s - b, tau)) / s.size)


section("E. the intercept-only minimiser: closed form, grid search, scipy")
grid_sample = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
GRID_LO = float(grid_sample.min() - 5.0)
GRID_HI = float(grid_sample.max() + 5.0)
GRID_PTS = 400001
GRID_STEP = (GRID_HI - GRID_LO) / (GRID_PTS - 1)
for tau in (0.05, 0.25, 0.75, 0.95):
    closed = inverse_ecdf_quantile(grid_sample, tau)
    # Brute force over a dense grid.
    grid = np.linspace(GRID_LO, GRID_HI, GRID_PTS)
    vals = np.array([pinball_objective_intercept(grid_sample, tau, b) for b in grid])
    best = float(grid[int(np.argmin(vals))])
    closed_val = pinball_objective_intercept(grid_sample, tau, closed)
    # THE COMPARISON IS ONE-SIDED, AND THAT IS THE POINT.
    #
    # The objective is PIECEWISE LINEAR with its kinks exactly at the sample
    # values, so the minimiser is AT a sample value. The grid cannot land
    # there (its spacing is 3.75e-5), so the grid's minimum is strictly
    # ABOVE the true minimum -- by an amount of order the grid spacing. The
    # first draft of this oracle compared the two values with a TWO-SIDED
    # equality and failed by exactly 3.75e-7 = one grid step. The correct
    # statement is that the closed form attains a value NO GREATER than the
    # grid's best, and the grid's best is within the grid's resolution of it.
    check(closed_val <= float(vals.min()) + 1e-12,
          "tau=%.2f: the closed-form quantile attains a value no greater than "
          "the grid's minimum (closed %.17g, grid %.17g)"
          % (tau, closed_val, float(vals.min())))
    check(abs(best - closed) <= 2.0 * GRID_STEP,
          "tau=%.2f: the grid argmin (%.17g) is within the grid's resolution of "
          "the closed-form quantile (%.17g)" % (tau, best, closed))
    # scipy, independently. Its BOUNDED method cannot resolve a kink to
    # machine precision -- it converges on a FLAT piece -- so the comparison
    # is on the VALUE, with a tolerance that reflects that.
    res = optimize.minimize_scalar(
        lambda b: pinball_objective_intercept(grid_sample, tau, b),
        bounds=(GRID_LO, GRID_HI),
        method="bounded", options={"xatol": 1e-14})
    close(closed_val, float(res.fun), 1e-7,
          "tau=%.2f: the closed-form quantile's value matches scipy's optimum "
          "value" % tau)
    # N*tau must not be an integer for the minimiser to be UNIQUE. Assert the
    # interval collapses: moving to the next order statistic must be worse.
    n = grid_sample.size
    k = int(np.ceil(n * tau))
    lo = float(np.sort(grid_sample)[k - 1])
    hi = float(np.sort(grid_sample)[k]) if k < n else lo
    check(abs(n * tau - round(n * tau)) > 1e-12,
          "tau=%.2f: N*tau is not an integer, so the minimiser is unique" % tau)
    if hi > lo:
        check(pinball_objective_intercept(grid_sample, tau, hi)
              > pinball_objective_intercept(grid_sample, tau, closed) + 1e-12,
              "tau=%.2f: the next order statistic is strictly worse" % tau)

# A non-grid sample, where the closed form must still hold.
ragged = np.array([3.0, -1.0, 7.5, 0.25, 2.0, 9.0, 4.5, -2.0, 6.0, 1.0])
for tau in (0.05, 0.25, 0.75, 0.95):
    closed = inverse_ecdf_quantile(ragged, tau)
    grid = np.linspace(ragged.min() - 5.0, ragged.max() + 5.0, GRID_PTS)
    vals = np.array([pinball_objective_intercept(ragged, tau, b) for b in grid])
    closed_val = pinball_objective_intercept(ragged, tau, closed)
    check(closed_val <= float(vals.min()) + 1e-12,
          "tau=%.2f: ragged sample, the closed form attains a value no greater "
          "than the grid's minimum" % tau)

# =============================================================================
# E2. THE INTERCEPT-ONLY TRAINING RUN -- AND THEREFORE THE C++ TOLERANCE
# =============================================================================
# The pinball objective is PIECEWISE LINEAR, so gradient descent on it does
# not converge to a point: it OSCILLATES around the minimiser. The test's
# tolerance must therefore be DERIVED from a simulation of the exact update
# the C++ performs, not guessed.
#
# The update the C++ performs, derived from the shipped code:
#   pinball_gradient gives dL/dpred_ik = -(1/N) * (tau_k if e >= 0 else tau_k - 1)
#   backward accumulates grad_bias_k = sum_i dL/dpred_ik
#   update_weights does bias_k -= lr * grad_bias_k
# With S = #{y_i < b} and T = #{y_i >= b} (S + T = N):
#   grad_bias_k = (1/N) * ( -tau_k * T + (1 - tau_k) * S ) = (S - N*tau_k) / N
# so b moves DOWN when S > N*tau_k and UP when S < N*tau_k. The fixed point
# is S = N*tau_k, which for non-integral N*tau_k is the unique minimiser
# y_(ceil(N*tau_k)), reached by oscillation.
#
# WHY THE CONVERGENCE IS SLOW AT A SMALL LEARNING RATE, AND WHY THAT WAS THE
# FIRST DRAFT'S BUG: the gradient magnitude is |S - N*tau_k| / N, and the
# INTERCEPT HAS TO TRAVEL the whole gap between the sample values, one step
# of size lr*|S - N*tau_k|/N at a time. At lr = 0.05 the per-step move is
# tiny and 400 steps do not arrive (measured: the worst level was still
# 1.315 away from its target). The learning rate has to be LARGE ENOUGH that
# the travel completes -- and the tolerance has to be taken over the whole
# TRAJECTORY, because the last steps oscillate around the target.
section("E2. the intercept-only training run, and the tolerance it implies")

F4_TAUS_LIST = [0.05, 0.25, 0.75, 0.95]


def simulate_intercept_training(sample, taus, lr, steps, tail_frac=0.2):
    """Mirror the C++ update exactly: bias_k -= lr * (S - N*tau_k)/N.

    Returns the FINAL intercepts and the worst deviation seen over the TAIL
    of the trajectory -- the STEADY-STATE OSCILLATION around the target.

    The tail, not the whole trajectory, because the trajectory's START is far
    from the target by construction (the intercept starts at 0 and the target
    is up to 9.5 away), so a whole-trajectory measure is dominated by the
    TRAVEL and says nothing about how tightly the fit settles. The first
    draft measured the whole trajectory and reported a worst deviation of
    9.45 -- which is just the distance the intercept had to travel.
    """
    n = sample.size
    b = np.zeros(len(taus))
    tail_start = int(steps * (1.0 - tail_frac))
    worst = 0.0
    for step in range(steps):
        for k, tau in enumerate(taus):
            S = int(np.sum(sample < b[k]))
            grad = (S - n * tau) / n
            b[k] -= lr * grad
        if step >= tail_start:
            for k, tau in enumerate(taus):
                worst = max(worst, abs(b[k] - inverse_ecdf_quantile(sample, tau)))
    return b, worst


F4_LR = 0.5
F4_STEPS = 200
# Each configuration is checked against what it ACTUALLY does, measured.
#
# THE SMALL-LEARNING-RATE CONFIGURATION GENUINELY DOES NOT CONVERGE, AND THAT
# IS THE MEASUREMENT THAT JUSTIFIES THE CHOICE OF lr. The first draft of this
# oracle asserted "every configuration converges within 0.5", and the
# lr = 0.05 case failed at 1.91 -- which is not a defect in the oracle but a
# fact about the dynamics: the intercept has to TRAVEL the gap between the
# sample values, one step of size lr*|S - N*tau|/N at a time, and at a small
# lr it has not arrived when the step budget runs out. Asserting that a
# non-converging configuration converges would be asserting a falsehood; the
# honest statement is that it does NOT converge, and that is why the shipped
# configuration uses a larger learning rate.
for lr, steps, converges in ((0.05, 400, False),
                             (0.1, 400, True),
                             (0.2, 2000, True),
                             (0.5, 200, True),
                             (0.5, 2000, True)):
    b, worst = simulate_intercept_training(grid_sample, F4_TAUS_LIST, lr, steps)
    print("   lr=%.3f steps=%d: the worst TAIL deviation is %.6g (%s)"
          % (lr, steps, worst, "converged" if worst < 0.5 else "NOT converged"))
    if converges:
        check(worst < 0.5,
              "lr=%.3f steps=%d: the simulated training settles within 0.5 of "
              "the closed-form quantile (worst %.6g)" % (lr, steps, worst))
    else:
        check(worst > 0.5,
              "lr=%.3f steps=%d: the simulated training does NOT settle within "
              "0.5 (worst %.6g) -- this is why the shipped configuration uses a "
              "larger learning rate" % (lr, steps, worst))

# The configuration the C++ test will use, and the tolerance it implies.
F4_SIM, F4_TRAJ_WORST = simulate_intercept_training(
    grid_sample, F4_TAUS_LIST, F4_LR, F4_STEPS)
# The tolerance covers the whole trajectory's deviation, plus a factor for the
# C++'s own float arithmetic. It is DERIVED, not guessed.
F4_TOL = float(np.ceil(F4_TRAJ_WORST * 1000.0) / 1000.0 + 0.001)
print("   the derived C++ tolerance is %.17g (lr=%.3f, steps=%d)"
      % (F4_TOL, F4_LR, F4_STEPS))
check(F4_TOL < 0.5, "the derived tolerance is tight enough to be meaningful")
for k, tau in enumerate(F4_TAUS_LIST):
    check(abs(F4_SIM[k] - inverse_ecdf_quantile(grid_sample, tau)) <= F4_TOL,
          "the simulated level %d endpoint is within the derived tolerance" % k)

# =============================================================================
# THE LAYER, IN NUMPY
# =============================================================================

def layer_forward(X, W, b):
    """(N, D) x (K, D)^T + (1, K) -> (N, K)."""
    return X @ W.T + b


def layer_total_loss(pred, y, taus):
    """(1/N) * sum_i sum_k rho_{tau_k}(y_i - pred_ik).  y is (N, 1)."""
    n = pred.shape[0]
    e = y - pred                      # (N, K) by broadcasting
    total = 0.0
    for k, tau in enumerate(taus):
        total += float(np.sum(rho_tau(e[:, k], tau)))
    return total / n


def layer_total_gradient(pred, y, taus):
    """dL/d(pred), shape (N, K)."""
    n = pred.shape[0]
    g = np.zeros_like(pred)
    e = y - pred
    for k, tau in enumerate(taus):
        g[:, k] = rho_tau_grad_pred(e[:, k], tau) / n
    return g


def layer_param_grads(X, pred, y, taus):
    """(dL/dW (K, D), dL/db (1, K), dL/dX (N, D))."""
    g = layer_total_gradient(pred, y, taus)     # (N, K)
    gW = g.T @ X                                 # (K, D)
    gb = g.sum(axis=0, keepdims=True)            # (1, K)
    gX = g @ pred.shape[1] * 0.0                 # placeholder, replaced below
    return gW, gb, g


def layer_param_grads_full(X, W, b, y, taus):
    g = layer_total_gradient(layer_forward(X, W, b), y, taus)   # (N, K)
    gW = g.T @ X                                                 # (K, D)
    gb = g.sum(axis=0, keepdims=True)                            # (1, K)
    gX = g @ W                                                   # (N, D)
    return gW, gb, gX


# =============================================================================
# F. THE FORWARD PASS
# =============================================================================

section("F. the forward pass against the direct formula")

F2_X = np.array([[0.5, -1.25],
                 [-2.0, 0.75],
                 [1.5, 2.0],
                 [0.0, -0.5],
                 [3.25, 1.0]])
F2_W = np.array([[0.7, -0.4],
                 [-1.1, 0.9],
                 [0.25, 0.6]])
F2_B = np.array([[0.1, -0.25, 0.4]])
F2_TAUS = [0.1, 0.5, 0.9]
F2_Y = np.array([[1.0], [-0.5], [2.25], [0.0], [1.75]])

F2_PRED = layer_forward(F2_X, F2_W, F2_B)
for i in range(F2_X.shape[0]):
    for k in range(len(F2_TAUS)):
        direct = float(np.dot(F2_X[i], F2_W[k]) + F2_B[0, k])
        close(direct, F2_PRED[i, k], 1e-14,
              "forward[%d][%d] matches x . w_k + b_k" % (i, k))

# =============================================================================
# G. THE ANALYTIC GRADIENTS AGAINST FINITE DIFFERENCES OF THE TOTAL LOSS
# =============================================================================

section("G. the analytic parameter gradients against finite differences")
gW, gb, gX = layer_param_grads_full(F2_X, F2_W, F2_B, F2_Y, F2_TAUS)

eps = 1e-6
worst = 0.0
for k in range(F2_W.shape[0]):
    for d in range(F2_W.shape[1]):
        Wp = F2_W.copy()
        Wp[k, d] += eps
        Wmm = F2_W.copy()
        Wmm[k, d] -= eps
        fd = (layer_total_loss(layer_forward(F2_X, Wp, F2_B), F2_Y, F2_TAUS)
              - layer_total_loss(layer_forward(F2_X, Wmm, F2_B), F2_Y, F2_TAUS)) / (2.0 * eps)
        worst = max(worst, abs(fd - gW[k, d]))
        close(fd, gW[k, d], 1e-7,
              "dL/dW[%d][%d] matches the finite difference" % (k, d))
for k in range(F2_B.shape[1]):
    bp = F2_B.copy()
    bp[0, k] += eps
    bm = F2_B.copy()
    bm[0, k] -= eps
    fd = (layer_total_loss(layer_forward(F2_X, F2_W, bp), F2_Y, F2_TAUS)
          - layer_total_loss(layer_forward(F2_X, F2_W, bm), F2_Y, F2_TAUS)) / (2.0 * eps)
    close(fd, gb[0, k], 1e-7, "dL/db[%d] matches the finite difference" % k)
for i in range(F2_X.shape[0]):
    for d in range(F2_X.shape[1]):
        Xp = F2_X.copy()
        Xp[i, d] += eps
        Xmm = F2_X.copy()
        Xmm[i, d] -= eps
        fd = (layer_total_loss(layer_forward(Xp, F2_W, F2_B), F2_Y, F2_TAUS)
              - layer_total_loss(layer_forward(Xmm, F2_W, F2_B), F2_Y, F2_TAUS)) / (2.0 * eps)
        close(fd, gX[i, d], 1e-7, "dL/dX[%d][%d] matches the finite difference" % (i, d))

check(worst < 1e-7,
      "the worst parameter-gradient disagreement is %.3g (the pinball objective is "
      "piecewise linear, so this check is tight)" % worst)

# The FD check is only valid if no kink was crossed. Assert that the sign of
# every residual is unchanged by the perturbation.
for k, tau in enumerate(F2_TAUS):
    e = F2_Y[:, 0] - F2_PRED[:, k]
    check(np.all(np.abs(e) > 1e-3),
          "level %d: every residual is far from the kink, so the FD check is valid"
          % k)

# =============================================================================
# H. LEVEL DECOUPLING
# =============================================================================

section("H. level decoupling: the total loss separates into the per-level losses")
g_total = layer_total_gradient(F2_PRED, F2_Y, F2_TAUS)
for k, tau in enumerate(F2_TAUS):
    single = F2_PRED[:, k]
    g_single = rho_tau_grad_pred(F2_Y[:, 0] - single, tau) / F2_X.shape[0]
    close(float(np.sum(np.abs(g_single - g_total[:, k]))), 0.0, 0.0,
          "level %d: the total-loss gradient COLUMN equals the single-level one"
          % k)
    # And the per-level parameter gradients must match the single-level ones.
    # gW_single[k][d] = sum_i g_single[i] * X[i][d], i.e. an OUTER PRODUCT
    # summed over the ROWS. The first draft wrote np.outer(g_single, Xs),
    # which is an (N, N) matrix, not the (D,) vector this needs -- a shape
    # error, caught by numpy's own broadcast check.
    gW_single = np.einsum("i,id->d", g_single, F2_X)
    close(float(np.max(np.abs(gW_single - gW[k]))), 0.0, 1e-15,
          "level %d: the per-level dL/dW equals the single-level dL/dW" % k)

# =============================================================================
# I. CROSSING
# =============================================================================

def crossing_count(pred):
    """The number of ADJACENT inverted pairs, summed over the rows."""
    n, k = pred.shape
    total = 0
    for i in range(n):
        for j in range(k - 1):
            if pred[i, j] > pred[i, j + 1]:
                total += 1
    return total


def monotone(pred):
    """Each row sorted ascending."""
    out = np.array(pred, dtype=np.float64, copy=True)
    for i in range(out.shape[0]):
        out[i, :] = np.sort(out[i, :])
    return out


section("I. the crossing diagnostic and the monotone rearrangement")

F3_PRED = np.array([[3.0, 1.0, 2.0],
                     [0.0, 5.0, 4.0],
                     [1.0, 2.0, 3.0]])
check(crossing_count(F3_PRED) == 2,
      "the constructed fixture has exactly 2 adjacent inversions (got %d)"
      % crossing_count(F3_PRED))
check(crossing_count(monotone(F3_PRED)) == 0,
      "the rearrangement has no crossings")
check(np.array_equal(monotone(F3_PRED),
                     np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [1.0, 2.0, 3.0]])),
      "the rearrangement sorts each row ascending")

# Random sweeps: the rearrangement is idempotent, is a permutation of the row,
# and always produces zero crossings.
for _ in range(500):
    p = rng.normal(size=(4, 5))
    m = monotone(p)
    check(crossing_count(m) == 0, "random sweep: the rearrangement has no crossings")
    for i in range(p.shape[0]):
        check(np.array_equal(np.sort(p[i, :]), m[i, :]),
              "random sweep: the rearrangement preserves the row's multiset")
    check(np.array_equal(monotone(m), m), "random sweep: the rearrangement is idempotent")

# ---- A HUNTED CROSSING FIXTURE --------------------------------------------
# The crossing is not a hypothetical: fit the levels INDEPENDENTLY on a small
# dataset (exactly as the layer does -- one pinball objective per level) and
# look for one whose fitted quantile LINES cross.
#
# The fit uses scipy's Nelder-Mead on the pinball objective. The pinball loss
# is convex and piecewise linear, so the minimiser is well defined; the
# witness the C++ test needs is the FITTED COEFFICIENTS, which the test then
# sets directly. The C++ does NOT have to reproduce an optimiser -- it pins
# the coefficients the oracle found and checks the crossing arithmetic.
def fit_level(X, y, tau):
    """Minimise (1/n) * sum_i rho_tau(y_i - (w.x_i + b)) over (w, b)."""
    n = X.shape[0]

    def obj(p):
        r = y - (X @ p[:X.shape[1]] + p[X.shape[1]])
        return float(np.sum(rho_tau(r, tau)) / n)

    res = optimize.minimize(obj, np.zeros(X.shape[1] + 1), method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-14,
                                     "maxiter": 20000, "maxfev": 20000})
    return res.x


hunt_rng = np.random.default_rng(7)
hunted = None
for trial in range(600):
    n = 8
    x = hunt_rng.normal(size=(n, 1))
    # A response whose conditional spread SHRINKS with x, which is the classic
    # setting where the fitted quantile slopes disagree in sign.
    y = x[:, 0] + hunt_rng.normal(size=n) * np.exp(-x[:, 0])
    taus = (0.2, 0.8)
    lines = []
    for tau in taus:
        p = fit_level(x, y, tau)
        lines.append((float(p[0]), float(p[1])))
    (w0, b0), (w1, b1) = lines
    den = (w0 - w1)
    if abs(den) < 1e-9:
        continue
    xc = (b1 - b0) / den
    if -3.0 <= xc <= 3.0:
        # And the crossing must be a REAL inversion: the low level above the
        # high level at some x.
        if (w0 * xc + b0) > (w1 * xc + b1):
            hunted = (x.ravel().copy(), y.copy(), lines, xc)
            break

if hunted is None:
    check(False, "the crossing hunt found NO fixture in 600 trials")
else:
    hx, hy, hlines, hxc = hunted
    check(True, "the crossing hunt found a fixture that crosses at x = %.6f" % hxc)
    # Pin the fixture for the C++ test: the fitted slopes and intercepts.
    print("   HUNTED x      = " + ", ".join("%.17g" % v for v in hx))
    print("   HUNTED y      = " + ", ".join("%.17g" % v for v in hy))
    for i, (w, b) in enumerate(hlines):
        print("   HUNTED level%d w = %.17g  b = %.17g" % (i, w, b))
    # And assert the crossing is real in the fitted functions.
    (w0, b0), (w1, b1) = hlines
    check(w0 * hxc + b0 > w1 * hxc + b1,
          "the hunted fixture really does have q_low > q_high at the crossing")
    # And that the fit is a genuine minimiser of the pinball objective: a
    # perturbation in EITHER direction must not improve it.
    for i, tau in enumerate((0.2, 0.8)):
        p = np.array([hlines[i][0], hlines[i][1]])
        base = float(np.sum(rho_tau(hy - (p[0] * hx + p[1]), tau)) / hx.size)
        for dim in (0, 1):
            for delta in (1e-3, -1e-3):
                q = p.copy()
                q[dim] += delta
                val = float(np.sum(rho_tau(hy - (q[0] * hx + q[1]), tau)) / hx.size)
                check(val >= base - 1e-12,
                      "hunted level %d: perturbing the fit in dim %d by %g does not "
                      "improve the objective" % (i, dim, delta))

# A second, INDEPENDENT crossing witness: two explicitly-crossing linear
# predictors, which is the shape the rearrangement has to repair.
section("I(b). an explicit crossing witness")
cw_x = np.array([[-1.0], [0.0], [1.0]])
cw_lo = (2.0, 0.0)     # q_0.2(x) = 2x
cw_hi = (-2.0, 0.0)    # q_0.8(x) = -2x
cw_pred = np.array([[cw_lo[0] * -1.0 + cw_lo[1], cw_hi[0] * -1.0 + cw_hi[1]],
                     [cw_lo[0] * 0.0 + cw_lo[1], cw_hi[0] * 0.0 + cw_hi[1]],
                     [cw_lo[0] * 1.0 + cw_lo[1], cw_hi[0] * 1.0 + cw_hi[1]]])
check(crossing_count(cw_pred) == 1,
      "the explicit witness has exactly 1 adjacent inversion (got %d)"
      % crossing_count(cw_pred))
check(crossing_count(monotone(cw_pred)) == 0,
      "the explicit witness is repaired by the rearrangement")

# =============================================================================
# J. THE C++ FIXTURE CONSTANTS
# =============================================================================

section("J. the C++ fixture constants")


def fmt(a):
    flat = np.asarray(a, dtype=np.float64).ravel()
    return ", ".join("%.17g" % v for v in flat)


# --- F1: the K = 1 fixture, for the bit-for-bit equality with QuantileLoss ---
F1_X = np.array([[0.5], [1.5], [-0.5], [2.0]])
F1_W = np.array([[0.7]])
F1_B = np.array([[0.1]])
F1_TAUS = [0.3]
F1_Y = np.array([[1.0], [2.0], [0.2], [1.5]])
F1_PRED = layer_forward(F1_X, F1_W, F1_B)
F1_LOSS = layer_total_loss(F1_PRED, F1_Y, F1_TAUS)
F1_G = layer_total_gradient(F1_PRED, F1_Y, F1_TAUS)
F1_GW, F1_GB, F1_GX = layer_param_grads_full(F1_X, F1_W, F1_B, F1_Y, F1_TAUS)

# The K = 1 layer MUST equal the shipped QuantileLoss, which computes
#   total = sum_{b,k} rho_q(y_bk - pred_bk) ; out = total / N
shipped_loss = float(np.sum(rho_tau(F1_Y - F1_PRED, 0.3)) / F1_X.shape[0])
close(F1_LOSS, shipped_loss, 0.0,
      "the K=1 layer's loss equals the shipped QuantileLoss formula EXACTLY")
shipped_grad = rho_tau_grad_pred(F1_Y - F1_PRED, 0.3) / F1_X.shape[0]
check(np.array_equal(F1_G, shipped_grad),
      "the K=1 layer's gradient equals the shipped QuantileLoss formula EXACTLY")

print("// BEGIN ORACLE-GENERATED")
print("// F1 -- the K = 1 fixture (the bit-for-bit QuantileLoss equality).")
print("static const size_t ORC_F1_N = 4;")
print("static const size_t ORC_F1_D = 1;")
print("static const size_t ORC_F1_K = 1;")
print("static const double ORC_F1_TAU[1] = {%.17g};" % F1_TAUS[0])
print("static const double ORC_F1_X[4] = {%s};" % fmt(F1_X))
print("static const double ORC_F1_W[1] = {%s};" % fmt(F1_W))
print("static const double ORC_F1_B[1] = {%s};" % fmt(F1_B))
print("static const double ORC_F1_Y[4] = {%s};" % fmt(F1_Y))
print("static const double ORC_F1_PRED[4] = {%s};" % fmt(F1_PRED))
print("static const double ORC_F1_LOSS = %.17g;" % F1_LOSS)
print("static const double ORC_F1_GRAD[4] = {%s};" % fmt(F1_G))
print("static const double ORC_F1_GRADW[1] = {%s};" % fmt(F1_GW))
print("static const double ORC_F1_GRADB[1] = {%s};" % fmt(F1_GB))
print("static const double ORC_F1_GRADX[4] = {%s};" % fmt(F1_GX))

print("// F2 -- the multi-level fixture.")
print("static const size_t ORC_F2_N = 5;")
print("static const size_t ORC_F2_D = 2;")
print("static const size_t ORC_F2_K = 3;")
print("static const double ORC_F2_TAU[3] = {%s};" % fmt(F2_TAUS))
print("static const double ORC_F2_X[10] = {%s};" % fmt(F2_X))
print("static const double ORC_F2_W[6] = {%s};" % fmt(F2_W))
print("static const double ORC_F2_B[3] = {%s};" % fmt(F2_B))
print("static const double ORC_F2_Y[5] = {%s};" % fmt(F2_Y))
print("static const double ORC_F2_PRED[15] = {%s};" % fmt(F2_PRED))
print("static const double ORC_F2_LOSS = %.17g;" % layer_total_loss(F2_PRED, F2_Y, F2_TAUS))
print("static const double ORC_F2_GRAD[15] = {%s};" % fmt(g_total))
print("static const double ORC_F2_GRADW[6] = {%s};" % fmt(gW))
print("static const double ORC_F2_GRADB[3] = {%s};" % fmt(gb))
print("static const double ORC_F2_GRADX[10] = {%s};" % fmt(gX))

print("// F3 -- the constructed crossing fixture.")
print("static const double ORC_F3_PRED[9] = {%s};" % fmt(F3_PRED))

print("// F4 -- the intercept-only training fixture (N*tau is never an integer).")
print("static const size_t ORC_F4_N = 10;")
print("static const size_t ORC_F4_K = 4;")
print("static const double ORC_F4_TAU[4] = {%s};" % fmt([0.05, 0.25, 0.75, 0.95]))
print("static const double ORC_F4_Y[10] = {%s};" % fmt(grid_sample))
F4_TARGET = np.array([inverse_ecdf_quantile(grid_sample, t)
                       for t in (0.05, 0.25, 0.75, 0.95)])
print("static const double ORC_F4_TARGET[4] = {%s};" % fmt(F4_TARGET))
print("static const double ORC_F4_SIM[4] = {%s};" % fmt(F4_SIM))
print("static const double ORC_F4_TOL = %.17g;" % F4_TOL)
print("static const double ORC_F4_LR = %.17g;" % F4_LR)
print("static const int ORC_F4_STEPS = %d;" % F4_STEPS)
# And the objective value at the target, which the trained intercept must beat.
F4_LOSS_AT_TARGET = np.array(
    [pinball_objective_intercept(grid_sample, t, F4_TARGET[i])
     for i, t in enumerate((0.05, 0.25, 0.75, 0.95))])
print("static const double ORC_F4_LOSS_AT_TARGET[4] = {%s};" % fmt(F4_LOSS_AT_TARGET))

if hunted is not None:
    hx, hy, hlines, hxc = hunted
    print("// F5 -- the HUNTED crossing fixture (independently fitted levels).")
    print("static const size_t ORC_F5_N = 8;")
    print("static const double ORC_F5_X[8] = {%s};" % fmt(hx))
    print("static const double ORC_F5_Y[8] = {%s};" % fmt(hy))
    print("static const double ORC_F5_W[2] = {%s};" % fmt([l[0] for l in hlines]))
    print("static const double ORC_F5_B[2] = {%s};" % fmt([l[1] for l in hlines]))
    print("static const double ORC_F5_XC = %.17g;" % hxc)
    # The predictions of the two levels on the fixture's x values.
    p_lo = np.array([hlines[0][0] * v + hlines[0][1] for v in hx])
    p_hi = np.array([hlines[1][0] * v + hlines[1][1] for v in hx])
    print("static const double ORC_F5_PRED_LO[8] = {%s};" % fmt(p_lo))
    print("static const double ORC_F5_PRED_HI[8] = {%s};" % fmt(p_hi))

print("// END ORACLE-GENERATED")

# =============================================================================
# SUMMARY
# =============================================================================

print("")
print("=== Summary: %d passed, %d failed ===" % (PASS, FAIL))
if FAIL:
    print("Failed checks:")
    for m in FAILURES:
        print("  - " + m)
    sys.exit(1)
sys.exit(0)
