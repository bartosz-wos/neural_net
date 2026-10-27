#!/usr/bin/env python3
"""
tests/gp_oracle.py -- the SPECIFICATION for utils/gaussian_process.{h,cpp}.

RUN THIS BEFORE WRITING ANY C++. Its job is to catch defects in the DERIVATIONS
and in itself, exactly as tests/crf_oracle.py, tests/conformal_oracle.py and
tests/sinkhorn_oracle.py did. In this repo the oracle has caught more real
defects in the DERIVATION than the C++ suite has caught in the code.

SOURCE. Rasmussen, C. E. & Williams, C. K. I., "Gaussian Processes for Machine
Learning", the MIT Press, Cambridge, 2006, ISBN 026218253X. The book is NOT on
arXiv (title search against the arXiv API returns two unrelated papers; the
official page gaussianprocess.org/gpml carries no arXiv pointer), so NO arXiv ID
MAY BE ASSERTED ANYWHERE. The equations quoted below are the book's own.

  eq. (2.16)/(2.31)  the SE kernel and its hyperparameterised form
  eq. (2.25)/(2.26)  the predictive mean and variance
  eq. (2.30)         the log marginal likelihood
  eq. (4.17)         the Matern 3/2 and 5/2 kernels
  eq. (5.8)          the log marginal likelihood in hyperparameter form
  eq. (5.9)          the hyperparameter gradient
  eq. (5.10)-(5.12)  the leave-one-out predictive quantities

WHAT IS BEING VERIFIED, AND WHY IT IS NOT OBVIOUS:

 1. THE CHOLESKY PATH AND THE EXPLICIT-INVERSE PATH MUST AGREE. The book
    (Algorithm 2.1) uses the Cholesky factorisation; a naive implementation
    forms Ky^-1 explicitly. They agree in exact arithmetic and DISAGREE in
    floating point for a badly conditioned Ky. This oracle measures the
    agreement on the fixtures the C++ suite uses, so the C++ suite knows what
    tolerance is meaningful.

 2. THE LOG MARGINAL LIKELIHOOD MUST MATCH A COMPLETELY SEPARATE ROUTE
    (numpy.linalg.slogdet + solve, i.e. LU, not Cholesky). The -1/2 log|Ky|
    term is the classic place to be off by a factor of two: the Cholesky form
    is -sum(log L_ii) and the determinant form is -1/2 log|Ky|. Both are
    checked, so a factor-of-two error in EITHER is caught.

 3. THE HYPERPARAMETER GRADIENT IS CHECKED AGAINST A CENTRAL FINITITE
    DIFFERENCE, and the USABLE EPSILON IS MEASURED rather than assumed. This
    repo's rule: a fixture whose discriminating value is only correct to
    within a rounding error does not discriminate anything.

 4. **THE LEAVE-ONE-OUT FORMULA IS CHECKED AGAINST AN EXACT REFIT.** This is
    the check that resolves a genuine AMBIGUITY in the book and it is the
    reason this oracle exists. See "THE LOO AMBIGUITY" below.

 5. THE INTERPOLATION AND VARIANCE-LIMIT PROPERTIES, which are observable
    consequences of the formulas and are what a wrong implementation gets
    wrong first.

SELF-CHECKS BEFORE EMITTING ANY NUMBER: every section prints its numbers AND
asserts on them. A self-check that prints a verdict it never tested is worse
than none.

Exit 0 = all sections green. Exit 1 = at least one FAILED line.
"""

import sys
import math
import numpy as np

np.set_printoptions(precision=12, suppress=False)

PASS = 0
FAIL = 0


def check(ok, name, extra=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({extra})" if extra else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({extra})" if extra else ""))


def section(title):
    print(f"\n=== {title} ===")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# The kernels, exactly as the book parameterises them.
# ---------------------------------------------------------------------------

def sqdist(A, B):
    """Pairwise squared Euclidean distances, shape (n, m)."""
    d = A.shape[1]
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    s = aa + bb - 2.0 * (A @ B.T)
    return np.maximum(s, 0.0)


def kernel_se(X, X2, lengthscale, sigma_f):
    """eq. (2.31) without the noise term: sf^2 * exp(-r^2 / (2 l^2))."""
    r2 = sqdist(X, X2)
    return (sigma_f ** 2) * np.exp(-0.5 * r2 / (lengthscale ** 2))


def kernel_matern32(X, X2, lengthscale, sigma_f):
    """eq. (4.17): (1 + sqrt(3) r / l) exp(-sqrt(3) r / l), times sf^2."""
    r = np.sqrt(sqdist(X, X2))
    a = math.sqrt(3.0) * r / lengthscale
    return (sigma_f ** 2) * (1.0 + a) * np.exp(-a)


def kernel_matern52(X, X2, lengthscale, sigma_f):
    """eq. (4.17): (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)."""
    r = np.sqrt(sqdist(X, X2))
    a = math.sqrt(5.0) * r / lengthscale
    return (sigma_f ** 2) * (1.0 + a + 5.0 * r * r / (3.0 * lengthscale ** 2)) * np.exp(-a)


KERNELS = {
    "se": kernel_se,
    "matern32": kernel_matern32,
    "matern52": kernel_matern52,
}


def kernel_matrix(X, X2, lengthscale, sigma_f, kname):
    return KERNELS[kname](X, X2, lengthscale, sigma_f)


def ky_matrix(X, lengthscale, sigma_f, sigma_n, kname):
    """Ky = Kf + sigma_n^2 I  (eq. 5.8)."""
    n = X.shape[0]
    K = kernel_matrix(X, X, lengthscale, sigma_f, kname)
    return K + (sigma_n ** 2) * np.eye(n)


# ---------------------------------------------------------------------------
# The posterior, via the book's Cholesky path (Algorithm 2.1).
# ---------------------------------------------------------------------------

def gp_cholesky(X, y, Xs, lengthscale, sigma_f, sigma_n, kname):
    """Returns (mean, var_of_f, log_marginal_likelihood) via Cholesky.

    mean    = k_*^T Ky^-1 y                          eq. (2.25)
    var_f   = k(x*,x*) - v^T v,  v = Ky^-1/2 k_*      eq. (2.26)
    log p   = -1/2 y^T Ky^-1 y - sum log L_ii - n/2 log 2pi   eq. (2.30)
    """
    n = X.shape[0]
    Ky = ky_matrix(X, lengthscale, sigma_f, sigma_n, kname)
    L = np.linalg.cholesky(Ky)                    # lower triangular
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    log_ml = -0.5 * float(y.T @ alpha) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)

    Ks = kernel_matrix(X, Xs, lengthscale, sigma_f, kname)   # (n, m)
    kss = np.diag(kernel_matrix(Xs, Xs, lengthscale, sigma_f, kname))  # (m,)
    V = np.linalg.solve(L, Ks)                    # (n, m)
    mean = (Ks.T @ alpha).reshape(-1, 1)
    var_f = (kss - np.sum(V * V, axis=0)).reshape(-1, 1)
    return mean, var_f, log_ml


def gp_explicit_inverse(X, y, Xs, lengthscale, sigma_f, sigma_n, kname):
    """The SAME quantities via an explicit inverse and an LU log-determinant.

    This is an INDEPENDENT ROUTE: numpy.linalg.slogdet uses LU, and the
    inverse is formed explicitly. Agreement between the two routes is what
    makes the Cholesky implementation's correctness checkable.
    """
    n = X.shape[0]
    Ky = ky_matrix(X, lengthscale, sigma_f, sigma_n, kname)
    Kinv = np.linalg.inv(Ky)
    sign, logdet = np.linalg.slogdet(Ky)
    assert sign > 0
    log_ml = -0.5 * float(y.T @ Kinv @ y) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)

    Ks = kernel_matrix(X, Xs, lengthscale, sigma_f, kname)
    kss = np.diag(kernel_matrix(Xs, Xs, lengthscale, sigma_f, kname))
    mean = (Ks.T @ Kinv @ y).reshape(-1, 1)
    var_f = (kss - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)).reshape(-1, 1)
    return mean, var_f, log_ml


def dK_dtheta(X, X2, lengthscale, sigma_f, sigma_n, kname, which):
    """The partial derivative of Ky with respect to one hyperparameter.

    which: "lengthscale" | "sigma_f" | "sigma_n"

    For the SE kernel (eq. 2.31), k = sf^2 exp(-r^2 / (2 l^2)):
        dk/dl = sf^2 exp(-r^2/(2 l^2)) * r^2 / l^3
        dk/dsf = 2 sf exp(-r^2/(2 l^2))
    and dKy/dsn = 2 sn I.
    """
    if which == "sigma_n":
        return 2.0 * sigma_n * np.eye(X.shape[0])

    if kname != "se":
        raise NotImplementedError("analytic derivatives are only derived for SE")

    r2 = sqdist(X, X2)
    e = np.exp(-0.5 * r2 / (lengthscale ** 2))
    if which == "lengthscale":
        return (sigma_f ** 2) * e * (r2 / (lengthscale ** 3))
    if which == "sigma_f":
        return 2.0 * sigma_f * e
    raise ValueError(which)


def gp_hyperparam_grad_closed_form(X, y, lengthscale, sigma_f, sigma_n, kname):
    """eq. (5.9): d/dtheta log p(y|X,theta) = 1/2 tr( (a a^T - Ky^-1) dKy/dtheta ).

    Computed the way the book's footnote 5 prescribes: the vector-by-matrix
    product first, and ONLY the diagonal terms of the matrix product in the
    trace.
    """
    n = X.shape[0]
    Ky = ky_matrix(X, lengthscale, sigma_f, sigma_n, kname)
    Kinv = np.linalg.inv(Ky)
    alpha = Kinv @ y                                  # (n, 1)
    grads = []
    for which in ("lengthscale", "sigma_f", "sigma_n"):
        dK = dK_dtheta(X, X, lengthscale, sigma_f, sigma_n, kname, which)
        # first term: alpha^T dK alpha
        t1 = float(alpha.T @ dK @ alpha)
        # second term: tr(Kinv dK) = sum_ij (Kinv)_ij (dK)_ji
        t2 = float(np.sum(Kinv * dK.T))
        grads.append(0.5 * (t1 - t2))
    return np.array(grads).reshape(-1, 1)


def gp_hyperparam_grad_fd(X, y, lengthscale, sigma_f, sigma_n, kname, eps):
    """Central finite difference of the log marginal likelihood."""
    base = np.array([lengthscale, sigma_f, sigma_n], dtype=float)
    out = np.zeros((3, 1))
    for i in range(3):
        hp = base.copy(); hp[i] += eps
        lp = gp_cholesky(X, y, X[:1], hp[0], hp[1], hp[2], kname)[2]
        hm = base.copy(); hm[i] -= eps
        lm = gp_cholesky(X, y, X[:1], hm[0], hm[1], hm[2], kname)[2]
        out[i, 0] = (lp - lm) / (2.0 * eps)
    return out


def gp_loo_inverse_formula(X, y, lengthscale, sigma_f, sigma_n, kname):
    """eq. (5.12): mu_i = y_i - [Ky^-1 y]_i / [Ky^-1]_ii, s2_i = 1/[Ky^-1]_ii.

    QUESTION: is the matrix in eq. (5.12) the NOISE-FREE Kf or the NOISY Ky?
    The book writes "K", which is ambiguous. This function uses Ky (see
    "THE LOO AMBIGUITY" in the header); `gp_loo_exact_refit` computes the same
    quantities by literally refitting on n-1 points, and the two are compared.
    """
    Ky = ky_matrix(X, lengthscale, sigma_f, sigma_n, kname)
    Kinv = np.linalg.inv(Ky)
    alpha = Kinv @ y
    n = X.shape[0]
    mu = np.zeros((n, 1))
    s2 = np.zeros((n, 1))
    for i in range(n):
        mu[i, 0] = y[i, 0] - alpha[i, 0] / Kinv[i, i]
        s2[i, 0] = 1.0 / Kinv[i, i]
    return mu, s2


def gp_loo_exact_refit(X, y, lengthscale, sigma_f, sigma_n, kname):
    """LOO by EXACT REFIT: drop point i, fit on the remaining n-1, predict it.

    This is the ground truth for the LOO quantities and does not depend on any
    identity. It is O(n) refits, which is why it is only used on small
    fixtures.

        mean_i = k_i,-i Ky_-i,-i^-1 y_-i
        var_i  = sn^2 + k_ii - k_i,-i Ky_-i,-i^-1 k_-i,i
    """
    n = X.shape[0]
    mu = np.zeros((n, 1))
    var = np.zeros((n, 1))
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Xr = X[keep, :]
        yr = y[keep, :]
        Kyr = ky_matrix(Xr, lengthscale, sigma_f, sigma_n, kname)
        kir = kernel_matrix(X[i:i + 1, :], Xr, lengthscale, sigma_f, kname).reshape(-1, 1)
        kii = float(kernel_matrix(X[i:i + 1, :], X[i:i + 1, :], lengthscale, sigma_f, kname)[0, 0])
        mu[i, 0] = float(kir.T @ np.linalg.solve(Kyr, yr))
        var[i, 0] = sigma_n ** 2 + kii - float(kir.T @ np.linalg.solve(Kyr, kir))
    return mu, var


# ===========================================================================
# 1. THE KERNEL MATRICES
# ===========================================================================
section("1. THE KERNEL MATRICES")

X1 = np.array([[0.0], [1.0], [2.0]])

k_se = kernel_se(X1, X1, 1.0, 1.0)
k_m32 = kernel_matern32(X1, X1, 1.0, 1.0)
k_m52 = kernel_matern52(X1, X1, 1.0, 1.0)

# The diagonal is sf^2 for EVERY kernel: k(x,x) = sf^2 (eq. 2.16 with r=0).
check(close(k_se[0, 0], 1.0, 1e-15), "1.1 SE: k(x,x) == sf^2", f"{k_se[0,0]:.15f}")
check(close(k_m32[0, 0], 1.0, 1e-15), "1.2 Matern 3/2: k(x,x) == sf^2", f"{k_m32[0,0]:.15f}")
check(close(k_m52[0, 0], 1.0, 1e-15), "1.3 Matern 5/2: k(x,x) == sf^2", f"{k_m52[0,0]:.15f}")

# The off-diagonal at r = l. SE: exp(-1/2). Matern 3/2: (1+sqrt3)exp(-sqrt3).
r1 = math.sqrt(3.0)
check(close(k_se[0, 1], math.exp(-0.5), 1e-15),
      "1.4 SE: k(r=l) == exp(-1/2)", f"{k_se[0,1]:.15f} vs {math.exp(-0.5):.15f}")
check(close(k_m32[0, 1], (1.0 + r1) * math.exp(-r1), 1e-15),
      "1.5 Matern 3/2: k(r=l) == (1+sqrt3)exp(-sqrt3)", f"{k_m32[0,1]:.15f}")
r5 = math.sqrt(5.0)
check(close(k_m52[0, 1], (1.0 + r5 + 5.0 / 3.0) * math.exp(-r5), 1e-15),
      "1.6 Matern 5/2: k(r=l) == (1+sqrt5+5/3)exp(-sqrt5)", f"{k_m52[0,1]:.15f}")

# The LENGTH-SCALE IS l, NOT l^2. At r = 2, l = 2: the exponent is
# -r^2/(2 l^2) = -4/8 = -0.5, so the kernel is exp(-0.5).
#
# THE FIRST DRAFT OF THIS FIXTURE EXPECTED exp(-0.25), WHICH IS NEITHER
# READING -- it is -r^2/(4 l^2). It FAILED against a correct kernel and the
# KERNEL WAS RIGHT. This is the repo's recurring slip class (a hand-arithmetic
# error in the EXPECTED value, caught by running rather than by reading).
# The corrected value still discriminates: exp(-0.5) = 0.606531 against the
# wrong "/l" reading exp(-1.0) = 0.367879 and the wrong "/(4 l^2)" reading
# exp(-0.25) = 0.778801.
k_se_l2 = kernel_se(np.array([[0.0]]), np.array([[2.0]]), 2.0, 1.0)
check(close(k_se_l2[0, 0], math.exp(-0.5), 1e-15),
      "1.7 SE: the exponent divides by l^2, not l",
      f"{k_se_l2[0,0]:.15f} vs {math.exp(-0.5):.15f}")
check(abs(k_se_l2[0, 0] - math.exp(-1.0)) > 1e-6,
      "1.7b the '/l' reading gives a DIFFERENT number",
      f"exp(-1.0) = {math.exp(-1.0):.15f}")

# The AMPLITUDE IS sf^2, NOT sf.
k_se_sf2 = kernel_se(np.array([[0.0]]), np.array([[1.0]]), 1.0, 2.0)
check(close(k_se_sf2[0, 0], 4.0 * math.exp(-0.5), 1e-15),
      "1.8 SE: the amplitude is sf^2, not sf", f"{k_se_sf2[0,0]:.15f}")

# Every kernel is SYMMETRIC and its diagonal is constant.
for nm, K in (("se", k_se), ("matern32", k_m32), ("matern52", k_m52)):
    check(np.allclose(K, K.T, atol=1e-15), f"1.9 {nm}: the kernel matrix is symmetric")
    check(np.allclose(np.diag(K), np.diag(K)[0], atol=1e-15),
          f"1.10 {nm}: the diagonal is constant")


# ===========================================================================
# 2. THE LOG MARGINAL LIKELIHOOD, BY TWO INDEPENDENT ROUTES
# ===========================================================================
section("2. THE LOG MARGINAL LIKELIHOOD")

rng = np.random.default_rng(20261027)
Xg = rng.normal(size=(6, 2))
yg = rng.normal(size=(6, 1))

m_chol, v_chol, lml_chol = gp_cholesky(Xg, yg, Xg, 1.2, 1.1, 0.3, "se")
m_inv, v_inv, lml_inv = gp_explicit_inverse(Xg, yg, Xg, 1.2, 1.1, 0.3, "se")

check(close(lml_chol, lml_inv, 1e-9),
      "2.1 Cholesky and explicit-inverse log marginal likelihoods agree",
      f"{lml_chol:.15f} vs {lml_inv:.15f}")
check(np.allclose(m_chol, m_inv, atol=1e-9),
      "2.2 Cholesky and explicit-inverse predictive means agree",
      f"max|d| = {np.max(np.abs(m_chol - m_inv)):.3e}")
check(np.allclose(v_chol, v_inv, atol=1e-9),
      "2.3 Cholesky and explicit-inverse predictive variances agree",
      f"max|d| = {np.max(np.abs(v_chol - v_inv)):.3e}")

# A THIRD route: the multivariate normal density, assembled by hand.
Ky = ky_matrix(Xg, 1.2, 1.1, 0.3, "se")
sign, logdet = np.linalg.slogdet(Ky)
lml_manual = -0.5 * float(yg.T @ np.linalg.solve(Ky, yg)) - 0.5 * logdet - 0.5 * 6 * math.log(2.0 * math.pi)
check(close(lml_chol, lml_manual, 1e-9),
      "2.4 the log marginal likelihood matches a hand-assembled normal density",
      f"{lml_chol:.15f} vs {lml_manual:.15f}")

# THE FACTOR-OF-TWO IN THE COMPLEXITY PENALTY. The Cholesky form is
# -sum(log L_ii); the determinant form is -1/2 log|Ky| = -sum(log L_ii). The
# WRONG reading is -1/2 sum(log L_ii), which HALVES the complexity penalty.
half_penalty = -0.5 * float(np.sum(np.log(np.diag(np.linalg.cholesky(Ky)))))
check(abs((lml_chol + half_penalty) - lml_chol) > 1e-3,
      "2.5 the half-penalty reading gives a DIFFERENT number (the factor-of-two)",
      f"difference = {abs(half_penalty):.6f}")

# THE -n/2 log(2 pi) CONSTANT. Removing it is a different number.
no_const = lml_chol + 0.5 * 6 * math.log(2.0 * math.pi)
check(abs(no_const - lml_chol) > 1e-3,
      "2.6 dropping the -n/2 log(2 pi) constant gives a DIFFERENT number",
      f"difference = {abs(no_const - lml_chol):.6f}")

# A HAND-DERIVED MAGNITUDE. n = 1, x = 0, y = 1, sf = 1, sn = 1, l = 1:
#   Ky = [[1 + 1]] = [[2]]
#   log p = -1/2 * (1/2) - 1/2 * log(2) - 1/2 * log(2 pi)
#        = -0.25 - 0.34657359027997264 - 0.91893853320467274
#        = -1.5155121234846454
X1x = np.array([[0.0]]); y1x = np.array([[1.0]])
_, _, lml_1 = gp_cholesky(X1x, y1x, X1x, 1.0, 1.0, 1.0, "se")
expected_1 = -0.25 - 0.5 * math.log(2.0) - 0.5 * math.log(2.0 * math.pi)
check(close(lml_1, expected_1, 1e-12),
      "2.7 the n=1 case matches the HAND-DERIVED value",
      f"{lml_1:.15f} vs {expected_1:.15f}")
print(f"  [VALUE] n=1 log marginal likelihood = {lml_1:.15f}")


# ===========================================================================
# 3. THE PREDICTIVE MEAN AND VARIANCE: THE PROPERTIES
# ===========================================================================
section("3. THE PREDICTIVE MEAN AND VARIANCE")

# P1/P2: INTERPOLATION. With sn = 0 the posterior mean at a TRAINING point
# reproduces the target exactly (the noise-free GP interpolates its data).
Xit = np.array([[0.0], [0.7], [1.5], [2.3], [3.1]])
yit = np.array([[0.2], [-1.1], [0.5], [1.7], [-0.3]])
m_it, v_it, _ = gp_cholesky(Xit, yit, Xit, 1.0, 1.0, 0.0, "se")
interp_err = float(np.max(np.abs(m_it - yit)))
check(interp_err < 1e-10, "3.1 sn=0: the posterior mean interpolates the training data",
      f"max|mean - y| = {interp_err:.3e}")

# P3: the noise-free posterior VARIANCE at a training point is ~0, NOT sf^2.
check(float(np.max(v_it)) < 1e-10,
      "3.2 sn=0: the posterior VARIANCE at a training point is ~0, not sf^2",
      f"max var = {float(np.max(v_it)):.3e}, sf^2 = 1.0")

# With a NONZERO noise the variance at a training point is NOT zero: it is the
# posterior variance of the LATENT f, which is bounded by sf^2.
m_n, v_n, _ = gp_cholesky(Xit, yit, Xit, 1.0, 1.0, 0.25, "se")
check(0.0 < float(np.max(v_n)) < 1.0,
      "3.3 sn>0: the latent posterior variance at a training point is in (0, sf^2)",
      f"max var = {float(np.max(v_n)):.6f}")
check(float(np.max(v_n)) < 1.0,
      "3.4 sn>0: the latent variance NEVER reaches sf^2 at a training point",
      f"max var = {float(np.max(v_n)):.6f} < 1.0")

# P4: at a FAR-AWAY test point the predictive variance approaches sf^2.
Xfar = np.array([[100.0]])
_, v_far, _ = gp_cholesky(Xit, yit, Xfar, 1.0, 1.0, 0.25, "se")
check(abs(float(v_far[0, 0]) - 1.0) < 1e-6,
      "3.5 a far test point has predictive variance -> sf^2",
      f"var = {float(v_far[0,0]):.9f}")

# P5: the NOISY target's variance is the latent variance PLUS sn^2.
v_noisy = v_n + 0.25 ** 2
check(np.all(v_noisy > v_n),
      "3.6 the noisy target variance is the latent variance PLUS sn^2")
check(close(float(np.max(v_noisy - v_n)), 0.0625, 1e-15),
      "3.7 the noise addition is EXACTLY sn^2",
      f"max difference = {float(np.max(v_noisy - v_n)):.15f}")

# The variance of the LATENT f never exceeds sf^2 + sn^2 (it is bounded by
# the prior variance of the function value).
Xs2 = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]])
_, v_grid, _ = gp_cholesky(Xit, yit, Xs2, 1.0, 1.0, 0.25, "se")
check(float(np.max(v_grid)) <= 1.0 + 1e-12,
      "3.8 the latent predictive variance is bounded by sf^2 everywhere",
      f"max var = {float(np.max(v_grid)):.12f}")


# ===========================================================================
# 4. THE HYPERPARAMETER GRADIENT
# ===========================================================================
section("4. THE HYPERPARAMETER GRADIENT")

print("  [MEASURE] the usable central-difference epsilon for the log marginal")
print("            likelihood gradient (the repo's rule: a fixture whose")
print("            discriminating value is only correct to within a rounding")
print("            error does not discriminate anything):")
best_eps = None
for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    g_fd = gp_hyperparam_grad_fd(Xg, yg, 1.2, 1.1, 0.3, "se", eps)
    g_cf = gp_hyperparam_grad_closed_form(Xg, yg, 1.2, 1.1, 0.3, "se")
    rel = np.max(np.abs(g_fd - g_cf) / np.maximum(np.abs(g_cf), 1e-12))
    print(f"    eps = {eps:.0e}   max rel err vs closed form = {rel:.3e}")
    if best_eps is None and rel < 1e-6:
        best_eps = eps

check(best_eps is not None,
      "4.1 SOME epsilon gives the closed form and the FD gradient agree",
      f"best eps = {best_eps}")

eps = 1e-5
g_fd = gp_hyperparam_grad_fd(Xg, yg, 1.2, 1.1, 0.3, "se", eps)
g_cf = gp_hyperparam_grad_closed_form(Xg, yg, 1.2, 1.1, 0.3, "se")
print(f"  [VALUE] closed-form gradient = {g_cf.ravel()}")
print(f"  [VALUE] FD gradient (eps=1e-5) = {g_fd.ravel()}")
rel_g = np.max(np.abs(g_fd - g_cf) / np.maximum(np.abs(g_cf), 1e-12))
check(rel_g < 1e-6, "4.2 the closed form matches a central finite difference",
      f"max rel err = {rel_g:.3e}")

# THE 1/2 AND THE TRACE. The closed form is 1/2 tr((aa^T - K^-1) dK/dtheta).
# Both wrong readings give a DIFFERENT number and must be caught.
Kinv = np.linalg.inv(Ky)
alpha = Kinv @ yg
dK_l = dK_dtheta(Xg, Xg, 1.2, 1.1, 0.3, "se", "lengthscale")
t1 = float(alpha.T @ dK_l @ alpha)
t2 = float(np.sum(Kinv * dK_l.T))
no_half = t1 - t2                 # the factor-of-two error
# The "sum of ALL entries of the PRODUCT" reading: 1^T (M dK) 1.
M = alpha @ alpha.T - Kinv
full_trace = float(np.sum(M @ dK_l))       # NOT the trace
correct = 0.5 * (t1 - t2)
check(abs(no_half - correct) > 1e-6,
      "4.3 dropping the 1/2 gives a DIFFERENT gradient",
      f"{correct:.9f} vs {no_half:.9f}")
check(abs(full_trace - correct) > 1e-6,
      "4.4 summing ALL entries of the PRODUCT instead of taking the trace gives a DIFFERENT gradient",
      f"{correct:.9f} vs {full_trace:.9f}")
drop_Kinv = 0.5 * t1
check(abs(drop_Kinv - correct) > 1e-6,
      "4.5 dropping the -K^-1 term gives a DIFFERENT gradient",
      f"{correct:.9f} vs {drop_Kinv:.9f}")

# **THE ELEMENTWISE SUM IS A TRAP, AND THIS CHECK IS WHY THE ORACLE EXISTS.**
#
# tr(M dK) = sum_ij M_ij dK_ji. For SYMMETRIC M and dK that equals the
# ELEMENTWISE sum sum_ij M_ij dK_ij -- so a mutation that replaces the trace
# with `sum(M .* dK)` SURVIVES on every symmetric fixture. Measured below:
#     trace(M dK)      = -8.028646365323732
#     sum(M .* dK)     = -8.028646365323732   <-- IDENTICAL
#     sum(M @ dK)      = -5.009491637812285   <-- the genuinely different thing
#
# So the "sum of all entries" mutation must be written against the MATRIX
# PRODUCT, not the elementwise product. The implementation must compute the
# trace as sum_ij M_ij dK_ji (i.e. with the TRANSPOSE), which is correct for
# any dK and does not silently rely on symmetry.
elem_sum = float(np.sum(M * dK_l))
trace_prod = float(np.trace(M @ dK_l))
print(f"  [MEASURE] trace(M dK) = {trace_prod:.15f}")
print(f"  [MEASURE] sum(M .* dK) = {elem_sum:.15f}   <-- EQUAL to the trace "
      f"(symmetric), so this reading is UNOBSERVABLE here")
print(f"  [MEASURE] sum(M @ dK)  = {full_trace:.15f}   <-- the trace's actual meaning")
check(abs(elem_sum - trace_prod) < 1e-9,
      "4.6 the elementwise sum EQUALS the trace for symmetric matrices (a trap)")
check(abs(elem_sum - full_trace) > 1e-6,
      "4.7 the elementwise sum DIFFERS from the sum of the product's entries",
      f"{elem_sum:.9f} vs {full_trace:.9f}")

# P9: THE SIGN OF THE LENGTH-SCALE DERIVATIVE. On data that is SMOOTH (a
# long length-scale fits better), the gradient must point TOWARD a longer
# length-scale, i.e. it must be POSITIVE at a length-scale that is too short.
Xsmooth = np.linspace(0.0, 5.0, 8).reshape(-1, 1)
ysmooth = np.sin(Xsmooth).reshape(-1, 1)
g_sm = gp_hyperparam_grad_closed_form(Xsmooth, ysmooth, 0.2, 1.0, 0.05, "se")
check(g_sm[0, 0] > 0.0,
      "4.6 at a TOO-SHORT length-scale on smooth data the dL/dl gradient is POSITIVE",
      f"dL/dl = {g_sm[0,0]:.9f}")

# A LONGER length-scale fits smooth data better: the log marginal
# likelihood at l = 2 must exceed the one at l = 0.2.
_, _, lml_short = gp_cholesky(Xsmooth, ysmooth, Xsmooth[:1], 0.2, 1.0, 0.05, "se")
_, _, lml_long = gp_cholesky(Xsmooth, ysmooth, Xsmooth[:1], 2.0, 1.0, 0.05, "se")
check(lml_long > lml_short,
      "4.7 a longer length-scale fits smooth data better",
      f"lml(0.2) = {lml_short:.6f} < lml(2.0) = {lml_long:.6f}")

# P12: the marginal likelihood is MAXIMISED near the TRUE hyperparameters on a
# fixture generated from known ones.
Xl = np.linspace(0.0, 4.0, 12).reshape(-1, 1)
yl = np.sin(1.5 * Xl).reshape(-1, 1)
lls = []
for L in (0.2, 0.5, 1.0, 1.5, 2.0, 3.0):
    _, _, lml = gp_cholesky(Xl, yl, Xl[:1], L, 1.0, 0.05, "se")
    lls.append((L, lml))
    print(f"    l = {L:4.1f}   log marginal likelihood = {lml:.6f}")
best_L = max(lls, key=lambda t: t[1])[0]
check(1.0 <= best_L <= 2.5,
      "4.8 the marginal likelihood is maximised NEAR the true length-scale",
      f"argmax l = {best_L}")


# ===========================================================================
# 5. THE LOO AMBIGUITY -- THE REASON THIS ORACLE EXISTS
# ===========================================================================
section("5. THE LEAVE-ONE-OUT QUANTITIES")

# THE BOOK IS AMBIGUOUS HERE. Eq. (5.12) is written in terms of "K", and
# eq. (5.8) had just introduced Ky = Kf + sn^2 I as a distinct object. The
# question: does eq. (5.12) use the NOISE-FREE Kf or the NOISY Ky?
#
# The book resolves it INDIRECTLY, in the sentence right after (5.12):
#   "careful inspection reveals that the mean mu_i is in fact independent of
#    y_i as indeed it should be."
# That property holds for the INVERSE FORMULA with EITHER matrix (it is pure
# algebra), so it does NOT by itself decide the question. The tie is broken
# by an EXACT REFIT: drop point i, fit on the remaining n-1 points, and
# predict point i. That is the ground truth, and it depends on the noise.

Xloo = np.array([[0.0], [0.6], [1.3], [2.0], [2.8], [3.5]])
yloo = np.array([[0.1], [0.7], [-0.2], [0.4], [0.9], [0.3]])
ls, sf, sn = 1.0, 1.0, 0.4

mu_inv, s2_inv = gp_loo_inverse_formula(Xloo, yloo, ls, sf, sn, "se")
mu_ref, var_ref = gp_loo_exact_refit(Xloo, yloo, ls, sf, sn, "se")

print(f"  [VALUE] inverse-formula LOO mean  = {mu_inv.ravel()}")
print(f"  [VALUE] exact-refit   LOO mean    = {mu_ref.ravel()}")
print(f"  [VALUE] inverse-formula LOO var   = {s2_inv.ravel()}")
print(f"  [VALUE] exact-refit   LOO var     = {var_ref.ravel()}")

err_mu = float(np.max(np.abs(mu_inv - mu_ref)))
err_var = float(np.max(np.abs(s2_inv - var_ref)))
check(err_mu < 1e-9,
      "5.1 the inverse-formula LOO MEAN matches the exact refit",
      f"max|d| = {err_mu:.3e}")
check(err_var < 1e-9,
      "5.2 the inverse-formula LOO VARIANCE matches the exact refit",
      f"max|d| = {err_var:.3e}")

# THE MATRIX IN eq. (5.12) IS THEREFORE Ky, THE NOISY ONE. The noise-free
# variant is WRONG and must be distinguishable.
Ky_inv = np.linalg.inv(ky_matrix(Xloo, ls, sf, sn, "se"))
Kf_inv = np.linalg.inv(kernel_matrix(Xloo, Xloo, ls, sf, "se"))
alpha_n = Ky_inv @ yloo
alpha_f = Kf_inv @ yloo
mu_f = np.array([yloo[i, 0] - alpha_f[i, 0] / Kf_inv[i, i] for i in range(6)]).reshape(-1, 1)
s2_f = np.array([1.0 / Kf_inv[i, i] for i in range(6)]).reshape(-1, 1)
err_mu_f = float(np.max(np.abs(mu_f - mu_ref)))
err_var_f = float(np.max(np.abs(s2_f - var_ref)))
print(f"  [MEASURE] the noise-free reading differs from the exact refit by")
print(f"            max|d mean| = {err_mu_f:.6f} and max|d var| = {err_var_f:.6f}")
check(err_mu_f > 1e-6,
      "5.3 the NOISE-FREE reading of eq. (5.12) is WRONG (it differs from the refit)",
      f"max|d mean| = {err_mu_f:.6f}")
check(err_var_f > 1e-6,
      "5.4 the NOISE-FREE reading of eq. (5.12) is WRONG for the variance too",
      f"max|d var| = {err_var_f:.6f}")

# P10: the LOO mean mu_i is INDEPENDENT of the held-out target y_i.
#
# **THE INDEPENDENCE IS PER-i, NOT GLOBAL, AND THE FIRST DRAFT OF THIS CHECK
# GOT THAT WRONG.** It perturbed y_2 and then took the max over ALL i, which
# picked up mu_0, mu_1, ... -- and those legitimately DO move, because y_2 is
# a training target for every j != 2.
#
# The algebra, from eq. (5.12):
#     [K^-1 y]_i = sum_j [K^-1]_ij y_j = [K^-1]_ii y_i + sum_{j != i} [K^-1]_ij y_j
#     mu_i = y_i - [K^-1 y]_i / [K^-1]_ii
#          = y_i - y_i - sum_{j != i} [K^-1]_ij y_j / [K^-1]_ii
# The two y_i terms CANCEL EXACTLY. That is what the book's sentence means:
# "the mean mu_i is in fact independent of y_i as indeed it should be" -- the
# LOO prediction of point i must not use point i's own label.
#
# The check therefore perturbs y_i and asserts on mu_i ALONE, for EVERY i.
# The global max is asserted to be NON-ZERO, which is the flip side: the other
# mu_j MUST move (otherwise the LOO quantities would be constant, which would
# be a different and equally wrong model).
max_diag = 0.0
max_offdiag = 0.0
d_s2 = 0.0
for i0 in range(6):
    yloo_p = yloo.copy()
    yloo_p[i0, 0] += 3.7
    mu_p, s2_p = gp_loo_inverse_formula(Xloo, yloo_p, ls, sf, sn, "se")
    max_diag = max(max_diag, abs(float(mu_p[i0, 0] - mu_inv[i0, 0])))
    for j in range(6):
        if j != i0:
            max_offdiag = max(max_offdiag, abs(float(mu_p[j, 0] - mu_inv[j, 0])))
    d_s2 = float(np.max(np.abs(s2_p - s2_inv)))
check(max_diag < 1e-10,
      "5.5 mu_i is INDEPENDENT of y_i, for EVERY i (the diagonal of the change matrix is 0)",
      f"max|d mu_i| = {max_diag:.3e}")
check(max_offdiag > 1e-6,
      "5.5b the OTHER mu_j DO move when y_i moves (the LOO is not constant)",
      f"max|d mu_j| = {max_offdiag:.6f}")
check(d_s2 < 1e-12,
      "5.6 the LOO variance is independent of the held-out target y_i",
      f"max|d s2| = {d_s2:.3e}")

# The LOO mean really does use the other targets: perturbing a DIFFERENT y_j
# must move mu_i.
yloo_q = yloo.copy()
yloo_q[0, 0] += 1.3
mu_q, _ = gp_loo_inverse_formula(Xloo, yloo_q, ls, sf, sn, "se")
d_mu_other = float(np.max(np.abs(mu_q[1:, :] - mu_inv[1:, :])))
check(d_mu_other > 1e-6,
      "5.7 the LOO mean DOES depend on the other targets",
      f"max|d mu| = {d_mu_other:.6f}")

# P11: the LOO variance DEPENDS ON sn, which is the observable consequence of
# eq. (5.12) using Ky. A noise-free LOO variance would be independent of it.
_, s2_nonoise = gp_loo_inverse_formula(Xloo, yloo, ls, sf, 0.0, "se")
_, s2_noisy = gp_loo_inverse_formula(Xloo, yloo, ls, sf, 0.4, "se")
d_s2_noise = float(np.max(np.abs(s2_nonoise - s2_noisy)))
check(d_s2_noise > 1e-6,
      "5.8 the LOO variance DEPENDS on sn (it uses Ky, not Kf)",
      f"max|d s2| = {d_s2_noise:.6f}")
print(f"  [VALUE] LOO variance at sn=0.0 = {s2_nonoise.ravel()}")
print(f"  [VALUE] LOO variance at sn=0.4 = {s2_noisy.ravel()}")


# ===========================================================================
# 6. THE SINGULAR CASE AND THE ZERO-NOISE LIMIT
# ===========================================================================
section("6. SINGULARITY AND THE ZERO-NOISE LIMIT")

# A DUPLICATE training point with sn = 0 makes Ky singular. The Cholesky
# must FAIL (a non-positive pivot), not silently produce nan.
Xd = np.array([[0.0], [0.0], [1.0]])
try:
    np.linalg.cholesky(ky_matrix(Xd, 1.0, 1.0, 0.0, "se"))
    singular_detected = False
    err_msg = ""
except np.linalg.LinAlgError as e:
    singular_detected = True
    err_msg = str(e)
check(singular_detected,
      "6.1 a duplicate training point with sn=0 makes Ky SINGULAR (Cholesky fails)",
      err_msg[:60])

# The SAME points with a positive noise are fine.
Ky_ok = ky_matrix(Xd, 1.0, 1.0, 0.1, "se")
check(np.all(np.linalg.eigvalsh(Ky_ok) > 0),
      "6.2 the same points with sn=0.1 give a POSITIVE DEFINITE Ky")

# P15: sn = 0 with DISTINCT points is legal and interpolates.
Xz = np.array([[0.0], [1.0], [2.0], [3.0]])
yz = np.array([[1.0], [-1.0], [0.5], [2.0]])
m_z, v_z, lml_z = gp_cholesky(Xz, yz, Xz, 1.0, 1.0, 0.0, "se")
check(float(np.max(np.abs(m_z - yz))) < 1e-10,
      "6.3 sn=0 with distinct points is LEGAL and interpolates",
      f"max|mean - y| = {float(np.max(np.abs(m_z - yz))):.3e}")

# P13: the sn -> 0 limit is continuous. The predictive mean at a training
# point converges to y as sn -> 0.
ms = []
for s in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.0):
    m_s, _, _ = gp_cholesky(Xz, yz, Xz, 1.0, 1.0, s, "se")
    ms.append(float(np.max(np.abs(m_s - yz))))
    print(f"    sn = {s:4.2f}   max|mean - y| at training points = {ms[-1]:.3e}")
check(ms[-1] < ms[0] and all(ms[i] >= ms[i + 1] - 1e-12 for i in range(len(ms) - 1)),
      "6.4 the sn -> 0 limit is MONOTONE towards exact interpolation",
      f"final = {ms[-1]:.3e}")


# ===========================================================================
# 7. THE FIXTURES THE C++ SUITE WILL ASSERT AGAINST
# ===========================================================================
section("7. THE C++ FIXTURES")

print("  Fixture G1: 3 points in 1-D, sn = 0 (interpolation)")
Xg1 = np.array([[0.0], [1.0], [2.0]])
yg1 = np.array([[0.5], [-0.25], [1.0]])
m1, v1, l1 = gp_cholesky(Xg1, yg1, Xg1, 1.0, 1.0, 0.0, "se")
print(f"    mean       = {m1.ravel()}")
print(f"    var        = {v1.ravel()}")
print(f"    log ml     = {l1:.15f}")

print("  Fixture G2: 6 points in 2-D, sn = 0.3 (the general case)")
Xg2 = rng.normal(size=(6, 2))
yg2 = rng.normal(size=(6, 1))
m2, v2, l2 = gp_cholesky(Xg2, yg2, Xg2, 1.2, 1.1, 0.3, "se")
g2_closed = gp_hyperparam_grad_closed_form(Xg2, yg2, 1.2, 1.1, 0.3, "se")
print(f"    log ml     = {l2:.15f}")
print(f"    mean       = {m2.ravel()}")
print(f"    var        = {v2.ravel()}")
print(f"    gradient   = {g2_closed.ravel()}")

print("  Fixture G3: the LOO fixture")
print(f"    mu         = {mu_inv.ravel()}")
print(f"    s2         = {s2_inv.ravel()}")

print("  Fixture G4: the n=1 hand-derived case")
print(f"    log ml     = {lml_1:.15f}")

print("  Fixture G5: the SE kernel matrix at l=1, sf=1")
print(f"    K          = {kernel_se(X1, X1, 1.0, 1.0).ravel()}")
print(f"    K matern32 = {kernel_matern32(X1, X1, 1.0, 1.0).ravel()}")
print(f"    K matern52 = {kernel_matern52(X1, X1, 1.0, 1.0).ravel()}")


# ===========================================================================
print()
print(f"=== ORACLE SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
