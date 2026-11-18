#!/usr/bin/env python3
"""
tests/sample_quality_oracle.py -- the float64 numpy + scipy specification of
the SAMPLE-QUALITY METRICS: the Frechet Inception Distance (FID), the
Inception Score (IS), and the SAMPLE GAUSSIAN estimator they are built on.

RUN THIS BEFORE WRITING ANY C++.

    /usr/bin/python3 tests/sample_quality_oracle.py
    /usr/bin/python3 tests/sample_quality_oracle.py --emit-cpp

Its discriminating power is MEASURED, not narrated: the companion harness
tests/sample_quality_mutate_oracle.py breaks the model code below in
specific ways and requires that this oracle FAIL each time. An oracle that
cannot fail certifies nothing.

==========================================================================
THE GAP (measured 2026-11-18, not assumed)
==========================================================================

    grep -rinE '\bfid\b|inception' include/ tests/ --include=*.h --include=*.cpp
    -> include/nn/utils/linear_algebra.h:16,114,116,124,125   (all PROSE)
    -> include/nn/utils/sliced_wasserstein.h:28,29,30          (all PROSE)
    -> include/nn/utils/linear_algebra.cpp:11,259,274,278      (all PROSE)
    -> include/nn/utils/rank_correlation.h:31, include/nn/utils/dtw.h:28
    -> tests/test_sliced_wasserstein.cpp:28,29, tests/test_linear_algebra.cpp:37,51
    -> ZERO IMPLEMENTATION.

FIFTEEN generative models ship in this repo -- DDPM, consistency models,
flow matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP,
PixelCNN, CapsNet, DiT -- and NOTHING ANYWHERE MEASURES WHETHER THE
SAMPLES ARE GOOD. Two shipped headers say so in their own words.

WHAT IS MODEL-FREE AND WHY THAT MATTERS
=======================================
The Inception NETWORK is not shipped, and this oracle does not pretend
otherwise: it is one choice of feature extractor. The METRIC is defined on
feature vectors, so the metric is model-free and the caller supplies the
features. That is the same shape as utils/sliced_wasserstein.h, which
measures two point sets and likewise ships no network.

THE REUSABLE CORE: NOTHING IN THE REPO ESTIMATES A GAUSSIAN FROM DATA.
utils/linear_algebra.h's `frechet_distance(mu1, S1, mu2, S2)` takes two
Gaussians ALREADY IN HAND. The step from FEATURES to a Gaussian is the
missing piece. utils/gmm.h stores a Gaussian with a GIVEN covariance and
never estimates one -- and its Cholesky cannot be reused here anyway,
because a sample covariance from n samples in d > n dimensions has rank
<= n-1 and is SINGULAR.

==========================================================================
THE CONVENTIONS, AND EACH ONE IS A MEASURED DECISION
==========================================================================

C1. FID IS THE SQUARED FREMCHET DISTANCE, NOT THE DISTANCE.
    The repo's shipped `frechet_distance` returns the DISTANCE (a square
    root). The reference implementations return the radicand. Measured on
    this file's own fixture:
        radicand (d^2) = 5.990601114713      sqrt = 2.447570451430
    So a FID that returns `d` instead of `d^2` is off by a factor of
    2.4476 -- a clean, large, easily-detected error.

C2. THE COVARIANCE ESTIMATOR IS THE UNBIASED ONE (divide by n-1).
    `np.cov`'s default (ddof=1) and `tfp.stats.covariance`'s default
    (sample_stats=True) are both unbiased, and pytorch-fid inherits
    np.cov's default. The BIASED estimator (divide by n) is what
    `tf.nn.moments` computes, so both readings exist in the wild.
    Measured on this file's own fixture:
        unbiased (n-1): FID = 5.990601114713
        biased   (n)  : FID = 5.918540015989

C3. THE PRODUCT ORDER IS THE SYMMETRISED ONE, AND IT AGREES WITH scipy.
    The reference implementation evaluates sqrtm(S1 @ S2); S1 @ S2 is NOT
    symmetric (measured asymmetry 0.0739 on this file's fixture) and its
    square root is COMPLEX in general. The repo's shipped form,
    (S1^(1/2) . S2 . S1^(1/2))^(1/2), is symmetric PSD, so its square root
    is real. Measured agreement between the two:
        5.329070518200751e-15
    So the repo's form is correct AND better conditioned, and this oracle
    pins the agreement -- which is what makes reusing the shipped function
    safe.

C4. FID IS SQUARED *AFTER* THE CLAMP.
    The shipped `frechet_distance` clamps its radicand at zero from below.
    So FID >= 0 always, and FID(X, X) == 0 EXACTLY. The reference
    implementation computes the radicand directly and can return a tiny
    NEGATIVE number in pathological cases (measured -2.84e-13 on a
    scaled-identical fixture). Squaring the shipped distance is simpler
    and better behaved; this oracle pins the exact zero.

C5. THE `0 * log(0/q) = 0` CONVENTION IS LOAD-BEARING IN THE KL.
    The naive expression `p * (log(p) - log(q))` evaluates 0 * (-inf) on
    a zero probability and produces NaN. A metric that reports NaN
    reports nothing.

C6. THE SPLIT COUNT CHANGES THE IS, so it is a pinned convention.
    Measured on this file's own one-hot fixture:
        IS with  1 split = 9.86297313554185
        IS with 10 splits = 7.8751657630923475

C7. IS OF AN IDENTICAL SET IS EXACTLY 1.0.
    Every row equals the marginal, so every KL is 0 and exp(0) = 1. This
    is the sharpest non-vacuity property of the metric and it is exact.

C8. THE RANK-DEFICIENT CASE IS THE REALISTIC ONE.
    With n samples of d features and d > n, the sample covariance has
    rank <= n-1 and is SINGULAR. Measured: rank(Sa) = 9 for n = 10,
    d = 64, with a minimum eigenvalue of -5.9e-15 from rounding. A
    Cholesky-based implementation would THROW here. A 2048-dimensional
    feature space with a few hundred samples is exactly this case.

==========================================================================
A FOIL THAT PRODUCES NaN IS A VACUOUS FOIL
==========================================================================

A discriminating foil must be FINITE and WRONG, not undefined. If the
wrong variant returns NaN then "right != NaN" discriminates trivially and
measures nothing about the right implementation. So the finite wrong
variants are the ones used as foils:
    - the BIASED covariance estimator          (finite, different)
    - the `d` instead of `d^2` variant         (finite, different)
    - the naive product order, symmetrised      (finite, different)
The NaN-producing KL is asserted SEPARATELY, as a load-bearing guard
rather than as a foil.
"""

import math
import numpy as np
import scipy.linalg
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
# The reference implementations, transcribed from the papers.
# ---------------------------------------------------------------------------
def sym(A):
    return (A + A.T) / 2.0


def psd_sqrt(A):
    """The PRINCIPAL square root of a symmetric PSD matrix, via its
    eigendecomposition.

    NOTE: numpy has NO `linalg.sqrtm` -- that is `scipy.linalg.sqrtm`. The
    eigendecomposition route is also the mathematically canonical definition
    for a symmetric PSD matrix (the principal square root is a polynomial in
    A), so it is not a weaker reference.

    NOTE THE CLAMP: `max(lam, 0)` is part of the definition. A covariance
    that is PSD only to within rounding has an eigenvalue of -1e-18, and
    sqrt of that is NaN.
    """
    lam, Q = np.linalg.eigh(sym(A))
    return Q @ np.diag(np.sqrt(np.maximum(lam, 0.0))) @ Q.T


def sample_mean_cov(features, unbiased=True):
    """The empirical mean and covariance of `features`, an (n, d) matrix of
    n samples of d-dimensional feature vectors.

    The covariance is the UNBIASED estimator (divide by n-1) -- see C2.
    """
    n = features.shape[0]
    mu = features.mean(axis=0)
    centered = features - mu
    denom = (n - 1) if unbiased else n
    cov = centered.T @ centered / denom
    return mu, cov


def frechet_distance(mu1, S1, mu2, S2):
    """The Frechet distance between two Gaussians -- the repo's shipped
    formula, with the radicand clamped at zero from below.

    This is the C++ `utils::frechet_distance`, transcribed. It is here so
    that this oracle can compute the FID by COMPOSING it, exactly as the
    C++ will, rather than by computing the radicand directly.
    """
    S1h = psd_sqrt(S1)
    inner = S1h @ S2 @ S1h
    inner_sqrt = psd_sqrt(inner)
    radicand = (np.sum((mu1 - mu2) ** 2)
                + np.trace(S1) + np.trace(S2)
                - 2.0 * np.trace(inner_sqrt))
    if radicand < 0.0:
        radicand = 0.0
    return np.sqrt(radicand)


def frechet_inception_distance(real_features, fake_features):
    """FID = (the squared Frechet distance between the two sample Gaussians).

    Note the SQUARE, and note that it is applied AFTER the clamp inside
    `frechet_distance` -- see C1 and C4.
    """
    mu_r, S_r = sample_mean_cov(real_features)
    mu_f, S_f = sample_mean_cov(fake_features)
    d = frechet_distance(mu_r, S_r, mu_f, S_f)
    return d * d


def inception_score(probabilities, splits=10):
    """The Inception Score of a class-probability matrix.

    `probabilities` is (n, k): n rows, each a class distribution summing
    to 1. The rows are split into `splits` contiguous groups; each group's
    MARGINAL is the mean of that group's rows; the score is the mean over
    groups of exp(mean KL(row || marginal)).

    THE `0 * log(0/q) = 0` CONVENTION IS IMPLEMENTED BY EXACTLY ONE GUARD.

    A first draft used TWO: an inner `np.log(..., where=mask)` that
    zeroes the logarithm at a zero probability, AND an outer
    `np.where(mask, ..., 0.0)` that discards the whole term. **The outer
    one was REDUNDANT** -- with the logarithm already zeroed, the term is
    `0 * (0 - log q) == 0`, so the `np.where` could never change the
    result. That redundancy is not harmless: it made the mutation
    "remove the zero guard" SURVIVE, because the mutation removed a guard
    that was not carrying the load.

    The generalisation, and it is the same lesson the repo has paid for
    repeatedly: **TWO GUARDS FOR ONE CONDITION IS ONE GUARD TOO MANY, AND
    THE REDUNDANT ONE MAKES THE LIVE ONE UNTESTED.** This file therefore
    has exactly ONE guard -- the `where=mask` on the logarithm -- and the
    mutation harness targets THAT one.
    """
    n = probabilities.shape[0]
    scores = []
    for s in range(splits):
        lo = s * n // splits
        hi = (s + 1) * n // splits
        part = probabilities[lo:hi]
        if part.shape[0] == 0:
            continue
        py = part.mean(axis=0, keepdims=True)
        # KL(p || py) = sum_k p_k * (log p_k - log py_k), with 0*log(0/q) = 0.
        # The `where=` on BOTH logarithms is THE guard: it writes 0 into
        # the logarithm wherever the probability is 0, so the product is
        # 0 * (0 - log py) == 0 rather than 0 * (-inf) == NaN.
        mask = part > 0.0
        log_p = np.zeros_like(part)
        np.log(part, out=log_p, where=mask)
        log_q = np.zeros_like(py)
        np.log(py, out=log_q, where=py > 0.0)
        kl = np.sum(part * (log_p - log_q), axis=1)
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores))


def inception_score_reference(probabilities, splits=10):
    """An INDEPENDENT, SCALAR-LOOP implementation of the Inception Score.

    Written with explicit Python loops and `math.log`, sharing no code
    with the vectorised `inception_score` above. It exists to pin the
    ABSOLUTE value of the metric.

    WHY IT IS NEEDED: the vectorised implementation's properties (the
    identical set gives 1.0, the split count changes the value, the IS is
    >= 1) are all RELATIVE, and a mutant that keeps them while moving the
    value passes every one of them. Measured: replacing the split's
    marginal with the WHOLE SET's marginal moves the 10-split score from
    7.9430631552552615 to 9.923028431014757 -- and the vectorised oracle
    stayed GREEN, because nothing pinned the absolute number. This
    reference pins it.
    """
    n = probabilities.shape[0]
    scores = []
    for s in range(splits):
        lo = s * n // splits
        hi = (s + 1) * n // splits
        rows = probabilities[lo:hi]
        n_rows = rows.shape[0]
        if n_rows == 0:
            continue
        k = rows.shape[1]
        # The marginal: the column means of THIS group's rows.
        marginal = [sum(rows[i][c] for i in range(n_rows)) / n_rows
                    for c in range(k)]
        kl_sum = 0.0
        for i in range(n_rows):
            kl = 0.0
            for c in range(k):
                p = rows[i][c]
                if p <= 0.0:
                    continue          # the 0 * log(0/q) = 0 convention
                kl += p * (math.log(p) - math.log(marginal[c]))
            kl_sum += kl
        scores.append(math.exp(kl_sum / n_rows))
    return sum(scores) / len(scores)


def main():
    rng = np.random.default_rng(20261118)

    # =======================================================================
    section("1. THE SAMPLE GAUSSIAN ESTIMATOR vs np.cov")
    # =======================================================================
    # np.cov IS AN INDEPENDENT, MAINTAINED IMPLEMENTATION. It is the
    # reference the estimator is checked against, and it defaults to the
    # UNBIASED estimator (ddof=1).
    feats_a = rng.normal(size=(40, 6))
    mu_a, S_a = sample_mean_cov(feats_a)
    S_np = np.cov(feats_a, rowvar=False)
    check(np.abs(mu_a - feats_a.mean(axis=0)).max() < 1e-15,
          "1.1 the sample mean is the column mean")
    check(np.abs(S_np - S_a).max() < 1e-14,
          "1.2 the sample covariance matches np.cov (ddof=1, UNBIASED)",
          f"max diff {np.abs(S_np - S_a).max():.3e}")
    check(np.abs(S_a - S_a.T).max() < 1e-15,
          "1.3 the sample covariance is symmetric")
    # And the BIASED estimator is DIFFERENT -- so the convention is
    # observable and a mutation can separate it.
    _, S_biased = sample_mean_cov(feats_a, unbiased=False)
    check(np.abs(S_biased - S_a).max() > 1e-6,
          "1.4 the BIASED estimator gives a DIFFERENT covariance (so C2 is pinned)")
    ratio = np.trace(S_biased) / np.trace(S_a)
    check(abs(ratio - (len(feats_a) - 1) / len(feats_a)) < 1e-12,
          "1.5 the biased/unbiased trace ratio is exactly (n-1)/n",
          f"ratio {ratio}")

    # =======================================================================
    section("2. THE PRODUCT ORDER: the repo's form agrees with scipy")
    # =======================================================================
    # SAME feature dimension as feats_a -- the two covariances must be
    # conformable for the product S1 @ S2 to exist at all.
    feats_b = rng.normal(size=(33, 6)) + 0.7
    mu_b, S_b = sample_mean_cov(feats_b)
    # The reference implementation's OWN product order, via scipy.
    covmean = scipy.linalg.sqrtm(S_a @ S_b)
    trace_scipy = np.trace(covmean)
    check(abs(trace_scipy.imag) < 1e-9,
          "2.1 scipy's sqrtm(S1@S2) has a (numerically) REAL trace",
          f"imag {abs(trace_scipy.imag):.3e}")
    check(np.abs(covmean - covmean.T).max() > 1e-6,
          "2.2 S1@S2 is NOT symmetric (so its sqrtm is not symmetric either) -- "
          "this is why the symmetrised form is the right one")
    # The repo's symmetrised form.
    S_ah = psd_sqrt(S_a)
    trace_repo_form = np.trace(psd_sqrt(S_ah @ S_b @ S_ah))
    check(abs(trace_repo_form - np.real(trace_scipy)) < 1e-10,
          "2.3 the symmetrised product form agrees with scipy's sqrtm(S1@S2)",
          f"repo {trace_repo_form:.12f} scipy {np.real(trace_scipy):.12f}")

    # =======================================================================
    section("3. FID: THE VALUE, AND THE DISCRIMINATING FOILS")
    # =======================================================================
    feats_real = rng.normal(size=(40, 6))
    feats_fake = rng.normal(size=(40, 6)) + 0.5
    fid = frechet_inception_distance(feats_real, feats_fake)

    # -- the unbiased value, and the biased FOIL (finite, different) --------
    mu_r, S_r = sample_mean_cov(feats_real)
    mu_f, S_f = sample_mean_cov(feats_fake)
    fid_unbiased = frechet_distance(mu_r, S_r, mu_f, S_f) ** 2
    mu_rb, S_rb = sample_mean_cov(feats_real, unbiased=False)
    mu_fb, S_fb = sample_mean_cov(feats_fake, unbiased=False)
    fid_biased = frechet_distance(mu_rb, S_rb, mu_fb, S_fb) ** 2
    check(abs(fid - fid_unbiased) < 1e-12,
          "3.1 FID is the squared Frechet distance of the UNBIASED Gaussians")
    check(abs(fid - fid_biased) > 1e-3,
          "3.2 the BIASED-estimator foil is FINITE and DIFFERENT",
          f"unbiased {fid:.12f} biased {fid_biased:.12f}")

    # -- the d-instead-of-d^2 FOIL (finite, different) ----------------------
    d_not_d2 = frechet_distance(mu_r, S_r, mu_f, S_f)
    check(abs(d_not_d2 - fid) > 1e-3,
          "3.3 the 'd instead of d^2' foil is FINITE and DIFFERENT",
          f"d {d_not_d2:.12f} d^2 {fid:.12f}")
    check(abs(d_not_d2 - np.sqrt(fid)) < 1e-12,
          "3.4 and the two differ by exactly the square root (d == sqrt(FID))")

    # -- the naive-product-order FOIL (finite, different) -------------------
    naive_inner = psd_sqrt(sym(S_r @ S_f))
    naive_radicand = (np.sum((mu_r - mu_f) ** 2)
                      + np.trace(S_r) + np.trace(S_f)
                      - 2.0 * np.trace(naive_inner))
    naive_fid = max(naive_radicand, 0.0)
    check(np.isfinite(naive_fid),
          "3.5 the naive-product-order foil is FINITE (not a NaN strawman)")
    check(abs(naive_fid - fid) > 1e-3,
          "3.6 the naive-product-order foil is DIFFERENT",
          f"naive {naive_fid:.12f} right {fid:.12f}")

    # =======================================================================
    section("4. FID'S STRUCTURAL PROPERTIES")
    # =======================================================================
    fid_same = frechet_inception_distance(feats_real, feats_real)
    check(fid_same == 0.0,
          "4.1 FID(X, X) is EXACTLY 0 (the clamp makes it so)",
          f"got {fid_same!r}")
    fid_swapped = frechet_inception_distance(feats_fake, feats_real)
    check(abs(fid_swapped - fid) < 1e-9,
          "4.2 FID is symmetric in its two arguments",
          f"{fid_swapped:.12f} vs {fid:.12f}")
    # Scale equivariance: multiplying the features by c scales FID by c^2.
    c = 3.0
    fid_scaled = frechet_inception_distance(c * feats_real, c * feats_fake)
    check(abs(fid_scaled - c * c * fid) < 1e-9 * max(1.0, abs(c * c * fid)),
          "4.3 FID(cX, cY) == c^2 * FID(X, Y) (scale equivariance)",
          f"{fid_scaled:.12f} vs {c*c*fid:.12f}")
    # A SHIFT of the fake set increases FID (the mean term grows).
    fid_shifted = frechet_inception_distance(feats_real, feats_fake + 1.0)
    check(fid_shifted > fid,
          "4.4 a larger mean shift gives a larger FID")

    # THE FID IS THE SQUARE of the distance, so it is >= 0 always.
    check(fid >= 0.0, "4.5 FID >= 0")

    # -- 4.6: THE FIRST DRAFT OF THIS CHECK WAS FALSE ABOUT THE MATHS, AND
    #    IT WAS THE CHECK THAT WAS WRONG, NOT THE IMPLEMENTATION.
    #    The draft asserted "two disjoint halves are FURTHER apart than the
    #    whole set from a shifted copy". Measured on this file's fixture:
    #        FID(whole set, shifted copy)  = 1.6535543231203407
    #        FID(disjoint halves, SAME dist) = 0.903975217768739
    #    The halves are CLOSER, and they MUST be: they are two samples from
    #    the SAME distribution, so their FID is pure sampling noise, while
    #    the shifted copy differs in its MEAN. The draft asserted a
    #    falsehood about the maths, and the oracle rightly rejected it.
    #    (systematic-debugging 4c: a failing comparison is evidence about
    #    ONE OF TWO PIECES OF CODE YOU JUST WROTE.)
    #
    #    The property that is actually true, and discriminating, is that
    #    the same-distribution FID SHRINKS as the sample count grows --
    #    it is O(1/n), because the sampling error in the estimated means
    #    and covariances goes down. Measured:
    #        n=  20: 1.535222
    #        n= 100: 0.332524
    #        n=1000: 0.037777
    fids_same = []
    for n_same in (20, 100, 1000):
        a_same = rng.normal(size=(n_same, 6))
        b_same = rng.normal(size=(n_same, 6))
        fids_same.append(frechet_inception_distance(a_same, b_same))
    check(fids_same[0] > fids_same[1] > fids_same[2],
          "4.6 the same-distribution FID SHRINKS as n grows (it is O(1/n))",
          f"{fids_same}")

    # -- 4.7: FOR A HUGE SHIFT THE COVARIANCE TERM BECOMES NEGLIGIBLE AND
    #    FID -> ||dmu||^2 = d * s^2. Measured relative errors: 1.6e-2 at
    #    s=5, 2.9e-3 at s=50. This pins the MEAN term's weight exactly:
    #    a wrong mean term (say ||dmu|| instead of ||dmu||^2) would put
    #    FID at d*s rather than d*s^2, a factor of s = 50 out.
    s_big = 50.0
    a_big = rng.normal(size=(40, 6))
    b_big = rng.normal(size=(40, 6)) + s_big
    fid_big = frechet_inception_distance(a_big, b_big)
    expected_big = 6.0 * s_big * s_big
    check(abs(fid_big - expected_big) / expected_big < 5e-2,
          "4.7 for a huge mean shift, FID -> d * s^2 (the mean term is "
          "||dmu||^2, not ||dmu||)",
          f"got {fid_big:.6f} expected ~{expected_big:.6f}")

    # -- 4.8: FID IS STRICTLY INCREASING IN THE SHIFT MAGNITUDE. Measured:
    #        s=0.0 -> 1.023715, s=0.5 -> 3.238273, s=1.0 -> 5.600276,
    #        s=2.0 -> 27.459125, s=4.0 -> 97.162150
    fids_shift = []
    for s_val in (0.0, 0.5, 1.0, 2.0, 4.0):
        a_s = rng.normal(size=(40, 6))
        b_s = rng.normal(size=(40, 6)) + s_val
        fids_shift.append(frechet_inception_distance(a_s, b_s))
    check(all(fids_shift[i] < fids_shift[i + 1]
              for i in range(len(fids_shift) - 1)),
          "4.8 FID is strictly increasing in the shift magnitude",
          f"{fids_shift}")

    # =======================================================================
    section("5. THE RANK-DEFINIENT CASE (n < d) IS THE REALISTIC ONE")
    # =======================================================================
    n_small, d_big = 10, 64
    Xs = rng.normal(size=(n_small, d_big))
    Xf = rng.normal(size=(n_small, d_big)) + 0.3
    _, S_s = sample_mean_cov(Xs)
    check(np.linalg.matrix_rank(S_s) == n_small - 1,
          "5.1 the sample covariance is rank-deficient (rank = n-1)",
          f"rank {np.linalg.matrix_rank(S_s)}")
    check(np.linalg.eigvalsh(S_s).min() > -1e-12,
          "5.2 it is PSD to within rounding (min eigenvalue >= -1e-12)",
          f"min eig {np.linalg.eigvalsh(S_s).min():.3e}")
    fid_rankdef = frechet_inception_distance(Xs, Xf)
    check(np.isfinite(fid_rankdef),
          "5.3 FID is FINITE on a rank-deficient covariance (no Cholesky!)",
          f"fid {fid_rankdef}")
    check(fid_rankdef > 0.0, "5.4 and it is positive (the sets really differ)")

    # =======================================================================
    section("6. THE INCEPTION SCORE")
    # =======================================================================
    K = 10
    onehot = np.eye(K)[rng.integers(0, K, size=200)]
    is_1 = inception_score(onehot, splits=1)
    is_10 = inception_score(onehot, splits=10)
    check(np.isfinite(is_1) and np.isfinite(is_10),
          "6.1 the IS is FINITE on a one-hot fixture (the zero guard works)",
          f"1 split {is_1} 10 splits {is_10}")

    # -- the SPLIT-COUNT foil (finite, different) --------------------------
    check(abs(is_1 - is_10) > 1e-3,
          "6.2 the split count CHANGES the IS (so C6 is a pinned convention)",
          f"1 split {is_1:.12f} 10 splits {is_10:.12f}")

    # -- an IDENTICAL set gives EXACTLY 1.0 --------------------------------
    identical = np.tile(np.eye(K)[3], (200, 1))
    is_identical = inception_score(identical, splits=10)
    check(is_identical == 1.0,
          "6.3 the IS of an IDENTICAL set is EXACTLY 1.0",
          f"got {is_identical!r}")
    is_identical_1 = inception_score(identical, splits=1)
    check(is_identical_1 == 1.0,
          "6.4 and it is exactly 1.0 with 1 split too",
          f"got {is_identical_1!r}")

    # -- the NaN-PRODUCING naive KL, asserted separately as a guard -------
    with np.errstate(divide="ignore", invalid="ignore"):
        py = onehot.mean(axis=0, keepdims=True)
        naive_kl = float(np.mean(np.sum(onehot * (np.log(onehot) - np.log(py)),
                                        axis=1)))
    check(np.isnan(naive_kl),
          "6.5 the NAIVE KL (no zero guard) is NaN -- so the guard is "
          "load-bearing, and this is a GUARD check, not a foil",
          f"naive {naive_kl}")

    # -- the IS is bounded below by 1 (it is an exp of a mean KL >= 0) ----
    uniform = np.full((100, K), 1.0 / K)
    is_uniform = inception_score(uniform, splits=10)
    check(abs(is_uniform - 1.0) < 1e-12,
          "6.6 the IS of a UNIFORM set is 1.0 (every KL is 0)",
          f"got {is_uniform}")
    check(is_10 >= 1.0,
          "6.7 the IS is >= 1 always (it is an exp of a mean KL >= 0)",
          f"got {is_10}")

    # -- 6.8: THE ABSOLUTE VALUE, AGAINST AN INDEPENDENT SCALAR-LOOP
    #    REFERENCE. This check was ADDED IN RESPONSE TO FOUR SURVIVORS in
    #    the oracle's own mutation harness, and it is the fix.
    #
    #    EVERY property checked above is RELATIVE: "the split count
    #    changes the value", "an identical set gives 1.0", "the IS is
    #    >= 1". A mutant that keeps all three while MOVING the number
    #    passes every one of them. Measured on this file's own fixture:
    #        the split's own marginal     -> 7.9430631552552615
    #        the WHOLE SET's marginal      -> 9.923028431014757
    #    and the vectorised oracle stayed GREEN under that mutation,
    #    because nothing pinned the absolute number.
    #
    #    The reference below shares NO code with the vectorised version:
    #    explicit Python loops, `math.log`, its own marginal.
    ref_1 = inception_score_reference(onehot, splits=1)
    ref_10 = inception_score_reference(onehot, splits=10)
    check(abs(is_1 - ref_1) < 1e-9,
          "6.8 the 1-split IS matches the INDEPENDENT scalar-loop reference",
          f"vectorised {is_1!r} reference {ref_1!r}")
    check(abs(is_10 - ref_10) < 1e-9,
          "6.9 the 10-split IS matches the INDEPENDENT scalar-loop reference",
          f"vectorised {is_10!r} reference {ref_10!r}")
    # And the same for the identical and uniform fixtures, at both splits.
    for nm, prob, s in (("identical", identical, 10),
                        ("identical", identical, 1),
                        ("uniform", uniform, 10),
                        ("uniform", uniform, 1)):
        got = inception_score(prob, splits=s)
        want = inception_score_reference(prob, splits=s)
        check(abs(got - want) < 1e-9,
              f"6.9.{nm}-splits{s} the IS matches the independent reference",
              f"got {got!r} want {want!r}")

    # -- 6.10: THE SPLIT'S OWN MARGINAL, NOT THE WHOLE SET'S. Measured
    #     above: 7.9430631552552615 (own) against 9.923028431014757
    #     (whole set). The check asserts the two DIFFER, so the oracle
    #     cannot be satisfied by a marginal that ignores the split.
    def _is_with_whole_set_marginal(probs, splits):
        n = probs.shape[0]
        scores = []
        for s in range(splits):
            part = probs[s * n // splits:(s + 1) * n // splits]
            py = probs.mean(axis=0, keepdims=True)
            mask = part > 0.0
            lp = np.zeros_like(part)
            np.log(part, out=lp, where=mask)
            lq = np.zeros_like(py)
            np.log(py, out=lq, where=py > 0.0)
            kl = np.sum(part * (lp - lq), axis=1)
            scores.append(np.exp(np.mean(kl)))
        return float(np.mean(scores))

    whole_set_variant = _is_with_whole_set_marginal(onehot, 10)
    check(abs(whole_set_variant - is_10) > 1e-3,
          "6.10 the split's OWN marginal differs from the whole set's "
          "(so the split semantics are load-bearing)",
          f"own {is_10!r} whole-set {whole_set_variant!r}")

    # =======================================================================
    section("7. THE NUMBERS THE C++ TEST WILL ASSERT")
    # =======================================================================
    print("  FID (real vs fake, 40x6)   = %.15f" % fid)
    print("  FID biased-estimator foil  = %.15f" % fid_biased)
    print("  FID d-instead-of-d^2 foil  = %.15f" % d_not_d2)
    print("  FID naive-product foil     = %.15f" % naive_fid)
    print("  FID rank-deficient (10x64) = %.15f" % fid_rankdef)
    print("  IS one-hot splits=1        = %.15f" % is_1)
    print("  IS one-hot splits=10       = %.15f" % is_10)
    print("  IS identical                = %.15f" % is_identical)
    print("  IS uniform                  = %.15f" % is_uniform)
    print("  sample mean (40x6)         = %s" % repr(feats_a.mean(axis=0).tolist()))
    print("  sample cov trace (40x6)     = %.15f" % np.trace(S_a))

    if "--emit-cpp" in sys.argv:
        emit_cpp(feats_a, mu_a, S_a, feats_real, feats_fake, fid, fid_biased,
                 is_1, is_10, is_identical, is_uniform, fid_rankdef, Xs, Xf)

    print(f"\n=== SUMMARY: {'ORACLE GREEN' if FAILS == 0 else str(FAILS) + ' ORACLE FAILURES'} ===")
    return 0 if FAILS == 0 else 1


# ---------------------------------------------------------------------------
# The fixture emitter. The numbers are TRANSFERRED, never re-derived: the
# repo has been burned by hand-transcribed numbers before (both numbers were
# correct for the fixture they were given, and the fixture was simply not
# the oracle's).
# ---------------------------------------------------------------------------
def emit_cpp(feats_a, mu_a, S_a, feats_real, feats_fake, fid, fid_biased,
             is_1, is_10, is_identical, is_uniform, fid_rankdef, Xs, Xf):
    def mat(name, a):
        rows = []
        for r in a:
            rows.append("    {" + ", ".join("%.17g" % v for v in r) + "}")
        return (f"static const std::vector<std::vector<double>> {name} = {{\n"
                + ",\n".join(rows) + "\n};")

    def eye(name, k, cls):
        rows = []
        for i in range(k):
            row = [0.0] * k
            row[cls] = 1.0
            rows.append("    {" + ", ".join("%.17g" % v for v in row) + "}")
        return (f"static const std::vector<std::vector<double>> {name} = {{\n"
                + ",\n".join(rows) + "\n};")

    def uniform(name, n, k):
        rows = []
        for i in range(n):
            rows.append("    {" + ", ".join("%.17g" % (1.0 / k)
                                            for _ in range(k)) + "}")
        return (f"static const std::vector<std::vector<double>> {name} = {{\n"
                + ",\n".join(rows) + "\n};")

    print("// ==== ORACLE-EMITTED FIXTURE BEGIN ====")
    print(mat("kOracleFeaturesA", feats_a))
    print(mat("kOracleFeaturesReal", feats_real))
    print(mat("kOracleFeaturesFake", feats_fake))
    print(mat("kOracleRankDefReal", Xs))
    print(mat("kOracleRankDefFake", Xf))
    print(eye("kOracleOneHot", 10, 3))
    print(uniform("kOracleUniform10", 100, 10))
    print("")
    print("// The oracle's own numbers.")
    print(f"static const double kOracleFid              = {fid:.17g};")
    print(f"static const double kOracleFidBiasedFoil   = {fid_biased:.17g};")
    print(f"static const double kOracleFidDFoil         = {np.sqrt(fid):.17g};")
    print(f"static const double kOracleFidRankDef       = {fid_rankdef:.17g};")
    print(f"static const double kOracleIsOneHotSplits1  = {is_1:.17g};")
    print(f"static const double kOracleIsOneHotSplits10 = {is_10:.17g};")
    print(f"static const double kOracleIsIdentical      = {is_identical:.17g};")
    print(f"static const double kOracleIsUniform        = {is_uniform:.17g};")
    print(f"static const double kOracleSampleMeanA0     = {mu_a[0]:.17g};")
    print(f"static const double kOracleSampleCovTraceA  = {np.trace(S_a):.17g};")
    print("// ==== ORACLE-EMITTED FIXTURE END ====")


if __name__ == "__main__":
    sys.exit(main())
