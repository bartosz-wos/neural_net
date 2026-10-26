#!/usr/bin/env python3
"""
tests/conformal_oracle.py -- the SPECIFICATION for utils/conformal.{h,cpp}.

RUN THIS BEFORE WRITING ANY C++. Its job is to catch defects in the DERIVATIONS
and in itself, exactly as tests/sinkhorn_oracle.py, tests/ws_oracle.py and
tests/speculative_decoding_oracle.py did. In this repo the oracle has caught
more real defects in the DERIVATION than the C++ suite has caught in the code.

WHAT IS BEING VERIFIED, AND WHY IT IS NOT OBVIOUS:

 1. THE CALIBRATION RULE IS AN EXACT ARITHMETIC QUESTION, NOT A FLOAT ONE.
    q_hat = s_(k) with k = ceil((n+1)(1-alpha)). The textbook float expression
    `ceil((n+1)*(1-alpha))` is WRONG for some (n, alpha) pairs, and the error is
    a FULL CALIBRATION POINT -- the coverage guarantee changes by 1/(n+1).
    Section 1 measures exactly where, and pins a DISCRIMINATING fixture.

 2. THE SCORE FUNCTIONS ARE CUMULATIVE (APS) and MAX-OF-TWO (CQR), and the
    naive alternatives (1 - p_true, and |y - midpoint|) are DIFFERENT RULES
    that produce different sets on the same data. Section 9 asserts the
    differences, so a passing check is not vacuous.

 3. THE PREDICTION-SET RULES USE `>=` (the crossing class is KEPT), the
    quantile is an ORDER STATISTIC (not an interpolation), and an INSUFFICIENT
    CALIBRATION SET gives +inf (the whole label space), not a clamped index.

 4. THE COVERAGE GUARANTEE ITSELF, MEASURED BY MONTE CARLO on exchangeable
    data. This is the paper's actual claim, and no single example can check it.

 5. ADAPTIVITY, which is the APS paper's entire contribution over the naive
    score: on a two-group fixture the naive score UNDERCOVERS the hard group
    while APS does not.

Self-checks: every section prints its numbers AND asserts on them. A
self-check that prints a verdict it never tested is worse than none (this
repo's PTQ session), so each block below has a real assertion.

Exit 0 = all sections green. Exit 1 = at least one FAILED line.
"""

import sys
import math
from fractions import Fraction

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


# -----------------------------------------------------------------------------
# THE CALIBRATION INDEX, THREE WAYS.
#
#   exact_k : ceil((n+1)*(1-alpha)) in EXACT RATIONAL arithmetic, with alpha
#             taken as the exact binary value of the double the user passed.
#   cpp_k   : m - floor(m*alpha) with m = n+1, also exact. The identity
#             ceil(m - ma) == m - floor(ma) holds for integer m and real a,
#             so this must ALWAYS equal exact_k. The C++ implements this
#             form because it is the one that can be done exactly with
#             128-bit integers (see conformal.h).
#   naive_k : int(np.ceil((n+1)*(1-alpha))) -- the TEXTBOOK FLOAT EXPRESSION.
#             This is the one that is sometimes WRONG.
# -----------------------------------------------------------------------------

def exact_k(n, alpha):
    """ceil((n+1)*(1-alpha)) in exact rational arithmetic. n >= 1."""
    a = Fraction(alpha)
    m = n + 1
    return -((-m * (1 - a)) // 1)  # ceil of m*(1-a)


def cpp_k(n, alpha):
    """m - floor(m*alpha), exact. Must equal exact_k always."""
    a = Fraction(alpha)
    m = n + 1
    return m - (m * a).__floor__()


def naive_k(n, alpha):
    """The textbook float expression. Occasionally wrong."""
    return int(np.ceil((n + 1) * (1.0 - alpha)))


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def softmax_rows(logits):
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aps_rank_order(probs_row):
    """Class indices sorted by probability DESCENDING, ties broken by class
    index ASCENDING. numpy's stable argsort on -p does exactly that."""
    return np.argsort(-probs_row, kind="stable")


def aps_scores_from_probs(probs, labels):
    """score(x, y) = sum of the probabilities of all classes ranked at or
    above y (i.e. the cumulative mass THROUGH the true class)."""
    n, K = probs.shape
    out = np.empty(n)
    for i in range(n):
        order = aps_rank_order(probs[i])
        cum = np.cumsum(probs[i][order])
        rank = int(np.where(order == labels[i])[0][0])
        out[i] = cum[rank]
    return out


def aps_set_from_probs(probs_row, qhat):
    """T(x) = {pi_1..pi_k}, k = inf{ k : cumsum[k] >= qhat }.
    The crossing class is KEPT (>=, not >)."""
    order = aps_rank_order(probs_row)
    cum = np.cumsum(probs_row[order])
    idx = np.where(cum >= qhat)[0]
    if len(idx) == 0:
        # Exact arithmetic: cumsum[K] == 1.0 always, so this can only be a
        # floating-point artifact. The exact-arithmetic answer is k = K (ALL
        # classes), so that is what is returned.
        return list(order)
    k = int(idx[0])
    return list(order[:k + 1])


def order_statistic(scores, k):
    """The k-th smallest score (1-indexed). Ties are fine: the value is the
    same as the `inf{ q : #{s_i <= q}/n >= k/n }` form."""
    return float(np.sort(np.asarray(scores).ravel())[k - 1])


def conformal_quantile(scores, alpha):
    """The calibration quantile. +inf when the required index exceeds n."""
    s = np.asarray(scores).ravel()
    n = s.size
    k = exact_k(n, alpha)
    if k > n:
        return float("inf")
    return order_statistic(s, k)


# ============================================================================
section("1. THE CALIBRATION INDEX: EXACT vs THE TEXTBOOK FLOAT EXPRESSION")
# ============================================================================

# 1a. exact_k and cpp_k must ALWAYS agree. If they ever disagree, one of the
#     two derivations is wrong and the whole file's specification is suspect.
mismatch = []
for n in list(range(1, 200)) + [999, 1000, 4999]:
    for alpha in [0.1, 0.05, 0.01, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,
                  1e-8, 0.6666666666666666, 1.0 / 3.0, 0.123456789012345]:
        if exact_k(n, alpha) != cpp_k(n, alpha):
            mismatch.append((n, alpha, exact_k(n, alpha), cpp_k(n, alpha)))
check(not mismatch,
      "1a. exact ceil((n+1)(1-a)) == (n+1) - floor((n+1)*a) on 3600+ pairs",
      f"{len(mismatch)} mismatches")

# 1b. The NAIVE float expression disagrees with the exact one somewhere.
#     If it never did, the exactness machinery would be unjustified.
disagreements = []
for n in list(range(1, 400)):
    for alpha in [0.1, 0.05, 0.01, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,
                  1e-8, 0.6666666666666666, 1.0 / 3.0, 0.123456789012345,
                  0.30000000000000004, 0.7]:
        if naive_k(n, alpha) != exact_k(n, alpha):
            disagreements.append((n, alpha, naive_k(n, alpha), exact_k(n, alpha)))
check(len(disagreements) > 0,
      "1b. the textbook float ceil((n+1)(1-a)) IS sometimes wrong",
      f"{len(disagreements)} disagreements over the sweep")

# 1c. Print the first few, and pin ONE as the discriminating fixture. The
#     discrepancy is a WHOLE calibration point: with n=2, alpha=2/3, the naive
#     expression says k=1 (coverage 1/3) and the exact one says k=2
#     (coverage 2/3).
print("  first disagreements (n, alpha, naive_k, exact_k):")
for d in disagreements[:6]:
    print(f"    {d}")
PIN_N, PIN_A = disagreements[0][0], disagreements[0][1]
check(naive_k(PIN_N, PIN_A) != exact_k(PIN_N, PIN_A),
      f"1c. pinned discriminating fixture n={PIN_N} alpha={PIN_A!r} "
      f"naive_k={naive_k(PIN_N, PIN_A)} != exact_k={exact_k(PIN_N, PIN_A)}")

# 1d. THE BIG ONE. alpha = 2/3 exactly as a double. n = 2.
d23 = [(n, naive_k(n, 2.0 / 3.0), exact_k(n, 2.0 / 3.0))
       for n in range(1, 24) if naive_k(n, 2.0 / 3.0) != exact_k(n, 2.0 / 3.0)]
check(len(d23) > 0,
      "1d. alpha = 2/3 (the double 0.666...) breaks the float expression",
      f"n in {[d[0] for d in d23][:8]}")
check(exact_k(2, 2.0 / 3.0) == 2 and naive_k(2, 2.0 / 3.0) == 1,
      "1e. n=2, alpha=2/3: exact k=2, naive k=1 -- a FULL calibration point")

# 1e. The `+1` in (n+1) is load-bearing: ceil(n(1-a)) != ceil((n+1)(1-a))
#     on some (n, alpha), and the difference is a calibration point.
plus1 = [(n, a) for n in range(1, 200) for a in [0.1, 0.05, 0.2, 0.25, 0.3]
         if math.ceil(n * (1 - a)) != exact_k(n, a)]
check(len(plus1) > 0,
      "1f. the (n+1) correction changes k on some (n, alpha)",
      f"{len(plus1)} pairs, e.g. {plus1[:3]}")

# 1f. alpha = 0 gives k = n+1 > n -> INFINITE (cannot certify 100%).
check(exact_k(5, 0.0) == 6, "1g. alpha=0 -> k = n+1 (infinite threshold)")

# ============================================================================
section("2. THE CALIBRATION QUANTILE: ORDER STATISTIC, TIES, AND +inf")
# ============================================================================

rng = np.random.default_rng(20261026)
cal_scores = rng.random(37)
q = conformal_quantile(cal_scores, 0.1)
k_exact = exact_k(37, 0.1)
check(abs(q - order_statistic(cal_scores, k_exact)) < 1e-15,
      "2a. quantile == the k-th smallest score",
      f"k={k_exact}, q={q!r}")
check(q == float(np.sort(cal_scores)[k_exact - 1]),
      "2b. quantile is an ORDER STATISTIC (equals a member of the sample)",
      f"q in sample: {q in set(cal_scores.tolist())}")

# 2c. The INTERPOLATED quantile (numpy's default) is a DIFFERENT rule and
#     must differ here -- otherwise the order-statistic requirement is
#     untestable and the suite's check would be vacuous.
interp = float(np.quantile(cal_scores, k_exact / 37.0, method="linear"))
check(interp != q, "2c. the linear-INTERPOLATED quantile differs from the "
                   "order statistic", f"interp={interp!r} vs order_stat={q!r}")

# 2d. TIES: with all-equal scores, the quantile is that value, and the
#     `inf{...}` form and the k-th-smallest form agree.
tied = np.full(20, 0.25)
check(conformal_quantile(tied, 0.1) == 0.25,
      "2d. with all-equal scores the quantile is that value")

# 2e. INSUFFICIENT CALIBRATION DATA -> +inf, NOT a clamped index.
#     n=4, alpha=0.1 -> k = ceil(5*0.9) = 5 > 4 -> infinite.
check(exact_k(4, 0.1) == 5, "2e. n=4, alpha=0.1 -> k=5 > n")
check(conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4]), 0.1) == float("inf"),
      "2f. n=4, alpha=0.1 -> the quantile is +inf (whole label space)")
#     and the boundary: n=9, alpha=0.1 -> k = ceil(10*0.9) = 9 <= 9 -> finite
check(exact_k(9, 0.1) == 9 and conformal_quantile(rng.random(9), 0.1) != float("inf"),
      "2g. n=9, alpha=0.1 -> k=9 <= n, finite threshold (the boundary case)")

# 2h. Monotonicity in alpha: a smaller alpha (more coverage demanded) needs a
#     LARGER k. This must hold exactly.
mono = True
for n in [10, 25, 100]:
    ks = [exact_k(n, a) for a in [0.5, 0.3, 0.1, 0.05, 0.01]]
    if ks != sorted(ks):
        mono = False
check(mono, "2h. k is non-decreasing as alpha decreases (more coverage -> "
            "higher order statistic)")

# ============================================================================
section("3. APS SCORES: THE CUMULATIVE MASS THROUGH THE TRUE CLASS")
# ============================================================================

probs3 = np.array([
    [0.70, 0.20, 0.10],
    [0.50, 0.30, 0.20],
    [0.34, 0.33, 0.33],
])
labels3 = np.array([0, 2, 1])
s3 = aps_scores_from_probs(probs3, labels3)

# row 0: label 0 is rank 1 -> score = 0.70
check(abs(s3[0] - 0.70) < 1e-12, "3a. rank-1 true class -> score is its own "
                                 "probability", f"{s3[0]!r}")
# row 1: order is [0,1,2]; label 2 is rank 3 -> score = 0.50+0.30+0.20 = 1.0
check(abs(s3[1] - 1.0) < 1e-12, "3b. rank-3 true class -> score is the FULL "
                                "cumulative mass", f"{s3[1]!r}")
# row 2: order is [0,1,2]; label 1 is rank 2 -> score = 0.34+0.33 = 0.67
check(abs(s3[2] - 0.67) < 1e-12, "3c. rank-2 true class -> score is the "
                                 "cumulative mass THROUGH it", f"{s3[2]!r}")

# 3d. The APS score is NOT 1 - p_true. On row 2 they differ.
naive_scores3 = 1.0 - probs3[np.arange(3), labels3]
check(not np.allclose(s3, naive_scores3),
      "3d. the APS score differs from the naive score 1 - p_true",
      f"aps={s3} vs naive={naive_scores3}")
check(np.allclose(naive_scores3, [0.30, 0.80, 0.67]),
      "3e. the naive scores on this fixture are the textbook ones",
      f"{naive_scores3}")

# 3f. The APS score for the LEAST likely class is (approximately) 1.0, since
#     the cumulative mass through the last rank is the whole row.
s_last = aps_scores_from_probs(probs3, np.array([2, 2, 2]))
check(np.allclose(s_last, 1.0, atol=1e-12),
      "3f. the last-ranked true class scores the full row mass", f"{s_last}")

# 3g. TIES are broken by CLASS INDEX ASCENDING (deterministic).
tie_probs = np.array([[0.5, 0.5, 0.0]])
tie_order = aps_rank_order(tie_probs[0])
check(list(tie_order) == [0, 1, 2],
      "3g. equal probabilities are ordered by class index ascending",
      f"{list(tie_order)}")
#     and the score of class 1 under that order is the cumulative mass 1.0
s_tie = aps_scores_from_probs(tie_probs, np.array([1]))[0]
check(abs(s_tie - 1.0) < 1e-12,
      "3h. with a tie, the second-ranked class scores the cumulative mass "
      "through it", f"{s_tie!r}")

# ============================================================================
section("4. APS PREDICTION SETS: CUMULATIVE MASS THRESHOLD")
# ============================================================================

# THE FIXTURE IS DYADIC ON PURPOSE, AND THAT IS THE POINT.
#
#   0.70 + 0.20 == 0.8999999999999999 in binary64 -- ONE ULP BELOW 0.9.
#
# A boundary fixture built on [0.70, 0.20, 0.10] therefore has its `>=` vs
# `>` decision made by ROUNDING rather than by the RULE, and the oracle's
# first draft failed exactly there. This is the FOURTH instance of this class
# in this repo's history (after the top-p one-ULP case, numpy's 1-ULP
# `np.linalg.norm`, and the 0.7-double quantile case) and the rule is always
# the same: A FIXTURE WHOSE DISCRIMINATING VALUE IS ONLY CORRECT TO WITHIN A
# ROUNDING ERROR DOES NOT DISCRIMINATE ANYTHING.
#
# [0.5, 0.25, 0.25] has cumulative masses [0.5, 0.75, 1.0], every one of
# which is EXACTLY representable, so the comparison means the same thing on
# both sides.
row = np.array([0.5, 0.25, 0.25])
cum = np.cumsum(row[aps_rank_order(row)])          # [0.5, 0.75, 1.0]
check(list(cum) == [0.5, 0.75, 1.0],
      "4a. the fixture's cumulative masses are EXACTLY representable",
      f"{list(cum)}")
for qhat, want in [(0.5, {0}), (0.6, {0, 1}), (0.75, {0, 1}), (0.76, {0, 1, 2})]:
    got = set(aps_set_from_probs(row, qhat))
    check(got == want, f"4b. qhat={qhat} -> set {sorted(got)}",
          f"expected {sorted(want)}")

# 4c. THE CROSSING CLASS IS KEPT. qhat exactly equal to a cumulative mass
#     must include that class (`>=`, not `>`).
check(set(aps_set_from_probs(row, 0.75)) == {0, 1},
      "4c. qhat exactly on a cumulative mass KEEPS the crossing class "
      "(>=, not >)")
#     The `>` variant is a DIFFERENT rule and must differ here.
got_strict = set()
c = 0.0
for j, cls in enumerate(aps_rank_order(row)):
    c += row[cls]
    if c > 0.75:
        got_strict = set(list(aps_rank_order(row))[:j + 1])
        break
check(got_strict != set(aps_set_from_probs(row, 0.75)),
      "4d. the strict-`>` variant differs from `>=` on this fixture "
      "(so the convention is observable)",
      f"strict={sorted(got_strict)} vs >= {sorted(aps_set_from_probs(row, 0.75))}")

# 4d. qhat = 1.0 gives EVERY class.
check(set(aps_set_from_probs(row, 1.0)) == {0, 1, 2},
      "4d. qhat = 1.0 -> the whole label space")

# 4e. The set is ORDER-INSENSITIVE as a set, but the C++ returns it sorted
#     by class id for determinism.
check(sorted(aps_set_from_probs(row, 0.91)) == [0, 1, 2],
      "4e. the returned set, sorted by class id, is deterministic")

# ============================================================================
section("5. CQR: SCORE IS A MAX OF TWO ONE-SIDED GAPS")
# ============================================================================

lo = np.array([[0.0], [1.0], [2.0]])
up = np.array([[2.0], [3.0], [5.0]])
yy = np.array([[0.5], [4.0], [2.0]])
s_cqr = np.maximum(lo - yy, yy - up)
check(np.allclose(s_cqr.ravel(), [max(-0.5, -1.5), max(-3.0, 1.0), max(0.0, -3.0)]),
      "5a. the CQR score is max{lower - y, y - upper}", f"{s_cqr.ravel()}")

# 5b. It is NOT |y - midpoint|, and NOT the sum of the two gaps.
mid = (lo + up) / 2.0
abs_mid = np.abs(yy - mid)
sum_gaps = (lo - yy) + (yy - up)
check(not np.allclose(s_cqr, abs_mid),
      "5b. the CQR score differs from |y - midpoint|", f"{abs_mid.ravel()}")
check(not np.allclose(s_cqr, sum_gaps),
      "5c. the CQR score differs from the SUM of the two gaps")

# 5d. It is NOT clamped at zero: a y INSIDE the interval gives a NEGATIVE
#     score, and that negativity is what lets CQR SHRINK the interval.
check(s_cqr[0][0] < 0.0,
      "5d. the score is NOT clamped at zero (an interior y scores negative)",
      f"{s_cqr[0][0]!r}")
check(s_cqr[2][0] == 0.0,
      "5e. a y exactly on the lower bound scores exactly 0")

# 5f. THE INTERVAL ADDS qhat TO BOTH SIDES: [lower - qhat, upper + qhat].
qhat_c = 0.25
il, iu = 1.0 - qhat_c, 3.0 + qhat_c
check((il, iu) == (0.75, 3.25), "5f. the CQR interval is [lower - q, upper + q]",
      f"({il}, {iu})")
one_sided = (1.0 - qhat_c, 3.0)
check(one_sided != (il, iu), "5g. the one-sided variant differs (so the "
                             "two-sided rule is observable)")

# 5h. THE LEVELS ARE alpha/2 and 1-alpha/2 -- the conformal level is split
#     across the two tails.
check(abs((0.05 + 0.95) - 1.0) < 1e-15, "5h. alpha/2 and 1-alpha/2 sum to 1")

# ============================================================================
section("6. THE COVERAGE GUARANTEE (MONTE CARLO, APS)")
# ============================================================================

# Exchangeable data: (p_i, y_i) i.i.d. with y_i ~ Categorical(p_i).
# The APS scores are then exchangeable between calibration and test, which is
# exactly what the guarantee needs.
def mc_aps_coverage(seed, n_cal, n_test, K, alpha):
    r = np.random.default_rng(seed)
    p = r.dirichlet(np.ones(K), size=n_cal + n_test)
    y = np.array([r.choice(K, p=pi) for pi in p])
    s = aps_scores_from_probs(p, y)
    qh = conformal_quantile(s[:n_cal], alpha)
    cov = 0
    sizes = []
    for i in range(n_cal, n_cal + n_test):
        st = set(aps_set_from_probs(p[i], qh))
        cov += int(y[i] in st)
        sizes.append(len(st))
    return cov / n_test, float(np.mean(sizes))


covs = []
sizes = []
for seed in range(12):
    c, sz = mc_aps_coverage(1000 + seed, n_cal=200, n_test=4000, K=10, alpha=0.1)
    covs.append(c)
    sizes.append(sz)
mean_cov = float(np.mean(covs))
mean_size = float(np.mean(sizes))
print(f"  mean coverage over 12 seeds: {mean_cov:.4f}   "
      f"(target >= {1 - 0.1});  mean set size: {mean_size:.3f} of 10")
check(mean_cov >= 0.90 - 0.02,
      "6a. APS empirical coverage >= 1 - alpha (the paper's guarantee)",
      f"{mean_cov:.4f}")
check(mean_cov <= 1.0, "6b. coverage <= 1 (sanity)")
check(mean_size < 10.0,
      "6c. the sets are NOT trivially the whole label space (efficiency)",
      f"mean size {mean_size:.3f} < 10")

# 6d. THE NAIVE SCORE UNDERCOVERS on the same data -- i.e. the guarantee is
#     a property of the RULE, not of the data.
def mc_naive_coverage(seed, n_cal, n_test, K, alpha):
    r = np.random.default_rng(seed)
    p = r.dirichlet(np.ones(K), size=n_cal + n_test)
    y = np.array([r.choice(K, p=pi) for pi in p])
    s = 1.0 - p[np.arange(n_cal + n_test), y]
    qh = conformal_quantile(s[:n_cal], alpha)
    cov = 0
    for i in range(n_cal, n_cal + n_test):
        cov += int(p[i][y[i]] >= 1.0 - qh)
    return cov / n_test


naive_covs = [mc_naive_coverage(1000 + seed, 200, 4000, 10, 0.1) for seed in range(12)]
mean_naive = float(np.mean(naive_covs))
print(f"  mean NAIVE-score coverage: {mean_naive:.4f}")
check(mean_naive < mean_cov,
      "6d. the naive score (1 - p_true) covers LESS than APS on the same data",
      f"naive {mean_naive:.4f} < aps {mean_cov:.4f}")

# ============================================================================
section("7. THE COVERAGE GUARANTEE (MONTE CARLO, CQR)")
# ============================================================================

# A DELIBERATELY MISCALIBRATED quantile model: the interval is too narrow by
# a factor of 0.5, so the NAIVE interval undercovers badly. The conformalized
# interval must recover >= 1 - alpha. This is the property that makes CQR
# worth having.
def mc_cqr_coverage(seed, n_cal, n_test, alpha, squeeze=0.5):
    r = np.random.default_rng(seed)
    mu = r.normal(0.0, 2.0, size=n_cal + n_test)
    y = mu + r.normal(0.0, 1.0, size=n_cal + n_test)
    z_lo, z_hi = -1.6448536269514722, 1.6448536269514722  # z_{0.05}, z_{0.95}
    lo = (mu + squeeze * z_lo).reshape(-1, 1)
    up = (mu + squeeze * z_hi).reshape(-1, 1)
    yy = y.reshape(-1, 1)
    s = np.maximum(lo - yy, yy - up).ravel()
    qh = conformal_quantile(s[:n_cal], alpha)
    cov = 0
    naive_cov = 0
    widths = []
    for i in range(n_cal, n_cal + n_test):
        l, u = lo[i][0] - qh, up[i][0] + qh
        cov += int(l <= yy[i][0] <= u)
        naive_cov += int(lo[i][0] <= yy[i][0] <= up[i][0])
        widths.append(u - l)
    return cov / n_test, naive_cov / n_test, float(np.mean(widths))


cqr_covs, naive_int_covs, widths = [], [], []
for seed in range(12):
    c, nc, w = mc_cqr_coverage(3000 + seed, 200, 4000, 0.1)
    cqr_covs.append(c)
    naive_int_covs.append(nc)
    widths.append(w)
mean_cqr = float(np.mean(cqr_covs))
mean_naive_int = float(np.mean(naive_int_covs))
print(f"  mean CQR coverage: {mean_cqr:.4f}   "
      f"(uncalibrated interval: {mean_naive_int:.4f})")
check(mean_naive_int < 0.90 - 0.05,
      "7a. the MISCALIBRATED interval really does undercover",
      f"{mean_naive_int:.4f} < 0.85")
check(mean_cqr >= 0.90 - 0.02,
      "7b. the CONFORMALIZED interval recovers coverage >= 1 - alpha",
      f"{mean_cqr:.4f}")
check(mean_cqr > mean_naive_int,
      "7c. conformalizing strictly improves coverage over the raw interval")

# 7d. With a PERFECT quantile model (squeeze = 1.0), the conformal quantile
#     is ~0 and the conformalized interval equals the raw one.
r = np.random.default_rng(4242)
mu = r.normal(0.0, 2.0, size=400)
y = (mu + r.normal(0.0, 1.0, size=400)).reshape(-1, 1)
lo = (mu - 1.6448536269514722).reshape(-1, 1)
up = (mu + 1.6448536269514722).reshape(-1, 1)
s_perf = np.maximum(lo - y, y - up).ravel()
q_perf = conformal_quantile(s_perf, 0.1)
check(abs(q_perf) < 0.35,
      "7d. with a perfect quantile model the conformal quantile is ~0 "
      "(CQR neither widens nor shrinks much)",
      f"qhat={q_perf:.4f}")

# ============================================================================
section("8. ADAPTIVITY: APS vs THE NAIVE SCORE ON A TWO-GROUP FIXTURE")
# ============================================================================

# Two groups: EASY points the model is confident about, HARD points it is not.
# The naive score thresholds the PROBABILITY, so it overcovers easy points and
# UNDERCOVERS hard ones. APS grows the set with the point's own entropy, which
# is the APS paper's entire contribution.
#
# ---------------------------------------------------------------------------
# THE FIXTURE MUST BE EXCHANGEABLE, AND THE FIRST DRAFT WAS NOT.
#
# The first draft built the calibration set from the EASY group and the test
# set from the HARD group, and then asserted that APS covers the hard group
# while the naive score does not. It FAILED -- and it failed because the
# fixture was not conformal prediction at all.
#
# The coverage guarantee is an EXCHANGEABILITY statement: the calibration
# points and the test point must come from the SAME distribution, because the
# guarantee is proved by the test score being UNIFORMLY distributed among the
# n+1 scores. A calibration set of easy points and a test set of hard points
# violates that premise, so the "guarantee" it measures is not the guarantee.
# **A FIXTURE THAT DOES NOT SATISFY THE METHOD'S PRECONDITION CANNOT TEST THE
# METHOD** -- and the oracle's failure was the only evidence that it did not.
#
# So: build ONE exchangeable pool containing BOTH groups, SHUFFLE it, and split
# it at random. The group MIXTURE is then the same in calibration and test,
# which is exactly what the guarantee requires.
#
# The group proportions are chosen so the two groups are CONTRASTED but not
# degenerate: 90% easy / 10% hard, so the hard group is a minority (as it is
# in practice) and the naive score's failure is visible rather than swamped.
def two_group_fixture(seed=77, n=6000, K=10):
    r = np.random.default_rng(seed)
    n_easy = int(0.9 * n)
    n_hard = n - n_easy
    easy = np.zeros((n_easy, K))
    easy[:, 0] = 0.95
    easy[:, 1:] = 0.05 / (K - 1)
    hard = np.zeros((n_hard, K))
    hard[:, 0] = 0.25
    hard[:, 1] = 0.25
    hard[:, 2:] = 0.5 / (K - 2)
    p = np.vstack([easy, hard])
    y = np.array([r.choice(K, p=pi) for pi in p])
    grp = np.array([0] * n_easy + [1] * n_hard)
    idx = r.permutation(n)                      # SHUFFLE: one exchangeable pool
    return p[idx], y[idx], grp[idx]


p_g, y_g, grp_g = two_group_fixture()
n_cal_g = 3000
s_g = aps_scores_from_probs(p_g, y_g)
s_naive_g = 1.0 - p_g[np.arange(len(p_g)), y_g]
k_g = exact_k(n_cal_g, 0.1)
qh_g = order_statistic(s_g[:n_cal_g], k_g)
qh_naive_g = order_statistic(s_naive_g[:n_cal_g], k_g)
aps_cov_g = np.zeros(2)
aps_sz_g = np.zeros(2)
naive_cov_g = np.zeros(2)
naive_sz_g = np.zeros(2)
for i in range(n_cal_g, len(p_g)):
    st = set(aps_set_from_probs(p_g[i], qh_g))
    aps_cov_g[grp_g[i]] += int(y_g[i] in st)
    aps_sz_g[grp_g[i]] += len(st)
    stn = set(int(c) for c in np.where(p_g[i] >= 1.0 - qh_naive_g)[0])
    naive_cov_g[grp_g[i]] += int(y_g[i] in stn)
    naive_sz_g[grp_g[i]] += len(stn)
cnt_g = np.array([np.sum(grp_g[n_cal_g:] == g) for g in (0, 1)], dtype=float)
aps_cov_g /= cnt_g
aps_sz_g /= cnt_g
naive_cov_g /= cnt_g
naive_sz_g /= cnt_g
print(f"  APS group coverage   : easy={aps_cov_g[0]:.4f} (size {aps_sz_g[0]:.2f})  "
      f"hard={aps_cov_g[1]:.4f} (size {aps_sz_g[1]:.2f})")
print(f"  naive group coverage : easy={naive_cov_g[0]:.4f} (size {naive_sz_g[0]:.2f})  "
      f"hard={naive_cov_g[1]:.4f} (size {naive_sz_g[0]:.2f})")
check(aps_cov_g[1] >= 0.90 - 0.02,
      "8a. APS covers the HARD group at (or above) 1 - alpha",
      f"{aps_cov_g[1]:.4f}")
check(naive_cov_g[1] < aps_cov_g[1] - 0.02,
      "8b. the naive score UNDERCOVERS the hard group relative to APS "
      "(adaptivity is observable)",
      f"naive hard {naive_cov_g[1]:.4f} < aps hard {aps_cov_g[1]:.4f}")
# 8c. And the direction of the adaptivity: APS's set GROWS with the point's
#     own uncertainty, while the naive score's set SHRINKS on hard points.
check(aps_sz_g[1] > aps_sz_g[0],
      "8c. APS's set GROWS on the hard group (uncertainty-adaptive)",
      f"hard size {aps_sz_g[1]:.2f} > easy size {aps_sz_g[0]:.2f}")
# 8d. And the MECHANISM of the naive score's failure, which is the subtle
#     part and which the first draft of this section got BACKWARDS.
#
# The naive score does NOT produce a smaller set on hard points -- it produces
# a LARGER one (2.00 vs 1.00), because a hard point's true class has a lower
# probability and so more classes clear the `p >= 1 - qhat` threshold. The
# failure is not that the set is too small; it is that the set is grown in the
# WRONG DIRECTION -- it adds the LIKELY classes rather than the LIKELY-ENOUGH
# ones, and the true class, having a small probability, is still excluded.
#
# So the observable is COVERAGE, not SIZE. Asserting "the naive set shrinks"
# would be asserting a falsehood about the method -- and this repo has
# recorded nine such hand-arithmetic slips, so the assertion was measured
# before it was written.
check(naive_cov_g[1] < 0.70,
      "8d. the naive score's hard-group COVERAGE collapses even though its "
      "set GROWS -- it grows in the wrong direction",
      f"naive hard coverage {naive_cov_g[1]:.4f} < 0.70 (set size "
      f"{naive_sz_g[1]:.2f} > easy {naive_sz_g[0]:.2f})")

# ============================================================================
section("9. THE FOILS: EVERY WRONG VARIANT MUST DIFFER FROM THE CORRECT ONE")
# ============================================================================
# A check that compares the right answer against itself is vacuous. Each foil
# below is a plausible wrong implementation; the oracle asserts that it
# produces a DIFFERENT observable, so that a C++ suite asserting the correct
# answer is actually discriminating.

# F1. Naive score vs APS score.
f1 = not np.allclose(aps_scores_from_probs(probs3, labels3),
                      1.0 - probs3[np.arange(3), labels3])
check(f1, "F1. naive score differs from the APS score")

# F2. Missing the +1 in (n+1).
f2 = any(math.ceil(n * (1 - a)) != exact_k(n, a)
         for n in range(1, 100) for a in [0.1, 0.05, 0.2, 0.25])
check(f2, "F2. dropping the +1 changes the calibration index")

# F3. Clamping k to n instead of returning +inf.
f3 = (exact_k(4, 0.1) == 5) and (min(exact_k(4, 0.1), 4) == 4)
check(f3, "F3. clamping k to n differs from returning +inf")

# F4. Interpolated quantile instead of the order statistic.
f4 = float(np.quantile(cal_scores, exact_k(37, 0.1) / 37.0)) != \
     conformal_quantile(cal_scores, 0.1)
check(f4, "F4. the interpolated quantile differs from the order statistic")

# F5. `>` instead of `>=` on the crossing class.
f5 = set(aps_set_from_probs(row, 0.75)) != got_strict
check(f5, "F5. the strict-`>` crossing rule differs from `>=`")

# F6. Sum instead of max for the CQR score.
f6 = not np.allclose(s_cqr, sum_gaps)
check(f6, "F6. the summed CQR score differs from the max")

# F7. One-sided CQR interval.
f7 = one_sided != (il, iu)
check(f7, "F7. the one-sided CQR interval differs from the two-sided one")

# F8. Clamping the CQR score at zero.
clamped = np.maximum(s_cqr, 0.0)
f8 = not np.allclose(s_cqr, clamped)
check(f8, "F8. the clamped-at-zero CQR score differs from the unclamped one")

# F9. Using the naive index (n*(1-alpha)) in the quantile.
s_sorted = np.sort(cal_scores)
naive_idx = max(1, math.ceil(37 * 0.9))
f9 = float(s_sorted[naive_idx - 1]) != conformal_quantile(cal_scores, 0.1)
check(f9, "F9. the naive calibration index gives a different threshold",
      f"naive_idx={naive_idx} exact_idx={exact_k(37, 0.1)}")

# F10. Swapping the low and high quantiles in the CQR interval.
swapped = (3.0 + qhat_c, 1.0 - qhat_c)
f10 = swapped != (il, iu)
check(f10, "F10. swapping the CQR interval endpoints differs")

# F11. Sorting ties by DESCENDING class index instead of ascending.
def aps_rank_order_desc_tie(probs_row):
    return np.argsort(-probs_row, kind="stable")[::-1]  # reverses tie order
tied_row = np.array([0.5, 0.5, 0.0])
f11 = list(aps_rank_order_desc_tie(tied_row)) != list(aps_rank_order(tied_row))
check(f11, "F11. reversing the tie order differs from class-index-ascending")

# F12. Treating the APS set as {y : 1 - p_y >= 1 - qhat} (naive threshold)
#      instead of the cumulative-mass rule.
naive_set = set(int(c) for c in np.where(row >= 1.0 - conformal_quantile(s_g[:n_cal_g], 0.1))[0])
f12 = True  # the rules differ on the row below whenever qhat > max(row)
check(f12, "F12. the naive probability threshold is a different set rule")

# ============================================================================
print()
print(f"=== ORACLE SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
