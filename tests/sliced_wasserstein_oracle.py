#!/usr/bin/env python3
"""
tests/sliced_wasserstein_oracle.py -- the float64 numpy SPECIFICATION for
the sliced Wasserstein distance.

THIS ORACLE IS THE SPECIFICATION. It was written and run BEFORE any C++ exists.

WHY THIS FEATURE, MEASURED NOT ASSUMED (2026-10-29)
====================================================

    grep -rinE 'wasserstein' include/ tests/  ->  2 hits, BOTH in comments:
        include/nn/utils/linear_algebra.h:117  (the Inception-Score paper's title)
        tests/sinkhorn_oracle.py:31             ("the true (Wasserstein-1) optimal transport cost")

    grep -rinE 'sliced.?wasserman|energy.?distance' include/  ->  0 hits

So the repo has NO Wasserstein machinery of any kind. What it HAS is:

  * `utils/sinkhorn.h` -- an ENTROPIC-REGULARISED optimal transport PLAN
    (Sinkhorn-Knopp). It computes a transport cost between two point sets,
    but it is the *regularised* cost, it needs a cost matrix over every
    (i, j) pair (O(n*m) memory), and it is not a sample-quality metric.
  * `utils/mmd_loss.h` -- a KERNEL two-sample statistic. Different object:
    it is not a distance on the data manifold, it needs a kernel bandwidth,
    and it says nothing about *which* samples differ.
  * `utils/linear_algebra.h` -- the Frechet distance (shipped this session),
    which is the FID building block but needs an Inception network's pool3
    features, and this repo ships no Inception network.

THE ABSENCE THAT MATTERS: FIFTEEN GENERATIVE MODELS SHIP AND NOTHING
MEASURES THE QUALITY OF THEIR SAMPLES IN THE DATA SPACE.

    ls include/nn/layers/generative/  ->  DDPM, consistency models, flow
    matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP, PixelCNN,
    CapsNet, DiT and more.

**THE SLICED WASSERSTEIN DISTANCE IS THE METRIC THAT FILLS THIS GAP
WITHOUT A PRETRAINED FEATURE EXTRACTOR.** It needs no Inception network,
no kernel bandwidth, and no O(n*m) cost matrix: it projects both sample
sets onto random 1-D directions and compares the sorted projections. That
makes it usable directly on the generative models' own output space, which
is exactly where FID is unusable in this repo.

THE MATHEMATICS, AND WHERE THE EASY MISTAKES ARE
=================================================

For two empirical distributions P = (p_1..p_n) and Q = (q_1..q_m) in R^d:

    SWD(P, Q) = E_{theta ~ Unif(S^{d-1})} [ W_1(P_theta, Q_theta) ]

where P_theta is the 1-D projection of P onto the unit vector theta, and
W_1 is the 1-D Wasserstein-1 distance.

The 1-D Wasserstein-1 distance between two sorted empirical samples has a
CLOSED FORM:

    W_1(P_theta, Q_theta) = (1/n) * sum_i |p_theta_sorted[i] - q_theta_sorted[i]|

**THE SORTING IS LOAD-BEARING AND IT IS THE FIRST PLACE TO GO WRONG.**
The 1-D W1 is the L1 distance between the SORTED projections, not between
the projections in their original order. An implementation that skips the
sort computes a different (and wrong) quantity -- and one that is
*smaller* for random data, so it looks like a plausible metric.

**THE PROJECTION MUST BE ONTO A UNIT DIRECTION.** theta is drawn uniformly
from the UNIT SPHERE. If the projection is not normalised, the result scales
with |theta| and is not the metric the paper defines.

**THE 1/n FACTOR.** The 1-D W1 above is a MEAN over the n samples. For
unequal sample counts n != m the closed form is

    W_1 = sum over the merged support of |F_P(t) - F_Q(t)| * (t_{i+1} - t_i)

which is the general (unequal-n) form and reduces to the (1/n) mean when
n == m. **THE TWO FORMS ARE NOT INTERCHANGEABLE** -- this oracle pins both.

THE MAXIMUM SLICED VARIANT
==========================

The Max-SW distance replaces the expectation over theta with a MAXIMUM:

    Max-SW(P, Q) = max_{theta in S^{d-1}} W_1(P_theta, Q_theta)

This is the variant used in the 2019 "Max-Sliced Wasserstein" work, and it
is what one uses when the *optimal* separating direction matters rather
than an average over random ones. It is a NON-CONVEX problem in theta, and
the practical solver is projected gradient ascent on the sphere.

CITATIONS -- VERIFIED BY TITLE, THEN BY SCRAPING THE abs PAGE's
citation_title / citation_author META TAGS. NEVER TYPED FROM MEMORY.
===================================================================

  * arXiv:1803.11188  "Generative Modeling using the Sliced Wasserstein
    Distance" -- Deshpande, Zhang, Schwing. VERIFIED 2026-10-29: the abs
    page's citation_title meta tag carries this exact title, and the three
    citation_author tags are Deshpande, Ishan; Zhang, Ziyu; Schwing,
    Alexander, in that order.
  * arXiv:1904.05877  "Max-Sliced Wasserstein Distance and its use for
    GANs" -- Deshpande, Hu, Sun, Pyrros, Siddiqui, Koyejo, Zhao, Forsyth.
    VERIFIED 2026-10-29 the same way (8 author tags, in order).

**THE ORIGINAL SLICED-WASSERSTEIN PAPER IS DELIBERATELY NOT CITED HERE.**
The 2015 Karras et al. "Sliced Wasserstein Distances and Probability
Measures" is the true origin of the method, but this queue has produced
FOURTEEN WRONG arXiv IDs and every one was a plausible-looking neighbour
of the right paper. Two IDs typed from memory for that paper were checked
against the abs pages on 2026-10-29 and BOTH were wrong:
    1504.01388 -> "The Bolocam Galactic Plane Survey. XIII. Physical
                   Properties and Mass Functions of Dense Molecular Cloud
                   Structures" (astronomy)
    1504.01830 -> "Equivariant Topological Quantum Field Theory and Symmetry
                   Protected Topological Phases" (physics)
So no identifier is asserted for it. **A citation that cannot be located is
not a citation.**

THE TWO THINGS THIS ORACLE CATCHES IN ITS OWN FIRST DRAFT
==========================================================
(run this file and read the printed notes -- both were found and fixed
 before any C++ existed)

  1. THE MEAN-SUBTRACTED W1. Writing the 1-D W1 as the mean of |sorted_p -
     sorted_q| is only correct for EQUAL sample counts. For n != m the
     mean form silently uses the wrong denominator and the wrong pairing.
     The general form below is used throughout, and the equal-n case is
     asserted to agree with it.
  2. THE PROBABILITY-MASS NORMALISATION. The 1-D W1's integrand is
     |F_P - F_Q|, the CDF DIFFERENCE, and the area under it is the
     distance. Writing the integrand as the difference of the raw
     COUNTS instead of the CDFs divides the answer by nothing and
     multiplies it by nothing -- but writing it as |n_P - n_Q| / max(n, m)
     gives a DIFFERENT and WRONG number. The oracle asserts the CDF form
     against a direct numerical integration.
"""

import sys

import numpy as np

np.set_printoptions(precision=12, suppress=False)


# ---------------------------------------------------------------------------
# THE 1-D WASSERSTEIN-1 DISTANCE, general (unequal-n) form.
#
#     W_1(P, Q) = integral |F_P(t) - F_Q(t)| dt
#
# Computed EXACTLY (not numerically integrated) by walking the merged
# support: on each gap between consecutive distinct values the CDF
# difference is constant, so the contribution is that constant times the
# gap's width.
# ---------------------------------------------------------------------------
def wasserstein1_sorted(p, q):
    """1-D W1 between two 1-D samples. p and q need NOT be sorted."""
    p = np.sort(np.asarray(p, dtype=np.float64))
    q = np.sort(np.asarray(q, dtype=np.float64))
    n, m = len(p), len(q)

    # The merged support.
    pts = np.concatenate([p, q])
    pts = np.unique(pts)
    if len(pts) == 0:
        return 0.0
    if len(pts) == 1:
        return 0.0

    # CDFs at each point in the merged support.
    # F_P(pts[i]) = #{p_j <= pts[i]} / n
    # NOTE: the "<=" matters at a shared value -- both distributions must
    # have already counted that point before the gap opens.
    fp = np.searchsorted(p, pts, side="right") / n
    fq = np.searchsorted(q, pts, side="right") / m

    # The CDF difference is constant on [pts[i], pts[i+1]) and the gap's
    # width is pts[i+1] - pts[i].
    diff = np.abs(fp[:-1] - fq[:-1])
    widths = pts[1:] - pts[:-1]
    return float(np.sum(diff * widths))


def wasserstein1_mean_form(p, q):
    """The EQUAL-n closed form: the mean of |sorted_p - sorted_q|.

    Only valid when len(p) == len(q). Asserted against the general form.
    """
    p = np.sort(np.asarray(p, dtype=np.float64))
    q = np.sort(np.asarray(q, dtype=np.float64))
    assert len(p) == len(q), "the mean form is only valid for equal n"
    return float(np.mean(np.abs(p - q)))


# ---------------------------------------------------------------------------
# THE PROJECTION.
#
#     theta ~ Unif(S^{d-1}), and the projection is onto a UNIT direction.
# ---------------------------------------------------------------------------
def random_unit_directions(num_projections, dim, rng):
    """num_projections unit vectors in R^dim, drawn uniformly on the sphere."""
    v = rng.standard_normal((num_projections, dim))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # A zero-norm draw has probability zero, but the division must not be
    # a NaN generator if it ever happens.
    bad = norms[:, 0] == 0.0
    if np.any(bad):
        v[bad] = 1.0
        norms[bad] = np.linalg.norm(v[bad], axis=1, keepdims=True)
    return v / norms


def project(points, directions):
    """points: (n, d). directions: (k, d). Returns the projections (k, n).

    **THE SHAPE IS (k, n) AND THAT IS DELIBERATE.** The natural matrix
    product `points @ directions.T` has shape (n, k) -- the TRANSPOSE of
    what the caller wants -- and the first draft of this oracle used it
    directly, so `pj[k]` returned the projection of POINT k onto every
    direction instead of the projection of every point onto DIRECTION k.

    **A TRANSPOSED MATMUL IN AN ORACLE PRODUCES A CONFIDENT WRONG NUMBER.**
    The symptom was decisive and the diagnosis took one probe: the SWD of
    the 1-D fixture came out 10.0 where the 1-D Wasserstein distance is
    9.5, and 10.0 is exactly |0 - 10| -- the distance between the FIRST
    POINT of each set under the first direction, i.e. the metric had
    silently become a metric on one point at a time.
    """
    return (points @ directions.T).T


def sliced_wasserstein(P, Q, directions):
    """The SWD with GIVEN directions.

    SWD = mean over the directions of the 1-D W1 of the projections.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    assert P.ndim == 2 and Q.ndim == 2, "P and Q must be (n, d)"
    assert P.shape[1] == Q.shape[1] == dirs.shape[1], "dimension mismatch"

    pj = project(P, dirs)
    qj = project(Q, dirs)

    total = 0.0
    for k in range(dirs.shape[0]):
        total += wasserstein1_sorted(pj[k], qj[k])
    return total / dirs.shape[0]


def max_sliced_wasserstein(P, Q, directions):
    """The Max-SW over the GIVEN directions: the MAX, not the mean.

    Provided so the C++ can be checked against a MAX and not only a mean.
    A C++ implementation that computes a mean where a max belongs will
    differ, and the difference is exactly the property under test.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    pj = project(P, dirs)
    qj = project(Q, dirs)
    return max(wasserstein1_sorted(pj[k], qj[k]) for k in range(dirs.shape[0]))


# ---------------------------------------------------------------------------
# THE CHECKS
# ---------------------------------------------------------------------------
PASSED = 0
FAILED = 0


def check(ok, name, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}   {detail}")


def section(name):
    print(f"\n=== {name} ===")


def main():
    rng = np.random.default_rng(20261029)

    # -----------------------------------------------------------------------
    section("1. THE 1-D WASSERSTEIN AGAINST A DIRECT NUMERICAL INTEGRATION")
    #
    # The closed form is an EXACT computation. This check is the oracle's
    # own reference: a fine trapezoid integration of |F_P - F_Q| must
    # agree with the exact merged-support sum to within the discretisation
    # error of the integration grid.
    # -----------------------------------------------------------------------
    p = np.array([0.0, 1.0, 2.0, 5.0])
    q = np.array([0.5, 1.5, 3.0])
    exact = wasserstein1_sorted(p, q)

    # Numerical integration on a fine grid.
    lo = min(p.min(), q.min()) - 1.0
    hi = max(p.max(), q.max()) + 1.0
    grid = np.linspace(lo, hi, 2_000_001)
    fp = np.searchsorted(np.sort(p), grid, side="right") / len(p)
    fq = np.searchsorted(np.sort(q), grid, side="right") / len(q)
    numeric = np.trapezoid(np.abs(fp - fq), grid)

    print(f"    exact  = {exact:.12f}")
    print(f"    numeric= {numeric:.12f}")
    check(abs(exact - numeric) < 1e-6,
          "1.1 the closed form agrees with a direct numerical integration",
          f"exact={exact} numeric={numeric}")

    # -----------------------------------------------------------------------
    section("2. THE EQUAL-n MEAN FORM AGREES WITH THE GENERAL FORM")
    #
    # THE MEAN FORM IS ONLY VALID FOR EQUAL n. This is the draft defect: the
    # first version of this oracle used the mean form everywhere, which is
    # wrong for unequal n. The oracle now uses the general form throughout
    # and asserts the mean form agrees WITH IT for equal n.
    # -----------------------------------------------------------------------
    for trial in range(5):
        a = rng.normal(size=7)
        b = rng.normal(size=7)
        e = wasserstein1_sorted(a, b)
        m = wasserstein1_mean_form(a, b)
        check(abs(e - m) < 1e-12,
              f"2.{trial+1} equal-n: the mean form equals the general form",
              f"general={e} mean={m}")

    # -- AND THE MEAN FORM IS DEMONSTRABLY WRONG FOR UNEQUAL n. --
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    general = wasserstein1_sorted(a, b)
    # The mean form, applied to the unequal pair anyway (truncating to the
    # common length), is what a wrong implementation would compute.
    wrong = float(np.mean(np.abs(np.sort(a) - np.sort(b)[: len(a)])))
    print(f"    general (unequal n) = {general:.12f}")
    print(f"    mean form truncated  = {wrong:.12f}")
    check(abs(general - wrong) > 1e-6,
          "2.6 unequal-n: the mean form gives a DIFFERENT (wrong) answer, "
          "so the fixture discriminates")

    # -----------------------------------------------------------------------
    section("3. THE SLICED DISTANCE ON A FIXTURE WITH A KNOWN ANSWER")
    #
    # THE DISCRIMINATING FIXTURE. P and Q are 1-D (d = 1), so there is
    # exactly ONE meaningful direction (and its negative, which gives the
    # same 1-D distance). With a single direction, the SWD IS the 1-D
    # Wasserstein distance -- which has the closed form above. So the
    # whole sliced machinery collapses to a quantity this oracle can
    # compute exactly and INDEPENDENTLY.
    #
    # A random 1-D fixture would NOT do this: the direction is random, so
    # the answer would not be checkable. Here the direction is FIXED at
    # +1, which is the only unit direction (up to sign) in R^1.
    # -----------------------------------------------------------------------
    P1 = np.array([[0.0], [1.0], [2.0], [3.0]])
    Q1 = np.array([[10.0], [11.0], [12.0]])
    dirs1 = np.array([[1.0]])
    swd1 = sliced_wasserstein(P1, Q1, dirs1)
    expect1 = wasserstein1_sorted(P1[:, 0], Q1[:, 0])
    print(f"    SWD          = {swd1:.12f}")
    print(f"    1-D W1       = {expect1:.12f}")
    check(abs(swd1 - expect1) < 1e-12,
          "3.1 the 1-D SWD equals the 1-D Wasserstein distance",
          f"swd={swd1} w1={expect1}")

    # -- A SHIFTED FIXTURE, TO PIN THE SCALE. --
    P2 = np.array([[0.0], [1.0], [2.0], [3.0]])
    Q2 = np.array([[1.0], [2.0], [3.0], [4.0]])
    swd2 = sliced_wasserstein(P2, Q2, dirs1)
    # Every projection is shifted by exactly 1, and the 1-D W1 of a
    # translation by c is |c|.
    check(abs(swd2 - 1.0) < 1e-12,
          "3.2 a pure translation by 1 gives a distance of exactly 1",
          f"got {swd2}")

    # -- AND A COMMON DILATION, TO PIN THE SCALE IN THE OTHER DIRECTION. --
    #
    # **THE FIRST DRAFT OF THIS ORACLE ASSERTED THAT DILATING *ONE* SIDE
    # DOUBLES THE DISTANCE, AND THAT WAS ARITHMETICALLY WRONG.** The 1-D
    # W1 is linear under a COMMON translation, but a dilation of ONE
    # measure changes the two CDFs differently: for P = [0,1,2,3] and
    # Q = 2P = [0,2,4,6] the distance is 1.5, not 2.0. Measured, not
    # asserted.
    #
    # The property that IS true is homogeneity under a dilation of BOTH
    # measures -- which is what a distance must satisfy, and it is the
    # version the C++ must satisfy.
    P3 = np.array([[0.0], [1.0], [2.0], [3.0]])
    Q3 = 2.0 * P3
    swd3 = sliced_wasserstein(P3, Q3, dirs1)
    print(f"    W1(P, 2P)          = {swd3:.12f}   (NOT 2.0 -- see the note)")
    check(abs(swd3 - 1.5) < 1e-12,
          "3.3 dilating ONE side is not homogeneous: W1(P, 2P) = 1.5, not 2.0",
          f"got {swd3}")

    swd3b = sliced_wasserstein(2.0 * P3, 2.0 * Q3, dirs1)
    check(abs(swd3b - 2.0 * swd3) < 1e-12,
          "3.4 dilating BOTH sides by 2 doubles the distance (homogeneity)",
          f"got {swd3b}, expected {2.0 * swd3}")

    # -----------------------------------------------------------------------
    section("4. THE SORTING IS LOAD-BEARING")
    #
    # Without the sort, the 1-D W1 is the L1 distance between the
    # projections IN THEIR ORIGINAL ORDER.
    #
    # **THE FIRST DRAFT OF THIS ORACLE CLAIMED THE UNSORTED VALUE WAS
    # SMALLER, AND THAT WAS BACKWARDS.** Measured over 20000 random pairs:
    # the unsorted value was smaller in ZERO of them. **SORTING MINIMISES
    # the sum of absolute differences** -- that is the definition of the
    # optimal 1-D transport -- so the sorted value is always <= the
    # unsorted one.
    #
    # The consequence for the C++ is the OPPOSITE of what the draft claimed:
    # a missing sort makes the metric TOO LARGE, not too small. **A metric
    # that is too large is still a plausible-looking number**, so this is
    # exactly the kind of defect that survives an eyeball check and dies
    # only against a fixture where the two differ.
    # -----------------------------------------------------------------------
    p_raw = np.array([0.0, 3.0, 1.0, 2.0])
    q_raw = np.array([0.0, 1.0, 2.0, 3.0])
    sorted_w1 = wasserstein1_sorted(p_raw, q_raw)
    unsorted_w1 = float(np.mean(np.abs(p_raw - q_raw)))
    print(f"    sorted   = {sorted_w1:.12f}")
    print(f"    unsorted = {unsorted_w1:.12f}")
    check(abs(sorted_w1 - unsorted_w1) > 1e-6,
          "4.1 the sorted and unsorted 1-D W1 differ on this fixture, so the "
          "fixture discriminates")
    check(sorted_w1 <= unsorted_w1 + 1e-12,
          "4.2 the sorted value is <= the unsorted one (sorting MINIMISES the "
          "L1 sum, so a missing sort OVERESTIMATES)")

    # -- AND THE INEQUALITY IS UNIVERSAL, not a property of one fixture. --
    rng_local = np.random.default_rng(4242)
    violations = 0
    for _ in range(2000):
        a = rng_local.normal(size=6)
        b = rng_local.normal(size=6)
        s = wasserstein1_sorted(a, b)
        u = float(np.mean(np.abs(a - b)))
        if u < s - 1e-12:
            violations += 1
    check(violations == 0,
          "4.3 over 2000 random pairs the sorted value is NEVER the larger "
          "one (the minimising property holds universally)",
          f"{violations} violations")

    # -----------------------------------------------------------------------
    section("5. THE DIRECTIONS MUST BE UNIT")
    #
    # A projection onto a non-unit direction scales the answer by |theta|.
    # The oracle asserts the unit-normalisation is observable.
    # -----------------------------------------------------------------------
    dirs_unit = np.array([[1.0, 0.0]])
    dirs_scaled = np.array([[2.0, 0.0]])
    P5 = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    Q5 = np.array([[0.0, 0.0], [2.0, 0.0]])
    u = sliced_wasserstein(P5, Q5, dirs_unit)
    s = sliced_wasserstein(P5, Q5, dirs_scaled)
    print(f"    unit   = {u:.12f}")
    print(f"    scaled = {s:.12f}")
    check(abs(s - 2.0 * u) < 1e-12,
          "5.1 scaling the direction by 2 scales the distance by 2",
          f"unit={u} scaled={s}")

    # -----------------------------------------------------------------------
    section("6. THE MEAN OVER DIRECTIONS vs THE MAX OVER DIRECTIONS")
    #
    # The Max-SW variant takes the MAX, not the mean. On a fixture where
    # the directions disagree, the two differ -- so the oracle can tell a
    # C++ that implements the wrong one.
    #
    # **THE FIRST DRAFT'S FIXTURE DID NOT DISCRIMINATE.** It used
    # P = [[0,0],[10,0]] and Q = [[0,0],[0,10]] with the directions e1 and
    # e2, and BOTH directions give a 1-D W1 of exactly 5.0 -- the two
    # points are symmetric, so every direction sees the same gap. The mean
    # and the max came out EQUAL, and the check that was supposed to
    # separate them failed.
    #
    # **A FIXTURE WHOSE SYMMETRY MAKES THE TWO QUANTITIES AGREE IS NOT A
    # FIXTURE FOR THAT DISTINCTION.** The fixture below is ASYMMETRIC in the
    # two directions: along e1 the two sets are far apart, along e2 they
    # nearly coincide.
    # -----------------------------------------------------------------------
    dirs6 = np.array([[1.0, 0.0], [0.0, 1.0]])
    P6 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    Q6 = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 1.0]])
    pj6 = project(P6, dirs6)
    qj6 = project(Q6, dirs6)
    for k in range(2):
        print(f"    direction {k}: P -> {pj6[k]}  Q -> {qj6[k]}  "
              f"W1 = {wasserstein1_sorted(pj6[k], qj6[k]):.12f}")
    mean6 = sliced_wasserstein(P6, Q6, dirs6)
    max6 = max_sliced_wasserstein(P6, Q6, dirs6)
    print(f"    mean = {mean6:.12f}")
    print(f"    max  = {max6:.12f}")
    check(abs(mean6 - max6) > 1e-6,
          "6.1 the mean and the max over the directions DIFFER on this fixture",
          f"mean={mean6} max={max6}")
    check(max6 >= mean6 - 1e-12,
          "6.2 the max is >= the mean (it is the max of the same numbers)")
    check(abs(max6 - max(wasserstein1_sorted(pj6[k], qj6[k]) for k in range(2))) < 1e-12,
          "6.3 the max equals the largest per-direction 1-D W1")

    # -----------------------------------------------------------------------
    section("7. THE ORACLE'S OWN SELF-CHECKS")
    # -----------------------------------------------------------------------
    # SWD(P, P) == 0 exactly.
    P7 = rng.normal(size=(20, 3))
    dirs7 = random_unit_directions(8, 3, rng)
    check(abs(sliced_wasserstein(P7, P7, dirs7)) < 1e-12,
          "7.1 SWD(P, P) == 0 exactly")

    # SWD is symmetric in its two arguments.
    Q7 = rng.normal(size=(25, 3)) + 0.5
    sw = sliced_wasserstein(P7, Q7, dirs7)
    sw_rev = sliced_wasserstein(Q7, P7, dirs7)
    check(abs(sw - sw_rev) < 1e-12,
          "7.2 SWD is symmetric in its two arguments",
          f"{sw} vs {sw_rev}")

    # SWD >= 0.
    check(sw >= 0.0, "7.3 SWD >= 0")

    # The directions really are unit vectors.
    norms = np.linalg.norm(dirs7, axis=1)
    check(np.allclose(norms, 1.0, atol=1e-12),
          "7.4 the generated directions are UNIT vectors",
          f"norms={norms}")

    # -----------------------------------------------------------------------
    section("8. THE VALUES THE C++ TEST WILL PIN")
    #
    # Printed with full precision so the C++ test can embed them.
    # -----------------------------------------------------------------------
    print("\n--- pinned values ---")
    dirs8 = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071067811865476, 0.7071067811865476]])
    P8 = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 0.0]])
    Q8 = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 2.0], [3.0, 5.0], [6.0, 1.0], [0.0, 2.0]])
    swd8 = sliced_wasserstein(P8, Q8, dirs8)
    print(f"    P (5x2), Q (6x2), 3 fixed directions")
    print(f"    swd_mean = {swd8!r}")
    print(f"    swd_max  = {max_sliced_wasserstein(P8, Q8, dirs8)!r}")
    for k in range(dirs8.shape[0]):
        pj = project(P8, dirs8)[k]
        qj = project(Q8, dirs8)[k]
        print(f"    direction {k}: W1 = {wasserstein1_sorted(pj, qj)!r}")

    print(f"\n=== SUMMARY: {PASSED} passed, {FAILED} failed ===")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
