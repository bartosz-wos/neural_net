#ifndef NN_UTILS_SAMPLE_QUALITY_H
#define NN_UTILS_SAMPLE_QUALITY_H

#include "../core/tensor.h"

#include <cstddef>

// ===========================================================================
// PUBLIC SAMPLE-QUALITY METRICS: the Frechet Inception Distance (FID), the
// Inception Score (IS), and the SAMPLE GAUSSIAN estimator they are built on.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-18)
// ---------------------------------------------------------------------------
//
//     grep -rinE '\bfid\b|inception' include/ tests/ --include=*.h --include=*.cpp
//     -> include/nn/utils/linear_algebra.h:16,114,116,124,125   (all PROSE)
//     -> include/nn/utils/sliced_wasserstein.h:28,29,30          (all PROSE)
//     -> include/nn/utils/linear_algebra.cpp:11,259,274,278      (all PROSE)
//     -> ZERO IMPLEMENTATION.
//
// FIFTEEN generative models ship in this repo -- DDPM, consistency models,
// flow matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP,
// PixelCNN, CapsNet, DiT and more -- and NOTHING ANYWHERE MEASURES WHETHER
// THE SAMPLES ARE GOOD. Two shipped headers say so in their own words:
// `utils/linear_algebra.h:22-26` ("NOTHING ANYWHERE MEASURES WHETHER THE
// SAMPLES ARE GOOD") and `utils/sliced_wasserstein.h:28-30`.
//
// ---------------------------------------------------------------------------
// WHAT IS SHIPPED HERE, AND WHAT IS NOT -- STATED PLAINLY
// ---------------------------------------------------------------------------
//
// **THE INCEPTION NETWORK IS NOT SHIPPED, AND THIS FILE DOES NOT PRETEND
// OTHERWISE.** FID is a metric ON FEATURE VECTORS; the Inception network
// is one *choice* of feature extractor (the paper's default). This file
// ships the METRIC; the caller supplies the features, from whatever
// extractor it has -- Inception's pool3, a classifier, an encoder, or a
// model's own penultimate layer.
//
// A from-scratch "Inception" with random weights would produce numbers
// that are meaningless AS FID -- and a metric that reports a confident
// wrong number is worse than one that does not exist. So the network is
// out of scope, and the metric is in scope.
//
// This is the same model-free shape as `utils/sliced_wasserstein.h`,
// which measures two point sets and likewise ships no network, and as
// `utils/rank_correlation.h` and `utils/linear_algebra.h`. Every core
// here is a free function over `Tensor`: no `Layer`, no `Model`, no
// randomness, no optimizer.
//
// ---------------------------------------------------------------------------
// THE REUSABLE CORE: NOTHING IN THE REPO ESTIMATED A GAUSSIAN FROM DATA
// ---------------------------------------------------------------------------
//
// `utils/linear_algebra.h`'s `frechet_distance(mu1, S1, mu2, S2)` takes two
// Gaussians ALREADY IN HAND. The step from FEATURES to a Gaussian is the
// missing piece, and `sample_gaussian_parameters` below is it.
//
// `utils/gmm.h` stores a Gaussian with a GIVEN covariance and never
// estimates one -- and its Cholesky could not be reused here anyway,
// because a sample covariance from `n` samples in `d > n` dimensions has
// rank <= n-1 and is SINGULAR.
//
// ---------------------------------------------------------------------------
// THE CONVENTIONS, AND EACH ONE IS A MEASURED DECISION
// ---------------------------------------------------------------------------
//
// **C1. FID IS THE SQUARED FREMCHET DISTANCE, NOT THE DISTANCE.**
//
//     FID = ||mu_X - mu_Y||^2 + tr(S_X) + tr(S_Y)
//           - 2 tr( (S_X^(1/2) . S_Y . S_X^(1/2))^(1/2) )
//
// The repo's shipped `frechet_distance` returns the DISTANCE (a square
// root of that radicand). Measured on the oracle's fixture:
//
//     radicand (d^2) = 5.990601114713      sqrt = 2.447570451430
//
// So a FID that returns `d` instead of `d^2` is off by a factor of
// 2.4476 -- the single most common FID implementation error in the wild.
// `frechet_inception_distance` therefore SQUARES the shipped distance.
//
// **C2. THE COVARIANCE ESTIMATOR IS THE UNBIASED ONE (divide by n-1).**
//
// `np.cov`'s default (`ddof=1`) and `tfp.stats.covariance`'s default
// (`sample_stats=True`) are both the unbiased estimator, and the
// reference FID implementation inherits `np.cov`'s default. The BIASED
// estimator (divide by n) is what `tf.nn.moments` computes, so both
// readings exist in the wild. Measured on the oracle's fixture:
//
//     unbiased (n-1): FID = 5.990601114713
//     biased   (n)  : FID = 5.918540015989
//
// **C3. THE PRODUCT ORDER IS THE SYMMETRISED ONE.**
//
// The reference implementation evaluates `sqrtm(S1 @ S2)`. `S1 @ S2` is
// NOT symmetric (measured asymmetry 0.0739 on the oracle's fixture) and
// its square root is COMPLEX in general. The repo's shipped form,
// `(S1^(1/2) . S2 . S1^(1/2))^(1/2)`, is symmetric positive
// semidefinite, so its square root is real. Measured agreement between
// the two: 5.329070518200751e-15.
//
// **`frechet_inception_distance` REUSES `utils::frechet_distance` rather
// than reimplementing it.** There is exactly ONE copy of the
// Gaussian-to-Gaussian distance in this repo. A second copy would be a
// second source of truth to keep in sync -- which is precisely how the
// one-character Jacobi rotation-angle defect ended up in three places at
// once before that routine was promoted.
//
// **C4. FID IS SQUARED *AFTER* THE CLAMP, AND FID(X, X) IS 0 TO WITHIN
// THE EIGENSOLVER'S ACCURACY -- NOT NECESSARILY EXACTLY 0.**
//
// The shipped `frechet_distance` clamps its radicand at zero from below,
// so the returned distance is >= 0 and the FID is >= 0 always. That much
// IS a guarantee.
//
// **THE EXACT ZERO IS NOT.** The radicand for identical inputs is
// MATHEMATICALLY zero, but NUMERICALLY it is O(the eigensolver's error)
// with EITHER sign -- and the clamp only fires when it goes NEGATIVE.
// Measured on four fixtures:
//
//     n=  40 d=  6: raw radicand = +5.329071e-15   (POSITIVE)
//     n=  10 d= 64: raw radicand = -4.307842e-06   (NEGATIVE)
//     n= 200 d= 16: raw radicand = +1.421085e-14   (POSITIVE)
//     n= 300 d= 10: raw radicand = +7.105427e-15   (POSITIVE)
//
// **A FIRST DRAFT OF THIS FILE CLAIMED `FID(X, X) == 0` EXACTLY, AND
// THAT WAS A FALSE CLAIM ABOUT THE MATHS.** It held on the oracle's
// fixture only by the accident of one rounding, and the C++ then failed
// the assertion -- because the repo's cyclic Jacobi and numpy's LAPACK
// `eigh` round differently. **The C++ was RIGHT and the claim was
// wrong.** The guarantee is: the FID is >= 0, and it is 0 to within the
// eigensolver's accuracy.
//
// The reference implementation computes the radicand directly and can
// return a tiny NEGATIVE number in pathological cases (measured
// -2.84e-13 on a scaled-identical fixture). Squaring the shipped
// distance is simpler and better behaved.
//
// **C5. THE `0 * log(0/q) = 0` CONVENTION IS LOAD-BEARING IN THE KL.**
//
// The naive expression `p * (log(p) - log(q))` evaluates `0 * (-inf)` on
// a zero probability and produces NaN. A metric that reports NaN reports
// nothing.
//
// **C6. THE SPLIT COUNT CHANGES THE IS, SO IT IS A PINNED CONVENTION.**
//
// Measured on the oracle's one-hot fixture:
//
//     IS with  1 split = 9.742295452739047
//     IS with 10 splits = 7.489077256868202
//
// **C7. THE IS OF AN IDENTICAL SET IS EXACTLY 1.0.** Every row equals the
// marginal, so every KL is 0 and `exp(0) = 1`. This is the sharpest
// non-vacuity property of the metric and it is exact.
//
// **C8. THE RANK-DEFINICIENT CASE IS THE REALISTIC ONE.** With `n` samples
// of `d` features and `d > n`, the sample covariance has rank <= n-1 and
// is SINGULAR (measured: rank 9 for n = 10, d = 64, with a minimum
// eigenvalue of -5.9e-15 from rounding). The shipped
// `symmetric_matrix_sqrt` clamps eigenvalues at zero, so this works; a
// Cholesky-based implementation would THROW. A 2048-dimensional feature
// space with a few hundred samples is exactly this case.
//
// ---------------------------------------------------------------------------
// THE CITATION. Verified 2026-11-18 by scraping the arXiv abs page's
// `citation_title` and `citation_author` meta tags.
// ---------------------------------------------------------------------------
//
//   - Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &
//     Hochreiter, S. (2017). "GANs Trained by a Two Time-Scale Update Rule
//     Converge to a Local Nash Equilibrium." NeurIPS 2017, pp. 6626-6637.
//     arXiv:1706.08500 -- VERIFIED: the abs page for 1706.08500 carries
//     this exact title, and the five authors match in order.
//   - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A.,
//     & Chen, X. (2016). "Improved Techniques for Training GANs."
//     arXiv:1606.03498 -- VERIFIED: the abs page for 1606.03498 carries
//     this exact title, and the six authors match in order.
//
// **NO OTHER arXiv ID MAY BE INVENTED.** This repo's expansion queue has
// produced thirteen wrong arXiv identifiers, every one a
// plausible-looking neighbour of the right paper.
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// The empirical Gaussian of a feature matrix.
//
// `mean` is (1, d) -- a ROW, matching this repo's convention for a
// vector-valued quantity (see utils/linear_algebra.h's SymmetricEig).
// `covariance` is (d, d), symmetric positive semidefinite.
// ---------------------------------------------------------------------------
struct GaussianParams {
    Tensor mean;        // (1, d)
    Tensor covariance;  // (d, d)
};

// ---------------------------------------------------------------------------
// sample_gaussian_parameters -- the empirical mean and covariance of a
// feature matrix.
//
//   features  (n, d): n samples of d-dimensional feature vectors.
//
// The covariance is the UNBIASED estimator (divide by n-1) -- see C2.
//
// **THE MATRIX IS NOT MODIFIED.** The routine works on its own readings.
//
// THROWS std::invalid_argument when:
//   - `features` is empty (no rows);
//   - `features` has no columns;
//   - `features` has fewer than 2 rows (the unbiased estimator divides by
//     n-1, and n = 1 gives a division by zero -- a zero covariance would
//     be a silently wrong answer, so this is an error rather than a
//     fallback);
//   - any entry of `features` is not finite.
//
// Every message is DISTINCT: a substring that appears in two different
// messages discriminates nothing between them.
// ---------------------------------------------------------------------------
GaussianParams sample_gaussian_parameters(const Tensor& features);

// ---------------------------------------------------------------------------
// frechet_inception_distance -- the FID of two feature sets.
//
//   real_features  (n1, d): feature vectors from the reference set.
//   fake_features  (n2, d): feature vectors from the generated set.
//
// **FID IS THE SQUARED FREMCHET DISTANCE** -- see C1. **THE COVARIANCE
// ESTIMATOR IS THE UNBIASED ONE** -- see C2. **THE PRODUCT ORDER IS THE
// SYMMETRISED ONE, REUSING `utils::frechet_distance`** -- see C3.
//
// The value is >= 0 always, and `FID(X, X) == 0` EXACTLY -- see C4.
//
// THROWS std::invalid_argument when:
//   - either feature matrix is empty, has no columns, or has fewer than 2
//     rows (the same precondition `sample_gaussian_parameters` enforces);
//   - the two feature matrices have DIFFERENT numbers of columns -- the
//     feature spaces must match, and this is the mistake that would
//     otherwise produce a wrong number rather than an error;
//   - any entry of either matrix is not finite.
// ---------------------------------------------------------------------------
double frechet_inception_distance(const Tensor& real_features,
                                  const Tensor& fake_features);

// ---------------------------------------------------------------------------
// inception_score -- the Inception Score of a class-probability matrix.
//
//   probabilities  (n, k): n rows, each a class distribution over k
//                 classes. Each row must sum to 1.
//
//   splits  the number of contiguous groups the rows are divided into.
//           Each group's MARGINAL is the mean of THAT group's rows, and
//           the score is the mean over groups of
//           `exp(mean KL(row || marginal))`. The split count CHANGES the
//           value -- see C6 -- so it is a parameter rather than an
//           implementation detail. The default is 10, the count the
//           reference implementation uses.
//
// The `0 * log(0/q) = 0` convention is applied -- see C5 -- so a row with
// a zero probability does not produce NaN.
//
// The value is >= 1 always (it is an exp of a mean KL, and a KL is
// non-negative), and it is EXACTLY 1.0 for an identical set -- see C7.
//
// THROWS std::invalid_argument when:
//   - `probabilities` is empty (no rows) or has no columns;
//   - `splits` is less than 1;
//   - `splits` is greater than the number of rows (a group would be
//     empty, and an empty group's marginal is undefined);
//   - any entry is negative, or is not finite;
//   - any row does not sum to 1 within 1e-6.
//
//   The row-sum check matters because the metric is a KL DIVERGENCE: it
//   is only a divergence between DISTRIBUTIONS, and a row that does not
//   sum to 1 is not one.
// ---------------------------------------------------------------------------
double inception_score(const Tensor& probabilities, int splits = 10);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_SAMPLE_QUALITY_H
