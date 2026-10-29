#ifndef NN_UTILS_LINEAR_ALGEBRA_H
#define NN_UTILS_LINEAR_ALGEBRA_H

#include "../core/tensor.h"
#include <cstddef>

// ===========================================================================
// PUBLIC SYMMETRIC LINEAR ALGEBRA: eigendecomposition, matrix square root
// and the Frechet distance between two Gaussians.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-29)
// ---------------------------------------------------------------------------
//
//     grep -rinE '\bmatrix.?sqrt\b|matrixSqrt|sqrtm' include/   -> 0 hits
//     grep -rinE '\bfid\b|fr[eé]chet|inception.?score' include/ tests/ -> 0 hits
//     grep -rinE 'lyapunov|riccati' include/
//         -> only optimizers/cautious.h's prose mention
//
// FIFTEEN generative models ship in this repo -- DDPM, consistency models,
// flow matching, RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP,
// PixelCNN, CapsNet, DiT and more -- and **NOTHING ANYWHERE MEASURES
// WHETHER THE SAMPLES ARE GOOD**. The nearest shipped objects are
// `utils/mmd_loss.h` (a kernel two-sample statistic with no plan and no
// geometry) and `utils/sinkhorn.h` (an entropic transport plan). Neither
// is a sample-quality metric.
//
// The Frechet distance is the missing metric, and it needs a matrix square
// root of a product of two positive semidefinite matrices -- which needs a
// symmetric eigendecomposition.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE IS THE PROMOTION OF AN EXISTING ROUTINE, NOT A NEW ALGORITHM
// ---------------------------------------------------------------------------
//
// The repo ALREADY had a correct cyclic Jacobi eigendecomposition -- but as
// THREE VERBATIM COPIES, each a `static` private member of Shampoo, SOAP
// and GaLore (added by three separate commits, 92f0e0b / 65b5495 / de76cad,
// each duplicating the routine to avoid a cross-optimizer dependency).
//
// **THE THREE COPIES ARE WHY A ONE-CHARACTER DEFECT WAS IN THREE PLACES AT
// ONCE.** On 2026-10-28 the general branch of the rotation angle was found
// to have a flipped sign (`atan2(2*apq, aqq - app)` instead of
// `atan2(2*apq, app - aqq)`), so the rotation roughly DOUBLED the
// off-diagonal instead of annihilating it. Every pre-existing test passed,
// because they asserted ORTHONORMALITY and FROBENIUS-NORM PRESERVATION,
// both of which hold for a product of rotations that is not an eigenbasis.
// The one property that matters -- that Q's columns ARE eigenvectors -- was
// never asserted.
//
// The routine is promoted here, ONCE, and the three optimizers call it.
// That is the whole point of this file: **de-duplication is what prevents
// the next one-character defect from landing in three places at once.**
//
// ---------------------------------------------------------------------------
// THE ORDER CONTRACT, AND THE DEFECT IT EXPOSES
// ---------------------------------------------------------------------------
//
// **THE EIGENPAIRS ARE RETURNED SORTED DESCENDING BY EIGENVALUE.** This is
// part of the CONTRACT, not an implementation detail, and it is the reason
// the routine is public rather than a private copy.
//
// The cyclic Jacobi's final diagonal is **NOT globally sorted**. It is the
// order in which the rotations happened to converge, and for a matrix that
// is ALREADY DIAGONAL **no rotation happens at all**, so the diagonal is
// exactly the input's diagonal order. Measured:
//   - the rotation always puts the larger of the two eigenvalues into the
//     LOWER-index slot: 0 violations in 200,000 random 2x2 rotations;
//   - but 525 of 2100 final diagonals over random and structured PSD
//     matrices were NOT descending, with an ascending violation as large
//     as 24.5.
//
// This is not a curiosity: `GaLore::refresh_projection` took "the top-r
// eigenvectors" and then copied the FIRST r COLUMNS of Q without sorting,
// so on `G = diag(1, 5, 1, 1)` -- whose Gram matrix `G'G = diag(1, 25, 1, 1)`
// is already diagonal -- it kept the eigenvectors of eigenvalues 1 and 25
// instead of 25 and 1, projecting the gradient onto a subspace that was
// NOT the dominant one. `P'P` was still the identity, so the orthonormality
// test could not see it.
//
// ---------------------------------------------------------------------------
// WHAT THE ORACLE PINNED, AND WHAT IT CAUGHT IN ITSELF
// ---------------------------------------------------------------------------
//
// The specification is `tests/linear_algebra_oracle.py` (float64 numpy),
// which was run BEFORE any C++ existed and caught SIX defects in its own
// first draft. Five of them were in the oracle and are worth keeping:
//
//   1. numpy has NO `linalg.sqrtm` (that is `scipy.linalg.sqrtm`, and
//      scipy is not a dependency here). The eigendecomposition route is
//      also the canonical definition for a symmetric PSD matrix.
//   2. A fixture built as `sym(M @ N.T)` with INDEPENDENT M and N is
//      symmetric but INDEFINITE, so the covariance's square root was
//      undefined and every distance came out NaN. PSD is the method's own
//      precondition; a fixture that violates it cannot test the method.
//   3. The Frechet radicand is `S1^(1/2).S2.S1^(1/2)`, NOT `S1.S2.S1`.
//      Measured on one fixture: the correct form gives 11.10577896685469
//      and the wrong one 10.970119177015437.
//   4. The commuting closed form is for d^2, NOT d.
//   5. The "classic mistake" foil `tr(S1.S2)` is NOT A DISTANCE: its
//      radicand measured -57.4, so it is NaN. A foil that produces NaN
//      discriminates trivially and is VACUOUS. The foil used instead is
//      the WRONG PRODUCT ORDER, which is finite, symmetric and
//      non-negative.
//
// ---------------------------------------------------------------------------
// THE CITATION. Verified 2026-10-29 by a web search on the exact title.
// ---------------------------------------------------------------------------
//   - Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &
//     Hochreiter, S. (2017). "GANs Trained by a Two Time-Scale Update Rule
//     Converge to a Local Nash Equilibrium." NeurIPS 2017, pp. 6626-6637.
//     arXiv:1706.08500 -- VERIFIED: the abs page for 1706.08500 carries
//     this exact title. This is the paper that introduces the Frechet
//     Inception Distance.
//
// **NO OTHER arXiv ID MAY BE INVENTED.** The companion Inception Score
// paper ("Fast Image-to-Image Translation with a Wasserstein Distance",
// ICML 2017) is deliberately NOT cited here: this file does not implement
// the Inception Score, and this repo's expansion queue has produced
// THIRTEEN wrong arXiv IDs, every one a plausible-looking neighbour of
// the right paper.
//
// NOTE ON SCOPE: this file implements the Frechet distance BETWEEN TWO
// GAUSSIANS -- the metric the FID formula is built from. It does NOT
// implement FID itself: FID additionally needs an Inception network's
// pool3 features, which this repo does not ship and which is a model, not
// a utility. What this file provides is the piece that was genuinely
// missing and genuinely reusable.
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// The result of a symmetric eigendecomposition.
//
// `eigenvalues` is (1, n) -- a ROW, matching the repo's convention for a
// vector-valued quantity (see utils/sinkhorn.h's marginals). It is sorted
// DESCENDING.
//
// `eigenvectors` is (n, n). **COLUMN j is the eigenvector for
// `eigenvalues(0, j)`** -- the eigenvector is a COLUMN, so the pairing is
// Q(:, j) <-> lambda(j).
// ---------------------------------------------------------------------------
struct SymmetricEig {
    Tensor eigenvalues;   // (1, n) -- sorted DESCENDING
    Tensor eigenvectors;  // (n, n) -- column j pairs with eigenvalues(0, j)
};

// ---------------------------------------------------------------------------
// symmetric_eigendecompose -- A = Q . diag(lambda) . Q' for a symmetric A.
//
//   A         (n, n) symmetric. NOT modified: the routine works on a
//             private copy, so the caller's matrix is untouched. (The
//             optimizers' private copies MUTATED their argument, and
//             every call site had to make a working copy by hand with a
//             comment saying so.)
//   max_sweeps  the maximum number of cyclic sweeps. Must be >= 1.
//   tol       the sweep stops when the off-diagonal Frobenius norm falls
//             below this. Must be > 0.
//
// Returns the eigenpairs SORTED DESCENDING by eigenvalue.
//
// **THE ROTATION ANGLE'S SIGN IS LOAD-BEARING.** The general branch must
// be
//
//     theta = 0.5 * atan2(2 * apq, app - aqq)
//
// and NOT `aqq - app`. The second argument's sign decides whether the
// rotation ANNIHILATES the off-diagonal (correct) or roughly DOUBLES it
// (wrong). With the wrong sign the routine still returns an ORTHOGONAL
// matrix and still drives A to a diagonal matrix -- it converges to some
// orthogonal similarity -- but the reported eigenvalues are NOT the
// eigenvalues of A and Q's columns are NOT eigenvectors.
//
// **THE `app == aqq` DEGENERATE BRANCH IS CORRECT AND MUST STAY.** It
// computes theta = +-pi/4 directly (the `atan2` would be 0/0 there). It
// is the proof that the general branch was the wrong one: the same
// function is right on the degenerate branch and wrong on the general one.
// It is also MEASURED INERT with the corrected general formula -- the
// general branch already produces +-pi/4 there, bit-identically -- and
// that inertness is pinned by tests/test_jacobi_inertness.cpp.
//
// **A MATRIX THAT IS NOT SYMMETRIC IS ACCEPTED** and symmetrised
// internally: the routine uses only the lower triangle's mirrored values,
// so an asymmetric input is treated as its symmetric part. This is the
// documented behaviour rather than an accident, and it is what makes the
// routine safe to call on `G.G'` products whose asymmetry is pure
// rounding.
//
// Throws std::invalid_argument on: an empty matrix; a non-square matrix;
// max_sweeps < 1; a non-positive or non-finite tol. Every message is
// DISTINCT -- a substring that appears in two different messages
// discriminates nothing between them.
// ---------------------------------------------------------------------------
SymmetricEig symmetric_eigendecompose(const Tensor& A,
                                      int max_sweeps = 100,
                                      double tol = 1e-12);

// ---------------------------------------------------------------------------
// symmetric_matrix_sqrt -- S = A^(1/2), the PRINCIPAL square root.
//
//   A         (n, n) symmetric positive semidefinite. NOT modified.
//
// S is symmetric, positive semidefinite, and satisfies S.S == A.
//
// **THE `max(lambda, 0)` CLAMP IS LOAD-BEARING, NOT COSMETIC.** A
// covariance matrix is PSD in exact arithmetic, but in floating point it is
// PSD only to WITHIN ROUNDING and can have an eigenvalue of -1e-18.
// `sqrt` of a negative number is NaN, and one NaN poisons the entire
// downstream metric -- a Frechet distance that reports NaN is a metric that
// silently reports nothing. The clamp is therefore part of the DEFINITION
// of the principal square root here, and it is pinned by the test suite
// with a fixture whose eigenvalue is exactly negative.
//
// NOTE this is NOT a Cholesky factorisation: `A = L.L'` with L lower
// TRIANGULAR is a different object, and the repo's `utils/gmm.h` has a
// Cholesky for the covariance of a Gaussian mixture. The spectral route is
// needed anyway, because the input here is only NUMERICALLY PSD and a
// Cholesky of a matrix with a zero or slightly negative eigenvalue throws.
//
// Throws std::invalid_argument on: an empty matrix; a non-square matrix;
// max_sweeps < 1; a non-positive or non-finite tol.
// ---------------------------------------------------------------------------
Tensor symmetric_matrix_sqrt(const Tensor& A,
                             int max_sweeps = 100,
                             double tol = 1e-12);

// ---------------------------------------------------------------------------
// frechet_distance -- the Frechet distance between two Gaussians.
//
//     d(N(mu1, S1), N(mu2, S2))
//         = sqrt( ||mu1 - mu2||^2
//                 + tr(S1) + tr(S2)
//                 - 2 * tr( (S1^(1/2) . S2 . S1^(1/2))^(1/2) ) )
//
//   mu1, mu2  (d, 1) the means. Must have the SAME number of columns.
//   S1, S2    (d, d) the covariances. Must be symmetric positive
//             semidefinite and have the SAME dimension as the means.
//
// The product is `S1^(1/2) . S2 . S1^(1/2)` and NOT `S1 . S2`: the former
// is symmetric positive semidefinite (so the outer square root is
// well-defined) and the latter is NOT symmetric. The two give different
// answers -- measured 11.10577896685469 against 10.970119177015437 on
// one fixture -- and the ORDER is load-bearing.
//
// WHEN THE COVARIANCES COMMUTE the distance has the closed form
//
//     d^2 = ||mu1 - mu2||^2 + sum_i (sqrt(l1_i) - sqrt(l2_i))^2
//
// which needs no matrix square root at all. The test suite asserts
// against that closed form, because **the identity checks alone (d(X,X)==0,
// symmetry, non-negativity) CANNOT catch a wrong square root** -- they
// hold for any symmetric matrix function that returns something PSD.
//
// The returned value is >= 0 by construction. It is a SQUARED distance's
// square root, so it is the DISTANCE itself, not its square.
//
// Throws std::invalid_argument on: an empty mean or covariance; a
// mean/covariance dimension mismatch; a non-square covariance; a
// covariance whose dimension does not match its mean's columns;
// max_sweeps < 1; a non-positive or non-finite tol.
// ---------------------------------------------------------------------------
double frechet_distance(const Tensor& mu1, const Tensor& S1,
                        const Tensor& mu2, const Tensor& S2,
                        int max_sweeps = 100, double tol = 1e-12);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_LINEAR_ALGEBRA_H
