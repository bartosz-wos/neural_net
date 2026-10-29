#include "linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// utils/linear_algebra.cpp -- the public symmetric eigendecomposition, the
// symmetric matrix square root and the Frechet distance.
//
// THE SPECIFICATION is tests/linear_algebra_oracle.py (float64 numpy),
// which was run BEFORE this file existed and caught six defects in its own
// first draft. See the header for what those were.
//
// THE ORIGIN. The cyclic Jacobi below was PROMOTED from the three verbatim
// private copies in Shampoo, SOAP and GaLore. The rotation-angle formula
// and the `app == aqq` degenerate branch are unchanged from the corrected
// version (commit 022c490, 2026-10-28) -- they are pinned by
// tests/test_jacobi_eig.cpp and tests/test_jacobi_inertness.cpp, which
// still pass unchanged.
//
// WHAT CHANGED IN THE PROMOTION, and why each change is required:
//
//   1. IT TAKES A `const Tensor&` AND DOES NOT MUTATE IT. The private
//      copies took a non-const reference and destroyed their argument, so
//      every call site had to make a working copy by hand -- each with a
//      comment saying so, which is a comment that exists only because of
//      the API.
//
//   2. IT SORTS THE EIGENPAIRS DESCENDING. The private copies returned the
//      cyclic sweep's diagonal order, which is NOT sorted, and one of the
//      three callers (GaLore) silently relied on the first columns being
//      the dominant ones. The sort is now part of the CONTRACT.
//
//   3. IT SYMMETRISES THE INPUT internally, so a caller with an
//      asymmetric matrix (a `G.G'` product whose asymmetry is pure
//      rounding) gets the symmetric part rather than a silently wrong
//      answer.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// Guards. EVERY message here is DISTINCT. A substring that appears in two
// different messages discriminates nothing between them, and this repo has
// hit that trap several separate times (each time the assertion searched
// for a word the *other* guard's message also contained, so the mutant
// survived).
// ---------------------------------------------------------------------------
void validate_square(const Tensor& A, const char* who) {
    if (A.rows == 0 || A.cols == 0)
        throw std::invalid_argument(std::string(who) + ": matrix is empty");
    if (A.rows != A.cols)
        throw std::invalid_argument(std::string(who) + ": matrix must be square");
}

void validate_sweep_params(int max_sweeps, double tol, const char* who) {
    if (max_sweeps < 1)
        throw std::invalid_argument(std::string(who) + ": max_sweeps must be >= 1");
    if (!(tol > 0.0) || !std::isfinite(tol))
        throw std::invalid_argument(std::string(who) + ": tol must be finite and positive");
}

// The trace of a square matrix.
double trace_of(const Tensor& A) {
    double t = 0.0;
    for (size_t i = 0; i < A.rows; ++i) t += A(i, i);
    return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// symmetric_eigendecompose -- cyclic Jacobi, then a DESCENDING sort.
//
// THE ROTATION, unchanged from the corrected version:
//
//     theta = 0.5 * atan2(2 * apq, app - aqq)
//
// The second argument's sign is load-bearing: it decides whether the
// rotation annihilates the off-diagonal (correct) or roughly doubles it
// (wrong). The `app == aqq` degenerate branch computes +-pi/4 directly,
// because the atan2 would be 0/0 there.
//
// THE SORT. The cyclic sweep's final diagonal is the order in which the
// rotations happened to converge -- NOT a sorted order, and for an
// already-diagonal matrix no rotation happens at all, so it is exactly
// the input's diagonal order. The sort permutes the eigenVECTORS with the
// eigenvalues so the pairing Q(:, j) <-> lambda(j) is preserved; sorting
// the eigenvalues alone would silently break every caller's lockstep
// indexing.
// ---------------------------------------------------------------------------
SymmetricEig symmetric_eigendecompose(const Tensor& A,
                                      int max_sweeps,
                                      double tol) {
    validate_square(A, "symmetric_eigendecompose");
    validate_sweep_params(max_sweeps, tol, "symmetric_eigendecompose");

    const size_t n = A.rows;

    // The working copy. The input is NOT modified.
    //
    // The copy is SYMMETRISED on the way in: only the lower triangle is
    // read, and the mirror is written, so an asymmetric input is treated
    // as its symmetric part. That is the documented behaviour, and it
    // makes the routine safe on a `G.G'` product whose asymmetry is pure
    // floating-point rounding.
    Tensor M(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const double v = 0.5 * (A(i, j) + A(j, i));
            M(i, j) = v;
            M(j, i) = v;
        }
    }

    // Q starts as the identity.
    Tensor Q(n, n);
    for (size_t i = 0; i < n; ++i) Q(i, i) = 1.0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off_norm_sq = 0.0;
        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                off_norm_sq += 2.0 * M(p, q) * M(p, q);
            }
        }
        if (std::sqrt(off_norm_sq) < tol) break;

        for (size_t p = 0; p + 1 < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = M(p, q);
                if (std::abs(apq) < 1e-15) continue;

                const double app = M(p, p);
                const double aqq = M(q, q);

                // THE ROTATION ANGLE. See the header: the sign of the
                // second argument is load-bearing.
                double theta;
                if (std::abs(app - aqq) < 1e-30) {
                    // The degenerate branch: the atan2 would be 0/0.
                    theta = (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
                } else {
                    theta = 0.5 * std::atan2(2.0 * apq, app - aqq);
                }
                const double c = std::cos(theta);
                const double s = std::sin(theta);

                // Apply the rotation: M := J' . M . J.
                const double new_app = c * c * app + 2.0 * c * s * apq + s * s * aqq;
                const double new_aqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
                M(p, p) = new_app;
                M(q, q) = new_aqq;
                M(p, q) = 0.0;
                M(q, p) = 0.0;

                // Off-diagonal entries M(r, p) and M(r, q) for r != p, q.
                for (size_t r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = M(r, p);
                    const double arq = M(r, q);
                    const double new_arp = c * arp + s * arq;
                    const double new_arq = -s * arp + c * arq;
                    M(r, p) = new_arp;
                    M(p, r) = new_arp;
                    M(r, q) = new_arq;
                    M(q, r) = new_arq;
                }

                // Apply the same rotation to Q: Q := Q . J.
                for (size_t r = 0; r < n; ++r) {
                    const double qrp = Q(r, p);
                    const double qrq = Q(r, q);
                    Q(r, p) = c * qrp + s * qrq;
                    Q(r, q) = -s * qrp + c * qrq;
                }
            }
        }
    }

    // THE SORT, DESCENDING. The eigenvalues are collected with their
    // eigenvector INDICES, then both are permuted together -- sorting the
    // eigenvalues alone would break the Q(:, j) <-> lambda(j) pairing
    // that every caller indexes in lockstep.
    std::vector<size_t> order(n);
    std::vector<double> lam(n);
    for (size_t i = 0; i < n; ++i) {
        order[i] = i;
        lam[i] = M(i, i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&lam](size_t a, size_t b) { return lam[a] > lam[b]; });

    SymmetricEig out;
    out.eigenvalues = Tensor(1, n);
    out.eigenvectors = Tensor(n, n);
    for (size_t j = 0; j < n; ++j) {
        out.eigenvalues(0, j) = lam[order[j]];
        for (size_t i = 0; i < n; ++i) {
            out.eigenvectors(i, j) = Q(i, order[j]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// symmetric_matrix_sqrt -- S = A^(1/2) via the eigendecomposition.
//
//     A = Q . diag(lam) . Q'   =>   S = Q . diag(sqrt(max(lam, 0))) . Q'
//
// THE CLAMP IS LOAD-BEARING. A covariance that is PSD only to within
// rounding has an eigenvalue of -1e-18, and sqrt of that is NaN, and one
// NaN poisons the whole downstream metric. See the header.
// ---------------------------------------------------------------------------
Tensor symmetric_matrix_sqrt(const Tensor& A, int max_sweeps, double tol) {
    validate_square(A, "symmetric_matrix_sqrt");
    validate_sweep_params(max_sweeps, tol, "symmetric_matrix_sqrt");

    const size_t n = A.rows;
    SymmetricEig eig = symmetric_eigendecompose(A, max_sweeps, tol);

    // S = sum_j sqrt(max(lam_j, 0)) * q_j q_j'
    //
    // Written as an explicit sum of outer products rather than as two
    // matrix products: it is one pass over the eigenpairs, it cannot
    // pick up an asymmetry from a non-exactly-orthogonal Q, and it makes
    // the clamp's position (on the EIGENVALUE, before the square root)
    // impossible to misread.
    Tensor S(n, n);
    for (size_t j = 0; j < n; ++j) {
        const double root = std::sqrt(std::max(eig.eigenvalues(0, j), 0.0));
        for (size_t i = 0; i < n; ++i) {
            const double qi = eig.eigenvectors(i, j);
            for (size_t k = 0; k < n; ++k) {
                S(i, k) += root * qi * eig.eigenvectors(k, j);
            }
        }
    }

    // Symmetrise the result: the outer-product sum is symmetric in exact
    // arithmetic, but the accumulation order can leave a last-bit
    // asymmetry, and the contract says S is symmetric.
    Tensor Sym(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Sym(i, j) = 0.5 * (S(i, j) + S(j, i));
        }
    }
    return Sym;
}

// ---------------------------------------------------------------------------
// frechet_distance -- the distance between two Gaussians.
//
//     d = sqrt( ||mu1-mu2||^2 + tr(S1) + tr(S2)
//                 - 2 tr( (S1^(1/2) . S2 . S1^(1/2))^(1/2) ) )
//
// THE PRODUCT ORDER IS LOAD-BEARING. `S1^(1/2) . S2 . S1^(1/2)` is
// symmetric positive semidefinite, so the outer square root is
// well-defined; `S1 . S2` is NOT symmetric and gives a different answer.
// Measured: 11.10577896685469 against 10.970119177015437 on one fixture.
//
// The radicand is clamped at zero from below, for the same reason the
// square root clamps its eigenvalues: the arithmetic can push it a
// fraction below zero, and a NaN distance is a metric that reports
// nothing.
// ---------------------------------------------------------------------------
double frechet_distance(const Tensor& mu1, const Tensor& S1,
                        const Tensor& mu2, const Tensor& S2,
                        int max_sweeps, double tol) {
    if (mu1.rows == 0 || mu1.cols == 0)
        throw std::invalid_argument("frechet_distance: the first mean is empty");
    if (mu2.rows == 0 || mu2.cols == 0)
        throw std::invalid_argument("frechet_distance: the second mean is empty");
    validate_square(S1, "frechet_distance: the first covariance");
    validate_square(S2, "frechet_distance: the second covariance");
    validate_sweep_params(max_sweeps, tol, "frechet_distance");

    if (mu1.cols != mu2.cols)
        throw std::invalid_argument(
            "frechet_distance: the two means have different dimensions");
    if (S1.rows != mu1.cols)
        throw std::invalid_argument(
            "frechet_distance: the first covariance's dimension does not match the mean");
    if (S2.rows != mu2.cols)
        throw std::invalid_argument(
            "frechet_distance: the second covariance's dimension does not match the mean");

    // ||mu1 - mu2||^2. The means are treated as vectors of `cols` entries,
    // so a (1, d) and a (d, 1) mean are the same vector.
    double mean_sq = 0.0;
    for (size_t k = 0; k < mu1.cols; ++k) {
        const double a = mu1.rows == 1 ? mu1(0, k) : mu1(k, 0);
        const double b = mu2.rows == 1 ? mu2(0, k) : mu2(k, 0);
        const double diff = a - b;
        mean_sq += diff * diff;
    }

    // The covariance term.
    Tensor S1h = symmetric_matrix_sqrt(S1, max_sweeps, tol);
    Tensor inner = S1h * S2 * S1h;
    Tensor inner_sqrt = symmetric_matrix_sqrt(inner, max_sweeps, tol);

    double radicand = mean_sq + trace_of(S1) + trace_of(S2) - 2.0 * trace_of(inner_sqrt);
    if (radicand < 0.0) radicand = 0.0;

    return std::sqrt(radicand);
}

}  // namespace utils
}  // namespace nn
