// =============================================================================
// tests/test_jacobi_eig.cpp -- THE EIGENVECTOR INVARIANT of the Jacobi
// eigendecomposition used by Shampoo, SOAP and GaLore.
//
// WHAT THIS SUITE EXISTS TO PIN, AND WHY EVERY EXISTING TEST MISSED IT
// =============================================================================
//
// All three optimizers build a Kronecker-factored preconditioner from the
// eigenvalues of a symmetric positive semi-definite matrix:
//
//   * Shampoo : L = G·Gᵀ, R = Gᵀ·G      (shampoo.cpp)
//   * SOAP    : L = G·Gᵀ, R = Gᵀ·G      (soap.cpp)
//   * GaLore  : A = G_EMAᵀ·G_EMA        (galore.cpp)
//
// Each carries its own PRIVATE copy of a cyclic Jacobi rotation routine
// (`jacobi_eigendecompose`), and each needs a symmetric eigendecomposition.
//
// **THE DEFECT.** The general branch of the rotation angle is
//
//     theta = 0.5 * atan2(2·apq, aqq - app)
//
// while the update formulas in the SAME function
//
//     new_app = c²·app + 2cs·apq + s²·aqq
//     new_aqq = s²·app − 2cs·apq + c²·aqq
//
// are the formulas for the rotation that ANNIHILATES `apq`, and the angle
// that annihilates it is `theta = ½·atan2(2·apq, app − aqq)`. The second
// argument of the `atan2` has its SIGN FLIPPED, so the rotation roughly
// DOUBLES the off-diagonal instead of zeroing it.
//
// **THE DECISIVE EVIDENCE IS AN INTERNAL INCONSISTENCY, NOT A CODE READ.**
// The same function contains a DEGENERATE branch for `app == aqq` that
// computes `theta = ±π/4`, and that branch is CORRECT. Measured with a probe
// that computes the post-rotation off-diagonal directly:
//
//     DEGENERATE app==aqq     repo: offdiag_after=+1.554e-16   fix: +1.554e-16
//     GENERAL  app!=aqq       repo: offdiag_after=-1.269e+00   fix: +1.110e-16
//
// A Jacobi rotation that leaves −1.269 where it must leave 0 is not a
// rounding error; it is the wrong branch of `atan2`.
//
// **WHY EVERY EXISTING TEST PASSED.** The existing suites assert that `Q` is
// ORTHONORMAL and that the rotation PRESERVES THE FROBENIUS NORM. Both hold
// for a product of rotations that is not an eigenbasis: `Q` is orthogonal
// whatever the angle, and an orthogonal similarity preserves the norm. The
// one property that matters -- that `Q`'s COLUMNS ARE EIGENVECTORS -- was
// never asserted. Measured on the repo's real source against
// `numpy.linalg.eigvalsh`:
//
//     variant      max eigen error     ‖A·Q − Q·Λ‖
//     repo as-is          2.946              O(1)
//     one-char fix        5.3e-15        2.2e-16
//
// Note that the repo DOES diagonalize `A` (off-diagonal → 1e-16, trace
// preserved to 8.9e-16): it converges to SOME orthogonal similarity, but the
// eigenvalues it reports are not the eigenvalues of `A`.
//
// **THE TEST IS EXERCISED THROUGH THE PUBLIC API**, not by calling the
// private function: `Shampoo::get_U_L`, `SOAP::get_Q_L`, `GaLore::get_P`.
// A test that reached into the private routine would still pass after
// someone moved the routine, and would not pin the observable behaviour.
//
// **THE FIXTURES ARE NON-VACUOUS BY CONSTRUCTION.** A diagonal gradient `G`
// gives a DIAGONAL `L = G·Gᵀ`, whose eigendecomposition is the identity and
// which therefore cannot discriminate a correct implementation from a broken
// one. Every fixture here uses a gradient with a non-zero off-diagonal, and
// each asserts that the resulting `L` really is non-diagonal before making
// any claim about its eigenvectors.
//
// **THE REFERENCE IS INDEPENDENT.** The 2×2 eigenvalues are computed in the
// test from the closed form `λ² − tr·λ + det = 0` using `std::sqrt`. No
// eigensolver from the library under test is used to check the library under
// test.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// Relative error with a floor, so a zero-valued reference does not divide.
static double rel_err(double got, double want) {
    double denom = std::max(std::abs(got), std::abs(want));
    if (denom < 1e-12) return std::abs(got - want);
    return std::abs(got - want) / denom;
}

// ---------------------------------------------------------------------------
// AN INDEPENDENT 2x2 SYMMETRIC EIGENDECOMPOSITION REFERENCE.
//
// Closed form, from the characteristic polynomial λ² − tr·λ + det = 0.
// It shares NO code with the implementation under test -- that is the point.
// ---------------------------------------------------------------------------
struct Eig2 {
    double lambda[2];   // sorted ascending
    double v[2][2];     // v[:, k] is the eigenvector for lambda[k]
};

// Eigenvectors for a 2x2 symmetric matrix A = [[a, b], [b, d]].
//
// For each eigenvalue the eigenvector is (b, λ − a) -- or (λ − d, b) when b
// is zero, in which case the eigenvector is the axis vector. We normalise.
static Eig2 reference_eig2(double a, double b, double d) {
    const double tr = a + d;
    const double det = a * d - b * b;
    const double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));
    Eig2 out;
    out.lambda[0] = 0.5 * (tr - disc);
    out.lambda[1] = 0.5 * (tr + disc);

    for (int k = 0; k < 2; ++k) {
        double x, y;
        if (std::abs(b) > 1e-14) {
            x = b;
            y = out.lambda[k] - a;
        } else if (std::abs(out.lambda[k] - a) > 1e-14) {
            // b == 0: A is diagonal, the eigenvector is an axis vector.
            x = (std::abs(out.lambda[k] - a) < std::abs(out.lambda[k] - d)) ? 1.0 : 0.0;
            y = (x == 1.0) ? 0.0 : 1.0;
        } else {
            x = 1.0; y = 0.0;
        }
        const double n = std::sqrt(x * x + y * y);
        out.v[0][k] = x / n;
        out.v[1][k] = y / n;
    }
    return out;
}

// ---------------------------------------------------------------------------
// The eigenvector property, evaluated on the PUBLIC preconditioner state.
//
// `Q` is (n, n) with EIGENVECTORS AS COLUMNS (the convention the optimizers
// use: `U_L^T · G` rotates INTO the eigenbasis, so column k of Q pairs with
// eigenvalue k). `Lambda` holds the eigenvalues.
//
// The test is:  A·Q == Q·diag(Lambda)
//
// **A NOTE ON THE FIRST DRAFT OF THIS FUNCTION, WHICH CONTAINED A REAL
// BUG AND PRODUCED A FALSE POSITIVE.** The first draft computed the right
// side as `sum_k Q(k,j)·lambda[j]` -- summing over the ROW index k inside a
// loop that was supposed to be picking out the (i,j) entry. That is
// `Q·diag(Lambda)·(ones)`, not `Q·diag(Lambda)`, and it scored a residual of
// 6.363961 on a case that is in fact EXACT: the degenerate branch of the
// implementation produces U = (1,1)/√2 and (1,−1)/√2, which are the true
// eigenvectors of [[5,4],[4,5]] with eigenvalues 9 and 1. The correct
// expression is `(Q·diag(Lambda))[i,j] = Q(i,j)·lambda[j]`.
//
// **THE LESSON, AND IT IS THE ONE THIS REPO KEEPS RELEARNING: a failing
// comparison is evidence about ONE OF TWO PIECES OF CODE YOU JUST WROTE,
// not about the implementation.** The probe that settled it printed U's
// columns and their Rayleigh quotients (9.0000000000 and 1.0000000000),
// which are the eigenvalues -- so the implementation was right and the test
// was wrong. Had the "fix" been applied to the implementation, it would have
// broken a correct branch.
// ---------------------------------------------------------------------------
static double eigenvector_residual(const Tensor& A, const Tensor& Q,
                                   const std::vector<double>& lambda) {
    const size_t n = A.rows;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double aq = 0.0;
            for (size_t k = 0; k < n; ++k) {
                aq += A(k, i) * Q(k, j);
            }
            // (Q · diag(Lambda))[i,j] = Q(i,j) · lambda[j]
            const double ql = Q(i, j) * lambda[j];
            worst = std::max(worst, std::abs(aq - ql));
        }
    }
    return worst;
}

// The scale of the residual, so the comparison can be RELATIVE: an absolute
// tolerance would be meaningless if the matrix entries were tiny.
static double matrix_scale(const Tensor& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < A.cols; ++j)
            s = std::max(s, std::abs(A(i, j)));
    return std::max(s, 1e-12);
}

// Build a Dense layer whose weight gradient is `G` (out_dim x in_dim).
// Returns the layer so the caller can read its state back.
//
// NOTE THE ARGUMENT ORDER: `Dense(in_features, out_features)` -- the
// constructor takes the INPUT size first, and the weights tensor is
// (out_features, in_features). Getting this backwards produces a weight
// matrix of the wrong shape and a silently wrong gradient.
static Dense* make_dense_with_grad(Model& model, size_t out_dim, size_t in_dim,
                                    const std::vector<std::vector<double>>& G) {
    Dense* d = new Dense(in_dim, out_dim);
    model.add_layer(d);
    std::vector<Tensor*> grads = d->gradients();
    grads[0]->fill(0.0);
    for (size_t i = 0; i < G.size() && i < grads[0]->rows; ++i)
        for (size_t j = 0; j < G[i].size() && j < grads[0]->cols; ++j)
            (*grads[0])(i, j) = G[i][j];
    if (grads.size() > 1) grads[1]->fill(0.0);
    return d;
}

// The expected Gram matrix L = G·Gᵀ, computed HERE rather than written as a
// literal.
//
// **A HAND-WRITTEN EXPECTED VALUE IS A KNOWN SOURCE OF SLIPS IN THIS REPO**
// -- a hand-arithmetic typo in a test comment once asserted 6.6875 where the
// rule gives 7.1875, and the implementation was right. The expected value is
// therefore COMPUTED from the same rule the implementation uses, in code, so
// a reader can see which factor belongs where.
static Tensor gram(const std::vector<std::vector<double>>& G) {
    const size_t m = G.size();
    const size_t n = G[0].size();
    Tensor A(m, m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += G[i][k] * G[j][k];
            A(i, j) = s;
        }
    return A;
}

// =============================================================================
int main() {
    printf("=== Jacobi eigendecomposition: the eigenvector invariant ===\n");
    printf("(Shampoo / SOAP / GaLore -- private cyclic Jacobi copies)\n");

    // =========================================================================
    // 1. THE EIGENVECTOR PROPERTY ON A NON-DIAGONAL 2x2 GRAM MATRIX
    //
    // G = [[1, 2], [3, 4]]  =>  L = G·Gᵀ = [[5, 11], [11, 25]]
    //
    // (L[0,1] = 1·3 + 2·4 = 11. The expected value is COMPUTED by `gram()`
    // below rather than written as a literal -- see the note there.)
    //
    // L[0,1] = 11 != 0, so the matrix is genuinely non-diagonal and the
    // eigendecomposition is not the identity.
    // =========================================================================
    section("1. SHAMPOO: A·Q == Q·Λ on a non-diagonal Gram matrix");
    {
        const std::vector<std::vector<double>> G = {{1.0, 2.0}, {3.0, 4.0}};
        Tensor Lexp = gram(G);

        Model model;
        // Dense(in=2, out=2) has weights (2, 2) and bias (1, 2).
        Dense* d = make_dense_with_grad(model, 2, 2, G);

        Shampoo opt(1e-3, 0.0);   // beta=0 => L = G·Gᵀ exactly, no EMA lag
        opt.step(model);

        Tensor L = opt.get_L(d, 0);
        Tensor U = opt.get_U_L(d, 0);

        // -- NON-VACUITY: the fixture must actually be non-diagonal. --
        check(std::abs(L(0, 1)) > 1e-6,
              "1.1 the fixture's L is genuinely non-diagonal (L[0,1] != 0)");
        check(std::abs(L(0, 1) - Lexp(0, 1)) < 1e-9 &&
              std::abs(L(0, 0) - Lexp(0, 0)) < 1e-9 &&
              std::abs(L(1, 1) - Lexp(1, 1)) < 1e-9,
              "1.2 L == G·Gᵀ");

        // -- THE EIGENVECTOR PROPERTY. This is the assertion that fails on
        //    the unfixed code, with a residual of order 1. --
        // The eigenvalues come from the independent closed-form reference.
        Eig2 ref = reference_eig2(L(0, 0), L(0, 1), L(1, 1));

        // The optimizers store the eigenvalues implicitly: `U` is the
        // eigenvector matrix and the eigenvalues are recovered from the
        // Rayleigh quotient of each column, which is exact for an
        // eigenvector and is INDEPENDENT of how the implementation computed
        // them. (Using the Rayleigh quotient avoids depending on any
        // eigenvalue accessor that may not exist.)
        std::vector<double> lambda(2, 0.0);
        for (size_t k = 0; k < 2; ++k) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < 2; ++i) {
                double qi = U(i, k);
                double aqi = L(i, 0) * U(0, k) + L(i, 1) * U(1, k);
                num += qi * aqi;
                den += qi * qi;
            }
            lambda[k] = num / den;
        }

        double resid = eigenvector_residual(L, U, lambda);
        double scale = matrix_scale(L);
        printf("    L = [%g %g; %g %g]\n", L(0, 0), L(0, 1), L(1, 0), L(1, 1));
        printf("    λ = [%.12f, %.12f]   (reference: [%.12f, %.12f])\n",
               lambda[0], lambda[1], ref.lambda[0], ref.lambda[1]);
        printf("    residual ‖A·Q − Q·Λ‖ = %.6e  (scale %.3e)\n", resid, scale);

        check(resid / scale < 1e-10,
              "1.3 the returned Q's columns ARE eigenvectors of L (A·Q == Q·Λ)");
        check(rel_err(std::min(lambda[0], lambda[1]), ref.lambda[0]) < 1e-9 &&
              rel_err(std::max(lambda[0], lambda[1]), ref.lambda[1]) < 1e-9,
              "1.4 the eigenvalues match the closed-form reference");

        // -- ORTHONORMALITY, which the existing suites already checked. It is
        //    kept because it distinguishes "wrong eigenvalues" from "not an
        //    orthogonal matrix": both fail 1.3, only the latter fails this. --
        double orth = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < 2; ++k) s += U(k, i) * U(k, j);
                orth = std::max(orth, std::abs(s - (i == j ? 1.0 : 0.0)));
            }
        check(orth < 1e-12, "1.5 Q is orthogonal (this already held before the fix)");
    }

    // =========================================================================
    // 2. THE SAME PROPERTY ON A 3x3 GRAM MATRIX
    //
    // A 2x2 has only ONE off-diagonal pair, so a single rotation; a 3x3
    // requires several rotations to interact, which is where an inconsistent
    // angle shows up even more strongly.
    // =========================================================================
    section("2. SOAP: A·Q == Q·Λ on a 3x3 Gram matrix");
    {
        Model model;
        Dense* d = make_dense_with_grad(model, 3, 3, {{1.0, 2.0, 0.5},
                                                      {0.3, 1.0, 2.0},
                                                      {1.5, 0.2, 1.0}});
        SOAP opt(1e-3, 0.9, 0.95, 1e-8, 10);
        opt.step(model);

        Tensor L = opt.get_L(d, 0);
        Tensor Q = opt.get_Q_L(d, 0);

        check(std::abs(L(0, 1)) > 1e-6,
              "2.1 the fixture's L is genuinely non-diagonal (L[0,1] != 0)");
        check(std::abs(L(0, 2)) > 1e-6,
              "2.2 L[0,2] != 0 as well (more than one off-diagonal pair)");

        // Rayleigh quotients give the eigenvalues of each eigenvector column.
        const size_t n = 3;
        std::vector<double> lambda(n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double qi = Q(i, k);
                double aqi = 0.0;
                for (size_t j = 0; j < n; ++j) aqi += L(i, j) * Q(j, k);
                num += qi * aqi;
                den += qi * qi;
            }
            lambda[k] = num / den;
        }

        double resid = eigenvector_residual(L, Q, lambda);
        double scale = matrix_scale(L);
        printf("    residual ‖A·Q − Q·Λ‖ = %.6e  (scale %.3e)\n", resid, scale);
        check(resid / scale < 1e-10,
              "2.3 the returned Q's columns ARE eigenvectors of L (A·Q == Q·Λ)");

        // The eigenvalues must also be REAL and distinct enough to be
        // meaningful -- and the trace must be preserved (a similarity
        // transformation preserves the trace, so sum(λ) == trace(A)).
        double tr = 0.0, sum_l = 0.0;
        for (size_t i = 0; i < n; ++i) { tr += L(i, i); sum_l += lambda[i]; }
        check(std::abs(tr - sum_l) / std::max(1.0, std::abs(tr)) < 1e-10,
              "2.4 the eigenvalues sum to trace(L)");

        // The eigenvalues of a PSD matrix are non-negative.
        bool nonneg = true;
        for (size_t k = 0; k < n; ++k) if (lambda[k] < -1e-9) nonneg = false;
        check(nonneg, "2.5 every eigenvalue of a PSD matrix is non-negative");
    }

    // =========================================================================
    // 3. GaLore: THE PROJECTION MATRIX MUST SPAN THE DOMINANT EIGENSPACE
    //
    // GaLore takes the top-r EIGENVECTORS of G_EMAᵀ·G_EMA as its projection
    // P. If the eigenvectors are wrong, P spans the wrong subspace and the
    // projected update is meaningless -- yet Pᵀ·P is still the identity, so
    // the existing orthonormality test cannot see it.
    // =========================================================================
    section("3. GaLore: P spans an eigenspace of G_EMAᵀ·G_EMA");
    {
        Model model;
        // out_dim=4, in_dim=3 => the weight is (4, 3), so the Gram matrix is 3x3.
        Dense* d = make_dense_with_grad(model, 4, 3, {{1.0, 2.0, 0.5},
                                                      {0.3, 1.0, 2.0},
                                                      {1.5, 0.2, 1.0},
                                                      {0.7, 0.9, 1.1}});
        GaLore opt(1e-3, 0.9, 0.999, 1e-8, /*rank=*/3, /*proj_update_interval=*/1);
        opt.step(model);

        const Tensor& P = opt.get_P(d, 0);
        const Tensor& G_ema = opt.get_G_EMA(d, 0);

        // A = G_EMAᵀ · G_EMA, 3x3
        const size_t m = G_ema.rows, n = G_ema.cols;
        Tensor A(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < m; ++k) s += G_ema(k, i) * G_ema(k, j);
                A(i, j) = s;
            }

        check(std::abs(A(0, 1)) > 1e-6,
              "3.1 the Gram matrix is genuinely non-diagonal (A[0,1] != 0)");

        // P's columns must be eigenvectors of A. Because P holds only the top
        // r of the n columns, the property is checked as:
        //     A·P == P·Λ   with Λ diagonal,
        // which holds iff P's columns are eigenvectors.
        //
        // Λ is recovered per column by the Rayleigh quotient, and the
        // residual is normalised by the Rayleigh quotient's own magnitude so
        // that a column that is NOT an eigenvector scores O(1).
        double worst = 0.0;
        for (size_t k = 0; k < P.cols; ++k) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double pi_ = P(i, k);
                double api = 0.0;
                for (size_t j = 0; j < n; ++j) api += A(i, j) * P(j, k);
                num += pi_ * api;
                den += pi_ * pi_;
            }
            const double lam = num / den;
            for (size_t i = 0; i < n; ++i) {
                double api = 0.0;
                for (size_t j = 0; j < n; ++j) api += A(i, j) * P(j, k);
                worst = std::max(worst, std::abs(api - lam * P(i, k)));
            }
        }
        double scale = matrix_scale(A);
        printf("    residual over P's columns = %.6e  (scale %.3e)\n", worst, scale);
        check(worst / scale < 1e-10,
              "3.2 every column of P is an eigenvector of G_EMAᵀ·G_EMA");

        // Orthonormality of P (already checked by the existing suite; kept
        // because it separates the two failure modes).
        Tensor PtP = P.transpose() * P;
        double orth = 0.0;
        for (size_t i = 0; i < PtP.rows; ++i)
            for (size_t j = 0; j < PtP.cols; ++j)
                orth = std::max(orth, std::abs(PtP(i, j) - (i == j ? 1.0 : 0.0)));
        check(orth < 1e-9, "3.3 P's columns are orthonormal (this already held)");
    }

    // =========================================================================
    // 4. THE DEGENERATE BRANCH: app == aqq WITH apq != 0
    //
    // The repo's own code has a special case for `app == aqq` (where the
    // `atan2` would be 0/0) that computes theta = ±π/4. **That branch is
    // CORRECT** and the fix must not disturb it: it is the branch the fix
    // leaves untouched, and it is the branch that proves the general branch
    // was the wrong one.
    //
    // The fixture is a 2x2 with equal diagonal entries and a non-zero
    // off-diagonal: A = [[c, b], [b, c]]. Its eigenvectors are (1,1)/√2 and
    // (1,−1)/√2 with eigenvalues c+b and c−b.
    // =========================================================================
    section("4. THE DEGENERATE BRANCH (app == aqq) STILL ANNIHILATES THE OFF-DIAGONAL");
    {
        // G = [[1, 1], [1, −1]]  =>  L = G·Gᵀ = [[2, 0], [0, 2]] -- diagonal!
        // Not what we want. Instead use G with L = G·Gᵀ having equal
        // diagonal entries: G = [[1, 2], [2, 1]] => G·Gᵀ = [[5,4],[4,5]].
        // L(0,0) == L(1,1) == 5 and L(0,1) == 4 != 0.
        Model model;
        Dense* d = make_dense_with_grad(model, 2, 2, {{1.0, 2.0}, {2.0, 1.0}});
        Shampoo opt(1e-3, 0.0);
        opt.step(model);

        Tensor L = opt.get_L(d, 0);
        Tensor U = opt.get_U_L(d, 0);

        check(std::abs(L(0, 0) - L(1, 1)) < 1e-12,
              "4.1 the fixture has app == aqq (the degenerate branch)");
        check(std::abs(L(0, 1)) > 1e-6,
              "4.2 the fixture's off-diagonal is non-zero");

        Eig2 ref = reference_eig2(L(0, 0), L(0, 1), L(1, 1));
        check(std::abs(ref.lambda[0] - (L(0, 0) - std::abs(L(0, 1)))) < 1e-9 &&
              std::abs(ref.lambda[1] - (L(0, 0) + std::abs(L(0, 1)))) < 1e-9,
              "4.3 the reference eigenvalues are c±b for the equal-diagonal case");

        std::vector<double> lambda(2, 0.0);
        for (size_t k = 0; k < 2; ++k) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < 2; ++i) {
                double qi = U(i, k);
                double aqi = L(i, 0) * U(0, k) + L(i, 1) * U(1, k);
                num += qi * aqi;
                den += qi * qi;
            }
            lambda[k] = num / den;
        }

        double resid = eigenvector_residual(L, U, lambda);
        double scale = matrix_scale(L);
        printf("    residual ‖A·Q − Q·Λ‖ = %.6e  (scale %.3e)\n", resid, scale);
        check(resid / scale < 1e-10,
              "4.4 the degenerate branch's eigenvectors are correct");
        check(rel_err(std::min(lambda[0], lambda[1]), ref.lambda[0]) < 1e-9 &&
              rel_err(std::max(lambda[0], lambda[1]), ref.lambda[1]) < 1e-9,
              "4.5 the degenerate branch's eigenvalues are correct");
    }

    // =========================================================================
    // 5. THE VICTIM-FIXTURE CHECK: THE TEST CAN FAIL
    //
    // A suite that cannot fail proves nothing. This section deliberately
    // evaluates the SAME residual formula against a DELIBERATELY WRONG
    // eigenbasis (the columns of a random orthogonal matrix, which is exactly
    // what a broken implementation returns) and asserts that the residual is
    // LARGE. If the residual metric were broken -- e.g. normalised into
    // always-smallness -- this check would fail.
    // =========================================================================
    section("5. THE RESIDUAL METRIC CAN DETECT A WRONG EIGENBASIS");
    {
        // A = [[5, 4], [4, 5]]; its true eigenvectors are (1,1)/√2 and
        // (1,-1)/√2. Use the WRONG basis (1,0),(0,1) instead.
        Tensor A(2, 2);
        A(0, 0) = 5.0; A(0, 1) = 4.0; A(1, 0) = 4.0; A(1, 1) = 5.0;
        Tensor Q(2, 2);
        Q(0, 0) = 1.0; Q(1, 0) = 0.0;   // column 0 = e1
        Q(0, 1) = 0.0; Q(1, 1) = 1.0;   // column 1 = e2

        // Rayleigh quotients for the wrong basis.
        std::vector<double> lambda(2, 0.0);
        for (size_t k = 0; k < 2; ++k) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < 2; ++i) {
                double qi = Q(i, k);
                double aqi = A(i, 0) * Q(0, k) + A(i, 1) * Q(1, k);
                num += qi * aqi;
                den += qi * qi;
            }
            lambda[k] = num / den;
        }
        double resid = eigenvector_residual(A, Q, lambda) / matrix_scale(A);
        printf("    residual for a WRONG eigenbasis = %.6e\n", resid);
        check(resid > 1e-3,
              "5.1 a wrong eigenbasis scores a LARGE residual (the metric works)");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
