// =============================================================================
// tests/test_linear_algebra.cpp -- the PUBLIC symmetric linear algebra:
// eigendecomposition, matrix square root and Frechet distance.
//
// WHAT THIS SUITE EXISTS TO PIN, AND WHY IT IS NOT A DUPLICATE OF
// tests/test_jacobi_eig.cpp
// =============================================================================
//
// `tests/test_jacobi_eig.cpp` pins the EIGENVECTOR PROPERTY of the three
// optimizers' PRIVATE copies of the Jacobi routine, through the PUBLIC
// preconditioner accessors (`get_U_L`, `get_Q_L`, `get_P`). That suite is
// unchanged by this work and still passes unchanged -- it is the
// regression test for the de-duplication.
//
// THIS suite pins the PUBLIC ROUTINE, and it pins three things the
// optimizer suite CANNOT see:
//
//   1. THE EIGENPAIRS ARE SORTED DESCENDING. The optimizer suite asserts
//      that Q's columns are eigenvectors; it never asserts WHICH
//      eigenvectors. And the cyclic Jacobi's final diagonal is NOT
//      globally sorted -- it is the order in which the rotations happened
//      to converge, and for a matrix that is ALREADY DIAGONAL no rotation
//      happens at all, so it is exactly the input's diagonal order.
//
//      **THE FIXTURE MUST BE THE DISCRIMINATING ONE.** A random matrix's
//      Jacobi output happens to be nearly sorted, so it cannot tell a
//      sorted implementation from an unsorted one. The fixture here is
//      `G = diag(1, 5, 1, 1)`, whose Gram matrix `G'G = diag(1, 25, 1, 1)`
//      is ALREADY DIAGONAL: the sweep performs NO rotation, the diagonal
//      stays `1, 25, 1, 1`, and the LARGEST eigenvalue is in the MIDDLE.
//
//   2. THE MATRIX SQUARE ROOT. Nothing in the repo computes one, and the
//      `max(lam, 0)` clamp is load-bearing: a covariance that is PSD only
//      to within rounding has an eigenvalue of -1e-18, and `sqrt` of that
//      is NaN.
//
//   3. THE FRECHET DISTANCE, against the CLOSED FORM for the commuting
//      case, which needs no matrix square root at all. **The identity
//      checks alone (d(X,X)==0, symmetry, non-negativity) CANNOT catch a
//      wrong square root** -- they hold for any symmetric matrix function
//      that returns something PSD. Only the closed form discriminates.
//
// THE REFERENCE IS INDEPENDENT. The 2x2 eigenvalues are computed in this
// file from the characteristic polynomial `l^2 - tr.l + det = 0` using
// `std::sqrt`. No eigensolver from the library under test is used to check
// the library under test.
//
// THE ORACLE. tests/linear_algebra_oracle.py is the float64 numpy
// specification, and it was run BEFORE this file was written. It caught
// SIX defects in its own first draft (see the file's header), including
// the Frechet radicand and the fact that the commuting closed form is for
// d^2 and not d.
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

// The max-abs difference between two matrices, normalised by the max-abs of
// the reference so the comparison is RELATIVE (an absolute tolerance is
// meaningless when the entries are tiny).
static double mat_rel_diff(const Tensor& a, const Tensor& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            num = std::max(num, std::abs(a(i, j) - b(i, j)));
            den = std::max(den, std::abs(b(i, j)));
        }
    }
    return num / std::max(den, 1e-12);
}

// The trace of a square matrix.
static double trace_of(const Tensor& A) {
    double t = 0.0;
    for (size_t i = 0; i < A.rows; ++i) t += A(i, i);
    return t;
}

// ---------------------------------------------------------------------------
// AN INDEPENDENT 2x2 SYMMETRIC EIGENDECOMPOSITION REFERENCE.
//
// Closed form, from the characteristic polynomial l^2 - tr.l + det = 0.
// It shares NO code with the implementation under test -- that is the point.
// ---------------------------------------------------------------------------
struct Eig2 {
    double lambda[2];   // sorted ASCENDING
};

static Eig2 reference_eig2(double a, double b, double d) {
    const double tr = a + d;
    const double det = a * d - b * b;
    const double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));
    Eig2 out;
    out.lambda[0] = 0.5 * (tr - disc);
    out.lambda[1] = 0.5 * (tr + disc);
    return out;
}

// Build a Gram matrix G'G from a dense (n x n) G. PSD BY CONSTRUCTION.
static Tensor gram_of(const std::vector<std::vector<double>>& G) {
    const size_t m = G.size();
    const size_t n = G[0].size();
    Tensor A(m, m);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += G[i][k] * G[j][k];
            A(i, j) = s;
        }
    }
    return A;
}

// The residual of the EIGENVECTOR property: max |A.Q - Q.diag(lam)|.
static double eigen_residual(const Tensor& A, const Tensor& Q,
                             const std::vector<double>& lam) {
    const size_t n = A.rows;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double aq = 0.0;
            for (size_t k = 0; k < n; ++k) aq += A(k, i) * Q(k, j);
            worst = std::max(worst, std::abs(aq - Q(i, j) * lam[j]));
        }
    }
    return worst;
}

// The max-abs entry of a matrix, used as the scale for a relative check.
static double mat_scale(const Tensor& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < A.cols; ++j)
            s = std::max(s, std::abs(A(i, j)));
    return std::max(s, 1e-12);
}

// =============================================================================
int main() {
    printf("=== Public symmetric linear algebra ===\n");

    // =========================================================================
    // 1. THE EIGENPAIR ORDER: DESCENDING IS THE CONTRACT
    //
    // THE DISCRIMINATING FIXTURE. G = diag(1, 5, 1, 1), so
    //     G'G = diag(1, 25, 1, 1)
    // which is ALREADY DIAGONAL. The cyclic sweep performs NO rotation (every
    // off-diagonal is zero), so the final diagonal is exactly the INPUT's
    // order: 1, 25, 1, 1 -- with the LARGEST eigenvalue in the MIDDLE.
    //
    // An implementation that does not sort returns the eigenvalues in that
    // order, so `lam(0)` is 1 and `lam(1)` is 25. A sorted implementation
    // returns 25, 1, 1, 1. This is the assertion that discriminates.
    // =========================================================================
    section("1. THE EIGENPAIRS ARE SORTED DESCENDING (the discriminating fixture)");
    {
        const std::vector<std::vector<double>> G = {{1.0, 0.0, 0.0, 0.0},
                                                     {0.0, 5.0, 0.0, 0.0},
                                                     {0.0, 0.0, 1.0, 0.0},
                                                     {0.0, 0.0, 0.0, 1.0}};
        Tensor A = gram_of(G);

        // -- NON-VACUITY: the fixture must be genuinely DIAGONAL and must have
        //    its dominant eigenvalue NOT in the first slot. --
        bool is_diagonal = true;
        for (size_t i = 0; i < A.rows; ++i)
            for (size_t j = 0; j < A.cols; ++j)
                if (i != j && std::abs(A(i, j)) > 1e-12) is_diagonal = false;
        check(is_diagonal,
              "1.1 the fixture's Gram matrix is ALREADY DIAGONAL (so no rotation happens)");
        check(std::abs(A(1, 1) - 25.0) < 1e-12 && std::abs(A(0, 0) - 1.0) < 1e-12,
              "1.2 the fixture's dominant eigenvalue (25) is in the MIDDLE, not first");

        utils::SymmetricEig eig = utils::symmetric_eigendecompose(A);

        check(eig.eigenvalues.rows == 1 && eig.eigenvalues.cols == 4,
              "1.3 the eigenvalues come back as a (1, n) row");
        check(eig.eigenvectors.rows == 4 && eig.eigenvectors.cols == 4,
              "1.4 the eigenvectors come back as an (n, n) matrix");

        printf("    eigenvalues = [%.12f, %.12f, %.12f, %.12f]\n",
               eig.eigenvalues(0, 0), eig.eigenvalues(0, 1),
               eig.eigenvalues(0, 2), eig.eigenvalues(0, 3));

        // -- THE CONTRACT. --
        check(std::abs(eig.eigenvalues(0, 0) - 25.0) < 1e-9,
              "1.5 the LARGEST eigenvalue comes FIRST (this is what fails without the sort)");
        bool non_increasing = true;
        for (size_t j = 1; j < 4; ++j)
            if (eig.eigenvalues(0, j) > eig.eigenvalues(0, j - 1) + 1e-12)
                non_increasing = false;
        check(non_increasing, "1.6 the eigenvalues are non-increasing (descending order)");

        // -- THE EIGENVECTOR PROPERTY, with the SORTED pairing. --
        std::vector<double> lam(4);
        for (size_t j = 0; j < 4; ++j) lam[j] = eig.eigenvalues(0, j);
        double resid = eigen_residual(A, eig.eigenvectors, lam);
        printf("    residual ||A.Q - Q.Lambda|| = %.6e  (scale %.3e)\n",
               resid, mat_scale(A));
        check(resid / mat_scale(A) < 1e-10,
              "1.7 the columns of Q are eigenvectors, paired with the SORTED eigenvalues");

        // -- ORTHONORMALITY. It held before the sort as well; it is kept
        //    because it separates "wrong eigenvalues" from "not orthogonal". --
        Tensor QtQ = eig.eigenvectors.transpose() * eig.eigenvectors;
        double orth = 0.0;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                orth = std::max(orth, std::abs(QtQ(i, j) - (i == j ? 1.0 : 0.0)));
        check(orth < 1e-12, "1.8 Q is orthogonal");

        // -- THE INPUT IS NOT MUTATED. The routine takes a const reference;
        //    the optimizers' private copy MUTATED its argument and every
        //    call site had to make a working copy by hand. --
        check(std::abs(A(1, 1) - 25.0) < 1e-12 && std::abs(A(0, 1)) < 1e-12,
              "1.9 the routine does NOT mutate the caller's matrix");
    }

    // =========================================================================
    // 2. THE EIGENVECTOR PROPERTY ON A NON-DIAGONAL MATRIX, AGAINST THE
    //    INDEPENDENT 2x2 CLOSED-FORM REFERENCE
    //
    // A diagonal matrix cannot discriminate an eigensolver, so the fixture
    // here has a genuine off-diagonal, and the expected eigenvalues come
    // from `l^2 - tr.l + det = 0` computed in THIS file.
    // =========================================================================
    section("2. THE EIGENVECTOR PROPERTY AND THE EIGENVALUES (2x2, non-diagonal)");
    {
        const std::vector<std::vector<double>> G = {{1.0, 2.0},
                                                     {3.0, 4.0}};
        Tensor A = gram_of(G);
        // A = G'G = [[5, 11], [11, 25]]
        check(std::abs(A(0, 1) - 11.0) < 1e-9 && std::abs(A(0, 0) - 5.0) < 1e-9,
              "2.1 the fixture's Gram matrix is genuinely non-diagonal");

        utils::SymmetricEig eig = utils::symmetric_eigendecompose(A);
        Eig2 ref = reference_eig2(A(0, 0), A(0, 1), A(1, 1));

        // The reference is ASCENDING; the contract is DESCENDING.
        check(rel_err(eig.eigenvalues(0, 0), ref.lambda[1]) < 1e-9,
              "2.2 the larger eigenvalue matches the closed-form reference");
        check(rel_err(eig.eigenvalues(0, 1), ref.lambda[0]) < 1e-9,
              "2.3 the smaller eigenvalue matches the closed-form reference");

        std::vector<double> lam = {eig.eigenvalues(0, 0), eig.eigenvalues(0, 1)};
        double resid = eigen_residual(A, eig.eigenvectors, lam);
        printf("    residual ||A.Q - Q.Lambda|| = %.6e\n", resid);
        check(resid / mat_scale(A) < 1e-10,
              "2.4 the columns of Q are eigenvectors of A");
        check(std::abs(trace_of(A) - (lam[0] + lam[1])) < 1e-9,
              "2.5 the eigenvalues sum to trace(A)");
    }

    // =========================================================================
    // 3. THE MATRIX SQUARE ROOT
    // =========================================================================
    section("3. THE SYMMETRIC MATRIX SQUARE ROOT");
    {
        const std::vector<std::vector<double>> G = {{1.0, 2.0, 0.5},
                                                     {0.3, 1.0, 2.0},
                                                     {1.5, 0.2, 1.0}};
        Tensor A = gram_of(G);
        utils::SymmetricEig eig = utils::symmetric_eigendecompose(A);

        Tensor S = utils::symmetric_matrix_sqrt(A);

        check(S.rows == 3 && S.cols == 3, "3.1 the square root has the same shape as A");

        // S.S == A
        Tensor SS = S * S;
        printf("    max|S.S - A| = %.6e\n", mat_rel_diff(SS, A));
        check(mat_rel_diff(SS, A) < 1e-10, "3.2 S.S == A");

        // S is symmetric
        Tensor St = S.transpose();
        check(mat_rel_diff(S, St) < 1e-12, "3.3 S is symmetric");

        // S is PSD (its eigenvalues are the square roots of A's, hence >= 0)
        utils::SymmetricEig seig = utils::symmetric_eigendecompose(S);
        bool psd = true;
        for (size_t j = 0; j < 3; ++j) if (seig.eigenvalues(0, j) < -1e-12) psd = false;
        check(psd, "3.4 S is positive semidefinite");
        for (size_t j = 0; j < 3; ++j) {
            if (rel_err(seig.eigenvalues(0, j), std::sqrt(std::max(eig.eigenvalues(0, j), 0.0))) > 1e-9) {
                psd = false;
            }
        }
        check(psd, "3.5 S's eigenvalues are the square roots of A's");

        // A RANK-DEFICIENT matrix: the Gram matrix of a (3x2) G is 3x3 with a
        // zero eigenvalue. The square root must exist and be finite.
        const std::vector<std::vector<double>> G2 = {{1.0, 0.0},
                                                      {0.0, 1.0},
                                                      {0.0, 0.0}};
        Tensor A2 = gram_of(G2);
        utils::SymmetricEig e2 = utils::symmetric_eigendecompose(A2);
        check(std::abs(e2.eigenvalues(0, 2)) < 1e-12,
              "3.6 the rank-deficient fixture really has a zero eigenvalue");
        Tensor S2 = utils::symmetric_matrix_sqrt(A2);
        bool finite = true;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                if (!std::isfinite(S2(i, j))) finite = false;
        check(finite, "3.7 the square root of a rank-deficient matrix is FINITE");
        check(mat_rel_diff(S2 * S2, A2) < 1e-10, "3.8 S.S == A for the rank-deficient case");

        // THE CLAMP IS LOAD-BEARING: a matrix with an eigenvalue that is
        // negative to within rounding must still give a FINITE square root
        // rather than a NaN. A covariance that is PSD only to within
        // rounding has such an eigenvalue, and one NaN poisons the whole
        // downstream metric.
        Tensor A3(2, 2);
        A3(0, 0) = 1.0; A3(0, 1) = 0.0;
        A3(1, 0) = 0.0; A3(1, 1) = -1e-17;
        Tensor S3 = utils::symmetric_matrix_sqrt(A3);
        bool finite3 = true;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                if (!std::isfinite(S3(i, j))) finite3 = false;
        check(finite3, "3.9 the max(lam,0) clamp keeps the square root FINITE on a negative eigenvalue");
        check(std::abs(S3(0, 0) - 1.0) < 1e-9 && std::abs(S3(1, 1)) < 1e-9,
              "3.10 the negative eigenvalue contributes a zero, not a NaN");
    }

    // =========================================================================
    // 4. THE FRECHET DISTANCE
    //
    // The identity checks alone CANNOT catch a wrong square root, so the
    // DISCRIMINATING assertion is the CLOSED FORM for the commuting case:
    //
    //     d^2 = ||mu1 - mu2||^2 + sum_i (sqrt(l1_i) - sqrt(l2_i))^2
    //
    // which needs no matrix square root at all.
    // =========================================================================
    section("4. THE FRECHET DISTANCE");
    {
        // 4a. THE COMMUTING (DIAGONAL) CASE, against the closed form.
        Tensor mu1(1, 2); mu1(0, 0) = 1.0; mu1(0, 1) = 2.0;
        Tensor mu2(1, 2); mu2(0, 0) = 0.0; mu2(0, 1) = -1.0;
        Tensor S1(2, 2); S1(0, 0) = 4.0; S1(0, 1) = 0.0; S1(1, 0) = 0.0; S1(1, 1) = 1.0;
        Tensor S2(2, 2); S2(0, 0) = 9.0; S2(0, 1) = 0.0; S2(1, 0) = 0.0; S2(1, 1) = 1.0;

        // The closed form, computed here (no library function involved).
        // NOTE THE DECOMPOSITION: `cov_part` is the COVARIANCE contribution
        // ALONE, and `dmu` is the MEAN contribution alone. The first draft
        // stored only the total `closed_d2 = dmu + cov_part` and then
        // asserted `d3^2 - dmu3 - closed_d2 == 0`, which double-counts the
        // mean: `closed_d2` already contains the OLD mean term. The
        // implementation was RIGHT (d3^2 = 37 = 36 + 1) and the assertion
        // was wrong -- the repo's recurring slip class, caught by deriving
        // the identity in python before asserting it.
        const double l1[2] = {4.0, 1.0};
        const double l2[2] = {9.0, 1.0};
        double dmu = (1.0 - 0.0) * (1.0 - 0.0) + (2.0 - (-1.0)) * (2.0 - (-1.0));
        double cov_part = (std::sqrt(l1[0]) - std::sqrt(l2[0])) * (std::sqrt(l1[0]) - std::sqrt(l2[0]))
                          + (std::sqrt(l1[1]) - std::sqrt(l2[1])) * (std::sqrt(l1[1]) - std::sqrt(l2[1]));
        double closed_d2 = dmu + cov_part;
        double closed_d = std::sqrt(closed_d2);

        double d = utils::frechet_distance(mu1, S1, mu2, S2);
        printf("    commuting d      = %.12f\n", d);
        printf("    closed form d    = %.12f  (d^2 = %.12f)\n", closed_d, closed_d2);
        check(std::abs(d - closed_d) < 1e-9,
              "4.1 the commuting case matches the closed form");

        check(std::abs(utils::frechet_distance(mu1, S1, mu1, S1)) < 1e-12,
              "4.2 d(X, X) == 0");
        check(std::abs(utils::frechet_distance(mu1, S1, mu2, S2)
                     - utils::frechet_distance(mu2, S2, mu1, S1)) < 1e-12,
              "4.3 d is symmetric in its two arguments");
        check(utils::frechet_distance(mu1, S1, mu2, S2) >= 0.0,
              "4.4 d >= 0");

        // 4b. THE MEAN IS SEPARABLE: the mean enters ONLY through
        // ||mu1 - mu2||^2, so shifting one mean changes d^2 by exactly the
        // change in ||mu1 - mu2||^2 -- the covariance term is untouched.
        Tensor mu3(1, 2); mu3(0, 0) = 6.0; mu3(0, 1) = -1.0;
        double d3 = utils::frechet_distance(mu3, S1, mu2, S2);
        double dmu3 = (6.0 - 0.0) * (6.0 - 0.0) + (-1.0 - (-1.0)) * (-1.0 - (-1.0));
        check(std::abs(d3 * d3 - dmu3 - cov_part) < 1e-9,
              "4.5 the mean enters ONLY through ||mu1 - mu2||^2");

        // 4c. THE GENERAL (NON-DIAGONAL) CASE, against a closed form that
        //     still needs NO matrix square root.
        //
        //     **THE FIXTURE COMMUTES WITHOUT BEING DIAGONAL.** Take a
        //     non-trivial orthogonal R and set
        //         S1 = R.diag(4, 1).R'      S2 = R.diag(9, 1).R'
        //     The two commute (they share the eigenbasis R) but neither is
        //     diagonal in the standard basis, so the matrix square root is
        //     genuinely exercised -- a DIAGONAL covariance would make the
        //     whole square-root path degenerate, which is the matrix form of
        //     this repo's "identity transform at integer source coordinates"
        //     trap. And because they commute, the closed form
        //         d^2 = ||mu1-mu2||^2 + sum_i (sqrt(l1_i) - sqrt(l2_i))^2
        //     applies to the EIGENVALUES, which are 4 and 1 -- known in
        //     closed form, with no library function involved.
        //
        //     **A HAND-COPIED 3x3 MATRIX WOULD BE A TRANSCRIPTION HAZARD**
        //     (the repo has shipped nine wrong arXiv IDs and a hand-typed
        //     expected value was wrong in nine separate test slips), so the
        //     fixture is built from a rotation whose entries are computed
        //     here, not written as literals.
        const double ct = std::cos(0.7), st = std::sin(0.7);
        Tensor R(2, 2);
        R(0, 0) = ct; R(0, 1) = -st;
        R(1, 0) = st; R(1, 1) =  ct;
        Tensor D1(2, 2); D1(0, 0) = 4.0; D1(1, 1) = 1.0;
        Tensor D2(2, 2); D2(0, 0) = 9.0; D2(1, 1) = 1.0;
        Tensor C1 = R * D1 * R.transpose();
        Tensor C2 = R * D2 * R.transpose();

        // NON-VACUITY: the covariances must really be NON-DIAGONAL, or the
        // square-root path is degenerate and the whole section measures
        // nothing.
        bool offdiag = std::abs(C1(0, 1)) > 1e-6 && std::abs(C2(0, 1)) > 1e-6;
        check(offdiag,
              "4.6 the commuting covariances are NON-DIAGONAL (the square root is exercised)");
        check(std::abs(trace_of(C1) - 5.0) < 1e-9 && std::abs(trace_of(C2) - 10.0) < 1e-9,
              "4.7 the commuting covariances have the expected traces");

        Tensor mA(1, 2); mA(0, 0) = 1.0; mA(0, 1) = 2.0;
        Tensor mB(1, 2); mB(0, 0) = 0.0; mB(0, 1) = -1.0;
        double dmu_gen = 1.0 + 9.0;   // ||mA - mB||^2
        // The closed form, using the KNOWN eigenvalues 4, 1 and 9, 1.
        double closed_gen_d2 = dmu_gen + (std::sqrt(4.0) - std::sqrt(9.0)) * (std::sqrt(4.0) - std::sqrt(9.0))
                                           + (std::sqrt(1.0) - std::sqrt(1.0)) * (std::sqrt(1.0) - std::sqrt(1.0));
        double d_gen = utils::frechet_distance(mA, C1, mB, C2);
        printf("    general (commuting, non-diagonal) d = %.12f\n", d_gen);
        printf("    closed form d                        = %.12f\n", std::sqrt(closed_gen_d2));
        check(std::abs(d_gen - std::sqrt(closed_gen_d2)) < 1e-9,
              "4.8 the non-diagonal commuting case matches the closed form");

        // 4d. THE DISCRIMINATING FOIL. Two NON-COMMUTING PSD covariances,
        //     where the WRONG PRODUCT ORDER gives a different answer.
        //
        //     The foil is `tr((S1.S2)^(1/2))` instead of
        //     `tr((S1^(1/2).S2.S1^(1/2))^(1/2))`. It is FINITE, SYMMETRIC
        //     and NON-NEGATIVE -- so it cannot be dismissed as a strawman,
        //     and it discriminates the ORDER, which is the part that is
        //     easy to get wrong.
        Tensor P1(3, 3);
        Tensor P2(3, 3);
        double p1[3][3] = {{2.0, 1.0, 0.0}, {1.0, 2.0, 1.0}, {0.0, 1.0, 2.0}};
        double p2[3][3] = {{3.0, 0.5, 0.0}, {0.2, 1.0, 0.8}, {0.0, 0.4, 2.0}};
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j) { P1(i, j) = p1[i][j]; P2(i, j) = p2[i][j]; }
        // Symmetrise to be exactly symmetric, then make them PSD by
        // construction: S = P.P' is PSD BY DEFINITION.
        Tensor NC1 = P1 * P1.transpose();
        Tensor NC2 = P2 * P2.transpose();
        Tensor nA(1, 3); nA(0, 0) = 0.5; nA(0, 1) = -1.0; nA(0, 2) = 0.25;
        Tensor nB(1, 3); nB(0, 0) = -0.5; nB(0, 1) =  0.0; nB(0, 2) = 1.0;

        double d_noncomm = utils::frechet_distance(nA, NC1, nB, NC2);

        // The foil, computed from the PUBLIC matrix square root.
        Tensor N1h = utils::symmetric_matrix_sqrt(NC1);
        Tensor foil_inner = N1h * NC2 * N1h;
        Tensor foil_sqrt = utils::symmetric_matrix_sqrt(foil_inner);
        Tensor foil_alt = utils::symmetric_matrix_sqrt(NC1 * NC2);
        double dmu_nc = 0.0;
        for (size_t k = 0; k < 3; ++k) {
            const double diff = nA(0, k) - nB(0, k);
            dmu_nc += diff * diff;
        }
        double d_foil = std::sqrt(dmu_nc + trace_of(NC1) + trace_of(NC2) - 2.0 * trace_of(foil_alt));

        printf("    non-commuting d        = %.12f\n", d_noncomm);
        printf("    wrong-order foil       = %.12f\n", d_foil);
        check(std::isfinite(d_noncomm), "4.9 the non-commuting distance is FINITE");
        check(std::isfinite(d_foil), "4.10 the foil is FINITE (not a NaN-producing strawman)");
        check(std::abs(d_noncomm - d_foil) > 1e-6,
              "4.11 the non-commuting case DISCRIMINATES the product order");
        check(std::abs(d_noncomm * d_noncomm - (dmu_nc + trace_of(NC1) + trace_of(NC2)
                                                - 2.0 * trace_of(foil_sqrt))) < 1e-9,
              "4.12 the non-commuting case uses tr((S1^(1/2).S2.S1^(1/2))^(1/2))");
    }

    // =========================================================================
    // 5. THE GUARDS. Every message must be DISTINCT -- a substring that
    //    appears in two messages discriminates nothing.
    // =========================================================================
    section("5. THE GUARDS");
    {
        Tensor empty;
        Tensor good(2, 2);
        good.fill(1.0);

        auto throws_with = [](const std::function<void()>& fn, const char* needle) {
            try { fn(); } catch (const std::invalid_argument& e) {
                return std::string(e.what()).find(needle) != std::string::npos;
            } catch (...) { return false; }
            return false;
        };

        check(throws_with([&]() { utils::symmetric_eigendecompose(empty); }, "empty"),
              "5.1 an empty matrix is rejected");
        check(throws_with([&]() { utils::symmetric_eigendecompose(Tensor(2, 3)); }, "square"),
              "5.2 a non-square matrix is rejected");
        check(throws_with([&]() { utils::symmetric_matrix_sqrt(empty); }, "empty"),
              "5.3 the square root rejects an empty matrix");
        check(throws_with([&]() { utils::symmetric_matrix_sqrt(Tensor(2, 3)); }, "square"),
              "5.4 the square root rejects a non-square matrix");

        Tensor mu(1, 2);
        Tensor cov(2, 2);
        cov.fill(1.0);
        check(throws_with([&]() { utils::frechet_distance(mu, cov, Tensor(1, 3), cov); }, "dimension"),
              "5.5 the Frechet distance rejects a mean/covariance dimension mismatch");
        check(throws_with([&]() { utils::frechet_distance(mu, cov, Tensor(1, 2), Tensor(2, 3)); }, "square"),
              "5.6 the Frechet distance rejects a non-square covariance");
        check(throws_with([&]() { utils::frechet_distance(mu, Tensor(2, 3), mu, cov); }, "square"),
              "5.7 the Frechet distance rejects a non-square first covariance");
    }

    // =========================================================================
    // 6. THE VICTIM-FIXTURE CHECK: THE SUITE CAN FAIL
    //
    // A suite that cannot fail proves nothing. This section evaluates the
    // SAME residual metric against a DELIBERATELY WRONG eigenbasis (the
    // columns of the identity on a matrix that is NOT diagonal) and asserts
    // that the residual is LARGE. If the metric were broken -- normalised
    // into always-smallness -- this check would fail.
    // =========================================================================
    section("6. THE RESIDUAL METRIC CAN DETECT A WRONG EIGENBASIS");
    {
        Tensor A(2, 2);
        A(0, 0) = 5.0; A(0, 1) = 4.0; A(1, 0) = 4.0; A(1, 1) = 5.0;
        Tensor Q(2, 2);
        Q(0, 0) = 1.0; Q(1, 0) = 0.0;   // column 0 = e1 (the WRONG basis)
        Q(0, 1) = 0.0; Q(1, 1) = 1.0;   // column 1 = e2
        std::vector<double> lam = {5.0, 5.0};   // the Rayleigh quotients of the wrong basis
        double resid = eigen_residual(A, Q, lam) / mat_scale(A);
        printf("    residual for a WRONG eigenbasis = %.6e\n", resid);
        check(resid > 1e-3,
              "6.1 a wrong eigenbasis scores a LARGE residual (the metric works)");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
