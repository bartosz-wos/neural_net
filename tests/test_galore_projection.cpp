// =============================================================================
// tests/test_galore_projection.cpp -- GaLore's "top-r" CONTRACT.
//
// THE DEFECT THIS SUITE PINS, AND WHY IT WAS INVISIBLE
// =============================================================================
//
// GaLore's header documents its projection matrix as
//
//     P  (n x r) — projection matrix (top-r right singular vectors of G_EMA)
//
// and `GaLore::refresh_projection` says it takes "the top-r eigenvectors
// (the ones with the largest eigenvalues)". The implementation, however,
// NEVER SORTS. It eigendecomposes `G_EMA' * G_EMA` and copies the FIRST r
// COLUMNS of the eigenvector matrix.
//
// **THAT IS ONLY CORRECT IF THE EIGENDECOMPOSITION RETURNS THE EIGENPAIRS IN
// DESCENDING ORDER -- AND THE CYCLIC JACOBI DOES NOT.** The sweep's final
// diagonal is the order in which the rotations happened to converge, and
// for a matrix that is ALREADY DIAGONAL **no rotation happens at all**, so
// the diagonal is exactly the input's diagonal order.
//
// THE DISCRIMINATING FIXTURE.
//
//     G = diag(1, 5, 1, 1)
//
// so `G'G = diag(1, 25, 1, 1)`, which is ALREADY DIAGONAL. The sweep
// performs no rotation and the eigenvalues come out in the INPUT order
// 1, 25, 1, 1 -- with the LARGEST eigenvalue in the MIDDLE.
//
// **THE `rank = 2` CASE DOES NOT DISCRIMINATE.** The broken implementation
// keeps the columns of eigenvalues {1, 25}; the correct one keeps {25, 1}.
// **AS A SET those are THE SAME**, so an assertion comparing the two
// Rayleigh quotients as an unordered pair PASSES on the broken code. That
// is this repo's recorded trap: a fixture whose discriminating value is
// only correct to within a reordering does not discriminate anything.
//
// **THE `rank = 1` CASE DOES.** The broken implementation keeps the FIRST
// column, whose eigenvalue is 1 -- the SMALLEST. The correct
// implementation keeps the column whose eigenvalue is 25 -- the LARGEST.
// So the kept subspace is the dominant one versus a subspace that misses
// the dominant direction entirely.
//
// **WHY NO EXISTING TEST COULD SEE IT.** `tests/test_galore.cpp` asserts
// that P has the right SHAPE and that `P'P` is the identity. Both hold for
// ANY set of orthonormal eigenvectors, including the wrong ones. And
// `tests/test_jacobi_eig.cpp` asserts that P's columns ARE eigenvectors of
// `G_EMA'G_EMA` -- which is ALSO true of the wrong ones. Neither test asks
// WHICH eigenvectors.
//
// **WHAT THE FIX IS.** The shared routine
// `utils::symmetric_eigendecompose` returns the eigenpairs SORTED
// DESCENDING as part of its contract, so "the first r columns" IS "the top
// r" and GaLore needs no change of its own. The sort lives in the shared
// routine precisely so that every caller gets it.
//
// THIS SUITE IS EXERCISED THROUGH THE PUBLIC API (`GaLore::get_P`,
// `GaLore::get_G_EMA`), so it stays valid if the implementation is
// refactored again.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// The Rayleigh quotient of each column of `P` with respect to `A` -- the
// eigenvalue of A that the column corresponds to. For an eigenvector this
// is exact, and it is INDEPENDENT of how the implementation computed the
// eigenvalues.
static std::vector<double> rayleigh_quotients(const Tensor& A, const Tensor& P) {
    const size_t n = A.rows;
    std::vector<double> out;
    for (size_t c = 0; c < P.cols; ++c) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double api = 0.0;
            for (size_t j = 0; j < n; ++j) api += A(i, j) * P(j, c);
            num += P(i, c) * api;
            den += P(i, c) * P(i, c);
        }
        out.push_back(num / den);
    }
    return out;
}

// A = G_EMA' * G_EMA
static Tensor gram_of(const Tensor& G) {
    const size_t m = G.rows, n = G.cols;
    Tensor A(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < m; ++k) s += G(k, i) * G(k, j);
            A(i, j) = s;
        }
    }
    return A;
}

// Build a Dense whose weight gradient is `G`, and step a GaLore optimizer
// with the given rank.
//
// `beta2 = 0` so the gradient EMA is EXACTLY the gradient after one step
// (no lag), which makes the fixture's Gram matrix exactly `G'G`.
static Dense* make_dense_with_grad(Model& model, size_t out_dim, size_t in_dim,
                                   const std::vector<std::vector<double>>& G) {
    Dense* d = new Dense(in_dim, out_dim);
    model.add_layer(d);
    std::vector<Tensor*> grads = d->gradients();
    grads[0]->fill(0.0);
    for (size_t i = 0; i < G.size() && i < grads[0]->rows; ++i) {
        for (size_t j = 0; j < G[i].size() && j < grads[0]->cols; ++j) {
            (*grads[0])(i, j) = G[i][j];
        }
    }
    if (grads.size() > 1) grads[1]->fill(0.0);
    return d;
}

// =============================================================================
int main() {
    printf("=== GaLore: the top-r dominant-eigenspace contract ===\n");

    // =========================================================================
    // 1. rank = 1: THE DISCRIMINATING CASE
    //
    // G = diag(1, 5, 1, 1), so G'G = diag(1, 25, 1, 1) -- already diagonal,
    // with the LARGEST eigenvalue (25) in the MIDDLE.
    //
    // With rank = 1 the broken implementation keeps the FIRST column, whose
    // eigenvalue is 1. The correct implementation keeps the column whose
    // eigenvalue is 25.
    // =========================================================================
    section("1. rank = 1 keeps the DOMINANT eigenvector");
    {
        Model model;
        Dense* d = make_dense_with_grad(model, 4, 4, {{1.0, 0.0, 0.0, 0.0},
                                                       {0.0, 5.0, 0.0, 0.0},
                                                       {0.0, 0.0, 1.0, 0.0},
                                                       {0.0, 0.0, 0.0, 1.0}});
        // beta2 = 0 => G_EMA == G exactly after one step.
        GaLore opt(1e-3, 0.9, /*beta2=*/0.0, 1e-8, /*rank=*/1, /*interval=*/1);
        opt.step(model);

        const Tensor& P = opt.get_P(d, 0);
        const Tensor& Ge = opt.get_G_EMA(d, 0);
        Tensor A = gram_of(Ge);

        // -- NON-VACUITY: the fixture must really be the already-diagonal
        //    Gram matrix with the dominant eigenvalue in the middle. --
        bool is_diagonal = true;
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < A.cols; ++j) {
                if (i != j && std::abs(A(i, j)) > 1e-9) is_diagonal = false;
            }
        }
        check(is_diagonal,
              "1.1 the fixture's Gram matrix is ALREADY DIAGONAL (no rotation happens)");
        check(std::abs(A(1, 1) - 25.0) < 1e-9 && std::abs(A(0, 0) - 1.0) < 1e-9,
              "1.2 the fixture's largest eigenvalue (25) is in the MIDDLE");
        check(std::abs(A(1, 1) - A(0, 0)) > 1.0,
              "1.3 the fixture's eigenvalues are genuinely DISTINCT");

        check(P.rows == 4 && P.cols == 1, "1.4 P has the expected shape (4, 1)");

        std::vector<double> rq = rayleigh_quotients(A, P);
        printf("    kept subspace's Rayleigh quotient = %.9f\n", rq[0]);
        printf("    eigenvalues of the Gram matrix     = 1, 25, 1, 1\n");

        // -- THE CONTRACT. --
        check(std::abs(rq[0] - 25.0) < 1e-6,
              "1.5 the kept subspace is the DOMINANT one (Rayleigh quotient == 25)");

        // -- AND THE ORTHONORMALITY, which the existing suite already checks
        //    and which CANNOT see the defect. --
        Tensor PtP = P.transpose() * P;
        double orth = 0.0;
        for (size_t i = 0; i < PtP.rows; ++i) {
            for (size_t j = 0; j < PtP.cols; ++j) {
                orth = std::max(orth, std::abs(PtP(i, j) - (i == j ? 1.0 : 0.0)));
            }
        }
        check(orth < 1e-9,
              "1.6 P's columns are orthonormal (this already held before the fix)");
    }

    // =========================================================================
    // 2. rank = 2: the kept subspace must span the TWO LARGEST eigenvalues
    //
    // NOTE this case does NOT discriminate on its own -- {1, 25} and {25, 1}
    // are the same set -- so it is asserted as a SORTED pair, and it is the
    // `rank = 1` case above that carries the discriminating power. It is
    // kept because it pins the contract for the general rank.
    // =========================================================================
    section("2. rank = 2 keeps the TWO largest eigenvalues");
    {
        Model model;
        Dense* d = make_dense_with_grad(model, 4, 4, {{1.0, 0.0, 0.0, 0.0},
                                                       {0.0, 5.0, 0.0, 0.0},
                                                       {0.0, 0.0, 1.0, 0.0},
                                                       {0.0, 0.0, 0.0, 1.0}});
        GaLore opt(1e-3, 0.9, /*beta2=*/0.0, 1e-8, /*rank=*/2, /*interval=*/1);
        opt.step(model);

        const Tensor& P = opt.get_P(d, 0);
        const Tensor& Ge = opt.get_G_EMA(d, 0);
        Tensor A = gram_of(Ge);

        check(P.rows == 4 && P.cols == 2, "2.1 P has the expected shape (4, 2)");

        std::vector<double> rq = rayleigh_quotients(A, P);
        std::sort(rq.begin(), rq.end(), std::greater<double>());
        printf("    kept subspace's Rayleigh quotients = %.9f, %.9f\n", rq[0], rq[1]);

        check(std::abs(rq[0] - 25.0) < 1e-6 && std::abs(rq[1] - 1.0) < 1e-6,
              "2.2 the kept subspace spans the two largest eigenvalues {25, 1}");
    }

    // =========================================================================
    // 3. A FIXTURE WITH A NON-DIAGONAL GRAM MATRIX
    //
    // The already-diagonal fixture is the DISCRIMINATING one, but it is
    // degenerate in the sense that the sweep does no work at all. This one
    // has a genuinely non-diagonal Gram matrix, so the sweep really runs --
    // and the contract must still hold.
    // =========================================================================
    section("3. a NON-DIAGONAL Gram matrix also honours the contract");
    {
        // A symmetric positive definite matrix with distinct eigenvalues,
        // whose dominant eigenvector is NOT axis-aligned.
        const double c = std::cos(0.7), s = std::sin(0.7);
        Tensor R(2, 2);
        R(0, 0) = c;  R(0, 1) = -s;
        R(1, 0) = s;  R(1, 1) =  c;
        Tensor D(2, 2); D(0, 0) = 25.0; D(1, 1) = 1.0;
        Tensor A_target = R * D * R.transpose();

        // G = sqrt(A_target) would be needed for G'G == A_target, but the
        // optimizer only needs SOME G whose Gram matrix has the same
        // dominant eigenvector -- and a symmetric positive definite G has
        // G'G = G^2, whose eigenvectors are G's. So use G = A_target
        // directly: G'G = A_target^2, which has the SAME eigenvectors with
        // eigenvalues 625 and 1.
        Model model;
        Dense* d = make_dense_with_grad(model, 2, 2, {{A_target(0, 0), A_target(0, 1)},
                                                       {A_target(1, 0), A_target(1, 1)}});
        GaLore opt(1e-3, 0.9, /*beta2=*/0.0, 1e-8, /*rank=*/1, /*interval=*/1);
        opt.step(model);

        const Tensor& P = opt.get_P(d, 0);
        const Tensor& Ge = opt.get_G_EMA(d, 0);
        Tensor A = gram_of(Ge);

        bool offdiag = std::abs(A(0, 1)) > 1e-6;
        check(offdiag, "3.1 the fixture's Gram matrix is genuinely NON-DIAGONAL");

        std::vector<double> rq = rayleigh_quotients(A, P);
        printf("    kept subspace's Rayleigh quotient = %.9f\n", rq[0]);
        printf("    the largest eigenvalue of the Gram matrix = %.9f\n", 625.0);

        check(std::abs(rq[0] - 625.0) < 1e-3,
              "3.2 the kept subspace is the DOMINANT one");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
