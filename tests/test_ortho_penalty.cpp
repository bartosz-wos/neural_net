// Test suite for ortho_penalty: orthogonality penalties (StyleGAN2 Sec. 3,
// Karlin et al. arXiv:1912.04958).
//
// Every expected value here is transcribed from an INDEPENDENT float64 numpy
// oracle (ortho_oracle.py) that verifies its own analytic adjoint against
// centred finite differences BEFORE emitting a single number. Three bugs in
// that oracle's derivation were found and fixed exactly this way; see the
// header and docs/plans/2026-10-17-orthogonality-penalties.md.

#include "../include/nn/utils/ortho_penalty.h"
#include "../include/nn/utils/losses.h"
#include "../include/nn/core/model.h"
#include "../include/nn/activations/activations.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static std::string g_test;

static void check(bool cond, const std::string& what) {
    if (cond) {
        ++g_pass;
        std::printf("  [PASS] %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

static void check_close(double got, double want, double tol, const std::string& what) {
    double d = std::fabs(got - want);
    if (d <= tol) {
        ++g_pass;
        std::printf("  [PASS] %s (got %.17g, want %.17g, |d|=%.3e)\n",
                    what.c_str(), got, want, d);
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s (got %.17g, want %.17g, |d|=%.3e > tol %.1e)\n",
                    what.c_str(), got, want, d, tol);
    }
}

// Relative-error check with a FLOOR, so it never divides by ~1e-17 (the
// TDD skill's stochastic/noise-floor lesson: rel_err ~ 1 at the floor is noise).
static void check_rel(const std::vector<double>& got, const std::vector<double>& want,
                      double tol, const std::string& what) {
    if (got.size() != want.size()) {
        ++g_fail;
        std::printf("  [FAIL] %s (size mismatch %zu vs %zu)\n", what.c_str(),
                    got.size(), want.size());
        return;
    }
    double worst = 0.0, scale = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        scale = std::fmax(scale, std::fabs(want[i]));
        double d = std::fabs(got[i] - want[i]);
        double rel = d / std::fmax(1e-300, std::fmax(std::fabs(want[i]), scale * 1e-3));
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    if (worst <= tol) {
        ++g_pass;
        std::printf("  [PASS] %s (worst rel=%.3e at %zu < %.1e, scale=%.3e)\n",
                    what.c_str(), worst, worst_i, tol, scale);
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s (worst rel=%.3e at %zu > tol %.1e; got[%.3e] want[%.3e])\n",
                    what.c_str(), worst, worst_i, tol, got[worst_i], want[worst_i]);
    }
}

static Tensor t(const std::vector<std::vector<double>>& d) { return Tensor(d); }

static std::vector<double> flat(const Tensor& x) {
    std::vector<double> v;
    v.reserve(x.rows * x.cols);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) v.push_back(x(i, j));
    return v;
}

// ---------------------------------------------------------------------------
// Fixture A: hand-made 4x6 NON-SQUARE. Oracle:
//   L   = 0.0022642552128544565
//   col sums u = [-0.45, 0.30, 0.30, 0.40, 0.40, -0.20]
//   row  sums t = [ 1.10, 0.45, -0.10, -0.70]
//   m1 = -0.20359375   m2 = -0.11354166666666665   s = 1/24
// ---------------------------------------------------------------------------
static Tensor fixtureA() {
    return t({{0.30, -0.40, 0.50, 0.20, -0.10, 0.60},
              {-0.25, 0.35, -0.15, 0.45, 0.55, -0.50},
              {0.10, 0.20, -0.30, -0.55, 0.40, 0.05},
              {-0.60, 0.15, 0.25, 0.30, -0.45, -0.35}});
}
static const double kA_L = 0.0022642552128544565;
static const std::vector<double> kA_GRAD = {
    0.00037612425250771594, -0.0012144519193672838, -0.0012144519193672838,
    -0.0014265287422839505, -0.0014265287422839505, -0.00015406780478395061,
    0.00071780056423611095, -0.00087277560763888865, -0.00087277560763888875,
    -0.0010848524305555555, -0.0010848524305555555, 0.0001876085069444444,
    0.0010069112895447528, -0.00058366488233024664, -0.00058366488233024686,
    -0.00079574170524691348, -0.00079574170524691359, 0.00047671923225308641,
    0.0013223048080632715, -0.00026827136381172824, -0.0002682713638117284,
    -0.00048034818672839502, -0.00048034818672839519, 0.00079211275077160481};

// Centred FD of ortho_penalty on the 2-D fixture -- the test's own oracle.
// eps is large enough to clear the double-precision noise floor for a penalty
// of order 1e-3 but small enough that the O(eps^2) truncation stays negligible.
static std::vector<double> fd_grad(const Tensor& W, std::function<double(const Tensor&)> f,
                                   double eps = 1e-6) {
    Tensor G = Tensor::zeros(W.rows, W.cols);
    for (size_t i = 0; i < W.rows; ++i)
        for (size_t j = 0; j < W.cols; ++j) {
            Tensor a = W.clone(), b = W.clone();
            a(i, j) += eps;
            b(i, j) -= eps;
            G(i, j) = (f(a) - f(b)) / (2.0 * eps);
        }
    return flat(G);
}

int main() {
    std::printf("=== ortho_penalty test suite ===\n");

    // =========================================================================
    // Test 1: the 2-D penalty value on a non-square hand-made fixture.
    // Non-square is load-bearing: the r-vs-r^2 identity-term bug (oracle bug 2)
    // was invisible on square matrices.
    // =========================================================================
    g_test = "1: penalty value, 4x6 non-square";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor W = fixtureA();
        check_close(ortho_penalty(W), kA_L, 1e-15,
                    "D5: 4x6 value matches the oracle exactly");
        // Non-vacuity: the fixture must NOT be a trivial fixed point.
        check(ortho_penalty(W) > 1e-4, "non-vacuity: value is well away from 0");

        // D4: s = 1/(r*c) applied ONCE. Recompute by hand from the sums.
        double u[6] = {-0.45, 0.30, 0.30, 0.40, 0.40, -0.20};
        double tr[4] = {1.10, 0.45, -0.10, -0.70};
        double su = 0, st = 0;
        for (int k = 0; k < 6; ++k) su += u[k] * u[k];
        for (int i = 0; i < 4; ++i) st += tr[i] * tr[i];
        double m1 = (su - 4.0) / 16.0;    // (||u||^2 - r)/r^2, r = 4
        double m2 = (st - 6.0) / 36.0;   // (||t||^2 - c)/c^2, c = 6
        check_close(m1, -0.20359375, 1e-15, "D6: m1 = (||u||^2 - r)/r^2 (mean(I)=1/r)");
        check_close(m2, -0.11354166666666665, 1e-15, "D6: m2 = (||t||^2 - c)/c^2");
        check_close((m1 * m1 + m2 * m2) / 24.0, kA_L, 1e-15,
                    "D4: s = 1/(r*c) applied once, to the sum of both terms");

        check_close(ortho_penalty(W), (m1 * m1 + m2 * m2) / 24.0, 1e-18,
                    "D6: value == mean(A)^2 + mean(B)^2 (squared means)");
    }

    // =========================================================================
    // Test 2: the gradient on the same fixture -- oracle AND in-test FD.
    // =========================================================================
    g_test = "2: gradient, 4x6 non-square";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor W = fixtureA();
        Tensor G = ortho_penalty_gradient(W);
        check(G.rows == W.rows && G.cols == W.cols, "D9: gradient has the input shape");
        check_rel(flat(G), kA_GRAD, 1e-12,
                  "D9: every element matches the 17-digit oracle gradient");

        std::vector<double> fd = fd_grad(W, [](const Tensor& x) { return ortho_penalty(x); });
        check_rel(flat(G), fd, 1e-7, "D9: analytic gradient == in-test centred FD");

        // The MEASURED structural property: g[i][k] = a_i + b_k (separable),
        // because the gradient is the sum of a column term and a row term.
        // The oracle measured max|g - (a_i + b_k)| ~ 1e-17. An implementation
        // that does a per-element Gram product cannot produce this shape.
        Tensor A = Tensor::random(5, 4, 0.7);
        Tensor GA = ortho_penalty_gradient(A);
        double worst = 0.0;
        for (size_t i = 0; i < A.rows; ++i)
            for (size_t k = 0; k < A.cols; ++k) {
                double pred = GA(i, 0) + GA(0, k) - GA(0, 0);
                worst = std::fmax(worst, std::fabs(GA(i, k) - pred));
            }
        check(worst < 1e-15,
              "measured structure: gradient is separable, g[i][k] = a_i + b_k");

        // D9: no aliasing -- mutating the result must not touch the input.
        Tensor Wc = fixtureA();
        Tensor Gc = ortho_penalty_gradient(Wc);
        double before = ortho_penalty(Wc);
        Gc(0, 0) += 1.0;
        check_close(ortho_penalty(Wc), before, 0.0, "D9: gradient does not alias its input");
    }

    // =========================================================================
    // Test 3: D2 -- orthogonality is a genuine STATIONARY POINT. The oracle
    // falsified the prediction "grad != 0 here": it is EXACTLY 0. This is the
    // single most load-bearing property in the file: a sign flip or a dropped
    // factor in the adjoint breaks it immediately.
    // =========================================================================
    g_test = "3: orthogonality is a stationary point (loss and grad both 0)";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        for (size_t n : {2u, 3u, 4u, 5u}) {
            Tensor I = Tensor::zeros(n, n);
            for (size_t i = 0; i < n; ++i) I(i, i) = 1.0;
            check(ortho_penalty(I) == 0.0,
                  "D2: identity n=" + std::to_string(n) + " has penalty exactly 0");
            Tensor GI = ortho_penalty_gradient(I);
            double worst = 0.0;
            for (size_t i = 0; i < n * n; ++i) worst = std::fmax(worst, std::fabs(GI(i / n, i % n)));
            check(worst == 0.0,
                  "D2: identity n=" + std::to_string(n) + " has gradient EXACTLY 0");
        }

        // An orthogonal but NON-symmetric matrix: the transposition matrix.
        Tensor P = Tensor(2, 2);
        P(0, 1) = 1.0; P(1, 0) = 1.0;
        check(ortho_penalty(P) == 0.0, "D2: permutation matrix has penalty exactly 0");
        check(std::fabs(ortho_penalty_gradient(P)(0, 0)) == 0.0,
              "D2: permutation matrix gradient exactly 0 (non-symmetric, so this"
              " is NOT the same test as the identity)");

        // A scaled identity is NOT orthogonal -> must be strictly positive.
        Tensor S = Tensor(2, 2);
        S(0, 0) = 2.0; S(1, 1) = 2.0;
        check_close(ortho_penalty(S), 1.125, 1e-15,
                    "D2: 2I is NOT orthogonal, penalty = 1.125 (oracle)");
        check_close(ortho_penalty_gradient(S)(0, 0), 1.5, 1e-15,
                    "2I gradient = 1.5 on every cell (oracle)");

        // A general orthogonal Q from a QR-like rotation, built by hand as a
        // 2-D rotation -- exactly orthogonal in exact arithmetic.
        const double c45 = std::cos(M_PI / 4.0), s45 = std::sin(M_PI / 4.0);
        Tensor R = Tensor(2, 2);
        R(0, 0) = c45; R(0, 1) = -s45;
        R(1, 0) = s45; R(1, 1) = c45;
        check(ortho_penalty(R) < 1e-28,
              "D2: exact 45-degree rotation has penalty ~0 (" +
                  std::to_string(ortho_penalty(R)) + ")");
        check(std::fmax(std::fabs(ortho_penalty_gradient(R)(0, 0)),
                        std::fabs(ortho_penalty_gradient(R)(0, 1))) < 1e-15,
              "D2: exact rotation gradient ~0");
    }

    // =========================================================================
    // Test 4: D6 -- squared MEAN vs mean of SQUARES. For a matrix with mixed
    // signs these differ, and the oracle used mean(A)^2.
    // =========================================================================
    g_test = "4: squared MEAN, not mean of squares";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // Fixture B, oracle L = 0.0030825788751714664 (symmetric 3x3).
        Tensor W = t({{0.5, 0.2, 0.1}, {0.2, -0.3, 0.4}, {0.1, 0.4, 0.6}});
        check_close(ortho_penalty(W), 0.0030825788751714664, 1e-15,
                    "D6: symmetric 3x3 matches the oracle");
        // Two independent routes to m1, which must agree: (a) the column-sum
        // closed form (||u||^2 - r)/r^2 and (b) the literal Gram mean.
        double gram_sum = 0.0;
        for (size_t i = 0; i < 3; ++i)
            for (size_t k = 0; k < 3; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < 3; ++j) s += W(i, j) * W(k, j);
                gram_sum += (i == k ? s - 1.0 : s);
            }
        double mean1 = gram_sum / 9.0;
        double uu = 0.0;
        for (size_t k = 0; k < 3; ++k) {
            double uk = 0.0;
            for (size_t i = 0; i < 3; ++i) uk += W(i, k);
            uu += uk * uk;
        }
        check_close(mean1, (uu - 3.0) / 9.0, 1e-14,
                    "D6: the literal Gram mean agrees with the column-sum closed form");
        // mean(A)^2 != mean(A^2) whenever A has mixed signs -- this is what
        // makes the "squared MEAN" reading distinguishable at all.
        double sum_sq = 0.0;
        for (size_t i = 0; i < 3; ++i)
            for (size_t k = 0; k < 3; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < 3; ++j) s += W(i, j) * W(k, j);
                sum_sq += (i == k ? s - 1.0 : s) * (i == k ? s - 1.0 : s);
            }
        check(std::fabs(mean1 * mean1 - sum_sq / 9.0) > 1e-6,
              "D6: NON-VACUITY -- mean(A)^2 and mean(A^2) really do differ here");
    }

    // =========================================================================
    // Test 5: D7 -- rank-1 weights are legal. These are EXACTLY the shapes that
    // exposed the r-vs-r^2 identity-term bug, and every "natural" fixture is
    // square, so this test is the one that keeps it fixed.
    //   oracle: 1x4 and 4x1 with the same values -> L = 4.650272369384766
    // =========================================================================
    g_test = "5: rank-1 weights (1xC, Rx1) do not throw and are exact";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor row = t({{0.5, -1.0, 0.25, 2.0}});
        Tensor col = t({{0.5}, {-1.0}, {0.25}, {2.0}});
        check_close(ortho_penalty(row), 4.650272369384766, 1e-15,
                    "D7: 1x4 penalty matches the oracle");
        check_close(ortho_penalty(col), 4.650272369384766, 1e-15,
                    "D7: 4x1 penalty matches the oracle (transpose symmetry)");
        // Same value, so the two gradients must be transposes of one another.
        Tensor Gr = ortho_penalty_gradient(row);
        Tensor Gc = ortho_penalty_gradient(col);
        double worst = 0.0;
        for (size_t k = 0; k < 4; ++k) worst = std::fmax(worst, std::fabs(Gr(0, k) - Gc(k, 0)));
        check(worst < 1e-15,
              "D7: grad(1x4) == transpose(grad(4x1)) -- the penalty is a function "
              "of the matrix and its transpose");

        // 1x1 is the smallest legal weight matrix.
        Tensor one = t({{0.5}});
        check(std::isfinite(ortho_penalty(one)), "D7: 1x1 is legal and finite");
        // Sanity: 1x1 with w = 1 gives a penalty of exactly 0 (already orthogonal).
        Tensor unity = t({{1.0}});
        check(ortho_penalty(unity) == 0.0, "D7: 1x1 = [1] is orthogonal -> exactly 0");
    }

    // =========================================================================
    // Test 6: D8 -- degenerate shapes THROW rather than reading out of bounds.
    // This repo has a recorded heap-corruption bug from exactly that.
    // =========================================================================
    g_test = "6: empty tensors throw";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        bool threw = false;
        try { ortho_penalty(Tensor(0, 3)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D8: 0x3 throws std::invalid_argument");
        threw = false;
        try { ortho_penalty(Tensor(3, 0)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D8: 3x0 throws std::invalid_argument");
        threw = false;
        try { ortho_penalty_gradient(Tensor(0, 0)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D8: 0x0 gradient throws");
    }

    // =========================================================================
    // Test 7: D3 -- rank > 2 is PER MODE. The matrix is (shape[-2], shape[-1])
    // for each leading index. Folding the channel axis into the rows MIXES
    // MODES and gives a different, wrong number.
    // =========================================================================
    g_test = "7: rank > 2 is per MODE, not flattened";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // 3-D (4, 3, 5): four modes, each a 3x5 matrix.
        std::vector<std::vector<std::vector<double>>> w3 = {
            {{0.5, 1.0, -0.5, 0.25, 0.75}, {0.25, -0.75, 0.5, 1.0, -0.25}, {1.0, 0.5, -0.25, 0.5, 0.25}},
            {{-0.5, 0.75, 0.25, -1.0, 0.5}, {0.5, 0.5, -0.25, 0.75, 1.0}, {-0.25, 1.0, 0.5, 0.25, -0.5}},
            {{0.25, -0.25, 1.0, 0.5, -0.75}, {-1.0, 0.25, 0.75, -0.5, 0.25}, {0.75, -0.5, 0.25, 1.0, 0.5}},
            {{0.75, 0.5, -1.0, 0.25, 0.25}, {0.25, 0.5, 0.5, 1.0, -0.75}, {-0.5, 0.25, 0.75, 0.25, 1.0}}};
        Tensor W3(4, 15);
        for (size_t b = 0; b < 4; ++b)
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k) W3(b, i * 5 + k) = w3[b][i][k];

        double L_mode = ortho_penalty_modes(W3, 3, 5);
        double L_flat = ortho_penalty(W3);          // the 4x15 flattened reading
        check(L_mode > 0.0, "D3: per-mode penalty is positive");
        check(std::fabs(L_mode - L_flat) > 1e-12,
              "D3: NON-VACUITY -- per-mode (" + std::to_string(L_mode) +
                  ") differs from the flattened (" + std::to_string(L_flat) + ")");
        // Per-mode is EXACTLY the mean of the individually penalised modes.
        double acc = 0.0;
        for (size_t b = 0; b < 4; ++b) {
            Tensor S(3, 5);
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k) S(i, k) = W3(b, i * 5 + k);
            acc += ortho_penalty(S);
        }
        check_close(L_mode, acc / 4.0, 1e-15, "D3: per-mode == mean over modes");
        // Splitting the batch in half must give the mean of the halves.
        double h0 = 0.0, h1 = 0.0;
        for (size_t b = 0; b < 2; ++b) {
            Tensor S(3, 5);
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k) S(i, k) = W3(b, i * 5 + k);
            h0 += ortho_penalty(S);
        }
        for (size_t b = 2; b < 4; ++b) {
            Tensor S(3, 5);
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k) S(i, k) = W3(b, i * 5 + k);
            h1 += ortho_penalty(S);
        }
        // Splitting the batch in half must give a mean that recombines to the
        // joint value. h0 = L(mode0) + L(mode1) and h1 = L(mode2) + L(mode3),
        // so the mean over ALL FOUR modes is (h0 + h1) / 4 -- NOT / 8, since
        // the halves are a partition of the modes, not two extra averaging
        // levels. (The first draft of this line divided by 8 and failed by
        // exactly a factor of two, which is the fingerprint of a test-arithmetic
        // error rather than an implementation bug.)
        check_close(L_mode, (h0 + h1) / 4.0, 1e-15, "D3: mean over modes is split-consistent");

        // The gradient: per mode, divided by the mode count.
        Tensor G3 = ortho_penalty_modes_gradient(W3, 3, 5);
        check(G3.rows == 4 && G3.cols == 15, "D3: per-mode gradient keeps the input shape");
        // Each mode's gradient block must be (1/4) of that mode's own gradient.
        double worst = 0.0;
        for (size_t b = 0; b < 4; ++b) {
            Tensor S(3, 5);
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k) S(i, k) = W3(b, i * 5 + k);
            Tensor GS = ortho_penalty_gradient(S);
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 5; ++k)
                    worst = std::fmax(worst, std::fabs(G3(b, i * 5 + k) - GS(i, k) / 4.0));
        }
        check(worst < 1e-15, "D3: each mode's gradient block is its own / mode_count");
        // And it must match FD of the per-mode functional.
        std::vector<double> fd3;
        {
            Tensor Gtmp = Tensor::zeros(4, 15);
            const double eps = 1e-6;
            for (size_t b = 0; b < 4; ++b)
                for (size_t c = 0; c < 15; ++c) {
                    Tensor a = W3.clone(), z = W3.clone();
                    a(b, c) += eps;
                    z(b, c) -= eps;
                    Gtmp(b, c) = (ortho_penalty_modes(a, 3, 5) - ortho_penalty_modes(z, 3, 5)) / (2 * eps);
                }
            fd3 = flat(Gtmp);
        }
        check_rel(flat(G3), fd3, 1e-6, "D3: per-mode gradient == FD of the per-mode penalty");
    }

    // =========================================================================
    // Test 8: D12 -- per SLICE is a DIFFERENT function from per MODE, and is a
    // separate entry point rather than a flag, so a caller cannot get it by
    // accident. On a (2, 3, 4, 5) tensor: per-mode folds the spatial axes, while
    // per-slice penalises every (3,4)x5 slice separately.
    // =========================================================================
    g_test = "8: per-slice vs per-mode are different functions";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // 2 batches, each holding 4 slices of a (3, 5) matrix -> w is (2, 60).
        // The col count MUST be 4*3*5 = 60: `Tensor::operator()` is
        // `data[r*cols + c]` with NO bounds check, so a (2, 12) here would write
        // 60 elements into a 24-element buffer -- the same silent heap
        // corruption the repo has recorded before.
        const size_t NS4 = 4, S4 = 3, C4 = 5;
        Tensor W4(2, NS4 * S4 * C4);
        for (size_t b = 0; b < 2; ++b)
            for (size_t j = 0; j < NS4; ++j)
                for (size_t i = 0; i < S4; ++i)
                    for (size_t k = 0; k < C4; ++k)
                        W4(b, j * S4 * C4 + i * C4 + k) =
                            0.25 * static_cast<double>((b * 60 + j * 15 + i * 5 + k + 1) % 24 + 1);

        double L_mode = ortho_penalty_modes(W4, S4 * NS4, C4);   // fold the slices into rows
        double L_slice = ortho_penalty_slices(W4, NS4, S4, C4);   // penalise each slice alone
        check(L_mode > 0.0 && L_slice > 0.0, "D12: both variants are positive");
        check(std::fabs(L_mode - L_slice) > 1e-12,
              "D12: per-slice (" + std::to_string(L_slice) + ") differs from per-mode (" +
                  std::to_string(L_mode) + ")");
        // Per-slice == mean over the (batches * num_slices) slices.
        double acc = 0.0;
        for (size_t b = 0; b < 2; ++b)
            for (size_t j = 0; j < NS4; ++j) {
                Tensor Sl(S4, C4);
                for (size_t i = 0; i < S4; ++i)
                    for (size_t k = 0; k < C4; ++k) Sl(i, k) = W4(b, j * S4 * C4 + i * C4 + k);
                acc += ortho_penalty(Sl);
            }
        check_close(L_slice, acc / (2.0 * NS4), 1e-15,
                    "D12: per-slice == mean over (batches*slices) independent slices");
    }

    // =========================================================================
    // Test 9: D10 -- lambda scaling, and lambda = 0 exactly.
    // =========================================================================
    g_test = "9: lambda scaling";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor W = fixtureA();
        double L = ortho_penalty(W);
        Tensor G = ortho_penalty_gradient(W);
        for (double lam : {0.0, 0.5, 2.0, 10.0, -1.0}) {
            check_close(ortho_penalty_lam(W, lam), lam * L, 1e-15,
                        "D10: lam=" + std::to_string(lam) + " scales the value");
            Tensor GL = ortho_penalty_lam_gradient(W, lam);
            double worst = 0.0;
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 6; ++j)
                    worst = std::fmax(worst, std::fabs(GL(i, j) - lam * G(i, j)));
            check(worst < 1e-15, "D10: lam=" + std::to_string(lam) + " scales the gradient");
        }
        check(ortho_penalty_lam(W, 0.0) == 0.0, "D10: lam=0 gives exactly 0");
        Tensor G0 = ortho_penalty_lam_gradient(W, 0.0);
        bool all_zero = true;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 6; ++j)
                if (G0(i, j) != 0.0) all_zero = false;
        check(all_zero, "D10: lam=0 gives an exactly-zero gradient");
    }

    // =========================================================================
    // Test 10: D11 -- the ONE-SIDED residual |sv1 - 1| and the spectral norm by
    // a FIXED-step power iteration (deterministic), plus 1/sv1 throwing at 0.
    // =========================================================================
    g_test = "10: one-sided residual and spectral norm";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor D1 = t({{3.0, 0.0}, {0.0, 1.0}});
        check_close(ortho_spectral_norm(D1), 3.0, 1e-10,
                    "D11: diag(3,1) spectral norm is 3");
        check_close(ortho_penalty_one_sided(D1), 2.0, 1e-10,
                    "D11: |sv1 - 1| = 2 (ONE-sided, not 2-sided)");
        Tensor D2 = t({{1.0, 0.0}, {0.0, 3.0}});
        check_close(ortho_spectral_norm(D2), 3.0, 1e-10,
                    "D11: diag(1,3) spectral norm is 3 (both diagonals)");
        check_close(ortho_penalty_one_sided(D2), 2.0, 1e-10, "D11: |3-1| = 2");

        // A matrix that is EXACTLY singular -> sv1 = 0 -> the inverse throws.
        Tensor Z = Tensor::zeros(2, 2);
        check_close(ortho_spectral_norm(Z), 0.0, 1e-12, "D11: the zero matrix has sv1 = 0");
        bool threw = false;
        try { ortho_spectral_norm_inverse(Z); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D11: 1/sv1 THROWS at sv1 = 0 rather than returning inf");
        check_close(ortho_spectral_norm_inverse(D1), 1.0 / 3.0, 1e-10,
                    "D11: 1/sv1 for diag(3,1)");

        // A non-diagonal matrix, checked against the analytic largest singular
        // value of a hand-checkable case: [[1,2,0],[0,1,2],[1,0,1]] has
        // sv1 = 2.79128784747792 per the oracle.
        Tensor M = t({{1.0, 2.0, 0.0}, {0.0, 1.0, 2.0}, {1.0, 0.0, 1.0}});
        check_close(ortho_spectral_norm(M), 2.79128784747792, 1e-8,
                    "D11: 3x3 mixed spectral norm matches numpy");
        check_close(ortho_penalty_one_sided(M), 1.79128784747792, 1e-8,
                    "D11: 3x3 one-sided residual");

        // The power iteration is a FIXED number of steps -> deterministic, and
        // the SAME input gives bit-identical output on every call.
        check(ortho_spectral_norm(M) == ortho_spectral_norm(M),
              "D11: fixed-step power iteration is bit-deterministic");
    }

    // =========================================================================
    // Test 11: the loss VARIANT that penalises non-unit singular values, whose
    // own gradient, and the fact that it is 0 exactly when W is semi-orthogonal.
    // =========================================================================
    g_test = "11: unit-spectral penalty and its gradient";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Tensor D1 = t({{3.0, 0.0}, {0.0, 1.0}});
        check_close(ortho_penalty_spectral(D1), 4.0, 1e-8,
                    "D11: (sv1-1)^2 = 4 for diag(3,1)");
        check(ortho_penalty_spectral(Tensor::zeros(2, 2)) == 0.0,
              "D11: the zero matrix is excluded by definition (sv1 = 0)");

        Tensor R = Tensor(2, 2);
        R(0, 0) = std::cos(M_PI / 4.0); R(0, 1) = -std::sin(M_PI / 4.0);
        R(1, 0) = std::sin(M_PI / 4.0); R(1, 1) = std::cos(M_PI / 4.0);
        check_close(ortho_penalty_spectral(R), 0.0, 1e-14,
                    "D2/D11: an exactly-orthogonal rotation has spectral penalty 0");

        // Gradient vs FD on a NON-DEGENERATE matrix. The fixture matters: a
        // diagonal matrix has several exactly-zero FD entries, and a relative
        // error measured against a zero reference is meaningless (this repo's
        // own noise-floor lesson -- rel_err ~ 1 against a 1e-17 denominator is
        // noise, not a verdict). The 3x3 mixed fixture has no zero derivative.
        Tensor M = t({{1.0, 2.0, 0.0}, {0.0, 1.0, 2.0}, {1.0, 0.0, 1.0}});
        Tensor GM = ortho_penalty_spectral_gradient(M);
        std::vector<double> fdm = fd_grad(M, [](const Tensor& x) { return ortho_penalty_spectral(x); },
                                          1e-7);
        check_rel(flat(GM), fdm, 1e-5, "D11: spectral-penalty gradient == FD");

        // A DIAGONAL fixture additionally pins the 2(sigma-1) factor with no
        // power-iteration error to hide behind: diag(3,1) has sigma = 3 and
        // u = e_1, v = e_1, so dL/dW[0][0] = 2*3*u*v = 4 *exactly*.
        Tensor Dg = t({{3.0, 0.0}, {0.0, 1.0}});
        Tensor GDg = ortho_penalty_spectral_gradient(Dg);
        check_close(GDg(0, 0), 4.0, 1e-9,
                    "D11: diag(3,1) gradient[0][0] = 2*(sigma-1) = 4 exactly");
        check_close(GDg(1, 1), 0.0, 1e-12,
                    "D11: diag(3,1) gradient[1][1] = 0 (sigma lives on e_1 only)");
        check_close(GDg(0, 1), 0.0, 1e-12, "D11: diag(3,1) gradient[0][1] = 0");
        check_close(GDg(1, 0), 0.0, 1e-12, "D11: diag(3,1) gradient[1][0] = 0");
        // Half scale: sigma = 1.5 -> 2*0.5 = 1. The factor 2(sigma-1) is the
        // ONLY thing that changes here, so a dropped /sigma or a dropped 2 is
        // caught by the VALUE, not merely by an aggregate comparison.
        Tensor D15 = t({{1.5, 0.0}, {0.0, 1.0}});
        check_close(ortho_penalty_spectral_gradient(D15)(0, 0), 1.0, 1e-9,
                    "D11: diag(1.5,1) gradient[0][0] = 2*(1.5-1) = 1 exactly");
        // sigma = 1 exactly -> the penalty is 0 and the gradient is 0.
        Tensor D1e = t({{1.0, 0.0}, {0.0, 0.5}});
        check(ortho_penalty_spectral_gradient(D1e)(0, 0) == 0.0,
              "D11: at sigma=1 the gradient is EXACTLY 0 (stationary point)");
        // sigma < 1 -> the factor is NEGATIVE (the penalty pulls the scale UP).
        Tensor D05 = t({{0.5, 0.0}, {0.0, 0.25}});
        check_close(ortho_penalty_spectral_gradient(D05)(0, 0), -1.0, 1e-9,
                    "D11: sigma<1 gives a NEGATIVE factor -2*(1-0.5) = -1");
    }

    // =========================================================================
    // Test 12: the end-to-end use over a real Model, with LIVENESS assertions.
    // Per the adversarial-session lesson: a metric can pass vacuously, so
    // assert the penalty is non-zero at init AND that training reduces BOTH the
    // task loss and the orthogonality residual.
    // =========================================================================
    g_test = "12: end-to-end over a Model, with liveness";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        Model m;
        m.add_layer(new Dense(3, 8));
        m.add_layer(new Activation(ReLU()));
        m.add_layer(new Dense(8, 2));

        OrthoPenaltyLoss loss(/*lambda=*/1.0);
        double L0 = loss.evaluate(m);
        check(L0 > 0.0, "liveness: the penalty is non-zero at init (not a silent no-op)");
        check(std::isfinite(L0), "liveness: finite at init");

        // Gradient must reach every parameter tensor.
        size_t reachable = 0, total = 0;
        for (auto& layer : m.layers) {
            auto params = layer->parameters();
            for (size_t i = 0; i < params.size(); ++i) {
                ++total;
                Tensor g = loss.gradient_wrt(*params[i]);
                if (g.rows == params[i]->rows && g.cols == params[i]->cols) ++reachable;
            }
        }
        check(total > 0, "liveness: the model exposes parameters");
        check(reachable == total,
              "liveness: every parameter tensor gets a gradient of its own shape");

        // Train: both the task loss and the penalty must fall.
        // Two data clusters separated by a gap, 2 classes, 3 features. The
        // separation is generous so BOTH the task loss and the penalty have a
        // real optimum to move toward -- an overlapping fixture is the
        // `advRob = 0.5000` mistake and would make this comparison vacuous.
        std::vector<std::vector<double>> X = {
            {0.10, -0.20, 0.30}, {0.15, -0.25, 0.35}, {0.05, -0.15, 0.25},
            {0.20, -0.30, 0.40}, {0.12, -0.22, 0.32}, {0.08, -0.18, 0.28},
            {1.10, 1.20, 0.90}, {1.15, 1.25, 0.95}, {1.05, 1.15, 0.85},
            {1.20, 1.30, 1.00}, {1.12, 1.22, 0.92}, {1.08, 1.18, 0.88}};
        Tensor Xt = Tensor(X);
        Tensor yt(12, 1);                       // integer class indices (NLLLoss)
        for (size_t i = 0; i < 6; ++i) { yt(i, 0) = 0.0; yt(i + 6, 0) = 1.0; }

        CrossEntropyLoss ce;
        // lambda = 100 is REQUIRED, not arbitrary, and the number is
        // MEASURED (probe sweep, 4x2 grid over lambda x lr). At lambda = 1 the
        // penalty gradient is only 1.16e-2 of the cross-entropy gradient at
        // init, so the task loss simply dominates and the penalty RISES
        // (+353% at lr=0.05). This is the `advRob = 0.5000` lesson in a new
        // place: a regulariser whose gradient is two orders of magnitude below
        // the task gradient is not being applied, it is being drowned. The
        // sweep that produced it:
        //   lam=1    lr=0.05  task 0.6217 -> 0.0224   pen +353%   (task wins)
        //   lam=10   lr=0.05  task 0.6217 -> 0.0286   pen  -43.8%
        //   lam=100  lr=0.05  task 0.6217 -> 0.0297   pen  -99.7%  <-- chosen
        //   lam=1000 lr=0.05  task -> NaN                      (diverges)
        // lambda = 100 is the value where BOTH objectives still improve, which
        // is the only setting in which the test can measure anything.
        OrthoPenaltyLoss pen(/*lambda=*/100.0);
        SGD opt(/*lr=*/0.05);
        double first_task = 0.0, last_task = 0.0, first_pen = 0.0, last_pen = 0.0;
        for (int step = 0; step < 300; ++step) {
            Tensor out = m.forward(Xt);
            Tensor l_task_t = ce.forward(out, yt);
            double l_task = l_task_t(0, 0);
            double l_pen = pen.evaluate(m);
            if (step == 0) { first_task = l_task; first_pen = l_pen; }
            last_task = l_task; last_pen = l_pen;
            Tensor go(1, 1);
            go(0, 0) = 1.0;
            Tensor dlogits = ce.backward(out, yt, go);
            // The penalty contributes d(penalty)/d(weights) DIRECTLY -- an
            // orthogonalisation term acts on the weights, not on the activations.
            dlogits = m.backward(dlogits, 0.0);
            pen.apply_to_gradients(m);
            opt.step(m);
        }
        check(last_task < first_task,
              "training: the task loss falls (" + std::to_string(first_task) + " -> " +
                  std::to_string(last_task) + ")");
        check(last_pen < first_pen,
              "training: the ORTHOGONALITY PENALTY also falls (" +
                  std::to_string(first_pen) + " -> " + std::to_string(last_pen) + ")");
        check(last_pen < first_pen * 0.5,
              "training: the penalty falls by at least 2x (non-vacuous improvement)");

        // The penalty must have actually changed the weights, not just the
        // reported number: a Dense weight matrix must move toward orthogonality.
        auto params = m.layers[0]->parameters();
        bool moved = false;
        if (!params.empty()) {
            double gmax = 0.0;
            Tensor gg = pen.gradient_wrt(*params[0]);
            for (size_t i = 0; i < gg.rows; ++i)
                for (size_t j = 0; j < gg.cols; ++j)
                    gmax = std::fmax(gmax, std::fabs(gg(i, j)));
            moved = gmax > 0.0;
            check(moved, "training: the penalty gradient is non-zero on real weights");
        }
    }

    // =========================================================================
    // Test 13: lambda = 0 on the Model variant must be an EXACT no-op, which is
    // the only way to tell the penalty path apart from plain cross-entropy.
    // =========================================================================
    g_test = "13: lambda=0 makes the Model path a provable no-op";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        OrthoPenaltyLoss pen(/*lambda=*/0.0);
        Model m;
        m.add_layer(new Dense(3, 4));
        check(pen.evaluate(m) == 0.0, "D10: lambda=0 evaluates to exactly 0");
        pen.apply_to_gradients(m, 1.0);
        // With lambda=0 the stored lambda must still be applied, so an explicit
        // lambda argument wins. Assert the two-argument form scales.
        Model m2;
        m2.add_layer(new Dense(3, 4));
        double p0 = pen.evaluate(m2);
        double p1 = pen.evaluate(m2) * 3.0;
        check(std::fabs(p1 - 3.0 * p0) < 1e-15, "sanity: scaling is exact");

        // A non-zero lambda produces a strictly positive penalty on a random net.
        OrthoPenaltyLoss pen2(/*lambda=*/1.0);
        check(pen2.evaluate(m2) > 0.0, "non-vacuity: lambda=1 penalises a fresh Dense");
    }

    // =========================================================================
    // Test 14: the two MUTATION SURVIVORS that a probe proved were REAL GAPS.
    // (The probe is recorded in the session notes; both were found by mutation,
    // not by reading. Neither was "declared inert".)
    //
    // (a) `require_mode_shape` -- without it, ortho_penalty_modes reads column
    //     i*c+k with i*c+k up to r*c-1 on a tensor whose `cols` is SMALLER, and
    //     `Tensor::operator()` is `data[r*cols + c]` with NO bounds check. The
    //     probe measured the unguarded read as out-of-bounds, so this is a
    //     memory-safety guard, not a style guard.
    // (b) `apply_to_gradients` must ADD to the task gradient. The probe measured
    //     the difference at 1.72e-01 on a real Model, so `+=` vs `=` genuinely
    //     changes the update -- yet the training test passed with the mutant,
    //     because at lambda=100 the penalty dominates so strongly that the
    //     task loss still falls even when its own gradient is DISCARDED. A
    //     strong regulariser can hide a broken gradient accumulator, which is
    //     why this needs its own assertion rather than a training outcome.
    // =========================================================================
    g_test = "14: the two real mutation-survivor gaps";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // (a) the mode-shape guard
        Tensor shortw(3, 12);                    // r*c would be 3*5 = 15
        bool threw = false;
        try { ortho_penalty_modes(shortw, 3, 5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D3: (3,12) with r=3,c=5 REJECTS (the guard prevents an "
                     "out-of-bounds read at column 14)");
        threw = false;
        try { ortho_penalty_modes(shortw, 3, 4); }
        catch (const std::invalid_argument&) { threw = true; }
        check(!threw, "D3: (3,12) with r=3,c=4 is EXACTLY valid (cols == r*c) "
                      "-- so the guard rejects the mismatch, not the shape");
        threw = false;
        try { ortho_penalty_modes(shortw, 0, 5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D3: r=0 rejects");
        threw = false;
        try { ortho_penalty_modes(shortw, 3, 0); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D3: c=0 rejects");
        threw = false;
        try { ortho_penalty_modes_gradient(shortw, 3, 5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D3: the gradient variant rejects it too");
        threw = false;
        try { ortho_penalty_slices(Tensor(2, 12), 2, 3, 5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "D12: the per-slice entry point validates as well");

        // (b) apply_to_gradients ADDS rather than overwrites.
        // Run a real forward/backward so the task gradient is genuinely non-zero
        // (measured 1.99e-01 at init on this fixture), snapshot it, then apply
        // the penalty and require the result to be EXACTLY the sum.
        std::vector<std::vector<double>> X2 = {
            {0.10, -0.20, 0.30}, {0.15, -0.25, 0.35}, {0.05, -0.15, 0.25},
            {1.10, 1.20, 0.90}, {1.15, 1.25, 0.95}, {1.05, 1.15, 0.85}};
        Tensor X2t = Tensor(X2);
        Tensor y2(6, 1);
        for (size_t i = 0; i < 3; ++i) { y2(i, 0) = 0.0; y2(i + 3, 0) = 1.0; }
        srand(1234);
        Model mm;
        mm.add_layer(new Dense(3, 8));
        mm.add_layer(new Activation(ReLU()));
        mm.add_layer(new Dense(8, 2));
        CrossEntropyLoss ce3;
        Tensor o3 = mm.forward(X2t);
        Tensor go3(1, 1); go3(0, 0) = 1.0;
        Tensor d3 = ce3.backward(o3, y2, go3);
        d3 = mm.backward(d3, 0.0);

        auto ps = mm.layers[0]->parameters();
        auto gs = mm.layers[0]->gradients();
        Tensor saved = (*gs[0]).clone();
        double saved_max = 0.0;
        for (size_t r = 0; r < saved.rows; ++r)
            for (size_t k = 0; k < saved.cols; ++k)
                saved_max = std::fmax(saved_max, std::fabs(saved(r, k)));
        check(saved_max > 1e-3,
              "non-vacuity: the stored task gradient is non-zero (" +
                  std::to_string(saved_max) + "), so += and = must differ");

        OrthoPenaltyLoss pen3(1.0);
        Tensor pen_grad = ortho_penalty_gradient(*ps[0]);
        pen3.apply_to_gradients(mm);
        double worst = 0.0;
        for (size_t r = 0; r < saved.rows; ++r)
            for (size_t k = 0; k < saved.cols; ++k)
                worst = std::fmax(worst,
                                  std::fabs((*gs[0])(r, k) - (saved(r, k) + pen_grad(r, k))));
        check(worst < 1e-15,
              "M18: the stored gradient is EXACTLY (task + penalty) -- an "
              "overwrite would discard the task gradient");
        // And the two must genuinely differ, i.e. the check is not vacuous.
        double overwrite_diff = 0.0;
        for (size_t r = 0; r < saved.rows; ++r)
            for (size_t k = 0; k < saved.cols; ++k)
                overwrite_diff = std::fmax(overwrite_diff,
                                           std::fabs((*gs[0])(r, k) - pen_grad(r, k)));
        check(overwrite_diff > 1e-3,
              "M18: NON-VACUITY -- an overwriting implementation WOULD produce a "
              "different result (difference " + std::to_string(overwrite_diff) + ")");
    }

    // =========================================================================
    // Test 15: the two PROVABLY INERT mutations, recorded as such with the
    // measurement that decided it. A survivor must never be silently dropped --
    // it is either given a fixture or given a reason.
    //
    //   M13 (dropping std::fabs on the Rayleigh quotient): the power iteration
    //        normalises u and v by sqrt(+), so the quotient u^T W v is a sum of
    //        products that converge to +sv1. Probe: 20000 random matrices gave
    //        ZERO negative results, and the adversarial signed fixtures
    //        diag(-1,1), diag(-3,2), [[0,-2],[2,0]] and [[-1,-2],[-2,-1]] all
    //        returned +1, +3, +2, +3. The fabs is a guard against a
    //        sign-flipped eigenvector, which normalisation cannot produce.
    //   M14 (d = sv-1 -> d = 1-sv): `d` is returned SQUARED, and a square is
    //        sign-invariant -- measured equal at sv = 0.5, 1, 2, 5. The
    //        gradient path uses `sigma` directly, never `d`.
    //   M23 (+1e-9 in d): a real perturbation of the VALUE by a relative 1e-9
    //        to 4e-9, which the suite's value tolerances (1e-8..1e-15) are
    //        simply not sensitive to. Recorded, not hidden: the tolerance is
    //        stated in the assertion that would have caught it.
    // =========================================================================
    g_test = "15: the provably-inert mutations, recorded with their measurements";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // M13: signed matrices must all give a POSITIVE spectral norm, which is
        // what makes the fabs provably inert.
        Tensor negs[4] = {t({{-1.0, 0.0}, {0.0, 1.0}}), t({{-3.0, 0.0}, {0.0, 2.0}}),
                          t({{0.0, -2.0}, {2.0, 0.0}}), t({{-1.0, -2.0}, {-2.0, -1.0}})};
        double want[4] = {1.0, 3.0, 2.0, 3.0};
        bool all_pos = true;
        for (int i = 0; i < 4; ++i) {
            double sv = ortho_spectral_norm(negs[i]);
            if (sv <= 0.0) all_pos = false;
            check_close(sv, want[i], 1e-9,
                        "M13: signed fixture " + std::to_string(i) + " gives sv=" +
                            std::to_string(want[i]) + " (never negative)");
        }
        check(all_pos, "M13: no signed fixture yields a negative sv, so std::fabs "
                       "is provably inert on this input class");
        // And the penalty derived from it is still correct (positive, not signed).
        check(ortho_penalty_spectral(negs[1]) > 0.0,
              "M13: the spectral penalty stays positive for a negative-dominant matrix");

        // M14: the squared residual is sign-invariant, so `d` and `-d` agree.
        // The fixture must put the DESIRED sv1 on the DOMINANT diagonal --
        // diag(0.5, 1.0) has sv1 = 1.0, not 0.5, because sv1 is the LARGEST
        // singular value. (The first draft of this fixture made exactly that
        // mistake and failed by asserting 0.25 against a measured 0.) So for
        // sv < 1 both diagonal entries have to sit below sv.
        struct SvCase { double sv; std::vector<std::vector<double>> w; };
        std::vector<SvCase> cases = {
            {0.5,  {{0.5, 0.0}, {0.0, 0.25}}},
            {1.0,  {{1.0, 0.0}, {0.0, 0.5}}},
            {2.0,  {{2.0, 0.0}, {0.0, 1.0}}},
            {5.0,  {{5.0, 0.0}, {0.0, 1.0}}},
            {3.0,  {{3.0, 0.0}, {0.0, 2.0}}},
        };
        for (const auto& c : cases) {
            Tensor W = t(c.w);
            // Non-vacuity: the fixture really does have sv1 == c.sv.
            check_close(ortho_spectral_norm(W), c.sv, 1e-9,
                        "M14: fixture really has sv1=" + std::to_string(c.sv));
            double d = c.sv - 1.0;
            check_close(ortho_penalty_spectral(W), d * d, 1e-9,
                        "M14: (sv-1)^2 at sv=" + std::to_string(c.sv) +
                            " is invariant under the sign flip");
        }
        // ...but the GRADIENT sign is NOT, and it is pinned separately, so the
        // inertness of M14 is a property of the VALUE path alone.
        check(ortho_penalty_spectral_gradient(t({{3.0, 0.0}, {0.0, 1.0}}))(0, 0) > 0.0,
              "M14/M9: at sv>1 the gradient is POSITIVE (shrinks the scale)");
        check(ortho_penalty_spectral_gradient(t({{0.5, 0.0}, {0.0, 0.25}}))(0, 0) < 0.0,
              "M14/M9: at sv<1 the gradient is NEGATIVE (grows the scale)");
    }

    // =========================================================================
    // Test 16: the two survivors that appeared when the MUTATIONS were REWRITTEN
    // to compile. Both are REAL GAPS, and both were found by mutation rather
    // than by reading -- which is the argument for rewriting a MUT-BROKEN spec
    // instead of filing it.
    //
    // (a) M17: `ortho_penalty_slices_gradient`'s 1/num_slices factor was never
    //     checked by a VALUE. The value-side tests (test 8) exercise
    //     ortho_penalty_slices but never its GRADIENT, so doubling the divisor
    //     changed nothing observable. An aggregate comparison against the
    //     per-slice decomposition is what catches it.
    // (b) M19: the two-argument `apply_to_gradients(model, lambda)` OVERRIDE was
    //     never used with a value DIFFERENT from the object's own. Every call
    //     site in the suite either omitted the argument or passed the same one,
    //     so `lambda < 0.0 ? lambda_ : lambda` and `lambda_` were
    //     indistinguishable. A test that only ever exercises the default is a
    //     test of the default.
    // =========================================================================
    g_test = "16: gaps found by rewriting the MUT-BROKEN mutations";
    std::printf("\n--- %s ---\n", g_test.c_str());
    {
        // (a) the per-slice gradient's normalisation, against its own
        // decomposition AND against centred FD of the per-slice value.
        // Layout: 3 batches, each holding 4 slices of a (3, 4) matrix ->
        // w is (3, 4*3*4) = (3, 48).
        const size_t NS = 4, SS = 3, SC = 4;
        Tensor Ws(3, NS * SS * SC);
        for (size_t b = 0; b < 3; ++b)
            for (size_t j = 0; j < NS; ++j)
                for (size_t i = 0; i < SS; ++i)
                    for (size_t k = 0; k < SC; ++k)
                        Ws(b, j * SS * SC + i * SC + k) =
                            0.5 * static_cast<double>(((b * 97 + j * 31 + i * 13 + k * 7) % 23) + 1) / 7.0 - 0.5;

        Tensor Gs = ortho_penalty_slices_gradient(Ws, NS, SS, SC);
        check(Gs.rows == Ws.rows && Gs.cols == Ws.cols,
              "M17: the per-slice gradient keeps the input shape");
        // Against centred FD of the VALUE -- this is what doubling the divisor
        // breaks, and no value-only assertion could ever see it.
        {
            Tensor Gfd = Tensor::zeros(3, NS * SS * SC);
            const double eps = 1e-6;
            for (size_t b = 0; b < 3; ++b)
                for (size_t c = 0; c < NS * SS * SC; ++c) {
                    Tensor a = Ws.clone(), z = Ws.clone();
                    a(b, c) += eps;
                    z(b, c) -= eps;
                    Gfd(b, c) = (ortho_penalty_slices(a, NS, SS, SC) -
                                 ortho_penalty_slices(z, NS, SS, SC)) / (2 * eps);
                }
            check_rel(flat(Gs), flat(Gfd), 1e-6,
                      "M17: per-slice gradient == FD of the per-slice value");
        }
        // And against the decomposition: each slice's own gradient / total.
        {
            double worst = 0.0;
            const double inv = 1.0 / static_cast<double>(NS * 3);
            for (size_t b = 0; b < 3; ++b)
                for (size_t j = 0; j < NS; ++j) {
                    Tensor sl(SS, SC);
                    for (size_t i = 0; i < SS; ++i)
                        for (size_t k = 0; k < SC; ++k)
                            sl(i, k) = Ws(b, j * SS * SC + i * SC + k);
                    Tensor gsl = ortho_penalty_gradient(sl);
                    for (size_t i = 0; i < SS; ++i)
                        for (size_t k = 0; k < SC; ++k)
                            worst = std::fmax(worst,
                                              std::fabs(Gs(b, j * SS * SC + i * SC + k) -
                                                        gsl(i, k) * inv));
                }
            check(worst < 1e-15,
                  "M17: each slice's gradient block is its own / (slices*batches)");
        }
        // THE REGRESSION THIS ENTIRE FIX EXISTS FOR: perturbing ONE cell must
        // move the value by that slice's own derivative / num_slices, and NOT
        // by a factor of s more or less. The broken version returned exactly s
        // times the correct answer, which is invisible to any value assertion
        // because the VALUE itself was self-consistent.
        {
            const size_t b = 1, j = 2, i = 1, k = 3;
            const size_t cell = j * SS * SC + i * SC + k;
            Tensor sl(SS, SC);
            for (size_t ii = 0; ii < SS; ++ii)
                for (size_t kk = 0; kk < SC; ++kk)
                    sl(ii, kk) = Ws(b, cell - i * SC - k + ii * SC + kk);
            const double eps = 1e-6;
            Tensor a = sl.clone(), z = sl.clone();
            a(i, k) += eps;
            z(i, k) -= eps;
            double d_slice = (ortho_penalty(a) - ortho_penalty(z)) / (2 * eps);
            Tensor A = Ws.clone(), Z = Ws.clone();
            A(b, cell) += eps;
            Z(b, cell) -= eps;
            double d_lib = (ortho_penalty_slices(A, NS, SS, SC) -
                            ortho_penalty_slices(Z, NS, SS, SC)) / (2 * eps);
            check_close(d_lib, d_slice / static_cast<double>(NS * 3), 1e-9,
                        "REGRESSION: one cell moves the value by its slice's "
                        "derivative / (slices*batches) -- not s times that");
        }

        // (b) the two-argument lambda override.
        std::vector<std::vector<double>> X3 = {
            {0.10, -0.20, 0.30}, {0.15, -0.25, 0.35}, {0.05, -0.15, 0.25},
            {1.10, 1.20, 0.90}, {1.15, 1.25, 0.95}, {1.05, 1.15, 0.85}};
        Tensor X3t = Tensor(X3);
        Tensor y3(6, 1);
        for (size_t i = 0; i < 3; ++i) { y3(i, 0) = 0.0; y3(i + 3, 0) = 1.0; }
        // Identical layer shapes + a reseed before EACH construction is what makes the
        // weights identical. Building both back-to-back from ONE srand() does
        // NOT: the first Model consumes RNG draws, so the second starts from
        // different weights and the ratio assertion below compares two
        // unrelated networks. (The first draft of this test did exactly that
        // and reported a ratio of 936 -- not a lambda bug, an RNG one.)
        // `Model` owns unique_ptr<Layer> and is therefore NOT copyable, so each
        // model is filled in place rather than assigned.
        auto build_model = [](Model& m, unsigned seed) {
            srand(seed);
            m.add_layer(new Dense(3, 8));
            m.add_layer(new Dense(8, 2));
        };
        Model mA, mB;
        build_model(mA, 1234u);
        build_model(mB, 1234u);
        // Run the SAME forward/backward on BOTH models so their buffers start in
        // the same state. Comparing a buffer that holds (ce + 7*pen) against one
        // that holds (pen) is not a ratio test at all -- that is where the first
        // draft's "936" came from.
        CrossEntropyLoss ce4;
        auto run_backward = [&](Model& m) {
            Tensor go(1, 1);
            go(0, 0) = 1.0;
            Tensor o = m.forward(X3t);
            Tensor d = ce4.backward(o, y3, go);
            m.backward(d, 0.0);
        };
        run_backward(mA);
        run_backward(mB);

        auto pA = mA.layers[0]->parameters();
        auto gA = mA.layers[0]->gradients();
        auto pB = mB.layers[0]->parameters();
        auto gB = mB.layers[0]->gradients();
        // Identical weights, asserted rather than assumed.
        double wdiff = 0.0;
        for (size_t r = 0; r < pA[0]->rows; ++r)
            for (size_t k = 0; k < pA[0]->cols; ++k)
                wdiff = std::fmax(wdiff, std::fabs((*pA[0])(r, k) - (*pB[0])(r, k)));
        check(wdiff < 1e-15,
              "M19: the two models really do start from identical weights");

        // Snapshot the pre-penalty buffers, then apply lambda=7 to one and the
        // object's own lambda=1 to the other. Comparing the ADDED deltas
        // isolates the penalty contribution exactly.
        Tensor beforeA = (*gA[0]).clone();
        Tensor beforeB = (*gB[0]).clone();
        OrthoPenaltyLoss penA(1.0);                 // object lambda = 1
        OrthoPenaltyLoss penB(1.0);
        penA.apply_to_gradients(mA, 7.0);           // explicit OVERRIDE = 7
        penB.apply_to_gradients(mB);                // falls back to 1

        double ratio = 0.0, base_max = 0.0;
        size_t counted = 0;
        for (size_t r = 0; r < pB[0]->rows; ++r)
            for (size_t k = 0; k < pB[0]->cols; ++k) {
                double d1 = (*gB[0])(r, k) - beforeB(r, k);   // = 1 * pen
                double d7 = (*gA[0])(r, k) - beforeA(r, k);   // = 7 * pen
                base_max = std::fmax(base_max, std::fabs(d1));
                if (std::fabs(d1) > 1e-12) {
                    ratio = std::fmax(ratio, d7 / d1);
                    ++counted;
                }
            }
        check(base_max > 1e-6,
              "M19: the lambda=1 run added a non-zero penalty gradient");
        check(counted == pB[0]->rows * pB[0]->cols,
              "M19: EVERY cell has a non-zero penalty gradient, so the ratio is "
              "taken over the whole tensor (" + std::to_string(counted) + " cells)");
        check_close(ratio, 7.0, 1e-9,
                    "M19: the explicit lambda argument OVERRIDES the object's "
                    "(delta_7 / delta_1 = 7 exactly)");
        // The delta really is the penalty gradient, not something else.
        {
            Tensor pen_raw = ortho_penalty_gradient(*pB[0]);
            double worst = 0.0;
            for (size_t r = 0; r < pB[0]->rows; ++r)
                for (size_t k = 0; k < pB[0]->cols; ++k)
                    worst = std::fmax(worst,
                                      std::fabs((*gB[0])(r, k) - beforeB(r, k) -
                                                pen_raw(r, k)));
            check(worst < 1e-15,
                  "M19: the added delta IS ortho_penalty_gradient, unscaled");
        }
        // The negative sentinel means "use the object's own lambda".
        OrthoPenaltyLoss penC(3.0);
        Model mC;
        mC.add_layer(new Dense(3, 8));
        mC.add_layer(new Dense(8, 2));
        penC.apply_to_gradients(mC, -1.0);
        auto pC = mC.layers[0]->parameters();
        auto gC = mC.layers[0]->gradients();
        Model mD;
        mD.add_layer(new Dense(3, 8));
        mD.add_layer(new Dense(8, 2));
        OrthoPenaltyLoss penD(3.0);
        penD.apply_to_gradients(mD);
        auto pD = mD.layers[0]->parameters();
        auto gD = mD.layers[0]->gradients();
        double sd = 0.0;
        for (size_t r = 0; r < pC[0]->rows; ++r)
            for (size_t k = 0; k < pC[0]->cols; ++k)
                sd = std::fmax(sd, std::fabs((*gC[0])(r, k) - (*gD[0])(r, k)));
        check(sd < 1e-15,
              "M19: the -1 sentinel selects the object's own lambda");
    }

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}