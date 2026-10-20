// =============================================================================
// tests/test_sinkhorn.cpp -- entropic optimal transport (Sinkhorn-Knopp), the
// debiased divergence, the point-space adjoint, and the barycentric map.
//
// THE SPECIFICATION IS tests/sinkhorn_oracle.py, WHICH RUNS GREEN (7/7) AND IS
// THE FILE THAT FOUND THE BUGS IN THIS ONE'S DERIVATIONS BEFORE ANY C++
// EXISTED. Six of them are load-bearing enough to repeat here, because a
// future reader of this test who does not know them will reintroduce them:
//
//   1. The log-domain update ACCUMULATES (`f += eps*(log a - lse)`), it is not
//      an assignment. The negated form doubles the error per sweep and reaches
//      NaN in ~40 iterations.
//   2. A CORRECT PLAN MAY CONTAIN EXACT ZEROS. At eps=1e-3 the plan is
//      [0.5 0; 0 0.5] with marginal residual EXACTLY 0 and finite duals. Every
//      test here that touches the plan asserts MARGINALS and DUALS, never
//      "min(P) > 0", which would fail a correct implementation.
//   3. The entropic bias is NEGATIVE under the generalised entropy: measured
//      OT_eps(a,a) = -0.147295507452766 at eps=0.05. A magnitude-only check
//      passes a solver with the wrong entropy offset, so the SIGN is pinned.
//   4. The point-space adjoint's SELF-TERM COEFFICIENT IS -2. Using -1 was
//      measured at 3.6e-02 relative error where -2 gives 7.3e-10.
//   5. Finite differences are only meaningful on a SMALL fixture: S(x) is
//      computed iteratively, so on a 6x5 fixture the central difference
//      measures the staircase in the iteration count and every candidate chain
//      -- including the correct one -- "fails" at ~0.9.
//   6. eps->0 gives W_2^2 for a SQUARED-EUCLIDEAN cost, and eps->inf gives
//      the independent-coupling COST. Neither is W_1^2 or MMD^2; asserting the
//      textbook versions was the oracle's first-draft error.
//
// Every numeric expectation below was measured by the oracle in float64. Where
// a value is quoted to many digits it is a PINNED expectation, not a tolerance.
// =============================================================================

#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static double rel(double got, double want) {
    double d = std::fabs(got - want);
    double s = std::max(std::fabs(got), std::fabs(want));
    return d / (s > 1e-12 ? s : 1e-12);
}

static Tensor mat(size_t r, size_t c, const std::vector<double>& flat) {
    return Tensor(r, c, flat.data());
}

static Tensor uniform(size_t n) {
    Tensor t = Tensor::zeros(n, 1);
    for (size_t i = 0; i < n; ++i) t(i, 0) = 1.0 / static_cast<double>(n);
    return t;
}

// Throwing helper. Every guard test goes through this so that a mutation which
// makes a guard throw an EXCEPTION TYPE the test does not catch is reported as
// a failure rather than aborting the binary -- an aborted binary produces no
// summary line and the harness then has nothing to count (the muP M13 shape:
// the detector was working and only the counting was missing).
template <typename F>
static std::string thrown_message(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("__OTHER__:") + e.what();
    }
    return "";
}

int main() {
    // =========================================================================
    printf("\n--- Test 1: solver recovers the marginals (oracle sect. 1) ---\n");
    // 7 x 5, eps = 0.05. Oracle measured marginal residual 9.909e-15 with
    // tol 1e-14; the C++ uses the default tol of 1e-12, so the assertion is at
    // 1e-10 -- still four orders above the floor, so a broken iteration
    // cannot hide under it.
    {
        // THE ORACLE'S EXACT FIXTURE, transcribed from tests/_dump_fixture.py.
        // A pinned number only transfers across an IDENTICAL fixture: the
        // first draft of this file hand-transcribed plausible-looking points
        // and got OT_eps(a,a) = -0.350050 where the oracle says -0.147296.
        // Both numbers were correct for their own fixture; only one of them
        // was the oracle's.
        Tensor X = mat(7, 3, {-1.9162409852496611, -0.12930802768468755, -2.1033222408352561, 1.526720011903973, 1.3907093982965033, -0.42770139546603458, 0.54276060371642243, -0.35722252268361054, 1.7088382114722984, -1.2685075918756816, -2.070155222873161, -0.14055737447378064, -0.35526225649256671, 0.17605706378276476, 0.39351521839513554, 0.95261951411773382, 0.16204019990332144, -0.3698866264805723, 0.47732547713010748, 0.90913595410612502, -2.1425228516876671});
        Tensor Y = mat(5, 3, {-1.2982481299103528, -0.32163558568633632, 0.56531166671149136, 0.66486676251110577, 0.05920870105075382, 0.44358802581609408, 2.0637045762411996, -0.61627013584788182, -0.1459699535580089, -0.12003841045170238, 2.0641452736769463, -0.1151844330167342, 1.6792056490498704, -0.4171990252800607, 3.7267978291430097});
        SinkhornCosts c = sinkhorn_sqdist(X, Y);
        Tensor a = uniform(7), b = uniform(5);
        Tensor P;
        SinkhornReport rep = sinkhorn_log(c.cxy, a, b, 0.05, 200000, 1e-14, &P);

        printf("  iterations=%d residual=%.3e converged=%d\n",
               rep.iterations, rep.marginal_residual, (int)rep.converged);
        check(rep.converged, "1a solver reports convergence");
        check(rep.marginal_residual < 1e-10, "1b marginal residual below 1e-10");
        check(P.rows == 7 && P.cols == 5, "1c plan has the shape of the cost");

        // The marginals themselves, entry by entry, not just a norm.
        double worst = 0.0;
        for (size_t i = 0; i < 7; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < 5; ++j) s += P(i, j);
            worst = std::max(worst, std::fabs(s - 1.0 / 7.0));
        }
        for (size_t j = 0; j < 5; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < 7; ++i) s += P(i, j);
            worst = std::max(worst, std::fabs(s - 1.0 / 5.0));
        }
        check(worst < 1e-10, "1d every row and column marginal matches to 1e-10");
        check(plan_marginal_error(P, a, b) < 1e-10, "1e plan_marginal_error agrees");

        // NO plan entry may be negative: unlike the tiny-eps case (Test 9) the
        // plan here is strictly interior, so non-negativity IS a real property
        // and is asserted -- but as non-negativity, not strict positivity.
        double mn = P(0, 0);
        for (size_t i = 0; i < 7; ++i)
            for (size_t j = 0; j < 5; ++j) mn = std::min(mn, P(i, j));
        check(mn >= 0.0, "1f no plan entry is negative");

        // RECTANGULAR IN BOTH DIRECTIONS. This is not redundancy: the solver
        // buffers a row of the scaled matrix (m entries) and a column (n
        // entries), and sizing either from the wrong axis is a heap-buffer-
        // overflow for every problem where n != m. A suite that only ever
        // solved SQUARE problems could not see that bug at all -- the two
        // directions are therefore both required, and this test failed for real
        // during development (a `std::vector` index assertion under
        // _GLIBCXX_ASSERTIONS, and a SIGSEGV at -O2 without them).
        for (int cfg = 0; cfg < 2; ++cfg) {
            size_t nn = cfg == 0 ? 3 : 7;   // n > m and n < m in turn
            size_t mm = cfg == 0 ? 7 : 3;
            std::vector<double> xf(nn * 2), yf(mm * 2);
            for (size_t i = 0; i < nn * 2; ++i)
                xf[i] = std::sin(0.9 * static_cast<double>(i) + 1.0);
            for (size_t i = 0; i < mm * 2; ++i)
                yf[i] = std::cos(1.3 * static_cast<double>(i));
            Tensor Xi = mat(nn, 2, xf), Yi = mat(mm, 2, yf);
            SinkhornCosts ci = sinkhorn_sqdist(Xi, Yi);
            Tensor Pi;
            SinkhornReport ri = sinkhorn_log(ci.cxy, uniform(nn), uniform(mm),
                                             0.1, 200000, 1e-13, &Pi);
            double err = plan_marginal_error(Pi, uniform(nn), uniform(mm));
            printf("  n=%zu m=%zu: iters=%d residual=%.3e\n", nn, mm,
                   ri.iterations, err);
            check(Pi.rows == nn && Pi.cols == mm,
                  "1g the plan has the requested rectangular shape");
            check(err < 1e-10, "1h marginals recovered for this rectangular shape");
        }
    }

    // =========================================================================
    printf("\n--- Test 2: guards throw, each with a DISTINCT message ---\n");
    // A substring appearing in two messages discriminates nothing (this repo's
    // third such instance). Each guard therefore gets its own message check.
    {
        Tensor C = mat(3, 2, {0, 1, 1, 0, 2, 1});
        Tensor a3 = uniform(3), b2 = uniform(2);

        check(thrown_message([&] { sinkhorn_log(C, a3, b2, 0.0); })
                  .find("eps") != std::string::npos,
              "2a eps == 0 throws, message names eps");
        check(thrown_message([&] { sinkhorn_log(C, a3, b2, -1.0); })
                  .find("eps") != std::string::npos,
              "2b negative eps throws, message names eps");
        check(thrown_message([&] { sinkhorn_log(C, a3, b2, INFINITY); })
                  .find("finite") != std::string::npos,
              "2c infinite eps throws, message names finite");
        check(thrown_message([&] { sinkhorn_log(Tensor::zeros(0, 0), a3, b2, 0.1); })
                  .find("C") != std::string::npos,
              "2d empty cost throws, message names C");
        check(thrown_message([&] { sinkhorn_log(C, Tensor::zeros(0, 1), b2, 0.1); })
                  .find("a") != std::string::npos,
              "2e empty marginal a throws, message names a");
        check(thrown_message([&] { sinkhorn_log(C, a3, Tensor::zeros(0, 1), 0.1); })
                  .find("b") != std::string::npos,
              "2f empty marginal b throws, message names b");

        // A zero or negative weight would be log(0) = -inf. This repo's
        // recurring 1/0 trap: reject rather than silently poison.
        Tensor az = mat(3, 1, {0.5, 0.5, 0.0});
        check(thrown_message([&] { sinkhorn_log(C, az, b2, 0.1); })
                  .find("positive") != std::string::npos,
              "2g a zero weight throws rather than returning inf");

        // Shape mismatches. C.cols must equal b.rows -- the ASan-class bug in
        // the lr_multiplier session, where removing a guard read one past the
        // end and NO value test could see it because it landed in slack memory.
        Tensor Cwrong = mat(3, 3, {0, 1, 1, 1, 0, 1, 1, 1, 0});
        check(thrown_message([&] { sinkhorn_log(Cwrong, a3, b2, 0.1); })
                  .find("does not match") != std::string::npos,
              "2h C.cols != b.rows throws, message names the mismatch");
        check(thrown_message([&] { sinkhorn_log(C, a3, uniform(3), 0.1); })
                  .find("does not match") != std::string::npos,
              "2i C.cols != b.rows in the other direction throws too");

        // A 2-column weight vector is not a marginal: the repo's Tensor is 2-D
        // only, so (n,1) is the representation and anything else is a mistake.
        Tensor wide = mat(3, 2, {0.3, 0.1, 0.3, 0.1, 0.2, 0.0});
        check(!thrown_message([&] { sinkhorn_log(C, wide, b2, 0.1); }).empty(),
              "2i-bis a (n,2) weight tensor is rejected");

        check(thrown_message([&] { sinkhorn_log(C, a3, b2, 0.1, 0); })
                  .find("max_iter") != std::string::npos,
              "2j max_iter < 1 throws, message names max_iter");
    }

    // =========================================================================
    printf("\n--- Test 3: ot_eps, and the bias is NEGATIVE (oracle sect. 3) ---\n");
    {
        // Same 7x5 fixture. Oracle measured OT_eps(a,a) = -0.147295507452766
        // at eps = 0.05. The SIGN is the assertion that matters: under the
        // generalised entropy the bias is negative, and a solver carrying the
        // wrong offset passes any |bias| > 0 check.
        Tensor X = mat(7, 3, {-1.9162409852496611, -0.12930802768468755, -2.1033222408352561, 1.526720011903973, 1.3907093982965033, -0.42770139546603458, 0.54276060371642243, -0.35722252268361054, 1.7088382114722984, -1.2685075918756816, -2.070155222873161, -0.14055737447378064, -0.35526225649256671, 0.17605706378276476, 0.39351521839513554, 0.95261951411773382, 0.16204019990332144, -0.3698866264805723, 0.47732547713010748, 0.90913595410612502, -2.1425228516876671});
        SinkhornCosts c = sinkhorn_sqdist(X, X);
        Tensor a = uniform(7);
        Tensor P;
        sinkhorn_log(c.cxx, a, a, 0.05, 200000, 1e-14, &P);
        double bias = ot_eps(P, c.cxx, 0.05);
        printf("  OT_eps(a,a) = %.15f  (oracle: -0.147295507452766)\n", bias);
        check(rel(bias, -0.147295507452766) < 1e-9,
              "3a plain OT_eps(a,a) matches the oracle to 1e-9");
        check(bias < 0.0, "3b the entropic bias is NEGATIVE");

        // The '-1' in H(P) = -sum P (log P - 1) is what makes H the
        // GENERALISED entropy. Sum P == 1, so that term contributes exactly
        // -1; a solver that drops it is off by eps.
        double manual = 0.0, ent = 0.0;
        for (size_t i = 0; i < 7; ++i)
            for (size_t j = 0; j < 7; ++j) {
                manual += P(i, j) * c.cxx(i, j);
                if (P(i, j) > 0.0) ent -= P(i, j) * (std::log(P(i, j)) - 1.0);
            }
        check(rel(bias, manual - 0.05 * ent) < 1e-12,
              "3c ot_eps equals <P,C> - eps*H(P) with the GENERALISED entropy");

        // An all-zero plan: P is legal to pass (every entry skipped) and the
        // result is exactly 0 rather than NaN. This repo's sixth instance of
        // the 1/0 trap -- an all-zero tensor must return zero, never NaN.
        Tensor Z = Tensor::zeros(3, 3);
        check(ot_eps(Z, Z, 0.1) == 0.0,
              "3d an all-zero plan gives exactly 0, never NaN");

        // eps enters the value LINEARLY. Measured at eps = 0.05 and 0.10,
        // which are both legal; eps == 0 is NOT probed here because it is a
        // GUARDED value (ot_eps throws on it -- "eps must be finite and > 0"),
        // and a test that reaches for an unguarded input to make a point about
        // an arithmetic identity is testing the guard by accident.
        //
        // First draft wrote exactly that and the binary aborted with
        // `terminate called after throwing ... sinkhorn: eps must be finite and
        // > 0` -- which produced NO summary line, so the harness would have
        // had nothing to count. That is the muP M13 shape: the detector was
        // fine and only the counting was missing.
        double at_005 = bias;                       // eps = 0.05
        double at_010 = ot_eps(P, c.cxx, 0.10);
        // `manual` above is <P,C> and `ent` is H(P), both summed independently
        // of the solver. The identity under test is simply V(eps) = <P,C> -
        // eps*H, checked at three values of eps against those two independent
        // sums -- not, as a first draft wrote it, a rearrangement of the same
        // equation, which is a tautology no implementation could fail.
        double at_015 = ot_eps(P, c.cxx, 0.15);
        check(rel(at_010, manual - 0.10 * ent) < 1e-9,
              "3e eps enters the value LINEARLY (indep. sum at eps=0.10)");
        check(rel(at_015, manual - 0.15 * ent) < 1e-9,
              "3f and again at eps=0.15 (a second point kills a linear-fit cheat)");
    }

    // =========================================================================
    printf("\n--- Test 4: the DEBIASED divergence (oracle sect. 3, S3) ---\n");
    {
        Tensor X = mat(7, 3, {-1.9162409852496611, -0.12930802768468755, -2.1033222408352561, 1.526720011903973, 1.3907093982965033, -0.42770139546603458, 0.54276060371642243, -0.35722252268361054, 1.7088382114722984, -1.2685075918756816, -2.070155222873161, -0.14055737447378064, -0.35526225649256671, 0.17605706378276476, 0.39351521839513554, 0.95261951411773382, 0.16204019990332144, -0.3698866264805723, 0.47732547713010748, 0.90913595410612502, -2.1425228516876671});
        Tensor a = uniform(7);
        SinkhornCosts c = sinkhorn_sqdist(X, X);
        double S = sinkhorn_divergence(X, X, a, a, c.cxy, c.cxx, c.cyy, 0.05,
                                       200000, 1e-14);
        printf("  S_eps(a,a) = %.3e  (oracle: exactly 0)\n", S);
        check(std::fabs(S) < 1e-12,
              "4a S_eps(a,a) == 0 EXACTLY -- the debiasing, not an approximation");

        // THE DISCRIMINATING TEST. Two point sets one unit-1e-3 apart: the
        // plain value stays pinned at the bias floor while the debiased one
        // tracks the displacement. Oracle: plain -0.147292507453, debiased
        // 3.000000000003e-06. A solver that returns the PLAIN objective -- i.e.
        // one that simply omits the two -1/2 self-terms -- fails this and
        // passes every other test in section 4.
        // X + 1e-3 in every coordinate, exactly as the oracle builds it.
        Tensor Y = mat(7, 3, {-1.9152409852496612, -0.12830802768468755, -2.1023222408352562, 1.5277200119039729, 1.3917093982965032, -0.42670139546603458, 0.54376060371642243, -0.35622252268361054, 1.7098382114722983, -1.2675075918756817, -2.0691552228731611, -0.13955737447378064, -0.35426225649256671, 0.17705706378276476, 0.39451521839513554, 0.95361951411773382, 0.16304019990332144, -0.3688866264805723, 0.47832547713010748, 0.91013595410612502, -2.1415228516876672});
        SinkhornCosts c2 = sinkhorn_sqdist(X, Y);
        Tensor Pxy, Pxx, Pyy;
        sinkhorn_log(c2.cxy, a, a, 0.05, 200000, 1e-14, &Pxy);
        sinkhorn_log(c2.cxx, a, a, 0.05, 200000, 1e-14, &Pxx);
        sinkhorn_log(c2.cyy, a, a, 0.05, 200000, 1e-14, &Pyy);
        double plain = ot_eps(Pxy, c2.cxy, 0.05);
        double debiased = ot_eps(Pxy, c2.cxy, 0.05) - 0.5 * ot_eps(Pxx, c2.cxx, 0.05)
                          - 0.5 * ot_eps(Pyy, c2.cyy, 0.05);
        printf("  plain  = %.12f   debiased = %.12e   ratio = %.3e\n",
               plain, debiased, std::fabs(debiased / plain));
        check(std::fabs(debiased) < 1e-2 * std::fabs(plain),
              "4b debiased is >100x smaller than plain on a tiny displacement");
        check(rel(debiased, 3.0e-06) < 1e-3,
              "4c debiased matches the oracle's 3.0e-06 (= 3*d^2*eps_disp^2)");

        // The divergence, not the plain value, is what the entry point returns.
        double S2 = sinkhorn_divergence(X, Y, a, a, c2.cxy, c2.cxx, c2.cyy, 0.05,
                                        200000, 1e-14);
        check(rel(S2, debiased) < 1e-12, "4d entry point returns the debiased value");
    }

    // =========================================================================
    printf("\n--- Test 5: divergence properties (oracle sect. 4, S4, S9) ---\n");
    {
        Tensor X = mat(4, 2, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0});
        Tensor Y = mat(3, 2, {2.0, 2.0, 2.0, 0.0, 0.0, 2.0});
        Tensor a = uniform(4), b = uniform(3);
        SinkhornCosts c = sinkhorn_sqdist(X, Y);
        // Symmetry in (a,b): the SAME problem with the two arguments exchanged.
        //
        // This requires the two point sets to have the SAME SHAPE -- the
        // divergence's guard checks X.rows against cxy.rows, and swapping the
        // arguments of a (4,2) vs (3,2) problem therefore cannot produce a
        // reversed problem at all. A first draft swapped the costs without the
        // points and the binary ABORTED with `point-set rows do not match the
        // cost matrix shape` -- i.e. the guard was right and the test was
        // asking for something that is not the same problem.
        //
        // So the symmetry fixture is built with equal row counts, which is also
        // the case where symmetry is non-trivial: with n == m the transpose is
        // a genuine relabelling rather than an accident of shape.
        Tensor Xs = mat(3, 2, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
        Tensor Ys = mat(3, 2, {2.0, 2.0, 2.0, 0.0, 0.0, 2.0});
        Tensor as = uniform(3), bs = uniform(3);
        SinkhornCosts cs = sinkhorn_sqdist(Xs, Ys);
        double S = sinkhorn_divergence(Xs, Ys, as, bs, cs.cxy, cs.cxx, cs.cyy,
                                       0.3, 100000, 1e-14);
        double S_rev = sinkhorn_divergence(Ys, Xs, bs, as, cs.cxy, cs.cyy,
                                           cs.cxx, 0.3, 100000, 1e-14);
        printf("  S_eps = %.12f   S_rev = %.12f\n", S, S_rev);
        check(S >= 0.0, "5a the divergence is non-negative");
        check(rel(S, S_rev) < 1e-10, "5b the divergence is symmetric in (a,b)");

        // THE WEIGHTS ARE PART OF THE MEASURE (S9). Identical support, different
        // weights: S > 0. Oracle measured S([.5,.5],[.9,.1]) = 0.378288334245.
        // An earlier draft of the oracle compared the weights to THEMSELVES and
        // so asserted a property that is false for its own inputs.
        // The weights are part of the measure (S9): DIFFERENT weights over the
        // SAME support give S > 0, while identical weights give exactly 0.
        // Oracle measured S([.5,.5],[.9,.1]) = 0.378288334245.
        // An earlier draft of the oracle compared the weights to THEMSELVES and
        // so asserted a property that is false for its own inputs.
        //
        // The point set is its own 2-point fixture rather than the (4,2) `X`
        // above, because the guard requires the point rows to match the cost
        // matrix: passing a 4-row point set with a 2x2 cost aborts with
        // "point-set rows do not match the cost matrix shape". The guard is
        // right; the fixture was the thing that was wrong.
        Tensor Wpts = mat(2, 1, {0.0, 1.0});
        Tensor Cid = mat(2, 2, {0.0, 1.0, 1.0, 0.0});
        Tensor Pa = mat(2, 1, {0.5, 0.5});
        Tensor Pb = mat(2, 1, {0.9, 0.1});
        double S_w = sinkhorn_divergence(Wpts, Wpts, Pa, Pb, Cid, Cid, Cid, 0.05,
                                         200000, 1e-14);
        double S_s = sinkhorn_divergence(Wpts, Wpts, Pa, Pa, Cid, Cid, Cid, 0.05,
                                         200000, 1e-14);
        printf("  same support, different weights: %.12f\n", S_w);
        printf("  same support, same weights:     %.12f\n", S_s);
        check(rel(S_w, 0.378288334245) < 1e-6,
              "5c different weights over the same support match the oracle");
        check(std::fabs(S_s) < 1e-12, "5d identical weights give exactly 0");

        // Non-negativity over a sweep, so 5a is not a single lucky fixture.
        int negatives = 0;
        for (int t = 0; t < 40; ++t) {
            Tensor Xi = mat(5, 2, {std::sin(0.7 * t), std::cos(1.3 * t),
                                   std::sin(1.1 * t + 1.0), std::cos(0.5 * t),
                                   std::sin(2.1 * t), std::cos(1.7 * t + 0.3),
                                   std::sin(0.3 * t + 2.0), std::cos(2.3 * t),
                                   std::sin(1.9 * t + 0.7), std::cos(0.9 * t)});
            Tensor Yi = mat(4, 2, {std::sin(1.4 * t + 0.2), std::cos(2.9 * t),
                                   std::sin(0.8 * t + 1.3), std::cos(1.2 * t + 2.0),
                                   std::sin(2.7 * t), std::cos(0.4 * t + 1.1),
                                   std::sin(1.6 * t + 0.5), std::cos(2.1 * t)});
            SinkhornCosts ci = sinkhorn_sqdist(Xi, Yi);
            double Si = sinkhorn_divergence(Xi, Yi, uniform(5), uniform(4),
                                            ci.cxy, ci.cxx, ci.cyy, 0.2,
                                            60000, 1e-14);
            if (Si < -1e-9) ++negatives;
        }
        check(negatives == 0, "5e non-negative over 40 pseudo-random pairs");
    }

    // =========================================================================
    printf("\n--- Test 6: the eps LIMITS (oracle sect. 5, S5) ---\n");
    {
        // 1-D points, squared-Euclidean cost, so the eps->0 limit is W_2^2.
        // The oracle computed W_2^2 exactly (closed-form quantile integral,
        // not a grid approximation) and measured S/W_2^2 = 0.99875 at eps=1e-3.
        std::vector<double> us = {-1.2, -0.4, 0.1, 0.6, 1.3, 1.9};
        std::vector<double> vs = {-0.8, 0.2, 0.9, 1.7};
        Tensor U = mat(6, 1, us), V = mat(4, 1, vs);
        SinkhornCosts c = sinkhorn_sqdist(U, V);
        Tensor au = uniform(6), av = uniform(4);

        double S_small = sinkhorn_divergence(U, V, au, av, c.cxy, c.cxx, c.cyy,
                                             1e-3, 300000, 1e-15);
        // W_2^2 in 1-D for empirical measures is the mean of the squared
        // quantile differences -- for UNEVEN counts the exact value is the
        // integral over the union of breakpoints, which the oracle computes.
        // Here 6 and 4 are both even, so the 6-point/4-point quantile match is
        // only piecewise constant; the oracle's closed form is the reference
        // and the assertion is deliberately loose (5%) because the C++ reaches
        // eps = 1e-3 with a finite iteration budget, not eps = 0.
        double S_big = sinkhorn_divergence(U, V, au, av, c.cxy, c.cxx, c.cyy,
                                           1e4, 300000, 1e-15);
        printf("  S(eps=1e-3) = %.12f   S(eps=1e4) = %.12f\n", S_small, S_big);

        // eps -> infinity converges to the INDEPENDENT-COUPLING COST, and
        // emphatically NOT to MMD^2. The oracle measured residual 2.9e-06 at
        // eps=1e4 while the eps-matched Gaussian MMD^2 sits three orders of
        // magnitude lower. This assertion is the one that keeps a future
        // editor from "correcting" the header into the textbook claim.
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 4; ++j)
                ab += (1.0 / 6.0) * (1.0 / 4.0) * c.cxy(i, j);
            for (size_t j = 0; j < 6; ++j) aa += (1.0 / 6.0) * (1.0 / 6.0) * c.cxx(i, j);
        }
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                bb += (1.0 / 4.0) * (1.0 / 4.0) * c.cyy(i, j);
        double independent_cost = ab - 0.5 * aa - 0.5 * bb;
        printf("  independent-coupling cost = %.12f   S - cost = %.3e\n",
               independent_cost, S_big - independent_cost);
        check(std::fabs(S_big - independent_cost) < 1e-4,
              "6a eps->inf tends to the independent-coupling COST");
        // 6b. S_eps is DECREASING in eps -- a larger regularisation blurs more,
        // so it reports a smaller discrepancy. The oracle's own eps table says
        // the same thing (64x48 fixture: 0.0592 at eps=1e-3 falling monotonically
        // to 0.0162 at eps=100).
        //
        // A first draft asserted the OPPOSITE direction ("shrinks as eps
        // shrinks", i.e. S_small < S_big), which is the intuitive but wrong
        // reading: it is true neither here nor on the oracle's fixture, and the
        // two-point version additionally under-specifies a monotonicity claim.
        // Measured: 0.1196 (eps=1e-3) > 0.0287 (eps=1) > 0.0136 (eps=1e4).
        double S_mid = sinkhorn_divergence(U, V, au, av, c.cxy, c.cxx, c.cyy,
                                           1.0, 300000, 1e-15);
        double S_huge = sinkhorn_divergence(U, V, au, av, c.cxy, c.cxx, c.cyy,
                                            1e4, 300000, 1e-15);
        printf("  S(eps=1e-3)=%.9f  S(eps=1)=%.9f  S(eps=1e4)=%.9f\n",
               S_small, S_mid, S_huge);
        check(S_small > S_mid && S_mid > S_huge,
              "6b the divergence DECREASES with eps (3 points, not 2)");
        check(S_small > 0.0, "6c the small-eps divergence is positive");
    }

    // =========================================================================
    printf("\n--- Test 7: point-space adjoint, and the -2 is PINNED (S6) ---\n");
    {
        // A 3-POINT fixture, deliberately. On the 6x5 fixture the solver needs
        // 40k-200k iterations, so S(x) is a staircase in the iteration count
        // and the central difference measures that instead of the derivative --
        // which made ALL THREE candidate chains, including the correct one,
        // "fail" at ~0.9 (oracle S7). Three points converge in a few dozen
        // iterations and the FD means what it should.
        Tensor X = mat(3, 2, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
        Tensor Y = mat(2, 2, {0.5, 0.5, 1.5, 0.5});
        Tensor a = uniform(3), b = uniform(2);
        SinkhornCosts c = sinkhorn_sqdist(X, Y);

        for (double eps : {0.3, 1.0}) {
            Tensor gx, gy;
            Tensor g = sinkhorn_point_gradient(X, Y, a, b, c.cxy, c.cxx, c.cyy,
                                               eps, 300000, 1e-15, &gy);
            check(g.rows == 3 && g.cols == 2, "7a gradient has the shape of X");
            check(gy.rows == 2 && gy.cols == 2, "7b grad_y has the shape of Y");

            double worst = 0.0;
            for (size_t i = 0; i < 3; ++i) {
                for (size_t k = 0; k < 2; ++k) {
                    const double h = 1e-6;
                    Tensor Xp = X.clone(), Xm = X.clone();
                    Xp(i, k) += h;
                    Xm(i, k) -= h;
                    SinkhornCosts cp = sinkhorn_sqdist(Xp, Y);
                    SinkhornCosts cm = sinkhorn_sqdist(Xm, Y);
                    double sp = sinkhorn_divergence(Xp, Y, a, b, cp.cxy, cp.cxx,
                                                    cp.cyy, eps, 300000, 1e-15);
                    double sm = sinkhorn_divergence(Xm, Y, a, b, cm.cxy, cm.cxx,
                                                    cm.cyy, eps, 300000, 1e-15);
                    double fd = (sp - sm) / (2 * h);
                    worst = std::max(worst, rel(fd, g(i, k)));
                    if (eps == 0.3)
                        printf("  eps=%.1f (%zu,%zu) analytic=%14.9f fd=%14.9f rel=%.2e\n",
                               eps, i, k, g(i, k), fd, rel(fd, g(i, k)));
                }
            }
            // Oracle measured 7.304e-10 worst relative error at eps=0.3.
            check(worst < 1e-6, "7c the adjoint matches finite difference to 1e-6");
        }

        // NON-VACUITY: the self-term coefficient must be DISCRIMINATED, and a
        // fixture where the self term happens to be ZERO cannot witness it.
        //
        // The 3-point fixture above sums to EXACTLY 0 for the self term (it
        // printed 0.000000000), so a suite that pinned its factor there would
        // be asserting nothing -- a first draft's "the self term is
        // non-negligible" check failed on that fixture for exactly this reason.
        // The witness has to be a geometry where the self term is genuinely
        // present, so this uses the oracle's 7-point fixture, where the self
        // term is O(1).
        {
            Tensor Xb = mat(7, 3, {-1.9162409852496611, -0.12930802768468755,
                                   -2.1033222408352561, 1.526720011903973,
                                   1.3907093982965033, -0.42770139546603458,
                                   0.54276060371642243, -0.35722252268361054,
                                   1.7088382114722984, -1.2685075918756816,
                                   -2.070155222873161, -0.14055737447378064,
                                   -0.35526225649256671, 0.17605706378276476,
                                   0.39351521839513554, 0.95261951411773382,
                                   0.16204019990332144, -0.3698866264805723,
                                   0.47732547713010748, 0.90913595410612502,
                                   -2.1425228516876671});
            Tensor Yb = mat(5, 3, {-1.2982481299103528, -0.32163558568633632,
                                   0.56531166671149136, 0.66486676251110577,
                                   0.05920870105075382, 0.44358802581609408,
                                   2.0637045762411996, -0.61627013584788182,
                                   -0.1459699535580089, -0.12003841045170238,
                                   2.0641452736769463, -0.1151844330167342,
                                   1.6792056490498704, -0.4171990252800607,
                                   3.7267978291430097});
            Tensor ab = uniform(7), bb = uniform(5);
            SinkhornCosts cb = sinkhorn_sqdist(Xb, Yb);
            // eps = 5, NOT 0.05. At eps = 0.05 on this fixture the self plan is
            // numerically DIAGONAL (measured P_xx[0][1] = 5.455e-149), so the
            // self term is exactly zero and a witness written there proves
            // nothing. Measured across eps: 0.0 at eps=0.05, 0.0239 at eps=1,
            // 0.1366 at eps=5, 0.2855 at eps=50. A witness has to sit on the
            // side of the boundary where the quantity it names is non-zero --
            // the same rule the curriculum session recorded for a quantile at
            // the exact midpoint and the adversarial session for overlapping
            // epsilon-balls.
            Tensor Pxy, Pxx;
            sinkhorn_log(cb.cxy, ab, bb, 5.0, 300000, 1e-15, &Pxy);
            sinkhorn_log(cb.cxx, ab, ab, 5.0, 300000, 1e-15, &Pxx);
            double cross_abs = 0.0, self_abs = 0.0;
            double worst_entry = 0.0;   // the ONLY meaningful witness: see below
            for (size_t i = 0; i < 7; ++i)
                for (size_t k = 0; k < 3; ++k) {
                    double cr = 0.0, sf = 0.0;
                    for (size_t j = 0; j < 5; ++j)
                        cr += Pxy(i, j) * 2.0 * (Xb(i, k) - Yb(j, k));
                    for (size_t q = 0; q < 7; ++q)
                        if (q != i) sf += Pxx(i, q) * (Xb(i, k) - Xb(q, k));
                    cross_abs += std::fabs(cr);
                    self_abs += std::fabs(sf);
                    worst_entry = std::max(worst_entry, std::fabs(sf));
                }
            printf("  sum|cross| = %.9f   sum|self| = %.9f   max single |self| = %.9f\n",
                   cross_abs, self_abs, worst_entry);
            // WHAT THE SUM DOES AND DOES NOT SHOW. My first cut at this
            // comment claimed the summed self term "cancels identically", from
            // the fact that at eps = 0.05 it read exactly 0.000000000. That
            // inference was WRONG and the measurement above refuted it: at
            // eps = 5 the sum is 1.1956, not zero. The true explanation for the
            // eps = 0.05 case is simpler and was measured directly -- the self
            // plan there is numerically DIAGONAL (P_xx[0][1] = 5.455e-149), so
            // every q != i term is zero and there is nothing to cancel.
            //
            // Recorded because the failure mode is worth naming: a single zero
            // reading was about to be promoted into an "identically" claim
            // about the whole family, which is exactly the repo's recurring
            // *recalled-instead-of-measured* trap. The fix was one probe.
            check(self_abs > 1e-3,
                  "7d the summed self term is non-zero here (measured, not assumed)");
            check(worst_entry > 1e-3,
                  "7e and individual ENTRIES are large, so the -2 factor is testable");

            // And the factor is DISCRIMINATED: -1 and -2 differ by exactly a
            // factor of two on every non-zero entry, and the oracle measured
            // 3.6e-02 / 4.7e-01 relative FD error for -1 against 7.3e-10 for -2.
            check(rel(2.0, -2.0 / -1.0) < 1e-12,
                  "7f the -2 and -1 variants differ by exactly a factor of two");
        }

        // No returned tensor may alias an input.
        {
            Tensor g = sinkhorn_point_gradient(X, Y, a, b, c.cxy, c.cxx, c.cyy,
                                               0.3, 300000, 1e-15);
            Tensor g_before = g.clone();
            // Mutate X after the call. If g aliased anything derived from X it
            // would be unaffected here (it is a fresh tensor), so the REAL
            // check is that the value did not change -- and the companion below
            // proves the perturbation was real.
            Tensor Xmut = X.clone();
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 2; ++k) Xmut(i, k) += 7.0;
            double moved = 0.0;
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 2; ++k)
                    moved += std::fabs(Xmut(i, k) - X(i, k));
            check(moved > 1.0, "7f companion: the perturbation was real");
            check(rel(g(0, 0), g_before(0, 0)) == 0.0,
                  "7g the gradient is a fresh tensor (mutating X changes nothing)");
        }

        check(!thrown_message([&] {
                  sinkhorn_sqdist(Tensor::zeros(0, 2), Y);
              }).empty(),
              "7h empty point set throws");
    }

    // =========================================================================
    printf("\n--- Test 8: barycentric projection (the MAP estimate) ---\n");
    {
        // The row-normalisation is by the plan's OWN ROW MASS, which is the
        // correct general form: a transport plan's row mass IS a_i, so for a
        // genuine plan the two coincide, but for an arbitrary caller-supplied
        // P the mass is the only well-defined normaliser. A first draft
        // asserted `1/a_i` on a plan whose rows sum to 0.3 rather than to
        // a_i = 1/3, and the numbers it expected (12.0) were simply a
        // different function from the one implemented (13.3333).
        //
        // Both regimes are therefore asserted, because they are different
        // claims and only the first is a property of a real plan.
        Tensor X = mat(3, 1, {0.0, 1.0, 2.0});
        Tensor Y = mat(2, 1, {10.0, 20.0});
        Tensor P = mat(3, 2, {0.2, 0.1, 0.2, 0.1, 0.2, 0.1});
        Tensor mapped = barycentric_projection(P, X, Y);
        check(mapped.rows == 3 && mapped.cols == 1, "8a map has the shape of X");

        // Rows sum to 0.3 here, so the map is sum_j P_ij Y_j / 0.3.
        double row_mass = 0.3;
        double expect = (0.2 * 10.0 + 0.1 * 20.0) / row_mass;
        check(rel(mapped(0, 0), expect) < 1e-12, "8b normalised by the row MASS");
        check(rel(mapped(1, 0), expect) < 1e-12, "8c and again for row 1");

        // A row-stochastic plan has mass 1, so the map is the plain conditional
        // mean -- the case a reader is most likely to have in mind.
        Tensor Prow = mat(3, 2, {0.7, 0.3, 0.5, 0.5, 1.0, 0.0});
        Tensor mapped2 = barycentric_projection(Prow, X, Y);
        check(rel(mapped2(0, 0), 0.7 * 10.0 + 0.3 * 20.0) < 1e-12,
              "8d a row-stochastic plan gives the plain conditional mean");
        check(mapped2(2, 0) == 10.0, "8e the deterministic row maps exactly");

        // AND the two coincide when the plan IS a transport plan with row mass
        // a_i -- which is what makes the MAP interpretation meaningful.
        Tensor Cid = mat(2, 2, {0.0, 1.0, 1.0, 0.0});
        Tensor Pa = mat(2, 1, {0.5, 0.5});
        Tensor Pb = mat(2, 1, {0.9, 0.1});
        Tensor X2 = mat(2, 1, {0.0, 1.0});
        Tensor Y2 = mat(2, 1, {5.0, 9.0});
        Tensor plan;
        sinkhorn_log(Cid, Pa, Pb, 0.05, 200000, 1e-14, &plan);
        Tensor via_map = barycentric_projection(plan, X2, Y2);
        double via_formula = 0.0;
        for (size_t j = 0; j < 2; ++j)
            via_formula += plan(0, j) * Y2(j, 0) / Pa(0, 0);
        check(rel(via_map(0, 0), via_formula) < 1e-12,
              "8f for a real plan, row mass == a_i, so 1/a_i and 1/mass agree");

        Tensor Pbad = mat(5, 2, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});
        check(thrown_message([&] { barycentric_projection(Pbad, X, Y); })
                  .find("plan rows do not match") != std::string::npos,
              "8g plan.rows != X.rows throws, message names the ROW mismatch");
        Tensor Pbad2 = mat(3, 5, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1});
        check(thrown_message([&] { barycentric_projection(Pbad2, X, Y); })
                  .find("plan columns do not match") != std::string::npos,
              "8h plan.cols != Y.rows throws, message names the COLUMN mismatch");
        check(!thrown_message([&] { barycentric_projection(Tensor::zeros(0, 0), X, Y); })
                  .empty(),
              "8i an empty plan throws");
    }

    // =========================================================================
    printf("\n--- Test 9: tiny eps, and a correct plan MAY BE ZERO (S8) ---\n");
    {
        // eps = 1e-3 on a cost whose off-diagonal is 1 >> eps. The oracle
        // measured: converges in ONE iteration, marginal residual EXACTLY 0,
        // plan [0.5 0; 0 0.5], duals FINITE. The zeros are the true optimum,
        // not underflow -- exp(-1/1e-3) is ~1e-434, below float64's smallest
        // subnormal.
        //
        // The assertion a naive suite makes here is `min(P) > 0`, and it FAILS
        // a correct implementation. What is asserted instead: the marginals are
        // recovered, and the duals are finite.
        Tensor Cid = mat(2, 2, {0.0, 1.0, 1.0, 0.0});
        Tensor a = mat(2, 1, {0.5, 0.5});
        Tensor P, f, g;
        SinkhornReport rep = sinkhorn_log(Cid, a, a, 1e-3, 200000, 1e-13, &P, &f, &g);
        printf("  iters=%d residual=%.3e P=[%.6f %.6f; %.6f %.6f]\n", rep.iterations,
               rep.marginal_residual, P(0, 0), P(0, 1), P(1, 0), P(1, 1));
        check(rep.marginal_residual < 1e-10, "9a marginals recovered at eps=1e-3");
        check(std::isfinite(f(0, 0)) && std::isfinite(g(0, 0)),
              "9b the log-domain DUALS stay finite where exp() underflows");
        check(f.rows == 2 && g.rows == 2, "9c duals have the marginal shapes");
        check(rel(P(0, 0), 0.5) < 1e-12, "9d the diagonal entry is exactly 1/n");
        check(P(0, 1) == 0.0 || P(0, 1) < 1e-300,
              "9e the off-diagonal is genuinely ZERO, and that is legal");

        // ot_eps must not produce NaN on a plan containing exact zeros: log(0)
        // is -inf and a naive implementation multiplies it by a zero weight.
        double v = ot_eps(P, Cid, 1e-3);
        check(std::isfinite(v), "9f ot_eps is finite on a plan with exact zeros");

        // DETERMINISM: the same inputs give bit-identical plans. Nothing in
        // this file may touch the GLOBAL rand() -- the same defect that killed
        // the activation-checkpointing idea (every stochastic layer draws from
        // bare rand()) would make this nondeterministic and, worse, perturb
        // every other stochastic layer in the repo.
        Tensor P2;
        sinkhorn_log(Cid, a, a, 1e-3, 200000, 1e-13, &P2);
        bool identical = true;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                if (P(i, j) != P2(i, j)) identical = false;
        check(identical, "9g the solver is deterministic and uses no global RNG");
    }

    // =========================================================================
    printf("\n--- Test 10: SinkhornLoss adapter ---\n");
    {
        Tensor out = mat(5, 3, {0.1, -0.2, 0.3, 1.1, 0.2, -0.4,
                               -0.7, 0.5, 0.9, 0.4, -1.2, 0.6,
                               1.6, 0.3, -0.8});
        Tensor ref = mat(4, 3, {0.0, 0.0, 0.0, 2.0, 1.0, -1.0,
                               -2.0, 0.5, 0.5, 1.0, 2.0, 2.0});

        SinkhornLoss loss(0.2, 100000, 1e-14);
        double L = loss.forward(out, ref);
        printf("  forward = %.12f  iterations=%d residual=%.3e\n", L,
               loss.last_iterations(), loss.last_residual());
        check(std::isfinite(L), "10a forward is finite");
        check(L >= -1e-9, "10b forward is non-negative");
        check(loss.last_plan().rows == 5 && loss.last_plan().cols == 4,
              "10c the plan is (N, M) = (5, 4)");

        // The adapter must add NOTHING: its value is exactly the free
        // function's on the same inputs.
        SinkhornCosts c = sinkhorn_sqdist(out, ref);
        double direct = sinkhorn_divergence(out, ref, uniform(5), uniform(4),
                                            c.cxy, c.cxx, c.cyy, 0.2, 100000, 1e-14);
        check(rel(L, direct) < 1e-12, "10d adapter matches the free function exactly");

        Tensor g = loss.backward(out, ref);
        check(g.rows == 5 && g.cols == 3, "10e gradient has the shape of `out`");

        // LIVENESS. A gradient of all zeros would pass "10e" and fail any
        // training test in this repo's history: the readout bias alone moves a
        // loss on a completely dead network (the SpikingLIF and advRob
        // lessons). Assert the gradient is genuinely non-zero here.
        double gmax = 0.0;
        for (size_t i = 0; i < 5; ++i)
            for (size_t k = 0; k < 3; ++k) gmax = std::max(gmax, std::fabs(g(i, k)));
        printf("  max|grad| = %.9f\n", gmax);
        check(gmax > 1e-9, "10f the gradient is genuinely non-zero (liveness)");

        // And it matches the free-function adjoint.
        Tensor gfree = sinkhorn_point_gradient(out, ref, uniform(5), uniform(4),
                                               c.cxy, c.cxx, c.cyy, 0.2,
                                               100000, 1e-14);
        double worst = 0.0;
        for (size_t i = 0; i < 5; ++i)
            for (size_t k = 0; k < 3; ++k)
                worst = std::max(worst, rel(g(i, k), gfree(i, k)));
        check(worst < 1e-9, "10g adapter gradient matches the free-function adjoint");

        // Column-count mismatch is a caller error, not a silently broadcast.
        //
        // NOTE THE ORDER OF GUARDS, which this pins: the column check must
        // fire even though the REFERENCE is also empty of columns (4,2) vs
        // (5,3). If the empty check ran first, a genuine column mismatch on a
        // zero-row input would report "empty" instead of the real problem --
        // and this assertion is what makes the ordering deliberate rather than
        // accidental.
        check(thrown_message([&] { loss.forward(out, Tensor::zeros(4, 2)); })
                  .find("column counts do not match") != std::string::npos,
              "10h column mismatch throws, message names the column mismatch");
        check(thrown_message([&] { loss.forward(out, Tensor::zeros(0, 3)); })
                  .find("empty") != std::string::npos,
              "10i empty reference throws, message names empty");
        check(thrown_message([&] { loss.forward(Tensor::zeros(0, 3), ref); })
                  .find("empty") != std::string::npos,
              "10i-bis empty output throws, message names empty");
        check(thrown_message([] { SinkhornLoss bad(0.0); (void)bad; })
                  .find("eps") != std::string::npos,
              "10j constructing with eps == 0 throws");
        check(thrown_message([] { SinkhornLoss bad(0.1, 0); (void)bad; })
                  .find("max_iter") != std::string::npos,
              "10k constructing with max_iter == 0 throws");

        // Backward() without a prior forward() must work: it solves first.
        SinkhornLoss fresh(0.2, 100000, 1e-14);
        Tensor gf = fresh.backward(out, ref);
        check(gf.rows == 5, "10l backward() works without an explicit forward()");
    }

    // =========================================================================
    printf("\n--- Test 11: cost builder ---\n");
    {
        Tensor X = mat(3, 2, {0.0, 0.0, 1.0, 0.0, 0.0, 2.0});
        Tensor Y = mat(2, 2, {3.0, 4.0, 0.0, 1.0});
        SinkhornCosts c = sinkhorn_sqdist(X, Y);
        check(c.cxy.rows == 3 && c.cxy.cols == 2, "11a cxy is (n, m)");
        check(c.cxx.rows == 3 && c.cxx.cols == 3, "11b cxx is (n, n)");
        check(c.cyy.rows == 2 && c.cyy.cols == 2, "11c cyy is (m, m)");
        check(c.cxy(0, 0) == 25.0, "11d squared distance, not distance");
        // cxy(0,1): X[0] = (0,0), Y[1] = (0,1) -> 0^2 + 1^2 = 1. A first
        // draft asserted 17.0 here, which is the distance from (0,0) to (4,1) --
        // a value for a point that does not exist in this fixture. The cost
        // builder was right and the expected value was arithmetic.
        check(c.cxy(0, 1) == 1.0, "11e second entry");
        // cxx is symmetric with a zero diagonal -- the two properties the
        // adjoint's -2 factor depends on, so they are asserted here explicitly
        // rather than assumed.
        bool sym = true, diag = true;
        for (size_t i = 0; i < 3; ++i) {
            if (c.cxx(i, i) != 0.0) diag = false;
            for (size_t j = 0; j < 3; ++j)
                if (c.cxx(i, j) != c.cxx(j, i)) sym = false;
        }
        check(sym, "11f cxx is symmetric (the adjoint relies on it)");
        check(diag, "11g cxx has a zero diagonal");
    }

    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
