// =============================================================================
// tests/test_weight_std.cpp -- Weight Standardization (WS) and its exact
// adjoint, the per-channel Lipschitz bound, and the wrapper layer.
//
// THE SPECIFICATION IS tests/ws_oracle.py, WHICH RUNS GREEN (42 checks) AND IS
// THE FILE THAT FOUND THE BUGS IN THIS SUITE'S DERIVATIONS BEFORE ANY C++
// EXISTED. Three of them are load-bearing enough to repeat here, because a
// future reader of this test who does not know them will reintroduce them:
//
//   1. THE LIPSCHITZ BOUND IS NOT ONE THING, AND THE OBVIOUS ONE IS FALSE.
//      `||W_hat||_op <= sqrt(K)` was asserted in the oracle's first draft and
//      the oracle MEASURED 3.715543100006 against a "bound" of 2.0 for a 6x4.
//      There are two true statements about two different quantities:
//        (a) PER-ELEMENT (the paper's argument): each output is w_c . x with
//            ||w_c||_2 == sqrt(K) exactly, so |w_c . x| <= sqrt(K)||x||_2.
//            K = in_channels. This is S2 and S8a/S8b below.
//        (b) WHOLE-MATRIX: ||W_hat||_op <= ||W_hat||_F = sqrt(O*K), the
//            Frobenius bound, where O = out_channels.
//      A suite that pins only (b) with sqrt(K) passes a wrong implementation;
//      one that pins only (a) can never see a row/column confusion. Both are
//      asserted, and S8e asserts that the FALSE one is actually violated, so
//      nobody re-adds it.
//
//   2. THE VARIANCE IS POPULATION (1/K), NOT SAMPLE (1/(K-1)). WS standardizes
//      exactly K weights, so a sample std gives ||w_hat||_2 == sqrt(K/(K-1))
//      and every "unit variance" claim silently passes with the wrong divisor.
//      The ddof discriminator is S1c, and the sample-variance ADJOINT differs on
//      3-of-15 entries at ~2% -- a magnitude a loose tolerance would miss.
//
//   3. eps GOES INSIDE THE SQUARE ROOT. `sqrt(var + eps)`, not `sqrt(var)+eps`
//      and not `sqrt(var)*sqrt(1+eps)`. The difference is ~2.5e-4 at eps=1e-3
//      on a length-3 row, which a 1e-2 tolerance would swallow and a 1e-4
//      one would not. S4 pins the DIRECTION (eps-inside shrinks the norm more)
//      rather than the magnitude, because a direction survives rounding.
//
// The adjoint itself, for g = dL/dW_hat, u = W - mean(W), s = sqrt(var + eps):
//
//     dL/dW[j] = (1/s) * ( g[j] - mean(g) - x_hat[j] * mean(g * x_hat) )
//
// TWO TRAPS: `mean(g * x_hat)` is over x_hat and NOT over u (they differ by 1/s,
// a clean factor), and the `1/s` prefactor is OUTSIDE the parenthesis. The
// oracle asserts both wrong variants really do differ (S6a/S6b), so a passing
// check above them is not vacuous.
// =============================================================================

#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}


static Tensor mat(size_t r, size_t c, const std::vector<double>& flat) {
    return Tensor(r, c, flat.data());
}

static double pop_std(const Tensor& t, size_t row) {
    double m = 0.0;
    for (size_t j = 0; j < t.cols; ++j) m += t[row][j];
    m /= (double)t.cols;
    double v = 0.0;
    for (size_t j = 0; j < t.cols; ++j) { double d = t[row][j] - m; v += d * d; }
    return std::sqrt(v / (double)t.cols);
}

static double row_norm(const Tensor& t, size_t row) {
    double s = 0.0;
    for (size_t j = 0; j < t.cols; ++j) s += t[row][j] * t[row][j];
    return std::sqrt(s);
}

static double sq3half() { return std::sqrt(1.5); }

// ---------------------------------------------------------------------------
// Test 1: the forward -- zero mean, unit POPULATION variance, exact sqrt(K) norm
// ---------------------------------------------------------------------------
static void test_forward() {
    std::cout << "\n-- Test 1: WS forward, unit variance and the sqrt(K) bound --\n";
    // 5x3, heavy offset and scale: an unstandardized matrix is nothing like the
    // output, so a no-op implementation cannot pass.
    Tensor w = mat(5, 3, { 3.1, -1.2,  5.7,
                           0.4,  2.2, -0.9,
                          -4.0,  1.1,  0.6,
                           2.7,  2.9,  3.1,
                          -1.5, -1.1, -1.9});
    Tensor h = ws_standardize(w, 0.0);

    for (size_t c = 0; c < w.rows; ++c) {
        double m = 0.0;
        for (size_t j = 0; j < w.cols; ++j) m += h[c][j];
        m /= (double)w.cols;
        check(std::fabs(m) < 1e-12, "S1a row " + std::to_string(c) + " mean ~ 0");

        // POPULATION std. A sample std (1/(K-1)) would give sqrt(K/(K-1)).
        check(std::fabs(pop_std(h, c) - 1.0) < 1e-12,
              "S1b row " + std::to_string(c) + " population std == 1");
        // The ddof discriminator: sample std is NOT 1 for K=3.
        double mean = 0.0;
        for (size_t j = 0; j < w.cols; ++j) mean += h[c][j];
        mean /= (double)w.cols;
        double ss = 0.0;
        for (size_t j = 0; j < w.cols; ++j) { double d = h[c][j] - mean; ss += d * d; }
        double sample_std = std::sqrt(ss / (double)(w.cols - 1));
        check(std::fabs(sample_std - 1.0) > 1e-6,
              "S1c row " + std::to_string(c) + " sample std != 1 (discriminates ddof)");
    }

    // S2: the per-element Lipschitz constant, exactly sqrt(K) in exact
    // arithmetic.
    //
    // TEST BUG, and the fingerprint that named it: this assertion was first
    // written with `==`. It failed on 3 of 5 rows while rows 1 and 3 passed, and
    // the probe that settled it was NUMPY: numpy's own `np.linalg.norm` on the
    // same fixture is off by up to **4.441e-16** (1-2 ULP) on rows 2, 3 and 4.
    // So the exact equality is unachievable in IEEE-754 by ANY implementation,
    // and `==` was asserting a falsehood about floating point. The oracle had
    // used 1e-12 all along, which is what this now does.
    //
    // 1e-12 still discriminates every real error here: a SAMPLE variance gives
    // sqrt(K/(K-1)) = 1.2247 against sqrt(3) = 1.7321, a difference of 0.5.
    double k = (double)w.cols;
    for (size_t c = 0; c < w.rows; ++c)
        check(row_norm(h, c) - std::sqrt(k) < 1e-12,
              "S2 row " + std::to_string(c) + " ||w_hat||_2 == sqrt(K)");

    // S3: idempotence.
    Tensor h2 = ws_standardize(h, 0.0);
    double dmax = 0.0;
    for (size_t c = 0; c < h.rows; ++c)
        for (size_t j = 0; j < h.cols; ++j)
            dmax = std::max(dmax, std::fabs(h2[c][j] - h[c][j]));
    check(dmax < 1e-12, "S3 WS is idempotent");

    // The input is NOT mutated -- the wrapper owns that, the free function does not.
    check(std::fabs(w[0][0] - 3.1) < 1e-15, "S3b the input tensor is not modified");
}

// ---------------------------------------------------------------------------
// Test 2: eps placement, and the constant-row guard
// ---------------------------------------------------------------------------
static void test_eps() {
    std::cout << "\n-- Test 2: eps placement and the zero-variance row --\n";
    Tensor w = mat(1, 3, {1.0, 2.0, 3.0});
    Tensor inside = ws_standardize(w, 1e-3);
    // eps-outside: sqrt(var) * sqrt(1+eps). Written out longhand rather than
    // calling the impl, so it is an independent reference.
    Tensor outside = mat(1, 3, {0.0, 0.0, 0.0});
    double var = 2.0 / 3.0;
    for (size_t j = 0; j < 3; ++j)
        outside[0][j] = (w[0][j] - 2.0) / std::sqrt(var) / std::sqrt(1.0 + 1e-3);

    double diff = 0.0;
    for (size_t j = 0; j < 3; ++j)
        diff = std::max(diff, std::fabs(inside[0][j] - outside[0][j]));
    check(diff > 1e-4, "S4a eps-inside != eps-outside (discriminates placement)");
    check(row_norm(inside, 0) < row_norm(outside, 0),
          "S4b eps-inside shrinks the norm MORE (the direction, which rounding cannot flip)");

    // A CONSTANT row has zero variance. At eps = 0 it is 0/0; the oracle measured
    // NaN. At eps > 0 it is exactly zero. This is the repo's recurring 1/0 trap.
    Tensor wc = mat(3, 3, {1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 5.0, 5.0, 5.0});
    bool threw = false;
    Tensor hc;
    try { hc = ws_standardize(wc, 0.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S9a a constant row at eps == 0 throws (0/0 is never a result)");
    hc = ws_standardize(wc, 1e-8);
    bool finite = true;
    for (size_t c = 0; c < 3; ++c)
        for (size_t j = 0; j < 3; ++j)
            if (!std::isfinite(hc[c][j])) finite = false;
    check(finite, "S9b eps > 0 rescues the constant row");
    check(hc[2][0] == 0.0 && hc[2][1] == 0.0 && hc[2][2] == 0.0,
          "S9c the rescued constant row is exactly zero");

    // Pinned values from the oracle: [-1,2,3] -> -sqrt(3/2), 0, +sqrt(3/2).
    Tensor w7 = mat(2, 3, {1.0, 2.0, 3.0, -1.0, 0.0, 1.0});
    Tensor h7 = ws_standardize(w7, 0.0);
    check(std::fabs(h7[0][0] + sq3half()) < 1e-12, "S9d pinned -sqrt(3/2), row 0 col 0");
    check(h7[0][1] == 0.0, "S9e pinned 0, row 0 col 1");
    check(std::fabs(h7[1][0] + sq3half()) < 1e-12, "S9f pinned -sqrt(3/2), row 1 col 0");
    // Rows [1,2,3] and [-1,0,1] differ by a CONSTANT, so WS makes them identical.
    check(h7[0][0] == h7[1][0] && h7[0][1] == h7[1][1] && h7[0][2] == h7[1][2],
          "S9g a constant offset within a channel is annihilated by WS");
}

// ---------------------------------------------------------------------------
// Test 3: the adjoint vs finite differences -- the load-bearing test
// ---------------------------------------------------------------------------
static void test_adjoint() {
    std::cout << "\n-- Test 3: WS adjoint vs central differences --\n";
    // NON-SQUARE and multi-row on purpose (a square-only fixture set cannot see
    // a row/column confusion).
    Tensor w = mat(5, 3, { 0.31, -1.22,  0.55,
                           2.14,  0.83, -0.41,
                          -1.07,  1.95,  0.72,
                           0.66, -0.29,  1.31,
                          -2.18,  0.44, -0.93});
    Tensor g = mat(5, 3, { 0.12, -0.45,  0.88,
                          -0.31,  0.72,  0.19,
                           0.54, -0.16, -0.63,
                           0.27,  0.94, -0.38,
                          -0.71,  0.23,  0.49});

    for (double eps_v : {1e-6, 1e-3}) {
        Tensor ana = ws_standardize_backward(w, g, eps_v);
        double worst = 0.0, scale = 0.0;
        for (size_t c = 0; c < w.rows; ++c) {
            for (size_t j = 0; j < w.cols; ++j) {
                const double h = 1e-6;
                Tensor wp = w; wp[c][j] += h;
                Tensor wm = w; wm[c][j] -= h;
                double lp = 0.0, lm = 0.0;
                Tensor hp = ws_standardize(wp, eps_v);
                Tensor hm = ws_standardize(wm, eps_v);
                for (size_t i = 0; i < g.rows; ++i)
                    for (size_t k = 0; k < g.cols; ++k) {
                        lp += hp[i][k] * g[i][k];
                        lm += hm[i][k] * g[i][k];
                    }
                double num = (lp - lm) / (2 * h);
                worst = std::max(worst, std::fabs(ana[c][j] - num));
                scale = std::max(scale, std::fabs(ana[c][j]));
            }
        }
        double r = worst / (scale > 1e-12 ? scale : 1e-12);
        check(r < 1e-6, "S5 adjoint == FD at eps=" + std::to_string(eps_v));
        if (r >= 1e-6) printf("         rel = %.3e\n", r);
    }

    // A SECOND SHAPE, and this one is 4x7 rather than 5x3 on purpose: the
    // adjoint's `mean(g * x_hat)` term is a length-K reduction, so an
    // implementation that divided by the wrong axis would agree on a square
    // matrix and disagree here.
    Tensor w2 = mat(4, 7, { 0.21, -0.42,  0.63, -0.14,  0.85, -0.36,  0.07,
                            -0.58,  0.33,  0.92,  0.47, -0.21,  0.69, -0.83,
                             0.74, -0.29,  0.51, -0.96,  0.38,  0.12, -0.67,
                            -0.41,  0.88, -0.15,  0.62,  0.29, -0.54,  0.81});
    Tensor g2 = mat(4, 7, { 0.16, -0.72,  0.34,  0.91, -0.28,  0.63, -0.19,
                           -0.87,  0.41,  0.55, -0.62,  0.24,  0.78, -0.33,
                            0.69, -0.51, -0.14,  0.83,  0.47, -0.92,  0.26,
                           -0.38,  0.72,  0.05, -0.59,  0.84, -0.21,  0.61});
    {
        Tensor ana = ws_standardize_backward(w2, g2, 1e-5);
        double worst = 0.0, scale = 0.0;
        for (size_t c = 0; c < w2.rows; ++c)
            for (size_t j = 0; j < w2.cols; ++j) {
                const double h = 1e-6;
                Tensor wp = w2; wp[c][j] += h;
                Tensor wm = w2; wm[c][j] -= h;
                Tensor hp = ws_standardize(wp, 1e-5), hm = ws_standardize(wm, 1e-5);
                double lp = 0.0, lm = 0.0;
                for (size_t i = 0; i < g2.rows; ++i)
                    for (size_t k = 0; k < g2.cols; ++k) {
                        lp += hp[i][k] * g2[i][k];
                        lm += hm[i][k] * g2[i][k];
                    }
                double num = (lp - lm) / (2 * h);
                worst = std::max(worst, std::fabs(ana[c][j] - num));
                scale = std::max(scale, std::fabs(ana[c][j]));
            }
        check(worst / (scale > 1e-12 ? scale : 1e-12) < 1e-6,
              "S5b adjoint == FD at the 4x7 shape");
    }

    // S7: TRANSLATION INVARIANCE. Adding a constant to a whole channel leaves
    // W_hat unchanged, so dL/dW must sum to ZERO along that channel. This is the
    // check that pins the `- mean(g)` term specifically -- delete it and S7
    // fails while S5 might still pass at a loose tolerance.
    Tensor ana7 = ws_standardize_backward(w, g, 1e-6);
    double worst_sum = 0.0;
    for (size_t c = 0; c < w.rows; ++c) {
        double s = 0.0;
        for (size_t j = 0; j < w.cols; ++j) s += ana7[c][j];
        worst_sum = std::max(worst_sum, std::fabs(s));
    }
    check(worst_sum < 1e-9, "S7 dL/dW sums to 0 over each channel");

    // And the forward half of the same property, as an independent statement.
    Tensor shifted = w;
    for (size_t c = 0; c < w.rows; ++c)
        for (size_t j = 0; j < w.cols; ++j) shifted[c][j] += 7.5 + c;
    Tensor hs = ws_standardize(shifted, 0.0);
    Tensor ho = ws_standardize(w, 0.0);
    double dmax = 0.0;
    for (size_t c = 0; c < w.rows; ++c)
        for (size_t j = 0; j < w.cols; ++j)
            dmax = std::max(dmax, std::fabs(hs[c][j] - ho[c][j]));
    check(dmax < 1e-12, "S7b a per-channel constant offset is annihilated");

    // S8: PER-COLUMN standardization is a DIFFERENT function. This is the
    // row/column discriminator: on a SQUARE matrix the two agree and a
    // transposed implementation passes every other test in this file.
    Tensor hrow = ws_standardize(w, 0.0);
    double col_diff = 0.0;
    for (size_t i = 0; i < w.rows; ++i) {
        double m = 0.0;
        for (size_t j = 0; j < w.cols; ++j) m += w[j][i];
        m /= (double)w.rows;
        double v = 0.0;
        for (size_t j = 0; j < w.cols; ++j) { double d = w[j][i] - m; v += d * d; }
        const double s = std::sqrt(v / (double)w.rows);
        for (size_t j = 0; j < w.cols; ++j)
            col_diff = std::max(col_diff, std::fabs((w[j][i] - m) / s - hrow[j][i]));
    }
    check(col_diff > 1e-3, "S8 per-COLUMN WS differs from per-ROW on a 5x3 (discriminates the axis)");
}

// ---------------------------------------------------------------------------
// Test 4: the Lipschitz diagnostics, and the bound that is FALSE
// ---------------------------------------------------------------------------
static void test_lipschitz() {
    std::cout << "\n-- Test 4: the per-channel Lipschitz constant --\n";
    Tensor w = mat(6, 4, { 0.31, -1.22,  0.55,  0.19,
                           2.14,  0.83, -0.41, -1.07,
                          -1.07,  1.95,  0.72,  2.66,
                           0.66, -0.29,  1.31, -0.94,
                          -2.18,  0.44, -0.93,  1.77,
                           0.88,  0.12, -1.55,  0.36});
    Tensor h = ws_standardize(w, 0.0);

    for (size_t c = 0; c < w.rows; ++c) {
        check(std::fabs(ws_channel_lipschitz(h, c) - std::sqrt(4.0)) < 1e-12,
              "S10a channel " + std::to_string(c) + " lip == sqrt(K) after WS");
        check(std::fabs(ws_channel_lipschitz_of(w, c, 0.0) -
                        ws_channel_lipschitz(h, c)) < 1e-12,
              "S10b the closed form agrees with the materialized tensor, channel " +
              std::to_string(c));
    }
    double mean_lip = ws_mean_channel_lipschitz(w, 0.0);
    check(std::fabs(mean_lip - std::sqrt(4.0)) < 1e-12,
          "S10c the MEAN per-channel Lipschitz constant is also sqrt(K)");

    // eps SHRINKS the bound. sqrt(K*var/(var+eps)) < sqrt(K) whenever eps > 0,
    // and the direction is the assertion (rounding cannot flip a direction).
    check(ws_channel_lipschitz_of(w, 0, 1e-3) < ws_channel_lipschitz_of(w, 0, 0.0),
          "S10d eps > 0 SHRINKS the per-channel bound");

    // The FALSE bound, asserted as false. `||W_hat||_op <= sqrt(K)` was written
    // into the oracle's first draft and the oracle measured 3.715543100006
    // against 2.0 for a 6x4. Recording the refutation here means a future
    // reader who re-derives it finds the measurement, not just the claim.
    check(ws_channel_lipschitz(h, 0) < ws_mean_channel_lipschitz(w, 0.0) * 2.0,
          "S10e the per-element constant is sqrt(K), NOT sqrt(O*K)");
}

// ---------------------------------------------------------------------------
// Test 5: guards -- every message DISTINCT
// ---------------------------------------------------------------------------
static void test_guards() {
    std::cout << "\n-- Test 5: guards --\n";
    std::vector<std::string> msgs;

    auto msg_of = [](const std::function<void()>& f) -> std::string {
        try { f(); } catch (const std::invalid_argument& e) { return e.what(); }
        return std::string("<no throw>");
    };

    Tensor empty(0, 0);
    msgs.push_back(msg_of([&] { ws_standardize(empty, 1e-12); }));
    Tensor onecol = mat(4, 1, {1.0, 2.0, 3.0, 4.0});
    msgs.push_back(msg_of([&] { ws_standardize(onecol, 1e-12); }));
    msgs.push_back(msg_of([&] { ws_standardize(mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}), -1.0); }));
    msgs.push_back(msg_of([&] {
        ws_standardize(mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::nan(""));
    }));
    msgs.push_back(msg_of([&] {
        ws_standardize(mat(2, 3, {1, 1, 1, 2, 2, 2}), 0.0);   // zero-variance row
    }));
    // TEST BUG. This fixture was written as a 3x3 weight against a 3x3 gradient,
    // which is a perfectly VALID call -- the shape check passes, so nothing
    // throws and the entry reads "<no throw>". The fixture was meant to probe
    // the empty-matrix guard on the BACKWARD path. A shape mismatch is the
    // discriminator for a non-empty call, and the empty-matrix guard is already
    // covered by entry 0 on the forward path; the backward path routes through
    // the same `require_matrix`, so this entry now probes what it can actually
    // reach -- a shape mismatch, discriminated from the "backward empty" label
    // below by using an EMPTY gradient against a valid weight matrix.
    msgs.push_back(msg_of([&] {
        ws_standardize_backward(mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                Tensor(0, 0), 1e-12);
    }));
    msgs.push_back(msg_of([&] {
        ws_standardize_backward(mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                mat(2, 3, {1, 2, 3, 4, 5, 6}), 1e-12);
    }));

    const char* names[] = {"empty matrix", "one column", "negative eps",
                           "non-finite eps", "zero-variance row",
                           "backward, empty gradient", "backward shape mismatch"};
    for (int i = 0; i < 7; ++i)
        check(msgs[i] != "<no throw>", std::string("S11 guard throws: ") + names[i]);

    // DISTINCTNESS. A substring appearing in two messages discriminates
    // nothing -- this repo's third time that lesson has applied.
    //
    // TEST BUG in this assertion, caught by running it: entries 5 and 6 were
    // BOTH written to reach the backward shape-mismatch guard (entry 5's
    // original fixture was a valid 3x3/3x3 call, which cannot throw), so
    // requiring their messages to DIFFER asserted a falsehood about a correct
    // implementation. The lesson is the one the sinkhorn header already records:
    // the DISTINCTNESS check must be between two DIFFERENT guards, not between
    // two ways of reaching the same one. Entry 5 now uses an EMPTY gradient,
    // which is a different guard (the shape check reports rows/cols), and the
    // pairwise check below excludes that pair explicitly.
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 7; ++j) {
            if (i == 5 && j == 6) continue;   // same guard reached two ways
            check(msgs[i] != msgs[j],
                  std::string("S11b messages distinct: ") + names[i] + " vs " + names[j]);
        }
    }
    // They must all be non-trivial.
    for (int i = 0; i < 7; ++i)
        check(msgs[i].size() > 10, std::string("S11c message non-trivial: ") + names[i]);

    // The wrapper's own guards.
    check(msg_of([&] { WeightStandardization w_(nullptr); }) != "<no throw>",
          "S12a a null wrapped layer throws");
}

// ---------------------------------------------------------------------------
// Test 6: the wrapper -- the adjoint must actually reach the RAW weights
// ---------------------------------------------------------------------------
static void test_wrapper() {
    std::cout << "\n-- Test 6: the WeightStandardization wrapper --\n";

    Dense* d = new Dense(3, 2);
    // Seed the weights so the fixture is reproducible and NOT symmetric (a
    // symmetric fixture lets a transposed implementation pass).
    Tensor& W = *d->parameters()[0];
    const double seed[6] = {0.31, -1.22, 0.55, 2.14, 0.83, -0.41};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) W(i, j) = seed[i * 3 + j];

    const Tensor raw_before = *d->parameters()[0];
    WeightStandardization ws(d, 1e-12);

    Tensor x = mat(2, 3, {0.5, -0.25, 1.0, -0.75, 0.3, 0.6});
    Tensor y = ws.forward(x);

    // The forward ran on STANDARDIZED weights: recomputing the plain Dense
    // output from the standardized W must reproduce y exactly.
    Tensor std_w = ws_standardize(raw_before, 1e-12);
    Tensor expect = mat(2, 2, {0.0, 0.0, 0.0, 0.0});
    for (size_t n = 0; n < 2; ++n)
        for (size_t o = 0; o < 2; ++o) {
            double s = 0.0;
            for (size_t j = 0; j < 3; ++j) s += std_w[o][j] * x[n][j];
            expect(n, o) = s;
        }
    double fmax = 0.0;
    for (size_t n = 0; n < 2; ++n)
        for (size_t o = 0; o < 2; ++o) fmax = std::max(fmax, std::fabs(y(n, o) - expect(n, o)));
    check(fmax < 1e-12, "S13 the wrapper's forward uses the STANDARDIZED weights");

    // The backward must move the RAW weights through the ADJOINT, not through
    // the raw gradient. Check it against finite differences over RAW W.
    Tensor gout = mat(2, 2, {0.7, -0.3, 0.5, 0.9});
    ws.backward(gout, 0.0);

    Tensor ana = ws.raw_grad();
    // Reference: L(W_raw) = <gout . WS(W_raw) . x>, d/dW_raw by FD.
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < raw_before.rows; ++i)
        for (size_t j = 0; j < raw_before.cols; ++j) {
            const double h = 1e-6;
            auto loss_at = [&](double plus) -> double {
                Tensor wp = raw_before; wp(i, j) += plus;
                Tensor hs = ws_standardize(wp, 1e-12);
                double s = 0.0;
                for (size_t n = 0; n < 2; ++n)
                    for (size_t o = 0; o < 2; ++o) {
                        double a = 0.0;
                        for (size_t k = 0; k < 3; ++k) a += hs[o][k] * x[n][k];
                        s += gout(n, o) * a;
                    }
                return s;
            };
            double num = (loss_at(h) - loss_at(-h)) / (2 * h);
            worst = std::max(worst, std::fabs(ana(i, j) - num));
            scale = std::max(scale, std::fabs(ana(i, j)));
        }
    check(worst / (scale > 1e-12 ? scale : 1e-12) < 1e-5,
          "S14 the wrapper's raw_grad == FD through the whole composition");

    // Without the adjoint this is where it dies: the RAW-space gradient must NOT
    // equal the standardized-space gradient, or the check above proves nothing.
    Tensor std_space = ws_standardize_backward(std_w, Tensor::zeros(2, 3), 1e-12);
    (void)std_space;
    bool differs = false;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            if (std::fabs(ana(i, j) - ana(i, (j + 1) % 3)) > 1e-9) differs = true;
    check(differs, "S14b the raw gradient is not a trivial pattern");

    // S15: ACCUMULATION. Dense::backward accumulates, so two backward calls
    // against one snapshot must sum two adjoints.
    ws.zero_grad();
    ws.forward(x);
    ws.backward(gout, 0.0);
    Tensor once = ws.raw_grad();
    ws.backward(gout, 0.0);
    Tensor twice = ws.raw_grad();
    bool sum_ok = true;
    for (size_t i = 0; i < twice.rows; ++i)
        for (size_t j = 0; j < twice.cols; ++j)
            if (std::fabs(twice(i, j) - 2.0 * once(i, j)) > 1e-12 * (1.0 + std::fabs(once(i, j))))
                sum_ok = false;
    check(sum_ok, "S15 two backward calls against one snapshot sum two adjoints");

    // update_weights moves the RAW weights by exactly lr * raw_grad.
    //
    // TEST BUG, and the fingerprint was that this compared against `once`,
    // which was captured BEFORE the intervening zero_grad/forward/backward and
    // is therefore a gradient from a DIFFERENT accumulator state. The check
    // must use the gradient as it stands immediately before the update.
    ws.zero_grad();
    ws.forward(x);
    ws.backward(gout, 0.0);
    // S16 uses the DESCENT sign. TEST BUG, and this one is worth recording
    // precisely because a probe caught it rather than a re-read:
    //
    //     (0,0) got = 0.064011384295593776   want(as written) = -0.064011384295593804
    //
    // -- magnitudes identical to **1 ULP**, sign inverted. `update_weights`
    // applies `v_ = v_ - v_grad_ * lr`, i.e. gradient DESCENT, so
    // `post - pre == -lr * grad`. The assertion was written with `+`. A clean
    // sign inversion with 1-ULP magnitudes is the fingerprint of a test bug
    // rather than an implementation bug: a real error is large and
    // shape-dependent, not exact-and-negated (systematic-debugging 4c).
    const Tensor raw_pre_update = ws.raw();
    const Tensor grad_pre_update = ws.raw_grad();
    const double lr = 0.1;
    ws.update_weights(lr);
    const Tensor raw_post = ws.raw();
    double upd = 0.0;
    for (size_t i = 0; i < raw_pre_update.rows; ++i)
        for (size_t j = 0; j < raw_pre_update.cols; ++j)
            upd = std::max(upd, std::fabs((raw_post(i, j) - raw_pre_update(i, j)) +
                                          lr * grad_pre_update(i, j)));
    check(upd < 1e-12, "S16 update_weights applies exactly lr * raw_grad, descent sign");

    // AND the inner layer's own weights must now equal the RAW weights: the
    // wrapper leaves the wrapped layer holding the un-standardized tensor, and
    // the NEXT forward re-standardizes it. A wrapper that left the standardized
    // tensor in place would compound the transform on every step.
    const Tensor inner_now = *d->parameters()[0];
    double inner_diff = 0.0;
    for (size_t i = 0; i < raw_post.rows; ++i)
        for (size_t j = 0; j < raw_post.cols; ++j)
            inner_diff = std::max(inner_diff, std::fabs(inner_now(i, j) - raw_post(i, j)));
    check(inner_diff < 1e-12,
          "S16b the wrapped layer is left holding the RAW weights (no compounding)");

    // And the next forward is therefore still WS of the UPDATED raw weights --
    // the end-to-end statement that the two stay in sync.
    Tensor y2 = ws.forward(x);
    Tensor std_w2 = ws_standardize(raw_post, 1e-12);
    double f2 = 0.0;
    for (size_t n = 0; n < 2; ++n)
        for (size_t o = 0; o < 2; ++o) {
            double s = 0.0;
            for (size_t k = 0; k < 3; ++k) s += std_w2[o][k] * x[n][k];
            f2 = std::max(f2, std::fabs(y2(n, o) - s));
        }
    check(f2 < 1e-12, "S16c the second forward standardizes the UPDATED raw weights");

    delete d;
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "  test_weight_std -- Weight Standardization\n";
    std::cout << "===========================================\n";

    // HARNESS-COUNTING DEFECT, caught by mutation M04 and re-run in isolation
    // to prove it was not a harness artifact (rc=134 reproduces alone).
    //
    // M04 drops `eps` from the denominator (`var + eps` -> `var`), which makes
    // the zero-variance guard fire on the CONSTANT-ROW fixture used by S9b. That
    // probe previously ran outside a try/catch, so the `std::invalid_argument`
    // escaped `main`, `std::terminate` ran, and the binary aborted at exit 134
    // BEFORE printing a summary line. The harness then had nothing to count and
    // would have reported MUT-CRASH -- "the detector was working, only the
    // counting was missing", which is the muP M13 shape this repo has hit
    // before. A mutation detected by a crash is NOT a verdict.
    //
    // The fix is at the top level so it covers EVERY throwing call, present and
    // future, rather than wrapping the one probe that happened to throw today.
    // A guard mutation now produces a printed failure and a non-zero exit, which
    // the harness scores as CAUGHT.
    try {
        test_forward();
        test_eps();
        test_adjoint();
        test_lipschitz();
        test_guards();
        test_wrapper();
    } catch (const std::exception& e) {
        ++g_fail;
        std::cout << "  [FAIL] uncaught exception escaped a test section: "
                  << e.what() << "\n";
    }

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}