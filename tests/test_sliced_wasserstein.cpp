// =============================================================================
// tests/test_sliced_wasserstein.cpp -- the sliced Wasserstein distance.
//
// THE METRIC, AND WHY IT IS BEING ADDED
// =============================================================================
//
//     SWD(P, Q) = E_{theta ~ Unif(S^{d-1})} [ W_1(P_theta, Q_theta) ]
//
// where P_theta is the 1-D projection of P onto the unit vector theta, and
// W_1 is the 1-D Wasserstein-1 distance.
//
// **THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-29).**
//
//     grep -rinE 'wasserstein' include/ tests/  ->  2 hits, BOTH in comments
//     grep -rinE 'sliced.?wasserman|energy.?distance' include/  ->  0 hits
//
// So the repo shipped NO Wasserstein machinery at all. What it had:
//
//   * `utils/sinkhorn.h` -- an ENTROPIC-REGULARISED optimal transport PLAN
//     (Sinkhorn-Knopp). It computes a transport COST between two point sets,
//     but the regularised cost is not the Wasserstein distance, it needs a
//     cost matrix over every (i, j) pair (O(n*m) memory), and it is not a
//     sample-quality metric.
//   * `utils/mmd_loss.h` -- a KERNEL two-sample statistic. A different
//     object: it is not a distance on the data manifold, it needs a kernel
//     bandwidth, and it says nothing about WHICH samples differ.
//   * `utils/linear_algebra.h` -- the Frechet distance (shipped this
//     session), the FID building block, which needs an Inception network's
//     pool3 features. **This repo ships no Inception network**, so FID
//     cannot be completed here.
//
// **FIFTEEN GENERATIVE MODELS SHIP (DDPM, consistency models, flow matching,
// RealNVP, neural spline flows, VQ-VAE, VAE, WGAN-GP, PixelCNN, CapsNet,
// DiT, ...) AND NOTHING MEASURES THE QUALITY OF THEIR SAMPLES IN THE DATA
// SPACE.** The sliced Wasserstein distance is the metric that fills that
// gap WITHOUT a pretrained feature extractor, WITHOUT a kernel bandwidth
// and WITHOUT an O(n*m) cost matrix.
//
// THE SPECIFICATION IS tests/sliced_wasserstein_oracle.py (float64 numpy),
// which was run BEFORE this file existed and caught FOUR defects in its own
// first draft. See the oracle's header for what those were; the two that
// matter most to the C++ are:
//
//   1. THE SORTING IS LOAD-BEARING, AND OMITTING IT OVERESTIMATES.
//      The 1-D W1 is the L1 distance between the SORTED projections.
//      Sorting MINIMISES the sum of absolute differences -- that is what
//      optimal 1-D transport IS -- so an implementation that forgets to
//      sort returns a value that is TOO LARGE. Measured over 20000 random
//      pairs: the unsorted value was smaller in ZERO of them.
//
//   2. THE PROJECTIONS MUST BE ONTO UNIT DIRECTIONS. A projection onto a
//      non-unit direction scales the answer by |theta|, which is not the
//      metric the definition names.
//
// WHY THE FIXTURES ARE NOT RANDOM
// =============================================================================
//
// A RANDOM fixture would not be checkable: the directions would be random
// too, so there would be no independent value to compare against. So the
// discriminating fixtures are FIXED:
//
//   * the 1-D fixture (d = 1) collapses the whole sliced machinery onto the
//     single unit direction, so the SWD IS the 1-D Wasserstein distance,
//     which the test computes INDEPENDENTLY from the CDF form
//     `integral |F_P - F_Q| dt`;
//   * the unequal-n fixture exercises the general 1-D W1, where the
//     tempting `mean(|sorted_p - sorted_q|)` shortcut is simply WRONG;
//   * the equal-n fixture is where the shortcut is valid, so the test
//     asserts the two forms agree there -- which is what pins the general
//     form to be the general one.
//
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// ---------------------------------------------------------------------------
// A guarded call.
//
// **THE MUTATION HARNESS FOUND THIS THE HARD WAY.** The mutation "the random
// directions are not normalised" makes the unit check fire, and the
// resulting std::invalid_argument ESCAPED main -- the binary aborted with
// rc=134, the harness reported MUT-CRASH, and the whole suite's verdict
// count was lost. That is this repo's recorded defect class: "every
// throwing call in a test is inside call_guarded, or an inverted guard
// ABORTS the binary and is reported MUT-CRASH with nothing countable".
//
// Every call that CAN throw is wrapped here, so a defect becomes a FAIL
// with a count instead of an abort with nothing.
// ---------------------------------------------------------------------------
template <typename F>
static bool call_guarded(F fn, double* out) {
    try {
        *out = fn();
        return true;
    } catch (const std::exception& e) {
        printf("    [THREW] %s\n", e.what());
        return false;
    }
}

static double rel_err(double got, double want) {
    double denom = std::max(std::abs(got), std::abs(want));
    if (denom < 1e-12) return std::abs(got - want);
    return std::abs(got - want) / denom;
}

// ---------------------------------------------------------------------------
// An INDEPENDENT 1-D WASSERSTEIN-1 REFERENCE, computed in THIS file.
//
// It shares no code with the implementation under test: it walks the merged
// support and accumulates |F_P - F_Q| over each gap, which is the
// definition of the 1-D Wasserstein-1 distance.
//
// This is the same algorithm the oracle uses, but the oracle's is numpy and
// this one's is C++ -- the point is that neither calls the library.
// ---------------------------------------------------------------------------
static double wasserstein1_reference(std::vector<double> p, std::vector<double> q) {
    if (p.empty() || q.empty()) return 0.0;
    std::sort(p.begin(), p.end());
    std::sort(q.begin(), q.end());

    // The merged support.
    std::vector<double> pts;
    pts.reserve(p.size() + q.size());
    std::set_union(p.begin(), p.end(), q.begin(), q.end(), std::back_inserter(pts));

    double total = 0.0;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        // The CDFs AT pts[i], counting every sample <= pts[i].
        size_t np = static_cast<size_t>(
            std::upper_bound(p.begin(), p.end(), pts[i]) - p.begin());
        size_t nq = static_cast<size_t>(
            std::upper_bound(q.begin(), q.end(), pts[i]) - q.begin());
        double fp = static_cast<double>(np) / static_cast<double>(p.size());
        double fq = static_cast<double>(nq) / static_cast<double>(q.size());
        total += std::abs(fp - fq) * (pts[i + 1] - pts[i]);
    }
    return total;
}

// A helper to build a Tensor from a vector of rows.
static Tensor make_tensor(const std::vector<std::vector<double>>& rows) {
    const size_t n = rows.size();
    const size_t d = n > 0 ? rows[0].size() : 0;
    Tensor T(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d && j < rows[i].size(); ++j) {
            T(i, j) = rows[i][j];
        }
    }
    return T;
}

// =============================================================================
int main() {
    printf("=== Sliced Wasserstein distance ===\n");

    // =========================================================================
    // 1. THE 1-D FIXTURE: THE SLICED MACHINERY COLLAPSES ONTO THE 1-D W1
    //
    // In R^1 the only unit direction is +1 (and -1, which gives the same
    // 1-D distance), so with one direction the SWD IS the 1-D Wasserstein
    // distance -- a quantity the test computes INDEPENDENTLY.
    //
    // The fixture is UNEQUAL-SIZED (4 vs 3), so the general 1-D W1 form is
    // exercised: the tempting `mean(|sorted_p - sorted_q|)` shortcut is
    // only valid for equal n and would give a different answer.
    // =========================================================================
    section("1. the 1-D fixture: SWD == the 1-D Wasserstein distance");
    {
        Tensor P = make_tensor({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor Q = make_tensor({{10.0}, {11.0}, {12.0}});
        Tensor dirs = make_tensor({{1.0}});

        // -- NON-VACUITY: the two sets really are different sizes, so the
        //    general 1-D W1 form is the one under test. --
        check(P.rows == 4 && Q.rows == 3,
              "1.1 the fixture's two sample counts genuinely DIFFER (4 vs 3)");

        double swd = utils::sliced_wasserstein_distance(P, Q, dirs);
        double ref = wasserstein1_reference({0.0, 1.0, 2.0, 3.0},
                                            {10.0, 11.0, 12.0});
        printf("    SWD      = %.12f\n", swd);
        printf("    1-D W1   = %.12f\n", ref);
        check(rel_err(swd, ref) < 1e-12,
              "1.2 the SWD equals the independently computed 1-D W1");

        // -- AND THE SHORTCUT IS DEMONSTRABLY WRONG HERE. --
        double shortcut = 0.0;
        {
            std::vector<double> ps = {0.0, 1.0, 2.0, 3.0};
            std::vector<double> qs = {10.0, 11.0, 12.0};
            std::sort(ps.begin(), ps.end());
            std::sort(qs.begin(), qs.end());
            for (size_t i = 0; i < ps.size(); ++i)
                shortcut += std::abs(ps[i] - qs[std::min(i, qs.size() - 1)]);
            shortcut /= static_cast<double>(ps.size());
        }
        printf("    shortcut = %.12f  (the mean form, truncated -- WRONG)\n", shortcut);
        check(std::abs(ref - shortcut) > 1e-6,
              "1.3 the mean-form shortcut gives a DIFFERENT (wrong) answer here, "
              "so the fixture discriminates");

        // -- A PURE TRANSLATION: the distance must be exactly the shift. --
        Tensor P2 = make_tensor({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor Q2 = make_tensor({{1.0}, {2.0}, {3.0}, {4.0}});
        double swd_shift = utils::sliced_wasserstein_distance(P2, Q2, dirs);
        printf("    translated SWD = %.12f\n", swd_shift);
        check(std::abs(swd_shift - 1.0) < 1e-12,
              "1.4 a pure translation by 1 gives a distance of exactly 1");

        // -- A COMMON DILATION: the distance must scale. --
        Tensor P3 = make_tensor({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor Q3 = make_tensor({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor P3d = make_tensor({{0.0}, {2.0}, {4.0}, {6.0}});
        Tensor Q3d = make_tensor({{0.0}, {2.0}, {4.0}, {6.0}});
        double d_scaled = utils::sliced_wasserstein_distance(P3d, Q3d, dirs);
        double d_base = utils::sliced_wasserstein_distance(P3, Q3, dirs);
        check(std::abs(d_base) < 1e-12,
              "1.5 the identical sets have distance exactly 0");
        check(std::abs(d_scaled - 2.0 * d_base) < 1e-12,
              "1.6 dilating BOTH sets by 2 scales the distance by 2");
    }

    // =========================================================================
    // 2. THE EQUAL-n FIXTURE: THE TWO 1-D W1 FORMS AGREE
    //
    // The general form must reduce to the `mean(|sorted_p - sorted_q|)`
    // shortcut when the sample counts are equal. Asserting THAT is what pins
    // the general form to be the general one rather than a coincidentally
    // similar quantity.
    // =========================================================================
    section("2. the equal-n fixture: the general and mean forms agree");
    {
        Tensor P = make_tensor({{0.0}, {3.0}, {1.0}, {2.0}});
        Tensor Q = make_tensor({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor dirs = make_tensor({{1.0}});

        std::vector<double> ps = {0.0, 3.0, 1.0, 2.0};
        std::vector<double> qs = {0.0, 1.0, 2.0, 3.0};
        std::sort(ps.begin(), ps.end());
        std::sort(qs.begin(), qs.end());
        double mean_form = 0.0;
        for (size_t i = 0; i < ps.size(); ++i)
            mean_form += std::abs(ps[i] - qs[i]);
        mean_form /= static_cast<double>(ps.size());

        double general = wasserstein1_reference(ps, qs);
        check(std::abs(general - mean_form) < 1e-12,
              "2.1 for equal n the general form EQUALS the mean form");

        double swd = utils::sliced_wasserstein_distance(P, Q, dirs);
        printf("    SWD = %.12f   mean form = %.12f\n", swd, mean_form);
        check(std::abs(swd - mean_form) < 1e-12,
              "2.2 the SWD equals the mean form on the equal-n fixture");

        // -- AND THE SORTING IS LOAD-BEARING: the UNSORTED pairing gives a
        //    DIFFERENT (larger) answer. --
        double unsorted = 0.0;
        const std::vector<double> ps_raw = {0.0, 3.0, 1.0, 2.0};
        const std::vector<double> qs_raw = {0.0, 1.0, 2.0, 3.0};
        for (size_t i = 0; i < ps_raw.size(); ++i)
            unsorted += std::abs(ps_raw[i] - qs_raw[i]);
        unsorted /= static_cast<double>(ps_raw.size());
        printf("    unsorted pairing = %.12f\n", unsorted);
        check(std::abs(unsorted - mean_form) > 1e-6,
              "2.3 the unsorted pairing gives a DIFFERENT answer, so the "
              "fixture discriminates a missing sort");
        check(mean_form <= unsorted + 1e-12,
              "2.4 the sorted value is <= the unsorted one (sorting MINIMISES "
              "the L1 sum, so a missing sort OVERESTIMATES)");
    }

    // =========================================================================
    // 3. THE PINNED MULTI-DIMENSIONAL VALUES
    //
    // The oracle's section 8 fixture: P is 5x2, Q is 6x2, and the three
    // directions are fixed. The oracle's numbers are the specification.
    //
    //     swd_mean = 0.9039482117703043
    //     swd_max  = 1.1785113019775795
    //     direction 0: W1 = 0.5333333333333334
    //     direction 1: W1 = 1.0
    //     direction 2: W1 = 1.1785113019775795
    // =========================================================================
    section("3. the pinned multi-dimensional values");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0},
                                 {2.0, 4.0}, {5.0, 0.0}});
        Tensor Q = make_tensor({{1.0, 1.0}, {2.0, 3.0}, {4.0, 2.0},
                                 {3.0, 5.0}, {6.0, 1.0}, {0.0, 2.0}});
        const double s = 0.7071067811865476;
        Tensor dirs = make_tensor({{1.0, 0.0}, {0.0, 1.0}, {s, s}});

        // -- NON-VACUITY: the three per-direction values really do differ,
        //    so the MEAN and the MAX are different numbers. --
        double mean_sw = utils::sliced_wasserstein_distance(P, Q, dirs);
        double max_sw = utils::max_sliced_wasserstein_distance(P, Q, dirs);
        printf("    mean = %.12f   (oracle: 0.9039482117703043)\n", mean_sw);
        printf("    max  = %.12f   (oracle: 1.1785113019775795)\n", max_sw);
        check(std::abs(mean_sw - max_sw) > 1e-6,
              "3.1 the mean and the max DIFFER on this fixture, so it "
              "discriminates the two aggregation rules");

        check(rel_err(mean_sw, 0.9039482117703043) < 1e-9,
              "3.2 the MEAN over the directions matches the oracle");
        check(rel_err(max_sw, 1.1785113019775795) < 1e-9,
              "3.3 the MAX over the directions matches the oracle");

        // -- The MAX must be >= the MEAN (it is the max of the same
        //    numbers), and the MEAN must lie BETWEEN the min and the max. --
        check(max_sw >= mean_sw - 1e-12,
              "3.4 the max is >= the mean");

        // -- And the individual per-direction values, so an implementation
        //    that got the AGGREGATION right but the per-direction value
        //    wrong cannot hide. --
        double w0 = utils::sliced_wasserstein_distance(
            P, Q, make_tensor({{1.0, 0.0}}));
        double w1 = utils::sliced_wasserstein_distance(
            P, Q, make_tensor({{0.0, 1.0}}));
        double w2 = utils::sliced_wasserstein_distance(
            P, Q, make_tensor({{s, s}}));
        printf("    per-direction: %.12f, %.12f, %.12f\n", w0, w1, w2);
        check(rel_err(w0, 0.5333333333333334) < 1e-9,
              "3.5 direction 0's 1-D W1 matches the oracle");
        check(rel_err(w1, 1.0) < 1e-9,
              "3.6 direction 1's 1-D W1 matches the oracle");
        check(rel_err(w2, 1.1785113019775795) < 1e-9,
              "3.7 direction 2's 1-D W1 matches the oracle");
        check(rel_err((w0 + w1 + w2) / 3.0, mean_sw) < 1e-12,
              "3.8 the mean is the average of the per-direction values");
    }

    // =========================================================================
    // 4. THE PROJECTION MUST BE ONTO A UNIT DIRECTION
    //
    // The oracle measured that scaling a direction by 2 scales the answer
    // by exactly 2. **If the implementation NORMALISED the directions, this
    // check would FAIL** -- and that is the point: the definition says the
    // directions are unit vectors, and a caller who passes a non-unit
    // direction has passed a different metric. The C++ must therefore
    // EITHER normalise (and say so) OR reject it.
    //
    // The contract chosen here is: the routine REQUIRES unit directions and
    // REJECTS a non-unit one with a distinct message. That is the stricter
    // contract, and it makes a silently-wrong scale impossible.
    // =========================================================================
    section("4. the directions must be unit");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 0.0}, {3.0, 0.0}});
        Tensor Q = make_tensor({{0.0, 0.0}, {2.0, 0.0}});
        Tensor unit_dirs = make_tensor({{1.0, 0.0}});
        Tensor scaled_dirs = make_tensor({{2.0, 0.0}});

        double u = utils::sliced_wasserstein_distance(P, Q, unit_dirs);
        printf("    unit   = %.12f\n", u);

        bool threw = false;
        std::string msg;
        try {
            utils::sliced_wasserstein_distance(P, Q, scaled_dirs);
        } catch (const std::invalid_argument& e) {
            threw = true;
            msg = e.what();
        }
        check(threw,
              "4.1 a NON-UNIT direction is REJECTED (not silently rescaled)");
        if (threw) {
            printf("    message: \"%s\"\n", msg.c_str());
            check(msg.find("unit") != std::string::npos,
                  "4.2 the rejection message mentions the unit requirement");
        }
    }

    // =========================================================================
    // 5. THE PROPERTIES A DISTANCE MUST HAVE
    // =========================================================================
    section("5. the distance properties");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0},
                                 {2.0, 4.0}, {5.0, 0.0}});
        Tensor Q = make_tensor({{1.0, 1.0}, {2.0, 3.0}, {4.0, 2.0},
                                 {3.0, 5.0}, {6.0, 1.0}, {0.0, 2.0}});
        Tensor dirs = make_tensor({{1.0, 0.0}, {0.0, 1.0}, {0.6, 0.8}});

        double d = utils::sliced_wasserstein_distance(P, Q, dirs);
        double d_rev = utils::sliced_wasserstein_distance(Q, P, dirs);
        double d_self = utils::sliced_wasserstein_distance(P, P, dirs);

        check(std::abs(d_self) < 1e-12,
              "5.1 d(X, X) == 0 exactly");
        check(d >= 0.0,
              "5.2 d >= 0");
        check(rel_err(d, d_rev) < 1e-12,
              "5.3 d is SYMMETRIC in its two arguments");
        check(rel_err(d, utils::max_sliced_wasserstein_distance(P, Q, dirs)) > 1e-6,
              "5.4 the mean and the max differ on this fixture too");
    }

    // =========================================================================
    // 6. THE RANDOM-DIRECTION API
    //
    // The convenience form draws the directions itself. The properties it
    // must have are checked here:
    //   * the directions it produces are UNIT (it must normalise them);
    //   * the result is >= 0 and symmetric;
    //   * the SAME seed gives the SAME answer (reproducibility), and a
    //     DIFFERENT seed generally gives a different one (the seed is
    //     actually used).
    //
    // **THE SEED IS PART OF THE CONTRACT.** A metric whose value depends on
    // an unseeded RNG is not reproducible, and a "reproducibility" test
    // that only checks `d >= 0` would pass for an implementation that
    // ignored the seed entirely.
    // =========================================================================
    section("6. the random-direction API");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0},
                                 {2.0, 4.0}, {5.0, 0.0}});
        Tensor Q = make_tensor({{1.0, 1.0}, {2.0, 3.0}, {4.0, 2.0},
                                 {3.0, 5.0}, {6.0, 1.0}, {0.0, 2.0}});

        Tensor dirs = utils::random_unit_directions(16, 2, 12345);
        check(dirs.rows == 16 && dirs.cols == 2,
              "6.1 the generated directions have the requested shape (16, 2)");
        double worst_norm = 0.0;
        for (size_t i = 0; i < dirs.rows; ++i) {
            double n = 0.0;
            for (size_t j = 0; j < dirs.cols; ++j) n += dirs(i, j) * dirs(i, j);
            n = std::sqrt(n);
            worst_norm = std::max(worst_norm, std::abs(n - 1.0));
        }
        printf("    worst |‖theta‖ - 1| = %.3e\n", worst_norm);
        check(worst_norm < 1e-12,
              "6.2 EVERY generated direction is a UNIT vector");

        double a = 0.0, b = 0.0, c = 0.0, d_rev = 0.0;
        const bool ok_a = call_guarded(
            [&]() { return utils::sliced_wasserstein_distance(P, Q, 16, 12345); }, &a);
        const bool ok_b = call_guarded(
            [&]() { return utils::sliced_wasserstein_distance(P, Q, 16, 12345); }, &b);
        const bool ok_c = call_guarded(
            [&]() { return utils::sliced_wasserstein_distance(P, Q, 16, 999); }, &c);
        const bool ok_r = call_guarded(
            [&]() { return utils::sliced_wasserstein_distance(Q, P, 16, 12345); }, &d_rev);

        check(ok_a && ok_b && ok_c && ok_r,
              "6.3 the random-direction form returns a value (it does not throw)");
        if (!ok_a || !ok_b || !ok_c || !ok_r) {
            // Nothing further in this section is meaningful if the call
            // threw, and every check below would be a comparison against an
            // UNINITIALISED value.
            printf("    seeded SWD = <unavailable>\n");
        } else {
            printf("    seeded SWD = %.12f\n", a);
            check(std::abs(a - b) < 1e-15,
                  "6.4 the SAME seed gives the SAME answer (reproducible)");
            check(std::abs(a - c) > 1e-9,
                  "6.5 a DIFFERENT seed generally gives a DIFFERENT answer "
                  "(the seed is actually used)");
            check(rel_err(a, d_rev) < 1e-12,
                  "6.6 the random-direction form is symmetric too");
            check(a >= 0.0, "6.7 the random-direction form is >= 0");
        }
    }

    // =========================================================================
    // 7. THE GUARDS
    //
    // Every message is DISTINCT -- a substring that appears in two
    // different messages discriminates nothing between them.
    // =========================================================================
    section("7. the guards");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 2.0}});
        Tensor Q = make_tensor({{1.0, 1.0}, {2.0, 3.0}});
        Tensor dirs = make_tensor({{1.0, 0.0}});
        Tensor wrong_dim_dirs = make_tensor({{1.0, 0.0, 0.0}});
        // A sample set with a DIFFERENT NUMBER OF COLUMNS.
        Tensor wrong_dim_samples = make_tensor({{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}});

        auto throws_with = [](const std::function<void()>& fn, const char* needle) {
            try { fn(); } catch (const std::invalid_argument& e) {
                return std::string(e.what()).find(needle) != std::string::npos;
            } catch (...) { return false; }
            return false;
        };

        // **THE NEEDLES ARE THE FULL MESSAGES, NOT THE BARE WORDS.** The
        // words "empty" and "dimension" each appear in SEVERAL guard
        // messages, so a bare-word needle would be satisfied by whichever
        // guard happened to fire first -- and removing the one under test
        // would go unnoticed. Each check below names its own guard.
        check(throws_with([&]() {
                  utils::sliced_wasserstein_distance(Tensor(), Q, dirs);
              }, "the first sample set is empty"),
              "7.1 an empty sample set is rejected");

        check(throws_with([&]() {
                  utils::sliced_wasserstein_distance(P, Q, Tensor());
              }, "the direction set is empty"),
              "7.2 an empty direction set is rejected");

        check(throws_with([&]() {
                  utils::sliced_wasserstein_distance(P, Q, wrong_dim_dirs);
              }, "the directions' dimension does not match the samples'"),
              "7.3 a direction set of the wrong dimension is rejected");

        // **THE FIXTURE MUST VARY IN THE DIMENSION THE GUARD IS ABOUT.** The
        // first draft of this check used P = 1x2 and Q = 2x2 -- which have
        // the SAME number of COLUMNS, so there was no dimension mismatch at
        // all and the guard never fired. A row-count difference is not a
        // dimension difference.
        check(P.cols == 2 && wrong_dim_samples.cols == 3,
              "7.4 the fixture's two sample sets really do have different "
              "COLUMN counts");
        check(throws_with([&]() {
                  utils::sliced_wasserstein_distance(wrong_dim_samples, Q, dirs);
              }, "the two sample sets have different dimensions"),
              "7.5 two sample sets of different dimensions are rejected");

        check(throws_with([&]() {
                  utils::sliced_wasserstein_distance(P, Q, 0, 12345);
              }, "the projection count must be >= 1"),
              "7.6 a non-positive projection count is rejected");

        check(throws_with([&]() {
                  utils::random_unit_directions(4, 0, 1);
              }, "the dimension must be >= 1"),
              "7.7 a zero-dimensional direction space is rejected");
    }

    // =========================================================================
    // 8. THE VICTIM-FIXTURE CHECK: THE SUITE CAN FAIL
    //
    // A suite that cannot fail proves nothing. This section asserts that
    // the discriminating fixtures really do DISCRIMINATE, by evaluating
    // the WRONG variants of the metric and asserting they differ from the
    // right one.
    // =========================================================================
    section("8. the suite can fail: the wrong variants are caught");
    {
        Tensor P = make_tensor({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0},
                                 {2.0, 4.0}, {5.0, 0.0}});
        Tensor Q = make_tensor({{1.0, 1.0}, {2.0, 3.0}, {4.0, 2.0},
                                 {3.0, 5.0}, {6.0, 1.0}, {0.0, 2.0}});
        Tensor dirs = make_tensor({{1.0, 0.0}, {0.0, 1.0}, {0.6, 0.8}});

        double right = utils::sliced_wasserstein_distance(P, Q, dirs);

        // WRONG VARIANT 1: the MAX instead of the MEAN.
        double wrong_max = utils::max_sliced_wasserstein_distance(P, Q, dirs);
        check(std::abs(right - wrong_max) > 1e-6,
              "8.1 the max-aggregation foil DIFFERS from the right answer");

        // WRONG VARIANT 2: the MEAN form of the 1-D W1 on an unequal-n
        // fixture (which is what a shortcut implementation would use).
        Tensor Pu = make_tensor({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0}});
        Tensor Qu = make_tensor({{1.0, 1.0}, {2.0, 3.0}, {4.0, 2.0},
                                 {3.0, 5.0}, {6.0, 1.0}});
        double right_u = utils::sliced_wasserstein_distance(Pu, Qu, dirs);
        double foil_u = 0.0;
        for (size_t k = 0; k < dirs.rows; ++k) {
            std::vector<double> pv, qv;
            for (size_t i = 0; i < Pu.rows; ++i) {
                double t = 0.0;
                for (size_t j = 0; j < Pu.cols; ++j) t += Pu(i, j) * dirs(k, j);
                pv.push_back(t);
            }
            for (size_t i = 0; i < Qu.rows; ++i) {
                double t = 0.0;
                for (size_t j = 0; j < Qu.cols; ++j) t += Qu(i, j) * dirs(k, j);
                qv.push_back(t);
            }
            std::sort(pv.begin(), pv.end());
            std::sort(qv.begin(), qv.end());
            double acc = 0.0;
            for (size_t i = 0; i < pv.size(); ++i)
                acc += std::abs(pv[i] - qv[std::min(i, qv.size() - 1)]);
            foil_u += acc / static_cast<double>(pv.size());
        }
        foil_u /= static_cast<double>(dirs.rows);
        printf("    right = %.12f   mean-form foil = %.12f\n", right_u, foil_u);
        check(std::abs(right_u - foil_u) > 1e-6,
              "8.2 the mean-form foil DIFFERS from the right answer on the "
              "unequal-n fixture");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
