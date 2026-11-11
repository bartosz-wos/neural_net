// =============================================================================
// tests/test_isotonic_differential.cpp -- the RANDOMIZED differential check
// against an EXACT optimum computed by exhaustive enumeration.
//
// WHY THIS FILE EXISTS SEPARATELY FROM tests/test_isotonic.cpp
// =============================================================================
//
// The pinned suite pins EXACT VALUES on a handful of hand-written fixtures.
// That is necessary and it is not sufficient: a pinned fixture only sees the
// fixture it was written for, and the whole point of a randomised sweep is to
// hit the shapes nobody thought of -- HEAVY TIES in x, UNEQUAL weights, and
// both directions.
//
// THE REFERENCE IS AN EXACT OPTIMUM, AND IT IS EXACT
// =============================================================================
//
// The reference enumerates EVERY partition of the distinct-x GROUPS into
// consecutive blocks, keeps the monotone ones, and takes the smallest
// weighted SSE. That is the definition of the weighted-L2 isotonic
// regression, so the reference is exact by construction rather than by
// agreement with anything.
//
// THE REFERENCE NEVER COLLAPSES THE TIES ITSELF, AND THAT IS THE POINT
// =============================================================================
//
// The shipped implementation collapses tied x to a weighted mean before
// running PAVA. If the reference did the same, a defect in the collapse rule
// would be invisible here: both sides would collapse the same wrong way.
//
// So the reference works on the ORIGINAL points. A block's optimal value is
// the weighted mean of the ORIGINAL points it contains,
//
//     v_b = ( SUM_{i in b} w_i y_i ) / ( SUM_{i in b} w_i ),
//
// and the constraint that all points sharing an x get ONE value is imposed
// by partitioning the GROUPS (never a group), not by pre-collapsing.
//
// **THE WEIGHTED-MEAN TIE-COLLAPSE RULE IS THEREFORE EMERGENT HERE**: the
// tie group's own weighted mean is forced by least squares once the group
// sits inside a block, rather than assumed. A collapse that used the
// UNWEIGHTED mean, or that forgot to SUM the weights, produces a different
// optimum and is caught here.
//
// The zero-weight rule is handled the same way: a point with weight zero
// contributes nothing to the objective, so it is dropped from BOTH sides.
//
// THE BRUTE FORCE IS EXPONENTIAL, AND THAT IS WHY IT IS USEFUL
// =============================================================================
//
// The x values are drawn from a SMALL INTEGER GRID, so the number of
// distinct x values -- and therefore the number of partitions -- stays
// tiny. The reference is obviously correct BECAUSE it is exhaustive, and it
// is affordable BECAUSE the grid is small.
//
//   Build:  make build/test_isotonic_differential
//   Run:    ./build/test_isotonic_differential
// =============================================================================

#include "nn/utils/isotonic_regression.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// `Tensor` is at GLOBAL scope in this repo (include/nn/core/tensor.h has no
// namespace); the utilities live in `nn::utils`.
using nn::utils::fit_isotonic;
using nn::utils::IsotonicDirection;
using nn::utils::IsotonicFit;

static int g_pass = 0;
static int g_fail = 0;

static Tensor column(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// The tolerance for comparing two fitted VALUES. The optimum is rational
// arithmetic over small integers, so two values are either exactly equal or
// differ by a LOT -- 1e-9 sits far below the smallest possible gap and far
// above the double-precision noise floor.
static const double kValueTol = 1e-9;

// The tolerance for deciding that two ADJACENT fitted values are EQUAL, and
// therefore that the two x's belong to the same run of the step function.
static const double kEqualTol = 1e-9;

// -----------------------------------------------------------------------------
// The exact optimum, by exhaustive enumeration of the block partitions.
// -----------------------------------------------------------------------------
namespace {

// A group of input points that share the same x.
struct Group {
    double x = 0.0;
    std::vector<size_t> points;  // the indices of the points in this group
};

// The result of the exhaustive search.
struct Optimum {
    std::vector<double> distinct_x;   // the distinct x, ascending
    std::vector<double> value;        // the fitted value at each distinct x
    std::vector<size_t> knot_group;   // the group index of each retained knot
    bool valid = false;
};

// The exact weighted-L2 isotonic regression of y on x.
//
// The fit is a step function whose blocks are CONSECUTIVE ranges of the
// sorted points, so enumerating every partition of the distinct-x GROUPS,
// keeping the monotone ones, and taking the smallest weighted SSE gives the
// exact optimum -- by the DEFINITION of the problem, not by agreement with
// any other implementation.
Optimum exact_optimum(const std::vector<double>& x,
                       const std::vector<double>& y,
                       const std::vector<double>& w, bool increasing) {
    Optimum out;

    // A zero-weight point contributes nothing to the objective, so it is
    // dropped from BOTH sides. (It must not shift the fitted range either.)
    std::vector<size_t> idx;
    idx.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        if (w[i] > 0.0) idx.push_back(i);
    if (idx.empty()) return out;

    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) { return x[a] < x[b]; });

    // The groups of points that share an x.
    std::vector<Group> groups;
    for (size_t k = 0; k < idx.size();) {
        Group g;
        g.x = x[idx[k]];
        size_t j = k;
        while (j < idx.size() && x[idx[j]] == g.x) {
            g.points.push_back(idx[j]);
            ++j;
        }
        groups.push_back(g);
        k = j;
    }
    const size_t G = groups.size();

    const size_t n_cuts = G - 1;
    const size_t n_masks = static_cast<size_t>(1) << n_cuts;

    double best_sse = 0.0;
    bool have_best = false;
    std::vector<double> best_value(G, 0.0);

    for (size_t mask = 0; mask < n_masks; ++mask) {
        // The block boundaries: a cut after group e when bit e of the mask
        // is set. The value of a block is the weighted mean of the ORIGINAL
        // points it contains -- never a mean of pre-collapsed values.
        std::vector<double> block_value;
        std::vector<size_t> block_of_group(G, 0);
        double sse = 0.0;
        bool ok = true;

        size_t g = 0;
        while (g < G) {
            size_t e = g;
            double sw = 0.0, swy = 0.0;
            for (size_t q = g; q <= e; ++q) {
                for (size_t p : groups[q].points) {
                    sw += w[p];
                    swy += w[p] * y[p];
                }
            }
            // Extend the block while the next group is NOT cut off.
            while (true) {
                const bool last = (e + 1 == G);
                const bool cut = !last && ((mask >> e) & static_cast<size_t>(1));
                if (last || cut) break;
                ++e;
                for (size_t q = e; q <= e; ++q) {
                    for (size_t p : groups[q].points) {
                        sw += w[p];
                        swy += w[p] * y[p];
                    }
                }
            }
            const double v = swy / sw;
            for (size_t q = g; q <= e; ++q) block_of_group[q] = block_value.size();
            block_value.push_back(v);
            for (size_t q = g; q <= e; ++q)
                for (size_t p : groups[q].points)
                    sse += w[p] * (y[p] - v) * (y[p] - v);
            g = e + 1;
        }

        // The monotonicity constraint.
        for (size_t b = 0; b + 1 < block_value.size(); ++b) {
            if (increasing && block_value[b] > block_value[b + 1] + 1e-12) {
                ok = false;
                break;
            }
            if (!increasing && block_value[b] < block_value[b + 1] - 1e-12) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        if (!have_best || sse < best_sse - 1e-14) {
            have_best = true;
            best_sse = sse;
            for (size_t q = 0; q < G; ++q) best_value[q] = block_value[block_of_group[q]];
        }
    }

    if (!have_best) return out;

    out.valid = true;
    for (size_t q = 0; q < G; ++q) {
        out.distinct_x.push_back(groups[q].x);
        out.value.push_back(best_value[q]);
    }

    // The retained knots are the boundaries of the maximal runs of EQUAL
    // fitted values: an interior point whose fitted value equals BOTH
    // neighbours' is dropped, so a run of equal values keeps only its
    // first and last point.
    //
    // This is derived from the DEFINITION, and it is applied to the exact
    // optimum's values rather than to anything the implementation produced.
    std::vector<size_t> keep(G, 1);
    for (size_t q = 1; q + 1 < G; ++q) {
        const bool same_left =
            std::fabs(out.value[q] - out.value[q - 1]) <= kEqualTol;
        const bool same_right =
            std::fabs(out.value[q] - out.value[q + 1]) <= kEqualTol;
        if (same_left && same_right) keep[q] = 0;
    }
    for (size_t q = 0; q < G; ++q)
        if (keep[q]) out.knot_group.push_back(q);

    return out;
}

// The step function's value at `qx`, reconstructed from the RETAINED knots.
//
// The retained knots are the boundaries of the maximal runs of equal fitted
// values, so the value at an x is the value of the LAST knot at or before
// it. (A run's interior x's all carry the run's value, and a run's two
// boundaries carry it too, so this is exact.)
double value_from_knots(const IsotonicFit& fit, double qx) {
    double v = fit.y_knots.empty() ? 0.0 : fit.y_knots.front();
    for (size_t k = 0; k < fit.x_knots.size(); ++k) {
        if (fit.x_knots[k] > qx) break;
        v = fit.y_knots[k];
    }
    return v;
}

}  // namespace

// =============================================================================

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "test_isotonic_differential -- C++ vs the EXACT optimum"
              << std::endl;
    std::cout << "====================================================" << std::endl;

    const int kCases = 400;
    const unsigned kSeed = 20261110u;

    std::mt19937 rng(kSeed);
    // A SMALL integer grid for x, so ties are frequent BY CONSTRUCTION.
    std::uniform_int_distribution<int> pick_x(-3, 3);
    std::uniform_int_distribution<int> pick_y(0, 3);
    // A small integer grid for w, so unequal-weight ties occur and every
    // weight is exactly representable.
    std::uniform_int_distribution<int> pick_w(0, 3);
    std::uniform_int_distribution<int> pick_n(2, 14);

    int value_mismatches = 0;
    int knot_count_mismatches = 0;
    int knot_value_mismatches = 0;
    int ambiguous = 0;
    int cases_run = 0;
    double worst_value_diff = 0.0;
    double worst_knot_diff = 0.0;
    std::string first_failure;

    for (int c = 0; c < kCases; ++c) {
        const int n = pick_n(rng);
        std::vector<double> x(n), y(n), w(n);
        for (int i = 0; i < n; ++i) {
            x[i] = pick_x(rng);
            y[i] = pick_y(rng);
            w[i] = pick_w(rng);
        }
        // Every fourth case drops its zero weights entirely, so the fit has
        // something to do; the zero-weight rule is still exercised by the
        // rest.
        if (c % 4 == 0)
            for (int i = 0; i < n; ++i)
                if (w[i] == 0.0) w[i] = 1.0;
        bool any_positive = false;
        for (int i = 0; i < n; ++i)
            if (w[i] > 0.0) any_positive = true;
        if (!any_positive) w[0] = 1.0;

        const bool increasing = (c % 5 != 0);
        const IsotonicDirection dir = increasing ? IsotonicDirection::Increasing
                                                 : IsotonicDirection::Decreasing;

        const Optimum opt = exact_optimum(x, y, w, increasing);
        if (!opt.valid) continue;
        ++cases_run;

        const IsotonicFit fit = fit_isotonic(column(x), column(y), column(w), dir);

        // ---- the fitted VALUE at every distinct x.
        bool values_ok = true;
        for (size_t q = 0; q < opt.distinct_x.size(); ++q) {
            const double got = value_from_knots(fit, opt.distinct_x[q]);
            const double want = opt.value[q];
            const double d = std::fabs(got - want);
            if (d > worst_value_diff) worst_value_diff = d;
            if (d > kValueTol) {
                values_ok = false;
                if (value_mismatches == 0) {
                    first_failure = "case " + std::to_string(c) + ": value at x=" +
                                   std::to_string(opt.distinct_x[q]) + " got " +
                                   std::to_string(got) + " want " +
                                   std::to_string(want);
                }
                ++value_mismatches;
            }
        }

        // ---- the retained knot COUNT.
        //
        // The count is compared only when the optimum's adjacent values are
        // UNAMBIGUOUS -- either exactly equal or far apart. Two adjacent
        // values that differ by a hair could be one run in one
        // implementation and two in the other, and the knot count would then
        // differ for a reason that is not a defect. Such cases are counted
        // and reported rather than silently passed.
        bool ambiguous_case = false;
        for (size_t q = 1; q < opt.value.size(); ++q) {
            const double d = std::fabs(opt.value[q] - opt.value[q - 1]);
            if (d > kEqualTol && d < 1e-6) ambiguous_case = true;
        }

        if (!ambiguous_case) {
            if (fit.x_knots.size() != opt.knot_group.size()) {
                ++knot_count_mismatches;
                if (first_failure.empty())
                    first_failure = "case " + std::to_string(c) + ": knot count " +
                                   std::to_string(fit.x_knots.size()) + " vs the exact " +
                                   std::to_string(opt.knot_group.size());
            } else {
                for (size_t k = 0; k < opt.knot_group.size(); ++k) {
                    const double dx =
                        std::fabs(fit.x_knots[k] - opt.distinct_x[opt.knot_group[k]]);
                    const double dy =
                        std::fabs(fit.y_knots[k] - opt.value[opt.knot_group[k]]);
                    if (dx > worst_knot_diff) worst_knot_diff = dx;
                    if (dy > worst_knot_diff) worst_knot_diff = dy;
                    if (dx > kValueTol || dy > kValueTol) {
                        ++knot_value_mismatches;
                        if (first_failure.empty())
                            first_failure = "case " + std::to_string(c) + ": knot " +
                                           std::to_string(k) + " got (" +
                                           std::to_string(fit.x_knots[k]) + ", " +
                                           std::to_string(fit.y_knots[k]) + ") vs (" +
                                           std::to_string(opt.distinct_x[opt.knot_group[k]]) +
                                           ", " + std::to_string(opt.value[opt.knot_group[k]]) + ")";
                    }
                }
            }
        } else {
            ++ambiguous;
        }

        if (!values_ok) {
            // The value check already reported; nothing more to do here.
        }
    }

    std::cout << "  cases:               " << cases_run << std::endl;
    std::cout << "  worst value diff:    " << worst_value_diff << std::endl;
    std::cout << "  worst knot diff:     " << worst_knot_diff << std::endl;
    std::cout << "  ambiguous cases:     " << ambiguous << std::endl;

    const bool ok = (value_mismatches == 0 && knot_count_mismatches == 0 &&
                     knot_value_mismatches == 0);
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] every randomised case matches the EXACT optimum"
                  << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << value_mismatches << " value, "
                  << knot_count_mismatches << " knot-count and "
                  << knot_value_mismatches << " knot-value mismatches"
                  << std::endl;
        if (!first_failure.empty())
            std::cout << "      first: " << first_failure << std::endl;
    }

    std::cout << "=== SUMMARY: " << g_pass << " passed, " << g_fail
              << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
