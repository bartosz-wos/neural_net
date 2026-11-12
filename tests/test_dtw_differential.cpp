// =============================================================================
// tests/test_dtw_differential.cpp -- the RANDOMIZED differential check for
// the dynamic time warping utilities.
//
// THE REFERENCE IS THE DEFINITION, NOT AN ALGORITHM
// =============================================================================
//
// The shipped implementation is a dynamic program over the warping lattice.
// Checking it against a second dynamic program written the same way catches
// transcription errors and nothing else -- the two would share every
// modelling mistake.
//
// So the reference here ENUMERATES every monotone path through the lattice
// and sums the local cost along each one. The DTW distance is, by
// definition, the MINIMUM over all such paths. The enumeration shares no
// structure with the DP: no table, no fill order, no recurrence. It is
// exponential, which is why it is affordable only on short series and why
// it is worth so much as a cross-check.
//
// The enumeration also yields the SET of optimal paths, which lets the
// reference answer a second question the DP cannot: is the shipped path
// actually an optimal one? That is checked by comparing the shipped path's
// own cost against the enumerated minimum, computed independently.
//
// =============================================================================

#include "nn/utils/dtw.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using nn::utils::dtw_align;
using nn::utils::dtw_distance;
using nn::utils::DtwPath;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& what, const std::string& detail = "") {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::cout << "  FAIL: " << what;
        if (!detail.empty()) std::cout << "  -- " << detail;
        std::cout << std::endl;
    }
}

static void section(const char* title) {
    std::cout << std::endl << "--- " << title << " ---" << std::endl;
}

// A deterministic LCG, so the sweep is reproducible from run to run.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed ? seed : 1) {}
    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state >> 17;
    }
    int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

// A (T, D) series.
static Tensor series(const std::vector<double>& values, size_t dim) {
    const size_t T = values.size() / dim;
    Tensor t(T, dim);
    for (size_t i = 0; i < T; ++i) {
        for (size_t k = 0; k < dim; ++k) t(i, k) = values[i * dim + k];
    }
    return t;
}

// The local cost: the Euclidean distance between two observations.
static double local_cost(const Tensor& x, size_t xi, const Tensor& y, size_t yi) {
    double acc = 0.0;
    for (size_t k = 0; k < x.cols; ++k) {
        const double d = x(xi, k) - y(yi, k);
        acc += d * d;
    }
    return std::sqrt(acc);
}

// =============================================================================
// THE REFERENCE -- every monotone path, enumerated
// =============================================================================
//
// A path is a sequence of (i,j) pairs starting at (0,0) and ending at
// (T-1,S-1), where each step advances by (1,0), (0,1) or (1,1). Its cost is
// the sum of the local costs of the cells it visits.
//
// THE CONVENTION THAT MATTERS: the running cost INCLUDES the cell the walk
// is currently standing on. So the origin's local cost is charged, and the
// final cell's local cost is charged. Charging the cell you are LEAVING
// instead would drop the final cell's cost entirely -- and it did, the first
// time this reference was written. The symptom was 312 mismatches over 120
// cases, i.e. every case failing, which is what a systematic off-by-one in
// the REFERENCE looks like.
//
// The enumeration is a DEPTH-FIRST walk over the path tree, carrying the
// running cost and pruning any branch that already exceeds the best found.
struct PathEnumerator {
    const Tensor& x;
    const Tensor& y;
    double best = std::numeric_limits<double>::infinity();

    void walk(size_t i, size_t j, double cost) {
        if (cost >= best) return;  // prune: no extension can be cheaper
        if (i + 1 == x.rows && j + 1 == y.rows) {
            best = cost;
            return;
        }
        // The three symmetric1 steps. The enumeration order does not affect
        // the MINIMUM, only which optimal path is found first.
        if (i + 1 < x.rows && j + 1 < y.rows) {
            walk(i + 1, j + 1, cost + local_cost(x, i + 1, y, j + 1));
        }
        if (i + 1 < x.rows) {
            walk(i + 1, j, cost + local_cost(x, i + 1, y, j));
        }
        if (j + 1 < y.rows) {
            walk(i, j + 1, cost + local_cost(x, i, y, j + 1));
        }
    }
};

static double enumerate_dtw(const Tensor& x, const Tensor& y) {
    PathEnumerator e{x, y};
    e.walk(0, 0, local_cost(x, 0, y, 0));
    return e.best;
}

// The banded reference: a path may only visit cells with |i-j| <= band, on
// the 0-based DATA indices -- which is the same condition as on the 1-based
// cell indices, because |i-j| == |(i-1)-(j-1)|.
struct BandedPathEnumerator {
    const Tensor& x;
    const Tensor& y;
    int band;
    double best = std::numeric_limits<double>::infinity();

    bool allowed(size_t i, size_t j) const {
        const long d = static_cast<long>(i) - static_cast<long>(j);
        return -static_cast<long>(band) <= d && d <= static_cast<long>(band);
    }

    void walk(size_t i, size_t j, double cost) {
        if (cost >= best) return;
        if (i + 1 == x.rows && j + 1 == y.rows) {
            best = cost;
            return;
        }
        if (i + 1 < x.rows && j + 1 < y.rows && allowed(i + 1, j + 1)) {
            walk(i + 1, j + 1, cost + local_cost(x, i + 1, y, j + 1));
        }
        if (i + 1 < x.rows && allowed(i + 1, j)) {
            walk(i + 1, j, cost + local_cost(x, i + 1, y, j));
        }
        if (j + 1 < y.rows && allowed(i, j + 1)) {
            walk(i, j + 1, cost + local_cost(x, i, y, j + 1));
        }
    }
};

static double enumerate_dtw_banded(const Tensor& x, const Tensor& y, int band) {
    BandedPathEnumerator e{x, y, band};
    e.walk(0, 0, local_cost(x, 0, y, 0));
    return e.best;
}

// =============================================================================
// THE SWEEP
// =============================================================================

static void section_unbanded() {
    section("the DTW distance against an exhaustive enumeration of every monotone path");

    Rng rng(20261117);
    int mismatches = 0;
    int cases = 0;
    for (int t = 0; t < 120; ++t) {
        const size_t T = static_cast<size_t>(rng.below(4) + 1);
        const size_t S = static_cast<size_t>(rng.below(4) + 1);
        const size_t dim = static_cast<size_t>(rng.below(2) + 1);
        std::vector<double> xv(T * dim), yv(S * dim);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = static_cast<double>(rng.below(5));
        for (size_t i = 0; i < yv.size(); ++i) yv[i] = static_cast<double>(rng.below(5));
        const Tensor x = series(xv, dim);
        const Tensor y = series(yv, dim);

        const double want = enumerate_dtw(x, y);
        const double got = dtw_distance(x, y);
        ++cases;
        if (std::abs(got - want) > 1e-9) ++mismatches;

        // THE SHIPPED PATH MUST BE AN OPTIMAL ONE. Its cost is recomputed
        // here from the path's cells, independently of the DP that produced
        // it.
        const DtwPath p = dtw_align(x, y);
        double path_cost = 0.0;
        for (const auto& cell : p.cells) {
            path_cost += local_cost(x, cell.first, y, cell.second);
        }
        if (std::abs(path_cost - want) > 1e-9) ++mismatches;
        if (std::abs(p.cost - want) > 1e-9) ++mismatches;
    }
    check(mismatches == 0,
          "the DTW distance and the shipped path's cost match the exhaustive "
          "enumeration on 120 random series pairs",
          std::to_string(mismatches) + " mismatches / " + std::to_string(cases) + " cases");
}

static void section_banded() {
    section("the banded DTW distance against a banded path enumeration");

    Rng rng(20261118);
    int mismatches = 0;
    int cases = 0;
    int feasible = 0;
    for (int t = 0; t < 120; ++t) {
        const size_t T = static_cast<size_t>(rng.below(4) + 1);
        const size_t S = static_cast<size_t>(rng.below(4) + 1);
        const size_t dim = 1;
        std::vector<double> xv(T), yv(S);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = static_cast<double>(rng.below(5));
        for (size_t i = 0; i < yv.size(); ++i) yv[i] = static_cast<double>(rng.below(5));
        const Tensor x = series(xv, dim);
        const Tensor y = series(yv, dim);
        const int band = rng.below(4);
        const double want = enumerate_dtw_banded(x, y, band);
        const double got = dtw_distance(x, y, band);
        ++cases;
        if (std::isfinite(want)) ++feasible;
        const bool got_inf = !std::isfinite(got);
        const bool want_inf = !std::isfinite(want);
        if (got_inf != want_inf) {
            ++mismatches;
        } else if (!got_inf && std::abs(got - want) > 1e-9) {
            ++mismatches;
        }
    }
    check(mismatches == 0,
          "the banded DTW distance matches the banded path enumeration on 120 "
          "random series pairs",
          std::to_string(mismatches) + " mismatches / " + std::to_string(cases) +
              " cases (" + std::to_string(feasible) + " with a feasible band)");
    check(feasible > 0 && feasible < cases,
          "the sweep exercised BOTH feasible and infeasible bands",
          std::to_string(feasible) + " feasible of " + std::to_string(cases));
}

static void section_path_shape() {
    section("the shipped path is a legal monotone path on every sweep case");

    Rng rng(20261119);
    int illegal = 0;
    int wrong_endpoints = 0;
    for (int t = 0; t < 200; ++t) {
        const size_t T = static_cast<size_t>(rng.below(5) + 1);
        const size_t S = static_cast<size_t>(rng.below(5) + 1);
        std::vector<double> xv(T), yv(S);
        for (size_t i = 0; i < T; ++i) xv[i] = static_cast<double>(rng.below(9));
        for (size_t i = 0; i < S; ++i) yv[i] = static_cast<double>(rng.below(9));
        const DtwPath p = dtw_align(series(xv, 1), series(yv, 1));
        if (p.cells.empty()) { ++illegal; continue; }
        if (p.cells.front().first != 0 || p.cells.front().second != 0) ++illegal;
        if (p.cells.back().first != T - 1 || p.cells.back().second != S - 1) {
            ++wrong_endpoints;
        }
        for (size_t k = 0; k + 1 < p.cells.size(); ++k) {
            const long di = static_cast<long>(p.cells[k + 1].first) -
                            static_cast<long>(p.cells[k].first);
            const long dj = static_cast<long>(p.cells[k + 1].second) -
                            static_cast<long>(p.cells[k].second);
            if (!((di == 1 && dj == 0) || (di == 0 && dj == 1) || (di == 1 && dj == 1))) {
                ++illegal;
            }
        }
    }
    check(illegal == 0, "every path is a legal monotone path", std::to_string(illegal) + " illegal");
    check(wrong_endpoints == 0, "every path starts at (0,0) and ends at (T-1,S-1)",
          std::to_string(wrong_endpoints) + " wrong endpoints");
}

// =============================================================================

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "test_dtw_differential -- the randomised differential" << std::endl;
    std::cout << "check against an exhaustive path enumeration" << std::endl;
    std::cout << "====================================================" << std::endl;

    section_unbanded();
    section_banded();
    section_path_shape();

    std::cout << std::endl;
    std::cout << "=== SUMMARY: " << g_pass << " passed, " << g_fail
              << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
