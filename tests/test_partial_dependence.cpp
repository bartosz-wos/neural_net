// test_partial_dependence.cpp — Tests for Partial Dependence / ICE curves
// (Greenwell, "Partial Dependence Plots", R Journal 2017, 9(1):421-436,
// https://journal.r-project.org/articles/RJ-2017-016/RJ-2017-016.pdf).
//
// CITATION NOTE: EXPANSION_QUEUE.md attributed this to "Greenwell arXiv:1409.7002".
// That ID is wrong — the arXiv API returns "Entropy and Optimization of Portfolios"
// for 1409.7002, a Markowitz paper. The R Journal article above is the real source.
//
//   PD(z)   = (1/N) Σ_i f(x_i with x_j := z)     — average response as feature j varies
//   ICE[i](z) = f(x_i with x_j := z)             — the same curve, per instance
//
// PD is what LIME's neighborhood sampling already needs, and the pairing is
// complementary: PD answers "what does feature j do ON AVERAGE", LIME answers
// "what did feature j do for THIS prediction".
//
// CLOSED-FORM ORACLE (why this is unusually testable). For a LINEAR black box
// f(x) = Σ_k c_k x_k (no bias), PD has an exact answer independent of X's spread:
//
//   PD(z)   = c_j·z + Σ_{k≠j} c_k·mean(X[:,k])
//   ICE[i](z) = c_j·z + Σ_{k≠j} c_k·X[i][k]
//
// So the PD curve is an exact line of slope c_j whose intercept drops feature j.
// That identity pins the grid, the column replacement, the averaging, and the
// absence of any hidden normalization in one assertion. It is the same class of
// oracle as LIME's β_j = c_j·σ_j.
//
// Semantics read from reference source (sklearn main, 2026-10-09), not memory:
//   _partial_dependence.py:377   percentiles=(0.05, 0.95)   <- NOT (0,1), NOT min/max
//   _partial_dependence.py:378   grid_resolution=100, np.linspace(..., endpoint=True)
//   _partial_dependence.py:138-142  if n_uniques < grid_resolution, use the uniques
//   _partial_dependence.py:144-148  if the two empirical percentiles are allclose, RAISE
//   utils/stats.py:13-38         np.nanquantile -> LINEAR interpolation
//   _plot/partial_dependence.py:1249,1254  preds -= preds[..., 0, None]
//   _partial_dependence.py:498-499  subsample limits ICE curves DRAWN; the PD
//                                    average still uses ALL of X
//
// Coverage:
//   1.  pd_grid linspace with linear-interpolating quantiles (hand-computed)
//   2.  pd_grid low-resolution fallback to sorted uniques
//   3.  pd_grid percentiles (0,1) spans the observed range
//   4.  pd_grid grid_resolution == 1 is invalid
//   5.  pd_grid validation errors
//   6.  pd_grid degenerate (zero-width) column raises
//   7.  pd_grid does not mutate / reorder the caller's X
//   8.  average_ice == column means; empty and ragged rejected
//   9.  center_pd subtracts element [0], NOT the mean
//  10.  center_ice subtracts each ROW's own first element
//  11.  partial_dependence linear-model PD oracle (the headline test)
//  12.  partial_dependence linear-model ICE oracle (per instance)
//  13.  PD curve is the exact partial dependence of a linear model (slope test)
//  14.  constant model -> PD == bias at every grid point
//  15.  centered=true puts average[0] at exactly 0
//  16.  subsample caps ICE curves but PD averages over ALL of X
//  17.  determinism (same seed bit-identical; different seed same average)
//  18.  nonlinear model: PD != ICE[0] and PD != mean of a single row
//  19.  result metadata
//  20.  validation errors + no gradient residue

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/nn/nn.h"
#include "../include/nn/interpretability/partial_dependence.h"

static int passed = 0, failed = 0;

static void check(bool cond, const std::string& msg) {
    if (cond) { passed++; }
    else { failed++; std::cout << "  FAIL: " << msg << std::endl; }
}

static void check_close(double a, double b, double tol, const std::string& msg) {
    if (std::abs(a - b) <= tol) { passed++; }
    else {
        failed++;
        std::cout << "  FAIL: " << msg << " (got " << a << ", expected " << b
                  << ", tol " << tol << ")" << std::endl;
    }
}

static void check_throws(bool threw, const std::string& msg) {
    if (threw) { passed++; }
    else { failed++; std::cout << "  FAIL: " << msg << std::endl; }
}

// TRANSPOSITION WARNING (same trap as test_lime.cpp): Dense::forward computes
// `input * weights.transpose()` with `weights` shaped (out_features, in_features),
// so f(x)[j] == Σ_k weights[j][k]·x[k]. This helper takes W in the natural
// (in_dim, out_dim) human layout and stores it transposed.
static Model make_linear_model(size_t in_dim, size_t out_dim,
                               const std::vector<std::vector<double>>& W) {
    Model m;
    Dense* d = new Dense(in_dim, out_dim);
    d->weights.fill(0.0);
    for (size_t i = 0; i < in_dim; ++i)
        for (size_t j = 0; j < out_dim; ++j)
            d->weights[j][i] = W[i][j];
    d->bias.fill(0.0);
    m.add_layer(d);
    return m;
}

static Tensor col(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t[i][0] = v[i];
    return t;
}

// numpy's 'linear' quantile, restated INDEPENDENTLY of the implementation:
// sorted v, h = q*(n-1), i = floor(h), frac = h - i, result = v[i]*(1-frac)+v[i+1]*frac.
static double quantile_linear(const std::vector<double>& unsorted, double q) {
    std::vector<double> v = unsorted;
    std::sort(v.begin(), v.end());
    const double h = q * static_cast<double>(v.size() - 1);
    const size_t i = static_cast<size_t>(std::floor(h));
    const double frac = h - static_cast<double>(i);
    if (i + 1 >= v.size()) return v.back();
    return v[i] * (1.0 - frac) + v[i + 1] * frac;
}

static double col_mean(const Tensor& t, size_t j) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i) s += t[i][j];
    return s / static_cast<double>(t.rows);
}

// --- 1: the grid -----------------------------------------------------------

static void test_pd_grid_linspace() {
    // N = 10, values 0..9. numpy linear quantile at q: h = q*9.
    //   q=0.05 -> h=0.45 -> 0*(1-.45) + 1*.45 = 0.45
    //   q=0.95 -> h=8.55 -> 8*(1-.55) + 9*.55 = 8.55
    // grid_resolution=3 -> linspace(0.45, 8.55, 3, endpoint=True).
    const std::vector<double> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor x = col(v);

    std::vector<double> g = pd_grid(x, 0, 0.05, 0.95, 3);
    check(g.size() == 3, "grid has grid_resolution points");
    const double lo = 0.45, hi = 8.55;
    for (size_t k = 0; k < g.size(); ++k) {
        const double want = lo + (hi - lo) * static_cast<double>(k) / 2.0;
        check_close(g[k], want, 1e-12, "grid point " + std::to_string(k) +
                                       " == hand-computed linspace");
    }
    check_close(g.front(), 0.45, 1e-12, "grid endpoint == 5th-percentile quantile");
    check_close(g.back(), 8.55, 1e-12, "grid endpoint == 95th-percentile quantile");

    // The load-bearing detail: the quantile is LINEAR-INTERPOLATED, not a
    // nearest order statistic. A 'lower' rule would give 0.0 here.
    check(g.front() > 0.4 && g.front() < 0.5,
          "quantile interpolates (0.45), it is not the nearest order statistic (0)");

    // Unsorted input must give the same grid (the impl sorts a copy).
    Tensor unsorted = col({9, 0, 5, 3, 8, 1, 7, 2, 6, 4});
    std::vector<double> gu = pd_grid(unsorted, 0, 0.05, 0.95, 3);
    check_close(gu[1], g[1], 1e-12, "grid is order-independent");
}

// --- 2: low-resolution fallback --------------------------------------------

static void test_pd_grid_low_resolution_fallback() {
    // 10 unique values but grid_resolution=11 -> the reference uses the
    // SORTED UNIQUES (10 points), not an 11-point linspace.
    Tensor x = col({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::vector<double> g = pd_grid(x, 0, 0.05, 0.95, 11);
    check(g.size() == 10, "fewer uniques than grid_resolution -> uses the uniques");
    for (size_t k = 0; k < g.size(); ++k) {
        check_close(g[k], static_cast<double>(k), 1e-15,
                    "fallback grid point " + std::to_string(k) + " == sorted unique");
    }

    // Equal counts (uniques == grid_resolution) takes the LINALSPHERE branch per
    // the reference's `uniques.shape[0] < grid_resolution` (strict less-than), so
    // with 10 uniques and grid_resolution=10 we get a 10-point linspace whose
    // endpoints are the 5th/95th percentiles, NOT the min and max.
    std::vector<double> ge = pd_grid(x, 0, 0.05, 0.95, 10);
    check(ge.size() == 10, "uniques == grid_resolution -> linspace branch, 10 points");
    check_close(ge.front(), 0.45, 1e-12,
                "uniques == grid_resolution uses percentiles (strict < in reference)");
    check_close(ge.back(), 8.55, 1e-12, "...at the high end too");
}

// --- 3: percentiles (0,1) spans the observed range -------------------------

static void test_pd_grid_full_range() {
    // grid_resolution=3 with 4 uniques -> the LINALSPHERE branch (uniques
    // 4 >= 3), so the percentiles govern. A column of 4 values at spacing 2 with
    // p=(0,1) gives lo=2, hi=8 and a 3-point axis with the exact midpoint 5.
    Tensor x = col({2.0, 4.0, 6.0, 8.0});
    std::vector<double> g = pd_grid(x, 0, 0.0, 1.0, 3);
    check(g.size() == 3, "full-range grid has grid_resolution points");
    check_close(g.front(), 2.0, 1e-12, "p_lo=0 -> grid starts at the min");
    check_close(g.back(), 8.0, 1e-12, "p_hi=1 -> grid ends at the max");
    check_close(g[1], 5.0, 1e-12, "midpoint of a full-range grid is the midpoint");

    // BRANCH-ORDER NOTE: the reference tests the low-cardinality fallback BEFORE
    // computing percentiles, so with grid_resolution=5 this same 4-unique column
    // yields the 4 UNIQUES, not a 5-point linspace. Asserted explicitly because
    // getting this order wrong is easy and looks harmless.
    std::vector<double> gu = pd_grid(x, 0, 0.0, 1.0, 5);
    check(gu.size() == 4, "uniques(4) < grid_resolution(5) -> uniques branch, 4 points");
    check_close(gu.front(), 2.0, 1e-12, "uniques branch still starts at the min");
    check_close(gu.back(), 8.0, 1e-12, "uniques branch still ends at the max");

    // A 2-feature matrix: the grid must come from the REQUESTED column.
    Tensor two = Tensor::zeros(4, 2);
    for (size_t i = 0; i < 4; ++i) { two[i][0] = 2.0 + 2.0 * static_cast<double>(i);
                                     two[i][1] = 100.0 - static_cast<double>(i); }
    std::vector<double> g0 = pd_grid(two, 0, 0.0, 1.0, 3);
    std::vector<double> g1 = pd_grid(two, 1, 0.0, 1.0, 3);
    check_close(g0.front(), 2.0, 1e-12, "column 0 grid starts at column 0's min");
    check_close(g0.back(), 8.0, 1e-12, "column 0 grid ends at column 0's max");
    check_close(g1.front(), 97.0, 1e-12, "column 1 grid starts at column 1's min");
    check_close(g1.back(), 100.0, 1e-12, "column 1 grid ends at column 1's max");
}

// --- 4, 5: validation ------------------------------------------------------

static void test_pd_grid_resolution_one() {
    Tensor x = col({1, 2, 3});
    bool threw = false;
    try { pd_grid(x, 0, 0.05, 0.95, 1); } catch (...) { threw = true; }
    check_throws(threw, "grid_resolution == 1 throws (reference requires > 1)");
}

static void test_pd_grid_validation() {
    Tensor x = col({1, 2, 3});
    bool t1 = false, t2 = false, t3 = false, t4 = false, t5 = false, t6 = false;
    Tensor two = Tensor::zeros(3, 2);
    try { pd_grid(Tensor::zeros(0, 1), 0); } catch (...) { t1 = true; }
    try { pd_grid(two, 5); } catch (...) { t2 = true; }
    try { pd_grid(two, 0, 0.9, 0.1); } catch (...) { t3 = true; }
    try { pd_grid(two, 0, -0.1, 0.9); } catch (...) { t4 = true; }
    try { pd_grid(two, 0, 0.1, 1.5); } catch (...) { t5 = true; }
    try { pd_grid(two, 0, 0.05, 0.95, 0); } catch (...) { t6 = true; }
    check_throws(t1, "empty X throws");
    check_throws(t2, "feature index out of range throws");
    check_throws(t3, "p_lo >= p_hi throws");
    check_throws(t4, "percentile below [0,1] throws");
    check_throws(t5, "percentile above [0,1] throws");
    check_throws(t6, "grid_resolution == 0 throws");
}

// --- 6: degenerate column --------------------------------------------------

static void test_pd_grid_degenerate_column() {
    // BRANCH ORDER IS THE WHOLE STORY HERE. The reference evaluates the
    // low-cardinality fallback BEFORE computing percentiles, so a CONSTANT
    // column (1 unique) with any grid_resolution > 1 returns the single unique
    // value and never reaches the allclose check. It does not raise.
    Tensor flat = col({5.0, 5.0, 5.0, 5.0});
    std::vector<double> gf = pd_grid(flat, 0, 0.05, 0.95, 5);
    check(gf.size() == 1, "constant column takes the uniques branch -> 1 point");
    check_close(gf[0], 5.0, 1e-15, "...at that constant value");

    // The reference's "percentiles are too close" error therefore only fires on
    // the LINALSPHERE branch, which needs n_uniques >= grid_resolution. Reach it
    // with two distinct values 1e-10 apart: uniques(2) >= grid_resolution(2) is
    // false on the fallback, so the percentile branch runs and 8.5e-11 is far
    // under np.allclose's default atol=1e-8. A silently flat curve would read as
    // "this feature has no effect" rather than "this column has no spread",
    // which are completely different claims, so this must raise.
    Tensor tiny = col({0.0, 1e-10, 1e-10, 1e-10});
    bool threw = false;
    try { pd_grid(tiny, 0, 0.05, 0.95, 2); } catch (...) { threw = true; }
    check_throws(threw, "degenerate percentile range on the linspace branch throws");

    // A genuinely spread column at the same config must succeed — the guard is
    // about the WIDTH, not about having exactly two uniques.
    Tensor two_vals = col({0.0, 5.0, 0.0, 5.0});
    bool ok = true;
    std::vector<double> gt;
    try { gt = pd_grid(two_vals, 0, 0.05, 0.95, 2); } catch (...) { ok = false; }
    check(ok, "a genuinely spread column is not mistaken for degenerate");
    check(gt.size() == 2, "...and returns the requested number of points");
    // sorted = [0,0,5,5]; q=0.05 -> h=0.15 -> v[0]=0 (i=0, frac 0.15), so lo
    // is EXACTLY 0.0, and q=0.95 -> h=2.85 -> 5*(1-0.85)+5*0.85 = 5.0.
    check_close(gt.front(), 0.0, 1e-15, "5th-percentile endpoint is exactly 0.0");
    check_close(gt.back(), 5.0, 1e-15, "95th-percentile endpoint is exactly 5.0");
    check_close(gt.back() - gt.front(), 5.0, 1e-15,
                "...so the axis spans the full 5.0 width");
}

// --- 7: X is not mutated ---------------------------------------------------

static void test_pd_grid_does_not_mutate_x() {
    Tensor x = col({9, 0, 5, 3, 8});
    const std::vector<double> before(x.data.begin(), x.data.end());
    (void)pd_grid(x, 0, 0.05, 0.95, 7);
    bool same = true;
    for (size_t i = 0; i < before.size(); ++i) {
        if (x.data[i] != before[i]) { same = false; }
    }
    check(same, "pd_grid does not reorder or mutate the caller's X");
}

// --- 8: averaging ----------------------------------------------------------

static void test_average_ice_is_column_mean() {
    // 3 instances x 3 points. Column means are hand-computed.
    const std::vector<std::vector<double>> ice = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 7.0},
        {3.0, 6.0, 11.0},
    };
    std::vector<double> avg = average_ice(ice);
    check(avg.size() == 3, "average_ice returns one value per grid point");
    check_close(avg[0], 2.0, 1e-15, "column 0 mean == 2.0");
    check_close(avg[1], 4.0, 1e-15, "column 1 mean == 4.0");
    check_close(avg[2], 7.0, 1e-15, "column 2 mean == 7.0");

    // MUTATION GUARD: dividing by n-1 instead of n is a clean 3/2 factor here.
    // The fixture is chosen so this is a visible, non-floating-point-noise error.
    check(std::abs(avg[0] - 3.0) > 1e-9, "average_ice divides by n, not n-1");
}

static void test_average_ice_validation() {
    bool t1 = false, t2 = false, t3 = false;
    try { average_ice({}); } catch (...) { t1 = true; }
    try { average_ice({{}}); } catch (...) { t2 = true; }
    try { average_ice({{1.0, 2.0}, {3.0}}); } catch (...) { t3 = true; }
    check_throws(t1, "empty ICE matrix throws");
    check_throws(t2, "ICE with zero columns throws");
    check_throws(t3, "ragged ICE matrix throws");
}

// --- 9: PD centering subtracts element [0], not the mean --------------------

static void test_center_pd_subtracts_element_zero() {
    std::vector<double> pd = {1.0, 100.0, 100.0};
    std::vector<double> c = center_pd(pd);
    check(c.size() == 3, "center_pd preserves length");
    check_close(c[0], 0.0, 1e-15, "centered curve starts at exactly 0");
    check_close(c[1], 99.0, 1e-15, "centered value == pd[k] - pd[0]");
    check_close(c[2], 99.0, 1e-15, "...for the last point too");

    // MUTATION GUARD: mean-centering this fixture gives [0, 33, 33] instead of
    // [0, 99, 99]. The fixture is chosen so pd[0] != mean(pd) by a wide margin.
    const double mean = (1.0 + 100.0 + 100.0) / 3.0;
    check(std::abs(c[1] - (100.0 - mean)) > 1.0,
          "center_pd subtracts element [0], NOT the mean");
}

// --- 10: ICE centering is per row -----------------------------------------

static void test_center_ice_is_per_row() {
    std::vector<std::vector<double>> ice = {
        {1.0, 2.0, 3.0},
        {10.0, 20.0, 25.0},
    };
    center_ice(ice);
    check_close(ice[0][0], 0.0, 1e-15, "row 0 starts at 0");
    check_close(ice[0][1], 1.0, 1e-15, "row 0 shifted by its OWN first element");
    check_close(ice[0][2], 2.0, 1e-15, "row 0 last point");
    check_close(ice[1][0], 0.0, 1e-15, "row 1 starts at 0");
    check_close(ice[1][1], 10.0, 1e-15, "row 1 shifted by 10, not by row 0's 1");
    check_close(ice[1][2], 15.0, 1e-15, "row 1 last point");

    // A shared-offset (subtract row 0's first element from every row) bug would
    // give row 1 = [9, 19, 24]; the per-row form gives [0, 10, 15].
    check(std::abs(ice[1][1] - 19.0) > 1.0,
          "center_ice shifts each row by its own first element");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_partial_dependence\n";
    std::cout << "============================================================\n";

    test_pd_grid_linspace();
    test_pd_grid_low_resolution_fallback();
    test_pd_grid_full_range();
    test_pd_grid_resolution_one();
    test_pd_grid_validation();
    test_pd_grid_degenerate_column();
    test_pd_grid_does_not_mutate_x();
    test_average_ice_is_column_mean();
    test_average_ice_validation();
    test_center_pd_subtracts_element_zero();
    test_center_ice_is_per_row();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}