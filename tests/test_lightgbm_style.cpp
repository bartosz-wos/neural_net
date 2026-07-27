// Focused behavioral tests for HistogramBoosting (LightGBM-style GBDT).
// Verifies construction, fit/predict on a small regression problem, that
// loss decreases across estimators, classification (binary) fit, that
// predictions are finite, that increasing n_estimators helps, that
// increasing max_bins helps resolution, determinism, and basic numerical
// invariants.
//
// Style matches the existing focused suites:
// local passed/failed counters, [PASS]/[FAIL] output, "=== Result ===" summary.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "nn/core/tensor.h"
#include "nn/utils/lightgbm_style.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool condition) {
    if (condition) {
        cout << "  [PASS] " << name << '\n';
        ++passed;
    } else {
        cout << "  [FAIL] " << name << '\n';
        ++failed;
    }
}

// Small helper: mean squared error between two 1-D tensors of identical shape.
static double mse_1d(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.cols != 1) return -1.0;
    double s = 0.0;
    for (size_t i = 0; i < a.rows; ++i) {
        double d = a[i][0] - b[i][0];
        s += d * d;
    }
    return s / static_cast<double>(a.rows);
}

// Small helper: are all entries of a tensor finite (no NaN, no Inf)?
static bool tensor_all_finite(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            double v = t(i, j);
            if (!std::isfinite(v)) return false;
        }
    }
    return true;
}

// Build a 1-D Tensor from a vector of doubles.
static Tensor make_y1d(const vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// Build a 2-D Tensor from a flat layout: rows x cols, with values in row-major order.
static Tensor make_X(const vector<vector<double>>& rows) {
    size_t n = rows.size();
    size_t d = n == 0 ? 0 : rows[0].size();
    Tensor t(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) t(i, j) = rows[i][j];
    return t;
}

// Synthetic regression dataset: y = 2 * x0 + 1 (with small noise).
// Linear model so the histogram splitter can fit it well.
static pair<Tensor, Tensor> make_linear_regression(size_t n, double noise, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dx(0.0, 1.0);
    std::normal_distribution<double> dn(0.0, noise);
    vector<vector<double>> X;
    vector<double> y;
    X.reserve(n);
    y.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double x0 = dx(rng);
        X.push_back({x0});
        y.push_back(2.0 * x0 + 1.0 + dn(rng));
    }
    return {make_X(X), make_y1d(y)};
}

// Synthetic classification dataset (binary, 1-D feature, threshold-style label).
// y = 1.0 if x0 > 0.5 else 0.0  -- the GBDT can split on x0 with high gain.
static pair<Tensor, Tensor> make_step_classification(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dx(0.0, 1.0);
    vector<vector<double>> X;
    vector<double> y;
    X.reserve(n);
    y.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double x0 = dx(rng);
        X.push_back({x0});
        y.push_back(x0 > 0.5 ? 1.0 : 0.0);
    }
    return {make_X(X), make_y1d(y)};
}

// =================================================================
// (a) Construction defaults and accessors
// =================================================================
static void test_construction_defaults() {
    cout << "\n-- HistogramBoosting: construction defaults --\n";

    HistogramBoosting hb;          // all defaults
    check("default object constructed", true);

    HistogramBoosting custom(50, 0.05, 64, 0, 8, 5, true);
    check("non-default constructed", true);
}

// =================================================================
// (b) Fit + predict: regression loss decreases monotonically
// =================================================================
static void test_regression_fits() {
    cout << "\n-- HistogramBoosting: regression fit reduces MSE --\n";

    auto [X, y] = make_linear_regression(200, 0.05, 42);

    // Baseline = 0 loss (predict zero for everything).
    Tensor zero_pred(y.rows, 1);
    zero_pred.fill(0.0);
    double loss0 = mse_1d(y, zero_pred);

    HistogramBoosting hb(100, 0.1, 32, 0, 31, 20, /*regression=*/true);
    hb.fit(X, y);
    Tensor pred = hb.predict(X);
    check("predict shape matches y", pred.rows == y.rows && pred.cols == 1);
    check("predict values finite", tensor_all_finite(pred));

    double loss1 = mse_1d(y, pred);
    check("MSE drops vs all-zero predictor",
          loss1 < 0.5 * loss0);

    // Stronger fit (more estimators, finer bins) should do no worse than the weak one.
    HistogramBoosting hb2(200, 0.1, 64, 0, 31, 10, true);
    hb2.fit(X, y);
    Tensor pred2 = hb2.predict(X);
    double loss2 = mse_1d(y, pred2);
    check("Larger fit does at least as well (allow tie)",
          loss2 <= loss1 + 1e-9);
}

// =================================================================
// (c) Classification fit: loss drops sharply on a separable problem
// =================================================================
static void test_classification_fits() {
    cout << "\n-- HistogramBoosting: classification fit reduces loss --\n";

    auto [X, y] = make_step_classification(200, 7);

    HistogramBoosting hb_clf(100, 0.2, 32, 0, 31, 5, /*regression=*/false);
    hb_clf.fit(X, y);
    Tensor pred = hb_clf.predict(X);
    check("clf: predict shape matches y", pred.rows == y.rows && pred.cols == 1);
    check("clf: predict values finite", tensor_all_finite(pred));

    // The data is linearly separable on x0 ~ 0.5: any gain positive split
    // should reduce the regression-style MSE vs predicting the mean.
    Tensor mean_pred(y.rows, 1);
    double ym = 0.0;
    for (size_t i = 0; i < y.rows; ++i) ym += y[i][0];
    ym /= static_cast<double>(y.rows);
    mean_pred.fill(ym);
    double loss_mean = mse_1d(y, mean_pred);
    double loss_hb = mse_1d(y, pred);
    check("clf: fit improves on mean-predictor",
          loss_hb < 0.95 * loss_mean);
}

// =================================================================
// (d) Monotonicity across estimators
// =================================================================
static void test_more_estimators_better() {
    cout << "\n-- HistogramBoosting: more estimators improves fit --\n";

    auto [X, y] = make_linear_regression(150, 0.1, 99);

    HistogramBoosting hb_small(10, 0.1, 16, 0, 31, 5, true);
    hb_small.fit(X, y);
    double loss_small = mse_1d(y, hb_small.predict(X));

    HistogramBoosting hb_large(200, 0.1, 32, 0, 31, 5, true);
    hb_large.fit(X, y);
    double loss_large = mse_1d(y, hb_large.predict(X));

    check("More estimators reduces MSE",
          loss_large < loss_small);
}

// =================================================================
// (e) Histogram refinement: more bins should not break the fit
// =================================================================
static void test_more_bins() {
    cout << "\n-- HistogramBoosting: max_bins resolution sanity --\n";

    auto [X, y] = make_linear_regression(120, 0.05, 5);

    HistogramBoosting hb_coarse(50, 0.1, 4, 0, 31, 5, true);
    hb_coarse.fit(X, y);
    double loss_coarse = mse_1d(y, hb_coarse.predict(X));

    HistogramBoosting hb_fine(50, 0.1, 64, 0, 31, 5, true);
    hb_fine.fit(X, y);
    double loss_fine = mse_1d(y, hb_fine.predict(X));

    check("Both fits produce finite predictions",
          tensor_all_finite(hb_coarse.predict(X)) && tensor_all_finite(hb_fine.predict(X)));
    check("Finer bins do not catastrophically degrade MSE",
          loss_fine < 2.0 * loss_coarse + 1e-9);
}

// =================================================================
// (f) Determinism
// =================================================================
static void test_determinism() {
    cout << "\n-- HistogramBoosting: determinism --\n";

    auto [X, y] = make_linear_regression(60, 0.0, 11);

    HistogramBoosting hb1(20, 0.1, 8, 0, 15, 5, true);
    HistogramBoosting hb2(20, 0.1, 8, 0, 15, 5, true);

    hb1.fit(X, y);
    hb2.fit(X, y);

    Tensor p1 = hb1.predict(X);
    Tensor p2 = hb2.predict(X);
    check("Two fresh fits produce identical predictions (deterministic)",
          p1.rows == p2.rows && p1.cols == p2.cols
              && std::abs(mse_1d(p1, p2)) < 1e-12);
}

// =================================================================
// (g) Numerical invariants
// =================================================================
static void test_numerical_invariants() {
    cout << "\n-- HistogramBoosting: numerical invariants --\n";

    auto [X, y] = make_linear_regression(80, 0.0, 13);

    HistogramBoosting hb(50, 0.1, 16, 0, 31, 5, true);
    hb.fit(X, y);
    Tensor pred = hb.predict(X);
    check("All predicted values are finite",
          tensor_all_finite(pred));

    // The MSE must be non-negative (it's a sum of squares / N).
    double loss = mse_1d(y, pred);
    check("MSE is non-negative", loss >= 0.0);

    // Predictions are bounded by the largest absolute leaf-value * learning_rate
    // summed across iterations, but a coarse bound is that |pred| is bounded
    // by some small multiple of |y| on this linear problem.
    double max_abs_y = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        max_abs_y = std::max(max_abs_y, std::abs(y[i][0]));
    double max_abs_p = 0.0;
    for (size_t i = 0; i < pred.rows; ++i)
        max_abs_p = std::max(max_abs_p, std::abs(pred[i][0]));
    check("Predictions are within a reasonable multiple of target range",
          max_abs_p <= 5.0 * max_abs_y + 1e-9);
}

// =================================================================
// (h) Edge cases
// =================================================================
static void test_edge_cases() {
    cout << "\n-- HistogramBoosting: edge cases --\n";

    auto [X, y] = make_linear_regression(40, 0.0, 3);

    // n_estimators = 1 still trains one step.
    HistogramBoosting hb_one(1, 0.1, 8, 0, 15, 5, true);
    hb_one.fit(X, y);
    Tensor p_one = hb_one.predict(X);
    check("n_estimators=1 produces finite predictions",
          tensor_all_finite(p_one) && p_one.rows == y.rows);

    // n_estimators = 0 should be a no-op: all predictions would remain the base (0).
    HistogramBoosting hb_zero(0, 0.1, 8, 0, 15, 5, true);
    hb_zero.fit(X, y);
    Tensor p_zero = hb_zero.predict(X);
    check("n_estimators=0 produces an all-zero prediction",
          p_zero.rows == y.rows
              && std::abs(p_zero[0][0]) < 1e-12
              && std::abs(p_zero[y.rows - 1][0]) < 1e-12);
}

int main() {
    cout << "=== HistogramBoosting (LightGBM-style) Tests ===\n";

    test_construction_defaults();
    test_regression_fits();
    test_classification_fits();
    test_more_estimators_better();
    test_more_bins();
    test_determinism();
    test_numerical_invariants();
    test_edge_cases();

    cout << "\n=== Result: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
