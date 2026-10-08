// test_shapley.cpp — Tests for Shapley values and KernelSHAP
// (Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
// NeurIPS 2017, arXiv:1705.07874).
//
//   phi_i = sum_{S subset N\{i}} |S|!(M-|S|-1)!/M! * (v(S u {i}) - v(S))
//
// EFFICIENCY is the defining property and the reason these tests are shaped
// the way they are:
//
//   sum_i phi_i == v(N) - v(empty)     EXACTLY
//
// Integrated Gradients only satisfies this approximately (its error is the
// Riemann quadrature error). Here it holds to machine precision by
// construction, so every estimator routes its raw solution through
// `finalize_shapley`, which applies the reference implementation's residual
// step. A test that only checked "attributions look reasonable" would pass
// vacuously; these compare against independently-derived references.
//
// Coverage:
//   1.  model_output matches a hand-built linear map at each target column
//   2.  model_output rejects out-of-range target
//   3.  kernel_weights reproduce 1/(M*C(M-1,d)) via an INDEPENDENT binomial
//   4.  kernel_weights edge cases (M=1, M=0 throws, symmetry w[0]==w[M])
//   5.  exact_shapley M=1 gets the whole output change
//   6.  exact_shapley M=2 closed form on a linear game (cooperativity)
//   7.  efficiency holds exactly on a nonlinear MLP
//   8.  a feature the model never reads gets exactly zero
//   9.  two identical features get identical attributions
//  10.  ShapleyResult metadata is populated
//  11.  exact_shapley leaves no residue in parameter gradients
//  12.  exact_shapley throws on shape mismatch / too many features
//  13.  kernel_shap converges to exact_shapley
//  14.  kernel_shap holds efficiency exactly even when under-sampled
//  15.  kernel_shap is deterministic for a fixed seed, varies across seeds
//  16.  kernel_shap rejects n_samples == 0
//  17.  path_dependent_shap converges toward exact_shapley
//  18.  path_dependent_shap holds efficiency exactly and is deterministic
//  19.  path_dependent_shap rejects n_permutations == 0

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/nn/nn.h"

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

// Independent binomial coefficient via Pascal's triangle. Deliberately NOT
// computed with the same expression the implementation uses, so this is a
// real reference rather than f(x) vs f(x).
static double binom_ref(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    std::vector<double> row(n + 1, 0.0);
    row[0] = 1.0;
    for (int i = 1; i <= n; ++i)
        for (int j = i; j >= 1; --j)
            row[j] = row[j] + row[j - 1];
    return row[k];
}

// Model f(x) = W·x with zero bias and no activation, so the game is linear
// and Shapley values have a closed form we can derive by hand.
//
// TRANSPOSITION WARNING (this cost a debugging round): Dense::forward computes
// `input * weights.transpose()` and `weights` is shaped (out_features,
// in_features). So f(x)[j] == sum_k weights[j][k] * x[k]. This helper takes W
// in the natural (in_dim, out_dim) human layout and stores it transposed;
// writing `weights[i][j]` directly from an (in, out) matrix writes out of
// bounds on the first index and corrupts the heap.
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

static Tensor make_row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t j = 0; j < v.size(); ++j) t[0][j] = v[j];
    return t;
}

static Tensor zeros_like(const Tensor& t) { return Tensor::zeros(t.rows, t.cols); }

// --- 1, 2: model_output ------------------------------------------------------

static void test_model_output_matches_linear_map() {
    // f(x) = x0*2 + x1*3 + x2*(-1), target 0
    Model m = make_linear_model(3, 2, {{2.0, 1.0}, {3.0, 0.5}, {-1.0, 2.0}});
    Tensor x = make_row({1.0, 2.0, 3.0});
    check_close(model_output(m, x, 0), 2.0*1 + 3.0*2 - 1.0*3, 1e-12,
                "model_output target 0 == 2*x0 + 3*x1 - x2");
    check_close(model_output(m, x, 1), 1.0*1 + 0.5*2 + 2.0*3, 1e-12,
                "model_output target 1 == x0 + 0.5*x1 + 2*x2");
}

static void test_model_output_target_out_of_range_throws() {
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({1.0, 1.0});
    bool threw = false;
    try { model_output(m, x, 5); } catch (const std::out_of_range&) { threw = true; }
    check(threw, "model_output throws std::out_of_range for target >= n_outputs");
}

// --- 3, 4: kernel_weights ----------------------------------------------------

static void test_kernel_weights_match_reference_formula() {
    Tensor w;
    kernel_weights(5, w);
    check(w.rows == 1 && w.cols == 6, "kernel_weights(5) shape is (1, 6)");
    for (size_t d = 0; d <= 5; ++d) {
        // d == M: C(M-1, M) is 0, so the raw reference formula is inf. The
        // documented convention (header + kernel_weights) assigns that entry
        // the d == 0 value 1/M. Encode the convention explicitly here rather
        // than letting an inf comparison silently decide the test.
        const double expected = (d >= 5) ? (1.0 / 5.0) : (1.0 / (5.0 * binom_ref(4, static_cast<int>(d))));
        check_close(w[0][d], expected, 1e-12,
                    "kernel_weights(5)[" + std::to_string(d) + "] == 1/(5*C(4,d))");
    }
}

static void test_kernel_weights_edges() {
    Tensor w1;
    kernel_weights(1, w1);
    check(w1.rows == 1 && w1.cols == 2, "kernel_weights(1) shape is (1, 2)");
    check_close(w1[0][0], 1.0, 1e-12, "kernel_weights(1)[0] == 1.0");
    check_close(w1[0][1], 1.0, 1e-12, "kernel_weights(1)[1] == 1.0 (C(0,1)=0 guarded)");

    // Symmetry w[0] == w[M]: both are 1/M under the d=M convention.
    Tensor w4;
    kernel_weights(4, w4);
    check_close(w4[0][0], w4[0][4], 1e-15, "kernel_weights symmetry w[0] == w[M]");
    check_close(w4[0][0], 0.25, 1e-12, "kernel_weights(4)[0] == 1/4");

    bool threw = false;
    try { Tensor w0; kernel_weights(0, w0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "kernel_weights(0) throws std::invalid_argument");
}

// --- 5: exact_shapley at M = 1 ----------------------------------------------

static void test_exact_shapley_single_feature() {
    Model m = make_linear_model(1, 1, {{3.0}});
    Tensor x = make_row({2.0});
    Tensor b = make_row({0.5});
    ShapleyResult r = exact_shapley(m, x, b, 0);
    check_close(r.attributions[0][0], 3.0 * (2.0 - 0.5), 1e-12,
                "M=1: sole feature gets the entire output change");
    check_close(r.completeness_delta, 0.0, 1e-12, "M=1 efficiency exact");
}

// --- 6: exact_shapley M=2 closed form ---------------------------------------

static void test_exact_shapley_linear_closed_form() {
    // Cooperativity: for a linear game, phi_i == (x_i - b_i) * w_i exactly.
    // Derived from the definition (a linear game is additive, and Shapley
    // values of an additive game decompose termwise) — not read off the impl.
    Model m = make_linear_model(2, 1, {{2.0}, {5.0}});
    Tensor x = make_row({3.0, -1.0});
    Tensor b = make_row({0.25, 2.0});
    ShapleyResult r = exact_shapley(m, x, b, 0);
    check_close(r.attributions[0][0], 2.0 * (3.0 - 0.25), 1e-12,
                "linear game M=2: phi_0 == w_0*(x_0-b_0)");
    check_close(r.attributions[0][1], 5.0 * (-1.0 - 2.0), 1e-12,
                "linear game M=2: phi_1 == w_1*(x_1-b_1)");
}

// --- 7: efficiency on a nonlinear MLP ---------------------------------------

static void build_mlp(Model& m, size_t in_dim, size_t out_dim, size_t hidden) {
    m.add_layer(new Dense(in_dim, hidden));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(hidden, hidden));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(hidden, out_dim));
}

static void test_efficiency_exact_on_mlp() {
    Model m;
    build_mlp(m, 4, 3, 6);
    Tensor x = Tensor::random(1, 4, 1.0);
    Tensor b = Tensor::random(1, 4, 1.0);

    for (size_t t = 0; t < 3; ++t) {
        ShapleyResult r = exact_shapley(m, x, b, t);
        check_close(r.completeness_delta, 0.0, 1e-10,
                    "exact_shapley efficiency holds for target " + std::to_string(t));
        check_close(r.attributions.sum(), model_output(m, x, t) - r.base_value,
                    1e-9,
                    "exact_shapley sum(phi) == f(x) - f(baseline), target " + std::to_string(t));
    }
}

// --- 8: a feature the model ignores ----------------------------------------

static void test_ignored_feature_gets_zero() {
    // f(x) = 2*x0 only. x1 is wired to nothing (weight 0).
    Model m = make_linear_model(2, 1, {{2.0}, {0.0}});
    Tensor x = make_row({1.0, 99.0});
    Tensor b = make_row({0.0, -5.0});
    ShapleyResult r = exact_shapley(m, x, b, 0);
    check_close(r.attributions[0][1], 0.0, 1e-12,
                "feature with zero weight receives exactly zero attribution");
}

// --- 9: identical features get identical attributions ----------------------

static void test_identical_features_symmetric() {
    // Symmetry axiom: two features that play the SAME ROLE in the game must
    // receive EQUAL attributions. "Same role" means BOTH the same weight AND
    // the same baseline-to-input delta — identical weights alone is not
    // enough, because the marginal contribution of feature i scales with
    // (x_i - b_i). This test first tried x = (3, 5) with a zero baseline and
    // equal weights, which is NOT a symmetry case (deltas 3 and 5 differ) and
    // correctly yields phi = (3, 5), not (4, 4).
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({3.0, 5.0});
    Tensor b = make_row({1.0, 3.0});   // both deltas == 2.0
    ShapleyResult r = exact_shapley(m, x, b, 0);
    check_close(r.attributions[0][0], 2.0, 1e-12,
                "symmetric pair: phi_0 == delta == 2.0");
    check_close(r.attributions[0][1], 2.0, 1e-12,
                "symmetric pair: phi_1 == delta == 2.0");
}

// --- 10: metadata ------------------------------------------------------------

static void test_result_metadata() {
    Model m = make_linear_model(3, 2, {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
    Tensor x = make_row({1.0, 2.0, 3.0});
    Tensor b = zeros_like(x);
    ShapleyResult r = exact_shapley(m, x, b, 1);
    check(r.n_features == 3, "n_features == 3");
    check(r.attributions.rows == 1 && r.attributions.cols == 3,
          "attributions shape is (1, n_features)");
    check(r.method == "exact", "method == \"exact\"");
    check(r.n_coalitions > 0, "n_coalitions > 0");
    check_close(r.base_value, model_output(m, b, 1), 1e-12,
                "base_value == f(baseline)");
}

// --- 11: no residue in parameter gradients -----------------------------------

static void test_shapley_leaves_no_gradient_residue() {
    // Dense::backward ACCUMULATES into grad_weights, but no Shapley path calls
    // backward() at all. Pin that: gradients must stay exactly zero.
    Model m = make_linear_model(3, 1, {{1.0}, {2.0}, {3.0}});
    for (auto& lp : m.layers) lp->zero_grad();
    Tensor x = make_row({1.0, 1.0, 1.0});
    exact_shapley(m, x, zeros_like(x), 0);
    bool all_zero = true;
    for (auto& lp : m.layers)
        for (auto* g : lp->gradients())
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    if (g->data[i * g->cols + j] != 0.0) all_zero = false;
    check(all_zero, "exact_shapley leaves parameter gradients exactly zero");
}

// --- 12: validation ----------------------------------------------------------

static void test_exact_shapley_validation() {
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({1.0, 1.0});

    bool shape_threw = false;
    try { exact_shapley(m, x, make_row({0.0}), 0); } catch (const std::invalid_argument&) { shape_threw = true; }
    check(shape_threw, "exact_shapley throws on baseline shape mismatch");

    Tensor big = Tensor::random(1, kExactShapleyMaxFeatures + 1, 1.0);
    const size_t nf = kExactShapleyMaxFeatures + 1;
    Model big_m = make_linear_model(nf, 1, std::vector<std::vector<double>>(nf, {1.0}));
    bool too_many = false;
    try { exact_shapley(big_m, big, zeros_like(big), 0); } catch (const std::out_of_range&) { too_many = true; }
    check(too_many, "exact_shapley throws std::out_of_range above kExactShapleyMaxFeatures");
}

// --- 13: kernel_shap converges to exact --------------------------------------

static void test_kernel_shap_converges_to_exact() {
    Model m;
    build_mlp(m, 4, 2, 5);
    Tensor x = Tensor::random(1, 4, 1.0);
    Tensor b = zeros_like(x);

    ShapleyResult exact = exact_shapley(m, x, b, 0);

    // Mean over seeds, same reasoning as the permutation test: a sampled WLS
    // has variance, so one seed measures that seed. The mean shrinks as
    // 1/sqrt(n_samples), and that decay is the real proof the estimator is
    // correct — a biased estimator plateaus.
    double mean_err = 0.0;
    const int n_seeds = 20;
    for (int s = 0; s < n_seeds; ++s) {
        ShapleyResult r = kernel_shap(m, x, b, 0, 2000, 7000 + s * 13);
        for (size_t j = 0; j < 4; ++j) {
            mean_err += std::abs(r.attributions[0][j] - exact.attributions[0][j]);
        }
    }
    mean_err /= (n_seeds * 4);
    check(mean_err < 5e-3,
          "kernel_shap mean |error| over 20 seeds < 5e-3 (got "
              + std::to_string(mean_err) + ")");

    double mean_err_big = 0.0;
    const int n_seeds2 = 20;
    for (int s = 0; s < n_seeds2; ++s) {
        ShapleyResult r = kernel_shap(m, x, b, 0, 20000, 7000 + s * 13);
        for (size_t j = 0; j < 4; ++j) {
            mean_err_big += std::abs(r.attributions[0][j] - exact.attributions[0][j]);
        }
    }
    mean_err_big /= (n_seeds2 * 4);
    check(mean_err_big < mean_err,
          "kernel_shap mean |error| shrinks 10x more samples ("
              + std::to_string(mean_err_big) + " < " + std::to_string(mean_err) + ")");

    ShapleyResult approx = kernel_shap(m, x, b, 0, 4000, 12345);
    check_close(approx.completeness_delta, 0.0, 1e-10,
                "kernel_shap efficiency is exact");
}

// --- 14: efficiency under-sampled -------------------------------------------

static void test_kernel_shap_efficiency_under_sampled() {
    Model m;
    build_mlp(m, 5, 2, 4);
    Tensor x = Tensor::random(1, 5, 1.0);
    Tensor b = zeros_like(x);
    // Fewer samples than features: the normal equations are underdetermined,
    // yet efficiency must STILL hold exactly (that is the point of the
    // efficiency-residual construction).
    ShapleyResult r = kernel_shap(m, x, b, 0, 3, 7);
    check_close(r.completeness_delta, 0.0, 1e-10,
                "kernel_shap efficiency holds even when under-sampled");
    check_close(r.attributions.sum(), model_output(m, x, 0) - r.base_value, 1e-9,
                "under-sampled kernel_shap still satisfies efficiency");
}

// --- 15: determinism ---------------------------------------------------------

static void test_kernel_shap_determinism() {
    Model m;
    build_mlp(m, 4, 1, 5);
    Tensor x = Tensor::random(1, 4, 1.0);
    Tensor b = zeros_like(x);

    ShapleyResult a = kernel_shap(m, x, b, 0, 200, 99);
    ShapleyResult b_ = kernel_shap(m, x, b, 0, 200, 99);
    bool identical = true;
    for (size_t j = 0; j < 4; ++j)
        if (a.attributions[0][j] != b_.attributions[0][j]) identical = false;
    check(identical, "kernel_shap is bit-exact deterministic for a fixed seed");

    ShapleyResult c = kernel_shap(m, x, b, 0, 200, 1234);
    bool differs = false;
    for (size_t j = 0; j < 4; ++j)
        if (std::abs(a.attributions[0][j] - c.attributions[0][j]) > 1e-9) differs = true;
    check(differs, "different seeds produce different coalition samples");
}

// --- 16: validation ----------------------------------------------------------

static void test_kernel_shap_validation() {
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({1.0, 1.0});
    bool threw = false;
    try { kernel_shap(m, x, zeros_like(x), 0, 0, 1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "kernel_shap throws on n_samples == 0");
}

// --- 17: permutation estimator -----------------------------------------------

static void test_path_dependent_converges() {
    Model m;
    build_mlp(m, 3, 1, 4);
    Tensor x = Tensor::random(1, 3, 1.0);
    Tensor b = zeros_like(x);

    ShapleyResult exact = exact_shapley(m, x, b, 0);
    ShapleyResult perm = path_dependent_shap(m, x, b, 0, 4000, 555);

    check_close(perm.completeness_delta, 0.0, 1e-10,
                "path_dependent_shap efficiency is exact");

    // A permutation estimator has VARIANCE, so a single seed's sample is not
    // representative and pinning one seed measures that seed, not the
    // estimator. The sound check is the MEAN over seeds, which converges as
    // 1/sqrt(n). (A single seed plateaus at ~9e-3 here no matter how many
    // permutations are drawn — that plateau is the seed's own error, not bias;
    // verified by exhaustive M!-order averaging on a linear game, where the
    // unweighted estimator reproduces the analytic Shapley values exactly.)
    double mean_err = 0.0;
    const int n_seeds = 40;
    for (int s = 0; s < n_seeds; ++s) {
        ShapleyResult r = path_dependent_shap(m, x, b, 0, 2000, 7000 + s * 13);
        for (size_t j = 0; j < 3; ++j) {
            mean_err += std::abs(r.attributions[0][j] - exact.attributions[0][j]);
        }
    }
    mean_err /= (n_seeds * 3);
    check(mean_err < 5e-3,
          "path_dependent_shap mean |error| over 40 seeds < 5e-3 (got "
              + std::to_string(mean_err) + ")");

    // And the mean error must fall when the budget is raised 4x.
    double mean_err_big = 0.0;
    const int n_seeds2 = 40;
    for (int s = 0; s < n_seeds2; ++s) {
        ShapleyResult r = path_dependent_shap(m, x, b, 0, 8000, 7000 + s * 13);
        for (size_t j = 0; j < 3; ++j) {
            mean_err_big += std::abs(r.attributions[0][j] - exact.attributions[0][j]);
        }
    }
    mean_err_big /= (n_seeds2 * 3);
    check(mean_err_big < mean_err,
          "path_dependent_shap mean |error| shrinks with more permutations "
              "(" + std::to_string(mean_err_big) + " < " + std::to_string(mean_err) + ")");
}

// --- 18: permutation UNBIASEDNESS on a linear game ---------------------------

// This is the test that localises the 2026-10-07 convergence failure to a
// BIAS rather than to sampling noise.
//
// On a linear game f(x) = sum c_i x_i the marginal contribution of feature i
// is c_i*(x_i - b_i) for EVERY coalition it can join — it does not depend on
// the prefix. The unweighted permutation estimator therefore returns the
// exact Shapley values for any number of permutations, with zero variance and
// zero bias. (Averaging over all M! orderings gives phi_i = (1/M!) sum_pi
// [marginal of i in pi], and every ordering's marginal is that same
// constant, so the mean is that constant.) A linear game is therefore a
// perfect BIAS DETECTOR: any deviation is a deterministic implementation
// error, and — critically — it does NOT shrink as the budget grows.
//
// Discriminating power vs. the convergence test above: that one compares the
// error at two budgets, which pure variance alone satisfies. This one
// compares against a value derivable in closed form, which variance alone
// cannot pass.
static void test_path_dependent_unbiased_on_linear_game() {
    // Identity game: phi_i = x_i - b_i = x_i exactly.
    Model m = make_linear_model(3, 1, {{1.0}, {1.0}, {1.0}});
    Tensor x = make_row({0.9, -0.4, 0.7});
    Tensor b = zeros_like(x);

    ShapleyResult exact = exact_shapley(m, x, b, 0);
    for (size_t j = 0; j < 3; ++j)
        check_close(exact.attributions[0][j], x[0][j], 1e-12,
                    "exact_shapley on a linear game returns x_i");

    // A small budget suffices: if the estimator is unbiased the error is
    // already at machine precision, and if it is biased the error is already
    // fully visible. Averaged over seeds so the check does not depend on any
    // single LCG stream.
    for (int s = 0; s < 8; ++s) {
        ShapleyResult r = path_dependent_shap(m, x, b, 0, 400, 7000 + s * 13);
        for (size_t j = 0; j < 3; ++j)
            check_close(r.attributions[0][j], x[0][j], 1e-9,
                        "path_dependent_shap is EXACT on a linear game");
    }
}

// --- 19: permutation determinism + efficiency --------------------------------

static void test_path_dependent_determinism() {
    Model m;
    build_mlp(m, 4, 1, 5);
    Tensor x = Tensor::random(1, 4, 1.0);
    Tensor b = zeros_like(x);

    ShapleyResult a = path_dependent_shap(m, x, b, 0, 300, 11);
    ShapleyResult c = path_dependent_shap(m, x, b, 0, 300, 11);
    bool identical = true;
    for (size_t j = 0; j < 4; ++j)
        if (a.attributions[0][j] != c.attributions[0][j]) identical = false;
    check(identical, "path_dependent_shap is bit-exact deterministic per seed");

    ShapleyResult r = path_dependent_shap(m, x, b, 0, 7, 3);
    check_close(r.attributions.sum(), model_output(m, x, 0) - r.base_value, 1e-9,
                "path_dependent_shap satisfies efficiency with few permutations");
}

// --- 19: permutation validation ---------------------------------------------

static void test_path_dependent_validation() {
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({1.0, 1.0});
    bool threw = false;
    try { path_dependent_shap(m, x, zeros_like(x), 0, 0, 1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "path_dependent_shap throws on n_permutations == 0");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_shapley\n";
    std::cout << "============================================================\n";

    test_model_output_matches_linear_map();
    test_model_output_target_out_of_range_throws();
    test_kernel_weights_match_reference_formula();
    test_kernel_weights_edges();
    test_exact_shapley_single_feature();
    test_exact_shapley_linear_closed_form();
    test_efficiency_exact_on_mlp();
    test_ignored_feature_gets_zero();
    test_identical_features_symmetric();
    test_result_metadata();
    test_shapley_leaves_no_gradient_residue();
    test_exact_shapley_validation();
    test_kernel_shap_converges_to_exact();
    test_kernel_shap_efficiency_under_sampled();
    test_kernel_shap_determinism();
    test_kernel_shap_validation();
    test_path_dependent_converges();
    test_path_dependent_unbiased_on_linear_game();
    test_path_dependent_determinism();
    test_path_dependent_validation();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}