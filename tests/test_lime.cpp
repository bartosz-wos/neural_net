// test_lime.cpp — Tests for LIME (Ribeiro, Singh, Guestrin, "Why Should I Trust
// You?: Explaining the Predictions of Any Classifier", KDD 2016,
// arXiv:1606.03878).
//
//   g(z) = β₀ + Σ_j β_j·z_j      fit on a Gaussian neighborhood around x,
//   w_i = sqrt(exp(-d_i²/ω²))    weighted by distance to the instance
//
// This is the one attribution method in the repo that uses NO gradients. The
// reference IS the weighted least-squares solve, so the whole method is
// verifiable in CLOSED FORM rather than against finite differences — which is
// why these tests use an exact oracle instead of tolerances:
//
//   linear model f(x) = Σ_j c_j x_j, neighborhood raw[i][j] = z[i][j]·σ_j + μ_j
//     ⟹ f(raw[i]) = Σ_j c_j σ_j z[i][j] + Σ_j c_j μ_j   (exactly affine in z)
//     ⟹ β_j = c_j·σ_j,  β₀ = Σ_j c_j μ_j,  local_pred = Σ_j β_j z[0][j] + β₀
//
// That identity pins the kernel, the weighting, the centering, the intercept
// reconstruction, the perturbation scale AND the distance metric in one shot.
// A bug anywhere in that chain moves the answer; nothing here is a threshold
// that a wrong implementation can still pass.
//
// Coverage:
//   1.  exp_kernel matches sqrt(exp(-d²/ω²)) at hand-computed points
//   2.  exp_kernel default width == sqrt(M)·0.75
//   3.  exp_kernel is 1.0 at d == 0 and decreasing in d
//   4.  fit_local_surrogate recovers an EXACTLY linear relationship
//   5.  fit_local_surrogate reproduces the hand-derived ridge shrinkage
//   6.  fit_local_surrogate intercept matches the sklearn weighted-mean rule
//   7.  fit_local_surrogate score == weighted R² (hand-computed)
//   8.  fit_local_surrogate validation errors
//   9.  lime_explain closed-form oracle on a linear model (the headline test)
//  10.  lime_explain local_prediction == model_output at x
//  11.  lime_explain feature ranking uses |β| (negative-dominant feature)
//  12.  lime_explain num_features selection actually restricts the fit
//  13.  lime_explain determinism per seed, variation across seeds
//  14.  lime_explain validation errors
//  15.  lime_explain leaves no parameter-gradient residue
//  16.  lime_explain result metadata
//  17.  pivoting fixture (forces a row swap) + standardized-distance fixture

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

// TRANSPOSITION WARNING (same trap as test_shapley.cpp): Dense::forward computes
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

static Tensor make_row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t j = 0; j < v.size(); ++j) t[0][j] = v[j];
    return t;
}

// Weighted mean — an INDEPENDENT restatement of the rule the implementation
// must follow (sample-weighted mean of y), deliberately not sharing code.
static double weighted_mean(const std::vector<double>& y,
                            const std::vector<double>& w) {
    double sw = 0.0, swy = 0.0;
    for (size_t i = 0; i < y.size(); ++i) { sw += w[i]; swy += w[i] * y[i]; }
    return swy / sw;
}

// --- 1, 2, 3: the kernel ----------------------------------------------------

static void test_exp_kernel_values() {
    const double w = 2.0;
    // Reference: sqrt(exp(-(d²)/w²)).
    check_close(exp_kernel(0.0, w), 1.0, 1e-15, "exp_kernel(0) == 1.0");
    check_close(exp_kernel(1.0, w), std::sqrt(std::exp(-1.0 / 4.0)), 1e-15,
                "exp_kernel(1, 2) == sqrt(exp(-1/4))");
    check_close(exp_kernel(2.0, w), std::sqrt(std::exp(-1.0)), 1e-15,
                "exp_kernel(2, 2) == sqrt(exp(-1))");
    // This is the load-bearing detail: the reference applies sqrt OUTSIDE the
    // exp, so the value is exp(-d²/2w²), NOT exp(-d²/w²). A plain
    // exp(-d²/w²) would give 0.3679 here instead of 0.6065.
    const double got = exp_kernel(2.0, 2.0);
    check(std::abs(got - std::exp(-1.0)) > 1e-6,
          "exp_kernel is sqrt(exp(...)), NOT exp(-d²/w²)");
}

static void test_exp_kernel_default_width() {
    check_close(exp_kernel_width(9), std::sqrt(9.0) * 0.75, 1e-15,
                "default kernel width == sqrt(M)*0.75");
    check_close(exp_kernel_width(16), 4.0 * 0.75, 1e-15,
                "default kernel width at M=16 == 3.0");
}

static void test_exp_kernel_monotone() {
    bool decreasing = true;
    double prev = exp_kernel(0.0, 1.5);
    for (double d = 0.25; d <= 6.0; d += 0.25) {
        const double cur = exp_kernel(d, 1.5);
        if (cur > prev) decreasing = false;
        prev = cur;
    }
    check(decreasing, "exp_kernel is monotonically decreasing in d");
}

// --- 4, 5, 6, 7: the weighted ridge solver ----------------------------------

static void test_fit_recovers_exact_linear_relationship() {
    // y = 3·z0 - 2·z1 exactly, so ridge with alpha=0 must reproduce it.
    std::vector<double> z0 = {1.0, 2.0, 3.0, -1.0, 0.5};
    std::vector<double> z1 = {0.0, 1.0, -1.0, 2.0, 3.0};
    std::vector<double> y, w;
    for (size_t i = 0; i < z0.size(); ++i) {
        y.push_back(3.0 * z0[i] - 2.0 * z1[i]);
        w.push_back(1.0 + 0.1 * static_cast<double>(i));
    }
    LocalSurrogate s = fit_local_surrogate({z0, z1}, y, w, 0.0);
    check_close(s.coefficients[0], 3.0, 1e-9, "exact linear: beta_0 == 3.0");
    check_close(s.coefficients[1], -2.0, 1e-9, "exact linear: beta_1 == -2.0");
    check_close(s.intercept, 0.0, 1e-9, "exact linear: intercept == 0");
    check_close(s.score, 1.0, 1e-9, "exact linear: weighted R² == 1.0");
}

static void test_fit_ridge_shrinkage_matches_closed_form() {
    // Two points, one feature — enough that the ridge shrinkage is visible and
    // the system is not rank-deficient.
    //
    // The fit CENTERS first (sklearn fit_intercept), so the closed form is the
    // CENTERED one, not the naive ratio:
    //     beta = (Σ w (x-x̄)(y-ȳ)) / (Σ w (x-x̄)² + alpha)
    // Working it out with w=(2,1), x=(4,3), y=(5,7), alpha=7:
    //     x̄ = 11/3, ȳ = 17/3
    //     x-x̄ = (1/3, -2/3),  y-ȳ = (-2/3, 4/3)
    //     numerator   = 2*(1/3)(-2/3) + 1*(-2/3)(4/3) = -4/9 - 8/9 = -4/3
    //     denominator = 2*(1/9) + 1*(4/9) + 7         = 2/9 + 4/9 + 7 = 23/3
    //     beta = (-4/3)/(23/3) = -4/23 ≈ -0.173913
    // The uncentered ratio 61/48 ≈ 1.2708 is the number a fit that skipped
    // centering would return, so this test also pins the centering.
    std::vector<double> z0 = {4.0, 3.0};
    std::vector<double> z1 = {0.0, 0.0};
    std::vector<double> y   = {5.0, 7.0};
    std::vector<double> w   = {2.0, 1.0};
    const double alpha = 7.0;
    LocalSurrogate s = fit_local_surrogate({z0, z1}, y, w, alpha);
    check_close(s.coefficients[0], -4.0 / 23.0, 1e-9,
                "ridge shrinkage == centered (Σw·xc·yc)/(Σw·xc²+alpha) — +alpha·I is real");
    check(std::abs(s.coefficients[0] - 61.0 / 48.0) > 1e-3,
          "fixture is non-degenerate: the uncentered answer differs");

    // alpha is monotone: a larger penalty must shrink |beta| toward zero.
    LocalSurrogate s0 = fit_local_surrogate({z0, z1}, y, w, 0.0);
    LocalSurrogate s_big = fit_local_surrogate({z0, z1}, y, w, 1000.0);
    check(std::abs(s.coefficients[0]) < std::abs(s0.coefficients[0]),
          "alpha>0 shrinks the coefficient relative to alpha=0");
    check(std::abs(s_big.coefficients[0]) < std::abs(s.coefficients[0]),
          "a larger alpha shrinks it further");
}

static void test_fit_intercept_uses_weighted_mean() {
    // y = 2 + 5·z exactly. The intercept is 2 ONLY if the implementation
    // centers by the SAMPLE-WEIGHTED mean. Centering by the unweighted mean
    // gives a different β₀ here, because the weights are deliberately skewed.
    std::vector<double> z0 = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> z1 = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> y(4), w(4);
    for (size_t i = 0; i < 4; ++i) { y[i] = 2.0 + 5.0 * z0[i]; w[i] = (i == 0) ? 50.0 : 1.0; }
    LocalSurrogate s = fit_local_surrogate({z0, z1}, y, w, 0.0);
    check_close(s.coefficients[0], 5.0, 1e-6, "skewed weights: slope still 5.0");
    check_close(s.intercept, 2.0, 1e-6, "intercept uses the WEIGHTED mean (== 2.0)");
    // And confirm the weighted and unweighted means genuinely differ here, so
    // this test would actually fail under an unweighted-centering mutation.
    const double wm = weighted_mean(y, w);
    double um = 0.0;
    for (double v : y) um += v;
    um /= 4.0;
    check(std::abs(wm - um) > 1e-3,
          "fixture is non-degenerate: weighted mean != unweighted mean");
}

static void test_fit_score_is_weighted_r2() {
    // Predictions are known by hand; SS_res and SS_tot use the same weights,
    // with SS_tot about the WEIGHTED mean.
    std::vector<double> z0 = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> z1 = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> y   = {1.0, 2.0, 4.0, 8.0};
    std::vector<double> w   = {1.0, 2.0, 3.0, 4.0};
    LocalSurrogate s = fit_local_surrogate({z0, z1}, y, w, 0.0);
    // Recompute R² independently: predictions from the fitted line.
    double ss_res = 0.0, ss_tot = 0.0;
    const double ybar = weighted_mean(y, w);
    for (size_t i = 0; i < y.size(); ++i) {
        const double pred = s.intercept + s.coefficients[0] * z0[i];
        ss_res += w[i] * (y[i] - pred) * (y[i] - pred);
        ss_tot += w[i] * (y[i] - ybar) * (y[i] - ybar);
    }
    check_close(s.score, 1.0 - ss_res / ss_tot, 1e-12,
                "score == 1 - SS_res/SS_tot with weights on both sums");
    check(s.score < 1.0, "fixture is genuinely imperfect (score < 1)");
}

static void test_fit_validation() {
    std::vector<double> z0 = {1.0, 2.0};
    std::vector<double> z1 = {1.0, 2.0};
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> w = {1.0, 1.0};

    bool threw = false;
    try { fit_local_surrogate({z0, z1}, y, w, -1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "fit_local_surrogate throws on negative alpha");

    threw = false;
    try { std::vector<double> w0 = {1.0}; fit_local_surrogate({z0, z1}, y, w0, 1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "fit_local_surrogate throws on mismatched weight length");

    threw = false;
    try { std::vector<double> yy = {1.0}; fit_local_surrogate({z0, z1}, yy, w, 1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "fit_local_surrogate throws on mismatched response length");
}

// --- 9, 10: the closed-form oracle (headline) -------------------------------

static void test_lime_closed_form_oracle() {
    // f(x) = Σ c_j x_j with NO bias, so
    //   β_j = c_j·σ_j,  intercept = Σ c_j μ_j,  local_pred = f(x).
    const std::vector<double> c  = {2.0, -1.5, 0.75, 3.0};
    const std::vector<double> mu = {0.5, -2.0, 1.0, 0.25};
    const std::vector<double> sd = {2.0, 0.5, 1.5, 4.0};
    Model m = make_linear_model(4, 1, {{2.0}, {-1.5}, {0.75}, {3.0}});

    // Place the instance ~1.4 standard deviations out from the mean, i.e. a
    // TYPICAL LIME query. This matters for conditioning: the earlier version of
    // this fixture sat 8.1 sigma out (z = [0.25, 8, -1.33, 0.06]) against a
    // kernel width of omega = sqrt(4)*0.75 = 1.5, so every sampled row landed
    // more than 5 widths away and every kernel weight underflowed to ~0. The
    // effective sample size collapsed to roughly 1. It still solved exactly at
    // alpha=0, but it was testing the near-rank-deficient path rather than the
    // method.
    const std::vector<double> zq = {0.5, -1.0, 0.8, 0.3};
    std::vector<double> xv(4);
    for (size_t j = 0; j < 4; ++j) xv[j] = mu[j] + sd[j] * zq[j];
    Tensor x = make_row(xv);

    LimeConfig cfg;
    cfg.mean = make_row(mu);
    cfg.std = make_row(sd);
    cfg.n_samples = 4000;
    cfg.seed = 7;
    // alpha = 0 is REQUIRED for exact recovery, and that is a property of the
    // method rather than a shortcut: the response is EXACTLY affine in the
    // interpretable coordinates for a linear black box, so with no penalty the
    // least-squares solution is exact for ANY neighborhood and ANY weighting —
    // there is no sampling noise to average out. With the reference's default
    // alpha = 1 the coefficients are deliberately shrunk toward zero (the ridge
    // penalty is real; test 5 pins it), so this oracle would fail for a reason
    // that has nothing to do with correctness.
    cfg.alpha = 0.0;

    LimeResult r = lime_explain(m, x, cfg, 0);

    for (size_t j = 0; j < 4; ++j) {
        check_close(r.coefficients[j], c[j] * sd[j], 1e-7,
                    "oracle: beta_" + std::to_string(j) + " == c_j * sigma_j");
    }
    double expect_intercept = 0.0;
    for (size_t j = 0; j < 4; ++j) expect_intercept += c[j] * mu[j];
    check_close(r.intercept, expect_intercept, 1e-7,
                "oracle: intercept == Σ c_j·μ_j (reconstructed after centering)");
    check_close(r.local_prediction, model_output(m, x, 0), 1e-9,
                "oracle: local surrogate prediction == f(x)");
    check(r.score > 0.999, "oracle: a perfectly linear model is fit almost exactly");
}

static void test_lime_local_prediction_matches_model() {
    Model m = make_linear_model(3, 2, {{2.0, 1.0}, {-1.0, 0.5}, {0.5, 2.0}});
    Tensor x = make_row({0.7, -1.3, 2.2});
    LimeConfig cfg;
    cfg.n_samples = 3000;
    cfg.seed = 3;
    // alpha = 0 so that "local prediction == f(x)" is an exact statement for a
    // linear black box rather than a statement about how much a ridge penalty
    // shrank the answer. The shrinkage itself is pinned in test 5.
    cfg.alpha = 0.0;
    LimeResult r1 = lime_explain(m, x, cfg, 0);
    LimeResult r2 = lime_explain(m, x, cfg, 1);
    check_close(r1.local_prediction, model_output(m, x, 0), 1e-9,
                "target 0: local prediction == f_0(x)");
    check_close(r2.local_prediction, model_output(m, x, 1), 1e-9,
                "target 1: local prediction == f_1(x)");
    check_close(r2.base_prediction, model_output(m, x, 1), 1e-12,
                "base prediction is the model output at the instance");

    // And a separate check that the DEFAULT config (alpha = 1, the reference's
    // value) is deliberately NOT exact — it shrinks. Without this, "alpha is
    // wired up" would go untested by the exact assertions above.
    LimeConfig dflt;
    dflt.n_samples = 3000;
    dflt.seed = 3;
    LimeResult shrunk = lime_explain(m, x, dflt, 0);
    check(std::abs(shrunk.local_prediction - model_output(m, x, 0)) > 1e-12,
          "the default alpha=1 genuinely applies a ridge penalty");
}

// --- 11, 12: ranking and selection -------------------------------------------

static void test_ranking_uses_absolute_coefficient() {
    // Make one feature clearly dominant and NEGATIVE. A rank-by-β (not |β|)
    // implementation would put it last instead of first.
    Model m = make_linear_model(3, 1, {{0.1}, {0.1}, {-5.0}});
    Tensor x = make_row({0.5, 0.5, 1.0});
    LimeConfig cfg;
    cfg.n_samples = 3000;
    cfg.seed = 5;
    cfg.alpha = 0.0;   // exact recovery, see the oracle test's note
    LimeResult r = lime_explain(m, x, cfg, 0);
    check(r.ranking[0] == 2,
          "the largest-magnitude coefficient ranks first even when negative");
    check_close(r.coefficients[2], -5.0 * r.config.std[0][2], 1e-6,
                "dominant negative feature recovered exactly");

    // And the returned importance is |β|, hence non-negative.
    bool all_nonneg = true;
    for (double v : r.importances) if (v < 0.0) all_nonneg = false;
    check(all_nonneg, "importances are |β| and therefore non-negative");
}

static void test_num_features_restricts_the_surrogate() {
    Model m = make_linear_model(4, 1, {{5.0}, {0.0}, {0.0}, {0.0}});
    Tensor x = make_row({1.0, 1.0, 1.0, 1.0});
    LimeConfig cfg;
    cfg.n_samples = 3000;
    cfg.seed = 9;
    cfg.num_features = 1;
    cfg.alpha = 0.0;   // exact recovery

    LimeResult r = lime_explain(m, x, cfg, 0);
    check(r.coefficients.size() == 1,
          "num_features=1 returns a single coefficient");
    check(r.ranking.size() == 1, "ranking is also truncated to num_features");
    check(r.ranking[0] == 0, "the retained feature is feature 0");
    // Read the scale back from the ECHOED config, not from the input config:
    // this fixture never sets `std`, so cfg.std is an empty Tensor whose
    // subscript has no backing buffer. lime_explain echoes the resolved
    // standardization (defaults 0/1 here), and reading it also proves the echo
    // is populated rather than a copy of the empty input.
    check_close(r.coefficients[0], 5.0 * r.config.std[0][0], 1e-6,
                "the single retained feature recovers 5.0 * sigma_0");
    check_close(r.config.std[0][0], 1.0, 1e-12,
                "config echo resolves the default standard deviation to 1.0");

    cfg.num_features = 0;  // 0 == "all features", the reference's convention
    LimeResult full = lime_explain(m, x, cfg, 0);
    check(full.coefficients.size() == 4, "num_features=0 means all features");
}

// --- 13, 14, 15, 16: determinism, validation, hygiene, metadata -------------

static void test_lime_determinism() {
    Model m = make_linear_model(4, 1, {{1.0}, {-2.0}, {0.5}, {3.0}});
    Tensor x = Tensor::random(1, 4, 1.0);
    LimeConfig cfg;
    cfg.n_samples = 2000;
    cfg.seed = 11;

    LimeResult a = lime_explain(m, x, cfg, 0);
    LimeResult b = lime_explain(m, x, cfg, 0);
    bool identical = true;
    for (size_t j = 0; j < a.coefficients.size(); ++j)
        if (a.coefficients[j] != b.coefficients[j]) identical = false;
    check(identical, "lime_explain is bit-exact deterministic for a fixed seed");

    cfg.seed = 12;
    LimeResult c = lime_explain(m, x, cfg, 0);
    double diff = 0.0;
    for (size_t j = 0; j < a.coefficients.size(); ++j)
        diff += std::abs(a.coefficients[j] - c.coefficients[j]);
    check(diff > 1e-9, "a different seed produces a different neighborhood");
}

static void test_lime_validation() {
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({1.0, 1.0});

    bool threw = false;
    LimeConfig zero; zero.n_samples = 0;
    try { lime_explain(m, x, zero, 0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "lime_explain throws on n_samples == 0");

    threw = false;
    LimeConfig bad_mean; bad_mean.mean = make_row({1.0});  // 1 col, model wants 2
    try { lime_explain(m, x, bad_mean, 0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "lime_explain throws on a mean whose width != n_features");

    threw = false;
    LimeConfig zero_sd; zero_sd.std = make_row({1.0, 0.0});
    try { lime_explain(m, x, zero_sd, 0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "lime_explain throws on a zero standard deviation");

    threw = false;
    try { lime_explain(m, x, LimeConfig(), 9); } catch (const std::out_of_range&) { threw = true; }
    check(threw, "lime_explain throws std::out_of_range for a bad target");
}

static void test_lime_leaves_no_gradient_residue() {
    // Deliberately seed NONZERO parameter gradients first. LIME never calls
    // backward(), so a suite that merely asserted "still zero" after the call
    // would pass vacuously — the values would have been zero to begin with.
    // Seeding them proves the zeroing actually happens.
    Model m = make_linear_model(3, 1, {{1.0}, {1.0}, {1.0}});
    double seeded = 0.0;
    for (auto& lp : m.layers) {
        Dense* dp = dynamic_cast<Dense*>(lp.get());
        if (dp) { dp->grad_weights.fill(7.0); dp->grad_bias.fill(5.0); seeded = 7.0; }
    }
    check(seeded != 0.0, "fixture: gradients really were seeded nonzero");

    Tensor x = make_row({0.1, 0.2, 0.3});
    LimeConfig cfg;
    cfg.n_samples = 500;
    cfg.seed = 1;
    lime_explain(m, x, cfg, 0);

    bool all_zero = true;
    for (auto& lp : m.layers)
        for (auto* g : lp->gradients())
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    if (g->data[i * g->cols + j] != 0.0) all_zero = false;
    check(all_zero, "lime_explain zeroes pre-existing parameter gradients");
}

static void test_lime_result_metadata() {
    Model m = make_linear_model(3, 1, {{1.0}, {2.0}, {3.0}});
    Tensor x = make_row({0.5, 0.5, 0.5});
    LimeConfig cfg;
    cfg.n_samples = 1500;
    cfg.seed = 21;
    cfg.num_features = 2;
    LimeResult r = lime_explain(m, x, cfg, 0);

    check(r.n_samples == 1500, "metadata: n_samples echoes the config");
    check(r.n_features == 3, "metadata: n_features is the input width");
    check(r.target == 0, "metadata: target is recorded");
    check(r.ranking.size() == 2, "metadata: ranking honors num_features");
    check(std::isfinite(r.score) && std::isfinite(r.intercept) &&
          std::isfinite(r.local_prediction),
          "metadata: score/intercept/prediction are all finite");
    // Config is echoed back so the caller can see the standardization used.
    check(r.config.std.cols == 3, "metadata: config.std is echoed with full width");
}

// --- 17: fixtures that exercise the solver's pivoting and standardization ---

static void test_pivoting_is_exercised() {
    // The Gram matrix here forces a partial-pivoting ROW SWAP at column 0:
    //     A = [ 2.00  -5.00 ]
    //         [-5.00  12.83 ]
    // so |A[1][0]| = 5 > |A[0][0]| = 2 and the elimination must swap rows.
    //
    // Why this fixture exists: the mutation harness showed that swapping only
    // the pivot ELEMENT instead of the whole row (a real bug this session
    // found) is NOT caught by any other test in the suite. Every other
    // fixture happens to produce a Gram matrix whose leading pivot is already
    // maximal, so the swap branch never runs and the bug is invisible. A
    // mutation that is invisible everywhere else is precisely the case that
    // needs a targeted fixture.
    //
    // Exact solution (verified independently, alpha = 0):
    //     beta  = (16, 7)
    //     b0    = -31
    // which reproduces y = [-31 + 16*2 + 7*0, -31 + 16*0 + 7*5, -31 + 16*1 + 7*3]
    //              = [1, 4, 6] = y exactly, hence score == 1.
    std::vector<double> z0 = {2.0, 0.0, 1.0};
    std::vector<double> z1 = {0.0, 5.0, 3.0};
    std::vector<double> y  = {1.0, 4.0, 6.0};
    std::vector<double> w  = {1.0, 1.0, 4.0};

    // Precondition: this fixture really does force a swap. Asserted so the
    // test cannot silently degrade into the no-swap case if the weighting is
    // ever changed.
    const double sw = w[0] + w[1] + w[2];
    const double xm0 = (w[0]*z0[0] + w[1]*z0[1] + w[2]*z0[2]) / sw;
    const double xm1 = (w[0]*z1[0] + w[1]*z1[1] + w[2]*z1[2]) / sw;
    const double a00 = w[0]*(z0[0]-xm0)*(z0[0]-xm0) + w[1]*(z0[1]-xm0)*(z0[1]-xm0)
                     + w[2]*(z0[2]-xm0)*(z0[2]-xm0);
    const double a10 = w[0]*(z0[0]-xm0)*(z1[0]-xm1) + w[1]*(z0[1]-xm0)*(z1[1]-xm1)
                     + w[2]*(z0[2]-xm0)*(z1[2]-xm1);
    check(std::abs(a10) > std::abs(a00),
          "fixture forces a pivoting row swap (|A[1][0]| > |A[0][0]|)");

    LocalSurrogate s = fit_local_surrogate({z0, z1}, y, w, 0.0);
    check_close(s.coefficients[0], 16.0, 1e-9, "pivoting fixture: beta_0 == 16.0");
    check_close(s.coefficients[1], 7.0, 1e-9, "pivoting fixture: beta_1 == 7.0");
    check_close(s.intercept, -31.0, 1e-9, "pivoting fixture: intercept == -31.0");
    check_close(s.score, 1.0, 1e-9, "pivoting fixture: exact fit, score == 1.0");
}

static void test_distance_is_standardized() {
    // Two features with a 100x scale gap. The instance sits at z = [1, 1] —
    // one standardized unit in each direction — but in RAW coordinates the
    // first feature is 100 units away and the second only 1. A surrogate that
    // measured distance on raw coordinates would put essentially all the
    // kernel weight on feature 0 and collapse.
    //
    // MEASURED (this is why the test uses the DEFAULT alpha): for a linear
    // black box at alpha = 0 the exact solution beta_j = c_j·sigma_j is
    // INDEPENDENT of the weights, so the distance metric is mathematically
    // unobservable there — measured: raw-distance mutation gives exactly the
    // same [100, 1], local_pred 101, score 1. The distance only enters through
    // the ridge penalty. At the reference default alpha = 1 the mutation is
    // glaring:
    //     standardized distances -> beta = [99.859, 0.9985], score = 0.999998
    //     raw distances          -> beta = [0.154,  0.894 ], score = 0.287
    // So the discriminating assertion is the SHRUNKEN one, and the exact
    // recovery is asserted only as a secondary invariant.
    Model m = make_linear_model(2, 1, {{1.0}, {1.0}});
    Tensor x = make_row({100.0, 1.0});

    LimeConfig base;
    base.n_samples = 4000;
    base.seed = 17;
    base.mean = make_row({0.0, 0.0});
    base.std = make_row({100.0, 1.0});

    // The discriminating case: the reference default alpha = 1.
    LimeConfig shrunk = base;                 // alpha defaults to 1
    LimeResult r = lime_explain(m, x, shrunk, 0);
    check(r.score > 0.999,
          "standardized distance: local fit stays near-exact (score > 0.999)");
    check_close(r.coefficients[0], 99.86, 1e-2,
                "standardized distance: beta_0 ~ 99.86 (ridge-shrunk from 100)");
    check_close(r.coefficients[1], 0.998, 1e-3,
                "standardized distance: beta_1 ~ 0.998 (ridge-shrunk from 1)");

    // Secondary invariant: at alpha = 0 the exact recovery holds, and it holds
    // regardless of the distance metric, so this is an assertion about the
    // closed-form oracle rather than about locality.
    LimeConfig exact_cfg = base;
    exact_cfg.alpha = 0.0;
    LimeResult e = lime_explain(m, x, exact_cfg, 0);
    check_close(e.coefficients[0], 100.0, 1e-6,
                "alpha=0: beta_0 == c_0 * sigma_0 exactly");
    check_close(e.coefficients[1], 1.0, 1e-6,
                "alpha=0: beta_1 == c_1 * sigma_1 exactly");
    check_close(e.local_prediction, 101.0, 1e-9,
                "alpha=0: local prediction == f(x) == 101");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_lime\n";
    std::cout << "============================================================\n";

    test_exp_kernel_values();
    test_exp_kernel_default_width();
    test_exp_kernel_monotone();
    test_fit_recovers_exact_linear_relationship();
    test_fit_ridge_shrinkage_matches_closed_form();
    test_fit_intercept_uses_weighted_mean();
    test_fit_score_is_weighted_r2();
    test_fit_validation();
    test_lime_closed_form_oracle();
    test_lime_local_prediction_matches_model();
    test_ranking_uses_absolute_coefficient();
    test_num_features_restricts_the_surrogate();
    test_lime_determinism();
    test_lime_validation();
    test_lime_leaves_no_gradient_residue();
    test_lime_result_metadata();
    test_pivoting_is_exercised();
    test_distance_is_standardized();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}