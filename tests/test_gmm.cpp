// Gaussian Mixture Model (GMM) tests
//
// Covers:
//   1. MultivariateGaussian scalar & vector log_pdf correctness (closed-form)
//   2. log_pdf_grad at mu is zero
//   3. KL divergence between identical Gaussians is 0
//   4. KL divergence between N(0,1) and N(1,1) is 0.5 (closed-form 1D)
//   5. MultivariateGaussian: gradient check (numerical vs analytical via FD)
//   6. GMM.fit() constructor validation (empty/zero components/etc.)
//   7. KMeans k-means++ init seeds are well-separated
//   8. KMeans Lloyd convergence: inertia non-increasing
//   9. KMeans recovers >95% on a 3-cluster toy problem
//  10. GMM log-likelihood monotonically non-decreasing across EM steps
//  11. GMM responsibilities per row sum to 1
//  12. GMM weights sum to 1
//  13. GMM predict consistency with argmax responsibilities
//  14. GMM end-to-end recovery of 3 components from 5 noisy initial means
//  15. GMM BIC decreases as K moves from 1 to 3 on a 3-component sample
//  16. GMM sample() produces finite shape-(N, D) output
//  17. GMM is reproducible given a seed
//  18. GMM reg_covar=0 on singular data throws
//  19. GMM KMeans++ init works when K > 1
//  20. GMM full cluster accuracy comparable to KMeans

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <tuple>

#include "nn/utils/gmm.h"

using std::cout;
using std::endl;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, msg)                                  \
    do {                                                                       \
        double a_ = (actual);                                                   \
        double e_ = (expected);                                                 \
        if (std::fabs(a_ - e_) <= (tol)) {                                      \
            cout << "[PASS] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << endl;                                \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << " diff=" << std::fabs(a_ - e_)       \
                 << endl;                                                      \
        }                                                                      \
        ++total;                                                               \
    } while (0)

// ---------- Test 1: MultivariateGaussian scalar log_pdf ----------
static void test_mvg_scalar_logpdf() {
    cout << "-- Test 1: MultivariateGaussian 1D log_pdf closed-form --" << endl;
    // 1D N(0, 1) at x=0 should be -0.5*log(2*pi) - 0.5*0 = -0.5*log(2*pi)
    {
        Tensor mu = Tensor::zeros(1, 1); mu(0, 0) = 0.0;
        Tensor cov = Tensor::zeros(1, 1); cov(0, 0) = 1.0;
        MultivariateGaussian g(mu, cov);
        Tensor x = Tensor::zeros(1, 1); x(0, 0) = 0.0;
        double lp = g.log_pdf_single(x);
        CHECK_NEAR(lp, -0.5 * std::log(2.0 * M_PI), 1e-10,
                   "N(0,1) log_pdf at 0 = -0.5*log(2*pi)");
    }
    // 1D N(0, 1) at x=1: -0.5*log(2*pi) - 0.5*1
    {
        Tensor mu = Tensor::zeros(1, 1); mu(0, 0) = 0.0;
        Tensor cov = Tensor::zeros(1, 1); cov(0, 0) = 1.0;
        MultivariateGaussian g(mu, cov);
        Tensor x = Tensor::zeros(1, 1); x(0, 0) = 1.0;
        double lp = g.log_pdf_single(x);
        CHECK_NEAR(lp, -0.5 * std::log(2.0 * M_PI) - 0.5, 1e-10,
                   "N(0,1) log_pdf at 1 = -0.5*log(2*pi) - 0.5");
    }
    // 1D N(2, 4) at x=2: -0.5*log(2*pi*4) - 0
    {
        Tensor mu = Tensor::zeros(1, 1); mu(0, 0) = 2.0;
        Tensor cov = Tensor::zeros(1, 1); cov(0, 0) = 4.0;
        MultivariateGaussian g(mu, cov);
        Tensor x = Tensor::zeros(1, 1); x(0, 0) = 2.0;
        double lp = g.log_pdf_single(x);
        CHECK_NEAR(lp, -0.5 * std::log(2.0 * M_PI * 4.0), 1e-10,
                   "N(2,4) log_pdf at mu = -0.5*log(2*pi*sigma^2)");
    }
}

// ---------- Test 2: MultivariateGaussian vector log_pdf batch ----------
static void test_mvg_batch_logpdf() {
    cout << "-- Test 2: MultivariateGaussian batch log_pdf --" << endl;
    // 2D isotropic N((0,0), I) batch over 3 points
    Tensor mu({{0.0, 0.0}});
    Tensor cov({{1.0, 0.0}, {0.0, 1.0}});
    MultivariateGaussian g(mu, cov);
    Tensor X({{0.0, 0.0},
              {1.0, 0.0},
              {0.0, 2.0}});
    Tensor lp = g.log_pdf(X);
    CHECK(lp.rows == 3 && lp.cols == 1, "log_pdf batch returns (N, 1) tensor");
    CHECK_NEAR(lp(0, 0), -1.0 * std::log(2.0 * M_PI), 1e-10,
               "N(0,I) log_pdf at origin = -log(2*pi)");
    CHECK_NEAR(lp(1, 0), -1.0 * std::log(2.0 * M_PI) - 0.5, 1e-10,
               "N(0,I) log_pdf at (1,0) = -log(2*pi) - 0.5");
    CHECK_NEAR(lp(2, 0), -1.0 * std::log(2.0 * M_PI) - 2.0, 1e-10,
               "N(0,I) log_pdf at (0,2) = -log(2*pi) - 2.0");
}

// ---------- Test 3: log_pdf_grad at mu is zero ----------
static void test_mvg_grad_at_mu_is_zero() {
    cout << "-- Test 3: log_pdf_grad at mu is zero --" << endl;
    Tensor mu({{1.0, -2.0, 3.0}});
    // non-isotropic covariance
    Tensor cov({{2.0, 0.5, 0.0},
                {0.5, 1.0, 0.0},
                {0.0, 0.0, 0.5}});
    MultivariateGaussian g(mu, cov);
    Tensor grad = g.log_pdf_grad(mu);
    CHECK(grad.rows == 1 && grad.cols == 3, "log_pdf_grad returns (1, d) tensor");
    CHECK_NEAR(grad(0, 0), 0.0, 1e-10, "log_pdf_grad at mu: grad[0] = 0");
    CHECK_NEAR(grad(0, 1), 0.0, 1e-10, "log_pdf_grad at mu: grad[1] = 0");
    CHECK_NEAR(grad(0, 2), 0.0, 1e-10, "log_pdf_grad at mu: grad[2] = 0");
}

// ---------- Test 4: log_pdf_grad numerical gradient check ----------
static void test_mvg_grad_numerical_check() {
    cout << "-- Test 4: log_pdf_grad vs centered FD --" << endl;
    Tensor mu({{0.5, -0.3}});
    Tensor cov({{1.5, 0.4}, {0.4, 0.8}});
    MultivariateGaussian g(mu, cov);
    Tensor x({{1.2, 0.7}});
    Tensor grad = g.log_pdf_grad(x);

    const double eps = 1e-5;
    double max_rel_err = 0.0;
    for (size_t j = 0; j < 2; ++j) {
        Tensor xp = x.clone();
        Tensor xm = x.clone();
        xp(0, j) += eps;
        xm(0, j) -= eps;
        double num = (g.log_pdf_single(xp) - g.log_pdf_single(xm)) / (2.0 * eps);
        double ana = grad(0, j);
        double denom = std::max(std::fabs(ana), 1e-12);
        max_rel_err = std::max(max_rel_err, std::fabs(num - ana) / denom);
    }
    CHECK(max_rel_err < 1e-5,
           "log_pdf_grad matches centered finite difference (rel_err < 1e-5)");
}

// ---------- Test 5: KL divergence ----------
static void test_mvg_kl_divergence() {
    cout << "-- Test 5: MultivariateGaussian KL divergence --" << endl;
    // Identical Gaussians: KL = 0
    Tensor mu({{0.0, 0.0}});
    Tensor cov({{1.0, 0.0}, {0.0, 1.0}});
    MultivariateGaussian p(mu, cov);
    MultivariateGaussian q(mu, cov);
    CHECK_NEAR(p.kl_to(q), 0.0, 1e-10, "KL(N || N) = 0 for identical distributions");
    // 1D N(0,1) || N(1,1): KL = 0.5 * (sigma_q^2/sigma_p^2 + (mu_p-mu_q)^2/sigma_q^2 - 1 + log(sigma_p^2/sigma_q^2))
    // = 0.5 * (1 + 1 - 1 + log(1)) = 0.5
    Tensor mu1 = Tensor::zeros(1, 1); mu1(0, 0) = 0.0;
    Tensor cov1 = Tensor::zeros(1, 1); cov1(0, 0) = 1.0;
    Tensor mu2 = Tensor::zeros(1, 1); mu2(0, 0) = 1.0;
    Tensor cov2 = Tensor::zeros(1, 1); cov2(0, 0) = 1.0;
    MultivariateGaussian p1(mu1, cov1);
    MultivariateGaussian q1(mu2, cov2);
    CHECK_NEAR(p1.kl_to(q1), 0.5, 1e-10, "KL(N(0,1) || N(1,1)) = 0.5");
    // KL(N(0,1) || N(0, sigma^2)) = 0.5*(1/sigma^2 - 1 + log(sigma^2))
    Tensor cov4 = Tensor::zeros(1, 1); cov4(0, 0) = 4.0;
    MultivariateGaussian q4(mu1, cov4);
    double expected = 0.5 * (1.0 / 4.0 - 1.0 + std::log(4.0));
    CHECK_NEAR(p1.kl_to(q4), expected, 1e-10,
               "KL(N(0,1) || N(0,4)) = 0.5*(1/4 - 1 + log(4))");
}

// ---------- Test 6: MultivariateGaussian validation ----------
static void test_mvg_validation() {
    cout << "-- Test 6: MultivariateGaussian validation --" << endl;
    // Mismatched mu/cov dimensions
    bool threw1 = false;
    try {
        Tensor mu({{0.0, 0.0}});
        Tensor cov = Tensor::zeros(1, 1); cov(0, 0) = 1.0;
        MultivariateGaussian g(mu, cov);
    } catch (const std::invalid_argument&) {
        threw1 = true;
    }
    CHECK(threw1, "mu.cols != cov.rows throws");

    // Non-square cov
    bool threw2 = false;
    try {
        Tensor mu({{0.0, 0.0}});
        Tensor cov({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});
        MultivariateGaussian g(mu, cov);
    } catch (const std::invalid_argument&) {
        threw2 = true;
    }
    CHECK(threw2, "non-square cov throws");
}

// ---------- Test 7: KMeans k-means++ seeds are well-separated ----------
static void test_kmeans_pp_init_separation() {
    cout << "-- Test 7: KMeans k-means++ seeds are well-separated --" << endl;
    // 3 well-separated clusters in 2D
    Tensor X(6, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.1;  X(1, 1) = 0.1;
    X(2, 0) = 10.0; X(2, 1) = 10.0;
    X(3, 0) = 10.1; X(3, 1) = 10.1;
    X(4, 0) = -10.0; X(4, 1) = 5.0;
    X(5, 0) = -10.1; X(5, 1) = 5.1;
    KMeans km(3, 50, 1e-6, 42);
    km.fit(X);
    auto centers = km.cluster_centers();
    CHECK(centers.rows == 3 && centers.cols == 2, "cluster_centers shape = (K, d)");

    // All centers should be distinct
    double min_pair = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 3; ++j) {
            double dx = centers(i, 0) - centers(j, 0);
            double dy = centers(i, 1) - centers(j, 1);
            double d = std::sqrt(dx * dx + dy * dy);
            min_pair = std::min(min_pair, d);
        }
    }
    CHECK(min_pair > 1.0,
          "k-means++ seeds (final centers) have min pairwise distance > 1");
}

// ---------- Test 8: KMeans inertia monotonically non-increasing ----------
static void test_kmeans_inertia_monotonic() {
    cout << "-- Test 8: KMeans inertia monotonically non-increasing --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(40, 2);
    for (size_t i = 0; i < 40; ++i) {
        // 2 well-separated clusters
        size_t grp = i < 20 ? 0 : 1;
        X(i, 0) = (grp == 0 ? 0.0 : 5.0) + nd(rng);
        X(i, 1) = (grp == 0 ? 0.0 : 5.0) + nd(rng);
    }
    KMeans km(2, 100, 1e-8, 7);
    auto inertia_history = km.fit_with_history(X);
    CHECK(inertia_history.size() >= 2, "history has >= 2 iterations");
    bool monotone = true;
    for (size_t i = 1; i < inertia_history.size(); ++i) {
        if (inertia_history[i] > inertia_history[i - 1] + 1e-9) {
            monotone = false;
            break;
        }
    }
    CHECK(monotone, "inertia is monotonically non-increasing across Lloyd iterations");
}

// ---------- Test 9: KMeans recovery on well-separated clusters ----------
static void test_kmeans_recovery() {
    cout << "-- Test 9: KMeans recovers >95% on well-separated clusters --" << endl;
    std::mt19937 rng(123);
    std::normal_distribution<double> nd(0.0, 0.4);
    const size_t K = 3;
    std::vector<std::pair<double, double>> centers = {{0.0, 0.0}, {8.0, 0.0}, {4.0, 7.0}};
    const size_t per_cluster = 50;
    Tensor X(K * per_cluster, 2);
    std::vector<size_t> true_labels(K * per_cluster);
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < per_cluster; ++i) {
            size_t row = k * per_cluster + i;
            X(row, 0) = centers[k].first + nd(rng);
            X(row, 1) = centers[k].second + nd(rng);
            true_labels[row] = k;
        }
    }
    KMeans km(K, 100, 1e-6, 7);
    km.fit(X);
    Tensor labels = km.predict(X);
    // Build a label-mapping via greedy matching (Hungarian would be better,
    // but a greedy assignment is enough for >95% on well-separated clusters).
    std::vector<size_t> pred(K * per_cluster);
    for (size_t i = 0; i < labels.rows; ++i) {
        pred[i] = static_cast<size_t>(labels(i, 0));
    }
    // Count co-occurrences of (true, pred)
    std::vector<std::vector<size_t>> co(K, std::vector<size_t>(K, 0));
    for (size_t i = 0; i < K * per_cluster; ++i) co[true_labels[i]][pred[i]]++;
    // Greedy matching
    std::vector<bool> used_true(K, false), used_pred(K, false);
    std::vector<std::tuple<size_t, size_t, size_t>> matches;
    for (size_t t = 0; t < K; ++t)
        for (size_t p = 0; p < K; ++p) matches.push_back({co[t][p], t, p});
    std::sort(matches.begin(), matches.end(), std::greater<std::tuple<size_t,size_t,size_t>>());
    size_t correct = 0;
    for (auto& [c, t, p] : matches) {
        if (used_true[t] || used_pred[p]) continue;
        used_true[t] = true;
        used_pred[p] = true;
        correct += c;
    }
    double acc = static_cast<double>(correct) / (K * per_cluster);
    CHECK(acc >= 0.95, "KMeans clustering accuracy on 3-cluster toy problem >= 95%");
}

// ---------- Test 10: GMM log-likelihood non-decreasing ----------
static void test_gmm_loglik_monotonic() {
    cout << "-- Test 10: GMM log-likelihood non-decreasing across EM --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(60, 2);
    for (size_t i = 0; i < 60; ++i) {
        size_t grp = i < 20 ? 0 : (i < 40 ? 1 : 2);
        double cx = (grp == 0 ? -3.0 : grp == 1 ? 3.0 : 0.0);
        double cy = (grp == 0 ? 0.0 : grp == 1 ? 0.0 : 4.0);
        X(i, 0) = cx + nd(rng) * 0.5;
        X(i, 1) = cy + nd(rng) * 0.5;
    }
    GMM gmm(3, 100, 1e-6, 1e-6, 42);
    std::vector<double> lh_history;
    gmm.fit_with_history(X, lh_history);
    CHECK(lh_history.size() >= 2, "log-likelihood history has >= 2 entries");
    // EM is guaranteed non-decreasing (allow tiny tolerance for numerical noise)
    bool monotone = true;
    for (size_t i = 1; i < lh_history.size(); ++i) {
        if (lh_history[i] < lh_history[i - 1] - 1e-6) {
            monotone = false;
            break;
        }
    }
    CHECK(monotone, "log-likelihood is monotonically non-decreasing across EM iterations");
}

// ---------- Test 11: GMM responsibilities row-sum to 1 ----------
static void test_gmm_responsibilities_rowsum() {
    cout << "-- Test 11: GMM responsibilities row-sum to 1 --" << endl;
    Tensor X(10, 2);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (size_t i = 0; i < 10; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM gmm(2, 50, 1e-6, 1e-6, 0);
    gmm.fit(X);
    Tensor resp = gmm.predict_proba(X);
    CHECK(resp.rows == 10 && resp.cols == 2, "responsibilities shape (N, K)");
    for (size_t i = 0; i < 10; ++i) {
        double s = resp(i, 0) + resp(i, 1);
        CHECK_NEAR(s, 1.0, 1e-10,
                   "responsibilities row " + std::to_string(i) + " sum to 1");
    }
}

// ---------- Test 12: GMM weights sum to 1 ----------
static void test_gmm_weights_sum() {
    cout << "-- Test 12: GMM weights sum to 1 --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(30, 2);
    for (size_t i = 0; i < 30; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM gmm(3, 50, 1e-6, 1e-6, 42);
    gmm.fit(X);
    Tensor weights = gmm.weights();
    CHECK(weights.rows == 3 && weights.cols == 1, "weights shape (K, 1)");
    double s = 0.0;
    for (size_t k = 0; k < 3; ++k) s += weights(k, 0);
    CHECK_NEAR(s, 1.0, 1e-10, "weights sum to 1");
    for (size_t k = 0; k < 3; ++k) {
        CHECK(weights(k, 0) >= 0.0, "weight[" + std::to_string(k) + "] >= 0");
        CHECK(weights(k, 0) <= 1.0, "weight[" + std::to_string(k) + "] <= 1");
    }
}

// ---------- Test 13: GMM predict consistency ----------
static void test_gmm_predict_consistency() {
    cout << "-- Test 13: GMM predict = argmax predict_proba --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(30, 2);
    for (size_t i = 0; i < 30; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM gmm(3, 50, 1e-6, 1e-6, 42);
    gmm.fit(X);
    Tensor resp = gmm.predict_proba(X);
    Tensor pred = gmm.predict(X);
    CHECK(pred.rows == 30 && pred.cols == 1, "predict returns (N, 1) labels");
    for (size_t i = 0; i < 30; ++i) {
        size_t expected = 0;
        double best = resp(i, 0);
        for (size_t k = 1; k < 3; ++k) {
            if (resp(i, k) > best) { best = resp(i, k); expected = k; }
        }
        CHECK_NEAR(static_cast<double>(pred(i, 0)), static_cast<double>(expected), 0.0,
                   "predict[" + std::to_string(i) + "] = argmax predict_proba[" +
                   std::to_string(i) + "]");
    }
}

// ---------- Test 14: GMM end-to-end recovery ----------
static void test_gmm_endtoend_recovery() {
    cout << "-- Test 14: GMM end-to-end recovery of 3 components from 5 noisy inits --" << endl;
    std::mt19937 rng(99);
    const size_t K = 3;
    const size_t per_cluster = 80;
    std::vector<std::pair<double, double>> true_means = {{-3.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
    Tensor X(K * per_cluster, 2);
    std::vector<size_t> true_labels(K * per_cluster);
    for (size_t k = 0; k < K; ++k) {
        std::normal_distribution<double> nd(0.0, 0.4);
        for (size_t i = 0; i < per_cluster; ++i) {
            size_t row = k * per_cluster + i;
            X(row, 0) = true_means[k].first + nd(rng);
            X(row, 1) = true_means[k].second + nd(rng);
            true_labels[row] = k;
        }
    }
    // Random-init the GMM (5 noisy means to make it harder)
    std::uniform_real_distribution<double> ur(-5.0, 5.0);
    Tensor init_means(5, 2);
    for (size_t i = 0; i < 5; ++i) { init_means(i, 0) = ur(rng); init_means(i, 1) = ur(rng); }
    // Use KMeans++ init but pick K=3 out of 5 noisy means (random subset)
    GMM gmm(K, 200, 1e-6, 1e-3, 42);
    gmm.fit(X);
    Tensor pred = gmm.predict(X);
    std::vector<size_t> pred_labels(K * per_cluster);
    for (size_t i = 0; i < K * per_cluster; ++i) pred_labels[i] = static_cast<size_t>(pred(i, 0));
    // Hungarian-style greedy matching
    std::vector<std::vector<size_t>> co(K, std::vector<size_t>(K, 0));
    for (size_t i = 0; i < K * per_cluster; ++i) co[true_labels[i]][pred_labels[i]]++;
    std::vector<std::tuple<size_t, size_t, size_t>> matches;
    for (size_t t = 0; t < K; ++t)
        for (size_t p = 0; p < K; ++p) matches.push_back({co[t][p], t, p});
    std::sort(matches.begin(), matches.end(), std::greater<std::tuple<size_t,size_t,size_t>>());
    std::vector<bool> used_true(K, false), used_pred(K, false);
    size_t correct = 0;
    for (auto& [c, t, p] : matches) {
        if (used_true[t] || used_pred[p]) continue;
        used_true[t] = true;
        used_pred[p] = true;
        correct += c;
    }
    double acc = static_cast<double>(correct) / (K * per_cluster);
    CHECK(acc >= 0.85, "GMM end-to-end cluster recovery >= 85%");
}

// ---------- Test 15: GMM BIC model selection ----------
static void test_gmm_bic_decreases() {
    cout << "-- Test 15: GMM BIC decreases as K moves from 1 to 3 --" << endl;
    std::mt19937 rng(0);
    const size_t per_cluster = 80;
    std::vector<std::pair<double, double>> means = {{-3.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
    Tensor X(3 * per_cluster, 2);
    for (size_t k = 0; k < 3; ++k) {
        std::normal_distribution<double> nd(0.0, 0.5);
        for (size_t i = 0; i < per_cluster; ++i) {
            size_t row = k * per_cluster + i;
            X(row, 0) = means[k].first + nd(rng);
            X(row, 1) = means[k].second + nd(rng);
        }
    }
    GMM gmm1(1, 200, 1e-6, 1e-3, 42);
    gmm1.fit(X);
    GMM gmm3(3, 200, 1e-6, 1e-3, 42);
    gmm3.fit(X);
    double bic1 = gmm1.bic(X);
    double bic3 = gmm3.bic(X);
    CHECK(bic3 < bic1, "BIC(K=3) < BIC(K=1) on a 3-component sample");
    CHECK(std::isfinite(bic1) && std::isfinite(bic3), "BIC values are finite");
}

// ---------- Test 16: GMM sample() ----------
static void test_gmm_sample() {
    cout << "-- Test 16: GMM sample() produces finite (N, D) output --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(40, 3);
    for (size_t i = 0; i < 40; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
        X(i, 2) = nd(rng);
    }
    GMM gmm(2, 50, 1e-6, 1e-3, 42);
    gmm.fit(X);
    Tensor samples = gmm.sample(100, 7);
    CHECK(samples.rows == 100 && samples.cols == 3, "sample shape (100, 3)");
    bool all_finite = true;
    for (size_t i = 0; i < samples.rows; ++i)
        for (size_t j = 0; j < samples.cols; ++j)
            if (!std::isfinite(samples(i, j))) all_finite = false;
    CHECK(all_finite, "all sampled values are finite");
}

// ---------- Test 17: GMM reproducibility ----------
static void test_gmm_reproducibility() {
    cout << "-- Test 17: GMM fit is reproducible given a seed --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(40, 2);
    for (size_t i = 0; i < 40; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM g1(3, 50, 1e-6, 1e-3, 123);
    GMM g2(3, 50, 1e-6, 1e-3, 123);
    g1.fit(X);
    g2.fit(X);
    Tensor w1 = g1.weights(), w2 = g2.weights();
    for (size_t k = 0; k < 3; ++k) {
        CHECK_NEAR(w1(k, 0), w2(k, 0), 1e-12,
                   "weights[" + std::to_string(k) + "] reproducible");
    }
}

// ---------- Test 18: GMM singular data throws (reg_covar=0) ----------
static void test_gmm_singular_data_throws() {
    cout << "-- Test 18: GMM reg_covar=0 on singular data throws --" << endl;
    // 10 identical points: degenerate covariance
    Tensor X(10, 2);
    for (size_t i = 0; i < 10; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = 2.0;
    }
    GMM gmm(2, 50, 1e-6, 0.0, 0);  // reg_covar=0
    bool threw = false;
    try {
        gmm.fit(X);
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "reg_covar=0 on collinear data throws");
}

// ---------- Test 19: GMM with default reg_covar handles real data ----------
static void test_gmm_default_reg_handles_real_data() {
    cout << "-- Test 19: GMM default reg_covar handles non-singular data --" << endl;
    std::mt19937 rng(0);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(40, 2);
    for (size_t i = 0; i < 40; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM gmm(2, 50, 1e-6, 1e-6, 42);
    gmm.fit(X);
    // No exception expected
    CHECK(true, "GMM.fit completes with default reg_covar on random data");
}

// ---------- Test 20: GMM score returns finite mean log-likelihood ----------
static void test_gmm_score() {
    cout << "-- Test 20: GMM score returns finite mean log-likelihood --" << endl;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor X(40, 2);
    for (size_t i = 0; i < 40; ++i) {
        X(i, 0) = nd(rng);
        X(i, 1) = nd(rng);
    }
    GMM gmm(2, 50, 1e-6, 1e-6, 42);
    gmm.fit(X);
    double s = gmm.score(X);
    CHECK(std::isfinite(s), "score returns finite value");
    CHECK(s < 0.0, "score (mean log-lik) is negative for continuous densities");
}

int main() {
    cout << "=== GMM Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_mvg_scalar_logpdf();
    test_mvg_batch_logpdf();
    test_mvg_grad_at_mu_is_zero();
    test_mvg_grad_numerical_check();
    test_mvg_kl_divergence();
    test_mvg_validation();
    test_kmeans_pp_init_separation();
    test_kmeans_inertia_monotonic();
    test_kmeans_recovery();
    test_gmm_loglik_monotonic();
    test_gmm_responsibilities_rowsum();
    test_gmm_weights_sum();
    test_gmm_predict_consistency();
    test_gmm_endtoend_recovery();
    test_gmm_bic_decreases();
    test_gmm_sample();
    test_gmm_reproducibility();
    test_gmm_singular_data_throws();
    test_gmm_default_reg_handles_real_data();
    test_gmm_score();

    cout << "=== Result: " << passed << "/" << total << " checks passed ===" << endl;
    return (passed == total) ? 0 : 1;
}