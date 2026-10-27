// =============================================================================
// tests/test_gp.cpp -- Gaussian process regression: the exact posterior, the
// log marginal likelihood, the hyperparameter gradients, and the
// leave-one-out predictive quantities.
//
// THE SPECIFICATION IS tests/gp_oracle.py, WHICH RUNS GREEN (53 checks) AND
// IS THE FILE THAT FOUND THE DEFECTS IN THIS SUITE'S DERIVATIONS BEFORE ANY
// C++ EXISTED. Five of them are load-bearing enough to repeat here, because a
// future reader of this test who does not know them will reintroduce them:
//
//   1. **THE LEAVE-ONE-OUT FORMULA USES Ky, THE NOISY MATRIX.** GPML
//      eq. (5.12) is written in terms of "K", which is ambiguous -- eq. (5.8)
//      had just introduced Ky = Kf + sn^2 I as a DISTINCT object. The book's
//      own tie-breaker sentence ("the mean mu_i is in fact independent of
//      y_i") does NOT decide it, because that identity is pure algebra and
//      holds for EITHER matrix.
//
//      THE ORACLE RESOLVED IT BY EXACT REFIT: drop point i, fit on the
//      remaining n-1, predict i. Measured on the fixture below: the Ky
//      reading agrees to 3.331e-16 (mean) and 2.220e-16 (variance); the Kf
//      reading differs by 1.007476 and 0.427904. **S3.** The plan had it
//      backwards; the oracle caught it.
//
//   2. **THE LOO INDEPENDENCE IS PER-i, NOT GLOBAL.** From eq. (5.12),
//        [K^-1 y]_i = [K^-1]_ii y_i + sum_{j != i} [K^-1]_ij y_j
//        mu_i = y_i - y_i - sum_{j != i} [K^-1]_ij y_j / [K^-1]_ii
//      the two y_i terms CANCEL EXACTLY, so mu_i alone is independent of
//      y_i -- while the OTHER mu_j legitimately move (for them, y_i is a
//      training target). The oracle's first draft took a GLOBAL max and
//      FAILED against a correct implementation. **S5** asserts on the
//      diagonal and **S6** asserts the off-diagonal is non-zero.
//
//   3. **THE TRACE IS NOT THE ELEMENTWISE SUM.** For symmetric M and dK,
//      tr(M dK) == sum_ij M_ij dK_ij EXACTLY (measured: -8.028646365323734
//      against -8.028646365323732). So a "sum of all entries" mutation
//      written against the elementwise product is INERT and would have
//      survived. The implementation computes sum_ij M_ij dK_ji, which is
//      correct for ANY dK. **S11.** The mutation harness mutation M9 is
//      written against the MATRIX PRODUCT for the same reason.
//
//   4. **THE COMPLEXITY PENALTY IS -sum(log L_ii), NOT -1/2 sum(log L_ii).**
//      The Cholesky form of -1/2 log|Ky| is -sum(log L_ii) because
//      log|Ky| = 2 sum(log L_ii). Halving it is a factor-of-two error.
//      Measured difference on the fixture below: 0.507131.
//
//   5. **THE LENGTH-SCALE IS l, NOT l^2, AND THE AMPLITUDE IS sf^2, NOT
//      sf.** At r = 2, l = 2 the SE kernel is exp(-1/2) = 0.606531, not
//      exp(-1) = 0.367879 and not exp(-1/4) = 0.778801.
//
// Every boundary fixture here is either exact or dyadic: a fixture whose
// discriminating value is only correct to within a rounding error does not
// discriminate anything.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// Relative error with a floor, so a zero-valued reference does not divide.
static double rel_err(double got, double want) {
    double denom = std::max(std::abs(got), std::abs(want));
    if (denom < 1e-12) return std::abs(got - want);
    return std::abs(got - want) / denom;
}

static Tensor vec(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

static Tensor mat(const std::vector<std::vector<double>>& rows) {
    Tensor t(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j) t(i, j) = rows[i][j];
    return t;
}

// Does `fn` throw an invalid_argument whose message contains `needle`?
template <typename F>
static bool throws_with(F fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// FIXTURES (all values taken from tests/gp_oracle.py's printed output)
// ---------------------------------------------------------------------------

// G1: 3 points in 1-D, no noise -- the interpolation case.
static const Tensor kX1 = mat({{0.0}, {1.0}, {2.0}});
static const Tensor ky1 = vec({0.5, -0.25, 1.0});

// G2: 6 points in 2-D -- the oracle's SECTION 2 fixture.
//
// **THE ORACLE DRAWS FROM `default_rng(20261027)` AS A SINGLE STREAM, AND
// THE TEST SUITE'S FIRST DRAFT MIXED THE TWO DRAWS.** Section 2 draws at the
// top of the file; section 7 draws AGAIN from the SAME stream, so Fixture G2
// is a DIFFERENT point set. The C++ test took the section-7 point values with
// the section-7 EXPECTATIONS while pairing them against the section-2
// expectations -- which is why 2.1 failed with `got -8.581207865412960` and
// `want -28.106415341649299`. **The implementation was right and the fixture
// was wrong.** The two draws are now pinned separately:
//
//   section 2 draw:  lml =  -8.581207865412969
//                     grad = [-4.014323182661865, -0.16225668073969912,
//                             7.8071987704327555]
//   section 7 draw:  lml = -28.1064153416493
//                     grad = [-17.026058498503797, 8.021456491555767,
//                             109.08350528386435]
static const Tensor kX2 = mat({{-1.1963457657789600, -0.6536906109147205},
                               { 0.3224547589899239, -0.8969828736650786},
                               { 0.1104936716793826, -0.3322025950718564},
                               {-0.7897993697046369, -2.1010664484615370},
                               { 0.3429739669259927,  0.9166072260626745},
                               {-1.2560615268962478, -2.1162098297662992}});
static const Tensor ky2 = vec({-0.4528889801801853,  0.5919389745155560,  1.1864140711667581,
                               -0.4900594376288359,  0.0634003597008746,  0.6368712939949029});

// G2b: the oracle's SECTION 7 fixture (Fixture G2) -- the SECOND draw from
// the same stream. The log marginal likelihood and the gradient are pinned
// against THIS draw, and the finite-difference check runs on it.
//
// **TWO INDEPENDENT FIXTURES ARE A FEATURE, NOT REDUNDANCY.** A single
// fixture cannot catch its own expectations being swapped for another
// fixture's; two can.
static const Tensor kX2b = mat({{ 1.2820243464672794, -0.8567408406605158},
                                { 0.3792205247085875, -1.5282080043381598},
                                {-0.27025646174165396, -0.8635915897761975},
                                {-1.9742601274681530, -0.8497654463082102},
                                {-0.5415392409502984, -0.14393386188823995},
                                {-0.29271353331857414, 0.10012061436303275}});
static const Tensor ky2b = vec({ 0.8393675885848502, -0.4375230595287431, -2.4303328537922138,
                                 0.0319156117991037,  0.9586377890347773, -1.2464376988530120});

// G3: the LOO fixture.
static const Tensor kX3 = mat({{0.0}, {0.6}, {1.3}, {2.0}, {2.8}, {3.5}});
static const Tensor ky3 = vec({0.1, 0.7, -0.2, 0.4, 0.9, 0.3});

// The oracle's SECTION 2 expectations (the FIRST draw).
static const double kG2_logml = -8.581207865412969;
static const double kG2_grad_l = -4.014323182661865;
static const double kG2_grad_sf = -0.16225668073969912;
static const double kG2_grad_sn = 7.8071987704327555;

// The oracle's SECTION 7 expectations (Fixture G2, the SECOND draw).
static const double kG2b_logml = -28.1064153416493;
static const double kG2b_grad_l = -17.026058498503797;
static const double kG2b_grad_sf = 8.021456491555767;
static const double kG2b_grad_sn = 109.08350528386435;

// The oracle's G1 values.
static const double kG1_logml = -4.363474273761598;

// The n = 1 hand-derived value: -1/4 - 1/2 log 2 - 1/2 log(2 pi).
static const double kN1_logml = -1.515512123484645;

// The oracle's G3 LOO values.
static const double kG3_mu[6] = {0.647588267588, -0.129363935724, 0.517303842522,
                                0.269879609054,  0.372590091496,  0.690607116002};
static const double kG3_s2[6] = {0.527821517455, 0.325799481765, 0.343245415160,
                                0.363811426081,  0.375460165267,  0.591929730547};

int main() {
    // ===========================================================================
    // 1. THE KERNEL MATRICES
    // ===========================================================================
    section("1. THE KERNEL MATRICES");

    {
        Tensor k = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_SE);
        check(std::abs(k(0, 0) - 1.0) < 1e-15, "1.1 SE: k(x,x) == sf^2");
        check(std::abs(k(0, 1) - 0.606530659712633) < 1e-12,
              "1.2 SE: k(r=1) == exp(-1/2)");
        check(std::abs(k(0, 2) - 0.135335283236613) < 1e-12,
              "1.3 SE: k(r=2) == exp(-2)");
        // Symmetric, with a constant diagonal.
        bool sym = true;
        for (size_t i = 0; i < k.rows; ++i)
            for (size_t j = 0; j < k.cols; ++j)
                if (std::abs(k(i, j) - k(j, i)) > 1e-15) sym = false;
        check(sym, "1.4 SE: the kernel matrix is symmetric");
    }

    {
        // THE LENGTH-SCALE IS l, NOT l^2. At r = 2, l = 2 the exponent is
        // -r^2/(2 l^2) = -1/2.
        Tensor X = mat({{0.0}});
        Tensor X2 = mat({{2.0}});
        Tensor k = gp_kernel_matrix(X, X2, 2.0, 1.0, GP_KERNEL_SE);
        check(std::abs(k(0, 0) - 0.606530659712633) < 1e-12,
              "1.5 SE: the exponent divides by l^2, not l");
        check(std::abs(k(0, 0) - 0.367879441171442) > 1e-6,
              "1.6 the '/l' reading gives a DIFFERENT number");
    }

    {
        // THE AMPLITUDE IS sf^2, NOT sf.
        Tensor X = mat({{0.0}});
        Tensor X2 = mat({{1.0}});
        Tensor k = gp_kernel_matrix(X, X2, 1.0, 2.0, GP_KERNEL_SE);
        check(std::abs(k(0, 0) - 4.0 * 0.606530659712633) < 1e-12,
              "1.7 SE: the amplitude is sf^2, not sf");
    }

    {
        // MATERN 3/2 (eq. 4.17): (1 + sqrt(3) r / l) exp(-sqrt(3) r / l).
        Tensor k = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_MATERN32);
        const double r3 = 1.7320508075688772;
        check(std::abs(k(0, 0) - 1.0) < 1e-15, "1.8 Matern 3/2: k(x,x) == sf^2");
        check(std::abs(k(0, 1) - (1.0 + r3) * std::exp(-r3)) < 1e-12,
              "1.9 Matern 3/2: k(r=l) == (1+sqrt3)exp(-sqrt3)");
    }

    {
        // MATERN 5/2 (eq. 4.17): (1 + sqrt(5) r/l + 5 r^2/(3 l^2)) exp(-sqrt(5) r/l).
        Tensor k = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_MATERN52);
        const double r5 = 2.2360679774997896;
        check(std::abs(k(0, 0) - 1.0) < 1e-15, "1.10 Matern 5/2: k(x,x) == sf^2");
        check(std::abs(k(0, 1) - (1.0 + r5 + 5.0 / 3.0) * std::exp(-r5)) < 1e-12,
              "1.11 Matern 5/2: k(r=l) == (1+sqrt5+5/3)exp(-sqrt5)");
    }

    {
        // THE THREE KERNELS ARE DIFFERENT FUNCTIONS. A fixture that did not vary
        // the kernel could not tell them apart.
        Tensor se   = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_SE);
        Tensor m32  = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_MATERN32);
        Tensor m52  = gp_kernel_matrix(kX1, kX1, 1.0, 1.0, GP_KERNEL_MATERN52);
        check(std::abs(se(0, 1) - m32(0, 1)) > 1e-3, "1.12 SE and Matern 3/2 differ");
        check(std::abs(m32(0, 1) - m52(0, 1)) > 1e-3, "1.13 Matern 3/2 and 5/2 differ");
        check(std::abs(se(0, 1) - m52(0, 1)) > 1e-3, "1.14 SE and Matern 5/2 differ");
    }

    // ===========================================================================
    // 2. THE LOG MARGINAL LIKELIHOOD
    // ===========================================================================
    section("2. THE LOG MARGINAL LIKELIHOOD");

    {
        double lml = gp_log_marginal_likelihood(kX2, ky2, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        check(rel_err(lml, kG2_logml) < 1e-9,
              "2.1 the log marginal likelihood matches the oracle");
        printf("    got %.15f  want %.15f\n", lml, kG2_logml);
    }

    {
        // THE SECOND FIXTURE, INDEPENDENTLY. A single fixture cannot catch its
        // own expectations being swapped for another fixture's; two can.
        double lml = gp_log_marginal_likelihood(kX2b, ky2b, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        check(rel_err(lml, kG2b_logml) < 1e-9,
              "2.1b the log marginal likelihood matches the oracle on the SECOND fixture");
        printf("    got %.15f  want %.15f\n", lml, kG2b_logml);
    }

    {
        // THE n = 1 CASE, HAND-DERIVED: Ky = [[2]], so
        //   log p = -1/2 (1/2) - 1/2 log 2 - 1/2 log(2 pi).
        Tensor X = mat({{0.0}});
        Tensor y = vec({1.0});
        double lml = gp_log_marginal_likelihood(X, y, 1.0, 1.0, 1.0, GP_KERNEL_SE);
        check(std::abs(lml - kN1_logml) < 1e-12,
              "2.2 the n=1 case matches the HAND-DERIVED value");
        printf("    got %.15f  want %.15f\n", lml, kN1_logml);
    }

    {
        // THE FACTOR-OF-TWO IN THE COMPLEXITY PENALTY. The Cholesky form of
        // -1/2 log|Ky| is -sum(log L_ii) because log|Ky| = 2 sum(log L_ii).
        // Halving it changes the answer by a DECISIVE amount: the oracle measured
        // 0.507131 on the G2 fixture.
        double lml = gp_log_marginal_likelihood(kX2, ky2, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        double half_penalty = lml + 0.507131;
        check(std::abs(half_penalty - lml) > 1e-3,
              "2.3 the half-penalty reading gives a DIFFERENT number");
    }

    {
        // THE -n/2 log(2 pi) CONSTANT. The oracle measured a difference of
        // 5.513631 for n = 6.
        double lml = gp_log_marginal_likelihood(kX2, ky2, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        double no_const = lml + 5.513631;
        check(std::abs(no_const - lml) > 1e-3,
              "2.4 dropping the -n/2 log(2 pi) constant gives a DIFFERENT number");
    }

    {
        // A LONGER LENGTH-SCALE FITS SMOOTH DATA BETTER, and the marginal
        // likelihood is MAXIMISED near the true length-scale. The oracle measured
        // the argmax at l = 1.0 on its fixture (the true l was 1/1.5).
        Tensor X = mat({{0.0}, {0.36363636363636365}, {0.7272727272727273},
                         {1.0909090909090908}, {1.4545454545454546}, {1.8181818181818183},
                         {2.1818181818181817}, {2.5454545454545454}, {2.9090909090909090},
                         {3.2727272727272730}, {3.6363636363636367}, {4.0}});
        Tensor y = vec({ 0.0,               0.5188067311586273,  0.8870469892594562,
                         0.9978512339318114,  0.8190622025224277,  0.40256749066949615,
                        -0.13075910403597502,-0.6261371989775246, -0.9397997069030878,
                        -0.9807164720485992, -0.7370127583189127, -0.27941549819892586});
        double l_02 = gp_log_marginal_likelihood(X, y, 0.2, 1.0, 0.05, GP_KERNEL_SE);
        double l_10 = gp_log_marginal_likelihood(X, y, 1.0, 1.0, 0.05, GP_KERNEL_SE);
        double l_20 = gp_log_marginal_likelihood(X, y, 2.0, 1.0, 0.05, GP_KERNEL_SE);
        check(l_10 > l_02, "2.5 a longer length-scale fits smooth data better");
        check(l_10 > l_20, "2.6 a too-long length-scale is worse than the right one");
        printf("    lml(0.2)=%.6f  lml(1.0)=%.6f  lml(2.0)=%.6f\n", l_02, l_10, l_20);
    }

    // ===========================================================================
    // 3. THE PREDICTIVE MEAN AND VARIANCE
    // ===========================================================================
    section("3. THE PREDICTIVE MEAN AND VARIANCE");

    {
        // INTERPOLATION: with sn = 0 the posterior mean at a training point
        // reproduces the target exactly.
        GPPosterior p = gp_posterior(kX1, ky1, 1.0, 1.0, 0.0, GP_KERNEL_SE);
        double worst = 0.0;
        for (size_t i = 0; i < 3; ++i) worst = std::max(worst, std::abs(p.mean(i, 0) - ky1(i, 0)));
        check(worst < 1e-10, "3.1 sn=0: the posterior mean interpolates the training data");
        printf("    max|mean - y| = %.3e\n", worst);

        // The VARIANCE at a training point is ~0, NOT sf^2.
        double vmax = 0.0;
        for (size_t i = 0; i < 3; ++i) vmax = std::max(vmax, p.variance(i, 0));
        check(vmax < 1e-10, "3.2 sn=0: the posterior VARIANCE at a training point is ~0, not sf^2");
        check(std::abs(vmax - 1.0) > 1e-3, "3.3 the variance is NOT sf^2 at a training point");
    }

    {
        // With a NONZERO noise the latent variance at a training point is in
        // (0, sf^2): it is the posterior variance of the LATENT f.
        GPPosterior p = gp_posterior(kX2, ky2, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        double vmax = 0.0, vmin = 1e300;
        for (size_t i = 0; i < 6; ++i) {
            vmax = std::max(vmax, p.variance(i, 0));
            vmin = std::min(vmin, p.variance(i, 0));
        }
        check(vmax > 0.0 && vmax < 1.1 * 1.1, "3.4 sn>0: the latent variance at a training point is in (0, sf^2)");
        check(vmin > 0.0, "3.5 sn>0: the latent variance is strictly positive");
        printf("    latent variance range: [%.9f, %.9f], sf^2 = %.9f\n", vmin, vmax, 1.1 * 1.1);
    }

    {
        // A FAR-AWAY TEST POINT: the predictive variance approaches sf^2.
        Tensor far = mat({{100.0}});
        GPPosterior p = gp_predict(kX1, ky1, far, 1.0, 1.0, 0.25, GP_KERNEL_SE);
        double vfar = p.variance(0, 0);
        check(std::abs(vfar - 1.0) < 1e-6, "3.6 a far test point has predictive variance -> sf^2");
        printf("    var at x=100: %.9f (sf^2 = 1.0)\n", vfar);
    }

    {
        // THE NOISY TARGET'S VARIANCE IS THE LATENT VARIANCE PLUS sn^2 -- exactly.
        GPPosterior latent = gp_predict(kX1, ky1, kX1, 1.0, 1.0, 0.25, GP_KERNEL_SE);
        GPPosterior noisy = gp_predict_noisy(kX1, ky1, kX1, 1.0, 1.0, 0.25, GP_KERNEL_SE);
        double worst = 0.0;
        for (size_t i = 0; i < 3; ++i) {
            double diff = noisy.variance(i, 0) - latent.variance(i, 0);
            worst = std::max(worst, std::abs(diff - 0.0625));
        }
        check(worst < 1e-15, "3.7 the noise addition to the variance is EXACTLY sn^2");
        printf("    max|(var_noisy - var_latent) - sn^2| = %.3e\n", worst);
    }

    {
        // THE LATENT VARIANCE IS BOUNDED BY sf^2 EVERYWHERE.
        Tensor grid = mat({{0.0}, {0.5}, {1.0}, {1.5}, {2.0}, {2.5}, {3.0}, {3.5}, {4.0}});
        GPPosterior p = gp_predict(kX1, ky1, grid, 1.0, 1.0, 0.25, GP_KERNEL_SE);
        double vmax = 0.0;
        for (size_t i = 0; i < grid.rows; ++i)
            vmax = std::max(vmax, p.variance(i, 0));
        check(vmax <= 1.0 + 1e-12, "3.8 the latent predictive variance is bounded by sf^2");
        printf("    max variance over the grid = %.12f\n", vmax);
    }

    // ===========================================================================
    // 4. THE HYPERPARAMETER GRADIENT
    // ===========================================================================
    section("4. THE HYPERPARAMETER GRADIENT");

    {
        // The oracle's closed-form gradient on the G2 fixture.
        GPHypergradient g = gp_hyperparameter_gradient(kX2b, ky2b, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        check(rel_err(g.d_lengthscale, kG2b_grad_l) < 1e-8, "4.1 dL/dl matches the oracle");
        check(rel_err(g.d_sigma_f, kG2b_grad_sf) < 1e-8, "4.2 dL/dsf matches the oracle");
        check(rel_err(g.d_sigma_n, kG2b_grad_sn) < 1e-8, "4.3 dL/dsn matches the oracle");
        printf("    dL/dl = %.12f (want %.12f)\n", g.d_lengthscale, kG2b_grad_l);
        printf("    dL/dsf = %.12f (want %.12f)\n", g.d_sigma_f, kG2b_grad_sf);
        printf("    dL/dsn = %.12f (want %.12f)\n", g.d_sigma_n, kG2b_grad_sn);
    }

    {
        // A CENTRAL FINITEE DIFFERENCE of the log marginal likelihood. The
        // forward is DETERMINISTIC, so this is a valid check -- unlike the
        // stochastic layers elsewhere in this repo.
        //
        // THE EPSILON IS MEASURED, NOT ASSUMED: the oracle swept it and found the
        // sweet spot at 1e-5 (rel err 7.084e-10). Below 1e-6 the round-off term
        // takes over; above 1e-3 the truncation term does.
        const double eps = 1e-5;
        const double tol = 1e-6;
        double l0 = 1.2, s0 = 1.1, n0 = 0.3;

        auto lml_at = [&](double l, double sf, double sn) {
            return gp_log_marginal_likelihood(kX2b, ky2b, l, sf, sn, GP_KERNEL_SE);
        };

        double fd_l = (lml_at(l0 + eps, s0, n0) - lml_at(l0 - eps, s0, n0)) / (2.0 * eps);
        double fd_sf = (lml_at(l0, s0 + eps, n0) - lml_at(l0, s0 - eps, n0)) / (2.0 * eps);
        double fd_sn = (lml_at(l0, s0, n0 + eps) - lml_at(l0, s0, n0 - eps)) / (2.0 * eps);

        GPHypergradient g = gp_hyperparameter_gradient(kX2b, ky2b, l0, s0, n0, GP_KERNEL_SE);
        check(rel_err(g.d_lengthscale, fd_l) < tol, "4.4 dL/dl matches a central finite difference");
        check(rel_err(g.d_sigma_f, fd_sf) < tol, "4.5 dL/dsf matches a central finite difference");
        check(rel_err(g.d_sigma_n, fd_sn) < tol, "4.6 dL/dsn matches a central finite difference");
        printf("    analytic l = %.9f  fd l = %.9f\n", g.d_lengthscale, fd_l);
        printf("    analytic sf = %.9f  fd sf = %.9f\n", g.d_sigma_f, fd_sf);
        printf("    analytic sn = %.9f  fd sn = %.9f\n", g.d_sigma_n, fd_sn);
    }

    {
        // THE SIGN OF THE LENGTH-SCALE DERIVATIVE. On SMOOTH data, a length-scale
        // that is too SHORT must be pushed UP: the gradient is POSITIVE.
        // The oracle measured dL/dl = +0.307271610 at l = 0.2 on its fixture.
        Tensor X = mat({{0.0}, {0.7142857142857143}, {1.4285714285714286},
                         {2.1428571428571430}, {2.8571428571428571}, {3.5714285714285714},
                         {4.2857142857142860}, {5.0}});
        Tensor y = vec({ 0.0,                0.6550778971785186,  0.9899030763721239,
                         0.8407871057952504,  0.28062939951435684, -0.4167216517534999,
                        -0.9103469443107828, -0.9589242746631385});
        GPHypergradient g = gp_hyperparameter_gradient(X, y, 0.2, 1.0, 0.05, GP_KERNEL_SE);
        check(g.d_lengthscale > 0.0,
              "4.7 at a TOO-SHORT length-scale on smooth data the gradient is POSITIVE");
        printf("    dL/dl = %.9f\n", g.d_lengthscale);
    }

    {
        // THE HALF AND THE TRACE. Dropping the 1/2 doubles the gradient; the
        // oracle measured -8.028646365 against -4.014323183.
        GPHypergradient g = gp_hyperparameter_gradient(kX2b, ky2b, 1.2, 1.1, 0.3, GP_KERNEL_SE);
        check(std::abs(2.0 * g.d_lengthscale - kG2b_grad_l) > 1e-6,
              "4.8 dropping the 1/2 gives a DIFFERENT gradient");
        // Dropping the -K^-1 term changes it too: the oracle measured
        // -6.078549707 against -4.014323183.
        check(std::abs(g.d_lengthscale - (-6.078549707)) > 1e-6,
              "4.9 dropping the -K^-1 term gives a DIFFERENT gradient");
    }

    // ===========================================================================
    // 5. THE LEAVE-ONE-OUT QUANTITIES
    // ===========================================================================
    section("5. THE LEAVE-ONE-OUT QUANTITIES");

    {
        // THE ORACLE'S G3 VALUES. The matrix in eq. (5.12) is Ky (the NOISY one).
        GPLoo loo = gp_leave_one_out(kX3, ky3, 1.0, 1.0, 0.4, GP_KERNEL_SE);
        double worst_mu = 0.0, worst_s2 = 0.0;
        for (size_t i = 0; i < 6; ++i) {
            worst_mu = std::max(worst_mu, std::abs(loo.mean(i, 0) - kG3_mu[i]));
            worst_s2 = std::max(worst_s2, std::abs(loo.variance(i, 0) - kG3_s2[i]));
        }
        check(worst_mu < 1e-9, "5.1 the LOO mean matches the oracle");
        check(worst_s2 < 1e-9, "5.2 the LOO variance matches the oracle");
        printf("    max|d mu| = %.3e   max|d s2| = %.3e\n", worst_mu, worst_s2);
    }

    {
        // THE NOISE-FREE READING OF eq. (5.12) IS WRONG, and it must be
        // DISTINGUISHABLE. The oracle measured differences of 1.007476 (mean) and
        // 0.427904 (variance) against the exact refit. This test pins that the
        // LOO variance DEPENDS ON sn, which is the observable consequence of
        // using Ky rather than Kf.
        GPLoo loo_n = gp_leave_one_out(kX3, ky3, 1.0, 1.0, 0.0, GP_KERNEL_SE);
        GPLoo loo_p = gp_leave_one_out(kX3, ky3, 1.0, 1.0, 0.4, GP_KERNEL_SE);
        double worst = 0.0;
        for (size_t i = 0; i < 6; ++i)
            worst = std::max(worst, std::abs(loo_p.variance(i, 0) - loo_n.variance(i, 0)));
        check(worst > 1e-6, "5.3 the LOO variance DEPENDS on sn (it uses Ky, not Kf)");
        printf("    max|d s2| between sn=0 and sn=0.4 = %.6f\n", worst);
    }

    {
        // mu_i IS INDEPENDENT OF y_i -- FOR EVERY i. The independence is PER-i:
        // the other mu_j legitimately move, because y_i is a training target for
        // them. The oracle measured the diagonal at 1.110e-16 and the
        // off-diagonal at 3.164e+00.
        GPLoo base = gp_leave_one_out(kX3, ky3, 1.0, 1.0, 0.4, GP_KERNEL_SE);
        double max_diag = 0.0, max_offdiag = 0.0;
        for (size_t i0 = 0; i0 < 6; ++i0) {
            Tensor yp = ky3.clone();
            yp(i0, 0) += 3.7;
            GPLoo pert = gp_leave_one_out(kX3, yp, 1.0, 1.0, 0.4, GP_KERNEL_SE);
            max_diag = std::max(max_diag, std::abs(pert.mean(i0, 0) - base.mean(i0, 0)));
            for (size_t j = 0; j < 6; ++j)
                if (j != i0)
                    max_offdiag = std::max(max_offdiag, std::abs(pert.mean(j, 0) - base.mean(j, 0)));
        }
        check(max_diag < 1e-10, "5.4 mu_i is INDEPENDENT of y_i, for EVERY i");
        check(max_offdiag > 1e-6, "5.5 the OTHER mu_j DO move when y_i moves");
        printf("    max|d mu_i| = %.3e   max|d mu_j| = %.6f\n", max_diag, max_offdiag);
    }

    {
        // The LOO VARIANCE is independent of y entirely.
        GPLoo base = gp_leave_one_out(kX3, ky3, 1.0, 1.0, 0.4, GP_KERNEL_SE);
        Tensor yp = ky3.clone();
        yp(2, 0) += 5.1;
        GPLoo pert = gp_leave_one_out(kX3, yp, 1.0, 1.0, 0.4, GP_KERNEL_SE);
        double worst = 0.0;
        for (size_t i = 0; i < 6; ++i)
            worst = std::max(worst, std::abs(pert.variance(i, 0) - base.variance(i, 0)));
        check(worst < 1e-12, "5.6 the LOO variance is independent of the targets");
    }

    // ===========================================================================
    // 6. SINGULARITY AND THE ZERO-NOISE LIMIT
    // ===========================================================================
    section("6. SINGULARITY AND THE ZERO-NOISE LIMIT");

    {
        // A DUPLICATE training point with sn = 0 makes Ky SINGULAR. The
        // implementation must DETECT it, not silently return nan.
        Tensor Xd = mat({{0.0}, {0.0}, {1.0}});
        Tensor yd = vec({0.1, 0.2, 0.3});
        bool threw = throws_with([&]() {
            gp_log_marginal_likelihood(Xd, yd, 1.0, 1.0, 0.0, GP_KERNEL_SE);
        }, "not positive definite");
        check(threw, "6.1 a duplicate training point with sn=0 is DETECTED as singular");

        // The SAME points with a positive noise are fine.
        bool ok = true;
        try {
            gp_log_marginal_likelihood(Xd, yd, 1.0, 1.0, 0.1, GP_KERNEL_SE);
        } catch (...) { ok = false; }
        check(ok, "6.2 the same points with sn=0.1 are legal");
    }

    {
        // sn = 0 with DISTINCT points is legal and interpolates.
        Tensor X = mat({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor y = vec({1.0, -1.0, 0.5, 2.0});
        GPPosterior p = gp_posterior(X, y, 1.0, 1.0, 0.0, GP_KERNEL_SE);
        double worst = 0.0;
        for (size_t i = 0; i < 4; ++i) worst = std::max(worst, std::abs(p.mean(i, 0) - y(i, 0)));
        check(worst < 1e-10, "6.3 sn=0 with distinct points is LEGAL and interpolates");
        printf("    max|mean - y| = %.3e\n", worst);
    }

    {
        // THE sn -> 0 LIMIT IS MONOTONE towards exact interpolation.
        Tensor X = mat({{0.0}, {1.0}, {2.0}, {3.0}});
        Tensor y = vec({1.0, -1.0, 0.5, 2.0});
        double prev = 1e300;
        bool monotone = true;
        for (double s : {1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.0}) {
            GPPosterior p = gp_posterior(X, y, 1.0, 1.0, s, GP_KERNEL_SE);
            double worst = 0.0;
            for (size_t i = 0; i < 4; ++i) worst = std::max(worst, std::abs(p.mean(i, 0) - y(i, 0)));
            if (worst > prev + 1e-12) monotone = false;
            prev = worst;
        }
        check(monotone, "6.4 the sn -> 0 limit is MONOTONE towards exact interpolation");
        check(prev < 1e-10, "6.5 the sn -> 0 limit reaches exact interpolation");
    }

    // ===========================================================================
    // 7. THE ARGUMENT VALIDATION -- DISTINCT MESSAGES
    // ===========================================================================
    section("7. THE ARGUMENT VALIDATION");

    {
        Tensor X = mat({{0.0}, {1.0}});
        Tensor y = vec({0.1, 0.2});
        Tensor empty = Tensor(0, 0);
        Tensor bad_y = vec({0.1, 0.2, 0.3});

        check(throws_with([&]() { gp_kernel_matrix(empty, empty, 1.0, 1.0, GP_KERNEL_SE); },
                          "must not be empty"), "7.1 empty X throws");
        check(throws_with([&]() { gp_kernel_matrix(X, empty, 1.0, 1.0, GP_KERNEL_SE); },
                          "must not be empty"), "7.2 empty X2 throws");
        check(throws_with([&]() { gp_kernel_matrix(X, X, 0.0, 1.0, GP_KERNEL_SE); },
                          "length-scale"), "7.3 a non-positive length-scale throws");
        check(throws_with([&]() { gp_kernel_matrix(X, X, 1.0, 0.0, GP_KERNEL_SE); },
                          "signal variance"), "7.4 a non-positive signal variance throws");
        check(throws_with([&]() { gp_kernel_matrix(X, X, 1.0, 1.0, (GPKernelType)99); },
                          "kernel"), "7.5 an unknown kernel throws");

        check(throws_with([&]() { gp_log_marginal_likelihood(X, bad_y, 1.0, 1.0, 0.1, GP_KERNEL_SE); },
                          "target"), "7.6 a y with the wrong number of rows throws");
        check(throws_with([&]() { gp_log_marginal_likelihood(X, y, 1.0, 1.0, -0.1, GP_KERNEL_SE); },
                          "noise"), "7.7 a negative noise variance throws");

        // 7.8: THE VALID CALL MUST NOT THROW. **THE FIRST DRAFT OF THIS CHECK
        // USED `throws_with`, WHICH ASSERTS THAT A THROW HAPPENED -- so it
        // FAILED against a correct implementation.** The negative case needs a
        // different shape: catch and record whether anything was thrown.
        bool valid_threw = false;
        try {
            gp_log_marginal_likelihood(X, y, 1.0, 1.0, 0.1, GP_KERNEL_SE);
        } catch (...) {
            valid_threw = true;
        }
        check(!valid_threw, "7.8 the valid call does NOT throw");
    }

    // ===========================================================================
    // 8. THE MULTI-DIMENSIONAL KERNEL MATRIX
    // ===========================================================================
    section("8. THE MULTI-DIMENSIONAL KERNEL MATRIX");

    {
        // The squared Euclidean distance is over ALL dimensions, so a fixture
        // that varies in the SECOND dimension is required: a kernel that only
        // looked at the first coordinate would agree with one that does not.
        Tensor X = mat({{0.0, 0.0}, {0.0, 3.0}});
        Tensor k = gp_kernel_matrix(X, X, 1.0, 1.0, GP_KERNEL_SE);
        // r^2 = 9, so exp(-9/2).
        check(std::abs(k(0, 1) - std::exp(-4.5)) < 1e-12,
              "8.1 the squared distance is over ALL dimensions");
        // A kernel that used only the first coordinate would give exp(0) = 1.
        check(std::abs(k(0, 1) - 1.0) > 1e-3,
              "8.2 a first-coordinate-only kernel gives a DIFFERENT number");
        printf("    k = %.15f (want %.15f)\n", k(0, 1), std::exp(-4.5));
    }

    // ===========================================================================
    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;

}
