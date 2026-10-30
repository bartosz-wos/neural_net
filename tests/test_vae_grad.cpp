// =============================================================================
// tests/test_vae_grad.cpp -- the plain VAE layer's KL GRADIENT.
//
// WHY THIS FILE EXISTS
// =============================================================================
//
// `tests/test_vae.cpp` pins the VAE's LOSS VALUES (and found that the KL
// was a SUM over the batch while the reconstruction loss was a MEAN).
// **A LOSS AND ITS GRADIENT ARE TWO DIFFERENT CLAIMS, AND CHECKING ONE
// DOES NOT CHECK THE OTHER.** This file pins the gradient.
//
// WHAT IT PINS, AND WHY EACH MEASUREMENT IS THE WAY IT IS
// =============================================================================
//
// THE KL'S CLOSED FORM (per latent coordinate, before the batch mean):
//
//     kl_ij = -0.5 * (1 + logvar_ij - mu_ij^2 - exp(logvar_ij))
//
// so, for the MEAN-form loss this repo now uses (the sum over rows divided
// by the batch size N):
//
//     d(mean kl)/d mu_ij     = (1/N) * (+mu_ij)
//     d(mean kl)/d logvar_ij = (1/N) * 0.5*(exp(logvar_ij) - 1)
//
// **THE SIGN ON THE mu TERM IS THE TRAP.** The closed form carries
// `- mu^2` with a MINUS in front of it, so
//
//     d/dmu [ -0.5 * (1 + logvar - mu^2 - exp(logvar)) ]
//         = -0.5 * ( -2*mu )
//         = +mu
//
// and the layer's own comment claimed the opposite ("dKL/dmu = -mu_ij
// (NOT +mu_ij -- the sign is negative)"). **A COMMENT THAT ASSERTS A SIGN
// IS NOT EVIDENCE OF ONE.** Measured, the layer produced -0.484948 where
// the finite difference gave +0.484948 -- a ratio of exactly -1.0 for a
// 1-row batch.
//
// AND THE SIGN IS NOT COSMETIC. With the sign as it was, gradient DESCENT
// moved `mu` AWAY from zero, which is the direction that INCREASES the KL
// -- the KL term was actively pushing the encoder away from the prior.
//
// THE 1/N FACTOR IS A SECOND, SEPARATE DEFECT. Measured ratios against the
// finite difference, for the layer as it stood:
//
//     N=1:  mu-bias ratio -1.000000    logvar-bias ratio +1.000013
//     N=2:  mu-bias ratio -2.000000    logvar-bias ratio +2.000025
//     N=3:  mu-bias ratio -3.000000    logvar-bias ratio +3.000038
//
// The mu ratio is -N (wrong sign AND missing the 1/N) and the logvar ratio
// is +N (right sign, missing the 1/N). **THE RATIO SCALING WITH N IS WHAT
// IDENTIFIES THE MISSING 1/N**, and it is why the test asserts the ratio
// for SEVERAL batch sizes rather than only one.
//
// WHY THE BIAS IS THE PROBE, AND NOT A WEIGHT
// =============================================================================
//
// The encoder's bias shifts `mu` (or `logvar`) DIRECTLY:
//
//     mu_ij     = W_ij . h + b_j
//     logvar_ij = V_ij . h + c_j
//
// so d(mu_ij)/d b_j = 1, with no weight path, no ReLU and no other layer
// in between. A weight probe would be measuring the ENCODER's own
// gradient path as well as the KL's, and the two would be confounded.
//
// **AND THE KL IS THE RIGHT THING TO DIFFERENTIATE.** The reparameterisation
// draws a fresh `eps ~ N(0,1)` on every forward, so the finite difference
// of `total_loss()` is the difference of two UNCORRELATED noisy values --
// the repo's recorded "stochastic forward" trap, where the central
// difference measures noise rather than a derivative. The KL does not
// depend on `eps` at all (measured: two forwards of the same input give a
// BIT-IDENTICAL KL), so its finite difference IS a valid derivative.
//
// The upstream gradient is set to ZERO, so the measured gradient is PURELY
// the KL's own contribution -- which is exactly what is under test.
//
// =============================================================================

#include "nn/nn.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// The finite-difference derivative of kl_loss() with respect to one entry
// of a bias, holding everything else fixed.
//
// The KL does not depend on the reparameterisation's noise, so this is a
// valid derivative -- unlike the same measurement on total_loss().
static double fd_kl_wrt_bias(VAE& v, Tensor* bias, size_t col,
                             const Tensor& x, double eps) {
    const double orig = (*bias)(0, col);
    (*bias)(0, col) = orig + eps;
    v.forward(x);
    const double kp = v.kl_loss();
    (*bias)(0, col) = orig - eps;
    v.forward(x);
    const double km = v.kl_loss();
    (*bias)(0, col) = orig;
    return (kp - km) / (2.0 * eps);
}

// A deterministic, non-trivial encoder: the biases are set away from zero
// so the KL is non-trivial, and the mu weights are set to a deterministic
// non-zero pattern so `mu` depends on the input rather than on the bias
// alone.
//
// The bias values are what make the KL's own gradient non-zero: with a
// zero encoder the KL is exactly 0 and its derivative is 0, which would
// make the whole measurement vacuous.
static void configure(VAE& v, std::vector<Tensor*>& params) {
    params = v.parameters();
    Tensor* Wmu = params[2];
    Tensor* bmu = params[3];
    Tensor* blv = params[5];
    for (size_t j = 0; j < bmu->cols; ++j) (*bmu)(0, j) = 0.3 + 0.1 * static_cast<double>(j);
    for (size_t j = 0; j < blv->cols; ++j) (*blv)(0, j) = 0.2 + 0.1 * static_cast<double>(j);
    for (size_t i = 0; i < Wmu->rows; ++i) {
        for (size_t j = 0; j < Wmu->cols; ++j) {
            (*Wmu)(i, j) = 0.1 * std::sin(3.0 * static_cast<double>(i * Wmu->cols + j)) + 0.05;
        }
    }
}

int main() {
    printf("=== VAE: the KL gradient ===\n");

    // =========================================================================
    // 1. THE mu-BIAS GRADIENT: THE SIGN AND THE 1/N
    //
    // The expected gradient of the MEAN-form KL w.r.t. a mu-bias entry is
    //     (1/N) * sum_i mu_i0
    // and the finite difference measures exactly that.
    // =========================================================================
    section("1. the mu-bias gradient: the sign and the 1/N");
    {
        for (size_t N = 1; N <= 3; ++N) {
            VAE v(3, 8, 2);
            std::vector<Tensor*> params;
            configure(v, params);
            Tensor* bmu = params[3];

            Tensor x(N, 3);
            for (size_t i = 0; i < N; ++i) { x(i, 0) = 1.0; x(i, 1) = -2.0; x(i, 2) = 0.5; }

            // -- NON-VACUITY: the KL must be non-zero, or its derivative is
            //    zero and the measurement says nothing.
            v.forward(x);
            check(std::abs(v.kl_loss()) > 1e-6,
                  "1." + std::to_string(N) + " the KL is non-trivial (so its "
                  "derivative is measurable)");

            const double fd = fd_kl_wrt_bias(v, bmu, 0, x, 1e-6);

            v.zero_grad();
            Tensor go(N, 3);
            go.fill(0.0);
            v.backward(go, 0.0);
            std::vector<Tensor*> grads = v.gradients();
            const double ana = (*grads[3])(0, 0);

            const double ratio = (fd != 0.0) ? ana / fd : 0.0;
            printf("    N=%zu  FD = %+.12f   analytical = %+.12f   ratio = %+.6f\n",
                   N, fd, ana, ratio);

            check(std::abs(ratio - 1.0) < 1e-4,
                  "1." + std::to_string(N) + "a the analytical gradient EQUALS the "
                  "finite difference (N=" + std::to_string(N) + ")");
            check(ana > 0.0,
                  "1." + std::to_string(N) + "b the analytical gradient is POSITIVE "
                  "-- gradient descent must move mu TOWARD zero, which is the "
                  "direction that REDUCES the KL");
        }
    }

    // =========================================================================
    // 2. THE LOGVAR-BIAS GRADIENT: THE 1/N
    //
    // The expected gradient is (1/N) * 0.5*(exp(logvar) - 1), summed over
    // the rows. Its SIGN was already correct; the 1/N was missing.
    // =========================================================================
    section("2. the logvar-bias gradient: the 1/N");
    {
        for (size_t N = 1; N <= 3; ++N) {
            VAE v(3, 8, 2);
            std::vector<Tensor*> params;
            configure(v, params);
            Tensor* blv = params[5];

            Tensor x(N, 3);
            for (size_t i = 0; i < N; ++i) { x(i, 0) = 1.0; x(i, 1) = -2.0; x(i, 2) = 0.5; }

            const double fd = fd_kl_wrt_bias(v, blv, 0, x, 1e-6);

            v.zero_grad();
            Tensor go(N, 3);
            go.fill(0.0);
            v.backward(go, 0.0);
            std::vector<Tensor*> grads = v.gradients();
            const double ana = (*grads[5])(0, 0);

            const double ratio = (fd != 0.0) ? ana / fd : 0.0;
            printf("    N=%zu  FD = %+.12f   analytical = %+.12f   ratio = %+.6f\n",
                   N, fd, ana, ratio);

            check(std::abs(ratio - 1.0) < 1e-4,
                  "2." + std::to_string(N) + " the analytical gradient EQUALS the "
                  "finite difference (N=" + std::to_string(N) + ")");
        }
    }

    // =========================================================================
    // 3. THE GRADIENT DOES NOT DEPEND ON THE NOISE DRAW
    //
    // The KL's gradient is a function of the encoder's output alone, so two
    // backwards passes on the same input must produce the SAME gradient.
    // This is what makes the finite-difference measurement above valid, and
    // it is worth asserting on its own.
    // =========================================================================
    section("3. the KL gradient is deterministic");
    {
        VAE v(3, 8, 2);
        std::vector<Tensor*> params;
        configure(v, params);
        Tensor x(2, 3);
        for (size_t i = 0; i < 2; ++i) { x(i, 0) = 1.0; x(i, 1) = -2.0; x(i, 2) = 0.5; }

        Tensor go(2, 3);
        go.fill(0.0);

        v.forward(x);
        v.zero_grad();
        v.backward(go, 0.0);
        std::vector<Tensor*> g1 = v.gradients();
        const double a1 = (*g1[3])(0, 0);

        v.forward(x);   // a fresh noise draw
        v.zero_grad();
        v.backward(go, 0.0);
        std::vector<Tensor*> g2 = v.gradients();
        const double a2 = (*g2[3])(0, 0);

        printf("    backward 1: %.15f\n    backward 2: %.15f\n", a1, a2);
        check(a1 == a2,
              "3.1 two backwards passes on the same input give a BIT-IDENTICAL "
              "KL gradient (it does not depend on the noise draw)");
    }

    // =========================================================================
    // 4. THE KL GRADIENT PUSHES mu TOWARD ZERO
    //
    // The functional statement behind the sign: the KL is minimised at
    // mu = 0, so the gradient of the KL must have the SAME SIGN as mu (so
    // that gradient descent, which SUBTRACTS the gradient, moves mu toward
    // zero). **A SIGN THAT IS WRONG HERE DOES NOT JUST HAVE THE WRONG
    // MAGNITUDE -- IT MAKES THE OPTIMISATION GO THE WRONG WAY.**
    //
    // The check is on the SIGN, computed from the layer's own output
    // rather than from a hard-coded expectation, and the two must AGREE.
    // =========================================================================
    section("4. the gradient's sign matches mu's sign");
    {
        // A NEGATIVE mu: the KL gradient must then be NEGATIVE too, so
        // that gradient descent moves mu up toward zero.
        VAE v(3, 8, 2);
        std::vector<Tensor*> params;
        configure(v, params);
        Tensor* bmu = params[3];
        for (size_t j = 0; j < bmu->cols; ++j) (*bmu)(0, j) = -0.4 - 0.1 * static_cast<double>(j);

        Tensor x(1, 3);
        x(0, 0) = 1.0; x(0, 1) = -2.0; x(0, 2) = 0.5;

        v.forward(x);
        const double fd = fd_kl_wrt_bias(v, bmu, 0, x, 1e-6);

        v.zero_grad();
        Tensor go(1, 3);
        go.fill(0.0);
        v.backward(go, 0.0);
        std::vector<Tensor*> grads = v.gradients();
        const double ana = (*grads[3])(0, 0);

        printf("    FD = %+.12f   analytical = %+.12f\n", fd, ana);
        check(fd < 0.0,
              "4.1 the finite difference is NEGATIVE for a negative mu");
        check(ana < 0.0,
              "4.2 the analytical gradient is NEGATIVE too -- gradient descent "
              "must move a negative mu UP toward zero");
        check(std::abs(ana - fd) < 1e-4,
              "4.3 and the two agree");
    }

    // =========================================================================
    // 5. A TRAINING STEP MOVES mu TOWARD ZERO
    //
    // The behavioural statement of section 4: after a step, the encoder's
    // mu output must be CLOSER to zero than it was. **THE SIGNS IN THE KL
    // GRADIENT ARE THE ONLY THING THAT DECIDES THIS**, so it is the
    // assertion that would catch a regression the sign check above
    // describes.
    // =========================================================================
    section("5. a training step moves mu toward zero");
    {
        VAE v(3, 8, 2);
        std::vector<Tensor*> params;
        configure(v, params);

        Tensor x(1, 3);
        x(0, 0) = 1.0; x(0, 1) = -2.0; x(0, 2) = 0.5;

        // The mu-bias before and after one gradient-descent step.
        v.forward(x);
        const double before = (*params[3])(0, 0);

        v.zero_grad();
        Tensor go(1, 3);
        go.fill(0.0);
        v.backward(go, 0.0);
        v.update_weights(0.5);   // a big step, so the direction is visible

        const double after = (*params[3])(0, 0);
        printf("    mu-bias before = %+.12f   after = %+.12f\n", before, after);
        check(std::abs(after) < std::abs(before),
              "5.1 one gradient-descent step moves the mu-bias CLOSER to zero");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
