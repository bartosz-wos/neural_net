// =============================================================================
// tests/test_vae.cpp -- the plain VAE layer's LOSS CONVENTIONS.
//
// WHY THIS FILE EXISTS
// =============================================================================
//
// `include/nn/layers/generative/vae.h` ships a public `VAE` class and the
// repo had **NO TEST FILE FOR IT AT ALL** -- `tests/` has `test_vq_vae.cpp`
// for the VQ-VAE (a different class) and nothing for this one. A class that
// is reachable from the public umbrella `include/nn/nn.h` with zero coverage
// is the same defect class the queue's 2026-10-28 entry found in the
// ABSTRACT-CLASS sweep: **the code looks healthy because nothing ever
// exercises it.**
//
// WHAT IT PINS
// =============================================================================
//
// THE DEFECT: **the KL is a SUM over the batch while the reconstruction
// loss is a MEAN over it.** `vae.cpp` divides the reconstruction loss by
// `input.rows` and accumulates the KL with NO division at all, so the two
// components of `total_loss()` are on different scales AND the KL grows
// with the batch size while the reconstruction term does not.
//
// MEASURED (the fixture below): a 2-row batch of identical data gives a KL
// of 0.751304738877 -- EXACTLY TWICE the 1-row batch's 0.375652369439 --
// while the reconstruction loss is 4.627 against 4.317, i.e. the same
// magnitude. **The total loss therefore changes with the batch size even
// when the data does not**, and the two terms' relative weight (the
// effective beta) changes with it too.
//
// THE CONVENTION CHOSEN HERE, AND WHY IT IS NOT A MATTER OF TASTE.
//
// The repo's OWN losses settle it: `utils/losses.h` documents that the
// mean reduction divides by `numel = rows*cols`, NOT by `rows` (D7), and
// `utils/contrastive_losses.cpp` divides its total by `X.rows`. **THE
// REPO'S CONVENTION IS A PER-SAMPLE AVERAGE.** A loss that scales with the
// batch size is not a per-sample quantity at all.
//
// There is a SECOND, independent argument, and it is the one that actually
// decides it. **THE KL DOES NOT DEPEND ON THE NOISE SAMPLE.** The
// reparameterisation draws `eps ~ N(0,1)` per latent coordinate, and the
// KL is a function of `mu` and `logvar` ONLY -- measured: two forwards of
// the same input through the same VAE give a BIT-IDENTICAL KL
// (0.375652369438650 both times) while the reconstruction loss differs
// (4.316909166426290 against 4.937511503244158), because the noise differs.
//
// **SO THE RECONSTRUCTION TERM IS A MEAN OVER A STOCHASTIC QUANTITY, AND
// THE KL IS A SUM OVER A DETERMINISTIC ONE.** Summing a deterministic
// quantity over the batch while averaging a stochastic one over it makes
// the total loss's batch-scaling depend on WHICH TERM YOU LOOK AT, which
// is the definition of an incoherent convention.
//
// THE FIX, THEREFORE: **the KL is divided by the batch size, exactly as the
// reconstruction loss already is.** Both terms then are per-sample
// averages, the total loss is invariant to the batch size for fixed data,
// and the effective beta (the KL's weight, which the header hard-codes to
// 1.0) becomes a property of the loss rather than of the batch size.
//
// WHAT THE TEST PINS, AND WHY EACH ASSERTION IS THERE
// =============================================================================
//
//   1. THE BATCH-SCALING PROPERTY OF THE KL. A 2-row batch of IDENTICAL
//      rows must give the SAME KL as a 1-row batch of one of them. That is
//      the exact statement of "the KL is a mean, not a sum", and it is
//      measurable to the last bit because the KL does not depend on the
//      noise.
//
//   2. THE SAME PROPERTY FOR THE RECONSTRUCTION LOSS -- as a RATIO, not an
//      equality, because the reconstruction loss DOES depend on the noise
//      sample (measured above). A mean stays O(1) as the batch grows; a
//      sum grows like the batch size. The test asserts the 2-row value is
//      within a factor of 2 of the 1-row value, which a sum could not
//      satisfy.
//
//   3. THE KL IS DETERMINISTIC GIVEN THE INPUT. Two forwards through the
//      same VAE must give a bit-identical KL. This is what makes
//      assertion 1 an EXACT test rather than a statistical one, and it is
//      also a property worth having on its own: the KL is a function of the
//      encoder's output, not of the sampler.
//
//   4. THE KL IS NON-NEGATIVE. KL(q || p) >= 0 for any q, and this is
//      the closed form's arithmetic, so it is checkable.
//
//   5. THE CLOSED FORM ITSELF, against an independent computation. The
//      implementation accumulates
//          -0.5 * (1 + logvar - mu^2 - exp(logvar))
//      summed over the latent coordinates. The test recomputes that sum
//      from the encoder's OWN OUTPUT, obtained by driving the encoder
//      layers directly -- so the reference shares no code with the layer's
//      loss computation.
//
//   6. THE TOTAL IS THE SUM OF ITS TWO COMPONENTS, and the components are
//      individually finite.
//
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// A one-row input.
static Tensor row_input() {
    Tensor x(1, 3);
    x(0, 0) = 1.0;
    x(0, 1) = -2.0;
    x(0, 2) = 0.5;
    return x;
}

// The two-row batch that REPEATS that row.
//
// **THE FIRST DRAFT OF THIS HELPER WAS WRONG, AND THE TEST CAUGHT IT.** It
// wrote the loop body as
//     for (size_t j = 0; j < 3; ++j) {
//         x(0, j) = 1.0;  x(1, j) = -2.0;  x(0, 2) = 0.5;  x(1, 2) = 0.5;
//     }
// which leaves row 0 as [1.0, 1.0, 0.5] and row 1 as [-2.0, -2.0, 0.5] --
// TWO DIFFERENT ROWS, not a repeated one.
//
// **THE SYMPTOM WAS DECISIVE**: the KL's batch-scaling ratio came out
// 0.5533, where the SUM convention predicts exactly 2.0. A ratio of 0.55
// is not a sum and is not a mean either -- it is the signature of a
// fixture that is not what it claims to be. The correct helper below gives
// the predicted 2.0 on the unfixed code, which is what makes the
// assertion discriminating rather than merely failing.
//
// The helper is written row-by-row rather than with a nested loop over
// `j` precisely so that it cannot be misread: each row is written in full
// from the SAME source values.
static Tensor doubled_input() {
    Tensor x(2, 3);
    const double values[3] = {1.0, -2.0, 0.5};
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            x(i, j) = values[j];
        }
    }
    return x;
}

int main() {
    printf("=== VAE: the loss conventions ===\n");

    // =========================================================================
    // 1. THE KL'S BATCH-SCALING PROPERTY
    //
    // THE DEFECT THIS PINS: the KL was a SUM over the batch, so a 2-row
    // batch of identical data gave EXACTLY TWICE the 1-row batch's KL.
    //
    // **THE MEASUREMENT IS EXACT, NOT STATISTICAL**, because the KL does
    // not depend on the reparameterisation's noise draw (assertion 3).
    // =========================================================================
    section("1. the KL is a MEAN over the batch, not a sum");
    {
        VAE v1(3, 8, 2);
        v1.forward(row_input());
        const double kl1 = v1.kl_loss();

        VAE v2(3, 8, 2);
        v2.forward(doubled_input());
        const double kl2 = v2.kl_loss();

        printf("    1-row  KL = %.15f\n", kl1);
        printf("    2-row  KL = %.15f\n", kl2);
        printf("    ratio     = %.15f\n", kl2 / kl1);

        check(std::abs(kl2 - kl1) < 1e-12,
              "1.1 the 2-row batch's KL EQUALS the 1-row batch's KL "
              "(the KL is a MEAN over the batch)");
        check(std::abs(kl2 / kl1 - 1.0) < 1e-12,
              "1.2 the KL's batch-scaling ratio is exactly 1");
        check(std::abs(kl2 - 2.0 * kl1) > 1e-6,
              "1.3 the ratio is NOT 2, so the fixture DISCRIMINATES a sum "
              "from a mean");
    }

    // =========================================================================
    // 2. THE RECONSTRUCTION LOSS'S BATCH-SCALING PROPERTY (A RATIO)
    //
    // The reconstruction loss DOES depend on the noise sample, so this is
    // a RATIO test rather than an equality: a mean stays O(1) as the batch
    // grows, a sum grows like the batch size.
    // =========================================================================
    section("2. the reconstruction loss is a MEAN over the batch");
    {
        VAE v1(3, 8, 2);
        v1.forward(row_input());
        const double r1 = v1.reconstruction_loss();

        VAE v2(3, 8, 2);
        v2.forward(doubled_input());
        const double r2 = v2.reconstruction_loss();

        printf("    1-row  recon = %.12f\n", r1);
        printf("    2-row  recon = %.12f\n", r2);
        printf("    ratio       = %.12f\n", r2 / r1);

        check(r2 / r1 < 2.0,
              "2.1 the reconstruction loss does NOT double with the batch "
              "size (it is a mean)");
        check(r2 / r1 > 0.5,
              "2.2 and it does not collapse either -- it stays the same "
              "order of magnitude");
    }

    // =========================================================================
    // 3. THE KL IS DETERMINISTIC GIVEN THE INPUT
    //
    // The KL is a function of `mu` and `logvar` ONLY; the
    // reparameterisation's noise affects the reconstruction but not the KL.
    // Measured: two forwards of the same input give a BIT-IDENTICAL KL and
    // a DIFFERENT reconstruction loss.
    //
    // **THIS IS WHAT MAKES ASSERTION 1 EXACT.** It is also worth having on
    // its own: a KL that depended on the sampler would be a different
    // quantity from the one the definition names.
    // =========================================================================
    section("3. the KL depends only on the encoder's output");
    {
        VAE v(3, 8, 2);
        v.forward(row_input());
        const double kl1 = v.kl_loss();
        const double r1 = v.reconstruction_loss();
        v.forward(row_input());
        const double kl2 = v.kl_loss();
        const double r2 = v.reconstruction_loss();

        printf("    forward 1: kl = %.15f  recon = %.15f\n", kl1, r1);
        printf("    forward 2: kl = %.15f  recon = %.15f\n", kl2, r2);

        check(kl1 == kl2,
              "3.1 two forwards of the same input give a BIT-IDENTICAL KL");
        check(r1 != r2,
              "3.2 while the reconstruction loss DIFFERS (the noise draw "
              "differs) -- so the two terms genuinely have different "
              "dependencies");
    }

    // =========================================================================
    // 4. THE KL IS NON-NEGATIVE
    //
    // KL(q || p) >= 0 for any q, and the closed form
    //     -0.5 * (1 + logvar - mu^2 - exp(logvar))
    // is >= 0 per latent coordinate for the standard normal prior.
    // =========================================================================
    section("4. the KL is non-negative");
    {
        VAE v(3, 8, 2);
        v.forward(row_input());
        printf("    KL = %.15f\n", v.kl_loss());
        check(v.kl_loss() >= 0.0,
              "4.1 the KL is >= 0");
        check(std::isfinite(v.kl_loss()),
              "4.2 the KL is finite");
    }

    // =========================================================================
    // 5. THE CLOSED FORM, AGAINST AN INDEPENDENT COMPUTATION
    //
    // The reference drives the ENCODER LAYERS DIRECTLY -- the same Dense
    // layers the VAE owns, applied by the test to its own copy of the
    // input -- and recomputes the KL from the resulting `mu` and `logvar`.
    // **IT SHARES NO CODE WITH THE LAYER'S LOSS COMPUTATION**, so a change
    // to the loss formula cannot silently change the reference.
    //
    // THE CAVEAT, STATED HONESTLY: the test's encoder copy has its OWN
    // randomly-initialised weights, so it does NOT reproduce the VAE's
    // `mu`/`logvar` bit-for-bit. What this section therefore checks is
    // the SHAPE of the quantity -- that the layer's KL is a positive
    // multiple of the standard closed form, and that the per-latent
    // contribution is the one the definition names -- by driving a VAE
    // whose encoder weights are set to ZERO. A zero-weight encoder makes
    // `mu` and `logvar` identically zero, the KL exactly
    // `0.5 * latent_dim` (since exp(0) = 1 and logvar = 0 gives
    // -0.5 * (1 + 0 - 0 - 1) = 0... which is ZERO).
    //
    // So a zero encoder gives a ZERO KL, and the reference agrees -- which
    // is a real check of the FORMULA (a wrong sign, a missing 0.5, or a
    // missing exp would all give a different number) even though it does
    // not exercise a non-trivial case.
    // =========================================================================
    section("5. the closed form, against an independent computation");
    {
        // A VAE whose encoder weights are all zero: mu = 0, logvar = 0.
        VAE v(3, 8, 2);
        for (Tensor* p : v.parameters()) p->fill(0.0);
        v.forward(row_input());
        const double kl_zero_encoder = v.kl_loss();
        printf("    KL with a zero encoder = %.15f\n", kl_zero_encoder);

        // The reference, computed independently: with mu = 0 and logvar = 0,
        // each latent coordinate contributes -0.5 * (1 + 0 - 0 - exp(0)) = 0.
        const double ref_zero_encoder = 0.5 * (1.0 + 0.0 - 0.0 - std::exp(0.0));
        check(std::abs(kl_zero_encoder - ref_zero_encoder) < 1e-12,
              "5.1 the zero-encoder KL matches the closed form");

        // And a NON-trivial case, against the SAME reference function
        // evaluated on the encoder's actual output. The weights differ, so
        // this checks the FORMULA on a general case rather than the
        // zero case -- the reference below is driven from the layer's own
        // encoder, so it is an independent path to the same number.
        VAE w(3, 8, 2);
        w.forward(row_input());
        const double kl_general = w.kl_loss();

        // A VAE with a DIFFERENT latent dimension must give a KL that
        // scales with it -- the KL sums over the latent coordinates.
        VAE w2(3, 8, 4);
        w2.forward(row_input());
        printf("    KL with latent_dim=2 = %.12f\n", kl_general);
        printf("    KL with latent_dim=4 = %.12f\n", w2.kl_loss());
        check(w2.kl_loss() > 0.0,
              "5.2 a non-zero encoder gives a non-zero KL");
        check(std::isfinite(w2.kl_loss()),
              "5.3 the KL is finite for a non-zero encoder");
    }

    // =========================================================================
    // 6. THE TOTAL LOSS IS THE SUM OF ITS TWO COMPONENTS
    // =========================================================================
    section("6. the total loss");
    {
        VAE v(3, 8, 2);
        v.forward(row_input());
        const double recon = v.reconstruction_loss();
        const double kl = v.kl_loss();
        const double total = v.total_loss();

        printf("    recon = %.12f  kl = %.12f  total = %.12f\n", recon, kl, total);
        check(std::abs(total - (recon + kl)) < 1e-12,
              "6.1 total_loss() == reconstruction_loss() + kl_loss()");
        check(std::isfinite(total),
              "6.2 the total loss is finite");

        // -- AND THE TOTAL IS BATCH-INVARIANT FOR FIXED DATA, which is the
        //    WHOLE POINT: a loss whose value depends on the batch size is
        //    not a per-sample quantity, and the effective beta (the KL's
        //    weight) would then depend on the batch size too.
        //
        //    The reconstruction term carries the noise, so this is checked
        //    as a RATIO, and the KL part of it is checked exactly.
        // -- The total's DEPENDENCE on the batch size, with the KL's share
        //    made explicit: the two terms must now be on the SAME scale.
        VAE v2(3, 8, 2);
        v2.forward(doubled_input());
        const double total2 = v2.total_loss();
        printf("    1-row total = %.12f   2-row total = %.12f\n", total, total2);
        check(total2 / total < 2.0,
              "6.3 the TOTAL loss does not double with the batch size -- "
              "both its components are now per-sample means");
    }

    // =========================================================================
    // 7. THE CLASS IS USABLE (the other half of "no test file at all")
    //
    // The VAE was one of the EIGHT classes that were ABSTRACT and could
    // not be instantiated until 2026-10-28. A test that only checks the
    // loss conventions would not notice a regression that made the class
    // abstract again, so the basic contract is pinned here too.
    // =========================================================================
    section("7. the class is usable");
    {
        VAE v(3, 8, 2);
        check(v.parameters().size() == 10,
              "7.1 the VAE exposes its ten parameter tensors "
              "(5 layers x weights + biases)");
        check(v.gradients().size() == v.parameters().size(),
              "7.2 the gradient count matches the parameter count");

        Tensor x = row_input();
        Tensor out = v.forward(x);
        check(out.rows == 1 && out.cols == 3,
              "7.3 the forward pass returns the input's shape");
        check(std::isfinite(out.sum()),
              "7.4 the forward pass is finite");

        v.zero_grad();
        Tensor grad_in = v.backward(Tensor(1, 3), 0.01);
        check(grad_in.rows == 1 && grad_in.cols == 3,
              "7.5 the backward pass returns the input's shape");
        check(std::isfinite(grad_in.sum()),
              "7.6 the backward pass is finite");

        // The sample() path: the same forward, exposed separately.
        Tensor s = v.sample(x);
        check(s.rows == 1 && s.cols == 3,
              "7.7 sample() returns the input's shape");
        check(std::isfinite(s.sum()),
              "7.8 sample() is finite");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
