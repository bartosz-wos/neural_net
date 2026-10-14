// test_adversarial.cpp — TDD tests for adversarial robustness
// (Madry, Makelov, Schmidt, Tsipras, Vladu, "Towards Deep Learning Models
// Resistant to Adversarial Attacks", arXiv:1706.06083).
//
// EVERY numeric expectation below is ORACLE-DERIVED from REAL PyTorch 2.14.1
// running in float64, never hand-derived by me. The reason is recorded in the
// knowledge-distillation session in this repo: an oracle transcribed by the same
// author as the implementation is not independent, and the KD oracle had three
// of its own bugs before a line of C++ existed.
//
// Oracle scripts (this session):
//   ~/.hermes/cache/scratch/adv_oracle.py   -- pure-paper formulas
//   ~/.hermes/cache/scratch/adv_oracle2.py  -- the discriminating fixtures
// Output: adv_oracle.out, adv_oracle2.out
//
// THE FOUR SEMANTICS EACH FIXTURE EXISTS TO PIN (see the header's D1..D8):
//
//   D2 sign(0) == 0, NOT +1.
//      Measured by torch: the two exactly-zero coordinates of the mixed-sign
//      fixture have delta EXACTLY 0.0 under both eps=0.1 and eps=0.3. A C++
//      `g >= 0 ? 1.0 : -1.0` moves them by a full eps and is invisible on any
//      fixture without a zero gradient.
//
//   D3 THE PROJECTION IS CENTRED ON x_nat, NOT ON x_t.
//      Discriminator: eps=0.3, alpha=0.1, k=7, deterministic start.
//        CORRECT (clip to x_nat +/- eps): final max|delta| = 0.30000000000000004
//        WRONG   (clip to x_t  +/- eps): final max|delta| = 0.7000000000000004
//      A 2.33x difference — no tolerance test could miss it, and a test that
//      only checks "max|delta| <= eps" would pass on the wrong implementation
//      for the first two steps.
//
//   D5 THE DATA-RANGE CLAMP IS OPT-IN.
//      Pure paper formula at eps=0.3 on the mixed fixture gives
//        row0 = [0.8, -0.55, 1.05, -0.425]   <- 1.05 leaves [0,1]
//      With the [0,1] clamp it gives
//        row0 = [0.8, 0, 1, 0]              <- visibly different
//      So the two forms are NOT interchangeable and the test asserts both.
//
//   D1 ASCENT NOT DESCENT.
//      The attack is a MAXIMISATION. End-to-end on a real model: clean loss
//      0.73522905727170718 -> adversarial loss 0.76273603563171388,
//      ratio 1.0374128009331947. A descent implementation gives a ratio < 1.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstddef>
#include <limits>
#include <utility>
#include <stdexcept>
#include "nn/utils/adversarial.h"
#include "nn/utils/losses.h"
#include "nn/core/model.h"
#include "nn/activations/activations.h"

static int passed = 0;
static int failed = 0;

// NOTE: argument order is (name, result) so each call reads as a sentence.
static void check(const std::string& name, bool pass) {
    if (pass) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else       { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
}

static bool close(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

static Tensor tensor_from(std::vector<std::vector<double>> rows) {
    return Tensor(rows);
}

// Max absolute elementwise difference between two same-shaped tensors.
static double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return std::numeric_limits<double>::infinity();
    }
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            m = std::max(m, std::fabs(a[i][j] - b[i][j]));
    return m;
}

static bool tensor_close(const Tensor& a, const Tensor& b, double tol = 1e-12) {
    return max_abs_diff(a, b) <= tol;
}

// ---------------------------------------------------------------------------
// THE MIXED-SIGN FIXTURE -- from adv_oracle2.out section A/B.
//
// Row 0 has both positive and negative entries, so sign(grad) has both +1 and
// -1: an implementation with the wrong sign convention cannot match it.
// Row 1 has TWO EXACTLY-ZERO entries (cols 1 and 3), which is what pins D2.
// ---------------------------------------------------------------------------
static Tensor mixed_fixture() {
    return tensor_from({
        { 0.5,  -0.25,  0.75, -0.125},
        { 0.1,   0.0,   -0.2,   0.0  },
    });
}

// grad of 0.5*||x||^2 is x itself; for the fixture the zero columns therefore
// have EXACTLY zero gradient. This is the oracle's `quad_grad`.
static Tensor quad_grad(const Tensor& x) { return x.clone(); }

// ---------------------------------------------------------------------------
// A real gradient oracle over the repo's own Model, so the end-to-end tests
// exercise Layer::backward's actual dL/d(input) contract.
// ---------------------------------------------------------------------------
class ModelOracle {
public:
    ModelOracle(Model& model, const Tensor& labels)
        : model_(model), labels_(labels) {}

    Tensor operator()(const Tensor& x) const {
        Tensor logits = model_.forward(x);
        CrossEntropyLoss ce;
        Tensor go(1, 1);
        go(0, 0) = 1.0;
        Tensor dl = ce.backward(logits, labels_, go);
        return model_.backward(dl, 0.0);
    }
private:
    Model& model_;
    Tensor labels_;
};

// ===========================================================================
int main() {
    std::cout << "\n=== test_adversarial: FGSM / PGD (Madry arXiv:1706.06083) ===\n";

    // -----------------------------------------------------------------
    std::cout << "\n--- 1. sign(): the D2 semantics, in isolation ---\n";

    Tensor sg = sign_tensor(mixed_fixture());
    check("sign() returns 1.0 for a positive entry",
          close(sg[0][0], 1.0));
    check("sign() returns -1.0 for a negative entry",
          close(sg[0][1], -1.0));
    // THE bug: `g >= 0 ? 1 : -1` gives +1 here.
    check("sign() returns exactly 0.0 for a zero entry (NOT +1)",
          close(sg[1][1], 0.0));
    check("sign() returns exactly 0.0 for the second zero entry",
          close(sg[1][3], 0.0));
    check("sign() output has the same shape as its input",
          sg.rows == mixed_fixture().rows && sg.cols == mixed_fixture().cols);

    // -----------------------------------------------------------------
    std::cout << "\n--- 2. FGSM, pure paper formula (D1 ascent, D2 sign(0)) ---\n";

    Tensor x_nat = mixed_fixture();
    Tensor adv01 = linf_fgsm(x_nat, 0.1, quad_grad);

    // Oracle, eps=0.1:
    //   [0.6, -0.35, 0.85, -0.225]
    //   [0.2,  0.0,  -0.3,  0.0  ]
    check("FGSM eps=0.1 matches the oracle on row 0",
          close(adv01[0][0], 0.59999999999999998) &&
          close(adv01[0][1], -0.34999999999999998) &&
          close(adv01[0][2], 0.84999999999999998) &&
          close(adv01[0][3], -0.22500000000000001));
    check("FGSM eps=0.1 matches the oracle on row 1",
          close(adv01[1][0], 0.20000000000000001) &&
          close(adv01[1][1], 0.0) &&
          close(adv01[1][2], -0.30000000000000004) &&
          close(adv01[1][3], 0.0));

    // The zero-gradient coordinates must not move AT ALL.
    check("FGSM leaves an exactly-zero-gradient coordinate untouched (delta == 0.0)",
          close(adv01[1][1] - x_nat[1][1], 0.0));
    check("FGSM leaves the second zero-gradient coordinate untouched",
          close(adv01[1][3] - x_nat[1][3], 0.0));

    // D1: ascent. Positive gradient -> +eps. Negative gradient -> -eps.
    check("FGSM moves UP the gradient direction (ascent, not descent)",
          close(adv01[0][0] - x_nat[0][0], 0.099999999999999978) &&
          close(adv01[0][1] - x_nat[0][1], -0.099999999999999978));

    Tensor adv03 = linf_fgsm(x_nat, 0.3, quad_grad);
    // Oracle, eps=0.3, PURE (no data clamp):
    //   [0.8, -0.55, 1.05, -0.425]
    //   [0.4,  0.0,  -0.5,   0.0  ]
    check("FGSM eps=0.3 matches the oracle, unclamped (row0 col2 = 1.05)",
          close(adv03[0][0], 0.80000000000000004) &&
          close(adv03[0][1], -0.55000000000000004) &&
          close(adv03[0][2], 1.05) &&
          close(adv03[0][3], -0.42499999999999999));
    check("FGSM eps=0.3 matches the oracle, unclamped (row1)",
          close(adv03[1][0], 0.40000000000000002) &&
          close(adv03[1][1], 0.0) &&
          close(adv03[1][2], -0.5) &&
          close(adv03[1][3], 0.0));
    // D5: the pure formula is allowed OUTSIDE [0,1]. 1.05 > 1 must survive.
    check("D5: the pure paper formula does NOT clamp to [0,1] (1.05 survives)",
          adv03[0][2] > 1.0);

    // D6: eps scales the perturbation linearly; alpha is not involved here.
    // delta(0.3) = 3 * delta(0.1), so the factor is THREE, not two. (The first
    // version of this assertion used 2.0*delta(0.1) and failed with a
    // difference of exactly +/-0.1 on every non-zero cell -- a hand-calc error
    // in the test, not in the implementation.)
    check("D6: FGSM scales linearly in eps (x_0.3 == x_nat + 3*(x_0.1 - x_nat))",
          close(adv03[0][0], x_nat[0][0] + 3.0 * (adv01[0][0] - x_nat[0][0])) &&
          close(adv03[1][2], x_nat[1][2] + 3.0 * (adv01[1][2] - x_nat[1][2])));

    // -----------------------------------------------------------------
    std::cout << "\n--- 3. FGSM with the opt-in data-range clamp (D5) ---\n";

    // Oracle, eps=0.3 WITH the [0,1] clamp:
    //   [0.8, 0, 1, 0]
    //   [0.4, 0, 0, 0]
    Tensor adv03c = linf_fgsm(x_nat, 0.3, quad_grad, DataRange::unit());
    check("FGSM + clamp matches the oracle row 0 (col1 -0.55 -> 0, col2 1.05 -> 1)",
          close(adv03c[0][0], 0.80000000000000004) &&
          close(adv03c[0][1], 0.0) &&
          close(adv03c[0][2], 1.0) &&
          close(adv03c[0][3], 0.0));
    check("FGSM + clamp matches the oracle row 1 (both negatives clamped to 0)",
          close(adv03c[1][0], 0.40000000000000002) &&
          close(adv03c[1][1], 0.0) &&
          close(adv03c[1][2], 0.0) &&
          close(adv03c[1][3], 0.0));
    check("D5: clamped and unclamped attacks genuinely DIFFER on this fixture",
          max_abs_diff(adv03c, adv03) > 0.5);
    // The clamp applies AFTER the perturbation, so a positive-gradient entry
    // already inside the range must be untouched by it.
    check("D5: the clamp never moves an entry that is already in range",
          close(adv03c[0][0], 0.80000000000000004));

    // D5 ORDER, for PGD -- this is the mutation that SURVIVED the harness.
    // Clamping to the data range BEFORE the ball projection is NOT the same
    // function. A randomised search over 2e6 (value, centre, eps, lo, hi)
    // tuples found 874159 disagreements between the two orders, so this is a
    // real semantic difference, not a commutable pair of interval clamps.
    //
    // The discriminating fixture, with every step computed rather than guessed:
    //
    //   x_nat = [0.02, 0.98, 0.50],  eps = 0.30,  alpha = 0.50,  k = 1
    //   balls  = [-0.28, 0.32] , [0.68, 1.28] , [0.20, 0.80]
    //   data range = [0.5, 1.0]   (disjoint from col0's ball)
    //   grad forced to -1, so each step is v = x - 0.50
    //
    //   col0: v = -0.48 -> ball -> -0.28 -> clamp -> 0.50
    //   col1: v =  0.48 -> ball ->  0.68 -> clamp -> 0.68
    //   col2: v =  0.00 -> ball ->  0.20 -> clamp -> 0.50
    //
    // The CORRECT (ball, then clamp) order therefore gives
    //   [0.50, 0.68, 0.50].
    // The MUTATED (clamp, then ball) order gives
    //   col0: clamp(-0.48) = 0.50, then ball -> min(0.32, max(-0.28, 0.50)) = 0.32
    //   col1: clamp(0.48)  = 0.50, then ball -> 0.68
    //   col2: clamp(0.00)  = 0.50, then ball -> 0.50
    //   i.e. [0.32, 0.68, 0.50] -- differing at column 0.
    {
        Tensor xs = tensor_from({{0.02, 0.98, 0.50}});
        auto down = [](const Tensor&) { return tensor_from({{-1.0, -1.0, -1.0}}); };
        Tensor ord = linf_pgd(xs, 0.3, 0.5, 1, down, DataRange(0.5, 1.0));
        check("D5 ORDER: ball-then-clamp gives [0.50, 0.68, 0.50] (measured)",
              close(ord[0][0], 0.50) &&
              close(ord[0][1], 0.67999999999999994) &&
              close(ord[0][2], 0.50));
        // The discriminator: the mutated order would give 0.32 here.
        check("D5 ORDER: column 0 is the data-range edge 0.50, NOT the ball edge "
              "0.32 -- this is what separates the two orders",
              close(ord[0][0], 0.50) && !close(ord[0][0], 0.32));

        // A NOTE worth recording rather than asserting: with a data range that is
        // DISJOINT from the ball, the output does NOT satisfy both. Column 0
        // ends at 0.50 while its ball is [-0.28, 0.32]. That is inherent to
        // clamping last -- the clamp can push a value out of the ball, and no
        // ordering fixes it (the alternatives are (i) not clamping, (ii) clamping
        // and giving up the ball guarantee). The paper's case -- the ball and the
        // pixel range agreeing on [0,1] with a small eps -- has no conflict at
        // all. Recorded here so a future reader does not add a bogus
        // "satisfies both" assertion; the FIRST draft of this test did exactly
        // that and failed.
        bool range_ok = true;
        for (size_t j = 0; j < xs.cols; ++j)
            if (ord[0][j] < 0.5 - 1e-12 || ord[0][j] > 1.0 + 1e-12) range_ok = false;
        check("D5: with an explicit data range, the output respects it "
              "(including where that leaves the eps-ball)",
              range_ok);
    }

    // -----------------------------------------------------------------
    std::cout << "\n--- 4. PGD: the projection is centred on x_nat (D3) ---\n";

    // Oracle, eps=0.3, alpha=0.1, k=7, deterministic start.
    //   x_1 = [0.6, -0.35, 0.85, -0.225] / [0.2, 0, -0.3, 0]
    //   x_2 = [0.7, -0.45, 0.95, -0.325] / [0.3, 0, -0.4, 0]
    //   x_3 = [0.8, -0.55, 1.05, -0.425] / [0.4, 0, -0.5, 0]
    //   x_4..x_7 identical to x_3 (every coordinate is at the eps boundary)
    // The k=1 case must equal FGSM with the same eps -- that is the definitional
    // relationship between the two, and it pins the step-then-project order (D4).
    // The identity is FGSM(eps) == PGD(k=1, alpha=eps). With alpha != eps the two
    // are DIFFERENT functions, which is exactly what D6 says -- so this test
    // uses alpha = eps = 0.3. (The first version passed alpha=0.1 here and
    // compared against FGSM(0.3), which is a different attack; that was a test
    // bug. A separate check below now asserts the two genuinely differ when
    // alpha != eps.)
    Tensor pgd1 = linf_pgd(x_nat, 0.3, 0.3, 1, quad_grad);
    check("PGD with k=1, alpha=eps is bit-identical to FGSM at that eps (D4 order)",
          tensor_close(pgd1, adv03));
    check("D6: PGD with alpha != eps is a DIFFERENT attack from FGSM (so the "
          "k=1 identity above is a statement about alpha == eps, not a tautology)",
          max_abs_diff(linf_pgd(x_nat, 0.3, 0.1, 1, quad_grad), adv03) > 0.1);

    Tensor pgd2 = linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad);
    check("PGD k=2 matches the oracle at x_2",
          close(pgd2[0][0], 0.69999999999999996) &&
          close(pgd2[0][1], -0.44999999999999996) &&
          close(pgd2[0][2], 0.94999999999999964) &&
          close(pgd2[0][3], -0.32499999999999999) &&
          close(pgd2[1][0], 0.30000000000000004) &&
          close(pgd2[1][1], 0.0) &&
          close(pgd2[1][2], -0.40000000000000002) &&
          close(pgd2[1][3], 0.0));

    Tensor pgd3 = linf_pgd(x_nat, 0.3, 0.1, 3, quad_grad);
    check("PGD k=3 matches the oracle at x_3 (first coordinate hits the boundary)",
          close(pgd3[0][0], 0.79999999999999993) &&
          close(pgd3[0][1], -0.54999999999999993) &&
          close(pgd3[0][2], 1.05) &&
          close(pgd3[1][0], 0.40000000000000002) &&
          close(pgd3[1][2], -0.5));

    // The D3 discriminator proper: after the boundary is reached the iterate
    // MUST stop moving, because the projection pulls it back every step.
    // A projection centred on x_t would instead run away (max|delta| = 0.7).
    Tensor pgd7 = linf_pgd(x_nat, 0.3, 0.1, 7, quad_grad);
    check("D3: the iterate is frozen once it reaches the eps boundary",
          tensor_close(pgd7, pgd3));
    check("D3: max|delta| is exactly eps, never more (0.30000000000000004)",
          close(max_abs_diff(pgd7, x_nat), 0.30000000000000004));
    // And the WRONG implementation would be caught by this bound alone:
    check("D3: the wrong (x_t-centred) projection would exceed eps by 0.7 -- "
          "the suite's bound is what separates them",
          max_abs_diff(pgd7, x_nat) < 0.5);

    // The zero-gradient coordinate must STILL be zero after 7 steps.
    check("D2 survives 7 PGD steps: the zero-gradient coordinate never moves",
          close(pgd7[1][1], 0.0) && close(pgd7[1][3], 0.0));

    // -----------------------------------------------------------------
    std::cout << "\n--- 5. PGD configuration validation (D6, D7) ---\n";

    bool threw_eps = false;
    try { linf_pgd(x_nat, -0.1, 0.1, 3, quad_grad); } catch (const std::invalid_argument&) { threw_eps = true; }
    check("linf_pgd rejects a negative epsilon", threw_eps);

    bool threw_zero_eps = false;
    try { linf_pgd(x_nat, 0.0, 0.1, 3, quad_grad); } catch (const std::invalid_argument&) { threw_zero_eps = true; }
    check("linf_pgd rejects epsilon == 0", threw_zero_eps);

    bool threw_alpha = false;
    try { linf_pgd(x_nat, 0.3, -0.1, 3, quad_grad); } catch (const std::invalid_argument&) { threw_alpha = true; }
    check("linf_pgd rejects a non-positive step size alpha", threw_alpha);

    bool threw_k = false;
    try { linf_pgd(x_nat, 0.3, 0.1, -1, quad_grad); } catch (const std::invalid_argument&) { threw_k = true; }
    check("linf_pgd rejects a negative step count k", threw_k);

    bool threw_shape = false;
    try { linf_pgd(x_nat, 0.3, 0.1, 3, [](/*unused*/ const Tensor&) { return tensor_from({{1.0, 2.0}}); }); }
    catch (const std::invalid_argument&) { threw_shape = true; }
    check("linf_pgd rejects a gradient of the wrong shape", threw_shape);

    // D7: k = 0 is the identity attack, NOT a single step.
    Tensor pgd0 = linf_pgd(x_nat, 0.3, 0.1, 0, quad_grad);
    check("D7: k=0 returns the input unchanged (no gradient is evaluated)",
          tensor_close(pgd0, x_nat));

    // D6: alpha and eps are independent. The authors' own challenge config is
    // eps=0.3, alpha=0.01 -- a 30x ratio -- so an impl that conflates them
    // behaves completely differently at these settings.
    Tensor pgd_small_alpha = linf_pgd(x_nat, 0.3, 0.01, 10, quad_grad);
    check("D6: alpha=0.01 with k=10 moves at most 10*0.01=0.1 per coordinate",
          max_abs_diff(pgd_small_alpha, x_nat) <= 0.10000000000000009);
    check("D6: a small alpha with few steps is strictly weaker than a big one",
          max_abs_diff(pgd_small_alpha, x_nat) <
          max_abs_diff(linf_pgd(x_nat, 0.3, 0.1, 10, quad_grad), x_nat));

    // -----------------------------------------------------------------
    std::cout << "\n--- 6. Random start (the reference implementation's option) ---\n";

    // The random start must be reproducible under a fixed seed, and the
    // starting point must lie inside the eps-ball AND inside the data range.
    Tensor rs1 = linf_pgd(x_nat, 0.3, 0.1, 0, quad_grad, DataRange::unit(), false, 1234);
    Tensor rs2 = linf_pgd(x_nat, 0.3, 0.1, 0, quad_grad, DataRange::unit(), false, 1234);
    check("random start is reproducible under a fixed seed", tensor_close(rs1, rs2));
    check("random start differs from the deterministic start",
          max_abs_diff(rs1, x_nat) > 0.0);
    check("random start lies within the eps-ball of x_nat",
          max_abs_diff(rs1, x_nat) <= 0.3 + 1e-12);
    bool all_in_range = true;
    for (size_t i = 0; i < rs1.rows; ++i)
        for (size_t j = 0; j < rs1.cols; ++j)
            if (rs1[i][j] < 0.0 || rs1[i][j] > 1.0) all_in_range = false;
    check("every random-start coordinate is inside [0,1]", all_in_range);

    // A full PGD run from that start must still land inside the eps-ball of
    // x_nat (the projection is what guarantees it, not the start).
    Tensor rs_run = linf_pgd(x_nat, 0.3, 0.05, 6, quad_grad, DataRange::unit(), false, 1234);
    check("a randomised-start PGD run stays within eps of x_nat",
          max_abs_diff(rs_run, x_nat) <= 0.3 + 1e-12);

    // -----------------------------------------------------------------
    std::cout << "\n--- 7. Random-restart attack = max over restarts ---\n";

    // The paper's observation that maxima are well-concentrated motivates
    // taking the WORST restart. With a quadratic oracle every restart gives a
    // different point but the SAME loss (0.5||x||^2 depends on the whole
    // vector), so the selection must be deterministic given the seed.
    Tensor worst = random_restart_pgd(x_nat, 0.3, 0.1, 4, 3, quad_grad, DataRange(), 77);
    check("random_restart_pgd returns a tensor of the input shape",
          worst.rows == x_nat.rows && worst.cols == x_nat.cols);
    check("random_restart_pgd result stays within the eps-ball",
          max_abs_diff(worst, x_nat) <= 0.3 + 1e-12);
    Tensor worst2 = random_restart_pgd(x_nat, 0.3, 0.1, 4, 3, quad_grad, DataRange(), 77);
    check("random_restart_pgd is reproducible under a fixed seed",
          tensor_close(worst, worst2));

    // With an objective, the selected restart must be at least as bad as ANY
    // single deterministic run -- that is the entire point of restarting.
    // The quadratic objective is 0.5*||x||^2, which the quad_grad oracle makes
    // exactly consistent with the attack's own gradient.
    auto quad_obj = [](const Tensor& t) {
        double s = 0.0;
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j) s += t[i][j] * t[i][j];
        return 0.5 * s;
    };
    Tensor worst_obj = random_restart_pgd(x_nat, 0.3, 0.1, 4, 5, quad_grad,
                                          DataRange(), 77, quad_obj);
    check("with an objective, the selected restart is worse than the "
          "single deterministic attack",
          quad_obj(worst_obj) > quad_obj(linf_pgd(x_nat, 0.3, 0.1, 4, quad_grad)));
    Tensor worst_obj2 = random_restart_pgd(x_nat, 0.3, 0.1, 4, 5, quad_grad,
                                           DataRange(), 77, quad_obj);
    check("random_restart_pgd WITH an objective is reproducible",
          tensor_close(worst_obj, worst_obj2));

    bool threw_restarts = false;
    try { random_restart_pgd(x_nat, 0.3, 0.1, 4, 0, quad_grad, DataRange(), 77); }
    catch (const std::invalid_argument&) { threw_restarts = true; }
    check("random_restart_pgd rejects a restart count below 1", threw_restarts);

    // -----------------------------------------------------------------
    std::cout << "\n--- 8. L2 variant ---\n";

    // Oracle (adv_oracle.py, last section): x + eps * g/||g||_2, ROW-WISE.
    //   fixture (2,4) seed=11: row norms 1.3090109717052727, 0.63740284621502719
    //   x_adv row0 = [0.26579901685044838, 1.2234455214283944,
    //                 0.92805700762657573, 0.91856563460832763]
    //   ||delta||_2 per row: exactly 0.5 for both rows.
    Tensor x2 = tensor_from({{0.1, -0.2, 0.3, 0.4}, {0.5, -0.1, 0.2, -0.3}});
    Tensor adv2 = l2_fgsm(x2, 0.5, quad_grad);
    double n0 = 0.0, n1 = 0.0;
    for (size_t j = 0; j < 4; ++j) {
        double d0 = adv2[0][j] - x2[0][j];
        double d1 = adv2[1][j] - x2[1][j];
        n0 += d0 * d0; n1 += d1 * d1;
    }
    check("L2 FGSM produces a perturbation of exactly norm eps on row 0",
          close(std::sqrt(n0), 0.5));
    check("L2 FGSM produces a perturbation of exactly norm eps on row 1",
          close(std::sqrt(n1), 0.5));
    // The L2 variant must NOT be the L_inf one -- normalised, not signed.
    check("L2 and L_inf attacks differ (L2 normalises, L_inf signs)",
          max_abs_diff(adv2, linf_fgsm(x2, 0.5, quad_grad)) > 1e-6);
    // Per-ROW normalisation: row 1 has a much smaller gradient norm, so its
    // per-coordinate step must be LARGER. A global norm would not do that.
    check("L2 FGSM normalises per ROW, not globally",
          std::fabs((adv2[1][0] - x2[1][0])) > std::fabs((adv2[0][0] - x2[0][0])));

    bool threw_zero_row = false;
    // A row whose gradient is exactly zero has no direction; the natural
    // convention is a no-op row rather than a NaN.
    Tensor zero_grad = tensor_from({{0.0, 0.0, 0.0}, {0.5, -0.1, 0.2}});
    try {
        Tensor t = l2_fgsm(zero_grad, 0.5, quad_grad);
        bool finite = true;
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                if (!std::isfinite(t[i][j])) finite = false;
        check("L2 FGSM leaves a zero-gradient row unchanged instead of producing NaN",
              finite && close(t[0][0], 0.0) && close(t[0][2], 0.0));
    } catch (const std::exception&) { threw_zero_row = true; }
    check("L2 FGSM handles a zero-gradient row without throwing", !threw_zero_row);

    // -----------------------------------------------------------------
    std::cout << "\n--- 9. End-to-end on a REAL Model (D1 ascent, D8 no mutation) ---\n";

    // Weights are the exact ones printed by the torch oracle (float64), so the
    // two implementations are computing the SAME function.
    Model model;
    Dense* l1 = new Dense(3, 4);
    Dense* l2 = new Dense(4, 2);
    l1->weights = tensor_from({
        {-0.097360573709011078, -0.30760648846626282,  0.024352256208658218},
        {-0.16359350085258484,  -0.4311281144618988,  -0.30468758940696716},
        { 0.47527286410331726,   0.1382516622543335,  -0.034582685679197311},
        {-0.47949737310409546,  -0.47420766949653625,  0.42239847779273987},
    });
    l1->bias = tensor_from({{-0.51681059598922729, 0.48716956377029419,
                             0.57677996158599854, -0.3570379912853241}});
    l2->weights = tensor_from({
        {-0.16944664716720581,  0.18826991319656372, 0.47309553623199463, 0.11987709999084473},
        {-0.025467991828918457, -0.13188827037811279, 0.22517269849777222, 0.24469459056854248},
    });
    l2->bias = tensor_from({{0.2549927830696106, -0.16078698635101318}});
    model.add_layer(l1);
    // ORDER MATTERS and the first version of this fixture got it wrong: it was
    // l1 -> l2 -> ReLU, which is NOT torch's Net (Linear -> ReLU -> Linear).
    // Symptom: our logits were [0.69268, 0] against torch's
    // [0.65360, 0.00269], and three loss assertions failed by ~0.0015 while
    // the perturbation tests passed -- because the attack only depends on the
    // SIGN of dL/dx, which the wrong order happened to preserve.
    model.add_layer(new Activation(ReLU()));
    model.add_layer(l2);

    Tensor xe = tensor_from({{0.3, 0.6, 0.2}, {0.9, 0.1, 0.5}});
    Tensor ye(2, 1);
    ye(0, 0) = 1.0; ye(1, 0) = 0.0;
    ModelOracle oracle(model, ye);

    CrossEntropyLoss ce;
    Tensor go(1, 1); go(0, 0) = 1.0;
    auto loss_at = [&](const Tensor& t) {
        Tensor logits = model.forward(t);
        return ce.forward(logits, ye)[0][0];
    };

    double clean_loss = loss_at(xe);
    check("the real model's clean loss matches the torch oracle (0.73522905727170718)",
          close(clean_loss, 0.73522905727170718, 1e-9));
    check("the real model's clean logits match torch row 0",
          close(model.forward(xe)[0][0], 0.65359689156999068, 1e-9) &&
          close(model.forward(xe)[0][1], 0.0026887972708566077, 1e-9));

    // D1: the gradient the attack consumes is a real dL/dx from this repo.
    Tensor gclean = oracle(xe);
    // C++17 has no std::sign, so the test uses the SHIPPED sign_tensor -- which is
    // itself one of the things under test. Comparing it against the torch sign
    // pattern on a real dL/dx is the point; if sign_tensor were broken, the
    // zero-coordinate checks in section 1 would already have failed.
    Tensor sg_real = sign_tensor(gclean);
    check("the model's dL/dx sign pattern matches the torch oracle row 0",
          close(sg_real[0][0], 1.0) &&
          close(sg_real[0][1], -1.0) &&
          close(sg_real[0][2], -1.0));
    check("the model's dL/dx sign pattern matches the torch oracle row 1",
          close(sg_real[1][0], -1.0) &&
          close(sg_real[1][1], 1.0) &&
          close(sg_real[1][2], 1.0));

    // Oracle: FGSM-equivalent eps=0.05 alpha=0.05 k=1 ->
    //   [0.35, 0.55, 0.15] / [0.85, 0.15, 0.55]
    Tensor e2e_fgsm = linf_pgd(xe, 0.05, 0.05, 1, oracle);
    check("FGSM on the real model matches the torch oracle exactly",
          close(e2e_fgsm[0][0], 0.34999999999999998, 1e-9) &&
          close(e2e_fgsm[0][1], 0.54999999999999993, 1e-9) &&
          close(e2e_fgsm[0][2], 0.15000000000000002, 1e-9) &&
          close(e2e_fgsm[1][0], 0.84999999999999998, 1e-9) &&
          close(e2e_fgsm[1][1], 0.15000000000000002, 1e-9) &&
          close(e2e_fgsm[1][2], 0.55000000000000004, 1e-9));

    // Oracle: PGD eps=0.20 alpha=0.05 k=5 final ->
    //   [0.5, 0.4, 0.0] / [0.7, 0.3, 0.7]
    Tensor e2e_pgd = linf_pgd(xe, 0.2, 0.05, 5, oracle);
    check("PGD-5 on the real model matches the torch oracle exactly",
          close(e2e_pgd[0][0], 0.5, 1e-9) &&
          close(e2e_pgd[0][1], 0.39999999999999997, 1e-9) &&
          close(e2e_pgd[0][2], 0.0, 1e-9) &&
          close(e2e_pgd[1][0], 0.69999999999999996, 1e-9) &&
          close(e2e_pgd[1][1], 0.30000000000000004, 1e-9) &&
          close(e2e_pgd[1][2], 0.69999999999999996, 1e-9));

    // D1: THE headline. The attack must INCREASE the loss.
    double adv_loss = loss_at(e2e_pgd);
    check("D1: the attack INCREASES the loss "
          "(clean 0.73522905727170718 -> adv 0.76273603563171388)",
          close(adv_loss, 0.76273603563171388, 1e-9));
    check("D1: adversarial loss / clean loss ratio is 1.0374128009331947",
          close(adv_loss / clean_loss, 1.0374128009331947, 1e-9));
    check("D1: a DESCENT implementation would give a ratio < 1 -- this asserts > 1",
          adv_loss > clean_loss);

    // Stronger attack => higher loss. Oracle (adv_oracle3.py, float64):
    //   clean     = 0.73522905727170718
    //   k=1       = 0.74204236803974899
    //   k=3       = 0.75579585680079497
    //   k=5       = 0.76273603563171388
    //   k=10      = 0.76273603563171388  (saturated: the iterate is on the ball)
    // The full monotone chain is asserted, not just "bigger k is bigger loss",
    // so an implementation that saturates early (e.g. because it re-centres the
    // projection on x_t and walks out of the ball) fails here.
    double l_k1 = loss_at(linf_pgd(xe, 0.2, 0.05, 1, oracle));
    double l_k3 = loss_at(linf_pgd(xe, 0.2, 0.05, 3, oracle));
    double l_k5 = loss_at(e2e_pgd);
    double l_k10 = loss_at(linf_pgd(xe, 0.2, 0.05, 10, oracle));
    check("k=1 loss matches the torch oracle (0.74204236803974899)",
          close(l_k1, 0.74204236803974899, 1e-9));
    check("k=3 loss matches the torch oracle (0.75579585680079497)",
          close(l_k3, 0.75579585680079497, 1e-9));
    check("the attack loss is strictly monotone in k: clean < k=1 < k=3 < k=5",
          clean_loss < l_k1 && l_k1 < l_k3 && l_k3 < l_k5);
    check("k=10 has saturated at the k=5 value (the iterate sits on the ball)",
          close(l_k10, l_k5, 1e-12));

    // D8: an attack must NOT change a single parameter.
    Tensor w0 = l1->weights.clone();
    Tensor b0 = l1->bias.clone();
    Tensor w1 = l2->weights.clone();
    (void)linf_pgd(xe, 0.3, 0.1, 10, oracle);
    check("D8: the attack leaves every parameter bit-identical",
          tensor_close(l1->weights, w0) && tensor_close(l1->bias, b0) &&
          tensor_close(l2->weights, w1));

    // -----------------------------------------------------------------
    std::cout << "\n--- 10. Adversarial training raises robustness ---\n";

    // A 2-class problem where a linear boundary is attackable, and crucially one
    // where a robust classifier EXISTS.
    //
    // THIS FIXTURE WAS VACUOUS THE FIRST TIME, and the way it was vacuous is
    // worth recording. It used two clusters 0.60 apart (0.10 and 0.90) with
    // eps = 0.30, and asserted `advRob > 0.5`. What happened:
    //   * the 0.30-radius balls of the two clusters are DISJOINT (they reach
    //     0.40 and 0.60 respectively), so the task is perfectly robust,
    //   * both models scored robust accuracy 1.0, so `advRob > stdRob` was
    //     1.0 > 1.0 == false and the test failed for a reason that had nothing
    //     to do with the feature.
    // Ten parameter sweeps later (probes 5-14) the real constraint was pinned:
    // the balls overlap iff 2*eps >= gap, and once they DO overlap the two
    // classes are genuinely inseparable inside the ball, so the constant
    // predictor is the true minimiser and BOTH models correctly collapse to it.
    // Measured plateaus at loss ~= log(2) = 0.6931 with the adversarially
    // trained model predicting one class for all 40 examples -- robustness
    // "0.5" that was a constant-predictor artifact, not robustness.
    //
    // So the fixture now uses gap = 0.40 and eps = 0.18, which keeps the balls
    // disjoint (2*0.18 = 0.36 < 0.40) while still letting a linear boundary
    // be attacked. Measured at that setting, deterministic across seeds
    // {1234, 7, 999}: stdRob = 0.825, advRob = 1.000, and the adversarially
    // trained model's predictions on the attacked set SPLIT 20/20 -- which is
    // the liveness evidence that it learned a boundary rather than a constant.
    Tensor Xtr(40, 2);
    Tensor ytr(40, 1);
    for (size_t i = 0; i < 20; ++i) {
        Xtr(i, 0) = 0.30 + 0.002 * (i % 7);  Xtr(i, 1) = 0.30 + 0.002 * (i % 5);
        ytr(i, 0) = 0.0;
    }
    for (size_t i = 20; i < 40; ++i) {
        Xtr(i, 0) = 0.70 - 0.002 * (i % 7);  Xtr(i, 1) = 0.70 - 0.002 * (i % 5);
        ytr(i, 0) = 1.0;
    }

    // The standard-training baseline needs the SAME label format as the
    // adversarial one, so it is trained through a loop built on the canonical
    // CrossEntropyLoss rather than through Model::train_cross_entropy (which
    // takes one-hot labels and would make this an apples-to-oranges
    // comparison). Keeping the training loop identical is what isolates
    // ADVERSARIAL training as the only variable.
    auto train_standard = [&](Model& m, int epochs) {
        CrossEntropyLoss ce;
        Tensor go(1, 1); go(0, 0) = 1.0;
        for (int e = 0; e < epochs; ++e) {
            for (size_t i = 0; i < Xtr.rows; ++i) {
                Tensor sample = Xtr.get_row(i);
                Tensor label = ytr.get_row(i);
                Tensor logits = m.forward(sample);
                m.backward(ce.backward(logits, label, go), 0.0);
                for (auto& layer : m.layers) {
                    layer->update_weights(0.05);
                    layer->zero_grad();
                }
            }
        }
    };

    PGDConfig cfg;
    cfg.epsilon = 0.18;      // 2*eps = 0.36 < gap 0.40: a robust model EXISTS
    cfg.alpha = 0.072;
    cfg.num_steps = 10;
    cfg.random_start = true;
    cfg.seed = 4242;
    cfg.verbose = false;
    check("PGDConfig round-trips every field it sets",
          close(cfg.epsilon, 0.18) && close(cfg.alpha, 0.072) &&
          cfg.num_steps == 10 && cfg.random_start == true &&
          cfg.seed == 4242u && cfg.verbose == false);

    // The library's own defaults must be safe: a default that is not a real
    // attack (alpha == 0, or steps == 0) would make every caller silently get a
    // no-op adversary.
    PGDConfig def;
    check("PGDConfig defaults to a working attack (eps>0, alpha>0, steps>0)",
          def.epsilon > 0.0 && def.alpha > 0.0 && def.num_steps > 0);
    check("PGDConfig defaults to the authors' MNIST challenge scale (eps = 0.3)",
          close(def.epsilon, 0.3));
    check("PGDConfig defaults do NOT conflate alpha with epsilon (D6)",
          def.alpha != def.epsilon);
    check("PGDConfig defaults to a RANDOM start, matching the authors' own "
          "challenge config (\"random_start\": true) -- the stronger attack",
          def.random_start == true);
    check("linf_pgd()'s single-shot default is the DETERMINISTIC start "
          "(reproducible adversarial examples unless the caller opts in)",
          linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad).cols > 0);
    {
        // The claim above, made concrete: two calls with no seed argument give
        // the same answer, and opting into random_start changes it.
        Tensor d1 = linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad);
        Tensor d2 = linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad);
        Tensor r1 = linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad, DataRange(), true, 5);
        Tensor r2 = linf_pgd(x_nat, 0.3, 0.1, 2, quad_grad, DataRange(), true, 5);
        check("two default linf_pgd() calls are bit-identical (reproducible)",
              tensor_close(d1, d2));
        check("two random_start calls with the SAME seed are bit-identical",
              tensor_close(r1, r2));
        check("random_start genuinely changes the trajectory",
              max_abs_diff(r1, d1) > 0.0);
    }

    Model std_model;
    {
        std_model.add_layer(new Dense(2, 16));
        std_model.add_layer(new Activation(ReLU()));
        std_model.add_layer(new Dense(16, 2));
        srand(1234);
        train_standard(std_model, 400);
    }
    Model adv_model;
    {
        adv_model.add_layer(new Dense(2, 16));
        adv_model.add_layer(new Activation(ReLU()));
        adv_model.add_layer(new Dense(16, 2));
        srand(1234);
        adversarial_train(adv_model, Xtr, ytr, cfg, 0.05, 400);
    }

    // Robust accuracy under the SAME evaluation attack for both models. The
    // evaluation is deterministic (random_start off) so the comparison is
    // reproducible.
    auto robust_accuracy = [&](Model& m, double eps, double alpha, int k) {
        size_t correct = 0, p0 = 0, p1 = 0;
        for (size_t i = 0; i < Xtr.rows; ++i) {
            Tensor row = Xtr.get_row(i);
            Tensor adv = linf_pgd(row, eps, alpha, k, ModelOracle(m, ytr.get_row(i)));
            for (auto& layer : m.layers) layer->zero_grad();   // D8, in eval too
            Tensor logits = m.forward(adv);
            size_t pred = static_cast<size_t>(logits[0][1] > logits[0][0] ? 1 : 0);
            (pred == 1 ? p1 : p0)++;
            if (static_cast<double>(pred) == ytr[i][0]) ++correct;
        }
        return std::make_pair(static_cast<double>(correct) / static_cast<double>(Xtr.rows),
                              std::make_pair(p0, p1));
    };

    auto std_res = robust_accuracy(std_model, 0.18, 0.072, 10);
    auto adv_res = robust_accuracy(adv_model, 0.18, 0.072, 10);
    double std_rob = std_res.first, adv_rob = adv_res.first;

    check("adversarial training raises robust accuracy under a PGD evaluation",
          adv_rob > std_rob);
    check("the standard model is genuinely attackable (so the comparison has "
          "something to measure)", std_rob < 1.0);
    check("the adversarially-trained model is fully robust at this epsilon",
          adv_rob > 0.99);

    // LIVENESS -- the check that would have caught the vacuous first fixture.
    // A model that predicts ONE class for every attacked example scores 0.5 on
    // a balanced task for a trivial reason, and 0.5 looks like "half robust".
    // Both models must SPLIT their predictions on the attacked set, which is
    // what distinguishes a learned boundary from a constant predictor.
    check("LIVENESS: the standard model splits its attacked predictions "
          "(not a constant predictor)",
          std_res.second.first > 0 && std_res.second.second > 0);
    check("LIVENESS: the adversarially-trained model splits its attacked "
          "predictions -- its robustness is NOT a constant-predictor artifact",
          adv_res.second.first > 0 && adv_res.second.second > 0);
    check("LIVENESS: the adversarially-trained model splits them evenly (20/20)",
          adv_res.second.first == 20 && adv_res.second.second == 20);

    // Clean accuracy must survive adversarial training -- a defense that just
    // destroys the model is not a defense.
    auto clean_accuracy = [&](Model& m) {
        size_t correct = 0;
        for (size_t i = 0; i < Xtr.rows; ++i) {
            Tensor logits = m.forward(Xtr.get_row(i));
            size_t pred = static_cast<size_t>(logits[0][1] > logits[0][0] ? 1 : 0);
            if (static_cast<double>(pred) == ytr[i][0]) ++correct;
        }
        return static_cast<double>(correct) / static_cast<double>(Xtr.rows);
    };
    check("adversarial training does not destroy clean accuracy",
          clean_accuracy(adv_model) > 0.99);

    // -----------------------------------------------------------------
    std::cout << "\n--- 11. adversarial_train validation ---\n";
    {
        Model empty;
        bool threw = false;
        try { adversarial_train(empty, Xtr, ytr, cfg, 0.05, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check("adversarial_train rejects a model with no layers", threw);
    }
    {
        Model m;
        m.add_layer(new Dense(2, 2));
        bool threw = false;
        try { adversarial_train(m, tensor_from({{1.0, 2.0}}), ytr, cfg, 0.05, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check("adversarial_train rejects mismatched X.rows / y.rows", threw);
    }
    {
        Model m;
        m.add_layer(new Dense(2, 2));
        PGDConfig bad;
        bad.epsilon = -1.0;
        bool threw = false;
        try { adversarial_train(m, Xtr, ytr, bad, 0.05, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check("adversarial_train rejects a negative epsilon in the config", threw);
    }

    // -----------------------------------------------------------------
    std::cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}