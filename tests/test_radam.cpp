// RAdam optimizer tests — Liu et al. 2020 "On the Variance of the Adaptive
// Learning Rate and Beyond" (https://arxiv.org/abs/1908.03265)
//
// RAdam signature update (per parameter):
//   m_t = β1·m_{t-1} + (1-β1)·g_t
//   v_t = β2·v_{t-1} + (1-β2)·g_t²
//   m̂_t = m_t / (1-β1^t)
//
//   ρ_∞ = 2/(1-β2) − 1
//   ρ_t = ρ_∞ − 2·t·β2^t / (1-β2^t)
//
//   if ρ_t > 4:
//       l_t = sqrt(((ρ_t-4)(ρ_t-2)ρ_∞) / ((ρ_∞-4)(ρ_∞-2)ρ_t))
//       param = param − lr·(l_t·m̂_t / (sqrt(v̂_t)+ε) + wd·param)
//   else:
//       param = param − lr·(m̂_t + wd·param)
//
// The CORE tests below validate:
//  (1)  Constructor defaults + accessors
//  (2)  Validation throws (β1/β2 ∈ (0,1), ε > 0)
//  (3)  State shape + lazy-init per layer
//  (4)  t increments correctly
//  (5)  ρ_∞ formula matches paper
//  (6)  ρ_t formula matches paper (exact closed-form for t=1..5, β2=0.999)
//  (7)  Early-step branch (ρ_t ≤ 4): SGD-like update — NO v denominator
//  (8)  Late-step branch (ρ_t > 4): adaptive update with l_t rectification
//  (9)  l_t formula hand-verified at t=5, β2=0.999
//  (10) Weight decay (AdamW-style): shrinks param at zero gradient
//  (11) Determinism: two fresh instances produce bit-identical params
//  (12) End-to-end: 60-step training on linear regression reduces MSE loss
//  (13) RAdam-vs-Adam signature: at t=1 with same grad, RAdam takes SGD-like
//       branch (param move = lr·m̂), Adam uses adaptive denom (lr·m̂/sqrt(v̂)+ε)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <random>

#include "nn/optimizers/radam.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"

using namespace std;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "  [PASS] " << msg << endl;                                \
            ++passed;                                                          \
        } else {                                                               \
            cout << "  [FAIL] " << msg << endl;                                \
        }                                                                      \
        ++total;                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                            \
    do {                                                                       \
        double aa = (a), bb = (b);                                             \
        bool ok = std::isfinite(aa) && std::isfinite(bb)                       \
                  && std::abs(aa - bb) <= (tol);                               \
        if (ok) {                                                              \
            cout << "  [PASS] " << msg << endl;                                \
            ++passed;                                                          \
        } else {                                                               \
            cout << "  [FAIL] " << msg << "  (got=" << aa                      \
                 << " want=" << bb << " diff=" << std::abs(aa-bb)              \
                 << " tol=" << tol << ")" << endl;                             \
        }                                                                      \
        ++total;                                                               \
    } while (0)

static double rel_err(double a, double b) {
    double m = std::max(std::abs(a), std::abs(b));
    if (m < 1e-12) return std::abs(a - b);
    return std::abs(a - b) / m;
}

struct Owned {
    Model* model;
    Dense* d;
};
static Owned make_owned_dense(int in_dim, int out_dim) {
    auto* m = new Model();
    auto* d = new Dense(in_dim, out_dim);
    m->add_layer(d);
    return {m, d};
}

int main() {
    cout << "=== RAdam Optimizer Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // ====================================================================
    // SECTION 1: Constructor defaults + accessors
    // ====================================================================
    {
        cout << "[1] Defaults + accessors" << endl;
        RAdam r;
        CHECK(r.get_lr()           == 1e-3,  "default lr = 1e-3");
        CHECK(r.get_beta1()        == 0.9,   "default beta1 = 0.9");
        CHECK(r.get_beta2()        == 0.999, "default beta2 = 0.999");
        CHECK(r.get_epsilon()      == 1e-8,  "default epsilon = 1e-8");
        CHECK(r.get_weight_decay() == 0.0,   "default weight_decay = 0");
        CHECK(r.get_t()            == 1,     "default t = 1");

        RAdam r2(0.05, 0.85, 0.95, 1e-6, 0.01);
        CHECK(r2.get_lr() == 0.05 && r2.get_beta1() == 0.85 && r2.get_beta2() == 0.95
              && r2.get_epsilon() == 1e-6 && r2.get_weight_decay() == 0.01,
              "non-default constructor args");
    }

    // ====================================================================
    // SECTION 2: Validation throws
    // ====================================================================
    {
        cout << "[2] Validation throws" << endl;
        {
            bool threw = false;
            try { RAdam r(0.1, 0.0, 0.9, 1e-8); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "beta1=0 throws");
        }
        {
            bool threw = false;
            try { RAdam r(0.1, 1.1, 0.9, 1e-8); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "beta1>=1 throws");
        }
        {
            bool threw = false;
            try { RAdam r(0.1, 0.9, 0.0, 1e-8); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "beta2=0 throws");
        }
        {
            bool threw = false;
            try { RAdam r(0.1, 0.9, 1.0, 1e-8); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "beta2=1 throws");
        }
        {
            bool threw = false;
            try { RAdam r(0.1, 0.9, 0.9, 0.0); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "epsilon=0 throws");
        }
        {
            bool threw = false;
            try { RAdam r(0.1, 0.9, 0.9, -1e-8); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "negative epsilon throws");
        }
        {
            // Mutator validation
            bool threw = false;
            RAdam r;
            try { r.set_beta2(2.0); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "set_beta2>=1 throws");
        }
        {
            bool threw = false;
            RAdam r;
            try { r.set_epsilon(-1.0); }
            catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "set_epsilon<=0 throws");
        }
    }

    // ====================================================================
    // SECTION 3: State lazy-init
    // ====================================================================
    {
        cout << "[3] State shape + lazy-init" << endl;
        Owned ow = make_owned_dense(3, 2);
        RAdam r(0.001, 0.9, 0.999, 1e-8, 0.0);

        // Before any step: get_m/get_v return empty tensors
        Tensor m_empty = r.get_m(ow.d, 0);
        Tensor v_empty = r.get_v(ow.d, 0);
        CHECK(m_empty.rows == 0 && m_empty.cols == 0, "get_m returns (0,0) before step");
        CHECK(v_empty.rows == 0 && v_empty.cols == 0, "get_v returns (0,0) before step");

        // Inject a fixed gradient and step
        ow.d->grad_weights(0, 0) = 1.0;
        ow.d->grad_bias(0, 0)    = 0.5;
        r.step(*ow.model);

        // After step: m/v have correct shape (matches param shape)
        // Dense(3, 2) → weights shape (out=2, in=3) → (2, 3)
        Tensor m_w = r.get_m(ow.d, 0);  // weights is (2, 3)
        Tensor v_w = r.get_v(ow.d, 0);
        Tensor m_b = r.get_m(ow.d, 1);  // bias is (1, 2)
        Tensor v_b = r.get_v(ow.d, 1);
        CHECK(m_w.rows == 2 && m_w.cols == 3, "m of weights has shape (2, 3)");
        CHECK(v_w.rows == 2 && v_w.cols == 3, "v of weights has shape (2, 3)");
        CHECK(m_b.rows == 1 && m_b.cols == 2, "m of bias has shape (1, 2)");
        CHECK(v_b.rows == 1 && v_b.cols == 2, "v of bias has shape (1, 2)");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 4: t increments
    // ====================================================================
    {
        cout << "[4] t increments" << endl;
        Owned ow = make_owned_dense(2, 1);
        ow.d->grad_weights(0, 0) = 1.0;

        RAdam r(0.01, 0.9, 0.9, 1e-8, 0.0);
        CHECK(r.get_t() == 1, "t starts at 1");
        r.step(*ow.model);
        CHECK(r.get_t() == 2, "t = 2 after step 1");
        r.step(*ow.model);
        CHECK(r.get_t() == 3, "t = 3 after step 2");
        r.step(*ow.model);
        r.step(*ow.model);
        r.step(*ow.model);
        CHECK(r.get_t() == 6, "t = 6 after 5 total steps");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 5: ρ_∞ formula
    // ρ_∞ = 2/(1-β2) − 1
    // ====================================================================
    {
        cout << "[5] rho_infty formula" << endl;
        Owned ow = make_owned_dense(1, 1);
        RAdam r(0.001, 0.9, 0.999, 1e-8, 0.0);

        // β2 = 0.999 → ρ_∞ = 2/0.001 − 1 = 1999
        CHECK_NEAR(r.get_rho_infty(), 1999.0, 1e-9, "rho_infty(β2=0.999) = 1999");

        // Test mutator triggers recompute
        r.set_beta2(0.9);
        // β2 = 0.9 → ρ_∞ = 2/0.1 − 1 = 19
        CHECK_NEAR(r.get_rho_infty(), 19.0, 1e-9, "rho_infty(β2=0.9) = 19 after set_beta2");

        // Test β2 = 0.5 → ρ_∞ = 2/0.5 − 1 = 3
        r.set_beta2(0.5);
        CHECK_NEAR(r.get_rho_infty(), 3.0, 1e-9, "rho_infty(β2=0.5) = 3");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 6: ρ_t formula
    // ρ_t = ρ_∞ − 2·t·β2^t / (1 − β2^t)
    //
    // β2=0.999, ρ_∞=1999:
    //   t=1: ρ_1 = 1999 − 2·1·0.999/(1−0.999) = 1999 − 1998 = 1.0  (exact)
    //   t=2: ρ_2 = 1999 − 2·2·0.998001/(1−0.998001) ≈ 1999 − 1997.001 ≈ 1.999
    //   t=3: ρ_3 ≈ 2.998
    //   t=4: ρ_4 ≈ 3.998
    //   t=5: ρ_5 ≈ 4.996  (first time > 4 → adaptive branch)
    //
    // Series: ρ_t - 1 grows ~linearly from t=1; the gap to ρ_∞ is huge in the
    // warmup phase. Tolerances are tight on t=1 (exact) and loosen for larger t.
    // ====================================================================
        {
            cout << "[6] rho_t formula" << endl;
            Owned ow = make_owned_dense(1, 1);
            RAdam r(0.001, 0.9, 0.999, 1e-8, 0.0);

            CHECK_NEAR(r.get_rho_t(1), 1.0,  1e-9, "rho_t(β2=0.999, t=1) = 1.0");
            CHECK_NEAR(r.get_rho_t(2), 1.999, 1e-3, "rho_t(β2=0.999, t=2) ≈ 1.999");
            CHECK_NEAR(r.get_rho_t(3), 2.998, 1e-3, "rho_t(β2=0.999, t=3) ≈ 2.998");
            CHECK_NEAR(r.get_rho_t(4), 3.998, 1e-3, "rho_t(β2=0.999, t=4) ≈ 3.998");
            CHECK_NEAR(r.get_rho_t(5), 4.996, 1e-3, "rho_t(β2=0.999, t=5) ≈ 4.996");

            // At t=1 with β2=0.999, ρ_t = 1 < 4 → SGD-like branch
            CHECK(r.get_rho_t(1) <= 4.0, "rho_t(β2=0.999, t=1) is in SGD-like branch");
            // At t=5 with β2=0.999, ρ_t ≈ 4.996 > 4 → adaptive branch
            CHECK(r.get_rho_t(5) > 4.0,  "rho_t(β2=0.999, t=5) is in adaptive branch");

            delete ow.model;
        }

    // ====================================================================
    // SECTION 7: Early-step branch (ρ_t ≤ 4): SGD-like update
    //
    // At t=1, β1=0.9, β2=0.999 (so ρ_1=1 < 4 → SGD-like branch):
    //   m_new = 0.9·0 + 0.1·g = 0.1·g
    //   m_hat = 0.1·g / (1−0.9) = g
    //   param_new = param − lr·g   (NO v denominator)
    //
    // For g=2, lr=0.1: param move = −0.2
    // ====================================================================
    {
        cout << "[7] Early-step SGD-like branch (rho_t <= 4)" << endl;
        Owned ow = make_owned_dense(1, 1);
        ow.d->weights.fill(1.0);
        ow.d->bias.fill(0.0);
        ow.d->grad_weights(0, 0) = 2.0;
        ow.d->grad_bias(0, 0)    = 0.0;  // only test weights

        RAdam r(0.1, 0.9, 0.999, 1e-8, 0.0);  // lr=0.1, β2=0.999 → t=1 is SGD-like
        r.step(*ow.model);

        // Expected: weights[0][0] = 1.0 − 0.1 · 2.0 = 0.8
        CHECK_NEAR(ow.d->weights[0][0], 0.8, 1e-12,
                   "t=1 SGD-like: param move = -lr·m̂ = -0.1·2.0 = -0.2 (so weights[0][0] = 0.8)");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 8: Late-step branch (ρ_t > 4): adaptive update with l_t
    //
    // Strategy: drive t up to ~5 with β2=0.9 so ρ_t > 4 there, then test.
    // β2=0.9, t=5: ρ_∞=19, ρ_5 = 19 − 2·5·0.9⁵/(1−0.9⁵) = 19 − 5.9052/0.40951 = 19 − 14.4205 = 4.5795
    // > 4 → adaptive branch
    //
    // At t=5, β1=0.9, β2=0.9, ε=1e-8, wd=0:
    //   With grad=2, m_prev=0:
    //     m_new = 0.1·2 = 0.2
    //     v_new = 0.1·4 = 0.4   (after 5 steps, accumulated from prior g²=4)
    //   bias-correction: b1_c = 1−0.9⁵ = 0.40951, b2_c = 1−0.9⁵ = 0.40951
    //     m_hat = 0.2/0.40951 = 0.4884
    //     v_hat = 0.4/0.40951 = 0.9768 (after just 1 step the state is from g²=4)
    //   l_5 ≈ sqrt(((4.5795−4)(4.5795−2)·19) / ((19−4)(19−2)·4.5795))
    //        = sqrt((0.5795 · 2.5795 · 19) / (15 · 17 · 4.5795))
    //        = sqrt(28.4043 / 1167.76)
    //        = sqrt(0.02432)
    //        = 0.156
    //   denom = sqrt(0.9768) + 1e-8 ≈ 0.9883
    //   step = l_t · m̂ / denom = 0.156 · 0.4884 / 0.9883 ≈ 0.0771
    //   param move = -lr · 0.0771
    //
    // To make this deterministic, do 5 steps first to enter the adaptive branch,
    // then a 6th step that we can compute exactly. Actually let's make t=5 the
    // "current" step (after 4 prior steps).
    // ====================================================================
    {
        cout << "[8] Late-step adaptive branch (rho_t > 4)" << endl;
        Owned ow = make_owned_dense(1, 1);
        ow.d->weights.fill(1.0);
        ow.d->bias.fill(0.0);
        // Initialize the state by doing 4 steps with zero grad (m stays 0, v stays 0)
        ow.d->grad_weights.fill(0.0);
        ow.d->grad_bias.fill(0.0);

        RAdam r(0.1, 0.9, 0.9, 1e-8, 0.0);  // β2=0.9 so ρ_5 > 4
        r.step(*ow.model);  // t=1: ρ_1 = 19 − 2·1·0.9/0.1 = 19 − 18 = 1, SGD-like
        r.step(*ow.model);  // t=2: ρ_2 = 19 − 2·2·0.81/0.19 = 19 − 17.0526 = 1.947, SGD-like
        r.step(*ow.model);  // t=3: ρ_3 = 19 − 2·3·0.729/0.271 = 19 − 16.1402 = 2.860, SGD-like
        r.step(*ow.model);  // t=4: ρ_4 = 19 − 2·4·0.6561/0.3439 = 19 − 15.265 = 3.735, SGD-like
        // t=5: ρ_5 ≈ 4.58 → adaptive branch!

        // Inject a non-zero grad and step
        double w_before = ow.d->weights[0][0];
        ow.d->grad_weights(0, 0) = 2.0;
        r.step(*ow.model);
        double w_after = ow.d->weights[0][0];

        // Verify the move is non-zero (proves adaptive branch ran)
        CHECK(std::abs(w_after - w_before) > 1e-6,
              "t=5 adaptive branch: weights change after grad injection");

        // The move should be different from a pure SGD-momentum step would be.
        // At t=5, β1=0.9, β2=0.9, m_prev=0, grad=2:
        //   m_new = 0.2, b1_c = 1−0.9⁵ = 0.40951, m̂ = 0.4884
        //   v_new = 0.4 (after only this step), b2_c = 0.40951, v̂ = 0.9768
        //   l_5 = sqrt(((ρ_t−4)(ρ_t−2)ρ_∞) / ((ρ_∞−4)(ρ_∞−2)ρ_t))
        //        = sqrt(((0.5795)(2.5795)(19)) / ((15)(17)(4.5795)))
        //        = sqrt(28.4043 / 1167.76) = sqrt(0.024325) ≈ 0.15597
        //   denom = sqrt(0.9768) + 1e-8 ≈ 0.98835
        //   step = l_5 · m̂ / denom = 0.15597 · 0.4884 / 0.98835 ≈ 0.07706
        //   param move = -lr · step = -0.1 · 0.07706 ≈ -0.007706
        double expected_step = 0.15597 * (0.2 / 0.40951) / (std::sqrt(0.4 / 0.40951) + 1e-8);
        double expected_w_after = w_before - 0.1 * expected_step;
        CHECK_NEAR(w_after, expected_w_after, 1e-3,
                   "t=5 adaptive branch: weights move matches analytical l_t formula");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 9: l_t formula hand-verified
    // ====================================================================
    {
        cout << "[9] l_t formula" << endl;
        Owned ow = make_owned_dense(1, 1);
        RAdam r(0.001, 0.9, 0.999, 1e-8, 0.0);

        // At t=5, β2=0.999: ρ_t ≈ 7.986 (computed in section 6)
        double rho_t = r.get_rho_t(5);
        double rho_inf = r.get_rho_infty();
        double l_t = r.get_l_t(5);
        // l_t = sqrt(((ρ_t-4)(ρ_t-2)ρ_∞) / ((ρ_∞-4)(ρ_∞-2)ρ_t))
        double expected = std::sqrt(
            ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) /
            ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
        );
        CHECK_NEAR(l_t, expected, 1e-9, "l_t formula matches hand-derived expression");

        // Sanity: l_t must be ≤ 1 always (RAdam bounds the variance amplification)
        // At t=5 with β2=0.999, ρ_t > ρ_∞/2 → l_t < 1
        CHECK(l_t <= 1.0, "l_t <= 1 (variance bound)");

        // Also: at large t (ρ_t → ρ_∞), l_t → 1
        CHECK_NEAR(r.get_l_t(10000), 1.0, 1e-3, "l_t → 1 as t → ∞");

        // For ρ_t ≤ 4, l_t is NOT used — get_l_t is still callable but reflects
        // the formula. Check the boundary:
        CHECK(r.get_rho_t(1) <= 4.0, "at t=1 we're below the boundary (ρ_t=1)");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 10: Weight decay
    // ====================================================================
    {
        cout << "[10] Weight decay" << endl;
        Owned ow = make_owned_dense(1, 1);
        ow.d->weights.fill(1.0);
        ow.d->bias.fill(0.0);
        ow.d->grad_weights.fill(0.0);  // zero grad
        ow.d->grad_bias.fill(0.0);

        // At t=1, ρ_1 ≤ 4 → SGD-like branch:
        //   param_new = param − lr·(m̂ + wd·param)
        //   m_new = 0.9·0 + 0.1·0 = 0, m̂ = 0/(1−0.9) = 0
        //   param_new = 1.0 − 0.01·(0 + 0.1·1.0) = 1.0 − 0.001 = 0.999
        RAdam r(0.01, 0.9, 0.999, 1e-8, 0.1);  // wd=0.1
        r.step(*ow.model);

        CHECK_NEAR(ow.d->weights[0][0], 0.999, 1e-9,
                   "weight decay shrinks param at zero grad (SGD-like branch)");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 11: Determinism
    // ====================================================================
    {
        cout << "[11] Determinism" << endl;
        // Train two fresh RAdam instances on the same grad sequence; params match
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);

        Owned ow1 = make_owned_dense(3, 2);
        Owned ow2 = make_owned_dense(3, 2);
        // Copy weights exactly
        for (size_t i = 0; i < ow1.d->weights.rows; ++i)
            for (size_t j = 0; j < ow1.d->weights.cols; ++j)
                ow2.d->weights[i][j] = ow1.d->weights[i][j];
        for (size_t j = 0; j < ow1.d->bias.cols; ++j)
            ow2.d->bias[0][j] = ow1.d->bias[0][j];

        RAdam r1(0.001, 0.9, 0.999, 1e-8, 0.0);
        RAdam r2(0.001, 0.9, 0.999, 1e-8, 0.0);

        for (int step = 0; step < 30; ++step) {
            // Same random grad for both
            for (size_t i = 0; i < ow1.d->grad_weights.rows; ++i)
                for (size_t j = 0; j < ow1.d->grad_weights.cols; ++j) {
                    double g = dist(rng);
                    ow1.d->grad_weights[i][j] = g;
                    ow2.d->grad_weights[i][j] = g;
                }
            for (size_t j = 0; j < ow1.d->grad_bias.cols; ++j) {
                double g = dist(rng);
                ow1.d->grad_bias[0][j] = g;
                ow2.d->grad_bias[0][j] = g;
            }
            r1.step(*ow1.model);
            r2.step(*ow2.model);
        }

        double max_rel = 0.0;
        for (size_t i = 0; i < ow1.d->weights.rows; ++i)
            for (size_t j = 0; j < ow1.d->weights.cols; ++j)
                max_rel = std::max(max_rel, rel_err(ow1.d->weights[i][j], ow2.d->weights[i][j]));
        for (size_t j = 0; j < ow1.d->bias.cols; ++j)
            max_rel = std::max(max_rel, rel_err(ow1.d->bias[0][j], ow2.d->bias[0][j]));

        CHECK(max_rel < 1e-12, "two fresh RAdam instances with identical grad sequences produce bit-identical params");

        delete ow1.model;
        delete ow2.model;
    }

    // ====================================================================
    // SECTION 12: End-to-end training on linear regression
    // ====================================================================
    {
        cout << "[12] End-to-end training reduces loss" << endl;
        // y = 2x + 1
        Owned ow = make_owned_dense(1, 1);
        ow.d->weights.fill(0.1);
        ow.d->bias.fill(0.0);

        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor target(1, 1);
        target[0][0] = 3.0;  // y = 2*1 + 1 = 3

        RAdam r(0.05, 0.9, 0.999, 1e-8, 0.0);

        double initial_loss = -1.0;
        double final_loss = -1.0;
        double w_plus_b = -1.0;
        for (int step = 0; step < 200; ++step) {
            Tensor out = ow.d->forward(input);
            double err = out[0][0] - target[0][0];
            double loss = 0.5 * err * err;
            if (step == 0) initial_loss = loss;
            if (step == 199) {
                final_loss = loss;
                w_plus_b = ow.d->weights[0][0] + ow.d->bias[0][0];
            }
            Tensor grad_out(1, 1);
            grad_out[0][0] = err;
            ow.d->backward(grad_out, 0.0);
            r.step(*ow.model);
        }

        cout << "    initial_loss = " << initial_loss
             << ", final_loss = " << final_loss
             << ", W+b = " << w_plus_b << endl;
        CHECK(final_loss < initial_loss * 0.1,
              "200-step RAdam reduces loss by > 90%");
        CHECK(std::abs(w_plus_b - 3.0) < 0.1,
              "200-step RAdam drives W + b close to target 3.0");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 13: RAdam vs Adam signature test
    //
    // At t=1 with β2=0.999 (RAdam SGD-like branch), RAdam's update is
    //   param = param − lr·m̂
    // while Adam's update is
    //   param = param − lr·m̂ / (sqrt(v̂) + ε)
    // For small v (early training) Adam's adaptive LR is much larger than RAdam's
    // SGD-like LR. So a single step should produce noticeably different param moves.
    // ====================================================================
    {
        cout << "[13] RAdam vs Adam signature" << endl;
        Owned ow_r = make_owned_dense(1, 1);
        Owned ow_a = make_owned_dense(1, 1);
        ow_r.d->weights.fill(1.0);
        ow_r.d->bias.fill(0.0);
        ow_a.d->weights.fill(1.0);
        ow_a.d->bias.fill(0.0);
        ow_r.d->grad_weights(0, 0) = 2.0;
        ow_a.d->grad_weights(0, 0) = 2.0;

        RAdam r(0.1, 0.9, 0.999, 1e-8, 0.0);
        r.step(*ow_r.model);
        // Adam: m_new = 0.2, v_new = 0.4, m̂=2, v̂=4, denom = sqrt(4)+1e-8 ≈ 2
        //   step = m̂ / denom = 2 / 2 = 1; param move = -0.1·1 = -0.1 → weights[0][0] = 0.9
        // RAdam: SGD-like → param move = -0.1·2 = -0.2 → weights[0][0] = 0.8
        CHECK_NEAR(ow_r.d->weights[0][0], 0.8, 1e-12,
                   "RAdam at t=1 with β2=0.999: SGD-like branch (no v denominator)");
        // Adam would give 0.9 — make sure we don't accidentally match Adam
        CHECK(std::abs(ow_r.d->weights[0][0] - 0.9) > 0.05,
               "RAdam at t=1 ≠ Adam at t=1 (signatures differ — adaptive LR is gated)");

        delete ow_r.model;
        delete ow_a.model;
    }

    // ====================================================================
    // SECTION 14: handles_weight_decay() returns true
    // ====================================================================
    {
        cout << "[14] handles_weight_decay()" << endl;
        RAdam r;
        CHECK(r.handles_weight_decay() == true, "handles_weight_decay() == true");
    }

    cout << "\n=== Summary: " << passed << "/" << total << " checks passed ===" << endl;
    return (passed == total) ? 0 : 1;
}