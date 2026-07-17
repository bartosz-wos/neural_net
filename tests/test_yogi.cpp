// Yogi optimizer tests — Zaheer et al. 2019 "Adaptive Methods for Nonconvex Optimization"
//
// Yogi is a self-correcting Adam variant. The signature update is:
//   v_t = v_{t-1} - (1 - b2) * sign(v_{t-1} - g^2) * g^2
//   (instead of Adam's v_t = b2 * v_{t-1} + (1 - b2) * g^2)
//
// The CORE tests below validate:
//
//  (1) Constructor defaults, accessors, mutators, validation throws.
//  (2) State shape, lazy-init per layer.
//  (3) First-step behavior of the v_t modification:
//      - v_prev=0, g=2 → v_new = +(1-b2) * 4 (always positive regardless of g's sign)
//      - v_prev huge vs g^2 → v_decreases
//      - v_prev tiny vs g^2 → v_increases
//  (4) Bias correction: m_hat = m_t / (1 - b1), v_hat = v_t / (1 - b2) at t=1, t=2.
//  (5) Step at t=1, grad=2, m_prev=0, v_prev=0:
//      matches the closed-form Adam-style update with v computed via Yogi rule.
//  (6) Weight decay: param strictly shrinks when wd>0 even with zero gradient.
//  (7) Determinism.
//  (8) End-to-end: 60-step Yogi on linear regression reduces MSE loss.
//  (9) Cross-optimizer signature: v_prev=0 case behaves differently across
//      gradient magnitudes (matches the rule exactly), confirming the
//      control-direction logic is wired in correctly.
//
// Note on memory: Model::add_layer takes ownership via unique_ptr<Layer>,
// so all Dense* pointers are created with `new Dense(...)` to avoid the
// stack-local + add_layer → double-free pattern. Each test block ends with
// `delete d;` after the model goes out of scope (or just lets the model
// destroy it — both work, but explicit delete makes the lifetime visible).

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <random>

#include "nn/optimizers/yogi.h"
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
                 << " tol=" << tol << ")" << endl;                            \
        }                                                                      \
        ++total;                                                               \
    } while (0)


// Helper: build a fresh (Dense, Model) pair and return pointers owned
// (caller is responsible for keeping Dense alive). We use the "new Dense"
// convention (matches adabelief/adafactor tests) since Model takes ownership.

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
    m->add_layer(d);  // model now owns `d`
    return {m, d};
}

int main() {
    cout << "=== Yogi Optimizer Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // ====================================================================
    // SECTION 1: Constructor defaults, accessors, validation
    // ====================================================================
    {
        cout << "[1] Defaults + accessors + validation" << endl;
        Yogi y;
        CHECK(y.get_lr()         == 1e-3, "default lr = 1e-3");
        CHECK(y.get_beta1()      == 0.9,  "default beta1 = 0.9");
        CHECK(y.get_beta2()      == 0.999,"default beta2 = 0.999");
        CHECK(y.get_epsilon()    == 1e-3, "default epsilon = 1e-3");
        CHECK(y.get_weight_decay() == 0.0,"default weight_decay = 0");
        CHECK(y.get_t()          == 1,    "default t = 1");

        Yogi y2(0.05, 0.85, 0.95, 1e-6, 0.01);
        CHECK(y2.get_lr() == 0.05 && y2.get_beta1() == 0.85 && y2.get_beta2() == 0.95
              && y2.get_epsilon() == 1e-6 && y2.get_weight_decay() == 0.01,
              "non-default constructor args");
    }

    // Validation throws
    {
        bool threw = false;
        try { Yogi y(0.1, 0.0, 0.9, 1e-8); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "beta1=0 throws");
    }
    {
        bool threw = false;
        try { Yogi y(0.1, 1.1, 0.9, 1e-8); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "beta1>=1 throws");
    }
    {
        bool threw = false;
        try { Yogi y(0.1, 0.9, 1.0, 1e-8); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "beta2>=1 throws");
    }
    {
        bool threw = false;
        try { Yogi y(0.1, 0.9, 0.0, 1e-8); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "beta2=0 throws");
    }
    {
        bool threw = false;
        try { Yogi y(0.1, 0.9, 0.9, 0.0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "epsilon<=0 throws");
    }
    {
        bool threw = false;
        try { Yogi y(0.1, 0.9, 0.9, -1e-8); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "negative epsilon throws");
    }
    {
        // Mutators validate too
        bool threw = false;
        Yogi y;
        try { y.set_beta1(2.0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "set_beta1>=1 throws");
    }

    // ====================================================================
    // SECTION 2: State shape + lazy-init
    // ====================================================================
    cout << "\n[2] State shape + lazy-init" << endl;
    {
        Owned ow = make_owned_dense(2, 3);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights.fill(0.0); d->bias.fill(0.0);
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        Yogi y;
        CHECK(!y.has_state(d), "no state before step()");
        // Force grad by running forward+backward on dummy input
        Tensor inp(1, 2); inp(0, 0) = 1.0; inp(0, 1) = 0.5;
        Tensor out = d->forward(inp);
        Tensor grad_loss(1, 3);
        for (size_t i = 0; i < 3; ++i) grad_loss(0, i) = 2.0 * out(0, i) / 3.0;
        d->backward(grad_loss, 0.0);
        y.step(m);
        CHECK(y.has_state(d), "state initialized after first step()");
        CHECK(y.get_m(d, 0).rows == 3 && y.get_m(d, 0).cols == 2,
              "m shape matches grad_weights (3,2)");
        CHECK(y.get_v(d, 0).rows == 3 && y.get_v(d, 0).cols == 2,
              "v shape matches grad_weights (3,2)");
        CHECK(y.get_m(d, 1).rows == 1 && y.get_v(d, 1).cols == 3,
              "bias m/v shape matches grad_bias (1,3)");
        CHECK(y.get_t() == 2, "t incremented to 2 after step()");

        delete ow.model;  // model owns `d`, so deleting the model deletes both
    }

    // ====================================================================
    // SECTION 3: First-step v_t modification behavior
    // ====================================================================
    // Yogi rule: v_t = v_{t-1} - (1-b2) * sign(v_{t-1} - g^2) * g^2
    // When v_prev = 0, g^2 > 0, delta = -g^2 < 0, sign = -1, so:
    //   v_new = 0 - (1-b2) * (-1) * g^2 = (1-b2) * g^2   (always positive)
    // When v_prev >> g^2, delta > 0, sign = +1, so:
    //   v_new = v_prev - (1-b2) * g^2   (decreases)
    // When v_prev << g^2, delta < 0, sign = -1, so:
    //   v_new = v_prev + (1-b2) * g^2   (increases)
    cout << "\n[3] First-step v_t modification (the Yogi 'soul')" << endl;
    {
        // (3a) v_prev = 0, grad = 2 → v_new = 0.1 * 4 = 0.4
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        const double b2 = 0.9;  // 1-b2 = 0.1
        Yogi y(0.01, 0.9, b2, 1e-8, 0.0);

        d->grad_weights(0, 0) = 2.0;
        y.step(m);
        double v1 = y.get_v(d, 0)(0, 0);
        CHECK_NEAR(v1, 0.4, 1e-12, "(3a) v_prev=0 g=2: v_new = (1-b2) * g^2 = 0.4");

        // (3b) v_prev = 0, grad = -3 → still positive 0.1 * 9 = 0.9
        Owned ow2 = make_owned_dense(1, 1);
        Model& m2 = *ow2.model;
        Dense* d2 = ow2.d;
        d2->weights(0, 0) = 0.0;
        d2->bias(0, 0)    = 0.0;
        d2->grad_weights.fill(0.0);
        d2->grad_bias.fill(0.0);
        Yogi y2(0.01, 0.9, b2, 1e-8, 0.0);
        d2->grad_weights(0, 0) = -3.0;
        y2.step(m2);
        double v1b = y2.get_v(d2, 0)(0, 0);
        CHECK_NEAR(v1b, 0.9, 1e-12, "(3b) v_prev=0 g=-3: v_new = (1-b2) * g^2 = 0.9");

        // (3c) v_prev huge vs g^2 → v decreases
        // Build up a large v_prev by running Yogi with g=10 a few times.
        Owned ow3 = make_owned_dense(1, 1);
        Model& m3 = *ow3.model;
        Dense* d3 = ow3.d;
        d3->weights(0, 0) = 0.0;
        d3->bias(0, 0)    = 0.0;
        d3->grad_weights.fill(0.0);
        d3->grad_bias.fill(0.0);
        Yogi y3(0.01, 0.9, b2, 1e-8, 0.0);
        d3->grad_weights(0, 0) = 10.0;
        for (int s = 0; s < 5; ++s) y3.step(m3);
        double v_before = y3.get_v(d3, 0)(0, 0);
        d3->grad_weights(0, 0) = 1.0;
        y3.step(m3);
        double v_after = y3.get_v(d3, 0)(0, 0);
        CHECK(v_after < v_before, "(3c) v decreases when v_prev > g^2 (control direction)");

        // (3d) Cross-check the magnitude of growth under small-v_prev + large-g
        Owned ow4 = make_owned_dense(1, 1);
        Model& m4 = *ow4.model;
        Dense* d4 = ow4.d;
        d4->weights(0, 0) = 0.0;
        d4->bias(0, 0)    = 0.0;
        d4->grad_weights.fill(0.0);
        d4->grad_bias.fill(0.0);
        Yogi y4(0.01, 0.9, b2, 1e-8, 0.0);
        // Build up a sizeable v_prev via many small-grad steps.
        d4->grad_weights(0, 0) = 1.0;
        for (int s = 0; s < 20; ++s) y4.step(m4);
        double v_pre = y4.get_v(d4, 0)(0, 0);
        // Now a HUGE grad step should grow v by exactly (1-b2)*g^2
        // (because v_prev << g^2 ⇒ delta<0 ⇒ sign=-1 ⇒ + term).
        d4->grad_weights(0, 0) = 100.0;
        y4.step(m4);
        double v_post = y4.get_v(d4, 0)(0, 0);
        CHECK_NEAR(v_post - v_pre, 0.1 * 100.0 * 100.0, 1e-6,
                   "(3d) v grows by (1-b2)*g^2 when v_prev << g^2");

        // (3e) Stagnation behavior: g=0 leaves v exactly unchanged
        Owned ow5 = make_owned_dense(1, 1);
        Model& m5 = *ow5.model;
        Dense* d5 = ow5.d;
        d5->weights(0, 0) = 0.0;
        d5->bias(0, 0)    = 0.0;
        d5->grad_weights.fill(0.0);
        d5->grad_bias.fill(0.0);
        Yogi y5(0.01, 0.9, b2, 1e-8, 0.0);
        d5->grad_weights(0, 0) = 5.0;
        y5.step(m5);
        double v0g0 = y5.get_v(d5, 0)(0, 0);
        // Now run with grad=0 — v should stay the same (g^2=0, no sign contribution)
        d5->grad_weights(0, 0) = 0.0;
        y5.step(m5);
        double v1g0 = y5.get_v(d5, 0)(0, 0);
        CHECK_NEAR(v1g0, v0g0, 1e-12, "(3e) v unchanged when g=0 (v_new = v_prev - (1-b2)*0)");

        delete ow.model; delete ow2.model; delete ow3.model;
        delete ow4.model; delete ow5.model;
    }

    // ====================================================================
    // SECTION 4: First-step closed-form (m_hat, v_hat, denom, step)
    // ====================================================================
    // lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0
    // grad = 2 on a 1x1 param starting at 0
    //   m_new = 0.9*0 + 0.1*2 = 0.2
    //   v_new (Yogi): v_prev=0, g=2, delta=-4<0, sign=-1 ⇒ v_new = -(1-0.999)*(-1)*4 = 0.004
    //   b1_c = 1 - 0.9 = 0.1
    //   b2_c = 1 - 0.999 = 0.001
    //   m_hat = 0.2 / 0.1  = 2.0
    //   v_hat = 0.004 / 0.001 = 4.0
    //   denom = sqrt(|4.0|) + 1e-8 ≈ 2.0
    //   step  = 2.0 / 2.0 = 1.0
    //   param = 0 - 0.1 * 1.0 = -0.1
    cout << "\n[4] First-step closed-form parameter update" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        Yogi y(0.1, 0.9, 0.999, 1e-8, 0.0);
        d->grad_weights(0, 0) = 2.0;
        y.step(m);

        double w_after = d->weights(0, 0);
        double m_val   = y.get_m(d, 0)(0, 0);
        double v_val   = y.get_v(d, 0)(0, 0);
        CHECK_NEAR(m_val, 0.2, 1e-12, "m_new at t=1");
        CHECK_NEAR(v_val, 0.004, 1e-12, "v_new via Yogi rule at t=1");
        // Note: with b2=0.999 and 1-b2=0.001, the precision of `std::pow(beta2,t)`
        // and the surrounding divisions leaves ~1e-10 relative noise. Even at
        // machine-precision the closed-form derivation here is exact in real
        // arithmetic; in IEEE-754 the chain b2_c = 1 - pow(b2, 1) introduces
        // ~10⁻¹⁶ relative error per division. 1e-7 is a generous floor.
        CHECK_NEAR(w_after, -0.1, 1e-7, "first-step closed-form param update");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 5: Weight decay shrinks param even with zero gradient
    // ====================================================================
    cout << "\n[5] Weight decay (AdamW-style, decoupled)" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 1.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        Yogi y(0.1, 0.9, 0.999, 1e-8, 0.01);  // wd = 0.01
        d->grad_weights(0, 0) = 0.0;            // no grad
        y.step(m);
        // Expected: param -= lr * wd * param = 1 - 0.1*0.01*1 = 0.999
        CHECK_NEAR(d->weights(0, 0), 1.0 - 0.1 * 0.01 * 1.0, 1e-12,
                   "decoupled wd shrinks param by lr*wd*param even with zero grad");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 6: Determinism (two fresh yogi instances, identical updates)
    // ====================================================================
    cout << "\n[6] Determinism" << endl;
    {
        Owned ow1 = make_owned_dense(2, 2);
        Owned ow2 = make_owned_dense(2, 2);
        Model& m1 = *ow1.model;
        Model& m2 = *ow2.model;
        Dense* d1 = ow1.d;
        Dense* d2 = ow2.d;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                d1->weights(i, j) = 0.01 * (i + j + 1);
                d1->bias(i, 0)    = 0.0;
                d2->weights(i, j) = 0.01 * (i + j + 1);
                d2->bias(i, 0)    = 0.0;
            }
        }
        d1->grad_weights = Tensor(2, 2); d1->grad_weights.fill(0.0);
        d1->grad_bias    = Tensor(1, 2); d1->grad_bias.fill(0.0);
        d2->grad_weights = Tensor(2, 2); d2->grad_weights.fill(0.0);
        d2->grad_bias    = Tensor(1, 2); d2->grad_bias.fill(0.0);

        Yogi y1(0.01, 0.9, 0.999, 1e-8, 0.001);
        Yogi y2(0.01, 0.9, 0.999, 1e-8, 0.001);

        std::mt19937 rng(1234);
        std::normal_distribution<double> nd;
        for (int step = 0; step < 20; ++step) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    double g = nd(rng);
                    d1->grad_weights(i, j) = g;
                    d2->grad_weights(i, j) = g;
                }
            }
            d1->grad_bias(0, 0) = nd(rng); d1->grad_bias(0, 1) = nd(rng);
            d2->grad_bias(0, 0) = d1->grad_bias(0, 0); d2->grad_bias(0, 1) = d1->grad_bias(0, 1);
            y1.step(m1);
            y2.step(m2);
        }

        bool all_match = true;
        for (int i = 0; i < 2 && all_match; ++i)
            for (int j = 0; j < 2 && all_match; ++j)
                if (rel_err(d1->weights(i, j), d2->weights(i, j)) > 1e-12) all_match = false;
        CHECK(all_match, "two fresh Yogi instances with identical grads produce identical params");

        delete ow1.model; delete ow2.model;
    }

    // ====================================================================
    // SECTION 7: End-to-end loss reduction on linear regression
    // ====================================================================
    // Task: minimize (W · x + b - y)^2 with x=1, y=3 (target: W + b = 3).
    // 60 steps of Yogi should drive W toward 2 and b toward 1.
    cout << "\n[7] End-to-end training (60 steps, MSE-driven)" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights = Tensor(1, 1); d->grad_weights.fill(0.0);
        d->grad_bias    = Tensor(1, 1); d->grad_bias.fill(0.0);

        Yogi y(0.05, 0.9, 0.999, 1e-8, 0.0);

        // single sample: x=1, target y=3
        Tensor inp(1, 1); inp(0, 0) = 1.0;
        Tensor tgt(1, 1); tgt(0, 0) = 3.0;

        Tensor o0 = d->forward(inp);
        double L0 = 0.5 * (o0(0, 0) - 3.0) * (o0(0, 0) - 3.0);
        CHECK(L0 > 0.5, "initial loss nonzero (sanity)");

        // Train loop
        for (int step = 0; step < 60; ++step) {
            // fresh forward + backward to populate grads
            Tensor out = d->forward(inp);
            // grad of 0.5*(out - y)^2 wrt out is (out - y)
            Tensor gl(1, 1);
            gl(0, 0) = (out(0, 0) - tgt(0, 0));
            d->backward(gl, 0.0);
            y.step(m);  // Yogi zeroes grads after stepping
        }

        Tensor o = d->forward(inp);
        double Lf = 0.5 * (o(0, 0) - 3.0) * (o(0, 0) - 3.0);
        CHECK(Lf < 0.1, "loss decreased substantially (final loss < 0.1)");
        CHECK(std::abs(d->weights(0, 0) + d->bias(0, 0) - 3.0) < 0.5,
              "W + b ≈ 3 (target fit on x=1, y=3)");

        delete ow.model;
    }

    // ====================================================================
    // SECTION 8: Signature difference vs naive EMA (sanity vs Adam at v_prev=0)
    // ====================================================================
    // At v_prev=0 the Yogi rule coincides with Adam's v update. The KEY
    // Yogi property is the sign-controlled growth in later steps. The
    // tests in section 3 (especially 3c — direction flip) verify that.
    // Here we sanity-check that the implementation at v_prev=0 produces
    // (1-b2)*g^2 exactly across multiple gradient magnitudes.
    cout << "\n[8] At v_prev=0, v_new matches (1-b2)*g^2 across magnitudes" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        Yogi y(0.05, 0.9, 0.5, 1e-8, 0.0);  // b2 = 0.5
        d->grad_weights(0, 0) = 7.0;
        y.step(m);
        // (1-b2)*g^2 = 0.5 * 49 = 24.5
        CHECK_NEAR(y.get_v(d, 0)(0, 0), 24.5, 1e-12,
                   "v_new at t=1 with v_prev=0 equals (1-b2)*g^2");
        delete ow.model;
    }

    // ====================================================================
    // SECTION 9: Cross-check Yogi v_new != Adam v_new when v_prev != 0
    // ====================================================================
    // Adam rule:    v_t = β2 * v_{t-1} + (1-β2) * g²
    // Yogi rule:    v_t = v_{t-1} − (1-β2) * sign(v_{t-1} − g²) * g²
    //
    // When v_prev = 1.0, β2 = 0.5, g = 0.1:
    //   Adam:  v_new = 0.5 * 1.0 + 0.5 * 0.01 = 0.505
    //   Yogi:  delta = 1.0 - 0.01 = 0.99 > 0, sign = +1
    //          v_new = 1.0 - 0.5 * 0.01 = 0.995
    //
    // Hence if the implementation is "drop-in Adam with the same state shape",
    // v_new = 0.505 — and this test will FAIL. This is the strongest
    // non-vacuous disambiguation between Yogi and Adam.
    cout << "\n[9] Yogi v_new differs from Adam v_new at v_prev != 0" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        // Manually populate v_prev=1.0 by performing one Yogi step with a
        // huge grad (we can verify it manually below). Easier: we put v_prev
        // directly by running one Yogi step with a config that yields a
        // known v_new, then setting v_prev to 1.0 indirectly isn't easy.
        //
        // Trick: pick a config where the Yogi rule yields v_new = 0.5 (via a
        // hypothetical previous state). We instead just directly assert
        // behavior from a fresh state, using the formula at v_prev=1.0:
        Owned ow_check = make_owned_dense(1, 1);
        Model& mc = *ow_check.model;
        Dense* dc = ow_check.d;
        dc->weights(0, 0) = 0.0;
        dc->bias(0, 0)    = 0.0;
        dc->grad_weights.fill(0.0);
        dc->grad_bias.fill(0.0);

        Yogi yc(0.01, 0.9, 0.5, 1e-8, 0.0);
        // Force v_prev to 1.0 by initializing state via one step at grad=sqrt(2/(1-b2)).
        // We need v_prev=1.0 ⇒ after a step with v_prev=0, v_new = (1-b2)*g^2 = 1.0
        // ⇒ g² = 2.0 ⇒ g = sqrt(2.0).
        double g_setup = std::sqrt(2.0);
        dc->grad_weights(0, 0) = g_setup;
        yc.step(mc);
        double v_after_setup = yc.get_v(dc, 0)(0, 0);
        CHECK_NEAR(v_after_setup, 1.0, 1e-12, "v_prev seeded to 1.0 via v_new = (1-b2)*g^2");

        // Now run with grad=0.1; Adam would give 0.505; Yogi should give 0.995
        dc->grad_weights(0, 0) = 0.1;
        yc.step(mc);
        double v_yogi = yc.get_v(dc, 0)(0, 0);
        CHECK_NEAR(v_yogi, 0.995, 1e-12, "v_new = 0.995 (Yogi), not 0.505 (Adam) — proves non-Adam");

        delete ow_check.model; delete ow.model;
    }

    // ====================================================================
    // SECTION 10: t increments through many steps
    // ====================================================================
    cout << "\n[10] Step counter t increments correctly" << endl;
    {
        Owned ow = make_owned_dense(1, 1);
        Model& m = *ow.model;
        Dense* d = ow.d;
        d->weights(0, 0) = 0.0;
        d->bias(0, 0)    = 0.0;
        d->grad_weights.fill(0.0);
        d->grad_bias.fill(0.0);

        Yogi y(0.01, 0.9, 0.9, 1e-8, 0.0);
        CHECK(y.get_t() == 1, "t starts at 1");
        d->grad_weights(0, 0) = 1.0;
        y.step(m);
        CHECK(y.get_t() == 2, "t = 2 after step 1");
        y.step(m);
        CHECK(y.get_t() == 3, "t = 3 after step 2");
        y.step(m);
        y.step(m);
        y.step(m);
        CHECK(y.get_t() == 6, "t = 6 after 5 total steps");

        delete ow.model;
    }

    cout << "\n=== Summary: " << passed << "/" << total << " checks passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
