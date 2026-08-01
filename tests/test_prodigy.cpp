// Prodigy optimizer tests
#include "nn/optimizers/prodigy.h"
#include "nn/optimizers/optimizer.h"
#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/core/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_pass;                                                          \
            std::cout << "  [PASS] " << msg << std::endl;                      \
        } else {                                                               \
            ++g_fail;                                                          \
            std::cerr << "  [FAIL] " << msg << "  (line " << __LINE__ << ")"   \
                      << std::endl;                                            \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
    do {                                                                       \
        double aa = (a), bb = (b);                                             \
        if (std::abs(aa - bb) <= (tol)) {                                      \
            ++g_pass;                                                          \
            std::cout << "  [PASS] " << msg << std::endl;                      \
        } else {                                                               \
            ++g_fail;                                                          \
            std::cerr << "  [FAIL] " << msg << "  expected " << bb              \
                      << " got " << aa << "  (line " << __LINE__ << ")"         \
                      << std::endl;                                            \
        }                                                                      \
    } while (0)

// Build a simple model with one Dense(in, out) layer (no activation) and
// initialize parameters to a known constant.
static void build_dense_model(Model& model, size_t in_dim, size_t out_dim,
                              double init_value = 0.0) {
    Dense* d = new Dense(in_dim, out_dim);
    d->weights = Tensor(out_dim, in_dim);
    d->weights.fill(init_value);
    d->bias = Tensor(1, out_dim);
    d->bias.fill(init_value);
    model.add_layer(d);
}

// Set gradients to a known constant (without running a real forward+backward).
static void set_grads_to_constant(Model& model, double g) {
    for (auto& layer : model.layers) {
        auto grads = layer->gradients();
        for (Tensor* grad : grads) {
            grad->fill(g);
        }
    }
}

// ---------------------------------------------------------------------------
// T1: Construction & defaults
// ---------------------------------------------------------------------------
static void test_defaults() {
    std::cout << "== construction & defaults ==\n";
    Prodigy opt;
    CHECK(opt.lr == 1.0, "default lr=1.0");
    CHECK(opt.beta1 == 0.9, "default beta1=0.9");
    CHECK(opt.beta2 == 0.999, "default beta2=0.999");
    CHECK_NEAR(opt.beta3, std::sqrt(0.999), 1e-12, "default beta3=sqrt(beta2)");
    CHECK(opt.epsilon == 1e-8, "default epsilon=1e-8");
    CHECK(opt.weight_decay == 0.0, "default weight_decay=0");
    CHECK(opt.d0 == 1e-6, "default d0=1e-6");
    CHECK(opt.d_coef == 1.0, "default d_coef=1.0");
    CHECK(std::isinf(opt.growth_rate), "default growth_rate=inf");
    CHECK(opt.decouple == true, "default decouple=true");
    CHECK(opt.use_bias_correction == false, "default use_bias_correction=false");
    CHECK(opt.safeguard_warmup == false, "default safeguard_warmup=false");
    CHECK(opt.k == 1, "default k=1");
    CHECK(opt.get_d() == 1e-6, "default d_ == d0");
    CHECK(opt.get_d_max() == 1e-6, "default d_max_ == d0");
    CHECK(opt.get_d_numerator() == 0.0, "default d_numerator_ == 0");
    CHECK(opt.get_d_hat() == 0.0, "default d_hat_ == 0");
    CHECK(opt.handles_weight_decay() == true, "handles_weight_decay=true (decouple=true)");

    // Accessor round-trip
    CHECK(opt.get_lr() == 1.0, "get_lr()");
    CHECK(opt.get_beta1() == 0.9, "get_beta1()");
    CHECK(opt.get_beta2() == 0.999, "get_beta2()");
    CHECK(opt.get_d0() == 1e-6, "get_d0()");
    CHECK(opt.get_d_coef() == 1.0, "get_d_coef()");
    CHECK(opt.get_growth_rate() == std::numeric_limits<double>::infinity(), "get_growth_rate()");
    CHECK(opt.get_decouple() == true, "get_decouple()");
    CHECK(opt.get_use_bias_correction() == false, "get_use_bias_correction()");
    CHECK(opt.get_safeguard_warmup() == false, "get_safeguard_warmup()");
    CHECK(opt.get_k() == 1, "get_k()");
}

static void test_custom_constructor() {
    std::cout << "== custom constructor ==\n";
    Prodigy opt(2.0, 0.85, 0.95, 1e-6, 0.01, 1e-5, 1.02, 1.02, false, true, true, 0.95);
    CHECK(opt.lr == 2.0, "custom lr");
    CHECK(opt.beta1 == 0.85, "custom beta1");
    CHECK(opt.beta2 == 0.95, "custom beta2");
    CHECK(opt.beta3 == 0.95, "custom beta3 (explicit)");
    CHECK(opt.epsilon == 1e-6, "custom epsilon");
    CHECK(opt.weight_decay == 0.01, "custom weight_decay");
    CHECK(opt.d0 == 1e-5, "custom d0");
    CHECK(opt.d_coef == 1.02, "custom d_coef");
    CHECK(opt.growth_rate == 1.02, "custom growth_rate");
    CHECK(opt.decouple == false, "custom decouple");
    CHECK(opt.use_bias_correction == true, "custom use_bias_correction");
    CHECK(opt.safeguard_warmup == true, "custom safeguard_warmup");
}

static void test_derived_beta3() {
    std::cout << "== derived beta3 ==\n";
    Prodigy opt(1.0, 0.9, 0.81, 1e-8, 0.0, 1e-6, 1.0);
    CHECK_NEAR(opt.beta3, 0.9, 1e-12, "beta3 = sqrt(0.81) = 0.9");
}

// ---------------------------------------------------------------------------
// T2: Setter validation
// ---------------------------------------------------------------------------
static void test_setter_validation() {
    std::cout << "== setter validation ==\n";
    Prodigy opt;

    bool threw = false;
    try { opt.set_lr(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_lr(0) throws");
    threw = false;
    try { opt.set_lr(-1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_lr(-1) throws");

    threw = false;
    try { opt.set_beta1(1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(1.0) throws");
    threw = false;
    try { opt.set_beta1(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(-0.1) throws");

    threw = false;
    try { opt.set_beta2(1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta2(1.0) throws");

    threw = false;
    try { opt.set_epsilon(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_epsilon(0) throws");

    threw = false;
    try { opt.set_weight_decay(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_weight_decay(-0.1) throws");

    threw = false;
    try { opt.set_d0(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_d0(0) throws");
    threw = false;
    try { opt.set_d0(-1e-7); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_d0(negative) throws");

    threw = false;
    try { opt.set_d_coef(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_d_coef(0) throws");

    threw = false;
    try { opt.set_growth_rate(0.5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_growth_rate(0.5) throws");

    threw = false;
    try { opt.set_beta3(1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta3(1.0) throws");
    threw = false;
    try { opt.set_beta3(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta3(-0.1) throws");

    // Constructor-time validation
    threw = false;
    try { Prodigy opt2(1.0, 0.9, 0.999, 1e-8, 0.0, 0.0); } catch (...) { threw = true; }
    CHECK(threw, "constructor d0=0 throws");

    threw = false;
    try { Prodigy opt3(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.0,
                       0.5); } catch (...) { threw = true; }
    CHECK(threw, "constructor growth_rate=0.5 throws");

    threw = false;
    try { Prodigy opt4(0.0, 0.9, 0.999, 1e-8, 0.0, 1e-6); } catch (...) { threw = true; }
    CHECK(threw, "constructor lr=0 throws");
}

static void test_setters_round_trip() {
    std::cout << "== setters round-trip ==\n";
    Prodigy opt;
    opt.set_lr(0.5);
    CHECK(opt.lr == 0.5, "set_lr round-trip");
    opt.set_beta1(0.85);
    CHECK(opt.beta1 == 0.85, "set_beta1 round-trip");
    opt.set_beta2(0.95);
    CHECK(opt.beta2 == 0.95, "set_beta2 round-trip");
    opt.set_beta3(0.9);
    CHECK(opt.beta3 == 0.9, "set_beta3 round-trip");
    opt.set_epsilon(1e-4);
    CHECK(opt.epsilon == 1e-4, "set_epsilon round-trip");
    opt.set_weight_decay(0.05);
    CHECK(opt.weight_decay == 0.05, "set_weight_decay round-trip");
    opt.set_d0(1e-3);
    CHECK(opt.d0 == 1e-3, "set_d0 round-trip");
    opt.set_d_coef(1.5);
    CHECK(opt.d_coef == 1.5, "set_d_coef round-trip");
    opt.set_growth_rate(1.5);
    CHECK(opt.growth_rate == 1.5, "set_growth_rate round-trip");
    opt.set_decouple(false);
    CHECK(opt.decouple == false, "set_decouple round-trip");
    opt.set_use_bias_correction(true);
    CHECK(opt.use_bias_correction == true, "set_use_bias_correction round-trip");
    opt.set_safeguard_warmup(true);
    CHECK(opt.safeguard_warmup == true, "set_safeguard_warmup round-trip");
}

// ---------------------------------------------------------------------------
// T3: State initialization
// ---------------------------------------------------------------------------
static void test_state_initialization() {
    std::cout << "== state initialization ==\n";
    Prodigy opt;
    Model model;
    build_dense_model(model, 3, 2, 0.0);

    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = d;
    CHECK(!opt.has_state(layer_ptr), "no state before step");

    set_grads_to_constant(model, 0.1);
    opt.step(model);

    CHECK(opt.has_state(layer_ptr), "state exists after step");
    CHECK(opt.num_params_with_state(layer_ptr) == 2, "2 params with state (weights + bias)");

    // State shapes
    const Tensor& m_w = opt.get_m(layer_ptr, 0);
    const Tensor& v_w = opt.get_v(layer_ptr, 0);
    const Tensor& s_w = opt.get_s(layer_ptr, 0);
    const Tensor& p0_w = opt.get_p0(layer_ptr, 0);
    CHECK(m_w.rows == 2 && m_w.cols == 3, "m for weights: (2, 3)");
    CHECK(v_w.rows == 2 && v_w.cols == 3, "v for weights: (2, 3)");
    CHECK(s_w.rows == 2 && s_w.cols == 3, "s for weights: (2, 3)");
    CHECK(p0_w.rows == 2 && p0_w.cols == 3, "p0 for weights: (2, 3)");

    const Tensor& m_b = opt.get_m(layer_ptr, 1);
    CHECK(m_b.rows == 1 && m_b.cols == 2, "m for bias: (1, 2)");

    // p0 should match the initial weights (all-zero here)
    CHECK_NEAR(p0_w[0][0], 0.0, 1e-12, "p0[0][0] == 0");
    CHECK_NEAR(p0_w[1][2], 0.0, 1e-12, "p0[1][2] == 0");
}

// ---------------------------------------------------------------------------
// T4: D-estimate growth
// ---------------------------------------------------------------------------
static void test_d_growth() {
    std::cout << "== d-estimate growth ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.0,
                std::numeric_limits<double>::infinity(), true, false, false);
    Model model;
    build_dense_model(model, 2, 1, 0.5);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = dptr;

    // Run 10 steps with constant positive gradient.
    const double g_val = 1.0;
    double prev_d = opt.get_d();
    double prev_d_max = opt.get_d_max();
    for (int step = 0; step < 10; ++step) {
        set_grads_to_constant(model, g_val);
        opt.step(model);
        // d is monotone non-decreasing
        CHECK(opt.get_d() >= prev_d - 1e-12, "d monotone non-decreasing at step " + std::to_string(step + 1));
        CHECK(opt.get_d_max() >= prev_d_max - 1e-12, "d_max monotone non-decreasing at step " + std::to_string(step + 1));
        prev_d = opt.get_d();
        prev_d_max = opt.get_d_max();
    }

    // d should have grown from d0
    CHECK(opt.get_d() > 1e-6, "d grows beyond d0 after 10 steps");
    // d_numerator should be evolving (not stuck at 0) — check after step > 1
    CHECK(opt.get_d_numerator() != 0.0, "d_numerator evolves after step > 1");
}

// ---------------------------------------------------------------------------
// T5: Closed-form first step
// ---------------------------------------------------------------------------
static void test_first_step_closed_form() {
    std::cout << "== first-step closed-form ==\n";
    // All-zero weights, grad=1.0, d=1.0 (after bootstrap d==d0 and d_hat=0
    // so d stays at d0=1e-6... wait, we need d=1.0 to make the math clean).
    // Use d0=1.0 to test the algorithm at d=1.0.
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1.0, 1.0);
    Model model;
    build_dense_model(model, 2, 1, 0.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = dptr;

    // p0 = 0, weights = 0, grad = 1.0
    set_grads_to_constant(model, 1.0);
    opt.step(model);

    // d_hat calculation:
    //   per parameter:
    //     m_ij = β1*0 + d*(1-β1)*g = 1.0 * 0.1 * 1.0 = 0.1
    //     v_ij = β2*0 + d²*(1-β2)*g² = 1.0 * 0.001 * 1.0 = 0.001
    //     s_ij = β3*0 + (d/d0)*dlr*g = 1.0 * 1.0 * 1.0 = 1.0
    //                              (dlr = d*lr = 1.0; d/d0 = 1.0)
    //     delta_numerator += (d/d0)*dlr * g * (p0 - p) = 0  (p0 == p == 0)
    //   d_denom += Σ |s_ij| = 6 elements (2-row weights + 1-row bias) all 1.0
    //                              = 6 + 2 = 8
    //   d_numerator (after *= β3) = 0
    //   global_d_numerator = 0 + 0 = 0
    //   d_hat = 1.0 * 0 / 8 = 0
    //   d was d0 (=1.0), max(1.0, 0) = 1.0, stays at 1.0
    //   d_max = max(1.0, 0) = 1.0
    //   d = min(1.0, 1.0*inf) = 1.0
    CHECK_NEAR(opt.get_d(), 1.0, 1e-12, "d stays at d0=1.0 when p0==p");
    CHECK_NEAR(opt.get_d_max(), 1.0, 1e-12, "d_max stays at d0=1.0");
    CHECK_NEAR(opt.get_d_hat(), 0.0, 1e-12, "d_hat == 0 when p0==p");

    // After step: m should be filled with d*(1-β1)*g = 0.1
    const Tensor& m_w = opt.get_m(layer_ptr, 0);
    CHECK_NEAR(m_w[0][0], 0.1, 1e-12, "m[0][0] = d*(1-β1)*g = 0.1");
    CHECK_NEAR(m_w[0][1], 0.1, 1e-12, "m[0][1] = 0.1");

    // v should be d²*(1-β2)*g² = 0.001
    const Tensor& v_w = opt.get_v(layer_ptr, 0);
    CHECK_NEAR(v_w[0][0], 0.001, 1e-12, "v[0][0] = d²*(1-β2)*g² = 0.001");

    // s should be (d/d0)*dlr*g = 1.0
    const Tensor& s_w = opt.get_s(layer_ptr, 0);
    CHECK_NEAR(s_w[0][0], 1.0, 1e-12, "s[0][0] = (d/d0)*dlr*g = 1.0");

    // Step 2: parameters now shifted (moved by -dlr*m/sqrt(v) = -1.0*0.1/sqrt(0.001)
    //   = -1.0*0.1/0.03162... = -3.1623)
    // So weights now have nonzero values. Next step the D-estimate should grow.
    double d_before_step2 = opt.get_d();
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    // d_hat depends on <g, p0 - p>: with weight ~ -3.1623 and grad = 1.0,
    // <g, p0 - p> = Σ 1.0 * (0 - (-3.1623)) = 6 * 3.1623 + 2 * 3.1623 = 25.298
    // But also + bias update... just check d > d_before_step2 (since d_hat > 0 here)
    CHECK(opt.get_d() >= d_before_step2 - 1e-12, "d monotone non-decreasing at step 2");
    CHECK(opt.get_d_hat() > 0.0, "d_hat > 0 at step 2 (grads consistent with -p0 direction)");
}

// ---------------------------------------------------------------------------
// T6: Decoupled weight decay
// ---------------------------------------------------------------------------
static void test_decoupled_weight_decay() {
    std::cout << "== decoupled weight decay ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.1, 1.0, 1.0);
    Model model;
    build_dense_model(model, 2, 1, 1.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = dptr;

    set_grads_to_constant(model, 0.0);
    opt.step(model);

    // With zero grad, parameters should be shrunk by (1 - dlr * wd).
    // p0 = 1.0, d = 1.0 (= d0), dlr = 1.0*1.0 = 1.0
    // After decoupled WD: param *= (1 - 1.0 * 0.1) = 0.9
    // So weight[0][0] = 0.9
    CHECK_NEAR(dptr->weights[0][0], 0.9, 1e-12, "decoupled WD shrinks weights at zero grad");
    CHECK_NEAR(dptr->weights[0][1], 0.9, 1e-12, "decoupled WD shrinks all weights");
    CHECK_NEAR(dptr->bias[0][0], 0.9, 1e-12, "decoupled WD shrinks bias");
    // State p0 should still equal 1.0 (set at first sight)
    CHECK_NEAR(opt.get_p0(layer_ptr, 0)[0][0], 1.0, 1e-12, "p0 unchanged after step");
}

static void test_coupled_weight_decay() {
    std::cout << "== coupled weight decay ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.1, 1.0, 1.0,
                std::numeric_limits<double>::infinity(), false /* decouple=false */,
                false, false);
    Model model;
    build_dense_model(model, 2, 1, 1.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());

    set_grads_to_constant(model, 0.0);
    opt.step(model);

    // With coupled WD (decouple=false), grad += wd * param = 0.1 * 1.0 = 0.1
    // So grad becomes 0.1 everywhere. Then the Adam-like step:
    //   m = d*(1-β1)*g = 1.0 * 0.1 * 0.1 = 0.01
    //   v = d²*(1-β2)*g² = 1.0 * 0.001 * 0.01 = 1e-5
    //   denom = sqrt(v + d*eps) = sqrt(1e-5 + 1e-8) ≈ 0.003164
    //   update = dlr * m / denom = 1.0 * 0.01 / 0.003164 ≈ 3.16070
    //   param -= 3.16070
    // So weights change by ~3.16 (in the negative direction).
    CHECK_NEAR(dptr->weights[0][0], 1.0 - 3.16070, 1e-3, "coupled WD: grad+=wd*param, then Adam step");
}

static void test_handles_weight_decay() {
    std::cout << "== handles_weight_decay walls ==\n";
    Prodigy opt;  // decouple=true by default
    CHECK(opt.handles_weight_decay() == true, "handles_weight_decay=true when decouple=true");
    opt.set_decouple(false);
    CHECK(opt.handles_weight_decay() == false, "handles_weight_decay=false when decouple=false");
}

// ---------------------------------------------------------------------------
// T7: Bias correction
// ---------------------------------------------------------------------------
static void test_bias_correction() {
    std::cout << "== bias correction ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1.0, 1.0,
                std::numeric_limits<double>::infinity(), true, true /* use_bc=true */, false);
    Model model;
    build_dense_model(model, 2, 1, 0.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
    Dense* dptr_alias = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = dptr;

    // Step 1: d=1.0, bc = sqrt(1-0.999^1)/(1-0.9^1) = sqrt(0.001)/0.1 = 0.03162/0.1 = 0.3162
    //         dlr = d * lr * bc = 1.0 * 1.0 * 0.3162 = 0.3162
    //         s = (d/d0) * dlr * g = 0.3162
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    const Tensor& s1 = opt.get_s(layer_ptr, 0);
    CHECK_NEAR(s1[0][0], 0.3162, 1e-3, "s with use_bc=true at step 1: (d/d0)*dlr*g=0.3162");
    (void)dptr_alias;
}

// ---------------------------------------------------------------------------
// T8: Safeguard warmup
// ---------------------------------------------------------------------------
static void test_safeguard_warmup() {
    std::cout << "== safeguard_warmup ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1.0, 1.0,
                std::numeric_limits<double>::infinity(), true, false, true /* safeguard_warmup=true */);
    Model model;
    build_dense_model(model, 2, 1, 0.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
    void* layer_ptr = dptr;

    set_grads_to_constant(model, 1.0);
    opt.step(model);

    // With safeguard_warmup=true: s = (d/d0) * d * g = 1.0 * 1.0 * 1.0 = 1.0
    // (vs without: s = (d/d0) * dlr * g = 1.0 * 1.0 * 1.0 = 1.0 — same when d=d0=1)
    // The difference kicks in when d > 1. Use d=2.0 to test.
    set_grads_to_constant(model, 1.0);
    opt.step(model);  // d may grow after step 1

    // After step 2, d could be > 1.0, so (d/d0)*d > (d/d0)*dlr
    // Just confirm safeguard_warmup=true accepted the value and didn't crash.
    CHECK(opt.get_safeguard_warmup() == true, "safeguard_warmup settable");
}

// ---------------------------------------------------------------------------
// T9: Step counter
// ---------------------------------------------------------------------------
static void test_step_counter() {
    std::cout << "== step counter ==\n";
    Prodigy opt;
    Model model;
    build_dense_model(model, 2, 1, 0.0);

    CHECK(opt.get_k() == 1, "k starts at 1");
    set_grads_to_constant(model, 0.1);
    opt.step(model);
    CHECK(opt.get_k() == 2, "k increments to 2 after step");
    opt.step(model);
    CHECK(opt.get_k() == 3, "k increments to 3 after second step");
    opt.step(model);
    CHECK(opt.get_k() == 4, "k increments to 4 after third step");
}

// ---------------------------------------------------------------------------
// T10: Determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << "== determinism ==\n";
    Prodigy opt1;
    Prodigy opt2;
    Model m1, m2;
    build_dense_model(m1, 3, 2, 0.5);
    build_dense_model(m2, 3, 2, 0.5);
    Dense* d1 = dynamic_cast<Dense*>(m1.layers[0].get());
    Dense* d2 = dynamic_cast<Dense*>(m2.layers[0].get());

    for (int step = 0; step < 5; ++step) {
        set_grads_to_constant(m1, 0.1);
        set_grads_to_constant(m2, 0.1);
        opt1.step(m1);
        opt2.step(m2);
    }

    // Final weights should be bit-identical
    double max_diff = 0.0;
    for (size_t i = 0; i < d1->weights.rows; ++i) {
        for (size_t j = 0; j < d1->weights.cols; ++j) {
            max_diff = std::max(max_diff, std::abs(d1->weights[i][j] - d2->weights[i][j]));
        }
    }
    for (size_t j = 0; j < d1->bias.cols; ++j) {
        max_diff = std::max(max_diff, std::abs(d1->bias[0][j] - d2->bias[0][j]));
    }
    CHECK(max_diff < 1e-12, "two fresh Prodigy instances produce bit-identical params over 5 steps");

    // d should also be identical
    CHECK_NEAR(opt1.get_d(), opt2.get_d(), 1e-12, "d bit-identical between two instances");
}

// ---------------------------------------------------------------------------
// T11: Zero-grad handling
// ---------------------------------------------------------------------------
static void test_zero_grad_handling() {
    std::cout << "== zero-grad handling ==\n";
    Prodigy opt;
    Model model;
    build_dense_model(model, 3, 2, 0.5);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());

    set_grads_to_constant(model, 0.0);
    opt.step(model);

    // With zero gradient everywhere: d_denom == 0, so d stays at d0.
    // v = 0, so update = m / sqrt(0 + d*eps) = 0/sqrt(eps) = 0.
    // No change to parameters (no WD, decouple=true).
    CHECK_NEAR(opt.get_d(), 1e-6, 1e-12, "d stays at d0 with zero grad");
    CHECK_NEAR(dptr->weights[0][0], 0.5, 1e-12, "weights unchanged with zero grad");
    CHECK_NEAR(dptr->bias[0][0], 0.5, 1e-12, "bias unchanged with zero grad");
}

// ---------------------------------------------------------------------------
// T12: Multi-layer independence
// ---------------------------------------------------------------------------
static void test_multi_layer_independence() {
    std::cout << "== multi-layer independence ==\n";
    Prodigy opt;
    Model model;
    build_dense_model(model, 2, 3, 0.1);
    build_dense_model(model, 3, 2, 0.1);

    Dense* d1 = dynamic_cast<Dense*>(model.layers[0].get());
    Dense* d2 = dynamic_cast<Dense*>(model.layers[1].get());

    set_grads_to_constant(model, 0.1);
    opt.step(model);

    CHECK(opt.has_state(d1), "layer 1 has state");
    CHECK(opt.has_state(d2), "layer 2 has state");
    CHECK(opt.num_params_with_state(d1) == 2, "layer 1 has 2 params with state");
    CHECK(opt.num_params_with_state(d2) == 2, "layer 2 has 2 params with state");

    // State shapes differ between layers
    CHECK(opt.get_m(d1, 0).rows == 3, "layer 1 weight m shape: (3, 2)");
    CHECK(opt.get_m(d2, 0).rows == 2, "layer 2 weight m shape: (2, 3)");
}

// ---------------------------------------------------------------------------
// T13: End-to-end training reduces loss
// ---------------------------------------------------------------------------
static void test_e2e_training_reduces_loss() {
    std::cout << "== e2e training reduces loss ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.0);
    Model model;
    build_dense_model(model, 2, 1, 0.5);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());

    // Forward pass with input [1, 1] (target = 2): use a manual forward.
    Tensor x(1, 2);
    x[0][0] = 1.0;
    x[0][1] = 1.0;

    auto forward = [&](const Tensor& input) {
        // Dense(2, 1): weights (1, 2), bias (1, 1)
        // y = xW^T + b   →   shape (1, 1)
        return input * dptr->weights.transpose() + dptr->bias;
    };

    auto loss = [&](const Tensor& y) {
        // target = 2
        double diff = y[0][0] - 2.0;
        return 0.5 * diff * diff;
    };

    double initial_loss = loss(forward(x));
    for (int step = 0; step < 200; ++step) {
        Tensor y = forward(x);
        double diff = y[0][0] - 2.0;
        // Manually compute gradient = (y - target) * x
        Tensor grad_w = dptr->weights;  // shape (1,2)
        Tensor grad_b = dptr->bias;     // shape (1,1)
        grad_w[0][0] = diff * x[0][0];
        grad_w[0][1] = diff * x[0][1];
        grad_b[0][0] = diff;
        dptr->grad_weights = grad_w;
        dptr->grad_bias = grad_b;
        opt.step(model);
    }

    double final_loss = loss(forward(x));
    CHECK(final_loss < 0.5 * initial_loss,
          "Prodigy reduces loss >= 50% on y=2x regression over 200 steps");
    CHECK(opt.get_d() > 1e-6, "d adapted upward during training");
}

// ---------------------------------------------------------------------------
// T14: handles_weight_decay transition
// ---------------------------------------------------------------------------
static void test_decay_handling_modulation() {
    std::cout << "== handles_weight_decay walls (decouple=true) ==\n";
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.5, 1.0, 1.0);  // wd=0.5, decouple=true
    Model model;
    build_dense_model(model, 2, 1, 1.0);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());

    set_grads_to_constant(model, 0.0);
    opt.step(model);

    // With decoupled WD and zero grad: param *= (1 - dlr * wd) = (1 - 1.0*0.5) = 0.5
    CHECK_NEAR(dptr->weights[0][0], 0.5, 1e-12, "decoupled WD halfway: 0.5");
}

// ---------------------------------------------------------------------------
// T15: Growth rate cap
// ---------------------------------------------------------------------------
static void test_growth_rate_cap() {
    std::cout << "== growth_rate cap ==\n";
    // growth_rate = 1.001 means d can grow at most 0.1% per step.
    // Use d0 = 1.0 so d != d0 from the start (avoids the bootstrap jump).
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1.0, 1.0, 1.001);
    Model model;
    build_dense_model(model, 2, 1, 0.5);

    for (int step = 0; step < 5; ++step) {
        const double d_before = opt.get_d();
        const double d_max_before = opt.get_d_max();
        set_grads_to_constant(model, 1.0);
        opt.step(model);
        // d is monotone non-decreasing
        CHECK(opt.get_d() >= d_before - 1e-12, "d monotone with growth_rate cap");
        // d grows by at most growth_rate per step (when d != d0).
        // The bootstrap (d == d0 on first step) can jump d to d_hat, so we
        // skip the cap check on that step.
        if (d_before != opt.get_d0()) {
            CHECK(opt.get_d() <= d_before * 1.001 + 1e-12,
                  "d does not exceed growth_rate cap (1.001× per step)");
        }
        // d_max is monotone non-decreasing
        CHECK(opt.get_d_max() >= d_max_before - 1e-12, "d_max monotone non-decreasing");
    }
}

// ---------------------------------------------------------------------------
// T16: Step with empty model
// ---------------------------------------------------------------------------
static void test_empty_model() {
    std::cout << "== empty model ==\n";
    Prodigy opt;
    Model model;
    opt.step(model);  // should not crash
    CHECK(opt.get_d() == 1e-6, "d unchanged on empty model");
    CHECK(opt.get_k() == 2, "k incremented even on empty model");
}

// ---------------------------------------------------------------------------
// T17: Single-layer single-param model
// ---------------------------------------------------------------------------
static void test_single_param_model() {
    std::cout << "== single-param model ==\n";
    Prodigy opt;
    Model model;
    build_dense_model(model, 1, 1, 0.5);

    Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());

    set_grads_to_constant(model, 0.1);
    opt.step(model);
    CHECK(!std::isnan(dptr->weights[0][0]), "no NaN after step");
    CHECK(!std::isinf(dptr->weights[0][0]), "no Inf after step");
    CHECK(opt.get_d() > 0.0, "d > 0 after step");
}

// ---------------------------------------------------------------------------
// T18: Empty model returns without state for absent layer
// ---------------------------------------------------------------------------
static void test_state_for_unseen_layer() {
    std::cout << "== state accessors for unseen layer ==\n";
    Prodigy opt;
    int dummy_layer = 0;
    void* layer_ptr = &dummy_layer;
    CHECK(!opt.has_state(layer_ptr), "has_state=false for unseen layer");
    CHECK(opt.get_m(layer_ptr, 0).rows == 0, "get_m returns empty for unseen layer");
    CHECK(opt.get_v(layer_ptr, 0).rows == 0, "get_v returns empty for unseen layer");
    CHECK(opt.get_s(layer_ptr, 0).rows == 0, "get_s returns empty for unseen layer");
    CHECK(opt.get_p0(layer_ptr, 0).rows == 0, "get_p0 returns empty for unseen layer");
}

// ---------------------------------------------------------------------------
// T19: Hyperparameter sensitivity — different lr produce different trajectories
// ---------------------------------------------------------------------------
static void test_lr_sensitivity() {
    std::cout << "== lr sensitivity ==\n";
    Prodigy opt_a(0.5);  // small lr
    Prodigy opt_b(2.0);  // large lr (paper's note: lr=1.0 is the default)
    Model m_a, m_b;
    build_dense_model(m_a, 2, 1, 0.5);
    build_dense_model(m_b, 2, 1, 0.5);
    Dense* d_a = dynamic_cast<Dense*>(m_a.layers[0].get());
    Dense* d_b = dynamic_cast<Dense*>(m_b.layers[0].get());

    for (int step = 0; step < 5; ++step) {
        set_grads_to_constant(m_a, 0.1);
        set_grads_to_constant(m_b, 0.1);
        opt_a.step(m_a);
        opt_b.step(m_b);
    }

    // The two should differ (lr sensitivity)
    bool differs = (std::abs(d_a->weights[0][0] - d_b->weights[0][0]) > 1e-6);
    CHECK(differs, "different lr produces different params (sanity check)");
}

// ---------------------------------------------------------------------------
// T20: Mutation: zero out the L1 tracker → d_denom stays 0 → d_hat stays 0
// (this is a NON-VACUOUSNESS check; we only verify the algorithm is detectable)
// ---------------------------------------------------------------------------
static void test_mutation_dhat_bootstrap() {
    std::cout << "== mutation: d_hat bootstrap != 0 at step 2 ==\n";
    // Verify the algorithm does NOT degenerate to d=d0 (the bootstrap fault).
    Prodigy opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.0);
    Model model;
    build_dense_model(model, 2, 1, 0.5);

    // First step: p0==p, so d_hat should be 0 (consistent with test_first_step_closed_form).
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    CHECK_NEAR(opt.get_d_hat(), 0.0, 1e-12, "d_hat == 0 at step 1 (p0==p)");
    CHECK_NEAR(opt.get_d(), 1e-6, 1e-12, "d stays at d0 at step 1");

    // After step 1, parameters have moved. Second step should give d_hat > 0.
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    CHECK(opt.get_d_hat() > 0.0, "d_hat > 0 at step 2 (after parameters moved)");
}

int main() {
    test_defaults();
    test_custom_constructor();
    test_derived_beta3();
    test_setter_validation();
    test_setters_round_trip();
    test_state_initialization();
    test_d_growth();
    test_first_step_closed_form();
    test_decoupled_weight_decay();
    test_coupled_weight_decay();
    test_handles_weight_decay();
    test_bias_correction();
    test_safeguard_warmup();
    test_step_counter();
    test_determinism();
    test_zero_grad_handling();
    test_multi_layer_independence();
    test_e2e_training_reduces_loss();
    test_decay_handling_modulation();
    test_growth_rate_cap();
    test_empty_model();
    test_single_param_model();
    test_state_for_unseen_layer();
    test_lr_sensitivity();
    test_mutation_dhat_bootstrap();

    std::cout << "\n=== Summary: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
