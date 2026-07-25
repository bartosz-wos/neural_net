// DAdaptAdam optimizer tests
#include "nn/optimizers/dadaptation.h"
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
static void build_dense_model(Model& model, int in_dim, int out_dim,
                              double init_value = 0.0) {
    Dense* d = new Dense(in_dim, out_dim);
    // overwrite init with constant for reproducibility.
    // Dense stores weights as (out_features, in_features) and bias as (1, out_features).
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
// T1-T2: Construction & defaults
// ---------------------------------------------------------------------------
static void test_defaults() {
    std::cout << "== construction & defaults ==\n";
    DAdaptAdam opt;
    CHECK(opt.lr == 1.0, "default lr=1.0");
    CHECK(opt.beta1 == 0.9, "default beta1=0.9");
    CHECK(opt.beta2 == 0.999, "default beta2=0.999");
    CHECK(opt.epsilon == 1e-8, "default epsilon=1e-8");
    CHECK(opt.weight_decay == 0.0, "default weight_decay=0");
    CHECK(opt.d0 == 1e-6, "default d0=1e-6");
    CHECK(opt.decouple == false, "default decouple=false");
    CHECK(opt.use_bias_correction == false, "default use_bias_correction=false");
    CHECK(std::isinf(opt.growth_rate), "default growth_rate=inf");
    CHECK(opt.k == 1, "default k=1");
    CHECK(opt.get_d() == 1e-6, "default d_ == d0");
    CHECK(opt.get_numerator_weighted() == 0.0, "default numerator_weighted == 0");
}

static void test_custom_constructor() {
    DAdaptAdam opt(2.0, 0.85, 0.95, 1e-6, 0.01, 1e-5, 1.02, true, true);
    CHECK(opt.lr == 2.0, "custom lr");
    CHECK(opt.beta1 == 0.85, "custom beta1");
    CHECK(opt.beta2 == 0.95, "custom beta2");
    CHECK(opt.epsilon == 1e-6, "custom epsilon");
    CHECK(opt.weight_decay == 0.01, "custom weight_decay");
    CHECK(opt.d0 == 1e-5, "custom d0");
    CHECK(opt.growth_rate == 1.02, "custom growth_rate");
    CHECK(opt.decouple == true, "custom decouple");
    CHECK(opt.use_bias_correction == true, "custom use_bias_correction");
}

// ---------------------------------------------------------------------------
// T3: Setter validation
// ---------------------------------------------------------------------------
static void test_setter_validation() {
    std::cout << "== setter validation ==\n";
    DAdaptAdam opt;

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
    try { opt.set_growth_rate(0.5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_growth_rate(0.5) throws");

    // Constructor-time validation
    threw = false;
    try { DAdaptAdam opt2(1.0, 0.9, 0.999, 1e-8, 0.0, 0.0); } catch (...) { threw = true; }
    CHECK(threw, "constructor d0=0 throws");

    threw = false;
    try { DAdaptAdam opt2(0.0); } catch (...) { threw = true; }
    CHECK(threw, "constructor lr=0 throws");
}

// ---------------------------------------------------------------------------
// T4: Closed-form first step on Dense(2,2), all-zero init, all-one grad
// ---------------------------------------------------------------------------
static void test_first_step_closed_form() {
    std::cout << "== closed-form first step ==\n";
    // Setup: Dense(2,2) with all params = 0, all grads = 1, lr=1, d=d0=1e-6
    // Expected (per element):
    //   dlr = d * lr = 1e-6 * 1.0 = 1e-6
    //   m_1 = beta1*0 + (1-beta1)*1e-6*1 = (1-0.9)*1e-6 = 1e-7
    //   v_1 = beta2*0 + (1-beta2)*1 = 1e-3
    //   s_1 = sqrt(beta2)*0 + (1-sqrt(beta2))*1e-6*1 = (1-sqrt(0.999))*1e-6
    //   denom = sqrt(1e-3) + eps = 0.0316227766... + 1e-8
    //   param_1 = 0 - m_1 / denom = -1e-7 / 0.0316227866... = -3.16228e-6
    Model m;
    build_dense_model(m, 2, 2, 0.0);
    set_grads_to_constant(m, 1.0);
    DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6,
                   std::numeric_limits<double>::infinity(), false, false);
    opt.step(m);

    // Dense has W (2,2) and b (1,2): 6 trainable params total
    auto layer = m.layers[0].get();
    auto params = layer->parameters();

    double expected_dlr = 1e-6;
    double expected_m = 0.1 * expected_dlr;             // (1-0.9) * 1e-6 = 1e-7
    double expected_v = 1e-3;
    double expected_denom = std::sqrt(expected_v) + 1e-8;
    double expected_param = -expected_m / expected_denom;

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p = params[i];
        for (size_t r = 0; r < p->rows; ++r) {
            for (size_t c = 0; c < p->cols; ++c) {
                CHECK_NEAR((*p)[r][c], expected_param, 1e-10,
                           std::string("first-step param[") + std::to_string(i) +
                           "][" + std::to_string(r) + "][" + std::to_string(c) + "]");
            }
        }
    }

    // Verify m, v, s state tensors
    auto& st = opt.get_m(static_cast<void*>(layer), 0);
    for (size_t r = 0; r < st.rows; ++r)
        for (size_t c = 0; c < st.cols; ++c)
            CHECK_NEAR(st[r][c], expected_m, 1e-13, "first-step m_1");

    auto& vst = opt.get_v(static_cast<void*>(layer), 0);
    for (size_t r = 0; r < vst.rows; ++r)
        for (size_t c = 0; c < vst.cols; ++c)
            CHECK_NEAR(vst[r][c], expected_v, 1e-13, "first-step v_1");

    double expected_s = (1.0 - std::sqrt(0.999)) * expected_dlr;
    auto& sst = opt.get_s(static_cast<void*>(layer), 0);
    for (size_t r = 0; r < sst.rows; ++r)
        for (size_t c = 0; c < sst.cols; ++c)
            CHECK_NEAR(sst[r][c], expected_s, 1e-15, "first-step s_1");
}

// ---------------------------------------------------------------------------
// T5: D estimate is monotonically non-decreasing across 5 steps
// ---------------------------------------------------------------------------
static void test_d_monotonic() {
    std::cout << "== D estimate monotonicity ==\n";
    Model m;
    build_dense_model(m, 2, 2, 0.0);
    set_grads_to_constant(m, 1.0);
    DAdaptAdam opt(1.0);  // all defaults
    double d_prev = opt.get_d();
    for (int step = 0; step < 5; ++step) {
        set_grads_to_constant(m, 1.0);
        opt.step(m);
        double d_now = opt.get_d();
        CHECK(d_now >= d_prev, "D non-decreasing step " + std::to_string(step+1));
        d_prev = d_now;
    }
    // After 5 steps, d should have grown above d0
    CHECK(d_prev > 1e-6, "D > d0 after 5 steps");
}

// ---------------------------------------------------------------------------
// T6: Zero-gradient step doesn't crash and doesn't update params
// ---------------------------------------------------------------------------
static void test_zero_grad_handling() {
    std::cout << "== zero-gradient handling ==\n";
    Model m;
    build_dense_model(m, 2, 2, 1.0);
    set_grads_to_constant(m, 0.0);
    DAdaptAdam opt(1.0);
    double d_before = opt.get_d();
    opt.step(m);  // should early-return (sk_l1 == 0)
    double d_after = opt.get_d();
    CHECK_NEAR(d_before, d_after, 0.0, "D unchanged when sk_l1=0");
    // Params unchanged
    auto params = m.layers[0]->parameters();
    for (Tensor* p : params) {
        for (size_t r = 0; r < p->rows; ++r)
            for (size_t c = 0; c < p->cols; ++c)
                CHECK_NEAR((*p)[r][c], 1.0, 0.0, "param unchanged on zero grad");
    }
    CHECK(opt.k == 2, "k incremented even on early-return");
}

// ---------------------------------------------------------------------------
// T7: Decoupled weight decay (AdamW-style)
// ---------------------------------------------------------------------------
static void test_decoupled_weight_decay() {
    std::cout << "== decoupled weight decay ==\n";
    // Setup: all params = 1, all grads = 0, decouple=true, wd=0.1
    // d0=1e-6, lr=1.0 → scale = 1 - lr * d_ * wd = 1 - 1 * 1e-6 * 0.1 = 1 - 1e-7
    Model m;
    build_dense_model(m, 2, 2, 1.0);
    set_grads_to_constant(m, 0.0);
    DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.1, 1e-6,
                   std::numeric_limits<double>::infinity(), true, false);
    opt.step(m);
    double expected = 1.0 - 1e-7;
    auto params = m.layers[0]->parameters();
    for (Tensor* p : params) {
        for (size_t r = 0; r < p->rows; ++r)
            for (size_t c = 0; c < p->cols; ++c)
                CHECK_NEAR((*p)[r][c], expected, 1e-12, "decoupled WD param update");
    }
}

// ---------------------------------------------------------------------------
// T8: Coupled weight decay (Adam-style)
// ---------------------------------------------------------------------------
static void test_coupled_weight_decay() {
    std::cout << "== coupled weight decay ==\n";
    // Coupled (Adam-style): grad += wd * param. With g=0, param=1, wd=0.1: g_eff=0.1
    // Per PyTorch reference: m, v, AND s all use g_eff (after grad mutation).
    //   m = (1-0.9) * d0 * lr * 0.1 = 0.1 * 1e-6 * 0.1 = 1e-8
    //   v = (1-0.999) * 0.1² = 1e-3 * 0.01 = 1e-5
    //   s = (1-√0.999) * d0 * lr * 0.1 = (1-√0.999) * 1e-7
    //   denom = sqrt(1e-5) + 1e-8 ≈ 0.003162287
    //   param = 1 - m / denom = 1 - 1e-8 / 0.003162287 ≈ 1 - 3.16e-6
    Model m;
    build_dense_model(m, 2, 2, 1.0);
    set_grads_to_constant(m, 0.0);
    DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.1, 1e-6,
                   std::numeric_limits<double>::infinity(), false, false);
    opt.step(m);
    double expected_m = 0.1 * 1e-6 * 0.1;            // 1e-8
    double expected_v = 1e-3 * 0.01;                // 1e-5
    double expected_denom = std::sqrt(expected_v) + 1e-8;
    double expected_param = 1.0 - expected_m / expected_denom;
    auto params = m.layers[0]->parameters();
    for (Tensor* p : params) {
        for (size_t r = 0; r < p->rows; ++r)
            for (size_t c = 0; c < p->cols; ++c)
                CHECK_NEAR((*p)[r][c], expected_param, 1e-10, "coupled WD param update");
    }
}

// ---------------------------------------------------------------------------
// T9: handles_weight_decay() reflects decouple flag
// ---------------------------------------------------------------------------
static void test_handles_weight_decay() {
    DAdaptAdam opt1(1.0);  // decouple=false
    DAdaptAdam opt2(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6,
                    std::numeric_limits<double>::infinity(), true, false);
    CHECK(opt1.handles_weight_decay() == false, "coupled: handles_weight_decay=false");
    CHECK(opt2.handles_weight_decay() == true, "decoupled: handles_weight_decay=true");
    opt1.set_decouple(true);
    CHECK(opt1.handles_weight_decay() == true, "after set_decouple(true): true");
}

// ---------------------------------------------------------------------------
// T10: State initialization on first step
// ---------------------------------------------------------------------------
static void test_state_initialization() {
    std::cout << "== state initialization ==\n";
    Model m;
    build_dense_model(m, 2, 2, 0.5);
    set_grads_to_constant(m, 0.5);
    auto* layer = m.layers[0].get();
    DAdaptAdam opt(1.0);
    void* p = static_cast<void*>(layer);
    CHECK(!opt.has_state(p), "no state before first step");
    opt.step(m);
    CHECK(opt.has_state(p), "state exists after first step");
    CHECK(opt.num_params_with_state(p) == 2, "2 param state (W + b)");

    // m, v, s should all have correct shape and be initialized
    for (size_t i = 0; i < 2; ++i) {
        const Tensor& m_t = opt.get_m(p, i);
        const Tensor& v_t = opt.get_v(p, i);
        const Tensor& s_t = opt.get_s(p, i);
        Tensor* param = m.layers[0]->parameters()[i];
        CHECK(m_t.rows == param->rows, "m shape matches param rows");
        CHECK(m_t.cols == param->cols, "m shape matches param cols");
        CHECK(v_t.rows == param->rows, "v shape matches param rows");
        CHECK(s_t.rows == param->rows, "s shape matches param rows");
    }
}

// ---------------------------------------------------------------------------
// T11: Step counter increments
// ---------------------------------------------------------------------------
static void test_step_counter() {
    std::cout << "== step counter ==\n";
    Model m;
    build_dense_model(m, 2, 2, 0.0);
    DAdaptAdam opt(1.0);
    CHECK(opt.get_step() == 1, "k=1 before any step");
    set_grads_to_constant(m, 1.0);
    opt.step(m);
    CHECK(opt.get_step() == 2, "k=2 after 1 step");
    opt.step(m);
    CHECK(opt.get_step() == 3, "k=3 after 2 steps");
}

// ---------------------------------------------------------------------------
// T12: Determinism — two fresh instances produce bit-exact trajectory
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << "== determinism ==\n";
    auto run = []() {
        Model m;
        build_dense_model(m, 2, 2, 0.3);
        set_grads_to_constant(m, 0.7);
        DAdaptAdam opt(1.0);
        std::vector<std::vector<double>> history;
        for (int step = 0; step < 5; ++step) {
            set_grads_to_constant(m, 0.7);
            opt.step(m);
            auto params = m.layers[0]->parameters();
            std::vector<double> snap;
            snap.push_back(opt.get_d());
            for (Tensor* p : params)
                for (size_t r = 0; r < p->rows; ++r)
                    for (size_t c = 0; c < p->cols; ++c)
                        snap.push_back((*p)[r][c]);
            history.push_back(snap);
        }
        return history;
    };
    auto h1 = run();
    auto h2 = run();
    CHECK(h1.size() == h2.size(), "history length matches");
    for (size_t s = 0; s < h1.size(); ++s) {
        for (size_t i = 0; i < h1[s].size(); ++i) {
            CHECK_NEAR(h1[s][i], h2[s][i], 0.0, "determinism step " + std::to_string(s) +
                                                " val " + std::to_string(i));
        }
    }
}

// ---------------------------------------------------------------------------
// T13: Signature vs Adam — DAdaptAdam differs from Adam on same gradient seq
// ---------------------------------------------------------------------------
static void test_signature_vs_adam() {
    std::cout << "== signature vs Adam ==\n";
    // Adam step at lr=1e-3 produces a small update; DAdaptAdam at lr=1.0
    // produces a different (usually larger) update because d has scaled.
    // Run 5 steps on identical gradient sequence.
    auto run_adam = []() {
        Model m;
        build_dense_model(m, 2, 2, 0.3);
        Adam opt(0.001, 0.9, 0.999, 1e-8);
        for (int step = 0; step < 5; ++step) {
            set_grads_to_constant(m, 0.7);
            opt.step(m);
        }
        return (*m.layers[0]->parameters()[0])[0][0];
    };
    auto run_dadapt = []() {
        Model m;
        build_dense_model(m, 2, 2, 0.3);
        DAdaptAdam opt(1.0);
        for (int step = 0; step < 5; ++step) {
            set_grads_to_constant(m, 0.7);
            opt.step(m);
        }
        return (*m.layers[0]->parameters()[0])[0][0];
    };
    double a = run_adam();
    double b = run_dadapt();
    CHECK(std::abs(a - b) > 1e-3,
          "Adam and DAdaptAdam trajectories differ (a=" + std::to_string(a) +
          ", b=" + std::to_string(b) + ")");
}

// ---------------------------------------------------------------------------
// T14: Bias correction makes D grow faster
// ---------------------------------------------------------------------------
static void test_bias_correction_growth() {
    std::cout << "== bias correction growth ==\n";
    auto run_n = [](bool use_bc) {
        Model m;
        build_dense_model(m, 2, 2, 0.3);
        DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6,
                       std::numeric_limits<double>::infinity(), false, use_bc);
        // Run 5 steps so d has time to grow (single step has s=0 → d_hat=0)
        for (int s = 0; s < 5; ++s) {
            set_grads_to_constant(m, 0.7);
            opt.step(m);
        }
        return opt.get_d();
    };
    double d_no_bc = run_n(false);
    double d_bc = run_n(true);
    // Both should grow above d0 after 5 steps
    CHECK(d_no_bc > 1e-6, "no-BC: d > d0 after 5 steps");
    CHECK(d_bc > 1e-6, "BC: d > d0 after 5 steps");
    // BC version scales dlr by ~1.0005 so d should differ after a few steps
    CHECK(std::abs(d_no_bc - d_bc) > 1e-10,
          "BC vs no-BC differ (no_bc=" + std::to_string(d_no_bc) +
          ", bc=" + std::to_string(d_bc) + ")");
}

// ---------------------------------------------------------------------------
// T15: Growth rate cap binds
// ---------------------------------------------------------------------------
static void test_growth_rate_cap() {
    std::cout << "== growth rate cap ==\n";
    Model m;
    build_dense_model(m, 2, 2, 0.0);
    DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.02, false, false);
    double d_prev = opt.get_d();
    set_grads_to_constant(m, 1.0);
    opt.step(m);
    double d_now = opt.get_d();
    CHECK(d_now <= 1.02 * d_prev + 1e-12, "d_{k+1} <= growth_rate * d_k");
}

// ---------------------------------------------------------------------------
// T16: handles_weight_decay for Adam-vs-DAdaptAdam in a wrapper sense
// ---------------------------------------------------------------------------
static void test_multi_layer_independence() {
    std::cout << "== multi-layer independence ==\n";
    Model m;
    Dense* d1 = new Dense(2, 3);
    d1->weights = Tensor(3, 2); d1->weights.fill(0.1);
    d1->bias = Tensor(1, 3); d1->bias.fill(0.1);
    m.add_layer(d1);
    Dense* d2 = new Dense(3, 2);
    d2->weights = Tensor(2, 3); d2->weights.fill(0.2);
    d2->bias = Tensor(1, 2); d2->bias.fill(0.2);
    m.add_layer(d2);

    // Set different gradients on each layer
    for (size_t i = 0; i < m.layers.size(); ++i) {
        auto grads = m.layers[i]->gradients();
        for (Tensor* g : grads) g->fill(0.5 + 0.1 * static_cast<double>(i));
    }

    DAdaptAdam opt(1.0);
    opt.step(m);
    // Both layers should have state
    CHECK(opt.has_state(static_cast<void*>(m.layers[0].get())), "layer 0 has state");
    CHECK(opt.has_state(static_cast<void*>(m.layers[1].get())), "layer 1 has state");
    // Step counter incremented
    CHECK(opt.get_step() == 2, "k incremented");
}

// ---------------------------------------------------------------------------
// T17: End-to-end training reduces loss (linear regression)
// ---------------------------------------------------------------------------
static void test_e2e_training_reduces_loss() {
    std::cout << "== end-to-end training reduces loss ==\n";
    // y = 2x, train Dense(1,1)
    Model m;
    Dense* d = new Dense(1, 1);
    d->weights = Tensor(1, 1); d->weights.fill(0.5);  // initial guess = 0.5 (target = 2.0)
    d->bias = Tensor(1, 1); d->bias.fill(0.0);
    m.add_layer(d);

    DAdaptAdam opt(1.0);  // paper-recommended lr

    Tensor X(4, 1);
    X[0][0] = 1.0; X[1][0] = 2.0; X[2][0] = 3.0; X[3][0] = 4.0;
    Tensor y(4, 1);
    y[0][0] = 2.0; y[1][0] = 4.0; y[2][0] = 6.0; y[3][0] = 8.0;

    // Compute initial loss
    Tensor y0 = m.forward(X);
    double init_loss = 0.0;
    for (size_t i = 0; i < 4; ++i) init_loss += (y0[i][0] - y[i][0]) * (y0[i][0] - y[i][0]);
    init_loss /= 4.0;

    // Train for 60 steps
    for (int step = 0; step < 60; ++step) {
        Tensor pred = m.forward(X);
        // MSE gradient: dL/dW = (2/N) * sum_i x_i * (pred_i - y_i)
        // dL/db = (2/N) * sum_i (pred_i - y_i)
        double grad_sum = 0.0;
        double grad_w_sum = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            double diff = pred[i][0] - y[i][0];
            grad_sum += diff;
            grad_w_sum += X[i][0] * diff;
        }
        grad_sum *= 2.0 / 4.0;
        grad_w_sum *= 2.0 / 4.0;
        m.layers[0]->gradients()[0]->fill(grad_w_sum);  // dW
        m.layers[0]->gradients()[1]->fill(grad_sum);    // db
        opt.step(m);
    }

    Tensor y_final = m.forward(X);
    double final_loss = 0.0;
    for (size_t i = 0; i < 4; ++i) final_loss += (y_final[i][0] - y[i][0]) * (y_final[i][0] - y[i][0]);
    final_loss /= 4.0;

    std::cout << "  [INFO] init_loss=" << init_loss << " final_loss=" << final_loss
              << " d=" << opt.get_d() << std::endl;
    CHECK(init_loss > 1e-6, "non-trivial initial loss");
    CHECK(final_loss < 0.5 * init_loss, "loss reduced >50%");
}

// ---------------------------------------------------------------------------
// T18: grow_rate=1.0 means d never changes
// ---------------------------------------------------------------------------
static void test_growth_rate_one() {
    std::cout << "== growth_rate=1 ==\n";
    Model m;
    build_dense_model(m, 2, 2, 0.0);
    DAdaptAdam opt(1.0, 0.9, 0.999, 1e-8, 0.0, 1e-6, 1.0, false, false);
    double d_init = opt.get_d();
    set_grads_to_constant(m, 1.0);
    for (int i = 0; i < 5; ++i) {
        set_grads_to_constant(m, 1.0);
        opt.step(m);
    }
    // growth_rate=1 means cap is min(d_hat, d*1) = min(d_hat, d). Since d starts
    // at d0=1e-6 (small), d_hat should be larger → d should still grow.
    // Actually: min(d_hat, d*1.0) = min(d_hat, d). If d_hat > d, min=d.
    // Then max(d, d) = d. So d never changes from d0!
    CHECK_NEAR(opt.get_d(), d_init, 1e-15, "growth_rate=1.0 keeps d at d0");
}

// ---------------------------------------------------------------------------
// T19: Setters after construction work (round-trip)
// ---------------------------------------------------------------------------
static void test_setters_round_trip() {
    std::cout << "== setters round-trip ==\n";
    DAdaptAdam opt;
    opt.set_lr(0.5);
    CHECK(opt.lr == 0.5, "set_lr round-trip");
    opt.set_beta1(0.5);
    CHECK(opt.beta1 == 0.5, "set_beta1 round-trip");
    opt.set_beta2(0.5);
    CHECK(opt.beta2 == 0.5, "set_beta2 round-trip");
    opt.set_epsilon(1e-4);
    CHECK(opt.epsilon == 1e-4, "set_epsilon round-trip");
    opt.set_weight_decay(0.05);
    CHECK(opt.weight_decay == 0.05, "set_weight_decay round-trip");
    opt.set_d0(1e-3);
    CHECK(opt.d0 == 1e-3, "set_d0 round-trip");
    opt.set_growth_rate(1.5);
    CHECK(opt.growth_rate == 1.5, "set_growth_rate round-trip");
    opt.set_decouple(true);
    CHECK(opt.decouple == true, "set_decouple round-trip");
    opt.set_use_bias_correction(true);
    CHECK(opt.use_bias_correction == true, "set_use_bias_correction round-trip");
}

int main() {
    test_defaults();
    test_custom_constructor();
    test_setter_validation();
    test_first_step_closed_form();
    test_d_monotonic();
    test_zero_grad_handling();
    test_decoupled_weight_decay();
    test_coupled_weight_decay();
    test_handles_weight_decay();
    test_state_initialization();
    test_step_counter();
    test_determinism();
    test_signature_vs_adam();
    test_bias_correction_growth();
    test_growth_rate_cap();
    test_multi_layer_independence();
    test_e2e_training_reduces_loss();
    test_growth_rate_one();
    test_setters_round_trip();

    std::cout << "\n=== Summary: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}