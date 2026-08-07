// GaLore optimizer tests
// Reference: Zhao, Yang, Wang, Shen, Zhang, Rajan, Tan, Khan (2024),
// "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
// https://arxiv.org/abs/2403.03508
#include "nn/optimizers/galore.h"
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

// Build a 1-layer Dense model with weights initialized to a known constant.
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
    GaLore opt;
    CHECK(opt.lr == 0.001, "default lr=0.001");
    CHECK(opt.beta1 == 0.9, "default beta1=0.9");
    CHECK(opt.beta2 == 0.999, "default beta2=0.999");
    CHECK(opt.epsilon == 1e-8, "default epsilon=1e-8");
    CHECK(opt.rank == 4, "default rank=4");
    CHECK(opt.proj_update_interval == 200, "default proj_update_interval=200");
    CHECK(opt.weight_decay == 0.0, "default weight_decay=0");
    CHECK(opt.scale == 1.0, "default scale=1.0");
    CHECK(opt.handles_weight_decay() == true, "handles_weight_decay=true");
    CHECK(opt.get_step_count() == 1, "default step counter=1");

    // Accessor round-trip
    CHECK(opt.get_lr() == 0.001, "get_lr()");
    CHECK(opt.get_beta1() == 0.9, "get_beta1()");
    CHECK(opt.get_beta2() == 0.999, "get_beta2()");
    CHECK(opt.get_epsilon() == 1e-8, "get_epsilon()");
    CHECK(opt.get_rank() == 4, "get_rank()");
    CHECK(opt.get_proj_update_interval() == 200, "get_proj_update_interval()");
    CHECK(opt.get_weight_decay() == 0.0, "get_weight_decay()");
    CHECK(opt.get_scale() == 1.0, "get_scale()");
}

// ---------------------------------------------------------------------------
// T2: Constructor with non-default args
// ---------------------------------------------------------------------------
static void test_non_default_constructor() {
    std::cout << "== non-default constructor ==\n";
    GaLore opt(0.01, 0.85, 0.95, 1e-6, 8, 50, 0.01, 0.5);
    CHECK(opt.lr == 0.01, "lr=0.01");
    CHECK(opt.beta1 == 0.85, "beta1=0.85");
    CHECK(opt.beta2 == 0.95, "beta2=0.95");
    CHECK(opt.epsilon == 1e-6, "epsilon=1e-6");
    CHECK(opt.rank == 8, "rank=8");
    CHECK(opt.proj_update_interval == 50, "proj_update_interval=50");
    CHECK(opt.weight_decay == 0.01, "weight_decay=0.01");
    CHECK(opt.scale == 0.5, "scale=0.5");
}

// ---------------------------------------------------------------------------
// T3: Setter validation
// ---------------------------------------------------------------------------
static void test_setter_validation() {
    std::cout << "== setter validation ==\n";
    GaLore opt;

    // lr must be > 0
    bool threw = false;
    try { opt.set_lr(-0.001); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_lr(-0.001) throws");

    threw = false;
    try { opt.set_lr(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_lr(0) throws");

    // beta1 must be in [0, 1)
    threw = false;
    try { opt.set_beta1(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(-0.1) throws");

    threw = false;
    try { opt.set_beta1(1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(1.0) throws");

    // beta2 must be in [0, 1)
    threw = false;
    try { opt.set_beta2(1.5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta2(1.5) throws");

    // epsilon must be > 0
    threw = false;
    try { opt.set_epsilon(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_epsilon(0) throws");

    // rank must be >= 1
    threw = false;
    try { opt.set_rank(0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_rank(0) throws");

    // proj_update_interval must be >= 1
    threw = false;
    try { opt.set_proj_update_interval(0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_proj_update_interval(0) throws");

    // weight_decay must be >= 0
    threw = false;
    try { opt.set_weight_decay(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_weight_decay(-0.1) throws");

    // scale must be > 0
    threw = false;
    try { opt.set_scale(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_scale(0) throws");

    // Constructor-time validation
    threw = false;
    try { GaLore bad(0.0, 0.9, 0.999, 1e-8, 4, 200); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "constructor with lr=0 throws");

    threw = false;
    try { GaLore bad(0.001, 0.9, 0.999, 1e-8, 0, 200); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "constructor with rank=0 throws");
}

// ---------------------------------------------------------------------------
// T4: First-step bias correction
// ---------------------------------------------------------------------------
static void test_first_step_bias_correction() {
    std::cout << "== first step bias correction ==\n";
    // First step: weights (2,2)=0.5, grad=1.0, lr=0.001, rank=2 (n=2).
    //   G = [[1,1],[1,1]], G_EMA = 0.001*G after step 1.
    //   P = top-2 right singular vectors of G_EMA = (1/√2)[[1,1],[1,-1]]
    //   g_low = G @ P = [[√2, 0],[√2, 0]]
    //   At step 1, t=1, m_low = 0.1*g_low, m_hat = g_low.
    //   v_low = 0.001*g_low² = [[0.002, 0],[0.002, 0]], v_hat = g_low².
    //   update_low = m_hat/(sqrt(v_hat)+ε) ≈ sign(g_low) = [[1, 0],[1, 0]]
    //   update = update_low @ P^T = [[1, 0],[1, 0]] @ (1/√2)[[1, 1],[1, -1]]
    //          = (1/√2) * [[1, 1],[1, 1]] ≈ 0.7071 * [[1,1],[1,1]]
    //   param -= lr * update = 0.001 * 0.7071 = 0.000707
    //   So param goes from 0.5 to ~0.499293.
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 2, 2, 0.5);
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    CHECK_NEAR(d->weights[0][0], 0.5 - 0.001 / std::sqrt(2.0), 1e-6, "first step on weights: ~0.499293");
    CHECK_NEAR(d->bias[0][0], 0.5 - 0.001 / std::sqrt(2.0), 1e-6, "first step on bias: ~0.499293");
}

// ---------------------------------------------------------------------------
// T5: Projection matrix shape and orthonormality
// ---------------------------------------------------------------------------
static void test_projection_shape_orthonormality() {
    std::cout << "== projection shape & orthonormality ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 3, 4, 0.0);  // weights (4, 3), rank=2 → P (3, 2)
    set_grads_to_constant(model, 1.0);
    opt.step(model);

    // Read P for weights via pointer to layer
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    const Tensor& P = opt.get_P(d, 0);
    CHECK(P.rows == 3, "P has 3 rows (n=in_dim)");
    CHECK(P.cols == 2, "P has 2 cols (rank)");

    // P^T P should be (2, 2) identity (within tolerance)
    Tensor PtP = P.transpose() * P;
    CHECK_NEAR(PtP[0][0], 1.0, 1e-9, "P^T P [0][0] = 1");
    CHECK_NEAR(PtP[1][1], 1.0, 1e-9, "P^T P [1][1] = 1");
    CHECK_NEAR(PtP[0][1], 0.0, 1e-9, "P^T P [0][1] = 0");
    CHECK_NEAR(PtP[1][0], 0.0, 1e-9, "P^T P [1][0] = 0");

    // m_low and v_low should be (m, r) = (4, 2)
    const Tensor& m_low = opt.get_m_low(d, 0);
    CHECK(m_low.rows == 4, "m_low has 4 rows (m=out_dim)");
    CHECK(m_low.cols == 2, "m_low has 2 cols (rank)");
    const Tensor& v_low = opt.get_v_low(d, 0);
    CHECK(v_low.rows == 4, "v_low has 4 rows");
    CHECK(v_low.cols == 2, "v_low has 2 cols");
}

// ---------------------------------------------------------------------------
// T6: Projection refresh interval
// ---------------------------------------------------------------------------
static void test_proj_update_interval() {
    std::cout << "== projection refresh interval ==\n";
    // proj_update_interval = 3 → P should refresh on step 1, 4, 7, ...
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 3);
    Model model;
    build_dense_model(model, 3, 4, 0.0);
    set_grads_to_constant(model, 1.0);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());

    opt.step(model);
    Tensor P_after_1 = opt.get_P(d, 0).clone();
    int step_after_1 = opt.get_step_proj(d, 0);
    CHECK(step_after_1 == 1, "P refreshed on step 1");

    // Step 2: with constant grads, G_EMA doesn't change, but P should NOT refresh (interval=3)
    opt.step(model);
    CHECK(opt.get_step_proj(d, 0) == step_after_1, "P unchanged on step 2 (interval=3)");

    // Step 3: still not refreshed
    opt.step(model);
    CHECK(opt.get_step_proj(d, 0) == step_after_1, "P unchanged on step 3 (interval=3)");

    // Step 4: should refresh
    opt.step(model);
    CHECK(opt.get_step_proj(d, 0) == 4, "P refreshed on step 4");
}

// ---------------------------------------------------------------------------
// T7: Bias correction in steady state
// ---------------------------------------------------------------------------
static void test_bias_correction_steady_state() {
    std::cout << "== bias correction in steady state ==\n";
    // After enough steps with constant grad, m_hat ≈ g (geometric series).
    // We'll just verify m_low and v_low stay in projected space, and that the
    // step counter increments.
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 2, 2, 0.5);
    set_grads_to_constant(model, 1.0);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());

    int prev_step = opt.get_step_count();
    for (int i = 0; i < 5; ++i) {
        opt.step(model);
        CHECK(opt.get_step_count() == prev_step + i + 1, "step counter incremented");
    }
    CHECK(opt.get_step_count() == 6, "step counter = 6 after 5 steps");

    // m_low values: g_low[0][0] = G[0,:] @ P[:,0] = 1*P[0,0] + 1*P[1,0] = √2.
    // After 5 step() calls, m_low[0][0] = (1 - 0.9^5) * g_low[0][0] = (1-0.59049)*√2
    const Tensor& m_low = opt.get_m_low(d, 0);
    double expected = (1 - std::pow(0.9, 5)) * std::sqrt(2.0);
    CHECK_NEAR(m_low[0][0], expected, 1e-9, "m_low[0][0] = (1 - beta1^5) * sqrt(2)");
}

// ---------------------------------------------------------------------------
// T8: Multi-step determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << "== determinism ==\n";
    auto build = []() {
        GaLore opt(0.01, 0.9, 0.999, 1e-8, 2, 3);
        Model model;
        build_dense_model(model, 3, 4, 0.1);
        Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
        d->weights[0][0] = 0.5;
        d->weights[1][2] = -0.3;
        d->bias[0][0] = 0.7;
        // Set gradients to a non-trivial pattern
        d->grad_weights[0][0] = 0.1;
        d->grad_weights[1][1] = -0.2;
        d->grad_weights[2][2] = 0.05;
        d->grad_bias[0][0] = 0.1;
        d->grad_bias[0][2] = -0.05;
        return std::make_pair(std::move(opt), std::move(model));
    };
    auto p1 = build();
    auto p2 = build();
    for (int i = 0; i < 8; ++i) {
        p1.first.step(p1.second);
        p2.first.step(p2.second);
    }
    Dense* d1 = dynamic_cast<Dense*>(p1.second.layers[0].get());
    Dense* d2 = dynamic_cast<Dense*>(p2.second.layers[0].get());
    bool same = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(d1->weights[i][j] - d2->weights[i][j]) > 1e-15) {
                same = false;
            }
        }
    }
    CHECK(same, "two fresh GaLore instances produce bit-identical weights after 8 steps");
}

// ---------------------------------------------------------------------------
// T9: Decoupled weight decay
// ---------------------------------------------------------------------------
static void test_decoupled_weight_decay() {
    std::cout << "== decoupled weight decay ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200, 0.1);  // wd=0.1
    Model model;
    build_dense_model(model, 2, 2, 1.0);
    set_grads_to_constant(model, 0.0);  // zero gradient, so weight decay dominates
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());

    // First step: param *= (1 - lr*wd) = 1 - 0.001*0.1 = 0.9999
    opt.step(model);
    CHECK_NEAR(d->weights[0][0], 0.9999, 1e-9, "weight decay: 1.0 → 0.9999");
    CHECK_NEAR(d->bias[0][0], 0.9999, 1e-9, "bias weight decay: same");

    // Same 100 steps: param ≈ 1.0 * (1 - 0.0001)^100 = 0.99005 (geometric)
    for (int i = 0; i < 100; ++i) {
        opt.step(model);
    }
    double expected = std::pow(1 - 0.001 * 0.1, 101);
    CHECK_NEAR(d->weights[0][0], expected, 1e-9, "100 steps weight decay (geometric)");
}

// ---------------------------------------------------------------------------
// T10: Multi-layer independence
// ---------------------------------------------------------------------------
static void test_multi_layer() {
    std::cout << "== multi-layer independence ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    Dense* d1 = new Dense(2, 2);
    d1->weights = Tensor(2, 2);
    d1->weights.fill(0.1);
    d1->bias = Tensor(1, 2);
    d1->bias.fill(0.1);
    d1->grad_weights.fill(0.5);
    d1->grad_bias.fill(0.5);
    Dense* d2 = new Dense(2, 2);
    d2->weights = Tensor(2, 2);
    d2->weights.fill(0.2);
    d2->bias = Tensor(1, 2);
    d2->bias.fill(0.2);
    d2->grad_weights.fill(-0.5);
    d2->grad_bias.fill(-0.5);
    model.add_layer(d1);
    model.add_layer(d2);

    opt.step(model);

    // Both layers should have moved (not zero)
    CHECK(std::abs(d1->weights[0][0] - 0.1) > 1e-12, "layer 1 weights moved");
    CHECK(std::abs(d2->weights[0][0] - 0.2) > 1e-12, "layer 2 weights moved");
    // Opposite gradient direction → opposite movement direction
    CHECK(d1->weights[0][0] < 0.1, "layer 1 (positive grad) decreased");
    CHECK(d2->weights[0][0] > 0.2, "layer 2 (negative grad) increased");
}

// ---------------------------------------------------------------------------
// T11: End-to-end training reduces loss
// ---------------------------------------------------------------------------
static void test_end_to_end_training() {
    std::cout << "== end-to-end training reduces loss ==\n";
    // y = 2x regression with one Dense(1, 1) layer.
    // Use a tiny model and standard SGD/Adam-like update.
    GaLore opt(0.05, 0.9, 0.999, 1e-8, 1, 50);
    Model model;
    Dense* d = new Dense(1, 1);
    d->weights = Tensor(1, 1);
    d->weights[0][0] = 0.5;
    d->bias = Tensor(1, 1);
    d->bias[0][0] = 0.0;
    model.add_layer(d);

    // Training data: x = 1, y = 2
    Tensor x(1, 1);
    x[0][0] = 1.0;
    Tensor y_true(1, 1);
    y_true[0][0] = 2.0;

    auto compute_loss = [&](Model& m) {
        Tensor y_pred = m.layers[0]->forward(x);
        double diff = y_pred[0][0] - y_true[0][0];
        return 0.5 * diff * diff;
    };

    double initial_loss = compute_loss(model);
    for (int i = 0; i < 200; ++i) {
        Tensor y_pred = model.layers[0]->forward(x);
        Tensor grad = y_pred - y_true;
        Dense* dptr = dynamic_cast<Dense*>(model.layers[0].get());
        // grad_layer wrt weights: grad^T @ x → (1, 1) (since x is (1, 1))
        Tensor grad_w(1, 1);
        grad_w[0][0] = grad[0][0] * x[0][0];
        dptr->grad_weights = grad_w.clone();
        dptr->grad_bias = grad.clone();
        opt.step(model);
    }
    double final_loss = compute_loss(model);
    std::cout << "  initial loss: " << initial_loss << ", final loss: " << final_loss << "\n";
    CHECK(final_loss < initial_loss * 0.5, "loss reduced by >50% in 200 steps");
}

// ---------------------------------------------------------------------------
// T12: Different rank values
// ---------------------------------------------------------------------------
static void test_rank_variants() {
    std::cout << "== rank variants ==\n";
    for (int r = 1; r <= 4; ++r) {
        GaLore opt(0.001, 0.9, 0.999, 1e-8, r, 200);
        Model model;
        build_dense_model(model, 4, 4, 0.0);
        set_grads_to_constant(model, 1.0);
        opt.step(model);
        Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
        const Tensor& P = opt.get_P(d, 0);
        CHECK(P.rows == 4, "P has 4 rows for rank test");
        CHECK(P.cols == (size_t)r, "P has correct rank");
    }
}

// ---------------------------------------------------------------------------
// T13: Zero gradient doesn't crash
// ---------------------------------------------------------------------------
static void test_zero_gradient() {
    std::cout << "== zero gradient doesn't crash ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 2, 2, 0.5);
    set_grads_to_constant(model, 0.0);
    opt.step(model);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    // All params should remain unchanged (no gradient, no decay)
    CHECK_NEAR(d->weights[0][0], 0.5, 1e-15, "zero grad leaves weights unchanged");
    CHECK_NEAR(d->bias[0][0], 0.5, 1e-15, "zero grad leaves bias unchanged");
}

// ---------------------------------------------------------------------------
// T14: State accessors before step
// ---------------------------------------------------------------------------
static void test_state_accessors_before_step() {
    std::cout << "== state accessors before step ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 2, 2, 0.0);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    // Has_state should be false before any step
    CHECK(!opt.has_state(d), "has_state false before step");
    // Accessors should return empty tensors
    const Tensor& P = opt.get_P(d, 0);
    CHECK(P.rows == 0 && P.cols == 0, "P is empty before step");
    const Tensor& m = opt.get_m_low(d, 0);
    CHECK(m.rows == 0 && m.cols == 0, "m_low is empty before step");
    const Tensor& v = opt.get_v_low(d, 0);
    CHECK(v.rows == 0 && v.cols == 0, "v_low is empty before step");
}

// ---------------------------------------------------------------------------
// T15: Empty model doesn't crash
// ---------------------------------------------------------------------------
static void test_empty_model() {
    std::cout << "== empty model doesn't crash ==\n";
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    opt.step(model);  // should not crash
    CHECK(true, "step on empty model completed");
}

// ---------------------------------------------------------------------------
// T16: Quadratic projection (rank matches full rank)
// ---------------------------------------------------------------------------
static void test_rank_full() {
    std::cout << "== rank = full dim ==\n";
    // For a 2x2 tensor with rank=2, projection should be identity-ish
    GaLore opt(0.001, 0.9, 0.999, 1e-8, 2, 200);
    Model model;
    build_dense_model(model, 2, 2, 0.0);
    set_grads_to_constant(model, 1.0);
    opt.step(model);
    Dense* d = dynamic_cast<Dense*>(model.layers[0].get());
    const Tensor& P = opt.get_P(d, 0);
    // P should be (2, 2) and orthonormal
    CHECK(P.rows == 2, "P has 2 rows for 2x2 weight");
    CHECK(P.cols == 2, "P has 2 cols for rank=2");
    Tensor PtP = P.transpose() * P;
    CHECK_NEAR(PtP[0][0], 1.0, 1e-9, "P^T P [0][0] = 1");
    CHECK_NEAR(PtP[1][1], 1.0, 1e-9, "P^T P [1][1] = 1");
    CHECK_NEAR(PtP[0][1], 0.0, 1e-9, "P^T P [0][1] = 0");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== GaLore Tests ===\n";
    test_defaults();
    test_non_default_constructor();
    test_setter_validation();
    test_first_step_bias_correction();
    test_projection_shape_orthonormality();
    test_proj_update_interval();
    test_bias_correction_steady_state();
    test_determinism();
    test_decoupled_weight_decay();
    test_multi_layer();
    test_end_to_end_training();
    test_rank_variants();
    test_zero_gradient();
    test_state_accessors_before_step();
    test_empty_model();
    test_rank_full();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
