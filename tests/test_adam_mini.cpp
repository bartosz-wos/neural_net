// Adam-mini optimizer tests
#include "nn/optimizers/adam_mini.h"
#include "nn/optimizers/optimizer.h"
#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/core/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
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

// ---------------------------------------------------------------------------
// T1-T2: Construction & defaults
// ---------------------------------------------------------------------------
static void test_defaults() {
    std::cout << "== construction & defaults ==\n";
    AdamMini opt;
    CHECK(opt.lr == 1e-3, "default lr=1e-3");
    CHECK(opt.beta1 == 0.9, "default beta1=0.9");
    CHECK(opt.beta2 == 0.999, "default beta2=0.999");
    CHECK(opt.epsilon == 1e-8, "default epsilon=1e-8");
    CHECK(opt.weight_decay == 0.0, "default weight_decay=0");
    CHECK(opt.t == 1, "default t=1");
    CHECK(opt.default_mode == AdamMini::BlockMode::AUTO, "default block-mode AUTO");
}

static void test_custom_constructor() {
    AdamMini opt(2e-3, 0.85, 0.95, 1e-6, 0.01, AdamMini::BlockMode::ROW_MEAN);
    CHECK(opt.lr == 2e-3, "custom lr");
    CHECK(opt.beta1 == 0.85, "custom beta1");
    CHECK(opt.beta2 == 0.95, "custom beta2");
    CHECK(opt.epsilon == 1e-6, "custom epsilon");
    CHECK(opt.weight_decay == 0.01, "custom weight_decay");
    CHECK(opt.default_mode == AdamMini::BlockMode::ROW_MEAN, "custom block-mode");
}

// ---------------------------------------------------------------------------
// T3: Setter validation
// ---------------------------------------------------------------------------
static void test_setter_validation() {
    std::cout << "== setter validation ==\n";
    AdamMini opt;
    bool threw = false;
    try { opt.set_lr(-1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_lr(-1) throws");

    threw = false;
    try { opt.set_beta1(1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(1.0) throws");

    threw = false;
    try { opt.set_beta1(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta1(-0.1) throws");

    threw = false;
    try { opt.set_beta2(1.5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_beta2(1.5) throws");

    threw = false;
    try { opt.set_epsilon(0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_epsilon(0) throws");

    threw = false;
    try { opt.set_epsilon(-1e-9); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_epsilon(-1e-9) throws");

    threw = false;
    try { opt.set_weight_decay(-0.01); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "set_weight_decay(-0.01) throws");

    // Valid mutations should succeed
    opt.set_lr(1e-2);
    CHECK(opt.lr == 1e-2, "set_lr(1e-2) succeeds");
    opt.set_beta1(0.5);
    CHECK(opt.beta1 == 0.5, "set_beta1(0.5) succeeds");
    opt.set_beta2(0.5);
    CHECK(opt.beta2 == 0.5, "set_beta2(0.5) succeeds");
    opt.set_epsilon(1e-4);
    CHECK(opt.epsilon == 1e-4, "set_epsilon(1e-4) succeeds");
    opt.set_weight_decay(0.05);
    CHECK(opt.weight_decay == 0.05, "set_weight_decay(0.05) succeeds");
}

// ---------------------------------------------------------------------------
// T4: hand-derived FULL mode closed form (1-D, single step)
// β1=β2=0.5, lr=1, ε=1, g=1, init=0
//   m_1 = 0.5*0 + 0.5*1 = 0.5; v_1 = 0.5*0 + 0.5*1 = 0.5
//   m̂_1 = 0.5 / (1-0.5) = 1.0; v̂_1 = 0.5 / (1-0.5) = 1.0
//   denom = sqrt(1) + 1 = 2; update = 1/2 = 0.5
//   new theta = 0 - 1*0.5 = -0.5
// ---------------------------------------------------------------------------
static void test_full_mode_closed_form() {
    std::cout << "== FULL mode closed-form (1-D, single step) ==\n";
    Model model;
    Dense* dense = new Dense(1, 1);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(1.0);

    AdamMini opt(1.0, 0.5, 0.5, 1.0, 0.0, AdamMini::BlockMode::FULL);
    opt.step(model);

    CHECK_NEAR((*params[0])[0][0], -0.5, 1e-12, "FULL mode: weight closed-form");
}

// ---------------------------------------------------------------------------
// T5: hand-derived ROW_MEAN mode closed form (2-D, single step)
// β1=β2=0.5, lr=1, ε=1, g=ones(4x3), init=0
//   vmean_per_row_step1 = mean(1² over 3) = 1 for each row
//   vmean_1 = 0.5*0 + 0.5*1 = 0.5 (same for all 4 rows)
//   v̂_1 = 0.5/(1-0.5) = 1.0
//   m_1 = 0.5; m̂_1 = 1.0
//   denom = sqrt(1)+1 = 2
//   update = 1/2 = 0.5
//   new theta = 0 - 1*0.5 = -0.5 (all entries)
// ---------------------------------------------------------------------------
static void test_row_mean_mode_closed_form() {
    std::cout << "== ROW_MEAN mode closed-form (2-D all-equal g) ==\n";
    Model model;
    Dense* dense = new Dense(3, 4);  // 3 inputs → 4 outputs, weight is (4, 3)
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(1.0);

    AdamMini opt(1.0, 0.5, 0.5, 1.0, 0.0, AdamMini::BlockMode::ROW_MEAN);
    opt.step(model);

    // All 12 entries should be -0.5
    bool all_correct = true;
    for (size_t r = 0; r < params[0]->rows; ++r)
        for (size_t c = 0; c < params[0]->cols; ++c)
            if (std::abs((*params[0])[r][c] - (-0.5)) > 1e-12) all_correct = false;
    CHECK(all_correct, "ROW_MEAN: all 12 entries are -0.5");

    // vmean state should be shape (rows, 1) = (4, 1), each entry 0.5
    auto& vmean = opt.get_vmean(dense, 0);
    CHECK(vmean.rows == 4, "vmean shape rows=4");
    CHECK(vmean.cols == 1, "vmean shape cols=1");
    bool all_v = true;
    for (size_t r = 0; r < vmean.rows; ++r)
        if (std::abs(vmean[r][0] - 0.5) > 1e-12) all_v = false;
    CHECK(all_v, "vmean[0..3][0] = 0.5");
}

// ---------------------------------------------------------------------------
// T6: ROW_MEAN distinguishes between rows with different magnitudes
// g = [[1,1,1],[1,1,1],[2,2,2],[2,2,2]]
// Row 0,1: mean(1²) = 1, row 2,3: mean(2²) = 4
// After 1 step with β2=0, lr=1, ε=1, β1=0 (so m = g directly, m̂ = g):
//   v̂_row0,1 = 1, denom = 1+1 = 2, update = g/2
//   v̂_row2,3 = 4, denom = 2+1 = 3, update = g/3
// ---------------------------------------------------------------------------
static void test_row_mean_distinguishes_rows() {
    std::cout << "== ROW_MEAN distinguishes rows ==\n";
    Model model;
    Dense* dense = new Dense(3, 4);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(1.0);
    for (size_t c = 0; c < 3; ++c) {
        (*grads[0])[2][c] = 2.0;
        (*grads[0])[3][c] = 2.0;
    }

    AdamMini opt(1.0, 0.0, 0.0, 1.0, 0.0, AdamMini::BlockMode::ROW_MEAN);
    opt.step(model);

    CHECK_NEAR((*params[0])[0][0], -0.5, 1e-12, "row 0 col 0 (g=1, denom=2)");
    CHECK_NEAR((*params[0])[1][2], -0.5, 1e-12, "row 1 col 2 (g=1, denom=2)");
    CHECK_NEAR((*params[0])[2][1], -2.0 / 3.0, 1e-12, "row 2 col 1 (g=2, denom=3)");
    CHECK_NEAR((*params[0])[3][0], -2.0 / 3.0, 1e-12, "row 3 col 0 (g=2, denom=3)");
}

// ---------------------------------------------------------------------------
// T6b: ROW_MEAN reduces WITHIN a row (variance within a row matters)
// g = [[0, 2, 0], ...]: row 0 mean(g²) = 4/3 ≈ 1.333, NOT 0 (which a buggy
// "use col 0 only" mutation would produce).
// ---------------------------------------------------------------------------
static void test_row_mean_within_row_reduction() {
    std::cout << "== ROW_MEAN within-row reduction ==\n";
    Model model;
    Dense* dense = new Dense(3, 2);  // weight (2, 3)
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    // Row 0: g = [0, 2, 0] → mean(g²) = 4/3 ≈ 1.333
    // Row 1: g = [3, 0, 0] → mean(g²) = 9/3 = 3.0
    (*grads[0])[0][0] = 0.0; (*grads[0])[0][1] = 2.0; (*grads[0])[0][2] = 0.0;
    (*grads[0])[1][0] = 3.0; (*grads[0])[1][1] = 0.0; (*grads[0])[1][2] = 0.0;

    AdamMini opt(1.0, 0.0, 0.0, 1.0, 0.0, AdamMini::BlockMode::ROW_MEAN);
    opt.step(model);

    // Row 0: β1=0, lr=1, ε=1
    //   v_1 = (1-0)*4/3 = 4/3; v̂_1 = 4/3; denom = sqrt(4/3)+1 ≈ 2.1547
    //   m_1 = g; m̂ = g (bc1=1)
    //   update = g / denom
    //   For col 0 (g=0): update=0, new θ = 0
    //   For col 1 (g=2): update = 2/2.1547 ≈ 0.9283, new θ = -0.9283
    //   For col 2 (g=0): update=0, new θ = 0
    double v_hat_r0 = 4.0 / 3.0;
    double denom_r0 = std::sqrt(v_hat_r0) + 1.0;
    double upd_r0_c1 = 2.0 / denom_r0;
    CHECK_NEAR((*params[0])[0][1], -upd_r0_c1, 1e-12, "row 0 col 1 (g=2, denom from row mean)");

    // Row 1: v_1 = 9/3 = 3; v̂_1 = 3; denom = sqrt(3)+1 ≈ 2.732
    //   For col 0 (g=3): update = 3/2.732 ≈ 1.098, new θ = -1.098
    double v_hat_r1 = 3.0;
    double denom_r1 = std::sqrt(v_hat_r1) + 1.0;
    double upd_r1_c0 = 3.0 / denom_r1;
    CHECK_NEAR((*params[0])[1][0], -upd_r1_c0, 1e-12, "row 1 col 0 (g=3, denom from row mean)");

    // Col 0 in row 0 has g=0, so update=0 regardless of vmean
    CHECK_NEAR((*params[0])[0][0], 0.0, 1e-12, "row 0 col 0 (g=0, no update)");
}

// ---------------------------------------------------------------------------
// T7: SCALAR mode closed form (asymmetric gradient)
// g = [[1,1,1],[1,1,1],[2,2,2],[2,2,2]] over 12 entries; mean(g²) = (6*1 + 6*4)/12 = 30/12 = 2.5
// v̂_1 = 2.5; denom = sqrt(2.5)+1 ≈ 2.5811
// β1=0, so m = g directly. m̂ = g (bc1 = 1).
// Row 0,1: update = 1/2.5811; new θ = -1/2.5811
// Row 2,3: update = 2/2.5811; new θ = -2/2.5811
// ---------------------------------------------------------------------------
static void test_scalar_mode_closed_form() {
    std::cout << "== SCALAR mode closed-form (asymmetric g) ==\n";
    Model model;
    Dense* dense = new Dense(3, 4);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(1.0);
    for (size_t c = 0; c < 3; ++c) {
        (*grads[0])[2][c] = 2.0;
        (*grads[0])[3][c] = 2.0;
    }

    AdamMini opt(1.0, 0.0, 0.0, 1.0, 0.0, AdamMini::BlockMode::SCALAR);
    opt.step(model);

    double mean_gg = (6.0 * 1.0 + 6.0 * 4.0) / 12.0;  // 2.5
    double denom = std::sqrt(mean_gg) + 1.0;  // ~2.5811
    double upd_row01 = 1.0 / denom;
    double upd_row23 = 2.0 / denom;

    CHECK_NEAR((*params[0])[0][0], -upd_row01, 1e-12, "scalar: row 0 (g=1)");
    CHECK_NEAR((*params[0])[3][0], -upd_row23, 1e-12, "scalar: row 3 (g=2)");

    auto& vmean = opt.get_vmean(dense, 0);
    CHECK(vmean.rows == 1 && vmean.cols == 1, "vmean scalar shape (1,1)");
    CHECK_NEAR(vmean[0][0], mean_gg, 1e-12, "vmean scalar value");
}

// ---------------------------------------------------------------------------
// T8: Bias correction on step 2 differs from step 1
// Step 1: bc1 = 1-β1 = 0.5, m̂ = m/0.5 = 2m
// Step 2: bc1 = 1-β1² = 0.75, m̂ = m/0.75 = 1.333m
// ---------------------------------------------------------------------------
static void test_bias_correction_step2() {
    std::cout << "== bias correction step 2 ==\n";
    Model model;
    Dense* dense = new Dense(1, 1);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(1.0);

    AdamMini opt(1.0, 0.5, 0.5, 1.0, 0.0, AdamMini::BlockMode::FULL);
    opt.step(model);  // t=1
    double after_step1 = (*params[0])[0][0];
    CHECK_NEAR(after_step1, -0.5, 1e-12, "step 1 closed form");

    grads[0]->fill(1.0);
    opt.step(model);  // t=2
    double after_step2 = (*params[0])[0][0];
    // Step 2:
    //   m_2 = 0.5*0.5 + 0.5*1 = 0.75
    //   v_2 = 0.5*0.5 + 0.5*1 = 0.75
    //   bc1 = 1-0.25 = 0.75, m̂ = 0.75/0.75 = 1.0
    //   bc2 = 1-0.25 = 0.75, v̂ = 0.75/0.75 = 1.0
    //   denom = 1+1 = 2, update = 0.5
    //   new θ = -0.5 - 0.5 = -1.0
    CHECK_NEAR(after_step2, -1.0, 1e-12, "step 2 closed form (-1.0)");

    CHECK(opt.t == 3, "t increments to 3 after step 2");

    grads[0]->fill(1.0);
    opt.step(model);
    double after_step3 = (*params[0])[0][0];
    // m_3 = 0.5*0.75 + 0.5*1 = 0.875; v_3 = 0.875; bc1 = 1-0.125 = 0.875; bc2 = 0.875
    // m̂ = 0.875/0.875 = 1.0; v̂ = 1.0; denom = 2; update = 0.5
    // new θ = -1.0 - 0.5 = -1.5
    CHECK_NEAR(after_step3, -1.5, 1e-12, "step 3 closed form (-1.5)");
}

// ---------------------------------------------------------------------------
// T9: State shape correctness
// ---------------------------------------------------------------------------
static void test_state_shapes() {
    std::cout << "== state shape correctness ==\n";
    Model model;
    Dense* dense = new Dense(3, 4);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(0.0);
    grads[0]->fill(0.1);
    grads[1]->fill(0.1);

    AdamMini opt(1e-2, 0.9, 0.999, 1e-8, 0.0);  // default AUTO mode
    opt.step(model);

    CHECK(opt.has_state(dense), "state exists after step");
    CHECK(opt.num_params_with_state(dense) == 2, "state has 2 params");

    auto& m_w = opt.get_m(dense, 0);
    auto& v_w = opt.get_vmean(dense, 0);
    CHECK(m_w.rows == 4 && m_w.cols == 3, "weight m shape (4,3)");
    CHECK(v_w.rows == 4 && v_w.cols == 1, "weight vmean shape (4,1) for ROW_MEAN");
    CHECK(opt.get_block_mode(dense, 0) == AdamMini::BlockMode::ROW_MEAN, "weight mode = ROW_MEAN");

    auto& m_b = opt.get_m(dense, 1);
    auto& v_b = opt.get_vmean(dense, 1);
    CHECK(m_b.rows == 1 && m_b.cols == 4, "bias m shape (1,4)");
    CHECK(v_b.rows == 1 && v_b.cols == 4, "bias vmean shape (1,4) for FULL");
    CHECK(opt.get_block_mode(dense, 1) == AdamMini::BlockMode::FULL, "bias mode = FULL");
}

// ---------------------------------------------------------------------------
// T10: Decoupled weight decay
// ---------------------------------------------------------------------------
static void test_weight_decay() {
    std::cout << "== decoupled weight decay ==\n";
    Model model;
    Dense* dense = new Dense(1, 1);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) p->fill(1.0);
    grads[0]->fill(0.0);

    AdamMini opt(0.1, 0.0, 0.0, 1e-8, 0.1, AdamMini::BlockMode::FULL);
    opt.step(model);

    // θ_new = 1.0 * (1 - 0.1*0.1) - 0.1 * 0 = 1.0 * 0.99 - 0 = 0.99
    CHECK_NEAR((*params[0])[0][0], 0.99, 1e-12, "weight decay: 1.0 → 0.99 with wd=0.1");
}

// ---------------------------------------------------------------------------
// T11: Zero gradient doesn't crash and doesn't move params
// ---------------------------------------------------------------------------
static void test_zero_grad() {
    std::cout << "== zero gradient step ==\n";
    Model model;
    Dense* dense = new Dense(2, 3);
    model.add_layer(dense);
    auto params = dense->parameters();
    auto grads = dense->gradients();
    for (auto* p : params) {
        for (size_t r = 0; r < p->rows; ++r)
            for (size_t c = 0; c < p->cols; ++c)
                (*p)[r][c] = 0.5;
    }
    grads[0]->fill(0.0);
    grads[1]->fill(0.0);

    AdamMini opt(0.1, 0.9, 0.999, 1e-8, 0.0);
    opt.step(model);

    bool unchanged = true;
    for (auto* p : params) {
        for (size_t r = 0; r < p->rows; ++r)
            for (size_t c = 0; c < p->cols; ++c)
                if (std::abs((*p)[r][c] - 0.5) > 1e-12) unchanged = false;
    }
    CHECK(unchanged, "zero grad: params unchanged");
}

// ---------------------------------------------------------------------------
// T12: Multi-layer model with independent state
// ---------------------------------------------------------------------------
static void test_multi_layer() {
    std::cout << "== multi-layer independent state ==\n";
    Model model;
    Dense* d1 = new Dense(2, 3);
    Dense* d2 = new Dense(3, 2);
    model.add_layer(d1);
    model.add_layer(d2);
    auto p1 = d1->parameters();
    auto p2 = d2->parameters();
    auto g1 = d1->gradients();
    auto g2 = d2->gradients();
    for (auto* p : p1) p->fill(0.0);
    for (auto* p : p2) p->fill(0.0);
    g1[0]->fill(1.0);
    g2[0]->fill(2.0);

    AdamMini opt(1.0, 0.0, 0.0, 1.0, 0.0, AdamMini::BlockMode::ROW_MEAN);
    opt.step(model);

    CHECK_NEAR((*p1[0])[0][0], -0.5, 1e-12, "layer1 weight[0][0]");
    CHECK_NEAR((*p2[0])[0][0], -2.0 / 3.0, 1e-12, "layer2 weight[0][0]");

    CHECK(opt.has_state(d1), "layer1 has state");
    CHECK(opt.has_state(d2), "layer2 has state");
    CHECK_NEAR(opt.get_vmean(d1, 0)[0][0], 1.0, 1e-12, "layer1 vmean[0]=1");
    CHECK_NEAR(opt.get_vmean(d2, 0)[0][0], 4.0, 1e-12, "layer2 vmean[0]=4");
}

// ---------------------------------------------------------------------------
// T13: Determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << "== determinism ==\n";
    auto run = []() {
        Model m;
        Dense* d = new Dense(3, 4);
        m.add_layer(d);
        auto p = d->parameters();
        auto g = d->gradients();
        for (auto* pp : p) {
            for (size_t r = 0; r < pp->rows; ++r)
                for (size_t c = 0; c < pp->cols; ++c)
                    (*pp)[r][c] = std::sin(r * 7.3 + c * 1.7) * 0.3;
        }
        for (auto* gg : g) {
            for (size_t r = 0; r < gg->rows; ++r)
                for (size_t c = 0; c < gg->cols; ++c)
                    (*gg)[r][c] = std::cos(r * 5.1 + c * 3.3) * 0.5 + 0.1;
        }
        AdamMini opt(1e-2, 0.9, 0.999, 1e-8, 0.0);
        for (int s = 0; s < 5; ++s) opt.step(m);
        std::vector<double> snapshot;
        for (size_t r = 0; r < p[0]->rows && r < 2; ++r)
            for (size_t c = 0; c < p[0]->cols && c < 2; ++c)
                snapshot.push_back((*p[0])[r][c]);
        return snapshot;
    };
    auto a = run();
    auto b = run();
    bool same = (a.size() == b.size());
    for (size_t i = 0; i < a.size() && same; ++i)
        if (a[i] != b[i]) same = false;
    CHECK(same, "two fresh instances produce bit-exact trajectory");
}

// ---------------------------------------------------------------------------
// T14: End-to-end training loss reduction
// ---------------------------------------------------------------------------
static void test_end_to_end() {
    std::cout << "== end-to-end training (y=2x) ==\n";
    Model model;
    Dense* dense = new Dense(1, 1);
    model.add_layer(dense);
    auto p = dense->parameters();
    // Initialize weight to 0.5 (target is 2.0)
    (*p[0])[0][0] = 0.5;
    (*p[1])[0][0] = 0.0;

    Tensor x(4, 1);
    Tensor y(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        x[i][0] = static_cast<double>(i) + 1.0;
        y[i][0] = 2.0 * x[i][0];
    }

    AdamMini opt(0.05, 0.9, 0.999, 1e-8, 0.0);

    double initial_loss = 0;
    {
        Tensor pred = dense->forward(x);
        double s = 0;
        for (size_t i = 0; i < 4; ++i) {
            double d = pred[i][0] - y[i][0];
            s += d * d;
        }
        initial_loss = s / 4.0;
    }

    for (int step = 0; step < 200; ++step) {
        Tensor pred = dense->forward(x);
        Tensor grad_out(4, 1);
        for (size_t i = 0; i < 4; ++i)
            grad_out[i][0] = 2.0 * (pred[i][0] - y[i][0]) / 4.0;
        dense->backward(grad_out, 0.0);
        opt.step(model);
    }

    Tensor pred = dense->forward(x);
    double final_loss = 0;
    for (size_t i = 0; i < 4; ++i) {
        double d = pred[i][0] - y[i][0];
        final_loss += d * d;
    }
    final_loss /= 4.0;

    CHECK(initial_loss > 1.0, "initial loss is non-trivial");
    CHECK(final_loss < 0.1, "final loss is small (< 0.1)");
    double reduction = (initial_loss - final_loss) / initial_loss;
    CHECK(reduction > 0.9, "loss reduced by >90%");
}

// ---------------------------------------------------------------------------
// T15: Single Dense layer step doesn't crash
// ---------------------------------------------------------------------------
static void test_empty_layer() {
    std::cout << "== single Dense step ==\n";
    Model model;
    Dense* dense = new Dense(2, 2);
    model.add_layer(dense);
    auto p = dense->parameters();
    auto g = dense->gradients();
    for (auto* pp : p) pp->fill(0.0);
    for (auto* gg : g) gg->fill(0.1);

    AdamMini opt(0.01, 0.9, 0.999, 1e-8, 0.0);
    opt.step(model);
    CHECK(true, "step on single Dense doesn't crash");
}

// ---------------------------------------------------------------------------
// T16: handles_weight_decay returns true
// ---------------------------------------------------------------------------
static void test_handles_weight_decay() {
    std::cout << "== handles_weight_decay ==\n";
    AdamMini opt;
    CHECK(opt.handles_weight_decay() == true, "handles_weight_decay returns true");
}

// ---------------------------------------------------------------------------
// T17: Signature vs Adam (different trajectories under same gradients)
// ---------------------------------------------------------------------------
static void test_signature_vs_adam() {
    std::cout << "== signature vs Adam ==\n";
    Model m1, m2;
    Dense* d1 = new Dense(3, 4);
    Dense* d2 = new Dense(3, 4);
    m1.add_layer(d1);
    m2.add_layer(d2);
    auto p1 = d1->parameters();
    auto p2 = d2->parameters();
    auto g1 = d1->gradients();
    auto g2 = d2->gradients();
    for (size_t i = 0; i < p1.size(); ++i) {
        for (size_t r = 0; r < p1[i]->rows; ++r)
            for (size_t c = 0; c < p1[i]->cols; ++c) {
                double v = std::sin(r * 7.3 + c * 1.7) * 0.3;
                (*p1[i])[r][c] = v;
                (*p2[i])[r][c] = v;
            }
    }
    for (auto* gg : g1) {
        for (size_t r = 0; r < gg->rows; ++r)
            for (size_t c = 0; c < gg->cols; ++c)
                (*gg)[r][c] = std::cos(r * 5.1 + c * 3.3) * 0.5 + 0.1;
    }
    for (auto* gg : g2) {
        for (size_t r = 0; r < gg->rows; ++r)
            for (size_t c = 0; c < gg->cols; ++c)
                (*gg)[r][c] = std::cos(r * 5.1 + c * 3.3) * 0.5 + 0.1;
    }

    AdamMini opt_mini(0.01, 0.9, 0.999, 1e-8, 0.0);
    Adam opt_adam(0.01, 0.9, 0.999, 1e-8);

    opt_mini.step(m1);
    opt_adam.step(m2);

    double dist = 0;
    size_t n = 0;
    for (size_t i = 0; i < p1.size(); ++i) {
        for (size_t r = 0; r < p1[i]->rows; ++r)
            for (size_t c = 0; c < p1[i]->cols; ++c) {
                double dd = (*p1[i])[r][c] - (*p2[i])[r][c];
                dist += dd * dd;
                ++n;
            }
    }
    dist = std::sqrt(dist / n);
    CHECK(dist > 1e-4, "AdamMini produces a different trajectory than Adam");
}

// ---------------------------------------------------------------------------
// T18: Per-parameter block-mode override
// ---------------------------------------------------------------------------
static void test_param_block_mode_override() {
    std::cout << "== per-param block-mode override ==\n";
    Model model;
    Dense* dense = new Dense(3, 4);
    model.add_layer(dense);
    auto p = dense->parameters();
    auto g = dense->gradients();
    for (auto* pp : p) pp->fill(0.0);
    g[0]->fill(1.0);
    g[1]->fill(1.0);

    AdamMini opt(1.0, 0.0, 0.0, 1.0, 0.0, AdamMini::BlockMode::AUTO);
    opt.set_param_block_mode(dense, 0, AdamMini::BlockMode::SCALAR);
    opt.step(model);

    CHECK(opt.get_block_mode(dense, 0) == AdamMini::BlockMode::SCALAR, "weight overridden to SCALAR");
    CHECK(opt.get_block_mode(dense, 1) == AdamMini::BlockMode::FULL, "bias auto → FULL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_defaults();
    test_custom_constructor();
    test_setter_validation();
    test_full_mode_closed_form();
    test_row_mean_mode_closed_form();
    test_row_mean_distinguishes_rows();
    test_row_mean_within_row_reduction();
    test_scalar_mode_closed_form();
    test_bias_correction_step2();
    test_state_shapes();
    test_weight_decay();
    test_zero_grad();
    test_multi_layer();
    test_determinism();
    test_end_to_end();
    test_empty_layer();
    test_handles_weight_decay();
    test_signature_vs_adam();
    test_param_block_mode_override();
    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
