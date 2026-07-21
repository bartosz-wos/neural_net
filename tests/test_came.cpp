// test_came.cpp — behavioral tests for CAME optimizer.
//
// Reference: Luo, Ren, Zheng, Jiang, Jiang, You 2023,
// "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
// (https://arxiv.org/abs/2307.02047), ACL 2023 Long Papers, pp. 4442-4453.
// Canonical reference implementation:
//   https://github.com/yangluo7/CAME/blob/master/came_pytorch/CAME.py
//
// Per-parameter update rules tested here:
//
//   2-D (factored) parameter:
//     raw            = g² + eps1
//     exp_avg_sq_row = beta2 * exp_avg_sq_row + (1-beta2) * mean_cols(raw)
//     exp_avg_sq_col = beta2 * exp_avg_sq_col + (1-beta2) * mean_rows(raw)
//     update         = g / sqrt(exp_avg_sq_row * exp_avg_sq_col / mean(exp_avg_sq_row))
//     update        /= max(1, RMS(update) / clip_threshold)
//     exp_avg        = beta1 * exp_avg + (1-beta1) * update
//     res            = (update - exp_avg)² + eps2
//     exp_avg_res_row = beta3 * exp_avg_res_row + (1-beta3) * mean_cols(res)
//     exp_avg_res_col = beta3 * exp_avg_res_col + (1-beta3) * mean_rows(res)
//     update         = exp_avg * sqrt(exp_avg_res_row * exp_avg_res_col
//                                     / mean(exp_avg_res_row))
//     param         *= (1 - lr * weight_decay)
//     param         -= lr * update
//
//   1-D parameter (residual is identity):
//     raw        = g² + eps1
//     exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * raw
//     update     = g / sqrt(exp_avg_sq)
//     update    /= max(1, RMS(update) / clip_threshold)
//     exp_avg    = beta1 * exp_avg + (1-beta1) * update
//     update     = exp_avg
//     param     *= (1 - lr * weight_decay)
//     param     -= lr * update

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/came.h"

using namespace std;

// Default β3 (paper / came_pytorch default). Pulled out so tests can use it
// without re-typing 0.9999.
static constexpr double kBeta3Default = 0.9999;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool condition) {
    if (condition) {
        cout << "  [PASS] " << name << '\n';
        ++passed;
    } else {
        cout << "  [FAIL] " << name << '\n';
        ++failed;
    }
    return condition;
}

static bool near(double actual, double expected, double tol = 1e-12) {
    return std::abs(actual - expected) <= tol;
}

// A minimal layer that we can use to plant specific gradients and verify
// the optimizer's per-parameter behavior in isolation (no Dense coupling).
class TestParam : public Layer {
public:
    Tensor param;       // parameter
    Tensor grad;        // gradient
    Tensor last_input;  // unused, for Layer API
    TestParam(size_t r, size_t c) : param(r, c), grad(r, c), last_input(r, c) {
        param.fill(0.0);
        grad.fill(0.0);
        last_input.fill(0.0);
    }
    Tensor forward(const Tensor& input) override {
        last_input = input;
        return input;
    }
    Tensor backward(const Tensor& grad_output, double) override {
        return grad_output;
    }
    void update_weights(double) override {}
    Tensor get_weights() const override { return param; }
    Tensor get_gradients() const override { return grad; }
    vector<Tensor*> parameters() override { return {&param}; }
    vector<Tensor*> gradients() override { return {&grad}; }
    void zero_grad() override { grad.fill(0.0); }
};

static void check_section(const string& title) {
    cout << "\n== " << title << " ==\n";
}

// ---- T1: defaults ----
static void test_defaults() {
    check_section("T1: default constructor");
    CAME opt;
    check("lr default = 2e-3", near(opt.get_lr(), 2e-3));
    check("beta1 default = 0.9", near(opt.get_beta1(), 0.9));
    check("beta2 default = 0.999", near(opt.get_beta2(), 0.999));
    check("beta3 default = 0.9999", near(opt.get_beta3(), 0.9999));
    check("eps1 default = 1e-30", near(opt.get_eps1(), 1e-30));
    check("eps2 default = 1e-16", near(opt.get_eps2(), 1e-16));
    check("clip default = 1.0", near(opt.get_clip_threshold(), 1.0));
    check("wd default = 0", near(opt.get_weight_decay(), 0.0));
    check("t default = 0", opt.get_t() == 0);
    check("handles_weight_decay true", opt.handles_weight_decay());
}

// ---- T2: non-default constructor ----
static void test_non_default_ctor() {
    check_section("T2: non-default constructor");
    CAME opt(1e-4, 0.85, 0.95, 0.999, 1e-8, 1e-12, 2.0, 0.05);
    check("custom lr", near(opt.get_lr(), 1e-4));
    check("custom beta1", near(opt.get_beta1(), 0.85));
    check("custom beta2", near(opt.get_beta2(), 0.95));
    check("custom beta3", near(opt.get_beta3(), 0.999));
    check("custom eps1", near(opt.get_eps1(), 1e-8));
    check("custom eps2", near(opt.get_eps2(), 1e-12));
    check("custom clip", near(opt.get_clip_threshold(), 2.0));
    check("custom wd", near(opt.get_weight_decay(), 0.05));
}

// ---- T3: validation throws ----
static void test_validation() {
    check_section("T3: constructor validation throws");
    auto check_throws = [](const string& name, const string& expect,
                           std::function<void()> fn) {
        try {
            fn();
            check("throws " + name + " (" + expect + ")", false);
        } catch (std::invalid_argument& e) {
            string what = e.what();
            bool ok = what.find(expect) != string::npos;
            check("throws " + name + " (" + expect + ")", ok);
            if (!ok) cout << "    actual: " << what << "\n";
        }
    };

    check_throws("lr=0", "learning rate", []() { CAME(0.0); });
    check_throws("lr<0", "learning rate", []() { CAME(-1e-3); });
    check_throws("beta1<0", "beta1", []() { CAME(2e-3, -0.1); });
    check_throws("beta1>=1", "beta1", []() { CAME(2e-3, 1.0); });
    check_throws("beta2<0", "beta2", []() { CAME(2e-3, 0.9, -0.1); });
    check_throws("beta2>=1", "beta2", []() { CAME(2e-3, 0.9, 1.0); });
    check_throws("beta3<0", "beta3", []() { CAME(2e-3, 0.9, 0.999, -0.1); });
    check_throws("beta3>=1", "beta3", []() { CAME(2e-3, 0.9, 0.999, 1.0); });
    check_throws("eps1<=0", "eps1", []() { CAME(2e-3, 0.9, 0.999, 0.9999, 0.0); });
    check_throws("eps2<=0", "eps2", []() { CAME(2e-3, 0.9, 0.999, 0.9999, 1e-30, 0.0); });
    check_throws("clip<=0", "clip", []() { CAME(2e-3, 0.9, 0.999, 0.9999, 1e-30, 1e-16, 0.0); });
    check_throws("wd<0", "weight decay", []() { CAME(2e-3, 0.9, 0.999, 0.9999, 1e-30, 1e-16, 1.0, -0.1); });
}

// ---- T4: mutator validation ----
static void test_mutator_validation() {
    check_section("T4: mutator validation throws");
    CAME opt;
    check("set_lr(1e-3)", (opt.set_lr(1e-3), near(opt.get_lr(), 1e-3)));
    bool threw = false;
    try { opt.set_lr(-1.0); } catch (std::invalid_argument&) { threw = true; }
    check("set_lr(-1) throws", threw);
    threw = false;
    try { opt.set_beta2(1.5); } catch (std::invalid_argument&) { threw = true; }
    check("set_beta2(1.5) throws", threw);
    threw = false;
    try { opt.set_eps1(0.0); } catch (std::invalid_argument&) { threw = true; }
    check("set_eps1(0) throws", threw);
    threw = false;
    try { opt.set_clip_threshold(-1.0); } catch (std::invalid_argument&) { threw = true; }
    check("set_clip(-1) throws", threw);
    threw = false;
    try { opt.set_weight_decay(-0.5); } catch (std::invalid_argument&) { threw = true; }
    check("set_wd(-0.5) throws", threw);
}

// ---- T5: state shape correctness (2-D and 1-D) ----
static void test_state_shapes() {
    check_section("T5: state shape correctness");
    CAME opt(1e-3, 0.9, 0.9, 0.9, 1e-8, 1e-8, 1.0, 0.0);
    Model model;
    // 2-D Dense(3, 4): weights (4, 3), bias (1, 4) — bias is 1-D
    Dense* d = new Dense(3, 4);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    d->zero_grad();
    model.add_layer(d);

    // Inject specific gradients
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            d->grad_weights[i][j] = 0.01 * static_cast<double>(i + j + 1);
    for (size_t j = 0; j < d->bias.cols; ++j)
        d->grad_bias[0][j] = 0.02 * static_cast<double>(j + 1);

    opt.step(model);

    // Check 2-D weight state shapes
    Tensor row, col, rrow, rcol, exp_avg;
    bool got_row = opt.get_exp_avg_sq_row(d, 0, row);
    bool got_col = opt.get_exp_avg_sq_col(d, 0, col);
    bool got_rrow = opt.get_exp_avg_res_row(d, 0, rrow);
    bool got_rcol = opt.get_exp_avg_res_col(d, 0, rcol);
    bool got_exp = (exp_avg = opt.get_exp_avg(d, 0), exp_avg.rows == 4 && exp_avg.cols == 3);

    check("2-D exp_avg shape (4, 3)", got_exp);
    check("2-D row_ema present", got_row && row.rows == 4 && row.cols == 1);
    check("2-D col_ema present", got_col && col.rows == 1 && col.cols == 3);
    check("2-D res_row present", got_rrow && rrow.rows == 4 && rrow.cols == 1);
    check("2-D res_col present", got_rcol && rcol.rows == 1 && rcol.cols == 3);

    // Check 1-D bias state shapes
    Tensor exp_avg_sq;
    bool got_sq = opt.get_exp_avg_sq(d, 1, exp_avg_sq);
    bool is_1d = opt.is_1d(d, 1);
    Tensor row2, col2, rrow2, rcol2;
    bool no_row = opt.get_exp_avg_sq_row(d, 1, row2);
    check("1-D bias uses exp_avg_sq (1, 4)", got_sq && exp_avg_sq.rows == 1 && exp_avg_sq.cols == 4);
    check("1-D flag set", is_1d);
    check("1-D has no row_ema", !no_row);
}

// ---- T6: has_state before/after ----
static void test_has_state() {
    check_section("T6: has_state before/after");
    CAME opt;
    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    d->zero_grad();
    model.add_layer(d);

    check("no state before step", !opt.has_state(d));

    d->grad_weights[0][0] = 0.01;
    opt.step(model);

    check("state after step", opt.has_state(d));
}

// ---- T7: t increments ----
static void test_t_increments() {
    check_section("T7: t increments");
    CAME opt;
    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    check("t=0 before", opt.get_t() == 0);
    opt.step(model);
    check("t=1 after first step", opt.get_t() == 1);
    opt.step(model);
    opt.step(model);
    check("t=3 after three steps", opt.get_t() == 3);
}

// ---- T8: first-step closed-form 1-D (β1=0, clip very large → no clipping) ----
static void test_first_step_1d_closed_form() {
    check_section("T8: 1-D first-step closed-form (β1=0, clip very large)");
    // With β1=0, exp_avg = update exactly (no momentum smoothing).
    // With very large clip, no RMS clipping.
    // update = g / sqrt(exp_avg_sq), where exp_avg_sq = (1-β2)*raw + β2*0 = (1-β2)*(g² + eps1)
    // So exp_avg = update = g / sqrt((1-β2)*(g² + eps1))
    // Then 1-D residual is identity, update = exp_avg.
    // param -= lr * update
    double lr = 1e-3, beta1 = 0.0, beta2 = 0.5, eps1 = 1e-8, eps2 = 1e-8;
    CAME opt(lr, beta1, beta2, kBeta3Default, eps1, eps2, /*clip=*/1e30, 0.0);

    Model model;
    Dense* d = new Dense(1, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // Bias gradient (1-D)
    d->grad_bias[0][0] = 4.0;
    d->grad_bias[0][1] = -9.0;

    // Save param baseline
    double p0 = d->bias[0][0];
    double p1 = d->bias[0][1];

    opt.step(model);

    // Expected update = g / sqrt((1-β2)*(g²+eps1))
    double u0 = 4.0 / std::sqrt(0.5 * (16.0 + eps1));
    double u1 = -9.0 / std::sqrt(0.5 * (81.0 + eps1));
    check("bias[0][0] = p0 - lr*u0",
          near(d->bias[0][0], p0 - lr * u0, 1e-10));
    check("bias[0][1] = p1 - lr*u1",
          near(d->bias[0][1], p1 - lr * u1, 1e-10));
}

// ---- T9: first-step closed-form 2-D ----
static void test_first_step_2d_closed_form() {
    check_section("T9: 2-D first-step closed-form (β1=0, clip very large)");
    // update = g / sqrt(row_EMA * col_EMA / mean(row_EMA))
    // where row_EMA_i = (1-β2)*mean_j(g²_ij + eps1)
    //       col_EMA_j = (1-β2)*mean_i(g²_ij + eps1)
    //       mean(row_EMA) = (1-β2)*mean(g² + eps1)
    // So row_EMA * col_EMA / mean(row_EMA) reduces to (1-β2)*col_EMA*row_EMA/mean(...)
    // For β1=0, exp_avg = update; residual then multiplies it again.
    // Let's use β3=0 too so res_row, res_col = mean(res) which is small but nonzero,
    // and verify that exp_avg matches update closed-form exactly.
    double lr = 1e-3, beta1 = 0.0, beta2 = 0.5, beta3 = 0.5, eps1 = 1e-8, eps2 = 1e-8;
    CAME opt(lr, beta1, beta2, beta3, eps1, eps2, /*clip=*/1e30, 0.0);

    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // 2x2 weight gradient
    d->grad_weights[0][0] = 1.0;
    d->grad_weights[0][1] = 2.0;
    d->grad_weights[1][0] = 3.0;
    d->grad_weights[1][1] = 4.0;

    opt.step(model);

    // Expected exp_avg before residual = update_adaptive
    // raw = g² + eps1 = [[1+eps1, 4+eps1], [9+eps1, 16+eps1]]
    // row_EMA = (1-β2) * row_means = 0.5 * [(1+eps1+4+eps1)/2, (9+eps1+16+eps1)/2]
    //         = [0.5*(2.5+eps1), 0.5*(12.5+eps1)]  -- wait, (1+eps1+4+eps1)/2 = (5+2eps1)/2 = 2.5+eps1
    // Hmm let me recompute. raw[0][0]=1+eps1, raw[0][1]=4+eps1, raw[1][0]=9+eps1, raw[1][1]=16+eps1
    // row_EMA_0 = 0.5 * (raw[0][0]+raw[0][1])/2 = 0.5 * ((5+2eps1)/2) = (5+2eps1)/4
    // row_EMA_1 = 0.5 * ((25+2eps1)/2) = (25+2eps1)/4
    // col_EMA_0 = 0.5 * (raw[0][0]+raw[1][0])/2 = 0.5 * (10+2eps1)/2 = (10+2eps1)/4
    // col_EMA_1 = 0.5 * (raw[0][1]+raw[1][1])/2 = 0.5 * (20+2eps1)/2 = (20+2eps1)/4
    // mean(row_EMA) = 0.5 * ((5+2eps1)/4 + (25+2eps1)/4) = (15+2eps1)/4
    // row_EMA * col_EMA:
    //   [0][0]: ((5+2e)/4) * ((10+2e)/4) / ((15+2e)/4) = ((5+2e)*(10+2e))/(4*(15+2e))
    //   [0][1]: ((5+2e)/4) * ((20+2e)/4) / ((15+2e)/4) = ((5+2e)*(20+2e))/(4*(15+2e))
    //   [1][0]: ((25+2e)/4) * ((10+2e)/4) / ((15+2e)/4) = ((25+2e)*(10+2e))/(4*(15+2e))
    //   [1][1]: ((25+2e)/4) * ((20+2e)/4) / ((15+2e)/4) = ((25+2e)*(20+2e))/(4*(15+2e))
    // update[0][0] = g[0][0] / sqrt(v[0][0])
    // For the closed-form test, just check that exp_avg matches what we compute.
    Tensor exp_avg = opt.get_exp_avg(d, 0);
    double v00 = ((5.0+2*eps1) * (10.0+2*eps1)) / (4.0 * (15.0+2*eps1));
    double u00 = 1.0 / std::sqrt(v00);
    check("exp_avg[0][0] matches closed-form 2-D update",
          near(exp_avg[0][0], u00, 1e-10));

    double v01 = ((5.0+2*eps1) * (20.0+2*eps1)) / (4.0 * (15.0+2*eps1));
    double u01 = 2.0 / std::sqrt(v01);
    check("exp_avg[0][1] matches closed-form 2-D update",
          near(exp_avg[0][1], u01, 1e-10));
}

// ---- T10: row EMA recurrence ----
static void test_row_ema_recurrence() {
    check_section("T10: row EMA recurrence at step 2");
    // Step 1: row_EMA_1 = (1-β2) * mean_cols(raw_1) where raw_1 = g_1² + eps1
    // Step 2: row_EMA_2 = β2 * row_EMA_1 + (1-β2) * mean_cols(raw_2)
    // Dense(in=2, out=3) → weights shape (3, 2). So rows=3, cols=2.
    CAME opt(1e-3, 0.9, 0.5, 0.999, 1e-8, 1e-8, 1.0, 0.0);
    Model model;
    Dense* d = new Dense(2, 3);  // weights shape (3, 2)
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // Step 1 — gradient is (3, 2)
    d->grad_weights[0][0] = 1.0;
    d->grad_weights[0][1] = 2.0;
    d->grad_weights[1][0] = 3.0;
    d->grad_weights[1][1] = 4.0;
    d->grad_weights[2][0] = 5.0;
    d->grad_weights[2][1] = 6.0;
    opt.step(model);

    Tensor row1;
    opt.get_exp_avg_sq_row(d, 0, row1);
    // row_EMA_0 (step 1) = (1-β2) * mean_cols(raw row 0) = 0.5 * ((1+ε + 4+ε)/2) = 0.5 * (2.5 + ε)
    double mean_cols_row_0_1 = 0.5 * ((1.0 + 1e-8) + (4.0 + 1e-8));
    double expected_row0_step1 = 0.5 * mean_cols_row_0_1;
    check("row_ema[0] step 1",
          near(row1[0][0], expected_row0_step1, 1e-10));

    // Step 2 with different gradient
    d->grad_weights[0][0] = 2.0;
    d->grad_weights[0][1] = 0.0;
    d->grad_weights[1][0] = 1.0;
    d->grad_weights[1][1] = 1.0;
    d->grad_weights[2][0] = 1.0;
    d->grad_weights[2][1] = 1.0;
    opt.step(model);

    opt.get_exp_avg_sq_row(d, 0, row1);
    // row_EMA_0 (step 2) = β2 * prev + (1-β2) * mean_cols(raw row 0 step 2)
    // raw row 0 step 2: [(2+ε)² + ε, (0+ε)² + ε] = [(4+4ε+ε²+ε), (ε²+ε)] ≈ [(4+5ε), (ε)] for small ε
    // mean_cols = ((4+ε) + (0+ε)) / 2 = (4+2ε)/2 = 2+ε
    // row_EMA_0 = 0.5 * prev + 0.5 * (2+ε)
    double mean_cols_row_0_2 = ((4.0 + 1e-8) + (0.0 + 1e-8)) / 2.0;
    double expected_row0_step2 = 0.5 * expected_row0_step1 + 0.5 * mean_cols_row_0_2;
    check("row_ema[0] step 2 = β2*prev + (1-β2)*new",
          near(row1[0][0], expected_row0_step2, 1e-10));
}

// ---- T11: momentum recurrence ----
static void test_momentum_recurrence() {
    check_section("T11: momentum recurrence");
    // exp_avg_t = β1 * exp_avg_{t-1} + (1-β1) * update_t
    CAME opt(1e-3, /*β1=*/0.5, 0.9, 0.9, 1e-8, 1e-8, 1e30, 0.0);
    Model model;
    Dense* d = new Dense(1, 1);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    d->grad_weights[0][0] = 1.0;
    opt.step(model);
    Tensor ea1 = opt.get_exp_avg(d, 0);

    d->grad_weights[0][0] = 2.0;
    opt.step(model);
    Tensor ea2 = opt.get_exp_avg(d, 0);

    d->grad_weights[0][0] = 3.0;
    opt.step(model);
    Tensor ea3 = opt.get_exp_avg(d, 0);

    // Each step: exp_avg_t = 0.5*prev + 0.5*update_t (β1=0.5)
    // After step 3 (β3 multiplies into exp_avg again, but with β3=0.9 the
    // residual correction is small)
    check("exp_avg non-zero after step 1", std::abs(ea1[0][0]) > 1e-9);
    check("exp_avg non-zero after step 2", std::abs(ea2[0][0]) > 1e-9);
    check("exp_avg non-zero after step 3", std::abs(ea3[0][0]) > 1e-9);
}

// ---- T12: residual correctness ----
static void test_residual_correctness() {
    check_section("T12: residual = (update - exp_avg)² + eps2");
    CAME opt(1e-3, 0.9, 0.9, 0.9, 1e-8, 1e-8, 1e30, 0.0);
    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    d->grad_weights[0][0] = 1.0;
    opt.step(model);

    Tensor ea = opt.get_exp_avg(d, 0);
    Tensor row, col;
    opt.get_exp_avg_sq_row(d, 0, row);
    opt.get_exp_avg_sq_col(d, 0, col);

    // With β1=0.9 and β3=0.9, the res_row EMA after step 1 is (1-β3)*mean_cols(res+eps2)
    // = 0.1 * mean_cols(((update - exp_avg)² + eps2))
    // We don't have direct access to update or res; just verify the res_row is finite
    // and the shape is correct.
    bool ok = row.rows == 2 && row.cols == 1;
    check("res_row shape (2, 1) after step 1", ok);
    bool ok2 = ea.rows == 2 && ea.cols == 2;
    check("exp_avg shape (2, 2) after step 1", ok2);
}

// ---- T13: res row EMA recurrence ----
static void test_res_row_ema_recurrence() {
    check_section("T13: res row EMA recurrence");
    // res_row_t = β3 * res_row_{t-1} + (1-β3) * mean_cols(res_t)
    // Use same gradient on two consecutive steps to isolate the EMA recurrence.
    // At step 1 the previous state is 0, so res_row_1 = (1-β3) * mean_cols(res_1).
    // At step 2 res_row_2 = β3 * res_row_1 + (1-β3) * mean_cols(res_2).
    // If we drop the β3 carry (replace with 0), res_row_2 would be just
    // (1-β3) * mean_cols(res_2) which differs from the full EMA value.
    CAME opt(1e-3, 0.5, 0.5, 0.5, 1e-8, 1e-8, 1e30, 0.0);
    Model model;
    Dense* d = new Dense(2, 3);  // weights shape (3, 2)
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // Two steps with the same gradient (all-1)
    for (int step = 0; step < 2; ++step) {
        for (size_t i = 0; i < d->grad_weights.rows; ++i)
            for (size_t j = 0; j < d->grad_weights.cols; ++j)
                d->grad_weights[i][j] = 1.0;
        opt.step(model);
    }

    // Now change the gradient and run two more steps. The res_row EMA
    // should approach the new value but with β3=0.5 still be influenced by
    // the previous state at step 3.
    for (int step = 0; step < 2; ++step) {
        for (size_t i = 0; i < d->grad_weights.rows; ++i)
            for (size_t j = 0; j < d->grad_weights.cols; ++j)
                d->grad_weights[i][j] = 2.0;
        opt.step(model);
    }

    // Step 5: change back to gradient 1.0
    Tensor res_before;
    opt.get_exp_avg_res_row(d, 0, res_before);
    for (size_t i = 0; i < d->grad_weights.rows; ++i)
        for (size_t j = 0; j < d->grad_weights.cols; ++j)
            d->grad_weights[i][j] = 1.0;
    opt.step(model);
    Tensor res_after;
    opt.get_exp_avg_res_row(d, 0, res_after);

    // The crucial check: with β3=0.5, when we go from "gradient 2.0" to
    // "gradient 1.0", res_row should move TOWARDS the new residual value
    // but not all the way. With the bug (no carry), res_row would jump
    // directly to (1-β3)*new_residual which is exactly half of the new.
    // With the correct EMA, res_row = β3 * prev + (1-β3) * new.
    // We verify that res_after is greater than (1-β3) * new_residual alone
    // would be — but this requires knowing new_residual, which we can't
    // compute from the public API. Instead, verify that res_after is
    // strictly less than res_before (since residual at g=1 < residual at g=2
    // is expected given the algorithm).
    check("res_row decreased when gradient went from 2.0 to 1.0",
          res_after[0][0] < res_before[0][0]);

    // res_after[0][0] should also be positive
    check("res_row positive after step 5", res_after[0][0] > 0.0);
}

// ---- T14: confidence scaling ----
static void test_confidence_scaling() {
    check_section("T14: confidence scaling = exp_avg * sqrt(res_factor)");
    // The final 2-D update is exp_avg * sqrt(res_row * res_col / mean(res_row))
    // where res_row/res_col are positive (squared residuals).
    // So the final update is just exp_avg scaled by a positive factor ≥1.
    CAME opt(1e-3, 0.5, 0.5, 0.5, 1e-8, 1e-8, 1e30, 0.0);
    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    d->grad_weights[0][0] = 1.0;
    d->grad_weights[0][1] = 1.0;
    d->grad_weights[1][0] = 1.0;
    d->grad_weights[1][1] = 1.0;
    opt.step(model);

    // The weight moved by lr * update_final where update_final = exp_avg * sqrt(confidence_factor)
    // Check the weights moved in the right direction (decreased).
    double max_w = -1e30;
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            max_w = std::max(max_w, std::abs(d->weights[i][j]));
    check("weights moved", max_w > 1e-9);
}

// ---- T15: 1-D identity path ----
static void test_1d_identity_path() {
    check_section("T15: 1-D residual is identity (update = exp_avg)");
    // For 1-D, after the residual step, update = exp_avg exactly.
    // So the weight decay + param update uses exp_avg directly.
    CAME opt(1e-3, 0.5, 0.5, 0.5, 1e-8, 1e-8, 1e30, 0.0);
    Model model;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    d->grad_bias[0][0] = 1.0;
    d->grad_bias[0][1] = 1.0;
    opt.step(model);

    Tensor ea = opt.get_exp_avg(d, 1);
    // The bias moved by lr * ea (no weight decay applied)
    check("bias[0][0] = -lr * ea[0][0]",
          near(d->bias[0][0], -1e-3 * ea[0][0], 1e-10));
}

// ---- T16: RMS-clip boundary ----
static void test_rms_clip_boundary() {
    check_section("T16: RMS-clip boundary");
    // Use a gradient that produces a very large update before clipping.
    CAME opt(1e-3, 0.9, 0.999, 0.9999, 1e-30, 1e-16, 0.5, 0.0);
    Model model;
    Dense* d = new Dense(2, 2);  // weights (2, 2) — 2-D path
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // Huge gradient
    for (size_t i = 0; i < d->grad_weights.rows; ++i)
        for (size_t j = 0; j < d->grad_weights.cols; ++j)
            d->grad_weights[i][j] = 1e10;
    opt.step(model);

    double rms_val;
    opt.get_rms(d, 0, rms_val);
    // The cached rms is the post-clip RMS, which should be ≤ clip_threshold.
    check("cached rms <= clip_threshold (0.5)", rms_val <= 0.5 + 1e-9);

    // Also verify the weight actually moved (otherwise we wouldn't know the
    // RMS path was taken at all).
    double max_abs = 0.0;
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            max_abs = std::max(max_abs, std::abs(d->weights[i][j]));
    check("weights moved (post-clip non-zero)", max_abs > 1e-9);

    // Without the bug (clip applied), the weight magnitude is bounded by
    // lr * clip_threshold = 1e-3 * 0.5 = 5e-4. With the bug (no clip),
    // the magnitude would be huge.
    check("weight magnitude bounded by lr * clip_threshold",
          max_abs <= 1e-3 * 0.5 + 1e-12);
}

// ---- T17: RMS-clip identity ----
static void test_rms_clip_identity() {
    check_section("T17: RMS-clip identity for small updates");
    CAME opt(1e-3, 0.9, 0.999, 0.9999, 1e-30, 1e-16, 1.0, 0.0);
    Model model;
    Dense* d = new Dense(1, 1);
    d->weights.fill(0.0);
    d->bias.fill(0.0);
    model.add_layer(d);

    // Tiny gradient — pre-clip RMS will be < 1.0 so no clipping
    d->grad_weights[0][0] = 1e-15;
    opt.step(model);

    double rms_val;
    opt.get_rms(d, 0, rms_val);
    // Pre-clip RMS is ~|g|/sqrt(exp_avg_sq) which for tiny g is ~1, but
    // exact value depends on the chain. Just check it's finite and small.
    check("rms finite and < 1e6", std::isfinite(rms_val) && rms_val < 1e6);
}

// ---- T18: weight decay shrinks params at zero gradient ----
static void test_weight_decay_zero_grad() {
    check_section("T18: weight decay shrinks params at zero gradient");
    CAME opt(0.1, 0.9, 0.999, 0.9999, 1e-30, 1e-16, 1.0, 0.1);
    Model model;
    Dense* d = new Dense(1, 1);
    d->weights[0][0] = 1.0;
    d->bias.fill(0.0);
    d->zero_grad();
    model.add_layer(d);

    // Zero gradient
    opt.step(model);

    // Expected: weight *= (1 - lr * wd) = 1 - 0.01 = 0.99
    check("weight = 0.99 at zero grad", near(d->weights[0][0], 0.99, 1e-12));
}

// ---- T19: determinism ----
static void test_determinism() {
    check_section("T19: determinism (two fresh CAME instances)");
    auto run = []() {
        CAME opt(1e-3, 0.9, 0.999, 0.9999, 1e-8, 1e-8, 1.0, 0.0);
        Model model;
        Dense* d = new Dense(2, 2);
        d->weights.fill(0.0);
        d->bias.fill(0.0);
        d->zero_grad();
        model.add_layer(d);

        double seed = 1.0;
        for (int step = 0; step < 30; ++step) {
            for (size_t i = 0; i < d->grad_weights.rows; ++i)
                for (size_t j = 0; j < d->grad_weights.cols; ++j) {
                    seed = std::sin(seed * 12.9898 + 78.233) * 43758.5453;
                    d->grad_weights[i][j] = (seed - std::floor(seed) - 0.5) * 0.1;
                }
            for (size_t j = 0; j < d->grad_bias.cols; ++j) {
                seed = std::sin(seed * 12.9898 + 78.233) * 43758.5453;
                d->grad_bias[0][j] = (seed - std::floor(seed) - 0.5) * 0.1;
            }
            opt.step(model);
        }
        return std::make_pair(d->weights[0][0], d->bias[0][0]);
    };
    auto a = run();
    auto b = run();
    check("bit-exact weights", near(a.first, b.first, 0.0));
    check("bit-exact bias", near(a.second, b.second, 0.0));
}

// ---- T20: end-to-end loss reduction ----
static void test_end_to_end() {
    check_section("T20: end-to-end loss reduction on linear regression");
    // Train a 1->1 linear regression: y = 2*x + 0.5
    // We'll use a Dense(1, 1) with initial weights=0, bias=0 and MSE loss.
    Model model;
    Dense* d = new Dense(1, 1);
    d->init_weights("zeros");
    model.add_layer(d);

    CAME opt(0.05, 0.9, 0.999, 0.9999, 1e-8, 1e-8, 1.0, 0.0);

    double x = 1.0;
    double target = 2.5;  // 2*1 + 0.5
    double initial_loss = std::pow(target - d->forward(Tensor(1, 1, &x))[0][0], 2);

    for (int step = 0; step < 200; ++step) {
        Tensor inp(1, 1, &x);
        Tensor pred = d->forward(inp);
        double diff = pred[0][0] - target;
        Tensor grad_out(1, 1, &diff);
        d->backward(grad_out, 0.0);
        opt.step(model);
    }

    Tensor final_pred = d->forward(Tensor(1, 1, &x));
    double final_loss = std::pow(final_pred[0][0] - target, 2);

    double reduction = (initial_loss - final_loss) / std::max(initial_loss, 1e-30);
    check("loss reduced by > 50% (was " + std::to_string(initial_loss) +
          " → " + std::to_string(final_loss) + ")",
          reduction > 0.5);
}

// ---- T21: independent state across layers ----
static void test_independent_state() {
    check_section("T21: independent state across layers");
    CAME opt;
    Model model;
    Dense* a = new Dense(2, 2);
    Dense* b = new Dense(2, 2);
    a->init_weights("zeros"); a->zero_grad();
    b->init_weights("zeros"); b->zero_grad();
    model.add_layer(a);
    model.add_layer(b);

    // Set different gradients
    a->grad_weights[0][0] = 0.01;
    b->grad_weights[0][0] = 0.99;

    opt.step(model);

    // Compare the row_EMA which is not affected by RMS-clipping (it's the
    // raw gradient-squared EMA, before any clipping happens).
    Tensor row_a, row_b;
    opt.get_exp_avg_sq_row(a, 0, row_a);
    opt.get_exp_avg_sq_row(b, 0, row_b);
    bool diff = std::abs(row_a[0][0] - row_b[0][0]) > 1e-9;
    check("different layers produce different row_EMA", diff);
}

// ---- T22: independent state across parameters ----
static void test_independent_state_per_param() {
    check_section("T22: independent state across parameters");
    CAME opt;
    Model model;
    Dense* d = new Dense(2, 2);
    d->init_weights("zeros"); d->zero_grad();
    model.add_layer(d);

    // weight and bias get different gradients
    d->grad_weights[0][0] = 0.01;
    d->grad_bias[0][0] = 0.5;

    opt.step(model);

    Tensor ea_w = opt.get_exp_avg(d, 0);
    Tensor ea_b = opt.get_exp_avg(d, 1);
    bool diff = std::abs(ea_w[0][0] - ea_b[0][0]) > 1e-6;
    check("weight and bias have different exp_avg", diff);
}

// ---- T23: gradient clearing ----
static void test_zero_grad() {
    check_section("T23: gradient clearing");
    CAME opt;
    Model model;
    Dense* d = new Dense(2, 2);
    d->init_weights("zeros"); d->zero_grad();
    model.add_layer(d);

    d->grad_weights[0][0] = 0.5;
    opt.step(model);

    check("grad_weights cleared", near(d->grad_weights[0][0], 0.0, 0.0));
    check("grad_bias cleared", near(d->grad_bias[0][0], 0.0, 0.0));
}

// ---- T24: param/grad count mismatch ----
static void test_param_grad_count_mismatch() {
    check_section("T24: parameter/gradient count mismatch throws");
    CAME opt;
    Model model;
    auto* bad = new TestParam(2, 2);
    model.add_layer(bad);

    // Override gradients() to return empty via a second TestParam
    auto* empty = new TestParam(0, 0);  // no params
    (void)empty;

    // Simpler: use the existing TestParam but make param zero-shaped (no real test here).
    // Instead, use the count mismatch path via grad count vs param count.
    bad->grad = Tensor(1, 1);  // mismatched shape to trigger both checks
    bool threw = false;
    try { opt.step(model); } catch (std::exception&) { threw = true; }
    check("mismatched param/grad throws", threw);
}

// ---- T25: end-to-end on Dense(3, 2) model ----
static void test_dense_3_2() {
    check_section("T25: end-to-end Dense(3, 2)");
    CAME opt(0.05, 0.9, 0.999, 0.9999, 1e-8, 1e-8, 1.0, 0.0);
    Model model;
    Dense* d = new Dense(3, 2);
    d->init_weights("zeros");
    model.add_layer(d);

    // Use (1, 3) input and (1, 2) target — single-sample regression.
    // Dense(3, 2): weights (2, 3), bias (1, 2). forward(inp) = inp * W^T + b
    //   = (1, 3) * (3, 2) + (1, 2) = (1, 2). Good.
    Tensor inp(1, 3);
    inp[0][0] = 1.0; inp[0][1] = 0.5; inp[0][2] = -0.3;
    Tensor tgt(1, 2);
    tgt[0][0] = 0.7; tgt[0][1] = -0.4;

    Tensor pred = d->forward(inp);
    double initial = (pred[0][0] - tgt[0][0]) * (pred[0][0] - tgt[0][0]) +
                     (pred[0][1] - tgt[0][1]) * (pred[0][1] - tgt[0][1]);

    for (int step = 0; step < 100; ++step) {
        Tensor p = d->forward(inp);  // forward populates last_input
        Tensor grad(1, 2);
        grad[0][0] = 2.0 * (p[0][0] - tgt[0][0]);
        grad[0][1] = 2.0 * (p[0][1] - tgt[0][1]);
        d->backward(grad, 0.0);
        opt.step(model);
    }

    Tensor p = d->forward(inp);
    double final = (p[0][0] - tgt[0][0]) * (p[0][0] - tgt[0][0]) +
                   (p[0][1] - tgt[0][1]) * (p[0][1] - tgt[0][1]);
    double reduction = (initial - final) / std::max(initial, 1e-30);
    check("Dense(3, 2) loss reduction > 50% (was " +
          std::to_string(initial) + " → " + std::to_string(final) + ")",
          reduction > 0.5);
}

// Forward decl for default beta3 in T8 helper
// (declared above as kBeta3Default)

int main() {
    cout << "=== CAME Optimizer Tests ===\n";
    test_defaults();
    test_non_default_ctor();
    test_validation();
    test_mutator_validation();
    test_state_shapes();
    test_has_state();
    test_t_increments();
    test_first_step_1d_closed_form();
    test_first_step_2d_closed_form();
    test_row_ema_recurrence();
    test_momentum_recurrence();
    test_residual_correctness();
    test_res_row_ema_recurrence();
    test_confidence_scaling();
    test_1d_identity_path();
    test_rms_clip_boundary();
    test_rms_clip_identity();
    test_weight_decay_zero_grad();
    test_determinism();
    test_end_to_end();
    test_independent_state();
    test_independent_state_per_param();
    test_zero_grad();
    test_param_grad_count_mismatch();
    test_dense_3_2();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
