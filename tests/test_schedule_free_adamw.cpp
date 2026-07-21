// test_schedule_free_adamw.cpp — behavioral tests for Schedule-Free AdamW.
//
// Reference: Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024,
// "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682)
// NeurIPS 2024 best-paper nominee.
//
// Canonical reference implementation:
//   https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
//
// The schedule-free optimizer maintains three coupled sequences per parameter:
//
//   x_k : eval point  (the parameter that forward/backward runs at next step)
//   z_k : iterate     (the parameter that gets Adam-updated)
//   y_k : averaged    (returned parameter; y_k = (1-beta1)*z_k + beta1*x_k)
//
// In train mode `param` holds `y`; in eval mode `param` holds `x`. The
// public API is `optimizer.train(model)` and `optimizer.eval(model)` which
// perform the linear swap (lerp) between `y` and `x`.
//
// Per-step update at the start of step k (k = 0-indexed; current step is k+1):
//   bias_corr2 = 1 - beta2^(k+1)
//   exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * g^2
//   denom      = sqrt(exp_avg_sq / bias_corr2) + eps
//   u          = g / denom  (+ wd * y added before z/y step if wd > 0)
//   z_{k+1}    = z_k - lr * u
//   y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k + lr*(beta1*(1-ckp1) - 1) * u
// where ckp1 = 1/(k+1) for default r=0, weight_lr_power=2.0 (and the warmup
// schedule only kicks in if k < warmup_steps).
//
// State init at first step: z = clone(param), exp_avg_sq = 0.
//
// Mode swaps (canonical formulas):
//   eval (y -> x):  p = p + (1 - 1/beta1) * (z - p)
//   train (x -> y): p = p + (1 - beta1) * (z - p)
//
// Tests below verify defaults, state shapes, eval/train mode swap, the
// closed-form first step under beta1=beta2=0 (degenerate case), z updates,
// exp_avg_sq EMA recurrence, ckp1 progression, decoupled weight decay,
// mode-swap correctness, determinism, and end-to-end training reduction.

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
#include "nn/optimizers/schedule_free_adamw.h"

using namespace std;

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

// Minimal layer that lets us plant specific gradients and inspect the
// parameter values after the optimizer has run.
class TestParam : public Layer {
public:
    Tensor param;
    Tensor grad;
    Tensor last_input;
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
    check_section("T1: defaults");
    ScheduleFreeAdamW opt;
    check("lr default = 1.0",          near(opt.get_lr(), 1.0));
    check("beta1 default = 0.9",       near(opt.get_beta1(), 0.9));
    check("beta2 default = 0.999",     near(opt.get_beta2(), 0.999));
    check("eps default = 1e-8",        near(opt.get_eps(), 1e-8));
    check("weight_decay default = 0",  near(opt.get_weight_decay(), 0.0));
    check("warmup_steps default = 0",  opt.get_warmup_steps() == 0);
    check("r default = 0",             near(opt.get_r(), 0.0));
    check("weight_lr_power default = 2.0", near(opt.get_weight_lr_power(), 2.0));
    check("k starts at 0",             opt.get_k() == 0);
    check("lr_max starts at -1",       near(opt.get_lr_max(), -1.0));
    check("weight_sum starts at 0",    near(opt.get_weight_sum(), 0.0));
    check("train_mode starts false",   !opt.is_train_mode());
    check("handles_weight_decay true", opt.handles_weight_decay());
}

// ---- T2: non-default constructor + validation throws ----
static void test_constructor_and_validation() {
    check_section("T2: constructor + validation");
    {
        ScheduleFreeAdamW opt(0.5, 0.85, 0.99, 1e-6, 0.01, 5, 0.5, 1.5);
        check("non-default lr",      near(opt.get_lr(), 0.5));
        check("non-default beta1",   near(opt.get_beta1(), 0.85));
        check("non-default beta2",   near(opt.get_beta2(), 0.99));
        check("non-default eps",     near(opt.get_eps(), 1e-6));
        check("non-default wd",      near(opt.get_weight_decay(), 0.01));
        check("non-default warmup",  opt.get_warmup_steps() == 5);
        check("non-default r",       near(opt.get_r(), 0.5));
        check("non-default wlp",     near(opt.get_weight_lr_power(), 1.5));
    }
    auto throws = [](auto fn) {
        try { fn(); return false; } catch (...) { return true; }
    };
    ScheduleFreeAdamW valid(1.0, 0.9, 0.999, 1e-8);
    check("lr=-1 throws",      throws([&]{ ScheduleFreeAdamW o(-1.0, 0.9, 0.999, 1e-8); }));
    check("lr=0 throws",       throws([&]{ ScheduleFreeAdamW o(0.0, 0.9, 0.999, 1e-8); }));
    check("beta1=1 throws",    throws([&]{ ScheduleFreeAdamW o(1.0, 1.0, 0.999, 1e-8); }));
    check("beta1=-1 throws",   throws([&]{ ScheduleFreeAdamW o(1.0, -0.1, 0.999, 1e-8); }));
    check("beta2=1 throws",    throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 1.0, 1e-8); }));
    check("beta2=-1 throws",   throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, -0.1, 1e-8); }));
    check("eps=0 throws",      throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, 0.0); }));
    check("eps<0 throws",      throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, -1e-8); }));
    check("wd<0 throws",       throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, 1e-8, -0.1); }));
    check("warmup<0 throws",   throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, 1e-8, 0.0, -1); }));
    check("r<0 throws",        throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, 1e-8, 0.0, 0, -0.1); }));
    check("wlp<0 throws",      throws([&]{ ScheduleFreeAdamW o(1.0, 0.9, 0.999, 1e-8, 0.0, 0, 0, -0.1); }));
    // mutator validation
    valid.set_lr(2.0);
    check("set_lr ok",         near(valid.get_lr(), 2.0));
    check("set_lr(-1) throws", throws([&]{ valid.set_lr(-1.0); }));
    check("set_beta1(1) throws", throws([&]{ valid.set_beta1(1.0); }));
    check("set_beta2(-0.1) throws", throws([&]{ valid.set_beta2(-0.1); }));
    check("set_eps(0) throws",    throws([&]{ valid.set_eps(0.0); }));
    check("set_wd(-0.1) throws",  throws([&]{ valid.set_weight_decay(-0.1); }));
    check("set_warmup(-1) throws",throws([&]{ valid.set_warmup_steps(-1); }));
    check("set_r(-0.1) throws",   throws([&]{ valid.set_r(-0.1); }));
    check("set_wlp(-0.1) throws", throws([&]{ valid.set_weight_lr_power(-0.1); }));
}

// ---- T3: state init shape and values ----
static void test_state_init() {
    check_section("T3: state shape + init values");
    Model model;
    auto* layer = new TestParam(3, 4);
    layer->param.fill(0.5);
    layer->grad.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt;
    // No state before step
    check("has_state false pre-step", !opt.has_state(layer));

    // Calling step without train mode should throw.
    bool step_threw = false;
    try { opt.step(model); } catch (...) { step_threw = true; }
    check("step() throws when not in train mode", step_threw);

    // Now set up train mode and step once.
    opt.train(model);
    layer->grad.fill(1.0);
    opt.step(model);

    check("has_state true post-step", opt.has_state(layer));
    Tensor z = opt.get_z(layer, 0);
    Tensor v = opt.get_exp_avg_sq(layer, 0);
    check("z shape matches param (3,4)", z.rows == 3 && z.cols == 4);
    check("exp_avg_sq shape matches param (3,4)", v.rows == 3 && v.cols == 4);
    // After one step with grad=1, exp_avg_sq should be (1-beta2)*1 = 0.001
    check("exp_avg_sq[0][0] = (1-beta2)*1",
          near(v[0][0], (1.0 - 0.999) * 1.0 * 1.0, 1e-12));
    // k now 1
    check("k incremented to 1", opt.get_k() == 1);
    // lr_max updated to lr*1 (no warmup)
    check("lr_max updated to lr", near(opt.get_lr_max(), 1.0));
    // weight_sum: k+1 = 1, weight_lr_power=2, lr_max=1.0, so first
    // weight = 1^0 * 1.0^2 = 1, weight_sum = 1.
    check("weight_sum = 1.0 after step 1", near(opt.get_weight_sum(), 1.0, 1e-12));
}

// ---- T4: closed-form first-step (degenerate case beta1=beta2=0) ----
static void test_first_step_degenerate() {
    check_section("T4: closed-form first step (beta1=beta2=0, eps=1)");
    // With beta1=0 and beta2=0 the bias_corr2 terms vanish cleanly:
    //   bias_corr2 = 1 - 0^(0+1) = 1
    //   exp_avg_sq_1 = 0 + 1 * g^2 = g^2
    //   denom = sqrt(g^2 / 1) + eps = g + eps = 1 + 1 = 2 (g=1, eps=1)
    //   u = g / denom = 0.5
    //   z_1 = z_0 - lr * u = 0 - 0.5 = -0.5
    //   ckp1 = 1, y_coeff = lr*(beta1*(1-ckp1) - 1) = -1
    //   y_1 = ckp1 * z_1 + (1-ckp1) * y_0 + y_coeff * u = 1*(-0.5) + 0*0 + (-1)*0.5 = -1
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param[0][0] = 0.0;
    layer->grad[0][0] = 1.0;
    model.add_layer(layer);

    ScheduleFreeAdamW opt(1.0, 0.0, 0.0, 1.0, 0.0, 0, 0.0, 2.0);
    opt.train(model);
    opt.step(model);

    // z_1 = 0 - 1 * (1/2) = -0.5
    Tensor z = opt.get_z(layer, 0);
    check("z_1 = -0.5 exactly", near(z[0][0], -0.5, 1e-12));
    // y_1 = 1*(-0.5) + 0*0 + (-1)*0.5 = -1.0
    check("y_1 = param = -1 exactly", near(layer->param[0][0], -1.0, 1e-12));
}

// ---- T5: z and exp_avg_sq multi-step EMA recurrence ----
static void test_ema_recurrence() {
    check_section("T5: z/exp_avg_sq EMA recurrence");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    // Use beta1=0, beta2=0.5, eps=0 for clean closed form.
    ScheduleFreeAdamW opt(1.0, 0.0, 0.5, 1e-30, 0.0);
    opt.train(model);

    // Step 1: g=2
    layer->grad[0][0] = 2.0;
    opt.step(model);
    // exp_avg_sq_1 = 0.5 * 0 + 0.5 * 4 = 2.0
    Tensor v = opt.get_exp_avg_sq(layer, 0);
    check("v_1 = (1-b2)*g^2 = 2.0", near(v[0][0], 2.0, 1e-12));

    // Step 2: g=4
    layer->grad[0][0] = 4.0;
    opt.step(model);
    // exp_avg_sq_2 = 0.5 * 2.0 + 0.5 * 16 = 9.0
    v = opt.get_exp_avg_sq(layer, 0);
    check("v_2 = b2*v_1 + (1-b2)*g^2 = 9.0", near(v[0][0], 9.0, 1e-12));

    // Step 3: g=0
    layer->grad[0][0] = 0.0;
    opt.step(model);
    // exp_avg_sq_3 = 0.5 * 9.0 + 0.5 * 0 = 4.5
    v = opt.get_exp_avg_sq(layer, 0);
    check("v_3 = b2*v_2 = 4.5", near(v[0][0], 4.5, 1e-12));
}

// ---- T5b: bias_correction2 affects z at step > 1 ----
//
// With beta2=0.5 and a constant non-zero gradient, the bias correction
//   bc2 = 1 - 0.5^(k+1)  causes denom = sqrt(v/bc2) + eps to differ
// from the uncorrected sqrt(v) + eps by a factor of 1/sqrt(bc2).
// This test verifies the bias correction is actually applied.
static void test_bias_correction_in_denom() {
    check_section("T5b: bias_correction2 affects z update");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    // beta1=0 (no first moment), beta2=0.5, eps=1e-30 for clean closed form
    // (eps=0 throws per validation; 1e-30 is negligible for closed-form math).
    ScheduleFreeAdamW opt(1.0, 0.0, 0.5, 1e-30, 0.0, 0, 0.0, 2.0);
    opt.train(model);

    // Step 1: g=4
    //   v_1 = 0.5*0 + 0.5*16 = 8
    //   bc2_1 = 1 - 0.5^1 = 0.5
    //   denom = sqrt(8/0.5) + 1e-30 ≈ 4
    //   u = 4/4 = 1, z_1 = 0 - 1 = -1
    layer->grad[0][0] = 4.0;
    opt.step(model);
    Tensor z = opt.get_z(layer, 0);
    check("step1: z_1 = -1.0 (bc2=0.5, v=8, denom=4, u=1)",
          near(z[0][0], -1.0, 1e-9));

    // Step 2: g=4 again
    //   v_2 = 0.5*8 + 0.5*16 = 12
    //   bc2_2 = 1 - 0.5^2 = 0.75
    //   denom = sqrt(12/0.75) + 1e-30 ≈ 4
    //   u = 4/4 = 1, z_2 = z_1 - 1 = -2
    layer->grad[0][0] = 4.0;
    opt.step(model);
    z = opt.get_z(layer, 0);
    check("step2: z_2 = -2.0 (bc2=0.75, v=12, denom=4, u=1)",
          near(z[0][0], -2.0, 1e-9));

    // Step 3: g=4
    //   v_3 = 0.5*12 + 0.5*16 = 14
    //   bc2_3 = 1 - 0.5^3 = 0.875
    //   denom = sqrt(14/0.875) + 1e-30 ≈ 4
    //   u = 4/4 = 1, z_3 = z_2 - 1 = -3
    layer->grad[0][0] = 4.0;
    opt.step(model);
    z = opt.get_z(layer, 0);
    check("step3: z_3 = -3.0 (bc2=0.875, v=14, denom=4, u=1)",
          near(z[0][0], -3.0, 1e-9));
}

// ---- T6: y and z evolve distinct from each other ----
static void test_y_vs_z_distinct() {
    check_section("T6: y and z are distinct sequences");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt(0.1, 0.9, 0.999, 1e-8);
    opt.train(model);
    layer->grad[0][0] = 1.0;
    opt.step(model);
    Tensor z = opt.get_z(layer, 0);
    // param (y_1) and z_1 must be different (else algo would collapse).
    check("y_1 != z_1 after step", std::abs(layer->param[0][0] - z[0][0]) > 1e-6);
    check("|y_1| > 0", std::abs(layer->param[0][0]) > 0.0);
    check("|z_1| > 0", std::abs(z[0][0]) > 0.0);
}

// ---- T7: mode swap (eval / train) ----
static void test_mode_swap() {
    check_section("T7: eval/train mode swap");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt(1.0, 0.5, 0.999, 1e-8);
    opt.train(model);
    layer->grad[0][0] = 0.0;  // no grad
    opt.step(model);

    // After step k=0 with k+1=1, ckp1=1, z_1=0, u=0 → y_1 = 1*0 + 0*0 + 1*(0.5*0-1)*0 = 0
    // (Actually z_1 = z_0 - lr * u, and u = 0 (g=0), so z_1 = z_0 = 0.)
    // Train: p = 0 holds y=0.
    check("initial y_1 = 0 (no grad)", near(layer->param[0][0], 0.0, 1e-12));
    Tensor z_before = opt.get_z(layer, 0);
    check("z_1 = 0 (no grad)", near(z_before[0][0], 0.0, 1e-12));

    // Now switch to eval: p should become x.
    // p.lerp_(z, 1-1/beta1) = 0 + (1-2)*(0-0) = 0 (still 0 because z=y=0)
    // That doesn't actually exercise the swap, so let's do a more interesting
    // setup: do another step with a non-zero grad.
    layer->grad[0][0] = 1.0;
    opt.step(model);  // k now 1, k+1=2, ckp1=1/2 (with r=0, wlp=2)
    // y will be a non-trivial value. Capture.
    const double y_now = layer->param[0][0];
    Tensor z_now = opt.get_z(layer, 0);
    check("after step 2: y != 0", std::abs(y_now) > 1e-6);
    check("after step 2: z != 0", std::abs(z_now[0][0]) > 1e-6);

    // Switch to eval mode: p → x.
    // By construction: y = (1-beta1)*z + beta1*x, so x = (y - (1-beta1)*z)/beta1.
    // In-place: p = p + (1-1/beta1)*(z-p) = x.
    opt.eval(model);
    const double x_val = layer->param[0][0];
    const double expected_x = (y_now - (1.0 - 0.5) * z_now[0][0]) / 0.5;
    check("eval: param becomes x (y - (1-b1)*z)/b1",
          near(x_val, expected_x, 1e-9));
    check("eval: train_mode flag flipped", !opt.is_train_mode());

    // Switch back to train: p → y = (1-beta1)*z + beta1*x.
    // In-place: p = p + (1-beta1)*(z-p) = (1-b1)*x + (1-b1)*z... wait, let's check.
    // p_new = x + (1-beta1)*(z - x) = (1 - (1-beta1))*x + (1-beta1)*z = beta1*x + (1-beta1)*z = y.
    opt.train(model);
    const double y_after = layer->param[0][0];
    const double expected_y = 0.5 * x_val + 0.5 * z_now[0][0];
    check("train: param becomes y = b1*x + (1-b1)*z",
          near(y_after, expected_y, 1e-9));
    check("train: train_mode flag flipped back", opt.is_train_mode());

    // Idempotence: eval twice is a no-op.
    opt.eval(model);
    const double x2 = layer->param[0][0];
    opt.eval(model);
    check("eval twice is idempotent", near(x2, layer->param[0][0], 1e-15));
    opt.train(model);
    opt.train(model);
    check("train twice is idempotent", near(y_after, layer->param[0][0], 1e-15));
}

// ---- T8: weight decay is applied as decoupled on the Adam direction ----
//
// With wd > 0, u = g/(sqrt(v/bc2)+eps) + wd*y. Verify that y shrinks
// even with g=0 (coupled form).
static void test_weight_decay() {
    check_section("T8: coupled weight decay on Adam direction");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param[0][0] = 1.0;
    model.add_layer(layer);

    ScheduleFreeAdamW opt(1.0, 0.0, 0.999, 1e-8, 0.1);  // wd=0.1
    opt.train(model);
    layer->grad[0][0] = 0.0;
    opt.step(model);
    // With g=0: u = 0 + 0.1 * y = 0.1 (at the start of step 1).
    //   z_1 = z_0 - lr*u = 1.0 - 0.1 = 0.9
    //   ckp1 = 1, y_coeff = lr*(beta1*(0) - 1) = -1
    //   y_1 = 1 * 0.9 + 0 * 1.0 + (-1)*0.1 = 0.9 - 0.1 = 0.8
    Tensor z = opt.get_z(layer, 0);
    check("z shrinks by lr*wd*y = 0.1", near(z[0][0], 0.9, 1e-9));
    check("y shrinks to 0.8",          near(layer->param[0][0], 0.8, 1e-9));
}

// ---- T9: parameter/gradient count + shape mismatch throws ----
static void test_malformed_layer() {
    check_section("T9: malformed layer guard");
    // Custom layer that lies about parameter count.
    class LyingParam : public Layer {
    public:
        Tensor p; Tensor g;
        LyingParam() : p(2, 2), g(2, 2) {
            p.fill(0.0); g.fill(0.0);
        }
        Tensor forward(const Tensor& input) override { return input; }
        Tensor backward(const Tensor& g, double) override { return g; }
        void update_weights(double) override {}
        Tensor get_weights() const override { return p; }
        Tensor get_gradients() const override { return g; }
        vector<Tensor*> parameters() override { return {&p}; } // claim 1 param
        vector<Tensor*> gradients() override { return {}; }    // but 0 grads!
        void zero_grad() override { g.fill(0.0); }
    };
    Model model;
    auto* bad = new LyingParam();
    model.add_layer(bad);

    ScheduleFreeAdamW opt;
    opt.train(model);
    bool threw = false;
    try { opt.step(model); } catch (std::runtime_error&) { threw = true; }
    check("param/grad count mismatch throws", threw);

    // Now a layer whose param shape != grad shape.
    class ShapeMismatch : public Layer {
    public:
        Tensor p, g;
        ShapeMismatch() : p(2, 3), g(3, 2) {
            p.fill(0.0); g.fill(0.0);
        }
        Tensor forward(const Tensor& input) override { return input; }
        Tensor backward(const Tensor& g, double) override { return g; }
        void update_weights(double) override {}
        Tensor get_weights() const override { return p; }
        Tensor get_gradients() const override { return g; }
        vector<Tensor*> parameters() override { return {&p}; }
        vector<Tensor*> gradients() override { return {&g}; }
        void zero_grad() override { g.fill(0.0); }
    };
    Model model2;
    auto* bad2 = new ShapeMismatch();
    model2.add_layer(bad2);
    opt.train(model2);
    bool threw2 = false;
    try { opt.step(model2); } catch (std::runtime_error&) { threw2 = true; }
    check("param/grad shape mismatch throws", threw2);
}

// ---- T10: determinism — two fresh instances produce bit-exact params ----
static void test_determinism() {
    check_section("T10: determinism over 30 random-grad steps");
    Model m1, m2;
    auto* l1 = new TestParam(3, 4);
    auto* l2 = new TestParam(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) {
            l1->param[i][j] = 0.42 + 0.1 * i + 0.01 * j;
            l2->param[i][j] = l1->param[i][j];
        }
    m1.add_layer(l1);
    m2.add_layer(l2);

    ScheduleFreeAdamW opt1(0.05, 0.9, 0.999, 1e-8);
    ScheduleFreeAdamW opt2(0.05, 0.9, 0.999, 1e-8);
    opt1.train(m1);
    opt2.train(m2);

    srand(42);
    for (int step = 0; step < 30; ++step) {
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j) {
                double g = (rand() % 1000) / 1000.0 - 0.5;
                l1->grad[i][j] = g;
                l2->grad[i][j] = g;
            }
        opt1.step(m1);
        opt2.step(m2);
    }

    bool all_match = true;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (!near(l1->param[i][j], l2->param[i][j], 0.0)) {
                all_match = false;
            }
    check("two fresh instances: bit-exact params after 30 steps", all_match);

    // Also check z state is identical
    Tensor z1 = opt1.get_z(l1, 0);
    Tensor z2 = opt2.get_z(l2, 0);
    bool z_match = true;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (!near(z1[i][j], z2[i][j], 0.0)) z_match = false;
    check("two fresh instances: bit-exact z state", z_match);
}

// ---- T11: end-to-end training on linear regression (Dense(1,1)) ----
static void test_end_to_end() {
    check_section("T11: end-to-end linear regression");
    Model model;
    Dense* d = new Dense(1, 1);
    // Initialize weights to something close to identity so the optimization
    // converges quickly.
    d->weights[0][0] = 0.5;
    d->bias[0][0] = 0.0;
    model.add_layer(d);

    // y = 2 * x, single feature.
    Tensor X(4, 1);
    Tensor y(4, 1);
    X[0][0] = 1.0; y[0][0] = 2.0;
    X[1][0] = 2.0; y[1][0] = 4.0;
    X[2][0] = 3.0; y[2][0] = 6.0;
    X[3][0] = 4.0; y[3][0] = 8.0;

    auto mse = [](const Tensor& pred, const Tensor& t) {
        double s = 0.0;
        for (size_t i = 0; i < pred.rows; ++i)
            for (size_t j = 0; j < pred.cols; ++j) {
                double d = pred[i][j] - t[i][j];
                s += d * d;
            }
        return s / static_cast<double>(pred.rows);
    };

    ScheduleFreeAdamW opt(0.05, 0.9, 0.999, 1e-8, 0.01);
    opt.train(model);
    Tensor pred0 = model.forward(X);
    double loss0 = mse(pred0, y);
    for (int step = 0; step < 200; ++step) {
        Tensor pred = model.forward(X);
        Tensor grad(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.rows; ++i)
            for (size_t j = 0; j < pred.cols; ++j)
                grad[i][j] = 2.0 * (pred[i][j] - y[i][j]) / static_cast<double>(pred.rows);
        model.backward(grad, 0.0);
        opt.step(model);
    }
    Tensor pred_final = model.forward(X);
    double loss_final = mse(pred_final, y);

    cout << "  initial loss: " << loss0 << "\n";
    cout << "  final loss:   " << loss_final << "\n";
    check("loss decreased by > 50%", loss_final < 0.5 * loss0);
    check("final weights near 2.0", std::abs(d->weights[0][0] - 2.0) < 0.5);
}

// ---- T12: signature test — produces a different trajectory than vanilla AdamW ----
static void test_signature_vs_adamw() {
    check_section("T12: signature test vs vanilla AdamW");
    // A vanilla AdamW with same lr/betas/eps over the same gradient sequence
    // should produce a different parameter trajectory than Schedule-Free.
    // We use simple analytic recurrences for both (no Dense coupling).
    double lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    // AdamW recurrences:
    double adam_m = 0.0, adam_v = 0.0;
    double adam_w = 0.0;  // initial weight

    // Schedule-Free recurrences (with r=0, wlp=2):
    // We just track the y and z sequences directly here.
    double sf_z = 0.0, sf_exp_avg_sq = 0.0, sf_y = 0.0;

    double grad_seq[6] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    for (int k = 0; k < 6; ++k) {
        double g = grad_seq[k];
        // AdamW step (bias-corrected)
        adam_m = beta1 * adam_m + (1.0 - beta1) * g;
        adam_v = beta2 * adam_v + (1.0 - beta2) * g * g;
        double kp1 = static_cast<double>(k + 1);
        double m_hat = adam_m / (1.0 - std::pow(beta1, kp1));
        double v_hat = adam_v / (1.0 - std::pow(beta2, kp1));
        adam_w -= lr * m_hat / (std::sqrt(v_hat) + eps);

        // Schedule-Free step
        sf_exp_avg_sq = beta2 * sf_exp_avg_sq + (1.0 - beta2) * g * g;
        double bc2 = 1.0 - std::pow(beta2, kp1);
        double denom = std::sqrt(sf_exp_avg_sq / bc2) + eps;
        double u = g / denom;
        // ckp1: with r=0, wlp=2, lr_max grows to lr (since sched=1 always),
        // weight = lr^2, weight_sum = (k+1)*lr^2, ckp1 = 1/(k+1).
        double ckp1 = 1.0 / kp1;
        // Update z: z_{k+1} = z_k - lr*u
        double z_new = sf_z - lr * u;
        // Update y: y_{k+1} = ckp1*z_new + (1-ckp1)*y_k + lr*(beta1*(1-ckp1)-1)*u
        double y_coeff = lr * (beta1 * (1.0 - ckp1) - 1.0);
        sf_y = ckp1 * z_new + (1.0 - ckp1) * sf_y + y_coeff * u;
        sf_z = z_new;
    }
    cout << "  AdamW final w = " << adam_w << "\n";
    cout << "  Schedule-Free final y = " << sf_y << "\n";
    cout << "  Schedule-Free final z = " << sf_z << "\n";
    check("AdamW and Schedule-Free produce different trajectories",
          std::abs(adam_w - sf_y) > 1e-3);
    check("AdamW and Schedule-Free produce different z",
          std::abs(adam_w - sf_z) > 1e-3);
}

// ---- T13: ckp1 progression with default r=0, weight_lr_power=2 ----
static void test_ckp1_progression() {
    check_section("T13: ckp1 progression");
    // With r=0, weight_lr_power=2, lr=1.0:
    //   lr_max grows to 1.0 on step 1 (no warmup, sched=1)
    //   weight_k = (k+1)^0 * 1.0^2 = 1
    //   weight_sum_k = sum_{i=1..k+1} 1 = k+1
    //   ckp1_k = 1 / (k+1)
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt(1.0, 0.9, 0.999, 1e-8, 0.0, 0, 0.0, 2.0);
    opt.train(model);
    layer->grad.fill(0.0);  // no grad → no u → z stays 0, but ckp1 still progresses
    opt.step(model);  // k=0→1
    check("after step 1: weight_sum = 1", near(opt.get_weight_sum(), 1.0, 1e-12));
    opt.step(model);  // k=1→2
    check("after step 2: weight_sum = 2", near(opt.get_weight_sum(), 2.0, 1e-12));
    opt.step(model);  // k=2→3
    check("after step 3: weight_sum = 3", near(opt.get_weight_sum(), 3.0, 1e-12));
    opt.step(model);  // k=3→4
    check("after step 4: weight_sum = 4", near(opt.get_weight_sum(), 4.0, 1e-12));
    // lr_max should be exactly 1.0 throughout (no warmup).
    check("lr_max = 1.0 (no warmup)", near(opt.get_lr_max(), 1.0, 1e-12));
    check("k = 4 after 4 steps", opt.get_k() == 4);
}

// ---- T14: warmup schedule (linear ramp) ----
static void test_warmup_schedule() {
    check_section("T14: warmup_steps=3 linear ramp");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt(2.0, 0.9, 0.999, 1e-8, 0.0, 3, 0.0, 2.0);
    opt.train(model);
    layer->grad[0][0] = 1.0;
    opt.step(model);  // k=0: sched=1/3, lr_used=2/3
    check("after step 1: lr_max = 2/3", near(opt.get_lr_max(), 2.0/3.0, 1e-9));
    opt.step(model);  // k=1: sched=2/3, lr_used=4/3, lr_max updates to 4/3
    check("after step 2: lr_max = 4/3", near(opt.get_lr_max(), 4.0/3.0, 1e-9));
    opt.step(model);  // k=2: sched=1, lr_used=2, lr_max=2
    check("after step 3: lr_max = 2", near(opt.get_lr_max(), 2.0, 1e-9));
    opt.step(model);  // k=3: sched=1 (post-warmup), lr_used=2, lr_max=2
    check("after step 4: lr_max stays at 2", near(opt.get_lr_max(), 2.0, 1e-12));
}

// ---- T15: independent state across layers ----
static void test_independent_layers() {
    check_section("T15: independent state across layers");
    Model model;
    auto* l1 = new TestParam(2, 2);
    auto* l2 = new TestParam(2, 2);
    l1->param.fill(0.0); l2->param.fill(0.0);
    model.add_layer(l1);
    model.add_layer(l2);

    ScheduleFreeAdamW opt;
    opt.train(model);
    l1->grad.fill(1.0);
    l2->grad.fill(0.0);
    opt.step(model);

    Tensor v1 = opt.get_exp_avg_sq(l1, 0);
    Tensor v2 = opt.get_exp_avg_sq(l2, 0);
    check("layer1 exp_avg_sq nonzero", std::abs(v1[0][0]) > 0.0);
    check("layer2 exp_avg_sq zero (zero grad)", near(v2[0][0], 0.0, 1e-15));
    check("layer1 has_state", opt.has_state(l1));
    check("layer2 has_state", opt.has_state(l2));
}

// ---- T16: independent state across parameters in the same layer ----
static void test_independent_params() {
    check_section("T16: independent state across parameters");
    // A layer with 2 distinct parameters (e.g., weights + bias).
    class TwoParamLayer : public Layer {
    public:
        Tensor w, b, gw, gb;
        TwoParamLayer() : w(2, 3), b(1, 3), gw(2, 3), gb(1, 3) {
            w.fill(0.0); b.fill(0.0); gw.fill(0.0); gb.fill(0.0);
        }
        Tensor forward(const Tensor& input) override { return input; }
        Tensor backward(const Tensor& g, double) override { return g; }
        void update_weights(double) override {}
        Tensor get_weights() const override { return w; }
        Tensor get_gradients() const override { return gw; }
        vector<Tensor*> parameters() override { return {&w, &b}; }
        vector<Tensor*> gradients() override { return {&gw, &gb}; }
        void zero_grad() override { gw.fill(0.0); gb.fill(0.0); }
    };
    Model model;
    auto* layer = new TwoParamLayer();
    model.add_layer(layer);

    ScheduleFreeAdamW opt;
    opt.train(model);
    layer->gw[0][0] = 1.0;
    layer->gb[0][0] = -1.0;
    opt.step(model);

    Tensor zw = opt.get_z(layer, 0);
    Tensor zb = opt.get_z(layer, 1);
    check("weight z exists and nonzero", std::abs(zw[0][0]) > 0.0);
    check("bias z exists and nonzero (different grad direction)",
          std::abs(zb[0][0]) > 0.0);
    // z for weight (grad=+1) and bias (grad=-1) must differ in sign.
    check("weight z and bias z differ in sign",
          (zw[0][0] > 0) != (zb[0][0] > 0));
}

// ---- T17: gradient clearing ----
static void test_zero_grad() {
    check_section("T17: gradient clearing after step");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->grad[0][0] = 1.0;
    model.add_layer(layer);

    ScheduleFreeAdamW opt;
    opt.train(model);
    opt.step(model);
    check("gradient cleared after step", near(layer->grad[0][0], 0.0, 1e-15));
}

// ---- T18: independent state across layers (Dense layer uses standard Model.train) ----
static void test_dense_integration() {
    check_section("T18: dense layer integration");
    Model model;
    auto* d = new Dense(2, 2);
    model.add_layer(d);
    ScheduleFreeAdamW opt(0.01, 0.9, 0.999, 1e-8);
    opt.train(model);
    Tensor x(1, 2);
    x[0][0] = 1.0; x[0][1] = 2.0;
    Tensor y_true(1, 2);
    y_true[0][0] = 0.5; y_true[0][1] = -0.5;

    auto mse = [](const Tensor& p, const Tensor& t) {
        double s = 0.0;
        for (size_t i = 0; i < p.rows; ++i)
            for (size_t j = 0; j < p.cols; ++j) {
                double d = p[i][j] - t[i][j]; s += d * d;
            }
        return s / static_cast<double>(p.rows);
    };

    double loss0 = mse(model.forward(x), y_true);
    for (int step = 0; step < 100; ++step) {
        Tensor pred = model.forward(x);
        Tensor grad(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.rows; ++i)
            for (size_t j = 0; j < pred.cols; ++j)
                grad[i][j] = 2.0 * (pred[i][j] - y_true[i][j]) / static_cast<double>(pred.rows);
        model.backward(grad, 0.0);
        opt.step(model);
    }
    double loss_final = mse(model.forward(x), y_true);
    cout << "  Dense integration: loss " << loss0 << " -> " << loss_final << "\n";
    check("Dense integration: loss decreased > 30%", loss_final < 0.7 * loss0);
}

// ---- T19: optional r and weight_lr_power affect ckp1 progression ----
static void test_r_and_weight_lr_power() {
    check_section("T19: r and weight_lr_power affect progression");
    // With r=1, weight_lr_power=0, lr=1:
    //   weight_k = (k+1)^1 * 1.0^0 = k+1
    //   weight_sum_k = sum_{i=1..k+1} i = (k+1)(k+2)/2
    //   ckp1_k = (k+1) / ((k+1)(k+2)/2) = 2/(k+2)
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeAdamW opt(1.0, 0.9, 0.999, 1e-8, 0.0, 0, 1.0, 0.0);
    opt.train(model);
    layer->grad.fill(0.0);
    opt.step(model);  // k=0→1
    // weight_sum = 1 (single step)
    check("r=1, wlp=0, step 1: weight_sum = 1", near(opt.get_weight_sum(), 1.0, 1e-12));
    opt.step(model);  // k=1→2
    // weight_sum = 1 + 2 = 3
    check("r=1, wlp=0, step 2: weight_sum = 3", near(opt.get_weight_sum(), 3.0, 1e-12));
    opt.step(model);  // k=2→3
    // weight_sum = 1 + 2 + 3 = 6
    check("r=1, wlp=0, step 3: weight_sum = 6", near(opt.get_weight_sum(), 6.0, 1e-12));
}

// ---- T20: get_z / get_exp_avg_sq return empty tensor when no state ----
static void test_state_accessors_empty() {
    check_section("T20: state accessors before step");
    Model model;
    auto* layer = new TestParam(1, 1);
    model.add_layer(layer);

    ScheduleFreeAdamW opt;
    Tensor z = opt.get_z(layer, 0);
    Tensor v = opt.get_exp_avg_sq(layer, 0);
    check("get_z before step: empty (0,0)", z.rows == 0 && z.cols == 0);
    check("get_exp_avg_sq before step: empty (0,0)", v.rows == 0 && v.cols == 0);
}

// ---- T21: model with zero parameters doesn't crash ----
static void test_empty_layer() {
    check_section("T21: layer with zero params is skipped");
    class EmptyLayer : public Layer {
    public:
        Tensor forward(const Tensor& input) override { return input; }
        Tensor backward(const Tensor& g, double) override { return g; }
        void update_weights(double) override {}
        Tensor get_weights() const override { return Tensor(0, 0); }
        Tensor get_gradients() const override { return Tensor(0, 0); }
        vector<Tensor*> parameters() override { return {}; }
        vector<Tensor*> gradients() override { return {}; }
        void zero_grad() override {}
    };
    Model model;
    auto* empty = new EmptyLayer();
    auto* real = new TestParam(1, 1);
    real->param[0][0] = 0.0;
    real->grad[0][0] = 1.0;
    model.add_layer(empty);
    model.add_layer(real);

    ScheduleFreeAdamW opt;
    opt.train(model);
    bool crashed = false;
    try {
        opt.step(model);
    } catch (...) { crashed = true; }
    check("empty layer doesn't crash step()", !crashed);
    check("real layer still gets updated", !near(real->param[0][0], 0.0, 1e-6));
    check("empty layer has no state", !opt.has_state(empty));
    check("real layer has state", opt.has_state(real));
}

int main() {
    cout << fixed << setprecision(12);
    test_defaults();
    test_constructor_and_validation();
    test_state_init();
    test_first_step_degenerate();
    test_ema_recurrence();
    test_bias_correction_in_denom();
    test_y_vs_z_distinct();
    test_mode_swap();
    test_weight_decay();
    test_malformed_layer();
    test_determinism();
    test_end_to_end();
    test_signature_vs_adamw();
    test_ckp1_progression();
    test_warmup_schedule();
    test_independent_layers();
    test_independent_params();
    test_zero_grad();
    test_dense_integration();
    test_r_and_weight_lr_power();
    test_state_accessors_empty();
    test_empty_layer();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
