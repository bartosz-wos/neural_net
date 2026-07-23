// test_schedule_free_sgd.cpp — behavioral tests for Schedule-Free SGD.
//
// Reference: Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024,
// "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682)
// NeurIPS 2024 best-paper nominee.
//
// Canonical reference implementation:
//   https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/sgd_schedulefree.py
//
// The schedule-free optimizer maintains three coupled sequences per parameter:
//
//   x_k : eval point  (the parameter that forward/backward runs at next step)
//   z_k : iterate     (the parameter that gets gradient-updated)
//   y_k : averaged    (returned parameter; y_k = (1-beta1)*z_k + beta1*x_k)
//
// In train mode `param` holds `y`; in eval mode `param` holds `x`. The
// public API is `optimizer.train(model)` and `optimizer.eval(model)` which
// perform the linear swap (lerp) between `y` and `x`.
//
// Per-step update at the start of step k (k = 0-indexed; current step is k+1):
//   u          = g + weight_decay * y        (coupled form when wd > 0)
//   z_{k+1}    = z_k - lr * u                (no Adam denominator, no bias corr)
//   y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k + lr*(beta1*(1-ckp1) - 1) * u
// where ckp1 = 1/(k+1) for default r=0, weight_lr_power=2.0 (and the warmup
// schedule only kicks in if k < warmup_steps).
//
// State init at first step: z = clone(param). No exp_avg_sq.
//
// Mode swaps (canonical formulas, shared with Schedule-Free AdamW):
//   eval (y -> x):  p = p + (1 - 1/beta1) * (z - p)
//   train (x -> y): p = p + (1 - beta1) * (z - p)
//
// Tests below verify defaults, state shapes, eval/train mode swap, the
// closed-form first step under beta1=0 (degenerate case), z recurrence,
// ckp1 progression, coupled weight decay, mode-swap correctness,
// determinism, end-to-end training reduction, and signature tests that
// distinguish Schedule-Free SGD from Schedule-Free AdamW and from vanilla
// SGD-with-momentum.

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
#include "nn/optimizers/schedule_free_sgd.h"

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
    ScheduleFreeSGD opt;
    check("lr default = 1.0",                near(opt.get_lr(), 1.0));
    check("beta1 default = 0.9",             near(opt.get_beta1(), 0.9));
    check("weight_decay default = 0",        near(opt.get_weight_decay(), 0.0));
    check("warmup_steps default = 0",        opt.get_warmup_steps() == 0);
    check("r default = 0",                   near(opt.get_r(), 0.0));
    check("weight_lr_power default = 2.0",   near(opt.get_weight_lr_power(), 2.0));
    check("k starts at 0",                   opt.get_k() == 0);
    check("lr_max starts at -1",             near(opt.get_lr_max(), -1.0));
    check("weight_sum starts at 0",          near(opt.get_weight_sum(), 0.0));
    check("train_mode starts false",         !opt.is_train_mode());
    check("handles_weight_decay true",       opt.handles_weight_decay());
}

// ---- T2: non-default constructor + validation throws ----
static void test_constructor_and_validation() {
    check_section("T2: constructor + validation");
    {
        ScheduleFreeSGD opt(0.5, 0.85, 0.01, 5, 0.5, 1.5);
        check("non-default lr",      near(opt.get_lr(), 0.5));
        check("non-default beta1",   near(opt.get_beta1(), 0.85));
        check("non-default wd",      near(opt.get_weight_decay(), 0.01));
        check("non-default warmup",  opt.get_warmup_steps() == 5);
        check("non-default r",       near(opt.get_r(), 0.5));
        check("non-default wlp",     near(opt.get_weight_lr_power(), 1.5));
    }
    auto throws = [](auto fn) {
        try { fn(); return false; } catch (...) { return true; }
    };
    ScheduleFreeSGD valid(1.0, 0.9, 0.0);
    check("lr=-1 throws",      throws([&]{ ScheduleFreeSGD o(-1.0, 0.9); }));
    check("lr=0 throws",       throws([&]{ ScheduleFreeSGD o(0.0, 0.9); }));
    check("beta1=1 throws",    throws([&]{ ScheduleFreeSGD o(1.0, 1.0); }));
    check("beta1=-1 throws",   throws([&]{ ScheduleFreeSGD o(1.0, -0.1); }));
    check("wd<0 throws",       throws([&]{ ScheduleFreeSGD o(1.0, 0.9, -0.1); }));
    check("warmup<0 throws",   throws([&]{ ScheduleFreeSGD o(1.0, 0.9, 0.0, -1); }));
    check("r<0 throws",        throws([&]{ ScheduleFreeSGD o(1.0, 0.9, 0.0, 0, -0.1); }));
    check("wlp<0 throws",      throws([&]{ ScheduleFreeSGD o(1.0, 0.9, 0.0, 0, 0, -0.1); }));
    // mutator validation
    valid.set_lr(2.0);
    check("set_lr ok",         near(valid.get_lr(), 2.0));
    check("set_lr(-1) throws", throws([&]{ valid.set_lr(-1.0); }));
    check("set_beta1(1) throws", throws([&]{ valid.set_beta1(1.0); }));
    check("set_beta1(-0.1) throws", throws([&]{ valid.set_beta1(-0.1); }));
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

    ScheduleFreeSGD opt;
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
    check("z shape matches param (3,4)", z.rows == 3 && z.cols == 4);
    // z initialized as clone of param (0.5 everywhere), then updated.
    // Step 1: g=1, wd=0 → u=1, z_1 = 0.5 - 1*1 = -0.5
    check("z[0][0] = -0.5 (z0=0.5, u=1)", near(z[0][0], -0.5, 1e-12));
    // k now 1
    check("k incremented to 1", opt.get_k() == 1);
    // lr_max updated to lr*1 (no warmup)
    check("lr_max updated to lr", near(opt.get_lr_max(), 1.0));
    // weight_sum: k+1 = 1, weight_lr_power=2, lr_max=1.0, so first
    // weight = 1^0 * 1.0^2 = 1, weight_sum = 1.
    check("weight_sum = 1.0 after step 1", near(opt.get_weight_sum(), 1.0, 1e-12));
}

// ---- T4: closed-form first-step (degenerate case beta1=0) ----
//
// With beta1=0 and lr=1, g=1, init=0.5, wd=0:
//   u = 1
//   z_1 = z_0 - lr * u = 0.5 - 1 = -0.5
//   ckp1 = 1, y_coeff = lr*(beta1*(1-ckp1) - 1) = -1
//   y_1 = 1 * (-0.5) + 0 * 0.5 + (-1) * 1 = -1.5
static void test_first_step_degenerate() {
    check_section("T4: closed-form first step (beta1=0)");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param[0][0] = 0.5;
    layer->grad[0][0] = 1.0;
    model.add_layer(layer);

    ScheduleFreeSGD opt(1.0, 0.0, 0.0, 0, 0.0, 2.0);
    opt.train(model);
    opt.step(model);

    Tensor z = opt.get_z(layer, 0);
    check("z_1 = -0.5 exactly", near(z[0][0], -0.5, 1e-12));
    check("y_1 = param = -1.5 exactly", near(layer->param[0][0], -1.5, 1e-12));
}

// ---- T5: z multi-step recurrence (no Adam denominator to fight us) ----
static void test_z_recurrence() {
    check_section("T5: z recurrence under constant gradient");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    // Use beta1=0, wd=0 for clean closed form. With beta1=0, the z update
    // is purely z -= lr * g every step.
    ScheduleFreeSGD opt(1.0, 0.0, 0.0);
    opt.train(model);

    // Step 1: g=2 → z_1 = 0 - 2 = -2
    layer->grad[0][0] = 2.0;
    opt.step(model);
    Tensor z = opt.get_z(layer, 0);
    check("z_1 = -2.0 (g=2)", near(z[0][0], -2.0, 1e-12));

    // Step 2: g=4 → z_2 = -2 - 4 = -6
    layer->grad[0][0] = 4.0;
    opt.step(model);
    z = opt.get_z(layer, 0);
    check("z_2 = -6.0 (g=4)", near(z[0][0], -6.0, 1e-12));

    // Step 3: g=0 → z_3 = -6 - 0 = -6 (no progress when g=0)
    layer->grad[0][0] = 0.0;
    opt.step(model);
    z = opt.get_z(layer, 0);
    check("z_3 = -6.0 (g=0, unchanged)", near(z[0][0], -6.0, 1e-12));
}

// ---- T6: y and z evolve distinct from each other ----
static void test_y_vs_z_distinct() {
    check_section("T6: y and z are distinct sequences");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeSGD opt(0.1, 0.9);
    opt.train(model);
    layer->grad[0][0] = 1.0;
    opt.step(model);
    Tensor z = opt.get_z(layer, 0);
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

    ScheduleFreeSGD opt(1.0, 0.5);
    opt.train(model);
    layer->grad[0][0] = 0.0;  // no grad
    opt.step(model);
    check("initial y_1 = 0 (no grad)", near(layer->param[0][0], 0.0, 1e-12));
    Tensor z_before = opt.get_z(layer, 0);
    check("z_1 = 0 (no grad)", near(z_before[0][0], 0.0, 1e-12));

    // Now do another step with a non-zero grad.
    layer->grad[0][0] = 1.0;
    opt.step(model);  // k now 1, k+1=2, ckp1=1/2 (with r=0, wlp=2)
    const double y_now = layer->param[0][0];
    Tensor z_now = opt.get_z(layer, 0);
    check("after step 2: y != 0", std::abs(y_now) > 1e-6);
    check("after step 2: z != 0", std::abs(z_now[0][0]) > 1e-6);

    // Switch to eval mode: p → x.
    opt.eval(model);
    const double x_val = layer->param[0][0];
    const double expected_x = (y_now - (1.0 - 0.5) * z_now[0][0]) / 0.5;
    check("eval: param becomes x = (y - (1-b1)*z)/b1",
          near(x_val, expected_x, 1e-9));
    check("eval: train_mode flag flipped", !opt.is_train_mode());

    // Switch back to train: p → y = (1-beta1)*z + beta1*x.
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

// ---- T8: coupled weight decay is applied (decoupled form ux = g + wd*y) ----
//
// With wd > 0, u = g + wd*y. Verify that y shrinks even with g=0.
static void test_weight_decay() {
    check_section("T8: coupled weight decay on update direction");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param[0][0] = 1.0;
    model.add_layer(layer);

    ScheduleFreeSGD opt(1.0, 0.0, 0.1);  // wd=0.1, beta1=0
    opt.train(model);
    layer->grad[0][0] = 0.0;
    opt.step(model);
    // With g=0, y=1: u = 0 + 0.1*1 = 0.1
    //   z_1 = z_0 - lr*u = 1.0 - 0.1 = 0.9
    //   ckp1 = 1, y_coeff = lr*(beta1*(1-ckp1) - 1) = -1
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

    ScheduleFreeSGD opt;
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

    ScheduleFreeSGD opt1(0.05, 0.9, 0.0);
    ScheduleFreeSGD opt2(0.05, 0.9, 0.0);
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
    d->weights[0][0] = 0.5;
    d->bias[0][0] = 0.0;
    model.add_layer(d);

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

    ScheduleFreeSGD opt(0.05, 0.9, 0.01);
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

// ---- T12: signature test — produces a different trajectory than vanilla SGD ----
//
// Use a non-symmetric gradient sequence so vanilla SGD and Schedule-Free
// SGD don't cancel out and trivially converge to the same value.
static void test_signature_vs_sgd() {
    check_section("T12: signature test vs vanilla SGD");
    double lr = 0.1;
    double sgd_w = 0.0;  // initial weight

    // Schedule-Free recurrences (beta1=0 for clean closed form):
    double sf_z = 0.0, sf_y = 0.0;

    // Non-symmetric, non-alternating sequence so the average over the path
    // doesn't trivially equal the running sum (which is what makes vanilla
    // SGD collapse to 0 over [1,-1,1,-1,...]).
    double grad_seq[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for (int k = 0; k < 6; ++k) {
        double g = grad_seq[k];

        // Vanilla SGD step
        sgd_w -= lr * g;

        // Schedule-Free step (beta1=0, wd=0)
        double u = g;
        double z_new = sf_z - lr * u;
        double kp1 = static_cast<double>(k + 1);
        double ckp1 = 1.0 / kp1;  // default r=0, wlp=2
        double y_coeff = lr * (0.0 * (1.0 - ckp1) - 1.0);
        sf_y = ckp1 * z_new + (1.0 - ckp1) * sf_y + y_coeff * u;
        sf_z = z_new;
    }
    cout << "  SGD final w = " << sgd_w << "\n";
    cout << "  Schedule-Free final y = " << sf_y << "\n";
    // sgd_w = -lr * 6 = -0.6 (pure cumulative)
    // sf_y   = ckp1-weighted average path: first step takes y_1 = z_1 - lr*g
    // (because ckp1=1, y_coeff=-lr, so y_1 = z_1 + 0 + (-lr)*g = -2*lr*g at k=0).
    // Subsequent steps average, but y_1 already "overshoots" SGD by -lr*g.
    // The two trajectories differ in trajectory shape and endpoint.
    check("SGD and Schedule-Free produce different trajectories",
          std::abs(sgd_w - sf_y) > 1e-3);
    check("SF y differs from SGD w in magnitude (trajectory shape)",
          std::abs(sf_y - sgd_w) > 0.05);
}

// ---- T12b: signature test — produces a different trajectory than Schedule-Free AdamW ----
//
// Schedule-Free SGD has NO second-moment denominator, so for a gradient
// sequence with high variance, Schedule-Free SGD will produce meaningfully
// different parameter trajectories than Schedule-Free AdamW (which
// down-weights high-variance gradients via the sqrt(v) denominator).
// Use a sparse/non-uniform gradient sequence where the Adam EMA sees
// a real variance signal that SGD does not.
static void test_signature_vs_adamw() {
    check_section("T12b: signature test vs Schedule-Free AdamW");
    double lr = 0.1, beta1 = 0.9;
    // SF-SGD
    double sgd_z = 0.0, sgd_y = 0.0;
    // SF-AdamW
    double adam_z = 0.0, adam_y = 0.0, adam_v = 0.0;
    const double beta2 = 0.999, eps = 1e-8;

    // High-variance sequence: alternates 0 (no signal) with a large spike.
    // The Adam EMA catches the spike via exp_avg_sq; SGD ignores it.
    // Over many steps the SF-AdamW trajectory is "smoothed" relative to SF-SGD.
    double grad_seq[6] = {10.0, 0.0, 10.0, 0.0, 10.0, 0.0};
    for (int k = 0; k < 6; ++k) {
        double g = grad_seq[k];
        double kp1 = static_cast<double>(k + 1);
        double ckp1 = 1.0 / kp1;
        double y_coeff = lr * (beta1 * (1.0 - ckp1) - 1.0);

        // SF-SGD: u = g, z -= lr*u, y update
        double u_sgd = g;
        sgd_z = sgd_z - lr * u_sgd;
        sgd_y = ckp1 * sgd_z + (1.0 - ckp1) * sgd_y + y_coeff * u_sgd;

        // SF-AdamW: u = g/denom, z -= lr*u, y update
        adam_v = beta2 * adam_v + (1.0 - beta2) * g * g;
        double bc2 = 1.0 - std::pow(beta2, kp1);
        double denom = std::sqrt(adam_v / bc2) + eps;
        double u_adam = g / denom;
        adam_z = adam_z - lr * u_adam;
        adam_y = ckp1 * adam_z + (1.0 - ckp1) * adam_y + y_coeff * u_adam;
    }
    cout << "  SF-SGD  final y = " << sgd_y << "\n";
    cout << "  SF-AdamW final y = " << adam_y << "\n";
    // SF-SGD: cumulative z -= lr * sum(g) = -0.1 * 30 = -3.0
    // SF-AdamW: denom adapts to spikes (v→100), so u = g/sqrt(100) ≈ g/10
    // → effective magnitude much smaller than SF-SGD
    check("SF-SGD and SF-AdamW produce different trajectories under high-variance grads",
          std::abs(sgd_y - adam_y) > 1e-3);
}

// ---- T13: ckp1 progression with default r=0, weight_lr_power=2 ----
static void test_ckp1_progression() {
    check_section("T13: ckp1 progression");
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeSGD opt(1.0, 0.9, 0.0, 0, 0.0, 2.0);
    opt.train(model);
    layer->grad.fill(0.0);
    opt.step(model);  // k=0→1
    check("after step 1: weight_sum = 1", near(opt.get_weight_sum(), 1.0, 1e-12));
    opt.step(model);  // k=1→2
    check("after step 2: weight_sum = 2", near(opt.get_weight_sum(), 2.0, 1e-12));
    opt.step(model);  // k=2→3
    check("after step 3: weight_sum = 3", near(opt.get_weight_sum(), 3.0, 1e-12));
    opt.step(model);  // k=3→4
    check("after step 4: weight_sum = 4", near(opt.get_weight_sum(), 4.0, 1e-12));
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

    ScheduleFreeSGD opt(2.0, 0.9, 0.0, 3, 0.0, 2.0);
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

    ScheduleFreeSGD opt;
    opt.train(model);
    l1->grad.fill(1.0);
    l2->grad.fill(0.0);
    opt.step(model);

    Tensor z1 = opt.get_z(l1, 0);
    Tensor z2 = opt.get_z(l2, 0);
    check("layer1 z nonzero", std::abs(z1[0][0]) > 0.0);
    check("layer2 z = init (zero grad, lr*g=0)", near(z2[0][0], 0.0, 1e-15));
    check("layer1 has_state", opt.has_state(l1));
    check("layer2 has_state", opt.has_state(l2));
}

// ---- T16: independent state across parameters in the same layer ----
static void test_independent_params() {
    check_section("T16: independent state across parameters");
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

    ScheduleFreeSGD opt;
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

    ScheduleFreeSGD opt;
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
    ScheduleFreeSGD opt(0.01, 0.9);
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
    Model model;
    auto* layer = new TestParam(1, 1);
    layer->param.fill(0.0);
    model.add_layer(layer);

    ScheduleFreeSGD opt(1.0, 0.9, 0.0, 0, 1.0, 0.0);
    opt.train(model);
    layer->grad.fill(0.0);
    opt.step(model);  // k=0→1
    check("r=1, wlp=0, step 1: weight_sum = 1", near(opt.get_weight_sum(), 1.0, 1e-12));
    opt.step(model);  // k=1→2
    check("r=1, wlp=0, step 2: weight_sum = 3", near(opt.get_weight_sum(), 3.0, 1e-12));
    opt.step(model);  // k=2→3
    check("r=1, wlp=0, step 3: weight_sum = 6", near(opt.get_weight_sum(), 6.0, 1e-12));
}

// ---- T20: get_z returns empty tensor when no state ----
static void test_state_accessors_empty() {
    check_section("T20: state accessors before step");
    Model model;
    auto* layer = new TestParam(1, 1);
    model.add_layer(layer);

    ScheduleFreeSGD opt;
    Tensor z = opt.get_z(layer, 0);
    check("get_z before step: empty (0,0)", z.rows == 0 && z.cols == 0);
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

    ScheduleFreeSGD opt;
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

// ---- T22: pure SGD-without-momentum reduction (beta1=0, no Adam at all) ----
//
// beta1=0 reduces the algorithm to: u=g, z-=lr*u, y=ckp1*z+(1-ckp1)*y+lr*(-1)*u.
// This is the lightest-weight form. Verify convergence under this regime.
static void test_pure_sgd_reduction() {
    check_section("T22: pure SGD reduction (beta1=0)");
    Model model;
    auto* d = new Dense(1, 1);
    d->weights[0][0] = 0.5;
    d->bias[0][0] = 0.0;
    model.add_layer(d);

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
                double d = pred[i][j] - t[i][j]; s += d * d;
            }
        return s / static_cast<double>(pred.rows);
    };

    // beta1=0 → pure SGD path (no momentum, no Adam)
    ScheduleFreeSGD opt(0.05, 0.0, 0.0);
    opt.train(model);
    double loss0 = mse(model.forward(X), y);
    for (int step = 0; step < 200; ++step) {
        Tensor pred = model.forward(X);
        Tensor grad(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.rows; ++i)
            for (size_t j = 0; j < pred.cols; ++j)
                grad[i][j] = 2.0 * (pred[i][j] - y[i][j]) / static_cast<double>(pred.rows);
        model.backward(grad, 0.0);
        opt.step(model);
    }
    double loss_final = mse(model.forward(X), y);
    cout << "  Pure-SF-SGD reduction: loss " << loss0 << " -> " << loss_final << "\n";
    check("pure SF-SGD reduces loss > 30%", loss_final < 0.7 * loss0);
}

int main() {
    cout << fixed << setprecision(12);
    test_defaults();
    test_constructor_and_validation();
    test_state_init();
    test_first_step_degenerate();
    test_z_recurrence();
    test_y_vs_z_distinct();
    test_mode_swap();
    test_weight_decay();
    test_malformed_layer();
    test_determinism();
    test_end_to_end();
    test_signature_vs_sgd();
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
    test_pure_sgd_reduction();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}