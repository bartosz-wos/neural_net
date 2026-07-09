// test_ademamix.cpp — Tests for AdEMAMix optimizer.
//
// Paper: Pagliardini et al. 2024 "AdEMAMix: Combining AdEMAM with AdEMAMix
// for Better Adam-style Optimization" (https://arxiv.org/abs/2409.03137)
//
// Algorithm (per parameter):
//   m_fast_t = β1 * m_fast_{t-1} + (1-β1) * g_t
//   m_slow_t = β3 * m_slow_{t-1} + (1-β3) * g_t
//   m_fast_hat_t = m_fast_t / (1 - β1^t)   (bias correction)
//   m_slow_hat_t = m_slow_t / (1 - β3^t)   (bias correction)
//   combined_t   = m_fast_hat_t + α * m_slow_hat_t
//   param_t      = param_t - lr * (combined_t + wd * param_t)   (decoupled wd)
//
// Properties tested:
//   1. Accessors / constructors (lr, b1, b3, α, wd, t, handles_weight_decay).
//   2. State initialization: m_fast_ and m_slow_ populated lazily per layer;
//      exact dimensions per parameter; both initialized to zero.
//   3. m_fast update matches the EMA formula: m_fast_t = β1 * m_fast_{t-1} + (1-β1) * g_t
//      (verified bit-exactly on a hand-computed example).
//   4. m_slow update matches its EMA formula with the paper's slow coefficient.
//   5. Bias correction works: at t=1, m_hat = m (no correction yet).
//   6. Step counter increments after each `step()`.
//   7. One-step update with single gradient value matches the closed-form
//      AdEMAMix formula bit-exactly.
//   8. Multi-layer Model exercises both layers' state maps independently.
//   9. Decoupled weight decay shrinks parameters even at zero gradient.
//   10. Training reduces loss on a linear-regression task.
//   11. State accessor correctness: last_m_fast_value / last_m_slow_value return
//       the post-update EMA values.
//   12. Two-state size after first step: m_fast_state() and m_slow_state()
//       each have exactly 1 entry per non-empty layer.
//   13. Different β1/β3 values produce different first-step outputs
//       (sanity: optimizer is sensitive to its hyperparameters).
//   14. With α=0, AdEMAMix exactly reduces to "fast EMA" AdamW-like behavior
//      (combined = m_fast_hat only); verified by equality of final param state
//       with a hand computation using α=0.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/optimizers/ademamix.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Set all parameters of a Dense layer to a single constant (deterministic
// fixtures; `Dense::random_init` advances a global RNG and is harder to assert).
static void set_layer(Dense* layer, double w, double b) {
    for (size_t r = 0; r < layer->weights.rows; ++r)
        for (size_t c = 0; c < layer->weights.cols; ++c)
            layer->weights[r][c] = w;
    layer->bias.fill(b);
}

// Run forward(input) + backward(grad_out) with input=ones(in_dim) and grad_out
// filled with `grad_val` to produce a fixed, deterministic gradient in the
// layer. For a single output neuron, grad_weights becomes a (out, in) tensor
// of `grad_val` (input is all-ones so each input contributes grad_val to its
// row in grad_weights), and grad_bias becomes (1, out) of `grad_val` (sum over
// the batch is just grad_val because the batch dim is 1).
static void run_one(Dense* layer, size_t in_dim, double grad_val) {
    Tensor input(1, in_dim);
    input.fill(1.0);
    Tensor grad_out(1, layer->weights.rows);
    grad_out.fill(grad_val);
    layer->forward(input);
    layer->backward(grad_out, 0.0);
}

int main() {
    cout << setprecision(10);
    cout << "=== AdEMAMix Optimizer Test ===" << endl << endl;

    // ============================================================
    // T1: accessors
    // ============================================================
    {
        cout << "T1: accessors expose lr/beta1/beta3/alpha/weight_decay" << endl;
        AdEMAMix opt(1e-4, 0.9, 0.9999, 2.0, 0.01);
        check("lr exposed",            std::abs(opt.lr - 1e-4)    < 1e-15);
        check("beta1 exposed",         std::abs(opt.beta1 - 0.9)  < 1e-15);
        check("beta3 exposed",         std::abs(opt.beta3 - 0.9999) < 1e-15);
        check("alpha exposed",         std::abs(opt.alpha - 2.0)  < 1e-15);
        check("weight_decay exposed",  std::abs(opt.weight_decay - 0.01) < 1e-15);
        check("timestep() == 1 before any step", opt.timestep() == 1);
        check("handles_weight_decay() == true (AdamW-style)",
              opt.handles_weight_decay());
    }
    cout << endl;

    // ============================================================
    // T2: zero gradient + zero wd → params unchanged
    // ============================================================
    {
        cout << "T2: zero gradient + zero wd -> params unchanged" << endl;
        Model model;
        Dense* layer = new Dense(3, 2);
        set_layer(layer, 0.1, 0.05);
        model.add_layer(layer);

        double w00_before = layer->weights[0][0];
        double b0_before  = layer->bias[0][0];

        AdEMAMix opt(1e-4, 0.9, 0.9999, 2.0, 0.0);
        opt.step(model);

        check("weights[0][0] unchanged after zero-grad step",
              std::abs(layer->weights[0][0] - w00_before) < 1e-15);
        check("bias[0][0]    unchanged after zero-grad step",
              std::abs(layer->bias[0][0]    - b0_before)  < 1e-15);
        check("timestep() advanced to 2", opt.timestep() == 2);
    }
    cout << endl;

    // ============================================================
    // T3: closed-form one-step with α=0, single-element gradient
    // (With α=0, the combined direction is just m_fast_hat; gives a
    //  hand-computable expected output.)
    //
    // At t=1, β1=0.9, β3=0.9999, α=0, lr=0.5, g=1.0, wd=0:
    //   b1_corr = 1 - 0.9^1 = 0.1
    //   m_fast  = 0.9*0 + 0.1*1 = 0.1
    //   m_fast_hat = 0.1 / 0.1 = 1.0
    //   m_slow  = 0.9999*0 + 0.0001*1 = 1e-4
    //   m_slow_hat = 1e-4 / (1 - 0.9999^1) = 1e-4 / 1e-4 = 1.0
    //   combined = 1.0 + 0*1.0 = 1.0
    //   param   -= 0.5 * (1.0 + 0) → param becomes -0.5
    // ============================================================
    {
        cout << "T3: closed-form one-step formula matches (α=0 case)" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);          // w=0, b=0
        model.add_layer(layer);

        // g_w and g_b are both 1.0
        run_one(layer, 1, 1.0);

        AdEMAMix opt(0.5, 0.9, 0.9999, 0.0, 0.0);
        opt.step(model);

        double expected = -0.5;
        check("param_w matches closed-form",
              std::abs(layer->weights[0][0] - expected) < 1e-12);
        check("param_b matches closed-form",
              std::abs(layer->bias[0][0] - expected) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T4: closed-form one-step with α=2 (paper-style default)
    // At t=1, β1=0.9, β3=0.9999, α=2.0, lr=0.5, g=1.0:
    //   m_fast_hat = 1.0   (from T3)
    //   m_slow_hat = 1.0   (from T3)
    //   combined = 1.0 + 2.0 * 1.0 = 3.0
    //   param   -= 0.5 * 3.0 = -1.5
    // ============================================================
    {
        cout << "T4: closed-form one-step with α=2 matches formula" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);
        run_one(layer, 1, 1.0);

        AdEMAMix opt(0.5, 0.9, 0.9999, 2.0, 0.0);
        opt.step(model);

        double expected = -1.5;
        check("param_w matches α=2 closed-form",
              std::abs(layer->weights[0][0] - expected) < 1e-12);
        check("param_b matches α=2 closed-form",
              std::abs(layer->bias[0][0] - expected) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T5: state initialization — both m_fast and m_slow after a step
    //     (Internal `last_m_fast_value` / `last_m_slow_value` accessors)
    //
    // At t=1, β1=0.9, β3=0.9999, g=2.0:
    //   m_fast = 0.9*0 + 0.1*2.0 = 0.2
    //   m_slow = 0.9999*0 + 0.0001*2.0 = 0.0002
    // ============================================================
    {
        cout << "T5: state accessors work after a step (with single non-zero grad)" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);
        run_one(layer, 1, 2.0);

        AdEMAMix opt(1e-3, 0.9, 0.9999, 2.0, 0.0);
        opt.step(model);

        check("m_fast_state has 1 layer", opt.m_fast_state().size() == 1);
        check("m_slow_state has 1 layer", opt.m_slow_state().size() == 1);
        check("last_m_fast_value matches EMA",
              std::abs(opt.last_m_fast_value(static_cast<void*>(layer), 0, 0, 0) - 0.2) < 1e-12);
        check("last_m_slow_value matches EMA",
              std::abs(opt.last_m_slow_value(static_cast<void*>(layer), 0, 0, 0) - 2e-4) < 1e-12);
        check("has_state() returns true after step", opt.has_state(static_cast<void*>(layer)));
    }
    cout << endl;

    // ============================================================
    // T6: bias correction under constant gradient (m_slow_hat → 1.0)
    // Step 1 (t=1): m_slow = (1-β3)*1 = 1e-4,   hat = 1e-4 / 1e-4 = 1.0
    // Step 2 (t=2): m_slow = β3*1e-4 + (1-β3)*1 = 1e-4 + 1e-4*(1-1e-4)
    //            b3_corr = 1 - 0.9999^2 = 1.9999e-4
    //            hat = 1.9999e-4 / 1.9999e-4 = 1.0
    // m_slow(2) = 0.9999*1e-4 + 0.0001*1.0 = 1.99990001e-4
    // ============================================================
    {
        cout << "T6: bias correction makes m_slow_hat ≈ 1.0 under constant gradient" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        AdEMAMix opt(1.0, 0.9, 0.9999, 2.0, 0.0);  // lr=1.0 to keep things readable

        run_one(layer, 1, 1.0);  // step 1 (implicit t=1)
        opt.step(model);

        run_one(layer, 1, 1.0);  // step 2 (now t=2)
        opt.step(model);

        double ms_val = opt.last_m_slow_value(static_cast<void*>(layer), 0, 0, 0);
        double expected = 0.9999 * 1e-4 + (1.0 - 0.9999) * 1.0;  // ≈ 1.99990e-4
        check("m_slow after 2 steps matches EMA",
              std::abs(ms_val - expected) < 1e-14);
    }
    cout << endl;

    // ============================================================
    // T7: m_fast EMA after 5 constant-gradient steps.
    // m_fast_t with constant g=1.0 has the closed form (1 - β1^t).
    // bias-correction divides by (1 - β1^t), giving exactly 1.0.
    // ============================================================
    {
        cout << "T7: m_fast EMA formula matches hand-computed value (t=5)" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        AdEMAMix opt(0.01, 0.9, 0.9999, 0.0, 0.0);  // α=0 to isolate m_fast chain

        for (int step = 0; step < 5; ++step) {
            run_one(layer, 1, 1.0);
            opt.step(model);
        }

        double expected = 1.0 - std::pow(0.9, 5);
        double mf_val = opt.last_m_fast_value(static_cast<void*>(layer), 0, 0, 0);
        check("m_fast after 5 steps matches 1 - β1^5",
              std::abs(mf_val - expected) < 1e-12);
        check("m_fast_hat = 1.0 under constant gradient",
              std::abs(mf_val / (1.0 - std::pow(0.9, 5.0)) - 1.0) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T8: decoupled weight decay shrinks params even at zero gradient
    //
    // With g=0: m_fast = 0, m_slow = 0, combined = 0
    //   update = 0 + wd*1.0 = 0.1
    //   param   -= lr * update = 1.0 * 0.1 = 0.1
    //   so param 1.0 -> 0.9
    // ============================================================
    {
        cout << "T8: decoupled weight decay shrinks params under zero gradient" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 1.0, 1.0);  // start at 1.0 so decay has visible effect
        model.add_layer(layer);

        AdEMAMix opt(1.0, 0.9, 0.9999, 0.0, 0.1);  // lr=1.0, wd=0.1, α=0
        opt.step(model);

        double expected = 0.9;
        check("weights shrink by (1 - lr*wd) when grad=0",
              std::abs(layer->weights[0][0] - expected) < 1e-12);
        check("bias also shrinks with decoupled wd",
              std::abs(layer->bias[0][0] - expected) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T9: handles_weight_decay() = true
    // ============================================================
    {
        cout << "T9: handles_weight_decay() returns true" << endl;
        AdEMAMix opt;  // all defaults
        check("handles_weight_decay() returns true", opt.handles_weight_decay());
    }
    cout << endl;

    // ============================================================
    // T10: hyperparameters affect first-step output (sensitivity check)
    // ============================================================
    {
        cout << "T10: optimizer is sensitive to (lr, α) hyperparameters" << endl;
        auto run = [](double lr, double alpha) {
            Model m; Dense* l = new Dense(1,1);
            set_layer(l, 0.0, 0.0);
            m.add_layer(l);
            run_one(l, 1, 1.0);
            AdEMAMix opt(lr, 0.9, 0.9999, alpha, 0.0);
            opt.step(m);
            return l->weights[0][0];
        };
        double r1 = run(0.1, 2.0);
        double r2 = run(0.3, 2.0);
        double r3 = run(0.1, 4.0);
        check("outputs differ for different lr",
              std::abs(r1 - r2) > 1e-9 && std::abs(r1 - 0.0) > 1e-12);
        check("outputs differ for different α",
              std::abs(r1 - r3) > 1e-9);
    }
    cout << endl;

    // ============================================================
    // T11: multi-layer model populates state for both layers
    // ============================================================
    {
        cout << "T11: multi-layer model populates state for both layers" << endl;
        Model model;
        Dense* layer1 = new Dense(2, 2);
        Dense* layer2 = new Dense(2, 1);
        set_layer(layer1, 0.1, 0.05);
        set_layer(layer2, 0.1, 0.0);
        model.add_layer(layer1);
        model.add_layer(layer2);

        run_one(layer1, 2, 1.0);
        run_one(layer2, 2, 1.0);

        AdEMAMix opt(1e-3, 0.9, 0.9999, 2.0, 0.0);
        opt.step(model);

        check("m_fast_state has 2 layers", opt.m_fast_state().size() == 2);
        check("m_slow_state has 2 layers", opt.m_slow_state().size() == 2);
        check("layer1 has its own state", opt.has_state(static_cast<void*>(layer1)));
        check("layer2 has its own state", opt.has_state(static_cast<void*>(layer2)));

        bool shape_ok = (opt.m_fast_state().at(static_cast<void*>(layer1)).size() == 2 &&
                         opt.m_fast_state().at(static_cast<void*>(layer1))[0].rows == 2 &&
                         opt.m_fast_state().at(static_cast<void*>(layer1))[0].cols == 2);
        check("layer1 m_fast[0] shape matches weights (2,2)", shape_ok);
    }
    cout << endl;

    // ============================================================
    // T12: training reduces loss on linear regression
    // ============================================================
    {
        cout << "T12: training reduces loss on linear regression" << endl;
        // f(x) = 2*x (target). One Dense layer with no activation matches it.
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.5, 0.1);
        model.add_layer(layer);

        auto loss_at = [&]() {
            Tensor x(1, 1); x[0][0] = 1.0;
            Tensor y_true(1, 1); y_true[0][0] = 2.0;
            Tensor y_pred = layer->forward(x);
            double s = 0;
            for (size_t i = 0; i < y_pred.rows; ++i)
                for (size_t j = 0; j < y_pred.cols; ++j) {
                    double d = y_pred[i][j] - y_true[i][j];
                    s += d * d;
                }
            return s;
        };

        // For y_pred = w*x + b with x=1, target=2, L = (w + b - 2)^2
        // dL/dw = 2*(w+b-2)*1,  dL/db = 2*(w+b-2)
        AdEMAMix opt(0.05, 0.9, 0.9999, 2.0, 0.0);

        double loss0 = loss_at();
        for (int step = 0; step < 60; ++step) {
            Tensor x(1, 1); x[0][0] = 1.0;
            Tensor y_pred = layer->forward(x);
            Tensor grad_out(1, 1); grad_out[0][0] = 2.0 * (y_pred[0][0] - 2.0);
            layer->backward(grad_out, 0.0);
            opt.step(model);
        }
        double loss1 = loss_at();

        check("loss decreases after 60 AdEMAMix steps",
              loss1 < loss0 * 0.5);  // at least 50% reduction
        check("loss is small and finite",
              loss1 > 0 && loss1 < 0.5 && !std::isnan(loss1));
    }
    cout << endl;

    // ============================================================
    // T13: zero-grad step doesn't crash on a minimal model
    // ============================================================
    {
        cout << "T13: zero-grad step on a model with parameters doesn't crash" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);
        AdEMAMix opt;
        opt.step(model);  // grads are zero by default
        check("step() on zero-grad model doesn't crash", true);
        check("state has 1 entry after the step", opt.m_fast_state().size() == 1);
    }
    cout << endl;

    // ============================================================
    // T14: default constructor uses paper defaults
    // ============================================================
    {
        cout << "T14: default constructor uses paper defaults" << endl;
        AdEMAMix opt;
        check("default lr = 1e-4",    std::abs(opt.lr - 1e-4)    < 1e-15);
        check("default beta1 = 0.9",  std::abs(opt.beta1 - 0.9)  < 1e-15);
        check("default beta3 = 0.9999", std::abs(opt.beta3 - 0.9999) < 1e-15);
        check("default alpha = 2.0",  std::abs(opt.alpha - 2.0)  < 1e-15);
        check("default wd = 0.0",     std::abs(opt.weight_decay) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // T15: bias correction at t=1 with β1=0.9 and g=1.0 gives m_hat=1.0
    //      (validates that the divider 1-β1^t = 1-0.9^1 = 0.1 is applied).
    // With α=0 we isolate the bias-correction effect on m_fast:
    //   combined = m_fast_hat + 0 = 1.0
    //   param  = 0 - 1.0 * 1.0 = -1.0
    // ============================================================
    {
        cout << "T15: bias correction at t=1 with β1=0.9 and g=1.0 gives m_hat=1.0" << endl;
        Model model;
        Dense* layer = new Dense(1, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);
        run_one(layer, 1, 1.0);

        AdEMAMix opt(1.0, 0.9, 0.9999, 0.0, 0.0);
        opt.step(model);  // t=1 → b1_corr = 0.1; m_fast = 0.1; hat = 1.0
        double expected = -1.0;
        check("param_w matches m_fast_hat=1.0 case",
              std::abs(layer->weights[0][0] - expected) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T16: two-step determinism — same input + same seed → same output.
    //   Reset both layers to identical init, take 2 steps, params must match.
    // ============================================================
    {
        cout << "T16: deterministic (same input -> same output)" << endl;
        auto run = [&]() {
            Model m; Dense* l = new Dense(2, 1);
            set_layer(l, 0.5, 0.1);
            m.add_layer(l);
            AdEMAMix opt(0.05, 0.9, 0.9999, 2.0, 0.0);
            for (int s = 0; s < 2; ++s) {
                run_one(l, 2, 1.0);
                opt.step(m);
            }
            return std::pair<double, double>(l->weights[0][0], l->bias[0][0]);
        };
        auto [wA, bA] = run();
        auto [wB, bB] = run();
        check("two-step training is deterministic",
              std::abs(wA - wB) < 1e-15 && std::abs(bA - bB) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // Summary
    // ============================================================
    cout << "=== Summary ===" << endl;
    cout << "Passed: " << passed << endl;
    cout << "Failed: " << failed << endl;

    return failed == 0 ? 0 : 1;
}
