// test_sophia.cpp — Tests for Sophia (Liu et al. 2023) optimizer
// Paper: "Sophia: A Scalable Stochastic Second-order Optimizer" (https://arxiv.org/abs/2305.11242)
//
// Sophia update rule (per parameter θ):
//   g_t    = grad_t                                            (gradient at step t)
//   m_t    = β1 * m_{t-1} + (1 - β1) * g_t                     (first-moment EMA)
//   m̂_t   = m_t / (1 - β1^t)                                  (bias correction)
//   h_t    = β2 * h_{t-1} + (1 - β2) * h_diag_t                (diagonal Hessian EMA,
//                                                              re-estimated every k steps)
//   update = clip(m̂_t / max(h_t, ε), -ρ, ρ)                    (clipped update direction)
//   θ     -= lr * (update + wd * θ)                            (decoupled weight decay)
//
// Where:
//   - h_diag_t is the per-coordinate diagonal of the loss Hessian
//     (or Gauss-Newton / empirical Fisher surrogate).
//   - For testability this implementation supports TWO sources of h_diag:
//       (1) External: the user calls `set_hessian_estimates(layer_ptr, h_diag)`
//           before step() — typically with a Hutchinson trace estimate.
//       (2) Default (Sophia-G style): h_diag_t = g_t ⊙ g_t — the empirical Fisher.
//   - ρ is the clipping bound (paper default ρ = 1).
//
// Key properties tested:
//   - Zero grad + zero wd: params unchanged exactly
//   - With wd > 0, zero grad still shrinks params
//   - Default h_diag source: empirical Fisher (g ⊙ g)
//   - External h_diag overrides empirical Fisher
//   - Update magnitude is clipped at ±ρ
//   - Hessian EMA smooths over time (h_t converges to g^2 under constant grad)
//   - Bias correction on m̂_t at first step
//   - Memory-efficient state: TWO Tensors per parameter (m and h)
//   - Converges on a simple linear regression
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/optimizers/sophia.h"
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

int main() {
    cout << setprecision(10);
    cout << "=== Sophia Optimizer Test ===" << endl << endl;

    // ---------------------------------------------------------------
    // Test 1: Zero gradient + zero wd -> params unchanged exactly
    // ---------------------------------------------------------------
    cout << "Test 1: Zero gradient + zero wd -> params unchanged" << endl;
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        double init_w00 = layer->weights[0][0];
        double init_b0  = layer->bias[0][0];

        Sophia opt(0.1, 0.9, 0.99, 1e-12, 1.0, 10);  // lr=0.1, b1=0.9, b2=0.99, eps=1e-12, rho=1.0, k=10
        opt.step(model);

        bool ok = std::abs(layer->weights[0][0] - init_w00) < 1e-12
               && std::abs(layer->bias[0][0]    - init_b0)  < 1e-12;
        check("zero-grad step preserves params exactly", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 2: With weight decay, zero gradient still shrinks params
    //   param -= lr * (0 + wd * param) => param_new = param - lr*wd*param = param*(1 - lr*wd)
    // ---------------------------------------------------------------
    cout << "Test 2: Weight decay shrinks params even with zero gradient" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        layer->weights[0][0] = 1.0;
        layer->weights[0][1] = 0.5;
        layer->bias[0][0] = -0.3;
        model.add_layer(layer);

        Sophia opt(0.1, 0.9, 0.99, 1e-12, 1.0, 10);
        opt.weight_decay = 0.1;
        opt.step(model);  // zero gradient, but wd applies
        double w_after = layer->weights[0][0];

        // param_new = 1.0 - 0.1 * (0 + 0.1 * 1.0) = 1.0 - 0.01 = 0.99
        bool ok = std::abs(w_after - 0.99) < 1e-12;
        check("wd=0.1 produces exact 1.0 -> 0.99 on zero gradient", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 3: Default Hessian source is empirical Fisher (g ⊙ g)
    //   After one step with constant positive grad g:
    //     h_t  = (1 - b2) * g^2                (first EMA, no h_prev)
    //     m_t  = (1 - b1) * g                  (first EMA, no m_prev)
    //     m̂_t = m_t / (1 - b1) = g             (bias correction at t=1)
    //     update = clip(g / max(h_t, eps), -rho, rho)
    //            = clip(g / max((1-b2)*g^2, eps), -rho, rho)
    //   For g = 1.0, b2 = 0.99: h_t = 0.01, so update = clip(1.0/0.01, -1, 1) = 1.0 (clipped)
    //   param_new = param - lr * 1.0 = param - 0.1
    // ---------------------------------------------------------------
    cout << "Test 3: Default empirical-Fisher Hessian -> clipped update direction" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.5;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        // Build a fixed gradient by doing forward+backward
        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;

        Sophia opt(0.1, 0.9, 0.99, 1e-12, 1.0, 10);
        double w_before = layer->weights[0][0];
        layer->forward(input);
        layer->backward(grad_output, 0.0);
        opt.step(model);
        double w_after = layer->weights[0][0];

        // dL/dW = grad_output * input = 1.0 * 1.0 = 1.0
        // m_1 = 0.1 * 1.0 = 0.1; m̂_1 = 0.1 / (1-0.9) = 1.0
        // h_1 = 0.01 * 1.0^2 = 0.01
        // update = clip(1.0 / max(0.01, 1e-12), -1, 1) = clip(100, -1, 1) = 1.0
        // param_new = 0.5 - 0.1 * 1.0 = 0.4
        bool ok = std::abs(w_after - 0.4) < 1e-12;
        check("step-1 default (g=1.0, h=g^2) update clipped to +1 -> w decreases by exactly lr",
              ok);
        cout << "  w before: " << w_before << "  w after: " << w_after
             << "  diff: " << (w_after - w_before) << endl;
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 4: External Hessian estimate overrides empirical Fisher
    //   With grad g=1.0 and h_diag=4.0 supplied externally:
    //     m_1 = 0.1 * 1.0 = 0.1; m̂_1 = 1.0
    //     h_1 = 0.01 * 4.0 = 0.04
    //     update = clip(1.0 / max(0.04, eps), -1, 1) = clip(25, -1, 1) = 1.0 (clipped)
    //   param_new = 0.5 - 0.1 * 1.0 = 0.4 (still clipped at +1, but h_t differs)
    //   We verify by also checking h_state_: after step, h[0][0] should be 0.04 (the EMA of the
    //   external estimate), NOT 0.01 (the empirical-Fisher default).
    // ---------------------------------------------------------------
    cout << "Test 4: External Hessian estimates override empirical Fisher" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.5;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;
        layer->forward(input);
        layer->backward(grad_output, 0.0);

        // Set external Hessian: h_diag = 4.0 for the weights, 4.0 for the bias.
        // We can call set_hessian_estimates with a vector per parameter.
        vector<Tensor> h_diag;
        Tensor h_w(1, 1); h_w[0][0] = 4.0;
        Tensor h_b(1, 1); h_b[0][0] = 4.0;
        h_diag.push_back(h_w);
        h_diag.push_back(h_b);

        Sophia opt(0.1, 0.9, 0.99, 1e-12, 1.0, 10);
        opt.set_hessian_estimates(layer, h_diag);
        double w_before = layer->weights[0][0];
        opt.step(model);
        double w_after = layer->weights[0][0];

        // After step:
        //   h_t = 0.01 * 4.0 = 0.04 (NOT 0.01 from empirical Fisher)
        // We can verify by inspecting the optimizer's h_state.
        double h_stored = opt.last_h_value(layer, 0, 0, 0);  // layer, param_idx=0 (W), row=0, col=0
        cout << "  h stored after step with external h=4.0: " << h_stored << endl;

        // step effect on weights: m̂_1 = 1.0, h_t = 0.04, update = clip(25, -1, 1) = 1.0
        // param_new = 0.5 - 0.1 * 1.0 = 0.4
        bool ok_w = std::abs(w_after - w_before) - std::abs(-0.1) < 1e-12;
        bool ok_h = std::abs(h_stored - 0.04) < 1e-12;
        check("external h_diag used (weights moved by clipped update)", ok_w);
        check("h_state contains the external h_diag (not empirical Fisher)", ok_h);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 5: Clipping at ±ρ bounds the update direction
    //   With small Hessian (h=1e-4) and large momentum (m̂ large),
    //   m̂/h is huge -> clip to ±ρ = ±1.
    //   After many steps the per-step move should be exactly ±lr.
    // ---------------------------------------------------------------
    cout << "Test 5: Persistent positive grad -> params decrease by exactly lr (clipped)" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.0;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Sophia opt(0.05, 0.9, 0.99, 1e-12, 1.0, 10);
        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;  // positive gradient

        double w_prev = layer->weights[0][0];
        for (int step = 0; step < 5; ++step) {
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
            double w_curr = layer->weights[0][0];
            double diff = w_curr - w_prev;
            cout << "  step " << (step+1) << ": w=" << w_curr << "  delta=" << diff << endl;
            // Under default empirical Fisher (h=g^2=1), update direction saturates at +1 after
            // enough steps, so the steady-state per-step move should be exactly -lr.
            // (First step: h=0.01 -> update=clip(100, -1, 1)=+1 -> -lr per step;
            //  after step 1: h_t = b2*0.01 + (1-b2)*1 ≈ 0.0199, m̂_2 ≈ 1.0, update = clip(50, -1, 1)=+1.)
            bool ok = std::abs(diff - (-0.05)) < 1e-12;
            check("persistent +grad gives -lr per step (step " + to_string(step+1) + ")", ok);
            w_prev = w_curr;
        }
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 6: Hessian EMA — h_t after several steps with constant g converges to g^2
    //   h_t = b2 * h_{t-1} + (1 - b2) * g^2   (with empirical Fisher, h_diag = g^2)
    //   Closed form: h_t = (1 - b2^t) * g^2 + b2^t * 0 = (1 - b2^t) * g^2
    // ---------------------------------------------------------------
    cout << "Test 6: Hessian EMA converges to g^2 under constant gradient" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.0;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 0.5;  // constant g=0.5 -> h_diag = 0.25

        Sophia opt(0.01, 0.9, 0.5, 1e-12, 1.0, 10);  // b2=0.5 for faster convergence in test
        // Run 10 steps with constant g=0.5
        for (int step = 0; step < 10; ++step) {
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
        }

        // After 10 steps with b2=0.5, g=0.5: h_t = (1 - 0.5^10) * 0.25 = 0.9990234375 * 0.25 ≈ 0.249755859
        double h_expected = (1.0 - std::pow(0.5, 10)) * 0.25;
        double h_actual = opt.last_h_value(layer, 0, 0, 0);
        cout << "  h expected: " << h_expected << "  h actual: " << h_actual << endl;
        bool ok = std::abs(h_actual - h_expected) < 1e-9;
        check("h_t = (1 - b2^10) * g^2 with constant gradient", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 7: Bias correction at first step
    //   m̂_1 = m_1 / (1 - b1^1) = (1-b1)*g / (1-b1) = g
    //   m̂_2 = m_2 / (1 - b1^2) = ((1-b1)*g + b1*(1-b1)*g) / (1-b1^2) = g
    //   So m̂_t = g for any t with constant gradient. Verify m_state tracks this.
    // ---------------------------------------------------------------
    cout << "Test 7: First moment is bias-corrected (m̂_t = g under constant grad)" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.0;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 0.7;  // constant g=0.7

        Sophia opt(0.01, 0.9, 0.99, 1e-12, 1.0, 10);
        for (int step = 0; step < 3; ++step) {
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
        }

        // m_state after 3 steps: m_3 = (1-b1)*g * (1 + b1 + b1^2) = 0.1*0.7 * 1.711 = 0.11977
        // We can verify by re-running with stored gradient; alternatively, check that bias-corrected
        // m̂_3 equals 0.7 (constant). We do this by setting external h_diag = 1.0 and verifying
        // the resulting update direction saturates at +1.
        // Simpler: directly check m_state_ value against the closed form.
        double m_expected = (1.0 - 0.9) * 0.7 * (1.0 + 0.9 + 0.9*0.9);
        double m_actual = opt.last_m_value(layer, 0, 0, 0);
        cout << "  m expected: " << m_expected << "  m actual: " << m_actual << endl;
        bool ok = std::abs(m_actual - m_expected) < 1e-9;
        check("m_state after 3 constant-grad steps matches closed form", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 8: handles_weight_decay returns true (so WeightDecay wrapper skips)
    // ---------------------------------------------------------------
    cout << "Test 8: Sophia reports handles_weight_decay() = true" << endl;
    {
        Sophia opt(0.01);
        check("handles_weight_decay() returns true", opt.handles_weight_decay());
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 9: Sophia reduces loss on a simple linear regression
    //   y = 2*x1 - 1*x2 + 0.5; train to predict y from x.
    // ---------------------------------------------------------------
    cout << "Test 9: Sophia reduces loss on simple linear regression" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                layer->weights[r][c] = 0.0;
        layer->bias.fill(0.0);
        model.add_layer(layer);

        Sophia opt(0.05, 0.9, 0.99, 1e-12, 1.0, 10);

        vector<vector<double>> xs = {{1.0, 0.5}, {0.3, 0.8}, {0.7, 0.1}, {0.5, 0.5}};
        vector<double> ys = {2.0*1.0 - 1.0*0.5 + 0.5, 2.0*0.3 - 1.0*0.8 + 0.5,
                              2.0*0.7 - 1.0*0.1 + 0.5, 2.0*0.5 - 1.0*0.5 + 0.5};

        auto compute_loss = [&]() {
            double loss = 0.0;
            for (size_t i = 0; i < xs.size(); ++i) {
                Tensor x(1, 2);
                x[0][0] = xs[i][0]; x[0][1] = xs[i][1];
                Tensor y_hat = layer->forward(x);
                double r = y_hat[0][0] - ys[i];
                loss += r * r;
            }
            return loss / xs.size();
        };

        double loss0 = compute_loss();
        cout << "  initial loss: " << loss0 << endl;
        for (int step = 0; step < 200; ++step) {
            for (size_t i = 0; i < xs.size(); ++i) {
                Tensor x(1, 2);
                x[0][0] = xs[i][0]; x[0][1] = xs[i][1];
                Tensor y_hat = layer->forward(x);
                double r = y_hat[0][0] - ys[i];
                Tensor grad(1, 1);
                grad[0][0] = 2.0 * r / xs.size();
                layer->backward(grad, 0.0);
                opt.step(model);
            }
        }
        double loss1 = compute_loss();
        cout << "  final loss:   " << loss1 << endl;
        cout << "  reduction:    " << (1.0 - loss1 / loss0) * 100.0 << "%" << endl;
        bool ok = loss1 < loss0 * 0.5;  // at least 50% loss reduction
        check("Sophia reduces loss by >= 50% on linear regression", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 10: Sophia parameters remain finite across many random-gradient steps
    // ---------------------------------------------------------------
    cout << "Test 10: Sophia runs without crashing across many steps" << endl;
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        Sophia opt(0.01, 0.9, 0.99, 1e-12, 1.0, 10);

        srand(42);
        for (int step = 0; step < 50; ++step) {
            Tensor input(1, 3);
            for (int i = 0; i < 3; ++i) input[0][i] = (double)(rand() % 100) / 100.0;
            Tensor grad_output(1, 2);
            for (int i = 0; i < 2; ++i) grad_output[0][i] = (double)((rand() % 200) - 100) / 100.0;
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
        }
        bool all_finite = true;
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                if (!std::isfinite(layer->weights[r][c])) all_finite = false;
        check("parameters remain finite after 50 random-gradient steps", all_finite);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 11: Persistent negative gradient -> params increase by exactly lr per step
    //   (mirrors Test 5 with negative grad)
    // ---------------------------------------------------------------
    cout << "Test 11: Persistent negative grad -> params increase by exactly lr per step" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.0;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Sophia opt(0.05, 0.9, 0.99, 1e-12, 1.0, 10);
        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = -1.0;  // negative grad

        for (int step = 0; step < 5; ++step) {
            double w_prev = layer->weights[0][0];
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
            double w_curr = layer->weights[0][0];
            double diff = w_curr - w_prev;
            // Under default empirical Fisher, update direction is m̂/h.
            // With negative g, m̂ is negative, so update clips to -1, and param -= lr * (-1) = +lr.
            bool ok = std::abs(diff - 0.05) < 1e-12;
            check("persistent -grad gives +lr per step (step " + to_string(step+1) + ")", ok);
        }
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 12: External Hessian with a large value correctly clips the update at -ρ
    //   With grad g=1.0 (m̂_1 = 1.0) and h_diag = 100 (h_t = 1.0 after EMA),
    //   update = clip(1.0/1.0, -1, 1) = 1.0, param_new = 0.5 - 0.1*1.0 = 0.4.
    //   Same as default behavior here, but verify h_state captures the right value.
    // ---------------------------------------------------------------
    cout << "Test 12: Large external Hessian -> update magnitude saturates correctly" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.5;
        layer->bias[0][0] = 0.0;
        model.add_layer(layer);

        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;
        layer->forward(input);
        layer->backward(grad_output, 0.0);

        // Large external h_diag: h_t = 0.01 * 100 = 1.0
        vector<Tensor> h_diag;
        Tensor h_w(1, 1); h_w[0][0] = 100.0;
        Tensor h_b(1, 1); h_b[0][0] = 100.0;
        h_diag.push_back(h_w);
        h_diag.push_back(h_b);

        Sophia opt(0.1, 0.9, 0.99, 1e-12, 1.0, 10);
        opt.set_hessian_estimates(layer, h_diag);
        opt.step(model);

        // h_stored = 0.01 * 100 = 1.0
        double h_stored = opt.last_h_value(layer, 0, 0, 0);
        bool ok_h = std::abs(h_stored - 1.0) < 1e-12;
        // update = clip(1.0 / max(1.0, 1e-12), -1, 1) = clip(1.0, -1, 1) = 1.0
        // param_new = 0.5 - 0.1 * 1.0 = 0.4
        bool ok_w = std::abs(layer->weights[0][0] - 0.4) < 1e-12;
        check("h_state = (1-b2) * external_h = 0.01 * 100 = 1.0", ok_h);
        check("update direction = +1 (clipped to rho), weights move by -lr", ok_w);
        cout << endl;
    }

    cout << "=== Sophia: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
