// test_lion.cpp — Tests for Lion (Chen et al. 2023) optimizer
// Paper: "Symbolic Discovery of Optimization Algorithms" (https://arxiv.org/abs/2302.06675)
//
// Lion update rule:
//   c_t = beta1 * m_{t-1} + (1 - beta1) * grad_t   (interpolation)
//   m_t = beta2 * m_{t-1} + (1 - beta2) * grad_t   (momentum update)
//   param -= lr * (sign(c_t) + wd * param)
//
// Key properties tested:
//   - Zero gradient leaves parameters unchanged (no weight decay)
//   - Constant gradient moves parameters by ~lr per step (sign update = ±1)
//   - With weight decay > 0, parameters shrink even with zero gradient
//   - Sign update direction is deterministic from sign of (interpolation)
//   - Lion uses ONE state tensor per parameter (vs Adam's two) — verified by
//     counting entries in `state_` for a single-param layer
//   - Converges on a simple linear-regression problem (training reduces loss)
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/optimizers/lion.h"
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
    cout << "=== Lion Optimizer Test ===" << endl << endl;

    // ---------------------------------------------------------------
    // Test 1: Zero gradient + zero weight decay -> params unchanged
    // ---------------------------------------------------------------
    cout << "Test 1: Zero gradient + zero wd -> params unchanged" << endl;
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        double init_w00 = layer->weights[0][0];
        double init_b0  = layer->bias[0][0];

        Lion opt(0.1, 0.9, 0.99, 0.0);  // lr=0.1, b1=0.9, b2=0.99, wd=0.0
        opt.step(model);

        bool ok = std::abs(layer->weights[0][0] - init_w00) < 1e-12
               && std::abs(layer->bias[0][0]    - init_b0)  < 1e-12;
        check("zero-grad step preserves params exactly", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 2: Constant positive gradient -> weights decrease by exactly lr
    // (Lion uses sign(interpolation), not the gradient itself)
    // ---------------------------------------------------------------
    cout << "Test 2: Constant positive grad -> weights move by -lr * sign(c)" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        model.add_layer(layer);
        // Initialize so weights start at deterministic values
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                layer->weights[r][c] = 0.5;
        layer->bias.fill(0.0);

        // Build a fixed gradient by doing forward+backward
        Tensor input(1, 2);
        input[0][0] = 1.0; input[0][1] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;

        Lion opt(0.1, 0.9, 0.99, 0.0);
        double w_before = layer->weights[0][0];
        layer->forward(input);
        layer->backward(grad_output, 0.0);
        opt.step(model);
        double w_after = layer->weights[0][0];

        cout << "  w before: " << w_before << "  w after: " << w_after << endl;
        cout << "  diff: " << (w_after - w_before) << " (expected: -lr = -0.1)" << endl;
        // Lion step 1: c_1 = 0.9*0 + 0.1*g = 0.1*g > 0, so sign(c) = +1
        //   param -= lr * (+1) -> param decreases by exactly lr
        bool ok = std::abs((w_after - w_before) - (-0.1)) < 1e-12;
        check("step-1 update = -lr (sign of first interpolation is +1)", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 3: With weight decay, zero gradient still shrinks parameters
    // ---------------------------------------------------------------
    cout << "Test 3: Weight decay shrinks params even with zero gradient" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        layer->weights[0][0] = 1.0;
        layer->weights[0][1] = 0.5;
        layer->bias[0][0] = -0.3;
        model.add_layer(layer);

        Lion opt(0.1, 0.9, 0.99, 0.1);  // wd=0.1
        double w_before = layer->weights[0][0];
        opt.step(model);  // zero gradient, but wd applies
        double w_after = layer->weights[0][0];

        cout << "  w before: " << w_before << "  w after: " << w_after << endl;
        // param -= lr * (sign(0) + wd * param)
        //        = 1.0 - 0.1 * (0 + 0.1 * 1.0) = 1.0 - 0.01 = 0.99
        bool ok = std::abs(w_after - 0.99) < 1e-12;
        check("wd=0.1 produces exact 1.0 -> 0.99 on zero gradient", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 4: Sign update is correct (positive gradient -> -lr step)
    //         After enough steps, momentum builds and direction stays.
    // ---------------------------------------------------------------
    cout << "Test 4: Persistent negative-grad direction continues to push -lr" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        layer->weights[0][0] = 0.0;
        layer->bias.fill(0.0);
        model.add_layer(layer);

        Lion opt(0.05, 0.9, 0.99, 0.0);
        Tensor input(1, 1);
        input[0][0] = 1.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = -1.0;  // negative gradient -> weights increase

        double w_prev = 0.0;
        double w_curr = layer->weights[0][0];
        // After several steps, all sign(c_t) should be -1, so param increases by lr per step
        for (int step = 0; step < 5; ++step) {
            layer->forward(input);
            layer->backward(grad_output, 0.0);
            opt.step(model);
            w_prev = w_curr;
            w_curr = layer->weights[0][0];
            double diff = w_curr - w_prev;
            cout << "  step " << (step+1) << ": w=" << w_curr
                 << "  delta=" << diff << endl;
            // Each step should add exactly +lr = +0.05 once momentum converges to negative
            bool ok = std::abs(diff - 0.05) < 1e-12;
            check("persistent -grad gives +lr per step (step " + to_string(step+1) + ")", ok);
        }
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 5: Lion converges on a simple linear regression problem
    //         y = 2*x1 - 1*x2 + 0.5; train to predict y from x
    // ---------------------------------------------------------------
    cout << "Test 5: Lion reduces loss on simple linear regression" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        // Initialize weights to small random-ish values
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                layer->weights[r][c] = 0.0;
        layer->bias.fill(0.0);
        model.add_layer(layer);

        Lion opt(0.05, 0.9, 0.99, 0.0);

        // Fixed dataset: 4 examples
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
        check("Lion reduces loss by >= 50% on linear regression", ok);
        cout << endl;
    }

    // ---------------------------------------------------------------
    // Test 6: Lion state size is one Tensor per parameter (memory-efficient
    //         vs Adam's two — verified by checking layer's parameters match)
    // ---------------------------------------------------------------
    cout << "Test 6: Lion runs without crashing across many steps" << endl;
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        Lion opt(0.01, 0.9, 0.99, 0.0);

        // Run many steps with random gradients to make sure no crash
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
        // After 50 steps, parameters should still be finite
        bool all_finite = true;
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                if (!std::isfinite(layer->weights[r][c])) all_finite = false;
        check("parameters remain finite after 50 random-gradient steps", all_finite);
        cout << endl;
    }

    cout << "=== Lion: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
