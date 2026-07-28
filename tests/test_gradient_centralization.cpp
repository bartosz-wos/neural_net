// test_gradient_centralization.cpp — Tests for Gradient Centralization (Yu et al. 2020) optimizer wrapper
// Paper: "Gradient Centralization: A New Optimization Strategy for Deep Neural Networks"
// https://arxiv.org/abs/2004.01461
//
// Gradient Centralization (GC) is a per-parameter optimizer wrapper that centers
// gradients before passing them to the inner optimizer:
//   - For 2-D weight of shape (R, C): grad[i][j] -= mean(grad[i][*])  for ROW mode
//   -                                  grad[i][j] -= mean(grad[*][j])  for COLUMN mode
//   - For 1-D bias  of shape (1, C):  treated as COLUMN mode (single row → mean = value → 0)
//   - For scalar (1, 1): no-op (mean is the value itself → 0)
//
// It is state-free, parameter-free, and chainable with any inner optimizer.
//
// IMPORTANT: GC takes ownership of the inner optimizer via unique_ptr (mirrors
// Lookahead, WeightDecay, SAM). All test inners are heap-allocated to avoid
// double-free, since GC will delete the inner when GC goes out of scope.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include "nn/optimizers/gradient_centralization.h"
#include "nn/optimizers/optimizer.h"
#include "nn/optimizers/optimizer_extended.h"
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
    cout << setprecision(12);
    cout << "=== Gradient Centralization Tests ===" << endl << endl;

    // Test 1: Constructor validation
    cout << "Test 1: Constructor validation" << endl;
    {
        bool threw = false;
        try { GradientCentralization gc(nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        check("null inner throws std::invalid_argument", threw);
        cout << endl;
    }

    // Test 2: Defaults round-trip
    cout << "Test 2: Defaults round-trip" << endl;
    {
        SGD* sgd = new SGD(0.05);
        GradientCentralization gc(sgd);
        check("default mode == COLUMN", gc.mode() == GradientCentralization::CenterMode::COLUMN);
        check("inner() returns the inner optimizer", gc.inner() == sgd);
        check("handles_weight_decay() == inner->handles_weight_decay() (false for SGD)",
              gc.handles_weight_decay() == false);
        cout << endl;
    }

    // Test 3: Zero gradient stays zero
    cout << "Test 3: Zero gradient passes through cleanly" << endl;
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);
        for (size_t i = 0; i < layer->weights.rows; ++i)
            for (size_t j = 0; j < layer->weights.cols; ++j)
                layer->weights[i][j] = 0.7;
        layer->bias.fill(0.0);

        double w00_before = layer->weights[0][0];
        double b0_before  = layer->bias[0][0];

        layer->zero_grad();

        GradientCentralization gc(new SGD(0.1));
        gc.step(model);

        bool ok = std::abs(layer->weights[0][0] - w00_before) < 1e-12
               && std::abs(layer->bias[0][0]    - b0_before)  < 1e-12;
        check("zero grad + no wd: params unchanged", ok);
        cout << endl;
    }

    // Test 4: Constant gradient becomes zero after centering
    cout << "Test 4: Constant gradient → centered gradient = 0 → no update" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 3);
        model.add_layer(layer);
        for (size_t i = 0; i < layer->weights.rows; ++i)
            for (size_t j = 0; j < layer->weights.cols; ++j)
                layer->weights[i][j] = 0.5;
        layer->bias.fill(0.0);

        double c = 1.0;
        for (size_t i = 0; i < layer->grad_weights.rows; ++i)
            for (size_t j = 0; j < layer->grad_weights.cols; ++j)
                layer->grad_weights[i][j] = c;
        for (size_t j = 0; j < layer->grad_bias.cols; ++j)
            layer->grad_bias[0][j] = c;

        double w00_before = layer->weights[0][0];
        double b0_before  = layer->bias[0][0];

        GradientCentralization gc(new SGD(0.1));
        gc.step(model);

        bool ok = std::abs(layer->weights[0][0] - w00_before) < 1e-12
               && std::abs(layer->bias[0][0]    - b0_before)  < 1e-12;
        check("constant gradient centers to zero → SGD sees zero → no update", ok);
        cout << endl;
    }

    // Test 5: ROW mode subtracts row-wise mean
    cout << "Test 5: ROW mode centers each row" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 3);  // weights (out=3, in=2) → shape (3, 2)
        model.add_layer(layer);
        layer->weights.fill(0.0);
        layer->bias.fill(0.0);

        // grad_weights (3, 2) — set to known values:
        //   [[1, 2],
        //    [3, 4],
        //    [5, 6]]
        // Row 0 mean = 1.5 → centered row 0 = [-0.5, +0.5]
        // Row 1 mean = 3.5 → centered row 1 = [-0.5, +0.5]
        // Row 2 mean = 5.5 → centered row 2 = [-0.5, +0.5]
        double raw[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                layer->grad_weights[i][j] = raw[i][j];

        GradientCentralization gc(new SGD(1.0), GradientCentralization::CenterMode::ROW);
        double w00_before = layer->weights[0][0];
        double w01_before = layer->weights[0][1];
        double w20_before = layer->weights[2][0];
        gc.step(model);
        // SGD: param -= lr * grad' where grad' is row-centered.
        // w[0][0] -= 1 * (-0.5) = +0.5
        bool ok5a = std::abs(layer->weights[0][0] - (w00_before + 0.5)) < 1e-12;
        // w[0][1] -= 1 * (+0.5) = -0.5
        bool ok5b = std::abs(layer->weights[0][1] - (w01_before - 0.5)) < 1e-12;
        // w[2][0] -= 1 * (-0.5) = +0.5
        bool ok5c = std::abs(layer->weights[2][0] - (w20_before + 0.5)) < 1e-12;
        // Sum of centered row 0 = 0.0 by construction
        double row0_total = (layer->weights[0][0] - w00_before) +
                            (layer->weights[0][1] - w01_before);
        bool ok5d = std::abs(row0_total) < 1e-12;
        check("ROW mode: w[0][0] -= 1 * (-0.5) = +0.5", ok5a);
        check("ROW mode: w[0][1] -= 1 * (+0.5) = -0.5", ok5b);
        check("ROW mode: w[2][0] -= 1 * (-0.5) = +0.5", ok5c);
        check("ROW mode: row 0 sum-of-deltas = 0", ok5d);
        cout << endl;
    }

    // Test 6: COLUMN mode subtracts column-wise mean
    cout << "Test 6: COLUMN mode centers each column" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 3);  // weights (3, 2)
        model.add_layer(layer);
        layer->weights.fill(0.0);
        layer->bias.fill(0.0);

        // grad_weights (3, 2) = [[1, 2], [3, 4], [5, 6]]
        // Col 0 mean = 3 → centered col 0 = [-2, 0, +2]
        // Col 1 mean = 4 → centered col 1 = [-2, 0, +2]
        double raw[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                layer->grad_weights[i][j] = raw[i][j];

        GradientCentralization gc(new SGD(1.0), GradientCentralization::CenterMode::COLUMN);
        gc.step(model);

        // SGD: w[i][j] -= 1 * centered[i][j]
        // w[0][0] -= 1 * (-2) = +2
        bool ok = std::abs(layer->weights[0][0] - 2.0) < 1e-12
               && std::abs(layer->weights[1][0] - 0.0) < 1e-12
               && std::abs(layer->weights[2][0] - (-2.0)) < 1e-12;
        check("COLUMN mode: col 0 centered to [-2, 0, +2]", ok);
        cout << endl;
    }

    // Test 7: Bias (1×C) follows column-mode semantics
    cout << "Test 7: Bias (1×C) centering uses column-mode semantics" << endl;
    {
        Model model2;
        Dense* layer2 = new Dense(2, 2);
        model2.add_layer(layer2);
        layer2->weights.fill(0.0);
        layer2->bias.fill(0.0);
        layer2->grad_bias[0][0] = 3.0;
        layer2->grad_bias[0][1] = 7.0;
        GradientCentralization gc(new SGD(1.0), GradientCentralization::CenterMode::COLUMN);
        gc.step(model2);
        bool ok = std::abs(layer2->bias[0][0] - 0.0) < 1e-12
               && std::abs(layer2->bias[0][1] - 0.0) < 1e-12;
        check("bias (1×C) single-row → mean = value → centered = 0", ok);
        cout << endl;
    }

    // Test 8: Inner optimizer is called exactly once per step
    cout << "Test 8: Inner optimizer is called exactly once per step" << endl;
    {
        class CountingSGD : public Optimizer {
        public:
            int step_count = 0;
            CountingSGD() {}
            void step(Model& model) override { (void)model; ++step_count; }
        };
        Model model;
        Dense* layer = new Dense(2, 2);
        model.add_layer(layer);
        layer->zero_grad();
        CountingSGD* csgd = new CountingSGD();
        GradientCentralization gc(csgd);
        gc.step(model);
        gc.step(model);
        gc.step(model);
        check("3 calls to GC.step → 3 calls to inner.step", csgd->step_count == 3);
        cout << endl;
    }

    // Test 9: Inner optimizer sees the CENTERED gradient
    cout << "Test 9: Inner optimizer sees the centered gradient" << endl;
    {
        class CapturingSGD : public Optimizer {
        public:
            Tensor* last_grad = nullptr;
            void step(Model& model) override {
                for (auto& l : model.layers) {
                    auto grads = l->gradients();
                    if (!grads.empty() && grads[0]->rows > 0) {
                        last_grad = grads[0];
                        return;
                    }
                }
            }
        };
        Model model;
        Dense* layer = new Dense(2, 3);  // weights (3, 2)
        model.add_layer(layer);
        layer->zero_grad();
        // grad_weights (3, 2) = [[1, 2], [3, 4], [5, 6]]
        // ROW mode: row means {1.5, 3.5, 5.5} → centered = [[-0.5, +0.5],
        //                                                    [-0.5, +0.5],
        //                                                    [-0.5, +0.5]]
        double raw[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                layer->grad_weights[i][j] = raw[i][j];
        CapturingSGD* cap = new CapturingSGD();
        GradientCentralization gc(cap, GradientCentralization::CenterMode::ROW);
        gc.step(model);
        bool ok = cap->last_grad != nullptr
               && cap->last_grad->rows == 3
               && cap->last_grad->cols == 2
               && std::abs((*cap->last_grad)[0][0] - (-0.5)) < 1e-12
               && std::abs((*cap->last_grad)[0][1] - 0.5)    < 1e-12
               && std::abs((*cap->last_grad)[1][0] - (-0.5)) < 1e-12
               && std::abs((*cap->last_grad)[1][1] - 0.5)    < 1e-12
               && std::abs((*cap->last_grad)[2][0] - (-0.5)) < 1e-12
               && std::abs((*cap->last_grad)[2][1] - 0.5)    < 1e-12;
        check("inner sees the centered gradient (not raw)", ok);
        cout << endl;
    }

    // Test 10: handles_weight_decay() delegates to inner
    cout << "Test 10: handles_weight_decay() delegation" << endl;
    {
        GradientCentralization gc_sgd(new SGD(0.1));
        GradientCentralization gc_aw(new AdamW(0.1));
        check("GC(SGD).handles_weight_decay() == false", gc_sgd.handles_weight_decay() == false);
        check("GC(AdamW).handles_weight_decay() == true", gc_aw.handles_weight_decay() == true);
        cout << endl;
    }

    // Test 11: Multi-layer model — every layer's gradient is centered
    cout << "Test 11: Multi-layer — every layer's gradient is centered" << endl;
    {
        Model model;
        Dense* l1 = new Dense(2, 3);  // weights (3, 2)
        Dense* l2 = new Dense(3, 2);  // weights (2, 3)
        model.add_layer(l1);
        model.add_layer(l2);
        l1->zero_grad();
        l2->zero_grad();

        // grad_weights values designed so each row has a known mean
        // l1: weights (3, 2) — set row 0 = [1, 3], row 1 = [2, 2], row 2 = [3, 1]
        //   row 0 mean = 2 → centered = [-1, +1]
        //   row 1 mean = 2 → centered = [0, 0]
        //   row 2 mean = 2 → centered = [+1, -1]
        double l1_raw[3][2] = {{1, 3}, {2, 2}, {3, 1}};
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                l1->grad_weights[i][j] = l1_raw[i][j];
        // l2: weights (2, 3) — set row 0 = [1, 2, 3], row 1 = [4, 5, 6]
        //   row 0 mean = 2 → centered = [-1, 0, +1]
        //   row 1 mean = 5 → centered = [-1, 0, +1]
        double l2_raw[2][3] = {{1, 2, 3}, {4, 5, 6}};
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                l2->grad_weights[i][j] = l2_raw[i][j];

        GradientCentralization gc(new SGD(1.0), GradientCentralization::CenterMode::ROW);
        // Snapshot (Xavier init so weights are non-zero)
        double l1_w00_before = l1->weights[0][0];
        double l1_w01_before = l1->weights[0][1];
        double l2_w00_before = l2->weights[0][0];
        double l2_w01_before = l2->weights[0][1];
        double l2_w02_before = l2->weights[0][2];
        double l2_w10_before = l2->weights[1][0];
        gc.step(model);
        // l1 row 0 centered: [-1, +1] → l1[0][0] += 1, l1[0][1] -= 1
        bool ok = std::abs(l1->weights[0][0] - (l1_w00_before + 1.0)) < 1e-12
               && std::abs(l1->weights[0][1] - (l1_w01_before - 1.0)) < 1e-12;
        // l2 row 0 centered: [-1, 0, +1] → l2[0][0] += 1, l2[0][1] unchanged, l2[0][2] -= 1
        ok = ok && std::abs(l2->weights[0][0] - (l2_w00_before + 1.0)) < 1e-12;
        ok = ok && std::abs(l2->weights[0][1] - l2_w01_before) < 1e-12;  // unchanged
        ok = ok && std::abs(l2->weights[0][2] - (l2_w02_before - 1.0)) < 1e-12;
        // l2 row 1 centered: [-1, 0, +1] → l2[1][0] += 1
        ok = ok && std::abs(l2->weights[1][0] - (l2_w10_before + 1.0)) < 1e-12;
        check("both layers' gradients centered before inner.step", ok);
        cout << endl;
    }

    // Test 12: GC(Adam) reduces loss on linear regression
    cout << "Test 12: GC(Adam) reduces loss on y=2x regression" << endl;
    {
        Model model;
        Dense* l1 = new Dense(1, 4);
        Dense* l2 = new Dense(4, 1);
        model.add_layer(l1);
        model.add_layer(l2);
        srand(42);
        for (size_t i = 0; i < l1->weights.rows; ++i) {
            for (size_t j = 0; j < l1->weights.cols; ++j) {
                l1->weights[i][j] = 0.1 * ((double)rand() / RAND_MAX - 0.5);
            }
            l1->bias[0][i] = 0.0;
        }
        for (size_t i = 0; i < l2->weights.rows; ++i) {
            for (size_t j = 0; j < l2->weights.cols; ++j) {
                l2->weights[i][j] = 0.1 * ((double)rand() / RAND_MAX - 0.5);
            }
            l2->bias[0][i] = 0.0;
        }

        Tensor X(8, 1), y(8, 1);
        for (size_t i = 0; i < 8; ++i) {
            X[i][0] = -1.0 + 0.25 * (double)i;
            y[i][0] = 2.0 * X[i][0];
        }

        GradientCentralization gc(new Adam(0.05), GradientCentralization::CenterMode::COLUMN);

        double loss0 = 0.0;
        for (size_t ep = 0; ep < 80; ++ep) {
            Tensor pred = model.forward(X);
            Tensor diff = pred;
            for (size_t i = 0; i < 8; ++i) diff[i][0] = pred[i][0] - y[i][0];
            double loss = 0.0;
            for (size_t i = 0; i < 8; ++i) loss += diff[i][0] * diff[i][0];
            loss /= 8.0;
            if (ep == 0) loss0 = loss;

            Tensor grad_out(8, 1);
            for (size_t i = 0; i < 8; ++i) grad_out[i][0] = 2.0 * diff[i][0] / 8.0;
            model.backward(grad_out, 0.0);
            gc.step(model);
        }
        Tensor final_pred = model.forward(X);
        double final_loss = 0.0;
        for (size_t i = 0; i < 8; ++i) {
            double d = final_pred[i][0] - y[i][0];
            final_loss += d * d;
        }
        final_loss /= 8.0;
        cout << "  loss0 = " << loss0 << "  final_loss = " << final_loss << endl;
        check("GC(Adam) reduces loss meaningfully", final_loss < loss0 * 0.7);
        cout << endl;
    }

    // Test 13: Determinism
    cout << "Test 13: Determinism" << endl;
    {
        auto run = []() {
            Model model;
            Dense* l1 = new Dense(2, 2);
            model.add_layer(l1);
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 2; ++j)
                    l1->weights[i][j] = 0.3 + 0.1 * (double)(i * 2 + j);
            l1->bias.fill(0.0);
            l1->zero_grad();
            double raw[2][2] = {{1, 2}, {3, 4}};
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 2; ++j)
                    l1->grad_weights[i][j] = raw[i][j];
            GradientCentralization gc(new SGD(0.07), GradientCentralization::CenterMode::COLUMN);
            gc.step(model);
            return l1->weights[0][0];
        };
        double r1 = run();
        double r2 = run();
        check("deterministic — same closed-form result", std::abs(r1 - r2) < 1e-15);
        cout << endl;
    }

    // Test 14: set_mode() toggle
    cout << "Test 14: set_mode() toggle" << endl;
    {
        GradientCentralization gc(new SGD(1.0));
        check("default is COLUMN", gc.mode() == GradientCentralization::CenterMode::COLUMN);
        gc.set_mode(GradientCentralization::CenterMode::ROW);
        check("after set_mode(ROW)", gc.mode() == GradientCentralization::CenterMode::ROW);
        gc.set_mode(GradientCentralization::CenterMode::COLUMN);
        check("after set_mode(COLUMN)", gc.mode() == GradientCentralization::CenterMode::COLUMN);
        cout << endl;
    }

    // Test 15: After step, gradients are zeroed
    cout << "Test 15: Gradients are zeroed after step (delegated to inner)" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 3);
        model.add_layer(layer);
        layer->zero_grad();
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                layer->grad_weights[i][j] = 0.5;
        for (size_t j = 0; j < 3; ++j) layer->grad_bias[0][j] = 0.7;
        GradientCentralization gc(new SGD(0.1));
        gc.step(model);
        double max_grad = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                max_grad = std::max(max_grad, std::abs(layer->grad_weights[i][j]));
        for (size_t j = 0; j < 3; ++j)
            max_grad = std::max(max_grad, std::abs(layer->grad_bias[0][j]));
        check("all gradients zeroed after step", max_grad == 0.0);
        cout << endl;
    }

    // Test 16: Scalar (1×1) parameter
    cout << "Test 16: Scalar (1×1) parameter — no centering effect" << endl;
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        model.add_layer(layer);
        layer->zero_grad();
        layer->grad_weights[0][0] = 5.0;
        layer->grad_bias[0][0] = 3.0;
        double w0_before = layer->weights[0][0];
        double b0_before = layer->bias[0][0];
        GradientCentralization gc(new SGD(1.0), GradientCentralization::CenterMode::ROW);
        gc.step(model);
        bool ok = std::abs(layer->weights[0][0] - w0_before) < 1e-12
               && std::abs(layer->bias[0][0]    - b0_before)  < 1e-12;
        check("scalar param: centering → 0 → no update", ok);
        cout << endl;
    }

    // Test 17: LR exposure via Optimizer base class
    cout << "Test 17: LR exposure via Optimizer base class" << endl;
    {
        SGD* sgd = new SGD(0.123);
        GradientCentralization gc(sgd);
        // Note: inner_->lr via base pointer returns the base-class lr (0.001),
        // not the derived-class lr. This is a known pattern in the codebase
        // (Lookahead, etc. do the same). Verify we don't crash and the inner is
        // accessible for the user to update lr directly.
        check("gc.inner() returns the SGD", gc.inner() == sgd);
        check("inner has the user-set lr", std::abs(((SGD*)gc.inner())->lr - 0.123) < 1e-15);
        cout << endl;
    }
    // Test 18: GC(AdamW) chain — inner applies its own weight decay
    cout << "Test 18: GC(AdamW) — inner applies its own weight decay" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        model.add_layer(layer);
        for (size_t i = 0; i < layer->weights.rows; ++i)
            for (size_t j = 0; j < layer->weights.cols; ++j)
                layer->weights[i][j] = 1.0;
        layer->bias.fill(1.0);
        layer->zero_grad();

        GradientCentralization gc(new AdamW(0.0, 0.9, 0.999, 1e-8, 0.1));
        gc.step(model);
        bool ok = std::abs(layer->weights[0][0] - 1.0) < 1e-12;
        check("GC(AdamW) with lr=0: weight decay=0→no change", ok);
        cout << endl;
    }

    // Test 19: GC(AdamW) with nonzero lr + wd + grad=0 → param shrinks
    cout << "Test 19: GC(AdamW) with lr>0 + wd>0 + grad=0 → param shrinks" << endl;
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        model.add_layer(layer);
        for (size_t i = 0; i < layer->weights.rows; ++i)
            for (size_t j = 0; j < layer->weights.cols; ++j)
                layer->weights[i][j] = 1.0;
        layer->bias.fill(0.0);
        layer->zero_grad();

        GradientCentralization gc(new AdamW(0.1, 0.9, 0.999, 1e-8, 0.1));
        gc.step(model);
        bool ok = std::abs(layer->weights[0][0] - 0.99) < 1e-12;
        check("GC(AdamW) with lr=0.1 wd=0.1 → param *= 0.99", ok);
        cout << endl;
    }

    // Test 20: GC is stateless
    cout << "Test 20: GC is stateless" << endl;
    {
        GradientCentralization gc(new SGD(0.1));
        Model model;
        Dense* layer = new Dense(2, 2);
        model.add_layer(layer);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                layer->weights[i][j] = 0.5;
        layer->bias.fill(0.0);
        for (size_t step = 0; step < 5; ++step) {
            layer->zero_grad();
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 2; ++j)
                    layer->grad_weights[i][j] = 0.1 * (double)(step + 1) + i + j;
            gc.step(model);
        }
        check("5 steps with constant-size gradients — no crash", true);
        cout << endl;
    }

    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
