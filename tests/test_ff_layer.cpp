// Forward-Forward Algorithm layer tests — Hinton 2022
// "The Forward-Forward Algorithm: Some preliminary investigations"
// (https://www.cs.toronto.edu/~hinton/FFA.pdf)
//
// Tests for FFLayer (single layer, greedy FF training):
//   1.  Forward shape: (B, in) → (B, out)
//   2.  Forward values: y = relu(W@x + b), negative pre-activations → 0
//   3.  Constructor: W and b initialized, threshold/lr stored
//   4.  train_ff: pushes positive goodness UP and negative goodness DOWN
//       after a few steps on a toy problem
//   5.  train_ff: returns (g_pos, g_neg) pair
//   6.  Numerical gradient check vs chain-rule gradient on input
//       (verifies the standard backward path works — sanity check only,
//        not how FFLayer is trained)
//   7.  numerical gradient on W via the chain rule is consistent
//   8.  parameters()/gradients()/zero_grad() work as expected
//   9.  name() returns "FFLayer"
//
// Tests for FFNetwork (multi-layer composition):
//  10.  Constructor: builds hidden_dims_.size() layers
//  11.  forward(x_with_label) shape: (B, input_dim + num_classes) → (B, last_hidden)
//  12.  make_positive: concatenates x and one-hot label
//  13.  make_negative_random: produces a different label
//  14.  train_step: returns per-layer (g_pos, g_neg)
//  15.  train_step: after several steps, positive goodness > negative goodness
//        on a learnable toy problem (verifies the FF learning signal works)
//  16.  predict: chooses the correct label on a small synthetic problem
//        after enough training
//  17.  parameters() aggregates across all layers
//  18.  name() returns "FFNetwork"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "nn/layers/utility/ff_layer.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"

using namespace std;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

// Helper: relative error
static double rel_err(double a, double b) {
    return std::fabs(a - b) / std::max(std::fabs(b), 1e-12);
}

int main() {
    cout << "=== Forward-Forward Algorithm Layer Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // =================================================================
    // FFLayer tests
    // =================================================================
    cout << "\n--- FFLayer tests ---\n";

    // -----------------------------------------------------------------
    // Test 1: Forward shape (B=4, in=3) → (4, 5)
    // -----------------------------------------------------------------
    {
        FFLayer ff(3, 5, /*threshold=*/2.0, /*lr_ff=*/0.03);
        Tensor x(4, 3);
        x(0, 0) = 0.1; x(0, 1) = -0.2; x(0, 2) = 0.3;
        x(1, 0) = 0.4; x(1, 1) =  0.5; x(1, 2) = -0.6;
        x(2, 0) = -0.7; x(2, 1) = 0.8; x(2, 2) = 0.9;
        x(3, 0) = 1.0; x(3, 1) =  -1.0; x(3, 2) = 0.5;
        Tensor y = ff.forward(x);
        CHECK(y.rows == 4 && y.cols == 5,
              "Test 1: forward shape (4x5) when input is (4x3)");
    }

    // -----------------------------------------------------------------
    // Test 2: Forward values — ReLU; if all pre-activations are negative,
    // all activations are zero.
    // -----------------------------------------------------------------
    {
        FFLayer ff(3, 4, 2.0, 0.03);
        // Force negative pre-activations by setting W to all -1, b=0,
        // and inputs to all 1. Then h = -1*1 -1*1 -1*1 = -3 → relu → 0.
        for (size_t i = 0; i < ff.parameters()[0]->data.size(); ++i) {
            ff.parameters()[0]->data[i] = -1.0;
        }
        // Set bias to 0
        ff.parameters()[1]->fill(0.0);
        Tensor x(2, 3);
        x.fill(1.0);
        Tensor y = ff.forward(x);
        bool all_zero = true;
        for (size_t i = 0; i < y.data.size(); ++i)
            if (y.data[i] != 0.0) all_zero = false;
        CHECK(all_zero, "Test 2: relu(W@x+b) = 0 when pre-activations all negative");
    }

    // -----------------------------------------------------------------
    // Test 3: Constructor stores threshold and lr_ff
    // -----------------------------------------------------------------
    {
        FFLayer ff(5, 7, /*threshold=*/3.5, /*lr_ff=*/0.07);
        CHECK(ff.threshold() == 3.5, "Test 3a: threshold stored");
        CHECK(std::fabs(ff.lr_ff() - 0.07) < 1e-12, "Test 3b: lr_ff stored");
        CHECK(ff.in_features() == 5, "Test 3c: in_features correct");
        CHECK(ff.out_features() == 7, "Test 3d: out_features correct");
        // Setters
        ff.set_threshold(1.5);
        ff.set_lr_ff(0.01);
        CHECK(ff.threshold() == 1.5, "Test 3e: set_threshold works");
        CHECK(std::fabs(ff.lr_ff() - 0.01) < 1e-12, "Test 3f: set_lr_ff works");
    }

    // -----------------------------------------------------------------
    // Test 4: train_ff pushes positive goodness UP and negative DOWN
    // Build a toy problem: positive sample has a strong pattern in a
    // specific input direction; negative has the OPPOSITE direction.
    // After many FF steps, goodness should separate them.
    // -----------------------------------------------------------------
    {
        const size_t D = 4;
        FFLayer ff(D, D, /*threshold=*/1.0, /*lr_ff=*/0.05);

        Tensor pos_x(1, D);
        Tensor neg_x(1, D);
        for (size_t k = 0; k < D; ++k) {
            pos_x(0, k) = (k == 0) ? 2.0 : 0.0;  // pattern in dim 0
            neg_x(0, k) = (k == 0) ? -2.0 : 0.0; // opposite
        }
        // Initial goodness
        Tensor yp = ff.forward(pos_x);
        Tensor yn = ff.forward(neg_x);
        double g_pos_0 = 0.0, g_neg_0 = 0.0;
        for (size_t j = 0; j < D; ++j) {
            g_pos_0 += yp(0, j) * yp(0, j);
            g_neg_0 += yn(0, j) * yn(0, j);
        }
        g_pos_0 /= D; g_neg_0 /= D;

        // Many FF steps
        for (int step = 0; step < 200; ++step) {
            ff.train_ff(pos_x, neg_x);
        }
        Tensor yp2 = ff.forward(pos_x);
        Tensor yn2 = ff.forward(neg_x);
        double g_pos_1 = 0.0, g_neg_1 = 0.0;
        for (size_t j = 0; j < D; ++j) {
            g_pos_1 += yp2(0, j) * yp2(0, j);
            g_neg_1 += yn2(0, j) * yn2(0, j);
        }
        g_pos_1 /= D; g_neg_1 /= D;

        // Verify separation: g_pos_1 should be > g_pos_0 and g_neg_1 should
        // be < g_neg_0 (modulo the threshold interaction — strong enough
        // lr_ff + steps should produce clear separation).
        bool sep_ok = (g_pos_1 > g_neg_1 + 0.5);
        CHECK(sep_ok,
              "Test 4: FF training separates pos vs neg goodness (g_pos > g_neg by 0.5)");
        cout << "    [info] g_pos_0=" << g_pos_0 << " → g_pos_1=" << g_pos_1
             << ", g_neg_0=" << g_neg_0 << " → g_neg_1=" << g_neg_1 << endl;
    }

    // -----------------------------------------------------------------
    // Test 5: train_ff returns (g_pos, g_neg) pair
    // -----------------------------------------------------------------
    {
        FFLayer ff(2, 3, 1.5, 0.03);
        Tensor pos(2, 2), neg(2, 2);
        pos.fill(1.0); neg.fill(-1.0);
        auto stats = ff.train_ff(pos, neg);
        CHECK(stats.first >= 0.0 && stats.second >= 0.0,
              "Test 5: train_ff returns non-negative goodness pair");
    }

    // -----------------------------------------------------------------
    // Test 6: Standard backward chain rule vs numerical gradient
    // (NOT how FFLayer is trained — sanity check that the chain-rule
    // gradient through y = relu(W@x+b) is correct.)
    // -----------------------------------------------------------------
    {
        FFLayer ff(3, 2, 2.0, 0.03);
        Tensor x(2, 3);
        x(0, 0) = 0.5; x(0, 1) = -0.3; x(0, 2) = 0.1;
        x(1, 0) = 0.7; x(1, 1) =  0.2; x(1, 2) = -0.4;
        // Use a known seed-like weight pattern so we can perturb
        Tensor W = *ff.parameters()[0];
        Tensor b = *ff.parameters()[1];
        for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = 0.1 * static_cast<double>(i + 1) - 0.3;
        for (size_t j = 0; j < b.data.size(); ++j) b(0, j) = 0.0;

        // Analytical: forward, set up loss = sum(grad_out * y), backward
        Tensor y = ff.forward(x);
        Tensor grad_out(2, 2);
        grad_out(0, 0) = 0.3; grad_out(0, 1) = -0.7;
        grad_out(1, 0) = 0.4; grad_out(1, 1) =  0.5;
        Tensor grad_x = ff.backward(grad_out, 0.0);

        // Numerical: perturb x[0,0] by ±eps, recompute y, compute loss
        double eps = 1e-4;
        double orig = x(0, 0);
        x(0, 0) = orig + eps;
        Tensor yp = ff.forward(x);
        double Lp = 0.0;
        for (size_t i = 0; i < yp.rows; ++i)
            for (size_t j = 0; j < yp.cols; ++j)
                Lp += grad_out(i, j) * yp(i, j);

        x(0, 0) = orig - eps;
        Tensor ym = ff.forward(x);
        double Lm = 0.0;
        for (size_t i = 0; i < ym.rows; ++i)
            for (size_t j = 0; j < ym.cols; ++j)
                Lm += grad_out(i, j) * ym(i, j);
        x(0, 0) = orig;

        double numerical = (Lp - Lm) / (2.0 * eps);
        double analytical = grad_x(0, 0);
        double re = rel_err(analytical, numerical);
        CHECK(re < 1e-4,
              "Test 6: numerical vs analytical grad_x[0,0] rel_err < 1e-4");
    }

    // -----------------------------------------------------------------
    // Test 7: numerical vs analytical W gradient via chain rule
    // -----------------------------------------------------------------
    {
        FFLayer ff(2, 2, 2.0, 0.03);
        Tensor W = *ff.parameters()[0];
        Tensor b = *ff.parameters()[1];
        // All-positive weights so units are clearly active for x = (1, 1).
        W(0, 0) = 0.5; W(0, 1) = 0.3;
        W(1, 0) = 0.4; W(1, 1) = 0.6;
        for (size_t j = 0; j < b.data.size(); ++j) b(0, j) = 0.0;

        Tensor x(1, 2);
        x(0, 0) = 1.0; x(0, 1) = 1.0;

        Tensor y = ff.forward(x);
        // Both pre-activations: 0.5+0.3 = 0.8, 0.4+0.6 = 1.0. Both clearly > 0.
        cout << "    [info] y = (" << y(0, 0) << ", " << y(0, 1) << ")" << endl;
        Tensor grad_out(1, 2);
        grad_out(0, 0) = 0.5; grad_out(0, 1) = -0.3;

        // Analytical W grad: standard chain rule: dL/dW[j,k] = sum_i grad_out[i,j] * (y>0 ? 1 : 0) * x[i,k]
        // We use the manual computation here since the layer's backward only
        // returns dL/dx.
        std::vector<std::vector<double>> Wgrad(W.rows, std::vector<double>(W.cols, 0.0));
        for (size_t i = 0; i < y.rows; ++i) {
            for (size_t j = 0; j < y.cols; ++j) {
                double mask = (y(i, j) > 0.0) ? 1.0 : 0.0;
                double g = grad_out(i, j) * mask;
                for (size_t k = 0; k < W.cols; ++k) {
                    Wgrad[j][k] += g * x(i, k);
                }
            }
        }
        double analytical = Wgrad[0][0];

        // Numerical: perturb W[0,0]
        double eps = 1e-4;
        double orig = W(0, 0);
        W(0, 0) = orig + eps;
        Tensor yp = ff.forward(x);
        double Lp = grad_out(0, 0) * yp(0, 0) + grad_out(0, 1) * yp(0, 1);
        W(0, 0) = orig - eps;
        Tensor ym = ff.forward(x);
        double Lm = grad_out(0, 0) * ym(0, 0) + grad_out(0, 1) * ym(0, 1);
        W(0, 0) = orig;
        double numerical = (Lp - Lm) / (2.0 * eps);
        double re = rel_err(analytical, numerical);
        if (re >= 1e-4) {
            cout << "    [diag] analytical=" << analytical
                 << " numerical=" << numerical
                 << " re=" << re << endl;
        }
        CHECK(re < 1e-4,
              "Test 7: numerical vs analytical grad_W[0,0] rel_err < 1e-4");
    }

    // -----------------------------------------------------------------
    // Test 8: parameters()/gradients()/zero_grad() interfaces
    // -----------------------------------------------------------------
    {
        FFLayer ff(3, 4);
        auto p = ff.parameters();
        auto g = ff.gradients();
        CHECK(p.size() == 2, "Test 8a: parameters() returns 2 (W, b)");
        CHECK(g.size() == 2, "Test 8b: gradients() returns 2 (gW, gb)");
        // Set grads to nonzero
        for (Tensor* gi : g) gi->fill(0.5);
        ff.zero_grad();
        bool all_zero = true;
        for (Tensor* gi : g) {
            for (size_t i = 0; i < gi->data.size(); ++i)
                if (gi->data[i] != 0.0) all_zero = false;
        }
        CHECK(all_zero, "Test 8c: zero_grad() zeros all grads");
    }

    // -----------------------------------------------------------------
    // Test 9: name() returns "FFLayer"
    // -----------------------------------------------------------------
    {
        FFLayer ff(2, 2);
        CHECK(ff.name() == "FFLayer", "Test 9: name() = \"FFLayer\"");
    }

    // =================================================================
    // FFNetwork tests
    // =================================================================
    cout << "\n--- FFNetwork tests ---\n";

    // -----------------------------------------------------------------
    // Test 10: Constructor builds the right number of layers
    // -----------------------------------------------------------------
    {
        FFNetwork net(/*input_dim=*/4, /*hidden_dims=*/{8, 6}, /*num_classes=*/3);
        CHECK(net.num_layers() == 2, "Test 10a: 2 hidden_dims → 2 layers");
        CHECK(net.input_dim() == 4, "Test 10b: input_dim stored");
        CHECK(net.num_classes() == 3, "Test 10c: num_classes stored");
        CHECK(net.hidden_dims().size() == 2
              && net.hidden_dims()[0] == 8
              && net.hidden_dims()[1] == 6,
              "Test 10d: hidden_dims stored");
    }

    // -----------------------------------------------------------------
    // Test 11: forward(x_with_label) shape
    // -----------------------------------------------------------------
    {
        FFNetwork net(3, {5}, 2);
        // input must have cols = input_dim + num_classes = 5
        Tensor x(2, 5);
        x(0, 0) = 0.1; x(0, 1) = 0.2; x(0, 2) = 0.3; x(0, 3) = 1.0; x(0, 4) = 0.0;
        x(1, 0) = 0.4; x(1, 1) = 0.5; x(1, 2) = 0.6; x(1, 3) = 0.0; x(1, 4) = 1.0;
        Tensor y = net.forward(x);
        CHECK(y.rows == 2 && y.cols == 5,
              "Test 11: FFNetwork forward (2x5) → (2x5) for hidden_dims={5}");
    }

    // -----------------------------------------------------------------
    // Test 12: make_positive concatenates x and one-hot label
    // -----------------------------------------------------------------
    {
        FFNetwork net(2, {3}, 3);
        Tensor x(1, 2);
        x(0, 0) = 0.5; x(0, 1) = -0.5;
        Tensor pos = net.make_positive(x, /*label=*/1);
        CHECK(pos.cols == 2 + 3, "Test 12a: positive cols = input_dim + num_classes");
        CHECK(pos.rows == 1, "Test 12b: positive rows preserved");
        CHECK(pos(0, 0) == 0.5 && pos(0, 1) == -0.5,
              "Test 12c: data dims copied through");
        // one-hot at index input_dim + label = 2 + 1 = 3
        CHECK(pos(0, 3) == 1.0, "Test 12d: one-hot label at input_dim + label");
        CHECK(pos(0, 2) == 0.0 && pos(0, 4) == 0.0,
              "Test 12e: other label dims are 0");
    }

    // -----------------------------------------------------------------
    // Test 13: make_negative_random produces a different label
    // -----------------------------------------------------------------
    {
        FFNetwork net(2, {3}, 4);
        Tensor x(1, 2);
        x.fill(0.1);
        // Sample many times — the chosen label should never equal 1
        bool never_true_label = true;
        for (int trial = 0; trial < 100; ++trial) {
            Tensor neg = net.make_negative_random(x, /*true_label=*/1);
            // Find which label index was set
            int chosen = -1;
            for (int k = 0; k < 4; ++k) {
                if (neg(0, 2 + k) == 1.0) { chosen = k; break; }
            }
            if (chosen == 1) never_true_label = false;
        }
        CHECK(never_true_label, "Test 13: negative sample never equals true label");
    }

    // -----------------------------------------------------------------
    // Test 14: train_step returns per-layer (g_pos, g_neg) stats
    // -----------------------------------------------------------------
    {
        FFNetwork net(3, {4, 3}, 2);
        Tensor x(2, 3);
        x.fill(0.1);
        auto stats = net.train_step(x, /*label=*/0);
        CHECK(stats.size() == 2,
              "Test 14: 2 layers → 2 (g_pos, g_neg) pairs from train_step");
        for (size_t i = 0; i < stats.size(); ++i) {
            CHECK(stats[i].first >= 0.0 && stats[i].second >= 0.0,
                  "Test 14: non-negative goodness per layer");
        }
    }

    // -----------------------------------------------------------------
    // Test 15: FF training pushes positive goodness up vs negative.
    // We build a synthetic problem: class 0 has pattern "first dim = 1",
    // class 1 has pattern "first dim = -1". After enough FF steps, the
    // summed goodness across layers should separate.
    // -----------------------------------------------------------------
    {
        FFNetwork net(/*input_dim=*/3, /*hidden_dims=*/{6, 4}, /*num_classes=*/2,
                       /*threshold=*/0.5, /*lr_ff=*/0.04);
        Tensor x_pos(1, 3);
        Tensor x_neg(1, 3);
        // Class 0: dim 0 = +1; Class 1: dim 0 = -1.
        x_pos(0, 0) = +1.0; x_pos(0, 1) = 0.1; x_pos(0, 2) = -0.1;
        x_neg(0, 0) = -1.0; x_neg(0, 1) = 0.1; x_neg(0, 2) = -0.1;

        // Measure initial summed goodness
        auto g_init_pos = net.predict_goodness(x_pos);
        double g0_init = g_init_pos(0, 0);
        double g1_init = g_init_pos(0, 1);

        // Train
        for (int step = 0; step < 250; ++step) {
            net.train_step(x_pos, /*label=*/0);
        }

        auto g_after_pos = net.predict_goodness(x_pos);
        double g0_after = g_after_pos(0, 0);
        double g1_after = g_after_pos(0, 1);

        cout << "    [info] class 0 goodness: " << g0_init << " → " << g0_after
             << ", class 1 goodness: " << g1_init << " → " << g1_after << endl;

        bool sep = (g0_after > g1_after + 0.05);
        CHECK(sep, "Test 15: FF training makes class-0 goodness > class-1 on x_pos");
    }

    // -----------------------------------------------------------------
    // Test 16: predict() chooses the correct label after training.
    // Two-class problem: dim 0 distinguishes.
    // -----------------------------------------------------------------
    {
        FFNetwork net(2, {6, 4}, 2, /*threshold=*/0.5, /*lr_ff=*/0.05);
        Tensor x0(1, 2);
        Tensor x1(1, 2);
        x0(0, 0) = +1.0; x0(0, 1) = 0.0;
        x1(0, 0) = -1.0; x1(0, 1) = 0.0;

        // Train both classes (label the x0 sample as class 0, x1 as class 1).
        for (int step = 0; step < 300; ++step) {
            net.train_step(x0, /*label=*/0);
            net.train_step(x1, /*label=*/1);
        }

        int pred0 = net.predict(x0);
        int pred1 = net.predict(x1);
        cout << "    [info] x0 → predict = " << pred0
             << ", x1 → predict = " << pred1 << endl;
        CHECK(pred0 == 0 && pred1 == 1,
              "Test 16: predict() correctly classifies both classes after FF training");
    }

    // -----------------------------------------------------------------
    // Test 17: parameters() aggregates across all layers
    // -----------------------------------------------------------------
    {
        FFNetwork net(3, {5, 4}, 2);
        auto p = net.parameters();
        // 2 layers × 2 params each (W, b) = 4
        CHECK(p.size() == 4,
              "Test 17: parameters() returns 4 (2 layers × (W, b))");
    }

    // -----------------------------------------------------------------
    // Test 18: name() returns "FFNetwork"
    // -----------------------------------------------------------------
    {
        FFNetwork net(2, {3}, 2);
        CHECK(net.name() == "FFNetwork", "Test 18: name() = \"FFNetwork\"");
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}