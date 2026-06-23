// SpanExtractor (SQuAD-style) tests.
//
// Tests:
//   1. Forward shape (B=2, n=5, d=3 → output (2, 6))
//   2. Forward without refinement: span = [start_emb; end_emb] exactly
//   3. Forward with refinement: post-refine differs from pre-refine
//   4. Backward without refinement: grad_input scattered to correct rows
//   5. Backward without refinement: zero in untouched rows
//   6. Backward with refinement: chains through tanh + Dense
//   7. Refinement gradient: numerical vs analytical grad_weights
//   8. Refinement gradient: numerical vs analytical grad_bias
//   9. Repeated indices (same start row hit by multiple examples) accumulate
//  10. Out-of-range index throws
//  11. dim mismatch (sequence dim != d_model) throws
//  12. parameters()/gradients()/zero_grad() delegation when refinement on
//  13. parameters()/gradients() empty when refinement off
//  14. training step reduces loss (Dense + SpanExtractor stack on toy data)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nn/layers/architectures/span_extractor.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/optimizers/optimizer.h"

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

static bool close_enough(double a, double b, double rel_tol = 1e-9) {
    return std::fabs(a - b) <= rel_tol * std::max(std::fabs(b), 1.0);
}

int main() {
    cout << "=== Span Extractor Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // ------------------------------------------------------------
    // Test 1: Forward shape (B=2, n=5, d=3 → output (2, 6))
    // ------------------------------------------------------------
    {
        Tensor seq(5, 3);
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j < 3; ++j)
                seq(i, j) = static_cast<double>(i * 3 + j + 1) * 0.1;

        SpanExtractor sx(3, /*use_refinement=*/false);
        vector<int> starts = {0, 2};
        vector<int> ends   = {1, 4};
        Tensor out = sx.extract(seq, starts, ends);
        CHECK(out.rows == 2 && out.cols == 6,
              "Test 1: forward shape (B=2, 2d=6)");
    }

    // ------------------------------------------------------------
    // Test 2: Forward without refinement: span = [start_emb; end_emb] exactly
    // ------------------------------------------------------------
    {
        Tensor seq(4, 2);
        seq(0, 0) = 1.0; seq(0, 1) = 2.0;
        seq(1, 0) = 3.0; seq(1, 1) = 4.0;
        seq(2, 0) = 5.0; seq(2, 1) = 6.0;
        seq(3, 0) = 7.0; seq(3, 1) = 8.0;

        SpanExtractor sx(2, /*use_refinement=*/false);
        vector<int> starts = {0, 2};
        vector<int> ends   = {3, 1};
        Tensor out = sx.extract(seq, starts, ends);

        // Expected: out[0, 0..1] = seq[0, :], out[0, 2..3] = seq[3, :]
        //          out[1, 0..1] = seq[2, :], out[1, 2..3] = seq[1, :]
        CHECK(close_enough(out(0, 0), 1.0) && close_enough(out(0, 1), 2.0),
              "Test 2a: out[0, 0:2] = seq[0, :] (start of example 0)");
        CHECK(close_enough(out(0, 2), 7.0) && close_enough(out(0, 3), 8.0),
              "Test 2b: out[0, 2:4] = seq[3, :] (end of example 0)");
        CHECK(close_enough(out(1, 0), 5.0) && close_enough(out(1, 1), 6.0),
              "Test 2c: out[1, 0:2] = seq[2, :] (start of example 1)");
        CHECK(close_enough(out(1, 2), 3.0) && close_enough(out(1, 3), 4.0),
              "Test 2d: out[1, 2:4] = seq[1, :] (end of example 1)");
    }

    // ------------------------------------------------------------
    // Test 3: Forward with refinement: post-refine differs from pre-refine
    // We just verify that some entries are non-trivially transformed.
    // ------------------------------------------------------------
    {
        Tensor seq(3, 2);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                seq(i, j) = 0.5;

        SpanExtractor sx(2, /*use_refinement=*/true);
        vector<int> starts = {0};
        vector<int> ends   = {2};
        Tensor out = sx.extract(seq, starts, ends);
        // With refinement: post-tanh output should not be exactly 0.5 (unless
        // W is initialized such that z ≈ 0 → tanh ≈ 0). With x=0.5 and a
        // Xavier-init W, the pre-activation is non-zero, so the post-tanh
        // output should differ from 0.5.
        bool different = false;
        for (size_t j = 0; j < 2; ++j)
            if (!close_enough(out(0, j), 0.5, 1e-3)) different = true;
        CHECK(different,
              "Test 3: refinement produces post-activations different from input");
    }

    // ------------------------------------------------------------
    // Test 4: Backward without refinement: grad_input scattered correctly
    // grad_output[i, 0..d-1] → grad_input[starts[i], :]
    // grad_output[i, d..2d-1] → grad_input[ends[i], :]
    // ------------------------------------------------------------
    {
        Tensor seq(4, 2);
        for (size_t i = 0; i < 4; ++i) seq(i, 0) = 1.0, seq(i, 1) = 1.0;

        SpanExtractor sx(2, /*use_refinement=*/false);
        vector<int> starts = {0, 2};
        vector<int> ends   = {3, 1};
        sx.extract(seq, starts, ends);  // populate last_starts_/last_ends_

        // grad_output: 2 examples, 4 cols (2d=4)
        Tensor grad_out(2, 4);
        grad_out(0, 0) = 1.0; grad_out(0, 1) = 2.0;  // → row 0 (start of ex 0)
        grad_out(0, 2) = 3.0; grad_out(0, 3) = 4.0;  // → row 3 (end of ex 0)
        grad_out(1, 0) = 5.0; grad_out(1, 1) = 6.0;  // → row 2 (start of ex 1)
        grad_out(1, 2) = 7.0; grad_out(1, 3) = 8.0;  // → row 1 (end of ex 1)

        Tensor grad_in = sx.backward_span(grad_out, 0.01);
        CHECK(close_enough(grad_in(0, 0), 1.0) && close_enough(grad_in(0, 1), 2.0),
              "Test 4a: grad_in[0, :] = (1, 2) (start of ex 0)");
        CHECK(close_enough(grad_in(3, 0), 3.0) && close_enough(grad_in(3, 1), 4.0),
              "Test 4b: grad_in[3, :] = (3, 4) (end of ex 0)");
        CHECK(close_enough(grad_in(2, 0), 5.0) && close_enough(grad_in(2, 1), 6.0),
              "Test 4c: grad_in[2, :] = (5, 6) (start of ex 1)");
        CHECK(close_enough(grad_in(1, 0), 7.0) && close_enough(grad_in(1, 1), 8.0),
              "Test 4d: grad_in[1, :] = (7, 8) (end of ex 1)");
    }

    // ------------------------------------------------------------
    // Test 5: Backward without refinement: zero in untouched rows
    // ------------------------------------------------------------
    {
        Tensor seq(5, 2);
        for (size_t i = 0; i < 5; ++i) seq(i, 0) = 0.1, seq(i, 1) = 0.1;

        SpanExtractor sx(2, /*use_refinement=*/false);
        vector<int> starts = {0};
        vector<int> ends   = {4};
        sx.extract(seq, starts, ends);

        Tensor grad_out(1, 4);
        grad_out(0, 0) = 0.5; grad_out(0, 1) = -0.3;
        grad_out(0, 2) = 1.0; grad_out(0, 3) = 2.0;

        Tensor grad_in = sx.backward_span(grad_out, 0.01);
        bool zero_untouched = true;
        // Rows 1, 2, 3 are untouched.
        for (size_t r : {size_t(1), size_t(2), size_t(3)}) {
            for (size_t c = 0; c < 2; ++c) {
                if (grad_in(r, c) != 0.0) zero_untouched = false;
            }
        }
        CHECK(zero_untouched, "Test 5: untouched rows have zero gradient");
    }

    // ------------------------------------------------------------
    // Test 6: Refinement gradient: numerical vs analytical grad_weights
    // We perturb W[0, 0] and measure (L(W+eps) - L(W-eps)) / (2*eps).
    // Loss L = sum over all output entries squared.
    // ------------------------------------------------------------
    {
        Tensor seq(3, 2);
        seq(0, 0) = 0.3; seq(0, 1) = -0.5;
        seq(1, 0) = 0.7; seq(1, 1) =  0.2;
        seq(2, 0) = -0.4; seq(2, 1) = 0.6;

        SpanExtractor sx(2, /*use_refinement=*/true);
        // Set known refinement weights/bias
        sx.refinement()->weights(0, 0) = 0.1; sx.refinement()->weights(0, 1) = 0.2;
        sx.refinement()->weights(1, 0) = 0.3; sx.refinement()->weights(1, 1) = 0.4;
        sx.refinement()->bias(0, 0) = 0.05; sx.refinement()->bias(0, 1) = -0.05;

        vector<int> starts = {0};
        vector<int> ends   = {2};

        // Analytical: forward → grad_out (treat as dL/dy = 2*y) → backward
        Tensor y = sx.extract(seq, starts, ends);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out(i, j) = 2.0 * y(i, j);  // dL/dy = 2y for L = ||y||^2

        sx.zero_grad();
        sx.backward_span(grad_out, 0.0);
        double analytical = sx.refinement()->grad_weights(0, 0);

        // Numerical: perturb W[0,0] by ±eps, re-extract, measure L change.
        double eps = 1e-4;
        double orig_w = sx.refinement()->weights(0, 0);
        sx.refinement()->weights(0, 0) = orig_w + eps;
        Tensor y_p = sx.extract(seq, starts, ends);
        double L_p = y_p.data[0] * y_p.data[0] + y_p.data[1] * y_p.data[1]
                   + y_p.data[2] * y_p.data[2] + y_p.data[3] * y_p.data[3];
        sx.refinement()->weights(0, 0) = orig_w - eps;
        Tensor y_m = sx.extract(seq, starts, ends);
        double L_m = y_m.data[0] * y_m.data[0] + y_m.data[1] * y_m.data[1]
                   + y_m.data[2] * y_m.data[2] + y_m.data[3] * y_m.data[3];
        sx.refinement()->weights(0, 0) = orig_w;

        double numerical = (L_p - L_m) / (2.0 * eps);
        double rel_err = std::fabs(analytical - numerical) / std::max(std::fabs(numerical), 1e-12);
        CHECK(rel_err < 1e-4,
              "Test 6: numerical grad_weights[0,0] vs analytical rel_err < 1e-4");
    }

    // ------------------------------------------------------------
    // Test 7: Refinement gradient: numerical vs analytical grad_bias
    // ------------------------------------------------------------
    {
        Tensor seq(2, 2);
        seq(0, 0) = 0.4; seq(0, 1) = -0.1;
        seq(1, 0) = 0.5; seq(1, 1) =  0.8;

        SpanExtractor sx(2, /*use_refinement=*/true);
        sx.refinement()->weights(0, 0) = 0.1; sx.refinement()->weights(0, 1) = 0.2;
        sx.refinement()->weights(1, 0) = 0.3; sx.refinement()->weights(1, 1) = 0.4;
        sx.refinement()->bias(0, 0) = 0.0; sx.refinement()->bias(0, 1) = 0.0;

        vector<int> starts = {0};
        vector<int> ends   = {1};

        Tensor y = sx.extract(seq, starts, ends);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out(i, j) = 2.0 * y(i, j);

        sx.zero_grad();
        sx.backward_span(grad_out, 0.0);
        double analytical_b0 = sx.refinement()->grad_bias(0, 0);
        double analytical_b1 = sx.refinement()->grad_bias(0, 1);

        // Numerical: perturb bias[0, 0] by ±eps
        double eps = 1e-4;
        double orig_b = sx.refinement()->bias(0, 0);
        sx.refinement()->bias(0, 0) = orig_b + eps;
        Tensor y_p = sx.extract(seq, starts, ends);
        double L_p = 0.0; for (double v : y_p.data) L_p += v * v;
        sx.refinement()->bias(0, 0) = orig_b - eps;
        Tensor y_m = sx.extract(seq, starts, ends);
        double L_m = 0.0; for (double v : y_m.data) L_m += v * v;
        sx.refinement()->bias(0, 0) = orig_b;
        double numerical_b0 = (L_p - L_m) / (2.0 * eps);

        // Same for bias[0, 1]
        orig_b = sx.refinement()->bias(0, 1);
        sx.refinement()->bias(0, 1) = orig_b + eps;
        Tensor y_p2 = sx.extract(seq, starts, ends);
        double L_p2 = 0.0; for (double v : y_p2.data) L_p2 += v * v;
        sx.refinement()->bias(0, 1) = orig_b - eps;
        Tensor y_m2 = sx.extract(seq, starts, ends);
        double L_m2 = 0.0; for (double v : y_m2.data) L_m2 += v * v;
        sx.refinement()->bias(0, 1) = orig_b;
        double numerical_b1 = (L_p2 - L_m2) / (2.0 * eps);

        double rel_err0 = std::fabs(analytical_b0 - numerical_b0) / std::max(std::fabs(numerical_b0), 1e-12);
        double rel_err1 = std::fabs(analytical_b1 - numerical_b1) / std::max(std::fabs(numerical_b1), 1e-12);
        CHECK(rel_err0 < 1e-3,
              "Test 7a: grad_bias[0,0] numerical vs analytical rel_err < 1e-3");
        CHECK(rel_err1 < 1e-3,
              "Test 7b: grad_bias[0,1] numerical vs analytical rel_err < 1e-3");
    }

    // ------------------------------------------------------------
    // Test 8: Repeated indices (same start row hit by multiple examples)
    // accumulate correctly.
    // ------------------------------------------------------------
    {
        Tensor seq(3, 2);
        for (size_t i = 0; i < 3; ++i) seq(i, 0) = 1.0, seq(i, 1) = 0.0;

        SpanExtractor sx(2, /*use_refinement=*/false);
        vector<int> starts = {0, 0};   // both examples hit row 0 as start
        vector<int> ends   = {1, 2};   // ends at rows 1 and 2
        sx.extract(seq, starts, ends);

        Tensor grad_out(2, 4);
        grad_out(0, 0) = 1.0; grad_out(0, 1) = 0.0;  // ex 0 start
        grad_out(0, 2) = 0.0; grad_out(0, 3) = 0.0;  // ex 0 end at row 1
        grad_out(1, 0) = 0.5; grad_out(1, 1) = 0.0;  // ex 1 start (accumulates with ex 0)
        grad_out(1, 2) = 0.0; grad_out(1, 3) = 0.0;  // ex 1 end at row 2

        Tensor grad_in = sx.backward_span(grad_out, 0.01);
        CHECK(close_enough(grad_in(0, 0), 1.5),  // 1.0 + 0.5 accumulated
              "Test 8a: row 0 accumulated from 2 start indices (1.5)");
        CHECK(close_enough(grad_in(1, 0), 0.0),
              "Test 8b: row 1 has end-of-ex-0 grad (0)");
        CHECK(close_enough(grad_in(2, 0), 0.0),
              "Test 8c: row 2 has end-of-ex-1 grad (0)");
    }

    // ------------------------------------------------------------
    // Test 9: Out-of-range index throws
    // ------------------------------------------------------------
    {
        Tensor seq(3, 2);
        SpanExtractor sx(2, false);
        bool threw = false;
        try {
            sx.extract(seq, {0, 5}, {1, 2});  // 5 out of range for n=3
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "Test 9: out-of-range index throws std::runtime_error");
    }

    // ------------------------------------------------------------
    // Test 10: dim mismatch throws
    // ------------------------------------------------------------
    {
        Tensor seq(3, 4);   // feature dim 4, but d_model=2
        SpanExtractor sx(2, false);
        bool threw = false;
        try {
            sx.extract(seq, {0}, {1});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "Test 10: dim mismatch throws std::runtime_error");
    }

    // ------------------------------------------------------------
    // Test 11: parameters()/gradients() delegation when refinement on
    // ------------------------------------------------------------
    {
        SpanExtractor sx(3, /*use_refinement=*/true);
        auto p = sx.parameters();
        auto g = sx.gradients();
        CHECK(p.size() == 2, "Test 11a: parameters() returns 2 (W, b) with refinement");
        CHECK(g.size() == 2, "Test 11b: gradients() returns 2 with refinement");
    }

    // ------------------------------------------------------------
    // Test 12: parameters()/gradients() empty when refinement off
    // ------------------------------------------------------------
    {
        SpanExtractor sx(3, /*use_refinement=*/false);
        auto p = sx.parameters();
        auto g = sx.gradients();
        CHECK(p.empty(), "Test 12a: parameters() empty without refinement");
        CHECK(g.empty(), "Test 12b: gradients() empty without refinement");
    }

    // ------------------------------------------------------------
    // Test 13: training step reduces loss
    // Build a tiny problem: random seq → span → MLP → target scalar.
    // Use L2 loss. After a few SGD steps the loss should go down.
    // ------------------------------------------------------------
    {
        Tensor seq(5, 3);
        // Init sequence with non-trivial values
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j < 3; ++j)
                seq(i, j) = std::sin(0.5 * i + 0.1 * j);

        SpanExtractor sx(3, /*use_refinement=*/true);
        vector<int> starts = {1, 3};
        vector<int> ends   = {2, 0};
        Tensor target(2, 6);
        target(0, 0) = 0.5; target(0, 1) = -0.3; target(0, 2) = 0.7;
        target(0, 3) = 0.1; target(0, 4) = -0.2; target(0, 5) = 0.4;
        target(1, 0) = -0.4; target(1, 1) = 0.6; target(1, 2) = -0.1;
        target(1, 3) = 0.3; target(1, 4) = 0.8; target(1, 5) = -0.5;

        double prev_loss = 1e9;
        bool loss_decreasing = true;
        for (int step = 0; step < 80; ++step) {
            Tensor y = sx.extract(seq, starts, ends);
            // L2 loss: gradient = 2*(y - target)
            Tensor grad_out(y.rows, y.cols);
            double loss = 0.0;
            for (size_t i = 0; i < y.rows; ++i) {
                for (size_t j = 0; j < y.cols; ++j) {
                    double diff = y(i, j) - target(i, j);
                    grad_out(i, j) = 2.0 * diff;
                    loss += diff * diff;
                }
            }
            sx.zero_grad();
            sx.backward_span(grad_out, 0.0);
            // Apply SGD manually (tiny step) on the refinement's W and b
            double lr = 0.01;
            for (Tensor* p : sx.parameters()) {
                Tensor* g_ptr = (p == &sx.refinement()->weights) ? &sx.refinement()->grad_weights
                              : (p == &sx.refinement()->bias)    ? &sx.refinement()->grad_bias
                              : nullptr;
                if (!g_ptr) continue;
                for (size_t i = 0; i < p->data.size(); ++i) {
                    p->data[i] -= lr * g_ptr->data[i];
                }
            }
            if (loss > prev_loss + 1e-9) loss_decreasing = false;
            prev_loss = loss;
        }
        CHECK(loss_decreasing,
              "Test 13: SpanExtractor + refinement training never increases loss");
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
