// Spatial Transformer Network (STN) tests — Jaderberg et al. 2015
// "Spatial Transformer Networks" (https://arxiv.org/abs/1506.02025).
//
// Tests:
//   1. Constructor + parameter count: loc_conv + loc_dense1 + loc_dense2 = 6
//   2. Forward shape: input (N, C*H*W) -> output (N, C*H*W)
//   3. Forward determinism: same input → same output (twice)
//   4. Identity transformation: when theta is initialized at identity
//      (initialization is small noise so output ≈ input but not bit-exact;
//      we test that |out - input| is small)
//   5. Forced identity: set theta to [[1,0,0],[0,1,0]] → output bit-exact equals input
//   6. Forward output is finite (no NaN/Inf)
//   7. Backward shape: input grad (N, C*H*W)
//   8. parameters() returns 6 tensors; grad shapes match
//   9. zero_grad() clears all 6 gradients
//  10. Localization network output has 6 entries (affine theta)
//  11. get_theta() returns a (2, 3) tensor with rows [0,0]=1, [1,1]=1 and small others
//  12. Numerical gradient check on input — small (1, 1, 4, 4) config
//  13. Training loop runs without crashing and loss stays finite
//  14. Bilinear sampler backward: gradient flows through with non-identity theta
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nn/layers/convolutions/spatial_transformer.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/core/model.h"
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

// Build a uniform image with one bright pixel — easy to detect rotation/translation
static Tensor make_pattern(int N, int C, int H, int W) {
    Tensor t(N, C * H * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    // Background 0.1, bright pixel 0.9 at (H/2, W/2)
                    double v = 0.1;
                    if (i == H / 2 && j == W / 2) v = 0.9;
                    t(n, c * H * W + i * W + j) = v;
                }
            }
        }
    }
    return t;
}

// Compute relative error
static double rel_err(const Tensor& a, const Tensor& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            num = std::max(num, std::fabs(a(i, j) - b(i, j)));
            den = std::max(den, std::max(std::fabs(a(i, j)), std::fabs(b(i, j))));
        }
    }
    return den > 0 ? num / den : num;
}

// Mean squared error loss
static double mse_loss(const Tensor& pred, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < pred.rows; ++i) {
        for (size_t j = 0; j < pred.cols; ++j) {
            double d = pred(i, j) - target(i, j);
            s += d * d;
        }
    }
    return s / (pred.rows * pred.cols);
}

static Tensor mse_grad(const Tensor& pred, const Tensor& target) {
    Tensor g(pred.rows, pred.cols);
    double scale = 2.0 / (pred.rows * pred.cols);
    for (size_t i = 0; i < pred.rows; ++i) {
        for (size_t j = 0; j < pred.cols; ++j) {
            g(i, j) = scale * (pred(i, j) - target(i, j));
        }
    }
    return g;
}

int main() {
    cout << "=== Spatial Transformer Network Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // Build a small input: 2 samples, 1 channel, 4x4
    const int N = 2, C = 1, H = 4, W = 4;

    // ----------------------------------------------------------------
    // Test 1: Constructor + parameter count
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        auto params = stn.parameters();
        CHECK(params.size() == 6,
              "Test 1: STN has 6 parameters (loc_conv W, loc_conv b, loc_dense1 W, loc_dense1 b, loc_dense2 W, loc_dense2 b)");
        CHECK(stn.name() == "SpatialTransformer",
              "Test 1b: name() returns 'SpatialTransformer'");
    }

    // ----------------------------------------------------------------
    // Test 8: parameters/gradients consistency
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        auto params = stn.parameters();
        auto grads = stn.gradients();
        CHECK(params.size() == grads.size(),
              "Test 8a: parameters() and gradients() return same count");
        for (size_t i = 0; i < params.size(); ++i) {
            CHECK(params[i]->rows == grads[i]->rows && params[i]->cols == grads[i]->cols,
                  "Test 8b: param/grad shape match");
        }
    }

    // ----------------------------------------------------------------
    // Test 9: zero_grad
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        // forward+backward to populate grads
        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        Tensor grad_out(N, C * H * W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        grad_out(n, c * H * W + i * W + j) = 1.0;
        stn.backward(grad_out, 0.01);
        stn.zero_grad();
        bool all_zero = true;
        for (auto* g : stn.gradients()) {
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    if (std::fabs((*g)(i, j)) > 1e-15) all_zero = false;
        }
        CHECK(all_zero, "Test 9: zero_grad clears all gradients");
    }

    // ----------------------------------------------------------------
    // Test 2: Forward shape
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        CHECK(out.rows == (size_t)N && out.cols == (size_t)(C * H * W),
              "Test 2: forward output shape (N, C*H*W)");
    }

    // ----------------------------------------------------------------
    // Test 6: Forward finite
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        CHECK(finite, "Test 6: forward output is finite");
    }

    // ----------------------------------------------------------------
    // Test 3: Forward determinism
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        Tensor out1 = stn.forward(input);
        Tensor out2 = stn.forward(input);
        CHECK(rel_err(out1, out2) < 1e-12,
              "Test 3: forward is deterministic (two calls give same output)");
    }

    // ----------------------------------------------------------------
    // Test 4: Initial output ≈ input (initialization close to identity)
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        // max abs deviation should be small (loc_fc initialized so theta ≈ identity)
        double dev = 0.0;
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                dev = std::max(dev, std::fabs(out(i, j) - input(i, j)));
        CHECK(dev < 0.3,
              "Test 4: initial output close to input (max deviation < 0.3)");
    }

    // ----------------------------------------------------------------
    // Test 5: Forced identity via set_theta — output exactly equals input
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor theta(2, 3);
        theta(0, 0) = 1.0; theta(0, 1) = 0.0; theta(0, 2) = 0.0;
        theta(1, 0) = 0.0; theta(1, 1) = 1.0; theta(1, 2) = 0.0;
        stn.set_theta(theta);

        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        CHECK(rel_err(out, input) < 1e-12,
              "Test 5: identity theta produces output == input");
    }

    // ----------------------------------------------------------------
    // Test 10: Localization output has 6 entries
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        stn.forward(input);
        Tensor theta = stn.get_theta();
        CHECK(theta.rows == 2 && theta.cols == 3,
              "Test 10: theta is (2, 3)");
    }

    // ----------------------------------------------------------------
    // Test 11: Initial theta is close to identity
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        stn.forward(input);
        Tensor theta = stn.get_theta();
        CHECK(std::fabs(theta(0, 0) - 1.0) < 0.2,
              "Test 11a: theta[0,0] ≈ 1 (initial)");
        CHECK(std::fabs(theta(1, 1) - 1.0) < 0.2,
              "Test 11b: theta[1,1] ≈ 1 (initial)");
    }

    // ----------------------------------------------------------------
    // Test 7: Backward output shape
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);
        Tensor out = stn.forward(input);
        Tensor grad_out(N, C * H * W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        grad_out(n, c * H * W + i * W + j) = 1.0;
        Tensor d_input = stn.backward(grad_out, 0.01);
        CHECK(d_input.rows == (size_t)N && d_input.cols == (size_t)(C * H * W),
              "Test 7: backward returns input grad of shape (N, C*H*W)");
        bool finite = true;
        for (size_t i = 0; i < d_input.rows; ++i)
            for (size_t j = 0; j < d_input.cols; ++j)
                if (!std::isfinite(d_input(i, j))) finite = false;
        CHECK(finite, "Test 7b: backward grad is finite");
    }

    // ----------------------------------------------------------------
    // Test 12: Numerical gradient check on input (forced identity theta)
    // ----------------------------------------------------------------
    {
        SpatialTransformer stn(1, 1, 4, 4);
        Tensor theta(2, 3);
        theta(0, 0) = 1.0; theta(0, 1) = 0.0; theta(0, 2) = 0.0;
        theta(1, 0) = 0.0; theta(1, 1) = 1.0; theta(1, 2) = 0.0;
        stn.set_theta(theta);

        Tensor input = make_pattern(1, 1, 4, 4);
        double eps = 1e-5;

        // Analytical grad (use identity-like loss: grad_out = input)
        Tensor out = stn.forward(input);
        Tensor grad_out(1, 16);
        for (int i = 0; i < 16; ++i) grad_out(0, i) = 1.0;
        Tensor ana = stn.backward(grad_out, 0.0);

        // Numerical grad — restore from a clone of the input each iteration
        Tensor input_orig = input.clone();
        Tensor num(1, 16);
        for (int idx = 0; idx < 16; ++idx) {
            int r = idx / 4, c = idx % 4;
            // Restore from orig before reading v0 (input is (1, 16))
            for (int p = 0; p < 16; ++p) input(0, p) = input_orig(0, p);
            double v0 = input(0, r * 4 + c);

            input(0, r * 4 + c) = v0 + eps;
            Tensor op = stn.forward(input);
            double lp = 0.0;
            for (int k = 0; k < 16; ++k) lp += op(0, k) * 1.0;

            input(0, r * 4 + c) = v0 - eps;
            Tensor om = stn.forward(input);
            double lm = 0.0;
            for (int k = 0; k < 16; ++k) lm += om(0, k) * 1.0;

            input(0, r * 4 + c) = v0;
            num(0, idx) = (lp - lm) / (2 * eps);
        }

        double err = rel_err(ana, num);
        cout << "  input grad rel_err (identity theta) = " << scientific << err << endl;
        CHECK(err < 1e-4, "Test 12: numerical input gradient matches analytical (identity theta)");
    }

    // ----------------------------------------------------------------
    // Test 13: Training loop runs without crash and loss stays finite
    // ----------------------------------------------------------------
    {
        // Build a tiny problem: target = input shifted right by 1 (translation)
        // The STN's locator has multiple ReLU/dense layers and gradients can be
        // large at this scale, so a tiny lr is needed for stability.
        Tensor input = make_pattern(2, 1, 4, 4);
        Tensor target(2, 16);
        for (int n = 0; n < 2; ++n) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    int src_j = j - 1;
                    double v = 0.1;
                    if (src_j >= 0 && src_j < 4 && i == 4 / 2 && src_j == 4 / 2) v = 0.9;
                    target(n, i * 4 + j) = v;
                }
            }
        }

        SpatialTransformer* stn = new SpatialTransformer(2, 1, 4, 4);

        double l0 = mse_loss(stn->forward(input), target);
        cout << "  Initial loss: " << fixed << l0 << endl;

        SGD opt(0.01);
        Model m;
        m.add_layer(stn);  // takes ownership
        for (int step = 0; step < 30; ++step) {
            Tensor pred = stn->forward(input);
            Tensor grad_out = mse_grad(pred, target);
            stn->backward(grad_out, 0.0);
            opt.step(m);
            stn->zero_grad();
        }

        double l1 = mse_loss(stn->forward(input), target);
        cout << "  Training: loss " << l0 << " -> " << l1 << endl;
        CHECK(std::isfinite(l1) && std::isfinite(l0),
              "Test 13: training loss remains finite");
    }

    // ----------------------------------------------------------------
    // Test 14: Bilinear sampler backward — finite grad with non-identity theta
    // ----------------------------------------------------------------
    {
        // Use a forced shift via theta and check that input grad is finite
        SpatialTransformer stn(1, 1, 4, 4);
        Tensor theta(2, 3);
        // small +x shift
        theta(0, 0) = 1.0; theta(0, 1) = 0.0; theta(0, 2) = 0.2;
        theta(1, 0) = 0.0; theta(1, 1) = 1.0; theta(1, 2) = 0.0;
        stn.set_theta(theta);

        Tensor input = make_pattern(1, 1, 4, 4);

        // forward
        Tensor out = stn.forward(input);
        // backward with grad = +1 everywhere
        Tensor grad_out(1, 16);
        for (int i = 0; i < 16; ++i) grad_out(0, i) = 1.0;
        Tensor d_input = stn.backward(grad_out, 0.0);

        bool finite = true;
        for (size_t i = 0; i < d_input.rows; ++i)
            for (size_t j = 0; j < d_input.cols; ++j)
                if (!std::isfinite(d_input(i, j))) finite = false;
        CHECK(finite, "Test 14: input grad finite with non-identity theta");

        // Sanity: forward changed by shift — output should differ from input
        CHECK(rel_err(out, input) > 1e-3,
              "Test 14b: output differs from input when theta shifts x");
    }

    // ----------------------------------------------------------------
    // Test 15: Numerical input gradient check with NON-identity theta
    // ----------------------------------------------------------------
    // Identity-theta grad check is vacuous (wx1/wy1 are 0/1 integers at
    // integer source coords, so the bilinear backward d_grid contribution
    // is exactly zero regardless of correctness). Use a sub-pixel shift
    // so wx1/wy1 are fractional and d_grid must be computed correctly.
    {
        SpatialTransformer stn(1, 1, 4, 4);
        Tensor theta(2, 3);
        // sub-pixel x shift: 0.3 of a pixel
        theta(0, 0) = 1.0; theta(0, 1) = 0.0; theta(0, 2) = 0.3;
        theta(1, 0) = 0.0; theta(1, 1) = 1.0; theta(1, 2) = 0.0;
        stn.set_theta(theta);

        Tensor input = make_pattern(1, 1, 4, 4);
        double eps = 1e-5;

        Tensor out = stn.forward(input);
        // grad = out (squared error proxy), so loss = sum(out^2)/2 etc.
        Tensor grad_out(1, 16);
        for (int i = 0; i < 16; ++i) grad_out(0, i) = out(0, i);

        Tensor ana = stn.backward(grad_out, 0.0);

        // Numerical gradient
        Tensor num(1, 16);
        for (int j = 0; j < 16; ++j) {
            Tensor inp_p = input; inp_p(0, j) += eps;
            Tensor inp_m = input; inp_m(0, j) -= eps;
            double fp = 0.0, fm = 0.0;
            Tensor op = stn.forward(inp_p);
            Tensor om = stn.forward(inp_m);
            for (int k = 0; k < 16; ++k) {
                fp += op(0, k) * grad_out(0, k);
                fm += om(0, k) * grad_out(0, k);
            }
            num(0, j) = (fp - fm) / (2 * eps);
        }

        double err = rel_err(ana, num);
        cout << "  input grad rel_err (sub-pixel-shift theta) = " << scientific << err << endl;
        CHECK(err < 1e-4,
              "Test 15: numerical input gradient matches analytical (sub-pixel-shift theta)");
    }

    // ----------------------------------------------------------------
    // Test 16: Numerical gradient check on loc_dense2_b (full STN chain)
    // ----------------------------------------------------------------
    // With non-fixed theta, the bilinear d_grid -> grid_backward ->
    // localization_backward chain must flow gradients correctly to the
    // localization parameters. Verify loc_dense2_b gradient via FD.
    // (Tests: bilinear_backward_grid, grid_backward, dense_backward, relu_backward)
    {
        const int N=2, C=1, H=4, W=4;
        SpatialTransformer stn(N, C, H, W);
        Tensor input = make_pattern(N, C, H, W);

        double eps = 1e-5;

        // First, populate grads with grad_out = +1 everywhere.
        Tensor out = stn.forward(input);
        Tensor grad_out(N, C * H * W);
        for (int i = 0; i < N * C * H * W; ++i) grad_out(0, i) = 1.0;
        stn.backward(grad_out, 0.0);

        // Analytical: read grad_loc_dense2_b
        Tensor ana = stn.grad_loc_dense2_b;
        Tensor num(1, 6);

        // Numerical FD on each entry of loc_dense2_b
        for (int j = 0; j < 6; ++j) {
            Tensor saved = stn.loc_dense2_b;
            stn.loc_dense2_b(0, j) = saved(0, j) + eps;
            Tensor op = stn.forward(input);
            double fp = 0.0;
            for (int k = 0; k < N * C * H * W; ++k) fp += op(0, k);

            stn.loc_dense2_b(0, j) = saved(0, j) - eps;
            Tensor om = stn.forward(input);
            double fm = 0.0;
            for (int k = 0; k < N * C * H * W; ++k) fm += om(0, k);

            stn.loc_dense2_b(0, j) = saved(0, j);  // restore
            num(0, j) = (fp - fm) / (2 * eps);
        }

        double err = rel_err(ana, num);
        cout << "  loc_dense2_b grad rel_err = " << scientific << err << endl;
        CHECK(err < 1e-4,
              "Test 16: loc_dense2_b numerical gradient matches analytical");
    }

    cout << "\n=== Results: " << passed << "/" << total << " passed ===" << endl;
    return (passed == total) ? 0 : 1;
}