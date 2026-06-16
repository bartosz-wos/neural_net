// PixelCNN / Gated PixelCNN Tests
//   van den Oord et al. 2016, "Pixel Recurrent Neural Networks"
//   van den Oord et al. 2016, "Conditional Image Generation with PixelCNN Decoders"
//
// Tests:
//   1. MaskedConv2d (type A) forward shape
//   2. MaskedConv2d (type A) output is finite
//   3. MaskedConv2d (type A) input gradient check
//   4. MaskedConv2d (type A) weights gradient check
//   5. MaskedConv2d (type A) bias gradient check
//   6. MaskedConv2d (type B) forward shape
//   7. MaskedConv2d (type B) input gradient check
//   8. MaskedConv2d (type B) center pixel is allowed (mask type B includes (kH/2,kW/2))
//   9. MaskedConv2d (type A) center pixel is blocked (autoregressive property)
//  10. MaskedConv2d training step reduces loss
//  11. GatedPixelCNNBlock forward shape
//  12. GatedPixelCNNBlock output is finite
//  13. GatedPixelCNNBlock input gradient check
//  14. GatedPixelCNNBlock training step reduces loss
//  15. PixelCNN (full model) forward shape
//  16. PixelCNN output is finite
//  17. PixelCNN input gradient check
//  18. PixelCNN training step reduces loss
//  19. PixelCNN with conditioning vector (gated) — forward shape
//  20. PixelCNN parameters/gradients shape consistency

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include "nn/layers/generative/pixelcnn.h"

using namespace std;

static double relative_error(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
}

static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// Helper: find parameter by shape signature
struct ParamMatch {
    Tensor* p = nullptr;
    Tensor* g = nullptr;
    int seen = 0;
};
static ParamMatch find_param(vector<Tensor*>& params, vector<Tensor*>& grads,
                             size_t r, size_t c, int occurrence = 0) {
    ParamMatch pm;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == r && params[i]->cols == c) {
            if (pm.seen == occurrence) {
                pm.p = params[i];
                pm.g = grads[i];
                return pm;
            }
            pm.seen++;
        }
    }
    return pm;
}

// Make a deterministic random image (1 batch, C, H, W) flattened to (1, C*H*W)
static Tensor make_image(int C, int H, int W, double seed = 0.0) {
    Tensor img(1, C * H * W);
    int idx = 0;
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                img(0, idx) = 0.1 * sin(seed + 0.3 * c + 0.5 * i) + 0.05 * j;
                ++idx;
            }
    return img;
}

int main() {
    cout << "=== PixelCNN / Gated PixelCNN Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Common test config — keep small for tractable gradient checks
    const int C_in = 1, C_out = 2;
    const int H = 4, W = 4;
    const int kH = 3, kW = 3;

    // ------------------------------------------------------------
    // Test 1: MaskedConv2d (type A) forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: MaskedConv2d type A forward shape ---\n";
    {
        ++total;
        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor input = make_image(C_in, H, W, 1.0);
        Tensor output = conv.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == 1 && output.cols == C_out * H * W) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 1x" << (C_out * H * W) << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: MaskedConv2d (type A) output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: MaskedConv2d type A output is finite ---\n";
    {
        ++total;
        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor input = make_image(C_in, H, W, 2.0);
        Tensor output = conv.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!std::isfinite(output.data[i])) finite = false;
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: MaskedConv2d (type A) input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: MaskedConv2d type A input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input = make_image(C_in, H, W, 1.5);
        Tensor target(1, C_out * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.05 * (i % 7) - 0.02 * (i / 3);

        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor out = conv.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        conv.zero_grad();
        Tensor grad_x = conv.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        // check a few input elements (sample evenly)
        for (int c = 0; c < C_in && n_checked < 4; ++c) {
            for (int i = 0; i < H && n_checked < 4; ++i) {
                for (int j = 0; j < W && n_checked < 4; ++j) {
                    int idx = c * H * W + i * W + j;
                    double orig = input(0, idx);
                    input(0, idx) = orig + eps;
                    Tensor out_p = conv.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    input(0, idx) = orig - eps;
                    Tensor out_m = conv.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    input(0, idx) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = grad_x(0, idx);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.05) {
                        cout << "  x[0," << idx << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
        }
        cout << "Max err (over " << n_checked << " elements): " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: MaskedConv2d (type A) weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: MaskedConv2d type A weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input = make_image(C_in, H, W, 2.5);
        Tensor target(1, C_out * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.07 * (i % 5) + 0.01 * (i / 4);

        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor out = conv.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        conv.zero_grad();
        conv.backward(grad_loss, 0.0);

        // weights: (out_channels, in_channels * kH * kW) = (C_out, C_in*kH*kW)
        size_t w_rows = (size_t)C_out, w_cols = (size_t)(C_in * kH * kW);
        double max_err = 0.0;
        int n_checked = 0;
        for (size_t i = 0; i < w_rows && n_checked < 6; ++i) {
            for (size_t j = 0; j < w_cols && n_checked < 6; ++j) {
                double orig = conv.weights(i, j);
                conv.weights(i, j) = orig + eps;
                Tensor out_p = conv.forward(input);
                double Lp = l2_loss_value(out_p, target);
                conv.weights(i, j) = orig - eps;
                Tensor out_m = conv.forward(input);
                double Lm = l2_loss_value(out_m, target);
                conv.weights(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = conv.grad_weights(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
            }
        }
        cout << "Max err (over " << n_checked << " elements): " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] weights gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] weights gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: MaskedConv2d (type A) bias gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: MaskedConv2d type A bias gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input = make_image(C_in, H, W, 3.0);
        Tensor target(1, C_out * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.04 * (i % 3) - 0.01;

        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor out = conv.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        conv.zero_grad();
        conv.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (int o = 0; o < C_out; ++o) {
            double orig = conv.bias(o, 0);
            conv.bias(o, 0) = orig + eps;
            Tensor out_p = conv.forward(input);
            double Lp = l2_loss_value(out_p, target);
            conv.bias(o, 0) = orig - eps;
            Tensor out_m = conv.forward(input);
            double Lm = l2_loss_value(out_m, target);
            conv.bias(o, 0) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = conv.grad_bias(o, 0);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] bias gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] bias gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: MaskedConv2d (type B) forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 6: MaskedConv2d type B forward shape ---\n";
    {
        ++total;
        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'B');
        Tensor input = make_image(C_in, H, W, 4.0);
        Tensor output = conv.forward(input);
        if (output.rows == 1 && output.cols == (size_t)(C_out * H * W)) {
            cout << "[PASS] type B forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 1x" << (C_out * H * W)
                 << ", got " << output.rows << "x" << output.cols << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: MaskedConv2d (type B) input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: MaskedConv2d type B input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input = make_image(C_in, H, W, 4.5);
        Tensor target(1, C_out * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.06 * (i % 4) - 0.03;

        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'B');
        Tensor out = conv.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        conv.zero_grad();
        Tensor grad_x = conv.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        for (int c = 0; c < C_in && n_checked < 4; ++c) {
            for (int i = 0; i < H && n_checked < 4; ++i) {
                for (int j = 0; j < W && n_checked < 4; ++j) {
                    int idx = c * H * W + i * W + j;
                    double orig = input(0, idx);
                    input(0, idx) = orig + eps;
                    Tensor out_p = conv.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    input(0, idx) = orig - eps;
                    Tensor out_m = conv.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    input(0, idx) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = grad_x(0, idx);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] type B input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] type B input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: MaskedConv2d (type B) center pixel is allowed
    // ------------------------------------------------------------
    cout << "\n--- Test 8: MaskedConv2d type B center pixel is allowed ---\n";
    {
        ++total;
        // Sanity: with type B, the weights at center pixel position are non-zero
        // (initialized normally). We verify by checking that conv.weights has
        // non-zero entries at the center column (kH/2, kW/2) of the kernel.
        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'B');
        // center col index: c * kH*kW + (kH/2)*kW + (kW/2) = 0*9 + 1*3 + 1 = 4
        size_t center_col = (kH / 2) * kW + (kW / 2);
        bool any_nonzero = false;
        for (int o = 0; o < C_out; ++o) {
            if (std::abs(conv.weights(o, center_col)) > 1e-9) {
                any_nonzero = true;
                break;
            }
        }
        if (any_nonzero) {
            cout << "[PASS] type B allows center pixel (weights at center nonzero)\n";
            ++passed;
        } else {
            cout << "[FAIL] type B center pixel weights are all zero\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: MaskedConv2d (type A) center pixel is blocked
    // ------------------------------------------------------------
    cout << "\n--- Test 9: MaskedConv2d type A center pixel is blocked ---\n";
    {
        ++total;
        // Sanity: with type A, the weights at center pixel position are exactly zero.
        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        size_t center_col = (kH / 2) * kW + (kW / 2);
        bool all_zero = true;
        for (int o = 0; o < C_out; ++o) {
            if (std::abs(conv.weights(o, center_col)) > 1e-15) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            cout << "[PASS] type A blocks center pixel (weights at center are 0)\n";
            ++passed;
        } else {
            cout << "[FAIL] type A center pixel weights are non-zero\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: MaskedConv2d training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: MaskedConv2d training step reduces loss ---\n";
    {
        ++total;
        Tensor input = make_image(C_in, H, W, 5.0);
        Tensor target(1, C_out * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.05 * sin(0.7 * i) - 0.02;

        MaskedConv2d conv(C_in, C_out, kH, kW, H, W, 'A');
        Tensor out0 = conv.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
        for (int step = 0; step < 60; ++step) {
            conv.zero_grad();
            Tensor out = conv.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            conv.backward(grad_loss, 0.0);
            conv.update_weights(lr);
        }
        Tensor out1 = conv.forward(input);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not decrease loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: GatedPixelCNNBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 11: GatedPixelCNNBlock forward shape ---\n";
    {
        ++total;
        int hidden = 3;
        GatedPixelCNNBlock block(C_in, hidden, kH, kW, H, W, 'B');
        Tensor input = make_image(C_in, H, W, 6.0);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        // The block is residual: output dim equals input dim (= in_ch * H * W).
        if (output.rows == 1 && output.cols == (size_t)(C_in * H * W)) {
            cout << "[PASS] GatedPixelCNNBlock forward shape correct (residual)\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 1x" << (C_in * H * W)
                 << ", got " << output.rows << "x" << output.cols << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: GatedPixelCNNBlock output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 12: GatedPixelCNNBlock output is finite ---\n";
    {
        ++total;
        int hidden = 3;
        GatedPixelCNNBlock block(C_in, hidden, kH, kW, H, W, 'B');
        Tensor input = make_image(C_in, H, W, 6.5);
        Tensor output = block.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!std::isfinite(output.data[i])) finite = false;
        if (finite) {
            cout << "[PASS] GatedPixelCNNBlock output is finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: GatedPixelCNNBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 13: GatedPixelCNNBlock input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        int hidden = 3;
        Tensor input = make_image(C_in, H, W, 7.0);
        // Block output dim = in_ch * H * W (residual)
        Tensor target(1, C_in * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.04 * (i % 5) - 0.01;

        GatedPixelCNNBlock block(C_in, hidden, kH, kW, H, W, 'B');
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        for (int i = 0; i < H && n_checked < 4; ++i) {
            for (int j = 0; j < W && n_checked < 4; ++j) {
                int idx = i * W + j;
                double orig = input(0, idx);
                input(0, idx) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(0, idx) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(0, idx) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(0, idx);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] GatedPixelCNNBlock input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] GatedPixelCNNBlock input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: GatedPixelCNNBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 14: GatedPixelCNNBlock training step reduces loss ---\n";
    {
        ++total;
        int hidden = 3;
        Tensor input = make_image(C_in, H, W, 8.0);
        Tensor target(1, C_in * H * W);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.03 * sin(0.5 * i);

        GatedPixelCNNBlock block(C_in, hidden, kH, kW, H, W, 'B');
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 80; ++step) {
            block.zero_grad();
            Tensor out = block.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            block.backward(grad_loss, 0.0);
            block.update_weights(lr);
        }
        Tensor out1 = block.forward(input);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] GatedPixelCNNBlock training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] GatedPixelCNNBlock training did not decrease loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: PixelCNN (full model) forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 15: PixelCNN full model forward shape ---\n";
    {
        ++total;
        int hidden = 2;
        int n_blocks = 1;
        int n_vals = 4;  // pixel can take 4 discrete values
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A');
        Tensor input = make_image(C_in, H, W, 9.0);
        Tensor output = model.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        // Output: (1, C_in * n_vals * H * W) — predicts per-pixel, per-channel
        // softmax over n_vals classes
        size_t expected = (size_t)(C_in * n_vals * H * W);
        if (output.rows == 1 && output.cols == expected) {
            cout << "[PASS] PixelCNN forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 1x" << expected
                 << ", got " << output.rows << "x" << output.cols << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: PixelCNN output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 16: PixelCNN output is finite ---\n";
    {
        ++total;
        int hidden = 2;
        int n_blocks = 1;
        int n_vals = 4;
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A');
        Tensor input = make_image(C_in, H, W, 9.5);
        Tensor output = model.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!std::isfinite(output.data[i])) finite = false;
        if (finite) {
            cout << "[PASS] PixelCNN output is finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 17: PixelCNN input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 17: PixelCNN input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        int hidden = 2;
        int n_blocks = 1;
        int n_vals = 4;
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A');
        Tensor input = make_image(C_in, H, W, 10.0);
        size_t expected_cols = (size_t)(C_in * n_vals * H * W);
        Tensor target(1, expected_cols);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.02 * (i % 3) - 0.01;

        Tensor out = model.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        for (int i = 0; i < H && n_checked < 4; ++i) {
            for (int j = 0; j < W && n_checked < 4; ++j) {
                int idx = i * W + j;
                double orig = input(0, idx);
                input(0, idx) = orig + eps;
                Tensor out_p = model.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(0, idx) = orig - eps;
                Tensor out_m = model.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(0, idx) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(0, idx);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] PixelCNN input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] PixelCNN input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 18: PixelCNN training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 18: PixelCNN training step reduces loss ---\n";
    {
        ++total;
        int hidden = 2;
        int n_blocks = 1;
        int n_vals = 4;
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A');
        Tensor input = make_image(C_in, H, W, 11.0);
        size_t expected_cols = (size_t)(C_in * n_vals * H * W);
        Tensor target(1, expected_cols);
        for (size_t i = 0; i < target.data.size(); ++i)
            target.data[i] = 0.02 * cos(0.3 * i);

        Tensor out0 = model.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 80; ++step) {
            model.zero_grad();
            Tensor out = model.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            model.backward(grad_loss, 0.0);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward(input);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] PixelCNN training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] PixelCNN training did not decrease loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 19: PixelCNN with conditioning (gated) — forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 19: PixelCNN with conditioning (gated) ---\n";
    {
        ++total;
        int hidden = 2;
        int n_blocks = 1;
        int n_vals = 4;
        int cond_dim = 3;
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A', cond_dim);
        Tensor input = make_image(C_in, H, W, 12.0);
        Tensor cond(1, cond_dim);
        for (int i = 0; i < cond_dim; ++i) cond(0, i) = 0.1 * (i + 1);
        Tensor output = model.forward_with_cond(input, cond);
        size_t expected = (size_t)(C_in * n_vals * H * W);
        if (output.rows == 1 && output.cols == expected) {
            cout << "[PASS] PixelCNN with conditioning forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 1x" << expected
                 << ", got " << output.rows << "x" << output.cols << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 20: PixelCNN parameters/gradients shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 20: PixelCNN parameters/gradients shape consistency ---\n";
    {
        ++total;
        int hidden = 2;
        int n_blocks = 2;
        int n_vals = 4;
        PixelCNN model(C_in, hidden, kH, kW, H, W, n_blocks, n_vals, 'A');
        auto params = model.parameters();
        auto grads  = model.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
        } else {
            bool ok = true;
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows ||
                    params[i]->cols != grads[i]->cols) {
                    ok = false;
                    cout << "  Mismatch at param " << i
                         << ": " << params[i]->rows << "x" << params[i]->cols
                         << " vs grad " << grads[i]->rows << "x" << grads[i]->cols
                         << "\n";
                    break;
                }
            }
            // Should have:
            //   first_mask_conv: 2 params (weights, bias)
            //   per block: conv_v_ (2) + bias_v_ (1) + conv_u_ (2) = 5
            //   final classifier: 2
            //   n_blocks=2 => 2 + 2*5 + 2 = 14
            size_t expected = 2 + n_blocks * 5 + 2;
            if (ok && params.size() == expected) {
                cout << "[PASS] all " << params.size()
                     << " param/grad pairs shape-matched\n";
                ++passed;
            } else {
                cout << "[FAIL] expected " << expected
                     << " params, got " << params.size() << "\n";
            }
        }
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
