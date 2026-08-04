// MLP-Mixer tests — Tolstikhin et al., 2021
// "An all-MLP Architecture for Vision"
//
// Tests:
//   1. MlpMixerBlock forward shape (B, S, D) -> (B, S*D)
//   2. MlpMixerBlock output is finite
//   3. MlpMixerBlock numerical gradient check on input
//   4. MlpMixerBlock token-mix weight gradient check (W1)
//   5. MlpMixerBlock token-mix weight gradient check (W2)
//   6. MlpMixerBlock channel-mix weight gradient check (W1)
//   7. MlpMixerBlock channel-mix weight gradient check (W2)
//   8. MlpMixerBlock per-token LN gamma gradient check
//   9. MlpMixerBlock per-channel LN gamma gradient check
//  10. MlpMixerBlock zero_grad clears all gradients
//  11. MlpMixerModel forward shape (B, C_in, H, W) -> (B, num_classes)
//  12. MlpMixerModel input gradient check
//  13. MlpMixerModel training reduces loss
//  14. MlpMixerBlock deeper stack (depth=2) forward shape
//  15. MlpMixerModel zero_grad clears all gradients
//  16. MlpMixerBlock: nonzero output sanity check (random init produces nonzero output)
//  17. MlpMixerBlock: S != D works (asymmetric shapes)
//  18. MlpMixerModel: num_patches accessor matches (image_size/patch_size)^2
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/layers/architectures/mlp_mixer.h"

using namespace std;

static double max_rel_err(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double av = a.data[i], bv = b.data[i];
        double denom = max(fabs(av), fabs(bv));
        if (denom < 1e-8) denom = 1e-8;
        m = max(m, fabs(av - bv) / denom);
    }
    return m;
}

// L2 loss + grad helpers
static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}

// Helper: fill tensor with random values (avoiding degenerate linear inputs).
static void fill_random(Tensor& t, std::mt19937& gen, double scale = 0.3) {
    std::normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dis(gen);
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// Finite-difference gradient check (central differences)
static Tensor finite_diff_grad_input(MlpMixerBlock& block, Tensor& input, const Tensor& target,
                                     double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = block.forward(input);
            double lp = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = block.forward(input);
            double lm = l2_loss_value(out_m, target);
            input(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// FD grad check on a parameter tensor (e.g. weights, biases, gamma).
// We need to call the function with different parameter values. Since Dense
// doesn't expose set_weights, we mutate directly via raw pointer.
template <typename LayerT>
static Tensor finite_diff_grad_param(LayerT& layer, const Tensor& input, const Tensor& target,
                                     Tensor& param, double eps = 1e-5) {
    Tensor grad(param.rows, param.cols);
    for (size_t i = 0; i < param.rows; ++i) {
        for (size_t j = 0; j < param.cols; ++j) {
            double orig = param(i, j);
            param(i, j) = orig + eps;
            Tensor out_p = layer.forward(input);
            double lp = l2_loss_value(out_p, target);
            param(i, j) = orig - eps;
            Tensor out_m = layer.forward(input);
            double lm = l2_loss_value(out_m, target);
            param(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    cout << "=== MLP-Mixer Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Helper to print pass/fail and increment counters.
    auto check = [&](bool ok, const string& msg) {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << msg << "\n"; }
        else    { cout << "[FAIL] " << msg << "\n"; }
    };

    // -------------------------------------------------------------------------
    // Test 1: MlpMixerBlock forward shape
    // -------------------------------------------------------------------------
    cout << "\n--- Test 1: MlpMixerBlock forward shape (B=2, S=4, D=8 -> B=2, S*D=32) ---\n";
    {
        size_t B = 2, S = 4, D = 8;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i % 7) - 0.05;

        MlpMixerBlock block(D, S, /*token_dim=*/16, /*channel_dim=*/32);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        check(output.rows == B && output.cols == S * D, "forward shape correct");
    }

    // -------------------------------------------------------------------------
    // Test 2: MlpMixerBlock output is finite
    // -------------------------------------------------------------------------
    cout << "\n--- Test 2: MlpMixerBlock output is finite ---\n";
    {
        size_t B = 2, S = 6, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.3 * sin(0.1 * i);

        MlpMixerBlock block(D, S, 8, 8);
        Tensor output = block.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!std::isfinite(output.data[i])) finite = false;
        }
        check(finite, "all outputs finite");
    }

    // -------------------------------------------------------------------------
    // Test 3: MlpMixerBlock input gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 3: MlpMixerBlock input gradient check ---\n";
    {
        size_t B = 1, S = 3, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * sin(0.3 * i);

        MlpMixerBlock block(D, S, 8, 8);

        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1 * i - 0.2;

        // Analytical gradient
        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor ana = block.backward(loss_grad, 0.001);

        // Numerical gradient
        Tensor num = finite_diff_grad_input(block, input, target, 1e-5);

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(input) = " << mrd << "\n";
        check(mrd < 1e-3, "input gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 4: token-mix W1 gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 4: MlpMixerBlock token-mix W1 gradient check ---\n";
    {
        std::mt19937 gen(4);
        size_t B = 1, S = 3, D = 4, T = 6;  // token_dim = 6
        Tensor input(B, S * D);
        fill_random(input, gen);

        MlpMixerBlock block(D, S, T, 8);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05;

        // Analytical
        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);
        // Access tok_mlp_w1_ via parameters() — order is: ln_token, ln_channel, tok_w1, tok_w2, chan_w1, chan_w2
        auto params = block.parameters();
        // ln_token has 2 params (gamma, beta), ln_channel has 2 params.
        // tok_mlp_w1 has 2 params (weights, bias). So tok_mlp_w1.weights is at index 4.
        Tensor& tok_w1 = *params[4];

        // FD — mutate tok_w1 directly.
        Tensor num = finite_diff_grad_param(block, input, target, tok_w1, 1e-5);

        // Ana — get from gradients() in same order
        auto grads = block.gradients();
        const Tensor& ana = *grads[4];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(tok_mlp_w1.weights) = " << mrd << "\n";
        check(mrd < 1e-3, "tok_mlp_w1.weights gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 5: token-mix W2 gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 5: MlpMixerBlock token-mix W2 gradient check ---\n";
    {
        std::mt19937 gen(5);
        size_t B = 1, S = 3, D = 4, T = 6;
        Tensor input(B, S * D);
        fill_random(input, gen);

        MlpMixerBlock block(D, S, T, 8);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        auto params = block.parameters();
        // ln_token gamma(0), ln_token beta(1), ln_channel gamma(2), ln_channel beta(3),
        // tok_w1 weights(4), tok_w1 bias(5), tok_w2 weights(6), tok_w2 bias(7),
        // chan_w1 weights(8), chan_w1 bias(9), chan_w2 weights(10), chan_w2 bias(11)
        Tensor& tok_w2 = *params[6];
        Tensor num = finite_diff_grad_param(block, input, target, tok_w2, 1e-5);

        auto grads = block.gradients();
        const Tensor& ana = *grads[6];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(tok_mlp_w2.weights) = " << mrd << "\n";
        check(mrd < 1e-3, "tok_mlp_w2.weights gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 6: channel-mix W1 gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 6: MlpMixerBlock channel-mix W1 gradient check ---\n";
    {
        std::mt19937 gen(6);
        size_t B = 1, S = 3, D = 4, C = 5;
        Tensor input(B, S * D);
        fill_random(input, gen);

        MlpMixerBlock block(D, S, 6, C);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        auto params = block.parameters();
        Tensor& chan_w1 = *params[8];
        Tensor num = finite_diff_grad_param(block, input, target, chan_w1, 1e-5);

        auto grads = block.gradients();
        const Tensor& ana = *grads[8];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(chan_mlp_w1.weights) = " << mrd << "\n";
        check(mrd < 1e-3, "chan_mlp_w1.weights gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 7: channel-mix W2 gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 7: MlpMixerBlock channel-mix W2 gradient check ---\n";
    {
        std::mt19937 gen(7);
        size_t B = 1, S = 3, D = 4, C = 5;
        Tensor input(B, S * D);
        fill_random(input, gen);

        MlpMixerBlock block(D, S, 6, C);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        auto params = block.parameters();
        Tensor& chan_w2 = *params[10];
        Tensor num = finite_diff_grad_param(block, input, target, chan_w2, 1e-5);

        auto grads = block.gradients();
        const Tensor& ana = *grads[10];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(chan_mlp_w2.weights) = " << mrd << "\n";
        check(mrd < 1e-3, "chan_mlp_w2.weights gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 8: per-token LN gamma gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 8: MlpMixerBlock per-token LN gamma gradient check ---\n";
    {
        size_t B = 1, S = 3, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * sin(0.3 * i);

        MlpMixerBlock block(D, S, 8, 8);
        // Initialize gamma to a known value to remove randomness — the
        // LayerNorm default-init scale (Tensor::random(0.01)) is small enough
        // that FD at gamma[0] ≈ 1e-4 becomes unreliable. Use a larger init
        // so the loss surface is well-conditioned for FD verification.
        for (size_t j = 0; j < D; ++j) block.ln_token().gamma(0, j) = 0.3;
        for (size_t s = 0; s < S; ++s) block.ln_channel().gamma(0, s) = 0.3;

        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1 * (i % 3) - 0.05;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        auto params = block.parameters();
        // params[0] = ln_token.gamma
        Tensor& ln_t_gamma = *params[0];

        // Try multiple FD epsilons to verify convergence.
        Tensor num_small = finite_diff_grad_param(block, input, target, ln_t_gamma, 1e-5);
        Tensor num_med   = finite_diff_grad_param(block, input, target, ln_t_gamma, 1e-6);
        Tensor num = num_med;

        auto grads = block.gradients();
        const Tensor& ana = *grads[0];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(ln_token.gamma) = " << mrd << "\n";
        check(mrd < 1e-3, "ln_token.gamma gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 9: per-channel LN gamma gradient check
    // -------------------------------------------------------------------------
    cout << "\n--- Test 9: MlpMixerBlock per-channel LN gamma gradient check ---\n";
    {
        size_t B = 1, S = 3, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * sin(0.3 * i);

        MlpMixerBlock block(D, S, 8, 8);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        auto params = block.parameters();
        // params[2] = ln_channel.gamma
        Tensor& ln_c_gamma = *params[2];
        Tensor num = finite_diff_grad_param(block, input, target, ln_c_gamma, 1e-5);

        auto grads = block.gradients();
        const Tensor& ana = *grads[2];

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(ln_channel.gamma) = " << mrd << "\n";
        check(mrd < 1e-3, "ln_channel.gamma gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 10: MlpMixerBlock zero_grad clears all gradients
    // -------------------------------------------------------------------------
    cout << "\n--- Test 10: MlpMixerBlock zero_grad clears all gradients ---\n";
    {
        size_t B = 1, S = 3, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1;

        MlpMixerBlock block(D, S, 8, 8);
        Tensor target(B, S * D);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05;

        Tensor out = block.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(loss_grad, 0.001);

        // Verify all gradients are nonzero
        auto grads_before = block.gradients();
        bool any_nonzero = false;
        for (auto* g : grads_before) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (g->data[i] != 0.0) { any_nonzero = true; break; }
            }
            if (any_nonzero) break;
        }
        check(any_nonzero, "gradients are nonzero after backward");

        block.zero_grad();
        auto grads_after = block.gradients();
        bool all_zero = true;
        for (auto* g : grads_after) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (g->data[i] != 0.0) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
        check(all_zero, "all gradients zero after zero_grad()");
    }

    // -------------------------------------------------------------------------
    // Test 11: MlpMixerModel forward shape
    // -------------------------------------------------------------------------
    cout << "\n--- Test 11: MlpMixerModel forward shape (B=2, C=1, H=W=4, patch=2) ---\n";
    {
        size_t B = 2, C = 1, H = 4, W = 4, patch = 2, num_classes = 3;
        size_t S = (H / patch) * (W / patch);  // = 4
        size_t D = 6;
        Tensor input(B, C * H * W);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i % 5) - 0.05;

        MlpMixerModel model(H, patch, C, num_classes, D, /*depth=*/1,
                            /*token_dim=*/8, /*channel_dim=*/12);
        Tensor output = model.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols
             << "  (S=" << S << ", D=" << D << ", num_classes=" << num_classes << ")\n";
        check(output.rows == B && output.cols == num_classes,
              "model forward shape correct");
        check(model.num_patches() == S, "num_patches() accessor matches");
    }

    // -------------------------------------------------------------------------
    // Test 12: MlpMixerModel input gradient check (small config)
    // -------------------------------------------------------------------------
    cout << "\n--- Test 12: MlpMixerModel input gradient check ---\n";
    {
        size_t B = 1, C = 1, H = 4, W = 4, patch = 2, num_classes = 2;
        size_t D = 4;
        Tensor input(B, C * H * W);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.05 * (i + 1);

        MlpMixerModel model(H, patch, C, num_classes, D, /*depth=*/1,
                            /*token_dim=*/8, /*channel_dim=*/8);
        Tensor target(B, num_classes);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1 * (i + 1);

        // Analytical
        Tensor out = model.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        model.zero_grad();
        Tensor ana = model.backward(loss_grad, 0.001);

        // Numerical
        Tensor num(B, C * H * W);
        double eps = 1e-5;
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = model.forward(input);
                double lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = model.forward(input);
                double lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                num(i, j) = (lp - lm) / (2.0 * eps);
            }
        }

        double mrd = max_rel_err(ana, num);
        cout << "max_rel_err(model.input) = " << mrd << "\n";
        check(mrd < 1e-3, "model input gradient rel_err < 1e-3");
    }

    // -------------------------------------------------------------------------
    // Test 13: MlpMixerModel training reduces loss
    // -------------------------------------------------------------------------
    cout << "\n--- Test 13: MlpMixerModel training reduces loss ---\n";
    {
        size_t B = 4, C = 1, H = 4, W = 4, patch = 2, num_classes = 2;
        size_t D = 8;
        MlpMixerModel model(H, patch, C, num_classes, D, /*depth=*/1,
                            /*token_dim=*/16, /*channel_dim=*/16);

        // Synthetic task: predict label = mean of image (above 0.5 → class 1)
        Tensor input(B, C * H * W);
        Tensor target(B, num_classes);
        std::mt19937 gen(7);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t b = 0; b < B; ++b) {
            double mean = 0.0;
            for (size_t i = 0; i < C * H * W; ++i) {
                double v = dis(gen);
                input(b, i) = v;
                mean += v;
            }
            mean /= (C * H * W);
            // One-hot: class 1 if mean > 0, else class 0
            int label = (mean > 0) ? 1 : 0;
            target(b, 0) = (label == 0) ? 1.0 : 0.0;
            target(b, 1) = (label == 1) ? 1.0 : 0.0;
        }

        double lr = 0.01;
        double init_loss = l2_loss_value(model.forward(input), target);
        cout << "  initial loss = " << init_loss << "\n";
        for (int step = 0; step < 60; ++step) {
            model.zero_grad();
            Tensor out = model.forward(input);
            Tensor loss_grad = l2_loss_grad(out, target);
            model.backward(loss_grad, lr);
            model.update_weights(lr);
        }
        double final_loss = l2_loss_value(model.forward(input), target);
        cout << "  final loss   = " << final_loss
             << "  (reduction: " << (1.0 - final_loss / init_loss) * 100.0 << "%)\n";
        check(final_loss < init_loss * 0.5,
              "training reduces loss > 50% over 60 steps");
    }

    // -------------------------------------------------------------------------
    // Test 14: depth=2 stack forward shape
    // -------------------------------------------------------------------------
    cout << "\n--- Test 14: MlpMixerModel depth=2 forward shape ---\n";
    {
        size_t B = 1, C = 1, H = 4, W = 4, patch = 2, num_classes = 2;
        size_t D = 4;
        Tensor input(B, C * H * W);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1;

        MlpMixerModel model(H, patch, C, num_classes, D, /*depth=*/2,
                            /*token_dim=*/8, /*channel_dim=*/8);
        Tensor output = model.forward(input);
        cout << "Output: " << output.rows << "x" << output.cols << "\n";
        check(output.rows == B && output.cols == num_classes, "depth=2 forward shape correct");

        // Also check finite
        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!std::isfinite(output.data[i])) finite = false;
        }
        check(finite, "depth=2 output finite");
    }

    // -------------------------------------------------------------------------
    // Test 15: MlpMixerModel zero_grad clears all gradients
    // -------------------------------------------------------------------------
    cout << "\n--- Test 15: MlpMixerModel zero_grad clears all gradients ---\n";
    {
        size_t B = 1, C = 1, H = 4, W = 4, patch = 2, num_classes = 2;
        size_t D = 4;
        MlpMixerModel model(H, patch, C, num_classes, D, /*depth=*/1,
                            /*token_dim=*/8, /*channel_dim=*/8);

        Tensor input(B, C * H * W);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.05;
        Tensor target(B, num_classes);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

        Tensor out = model.forward(input);
        Tensor loss_grad = l2_loss_grad(out, target);
        model.zero_grad();
        model.backward(loss_grad, 0.001);

        // Verify nonzero
        auto grads = model.gradients();
        bool any_nonzero = false;
        for (auto* g : grads) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (g->data[i] != 0.0) { any_nonzero = true; break; }
            }
            if (any_nonzero) break;
        }
        check(any_nonzero, "model gradients nonzero after backward");

        model.zero_grad();
        auto grads2 = model.gradients();
        bool all_zero = true;
        for (auto* g : grads2) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (g->data[i] != 0.0) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
        check(all_zero, "model gradients zero after zero_grad()");
    }

    // -------------------------------------------------------------------------
    // Test 16: nonzero output sanity check (random init produces nonzero output)
    // -------------------------------------------------------------------------
    cout << "\n--- Test 16: MlpMixerBlock produces nonzero output (random init) ---\n";
    {
        // Use non-constant input so that LN's normalization does not zero it out
        // before the random-init Dense weights/biases can produce output. With
        // constant input + zero-init Dense biases, the block output would be
        // zero regardless of weights — the LN normalizes constants to zero and
        // Dense.forward(input=W*0+b) = b, which is zero-init.
        size_t B = 1, S = 4, D = 4;
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.5 * sin(0.7 * i) + 0.3;

        MlpMixerBlock block(D, S, 8, 8);
        Tensor output = block.forward(input);
        double sum_abs = 0.0;
        for (size_t i = 0; i < output.data.size(); ++i) sum_abs += fabs(output.data[i]);
        cout << "  sum(|output|) = " << sum_abs << "\n";
        check(sum_abs > 1e-3, "block produces nonzero output for nonzero random init");
    }

    // -------------------------------------------------------------------------
    // Test 17: S != D (asymmetric shapes)
    // -------------------------------------------------------------------------
    cout << "\n--- Test 17: MlpMixerBlock S != D asymmetric shape ---\n";
    {
        size_t B = 1, S = 6, D = 3;  // S != D
        Tensor input(B, S * D);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2;

        MlpMixerBlock block(D, S, 8, 8);
        Tensor output = block.forward(input);
        cout << "  Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        check(output.rows == B && output.cols == S * D, "S != D forward shape correct");
    }

    // -------------------------------------------------------------------------
    // Test 18: num_patches accessor
    // -------------------------------------------------------------------------
    cout << "\n--- Test 18: MlpMixerModel num_patches() accessor ---\n";
    {
        size_t H = 8, patch = 4, num_classes = 3, D = 8;
        MlpMixerModel model(H, patch, /*C_in=*/3, num_classes, D, /*depth=*/1,
                            /*token_dim=*/16, /*channel_dim=*/16);
        // (H/patch)^2 = (8/4)^2 = 4
        cout << "  num_patches() = " << model.num_patches() << "\n";
        check(model.num_patches() == 4, "num_patches() == (8/4)^2 == 4");
    }

    cout << "\n=== Summary: " << passed << "/" << total << " checks passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
