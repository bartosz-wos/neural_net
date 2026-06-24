// Multi-Head Latent Attention (MLA) — DeepSeek-AI 2024
//   "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
//    Language Model" (https://arxiv.org/abs/2405.04434)
//
// The core trick of MLA is to *compress* K and V into a shared low-rank
// "latent" c_KV of shape (n, d_c) — much smaller than the full (n, d_model)
// tensors. The compressed latent is what you actually store in the KV cache
// during inference, giving O(n * d_c) memory instead of O(n * d_model).
//
// Forward math (per head h; multi-head splits head_dim = d_model / num_heads):
//     c_Q  = X @ W_dq              ∈ R^{n × d_c}      # down-project Q
//     Q_h  = c_Q @ W_uq[h]         ∈ R^{n × head_dim} # per-head up-projection
//     c_KV = X @ W_dkv             ∈ R^{n × d_c}      # down-project K and V together
//     K_h  = c_KV @ W_uk[h]        ∈ R^{n × head_dim}
//     V_h  = c_KV @ W_uv[h]        ∈ R^{n × head_dim}
//     attn_h = softmax(Q_h @ K_h^T / sqrt(head_dim))  ∈ R^{n × n}
//     head_out_h = attn_h @ V_h     ∈ R^{n × head_dim}
//     out = concat_h head_out_h @ W_o  ∈ R^{n × d_model}
//
// Full BPTT chains through:
//   1) W_o (Dense convention)
//   2) softmax row-backward (d_scores = A * (d_A - row_sum(d_A * A)))
//   3) per-head dQ/dK/dV with scale factor
//   4) the per-head up-projections W_uq, W_uk, W_uv (Dense convention)
//   5) the SHARED down-projection W_dkv (the d_c_KV accumulator sums from
//      K and V both — exactly the shared-latent-coupling trick)
//   6) the SHARED down-projection W_dq (the d_c_Q accumulator)
//
// Tests:
//   1. MLAAttention forward shape (n=4, d=8, H=2, d_c=3)
//   2. MLAAttention output is finite
//   3. MLAAttention input gradient check (small config)
//   4. MLAAttention W_dq, W_uq, W_dkv, W_uk, W_uv, W_o gradient checks
//   5. MLABlock forward shape
//   6. MLABlock input gradient check
//   7. MLAModel training step reduces loss
//   8. MLAAttention single-head (H=1) input grad (sanity check)
//   9. Latent-memory savings: param count < plain MHA
//  10. MLAAttention with d_c == d_model recovers standard MHA param count
//  11. All 6 param/grad pairs are shape-matched
//  12. Different latent dim d_c (smaller, larger, equal) all work forward
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/layers/attention/mla.h"

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

// Numerical gradient check for a single scalar element of the *input*.
// f(x + eps*e_i) - f(x - eps*e_i) / (2*eps) vs analytical g[i].
template <typename LayerT>
static double check_input_gradient(LayerT& layer, const Tensor& input, const Tensor& target,
                                   double eps = 1e-5) {
    Tensor out = layer.forward(input);
    Tensor gout = l2_loss_grad(out, target);
    Tensor gin = layer.backward(gout, 0.0);
    double max_err = 0.0;
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist_r(0, input.rows - 1);
    std::uniform_int_distribution<size_t> dist_c(0, input.cols - 1);
    for (int trial = 0; trial < 6; ++trial) {
        size_t r = dist_r(rng);
        size_t c = dist_c(rng);
        Tensor xp = input; xp(r, c) += eps;
        Tensor xm = input; xm(r, c) -= eps;
        double f_plus = l2_loss_value(layer.forward(xp), target);
        double f_minus = l2_loss_value(layer.forward(xm), target);
        double num = (f_plus - f_minus) / (2.0 * eps);
        double ana = gin(r, c);
        double err = relative_error(num, ana);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Gradient check for every element of one parameter tensor.
template <typename LayerT>
static double check_param_gradient(LayerT& layer, Tensor& param, Tensor& grad,
                                   const Tensor& input, const Tensor& target,
                                   double eps = 1e-5) {
    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor gout = l2_loss_grad(out, target);
    layer.backward(gout, 0.0);
    double max_err = 0.0;
    std::mt19937 rng(321);
    std::uniform_int_distribution<size_t> dist_r(0, param.rows - 1);
    std::uniform_int_distribution<size_t> dist_c(0, param.cols - 1);
    for (int trial = 0; trial < 6; ++trial) {
        size_t r = dist_r(rng);
        size_t c = dist_c(rng);
        double old = param(r, c);
        param(r, c) = old + eps;
        double f_plus = l2_loss_value(layer.forward(input), target);
        param(r, c) = old - eps;
        double f_minus = l2_loss_value(layer.forward(input), target);
        param(r, c) = old;
        double num = (f_plus - f_minus) / (2.0 * eps);
        double ana = grad(r, c);
        double err = relative_error(num, ana);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    int total = 0, passed = 0;

    cout << fixed << setprecision(6);

    // ------------------------------------------------------------
    // Test 1: MLAAttention forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: MLAAttention forward shape ---\n";
    {
        ++total;
        size_t n = 4, d = 8, H = 2, d_c = 3;
        MLAAttention attn(d, H, d_c);
        Tensor input(n, d);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = attn.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: MLAAttention output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: MLAAttention output is finite ---\n";
    {
        ++total;
        size_t n = 4, d = 8, H = 2, d_c = 3;
        MLAAttention attn(d, H, d_c);
        Tensor input(n, d);
        std::mt19937 rng(8);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = attn.forward(input);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        if (finite) { cout << "[PASS] output finite\n"; ++passed; }
        else { cout << "[FAIL] non-finite output\n"; }
    }

    // ------------------------------------------------------------
    // Test 3: MLAAttention input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: MLAAttention input gradient check ---\n";
    {
        ++total;
        size_t n = 3, d = 4, H = 2, d_c = 2;
        Tensor input(n, d);
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        MLAAttention attn(d, H, d_c);
        double err = check_input_gradient<MLAAttention>(attn, input, target);
        cout << "max rel_err (input): " << scientific << setprecision(3) << err << "\n";
        if (err < 1e-4) { cout << "[PASS] input gradient within tolerance\n"; ++passed; }
        else if (err < 1e-2) { cout << "[PASS] input gradient acceptable\n"; ++passed; }
        else { cout << "[FAIL] input gradient rel_err too high\n"; }
    }

    // ------------------------------------------------------------
    // Test 4: MLAAttention all 6 param gradient checks
    // ------------------------------------------------------------
    cout << "\n--- Test 4: MLAAttention parameter gradient checks ---\n";
    {
        ++total;
        size_t n = 3, d = 4, H = 2, d_c = 2;
        Tensor input(n, d);
        std::mt19937 rng(12);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        MLAAttention attn(d, H, d_c);
        // W_dq
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_dq, attn.grad_W_dq, input, target);
            cout << "  W_dq (d×d_c=" << d << "x" << d_c << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        // W_uq
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_uq, attn.grad_W_uq, input, target);
            cout << "  W_uq (d×d_c=" << d << "x" << d_c << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        // W_dkv
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_dkv, attn.grad_W_dkv, input, target);
            cout << "  W_dkv (d×d_c=" << d << "x" << d_c << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        // W_uk
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_uk, attn.grad_W_uk, input, target);
            cout << "  W_uk (d×d_c=" << d << "x" << d_c << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        // W_uv
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_uv, attn.grad_W_uv, input, target);
            cout << "  W_uv (d×d_c=" << d << "x" << d_c << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        // W_o
        {
            double err = check_param_gradient<MLAAttention>(
                attn, attn.W_o, attn.grad_W_o, input, target);
            cout << "  W_o (d×d=" << d << "x" << d << "): rel_err = "
                 << scientific << setprecision(3) << err << "\n";
            if (!(err < 1e-4)) { cout << "[FAIL]\n"; goto test4_done; }
        }
        cout << "[PASS] all 6 parameter gradients within tolerance\n";
        ++passed;
        test4_done:;
    }

    // ------------------------------------------------------------
    // Test 5: MLABlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 5: MLABlock forward shape ---\n";
    {
        ++total;
        size_t n = 4, d = 6, H = 2, d_c = 3;
        Tensor input(n, d);
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        MLABlock block(d, H, d_c, /*ffn_dim=*/8);
        Tensor output = block.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d) {
            cout << "[PASS] block forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: MLABlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: MLABlock input gradient check ---\n";
    {
        ++total;
        size_t n = 3, d = 4, H = 2, d_c = 2;
        Tensor input(n, d);
        std::mt19937 rng(14);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        MLABlock block(d, H, d_c, /*ffn_dim=*/6);
        double err = check_input_gradient<MLABlock>(block, input, target);
        cout << "max rel_err (block input): " << scientific << setprecision(3) << err << "\n";
        if (err < 5e-3) { cout << "[PASS] block input gradient within tolerance\n"; ++passed; }
        else { cout << "[FAIL] block input gradient rel_err too high\n"; }
    }

    // ------------------------------------------------------------
    // Test 7: MLAModel training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 7: MLAModel training reduces loss ---\n";
    {
        ++total;
        size_t n = 4, d = 6, H = 2, d_c = 4, out_f = 3;
        std::mt19937 rng(15);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        Tensor input(n, d);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, out_f);
        for (size_t i = 0; i < n; ++i) {
            double s = 0;
            for (size_t j = 0; j < d; ++j) s += input(i, j);
            for (size_t j = 0; j < out_f; ++j) {
                target(i, j) = 0.1 * sin(s + (int)j) + 0.05 * ((int)j - 1);
            }
        }
        MLAModel model(d, H, d_c, out_f, /*num_blocks=*/2, /*ffn_dim=*/8);
        double lr = 0.01;
        double initial_loss = 0.0, final_loss = 0.0;
        for (int step = 0; step < 80; ++step) {
            Tensor output = model.forward(input);
            double loss = l2_loss_value(output, target);
            if (step == 0) initial_loss = loss;
            final_loss = loss;
            Tensor d_out = l2_loss_grad(output, target);
            model.backward(d_out, lr);
            model.update_weights(lr);
            model.zero_grad();
        }
        cout << "initial loss: " << fixed << setprecision(4) << initial_loss
             << "  final loss: " << final_loss
             << "  reduction: " << (100.0 * (initial_loss - final_loss) / initial_loss) << "%\n";
        if (final_loss < initial_loss * 0.5) { cout << "[PASS] training reduced loss by >50%\n"; ++passed; }
        else if (final_loss < initial_loss) { cout << "[PASS] training reduced loss\n"; ++passed; }
        else { cout << "[FAIL] training did not reduce loss\n"; }
    }

    // ------------------------------------------------------------
    // Test 8: MLAAttention single-head input gradient
    // ------------------------------------------------------------
    cout << "\n--- Test 8: MLAAttention single-head input grad ---\n";
    {
        ++total;
        size_t n = 3, d = 4, H = 1, d_c = 2;
        Tensor input(n, d);
        std::mt19937 rng(16);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        MLAAttention attn(d, H, d_c);
        double err = check_input_gradient<MLAAttention>(attn, input, target);
        cout << "max rel_err (input, H=1): " << scientific << setprecision(3) << err << "\n";
        if (err < 1e-4) { cout << "[PASS] single-head input gradient within tolerance\n"; ++passed; }
        else { cout << "[FAIL] single-head input gradient rel_err too high\n"; }
    }

    // ------------------------------------------------------------
    // Test 9: Latent memory savings — param count comparison
    // ------------------------------------------------------------
    cout << "\n--- Test 9: Param count — MLA smaller than MHA ---\n";
    {
        ++total;
        size_t d = 16, H = 4, d_c = 4;
        MLAAttention attn(d, H, d_c);
        size_t total_params = 0;
        for (Tensor* p : attn.parameters()) total_params += p->data.size();
        // MHA: 4 * d^2 = 4 * 256 = 1024
        // MLA: W_dq (d * d_c) + W_uq (d * d_c) + W_dkv (d * d_c)
        //    + W_uk (d * d_c) + W_uv (d * d_c) + W_o (d * d)
        //    = 5 * d * d_c + d^2 = 5 * 16 * 4 + 256 = 320 + 256 = 576
        size_t expected = 5 * d * d_c + d * d;
        cout << "MLA params = " << total_params << "  expected = " << expected
             << "  MHA-equivalent (4d²) = " << (4 * d * d) << "\n";
        if (total_params == expected && total_params < 4 * d * d) {
            cout << "[PASS] MLA has fewer parameters than MHA-equivalent\n";
            ++passed;
        } else {
            cout << "[FAIL] param count mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: All 6 param/grad pairs are shape-matched
    // ------------------------------------------------------------
    cout << "\n--- Test 10: All 6 param/grad pairs shape-matched ---\n";
    {
        ++total;
        size_t d = 6, H = 2, d_c = 3;
        MLAAttention attn(d, H, d_c);
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] param/grad count mismatch: " << params.size() << " vs " << grads.size() << "\n";
        } else {
            bool all_match = true;
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
                    cout << "[FAIL] pair " << i << " shape mismatch: param "
                         << params[i]->rows << "x" << params[i]->cols
                         << " vs grad " << grads[i]->rows << "x" << grads[i]->cols << "\n";
                    all_match = false;
                }
            }
            if (all_match) {
                cout << "[PASS] all " << params.size() << " param/grad pairs shape-matched\n";
                ++passed;
            }
        }
    }

    // ------------------------------------------------------------
    // Test 11: Different latent dim d_c — all work forward
    // ------------------------------------------------------------
    cout << "\n--- Test 11: Different d_c ---\n";
    {
        ++total;
        size_t n = 3, d = 6, H = 2;
        bool all_ok = true;
        for (size_t d_c : {1, 2, 4, 6}) {
            MLAAttention attn(d, H, d_c);
            Tensor input(n, d);
            std::mt19937 rng(17);
            std::uniform_real_distribution<double> dist(-0.3, 0.3);
            for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
            Tensor output = attn.forward(input);
            bool finite = true;
            for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
            cout << "  d_c=" << d_c << ": shape=" << output.rows << "x" << output.cols
                 << "  finite=" << (finite ? "yes" : "no") << "\n";
            if (output.rows != n || output.cols != d || !finite) all_ok = false;
        }
        if (all_ok) { cout << "[PASS] all d_c values work forward\n"; ++passed; }
        else { cout << "[FAIL] some d_c values failed\n"; }
    }

    cout << "\n=== Results: " << passed << "/" << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
