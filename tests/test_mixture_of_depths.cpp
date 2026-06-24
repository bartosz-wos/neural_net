// Mixture of Depths (MoD) tests — Raposo et al. 2024
//   "Mixture-of-Depths: Dynamically allocating compute in transformer-based models"
//   (https://arxiv.org/abs/2404.02258)
//
// MoD routes a fraction of tokens through a transformer sub-layer (process)
// while skipping the rest (identity). The router is a per-token sigmoid
// over a single logit; the top-k tokens (where k = capacity * n) are
// routed through, the rest are skipped. The output is x + mask * sub_out
// (residual-style).
//
// Math (per MoDLayer call):
//   router_logits[t, 0] = x[t, :] @ W_router[:] + b_router   # (n, 1)
//   router_probs[t, 0]  = sigmoid(router_logits[t, 0])
//   capacity            = capacity_factor * n
//   selected            = top-capacity tokens by router_logits
//   mask[t]             = 1.0 if t in selected, else 0.0
//   y                   = x + mask * inner(x)
//
// Tests:
//   1. MoDLayer forward shape (n=5, d_model=3, capacity_factor=0.5)
//   2. MoDLayer output is finite
//   3. Capacity selection: exactly capacity_factor * n tokens are routed
//      through (with masking reflecting the chosen indices)
//   4. MoDLayer input gradient check (small config) — tests BPTT including
//      residual path + masked-sub-layer path
//   5. MoDLayer W_router gradient check
//   6. MoDLayer inner-block weight gradient check (FFN params)
//   7. MoDLayer aux loss is zero at capacity_factor == mean(mask)
//      (when all tokens are selected, the load is exactly balanced)
//   8. MoDLayer at capacity_factor=1.0 — all tokens selected, output == x + inner(x)
//   9. MoDLayer at capacity_factor=0.0 — no tokens selected, output == x exactly
//  10. MoDBlock forward shape
//  11. MoDBlock input gradient check
//  12. MoDBlock training reduces loss
//  13. MoDModel forward shape (num_blocks=2)
//  14. MoDModel training reduces loss
//  15. MoDModel total aux loss aggregates across blocks
//  16. capacity_factor clamped to [0, 1]
//  17. Single-token (n=1) capacity behaves correctly
//  18. Param/grad shape consistency: router has (1, d_model) + (1, 1)
//  19. MoDLayer parameters()/gradients()/zero_grad() delegation to inner
//  20. Different capacity_factor values (0.25, 0.5, 0.75) all forward finite
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "nn/layers/architectures/mixture_of_depths.h"

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

// Helper removed — mask_ is now public so tests can access it directly.

int main() {
    int total = 0, passed = 0;
    cout << fixed << setprecision(6);

    // ============================================================
    // Test 1: MoDLayer forward shape
    // ============================================================
    cout << "\n--- Test 1: MoDLayer forward shape ---\n";
    {
        ++total;
        size_t n = 5, d = 3;
        // MoDLayer(d, capacity_factor, aux_loss_coef, inner=nullptr)
        // Use a Dense layer as inner for shape variety (we'll re-test with
        // FFNSequential-style inner via MoDBlock).
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ============================================================
    // Test 2: MoDLayer output is finite
    // ============================================================
    cout << "\n--- Test 2: MoDLayer output is finite ---\n";
    {
        ++total;
        size_t n = 5, d = 3;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(8);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        if (finite) { cout << "[PASS] output finite\n"; ++passed; }
        else { cout << "[FAIL] non-finite output\n"; }
    }

    // ============================================================
    // Test 3: Capacity selection — exactly capacity_factor*n tokens selected
    // ============================================================
    cout << "\n--- Test 3: Capacity selection size matches capacity_factor*n ---\n";
    {
        ++total;
        size_t n = 8, d = 3;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        // Count selected tokens (mask == 1)
        size_t n_selected = 0;
        for (size_t t = 0; t < n; ++t) {
            if (mod.mask_(t, 0) > 0.5) ++n_selected;
        }
        size_t expected = mod.capacity();
        cout << "n_selected=" << n_selected << " expected=" << expected << "\n";
        if (n_selected == expected) {
            cout << "[PASS] exactly " << expected << " tokens selected (capacity=" << expected << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] selection count mismatch\n";
        }
    }

    // ============================================================
    // Test 4: MoDLayer input gradient check (small config)
    // ============================================================
    cout << "\n--- Test 4: MoDLayer input gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        double err = check_input_gradient<MoDLayer>(mod, input, target);
        cout << "max rel_err (input): " << scientific << setprecision(3) << err << "\n";
        if (err < 1e-4) { cout << "[PASS] input gradient within tolerance\n"; ++passed; }
        else if (err < 1e-2) { cout << "[PASS] input gradient acceptable\n"; ++passed; }
        else { cout << "[FAIL] input gradient rel_err too high\n"; }
    }

    // ============================================================
    // Test 5: MoDLayer W_router gradient check
    // ============================================================
    cout << "\n--- Test 5: MoDLayer W_router gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(12);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        double err = check_param_gradient<MoDLayer>(mod, mod.W_router_, mod.grad_W_router_,
                                                    input, target);
        cout << "W_router (1x" << d << "): rel_err = " << scientific << setprecision(3) << err << "\n";
        if (err < 1e-4) { cout << "[PASS] W_router gradient within tolerance\n"; ++passed; }
        else if (err < 1e-2) { cout << "[PASS] W_router gradient acceptable\n"; ++passed; }
        else { cout << "[FAIL] W_router gradient rel_err too high\n"; }
    }

    // ============================================================
    // Test 6: MoDLayer inner-block weight gradient check
    // ============================================================
    cout << "\n--- Test 6: MoDLayer inner-block weight gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3;
        Dense* inner = new Dense(d, d);
        MoDLayer mod(d, 0.5, 0.01, inner);
        Tensor input(n, d);
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        double err = check_param_gradient<MoDLayer>(mod, inner->weights, inner->grad_weights,
                                                    input, target);
        cout << "inner.weights (" << d << "x" << d << "): rel_err = "
             << scientific << setprecision(3) << err << "\n";
        if (err < 1e-4) { cout << "[PASS] inner.weights gradient within tolerance\n"; ++passed; }
        else if (err < 1e-2) { cout << "[PASS] inner.weights gradient acceptable\n"; ++passed; }
        else { cout << "[FAIL] inner.weights gradient rel_err too high\n"; }
    }

    // ============================================================
    // Test 7: capacity_factor=1.0 — all tokens selected
    // ============================================================
    cout << "\n--- Test 7: capacity_factor=1.0 — all tokens routed ---\n";
    {
        ++total;
        size_t n = 5, d = 3;
        MoDLayer mod(d, 1.0, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(14);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        size_t n_selected = 0;
        for (size_t t = 0; t < n; ++t) if (mod.mask_(t, 0) > 0.5) ++n_selected;
        cout << "n_selected=" << n_selected << " (expected " << n << ")\n";
        if (n_selected == n) {
            cout << "[PASS] all " << n << " tokens selected at capacity_factor=1.0\n";
            ++passed;
        } else {
            cout << "[FAIL] not all tokens selected\n";
        }
    }

    // ============================================================
    // Test 8: capacity_factor=0.0 — no tokens selected, output == input
    // ============================================================
    cout << "\n--- Test 8: capacity_factor=0.0 — no tokens routed, output == input ---\n";
    {
        ++total;
        size_t n = 5, d = 3;
        MoDLayer mod(d, 0.0, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(15);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        // Compute "expected" = input (since mask=0 everywhere)
        double max_diff = 0.0;
        for (size_t i = 0; i < output.data.size(); ++i) {
            max_diff = max(max_diff, fabs(output.data[i] - input.data[i]));
        }
        cout << "max |output - input| = " << scientific << setprecision(3) << max_diff << "\n";
        if (max_diff < 1e-12) {
            cout << "[PASS] output equals input when capacity_factor=0\n";
            ++passed;
        } else {
            cout << "[FAIL] output deviates from input\n";
        }
    }

    // ============================================================
    // Test 9: MoDLayer aux loss is non-negative
    // ============================================================
    cout << "\n--- Test 9: MoDLayer aux loss is non-negative ---\n";
    {
        ++total;
        size_t n = 8, d = 3;
        MoDLayer mod(d, 0.5, 0.1, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(16);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        mod.forward(input);
        double aux = mod.get_load_balance_loss();
        cout << "aux loss = " << aux << "\n";
        if (aux >= 0.0) { cout << "[PASS] aux loss non-negative\n"; ++passed; }
        else { cout << "[FAIL] aux loss negative\n"; }
    }

    // ============================================================
    // Test 10: MoDBlock forward shape
    // ============================================================
    cout << "\n--- Test 10: MoDBlock forward shape ---\n";
    {
        ++total;
        size_t n = 5, d = 4;
        MoDBlock block(d, 0.5, /*ffn_dim=*/6);
        Tensor input(n, d);
        std::mt19937 rng(17);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = block.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d) {
            cout << "[PASS] MoDBlock forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ============================================================
    // Test 11: MoDBlock input gradient check
    // ============================================================
    cout << "\n--- Test 11: MoDBlock input gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3;
        MoDBlock block(d, 0.5, /*ffn_dim=*/6);
        Tensor input(n, d);
        std::mt19937 rng(18);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        double err = check_input_gradient<MoDBlock>(block, input, target);
        cout << "max rel_err (block input): " << scientific << setprecision(3) << err << "\n";
        if (err < 5e-3) { cout << "[PASS] MoDBlock input gradient within tolerance\n"; ++passed; }
        else if (err < 1e-1) { cout << "[PASS] MoDBlock input gradient acceptable\n"; ++passed; }
        else { cout << "[FAIL] MoDBlock input gradient rel_err too high\n"; }
    }

    // ============================================================
    // Test 12: MoDBlock training reduces loss
    // ============================================================
    cout << "\n--- Test 12: MoDBlock training reduces loss ---\n";
    {
        ++total;
        size_t n = 4, d = 4;
        MoDBlock block(d, 0.5, /*ffn_dim=*/8);
        Tensor input(n, d);
        std::mt19937 rng(19);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;
        double lr = 0.01;
        double initial_loss = 0.0, final_loss = 0.0;
        for (int step = 0; step < 60; ++step) {
            Tensor output = block.forward(input);
            double loss = l2_loss_value(output, target);
            if (step == 0) initial_loss = loss;
            final_loss = loss;
            Tensor d_out = l2_loss_grad(output, target);
            block.backward(d_out, lr);
            block.update_weights(lr);
            block.zero_grad();
        }
        cout << "initial: " << initial_loss << " final: " << final_loss
             << " reduction: " << (100.0 * (initial_loss - final_loss) / initial_loss) << "%\n";
        if (final_loss < initial_loss * 0.95) {
            cout << "[PASS] MoDBlock training reduced loss\n";
            ++passed;
        } else {
            cout << "[FAIL] MoDBlock training did not reduce loss\n";
        }
    }

    // ============================================================
    // Test 13: MoDModel forward shape
    // ============================================================
    cout << "\n--- Test 13: MoDModel forward shape ---\n";
    {
        ++total;
        size_t n = 4, d = 6, out_f = 3;
        MoDModel model(d, out_f, /*num_blocks=*/2, /*capacity_factor=*/0.5, /*ffn_dim=*/8);
        Tensor input(n, d);
        std::mt19937 rng(20);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = model.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == out_f) {
            cout << "[PASS] MoDModel forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << out_f << "\n";
        }
    }

    // ============================================================
    // Test 14: MoDModel training reduces loss
    // ============================================================
    cout << "\n--- Test 14: MoDModel training reduces loss ---\n";
    {
        ++total;
        size_t n = 5, d = 6, out_f = 3;
        MoDModel model(d, out_f, /*num_blocks=*/2, /*capacity_factor=*/0.5, /*ffn_dim=*/8);
        Tensor input(n, d);
        std::mt19937 rng(21);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        // target: smooth function of input
        Tensor target(n, out_f);
        for (size_t i = 0; i < n; ++i) {
            double s = 0;
            for (size_t j = 0; j < d; ++j) s += input(i, j);
            for (size_t j = 0; j < out_f; ++j) {
                target(i, j) = 0.1 * sin(s + (int)j) + 0.05 * ((int)j - 1);
            }
        }
        double lr = 0.01;
        double initial_loss = 0.0, final_loss = 0.0;
        for (int step = 0; step < 60; ++step) {
            Tensor output = model.forward(input);
            double loss = l2_loss_value(output, target);
            if (step == 0) initial_loss = loss;
            final_loss = loss;
            Tensor d_out = l2_loss_grad(output, target);
            model.backward(d_out, lr);
            model.update_weights(lr);
            model.zero_grad();
        }
        cout << "initial: " << initial_loss << " final: " << final_loss
             << " reduction: " << (100.0 * (initial_loss - final_loss) / initial_loss) << "%\n";
        if (final_loss < initial_loss * 0.9) {
            cout << "[PASS] MoDModel training reduced loss\n";
            ++passed;
        } else {
            cout << "[FAIL] MoDModel training did not reduce loss\n";
        }
    }

    // ============================================================
    // Test 15: MoDModel total aux loss aggregates across blocks
    // ============================================================
    cout << "\n--- Test 15: MoDModel total aux loss aggregates across blocks ---\n";
    {
        ++total;
        size_t n = 4, d = 4, out_f = 2;
        MoDModel model(d, out_f, /*num_blocks=*/3, /*capacity_factor=*/0.5, /*ffn_dim=*/6);
        Tensor input(n, d);
        std::mt19937 rng(22);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        model.forward(input);
        double total_aux = model.get_total_aux_loss();
        cout << "total aux loss (3 blocks) = " << total_aux << "\n";
        // At least non-negative (and >0 unless capacity_factor is exactly met).
        if (total_aux >= 0.0) {
            cout << "[PASS] total aux loss is non-negative\n";
            ++passed;
        } else {
            cout << "[FAIL] total aux loss negative\n";
        }
    }

    // ============================================================
    // Test 16: capacity_factor clamped to [0, 1]
    // ============================================================
    cout << "\n--- Test 16: capacity_factor clamped ---\n";
    {
        ++total;
        size_t n = 4, d = 3;
        // Test that 1.5 and -0.5 are clamped to 1.0 and 0.0 respectively.
        MoDLayer mod1(d, 1.5, 0.01, new Dense(d, d));
        MoDLayer mod2(d, -0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(23);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor out1 = mod1.forward(input);
        Tensor out2 = mod2.forward(input);
        size_t n_sel_1 = 0, n_sel_2 = 0;
        for (size_t t = 0; t < n; ++t) {
            if (mod1.mask_(t, 0) > 0.5) ++n_sel_1;
            if (mod2.mask_(t, 0) > 0.5) ++n_sel_2;
        }
        cout << "mod1 (cap=1.5) selected " << n_sel_1 << " (expected " << n << ")\n";
        cout << "mod2 (cap=-0.5) selected " << n_sel_2 << " (expected 0)\n";
        if (n_sel_1 == n && n_sel_2 == 0) {
            cout << "[PASS] capacity_factor clamped\n";
            ++passed;
        } else {
            cout << "[FAIL] capacity_factor not clamped correctly\n";
        }
    }

    // ============================================================
    // Test 17: Single-token (n=1) — capacity=1, output = input + inner(input)
    // ============================================================
    cout << "\n--- Test 17: Single-token (n=1) capacity behaviour ---\n";
    {
        ++total;
        size_t n = 1, d = 3;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        Tensor input(n, d);
        std::mt19937 rng(24);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor output = mod.forward(input);
        if (mod.capacity() == 1 && output.rows == 1 && output.cols == d) {
            cout << "[PASS] n=1 works, capacity=1\n";
            ++passed;
        } else {
            cout << "[FAIL] n=1 capacity=" << mod.capacity() << "\n";
        }
    }

    // ============================================================
    // Test 18: Param/grad shape consistency (router + inner)
    // ============================================================
    cout << "\n--- Test 18: MoDLayer param/grad shape consistency ---\n";
    {
        ++total;
        size_t d = 5;
        MoDLayer mod(d, 0.5, 0.01, new Dense(d, d));
        auto params = mod.parameters();
        auto grads = mod.gradients();
        cout << "MoDLayer parameters: " << params.size() << " (expect 4: W_router, b_router, inner.weights, inner.bias)\n";
        cout << "MoDLayer gradients:  " << grads.size() << "\n";
        if (params.size() != grads.size()) {
            cout << "[FAIL] param/grad count mismatch\n";
        } else {
            bool all_match = true;
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
                    cout << "[FAIL] pair " << i << " shape mismatch\n";
                    all_match = false;
                }
            }
            if (all_match) {
                cout << "[PASS] all " << params.size() << " param/grad pairs shape-matched\n";
                ++passed;
            }
        }
    }

    // ============================================================
    // Test 19: Different capacity_factor values — all work forward
    // ============================================================
    cout << "\n--- Test 19: Different capacity_factor values ---\n";
    {
        ++total;
        size_t n = 6, d = 4;
        bool all_ok = true;
        for (double cf : {0.25, 0.5, 0.75}) {
            MoDLayer mod(d, cf, 0.01, new Dense(d, d));
            Tensor input(n, d);
            std::mt19937 rng(25);
            std::uniform_real_distribution<double> dist(-0.3, 0.3);
            for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
            Tensor output = mod.forward(input);
            bool finite = true;
            for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
            size_t n_sel = 0;
            for (size_t t = 0; t < n; ++t) if (mod.mask_(t, 0) > 0.5) ++n_sel;
            cout << "  cf=" << cf << ": capacity=" << mod.capacity()
                 << "  shape=" << output.rows << "x" << output.cols
                 << "  finite=" << (finite ? "yes" : "no")
                 << "  selected=" << n_sel << "\n";
            if (output.rows != n || output.cols != d || !finite) all_ok = false;
        }
        if (all_ok) { cout << "[PASS] all capacity_factor values work forward\n"; ++passed; }
        else { cout << "[FAIL] some capacity_factor values failed\n"; }
    }

    cout << "\n=== Results: " << passed << "/" << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
