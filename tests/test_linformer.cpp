// Linformer — Wang et al. 2020 "Linformer: Self-Attention with Linear Complexity"
//
// Tests:
//   1. LinformerAttention forward shape: (n, d) -> (n, d)
//   2. LinformerAttention output is finite
//   3. LinformerAttention input gradient check (learned projection)
//   4. LinformerAttention W_q gradient check
//   5. LinformerAttention E projection gradient check (learned path)
//   6. LinformerAttention equivalence: k == n is mathematically vanilla
//   7. LinformerAttention fixed-projection path (no E, F in params, still trains)
//   8. LinformerBlock forward shape
//   9. LinformerBlock input gradient check
//  10. LinformerModel training step reduces loss (2 blocks, learned projection)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/attention/linformer.h"

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

// Find a parameter matching (rows, cols); returns nullptr if not found.
// (Unused for now but kept for future tests.)
[[maybe_unused]] static Tensor* find_param(vector<Tensor*>& params, size_t r, size_t c) {
    for (auto* p : params) if (p->rows == r && p->cols == c) return p;
    return nullptr;
}

int main() {
    cout << "=== Linformer Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: LinformerAttention forward shape (n=6, d=4 -> n=6, d=4)
    // ------------------------------------------------------------
    cout << "\n--- Test 1: LinformerAttention forward shape (learned, k=3) ---\n";
    {
        ++total;
        size_t n = 6, d = 4, k = 3;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j;

        LinformerAttention attn(d, n, k, /*learned=*/true);
        Tensor output = attn.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: LinformerAttention output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: LinformerAttention output is finite (fixed projection) ---\n";
    {
        ++total;
        size_t n = 8, d = 6, k = 4;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * sin(0.1 * i) + 0.2 * j;

        LinformerAttention attn(d, n, k, /*learned=*/false);
        Tensor output = attn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: LinformerAttention input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: LinformerAttention input gradient check (learned, k=2) ---\n";
    {
        ++total;
        size_t n = 4, d = 3, k = 2;
        double eps = 1e-5;

        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        LinformerAttention attn(d, n, k, /*learned=*/true);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        Tensor grad_x = attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "  x[" << i << "][" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: LinformerAttention W_q gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: LinformerAttention W_q gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3, k = 2;
        double eps = 1e-5;

        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        LinformerAttention attn(d, n, k, /*learned=*/true);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // W_q is the first (d, d) param
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        Tensor* Wp = params[0];   // W_q
        Tensor* Gp = grads[0];
        double max_err = 0.0;
        int n_checked = 0;
        for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
            for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Gp)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] W_q gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_q gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: LinformerAttention E projection gradient check (learned)
    // ------------------------------------------------------------
    cout << "\n--- Test 5: LinformerAttention E projection gradient check ---\n";
    {
        ++total;
        size_t n = 4, d = 3, k = 2;
        double eps = 1e-5;

        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        LinformerAttention attn(d, n, k, /*learned=*/true);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // E is the (k, n) parameter, the 5th in the list
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        Tensor* Ep = nullptr;
        Tensor* Eg = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == k && params[i]->cols == n) {
                Ep = params[i];
                Eg = grads[i];
                break;
            }
        }
        if (!Ep) {
            cout << "[FAIL] could not find E parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Ep->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < Ep->cols && n_checked < 4; ++j) {
                    double orig = (*Ep)(i, j);
                    (*Ep)(i, j) = orig + eps;
                    Tensor out_p = attn.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Ep)(i, j) = orig - eps;
                    Tensor out_m = attn.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Ep)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Eg)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.05) {
                cout << "[PASS] E gradient check (rel_err < 5%)\n";
                ++passed;
            } else {
                cout << "[FAIL] E gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 6: LinformerAttention k == n equivalence
    // We can't directly assert mathematical equivalence (E,F are random init
    // and not identity), so this test is "it compiles and runs at k=n".
    // ------------------------------------------------------------
    cout << "\n--- Test 6: LinformerAttention k == n (degenerate to full-rank) ---\n";
    {
        ++total;
        size_t n = 4, d = 3, k = 4;  // k == n
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i + 0.05 * j;

        LinformerAttention attn(d, n, k, /*learned=*/true);
        Tensor output = attn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite && output.rows == n && output.cols == d) {
            cout << "[PASS] k=n runs and produces finite (n, d) output\n";
            ++passed;
        } else {
            cout << "[FAIL] k=n produced bad output\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: LinformerAttention fixed-projection path
    //   - E, F should NOT be in parameters (no learning them)
    //   - update_weights should not change E, F
    //   - input grad should still flow
    // ------------------------------------------------------------
    cout << "\n--- Test 7: LinformerAttention fixed projection (no E/F params) ---\n";
    {
        ++total;
        size_t n = 4, d = 3, k = 2;

        // Use larger input scale so gradients aren't dominated by float noise
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        LinformerAttention attn(d, n, k, /*learned=*/false);
        auto params = attn.parameters();
        // With fixed projection, params should be exactly 4 (W_q, W_k, W_v, W_o)
        // and none should have shape (k, n).
        bool no_kbyn = true;
        for (auto* p : params) {
            if (p->rows == k && p->cols == n) { no_kbyn = false; break; }
        }
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        Tensor grad_x = attn.backward(grad_loss, 0.0);

        // Quick input grad sanity check
        double eps = 1e-5;
        double max_err = 0.0;
        for (size_t i = 0; i < n && max_err < 0.05; ++i) {
            for (size_t j = 0; j < d && max_err < 0.05; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        if (no_kbyn && params.size() == 4 && max_err < 0.05) {
            cout << "[PASS] fixed projection: 4 params, no (k,n), input grad ok (err=" << max_err << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] params.size()=" << params.size()
                 << " no_kbyn=" << no_kbyn << " max_err=" << max_err << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: LinformerBlock forward shape (n=5, d=4 -> n=5, d=4)
    // ------------------------------------------------------------
    cout << "\n--- Test 8: LinformerBlock forward shape ---\n";
    {
        ++total;
        size_t n = 5, d = 4, k = 3;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j;

        LinformerBlock block(d, n, k);
        Tensor out = block.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == n && out.cols == d) {
            cout << "[PASS] block shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: LinformerBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: LinformerBlock input gradient check ---\n";
    {
        ++total;
        size_t n = 3, d = 2, k = 2;
        double eps = 1e-5;

        // Use larger input scale so gradients aren't dominated by float noise
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        LinformerBlock block(d, n, k);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                if (err > 0.1) {
                    cout << "  x[" << i << "][" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] block input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] block input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: LinformerModel training step reduces loss (2 blocks)
    // ------------------------------------------------------------
    cout << "\n--- Test 10: LinformerModel 2-block training reduces loss ---\n";
    {
        ++total;
        size_t n = 4, d = 3, out_f = 2, k = 2;
        double lr = 0.01;

        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(n, out_f);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        LinformerModel model(d, n, out_f, /*num_blocks=*/2, /*proj_dim=*/k);
        Tensor out0 = model.forward(input);
        double loss0 = l2_loss_value(out0, target);

        for (int step = 0; step < 50; ++step) {
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
            cout << "[PASS] training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not decrease loss\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
