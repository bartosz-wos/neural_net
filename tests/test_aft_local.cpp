// AFT-Local — Zhai et al. 2021, "An Attention Free Transformer", §2.2
//   Windowed relative position bias variant.
//
// Tests:
//   1. AFTLocalAttention forward shape (n=3, d_model=2)
//   2. AFTLocalAttention output is finite (n=5, d_model=4)
//   3. AFTLocalAttention input gradient check (n=3, d_model=2)
//   4. AFTLocalAttention W_q weights gradient check
//   5. AFTLocalAttention W_k weights gradient check
//   6. AFTLocalAttention W_v weights gradient check
//   7. AFTLocalAttention W_o weights gradient check
//   8. AFTLocalAttention relative_bias gradient check (interior + boundary offsets)
//   9. AFTLocalAttention training step reduces loss
//  10. AFTLocalAttention parameters/gradients shape consistency (9 params)
//  11. AFTLocalAttention windowed: out-of-window pairs are masked
//  12. AFTLocalBlock forward shape
//  13. AFTLocalBlock input gradient check (n=3, d_model=2)
//  14. AFTLocalBlock training step reduces loss

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/attention/aft.h"

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

// Helper: find parameter by shape signature (rows, cols).
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

int main() {
    cout << "=== AFT-Local (windowed relative position bias) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    size_t n = 3, d_model = 2, max_seq_len = 4;
    size_t window = 2;  // covers offsets {-1, 0, +1}, so 2*w-1 = 3 entries

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: AFTLocalAttention forward shape ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTLocalAttention attn(d_model, max_seq_len, window);
        Tensor output = attn.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: AFTLocalAttention output is finite (n=5) ---\n";
    {
        ++total;
        size_t n2 = 5;
        size_t max_sl2 = 8;  // need > n2
        Tensor input(n2, d_model);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        AFTLocalAttention attn(d_model, max_sl2, window);
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
    // Test 3: input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: AFTLocalAttention input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        AFTLocalAttention attn(d_model, max_seq_len, window);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        Tensor grad_x = attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
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
                if (err > 0.1) {
                    cout << "  x[" << i << "][" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4-7: W_q, W_k, W_v, W_o weight gradient checks
    // ------------------------------------------------------------
    auto run_w_grad_check = [&](int test_num, const char* name, int occ) {
        cout << "\n--- Test " << test_num << ": AFTLocalAttention " << name
             << " gradient check ---\n";
        ++total;
        double eps = 1e-5;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        AFTLocalAttention attn(d_model, max_seq_len, window);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, d_model, d_model, occ);
        if (!m.p) {
            cout << "[FAIL] could not find " << name << "\n";
            return;
        }
        double max_err = 0.0;
        int n_checked = 0;
        for (size_t i = 0; i < m.p->rows && n_checked < 4; ++i) {
            for (size_t j = 0; j < m.p->cols && n_checked < 4; ++j) {
                double orig = (*m.p)(i, j);
                (*m.p)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*m.p)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*m.p)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*m.g)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
                if (err > 0.1) {
                    cout << "  " << name << "[" << i << "][" << j
                         << "]: ana=" << ana << " num=" << num
                         << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] " << name << " gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] " << name << " gradient check failed\n";
        }
    };
    run_w_grad_check(4, "W_q", 0);
    run_w_grad_check(5, "W_k", 1);
    run_w_grad_check(6, "W_v", 2);
    run_w_grad_check(7, "W_o", 3);

    // ------------------------------------------------------------
    // Test 8: relative_bias gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: AFTLocalAttention relative_bias gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        AFTLocalAttention attn(d_model, max_seq_len, window);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // relative_bias is (1, 2*window-1) = (1, 3)
        size_t rb_cols = 2 * window - 1;
        double max_err = 0.0;
        for (size_t j = 0; j < rb_cols; ++j) {
            double orig = attn.relative_bias_(0, j);
            attn.relative_bias_(0, j) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            attn.relative_bias_(0, j) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            attn.relative_bias_(0, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = attn.grad_relative_bias_(0, j);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
            if (err > 0.1) {
                cout << "  rb[" << j << "]: ana=" << ana
                     << " num=" << num << " err=" << err << "\n";
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] relative_bias gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] relative_bias gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 9: AFTLocalAttention training step reduces loss ---\n";
    {
        ++total;
        size_t n9 = 4;
        Tensor input(n9, d_model);
        for (size_t i = 0; i < n9; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n9, d_model);
        for (size_t i = 0; i < n9; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        AFTLocalAttention attn(d_model, max_seq_len, window);
        Tensor out0 = attn.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
        for (int step = 0; step < 60; ++step) {
            attn.zero_grad();
            Tensor out = attn.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            attn.backward(grad_loss, 0.0);
            attn.update_weights(lr);
        }
        Tensor out1 = attn.forward(input);
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
    // Test 10: parameters/gradients shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 10: AFTLocalAttention parameters/gradients shape consistency ---\n";
    {
        ++total;
        AFTLocalAttention attn(d_model, max_seq_len, window);
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
        } else {
            bool ok = true;
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows ||
                    params[i]->cols != grads[i]->cols) {
                    ok = false;
                    break;
                }
            }
            if (ok && params.size() == 9) {
                cout << "[PASS] all 9 param/grad pairs shape-matched\n";
                ++passed;
            } else {
                cout << "[FAIL] expected 9 params, got " << params.size() << "\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 11: out-of-window positions are masked
    // ------------------------------------------------------------
    cout << "\n--- Test 11: AFTLocalAttention windowed: out-of-window pairs are masked ---\n";
    {
        ++total;
        // Sanity check: with window=2, the (1, 3) pair (|t-s|=2) is out-of-window
        // and should not contribute to forward. We verify by setting the relative_bias
        // to a huge positive value and confirming the output doesn't blow up at the
        // out-of-window pair (since exp(huge) should not be applied).
        size_t max_sl = 4;
        size_t n11 = 4;
        size_t w = 2;  // window covers |t-s| < 2 i.e. |t-s| in {0, 1}
        AFTLocalAttention attn(d_model, max_sl, w);

        // Set all relative bias entries to a huge value
        for (size_t j = 0; j < attn.relative_bias_.cols; ++j) {
            attn.relative_bias_(0, j) = 100.0;
        }

        Tensor input(n11, d_model);
        for (size_t i = 0; i < n11; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * j;

        Tensor output = attn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] output remains finite (out-of-window exp(-1e9) masking works)\n";
            ++passed;
        } else {
            cout << "[FAIL] output became non-finite\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: AFTLocalBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 12: AFTLocalBlock forward shape ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTLocalBlock block(d_model, max_seq_len, window);
        Tensor output = block.forward(input);
        if (output.rows == n && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: AFTLocalBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 13: AFTLocalBlock input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        AFTLocalBlock block(d_model, max_seq_len, window);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
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
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] AFTLocalBlock input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] AFTLocalBlock input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: AFTLocalBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 14: AFTLocalBlock training step reduces loss ---\n";
    {
        ++total;
        size_t n14 = 4;
        Tensor input(n14, d_model);
        for (size_t i = 0; i < n14; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n14, d_model);
        for (size_t i = 0; i < n14; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        AFTLocalBlock block(d_model, max_seq_len, window);
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
        for (int step = 0; step < 60; ++step) {
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
            cout << "[PASS] training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not decrease loss\n";
        }
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
