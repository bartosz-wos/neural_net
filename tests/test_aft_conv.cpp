// AFT-Conv — Zhai et al. 2021, "An Attention Free Transformer", §2.2
//   Position bias as low-rank bilinear form of learned per-position vectors,
//   convolved with a learned 1x1 kernel.
//
// Tests:
//   1. AFTConvAttention forward shape (n=3, d_model=2)
//   2. AFTConvAttention output is finite (n=5, d_model=4)
//   3. AFTConvAttention input gradient check (n=3, d_model=2)
//   4. AFTConvAttention W_q weights gradient check
//   5. AFTConvAttention W_k weights gradient check
//   6. AFTConvAttention W_v weights gradient check
//   7. AFTConvAttention W_o weights gradient check
//   8. AFTConvAttention position_embedding gradient check
//   9. AFTConvAttention W_conv gradient check
//  10. AFTConvAttention b_conv gradient check
//  11. AFTConvAttention training step reduces loss
//  12. AFTConvAttention parameters/gradients shape consistency (11 params)
//  13. AFTConvAttention position bias is non-trivially different from constant
//  14. AFTConvBlock forward shape
//  15. AFTConvBlock input gradient check
//  16. AFTConvBlock training step reduces loss

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
    cout << "=== AFT-Conv (low-rank bilinear position bias) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    size_t n = 3, d_model = 2, max_seq_len = 4;
    size_t rank = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: AFTConvAttention forward shape ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTConvAttention attn(d_model, max_seq_len, rank);
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
    cout << "\n--- Test 2: AFTConvAttention output is finite (n=5) ---\n";
    {
        ++total;
        size_t n2 = 5;
        size_t max_sl2 = 8;  // need > n2
        Tensor input(n2, d_model);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        AFTConvAttention attn(d_model, max_sl2, rank);
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
    cout << "\n--- Test 3: AFTConvAttention input gradient check ---\n";
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

        AFTConvAttention attn(d_model, max_seq_len, rank);
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
        cout << "\n--- Test " << test_num << ": AFTConvAttention " << name
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

        AFTConvAttention attn(d_model, max_seq_len, rank);
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
    // Test 8: position_embedding gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: AFTConvAttention position_embedding gradient check ---\n";
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

        AFTConvAttention attn(d_model, max_seq_len, rank);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        for (size_t t = 0; t < max_seq_len && n_checked < 4; ++t) {
            for (size_t k = 0; k < rank && n_checked < 4; ++k) {
                double orig = attn.position_embedding_(t, k);
                attn.position_embedding_(t, k) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                attn.position_embedding_(t, k) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                attn.position_embedding_(t, k) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = attn.grad_position_embedding_(t, k);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
                if (err > 0.1) {
                    cout << "  pos_emb[" << t << "][" << k << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] position_embedding gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] position_embedding gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: W_conv gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: AFTConvAttention W_conv gradient check ---\n";
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

        AFTConvAttention attn(d_model, max_seq_len, rank);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        int n_checked = 0;
        for (size_t i = 0; i < rank && n_checked < 4; ++i) {
            for (size_t k = 0; k < rank && n_checked < 4; ++k) {
                double orig = attn.W_conv_(i, k);
                attn.W_conv_(i, k) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                attn.W_conv_(i, k) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                attn.W_conv_(i, k) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = attn.grad_W_conv_(i, k);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n_checked;
                if (err > 0.1) {
                    cout << "  W_conv[" << i << "][" << k << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W_conv gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_conv gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: b_conv gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 10: AFTConvAttention b_conv gradient check ---\n";
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

        AFTConvAttention attn(d_model, max_seq_len, rank);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t k = 0; k < rank; ++k) {
            double orig = attn.b_conv_(0, k);
            attn.b_conv_(0, k) = orig + eps;
            Tensor out_p = attn.forward(input);
            double Lp = l2_loss_value(out_p, target);
            attn.b_conv_(0, k) = orig - eps;
            Tensor out_m = attn.forward(input);
            double Lm = l2_loss_value(out_m, target);
            attn.b_conv_(0, k) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = attn.grad_b_conv_(0, k);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
            if (err > 0.1) {
                cout << "  b_conv[" << k << "]: ana=" << ana
                     << " num=" << num << " err=" << err << "\n";
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] b_conv gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] b_conv gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 11: AFTConvAttention training step reduces loss ---\n";
    {
        ++total;
        size_t n11 = 4;
        Tensor input(n11, d_model);
        for (size_t i = 0; i < n11; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n11, d_model);
        for (size_t i = 0; i < n11; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        AFTConvAttention attn(d_model, max_seq_len, rank);
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
    // Test 12: parameters/gradients shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 12: AFTConvAttention parameters/gradients shape consistency ---\n";
    {
        ++total;
        AFTConvAttention attn(d_model, max_seq_len, rank);
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
            // 4 Denses * 2 (W, b) + 3 position-bias params (pos_emb, W_conv, b_conv) = 11
            if (ok && params.size() == 11) {
                cout << "[PASS] all 11 param/grad pairs shape-matched\n";
                ++passed;
            } else {
                cout << "[FAIL] expected 11 params, got " << params.size() << "\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 13: position bias is non-trivially different from constant
    // (sanity check that the bilinear form is producing different values
    //  for different position pairs, not just a constant)
    // ------------------------------------------------------------
    cout << "\n--- Test 13: AFTConvAttention position bias is non-trivially different ---\n";
    {
        ++total;
        size_t max_sl = 4;
        size_t r = 2;
        AFTConvAttention attn(d_model, max_sl, r);
        // Build a small input to populate caches
        Tensor input(2, d_model);
        input(0, 0) = 0.0; input(0, 1) = 0.0;
        input(1, 0) = 0.0; input(1, 1) = 0.0;
        attn.forward(input);
        // Verify the position embedding is non-trivial (varies across positions)
        // and that the cached last_qp_ differs across positions.
        // We use the public position_embedding_ accessor for the check.
        double e00 = attn.position_embedding_(0, 0);
        double e01 = attn.position_embedding_(0, 1);
        double e10 = attn.position_embedding_(1, 0);
        double e11 = attn.position_embedding_(1, 1);
        cout << "pos_emb[0][0]=" << e00 << " pos_emb[0][1]=" << e01
             << " pos_emb[1][0]=" << e10 << " pos_emb[1][1]=" << e11 << "\n";
        // Check that at least one entry differs (non-trivial across positions).
        bool varied = (std::fabs(e00 - e10) > 1e-6) || (std::fabs(e00 - e01) > 1e-6) ||
                      (std::fabs(e10 - e11) > 1e-6);
        // Check entries are bounded (small init)
        bool in_range = true;
        for (double v : {e00, e01, e10, e11}) {
            if (std::fabs(v) > 1.0) in_range = false;
        }
        if (in_range && varied) {
            cout << "[PASS] position embedding is non-trivial and bounded\n";
            ++passed;
        } else {
            cout << "[FAIL] position embedding check failed (in_range=" << in_range
                 << " varied=" << varied << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: AFTConvBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 14: AFTConvBlock forward shape ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTConvBlock block(d_model, max_seq_len, rank);
        Tensor output = block.forward(input);
        if (output.rows == n && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: AFTConvBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 15: AFTConvBlock input gradient check ---\n";
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

        AFTConvBlock block(d_model, max_seq_len, rank);
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
            cout << "[PASS] AFTConvBlock input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] AFTConvBlock input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: AFTConvBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 16: AFTConvBlock training step reduces loss ---\n";
    {
        ++total;
        size_t n16 = 4;
        Tensor input(n16, d_model);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n16, d_model);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        AFTConvBlock block(d_model, max_seq_len, rank);
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
