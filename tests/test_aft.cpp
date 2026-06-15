// AFT (Attention Free Transformer) — Zhai et al. 2021
//   "An Attention Free Transformer" (https://arxiv.org/abs/2105.14103)
//
// Tests:
//   1. AFTAttention forward shape: (n, d_model) -> (n, d_model)
//   2. AFTAttention output is finite (n=5, d_model=4)
//   3. AFTAttention input gradient check (n=3, d_model=2)
//   4. AFTAttention W_q weights gradient check
//   5. AFTAttention W_k weights gradient check
//   6. AFTAttention W_v weights gradient check
//   7. AFTAttention W_o weights gradient check
//   8. AFTAttention position_bias gradient check
//   9. AFTAttention training step reduces loss
//  10. AFTAttention parameters/gradients shape consistency
//  11. AFTBlock forward shape
//  12. AFTBlock output is finite (n=5, d_model=4)
//  13. AFTBlock input gradient check (n=3, d_model=2)
//  14. AFTBlock training step reduces loss
//  15. AFTModel forward shape
//  16. AFTModel training step reduces loss

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
    cout << "=== AFT (Attention Free Transformer) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small, tractable config: n=3, d_model=2, max_seq_len=4
    size_t n = 3, d_model = 2, max_seq_len = 4;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: AFTAttention forward shape (n=3, d_model=2) ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTAttention attn(d_model, max_seq_len);
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
    cout << "\n--- Test 2: AFTAttention output is finite (n=4) ---\n";
    {
        ++total;
        size_t n2 = 4;
        Tensor input(n2, d_model);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        AFTAttention attn(d_model, max_seq_len);
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
    cout << "\n--- Test 3: AFTAttention input gradient check (n=3) ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
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
    // Test 4: W_q weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: AFTAttention W_q weights gradient check ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // W_q is the FIRST (d_model, d_model) parameter
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, d_model, d_model, 0);
        if (!m.p) {
            cout << "[FAIL] could not find W_q\n";
        } else {
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
                        cout << "  W_q[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] W_q weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] W_q weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 5: W_k weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: AFTAttention W_k weights gradient check ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // W_k is the SECOND (d_model, d_model) parameter (after W_q)
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, d_model, d_model, 1);
        if (!m.p) {
            cout << "[FAIL] could not find W_k\n";
        } else {
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
                        cout << "  W_k[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] W_k weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] W_k weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_v weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: AFTAttention W_v weights gradient check ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // W_v is the THIRD (d_model, d_model) parameter (after W_q, W_k)
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, d_model, d_model, 2);
        if (!m.p) {
            cout << "[FAIL] could not find W_v\n";
        } else {
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
                        cout << "  W_v[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] W_v weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] W_v weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 7: W_o weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: AFTAttention W_o weights gradient check ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // W_o is the FOURTH (d_model, d_model) parameter (after W_q, W_k, W_v)
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, d_model, d_model, 3);
        if (!m.p) {
            cout << "[FAIL] could not find W_o\n";
        } else {
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
                        cout << "  W_o[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] W_o weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] W_o weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 8: position_bias gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: AFTAttention position_bias gradient check ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // position_bias is (max_seq_len, max_seq_len) = (4, 4)
        auto params = attn.parameters();
        auto grads  = attn.gradients();
        auto m = find_param(params, grads, max_seq_len, max_seq_len, 0);
        if (!m.p) {
            cout << "[FAIL] could not find position_bias\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < m.p->rows && n_checked < 6; ++i) {
                for (size_t j = 0; j < m.p->cols && n_checked < 6; ++j) {
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
                        cout << "  pb[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] position_bias gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] position_bias gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 9: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 9: AFTAttention training step reduces loss ---\n";
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

        AFTAttention attn(d_model, max_seq_len);
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
    cout << "\n--- Test 10: AFTAttention parameters/gradients shape consistency ---\n";
    {
        ++total;
        AFTAttention attn(d_model, max_seq_len);
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
                    cout << "  shape mismatch at index " << i << ": "
                         << params[i]->rows << "x" << params[i]->cols
                         << " vs " << grads[i]->rows << "x" << grads[i]->cols << "\n";
                    break;
                }
            }
            // Expected: 4 Denses * 2 (W, b) + 1 position_bias = 9 params.
            if (ok && params.size() == 9) {
                cout << "[PASS] all 9 param/grad pairs shape-matched\n";
                ++passed;
            } else {
                cout << "[FAIL] expected 9 params, got " << params.size() << "\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 11: AFTBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 11: AFTBlock forward shape (n=3, d_model=2) ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTBlock block(d_model, max_seq_len);
        Tensor output = block.forward(input);
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
    // Test 12: AFTBlock output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 12: AFTBlock output is finite (n=4) ---\n";
    {
        ++total;
        size_t n2 = 4;
        Tensor input(n2, d_model);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        AFTBlock block(d_model, max_seq_len);
        Tensor output = block.forward(input);
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
    // Test 13: AFTBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 13: AFTBlock input gradient check (n=3) ---\n";
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

        AFTBlock block(d_model, max_seq_len);
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
                if (err > 0.1) {
                    cout << "  x[" << i << "][" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] AFTBlock input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] AFTBlock input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: AFTBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 14: AFTBlock training step reduces loss ---\n";
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

        AFTBlock block(d_model, max_seq_len);
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

    // ------------------------------------------------------------
    // Test 15: AFTModel forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 15: AFTModel forward shape (n=3, d_model=2, out=1) ---\n";
    {
        ++total;
        size_t out_features = 1;
        size_t num_blocks = 2;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        AFTModel model(d_model, max_seq_len, out_features, num_blocks);
        Tensor output = model.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == n && output.cols == out_features) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << n << "x" << out_features << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: AFTModel training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 16: AFTModel training step reduces loss ---\n";
    {
        ++total;
        size_t out_features = 1;
        size_t num_blocks = 2;
        size_t n16 = 4;
        Tensor input(n16, d_model);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n16, out_features);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < out_features; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        AFTModel model(d_model, max_seq_len, out_features, num_blocks);
        Tensor out0 = model.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
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

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
