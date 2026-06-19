// MultiHead-Conv Attention — Yang et al. 2023, "Convolutional Self-Attention Networks"
//
// Tests:
//   1. ConvAttention forward shape (single head, n=3, d=2)
//   2. ConvAttention output finite
//   3. ConvAttention input gradient check (single head, n=3, d=2, k=3)
//   4. ConvAttention Wq conv-weight gradient check
//   5. ConvAttention Wk conv-weight gradient check
//   6. ConvAttention Wv conv-weight gradient check
//   7. ConvAttention W_o output projection gradient check
//   8. ConvAttention bq / bk / bv bias gradient check
//   9. ConvAttention training step reduces loss
//  10. ConvAttention parameters/gradients shape consistency
//  11. Multi-head (num_heads=2, d=4) input gradient check
//  12. ConvAttentionBlock forward shape
//  13. ConvAttentionBlock input gradient check
//  14. ConvAttentionBlock training step reduces loss
//  15. ConvAttentionModel forward shape
//  16. ConvAttentionModel training step reduces loss
//  17. Different kernel sizes (k=5) sanity check

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/attention/conv_attention.h"

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

int main() {
    cout << "=== MultiHead-Conv Attention (Yang 2023) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Tractable config: n=3, d_model=2, single head, kernel=3
    size_t n = 3, d_model = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: ConvAttention forward shape (n=3, d=2) ---\n";
    {
        ++total;
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        ConvAttention attn(d_model, n, /*num_heads=*/1, /*kernel_size=*/3);
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
    // Test 2: output finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: ConvAttention output finite (n=4) ---\n";
    {
        ++total;
        size_t n2 = 4;
        Tensor input(n2, d_model);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        ConvAttention attn(d_model, n2, 1, 3);
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
    cout << "\n--- Test 3: ConvAttention input gradient check (n=3, d=2) ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
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
    // Test 4: Wq conv-weight gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: ConvAttention Wq gradient check ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        // Find Wq_w_ in grad list
        auto params = attn.parameters();
        auto grads = attn.gradients();
        Tensor* Wq_w = nullptr;
        Tensor* grad_Wq_w = nullptr;
        size_t ksz = 3;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_model && params[i]->cols == d_model * ksz) {
                Wq_w = params[i];
                grad_Wq_w = grads[i];
                break;
            }
        }
        if (!Wq_w || !grad_Wq_w) {
            cout << "[FAIL] could not find Wq_w_ in parameters\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < Wq_w->rows; ++i) {
                for (size_t j = 0; j < Wq_w->cols; ++j) {
                    double orig = (*Wq_w)(i, j);
                    (*Wq_w)(i, j) = orig + eps;
                    Tensor out_p = attn.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wq_w)(i, j) = orig - eps;
                    Tensor out_m = attn.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wq_w)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*grad_Wq_w)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] Wq gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] Wq gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 5: Wk conv-weight gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: ConvAttention Wk gradient check ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        Tensor* Wk_w = nullptr;
        Tensor* grad_Wk_w = nullptr;
        size_t ksz = 3;
        int seen = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_model && params[i]->cols == d_model * ksz) {
                if (seen == 1) {
                    Wk_w = params[i];
                    grad_Wk_w = grads[i];
                    break;
                }
                seen++;
            }
        }
        if (!Wk_w) {
            cout << "[FAIL] could not find Wk_w_ in parameters\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < Wk_w->rows; ++i) {
                for (size_t j = 0; j < Wk_w->cols; ++j) {
                    double orig = (*Wk_w)(i, j);
                    (*Wk_w)(i, j) = orig + eps;
                    Tensor out_p = attn.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wk_w)(i, j) = orig - eps;
                    Tensor out_m = attn.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wk_w)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*grad_Wk_w)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] Wk gradient check\n";
                ++passed;
            } else {
                cout << "[FAIL] Wk gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 6: Wv conv-weight gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: ConvAttention Wv gradient check ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        Tensor* Wv_w = nullptr;
        Tensor* grad_Wv_w = nullptr;
        size_t ksz = 3;
        int seen = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_model && params[i]->cols == d_model * ksz) {
                if (seen == 2) {
                    Wv_w = params[i];
                    grad_Wv_w = grads[i];
                    break;
                }
                seen++;
            }
        }
        if (!Wv_w) {
            cout << "[FAIL] could not find Wv_w_ in parameters\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < Wv_w->rows; ++i) {
                for (size_t j = 0; j < Wv_w->cols; ++j) {
                    double orig = (*Wv_w)(i, j);
                    (*Wv_w)(i, j) = orig + eps;
                    Tensor out_p = attn.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wv_w)(i, j) = orig - eps;
                    Tensor out_m = attn.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wv_w)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*grad_Wv_w)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] Wv gradient check\n";
                ++passed;
            } else {
                cout << "[FAIL] Wv gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 7: W_o output projection gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: ConvAttention W_o gradient check ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        Tensor* Wo_w = nullptr;
        Tensor* grad_Wo_w = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_model && params[i]->cols == d_model) {
                // W_o is the only (d, d) tensor in params.
                // (The convs are (d, d*k).) — but our (d, d) tensor could
                // also be the W_o weights inside Dense. We need to find
                // the right one.  W_o is index 6 in our parameter list
                // (after Wq, Wk, Wv weights and biases).
                // Simplest: pick the LAST (d, d) tensor.
                Wo_w = params[i];
                grad_Wo_w = grads[i];
            }
        }
        if (!Wo_w) {
            cout << "[FAIL] could not find W_o in parameters\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < Wo_w->rows; ++i) {
                for (size_t j = 0; j < Wo_w->cols; ++j) {
                    double orig = (*Wo_w)(i, j);
                    (*Wo_w)(i, j) = orig + eps;
                    Tensor out_p = attn.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wo_w)(i, j) = orig - eps;
                    Tensor out_m = attn.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wo_w)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*grad_Wo_w)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] W_o gradient check\n";
                ++passed;
            } else {
                cout << "[FAIL] W_o gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 8: bias gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: ConvAttention bias gradient check (Wq_b) ---\n";
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

        ConvAttention attn(d_model, n, 1, 3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        Tensor* Wq_b = nullptr;
        Tensor* grad_Wq_b = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_model && params[i]->cols == 1) {
                Wq_b = params[i];
                grad_Wq_b = grads[i];
                break;
            }
        }
        if (!Wq_b) {
            cout << "[FAIL] could not find bias tensor\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < Wq_b->rows; ++i) {
                double orig = (*Wq_b)(i, 0);
                (*Wq_b)(i, 0) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wq_b)(i, 0) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wq_b)(i, 0) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*grad_Wq_b)(i, 0);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] Wq_b gradient check\n";
                ++passed;
            } else {
                cout << "[FAIL] Wq_b gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 9: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 9: ConvAttention training step reduces loss ---\n";
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

        ConvAttention attn(d_model, n9, 1, 3);
        Tensor out0 = attn.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
        for (int step = 0; step < 80; ++step) {
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
            cout << "[FAIL] loss did not decrease\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: param/grad shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 10: param/grad shape consistency ---\n";
    {
        ++total;
        ConvAttention attn(d_model, n, 2, 3);
        attn.zero_grad();
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i + 0.2 * j;
        Tensor target(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.05 * i;
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        bool ok = (params.size() == grads.size());
        for (size_t i = 0; i < params.size() && ok; ++i) {
            if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
                ok = false;
            }
        }
        if (ok) {
            cout << "[PASS] all param/grad shapes match (" << params.size() << " pairs)\n";
            ++passed;
        } else {
            cout << "[FAIL] param/grad shape mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: multi-head input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 11: Multi-head (H=2, d=4) input gradient check ---\n";
    {
        ++total;
        size_t n11 = 3, d11 = 4;
        double eps = 1e-5;
        Tensor input(n11, d11);
        for (size_t i = 0; i < n11; ++i)
            for (size_t j = 0; j < d11; ++j)
                input(i, j) = 0.1 * i + 0.2 * j - 0.05;

        Tensor target(n11, d11);
        for (size_t i = 0; i < n11; ++i)
            for (size_t j = 0; j < d11; ++j)
                target(i, j) = 0.05 * i + 0.1 * j;

        ConvAttention attn(d11, n11, /*num_heads=*/2, /*kernel_size=*/3);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        Tensor grad_x = attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < n11; ++i) {
            for (size_t j = 0; j < d11; ++j) {
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
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] multi-head input gradient check\n";
            ++passed;
        } else {
            cout << "[FAIL] multi-head input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: ConvAttentionBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 12: ConvAttentionBlock forward shape ---\n";
    {
        ++total;
        ConvAttentionBlock block(d_model, n, 1, 3);
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i + 0.2 * j;
        Tensor out = block.forward(input);
        if (out.rows == n && out.cols == d_model) {
            cout << "[PASS] block forward shape\n";
            ++passed;
        } else {
            cout << "[FAIL] block forward shape (got " << out.rows << "x" << out.cols << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: ConvAttentionBlock input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 13: ConvAttentionBlock input gradient check ---\n";
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

        ConvAttentionBlock block(d_model, n, 1, 3);
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
            cout << "[PASS] block input gradient check\n";
            ++passed;
        } else {
            cout << "[FAIL] block input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: ConvAttentionBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 14: ConvAttentionBlock training step reduces loss ---\n";
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

        ConvAttentionBlock block(d_model, n14, 1, 3);
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 100; ++step) {
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
            cout << "[PASS] block training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] block loss did not decrease\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: ConvAttentionModel forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 15: ConvAttentionModel forward shape ---\n";
    {
        ++total;
        size_t out_features = 2;
        ConvAttentionModel model(d_model, n, out_features, /*num_blocks=*/2, 1, 3);
        Tensor input(n, d_model);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i + 0.2 * j;
        Tensor out = model.forward(input);
        if (out.rows == n && out.cols == out_features) {
            cout << "[PASS] model forward shape\n";
            ++passed;
        } else {
            cout << "[FAIL] model forward shape (got " << out.rows << "x" << out.cols << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: ConvAttentionModel training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 16: ConvAttentionModel training step reduces loss ---\n";
    {
        ++total;
        size_t n16 = 4, out_features = 2;
        Tensor input(n16, d_model);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(n16, out_features);
        for (size_t i = 0; i < n16; ++i)
            for (size_t j = 0; j < out_features; ++j)
                target(i, j) = 0.1 * (i + j);

        ConvAttentionModel model(d_model, n16, out_features, /*num_blocks=*/2, 1, 3);
        Tensor out0 = model.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 100; ++step) {
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
            cout << "[PASS] model training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] model loss did not decrease\n";
        }
    }

    // ------------------------------------------------------------
    // Test 17: kernel_size=5 sanity check
    // ------------------------------------------------------------
    cout << "\n--- Test 17: kernel_size=5 sanity check ---\n";
    {
        ++total;
        size_t n17 = 5;
        Tensor input(n17, d_model);
        for (size_t i = 0; i < n17; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * sin(0.3 * i) - 0.1 * j;

        ConvAttention attn(d_model, n17, 1, /*kernel_size=*/5);
        Tensor out = attn.forward(input);
        bool finite = (out.rows == n17) && (out.cols == d_model);
        for (size_t i = 0; i < out.rows && finite; ++i) {
            for (size_t j = 0; j < out.cols; ++j) {
                if (!std::isfinite(out(i, j))) finite = false;
            }
        }
        if (finite) {
            cout << "[PASS] kernel=5 forward ok\n";
            ++passed;
        } else {
            cout << "[FAIL] kernel=5 forward broken\n";
        }
    }

    // ------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------
    cout << "\n=== ConvAttention: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
