// H3 Block (Hungry Hungry Hippos) — Fu, Dao, Saab, Thomas, Rudra, Ré 2023
//   "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
//   https://arxiv.org/abs/2212.14052
//
// Tests:
//   1. Forward shape: (T, d_model) -> (T, d_model)
//   2. Output is finite
//   3. Shift SSM sliding-window property: K̄_t contains the last d K values
//   4. Diagonal SSM state propagation: Z_t = λ·Z_{t-1} + z_t
//   5. Input gradient check (analytical vs FD)
//   6. W_Q gradient check
//   7. W_K gradient check (tests the shift-SSM backward)
//   8. W_V gradient check (tests the diag-SSM backward through the outer product)
//   9. W_O gradient check (standard Dense backward)
//  10. λ_log gradient check (tests the diag-SSM λ gradient chain)
//  11. zero_grad clears all gradients
//  12. Training step reduces loss
//  13. Parameters()/gradients() shape consistency

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/h3.h"

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
    cout << "=== H3 (Hungry Hungry Hippos) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small config: T=4, d_model=2 (small enough for tractable grad checks)
    size_t T = 4, d_model = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: H3Block forward shape (T=4, d_model=2) ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        H3Block block(d_model);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: H3Block output is finite (T=5) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        H3Block block(d_model);
        Tensor output = block.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols && finite; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: Shift SSM sliding-window property
    //   We verify that K̄_t[i] = K_{t-i} when i ≤ t (the sliding-window property).
    //   We do this by:
    //     1) Running forward with a known input
    //     2) Computing K = input · W_K (the standard projection)
    //     3) Verifying last_K_bar_(t, i) == K(t - i, i) for i ≤ t
    // ------------------------------------------------------------
    cout << "\n--- Test 3: Shift SSM sliding-window property ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        // Use a distinctive input so we can verify the sliding-window property
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 + 0.1 * i + 0.2 * j;

        H3Block block(d_model);
        Tensor output = block.forward(input);  // populates last_K_bar_

        // Compute K = input · W_K.weights^T (the same projection K_t uses)
        Tensor K = block.W_K.forward(input);  // (T, d_model)

        bool ok = true;
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < d_model; ++i) {
                double expected;
                if (i <= t) {
                    expected = K(t - i, i);
                } else {
                    expected = 0.0;
                }
                double actual = block.last_K_bar_(t, i);
                if (fabs(expected - actual) > 1e-10) {
                    ok = false;
                    cout << "  K̄[" << t << "][" << i << "] expected " << expected
                         << ", got " << actual << "\n";
                }
            }
        }
        if (ok) {
            cout << "[PASS] sliding-window property holds\n";
            ++passed;
        } else {
            cout << "[FAIL] sliding-window property violated\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: Diagonal SSM state propagation
    //   We verify that Z_t[i,j] = λ[i,j] * Z_{t-1}[i,j] + z_t[i,j]
    //   using the cache.
    // ------------------------------------------------------------
    cout << "\n--- Test 4: Diagonal SSM state propagation ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.4 * (i + 1) - 0.3 * (j + 1);

        H3Block block(d_model);
        Tensor output = block.forward(input);  // populates last_Z_, last_lambda_, last_K_bar_, last_V_

        // Verify Z_t = λ * Z_{t-1} + z_t for t ≥ 1, Z_0 = z_0
        bool ok = true;
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < d_model; ++i) {
                for (size_t j = 0; j < d_model; ++j) {
                    double z_t = block.last_K_bar_(t, i) * block.last_V_(t, j);
                    double expected;
                    if (t == 0) {
                        expected = z_t;
                    } else {
                        double Z_prev = block.last_Z_((t - 1) * d_model + i, j);
                        expected = block.last_lambda_(i, j) * Z_prev + z_t;
                    }
                    double actual = block.last_Z_(t * d_model + i, j);
                    if (fabs(expected - actual) > 1e-10) {
                        ok = false;
                        cout << "  Z[" << t << "][" << i << "][" << j << "] expected "
                             << expected << ", got " << actual << "\n";
                    }
                }
            }
        }
        if (ok) {
            cout << "[PASS] diag-SSM state propagation correct\n";
            ++passed;
        } else {
            cout << "[FAIL] diag-SSM state propagation broken\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: H3Block input gradient check (T=4) ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < T; ++i) {
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
            cout << "[PASS] input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_Q gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: H3Block W_Q weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // W_Q.weights is (d_model, d_model). Find a representative entry.
        Tensor* Wp = &block.W_Q.weights;
        Tensor* Wg = &block.W_Q.grad_weights;
        double max_err = 0.0;
        for (size_t i = 0; i < Wp->rows; ++i) {
            for (size_t j = 0; j < Wp->cols; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Wg)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W_Q gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_Q gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: W_K gradient check (tests shift-SSM backward)
    // ------------------------------------------------------------
    cout << "\n--- Test 7: H3Block W_K weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        Tensor* Wp = &block.W_K.weights;
        Tensor* Wg = &block.W_K.grad_weights;
        double max_err = 0.0;
        for (size_t i = 0; i < Wp->rows; ++i) {
            for (size_t j = 0; j < Wp->cols; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Wg)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W_K gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_K gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: W_V gradient check (tests diag-SSM backward through outer product)
    // ------------------------------------------------------------
    cout << "\n--- Test 8: H3Block W_V weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        Tensor* Wp = &block.W_V.weights;
        Tensor* Wg = &block.W_V.grad_weights;
        double max_err = 0.0;
        for (size_t i = 0; i < Wp->rows; ++i) {
            for (size_t j = 0; j < Wp->cols; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Wg)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W_V gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_V gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: W_O gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: H3Block W_O weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        Tensor* Wp = &block.W_O.weights;
        Tensor* Wg = &block.W_O.grad_weights;
        double max_err = 0.0;
        for (size_t i = 0; i < Wp->rows; ++i) {
            for (size_t j = 0; j < Wp->cols; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Wg)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W_O gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_O gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: λ_log gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 10: H3Block lambda_log gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        H3Block block(d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        Tensor* Lp = &block.lambda_log;
        Tensor* Lg = &block.grad_lambda_log_;
        double max_err = 0.0;
        for (size_t i = 0; i < Lp->rows; ++i) {
            for (size_t j = 0; j < Lp->cols; ++j) {
                double orig = (*Lp)(i, j);
                (*Lp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp_v = l2_loss_value(out_p, target);
                (*Lp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm_v = l2_loss_value(out_m, target);
                (*Lp)(i, j) = orig;
                double num = (Lp_v - Lm_v) / (2.0 * eps);
                double ana = (*Lg)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] lambda_log gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] lambda_log gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: zero_grad clears all gradients
    // ------------------------------------------------------------
    cout << "\n--- Test 11: H3Block zero_grad clears all gradients ---\n";
    {
        ++total;
        H3Block block(d_model);
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(T, d_model);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.backward(grad_loss, 0.0);

        // Verify some gradients are non-zero before zero_grad
        double sum_before = 0.0;
        for (size_t i = 0; i < block.W_Q.grad_weights.data.size(); ++i)
            sum_before += fabs(block.W_Q.grad_weights.data[i]);
        for (size_t i = 0; i < block.grad_lambda_log_.data.size(); ++i)
            sum_before += fabs(block.grad_lambda_log_.data[i]);

        block.zero_grad();
        double sum_after = 0.0;
        for (size_t i = 0; i < block.W_Q.grad_weights.data.size(); ++i)
            sum_after += fabs(block.W_Q.grad_weights.data[i]);
        for (size_t i = 0; i < block.grad_lambda_log_.data.size(); ++i)
            sum_after += fabs(block.grad_lambda_log_.data[i]);

        cout << "Sum before zero_grad: " << sum_before
             << "  Sum after zero_grad: " << sum_after << "\n";
        if (sum_before > 1e-10 && sum_after < 1e-15) {
            cout << "[PASS] zero_grad clears all gradients\n";
            ++passed;
        } else {
            cout << "[FAIL] zero_grad did not clear all gradients\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 12: H3Block training step reduces loss ---\n";
    {
        ++total;
        size_t T2 = 4;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        H3Block block(d_model);
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 30; ++step) {
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
    // Test 13: parameters()/gradients() shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 13: H3Block parameters()/gradients() shape consistency ---\n";
    {
        ++total;
        H3Block block(d_model);
        auto params = block.parameters();
        auto grads  = block.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
        } else {
            // Expected: 4 Denses * 2 (weights+bias) + 1 (lambda_log) = 9 params
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
            if (ok && params.size() == 9) {
                cout << "[PASS] all " << params.size() << " params/grads shapes match\n";
                ++passed;
            } else {
                cout << "[FAIL] expected 9 params, got " << params.size() << "\n";
            }
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
