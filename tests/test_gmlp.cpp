// gMLP (gated MLP) tests — Liu et al. 2021, "Pay Attention to MLPs"
//
// Tests:
//   1. gMLPBlock forward shape (S, d) -> (S, d)
//   2. gMLPBlock output is finite
//   3. gMLPBlock numerical gradient check on input
//   4. gMLPBlock numerical gradient check on W_spatial
//   5. gMLPBlock numerical gradient check on fc_in weights
//   6. gMLPBlock numerical gradient check on alpha (residual scalar)
//   7. gMLPBlock zero_grad() clears all gradients
//   8. gMLPModel forward shape (S, d) -> (S, out_features)
//   9. gMLPModel numerical gradient check on input
//  10. gMLPModel training step reduces loss
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/architectures/gmlp.h"

using namespace std;

static double relative_error(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
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
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

int main() {
    cout << "=== gMLP Tests ===" << endl;
    cout.setf(std::ios::unitbuf);  // flush after each write so we see partial output on crash
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: gMLPBlock forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: gMLPBlock forward shape (S=6, d=4 -> S=6, d=4) ---\n";
    {
        ++total;
        size_t S = 6, d = 4;
        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j;

        gMLPBlock block(d, S);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == S && output.cols == d) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << S << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: gMLPBlock output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: gMLPBlock output is finite ---\n";
    {
        ++total;
        size_t S = 8, d = 6;
        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * sin(0.1 * i) + 0.2 * j;

        gMLPBlock block(d, S);
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
    // Test 3: gMLPBlock input gradient check (L2 loss)
    // ------------------------------------------------------------
    cout << "\n--- Test 3: gMLPBlock input gradient check (L2) ---\n";
    {
        ++total;
        size_t S = 4, d = 3;
        double eps = 1e-5;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPBlock block(d, S);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < S; ++i) {
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
    // Test 4: gMLPBlock W_spatial gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: gMLPBlock W_spatial gradient check (L2) ---\n";
    {
        ++total;
        size_t S = 4, d = 3;
        double eps = 1e-5;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPBlock block(d, S);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // W_spatial is the 5th param (after fc_in W, fc_in b, fc_out W, fc_out b,
        // LN gamma, LN beta = 6, then W_spatial = 7th). Let's just find it.
        Tensor* Wp = nullptr;
        Tensor* Gp = nullptr;
        auto params = block.parameters();
        auto grads  = block.gradients();
        for (size_t k = 0; k < params.size(); ++k) {
            if (params[k]->rows == S && params[k]->cols == S) {
                Wp = params[k];
                Gp = grads[k];
                break;
            }
        }
        if (!Wp) {
            cout << "[FAIL] could not find W_spatial in parameters\n";
        } else {
            double max_err = 0.0;
            int n = 0;
            // spot-check a few entries (S*S may be small here, check all)
            for (size_t i = 0; i < S && n < 6; ++i) {
                for (size_t j = 0; j < S && n < 6; ++j) {
                    double orig = (*Wp)(i, j);
                    (*Wp)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wp)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wp)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Gp)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.05) {
                cout << "[PASS] W_spatial gradient check (rel_err < 5%)\n";
                ++passed;
            } else {
                cout << "[FAIL] W_spatial gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 5: gMLPBlock fc_in weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: gMLPBlock fc_in W gradient check (L2) ---\n";
    {
        ++total;
        size_t S = 3, d = 2;
        double eps = 1e-5;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPBlock block(d, S);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // First param is fc_in.weights (2d, d).
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Wp = params[0];
        Tensor* Gp = grads[0];
        double max_err = 0.0;
        int n = 0;
        for (size_t i = 0; i < Wp->rows && n < 4; ++i) {
            for (size_t j = 0; j < Wp->cols && n < 4; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Gp)(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                ++n;
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) {
            cout << "[PASS] fc_in W gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] fc_in W gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: gMLPBlock alpha (residual scalar) gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: gMLPBlock alpha gradient check (L2) ---\n";
    {
        ++total;
        size_t S = 3, d = 2;
        double eps = 1e-5;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPBlock block(d, S, /*alpha_init=*/0.5);  // bigger init so alpha matters
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // Find alpha (1, 1) in params
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Ap = nullptr;
        Tensor* Ag = nullptr;
        for (size_t k = 0; k < params.size(); ++k) {
            if (params[k]->rows == 1 && params[k]->cols == 1) {
                Ap = params[k];
                Ag = grads[k];
                break;
            }
        }
        if (!Ap) {
            cout << "[FAIL] could not find alpha in parameters\n";
        } else {
            double orig_a = (*Ap)(0, 0);
            (*Ap)(0, 0) = orig_a + eps;
            Tensor out_p = block.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Ap)(0, 0) = orig_a - eps;
            Tensor out_m = block.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Ap)(0, 0) = orig_a;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Ag)(0, 0);
            double err = relative_error(num, ana);
            cout << "alpha: ana=" << ana << " num=" << num << " err=" << err << "\n";
            if (err < 0.05) {
                cout << "[PASS] alpha gradient check (rel_err < 5%)\n";
                ++passed;
            } else {
                cout << "[FAIL] alpha gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 7: gMLPBlock zero_grad() clears all gradients
    // ------------------------------------------------------------
    cout << "\n--- Test 7: gMLPBlock zero_grad() clears all gradients ---\n";
    {
        ++total;
        size_t S = 4, d = 3;
        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPBlock block(d, S);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.backward(grad_loss, 0.0);
        block.zero_grad();

        bool all_zero = true;
        for (auto* g : block.gradients()) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (std::fabs(g->data[i]) > 1e-12) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
        if (all_zero) {
            cout << "[PASS] all gradients zero after zero_grad()\n";
            ++passed;
        } else {
            cout << "[FAIL] some gradients non-zero after zero_grad()\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: gMLPModel forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 8: gMLPModel forward shape (S=6, d=4, 2 blocks, out=3) ---\n";
    {
        ++total;
        size_t S = 6, d = 4, out_f = 3;
        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j;

        gMLPModel model(d, S, out_f, /*num_blocks=*/2);
        Tensor out = model.forward(input);
        cout << "Input: " << input.rows << "x" << input.cols
             << "  Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == S && out.cols == out_f) {
            cout << "[PASS] model output shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << S << "x" << out_f << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: gMLPModel input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: gMLPModel input gradient check (L2, 2 blocks) ---\n";
    {
        ++total;
        size_t S = 3, d = 2, out_f = 2;
        double eps = 1e-5;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, out_f);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPModel model(d, S, out_f, 2);
        Tensor out = model.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < S; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = model.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = model.forward(input);
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
            cout << "[PASS] model input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] model input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: gMLPModel training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: gMLPModel training step reduces loss ---\n";
    {
        ++total;
        size_t S = 4, d = 3, out_f = 2;
        double lr = 0.01;

        Tensor input(S, d);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(S, out_f);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        gMLPModel model(d, S, out_f, 2);
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

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
