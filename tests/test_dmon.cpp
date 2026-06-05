// DMon (Diffusion Module Network) tests.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/architectures/dmon.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

// =============================================================
// Helpers
// =============================================================

// Small 5-node graph (chain + cross edges)
Tensor make_adj_5nodes() {
    Tensor adj(5, 5);
    adj(0, 1) = 1; adj(1, 0) = 1;
    adj(1, 2) = 1; adj(2, 1) = 1;
    adj(2, 3) = 1; adj(3, 2) = 1;
    adj(3, 4) = 1; adj(4, 3) = 1;
    adj(1, 4) = 1; adj(4, 1) = 1;
    return adj;
}

// L2 loss value and gradient helpers
Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    return output - target;
}
double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.rows; ++i)
        for (size_t j = 0; j < output.cols; ++j) {
            double d = output(i, j) - target(i, j);
            s += 0.5 * d * d;
        }
    return s;
}

int main() {
    cout << "=== DMon Tests ===" << endl;
    int total = 0, passed = 0;

    // ================================================================
    // Test 1: DMonLayer forward shape
    // ================================================================
    cout << "\n--- Test 1: DMonLayer forward shape (5 nodes, K=3, R=4) ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, out_f = 4, K = 3, R = 4;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;
        DMonLayer layer(in_f, out_f, K, R);
        Tensor output = layer.forward_with_adj(input, adj);
        cout << "Input:  " << input.rows << "x" << input.cols << "\n";
        cout << "Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == N && output.cols == out_f) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << out_f << "\n";
        }
    }

    // ================================================================
    // Test 2: DMonLayer output is finite
    // ================================================================
    cout << "\n--- Test 2: DMonLayer output is finite ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, out_f = 4;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.5 * sin(0.1 * i) + 0.3 * f;
        DMonLayer layer(in_f, out_f, 3, 4);
        Tensor output = layer.forward_with_adj(input, adj);
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

    // ================================================================
    // Test 3: DMonLayer numerical gradient check (input, L2 loss)
    // ================================================================
    cout << "\n--- Test 3: DMonLayer numerical gradient check (input) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, out_f = 2;
        double eps = 1e-5;

        // Triangle graph
        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.1 * i - 0.05 * j;

        DMonLayer layer(in_f, out_f, 2, 3);  // 2 scales, 3 Taylor terms
        Tensor out = layer.forward_with_adj(input, adj);
        Tensor grad_loss = l2_loss_grad(out, target);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = layer.forward_with_adj(input, adj);
                double loss_p = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = layer.forward_with_adj(input, adj);
                double loss_m = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num_grad = (loss_p - loss_m) / (2.0 * eps);
                double ana_grad = grad_x(i, j);
                double err = relative_error(num_grad, ana_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "input[" << i << "][" << j << "]: anal=" << ana_grad
                         << " num=" << num_grad << " rel_err=" << err << endl;
                }
            }
        }
        cout << "Max err: " << max_err << endl;
        if (max_err < 0.05) {
            cout << "[PASS] input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ================================================================
    // Test 4: DMonLayer numerical gradient check (W_out, L2 loss)
    // ================================================================
    cout << "\n--- Test 4: DMonLayer numerical gradient check (W_out) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, out_f = 2;
        double eps = 1e-5;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.1 * i - 0.05 * j;

        DMonLayer layer(in_f, out_f, 2, 3);
        Tensor out = layer.forward_with_adj(input, adj);
        Tensor grad_loss = l2_loss_grad(out, target);
        layer.zero_grad();
        layer.backward(grad_loss, 0.0);

        // W_out is in the first param slot of output_dense_ (the weights matrix).
        Tensor* Wp = layer.parameters()[0];
        Tensor* Gp = layer.gradients()[0];
        size_t W_rows = Wp->rows;
        size_t W_cols = Wp->cols;

        double max_err = 0.0;
        int n_checked = 0;
        // Spot-check a few W entries
        for (size_t r = 0; r < W_rows; ++r) {
            for (size_t c = 0; c < W_cols; ++c) {
                if (n_checked >= 3) break;  // just a few
                double orig = (*Wp)(r, c);
                (*Wp)(r, c) = orig + eps;
                Tensor out_p = layer.forward_with_adj(input, adj);
                double loss_p = l2_loss_value(out_p, target);
                (*Wp)(r, c) = orig - eps;
                Tensor out_m = layer.forward_with_adj(input, adj);
                double loss_m = l2_loss_value(out_m, target);
                (*Wp)(r, c) = orig;
                double num = (loss_p - loss_m) / (2.0 * eps);
                double ana = (*Gp)(r, c);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                cout << "W[" << r << "][" << c << "]: anal=" << ana
                     << " num=" << num << " rel_err=" << err << endl;
                ++n_checked;
            }
        }
        if (max_err < 0.05) {
            cout << "[PASS] W gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W gradient check failed\n";
        }
    }

    // ================================================================
    // Test 5: DMonModel forward shape (with input projection)
    // ================================================================
    cout << "\n--- Test 5: DMonModel forward shape (input proj + 2 layers) ---\n";
    {
        ++total;
        size_t N = 5, in_f = 2, hidden = 4, out_f = 3;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) - 0.05 * f;
        DMonModel model(N, in_f, hidden, out_f, /*num_layers=*/2,
                        /*num_scales=*/3, /*taylor_terms=*/4);
        Tensor out = model.forward_with_adj(input, adj);
        cout << "Input:  " << input.rows << "x" << input.cols << "\n";
        cout << "Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == N && out.cols == out_f) {
            cout << "[PASS] model output shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << out_f << "\n";
        }
    }

    // ================================================================
    // Test 6: DMonModel output is finite
    // ================================================================
    cout << "\n--- Test 6: DMonModel output is finite ---\n";
    {
        ++total;
        size_t N = 5, in_f = 2, hidden = 4, out_f = 3;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.3 * cos(0.2 * i) - 0.1 * f;
        DMonModel model(N, in_f, hidden, out_f, 2, 3, 4);
        Tensor out = model.forward_with_adj(input, adj);
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] all model outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite model output detected\n";
        }
    }

    // ================================================================
    // Test 7: DMonModel input gradient check (L2 loss)
    // ================================================================
    cout << "\n--- Test 7: DMonModel input gradient check (L2 loss) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3, out_f = 2;
        double eps = 1e-5;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.1 * i - 0.05 * j;

        DMonModel model(N, in_f, hidden, out_f, 2, 2, 3);
        Tensor out = model.forward_with_adj(input, adj);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = model.forward_with_adj(input, adj);
                double loss_p = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = model.forward_with_adj(input, adj);
                double loss_m = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num_grad = (loss_p - loss_m) / (2.0 * eps);
                double ana_grad = grad_x(i, j);
                double err = relative_error(num_grad, ana_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "input[" << i << "][" << j << "]: anal=" << ana_grad
                         << " num=" << num_grad << " rel_err=" << err << endl;
                }
            }
        }
        cout << "Max err: " << max_err << endl;
        if (max_err < 0.05) {
            cout << "[PASS] model input gradient check (rel_err < 5%)\n";
            ++passed;
        } else {
            cout << "[FAIL] model input gradient check failed\n";
        }
    }

    // ================================================================
    // Test 8: Training step reduces loss (sanity)
    // ================================================================
    cout << "\n--- Test 8: DMonModel training step reduces loss ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3, out_f = 2;
        double lr = 0.01;

        // Square graph
        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 3) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 1) = 1; adj(2, 3) = 1;
        adj(3, 0) = 1; adj(3, 2) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;
        input(3, 0) = 0.2;  input(3, 1) = -0.5;

        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                target(i, j) = 0.1 * i - 0.05 * j;

        DMonModel model(N, in_f, hidden, out_f, 2, 2, 3);

        Tensor out0 = model.forward_with_adj(input, adj);
        double loss0 = l2_loss_value(out0, target);

        for (int step = 0; step < 10; ++step) {
            model.zero_grad();
            Tensor out = model.forward_with_adj(input, adj);
            Tensor grad_loss = l2_loss_grad(out, target);
            model.backward(grad_loss, 0.0);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward_with_adj(input, adj);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << endl;
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
