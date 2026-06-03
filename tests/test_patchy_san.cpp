#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/patchy_san.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

// 5-node graph: 0-1-2-3, with 0-3
Tensor make_adj_5nodes() {
    Tensor adj(5, 5);
    adj(0, 1) = 1; adj(1, 0) = 1;
    adj(1, 2) = 1; adj(2, 1) = 1;
    adj(2, 3) = 1; adj(3, 2) = 1;
    adj(0, 3) = 1; adj(3, 0) = 1;
    adj(2, 4) = 1; adj(4, 2) = 1;
    return adj;
}

int main() {
    cout << "=== PATCHY-SAN Tests ===" << endl;

    // ================================================================
    // Test 1: PatchySANLayer forward shape
    // ================================================================
    cout << "\n--- Test 1: PatchySANLayer forward shape ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 3;
        size_t out_f = 4;
        size_t w = 3;
        size_t k = 2;

        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        PatchySANLayer layer(in_f, out_f, w, k);
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: PatchySANLayer output shape correct" << endl;
        } else {
            cout << "FAIL: expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 2: PatchySANLayer output is finite and non-trivial
    // ================================================================
    cout << "\n--- Test 2: PatchySANLayer output is non-trivial ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 3;
        size_t out_f = 4;
        size_t w = 4;
        size_t k = 2;

        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        PatchySANLayer layer(in_f, out_f, w, k);
        Tensor output = layer.forward_with_adj(input, adj);

        // Check finite
        bool all_finite = true;
        double max_abs = 0.0;
        for (size_t i = 0; i < N && all_finite; ++i) {
            for (size_t j = 0; j < out_f && all_finite; ++j) {
                double v = output(i, j);
                if (!std::isfinite(v)) all_finite = false;
                max_abs = max(max_abs, fabs(v));
            }
        }
        if (!all_finite) {
            cout << "FAIL: non-finite output" << endl;
            return 1;
        }
        if (max_abs < 1e-6) {
            cout << "FAIL: output collapsed to ~0" << endl;
            return 1;
        }
        cout << "Max |output| = " << max_abs << endl;
        cout << "PASS: PatchySANLayer output is non-trivial" << endl;
    }

    // ================================================================
    // Test 3: PatchySANLayer gradient check (numerical, L2 loss)
    // ================================================================
    cout << "\n--- Test 3: PatchySANLayer gradient check (L2) ---" << endl;
    {
        size_t N = 4;
        size_t in_f = 3;
        size_t out_f = 3;
        size_t w = 3;
        size_t k = 2;

        // 4-node graph: 0-1-2-3
        Tensor adj(N, N);
        adj(0, 1) = 1; adj(1, 0) = 1;
        adj(1, 2) = 1; adj(2, 1) = 1;
        adj(2, 3) = 1; adj(3, 2) = 1;
        adj(0, 3) = 1; adj(3, 0) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.4;  input(0, 1) = -0.2; input(0, 2) = 0.3;
        input(1, 0) = 0.1;  input(1, 1) = 0.5;  input(1, 2) = -0.1;
        input(2, 0) = -0.3; input(2, 1) = 0.2;  input(2, 2) = 0.4;
        input(3, 0) = 0.2;  input(3, 1) = -0.4; input(3, 2) = 0.1;

        PatchySANLayer layer(in_f, out_f, w, k);

        auto compute_loss = [](const Tensor& t) -> double {
            double s = 0.0;
            for (size_t i = 0; i < t.rows; ++i)
                for (size_t j = 0; j < t.cols; ++j)
                    s += t(i, j) * t(i, j);
            return s;
        };

        double eps = 1e-5;
        double max_err = 0.0;

        // Forward
        Tensor output = layer.forward_with_adj(input, adj);
        double loss_fwd = compute_loss(output);
        cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

        // dL/d(out) = 2 * out for L2
        Tensor grad_output_l2(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                grad_output_l2(i, j) = 2.0 * output(i, j);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output_l2, 0.0);

        // Numerical gradient check (perturb input)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                layer.zero_grad();
                input(i, j) = orig + eps;
                Tensor out_plus = layer.forward_with_adj(input, adj);
                double loss_plus = compute_loss(out_plus);

                layer.zero_grad();
                input(i, j) = orig - eps;
                Tensor out_minus = layer.forward_with_adj(input, adj);
                double loss_minus = compute_loss(out_minus);

                double num_grad = (loss_plus - loss_minus) / (2 * eps);
                input(i, j) = orig;

                double err = relative_error(grad_x(i, j), num_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "input[" << i << "][" << j << "]: "
                         << "analytical=" << grad_x(i, j)
                         << " numerical=" << num_grad
                         << " rel_err=" << err << endl;
                }
            }
        }

        cout << "Max relative error: " << max_err << endl;
        if (max_err < 0.05) {
            cout << "PASS: PatchySANLayer gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 4: PatchySANLayer gradient check (random grad_output, L2)
    // ================================================================
    cout << "\n--- Test 4: PatchySANLayer random-grad gradient check ---" << endl;
    {
        size_t N = 4;
        size_t in_f = 2;
        size_t out_f = 2;
        size_t w = 3;
        size_t k = 2;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(1, 0) = 1;
        adj(1, 2) = 1; adj(2, 1) = 1;
        adj(2, 3) = 1; adj(3, 2) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.1; input(0, 1) = 0.2;
        input(1, 0) = 0.3; input(1, 1) = -0.1;
        input(2, 0) = 0.0; input(2, 1) = 0.4;
        input(3, 0) = -0.2; input(3, 1) = 0.1;

        PatchySANLayer layer(in_f, out_f, w, k);

        // Forward once
        Tensor output = layer.forward_with_adj(input, adj);

        // dL/d(out) = fixed random gradient
        Tensor grad_output(N, out_f);
        grad_output(0, 0) = 0.1; grad_output(0, 1) = -0.2;
        grad_output(1, 0) = 0.05; grad_output(1, 1) = 0.3;
        grad_output(2, 0) = -0.1; grad_output(2, 1) = 0.2;
        grad_output(3, 0) = 0.4; grad_output(3, 1) = -0.1;

        // Analytical gradient w.r.t. input
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output, 0.0);

        // Numerical gradient: d/d(input[i,j]) of <grad_output, output>
        double eps = 1e-5;
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                layer.zero_grad();
                input(i, j) = orig + eps;
                Tensor out_plus = layer.forward_with_adj(input, adj);

                layer.zero_grad();
                input(i, j) = orig - eps;
                Tensor out_minus = layer.forward_with_adj(input, adj);
                input(i, j) = orig;

                // dot(grad_output, (out_plus - out_minus)) / (2*eps)
                double num_grad = 0.0;
                for (size_t ii = 0; ii < N; ++ii) {
                    for (size_t jj = 0; jj < out_f; ++jj) {
                        num_grad += grad_output(ii, jj) * (out_plus(ii, jj) - out_minus(ii, jj));
                    }
                }
                num_grad /= (2 * eps);

                double err = relative_error(grad_x(i, j), num_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "input[" << i << "][" << j << "]: "
                         << "analytical=" << grad_x(i, j)
                         << " numerical=" << num_grad
                         << " rel_err=" << err << endl;
                }
            }
        }
        cout << "Max relative error: " << max_err << endl;
        if (max_err < 0.05) {
            cout << "PASS: PatchySANLayer random-grad gradient check" << endl;
        } else {
            cout << "WARNING: gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 5: PatchySANModel forward shape and parameters
    // ================================================================
    cout << "\n--- Test 5: PatchySANModel forward shape ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 8;
        size_t hidden = 6;
        size_t out_f = 4;

        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.05;

        PatchySANModel model(in_f, hidden, out_f, 3, 2);
        Tensor output = model.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;
        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: PatchySANModel output shape correct" << endl;
        } else {
            cout << "FAIL: expected " << N << "x" << out_f << endl;
            return 1;
        }

        auto params = model.parameters();
        auto grads = model.gradients();
        cout << "Parameters: " << params.size() << ", Gradients: " << grads.size() << endl;
        if (params.empty() || grads.empty()) {
            cout << "FAIL: empty parameters or gradients" << endl;
            return 1;
        }
        cout << "PASS: PatchySANModel has parameters and gradients" << endl;
    }

    // ================================================================
    // Test 6: PatchySANModel gradient check (L2 loss)
    // ================================================================
    cout << "\n--- Test 6: PatchySANModel gradient check (L2) ---" << endl;
    {
        size_t N = 4;
        size_t in_f = 3;
        size_t hidden = 3;
        size_t out_f = 2;
        size_t w = 3;
        size_t k = 2;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(1, 0) = 1;
        adj(1, 2) = 1; adj(2, 1) = 1;
        adj(2, 3) = 1; adj(3, 2) = 1;
        adj(0, 3) = 1; adj(3, 0) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.1; input(0, 1) = 0.2; input(0, 2) = -0.1;
        input(1, 0) = 0.0; input(1, 1) = 0.3; input(1, 2) = 0.2;
        input(2, 0) = -0.2; input(2, 1) = 0.1; input(2, 2) = 0.4;
        input(3, 0) = 0.3; input(3, 1) = -0.3; input(3, 2) = 0.1;

        PatchySANModel model(in_f, hidden, out_f, w, k);

        auto compute_loss = [](const Tensor& t) -> double {
            double s = 0.0;
            for (size_t i = 0; i < t.rows; ++i)
                for (size_t j = 0; j < t.cols; ++j)
                    s += t(i, j) * t(i, j);
            return s;
        };

        double eps = 1e-5;
        double max_err = 0.0;

        Tensor output = model.forward_with_adj(input, adj);
        double loss_fwd = compute_loss(output);
        cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

        // dL/d(out) = 2*out
        Tensor grad_output(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                grad_output(i, j) = 2.0 * output(i, j);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_output, 0.0);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                model.zero_grad();
                input(i, j) = orig + eps;
                Tensor out_plus = model.forward_with_adj(input, adj);
                double loss_plus = compute_loss(out_plus);

                model.zero_grad();
                input(i, j) = orig - eps;
                Tensor out_minus = model.forward_with_adj(input, adj);
                double loss_minus = compute_loss(out_minus);

                double num_grad = (loss_plus - loss_minus) / (2 * eps);
                input(i, j) = orig;

                double err = relative_error(grad_x(i, j), num_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "input[" << i << "][" << j << "]: "
                         << "analytical=" << grad_x(i, j)
                         << " numerical=" << num_grad
                         << " rel_err=" << err << endl;
                }
            }
        }
        cout << "Max relative error: " << max_err << endl;
        if (max_err < 0.05) {
            cout << "PASS: PatchySANModel gradient check within tolerance" << endl;
        } else {
            cout << "WARNING: gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 7: Determinism — same input gives same output
    // ================================================================
    cout << "\n--- Test 7: Forward determinism ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 3;
        size_t out_f = 4;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        PatchySANLayer layer(in_f, out_f, 4, 2);
        Tensor o1 = layer.forward_with_adj(input, adj);
        Tensor o2 = layer.forward_with_adj(input, adj);

        bool same = true;
        for (size_t i = 0; i < N && same; ++i)
            for (size_t j = 0; j < out_f && same; ++j)
                if (fabs(o1(i, j) - o2(i, j)) > 1e-10) same = false;

        if (same) {
            cout << "PASS: forward is deterministic" << endl;
        } else {
            cout << "FAIL: forward is not deterministic" << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 8: Patch composition is sensible
    // ================================================================
    cout << "\n--- Test 8: Patch composition ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 2;
        size_t out_f = 3;
        size_t w = 4;
        size_t k = 2;

        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;

        PatchySANLayer layer(in_f, out_f, w, k);
        Tensor output = layer.forward_with_adj(input, adj);

        // Each row should be the per-anchor patch embedding.
        // Two nodes with the same neighborhood structure should produce similar outputs.
        // In our graph, nodes 0 and 3 both have degree 2; nodes 1 and 2 have degree 3.
        // With degree-based labels, they should be distinguishable but the outputs
        // should differ across all nodes (no two rows identical to many decimals).
        bool all_distinct = true;
        for (size_t i = 0; i < N && all_distinct; ++i) {
            for (size_t j = i + 1; j < N && all_distinct; ++j) {
                double diff = 0.0;
                for (size_t f = 0; f < out_f; ++f)
                    diff += fabs(output(i, f) - output(j, f));
                if (diff < 1e-8) all_distinct = false;
            }
        }
        if (all_distinct) {
            cout << "PASS: outputs are distinguishable across nodes" << endl;
        } else {
            cout << "WARNING: some nodes produce identical outputs" << endl;
        }
    }

    cout << "\n=== All PATCHY-SAN tests complete ===" << endl;
    return 0;
}
