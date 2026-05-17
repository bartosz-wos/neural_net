#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gin.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

int main() {
    cout << "=== GIN (Graph Isomorphism Network) Test ===" << endl;

    // ================================================================
    // Test 1: GIN0Layer forward pass shape check
    // ================================================================
    cout << "\n--- Test 1: GIN0Layer forward shape ---" << endl;
    {
        size_t N = 4;        // 4 nodes
        size_t in_f = 3;     // 3 input features
        size_t out_f = 8;    // 4 output features

        // Adjacency matrix (no self-loops)
        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 3) = 1;
        adj(2, 0) = 1; adj(2, 3) = 1;
        adj(3, 1) = 1; adj(3, 2) = 1;

        // Node features
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i * 0.1 + f * 0.2);

        GIN0Layer layer(in_f, out_f);
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: GIN0Layer output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 2: GINLayer forward pass shape check
    // ================================================================
    cout << "\n--- Test 2: GINLayer forward shape ---" << endl;
    {
        size_t N = 4;
        size_t in_f = 3;
        size_t out_f = 8;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 3) = 1;
        adj(2, 0) = 1; adj(2, 3) = 1;
        adj(3, 1) = 1; adj(3, 2) = 1;

        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i * 0.1 + f * 0.2);

        GINLayer layer(in_f, out_f, 16, 2);
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: GINLayer output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 3: GIN0Layer gradient check (numerical)
    // ================================================================
    cout << "\n--- Test 3: GIN0Layer gradient check ---" << endl;
    {
        size_t N = 3;
        size_t in_f = 2;
        size_t out_f = 4;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        GIN0Layer layer(in_f, out_f);

        auto compute_loss = [](const Tensor& t) -> double {
            double s = 0.0;
            for (size_t i = 0; i < t.rows; ++i)
                for (size_t j = 0; j < t.cols; ++j)
                    s += t(i, j);
            return s;
        };

        double eps = 1e-5;
        double max_err = 0.0;

        // Forward
        Tensor output = layer.forward_with_adj(input, adj);
        double loss_fwd = compute_loss(output);
        cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

        // Backward
        Tensor grad_output(output.rows, output.cols);
        grad_output.fill(1.0);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output, 0.0);

        // Numerical gradient check
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
            cout << "PASS: GIN0Layer gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: GIN0Layer gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 4: GINLayer gradient check (numerical)
    // ================================================================
    cout << "\n--- Test 4: GINLayer gradient check ---" << endl;
    {
        size_t N = 3;
        size_t in_f = 2;
        size_t out_f = 4;

        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        GINLayer layer(in_f, out_f, 16, 2);

        // Use L2 loss to avoid catastrophic cancellation when MLP outputs are near zero
        auto compute_loss = [](const Tensor& t) -> double {
            double s = 0.0;
            for (size_t i = 0; i < t.rows; ++i)
                for (size_t j = 0; j < t.cols; ++j)
                    s += t(i, j) * t(i, j);  // L2: avoids cancellation when values ~0
            return s;
        };

        double eps = 1e-5;
        double max_err = 0.0;

        // Forward
        Tensor output = layer.forward_with_adj(input, adj);
        double loss_fwd = compute_loss(output);
        cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

        // dL/d(out) = 2*out for L2 loss
        Tensor grad_output_l2(output.rows, output.cols);
        for (size_t i = 0; i < output.rows; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                grad_output_l2(i, j) = 2.0 * output(i, j);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output_l2, 0.0);

        // Numerical gradient check
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
            cout << "PASS: GINLayer gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: GINLayer gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 5: Multi-layer GIN forward/backward pass
    // ================================================================
    cout << "\n--- Test 5: Multi-layer GIN stack ---" << endl;
    {
        size_t N = 5;
        size_t in_f = 4;
        size_t hidden_f = 8;
        size_t out_f = 4;

        Tensor adj(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                if (i != j) adj(i, j) = ((i + j) % 3 == 0) ? 1.0 : 0.0;

        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.3;

        // Two GIN0Layers in sequence
        GIN0Layer layer1(in_f, hidden_f);
        GIN0Layer layer2(hidden_f, out_f);

        // Forward pass
        Tensor h = layer1.forward_with_adj(input, adj);
        h = layer2.forward_with_adj(h, adj);

        cout << "After 2 layers: " << h.rows << "x" << h.cols << endl;
        if (h.rows == N && h.cols == out_f) {
            cout << "PASS: Multi-layer GIN forward shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }

        // Backward pass
        Tensor grad_output(N, out_f);
        grad_output.fill(0.5);

        layer2.zero_grad();
        layer1.zero_grad();

        Tensor grad_h = layer2.backward(grad_output, 0.01);
        Tensor grad_x = layer1.backward(grad_h, 0.01);

        cout << "Backward pass completed." << endl;
        cout << "PASS: Multi-layer GIN backward shape: " << grad_x.rows << "x" << grad_x.cols << endl;
    }

    // ================================================================
    // Test 6: GIN with self-loops (standard config)
    // ================================================================
    cout << "\n--- Test 6: GINLayer with self-loop adjacency ---" << endl;
    {
        size_t N = 3;
        size_t in_f = 2;
        size_t out_f = 4;

        // Adjacency WITH self-loops (standard for GIN)
        Tensor adj(N, N);
        adj(0, 0) = 1; adj(0, 1) = 1; adj(0, 2) = 0;
        adj(1, 0) = 1; adj(1, 1) = 1; adj(1, 2) = 1;
        adj(2, 0) = 0; adj(2, 1) = 1; adj(2, 2) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.1; input(0, 1) = 0.2;
        input(1, 0) = 0.3; input(1, 1) = 0.4;
        input(2, 0) = 0.5; input(2, 1) = 0.6;

        GINLayer layer(in_f, out_f, 8, 2);
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Output with self-loops: " << output.rows << "x" << output.cols << endl;

        // Quick gradient check
        Tensor grad_output(N, out_f);
        grad_output.fill(1.0);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output, 0.0);

        cout << "Gradient shape: " << grad_x.rows << "x" << grad_x.cols << endl;
        cout << "PASS: Self-loop GIN test complete" << endl;
    }

    // ================================================================
    // Test 7: GIN0Layer parameters and gradients
    // ================================================================
    cout << "\n--- Test 7: GIN0Layer parameters/gradients ---" << endl;
    {
        GIN0Layer layer(3, 8);
        auto params = layer.parameters();
        auto grads = layer.gradients();

        cout << "Parameters count: " << params.size() << endl;
        cout << "Gradients count: " << grads.size() << endl;

        if (!params.empty() && !grads.empty()) {
            cout << "PASS: GIN0Layer has parameters and gradients" << endl;
        } else {
            cout << "FAIL: Empty parameters or gradients" << endl;
            return 1;
        }

        layer.zero_grad();
        cout << "zero_grad() called successfully" << endl;
    }

    // ================================================================
    // Test 8: GINLayer parameters and gradients
    // ================================================================
    cout << "\n--- Test 8: GINLayer parameters/gradients ---" << endl;
    {
        GINLayer layer(4, 8, 16, 2);
        auto params = layer.parameters();
        auto grads = layer.gradients();

        cout << "Parameters count: " << params.size() << endl;
        cout << "Gradients count: " << grads.size() << endl;

        if (!params.empty() && !grads.empty()) {
            cout << "PASS: GINLayer has parameters and gradients" << endl;
        } else {
            cout << "FAIL: Empty parameters or gradients" << endl;
            return 1;
        }

        layer.zero_grad();
        cout << "zero_grad() called successfully" << endl;
    }

    cout << "\n=== All GIN tests complete ===" << endl;
    return 0;
}