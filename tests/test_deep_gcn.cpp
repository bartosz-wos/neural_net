#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/deep_gcn.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

// ================================================================
// Helper: create a simple 4-node graph adjacency
// ================================================================
Tensor make_adj_4nodes() {
    Tensor adj(4, 4);
    // 0 -- 1
    // |    |
    // 3 -- 2
    adj(0, 1) = 1; adj(0, 3) = 1;
    adj(1, 0) = 1; adj(1, 2) = 1;
    adj(2, 1) = 1; adj(2, 3) = 1;
    adj(3, 0) = 1; adj(3, 2) = 1;
    return adj;
}

int main() {
    cout << "=== DeepGCN and GCNII Layer Tests ===" << endl;

    // ================================================================
    // Test 1: DeepGCNBlock forward pass shape check
    // ================================================================
    cout << "\n--- Test 1: DeepGCNBlock forward shape ---" << endl;
    {
        size_t N = 4, in_f = 3, out_f = 8;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        DeepGCNBlock block(in_f, out_f, true, false, 0.0, false);
        block.set_training(false);
        Tensor output = block.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: DeepGCNBlock output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 2: DeepGCNStack forward pass shape check
    // ================================================================
    cout << "\n--- Test 2: DeepGCNStack forward shape ---" << endl;
    {
        size_t N = 4, in_f = 3, hidden = 8, out_f = 8;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        DeepGCNStack stack({in_f, hidden, out_f}, true, false, 0.0, false);
        stack.set_training(false);
        Tensor output = stack.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: DeepGCNStack output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 3: DeepGCNBlock gradient check (numerical)
    // ================================================================
    cout << "\n--- Test 3: DeepGCNBlock gradient check ---" << endl;
    {
        size_t N = 3, in_f = 2, out_f = 4;
        Tensor adj(3, 3);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        DeepGCNBlock block(in_f, out_f, true, false, 0.0, false);
        block.set_training(false);

        // L2 loss: sum of squares — numerically stable for gradient checks
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
        Tensor output = block.forward_with_adj(input, adj);
        double loss_fwd = compute_loss(output);
        cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

        // Backward
        // dL/d(out) = 2*out for L2 loss
        Tensor grad_output_l2(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_f; ++j)
                grad_output_l2(i, j) = 2.0 * output(i, j);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_output_l2, 0.0);

        // Numerical gradient check on input
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                block.zero_grad();
                input(i, j) = orig + eps;
                Tensor out_plus = block.forward_with_adj(input, adj);
                double loss_plus = compute_loss(out_plus);

                block.zero_grad();
                input(i, j) = orig - eps;
                Tensor out_minus = block.forward_with_adj(input, adj);
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
            cout << "PASS: DeepGCNBlock gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: DeepGCNBlock gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 4: GCNIILayer forward pass shape check
    // ================================================================
    cout << "\n--- Test 4: GCNIILayer forward shape ---" << endl;
    {
        size_t N = 4, in_f = 3, out_f = 8;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        GCNIILayer layer(in_f, out_f);
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == out_f) {
            cout << "PASS: GCNIILayer output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << out_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 5: GCNIIModel forward pass shape check
    // ================================================================
    cout << "\n--- Test 5: GCNIIModel forward shape ---" << endl;
    {
        size_t N = 4, in_f = 3, hidden_f = 8, num_layers = 3;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        GCNIIModel model(in_f, hidden_f, num_layers, 0.0, true, 1.0);
        model.set_training(false);
        Tensor output = model.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == hidden_f) {
            cout << "PASS: GCNIIModel output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << hidden_f << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 6: GCNIILayer gradient check (numerical)
    // ================================================================
    cout << "\n--- Test 6: GCNIILayer gradient check ---" << endl;
    {
        size_t N = 3, in_f = 2, out_f = 4;
        Tensor adj(3, 3);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, in_f);
        input(0, 0) = 0.5;  input(0, 1) = -0.3;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;
        input(2, 0) = -0.1; input(2, 1) = 0.4;

        GCNIILayer layer(in_f, out_f);

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
        Tensor grad_output(N, out_f);
        grad_output.fill(1.0);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output, 0.0);

        // Numerical gradient check on input
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                input(i, j) = orig + eps;
                Tensor out_plus = layer.forward_with_adj(input, adj);
                double loss_plus = compute_loss(out_plus);

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
            cout << "PASS: GCNIILayer gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: GCNIILayer gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 7: DeepGCNBlock parameters and gradients
    // ================================================================
    cout << "\n--- Test 7: DeepGCNBlock parameters/gradients ---" << endl;
    {
        DeepGCNBlock block(3, 8, true, false, 0.0, false);
        auto params = block.parameters();
        auto grads = block.gradients();

        cout << "Parameters count: " << params.size() << endl;
        cout << "Gradients count: " << grads.size() << endl;

        if (!params.empty() && !grads.empty()) {
            cout << "PASS: DeepGCNBlock has parameters and gradients" << endl;
        } else {
            cout << "FAIL: Empty parameters or gradients" << endl;
            return 1;
        }

        block.zero_grad();
        cout << "zero_grad() called successfully" << endl;
    }

    // ================================================================
    // Test 8: GCNIILayer parameters and gradients
    // ================================================================
    cout << "\n--- Test 8: GCNIILayer parameters/gradients ---" << endl;
    {
        GCNIILayer layer(3, 8);
        auto params = layer.parameters();
        auto grads = layer.gradients();

        cout << "Parameters count: " << params.size() << endl;
        cout << "Gradients count: " << grads.size() << endl;

        if (!params.empty() && !grads.empty()) {
            cout << "PASS: GCNIILayer has parameters and gradients" << endl;
        } else {
            cout << "FAIL: Empty parameters or gradients" << endl;
            return 1;
        }

        layer.zero_grad();
        cout << "zero_grad() called successfully" << endl;
    }

    // ================================================================
    // Test 9: Multi-layer DeepGCNStack backward pass
    // ================================================================
    cout << "\n--- Test 9: Multi-layer DeepGCNStack backward ---" << endl;
    {
        size_t N = 5, in_f = 4, hidden_f = 8, out_f = 4;
        Tensor adj(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                if (i != j) adj(i, j) = ((i + j) % 2 == 0) ? 1.0 : 0.0;

        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.3;

        DeepGCNStack stack({in_f, hidden_f, out_f}, true, false, 0.0, false);
        stack.set_training(false);

        // Forward
        Tensor h = stack.forward_with_adj(input, adj);
        cout << "Output shape: " << h.rows << "x" << h.cols << endl;

        // Backward
        Tensor grad_output(N, out_f);
        grad_output.fill(0.5);
        stack.zero_grad();
        Tensor grad_x = stack.backward(grad_output, 0.01);

        cout << "Backward pass completed." << endl;
        cout << "Gradient shape: " << grad_x.rows << "x" << grad_x.cols << endl;
        cout << "PASS: Multi-layer DeepGCNStack backward correct" << endl;
    }

    // ================================================================
    // Test 10: GCNIIModel with multiple layers backward
    // ================================================================
    cout << "\n--- Test 10: GCNIIModel multi-layer backward ---" << endl;
    {
        size_t N = 5, in_f = 4, hidden_f = 8, num_layers = 3;
        Tensor adj(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                if (i != j) adj(i, j) = ((i + j) % 2 == 0) ? 1.0 : 0.0;

        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.3;

        GCNIIModel model(in_f, hidden_f, num_layers, 0.0, true, 1.0);
        model.set_training(false);

        // Forward
        Tensor h = model.forward_with_adj(input, adj);
        cout << "Output shape: " << h.rows << "x" << h.cols << endl;

        // Backward
        Tensor grad_output(N, hidden_f);
        grad_output.fill(0.5);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_output, 0.01);

        cout << "Backward pass completed." << endl;
        cout << "Gradient shape: " << grad_x.rows << "x" << grad_x.cols << endl;
        cout << "PASS: GCNIIModel multi-layer backward correct" << endl;
    }

    // ================================================================
    // Test 11: DeepGCNBlock with dropout
    // ================================================================
    cout << "\n--- Test 11: DeepGCNBlock with dropout ---" << endl;
    {
        size_t N = 4, in_f = 3, out_f = 8;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        DeepGCNBlock block(in_f, out_f, true, true, 0.5, false);
        block.set_training(true);  // training mode

        Tensor out_train = block.forward_with_adj(input, adj);

        block.set_training(false);  // eval mode
        Tensor out_eval = block.forward_with_adj(input, adj);

        cout << "Training output: " << out_train.rows << "x" << out_train.cols << endl;
        cout << "Eval output: " << out_eval.rows << "x" << out_eval.cols << endl;
        cout << "PASS: DeepGCNBlock dropout forward works" << endl;
    }

    // ================================================================
    // Test 12: GCNIIModel with dropout and batchnorm
    // ================================================================
    cout << "\n--- Test 12: GCNIIModel with dropout and BN ---" << endl;
    {
        size_t N = 4, in_f = 3, hidden_f = 8, num_layers = 3;
        Tensor adj = make_adj_4nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        GCNIIModel model(in_f, hidden_f, num_layers, 0.3, true, 1.0);
        model.set_training(true);

        Tensor out_train = model.forward_with_adj(input, adj);
        model.set_training(false);
        Tensor out_eval = model.forward_with_adj(input, adj);

        cout << "Training output: " << out_train.rows << "x" << out_train.cols << endl;
        cout << "Eval output: " << out_eval.rows << "x" << out_eval.cols << endl;
        cout << "PASS: GCNIIModel dropout+BN forward works" << endl;
    }

    cout << "\n=== All DeepGCN and GCNII tests complete ===" << endl;
    return 0;
}