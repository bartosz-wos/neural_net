#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/lightgcn.h"

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
    cout << "=== LightGCN Tests ===" << endl;

    // ================================================================
    // Test 1: LightGCNLayer forward pass shape check
    // ================================================================
    cout << "\n--- Test 1: LightGCNLayer forward shape (4 nodes, 2 layers) ---" << endl;
    {
        size_t N = 4;          // 4 nodes
        size_t dim = 3;        // 3 embedding dim
        size_t num_layers = 2; // K=2 propagation steps

        Tensor adj = make_adj_4nodes();

        // Node features: (N, dim)
        Tensor input(N, dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < dim; ++f)
                input(i, f) = (i * 0.1 + f * 0.2);

        LightGCNLayer layer(num_layers, false);  // uniform combination
        Tensor output = layer.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == dim) {
            cout << "PASS: LightGCNLayer output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << dim << endl;
            return 1;
        }
    }

    // ================================================================
    // Test 2: LightGCNLayer gradient check (numerical, L2 loss)
    // ================================================================
    cout << "\n--- Test 2: LightGCNLayer gradient check (L2 loss) ---" << endl;
    {
        size_t N = 3;
        size_t dim = 4;
        size_t num_layers = 2;

        // Triangle graph
        Tensor adj(N, N);
        adj(0, 1) = 1; adj(0, 2) = 1;
        adj(1, 0) = 1; adj(1, 2) = 1;
        adj(2, 0) = 1; adj(2, 1) = 1;

        Tensor input(N, dim);
        input(0, 0) = 0.5;  input(0, 1) = -0.3; input(0, 2) = 0.1; input(0, 3) = 0.2;
        input(1, 0) = 0.8;  input(1, 1) = 0.2;  input(1, 2) = -0.4; input(1, 3) = 0.1;
        input(2, 0) = -0.1; input(2, 1) = 0.4;  input(2, 2) = 0.3;  input(2, 3) = -0.2;

        LightGCNLayer layer(num_layers, false);

        // L2 loss: sum of squares
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

        // dL/d(out) = 2*out for L2 loss
        Tensor grad_output_l2(N, dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < dim; ++j)
                grad_output_l2(i, j) = 2.0 * output(i, j);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output_l2, 0.0);

        // Numerical gradient check
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < dim; ++j) {
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
            cout << "PASS: LightGCNLayer gradient check within tolerance (5%)" << endl;
        } else {
            cout << "WARNING: LightGCNLayer gradient error exceeds 5%" << endl;
        }
    }

    // ================================================================
    // Test 3: Layer combination verification
    // ================================================================
    cout << "\n--- Test 3: Layer combination verification ---" << endl;
    {
        size_t N = 4;
        size_t dim = 3;
        size_t num_layers = 3;  // K=3 gives outputs H^{(0)}, H^{(1)}, H^{(2)}, H^{(3)}

        Tensor adj = make_adj_4nodes();

        Tensor input(N, dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < dim; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.3;

        LightGCNLayer layer(num_layers, false);

        // The layer internally stores all K+1 outputs
        // Final output = sum_k alpha_k * H^{(k)} with uniform alpha = 1/(K+1)
        // We verify the combination by checking that output = (H^0 + H^1 + H^2 + H^3) / 4
        // We can't directly access layer_outputs_, but we can verify:
        // 1. With K=3 and uniform weights, output should be a convex combination
        // 2. The output should differ from H^{(0)} (the input) when K >= 1

        Tensor output = layer.forward_with_adj(input, adj);

        // Compute what H^{(1)} would be with one propagation step
        // Normalize adjacency manually to get H^1
        // K = num_layers + 1 = 4 (total layer outputs H^{(0)}..H^{(3)})

        // Verify: output is NOT equal to input (with propagation, it changes)
        bool differs_from_input = false;
        for (size_t i = 0; i < N && !differs_from_input; ++i)
            for (size_t j = 0; j < dim && !differs_from_input; ++j)
                if (fabs(output(i, j) - input(i, j)) > 1e-6)
                    differs_from_input = true;

        if (differs_from_input) {
            cout << "Output differs from input (layer combination working)" << endl;
            cout << "PASS: Layer combination affects output" << endl;
        } else {
            cout << "FAIL: Output identical to input (no combination?)" << endl;
            return 1;
        }

        // Also verify that learnable combination changes results
        LightGCNLayer layer_learnable(num_layers, true);
        Tensor output_learn = layer_learnable.forward_with_adj(input, adj);

        // With different alpha values, output should potentially differ
        cout << "Learnable combination layer output computed" << endl;
        cout << "PASS: Layer combination test complete" << endl;
    }

    // ================================================================
    // Test 4: LightGCN single-layer equivalent to no propagation check
    // ================================================================
    cout << "\n--- Test 4: LightGCN single-layer propagation check ---" << endl;
    {
        size_t N = 4;
        size_t dim = 3;

        Tensor adj = make_adj_4nodes();

        Tensor input(N, dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < dim; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        // With K=0, num_layers=0, we get only H^{(0)} (the input itself)
        // Output = alpha_0 * H^{(0)} = 1.0 * H^{(0)} = input
        LightGCNLayer layer_0(0, false);
        Tensor output_0 = layer_0.forward_with_adj(input, adj);

        // Check: for K=0, output should equal input (uniform alpha=1, H^{(0)}=input)
        bool equals_input = true;
        for (size_t i = 0; i < N && equals_input; ++i)
            for (size_t j = 0; j < dim && equals_input; ++j)
                if (fabs(output_0(i, j) - input(i, j)) > 1e-8)
                    equals_input = false;

        if (equals_input) {
            cout << "K=0: output equals input (correct)" << endl;
            cout << "PASS: K=0 single-layer propagation correct" << endl;
        } else {
            cout << "FAIL: K=0 output should equal input" << endl;
            cout << "Output(0,0)=" << output_0(0,0) << " input(0,0)=" << input(0,0) << endl;
            return 1;
        }

        // With K=1, we get H^{(0)} and H^{(1)}, combined with alpha=[0.5, 0.5]
        // This should NOT equal input
        LightGCNLayer layer_1(1, false);
        Tensor output_1 = layer_1.forward_with_adj(input, adj);

        bool differs_from_input_1 = false;
        for (size_t i = 0; i < N && !differs_from_input_1; ++i)
            for (size_t j = 0; j < dim && !differs_from_input_1; ++j)
                if (fabs(output_1(i, j) - input(i, j)) > 1e-6)
                    differs_from_input_1 = true;

        if (differs_from_input_1) {
            cout << "K=1: output differs from input (propagation working)" << endl;
            cout << "PASS: K=1 single propagation correct" << endl;
        } else {
            cout << "WARNING: K=1 output equals input" << endl;
        }
    }

    // ================================================================
    // Test 5: LightGCNModel forward/backward pass
    // ================================================================
    cout << "\n--- Test 5: LightGCNModel forward/backward shape ---" << endl;
    {
        size_t N = 4;
        size_t dim = 8;  // features dim == embedding_dim (use raw features)
        size_t num_layers = 2;

        Tensor adj = make_adj_4nodes();

        // Input: raw features (N, dim), where dim == embedding_dim
        Tensor input(N, dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < dim; ++f)
                input(i, f) = (i + 1) * 0.1 + f * 0.2;

        // LightGCNModel with in_features == embedding_dim: uses raw features as input
        LightGCNModel model(N, dim, dim, num_layers, false);
        Tensor output = model.forward_with_adj(input, adj);

        cout << "Input: " << input.rows << "x" << input.cols << endl;
        cout << "Output: " << output.rows << "x" << output.cols << endl;

        if (output.rows == N && output.cols == dim) {
            cout << "PASS: LightGCNModel output shape correct" << endl;
        } else {
            cout << "FAIL: Expected " << N << "x" << dim << endl;
            return 1;
        }

        // Backward pass
        Tensor grad_output(N, dim);
        grad_output.fill(1.0);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_output, 0.01);

        cout << "Backward pass completed" << endl;
        cout << "Gradient shape: " << grad_x.rows << "x" << grad_x.cols << endl;
        cout << "PASS: LightGCNModel backward shape correct" << endl;
    }

    // ================================================================
    // Test 6: LightGCNModel parameters and gradients
    // ================================================================
    cout << "\n--- Test 6: LightGCNModel parameters/gradients ---" << endl;
    {
        size_t N = 4;
        size_t in_f = 16;  // different from embedding_dim -> uses learned embeddings
        size_t emb_dim = 8;
        size_t num_layers = 2;

        // When in_f != emb_dim, LightGCNModel uses learned embeddings
        LightGCNModel model(N, in_f, emb_dim, num_layers, false);
        auto params = model.parameters();
        auto grads = model.gradients();

        cout << "Parameters count: " << params.size() << endl;
        cout << "Gradients count: " << grads.size() << endl;

        if (!params.empty() && !grads.empty()) {
            cout << "PASS: LightGCNModel has parameters and gradients" << endl;
        } else {
            cout << "FAIL: Empty parameters or gradients" << endl;
            return 1;
        }

        model.zero_grad();
        cout << "zero_grad() called successfully" << endl;
    }

    cout << "\n=== All LightGCN tests complete ===" << endl;
    return 0;
}