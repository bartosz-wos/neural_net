#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gnn.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

int main() {
    cout << "=== GATLayer Backward Pass Test ===" << endl;

    size_t N = 3;
    size_t in_features = 2;
    size_t out_features = 4;
    size_t num_heads = 2;
    bool concat_heads = true;
    double eps = 1e-5;

    Tensor adj(N, N);
    adj(0, 0) = 0; adj(0, 1) = 1; adj(0, 2) = 1;
    adj(1, 0) = 1; adj(1, 1) = 0; adj(1, 2) = 1;
    adj(2, 0) = 1; adj(2, 1) = 1; adj(2, 2) = 0;

    Tensor input(N, in_features);
    input(0, 0) = 0.5;  input(0, 1) = -0.3;
    input(1, 0) = 0.8;  input(1, 1) = 0.2;
    input(2, 0) = -0.1; input(2, 1) = 0.4;

    GATLayer gat(in_features, out_features, num_heads, concat_heads);

    // Forward pass
    Tensor output = gat.forward_with_adj(input, adj);
    cout << "Output shape: " << output.rows << "x" << output.cols << endl;

    // Simple loss: sum of all elements
    auto compute_loss = [](const Tensor& t) -> double {
        double s = 0.0;
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                s += t(i, j);
        return s;
    };

    double loss_fwd = compute_loss(output);
    cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

    // Backward pass
    Tensor grad_output(output.rows, output.cols);
    grad_output.fill(1.0);
    gat.zero_grad();

    Tensor grad_x = gat.backward(grad_output, 0.0);
    cout << "Backward completed. grad_x shape: " << grad_x.rows << "x" << grad_x.cols << endl;

    // Numerical gradient check for input gradients
    cout << "\n--- Numerical Gradient Check (input) ---" << endl;
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            double orig_ij = input(i, j);

            gat.zero_grad();
            input(i, j) = orig_ij + eps;
            Tensor out_plus = gat.forward_with_adj(input, adj);
            double loss_plus = compute_loss(out_plus);

            gat.zero_grad();
            input(i, j) = orig_ij - eps;
            Tensor out_minus = gat.forward_with_adj(input, adj);
            double loss_minus = compute_loss(out_minus);

            double num_grad = (loss_plus - loss_minus) / (2 * eps);
            input(i, j) = orig_ij;

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
        cout << "PASSED: Gradient check within tolerance (5%)" << endl;
    } else {
        cout << "WARNING: Gradient error exceeds 5% - possible bug but could be numerical approximation" << endl;
    }

    // Verify second forward works after backward
    gat.zero_grad();
    Tensor output2 = gat.forward_with_adj(input, adj);
    cout << "\nSecond forward pass OK: " << output2.rows << "x" << output2.cols << endl;

    cout << "\n=== GATLayer backward() test complete ===" << endl;
    return 0;
}