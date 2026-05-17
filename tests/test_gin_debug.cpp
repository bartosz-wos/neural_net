// Debug gradient check for GIN0Layer
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
    cout << "=== GIN0Layer Gradient Debug ===" << endl;

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

    // Print W
    const Tensor& W = *layer.parameters()[0];
    cout << "W shape: " << W.rows << "x" << W.cols << endl;
    cout << "W = " << endl;
    for (size_t i = 0; i < W.rows; ++i) {
        for (size_t j = 0; j < W.cols; ++j) {
            cout << " " << W(i, j);
        }
        cout << endl;
    }

    auto compute_loss = [](const Tensor& t) -> double {
        double s = 0.0;
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                s += t(i, j);
        return s;
    };

    double eps = 1e-5;

    // Forward
    Tensor output = layer.forward_with_adj(input, adj);
    double loss_fwd = compute_loss(output);
    cout << "\nLoss (forward): " << fixed << setprecision(8) << loss_fwd << endl;

    // Print aggregated values
    // Reconstruct: agg[i] = sum of neighbor features
    cout << "\n--- Forward pass details ---" << endl;
    for (size_t i = 0; i < N; ++i) {
        double agg0 = 0, agg1 = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 1e-9) {
                agg0 += input(j, 0);
                agg1 += input(j, 1);
            }
        }
        cout << "Node " << i << ": input=[" << input(i,0) << "," << input(i,1)
             << "] agg=[" << agg0 << "," << agg1 << "]"
             << " combined=[" << input(i,0)+agg0 << "," << input(i,1)+agg1 << "]" << endl;
    }

    // Backward (learning_rate=0 so W doesn't change)
    Tensor grad_output(output.rows, output.cols);
    grad_output.fill(1.0);
    layer.zero_grad();
    Tensor grad_x = layer.backward(grad_output, 0.0);

    cout << "\n--- Analytical gradients ---" << endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            cout << "grad_x[" << i << "][" << j << "] = " << grad_x(i, j) << endl;
        }
    }

    // Numerical check
    cout << "\n--- Numerical gradients (checking only input[0][0]) ---" << endl;
    double orig = input(0, 0);

    layer.zero_grad();
    input(0, 0) = orig + eps;
    Tensor out_plus = layer.forward_with_adj(input, adj);
    double loss_plus = compute_loss(out_plus);

    layer.zero_grad();
    input(0, 0) = orig - eps;
    Tensor out_minus = layer.forward_with_adj(input, adj);
    double loss_minus = compute_loss(out_minus);

    double num_grad = (loss_plus - loss_minus) / (2 * eps);
    input(0, 0) = orig;

    cout << "loss_plus = " << loss_plus << endl;
    cout << "loss_minus = " << loss_minus << endl;
    cout << "num_grad[0][0] = " << num_grad << endl;
    cout << "analytical[0][0] = " << grad_x(0, 0) << endl;
    cout << "rel_err = " << relative_error(grad_x(0, 0), num_grad) << endl;

    return 0;
}