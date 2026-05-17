// Debug gradient check for GINLayer - trace each step
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
    cout << "=== GINLayer Gradient Debug ===" << endl;

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

    auto compute_loss = [](const Tensor& t) -> double {
        double s = 0.0;
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                s += t(i, j);
        return s;
    };

    double eps = 1e-5;

    // Forward
    cout << "\n--- Forward pass ---" << endl;
    layer.zero_grad();
    Tensor output = layer.forward_with_adj(input, adj);
    double loss_fwd = compute_loss(output);
    cout << "Loss (forward): " << fixed << setprecision(8) << loss_fwd << endl;
    cout << "Output:" << endl;
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            cout << " " << output(i, j);
        }
        cout << endl;
    }

    // Backward
    cout << "\n--- Backward pass ---" << endl;
    Tensor grad_output(output.rows, output.cols);
    grad_output.fill(1.0);
    layer.zero_grad();
    Tensor grad_x = layer.backward(grad_output, 0.0);

    cout << "Grad input (analytical):" << endl;
    for (size_t i = 0; i < grad_x.rows; ++i) {
        for (size_t j = 0; j < grad_x.cols; ++j) {
            cout << " " << grad_x(i, j);
        }
        cout << endl;
    }

    // Numerical gradient check for input[0][0]
    cout << "\n--- Numerical gradient check for input[0][0] ---" << endl;
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

    double analytical = grad_x(0, 0);
    double err = relative_error(analytical, num_grad);

    cout << "analytical = " << analytical << endl;
    cout << "numerical  = " << num_grad << endl;
    cout << "rel_err    = " << err << endl;

    return 0;
}