// GAT: minimal single-head, less sensitive
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gnn.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    return output - target;
}

double l2_loss_value(const Tensor& output, const Tensor& target) {
    Tensor d = output - target;
    double s = 0.0;
    for (size_t i = 0; i < d.rows; ++i)
        for (size_t j = 0; j < d.cols; ++j)
            s += 0.5 * d(i, j) * d(i, j);
    return s;
}

int main() {
    cout << "=== GAT: single head, simple config ===" << endl;

    size_t N = 3;
    size_t in_features = 2;
    size_t out_features = 2;  // single head
    size_t num_heads = 1;
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

    Tensor target(N, out_features);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features; ++j)
            target(i, j) = 0.1 * i - 0.05 * j;

    GATLayer gat(in_features, out_features, num_heads, concat_heads);
    Tensor out = gat.forward_with_adj(input, adj);
    Tensor grad_loss = l2_loss_grad(out, target);
    gat.zero_grad();
    Tensor grad_input = gat.backward(grad_loss, 0.0);

    cout << "--- Input gradient check ---" << endl;
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = gat.forward_with_adj(input, adj);
            double loss_p = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = gat.forward_with_adj(input, adj);
            double loss_m = l2_loss_value(out_m, target);
            input(i, j) = orig;
            double num_grad = (loss_p - loss_m) / (2.0 * eps);
            double ana_grad = grad_input(i, j);
            double err = relative_error(num_grad, ana_grad);
            max_err = max(max_err, err);
            cout << "input[" << i << "][" << j << "]: anal=" << ana_grad
                 << " num=" << num_grad << " rel_err=" << err << endl;
        }
    }
    cout << "Max err: " << max_err << endl;

    cout << "\n--- W grad check ---" << endl;
    Tensor* Wp = gat.parameters()[0];
    Tensor* Gp = gat.gradients()[0];
    cout << "W[0][0]=" << (*Wp)(0,0) << " grad[0][0]=" << (*Gp)(0,0) << endl;
    double orig = (*Wp)(0, 0);
    (*Wp)(0, 0) = orig + eps;
    Tensor out_p = gat.forward_with_adj(input, adj);
    double loss_p = l2_loss_value(out_p, target);
    (*Wp)(0, 0) = orig - eps;
    Tensor out_m = gat.forward_with_adj(input, adj);
    double loss_m = l2_loss_value(out_m, target);
    (*Wp)(0, 0) = orig;
    double num = (loss_p - loss_m) / (2.0 * eps);
    double ana = (*Gp)(0, 0);
    cout << "W[0][0] anal=" << ana << " num=" << num
         << " rel_err=" << relative_error(num, ana) << endl;
    return 0;
}
