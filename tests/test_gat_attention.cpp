// GAT: direct test using the new attention/gat.h include path.
// Validates the refactor: GAT lives in layers/attention/, gnn.h re-exports.
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/attention/gat.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

double l2_loss_value(const Tensor& output, const Tensor& target) {
    Tensor d = output - target;
    double s = 0.0;
    for (size_t i = 0; i < d.rows; ++i)
        for (size_t j = 0; j < d.cols; ++j)
            s += 0.5 * d(i, j) * d(i, j);
    return s;
}

Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    return output - target;
}

int main() {
    cout << "=== GAT via attention/gat.h ===" << endl;

    size_t N = 4;
    size_t in_features = 3;
    size_t out_features = 6;  // 3 heads * 2 dim
    size_t num_heads = 3;
    bool concat_heads = true;

    Tensor adj(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            adj(i, j) = (i == j) ? 0.0 : 1.0;

    Tensor input(N, in_features);
    input(0, 0) = 0.5;  input(0, 1) = -0.3; input(0, 2) = 0.1;
    input(1, 0) = 0.8;  input(1, 1) = 0.2;  input(1, 2) = -0.4;
    input(2, 0) = -0.1; input(2, 1) = 0.4;  input(2, 2) = 0.6;
    input(3, 0) = 0.2;  input(3, 1) = 0.1;  input(3, 2) = -0.2;

    Tensor target(N, out_features);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features; ++j)
            target(i, j) = 0.05 * i - 0.02 * j;

    GATLayer gat(in_features, out_features, num_heads, concat_heads);
    Tensor out = gat.forward_with_adj(input, adj);
    Tensor grad_loss = l2_loss_grad(out, target);
    gat.zero_grad();
    Tensor grad_input = gat.backward(grad_loss, 0.0);

    cout << "Output shape: " << out.rows << "x" << out.cols << endl;
    cout << "grad_input shape: " << grad_input.rows << "x" << grad_input.cols << endl;

    int passes = 0;
    int total = 0;
    double eps = 1e-5;

    // 1) Numerical gradient check for input
    cout << "\n--- Input gradient check (3 heads, 4 nodes) ---" << endl;
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
        }
    }
    total++;   // one check: the WORST element over the whole sweep
    if (max_err < 1e-2) { cout << "  [PASS] max_err=" << max_err << endl; passes++; }
    else { cout << "  [FAIL] max_err=" << max_err << endl; }

    // 2) Numerical gradient check for W
    cout << "\n--- W gradient check ---" << endl;
    auto params = gat.parameters();
    auto grads = gat.gradients();
    double max_err_w = 0.0;
    for (size_t p = 0; p < params.size(); p += 2) {  // every other is W (heads_[h].W)
        Tensor* Wp = params[p];
        Tensor* Gp = grads[p];
        for (size_t i = 0; i < Wp->rows && i < 1; ++i) {
            for (size_t j = 0; j < Wp->cols; ++j) {
                double orig = (*Wp)(i, j);
                (*Wp)(i, j) = orig + eps;
                Tensor out_p = gat.forward_with_adj(input, adj);
                double loss_p = l2_loss_value(out_p, target);
                (*Wp)(i, j) = orig - eps;
                Tensor out_m = gat.forward_with_adj(input, adj);
                double loss_m = l2_loss_value(out_m, target);
                (*Wp)(i, j) = orig;
                double num = (loss_p - loss_m) / (2.0 * eps);
                double ana = (*Gp)(i, j);
                double err = relative_error(num, ana);
                max_err_w = max(max_err_w, err);
            }
        }
    }
    total++;   // one check: the WORST element over all heads' W
    if (max_err_w < 1e-2) { cout << "  [PASS] max_err=" << max_err_w << endl; passes++; }
    else { cout << "  [FAIL] max_err=" << max_err_w << endl; }

    // 3) Numerical gradient check for a
    cout << "\n--- a gradient check ---" << endl;
    double max_err_a = 0.0;
    for (size_t p = 1; p < params.size(); p += 2) {  // every other is a (heads_[h].a)
        Tensor* ap = params[p];
        Tensor* gp = grads[p];
        for (size_t i = 0; i < ap->rows; ++i) {
            double orig = (*ap)(i, 0);
            (*ap)(i, 0) = orig + eps;
            Tensor out_p = gat.forward_with_adj(input, adj);
            double loss_p = l2_loss_value(out_p, target);
            (*ap)(i, 0) = orig - eps;
            Tensor out_m = gat.forward_with_adj(input, adj);
            double loss_m = l2_loss_value(out_m, target);
            (*ap)(i, 0) = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = (*gp)(i, 0);
            double err = relative_error(num, ana);
            max_err_a = max(max_err_a, err);
        }
    }
    total++;   // one check: the WORST element over all heads' a
    if (max_err_a < 1e-2) { cout << "  [PASS] max_err=" << max_err_a << endl; passes++; }
    else { cout << "  [FAIL] max_err=" << max_err_a << endl; }

    // 4) Training step reduces loss
    cout << "\n--- Training step reduces loss ---" << endl;
    GATLayer gat2(in_features, out_features, num_heads, concat_heads);
    Tensor out1 = gat2.forward_with_adj(input, adj);
    double loss_before = l2_loss_value(out1, target);
    Tensor g_loss = l2_loss_grad(out1, target);
    gat2.zero_grad();
    gat2.backward(g_loss, 0.05);
    Tensor out2 = gat2.forward_with_adj(input, adj);
    double loss_after = l2_loss_value(out2, target);
    cout << "  loss_before=" << loss_before << " loss_after=" << loss_after << endl;
    total++;
    if (loss_after < loss_before) { cout << "  [PASS]" << endl; passes++; }
    else { cout << "  [FAIL]" << endl; }

    // 5) Average-heads mode (concat_heads=false)
    cout << "\n--- Average-heads mode (concat_heads=false) ---" << endl;
    GATLayer gat3(in_features, out_features, num_heads, false);
    Tensor out3 = gat3.forward_with_adj(input, adj);
    cout << "  out3 shape: " << out3.rows << "x" << out3.cols << endl;
    total++;
    if (out3.rows == N && out3.cols == out_features) { cout << "  [PASS]" << endl; passes++; }
    else { cout << "  [FAIL] wrong shape" << endl; }

    cout << "\n=== GAT via attention/gat.h: " << passes << "/" << total << " ===" << endl;
    return (passes == total) ? 0 : 1;
}
