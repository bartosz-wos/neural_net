#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/pna.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

// =============================================================
// Helper: small 5-node graph (chain + cross edges)
//  0 -- 1 -- 2 -- 3
//       |         |
//       +--- 4 ---+
// =============================================================
Tensor make_adj_5nodes() {
    Tensor adj(5, 5);
    adj(0, 1) = 1; adj(1, 0) = 1;
    adj(1, 2) = 1; adj(2, 1) = 1;
    adj(2, 3) = 1; adj(3, 2) = 1;
    adj(3, 4) = 1; adj(4, 3) = 1;
    adj(1, 4) = 1; adj(4, 1) = 1;
    return adj;
}

// L2 loss: 0.5 * sum (y - target)^2
Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g = output - target;
    return g;
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
    cout << "=== PNA Tests ===" << endl;
    int total = 0, passed = 0;

    // ================================================================
    // Test 1: PNALayer forward shape
    // ================================================================
    cout << "\n--- Test 1: PNALayer forward shape ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, out_f = 4;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;
        PNALayer layer(in_f, out_f, 1.0);
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
    // Test 2: PNALayer output is finite
    // ================================================================
    cout << "\n--- Test 2: PNALayer output is finite ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, out_f = 4;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.5 * sin(0.1 * i) + 0.3 * f;
        PNALayer layer(in_f, out_f, 1.0);
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
    // Test 3: PNALayer handles isolated node (no neighbours) gracefully
    // ================================================================
    cout << "\n--- Test 3: PNALayer with isolated node ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, out_f = 3;
        // 4 nodes, node 0 is isolated, others form a triangle
        Tensor adj(N, N);
        adj(1, 2) = 1; adj(2, 1) = 1;
        adj(2, 3) = 1; adj(3, 2) = 1;
        adj(1, 3) = 1; adj(3, 1) = 1;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * i + 0.5 * f;
        PNALayer layer(in_f, out_f, 1.0);
        Tensor output = layer.forward_with_adj(input, adj);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite && output.rows == N && output.cols == out_f) {
            cout << "[PASS] isolated node handled, all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] isolated node produced non-finite or wrong shape\n";
        }
    }

    // ================================================================
    // Test 4: PNALayer numerical gradient (L2 loss) — post_agg weights
    // ================================================================
    cout << "\n--- Test 4: PNALayer L2 loss gradient check (post_agg W) ---\n";
    {
        ++total;
        size_t in_f = 2, out_f = 2;
        Tensor adj = make_adj_5nodes();
        // Truncate adj to first 4 nodes
        Tensor adj4(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                adj4(i, j) = adj(i, j);
        Tensor input(4, in_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.3 + 0.1 * i + 0.05 * f;
        Tensor target(4, out_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < out_f; ++f)
                target(i, f) = 0.1 * f - 0.05 * i;

        PNALayer layer(in_f, out_f, 1.0);
        // Get analytical gradient for W[0][0]
        Tensor out = layer.forward_with_adj(input, adj4);
        Tensor grad_loss = l2_loss_grad(out, target);
        layer.backward(grad_loss, 0.0);  // learning_rate=0; just compute grads
        Tensor* Wp = layer.parameters()[0];  // post_agg_.weights
        Tensor analytical_W00 = (*Wp).clone();
        // Now numerical gradient
        double eps = 1e-5;
        double orig = (*Wp)(0, 0);
        // f(x + eps)
        (*Wp)(0, 0) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input, adj4);
        double loss_p = l2_loss_value(out_p, target);
        // f(x - eps)
        (*Wp)(0, 0) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input, adj4);
        double loss_m = l2_loss_value(out_m, target);
        (*Wp)(0, 0) = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        // Pull analytical gradient on weights — but the standard pattern uses update_weights.
        // We need to read grad_weights which gets accumulated by backward.
        Tensor* Gp = layer.gradients()[0];
        double ana_grad = (*Gp)(0, 0);
        double rel = relative_error(num_grad, ana_grad);
        cout << "W[0][0] analytical: " << ana_grad << ", numerical: " << num_grad
             << ", rel_err: " << rel << "\n";
        if (rel < 1e-2) {
            cout << "[PASS] post-agg weight gradient close to numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] rel_err too high: " << rel << "\n";
        }
    }

    // ================================================================
    // Test 5: PNALayer L2 loss gradient check — input gradient
    // ================================================================
    cout << "\n--- Test 5: PNALayer L2 loss gradient check (input) ---\n";
    {
        ++total;
        size_t in_f = 2, out_f = 2;
        Tensor adj4(4, 4);
        Tensor adj = make_adj_5nodes();
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                adj4(i, j) = adj(i, j);
        Tensor input(4, in_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.2 + 0.1 * i - 0.05 * f;
        Tensor target(4, out_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < out_f; ++f)
                target(i, f) = 0.1 * i - 0.05 * f;

        PNALayer layer(in_f, out_f, 1.0);
        Tensor out = layer.forward_with_adj(input, adj4);
        Tensor grad_loss = l2_loss_grad(out, target);
        Tensor grad_input = layer.backward(grad_loss, 0.0);

        double eps = 1e-5;
        // Check grad_input[0][0]
        size_t ri = 0, rf = 0;
        double orig = input(ri, rf);
        input(ri, rf) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input, adj4);
        double loss_p = l2_loss_value(out_p, target);
        input(ri, rf) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input, adj4);
        double loss_m = l2_loss_value(out_m, target);
        input(ri, rf) = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = grad_input(ri, rf);
        double rel = relative_error(num_grad, ana_grad);
        cout << "input[" << ri << "][" << rf << "] analytical: " << ana_grad
             << ", numerical: " << num_grad << ", rel_err: " << rel << "\n";
        if (rel < 5e-2) {  // PNA has many non-smooth operations; loosen bound
            cout << "[PASS] input gradient close to numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] rel_err too high: " << rel << "\n";
        }
    }

    // ================================================================
    // Test 6: PNAModel forward shape
    // ================================================================
    cout << "\n--- Test 6: PNAModel forward shape ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, hidden = 4, out_f = 2;
        Tensor adj = make_adj_5nodes();
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 + 0.05 * i + 0.1 * f;
        PNAModel model(N, in_f, hidden, out_f, 2, 1.0);
        Tensor output = model.forward_with_adj(input, adj);
        cout << "Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == N && output.cols == out_f) {
            cout << "[PASS] PNAModel output shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << out_f << "\n";
        }
    }

    // ================================================================
    // Test 7: PNAModel L2 loss gradient check — input
    // ================================================================
    cout << "\n--- Test 7: PNAModel L2 loss gradient check (input) ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3, out_f = 2;
        Tensor adj4(4, 4);
        Tensor adj = make_adj_5nodes();
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                adj4(i, j) = adj(i, j);
        Tensor input(4, in_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.2 + 0.1 * i - 0.05 * f;
        Tensor target(4, out_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < out_f; ++f)
                target(i, f) = 0.1 * i - 0.05 * f;
        PNAModel model(N, in_f, hidden, out_f, 2, 1.0);
        Tensor out = model.forward_with_adj(input, adj4);
        Tensor grad_loss = l2_loss_grad(out, target);
        Tensor grad_input = model.backward(grad_loss, 0.0);
        double eps = 1e-5;
        size_t ri = 1, rf = 0;
        double orig = input(ri, rf);
        input(ri, rf) = orig + eps;
        Tensor out_p = model.forward_with_adj(input, adj4);
        double loss_p = l2_loss_value(out_p, target);
        input(ri, rf) = orig - eps;
        Tensor out_m = model.forward_with_adj(input, adj4);
        double loss_m = l2_loss_value(out_m, target);
        input(ri, rf) = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = grad_input(ri, rf);
        double rel = relative_error(num_grad, ana_grad);
        cout << "input[" << ri << "][" << rf << "] analytical: " << ana_grad
             << ", numerical: " << num_grad << ", rel_err: " << rel << "\n";
        if (rel < 5e-2) {
            cout << "[PASS] PNAModel input gradient close to numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] rel_err too high: " << rel << "\n";
        }
    }

    // ================================================================
    // Test 8: PNAModel training step reduces L2 loss
    // ================================================================
    cout << "\n--- Test 8: PNAModel training step reduces loss ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 4, out_f = 2;
        Tensor adj4(4, 4);
        Tensor adj = make_adj_5nodes();
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                adj4(i, j) = adj(i, j);
        Tensor input(4, in_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * i + 0.2 * f;
        Tensor target(4, out_f);
        for (size_t i = 0; i < 4; ++i)
            for (size_t f = 0; f < out_f; ++f)
                target(i, f) = 0.1 * (f + 1) - 0.05 * i;
        PNAModel model(N, in_f, hidden, out_f, 2, 1.0);
        Tensor out0 = model.forward_with_adj(input, adj4);
        double loss0 = l2_loss_value(out0, target);
        double lr = 0.01;
        for (int step = 0; step < 30; ++step) {
            Tensor out = model.forward_with_adj(input, adj4);
            Tensor grad = l2_loss_grad(out, target);
            model.zero_grad();
            model.backward(grad, lr);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward_with_adj(input, adj4);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] loss decreased after training\n";
            ++passed;
        } else {
            cout << "[FAIL] loss did not decrease\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed (of " << total << ") ===" << endl;
    return (passed == total) ? 0 : 1;
}
