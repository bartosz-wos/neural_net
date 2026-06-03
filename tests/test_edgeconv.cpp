#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/edgeconv.h"

using namespace std;

static double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

int main() {
    cout << "=== EdgeConv / DGCNN Tests ===" << endl;
    int total = 0, passed = 0;

    // =================================================================
    // Test 1: EdgeConvLayer forward shape
    // =================================================================
    cout << "\n--- Test 1: EdgeConvLayer forward shape ---\n";
    {
        ++total;
        size_t N = 5, in_f = 3, out_f = 4, k = 2;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * i + 0.3 * f;

        EdgeConvLayer layer(in_f, out_f, k, /*self_loops=*/true);
        Tensor output = layer.forward(input);
        if (output.rows == N && output.cols == out_f) {
            cout << "[PASS] forward output shape = " << output.rows << "x"
                 << output.cols << "\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << out_f << " got "
                 << output.rows << "x" << output.cols << "\n";
        }
    }

    // =================================================================
    // Test 2: output is finite
    // =================================================================
    cout << "\n--- Test 2: EdgeConvLayer output is finite ---\n";
    {
        ++total;
        size_t N = 6, in_f = 4, out_f = 5, k = 3;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = sin(0.2 * i + 0.5 * f);

        EdgeConvLayer layer(in_f, out_f, k);
        Tensor output = layer.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) { cout << "[PASS] all outputs finite\n"; ++passed; }
        else        { cout << "[FAIL] non-finite output\n"; }
    }

    // =================================================================
    // Test 3: zero-input produces deterministic output (sanity)
    // =================================================================
    cout << "\n--- Test 3: zero-input is deterministic ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, out_f = 4, k = 2;
        Tensor input(N, in_f);  // all zeros
        EdgeConvLayer layer(in_f, out_f, k);
        Tensor out1 = layer.forward(input);
        Tensor out2 = layer.forward(input);
        bool same = true;
        for (size_t i = 0; i < out1.rows && same; ++i)
            for (size_t j = 0; j < out1.cols; ++j)
                if (fabs(out1(i, j) - out2(i, j)) > 1e-9) same = false;
        if (same) { cout << "[PASS] deterministic on zero input\n"; ++passed; }
        else      { cout << "[FAIL] not deterministic\n"; }
    }

    // =================================================================
    // Test 4: numerical gradient check on EdgeConvLayer input
    // =================================================================
    cout << "\n--- Test 4: numerical gradient check (EdgeConvLayer input) ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, out_f = 3, k = 4;  // k=N to avoid graph churn
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;

        Tensor target(N, out_f);

        EdgeConvLayer layer(in_f, out_f, k);
        Tensor out = layer.forward(input);
        Tensor grad_output(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < out_f; ++f)
                grad_output(i, f) = out(i, f) - target(i, f);
        layer.zero_grad();
        Tensor grad_x = layer.backward(grad_output, 0.0);

        double eps = 1e-5;
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                input(i, j) = orig + eps;
                Tensor out_p = layer.forward(input);
                double loss_p = 0.0;
                for (size_t r = 0; r < N; ++r)
                    for (size_t c = 0; c < out_f; ++c)
                        loss_p += 0.5 * (out_p(r, c) - target(r, c)) *
                                  (out_p(r, c) - target(r, c));

                input(i, j) = orig - eps;
                Tensor out_m = layer.forward(input);
                double loss_m = 0.0;
                for (size_t r = 0; r < N; ++r)
                    for (size_t c = 0; c < out_f; ++c)
                        loss_m += 0.5 * (out_m(r, c) - target(r, c)) *
                                  (out_m(r, c) - target(r, c));

                double num_grad = (loss_p - loss_m) / (2 * eps);
                input(i, j) = orig;

                double err = relative_error(grad_x(i, j), num_grad);
                max_err = max(max_err, err);
                if (err > 0.01) {
                    cout << "  input[" << i << "][" << j << "]: anal="
                         << grad_x(i, j) << " num=" << num_grad
                         << " rel_err=" << err << "\n";
                }
            }
        }
        if (max_err < 0.01) {
            cout << "[PASS] max rel err = " << max_err << "\n";
            ++passed;
        } else {
            cout << "[FAIL] max rel err = " << max_err << "\n";
        }
    }

    // =================================================================
    // Test 5: forward shape with model
    // =================================================================
    cout << "\n--- Test 5: EdgeConvModel forward shape ---\n";
    {
        ++total;
        size_t N = 6, in_f = 4, hidden = 8, out_f = 3, num_cls = 2;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.05 * i + 0.1 * f;

        EdgeConvModel model(in_f, hidden, out_f, num_cls, /*num_layers=*/2, /*k=*/3);
        Tensor out = model.forward(input);
        if (out.rows == N && out.cols == num_cls) {
            cout << "[PASS] model output = " << out.rows << "x" << out.cols << "\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << num_cls
                 << " got " << out.rows << "x" << out.cols << "\n";
        }
    }

    // =================================================================
    // Test 6: numerical gradient check on EdgeConvModel input
    // Note: EdgeConv's k-NN is recomputed at each forward. To get a
    // meaningful gradient check we use k=N (no truncation, graph
    // stays the same as the full set, no churn).
    // =================================================================
    cout << "\n--- Test 6: numerical gradient check (EdgeConvModel input) ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, hidden = 4, out_f = 4, num_cls = 3, k = 4;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;

        Tensor target(N, num_cls);

        EdgeConvModel model(in_f, hidden, out_f, num_cls, /*num_layers=*/1, k);
        Tensor out = model.forward(input);
        Tensor grad_output(N, num_cls);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < num_cls; ++f)
                grad_output(i, f) = out(i, f) - target(i, f);
        model.zero_grad();
        Tensor grad_x = model.backward(grad_output, 0.0);

        double eps = 1e-5;
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                double orig = input(i, j);

                input(i, j) = orig + eps;
                Tensor out_p = model.forward(input);
                double loss_p = 0.0;
                for (size_t r = 0; r < N; ++r)
                    for (size_t c = 0; c < num_cls; ++c)
                        loss_p += 0.5 * (out_p(r, c) - target(r, c)) *
                                  (out_p(r, c) - target(r, c));

                input(i, j) = orig - eps;
                Tensor out_m = model.forward(input);
                double loss_m = 0.0;
                for (size_t r = 0; r < N; ++r)
                    for (size_t c = 0; c < num_cls; ++c)
                        loss_m += 0.5 * (out_m(r, c) - target(r, c)) *
                                  (out_m(r, c) - target(r, c));

                double num_grad = (loss_p - loss_m) / (2 * eps);
                input(i, j) = orig;

                double err = relative_error(grad_x(i, j), num_grad);
                max_err = max(max_err, err);
                if (err > 0.05) {
                    cout << "  input[" << i << "][" << j << "]: anal="
                         << grad_x(i, j) << " num=" << num_grad
                         << " rel_err=" << err << "\n";
                }
            }
        }
        if (max_err < 0.05) {
            cout << "[PASS] max rel err = " << max_err << "\n";
            ++passed;
        } else {
            cout << "[FAIL] max rel err = " << max_err << "\n";
        }
    }

    // =================================================================
    // Test 7: numerical gradient check on EdgeConvLayer bias (parameters[1])
    // Bias is a cleaner target than weights: it's a pure additive term
    // that doesn't interact with ReLU dead-neuron issues the way
    // weight gradients do near the ReLU kink.
    // =================================================================
    cout << "\n--- Test 7: numerical gradient check (EdgeConvLayer bias) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, out_f = 2, k = 2;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;
        Tensor target(N, out_f);

        EdgeConvLayer layer(in_f, out_f, k);

        Tensor out = layer.forward(input);
        Tensor grad_output(N, out_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < out_f; ++f)
                grad_output(i, f) = out(i, f) - target(i, f);
        layer.zero_grad();
        layer.backward(grad_output, 0.0);

        // gradients()[1] is mlp_fc1_.grad_bias
        double b_orig = layer.parameters()[1]->operator()(0, 0);
        double gb = layer.gradients()[1]->operator()(0, 0);

        double eps = 1e-5;
        layer.parameters()[1]->operator()(0, 0) = b_orig + eps;
        Tensor out_p = layer.forward(input);
        double loss_p = 0.0;
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < out_f; ++c)
                loss_p += 0.5 * (out_p(r, c) - target(r, c)) *
                          (out_p(r, c) - target(r, c));
        layer.parameters()[1]->operator()(0, 0) = b_orig - eps;
        Tensor out_m = layer.forward(input);
        double loss_m = 0.0;
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < out_f; ++c)
                loss_m += 0.5 * (out_m(r, c) - target(r, c)) *
                          (out_m(r, c) - target(r, c));
        layer.parameters()[1]->operator()(0, 0) = b_orig;

        double num_grad = (loss_p - loss_m) / (2 * eps);
        double err = relative_error(gb, num_grad);
        if (err < 0.05) {
            cout << "[PASS] max rel err = " << err << "\n";
            ++passed;
        } else {
            cout << "[FAIL] bias(0,0): anal=" << gb << " num=" << num_grad
                 << " rel_err=" << err << "\n";
        }
    }

    // =================================================================
    // Test 8: EdgeConvModel can do a training step (loss decreases)
    // =================================================================
    cout << "\n--- Test 8: EdgeConvModel training step ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, hidden = 4, out_f = 4, num_cls = 2;
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) + 0.2 * f;
        Tensor target(N, num_cls);
        for (size_t i = 0; i < N; ++i)
            target(i, i % num_cls) = 1.0;

        EdgeConvModel model(in_f, hidden, out_f, num_cls, /*num_layers=*/1, /*k=*/2);

        auto loss_of = [&](const Tensor& y) {
            double s = 0.0;
            for (size_t r = 0; r < y.rows; ++r)
                for (size_t c = 0; c < y.cols; ++c)
                    s += 0.5 * (y(r, c) - target(r, c)) * (y(r, c) - target(r, c));
            return s;
        };

        Tensor out0 = model.forward(input);
        double loss0 = loss_of(out0);

        double lr = 0.01;
        for (int step = 0; step < 30; ++step) {
            Tensor out = model.forward(input);
            Tensor grad_output(N, num_cls);
            for (size_t r = 0; r < N; ++r)
                for (size_t c = 0; c < num_cls; ++c)
                    grad_output(r, c) = out(r, c) - target(r, c);
            model.zero_grad();
            model.backward(grad_output, lr);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward(input);
        double loss1 = loss_of(out1);
        cout << "  loss before=" << loss0 << " after=" << loss1 << "\n";
        if (loss1 < loss0) { cout << "[PASS] loss decreased\n"; ++passed; }
        else               { cout << "[FAIL] loss did not decrease\n"; }
    }

    cout << "\n=== Summary: " << passed << " passed, "
         << (total - passed) << " failed (of " << total << ") ===\n";
    return (passed == total) ? 0 : 1;
}
