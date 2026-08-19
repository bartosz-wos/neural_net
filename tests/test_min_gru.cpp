// MinGRU — Feng et al. 2024 ("Were RNNs All We Needed?")
//
// Tests:
//   1. MinGRUCell forward shape: (T, input_size) -> (T, hidden_size)
//   2. MinGRUCell output is finite (no NaN/Inf) over T=4
//   3. MinGRUCell hand-derived reference (T=1, single hidden unit): sanity
//   4. MinGRUCell input gradient check (T=3)
//   5. MinGRUCell W_g gradient check
//   6. MinGRUCell W_h gradient check
//   7. MinGRUCell b_g gradient check
//   8. MinGRUCell b_h gradient check
//   9. MinGRUCell zero_grad clears all gradients
//  10. MinGRUCell training step reduces loss
//  11. MinGRUCell longer sequence (T=6) gradient check (deeper BPTT)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/min_gru.h"

using namespace std;

static double relative_error(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
}

static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

int main() {
    cout << "=== MinGRU Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Tiny tractable config: T=2, input_size=2, hidden_size=2
    size_t T = 2, input_size = 2, hidden_size = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: MinGRUCell forward shape ---\n";
    {
        ++total;
        Tensor input(T, input_size);
        for (size_t i = 0; i < T * input_size; ++i) input.data[i] = 0.1 * (i + 1);

        MinGRU cell(input_size, hidden_size);
        Tensor output = cell.forward_sequence(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == hidden_size) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << hidden_size << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output is finite (T=4)
    // ------------------------------------------------------------
    cout << "\n--- Test 2: MinGRUCell output is finite (T=4) ---\n";
    {
        ++total;
        size_t T4 = 4;
        Tensor input(T4, input_size);
        for (size_t i = 0; i < T4 * input_size; ++i)
            input.data[i] = 0.1 * (i + 1) - 0.3;

        MinGRU cell(input_size, hidden_size);
        Tensor output = cell.forward_sequence(input);
        bool all_finite = true;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!std::isfinite(output.data[i])) { all_finite = false; break; }
        }
        if (all_finite) {
            cout << "[PASS] all output values finite\n";
            ++passed;
        } else {
            cout << "[FAIL] output contains NaN/Inf\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: hand-derived reference (T=1, single hidden unit)
    //
    // With T=1, h_0 = 0, h_1 = gates_1 * cand_1.
    // gates_1 = sigmoid(W_g[0][0]*x_0 + b_g[0][0])
    // cand_1  = W_h[0][0]*x_0 + b_h[0][0]
    // ------------------------------------------------------------
    cout << "\n--- Test 3: MinGRUCell hand-derived reference (T=1, h=1) ---\n";
    {
        ++total;
        MinGRU cell(1, 1);
        // Force known weights
        cell.W_g().fill(0.0); cell.b_g().fill(0.0);
        cell.W_h().fill(0.0); cell.b_h().fill(0.0);
        cell.W_g()(0,0) = 2.0;  cell.b_g()(0,0) = 0.5;
        cell.W_h()(0,0) = 1.0;  cell.b_h()(0,0) = 0.1;

        Tensor input(1, 1);
        input[0][0] = 0.3;
        Tensor output = cell.forward_sequence(input);
        // gates = sigmoid(2*0.3 + 0.5) = sigmoid(1.1)
        double gates = 1.0 / (1.0 + std::exp(-1.1));
        // cand  = 1*0.3 + 0.1 = 0.4
        // h_1 = gates * 0.4
        double expected = gates * 0.4;
        double got = output[0][0];
        double rel = relative_error(got, expected);
        cout << "  expected: " << expected << "  got: " << got
             << "  rel_err: " << rel << "\n";
        if (rel < 1e-9) {
            cout << "[PASS] hand-derived reference matches\n";
            ++passed;
        } else {
            cout << "[FAIL] hand-derived mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: input gradient check (T=3)
    // ------------------------------------------------------------
    cout << "\n--- Test 4: MinGRUCell input gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T4 = 3;
        Tensor input(T4, input_size);
        for (size_t i = 0; i < T4; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T4, hidden_size);
        for (size_t i = 0; i < T4; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.5 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        Tensor grad_x = cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < T4; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-3) {
            cout << "[PASS] input gradient check (rel_err < 0.1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: W_g gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: MinGRUCell W_g gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T5 = 3;
        Tensor input(T5, input_size);
        for (size_t i = 0; i < T5; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T5, hidden_size);
        for (size_t i = 0; i < T5; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.3 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                double orig = cell.W_g()(i,j);
                cell.W_g()(i,j) = orig + eps;
                cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
                double Lp = l2_loss_value(out_p, target);
                cell.W_g()(i,j) = orig - eps;
                cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
                double Lm = l2_loss_value(out_m, target);
                cell.W_g()(i,j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = cell.grad_W_g()(i,j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-3) {
            cout << "[PASS] W_g gradient check (rel_err < 0.1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_g gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_h gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: MinGRUCell W_h gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T6 = 3;
        Tensor input(T6, input_size);
        for (size_t i = 0; i < T6; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T6, hidden_size);
        for (size_t i = 0; i < T6; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.3 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                double orig = cell.W_h()(i,j);
                cell.W_h()(i,j) = orig + eps;
                cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
                double Lp = l2_loss_value(out_p, target);
                cell.W_h()(i,j) = orig - eps;
                cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
                double Lm = l2_loss_value(out_m, target);
                cell.W_h()(i,j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = cell.grad_W_h()(i,j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-3) {
            cout << "[PASS] W_h gradient check (rel_err < 0.1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W_h gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: b_g gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: MinGRUCell b_g gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T7 = 3;
        Tensor input(T7, input_size);
        for (size_t i = 0; i < T7; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T7, hidden_size);
        for (size_t i = 0; i < T7; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.3 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t j = 0; j < hidden_size; ++j) {
            double orig = cell.b_g()(0,j);
            cell.b_g()(0,j) = orig + eps;
            cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
            double Lp = l2_loss_value(out_p, target);
            cell.b_g()(0,j) = orig - eps;
            cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
            double Lm = l2_loss_value(out_m, target);
            cell.b_g()(0,j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = cell.grad_b_g()(0,j);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-3) {
            cout << "[PASS] b_g gradient check (rel_err < 0.1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] b_g gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: b_h gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: MinGRUCell b_h gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T8 = 3;
        Tensor input(T8, input_size);
        for (size_t i = 0; i < T8; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T8, hidden_size);
        for (size_t i = 0; i < T8; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.3 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t j = 0; j < hidden_size; ++j) {
            double orig = cell.b_h()(0,j);
            cell.b_h()(0,j) = orig + eps;
            cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
            double Lp = l2_loss_value(out_p, target);
            cell.b_h()(0,j) = orig - eps;
            cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
            double Lm = l2_loss_value(out_m, target);
            cell.b_h()(0,j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = cell.grad_b_h()(0,j);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-3) {
            cout << "[PASS] b_h gradient check (rel_err < 0.1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] b_h gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: zero_grad clears all gradients
    // ------------------------------------------------------------
    cout << "\n--- Test 9: MinGRUCell zero_grad clears all gradients ---\n";
    {
        ++total;
        MinGRU cell(input_size, hidden_size);
        Tensor input(T, input_size);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.5;
        Tensor target(T, hidden_size);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.0;
        Tensor out = cell.forward_sequence(input);
        cell.backward(l2_loss_grad(out, target), 0.0);
        cell.zero_grad();
        double sum = 0.0;
        for (Tensor* t : cell.gradients()) {
            for (size_t k = 0; k < t->data.size(); ++k) sum += std::fabs(t->data[k]);
        }
        if (sum == 0.0) {
            cout << "[PASS] zero_grad clears all grads\n";
            ++passed;
        } else {
            cout << "[FAIL] residual gradient sum: " << sum << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: MinGRUCell training reduces loss ---\n";
    {
        ++total;
        size_t Tt = 4;
        Tensor input(Tt, input_size);
        for (size_t i = 0; i < Tt; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 * (i + 1) + 0.05 * k;
        Tensor target(Tt, hidden_size);
        for (size_t i = 0; i < Tt; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.3 + 0.05 * (i + k);

        MinGRU cell(input_size, hidden_size);
        Tensor out0 = cell.forward_sequence(input);
        double L0 = l2_loss_value(out0, target);

        double lr = 0.1;
        for (int step = 0; step < 80; ++step) {
            Tensor out = cell.forward_sequence(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            cell.zero_grad();
            cell.backward(grad_loss, 0.0);
            cell.update_weights(lr);
        }
        Tensor outF = cell.forward_sequence(input);
        double LF = l2_loss_value(outF, target);
        cout << "  L0: " << L0 << "  LF: " << LF << "\n";
        if (LF < L0 * 0.8) {
            cout << "[PASS] loss reduced > 20%\n";
            ++passed;
        } else {
            cout << "[FAIL] loss did not reduce sufficiently\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: longer sequence (T=6) input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 11: MinGRUCell longer sequence input gradient check (T=6) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T11 = 6;
        Tensor input(T11, input_size);
        for (size_t i = 0; i < T11; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.1 + 0.03 * i + 0.02 * k;

        Tensor target(T11, hidden_size);
        for (size_t i = 0; i < T11; ++i)
            for (size_t k = 0; k < hidden_size; ++k)
                target(i, k) = 0.4 + 0.05 * k;

        MinGRU cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        Tensor grad_x = cell.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < T11; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                cell.reset_state(); Tensor out_p = cell.forward_sequence(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                cell.reset_state(); Tensor out_m = cell.forward_sequence(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-2) {
            cout << "[PASS] longer-sequence input grad check (rel_err < 1%)\n";
            ++passed;
        } else {
            cout << "[FAIL] longer-sequence input grad check failed\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, "
         << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
