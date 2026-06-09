// RetNet — Sun et al. 2023
//   "Retentive Network: A Successor to Transformer for Large Language Models"
//
// Tests:
//   1. RetNetRetention forward shape: (T, d) -> (T, d)
//   2. RetNetRetention output is finite
//   3. RetNetRetention input gradient check  (numerical vs analytical)
//   4. RetNetRetention W_Q gradient check    (tests rotation + dS q-path)
//   5. RetNetRetention W_K gradient check    (tests outer-product k-tensor path)
//   6. RetNetRetention W_V gradient check    (tests outer-product v-tensor path)
//   7. RetNetRetention W_O gradient check    (output projection)
//   8. RetNetRetention gamma_raw gradient check (decay; tests recurrence carrier)
//   9. RetNetRetention training step reduces loss
//  10. RetNetRetention parameters()/gradients() shape consistency
//  11. RetNetRetention multi-head forward+backward (H=2, d=4, D=2)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/retnet.h"

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

template <typename LayerType, typename ParamPtr>
static Tensor numerical_grad_param(
    LayerType& layer,
    const Tensor& input,
    const Tensor& target,
    ParamPtr param,
    double eps = 1e-5
) {
    size_t R = param->rows, C = param->cols;
    Tensor grad(R, C);
    double base_loss = l2_loss_value(layer.forward(input), target);
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            double orig = (*param)(i, j);
            (*param)(i, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            (*param)(i, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            (*param)(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    (void)base_loss;
    return grad;
}

static Tensor numerical_grad_input(
    RetNetRetention& layer,
    const Tensor& input,
    const Tensor& target,
    double eps = 1e-5
) {
    size_t R = input.rows, C = input.cols;
    Tensor grad(R, C);
    Tensor work = input.clone();
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            double orig = work(i, j);
            work(i, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(work), target);
            work(i, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(work), target);
            work(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor analytical_input_grad(
    RetNetRetention& layer,
    const Tensor& input,
    const Tensor& target
) {
    Tensor output = layer.forward(input);
    Tensor d_out = l2_loss_grad(output, target);
    return layer.backward(d_out, 0.0);
}

static double avg_rel_err(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1.0;
    double sum = 0.0;
    size_t n = a.data.size();
    for (size_t i = 0; i < n; ++i) sum += relative_error(a.data[i], b.data[i]);
    return sum / (double)n;
}
static double max_rel_err(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1.0;
    double m = 0.0;
    size_t n = a.data.size();
    for (size_t i = 0; i < n; ++i) m = max(m, relative_error(a.data[i], b.data[i]));
    return m;
}

int main() {
    cout << "=== RetNet Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small, tractable config: T=3, d=4, num_heads=1, head_dim=4
    size_t T = 3, d = 4, H = 1;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: RetNetRetention forward shape (T=3, d=4, H=1) ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        RetNetRetention layer(d, H);
        Tensor output = layer.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == d) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << d << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: RetNetRetention output is finite (T=5, d=4) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        RetNetRetention layer(d, H);
        Tensor output = layer.forward(input);
        bool finite = true;
        for (auto v : output.data) {
            if (!std::isfinite(v)) { finite = false; break; }
        }
        if (finite) {
            cout << "[PASS] all output values finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: RetNetRetention input gradient check (T=3, d=4) ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor ana = analytical_input_grad(layer, input, target);
        Tensor num = numerical_grad_input(layer, input, target);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] input gradient matches numerical (avg < 1e-2, max < 5e-2)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: W_Q gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: RetNetRetention W_Q gradient check ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_Q.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_Q.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_Q gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_Q gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: W_K gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: RetNetRetention W_K gradient check ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_K.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_K.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_K gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_K gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_V gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: RetNetRetention W_V gradient check ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_V.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_V.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_V gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_V gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: W_O gradient check (output projection)
    // ------------------------------------------------------------
    cout << "\n--- Test 7: RetNetRetention W_O gradient check ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_O.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_O.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_O gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_O gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: gamma_raw gradient check (decay)
    // ------------------------------------------------------------
    cout << "\n--- Test 8: RetNetRetention gamma_raw gradient check ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d, H);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.grad_gamma_raw_.clone();
        Tensor num(1, d);
        double eps = 1e-4;
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.gamma_raw(0, j);
            layer.gamma_raw(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.gamma_raw(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.gamma_raw(0, j) = orig;
            num(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] gamma_raw gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] gamma_raw gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 9: RetNetRetention training step reduces loss ---\n";
    {
        ++total;
        size_t T3 = 4;
        Tensor input(T3, d);
        for (size_t i = 0; i < T3; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i + 0.05 * j;
        Tensor target(T3, d);
        for (size_t i = 0; i < T3; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.2 * i - 0.1;

        RetNetRetention layer(d, H);
        double lr = 0.05;
        double first_loss = 0.0, last_loss = 0.0;
        for (int step = 0; step < 30; ++step) {
            Tensor out = layer.forward(input);
            double loss = l2_loss_value(out, target);
            if (step == 0) first_loss = loss;
            if (step == 29) last_loss = loss;
            Tensor d_out = l2_loss_grad(out, target);
            layer.backward(d_out, lr);
            layer.update_weights(lr);
            layer.zero_grad();
        }
        cout << "first_loss = " << first_loss << "  last_loss = " << last_loss << "\n";
        if (last_loss < first_loss * 0.95) {
            cout << "[PASS] training step decreases loss ("
                 << (first_loss - last_loss) / first_loss * 100.0 << "% reduction)\n";
            ++passed;
        } else {
            cout << "[FAIL] loss did not decrease enough\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: parameters()/gradients() shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 10: RetNetRetention parameters()/gradients() shape consistency ---\n";
    {
        ++total;
        RetNetRetention layer(d, H);
        auto params = layer.parameters();
        auto grads  = layer.gradients();
        cout << "params count: " << params.size() << ", grads count: " << grads.size() << "\n";
        bool ok = true;
        if (params.size() != grads.size()) ok = false;
        for (size_t i = 0; i < params.size() && ok; ++i) {
            if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
                cout << "  MISMATCH at i=" << i << ": param "
                     << params[i]->rows << "x" << params[i]->cols
                     << " vs grad " << grads[i]->rows << "x" << grads[i]->cols << "\n";
                ok = false;
            }
        }
        if (ok) {
            cout << "[PASS] all " << params.size() << " param/grad pairs shape-matched\n";
            ++passed;
        } else {
            cout << "[FAIL] param/grad shape mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: multi-head (H=2, d=4, D=2) input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 11: RetNetRetention multi-head (H=2, d=4, D=2) input gradient check ---\n";
    {
        ++total;
        size_t d2 = 4, H2 = 2, T2 = 3;
        Tensor input(T2, d2);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d2; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;
        Tensor target(T2, d2);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d2; ++j)
                target(i, j) = 0.2 * cos(0.7 * i) + 0.1 * j;

        RetNetRetention layer(d2, H2);
        Tensor ana = analytical_input_grad(layer, input, target);
        Tensor num = numerical_grad_input(layer, input, target);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] multi-head input gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] multi-head input gradient mismatch\n";
        }
    }

    cout << "\n=== Total: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
