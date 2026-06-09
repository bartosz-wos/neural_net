// RWKV — Peng et al. 2023
//   "RWKV: Reinventing RNNs for the Transformer Era"
//
// Tests:
//   1. RWKVTimeMix forward shape: (T, d) -> (T, d)
//   2. RWKVTimeMix output is finite
//   3. RWKVTimeMix input gradient check  (numerical vs analytical)
//   4. RWKVTimeMix W_r gradient check    (receptance projection)
//   5. RWKVTimeMix W_k gradient check    (key projection; tests WKV recurrence + bonus paths)
//   6. RWKVTimeMix W_v gradient check    (value projection; tests the kv contribution)
//   7. RWKVTimeMix log_w gradient check  (decay; tests the a[i]*p_{t-1}[i] recurrence carrier)
//   8. RWKVTimeMix u gradient check      (bonus; tests the (exp(u)-1)*kv diagonal term)
//   9. RWKVTimeMix mu_r/k/v gradient check (token-shift mixing coefficients)
//  10. RWKVTimeMix training step reduces loss
//  11. RWKVTimeMix parameters()/gradients() shape consistency

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/rwkv.h"

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

// Numerical gradient for a layer: perturb each element of `param` and measure
// loss change. Returns a tensor of the same shape as `param`.
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

// Numerical gradient for the layer INPUT. Operates on a working copy so the
// caller's tensor is not mutated.
static Tensor numerical_grad_input(
    RWKVTimeMix& layer,
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

// Analytical input gradient via forward+backward. Returns grad_input.
// Caller must pass in the original input — we will use it for forward.
static Tensor analytical_input_grad(
    RWKVTimeMix& layer,
    const Tensor& input,
    const Tensor& target
) {
    Tensor output = layer.forward(input);
    Tensor d_out = l2_loss_grad(output, target);
    return layer.backward(d_out, 0.0);
}

// Average rel-err over all elements between two tensors of the same shape.
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
    cout << "=== RWKV Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small, tractable config: T=3, d=2
    size_t T = 3, d = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: RWKVTimeMix forward shape (T=3, d=2) ---\n";
    {
        ++total;
        Tensor input(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        RWKVTimeMix layer(d);
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
    cout << "\n--- Test 2: RWKVTimeMix output is finite (T=5) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        RWKVTimeMix layer(d);
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
    cout << "\n--- Test 3: RWKVTimeMix input gradient check (T=3, d=2) ---\n";
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

        RWKVTimeMix layer(d);
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
    // Test 4: W_r gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: RWKVTimeMix W_r (receptance) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        // Analytical: do forward+backward to populate grad_weights
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_r.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_r.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_r gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_r gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: W_k gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: RWKVTimeMix W_k (key) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_k.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_k.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_k gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_k gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_v gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: RWKVTimeMix W_v (value) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.W_v.grad_weights.clone();
        Tensor num = numerical_grad_param(layer, input, target, &layer.W_v.weights, 1e-4);
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] W_v gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] W_v gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: log_w gradient check (decay)
    // ------------------------------------------------------------
    cout << "\n--- Test 7: RWKVTimeMix log_w (decay) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.grad_log_w_.clone();
        // Numerical: perturb log_w (1, d)
        Tensor num(1, d);
        double eps = 1e-4;
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.log_w(0, j);
            layer.log_w(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.log_w(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.log_w(0, j) = orig;
            num(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] log_w gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] log_w gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: u gradient check (bonus)
    // ------------------------------------------------------------
    cout << "\n--- Test 8: RWKVTimeMix u (bonus) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        Tensor ana = layer.grad_u_.clone();
        Tensor num(1, d);
        double eps = 1e-4;
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.u(0, j);
            layer.u(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.u(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.u(0, j) = orig;
            num(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg = avg_rel_err(ana, num);
        double mx  = max_rel_err(ana, num);
        cout << "avg_rel_err = " << scientific << setprecision(2) << avg
             << ", max_rel_err = " << mx << "\n";
        if (avg < 1e-2 && mx < 5e-2) {
            cout << "[PASS] u gradient matches numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] u gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: mu_r, mu_k, mu_v gradient check (token-shift)
    // ------------------------------------------------------------
    cout << "\n--- Test 9: RWKVTimeMix mu_r/k/v (token-shift) gradient check ---\n";
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

        RWKVTimeMix layer(d);
        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);
        // Check mu_r
        Tensor num_r(1, d);
        double eps = 1e-4;
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.mu_r(0, j);
            layer.mu_r(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.mu_r(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.mu_r(0, j) = orig;
            num_r(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg_r = avg_rel_err(layer.grad_mu_r_, num_r);
        double mx_r  = max_rel_err(layer.grad_mu_r_, num_r);
        // Check mu_k
        Tensor num_k(1, d);
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.mu_k(0, j);
            layer.mu_k(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.mu_k(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.mu_k(0, j) = orig;
            num_k(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg_k = avg_rel_err(layer.grad_mu_k_, num_k);
        double mx_k  = max_rel_err(layer.grad_mu_k_, num_k);
        // Check mu_v
        Tensor num_v(1, d);
        for (size_t j = 0; j < d; ++j) {
            double orig = layer.mu_v(0, j);
            layer.mu_v(0, j) = orig + eps;
            double lp = l2_loss_value(layer.forward(input), target);
            layer.mu_v(0, j) = orig - eps;
            double lm = l2_loss_value(layer.forward(input), target);
            layer.mu_v(0, j) = orig;
            num_v(0, j) = (lp - lm) / (2.0 * eps);
        }
        double avg_v = avg_rel_err(layer.grad_mu_v_, num_v);
        double mx_v  = max_rel_err(layer.grad_mu_v_, num_v);

        cout << "mu_r: avg=" << scientific << setprecision(2) << avg_r
             << " max=" << mx_r << "\n";
        cout << "mu_k: avg=" << avg_k << " max=" << mx_k << "\n";
        cout << "mu_v: avg=" << avg_v << " max=" << mx_v << "\n";

        bool ok = (avg_r < 1e-2 && mx_r < 5e-2) &&
                  (avg_k < 1e-2 && mx_k < 5e-2) &&
                  (avg_v < 1e-2 && mx_v < 5e-2);
        if (ok) {
            cout << "[PASS] mu_r, mu_k, mu_v gradients match numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] mu gradient mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: RWKVTimeMix training step reduces loss ---\n";
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

        RWKVTimeMix layer(d);
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
    // Test 11: parameters()/gradients() shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 11: RWKVTimeMix parameters()/gradients() shape consistency ---\n";
    {
        ++total;
        RWKVTimeMix layer(d);
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

    cout << "\n=== Total: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
