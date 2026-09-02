// MinLSTM — Feng et al. 2024 ("Were RNNs All We Needed?", https://arxiv.org/abs/2410.01201, Alg. 3)
//
// The minimalist LSTM companion to MinGRU. Two input-only gates (forget + input)
// are NORMALIZED to sum to 1, making the recurrence a convex combination:
//
//   f_t = sigmoid(W_f x_t + b_f)   i_t = sigmoid(W_i x_t + b_i)
//   c_t = W_h x_t + b_h
//   s_t = f_t + i_t ; f'_t = f_t/s_t ; i'_t = i_t/s_t   (f' + i' == 1)
//   h_t = f'_t * h_{t-1} + i'_t * c_t
//
// Tests:
//   1.  Constructor validation (input_dim=0, hidden_size=0 throw)
//   2.  forward_sequence shape (T, input_dim) -> (T, hidden_size)
//   3.  Forward finite for T=4 mixed-sign input
//   4.  Gate normalization invariant f'_t + i'_t == 1 (machine precision)
//   5.  Hand-derived reference (T=1, in=1, h=1)
//   6.  Input gradient FD check (T=3)
//   7.  W_f gradient FD check
//   8.  W_i gradient FD check
//   9.  W_h gradient FD check
//  10.  b_f gradient FD check
//  11.  b_i gradient FD check
//  12.  b_h gradient FD check
//  13.  zero_grad clears all 6 gradients
//  14.  update_weights moves all 6 parameters
//  15.  Training reduces loss over 80 SGD steps
//  16.  Longer sequence (T=6) input gradient FD check
//  17.  Equal-gate symmetry signature (f' == i' == 0.5 -> hand-computed EMA)
//  18.  parameters()/gradients() contract (6 tensors, shapes matched)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/layers/recurrent/min_lstm.h"

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
    for (size_t i = 0; i < output.data.size(); ++i)
        g.data[i] = output.data[i] - target.data[i];
    return g;
}

// Generic FD check for one parameter tensor of a MinLSTM.
static double param_fd_check(size_t input_size, size_t hidden_size,
                             const Tensor& input, const Tensor& target,
                             Tensor& (MinLSTM::*param)(),
                             const Tensor& (MinLSTM::*grad)() const,
                             const char* label) {
    double eps = 1e-5;
    MinLSTM cell(input_size, hidden_size);
    Tensor out = cell.forward_sequence(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    cell.reset_state();
    out = cell.forward_sequence(input);
    grad_loss = l2_loss_grad(out, target);
    cell.backward(grad_loss, 0.0);
    Tensor ana = (cell.*grad)();

    double max_err = 0.0;
    Tensor& P = (cell.*param)();
    for (size_t i = 0; i < P.rows; ++i) {
        for (size_t j = 0; j < P.cols; ++j) {
            double orig = P(i, j);
            P(i, j) = orig + eps;
            cell.reset_state();
            double Lp = l2_loss_value(cell.forward_sequence(input), target);
            P(i, j) = orig - eps;
            cell.reset_state();
            double Lm = l2_loss_value(cell.forward_sequence(input), target);
            P(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            max_err = max(max_err, relative_error(num, ana(i, j)));
        }
    }
    cout << "  " << label << " max_err: " << max_err << "\n";
    return max_err;
}

int main() {
    cout << "=== MinLSTM Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    const size_t input_size = 2, hidden_size = 2;

    // Shared FD fixture: T=3, non-uniform input & target.
    const size_t T3 = 3;
    Tensor fd_input(T3, input_size);
    for (size_t t = 0; t < T3; ++t)
        for (size_t k = 0; k < input_size; ++k)
            fd_input(t, k) = 0.15 + 0.11 * t - 0.07 * k;
    Tensor fd_target(T3, hidden_size);
    for (size_t t = 0; t < T3; ++t)
        for (size_t k = 0; k < hidden_size; ++k)
            fd_target(t, k) = 0.3 - 0.09 * t + 0.06 * k;

    // ------------------------------------------------------------
    cout << "\n--- Test 1: constructor validation ---\n";
    {
        ++total;
        bool threw_in = false, threw_h = false, valid_ok = true;
        try { MinLSTM bad(0, 3); } catch (const std::exception&) { threw_in = true; }
        try { MinLSTM bad(3, 0); } catch (const std::exception&) { threw_h = true; }
        try { MinLSTM good(3, 4); } catch (const std::exception&) { valid_ok = false; }
        cout << "  input_dim=0 threw: " << threw_in
             << "  hidden=0 threw: " << threw_h
             << "  valid ok: " << valid_ok << "\n";
        if (threw_in && threw_h && valid_ok) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 2: forward_sequence shape ---\n";
    {
        ++total;
        Tensor input(T3, input_size);
        for (size_t i = 0; i < T3 * input_size; ++i) input.data[i] = 0.1 * (i + 1);
        MinLSTM cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        cout << "  " << out.rows << "x" << out.cols << "\n";
        if (out.rows == T3 && out.cols == hidden_size) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL] expected " << T3 << "x" << hidden_size << "\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 3: forward finite (T=4, mixed sign) ---\n";
    {
        ++total;
        size_t T4 = 4;
        Tensor input(T4, input_size);
        for (size_t i = 0; i < T4 * input_size; ++i)
            input.data[i] = 0.35 * (i + 1) - 1.1;
        MinLSTM cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(input);
        bool ok = true;
        for (size_t i = 0; i < out.data.size(); ++i)
            if (!std::isfinite(out.data[i])) { ok = false; break; }
        if (ok) { cout << "[PASS] all finite\n"; ++passed; }
        else cout << "[FAIL] NaN/Inf present\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 4: gate normalization invariant f' + i' == 1 ---\n";
    {
        ++total;
        size_t T5 = 5;
        Tensor input(T5, input_size);
        for (size_t t = 0; t < T5; ++t)
            for (size_t k = 0; k < input_size; ++k)
                input(t, k) = 0.9 * std::sin(1.7 * t + 0.5 * k) - 0.3;
        MinLSTM cell(input_size, hidden_size);
        cell.forward_sequence(input);
        const Tensor& fn = cell.last_f_norm();
        const Tensor& in = cell.last_i_norm();
        double max_err = 0.0;
        bool shape_ok = (fn.rows == T5 && fn.cols == hidden_size &&
                         in.rows == T5 && in.cols == hidden_size);
        bool range_ok = true;
        if (shape_ok) {
            for (size_t t = 0; t < T5; ++t)
                for (size_t j = 0; j < hidden_size; ++j) {
                    max_err = max(max_err, fabs(fn(t, j) + in(t, j) - 1.0));
                    if (fn(t, j) <= 0.0 || fn(t, j) >= 1.0) range_ok = false;
                    if (in(t, j) <= 0.0 || in(t, j) >= 1.0) range_ok = false;
                }
        }
        cout << "  shape_ok: " << shape_ok << "  strictly in (0,1): " << range_ok
             << "  max |f'+i'-1|: " << max_err << "\n";
        if (shape_ok && range_ok && max_err < 1e-14) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 5: hand-derived reference (T=1, in=1, h=1) ---\n";
    {
        ++total;
        MinLSTM cell(1, 1);
        cell.W_f().fill(0.0); cell.b_f().fill(0.0);
        cell.W_i().fill(0.0); cell.b_i().fill(0.0);
        cell.W_h().fill(0.0); cell.b_h().fill(0.0);
        cell.W_f()(0, 0) = 2.0;  cell.b_f()(0, 0) = 0.5;
        cell.W_i()(0, 0) = -1.0; cell.b_i()(0, 0) = 0.25;
        cell.W_h()(0, 0) = 1.5;  cell.b_h()(0, 0) = -0.2;

        Tensor input(1, 1);
        input(0, 0) = 0.4;
        Tensor out = cell.forward_sequence(input);

        // f = sigmoid(2*0.4 + 0.5) = sigmoid(1.3)
        // i = sigmoid(-1*0.4 + 0.25) = sigmoid(-0.15)
        // c = 1.5*0.4 - 0.2 = 0.4
        // h_0 = 0 so h_1 = i' * c = (i/(f+i)) * 0.4
        double f = 1.0 / (1.0 + std::exp(-1.3));
        double ii = 1.0 / (1.0 + std::exp(0.15));
        double c = 0.4;
        double expected = (ii / (f + ii)) * c;
        double err = relative_error(out(0, 0), expected);
        cout << "  actual: " << setprecision(12) << out(0, 0)
             << "  expected: " << expected << "  rel_err: " << err << "\n";
        if (err < 1e-12) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 6: input gradient FD check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        MinLSTM cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(fd_input);
        Tensor grad_loss = l2_loss_grad(out, fd_target);
        cell.zero_grad();
        cell.reset_state();
        out = cell.forward_sequence(fd_input);
        grad_loss = l2_loss_grad(out, fd_target);
        Tensor grad_x = cell.backward(grad_loss, 0.0);

        Tensor input = fd_input;
        double max_err = 0.0;
        for (size_t t = 0; t < T3; ++t)
            for (size_t j = 0; j < input_size; ++j) {
                double orig = input(t, j);
                input(t, j) = orig + eps;
                cell.reset_state();
                double Lp = l2_loss_value(cell.forward_sequence(input), fd_target);
                input(t, j) = orig - eps;
                cell.reset_state();
                double Lm = l2_loss_value(cell.forward_sequence(input), fd_target);
                input(t, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                max_err = max(max_err, relative_error(num, grad_x(t, j)));
            }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-6) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    struct PCase { const char* name; Tensor& (MinLSTM::*p)(); const Tensor& (MinLSTM::*g)() const; };
    PCase cases[] = {
        {"W_f", &MinLSTM::W_f, &MinLSTM::grad_W_f},
        {"W_i", &MinLSTM::W_i, &MinLSTM::grad_W_i},
        {"W_h", &MinLSTM::W_h, &MinLSTM::grad_W_h},
        {"b_f", &MinLSTM::b_f, &MinLSTM::grad_b_f},
        {"b_i", &MinLSTM::b_i, &MinLSTM::grad_b_i},
        {"b_h", &MinLSTM::b_h, &MinLSTM::grad_b_h},
    };
    for (const auto& pc : cases) {
        cout << "\n--- Test: " << pc.name << " gradient FD check ---\n";
        ++total;
        double err = param_fd_check(input_size, hidden_size, fd_input, fd_target,
                                    pc.p, pc.g, pc.name);
        if (err < 1e-6) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 13: zero_grad clears all 6 gradients ---\n";
    {
        ++total;
        MinLSTM cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(fd_input);
        cell.backward(l2_loss_grad(out, fd_target), 0.0);
        double before = 0.0;
        for (Tensor* g : cell.gradients())
            for (double v : g->data) before += fabs(v);
        cell.zero_grad();
        double after = 0.0;
        for (Tensor* g : cell.gradients())
            for (double v : g->data) after += fabs(v);
        cout << "  |g| before: " << before << "  after: " << after << "\n";
        if (before > 1e-8 && after == 0.0) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 14: update_weights moves all 6 parameters ---\n";
    {
        ++total;
        MinLSTM cell(input_size, hidden_size);
        Tensor out = cell.forward_sequence(fd_input);
        cell.zero_grad();
        cell.reset_state();
        out = cell.forward_sequence(fd_input);
        cell.backward(l2_loss_grad(out, fd_target), 0.0);
        vector<Tensor> snap;
        for (Tensor* p : cell.parameters()) snap.push_back(*p);
        cell.update_weights(0.5);
        size_t moved = 0;
        auto params = cell.parameters();
        for (size_t k = 0; k < params.size(); ++k) {
            double d = 0.0;
            for (size_t m = 0; m < params[k]->data.size(); ++m)
                d += fabs(params[k]->data[m] - snap[k].data[m]);
            if (d > 1e-12) ++moved;
        }
        cout << "  moved: " << moved << " / " << params.size() << "\n";
        if (moved == params.size()) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 15: training reduces loss (80 SGD steps) ---\n";
    {
        ++total;
        size_t T = 4;
        Tensor input(T, input_size);
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < input_size; ++k)
                input(t, k) = 0.2 + 0.1 * t - 0.05 * k;
        Tensor target(T, hidden_size);
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < hidden_size; ++k)
                target(t, k) = 0.5 - 0.1 * k;

        MinLSTM cell(input_size, hidden_size);
        cell.reset_state();
        double L0 = l2_loss_value(cell.forward_sequence(input), target);
        double LF = L0;
        for (int step = 0; step < 80; ++step) {
            cell.zero_grad();
            cell.reset_state();
            Tensor out = cell.forward_sequence(input);
            LF = l2_loss_value(out, target);
            cell.backward(l2_loss_grad(out, target), 0.0);
            cell.update_weights(0.05);
        }
        cell.zero_grad();
        cell.reset_state();
        LF = l2_loss_value(cell.forward_sequence(input), target);
        cout << "  L0: " << L0 << "  LF: " << LF << "\n";
        if (LF < L0 * 0.8) { cout << "[PASS] loss reduced > 20%\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 16: longer sequence (T=6) input gradient FD check ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T6 = 6;
        Tensor input(T6, input_size);
        for (size_t t = 0; t < T6; ++t)
            for (size_t k = 0; k < input_size; ++k)
                input(t, k) = 0.1 + 0.04 * t + 0.03 * k;
        Tensor target(T6, hidden_size);
        for (size_t t = 0; t < T6; ++t)
            for (size_t k = 0; k < hidden_size; ++k)
                target(t, k) = 0.4 + 0.05 * k;

        MinLSTM cell(input_size, hidden_size);
        cell.reset_state();
        Tensor out = cell.forward_sequence(input);
        cell.zero_grad();
        cell.reset_state();
        out = cell.forward_sequence(input);
        Tensor grad_x = cell.backward(l2_loss_grad(out, target), 0.0);

        double max_err = 0.0;
        for (size_t t = 0; t < T6; ++t)
            for (size_t j = 0; j < input_size; ++j) {
                double orig = input(t, j);
                input(t, j) = orig + eps;
                cell.reset_state();
                double Lp = l2_loss_value(cell.forward_sequence(input), target);
                input(t, j) = orig - eps;
                cell.reset_state();
                double Lm = l2_loss_value(cell.forward_sequence(input), target);
                input(t, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                max_err = max(max_err, relative_error(num, grad_x(t, j)));
            }
        cout << "  max_err: " << max_err << "\n";
        if (max_err < 1e-6) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    // Signature test: with W_f == W_i and b_f == b_i, both gates are equal,
    // so f' == i' == 0.5 exactly and h_t = 0.5*h_{t-1} + 0.5*c_t.
    // Distinguishes MinLSTM's normalized dual gate from a single-gate reduction.
    // ------------------------------------------------------------
    cout << "\n--- Test 17: equal-gate symmetry signature ---\n";
    {
        ++total;
        size_t T = 4;
        MinLSTM cell(input_size, hidden_size);
        // Force W_f == W_i, b_f == b_i (arbitrary asymmetric values).
        for (size_t i = 0; i < input_size; ++i)
            for (size_t j = 0; j < hidden_size; ++j) {
                double v = 0.7 - 0.31 * i + 0.19 * j;
                cell.W_f()(i, j) = v;
                cell.W_i()(i, j) = v;
            }
        for (size_t j = 0; j < hidden_size; ++j) {
            cell.b_f()(0, j) = 0.4 - 0.2 * j;
            cell.b_i()(0, j) = 0.4 - 0.2 * j;
        }
        Tensor input(T, input_size);
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < input_size; ++k)
                input(t, k) = 0.5 * std::cos(0.9 * t + 0.4 * k);

        cell.reset_state();
        Tensor out = cell.forward_sequence(input);

        // Reference EMA: h_t = 0.5*h_{t-1} + 0.5*c_t with c_t = W_h x_t + b_h.
        double max_err = 0.0;
        vector<double> h(hidden_size, 0.0);
        for (size_t t = 0; t < T; ++t)
            for (size_t j = 0; j < hidden_size; ++j) {
                double c = cell.b_h()(0, j);
                for (size_t k = 0; k < input_size; ++k)
                    c += input(t, k) * cell.W_h()(k, j);
                h[j] = 0.5 * h[j] + 0.5 * c;
                max_err = max(max_err, relative_error(out(t, j), h[j]));
            }
        // Also verify the normalized gates are exactly 0.5.
        double gate_err = 0.0;
        for (size_t t = 0; t < T; ++t)
            for (size_t j = 0; j < hidden_size; ++j) {
                gate_err = max(gate_err, fabs(cell.last_f_norm()(t, j) - 0.5));
                gate_err = max(gate_err, fabs(cell.last_i_norm()(t, j) - 0.5));
            }
        cout << "  EMA max_err: " << max_err << "  gate dev from 0.5: " << gate_err << "\n";
        if (max_err < 1e-12 && gate_err < 1e-14) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    // ------------------------------------------------------------
    cout << "\n--- Test 18: parameters()/gradients() contract ---\n";
    {
        ++total;
        MinLSTM cell(3, 4);
        auto ps = cell.parameters();
        auto gs = cell.gradients();
        bool ok = (ps.size() == 6 && gs.size() == 6);
        if (ok)
            for (size_t k = 0; k < ps.size(); ++k)
                if (ps[k]->rows != gs[k]->rows || ps[k]->cols != gs[k]->cols) ok = false;
        bool shapes_ok = ok &&
            cell.W_f().rows == 3 && cell.W_f().cols == 4 &&
            cell.W_i().rows == 3 && cell.W_i().cols == 4 &&
            cell.W_h().rows == 3 && cell.W_h().cols == 4 &&
            cell.b_f().rows == 1 && cell.b_f().cols == 4 &&
            cell.b_i().rows == 1 && cell.b_i().cols == 4 &&
            cell.b_h().rows == 1 && cell.b_h().cols == 4;
        cout << "  params: " << ps.size() << "  grads: " << gs.size()
             << "  shapes_ok: " << shapes_ok << "  name: " << cell.name() << "\n";
        if (ok && shapes_ok) { cout << "[PASS]\n"; ++passed; }
        else cout << "[FAIL]\n";
    }

    cout << "\n=== Summary: " << passed << " passed, "
         << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
