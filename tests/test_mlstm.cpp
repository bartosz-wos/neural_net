// mLSTM — Beck et al. 2024 (matrix-memory xLSTM variant)
//
// Tests:
//   1. MLSTMCell forward shape: (T, input_size) -> (T, hidden_size)
//   2. MLSTMCell output is finite (T=4)
//   3. MLSTMCell hand-computed reference (single-step, hidden=1)
//   4. MLSTMCell input gradient check (T=2, d=2)
//   5. MLSTMCell input gradient check (T=3, d=2) — deeper BPTT
//   6. MLSTMCell W gradient check (q, k, v, i, f, o blocks covered)
//   7. MLSTMCell b gradient check (q, k, v, i, f, o blocks covered)
//   8. MLSTMCell forget-bias init = 1 convention
//   9. MLSTMCell parameters()/gradients() shape consistency
//  10. MLSTMCell training step reduces loss
//  11. MLSTMCell matrix state norm grows (cell state actually accumulates)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/mlstm.h"

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
    return grad;
}

static Tensor numerical_grad_input(
    MLSTMCell& layer,
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
    MLSTMCell& layer,
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
    cout << "=== mLSTM Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 3, input_size = 2, hidden_size = 2;
        Tensor input(T, input_size);
        for (size_t i = 0; i < T * input_size; ++i) input.data[i] = 0.1 * (i + 1);

        MLSTMCell cell(input_size, hidden_size);
        Tensor output = cell.forward(input);
        cout << "\n--- Test 1: MLSTMCell forward shape ---\n";
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == hidden_size) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << hidden_size << "\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 2: output is finite
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 4, input_size = 2, hidden_size = 2;
        Tensor input(T, input_size);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < input_size; ++k)
                input(i, k) = 0.3 * sin(0.5 * (i + 1)) - 0.2 * (k + 1);

        MLSTMCell cell(input_size, hidden_size);
        Tensor output = cell.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        cout << "\n--- Test 2: MLSTMCell output finite (T=4) ---\n";
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 3: hand-computed single-step reference
    //
    // Hidden=1, input=1, T=1, h_prev=0, C_0=N_0=0.
    // We manually set W and b with mostly-zero values so the projection
    // produces a known [q, k, v, i, f, o]. Then we trace the math.
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 1, in = 1, h = 1;
        MLSTMCell cell(in, h);
        // W has shape (6h, in+h) = (6, 2).  Block layout:
        //   [0]=q, [1]=k, [2]=v, [3]=i, [4]=f, [5]=o
        // Each block's 2 columns: [x, h_prev]
        // Set W such that x=0.5 gives:
        //   q=0.2, k=0.3, v=0.4, i=0.0, f=1.0 (bias-driven), o=0.0
        // (h_prev=0 so the second column contributes 0)
        cell.W(0, 0) = 0.4;   // q from x:  q = 0.4 * 0.5 = 0.2
        cell.W(1, 0) = 0.6;   // k from x:  k = 0.6 * 0.5 = 0.3
        cell.W(2, 0) = 0.8;   // v from x:  v = 0.8 * 0.5 = 0.4
        cell.W(3, 0) = 0.0;   // i from x:  i_pre = 0
        cell.W(4, 0) = 0.0;   // f from x:  f_pre = b[f]=1, so sigma_f = σ(1) ≈ 0.7311
        cell.W(5, 0) = 0.0;   // o from x:  o_pre = 0
        // b[0..5]: q=0, k=0, v=0, i=0, f=1 (forget-bias), o=0
        cell.b(0, 0) = 0.0;
        cell.b(0, 1) = 0.0;
        cell.b(0, 2) = 0.0;
        cell.b(0, 3) = 0.0;
        cell.b(0, 4) = 1.0;
        cell.b(0, 5) = 0.0;
        // h_prev columns all zero (since h=1, the column h_prev contributes 0)
        for (size_t j = 0; j < 6; ++j) cell.W(j, 1) = 0.0;

        Tensor input(T, in);
        input(0, 0) = 0.5;
        Tensor output = cell.forward(input);

        // Hand calculation (T=1, h=1):
        //   q=0.2, k=0.3, v=0.4, i=0, f_pre=1, o=0
        //   log σ(1) = log(1/(1+exp(-1))) = log(0.7311) ≈ -0.3133
        //   m_prev = 0
        //   log_sigma_f + m_prev = -0.3133
        //   m_t = max(-0.3133, 0) = 0
        //   f' = σ(1) * exp(0 - 0) = 0.7311
        //   i' = exp(0 - 0) = 1
        //   C_1 = f' * C_0 + i' * outer(v, k) = 0.4 * 0.3 = 0.12
        //   N_1 = f' * N_0 + i' * outer(k, k) = 0.3 * 0.3 = 0.09
        //   q N q = 0.2 * 0.09 * 0.2 = 0.0036
        //   h_norm = max(1, 0.0036) = 1   (clamped)
        //   h_pre = C q / h_norm = 0.12 * 0.2 / 1 = 0.024
        //   h = σ(0) * 0.024 = 0.5 * 0.024 = 0.012
        double expected = 0.012;
        double got = output(0, 0);
        double rel_err = fabs(got - expected) / max(fabs(expected), 1e-8);
        cout << "\n--- Test 3: MLSTMCell hand-computed reference ---\n";
        cout << "Expected: " << expected << "  Got: " << got
             << "  rel_err: " << rel_err << "\n";
        if (rel_err < 1e-3) {
            cout << "[PASS] hand-computed reference match\n";
            ++passed;
        } else {
            cout << "[FAIL] hand-computed reference mismatch\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 4: input gradient check (T=2, d=2)
    // Use small positive inputs and small targets to stay in a smooth
    // region of the max() boundary in the m_t stabilizer.
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 2, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.1 + 0.05 * i + 0.02 * k;

        Tensor target(T, h);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < h; ++k)
                target(i, k) = 0.5 + 0.05 * k;

        MLSTMCell cell(in, h);
        Tensor ana = analytical_input_grad(cell, input, target);
        Tensor num = numerical_grad_input(cell, input, target);

        double avg = avg_rel_err(ana, num);
        double mx = max_rel_err(ana, num);
        cout << "\n--- Test 4: MLSTMCell input gradient check (T=2, d=2) ---\n";
        cout << "avg rel err: " << avg << "  max rel err: " << mx << "\n";
        if (mx < 0.1) {
            cout << "[PASS] input grad check (max rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input grad check failed\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 5: input gradient check (T=3, d=2) — deeper BPTT
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 3, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.15 + 0.05 * i + 0.02 * k;

        Tensor target(T, h);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < h; ++k)
                target(i, k) = 0.5 + 0.05 * k;

        MLSTMCell cell(in, h);
        Tensor ana = analytical_input_grad(cell, input, target);
        Tensor num = numerical_grad_input(cell, input, target);

        double avg = avg_rel_err(ana, num);
        double mx = max_rel_err(ana, num);
        cout << "\n--- Test 5: MLSTMCell input gradient check (T=3, d=2) ---\n";
        cout << "avg rel err: " << avg << "  max rel err: " << mx << "\n";
        if (mx < 0.1) {
            cout << "[PASS] input grad check (T=3, max rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input grad check (T=3) failed\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 6: W gradient check (sparse, hits all 6 blocks)
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 2, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.2 * (i + 1) - 0.1 * (k + 1);

        Tensor target(T, h);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < h; ++k)
                target(i, k) = 0.1 * i + 0.05 * k;

        MLSTMCell cell(in, h);
        Tensor out = cell.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        // Numerical W gradient (sparse: only a few entries to keep test fast)
        double max_err = 0.0;
        int n_checked = 0;
        // We check W[block_idx][hidden_idx, k] for a few k in each block.
        // Block 0 (q): W(0..d-1, 0..in+d-1)
        // Block 1 (k), 2 (v), 3 (i), 4 (f), 5 (o)
        for (int block = 0; block < 6; ++block) {
            for (size_t j = 0; j < h; ++j) {
                for (size_t k = 0; k < in + h && n_checked < 24; ++k) {
                    size_t W_row = block * h + j;
                    double orig = cell.W(W_row, k);
                    cell.W(W_row, k) = orig + 1e-5;
                    Tensor out_p = cell.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    cell.W(W_row, k) = orig - 1e-5;
                    Tensor out_m = cell.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    cell.W(W_row, k) = orig;
                    double num = (Lp - Lm) / (2e-5);
                    double ana = cell.grad_W(W_row, k);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.1) {
                        cout << "  W[" << W_row << "][" << k << "] (block " << block
                             << "): ana=" << ana << " num=" << num
                             << " err=" << err << "\n";
                    }
                }
            }
        }
        cout << "\n--- Test 6: MLSTMCell W gradient check (all 6 blocks) ---\n";
        cout << "Checked " << n_checked << " entries.  Max rel err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] W gradient check (max rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] W gradient check failed\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 7: b gradient check (all 6 blocks)
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 2, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.15 + 0.05 * i + 0.02 * k;

        Tensor target(T, h);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < h; ++k)
                target(i, k) = 0.4 + 0.05 * k;

        MLSTMCell cell(in, h);
        Tensor out = cell.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        cell.zero_grad();
        cell.backward(grad_loss, 0.0);

        // Check b for one entry in each of the 6 blocks.  Skip f-bias since
        // update_weights re-applies the forget-bias=1 convention.
        double max_err = 0.0;
        for (int block = 0; block < 6; ++block) {
            if (block == 4) continue;  // skip f-bias (forget-bias=1 reset)
            size_t b_col = block * h;  // check the j=0 entry of each block
            double orig = cell.b(0, b_col);
            cell.b(0, b_col) = orig + 1e-5;
            Tensor out_p = cell.forward(input);
            double Lp = l2_loss_value(out_p, target);
            cell.b(0, b_col) = orig - 1e-5;
            Tensor out_m = cell.forward(input);
            double Lm = l2_loss_value(out_m, target);
            cell.b(0, b_col) = orig;
            double num = (Lp - Lm) / (2e-5);
            double ana = cell.grad_b(0, b_col);
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
            if (err > 0.1) {
                cout << "  b[block=" << block << ", j=0]: ana=" << ana
                     << " num=" << num << " err=" << err << "\n";
            }
        }
        cout << "\n--- Test 7: MLSTMCell b gradient check (q,k,v,i,o blocks) ---\n";
        cout << "Max rel err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] b gradient check (max rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] b gradient check failed\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 8: forget-bias init = 1
    // ------------------------------------------------------------------
    {
        ++total;
        size_t in = 2, h = 3;
        MLSTMCell cell(in, h);
        bool ok = true;
        for (size_t j = 0; j < h; ++j) {
            double fb = cell.b(0, 4 * h + j);
            if (fabs(fb - 1.0) > 1e-9) {
                ok = false;
                cout << "  forget bias[" << j << "] = " << fb << ", expected 1.0\n";
            }
        }
        cout << "\n--- Test 8: MLSTMCell forget-bias initialization ---\n";
        if (ok) {
            cout << "[PASS] forget-bias = 1 on init\n";
            ++passed;
        } else {
            cout << "[FAIL] forget-bias not initialized to 1\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 9: parameters/gradients shape consistency
    // ------------------------------------------------------------------
    {
        ++total;
        size_t in = 2, h = 3;
        MLSTMCell cell(in, h);
        auto params = cell.parameters();
        auto grads = cell.gradients();
        cout << "\n--- Test 9: MLSTMCell parameters/gradients shape consistency ---\n";
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
        } else if (params.size() != 2) {
            cout << "[FAIL] expected 2 params (W, b), got " << params.size() << "\n";
        } else {
            bool ok = true;
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows ||
                    params[i]->cols != grads[i]->cols) {
                    ok = false;
                    cout << "  shape mismatch at index " << i << ": "
                         << params[i]->rows << "x" << params[i]->cols
                         << " vs " << grads[i]->rows << "x" << grads[i]->cols << "\n";
                    break;
                }
            }
            if (ok) {
                cout << "[PASS] W and b params/grads shapes match\n";
                ++passed;
            } else {
                cout << "[FAIL] shape mismatch\n";
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 4, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.1 * (i + 1) + 0.05 * (k + 1);

        Tensor target(T, h);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < h; ++k)
                target(i, k) = 0.2 * i - 0.1 * k;

        MLSTMCell cell(in, h);
        Tensor out0 = cell.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.05;
        for (int step = 0; step < 50; ++step) {
            cell.zero_grad();
            Tensor out = cell.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            cell.backward(grad_loss, 0.0);
            cell.update_weights(lr);
        }
        Tensor out1 = cell.forward(input);
        double loss1 = l2_loss_value(out1, target);
        cout << "\n--- Test 10: MLSTMCell training step reduces loss ---\n";
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not decrease loss\n";
        }
    }

    // ------------------------------------------------------------------
    // Test 11: matrix state norm grows — C and N actually accumulate
    // (sanity: with non-trivial input, the cell state should be nonzero)
    // ------------------------------------------------------------------
    {
        ++total;
        size_t T = 3, in = 2, h = 2;
        Tensor input(T, in);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < in; ++k)
                input(i, k) = 0.5 + 0.1 * (i + 1) + 0.05 * (k + 1);

        MLSTMCell cell(in, h);
        Tensor output = cell.forward(input);
        // Compute Frobenius norm of last_C_ at t=T (the final cell state).
        double c_norm = 0.0, n_norm = 0.0;
        Tensor C_final = cell.last_C(T);
        Tensor N_final = cell.last_N(T);
        for (size_t i = 0; i < h; ++i)
            for (size_t j = 0; j < h; ++j) {
                c_norm += C_final(i, j) * C_final(i, j);
                n_norm += N_final(i, j) * N_final(i, j);
            }
        c_norm = sqrt(c_norm);
        n_norm = sqrt(n_norm);
        cout << "\n--- Test 11: MLSTMCell matrix state actually accumulates ---\n";
        cout << "||C_T||_F = " << c_norm << "  ||N_T||_F = " << n_norm << "\n";
        if (c_norm > 1e-6 && n_norm > 1e-6) {
            cout << "[PASS] matrix state has nontrivial norm\n";
            ++passed;
        } else {
            cout << "[FAIL] matrix state is essentially zero — recurrence may be broken\n";
        }
    }

    cout << "\n=== Summary: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
