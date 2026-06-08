// Mamba-2 / SSD — Dao & Gu 2024
//   "Transformers are SSMs: Structured State Space Duality"
//
// Tests:
//   1. Mamba2Block forward shape: (T, d_model) -> (T, d_model)
//   2. Mamba2Block output is finite (T=5)
//   3. Mamba2Block input gradient check (T=3) — tests the full BPTT through
//      the SSD recurrence and all 5 projections
//   4. Mamba2Block a_proj weights gradient check — tests the scalar decay path
//   5. Mamba2Block b_proj weights gradient check — tests the "value" path
//   6. Mamba2Block k_proj weights gradient check — tests the "key" path
//   7. Mamba2Block q_proj weights gradient check — tests the "query" path
//   8. Mamba2Block D_skip gradient check — tests the skip connection
//   9. Mamba2Block dt_bias gradient check — tests the per-head log-decay bias
//  10. Mamba2Block training step reduces loss
//  11. Mamba2Block parameters()/gradients() shape consistency

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/mamba2.h"

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

// Helper: find parameter by shape signature (rows, cols) skipping first match.
// Used when multiple params share a shape (e.g. k_proj, q_proj, b_proj all
// have shape (d_inner, d_model)).
struct ParamMatch {
    Tensor* p = nullptr;
    Tensor* g = nullptr;
    int seen = 0;  // which occurrence to return
};

static ParamMatch find_param(vector<Tensor*>& params, vector<Tensor*>& grads,
                             size_t r, size_t c, int occurrence = 0) {
    ParamMatch pm;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == r && params[i]->cols == c) {
            if (pm.seen == occurrence) {
                pm.p = params[i];
                pm.g = grads[i];
                return pm;
            }
            pm.seen++;
        }
    }
    return pm;
}

int main() {
    cout << "=== Mamba-2 / SSD Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small, tractable config: T=3, d_model=2, n_heads=2, d_inner=4 (default)
    // head_dim = d_inner / n_heads = 2.  Total per-head H: (head_dim, head_dim) = (2, 2).
    size_t T = 3, d_model = 2, n_heads = 2;
    size_t d_inner = 2 * d_model;  // = 4

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: Mamba2Block forward shape (T=3, d_model=2) ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        Mamba2Block block(d_model, n_heads);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: Mamba2Block output is finite (T=5) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor output = block.forward(input);
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

    // ------------------------------------------------------------
    // Test 3: input gradient check (T=3) — exercises the full backward
    //         through the SSD recurrence + all 5 projections
    // ------------------------------------------------------------
    cout << "\n--- Test 3: Mamba2Block input gradient check (T=3) ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < T; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                if (err > 0.1) {
                    cout << "  x[" << i << "][" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.1) {
            cout << "[PASS] input gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: a_proj weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 4: Mamba2Block a_proj weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // a_proj weights shape: (n_heads, d_model) = (2, 2)
        auto m = find_param(params, grads, n_heads, d_model, 0);
        if (!m.p) {
            cout << "[FAIL] could not find a_proj weights (shape "
                 << n_heads << "x" << d_model << ")\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < m.p->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < m.p->cols && n_checked < 4; ++j) {
                    double orig = (*m.p)(i, j);
                    (*m.p)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*m.p)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*m.p)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*m.g)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.1) {
                        cout << "  a_proj[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] a_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] a_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 5: b_proj weights gradient check (value path)
    // ------------------------------------------------------------
    cout << "\n--- Test 5: Mamba2Block b_proj weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // b_proj, k_proj, q_proj all have shape (d_inner, d_model) = (4, 2).
        // We need to identify which is which. The simplest approach: test the
        // FIRST such parameter (b_proj) and accept rel_err < 0.1.
        auto m = find_param(params, grads, d_inner, d_model, 0);
        if (!m.p) {
            cout << "[FAIL] could not find (d_inner, d_model) parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < m.p->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < m.p->cols && n_checked < 4; ++j) {
                    double orig = (*m.p)(i, j);
                    (*m.p)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*m.p)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*m.p)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*m.g)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.1) {
                        cout << "  b_proj[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] b_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] b_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 6: k_proj weights gradient check (key path)
    // ------------------------------------------------------------
    cout << "\n--- Test 6: Mamba2Block k_proj weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // k_proj is the SECOND (d_inner, d_model) parameter (after b_proj).
        auto m = find_param(params, grads, d_inner, d_model, 1);
        if (!m.p) {
            cout << "[FAIL] could not find k_proj parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < m.p->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < m.p->cols && n_checked < 4; ++j) {
                    double orig = (*m.p)(i, j);
                    (*m.p)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*m.p)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*m.p)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*m.g)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.1) {
                        cout << "  k_proj[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] k_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] k_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 7: q_proj weights gradient check (query path)
    // ------------------------------------------------------------
    cout << "\n--- Test 7: Mamba2Block q_proj weights gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // q_proj is the THIRD (d_inner, d_model) parameter (after b_proj, k_proj).
        auto m = find_param(params, grads, d_inner, d_model, 2);
        if (!m.p) {
            cout << "[FAIL] could not find q_proj parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < m.p->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < m.p->cols && n_checked < 4; ++j) {
                    double orig = (*m.p)(i, j);
                    (*m.p)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*m.p)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*m.p)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*m.g)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                    if (err > 0.1) {
                        cout << "  q_proj[" << i << "][" << j << "]: ana=" << ana
                             << " num=" << num << " err=" << err << "\n";
                    }
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] q_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] q_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 8: D_skip gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: Mamba2Block D_skip gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // D_skip is (1, d_inner) = (1, 4)
        auto m = find_param(params, grads, 1, d_inner, 0);
        if (!m.p) {
            cout << "[FAIL] could not find D_skip parameter\n";
        } else {
            double max_err = 0.0;
            for (size_t j = 0; j < m.p->cols; ++j) {
                double orig = (*m.p)(0, j);
                (*m.p)(0, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*m.p)(0, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*m.p)(0, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*m.g)(0, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] D_skip gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] D_skip gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 9: dt_bias gradient check (per-head scalar log-decay)
    // ------------------------------------------------------------
    cout << "\n--- Test 9: Mamba2Block dt_bias gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // dt_bias is (1, n_heads) = (1, 2)
        auto m = find_param(params, grads, 1, n_heads, 0);
        if (!m.p) {
            cout << "[FAIL] could not find dt_bias parameter\n";
        } else {
            double max_err = 0.0;
            for (size_t j = 0; j < m.p->cols; ++j) {
                double orig = (*m.p)(0, j);
                (*m.p)(0, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*m.p)(0, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*m.p)(0, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*m.g)(0, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                if (err > 0.1) {
                    cout << "  dt_bias[" << j << "]: ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] dt_bias gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] dt_bias gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: Mamba2Block training step reduces loss ---\n";
    {
        ++total;
        size_t T10 = 4;
        Tensor input(T10, d_model);
        for (size_t i = 0; i < T10; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T10, d_model);
        for (size_t i = 0; i < T10; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.15 * i - 0.1 * j;

        Mamba2Block block(d_model, n_heads);
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.02;
        for (int step = 0; step < 60; ++step) {
            block.zero_grad();
            Tensor out = block.forward(input);
            Tensor grad_loss = l2_loss_grad(out, target);
            block.backward(grad_loss, 0.0);
            block.update_weights(lr);
        }
        Tensor out1 = block.forward(input);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] training decreased loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not decrease loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: parameters()/gradients() shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 11: Mamba2Block parameters/gradients shape consistency ---\n";
    {
        ++total;
        Mamba2Block block(d_model, n_heads);
        auto params = block.parameters();
        auto grads  = block.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
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
            // Expected: 6 Denses * 2 (W, b) + 2 (D_skip, dt_bias) = 14 params.
            if (ok && params.size() == 14) {
                cout << "[PASS] all 14 param/grad pairs shape-matched\n";
                ++passed;
            } else {
                cout << "[FAIL] expected 14 params, got " << params.size() << "\n";
            }
        }
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
