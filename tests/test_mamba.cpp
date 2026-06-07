// Mamba (S6) — Gu & Dao 2023
//   "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//
// Tests:
//   1. MambaBlock forward shape: (T, d_model) -> (T, d_model)
//   2. MambaBlock output is finite
//   3. MambaBlock input gradient check
//   4. MambaBlock A_log gradient check (state matrix; tests selective scan backward)
//   5. MambaBlock D_skip gradient check (skip connection; simple but tests the y_t decomposition)
//   6. MambaBlock in_proj weights gradient check (Dense inside, but with sel-scan gradient path)
//   7. MambaBlock B_proj weights gradient check (tests B_proj → B_t → B̄_t → h_t → y_t path)
//   8. MambaBlock C_proj weights gradient check (tests y_t → grad_C_t path)
//   9. MambaBlock dt_proj weights gradient check (tests softplus + Δ path)
//  10. MambaBlock training step reduces loss
//  11. MambaBlock parameters()/gradients() shape consistency

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/mamba.h"

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

// Find a parameter matching (rows, cols); returns nullptr if not found.
[[maybe_unused]] static Tensor* find_param(vector<Tensor*>& params, size_t r, size_t c) {
    for (auto* p : params) if (p->rows == r && p->cols == c) return p;
    return nullptr;
}
[[maybe_unused]] static Tensor* find_grad(vector<Tensor*>& grads, size_t r, size_t c) {
    for (auto* g : grads) if (g->rows == r && g->cols == c) return g;
    return nullptr;
}

int main() {
    cout << "=== Mamba (S6) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small, tractable config: T=3, d_model=2, d_state=2, d_inner=2 (default = 2*d_model)
    size_t T = 3, d_model = 2, d_state = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: MambaBlock forward shape (T=3, d_model=2) ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        MambaBlock block(d_model, d_state);
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
    cout << "\n--- Test 2: MambaBlock output is finite (T=5) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        MambaBlock block(d_model, d_state);
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
    // Test 3: input gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 3: MambaBlock input gradient check (T=3) ---\n";
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

        MambaBlock block(d_model, d_state);
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
    // Test 4: A_log gradient check (state matrix)
    // ------------------------------------------------------------
    cout << "\n--- Test 4: MambaBlock A_log gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // A_log is the (d_inner, d_state) parameter. d_inner = 2*d_model = 4.
        size_t d_inner = 2 * d_model;
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Ap = nullptr;
        Tensor* Ag = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_inner && params[i]->cols == d_state) {
                Ap = params[i];
                Ag = grads[i];
                break;
            }
        }
        if (!Ap) {
            cout << "[FAIL] could not find A_log parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Ap->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < Ap->cols && n_checked < 4; ++j) {
                    double orig = (*Ap)(i, j);
                    (*Ap)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Ap)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Ap)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Ag)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] A_log gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] A_log gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 5: D_skip gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 5: MambaBlock D_skip gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        size_t d_inner = 2 * d_model;
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Dp = nullptr;
        Tensor* Dg = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == 1 && params[i]->cols == d_inner) {
                Dp = params[i];
                Dg = grads[i];
                break;
            }
        }
        if (!Dp) {
            cout << "[FAIL] could not find D_skip parameter\n";
        } else {
            double max_err = 0.0;
            for (size_t j = 0; j < Dp->cols; ++j) {
                double orig = (*Dp)(0, j);
                (*Dp)(0, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*Dp)(0, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*Dp)(0, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*Dg)(0, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.05) {
                cout << "[PASS] D_skip gradient check (rel_err < 5%)\n";
                ++passed;
            } else {
                cout << "[FAIL] D_skip gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 6: in_proj weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: MambaBlock in_proj weights gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // in_proj weights are the (2*d_inner, d_model) tensor (first weights in param list)
        size_t d_inner = 2 * d_model;
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Wp = params[0];
        Tensor* Wg = grads[0];
        if (Wp->rows != 2 * d_inner || Wp->cols != d_model) {
            cout << "[FAIL] in_proj weights shape mismatch: "
                 << Wp->rows << "x" << Wp->cols << "\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Wp->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < Wp->cols && n_checked < 4; ++j) {
                    double orig = (*Wp)(i, j);
                    (*Wp)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Wp)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Wp)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Wg)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] in_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] in_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 7: B_proj weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: MambaBlock B_proj weights gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // B_proj weights: (d_state, d_model). Find by shape.
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Bp = nullptr;
        Tensor* Bg = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_state && params[i]->cols == d_model) {
                Bp = params[i];
                Bg = grads[i];
                break;
            }
        }
        if (!Bp) {
            cout << "[FAIL] could not find B_proj weights parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Bp->rows && n_checked < 3; ++i) {
                for (size_t j = 0; j < Bp->cols && n_checked < 3; ++j) {
                    double orig = (*Bp)(i, j);
                    (*Bp)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Bp)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Bp)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Bg)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] B_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] B_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 8: C_proj weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: MambaBlock C_proj weights gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        auto params = block.parameters();
        auto grads  = block.gradients();
        // C_proj weights: (d_state, d_model). Could collide with B_proj by shape;
        // we differentiate by checking grad against a SECOND independent forward.
        // The simplest approach: iterate all (d_state, d_model) parameters and
        // check the first one not already taken by B_proj. But to keep this
        // robust, just check ALL of them with a low threshold — at least one
        // must be a C_proj grad and it will pass.
        double max_err_best = 1e9;
        for (size_t idx = 0; idx < params.size(); ++idx) {
            Tensor* Cp = params[idx];
            Tensor* Cg = grads[idx];
            if (Cp->rows != d_state || Cp->cols != d_model) continue;
            // Skip the FIRST (d_state, d_model) — assume that's B_proj.
            // We'll check the SECOND one.
            static int seen = 0;
            if (seen == 0) { ++seen; continue; }
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Cp->rows && n_checked < 3; ++i) {
                for (size_t j = 0; j < Cp->cols && n_checked < 3; ++j) {
                    double orig = (*Cp)(i, j);
                    (*Cp)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Cp)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Cp)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Cg)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            max_err_best = max_err;
            break;
        }
        cout << "Max err: " << max_err_best << "\n";
        if (max_err_best < 0.1) {
            cout << "[PASS] C_proj weights gradient check (rel_err < 10%)\n";
            ++passed;
        } else {
            cout << "[FAIL] C_proj weights gradient check failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: dt_proj weights gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: MambaBlock dt_proj weights gradient check ---\n";
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

        MambaBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(grad_loss, 0.0);

        // dt_proj weights: (d_inner, d_model). d_inner = 2*d_model = 4.
        size_t d_inner = 2 * d_model;
        auto params = block.parameters();
        auto grads  = block.gradients();
        Tensor* Tp = nullptr;
        Tensor* Tg = nullptr;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows == d_inner && params[i]->cols == d_model) {
                Tp = params[i];
                Tg = grads[i];
                break;
            }
        }
        if (!Tp) {
            cout << "[FAIL] could not find dt_proj weights parameter\n";
        } else {
            double max_err = 0.0;
            int n_checked = 0;
            for (size_t i = 0; i < Tp->rows && n_checked < 4; ++i) {
                for (size_t j = 0; j < Tp->cols && n_checked < 4; ++j) {
                    double orig = (*Tp)(i, j);
                    (*Tp)(i, j) = orig + eps;
                    Tensor out_p = block.forward(input);
                    double Lp = l2_loss_value(out_p, target);
                    (*Tp)(i, j) = orig - eps;
                    Tensor out_m = block.forward(input);
                    double Lm = l2_loss_value(out_m, target);
                    (*Tp)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*Tg)(i, j);
                    double err = relative_error(num, ana);
                    max_err = max(max_err, err);
                    ++n_checked;
                }
            }
            cout << "Max err: " << max_err << "\n";
            if (max_err < 0.1) {
                cout << "[PASS] dt_proj weights gradient check (rel_err < 10%)\n";
                ++passed;
            } else {
                cout << "[FAIL] dt_proj weights gradient check failed\n";
            }
        }
    }

    // ------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: MambaBlock training step reduces loss ---\n";
    {
        ++total;
        size_t T2 = 4;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        Tensor target(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        MambaBlock block(d_model, d_state);
        Tensor out0 = block.forward(input);
        double loss0 = l2_loss_value(out0, target);

        double lr = 0.01;
        for (int step = 0; step < 30; ++step) {
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
    cout << "\n--- Test 11: parameters()/gradients() shape consistency ---\n";
    {
        ++total;
        MambaBlock block(d_model, d_state);
        auto params = block.parameters();
        auto grads  = block.gradients();
        if (params.size() != grads.size()) {
            cout << "[FAIL] params.size()=" << params.size()
                 << " != grads.size()=" << grads.size() << "\n";
        } else if (params.size() != 12) {
            cout << "[FAIL] expected 12 params (5 dense * 2 + A_log + D_skip) = "
                 << params.size() << "\n";
        } else {
            // Verify shapes match pairwise
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
                cout << "[PASS] all 12 params/grads shapes match\n";
                ++passed;
            } else {
                cout << "[FAIL] shape mismatch\n";
            }
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
