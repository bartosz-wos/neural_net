// LinOSS — Orvieto et al. 2024/2025
// "From Principles to Learning Systems: State Space Models That Need No Discretization"
// https://arxiv.org/abs/2410.03943 (ICLR 2025)
//
// Tests (~24 focused checks):
//   1.  constructor validation (4 cases)
//   2.  forward shape (T=4, IM)
//   3.  forward shape (T=4, HEUN)
//   4.  forward finiteness + nonzero
//   5.  hand-derived T=1 reference (IM)
//   6.  hand-derived T=1 reference (HEUN)
//   7.  input gradient FD check (T=3, IM)
//   8.  B parameter gradient FD check
//   9.  C parameter gradient FD check
//  10.  log_damping gradient FD check
//  11.  freq gradient FD check
//  12.  log_dt gradient FD check
//  13.  bias gradient FD check
//  14.  longer-sequence (T=8) input grad FD check
//  15.  training reduces loss (single cell)
//  16.  determinism — bit-exact forward with copied params
//  17.  LinOSSModel forward shape + training reduces loss
//  18.  parameters/gradients count + zero_grad contract

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include "nn/layers/recurrent/linoss.h"

using namespace std;

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

// Set non-random init values for FD checks: B/C are non-trivial deterministic
// values that produce non-degenerate recurrences.
static void build_deterministic_cell(LinOSSCell& cell, int d_in, int d_osc, int d_out,
                                     double B_scale = 1.5, double C_scale = 1.0,
                                     double dt = 0.5) {
    for (int i = 0; i < d_in; ++i)
        for (int j = 0; j < 2 * d_osc; ++j)
            cell.B[i][j] = B_scale * (0.5 + 0.1 * (double)((i + 2 * j) % 5));

    for (int i = 0; i < 2 * d_osc; ++i)
        for (int j = 0; j < d_out; ++j)
            cell.C[i][j] = C_scale * (0.3 + 0.1 * (double)((i + 3 * j) % 4));

    for (int k = 0; k < d_osc; ++k) cell.log_damping[k][0] = std::log(0.5);
    for (int k = 0; k < d_osc; ++k) cell.freq[k][0] = 0.4 + 0.2 * (double)k;

    cell.log_dt[0][0] = std::log(std::exp(dt) - 1.0);

    for (int j = 0; j < d_out; ++j) cell.bias[j][0] = 0.0;
}

static void copy_params(const LinOSSCell& src, LinOSSCell& dst) {
    dst.B = src.B.clone();
    dst.C = src.C.clone();
    dst.log_damping = src.log_damping.clone();
    dst.freq = src.freq.clone();
    dst.log_dt = src.log_dt.clone();
    dst.bias = src.bias.clone();
}

int main() {
    cout << "=== LinOSS Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: constructor validation
    // ------------------------------------------------------------
    {
        ++total;
        bool threw_d_input = false;
        try { LinOSSCell cell(0, 3, 4); }
        catch (const std::exception&) { threw_d_input = true; }
        if (threw_d_input) { cout << "[PASS] d_input=0 throws\n"; ++passed; }
        else cout << "[FAIL] d_input=0 should throw\n";

        ++total;
        bool threw_d_output = false;
        try { LinOSSCell cell(2, 0, 4); }
        catch (const std::exception&) { threw_d_output = true; }
        if (threw_d_output) { cout << "[PASS] d_output=0 throws\n"; ++passed; }
        else cout << "[FAIL] d_output=0 should throw\n";

        ++total;
        bool threw_d_osc = false;
        try { LinOSSCell cell(2, 3, 0); }
        catch (const std::exception&) { threw_d_osc = true; }
        if (threw_d_osc) { cout << "[PASS] d_osc=0 throws\n"; ++passed; }
        else cout << "[FAIL] d_osc=0 should throw\n";

        ++total;
        bool ok = true;
        try { LinOSSCell cell(2, 3, 4); }
        catch (...) { ok = false; }
        if (ok) { cout << "[PASS] LinOSSCell(2,3,4) constructs\n"; ++passed; }
        else cout << "[FAIL] construction should succeed\n";
    }

    // ------------------------------------------------------------
    // Test 2 & 3: forward shape
    // ------------------------------------------------------------
    {
        ++total;
        LinOSSCell cell(3, 2, 4, LinOSSType::IMPLICIT_MIDPOINT);
        Tensor input(4, 3);
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = 0.1 * std::sin(0.3 * (double)i);
        Tensor out = cell.forward(input);
        if (out.rows == 4 && out.cols == 2) {
            cout << "[PASS] IM forward shape (4,2)\n"; ++passed;
        } else {
            cout << "[FAIL] expected (4,2), got (" << out.rows << "," << out.cols << ")\n";
        }
    }

    {
        ++total;
        LinOSSCell cell(3, 2, 4, LinOSSType::HEUN);
        Tensor input(4, 3);
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = 0.1 * std::cos(0.4 * (double)i);
        Tensor out = cell.forward(input);
        if (out.rows == 4 && out.cols == 2) {
            cout << "[PASS] HEUN forward shape (4,2)\n"; ++passed;
        } else {
            cout << "[FAIL] expected (4,2), got (" << out.rows << "," << out.cols << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: forward finiteness + nonzero
    // ------------------------------------------------------------
    {
        ++total;
        LinOSSCell cell(2, 3, 2, LinOSSType::IMPLICIT_MIDPOINT);
        Tensor input(5, 2);
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = 0.5 * ((double)i - 2.0);
        Tensor out = cell.forward(input);
        bool finite = true;
        for (size_t i = 0; i < out.data.size(); ++i)
            if (!std::isfinite(out.data[i])) finite = false;
        double any_nonzero = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) any_nonzero += std::fabs(out.data[i]);
        if (finite && any_nonzero > 1e-12) {
            cout << "[PASS] forward finite and nonzero (sum=" << any_nonzero << ")\n"; ++passed;
        } else {
            cout << "[FAIL] finite=" << finite << " nonzero=" << any_nonzero << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: hand-derived T=1 reference (IM)
    // ------------------------------------------------------------
    {
        ++total;
        LinOSSCell cell(1, 1, 1, LinOSSType::IMPLICIT_MIDPOINT);
        cell.log_damping[0][0] = 0.0;
        cell.freq[0][0] = 2.0;
        cell.log_dt[0][0] = std::log(std::exp(1.0) - 1.0);
        cell.B[0][0] = 0.3;
        cell.B[0][1] = -0.2;
        cell.C[0][0] = 1.5;
        cell.C[1][0] = 0.7;
        cell.bias[0][0] = 0.1;
        Tensor input(1, 1);
        input[0][0] = 0.5;
        Tensor out = cell.forward(input);
        double expected = 0.09308;
        double got = out[0][0];
        double err = std::fabs(got - expected);
        if (err < 1e-3) {
            cout << "[PASS] IM T=1 reference (got " << got << ", err " << err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] IM T=1 reference (got " << got << ", err " << err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: hand-derived T=1 reference (HEUN)
    // ------------------------------------------------------------
    {
        ++total;
        LinOSSCell cell(1, 1, 1, LinOSSType::HEUN);
        cell.log_damping[0][0] = 0.0;
        cell.freq[0][0] = 2.0;
        cell.log_dt[0][0] = std::log(std::exp(1.0) - 1.0);
        cell.B[0][0] = 0.3;
        cell.B[0][1] = -0.2;
        cell.C[0][0] = 1.5;
        cell.C[1][0] = 0.7;
        cell.bias[0][0] = 0.1;
        Tensor input(1, 1);
        input[0][0] = 0.5;
        Tensor out = cell.forward(input);
        double expected = -0.0775;
        double got = out[0][0];
        double err = std::fabs(got - expected);
        if (err < 1e-3) {
            cout << "[PASS] HEUN T=1 reference (got " << got << ", err " << err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] HEUN T=1 reference (got " << got << ", err " << err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 7: input gradient FD check (T=3, IM)
    // Use deterministic well-conditioned config + non-zero target offset.
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);

        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) {
            target[t][0] = 0.7 + 0.1 * t;
            target[t][1] = 0.3 - 0.1 * t;
        }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        Tensor grad_input_ana = cell.backward(grad_out, 0.0);

        double eps = 1e-4;
        double max_err = 0.0;
        for (size_t i = 0; i < input.data.size(); ++i) {
            double orig = input.data[i];
            input.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);

            input.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);

            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_input_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;

            input.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 1e-2) {
            cout << "[PASS] input grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] input grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: B parameter gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        Tensor grad_B_ana = cell.grad_B.clone();

        double eps = 1e-4;
        double max_err = 0.0;
        size_t n_check = std::min<size_t>(cell.B.data.size(), 8);
        for (size_t i = 0; i < n_check; ++i) {
            double orig = cell.B.data[i];
            cell.B.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            cell.B.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_B_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            cell.B.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 1e-2) {
            cout << "[PASS] B grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] B grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 9: C parameter gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        Tensor grad_C_ana = cell.grad_C.clone();

        double eps = 1e-4;
        double max_err = 0.0;
        size_t n_check = std::min<size_t>(cell.C.data.size(), 8);
        for (size_t i = 0; i < n_check; ++i) {
            double orig = cell.C.data[i];
            cell.C.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            cell.C.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_C_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            cell.C.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 1e-2) {
            cout << "[PASS] C grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] C grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 10: log_damping gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        Tensor grad_ld_ana = cell.grad_log_damping.clone();

        double eps = 1e-4;
        double max_err = 0.0;
        size_t n_check = std::min<size_t>(cell.log_damping.data.size(), 4);
        for (size_t i = 0; i < n_check; ++i) {
            double orig = cell.log_damping.data[i];
            cell.log_damping.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            cell.log_damping.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_ld_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            cell.log_damping.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 5e-2) {
            cout << "[PASS] log_damping grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] log_damping grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: freq gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        Tensor grad_freq_ana = cell.grad_freq.clone();

        double eps = 1e-4;
        double max_err = 0.0;
        size_t n_check = std::min<size_t>(cell.freq.data.size(), 4);
        for (size_t i = 0; i < n_check; ++i) {
            double orig = cell.freq.data[i];
            cell.freq.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            cell.freq.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_freq_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            cell.freq.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 5e-2) {
            cout << "[PASS] freq grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] freq grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: log_dt gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        double grad_dt_ana = cell.grad_log_dt[0][0];

        double eps = 1e-4;
        double orig = cell.log_dt[0][0];
        cell.log_dt[0][0] = orig + eps;
        cell.reset_state();
        Tensor out_p = cell.forward(input);
        double lp = l2_loss_value(out_p, target);
        cell.log_dt[0][0] = orig - eps;
        cell.reset_state();
        Tensor out_m = cell.forward(input);
        double lm = l2_loss_value(out_m, target);
        double fd = (lp - lm) / (2.0 * eps);
        double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(grad_dt_ana)));
        double rel = std::fabs(fd - grad_dt_ana) / denom;
        cell.log_dt[0][0] = orig;
        cell.reset_state();
        if (rel < 5e-2) {
            cout << "[PASS] log_dt grad FD check (rel_err " << rel << ")\n"; ++passed;
        } else {
            cout << "[FAIL] log_dt grad FD check (rel_err " << rel << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: bias gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.1 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.05 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.7; target[t][1] = 0.3; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        cell.backward(grad_out, 0.0);
        Tensor grad_bias_ana = cell.grad_bias.clone();

        double eps = 1e-4;
        double max_err = 0.0;
        size_t n_check = std::min<size_t>(cell.bias.data.size(), 4);
        for (size_t i = 0; i < n_check; ++i) {
            double orig = cell.bias.data[i];
            cell.bias.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            cell.bias.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_bias_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            cell.bias.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 1e-8) {
            cout << "[PASS] bias grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] bias grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: longer-sequence (T=8) input grad FD check
    // ------------------------------------------------------------
    {
        ++total;
        int T = 8;
        LinOSSCell cell(2, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        build_deterministic_cell(cell, 2, 2, 2, 1.5, 1.0, 0.5);
        Tensor input(T, 2);
        for (int t = 0; t < T; ++t) input[t][0] = 0.4 + 0.05 * t;
        for (int t = 0; t < T; ++t) input[t][1] = -0.2 + 0.03 * t;
        Tensor target(T, 2);
        for (int t = 0; t < T; ++t) { target[t][0] = 0.6; target[t][1] = 0.4; }

        Tensor output = cell.forward(input);
        Tensor grad_out = l2_loss_grad(output, target);
        Tensor grad_input_ana = cell.backward(grad_out, 0.0);

        double eps = 1e-4;
        double max_err = 0.0;
        for (size_t i = 0; i < input.data.size(); ++i) {
            double orig = input.data[i];
            input.data[i] = orig + eps;
            cell.reset_state();
            Tensor out_p = cell.forward(input);
            double lp = l2_loss_value(out_p, target);
            input.data[i] = orig - eps;
            cell.reset_state();
            Tensor out_m = cell.forward(input);
            double lm = l2_loss_value(out_m, target);
            double fd = (lp - lm) / (2.0 * eps);
            double ana = grad_input_ana.data[i];
            double denom = std::max(1e-8, std::max(std::fabs(fd), std::fabs(ana)));
            double rel = std::fabs(fd - ana) / denom;
            if (rel > max_err) max_err = rel;
            input.data[i] = orig;
        }
        cell.reset_state();
        if (max_err < 5e-2) {
            cout << "[PASS] longer-seq input grad FD check (max_rel_err " << max_err << ")\n"; ++passed;
        } else {
            cout << "[FAIL] longer-seq input grad FD check (max_rel_err " << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: training reduces loss (single cell)
    // ------------------------------------------------------------
    {
        ++total;
        int T = 4;
        int d_in = 2, d_out = 2;
        LinOSSCell cell(d_in, d_out, 2, LinOSSType::IMPLICIT_MIDPOINT);

        Tensor input(T, d_in);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.5 * std::sin(0.4 * (double)i);

        Tensor target(T, d_out);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.3;

        Tensor output = cell.forward(input);
        double l0 = l2_loss_value(output, target);

        double lr = 0.05;
        for (int step = 0; step < 80; ++step) {
            cell.zero_grad();
            cell.reset_state();
            output = cell.forward(input);
            Tensor grad_out = l2_loss_grad(output, target);
            cell.backward(grad_out, 0.0);
            cell.update_weights(lr);
        }
        cell.reset_state();
        output = cell.forward(input);
        double lf = l2_loss_value(output, target);
        if (lf < l0 * 0.5) {
            cout << "[PASS] training reduces loss (" << l0 << " -> " << lf << ")\n"; ++passed;
        } else {
            cout << "[FAIL] training did not reduce loss enough (" << l0 << " -> " << lf << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: determinism — bit-exact forward with copied params
    // ------------------------------------------------------------
    {
        ++total;
        int T = 4;
        LinOSSCell cell1(2, 3, 2, LinOSSType::IMPLICIT_MIDPOINT);
        LinOSSCell cell2(2, 3, 2, LinOSSType::IMPLICIT_MIDPOINT);
        copy_params(cell1, cell2);

        Tensor input(T, 2);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.3 * std::cos(0.5 * (double)i);

        cell1.reset_state();
        Tensor out1 = cell1.forward(input);
        cell2.reset_state();
        Tensor out2 = cell2.forward(input);

        double max_diff = 0.0;
        for (size_t i = 0; i < out1.data.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(out1.data[i] - out2.data[i]));

        if (max_diff < 1e-12) {
            cout << "[PASS] determinism (max_diff " << max_diff << ")\n"; ++passed;
        } else {
            cout << "[FAIL] determinism (max_diff " << max_diff << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 17: LinOSSModel forward shape + training reduces loss
    // ------------------------------------------------------------
    {
        ++total;
        int T = 3;
        LinOSSModel model(2, 2, 4, 2, 2, LinOSSType::IMPLICIT_MIDPOINT);
        Tensor input(T, 2);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.3 * std::sin(0.4 * (double)i);

        Tensor output = model.forward(input);
        bool shape_ok = (output.rows == 1 && output.cols == 2);
        bool finite_ok = true;
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!std::isfinite(output.data[i])) finite_ok = false;
        if (shape_ok && finite_ok) {
            cout << "[PASS] LinOSSModel forward shape (1,2) finite\n"; ++passed;
        } else {
            cout << "[FAIL] LinOSSModel forward shape (" << output.rows << "," << output.cols << ")\n";
        }

        ++total;
        Tensor target(1, 2);
        target[0][0] = 0.2; target[0][1] = -0.1;
        double l0 = l2_loss_value(output, target);
        double lr = 0.05;
        for (int step = 0; step < 80; ++step) {
            model.zero_grad();
            output = model.forward(input);
            Tensor grad_out = l2_loss_grad(output, target);
            model.backward(grad_out, 0.0);
            model.update_weights(lr);
        }
        output = model.forward(input);
        double lf = l2_loss_value(output, target);
        if (lf < l0 * 0.95) {
            cout << "[PASS] LinOSSModel training reduces loss (" << l0 << " -> " << lf << ")\n"; ++passed;
        } else {
            cout << "[FAIL] LinOSSModel training did not reduce loss enough (" << l0 << " -> " << lf << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 18: parameters/gradients count + zero_grad contract
    // ------------------------------------------------------------
    {
        ++total;
        LinOSSCell cell(2, 3, 4, LinOSSType::IMPLICIT_MIDPOINT);
        auto p = cell.parameters();
        if (p.size() == 6) {
            cout << "[PASS] parameters() returns 6 tensors\n"; ++passed;
        } else {
            cout << "[FAIL] parameters() size " << p.size() << "\n";
        }

        ++total;
        Tensor input(2, 2);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (double)i;
        Tensor output = cell.forward(input);
        Tensor grad_out(output.rows, output.cols);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = 1.0;
        cell.backward(grad_out, 0.0);

        auto g = cell.gradients();
        if (g.size() == 6) {
            cout << "[PASS] gradients() returns 6 tensors\n"; ++passed;
        } else {
            cout << "[FAIL] gradients() size " << g.size() << "\n";
        }

        ++total;
        cell.zero_grad();
        double sum_after_zero = 0.0;
        for (size_t i = 0; i < cell.grad_B.data.size(); ++i) sum_after_zero += std::fabs(cell.grad_B.data[i]);
        for (size_t i = 0; i < cell.grad_C.data.size(); ++i) sum_after_zero += std::fabs(cell.grad_C.data[i]);
        for (size_t i = 0; i < cell.grad_log_damping.data.size(); ++i) sum_after_zero += std::fabs(cell.grad_log_damping.data[i]);
        for (size_t i = 0; i < cell.grad_freq.data.size(); ++i) sum_after_zero += std::fabs(cell.grad_freq.data[i]);
        sum_after_zero += std::fabs(cell.grad_log_dt[0][0]);
        for (size_t i = 0; i < cell.grad_bias.data.size(); ++i) sum_after_zero += std::fabs(cell.grad_bias.data[i]);
        if (sum_after_zero < 1e-12) {
            cout << "[PASS] zero_grad clears all gradients\n"; ++passed;
        } else {
            cout << "[FAIL] zero_grad leaves residual gradient " << sum_after_zero << "\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed (" << total << " total) ===" << endl;
    return (passed == total) ? 0 : 1;
}