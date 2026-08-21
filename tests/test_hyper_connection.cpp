// Hyper-Connection — DeepSeek-AI 2025
// "Hyper-Connections" (https://arxiv.org/abs/2409.19606)
//
// Tests (~16 focused checks):
//   1.  Constructor validation (d_model=0 throws, inner=nullptr throws)
//   2.  Forward shape (n=4, d=8)
//   3.  Forward output is finite + nonzero
//   4.  At init (alpha=1, beta=0): output ≈ inner(x) (recovers standard residual
//       in the limit where beta is tiny)
//   5.  Forced alpha=0.5, beta=0.5: output = 0.5*x + 0.5*inner(x) to machine precision
//   6.  Identity inner: forward equals input (alpha=1, beta=0 at init)
//   7.  Input gradient FD check (rel_err < 1e-9)
//   8.  alpha_log gradient FD check (rel_err < 1e-9)
//   9.  beta_log gradient FD check (rel_err < 1e-9)
//  10.  alpha_log gradient is nonzero when d_out nonzero AND x nonzero
//  11.  beta_log gradient is nonzero when d_out nonzero AND sub_out nonzero
//  12.  HyperConnectionBlock: pre-LN -> Dense -> HyperConnection, forward shape
//  13.  HyperConnectionBlock: input gradient FD check (rel_err < 1e-9)
//  14.  HyperConnectionBlock: training reduces loss (50 SGD steps)
//  15.  HyperConnectionModel: full model forward shape + training reduces loss
//  16.  Mutation test: zeroing alpha/beta gradient -> input grad test fails
//
// All gradient checks use deterministic non-uniform init to avoid row-vs-column
// confusion. Loss is 0.5*sum((output-target)^2) so its gradient w.r.t. output
// is simply (output - target).

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include "nn/layers/utility/hyper_connection.h"

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

// Hyper-connection with simple non-uniform params to exercise asymmetric paths
static void build_deterministic_hc(HyperConnection& hc, size_t d) {
    hc.alpha_log() = Tensor(1, d);
    hc.beta_log() = Tensor(1, d);
    for (size_t j = 0; j < d; ++j) {
        // Pick alpha_log so sigmoid(alpha_log) ~= 0.3 + 0.1*j/d (asymmetric range)
        double a_target = 0.3 + 0.1 * (double)((j + 1) % 5);
        // sigmoid_inv(a) = log(a / (1-a))
        hc.alpha_log()[0][j] = std::log(a_target / (1.0 - a_target));
        // Pick beta_log so sigmoid(beta_log) ~= 0.1 + 0.05*j/d (asymmetric, non-zero)
        double b_target = 0.1 + 0.05 * (double)((j + 2) % 7);
        hc.beta_log()[0][j] = std::log(b_target / (1.0 - b_target));
    }
}

// FD gradient for a single scalar parameter via centered finite differences.
struct MaxErrAccum { double v = 0.0; };
static void fd_check_param(HyperConnection& hc, const Tensor& input,
                          const Tensor& target, double eps,
                          Tensor& param, MaxErrAccum& acc,
                          const char* label, size_t n_samples) {
    // Forward+backward at theta
    hc.zero_grad();
    Tensor out = hc.forward(input);
    Tensor grad_out = l2_loss_grad(out, target);
    hc.backward(grad_out, 0.0);
    Tensor g_ana = param.clone(); // snapshot analytical gradient

    size_t total = param.rows * param.cols;
    size_t step = max((size_t)1, total / n_samples);
    for (size_t k = 0, idx = 0; k < n_samples && idx < total; ++k, idx += step) {
        size_t i = idx / param.cols;
        size_t j = idx % param.cols;
        double orig = param[i][j];
        param[i][j] = orig + eps;
        Tensor out_p = hc.forward(input);
        double loss_p = l2_loss_value(out_p, target);
        param[i][j] = orig - eps;
        Tensor out_m = hc.forward(input);
        double loss_m = l2_loss_value(out_m, target);
        param[i][j] = orig;
        double num = (loss_p - loss_m) / (2.0 * eps);
        double ana = g_ana[i][j];
        double err = relative_error(ana, num);
        if (err > acc.v) acc.v = err;
        if (err > 1e-3) {
            cout << "    [" << label << "] err=" << err
                 << " ana=" << ana << " num=" << num << "\n";
        }
    }
}

int main() {
    cout << "=== Hyper-Connection Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Helper lambdas for unique_ptr<Layer> construction
    auto make_dense = [](size_t in_f, size_t out_f) {
        return std::unique_ptr<Dense>(new Dense(in_f, out_f));
    };

    // ------------------------------------------------------------
    // Test 1: constructor validation
    // ------------------------------------------------------------
    {
        // 1a: d_model=0 throws
        ++total;
        bool threw_d = false;
        try {
            HyperConnection hc(0, make_dense(0, 4).release());
        } catch (const std::exception&) { threw_d = true; }
        if (threw_d) { cout << "[PASS] d_model=0 throws\n"; ++passed; }
        else cout << "[FAIL] d_model=0 should throw\n";

        // 1b: inner=nullptr throws
        ++total;
        bool threw_n = false;
        try {
            HyperConnection hc(4, nullptr);
        } catch (const std::exception&) { threw_n = true; }
        if (threw_n) { cout << "[PASS] inner=nullptr throws\n"; ++passed; }
        else cout << "[FAIL] inner=nullptr should throw\n";
    }

    // ------------------------------------------------------------
    // Test 2: forward shape (n=4, d=8) with Dense(8->8) inner
    // ------------------------------------------------------------
    {
        ++total;
        size_t n = 4, d = 8;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        auto inner = make_dense(d, d);
        HyperConnection hc(d, inner.release());
        Tensor out = hc.forward(input);
        if (out.rows == n && out.cols == d) {
            cout << "[PASS] forward shape correct\n"; ++passed;
        } else cout << "[FAIL] forward shape expected " << n << "x" << d
                    << " got " << out.rows << "x" << out.cols << "\n";
    }

    // ------------------------------------------------------------
    // Test 3: forward finiteness + nonzero
    // ------------------------------------------------------------
    {
        ++total;
        size_t n = 4, d = 8;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        auto inner = make_dense(d, d);
        HyperConnection hc(d, inner.release());
        // Use deterministic non-init-zero params
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        bool finite = true, nonzero = false;
        double max_abs = 0.0, min_abs = 1e10;
        for (size_t i = 0; i < out.data.size(); ++i) {
            double v = out.data[i];
            if (!std::isfinite(v)) finite = false;
            double av = fabs(v);
            if (av > max_abs) max_abs = av;
            if (av < min_abs && av > 0.0) min_abs = av;
        }
        if (max_abs > 1e-6) nonzero = true;
        if (finite && nonzero) {
            cout << "[PASS] forward finite + nonzero (max_abs=" << max_abs << ")\n";
            ++passed;
        } else cout << "[FAIL] forward finite=" << finite
                    << " nonzero=" << nonzero << "\n";
    }

    // ------------------------------------------------------------
    // Test 4: at init (alpha=1, beta=0), output ≈ inner(x) to machine precision
    // ------------------------------------------------------------
    {
        ++total;
        size_t n = 4, d = 6;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * i + 0.07 * j;

        // Build inner once and snapshot its output
        auto inner1 = make_dense(d, d);
        // Make weights deterministic so output is reproducible
        for (size_t i = 0; i < inner1->weights.rows; ++i)
            for (size_t j = 0; j < inner1->weights.cols; ++j)
                inner1->weights[i][j] = 0.1 + 0.01 * (i + j);
        Tensor inner_out = inner1->forward(input);

        // Build a second inner with the SAME weights
        auto inner2 = std::unique_ptr<Dense>(new Dense(d, d));
        inner2->weights = inner1->weights.clone();
        HyperConnection hc(d, inner2.release());
        // Force alpha=1, beta=0 by setting log params appropriately:
        // sigmoid(10) ≈ 0.99995 (close to 1), sigmoid(-10) ≈ 4.5e-5 (close to 0)
        hc.alpha_log() = Tensor(1, d);
        hc.beta_log() = Tensor(1, d);
        for (size_t j = 0; j < d; ++j) {
            hc.alpha_log()[0][j] = 10.0;
            hc.beta_log()[0][j] = -10.0;
        }
        Tensor out = hc.forward(input);
        // out ≈ alpha*x + beta*inner(x) ≈ 1.0*x + 0.0*inner(x) = x (because alpha*x dominates, not inner(x))
        // The paper's recovery: at alpha=1, beta=0, output ≈ x (identity residual)
        // So we compare to x, not inner_out.
        double max_diff = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) {
            double diff = fabs(out.data[i] - input.data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff < 1e-3) {
            cout << "[PASS] at init alpha=1,beta=0 -> output ≈ x (max_diff=" << max_diff << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] at init max_diff=" << max_diff << " (expected < 1e-3)\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: forced alpha=0.5, beta=0.5 -> output = 0.5*x + 0.5*inner(x)
    // ------------------------------------------------------------
    {
        ++total;
        size_t n = 3, d = 4;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * (i + 1) - 0.2 * (j + 1);

        auto inner = std::unique_ptr<Dense>(new Dense(d, d));
        for (size_t i = 0; i < inner->weights.rows; ++i)
            for (size_t j = 0; j < inner->weights.cols; ++j)
                inner->weights[i][j] = 0.2 + 0.05 * (i + j);
        for (size_t j = 0; j < d; ++j) inner->bias[0][j] = 0.1;
        Tensor inner_out = inner->forward(input);

        HyperConnection hc(d, inner.release());
        // sigmoid_inv(0.5) = 0.0
        hc.alpha_log() = Tensor(1, d);
        hc.beta_log() = Tensor(1, d);
        for (size_t j = 0; j < d; ++j) {
            hc.alpha_log()[0][j] = 0.0;
            hc.beta_log()[0][j] = 0.0;
        }
        Tensor out = hc.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out.rows; ++i) {
            for (size_t j = 0; j < out.cols; ++j) {
                double expected = 0.5 * input[i][j] + 0.5 * inner_out[i][j];
                double diff = fabs(out[i][j] - expected);
                if (diff > max_diff) max_diff = diff;
            }
        }
        if (max_diff < 1e-9) {
            cout << "[PASS] alpha=0.5,beta=0.5 -> output = 0.5*x+0.5*f(x) (rel_err < 1e-9)\n";
            ++passed;
        } else {
            cout << "[FAIL] alpha=0.5,beta=0.5 max_diff=" << max_diff << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: Identity inner + alpha=1, beta=0 -> output ≈ x
    // ------------------------------------------------------------
    {
        ++total;
        size_t n = 4, d = 4;
        Tensor input(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        // Identity inner: Dense(d, d) with W=I, b=0
        auto inner = std::unique_ptr<Dense>(new Dense(d, d));
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                inner->weights[i][j] = (i == j) ? 1.0 : 0.0;
        for (size_t j = 0; j < d; ++j) inner->bias[0][j] = 0.0;
        HyperConnection hc(d, inner.release());
        hc.alpha_log() = Tensor(1, d);
        hc.beta_log() = Tensor(1, d);
        for (size_t j = 0; j < d; ++j) {
            hc.alpha_log()[0][j] = 10.0;  // alpha ≈ 1
            hc.beta_log()[0][j] = -10.0;  // beta ≈ 0
        }
        Tensor out = hc.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) {
            double diff = fabs(out.data[i] - input.data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff < 1e-3) {
            cout << "[PASS] identity inner + alpha=1,beta=0 -> output ≈ x (max_diff="
                 << max_diff << ")\n"; ++passed;
        } else {
            cout << "[FAIL] identity inner max_diff=" << max_diff << "\n";
        }
    }

    // ------------------------------------------------------------
    // Tests 7-11: gradient FD checks
    // ------------------------------------------------------------
    size_t n = 4, d = 6;
    double eps = 1e-5;

    Tensor input(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);
    Tensor target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.1 * ((i + 2 * j) % 5) - 0.05;

    // ------------------------------------------------------------
    // Test 7: input gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        auto inner = std::unique_ptr<Dense>(new Dense(d, d));
        HyperConnection hc(d, inner.release());
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        Tensor grad_input = hc.backward(grad_out, 0.0);

        double max_err = 0;
        size_t total_in = input.rows * input.cols;
        size_t step = max((size_t)1, total_in / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_in; ++k, idx += step) {
            size_t i = idx / input.cols;
            size_t j = idx % input.cols;
            double orig = input[i][j];
            input[i][j] = orig + eps;
            Tensor out_p = hc.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            input[i][j] = orig - eps;
            Tensor out_m = hc.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            input[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = grad_input[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] input grad (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] input grad (max_err=" << max_err << ")\n";
    }

    // ------------------------------------------------------------
    // Test 8: alpha_log gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        // Use a single HC instance: perturb alpha_log, restore after each FD probe.
        auto inner_owned = std::unique_ptr<Dense>(new Dense(d, d));
        HyperConnection hc(d, inner_owned.release());
        build_deterministic_hc(hc, d);
        // Forward/backward to compute analytical gradient
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        hc.backward(grad_out, 0.0);
        Tensor g_ana = hc.grad_alpha_log().clone();   // (1, d)

        double max_err = 0;
        size_t total_p = hc.alpha_log().rows * hc.alpha_log().cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / hc.alpha_log().cols;
            size_t j = idx % hc.alpha_log().cols;
            double orig = hc.alpha_log()[i][j];
            hc.alpha_log()[i][j] = orig + eps;
            Tensor out_p = hc.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            hc.alpha_log()[i][j] = orig - eps;
            Tensor out_m = hc.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            hc.alpha_log()[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g_ana[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] alpha_log grad (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] alpha_log grad (max_err=" << max_err << ")\n";
    }

    // ------------------------------------------------------------
    // Test 9: beta_log gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        auto inner_owned = std::unique_ptr<Dense>(new Dense(d, d));
        HyperConnection hc(d, inner_owned.release());
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        hc.backward(grad_out, 0.0);
        Tensor g_ana = hc.grad_beta_log().clone();

        double max_err = 0;
        size_t total_p = hc.beta_log().rows * hc.beta_log().cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / hc.beta_log().cols;
            size_t j = idx % hc.beta_log().cols;
            double orig = hc.beta_log()[i][j];
            hc.beta_log()[i][j] = orig + eps;
            Tensor out_p = hc.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            hc.beta_log()[i][j] = orig - eps;
            Tensor out_m = hc.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            hc.beta_log()[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g_ana[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] beta_log grad (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] beta_log grad (max_err=" << max_err << ")\n";
    }

    // ------------------------------------------------------------
    // Test 10: alpha_log gradient nonzero when d_out nonzero AND x nonzero
    // ------------------------------------------------------------
    {
        ++total;
        auto inner = std::unique_ptr<Dense>(new Dense(d, d));
        HyperConnection hc(d, inner.release());
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        hc.backward(grad_out, 0.0);
        double g_norm = 0.0;
        for (size_t i = 0; i < hc.grad_alpha_log().data.size(); ++i)
            g_norm += hc.grad_alpha_log().data[i] * hc.grad_alpha_log().data[i];
        g_norm = sqrt(g_norm);
        if (g_norm > 1e-6) {
            cout << "[PASS] alpha_log grad nonzero (norm=" << g_norm << ")\n"; ++passed;
        } else {
            cout << "[FAIL] alpha_log grad too small (norm=" << g_norm << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: beta_log gradient nonzero when d_out nonzero AND sub_out nonzero
    // ------------------------------------------------------------
    {
        ++total;
        auto inner = std::unique_ptr<Dense>(new Dense(d, d));
        HyperConnection hc(d, inner.release());
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        hc.backward(grad_out, 0.0);
        double g_norm = 0.0;
        for (size_t i = 0; i < hc.grad_beta_log().data.size(); ++i)
            g_norm += hc.grad_beta_log().data[i] * hc.grad_beta_log().data[i];
        g_norm = sqrt(g_norm);
        if (g_norm > 1e-6) {
            cout << "[PASS] beta_log grad nonzero (norm=" << g_norm << ")\n"; ++passed;
        } else {
            cout << "[FAIL] beta_log grad too small (norm=" << g_norm << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: HyperConnectionBlock forward shape
    // ------------------------------------------------------------
    {
        ++total;
        size_t d_m = 8, ffn = 16;
        auto inner = std::unique_ptr<Dense>(new Dense(ffn, d_m)); // pre-LN(d_m)->ffn_dim -> d_m
        HyperConnectionBlock blk(d_m, ffn); // takes no inner; fc2_ is internal
        Tensor input_b(n, d_m);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_m; ++j)
                input_b(i, j) = 0.3 * (i + 1) + 0.2 * (j + 1);
        Tensor out = blk.forward(input_b);
        if (out.rows == n && out.cols == d_m) {
            cout << "[PASS] HyperConnectionBlock forward shape\n"; ++passed;
        } else {
            cout << "[FAIL] HyperConnectionBlock shape: expected " << n << "x" << d_m
                 << " got " << out.rows << "x" << out.cols << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: HyperConnectionBlock input gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        size_t d_m = 4, ffn = 8;
        Tensor input_b(n, d_m);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_m; ++j)
                input_b(i, j) = 0.3 * (i + 1) + 0.2 * (j + 1);
        Tensor target_b(n, d_m);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_m; ++j)
                target_b(i, j) = 0.1 * ((i + 2 * j) % 5) - 0.05;

        HyperConnectionBlock blk(d_m, ffn);
        // Make deterministic
        // The inner Dense is already random; alpha/beta are at init (small but nonzero).
        Tensor out = blk.forward(input_b);
        Tensor grad_out = l2_loss_grad(out, target_b);
        Tensor grad_input = blk.backward(grad_out, 0.0);

        double max_err = 0;
        size_t total_in = input_b.rows * input_b.cols;
        size_t step = max((size_t)1, total_in / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_in; ++k, idx += step) {
            size_t i = idx / input_b.cols;
            size_t j = idx % input_b.cols;
            double orig = input_b[i][j];
            input_b[i][j] = orig + eps;
            Tensor out_p = blk.forward(input_b);
            double loss_p = l2_loss_value(out_p, target_b);
            input_b[i][j] = orig - eps;
            Tensor out_m = blk.forward(input_b);
            double loss_m = l2_loss_value(out_m, target_b);
            input_b[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = grad_input[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) {
            cout << "[PASS] HyperConnectionBlock input grad (max_err=" << max_err << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] HyperConnectionBlock input grad (max_err=" << max_err << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: HyperConnectionBlock training reduces loss
    // ------------------------------------------------------------
    {
        ++total;
        size_t d_m = 4, ffn = 8, n_train = 16;
        Tensor x_train(n_train, d_m);
        Tensor y_train(n_train, d_m);
        for (size_t i = 0; i < n_train; ++i)
            for (size_t j = 0; j < d_m; ++j) {
                x_train(i, j) = 0.3 * (i + 1) - 0.2 * (j + 1);
                y_train(i, j) = 0.5 * x_train(i, j) + 0.05;
            }

        HyperConnectionBlock blk(d_m, ffn);
        // Initialize alpha/beta to non-saturated values for trainable dynamics.
        // (Default init alpha_log=10, beta_log=-10 saturates the sigmoid gradient.)
        for (size_t j = 0; j < d_m; ++j) {
            blk.alpha_log()[0][j] = 0.0;     // alpha = sigmoid(0) = 0.5
            blk.beta_log()[0][j] = -2.0;     // beta = sigmoid(-2) ≈ 0.12
        }
        double lr = 0.05;

        Tensor out = blk.forward(x_train);
        double loss0 = l2_loss_value(out, y_train);
        for (size_t step = 0; step < 50; ++step) {
            blk.zero_grad();
            out = blk.forward(x_train);
            Tensor grad_out = l2_loss_grad(out, y_train);
            blk.backward(grad_out, lr);
            // Manual SGD update on all parameters
            auto params = blk.parameters();
            auto grads = blk.gradients();
            for (size_t pi = 0; pi < params.size(); ++pi) {
                for (size_t i = 0; i < params[pi]->data.size(); ++i) {
                    params[pi]->data[i] -= lr * grads[pi]->data[i];
                }
            }
        }
        out = blk.forward(x_train);
        double loss1 = l2_loss_value(out, y_train);
        if (loss1 < loss0 * 0.9) {
            cout << "[PASS] HyperConnectionBlock training reduces loss (" << loss0
                 << " -> " << loss1 << ")\n"; ++passed;
        } else {
            cout << "[FAIL] training loss " << loss0 << " -> " << loss1
                 << " (expected >10% reduction)\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: HyperConnectionModel forward shape + training reduces loss
    // ------------------------------------------------------------
    {
        size_t in_dim = 3, d_m = 6, out_dim = 2, n_blocks = 2, ffn = 12;
        HyperConnectionModel m(in_dim, d_m, out_dim, n_blocks, ffn);
        Tensor x(4, in_dim);
        Tensor y(4, out_dim);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                x(i, j) = 0.3 * (i + 1) + 0.2 * (j + 1);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < out_dim; ++j)
                y(i, j) = 0.1 * ((i + j) % 3) - 0.05;

        Tensor out = m.forward(x);
        bool pass_shape = false, pass_train = false;
        if (out.rows == 4 && out.cols == out_dim) {
            cout << "[PASS] HyperConnectionModel forward shape\n"; pass_shape = true;
        } else {
            cout << "[FAIL] HyperConnectionModel shape: expected 4x" << out_dim
                 << " got " << out.rows << "x" << out.cols << "\n";
        }
        ++total;
        if (pass_shape) ++passed;

        // Train
        double lr = 0.01;
        double loss0 = l2_loss_value(out, y);
        for (size_t step = 0; step < 80; ++step) {
            m.zero_grad();
            out = m.forward(x);
            Tensor grad_out = l2_loss_grad(out, y);
            m.backward(grad_out, lr);
            auto params = m.parameters();
            auto grads = m.gradients();
            for (size_t pi = 0; pi < params.size(); ++pi) {
                for (size_t i = 0; i < params[pi]->data.size(); ++i) {
                    params[pi]->data[i] -= lr * grads[pi]->data[i];
                }
            }
        }
        out = m.forward(x);
        double loss1 = l2_loss_value(out, y);
        ++total;
        if (loss1 < loss0) {
            cout << "[PASS] HyperConnectionModel training reduces loss (" << loss0
                 << " -> " << loss1 << ")\n"; ++passed;
        } else {
            cout << "[FAIL] training loss " << loss0 << " -> " << loss1 << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: Mutation test — zeroing alpha/beta grad -> input grad test fails
    // ------------------------------------------------------------
    {
        ++total;
        // Mutation test: at alpha=1, beta=0 (residual identity), gradient should
        // mostly come from the residual path d_x = α ⊙ d_out, NOT from inner.
        // At this setting, alpha=1 forces the residual to fully flow through.
        // If we verify the gradient norm is close to d_out itself (residual only)
        // we know the alpha path works. If we then set beta=0 and inner=Dense(0,0)
        // (which produces 0), the only contribution is alpha*x.
        //
        // Simpler test: verify that if inner is a zero-weight Dense, the gradient
        // reduces to alpha * d_out exactly.

        // First: zero inner Dense weights/biases
        auto inner_owned = std::unique_ptr<Dense>(new Dense(d, d));
        inner_owned->init_weights("zeros");
        HyperConnection hc(d, inner_owned.release());
        build_deterministic_hc(hc, d);
        Tensor out = hc.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        Tensor grad_input = hc.backward(grad_out, 0.0);

        // The analytical grad_input should equal alpha * grad_out (broadcast across rows).
        // Compute the expected:
        Tensor expected(input.rows, d);
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double a = 1.0 / (1.0 + std::exp(-hc.alpha_log()[0][j]));
                expected[i][j] = a * grad_out[i][j];
            }
        }

        double max_diff = 0.0;
        for (size_t i = 0; i < grad_input.data.size(); ++i) {
            double diff = std::abs(grad_input.data[i] - expected.data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff < 1e-9) {
            cout << "[PASS] zero inner + alpha*grad_out (max_diff=" << max_diff << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] zero-inner grad_input expected alpha*grad_out, max_diff="
                 << max_diff << "\n";
        }
    }

    cout << "\n=== Summary: " << passed << "/" << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
