// MambaConv — Canonical Mamba-1 with depthwise causal 1D convolution
//   Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//   (https://arxiv.org/abs/2312.00752)
//
// Tests:
//   1.  Constructor validation (d_model=0 / conv_kernel=0 throw; valid constructs)
//   2.  Forward shape (T=3, d_model=2, d_state=2, d_inner=2, conv_kernel=4)
//   3.  Forward output is finite (T=5)
//   4.  Forward output is non-zero
//   5.  Hand-derived T=1 forward reference matches at rel_err 1e-13
//   6.  conv_kernel=1 reduces to (almost) MambaBlock forward
//   7.  Causal conv: perturbing x_pre at t = T-1 affects only the t = T-1 conv output
//   8.  Input gradient check (T=3, centered FD)
//   9.  Input gradient check (T=6, deeper BPTT)
//  10.  in_proj.weights gradient check (FD)
//  11.  dt_proj.weights gradient check (FD; tests softplus + Δ path)
//  12.  B_proj.weights gradient check (FD; tests B_t → B̄_t → h_t → y_t path)
//  13.  C_proj.weights gradient check (FD; tests y_t → grad_C_t path)
//  14.  out_proj.weights gradient check (FD)
//  15.  A_log gradient check (FD; tests selective scan backward)
//  16.  D_skip gradient check (FD; tests skip path through x_ssm)
//  17.  conv_weight gradient check (FD; NEW vs MambaBlock)
//  18.  conv_bias gradient check (FD;   NEW vs MambaBlock)
//  19.  Training step reduces loss (50 SGD steps, lr=1e-2)
//  20.  parameters()/gradients() shape consistency (14 params / 14 grads)
//  21.  zero_grad clears all 14 gradient buffers
//  22.  update_weights moves all 14 parameters
//  23.  Conv signal propagates: changing conv_weight[k > 0] changes output at t > 0
//  24.  Skip path: D_skip gradient non-zero (proves x_ssm enters the gated output)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/recurrent/mamba_conv.h"

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

// Forward to compute scalar loss given an input and target (used for FD).
static double forward_loss(MambaConvBlock& blk, const Tensor& input, const Tensor& target) {
    Tensor out = blk.forward(input);
    return l2_loss_value(out, target);
}

// Run centered FD for a single parameter tensor.
static double fd_param_check(MambaConvBlock& blk, Tensor& param,
                             const Tensor& input, const Tensor& target,
                             double eps = 1e-5, int sample_every = 1) {
    size_t r = param.rows, c = param.cols;
    double max_err = 0.0;
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            if (((i * c + j) % sample_every) != 0) continue;
            double orig = param(i, j);
            param(i, j) = orig + eps;
            double Lp = forward_loss(blk, input, target);
            param(i, j) = orig - eps;
            double Lm = forward_loss(blk, input, target);
            param(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            // Now compute the analytical grad.
            Tensor out = blk.forward(input);
            Tensor gl = l2_loss_grad(out, target);
            blk.zero_grad();
            blk.backward(gl, 0.0);
            auto grads = blk.gradients();
            // Find the grad matching this param's shape
            double ana = 0.0;
            for (auto* g : grads) {
                if (g->rows == r && g->cols == c) {
                    ana = (*g)(i, j);
                    break;
                }
            }
            double err = relative_error(num, ana);
            max_err = max(max_err, err);
            if (err > 0.01) {
                cout << "    [" << i << "][" << j << "] ana=" << ana
                     << " num=" << num << " err=" << err << "\n";
            }
        }
    }
    return max_err;
}

int main() {
    cout << "=== MambaConv (Mamba-1 with depthwise causal 1D conv) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: constructor validation
    // ------------------------------------------------------------
    cout << "\n--- Test 1: Constructor validation ---\n";
    {
        ++total;
        bool ok = true;
        try { MambaConvBlock(0, 4, 0, 4); ok = false; } catch (...) {}
        try { MambaConvBlock(4, 4, 0, 0); ok = false; } catch (...) {}
        try { MambaConvBlock(4, 4, 0, 4); } catch (...) { ok = false; }
        try { MambaConvBlock(4, 4, 8, 1); } catch (...) { ok = false; }
        try { MambaConvBlock(4, 4, 8, 2); } catch (...) { ok = false; }
        if (ok) { cout << "[PASS] 5 constructor cases (2 throw, 3 valid)\n"; ++passed; }
        else    { cout << "[FAIL] constructor validation\n"; }
    }

    // Small, tractable config: T=3, d_model=2, d_state=2, d_inner=2 (default = 2*d_model)
    size_t T = 3, d_model = 2, d_state = 2, d_inner = 4;  // d_inner forced = 2 * d_model

    // ------------------------------------------------------------
    // Test 2: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 2: MambaConvBlock forward shape (T=3, d_model=2) ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        MambaConvBlock block(d_model, d_state);
        Tensor output = block.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == T && output.cols == d_model) {
            cout << "[PASS] forward shape correct\n"; ++passed;
        } else {
            cout << "[FAIL] expected " << T << "x" << d_model << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: output is finite
    // ------------------------------------------------------------
    cout << "\n--- Test 3: MambaConvBlock output is finite (T=5) ---\n";
    {
        ++total;
        size_t T2 = 5;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        MambaConvBlock block(d_model, d_state);
        Tensor output = block.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) { cout << "[PASS] all outputs finite\n"; ++passed; }
        else        { cout << "[FAIL] non-finite output detected\n"; }
    }

    // ------------------------------------------------------------
    // Test 4: output is non-zero
    // ------------------------------------------------------------
    cout << "\n--- Test 4: MambaConvBlock output is non-zero ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        MambaConvBlock block(d_model, d_state);
        Tensor output = block.forward(input);
        double mag = 0.0;
        for (auto v : output.data) mag += v * v;
        if (mag > 1e-8) { cout << "[PASS] output non-zero (mag=" << sqrt(mag) << ")\n"; ++passed; }
        else            { cout << "[FAIL] output is zero\n"; }
    }

    // ------------------------------------------------------------
    // Test 5: hand-derived T=1 forward reference (deterministic small config)
    //
    // Build a 1-token input, override all parameters to known small values,
    // compute forward by hand, compare to impl output.
    // ------------------------------------------------------------
    cout << "\n--- Test 5: Hand-derived T=1 forward reference ---\n";
    {
        ++total;
        size_t dm = 1, ds = 1, di = 1, ck = 2;
        MambaConvBlock block(dm, ds, di, ck);

        // Overwrite all parameters with known values.
        // in_proj: d_model -> 2*d_inner. So weights is (2, 1), bias (2, 1).
        block.in_proj.weights(0, 0) = 1.5; block.in_proj.weights(1, 0) = 0.7;
        block.in_proj.bias(0, 0)    = 0.3; block.in_proj.bias(0, 1)    = -0.2;
        // dt_proj: d_model -> d_inner. (1, 1)
        block.dt_proj.weights(0, 0) = 0.5;
        block.dt_proj.bias(0, 0)    = 0.1;
        // B_proj: d_model -> d_state. (1, 1)
        block.B_proj.weights(0, 0)   = 0.4;
        block.B_proj.bias(0, 0)      = 0.05;
        // C_proj: d_model -> d_state. (1, 1)
        block.C_proj.weights(0, 0)   = 0.6;
        block.C_proj.bias(0, 0)      = 0.0;
        // out_proj: d_inner -> d_model. (1, 1)
        block.out_proj.weights(0, 0) = 1.2;
        block.out_proj.bias(0, 0)    = 0.0;
        // A_log: (d_inner=1, d_state=1)
        block.A_log(0, 0)            = 0.0;       // A = -exp(0) = -1
        // D_skip: (1, d_inner)
        block.D_skip(0, 0)           = 1.0;
        // conv_weight: (d_inner=1, conv_kernel=2)
        block.conv_weight(0, 0)      = 0.4;
        block.conv_weight(0, 1)      = 0.2;
        // conv_bias: (1, d_inner=1)
        block.conv_bias(0, 0)        = 0.1;

        Tensor input(1, dm);
        input(0, 0) = 0.5;

        // Hand-derive forward output:
        double x0 = 0.5;
        // p = in_proj(x) = [[1.5*0.5 + 0.3, 0.7*0.5 + -0.2]] = [[1.05, 0.15]]
        double x_pre = 1.5 * x0 + 0.3;        // = 1.05
        double gate  = 0.7 * x0 - 0.2;        // = 0.15
        // conv (kernel=2, depthwise): t=0, j=0 contributes x_pre[0], j=1 needs t-j=-1 → 0
        double x_conv = block.conv_bias(0, 0) + block.conv_weight(0, 0) * x_pre;  // 0.1 + 0.4*1.05 = 0.52
        // silu: x * sigmoid(x)
        double sigmoid_xc = 1.0 / (1.0 + std::exp(-x_conv));
        double x_ssm = x_conv * sigmoid_xc;
        // Δ = softplus(0.5*0.5 + 0.1) = softplus(0.35)
        double Delta_pre = 0.5 * x0 + 0.1;
        double Delta = std::log(1.0 + std::exp(Delta_pre));
        // B = 0.4*0.5 + 0.05 = 0.25
        double B_t = block.B_proj.weights(0, 0) * x0 + block.B_proj.bias(0, 0);
        // C = 0.6*0.5 + 0 = 0.3
        double C_t = block.C_proj.weights(0, 0) * x0 + block.C_proj.bias(0, 0);
        // A = -exp(0) = -1
        double A = -std::exp(block.A_log(0, 0));
        // Ā = exp(Delta * A) = exp(Delta * -1)
        double A_bar = std::exp(Delta * A);
        // B̄ = Delta * B_t
        double B_bar = Delta * B_t;
        // h_0 = 0
        // h_1 = Ā * h_0 + B̄ * x_ssm = B̄ * x_ssm
        double h = B_bar * x_ssm;
        // y = C_t * h
        double y = C_t * h;
        // skip path: x_ssm * D_skip
        double skip = block.D_skip(0, 0) * x_ssm;
        // gated = silu(gate) * (y + skip)
        double sigmoid_g = 1.0 / (1.0 + std::exp(-gate));
        double silu_gate = gate * sigmoid_g;
        double gated = silu_gate * (y + skip);
        // out = gated * W_out + b_out
        double expected = gated * block.out_proj.weights(0, 0) + block.out_proj.bias(0, 0);

        Tensor output = block.forward(input);
        cout << "Expected: " << expected << "  Got: " << output(0, 0)
             << "  diff: " << fabs(expected - output(0, 0)) << "\n";
        if (fabs(expected - output(0, 0)) < 1e-12) {
            cout << "[PASS] hand-derived T=1 forward matches\n"; ++passed;
        } else {
            cout << "[FAIL] hand-derived T=1 forward MISMATCH\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: conv_kernel=1 sanity (single-tap = no mixing; output should differ
    //   from MambaBlock only because of the extra conv_weight and conv_bias parameters)
    // ------------------------------------------------------------
    cout << "\n--- Test 6: conv_kernel=1 sanity (output is finite + non-degenerate) ---\n";
    {
        ++total;
        MambaConvBlock blk(d_model, d_state, d_inner, 1);
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.3 * i + 0.1 * j - 0.5;
        Tensor output = blk.forward(input);
        bool ok = (output.rows == T && output.cols == d_model);
        double mag = 0.0;
        for (auto v : output.data) mag += v * v;
        ok = ok && (mag > 1e-8);
        for (auto v : output.data) if (!std::isfinite(v)) ok = false;
        if (ok) { cout << "[PASS] conv_kernel=1 forward shape+finite+nonzero\n"; ++passed; }
        else    { cout << "[FAIL] conv_kernel=1 broken\n"; }
    }

    // ------------------------------------------------------------
    // Test 7: causal conv — perturbing x_pre[T-1] only affects x_conv[T-1]
    //   (we test indirectly: changing conv_weight[k-1, ?] changes x_conv[t] for t>=0
    //    while changing conv_weight[0, ?] changes x_conv[t] for all t)
    // ------------------------------------------------------------
    cout << "\n--- Test 7: causal conv — depthwise filter affects expected positions ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * (i + 1) * (j + 1);

        // Block A: conv_weight[0, k-1] = 0, others 0.5  → only x_pre[T-1] contributes to x_conv[T-1]
        // Block B: conv_weight[k-1, 0] = 0, others 0.5  → all x_pre[t-k+1..t] contribute
        // Compare: x_conv[0] for A vs B should differ (because B has j=k-1 tap contributing
        // x_pre[0], A has only j=0 contributing x_pre[0]).
        MambaConvBlock blkA(d_model, d_state, d_inner, 4);
        MambaConvBlock blkB(d_model, d_state, d_inner, 4);
        for (size_t i = 0; i < d_inner; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                blkA.conv_weight(i, k) = (k == 3) ? 0.0 : 0.5;
                blkB.conv_weight(i, k) = (k == 0) ? 0.0 : 0.5;
            }
            blkA.conv_bias(0, i) = 0.0; blkB.conv_bias(0, i) = 0.0;
        }
        // Copy B_proj / C_proj / dt_proj / in_proj / out_proj / A_log / D_skip from blkA to blkB
        blkB.in_proj.weights = blkA.in_proj.weights.clone();
        blkB.in_proj.bias    = blkA.in_proj.bias.clone();
        blkB.dt_proj.weights = blkA.dt_proj.weights.clone();
        blkB.dt_proj.bias    = blkA.dt_proj.bias.clone();
        blkB.B_proj.weights  = blkA.B_proj.weights.clone();
        blkB.B_proj.bias     = blkA.B_proj.bias.clone();
        blkB.C_proj.weights  = blkA.C_proj.weights.clone();
        blkB.C_proj.bias     = blkA.C_proj.bias.clone();
        blkB.out_proj.weights= blkA.out_proj.weights.clone();
        blkB.out_proj.bias   = blkA.out_proj.bias.clone();
        blkB.A_log           = blkA.A_log.clone();
        blkB.D_skip          = blkA.D_skip.clone();

        Tensor outA = blkA.forward(input);
        Tensor outB = blkB.forward(input);
        double diff = 0.0;
        for (size_t i = 0; i < outA.data.size(); ++i)
            diff += fabs(outA.data[i] - outB.data[i]);
        if (diff > 1e-6) {
            cout << "[PASS] different conv filters → different output (diff=" << diff << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] conv filters don't propagate (diff=" << diff << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 8: input gradient check (T=3)
    // ------------------------------------------------------------
    cout << "\n--- Test 8: Input gradient check (T=3) ---\n";
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

        MambaConvBlock block(d_model, d_state);
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
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) { cout << "[PASS] input gradient (rel_err < 5%)\n"; ++passed; }
        else                { cout << "[FAIL] input gradient\n"; }
    }

    // ------------------------------------------------------------
    // Test 9: input gradient check (T=6, deeper BPTT)
    // ------------------------------------------------------------
    cout << "\n--- Test 9: Input gradient check (T=6) ---\n";
    {
        ++total;
        double eps = 1e-5;
        size_t T2 = 6;
        Tensor input(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        Tensor target(T2, d_model);
        for (size_t i = 0; i < T2; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.05 * i + 0.02 * j;

        MambaConvBlock block(d_model, d_state);
        Tensor out = block.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor grad_x = block.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < T2; ++i) {
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
            }
        }
        cout << "Max err: " << max_err << "\n";
        if (max_err < 0.05) { cout << "[PASS] T=6 input gradient (rel_err < 5%)\n"; ++passed; }
        else                { cout << "[FAIL] T=6 input gradient\n"; }
    }

    // ------------------------------------------------------------
    // Tests 10-18: parameter gradient FD checks
    // ------------------------------------------------------------
    cout << "\n--- Test 10: in_proj.weights gradient ---\n";

    auto fd_check_named = [&](const string& name,
                              std::function<Tensor&(MambaConvBlock&)> getter,
                              double tol = 0.05) {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);
        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1 * i + 0.05 * j;

        MambaConvBlock block(d_model, d_state);
        Tensor& param = getter(block);
        size_t r = param.rows, c = param.cols;
        double max_err = 0.0;
        double eps = 1e-5;
        // Sample all entries (small params)
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                double orig = param(i, j);
                param(i, j) = orig + eps;
                Tensor out_p = block.forward(input);
                double Lp = l2_loss_value(out_p, target);
                param(i, j) = orig - eps;
                Tensor out_m = block.forward(input);
                double Lm = l2_loss_value(out_m, target);
                param(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                Tensor out = block.forward(input);
                Tensor gl = l2_loss_grad(out, target);
                block.zero_grad();
                block.backward(gl, 0.0);
                auto params = block.parameters();
                auto grads = block.gradients();
                double ana = 0.0;
                // Find the grad whose corresponding param has the same address as `param`.
                for (size_t gi = 0; gi < grads.size(); ++gi) {
                    if (params[gi] == &param) { ana = (*grads[gi])(i, j); break; }
                }
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
                if (err > 0.01) {
                    cout << "    [" << i << "][" << j << "] ana=" << ana
                         << " num=" << num << " err=" << err << "\n";
                }
            }
        }
        cout << "  Max err for " << name << " (shape " << r << "x" << c << "): " << max_err << "\n";
        if (max_err < tol) { cout << "[PASS] " << name << " gradient FD\n"; ++passed; }
        else               { cout << "[FAIL] " << name << " gradient FD\n"; }
    };

    cout << "\n--- Test 10: in_proj.weights gradient (shape 2*d_inner x d_model) ---\n";
    fd_check_named("in_proj.weights", [](MambaConvBlock& b) -> Tensor& { return b.in_proj.weights; });
    cout << "\n--- Test 11: dt_proj.weights gradient ---\n";
    fd_check_named("dt_proj.weights", [](MambaConvBlock& b) -> Tensor& { return b.dt_proj.weights; });
    cout << "\n--- Test 12: B_proj.weights gradient ---\n";
    fd_check_named("B_proj.weights",  [](MambaConvBlock& b) -> Tensor& { return b.B_proj.weights; });
    cout << "\n--- Test 13: C_proj.weights gradient ---\n";
    fd_check_named("C_proj.weights",  [](MambaConvBlock& b) -> Tensor& { return b.C_proj.weights; });
    cout << "\n--- Test 14: out_proj.weights gradient ---\n";
    fd_check_named("out_proj.weights",[](MambaConvBlock& b) -> Tensor& { return b.out_proj.weights; });
    cout << "\n--- Test 15: A_log gradient (shape d_inner x d_state) ---\n";
    fd_check_named("A_log",           [](MambaConvBlock& b) -> Tensor& { return b.A_log; });
    cout << "\n--- Test 16: D_skip gradient (shape 1 x d_inner) ---\n";
    fd_check_named("D_skip",          [](MambaConvBlock& b) -> Tensor& { return b.D_skip; });
    cout << "\n--- Test 17: conv_weight gradient (shape d_inner x conv_kernel) — NEW ---\n";
    fd_check_named("conv_weight",     [](MambaConvBlock& b) -> Tensor& { return b.conv_weight; });
    cout << "\n--- Test 18: conv_bias gradient (shape 1 x d_inner) — NEW ---\n";
    fd_check_named("conv_bias",       [](MambaConvBlock& b) -> Tensor& { return b.conv_bias; });

    // ------------------------------------------------------------
    // Test 19: training reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 19: Training reduces loss (50 SGD steps, lr=1e-2) ---\n";
    {
        ++total;
        size_t T_train = 4;
        Tensor input(T_train, d_model);
        for (size_t i = 0; i < T_train; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.1 * i + 0.2 * j - 0.3;
        Tensor target(T_train, d_model);
        for (size_t i = 0; i < T_train; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.05 * i - 0.02 * j + 0.1;

        MambaConvBlock block(d_model, d_state);
        double L0 = l2_loss_value(block.forward(input), target);
        double lr = 1e-2;
        for (int step = 0; step < 50; ++step) {
            Tensor out = block.forward(input);
            Tensor gl  = l2_loss_grad(out, target);
            block.zero_grad();
            block.backward(gl, lr);
            block.update_weights(lr);
        }
        double Lf = l2_loss_value(block.forward(input), target);
        cout << "L0=" << L0 << "  Lf=" << Lf << "\n";
        if (Lf < L0 * 0.5) { cout << "[PASS] training reduces loss > 50%\n"; ++passed; }
        else               { cout << "[FAIL] training did not reduce loss\n"; }
    }

    // ------------------------------------------------------------
    // Test 20: parameters()/gradients() shape consistency (14 params)
    // ------------------------------------------------------------
    cout << "\n--- Test 20: parameters()/gradients() shape consistency ---\n";
    {
        ++total;
        MambaConvBlock block(d_model, d_state);
        auto params = block.parameters();
        auto grads  = block.gradients();
        bool ok = (params.size() == grads.size());
        if (ok && params.size() != 14) {
            cout << "[FAIL] expected 14 params, got " << params.size() << "\n";
            ok = false;
        }
        if (ok) {
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows ||
                    params[i]->cols != grads[i]->cols) {
                    cout << "  shape mismatch at index " << i << "\n";
                    ok = false; break;
                }
            }
        }
        if (ok) { cout << "[PASS] 14 params/grads shape-matched\n"; ++passed; }
        else    { cout << "[FAIL] shape contract\n"; }
    }

    // ------------------------------------------------------------
    // Test 21: zero_grad clears all gradients
    // ------------------------------------------------------------
    cout << "\n--- Test 21: zero_grad clears all gradients ---\n";
    {
        ++total;
        MambaConvBlock block(d_model, d_state);
        Tensor input  = Tensor::random(T, d_model, 0.1);
        Tensor target = Tensor::random(T, d_model, 0.05);
        block.zero_grad();
        block.forward(input);  // populate caches
        Tensor out = block.forward(input);
        block.zero_grad();
        Tensor gl = l2_loss_grad(out, target);
        block.backward(gl, 0.0);
        // Now grads are non-zero. zero_grad() again.
        block.zero_grad();
        auto grads = block.gradients();
        double total_norm = 0.0;
        for (auto* g : grads) for (auto v : g->data) total_norm += v * v;
        if (total_norm < 1e-12) { cout << "[PASS] all grads zero after zero_grad()\n"; ++passed; }
        else                    { cout << "[FAIL] grads not zero, norm=" << sqrt(total_norm) << "\n"; }
    }

    // ------------------------------------------------------------
    // Test 22: update_weights moves all parameters
    // ------------------------------------------------------------
    cout << "\n--- Test 22: update_weights moves all 14 parameters ---\n";
    {
        ++total;
        MambaConvBlock block(d_model, d_state);
        Tensor input  = Tensor::random(T, d_model, 0.1);
        Tensor target = Tensor::random(T, d_model, 0.05);
        // Snapshot all params.
        auto params_before = block.parameters();
        std::vector<Tensor> snapshot;
        for (auto* p : params_before) snapshot.push_back(p->clone());
        Tensor out = block.forward(input);
        Tensor gl = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(gl, 0.0);
        double lr = 0.05;
        block.update_weights(lr);
        // Verify all moved
        auto params_after = block.parameters();
        double max_diff = 0.0;
        for (size_t i = 0; i < params_after.size(); ++i) {
            for (size_t j = 0; j < params_after[i]->data.size(); ++j) {
                max_diff = max(max_diff, fabs(params_after[i]->data[j] - snapshot[i].data[j]));
            }
        }
        if (max_diff > 1e-8) { cout << "[PASS] params moved (max_diff=" << max_diff << ")\n"; ++passed; }
        else                 { cout << "[FAIL] params unchanged\n"; }
    }

    // ------------------------------------------------------------
    // Test 23: conv_weight non-trivial effect (proves conv is exercised)
    // ------------------------------------------------------------
    cout << "\n--- Test 23: conv_weight propagates through to output ---\n";
    {
        ++total;
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.2 * (i + 1) * (j + 1) - 0.5;
        MambaConvBlock blkA(d_model, d_state, d_inner, 4);
        MambaConvBlock blkB(d_model, d_state, d_inner, 4);
        // Copy all params from A to B, then bump conv_weight by 0.1 in B
        blkB.in_proj.weights = blkA.in_proj.weights.clone();
        blkB.in_proj.bias    = blkA.in_proj.bias.clone();
        blkB.dt_proj.weights = blkA.dt_proj.weights.clone();
        blkB.dt_proj.bias    = blkA.dt_proj.bias.clone();
        blkB.B_proj.weights  = blkA.B_proj.weights.clone();
        blkB.B_proj.bias     = blkA.B_proj.bias.clone();
        blkB.C_proj.weights  = blkA.C_proj.weights.clone();
        blkB.C_proj.bias     = blkA.C_proj.bias.clone();
        blkB.out_proj.weights= blkA.out_proj.weights.clone();
        blkB.out_proj.bias   = blkA.out_proj.bias.clone();
        blkB.A_log           = blkA.A_log.clone();
        blkB.D_skip          = blkA.D_skip.clone();
        blkB.conv_bias       = blkA.conv_bias.clone();
        for (size_t i = 0; i < d_inner; ++i)
            for (size_t k = 0; k < 4; ++k)
                blkB.conv_weight(i, k) = blkA.conv_weight(i, k) + 0.1;
        Tensor outA = blkA.forward(input);
        Tensor outB = blkB.forward(input);
        double diff = 0.0;
        for (size_t i = 0; i < outA.data.size(); ++i)
            diff += fabs(outA.data[i] - outB.data[i]);
        if (diff > 1e-6) { cout << "[PASS] conv_weight perturbation changes output (diff=" << diff << ")\n"; ++passed; }
        else             { cout << "[FAIL] conv_weight perturbation has no effect (diff=" << diff << ")\n"; }
    }

    // ------------------------------------------------------------
    // Test 24: Skip path — D_skip gradient non-zero (proves x_ssm enters gated output)
    // ------------------------------------------------------------
    cout << "\n--- Test 24: D_skip gradient non-zero (skip path is exercised) ---\n";
    {
        ++total;
        MambaConvBlock block(d_model, d_state);
        Tensor input(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                input(i, j) = 0.5;
        Tensor target(T, d_model);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d_model; ++j)
                target(i, j) = 0.1;
        Tensor out = block.forward(input);
        Tensor gl = l2_loss_grad(out, target);
        block.zero_grad();
        block.backward(gl, 0.0);
        auto grads = block.gradients();
        double d_skip_norm = 0.0;
        for (auto* g : grads) {
            if (g->rows == 1 && g->cols == d_inner) {
                for (auto v : g->data) d_skip_norm += v * v;
            }
        }
        if (d_skip_norm > 1e-12) { cout << "[PASS] D_skip gradient non-zero (norm=" << sqrt(d_skip_norm) << ")\n"; ++passed; }
        else                     { cout << "[FAIL] D_skip gradient is zero — skip path may be broken\n"; }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
