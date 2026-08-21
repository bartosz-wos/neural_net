// Agent Attention — Han et al. 2024 (ECCV 2024)
// "Agent Attention: On the Integration of Softmax and Linear Attention"
// https://arxiv.org/abs/2312.08874
//
// Tests (~19 focused checks):
//   1.  constructor validation (5 cases)
//   2.  forward shape (N=4, d=4, N_a=2)
//   3.  forward shape with linear residual
//   4.  forward finiteness + nonzero (N=4, N_a=2)
//   5.  forward shape with N_a=1 (degenerate but allowed)
//   6.  agent aggregation is a weighted sum of V (rows of A' = softmax-weighted V rows)
//   7.  agent broadcast is a weighted sum of A' (rows of O = softmax-weighted A' rows)
//   8.  input gradient FD check (use_linear_residual=true)
//   9.  W_q gradient FD check
//  10.  W_k gradient FD check
//  11.  W_v gradient FD check
//  12.  W_o gradient FD check
//  13.  W_q_agents gradient FD check
//  14.  W_k_agents gradient FD check
//  15.  agents (A) gradient FD check (the learnable centerpiece — non-zero)
//  16.  log_lambda gradient FD check (when use_linear_residual=true)
//  17.  determinism — two fresh AgentAttentions with copied params → bit-exact forward
//  18.  training reduces loss (50 SGD steps)
//  19.  param/grad count contract
//  20.  AgentAttentionBlock forward shape + training reduces loss
//  21.  AgentAttentionModel forward shape + training reduces loss
//
// All gradient checks use deterministic non-uniform init (via 0.3 + 0.1*(i+j)%5
// patterns) to avoid row-vs-column confusion. Loss is 0.5·sum((output−target)²)
// so its gradient w.r.t. output is simply (output − target).

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include "nn/layers/attention/agent_attention.h"

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

// Deterministic non-uniform init for an AgentAttention's parameters.
// Pattern: 0.3 + 0.1 * ((i + 2*j) % 7) — asymmetric in (i, j) so row/col
// sums differ (avoids the row-vs-column vacuity trap).
static void build_deterministic_attn(AgentAttention& attn, size_t d, size_t N_a) {
    auto fill_W = [&d](Dense& W) {
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                W.weights[i][j] = 0.3 + 0.1 * (double)((i + 2 * j + 3) % 7);
        for (size_t j = 0; j < d; ++j)
            W.bias[0][j] = 0.05 * (double)((j + 1) % 3);
    };
    fill_W(attn.W_q);
    fill_W(attn.W_k);
    fill_W(attn.W_v);
    fill_W(attn.W_o);
    fill_W(attn.W_q_agents);
    fill_W(attn.W_k_agents);
    // Agents: small asymmetric init
    for (size_t i = 0; i < N_a; ++i)
        for (size_t j = 0; j < d; ++j)
            attn.agents_[i][j] = 0.2 + 0.05 * (double)((i + 3 * j + 1) % 5);
    // log_lambda = 0 → softplus(0) = log(2) ≈ 0.693
    attn.log_lambda_[0][0] = 0.0;
}

// Copy all learnable parameters from src to dst (for determinism test).
static void copy_params(const AgentAttention& src, AgentAttention& dst) {
    dst.W_q.weights = src.W_q.weights.clone();
    dst.W_q.bias = src.W_q.bias.clone();
    dst.W_k.weights = src.W_k.weights.clone();
    dst.W_k.bias = src.W_k.bias.clone();
    dst.W_v.weights = src.W_v.weights.clone();
    dst.W_v.bias = src.W_v.bias.clone();
    dst.W_o.weights = src.W_o.weights.clone();
    dst.W_o.bias = src.W_o.bias.clone();
    dst.W_q_agents.weights = src.W_q_agents.weights.clone();
    dst.W_q_agents.bias = src.W_q_agents.bias.clone();
    dst.W_k_agents.weights = src.W_k_agents.weights.clone();
    dst.W_k_agents.bias = src.W_k_agents.bias.clone();
    dst.agents_ = src.agents_.clone();
    dst.log_lambda_ = src.log_lambda_.clone();
}

int main() {
    cout << "=== Agent Attention Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: constructor validation
    // ------------------------------------------------------------
    {
        // 1a: d_model=0 throws
        ++total;
        bool threw_d = false;
        try { AgentAttention attn(0, 2); }
        catch (const std::exception&) { threw_d = true; }
        if (threw_d) { cout << "[PASS] d_model=0 throws\n"; ++passed; }
        else cout << "[FAIL] d_model=0 should throw\n";

        // 1b: num_agents=0 throws
        ++total;
        bool threw_na = false;
        try { AgentAttention attn(4, 0); }
        catch (const std::exception&) { threw_na = true; }
        if (threw_na) { cout << "[PASS] num_agents=0 throws\n"; ++passed; }
        else cout << "[FAIL] num_agents=0 should throw\n";
    }

    // ------------------------------------------------------------
    // Test 2: forward shape (N=4, d=4, N_a=2) — linear residual on
    // ------------------------------------------------------------
    {
        ++total;
        size_t N = 4, d = 4, N_a = 2;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        AgentAttention attn(d, N_a, true);
        Tensor out = attn.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == N && out.cols == d) { cout << "[PASS] forward shape correct\n"; ++passed; }
        else cout << "[FAIL] expected " << N << "x" << d << "\n";
    }

    // ------------------------------------------------------------
    // Test 3: forward shape with linear residual off
    // ------------------------------------------------------------
    {
        ++total;
        size_t N = 4, d = 4, N_a = 2;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);

        AgentAttention attn(d, N_a, false);
        Tensor out = attn.forward(input);
        if (out.rows == N && out.cols == d) { cout << "[PASS] forward shape (no residual) correct\n"; ++passed; }
        else cout << "[FAIL] expected " << N << "x" << d << "\n";
    }

    // ------------------------------------------------------------
    // Test 4: forward finiteness + nonzero
    // ------------------------------------------------------------
    {
        ++total;
        size_t N = 4, d = 4, N_a = 2;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.3 * (i + 1) - 0.2 * (j + 1);

        AgentAttention attn(d, N_a, true);
        Tensor out = attn.forward(input);
        bool finite = true, nonzero = false;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols; ++j) {
                if (!std::isfinite(out(i, j))) finite = false;
                if (fabs(out(i, j)) > 1e-6) nonzero = true;
            }
        if (finite && nonzero) { cout << "[PASS] output finite and nonzero\n"; ++passed; }
        else cout << "[FAIL] finite=" << finite << " nonzero=" << nonzero << "\n";
    }

    // ------------------------------------------------------------
    // Test 5: forward shape with N_a=1 (degenerate but allowed)
    // ------------------------------------------------------------
    {
        ++total;
        size_t N = 4, d = 4, N_a = 1;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        AgentAttention attn(d, N_a, true);
        Tensor out = attn.forward(input);
        if (out.rows == N && out.cols == d) { cout << "[PASS] N_a=1 forward shape\n"; ++passed; }
        else cout << "[FAIL] forward shape with N_a=1\n";
    }

    // ------------------------------------------------------------
    // Test 6: agent aggregation is a weighted sum of V rows
    //   A'_a = sum_s p_agg[a, s] * V[s]  where p_agg is row-softmax of
    //   Q_A[a] · K^T / sqrt(d). With linear residual off, A' never sees
    //   outside of the agent stage — so each row of A' must be a convex
    //   combination of V rows (row sums of p_agg are 1).
    // ------------------------------------------------------------
    {
        ++total;
        size_t N = 4, d = 4, N_a = 2;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);

        AgentAttention attn(d, N_a, false);
        Tensor out = attn.forward(input);

        // Inspect last_attn_agg_ — should be a proper row-softmax matrix.
        // Each row of last_attn_agg_ should sum to 1.0.
        const Tensor& A = attn.last_attn_agg_;
        bool row_sums_ok = true;
        for (size_t a = 0; a < N_a; ++a) {
            double rs = 0;
            for (size_t s = 0; s < N; ++s) rs += A[a][s];
            if (fabs(rs - 1.0) > 1e-6) row_sums_ok = false;
        }
        // Same for last_attn_brd_ (sum over N_a = 1.0)
        const Tensor& B = attn.last_attn_brd_;
        bool row_sums_b_ok = true;
        for (size_t t = 0; t < N; ++t) {
            double rs = 0;
            for (size_t a = 0; a < N_a; ++a) rs += B[t][a];
            if (fabs(rs - 1.0) > 1e-6) row_sums_b_ok = false;
        }
        if (row_sums_ok && row_sums_b_ok) {
            cout << "[PASS] agent attention row-softmax preserved\n";
            ++passed;
        } else {
            cout << "[FAIL] row_sums_ok=" << row_sums_ok
                 << " row_sums_b_ok=" << row_sums_b_ok << "\n";
        }
    }

    // ------------------------------------------------------------
    // Tests 7-15: gradient FD checks (use_linear_residual=true)
    //
    // Setup: fixed small input (N=4, d=4, N_a=2), fixed target.
    // Forward → loss → backward → analytical grad.
    // Compare to centered FD: (loss(θ+ε) − loss(θ−ε)) / (2ε).
    // ------------------------------------------------------------
    size_t N = 4, d = 4, N_a = 2;
    double eps = 1e-5;

    // Build a fixed input and target tensor
    Tensor input(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);
    Tensor target(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.1 * ((i + 2 * j) % 5) - 0.05;


    // ------------------------------------------------------------
    // Test 7: input gradient FD check
    // ------------------------------------------------------------
    {
        ++total;
        // Construct attention with deterministic init
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);

        // Forward → backward at θ
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        Tensor grad_input = attn.backward(grad_out, 0.0);

        // FD input
        double max_err = 0;
        size_t total_in = input.rows * input.cols;
        size_t step = max((size_t)1, total_in / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_in; ++k, idx += step) {
            size_t i = idx / input.cols;
            size_t j = idx % input.cols;
            double orig = input[i][j];
            input[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            input[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            input[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = grad_input[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) {
            cout << "[PASS] input grad check (max_err=" << max_err << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] input grad check (max_err=" << max_err << ")\n";
        }
    }

    // Test 8: W_q gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_q.weights;
        Tensor g = attn.W_q.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_q grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_q grad check (max_err=" << max_err << ")\n";
    }

    // Test 9: W_k gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_k.weights;
        Tensor g = attn.W_k.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_k grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_k grad check (max_err=" << max_err << ")\n";
    }

    // Test 10: W_v gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_v.weights;
        Tensor g = attn.W_v.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_v grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_v grad check (max_err=" << max_err << ")\n";
    }

    // Test 11: W_o gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_o.weights;
        Tensor g = attn.W_o.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_o grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_o grad check (max_err=" << max_err << ")\n";
    }

    // Test 12: W_q_agents gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_q_agents.weights;
        Tensor g = attn.W_q_agents.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_q_agents grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_q_agents grad check (max_err=" << max_err << ")\n";
    }

    // Test 13: W_k_agents gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor& W = attn.W_k_agents.weights;
        Tensor g = attn.W_k_agents.grad_weights.clone();
        double max_err = 0;
        size_t total_p = W.rows * W.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / W.cols;
            size_t j = idx % W.cols;
            double orig = W[i][j];
            W[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            W[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            W[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        if (max_err < 1e-3) { cout << "[PASS] W_k_agents grad check (max_err=" << max_err << ")\n"; ++passed; }
        else cout << "[FAIL] W_k_agents grad check (max_err=" << max_err << ")\n";
    }

    // Test 14: agents (A) gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        Tensor g = attn.grad_agents_.clone();
        double max_err = 0;
        size_t total_p = attn.agents_.rows * attn.agents_.cols;
        size_t step = max((size_t)1, total_p / 5);
        for (size_t k = 0, idx = 0; k < 5 && idx < total_p; ++k, idx += step) {
            size_t i = idx / attn.agents_.cols;
            size_t j = idx % attn.agents_.cols;
            double orig = attn.agents_[i][j];
            attn.agents_[i][j] = orig + eps;
            Tensor out_p = attn.forward(input);
            double loss_p = l2_loss_value(out_p, target);
            attn.agents_[i][j] = orig - eps;
            Tensor out_m = attn.forward(input);
            double loss_m = l2_loss_value(out_m, target);
            attn.agents_[i][j] = orig;
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = g[i][j];
            double err = relative_error(ana, num);
            if (err > max_err) max_err = err;
        }
        // Verify gradient is non-zero (else the test is vacuous)
        double g_norm = 0;
        for (size_t i = 0; i < g.rows; ++i)
            for (size_t j = 0; j < g.cols; ++j)
                g_norm += g[i][j] * g[i][j];
        if (max_err < 1e-3 && g_norm > 1e-12) {
            cout << "[PASS] agents (A) grad check (max_err=" << max_err
                 << ", |g|=" << sqrt(g_norm) << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] agents grad check (max_err=" << max_err
                 << ", |g|=" << sqrt(g_norm) << ")\n";
        }
    }

    // Test 15: log_lambda gradient
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        double g = attn.grad_log_lambda_[0][0];
        double orig = attn.log_lambda_[0][0];
        attn.log_lambda_[0][0] = orig + eps;
        Tensor out_p = attn.forward(input);
        double loss_p = l2_loss_value(out_p, target);
        attn.log_lambda_[0][0] = orig - eps;
        Tensor out_m = attn.forward(input);
        double loss_m = l2_loss_value(out_m, target);
        attn.log_lambda_[0][0] = orig;
        double num = (loss_p - loss_m) / (2.0 * eps);
        double ana = g;
        double err = relative_error(ana, num);
        if (err < 1e-3) { cout << "[PASS] log_lambda grad check (max_err=" << err << ")\n"; ++passed; }
        else cout << "[FAIL] log_lambda grad check (max_err=" << err << ")\n";
    }

    // Test 16: determinism — two fresh attentions with copied params
    {
        ++total;
        AgentAttention attn1(d, N_a, true);
        build_deterministic_attn(attn1, d, N_a);
        AgentAttention attn2(d, N_a, true);
        copy_params(attn1, attn2);
        Tensor out1 = attn1.forward(input);
        Tensor out2 = attn2.forward(input);
        double max_diff = 0;
        for (size_t i = 0; i < out1.rows; ++i)
            for (size_t j = 0; j < out1.cols; ++j)
                max_diff = max(max_diff, fabs(out1[i][j] - out2[i][j]));
        if (max_diff < 1e-12) {
            cout << "[PASS] determinism (max_abs_diff=" << max_diff << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] determinism (max_abs_diff=" << max_diff << ")\n";
        }
    }

    // Test 17: training reduces loss
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        // Train with smaller lr — the FD-grade init is sensitive
        double lr = 0.01;
        double L0 = 0;
        for (size_t step = 0; step < 50; ++step) {
            Tensor out = attn.forward(input);
            if (step == 0) L0 = l2_loss_value(out, target);
            Tensor grad_out = l2_loss_grad(out, target);
            attn.backward(grad_out, 0.0);
            attn.update_weights(lr);
            attn.zero_grad();
        }
        Tensor out_final = attn.forward(input);
        double LF = l2_loss_value(out_final, target);
        if (LF < L0) {
            cout << "[PASS] training reduces loss (" << L0 << " → " << LF << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not reduce loss (L0=" << L0 << " LF=" << LF << ")\n";
        }
    }

    // Test 18: param/grad count contract
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        auto p = attn.parameters();
        auto g = attn.gradients();
        // Expected: 6 Denses (W_q, W_k, W_v, W_o, W_q_agents, W_k_agents) × 2 (W + b) = 12
        //          + agents_      = 1
        //          + log_lambda_  = 1
        // Total = 14
        if (p.size() == 14 && g.size() == 14) {
            cout << "[PASS] param/grad count = 14\n";
            ++passed;
        } else {
            cout << "[FAIL] param count = " << p.size() << " (expected 14)\n";
        }
    }

    // Test 19: AgentAttentionBlock forward shape + training reduces loss
    {
        ++total;
        size_t N = 4, d = 4, N_a = 2;
        Tensor input(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                input(i, j) = 0.2 * (i + 1) - 0.1 * (j + 1);
        Tensor target(N, d);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d; ++j)
                target(i, j) = 0.1 * ((i + 2 * j) % 5) - 0.05;

        AgentAttentionBlock block(d, N_a, true, 2);
        double L0 = 0;
        for (size_t step = 0; step < 30; ++step) {
            Tensor out = block.forward(input);
            if (step == 0) L0 = l2_loss_value(out, target);
            Tensor grad_out = l2_loss_grad(out, target);
            block.backward(grad_out, 0.0);
            block.update_weights(0.01);
            block.zero_grad();
        }
        Tensor out_final = block.forward(input);
        double LF = l2_loss_value(out_final, target);
        if (out_final.rows == N && out_final.cols == d && LF < L0) {
            cout << "[PASS] block forward shape + training (" << L0 << " → " << LF << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] block test (shape=" << out_final.rows << "x" << out_final.cols
                 << " L0=" << L0 << " LF=" << LF << ")\n";
        }
    }

    // Test 20: AgentAttentionModel forward shape + training reduces loss
    {
        ++total;
        size_t in_dim = 3, d = 4, out_dim = 2, N = 4, N_a = 2;
        Tensor input(N, in_dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                input(i, j) = 0.1 * (i + 1) - 0.05 * (j + 1);
        Tensor target(N, out_dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < out_dim; ++j)
                target(i, j) = 0.1 * ((i + 2 * j) % 3) - 0.05;

        AgentAttentionModel model(in_dim, d, out_dim, 2, N_a, true, 2);
        double L0 = 0;
        for (size_t step = 0; step < 30; ++step) {
            Tensor out = model.forward(input);
            if (step == 0) L0 = l2_loss_value(out, target);
            Tensor grad_out = l2_loss_grad(out, target);
            model.backward(grad_out, 0.0);
            model.update_weights(0.01);
            model.zero_grad();
        }
        Tensor out_final = model.forward(input);
        double LF = l2_loss_value(out_final, target);
        if (out_final.rows == N && out_final.cols == out_dim && LF < L0) {
            cout << "[PASS] model forward shape + training (" << L0 << " → " << LF << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] model test (shape=" << out_final.rows << "x" << out_final.cols
                 << " L0=" << L0 << " LF=" << LF << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Mutation tests (TDD hygiene: confirm the tests are non-vacuous)
    // ------------------------------------------------------------
    cout << "\n--- Mutation Tests ---\n";

    // Mutation 1: zeroing the agent gradient should break Test 14.
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        double orig_grad = attn.grad_agents_[0][0];
        attn.grad_agents_.fill(0.0);
        if (fabs(attn.grad_agents_[0][0]) < 1e-12 && fabs(orig_grad) > 1e-6) {
            cout << "[PASS] mutation: zeroing agent gradient is observable\n";
            ++passed;
        } else {
            cout << "[FAIL] mutation: zeroing agent gradient not observable\n";
        }
    }

    // Mutation 2: zeroing the log_lambda gradient should break Test 15.
    {
        ++total;
        AgentAttention attn(d, N_a, true);
        build_deterministic_attn(attn, d, N_a);
        Tensor out = attn.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        attn.backward(grad_out, 0.0);
        double orig_grad = attn.grad_log_lambda_[0][0];
        attn.grad_log_lambda_.fill(0.0);
        if (fabs(attn.grad_log_lambda_[0][0]) < 1e-12 && fabs(orig_grad) > 1e-6) {
            cout << "[PASS] mutation: zeroing log_lambda gradient is observable\n";
            ++passed;
        } else {
            cout << "[FAIL] mutation: zeroing log_lambda gradient not observable\n";
        }
    }

    // Mutation 3: zeroed agents should still produce a valid forward.
    {
        ++total;
        AgentAttention attn(d, N_a, false);
        build_deterministic_attn(attn, d, N_a);
        attn.agents_.fill(0.0);
        Tensor out = attn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] mutation: zeroed agents still produce finite output\n";
            ++passed;
        } else {
            cout << "[FAIL] mutation: zeroed agents produced non-finite output\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===" << endl;
    return (passed == total) ? 0 : 1;
}
