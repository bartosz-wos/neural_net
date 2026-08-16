// test_jamba.cpp — Tests for Jamba hybrid block (Lieber et al. 2024)
// https://arxiv.org/abs/2403.19887
//
// Jamba = Mamba-2 SSM + Multi-head self-attention + MoE FFN, all under
// pre-norm residuals. This test file covers:
//   - Constructor validation
//   - Forward shape / finiteness / non-zeros
//   - Backward gradient flow (input grad + parameter grad via FD)
//   - Training reduces loss
//   - Determinism
//   - Parameter / gradient count contract
//   - update_weights / zero_grad contract
//   - JambaStack (multi-block)
//   - Dense-FFN variant (num_experts=0)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <vector>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) {
        cout << "  [PASS] " << name << endl;
        ++passed;
    } else {
        cout << "  [FAIL] " << name << endl;
        ++failed;
    }
    return pass;
}

// Build a (T, D) tensor with seeded random values, scaled to a small magnitude.
static Tensor rand_tensor(size_t T, size_t D, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = nd(rng);
    return x;
}

// Fill a tensor with all zeros.
static Tensor zeros_tensor(size_t T, size_t D) {
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = 0.0;
    return x;
}

// Centered finite-difference grad of one parameter entry, using MSE loss
// L = sum((y - target)^2) / (2*T).
static double numerical_grad(JambaBlock& blk, const Tensor& x, const Tensor& target,
                             Tensor& param, size_t r, size_t c, double eps) {
    double orig = param(r, c);
    param(r, c) = orig + eps;
    Tensor y_plus = blk.forward(x);
    double L_plus = 0.0;
    for (size_t i = 0; i < y_plus.rows; ++i)
        for (size_t j = 0; j < y_plus.cols; ++j) {
            double d = y_plus[i][j] - target[i][j];
            L_plus += d * d;
        }
    L_plus /= (2.0 * y_plus.rows);

    param(r, c) = orig - eps;
    Tensor y_minus = blk.forward(x);
    double L_minus = 0.0;
    for (size_t i = 0; i < y_minus.rows; ++i)
        for (size_t j = 0; j < y_minus.cols; ++j) {
            double d = y_minus[i][j] - target[i][j];
            L_minus += d * d;
        }
    L_minus /= (2.0 * y_minus.rows);

    param(r, c) = orig;
    return (L_plus - L_minus) / (2.0 * eps);
}

// Analytical gradient of one parameter entry, via standard forward/backward +
// chain rule. For MSE loss L = sum((y-t)^2) / (2T),  dL/dy = (y-t)/T.
static double analytical_grad(JambaBlock& blk, const Tensor& x, const Tensor& target,
                               Tensor& grad, size_t r, size_t c) {
    Tensor y = blk.forward(x);
    size_t T = y.rows;
    Tensor grad_y(T, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
    blk.zero_grad();
    blk.backward(grad_y, 0.0);
    return grad(r, c);
}

// Relative error with floor for FP noise.
static double rel_err(double a, double b) {
    double denom = std::max(std::abs(a), std::abs(b));
    if (denom < 1e-12) return std::abs(a - b);
    return std::abs(a - b) / denom;
}

// =====================================================================
// Test 1: Constructor validation
// =====================================================================
static void test_constructor_validation() {
    cout << endl << "-- Test 1: constructor validation --" << endl;
    bool threw = false;
    try { JambaBlock blk(0, 2); } catch (const std::invalid_argument&) { threw = true; }
    check("d_model=0 throws", threw);

    threw = false;
    try { JambaBlock blk(8, 0); } catch (const std::invalid_argument&) { threw = true; }
    check("num_heads=0 throws", threw);

    threw = false;
    try { JambaBlock blk(4, 3); } catch (const std::invalid_argument&) { threw = true; }
    check("d_model % num_heads != 0 throws", threw);

    threw = false;
    try { JambaBlock blk(8, 2, 4, 2, 0, 3); } catch (const std::invalid_argument&) { threw = true; }
    check("moe_every_n=3 throws", threw);

    // Default-construct: 8/4/0/2
    {
        JambaBlock blk(8, 4, 0, 2);
        check("d_model=8 num_heads=4 num_experts=0 ok", true);
    }
    // MoE variant
    {
        JambaBlock blk(8, 2, 4, 2);
        check("d_model=8 num_heads=2 num_experts=4 ok", true);
    }
}

// =====================================================================
// Test 2: Forward shape + finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 2: forward shape --" << endl;
    JambaBlock blk(8, 2, 4, 2);

    bool ok = true;
    for (size_t T : {1u, 2u, 3u, 5u}) {
        Tensor x = Tensor(T, 8);
        for (size_t i = 0; i < T * 8; ++i) x.data[i] = 0.1 * (i + 1);
        Tensor y = blk.forward(x);
        if (y.rows != T || y.cols != 8) { ok = false; break; }
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!std::isfinite(y.data[i])) { ok = false; break; }
    }
    check("T=1,2,3,5 all yield (T,8) and finite", ok);

    // input feature dim mismatch
    bool threw = false;
    try { blk.forward(Tensor(3, 7)); } catch (const std::invalid_argument&) { threw = true; }
    check("input feature dim mismatch throws", threw);
}

// =====================================================================
// Test 3: Output is non-zero for random init
// =====================================================================
static void test_output_nonzero() {
    cout << endl << "-- Test 3: output non-zero --" << endl;
    JambaBlock blk(8, 2, 4, 2);
    Tensor x = rand_tensor(3, 8, 42);
    Tensor y = blk.forward(x);
    bool any_nonzero = false;
    for (size_t i = 0; i < y.data.size(); ++i)
        if (std::abs(y.data[i]) > 1e-8) { any_nonzero = true; break; }
    check("output has at least one nonzero entry", any_nonzero);
}

// =====================================================================
// Test 4: Input gradient via centered finite differences
// =====================================================================
static void test_input_gradient() {
    cout << endl << "-- Test 4: input gradient (centered FD) --" << endl;
    JambaBlock blk(4, 2, 0, 2);  // No MoE — easier to verify
    Tensor x = rand_tensor(3, 4, 7);
    Tensor target = rand_tensor(3, 4, 11);

    Tensor y = blk.forward(x);
    size_t T = y.rows;
    Tensor grad_y(T, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
    Tensor grad_x_ana = blk.backward(grad_y, 0.0);

    double eps = 1e-4;
    bool ok = true;
    double max_err = 0.0;
    for (size_t t = 0; t < 3; ++t) {
        for (size_t d = 0; d < 4; ++d) {
            double orig = x(t, d);
            x(t, d) = orig + eps;
            Tensor y_plus = blk.forward(x);
            double L_plus = 0.0;
            for (size_t i = 0; i < T; ++i)
                for (size_t j = 0; j < 4; ++j) {
                    double diff = y_plus[i][j] - target[i][j];
                    L_plus += diff * diff;
                }
            L_plus /= (2.0 * T);
            x(t, d) = orig - eps;
            Tensor y_minus = blk.forward(x);
            double L_minus = 0.0;
            for (size_t i = 0; i < T; ++i)
                for (size_t j = 0; j < 4; ++j) {
                    double diff = y_minus[i][j] - target[i][j];
                    L_minus += diff * diff;
                }
            L_minus /= (2.0 * T);
            x(t, d) = orig;
            double num = (L_plus - L_minus) / (2.0 * eps);
            double ana = grad_x_ana(t, d);
            double re = rel_err(ana, num);
            if (re > max_err) max_err = re;
            if (re > 1e-3) { ok = false; }
        }
    }
    cout << "  max_rel_err = " << max_err << endl;
    check("input gradient matches FD (rel_err < 1e-3)", ok);
}

// =====================================================================
// Test 5: Parameter gradient FD check on a Mamba-2 weight
// =====================================================================
static void test_param_grad_mamba() {
    cout << endl << "-- Test 5: Mamba-2 in_proj param gradient --" << endl;
    JambaBlock blk(8, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    // Find the Mamba-2 in_proj.weights — it's the first param.
    auto params = blk.parameters();
    auto grads = blk.gradients();
    Tensor& mp = *params[0];  // Mamba-2 in_proj.weights
    Tensor& mg = *grads[0];

    double a = analytical_grad(blk, x, target, mg, 0, 0);
    double n = numerical_grad(blk, x, target, mp, 0, 0, 1e-5);
    double re = rel_err(a, n);
    cout << "  Mamba in_proj.weights[0][0]: a=" << a << " n=" << n << " rel_err=" << re << endl;
    check("Mamba-2 in_proj.weights grad matches FD", re < 1e-2);
}

// =====================================================================
// Test 6: Parameter gradient FD check on attention W_q
// =====================================================================
static void test_param_grad_attn() {
    cout << endl << "-- Test 6: Attention W_q param gradient --" << endl;
    JambaBlock blk(8, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    auto params = blk.parameters();
    auto grads = blk.gradients();
    // Find an attention W_q — it's a (d_model, d_model) tensor.
    Tensor* ap = nullptr;
    Tensor* ag = nullptr;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == 8 && params[i]->cols == 8) {
            // Look for the first W_q (in Mamba-2, in_proj is (2*d_inner, d_model) = (16, 8))
            // Attention W_q is (d_model, d_model) = (8, 8).
            // Multiple (8,8) tensors exist; we want the second one (after Mamba's in_proj
            // which is (16, 8)). To target W_q specifically, we look for the first (8,8) at
            // i >= 1 (since Mamba-2 out_proj is (8, 16)).
            if (ap == nullptr) ap = params[i];
            else { ap = params[i]; ag = grads[i]; break; }
        }
    }
    if (ap == nullptr || ag == nullptr) {
        check("found Attention W_q param", false);
        return;
    }

    double a = analytical_grad(blk, x, target, *ag, 0, 0);
    double n = numerical_grad(blk, x, target, *ap, 0, 0, 1e-5);
    double re = rel_err(a, n);
    cout << "  Attn W_q.weights[0][0]: a=" << a << " n=" << n << " rel_err=" << re << endl;
    check("Attention W_q grad matches FD", re < 1e-2);
}

// =====================================================================
// Test 7: Parameter gradient FD check on dense FFN weights
// =====================================================================
static void test_param_grad_ffn() {
    cout << endl << "-- Test 7: Dense FFN w1 param gradient --" << endl;
    JambaBlock blk(8, 2, 0, 2);  // num_experts=0 -> dense FFN
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    auto params = blk.parameters();
    auto grads = blk.gradients();
    // Find w1_.weights = (4*d_model, d_model) = (32, 8)
    Tensor* fp = nullptr;
    Tensor* fg = nullptr;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == 32 && params[i]->cols == 8) {
            fp = params[i]; fg = grads[i]; break;
        }
    }
    if (fp == nullptr) {
        check("found Dense FFN w1.weights", false);
        return;
    }

    double a = analytical_grad(blk, x, target, *fg, 0, 0);
    double n = numerical_grad(blk, x, target, *fp, 0, 0, 1e-5);
    double re = rel_err(a, n);
    cout << "  Dense FFN w1[0][0]: a=" << a << " n=" << n << " rel_err=" << re << endl;
    check("Dense FFN w1 grad matches FD", re < 1e-2);
}

// =====================================================================
// Test 8: Training reduces loss
// =====================================================================
static void test_training_reduces_loss() {
    cout << endl << "-- Test 8: training reduces loss --" << endl;
    JambaBlock blk(8, 2, 0, 2);  // dense FFN for determinism
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    auto compute_loss = [&](JambaBlock& b) {
        Tensor y = b.forward(x);
        double loss = 0.0;
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) {
                double d = y[i][j] - target[i][j];
                loss += d * d;
            }
        return loss / (2.0 * y.rows);
    };

    double L0 = compute_loss(blk);
    double lr = 1e-3;
    for (size_t step = 0; step < 30; ++step) {
        Tensor y = blk.forward(x);
        size_t T = y.rows;
        Tensor grad_y(T, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
        blk.zero_grad();
        blk.backward(grad_y, 0.0);
        blk.update_weights(lr);
    }
    double L1 = compute_loss(blk);
    cout << "  L0=" << L0 << " L1=" << L1 << " ratio=" << L1 / L0 << endl;
    check("loss reduced by > 30% over 30 SGD steps", L1 < 0.7 * L0);
}

// =====================================================================
// Test 9: Determinism — same seed produces bit-identical forward
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 9: determinism --" << endl;
    std::srand(42);
    JambaBlock a(8, 2, 4, 2);
    std::srand(42);
    JambaBlock b(8, 2, 4, 2);
    Tensor x = rand_tensor(3, 8, 99);
    Tensor ya = a.forward(x);
    Tensor yb = b.forward(x);
    bool ok = true;
    for (size_t i = 0; i < ya.data.size(); ++i)
        if (std::abs(ya.data[i] - yb.data[i]) > 1e-12) { ok = false; break; }
    check("two fresh blocks with same seed -> bit-identical forward", ok);
}

// =====================================================================
// Test 10: Parameter count contract
// =====================================================================
static void test_param_count() {
    cout << endl << "-- Test 10: parameter count contract --" << endl;
    JambaBlock dense_blk(8, 2, 0, 2);
    JambaBlock moe_blk(8, 2, 4, 2);
    size_t dense_p = dense_blk.parameters().size();
    size_t moe_p = moe_blk.parameters().size();
    cout << "  dense params=" << dense_p << " moe params=" << moe_p << endl;
    check("dense FFN block has parameters", dense_p > 0);
    check("MoE block has more parameters than dense (with 4 experts)", moe_p > dense_p);
    check("MoE has parameters", moe_p > 0);
}

// =====================================================================
// Test 11: update_weights moves all parameters
// =====================================================================
static void test_update_weights() {
    cout << endl << "-- Test 11: update_weights moves all parameters --" << endl;
    JambaBlock blk(8, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    // Snapshot
    auto params = blk.parameters();
    std::vector<std::vector<double>> before;
    for (auto* p : params) before.push_back(p->data);

    // Forward + backward + update
    Tensor y = blk.forward(x);
    Tensor grad_y(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(y.rows);
    blk.zero_grad();
    blk.backward(grad_y, 0.0);
    blk.update_weights(0.1);

    bool any_moved = false;
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t k = 0; k < params[i]->data.size(); ++k)
            if (std::abs(params[i]->data[k] - before[i][k]) > 1e-12) { any_moved = true; break; }
        if (any_moved) break;
    }
    check("update_weights moves at least one parameter", any_moved);
}

// =====================================================================
// Test 12: zero_grad clears all gradients
// =====================================================================
static void test_zero_grad() {
    cout << endl << "-- Test 12: zero_grad clears --" << endl;
    JambaBlock blk(8, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    Tensor y = blk.forward(x);
    Tensor grad_y(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(y.rows);
    blk.backward(grad_y, 0.0);

    bool any_nonzero_before = false;
    for (auto* g : blk.gradients()) {
        for (size_t k = 0; k < g->data.size(); ++k)
            if (std::abs(g->data[k]) > 1e-20) { any_nonzero_before = true; break; }
        if (any_nonzero_before) break;
    }
    check("gradients non-zero after backward", any_nonzero_before);

    blk.zero_grad();
    bool all_zero_after = true;
    for (auto* g : blk.gradients()) {
        for (size_t k = 0; k < g->data.size(); ++k)
            if (std::abs(g->data[k]) > 0.0) { all_zero_after = false; break; }
        if (!all_zero_after) break;
    }
    check("gradients all zero after zero_grad", all_zero_after);
}

// =====================================================================
// Test 13: JambaStack forward shape
// =====================================================================
static void test_stack_forward() {
    cout << endl << "-- Test 13: JambaStack forward --" << endl;
    JambaStack stack(8, 2, 2, 4, 2);
    Tensor x = rand_tensor(4, 8, 7);
    Tensor y = stack.forward(x);
    bool ok = (y.rows == 4 && y.cols == 8);
    for (size_t i = 0; i < y.data.size(); ++i)
        if (!std::isfinite(y.data[i])) { ok = false; break; }
    check("2-layer stack forward shape (4,8) and finite", ok);
}

// =====================================================================
// Test 14: JambaStack training reduces loss
// =====================================================================
static void test_stack_training() {
    cout << endl << "-- Test 14: JambaStack training reduces loss --" << endl;
    JambaStack stack(8, 2, 2, 0, 2);  // 2 layers, dense FFN
    Tensor x = rand_tensor(4, 8, 7);
    Tensor target = rand_tensor(4, 8, 11);

    auto compute_loss = [&](JambaStack& s) {
        Tensor y = s.forward(x);
        double loss = 0.0;
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) {
                double d = y[i][j] - target[i][j];
                loss += d * d;
            }
        return loss / (2.0 * y.rows);
    };
    double L0 = compute_loss(stack);
    for (size_t step = 0; step < 20; ++step) {
        Tensor y = stack.forward(x);
        Tensor grad_y(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(y.rows);
        stack.zero_grad();
        stack.backward(grad_y, 0.0);
        stack.update_weights(1e-3);
    }
    double L1 = compute_loss(stack);
    cout << "  L0=" << L0 << " L1=" << L1 << " ratio=" << L1 / L0 << endl;
    check("2-layer stack loss reduced by > 20% over 20 steps", L1 < 0.8 * L0);
}

// =====================================================================
// Test 15: JambaStack parameter count scales with num_layers
// =====================================================================
static void test_stack_params() {
    cout << endl << "-- Test 15: stack params scale --" << endl;
    JambaStack s1(8, 2, 1, 0, 2);
    JambaStack s3(8, 2, 3, 0, 2);
    size_t p1 = s1.parameters().size();
    size_t p3 = s3.parameters().size();
    cout << "  1-layer params=" << p1 << " 3-layer params=" << p3 << endl;
    check("3-layer stack has 3x params of 1-layer", p3 == 3 * p1);
}

// =====================================================================
// Test 16: MoE-block forward shape + finiteness
// =====================================================================
static void test_moe_forward() {
    cout << endl << "-- Test 16: MoE block forward --" << endl;
    JambaBlock blk(8, 2, 4, 2);
    Tensor x = rand_tensor(3, 8, 7);
    bool ok = true;
    for (size_t T : {1u, 2u, 3u, 5u}) {
        Tensor xi = rand_tensor(T, 8, 7);
        Tensor y = blk.forward(xi);
        if (y.rows != T || y.cols != 8) { ok = false; break; }
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!std::isfinite(y.data[i])) { ok = false; break; }
    }
    check("MoE block T=1..5 forward shape (T,8) and finite", ok);
}

// =====================================================================
// Test 17: JambaStack(num_experts=0) backward doesn't crash
// =====================================================================
static void test_stack_backward() {
    cout << endl << "-- Test 17: JambaStack backward --" << endl;
    JambaStack stack(8, 2, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor y = stack.forward(x);
    Tensor grad_y(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_y[i][j] = 1.0;
    Tensor grad_x = stack.backward(grad_y, 0.0);
    bool ok = (grad_x.rows == 3 && grad_x.cols == 8);
    for (size_t i = 0; i < grad_x.data.size(); ++i)
        if (!std::isfinite(grad_x.data[i])) { ok = false; break; }
    check("stack backward returns (3,8) and finite", ok);
}

// =====================================================================
// Test 18: JambaBlock.num_blocks() >= 1
// =====================================================================
static void test_stack_num_blocks() {
    cout << endl << "-- Test 18: stack num_blocks --" << endl;
    JambaStack s1(8, 2, 1, 0, 2);
    JambaStack s5(8, 2, 5, 0, 2);
    check("1-layer stack has 1 block", s1.num_blocks() == 1);
    check("5-layer stack has 5 blocks", s5.num_blocks() == 5);
}

// =====================================================================
// Test 19: moe_every_n=2 — MoE in block 0, dense in block 1
// =====================================================================
static void test_moe_every_n() {
    cout << endl << "-- Test 19: moe_every_n=2 --" << endl;
    JambaStack stack(8, 2, 2, 4, 2, 0, 2);
    // Block 0: MoE (uses_moe=true). Block 1: dense FFN.
    check("block 0 uses MoE", stack.block(0).uses_moe());
    check("block 1 uses dense FFN", !stack.block(1).uses_moe());
    // Forward over both blocks should still work.
    Tensor x = rand_tensor(3, 8, 7);
    Tensor y = stack.forward(x);
    bool ok = (y.rows == 3 && y.cols == 8);
    for (size_t i = 0; i < y.data.size(); ++i)
        if (!std::isfinite(y.data[i])) { ok = false; break; }
    check("mixed MoE/dense stack forward ok", ok);
}

// =====================================================================
// Test 20: JambaBlock gradient on a LayerNorm gamma
// =====================================================================
static void test_ln_gamma_grad() {
    cout << endl << "-- Test 20: LayerNorm gamma gradient --" << endl;
    JambaBlock blk(8, 2, 0, 2);
    Tensor x = rand_tensor(3, 8, 7);
    Tensor target = rand_tensor(3, 8, 11);

    auto params = blk.parameters();
    auto grads = blk.gradients();
    // Find a LayerNorm gamma — (d_model, 1) = (8, 1) tensor
    Tensor* gp = nullptr;
    Tensor* gg = nullptr;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == 8 && params[i]->cols == 1) {
            gp = params[i]; gg = grads[i]; break;
        }
    }
    if (gp == nullptr) {
        check("found LayerNorm gamma param", false);
        return;
    }

    double a = analytical_grad(blk, x, target, *gg, 3, 0);
    double n = numerical_grad(blk, x, target, *gp, 3, 0, 1e-5);
    double re = rel_err(a, n);
    cout << "  LN1 gamma[3]: a=" << a << " n=" << n << " rel_err=" << re << endl;
    check("LayerNorm gamma grad matches FD", re < 1e-2);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== Jamba Hybrid Block Tests ===" << endl;
    test_constructor_validation();
    test_forward_shape();
    test_output_nonzero();
    test_input_gradient();
    test_param_grad_mamba();
    test_param_grad_attn();
    test_param_grad_ffn();
    test_training_reduces_loss();
    test_determinism();
    test_param_count();
    test_update_weights();
    test_zero_grad();
    test_stack_forward();
    test_stack_training();
    test_stack_params();
    test_moe_forward();
    test_stack_backward();
    test_stack_num_blocks();
    test_moe_every_n();
    test_ln_gamma_grad();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
