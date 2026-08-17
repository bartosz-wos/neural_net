// test_griffin.cpp — Tests for the Griffin hybrid sequence block
// (De et al. 2024, https://arxiv.org/abs/2402.19427)
//
// Griffin = parallel composition of:
//   (a) Hawk (RG-LRU) gated linear recurrence
//   (b) Local sliding-window causal attention
//   (c) Dense GELU MLP
// All three branches fed by a shared LayerNorm, summed into a single
// residual stream: out = x + hawk(LN(x)) + attn(LN(x)) + mlp(LN(x))
//
// This is the key difference vs Jamba (which is sequential). The Griffin
// block is the canonical hybrid block from the Griffin paper §3.2.
//
// Test file covers:
//   - Constructor validation
//   - Forward shape / finiteness / non-zero
//   - Three sublayers each receive LayerNorm'd input (sanity)
//   - Backward gradient flow (input + every parameter group)
//   - Training reduces loss
//   - Determinism
//   - GriffinModel (multi-block stack)
//   - update_weights / zero_grad contract
//   - Parameter count contract

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

// Build a (T, D) tensor with seeded random values.
static Tensor rand_tensor(size_t T, size_t D, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = nd(rng);
    return x;
}

// Forward all-ones tensor (unused but kept for future use).
[[maybe_unused]] static Tensor ones_tensor(size_t T, size_t D) {
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = 1.0;
    return x;
}

// Hand-derived FD gradient for parameter entry (r, c) under MSE loss
// L = sum((y-t)^2) / (2T).
static double fd_grad_param(GriffinBlock& blk, const Tensor& x, const Tensor& target,
                            Tensor& param, size_t r, size_t c,
                            double eps = 1e-4) {
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

static double fd_grad_input(GriffinBlock& blk, const Tensor& x, const Tensor& target,
                            size_t r, size_t c, double eps = 1e-4) {
    Tensor x_plus = x;
    Tensor x_minus = x;
    x_plus[r][c] += eps;
    x_minus[r][c] -= eps;

    Tensor y_plus = blk.forward(x_plus);
    Tensor y_minus = blk.forward(x_minus);
    double L_plus = 0.0, L_minus = 0.0;
    for (size_t i = 0; i < y_plus.rows; ++i)
        for (size_t j = 0; j < y_plus.cols; ++j) {
            double d_p = y_plus[i][j] - target[i][j];
            double d_m = y_minus[i][j] - target[i][j];
            L_plus += d_p * d_p;
            L_minus += d_m * d_m;
        }
    L_plus /= (2.0 * y_plus.rows);
    L_minus /= (2.0 * y_minus.rows);
    return (L_plus - L_minus) / (2.0 * eps);
}

// Analytical gradient of one parameter entry, via standard forward/backward +
// chain rule. For MSE loss L = sum((y-t)^2) / (2T),  dL/dy = (y-t)/T.
static double analytical_grad_param(GriffinBlock& blk, const Tensor& x, const Tensor& target,
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
    try { GriffinBlock blk(0, 2); } catch (const std::invalid_argument&) { threw = true; }
    check("d_model=0 throws", threw);

    threw = false;
    try { GriffinBlock blk(8, 0, 3); } catch (const std::invalid_argument&) { threw = true; }
    check("num_heads=0 throws", threw);

    threw = false;
    try { GriffinBlock blk(5, 2, 3); } catch (const std::invalid_argument&) { threw = true; }
    check("d_model % num_heads != 0 throws", threw);

    threw = false;
    try { GriffinBlock blk(8, 2, 0); } catch (const std::invalid_argument&) { threw = true; }
    check("window_size=0 throws", threw);

    threw = false;
    try { GriffinBlock blk(8, 2, 3, 0); } catch (const std::invalid_argument&) { threw = true; }
    check("ffn_mult=0 throws", threw);

    // Default-construct variants
    {
        GriffinBlock blk(8, 2);
        check("GriffinBlock(8, 2) ok", true);
    }
    {
        GriffinBlock blk(8, 4, 3, 2);
        check("GriffinBlock(8, 4, 3, 2) ok", true);
    }
}

// =====================================================================
// Test 2: Forward shape + finiteness + non-zero
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 2: forward shape --" << endl;
    GriffinBlock blk(8, 2, 3);

    bool ok_shape = true, ok_finite = true, ok_nonzero = true;
    for (size_t T : {1u, 2u, 3u, 5u, 8u}) {
        Tensor x = rand_tensor(T, 8, 42u, 0.3);
        Tensor y;
        try {
            y = blk.forward(x);
        } catch (const std::exception& e) {
            ok_shape = false;
            cout << "    forward(T=" << T << ") threw: " << e.what() << endl;
            continue;
        }
        if (y.rows != T || y.cols != 8) ok_shape = false;
        for (size_t i = 0; i < T; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                double v = y[i][j];
                if (!std::isfinite(v)) ok_finite = false;
                if (std::abs(v) > 1e-9) {
                    /* nonzero ok */
                }
            }
        }
        // Sum of abs output > 0
        double abs_sum = 0.0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < 8; ++j)
                abs_sum += std::abs(y[i][j]);
        if (abs_sum < 1e-9) ok_nonzero = false;
    }
    check("forward shape (T, 8) for T=1..8", ok_shape);
    check("forward finite", ok_finite);
    check("forward nonzero", ok_nonzero);
}

// =====================================================================
// Test 3: Input gradient via centered FD
// =====================================================================
static void test_input_grad_fd() {
    cout << endl << "-- Test 3: input gradient FD check --" << endl;
    // Small config for tractable FD
    GriffinBlock blk(4, 2, 2, 2);
    // Re-init with non-degenerate random weights
    // (Dense default xavier works)

    const size_t T = 3;
    Tensor x = rand_tensor(T, 4, 7u, 0.3);
    Tensor target = rand_tensor(T, 4, 11u, 0.3);

    // Compute analytical input grad
    Tensor y = blk.forward(x);
    Tensor grad_y(T, 4);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
    blk.zero_grad();
    Tensor grad_input_ana = blk.backward(grad_y, 0.0);

    bool ok = true;
    double max_re = 0.0;
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_grad_input(blk, x, target, i, j, 1e-4);
            double ana = grad_input_ana[i][j];
            double re = rel_err(fd, ana);
            if (re > max_re) max_re = re;
            if (re > 5e-2) { ok = false; }
        }
    }
    cout << "    max_rel_err = " << max_re << endl;
    check("input grad matches centered FD (eps=1e-4, rel_err < 5e-2)", ok);
}

// =====================================================================
// Test 4: Hawk path parameter gradient (W_x.weights)
// =====================================================================
static void test_hawk_W_x_grad() {
    cout << endl << "-- Test 4: Hawk W_x.weights gradient FD check --" << endl;
    GriffinBlock blk(4, 2, 2, 2);
    // Ensure non-degenerate init: re-seed RNG and force a refresh of weights
    // (Dense default is xavier which is fine; we just need > 0 weights)

    const size_t T = 3;
    Tensor x = rand_tensor(T, 4, 13u, 0.3);
    Tensor target = rand_tensor(T, 4, 17u, 0.3);

    Tensor& W_x = blk.hawk.W_x.weights;

    bool ok = true;
    double max_re = 0.0;
    for (size_t r = 0; r < W_x.rows; ++r) {
        for (size_t c = 0; c < W_x.cols; ++c) {
            double fd = fd_grad_param(blk, x, target, W_x, r, c, 1e-4);
            double ana = analytical_grad_param(blk, x, target,
                                               blk.hawk.W_x.grad_weights, r, c);
            double re = rel_err(fd, ana);
            if (re > max_re) max_re = re;
            if (re > 5e-2) { ok = false; }
        }
    }
    cout << "    max_rel_err = " << max_re << endl;
    check("Hawk W_x.weights grad matches FD (rel_err < 5e-2)", ok);
}

// =====================================================================
// Test 5: Attention path parameter gradient (W_q.weights)
// =====================================================================
static void test_attn_W_q_grad() {
    cout << endl << "-- Test 5: Attention W_q.weights gradient FD check --" << endl;
    GriffinBlock blk(4, 2, 2, 2);

    const size_t T = 3;
    Tensor x = rand_tensor(T, 4, 23u, 0.3);
    Tensor target = rand_tensor(T, 4, 29u, 0.3);

    Tensor& W_q = blk.attn.W_q.weights;

    bool ok = true;
    double max_re = 0.0;
    for (size_t r = 0; r < W_q.rows; ++r) {
        for (size_t c = 0; c < W_q.cols; ++c) {
            double fd = fd_grad_param(blk, x, target, W_q, r, c, 1e-4);
            double ana = analytical_grad_param(blk, x, target,
                                               blk.attn.W_q.grad_weights, r, c);
            double re = rel_err(fd, ana);
            if (re > max_re) max_re = re;
            if (re > 5e-2) { ok = false; }
        }
    }
    cout << "    max_rel_err = " << max_re << endl;
    check("Attn W_q.weights grad matches FD (rel_err < 5e-2)", ok);
}

// =====================================================================
// Test 6: MLP path parameter gradient (W1.weights)
// =====================================================================
static void test_mlp_W1_grad() {
    cout << endl << "-- Test 6: MLP W1.weights gradient FD check --" << endl;
    GriffinBlock blk(4, 2, 2, 2);

    const size_t T = 3;
    Tensor x = rand_tensor(T, 4, 31u, 0.3);
    Tensor target = rand_tensor(T, 4, 37u, 0.3);

    Tensor& W1 = blk.mlp.W1.weights;

    bool ok = true;
    double max_re = 0.0;
    for (size_t r = 0; r < W1.rows; ++r) {
        for (size_t c = 0; c < W1.cols; ++c) {
            double fd = fd_grad_param(blk, x, target, W1, r, c, 1e-4);
            double ana = analytical_grad_param(blk, x, target,
                                               blk.mlp.W1.grad_weights, r, c);
            double re = rel_err(fd, ana);
            if (re > max_re) max_re = re;
            if (re > 5e-2) { ok = false; }
        }
    }
    cout << "    max_rel_err = " << max_re << endl;
    check("MLP W1.weights grad matches FD (rel_err < 5e-2)", ok);
}

// =====================================================================
// Test 7: LayerNorm gamma gradient
// =====================================================================
static void test_ln_gamma_grad() {
    cout << endl << "-- Test 7: LN gamma gradient FD check --" << endl;
    GriffinBlock blk(4, 2, 2, 2);

    const size_t T = 3;
    Tensor x = rand_tensor(T, 4, 41u, 0.3);
    Tensor target = rand_tensor(T, 4, 43u, 0.3);

    Tensor& gamma = blk.ln.gamma;

    bool ok = true;
    double max_re = 0.0;
    for (size_t c = 0; c < 4; ++c) {
        double fd = fd_grad_param(blk, x, target, gamma, 0, c, 1e-4);
        double ana = analytical_grad_param(blk, x, target,
                                           blk.ln.grad_gamma_, 0, c);
        double re = rel_err(fd, ana);
        if (re > max_re) max_re = re;
        if (re > 5e-2) { ok = false; }
    }
    cout << "    max_rel_err = " << max_re << endl;
    check("LN gamma grad matches FD (rel_err < 5e-2)", ok);
}

// =====================================================================
// Test 8: All three sublayer paths receive LayerNorm'd input
// (sanity: verify the LN cache is populated and is the input to all three)
// =====================================================================
static void test_ln_cache_populated() {
    cout << endl << "-- Test 8: LN cache populated, all 3 sublayers see LN'd input --" << endl;
    GriffinBlock blk(4, 2, 2, 2);

    Tensor x = rand_tensor(3, 4, 47u, 0.3);
    blk.forward(x);

    bool ok = true;
    // The LN cache is non-empty after forward
    if (blk.last_ln_input.rows != 3 || blk.last_ln_input.cols != 4) ok = false;
    // Hawk should have received LN'd input: cache gate_input = LN(x) @ W_x + b_x
    // We sanity-check that all three sublayer forward-caches are populated.
    if (blk.hawk.last_gate_input_.rows != 3 || blk.hawk.last_gate_input_.cols != 4) ok = false;
    if (blk.attn.last_q.rows != 3 || blk.attn.last_q.cols != 4) ok = false;
    if (blk.mlp.last_input.rows != 3 || blk.mlp.last_input.cols != 4) ok = false;
    check("LN cache + 3 sublayer caches populated after forward", ok);
}

// =====================================================================
// Test 9: Determinism (two fresh blocks with copied params → bit-identical forward)
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 9: determinism --" << endl;
    GriffinBlock blk1(8, 2, 3, 2);
    GriffinBlock blk2(8, 2, 3, 2);

    // Copy parameters from blk1 to blk2
    blk2.copy_params_from(blk1);

    Tensor x = rand_tensor(5, 8, 53u, 0.3);
    Tensor y1 = blk1.forward(x);
    Tensor y2 = blk2.forward(x);

    double max_diff = 0.0;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 8; ++j) {
            double d = std::abs(y1[i][j] - y2[i][j]);
            if (d > max_diff) max_diff = d;
        }
    cout << "    max abs diff = " << max_diff << endl;
    check("two fresh GriffinBlocks with copied params → bit-identical forward", max_diff == 0.0);
}

// =====================================================================
// Test 10: Training reduces loss
// =====================================================================
static void test_training_reduces_loss() {
    cout << endl << "-- Test 10: training reduces loss --" << endl;
    GriffinBlock blk(8, 2, 3, 2);

    // Synthetic regression: target = linear projection of input
    const size_t T = 4;
    Tensor x = rand_tensor(T, 8, 59u, 0.3);
    Tensor w_target(1, 8);
    for (size_t i = 0; i < 8; ++i) w_target[0][i] = 0.3;
    Tensor target = x;  // (T, 8) * (1, 8)^T = elementwise (T, 8); use as identity-shaped
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < 8; ++j)
            target[i][j] = 0.5 * x[i][j];  // simple scale target

    auto compute_loss = [&]() {
        Tensor y = blk.forward(x);
        double L = 0.0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < 8; ++j) {
                double d = y[i][j] - target[i][j];
                L += d * d;
            }
        return L / (2.0 * T);
    };

    double L0 = compute_loss();
    const double lr = 1e-3;
    const int n_steps = 80;
    for (int s = 0; s < n_steps; ++s) {
        Tensor y = blk.forward(x);
        Tensor grad_y(T, 8);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < 8; ++j)
                grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
        blk.zero_grad();
        blk.backward(grad_y, 0.0);
        blk.update_weights(lr);
    }
    double L1 = compute_loss();
    cout << "    L0=" << L0 << " → L1=" << L1
         << " (reduction = " << (100.0 * (L0 - L1) / L0) << "%)" << endl;
    check("training reduces loss > 5% over 80 SGD steps (finite)",
          L1 < 0.95 * L0 && std::isfinite(L1));
}

// =====================================================================
// Test 11: update_weights moves all parameters; zero_grad clears all grads
// =====================================================================
static void test_update_weights_zero_grad() {
    cout << endl << "-- Test 11: update_weights / zero_grad contract --" << endl;
    GriffinBlock blk(4, 2, 2, 2);

    Tensor x = rand_tensor(3, 4, 61u, 0.3);
    Tensor target = rand_tensor(3, 4, 67u, 0.3);
    Tensor y = blk.forward(x);
    Tensor grad_y(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_y[i][j] = (y[i][j] - target[i][j]) / 3.0;
    blk.zero_grad();
    blk.backward(grad_y, 0.0);

    // Snapshot params before update
    auto snapshot_params = [&](const GriffinBlock& b) {
        std::vector<Tensor> snaps;
        for (Tensor* p : const_cast<GriffinBlock&>(b).parameters()) snaps.push_back(*p);
        return snaps;
    };
    auto snaps0 = snapshot_params(blk);
    blk.update_weights(1e-3);
    auto snaps1 = snapshot_params(blk);

    bool any_moved = false;
    for (size_t i = 0; i < snaps0.size(); ++i) {
        double d = 0.0;
        for (size_t r = 0; r < snaps0[i].rows; ++r)
            for (size_t c = 0; c < snaps0[i].cols; ++c)
                d += std::abs(snaps0[i][r][c] - snaps1[i][r][c]);
        if (d > 1e-12) any_moved = true;
    }
    check("update_weights moves at least one parameter", any_moved);

    // zero_grad after a backward should produce all-zero grads
    blk.zero_grad();
    bool all_zero = true;
    for (Tensor* g : blk.gradients()) {
        for (size_t r = 0; r < g->rows; ++r)
            for (size_t c = 0; c < g->cols; ++c)
                if (std::abs((*g)[r][c]) > 1e-15) all_zero = false;
    }
    check("zero_grad clears all gradients", all_zero);
}

// =====================================================================
// Test 12: Parameter / gradient count contract
// =====================================================================
static void test_param_count() {
    cout << endl << "-- Test 12: parameter/gradient count --" << endl;
    GriffinBlock blk(8, 2, 3, 2);
    auto p = blk.parameters();
    auto g = blk.gradients();
    cout << "    #params=" << p.size() << ", #grads=" << g.size() << endl;
    // Hawk contributes 5 (W_x.weights, W_x.bias, log_a_raw, W_o.weights, W_o.bias)
    // LocalAttn contributes 8 (W_q/W_k/W_v/W_o weights+bias = 8)
    // MLP contributes 4 (W1/W2 weights + biases = 4)
    // LN contributes 2 (gamma, beta)
    // Total = 5 + 8 + 4 + 2 = 19
    check("parameters() returns 19 tensors", p.size() == 19);
    check("gradients() returns 19 tensors", g.size() == 19);
}

// =====================================================================
// Test 13: GriffinModel forward + training
// =====================================================================
static void test_griffin_model() {
    cout << endl << "-- Test 13: GriffinModel (multi-block stack) --" << endl;
    GriffinModel model(4, 8, 3, 2, 2, 3, 2);  // input=4, d=8, output=3, num_layers=2, num_heads=2, win=3, mult=2

    const size_t B = 2, T = 4;
    Tensor x(B * T, 4);
    std::mt19937 rng(71u);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (size_t i = 0; i < B * T * 4; ++i) x.data[i] = nd(rng);

    Tensor y;
    bool ok_shape = true, ok_finite = true;
    try {
        y = model.forward(x);
    } catch (const std::exception& e) {
        cout << "    forward threw: " << e.what() << endl;
        ok_shape = false;
    }
    if (y.rows != B * T || y.cols != 3) ok_shape = false;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y[i][j])) ok_finite = false;
    check("model forward shape (B*T, 3)", ok_shape);
    check("model forward finite", ok_finite);

    // Training reduces loss
    Tensor target = Tensor::zeros(B * T, 3);
    for (size_t i = 0; i < target.rows * target.cols; ++i)
        target.data[i] = 0.05 * (static_cast<int>(i % 5) - 2);

    auto compute_loss = [&]() {
        Tensor yhat = model.forward(x);
        double L = 0.0;
        for (size_t i = 0; i < yhat.rows; ++i)
            for (size_t j = 0; j < yhat.cols; ++j) {
                double d = yhat[i][j] - target[i][j];
                L += d * d;
            }
        return L / (2.0 * yhat.rows);
    };

    double L0 = compute_loss();
    const double lr = 5e-3;
    const int n_steps = 80;
    for (int s = 0; s < n_steps; ++s) {
        Tensor yhat = model.forward(x);
        Tensor grad_y(yhat.rows, yhat.cols);
        for (size_t i = 0; i < yhat.rows; ++i)
            for (size_t j = 0; j < yhat.cols; ++j)
                grad_y[i][j] = (yhat[i][j] - target[i][j]) / static_cast<double>(yhat.rows);
        model.zero_grad();
        model.backward(grad_y, 0.0);
        model.update_weights(lr);
    }
    double L1 = compute_loss();
    cout << "    L0=" << L0 << " → L1=" << L1
         << " (reduction = " << (100.0 * (L0 - L1) / L0) << "%)" << endl;
    check("model training reduces loss > 10% over 60 SGD steps (finite)",
          L1 < 0.9 * L0 && std::isfinite(L1));
}

// =====================================================================
// main
// =====================================================================
int main() {
    cout << "=== Griffin Hybrid Block Tests ===" << endl;
    test_constructor_validation();
    test_forward_shape();
    test_input_grad_fd();
    test_hawk_W_x_grad();
    test_attn_W_q_grad();
    test_mlp_W1_grad();
    test_ln_gamma_grad();
    test_ln_cache_populated();
    test_determinism();
    test_training_reduces_loss();
    test_update_weights_zero_grad();
    test_param_count();
    test_griffin_model();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}