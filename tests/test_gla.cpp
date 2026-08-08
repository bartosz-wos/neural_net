// ==========================================================================
// tests/test_gla.cpp
//
// Focused tests for GatedLinearAttention (Yang et al. 2023, "Gated Linear
// Attention Transformers with Hardware-Efficient Training").
// Tests:
//   1. Constructor validation
//   2. Forward shape (T, d_model) -> (T, d_model)
//   3. Forward output is finite
//   4. State cache shape (n_heads, head_dim * head_dim)
//   5. State accumulation (norm > 0 after first step)
//   6. Gate output in (0, 1)
//   7. Input gradient check (analytical vs centered FD)
//   8. W_Q gradient check
//   9. W_K gradient check
//  10. W_V gradient check
//  11. W_O gradient check
//  12. W_gate gradient check
//  13. Training reduces loss (SGD on MSE loss)
//  14. Determinism (bit-identical with copied params)
//  15. Multi-head (n_heads=3, head_dim=2) forward shape
// ==========================================================================

#include "nn/nn.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

static int passed = 0;
static int failed = 0;

#define EXPECT(cond) do { \
    if (cond) { ++passed; } else { ++failed; std::cerr << "FAIL: " << #cond << " at line " << __LINE__ << " in " << __FILE__ << std::endl; } \
} while (0)

#define EXPECT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        ++failed; std::cerr << "FAIL: EXPECT_NEAR(" #a "=" << _a << ", " #b "=" << _b << ", tol=" << (tol) << ") at line " << __LINE__ << std::endl; \
    } else { ++passed; } \
} while (0)

static double compute_loss(GatedLinearAttention& gla, const Tensor& x, const Tensor& target) {
    Tensor y = gla.forward(x);
    double loss = 0.0;
    size_t N = y.rows * y.cols;
    for (size_t i = 0; i < N; ++i) {
        double diff = y.data[i] - target.data[i];
        loss += diff * diff;
    }
    return loss / (2.0 * N);
}

static Tensor numerical_input_grad(GatedLinearAttention& gla, const Tensor& x, const Tensor& target, double eps) {
    size_t T = x.rows, D = x.cols;
    Tensor grad(T, D);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            Tensor x_plus = x.clone();
            Tensor x_minus = x.clone();
            x_plus.data[t * D + d] += eps;
            x_minus.data[t * D + d] -= eps;
            double L_plus = compute_loss(gla, x_plus, target);
            double L_minus = compute_loss(gla, x_minus, target);
            grad.data[t * D + d] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor numerical_param_grad(GatedLinearAttention& gla, const Tensor& x, const Tensor& target,
                                    Dense& d, double eps) {
    Tensor grad(d.weights.rows, d.weights.cols);
    for (size_t i = 0; i < d.weights.rows; ++i) {
        for (size_t j = 0; j < d.weights.cols; ++j) {
            double orig = d.weights[i][j];
            d.weights[i][j] = orig + eps;
            double L_plus = compute_loss(gla, x, target);
            d.weights[i][j] = orig - eps;
            double L_minus = compute_loss(gla, x, target);
            d.weights[i][j] = orig;
            grad[i][j] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        m = std::max(m, std::abs(a.data[i] - b.data[i]));
    }
    return m;
}

static double rel_err(const Tensor& a, const Tensor& b) {
    double max_abs = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        max_abs = std::max(max_abs, std::max(std::abs(a.data[i]), std::abs(b.data[i])));
    }
    if (max_abs < 1e-12) return max_abs_diff(a, b);
    return max_abs_diff(a, b) / max_abs;
}

int main() {
    std::srand(42);

    // ---- Test 1: Constructor validation ----
    {
        bool threw = false;
        try { GatedLinearAttention g(0, 2); } catch (std::invalid_argument&) { threw = true; }
        EXPECT(threw);
    }
    {
        bool threw = false;
        try { GatedLinearAttention g(4, 0); } catch (std::invalid_argument&) { threw = true; }
        EXPECT(threw);
    }
    {
        bool threw = false;
        try { GatedLinearAttention g(5, 2); } catch (std::invalid_argument&) { threw = true; }  // 5 % 2 != 0
        EXPECT(threw);
    }
    {
        bool threw = false;
        try { GatedLinearAttention g(6, 2, 2); } catch (std::invalid_argument&) { threw = true; }
        EXPECT(threw);  // explicit head_dim must equal d_model / n_heads
    }

    // ---- Setup for forward/gradient tests ----
    const size_t T = 3;
    const size_t d_model = 4;
    const size_t n_heads = 2;
    const size_t head_dim = 2;
    const size_t d_inner = d_model;  // default d_inner = d_model
    GatedLinearAttention gla(d_model, n_heads, head_dim);

    // Random input and target
    Tensor x(T, d_model);
    Tensor target(T, d_model);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = dist(rng);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng);

    // ---- Test 2: Forward shape ----
    Tensor y = gla.forward(x);
    EXPECT(y.rows == T);
    EXPECT(y.cols == d_model);

    // ---- Test 3: Forward output is finite ----
    bool all_finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { all_finite = false; break; }
    }
    EXPECT(all_finite);

    // ---- Test 4: State cache shape ----
    Tensor S_T = gla.last_state();
    EXPECT(S_T.rows == n_heads);
    EXPECT(S_T.cols == head_dim * head_dim);

    // ---- Test 5: State accumulation ----
    double norm = 0.0;
    for (size_t i = 0; i < S_T.data.size(); ++i) norm += S_T.data[i] * S_T.data[i];
    EXPECT(norm > 0.0);

    // ---- Test 6: Gate output in (0, 1) ----
    Tensor gates = gla.last_gates();
    EXPECT(gates.rows == T);
    EXPECT(gates.cols == n_heads);
    bool all_in_unit = true;
    for (size_t i = 0; i < T * n_heads; ++i) {
        double g = gates.data[i];
        if (g <= 0.0 || g >= 1.0) { all_in_unit = false; break; }
    }
    EXPECT(all_in_unit);

    // ---- Test 7: Input gradient (analytical vs FD) ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        Tensor ana_grad = gla.backward(grad_out, 0.0);
        Tensor fd = numerical_input_grad(gla, x, target, 1e-5);
        double r = rel_err(ana_grad, fd);
        std::cerr << "  [info] input-grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 8: W_Q gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_Q_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_Q_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_Q grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 9: W_K gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_K_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_K_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_K grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 10: W_V gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_V_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_V_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_V grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 11: W_O gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_O_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_O_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_O grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 12: W_gate gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_gate_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_gate_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_gate grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 13: Training reduces loss ----
    {
        GatedLinearAttention train_gla(d_model, n_heads, head_dim);
        // Initialize with a fixed pattern so the gradient step has signal.
        // W_Q, W_K, W_V, W_O all have shape (d_model, d_model) = 16 elements.
        for (size_t i = 0; i < train_gla.W_Q_.weights.data.size(); ++i) {
            train_gla.W_Q_.weights.data[i] = 0.1 * (1.0 - 0.02 * static_cast<double>(i % 7));
            train_gla.W_K_.weights.data[i] = 0.05 + 0.01 * static_cast<double>(i % 5);
            train_gla.W_V_.weights.data[i] = 0.2 - 0.015 * static_cast<double>(i % 9);
            train_gla.W_O_.weights.data[i] = 0.3 + 0.02 * static_cast<double>(i % 11);
        }
        // W_gate has shape (n_heads, d_model) = (2, 4) = 8 elements.
        for (size_t i = 0; i < train_gla.W_gate_.weights.data.size(); ++i) {
            train_gla.W_gate_.weights.data[i] = 0.5 - 0.03 * static_cast<double>(i % 13);
        }

        double lr = 0.05;
        double initial_loss = compute_loss(train_gla, x, target);
        for (size_t step = 0; step < 50; ++step) {
            Tensor y_tr = train_gla.forward(x);
            Tensor grad_out(T, d_model);
            size_t N = T * d_model;
            for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y_tr.data[i] - target.data[i]) / N;
            train_gla.zero_grad();
            train_gla.backward(grad_out, 0.0);
            train_gla.update_weights(lr);
        }
        double final_loss = compute_loss(train_gla, x, target);
        std::cerr << "  [info] training: initial=" << initial_loss << " final=" << final_loss << std::endl;
        EXPECT(final_loss < initial_loss);
    }

    // ---- Test 14: Determinism ----
    {
        GatedLinearAttention a(d_model, n_heads, head_dim);
        GatedLinearAttention b(d_model, n_heads, head_dim);
        // Copy params from a to b. Note: W_Q/K/V/O have 16 elements each,
        // W_gate has only 8 elements (n_heads * d_model).
        for (size_t i = 0; i < a.W_Q_.weights.data.size(); ++i) {
            b.W_Q_.weights.data[i] = a.W_Q_.weights.data[i];
            b.W_K_.weights.data[i] = a.W_K_.weights.data[i];
            b.W_V_.weights.data[i] = a.W_V_.weights.data[i];
            b.W_O_.weights.data[i] = a.W_O_.weights.data[i];
        }
        for (size_t i = 0; i < a.W_gate_.weights.data.size(); ++i) {
            b.W_gate_.weights.data[i] = a.W_gate_.weights.data[i];
        }
        Tensor ya = a.forward(x);
        Tensor yb = b.forward(x);
        double r = max_abs_diff(ya, yb);
        std::cerr << "  [info] determinism max_abs_diff = " << r << std::endl;
        EXPECT(r < 1e-12);
    }

    // ---- Test 15: Multi-head (n_heads=3, head_dim=2) forward shape ----
    {
        GatedLinearAttention mh(6, 3, 2);
        Tensor x_mh(4, 6);
        for (size_t i = 0; i < x_mh.data.size(); ++i) x_mh.data[i] = dist(rng);
        Tensor y_mh = mh.forward(x_mh);
        EXPECT(y_mh.rows == 4);
        EXPECT(y_mh.cols == 6);
    }

    // ---- Test 16: zero_grad clears all gradient buffers ----
    {
        GatedLinearAttention zg(d_model, n_heads, head_dim);
        Tensor y0 = zg.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        zg.backward(grad_out, 0.0);
        // Check some gradients are non-zero
        bool any_nonzero = false;
        for (size_t i = 0; i < zg.W_Q_.grad_weights.data.size(); ++i) {
            if (zg.W_Q_.grad_weights.data[i] != 0.0) { any_nonzero = true; break; }
        }
        EXPECT(any_nonzero);
        zg.zero_grad();
        // After zero_grad, all param gradients must be zero
        bool all_zero = true;
        for (size_t i = 0; i < zg.W_Q_.grad_weights.data.size(); ++i) {
            if (zg.W_Q_.grad_weights.data[i] != 0.0) { all_zero = false; break; }
        }
        for (size_t i = 0; i < zg.W_K_.grad_weights.data.size(); ++i) {
            if (zg.W_K_.grad_weights.data[i] != 0.0) { all_zero = false; break; }
        }
        for (size_t i = 0; i < zg.W_gate_.grad_weights.data.size(); ++i) {
            if (zg.W_gate_.grad_weights.data[i] != 0.0) { all_zero = false; break; }
        }
        EXPECT(all_zero);
    }

    // ---- Test 17: name() and get_gradients() point to a real Dense grad buffer ----
    {
        GatedLinearAttention nu(d_model, n_heads, head_dim);
        EXPECT(nu.name() == "GatedLinearAttention");
        Tensor gw = nu.get_weights();
        EXPECT(gw.rows == d_inner);
        Tensor gg = nu.get_gradients();
        EXPECT(gg.rows == gw.rows);
        EXPECT(gg.cols == gw.cols);
    }

    // ---- Test 18: parameters() returns expected count ----
    {
        GatedLinearAttention pc(d_model, n_heads, head_dim);
        auto params = pc.parameters();
        // 5 Dense layers * 2 params (weights, bias) = 10 tensors
        EXPECT(params.size() == 10);
    }

    // ---- Test 19: GLA updates weights (training actually moves parameters) ----
    {
        GatedLinearAttention tu(d_model, n_heads, head_dim);
        // Snapshot W_O weights
        Tensor W_O_before = tu.W_O_.weights.clone();
        Tensor y0 = tu.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        tu.zero_grad();
        tu.backward(grad_out, 0.0);
        tu.update_weights(0.1);
        // Some weights must have moved
        double moved = max_abs_diff(W_O_before, tu.W_O_.weights);
        std::cerr << "  [info] update_weights W_O delta = " << moved << std::endl;
        EXPECT(moved > 0.0);
    }

    std::cerr << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}