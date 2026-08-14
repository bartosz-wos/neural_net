// ============================================================================
// HGRN (Hierarchically Gated Recurrent Neural Network) tests — Mehrdad 2023
//   "Hierarchically Gated Recurrent Neural Network"
//   https://arxiv.org/abs/2307.02226
// ============================================================================
//
// Test coverage: HGRN-1 (no output gate) and HGRN-2 (with output gate) for
// HGRNCell, plus a stacked HGRNModel. Verifies:
//   - constructor validation, forward shape, finiteness
//   - hand-derived forward reference (single channel, T=2)
//   - input gradient via centered finite differences (rel_err < 1e-5)
//   - all 6 parameter gradient checks for HGRN-1 (W_f.weights/bias,
//     W_i.weights/bias, W_z.weights/bias) via centered FD (rel_err < 1e-3)
//   - all 8 parameter gradient checks for HGRN-2 (+ W_o.weights/bias)
//   - hand-derived forward for HGRN-2 (verifies output gate path)
//   - determinism (two fresh instances with copied params produce bit-exact
//     forward)
//   - parameters()/gradients()/zero_grad() shape contract
//   - update_weights() actually moves parameters
//   - HGRNModel forward shape + finiteness
//   - HGRNModel training reduces loss over 50 SGD steps
//
// Layer-under-test: include/nn/layers/recurrent/hgrn.{h,cpp}
//
// Forward math (per channel c, time step t):
//     f_pre_t[c] = (W_f x_t + b_f)[c]
//     i_pre_t[c] = (W_i x_t + b_i)[c]
//     z_pre_t[c] = (W_z x_t + b_z)[c]
//     f_t[c]     = sigmoid(f_pre_t[c])   ∈ (0, 1)
//     i_t[c]     = sigmoid(i_pre_t[c])   ∈ (0, 1)
//     z_t[c]     = tanh(z_pre_t[c])      ∈ (-1, 1)
//     c_t[c]     = f_t[c] * c_{t-1}[c] + i_t[c] * z_t[c]
//     h_t[c]     = c_t[c]                                    (HGRN-1)
//     h_t[c]     = sigmoid(W_o x_t + b_o)[c] * c_t[c]       (HGRN-2)
//
// IMPORTANT: For the parameter-gradient tests, the loss function used for
// FD is L(y) = sum(grad_loss .* y), where `grad_loss` is the same vector
// passed to cell.backward() as the dL/dy gradient. This keeps the FD
// formula `(L(y+eps) - L(y-eps)) / (2*eps) = grad_loss .* dy/d_param`
// directly comparable to the analytical `cell.backward(grad_loss, 0).* grad_W`.
// If the loss were e.g. MSE the analytical `grad_output` would need to be
// `2*(y - target)/N` — we sidestep that by using the dot-product form.
// ============================================================================

#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/layers/recurrent/hgrn.h"
#include "nn/utils/gradient_check.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace {

int g_tests_passed = 0;
int g_tests_failed = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  [FAIL] %s (line %d)\n", msg, __LINE__); \
        ++g_tests_failed; \
    } else { \
        ++g_tests_passed; \
    } \
} while (0)

#define EXPECT_NEAR(a, b, tol, msg) do { \
    double aa = (a), bb = (b); \
    double err = std::fabs(aa - bb); \
    double scale = std::max({std::fabs(aa), std::fabs(bb), 1e-12}); \
    if (err > tol * scale || std::isnan(err)) { \
        std::fprintf(stderr, "  [FAIL] %s: %.6e vs %.6e (rel_err %.3e, line %d)\n", \
                     msg, aa, bb, err / scale, __LINE__); \
        ++g_tests_failed; \
    } else { \
        ++g_tests_passed; \
    } \
} while (0)

#define TOL_STRICT 1e-9
#define TOL_LOOSE  1e-4

// Build a deterministic (T, d) input tensor with small values.
Tensor make_input(size_t T, size_t d, double scale = 0.1) {
    Tensor x(T, d);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d; ++j) {
            x[i][j] = scale * static_cast<double>(i + 1 + j);
        }
    }
    return x;
}

// Initialize a 3-gate HGRNCell with small deterministic weights.
void init_cell_weights(HGRNCell& cell, double scale = 0.2) {
    // Dense stores weights as (out_features, in_features)
    for (size_t i = 0; i < cell.W_f.weights.rows; ++i) {
        for (size_t j = 0; j < cell.W_f.weights.cols; ++j) {
            double v = scale * static_cast<double>((i + 1) * (j + 1)) / static_cast<double>(cell.W_f.weights.cols);
            cell.W_f.weights[i][j] = v;
            cell.W_i.weights[i][j] = v * 0.5;
            cell.W_z.weights[i][j] = v * 0.3;
            if (cell.use_output_gate()) {
                cell.W_o.weights[i][j] = v * 0.7;
            }
        }
        cell.W_f.bias[0][i] = scale * 0.1 * static_cast<double>(i + 1);
        cell.W_i.bias[0][i] = scale * 0.1 * static_cast<double>(i + 1);
        cell.W_z.bias[0][i] = scale * 0.05 * static_cast<double>(i + 1);
        if (cell.use_output_gate()) {
            cell.W_o.bias[0][i] = scale * 0.05 * static_cast<double>(i + 1);
        }
    }
}

// Compute finite-difference gradient for a single parameter tensor by perturbing
// it entry-by-entry. Returns the FD gradient tensor of the same shape as `param_ptr`.
Tensor compute_fd_grad_param(
    Tensor* param_ptr,
    std::function<Tensor()> forward_fn,
    std::function<double(const Tensor&)> loss_fn,
    double eps = 1e-5)
{
    Tensor fd(param_ptr->rows, param_ptr->cols);
    fd.fill(0.0);
    Tensor saved = param_ptr->clone();
    for (size_t i = 0; i < param_ptr->rows; ++i) {
        for (size_t j = 0; j < param_ptr->cols; ++j) {
            double orig = (*param_ptr)[i][j];
            (*param_ptr)[i][j] = orig + eps;
            double L_plus = loss_fn(forward_fn());
            (*param_ptr)[i][j] = orig - eps;
            double L_minus = loss_fn(forward_fn());
            (*param_ptr)[i][j] = orig;
            fd[i][j] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    // Restore
    for (size_t i = 0; i < param_ptr->rows; ++i) {
        for (size_t j = 0; j < param_ptr->cols; ++j) {
            (*param_ptr)[i][j] = saved[i][j];
        }
    }
    return fd;
}

// Test 1: constructor + forward shape
void test_constructor_and_forward_shape() {
    HGRNCell cell(3, 4);
    EXPECT(cell.input_dim() == 3, "input_dim");
    EXPECT(cell.hidden_dim() == 4, "hidden_dim");
    EXPECT(!cell.use_output_gate(), "default no output gate");
    EXPECT(cell.name() == "HGRNCell", "name");

    Tensor x = make_input(5, 3, 0.1);
    Tensor y = cell.forward(x);
    EXPECT(y.rows == 5, "output rows == T");
    EXPECT(y.cols == 4, "output cols == hidden_dim");
    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "forward output finite");
}

// Test 2: forward with output gate (HGRN-2)
void test_forward_with_output_gate() {
    HGRNCell cell(3, 4, /*use_output_gate=*/true);
    EXPECT(cell.use_output_gate(), "output gate enabled");

    Tensor x = make_input(4, 3, 0.05);
    Tensor y = cell.forward(x);
    EXPECT(y.rows == 4, "rows");
    EXPECT(y.cols == 4, "cols");
    bool finite = true;
    double max_abs = 0.0;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
        max_abs = std::max(max_abs, std::fabs(y.data[i]));
    }
    EXPECT(finite, "finite");
    EXPECT(max_abs > 0.0, "non-zero output (weights init non-zero)");
}

// Test 3: hand-derived forward reference (single channel, T=2, HGRN-1).
void test_hand_derived_forward_hgrn1() {
    HGRNCell cell(1, 1, /*use_output_gate=*/false);
    const double w_f = 0.5, b_f = 0.1;
    const double w_i = 0.6, b_i = -0.2;
    const double w_z = 0.7, b_z = 0.3;
    cell.W_f.weights[0][0] = w_f;
    cell.W_f.bias[0][0]    = b_f;
    cell.W_i.weights[0][0] = w_i;
    cell.W_i.bias[0][0]    = b_i;
    cell.W_z.weights[0][0] = w_z;
    cell.W_z.bias[0][0]    = b_z;

    Tensor x(2, 1);
    x[0][0] = 0.4;
    x[1][0] = 0.6;

    Tensor y = cell.forward(x);

    auto sigmoid = [](double z) {
        if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    };

    const double f_pre_0 = w_f * x[0][0] + b_f;
    const double i_pre_0 = w_i * x[0][0] + b_i;
    const double z_pre_0 = w_z * x[0][0] + b_z;
    const double f_0 = sigmoid(f_pre_0);
    const double i_0 = sigmoid(i_pre_0);
    const double z_0 = std::tanh(z_pre_0);
    const double c_0 = f_0 * 0.0 + i_0 * z_0;
    const double h_0 = c_0;

    const double f_pre_1 = w_f * x[1][0] + b_f;
    const double i_pre_1 = w_i * x[1][0] + b_i;
    const double z_pre_1 = w_z * x[1][0] + b_z;
    const double f_1 = sigmoid(f_pre_1);
    const double i_1 = sigmoid(i_pre_1);
    const double z_1 = std::tanh(z_pre_1);
    const double c_1 = f_1 * c_0 + i_1 * z_1;
    const double h_1 = c_1;

    EXPECT_NEAR(y[0][0], h_0, TOL_STRICT, "hand-derived h[0]");
    EXPECT_NEAR(y[1][0], h_1, TOL_STRICT, "hand-derived h[1]");
}

// Test 4: hand-derived forward reference for HGRN-2 (output gate path).
void test_hand_derived_forward_hgrn2() {
    HGRNCell cell(1, 1, /*use_output_gate=*/true);
    cell.W_f.weights[0][0] = 0.5; cell.W_f.bias[0][0] = 0.1;
    cell.W_i.weights[0][0] = 0.6; cell.W_i.bias[0][0] = -0.2;
    cell.W_z.weights[0][0] = 0.7; cell.W_z.bias[0][0] = 0.3;
    cell.W_o.weights[0][0] = 0.4; cell.W_o.bias[0][0] = -0.1;

    Tensor x(2, 1);
    x[0][0] = 0.4;
    x[1][0] = 0.6;

    Tensor y = cell.forward(x);

    auto sigmoid = [](double z) {
        if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    };

    const double f_pre_0 = 0.5 * 0.4 + 0.1;
    const double i_pre_0 = 0.6 * 0.4 - 0.2;
    const double z_pre_0 = 0.7 * 0.4 + 0.3;
    const double o_pre_0 = 0.4 * 0.4 - 0.1;
    const double f_0 = sigmoid(f_pre_0);
    const double i_0 = sigmoid(i_pre_0);
    const double z_0 = std::tanh(z_pre_0);
    const double o_0 = sigmoid(o_pre_0);
    const double c_0 = f_0 * 0.0 + i_0 * z_0;
    const double h_0 = o_0 * c_0;

    const double f_pre_1 = 0.5 * 0.6 + 0.1;
    const double i_pre_1 = 0.6 * 0.6 - 0.2;
    const double z_pre_1 = 0.7 * 0.6 + 0.3;
    const double o_pre_1 = 0.4 * 0.6 - 0.1;
    const double f_1 = sigmoid(f_pre_1);
    const double i_1 = sigmoid(i_pre_1);
    const double z_1 = std::tanh(z_pre_1);
    const double o_1 = sigmoid(o_pre_1);
    const double c_1 = f_1 * c_0 + i_1 * z_1;
    const double h_1 = o_1 * c_1;

    EXPECT_NEAR(y[0][0], h_0, TOL_STRICT, "hand-derived HGRN-2 h[0]");
    EXPECT_NEAR(y[1][0], h_1, TOL_STRICT, "hand-derived HGRN-2 h[1]");
}

// Test 5: determinism — two fresh networks with identical params produce bit-exact forward.
void test_determinism() {
    HGRNCell a(3, 4);
    HGRNCell b(3, 4);
    init_cell_weights(a, 0.2);
    init_cell_weights(b, 0.2);
    Tensor x = make_input(4, 3, 0.3);
    Tensor ya = a.forward(x);
    Tensor yb = b.forward(x);
    double max_diff = 0.0;
    for (size_t i = 0; i < ya.data.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ya.data[i] - yb.data[i]));
    }
    EXPECT(max_diff < 1e-12, "determinism HGRN-1");

    HGRNCell c(3, 4, true);
    HGRNCell d(3, 4, true);
    init_cell_weights(c, 0.2);
    init_cell_weights(d, 0.2);
    Tensor yc = c.forward(x);
    Tensor yd = d.forward(x);
    double max_diff2 = 0.0;
    for (size_t i = 0; i < yc.data.size(); ++i) {
        max_diff2 = std::max(max_diff2, std::fabs(yc.data[i] - yd.data[i]));
    }
    EXPECT(max_diff2 < 1e-12, "determinism HGRN-2");
}

// Test 6: input gradient via centered FD.
void test_input_gradient_hgrn1() {
    HGRNCell cell(2, 3);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(4, 2, 0.3);
    Tensor y = cell.forward(x);
    Tensor grad_loss(4, 3);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) {
        grad_loss.data[i] = 0.5 * (i + 1);
    }
    cell.zero_grad();
    Tensor grad_x_analytical = cell.backward(grad_loss, 0.0);

    double eps = 1e-5;
    double max_rel_err = 0.0;
    for (size_t t = 0; t < 4; ++t) {
        for (size_t c = 0; c < 2; ++c) {
            double x_orig = x[t][c];
            x[t][c] = x_orig + eps;
            Tensor y_plus = cell.forward(x);
            x[t][c] = x_orig - eps;
            Tensor y_minus = cell.forward(x);
            x[t][c] = x_orig;

            double fd = 0.0;
            for (size_t i = 0; i < grad_loss.data.size(); ++i) {
                fd += grad_loss.data[i] * (y_plus.data[i] - y_minus.data[i]) / (2.0 * eps);
            }
            double ana = grad_x_analytical[t][c];
            double scale = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
            double rel = std::fabs(ana - fd) / scale;
            max_rel_err = std::max(max_rel_err, rel);
        }
    }
    std::printf("    HGRN-1 input grad: max rel_err = %.3e\n", max_rel_err);
    EXPECT(max_rel_err < 1e-5, "input grad FD");
}

// Test 7: parameter gradients HGRN-1 (all 6) via centered FD.
//
// Loss for FD: L = sum(grad_loss .* y_pred)  →  dL/dy_pred = grad_loss
// (matches the gradient we pass to cell.backward()).
void test_parameter_gradients_hgrn1() {
    HGRNCell cell(2, 2);
    init_cell_weights(cell, 0.15);
    Tensor x = make_input(3, 2, 0.25);
    Tensor y = cell.forward(x);
    Tensor grad_loss(3, 2);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) {
        grad_loss.data[i] = 0.5 * (i + 1);
    }
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad_loss.data.size(); ++i) {
            s += grad_loss.data[i] * y_pred.data[i];
        }
        return s;
    };
    auto forward_fn = [&]() { return cell.forward(x); };

    struct ParamPair { Tensor* param; Tensor* grad; const char* name; };
    std::vector<ParamPair> params = {
        {&cell.W_f.weights, &cell.W_f.grad_weights, "W_f.weights"},
        {&cell.W_f.bias,    &cell.W_f.grad_bias,    "W_f.bias"},
        {&cell.W_i.weights, &cell.W_i.grad_weights, "W_i.weights"},
        {&cell.W_i.bias,    &cell.W_i.grad_bias,    "W_i.bias"},
        {&cell.W_z.weights, &cell.W_z.grad_weights, "W_z.weights"},
        {&cell.W_z.bias,    &cell.W_z.grad_bias,    "W_z.bias"},
    };

    for (auto& pp : params) {
        Tensor fd = compute_fd_grad_param(pp.param, forward_fn, loss_fn, 1e-5);
        double max_rel = 0.0;
        for (size_t i = 0; i < pp.grad->rows; ++i) {
            for (size_t j = 0; j < pp.grad->cols; ++j) {
                double ana = (*pp.grad)[i][j];
                double fn = fd[i][j];
                double scale = std::max({std::fabs(ana), std::fabs(fn), 1e-12});
                max_rel = std::max(max_rel, std::fabs(ana - fn) / scale);
            }
        }
        std::printf("    HGRN-1 %s: max rel_err = %.3e\n", pp.name, max_rel);
        EXPECT(max_rel < 1e-3, pp.name);
    }
}

// Test 8: parameter gradients HGRN-2 (all 8) via centered FD.
void test_parameter_gradients_hgrn2() {
    HGRNCell cell(2, 2, /*use_output_gate=*/true);
    init_cell_weights(cell, 0.15);
    Tensor x = make_input(3, 2, 0.25);
    Tensor y = cell.forward(x);
    Tensor grad_loss(3, 2);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) {
        grad_loss.data[i] = 0.5 * (i + 1);
    }
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad_loss.data.size(); ++i) {
            s += grad_loss.data[i] * y_pred.data[i];
        }
        return s;
    };
    auto forward_fn = [&]() { return cell.forward(x); };

    struct ParamPair { Tensor* param; Tensor* grad; const char* name; };
    std::vector<ParamPair> params = {
        {&cell.W_f.weights, &cell.W_f.grad_weights, "W_f.weights"},
        {&cell.W_f.bias,    &cell.W_f.grad_bias,    "W_f.bias"},
        {&cell.W_i.weights, &cell.W_i.grad_weights, "W_i.weights"},
        {&cell.W_i.bias,    &cell.W_i.grad_bias,    "W_i.bias"},
        {&cell.W_z.weights, &cell.W_z.grad_weights, "W_z.weights"},
        {&cell.W_z.bias,    &cell.W_z.grad_bias,    "W_z.bias"},
        {&cell.W_o.weights, &cell.W_o.grad_weights, "W_o.weights"},
        {&cell.W_o.bias,    &cell.W_o.grad_bias,    "W_o.bias"},
    };

    for (auto& pp : params) {
        Tensor fd = compute_fd_grad_param(pp.param, forward_fn, loss_fn, 1e-5);
        double max_rel = 0.0;
        for (size_t i = 0; i < pp.grad->rows; ++i) {
            for (size_t j = 0; j < pp.grad->cols; ++j) {
                double ana = (*pp.grad)[i][j];
                double fn = fd[i][j];
                double scale = std::max({std::fabs(ana), std::fabs(fn), 1e-12});
                max_rel = std::max(max_rel, std::fabs(ana - fn) / scale);
            }
        }
        std::printf("    HGRN-2 %s: max rel_err = %.3e\n", pp.name, max_rel);
        EXPECT(max_rel < 1e-3, pp.name);
    }
}

// Test 9: parameters/gradients/zero_grad contract.
void test_parameters_gradients_contract() {
    {
        HGRNCell cell(2, 3);
        auto p = cell.parameters();
        EXPECT(p.size() == 6, "HGRN-1 params count (3 Dense × 2 = 6)");
        auto g = cell.gradients();
        EXPECT(g.size() == 6, "HGRN-1 grads count");
    }
    {
        HGRNCell cell(2, 3, /*use_output_gate=*/true);
        auto p = cell.parameters();
        EXPECT(p.size() == 8, "HGRN-2 params count (4 Dense × 2 = 8)");
        auto g = cell.gradients();
        EXPECT(g.size() == 8, "HGRN-2 grads count");
    }
}

// Test 10: zero_grad clears all gradients.
void test_zero_grad() {
    HGRNCell cell(2, 2);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(3, 2);
    Tensor y = cell.forward(x);
    Tensor grad_loss(3, 2);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) grad_loss.data[i] = 0.1;
    cell.backward(grad_loss, 0.0);

    bool nonzero = false;
    for (size_t i = 0; i < cell.W_f.grad_weights.data.size() && !nonzero; ++i) {
        if (std::fabs(cell.W_f.grad_weights.data[i]) > 1e-12) nonzero = true;
    }
    EXPECT(nonzero, "grad non-zero before zero_grad");

    cell.zero_grad();
    bool zero = true;
    for (size_t i = 0; i < cell.W_f.grad_weights.data.size(); ++i) {
        if (std::fabs(cell.W_f.grad_weights.data[i]) > 1e-12) { zero = false; break; }
    }
    for (size_t i = 0; i < cell.W_i.grad_weights.data.size(); ++i) {
        if (std::fabs(cell.W_i.grad_weights.data[i]) > 1e-12) { zero = false; break; }
    }
    for (size_t i = 0; i < cell.W_z.grad_weights.data.size(); ++i) {
        if (std::fabs(cell.W_z.grad_weights.data[i]) > 1e-12) { zero = false; break; }
    }
    EXPECT(zero, "all grads zero after zero_grad");
}

// Test 11: update_weights moves parameters.
void test_update_weights() {
    HGRNCell cell(2, 2);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(3, 2);
    Tensor y = cell.forward(x);
    Tensor grad_loss(3, 2);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) grad_loss.data[i] = 0.5;
    cell.backward(grad_loss, 0.0);

    Tensor w_before = cell.W_f.weights.clone();
    cell.update_weights(0.01);
    bool moved = false;
    for (size_t i = 0; i < w_before.data.size(); ++i) {
        if (std::fabs(cell.W_f.weights.data[i] - w_before.data[i]) > 1e-12) {
            moved = true; break;
        }
    }
    EXPECT(moved, "update_weights changes W_f.weights");
}

// Test 12: HGRNModel forward shape + finiteness.
void test_model_forward() {
    HGRNModel model(3, 5, 2, /*num_layers=*/2, /*use_output_gate=*/true);
    EXPECT(model.num_layers() == 2, "num_layers");
    Tensor x = make_input(5, 3, 0.2);
    Tensor y = model.forward(x);
    EXPECT(y.rows == 1, "model output rows");
    EXPECT(y.cols == 2, "model output cols (output_dim)");
    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "model output finite");
}

// Test 13: HGRNModel training reduces loss.
void test_training_reduces_loss() {
    HGRNModel model(2, 4, 1, /*num_layers=*/1, /*use_output_gate=*/false);
    Tensor x(4, 2);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j)
            x[i][j] = 0.2 * (i + j + 1);
    const double target = 0.5;

    Tensor y0 = model.forward(x);
    double L0 = std::pow(y0[0][0] - target, 2);

    const double lr = 0.05;
    for (size_t step = 0; step < 80; ++step) {
        Tensor y = model.forward(x);
        Tensor grad(1, 1);
        grad[0][0] = 2.0 * (y[0][0] - target);
        model.zero_grad();
        model.backward(grad, 0.0);
        model.update_weights(lr);
    }
    Tensor y1 = model.forward(x);
    double L1 = std::pow(y1[0][0] - target, 2);
    std::printf("    training: L0=%.4f → L1=%.4f (%.1f%% reduction)\n",
                L0, L1, 100.0 * (L0 - L1) / std::max(L0, 1e-12));
    EXPECT(L1 < L0, "training reduces loss");
}

// Test 14: longer sequence T=6, hidden=4 (verifies carry-forward chain).
void test_longer_sequence_hgrn1() {
    HGRNCell cell(3, 4);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(6, 3, 0.3);
    Tensor y = cell.forward(x);
    EXPECT(y.rows == 6 && y.cols == 4, "shape T=6, hidden=4");

    bool differs = false;
    for (size_t c = 0; c < 4; ++c) {
        if (std::fabs(y[5][c] - y[0][c]) > 1e-9) { differs = true; break; }
    }
    EXPECT(differs, "y[T-1] differs from y[0] (state evolved)");

    Tensor grad_loss(6, 4);
    for (size_t i = 0; i < grad_loss.data.size(); ++i) grad_loss.data[i] = 0.1 * (i + 1);
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad_loss.data.size(); ++i) {
            s += grad_loss.data[i] * y_pred.data[i];
        }
        return s;
    };
    auto forward_fn = [&]() { return cell.forward(x); };

    Tensor fd_Wf = compute_fd_grad_param(&cell.W_f.weights, forward_fn, loss_fn, 1e-5);
    double max_rel = 0.0;
    for (size_t i = 0; i < cell.W_f.grad_weights.rows; ++i) {
        for (size_t j = 0; j < cell.W_f.grad_weights.cols; ++j) {
            double ana = cell.W_f.grad_weights[i][j];
            double fn = fd_Wf[i][j];
            double scale = std::max({std::fabs(ana), std::fabs(fn), 1e-12});
            max_rel = std::max(max_rel, std::fabs(ana - fn) / scale);
        }
    }
    std::printf("    T=6 W_f.weights grad: max rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-3, "W_f.weights grad FD on T=6");
}

// Test 15: sigmoid(0) = 0.5 → f=0.5, i=0.5 → c_t = 0.5*c_{t-1} + 0.5*z_t.
// Verify zero input → zero output (z=tanh(0)=0).
void test_zero_input_zero_output() {
    HGRNCell cell(1, 1, /*use_output_gate=*/false);
    cell.W_f.weights[0][0] = 0.0; cell.W_f.bias[0][0] = 0.0;
    cell.W_i.weights[0][0] = 0.0; cell.W_i.bias[0][0] = 0.0;
    cell.W_z.weights[0][0] = 0.0; cell.W_z.bias[0][0] = 0.0;

    Tensor x(3, 1);
    x[0][0] = 0.0; x[1][0] = 0.0; x[2][0] = 0.0;
    Tensor y = cell.forward(x);
    EXPECT_NEAR(y[0][0], 0.0, TOL_STRICT, "zero-input → zero-output step 0");
    EXPECT_NEAR(y[1][0], 0.0, TOL_STRICT, "zero-input → zero-output step 1");
    EXPECT_NEAR(y[2][0], 0.0, TOL_STRICT, "zero-input → zero-output step 2");
}

// Test 16: HGRNModel parameter gradient check on the last-step classifier
// (verifies the last-timestep → classifier chain in the model).
void test_model_classifier_grad() {
    HGRNModel model(2, 3, 2, /*num_layers=*/1, /*use_output_gate=*/false);
    Tensor x(3, 2);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 2; ++j)
            x[i][j] = 0.2 * (i + j + 1);
    Tensor y = model.forward(x);
    Tensor grad_loss(1, 2);
    grad_loss[0][0] = 0.5;
    grad_loss[0][1] = 0.3;
    model.zero_grad();
    model.backward(grad_loss, 0.0);

    // Loss for FD: grad_loss .* y_pred
    auto loss_fn = [&](const Tensor& y_pred) {
        return grad_loss[0][0] * y_pred[0][0] + grad_loss[0][1] * y_pred[0][1];
    };
    auto forward_fn = [&]() { return model.forward(x); };

    Tensor fd_classifier = compute_fd_grad_param(&model.classifier.weights, forward_fn, loss_fn, 1e-5);
    double max_rel = 0.0;
    for (size_t i = 0; i < model.classifier.grad_weights.rows; ++i) {
        for (size_t j = 0; j < model.classifier.grad_weights.cols; ++j) {
            double ana = model.classifier.grad_weights[i][j];
            double fn = fd_classifier[i][j];
            double scale = std::max({std::fabs(ana), std::fabs(fn), 1e-12});
            max_rel = std::max(max_rel, std::fabs(ana - fn) / scale);
        }
    }
    std::printf("    model classifier W grad: max rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-3, "model classifier W grad");
}

}  // namespace

int main() {
    std::printf("=== HGRN (Hierarchically Gated Recurrent Neural Network) Tests ===\n");

    test_constructor_and_forward_shape();
    test_forward_with_output_gate();
    test_hand_derived_forward_hgrn1();
    test_hand_derived_forward_hgrn2();
    test_determinism();
    test_input_gradient_hgrn1();
    test_parameter_gradients_hgrn1();
    test_parameter_gradients_hgrn2();
    test_parameters_gradients_contract();
    test_zero_grad();
    test_update_weights();
    test_model_forward();
    test_training_reduces_loss();
    test_longer_sequence_hgrn1();
    test_zero_input_zero_output();
    test_model_classifier_grad();

    std::printf("\n=== Summary: %d passed, %d failed ===\n",
                g_tests_passed, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
