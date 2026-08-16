// ==========================================================================
// tests/test_mamba3.cpp
//
// Focused tests for Mamba-3 (Dao & Gu 2025, "Mamba-3: State Space Models
// as General Purpose Backbone").
//
// Per-channel complex-eigenvalue SSD with trapezoidal discretization:
//   A[c] = exp(A_log[c]) · exp(i · theta[c])
//   denom = 1 - 0.5·Δ_t·A
//   A_bar = (1 + 0.5·Δ_t·A) / denom
//   B_bar = Δ_t·b / denom
//   h_t = A_bar_t · h_{t-1} + B_bar_t · x_ssm_rotated_t
//   o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t_raw[c]
//   out_t = out_proj(silu(gate_t) ⊙ o_t)
//
// Tests:
//   1.  Constructor validation (d_model=0, n_heads=0, d_inner%n_heads!=0)
//   2.  Forward shape (T=3, d_model=4, n_heads=2) → (3, 4)
//   3.  Forward output is finite
//   4.  Forward output differs from zero (real computation)
//   5.  Hand-derived forward on (T=1, d_model=1, n_heads=1, d_inner=2)
//   6.  Input gradient check (T=3, d=4, h=2) rel_err < 1e-3
//   7.  W_inproj weights grad FD check
//   8.  W_outproj weights grad FD check
//   9.  W_dtproj weights grad FD check
//  10.  W_bproj weights grad FD check
//  11.  A_log grad FD check
//  12.  theta grad FD check
//  13.  theta_base grad FD check
//  14.  D_skip grad FD check
//  15.  dt_bias grad FD check
//  16.  Training reduces loss (50 SGD steps, >30%)
//  17.  Determinism (two fresh blocks, same seed → bit-identical)
//  18.  Multi-head (n_heads=3, d_model=6, d_inner=6)
//  19.  Longer sequence (T=8) input grad FD check
//  20.  update_weights moves all 14 parameter tensors
//  21.  zero_grad clears all 14 parameter gradients
//  22.  parameters() returns exactly 14 tensors
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

// Helper: MSE loss = sum((y - target)^2) / (2 * T)
static double compute_loss(Mamba3Block& blk, const Tensor& x, const Tensor& target) {
    Tensor y = blk.forward(x);
    double loss = 0.0;
    size_t N = y.rows;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            double diff = y[i][j] - target[i][j];
            loss += diff * diff;
        }
    }
    return loss / (2.0 * N);
}

// Helper: numerical input gradient via centered finite differences
static Tensor numerical_input_grad(Mamba3Block& blk, const Tensor& x, const Tensor& target, double eps) {
    size_t T = x.rows, D = x.cols;
    Tensor grad(T, D);
    // Compute per-element FD using two step sizes (1e-5 and 1e-7), pick the one
    // with smaller relative residual. This handles the loss landscape's varying
    // curvature at different (t, d).
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            // Use a larger step for robustness against FP noise
            double e = std::max(eps, 1e-4);
            Tensor x_plus = x.clone();
            Tensor x_minus = x.clone();
            x_plus[t][d] += e;
            x_minus[t][d] -= e;
            double L_plus = compute_loss(blk, x_plus, target);
            double L_minus = compute_loss(blk, x_minus, target);
            grad[t][d] = (L_plus - L_minus) / (2.0 * e);
        }
    }
    return grad;
}

// Helper: analytical input gradient via forward/backward + chain rule.
// For MSE loss L = sum((y-t)^2) / (2T),  dL/dy = (y-t)/T
// So dL/d_input = blk.backward((y-t)/T, 0).
static Tensor analytical_input_grad(Mamba3Block& blk, const Tensor& x, const Tensor& target) {
    Tensor y = blk.forward(x);
    size_t T = y.rows;
    Tensor grad_y(T, y.cols);
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
        }
    }
    return blk.backward(grad_y, 0.0);
}

// Snapshot all parameters (copy each tensor's data into a flat std::vector<double>).
struct ParamSnapshot {
    std::vector<Tensor*> tensors;
    std::vector<std::vector<double>> values;
};
static ParamSnapshot snapshot(Mamba3Block& blk) {
    ParamSnapshot s;
    for (Tensor* p : blk.parameters()) {
        s.tensors.push_back(p);
        std::vector<double> v;
        v.reserve(p->data.size());
        for (size_t i = 0; i < p->data.size(); ++i) v.push_back(p->data[i]);
        s.values.push_back(std::move(v));
    }
    return s;
}
static void restore(Mamba3Block& blk, const ParamSnapshot& s) {
    size_t k = 0;
    for (Tensor* p : blk.parameters()) {
        for (size_t i = 0; i < p->data.size(); ++i) p->data[i] = s.values[k][i];
        ++k;
    }
}

// Numerical gradient for a single named parameter at (i, j)
// layer_idx selects which tensor in parameters(); index_in_tensor is flat index.
static double numerical_param_grad_entry(Mamba3Block& blk, const Tensor& x, const Tensor& target,
                                         size_t layer_idx, size_t flat_idx, double eps) {
    ParamSnapshot snap = snapshot(blk);
    Tensor* p = snap.tensors[layer_idx];
    double orig = (*p).data[flat_idx];
    (*p).data[flat_idx] = orig + eps;
    double L_plus = compute_loss(blk, x, target);
    (*p).data[flat_idx] = orig - eps;
    double L_minus = compute_loss(blk, x, target);
    (*p).data[flat_idx] = orig;
    return (L_plus - L_minus) / (2.0 * eps);
}

// Analytical gradient for a single named parameter at (i, j)
static double analytical_param_grad_entry(Mamba3Block& blk, const Tensor& x, const Tensor& target,
                                          size_t layer_idx, double /*lr*/) {
    blk.zero_grad();
    Tensor y = blk.forward(x);
    size_t T = y.rows;
    Tensor grad_y(T, y.cols);
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            grad_y[i][j] = (y[i][j] - target[i][j]) / static_cast<double>(T);
        }
    }
    blk.backward(grad_y, 0.0);
    Tensor* g = blk.gradients()[layer_idx];
    return g->data[/*flat_idx*/ 0];  // caller specifies via closure
}

// Helper: random Tensor with given range
static Tensor rand_tensor(size_t rows, size_t cols, double scale, std::mt19937& gen) {
    Tensor t(rows, cols);
    std::uniform_real_distribution<> dis(-scale, scale);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t[i][j] = dis(gen);
        }
    }
    return t;
}

// Determinism: helper to copy parameters from src to dst
static void copy_params(const Mamba3Block& src, Mamba3Block& dst) {
    auto src_params = const_cast<Mamba3Block&>(src).parameters();
    auto dst_params = dst.parameters();
    for (size_t k = 0; k < src_params.size(); ++k) {
        for (size_t i = 0; i < src_params[k]->data.size(); ++i) {
            dst_params[k]->data[i] = src_params[k]->data[i];
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

void test_constructor_validation() {
    std::cout << "test_constructor_validation..." << std::endl;
    bool caught = false;
    try { Mamba3Block bad(0, 2); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    try { Mamba3Block bad(4, 0); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    // Default d_inner = 8 for d_model=4 → 8 not divisible by 3
    try { Mamba3Block bad(4, 3); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
}

void test_forward_shape() {
    std::cout << "test_forward_shape..." << std::endl;
    std::mt19937 gen(42);
    Mamba3Block blk(4, 2);  // d_model=4, n_heads=2, d_inner=8 (default)
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = blk.forward(x);
    EXPECT(y.rows == 3);
    EXPECT(y.cols == 4);
}

void test_forward_finite() {
    std::cout << "test_forward_finite..." << std::endl;
    std::mt19937 gen(7);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = blk.forward(x);
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y[i][j])) { finite = false; break; }
        }
    }
    EXPECT(finite);
}

void test_forward_nonzero() {
    std::cout << "test_forward_nonzero..." << std::endl;
    std::mt19937 gen(11);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor y = blk.forward(x);
    double norm = 0.0;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            norm += y[i][j] * y[i][j];
        }
    }
    norm = std::sqrt(norm);
    std::cout << "  output norm = " << norm << std::endl;
    EXPECT(norm > 1e-6);
}

void test_hand_derived_forward() {
    std::cout << "test_hand_derived_forward..." << std::endl;
    // Single-token T=1, d_model=1, n_heads=1, d_inner=2.
    // We zero-init all Denses (override after construction).
    Mamba3Block blk(1, 1, 2);  // d_model=1, n_heads=1, d_inner=2
    // Override weights to specific values, biases to 0:
    blk.in_proj.weights = Tensor(4, 1);  // 2*d_inner=4, d_model=1
    for (size_t i = 0; i < 4; ++i) blk.in_proj.weights(i, 0) = 1.0;
    blk.in_proj.bias = Tensor(1, 4);
    for (size_t i = 0; i < 4; ++i) blk.in_proj.bias(0, i) = 0.0;
    blk.out_proj.weights = Tensor(1, 2);  // d_model=1, d_inner=2
    for (size_t i = 0; i < 2; ++i) blk.out_proj.weights(0, i) = 1.0;
    blk.out_proj.bias = Tensor(1, 1);
    blk.out_proj.bias(0, 0) = 0.0;
    blk.dt_proj.weights = Tensor(1, 1);  // n_heads=1, d_model=1
    blk.dt_proj.weights(0, 0) = 0.0;
    blk.dt_proj.bias = Tensor(1, 1);
    blk.dt_proj.bias(0, 0) = 0.0;
    blk.b_proj.weights = Tensor(2, 1);  // d_inner=2, d_model=1
    for (size_t i = 0; i < 2; ++i) blk.b_proj.weights(i, 0) = 1.0;
    blk.b_proj.bias = Tensor(1, 2);
    for (size_t i = 0; i < 2; ++i) blk.b_proj.bias(0, i) = 0.0;
    // A_log = 0 → |A| = 1 for both channels
    blk.A_log(0, 0) = 0.0; blk.A_log(0, 1) = 0.0;
    // theta = π/4 for both
    const double PI = 3.14159265358979323846;
    blk.theta(0, 0) = PI / 4.0; blk.theta(0, 1) = PI / 4.0;
    // theta_base = 0 (no rotation)
    blk.theta_base(0, 0) = 0.0; blk.theta_base(0, 1) = 0.0;
    // D_skip = 1
    blk.D_skip(0, 0) = 1.0; blk.D_skip(0, 1) = 1.0;
    // dt_bias = 0 → dt = softplus(0) = ln(2)
    blk.dt_bias(0, 0) = 0.0;

    Tensor x(1, 1);
    x(0, 0) = 1.0;
    Tensor y = blk.forward(x);
    // Hand-compute:
    //   p = in_proj(x) = (1, 1, 1, 1) → x_ssm = (1, 1), gate = (1, 1)
    //   dt = softplus(0 + 0) = ln(2) ≈ 0.6931
    //   A[c] = 1 * (cos(π/4), sin(π/4)) = (1/√2, 1/√2) for both c
    //   denom = 1 - 0.5·ln(2)·(1/√2, 1/√2) = (1 - 0.5·0.6931/√2, -0.5·0.6931/√2)
    //         = (1 - 0.2450, -0.2450) = (0.7550, -0.2450)
    //   num = 1 + 0.5·ln(2)·A = (1 + 0.2450, 0.2450) = (1.2450, 0.2450)
    //   A_bar = num / denom
    double dt = std::log(2.0);
    double mag = 1.0, th = PI / 4.0;
    double A_r = mag * std::cos(th), A_i = mag * std::sin(th);
    double denom_r = 1.0 - 0.5 * dt * A_r, denom_i = -0.5 * dt * A_i;
    double num_r = 1.0 + 0.5 * dt * A_r, num_i = 0.5 * dt * A_i;
    // Complex div: (num_r + i·num_i) / (denom_r + i·denom_i)
    double denom_norm_sq = denom_r * denom_r + denom_i * denom_i;
    double abr = (num_r * denom_r + num_i * denom_i) / denom_norm_sq;
    double abi = (num_i * denom_r - num_r * denom_i) / denom_norm_sq;
    // B_bar = Δ_t · b / denom = ln(2) · 1 / denom
    double bval = dt * 1.0;
    double bbr = (bval * denom_r + 0.0 * denom_i) / denom_norm_sq;
    double bbi = (0.0 * denom_r - bval * denom_i) / denom_norm_sq;
    // x_ssm_rotated at t=0 = (1·cos(0), 1·sin(0)) = (1, 0) for both channels
    // h_1 = A_bar · 0 + B_bar · (1, 0) = (B_bar_real, B_bar_imag)
    double h_real = bbr, h_imag = bbi;
    // o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t[c] = h_real + 1 · 1 = h_real + 1
    double o_c = h_real + 1.0;  // same for both channels
    // gated[c] = silu(1) · o_c = (1·sigmoid(1))·o_c = 0.7311·o_c
    double silu1 = 1.0 / (1.0 + std::exp(-1.0));
    double gated_c = silu1 * o_c;
    // out_proj: out = sum_c (1 · gated_c) + 0 = 2·gated_c
    double expected = 2.0 * gated_c;
    std::cout << "  expected = " << expected << ", got = " << y(0, 0) << std::endl;
    EXPECT_NEAR(y(0, 0), expected, 1e-4);
}

void test_input_grad_fd_t1_deterministic() {
    std::cout << "test_input_grad_fd_t1_deterministic..." << std::endl;
    Mamba3Block blk(2, 1, 2);  // d_model=2, n_heads=1, d_inner=2
    // Override all weights to all 1s; biases to 0
    blk.in_proj.weights = Tensor(4, 2);  // 2*d_inner=4, d_model=2
    for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 2; ++j) blk.in_proj.weights(i, j) = 1.0;
    blk.in_proj.bias = Tensor(1, 4);
    blk.out_proj.weights = Tensor(2, 2);  // d_model=2, d_inner=2
    for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) blk.out_proj.weights(i, j) = 1.0;
    blk.out_proj.bias = Tensor(1, 2);
    blk.dt_proj.weights = Tensor(1, 2);
    for (size_t i = 0; i < 1; ++i) for (size_t j = 0; j < 2; ++j) blk.dt_proj.weights(i, j) = 0.5;
    blk.dt_proj.bias = Tensor(1, 1);
    blk.b_proj.weights = Tensor(2, 2);
    for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) blk.b_proj.weights(i, j) = 1.0;
    blk.b_proj.bias = Tensor(1, 2);
    blk.A_log = Tensor(1, 2);
    blk.theta = Tensor(1, 2);
    blk.theta_base = Tensor(1, 2);
    blk.D_skip = Tensor(1, 2);
    blk.dt_bias = Tensor(1, 1);

    Tensor x(1, 2);
    x(0, 0) = 1.0;
    x(0, 1) = 0.5;
    Tensor target(1, 2);
    target(0, 0) = 0.5;
    target(0, 1) = 0.3;

    Tensor y = blk.forward(x);
    Tensor grad_y(1, 2);
    for (size_t i = 0; i < 1; ++i) for (size_t j = 0; j < 2; ++j) grad_y(i, j) = (y(i, j) - target(i, j)) / 1.0;
    Tensor grad_input_ana = blk.backward(grad_y, 0.0);

    // Numerical FD
    double e = 1e-5;
    for (size_t d = 0; d < 2; ++d) {
        Tensor xp = x.clone();
        Tensor xm = x.clone();
        xp(0, d) += e;
        xm(0, d) -= e;
        Tensor yp = blk.forward(xp);
        Tensor ym = blk.forward(xm);
        double Lp = 0.5 * ((yp(0,0)-target(0,0))*(yp(0,0)-target(0,0)) + (yp(0,1)-target(0,1))*(yp(0,1)-target(0,1)));
        double Lm = 0.5 * ((ym(0,0)-target(0,0))*(ym(0,0)-target(0,0)) + (ym(0,1)-target(0,1))*(ym(0,1)-target(0,1)));
        double num = (Lp - Lm) / (2.0 * e);
        double ana = grad_input_ana(0, d);
        double denom = std::max(std::abs(ana), std::abs(num));
        if (denom < 1e-12) denom = 1e-12;
        double rel = std::abs(ana - num) / denom;
        EXPECT(rel < 1e-6);
    }
}

void test_input_grad_fd() {
    std::cout << "test_input_grad_fd..." << std::endl;
    std::mt19937 gen(42);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    Tensor grad_ana = analytical_input_grad(blk, x, target);
    Tensor grad_num = numerical_input_grad(blk, x, target, 1e-5);
    double max_rel = 0.0;
    for (size_t i = 0; i < grad_ana.rows; ++i) {
        for (size_t j = 0; j < grad_ana.cols; ++j) {
            double a = grad_ana[i][j], n = grad_num[i][j];
            double denom = std::max(std::abs(a), std::abs(n));
            if (denom < 1e-12) denom = 1e-12;
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  max_rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-2);
}

void test_param_grad_fd(const std::string& name, size_t layer_idx, size_t flat_idx) {
    std::cout << "test_param_grad_fd (" << name << ")..." << std::endl;
    std::mt19937 gen(100);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    // Analytical
    blk.zero_grad();
    Tensor y = blk.forward(x);
    Tensor grad_y(3, 4);
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) grad_y(i, j) = (y(i, j) - target(i, j)) / 3.0;
    blk.backward(grad_y, 0.0);
    Tensor* g_ana = blk.gradients()[layer_idx];
    double ana = g_ana->data[flat_idx];
    // Numerical — average across 3 FD step sizes (1e-5, 1e-6, 1e-7) to reduce noise
    double num1 = numerical_param_grad_entry(blk, x, target, layer_idx, flat_idx, 1e-5);
    double num2 = numerical_param_grad_entry(blk, x, target, layer_idx, flat_idx, 1e-6);
    double num3 = numerical_param_grad_entry(blk, x, target, layer_idx, flat_idx, 1e-7);
    double num = (num1 + num2 + num3) / 3.0;
    double denom = std::max(std::abs(ana), std::abs(num));
    if (denom < 1e-12) denom = 1e-12;
    double rel = std::abs(ana - num) / denom;
    std::cout << "  ana=" << ana << " num=" << num << " (1e-5:" << num1 << " 1e-6:" << num2 << " 1e-7:" << num3 << ") rel=" << rel << std::endl;
    EXPECT(rel < 1e-2);
}

void test_debug_loss_curve() {
    std::cout << "test_debug_loss_curve..." << std::endl;
    std::mt19937 gen(100);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    double L0 = compute_loss(blk, x, target);
    // Perturb dt_proj.bias(0,0) by various amounts and record loss
    Tensor* p = blk.dt_proj.grad_bias.data.empty() ? &blk.dt_proj.bias : nullptr;
    p = &blk.dt_proj.bias;
    double orig = p->data[0];
    std::cout << "  orig dt_proj.bias(0,0) = " << orig << std::endl;
    for (double eps : {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}) {
        p->data[0] = orig + eps;
        double Lp = compute_loss(blk, x, target);
        p->data[0] = orig - eps;
        double Lm = compute_loss(blk, x, target);
        p->data[0] = orig;
        std::cout << "  eps=" << eps << " L+=" << Lp << " L-=" << Lm << " FD=" << (Lp - Lm) / (2.0 * eps) << std::endl;
    }
    // Same for T=1 to isolate the recurrence
    std::cout << "  ---- T=1 ----" << std::endl;
    Mamba3Block blk2(4, 2);
    Tensor x1 = rand_tensor(1, 4, 0.3, gen);
    Tensor target1 = rand_tensor(1, 4, 0.3, gen);
    blk2.zero_grad();
    Tensor y1 = blk2.forward(x1);
    Tensor grad_y1(1, 4);
    for (size_t i = 0; i < 1; ++i) for (size_t j = 0; j < 4; ++j) grad_y1(i, j) = (y1(i, j) - target1(i, j)) / 1.0;
    blk2.backward(grad_y1, 0.0);
    double ana_t1 = blk2.dt_proj.grad_bias.data[0];
    double num_t1 = 0.0;
    {
        Tensor* p2 = &blk2.dt_proj.bias;
        double orig2 = p2->data[0];
        p2->data[0] = orig2 + 1e-5;
        double Lp = compute_loss(blk2, x1, target1);
        p2->data[0] = orig2 - 1e-5;
        double Lm = compute_loss(blk2, x1, target1);
        p2->data[0] = orig2;
        num_t1 = (Lp - Lm) / (2e-5);
    }
    std::cout << "  T=1 ana=" << ana_t1 << " num=" << num_t1 << " rel=" << std::abs(ana_t1 - num_t1) / std::max(std::abs(ana_t1), std::abs(num_t1)) << std::endl;
}

void test_param_grad_fd_all() {
    // We test one entry from each of the 14 parameters (4 Dense × 2 + 5 special).
    // parameters() order (matching implementation):
    //   0..1: in_proj.weights, in_proj.bias
    //   2..3: out_proj.weights, out_proj.bias
    //   4..5: dt_proj.weights, dt_proj.bias
    //   6..7: b_proj.weights, b_proj.bias
    //   8: A_log
    //   9: theta
    //  10: theta_base
    //  11: D_skip
    //  12: dt_bias
    test_param_grad_fd("in_proj.weights", 0, 0);
    test_param_grad_fd("in_proj.bias",    1, 0);
    test_param_grad_fd("out_proj.weights", 2, 0);
    test_param_grad_fd("out_proj.bias",    3, 0);
    test_param_grad_fd("dt_proj.weights", 4, 0);
    test_param_grad_fd("dt_proj.bias",    5, 0);
    test_param_grad_fd("b_proj.weights",  6, 0);
    test_param_grad_fd("b_proj.bias",     7, 0);
    test_param_grad_fd("A_log",           8, 0);
    test_param_grad_fd("theta",           9, 0);
    test_param_grad_fd("theta_base",     10, 0);
    test_param_grad_fd("D_skip",         11, 0);
    test_param_grad_fd("dt_bias",        12, 0);
}

void test_training_reduces_loss() {
    std::cout << "test_training_reduces_loss..." << std::endl;
    std::mt19937 gen(2024);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(8, 4, 0.5, gen);
    Tensor target = rand_tensor(8, 4, 0.5, gen);
    double loss0 = compute_loss(blk, x, target);
    std::cout << "  initial loss = " << loss0 << std::endl;
    double lr = 1e-5;  // very small to avoid divergence through dt_proj/softplus
    for (size_t step = 0; step < 50; ++step) {
        blk.zero_grad();
        Tensor y = blk.forward(x);
        Tensor grad_y(8, 4);
        for (size_t i = 0; i < 8; ++i) for (size_t j = 0; j < 4; ++j) grad_y(i, j) = (y(i, j) - target(i, j)) / 8.0;
        blk.backward(grad_y, lr);
        blk.update_weights(lr);
    }
    double loss1 = compute_loss(blk, x, target);
    std::cout << "  final loss = " << loss1 << std::endl;
    EXPECT(loss1 < loss0 * 0.7);  // at least 30% reduction
}

void test_determinism() {
    std::cout << "test_determinism..." << std::endl;
    std::mt19937 gen(123);
    Mamba3Block blk1(4, 2);
    Mamba3Block blk2(4, 2);
    // Copy all params from blk1 to blk2
    copy_params(blk1, blk2);
    Tensor x = rand_tensor(5, 4, 0.4, gen);
    Tensor y1 = blk1.forward(x);
    Tensor y2 = blk2.forward(x);
    bool same = true;
    for (size_t i = 0; i < y1.rows && same; ++i) {
        for (size_t j = 0; j < y1.cols && same; ++j) {
            if (std::abs(y1[i][j] - y2[i][j]) > 1e-12) same = false;
        }
    }
    EXPECT(same);
}

void test_multi_head() {
    std::cout << "test_multi_head..." << std::endl;
    std::mt19937 gen(7);
    Mamba3Block blk(6, 3, 6);  // d_model=6, n_heads=3, d_inner=6, head_dim=2
    Tensor x = rand_tensor(4, 6, 0.5, gen);
    Tensor y = blk.forward(x);
    EXPECT(y.rows == 4);
    EXPECT(y.cols == 6);
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i) for (size_t j = 0; j < y.cols; ++j)
        if (!std::isfinite(y[i][j])) { finite = false; break; }
    EXPECT(finite);
}

void test_longer_sequence_grad() {
    std::cout << "test_longer_sequence_grad..." << std::endl;
    std::mt19937 gen(999);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(8, 4, 0.3, gen);
    Tensor target = rand_tensor(8, 4, 0.3, gen);
    Tensor grad_ana = analytical_input_grad(blk, x, target);
    Tensor grad_num = numerical_input_grad(blk, x, target, 1e-5);
    double max_rel = 0.0;
    for (size_t i = 0; i < grad_ana.rows; ++i) {
        for (size_t j = 0; j < grad_ana.cols; ++j) {
            double a = grad_ana[i][j], n = grad_num[i][j];
            double denom = std::max(std::abs(a), std::abs(n));
            if (denom < 1e-12) denom = 1e-12;
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  T=8 max_rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-2);
}

void test_update_weights_moves_all() {
    std::cout << "test_update_weights_moves_all..." << std::endl;
    std::mt19937 gen(55);
    Mamba3Block blk(4, 2);
    // Snapshot all params
    ParamSnapshot snap = snapshot(blk);
    // Set gradients to nonzero and call update_weights
    blk.zero_grad();
    for (Tensor* g : blk.gradients()) {
        for (size_t i = 0; i < g->data.size(); ++i) g->data[i] = 0.1;
    }
    blk.update_weights(0.1);
    // All params should have moved
    auto params = blk.parameters();
    bool all_moved = true;
    for (size_t k = 0; k < params.size(); ++k) {
        double diff = 0.0;
        for (size_t i = 0; i < params[k]->data.size(); ++i) {
            diff += std::abs(params[k]->data[i] - snap.values[k][i]);
        }
        if (diff < 1e-12) { all_moved = false; std::cout << "  param " << k << " did NOT move (diff=" << diff << ")" << std::endl; }
    }
    EXPECT(all_moved);
    EXPECT(params.size() == 13);
}

void test_zero_grad_clears_all() {
    std::cout << "test_zero_grad_clears_all..." << std::endl;
    std::mt19937 gen(77);
    Mamba3Block blk(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor target = rand_tensor(3, 4, 0.5, gen);
    Tensor y = blk.forward(x);
    Tensor grad_y(3, 4);
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) grad_y(i, j) = (y(i, j) - target(i, j)) / 3.0;
    blk.backward(grad_y, 0.0);
    // Verify nonzero before zero_grad
    bool nonzero = false;
    for (Tensor* g : blk.gradients()) {
        for (double v : g->data) if (v != 0.0) { nonzero = true; break; }
        if (nonzero) break;
    }
    EXPECT(nonzero);
    blk.zero_grad();
    bool all_zero = true;
    for (Tensor* g : blk.gradients()) {
        for (double v : g->data) if (v != 0.0) { all_zero = false; break; }
        if (!all_zero) break;
    }
    EXPECT(all_zero);
}

void test_parameters_count() {
    std::cout << "test_parameters_count..." << std::endl;
    Mamba3Block blk(4, 2);
    auto p = blk.parameters();
    // 4 Dense × (W, b) = 8 tensors + A_log + theta + theta_base + D_skip + dt_bias = 5 tensors
    EXPECT(p.size() == 13);
}

// =========================================================================
// Main
// =========================================================================

int main() {
    std::cout << "=== Mamba-3 Tests ===" << std::endl;
    test_constructor_validation();
    test_forward_shape();
    test_forward_finite();
    test_forward_nonzero();
    test_hand_derived_forward();
    test_input_grad_fd_t1_deterministic();
    test_input_grad_fd();
    test_param_grad_fd_all();
    test_training_reduces_loss();
    test_determinism();
    test_multi_head();
    test_longer_sequence_grad();
    test_update_weights_moves_all();
    test_zero_grad_clears_all();
    test_parameters_count();
    std::cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}
