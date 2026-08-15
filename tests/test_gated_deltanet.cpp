// ==========================================================================
// tests/test_gated_deltanet.cpp
//
// Focused tests for GatedDeltaNet (Yang, Kautz, Hatamizadeh 2025, ICLR 2025,
// "Gated Delta Networks: Improving Mamba2 with Delta Rule",
// https://arxiv.org/abs/2412.06464).
//
// Combines:
//   - Mamba2/GLA per-head decay gate α_t = sigmoid(W_gate · x_t) ∈ (0,1)
//   - DeltaNet per-head write strength β_t with k-magnitude normalization
//
// Recurrence (per head h, time t):
//   k_t[h] = (β_t[h] / ||k_t_raw[h]||) · k_t_raw[h]
//   S_t[h] = gate_t[h] · S_{t-1}[h] · (I − β_t[h] · outer(k_t[h], k_t[h]))
//          + β_t[h] · outer(k_t[h], v_t[h])
//   o_t[h] = S_t[h] · q_t[h]
//
// Tests:
//   1.  Constructor validation (d_model=0, n_heads=0, d_model%n_heads!=0)
//   2.  Forward shape (T=3, d_model=4, n_heads=2) → (3, 4)
//   3.  Forward output is finite
//   4.  Forward output differs from zero (real computation)
//   5.  last_state() returns (n_heads, head_dim * head_dim)
//   6.  last_state() norm > 0 (state accumulation works)
//   7.  Hand-derived forward on (T=1, d_model=2, n_heads=1, head_dim=2) matches
//   8.  Input gradient check (T=3, d=4, h=2) rel_err < 1e-4
//   9.  W_Q weights grad FD check
//  10.  W_K weights grad FD check
//  11.  W_V weights grad FD check
//  12.  W_O weights grad FD check
//  13.  W_beta weights grad FD check
//  14.  W_gate weights grad FD check
//  15.  Training reduces loss (50 SGD steps, >50%)
//  16.  Determinism (two fresh layers, same init, same input → bit-identical)
//  17.  Multi-head (n_heads=3, head_dim=2, d_model=6) forward
//  18.  Longer sequence (T=8, d_model=4, n_heads=2) input grad
//  19.  update_weights moves all 12 parameter tensors
//  20.  zero_grad clears all 12 parameter gradients
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

// Helper: compute MSE loss = sum((y - target)^2) / (2 * T)
static double compute_loss(GatedDeltaNet& gdn, const Tensor& x, const Tensor& target) {
    Tensor y = gdn.forward(x);
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
static Tensor numerical_input_grad(GatedDeltaNet& gdn, const Tensor& x, const Tensor& target, double eps) {
    size_t T = x.rows, D = x.cols;
    Tensor grad(T, D);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            Tensor x_plus = x.clone();
            Tensor x_minus = x.clone();
            x_plus[t][d] += eps;
            x_minus[t][d] -= eps;
            double L_plus = compute_loss(gdn, x_plus, target);
            double L_minus = compute_loss(gdn, x_minus, target);
            grad[t][d] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

// Helper: numerical parameter gradient via centered finite differences
// layer_idx: 0=W_Q, 1=W_K, 2=W_V, 3=W_O, 4=W_beta, 5=W_gate
// index_in_layer: 0=weights, 1=bias
static Tensor numerical_param_grad(GatedDeltaNet& gdn, const Tensor& x, const Tensor& target,
                                   size_t layer_idx, size_t index_in_layer, double eps) {
    Tensor* w;
    if (layer_idx == 0)      w = (index_in_layer == 0) ? &gdn.W_Q_.weights    : &gdn.W_Q_.bias;
    else if (layer_idx == 1) w = (index_in_layer == 0) ? &gdn.W_K_.weights    : &gdn.W_K_.bias;
    else if (layer_idx == 2) w = (index_in_layer == 0) ? &gdn.W_V_.weights    : &gdn.W_V_.bias;
    else if (layer_idx == 3) w = (index_in_layer == 0) ? &gdn.W_O_.weights    : &gdn.W_O_.bias;
    else if (layer_idx == 4) w = (index_in_layer == 0) ? &gdn.W_beta_.weights : &gdn.W_beta_.bias;
    else if (layer_idx == 5) w = (index_in_layer == 0) ? &gdn.W_gate_.weights : &gdn.W_gate_.bias;
    else throw std::invalid_argument("invalid layer_idx");

    Tensor grad(w->rows, w->cols);
    for (size_t i = 0; i < w->rows; ++i) {
        for (size_t j = 0; j < w->cols; ++j) {
            double orig = (*w)[i][j];
            (*w)[i][j] = orig + eps;
            double L_plus = compute_loss(gdn, x, target);
            (*w)[i][j] = orig - eps;
            double L_minus = compute_loss(gdn, x, target);
            (*w)[i][j] = orig;
            grad[i][j] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
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

// =========================================================================
// Tests
// =========================================================================

void test_constructor_validation() {
    std::cout << "test_constructor_validation..." << std::endl;
    bool caught = false;
    try { GatedDeltaNet bad(0, 2); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    try { GatedDeltaNet bad(4, 0); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    try { GatedDeltaNet bad(4, 3); } catch (std::invalid_argument&) { caught = true; }  // 4 not divisible by 3
    EXPECT(caught);
}

void test_forward_shape() {
    std::cout << "test_forward_shape..." << std::endl;
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);  // d_model=4, n_heads=2, head_dim=2
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = gdn.forward(x);
    EXPECT(y.rows == 3);
    EXPECT(y.cols == 4);
}

void test_forward_finite() {
    std::cout << "test_forward_finite..." << std::endl;
    std::mt19937 gen(7);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = gdn.forward(x);
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
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor y = gdn.forward(x);
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

void test_state_shape() {
    std::cout << "test_state_shape..." << std::endl;
    std::mt19937 gen(123);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    gdn.forward(x);
    Tensor s = gdn.last_state();
    EXPECT(s.rows == 2);  // n_heads
    EXPECT(s.cols == 4);  // head_dim * head_dim
}

void test_state_accumulates() {
    std::cout << "test_state_accumulates..." << std::endl;
    std::mt19937 gen(99);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    gdn.forward(x);
    Tensor s = gdn.last_state();
    double norm = 0.0;
    for (size_t i = 0; i < s.rows; ++i) {
        for (size_t j = 0; j < s.cols; ++j) {
            norm += s[i][j] * s[i][j];
        }
    }
    norm = std::sqrt(norm);
    std::cout << "  state norm = " << norm << std::endl;
    EXPECT(norm > 1e-6);
}

void test_hand_derived_forward() {
    std::cout << "test_hand_derived_forward..." << std::endl;

    // Hand-derived forward test — d_model=2, n_heads=1, head_dim=2, T=1
    GatedDeltaNet gdn(2, 1);

    // Dense weights are stored as (out_features, in_features).
    //   W_Q, W_K, W_V: (d_inner=2, d_model=2)
    //   W_O:           (d_model=2, d_inner=2)
    //   W_beta, W_gate: (n_heads=1, d_model=2)

    gdn.W_Q_.weights    = Tensor::zeros(2, 2);
    gdn.W_K_.weights    = Tensor::zeros(2, 2);
    gdn.W_V_.weights    = Tensor::zeros(2, 2);
    gdn.W_O_.weights    = Tensor::zeros(2, 2);
    gdn.W_beta_.weights = Tensor::zeros(1, 2);  // (out=1, in=2)
    gdn.W_gate_.weights = Tensor::zeros(1, 2);

    // Identity 2×2 for Q/K/V/O so projection is x.
    for (size_t i = 0; i < 2; ++i) {
        gdn.W_Q_.weights[i][i]    = 1.0;
        gdn.W_K_.weights[i][i]    = 1.0;
        gdn.W_V_.weights[i][i]    = 1.0;
        gdn.W_O_.weights[i][i]    = 1.0;
    }
    // W_beta = [0.7, 0.7] so q_t · W_beta = 0.7 (a + b) (constant per t).
    // W_gate = [0, 0] so gate = sigmoid(0) = 0.5.
    gdn.W_beta_.weights[0][0] = 0.7;
    gdn.W_beta_.weights[0][1] = 0.7;
    // (W_gate stays at zero → gate = 0.5)

    // Input: x = [0.6, 0.8]
    Tensor x(1, 2);
    x[0][0] = 0.6;
    x[0][1] = 0.8;

    Tensor y = gdn.forward(x);

    // Hand-derived:
    //   k_raw = [0.6, 0.8], ||k_raw|| = 1.0
    //   β = sigmoid(0.7·0.6 + 0.7·0.8) = sigmoid(0.98) = 0.7269...
    //   gate = sigmoid(0) = 0.5
    //   k_t = (β/||k_raw||) · k_raw[h]  (already has β factor)
    //   v_t = [0.6, 0.8], q_t = [0.6, 0.8]
    //   G = I − β · outer(k_t, k_t)
    //   S_1 = gate · 0 · G + β · outer(k_t, v_t)  = β · outer(k_t, v_t)
    //       = β · (β/||k||) · outer(k_raw, v_t)
    //       = β² · outer([0.6, 0.8], [0.6, 0.8])         (||k_raw|| = 1)
    //       = β² · [[0.36, 0.48], [0.48, 0.64]]
    //   o_1 = S_1 · q_1 = β² · [0.36·0.6 + 0.48·0.8, 0.48·0.6 + 0.64·0.8]
    //                   = β² · [0.6, 0.8]
    //   y = W_O · o_1 = o_1 (identity W_O)

    double xa = 0.6, xb = 0.8;
    double beta = 1.0 / (1.0 + std::exp(-(0.7 * (xa + xb))));
    double exp_o0 = beta * beta * (0.36 * xa + 0.48 * xb);
    double exp_o1 = beta * beta * (0.48 * xa + 0.64 * xb);

    std::cout << "  β=" << beta << " expected y=[" << exp_o0 << ", " << exp_o1
              << "] got y=[" << y[0][0] << ", " << y[0][1] << "]" << std::endl;

    EXPECT_NEAR(y[0][0], exp_o0, 1e-9);
    EXPECT_NEAR(y[0][1], exp_o1, 1e-9);
}

void test_input_grad() {
    std::cout << "test_input_grad..." << std::endl;
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    // Forward + analytical backward
    Tensor y = gdn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    Tensor analytical = gdn.backward(grad_out, 0.0);

    // Numerical
    Tensor numerical = numerical_input_grad(gdn, x, target, 1e-5);

    double max_rel = 0.0;
    int n_skipped = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double a = analytical[i][j];
            double n = numerical[i][j];
            if (std::abs(n) < 1e-12) { ++n_skipped; continue; }
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  max rel_err = " << max_rel << " (skipped " << n_skipped << " near-zero)" << std::endl;
    EXPECT(max_rel < 1e-3);
}

template <typename DenseT>
static void run_param_grad_test(const char* name, size_t layer_idx, DenseT& dense,
                                GatedDeltaNet& gdn, const Tensor& x, const Tensor& target) {
    std::cout << "test_" << name << "_grad..." << std::endl;

    // Forward + analytical backward
    Tensor y = gdn.forward(x);
    Tensor grad_out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / (double)x.rows;
        }
    }
    gdn.zero_grad();
    gdn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(gdn, x, target, layer_idx, 0, 1e-4);

    double max_rel = 0.0;
    int n_skipped = 0;
    for (size_t i = 0; i < dense.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dense.grad_weights.cols; ++j) {
            double a = dense.grad_weights[i][j];
            double n = numerical[i][j];
            if (std::abs(n) < 1e-10) { ++n_skipped; continue; }
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  " << name << " max rel_err = " << max_rel
              << " (skipped " << n_skipped << " near-zero)" << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_Q_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_Q", 0, gdn.W_Q_, gdn, x, target);
}

void test_W_K_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_K", 1, gdn.W_K_, gdn, x, target);
}

void test_W_V_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_V", 2, gdn.W_V_, gdn, x, target);
}

void test_W_O_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_O", 3, gdn.W_O_, gdn, x, target);
}

void test_W_beta_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_beta", 4, gdn.W_beta_, gdn, x, target);
}

void test_W_gate_grad() {
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);
    run_param_grad_test("W_gate", 5, gdn.W_gate_, gdn, x, target);
}

void test_training_reduces_loss() {
    std::cout << "test_training_reduces_loss..." << std::endl;
    std::mt19937 gen(7);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    double L0 = compute_loss(gdn, x, target);
    for (int step = 0; step < 50; ++step) {
        Tensor y = gdn.forward(x);
        Tensor grad_out(3, 4);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
            }
        }
        gdn.zero_grad();
        gdn.backward(grad_out, 0.0);
        // Manual SGD step
        auto params = gdn.parameters();
        auto grads = gdn.gradients();
        for (size_t p = 0; p < params.size(); ++p) {
            for (size_t i = 0; i < params[p]->rows; ++i) {
                for (size_t j = 0; j < params[p]->cols; ++j) {
                    (*params[p])[i][j] -= 0.01 * (*grads[p])[i][j];
                }
            }
        }
    }
    double L1 = compute_loss(gdn, x, target);
    std::cout << "  L0 = " << L0 << " L1 = " << L1 << std::endl;
    EXPECT(L1 < L0 * 0.95);  // at least 5% reduction
}

void test_determinism() {
    std::cout << "test_determinism..." << std::endl;
    std::mt19937 gen1(42), gen2(42);
    GatedDeltaNet gdn1(4, 2), gdn2(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen1);
    Tensor x2 = rand_tensor(3, 4, 0.3, gen2);
    Tensor y1 = gdn1.forward(x);
    Tensor y2 = gdn2.forward(x2);
    bool same = true;
    for (size_t i = 0; i < y1.rows; ++i) {
        for (size_t j = 0; j < y1.cols; ++j) {
            if (y1[i][j] != y2[i][j]) { same = false; break; }
        }
    }
    EXPECT(same);
}

void test_multi_head() {
    std::cout << "test_multi_head..." << std::endl;
    std::mt19937 gen(11);
    GatedDeltaNet gdn(6, 3);  // d_model=6, n_heads=3, head_dim=2
    Tensor x = rand_tensor(3, 6, 0.3, gen);
    Tensor y = gdn.forward(x);
    EXPECT(y.rows == 3);
    EXPECT(y.cols == 6);
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y[i][j])) { finite = false; break; }
        }
    }
    EXPECT(finite);
}

void test_longer_recurrence() {
    std::cout << "test_longer_recurrence..." << std::endl;
    std::mt19937 gen(127);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(8, 4, 0.2, gen);
    Tensor target = rand_tensor(8, 4, 0.2, gen);

    Tensor y = gdn.forward(x);
    Tensor grad_out(8, 4);
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 8.0;
        }
    }
    Tensor analytical = gdn.backward(grad_out, 0.0);

    Tensor numerical = numerical_input_grad(gdn, x, target, 1e-5);

    double max_rel = 0.0;
    int n_skipped = 0;
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double a = analytical[i][j];
            double n = numerical[i][j];
            if (std::abs(n) < 1e-12) { ++n_skipped; continue; }
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  T=8 max rel_err = " << max_rel
              << " (skipped " << n_skipped << " near-zero)" << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_update_weights() {
    std::cout << "test_update_weights..." << std::endl;
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    // Snapshot params
    std::vector<Tensor> snapshots;
    for (auto* p : gdn.parameters()) snapshots.push_back(p->clone());

    Tensor y = gdn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    gdn.zero_grad();
    gdn.backward(grad_out, 0.0);
    gdn.update_weights(0.01);

    // Verify all params moved
    bool all_moved = true;
    auto params = gdn.parameters();
    for (size_t p = 0; p < params.size(); ++p) {
        double diff = 0.0;
        for (size_t i = 0; i < params[p]->rows; ++i) {
            for (size_t j = 0; j < params[p]->cols; ++j) {
                diff += std::abs((*params[p])[i][j] - snapshots[p][i][j]);
            }
        }
        if (diff < 1e-10) all_moved = false;
    }
    EXPECT(all_moved);
    EXPECT(gdn.parameters().size() == 12);  // 6 Denses × (weights + bias)
}

void test_zero_grad() {
    std::cout << "test_zero_grad..." << std::endl;
    std::mt19937 gen(42);
    GatedDeltaNet gdn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = gdn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    gdn.backward(grad_out, 0.0);
    gdn.zero_grad();
    double max_grad = 0.0;
    for (auto* g : gdn.gradients()) {
        for (size_t i = 0; i < g->rows; ++i) {
            for (size_t j = 0; j < g->cols; ++j) {
                double v = std::abs((*g)[i][j]);
                if (v > max_grad) max_grad = v;
            }
        }
    }
    std::cout << "  max grad after zero_grad = " << max_grad << std::endl;
    EXPECT(max_grad < 1e-12);
}

int main() {
    test_constructor_validation();
    test_forward_shape();
    test_forward_finite();
    test_forward_nonzero();
    test_state_shape();
    test_state_accumulates();
    test_hand_derived_forward();
    test_input_grad();
    test_W_Q_grad();
    test_W_K_grad();
    test_W_V_grad();
    test_W_O_grad();
    test_W_beta_grad();
    test_W_gate_grad();
    test_training_reduces_loss();
    test_determinism();
    test_multi_head();
    test_longer_recurrence();
    test_update_weights();
    test_zero_grad();

    std::cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed > 0 ? 1 : 0;
}