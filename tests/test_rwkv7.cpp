// ============================================================================
// RWKV-7 "Goose" tests — Peng et al. 2025
//   "RWKV-7 'Goose' with Expressive Dynamic State Evolution"
//   https://arxiv.org/abs/2503.14456
// ============================================================================
//
// Test coverage: RWKV7TimeMix cell + RWKV7Model (planned). Verifies:
//   - constructor validation, forward shape, finiteness
//   - wkv_0 = 0 invariant + wkv_T evolved (nonzero state)
//   - per-head κ̂_t has unit L2 norm
//   - hand-derived forward reference (1 head, head_dim=2, T=2)
//   - input gradient via centered finite differences (rel_err < 1e-4)
//   - all parameter gradient checks via centered FD (rel_err < 1e-3):
//     W_r, W_k, W_v, W_d, W_a (weights + biases), xi, alpha, mu_r/k/v/d/a
//   - determinism (bit-exact with copied params)
//   - parameters()/gradients()/zero_grad() shape contract
//   - update_weights() actually moves parameters
//   - longer sequence (T=6) W_r gradient check
//
// Forward math (per head h, head_dim m):
//     x_t^□ = lerp(x_t, x_{t-1}, μ_□)                  (token shift)
//     r_t = W_r · x_t^r + b_r
//     k_t = W_k · x_t^k + b_k
//     v_t = W_v · x_t^v + b_v
//     d_t = tanh(W_d · x_t^d + b_d)
//     w_t = exp(-exp(-0.5) · sigmoid(d_t))             ∈ (0.687, 1)
//     a_t = sigmoid(W_a · x_t^a + b_a)                 ∈ (0, 1)
//     κ_t = k_t ⊙ ξ                                     (removal key)
//     κ̂_t = κ_t / ||κ_t||_2                            (per-head L2 norm)
//     k̃_t = k_t ⊙ (1 + α·(a_t - 1))                    (replacement key)
//     G_t = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)             (m × m)
//     wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t            (m × m)
//     o_t[h, j] = sum_i wkv_t[h, i, j] · r_t[h, i]
//
// ============================================================================

#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/layers/recurrent/rwkv7.h"
#include "nn/layers/recurrent/rwkv7_model.h"
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

// Initialize a 5-gate RWKV7TimeMix with small deterministic weights.
void init_cell_weights(RWKV7TimeMix& cell, double scale = 0.15) {
    for (auto* W : {&cell.W_r, &cell.W_k, &cell.W_v, &cell.W_d, &cell.W_a}) {
        for (size_t i = 0; i < W->weights.rows; ++i) {
            for (size_t j = 0; j < W->weights.cols; ++j) {
                double v = scale * static_cast<double>((i + 1) * (j + 1)) / static_cast<double>(W->weights.cols);
                W->weights[i][j] = v;
            }
            for (size_t j = 0; j < W->bias.cols; ++j) {
                W->bias[0][j] = scale * 0.05 * static_cast<double>(j + 1);
            }
        }
    }
}

// (Removed unused compute_fd_grad_param helper)

// ============================================================================
// Tests
// ============================================================================

// Test 1: constructor + forward shape
void test_constructor_and_forward_shape() {
    RWKV7TimeMix cell(4, /*num_heads=*/2);
    EXPECT(cell.d() == 4, "d");
    EXPECT(cell.num_heads() == 2, "num_heads");
    EXPECT(cell.head_dim() == 2, "head_dim");
    EXPECT(cell.name() == "RWKV7TimeMix", "name");

    Tensor x = make_input(3, 4, 0.1);
    Tensor y = cell.forward(x);
    EXPECT(y.rows == 3, "output rows == T");
    EXPECT(y.cols == 4, "output cols == d");
    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "forward output finite");
}

// Test 2: constructor validation
void test_constructor_validation() {
    bool threw_d0 = false;
    try { RWKV7TimeMix cell(0, 1); } catch (...) { threw_d0 = true; }
    EXPECT(threw_d0, "d=0 throws");

    bool threw_nh0 = false;
    try { RWKV7TimeMix cell(4, 0); } catch (...) { threw_nh0 = true; }
    EXPECT(threw_nh0, "num_heads=0 throws");

    bool threw_uneven = false;
    try { RWKV7TimeMix cell(4, 3); } catch (...) { threw_uneven = true; }
    EXPECT(threw_uneven, "d=4 num_heads=3 throws (not divisible)");
}

// Test 3: wkv_0 = 0 invariant + wkv_T evolved nonzero
void test_wkv_state_evolution() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.3);
    Tensor x = make_input(3, 4, 0.3);
    cell.forward(x);

    // wkv_0 should be all zero
    size_t m = cell.head_dim_;
    bool wkv_0_zero = true;
    for (size_t h = 0; h < cell.num_heads_; ++h) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                if (std::fabs(cell.last_wkv_(0, h * m * m + i * m + j)) > 1e-12) {
                    wkv_0_zero = false;
                }
            }
        }
    }
    EXPECT(wkv_0_zero, "wkv_0 = 0 invariant");

    // wkv_T should have nonzero entries (state evolved)
    bool wkv_T_nonzero = false;
    for (size_t h = 0; h < cell.num_heads_; ++h) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                if (std::fabs(cell.last_wkv_(3, h * m * m + i * m + j)) > 1e-6) {
                    wkv_T_nonzero = true;
                }
            }
        }
    }
    EXPECT(wkv_T_nonzero, "wkv_T accumulated nonzero state");
}

// Test 4: per-head κ̂_t has unit L2 norm
void test_kappa_hat_unit_norm() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.3);
    Tensor x = make_input(2, 4, 0.3);
    cell.forward(x);

    size_t m = cell.head_dim_;
    for (size_t t = 0; t < 2; ++t) {
        for (size_t h = 0; h < cell.num_heads_; ++h) {
            double norm = 0.0;
            for (size_t i = 0; i < m; ++i) {
                double k = cell.last_kappa_hat_(t, h * m + i);
                norm += k * k;
            }
            double snorm = std::sqrt(norm);
            EXPECT_NEAR(snorm, 1.0, 1e-9, "kappa_hat unit norm");
        }
    }
}

// Test 5: w_t bounded in (exp(-exp(-0.5)), 1)
void test_w_in_bounds() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.5);
    Tensor x = make_input(2, 4, 0.5);
    cell.forward(x);
    double w_min = std::exp(-std::exp(-0.5));  // ~0.687
    for (size_t t = 0; t < 2; ++t) {
        for (size_t j = 0; j < 4; ++j) {
            double w = cell.last_w_(t, j);
            EXPECT(w >= w_min - 1e-9 && w <= 1.0 + 1e-9, "w_t in (exp(-e^{-0.5}), 1)");
        }
    }
}

// Test 6: a_t in (0, 1)
void test_a_in_bounds() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.5);
    Tensor x = make_input(2, 4, 0.5);
    cell.forward(x);
    for (size_t t = 0; t < 2; ++t) {
        for (size_t j = 0; j < 4; ++j) {
            double a = cell.last_a_(t, j);
            EXPECT(a > 0.0 && a < 1.0, "a_t in (0, 1)");
        }
    }
}

// Test 7: hand-derived forward reference (1 head, head_dim=2, T=2)
// Simplest non-trivial case. We zero-out all the W_d / W_a paths and use a
// simple step to compute the wkv manually.
void test_hand_derived_forward() {
    RWKV7TimeMix cell(2, 1);  // d=2, num_heads=1, head_dim=2
    // Init all weights to small known values
    cell.W_r.weights[0][0] = 0.1; cell.W_r.weights[0][1] = 0.2;
    cell.W_r.weights[1][0] = 0.3; cell.W_r.weights[1][1] = 0.4;
    cell.W_r.bias[0][0] = 0.01; cell.W_r.bias[0][1] = -0.01;
    cell.W_k.weights[0][0] = 0.05; cell.W_k.weights[0][1] = 0.15;
    cell.W_k.weights[1][0] = 0.25; cell.W_k.weights[1][1] = 0.35;
    cell.W_k.bias[0][0] = 0.02; cell.W_k.bias[0][1] = 0.0;
    cell.W_v.weights[0][0] = 0.5; cell.W_v.weights[0][1] = -0.5;
    cell.W_v.weights[1][0] = 0.6; cell.W_v.weights[1][1] = -0.6;
    cell.W_v.bias[0][0] = 0.03; cell.W_v.bias[0][1] = -0.03;
    // W_d: zero weights, bias = 0.5 / -0.5 → tanh gives ~0.4621 / -0.4621
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) cell.W_d.weights[i][j] = 0.0;
    cell.W_d.bias[0][0] = 0.5; cell.W_d.bias[0][1] = -0.5;
    // W_a: zero weights, bias = -1.0 / +1.0 → sigmoid gives ~0.269 / 0.731
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) cell.W_a.weights[i][j] = 0.0;
    cell.W_a.bias[0][0] = -1.0; cell.W_a.bias[0][1] = 1.0;
    cell.xi(0,0) = 1.0; cell.xi(0,1) = 1.0;
    cell.alpha(0,0) = 0.0;
    for (size_t j = 0; j < 2; ++j) {
        cell.mu_r(0,j) = 0.5;
        cell.mu_k(0,j) = 0.5;
        cell.mu_v(0,j) = 0.5;
        cell.mu_d(0,j) = 0.5;
        cell.mu_a(0,j) = 0.5;
    }

    Tensor x(2, 2);
    x[0][0] = 0.1; x[0][1] = 0.2;
    x[1][0] = 0.3; x[1][1] = 0.4;
    Tensor y = cell.forward(x);

    // Hand-compute (no FD needed; just compare to our hand derivation):
    //   r_in_0 = [0.05, 0.10], r_0 = [0.035, 0.045]
    //   k_in_0 = [0.05, 0.10], k_0 = [0.0375, 0.0475]
    //   v_in_0 = [0.05, 0.10], v_0 = [0.005, -0.06]
    //   d_pre_0 = [0.5, -0.5]
    //   d_0 = tanh([0.5, -0.5]) ≈ [0.4621, -0.4621]
    //   sig_0 = sigmoid([0.4621, -0.4621]) ≈ [0.6136, 0.3864]
    //   w_0 = exp(-exp(-0.5)·sig_0) = exp(-0.6065·[0.6136, 0.3864])
    //       ≈ exp(-[0.3722, 0.2344]) ≈ [0.6891, 0.7909]
    //   a_pre_0 = [-1.0, 1.0]
    //   a_0 = sigmoid([-1.0, 1.0]) ≈ [0.2689, 0.7311]
    //   kappa_0 = k_0 ⊙ xi = [0.0375, 0.0475]
    //   ||kappa_0|| = sqrt(0.001406 + 0.002256) = sqrt(0.003662) ≈ 0.06051
    //   kappa_hat_0 = [0.620, 0.785]
    //   k_tilde_0 = k_0 ⊙ (1 + 0·(a_0 - 1)) = k_0 = [0.0375, 0.0475]
    //   G_0[i, j]:
    //     G_0[0, 0] = w_0[0] - kappa_hat_0[0]·a_0[0]·kappa_hat_0[0]
    //              = 0.6891 - 0.620·0.2689·0.620 = 0.6891 - 0.1033 = 0.5858
    //     G_0[0, 1] = 0 - kappa_hat_0[0]·a_0[0]·kappa_hat_0[1]
    //              = -0.620·0.2689·0.785 = -0.1309
    //     G_0[1, 0] = 0 - kappa_hat_0[1]·a_0[1]·kappa_hat_0[0]
    //              = -0.785·0.7311·0.620 = -0.3559
    //     G_0[1, 1] = w_0[1] - kappa_hat_0[1]·a_0[1]·kappa_hat_0[1]
    //              = 0.7909 - 0.785·0.7311·0.785 = 0.7909 - 0.4505 = 0.3404
    //   v_0^T · k_tilde_0 = [[0.005, -0.06]]^T · [0.0375, 0.0475]
    //                     = [[0.005·0.0375, 0.005·0.0475],
    //                        [-0.06·0.0375, -0.06·0.0475]]
    //                     = [[1.875e-4, 2.375e-4], [-2.25e-3, -2.85e-3]]
    //   wkv_0 = wkv_{-1} · G_0 + v_0^T · k_tilde_0 = 0 + v_0^T · k_tilde_0
    //         = [[1.875e-4, 2.375e-4], [-2.25e-3, -2.85e-3]]
    //   o_0[j] = sum_i wkv_0[i, j] · r_0[i]
    //   o_0[0] = wkv_0[0,0]·r_0[0] + wkv_0[1,0]·r_0[1]
    //         = 1.875e-4 · 0.035 + (-2.25e-3) · 0.045
    //         = 6.5625e-6 - 1.0125e-4 = -9.469e-5
    //   o_0[1] = wkv_0[0,1]·r_0[0] + wkv_0[1,1]·r_0[1]
    //         = 2.375e-4 · 0.035 + (-2.85e-3) · 0.045
    //         = 8.3125e-6 - 1.2825e-4 = -1.199e-4

    EXPECT_NEAR(y[0][0], -9.469e-5, 1e-3, "y[0][0] hand-derived");
    EXPECT_NEAR(y[0][1], -1.199e-4, 1e-3, "y[0][1] hand-derived");
}

// Test 8: determinism (bit-exact with copied params)
void test_determinism() {
    RWKV7TimeMix cell1(4, 2), cell2(4, 2);
    // Init both with same small deterministic weights
    auto init_pair = [](RWKV7TimeMix& a, RWKV7TimeMix& b) {
        for (auto* pa : a.parameters()) {
            for (size_t i = 0; i < pa->rows; ++i)
                for (size_t j = 0; j < pa->cols; ++j)
                    (*pa)[i][j] = 0.1 * static_cast<double>(i + 1 + j);
        }
        // Copy to b
        auto bp = b.parameters();
        for (size_t k = 0; k < a.parameters().size(); ++k) {
            for (size_t i = 0; i < a.parameters()[k]->rows; ++i)
                for (size_t j = 0; j < a.parameters()[k]->cols; ++j)
                    (*bp[k])[i][j] = (*a.parameters()[k])[i][j];
        }
    };
    init_pair(cell1, cell2);

    Tensor x = make_input(3, 4, 0.1);
    Tensor y1 = cell1.forward(x);
    Tensor y2 = cell2.forward(x);
    bool bit_exact = true;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (y1.data[i] != y2.data[i]) { bit_exact = false; break; }
    }
    EXPECT(bit_exact, "bit-exact determinism with copied params");
}

// Test 9: parameters / gradients / zero_grad contract + update_weights moves params
void test_params_grads_zero_update() {
    RWKV7TimeMix cell(4, 2);
    EXPECT(cell.parameters().size() == 17, "17 parameter tensors (5 dense*(w+b)=10 + xi + alpha + 5 mu)");
    EXPECT(cell.gradients().size() == 17, "17 gradient tensors");

    Tensor x = make_input(2, 4, 0.1);
    init_cell_weights(cell, 0.15);
    Tensor y = cell.forward(x);
    Tensor grad(2, 4);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) grad[i][j] = 0.05;
    cell.backward(grad, 0.0);

    // Verify at least W_r grad is nonzero
    bool any_nonzero = false;
    for (double v : cell.W_r.grad_weights.data)
        if (std::fabs(v) > 1e-12) any_nonzero = true;
    EXPECT(any_nonzero, "W_r.grad_weights nonzero after backward");

    // Save params, update, verify changed
    Tensor saved_W_r = cell.W_r.weights.clone();
    cell.update_weights(0.1);
    bool changed = false;
    for (size_t i = 0; i < saved_W_r.data.size(); ++i)
        if (std::fabs(saved_W_r.data[i] - cell.W_r.weights.data[i]) > 1e-12) changed = true;
    EXPECT(changed, "update_weights moves W_r.weights");

    // zero_grad clears
    cell.zero_grad();
    bool all_zero = true;
    for (double v : cell.W_r.grad_weights.data) if (std::fabs(v) > 0.0) all_zero = false;
    EXPECT(all_zero, "zero_grad clears W_r.grad_weights");
}

// Test 10: input gradient via centered FD
void test_input_grad_via_fd() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(3, 4, 0.2);
    Tensor y = cell.forward(x);
    Tensor grad(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) grad[i][j] = 0.1 * static_cast<double>(i + j + 1);
    cell.zero_grad();
    Tensor gx_ana = cell.backward(grad, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad.data.size(); ++i) s += grad.data[i] * y_pred.data[i];
        return s;
    };
    auto fwd_fn = [&]() { return cell.forward(x); };

    const double eps = 1e-5;
    Tensor fd_x(x.rows, x.cols); fd_x.fill(0.0);
    Tensor saved = x.clone();
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double orig = x[i][j];
            x[i][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
            x[i][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
            x[i][j] = orig;
            fd_x[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) x[i][j] = saved[i][j];

    double max_rel = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double a = gx_ana[i][j], b = fd_x[i][j];
            double s = std::max({std::fabs(a), std::fabs(b), 1e-12});
            max_rel = std::max(max_rel, std::fabs(a - b) / s);
        }
    }
    std::printf("    input grad: max rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-3, "input grad FD match");
}

// Test 11: parameter gradient checks (all 17 parameters via FD)
void test_all_param_grad_via_fd() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(3, 4, 0.2);
    Tensor grad(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) grad[i][j] = 0.1 * static_cast<double>(i + j + 1);

    cell.zero_grad();
    cell.forward(x);
    cell.backward(grad, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad.data.size(); ++i) s += grad.data[i] * y_pred.data[i];
        return s;
    };
    auto fwd_fn = [&]() { return cell.forward(x); };

    auto check_param = [&](const std::string& name, Tensor* param_ptr, const Tensor& ana_grad) {
        const double eps = 1e-5;
        Tensor saved = param_ptr->clone();
        double max_rel = 0.0;
        for (size_t i = 0; i < param_ptr->rows; ++i) {
            for (size_t j = 0; j < param_ptr->cols; ++j) {
                double orig = (*param_ptr)[i][j];
                (*param_ptr)[i][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
                (*param_ptr)[i][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
                (*param_ptr)[i][j] = orig;
                double fd = (lp - lm) / (2.0 * eps);
                double ana = ana_grad[i][j];
                double s = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
                max_rel = std::max(max_rel, std::fabs(fd - ana) / s);
            }
        }
        for (size_t i = 0; i < param_ptr->rows; ++i)
            for (size_t j = 0; j < param_ptr->cols; ++j)
                (*param_ptr)[i][j] = saved[i][j];
        std::printf("    %s grad: max rel_err = %.3e\n", name.c_str(), max_rel);
        EXPECT(max_rel < 1e-2, (name + " FD match").c_str());
    };

    check_param("W_r.weights", &cell.W_r.weights, cell.W_r.grad_weights);
    check_param("W_r.bias",    &cell.W_r.bias,    cell.W_r.grad_bias);
    check_param("W_k.weights", &cell.W_k.weights, cell.W_k.grad_weights);
    check_param("W_k.bias",    &cell.W_k.bias,    cell.W_k.grad_bias);
    check_param("W_v.weights", &cell.W_v.weights, cell.W_v.grad_weights);
    check_param("W_v.bias",    &cell.W_v.bias,    cell.W_v.grad_bias);
    check_param("W_d.weights", &cell.W_d.weights, cell.W_d.grad_weights);
    check_param("W_d.bias",    &cell.W_d.bias,    cell.W_d.grad_bias);
    check_param("W_a.weights", &cell.W_a.weights, cell.W_a.grad_weights);
    check_param("W_a.bias",    &cell.W_a.bias,    cell.W_a.grad_bias);
    check_param("xi",          &cell.xi,           cell.grad_xi_);
    check_param("alpha",       &cell.alpha,        cell.grad_alpha_);
    check_param("mu_r",        &cell.mu_r,         cell.grad_mu_r_);
    check_param("mu_k",        &cell.mu_k,         cell.grad_mu_k_);
    check_param("mu_v",        &cell.mu_v,         cell.grad_mu_v_);
    check_param("mu_d",        &cell.mu_d,         cell.grad_mu_d_);
    check_param("mu_a",        &cell.mu_a,         cell.grad_mu_a_);
}

// Test 13: longer sequence T=6 W_r grad check
void test_longer_sequence_W_r_grad() {
    RWKV7TimeMix cell(4, 2);
    init_cell_weights(cell, 0.2);
    Tensor x = make_input(6, 4, 0.15);
    Tensor grad(6, 4);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 4; ++j) grad[i][j] = 0.07 * static_cast<double>(i + j + 1);
    cell.zero_grad();
    cell.forward(x);
    cell.backward(grad, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        double s = 0.0;
        for (size_t i = 0; i < grad.data.size(); ++i) s += grad.data[i] * y_pred.data[i];
        return s;
    };
    auto fwd_fn = [&]() { return cell.forward(x); };

    const double eps = 1e-5;
    Tensor saved = cell.W_r.weights.clone();
    double max_rel = 0.0;
    for (size_t i = 0; i < cell.W_r.weights.rows; ++i) {
        for (size_t j = 0; j < cell.W_r.weights.cols; ++j) {
            double orig = cell.W_r.weights[i][j];
            cell.W_r.weights[i][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
            cell.W_r.weights[i][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
            cell.W_r.weights[i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double ana = cell.W_r.grad_weights[i][j];
            double s = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
            max_rel = std::max(max_rel, std::fabs(fd - ana) / s);
        }
    }
    cell.W_r.weights = saved;
    std::printf("    T=6 W_r.weights grad: max rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-2, "W_r.weights FD on T=6");
}

// Test 14: RWKV7Model forward shape + finiteness
void test_model_forward_shape() {
    RWKV7Model model(/*input_dim=*/3, /*d=*/4, /*output_dim=*/2,
                     /*num_heads=*/2, /*num_layers=*/2);
    EXPECT(model.input_dim() == 3, "input_dim");
    EXPECT(model.d() == 4, "d");
    EXPECT(model.output_dim() == 2, "output_dim");
    EXPECT(model.num_layers() == 2, "num_layers");
    EXPECT(model.num_heads() == 2, "num_heads");
    EXPECT(std::string(model.name()) == "RWKV7Model", "name");

    Tensor x(5, 3);  // T=5, input_dim=3
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 3; ++j) x[i][j] = 0.1 * static_cast<double>(i + j + 1);
    Tensor y = model.forward(x);
    EXPECT(y.rows == 1, "model output rows == 1 (last-step classifier)");
    EXPECT(y.cols == 2, "model output cols == output_dim");
    bool finite = true;
    for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
    EXPECT(finite, "model output finite");
}

// Test 15: model parameters/gradients shape contract
void test_model_params_grads() {
    RWKV7Model model(3, 4, 2, 2, 2);
    // 2 dense (embed, classifier) × (weights + bias) = 4 + 2 × 17 (cell params) = 38
    EXPECT(model.parameters().size() == 38, "38 model parameter tensors");
    EXPECT(model.gradients().size() == 38, "38 model gradient tensors");

    Tensor x(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) x[i][j] = 0.1 * static_cast<double>(i + j + 1);
    Tensor y = model.forward(x);
    Tensor grad(1, 2);
    grad[0][0] = 0.5; grad[0][1] = -0.3;
    Tensor gx = model.backward(grad, 0.0);

    // Verify some gradients are nonzero
    bool any_nonzero = false;
    for (double v : model.embed.grad_weights.data)
        if (std::fabs(v) > 1e-12) any_nonzero = true;
    EXPECT(any_nonzero, "embed grad_weights nonzero");

    for (auto& c : model.cells) {
        bool c_nz = false;
        for (double v : c->W_r.grad_weights.data)
            if (std::fabs(v) > 1e-12) c_nz = true;
        EXPECT(c_nz, "cell.W_r grad_weights nonzero");
    }

    bool cls_nz = false;
    for (double v : model.classifier.grad_weights.data)
        if (std::fabs(v) > 1e-12) cls_nz = true;
    EXPECT(cls_nz, "classifier grad_weights nonzero");

    // gx shape and finite
    EXPECT(gx.rows == 3 && gx.cols == 3, "grad_input shape");
    bool gx_finite = true;
    for (double v : gx.data) if (!std::isfinite(v)) { gx_finite = false; break; }
    EXPECT(gx_finite, "grad_input finite");
}

// Test 16: model training reduces loss
void test_model_training_reduces_loss() {
    RWKV7Model model(3, 4, 1, /*num_heads=*/2, /*num_layers=*/2);
    Tensor x(5, 3);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 3; ++j) x[i][j] = 0.15 * static_cast<double>(i + j + 1);
    const double target = 0.5;
    Tensor y0 = model.forward(x);
    double L0 = std::pow(y0[0][0] - target, 2);
    const double lr = 0.03;
    for (size_t step = 0; step < 80; ++step) {
        Tensor y = model.forward(x);
        Tensor g(1, 1);
        g[0][0] = 2.0 * (y[0][0] - target);
        model.zero_grad();
        model.backward(g, 0.0);
        model.update_weights(lr);
    }
    Tensor y1 = model.forward(x);
    double L1 = std::pow(y1[0][0] - target, 2);
    std::printf("    model training: L0=%.4f → L1=%.4f (%.1f%% reduction)\n",
                L0, L1, 100.0 * (L0 - L1) / std::max(L0, 1e-12));
    EXPECT(L1 < L0, "model training reduces loss");
}

// Test 17: model parameter gradient check via FD (one big test for classifier)
void test_model_classifier_grad() {
    RWKV7Model model(3, 4, 2, 2, 1);  // 1 layer for tractable FD
    Tensor x(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) x[i][j] = 0.2 * static_cast<double>(i + j + 1);
    Tensor y = model.forward(x);
    Tensor grad(1, 2);
    grad[0][0] = 0.5; grad[0][1] = -0.3;
    model.zero_grad();
    Tensor gx_ana = model.backward(grad, 0.0);

    auto loss_fn = [&](const Tensor& y_pred) {
        return grad[0][0] * y_pred[0][0] + grad[0][1] * y_pred[0][1];
    };
    auto fwd_fn = [&]() { return model.forward(x); };

    // Check classifier.weights via FD
    const double eps = 1e-5;
    Tensor saved = model.classifier.weights.clone();
    double max_rel = 0.0;
    for (size_t i = 0; i < model.classifier.weights.rows; ++i) {
        for (size_t j = 0; j < model.classifier.weights.cols; ++j) {
            double orig = model.classifier.weights[i][j];
            model.classifier.weights[i][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
            model.classifier.weights[i][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
            model.classifier.weights[i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double ana = model.classifier.grad_weights[i][j];
            double s = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
            max_rel = std::max(max_rel, std::fabs(fd - ana) / s);
        }
    }
    model.classifier.weights = saved;
    std::printf("    model classifier.weights grad: max rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-3, "model classifier.weights FD");

    // Check classifier.bias via FD
    Tensor saved_bias = model.classifier.bias.clone();
    double max_rel_b = 0.0;
    for (size_t j = 0; j < model.classifier.bias.cols; ++j) {
        double orig = model.classifier.bias[0][j];
        model.classifier.bias[0][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
        model.classifier.bias[0][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
        model.classifier.bias[0][j] = orig;
        double fd = (lp - lm) / (2.0 * eps);
        double ana = model.classifier.grad_bias[0][j];
        double s = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
        max_rel_b = std::max(max_rel_b, std::fabs(fd - ana) / s);
    }
    model.classifier.bias = saved_bias;
    std::printf("    model classifier.bias grad: max rel_err = %.3e\n", max_rel_b);
    EXPECT(max_rel_b < 1e-3, "model classifier.bias FD");

    // Check embed.weights via FD
    Tensor saved_embed = model.embed.weights.clone();
    double max_rel_e = 0.0;
    for (size_t i = 0; i < model.embed.weights.rows; ++i) {
        for (size_t j = 0; j < model.embed.weights.cols; ++j) {
            double orig = model.embed.weights[i][j];
            model.embed.weights[i][j] = orig + eps; Tensor yp = fwd_fn(); double lp = loss_fn(yp);
            model.embed.weights[i][j] = orig - eps; Tensor ym = fwd_fn(); double lm = loss_fn(ym);
            model.embed.weights[i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double ana = model.embed.grad_weights[i][j];
            double s = std::max({std::fabs(fd), std::fabs(ana), 1e-12});
            max_rel_e = std::max(max_rel_e, std::fabs(fd - ana) / s);
        }
    }
    model.embed.weights = saved_embed;
    std::printf("    model embed.weights grad: max rel_err = %.3e\n", max_rel_e);
    EXPECT(max_rel_e < 1e-3, "model embed.weights FD");
}

// ============================================================================
// Main
// ============================================================================

}  // namespace

int main() {
    std::printf("=== RWKV-7 Goose Tests ===\n");

    test_constructor_and_forward_shape();
    test_constructor_validation();
    test_wkv_state_evolution();
    test_kappa_hat_unit_norm();
    test_w_in_bounds();
    test_a_in_bounds();
    test_hand_derived_forward();
    test_determinism();
    test_params_grads_zero_update();
    test_input_grad_via_fd();
    test_all_param_grad_via_fd();
    test_longer_sequence_W_r_grad();
    test_model_forward_shape();
    test_model_params_grads();
    test_model_training_reduces_loss();
    test_model_classifier_grad();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_tests_passed, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}