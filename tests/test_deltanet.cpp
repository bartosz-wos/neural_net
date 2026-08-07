// ==========================================================================
// tests/test_deltanet.cpp
//
// Focused tests for DeltaNet (Yang et al. 2024, "Linear Attention with the
// Delta Rule"). Tests:
//   1. Constructor validation
//   2. Forward shape (T, d_model) -> (T, d_model)
//   3. Forward output is finite
//   4. State cache shape (n_heads, head_dim * head_dim)
//   5. State accumulation (norm > 0)
//   6. Input gradient check (analytical vs centered finite difference)
//   7. W_Q gradient check
//   8. W_K gradient check
//   9. W_V gradient check
//  10. W_O gradient check
//  11. W_beta gradient check
//  12. Training reduces loss (SGD on MSE loss)
//  13. Determinism (bit-identical with copied params)
//  14. Multi-head (n_heads=3, head_dim=2) forward shape
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

// Helper: compute MSE loss
static double compute_loss(DeltaNet& dn, const Tensor& x, const Tensor& target) {
    Tensor y = dn.forward(x);
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
static Tensor numerical_input_grad(DeltaNet& dn, const Tensor& x, const Tensor& target, double eps) {
    size_t T = x.rows, D = x.cols;
    Tensor grad(T, D);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            Tensor x_plus = x.clone();
            Tensor x_minus = x.clone();
            x_plus[t][d] += eps;
            x_minus[t][d] -= eps;
            double L_plus = compute_loss(dn, x_plus, target);
            double L_minus = compute_loss(dn, x_minus, target);
            grad[t][d] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

// Helper: numerical parameter gradient via centered finite differences
// layer_idx: 0 = W_Q, 1 = W_K, 2 = W_V, 3 = W_O, 4 = W_beta
// index_in_layer: 0 = weights, 1 = bias
static Tensor numerical_param_grad(DeltaNet& dn, const Tensor& x, const Tensor& target,
                                   size_t layer_idx, size_t index_in_layer, double eps) {
    Tensor* w;
    if (layer_idx == 0) w = (index_in_layer == 0) ? &dn.W_Q_.weights : &dn.W_Q_.bias;
    else if (layer_idx == 1) w = (index_in_layer == 0) ? &dn.W_K_.weights : &dn.W_K_.bias;
    else if (layer_idx == 2) w = (index_in_layer == 0) ? &dn.W_V_.weights : &dn.W_V_.bias;
    else if (layer_idx == 3) w = (index_in_layer == 0) ? &dn.W_O_.weights : &dn.W_O_.bias;
    else if (layer_idx == 4) w = (index_in_layer == 0) ? &dn.W_beta_.weights : &dn.W_beta_.bias;
    else throw std::invalid_argument("invalid layer_idx");

    Tensor grad(w->rows, w->cols);
    for (size_t i = 0; i < w->rows; ++i) {
        for (size_t j = 0; j < w->cols; ++j) {
            double orig = (*w)[i][j];
            (*w)[i][j] = orig + eps;
            double L_plus = compute_loss(dn, x, target);
            (*w)[i][j] = orig - eps;
            double L_minus = compute_loss(dn, x, target);
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
    try { DeltaNet bad(0, 2); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    try { DeltaNet bad(4, 0); } catch (std::invalid_argument&) { caught = true; }
    EXPECT(caught);
    caught = false;
    try { DeltaNet bad(4, 3); } catch (std::invalid_argument&) { caught = true; }  // 4 not divisible by 3
    EXPECT(caught);
}

void test_forward_shape() {
    std::cout << "test_forward_shape..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = dn.forward(x);
    EXPECT(y.rows == 3);
    EXPECT(y.cols == 4);
}

void test_forward_finite() {
    std::cout << "test_forward_finite..." << std::endl;
    std::mt19937 gen(7);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    Tensor y = dn.forward(x);
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i) {
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y[i][j])) { finite = false; break; }
        }
    }
    EXPECT(finite);
}

void test_state_shape() {
    std::cout << "test_state_shape..." << std::endl;
    std::mt19937 gen(123);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    dn.forward(x);
    Tensor s = dn.last_state();
    EXPECT(s.rows == 2);  // n_heads
    EXPECT(s.cols == 4);  // head_dim * head_dim
}

void test_state_accumulates() {
    std::cout << "test_state_accumulates..." << std::endl;
    std::mt19937 gen(99);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.5, gen);
    dn.forward(x);
    Tensor s = dn.last_state();
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

void test_input_grad() {
    std::cout << "test_input_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    // Forward + analytical backward
    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    Tensor analytical = dn.backward(grad_out, 0.0);

    // Numerical
    Tensor numerical = numerical_input_grad(dn, x, target, 1e-5);

    double max_rel = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double a = analytical[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_Q_grad() {
    std::cout << "test_W_Q_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.zero_grad();
    dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(dn, x, target, 0, 0, 1e-4);

    double max_rel = 0.0;
    for (size_t i = 0; i < dn.W_Q_.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dn.W_Q_.grad_weights.cols; ++j) {
            double a = dn.W_Q_.grad_weights[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  W_Q max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_K_grad() {
    std::cout << "test_W_K_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.zero_grad();
    dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(dn, x, target, 1, 0, 1e-4);

    double max_rel = 0.0;
    for (size_t i = 0; i < dn.W_K_.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dn.W_K_.grad_weights.cols; ++j) {
            double a = dn.W_K_.grad_weights[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;

        }
    }
    std::cout << "  W_K max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_V_grad() {
    std::cout << "test_W_V_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.zero_grad();
    dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(dn, x, target, 2, 0, 1e-4);

    double max_rel = 0.0;
    for (size_t i = 0; i < dn.W_V_.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dn.W_V_.grad_weights.cols; ++j) {
            double a = dn.W_V_.grad_weights[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  W_V max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_O_grad() {
    std::cout << "test_W_O_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.zero_grad();
    dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(dn, x, target, 3, 0, 1e-4);

    double max_rel = 0.0;
    for (size_t i = 0; i < dn.W_O_.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dn.W_O_.grad_weights.cols; ++j) {
            double a = dn.W_O_.grad_weights[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  W_O max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_W_beta_grad() {
    std::cout << "test_W_beta_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.zero_grad();
    dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_param_grad(dn, x, target, 4, 0, 1e-4);

    double max_rel = 0.0;
    int n_skipped = 0;
    for (size_t i = 0; i < dn.W_beta_.grad_weights.rows; ++i) {
        for (size_t j = 0; j < dn.W_beta_.grad_weights.cols; ++j) {
            double a = dn.W_beta_.grad_weights[i][j];
            double n = numerical[i][j];
            // beta can be small (sigmoid) → gradients may be tiny, so use a tighter absolute threshold
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-10);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
            if (std::abs(n) < 1e-12) ++n_skipped;
        }
    }
    std::cout << "  W_beta max rel_err = " << max_rel << " (skipped " << n_skipped << " near-zero cells)" << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_training_reduces_loss() {
    std::cout << "test_training_reduces_loss..." << std::endl;
    std::mt19937 gen(7);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    double L0 = compute_loss(dn, x, target);
    for (int step = 0; step < 30; ++step) {
        Tensor y = dn.forward(x);
        Tensor grad_out(3, 4);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
            }
        }
        dn.zero_grad();
        dn.backward(grad_out, 0.0);
        // Manual SGD step
        auto params = dn.parameters();
        auto grads = dn.gradients();
        for (size_t p = 0; p < params.size(); ++p) {
            for (size_t i = 0; i < params[p]->rows; ++i) {
                for (size_t j = 0; j < params[p]->cols; ++j) {
                    (*params[p])[i][j] -= 0.01 * (*grads[p])[i][j];
                }
            }
        }
    }
    double L1 = compute_loss(dn, x, target);
    std::cout << "  L0 = " << L0 << " L1 = " << L1 << std::endl;
    EXPECT(L1 < L0);
}

void test_determinism() {
    std::cout << "test_determinism..." << std::endl;
    std::mt19937 gen1(42), gen2(42);
    DeltaNet dn1(4, 2), dn2(4, 2);
    // Both have same init (deterministic given same seed: 42 in Dense::init_weights)
    Tensor x = rand_tensor(3, 4, 0.3, gen1);
    Tensor x2 = rand_tensor(3, 4, 0.3, gen2);
    // Expect x == x2 since same seed
    Tensor y1 = dn1.forward(x);
    Tensor y2 = dn2.forward(x2);
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
    DeltaNet dn(8, 4);  // d_model=8, n_heads=4, head_dim=2
    Tensor x = rand_tensor(3, 8, 0.3, gen);
    Tensor y = dn.forward(x);
    EXPECT(y.rows == 3);
    EXPECT(y.cols == 8);
}

void test_longer_sequence() {
    std::cout << "test_longer_sequence..." << std::endl;
    std::mt19937 gen(33);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(5, 4, 0.2, gen);
    Tensor y = dn.forward(x);
    EXPECT(y.rows == 5);
    EXPECT(y.cols == 4);
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
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(10, 4, 0.2, gen);
    Tensor target = rand_tensor(10, 4, 0.2, gen);

    // Forward + analytical backward
    Tensor y = dn.forward(x);
    Tensor grad_out(10, 4);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 10.0;
        }
    }
    Tensor analytical = dn.backward(grad_out, 0.0);

    Tensor numerical = numerical_input_grad(dn, x, target, 1e-5);

    double max_rel = 0.0;
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double a = analytical[i][j];
            double n = numerical[i][j];
            double denom = std::max(std::max(std::abs(a), std::abs(n)), 1e-12);
            double rel = std::abs(a - n) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    std::cout << "  T=10 max rel_err = " << max_rel << std::endl;
    EXPECT(max_rel < 1e-3);
}

void test_zero_grad() {
    std::cout << "test_zero_grad..." << std::endl;
    std::mt19937 gen(42);
    DeltaNet dn(4, 2);
    Tensor x = rand_tensor(3, 4, 0.3, gen);
    Tensor target = rand_tensor(3, 4, 0.3, gen);

    Tensor y = dn.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
        }
    }
    dn.backward(grad_out, 0.0);
    // Now zero
    dn.zero_grad();
    double max_grad = 0.0;
    for (auto* g : dn.gradients()) {
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
    test_state_shape();
    test_state_accumulates();
    test_input_grad();
    test_W_Q_grad();
    test_W_K_grad();
    test_W_V_grad();
    test_W_O_grad();
    test_W_beta_grad();
    test_training_reduces_loss();
    test_determinism();
    test_multi_head();
    test_longer_sequence();
    test_longer_recurrence();
    test_zero_grad();

    std::cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
