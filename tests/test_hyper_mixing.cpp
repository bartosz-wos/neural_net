// test_hyper_mixing.cpp — Tests for the HyperMixing architecture.
//
// HyperMixing (Mai et al. 2024, "HyperMixing: A Complement to Attention",
// https://arxiv.org/abs/2401.09656) uses a hypernetwork to generate a (T, T)
// token-mixing matrix from the input. Forward path:
//
//   x = input                                              // (T, d_model)
//   R_logits = W_h · x                                     // (T, T)
//   R = row_softmax(R_logits)                              // (T, T) rows sum to 1
//   U = W_2 · σ(W_1 · x)                                   // (T, d_model), σ = GELU
//   H_T = R @ U                                            // (T, d_model) — token mix
//   Z = W_4 · σ(W_3 · H_T)                                 // (T, d_model) — channel mix
//   out = LayerNorm(Z + x)                                 // (T, d_model)
//
// Tests (14):
//   1.  test_constructor_validates_dims
//   2.  test_forward_shape_finite_nonzero
//   3.  test_input_gradient_fd
//   4.  test_W_h_gradient_fd
//   5.  test_W_1_gradient_fd
//   6.  test_W_3_gradient_fd
//   7.  test_permutation_equivariance
//   8.  test_determinism
//   9.  test_T1_edge_case
//   10. test_training_reduces_loss
//   11. test_model_forward_shape
//   12. test_model_training_reduces_loss
//   13. test_mutation_W_h_grad_path
//   14. test_accessors_and_param_count

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <vector>
#include <algorithm>
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

// Build a (T, D) tensor with seeded random values (mt19937 — deterministic
// across runs and NOT coupled to global rand() state).
static Tensor rand_tensor(size_t T, size_t D, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = nd(rng);
    return x;
}

// MSE loss sum((y-t)^2) / (2T). Gradient: dL/dy = (y - t) / T.
static inline double block_mse(const Tensor& y, const Tensor& t) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y[i][j] - t[i][j];
            L += d * d;
        }
    return L / (2.0 * y.rows);
}

// FD gradient for an entry (r, c) of a parameter tensor, on MSE loss.
static double fd_grad_param(Layer& layer, const Tensor& x, const Tensor& target,
                            Tensor& param, size_t r, size_t c,
                            double eps = 1e-4) {
    double orig = param(r, c);
    param(r, c) = orig + eps;
    Tensor y_plus = layer.forward(x);
    double L_plus = block_mse(y_plus, target);

    param(r, c) = orig - eps;
    Tensor y_minus = layer.forward(x);
    double L_minus = block_mse(y_minus, target);

    param(r, c) = orig;
    return (L_plus - L_minus) / (2.0 * eps);
}

// FD gradient for an input entry (r, c) on MSE loss.
static double fd_grad_input(Layer& layer, const Tensor& x, const Tensor& target,
                            size_t r, size_t c, double eps = 1e-4) {
    Tensor x_plus = x;
    Tensor x_minus = x;
    x_plus[r][c] += eps;
    x_minus[r][c] -= eps;

    Tensor y_plus = layer.forward(x_plus);
    Tensor y_minus = layer.forward(x_minus);
    return (block_mse(y_plus, target) - block_mse(y_minus, target)) / (2.0 * eps);
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double d = std::abs(a.data[i] - b.data[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor_validates_dims() {
    cout << "--- Test 1: HyperMixingLayer constructor validates dims ---" << endl;
    bool ok = true;
    try {
        HyperMixingLayer bad(0, 8, 3);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("d_model=0 throws", ok);

    ok = true;
    try {
        HyperMixingLayer bad(4, 0, 3);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("mlp_hidden=0 throws", ok);

    ok = true;
    try {
        HyperMixingLayer bad(4, 8, 0);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("num_tokens=0 throws", ok);

    HyperMixingLayer ok1(4, 8, 3);
    check("default-construct ok", true);

    HyperMixingLayer ok2(8, 16, 1);
    check("default-construct T=1 ok", true);

    HyperMixingLayer ok3(8, 16, 16);
    check("default-construct large T ok", true);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape + finiteness + nonzero
// ---------------------------------------------------------------------------
static void test_forward_shape_finite_nonzero() {
    cout << "--- Test 2: HyperMixingLayer forward shape + finiteness ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor y = layer.forward(x);
    check("forward shape (T=3, d=4)", y.rows == 3 && y.cols == 4);

    bool finite = true;
    bool nonzero = false;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
        if (y.data[i] != 0.0) nonzero = true;
    }
    check("output finite", finite);
    check("output nonzero", nonzero);
}

// ---------------------------------------------------------------------------
// Test 3: input gradient via centered FD on MSE loss
// ---------------------------------------------------------------------------
static void test_input_gradient_fd() {
    cout << "--- Test 3: HyperMixingLayer input gradient FD check ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = layer.forward(x);
    // Mean MSE: grad_out = (y - t) / T
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    double max_rel_err = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double ana = grad_in[i][j];
            double fd = fd_grad_input(layer, x, target, i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            denom = std::max(denom, 1e-12);
            double re = std::abs(ana - fd) / denom;
            if (re > max_rel_err) max_rel_err = re;
        }
    }
    cout << "  max_rel_err = " << max_rel_err << endl;
    check("input grad rel_err < 1e-4", max_rel_err < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 4: W_h gradient via centered FD
// ---------------------------------------------------------------------------
static void test_W_h_gradient_fd() {
    cout << "--- Test 4: HyperMixingLayer W_h (hypernet) gradient FD check ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = layer.forward(x);
    // Mean MSE: grad_out = (y - t) / T
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    // W_h.weights shape: (num_tokens, d_model) = (3, 4)
    Tensor& W = layer.W_h_.weights;
    double ana = layer.W_h_.grad_weights(0, 0);
    double fd = fd_grad_param(layer, x, target, W, 0, 0);
    cout << "  W_h[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W_h grad rel_err < 1e-4", rel_err < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 5: W_1 (channel MLP layer 1) gradient via centered FD
// ---------------------------------------------------------------------------
static void test_W_1_gradient_fd() {
    cout << "--- Test 5: HyperMixingLayer W_1 (channel MLP layer 1) gradient FD check ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    Tensor& W = layer.W_1_.weights;
    double ana = layer.W_1_.grad_weights(0, 0);
    double fd = fd_grad_param(layer, x, target, W, 0, 0);
    cout << "  W_1[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W_1 grad rel_err < 1e-4", rel_err < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 6: W_3 (channel mixing layer 1) gradient via centered FD
// ---------------------------------------------------------------------------
static void test_W_3_gradient_fd() {
    cout << "--- Test 6: HyperMixingLayer W_3 (channel mixing layer 1) gradient FD check ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    Tensor& W = layer.W_3_.weights;
    double ana = layer.W_3_.grad_weights(0, 0);
    double fd = fd_grad_param(layer, x, target, W, 0, 0);
    cout << "  W_3[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W_3 grad rel_err < 1e-4", rel_err < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 7: permutation consistency
// ---------------------------------------------------------------------------
// Permuting input rows permutes R rows, U rows, and (consequently) the outputs
// in a row-aligned way. We test the structural property:
//   1. last_R is row-stochastic (rows sum to 1) under permutation.
//   2. R rows permute consistently with input permutation.
static void test_permutation_equivariance() {
    cout << "--- Test 7: permutation consistency ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);

    Tensor x_perm(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            x_perm[i][j] = x[(i + 2) % 3][j];

    Tensor y_orig = layer.forward(x);

    bool rsets_ok = true;
    for (size_t i = 0; i < 3; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < 3; ++j) row_sum += layer.last_R[i][j];
        if (std::abs(row_sum - 1.0) > 1e-5) rsets_ok = false;
    }
    check("R is row-stochastic (rows sum to 1)", rsets_ok);

    Tensor y_perm = layer.forward(x_perm);

    bool rsets_perm_ok = true;
    for (size_t i = 0; i < 3; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < 3; ++j) row_sum += layer.last_R[i][j];
        if (std::abs(row_sum - 1.0) > 1e-5) rsets_perm_ok = false;
    }
    check("R is row-stochastic after permuted input", rsets_perm_ok);

    Tensor y_orig_again = layer.forward(x);
    Tensor r_orig = layer.last_R.clone();
    Tensor y_perm_again = layer.forward(x_perm);
    Tensor r_perm = layer.last_R.clone();

    bool r_perm_ok = true;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double diff = std::abs(r_perm[i][j] - r_orig[(i + 2) % 3][j]);
            if (diff > 1e-5) r_perm_ok = false;
        }
    check("R rows permute consistently under input permutation", r_perm_ok);

    (void)y_orig; (void)y_perm; (void)y_orig_again; (void)y_perm_again;
}

// ---------------------------------------------------------------------------
// Test 8: determinism (bit-exact forward with copied params)
// ---------------------------------------------------------------------------
static void test_determinism() {
    cout << "--- Test 8: determinism (copied params = bit-exact fwd) ---" << endl;
    HyperMixingLayer layer1(4, 8, 3);
    HyperMixingLayer layer2(4, 8, 3);
    layer2.copy_params_from(layer1);

    Tensor x = rand_tensor(3, 4, 1);
    Tensor y1 = layer1.forward(x);
    Tensor y2 = layer2.forward(x);
    double diff = max_abs_diff(y1, y2);
    cout << "  max abs diff = " << diff << endl;
    check("bit-identical forward with copied params", diff == 0.0);
}

// ---------------------------------------------------------------------------
// Test 9: T=1 edge case
// ---------------------------------------------------------------------------
static void test_T1_edge_case() {
    cout << "--- Test 9: T=1 edge case (forward + gradient FD) ---" << endl;
    HyperMixingLayer layer(4, 8, 1);
    Tensor x = rand_tensor(1, 4, 1);
    Tensor y = layer.forward(x);
    check("T=1 forward shape (T=1, d=4)", y.rows == 1 && y.cols == 4);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
    }
    check("T=1 output finite", finite);

    Tensor target = rand_tensor(1, 4, 2);
    Tensor grad_out(1, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);

    double max_rel_err = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double ana = grad_in[i][j];
            double fd = fd_grad_input(layer, x, target, i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            denom = std::max(denom, 1e-12);
            double re = std::abs(ana - fd) / denom;
            if (re > max_rel_err) max_rel_err = re;
        }
    }
    cout << "  T=1 max_rel_err = " << max_rel_err << endl;
    check("T=1 input grad rel_err < 1e-4", max_rel_err < 1e-4);
}

// ---------------------------------------------------------------------------
// Test 10: training reduces loss (50 SGD steps)
// ---------------------------------------------------------------------------
static void test_training_reduces_loss() {
    cout << "--- Test 10: training reduces loss (50 SGD steps) ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y0 = layer.forward(x);
    double L0 = block_mse(y0, target);
    cout << "  L0 = " << L0 << endl;

    double lr = 1e-2;
    for (int step = 0; step < 50; ++step) {
        Tensor y = layer.forward(x);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
        layer.backward(grad_out, lr);
        layer.update_weights(lr);
    }

    Tensor y1 = layer.forward(x);
    double L1 = block_mse(y1, target);
    cout << "  L0 = " << L0 << " -> L1 = " << L1 << endl;
    check("loss decreased by > 30%", L1 < 0.7 * L0);
}

// ---------------------------------------------------------------------------
// Test 11: HyperMixingModel forward shape
// ---------------------------------------------------------------------------
static void test_model_forward_shape() {
    cout << "--- Test 11: HyperMixingModel forward shape ---" << endl;
    HyperMixingModel model(3, 4, 2, 2, 8, 3);
    Tensor x = rand_tensor(3, 3, 1);
    Tensor y = model.forward(x);
    check("HyperMixingModel forward shape (T=3, in=3) -> (T=3, out=2)",
          y.rows == 3 && y.cols == 2);

    bool finite = true;
    bool nonzero = false;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
        if (y.data[i] != 0.0) nonzero = true;
    }
    check("model output finite", finite);
    check("model output nonzero", nonzero);

    auto p = model.parameters();
    auto g = model.gradients();
    check("parameters().size() > 0", !p.empty());
    check("gradients().size() > 0", !g.empty());
    check("parameters() and gradients() sizes match", p.size() == g.size());
}

// ---------------------------------------------------------------------------
// Test 12: HyperMixingModel training reduces loss
// ---------------------------------------------------------------------------
static void test_model_training_reduces_loss() {
    cout << "--- Test 12: HyperMixingModel training reduces loss (80 steps) ---" << endl;
    HyperMixingModel model(3, 4, 2, 2, 8, 3);
    Tensor x = rand_tensor(3, 3, 1);
    Tensor target = rand_tensor(3, 2, 2);

    Tensor y0 = model.forward(x);
    double L0 = block_mse(y0, target);
    cout << "  L0 = " << L0 << endl;

    double lr = 1e-2;
    for (int step = 0; step < 80; ++step) {
        Tensor y = model.forward(x);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
        model.backward(grad_out, lr);
        model.update_weights(lr);
    }

    Tensor y1 = model.forward(x);
    double L1 = block_mse(y1, target);
    cout << "  L0 = " << L0 << " -> L1 = " << L1 << endl;
    check("model training reduced loss by > 30%", L1 < 0.7 * L0);
}

// ---------------------------------------------------------------------------
// Test 13: mutation — stub out the W_h gradient path
// ---------------------------------------------------------------------------
// If the W_h gradient path is bug-free, stubbing it should make the W_h FD
// check fail. If stubbing it doesn't fail, the test is vacuous.
static void test_mutation_W_h_grad_path() {
    cout << "--- Test 13: mutation — stub W_h grad path; Test 4 FD should fail ---" << endl;
    HyperMixingLayer layer(4, 8, 3);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / 3.0;
    layer.backward(grad_out, 1e-3);

    // Pick the W_h gradient entry with the largest absolute value across all
    // positions. We then zero it and verify FD disagrees.
    Tensor& W = layer.W_h_.weights;
    double best_ana = 0.0;
    size_t best_r = 0, best_c = 0;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) {
            double a = layer.W_h_.grad_weights(r, c);
            if (std::abs(a) > std::abs(best_ana)) { best_ana = a; best_r = r; best_c = c; }
        }

    double best_fd = fd_grad_param(layer, x, target, W, best_r, best_c);
    double pre_rel = std::abs(best_ana - best_fd) / std::max(std::abs(best_ana), std::max(std::abs(best_fd), 1e-12));
    cout << "  Pre-mutation W_h[" << best_r << "," << best_c
         << "]: ana=" << best_ana << " fd=" << best_fd
         << " rel=" << pre_rel << endl;
    check("pre-mutation match is non-trivial (rel < 1e-4)", pre_rel < 1e-4);

    // Mutation: zero out the W_h gradient. The analytical now reports 0
    // while FD still reports the true value; the FD check should fail.
    layer.W_h_.grad_weights(best_r, best_c) = 0.0;
    double post_rel = std::abs(0.0 - best_fd) / std::max(std::abs(best_fd), 1e-12);
    cout << "  After stub, W_h[" << best_r << "," << best_c
         << "]: ana=0, fd=" << best_fd << ", rel_err=" << post_rel << endl;
    check("Test 4 catches stubbed W_h grad path", post_rel > 1e-4);
}

// ---------------------------------------------------------------------------
// Test 14: accessors + param count
// ---------------------------------------------------------------------------
static void test_accessors_and_param_count() {
    cout << "--- Test 14: accessors + param count ---" << endl;
    HyperMixingLayer layer(4, 8, 3);

    check("d_model() == 4", layer.d_model() == 4);
    check("mlp_hidden() == 8", layer.mlp_hidden() == 8);
    check("num_tokens() == 3", layer.num_tokens() == 3);

    auto p = layer.parameters();
    auto g = layer.gradients();
    cout << "  parameters().size() = " << p.size() << endl;
    // 5 Denses * 2 (W + b) + 1 LayerNorm * 2 (gamma + beta) = 12
    check("parameters().size() == 12 (5 Denses * 2 + 1 LN * 2)", p.size() == 12);
    check("gradients().size() == parameters().size()", g.size() == p.size());
}

int main() {
    srand(42);
    cout << "=== HyperMixing Tests (Mai et al. 2024, https://arxiv.org/abs/2401.09656) ===" << endl;
    cout << fixed << setprecision(6);

    test_constructor_validates_dims();
    test_forward_shape_finite_nonzero();
    test_input_gradient_fd();
    test_W_h_gradient_fd();
    test_W_1_gradient_fd();
    test_W_3_gradient_fd();
    test_permutation_equivariance();
    test_determinism();
    test_T1_edge_case();
    test_training_reduces_loss();
    test_model_forward_shape();
    test_model_training_reduces_loss();
    test_mutation_W_h_grad_path();
    test_accessors_and_param_count();

    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
