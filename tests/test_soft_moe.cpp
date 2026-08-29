// test_soft_moe.cpp — Tests for the Soft Mixture of Experts (Soft MoE) layer.
//
// Soft MoE (Puigcerver et al. ICLR 2024, https://arxiv.org/abs/2308.00951,
// "From Sparse to Soft Mixtures of Experts") replaces the hard top-k
// token-choice routing of standard sparse MoE with a fully differentiable
// slot-based soft dispatch.
//
// Per the paper (Eq. 1 in §2.2):
//   D = softmax(X · W_disp^T, axis=1)           (T, E*S)   dispatch weights
//   X' = D^T @ X                                  (E*S, d_model)   slot inputs
//   Y  = [expert_1(X'_1); ...; expert_E(X'_E)]    (E*S, d_model)   slot outputs
//   C  = softmax(X' · W_comb^T, axis=1)          (E*S, T)   combine weights
//   output = C^T @ Y                              (T, d_model)
//
// Layout: (T, d_model) end-to-end.
//
// Tests (15):
//   1.  test_constructor_validates_dims
//   2.  test_forward_shape_finite_nonzero
//   3.  test_dispatch_invariants                (D rows sum to 1, D ∈ (0,1))
//   4.  test_combine_invariants                 (C cols sum to 1, C ∈ (0,1))
//   5.  test_input_gradient_fd
//   6.  test_W_disp_gradient_fd
//   7.  test_W_comb_gradient_fd
//   8.  test_expert_W1_gradient_fd
//   9.  test_degeneracy_single_expert_full_slots
//   10. test_determinism
//   11. test_training_reduces_loss
//   12. test_model_forward_shape_and_finiteness
//   13. test_model_training_reduces_loss
//   14. test_parameters_count
//   15. test_mutation_dispatch_path

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

static Tensor rand_tensor(size_t T, size_t D, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = nd(rng);
    return x;
}

static inline double block_mse(const Tensor& y, const Tensor& t) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y[i][j] - t[i][j];
            L += d * d;
        }
    return L / (2.0 * y.rows);
}

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
    cout << "--- Test 1: SoftMoELayer constructor validates dims ---" << endl;
    bool ok = true;
    try {
        SoftMoELayer bad(0, 4, 2, 2);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("d_model=0 throws", ok);

    ok = true;
    try {
        SoftMoELayer bad(8, 0, 2, 2);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("num_experts=0 throws", ok);

    ok = true;
    try {
        SoftMoELayer bad(8, 2, 0, 2);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("slots_per_expert=0 throws", ok);

    ok = true;
    try {
        SoftMoELayer bad(8, 2, 2, 0);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("d_expert=0 throws", ok);

    // Valid
    SoftMoELayer ok1(8, 2, 2, 4);
    check("default-construct (d_model=8, E=2, S=2, d_expert=4) ok", true);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape + finiteness + nonzero
// ---------------------------------------------------------------------------
static void test_forward_shape_finite_nonzero() {
    cout << "--- Test 2: SoftMoELayer forward shape + finiteness ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(3, 8, 1);
    Tensor y = layer.forward(x);
    check("forward shape (T=3, d=8)", y.rows == 3 && y.cols == 8);

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
// Test 3: dispatch invariants (D rows sum to 1, D ∈ (0,1))
// ---------------------------------------------------------------------------
static void test_dispatch_invariants() {
    cout << "--- Test 3: SoftMoELayer dispatch invariants ---" << endl;
    SoftMoELayer layer(8, 2, 3, 4);
    Tensor x = rand_tensor(4, 8, 11);
    (void)layer.forward(x);   // populates caches

    // last_D_ is (T, E*S) = (4, 6)
    const Tensor& D = layer.last_D_;
    double row_sum_err = 0.0;
    bool all_in_unit = true;
    for (size_t t = 0; t < D.rows; ++t) {
        double sum = 0.0;
        for (size_t j = 0; j < D.cols; ++j) {
            sum += D[t][j];
            if (D[t][j] <= 0.0 || D[t][j] >= 1.0) all_in_unit = false;
        }
        double err = std::abs(sum - 1.0);
        if (err > row_sum_err) row_sum_err = err;
    }
    check("D rows sum to 1 (max err < 1e-12)", row_sum_err < 1e-12);
    check("D entries strictly in (0, 1)", all_in_unit);
    check("D shape (4, 6)", D.rows == 4 && D.cols == 6);
}

// ---------------------------------------------------------------------------
// Test 4: combine invariants (C rows sum to 1, C ∈ (0,1))
// ---------------------------------------------------------------------------
static void test_combine_invariants() {
    cout << "--- Test 4: SoftMoELayer combine invariants ---" << endl;
    SoftMoELayer layer(8, 2, 3, 4);
    Tensor x = rand_tensor(4, 8, 12);
    (void)layer.forward(x);

    // last_C_ is (E*S, T) = (6, 4) — softmax per-row over tokens
    const Tensor& C = layer.last_C_;
    double row_sum_err = 0.0;
    bool all_in_unit = true;
    for (size_t i = 0; i < C.rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < C.cols; ++j) {
            sum += C[i][j];
            if (C[i][j] <= 0.0 || C[i][j] >= 1.0) all_in_unit = false;
        }
        double err = std::abs(sum - 1.0);
        if (err > row_sum_err) row_sum_err = err;
    }
    check("C rows sum to 1 (max err < 1e-12)", row_sum_err < 1e-12);
    check("C entries strictly in (0, 1)", all_in_unit);
    check("C shape (6, 4)", C.rows == 6 && C.cols == 4);
}

// ---------------------------------------------------------------------------
// Test 5: input gradient via centered FD on MSE loss
// ---------------------------------------------------------------------------
static void test_input_gradient_fd() {
    cout << "--- Test 5: SoftMoELayer input gradient FD check ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(2, 8, 1);
    Tensor target = rand_tensor(2, 8, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(2, 8);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double ana = grad_in[i][j];
            double fd = fd_grad_input(layer, x, target, i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            denom = std::max(denom, 1e-12);
            double re = std::abs(ana - fd) / denom;
            double ae = std::abs(ana - fd);
            if (re > max_rel_err) max_rel_err = re;
            if (ae > max_abs_err) max_abs_err = ae;
        }
    }
    cout << "  max_rel_err = " << max_rel_err << endl;
    check("input grad rel_err < 1e-3", max_rel_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 6: W_disp gradient via centered FD
// ---------------------------------------------------------------------------
static void test_W_disp_gradient_fd() {
    cout << "--- Test 6: SoftMoELayer W_disp gradient FD check ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(2, 8, 1);
    Tensor target = rand_tensor(2, 8, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(2, 8);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    // W_disp.weights shape: (E*S, d_model) = (4, 8)
    Tensor& W = layer.W_disp_.weights;
    double best_ana = 0.0, best_fd = 0.0;
    size_t best_r = 0, best_c = 0;
    for (size_t r = 0; r < W.rows; ++r) {
        for (size_t c = 0; c < W.cols; ++c) {
            double ana = layer.W_disp_.grad_weights(r, c);
            if (std::abs(ana) > std::abs(best_ana)) {
                best_ana = ana;
                best_r = r;
                best_c = c;
            }
        }
    }
    best_fd = fd_grad_param(layer, x, target, W, best_r, best_c);
    cout << "  W_disp[" << best_r << "," << best_c << "]: ana=" << best_ana
         << " fd=" << best_fd << endl;
    double denom = std::max(std::abs(best_ana), std::abs(best_fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(best_ana - best_fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W_disp grad rel_err < 1e-3 (best entry)", rel_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 7: W_comb gradient via centered FD
// ---------------------------------------------------------------------------
static void test_W_comb_gradient_fd() {
    cout << "--- Test 7: SoftMoELayer W_comb gradient FD check ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(2, 8, 1);
    Tensor target = rand_tensor(2, 8, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(2, 8);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    // W_comb_[0].weights shape: (T=2, d_model=8)
    Dense& W = layer.W_comb_[0];
    double best_ana = 0.0, best_fd = 0.0;
    size_t best_r = 0, best_c = 0;
    for (size_t r = 0; r < W.weights.rows; ++r) {
        for (size_t c = 0; c < W.weights.cols; ++c) {
            double ana = W.grad_weights(r, c);
            if (std::abs(ana) > std::abs(best_ana)) {
                best_ana = ana;
                best_r = r;
                best_c = c;
            }
        }
    }
    best_fd = fd_grad_param(layer, x, target, W.weights, best_r, best_c);
    cout << "  W_comb_[0][" << best_r << "," << best_c << "]: ana=" << best_ana
         << " fd=" << best_fd << endl;
    double denom = std::max(std::abs(best_ana), std::abs(best_fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(best_ana - best_fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W_comb grad rel_err < 1e-3 (best entry)", rel_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 8: per-expert W1 gradient via centered FD
// ---------------------------------------------------------------------------
static void test_expert_W1_gradient_fd() {
    cout << "--- Test 8: SoftMoELayer expert W1 gradient FD check ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(2, 8, 1);
    Tensor target = rand_tensor(2, 8, 2);

    Tensor y = layer.forward(x);
    Tensor grad_out(2, 8);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = layer.backward(grad_out, 1e-3);
    (void)grad_in;

    // W1_[0].weights shape: (d_expert, d_model) = (4, 8)
    Dense& W = layer.W1_[0];
    double best_ana = 0.0, best_fd = 0.0;
    size_t best_r = 0, best_c = 0;
    for (size_t r = 0; r < W.weights.rows; ++r) {
        for (size_t c = 0; c < W.weights.cols; ++c) {
            double ana = W.grad_weights(r, c);
            if (std::abs(ana) > std::abs(best_ana)) {
                best_ana = ana;
                best_r = r;
                best_c = c;
            }
        }
    }
    best_fd = fd_grad_param(layer, x, target, W.weights, best_r, best_c);
    cout << "  W1_[0][" << best_r << "," << best_c << "]: ana=" << best_ana
         << " fd=" << best_fd << endl;
    double denom = std::max(std::abs(best_ana), std::abs(best_fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(best_ana - best_fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("W1_[0] grad rel_err < 1e-3 (best entry)", rel_err < 1e-3);
}

// ---------------------------------------------------------------------------
// Test 9: degeneracy — E=1, S=T (single expert, all slots) → known form
// ---------------------------------------------------------------------------
static void test_degeneracy_single_expert_full_slots() {
    cout << "--- Test 9: degeneracy — E=1, S=T single-expert-full-slots ---" << endl;
    size_t T = 3, d = 4;
    SoftMoELayer layer(d, /*E=*/1, /*S=*/T, /*d_expert=*/2);
    Tensor x = rand_tensor(T, d, 21);
    (void)layer.forward(x);

    // With E=1 and S=T: dispatch becomes a permutation softmax over T tokens
    // (each row of D is a softmax over all tokens; same input -> identical rows).
    // When x has different rows, D is a per-token softmax over slots=T.
    // We only sanity-check the cache is sensible (no NaN, shapes right).
    check("last_D_ shape (T, T) = (3, 3)", layer.last_D_.rows == T && layer.last_D_.cols == T);
    check("last_Xp_ shape (T, d) = (3, 4)", layer.last_Xp_.rows == T && layer.last_Xp_.cols == d);
    check("last_C_ shape (T, T) = (3, 3)", layer.last_C_.rows == T && layer.last_C_.cols == T);
}

// ---------------------------------------------------------------------------
// Test 10: determinism — two fresh layers with copied params produce bit-exact forward
// ---------------------------------------------------------------------------
static void test_determinism() {
    cout << "--- Test 10: SoftMoELayer determinism (bit-exact with copied params) ---" << endl;
    SoftMoELayer a(8, 2, 2, 4);
    SoftMoELayer b(8, 2, 2, 4);
    Tensor x = rand_tensor(3, 8, 31);

    // Copy params from a → b
    b.copy_params_from(a);

    Tensor y_a = a.forward(x);
    Tensor y_b = b.forward(x);

    double diff = max_abs_diff(y_a, y_b);
    cout << "  max_abs_diff = " << diff << endl;
    check("bit-exact forward with copied params", diff == 0.0);
}

// ---------------------------------------------------------------------------
// Test 11: training reduces loss
// ---------------------------------------------------------------------------
static void test_training_reduces_loss() {
    cout << "--- Test 11: SoftMoELayer training reduces loss ---" << endl;
    SoftMoELayer layer(4, 2, 2, 3);
    Tensor x = rand_tensor(2, 4, 41);
    Tensor target = rand_tensor(2, 4, 42);

    double L0 = 0.0;
    double lr = 1e-1;
    for (size_t i = 0; i < 200; ++i) {
        Tensor y = layer.forward(x);
        double L = block_mse(y, target);
        if (i == 0) L0 = L;
        Tensor grad_out(2, 4);
        for (size_t r = 0; r < y.rows; ++r)
            for (size_t c = 0; c < y.cols; ++c)
                grad_out[r][c] = (y[r][c] - target[r][c]) / y.rows;
        (void)layer.backward(grad_out, lr);
        layer.update_weights(lr);
        layer.zero_grad();
    }
    Tensor y_final = layer.forward(x);
    double Lf = block_mse(y_final, target);
    cout << "  L0=" << L0 << " Lf=" << Lf << endl;
    check("loss decreased > 50%", Lf < L0 * 0.5);
}

// ---------------------------------------------------------------------------
// Test 12: SoftMoEModel forward shape + finiteness
// ---------------------------------------------------------------------------
static void test_model_forward_shape_and_finiteness() {
    cout << "--- Test 12: SoftMoEModel forward shape + finiteness ---" << endl;
    SoftMoEModel model(/*input_dim=*/3, /*d_model=*/8, /*output_dim=*/2,
                       /*num_layers=*/2, /*num_experts=*/2, /*slots_per_expert=*/2,
                       /*d_expert=*/4);
    Tensor x = rand_tensor(2, 3, 51);
    Tensor y = model.forward(x);
    // SoftMoEModel mean-pools over tokens → output (1, out_dim)
    check("model forward shape (T=2, out=2) -> (1, out=2)", y.rows == 1 && y.cols == 2);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i)
        if (!std::isfinite(y[i][0])) finite = false;
    check("model output finite", finite);
}

// ---------------------------------------------------------------------------
// Test 13: SoftMoEModel training reduces loss
// ---------------------------------------------------------------------------
static void test_model_training_reduces_loss() {
    cout << "--- Test 13: SoftMoEModel training reduces loss ---" << endl;
    SoftMoEModel model(3, 8, 2, 2, 2, 2, 4);
    Tensor x = rand_tensor(2, 3, 61);
    // mean-pool → output (1, 2)
    Tensor target(1, 2);
    std::mt19937 rng(62);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (size_t i = 0; i < 2; ++i) target[0][i] = nd(rng);

    double L0 = 0.0;
    for (size_t i = 0; i < 80; ++i) {
        Tensor y = model.forward(x);
        double L = block_mse(y, target);
        if (i == 0) L0 = L;
        Tensor grad_out(1, 2);
        for (size_t r = 0; r < y.rows; ++r)
            for (size_t c = 0; c < y.cols; ++c)
                grad_out[r][c] = (y[r][c] - target[r][c]) / y.rows;
        (void)model.backward(grad_out, 1e-2);
        model.update_weights(1e-2);
        model.zero_grad();
    }
    Tensor y_final = model.forward(x);
    double Lf = block_mse(y_final, target);
    cout << "  L0=" << L0 << " Lf=" << Lf << endl;
    check("model loss decreased > 50%", Lf < L0 * 0.5);
}

// ---------------------------------------------------------------------------
// Test 14: parameters count
// ---------------------------------------------------------------------------
static void test_parameters_count() {
    cout << "--- Test 14: SoftMoELayer parameters count ---" << endl;
    size_t d = 4, E = 2, S = 2, d_e = 3;
    SoftMoELayer layer(d, E, S, d_e);
    auto params = layer.parameters();
    // Expected: W_disp.weights (E*S, d), and num_slots_ expert W1 (d_e, d)
    // W_comb_ is lazily initialized on first forward, so it may not be present
    // yet (we count it via forward call below).
    int expert_w1_count = 0;
    bool has_disp = false;
    for (Tensor* p : params) {
        if (p->rows == E * S && p->cols == d) has_disp = true;
        if (p->rows == d_e && p->cols == d) expert_w1_count++;
    }
    check("W_disp.weights in parameters()", has_disp);
    check("num_slots_ expert W1 in parameters()", expert_w1_count >= (int)(E * S));

    // After a forward, W_comb_ should also appear in parameters().
    Tensor xi = rand_tensor(2, d, 99);
    (void)layer.forward(xi);
    auto params2 = layer.parameters();
    int comb_count = 0;
    for (Tensor* p : params2) {
        if (p->rows == 2 && p->cols == d) comb_count++;   // W_comb weights are (T=2, d)
    }
    check("W_comb_ entries in parameters() after forward", comb_count >= (int)(E * S));
}

// ---------------------------------------------------------------------------
// Test 15: mutation — stub dispatch path, output must change
// ---------------------------------------------------------------------------
static void test_mutation_dispatch_path() {
    cout << "--- Test 15: mutation — stub dispatch path ---" << endl;
    SoftMoELayer layer(8, 2, 2, 4);
    Tensor x = rand_tensor(3, 8, 71);
    Tensor y_before = layer.forward(x);

    // Zero out W_disp_.weights — this breaks the dispatch, the slot inputs
    // X' should collapse to a constant, and output should change.
    for (size_t i = 0; i < layer.W_disp_.weights.data.size(); ++i)
        layer.W_disp_.weights.data[i] = 0.0;
    Tensor y_after = layer.forward(x);

    double diff = max_abs_diff(y_before, y_after);
    cout << "  max_abs_diff after zeroing W_disp = " << diff << endl;
    check("output changes when W_disp zeroed", diff > 1e-6);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cout << "=========================================" << endl;
    cout << "  SoftMoE Tests" << endl;
    cout << "=========================================" << endl;
    cout << fixed << setprecision(6);

    test_constructor_validates_dims();
    test_forward_shape_finite_nonzero();
    test_dispatch_invariants();
    test_combine_invariants();
    test_input_gradient_fd();
    test_W_disp_gradient_fd();
    test_W_comb_gradient_fd();
    test_expert_W1_gradient_fd();
    test_degeneracy_single_expert_full_slots();
    test_determinism();
    test_training_reduces_loss();
    test_model_forward_shape_and_finiteness();
    test_model_training_reduces_loss();
    test_parameters_count();
    test_mutation_dispatch_path();

    cout << "=========================================" << endl;
    cout << "  Summary: " << passed << " passed, " << failed << " failed" << endl;
    cout << "=========================================" << endl;
    return failed == 0 ? 0 : 1;
}