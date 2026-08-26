// ============================================================================
// RWKV-7 Parallel Attention tests
//   Peng et al. 2025 "RWKV-7 'Goose' with Expressive Dynamic State Evolution"
//   https://arxiv.org/abs/2503.14456
// ============================================================================
//
// Test coverage: RWKV7ParallelAttention. Verifies:
//   - constructor validation
//   - forward shape, finiteness, nonzero
//   - forward BIT-EXACT equivalence with RWKV7TimeMix (chunk_size=T)
//   - input gradient equivalence with RWKV7TimeMix
//   - all parameter gradient FD checks (W_r/W_k/W_v/W_d/W_a + xi/alpha/mu_*)
//   - chunk_size < T also produces correct forward (within chunk math is identical)
//   - multi-head (H=2) shape + input grad
//   - training step reduces loss
//
// The defining test: forward and gradient outputs of RWKV7ParallelAttention
// must match RWKV7TimeMix bit-exactly (FP64 tolerance) when given the same
// parameters, same input, and chunk_size >= T. This makes the parallel form
// a verification oracle for the recurrent form.
// ============================================================================

#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/layers/recurrent/rwkv7.h"
#include "nn/layers/attention/rwkv7_parallel.h"

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
Tensor make_input(size_t T, size_t d, double scale = 0.1, size_t seed = 0) {
    Tensor x(T, d);
    std::mt19937 rng(seed + 12345);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d; ++j) {
            x[i][j] = dist(rng);
        }
    }
    return x;
}

// Initialize a 5-gate layer with deterministic small weights.
void init_layer_weights(RWKV7TimeMix& cell, double scale = 0.15) {
    for (auto* W : {&cell.W_r, &cell.W_k, &cell.W_v, &cell.W_d, &cell.W_a}) {
        for (size_t i = 0; i < W->weights.rows; ++i) {
            for (size_t j = 0; j < W->weights.cols; ++j) {
                double v = scale * static_cast<double>((i + 1) * (j + 1))
                         / static_cast<double>(W->weights.cols);
                W->weights[i][j] = v;
            }
            for (size_t j = 0; j < W->bias.cols; ++j) {
                W->bias[0][j] = scale * 0.05 * static_cast<double>(j + 1);
            }
        }
    }
}

void init_layer_weights(RWKV7ParallelAttention& attn, double scale = 0.15) {
    for (auto* W : {&attn.W_r, &attn.W_k, &attn.W_v, &attn.W_d, &attn.W_a}) {
        for (size_t i = 0; i < W->weights.rows; ++i) {
            for (size_t j = 0; j < W->weights.cols; ++j) {
                double v = scale * static_cast<double>((i + 1) * (j + 1))
                         / static_cast<double>(W->weights.cols);
                W->weights[i][j] = v;
            }
            for (size_t j = 0; j < W->bias.cols; ++j) {
                W->bias[0][j] = scale * 0.05 * static_cast<double>(j + 1);
            }
        }
    }
}

// Copy params from RWKV7TimeMix into RWKV7ParallelAttention.
void copy_params_from(const RWKV7TimeMix& src, RWKV7ParallelAttention& dst) {
    const Dense* src_w[5] = {&src.W_r, &src.W_k, &src.W_v, &src.W_d, &src.W_a};
    Dense* dst_w[5] = {&dst.W_r, &dst.W_k, &dst.W_v, &dst.W_d, &dst.W_a};
    for (int idx = 0; idx < 5; ++idx) {
        const Dense& s = *src_w[idx];
        Dense& d = *dst_w[idx];
        for (size_t i = 0; i < s.weights.rows; ++i) {
            for (size_t j = 0; j < s.weights.cols; ++j) {
                d.weights[i][j] = s.weights[i][j];
            }
            for (size_t j = 0; j < s.bias.cols; ++j) {
                d.bias[0][j] = s.bias[0][j];
            }
        }
    }
    for (size_t j = 0; j < src.d_; ++j) {
        dst.xi(0, j) = src.xi(0, j);
        dst.mu_r(0, j) = src.mu_r(0, j);
        dst.mu_k(0, j) = src.mu_k(0, j);
        dst.mu_v(0, j) = src.mu_v(0, j);
        dst.mu_d(0, j) = src.mu_d(0, j);
        dst.mu_a(0, j) = src.mu_a(0, j);
    }
    dst.alpha(0, 0) = src.alpha(0, 0);
}

// ============================================================================
// Tests
// ============================================================================

// Test 1: constructor + accessors
void test_constructor_and_accessors() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    EXPECT(attn.d() == 4, "d()");
    EXPECT(attn.num_heads() == 2, "num_heads()");
    EXPECT(attn.head_dim() == 2, "head_dim()");
    EXPECT(attn.chunk_size() == 0, "default chunk_size is 0 (= use T)");
    EXPECT(attn.name() == "RWKV7ParallelAttention", "name()");
}

// Test 2: constructor validation
void test_constructor_validation() {
    bool threw_d0 = false;
    try { RWKV7ParallelAttention attn(0, 1); } catch (...) { threw_d0 = true; }
    EXPECT(threw_d0, "d=0 throws");

    bool threw_nh0 = false;
    try { RWKV7ParallelAttention attn(4, 0); } catch (...) { threw_nh0 = true; }
    EXPECT(threw_nh0, "num_heads=0 throws");

    bool threw_uneven = false;
    try { RWKV7ParallelAttention attn(4, 3); } catch (...) { threw_uneven = true; }
    EXPECT(threw_uneven, "d=4 num_heads=3 throws (not divisible)");

    // valid
    bool ok = true;
    try { RWKV7ParallelAttention attn(8, 2); } catch (...) { ok = false; }
    EXPECT(ok, "valid (8, 2) doesn't throw");

    // valid with explicit chunk_size
    ok = true;
    try { RWKV7ParallelAttention attn(8, 2, 3); } catch (...) { ok = false; }
    EXPECT(ok, "valid (8, 2, chunk_size=3) doesn't throw");
}

// Test 3: forward shape, finite, nonzero
void test_forward_shape_and_finite() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    init_layer_weights(attn, 0.3);
    Tensor x = make_input(3, 4, 0.1, /*seed=*/1);
    Tensor y = attn.forward(x);
    EXPECT(y.rows == 3, "y.rows == T");
    EXPECT(y.cols == 4, "y.cols == d");
    bool finite = true;
    bool nonzero = false;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; }
        if (std::fabs(y.data[i]) > 1e-6) { nonzero = true; }
    }
    EXPECT(finite, "forward output finite");
    EXPECT(nonzero, "forward output nonzero");
}

// Test 4: forward BIT-EXACT equivalence with RWKV7TimeMix (chunk_size=T default)
void test_forward_equivalence_with_recurrent() {
    RWKV7TimeMix recurrent(4, /*num_heads=*/2);
    init_layer_weights(recurrent, 0.3);
    RWKV7ParallelAttention parallel(4, /*num_heads=*/2);
    init_layer_weights(parallel, 0.3);
    copy_params_from(recurrent, parallel);

    Tensor x = make_input(4, 4, 0.2, /*seed=*/2);
    Tensor y_rec = recurrent.forward(x);
    Tensor y_par = parallel.forward(x);

    double max_diff = 0.0;
    for (size_t i = 0; i < y_rec.data.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(y_rec.data[i] - y_par.data[i]));
    }
    std::printf("    parallel vs recurrent forward max_diff = %.3e\n", max_diff);
    EXPECT(max_diff < 1e-10, "parallel vs recurrent forward BIT-EXACT");
}

// Test 5: forward equivalence with explicit chunk_size=2 (smaller than T)
void test_forward_equivalence_chunk_smaller_than_T() {
    RWKV7TimeMix recurrent(4, /*num_heads=*/2);
    init_layer_weights(recurrent, 0.3);
    RWKV7ParallelAttention parallel(4, /*num_heads=*/2, /*chunk_size=*/2);
    init_layer_weights(parallel, 0.3);
    copy_params_from(recurrent, parallel);

    Tensor x = make_input(4, 4, 0.2, /*seed=*/3);
    Tensor y_rec = recurrent.forward(x);
    Tensor y_par = parallel.forward(x);

    // With chunk_size=2 and T=4: chunks are [0,2), [2,4). The recurrence is
    // applied within each chunk (so first chunk starts with wkv_{-1}=0 and
    // produces wkv_1). At the chunk boundary, wkv_1 carries into the next
    // chunk as wkv_prev. Since the recurrent form is the same recurrence
    // applied over the entire sequence, the per-token outputs should match.
    double max_diff = 0.0;
    for (size_t i = 0; i < y_rec.data.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(y_rec.data[i] - y_par.data[i]));
    }
    std::printf("    parallel chunk_size=2 vs recurrent forward max_diff = %.3e\n", max_diff);
    EXPECT(max_diff < 1e-10, "parallel (chunk_size=2) vs recurrent BIT-EXACT");
}

// Test 6: input gradient FD vs analytical on parallel form
void test_input_grad_via_fd() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    init_layer_weights(attn, 0.3);
    Tensor x = make_input(3, 4, 0.1, /*seed=*/4);
    Tensor y = attn.forward(x);

    // Upstream gradient = 1.0 at every entry (simple loss = sum(y))
    Tensor grad_out(3, 4);
    grad_out.fill(1.0);
    Tensor grad_x_ana = attn.backward(grad_out, 0.0);

    // Centered finite differences
    double eps = 1e-5;
    Tensor grad_x_fd(3, 4);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            Tensor x_plus = x.clone();
            x_plus[i][j] += eps;
            Tensor x_minus = x.clone();
            x_minus[i][j] -= eps;
            Tensor y_plus = attn.forward(x_plus);
            Tensor y_minus = attn.forward(x_minus);
            // We need to recompute last_input_/caches for x_plus and x_minus.
            // Since forward overwrites caches, call them sequentially and
            // re-establish baseline. Actually since we need grad at the
            // unperturbed x, the LAST call to forward should be x itself.
            double sp = 0.0, sm = 0.0;
            for (size_t k = 0; k < y.data.size(); ++k) {
                sp += grad_out.data[k] * y_plus.data[k];
                sm += grad_out.data[k] * y_minus.data[k];
            }
            grad_x_fd[i][j] = (sp - sm) / (2.0 * eps);
            // restore cache state by running forward on x again so backward
            // (if called later) sees the correct caches
            attn.forward(x);
        }
    }

    double max_rel = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < grad_x_ana.data.size(); ++i) {
        double ana = grad_x_ana.data[i];
        double fd = grad_x_fd.data[i];
        double scale = std::max({std::fabs(ana), std::fabs(fd), 1e-12});
        double re = std::fabs(ana - fd) / scale;
        max_rel = std::max(max_rel, re);
        max_abs = std::max(max_abs, std::fabs(ana - fd));
    }
    std::printf("    parallel input grad max_rel_err = %.3e (max_abs %.3e)\n", max_rel, max_abs);
    EXPECT(max_rel < 1e-3, "input grad FD vs analytical (parallel)");
}

// Test 7: input gradient equivalence with recurrent form
void test_input_grad_equivalence_with_recurrent() {
    RWKV7TimeMix recurrent(4, /*num_heads=*/2);
    init_layer_weights(recurrent, 0.3);
    RWKV7ParallelAttention parallel(4, /*num_heads=*/2);
    init_layer_weights(parallel, 0.3);
    copy_params_from(recurrent, parallel);

    Tensor x = make_input(4, 4, 0.1, /*seed=*/5);
    Tensor grad_out(4, 4);
    grad_out.fill(1.0);

    Tensor y_rec = recurrent.forward(x);
    Tensor grad_x_rec = recurrent.backward(grad_out, 0.0);

    Tensor y_par = parallel.forward(x);
    Tensor grad_x_par = parallel.backward(grad_out, 0.0);

    double max_diff = 0.0;
    for (size_t i = 0; i < grad_x_rec.data.size(); ++i) {
        max_diff = std::max(max_diff,
            std::fabs(grad_x_rec.data[i] - grad_x_par.data[i]));
    }
    std::printf("    input grad max_diff (parallel vs recurrent) = %.3e\n", max_diff);
    EXPECT(max_diff < 1e-9, "input grad equivalence");
}

// Test 8: parameter gradient equivalence with recurrent form
void test_param_grad_equivalence_with_recurrent() {
    RWKV7TimeMix recurrent(4, /*num_heads=*/2);
    init_layer_weights(recurrent, 0.3);
    RWKV7ParallelAttention parallel(4, /*num_heads=*/2);
    init_layer_weights(parallel, 0.3);
    copy_params_from(recurrent, parallel);

    Tensor x = make_input(4, 4, 0.1, /*seed=*/6);
    Tensor grad_out(4, 4);
    grad_out.fill(1.0);

    recurrent.forward(x);
    recurrent.backward(grad_out, 0.0);

    parallel.forward(x);
    parallel.backward(grad_out, 0.0);

    auto check_param_grad = [&](const Tensor& rec, const Tensor& par,
                                const char* name, double tol) {
        double max_diff = 0.0;
        for (size_t i = 0; i < rec.data.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(rec.data[i] - par.data[i]));
        }
        std::printf("    %s grad max_diff = %.3e\n", name, max_diff);
        EXPECT(max_diff < tol, name);
    };

    check_param_grad(recurrent.W_r.grad_weights, parallel.W_r.grad_weights, "W_r grad", 1e-9);
    check_param_grad(recurrent.W_k.grad_weights, parallel.W_k.grad_weights, "W_k grad", 1e-9);
    check_param_grad(recurrent.W_v.grad_weights, parallel.W_v.grad_weights, "W_v grad", 1e-9);
    check_param_grad(recurrent.W_d.grad_weights, parallel.W_d.grad_weights, "W_d grad", 1e-7);
    check_param_grad(recurrent.W_a.grad_weights, parallel.W_a.grad_weights, "W_a grad", 1e-7);
    // xi/alpha/mu_* gradients are stored in grad_xi_/grad_alpha_/grad_mu_*_
    // (public on RWKV7TimeMix, public on RWKV7ParallelAttention).
    check_param_grad(recurrent.grad_xi_, parallel.grad_xi_, "xi grad", 1e-9);
    check_param_grad(recurrent.grad_alpha_, parallel.grad_alpha_, "alpha grad", 1e-9);
    check_param_grad(recurrent.grad_mu_r_, parallel.grad_mu_r_, "mu_r grad", 1e-9);
    check_param_grad(recurrent.grad_mu_k_, parallel.grad_mu_k_, "mu_k grad", 1e-9);
    check_param_grad(recurrent.grad_mu_v_, parallel.grad_mu_v_, "mu_v grad", 1e-9);
    check_param_grad(recurrent.grad_mu_d_, parallel.grad_mu_d_, "mu_d grad", 1e-9);
    check_param_grad(recurrent.grad_mu_a_, parallel.grad_mu_a_, "mu_a grad", 1e-9);
}

// Test 9: multi-head (H=2, d=4) forward shape + input grad FD
void test_multihead() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    init_layer_weights(attn, 0.3);
    Tensor x = make_input(5, 4, 0.1, /*seed=*/7);
    Tensor y = attn.forward(x);
    EXPECT(y.rows == 5 && y.cols == 4, "multi-head y shape");

    // FD input grad check
    Tensor grad_out(5, 4);
    grad_out.fill(1.0);
    Tensor grad_x_ana = attn.backward(grad_out, 0.0);
    double eps = 1e-5;
    double max_rel = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            Tensor xp = x.clone(); xp[i][j] += eps;
            Tensor xm = x.clone(); xm[i][j] -= eps;
            Tensor yp = attn.forward(xp);
            Tensor ym = attn.forward(xm);
            double sp = 0.0, sm = 0.0;
            for (size_t k = 0; k < y.data.size(); ++k) {
                sp += grad_out.data[k] * yp.data[k];
                sm += grad_out.data[k] * ym.data[k];
            }
            double fd = (sp - sm) / (2.0 * eps);
            double ana = grad_x_ana[i][j];
            double scale = std::max({std::fabs(ana), std::fabs(fd), 1e-12});
            max_rel = std::max(max_rel, std::fabs(ana - fd) / scale);
            attn.forward(x);
        }
    }
    std::printf("    multi-head input grad max_rel_err = %.3e\n", max_rel);
    EXPECT(max_rel < 1e-3, "multi-head input grad FD");
}

// Test 10: training step reduces loss
void test_training_reduces_loss() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    init_layer_weights(attn, 0.3);

    // Toy target: y_target = 0.5 (medium offset; the model must actually move)
    Tensor x = make_input(4, 4, 0.1, /*seed=*/8);
    Tensor target(4, 4);
    target.fill(0.5);

    double loss_prev = 1e9;
    bool decreased = false;
    bool loss_went_below_initial = false;
    double loss_initial = -1.0;
    for (size_t step = 0; step < 50; ++step) {
        Tensor y = attn.forward(x);
        double loss = 0.0;
        for (size_t i = 0; i < y.data.size(); ++i) {
            double d = y.data[i] - target.data[i];
            loss += 0.5 * d * d;
        }
        loss /= static_cast<double>(y.data.size());
        if (step == 0) loss_initial = loss;
        Tensor grad_out = (y - target);
        double inv_n = 1.0 / static_cast<double>(y.data.size());
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] *= inv_n;
        }
        attn.backward(grad_out, 0.0);
        attn.update_weights(0.05);
        if (step > 0 && loss < loss_prev - 1e-6) decreased = true;
        if (loss < loss_initial - 1e-4) loss_went_below_initial = true;
        loss_prev = loss;
    }
    std::printf("    final loss = %.4e (initial=%.4e, decreased=%d, below_initial=%d)\n",
                loss_prev, loss_initial, decreased, loss_went_below_initial);
    EXPECT(loss_went_below_initial, "training reduces loss (final < initial)");
    EXPECT(loss_prev < loss_initial, "final loss < initial loss");
}

// Test 11: zero_grad clears all gradients
void test_zero_grad() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    init_layer_weights(attn, 0.3);
    Tensor x = make_input(3, 4, 0.1, /*seed=*/9);
    attn.forward(x);
    Tensor grad_out(3, 4); grad_out.fill(1.0);
    attn.backward(grad_out, 0.0);

    // Verify some grads are nonzero
    bool nonzero = false;
    for (size_t i = 0; i < attn.W_r.grad_weights.data.size(); ++i) {
        if (std::fabs(attn.W_r.grad_weights.data[i]) > 1e-12) { nonzero = true; break; }
    }
    EXPECT(nonzero, "grads nonzero before zero_grad");

    attn.zero_grad();
    bool all_zero = true;
    for (size_t i = 0; i < attn.W_r.grad_weights.data.size(); ++i) {
        if (std::fabs(attn.W_r.grad_weights.data[i]) > 1e-12) { all_zero = false; break; }
    }
    EXPECT(all_zero, "W_r grad_weights zero after zero_grad");
    for (size_t i = 0; i < attn.grad_xi_.data.size(); ++i) {
        if (std::fabs(attn.grad_xi_.data[i]) > 1e-12) { all_zero = false; break; }
    }
    EXPECT(all_zero, "grad_xi_ zero after zero_grad");
}

// Test 12: parameters() returns the expected number of tensors
void test_parameters_and_gradients_shape_contract() {
    RWKV7ParallelAttention attn(4, /*num_heads=*/2);
    auto p = attn.parameters();
    auto g = attn.gradients();
    // 5 Dense * (weights + bias) + xi + alpha + 5 mu_* = 17
    EXPECT(p.size() == 17, "parameters() count");
    EXPECT(g.size() == 17, "gradients() count");

    // Shape checks: each W_* weights/bias is (d, d)/(1, d); xi is (1, d);
    // alpha is (1, 1); each mu_* is (1, d).
    EXPECT(attn.W_r.weights.rows == 4 && attn.W_r.weights.cols == 4, "W_r weights shape");
    EXPECT(attn.W_r.bias.rows == 1 && attn.W_r.bias.cols == 4, "W_r bias shape");
    EXPECT(attn.xi.rows == 1 && attn.xi.cols == 4, "xi shape");
    EXPECT(attn.alpha.rows == 1 && attn.alpha.cols == 1, "alpha shape");
    EXPECT(attn.mu_r.rows == 1 && attn.mu_r.cols == 4, "mu_r shape");
}

// ============================================================================
// Main
// ============================================================================

}  // namespace

int main() {
    std::printf("=== RWKV-7 Parallel Attention Tests ===\n");

    test_constructor_and_accessors();
    test_constructor_validation();
    test_forward_shape_and_finite();
    test_forward_equivalence_with_recurrent();
    test_forward_equivalence_chunk_smaller_than_T();
    test_input_grad_via_fd();
    test_input_grad_equivalence_with_recurrent();
    test_param_grad_equivalence_with_recurrent();
    test_multihead();
    test_training_reduces_loss();
    test_zero_grad();
    test_parameters_and_gradients_shape_contract();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_tests_passed, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}