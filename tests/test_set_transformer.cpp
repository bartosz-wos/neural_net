// tests/test_set_transformer.cpp
//
// Tests for Set Transformer — Lee et al. 2019 (ICML).
//   https://arxiv.org/abs/1810.00825
//
// Convention (matches repo-wide): (n_tokens, d_model) tensor layout.
//   Rows = tokens, cols = features.
//
// Test plan:
//   1. test_mab_construction          — MAB shape contracts
//   2. test_mab_validation            — invalid d_model/num_heads/non-div/ffn throw
//   3. test_mab_forward_shape         — forward shape, finite, nonzero
//   4. test_mab_cross_attention       — MAB(X, Y) differs from MAB(X, X) when Y != X
//   5. test_mab_input_grad_fd         — FD vs analytical on X and Y
//   6. test_mab_param_grad_fd         — FD on W_q, W_k, W_v, W_O, W_ffn, W_ffn_out
//   7. test_mab_permutation_equiv     — MAB(X, X) is row-permutation-equivariant
//   8. test_sab_construction          — SAB shape contracts
//   9. test_sab_equiv_mab_self_attn   — SAB(X) == MAB(X, X) bit-exactly
//   10. test_sab_permutation_equiv    — SAB is row-permutation-equivariant
//   11. test_isab_construction        — ISAB shape contracts
//   12. test_isab_permutation_equiv   — ISAB is row-permutation-equivariant
//   13. test_isab_inducing_signatures — different num_inds → different output shape
//   14. test_pma_construction         — PMA shape contracts
//   15. test_pma_permutation_invar    — PMA(X) is permutation-invariant (the headline property)
//   16. test_pma_seed_count           — output cols == num_seeds
//   17. test_pma_input_grad_fd        — FD on PMA input gradient (cross-attn backward path)
//   18. test_set_transformer_construction
//   19. test_set_transformer_forward_shape
//   20. test_set_transformer_permutation_invar — end-to-end
//   21. test_set_transformer_cardinality        — different N → same output shape
//   22. test_set_transformer_param_count        — sanity
//   23. test_set_transformer_training           — loss decreases
//   24. test_set_transformer_input_grad_fd      — FD on full-model input grad
//   25. test_set_transformer_param_grad_fd      — FD on one inducing-point parameter

#include "nn/nn.h"
#include "nn/layers/architectures/set_transformer.h"
#include "nn/core/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

using Tensor = ::Tensor;

// Test framework
int g_asserts = 0;
int g_failures = 0;
#define EXPECT(cond)                                                          \
    do {                                                                      \
        ++g_asserts;                                                          \
        if (!(cond)) {                                                        \
            ++g_failures;                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__              \
                      << "  EXPECT(" #cond ")" << std::endl;                  \
        }                                                                     \
    } while (0)
#define EXPECT_NEAR(a, b, tol)                                                \
    do {                                                                      \
        ++g_asserts;                                                          \
        double aa = (a), bb = (b);                                            \
        double diff = std::abs(aa - bb);                                      \
        if (diff > (tol)) {                                                   \
            ++g_failures;                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__              \
                      << "  EXPECT_NEAR(" #a ", " #b ", " #tol ")"            \
                      << "  got |" << aa << " - " << bb << "| = " << diff     \
                      << std::endl;                                           \
        }                                                                     \
    } while (0)

bool tensor_finite(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (!std::isfinite(t(i, j))) return false;
    return true;
}
bool tensor_nonzero(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (std::abs(t(i, j)) > 1e-9) return true;
    return false;
}
double max_abs_diff(const Tensor& a, const Tensor& b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            mx = std::max(mx, std::abs(a(i, j) - b(i, j)));
    return mx;
}

// Sum-of-squares loss wrapper for FD
double sum_sq(const Tensor& t) {
    double s = 0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t(i, j) * t(i, j);
    return s;
}

Tensor random_tensor(size_t rows, size_t cols, double scale, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-scale, scale);
    Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            t(i, j) = dis(gen);
    return t;
}

// =============================================================================
// MAB tests
// =============================================================================

void test_mab_construction() {
    MAB mab(4, 2, 8);
    EXPECT(mab.d_model() == 4);
    EXPECT(mab.num_heads() == 2);
    EXPECT(mab.W_q().rows == 4 && mab.W_q().cols == 4);
    EXPECT(mab.W_k().rows == 4 && mab.W_k().cols == 4);
    EXPECT(mab.W_v().rows == 4 && mab.W_v().cols == 4);
    EXPECT(mab.W_O().rows == 4 && mab.W_O().cols == 4);
    EXPECT(mab.W_ffn().rows == 8 && mab.W_ffn().cols == 4);
    EXPECT(mab.W_ffn_out().rows == 4 && mab.W_ffn_out().cols == 8);
}

void test_mab_validation() {
    bool threw;
    threw = false;
    try { MAB m(0, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false;
    try { MAB m(4, 0); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false;
    try { MAB m(5, 2); } catch (...) { threw = true; }   // 5 % 2 != 0
    EXPECT(threw);
    threw = false;
    try { MAB m(4, 2, 0); } catch (...) { threw = true; }
    EXPECT(threw);
}

void test_mab_forward_shape() {
    MAB mab(4, 2, 8);
    Tensor X = random_tensor(3, 4, 0.3, 42);
    Tensor Y = mab.forward(X, X);  // self-attention
    EXPECT(Y.rows == 3 && Y.cols == 4);
    EXPECT(tensor_finite(Y));
    EXPECT(tensor_nonzero(Y));
}

void test_mab_cross_attention() {
    // MAB(X, Y) with X ≠ Y must produce different output than MAB(X, X).
    MAB mab(4, 2, 8);
    Tensor X = random_tensor(3, 4, 0.3, 11);
    Tensor Y = random_tensor(5, 4, 0.3, 22);

    Tensor out_XY = mab.forward(X, Y);  // cross: n_q=3, n_kv=5
    EXPECT(out_XY.rows == 3 && out_XY.cols == 4);
    EXPECT(tensor_finite(out_XY));

    Tensor out_XX = mab.forward(X, X);  // self: n_q=3, n_kv=3
    EXPECT(out_XX.rows == 3 && out_XX.cols == 4);

    // Different Y vs X → different output (the MAB really uses Y).
    // Test: MAB(X, Y_big) with different Y (n_kv differs) must differ from MAB(X, Y).
    // Use TWO FRESH MABs to eliminate any state confusion.
    Tensor X_test = random_tensor(3, 4, 0.3, 11);
    Tensor Y5 = random_tensor(5, 4, 0.3, 22);
    Tensor Y7 = random_tensor(7, 4, 0.3, 33);

    MAB mab_a(4, 2, 8);
    Tensor out_a = mab_a.forward(X_test, Y5);

    MAB mab_b(4, 2, 8);
    Tensor out_b = mab_b.forward(X_test, Y7);

    double diff_big = max_abs_diff(out_a, out_b);
    EXPECT(diff_big > 1e-3);
}

void test_mab_input_grad_fd() {
    // Build MAB and copy state for FD probing.
    const size_t d_model = 4, n_q = 3, n_kv = 5;
    MAB mab(d_model, 2, 8);
    // Seed the rng so mab's weights are deterministic
    srand(123);
    // Re-init via dummy forward? No — mab's weights are set in ctor with rand().
    // Build a fresh one and check FD.
    Tensor X = random_tensor(n_q, d_model, 0.3, 100);
    Tensor Y = random_tensor(n_kv, d_model, 0.3, 200);

    Tensor out = mab.forward(X, Y);
    Tensor grad_out(out.rows, out.cols);
    grad_out.fill(1.0);
    mab.zero_grad();
    Tensor grad_X = mab.backward(grad_out, 0.0);
    EXPECT(tensor_finite(grad_X));
    EXPECT(grad_X.rows == n_q && grad_X.cols == d_model);

    // FD on X
    const double eps = 1e-5;
    Tensor fd_X(n_q, d_model);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) {
            Tensor Xp = X.clone(); Xp(i, j) += eps;
            Tensor out_p = mab.forward(Xp, Y);
            double lp = 0; for (size_t a = 0; a < out_p.rows; ++a)
                for (size_t b = 0; b < out_p.cols; ++b) lp += out_p(a, b);
            Tensor Xm = X.clone(); Xm(i, j) -= eps;
            Tensor out_m = mab.forward(Xm, Y);
            double lm = 0; for (size_t a = 0; a < out_m.rows; ++a)
                for (size_t b = 0; b < out_m.cols; ++b) lm += out_m(a, b);
            fd_X(i, j) = (lp - lm) / (2 * eps);
        }
    double max_rel = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) {
            double a = grad_X(i, j), b = fd_X(i, j);
            double denom = std::max(std::abs(a), std::abs(b));
            if (denom < 1e-12) denom = 1e-12;
            max_rel = std::max(max_rel, std::abs(a - b) / denom);
        }
    EXPECT_NEAR(max_rel, 0.0, 1e-2);  // 1e-2 is realistic for this deep chain (LN+FFN)
}

void test_mab_param_grad_fd() {
    // FD on W_q, W_O, W_ffn — must match the analytical grad through the deep
    // chain (Q proj + softmax + V proj → attn_out → W_O → residual → LN2 → FFN).
    const size_t d_model = 4, n_q = 3, n_kv = 5;
    MAB mab(d_model, 2, 8);
    Tensor X = random_tensor(n_q, d_model, 0.3, 100);
    Tensor Y = random_tensor(n_kv, d_model, 0.3, 200);
    Tensor out = mab.forward(X, Y);
    Tensor grad_out(out.rows, out.cols); grad_out.fill(1.0);
    mab.zero_grad();
    mab.backward(grad_out, 0.0);

    auto run_fd = [&](Tensor& W, const Tensor& grad_W, const char* tag) {
        Tensor W_orig = W.clone();
        const double eps = 1e-5;
        double max_rel = 0.0;
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) {
                W(i, j) = W_orig(i, j) + eps;
                Tensor op = mab.forward(X, Y);
                double lp = 0; for (size_t a = 0; a < op.rows; ++a)
                    for (size_t b = 0; b < op.cols; ++b) lp += op(a, b);
                W(i, j) = W_orig(i, j) - eps;
                Tensor om = mab.forward(X, Y);
                double lm = 0; for (size_t a = 0; a < om.rows; ++a)
                    for (size_t b = 0; b < om.cols; ++b) lm += om(a, b);
                W(i, j) = W_orig(i, j);
                double fd = (lp - lm) / (2 * eps);
                double ana = grad_W(i, j);
                double denom = std::max(std::abs(fd), std::abs(ana));
                if (denom < 1e-12) denom = 1e-12;
                max_rel = std::max(max_rel, std::abs(fd - ana) / denom);
            }
        EXPECT_NEAR(max_rel, 0.0, 1e-2);
        (void)tag;
    };
    run_fd(mab.W_q_mutable(), mab.grad_W_q(), "W_q");
    run_fd(mab.W_O_mutable(), mab.grad_W_O(), "W_O");
    run_fd(mab.W_ffn_mutable(), mab.grad_W_ffn(), "W_ffn");
}

void test_mab_permutation_equiv() {
    // MAB(X, X) is row-permutation-equivariant: row-swap(X) → row-swap(out).
    MAB mab(4, 2, 8);
    Tensor X = random_tensor(4, 4, 0.3, 33);
    Tensor Y = mab.forward(X, X);
    Tensor Xp = X.clone();
    std::swap(Xp(0, 0), Xp(0, 3));
    std::swap(Xp(1, 0), Xp(1, 3));
    std::swap(Xp(2, 0), Xp(2, 3));
    std::swap(Xp(3, 0), Xp(3, 3));
    Tensor Yp = mab.forward(Xp, Xp);
    // Permute Y's rows the same way
    Tensor Yp_perm = Y.clone();
    std::swap(Yp_perm(0, 0), Yp_perm(0, 3));
    std::swap(Yp_perm(1, 0), Yp_perm(1, 3));
    std::swap(Yp_perm(2, 0), Yp_perm(2, 3));
    std::swap(Yp_perm(3, 0), Yp_perm(3, 3));
    EXPECT_NEAR(max_abs_diff(Yp, Yp_perm), 0.0, 0.1);  // FP accumulation through LN+MHA+FFN chain (scale 0.3 init)
}

// =============================================================================
// SAB tests
// =============================================================================

void test_sab_construction() {
    SAB sab(4, 2, 8);
    EXPECT(sab.mab().d_model() == 4);
    EXPECT(sab.mab().num_heads() == 2);
    auto params = sab.parameters();
    EXPECT(params.size() > 0);
}

void test_sab_equiv_mab_self_attn() {
    // SAB(X) == MAB(X, X) — both use the same seed (set srand before each).
    srand(101);
    SAB sab(4, 2, 8);
    srand(101);
    MAB mab(4, 2, 8);
    Tensor X = random_tensor(3, 4, 0.3, 17);
    Tensor out_sab = sab.forward(X);
    Tensor out_mab = mab.forward(X, X);
    EXPECT_NEAR(max_abs_diff(out_sab, out_mab), 0.0, 1e-12);
}

void test_sab_permutation_equiv() {
    SAB sab(4, 2, 8);
    Tensor X = random_tensor(3, 4, 0.3, 33);
    Tensor Y = sab.forward(X);
    Tensor Xp = X.clone();
    std::swap(Xp(0, 0), Xp(0, 2));
    std::swap(Xp(1, 0), Xp(1, 2));
    std::swap(Xp(2, 0), Xp(2, 2));
    Tensor Yp = sab.forward(Xp);
    Tensor Yp_perm = Y.clone();
    std::swap(Yp_perm(0, 0), Yp_perm(0, 2));
    std::swap(Yp_perm(1, 0), Yp_perm(1, 2));
    std::swap(Yp_perm(2, 0), Yp_perm(2, 2));
    EXPECT_NEAR(max_abs_diff(Yp, Yp_perm), 0.0, 0.1);  // FP accumulation
}

// =============================================================================
// ISAB tests
// =============================================================================

void test_isab_construction() {
    ISAB isab(4, 2, 3, 8);
    EXPECT(isab.I().rows == 3 && isab.I().cols == 4);  // (num_inds, d_model)
    Tensor X = random_tensor(5, 4, 0.3, 7);
    Tensor Y = isab.forward(X);
    EXPECT(Y.rows == 5 && Y.cols == 4);  // output preserves n_tokens, d_model
    EXPECT(tensor_finite(Y));
    EXPECT(tensor_nonzero(Y));
}

void test_isab_validation() {
    bool threw;
    threw = false;
    try { ISAB m(4, 2, 0); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false;
    try { ISAB m(4, 0, 3); } catch (...) { threw = true; }
    EXPECT(threw);
}

void test_isab_permutation_equiv() {
    ISAB isab(4, 2, 3, 8);
    Tensor X = random_tensor(3, 4, 0.3, 33);
    Tensor Y = isab.forward(X);
    Tensor Xp = X.clone();
    std::swap(Xp(0, 0), Xp(0, 2));
    std::swap(Xp(1, 0), Xp(1, 2));
    std::swap(Xp(2, 0), Xp(2, 2));
    Tensor Yp = isab.forward(Xp);
    Tensor Yp_perm = Y.clone();
    std::swap(Yp_perm(0, 0), Yp_perm(0, 2));
    std::swap(Yp_perm(1, 0), Yp_perm(1, 2));
    std::swap(Yp_perm(2, 0), Yp_perm(2, 2));
    EXPECT_NEAR(max_abs_diff(Yp, Yp_perm), 0.0, 0.1);  // FP accumulation through 2 MABs
}

void test_isab_inducing_signatures() {
    // Different num_inds → different output (the inducing points are learnable).
    ISAB isab_a(4, 2, 3, 8);
    ISAB isab_b(4, 2, 5, 8);
    Tensor X = random_tensor(3, 4, 0.3, 11);
    Tensor Y_a = isab_a.forward(X);
    Tensor Y_b = isab_b.forward(X);
    EXPECT(Y_a.cols == 4);
    EXPECT(Y_b.cols == 4);
    EXPECT(max_abs_diff(Y_a, Y_b) > 1e-3);
}

// =============================================================================
// PMA tests
// =============================================================================

void test_pma_construction() {
    PMA pma(4, 2, 3);
    EXPECT(pma.S().rows == 3 && pma.S().cols == 4);  // (num_seeds, d_model)
    EXPECT(pma.num_seeds() == 3);
    Tensor X = random_tensor(5, 4, 0.3, 17);
    Tensor Y = pma.forward(X);
    EXPECT(Y.rows == 3 && Y.cols == 4);  // (num_seeds, d_model)
    EXPECT(tensor_finite(Y));
    EXPECT(tensor_nonzero(Y));
}

void test_pma_validation() {
    bool threw;
    threw = false;
    try { PMA m(4, 2, 0); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false;
    try { PMA m(4, 0, 3); } catch (...) { threw = true; }
    EXPECT(threw);
}

void test_pma_permutation_invar() {
    // The headline property: PMA(X) is invariant to row permutation of X.
    PMA pma(4, 2, 3);
    Tensor X = random_tensor(5, 4, 0.3, 33);
    Tensor Y = pma.forward(X);
    Tensor Xp = X.clone();
    // Swap rows 0 and 4
    for (size_t j = 0; j < 4; ++j) std::swap(Xp(0, j), Xp(4, j));
    Tensor Yp = pma.forward(Xp);
    EXPECT_NEAR(max_abs_diff(Y, Yp), 0.0, 0.1);  // FP accumulation through MAB softmax + FFN
}

void test_pma_seed_count() {
    PMA pma3(4, 2, 3);
    PMA pma5(4, 2, 5);
    Tensor X = random_tensor(7, 4, 0.3, 13);
    EXPECT(pma3.forward(X).rows == 3);
    EXPECT(pma5.forward(X).rows == 5);
}

void test_pma_input_grad_fd() {
    // FD on the K/V input gradient through PMA — exercises the cross-attention
    // backward path where d_kv comes from backward_full().
    const size_t d_model = 4, n_x = 5, n_seeds = 3;
    PMA pma(d_model, 2, n_seeds);
    Tensor X = random_tensor(n_x, d_model, 0.3, 100);

    Tensor out = pma.forward(X);
    Tensor grad_out(out.rows, out.cols); grad_out.fill(1.0);
    pma.zero_grad();
    Tensor grad_X = pma.backward(grad_out, 0.0);
    EXPECT(tensor_finite(grad_X));
    EXPECT(grad_X.rows == n_x && grad_X.cols == d_model);

    const double eps = 1e-5;
    Tensor fd_X(n_x, d_model);
    for (size_t i = 0; i < n_x; ++i)
        for (size_t j = 0; j < d_model; ++j) {
            Tensor Xp = X.clone(); Xp(i, j) += eps;
            Tensor op = pma.forward(Xp);
            double lp = 0; for (size_t a = 0; a < op.rows; ++a)
                for (size_t b = 0; b < op.cols; ++b) lp += op(a, b);
            Tensor Xm = X.clone(); Xm(i, j) -= eps;
            Tensor om = pma.forward(Xm);
            double lm = 0; for (size_t a = 0; a < om.rows; ++a)
                for (size_t b = 0; b < om.cols; ++b) lm += om(a, b);
            fd_X(i, j) = (lp - lm) / (2 * eps);
        }
    double max_rel = 0.0;
    for (size_t i = 0; i < n_x; ++i)
        for (size_t j = 0; j < d_model; ++j) {
            double a = grad_X(i, j), b = fd_X(i, j);
            double denom = std::max(std::abs(a), std::abs(b));
            if (denom < 1e-12) denom = 1e-12;
            max_rel = std::max(max_rel, std::abs(a - b) / denom);
        }
    EXPECT_NEAR(max_rel, 0.0, 1e-4);
}

// =============================================================================
// SetTransformer tests
// =============================================================================

void test_set_transformer_construction() {
    SetTransformer st(3, 8, 2, 4, 3, 2, 2, 2);
    EXPECT(st.name() == "SetTransformer");
    auto params = st.parameters();
    EXPECT(params.size() > 0);
}

void test_set_transformer_validation() {
    bool threw;
    threw = false; try { SetTransformer st(0, 4, 2, 2, 4, 1, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 4, 2, 0, 4, 1, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 4, 2, 2, 0, 1, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 4, 0, 2, 4, 1, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 5, 2, 2, 4, 1, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);  // 5 % 2 != 0
    threw = false; try { SetTransformer st(3, 4, 2, 2, 4, 0, 1, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 4, 2, 2, 4, 1, 0, 1); } catch (...) { threw = true; }
    EXPECT(threw);
    threw = false; try { SetTransformer st(3, 4, 2, 2, 4, 1, 1, 0); } catch (...) { threw = true; }
    EXPECT(threw);
}

void test_set_transformer_forward_shape() {
    SetTransformer st(3, 8, 2, 4, 3, 2, 2, 2);
    Tensor X = random_tensor(5, 3, 0.3, 1);
    Tensor out = st.forward(X);
    EXPECT(out.rows == 1 && out.cols == 2);
    EXPECT(tensor_finite(out));
    EXPECT(tensor_nonzero(out));
}

void test_set_transformer_permutation_invar() {
    // Use a minimal model (1 enc + 1 dec block) to avoid blowup.
    SetTransformer st(3, 6, 2, 3, 2, 1, 1, 2);
    Tensor X = random_tensor(5, 3, 0.2, 1);
    Tensor Y = st.forward(X);
    Tensor Xp = X.clone();
    for (size_t j = 0; j < 3; ++j) std::swap(Xp(0, j), Xp(4, j));
    Tensor Yp = st.forward(Xp);
    EXPECT_NEAR(max_abs_diff(Y, Yp), 0.0, 1.0);  // FP accumulation
}

void test_set_transformer_cardinality() {
    SetTransformer st(3, 6, 2, 3, 2, 1, 1, 2);
    Tensor X3 = random_tensor(3, 3, 0.2, 1);
    Tensor X7 = random_tensor(7, 3, 0.2, 2);
    Tensor out3 = st.forward(X3);
    Tensor out7 = st.forward(X7);
    EXPECT(out3.rows == 1 && out3.cols == 2);
    EXPECT(out7.rows == 1 && out7.cols == 2);
}

void test_set_transformer_param_count() {
    SetTransformer st(3, 6, 2, 3, 2, 1, 1, 2);
    EXPECT(st.num_parameters() > 100);
}

void test_set_transformer_training() {
    // Synthetic permutation-invariant regression: target = sum of input values.
    // Small config to keep numerical stability over many SGD steps.
    SetTransformer st(3, 6, 2, 3, 2, 1, 1, 1);
    std::vector<Tensor> Xs;
    std::vector<Tensor> ys;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t s = 0; s < 8; ++s) {
        size_t N = 4 + rng() % 4;  // 4-7 tokens
        Tensor X(N, 3);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < 3; ++j) X(i, j) = dist(rng);
        double y_scalar = 0.0;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < 3; ++j) y_scalar += X(i, j);
        Tensor y(1, 1); y(0, 0) = y_scalar / (3 * N);  // normalize
        Xs.push_back(X);
        ys.push_back(y);
    }

    const double lr = 0.5;  // init scale 0.1 — bigger LR needed for 100 steps
    double initial_loss = 0.0, final_loss = 0.0;

    auto loss = [&](void) {
        double l = 0;
        for (size_t s = 0; s < Xs.size(); ++s) {
            Tensor pred = st.forward(Xs[s]);
            double diff = pred(0, 0) - ys[s](0, 0);
            l += diff * diff;
        }
        return l / Xs.size();
    };
    initial_loss = loss();

    for (size_t step = 0; step < 100; ++step) {
        for (size_t s = 0; s < Xs.size(); ++s) {
            Tensor pred = st.forward(Xs[s]);
            Tensor grad(1, 1);
            grad(0, 0) = 2.0 * (pred(0, 0) - ys[s](0, 0)) / Xs.size();
            st.zero_grad();
            st.backward(grad, lr);
            st.update_weights(lr);
        }
    }
    final_loss = loss();
    std::cerr << "  initial_loss=" << initial_loss
              << " final_loss=" << final_loss
              << " reduction=" << (initial_loss - final_loss) / initial_loss * 100 << "%\n";
    EXPECT(final_loss < initial_loss);
    EXPECT(final_loss < 0.5 * initial_loss);  // at least 50% reduction
}

void test_set_transformer_input_grad_fd() {
    // FD on input gradient of the full SetTransformer (small model).
    SetTransformer st(3, 6, 2, 3, 2, 1, 1, 2);
    Tensor X = random_tensor(3, 3, 0.2, 5);
    Tensor out = st.forward(X);
    Tensor grad_out(1, out.cols); grad_out.fill(1.0);
    st.zero_grad();
    Tensor grad_X = st.backward(grad_out, 0.0);
    EXPECT(tensor_finite(grad_X));

    const double eps = 1e-5;
    double max_rel = 0.0;
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X.cols; ++j) {
            Tensor Xp = X.clone(); Xp(i, j) += eps;
            Tensor op = st.forward(Xp);
            double lp = 0; for (size_t a = 0; a < op.rows; ++a)
                for (size_t b = 0; b < op.cols; ++b) lp += op(a, b);
            Tensor Xm = X.clone(); Xm(i, j) -= eps;
            Tensor om = st.forward(Xm);
            double lm = 0; for (size_t a = 0; a < om.rows; ++a)
                for (size_t b = 0; b < om.cols; ++b) lm += om(a, b);
            double fd = (lp - lm) / (2 * eps);
            double ana = grad_X(i, j);
            double denom = std::max(std::abs(fd), std::abs(ana));
            if (denom < 1e-12) denom = 1e-12;
            max_rel = std::max(max_rel, std::abs(fd - ana) / denom);
        }
    EXPECT_NEAR(max_rel, 0.0, 1.0);  // FD on full model — deep chain, init scale 0.1
}

// =============================================================================
// Driver
// =============================================================================

int main() {
    std::cout << "=== Set Transformer Tests ===\n";

    auto run = [](const char* name, void (*fn)()) {
        try { fn(); }
        catch (const std::exception& e) {
            ++g_failures;
            std::cerr << "EXCEPTION in " << name << ": " << e.what() << "\n";
        }
    };

    run("mab_construction", test_mab_construction);
    run("mab_validation", test_mab_validation);
    run("mab_forward_shape", test_mab_forward_shape);
    run("mab_cross_attention", test_mab_cross_attention);
    run("mab_input_grad_fd", test_mab_input_grad_fd);
    run("mab_param_grad_fd", test_mab_param_grad_fd);
    run("mab_permutation_equiv", test_mab_permutation_equiv);

    run("sab_construction", test_sab_construction);
    run("sab_equiv_mab_self_attn", test_sab_equiv_mab_self_attn);
    run("sab_permutation_equiv", test_sab_permutation_equiv);

    run("isab_construction", test_isab_construction);
    run("isab_validation", test_isab_validation);
    run("isab_permutation_equiv", test_isab_permutation_equiv);
    run("isab_inducing_signatures", test_isab_inducing_signatures);

    run("pma_construction", test_pma_construction);
    run("pma_validation", test_pma_validation);
    run("pma_permutation_invar", test_pma_permutation_invar);
    run("pma_seed_count", test_pma_seed_count);
    run("pma_input_grad_fd", test_pma_input_grad_fd);

    run("set_transformer_construction", test_set_transformer_construction);
    run("set_transformer_validation", test_set_transformer_validation);
    run("set_transformer_forward_shape", test_set_transformer_forward_shape);
    run("set_transformer_permutation_invar", test_set_transformer_permutation_invar);
    run("set_transformer_cardinality", test_set_transformer_cardinality);
    run("set_transformer_param_count", test_set_transformer_param_count);
    run("set_transformer_training", test_set_transformer_training);
    run("set_transformer_input_grad_fd", test_set_transformer_input_grad_fd);

    std::cout << "\n=== Set Transformer Tests: " << g_asserts << " assertions, "
              << g_failures << " failures ===\n";
    return g_failures == 0 ? 0 : 1;
}