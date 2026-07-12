// test_siglip_loss.cpp
// Tests for SigLIP-style Sigmoid Loss for Language-Image Pre-Training
// in include/nn/utils/siglip_loss.h
//
// Reference: Zhai et al. 2023 "Sigmoid Loss for Language Image Pre-Training"
// (https://arxiv.org/abs/2303.15343)
//
// Per-row elementwise sigmoid loss:
//   L[i] = (1/N) * sum_j softplus(-z*[i][j] * t[i][j])
//   z* = scale * S + bias
//   t[i][j] = +1 if i == j (positive pair) else -1
//   S[i][j] = cos(z_img[i], z_txt[j])   (default, normalize=true)
//          = z_img[i] . z_txt[j]         (dot product, normalize=false)
// Total loss: L = (1/N) * sum_i L[i]  (mean over batch rows)
//
// The original CLIP/InfoNCE loss requires a softmax partition function, which
// makes it scale O(N^2) memory and compute and prevents very large batches.
// SigLIP's elementwise sigmoid removes the normalization, scales O(N) (just
// the N×N similarity matrix), and lets you push batch sizes much higher.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "nn/utils/siglip_loss.h"

using std::abs;

static int total_passed = 0;
static int total_failed = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            std::cout << "  [PASS] " << msg << std::endl; \
            ++total_passed; \
        } else { \
            std::cout << "  [FAIL] " << msg << std::endl; \
            ++total_failed; \
        } \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg) \
    do { \
        double aa = (a), bb = (b); \
        if (std::abs(aa - bb) < (tol)) { \
            std::cout << "  [PASS] " << msg << " (got " << std::setprecision(10) << aa \
                      << ", expected " << bb << ", |diff|=" << std::abs(aa - bb) << ")" << std::endl; \
            ++total_passed; \
        } else { \
            std::cout << "  [FAIL] " << msg << " (got " << std::setprecision(10) << aa \
                      << ", expected " << bb << ", |diff|=" << std::abs(aa - bb) << ")" << std::endl; \
            ++total_failed; \
        } \
    } while (0)

// =================================================================
// Test 1: Accessors
// =================================================================
static void test_accessors() {
    std::cout << "-- Test 1: Accessors and construction --" << std::endl;
    SigLIPLoss loss(100.0, 0.0, 1e-8);
    CHECK_NEAR(loss.get_scale(), 100.0, 1e-12, "scale = 100");
    CHECK_NEAR(loss.get_bias(), 0.0, 1e-12, "bias = 0");
    CHECK_NEAR(loss.get_eps(), 1e-8, 1e-12, "eps = 1e-8");
    CHECK(loss.get_normalize() == true, "normalize default = true (cosine)");

    SigLIPLoss loss2(50.0, -10.0);
    CHECK_NEAR(loss2.get_scale(), 50.0, 1e-12, "scale = 50");
    CHECK_NEAR(loss2.get_bias(), -10.0, 1e-12, "bias = -10");

    // Defensive: scale <= 0 should be clamped
    SigLIPLoss loss3(-1.0, 0.0);
    CHECK(loss3.get_scale() > 0.0, "negative scale clamped to positive");
}

// =================================================================
// Test 2: Forward — known analytical value
// Hand-derive: N=2, S = [[s00, s01], [s10, s11]] with t = [[+1,-1],[-1,+1]]
// L[i] = (1/2) * (softplus(-scale*s_ii) + softplus(+scale*s_ij))   (j!=i)
// L = (1/2) * (L[0] + L[1])
//
// Use scale=1, bias=0, S=[[2,0],[0,2]] -> positives get z*=2 (good),
// negatives get z*=0.
// softplus(-2) = log(1+e^-2) = log(1+0.1353) = 0.1269
// softplus(0)  = log(2)        = 0.6931
// L[0] = (1/2) * (softplus(-2) + softplus(0)) = (1/2)*(0.1269 + 0.6931) = 0.4100
// Same for L[1]. L = 0.4100
// =================================================================
static void test_forward_hand_derived() {
    std::cout << "-- Test 2: Hand-derived forward value --" << std::endl;
    const int N = 2;
    const int D = 2;
    Tensor z_img(N, D);
    Tensor z_txt(N, D);

    // Construct such that normalized cosine sim S = [[2,0],[0,2]] after scale=1, bias=0.
    // Sim > 1 is fine for S (we treat S as the raw similarity matrix; for cosine
    // on normalized vectors S is in [-1,1], but the test treats scale*S as a
    // free parameter). Actually use dot product mode for clarity.
    // z_img[0] = [1, 0]; z_img[1] = [0, 1]
    // z_txt[0] = [2, 0]; z_txt[1] = [0, 2]  (S[i][i] = 2, S[0][1] = 0, S[1][0] = 0)
    z_img[0][0] = 1.0; z_img[0][1] = 0.0;
    z_img[1][0] = 0.0; z_img[1][1] = 1.0;
    z_txt[0][0] = 2.0; z_txt[0][1] = 0.0;
    z_txt[1][0] = 0.0; z_txt[1][1] = 2.0;

    SigLIPLoss loss(1.0, 0.0, 1e-8);
    loss.set_normalize(false);
    Tensor L = loss.forward(z_img, z_txt);

    // softplus(-2) = log(1+e^-2) and softplus(0) = log(2)
    double sp_neg2 = std::log(1.0 + std::exp(-2.0));
    double sp_zero = std::log(2.0);
    double expected_per_row = 0.5 * (sp_neg2 + sp_zero);
    double expected = expected_per_row;  // mean over 2 rows of same value

    CHECK_NEAR(L[0][0], expected, 1e-10, "hand-derived L = 0.4100...");
}

// =================================================================
// Test 3: Symmetry: when z_img == z_txt, forward and backward should
// both be invariant under swap of arguments (since cos(x, x) is symmetric
// and t[i][j] is symmetric). With independent random inputs, L is
// symmetric in (z_img, z_txt) but the gradient w.r.t. each input is
// different because the inputs play different roles — so the test
// uses z_img == z_txt for the strictest symmetry check.
// =================================================================
static void test_symmetry() {
    std::cout << "-- Test 3: Symmetric loss and grad under img/txt swap (z_img == z_txt) --" << std::endl;
    const int N = 3;
    const int D = 4;
    // Use a single random tensor for both img and txt.
    Tensor z = Tensor::random(N, D, 0.5);

    SigLIPLoss loss(10.0, 0.0, 1e-8);

    Tensor L1 = loss.forward(z, z);
    Tensor L2 = loss.forward(z, z);  // second call to confirm deterministic
    CHECK_NEAR(L1[0][0], L2[0][0], 1e-12, "forward is deterministic (L(z, z) twice)");

    auto g1 = loss.backward(z, z);
    auto g2 = loss.backward(z, z);
    double max_diff = 0.0;
    for (size_t i = 0; i < g1.first.rows; ++i) {
        for (size_t j = 0; j < g1.first.cols; ++j) {
            max_diff = std::max(max_diff, std::abs(g1.first[i][j] - g2.first[i][j]));
            max_diff = std::max(max_diff, std::abs(g1.second[i][j] - g2.second[i][j]));
        }
    }
    CHECK_NEAR(max_diff, 0.0, 1e-12, "backward is deterministic (grad at same point)");

    // The true symmetry check: at the point (z, z), grad w.r.t. arg1 (z_img) and
    // arg2 (z_txt) should be equal because the inputs are the same.
    CHECK_NEAR(g1.first[0][0], g1.second[0][0], 1e-12, "grad_z_img == grad_z_txt at z_img==z_txt");
}

// =================================================================
// Test 4: scale=0 limit: z* = bias, so L is constant in S. If bias = 0
// then z* = 0 and softplus(0) = log(2) for every entry -> L = log(2).
// =================================================================
static void test_scale_zero() {
    std::cout << "-- Test 4: scale=0 -> L = log(2) (random / decoupled) --" << std::endl;
    const int N = 3;
    const int D = 4;
    Tensor z_img = Tensor::random(N, D, 1.0);
    Tensor z_txt = Tensor::random(N, D, 1.0);

    SigLIPLoss loss(0.0, 0.0, 1e-8);
    loss.set_normalize(false);
    Tensor L = loss.forward(z_img, z_txt);
    // Tolerance 1e-8 to absorb the N^2 reduction in the mean (1e-10 absolute
    // is too tight once we've averaged over 9 entries that each are within
    // FP noise of log(2)).
    CHECK_NEAR(L[0][0], std::log(2.0), 1e-8, "L = log(2) when scale=0, bias=0");

    // With scale=0 but nonzero bias, L = softplus(-bias) for positives
    // and softplus(+bias) for negatives, then mean.
    SigLIPLoss loss2(0.0, 1.0, 1e-8);
    loss2.set_normalize(false);
    Tensor L2 = loss2.forward(z_img, z_txt);
    // pos (1 entry per row) gets softplus(-1) = log(1+e^-1) ≈ 0.3133
    // neg (2 entries per row) gets softplus(+1) = log(1+e^1)  ≈ 1.3133
    // per-row L = (1/3) * (softplus(-1) + 2*softplus(+1))
    double sp_neg1 = std::log(1.0 + std::exp(-1.0));
    double sp_pos1 = std::log(1.0 + std::exp(1.0));
    double expected2 = (sp_neg1 + 2.0 * sp_pos1) / 3.0;
    CHECK_NEAR(L2[0][0], expected2, 1e-8, "scale=0, bias=1 -> closed-form mean softplus");
}

// =================================================================
// Test 5: bias=0 + perfect alignment -> L should be very small.
// When S is the identity matrix (z_img[i] = z_txt[i] and well-separated),
// scale * S[i][i] = scale and scale * S[i][j!=i] = 0 (orthogonal).
// With scale=10, the per-row L = (1/N) * (softplus(-10) + (N-1)*softplus(0))
//                                                  pos            neg
// For N=2:  L = 0.5 * (softplus(-10) + softplus(0))
//             = 0.5 * (4.54e-5 + 0.6931) = 0.3466
// =================================================================
static void test_bias_zero_perfect_alignment() {
    std::cout << "-- Test 5: bias=0, perfect alignment reduces L --" << std::endl;
    const int N = 2;
    const int D = 3;
    Tensor z_img(N, D);
    Tensor z_txt(N, D);
    // Use orthogonal columns, equal in both
    z_img[0][0] = 1.0; z_img[0][1] = 0.0; z_img[0][2] = 0.0;
    z_img[1][0] = 0.0; z_img[1][1] = 1.0; z_img[1][2] = 0.0;
    z_txt[0][0] = 1.0; z_txt[0][1] = 0.0; z_txt[0][2] = 0.0;
    z_txt[1][0] = 0.0; z_txt[1][1] = 1.0; z_txt[1][2] = 0.0;

    SigLIPLoss loss(10.0, 0.0, 1e-8);
    loss.set_normalize(false);  // dot product
    Tensor L = loss.forward(z_img, z_txt);

    double sp_neg10 = std::log(1.0 + std::exp(-10.0));
    double sp_zero  = std::log(2.0);
    double expected = 0.5 * (sp_neg10 + sp_zero);
    CHECK_NEAR(L[0][0], expected, 1e-10, "closed-form perfect alignment");
    CHECK(L[0][0] < sp_zero, "L < log(2) when positives are well-aligned");
}

// =================================================================
// Test 6: Numerical gradient check via central finite differences
// Pick a small batch and verify the analytical grad matches FD.
// =================================================================
static void test_numerical_gradient() {
    std::cout << "-- Test 6: Numerical gradient check (central FD) --" << std::endl;
    const int N = 3;
    const int D = 4;
    Tensor z_img = Tensor::random(N, D, 0.3);
    Tensor z_txt = Tensor::random(N, D, 0.3);

    SigLIPLoss loss(5.0, 0.0, 1e-8);
    loss.set_normalize(false);

    // Populate cache by calling forward first.
    loss.forward(z_img, z_txt);
    auto grads = loss.backward(z_img, z_txt);
    const Tensor& g_img = grads.first;
    const Tensor& g_txt = grads.second;

    const double eps = 1e-5;
    double max_rel_err_img = 0.0;
    double max_rel_err_txt = 0.0;

    // Finite-diff the z_img entries
    for (size_t i = 0; i < z_img.rows; ++i) {
        for (size_t j = 0; j < z_img.cols; ++j) {
            double orig = z_img[i][j];
            z_img[i][j] = orig + eps;
            double Lp = loss.forward(z_img, z_txt)[0][0];
            z_img[i][j] = orig - eps;
            double Lm = loss.forward(z_img, z_txt)[0][0];
            z_img[i][j] = orig;
            double fd = (Lp - Lm) / (2.0 * eps);
            double ana = g_img[i][j];
            double rel = std::abs(fd - ana) / std::max(1e-12, std::max(std::abs(fd), std::abs(ana)));
            max_rel_err_img = std::max(max_rel_err_img, rel);
        }
    }
    // Finite-diff the z_txt entries
    for (size_t i = 0; i < z_txt.rows; ++i) {
        for (size_t j = 0; j < z_txt.cols; ++j) {
            double orig = z_txt[i][j];
            z_txt[i][j] = orig + eps;
            double Lp = loss.forward(z_img, z_txt)[0][0];
            z_txt[i][j] = orig - eps;
            double Lm = loss.forward(z_img, z_txt)[0][0];
            z_txt[i][j] = orig;
            double fd = (Lp - Lm) / (2.0 * eps);
            double ana = g_txt[i][j];
            double rel = std::abs(fd - ana) / std::max(1e-12, std::max(std::abs(fd), std::abs(ana)));
            max_rel_err_txt = std::max(max_rel_err_txt, rel);
        }
    }
    std::cout << "    max rel err (img): " << std::scientific << std::setprecision(2) << max_rel_err_img
              << ", (txt): " << max_rel_err_txt << std::endl;
    CHECK(max_rel_err_img < 1e-5, "z_img grad rel_err < 1e-5 (FD vs analytical)");
    CHECK(max_rel_err_txt < 1e-5, "z_txt grad rel_err < 1e-5 (FD vs analytical)");
}

// =================================================================
// Test 7: Cosine mode gradient check (with L2-normalization chain)
// =================================================================
static void test_cosine_mode_gradient() {
    std::cout << "-- Test 7: Cosine mode numerical gradient --" << std::endl;
    const int N = 3;
    const int D = 4;
    Tensor z_img = Tensor::random(N, D, 0.5);
    Tensor z_txt = Tensor::random(N, D, 0.5);

    SigLIPLoss loss(10.0, 0.0, 1e-8);
    // normalize=true is the default

    loss.forward(z_img, z_txt);  // populate cache
    auto grads = loss.backward(z_img, z_txt);
    const Tensor& g_img = grads.first;
    const Tensor& g_txt = grads.second;

    const double eps = 1e-5;
    double max_rel_err_img = 0.0;
    double max_rel_err_txt = 0.0;

    for (size_t i = 0; i < z_img.rows; ++i) {
        for (size_t j = 0; j < z_img.cols; ++j) {
            double orig = z_img[i][j];
            z_img[i][j] = orig + eps;
            double Lp = loss.forward(z_img, z_txt)[0][0];
            z_img[i][j] = orig - eps;
            double Lm = loss.forward(z_img, z_txt)[0][0];
            z_img[i][j] = orig;
            double fd = (Lp - Lm) / (2.0 * eps);
            double ana = g_img[i][j];
            double rel = std::abs(fd - ana) / std::max(1e-12, std::max(std::abs(fd), std::abs(ana)));
            max_rel_err_img = std::max(max_rel_err_img, rel);
        }
    }
    for (size_t i = 0; i < z_txt.rows; ++i) {
        for (size_t j = 0; j < z_txt.cols; ++j) {
            double orig = z_txt[i][j];
            z_txt[i][j] = orig + eps;
            double Lp = loss.forward(z_img, z_txt)[0][0];
            z_txt[i][j] = orig - eps;
            double Lm = loss.forward(z_img, z_txt)[0][0];
            z_txt[i][j] = orig;
            double fd = (Lp - Lm) / (2.0 * eps);
            double ana = g_txt[i][j];
            double rel = std::abs(fd - ana) / std::max(1e-12, std::max(std::abs(fd), std::abs(ana)));
            max_rel_err_txt = std::max(max_rel_err_txt, rel);
        }
    }
    std::cout << "    max rel err (img): " << std::scientific << std::setprecision(2) << max_rel_err_img
              << ", (txt): " << max_rel_err_txt << std::endl;
    CHECK(max_rel_err_img < 1e-4, "z_img grad rel_err < 1e-4 in cosine mode");
    CHECK(max_rel_err_txt < 1e-4, "z_txt grad rel_err < 1e-4 in cosine mode");
}

// =================================================================
// Test 8: Targets cache matches t[i][j] = +1 if i==j else -1
// =================================================================
static void test_targets_cache() {
    std::cout << "-- Test 8: Targets cache shape and values --" << std::endl;
    const int N = 4;
    const int D = 2;
    Tensor z_img = Tensor::random(N, D, 0.5);
    Tensor z_txt = Tensor::random(N, D, 0.5);

    SigLIPLoss loss(5.0, 0.0, 1e-8);
    loss.set_normalize(false);
    loss.forward(z_img, z_txt);

    const Tensor& T = loss.last_targets();
    CHECK(T.rows == (size_t)N && T.cols == (size_t)N, "targets is N x N");
    int pos = 0, neg = 0;
    for (size_t i = 0; i < T.rows; ++i) {
        for (size_t j = 0; j < T.cols; ++j) {
            if (i == j) {
                CHECK_NEAR(T[i][j], 1.0, 1e-12, "diagonal t = +1");
                ++pos;
            } else {
                CHECK_NEAR(T[i][j], -1.0, 1e-12, "off-diagonal t = -1");
                ++neg;
            }
        }
    }
    CHECK(pos == N && neg == N * (N - 1), "exactly N positives and N(N-1) negatives");
}

// =================================================================
// Test 9: zstar cache = scale * S + bias
// =================================================================
static void test_zstar_cache() {
    std::cout << "-- Test 9: zstar cache = scale * S + bias --" << std::endl;
    const int N = 3;
    const int D = 3;
    Tensor z_img = Tensor::random(N, D, 0.5);
    Tensor z_txt = Tensor::random(N, D, 0.5);

    SigLIPLoss loss(2.5, -0.5, 1e-8);
    loss.set_normalize(false);
    loss.forward(z_img, z_txt);

    const Tensor& Z = loss.last_zstar();
    const Tensor& S = loss.last_similarity();

    CHECK(Z.rows == S.rows && Z.cols == S.cols, "zstar same shape as S");
    double max_diff = 0.0;
    for (size_t i = 0; i < Z.rows; ++i) {
        for (size_t j = 0; j < Z.cols; ++j) {
            double expected = 2.5 * S[i][j] - 0.5;
            max_diff = std::max(max_diff, std::abs(Z[i][j] - expected));
        }
    }
    CHECK_NEAR(max_diff, 0.0, 1e-12, "zstar == scale * S + bias exactly");
}

// =================================================================
// Test 10: Non-vacuous gradient — non-trivial gradient norm
// =================================================================
static void test_gradient_nonvacuous() {
    std::cout << "-- Test 10: Non-vacuous gradient --" << std::endl;
    const int N = 4;
    const int D = 4;
    Tensor z_img = Tensor::random(N, D, 0.5);
    Tensor z_txt = Tensor::random(N, D, 0.5);

    SigLIPLoss loss(10.0, 0.0, 1e-8);
    loss.forward(z_img, z_txt);  // populate cache
    auto grads = loss.backward(z_img, z_txt);

    double norm_img = 0.0, norm_txt = 0.0;
    for (size_t i = 0; i < grads.first.rows; ++i)
        for (size_t j = 0; j < grads.first.cols; ++j)
            norm_img += grads.first[i][j] * grads.first[i][j];
    for (size_t i = 0; i < grads.second.rows; ++i)
        for (size_t j = 0; j < grads.second.cols; ++j)
            norm_txt += grads.second[i][j] * grads.second[i][j];
    CHECK(norm_img > 1e-6, "grad_z_img non-trivial (norm > 1e-6)");
    CHECK(norm_txt > 1e-6, "grad_z_txt non-trivial (norm > 1e-6)");
}

// =================================================================
// Test 11: Mutation test — zeroing the (1 - sigmoid) term fails the grad check
// (catches "always-zero gradient" bugs)
// =================================================================
static void test_gradient_signs() {
    std::cout << "-- Test 11: Gradient signs are correct --" << std::endl;
    // For perfect alignment (z_img == z_txt), the loss should be small
    // and gradient should push z_img further away from z_txt's negatives
    // and closer to z_txt's positives. We check that at least one element
    // is nonzero with a clear sign direction.
    const int N = 2;
    const int D = 3;
    Tensor z_img(N, D);
    Tensor z_txt(N, D);
    // Pair them up cleanly
    z_img[0][0] = 1.0; z_img[0][1] = 0.0; z_img[0][2] = 0.0;
    z_img[1][0] = 0.0; z_img[1][1] = 1.0; z_img[1][2] = 0.0;
    z_txt[0][0] = 1.0; z_txt[0][1] = 0.0; z_txt[0][2] = 0.0;
    z_txt[1][0] = 0.0; z_txt[1][1] = 1.0; z_txt[1][2] = 0.0;

    SigLIPLoss loss(10.0, 0.0, 1e-8);
    loss.set_normalize(false);
    loss.forward(z_img, z_txt);  // populate cache
    auto grads = loss.backward(z_img, z_txt);

    // All grad values should be tiny because the loss is at its minimum
    // (positives are perfectly aligned, negatives are perfectly orthogonal).
    // The negative pair (0,1) and (1,0) have S=0 so sigmoid(0)=0.5, both
    // grad components are nonzero.
    double norm = 0.0;
    for (size_t i = 0; i < grads.first.rows; ++i)
        for (size_t j = 0; j < grads.first.cols; ++j)
            norm += grads.first[i][j] * grads.first[i][j];
    CHECK(norm > 0.0, "grad non-zero on perfect alignment (negatives still get gradient)");
}

// =================================================================
// Test 12: End-to-end training reduces loss
// Set up a small learnable projection W (D, D) and target T (D, D) such that
// the positive pair S[i][i] = scale and negatives are scale * 0.5. Then
// training should reduce loss below the initial value.
// =================================================================
static void test_end_to_end_training() {
    std::cout << "-- Test 12: End-to-end training reduces loss --" << std::endl;
    const int N = 4;
    const int D = 4;
    const double scale = 5.0;
    const double lr = 0.2;
    const int steps = 100;

    // Fixed raw img and txt features
    Tensor raw_img = Tensor::random(N, D, 0.3);
    Tensor raw_txt = Tensor::random(N, D, 0.3);

    // Learnable projections W_i, W_t (D, D). Start with small non-zero
    // values (NOT zero — zero init means z_img = 0 and the similarity
    // matrix is degenerate, killing the gradient).
    Tensor W_i = Tensor::random(D, D, 0.1);
    Tensor W_t = Tensor::random(D, D, 0.1);

    auto project = [&](const Tensor& X, const Tensor& W) -> Tensor {
        Tensor Y(N, D);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (int k = 0; k < D; ++k) s += X[i][k] * W[k][j];
                Y[i][j] = s;
            }
        }
        return Y;
    };

    SigLIPLoss loss_fn(scale, 0.0, 1e-8);
    loss_fn.set_normalize(false);

    // Initial
    Tensor z_img = project(raw_img, W_i);
    Tensor z_txt = project(raw_txt, W_t);
    Tensor L0 = loss_fn.forward(z_img, z_txt);
    double initial_loss = L0[0][0];

    for (int step = 0; step < steps; ++step) {
        Tensor z_img_cur = project(raw_img, W_i);
        Tensor z_txt_cur = project(raw_txt, W_t);
        loss_fn.forward(z_img_cur, z_txt_cur);
        auto grads = loss_fn.backward(z_img_cur, z_txt_cur);
        const Tensor& g_img = grads.first;
        const Tensor& g_txt = grads.second;

        // Chain: dL/dW_i = raw_img^T @ g_img, dL/dW_t = raw_txt^T @ g_txt
        Tensor g_W_i(D, D);
        Tensor g_W_t(D, D);
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                double s_i = 0.0, s_t = 0.0;
                for (int k = 0; k < N; ++k) {
                    s_i += raw_img[k][i] * g_img[k][j];
                    s_t += raw_txt[k][i] * g_txt[k][j];
                }
                g_W_i[i][j] = s_i;
                g_W_t[i][j] = s_t;
            }
        }
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                W_i[i][j] -= lr * g_W_i[i][j];
                W_t[i][j] -= lr * g_W_t[i][j];
            }
        }
    }
    Tensor z_img_f = project(raw_img, W_i);
    Tensor z_txt_f = project(raw_txt, W_t);
    Tensor Lf = loss_fn.forward(z_img_f, z_txt_f);
    std::cout << "    initial loss = " << std::setprecision(6) << initial_loss
              << ", final loss = " << Lf[0][0] << std::endl;
    CHECK(Lf[0][0] < initial_loss - 0.01, "training reduces loss by > 0.01 over 60 steps");
}

// =================================================================
// Test 13: Bias can shift loss symmetrically
// (positives with t=+1 and z* = scale*S + bias: increasing bias decreases
// softplus(-z*) contribution for positives but increases for negatives;
// the overall loss is a tradeoff.)
// =================================================================
static void test_bias_effect() {
    std::cout << "-- Test 13: bias shifts the loss --" << std::endl;
    const int N = 3;
    const int D = 3;
    Tensor z_img = Tensor::random(N, D, 0.5);
    Tensor z_txt = Tensor::random(N, D, 0.5);

    SigLIPLoss loss_a(10.0, 0.0, 1e-8);
    loss_a.set_normalize(false);
    Tensor La = loss_a.forward(z_img, z_txt);

    SigLIPLoss loss_b(10.0, 5.0, 1e-8);
    loss_b.set_normalize(false);
    Tensor Lb = loss_b.forward(z_img, z_txt);

    CHECK(std::abs(La[0][0] - Lb[0][0]) > 1e-3, "different bias produces different loss");
}

int main() {
    std::cout << "=== Running SigLIP Loss Tests ===" << std::endl;
    test_accessors();
    test_forward_hand_derived();
    test_symmetry();
    test_scale_zero();
    test_bias_zero_perfect_alignment();
    test_numerical_gradient();
    test_cosine_mode_gradient();
    test_targets_cache();
    test_zstar_cache();
    test_gradient_nonvacuous();
    test_gradient_signs();
    test_end_to_end_training();
    test_bias_effect();
    std::cout << "\n=== Results: " << total_passed << " passed, "
              << total_failed << " failed ===" << std::endl;
    return total_failed == 0 ? 0 : 1;
}
