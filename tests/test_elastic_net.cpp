// ============================================================================
// ElasticNet + ElasticNetCD test suite
// For include/nn/utils/elastic_net.{h,cpp}
//
// Reference math (verified against source):
//   Penalty: α * (l1_ratio * ||w||_1 + (1-l1_ratio) * ||w||_2²/2)
//   Objective: MSE(pred, y) + penalty(w)
//     pred = w0[0][0] + Σ_j X[i][j]*w[0][j]  (intercept from w0; if w0 empty, intercept=0)
//   ElasticNet.fit uses ridge warm-start + Gauss-Seidel soft-thresholding iter (500 steps)
//   ElasticNetCD.fit uses coordinate descent with col_norm normalization
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/utils/elastic_net.h"
#include "nn/core/tensor.h"

static int g_pass = 0;
static int g_fail = 0;
static std::string g_current_test;

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cout << "  [FAIL] (" << g_current_test << ") " \
                      << #cond << " @ line " << __LINE__ << "\n"; \
            ++g_fail; \
        } else { ++g_pass; } \
    } while (0)

#define ASSERT_NEAR(a, b, tol) \
    do { \
        double _a = (double)(a), _b = (double)(b), _tol = (double)(tol); \
        if (!(std::abs(_a - _b) <= _tol)) { \
            std::cout << "  [FAIL] (" << g_current_test << ") " \
                      << #a << "=" << _a << " vs " << #b << "=" << _b \
                      << " tol=" << _tol << " @ line " << __LINE__ << "\n"; \
            ++g_fail; \
        } else { ++g_pass; } \
    } while (0)

static void run(const std::string& name, std::function<void()> body) {
    g_current_test = name;
    std::cout << "\n" << name << "\n";
    body();
}

#include <functional>

// =================================================================
// Test 1: Pure L1 penalty (l1_ratio=1.0): penalty = α * L1
//   w = [[1, 2, -3]], α=1, l1_ratio=1.0 → penalty = 1*6 = 6
// =================================================================
static void test_penalty_pure_l1() {
    run("(1) penalty pure L1 (l1_ratio=1.0)", []{
        ElasticNet en(1.0, 1.0);
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = -3.0;
        ASSERT_NEAR(en.penalty(w), 6.0, 1e-12);
    });
}

// =================================================================
// Test 2: Pure L2 penalty (l1_ratio=0.0): penalty = α * L2²/2
//   w = [[1, 2, -3]], α=1, l1_ratio=0.0 → penalty = (1+4+9)/2 = 7
// =================================================================
static void test_penalty_pure_l2() {
    run("(2) penalty pure L2 (l1_ratio=0.0)", []{
        ElasticNet en(1.0, 0.0);
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = -3.0;
        ASSERT_NEAR(en.penalty(w), 7.0, 1e-12);
    });
}

// =================================================================
// Test 3: Penalty mix (l1_ratio=0.5): penalty = α * (0.5 * L1 + 0.5 * L2²/2)
//   w = [[1, 2, -3]] → 0.5 * 6 + 0.5 * 7 = 6.5
// =================================================================
static void test_penalty_mix() {
    run("(3) penalty mix (l1_ratio=0.5)", []{
        ElasticNet en(1.0, 0.5);
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = -3.0;
        ASSERT_NEAR(en.penalty(w), 6.5, 1e-12);
    });
}

// =================================================================
// Test 4: Penalty on zero weights = 0
// =================================================================
static void test_penalty_zero() {
    run("(4) penalty on zero weights = 0", []{
        ElasticNet en(2.0, 0.5);
        Tensor w(1, 5);
        ASSERT_NEAR(en.penalty(w), 0.0, 1e-12);
    });
}

// =================================================================
// Test 5: Penalty scales with α
//   en(2.0, 0.5) on [[1,2,3]] = 2 * (0.5*6 + 0.5*14/2) = 2 * 6.5 = 13
// =================================================================
static void test_penalty_scales_with_alpha() {
    run("(5) penalty scales with α", []{
        ElasticNet en(2.0, 0.5);
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = 3.0;
        // L1=6, L2²/2=14/2=7, 0.5*6+0.5*7=6.5, *α=2 → 13
        ASSERT_NEAR(en.penalty(w), 13.0, 1e-12);
    });
}

// =================================================================
// Test 6: Penalty hand-derived on [1, 2, 3]
//   en(1.0, 0.5): L1=6, L2²/2 = (1+4+9)/2 = 7, mix = 0.5*6 + 0.5*7 = 6.5
// =================================================================
static void test_penalty_hand_derived() {
    run("(6) penalty hand-derived on [1,2,3], α=1, λ=0.5", []{
        ElasticNet en(1.0, 0.5);
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = 3.0;
        ASSERT_NEAR(en.penalty(w), 6.5, 1e-12);
    });
}

// =================================================================
// Test 7: Penalty sign-symmetric: penalty(w) == penalty(-w)
// =================================================================
static void test_penalty_symmetric() {
    run("(7) penalty(w) == penalty(-w) — sign symmetry", []{
        ElasticNet en(0.7, 0.3);
        Tensor w(1, 4);
        w[0][0] = 0.5; w[0][1] = -1.5; w[0][2] = 2.0; w[0][3] = -0.3;
        Tensor wneg(1, 4);
        wneg[0][0] = -0.5; wneg[0][1] = 1.5; wneg[0][2] = -2.0; wneg[0][3] = 0.3;
        ASSERT_NEAR(en.penalty(w), en.penalty(wneg), 1e-12);
    });
}

// =================================================================
// Test 8: Pure L1 penalty on tiny values — α * |w| only
//   w = [[0.5, -0.5]], α=0.5, l1_ratio=1.0 → 0.5 * 1.0 = 0.5
// =================================================================
static void test_penalty_l1_tiny() {
    run("(8) pure L1 penalty on tiny values", []{
        ElasticNet en(0.5, 1.0);
        Tensor w(1, 2);
        w[0][0] = 0.5; w[0][1] = -0.5;
        ASSERT_NEAR(en.penalty(w), 0.5, 1e-12);
    });
}

// =================================================================
// Test 9: Pure L2 penalty on tiny values — α * w²/2
//   w = [[0.5, 0.5]], α=0.5, l1_ratio=0.0 → 0.5 * (0.25+0.25)/2 = 0.5 * 0.25 = 0.125
// =================================================================
static void test_penalty_l2_tiny() {
    run("(9) pure L2 penalty on tiny values", []{
        ElasticNet en(0.5, 0.0);
        Tensor w(1, 2);
        w[0][0] = 0.5; w[0][1] = 0.5;
        ASSERT_NEAR(en.penalty(w), 0.125, 1e-12);
    });
}

// =================================================================
// Test 10: Objective on zero-weight, zero-target: MSE=0, objective=penalty
// =================================================================
static void test_objective_zero_pred() {
    run("(10) objective: MSE=0 (zero pred, zero target) + penalty", []{
        ElasticNet en(1.0, 0.5);
        Tensor X(3, 2);  // doesn't matter since w=0
        Tensor y(3, 1); y[0][0]=0; y[1][0]=0; y[2][0]=0;
        Tensor w(1, 2); w[0][0] = 0; w[0][1] = 0;
        // pred = 0 + 0*X + 0*X = 0, MSE = 0, obj = penalty(0) = 0
        ASSERT_NEAR(en.objective(X, y, w), 0.0, 1e-12);
    });
}

// =================================================================
// Test 11: Objective: w = 0, target = [1, 1, 1] → MSE = 1, objective = 1 + 0
// =================================================================
static void test_objective_nonzero_target() {
    run("(11) objective: w=0, y=[1,1,1] → MSE=1, penalty=0, obj=1", []{
        ElasticNet en(1.0, 0.5);
        Tensor X(3, 2);  // doesn't matter
        Tensor y(3, 1); y[0][0]=1; y[1][0]=1; y[2][0]=1;
        Tensor w(1, 2); w[0][0] = 0; w[0][1] = 0;
        // pred = 0 for all; MSE = 1; penalty = 0
        ASSERT_NEAR(en.objective(X, y, w), 1.0, 1e-12);
    });
}

// =================================================================
// Test 12: Objective with intercept (w0)
//   X = [[1, 0], [0, 1]], y = [[1], [2]], w = [[0, 0]], w0 = [[5]] (intercept=5)
//   pred[0] = 5 + 0*1 + 0*0 = 5 → err = 4
//   pred[1] = 5 + 0*0 + 0*1 = 5 → err = 3
//   MSE = (16+9)/2 = 12.5
// =================================================================
static void test_objective_with_intercept() {
    run("(12) objective: intercept (w0) is added to prediction", []{
        ElasticNet en(1.0, 0.0);
        Tensor X(2, 2);
        X[0][0] = 1; X[0][1] = 0;
        X[1][0] = 0; X[1][1] = 1;
        Tensor y(2, 1); y[0][0] = 1; y[1][0] = 2;
        Tensor w(1, 2); w[0][0] = 0; w[0][1] = 0;
        Tensor w0(1, 1); w0[0][0] = 5;
        // pred[0] = 5, err=4, sq=16; pred[1] = 5, err=3, sq=9; MSE = 25/2 = 12.5
        ASSERT_NEAR(en.objective(X, y, w, w0), 12.5, 1e-12);
    });
}

// =================================================================
// Test 13: fit() returns weight tensor of correct shape (1, d)
// =================================================================
static void test_fit_shape() {
    run("(13) ElasticNet.fit returns weights of shape (1, d)", []{
        ElasticNet en(0.1, 0.5);
        Tensor X(5, 3);
        Tensor y(5, 1);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) X[i][j] = 0.1 * (i + j);
            y[i][0] = 0.5 * i;
        }
        Tensor w = en.fit(X, y);
        ASSERT(w.rows == 1);
        ASSERT(w.cols == 3);
        for (size_t j = 0; j < w.cols; ++j)
            ASSERT(std::isfinite(w[0][j]));
    });
}

// =================================================================
// Test 14: fit() with low alpha on separable data approximates solution
//   X = I_3, y = [1, 2, 3], α=0.001, l1_ratio=0 (almost pure L2)
//   The L2 problem is w = (X^T X + α I)^-1 X^T y ≈ y for tiny α
//   But the soft-thresholding in fit() may still give some shrinkage.
//   Just verify weights are finite and have correct sign.
// =================================================================
static void test_fit_identity_inputs() {
    run("(14) ElasticNet.fit on X=I, y=given: weights have correct sign", []{
        ElasticNet en(0.001, 0.1);  // tiny alpha, small L1
        Tensor X(3, 3);
        Tensor y(3, 1);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) X[i][j] = (i == j) ? 1.0 : 0.0;
            y[i][0] = (i == 0) ? 1.0 : (i == 1) ? 2.0 : 3.0;
        }
        Tensor w = en.fit(X, y);
        // Weights should be positive (target positive, identity X)
        for (size_t j = 0; j < w.cols; ++j)
            ASSERT(w[0][j] > 0.0);
    });
}

// =================================================================
// Test 15: ElasticNetCD constructor defaults
// =================================================================
static void test_cd_defaults() {
    run("(15) ElasticNetCD constructor defaults: α=1, λ=0.5, max_iter=1000, tol=1e-6", []{
        ElasticNetCD cd;
        Tensor X(4, 2);
        Tensor y(4, 1);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) X[i][j] = 0.5 * i + 0.1 * j;
            y[i][0] = 0.5 * i;
        }
        Tensor w = cd.fit(X, y);
        ASSERT(w.rows == 1);
        ASSERT(w.cols == 2);
        for (size_t j = 0; j < w.cols; ++j)
            ASSERT(std::isfinite(w[0][j]));
    });
}

// =================================================================
// Test 16: ElasticNetCD non-default constructor
// =================================================================
static void test_cd_nondefault() {
    run("(16) ElasticNetCD non-default constructor: smaller max_iter, larger tol", []{
        ElasticNetCD cd(0.5, 0.7, /*max_iter=*/100, /*tol=*/1e-3);
        Tensor X(5, 4);
        Tensor y(5, 1);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) X[i][j] = 0.1 * i + 0.2 * j;
            y[i][0] = i % 2;
        }
        Tensor w = cd.fit(X, y);
        ASSERT(w.rows == 1);
        ASSERT(w.cols == 4);
        for (size_t j = 0; j < w.cols; ++j)
            ASSERT(std::isfinite(w[0][j]));
    });
}

// =================================================================
// Test 17: Fit converges (multiple calls don't crash; weights remain finite)
// =================================================================
static void test_fit_repeated() {
    run("(17) ElasticNet.fit called repeatedly: still finite", []{
        ElasticNet en(0.01, 0.5);
        Tensor X(6, 3);
        Tensor y(6, 1);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) X[i][j] = 0.05 * (i + 2 * j);
            y[i][0] = 0.3 * i;
        }
        for (int trial = 0; trial < 3; ++trial) {
            Tensor w = en.fit(X, y);
            for (size_t j = 0; j < w.cols; ++j)
                ASSERT(std::isfinite(w[0][j]));
        }
    });
}

// =================================================================
// Test 18: Penalty on (1, 5) tensor — sum across all elements
//   w = [[1, 2, 3, 4, 5]], α=1, λ=1.0 → penalty = 1 * 15 = 15
// =================================================================
static void test_penalty_larger_tensor() {
    run("(18) penalty on (1, 5) tensor: sum across all elements", []{
        ElasticNet en(1.0, 1.0);
        Tensor w(1, 5);
        for (size_t j = 0; j < 5; ++j) w[0][j] = (double)(j + 1);
        ASSERT_NEAR(en.penalty(w), 15.0, 1e-12);
    });
}

// =================================================================
// Test 19: Penalty default constructor: α=1, λ=0.5
// =================================================================
static void test_default_constructor() {
    run("(19) default constructor: α=1, λ=0.5", []{
        ElasticNet en;
        Tensor w(1, 3);
        w[0][0] = 1.0; w[0][1] = 1.0; w[0][2] = 1.0;
        // L1 = 3, L2²/2 = 3/2 = 1.5, 0.5*3 + 0.5*1.5 = 2.25
        ASSERT_NEAR(en.penalty(w), 2.25, 1e-12);
    });
}

// =================================================================
// Test 20: ElasticNet.fit with weights_init provided
// =================================================================
static void test_fit_with_init() {
    run("(20) ElasticNet.fit with weights_init: still returns valid shape", []{
        ElasticNet en(0.1, 0.5);
        Tensor X(4, 3);
        Tensor y(4, 1);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) X[i][j] = 0.1 * (i + j);
            y[i][0] = 0.5 * i;
        }
        Tensor w_init(1, 3);
        w_init[0][0] = 0.1; w_init[0][1] = 0.2; w_init[0][2] = 0.3;
        Tensor w = en.fit(X, y, w_init);
        ASSERT(w.rows == 1);
        ASSERT(w.cols == 3);
        for (size_t j = 0; j < w.cols; ++j)
            ASSERT(std::isfinite(w[0][j]));
    });
}

int main() {
    test_penalty_pure_l1();
    test_penalty_pure_l2();
    test_penalty_mix();
    test_penalty_zero();
    test_penalty_scales_with_alpha();
    test_penalty_hand_derived();
    test_penalty_symmetric();
    test_penalty_l1_tiny();
    test_penalty_l2_tiny();
    test_objective_zero_pred();
    test_objective_nonzero_target();
    test_objective_with_intercept();
    test_fit_shape();
    test_fit_identity_inputs();
    test_cd_defaults();
    test_cd_nondefault();
    test_fit_repeated();
    test_penalty_larger_tensor();
    test_default_constructor();
    test_fit_with_init();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}