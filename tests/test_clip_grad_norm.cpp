// ============================================================================
// clip_grad_norm_ test suite
// For include/nn/utils/clip_grad_norm.{h,cpp}
//
// Reference math (verified against source):
//   total_norm = sqrt(Σ_p Σ_ij g_p[i][j]²)
//   If total_norm > max_norm: every gradient scaled by max_norm/total_norm (in-place)
//   Otherwise: gradients unchanged
//   Returns total_norm (the original norm); if clipping happened, returns max_norm
//   (because post-clip norm == max_norm by construction)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/utils/clip_grad_norm.h"
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
// Test 1: Single small tensor; norm below max → returns norm, tensor unchanged
//   grad = [3, 4], norm = 5, max = 10 → no clip
// =================================================================
static void test_single_tensor_below_max() {
    run("(1) single tensor norm below max: tensor unchanged, returns norm", []{
        Tensor g(1, 2);
        g[0][0] = 3.0;
        g[0][1] = 4.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 10.0);
        ASSERT_NEAR(n, 5.0, 1e-12);
        ASSERT_NEAR(g[0][0], 3.0, 1e-12);
        ASSERT_NEAR(g[0][1], 4.0, 1e-12);
    });
}

// =================================================================
// Test 2: Single large tensor; norm exceeds max → clipped
//   grad = [6, 8], norm = 10, max = 5 → scale = 0.5
// =================================================================
static void test_single_tensor_clipped() {
    run("(2) single tensor norm > max: tensor scaled, returns max_norm", []{
        Tensor g(1, 2);
        g[0][0] = 6.0;
        g[0][1] = 8.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 5.0);
        ASSERT_NEAR(n, 5.0, 1e-12);
        ASSERT_NEAR(g[0][0], 3.0, 1e-12);
        ASSERT_NEAR(g[0][1], 4.0, 1e-12);
    });
}

// =================================================================
// Test 3: Multiple tensors; total norm accumulated
//   grad1 = [3, 4] (norm 5), grad2 = [0, 12] (norm 12) → total norm = 13
//   max = 6.5 → scale = 0.5
// =================================================================
static void test_multi_tensor_clip() {
    run("(3) multi-tensor: total norm accumulated, all scaled by same factor", []{
        Tensor g1(1, 2); g1[0][0] = 3.0; g1[0][1] = 4.0;
        Tensor g2(1, 2); g2[0][0] = 0.0; g2[0][1] = 12.0;
        std::vector<Tensor*> ps = {&g1, &g2};
        double n = clip_grad_norm_(ps, 6.5);
        ASSERT_NEAR(n, 6.5, 1e-9);
        ASSERT_NEAR(g1[0][0], 1.5, 1e-12);
        ASSERT_NEAR(g1[0][1], 2.0, 1e-12);
        ASSERT_NEAR(g2[0][0], 0.0, 1e-12);
        ASSERT_NEAR(g2[0][1], 6.0, 1e-12);
    });
}

// =================================================================
// Test 4: Zero gradients: norm=0, no scaling, return 0
// =================================================================
static void test_zero_gradients() {
    run("(4) zero gradients: norm=0, no scaling", []{
        Tensor g(1, 3);
        g[0][0] = 0.0; g[0][1] = 0.0; g[0][2] = 0.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 1.0);
        ASSERT_NEAR(n, 0.0, 1e-12);
        ASSERT_NEAR(g[0][0], 0.0, 1e-12);
        ASSERT_NEAR(g[0][1], 0.0, 1e-12);
        ASSERT_NEAR(g[0][2], 0.0, 1e-12);
    });
}

// =================================================================
// Test 5: Mixed signs — L2 norm squares them
//   grad = [-3, 4], norm = 5, max = 100 → no clip
// =================================================================
static void test_mixed_signs() {
    run("(5) mixed signs: L2 norm squares them properly", []{
        Tensor g(1, 2);
        g[0][0] = -3.0;
        g[0][1] = 4.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 100.0);
        ASSERT_NEAR(n, 5.0, 1e-12);
    });
}

// =================================================================
// Test 6: Null pointer in list is skipped (no crash)
// =================================================================
static void test_null_pointer_skipped() {
    run("(6) null pointer in list skipped, no crash", []{
        Tensor g(1, 2);
        g[0][0] = 3.0;
        g[0][1] = 4.0;
        std::vector<Tensor*> ps = {&g, nullptr, &g};
        double n = clip_grad_norm_(ps, 100.0);
        // norm = sqrt(9+16+9+16) = sqrt(50) ≈ 7.07
        ASSERT_NEAR(n, std::sqrt(50.0), 1e-9);
    });
}

// =================================================================
// Test 7: max_norm = 0, single tensor with non-zero values → scale = 0 → zeros out
//   grad = [3, 4], norm = 5, max = 0 → scale = 0
// =================================================================
static void test_max_norm_zero() {
    run("(7) max_norm=0: non-zero grads scale to zero (no div-by-zero crash)", []{
        Tensor g(1, 2);
        g[0][0] = 3.0;
        g[0][1] = 4.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 0.0);
        // norm = 5 > 0, scale = 0/5 = 0
        ASSERT_NEAR(n, 0.0, 1e-12);
        ASSERT_NEAR(g[0][0], 0.0, 1e-12);
        ASSERT_NEAR(g[0][1], 0.0, 1e-12);
    });
}

// =================================================================
// Test 8: max_norm = 0, all-zero grad → norm = 0 → no scaling (since 0 <= 0)
// =================================================================
static void test_max_norm_zero_zero_grad() {
    run("(8) max_norm=0, zero grad: norm=0, no scaling", []{
        Tensor g(1, 2);
        g[0][0] = 0.0;
        g[0][1] = 0.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 0.0);
        ASSERT_NEAR(n, 0.0, 1e-12);
        ASSERT_NEAR(g[0][0], 0.0, 1e-12);
        ASSERT_NEAR(g[0][1], 0.0, 1e-12);
    });
}

// =================================================================
// Test 9: Returns pre-clip norm when no clipping needed
// =================================================================
static void test_returns_pre_clip_norm() {
    run("(9) returns pre-clip norm when no clipping", []{
        Tensor g(1, 3);
        g[0][0] = 1.0; g[0][1] = 2.0; g[0][2] = 2.0;
        std::vector<Tensor*> ps = {&g};
        // norm = sqrt(1+4+4) = 3
        double n = clip_grad_norm_(ps, 5.0);
        ASSERT_NEAR(n, 3.0, 1e-12);
    });
}

// =================================================================
// Test 10: Returns max_norm when clipping happens (post-clip norm)
// =================================================================
static void test_returns_max_norm_when_clipping() {
    run("(10) returns max_norm when clipping happened", []{
        Tensor g(1, 2);
        g[0][0] = 30.0;
        g[0][1] = 40.0;
        std::vector<Tensor*> ps = {&g};
        // norm = 50, max = 7
        double n = clip_grad_norm_(ps, 7.0);
        ASSERT_NEAR(n, 7.0, 1e-12);
    });
}

// =================================================================
// Test 11: (1,1) tensor with single value
// =================================================================
static void test_single_element_tensor() {
    run("(11) (1,1) tensor: single element works", []{
        Tensor g(1, 1);
        g[0][0] = 5.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 2.5);
        ASSERT_NEAR(n, 2.5, 1e-12);
        ASSERT_NEAR(g[0][0], 2.5, 1e-12);
    });
}

// =================================================================
// Test 12: (4,4) tensor with shape normalization
//   All 16 entries = 1 → norm = sqrt(16) = 4, max = 2 → scale = 0.5
// =================================================================
static void test_4x4_tensor_clip() {
    run("(12) (4,4) tensor: all 1s, norm=4, clip to 2", []{
        Tensor g(4, 4);
        for (size_t i = 0; i < g.rows; ++i)
            for (size_t j = 0; j < g.cols; ++j)
                g[i][j] = 1.0;
        std::vector<Tensor*> ps = {&g};
        double n = clip_grad_norm_(ps, 2.0);
        ASSERT_NEAR(n, 2.0, 1e-9);
        for (size_t i = 0; i < g.rows; ++i)
            for (size_t j = 0; j < g.cols; ++j)
                ASSERT_NEAR(g[i][j], 0.5, 1e-12);
    });
}

// =================================================================
// Test 13: Cross-tensor scaling consistency: all tensors scaled by SAME factor
// =================================================================
static void test_cross_tensor_consistent_scale() {
    run("(13) cross-tensor scaling: all tensors scaled by same factor", []{
        Tensor g1(1, 2); g1[0][0] = 3.0; g1[0][1] = 4.0;  // norm 5
        Tensor g2(1, 2); g2[0][0] = 6.0; g2[0][1] = 8.0;  // norm 10
        std::vector<Tensor*> ps = {&g1, &g2};
        // total norm = sqrt(25+100) = sqrt(125) ≈ 11.18
        // max = 5.58, scale = 0.5
        double total = std::sqrt(125.0);
        double max = total * 0.5;
        double n = clip_grad_norm_(ps, max);
        ASSERT_NEAR(n, max, 1e-9);
        ASSERT_NEAR(g1[0][0], 1.5, 1e-12);
        ASSERT_NEAR(g1[0][1], 2.0, 1e-12);
        ASSERT_NEAR(g2[0][0], 3.0, 1e-12);
        ASSERT_NEAR(g2[0][1], 4.0, 1e-12);
    });
}

// =================================================================
// Test 14: Empty list → total_norm=0, no crash
// =================================================================
static void test_empty_list() {
    run("(14) empty list: norm=0, no crash", []{
        std::vector<Tensor*> ps;
        double n = clip_grad_norm_(ps, 1.0);
        ASSERT_NEAR(n, 0.0, 1e-12);
    });
}

// =================================================================
// Test 15: One tensor below max, one above (mixed situation)
//   g1 norm 1, g2 norm 100 → total = sqrt(10001) ≈ 100.005, max = 50 → scale ≈ 0.4999
// =================================================================
static void test_mixed_below_above() {
    run("(15) one below + one above max: all scaled by same factor", []{
        Tensor g1(1, 1); g1[0][0] = 1.0;
        Tensor g2(1, 1); g2[0][0] = 100.0;
        std::vector<Tensor*> ps = {&g1, &g2};
        double total = std::sqrt(1.0 + 10000.0);
        double max = total * 0.5;
        double scale = max / total;
        double n = clip_grad_norm_(ps, max);
        ASSERT_NEAR(n, max, 1e-9);
        ASSERT_NEAR(g1[0][0], 1.0 * scale, 1e-12);
        ASSERT_NEAR(g2[0][0], 100.0 * scale, 1e-12);
    });
}

int main() {
    test_single_tensor_below_max();
    test_single_tensor_clipped();
    test_multi_tensor_clip();
    test_zero_gradients();
    test_mixed_signs();
    test_null_pointer_skipped();
    test_max_norm_zero();
    test_max_norm_zero_zero_grad();
    test_returns_pre_clip_norm();
    test_returns_max_norm_when_clipping();
    test_single_element_tensor();
    test_4x4_tensor_clip();
    test_cross_tensor_consistent_scale();
    test_empty_list();
    test_mixed_below_above();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}