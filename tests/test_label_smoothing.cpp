// ============================================================================
// Label Smoothing Cross-Entropy test suite
// For include/nn/utils/label_smoothing.{h,cpp}
//
// Reference math (verified against source — PyTorch convention):
//   y_smooth[j] = (j == label) ? (1 - ε) : (ε / K)
//   Note: this does NOT sum to 1.0; sum = 1 - ε + ε = 1 only when label is in [0, K).
//   Actually for K > 1: true class gets (1-ε), and (K-1) off-classes each get ε/K.
//   Sum = (1-ε) + (K-1) * ε/K = 1 - ε + ε - ε/K = 1 - ε/K. Hmm — let me verify:
//     With K=2, ε=0.1: true class = 0.9, off class = 0.05. Sum = 0.95 = 1 - 0.05 = 1 - ε/K.
//     With K=4, ε=0.1: true class = 0.9, off classes = 0.025 each (3 of them). Sum = 0.9 + 0.075 = 0.975 = 1 - 0.025 = 1 - ε/K.
//
//   Forward: L = (1/B) * Σ_b Σ_j y_smooth[b][j] * (-log softmax(logits)[b][j])
//   Backward: ∂L/∂logits[b][j] = (softmax(logits)[b][j] - y_smooth[b][j]) / B
//
// Hard labels (target rows = (B, 1) with integer class indices): converted to "sparse" smoothed targets
// Soft labels (target rows = (B, K)): used as-is (no smoothing)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "nn/utils/label_smoothing.h"
#include "nn/core/tensor.h"

using std::abs;

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
// Test 1: Default constructor stores smoothing=0.1
// =================================================================
static void test_default_smoothing() {
    run("(1) default smoothing=0.1", []{
        LabelSmoothingCrossEntropy loss;
        ASSERT_NEAR(loss.get_smoothing(), 0.1, 0.0);
    });
}

// =================================================================
// Test 2: Non-default constructor
// =================================================================
static void test_custom_smoothing() {
    run("(2) custom smoothing=0.2 round-trip", []{
        LabelSmoothingCrossEntropy loss(0.2);
        ASSERT_NEAR(loss.get_smoothing(), 0.2, 1e-12);
    });
}

// =================================================================
// Test 3: Forward with smoothing=0 reduces to standard CE on a 1x2 case
//   With ε=0: y_smooth[j] = (j==label) ? 1.0 : 0.0 — standard one-hot
//   logits = [1.0, 0.5], target=0
//   softmax = [exp(1)/(exp(1)+exp(0.5)), exp(0.5)/(exp(1)+exp(0.5))]
//           ≈ [0.5987, 0.4013]
//   CE = -log(0.5987) ≈ 0.5133
// =================================================================
static void test_forward_smoothing_zero() {
    run("(3) forward smoothing=0 = standard cross-entropy", []{
        LabelSmoothingCrossEntropy loss(0.0);
        Tensor logits(1, 2);
        logits[0][0] = 1.0;
        logits[0][1] = 0.5;
        Tensor targets(1, 1);
        targets[0][0] = 0;

        Tensor out = loss.forward(logits, targets);
        double expected = -std::log(std::exp(1.0) / (std::exp(1.0) + std::exp(0.5)));
        ASSERT_NEAR(out[0][0], expected, 1e-9);
    });
}

// =================================================================
// Test 4: Forward with smoothing=0.1 hand-derived
//   logits = [1.0, 0.5], target=0, K=2, smoothing=0.1
//   PyTorch convention: smooth_targets = [(1-ε+ε/K), ε/K] = [0.95, 0.05]   (sums to 1)
//   softmax ≈ [0.5987, 0.4013]
//   L = -(0.95 * log(0.5987) + 0.05 * log(0.4013))
// =================================================================
static void test_forward_with_smoothing_hand_derived() {
    run("(4) forward smoothing=0.1 hand-derived (PyTorch convention sums to 1)", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(1, 2);
        logits[0][0] = 1.0;
        logits[0][1] = 0.5;
        Tensor targets(1, 1);
        targets[0][0] = 0;

        Tensor out = loss.forward(logits, targets);
        double s0 = std::exp(1.0), s1 = std::exp(0.5);
        double p0 = s0 / (s0 + s1), p1 = s1 / (s0 + s1);
        // PyTorch convention: true class = 1-ε+ε/K, off class = ε/K (sums to 1)
        double true_val = 1.0 - 0.1 + 0.1/2.0;  // 0.95
        double off_val = 0.1 / 2.0;              // 0.05
        double expected = -(true_val * std::log(p0) + off_val * std::log(p1));
        ASSERT_NEAR(out[0][0], expected, 1e-9);
    });
}

// =================================================================
// Test 5: Smoothed targets: with K=4, target=2, smoothing=0.1
//   PyTorch convention: smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K]
//                       = [0.025, 0.025, 0.925, 0.025]
//   Sum = 0.025+0.025+0.925+0.025 = 1.0
// =================================================================
static void test_smoothed_target_distribution() {
    run("(5) smoothed target distribution K=4 (PyTorch convention sums to 1)", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(1, 4);
        logits[0][0] = 0.5;
        logits[0][1] = -0.2;
        logits[0][2] = 1.0;
        logits[0][3] = 0.3;
        Tensor targets(1, 1);
        targets[0][0] = 2;

        Tensor out = loss.forward(logits, targets);
        ASSERT(out[0][0] > 0.0);
        ASSERT(std::isfinite(out[0][0]));
        // Hand-compute: softmax via stable form (subtract max=1.0)
        double mx = 1.0;
        double e0 = std::exp(0.5 - mx), e1 = std::exp(-0.2 - mx);
        double e2 = std::exp(1.0 - mx),  e3 = std::exp(0.3 - mx);
        double sum = e0 + e1 + e2 + e3;
        double p0 = e0 / sum, p1 = e1 / sum, p2 = e2 / sum, p3 = e3 / sum;
        // PyTorch convention: smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K]
        double expected = -(0.025 * std::log(p0) + 0.025 * std::log(p1)
                          + 0.925 * std::log(p2) + 0.025 * std::log(p3));
        ASSERT_NEAR(out[0][0], expected, 1e-9);
    });
}

// =================================================================
// Test 6: Soft-label input — (B, K) targets used as-is
// =================================================================
static void test_soft_labels_passed_through() {
    run("(6) soft labels (B,K) used as-is", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(1, 3);
        logits[0][0] = 1.0;
        logits[0][1] = 0.0;
        logits[0][2] = -1.0;

        // Soft labels (not one-hot; doesn't sum to 1)
        Tensor targets(1, 3);
        targets[0][0] = 0.4;
        targets[0][1] = 0.5;
        targets[0][2] = 0.1;

        Tensor out = loss.forward(logits, targets);
        // Hand-compute
        double mx = 1.0;
        double e0 = std::exp(1.0 - mx), e1 = std::exp(0.0 - mx), e2 = std::exp(-1.0 - mx);
        double sum = e0 + e1 + e2;
        double p0 = e0 / sum, p1 = e1 / sum, p2 = e2 / sum;
        double expected = -(0.4 * std::log(p0) + 0.5 * std::log(p1) + 0.1 * std::log(p2));
        ASSERT_NEAR(out[0][0], expected, 1e-9);
    });
}

// =================================================================
// Test 7: Backward gradient is (softmax - y_smooth) / B
//   PyTorch convention: true class = 1-ε+ε/K, off classes = ε/K
// =================================================================
static void test_backward_formula() {
    run("(7) backward = (softmax - smooth_targets) / B (PyTorch convention)", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(2, 3);
        logits[0][0] = 1.0;  logits[0][1] = 0.5;  logits[0][2] = -0.3;
        logits[1][0] = -0.5; logits[1][1] = 1.5;  logits[1][2] = 0.2;

        Tensor targets(2, 1);
        targets[0][0] = 0;
        targets[1][0] = 1;

        loss.forward(logits, targets);
        Tensor grad = loss.backward(logits, targets);

        // Recompute softmax row 0
        auto softmax_row = [&](int r) {
            double mx = logits[r][0];
            for (size_t j = 1; j < logits.cols; ++j)
                if (logits[r][j] > mx) mx = logits[r][j];
            std::vector<double> p(logits.cols);
            double sum = 0.0;
            for (size_t j = 0; j < logits.cols; ++j) {
                p[j] = std::exp(logits[r][j] - mx);
                sum += p[j];
            }
            for (size_t j = 0; j < logits.cols; ++j) p[j] /= sum;
            return p;
        };
        auto p0 = softmax_row(0), p1 = softmax_row(1);
        // PyTorch convention: smooth[row 0, target=0, ε=0.1, K=3] = [1-ε+ε/K, ε/K, ε/K] = [0.9333, 0.0333, 0.0333]
        // smooth[row 1, target=1, ε=0.1, K=3] = [ε/K, 1-ε+ε/K, ε/K] = [0.0333, 0.9333, 0.0333]
        double inv_B = 1.0 / 2.0;
        double true_val = 1.0 - 0.1 + 0.1/3.0;  // 0.9333
        double off_val = 0.1 / 3.0;              // 0.0333
        ASSERT_NEAR(grad[0][0], (p0[0] - true_val) * inv_B, 1e-12);
        ASSERT_NEAR(grad[0][1], (p0[1] - off_val)  * inv_B, 1e-12);
        ASSERT_NEAR(grad[0][2], (p0[2] - off_val)  * inv_B, 1e-12);
        ASSERT_NEAR(grad[1][0], (p1[0] - off_val)  * inv_B, 1e-12);
        ASSERT_NEAR(grad[1][1], (p1[1] - true_val) * inv_B, 1e-12);
        ASSERT_NEAR(grad[1][2], (p1[2] - off_val)  * inv_B, 1e-12);
    });
}

// =================================================================
// Test 8: Gradient row sum = 0 (PyTorch convention sums to 1, so Σ(p_j - y_smooth_j) = 0)
// =================================================================
static void test_gradient_row_sum_zero() {
    run("(8) gradient row sums to zero (PyTorch convention)", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(3, 4);
        for (size_t i = 0; i < logits.rows; ++i)
            for (size_t j = 0; j < logits.cols; ++j)
                logits[i][j] = 0.1 * i - 0.2 * j;

        Tensor targets(3, 1);
        targets[0][0] = 1;
        targets[1][0] = 2;
        targets[2][0] = 3;

        loss.forward(logits, targets);
        Tensor grad = loss.backward(logits, targets);
        for (size_t i = 0; i < grad.rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < grad.cols; ++j) sum += grad[i][j];
            ASSERT_NEAR(sum, 0.0, 1e-12);
        }
    });
}

// =================================================================
// Test 9: Numerical gradient check vs analytical backward (per-feature)
// =================================================================
static void test_numerical_gradient_single_feature() {
    run("(9) numerical gradient matches analytical per-feature", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(1, 3);
        logits[0][0] = 1.0;
        logits[0][1] = 0.5;
        logits[0][2] = -0.3;
        Tensor targets(1, 1);
        targets[0][0] = 0;

        Tensor grad_ana = loss.backward(logits, targets);

        const double eps = 1e-5;
        for (size_t j = 0; j < logits.cols; ++j) {
            double orig = logits[0][j];
            logits[0][j] = orig + eps;
            double Lp = loss.forward(logits, targets)[0][0];
            logits[0][j] = orig - eps;
            double Lm = loss.forward(logits, targets)[0][0];
            logits[0][j] = orig;

            double grad_num = (Lp - Lm) / (2.0 * eps);
            double ana = grad_ana[0][j];
            double scale = std::max(1e-12, std::max(std::abs(grad_num), std::abs(ana)));
            ASSERT(std::abs(grad_num - ana) / scale < 1e-5);
        }
    });
}

// =================================================================
// Test 10: Multi-batch gradient numerical check (B=3, K=3)
// =================================================================
static void test_numerical_gradient_multibatch() {
    run("(10) numerical gradient matches on multi-batch (B=3, K=3)", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(3, 3);
        logits[0][0] = 1.0; logits[0][1] = 0.5; logits[0][2] = -0.3;
        logits[1][0] = -0.5; logits[1][1] = 1.5; logits[1][2] = 0.2;
        logits[2][0] = 0.7; logits[2][1] = -0.4; logits[2][2] = 0.9;
        Tensor targets(3, 1);
        targets[0][0] = 0;
        targets[1][0] = 2;
        targets[2][0] = 1;

        Tensor grad_ana = loss.backward(logits, targets);

        const double eps = 1e-5;
        for (size_t i = 0; i < logits.rows; ++i) {
            for (size_t j = 0; j < logits.cols; ++j) {
                double orig = logits[i][j];
                logits[i][j] = orig + eps;
                double Lp = loss.forward(logits, targets)[0][0];
                logits[i][j] = orig - eps;
                double Lm = loss.forward(logits, targets)[0][0];
                logits[i][j] = orig;

                double grad_num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_ana[i][j];
                double scale = std::max(1e-12, std::max(std::abs(grad_num), std::abs(ana)));
                ASSERT(std::abs(grad_num - ana) / scale < 1e-5);
            }
        }
    });
}

// =================================================================
// Test 11: Larger smoothing (0.5) — forward still finite
// =================================================================
static void test_high_smoothing() {
    run("(11) smoothing=0.5: forward finite, gradient sums to zero", []{
        LabelSmoothingCrossEntropy loss(0.5);
        Tensor logits(2, 4);
        for (size_t i = 0; i < logits.rows; ++i)
            for (size_t j = 0; j < logits.cols; ++j)
                logits[i][j] = 0.3 * i + 0.1 * j - 0.2;
        Tensor targets(2, 1);
        targets[0][0] = 0;
        targets[1][0] = 3;

        Tensor fwd = loss.forward(logits, targets);
        ASSERT(std::isfinite(fwd[0][0]));
        ASSERT(fwd[0][0] > 0.0);

        Tensor grad = loss.backward(logits, targets);
        for (size_t i = 0; i < grad.rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < grad.cols; ++j) sum += grad[i][j];
            ASSERT_NEAR(sum, 0.0, 1e-12);
        }
    });
}

// =================================================================
// Test 12: smoothing=1.0 — uniform distribution: smooth = [1/K, 1/K, ..., 1/K]
//   grad = (softmax - 1/K) / B
// =================================================================
static void test_uniform_target_smoothing_one() {
    run("(12) smoothing=1.0: smooth = uniform [1/K, ..., 1/K], gradient = (softmax - 1/K)/B", []{
        LabelSmoothingCrossEntropy loss(1.0);
        Tensor logits(1, 3);
        logits[0][0] = 0.5;
        logits[0][1] = 0.0;
        logits[0][2] = -0.5;
        Tensor targets(1, 1);
        targets[0][0] = 0;

        Tensor grad = loss.backward(logits, targets);
        // Recompute softmax
        double mx = 0.5;
        double e0 = std::exp(0.5 - mx), e1 = std::exp(0.0 - mx), e2 = std::exp(-0.5 - mx);
        double sum = e0 + e1 + e2;
        double inv_B = 1.0; // B=1
        double inv_K = 1.0 / 3.0;
        // Smooth: with ε=1.0, true = 1-1+1/K = 1/K, off = 1/K → uniform
        ASSERT_NEAR(grad[0][0], (e0 / sum - inv_K) * inv_B, 1e-12);
        ASSERT_NEAR(grad[0][1], (e1 / sum - inv_K) * inv_B, 1e-12);
        ASSERT_NEAR(grad[0][2], (e2 / sum - inv_K) * inv_B, 1e-12);
    });
}

// =================================================================
// Test 13: Forward is non-negative
// =================================================================
static void test_forward_nonneg() {
    run("(13) forward is non-negative", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(2, 3);
        logits[0][0] = 0.1; logits[0][1] = -0.2; logits[0][2] = 0.3;
        logits[1][0] = -0.5; logits[1][1] = 0.4; logits[1][2] = -0.1;
        Tensor targets(2, 1);
        targets[0][0] = 2;
        targets[1][0] = 0;

        Tensor out = loss.forward(logits, targets);
        ASSERT(out[0][0] >= 0.0);
        ASSERT(std::isfinite(out[0][0]));
    });
}

// =================================================================
// Test 14: Backward gradient is finite and shape-matched
// =================================================================
static void test_backward_shape_and_finite() {
    run("(14) backward shape and finiteness", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(2, 4);
        for (size_t i = 0; i < logits.rows; ++i)
            for (size_t j = 0; j < logits.cols; ++j)
                logits[i][j] = 0.05 * (i * logits.cols + j);
        Tensor targets(2, 1);
        targets[0][0] = 0;
        targets[1][0] = 3;

        Tensor grad = loss.backward(logits, targets);
        ASSERT(grad.rows == 2);
        ASSERT(grad.cols == 4);
        for (size_t i = 0; i < grad.rows; ++i)
            for (size_t j = 0; j < grad.cols; ++j)
                ASSERT(std::isfinite(grad[i][j]));
    });
}

// =================================================================
// Test 15: Boosting the correct logit decreases loss
// =================================================================
static void test_loss_decreases_with_correct_logit() {
    run("(15) boosting correct logit decreases loss", []{
        LabelSmoothingCrossEntropy loss(0.1);

        Tensor baseline(1, 3);
        baseline[0][0] = 0.0;
        baseline[0][1] = 0.0;
        baseline[0][2] = 0.0;
        Tensor targets(1, 1);
        targets[0][0] = 0;
        Tensor L0 = loss.forward(baseline, targets);

        Tensor boosted(1, 3);
        boosted[0][0] = 5.0;
        boosted[0][1] = 0.0;
        boosted[0][2] = 0.0;
        Tensor L1 = loss.forward(boosted, targets);
        ASSERT(L1[0][0] < L0[0][0]);
    });
}

// =================================================================
// Test 16: BUG REGRESSION — backward() without prior forward() should not crash
// (Was crashing because backward() relied on the last_smooth_targets_ cache which
//  was default-constructed (rows=0, cols=0) when no forward() had run.)
// =================================================================
static void test_backward_without_prior_forward() {
    run("(16) backward() without prior forward() doesn't crash", []{
        LabelSmoothingCrossEntropy loss(0.1);
        Tensor logits(2, 3);
        for (size_t i = 0; i < logits.rows; ++i)
            for (size_t j = 0; j < logits.cols; ++j)
                logits[i][j] = 0.5 * (i + j);
        Tensor targets(2, 1);
        targets[0][0] = 0;
        targets[1][0] = 2;
        Tensor grad = loss.backward(logits, targets);  // no forward() called
        ASSERT(grad.rows == 2);
        ASSERT(grad.cols == 3);
        for (size_t i = 0; i < grad.rows; ++i)
            for (size_t j = 0; j < grad.cols; ++j)
                ASSERT(std::isfinite(grad[i][j]));
    });
}

// =================================================================
// Test 17: Consistent behavior — forward+backward agrees with backward-alone
//   (Verifies the impl computes smooth_targets correctly in both code paths.)
// =================================================================
static void test_forward_then_backward_equals_backward_alone() {
    run("(17) forward+backward gradient == backward-alone gradient", []{
        LabelSmoothingCrossEntropy loss(0.2);
        Tensor logits(2, 3);
        logits[0][0] = 0.1; logits[0][1] = 0.5; logits[0][2] = -0.2;
        logits[1][0] = 0.3; logits[1][1] = -0.4; logits[1][2] = 0.6;
        Tensor targets(2, 1);
        targets[0][0] = 1;
        targets[1][0] = 2;

        loss.forward(logits, targets);
        Tensor g1 = loss.backward(logits, targets);

        // Fresh instance, only call backward (no forward)
        LabelSmoothingCrossEntropy loss2(0.2);
        Tensor g2 = loss2.backward(logits, targets);

        for (size_t i = 0; i < g1.rows; ++i)
            for (size_t j = 0; j < g1.cols; ++j)
                ASSERT_NEAR(g1[i][j], g2[i][j], 1e-15);
    });
}

int main() {
    test_default_smoothing();
    test_custom_smoothing();
    test_forward_smoothing_zero();
    test_forward_with_smoothing_hand_derived();
    test_smoothed_target_distribution();
    test_soft_labels_passed_through();
    test_backward_formula();
    test_gradient_row_sum_zero();
    test_numerical_gradient_single_feature();
    test_numerical_gradient_multibatch();
    test_high_smoothing();
    test_uniform_target_smoothing_one();
    test_forward_nonneg();
    test_backward_shape_and_finite();
    test_loss_decreases_with_correct_logit();
    test_backward_without_prior_forward();
    test_forward_then_backward_equals_backward_alone();

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}