// test_distribution_losses.cpp
// Tests for KL Divergence, JS Divergence, Huber Loss, Quantile Loss
// in include/nn/utils/distribution_losses.h
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "nn/utils/distribution_losses.h"

using std::abs;  // Use std::abs to disambiguate from macro

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
// Test 1: KL Divergence forward — known analytical values
// KL([1,0] || [1,0]) = 0
// KL([1,0] || [0.5,0.5]) = log(2) ≈ 0.693147
// KL([0.5,0.5] || [1,0]) = log(2) (because 0.5 * log(0.5/1) = 0.5*log(0.5) and only the 0.5->1 contributes; sum: 0.5*log(0.5) + 0.5*log(0.5/0.5) = 0.5*log(0.5) but actually 0.5*log(0.5/1)=−log(2)/2 and the second 0.5*log(0.5/0)=−∞. So we only test the well-defined cases.)
// =================================================================
static void test_kl_forward_known_values() {
    std::cout << "-- Test 1: KL Divergence forward known values --" << std::endl;
    KLDivergence kl;

    // Same distribution: KL = 0
    Tensor p(1, 2);
    p[0][0] = 1.0; p[0][1] = 0.0;
    Tensor q_same(1, 2);
    q_same[0][0] = 1.0; q_same[0][1] = 0.0;

    Tensor out = kl.forward(p, q_same);
    CHECK_NEAR(out[0][0], 0.0, 1e-10, "KL([1,0]||[1,0]) = 0");

    // p = [1, 0], q = [0.5, 0.5]:
    //   KL = 1*log(1/0.5) + 0*log(0/0.5) = log(2)
    Tensor q_unif(1, 2);
    q_unif[0][0] = 0.5; q_unif[0][1] = 0.5;

    Tensor out2 = kl.forward(p, q_unif);
    CHECK_NEAR(out2[0][0], std::log(2.0), 1e-10, "KL([1,0]||[0.5,0.5]) = log(2)");

    // Batch of two rows: 0 + log(2) = log(2), then mean = log(2)/2
    Tensor p_batch(2, 2);
    p_batch[0][0] = 1.0; p_batch[0][1] = 0.0;
    p_batch[1][0] = 1.0; p_batch[1][1] = 0.0;
    Tensor q_batch(2, 2);
    q_batch[0][0] = 1.0; q_batch[0][1] = 0.0;
    q_batch[1][0] = 0.5; q_batch[1][1] = 0.5;

    Tensor out3 = kl.forward(p_batch, q_batch);
    CHECK_NEAR(out3[0][0], std::log(2.0) / 2.0, 1e-10, "KL batch mean = log(2)/2");
}

// =================================================================
// Test 2: KL Divergence with batching (uniform vs uniform = log(K))
// For uniform q over K classes and uniform p over K classes:
//   KL = sum_k (1/K) * log((1/K)/(1/K)) = 0
// For uniform p over 3 classes vs delta q at one index:
//   KL(p||q) where p = [1/3, 1/3, 1/3], q = [1, 0, 0]:
//     = (1/3)*log(1/3) + (1/3)*log(inf) - undefined
//   We test uniform-on-uniform = 0
// =================================================================
static void test_kl_zero_for_matching() {
    std::cout << "-- Test 2: KL zero when distributions match --" << std::endl;
    KLDivergence kl;

    // Both uniform on 4 classes -> KL = 0
    Tensor a(2, 4);
    Tensor b(2, 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = 0.25;
            b[i][j] = 0.25;
        }
    }
    Tensor out = kl.forward(a, b);
    CHECK_NEAR(out[0][0], 0.0, 1e-10, "KL(uniform||uniform) = 0");

    // Both equal at one non-trivial distribution -> KL = 0
    Tensor c(1, 3);
    c[0][0] = 0.5; c[0][1] = 0.3; c[0][2] = 0.2;
    Tensor d(1, 3);
    d[0][0] = 0.5; d[0][1] = 0.3; d[0][2] = 0.2;
    Tensor out2 = kl.forward(c, d);
    CHECK_NEAR(out2[0][0], 0.0, 1e-10, "KL(p||p) = 0 for arbitrary p");
}

// =================================================================
// Test 3: KL Divergence gradient via finite differences
// Forward: KL(p || q) = sum_i p_i * (log p_i - log q_i)
// For analytical gradient w.r.t. q_j (when q_j > 0):
//   dL/dq_j = -p_j / q_j  (mean over batch)
// =================================================================
static void test_kl_gradient() {
    std::cout << "-- Test 3: KL gradient via finite differences --" << std::endl;
    KLDivergence kl;

    // Use arbitrary distributions
    Tensor p(1, 3);
    p[0][0] = 0.5; p[0][1] = 0.3; p[0][2] = 0.2;

    Tensor q(1, 3);
    q[0][0] = 0.4; q[0][1] = 0.4; q[0][2] = 0.2;

    // Forward at q
    Tensor out = kl.forward(p, q);

    // Analytical gradient w.r.t. q
    Tensor grad_ana = kl.backward(p, q);

    // Finite difference for each element
    double eps = 1e-5;
    bool all_ok = true;
    for (int j = 0; j < 3; ++j) {
        double qj = q[0][j];
        q[0][j] = qj + eps;
        Tensor out_p = kl.forward(p, q);
        double loss_p = out_p[0][0];
        q[0][j] = qj - eps;
        Tensor out_m = kl.forward(p, q);
        double loss_m = out_m[0][0];
        q[0][j] = qj;

        double grad_num = (loss_p - loss_m) / (2.0 * eps);
        double ana = grad_ana[0][j];
        if (std::abs(grad_num - ana) > 1e-5) {
            std::cout << "  [FAIL] grad q[" << j << "]: analytical=" << ana
                      << " numerical=" << grad_num
                      << " |diff|=" << std::abs(grad_num - ana) << std::endl;
            all_ok = false;
        }
    }
    CHECK(all_ok, "KL gradient matches finite differences (all 3 entries)");

    // Negative: -p_j/q_j for j=0 → -0.5/0.4 = -1.25, divided by batch=1 → -1.25
    double expected_g0 = -p[0][0] / q[0][0];
    CHECK_NEAR(grad_ana[0][0], expected_g0, 1e-10, "dKL/dq_0 analytical = -p_0/q_0");
}

// =================================================================
// Test 4: KL forward returns a single scalar (mean over batch)
// =================================================================
static void test_kl_output_shape() {
    std::cout << "-- Test 4: KL output is scalar --" << std::endl;
    KLDivergence kl;
    Tensor p(3, 4), q(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < 4; ++j) { p[i][j] = 0.25; q[i][j] = 0.1 * (j + 1); s += q[i][j]; }
        for (size_t j = 0; j < 4; ++j) q[i][j] /= s;
    }
    Tensor out = kl.forward(p, q);
    CHECK(out.rows == 1 && out.cols == 1, "KL forward returns 1x1 tensor");
    CHECK(std::isfinite(out[0][0]) && out[0][0] >= 0.0, "KL forward returns finite non-negative scalar");
}

// =================================================================
// Test 5: Jensen-Shannon Divergence — symmetric and bounded
// JSD(p, q) = 0.5 * (KL(p || m) + KL(q || m))  where m = (p+q)/2
// JSD(p, p) = 0
// JSD is symmetric: JSD(p, q) = JSD(q, p)
// JSD is bounded: 0 <= JSD(p, q) <= log(2)
// =================================================================
static void test_js_properties() {
    std::cout << "-- Test 5: JSD properties (symmetric, bounded, self=0) --" << std::endl;
    JSDivergence js;

    // JSD(p, p) = 0
    Tensor p(1, 3);
    p[0][0] = 0.5; p[0][1] = 0.3; p[0][2] = 0.2;
    Tensor q(1, 3);
    q[0][0] = 0.5; q[0][1] = 0.3; q[0][2] = 0.2;
    Tensor out_same = js.forward(p, q);
    CHECK_NEAR(out_same[0][0], 0.0, 1e-10, "JSD(p,p) = 0");

    // Symmetry: JSD(p, q) = JSD(q, p)
    Tensor p2(1, 3);
    p2[0][0] = 0.7; p2[0][1] = 0.2; p2[0][2] = 0.1;
    Tensor q2(1, 3);
    q2[0][0] = 0.1; q2[0][1] = 0.7; q2[0][2] = 0.2;
    Tensor out_pq = js.forward(p2, q2);
    Tensor out_qp = js.forward(q2, p2);
    CHECK_NEAR(out_pq[0][0], out_qp[0][0], 1e-10, "JSD is symmetric");

    // Bounded: 0 <= JSD <= log(2)
    CHECK(out_pq[0][0] >= 0.0, "JSD(p,q) >= 0");
    CHECK(out_pq[0][0] <= std::log(2.0) + 1e-10, "JSD(p,q) <= log(2)");
}

// =================================================================
// Test 6: JS Divergence gradient via finite differences
// JSD(p, q) = 0.5 * (KL(p||m) + KL(q||m))  where m = (p+q)/2
// dL/dq_j = 0.5 * (dKL(p||m)/dq_j + dKL(q||m)/dq_j)
// Use chain rule carefully.
// =================================================================
static void test_js_gradient() {
    std::cout << "-- Test 6: JS gradient via finite differences --" << std::endl;
    JSDivergence js;

    Tensor p(1, 3);
    p[0][0] = 0.5; p[0][1] = 0.3; p[0][2] = 0.2;

    Tensor q(1, 3);
    q[0][0] = 0.4; q[0][1] = 0.4; q[0][2] = 0.2;

    Tensor out = js.forward(p, q);
    CHECK(std::isfinite(out[0][0]), "JSD forward finite");

    // Analytical gradient w.r.t. q
    Tensor grad_ana = js.backward(p, q);

    // Finite differences
    double eps = 1e-5;
    bool all_ok = true;
    for (int j = 0; j < 3; ++j) {
        double qj = q[0][j];
        q[0][j] = qj + eps;
        double loss_p = js.forward(p, q)[0][0];
        q[0][j] = qj - eps;
        double loss_m = js.forward(p, q)[0][0];
        q[0][j] = qj;
        double grad_num = (loss_p - loss_m) / (2.0 * eps);
        if (std::abs(grad_num - grad_ana[0][j]) > 1e-5) {
            std::cout << "  [FAIL] grad q[" << j << "]: analytical=" << grad_ana[0][j]
                      << " numerical=" << grad_num << std::endl;
            all_ok = false;
        }
    }
    CHECK(all_ok, "JSD gradient matches finite differences");
}

// =================================================================
// Test 7: Huber loss forward — piecewise quadratic/linear
// L(x) = 0.5 * x^2         if |x| <= delta
//      = delta * (|x| - 0.5*delta)   if |x| > delta
// Continuous at the boundary: at |x| = delta, both = 0.5*delta^2
// =================================================================
static void test_huber_forward() {
    std::cout << "-- Test 7: Huber loss forward — piecewise formula --" << std::endl;
    HuberLoss huber(1.0);

    Tensor pred(4, 1);
    pred[0][0] = 0.5;    // |err| = 0.4 → quadratic regime
    pred[1][0] = -0.5;   // |err| = 0.4 → quadratic regime (symmetric)
    pred[2][0] = 5.0;    // |err| = 1.1 → linear regime (delta=1)
    pred[3][0] = -5.0;   // |err| = 1.1 → linear regime

    Tensor y(4, 1);
    y[0][0] = -0.1;
    y[1][0] = -0.1;
    y[2][0] = 3.0;       // pred - y = 2.0 → |err|=2
    y[3][0] = -3.8;      // pred - y = -1.2 → |err|=1.2

    Tensor out = huber.forward(pred, y);

    // err[0] = 0.6, |0.6| <= 1, loss = 0.5*0.6^2 = 0.18
    double l0 = 0.5 * 0.6 * 0.6;
    // err[1] = -0.4, |0.4| <= 1, loss = 0.5*0.4^2 = 0.08
    double l1 = 0.5 * 0.4 * 0.4;
    // err[2] = 2.0, |2.0| > 1, loss = 1.0 * (2.0 - 0.5) = 1.5
    double l2 = 1.0 * (2.0 - 0.5);
    // err[3] = -1.2, |1.2| > 1, loss = 1.0 * (1.2 - 0.5) = 0.7
    double l3 = 1.0 * (1.2 - 0.5);
    double expected = (l0 + l1 + l2 + l3) / 4.0;

    CHECK_NEAR(out[0][0], expected, 1e-10, "Huber forward (mean over batch) matches piecewise formula");
}

// =================================================================
// Test 8: Huber loss continuous at boundary x = ±delta
// =================================================================
static void test_huber_continuity() {
    std::cout << "-- Test 8: Huber continuous at boundary --" << std::endl;
    double delta = 0.7;
    HuberLoss huber(delta);

    Tensor p(1, 1); p[0][0] = delta;     // |err| = delta exactly
    Tensor y(1, 1); y[0][0] = 0.0;       // err = delta
    double loss_at_bound = huber.forward(p, y)[0][0];

    // Both formulas give 0.5 * delta^2
    double closed_form = 0.5 * delta * delta;
    CHECK_NEAR(loss_at_bound, closed_form, 1e-12, "Huber at |err|=delta matches closed form");
}

// =================================================================
// Test 9: Huber loss gradient via finite differences
// dL/d(pred_i):
//   If |err_i| <= delta: err_i / N
//   If |err_i| > delta:  delta * sign(err_i) / N
// =================================================================
static void test_huber_gradient() {
    std::cout << "-- Test 9: Huber gradient via finite differences --" << std::endl;
    double delta = 1.0;
    HuberLoss huber(delta);

    Tensor pred(3, 1);
    pred[0][0] = 0.3;     // |err| = 0.3 < delta (quadratic, grad = err)
    pred[1][0] = -0.4;    // |err| = 0.4 < delta (quadratic, grad = err)
    pred[2][0] = 5.0;     // |err| = 2.0 > delta (linear, grad = -delta)

    Tensor y(3, 1);
    y[0][0] = 0.0;
    y[1][0] = 0.0;
    y[2][0] = 3.0;

    Tensor grad_ana = huber.backward(pred, y);

    // Hand-computed: divide by batch=3
    // row 0: err = 0.3, |err| < delta, dL_b/dpred = -err/N = (pred-y)/N
    //        with pred=0.3, y=0: grad = 0.3/3 = 0.1
    // row 1: err = -0.4, |err| < delta, grad = -err/N = (pred-y)/N
    //        with pred=-0.4, y=0: grad = -0.4/3 = -0.13333...
    // row 2: err = 2.0 (pred=5.0, y=3.0), |err| > delta.
    //        dL_b/dpred = +delta when pred > y (i.e. pinball reg slope is +delta).
    //        So grad = +delta/N = +1/3
    CHECK_NEAR(grad_ana[0][0], 0.1, 1e-12, "Huber grad quadratic regime (positive)");
    CHECK_NEAR(grad_ana[1][0], -0.4 / 3.0, 1e-12, "Huber grad quadratic regime (negative)");
    CHECK_NEAR(grad_ana[2][0], +1.0 / 3.0, 1e-12, "Huber grad linear regime (pred > y) = +delta/N");

    // Finite-difference check
    double eps = 1e-5;
    bool all_ok = true;
    for (size_t i = 0; i < 3; ++i) {
        double orig = pred[i][0];
        pred[i][0] = orig + eps;
        double l_p = huber.forward(pred, y)[0][0];
        pred[i][0] = orig - eps;
        double l_m = huber.forward(pred, y)[0][0];
        pred[i][0] = orig;
        double gnum = (l_p - l_m) / (2.0 * eps);
        if (std::abs(gnum - grad_ana[i][0]) > 1e-5) {
            std::cout << "  [FAIL] huber grad row " << i << ": ana=" << grad_ana[i][0]
                      << " num=" << gnum << std::endl;
            all_ok = false;
        }
    }
    CHECK(all_ok, "Huber gradient matches finite differences");
}

// =================================================================
// Test 10: Quantile (pinball) loss — asymmetric L1
// L(y, p, q) = max(q * (y - p), (q - 1) * (y - p))
//            = (y - p) * q     if y >= p  (under-prediction)
//            = (y - p) * (q-1) if y < p   (over-prediction, q-1 is negative)
// For q = 0.5: 0.5 * |y - p|  (half the MAE)
// =================================================================
static void test_quantile_forward() {
    std::cout << "-- Test 10: Quantile (pinball) loss forward --" << std::endl;
    // q = 0.7
    QuantileLoss ql(0.7);

    Tensor pred(4, 1);
    pred[0][0] = 1.0;  // err = y - p, err_under (y > p): loss = q * err
    pred[1][0] = 2.0;
    pred[2][0] = 3.0;  // err = y - p, err_over (y < p): loss = (q-1) * (y-p)
    pred[3][0] = 4.0;

    Tensor y(4, 1);
    y[0][0] = 2.0;    // err = +1 → under-prediction → 0.7 * 1 = 0.7
    y[1][0] = 1.0;    // err = -1 → over-prediction → (0.7-1) * -1 = 0.3
    y[2][0] = 0.0;    // err = -3 → over-prediction → (0.7-1) * -3 = 0.9
    y[3][0] = 5.0;    // err = +1 → under-prediction → 0.7 * 1 = 0.7

    Tensor out = ql.forward(pred, y);
    double expected = (0.7 + 0.3 + 0.9 + 0.7) / 4.0;
    CHECK_NEAR(out[0][0], expected, 1e-12, "Quantile forward (q=0.7) matches pinball formula");

    // q = 0.5 should equal 0.5 * mean(|err|)
    QuantileLoss qhalf(0.5);
    Tensor errs(4, 1);
    errs[0][0] = 1.0;
    errs[1][0] = -1.0;
    errs[2][0] = -3.0;
    errs[3][0] = 1.0;
    Tensor zeros(4, 1);
    for (size_t i = 0; i < 4; ++i) zeros[i][0] = 0.0;
    Tensor out_half = qhalf.forward(errs, zeros);
    // (errs[i] are the preds, y=0): err = y - pred = -errs[i]
    //   pred=1:   err=-1, over   -> (0.5-1)*-1 = 0.5
    //   pred=-1:  err=+1, under  -> 0.5*1 = 0.5
    //   pred=-3:  err=+3, under  -> 0.5*3 = 1.5
    //   pred=1:   err=-1, over   -> 0.5
    // mean = (0.5 + 0.5 + 1.5 + 0.5) / 4 = 0.75
    double expected_half = (0.5 + 0.5 + 1.5 + 0.5) / 4.0;
    CHECK_NEAR(out_half[0][0], expected_half, 1e-12, "Quantile q=0.5 forward equals hand-computed pinball");
}

// =================================================================
// Test 11: Quantile loss gradient via finite differences
// dL/d(pred_i):
//   if y >= p: grad = -q / N
//   if y <  p: grad = -(q-1) / N = (1-q) / N
// =================================================================
static void test_quantile_gradient() {
    std::cout << "-- Test 11: Quantile gradient via finite differences --" << std::endl;
    QuantileLoss ql(0.3);

    // Avoid exact-equal rows because pinball is non-differentiable at the kink
    // (numerical FD picks the subgradient (0.5-q)/N while the analytical picks
    // either -q/N or (1-q)/N — both valid subgradients).
    Tensor pred(3, 1);
    pred[0][0] = 1.0;  // y > p: under-pred, grad = -q/N
    pred[1][0] = 1.0;  // y < p: over-pred, grad = (1-q)/N
    pred[2][0] = 2.0;  // y > p (strictly): grad = -q/N
    Tensor y(3, 1);
    y[0][0] = 2.0;    // under-pred
    y[1][0] = 0.0;    // over-pred
    y[2][0] = 2.001;  // strictly under-pred (err = 0.001 > 0)

    Tensor grad_ana = ql.backward(pred, y);

    // row 0: under-pred, grad = -q/N = -0.3/3 = -0.1
    CHECK_NEAR(grad_ana[0][0], -0.1, 1e-12, "Quantile grad under-prediction = -q/N");
    // row 1: over-pred, grad = (1-q)/N = 0.7/3 ≈ 0.23333
    CHECK_NEAR(grad_ana[1][0], 0.7 / 3.0, 1e-12, "Quantile grad over-prediction = (1-q)/N");
    // row 2: strict under-pred, err = 0.001 > 0, grad = -q/N = -0.1
    CHECK_NEAR(grad_ana[2][0], -0.1, 1e-12, "Quantile grad strict under-prediction = -q/N");

    // Finite-difference check
    double eps = 1e-5;
    bool all_ok = true;
    for (size_t i = 0; i < 3; ++i) {
        double orig = pred[i][0];
        pred[i][0] = orig + eps;
        double l_p = ql.forward(pred, y)[0][0];
        pred[i][0] = orig - eps;
        double l_m = ql.forward(pred, y)[0][0];
        pred[i][0] = orig;
        double gnum = (l_p - l_m) / (2.0 * eps);
        if (std::abs(gnum - grad_ana[i][0]) > 1e-5) {
            std::cout << "  [FAIL] quantile grad row " << i << ": ana=" << grad_ana[i][0]
                      << " num=" << gnum << std::endl;
            all_ok = false;
        }
    }
    CHECK(all_ok, "Quantile gradient matches finite differences");
}

// =================================================================
// Test 12: End-to-end training — KL loss as VAE/KL regularizer
// Train a single Dense layer to make p_pred match p_target via KL(p_pred||p_target)
// =================================================================
static Tensor softmax_row(const Tensor& logits) {
    size_t N = logits.rows;
    size_t K = logits.cols;
    Tensor out(N, K);
    for (size_t b = 0; b < N; ++b) {
        double max_l = logits[b][0];
        for (size_t k = 0; k < K; ++k) max_l = std::max(max_l, logits[b][k]);
        double sum_e = 0.0;
        for (size_t k = 0; k < K; ++k) {
            out[b][k] = std::exp(logits[b][k] - max_l);
            sum_e += out[b][k];
        }
        for (size_t k = 0; k < K; ++k) out[b][k] /= sum_e;
    }
    return out;
}

static void test_kl_endtoend() {
    std::cout << "-- Test 12: KL end-to-end training reduces loss --" << std::endl;
    KLDivergence kl;

    // Target distribution
    Tensor p_target(1, 3);
    p_target[0][0] = 0.1; p_target[0][1] = 0.6; p_target[0][2] = 0.3;

    // Mock model: predict probabilities via softmax(logits)
    Tensor logits(1, 3);
    logits[0][0] = 0.0; logits[0][1] = 0.0; logits[0][2] = 0.0;

    // Initial loss: KL(softmax(logits=0) || p_target). softmax([0,0,0]) = [1/3, 1/3, 1/3]
    Tensor probs_init = softmax_row(logits);
    double loss_initial = kl.forward(probs_init, p_target)[0][0];

    // 50 SGD steps with lr=0.5, gradient on logits = probs - p_target (cross-entropy style)
    // This is the well-known gradient: d/dz_j KL(softmax(z) || p_target) = p_j - target_j.
    double lr = 0.5;
    for (int step = 0; step < 50; ++step) {
        Tensor probs = softmax_row(logits);
        for (int j = 0; j < 3; ++j) {
            double grad = probs[0][j] - p_target[0][j];
            logits[0][j] -= lr * grad;
        }
    }

    // Recompute probs and KL after training
    Tensor probs_final = softmax_row(logits);
    double loss_final = kl.forward(probs_final, p_target)[0][0];

    CHECK(loss_final < loss_initial, "KL end-to-end: loss decreased after 50 SGD steps");
    CHECK(loss_initial - loss_final > 0.1, "KL end-to-end: substantial loss reduction (>0.1)");
    // Also verify probs got close to target
    CHECK(std::abs(probs_final[0][0] - p_target[0][0]) < 0.05, "KL end-to-end: probs[0] close to target[0]");
}

// =================================================================
// Test 13: Huber loss end-to-end regression training
// Train a single scalar y_pred to match y=3.0 from y_pred_init=0.0
// =================================================================
static void test_huber_e2e() {
    std::cout << "-- Test 13: Huber end-to-end regression training --" << std::endl;
    HuberLoss huber(1.0);
    Tensor pred(1, 1);
    pred[0][0] = 0.0;
    Tensor y(1, 1);
    y[0][0] = 3.0;
    double lr = 0.1;
    double l0 = huber.forward(pred, y)[0][0];
    for (int step = 0; step < 30; ++step) {
        double grad = huber.backward(pred, y)[0][0];
        pred[0][0] -= lr * grad;
        double new_err = std::abs(pred[0][0] - y[0][0]);
        // If we entered linear regime, clamp the effective gradient to delta/N
        // (backward already does this; just simulate)
        if (new_err > 1.0) {
            // sign(err) * delta / N
            double sign = (pred[0][0] < y[0][0]) ? 1.0 : -1.0;
            double clamped = sign * 1.0;  // delta=1
            // Grad = clamped (already accounted by huber.backward)
            (void)clamped;
        }
    }
    double l1 = huber.forward(pred, y)[0][0];
    CHECK(l1 < l0, "Huber e2e: loss decreased");
    CHECK(pred[0][0] > 0.5, "Huber e2e: pred moved toward target (still sublinear if regime-dependent)");
}

// =================================================================
// Test 14: Edge cases
// =================================================================
static void test_edge_cases() {
    std::cout << "-- Test 14: Edge cases (q_zeros, p_zeros, q=1.0) --" << std::endl;
    KLDivergence kl;

    // p = [0, 1], q = [0, 1]: both delta on same class, KL = 0
    Tensor p(1, 2); p[0][0] = 0.0; p[0][1] = 1.0;
    Tensor q(1, 2); q[0][0] = 0.0; q[0][1] = 1.0;
    Tensor out = kl.forward(p, q);
    CHECK_NEAR(out[0][0], 0.0, 1e-10, "KL delta||delta on same class = 0");

    // Huber's gradient should be exactly zero at exact match
    HuberLoss huber(1.0);
    Tensor pm(1, 1); pm[0][0] = 2.0;
    Tensor ym(1, 1); ym[0][0] = 2.0;
    double zero_grad = huber.backward(pm, ym)[0][0];
    CHECK_NEAR(zero_grad, 0.0, 1e-12, "Huber gradient at exact match = 0");
}

int main() {
    std::cout << std::setprecision(12);
    std::cout << "=== Distribution Losses Tests ===" << std::endl;

    test_kl_forward_known_values();
    test_kl_zero_for_matching();
    test_kl_gradient();
    test_kl_output_shape();
    test_js_properties();
    test_js_gradient();
    test_huber_forward();
    test_huber_continuity();
    test_huber_gradient();
    test_quantile_forward();
    test_quantile_gradient();
    test_kl_endtoend();
    test_huber_e2e();
    test_edge_cases();

    std::cout << std::endl;
    std::cout << "=== Summary: " << total_passed << " passed, " << total_failed << " failed ===" << std::endl;
    return total_failed == 0 ? 0 : 1;
}
