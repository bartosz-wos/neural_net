// test_gumbel_softmax.cpp — Tests for the Gumbel-Softmax layer
// (Jang et al. 2017, Maddison et al. 2017).
//
// GumbelSoftmax maps (B, K) logits to (B, K) "soft categorical" samples.
// The hard (Straight-Through Estimator) variant returns one-hot at forward
// time but flows the gradient through the soft path during backward.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
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

// =====================================================================
// Test 1: Forward shape and finiteness (soft mode)
// =====================================================================
static void test_forward_shape_soft() {
    cout << endl << "-- Test 1: forward shape (soft) --" << endl;
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/42);
    Tensor logits(3, 5);
    logits[0][0] = 1.0;  logits[0][1] = 2.0;  logits[0][2] = 0.5;
    logits[0][3] = -1.0; logits[0][4] = 0.0;
    logits[1][0] = 0.1;  logits[1][1] = 0.2;  logits[1][2] = 0.3;
    logits[1][3] = 0.4;  logits[1][4] = 0.5;
    logits[2][0] = -2.0; logits[2][1] = -2.0; logits[2][2] = -2.0;
    logits[2][3] = -2.0; logits[2][4] = -2.0;

    Tensor out = gs.forward(logits);
    check("output shape (3, 5)", out.rows == 3 && out.cols == 5);
    bool finite = true;
    for (size_t i = 0; i < out.rows * out.cols; ++i) {
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    }
    check("output all finite", finite);

    // Rows sum to ~1 (softmax)
    bool row_sums_ok = true;
    for (size_t b = 0; b < out.rows; ++b) {
        double s = 0.0;
        for (size_t j = 0; j < out.cols; ++j) s += out[b][j];
        if (std::fabs(s - 1.0) > 1e-9) row_sums_ok = false;
    }
    check("soft output rows sum to 1", row_sums_ok);
}

// =====================================================================
// Test 2: Hard (straight-through) forward returns one-hot, exact row sum = 1
// =====================================================================
static void test_forward_hard_one_hot() {
    cout << endl << "-- Test 2: hard forward returns one-hot, row sum = 1 --" << endl;
    GumbelSoftmax gs(0.5, /*hard=*/true, /*seed=*/7);
    Tensor logits(2, 4);
    // Row 0: argmax should be j=2 (logit 5.0)
    logits[0][0] = 1.0; logits[0][1] = 2.0; logits[0][2] = 5.0; logits[0][3] = 0.5;
    // Row 1: argmax should be j=0 (logit 10.0)
    logits[1][0] = 10.0; logits[1][1] = -3.0; logits[1][2] = 1.0; logits[1][3] = -1.0;

    Tensor out = gs.forward(logits);
    check("hard output shape (2, 4)", out.rows == 2 && out.cols == 4);

    bool row_sums_one = true;
    bool one_one_per_row = true;
    bool argmax_correct = true;
    for (size_t b = 0; b < out.rows; ++b) {
        double s = 0.0;
        size_t one_idx = out.cols;
        size_t n_ones = 0;
        for (size_t j = 0; j < out.cols; ++j) {
            double v = out[b][j];
            s += v;
            // values are 0 or 1 exactly (no fp slop from any accumulation)
            if (v == 1.0) { one_idx = j; ++n_ones; }
            else if (v != 0.0) { one_idx = out.cols; }  // unexpected non-0/non-1
        }
        if (s != 1.0) row_sums_one = false;
        if (n_ones != 1) one_one_per_row = false;
        // Compare argmax from logits vs one-hot position
        size_t true_argmax = 0; double best = logits[b][0];
        for (size_t j = 1; j < logits.cols; ++j) {
            if (logits[b][j] > best) { best = logits[b][j]; true_argmax = j; }
        }
        if (one_idx != true_argmax) argmax_correct = false;
    }
    check("hard row sums exactly to 1", row_sums_one);
    check("hard each row has exactly one 1.0", one_one_per_row);
    check("hard 1.0 is at argmax(logits)", argmax_correct);
}

// =====================================================================
// Test 3: Temperature effect — small tau concentrates, large tau smooths
// =====================================================================
static void test_temperature_effect() {
    cout << endl << "-- Test 3: temperature concentrates / smooths --" << endl;
    Tensor logits(1, 4);
    logits[0][0] = 1.0; logits[0][1] = 2.0; logits[0][2] = 3.0; logits[0][3] = 4.0;

    // Small temperature: max prob should be much larger than uniform.
    GumbelSoftmax cold(0.1, /*hard=*/false, /*seed=*/123);
    Tensor out_cold = cold.forward(logits);
    double max_cold = 0.0;
    for (size_t j = 0; j < logits.cols; ++j) {
        if (out_cold[0][j] > max_cold) max_cold = out_cold[0][j];
    }

    // Large temperature: max prob should be near 0.25.
    GumbelSoftmax warm(10.0, /*hard=*/false, /*seed=*/123);
    Tensor out_warm = warm.forward(logits);
    double max_warm = 0.0;
    for (size_t j = 0; j < logits.cols; ++j) {
        if (out_warm[0][j] > max_warm) max_warm = out_warm[0][j];
    }

    // (With Gumbel noise on a 4-class logit vector, even at tau=0.1 the
    // argmax can occasionally shift — so test the *trend* in expectation
    // by averaging many samples rather than asserting exact values.)
    auto avg_max = [&](double tau) {
        GumbelSoftmax g(tau, /*hard=*/false, /*seed=*/0);
        const int N = 200;
        double s = 0.0;
        for (int i = 0; i < N; ++i) {
            Tensor o = g.forward(logits);
            double m = 0.0;
            for (size_t j = 0; j < logits.cols; ++j) {
                if (o[0][j] > m) m = o[0][j];
            }
            s += m;
        }
        return s / N;
    };
    double avg_cold = avg_max(0.1);
    double avg_warm = avg_max(10.0);
    cout << "    avg max-prob at tau=0.1:  " << avg_cold << endl;
    cout << "    avg max-prob at tau=10.0: " << avg_warm << endl;
    check("small tau -> larger avg max-prob than large tau", avg_cold > avg_warm);
    check("small tau avg max-prob > 0.5", avg_cold > 0.5);
    check("large tau avg max-prob < 0.4", avg_warm < 0.4);
}

// =====================================================================
// Test 4: Reproducibility with seed, distinct without
// =====================================================================
static void test_seed_reproducibility() {
    cout << endl << "-- Test 4: seeded RNG reproduces; eval mode is deterministic --" << endl;
    Tensor logits(2, 3);
    logits[0][0] = 0.1; logits[0][1] = 0.5; logits[0][2] = 0.9;
    logits[1][0] = 0.4; logits[1][1] = 0.0; logits[1][2] = 0.3;

    GumbelSoftmax a(1.0, /*hard=*/false, /*seed=*/99);
    GumbelSoftmax b(1.0, /*hard=*/false, /*seed=*/99);
    Tensor ya = a.forward(logits);
    Tensor yb = b.forward(logits);
    bool same = true;
    for (size_t i = 0; i < ya.rows * ya.cols; ++i) {
        if (ya.data[i] != yb.data[i]) { same = false; break; }
    }
    check("same seed -> identical soft forward", same);

    GumbelSoftmax c(1.0, /*hard=*/false, /*seed=*/1);
    GumbelSoftmax d(1.0, /*hard=*/false, /*seed=*/2);
    Tensor yc = c.forward(logits);
    Tensor yd = d.forward(logits);
    bool different = false;
    for (size_t i = 0; i < yc.rows * yc.cols; ++i) {
        if (yc.data[i] != yd.data[i]) { different = true; break; }
    }
    check("different seeds -> different soft forward", different);

    // Eval mode is deterministic across instances.
    GumbelSoftmax e1(1.0, /*hard=*/false, /*seed=*/111);
    GumbelSoftmax e2(1.0, /*hard=*/false, /*seed=*/222);
    e1.set_training(false);
    e2.set_training(false);
    Tensor ye1 = e1.forward(logits);
    Tensor ye2 = e2.forward(logits);
    bool eval_same = true;
    for (size_t i = 0; i < ye1.rows * ye1.cols; ++i) {
        if (ye1.data[i] != ye2.data[i]) { eval_same = false; break; }
    }
    check("eval mode -> deterministic across seeds", eval_same);

    // Eval mode output equals softmax(logits / tau) — no noise.
    Tensor eval_input(1, 3);
    eval_input[0][0] = 1.0; eval_input[0][1] = 2.0; eval_input[0][2] = 3.0;
    GumbelSoftmax ev(0.5, /*hard=*/false, /*seed=*/0);
    ev.set_training(false);
    Tensor ye = ev.forward(eval_input);
    // Reference: softmax([1, 2, 3] / 0.5) = softmax([2, 4, 6])
    // exp(2) = 7.389, exp(4) = 54.598, exp(6) = 403.429, sum = 465.416
    double e2_ = std::exp(2.0), e4 = std::exp(4.0), e6 = std::exp(6.0);
    double s = e2_ + e4 + e6;
    double expected0 = e2_ / s, expected1 = e4 / s, expected2 = e6 / s;
    bool eval_correct =
        std::fabs(ye[0][0] - expected0) < 1e-9 &&
        std::fabs(ye[0][1] - expected1) < 1e-9 &&
        std::fabs(ye[0][2] - expected2) < 1e-9;
    check("eval forward matches softmax(logits / tau)", eval_correct);
}

// =====================================================================
// Test 5: No learnable parameters
// =====================================================================
static void test_no_parameters() {
    cout << endl << "-- Test 5: no parameters / no gradients --" << endl;
    GumbelSoftmax gs(1.0, /*hard=*/true, /*seed=*/0);
    check("parameters() is empty", gs.parameters().empty());
    check("gradients() is empty", gs.gradients().empty());
    check("get_weights() is empty (0,0)", gs.get_weights().rows == 0 && gs.get_weights().cols == 0);
    check("get_gradients() is empty (0,0)", gs.get_gradients().rows == 0 && gs.get_gradients().cols == 0);
    // update_weights / zero_grad are no-ops; just verify they don't throw
    gs.update_weights(0.01);
    gs.zero_grad();
    check("update_weights and zero_grad do not throw", true);
}

// =====================================================================
// Test 6: Backward shape
// =====================================================================
static void test_backward_shape() {
    cout << endl << "-- Test 6: backward shape --" << endl;
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/0);
    Tensor logits(2, 4);
    for (size_t i = 0; i < logits.rows; ++i)
        for (size_t j = 0; j < logits.cols; ++j)
            logits[i][j] = 0.1 * (i * logits.cols + j);

    Tensor out = gs.forward(logits);
    Tensor grad_output(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            grad_output[i][j] = 1.0;  // sum-loss gradient

    Tensor grad_logits = gs.backward(grad_output, 0.0);
    check("backward returns (B, K)", grad_logits.rows == 2 && grad_logits.cols == 4);

    bool finite = true;
    for (size_t i = 0; i < grad_logits.rows * grad_logits.cols; ++i) {
        if (!std::isfinite(grad_logits.data[i])) { finite = false; break; }
    }
    check("backward output all finite", finite);
}

// =====================================================================
// Test 7: Backward — analytical matches numerical finite-difference (soft)
// =====================================================================
static void test_backward_soft_vs_numerical() {
    cout << endl << "-- Test 7: soft backward matches numerical --" << endl;
    // Use eval mode so forward is the deterministic softmax(logits / tau) —
    // otherwise the Gumbel noise sampled by each forward call would make the
    // central-difference numerical gradient inconsistent with the analytical
    // gradient (which uses the noise realization cached during the analytical
    // forward pass).
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/0);
    gs.set_training(false);

    Tensor logits(2, 4);
    // Use moderate logit magnitudes so softmax is well-conditioned.
    for (size_t i = 0; i < logits.rows; ++i)
        for (size_t j = 0; j < logits.cols; ++j)
            logits[i][j] = 0.4 * (i * logits.cols + j) - 0.3;

    // Sum-loss gradient
    Tensor y = gs.forward(logits);
    Tensor grad_output(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_output[i][j] = 1.0;
    Tensor d_logits_ana = gs.backward(grad_output, 0.0);

    // Numerical gradient via finite differences (central diff)
    const double eps = 1e-5;
    bool all_pass = true;
    double max_re = 0.0;
    for (size_t b = 0; b < logits.rows; ++b) {
        for (size_t k = 0; k < logits.cols; ++k) {
            Tensor lp = logits;  lp(b, k) += eps;
            Tensor lm = logits;  lm(b, k) -= eps;
            Tensor yp = gs.forward(lp);
            Tensor ym = gs.forward(lm);
            double sp = 0.0, sm = 0.0;
            for (size_t i = 0; i < yp.rows; ++i)
                for (size_t j = 0; j < yp.cols; ++j) sp += yp(i, j);
            for (size_t i = 0; i < ym.rows; ++i)
                for (size_t j = 0; j < ym.cols; ++j) sm += ym(i, j);
            double num = (sp - sm) / (2.0 * eps);
            double ana = d_logits_ana(b, k);
            double scale = std::max({1e-12, std::fabs(ana), std::fabs(num)});
            double re = std::fabs(ana - num) / scale;
            if (re > max_re) max_re = re;
            cout << "    (" << b << "," << k << ") ana=" << ana
                 << " num=" << num << " rel_err=" << re << endl;
            // Tolerance scales with the magnitude of the gradient — when
            // both analytical and numerical are at the finite-difference
            // noise floor (very small), we use an absolute tolerance
            // instead of a relative one.
            double tol = std::max(1e-6, 1e-3 * scale);
            if (std::fabs(ana - num) > tol) all_pass = false;
        }
    }
    cout << "    max rel_err: " << max_re << endl;
    check("soft backward matches numerical (rel_err < 1e-3) per entry", all_pass);
}

// =====================================================================
// Test 8: Backward — straight-through (hard forward) backward equals soft
// backward for the same cached y_soft. This is the canonical definition
// of the STE: the gradient flows through the soft relaxation regardless
// of whether the forward is hard or soft.
// (We cannot use a direct central-difference numerical gradient on the
// hard forward because the one-hot output is a step function — its
// finite-difference gradient is 0 almost everywhere.)
// =====================================================================
static void test_backward_hard_equals_soft() {
    cout << endl << "-- Test 8: hard (STE) backward == soft backward --" << endl;
    Tensor logits(1, 4);
    logits[0][0] = 1.5; logits[0][1] = 0.2; logits[0][2] = -0.4; logits[0][3] = 0.7;

    // Run twice with the same seed: once soft, once hard.
    GumbelSoftmax gs_soft(1.0, /*hard=*/false, /*seed=*/1234);
    Tensor y_soft_fwd = gs_soft.forward(logits);
    Tensor grad_output(1, 4);
    for (size_t i = 0; i < y_soft_fwd.rows; ++i)
        for (size_t j = 0; j < y_soft_fwd.cols; ++j)
            grad_output[i][j] = 1.0;
    Tensor d_soft = gs_soft.backward(grad_output, 0.0);

    GumbelSoftmax gs_hard(1.0, /*hard=*/true, /*seed=*/1234);
    Tensor y_hard_fwd = gs_hard.forward(logits);
    // The hard forward must actually be one-hot at the argmax of the soft one.
    bool hard_is_onehot = true;
    size_t n_ones = 0;
    size_t argmax_soft = 0;
    double best = y_soft_fwd[0][0];
    for (size_t j = 0; j < y_soft_fwd.cols; ++j) {
        if (y_soft_fwd[0][j] > best) { best = y_soft_fwd[0][j]; argmax_soft = j; }
        if (y_hard_fwd[0][j] == 1.0) ++n_ones;
        else if (y_hard_fwd[0][j] != 0.0) hard_is_onehot = false;
    }
    check("hard forward is one-hot", n_ones == 1 && hard_is_onehot);
    check("hard one is at argmax of soft", y_hard_fwd[0][argmax_soft] == 1.0);

    Tensor d_hard = gs_hard.backward(grad_output, 0.0);

    // STE backward must match soft backward exactly (same cached y_soft).
    bool identical = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < d_hard.rows * d_hard.cols; ++i) {
        double diff = std::fabs(d_hard.data[i] - d_soft.data[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-12) identical = false;
    }
    cout << "    max |d_hard - d_soft| = " << max_diff << endl;
    check("STE backward == soft backward bit-exactly", identical);

    // And: STE backward must be non-trivially different from "return grad_output as-is".
    bool not_passthrough = false;
    for (size_t i = 0; i < d_hard.rows * d_hard.cols; ++i) {
        if (std::fabs(d_hard.data[i] - grad_output.data[i]) > 1e-6) {
            not_passthrough = true; break;
        }
    }
    check("STE backward is not grad_output passthrough", not_passthrough);
}

// =====================================================================
// Test 9: set_temperature clamps and applies
// =====================================================================
static void test_set_temperature() {
    cout << endl << "-- Test 9: set_temperature / set_training --" << endl;
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/0);
    check("initial temperature is 1.0", gs.temperature() == 1.0);

    gs.set_temperature(0.5);
    check("set_temperature(0.5) -> 0.5", gs.temperature() == 0.5);

    // Negative / zero should clamp to a small positive value, not crash.
    gs.set_temperature(-1.0);
    check("set_temperature(-1.0) clamps to small positive", gs.temperature() > 0.0);
    check("set_temperature(-1.0) clamps to <= 1e-6", gs.temperature() <= 1e-6);

    gs.set_training(false);
    check("set_training(false)", !gs.training());
    gs.set_training(true);
    check("set_training(true)", gs.training());
}

// =====================================================================
// Test 10: MiniVAE-style end-to-end — Dense -> GumbelSoftmax in a loop
// reduces loss on a synthetic task. Verifies the layer is properly
// composable in a real training pipeline.
// =====================================================================
static void test_vae_style_endtoend() {
    cout << endl << "-- Test 10: VAE-style training reduces loss --" << endl;
    // Tiny "encoder": input (1, 4) -> Dense(4, 4) -> Dense(4, 3) -> GumbelSoftmax(3) -> loss.
    // We use SOFT Gumbel-Softmax here (not hard STE) so the loss is
    // differentiable everywhere and the gradient signal is continuous.
    // Target is a soft distribution the network has to match.

    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    Tensor x(1, 4);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = uni(rng);
    Tensor target(1, 3);
    target[0][0] = 0.1; target[0][1] = 0.1; target[0][2] = 0.8;

    Dense enc1(4, 4);
    Dense enc2(4, 3);
    // Use eval mode so forward is the deterministic temperature-scaled
    // softmax — same math as Softmax + cross-entropy, no Gumbel noise.
    // (The Gumbel-noised forward is tested separately in tests 1-9.)
    // This isolates the gradient-chain smoke test from noise-induced
    // gradient variance.
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/0);
    gs.set_training(false);

    auto sgd_step = [&]() {
        Tensor h1 = enc1.forward(x);
        Tensor logits = enc2.forward(h1);
        Tensor y = gs.forward(logits);
        // Soft cross-entropy: L = -sum target * log(y + eps).  y is a row
        // softmax (sums to 1) so the loss is a real number in [-log(1+eps), ~0].
        const double eps_l = 1e-9;
        double L = 0.0;
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                L += -target[i][j] * std::log(y[i][j] + eps_l);
        // dL/dy = -target / (y + eps_l)
        Tensor dLdy(target.rows, target.cols);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                dLdy[i][j] = -target[i][j] / (y[i][j] + eps_l);
        Tensor dlogits = gs.backward(dLdy, 0.0);
        Tensor dh1 = enc2.backward(dlogits, 0.0);
        enc1.backward(dh1, 0.0);
        const double lr = 0.05;
        enc1.update_weights(lr);
        enc2.update_weights(lr);
        enc1.zero_grad();
        enc2.zero_grad();
        return L;
    };

    const int STEPS = 80;
    double L0 = sgd_step();
    double L_last = L0;
    for (int i = 1; i < STEPS; ++i) {
        L_last = sgd_step();
    }
    cout << "    L0=" << L0 << " L_last=" << L_last << endl;
    check("end-to-end: initial loss is meaningful (not trivially 0)", L0 > 0.1);
    check("end-to-end: loss decreases over SGD steps", L_last < L0);
    check("end-to-end: loss dropped > 20%", L_last < 0.8 * L0);
}

// =====================================================================
// Test 11: Mutation hygiene — the gradient is *not* identity, and matches
// the analytical softmax Jacobian against the *no-noise* deterministic
// forward (eval mode).
// =====================================================================
static void test_gradient_is_not_identity() {
    cout << endl << "-- Test 11: gradient reflects softmax Jacobian --" << endl;
    GumbelSoftmax gs(1.0, /*hard=*/false, /*seed=*/0);
    gs.set_training(false);  // deterministic softmax(logits / tau) — no Gumbel noise
    Tensor logits(1, 4);
    logits[0][0] = 1.0; logits[0][1] = 2.0; logits[0][2] = 0.0; logits[0][3] = -1.0;
    Tensor y = gs.forward(logits);
    Tensor grad_output(1, 4);
    grad_output[0][0] = 1.0; grad_output[0][1] = 0.0;
    grad_output[0][2] = 0.0; grad_output[0][3] = 0.0;
    Tensor d_logits = gs.backward(grad_output, 0.0);

    // softmax([1,2,0,-1]) / 1.0:
    double e1 = std::exp(1), e2 = std::exp(2), e0 = std::exp(0), em1 = std::exp(-1);
    double s = e1 + e2 + e0 + em1;
    double y0 = e1 / s, y1 = e2 / s, y2 = e0 / s, y3 = em1 / s;
    // Jacobian diag*off-diag applied to grad = [1,0,0,0]:
    //   d_logits[0,k] = (1/tau) * y_k * (grad_k - sum_i grad_i * y_i)
    //                  = y_k * (delta_{k0} - y_0)
    double expected[4] = {
        y0 * (1.0 - y0),
        y1 * (0.0 - y0),
        y2 * (0.0 - y0),
        y3 * (0.0 - y0),
    };
    bool matches = true;
    for (size_t k = 0; k < 4; ++k) {
        if (std::fabs(d_logits[0][k] - expected[k]) > 1e-12) {
            matches = false;
            cout << "    k=" << k << " ana=" << d_logits[0][k]
                 << " expected=" << expected[k] << endl;
        }
    }
    check("gradient matches analytical softmax Jacobian", matches);

    // Also: if the bug were "return grad_output", d_logits[0][0] would be 1.0
    // not y0*(1-2*y0) ~= 0.196. Sanity-check that the magnitude differs.
    bool not_identity = std::fabs(d_logits[0][0] - 1.0) > 1e-3;
    check("gradient is not the identity", not_identity);
}

int main() {
    cout << "=== Gumbel-Softmax Tests ===" << endl;
    test_forward_shape_soft();
    test_forward_hard_one_hot();
    test_temperature_effect();
    test_seed_reproducibility();
    test_no_parameters();
    test_backward_shape();
    test_backward_soft_vs_numerical();
    test_backward_hard_equals_soft();
    test_set_temperature();
    test_vae_style_endtoend();
    test_gradient_is_not_identity();
    cout << endl << "================================" << endl;
    cout << "Total: " << (passed + failed) << ", Passed: " << passed
         << ", Failed: " << failed << endl;
    return failed == 0 ? 0 : 1;
}