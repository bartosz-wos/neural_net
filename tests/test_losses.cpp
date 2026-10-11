// ============================================================================
// Canonical classification losses test suite
// For include/nn/utils/losses.{h,cpp}
//
// Reference math (verified against PyTorch source 2026-10-11, NOT from memory):
//   aten/src/ATen/native/LossNLL.cpp:
//     :166-305  nll_loss_out_frame        forward + total_weight bookkeeping
//     :345-416  nll_loss_backward_out_frame  backward, ignored rows stay zero
//     :498-563  cross_entropy_loss_prob_target
//   aten/src/ATen/native/Loss.cpp:349-364  binary_cross_entropy_with_logits
//
// The three details every hand-rolled version of this gets wrong, each pinned
// by its own test:
//   D1  mean divides by total_weight = (B - n_ignored), NOT by the batch size
//   D2  an ignored row's gradient row is EXACTLY zero, not merely small
//   D3  with class weights, mean divides by the WEIGHT SUM, not the batch size
//   D7  BCE mean divides by numel (rows*cols), not by rows
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <functional>
#include <algorithm>
#include "nn/utils/losses.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"

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

// =================================================================
// Test 1: log_softmax — hand-derived, stable, row-sums
// =================================================================
static void test_log_softmax_hand() {
    run("(1a) log_softmax hand-derived", [] {
        Tensor x(1, 2); x[0][0] = 0.0; x[0][1] = std::log(2.0);
        Tensor lp = log_softmax(x);
        ASSERT_NEAR(lp[0][0], -std::log(3.0), 1e-12);
        ASSERT_NEAR(lp[0][1], std::log(2.0) - std::log(3.0), 1e-12);
        double s = std::exp(lp[0][0]) + std::exp(lp[0][1]);
        ASSERT_NEAR(s, 1.0, 1e-12);
    });
}

// A +1e4 logit must not overflow exp(); the max-subtraction is what saves it.
static void test_log_softmax_stability() {
    run("(1b) log_softmax stable at extreme logits", [] {
        Tensor x(1, 3);
        x[0][0] = 1e4; x[0][1] = -1e4; x[0][2] = 0.0;
        Tensor lp = log_softmax(x);
        ASSERT(std::isfinite(lp[0][0]) && std::isfinite(lp[0][1]));
        ASSERT_NEAR(lp[0][0], 0.0, 1e-9);          // the max row is ~0
        ASSERT_NEAR(lp[0][1], -2e4, 1e-6);         // dominated
        double s = std::exp(lp[0][0]) + std::exp(lp[0][1]) + std::exp(lp[0][2]);
        ASSERT_NEAR(s, 1.0, 1e-12);
    });
}

// log_softmax_backward: the row sum of the gradient is preserved (shift-invariance).
static void test_log_softmax_backward() {
    run("(1c) log_softmax_backward", [] {
        Tensor x(2, 3);
        x[0][0] = 0.5; x[0][1] = -1.0; x[0][2] = 2.0;
        x[1][0] = 0.0; x[1][1] = 0.25; x[1][2] = -0.5;
        Tensor lp = log_softmax(x);
        Tensor g(2, 3);
        g[0][0] = 1.0; g[0][1] = 0.5; g[0][2] = -1.5;
        g[1][0] = 0.0; g[1][1] = 1.0; g[1][2] = 0.0;
        Tensor dx = log_softmax_backward(g, lp);
        double r0 = dx[0][0] + dx[0][1] + dx[0][2];
        double r1 = dx[1][0] + dx[1][1] + dx[1][2];
        ASSERT_NEAR(r0, 0.0, 1e-12);   // sum over g is preserved exactly
        ASSERT_NEAR(r1, 0.0, 1e-12);
        // FD check on one cell.
        const double eps = 1e-6;
        Tensor up = x.clone(); up[0][1] += eps;
        Tensor dn = x.clone(); dn[0][1] -= eps;
        Tensor gx = g.clone();
        Tensor lpp = log_softmax(up);
        // d/dx of sum(g * log_softmax(x)) is exactly what log_softmax_backward gives.
        auto scalar = [&](const Tensor& xx) {
            Tensor l = log_softmax(xx);
            double s = 0.0;
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 3; ++j) s += gx[i][j] * l[i][j];
            return s;
        };
        double num = (scalar(up) - scalar(dn)) / (2 * eps);
        ASSERT_NEAR(num, dx[0][1], 1e-6);
        (void)lpp;
    });
}

// =================================================================
// Test 2: NLL — perfect vs wrong, and the uniform closed form
// =================================================================
static void test_nll_perfect_and_wrong() {
    run("(2a) NLL perfect vs uniform", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor lp(2, 2);
        lp[0][0] = -1e-9; lp[0][1] = -20.0;   // target 0, near-certain
        lp[1][0] = -20.0; lp[1][1] = -1e-9;   // target 1, near-certain
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        Tensor perfect = loss.forward(lp, t);
        ASSERT(perfect[0][0] >= 0.0);
        ASSERT(perfect[0][0] < 1e-6);

        // Uniform log-probs (-ln2) for K=2 give loss ln2 exactly.
        Tensor uni(1, 2); uni[0][0] = -std::log(2.0); uni[0][1] = -std::log(2.0);
        Tensor t1(1, 1); t1[0][0] = 0.0;
        NLLLoss l2(Reduction::Mean, -100);
        ASSERT_NEAR(l2.forward(uni, t1)[0][0], std::log(2.0), 1e-12);
    });
}

// D1 — mean divides by (B - n_ignored), NOT by B. THE headline test.
static void test_ignore_index_mean_divisor() {
    run("(2b) D1: ignore_index mean divisor", [] {
        NLLLoss loss(Reduction::Mean, /*ignore_index=*/1);
        Tensor lp(2, 2);
        lp[0][0] = -1.0; lp[0][1] = -std::log(3.0);
        lp[1][0] = 0.0;  lp[1][1] = -1.0;      // ignored row
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        Tensor out = loss.forward(lp, t);
        // Only row 0 contributes and its loss is 1.0. Divisor is 2-1 = 1.
        ASSERT_NEAR(loss.total_weight(), 1.0, 0.0);
        ASSERT_NEAR(out[0][0], 1.0, 1e-12);
    });
}

// D3 — with class weights, mean divides by the WEIGHT SUM.
static void test_weighted_mean_divisor() {
    run("(2c) D3: class-weight mean divisor", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor w(1, 3); w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = 3.0;
        loss.set_weight(w);
        Tensor lp(2, 3);
        lp[0][0] = -1.0; lp[0][1] = -5.0; lp[0][2] = -5.0;  // t=0, w=1 -> 1.0
        lp[1][0] = -5.0; lp[1][1] = -2.0; lp[1][2] = -5.0;  // t=1, w=2 -> 4.0
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        Tensor out = loss.forward(lp, t);
        ASSERT_NEAR(loss.total_weight(), 3.0, 0.0);      // 1 + 2
        ASSERT_NEAR(out[0][0], 5.0 / 3.0, 1e-12);
    });
}

// D2 — an ignored row's gradient is EXACTLY zero, not merely small.
static void test_ignored_row_gradient_is_zero() {
    run("(2d) D2: ignored row grad exactly zero", [] {
        NLLLoss loss(Reduction::Mean, 1);
        Tensor lp(2, 2);
        lp[0][0] = -0.5; lp[0][1] = -2.0;
        lp[1][0] = -0.7; lp[1][1] = -0.3;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;   // row 1 ignored
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = loss.backward(lp, t, go);
        ASSERT_NEAR(g[0][0], -1.0, 1e-12);
        ASSERT(g[0][1] == 0.0);
        ASSERT(g[1][0] == 0.0 && g[1][1] == 0.0);   // EXACTLY zero
    });
}

// Reduction::Sum must NOT divide; Reduction::None returns per-row.
static void test_sum_and_none() {
    run("(2e) sum and none reductions", [] {
        Tensor lp(2, 2);
        lp[0][0] = -1.0; lp[0][1] = -2.0;
        lp[1][0] = -3.0; lp[1][1] = -4.0;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        NLLLoss sum(Reduction::Sum, -100);
        ASSERT_NEAR(sum.forward(lp, t)[0][0], 1.0 + 4.0, 1e-12);
        NLLLoss none(Reduction::None, -100);
        Tensor per = none.forward(lp, t);
        ASSERT(per.rows == 2 && per.cols == 1);
        ASSERT_NEAR(per[0][0], 1.0, 1e-12);
        ASSERT_NEAR(per[1][0], 4.0, 1e-12);
    });
}

// Reduction::None backward takes a PER-ROW grad_output (LossNLL.cpp:378).
static void test_none_backward_per_row() {
    run("(2f) none backward per-row grad_output", [] {
        Tensor lp(2, 2);
        lp[0][0] = -1.0; lp[0][1] = -2.0;
        lp[1][0] = -3.0; lp[1][1] = -4.0;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        NLLLoss none(Reduction::None, -100);
        Tensor go(2, 1); go[0][0] = 2.0; go[1][0] = 5.0;
        Tensor g = none.backward(lp, t, go);
        ASSERT_NEAR(g[0][0], -2.0, 1e-12);
        ASSERT_NEAR(g[1][1], -5.0, 1e-12);
        ASSERT(g[0][1] == 0.0 && g[1][0] == 0.0);
    });
}

// Out-of-range target must throw, not wrap or read out of bounds.
static void test_out_of_range_throws() {
    run("(2g) out-of-range target throws", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor lp(1, 2); lp[0][0] = -0.5; lp[0][1] = -0.5;
        Tensor t(1, 1); t[0][0] = 7.0;
        bool threw = false;
        try { loss.forward(lp, t); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });
}

// =================================================================
// Test 3: CrossEntropyLoss
// =================================================================
// CE over logits == NLL(log_softmax(logits)). The equivalence IS the spec.
static void test_ce_equals_nll_of_log_softmax() {
    run("(3a) CE == NLL(log_softmax(logits))", [] {
        Tensor logits(3, 4);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                logits[i][j] = std::sin(3.0 * i + 1.7 * j) * 2.0;
        Tensor t(3, 1); t[0][0] = 0.0; t[1][0] = 3.0; t[2][0] = 2.0;
        NLLLoss nll(Reduction::Mean, -100);
        Tensor expected = nll.forward(log_softmax(logits), t);
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor got = ce.forward(logits, t);
        ASSERT_NEAR(got[0][0], expected[0][0], 1e-12);
    });
}

// Gradient: dCE/dlogits = softmax - onehot, over the batch. Known closed form.
static void test_ce_gradient_closed_form() {
    run("(3b) CE gradient == (softmax - onehot)/B", [] {
        Tensor logits(2, 3);
        logits[0][0] = 0.5; logits[0][1] = -1.0; logits[0][2] = 2.0;
        logits[1][0] = 0.0; logits[1][1] = 0.25; logits[1][2] = -0.5;
        Tensor t(2, 1); t[0][0] = 1.0; t[1][0] = 2.0;
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = ce.backward(logits, t, go);
        Tensor lp = log_softmax(logits);
        for (size_t i = 0; i < 2; ++i) {
            double rowsum = 0.0;
            for (size_t j = 0; j < 3; ++j) rowsum += g[i][j];
            ASSERT_NEAR(rowsum, 0.0, 1e-12);   // softmax-1 sums to 0 per row
            for (size_t j = 0; j < 3; ++j) {
                double onehot = (j == static_cast<size_t>(t[i][0])) ? 1.0 : 0.0;
                ASSERT_NEAR(g[i][j], (std::exp(lp[i][j]) - onehot) / 2.0, 1e-12);
            }
        }
    });
}

// D4: CE takes RAW LOGITS. Feeding probabilities where logits belong gives a
// DIFFERENT (wrong) number, because softmax([0.2,0.5,0.3]) is not [0.2,0.5,0.3]
// — the answer is logsumexp([0.2,0.5,0.3]) - 0.2 = 1.2398, not -ln(0.2) = 1.6094.
// Feeding the true LOGITS log([0.2,0.5,0.3]) recovers -ln(0.2) exactly.
// That gap between the two is the assertion; it is the whole point of D4.
static void test_ce_expects_raw_logits() {
    run("(3c) D4: CE expects raw logits", [] {
        Tensor probs(1, 3); probs[0][0] = 0.2; probs[0][1] = 0.5; probs[0][2] = 0.3;
        Tensor t(1, 1); t[0][0] = 0.0;
        CrossEntropyLoss ce(Reduction::Mean, -100);
        double as_probs = ce.forward(probs, t)[0][0];
        ASSERT_NEAR(as_probs, 1.2398310608444603, 1e-9);

        Tensor logits(1, 3);
        logits[0][0] = std::log(0.2); logits[0][1] = std::log(0.5);
        logits[0][2] = std::log(0.3);
        double as_logits = ce.forward(logits, t)[0][0];
        ASSERT_NEAR(as_logits, -std::log(0.2), 1e-12);
        // The two MUST disagree, or the fixture cannot tell the modes apart.
        ASSERT(std::abs(as_probs - as_logits) > 0.3);
    });
}

// CE with ignore_index must have an EXACTLY zero gradient row.
static void test_ce_ignore_index() {
    run("(3d) CE ignore_index", [] {
        Tensor logits(2, 3);
        logits[0][0] = 0.3; logits[0][1] = -0.8; logits[0][2] = 1.1;
        logits[1][0] = -0.2; logits[1][1] = 0.9; logits[1][2] = 0.4;
        Tensor t(2, 1); t[0][0] = 2.0; t[1][0] = 1;   // row 1 ignored
        CrossEntropyLoss ce(Reduction::Mean, 1);
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = ce.backward(logits, t, go);
        ASSERT(g[1][0] == 0.0 && g[1][1] == 0.0 && g[1][2] == 0.0);
        double rowsum = 0.0;
        for (size_t j = 0; j < 3; ++j) rowsum += g[0][j];
        ASSERT_NEAR(rowsum, 0.0, 1e-12);
    });
}

// FD oracle: the gradient is smooth in the logits, so centered FD is valid here.
static void test_ce_input_gradient_fd() {
    run("(3e) CE input gradient vs finite differences", [] {
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor logits(2, 3);
        logits[0][0] = 0.3; logits[0][1] = -0.8; logits[0][2] = 1.1;
        logits[1][0] = -0.2; logits[1][1] = 0.9; logits[1][2] = 0.4;
        Tensor t(2, 1); t[0][0] = 2.0; t[1][0] = 0.0;
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = ce.backward(logits, t, go);
        const double eps = 1e-6;
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 3; ++j) {
            Tensor up = logits.clone(); up[i][j] += eps;
            Tensor dn = logits.clone(); dn[i][j] -= eps;
            double num = (ce.forward(up, t)[0][0] - ce.forward(dn, t)[0][0]) / (2 * eps);
            double rel = std::abs(num - g[i][j]) / std::max(1e-9, std::abs(num));
            ASSERT(rel < 1e-5);
        }
    });
}

// =================================================================
// Test 4: BCEWithLogitsLoss
// =================================================================
// Stable form: max(x,0) - x*t + log1p(exp(-|x|))  (Loss.cpp:349-364)
static void test_bce_hand_derived() {
    run("(4a) BCE hand-derived", [] {
        BCEWithLogitsLoss loss(Reduction::None);
        Tensor z(2, 1); z[0][0] = 0.0; z[1][0] = 2.0;
        Tensor t(2, 1); t[0][0] = 1.0; t[1][0] = 0.0;
        Tensor per = loss.forward(z, t);
        ASSERT(per.rows == 2 && per.cols == 1);
        ASSERT_NEAR(per[0][0], std::log(2.0), 1e-12);                    // -log s(0)
        ASSERT_NEAR(per[1][0], 2.0 + std::log1p(std::exp(-2.0)), 1e-12); // -log s(-2)
    });
}

// D7: mean divides by numel = rows*cols, NOT by rows. Differs from mean over
// batch the moment cols > 1.
static void test_bce_mean_divides_by_numel() {
    run("(4b) D7: BCE mean divides by numel not rows", [] {
        Tensor z(2, 3); z.fill(0.0);
        Tensor t(2, 3); t.fill(1.0);
        BCEWithLogitsLoss sum(Reduction::Sum);
        double total = sum.forward(z, t)[0][0];
        BCEWithLogitsLoss mean(Reduction::Mean);
        ASSERT_NEAR(mean.forward(z, t)[0][0], total / 6.0, 1e-12);
        // And it must NOT be total/2 (mean over rows), which is the trap.
        ASSERT(std::abs(mean.forward(z, t)[0][0] - total / 2.0) > 0.3);
    });
}

// Extreme logits must not overflow: |z| = 1e4 gives a finite loss.
static void test_bce_extreme_logits() {
    run("(4c) BCE stable at |z| = 1e4", [] {
        BCEWithLogitsLoss loss(Reduction::Mean);
        Tensor z(1, 1); z[0][0] = 1e4;
        Tensor t(1, 1); t[0][0] = 0.0;
        double v = loss.forward(z, t)[0][0];
        ASSERT(std::isfinite(v));
        ASSERT_NEAR(v, 1e4, 1e-3);           // loss ~ max(x,0) - x*t = 1e4
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = loss.backward(z, t, go);
        ASSERT(std::isfinite(g[0][0]));
    });
}

// FD oracle on the logit gradient.
static void test_bce_gradient_fd() {
    run("(4d) BCE gradient vs finite differences", [] {
        BCEWithLogitsLoss loss(Reduction::Mean);
        Tensor z(2, 2);
        z[0][0] = 0.4; z[0][1] = -1.2; z[1][0] = 0.7; z[1][1] = 2.1;
        Tensor t(2, 2);
        t[0][0] = 1.0; t[0][1] = 0.0; t[1][0] = 0.0; t[1][1] = 1.0;
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = loss.backward(z, t, go);
        const double eps = 1e-6;
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) {
            Tensor up = z.clone(); up[i][j] += eps;
            Tensor dn = z.clone(); dn[i][j] -= eps;
            double num = (loss.forward(up, t)[0][0] - loss.forward(dn, t)[0][0]) / (2 * eps);
            ASSERT(std::abs(num - g[i][j]) / std::max(1e-9, std::abs(num)) < 1e-5);
        }
    });
}

// pos_weight reweights the POSITIVE term's log_sigmoid part
// (Loss.cpp:351-355): L = (1-t)*z - ((pos_weight-1)*t + 1)*log_sigmoid(z).
// Uses a (1,1) input so Reduction::None's per-row value is unambiguous.
static void test_bce_pos_weight() {
    run("(4e) BCE pos_weight", [] {
        BCEWithLogitsLoss plain(Reduction::None);
        BCEWithLogitsLoss loss(Reduction::None);
        Tensor pw(1, 1); pw[0][0] = 2.0;
        loss.set_pos_weight(pw);
        Tensor z(1, 1); z[0][0] = 0.5;
        Tensor tp(1, 1); tp[0][0] = 1.0;
        Tensor tn(1, 1); tn[0][0] = 0.0;

        const double ls = std::log1p(std::exp(-0.5));   // -log_sigmoid(0.5)
        // t=1 -> logw = 2: L = (1-1)*z - 2*log_sigmoid(z) = -2*log_sigmoid(z),
        // and log_sigmoid(0.5) is NEGATIVE, so the loss is +2*ls.
        ASSERT_NEAR(loss.forward(z, tp)[0][0], 2.0 * ls, 1e-12);
        // t=0 -> logw = 1: L = z - log_sigmoid(z) = 0.5 + ls, unchanged by pw.
        ASSERT_NEAR(loss.forward(z, tn)[0][0], plain.forward(z, tn)[0][0], 1e-12);
        ASSERT_NEAR(loss.forward(z, tn)[0][0], 0.5 + ls, 1e-12);
        // pos_weight must actually change the positive case
        ASSERT(std::abs(loss.forward(z, tp)[0][0] -
                        plain.forward(z, tp)[0][0]) > 0.1);
    });
}

// An elementwise `weight` multiplies the loss after the pointwise form.
static void test_bce_elementwise_weight() {
    run("(4f) BCE elementwise weight", [] {
        BCEWithLogitsLoss plain(Reduction::None);
        BCEWithLogitsLoss wt(Reduction::None);
        Tensor w(1, 2); w[0][0] = 2.0; w[0][1] = 3.0;
        wt.set_weight(w);
        Tensor z(1, 2); z[0][0] = 0.5; z[0][1] = -0.5;
        Tensor t(1, 2); t[0][0] = 1.0; t[0][1] = 0.0;
        // Reduction::None reduces over the CLASS axis, so the result is (rows,1)
        // and column 1 does not exist. Compare against a per-element reference
        // recomputed here instead of indexing into the reduced output.
        const double ls_pos = -std::log1p(std::exp(-0.5));    // log_sigmoid(0.5)
        const double ls_neg = -std::log1p(std::exp(0.5));     // log_sigmoid(-0.5)
        const double e00 = (1.0 - 1.0) * 0.5 - ls_pos;       // t=1, z=0.5
        const double e01 = (1.0 - 0.0) * -0.5 - ls_neg;      // t=0, z=-0.5
        Tensor b = wt.forward(z, t);
        ASSERT_NEAR(b[0][0], 2.0 * e00 + 3.0 * e01, 1e-12);
        ASSERT(std::abs(b[0][0] - plain.forward(z, t)[0][0]) > 1e-6);
    });
}

// BCE on a single logit z1 with target 0 is log(1 + e^z1), which is exactly
// CE([0, z1], target 0) = logsumexp([0,z1]) - 0. The K=2 limit, in closed form.
static void test_bce_is_k2_cross_entropy() {
    run("(4g) BCE is the K=2 limit of CE", [] {
        for (double z1 : {-2.0, 0.0, 1.5}) {
            Tensor pair(1, 2); pair[0][0] = 0.0; pair[0][1] = z1;
            Tensor tp(1, 1); tp[0][0] = 0.0;
            CrossEntropyLoss ce(Reduction::Mean, -100);
            double ce_v = ce.forward(pair, tp)[0][0];
            ASSERT_NEAR(ce_v, std::log1p(std::exp(z1)), 1e-12);

            Tensor single(1, 1); single[0][0] = z1;
            Tensor ts(1, 1); ts[0][0] = 0.0;
            BCEWithLogitsLoss bce(Reduction::Mean);
            double bce_v = bce.forward(single, ts)[0][0];
            ASSERT_NEAR(ce_v, bce_v, 1e-12);
        }
    });
}

// =================================================================
// Test 5: end-to-end — the loss must actually be trainable through a Dense
// stack (proves the gradient composes with Layer::backward's contract).
// =================================================================
static void test_end_to_end_training() {
    run("(5) end-to-end training reduces CE loss", [] {
        // 2-class linear model on 4 fixed samples.
        Dense d(2, 2);
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor X(4, 2);
        X[0][0] = 1.0; X[0][1] = 0.0;
        X[1][0] = 0.0; X[1][1] = 1.0;
        X[2][0] = 2.0; X[2][1] = 0.5;
        X[3][0] = 0.5; X[3][1] = 2.0;
        Tensor t(4, 1);
        t[0][0] = 0.0; t[1][0] = 1.0; t[2][0] = 0.0; t[3][0] = 1.0;

        double first = ce.forward(d.forward(X), t)[0][0];
        for (int step = 0; step < 200; ++step) {
            Tensor logits = d.forward(X);
            Tensor loss = ce.forward(logits, t);
            Tensor grad_out(4, 2);
            grad_out.fill(1.0);
            Tensor g = ce.backward(logits, t, grad_out);
            d.zero_grad();
            d.backward(g, 0.0);
            d.update_weights(0.5);
            (void)loss;
        }
        double last = ce.forward(d.forward(X), t)[0][0];
        ASSERT(first > 0.6);
        ASSERT(last < first * 0.2);
    });
}

// =================================================================
int main() {
    std::cout << "==============================================\n";
    std::cout << " Canonical Losses Test Suite\n";
    std::cout << "==============================================\n";

    test_log_softmax_hand();
    test_log_softmax_stability();
    test_log_softmax_backward();
    test_nll_perfect_and_wrong();
    test_ignore_index_mean_divisor();
    test_weighted_mean_divisor();
    test_ignored_row_gradient_is_zero();
    test_sum_and_none();
    test_none_backward_per_row();
    test_out_of_range_throws();
    test_ce_equals_nll_of_log_softmax();
    test_ce_gradient_closed_form();
    test_ce_expects_raw_logits();
    test_ce_ignore_index();
    test_ce_input_gradient_fd();
    test_bce_hand_derived();
    test_bce_mean_divides_by_numel();
    test_bce_extreme_logits();
    test_bce_gradient_fd();
    test_bce_pos_weight();
    test_bce_elementwise_weight();
    test_bce_is_k2_cross_entropy();
    test_end_to_end_training();

    std::cout << "\n==============================================\n";
    std::cout << "=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    std::cout << "==============================================\n";
    return g_fail == 0 ? 0 : 1;
}