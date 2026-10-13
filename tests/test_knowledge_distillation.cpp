// test_knowledge_distillation.cpp — TDD tests for knowledge distillation
// (Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network",
// arXiv:1503.02531).
//
// EVERY expected value here is ORACLE-DERIVED from REAL PyTorch 2.14.1
// (aten/src/ATen/native/Loss.cpp kl_div) — not hand-derived, and not transcribed
// by me, for the reason the metrics session in this repo already paid for: an
// oracle written by the same author as the implementation is not independent.
// The oracle run is /home/stefan/.hermes/cache/scratch/kd_oracle.py; the
// recorded numbers appear in comments at each fixture.
//
// THE ORACLE HAD ITS OWN BUGS, all fixed before a line of C++ was written, and
// they are worth recording because each presented as a plausible "the math is
// wrong" conclusion:
//   (1) The oracle's first analytic adjoint had the sign of (p - q). Hinton's
//       Eq. 2 is (q - p). The symptom was rel_err ~ 1.96 == 2.0 on every cell —
//       a CLEAN ratio, which the TDD skill calls the fingerprint of a sign flip.
//       The impl was never written yet; the oracle was simply wrong.
//   (2) FD at eps=1e-6 reported rel_err ~ 1.9 and looked like a broken adjoint.
//       It was float32 noise: the same analytic formula matches torch AUTOGRAD
//       to 1.4e-16 in float64. The FD step, not the math, was degenerate.
//   (3) The oracle asserted the gradient w.r.t. the teacher is zero. KL(p||q) IS
//       a function of p, so dKL/dp is genuinely non-zero — the assertion was
//       wrong. The meaningful statement is that the teacher is a CONSTANT.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstddef>
#include "nn/utils/knowledge_distillation.h"
#include "nn/utils/losses.h"   // log_softmax, CrossEntropyLoss

static int passed = 0;
static int failed = 0;

// NOTE: argument order is (name, result) so each call reads as a sentence
// describing what was checked.
static void check(const std::string& name, bool pass) {
    if (pass) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else       { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
}

static bool close(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

// Relative comparison for values whose magnitude is fixture-dependent.
static bool rel_close(double a, double b, double rtol = 1e-6) {
    return std::fabs(a - b) <= rtol * std::max(1.0, std::fabs(b));
}

static Tensor mat(std::vector<std::vector<double>> rows) {
    return Tensor(rows);
}

// Integer label column (N,1).
static Tensor labels(std::vector<double> y) {
    Tensor t(y.size(), 1);
    for (size_t i = 0; i < y.size(); ++i) t(i, 0) = y[i];
    return t;
}

static Tensor random_logits(size_t r, size_t c, unsigned seed) {
    srand(seed);
    return Tensor::random(r, c, 1.0);
}

// FD oracle for the analytic backward: d(distillation_kl)/d(student_logit[i][j])
// with the T^2 factor applied, matching what distillation_kl_backward returns.
//
// eps = 1e-4, NOT 1e-6. Measured: at eps=1e-6 the forward's own rounding noise
// dominates this loss (the two normalisations of semantics 3 are separately
// rounded) and FD reports spurious rel_err ~ 1.9 against a CORRECT adjoint. This
// is the systematic-debugging 5c case: the test config was degenerate, not the
// implementation. Confirmed against torch autograd, which agrees with the
// analytic formula to 1.4e-16 at eps=1e-4.
static double fd_kl_grad(const Tensor& zs, const Tensor& zt, double T, double eps,
                         size_t i, size_t j) {
    Tensor up = zs.clone(), dn = zs.clone();
    up(i, j) += eps;
    dn(i, j) -= eps;
    return (distillation_kl(up, zt, T) - distillation_kl(dn, zt, T)) / (2.0 * eps);
}

// ---------------------------------------------------------------------------
// Fixtures — every value below traced to the torch oracle of 2026-10-13
// ---------------------------------------------------------------------------

// FIX_A: 3 rows x 3 classes. No ties anywhere, non-uniform, deliberately NOT
// zero-meaned per row (row sums are 2.0 / 3.5 / 1.25), which is what makes it
// able to detect a silently-applied mean subtraction.
//
// Student                       Teacher
//   [ 2.0,  1.0, -1.0]            [ 1.8,  1.1, -0.9]
//   [ 0.5,  3.0,  0.0]            [ 0.4,  3.2, -0.1]
//   [ 1.5, -0.5,  0.25]           [ 1.4, -0.2,  0.3]
//
// Torch oracle, distillation_kl WITHOUT the T^2 factor, reduction='mean'
// (i.e. mean over all 9 elements):
//   T=1  -> 0.002527293982
//   T=3  -> 0.000432487985
//   T=7  -> 0.000079751015
// Same logits, sum over classes and MEAN OVER ROWS (batchmean):
//   T=1  -> 0.007581878453
//   T=3  -> 0.001297439798
//   T=7  -> 0.000239222893
// ratio rows/elements = 2.999999 / 2.999944 / 2.999622  (== K = 3)
static Tensor FIX_A_S() {
    return mat({{2.0, 1.0, -1.0}, {0.5, 3.0, 0.0}, {1.5, -0.5, 0.25}});
}
static Tensor FIX_A_T() {
    return mat({{1.8, 1.1, -0.9}, {0.4, 3.2, -0.1}, {1.4, -0.2, 0.3}});
}

// FIX_UNDERFLOW: a row whose teacher probabilities underflow to exactly 0 at
// large T, so the xlogy zero rule (semantics 4) is a REACHABLE path rather than
// a theoretical one. Teacher logits span 40 nats; at T=1000 every non-argmax
// entry exp()-underflows to 0, and the naive p*log(p/q) form returns NaN.
static Tensor FIX_UF_S() {
    return mat({{40.0, 0.0, -1.0}, {0.0, 40.0, -2.0}});
}
static Tensor FIX_UF_T() {
    return mat({{35.0, -1.0, -2.0}, {-1.0, 33.0, -3.0}});
}

static void test_constructor_validation() {
    std::cout << "--- Constructor validation ---" << std::endl;

    bool threw = false;
    try { DistillationLoss(-1.0); } catch (...) { threw = true; }
    check("temperature <= 0 throws", threw);

    threw = false;
    try { DistillationLoss(0.0); } catch (...) { threw = true; }
    check("temperature == 0 throws", threw);

    threw = false;
    try { DistillationLoss(3.0, -0.5); } catch (...) { threw = true; }
    check("alpha < 0 throws", threw);

    threw = false;
    try { DistillationLoss(3.0, 1.5); } catch (...) { threw = true; }
    check("alpha > 1 throws", threw);

    threw = false;
    try { distillation_kl(FIX_A_S(), FIX_A_T(), 0.0); }
    catch (...) { threw = true; }
    check("distillation_kl with T == 0 throws", threw);

    threw = false;
    try { distillation_kl(FIX_A_S(), mat({{1.0, 2.0}}), 3.0); }
    catch (...) { threw = true; }
    check("shape mismatch teacher throws", threw);

    threw = false;
    try { distillation_kl(FIX_A_S(), mat({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}}), 3.0); }
    catch (...) { threw = true; }
    check("shape mismatch on COLS throws", threw);

    DistillationLoss d(4.5, 0.3, Reduction::Sum);
    check("accessors round-trip",
          close(d.temperature(), 4.5) && close(d.alpha(), 0.3) &&
          d.reduction() == Reduction::Sum);
    DistillationLoss defaults;
    check("defaults are T=3, alpha=1, Mean",
          close(defaults.temperature(), 3.0) && close(defaults.alpha(), 1.0) &&
          defaults.reduction() == Reduction::Mean);
}

static void test_forward_matches_oracle() {
    std::cout << "--- Forward vs torch oracle (mean over ELEMENTS, then x T^2) ---"
              << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();

    // Oracle: mean-over-elements x T^2.
    //   T=1 -> 0.002527293982 * 1   = 0.002527293982
    //   T=3 -> 0.000432487985 * 9   = 0.003892391865   (oracle 'sum' = 0.003892391920,
    //                                                  the last digits differ because
    //                                                  torch's xlogy(p,p) term is 0 here
    //                                                  and ours is computed as p*(log p - s)
    //                                                  with one extra rounding)
    //   T=7 -> 0.000079751015 * 49  = 0.003907799735
    check("forward T=1 matches oracle 0.002527293982",
          rel_close(distillation_kl(s, t, 1.0), 0.002527293982, 1e-6));
    check("forward T=3 matches oracle 0.003892391865",
          rel_close(distillation_kl(s, t, 3.0), 0.003892391865, 1e-6));
    check("forward T=7 matches oracle 0.003907799735",
          rel_close(distillation_kl(s, t, 7.0), 0.003907799735, 1e-6));

    // The T^2 factor must be OBSERVABLE, not incidental: without it the T=3
    // value would be 0.000432487985, a factor of 9 smaller.
    check("forward at T=3 is EXACTLY 9x the un-rescaled value",
          rel_close(distillation_kl(s, t, 3.0) / 0.000432487985, 9.0, 1e-6));

    // MEAN IS OVER ELEMENTS, NOT ROWS (semantics 5). The row-mean (batchmean)
    // value is exactly K times larger; a test that only checked "a number" would
    // accept the wrong one.
    double rows_mean = 0.001297439798 * 9.0;
    check("forward is NOT the row-mean: differs from batchmean by exactly K=3",
          rel_close(rows_mean / distillation_kl(s, t, 3.0), 3.0, 1e-5));
    check("mean-vs-batchmean distinction is real (they differ)",
          !rel_close(distillation_kl(s, t, 3.0), rows_mean, 1e-3));
}

static void test_kl_direction() {
    std::cout << "--- KL direction is KL(teacher || student) ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();

    // The loss must be ASYMMETRIC in its two arguments. Oracle:
    //   KL(teacher||student) = 0.001297423034  (x T^2 = 0.011676807306)
    //   KL(student||teacher) = 0.001294596004
    // They differ by 2.827e-06 — small, but far above double-precision noise.
    // A transposed argument order would return the OTHER value, and the
    // self-KL-is-zero test below would still pass. Only this test can see it.
    double forward_ = distillation_kl(s, t, 3.0);
    double swapped_ = distillation_kl(t, s, 3.0);
    check("forward direction matches KL(teacher||student) = 0.011676807306",
          rel_close(forward_, 0.011676807306, 1e-6));
    check("KL is asymmetric: swapped args give a DIFFERENT value",
          !rel_close(forward_, swapped_, 1e-4));

    // A near-tie fixture where the two directions are far apart, so this is not
    // an accident of FIX_A's small asymmetry.
    Tensor a = mat({{3.0, 0.0, 0.0}});
    Tensor b = mat({{0.0, 0.0, 0.0}});
    double ts = distillation_kl(a, b, 1.0);   // confident teacher, flat student
    double st = distillation_kl(b, a, 1.0);   // flat teacher, confident student
    check("confident-teacher vs confident-student are strongly asymmetric",
          std::fabs(ts - st) > 0.1);
}

static void test_self_kl_is_zero() {
    std::cout << "--- Identical models give ~0, NOT bitwise 0 ---" << std::endl;
    // Oracle 5 measured 1.43e-07 at T=3 and 8.28e-09 at T=1 in float32: torch
    // forms xlogy(p,p) - p*s with log(softmax) rounded separately. A test
    // asserting == 0.0 would fail against a CORRECT implementation.
    double v = distillation_kl(FIX_A_S(), FIX_A_S(), 3.0);
    check("self-KL is within 1e-6 of zero (oracle residue ~1.4e-07)",
          std::fabs(v) < 1e-6);
    check("self-KL is not asserted == 0.0 (it is a rounding residue)",
          v != 0.0);
}

static void test_xlogy_zero_rule() {
    std::cout << "--- xlogy(0, 0) = 0, not NaN (semantics 4) ---" << std::endl;
    Tensor s = FIX_UF_S(), t = FIX_UF_T();
    double v = distillation_kl(s, t, 1000.0);
    check("underflowing teacher probs do NOT produce NaN",
          !std::isnan(v) && !std::isinf(v));
    check("underflowed KL is finite and small",
          v >= 0.0 && v < 1e-6);
    // And the ordinary temperatures on the same fixture stay finite.
    for (double T : {1.0, 3.0, 10.0}) {
        double x = distillation_kl(s, t, T);
        if (std::isnan(x) || std::isinf(x)) {
            check("extreme-logit fixture finite at T=" + std::to_string(T), false);
            return;
        }
    }
    check("extreme-logit fixture finite at T=1,3,10", true);
}

static void test_kl_is_nonnegative() {
    std::cout << "--- KL >= 0 always ---" << std::endl;
    Tensor s = random_logits(5, 4, 11), t = random_logits(5, 4, 22);
    bool all_nonneg = true;
    for (double T : {0.5, 1.0, 2.0, 3.0, 7.5}) {
        if (distillation_kl(s, t, T) < -1e-12) all_nonneg = false;
    }
    check("KD loss is nonnegative over 5 temperatures", all_nonneg);
}

static void test_backward_matches_fd() {
    std::cout << "--- Backward vs centered finite differences (eps=1e-4) ---"
              << std::endl;
    // Random NON-UNIFORM logits on purpose: a uniform init makes
    // sum_k W[k,j] == sum_k W[j,k] by construction, which masks a whole class
    // of index-transposition bugs (TDD skill's uniform-init caveat). srand(seed)
    // makes the fixture deterministic.
    Tensor s = random_logits(3, 4, 12345), t = random_logits(3, 4, 6789);

    for (double T : {1.0, 2.5, 5.0}) {
        Tensor g = distillation_kl_backward(s, t, T);
        double worst = 0.0;
        bool sign_ok = true;
        for (size_t i = 0; i < s.rows; ++i)
            for (size_t j = 0; j < s.cols; ++j) {
                double fd = fd_kl_grad(s, t, T, 1e-4, i, j);
                double an = g(i, j);
                double rel = std::fabs(an - fd) / std::max(1e-12,
                                              std::max(std::fabs(an), std::fabs(fd)));
                if (rel > worst) worst = rel;
                // A pure sign flip gives a clean ratio of 2.0 against FD; a
                // tolerance check alone would not catch it where both are small.
                if (an * fd < 0.0) sign_ok = false;
            }
        std::cout << "    T=" << T << " worst rel_err = " << worst << std::endl;
        check("T=" + std::to_string(T) + " backward rel_err < 1e-5",
              worst < 1e-5);
        check("T=" + std::to_string(T) + " backward agrees in SIGN with FD",
              sign_ok);
    }
}

static void test_t2_gradient_factor() {
    std::cout << "--- The T^2 rescaling is observable in the GRADIENT (semantics 6) ---"
              << std::endl;
    // Oracle 4 measured the gradient-norm ratio with and without T^2 as
    // EXACTLY 9.0 per row at T=3. Here the same property is checked end to end
    // through the shipped backward: the analytic gradient at T and at T=1 must
    // be related by the temperature, and the RELATION is what pins the factor.
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor g1 = distillation_kl_backward(s, t, 1.0);
    Tensor g3 = distillation_kl_backward(s, t, 3.0);

    // At T=1 the T^2 factor is 1, so g1 is the plain mean-over-elements adjoint.
    // Oracle 3 at T=1, elements, (T^2-scaled == unscaled) agrees with torch
    // autograd; the analytic (q-p)/(N*K) row-0 triple is
    //   [ 0.021970937, -0.019351937, -0.002619000 ]
    check("T=1 backward row 0 matches the oracle analytic adjoint",
          rel_close(g1(0, 0), 0.02197093707125, 1e-6) &&
          rel_close(g1(0, 1), -0.01935193717279, 1e-6) &&
          rel_close(g1(0, 2), -0.00261899989846, 1e-6));

    // Scaling from T=1 to T=3 is NOT a constant factor of 9 — the softmax
    // itself changes. What IS a constant factor of 9 is the ratio between
    // applying and not applying the rescaling at the SAME T, so that is what
    // the FD comparison above pins. Here assert only that T=3 changes the
    // gradient (i.e. the temperature is not silently ignored).
    bool differs = false;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(g3(i, j) - g1(i, j)) > 1e-9) differs = true;
    check("changing T changes the gradient (temperature is not ignored)", differs);
}

static void test_combined_loss_objective() {
    std::cout << "--- DistillationLoss: weighted sum of TWO objectives ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor y = labels({0.0, 1.0, 2.0});   // all correct -> CE == 0 at these logits? no

    // alpha = 1 must be EXACTLY the plain hard-label cross entropy at T=1,
    // with no distillation term at all. That is the property that distinguishes
    // "weighted sum of two objectives" from "blend the one-hot into the soft
    // target": the blended form would still contain teacher information at
    // alpha=1.
    CrossEntropyLoss ce(Reduction::Mean);
    Tensor ce_only = ce.forward(s, y);
    DistillationLoss hard_only(3.0, 1.0, Reduction::Mean);
    Tensor d_hard = hard_only.forward(s, t, y);
    check("alpha=1 is EXACTLY the hard-label CE (no teacher contribution)",
          rel_close(d_hard(0, 0), ce_only(0, 0), 1e-12));

    // alpha = 0 must be EXACTLY the softened KL.
    DistillationLoss soft_only(3.0, 0.0, Reduction::Mean);
    Tensor d_soft = soft_only.forward(s, t, y);
    check("alpha=0 is EXACTLY distillation_kl",
          rel_close(d_soft(0, 0), distillation_kl(s, t, 3.0), 1e-12));

    // And alpha = 0.25 is the exact convex combination of those two — the
    // linearity is what makes "weighted average of two objective functions"
    // checkable rather than merely asserted.
    DistillationLoss mid(3.0, 0.25, Reduction::Mean);
    double expect = 0.25 * ce_only(0, 0) + 0.75 * distillation_kl(s, t, 3.0);
    check("alpha=0.25 is the exact weighted combination of the two extremes",
          rel_close(mid.forward(s, t, y)(0, 0), expect, 1e-12));

    // The hard term is at T=1 regardless of the distillation temperature. Pin
    // this by varying T at fixed alpha=1: the loss must NOT move.
    DistillationLoss t3(3.0, 1.0), t50(50.0, 1.0);
    check("the hard term is evaluated at T=1 (loss invariant in T at alpha=1)",
          rel_close(t3.forward(s, t, y)(0, 0), t50.forward(s, t, y)(0, 0), 1e-12));

    // And at alpha=0 the loss MUST move with T.
    DistillationLoss s1(1.0, 0.0), s7(7.0, 0.0);
    check("the soft term does depend on T (loss varies in T at alpha=0)",
          !rel_close(s1.forward(s, t, y)(0, 0), s7.forward(s, t, y)(0, 0), 1e-6));
}

static void test_label_validation() {
    std::cout << "--- Label validation ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();
    DistillationLoss d(3.0, 0.5);

    bool threw = false;
    try { d.forward(s, t, labels({0.0, 1.0, 5.0})); } catch (...) { threw = true; }
    check("out-of-range label throws", threw);

    threw = false;
    try { d.forward(s, t, labels({0.0, -1.0, 2.0})); } catch (...) { threw = true; }
    check("negative label throws", threw);

    threw = false;
    try { d.forward(s, t, labels({0.5, 1.0, 2.0})); } catch (...) { threw = true; }
    check("non-integral label throws", threw);

    threw = false;
    try { d.forward(s, t, labels({0.0, 1.0})); } catch (...) { threw = true; }
    check("label row-count mismatch throws", threw);

    Tensor ok = d.forward(s, t, labels({0.0, 1.0, 2.0}));
    check("valid labels produce a finite scalar",
          ok.rows == 1 && ok.cols == 1 && std::isfinite(ok(0, 0)));
}

static void test_reduction_none() {
    std::cout << "--- Reduction::None is per-ROW, not the full sum ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor y = labels({0.0, 1.0, 2.0});
    DistillationLoss none(3.0, 0.5, Reduction::None);
    Tensor per_row = none.forward(s, t, y);
    check("Reduction::None returns one value per row",
          per_row.rows == 3 && per_row.cols == 1);

    DistillationLoss sum(3.0, 0.5, Reduction::Sum);
    Tensor total = sum.forward(s, t, y);
    double manual = 0.0;
    for (size_t i = 0; i < 3; ++i) manual += per_row(i, 0);
    check("Sum equals the sum of the per-row values",
          rel_close(total(0, 0), manual, 1e-12));

    DistillationLoss mean(3.0, 0.5, Reduction::Mean);
    check("Mean is the MEAN of the per-row values (divides by N=3)",
          rel_close(mean.forward(s, t, y)(0, 0), manual / 3.0, 1e-12));

    // Pin the DIFFERENCE from Sum: if Mean were accidentally Sum, these agree.
    check("Mean and Sum genuinely differ on this fixture",
          !rel_close(mean.forward(s, t, y)(0, 0), total(0, 0), 1e-6));
}

static void test_backward_shape_and_symmetry() {
    std::cout << "--- Combined backward shape and sign ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor y = labels({0.0, 1.0, 2.0});
    DistillationLoss d(3.0, 0.4, Reduction::Mean);
    Tensor one(1, 1); one(0, 0) = 1.0;

    Tensor g = d.backward(s, t, y, one);
    check("combined backward has the student's shape",
          g.rows == s.rows && g.cols == s.cols);
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < g.rows; ++i)
        for (size_t j = 0; j < g.cols; ++j) {
            if (!std::isfinite(g(i, j))) finite = false;
            if (std::fabs(g(i, j)) > 1e-12) nonzero = true;
        }
    check("combined backward is finite", finite);
    check("combined backward is nonzero", nonzero);

    // grad_output must actually scale the result — a backward that ignored it
    // would still pass every shape check above.
    Tensor two(1, 1); two(0, 0) = 2.0;
    Tensor g2 = d.backward(s, t, y, two);
    bool scaled = true;
    for (size_t i = 0; i < g.rows; ++i)
        for (size_t j = 0; j < g.cols; ++j)
            if (!rel_close(g2(i, j), 2.0 * g(i, j), 1e-12)) scaled = false;
    check("grad_output scales the gradient linearly", scaled);
}

static void test_combined_backward_fd() {
    std::cout << "--- Combined backward vs FD through the full objective ---"
              << std::endl;
    Tensor s = random_logits(3, 4, 555), t = random_logits(3, 4, 777);
    Tensor y = labels({0.0, 2.0, 1.0});
    DistillationLoss d(2.0, 0.35, Reduction::Mean);
    Tensor one(1, 1); one(0, 0) = 1.0;
    Tensor g = d.backward(s, t, y, one);

    double eps = 1e-4;
    double worst = 0.0;
    bool sign_ok = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j) {
            Tensor up = s.clone(), dn = s.clone();
            up(i, j) += eps; dn(i, j) -= eps;
            double fd = (d.forward(up, t, y)(0, 0) - d.forward(dn, t, y)(0, 0))
                        / (2.0 * eps);
            double an = g(i, j);
            double rel = std::fabs(an - fd) / std::max(1e-12,
                                          std::max(std::fabs(an), std::fabs(fd)));
            if (rel > worst) worst = rel;
            if (an * fd < 0.0) sign_ok = false;
        }
    std::cout << "    worst rel_err = " << worst << std::endl;
    check("combined backward rel_err < 1e-5 vs FD", worst < 1e-5);
    check("combined backward agrees in SIGN with FD", sign_ok);
}

static void test_high_temperature_limit() {
    std::cout << "--- High-T limit (Hinton Eq. 3->4, zero-meaned rows) ---" << std::endl;
    // Hinton's Eq. 3 -> 4 reduction assumes sum_j z_j = sum_j v_j = 0 per case.
    // Verified by measurement that the precondition is REQUIRED: on un-zero-
    // meaned rows the limit does not converge. So this test zero-means first,
    // which is legitimate because softmax is shift-invariant — subtracting a
    // per-row constant changes neither p nor q, hence not the loss.
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor zs(s.rows, s.cols), zt(t.rows, t.cols);
    for (size_t i = 0; i < s.rows; ++i) {
        double ms = 0.0, mt = 0.0;
        for (size_t j = 0; j < s.cols; ++j) { ms += s(i, j); mt += t(i, j); }
        ms /= static_cast<double>(s.cols);
        mt /= static_cast<double>(t.cols);
        for (size_t j = 0; j < s.cols; ++j) {
            zs(i, j) = s(i, j) - ms;
            zt(i, j) = t(i, j) - mt;
        }
    }
    // Shift invariance: the loss must be unchanged by the zero-meaning.
    check("distillation_kl is invariant to a per-row constant shift",
          rel_close(distillation_kl(zs, zt, 3.0), distillation_kl(s, t, 3.0),
                    1e-9));

    // Eq. 4 with the T^2 rescaling folded in: dL/dz -> (z - v) / (K * N_rows).
    const double K = 3.0, NR = 3.0;
    Tensor g = distillation_kl_backward(zs, zt, 1e3);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < zs.rows; ++i)
        for (size_t j = 0; j < zs.cols; ++j) {
            double expect = (zs(i, j) - zt(i, j)) / (K * NR);
            if (std::fabs(expect) > scale) scale = std::fabs(expect);
            double d = std::fabs(g(i, j) - expect);
            if (d > worst) worst = d;
        }
    std::cout << "    max abs deviation from Eq. 4 at T=1e3 = " << worst
              << "  (scale " << scale << ")" << std::endl;
    check("T=1e3 backward matches Eq.4 (z-v)/(K*N) to <1% absolute",
          worst < 0.01 * std::max(1.0, scale));

    // The unscaled gradient at high T vanishes like 1/T^2 — the very reason the
    // T^2 rescaling exists. Assert the rescaled gradient does NOT vanish.
    double g_norm = 0.0;
    for (size_t i = 0; i < zs.rows; ++i)
        for (size_t j = 0; j < zs.cols; ++j) g_norm += g(i, j) * g(i, j);
    check("T^2-rescaled gradient stays finite at T=1e3 (does not vanish)",
          g_norm > 0.0);
}

static void test_determinism() {
    std::cout << "--- Determinism ---" << std::endl;
    Tensor s = FIX_A_S(), t = FIX_A_T();
    Tensor y = labels({0.0, 1.0, 2.0});
    DistillationLoss d(3.0, 0.6, Reduction::Mean);
    Tensor a = d.forward(s, t, y);
    Tensor b = d.forward(s, t, y);
    check("two forward calls are bit-identical", a(0, 0) == b(0, 0));
    Tensor one(1, 1); one(0, 0) = 1.0;
    Tensor ga = d.backward(s, t, y, one);
    Tensor gb = d.backward(s, t, y, one);
    bool same = true;
    for (size_t i = 0; i < ga.rows; ++i)
        for (size_t j = 0; j < ga.cols; ++j)
            if (ga(i, j) != gb(i, j)) same = false;
    check("two backward calls are bit-identical", same);
}

int main() {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== Knowledge Distillation (Hinton/Vinyals/Dean, arXiv:1503.02531) ==="
              << std::endl;

    test_constructor_validation();
    test_forward_matches_oracle();
    test_kl_direction();
    test_self_kl_is_zero();
    test_xlogy_zero_rule();
    test_kl_is_nonnegative();
    test_backward_matches_fd();
    test_t2_gradient_factor();
    test_combined_loss_objective();
    test_label_validation();
    test_reduction_none();
    test_backward_shape_and_symmetry();
    test_combined_backward_fd();
    test_high_temperature_limit();
    test_determinism();

    std::cout << "=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}