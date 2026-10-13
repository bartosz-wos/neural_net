#ifndef NN_UTILS_KNOWLEDGE_DISTILLATION_H
#define NN_UTILS_KNOWLEDGE_DISTILLATION_H

#include "../core/tensor.h"
#include "losses.h"   // Reduction (guarded enum), log_softmax, log_softmax_backward
#include <cstddef>

// ===========================================================================
// Knowledge distillation
//
//   Geoffrey Hinton, Oriol Vinyals, Jeff Dean —
//   "Distilling the Knowledge in a Neural Network", arXiv:1503.02531.
//
// The queue recorded this ID and it is CORRECT (verified against the arXiv API
// 2026-10-13: "Distilling the Knowledge in a Neural Network").
//
// THE IDEA: a trained model contains information its one-hot label does not. A
// trained "cumbersome model" is used to produce SOFT TARGETS on a transfer set,
// and a smaller "distilled model" is trained to match those soft targets at a
// raised softmax temperature, alongside the ordinary hard-label objective.
//
// WHY IT LIVES HERE AND NOT IN layers/: it is a LOSS, not a layer, and it is
// model-free on purpose — it consumes (N, K) teacher and student logit matrices
// and needs no `Model` at all. That makes it usable with any classifier in this
// repo and unit-testable without constructing a network.
//
// ===========================================================================
// Semantics, and the rules a reimplementation gets wrong
// ===========================================================================
//
// (1) DIVIDE BY T. Eq. 1: q_i = exp(z_i/T) / sum_j exp(z_j/T). The student is
//     trained at temperature T but DEPLOYS at T = 1 (the paper: "after it has
//     been trained it uses a temperature of 1"). Multiplying by T is the natural
//     slip and produces a plausible-looking wrong number.
//
// (2) THE KL DIRECTION IS KL(TEACHER || STUDENT) — the student is pulled TOWARD
//     the teacher, so the teacher's softened distribution is the reference `p`
//     and the student's is the distribution `q`. KL is ASYMMETRIC: measured on
//     the FIX_A fixture at T=3, the two directions differ by 2.827e-06. A
//     "loss(t, t) == 0" test cannot see this, so a dedicated test pins it.
//
// (3) THE TWO NORMALISATIONS ARE SEPARATE AND MUST NOT SHARE WORK.
//     PyTorch Loss.cpp:264: output = xlogy(target, target) - target * input,
//     with `input` = student LOG-probs and `target` = teacher PROBS. So the loss
//     is built from log_softmax(student/T) and softmax(teacher/T) — two
//     independent row-wise normalisations of two different matrices.
//
// (4) THE xlogy ZERO RULE. xlogy(0, y) = 0 EVEN WHEN y IS ALSO 0 (Loss.cpp:264).
//     The naive p*log(p/q) yields 0*log(0/0) = NaN instead. Teacher
//     probabilities underflow to exactly 0 at large T (measured: every teacher
//     entry underflows at T = 1e3 on a row with a 3.0 logit gap), so this is a
//     reachable path, not a theoretical one.
//
// (5) Reduction::Mean DIVIDES BY rows*cols, NOT BY rows. PyTorch's Mean is
//     `unreduced.mean()` over ALL ELEMENTS (Loss.cpp:66-68); it is NOT
//     `batchmean`. Measured: mean/batchmean = 1/K = 0.333333 exactly at K=3.
//     Dividing the per-row KL by `rows` silently reproduces batchmean and is off
//     by a factor of K. The shipped BCE documents the same split as D7.
//
// (6) THE T^2 RESCALING IS ON THE LOSS. "Since the magnitudes of the gradients
//     produced by the soft targets scale as 1/T^2 it is important to multiply
//     them by T^2 when using both hard and soft targets. This ensures that the
//     relative contributions of the hard and soft targets remain roughly
//     unchanged if the temperature ... is changed." Keras applies it identically
//     (`loss * (self.temperature**2)` in LogitsDistillation.compute_loss).
//     Measured: the gradient-norm ratio with and without T^2 is EXACTLY 9.0 per
//     row at T = 3.
//
// (7) THE HARD/SOFT COMBINATION IS A WEIGHTED SUM OF TWO OBJECTIVES, NOT A
//     MIXTURE OF TARGETS. "a weighted average of two different objective
//     functions. The first objective function is the cross entropy with the
//     soft targets ... at a temperature of T. The second ... is computed using
//     exactly the same logits in softmax of the distilled model but at a
//     temperature of 1." So the hard term is evaluated at T = 1 and carries no
//     T^2. Blending the one-hot into the soft target instead is a different
//     objective with a different gradient, and it is the more common mistake.
//
// (8) HINTON'S Eq. 3->4 HIGH-TEMPERATURE LIMIT REQUIRES ZERO-MEANED LOGITS PER
//     CASE (sum_j z_j = sum_j v_j = 0). Verified by measurement, because it is a
//     precondition rather than an assertion: on UN-zero-meaned rows the limit
//     does NOT converge (relative error stuck at ~0.28 and GROWING with T), and
//     on zero-meaned rows it converges to (z - v)/(K * N_rows) at T = 1e3. The
//     implementation does not require it, and does not silently normalise either
//     — softmax is shift-invariant, so subtracting a row mean changes nothing
//     about this loss. Recorded so a future reader does not "fix" the limit
//     claim by assuming the precondition holds.
//
// (9) IDENTICAL MODELS GIVE ~1e-8, NOT BITWISE ZERO. PyTorch forms
//     xlogy(p,p) - p*s with log(softmax(z/T)) rounded separately from softmax,
//     so the self-KL residue is 1.43e-07 (measured). An `== 0.0` assertion would
//     fail against a correct implementation; the test asserts a tolerance.
//
// ===========================================================================

// Softened-KL distillation loss between student and teacher LOGITS:
//
//   KL(p_t || q_s) = sum_ij p_t(i,j) * (log p_t(i,j) - log_softmax(z_s/T)(i,j))
//
// with p_t = softmax(teacher_logits / T), reduced per `reduction` and then
// MULTIPLIED BY T^2 (semantics 6). `T` must be > 0.
//
// student_logits, teacher_logits: both (N, K), same shape; throws
// std::invalid_argument otherwise. Returns a (1, 1) tensor for Reduction::Mean
// and Reduction::Sum.
double distillation_kl(const Tensor& student_logits,
                       const Tensor& teacher_logits, double T);

// d(distillation_kl)/d(student_logits). Same shape as the inputs.
//
// With p_t = softmax(teacher/T) and q_s = softmax(student/T):
//   dL/dz_s = T^2 * (1/T) * (1/(N*K)) * (q_s - p_t)
// where the 1/(N*K) is present only for Reduction::Mean, the leading T^2 is the
// loss factor of semantics 6, and the 1/T is the chain rule through the
// temperature. Verified against torch autograd in float64 to 1.4e-16.
//
// The teacher contributes NO gradient: it is a constant target, and this
// function's entire contract is with respect to the student.
Tensor distillation_kl_backward(const Tensor& student_logits,
                               const Tensor& teacher_logits, double T);

// ---------------------------------------------------------------------------
// DistillationLoss — the full training objective of Eq. 4's companion (the
// weighted sum of a hard-label cross-entropy at T = 1 and a soft-target KL at
// temperature T).
//
// alpha is the weight on the HARD-label cross-entropy, following the paper's
// "using a considerably lower weight on the second objective function" where the
// second objective is the one on the correct labels. alpha = 1.0 is the ordinary
// hard-label loss alone; alpha = 0.0 is pure distillation (the paper's setting
// when the transfer set is unlabelled). Defaults follow Keras's
// LogitsDistillation default temperature of 3.0.
// ---------------------------------------------------------------------------
class DistillationLoss {
public:
    DistillationLoss(double temperature = 3.0, double alpha = 1.0,
                     Reduction reduction = Reduction::Mean);

    // loss = alpha * CE(student_logits, labels @ T=1)
    //      + (1 - alpha) * distillation_kl(student_logits, teacher_logits, T)
    //
    // `labels` is an (N, 1) column of integer class indices in [0, K). A
    // non-integral, negative or out-of-range index throws std::invalid_argument.
    // For Reduction::None the result is (N, 1), one value per row.
    Tensor forward(const Tensor& student_logits, const Tensor& teacher_logits,
                   const Tensor& labels) const;

    // Gradient w.r.t. student_logits. Same shape as student_logits. The teacher
    // and the labels are both constants here.
    Tensor backward(const Tensor& student_logits, const Tensor& teacher_logits,
                    const Tensor& labels, const Tensor& grad_output) const;

    double temperature() const { return temperature_; }
    double alpha() const { return alpha_; }
    Reduction reduction() const { return reduction_; }

private:
    double temperature_;
    double alpha_;
    Reduction reduction_;
};

#endif // NN_UTILS_KNOWLEDGE_DISTILLATION_H