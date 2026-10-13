#ifndef NN_UTILS_TEMPERATURE_SCALING_H
#define NN_UTILS_TEMPERATURE_SCALING_H

#include "../core/tensor.h"
#include <cstddef>

// ===========================================================================
// Temperature scaling
//
//   Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger —
//   "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599.
//
// CITATION NOTE: EXPANSION_QUEUE.md recorded this paper as arXiv:1707.06552.
// That ID is WRONG — the arXiv API returns "Towards a scientific blockchain
// framework for reproducible data analysis" for it (verified 2026-10-13).
//
// The paper's claim (Sec. 1): "a single-parameter variant of Platt scaling —
// which we refer to as temperature scaling — is often the most effective
// method at obtaining calibrated probabilities." It is a POST-PROCESSING step:
// the network is trained normally, then one scalar is fitted on held-out data.
//
// This is the standard fix for a modern network's characteristic failure: it
// is systematically OVERCONFIDENT, so sharpening the output with a single T > 1
// recovers calibrated probabilities without touching a single weight.
//
// Why it lives here and not in metrics.{h,cpp}: metrics SCORES a fixed
// prediction set, whereas temperature scaling PRODUCES a new one. It is also
// not a loss — it is fitted, but it is fitted against the same NLL a training
// loop already uses, and it is applied at inference. That makes it a utility.
//
// It is model-free on purpose: it consumes (N, K) logits and (N, 1) integer
// labels. Nothing here needs a `Model`, which means it composes with ANY
// classifier in this repo, and — the reason it was deferred until now — it is
// directly measurable with the already-shipped expected_calibration_error /
// classwise_ece.
//
// ===========================================================================
// Semantics, and the rules a reimplementation gets wrong
// ===========================================================================
//
// (1) DIVIDE. The forward is softmax(z / T), NOT softmax(z * T) (Eq. 9:
//     q_i = max_k softmax(z_i / T)^(k)). The authors' own implementation makes
//     the parameterisation explicit: "we implement temperature scaling by
//     inserting a nn.MulConstant between the logits and the softmax, whose
//     parameter is 1/T". Note this repo's Hopfield layer stores an INVERSE
//     temperature `beta`; the two conventions must not be conflated. This file
//     exposes T itself, the paper's parameterisation.
//
// (2) T > 0 is a hard precondition and T = 1 is the identity. Sec. 4.2: "With
//     T = 1, we recover the original probability p_i." T <= 0 throws.
//
// (3) THE CLASS PREDICTION CANNOT CHANGE. Dividing by a positive scalar
//     preserves argmax exactly, so "temperature scaling does not affect the
//     model's accuracy" (Sec. 4.2, confirmed in Table S2). This is the
//     property that makes it safe to deploy: calibration improves, accuracy is
//     untouched. It is broken only by the inverted op at T < 1.
//
// (4) THE LIMITS ARE ONE-SIDED AND BOTH ARE DEGENERATE. T -> infinity gives
//     q_i -> 1/K (maximum uncertainty); T -> 0 gives a point mass on the argmax
//     (Sec. 4.2). The T -> 0 end is what makes the FIT need a bracket: on a
//     perfectly separable validation set the NLL minimiser runs all the way to
//     T = 0 with NLL -> 0 (measured: T* = 6.9e-4, NLL = 0.0). An unbounded
//     search would return a meaningless temperature.
//
// (5) THE OBJECTIVE IS NLL ON A HELD-OUT SET (Sec. 4.2: "T is optimized with
//     respect to NLL on the validation set"), reduced as the MEAN over rows.
//     This is deliberately NOT the shipped `log_loss`: that one is sklearn's
//     clip-based form, whose 1-column and 2-column forms of the same data
//     differ. Here the mean is over rows of a fixed logit matrix.
//
// (6) THE OBJECTIVE IS UNIMODAL BUT NOT CONVEX in T, in log T, or in 1/T.
//     The paper calls it "a one-dimensional convex optimization problem"
//     (Sec. 4.2); that was measured against scipy on five fixtures and does not
//     hold — see fit_temperature's implementation comment. A line search needs
//     only unimodality, so the search below is a golden-section line search on
//     log T, not a convexity-assuming gradient step. Searching log T also keeps
//     the bracket two-sided over T in (0, infinity), which a symmetric bracket
//     in T cannot express when T* ranges from 0.03 to 11.7.
//
// ===========================================================================

// softmax(logits / T). T must be > 0.
// logits is (N, K) and the result is (N, K), each row summing to 1.
Tensor temperature_scale(const Tensor& logits, double T);

// Mean negative log-likelihood of `labels` under softmax(logits / T), reduced
// over ROWS. `labels` is an (N, 1) column of class indices in [0, K); a
// non-integer, negative or out-of-range index throws std::invalid_argument.
double temperature_nll(const Tensor& logits, const Tensor& labels, double T);

// dNLL/dT, the closed form (verified against centered finite differences to
// rel < 1e-9 in tests/test_temperature_scaling.cpp):
//
//   dL/dT = (1 / (n T^2)) * sum_i [ z_{i,y_i} - sum_j p_j(i) * z_j(i) ]
//
// with p = softmax(z/T). Derivation:
//   d/dT lse(z/T) = -(1/T^2) * sum_j p_j z_j
//   d/dT (z_y/T) = -z_y / T^2
//   => dL/dT = (1/(n T^2)) sum_i [ z_y - sum_j p_j z_j ].
//
// THE ORDER OF THE BRACKET IS THE WHOLE FORMULA. Reversed, the result is
// exactly -dL/dT — a rel error of precisely 2.0 against finite differences,
// which is the fingerprint this test file pins.
double temperature_nll_grad(const Tensor& logits, const Tensor& labels, double T);

// Fit T on held-out data by minimising temperature_nll over [t_min, t_max].
//
// Returns the best T seen. The returned value is ALWAYS inside
// [t_min, t_max]: on data whose optimum lies outside the bracket (or at the
// degenerate T -> 0 limit of a perfectly separable set) the boundary is
// returned rather than a value the caller did not ask for.
//
// t_min must be > 0, t_max > t_min; otherwise std::invalid_argument. The
// defaults [1e-2, 1e2] cover the range in which temperature scaling is useful
// in practice; the true optimum can be far outside them for a strongly
// overconfident or strongly underconfident model, and the caller should widen
// the bracket rather than trust the default.
//
// max_iter bounds the line search (200 is far more than the ~60 needed to
// converge on the fixtures in the test suite — the paper notes the optimum is
// reachable "in 10 iterations").
double fit_temperature(const Tensor& logits, const Tensor& labels,
                       double t_min = 1e-2, double t_max = 1e2,
                       int max_iter = 200);

// Convenience: fit T on `logits`/`labels`, then return softmax(logits / T*).
// This is the call a user actually wants after training a model.
Tensor calibrate_logits(const Tensor& logits, const Tensor& labels,
                        double t_min = 1e-2, double t_max = 1e2);

#endif // NN_UTILS_TEMPERATURE_SCALING_H