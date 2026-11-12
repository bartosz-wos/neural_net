#ifndef NN_LAYERS_DENSE_ORDINAL_REGRESSION_H
#define NN_LAYERS_DENSE_ORDINAL_REGRESSION_H

#include "../../core/layer.h"

#include <cstddef>
#include <vector>

// ===========================================================================
// ORDINAL REGRESSION -- THE CUMULATIVE LINK MODEL
//
//   McCullagh, P. -- "Regression analysis of ordinal data", Journal of the
//   Royal Statistical Society, Series B 42(2):109-142, 1980.
//
//   Ordinal logistic regression; the proportional-odds model. The label is
//   ORDERED (a Likert rating, a severity grade, a satisfaction level), and
//   the model says so: a higher latent score moves probability mass towards
//   HIGHER classes.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-12)
//
//   grep -rinE 'ordinal.regress|cumulative.link|probit|ordered.logit' include/
//   -> only `gaussian_process.h`'s prose mention of "Bernoulli-probit GPs"
//      and `utils/rank_correlation.cpp:8`'s note about not hand-rolling the
//      probit function. ZERO IMPLEMENTATION.
//
// The repo has ~250 layers and 60+ utility files and NO ordinal model. What
// it has instead is the NOMINAL half and the CALIBRATION half:
//
//   * utils/losses.h NLLLoss -- NLL against log-softmax. It assumes K
//     UNORDERED classes: the class index carries no order, so "2 vs 3" and
//     "2 vs 9" are equally wrong.
//   * utils/metrics.h -- accuracy / ECE / Brier. An accuracy over an ORDERED
//     label set is a poor metric, because being off by one is far less bad
//     than being off by nine.
//   * utils/temperature_scaling.h -- fits ONE scalar T. It recalibrates; it
//     does not ORDER.
//
// **THE COMPOSITION IS THE POINT.** An ordinal model emits a distribution
// over an ordered label set, so its natural readout is the EXPECTED CLASS
// E[Y|x] = sum_k k * P(Y=k|x) -- a real number, not an argmax. That is what
// a Likert-scale response needs, and nothing in the repo can produce it.
//
// ---------------------------------------------------------------------------
// THE MODEL, STATED FROM THE DEFINITION
// ---------------------------------------------------------------------------
// There is a LATENT variable
//
//     Y* = eta + eps,        eps ~ standard Logistic(0, 1)
//
// and K-1 ORDERED cut points theta_1 < ... < theta_{K-1} (with theta_0 = -inf
// and theta_K = +inf) such that
//
//     Y = k   iff   theta_{k-1} < Y* <= theta_k .
//
// Therefore, with f(u) = e^{-u}/(1+e^{-u})^2 the standard logistic density,
//
//     P(Y = k | x) = INTEGRAL_{theta_{k-1}}^{theta_k} f(z - eta) dz .
//
// and the latent score is eta = x . beta.
//
// **THERE IS NO INTERCEPT**, and that is a decision, not an omission.
// P(Y <= k) = sigmoid(theta_k - eta) is INVARIANT under shifting every theta
// by c and eta by c, so an intercept would be EXACTLY redundant with a
// uniform shift of the cut points. Pinning it to zero removes that
// degeneracy instead of hiding it behind an unidentifiable pair.
//
// ---------------------------------------------------------------------------
// THE PARAMETERISATION, AND WHY THE GAPS ARE KEPT SEPARATE
// ---------------------------------------------------------------------------
// The cut points are stored as K-1 RAW parameters r, with
//
//     theta_1 = r_1
//     theta_k = theta_{k-1} + gap_k,      gap_k := softplus(r_k) > 0
//
// so the ORDERING IS STRUCTURAL: no optimiser step can produce
// theta_k <= theta_{k-1}, and hence no class probability can ever be
// negative. The map r -> theta is a bijection onto the strictly increasing
// vectors, so nothing is lost.
//
// **THE GAPS MUST NOT BE RECOVERED BY DIFFERENCING THE THETAS.** The
// cumulative sum theta_k = theta_{k-1} + gap_k can ABSORB a tiny gap when
// theta_1 is large: theta_1 = 100 with gap = softplus(-100) = 3.7e-44 gives
// theta_2 == theta_1 EXACTLY in float64, so differencing reports a gap of
// 0. The gap enters the class probability AND the gradient, so the gap is
// kept as its own quantity, computed from r directly.
//
// ---------------------------------------------------------------------------
// THE FOUR THINGS THAT ARE EASY TO GET WRONG
// ---------------------------------------------------------------------------
// **1. THE PROBABILITY OF AN INTERIOR CLASS IS A DIFFERENCE OF TWO
// CUMULATIVES, AND THE NAIVE DIFFERENCE LOSES ALL OF IT.**
//
//     P(Y = k) = sigma(a_k) - sigma(a_{k-1}),   a_k = theta_k - eta.
//
// When eta is large and negative, BOTH a_k and a_{k-1} are large and
// positive, so BOTH sigmoids round to exactly 1.0 in float64 and the
// difference is EXACTLY ZERO -- the class is silently assigned probability 0
// while its true probability is around 2.7e-18. That is not a rounding
// curiosity: it is the difference between a class that is merely very
// unlikely and a class that is impossible, and the NLL of the latter is
// +inf.
//
// The stable identity is
//
//     sigma(u) - sigma(v) = sigma(u) * sigma(-v) * (1 - e^{-(u-v)})  (u > v)
//
// because sigma(u) - sigma(v) = (e^{-v} - e^{-u})/((1+e^{-u})(1+e^{-v}))
// and e^{-v} - e^{-u} = e^{-v}(1 - e^{-(u-v)}). Every factor is O(1), so the
// product keeps full RELATIVE accuracy where the naive difference loses it
// completely.
//
// **2. THE SAME CANCELLATION SITS INSIDE THE GRADIENT, ONE LEVEL DEEPER,
// WHICH IS WHY THIS LAYER OUTPUTS LOG PROBABILITIES.**
//
// The probability-space gradient carries the factor
//
//     d sigma(a) / d eta = -sigma(a) * (1 - sigma(a))
//
// and where sigma(a) has rounded to exactly 1.0 that factor is EXACTLY
// ZERO. A gradient computed in probability space is therefore ZERO where
// the true gradient is about -1 -- a 100% relative error -- and no amount
// of care in the probability computation can rescue it.
//
// In LOG space the derivative factors are
//
//     d log sigma(a) / d a      = 1 - sigma(a) = sigmoid(-a)   [stable]
//     d log sigma(-a) / d a   = -sigma(a)                    [stable]
//
// and the interior classes carry ONE extra term, because their probability
// depends on the GAP between two cut points and not only on their
// positions:
//
//     log P_k = log sigma(a_k) + log sigma(-a_{k-1}) + log(1 - e^{-gap_{k-1}})
//
// with gap_{k-1} = theta_k - theta_{k-1} (independent of eta), so
//
//     q_{k-1} := e^{-gap_{k-1}}/(1 - e^{-gap_{k-1}}) = 1/(e^{gap_{k-1}} - 1)
//
//     d log P_k / d a_k     =  (1 - c_k) + q_{k-1}
//     d log P_k / d a_{k-1} = -(c_{k-1} + q_{k-1})
//
// and for the two END classes, which have no gap:
//
//     d log P_1 / d a_1     =  (1 - c_1)
//     d log P_K / d a_{K-1} = -c_{K-1}
//
// **THE q TERM IS THE WHOLE POINT.** It is what makes the interior classes'
// gradient w.r.t. the cut points correct: the probability of an interior
// class depends on the SPREAD of two cut points. An implementation that
// differentiates only the positions gets the cut-point gradient wrong.
//
// **3. THE ORDERING IS STRUCTURAL** (see above), and the gaps must not be
// differenced.
//
// **4. THE INTERIOR CLASSES' PROBABILITY DEPENDS ON EVERY CUT POINT ABOVE
// THE GAP, NOT ON ONE.** A gap shift moves EVERY cut point above it, so it
// touches EVERY class at or above it -- in particular the last class, whose
// probability is 1 - sigma(a_{K-1}). Differentiating a gap therefore has
// THREE cases per class, and getting them wrong is easy:
//
//     gap j <= k-2  : BOTH theta_k and theta_{k-1} shift -> (e_k - c_{k-1})
//     gap j == k-1  : ONLY theta_k shifts, and the gap itself changes
//                     -> (e_k + q_{k-1})
//     gap j >= k    : nothing
//
// **AND THE q TERM CANCELS BETWEEN THE TWO theta-PARTIALS.** With a tiny
// gap q is astronomically large (2.7e43 for a 3.7e-44 gap) and the
// cancellation destroys the answer -- an implementation that sums the two
// theta-partials returns EXACTLY 0 where the true answer is -1. The
// implementation differentiates in (u, gaps) coordinates instead, where q
// never has to cancel.
//
// ---------------------------------------------------------------------------
// WHAT THE LAYER RETURNS, AND WHY IT MATTERS FOR THE COMPOSITION
// ---------------------------------------------------------------------------
// `forward` returns the LOG class probabilities, shape (N, K).
//
// This is not a stylistic choice -- it is item 2 above -- and it is also
// what makes the layer compose with the repo's EXISTING loss: the
// `utils/losses.h` `NLLLoss::backward` already produces exactly
// dL/d(log P), which is what `OrdinalRegression::backward` consumes. There
// is no glue code and no 1/P anywhere.
//
// Callers who want the probabilities call exp() on the result.
// ===========================================================================

class OrdinalRegression : public Layer {
public:
    // in_features: the number of input features D.
    // num_classes: the number of ORDERED classes K. There are K-1 cut
    //              points. K >= 2.
    //
    // Throws std::invalid_argument on K < 2 or D == 0.
    OrdinalRegression(size_t in_features, size_t num_classes);

    // (N, D) features -> (N, K) LOG class probabilities.
    //
    // Throws std::invalid_argument if the input's width is not D.
    Tensor forward(const Tensor& input) override;
    // `grad_output` is dL/d(log P), shape (N, K) -- the shape the repo's
    // NLLLoss::backward already produces. Accumulates into the parameter
    // gradients and returns dL/d(input), shape (N, D).
    //
    // Throws std::invalid_argument on a shape mismatch.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;

    // The parameter set is TWO tensors: the slope and the raw cut points.
    // `get_weights` returns the slope (the primary parameter); the cut
    // points are reachable through `parameters()` and `cut_points()`.
    Tensor get_weights() const override { return beta_; }
    Tensor get_gradients() const override { return grad_beta_; }
    std::vector<Tensor*> parameters() override { return {&beta_, &raw_cuts_}; }
    std::vector<Tensor*> gradients() override { return {&grad_beta_, &grad_raw_cuts_}; }
    void zero_grad() override;

    size_t num_classes() const { return num_classes_; }
    size_t in_features() const { return in_features_; }

    // The ORDERED cut points actually in force, shape (1, K-1).
    // Non-decreasing by construction.
    //
    // **THE FLOAT64 CUT POINTS CAN BE EQUAL EVEN THOUGH THE MODEL'S GAPS
    // ARE NOT.** The cumulative sum theta_k = theta_{k-1} + gap_k can
    // ABSORB a tiny gap: theta_1 = 100 with a gap of 3.7e-44 gives
    // theta_2 == theta_1 EXACTLY in float64. That is a limit of the
    // REPRESENTATION, not a defect of the model -- the gap the model uses
    // is still 3.7e-44, and `cut_gaps()` reports it. Differencing the
    // cut points to recover the gap would report 0, and the gap enters the
    // class probability AND the gradient, so the implementation never does.
    Tensor cut_points() const;

    // The K-2 GAPS between consecutive cut points, shape (1, K-2),
    // computed from the raw parameters as softplus(r_k) and NEVER recovered
    // by differencing the cut points. Every gap is strictly positive.
    Tensor cut_gaps() const;

    // The EXPECTED CLASS E[Y|x] = sum_k k * P(Y=k|x), shape (N, 1).
    // The natural readout of an ordinal model: a real number, not an argmax.
    Tensor expected_class(const Tensor& input) const;

    // ---- parameter accessors (used by the tests and by callers that want
    // ---- to set a model up by hand) ----

    double beta_at(size_t d) const;
    void set_beta(size_t d, double value);
    double raw_cut_at(size_t k) const;
    void set_raw_cut(size_t k, double value);

    // The accumulated parameter gradients, for inspection.
    Tensor beta_gradient() const { return grad_beta_; }
    Tensor cut_gradient() const { return grad_raw_cuts_; }

private:
    // The log class probabilities for `input`, WITHOUT touching the cached
    // last_input_. `expected_class` is a const reader, and it must not
    // clobber the state that `backward` depends on.
    Tensor log_probs(const Tensor& input) const;

    size_t in_features_;
    size_t num_classes_;
    Tensor beta_;            // (1, D)     the slope
    Tensor raw_cuts_;        // (1, K-1)   the raw cut-point parameters
    Tensor grad_beta_;       // (1, D)
    Tensor grad_raw_cuts_;   // (1, K-1)
    Tensor last_input_;      // cached for backward
};

#endif  // NN_LAYERS_DENSE_ORDINAL_REGRESSION_H
