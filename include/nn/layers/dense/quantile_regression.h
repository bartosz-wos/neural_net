#ifndef NN_LAYERS_DENSE_QUANTILE_REGRESSION_H
#define NN_LAYERS_DENSE_QUANTILE_REGRESSION_H

#include "../../core/layer.h"

#include <cstddef>
#include <vector>

// ===========================================================================
// QUANTILE REGRESSION -- the conditional quantile function
//
//   Koenker, R. and Bassett, G., Jr. -- "Regression Quantiles", Econometrica
//   46(1):33-50, 1978.
//
//   Quantile regression estimates the CONDITIONAL QUANTILE of a response,
//   not its conditional mean. Given features x and a level tau in (0, 1), it
//   fits the linear predictor
//
//       q_tau(x) = x . w_tau + b_tau
//
//   so that q_tau(x) estimates the tau-quantile of y given x. The mean is
//   the tau = 0.5 case; a model that emits SEVERAL levels at once is a
//   distributional model of y given x.
//
//   CITATION. Verified 2026-11-13 by a title search against the Econometric
//   Society's own index page, which gives the volume, issue and page range
//   verbatim: econometricsociety.org/publications/econometrica/1978/01/01/
//   regression-quantiles. **NO arXiv ID IS ASSERTED** -- the paper predates
//   arXiv by three decades and is not on it. This repo's expansion queue has
//   produced fifteen wrong arXiv identifiers, every one a plausible-looking
//   neighbour of the right paper.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-13)
//
//   grep -rinE 'quantile.regress|pinball|quantile.?head|non.?crossing'
//           include/ tests/ --include=*.h --include=*.cpp
//   -> include/nn/utils/conformal.h:11,25,74
//        (the CQR paper, and: "utils/distribution_losses.h ships QuantileLoss
//         (the pinball loss). That is the TRAINING LOSS CQR needs -- and it
//         is the strongest evidence that CQR belongs here: a shipped
//         primitive with NO CONSUMER.")
//   -> include/nn/utils/isotonic_regression.h:134
//        ("QUANTILE REGRESSION / CQR-style monotone quantile fitting. A
//         different objective (pinball loss, not squared loss).")
//   -> include/nn/utils/distribution_losses.h:106  (the QuantileLoss class)
//   -> tests/test_distribution_losses.cpp       (its own unit tests)
//
// ZERO IMPLEMENTATION of a quantile REGRESSION. The repo ships the
// CONSUMER and no PRODUCER:
//
//   // utils/conformal.h:259
//   Tensor cqr_score(const Tensor& lower, const Tensor& upper, const Tensor& y);
//
// `cqr_score` needs a lower and an upper quantile prediction. Nothing in
// ~250 layers produces one.
//
// It is NOT redundant with the shipped neighbours:
//   * utils/distribution_losses.h `QuantileLoss` is a LOSS and takes ONE q
//     for the whole (N, K) tensor, so it cannot express "column 0 is the
//     0.05-quantile, column 1 is the 0.95-quantile". It is the objective,
//     not the model.
//   * utils/isotonic_regression.h fits a monotone map on a single response.
//     It is a post-hoc calibration of a score, not a conditional quantile
//     model, and its header explicitly excludes the pinball objective.
//   * utils/conformal.h is POST-HOC: it calibrates a prediction that was
//     already made. A quantile regression produces the distribution FROM
//     THE DATA.
//
// ---------------------------------------------------------------------------
// THE COMPOSITION IS THE POINT
//
// The layer's output feeds the shipped conformal machinery with no glue:
//
//   Tensor pred = layer.forward(X);              // (N, K)
//   Tensor mono = layer.monotone(pred);           // (N, K), non-crossing
//   Tensor lower = layer.level_column(mono, 0);   // (N, 1)
//   Tensor upper = layer.level_column(mono, K-1); // (N, 1)
//   Tensor score = cqr_score(lower, upper, y);    // (N, 1)
//   Tensor iv = cqr_interval(l, u, qhat);         // (1, 2)
//
// ---------------------------------------------------------------------------
// THE PINBALL (QUANTILE) LOSS
//
// For level tau and residual e = y - pred,
//
//   rho_tau(e) = e * tau        if e >= 0   (under-prediction: pred <= y)
//              = e * (tau - 1)  if e <  0   (over-prediction:  pred >  y)
//
// which is the INTEGRAL OF ITS OWN SUBGRADIENT -- the definition-level form
//
//   rho_tau(u) = INTEGRAL_0^u ( tau - 1{v < 0} ) dv.                        (1)
//
// THE TIE CONVENTION IS NOT FREE, AND IT IS PINNED BY THE REPO.
//
// rho_tau is NOT differentiable at e = 0; its subgradient there is the SET
// [tau - 1, tau]. The repo's shipped `utils/distribution_losses.h`
// `QuantileLoss::backward` picks the RIGHT end (tau), i.e. the "e >= 0"
// branch, and its test asserts that. This layer MUST pick the same one, or
// the layer and the shipped loss disagree on the same data. The K = 1 case
// of this layer is therefore required to be BIT-FOR-BIT equal to the shipped
// QuantileLoss -- which pins the loss, the gradient, the tie convention and
// the reduction SIMULTANEOUSLY.
//
// THE REDUCTION IS "MEAN OVER THE BATCH, SUM OVER THE LEVELS":
//
//   L = (1/N) * sum_i sum_k rho_{tau_k}( y_i - pred_ik )
//
// which is what the shipped QuantileLoss does (it divides by N, the row
// count, and sums the columns). **THE CONSEQUENCE IS THAT THE TOTAL LOSS
// SCALES WITH K, AND SO DOES THE EFFECTIVE LEARNING RATE.** That is
// deliberate -- it is what makes the K = 1 equality hold.
//
// ---------------------------------------------------------------------------
// THE FITTED QUANTILE FUNCTIONS CAN CROSS, AND THE LAYER MEASURES IT
//
// The levels are fitted INDEPENDENTLY -- the classical Koenker-Bassett
// estimator -- because the k-th level's pinball loss depends only on the
// k-th column of the predictions and only on row k of the weight matrix.
// Nothing in the model therefore forces
//
//   q_{tau_1}(x) <= q_{tau_2}(x) <= ... <= q_{tau_K}(x)
//
// and the fitted functions CAN CROSS. That is a property of the ESTIMATOR,
// not a bug, and it is exactly what breaks a conformalised-quantile-
// regression interval (a lower bound above the upper bound gives a
// negative-width interval).
//
// So the layer ships THREE distinct things and the header is explicit about
// which is which:
//
//   forward(x)        -- the MODEL'S OWN output. It may cross. It is what
//                        the loss and the gradients are computed on.
//   crossing_count(p) -- a DIAGNOSTIC: how many adjacent level pairs are
//                        inverted. Zero iff every row is non-decreasing.
//   monotone(p)       -- a PROJECTION: each row sorted ascending. It is NOT
//                        the model's output, it is not differentiable in the
//                        useful sense (the sort is piecewise constant), and
//                        it must not be used as a training target.
//
// A caller who wants a guaranteed-ordered pair for `cqr_score` applies
// `monotone` to the model's predictions. A caller who wants to know whether
// the ordering held asks `crossing_count`.
//
// ---------------------------------------------------------------------------
// THE SHAPE OF y
//
// `y` is (N, 1) -- ONE response per row, compared against ALL K levels.
//
// This differs from the shipped `QuantileLoss`, which takes an ELEMENTWISE
// (N, K) target. Mixing the two up silently produces garbage, so the header
// says so twice and the runtime checks are explicit.
//
// ---------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, STATED SO IT CANNOT CREEP IN:
//   - NON-LINEAR quantile regression (a neural quantile model). A different
//     model class; the layer is deliberately LINEAR so that the fitted
//     quantile function is inspectable.
//   - A JOINT (non-crossing by construction) fit. The crossing is measured
//     and repaired by projection here; a constrained estimator is a different
//     algorithm with a different objective.
//   - SAMPLE WEIGHTS. The shipped `QuantileLoss` has none either.
//   - Any change to `utils/distribution_losses.h`.
//   - Any change to `utils/conformal.h`.
// ===========================================================================

class QuantileRegression : public Layer {
public:
    // in_features: the number of input features D. D >= 1.
    //
    // `quantiles`: the K levels. Each must lie in the OPEN interval (0, 1)
    // and they must be STRICTLY INCREASING -- the layer's columns are
    // ordered by level, and `level_column` and `monotone` rely on it.
    //
    // Throws std::invalid_argument on: D == 0; an empty level vector; a
    // level outside (0, 1); levels that are not strictly increasing. Every
    // message is DISTINCT -- a substring shared by two messages
    // discriminates nothing.
    QuantileRegression(size_t in_features, std::vector<double> quantiles);

    // (N, D) features -> (N, K) quantile predictions, column k at level
    // tau_k.
    //
    // **THE PREDICTIONS MAY CROSS.** See the header.
    //
    // Throws std::invalid_argument if the input's width is not D.
    Tensor forward(const Tensor& input) override;

    // `grad_output` is dL/d(pred), shape (N, K) -- exactly what
    // `pinball_gradient` produces. Accumulates into the parameter gradients
    // and returns dL/d(input), shape (N, D).
    //
    // Throws std::invalid_argument on a shape mismatch.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;

    // The parameter set is TWO tensors: the slope matrix (K, D) and the
    // intercept row (1, K). `get_weights` returns the slopes.
    Tensor get_weights() const override { return weights_; }
    Tensor get_gradients() const override { return grad_weights_; }
    std::vector<Tensor*> parameters() override { return {&weights_, &bias_}; }
    std::vector<Tensor*> gradients() override { return {&grad_weights_, &grad_bias_}; }
    void zero_grad() override;
    std::string name() const override { return "QuantileRegression"; }

    size_t num_levels() const { return quantiles_.size(); }
    size_t in_features() const { return in_features_; }

    // The K levels, in increasing order.
    const std::vector<double>& quantiles() const { return quantiles_; }

    // The pinball objective AT THE LAYER'S OWN LEVELS:
    //
    //   L = (1/N) * sum_i sum_k rho_{tau_k}( y_i - pred_ik )
    //
    // `pred` is (N, K) and `y` is (N, 1) -- ONE response per row, compared
    // against all K levels. The result is a (1, 1) tensor.
    //
    // Throws std::invalid_argument if `pred` does not have K columns, if
    // `y` is not (N, 1), or if the row counts disagree.
    Tensor pinball_loss(const Tensor& pred, const Tensor& y) const;

    // dL/d(pred), shape (N, K), with the SHIPPED tie convention: the
    // under-prediction branch (e >= 0) at e == 0.
    //
    // Throws the same way as `pinball_loss`.
    Tensor pinball_gradient(const Tensor& pred, const Tensor& y) const;

    // The pointwise monotone REARRANGEMENT: each row of `pred` sorted
    // ascending. A PROJECTION, not the model's output -- see the header.
    //
    // Throws std::invalid_argument if `pred` has no columns.
    Tensor monotone(const Tensor& pred) const;

    // The number of ADJACENT level pairs (k, k+1) whose predictions are
    // inverted, summed over the rows. Zero iff every row is non-decreasing
    // in the level.
    //
    // Throws std::invalid_argument if `pred` has no columns.
    size_t crossing_count(const Tensor& pred) const;

    // Column k of `pred` as (N, 1) -- the shape `cqr_score` consumes.
    //
    // Throws std::invalid_argument if k is out of range.
    Tensor level_column(const Tensor& pred, size_t k) const;

    // ---- parameter accessors (for the tests and for callers that want to
    // ---- set a model up by hand) ----
    double weight_at(size_t k, size_t d) const;
    void set_weight_at(size_t k, size_t d, double value);
    double bias_at(size_t k) const;
    void set_bias_at(size_t k, double value);

    // The accumulated parameter gradients, for inspection.
    Tensor weight_gradient() const { return grad_weights_; }
    Tensor bias_gradient() const { return grad_bias_; }

private:
    size_t in_features_;
    std::vector<double> quantiles_;

    Tensor weights_;       // (K, D)  the slope of each level
    Tensor bias_;          // (1, K)  the intercept of each level
    Tensor grad_weights_;  // (K, D)
    Tensor grad_bias_;     // (1, K)
    Tensor last_input_;    // (N, D)  cached for backward
};

#endif  // NN_LAYERS_DENSE_QUANTILE_REGRESSION_H
