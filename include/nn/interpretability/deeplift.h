#ifndef DEEPLIFT_H
#define DEEPLIFT_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <vector>

// =============================================================================
// DeepLIFT with the Rescale rule.
//
// Shrikumar, Greenside, Kundaje, "Learning Important Features Through Propagating
// Activation Differences", ICML 2017, arXiv:1704.02685.
//
// CITATION NOTE: EXPANSION_QUEUE.md and the implementation plan recorded this
// work as arXiv:1703.01385. That ID is wrong — the arXiv API returns "Truncated
// Bernoulli-Carlitz and truncated Cauchy-Carlitz numbers" for 1703.01385, a
// number-theory paper by Komatsu. The DeepLIFT paper is 1704.02685. Do not
// re-introduce 1703.01385; see docs/plans/2026-10-09-deeplift.md decision D1.
//
// THE RULE. Given an input x and a reference (baseline) x', propagate the
// DIFFERENCE through the network instead of the signal. For each neuron i of an
// elementwise nonlinear layer the Rescale multiplier is the secant slope of the
// activation across the pair (a, a'):
//
//   m_i = (f(a_i) - f(a'_i)) / (a_i - a'_i)
//
// and it rescales the running adjoint ("importance") tensor backward:
//
//   G_in  = G_out ⊙ m                    nonlinear layers
//   G_in  = G_out                        linear layers (identity multiplier)
//
// Each layer contributes (a - a') ⊙ G_in, and the INPUT layer's contribution
// IS the attribution. The choice of m is what distinguishes the rules: Rescale
// uses the secant slope (plan D2, captum `deep_lift.py:nonlinear()`); Reveal-
// Cancel would subtract the input's own gradient instead. Only Rescale is
// implemented — captum likewise ("This implementation supports only Rescale
// rule", dl_core.py:64-65).
//
// THE ε-FALLBACK (plan D3) — THE ONE DETAIL NO EYE-BALLING CATCHES. When
// a_i == a'_i the multiplier is 0/0. It is NOT 0 and NOT 1: the limit is
// f'(a), i.e. the true gradient. captum deep_lift.py:982-984 implements exactly
// this — `torch.where(abs(delta_in) < eps, grad_input, grad_out * delta_out /
// delta_in)` with eps = 1e-10, where grad_input is the ordinary dL/d(input).
// Getting this wrong is invisible except on x == x', where an unguarded
// division yields NaN and returning 0 yields a finite-but-wrong attribution
// that no completeness test would catch.
//
// COMPLETENESS (the axiom):
//
//   F(x) - F(x') == sum_i attribution_i
//
// Note this is the sum of the INPUT layer's contribution ALONE, NOT the sum
// over all layers' contributions. Summing every layer counts the same output
// change once per layer and is wrong — on a 5-layer Dense/ReLU net the input
// contribution sums to 0.575 == F(x) - F(x') while all five layers together
// sum to 2.875. (docs/plans/2026-10-09-deeplift.md:47 states the all-layers
// version in prose while its own pseudocode at :243 computes the correct one;
// the pseudocode is right.)
//
// WHY THIS IS CHEAPER THAN INTEGRATED GRADIENTS. IG needs `n_steps` forward +
// backward passes per instance because it integrates dF/dx along the path.
// DeepLIFT needs exactly TWO forward passes (baseline, then input) and ONE
// backward walk — O(1) in the number of path samples, at the price of assuming
// the network is piecewise linear between the pair (a, a') and of taking the
// secant slope rather than an averaged derivative. On a purely LINEAR network
// every multiplier is the identity and DeepLIFT reduces EXACTLY to
// attribution_i = (x_i - x'_i) · ∂F/∂x_i, which makes the whole method
// verifiable in closed form (see test_deeplift.cpp).
//
// CONVENTION — identical to attribution.h, shapley.h and lime.h: the model
// output is (rows, n_classes) and `target` selects which output column is the
// scalar being attributed. Attributions are on the LOGIT, never a softmax
// probability.
//
// SCOPE — 2-D Tensor MLPs. Reveal-Cancel, convolutional / maxpool / softmax
// modules, and gradients THROUGH the attributions (arXiv:1711.06104) are out of
// scope.
// =============================================================================

// Result of one DeepLIFT pass.
struct DeepLiftResult {
    // Per-element attributions, same (rows, cols) shape as the input. This is
    // the INPUT layer's contribution (a - a') ⊙ G_in.
    Tensor attributions;
    // sum(attributions) - (F(x) - F(x')). Zero is the completeness axiom.
    //
    // Unlike IG's counterpart this is NOT an approximation error: with the
    // Rescale rule the recursion is exact, so this is at machine precision on
    // any piecewise-linear net and should only ever be round-off. It stays in
    // the struct so a caller can assert the axiom without recomputing F.
    double completeness_delta = 0.0;
    // Layers traversed by the backward walk.
    size_t n_layers = 0;
};

// DeepLIFT attribution of logit `target` to `input`, relative to `baseline`.
//
// `baseline` is the reference point x' — conventionally zeros, an empirical
// mean image, or a blurred version of the input. Its shape must match
// `input`'s. `eps` is the threshold below which a multiplier is treated as
// degenerate and replaced by the derivative (see the header).
//
// Throws std::invalid_argument on a shape mismatch, a non-positive eps, or an
// empty model; std::out_of_range if `target` exceeds the model output width.
//
// Leaves no residue in the model's parameter gradients (zeroed before and
// after), matching input_gradient, integrated_gradients and lime_explain.
DeepLiftResult deep_lift(Model& model,
                         const Tensor& input,
                         const Tensor& baseline,
                         size_t target = 0,
                         double eps = 1e-10);

// Zero-baseline convenience overload (x' = 0).
DeepLiftResult deep_lift(Model& model,
                         const Tensor& input,
                         size_t target = 0,
                         double eps = 1e-10);

// DeepLIFT over K baselines, contributions averaged — the "DeepLIFT SHAP"
// variant of the paper's §4.2 and captum's DeepLiftShap (dl_core.py:593). The
// averaging over reference points is what removes the baseline-choice bias that
// single-reference DeepLIFT inherits from wherever x' was drawn.
//
// Throws std::invalid_argument if `baselines` is empty or any entry's shape
// differs from `input`'s; the remaining exceptions are deep_lift's.
DeepLiftResult deep_lift_shap(Model& model,
                              const Tensor& input,
                              const std::vector<Tensor>& baselines,
                              size_t target = 0,
                              double eps = 1e-10);

#endif // DEEPLIFT_H