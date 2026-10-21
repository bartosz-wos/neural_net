#ifndef NN_UTILS_WEIGHT_STD_H
#define NN_UTILS_WEIGHT_STD_H

#include "../core/tensor.h"
#include "../core/layer.h"
#include <cstddef>
#include <vector>

// ===========================================================================
// Weight Standardization (WS) and its exact adjoint.
//
//   Qiao, Wang, Liu, Shen, Yuille -- "Micro-Batch Training with Batch-Channel
//   Normalization and Weight Standardization", arXiv:1903.10520.
//
// ---------------------------------------------------------------------------
// CITATION. Verified 2026-10-22 by scraping the arXiv ABS PAGE for its
// `citation_title` / `citation_author` meta tags: the title matches verbatim and
// the five authors are Qiao, Wang, Liu, Shen, Yuille in that order. Equation
// (10) below was read from the paper's own HTML rendering (ar5iv), not from
// memory.
//
// THREE WRONG IDS, measured on the same day, that MUST NOT be reintroduced:
//   * arXiv:1903.10520 is NOT "Weight Standardization for Adversarial
//     Robustness" -- it is the micro-batch / BCN paper. It is the CORRECT id for
//     the paper that DEFINES WS; a memory-typed citation points the other way,
//     which is how this file's sibling queue has produced nine wrong ids.
//   * "Weight Standardization for Adversarial Robustness" (CVPR 2020) is NOT
//     LOCATABLE from this host: an arXiv `ti:` title query returns 0 and
//     OpenAlex `title.search` returns 0, and openaccess.thecvf.com serves a
//     2.6 kB block page at HTTP 200. Its design is the same formula, but NO
//     identifier is claimed for it.
//   * arXiv:1805.09555 = "Phase Retrieval via Polytope Optimization";
//     arXiv:1804.08868 = "Rational proofs for quantum computing". Both were
//     guessed as Lipschitz-constrained-NN papers. Neither is.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
//
// `grep -rinE 'weight.?standard|ws_norm|lipschitz|learnable_lip|gain.?schedul'
//  include/ tests/` returns ONE hit, and it is the NAME of a test in
// test_spectral_norm.cpp (`test_end_to_end_lipschitz`). No implementation of the
// concept exists anywhere in ~250 layers and 44 utility files.
//
// It is NOT redundant with the two mechanisms that share its name's neighbourhood:
//
//   * layers/normalization/weight_norm.h keeps a LEARNABLE LENGTH `g_i`,
//     does NOT center, is not a `Layer`, and has no adjoint. WS removes the
//     learnable length entirely ("removes the learnable weight length g",
//     paper Sec. IV) and adds the centering.
//   * layers/normalization/spectral_norm.h constrains the SPECTRAL NORM OF THE
//     WHOLE MATRIX to 1 by power iteration. WS bounds each CHANNEL separately
//     and exactly. Different quantities.
//
// ---------------------------------------------------------------------------
// THE TRANSFORM, and the two constants that decide whether it is right
//
//     W_hat[c][j] = (W[c][j] - mean_j W[c][j]) / sqrt(var_j W[c][j] + eps)
//
//   (1) PER ROW, not per column. The paper standardizes "each output channel",
//       and on this repo's 2-D `(out, in)` weight a channel is a ROW. This is
//       invisible on a square matrix, so the suite carries 5x3, 4x7 and 6x4
//       fixtures on purpose.
//   (2) eps goes INSIDE the sqrt. `sqrt(var + eps)`, not `sqrt(var) + eps` and
//       not `sqrt(var) * sqrt(1 + eps)`. At eps=1e-3 on a length-3 row the
//       difference is ~2.5e-4, which a 1e-2 tolerance swallows and a 1e-4 one
//       does not.
//   (3) The variance is POPULATION (divide by K), never sample (1/(K-1)). WS
//       standardizes exactly K weights, so the sample std gives
//       ||w_hat||_2 == sqrt(K/(K-1)) and every "unit variance" claim passes
//       with the wrong divisor.
//
// NO AFFINE TRANSFORMATION IS APPLIED -- the paper is explicit: "we do not have
// any affine transformation on W_hat", because a downstream BN/GN normalizes
// the output anyway and an affine here would confound the scale it is fixing.
//
// ---------------------------------------------------------------------------
// THE LIPSCHITZ BOUND: THERE ARE TWO, AND THE OBVIOUS ONE IS FALSE
//
// This is the finding that matters, and the oracle measured it:
//
//     ||W_hat||_op <= sqrt(K)      <-- FALSE, asserted in the oracle's first
//                                      draft and refuted by measurement
//                                      (6x4 random: 3.715543100006 vs 2.0)
//
// Two true statements, about two DIFFERENT quantities:
//
//   (a) PER-ELEMENT (the paper's argument, and what `ws_channel_lipschitz`
//       returns): each output is `w_c . x` with `||w_c||_2 == sqrt(K)` exactly,
//       so `|w_c . x| <= sqrt(K) ||x||_2` by Cauchy-Schwarz. K = in_channels.
//   (b) WHOLE-MATRIX: `||W_hat||_op <= ||W_hat||_F == sqrt(O*K)`, the Frobenius
//       bound, because each of the O standardized rows contributes exactly K to
//       the squared Frobenius norm. O = out_channels.
//
// Asserting only (b) with sqrt(K) passes a wrong implementation; asserting only
// (a) can never see a row/column confusion. This header exposes (a) as the
// primary function and the test suite asserts both, plus that the false one is
// actually violated.
// ===========================================================================

namespace nn {
namespace utils {

// Standardize each ROW of `w` to zero mean and unit population variance.
// `w` is NOT modified; the return value is a fresh tensor of the same shape.
// Throws std::invalid_argument if `w` is empty, if a row has fewer than 2
// entries (a variance needs a divisor), if `eps` is negative or not finite, or
// if a row's `var + eps` is not strictly positive -- a constant row at eps == 0
// is 0/0, and returning a plausible-looking garbage tensor there is exactly the
// trap this file refuses (cf. hutchinson's eps cancellation floor).
Tensor ws_standardize(const Tensor& w, double eps = 1e-12);

// The exact adjoint of `ws_standardize`: given `g = dL/dW_hat` it returns
// `dL/dW`. Throws the same guards as the forward, plus a shape mismatch.
//
//     dL/dW[c][j] = (1/s) * ( g[c][j] - mean_j g[c][j]
//                              - x_hat[c][j] * mean_j ( g[c][j] * x_hat[c][j] ) )
//
// TWO TRAPS, both of which a plausible implementation gets wrong:
//
//   (a) `mean(g * x_hat)` is over x_hat and NOT over `u = W - mean(W)`. The two
//       differ by exactly 1/s, so using u is a CLEAN FACTOR-OF-s error that a
//       loose relative tolerance can miss on small-variance rows.
//   (b) the `1/s` prefactor is OUTSIDE the parenthesis. Moving it inside turns
//       the update into a `1/s^2` weighting of the `x_hat * mean(g*x_hat)` term.
Tensor ws_standardize_backward(const Tensor& w, const Tensor& g, double eps = 1e-12);

// Apply WS IN PLACE, row by row. This is what the layer uses; the free
// function above is non-mutating because a caller may not expect it to write.
void ws_standardize_(Tensor& w, double eps = 1e-12);

// The per-channel (per-element) Lipschitz constant of channel `c`: the exact
// value is sqrt(K) at eps == 0 and sqrt(K * var / (var + eps)) at eps > 0.
// Reported for the ALREADY-standardized tensor.
double ws_channel_lipschitz(const Tensor& w_hat, size_t c);

// The per-channel Lipschitz constant BEFORE standardization, i.e. the value
// `ws_standardize` would produce for channel `c`, computed without
// materializing the standardized matrix.
double ws_channel_lipschitz_of(const Tensor& w, size_t c, double eps = 1e-12);

// The MEAN per-channel Lipschitz constant over every channel. Exposed because a
// single channel is rarely the one an architecture depends on.
double ws_mean_channel_lipschitz(const Tensor& w, double eps = 1e-12);

// ===========================================================================
// WeightStandardization -- a thin wrapper that standardizes a wrapped layer's
// first parameter tensor and routes the adjoint back to the RAW weights.
//
// WHY THE ADJOINT IS THE LOAD-BEARING PART. `Layer::backward` returns
// dL/d(input) and accumulates into the layer's OWN gradients, which are then
// dL/d(standardized weights) because that is what the forward computed with. A
// wrapper that standardizes and then simply calls `wrapped_->backward()` trains
// the PROJECTION rather than the weights -- a silent no-op that looks like it
// works and produces zero useful learning signal on the raw parameter. So the
// wrapper additionally converts the accumulated gradient back through
// `ws_standardize_backward` and accumulates THAT into the raw snapshot, which is
// what `v_` holds and what the wrapped optimizer then updates.
//
// ACCUMULATION IS DELIBERATE AND ASSERTED. `Dense::backward` and friends
// ACCUMULATE into `gradients()` (core/layer.cpp), so calling backward twice
// against one snapshot must sum both adjoints. Test S11 pins that.
// ===========================================================================
class WeightStandardization {
public:
    // Wraps `inner`, standardizing `inner->parameters()[0]`. A null pointer, a
    // layer with no parameters, or a non-2-D parameter all throw.
    explicit WeightStandardization(Layer* inner, double eps = 1e-12);
    ~WeightStandardization();

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output, double learning_rate);
    // Moves the RAW weights into `v_` (the wrapper's optimization target) and
    // applies them back through `inner`'s normal `update_weights`.
    void update_weights(double learning_rate);
    void zero_grad();

    Layer* inner() const { return inner_; }

    // The RAW weights, i.e. the tensor this wrapper optimizes. Exposed so a
    // caller can inspect or seed them; `parameters()` deliberately does NOT
    // return it, because an optimizer driving the wrapper would otherwise update
    // the raw tensor WITHOUT the projection and the two would silently diverge.
    const Tensor& raw() const { return v_; }

    // The gradient with respect to the RAW weights -- the quantity an optimizer
    // needs and the one the plain `gradients()` does not provide.
    const Tensor& raw_grad() const { return raw_grad_; }

    size_t raw_rows() const { return v_.rows; }
    size_t raw_cols() const { return v_.cols; }

private:
    Layer* inner_;
    double eps_;
    Tensor v_;          // RAW weights; the optimization target
    Tensor v_grad_;     // gradient wrt RAW weights (what update_weights uses)
    Tensor raw_grad_;   // public read-only view of v_grad_
    Tensor snapshot_;   // the standardized weights as the forward saw them

    void sync_from_inner();
    void standardize_params_();
};

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_WEIGHT_STD_H