#ifndef NN_UTILS_TTA_H
#define NN_UTILS_TTA_H

#include "../core/tensor.h"
#include <cstddef>
#include <vector>

// ===========================================================================
// Test-time augmentation
//
//   Divya Shanmugam, Davis Blalock, Guha Balakrishnan, John Guttag --
//   "Better Aggregation in Test-Time Augmentation", arXiv:2011.11156
//   (ICCV 2021).
//
// CITATION. Verified on 2026-10-16 against https://arxiv.org/abs/2011.11156
// directly (abs-page scrape: title, the four author links, the abstract). The
// arXiv EXPORT API was rate-limited from this host -- "Rate exceeded." on three
// attempts, http and https, with and without a User-Agent, 25 s apart -- so the
// ID was confirmed from three independent sources instead: the arXiv abs page
// itself, NASA ADS (2020arXiv201111156S, DOI 10.48550/arXiv.2011.11156,
// "E-Print: ICCV 2021") and emergentmind. This repo's expansion queue has
// produced EIGHT wrong arXiv IDs before, so the check is not optional.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//
// `Model::forward(const Tensor&)` returns ONE tensor for ONE input. Measured by
// content grep before this file was written:
//
//     grep -ril 'test.time.augment|test_time_augment|\btta\b|ensemble.*predict'
//          include/ tests/
//
// -> 0 HITS. The repo ships ~250 layers and ten inference-time utilities --
// among them `adversarial.{h,cpp}` (search for a worst-case delta),
// `randomized_smoothing.{h,cpp}` (bound any attack) and `vat.{h,cpp}` (smooth
// the model) -- and NOTHING that runs a model on several transformations of one
// input and combines the results.
//
// That is not a cosmetic omission. The paper's headline finding is that
// aggregating augmentations can produce a BETTER aggregate score while
// converting predictions that were CORRECT into ones that are WRONG, so a
// utility that reports only the aggregate accuracy cannot see its own failure.
// The report type below exists specifically to expose that count separately.
//
// ---------------------------------------------------------------------------
// THE PAPER'S OWN HEADLINE, verbatim from the abstract
// ---------------------------------------------------------------------------
//
//   "A key finding is that even when test-time augmentation produces a net
//    improvement in accuracy, it can change many correct predictions into
//    incorrect predictions."
//
// Hence `TtaFlipReport` returning `changed`, `correct_to_incorrect` and
// `incorrect_to_correct` as SEPARATE fields, and why `net_change()` is a
// convenience accessor rather than the primary result. On this suite's
// end-to-end fixture the aggregate nets +1 while LOSING one prediction the base
// transform got right -- both facts are measured, not asserted from the paper.
//
// ---------------------------------------------------------------------------
// SHAPES
// ---------------------------------------------------------------------------
//
// `Tensor` is strictly two-dimensional (`core/tensor.h`), so a logit bundle is
// carried as `(T, N * K)` -- TRANSFORM-MAJOR (D1). Element `(t, n * K + k)` is
// the logit of class `k` for row `n` under transform `t`. The alternative
// ordering, `(N, T, K)` flattened, mixes transforms of different rows into one
// accumulation and is the single easiest mistake here.
//
// Aggregated outputs are `(N, K)`. `tta_split_row(t, n, k, K)` is the accessor
// the tests use to read a bundle cell.

// Row of the bundle belonging to transform `t` -- a (N, K) view by VALUE.
Tensor tta_transform_view(const Tensor& bundle, size_t transform,
                          size_t num_classes);

// Number of transforms in a bundle, given the per-transform width.
size_t tta_num_transforms(const Tensor& bundle, size_t num_classes);

// ---------------------------------------------------------------------------
// PREDICTION UTILITIES
// ---------------------------------------------------------------------------

// Class index per row of an `(N, K)` tensor.
//
// D4: ties resolve to the LOWEST class index (strict `>`, matching the existing
// `argmax_row` in `metrics.cpp:38`). No tolerance-based assertion can observe a
// tie-break -- only an EXACT tie can -- so the suite's fixture carries
// bit-identical ties.
//
// THROWS std::invalid_argument on an empty tensor (no columns, or no rows).
std::vector<size_t> tta_argmax(const Tensor& t);

// ---------------------------------------------------------------------------
// TRANSFORMS
// ---------------------------------------------------------------------------

// `x` is (N, H*W) -- a BATCH of flattened images, one image per row.
// The identity view, returned as a COPY (never a reference into `x`).
Tensor tta_identity(const Tensor& x);

// Horizontal flip, applied WITHIN EACH IMAGE ROW (D10).
//
// The flip reverses the WIDTH axis of each (H, W) image independently. A
// whole-buffer reversal would also reverse the order of the IMAGES, which is a
// different -- and wrong -- transform; the suite asserts the two are
// distinguishable rather than assuming the row-locality is obvious.
//
// Involutory: `tta_hflip(tta_hflip(x, H, W), H, W) == x` bitwise.
//
// THROWS std::invalid_argument if H == 0, W == 0, or `H * W != x.cols`.
Tensor tta_hflip(const Tensor& x, size_t height, size_t width);

// The canonical two-view set {identity, hflip}, in that order. Transform 0 is
// the identity, so it can serve as the "base" prediction the flip report
// compares against.
std::vector<Tensor> tta_augment(const Tensor& x, size_t height, size_t width);

// ---------------------------------------------------------------------------
// FLIP REPORT (D9)
// ---------------------------------------------------------------------------

// What aggregation did to the base prediction, counted separately.
//
//     changed              -- rows whose predicted CLASS differs
//     correct_to_incorrect -- rows the base got RIGHT and the aggregate got wrong
//     incorrect_to_correct -- rows the base got WRONG and the aggregate fixed
//
// `changed == correct_to_incorrect + incorrect_to_correct` ALWAYS -- there are
// only two outcomes for a changed row, so the suite asserts the identity as an
// invariant rather than taking it on trust.
//
// `labels` is the ground truth, one entry per row.
//
// THROWS std::invalid_argument if the row counts of `base` and `candidate`
// differ, or if `labels.size()` disagrees with either.
struct TtaFlipReport {
    size_t changed = 0;
    size_t correct_to_incorrect = 0;
    size_t incorrect_to_correct = 0;

    // Convenience only. The paper's point is that this number ALONE hides the
    // regressions, so it is never the only thing a caller should look at.
    long net_change() const {
        return static_cast<long>(incorrect_to_correct) -
               static_cast<long>(correct_to_incorrect);
    }
};

TtaFlipReport tta_flip_report(const Tensor& base, const Tensor& candidate,
                              const std::vector<int>& labels);

// ===========================================================================
// AGGREGATIONS
// ===========================================================================
//
// Four ways to combine a (T, N*K) logit bundle into one (N, K) prediction.
//
//   tta_mean_logits    -- mean of the LOGITS. Returns LOGITS (D2).
//   tta_mean_probs     -- mean of the per-transform softmax probabilities.
//   tta_geo_mean_probs -- mean in LOG space, i.e. the geometric mean of the
//                         probabilities, NORMALISED (D3).
//   tta_majority_vote  -- plurality vote over per-transform argmaxes, returned
//                         as a ONE-HOT (N, K) tensor (D6).
//
// These are NOT interchangeable, and the suite asserts they disagree rather than
// treating any of them as a rescaling of another.
//
// D3, THE NORMALISATION IS LOAD-BEARING. The identity "normalised geometric mean
// of probabilities == softmax of the mean logits" is TRUE, and the obvious
// derivation of it is WRONG. Writing log softmax(z) = z - logsumexp(z) and
// averaging does not yield softmax(mean z), because logsumexp is CONVEX:
// lse(mean z) <= mean(lse(z_t)), strictly unless the members agree. The RAW
// geometric mean is therefore softmax(mean z) times a per-row factor strictly
// BELOW 1 -- a sub-probability vector, not a probability vector. Measured raw
// row sums on the suite fixture: 0.94526606993838813 and 0.90768902185758782,
// versus 1 for `tta_mean_probs`. Omitting the normalisation is caught by a VALUE
// assertion and NOT by an argmax assertion, because a positive per-row scale
// never changes the argmax.
//
// D7, SHIFT INVARIANCE IS ASYMMETRIC ACROSS THE AGGREGATIONS. softmax is
// shift-invariant PER TRANSFORM, so averaging the PROBABILITIES is
// shift-invariant; averaging the LOGITS is NOT, because the offset survives as a
// temperature-like rescale on that member. Measured: adding 7.0 to transform 1
// moves `tta_mean_logits` by 3.500000e+00 and moves `tta_mean_probs` and
// `tta_geo_mean_probs` by 0.000000e+00.

// Mean of the logits over transforms. The result is LOGITS (D2), not a
// probability vector -- callers wanting probabilities want `tta_mean_probs`.
//
// `weights` is optional; when non-empty it must have exactly T entries and is
// NORMALISED INTERNALLY (D8), so {1, 3} and {2, 6} give the same answer.
// Negative weights throw; T == 1 throws, since a single transform is not an
// aggregation and silently returning the input would hide a mis-wired pipeline.
Tensor tta_mean_logits(const Tensor& bundle, size_t num_classes,
                       const std::vector<double>& weights = {});

// Mean of the per-transform softmax probabilities. Output rows sum to 1.
//
// THROWS std::invalid_argument if any member's softmax row is degenerate
// (zero columns), which would make the mean meaningless.
Tensor tta_mean_probs(const Tensor& bundle, size_t num_classes,
                      const std::vector<double>& weights = {});

// Geometric mean of the per-transform probabilities, NORMALISED to sum to 1
// per row (D3). Mathematically equal to `softmax(tta_mean_logits(...))` up to
// floating point -- the suite asserts the two agree to 1e-15.
//
// Numerically this is computed in log space and shifted by each member's own
// logsumexp, so no intermediate is a raw `log(probability)` of a value that may
// have underflowed to zero.
Tensor tta_geo_mean_probs(const Tensor& bundle, size_t num_classes,
                          const std::vector<double>& weights = {});

// Plurality vote over the per-transform argmaxes, returned as a ONE-HOT
// (N, K) tensor (D6) -- not counts, and not class indices.
//
// D5: a tie goes to the LOWEST CLASS INDEX, not to the earliest transform. The
// suite feeds an exact 1-1 split with the transforms in both orders, because a
// suite that only ever sees 3-0 or 2-1 splits cannot tell those rules apart.
Tensor tta_majority_vote(const Tensor& bundle, size_t num_classes);

// Dispatch on the aggregation named. Every enum value must return the same
// thing as the corresponding free function above -- the suite asserts that
// identity for all four, so the dispatcher cannot drift from the cores.
enum class TtaAggregation { MeanLogits, MeanProbs, GeoMeanProbs, MajorityVote };

Tensor tta_aggregate(const Tensor& bundle, size_t num_classes,
                    TtaAggregation how,
                    const std::vector<double>& weights = {});

#endif // NN_UTILS_TTA_H