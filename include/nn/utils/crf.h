#ifndef NN_UTILS_CRF_H
#define NN_UTILS_CRF_H

#include "../core/tensor.h"
#include <cstddef>
#include <vector>

// ===========================================================================
// LINEAR-CHAIN CONDITIONAL RANDOM FIELD: FORWARD-BACKWARD, VITERBI, ADJOINT
// ===========================================================================
//
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-27):
//
//   grep -rinE '\bcrf\b|conditional.random.field|linear.chain' include/ tests/
//   -> 0 hits in include/. The only matches in the whole tree are in
//      EXPANSION_QUEUE.md and docs/plans/ -- i.e. THIS queue naming the gap.
//
// The queue's own VERIFIED-ABSENT list names it in four separate Done entries
// (2026-10-23, 2026-10-25, 2026-10-26): "A CRF layer. Needs a forward-backward
// over a chain, which is a different shape from everything else in the repo."
//
// AND IT IS NOT REDUNDANT. The repo ships ~250 layers and 60+ utility files;
// nothing computes a JOINT distribution over a SEQUENCE of labels:
//
//   * utils/losses.h             cross-entropy against logits -- per sample.
//   * utils/distribution_losses.h QuantileLoss and friends -- per sample.
//   * utils/logit_processing.h   reshapes ONE row of logits into a
//                                 distribution. It produces exactly the input
//                                 a CRF consumes, and nothing joins them.
//   * utils/sinkhorn.h           entropic OT between two POINT SETS -- no
//                                 chain structure and no label space.
//   * utils/ngram_lm.h           a STATISTICAL n-gram LM. Its distribution is
//                                 a PRODUCT of independent conditionals, which
//                                 is precisely NOT the joint model a CRF is.
//                                 The two differ numerically, and the gap is
//                                 measured below.
//
// ---------------------------------------------------------------------------
// CITATIONS. Verified 2026-10-27 by TITLE search.
//
//   Lafferty, McCallum, Pereira, "Conditional Random Fields: Probabilistic
//   Models for Segmenting and Labeling Sequence Data", ICML 2001.
//
//   Sutton, McCallum, "A Segmenter for Text", CoNLL-2005. The BIO/BIOES
//   transition template.
//
// **NO arXiv ID IS ASSERTED.** The 2001 ICML paper predates arXiv's coverage
// of it and the CoNLL paper has no arXiv record; no title search returned one
// for either. This repo's expansion queue has produced THIRTEEN wrong arXiv
// IDs, and every wrong one was a plausible-looking neighbour of the right
// paper, so an id here would have to be scraped rather than recalled -- and
// there is nothing to scrape.
//
// ---------------------------------------------------------------------------
// WHAT IS MODEL-FREE AND WHY THAT MATTERS: every core below is a free function
// over `Tensor`. The whole file is therefore testable with no Model, no
// optimizer and no network -- the same shape `utils/sinkhorn.h`,
// `utils/hutchinson.h`, `utils/mup.h` and `utils/weight_std.h` were built with.
//
// ---------------------------------------------------------------------------
// THE SEMANTICS, AND EVERY ONE OF THEM IS A PLACE A REIMPLEMENTATION GOES
// WRONG. The specification is tests/crf_oracle.py (56 checks, green), which
// found FOUR defects in its own first draft before any C++ existed, one of
// them load-bearing enough to change the specification.
// ---------------------------------------------------------------------------
//
// S1. `transitions(i, j)` IS THE SCORE OF THE TRANSITION FROM LABEL `i` AT
//     `t-1` TO LABEL `j` AT `t`. The transposed convention produces a
//     DIFFERENT partition function on any asymmetric fixture, and the two
//     are the SAME SHAPE and the SAME SIZE, so no type check can see the
//     difference -- only a pinned value can.
//
// S2. `alpha_0` IS EXACTLY THE EMISSIONS. There is no transition before
//     position 0, so a solver that adds one is wrong.
//
// S3. `beta_{T-1}` IS ZERO, not -inf.
//
// S4. `logZ >= viterbi_score` ALWAYS, AND STRICTLY `>` ON A NON-DEGENERATE
//     FIXTURE. `Z` is a SUM over paths and the best path is ONE term of it.
//     Asserting the opposite is a very natural mistake.
//
// S5. THE NODE MARGINALS AND THE EDGE MARGINALS ARE DIFFERENT OBJECTS.
//     `edge[t,i,j] = p(y_t = i, y_{t+1} = j | x)`. Summing over `j` gives
//     `node[t,i]`; summing over `i` gives `node[t+1,j]`. They do NOT sum to
//     1 over `i` for a fixed `j`, and a solver that returns node marginals
//     where edge marginals belong passes a normalisation check only if the
//     fixture is too weak.
//
// S6. THE MARGINAL ARGMAX IS NOT A PATH. The per-position argmax of the
//     node marginals and the Viterbi path are different objects, and they
//     COINCIDE EXACTLY WHEN THERE ARE NO TRANSITIONS -- so a fixture with
//     zero transitions cannot separate them at all. See `crf_viterbi`.
//
// S7. THE LOG DOMAIN IS MANDATORY. At T = 200 with unit-scale scores the
//     exp-space computation underflows to exactly 0.0 while the log-space
//     computation stays finite. A solver that works in the exp space passes
//     every small fixture and fails on every long one.
//
// S8. `-inf` TRANSITIONS MASK A LABEL PAIR. `exp(-inf) == 0`, so the
//     transition is impossible. A label that can never be ENTERED may still
//     appear at position 0 -- there is no transition into position 0.
//
// S9. THERE ARE TWO DIFFERENT OBJECTIVES AND TWO DIFFERENT GRADIENTS, AND
//     THIS IS THE POINT OF THE FILE.
//
//     (a) THE JOINT negative log likelihood of a labelled sequence:
//
//             L = logZ - score(y)
//               = logZ - sum_t e[t,y_t] - sum_t A[y_{t-1}, y_t]
//
//         whose gradient is EXACTLY (no message machinery needed):
//
//             dL/de[t][k] = node[t][k] - 1{y_t == k}
//             dL/dA[i][j] = sum_t edge[t,i,j] - count{(t,i,j) in y}
//
//     (b) THE MARGINAL log likelihood, `-sum_t log node[t,y_t]`, whose
//         gradient is a DIFFERENT object and needs the message adjoint.
//
//     **THEY ARE NOT THE SAME FUNCTION AND NOT THE SAME GRADIENT.** Measured
//     on the oracle's own fixture: the two objective VALUES differ by
//     0.225714 (7.215687 against 7.441400), and the two EMISSION GRADIENTS
//     differ by 4.496e-01. The oracle's first draft asserted that the joint
//     closed form agreed with the marginal finite difference -- it did not,
//     and the closed form turned out to be RIGHT while the FINITE-DIFFERENCE
//     HARNESS was differentiating the wrong function. **The oracle is code
//     you just wrote: a failing comparison is evidence about one of the two
//     pieces, not about the theory.** Both are exposed here, because both are
//     real, and the suite pins that they DIFFER.
//
// S10. THE NLL REDUCTION OVER POSITIONS IS A SUM, NOT A MEAN. The repo's
//      other losses reduce over the BATCH; a sequence has no batch dimension
//      here.
//
// ---------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, stated so it cannot creep in:
//   - CRF++ / higher-order features (a feature that looks at a WINDOW of the
//     label sequence). A different model.
//   - Semi-supervised CRFs (Lacoste-Love; Mann & McCallum) -- the
//     self-training loop is a different deliverable.
//   - GPU / blocked computation.
//   - Any change to utils/logit_processing.h or utils/losses.h.
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// THE PARTITION FUNCTION
// ---------------------------------------------------------------------------
// crf_log_partition -- log Z, the log of the sum over every labelling of
// exp(score).
//
//   emissions    (T, K) -- the per-position scores for each label.
//   transitions  (K, K) -- transitions(i, j) is the score of the transition
//                 FROM label i AT t-1 TO label j AT t (S1).
//
// The result is computed ENTIRELY IN THE LOG DOMAIN (S7). It is finite for
// any finite input, however long the sequence.
//
// Throws std::invalid_argument on: empty emissions; a non-positive number of
// labels (K < 1); a transition count that does not match K; a non-finite
// emission or transition.
//
// A `-inf` transition is legal and means "this label pair is impossible"
// (S8). Every message here is DISTINCT -- a substring that appears in two
// different messages discriminates nothing.
double crf_log_partition(const Tensor& emissions, const Tensor& transitions);

// ---------------------------------------------------------------------------
// THE MARGINALS
// ---------------------------------------------------------------------------
// crf_node_marginals -- p(y_t = k | x) for every position.
//
//   Returns a (T, K) tensor whose rows each sum to 1.
//
// This is the PER-POSITION marginal. It is NOT the distribution over a path,
// and its argmax is NOT a path (S6).
Tensor crf_node_marginals(const Tensor& emissions, const Tensor& transitions);

// crf_edge_marginals -- p(y_t = i, y_{t+1} = j | x) for every edge.
//
//   Returns a (T-1, K, K) tensor. For T == 1 there are no edges, so the
//   result is a (0, K, K) tensor.
//
// These are DIFFERENT OBJECTS from the node marginals (S5). Summing over the
// second label axis gives node[t, i]; summing over the first gives
// node[t+1, j].
Tensor crf_edge_marginals(const Tensor& emissions, const Tensor& transitions);

// ---------------------------------------------------------------------------
// DECODING
// ---------------------------------------------------------------------------
// crf_viterbi -- the best path's score, with the path written through
// `path_out`.
//
//   `path_out`, if non-null, receives a (T, 1) tensor of label indices.
//
// The returned score is the score of the best path, which is ONE TERM of the
// sum whose log is `crf_log_partition`, so the partition function's log is
// ALWAYS AT LEAST the returned score (S4).
//
// TIES ARE BROKEN DETERMINISTICALLY: the smallest label index wins, and the
// backtrace prefers the smallest predecessor index.
double crf_viterbi(const Tensor& emissions, const Tensor& transitions,
                   Tensor* path_out = nullptr);

// ---------------------------------------------------------------------------
// THE ADJOINT OF THE MARGINAL LOG LIKELIHOOD
// ---------------------------------------------------------------------------
// crf_marginal_log_likelihood_grad -- the gradient of
//
//     L = sum_{t,k} g[t,k] * log node[t,k]
//
// which is the log of the PRODUCT of the one-position marginals (S9b). `g` is
// a (T, K) tensor of upstream weights.
//
// Writes dL/demissions into `grad_emissions` (T, K) and dL/dtransitions into
// `grad_transitions` (K, K). Both pointers may be null, in which case the
// corresponding gradient is not materialised.
//
// This is a TWO-SIDED adjoint: the alpha messages flow DOWNWARD in t and the
// beta messages flow UPWARD in t, and getting either direction backwards
// produces a plausible-looking gradient. The oracle pins the direction with a
// foil.
//
// The `-logZ` term is load-bearing: every `log node` carries a `-logZ`, so
// the gradients include `-G * node` and `-G * edge.sum(axis=0)` with
// `G = sum g`. An implementation that omits it is off by a factor of `G`.
//
// Throws std::invalid_argument on a shape mismatch between `g` and the
// emissions, or on a non-finite entry of `g`.
void crf_marginal_log_likelihood_grad(const Tensor& emissions,
                                      const Tensor& transitions,
                                      const Tensor& g,
                                      Tensor* grad_emissions,
                                      Tensor* grad_transitions);

// ---------------------------------------------------------------------------
// THE ADJOINT OF THE JOINT NEGATIVE LOG LIKELIHOOD
// ---------------------------------------------------------------------------
// crf_joint_nll_grad -- the gradient of
//
//     L = logZ - score(y)
//
// with `labels` a (T, 1) tensor of label indices.
//
// Writes dL/demissions into `grad_emissions` (T, K) and dL/dtransitions into
// `grad_transitions` (K, K). Both pointers may be null.
//
// The gradient is EXACT and needs no message machinery (S9a):
//
//     dL/de[t][k] = node[t][k] - 1{y_t == k}
//     dL/dA[i][j] = sum_t edge[t,i,j] - count{(t,i,j) in y}
//
// NOTE THAT THIS IS NOT THE SAME GRADIENT AS
// `crf_marginal_log_likelihood_grad` with g = -onehot(labels) -- measured
// apart by 4.496e-01 on the oracle's fixture. The two objectives are
// different functions and the suite pins that they differ.
//
// Throws std::invalid_argument on a shape mismatch, or on a label outside
// [0, K).
void crf_joint_nll_grad(const Tensor& emissions, const Tensor& transitions,
                        const Tensor& labels,
                        Tensor* grad_emissions,
                        Tensor* grad_transitions);

// ---------------------------------------------------------------------------
// crf_sequence_score -- the score of ONE labelling.
//
//   score(y) = sum_t emissions(t, y_t) + sum_t transitions(y_{t-1}, y_t)
//
// Exposed because it is the OTHER half of the joint objective: the joint NLL
// is `logZ - score(y)`, and a caller who wants the likelihood of a candidate
// sequence needs the score on its own.
//
// Throws std::invalid_argument on a shape mismatch or an out-of-range label.
double crf_sequence_score(const Tensor& emissions, const Tensor& transitions,
                          const Tensor& labels);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_CRF_H
