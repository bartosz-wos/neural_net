#ifndef NN_UTILS_LOGIT_PROCESSING_H
#define NN_UTILS_LOGIT_PROCESSING_H

#include "../core/tensor.h"
#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// LOGIT PROCESSING AND DECODING STRATEGIES
// ============================================================================
//
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-20):
//
//   grep -rinE 'nucleus|top.?p\b|top_p_sampl|temperature_sampl|typical.?sampl
//               |min.?p\b|repetition.?penal|beam.?search' include/
//   -> 0 HITS in the whole tree.
//
// And a wider sweep for the concept under other names found only FALSES:
//   grep -rinE 'temperature|top_k|greedy|beam|sampling' include/nn/layers/
//               generative/*.h include/nn/core/*.h
//   -> 11 hits, every one a substring or a different concept: `up_sample_` /
//      `down_sample_` conv members in ddpm.h, `conv_transpose` in consistency.h,
//      "inverse pass (for sampling)" in coupling_layer.h, "greedy OT
//      assignment" in flow_matching.h (the min-cost-assignment coupling), and
//      an rng_ for the WGAN gradient penalty.
//
// IT IS NOT REDUNDANT. The queue's standing rule is that an absence proves a
// feature is missing, not that it belongs -- so the redundancy question was
// measured rather than argued. `layers/generative/pixelcnn.h:203-205` says, in
// the repo's own words:
//
//     "Per-pixel softmax: applied across the n_vals dimension inside each (i,j)
//      position. For training with cross-entropy we expose logits and let the
//      caller apply softmax. The output is the raw logits."
//
// PixelCNN models a CATEGORICAL DISTRIBUTION over `n_vals` values and the
// repository ships NO code that turns those logits into a token: `grep -n
// 'sample\|generate' pixelcnn.cpp` returns nothing. VQ-VAE's codebook is a
// second discrete-output model with the same missing decoder. So the DISCRETE
// half of generative/ has no inference path at all, and this file is it.
//
// WHAT IS MODEL-FREE AND WHY THAT MATTERS: every processor is a free function
// over `Tensor`. The whole file is therefore testable with no Model, no
// optimizer and no network -- the same shape utils/hutchinson.h, utils/mup.h,
// utils/sinkhorn.h and utils/ortho_penalty.h were built with. The one place a
// model appears is `ScoreFn`, a caller-supplied callback, which is what keeps
// `BeamSearchDecoder` and `autoregressive_sample` agnostic to WHICH model they
// are driving and lets the tests supply a hand-written oracle.
//
// -----------------------------------------------------------------------------
// CITATIONS. Verified BY TITLE on 2026-10-20, never typed from memory -- this
// repo's expansion queue has produced nine wrong arXiv IDs.
//   - Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text
//     Degeneration", ICLR 2020, arXiv:1904.09751. Nucleus (top-p) sampling,
//     min-p sampling, and the "unreliable tail" motivation. Title search
//     returned the result set titled `[1904.09751] The Curious Case of Neural
//     Text Degeneration`; the abstract snippet matches the paper's Section 3.1.
//   - Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation", ACL 2018,
//     arXiv:1805.04833. Top-k sampling. Title search returned
//     `[1805.04833] Hierarchical Neural Story Generation`; the fairseq README
//     for that paper documents `--sampling-topk`, which is the same knob.
//   - Beam search length penalty: the GNMT formulation `score / len^alpha`.
//   - Repetition / frequency / presence penalties: the CTRL-era mechanisms.
//     No arXiv ID is asserted for these -- they are folklore by mechanism name,
//     and asserting an unverified ID is the failure mode this file avoids.
// The arXiv EXPORT API is unreachable from the development host (`curl` returned
// an empty body with exit 0 for a control query that must return results), so
// both IDs come from a title search, which is the fallback the queue file
// itself prescribes.
// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// THE SEMANTICS, AND WHY EACH ONE IS PINNED BY A TEST
// -----------------------------------------------------------------------------
//
// S1. NUCLEUS `p` IS A CUMULATIVE-MASS CUT AND `>=` IS LOAD-BEARING. Sort
//     descending, accumulate, keep the smallest prefix whose mass is `>= p`.
//     The token that REACHES the threshold is kept and the loop stops. On
//     p = [0.5, 0.3, 0.15, 0.05] the cumulative masses are 0.5, 0.8, 0.95, 1.0,
//     and `p = 0.8` is the ONLY value in {0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}
//     that separates `cum >= p` from `cum > p`: `>=` gives {0,1}, `>` gives
//     {0,1,2}. That is the single most common hand-rolled-nucleus bug.
//
// S2. `p >= 1.0` IS A NO-OP, NOT AN ERROR. Mass reaches 1.0 only after all V
//     tokens, so the loop must terminate without running off the end.
//
// S3/S13. TOP-K KEEPS `>= k` AND `k >= V` IS A NO-OP. Not `>`; `k == V` must not
//     -inf the argmax, and a V=1 vocabulary must be legal.
//
// S4. MIN-P IS A RELATIVE FLOOR AGAINST THE MAXIMUM, NOT THE MEDIAN. Holtzman's
//     formulation keeps tokens with `p_i >= min_p * p_max`. A median- or
//     mean-relative threshold is a DIFFERENT ALGORITHM that happens to look
//     plausible: on p = [0.70, 0.15, 0.10, 0.05] at min_p = 0.1 the two give
//     {0,1,2} and {0,1,2,3} respectively.
//
// S5. THE REPETITION PENALTY SIGN-SPLITS ON NEGATIVE LOGITS. For penalty > 1:
//         logit < 0  ->  logit * penalty     (more negative => suppressed)
//         logit >= 0 ->  logit / penalty
//     An implementation that always multiplies SUPPRESSES positive tokens and
//     PROMOTES negative ones -- it inverts the penalty on half the vocabulary.
//     Measured on [-2.0, 4.0] with history [0,1] at penalty 2.0: correct is
//     [-4.0, 2.0]; always-multiply gives [-4.0, 8.0].
//
// S6. FREQUENCY IS LINEAR IN THE COUNT, PRESENCE IS A STEP.
//     `logit -= penalty * count` vs `logit -= penalty * (count > 0)`, both from
//     the history MULTISET. Neither is clipped at -count.
//
// S7. TEMPERATURE ACTS ON LOGITS, NOT PROBABILITIES. `logits / T`. Dividing the
//     softmax OUTPUT by T is a different function that does not even sum to 1:
//     on logits [2,1,0] at T=0.5, `softmax(logits/T)[0] = 0.8668133321973347`
//     while `softmax(probs/T)[0] = 1.3304819115496436`.
//
// S8. CHAIN ORDER IS PART OF THE CONTRACT -- BUT ONLY *TRUNCATION-ON-MASS*
//     PROCESSORS CAN PIN IT. Dividing by T > 0 is monotone, so top-k's survivor
//     SET is order-independent BY CONSTRUCTION and cannot witness the order.
//     Top-p can, because its cut depends on cumulative mass and temperature
//     changes the mass: measured on log([0.5,0.3,0.15,0.05]) at T=0.5 and
//     p=0.9, temperature-then-p keeps {0,1} while p-then-temperature keeps
//     {0,1,2}. Documented order: temperature first, then truncations, then the
//     penalties.
//
// S9. `-inf` MUST SURVIVE THE PIPELINE. A truncated token is `-inf`; temperature
//     divides it to `-inf` (fine), but a zeroing or add-with-NaN implementation
//     turns it into NaN, and a softmax without max-subtraction turns `inf/inf`
//     into NaN. Every processor here writes `-inf` and nothing else.
//
// S10. THE MAX-SUBTRACT IN `softmax` AND `sample_from_logits` IS LOAD-BEARING.
//     `exp(1000)` is `inf` and `inf/inf` is NaN. On logits [1000, 999] the
//     correct answer is [0.7310585786300049, 0.2689414213699951].
//
// S11. `apply_round_to_int` USES `std::round` = HALF AWAY FROM ZERO, so
//     round(-2.5) = -3 and round(0.5) = 1. NOT banker's rounding
//     (`std::nearbyint`) and NOT numpy's `np.round`, which is ties-to-EVEN.
//     That is worth stating because an oracle written in the wrong language with
//     the wrong rounding mode ships a C++ test whose every value is wrong.
//
// S12. TIE-BREAK IS FIRST-WINS ON THE ORIGINAL INDEX, in top-k, in top-p and in
//     greedy argmax. Deterministic regardless of sort implementation.
//
// S13. A VOCAB-SIZE-1 DISTRIBUTION IS LEGAL throughout.
//
// -----------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, stated so it cannot creep in:
//   - Speculative / lookahead / contrastive decoding.
//   - min-p applied to a live beam during beam search (a research variant, not
//     the paper's algorithm).
//   - Any change to layers/generative/pixelcnn.{h,cpp} or vq_vae.{h,cpp}. The
//     model adapter lives in the TEST, not in the library.
//   - Batched (2-D) processing. Every primitive is a single-row function on
//     purpose; `apply_row` drives a batch from the outside.
// ============================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// softmax -- max-subtracting. The subtraction is what keeps [1000, 999] finite
// (S10). Throws on an empty row rather than returning a zero-sum.
// ---------------------------------------------------------------------------
Tensor softmax(const Tensor& logits);

// ---------------------------------------------------------------------------
// Truncating processors. Each returns a NEW tensor of the same shape with the
// disallowed entries set to -inf. The INPUT IS NEVER MUTATED -- a processor
// that mutated its argument would make the chain non-reentrant and would make
// it impossible to compare orders (S8).
//
// A row may be (1, V) or (V, 1); the shape is preserved.
// ---------------------------------------------------------------------------

// Temperature on LOGITS (S7). T == 1 is the identity. T <= 0 throws.
Tensor apply_temperature(const Tensor& logits, double temperature);

// Top-k: keep the k largest logits (S3, S12). k >= V is a no-op.
Tensor apply_top_k(const Tensor& logits, size_t k);

// Top-p / nucleus: keep the smallest descending prefix with cumulative mass
// >= p (S1, S2, S12). p >= 1 is a no-op.
Tensor apply_top_p(const Tensor& logits, double p);

// Min-p: keep tokens with `p_i >= min_p * p_max` (S4). `temperature` is applied
// to the logits FIRST, so the floor is relative to the TEMPERATURE-adjusted
// maximum. min_p == 0 is a no-op.
Tensor apply_min_p(const Tensor& logits, double min_p, double temperature = 1.0);

// Typical-p / locally typical sampling (S15..S21). Keeps the tokens whose
// NEGATIVE LOG-PROBABILITY is closest to the model's own CONDITIONAL ENTROPY,
// greedily until their cumulative probability mass reaches `tau`.
//
// The ordering key is `| -log p_i - H |` in nats, where `H = -sum p log p` is
// the entropy of the (temperature-adjusted) distribution, and the tokens are
// ordered ASCENDING by that distance -- the one CLOSEST to the entropy first.
// The cut is the smallest such prefix whose cumulative mass reaches `tau`
// (S17).
//
// THIS IS NOT A MASS CUT LIKE TOP-P. Top-p keeps the most PROBABLE tokens;
// typical-p keeps the tokens whose INFORMATION CONTENT is closest to the
// model's EXPECTED information content, so the argmax is not privileged and
// can be excluded. On the pinned fixture (logits [0, -1, 0.5, -2]) the argmax
// is token 2 while the token closest to the entropy is token 0, and the two
// cuts keep different sets.
//
// `temperature` is applied to the LOGITS FIRST, so the entropy is that of the
// temperature-adjusted distribution.
//
// tau is a probability mass in [0, 1]. tau == 1 keeps everything (a no-op);
// tau == 0 keeps EXACTLY ONE token -- the closest to the entropy -- because
// the greedy loop takes a token before it tests the threshold. tau outside
// [0, 1] throws.
Tensor apply_typical_p(const Tensor& logits, double tau, double temperature = 1.0);

// ---------------------------------------------------------------------------
// Penalising processors. `history` is the multiset of previously emitted token
// ids. The three differ ONLY in how they weight the count (S5, S6):
//
//   repetition : sign-splitting multiplicative penalty, applied ONCE per token
//   frequency  : -penalty * count            (linear, NOT clipped)
//   presence   : -penalty * (count > 0)      (a step function)
//
// All three leave tokens absent from `history` untouched.
// ---------------------------------------------------------------------------
Tensor apply_repetition_penalty(const Tensor& logits,
                                const std::vector<int>& history, double penalty);
Tensor apply_frequency_penalty(const Tensor& logits,
                               const std::vector<int>& history, double penalty);
Tensor apply_presence_penalty(const Tensor& logits,
                              const std::vector<int>& history, double penalty);

// ---------------------------------------------------------------------------
// Output projectors. These do NOT truncate; they write a transformed value.
// ---------------------------------------------------------------------------

// Zero (i.e. -inf) any logit strictly below `epsilon` in PROBABILITY space.
Tensor apply_epsilon(const Tensor& logits, double epsilon);

// std::round: half AWAY FROM ZERO on both signs (S11). Not nearbyint.
Tensor apply_round_to_int(const Tensor& logits);

// ---------------------------------------------------------------------------
// Sampling. This is the ONLY place a categorical draw happens.
//
// GREEDY returns the argmax, ties broken first-wins (S12), and does NOT consult
// the rng -- so it is bit-reproducible regardless of the seed. MULTINOMIAL
// draws from the renormalised softmax. Both max-subtract first (S10).
// ---------------------------------------------------------------------------
enum class SamplingStrategy { GREEDY, MULTINOMIAL };

int sample_from_logits(const Tensor& logits, SamplingStrategy strategy,
                       std::mt19937& rng);

// ---------------------------------------------------------------------------
// LogitProcessorChain -- applies its processors LEFT TO RIGHT, which is the
// documented order (S8): temperature, then truncations, then penalties.
// Return-type-chaining builders, so a chain reads in the order it applies.
// ---------------------------------------------------------------------------
class LogitProcessorChain {
public:
    LogitProcessorChain& add_temperature(double temperature);
    LogitProcessorChain& add_top_k(size_t k);
    LogitProcessorChain& add_top_p(double p);
    LogitProcessorChain& add_min_p(double min_p, double temperature = 1.0);
    LogitProcessorChain& add_typical_p(double tau, double temperature = 1.0);
    LogitProcessorChain& add_repetition_penalty(const std::vector<int>& history,
                                                double penalty);
    LogitProcessorChain& add_frequency_penalty(const std::vector<int>& history,
                                               double penalty);
    LogitProcessorChain& add_presence_penalty(const std::vector<int>& history,
                                              double penalty);
    LogitProcessorChain& add_epsilon(double epsilon);

    Tensor apply(const Tensor& logits) const;
    size_t size() const { return processors_.size(); }
    void clear() { processors_.clear(); }

private:
    std::vector<std::function<Tensor(const Tensor&)>> processors_;
};

// ---------------------------------------------------------------------------
// Beam search.
//
// ScoreFn is the boundary that keeps this model-agnostic: the caller supplies
// `logits(batch_ids) -> (B, V) LOGITS` and the decoder never learns what
// generated them. That is what lets the tests drive it from a hand-written
// oracle model AND lets it drive a real PixelCNN, with no decoder change.
//
// LENGTH PENALTY IS APPLIED ONLY WHEN RANKING FINISHED HYPOTHESES, never
// during live beam expansion. Penalising a live beam for being short biases
// the whole search toward short outputs and is the classic bug here; the
// expansion compares hypotheses at equal length, so the penalty cancels.
//
// eod IS NOT eos. The eos id is an explicit constructor argument; any other id
// (including a caller's -1 "end of data" sentinel) is an ordinary token and
// does not terminate the search. This matches PyTorch's EncoderDecoderModel,
// which does not treat eod as eos.
// ---------------------------------------------------------------------------
struct BeamHypothesis {
    std::vector<int> tokens;
    double logprob = 0.0;
    bool finished = false;
};

struct BeamSearchConfig {
    size_t beam_width = 3;
    // GNMT: score / len^length_penalty, applied only to FINISHED hypotheses.
    double length_penalty = 0.0;
    // Stop as soon as beam_width hypotheses have finished.
    bool early_stopping = true;
    size_t max_length = 64;
    // -1 means "never terminates early".
    int eos_id = -1;
};

class BeamSearchDecoder {
public:
    using ScoreFn = std::function<Tensor(const Tensor& batch_ids)>;

    // Throws std::invalid_argument on vocab == 0 or beam_width == 0.
    BeamSearchDecoder(size_t vocab, BeamSearchConfig cfg, ScoreFn fn);

    // Returns (num_return, L), padded with eos_id. `start_ids` is (1, S).
    Tensor decode(const Tensor& start_ids, size_t num_return);

    // The ranked finished hypotheses, for tests and for callers that need the
    // scores rather than just the tokens.
    const std::vector<BeamHypothesis>& hypotheses() const { return finished_; }
    size_t vocab() const { return vocab_; }
    const BeamSearchConfig& config() const { return cfg_; }

private:
    size_t vocab_;
    BeamSearchConfig cfg_;
    ScoreFn score_fn_;
    std::vector<BeamHypothesis> finished_;
};

// ---------------------------------------------------------------------------
// The autoregressive loop -- the thing this whole file exists to make possible.
//
// `steps` is the number of tokens to emit, NOT a length the caller has to
// reconcile with the model's own sequence length; PixelCNN's canvas loop drives
// this with steps == H*W.
//
// `stop_on_eos` with `eos_id` breaks out of the loop the moment eos is drawn.
// ---------------------------------------------------------------------------
struct AutoregressiveResult {
    Tensor tokens;
    bool stopped_early = false;
};

AutoregressiveResult autoregressive_sample(
    const BeamSearchDecoder::ScoreFn& score_fn, const Tensor& start_ids,
    size_t steps, const LogitProcessorChain& chain, SamplingStrategy strategy,
    std::mt19937& rng, int eos_id = -1, bool stop_on_eos = false);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_LOGIT_PROCESSING_H