#ifndef NN_UTILS_NGRAM_LM_H
#define NN_UTILS_NGRAM_LM_H

#include <cstddef>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// NGramLM -- a STATISTICAL (count-based) n-gram language model with three
// smoothing schemes, plus the two classical corpus estimators that consume
// the same count tables.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-22)
//
//   grep -rinE 'ngram|n-gram|backoff|jelinek|stupid.?backoff|good.?turing
//               |heaps|kenlm|kneser|katz|bigram|trigram|unigram' include/ tests/
//   -> 0 HITS, with MATCHING LINES printed (not filenames).
//
// The idea sits on this repo's own expansion queue as a VERIFIED-ABSENT item
// (EXPANSION_QUEUE.md, the 2026-10-20 entry's "VERIFIED ABSENT, re-measure
// before taking" list) and was re-measured on 2026-10-22 with 0 hits.
//
// IT IS NOT REDUNDANT with what the repo already ships:
//   * utils/tokenizer.{h,cpp}      maps text <-> ids. No probabilities.
//   * utils/bpe_tokenizer.{h,cpp}  maps text <-> byte-level BPE ids. No
//                                  probabilities.
//   * utils/logit_processing.{h,cpp} reshapes a NEURAL NETWORK's logits into
//                                  a distribution and decodes from it.
//   * utils/losses.h               cross-entropy against a NEURAL NETWORK's
//                                  logits.
//   * layers/generative/pixelcnn.h  a neural autoregressive model.
// Nothing in the repo produces a STATISTICAL language model, and nothing
// produces a PERPLEXITY for a token sequence. This file is the canonical
// model-free baseline, and it is the natural consumer of the ids that this
// repo's own tokenizers produce.
//
// ---------------------------------------------------------------------------
// CITATIONS. All three are PRE-arXiv, so NO arXiv ID may be asserted for any
// of them. This repo's expansion queue has produced TWELVE wrong arXiv IDs
// and every one of them was a plausible-looking neighbour of the right paper,
// so a bare "arXiv:XXXX.XXXXX" here would be the repo's THIRTEENTH such error.
//
//   * Kneser, R. and Ney, H. (1995). "Three methods for the construction of
//     language models for large scale text retrieval." Eurospeech 1995,
//     pages 125-135. THE canonical smoothed n-gram LM.
//   * Witten, I. H. and Bell, T. C. (1991). "The statistical basis of
//     predictive word association." IEE Proceedings-B, 138(4), pages 375-380.
//   * Chen, S. F. and Goodman, J. (1999). "An empirical study of smoothing
//     techniques for language modeling." COLING-99, pages 291-304. The
//     source of the INTERPOLATED forms used here and of the survey of the
//     variants (absolute discounting, Witten-Bell, Jelinek-Mercer, Katz).
//
// The interpolated forms were cross-checked against the Stanford NLP
// smoothing tutorial (referenced from the Kneser-Ney article) and against
// NLTK's `nltk/probability.py`, a MAINTAINED reference implementation.
//
// ---------------------------------------------------------------------------
// THE PINNED SEMANTICS (the oracle in tests/ngram_oracle.py is the SPEC)
//
// S-1 TOKENIZATION. The LM works on INTEGER TOKEN IDS. That is what makes it
//     compose with this repo's own tokenizers, which emit std::vector<int>.
//
// S-2 NO CROSS-SENTENCE N-GRAMS. The corpus is a list of sentences and an
//     n-gram is counted only if it lies entirely inside ONE sentence.
//     Sentence boundaries are NOT modelled with a BOS symbol: a sentence of
//     length 1 contributes a unigram and nothing else.
//
// S-3 COUNTING. c(g) is the count of the n-gram g; c(h) = sum_w c(h,w) is the
//     count of the CONTEXT h; V_h = |{w : c(h,w) > 0}| is the number of
//     DISTINCT types following h.
//
//     **V_h IS NOT c(h).** 'the' occurs 6 times in the oracle's fixture but
//     is followed by only 4 DISTINCT words. Every smoothing scheme below
//     uses V_h where a naive implementation would use c(h), and the
//     difference is not rounding.
//
// S-4 KNESER-NEY (the INTERPOLATED variant), exactly:
//
//         p_n(w|h) = max(c(h,w) - D, 0) / c(h)
//                  + (D * V_h / c(h)) * p_{n-1}(w | h')
//
//     with the base case the CONTINUATION unigram
//
//         p_1(w) = N1+(w) / sum_{w'} N1+(w')
//
//     where N1+(w) = the number of DISTINCT bigram types (h', w) -- the
//     number of distinct things that can PRECEDE w.
//
//     FOUR properties, each of which is a natural mistake, each pinned:
//       (i)   the unigram is the CONTINUATION count N1+, not the raw count;
//       (ii)  the discount is ABSOLUTE (c - D) and CLAMPED AT ZERO;
//       (iii) the backoff weight is D * V_h / c(h) -- V_h is the number of
//             DISTINCT FOLLOWERS of h, which is NOT c(h);
//       (iv)  an UNSEEN CONTEXT backs off completely (weight 1 on the lower
//             order); it does NOT renormalize against the seen types.
//
//     D is constrained to 0 < D <= 1. That bound is not cosmetic: the
//     normalization proof needs c(h,w) >= D for every type w following h,
//     which holds because counts are integers >= 1.
//
//     **THE VARIANT.** Kneser-Ney has several published variants and they
//     are NOT interchangeable. This file implements the INTERPOLATED one.
//     NLTK implements the NON-INTERPOLATED (recursive, beta-weighted) one.
//     On the oracle's fixture the two differ by 0.2418 on
//     p(the-quick-brown) -- measured, not asserted, in the oracle's S10.
//
// S-5 WITTEN-BELL. Reserved-mass form:
//
//         seen at h:   p(w|h) = c(h,w) / (N_h + T_h)
//         unseen at h: p(w|h) = (T_h / (N_h + T_h)) * p_lower(w|h') / Z_h
//
//     where N_h = c(h), T_h = V_h, p_lower is the (n-1)-gram model, and
//         Z_h = sum over the tokens UNSEEN at h of p_lower(w | h').
//
//     **D-WB: the renormalization by Z_h is an EXPLICIT decision.** Without
//     it the model is not normalized: the reserved mass would be spread
//     over the WHOLE lower-order support rather than over the tokens that
//     are actually unseen at h.
//
//     The UNIGRAM level reserves the mass T/(N+T) for a single
//     OUT-OF-VOCABULARY class, which is exactly NLTK's
//     `WittenBellProbDist` with `bins = B + 1`. The oracle checks that
//     equality to floating point.
//
// S-6 ABSOLUTE DISCOUNTING WITH INTERPOLATION.
//
//         p_n(w|h) = (1 - lambda_h) * c(h,w)/c(h) + lambda_h * p_{n-1}(w|h')
//         lambda_h = D * V_h / c(h)
//
//     with the base case the RAW unigram MLE p_1(w) = c(w)/N.
//
// S-7 THE OOV CLASS. The vocabulary is the set of unigram types seen in
//     training, PLUS one implicit "unseen" class. Witten-Bell assigns it
//     real mass; Kneser-Ney and absolute discounting assign it 0. `prob()`
//     returns 0.0 for a token outside the trained vocabulary under those two
//     schemes, and `sequence_logprob` / `perplexity` return -inf / +inf in
//     that case. Both are documented; neither is silently clamped.
//
// S-8 PERPLEXITY.  ppl = exp( -1/N * sum_i log p(w_i | context_i) ), with
//     contexts built per SENTENCE (S-2) and N = the number of predicted
//     tokens.
//
// S-9 HEAPS' LAW AND GOOD-TURING. Two classical corpus estimators over the
//     same count tables: V(n) = K * n^beta fitted by least squares in log-log
//     space, and the Good-Turing discounted count c*_r = (r+1) N_{r+1}/N_r
//     with unseen mass P_0 = N_1/N.
//
// ===========================================================================

namespace nn {
namespace utils {

// The three smoothing schemes. `kKneserNey` is the default because it is the
// strongest of the three on held-out text for small corpora and the repo's
// queue names it first.
enum class NGramScheme {
    kKneserNey,
    kWittenBell,
    kAbsoluteDiscounting,
};

// The result of a Heaps' law fit.
struct HeapsFit {
    double k = 0.0;     // the growth constant
    double beta = 0.0;  // the exponent, 0 < beta < 1 for a natural language
    double r2 = 0.0;    // the coefficient of determination of the log-log fit
};

// A statistical n-gram language model over INTEGER TOKEN IDS.
//
// Usage:
//     NGramLM lm(/*max_order=*/3, /*discount=*/0.75);
//     lm.train({ {1, 2, 3}, {4, 5} });   // a corpus: one vector per sentence
//     double p = lm.kn_prob(3, /*context=*/{1, 2});
//     double ppl = lm.corpus_perplexity(corpus);
class NGramLM {
public:
    // Construct an untrained model.
    //
    //   max_order -- the highest n-gram order to count and use. Must be >= 1.
    //   discount  -- the absolute-discount value D. Must satisfy
    //                0 < D <= 1; the bound is required by the normalization
    //                proof of the Kneser-Ney form (see S-4).
    //   vocab_size_hint -- reserved for callers that know the vocabulary size
    //                in advance; 0 means "infer it from the corpus". It is
    //                accepted and ignored, so that a future version can use
    //                it without changing the signature.
    //
    // Throws std::invalid_argument if max_order < 1, or if discount is not
    // finite, or if discount <= 0 or discount > 1.
    explicit NGramLM(size_t max_order = 3, double discount = 0.75,
                     size_t vocab_size_hint = 0);

    // Count the corpus. An n-gram is counted only if it lies entirely inside
    // one sentence (S-2). Calling train() a second time REPLACES the count
    // tables; it does not accumulate them.
    void train(const std::vector<std::vector<int>>& corpus);

    // ---- accessors -------------------------------------------------------
    size_t max_order() const { return max_order_; }
    double discount() const { return discount_; }
    // The number of distinct token ids seen in training.
    size_t vocab_size() const { return unigram_.size(); }
    // The total number of tokens counted.
    size_t num_tokens() const { return num_tokens_; }
    // The number of distinct unigram types.
    size_t num_types() const { return unigram_.size(); }
    // The number of distinct bigram / trigram types.
    size_t num_bigrams() const { return bigram_.size(); }
    size_t num_trigrams() const { return trigram_.size(); }

    // c(w) -- the unigram count.
    size_t unigram_count(int token) const;
    // c(h, w) -- the bigram count. Returns 0 for a pair never seen.
    size_t bigram_count(int h, int w) const;
    // c(h1, h2, w) -- the trigram count. Returns 0 for a triple never seen.
    size_t trigram_count(int h1, int h2, int w) const;

    // V_h -- the number of DISTINCT token types that follow the unigram
    // context `h`. THIS IS NOT c(h): 'the' occurs 6 times in the oracle's
    // fixture and is followed by 4 distinct words.
    size_t context_type_count(int h) const;

    // N1+(w) -- the number of DISTINCT contexts that can precede w. This is
    // the CONTINUATION count, and it is the numerator of the Kneser-Ney
    // unigram.
    size_t continuation_count(int w) const;
    // sum_{w'} N1+(w') -- the denominator of the Kneser-Ney unigram.
    size_t continuation_total() const { return continuation_total_; }

    // ---- the three smoothed models ---------------------------------------
    // Each returns p(w | context). The context may be empty (the unigram),
    // one token (the bigram) or two tokens (the trigram). A context longer
    // than max_order - 1 uses the highest available order.
    //
    // A token outside the trained vocabulary returns 0.0 under Kneser-Ney and
    // absolute discounting, and a positive value under Witten-Bell (the OOV
    // class, S-7).
    double kn_prob(int w, const std::vector<int>& context = {}) const;
    double wb_prob(int w, const std::vector<int>& context = {}) const;
    double ad_prob(int w, const std::vector<int>& context = {}) const;

    // Convenience overloads for the fixed context lengths. A bigram
    // context is one token, a trigram context is two. They exist so a
    // caller does not have to build a temporary vector for the common case.
    double kn_prob(int w, int h) const { return kn_prob(w, std::vector<int>{h}); }
    double wb_prob(int w, int h) const { return wb_prob(w, std::vector<int>{h}); }
    double ad_prob(int w, int h) const { return ad_prob(w, std::vector<int>{h}); }
    double kn_prob(int w, int h1, int h2) const {
        return kn_prob(w, std::vector<int>{h1, h2});
    }
    double wb_prob(int w, int h1, int h2) const {
        return wb_prob(w, std::vector<int>{h1, h2});
    }
    double ad_prob(int w, int h1, int h2) const {
        return ad_prob(w, std::vector<int>{h1, h2});
    }

    // The unigram of each scheme, i.e. p_1(w). The Kneser-Ney one uses the
    // CONTINUATION count; the other two use the raw count.
    double kn_unigram(int w) const;
    double wb_unigram(int w) const;
    double ad_unigram(int w) const;

    // ---- sequence scoring -------------------------------------------------
    // The log-probability of a token SEQUENCE under the given scheme. The
    // context of the first token is empty, so the first token is scored by
    // the unigram. Returns -inf if any token has probability 0 (which is
    // what an out-of-vocabulary token does under KN and AD).
    double sequence_logprob(const std::vector<int>& tokens,
                            double discount = -1.0,
                            NGramScheme scheme = NGramScheme::kKneserNey) const;

    // The perplexity of a corpus, exp(-mean log p). Returns +inf if the
    // corpus is empty.
    double corpus_perplexity(const std::vector<std::vector<int>>& corpus,
                              double discount = -1.0,
                              NGramScheme scheme = NGramScheme::kKneserNey) const;

    // ---- Heaps' law -------------------------------------------------------
    // Fit V(n) = K * n^beta by least squares in log-log space, over the
    // corpus growth curve (the number of distinct types after n tokens).
    // With no arguments it uses the training corpus; with explicit
    // (n, V(n)) points it fits those instead.
    HeapsFit heaps_fit() const;
    HeapsFit heaps_fit(const std::vector<double>& n_points,
                        const std::vector<double>& v_points) const;

    // ---- Good-Turing ------------------------------------------------------
    // P_0 = N_1 / N -- the probability mass assigned to types never seen.
    double good_turing_p0() const;
    // The discounted count c*_r = (r+1) * N_{r+1} / N_r. Returns 0.0 when
    // N_{r+1} == 0 (the discounted count is then undefined).
    double good_turing_discounted_count(size_t r) const;
    // The Good-Turing probability of a token with count c, using the
    // discounted count where it is defined.
    double good_turing_probability(int w) const;

private:
    // The count tables.
    std::map<int, size_t> unigram_;                       // c(w)
    std::map<std::pair<int, int>, size_t> bigram_;        // c(h, w)
    std::map<std::pair<int, std::pair<int, int>>, size_t> trigram_;  // c(h1,h2,w)
    // c(h) and V_h for the unigram context h.
    std::map<int, size_t> ctx_count_;
    std::map<int, size_t> ctx_types_;
    // c(h1, h2) and V_{h1,h2} for the trigram context.
    std::map<std::pair<int, int>, size_t> ctx2_count_;
    std::map<std::pair<int, int>, size_t> ctx2_types_;
    // N1+(w) -- the number of distinct contexts preceding w.
    std::map<int, size_t> continuation_;

    size_t max_order_ = 3;
    double discount_ = 0.75;
    size_t num_tokens_ = 0;
    size_t continuation_total_ = 0;

    // The corpus growth curve, recorded during training: the number of
    // distinct types after n tokens. Heaps' law is a statement about THIS
    // curve, and it cannot be reconstructed from the count tables alone --
    // the type counts do not record the order the types were first seen in.
    // It is therefore snapshotted as training proceeds.
    std::vector<double> growth_n_;
    std::vector<double> growth_v_;

    // Internal helpers. `context` is the (possibly empty) context; `order`
    // is the n-gram order being evaluated (1 = unigram).
    double kn_order(size_t order, int w, const std::vector<int>& context) const;
    double wb_order(size_t order, int w, const std::vector<int>& context) const;
    double ad_order(size_t order, int w, const std::vector<int>& context) const;

    // The context key helpers.
    size_t ctx_count_of(const std::vector<int>& context) const;
    size_t ctx_types_of(const std::vector<int>& context) const;
    size_t ngram_count_of(const std::vector<int>& context, int w) const;
};

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_NGRAM_LM_H
