#include "ngram_lm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace nn {
namespace utils {
namespace {

// A context key for the trigram tables: the two most recent tokens.
using Ctx2 = std::pair<int, int>;

// THE OUT-OF-VOCABULARY CLASS.
//
// This is a SINGLE, FIXED sentinel, and it is deliberately OUTSIDE the range
// of any plausible token id (a caller passing 10000 for an unknown token
// must get the OOV class's probability, not "token 10000 does not exist").
//
// It is a class, not a token: every out-of-vocabulary id maps to it, so the
// Witten-Bell distribution is over (vocabulary + ONE OOV class) and sums to
// one. A per-id OOV would make the reserved mass depend on which unknown id
// was asked about.
constexpr int kOovClass = -1;

}  // namespace

// ---------------------------------------------------------------------------
// Construction and validation
// ---------------------------------------------------------------------------
NGramLM::NGramLM(size_t max_order, double discount, size_t vocab_size_hint)
    : max_order_(max_order), discount_(discount) {
    // The bound 0 < D <= 1 is required by the Kneser-Ney normalization
    // proof: the discounted term max(c(h,w) - D, 0) is nonzero for every
    // type w following h only if D <= 1, because counts are integers >= 1.
    if (max_order < 1) {
        throw std::invalid_argument("ngram: max_order must be >= 1");
    }
    if (!std::isfinite(discount) || discount <= 0.0 || discount > 1.0) {
        throw std::invalid_argument("ngram: discount must be finite and in (0, 1]");
    }
    // vocab_size_hint is accepted and ignored: the vocabulary is inferred
    // from the corpus. It exists so that a future version can use it without
    // changing this signature.
    (void)vocab_size_hint;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
void NGramLM::train(const std::vector<std::vector<int>>& corpus) {
    // Retraining REPLACES the tables rather than accumulating into them.
    unigram_.clear();
    bigram_.clear();
    trigram_.clear();
    ctx_count_.clear();
    ctx_types_.clear();
    ctx2_count_.clear();
    ctx2_types_.clear();
    continuation_.clear();
    num_tokens_ = 0;
    continuation_total_ = 0;
    growth_n_.clear();
    growth_v_.clear();

    // The set of distinct followers per context, accumulated locally and
    // turned into a count at the end. A std::set is used because V_h is the
    // number of DISTINCT followers, and a multiset would silently turn the
    // distinct-follower count into the token count -- the exact confusion
    // S3 warns about.
    std::map<int, std::set<int>> followers;
    std::map<Ctx2, std::set<int>> followers2;
    // The set of distinct predecessors per token, for the continuation count.
    std::map<int, std::set<int>> predecessors;

    for (const std::vector<int>& sentence : corpus) {
        for (size_t i = 0; i < sentence.size(); ++i) {
            const int w = sentence[i];
            ++unigram_[w];
            ++num_tokens_;

            // The corpus growth curve: the number of distinct types after n
            // tokens. Heaps' law is fitted to THIS, so it is recorded here
            // rather than reconstructed afterwards -- the count tables do not
            // record the order in which types were first seen.
            //
            // A POINT PER TOKEN, not one per new type. The growth curve is
            // V(n) for EVERY n, including the n's where no new type appears
            // (the curve has plateaus). Recording only the new types would
            // drop the plateaus and fit a DIFFERENT curve -- one with a
            // strictly increasing V -- which is a visibly different object.
            growth_n_.push_back(static_cast<double>(num_tokens_));
            growth_v_.push_back(static_cast<double>(unigram_.size()));

            if (i >= 1) {
                const int h = sentence[i - 1];
                ++bigram_[std::make_pair(h, w)];
                ++ctx_count_[h];
                followers[h].insert(w);
                predecessors[w].insert(h);
            }
            if (i >= 2) {
                const Ctx2 h(sentence[i - 2], sentence[i - 1]);
                ++trigram_[std::make_pair(h.first, std::make_pair(h.second, w))];
                ++ctx2_count_[h];
                followers2[h].insert(w);
            }
        }
    }

    for (const auto& kv : followers) ctx_types_[kv.first] = kv.second.size();
    for (const auto& kv : followers2) ctx2_types_[kv.first] = kv.second.size();
    for (const auto& kv : predecessors) {
        continuation_[kv.first] = kv.second.size();
        continuation_total_ += kv.second.size();
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
size_t NGramLM::unigram_count(int token) const {
    auto it = unigram_.find(token);
    return it == unigram_.end() ? 0 : it->second;
}

size_t NGramLM::bigram_count(int h, int w) const {
    auto it = bigram_.find(std::make_pair(h, w));
    return it == bigram_.end() ? 0 : it->second;
}

size_t NGramLM::trigram_count(int h1, int h2, int w) const {
    auto it = trigram_.find(std::make_pair(h1, std::make_pair(h2, w)));
    return it == trigram_.end() ? 0 : it->second;
}

size_t NGramLM::context_type_count(int h) const {
    auto it = ctx_types_.find(h);
    return it == ctx_types_.end() ? 0 : it->second;
}

size_t NGramLM::continuation_count(int w) const {
    auto it = continuation_.find(w);
    return it == continuation_.end() ? 0 : it->second;
}

// ---------------------------------------------------------------------------
// Internal context helpers
// ---------------------------------------------------------------------------
size_t NGramLM::ctx_count_of(const std::vector<int>& context) const {
    if (context.empty()) return 0;
    if (context.size() == 1) {
        auto it = ctx_count_.find(context[0]);
        return it == ctx_count_.end() ? 0 : it->second;
    }
    // The LAST TWO tokens form the trigram context. A longer context is
    // truncated, which is the standard "use the highest available order"
    // rule.
    const Ctx2 h(context[context.size() - 2], context[context.size() - 1]);
    auto it = ctx2_count_.find(h);
    return it == ctx2_count_.end() ? 0 : it->second;
}

size_t NGramLM::ctx_types_of(const std::vector<int>& context) const {
    if (context.empty()) return 0;
    if (context.size() == 1) {
        auto it = ctx_types_.find(context[0]);
        return it == ctx_types_.end() ? 0 : it->second;
    }
    const Ctx2 h(context[context.size() - 2], context[context.size() - 1]);
    auto it = ctx2_types_.find(h);
    return it == ctx2_types_.end() ? 0 : it->second;
}

size_t NGramLM::ngram_count_of(const std::vector<int>& context, int w) const {
    if (context.empty()) return unigram_count(w);
    if (context.size() == 1) return bigram_count(context[0], w);
    const int h1 = context[context.size() - 2];
    const int h2 = context[context.size() - 1];
    return trigram_count(h1, h2, w);
}

// ---------------------------------------------------------------------------
// The three unigrams
// ---------------------------------------------------------------------------
double NGramLM::kn_unigram(int w) const {
    // The CONTINUATION count: the number of DISTINCT contexts that can
    // precede w. A token that never occurs in the corpus has a continuation
    // count of 0 and therefore probability 0.
    if (continuation_total_ == 0) return 0.0;
    return static_cast<double>(continuation_count(w)) /
           static_cast<double>(continuation_total_);
}

double NGramLM::wb_unigram(int w) const {
    // Witten-Bell reserves the mass T/(N+T) for the out-of-vocabulary class,
    // where T is the number of observed types and N the token count. A token
    // with count 0 IS the OOV class and gets that reserved mass.
    //
    // EVERY out-of-vocabulary id maps to the single OOV class, so the
    // probability of an unknown token does not depend on WHICH unknown id
    // was asked about.
    const double n = static_cast<double>(num_tokens_);
    const double t = static_cast<double>(unigram_.size());
    if (n + t == 0.0) return 0.0;
    const double c = static_cast<double>(unigram_count(w));
    if (c > 0.0) return c / (n + t);
    return t / (n + t);
}

double NGramLM::ad_unigram(int w) const {
    // The RAW maximum-likelihood estimate. A token with count 0 gets 0.
    if (num_tokens_ == 0) return 0.0;
    return static_cast<double>(unigram_count(w)) /
           static_cast<double>(num_tokens_);
}

// ---------------------------------------------------------------------------
// KNESER-NEY
// ---------------------------------------------------------------------------
double NGramLM::kn_order(size_t order, int w,
                          const std::vector<int>& context) const {
    if (order <= 1) return kn_unigram(w);

    const size_t c_h = ctx_count_of(context);
    if (c_h == 0) {
        // (iv) An UNSEEN CONTEXT backs off COMPLETELY: the whole probability
        // comes from the lower order. It does NOT renormalize against the
        // seen types.
        // THE BACKOFF KEEPS THE SUFFIX of the context and drops its FIRST
        // token: h = (h1, h2) -> h' = (h2). The context of a language model
        // is a SUFFIX of the history, so dropping the OLDEST token is what
        // "the (n-1)-gram model" means. Taking the PREFIX instead (the
        // newest token) is a different model -- and for the fixture's
        // context (the,cat) it evaluates (cat) instead of (the), which
        // changes the answer.
        std::vector<int> shorter(context.begin() + 1, context.end());
        if (shorter.empty()) return kn_unigram(w);
        return kn_order(order - 1, w, shorter);
    }

    const double c_hw = static_cast<double>(ngram_count_of(context, w));
    const double v_h = static_cast<double>(ctx_types_of(context));
    const double d = discount_;
    const double ch = static_cast<double>(c_h);

    // (ii) The discount is ABSOLUTE and the term is CLAMPED AT ZERO.
    const double first = std::max(c_hw - d, 0.0) / ch;
    // (iii) The backoff weight uses V_h, the number of DISTINCT FOLLOWERS.
    const double lambda = d * v_h / ch;

    // The backoff keeps the SUFFIX of the context (see the note in
    // kn_order): h = (h1, h2) -> h' = (h2).
    std::vector<int> shorter(context.begin() + 1, context.end());
    const double lower = shorter.empty() ? kn_unigram(w)
                                         : kn_order(order - 1, w, shorter);
    return first + lambda * lower;
}

double NGramLM::kn_prob(int w, const std::vector<int>& context) const {
    return kn_order(context.size() + 1, w, context);
}

// ---------------------------------------------------------------------------
// WITTEN-BELL
// ---------------------------------------------------------------------------
double NGramLM::wb_order(size_t order, int w,
                          const std::vector<int>& context) const {
    if (order <= 1) return wb_unigram(w);

    const size_t c_h = ctx_count_of(context);
    if (c_h == 0) {
        // THE BACKOFF KEEPS THE SUFFIX of the context and drops its FIRST
        // token: h = (h1, h2) -> h' = (h2). The context of a language model
        // is a SUFFIX of the history, so dropping the OLDEST token is what
        // "the (n-1)-gram model" means. Taking the PREFIX instead (the
        // newest token) is a different model -- and for the fixture's
        // context (the,cat) it evaluates (cat) instead of (the), which
        // changes the answer.
        std::vector<int> shorter(context.begin() + 1, context.end());
        if (shorter.empty()) return wb_unigram(w);
        return wb_order(order - 1, w, shorter);
    }

    const double t_h = static_cast<double>(ctx_types_of(context));
    const double n_h = static_cast<double>(c_h);
    const double c_hw = static_cast<double>(ngram_count_of(context, w));

    if (c_hw > 0.0) {
        // A SEEN type at h gets c(h,w) / (N_h + T_h).
        return c_hw / (n_h + t_h);
    }

    // An UNSEEN type at h gets the reserved mass, distributed over the types
    // that are unseen at h. The renormalization constant Z_h is the sum of
    // the lower-order probabilities over that set (D-WB).
    //
    // The set of types unseen at h is the tokens with c(h,w) == 0 PLUS the
    // out-of-vocabulary class, which is unseen at every context by
    // construction. Leaving the OOV class out loses that share of the
    // reserved mass and the distribution stops summing to one.
    //
    // THE OOV CLASS IS SPELLED `kOovClass` AND IT IS A SINGLE, FIXED
    // SENTINEL. It is NOT "any token that is not in the vocabulary": a
    // caller may ask about an arbitrary out-of-vocabulary id, and every one
    // of them must receive the SAME probability -- the probability of the
    // class, not of a particular id. Treating the caller's id as the class
    // identity would make the reserved mass depend on which unknown id
    // happened to be asked about, and the distribution would no longer sum
    // to one over (vocabulary + one OOV class).
    const std::vector<int> shorter(context.begin(),
                                   context.begin() + (context.size() - 1));

    // The vocabulary tokens unseen at h, plus the single OOV class.
    std::vector<int> unseen;
    for (const auto& kv : unigram_) {
        if (ngram_count_of(context, kv.first) == 0) unseen.push_back(kv.first);
    }
    unseen.push_back(kOovClass);

    // Z_h and the lower-order probabilities of the unseen set.
    std::vector<double> lower_of_unseen;
    lower_of_unseen.reserve(unseen.size());
    double z = 0.0;
    for (int t : unseen) {
        const double p = shorter.empty() ? wb_unigram(t)
                                         : wb_order(order - 1, t, shorter);
        lower_of_unseen.push_back(p);
        z += p;
    }

    if (z == 0.0) return 0.0;

    // The lower-order probability of THIS token. A token that is unseen at
    // h gets its share of the reserved mass; a token that is SEEN at h was
    // handled above and never reaches here.
    double lower = 0.0;
    if (w == kOovClass || unigram_count(w) == 0) {
        // Either the OOV class itself, or a token that never occurs in the
        // corpus (which is the OOV class by definition).
        lower = wb_unigram(kOovClass);
        if (!shorter.empty()) lower = wb_order(order - 1, kOovClass, shorter);
    } else {
        for (size_t i = 0; i < unseen.size(); ++i) {
            if (unseen[i] == w) { lower = lower_of_unseen[i]; break; }
        }
    }

    return (t_h / (n_h + t_h)) * lower / z;
}

double NGramLM::wb_prob(int w, const std::vector<int>& context) const {
    // EVERY out-of-vocabulary id maps to the single OOV class, so the
    // probability of an unknown token does not depend on WHICH unknown id
    // was asked about. Without this mapping the reserved mass would be
    // split between the OOV class and the caller's specific id, and the
    // distribution would not sum to one.
    const int token = (unigram_count(w) > 0) ? w : kOovClass;
    return wb_order(context.size() + 1, token, context);
}

// ---------------------------------------------------------------------------
// ABSOLUTE DISCOUNTING WITH INTERPOLATION
// ---------------------------------------------------------------------------
double NGramLM::ad_order(size_t order, int w,
                          const std::vector<int>& context) const {
    if (order <= 1) return ad_unigram(w);

    const size_t c_h = ctx_count_of(context);
    if (c_h == 0) {
        // THE BACKOFF KEEPS THE SUFFIX of the context and drops its FIRST
        // token: h = (h1, h2) -> h' = (h2). The context of a language model
        // is a SUFFIX of the history, so dropping the OLDEST token is what
        // "the (n-1)-gram model" means. Taking the PREFIX instead (the
        // newest token) is a different model -- and for the fixture's
        // context (the,cat) it evaluates (cat) instead of (the), which
        // changes the answer.
        std::vector<int> shorter(context.begin() + 1, context.end());
        if (shorter.empty()) return ad_unigram(w);
        return ad_order(order - 1, w, shorter);
    }

    const double c_hw = static_cast<double>(ngram_count_of(context, w));
    const double v_h = static_cast<double>(ctx_types_of(context));
    const double d = discount_;
    const double ch = static_cast<double>(c_h);

    // The interpolation weight uses V_h, the DISTINCT-FOLLOWER count.
    const double lambda = d * v_h / ch;

    // The backoff keeps the SUFFIX of the context (see the note in
    // kn_order): h = (h1, h2) -> h' = (h2).
    std::vector<int> shorter(context.begin() + 1, context.end());
    const double lower = shorter.empty() ? ad_unigram(w)
                                         : ad_order(order - 1, w, shorter);
    return (1.0 - lambda) * c_hw / ch + lambda * lower;
}

double NGramLM::ad_prob(int w, const std::vector<int>& context) const {
    return ad_order(context.size() + 1, w, context);
}

// ---------------------------------------------------------------------------
// Sequence scoring
// ---------------------------------------------------------------------------
double NGramLM::sequence_logprob(const std::vector<int>& tokens,
                                  double discount,
                                  NGramScheme scheme) const {
    if (tokens.empty()) return 0.0;

    // A discount override of -1 (the default) means "use the model's own".
    // Any other value must be valid, because a negative or > 1 discount
    // would silently break the normalization.
    const double d = (discount < 0.0) ? discount_ : discount;
    if (!std::isfinite(d) || d <= 0.0 || d > 1.0) {
        throw std::invalid_argument("ngram: discount must be finite and in (0, 1]");
    }

    // The context of the first token is empty, so the first token is scored
    // by the unigram. The context length is capped at max_order - 1.
    const size_t max_ctx = (max_order_ > 1) ? (max_order_ - 1) : 0;

    // A discount override that differs from the model's own needs a model
    // that USES it, because the discount enters the backoff weight
    // D * V_h / c(h) and the clamped term max(c - D, 0). Rather than thread
    // the discount through every helper (and risk one of them silently
    // reading discount_ instead), the scoring path uses a private copy of
    // this model with the overridden discount. The copy is cheap: the count
    // tables are copied, not recomputed.
    if (d != discount_) {
        NGramLM tmp(*this);
        tmp.discount_ = d;
        return tmp.sequence_logprob(tokens, -1.0, scheme);
    }

    double total = 0.0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const size_t ctx_len = std::min(i, max_ctx);
        std::vector<int> ctx(tokens.begin() + (i - ctx_len), tokens.begin() + i);

        double p = 0.0;
        switch (scheme) {
            case NGramScheme::kKneserNey:
                p = kn_prob(tokens[i], ctx);
                break;
            case NGramScheme::kWittenBell:
                p = wb_prob(tokens[i], ctx);
                break;
            case NGramScheme::kAbsoluteDiscounting:
                p = ad_prob(tokens[i], ctx);
                break;
        }
        if (p <= 0.0) return -std::numeric_limits<double>::infinity();
        total += std::log(p);
    }
    return total;
}

double NGramLM::corpus_perplexity(const std::vector<std::vector<int>>& corpus,
                                   double discount,
                                   NGramScheme scheme) const {
    double total = 0.0;
    size_t n = 0;
    for (const std::vector<int>& sentence : corpus) {
        const double lp = sequence_logprob(sentence, discount, scheme);
        if (!std::isfinite(lp)) return std::numeric_limits<double>::infinity();
        total += lp;
        n += sentence.size();
    }
    if (n == 0) return std::numeric_limits<double>::infinity();
    return std::exp(-total / static_cast<double>(n));
}

// ---------------------------------------------------------------------------
// HEAPS' LAW
// ---------------------------------------------------------------------------
HeapsFit NGramLM::heaps_fit() const {
    // The corpus growth curve, recorded during training. Heaps' law is a
    // statement about the ORDER in which types first appear, and the count
    // tables do not record that order -- so the curve is snapshotted during
    // training rather than reconstructed here. Reconstructing it from the
    // unigram counts (in token-id order) would produce a DIFFERENT curve and
    // a different fit, which is the kind of quiet divergence a test cannot
    // see.
    return heaps_fit(growth_n_, growth_v_);
}

HeapsFit NGramLM::heaps_fit(const std::vector<double>& n_points,
                             const std::vector<double>& v_points) const {
    HeapsFit fit;
    if (n_points.size() != v_points.size()) {
        throw std::invalid_argument("ngram: heaps_fit points must have equal length");
    }
    if (n_points.size() < 2) {
        throw std::invalid_argument("ngram: heaps_fit needs at least 2 points");
    }
    // Least squares in log-log space: log V = log K + beta * log n.
    double sx = 0.0, sy = 0.0;
    size_t used = 0;
    for (size_t i = 0; i < n_points.size(); ++i) {
        if (!(n_points[i] > 0.0) || !(v_points[i] > 0.0)) continue;
        sx += std::log(n_points[i]);
        sy += std::log(v_points[i]);
        ++used;
    }
    if (used < 2) {
        throw std::invalid_argument("ngram: heaps_fit needs 2 positive points");
    }
    const double mx = sx / static_cast<double>(used);
    const double my = sy / static_cast<double>(used);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n_points.size(); ++i) {
        if (!(n_points[i] > 0.0) || !(v_points[i] > 0.0)) continue;
        const double dx = std::log(n_points[i]) - mx;
        num += dx * (std::log(v_points[i]) - my);
        den += dx * dx;
    }
    if (den == 0.0) {
        throw std::invalid_argument("ngram: heaps_fit points are degenerate");
    }
    const double beta = num / den;
    const double log_k = my - beta * mx;
    fit.beta = beta;
    fit.k = std::exp(log_k);

    // R^2 of the log-log fit.
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n_points.size(); ++i) {
        if (!(n_points[i] > 0.0) || !(v_points[i] > 0.0)) continue;
        const double y = std::log(v_points[i]);
        const double yhat = log_k + beta * std::log(n_points[i]);
        ss_res += (y - yhat) * (y - yhat);
        ss_tot += (y - my) * (y - my);
    }
    fit.r2 = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 1.0;
    return fit;
}

// ---------------------------------------------------------------------------
// GOOD-TURING
// ---------------------------------------------------------------------------
double NGramLM::good_turing_p0() const {
    // P_0 = N_1 / N: the probability mass assigned to types never seen.
    if (num_tokens_ == 0) return 0.0;
    size_t n1 = 0;
    for (const auto& kv : unigram_) {
        if (kv.second == 1) ++n1;
    }
    return static_cast<double>(n1) / static_cast<double>(num_tokens_);
}

double NGramLM::good_turing_discounted_count(size_t r) const {
    // N_r = the number of types whose count is exactly r.
    auto n_r = [&](size_t k) -> size_t {
        size_t n = 0;
        for (const auto& kv : unigram_) {
            if (kv.second == k) ++n;
        }
        return n;
    };
    const size_t nr = n_r(r);
    const size_t nr1 = n_r(r + 1);
    if (nr == 0 || nr1 == 0) return 0.0;
    return static_cast<double>(r + 1) * static_cast<double>(nr1) /
           static_cast<double>(nr);
}

double NGramLM::good_turing_probability(int w) const {
    const size_t c = unigram_count(w);
    if (c == 0) return 0.0;
    if (num_tokens_ == 0) return 0.0;
    const double c_star = good_turing_discounted_count(c);
    if (c_star > 0.0) return c_star / static_cast<double>(num_tokens_);
    // The discounted count is undefined for this r; fall back to the raw
    // count. This is the standard Good-Turing convention for the counts
    // above the switch point.
    return static_cast<double>(c) / static_cast<double>(num_tokens_);
}

}  // namespace utils
}  // namespace nn
