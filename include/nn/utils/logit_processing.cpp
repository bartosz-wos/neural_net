#include "logit_processing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nn {
namespace utils {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// Row access. A "row" is a flat view of size V over a (1, V) or (V, 1) tensor,
// so every processor is a single-distribution function and none of them needs a
// batch code path (the header says this is deliberate).
// ---------------------------------------------------------------------------
size_t vocab_of(const Tensor& t) {
    if (t.rows == 0 || t.cols == 0) {
        throw std::invalid_argument(
            "logit_processing: empty logits tensor");
    }
    if (t.rows != 1 && t.cols != 1) {
        throw std::invalid_argument(
            "logit_processing: expected a single distribution, got a "
            + std::to_string(t.rows) + "x" + std::to_string(t.cols) +
            " tensor; use apply_row() to drive a batch");
    }
    return std::max(t.rows, t.cols);
}

// Read entry j of a row regardless of whether it is laid out (1,V) or (V,1).
double get_at(const Tensor& t, size_t j) {
    return (t.rows == 1) ? t(0, j) : t(j, 0);
}

void set_at(Tensor& t, size_t j, double v) {
    if (t.rows == 1) t(0, j) = v;
    else              t(j, 0) = v;
}

// A fresh copy of `in` shaped like it, so processors never mutate the caller's
// tensor (S8 depends on this: comparing two orders needs both inputs intact).
Tensor like(const Tensor& in) { return in.clone(); }

// Descending-stable sort of indices: ties resolve to the LOWER original index
// (S12). std::stable_sort with a strict `>` comparator gives exactly that.
std::vector<size_t> descending_order(const std::vector<double>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t a, size_t b) { return v[a] > v[b]; });
    return idx;
}

std::vector<double> to_vec(const Tensor& t) {
    size_t v = vocab_of(t);
    std::vector<double> out(v);
    for (size_t j = 0; j < v; ++j) out[j] = get_at(t, j);
    return out;
}

// Multiplicative mean of the probabilities, used by min_p. Ties are excluded by
// the caller's threshold, so no tie handling is needed here.
double max_prob(const std::vector<double>& probs) {
    return *std::max_element(probs.begin(), probs.end());
}

std::vector<double> probabilities(const std::vector<double>& logits) {
    double m = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] - m);
        sum += p[i];
    }
    for (size_t i = 0; i < p.size(); ++i) p[i] /= sum;
    return p;
}

}  // namespace

// ===========================================================================
// softmax
// ===========================================================================
Tensor softmax(const Tensor& logits) {
    std::vector<double> lg = to_vec(logits);
    std::vector<double> p = probabilities(lg);
    Tensor out = like(logits);
    for (size_t j = 0; j < p.size(); ++j) set_at(out, j, p[j]);
    return out;
}

// ===========================================================================
// Temperature -- on the LOGITS (S7)
// ===========================================================================
Tensor apply_temperature(const Tensor& logits, double temperature) {
    if (!(temperature > 0.0) || !std::isfinite(temperature)) {
        throw std::invalid_argument(
            "apply_temperature: temperature must be finite and > 0");
    }
    Tensor out = like(logits);
    size_t v = vocab_of(logits);
    for (size_t j = 0; j < v; ++j) {
        double x = get_at(logits, j);
        // Dividing -inf by a positive number keeps it -inf, which is what lets a
        // truncated token survive the chain (S9). An explicit guard documents
        // the intent rather than relying on IEEE luck.
        set_at(out, j, std::isinf(x) ? x : x / temperature);
    }
    return out;
}

// ===========================================================================
// Top-k -- keep the k largest, ties first-wins (S3, S12, S13)
// ===========================================================================
Tensor apply_top_k(const Tensor& logits, size_t k) {
    std::vector<double> lg = to_vec(logits);
    size_t v = lg.size();
    Tensor out = like(logits);
    for (size_t j = 0; j < v; ++j) set_at(out, j, kNegInf);

    // `k >= v` is a no-op rather than an error: k == v must not -inf the argmax.
    if (k == 0) return out;
    if (k >= v) {
        for (size_t j = 0; j < v; ++j) set_at(out, j, get_at(logits, j));
        return out;
    }
    std::vector<size_t> order = descending_order(lg);
    for (size_t i = 0; i < k; ++i) set_at(out, order[i], get_at(logits, order[i]));
    return out;
}

// ===========================================================================
// Top-p / nucleus -- cumulative mass cut (S1, S2, S12)
// ===========================================================================
Tensor apply_top_p(const Tensor& logits, double p_threshold) {
    if (!(p_threshold > 0.0) || p_threshold > 1.0 + 1e-12) {
        throw std::invalid_argument(
            "apply_top_p: p must be in (0, 1]");
    }
    std::vector<double> lg = to_vec(logits);
    size_t v = lg.size();
    Tensor out = like(logits);
    for (size_t j = 0; j < v; ++j) set_at(out, j, kNegInf);

    std::vector<double> probs = probabilities(lg);
    std::vector<size_t> order = descending_order(lg);

    double cum = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        size_t idx = order[i];
        set_at(out, idx, get_at(logits, idx));
        cum += probs[idx];
        // S1: `>=`, not `>`. The token that REACHES the threshold is kept and the
        // loop stops there. On [0.5,0.3,0.15,0.05] with p = 0.8, 0.5+0.3 == 0.8
        // satisfies this and the result is {0,1}; a `>` here would admit {0,1,2}.
        if (cum >= p_threshold) break;
    }
    return out;
}

// ===========================================================================
// Min-p -- relative to the MAXIMUM, not the median (S4)
// ===========================================================================
Tensor apply_min_p(const Tensor& logits, double min_p, double temperature) {
    if (min_p < 0.0 || min_p > 1.0) {
        throw std::invalid_argument("apply_min_p: min_p must be in [0, 1]");
    }
    if (!(temperature > 0.0) || !std::isfinite(temperature)) {
        throw std::invalid_argument(
            "apply_min_p: temperature must be finite and > 0");
    }
    std::vector<double> lg = to_vec(logits);
    for (size_t j = 0; j < lg.size(); ++j) {
        if (!std::isinf(lg[j])) lg[j] /= temperature;
    }
    std::vector<double> probs = probabilities(lg);

    // The floor is min_p * p_max -- relative to the MOST LIKELY token. Using
    // the median or the mean is a different algorithm that looks plausible.
    double thr = min_p * max_prob(probs);

    Tensor out = like(logits);
    for (size_t j = 0; j < lg.size(); ++j) {
        bool keep = std::isinf(lg[j]) ? false : (probs[j] >= thr);
        set_at(out, j, keep ? get_at(logits, j) : kNegInf);
    }
    return out;
}

// ===========================================================================
// The three penalties (S5, S6)
// ===========================================================================
namespace {

// Validate the history against the vocabulary and return the token ids that are
// actually present. Out-of-range ids are a caller error, not a silent no-op --
// a token id of 7 into a 4-token vocabulary means the caller and the model
// disagree, and swallowing it would hide exactly the bug these tests hunt.
std::vector<size_t> validate_history(const std::vector<int>& history, size_t vocab,
                                     const char* who) {
    std::vector<size_t> ids;
    for (int t : history) {
        if (t < 0 || static_cast<size_t>(t) >= vocab) {
            throw std::invalid_argument(
                std::string(who) + ": token id " + std::to_string(t) +
                " is outside the vocabulary of size " + std::to_string(vocab));
        }
        ids.push_back(static_cast<size_t>(t));
    }
    return ids;
}

}  // namespace

Tensor apply_repetition_penalty(const Tensor& logits,
                                const std::vector<int>& history, double penalty) {
    size_t v = vocab_of(logits);
    validate_history(history, v, "apply_repetition_penalty");
    Tensor out = like(logits);

    // Deduplicate: the repetition penalty is applied ONCE per distinct token,
    // however many times it occurred.
    std::vector<bool> seen(v, false);
    for (int t : history) {
        size_t i = static_cast<size_t>(t);
        if (seen[i]) continue;
        seen[i] = true;
        double x = get_at(logits, i);
        if (std::isinf(x)) continue;   // -inf stays -inf (S9)
        // S5: THE SIGN SPLIT. Multiplying unconditionally would SUPPRESS
        // positive tokens and PROMOTE negative ones.
        set_at(out, i, x < 0.0 ? x * penalty : x / penalty);
    }
    return out;
}

Tensor apply_frequency_penalty(const Tensor& logits,
                               const std::vector<int>& history, double penalty) {
    size_t v = vocab_of(logits);
    validate_history(history, v, "apply_frequency_penalty");
    Tensor out = like(logits);

    std::vector<double> count(v, 0.0);
    for (int t : history) count[static_cast<size_t>(t)] += 1.0;

    for (size_t j = 0; j < v; ++j) {
        if (count[j] == 0.0) continue;
        double x = get_at(logits, j);
        if (std::isinf(x)) continue;
        // S6: LINEAR in the count, not clipped.
        set_at(out, j, x - penalty * count[j]);
    }
    return out;
}

Tensor apply_presence_penalty(const Tensor& logits,
                              const std::vector<int>& history, double penalty) {
    size_t v = vocab_of(logits);
    validate_history(history, v, "apply_presence_penalty");
    Tensor out = like(logits);

    std::vector<bool> present(v, false);
    for (int t : history) present[static_cast<size_t>(t)] = true;

    for (size_t j = 0; j < v; ++j) {
        if (!present[j]) continue;
        double x = get_at(logits, j);
        if (std::isinf(x)) continue;
        // S6: a STEP in the count -- 1x however many times it occurred.
        set_at(out, j, x - penalty);
    }
    return out;
}

// ===========================================================================
// Output projectors
// ===========================================================================
Tensor apply_epsilon(const Tensor& logits, double epsilon) {
    if (epsilon < 0.0 || epsilon > 1.0) {
        throw std::invalid_argument("apply_epsilon: epsilon must be in [0, 1]");
    }
    std::vector<double> lg = to_vec(logits);
    std::vector<double> probs = probabilities(lg);
    Tensor out = like(logits);
    for (size_t j = 0; j < probs.size(); ++j) {
        bool keep = std::isinf(lg[j]) ? false : (probs[j] >= epsilon);
        set_at(out, j, keep ? get_at(logits, j) : kNegInf);
    }
    return out;
}

Tensor apply_round_to_int(const Tensor& logits) {
    Tensor out = like(logits);
    size_t v = vocab_of(logits);
    for (size_t j = 0; j < v; ++j) {
        double x = get_at(logits, j);
        // S11: std::round is HALF AWAY FROM ZERO on both signs. std::nearbyint
        // would be ties-to-EVEN and get -2.5 wrong (-2 instead of -3).
        set_at(out, j, std::round(x));
    }
    return out;
}

// ===========================================================================
// The sampler (S10, S12)
// ===========================================================================
int sample_from_logits(const Tensor& logits, SamplingStrategy strategy,
                       std::mt19937& rng) {
    std::vector<double> lg = to_vec(logits);
    size_t v = lg.size();

    if (strategy == SamplingStrategy::GREEDY) {
        // S12: `>` (not `>=`) makes an exact tie resolve to the LOWER index,
        // and the rng is never consulted, so greedy is seed-independent.
        size_t best = 0;
        for (size_t j = 1; j < v; ++j) {
            if (lg[j] > lg[best]) best = j;
        }
        return static_cast<int>(best);
    }

    // S10: max-subtract BEFORE exp. exp(1000) is inf and inf/inf is NaN.
    std::vector<double> probs = probabilities(lg);

    double sum = 0.0;
    for (double p : probs) sum += p;
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        // Every entry is -inf (or NaN): there is nothing to draw from. Returning
        // the first index is total, whereas dividing by a zero sum is NaN and
        // walking a CDF that never reaches u reads off the end.
        return 0;
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double u = unif(rng) * sum;

    double acc = 0.0;
    for (size_t j = 0; j < v; ++j) {
        acc += probs[j];
        if (u < acc) return static_cast<int>(j);
    }
    // Only reachable through floating-point drift in the final CDF step.
    for (size_t j = v; j-- > 0;) {
        if (probs[j] > 0.0) return static_cast<int>(j);
    }
    return 0;
}

// ===========================================================================
// LogitProcessorChain
// ===========================================================================
LogitProcessorChain& LogitProcessorChain::add_temperature(double temperature) {
    double t = temperature;
    processors_.push_back([t](const Tensor& x) { return apply_temperature(x, t); });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_top_k(size_t k) {
    processors_.push_back([k](const Tensor& x) { return apply_top_k(x, k); });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_top_p(double p) {
    processors_.push_back([p](const Tensor& x) { return apply_top_p(x, p); });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_min_p(double min_p, double temperature) {
    processors_.push_back(
        [min_p, temperature](const Tensor& x) { return apply_min_p(x, min_p, temperature); });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_repetition_penalty(
    const std::vector<int>& history, double penalty) {
    processors_.push_back([history, penalty](const Tensor& x) {
        return apply_repetition_penalty(x, history, penalty);
    });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_frequency_penalty(
    const std::vector<int>& history, double penalty) {
    processors_.push_back([history, penalty](const Tensor& x) {
        return apply_frequency_penalty(x, history, penalty);
    });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_presence_penalty(
    const std::vector<int>& history, double penalty) {
    processors_.push_back([history, penalty](const Tensor& x) {
        return apply_presence_penalty(x, history, penalty);
    });
    return *this;
}

LogitProcessorChain& LogitProcessorChain::add_epsilon(double epsilon) {
    processors_.push_back(
        [epsilon](const Tensor& x) { return apply_epsilon(x, epsilon); });
    return *this;
}

// S8: LEFT TO RIGHT is the contract. Reverse this fold and a chain of
// temperature-then-top-p admits a token carrying ~0.049 of the mass that the
// documented order rejects.
Tensor LogitProcessorChain::apply(const Tensor& logits) const {
    Tensor cur = logits.clone();
    for (const auto& fn : processors_) cur = fn(cur);
    return cur;
}

// ===========================================================================
// BeamSearchDecoder
// ===========================================================================
BeamSearchDecoder::BeamSearchDecoder(size_t vocab, BeamSearchConfig cfg,
                                     ScoreFn fn)
    : vocab_(vocab), cfg_(cfg), score_fn_(std::move(fn)) {
    if (vocab == 0) {
        throw std::invalid_argument("BeamSearchDecoder: vocab must be > 0");
    }
    if (cfg_.beam_width == 0) {
        throw std::invalid_argument("BeamSearchDecoder: beam_width must be > 0");
    }
    if (cfg_.max_length == 0) {
        throw std::invalid_argument("BeamSearchDecoder: max_length must be > 0");
    }
    if (!score_fn_) {
        throw std::invalid_argument("BeamSearchDecoder: score function must not be null");
    }
    if (cfg_.eos_id >= 0 && static_cast<size_t>(cfg_.eos_id) >= vocab_) {
        throw std::invalid_argument(
            "BeamSearchDecoder: eos_id is outside the vocabulary");
    }
}

Tensor BeamSearchDecoder::decode(const Tensor& start_ids, size_t num_return) {
    if (start_ids.rows != 1) {
        throw std::invalid_argument("BeamSearchDecoder: start_ids must have 1 row");
    }
    if (num_return == 0) {
        throw std::invalid_argument("BeamSearchDecoder: num_return must be > 0");
    }
    finished_.clear();

    std::vector<BeamHypothesis> live;
    {
        BeamHypothesis h;
        h.tokens = start_ids[0] ? std::vector<int>() : std::vector<int>();
        for (size_t j = 0; j < start_ids.cols; ++j)
            h.tokens.push_back(static_cast<int>(start_ids(0, j)));
        h.logprob = 0.0;
        live.push_back(h);
    }

    size_t prompt_len = start_ids.cols;

    for (size_t step = 0; step < cfg_.max_length; ++step) {
        // Score every live hypothesis in ONE batched call: the rows of `batch`
        // are the live hypotheses' token sequences.
        Tensor batch(live.size(), 0);
        {
            size_t width = live.empty() ? 0 : live[0].tokens.size();
            batch = Tensor(live.size(), width);
            for (size_t b = 0; b < live.size(); ++b) {
                for (size_t j = 0; j < width && j < live[b].tokens.size(); ++j)
                    batch(b, j) = static_cast<double>(live[b].tokens[j]);
            }
        }

        Tensor next = score_fn_(batch);
        if (next.rows != live.size() || next.cols != vocab_) {
            throw std::runtime_error(
                "BeamSearchDecoder: score_fn returned a " +
                std::to_string(next.rows) + "x" + std::to_string(next.cols) +
                " tensor, expected " + std::to_string(live.size()) + "x" +
                std::to_string(vocab_));
        }

        // Every candidate (beam, token) scored by accumulated log-probability.
        struct Candidate {
            size_t beam;
            int token;
            double logprob;
        };
        std::vector<Candidate> cands;
        cands.reserve(live.size() * vocab_);

        for (size_t b = 0; b < live.size(); ++b) {
            Tensor row_logits(1, vocab_);
            for (size_t j = 0; j < vocab_; ++j) row_logits(0, j) = next(b, j);
            Tensor probs = softmax(row_logits);
            for (size_t j = 0; j < vocab_; ++j) {
                double p = probs(0, j);
                if (p <= 0.0) continue;
                cands.push_back({b, static_cast<int>(j),
                                 live[b].logprob + std::log(p)});
            }
        }
        if (cands.empty()) break;

        // Descending, ties to the lower (beam, token) pair -- deterministic.
        std::stable_sort(cands.begin(), cands.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.logprob > b.logprob;
                         });

        std::vector<BeamHypothesis> next_live;
        for (const Candidate& c : cands) {
            if (next_live.size() >= cfg_.beam_width) break;
            // Every hypothesis descends from a distinct prefix, so dropping one
            // parent wholesale is safe and avoids duplicate sequences.
            bool already = false;
            for (const BeamHypothesis& h : next_live)
                if (h.tokens == live[c.beam].tokens) { already = true; break; }
            if (already) continue;

            BeamHypothesis h = live[c.beam];
            h.tokens.push_back(c.token);
            h.logprob = c.logprob;
            // eod is NOT eos: only the configured eos_id terminates.
            if (c.token == cfg_.eos_id) {
                h.finished = true;
                finished_.push_back(h);
            } else {
                next_live.push_back(h);
            }
        }

        if (cfg_.early_stopping && finished_.size() >= cfg_.beam_width) break;
        live = next_live;
        if (live.empty()) break;
        if (live[0].tokens.size() >= prompt_len + cfg_.max_length) break;
    }

    // If nothing finished, the best live hypotheses are still the answer.
    if (finished_.empty()) {
        for (const BeamHypothesis& h : live) finished_.push_back(h);
    }

    // S14: THE LENGTH PENALTY IS APPLIED HERE AND NOWHERE ELSE. Applying it
    // during live expansion would penalise a beam for being short, which
    // biases the whole search toward short outputs -- and because live beams are
    // all compared at the SAME length, the penalty cancels exactly, so doing it
    // there is both wrong and pointless.
    std::stable_sort(finished_.begin(), finished_.end(),
                     [this](const BeamHypothesis& a, const BeamHypothesis& b) {
                         double sa = a.logprob;
                         double sb = b.logprob;
                         if (cfg_.length_penalty != 0.0) {
                             double la = static_cast<double>(
                                 std::max<size_t>(1, a.tokens.size()));
                             double lb = static_cast<double>(
                                 std::max<size_t>(1, b.tokens.size()));
                             sa /= std::pow(la, cfg_.length_penalty);
                             sb /= std::pow(lb, cfg_.length_penalty);
                         }
                         return sa > sb;
                     });

    size_t want = std::min(num_return, finished_.size());
    size_t width = 0;
    for (size_t i = 0; i < want; ++i)
        width = std::max(width, finished_[i].tokens.size());
    if (width == 0) width = 1;

    Tensor out(want, width);
    for (size_t i = 0; i < want; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out(i, j) = (j < finished_[i].tokens.size())
                            ? static_cast<double>(finished_[i].tokens[j])
                            : static_cast<double>(cfg_.eos_id);
        }
    }
    return out;
}

// ===========================================================================
// autoregressive_sample
// ===========================================================================
AutoregressiveResult autoregressive_sample(
    const BeamSearchDecoder::ScoreFn& score_fn, const Tensor& start_ids,
    size_t steps, const LogitProcessorChain& chain, SamplingStrategy strategy,
    std::mt19937& rng, int eos_id, bool stop_on_eos) {
    if (!score_fn) {
        throw std::invalid_argument(
            "autoregressive_sample: score function must not be null");
    }
    if (start_ids.rows != 1) {
        throw std::invalid_argument(
            "autoregressive_sample: start_ids must have 1 row");
    }

    size_t width = start_ids.cols + steps;
    Tensor ids(1, width);
    for (size_t j = 0; j < start_ids.cols; ++j)
        ids(0, j) = start_ids(0, j);

    AutoregressiveResult result;
    result.tokens = Tensor(1, start_ids.cols);
    for (size_t j = 0; j < start_ids.cols; ++j)
        result.tokens(0, j) = start_ids(0, j);

    for (size_t step = 0; step < steps; ++step) {
        // Score the prefix generated so far. The model sees ONLY what exists.
        size_t len = start_ids.cols + step;
        Tensor prefix(1, len);
        for (size_t j = 0; j < len; ++j) prefix(0, j) = ids(0, j);

        Tensor logits = score_fn(prefix);
        if (logits.rows != 1) {
            throw std::runtime_error(
                "autoregressive_sample: score_fn must return 1 row");
        }

        Tensor row_logits(1, logits.cols);
        for (size_t j = 0; j < logits.cols; ++j) row_logits(0, j) = logits(0, j);

        Tensor processed = chain.apply(row_logits);
        int tok = sample_from_logits(processed, strategy, rng);

        ids(0, len) = static_cast<double>(tok);

        Tensor grown(1, len + 1);
        for (size_t j = 0; j <= len; ++j) grown(0, j) = ids(0, j);
        result.tokens = grown;

        if (stop_on_eos && eos_id >= 0 && tok == eos_id) {
            result.stopped_early = true;
            break;
        }
    }
    return result;
}

}  // namespace utils
}  // namespace nn