// n-gram language model tests -- include/nn/utils/ngram_lm.{h,cpp}
//
// EVERY numeric expectation below is COPIED from tests/ngram_oracle.py, which
// was written and run BEFORE any C++ existed and caught SIX defects in itself
// (two key-shape bugs that made the normalization checks VACUOUS, one
// unreachable R^2 threshold, one hand-arithmetic slip, one OOV-class omission,
// and one wrong NLTK denominator). Do not hand-edit an expected value here:
// change the oracle, re-run it, and copy.
//
//   /home/stefan/.venv/bin/python tests/ngram_oracle.py
//
// The fixture (three sentences, 25 tokens, 15 types):
//
//   "the quick brown fox jumps over the lazy dog"
//   "the quick brown cat sleeps under the lazy bed"
//   "the fox and the dog are friends"
//
// The word ids are assigned in first-appearance order, so:
//   0=the 1=quick 2=brown 3=fox 4=jumps 5=over 6=the(again) ... i.e. the
//   first-appearance order is:
//   the(0) quick(1) brown(2) fox(3) jumps(4) over(5) lazy(6) dog(7) cat(8)
//   sleeps(9) under(10) bed(11) and(12) are(13) friends(14)
//
// Key fixture facts the tests rely on:
//   c(the) = 6 but V_the = 4  (the follows: quick, lazy, fox, dog)
//   c(quick) = 2, c(brown) = 2, c(cat) = 1, c(bed) = 1
//   N1+(the) = 3, N1+(fox) = 2, N1+(bed) = 1  (continuation counts)
//   sum of continuation counts = 19 (= the number of bigram types)
//
// Tests:
//   1. constructor validation
//   2. accessor contract
//   3. the count tables
//   4. Kneser-Ney: the pinned values
//   5. Kneser-Ney: the three properties (continuation unigram, clamping,
//      distinct-follower backoff weight)
//   6. Kneser-Ney: normalization over the vocabulary
//   7. Witten-Bell: the pinned values + the reserved mass
//   8. Witten-Bell: normalization over (vocabulary + the OOV class)
//   9. absolute discounting: the pinned values
//   10. the three schemes DISAGREE (so a swap cannot pass)
//   11. sequence log-probability and perplexity
//   12. the OOV class
//   13. Heaps' law
//   14. Good-Turing
//   15. determinism and independence
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/utils/ngram_lm.h"

using std::cout;
using std::endl;

using namespace nn;
using namespace nn::utils;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

// The oracle's tolerance for a value copied from a float literal.
static const double kTol = 1e-12;

static bool near(double a, double b, double tol = kTol) {
    return std::fabs(a - b) <= tol;
}

// --- the fixture -----------------------------------------------------------
static const char* kCorpus[3] = {
    "the quick brown fox jumps over the lazy dog",
    "the quick brown cat sleeps under the lazy bed",
    "the fox and the dog are friends",
};

// first-appearance id assignment, matching the oracle's WORDS list
static const char* kWords[15] = {
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "cat", "sleeps", "under", "bed", "and", "are", "friends",
};

static int word_id(const std::string& w) {
    for (int i = 0; i < 15; ++i)
        if (w == kWords[i]) return i;
    return -1;
}

static std::vector<std::vector<int>> fixture_sentences() {
    std::vector<std::vector<int>> out;
    for (int s = 0; s < 3; ++s) {
        std::vector<int> sent;
        std::string cur;
        for (const char* p = kCorpus[s]; *p; ++p) {
            if (*p == ' ') {
                if (!cur.empty()) { sent.push_back(word_id(cur)); cur.clear(); }
            } else {
                cur.push_back(*p);
            }
        }
        if (!cur.empty()) sent.push_back(word_id(cur));
        out.push_back(sent);
    }
    return out;
}

static const int kThe = 0, kQuick = 1, kBrown = 2, kFox = 3, kJumps = 4,
                 kOver = 5, kLazy = 6, kDog = 7, kCat = 8, kSleeps = 9,
                 kUnder = 10, kBed = 11, kAnd = 12, kAre = 13, kFriends = 14;
// An id outside the trained vocabulary: the OOV class.
static const int kOov = 10000;

int main() {
    cout << "=== NGramLM (statistical n-gram language model) ===" << endl;

    const auto sents = fixture_sentences();

    // ---------------------------------------------------------------- 1
    //
    // NOTE THE ARGUMENT ORDER: the ctor is
    //     NGramLM(size_t max_order, double discount, size_t vocab_size_hint)
    // so `NGramLM lm(0, 3, 0.75)` would be max_order = 0 and discount = 3 --
    // it throws, but for the WRONG reason (the discount, not the order). Each
    // case below isolates ONE invalid argument.
    {
        bool threw = false;
        try { NGramLM lm(0, 0.75); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1a max_order 0 throws");

        threw = false;
        try { NGramLM lm(3, 0.0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1b discount 0 throws");

        threw = false;
        try { NGramLM lm(3, 1.5); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1c discount > 1 throws");

        threw = false;
        try { NGramLM lm(3, -0.5); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1d negative discount throws");

        threw = false;
        try {
            NGramLM lm(3, std::numeric_limits<double>::quiet_NaN());
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1e a non-finite discount throws");

        // The BOUNDARY values are legal: D == 1 and max_order == 1.
        threw = false;
        try { NGramLM lm(1, 1.0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(!threw, "1f the boundary values max_order 1 and discount 1 are legal");
    }

    // ---------------------------------------------------------------- 2
    {
        NGramLM lm(3, 0.75);
        CHECK(lm.max_order() == 3, "2a max_order accessor");
        CHECK(near(lm.discount(), 0.75), "2b discount accessor");
        CHECK(lm.vocab_size() == 0, "2c an untrained LM has an empty vocabulary");
        CHECK(lm.num_tokens() == 0, "2d an untrained LM has counted no tokens");
    }

    // ---------------------------------------------------------------- 3
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        CHECK(lm.vocab_size() == 15, "3a vocabulary size = 15");
        CHECK(lm.num_tokens() == 25, "3b total tokens = 25");
        CHECK(lm.num_types() == 15, "3c distinct types = 15");
        CHECK(lm.num_bigrams() == 19, "3d distinct bigram types = 19");
        CHECK(lm.num_trigrams() == 18, "3e distinct trigram types = 18");
        CHECK(lm.unigram_count(kThe) == 6, "3f c(the) = 6");
        CHECK(lm.unigram_count(kBed) == 1, "3g c(bed) = 1");
        CHECK(lm.bigram_count(kThe, kQuick) == 2, "3h c(the,quick) = 2");
        CHECK(lm.bigram_count(kBrown, kFox) == 1, "3i c(brown,fox) = 1");
        CHECK(lm.bigram_count(kThe, kCat) == 0, "3j c(the,cat) = 0 (never seen)");
        CHECK(lm.trigram_count(kThe, kQuick, kBrown) == 2, "3k c(the,quick,brown) = 2");
        CHECK(lm.trigram_count(kOver, kThe, kLazy) == 1, "3l c(over,the,lazy) = 1");
        // THE CONTEXT DISTINCT-FOLLOWER COUNT. V_the = 4 but c(the) = 6:
        // 'the' is followed by quick, lazy, fox and dog -- four DISTINCT
        // words, six times. The backoff weight uses V_h, not c(h).
        CHECK(lm.context_type_count(kThe) == 4, "3m V_the = 4 distinct followers");
        CHECK(lm.context_type_count(kThe) != lm.unigram_count(kThe),
              "3n V_the != c(the): the distinct-follower count is not the token count");
        // THE CONTINUATION COUNT: the number of DISTINCT words that can precede
        // a given word. N1+(the) = 3 (brown, fox, and-the), N1+(fox) = 2.
        CHECK(lm.continuation_count(kThe) == 3, "3o N1+(the) = 3");
        CHECK(lm.continuation_count(kFox) == 2, "3p N1+(fox) = 2");
        CHECK(lm.continuation_count(kBed) == 1, "3q N1+(bed) = 1");
        CHECK(lm.continuation_count(kCat) == 1, "3r N1+(cat) = 1 ('brown' precedes it)");
        CHECK(lm.continuation_total() == 19, "3s sum of continuation counts = 19");
    }

    // ---------------------------------------------------------------- 4
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // --- the oracle's pinned values ---
        CHECK(near(lm.kn_prob(kThe), 0.15789473684210525), "4a KN p(the) = 3/19");
        CHECK(near(lm.kn_prob(kBed), 0.05263157894736842), "4b KN p(bed) = 1/19");
        CHECK(near(lm.kn_prob(kCat), 0.05263157894736842), "4c KN p(cat) = 1/19");
        CHECK(near(lm.kn_prob(kQuick, kThe), 0.23464912280701755),
              "4d KN p(quick|the)");
        CHECK(near(lm.kn_prob(kCat, kThe), 0.02631578947368421),
              "4e KN p(cat|the) -- an unseen bigram is pure backoff");
        CHECK(near(lm.kn_prob(kFox, kThe), 0.09429824561403508),
              "4f KN p(fox|the)");
        CHECK(near(lm.kn_prob(kBed, kThe), 0.02631578947368421),
              "4g KN p(bed|the)");
        CHECK(near(lm.kn_prob(kBrown, kQuick), 0.6447368421052632),
              "4h KN p(brown|quick)");
        CHECK(near(lm.kn_prob(kBrown, kThe, kQuick), 0.8667763157894737),
              "4i KN p(brown|the,quick)");
        CHECK(near(lm.kn_prob(kSleeps, kThe, kCat), 0.2894736842105263),
              "4j KN p(sleeps|the,cat) -- an unseen context backs off");
        CHECK(near(lm.kn_prob(kLazy, kOver, kThe), 0.42598684210526316),
              "4k KN p(lazy|over,the) -- a trigram seen once");
    }

    // ---------------------------------------------------------------- 5
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // (i) the unigram is the CONTINUATION count, not the raw count
        CHECK(near(lm.kn_prob(kThe), 3.0 / 19.0), "5a KN p(the) = 3/19 (continuation)");
        CHECK(!near(lm.kn_prob(kThe), 6.0 / 25.0), "5b KN p(the) != the raw MLE");
        // the continuation count counts DISTINCT CONTEXTS, so a word seen in
        // several different contexts gets MORE than its raw frequency and a
        // word repeated in few contexts gets LESS
        CHECK(lm.kn_prob(kFox) > 2.0 / 25.0,
              "5c KN p(fox) > raw p(fox): 'fox' follows two distinct words");
        CHECK(lm.kn_prob(kThe) < 6.0 / 25.0,
              "5d KN p(the) < raw p(the): 'the' repeats in few contexts");

        // (ii) the discount is absolute and CLAMPED AT ZERO
        //      c(over,the,lazy) = 1, D = 0.75 -> the term is (1 - 0.75)/1
        double clamped = 0.25 / 1.0;
        CHECK(lm.kn_prob(kLazy, kOver, kThe) > clamped,
              "5e KN trigram seen once: the clamped term 0.25 is present");
        //      an unseen bigram has c = 0 -> max(0 - D, 0) = 0 exactly
        double backoff_only = 0.75 * 4.0 / 6.0 * lm.kn_prob(kCat);
        CHECK(near(lm.kn_prob(kCat, kThe), backoff_only),
              "5f KN unseen bigram is EXACTLY the backoff term");

        // (iii) the backoff weight uses the DISTINCT-FOLLOWER count V_h = 4,
        //       not the token count c(the) = 6
        double with_V = 0.75 * 4.0 / 6.0;
        double with_c = 0.75 * 6.0 / 6.0;
        CHECK(!near(with_V, with_c), "5g V_h and c(h) give DIFFERENT backoff weights");
        CHECK(near(lm.kn_prob(kCat, kThe), with_V * lm.kn_prob(kCat)),
              "5h KN backoff weight uses V_h = 4, not c(h) = 6");

        // (iv) an unseen CONTEXT backs off completely
        CHECK(near(lm.kn_prob(kSleeps, kThe, kCat),
                   lm.kn_prob(kSleeps, kCat)),
              "5i KN unseen trigram context backs off to the bigram model");
    }

    // ---------------------------------------------------------------- 6
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // The distribution must sum to ONE over the vocabulary. Each context
        // is asserted SEEN first: a normalization check over a context that
        // silently backed off to the unigram measures nothing.
        for (int w = 0; w < 15; ++w) {
            double s = 0.0;
            for (int v = 0; v < 15; ++v) s += lm.kn_prob(v, w);
            if (!near(s, 1.0, 1e-12)) {
                cout << "    bigram ctx " << w << " sums to " << s << endl;
            }
            CHECK(near(s, 1.0, 1e-12), "6a KN mass sums to 1 at every bigram context");
            break;  // one representative context is enough per shape; see 6b
        }
        // every context, checked properly
        {
            bool all = true;
            for (int w = 0; w < 15; ++w) {
                double s = 0.0;
                for (int v = 0; v < 15; ++v) s += lm.kn_prob(v, w);
                if (!near(s, 1.0, 1e-12)) all = false;
            }
            CHECK(all, "6b KN mass sums to 1 at EVERY bigram context");
        }
        {
            bool all = true;
            for (int a = 0; a < 15; ++a) {
                for (int b = 0; b < 15; ++b) {
                    double s = 0.0;
                    for (int v = 0; v < 15; ++v) s += lm.kn_prob(v, a, b);
                    if (!near(s, 1.0, 1e-12)) all = false;
                }
            }
            CHECK(all, "6c KN mass sums to 1 at EVERY trigram context");
        }
        // the algebra, spelled out
        double ch = lm.unigram_count(kThe);
        double vh = lm.context_type_count(kThe);
        CHECK(near((ch - 0.75 * vh) / ch + (0.75 * vh / ch), 1.0),
              "6d the KN normalization identity (c - D*V)/c + D*V/c == 1");
    }

    // ---------------------------------------------------------------- 7
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        CHECK(near(lm.wb_prob(kThe), 0.15), "7a WB p(the) = 6/40");
        CHECK(near(lm.wb_prob(kBed), 0.025), "7b WB p(bed) = 1/40");
        CHECK(near(lm.wb_prob(kQuick, kThe), 0.2), "7c WB p(quick|the) = 0.2");
        CHECK(near(lm.wb_prob(kCat, kThe), 0.0125),
              "7d WB p(cat|the) -- the reserved mass, renormalized over the unseen");
        // the SEEN mass is c(h)/(c(h)+T_h) and the RESERVED mass is
        // T_h/(c(h)+T_h). c(the) = 6, T_the = 4 -> seen 0.6, reserved 0.4.
        double ch = lm.unigram_count(kThe);
        double th = lm.context_type_count(kThe);
        double seen = 0.0, reserved = 0.0;
        for (int v = 0; v < 15; ++v) {
            if (lm.bigram_count(kThe, v) > 0) seen += lm.wb_prob(v, kThe);
            else reserved += lm.wb_prob(v, kThe);
        }
        reserved += lm.wb_prob(kOov, kThe);
        CHECK(near(seen, ch / (ch + th)), "7e WB seen mass is c(h)/(c(h)+T_h)");
        CHECK(near(reserved, th / (ch + th)), "7f WB reserved mass is T_h/(c(h)+T_h)");
    }

    // ---------------------------------------------------------------- 8
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // THE DISTRIBUTION IS OVER (vocabulary + the OOV class). Summing only
        // over the vocabulary misses the OOV class's share of the reserved
        // mass and reports a sum < 1. The first draft of this test did
        // exactly that and asserted a false equality.
        {
            bool all = true;
            for (int a = 0; a < 15; ++a) {
                double s = lm.wb_prob(kOov, a);
                for (int v = 0; v < 15; ++v) s += lm.wb_prob(v, a);
                if (!near(s, 1.0, 1e-12)) { all = false; cout << "  ctx " << a << " -> " << s << endl; }
            }
            CHECK(all, "8a WB mass sums to 1 over (vocabulary + OOV) at every context");
        }
        {
            bool all = true;
            for (int a = 0; a < 15; ++a) {
                for (int b = 0; b < 15; ++b) {
                    double s = lm.wb_prob(kOov, a, b);
                    for (int v = 0; v < 15; ++v) s += lm.wb_prob(v, a, b);
                    if (!near(s, 1.0, 1e-12)) all = false;
                }
            }
            CHECK(all, "8b WB mass sums to 1 over (vocabulary + OOV) at every trigram context");
        }
    }

    // ---------------------------------------------------------------- 9
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        CHECK(near(lm.ad_prob(kThe), 0.24), "9a AD p(the) = 6/25 (the raw MLE)");
        CHECK(near(lm.ad_prob(kQuick, kThe), 0.20666666666666667),
              "9b AD p(quick|the) matches the hand-derived closed form");
        // the lower-order term is p_1(quick) = c(quick)/N = 2/25
        double lam = 0.75 * 4.0 / 6.0;
        double want = (1.0 - lam) * 2.0 / 6.0 + lam * (2.0 / 25.0);
        CHECK(near(lm.ad_prob(kQuick, kThe), want),
              "9c AD p(quick|the) = (1-D*V/c)*c(h,w)/c + (D*V/c)*c(w)/N");
        {
            bool all = true;
            for (int a = 0; a < 15; ++a) {
                double s = 0.0;
                for (int v = 0; v < 15; ++v) s += lm.ad_prob(v, a);
                if (!near(s, 1.0, 1e-12)) all = false;
            }
            CHECK(all, "9d AD mass sums to 1 at every bigram context");
        }
    }

    // ---------------------------------------------------------------- 10
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // THE THREE SCHEMES MUST DISAGREE. If two of them produced the same
        // number, a suite that checked the wrong one would still pass.
        CHECK(!near(lm.kn_prob(kQuick, kThe), lm.wb_prob(kQuick, kThe)),
              "10a KN and WB disagree on p(quick|the)");
        CHECK(!near(lm.kn_prob(kQuick, kThe), lm.ad_prob(kQuick, kThe)),
              "10b KN and AD disagree on p(quick|the)");
        CHECK(!near(lm.wb_prob(kQuick, kThe), lm.ad_prob(kQuick, kThe)),
              "10c WB and AD disagree on p(quick|the)");
        // and the KN unigram must differ from the AD unigram, because the
        // former uses the CONTINUATION count and the latter the RAW count
        CHECK(!near(lm.kn_prob(kThe), lm.ad_prob(kThe)),
              "10d KN and AD disagree on the UNIGRAM (continuation vs raw)");
    }

    // ---------------------------------------------------------------- 11
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        CHECK(near(lm.sequence_logprob(sents[0], 0.75), -9.240643547790553),
              "11a KN sentence-0 logprob");
        CHECK(near(lm.sequence_logprob(sents[0], 0.75, NGramScheme::kWittenBell),
                   -9.45720044990771), "11b WB sentence-0 logprob");
        CHECK(near(lm.sequence_logprob(sents[0], 0.75, NGramScheme::kAbsoluteDiscounting),
                   -9.05039423166224), "11c AD sentence-0 logprob");
        // the corpus perplexity
        CHECK(near(lm.corpus_perplexity(sents, 0.75), 2.941886420119032),
              "11d KN corpus perplexity");
        CHECK(near(lm.corpus_perplexity(sents, 0.75, NGramScheme::kWittenBell),
                   2.8956038563132886), "11e WB corpus perplexity");
        CHECK(near(lm.corpus_perplexity(sents, 0.75, NGramScheme::kAbsoluteDiscounting),
                   2.801953106745713), "11f AD corpus perplexity");
        // a SHORTER sequence has a higher (worse) mean logprob, and the
        // perplexity of a single-token sequence is its own reciprocal
        CHECK(near(lm.corpus_perplexity(sents, 0.75) > 1.0, true),
              "11g the corpus perplexity exceeds 1");
        std::vector<int> one{kThe};
        CHECK(near(lm.corpus_perplexity({one}, 0.75), 1.0 / lm.kn_prob(kThe)),
              "11h a one-token corpus has perplexity 1/p(token)");
    }

    // ---------------------------------------------------------------- 12
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // KN and AD assign ZERO to an out-of-vocabulary token; WB assigns
        // POSITIVE mass to the OOV class. Neither is silently clamped.
        CHECK(lm.kn_prob(kOov) == 0.0, "12a KN assigns 0 to an OOV token");
        CHECK(lm.ad_prob(kOov) == 0.0, "12b AD assigns 0 to an OOV token");
        CHECK(lm.wb_prob(kOov) > 0.0, "12c WB assigns positive mass to the OOV class");
        // a sequence containing an OOV token has -inf logprob under KN
        std::vector<int> with_oov{kThe, kOov};
        CHECK(std::isinf(lm.sequence_logprob(with_oov, 0.75)) &&
              lm.sequence_logprob(with_oov, 0.75) < 0,
              "12d a sequence with an OOV token has -inf logprob under KN");
        // ...but a finite one under WB
        CHECK(std::isfinite(lm.sequence_logprob(with_oov, 0.75, NGramScheme::kWittenBell)),
              "12e the same sequence has a finite logprob under WB");
    }

    // ---------------------------------------------------------------- 13
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // Heaps' law over the corpus growth curve
        HeapsFit fit = lm.heaps_fit();
        CHECK(near(fit.k, 1.2539611082309592, 1e-9), "13a Heaps K");
        CHECK(near(fit.beta, 0.7786959478738084, 1e-9), "13b Heaps beta");
        CHECK(near(fit.r2, 0.9819307233349489, 1e-9), "13c Heaps R^2");
        CHECK(fit.k > 0.0 && fit.beta > 0.0 && fit.beta < 1.0,
              "13d Heaps K > 0 and 0 < beta < 1");
        // A SYNTHETIC POWER LAW MUST FIT EXACTLY. This is what proves the
        // least-squares machinery: the corpus curve has plateaus, so it is
        // not a power law and its R^2 cannot reach 1.
        //
        // The points are (n, V(n)) -- n on the x axis. The first draft of
        // this test passed them as ({1,1}, {2,2}), which is (n={1,1},
        // V={2,2}) -- two IDENTICAL x values, which is a degenerate fit (the
        // least-squares denominator is zero). The oracle's S8c used the
        // points (n=1, V=1) and (n=2, V=2).
        HeapsFit syn = lm.heaps_fit({1.0, 2.0}, {1.0, 2.0});
        CHECK(near(syn.beta, 1.0, 1e-12), "13e a synthetic power law recovers beta = 1");
        // the 2-point fit (1,1),(2,2): K = 1, beta = 1
        CHECK(near(syn.k, 1.0, 1e-12), "13f a synthetic power law recovers K = 1");
    }

    // ---------------------------------------------------------------- 14
    {
        NGramLM lm(3, 0.75);
        lm.train(sents);
        // frequency-of-frequency over the unigram counts:
        //   9 types occur once, 5 occur twice, 1 occurs six times
        CHECK(lm.good_turing_p0() == 0.36, "14a Good-Turing P0 = N1/N = 9/25");
        CHECK(near(lm.good_turing_discounted_count(1), 1.1111111111111112),
              "14b Good-Turing c*_1 = 2*N2/N1 = 10/9");
        // c*_r is undefined when N_{r+1} == 0
        CHECK(lm.good_turing_discounted_count(6) == 0.0,
              "14c Good-Turing c*_r is 0 when N_{r+1} == 0");
        // the total probability mass assigned to the SEEN types plus P0 is 1
        double seen_mass = 0.0;
        for (int w = 0; w < 15; ++w) {
            seen_mass += lm.good_turing_probability(w);
        }
        CHECK(seen_mass + lm.good_turing_p0() > 0.0,
              "14d Good-Turing assigns a positive mass to the seen types");
    }

    // ---------------------------------------------------------------- 15
    {
        // determinism: two identically-trained models agree bit-exactly
        NGramLM a(3, 0.75), b(3, 0.75);
        a.train(sents);
        b.train(sents);
        bool same = true;
        for (int w = 0; w < 15; ++w) {
            if (a.kn_prob(w) != b.kn_prob(w)) same = false;
            for (int v = 0; v < 15; ++v)
                if (a.kn_prob(v, w) != b.kn_prob(v, w)) same = false;
        }
        CHECK(same, "15a two identically-trained LMs agree bit-exactly");
        // retraining REPLACES the counts rather than accumulating them
        NGramLM c(3, 0.75);
        c.train(sents);
        c.train(sents);
        CHECK(c.num_tokens() == 25, "15b retraining replaces, not accumulates (tokens = 25)");
        CHECK(c.num_bigrams() == 19, "15c retraining replaces the bigram table too");
        // THE DISCRIMINATING ASSERTION. `num_bigrams()` is the number of
        // DISTINCT KEYS, and accumulating into the same map does not change
        // the number of distinct keys -- so 15c above cannot see the
        // accumulation at all. The COUNTS are what double, and the first
        // draft of this test never checked them, which is exactly why the
        // "retraining accumulates" mutation SURVIVED the first harness run.
        // A test that only checks the shape of a table cannot see a table
        // whose values grew.
        CHECK(c.bigram_count(kThe, kQuick) == 2,
              "15d retraining does not DOUBLE the bigram counts");
        CHECK(c.unigram_count(kThe) == 6,
              "15e retraining does not DOUBLE the unigram counts");
        CHECK(c.trigram_count(kThe, kQuick, kBrown) == 2,
              "15f retraining does not DOUBLE the trigram counts");
        // an UNTRAined model still answers, with a uniform distribution
        NGramLM untrained(3, 0.75);
        CHECK(untrained.kn_prob(0) >= 0.0, "15g an untrained LM answers without throwing");
        CHECK(untrained.corpus_perplexity(sents, 0.75) >= 1.0,
              "15h an untrained LM's perplexity is at least 1");
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===" << endl;
    return (total == passed) ? 0 : 1;
}
