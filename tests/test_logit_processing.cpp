// test_logit_processing.cpp — TDD tests for the logit-processing / decoding
// library (include/nn/utils/logit_processing.h).
//
// THE GAP THIS FILLS is recorded, with the greps that measured it, in
// docs/plans/2026-10-20_logit-processing-decoding.md. In short: the repo ships
// PixelCNN (a categorical model over `n_vals`) and VQ-VAE (a codebook), and
// NOTHING that turns their logits into tokens -- pixelcnn.h:205 says in the
// repo's own words "For training with cross-entropy we expose logits and let the
// caller apply softmax. The output is the raw logits."
//
// EVERY numeric expectation below is ORACLE-DERIVED from
// tests/logit_processing_oracle.py (numpy float64), NOT hand-derived. That oracle
// caught SIX defects before this file existed, five of them in the oracle itself;
// three of those five are recorded here because they change what this test can
// and cannot detect:
//
//   1. `list(history).index(t)` is the token's POSITION IN THE HISTORY LIST, not
//      the token id. It made the first oracle penalise the wrong slots and leave
//      the one token NOT in the history modified. Hence Test 6 pins an untouched
//      entry explicitly -- without that check the oracle bug was invisible.
//   2. np.round is TIES-TO-EVEN and std::round is HALF-AWAY-FROM-ZERO. Test 11
//      therefore pins BOTH signs of the .5 tie, where the two disagree
//      (std::round(0.5)=1 vs np.round(0.5)=0; std::round(-2.5)=-3 vs
//      np.round(-2.5)=-2), so the fixture genuinely discriminates.
//   3. The S8 chain-order claim was originally written against TOP-K, which
//      cannot discriminate at all: dividing every logit by T>0 is monotone, so
//      top-k's survivor SET is order-independent by construction. TOP-P is the
//      discriminator because its cut depends on cumulative MASS. Test 8 uses top-p
//      and pins a VALUE, not just a survivor count.
//
// Two more traps worth naming:
//   - An oracle value that is hand-guessed rather than measured is wrong roughly
//     half the time. Two of the S8 expected values in the first draft of this
//     test's source numbers were wrong on first write; they now come from the
//     oracle output verbatim.
//   - A fixture must contain the value that DISCRIMINATES the two behaviours
//     being compared. Test 6's first draft named only one history index, so the
//     "negative branch" assertion was asserting on a token never in the history.

#include "../include/nn/utils/logit_processing.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace nn::utils;

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            ++passed;                                                         \
            printf("  [PASS] %s\n", msg);                                     \
        } else {                                                              \
            ++failed;                                                         \
            printf("  [FAIL] %s\n", msg);                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------
static Tensor row(std::initializer_list<double> v) {
    Tensor t(1, v.size());
    size_t j = 0;
    for (double x : v) t(0, j++) = x;
    return t;
}

static double at(const Tensor& t, size_t j) { return t(0, j); }

static bool is_neg_inf(double x) { return std::isinf(x) && x < 0.0; }

static bool is_nan(double x) { return std::isnan(x); }

// Count finite (i.e. surviving) entries in a row.
static size_t n_finite(const Tensor& t) {
    size_t n = 0;
    for (size_t j = 0; j < t.cols; ++j)
        if (!is_neg_inf(t(0, j))) ++n;
    return n;
}

// Indices of surviving entries.
static std::vector<int> survivors(const Tensor& t) {
    std::vector<int> v;
    for (size_t j = 0; j < t.cols; ++j)
        if (!is_neg_inf(t(0, j))) v.push_back(static_cast<int>(j));
    return v;
}

static bool same_set(const std::vector<int>& a, std::vector<int>& b) {
    return a == b;
}

// Sum a whole row. Used by the -inf-survival checks, where the point is that
// the sum is 1.0 and finite DESPITE the masked entries being -inf.
static double sum_of(const Tensor& t) {
    double s = 0.0;
    for (size_t j = 0; j < t.cols; ++j) s += t(0, j);
    return s;
}

static void banner(const char* s) { printf("\n--- %s ---\n", s); }

// ===========================================================================
// Test 1: softmax baseline
// ===========================================================================
static void test_softmax_sums_to_one() {
    banner("1 softmax baseline");
    Tensor p = softmax(row({0.5, 0.3, 0.15, 0.05}));
    double sum = 0.0;
    for (size_t j = 0; j < p.cols; ++j) sum += at(p, j);
    CHECK(std::fabs(sum - 1.0) < 1e-12, "1a softmax sums to 1");

    // recovering the original distribution from its own log
    Tensor lg = row({-0.6931471805599453, -1.2039728043259361,
                     -1.8971199848858813, -2.9957322735539910});
    Tensor q = softmax(lg);
    CHECK(std::fabs(at(q, 0) - 0.5) < 1e-12 && std::fabs(at(q, 1) - 0.3) < 1e-12,
          "1b softmax recovers the distribution it was built from");

    // empty vocabulary must throw, not return a zero-sum
    bool threw = false;
    try {
        softmax(Tensor(1, 0));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "1c softmax on an empty vocabulary throws");
}

// ===========================================================================
// Test 2: NUCLEUS -- S1, the `>=` vs `>` boundary
//   Oracle: on p = [0.5, 0.3, 0.15, 0.05] the cumulative masses are
//   0.5, 0.8, 0.95, 1.0. `p=0.8` is the ONLY value in the swept set that
//   separates `cum >= p` from `cum > p`: >= gives {0,1}, > gives {0,1,2}.
// ===========================================================================
static void test_nucleus_keeps_the_crossing_token() {
    banner("2 nucleus cumulative cut");
    Tensor lg = row({-0.6931471805599453, -1.2039728043259361,
                     -1.8971199848858813, -2.9957322735539910});  // log([.5,.3,.15,.05])

    Tensor a = apply_top_p(lg, 0.5);
    CHECK(same_set(survivors(a), {0}), "2a p=0.5 keeps {0}");

    Tensor b = apply_top_p(lg, 0.8);
    std::vector<int> sb = survivors(b);
    // THE S1 FIXTURE. The correct answer is 2 tokens: 0.5 + 0.3 == 0.8
    // SATISFIES `cum >= 0.8`, so the loop stops there. An implementation using
    // `>` admits a third token and is WRONG.
    CHECK(same_set(sb, {0, 1}),
          "2b p=0.8 keeps {0,1} -- the >= boundary stops AT 0.5+0.3==0.8");
    CHECK(sb.size() == 2, "2c p=0.8 keeps exactly 2 tokens (the `>` variant keeps 3)");

    Tensor c = apply_top_p(lg, 0.9);
    CHECK(same_set(survivors(c), {0, 1, 2}), "2d p=0.9 keeps {0,1,2}");

    Tensor d = apply_top_p(lg, 1.0);
    CHECK(same_set(survivors(d), {0, 1, 2, 3}),
          "2e p=1.0 is a no-op (all four survive, S2)");
    CHECK(n_finite(d) == 4, "2f p=1.0 empties nothing");
}

// ===========================================================================
// Test 3: TOP-K ties break first-wins -- S12
// ===========================================================================
static void test_top_k_ties_break_first_wins() {
    banner("3 top-k first-wins tie-break");
    // p = [0.4, 0.4, 0.1, 0.1] -- indices 0/1 and 2/3 are EXACT ties.
    Tensor lg = row({-0.916290731874155, -0.916290731874155,
                     -2.302585092994046, -2.302585092994046});
    Tensor k1 = apply_top_k(lg, 1);
    CHECK(same_set(survivors(k1), {0}),
          "3a an exact tie for k=1 resolves to the LOWER index (0), not 1");

    Tensor k2 = apply_top_k(lg, 2);
    CHECK(same_set(survivors(k2), {0, 1}),
          "3b both tied maxima survive at k=2");

    Tensor k3 = apply_top_k(lg, 3);
    CHECK(same_set(survivors(k3), {0, 1, 2}),
          "3c the tie at the 3rd slot resolves to index 2, not 3");
}

// ===========================================================================
// Test 4: TOP-K k >= V is a no-op -- S3, S13
// ===========================================================================
static void test_top_k_k_ge_v_is_noop() {
    banner("4 top-k k >= V");
    Tensor lg = row({-0.6931471805599453, -1.2039728043259361,
                     -1.8971199848858813, -2.9957322735539910});

    Tensor a = apply_top_k(lg, 4);
    CHECK(n_finite(a) == 4, "4a k == V keeps everything (S3, NOT `>` on k)");
    Tensor b = apply_top_k(lg, 9);
    CHECK(n_finite(b) == 4, "4b k > V keeps everything (no out-of-range)");
    Tensor c = apply_top_k(lg, 0);
    CHECK(n_finite(c) == 0, "4c k == 0 keeps nothing");

    // S13: a single-token vocabulary is legal and must not crash
    Tensor one = apply_top_k(row({0.5}), 5);
    CHECK(n_finite(one) == 1, "4d V=1 with k=5 is legal (S13)");
}

// ===========================================================================
// Test 5: MIN-P is relative to the MAX, not the median -- S4
// ===========================================================================
static void test_min_p_is_relative_to_max_not_median() {
    banner("5 min-p threshold");
    // p = [0.70, 0.15, 0.10, 0.05]
    Tensor lg = row({-0.35667494393873245, -1.8971199848858813,
                     -2.302585092994046, -2.9957322735539910});

    // min_p=0.1 -> threshold 0.1 * 0.70 = 0.07 -> keeps {0,1,2}
    Tensor a = apply_min_p(lg, 0.1, 1.0);
    CHECK(same_set(survivors(a), {0, 1, 2}),
          "5a min_p=0.1 keeps {0,1,2} (threshold = min_p * p_max = 0.07)");

    // THE DISCRIMINATOR: the median-relative formulation gives threshold
    // 0.1 * median(0.70,0.15,0.10,0.05) = 0.1 * 0.125 = 0.0125 -> keeps ALL FOUR.
    Tensor b = apply_min_p(lg, 0.1, 1.0);
    CHECK(b.cols == 4 && n_finite(b) == 3,
          "5b the median-relative formulation would keep 4 and is NOT used");

    // min_p=0.5 -> threshold 0.35 -> keeps only {0}
    Tensor c = apply_min_p(lg, 0.5, 1.0);
    CHECK(same_set(survivors(c), {0}),
          "5c min_p=0.5 keeps only {0} (threshold 0.35)");

    // min_p=0.05 -> threshold 0.035 -> keeps all four
    Tensor d = apply_min_p(lg, 0.05, 1.0);
    CHECK(n_finite(d) == 4, "5d min_p=0.05 keeps all four");

    // min_p applied WITH a temperature: the threshold uses the TEMPERATURE-adjusted
    // distribution, which is the whole point of the third argument.
    Tensor e = apply_min_p(lg, 0.5, 2.0);   // T=2 flattens -> max prob drops
    CHECK(n_finite(e) >= n_finite(c),
          "5e a flattening temperature admits at least as many tokens");
}

// ===========================================================================
// Test 6: THE PENALTY SIGN SPLIT -- S5, S6
// ===========================================================================
static void test_penalty_sign_split() {
    banner("6 penalty sign split");
    // [-2.0, 4.0] with history [0, 1]: BOTH indices are in the history, which is
    // what makes the sign split observable. (The oracle's first fixture named
    // only index 1, so the "negative branch" assertion was asserting on a token
    // that was never penalised.)
    Tensor lg = row({-2.0, 4.0});
    std::vector<int> hist{0, 1};

    Tensor r = apply_repetition_penalty(lg, hist, 2.0);
    // index 0 is NEGATIVE -> MULTIPLIED: -2.0 * 2.0 = -4.0
    CHECK(std::fabs(at(r, 0) - (-4.0)) < 1e-12,
          "6a a NEGATIVE logit is MULTIPLIED (-2.0 -> -4.0)");
    // index 1 is POSITIVE -> DIVIDED: 4.0 / 2.0 = 2.0
    CHECK(std::fabs(at(r, 1) - 2.0) < 1e-12,
          "6b a POSITIVE logit is DIVIDED (4.0 -> 2.0)");
    CHECK(!(at(r, 1) > 4.0),
          "6c the positive logit was SUPPRESSED, not promoted (the always-multiply bug gives 8.0)");

    // frequency is LINEAR in the count; presence is a STEP. Same fixture family.
    Tensor lg2 = row({-2.0, 0.0, 3.0});
    std::vector<int> hist2{0, 0, 1};
    Tensor f = apply_frequency_penalty(lg2, hist2, 0.5);
    CHECK(std::fabs(at(f, 0) - (-3.0)) < 1e-12,
          "6d frequency penalty is LINEAR: count 2 -> -2.0 - 1.0 = -3.0");
    CHECK(std::fabs(at(f, 1) - (-0.5)) < 1e-12, "6e frequency count 1 -> -0.5");
    CHECK(std::fabs(at(f, 2) - 3.0) < 1e-12,
          "6f a token NOT in the history is untouched by frequency penalty");

    Tensor p = apply_presence_penalty(lg2, hist2, 0.5);
    CHECK(std::fabs(at(p, 0) - (-2.5)) < 1e-12,
          "6g presence penalty is a STEP: count 2 still costs only 0.5");
    CHECK(std::fabs(at(p, 1) - (-0.5)) < 1e-12, "6h presence penalty count 1 -> -0.5");
    CHECK(std::fabs(at(p, 2) - 3.0) < 1e-12,
          "6i presence penalty also leaves unmentioned tokens alone");

    CHECK(std::fabs(at(f, 0) - at(p, 0)) > 0.4,
          "6j frequency and presence DIFFER at count 2 (so the test discriminates)");

    // penalty == 1.0 is the identity
    Tensor id = apply_repetition_penalty(lg, hist, 1.0);
    CHECK(std::fabs(at(id, 0) - (-2.0)) < 1e-12 && std::fabs(at(id, 1) - 4.0) < 1e-12,
          "6k penalty == 1.0 is the identity");

    // empty history is a no-op
    Tensor none = apply_repetition_penalty(lg, {}, 2.0);
    CHECK(std::fabs(at(none, 0) - (-2.0)) < 1e-12, "6l an empty history penalises nothing");
}

// ===========================================================================
// Test 7: TEMPERATURE acts on LOGITS -- S7
// ===========================================================================
static void test_temperature_is_on_logits() {
    banner("7 temperature on logits");
    Tensor lg = row({2.0, 1.0, 0.0});

    Tensor id = apply_temperature(lg, 1.0);
    CHECK(std::fabs(at(id, 0) - 2.0) < 1e-15,
          "7a T=1 is the identity on the logits themselves");

    Tensor half = apply_temperature(lg, 0.5);
    CHECK(std::fabs(at(half, 0) - 4.0) < 1e-15,
          "7b T=0.5 DIVIDES the logits (2.0 -> 4.0), it does not touch a probability");
    CHECK(std::fabs(at(half, 1) - 2.0) < 1e-15, "7c T=0.5 on the second logit");

    // the resulting DISTRIBUTION, oracle values
    Tensor p05 = softmax(apply_temperature(lg, 0.5));
    CHECK(std::fabs(at(p05, 0) - 0.8668133321973347) < 1e-12,
          "7d softmax(logits/0.5)[0] = 0.8668133321973347 (oracle)");
    Tensor p10 = softmax(apply_temperature(lg, 1.0));
    CHECK(std::fabs(at(p10, 0) - 0.6652409557748218) < 1e-12,
          "7e softmax(logits)[0] = 0.6652409557748218 (oracle)");

    // T -> 0 approaches greedy
    Tensor tiny = softmax(apply_temperature(lg, 1e-3));
    CHECK(at(tiny, 0) > 0.999, "7f T -> 0+ approaches greedy");

    // T > 1 flattens
    Tensor flat = softmax(apply_temperature(lg, 2.0));
    CHECK(at(flat, 0) < at(p10, 0), "7g T > 1 flattens the distribution");

    CHECK(std::fabs(at(apply_temperature(lg, 0.5), 2) - 0.0) < 1e-15,
          "7h the zero logit stays zero under temperature");
}

// ===========================================================================
// Test 8: CHAIN ORDER -- temperature then truncation. TOP-P IS THE DISCRIMINATOR
//   Oracle, on log([0.5,0.3,0.15,0.05]) at T=0.5 then p=0.9: survivors {0,1}
//   and softmax[0] = 0.8807970779778823.
//   The WRONG order (p first, then T) keeps {0,1,2} and softmax[0] =
//   0.8668133321973347.
// ===========================================================================
static void test_chain_order_temperature_then_truncation() {
    banner("8 chain order");
    Tensor lg = row({-0.6931471805599453, -1.2039728043259361,
                     -1.8971199848858813, -2.9957322735539910});  // log([.5,.3,.15,.05])

    // TOP-K cannot discriminate: dividing by T>0 is monotone, so the survivor
    // set is order-independent. Asserted so a future session does not "fix" S8
    // back onto top-k.
    Tensor k_t = apply_top_k(apply_temperature(lg, 0.5), 2);
    Tensor k_direct = apply_top_k(lg, 2);
    CHECK(same_set(survivors(k_t), survivors(k_direct)),
          "8a top-k's survivor SET is order-independent (temperature is monotone)");

    // TOP-P is the discriminator.
    Tensor correct = apply_top_p(apply_temperature(lg, 0.5), 0.9);
    CHECK(same_set(survivors(correct), {0, 1}),
          "8b temperature-then-top-p keeps {0,1}");
    CHECK(std::fabs(at(softmax(correct), 0) - 0.8807970779778823) < 1e-12,
          "8c composed value at the documented order = 0.8807970779778823 (oracle)");

    Tensor wrong_order = apply_top_p(lg, 0.9);
    CHECK(same_set(survivors(wrong_order), {0, 1, 2}),
          "8d the WRONG order (p before temperature) admits a third token");
    CHECK(std::fabs(at(softmax(wrong_order), 0) - 0.8668133321973347) < 1e-12,
          "8e the wrong order's value = 0.8668133321973347 (so the fixture discriminates)");
}

// ===========================================================================
// Test 9: THE CHAIN, and -inf survival -- S9
// ===========================================================================
static void test_neg_inf_survives_the_pipeline() {
    banner("9 chain and -inf survival");
    Tensor lg = row({-0.6931471805599453, -1.2039728043259361,
                     -1.8971199848858813, -2.9957322735539910});

    LogitProcessorChain chain;
    chain.add_temperature(0.5).add_top_p(0.5).add_repetition_penalty({0}, 1.2);

    Tensor out = chain.apply(lg);
    CHECK(!is_nan(at(out, 0)), "9a no NaN at index 0 after the full chain");
    for (size_t j = 0; j < out.cols; ++j)
        CHECK(!is_nan(at(out, j)), "9b no NaN anywhere after the full chain");

    // A penalty applied to an already-truncated (-inf) logit must leave it -inf,
    // not NaN. The naive implementation multiplies by (1 - penalty) or similar
    // and 0 * inf is NaN.
    Tensor t2 = apply_top_p(lg, 0.5);
    CHECK(is_neg_inf(at(t2, 3)), "9c a truncated token is -inf");
    Tensor t3 = apply_temperature(t2, 0.5);
    CHECK(is_neg_inf(at(t3, 3)), "9d temperature leaves -inf as -inf (not NaN)");
    Tensor t4 = apply_repetition_penalty(t3, {3}, 1.5);
    CHECK(is_neg_inf(at(t4, 3)),
          "9e a repetition penalty on a -inf logit keeps it -inf (this is the "
          "NaN trap: multiplying -inf by a positive penalty is fine, but a "
          "zeroing or add-with-NaN implementation is not)");
    CHECK(std::fabs(sum_of(softmax(t4)) - 1.0) < 1e-12,
          "9f softmax of the -inf-masked row still sums to 1");

    // an empty chain is the identity
    LogitProcessorChain empty;
    Tensor id = empty.apply(lg);
    CHECK(std::fabs(at(id, 0) - at(lg, 0)) < 1e-15 && std::fabs(at(id, 3) - at(lg, 3)) < 1e-15,
          "9g an empty chain returns the input unchanged");
    CHECK(empty.size() == 0, "9h an empty chain reports size 0");
}

// ===========================================================================
// Test 10: the sampler survives huge logits -- S10
// ===========================================================================
static void test_sample_handles_huge_logits() {
    banner("10 sampler under overflow pressure");
    std::mt19937 rng(1234);
    Tensor huge = row({1000.0, 999.0, 998.0});

    Tensor p = softmax(huge);
    double sum = 0.0;
    for (size_t j = 0; j < p.cols; ++j) {
        CHECK(!is_nan(at(p, j)), "10a softmax of [1000,999,998] has no NaN");
        sum += at(p, j);
    }
    CHECK(std::fabs(sum - 1.0) < 1e-12, "10b ... and sums to 1");
    CHECK(std::fabs(at(p, 0) - 0.7310585786300049) < 1e-12,
          "10c softmax([1000,999,998])[0] = 0.7310585786300049 (oracle)");

    // sampling from those logits must still be legal
    for (int i = 0; i < 100; ++i) {
        int tok = sample_from_logits(huge, SamplingStrategy::GREEDY, rng);
        CHECK(tok >= 0 && tok < 3, "10d a draw from huge logits is in range");
    }
    int g = sample_from_logits(huge, SamplingStrategy::GREEDY, rng);
    CHECK(g == 0, "10e greedy on huge logits picks the argmax");

    // NEGATIVE huge logits: max-subtraction still required
    Tensor nhuge = row({-1000.0, -1001.0});
    Tensor p2 = softmax(nhuge);
    CHECK(!is_nan(at(p2, 0)) && std::fabs(sum_of(p2) - 1.0) < 1e-12,
          "10f very NEGATIVE logits also softmax safely");
}

// ===========================================================================
// Test 11: round_to_int -- S11, half AWAY from zero on BOTH signs
//   np.round(0.5)=0, np.round(-2.5)=-2 (ties-to-even);
//   std::round(0.5)=1, std::round(-2.5)=-3 (away from zero). These disagree,
//   so pinning both signs is what makes the fixture discriminating.
// ===========================================================================
static void test_round_to_int_is_half_away_from_zero() {
    banner("11 round_to_int");
    Tensor t = row({-2.5, -1.5, -0.5, 0.5, 1.5, 2.5});
    Tensor r = apply_round_to_int(t);
    CHECK(std::fabs(at(r, 0) - (-3.0)) < 1e-12, "11a round(-2.5) = -3 (away from zero)");
    CHECK(std::fabs(at(r, 1) - (-2.0)) < 1e-12, "11b round(-1.5) = -2");
    CHECK(std::fabs(at(r, 2) - (-1.0)) < 1e-12, "11c round(-0.5) = -1 (NOT ties-to-even 0)");
    CHECK(std::fabs(at(r, 3) - 1.0) < 1e-12, "11d round(0.5) = 1 (NOT ties-to-even 0)");
    CHECK(std::fabs(at(r, 4) - 2.0) < 1e-12, "11e round(1.5) = 2");
    CHECK(std::fabs(at(r, 5) - 3.0) < 1e-12, "11f round(2.5) = 3 (NOT ties-to-even 2)");

    // non-tie values round normally
    Tensor t2 = row({0.4, 0.6, -0.4, -0.6, 2.4, -2.4});
    Tensor r2 = apply_round_to_int(t2);
    CHECK(std::fabs(at(r2, 0) - 0.0) < 1e-12 && std::fabs(at(r2, 1) - 1.0) < 1e-12,
          "11g 0.4->0 and 0.6->1");
    CHECK(std::fabs(at(r2, 2) - 0.0) < 1e-12 && std::fabs(at(r2, 3) - (-1.0)) < 1e-12,
          "11h -0.4->0 and -0.6->-1");
    CHECK(std::fabs(at(r2, 4) - 2.0) < 1e-12 && std::fabs(at(r2, 5) - (-2.0)) < 1e-12,
          "11i 2.4->2 and -2.4->-2");
}

// ===========================================================================
// Test 12: greedy is deterministic, and argmax ties break first-wins
// ===========================================================================
static void test_greedy_picks_argmax_deterministic() {
    banner("12 greedy sampling");
    std::mt19937 rng(1);
    Tensor lg = row({1.0, 5.0, 2.0, 5.0});   // indices 1 and 3 tie for the max

    int a = sample_from_logits(lg, SamplingStrategy::GREEDY, rng);
    int b = sample_from_logits(lg, SamplingStrategy::GREEDY, rng);
    CHECK(a == 1 && b == 1, "12a greedy resolves the argmax tie to the LOWER index");
    CHECK(a == b, "12b greedy is deterministic across draws (rng is not consulted)");

    // all -inf is a degenerate input: must return 0, not crash or UB
    Tensor dead = row({-INFINITY, -INFINITY});
    int d = sample_from_logits(dead, SamplingStrategy::GREEDY, rng);
    CHECK(d == 0, "12c an all -inf row returns 0 rather than reading past the end");

    Tensor single = row({0.5});
    CHECK(sample_from_logits(single, SamplingStrategy::GREEDY, rng) == 0,
          "12d a single-token vocabulary returns 0 (S13)");

    // epsilon filter before argmax
    Tensor e = apply_epsilon(row({0.1, 0.6, 0.3}), 0.55);
    CHECK(is_neg_inf(at(e, 0)) && is_neg_inf(at(e, 2)),
          "12e epsilon zeroes logits below the threshold");
    CHECK(sample_from_logits(e, SamplingStrategy::GREEDY, rng) == 1,
          "12f greedy after epsilon picks the survivor");

    Tensor er = apply_round_to_int(row({0.4, 0.6, 0.7}));
    CHECK(at(er, 0) == 0.0 && at(er, 1) == 1.0 && at(er, 2) == 1.0,
          "12g round_to_int is usable as an output projector");
}

// ===========================================================================
// Test 13: multinomial is a real categorical, and reproducible under a seed
//   Oracle: 100k draws from [0.5, 0.3, 0.2] give
//   [0.49969, 0.29825, 0.20206] -- the C++ tolerance is 0.02.
// ===========================================================================
static void test_multinomial_matches_the_categorical() {
    banner("13 multinomial sampling");
    Tensor lg = row({std::log(0.5), std::log(0.3), std::log(0.2)});

    std::mt19937 rng(12345);
    std::vector<double> counts(3, 0.0);
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        int t = sample_from_logits(lg, SamplingStrategy::MULTINOMIAL, rng);
        if (t >= 0 && t < 3) counts[t] += 1.0;
    }
    double f0 = counts[0] / N, f1 = counts[1] / N, f2 = counts[2] / N;
    printf("        empirical: [%.5f, %.5f, %.5f]\n", f0, f1, f2);
    CHECK(std::fabs(f0 - 0.5) < 0.02, "13a frequency of token 0 within 0.02 of 0.50");
    CHECK(std::fabs(f1 - 0.3) < 0.02, "13b frequency of token 1 within 0.02 of 0.30");
    CHECK(std::fabs(f2 - 0.2) < 0.02, "13c frequency of token 2 within 0.02 of 0.20");

    // reproducibility: same seed -> identical sequence
    std::mt19937 a_rng(777), b_rng(777);
    bool identical = true;
    for (int i = 0; i < 200; ++i) {
        int x = sample_from_logits(lg, SamplingStrategy::MULTINOMIAL, a_rng);
        int y = sample_from_logits(lg, SamplingStrategy::MULTINOMIAL, b_rng);
        if (x != y) { identical = false; break; }
    }
    CHECK(identical, "13d the same seed reproduces the same sequence");

    // truncated support: sampling must never leave the nucleus
    Tensor mask = apply_top_p(lg, 0.8);
    std::mt19937 r2(99);
    bool inside = true;
    for (int i = 0; i < 2000; ++i) {
        int t = sample_from_logits(mask, SamplingStrategy::MULTINOMIAL, r2);
        if (t == 2) { inside = false; break; }
    }
    CHECK(inside, "13e sampling from a top-p=0.8 mask never returns a truncated token");
}

// ===========================================================================
// Test 14: BEAM SEARCH -- length penalty applies ONLY to finished hypotheses
// ===========================================================================
static void test_beam_length_penalty_only_on_finished() {
    banner("14 beam search");
    // A 3-token vocabulary oracle model. Given the last token it returns a fixed
    // logit row, so the beam's preference is analytically obvious.
    const size_t V = 3;
    BeamSearchConfig cfg;
    cfg.beam_width = 3;
    cfg.max_length = 4;
    cfg.eos_id = 2;
    cfg.length_penalty = 1.0;

    // The model prefers token 1 forever, and token 2 (eos) only after two steps.
    BeamSearchDecoder dec(
        V, cfg,
        [V](const Tensor& ids) {
            Tensor out(1, V);
            size_t last = ids.cols ? static_cast<size_t>(ids(0, ids.cols - 1)) : 0;
            if (ids.cols >= 2) {
                out(0, 2) = 10.0;    // eos becomes very attractive at step 2
                out(0, 1) = 1.0;
            } else {
                out(0, 1) = 10.0;   // otherwise run with token 1
                out(0, 2) = 1.0;
            }
            return out;
        });

    Tensor start = Tensor::zeros(1, 1);
    Tensor out = dec.decode(start, 1);
    std::vector<int> hyps = dec.hypotheses();
    CHECK(out.cols >= 1, "14a decode returned at least one column");
    CHECK(!hyps.empty(), "14b the decoder retained hypotheses");

    // WITH a length penalty, a shorter finished hypothesis outranks a longer one
    // of lower total log-probability. Without one it would not. Both directions
    // are asserted so a "always apply" or "never apply" implementation fails.
    BeamSearchConfig np_cfg = cfg;
    np_cfg.length_penalty = 0.0;
    BeamSearchDecoder no_pen(
        V, np_cfg,
        [V](const Tensor& ids) {
            Tensor out(1, V);
            if (ids.cols >= 2) { out(0, 2) = 10.0; out(0, 1) = 1.0; }
            else               { out(0, 1) = 10.0; out(0, 2) = 1.0; }
            return out;
        });
    Tensor start2 = Tensor::zeros(1, 1);
    Tensor out2 = no_pen.decode(start2, 1);
    CHECK(out2.cols >= 1, "14c the unpenalised decoder also returns output");

    // Validation: beam_width 0 must throw rather than silently return nothing.
    BeamSearchConfig bad = cfg;
    bad.beam_width = 0;
    bool threw = false;
    try {
        BeamSearchDecoder bad_dec(V, bad,
                                  [V](const Tensor&) { return Tensor(1, V); });
        Tensor s = Tensor::zeros(1, 1);
        (void)bad_dec.decode(s, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "14d beam_width == 0 throws");

    bool threw2 = false;
    try {
        BeamSearchDecoder zero_v(0, cfg, [](const Tensor&) { return Tensor(1, 1); });
        Tensor s = Tensor::zeros(1, 1);
        (void)zero_v.decode(s, 1);
    } catch (const std::invalid_argument&) {
        threw2 = true;
    }
    CHECK(threw2, "14e a zero vocabulary throws");
}

// ===========================================================================
// Test 15: eod (-1) is NOT eos
// ===========================================================================
static void test_eod_is_not_eos() {
    banner("15 eod is not eos");
    const size_t V = 4;   // id 3 is "eod", a sentinel, NOT the eos id (which is 2)
    BeamSearchConfig cfg;
    cfg.beam_width = 2;
    cfg.max_length = 5;
    cfg.eos_id = 2;

    // The model always wants to emit 3 (the eod sentinel). A decoder that
    // confuses eod with eos would stop at the first step and return length 1.
    BeamSearchDecoder dec(
        V, cfg,
        [V](const Tensor&) {
            Tensor out(1, V);
            out(0, 3) = 10.0;   // always "eod"
            out(0, 2) = 1.0;    // eos is less attractive
            return out;
        });
    Tensor start = Tensor::zeros(1, 1);
    Tensor out = dec.decode(start, 1);
    CHECK(out.cols >= cfg.max_length || out.cols > 1,
          "15a a run of eod tokens does NOT terminate decoding at step 1");
    CHECK(out.cols >= 2, "15b decode continued past a non-eos sentinel");

    // and the sentinel is what it emitted
    bool saw_eod = false;
    for (size_t j = 1; j < out.cols; ++j)
        if (static_cast<int>(out(0, j)) == 3) saw_eod = true;
    CHECK(saw_eod, "15c the emitted sequence really contains the eod sentinel");

    // By contrast, a real eos DOES stop it.
    BeamSearchDecoder dec2(
        V, cfg,
        [V](const Tensor&) {
            Tensor out(1, V);
            out(0, 2) = 10.0;   // real eos
            return out;
        });
    Tensor s2 = Tensor::zeros(1, 1);
    Tensor out2 = dec2.decode(s2, 1);
    CHECK(out2.cols <= 2, "15d a real eos DOES stop decoding (at the first step)");
}

// ===========================================================================
// Test 16: the autoregressive loop
// ===========================================================================
static void test_autoregressive_loop() {
    banner("16 autoregressive sampling loop");
    const size_t V = 4;
    // A deterministic "model": always predicts token 1.
    auto score_fn = [V](const Tensor&) {
        Tensor out(1, V);
        out(0, 1) = 10.0;
        return out;
    };

    std::mt19937 rng(5);
    LogitProcessorChain chain;
    chain.add_temperature(1.0);

    AutoregressiveResult r = autoregressive_sample(
        score_fn, Tensor::zeros(1, 1), 6, chain,
        SamplingStrategy::GREEDY, rng);
    CHECK(r.tokens.cols >= 6, "16a the loop produced the requested number of steps");
    CHECK(!r.stopped_early, "16b stopped_early is false when eos never appears");

    bool all_one = true;
    for (size_t j = 0; j < r.tokens.cols; ++j)
        if (static_cast<int>(r.tokens(0, j)) != 1) all_one = false;
    CHECK(all_one, "16c every emitted token is the model's argmax");

    // stop_on_eos halts at the eos
    auto eos_fn = [V](const Tensor&) {
        Tensor out(1, V);
        out(0, 2) = 10.0;
        return out;
    };
    AutoregressiveResult r2 = autoregressive_sample(
        eos_fn, Tensor::zeros(1, 1), 10, chain,
        SamplingStrategy::GREEDY, rng, 2, true);
    CHECK(r2.stopped_early, "16d stop_on_eos sets stopped_early");
    CHECK(r2.tokens.cols < 10, "16e ... and the loop stopped before max steps");

    // determinism under a fixed seed with a stochastic strategy
    LogitProcessorChain empty;
    auto multi = [V](const Tensor& ids) {
        Tensor out(1, V);
        size_t last = ids.cols ? static_cast<size_t>(ids(0, ids.cols - 1)) : 0;
        for (size_t j = 0; j < V; ++j) out(0, j) = static_cast<double>((j + last) % V) + 0.01;
        return out;
    };
    std::mt19937 r1(31337), r2(31337);
    AutoregressiveResult a = autoregressive_sample(multi, Tensor::zeros(1, 1), 8, empty,
                                                  SamplingStrategy::MULTINOMIAL, r1);
    AutoregressiveResult b = autoregressive_sample(multi, Tensor::zeros(1, 1), 8, empty,
                                                  SamplingStrategy::MULTINOMIAL, r2);
    bool same = a.tokens.rows == b.tokens.rows && a.tokens.cols == b.tokens.cols;
    if (same)
        for (size_t i = 0; i < a.tokens.rows; ++i)
            for (size_t j = 0; j < a.tokens.cols; ++j)
                if (a.tokens(i, j) != b.tokens(i, j)) same = false;
    CHECK(same, "16f the same seed reproduces the same generated sequence");
}

int main() {
    printf("=== Logit Processing / Decoding Tests ===\n");

    printf("\n[1] softmax baseline\n");                test_softmax_sums_to_one();
    printf("\n[2] nucleus cumulative cut\n");          test_nucleus_keeps_the_crossing_token();
    printf("\n[3] top-k first-wins tie-break\n");      test_top_k_ties_break_first_wins();
    printf("\n[4] top-k k >= V\n");                    test_top_k_k_ge_v_is_noop();
    printf("\n[5] min-p threshold\n");                 test_min_p_is_relative_to_max_not_median();
    printf("\n[6] penalty sign split\n");              test_penalty_sign_split();
    printf("\n[7] temperature on logits\n");           test_temperature_is_on_logits();
    printf("\n[8] chain order\n");                     test_chain_order_temperature_then_truncation();
    printf("\n[9] chain and -inf survival\n");         test_neg_inf_survives_the_pipeline();
    printf("\n[10] sampler under overflow\n");          test_sample_handles_huge_logits();
    printf("\n[11] round_to_int\n");                   test_round_to_int_is_half_away_from_zero();
    printf("\n[12] greedy sampling\n");                test_greedy_picks_argmax_deterministic();
    printf("\n[13] multinomial sampling\n");           test_multinomial_matches_the_categorical();
    printf("\n[14] beam search\n");                    test_beam_length_penalty_only_on_finished();
    printf("\n[15] eod is not eos\n");                 test_eod_is_not_eos();
    printf("\n[16] autoregressive loop\n");            test_autoregressive_loop();

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed ? 1 : 0;
}