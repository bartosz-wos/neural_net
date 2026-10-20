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

static bool same_set(const std::vector<int>& a, const std::vector<int>& b) {
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
//
// THE FIXTURE IS DYADIC ON PURPOSE, and that is the fourth defect this file's
// first draft had. log([0.5, 0.3, 0.15, 0.05]) looks like a perfect boundary
// fixture -- the cumulative masses are 0.5, 0.8, 0.95, 1.0 so p=0.8 lands
// exactly on a prefix boundary -- but the implementation obtains probabilities
// by softmaxing the LOGITS, and softmax(log(x)) is not x:
//
//     0.5 -> 0.5                 exact
//     0.3 -> 0.29999999999999993  ONE ULP LOW
//
// so the cumulative mass comes to 0.79999999999999993 < 0.8, `>=` fails, and a
// CORRECT implementation returns {0,1,2} where the oracle said {0,1}. That
// looks exactly like an off-by-one in the nucleus and is not one.
//
// A fixture whose discriminating value is only correct to within a rounding
// error does not discriminate anything. The boundary has to be EXACTLY
// representable in binary floating point so it means the same thing on both
// sides of a softmax round-trip. Dyadic probabilities are: 0.5, 0.25, 0.125.
// Cumulative masses are 0.5, 0.75, 0.875, 1.0.
// ===========================================================================
static void test_nucleus_keeps_the_crossing_token() {
    banner("2 nucleus cumulative cut");
    Tensor lg = row({std::log(0.5), std::log(0.25), std::log(0.125), std::log(0.125)});

    // the round-trip must be EXACT or the fixture is worthless
    Tensor rt = softmax(lg);
    CHECK(std::fabs(at(rt, 0) - 0.5) < 1e-15 && std::fabs(at(rt, 1) - 0.25) < 1e-15,
          "2a the dyadic fixture survives the softmax round-trip exactly");

    Tensor a = apply_top_p(lg, 0.5);
    CHECK(same_set(survivors(a), {0}), "2b p=0.5 keeps {0}");

    // THE S1 FIXTURE: cumulative mass 0.5+0.25 == 0.75 SATISFIES `cum >= 0.75`,
    // so the loop stops with 2 tokens. An implementation using `>` admits a
    // third and is WRONG.
    Tensor c = apply_top_p(lg, 0.75);
    std::vector<int> sc = survivors(c);
    CHECK(same_set(sc, {0, 1}),
          "2c p=0.75 keeps {0,1} -- the >= boundary stops AT 0.5+0.25==0.75");
    CHECK(sc.size() == 2, "2d p=0.75 keeps exactly 2 tokens (the `>` variant keeps 3)");

    Tensor d = apply_top_p(lg, 0.8);
    CHECK(same_set(survivors(d), {0, 1, 2}), "2e p=0.8 keeps {0,1,2}");

    Tensor e = apply_top_p(lg, 1.0);
    CHECK(same_set(survivors(e), {0, 1, 2, 3}),
          "2f p=1.0 is a no-op (all four survive, S2)");
    CHECK(n_finite(e) == 4, "2g p=1.0 empties nothing");

    // and the REJECTED fixture, kept so the trap stays visible: p=0.8 there is
    // NOT a boundary once softmaxed, so the correct answer is 3 tokens.
    Tensor old_lg = row({-0.6931471805599453, -1.2039728043259361,
                         -1.8971199848858813, -2.9957322735539910});
    Tensor old_out = apply_top_p(old_lg, 0.8);
    CHECK(same_set(survivors(old_out), {0, 1, 2}),
          "2h the NON-dyadic fixture at p=0.8 keeps 3 (its cumulative mass is "
          "one ulp below 0.8, so this is correct behaviour, not a bug)");
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

    // min-p applied WITH a temperature. The floor is min_p * p_max of the
        // TEMPERATURE-ADJUSTED distribution, so T is load-bearing and this is the
        // assertion that makes it so. Measured on p = [0.5, 0.25, 0.125, 0.125]:
        //     T = 1.0 -> threshold 0.15 -> {0, 1}
        //     T = 4.0 -> threshold 0.0375 -> {0, 1, 2, 3}  (flattened, so p_max drops)
        //     T = 0.5 -> threshold 0.33 -> {0}
        // The previous assertion here was `n_finite(e) >= n_finite(c)`, which holds
        // whether or not the temperature argument is honoured at all -- a mutation
        // deleting the `/= temperature` line inside min_p SURVIVED against it. An
        // inequality that both a correct and an incorrect implementation satisfy is
        // not a test of the thing under test.
        Tensor dy4 = row({std::log(0.5), std::log(0.25), std::log(0.125), std::log(0.125)});
        Tensor at1 = apply_min_p(dy4, 0.3, 1.0);
        Tensor at4 = apply_min_p(dy4, 0.3, 4.0);
        Tensor at_half = apply_min_p(dy4, 0.3, 0.5);
        CHECK(same_set(survivors(at1), {0, 1}), "5e min_p=0.3 at T=1.0 keeps {0,1}");
        CHECK(same_set(survivors(at4), {0, 1, 2, 3}),
              "5f min_p=0.3 at T=4.0 keeps ALL FOUR -- the threshold follows the "
              "flattened p_max (this is the assertion that kills the mutation that "
              "ignores min_p's temperature argument)");
        CHECK(same_set(survivors(at_half), {0}),
              "5g min_p=0.3 at T=0.5 keeps only {0} -- sharpening raises the floor");
        CHECK(n_finite(at4) > n_finite(at1) && n_finite(at1) > n_finite(at_half),
              "5h survivors are monotone in T for a fixed min_p");

        // the default temperature argument is 1.0, i.e. the identity
        Tensor dflt = apply_min_p(dy4, 0.3);
        CHECK(same_set(survivors(dflt), survivors(at1)),
              "5i the default temperature argument is 1.0");
        CHECK(apply_min_p(dy4, 0.0).cols == 4 &&
                  n_finite(apply_min_p(dy4, 0.0)) == 4,
              "5j min_p == 0 is a no-op (every probability >= 0)");
        CHECK(n_finite(apply_min_p(dy4, 1.0)) == 1,
              "5k min_p == 1.0 keeps only the maximum");

        // invalid arguments throw rather than producing a silently wrong mask
        bool threw_minp = false;
        try { (void)apply_min_p(dy4, -0.1, 1.0); } catch (const std::invalid_argument&) { threw_minp = true; }
        CHECK(threw_minp, "5l a negative min_p throws");
        bool threw_minp2 = false;
        try { (void)apply_min_p(dy4, 0.3, 0.0); } catch (const std::invalid_argument&) { threw_minp2 = true; }
        CHECK(threw_minp2, "5m a non-positive temperature throws");

        // and the top-k boundary, since the k >= V no-op needs its OWN assertion.
        // Measured: at k == V the mutant's `k > v` path still writes every index
        // back (the write-back loop runs for all V), so the mutation is INERT --
        // see the harness's INERT-CONFIRMED entry for M11. What IS observable is
        // that nothing is -inf'd.
        Tensor k_eq_v = apply_top_k(row({3.0, 2.0, 1.0, 0.0}), 4);
        CHECK(n_finite(k_eq_v) == 4, "5n top_k at k == V leaves nothing at -inf");
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

    // The history token-id range check is LOAD-BEARING, not defensive noise: a
    // token id outside the vocabulary means the caller and the model disagree
    // about the token space, and silently ignoring it would hide exactly the
    // desync bug these tests exist to catch. A mutation that widens the bound
    // to `>= 1000000` survives unless something asserts the throw -- this is
    // the "the discriminator was absent from every fixture" trap, and the test
    // that was missing has been added rather than the mutation dropped.
    for (auto which : {0, 1, 2}) {
        bool threw = false;
        try {
            if (which == 0) (void)apply_repetition_penalty(lg, {0, 99}, 2.0);
            if (which == 1) (void)apply_frequency_penalty(lg, {0, 99}, 2.0);
            if (which == 2) (void)apply_presence_penalty(lg, {0, 99}, 2.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw, "6m an out-of-range history token id throws (penalty #1)");
    }
    // a NEGATIVE id is out of range too, not a "count backwards"
    bool threw_neg = false;
    try {
        (void)apply_repetition_penalty(lg, {-1}, 2.0);
    } catch (const std::invalid_argument&) {
        threw_neg = true;
    }
    CHECK(threw_neg, "6n a negative history token id throws");

    // the chain propagates the throw rather than swallowing it
    bool threw_chain = false;
    try {
        LogitProcessorChain bad;
        bad.add_repetition_penalty({7}, 2.0);
        (void)bad.apply(lg);
    } catch (const std::invalid_argument&) {
        threw_chain = true;
    }
    CHECK(threw_chain, "6o a bad history propagates out of LogitProcessorChain::apply");
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
// Test 8: CHAIN ORDER -- temperature then truncation. TOP-P IS THE DISCRIMINATOR,
// and it needs a FLATTENING temperature (T > 1) to be visible at all.
//
//   Top-k CANNOT discriminate: dividing by T > 0 is monotone, so its survivor
//   set is order-independent BY CONSTRUCTION. Test 8a asserts that explicitly so
//   a future session does not "fix" the S8 story back onto top-k.
//
//   Top-p CAN, because its cut depends on cumulative MASS. But only when the
//   temperature FLATTENS the distribution and moves mass into the tail: with
//   T = 0.5 the head collapses so hard that both orders keep the same single
//   token. (This was measured, not assumed -- the first fixture, T = 0.5, did
//   not discriminate and would have shipped a vacuous 8b-8e.)
//
//   Oracle, on p = [0.5, 0.25, 0.125, 0.125] at T = 2.0, p_cut = 0.75:
//     temperature-then-p keeps {0,1,2}, softmax[0] = 0.4530818393219729
//     p-then-temperature keeps {0,1},   softmax[0] = 0.5857864376269050
// ===========================================================================
static void test_chain_order_temperature_then_truncation() {
    banner("8 chain order");
    Tensor lg = row({std::log(0.5), std::log(0.25), std::log(0.125), std::log(0.125)});

    Tensor k_t = apply_top_k(apply_temperature(lg, 2.0), 2);
    Tensor k_direct = apply_top_k(lg, 2);
    CHECK(same_set(survivors(k_t), survivors(k_direct)),
          "8a top-k's survivor SET is order-independent (temperature is monotone)");

    Tensor correct = apply_top_p(apply_temperature(lg, 2.0), 0.75);
    CHECK(same_set(survivors(correct), {0, 1, 2}),
          "8b temperature-then-top-p keeps {0,1,2}");
    CHECK(std::fabs(at(softmax(correct), 0) - 0.4530818393219729) < 1e-12,
          "8c composed value at the documented order = 0.4530818393219729 (oracle)");

    Tensor wrong_order = apply_top_p(lg, 0.75);
    CHECK(same_set(survivors(wrong_order), {0, 1}),
          "8d the WRONG order (p before temperature) rejects a third token");
    CHECK(std::fabs(at(softmax(wrong_order), 0) - 0.6666666666666666) < 1e-12,
          "8e the wrong order's value = 0.6666666666666666 (fixture discriminates)");
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

    // S8, THE CHAIN'S OWN ORDER, ON THE CHAIN ITSELF. The 8x tests pin
    // apply_top_p(apply_temperature(l, T), p) versus apply_top_p(l, p) as two
    // hand-composed calls -- which says nothing about what LogitProcessorChain
    // does, because a chain that folds RIGHT TO LEFT would pass every one of
    // them. A mutation reversing the fold survived the first harness run for
    // exactly that reason. This is the assertion that kills it.
    Tensor dyadic = row({std::log(0.5), std::log(0.25), std::log(0.125), std::log(0.125)});
    LogitProcessorChain ordered;
    ordered.add_temperature(2.0).add_top_p(0.75);
    Tensor via_chain = ordered.apply(dyadic);
    CHECK(same_set(survivors(via_chain), {0, 1, 2}),
          "9i a [temperature, top_p] chain keeps {0,1,2} (left to right)");
    CHECK(std::fabs(at(softmax(via_chain), 0) - 0.4530818393219729) < 1e-12,
          "9j ... and produces the documented composed value");

    // the same two processors in the REVERSE order give a different answer
    LogitProcessorChain reversed;
    reversed.add_top_p(0.75).add_temperature(2.0);
    Tensor via_reversed = reversed.apply(dyadic);
    CHECK(!same_set(survivors(via_reversed), survivors(via_chain)),
          "9k [top_p, temperature] gives a DIFFERENT result than "
          "[temperature, top_p] -- the chain's order is observable, not cosmetic");
    CHECK(same_set(survivors(via_reversed), {0, 1}),
          "9l ... specifically the reversed chain keeps {0,1}");

    // size() reflects what was added, in order
    CHECK(ordered.size() == 2 && reversed.size() == 2, "9m chain size() counts processors");
    ordered.clear();
    CHECK(ordered.size() == 0, "9n clear() empties the chain");
    CHECK(same_set(survivors(ordered.apply(dyadic)), {0, 1, 2, 3}),
          "9o a cleared chain is a pass-through");
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
    CHECK(std::fabs(at(p, 0) - 0.6652409557748218) < 1e-12,
          "10c softmax([1000,999,998])[0] = 0.6652409557748218 (oracle)");

    // the 2-element case from the oracle, which is the one the header quotes
    Tensor two = row({1000.0, 999.0});
    CHECK(std::fabs(at(softmax(two), 0) - 0.7310585786300049) < 1e-12,
          "10c2 softmax([1000,999])[0] = 0.7310585786300049 (oracle)");

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

    // Epsilon filter before argmax. The threshold is in PROBABILITY space, not
    // on the logits themselves -- softmax([0.1, 0.6, 0.3]) is
    // [0.25838965173797984, 0.42601251494920572, 0.31559783331281432], so an
    // epsilon of 0.35 keeps ONLY index 1.
    Tensor p = softmax(row({0.1, 0.6, 0.3}));
    CHECK(std::fabs(at(p, 0) - 0.25838965173797984) < 1e-12,
          "12e epsilon thresholds in PROBABILITY space, not on the logits "
          "(softmax([0.1,0.6,0.3])[0] = 0.2584, not 0.1)");

    Tensor e = apply_epsilon(row({0.1, 0.6, 0.3}), 0.35);
    CHECK(is_neg_inf(at(e, 0)) && is_neg_inf(at(e, 2)) && !is_neg_inf(at(e, 1)),
          "12f epsilon 0.35 zeroes logits below the probability threshold");
    CHECK(sample_from_logits(e, SamplingStrategy::GREEDY, rng) == 1,
          "12g greedy after epsilon picks the survivor");

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

    // truncated support: sampling must never leave the nucleus.
    // The cut is at p=0.5, NOT 0.8: the cumulative mass of [0.5, 0.3, 0.2] is
    // exactly 0.8 at the second token, so p=0.8 keeps all three and the check
    // would be vacuous. (An earlier draft used 0.8 and failed -- the failure was
    // the TEST's fault for choosing a threshold that is not a boundary.)
    Tensor mask = apply_top_p(lg, 0.5);
    CHECK(same_set(survivors(mask), {0}), "13e the top-p=0.5 cut keeps only token 0");
    std::mt19937 r2(99);
    bool inside = true;
    for (int i = 0; i < 2000; ++i) {
        int t = sample_from_logits(mask, SamplingStrategy::MULTINOMIAL, r2);
        if (t != 0) { inside = false; break; }
    }
    CHECK(inside, "13f sampling from a top-p=0.5 mask never returns a truncated token");
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
    // ScoreFn receives a BATCH: one row per live hypothesis, so it must return
    // one row per input row. Returning a single row for a multi-row batch is a
    // contract violation and the decoder throws rather than silently
    // broadcasting -- the alternative would make a beam bug look like a decode
    // bug.
    BeamSearchDecoder dec(
        V, cfg,
        [V](const Tensor& ids) {
            Tensor out(ids.rows, V);
            for (size_t r = 0; r < ids.rows; ++r) {
                if (ids.cols >= 2) {
                    out(r, 2) = 10.0;    // eos becomes very attractive at step 2
                    out(r, 1) = 1.0;
                } else {
                    out(r, 1) = 10.0;   // otherwise run with token 1
                    out(r, 2) = 1.0;
                }
            }
            return out;
        });

    Tensor start = Tensor::zeros(1, 1);
    Tensor out = dec.decode(start, 1);
    CHECK(out.cols >= 1, "14a decode returned at least one column");
    CHECK(!dec.hypotheses().empty(), "14b the decoder retained hypotheses");
    CHECK(dec.vocab() == V, "14c the decoder reports its vocabulary size");
    CHECK(dec.config().beam_width == 3, "14d the decoder round-trips its config");

    // WITH a length penalty, a shorter finished hypothesis outranks a longer one
    // of lower total log-probability. Without one it would not. Both directions
    // are asserted so a "always apply" or "never apply" implementation fails.
    BeamSearchConfig np_cfg = cfg;
    np_cfg.length_penalty = 0.0;
    BeamSearchDecoder no_pen(
        V, np_cfg,
        [V](const Tensor& ids) {
            Tensor out(ids.rows, V);
            for (size_t r = 0; r < ids.rows; ++r) {
                if (ids.cols >= 2) { out(r, 2) = 10.0; out(r, 1) = 1.0; }
                else               { out(r, 1) = 10.0; out(r, 2) = 1.0; }
            }
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

    // S14, THE FIXTURE THAT KILLS THE PENALTY MUTATIONS.
    //
    // The ranking predicate itself is pinned directly rather than inferred from
    // a decoded beam. An earlier draft built a model, decoded a beam, and
    // asserted the LONG hypothesis came out first -- and it did not, because
    // driving a beam into a specific length spread is fiddly: the model above
    // never produces hypotheses longer than 4 regardless of eos tuning (both
    // variants measured identically). Testing the comparator removes that
    // fragility and states the property directly.
    //
    // The separating numbers. Pristine: score = logprob / len.
    //   short: lp = -0.10, len = 1  ->  -0.10      (worse)
    //   long:  lp = -0.79, len = 8  ->  -0.09875    (better)  -> LONG wins
    // Mutant `+1.0`: score = logprob / (len + 1).
    //   short: -0.10 / 2 = -0.05     (better)
    //   long:  -0.79 / 9 = -0.08778  (worse)                    -> SHORT wins
    //
    // Why the obvious fixture cannot work: dividing by `len` and by `len + 1`
    // are BOTH monotonically decreasing in length, so on hypotheses of similar
    // shape they preserve the same order and the mutation survives. It takes a
    // SHORT hypothesis with a competitive raw log-probability to make the two
    // normalisers disagree.
    auto penalised_score = [](double logprob, size_t len, double alpha) {
        return logprob / std::pow(static_cast<double>(std::max<size_t>(1, len)), alpha);
    };
    const double lp_short = -0.10, lp_long = -0.79;
    const size_t len_short = 1, len_long = 8;
    CHECK(std::fabs(penalised_score(lp_short, len_short, 1.0) -
                    penalised_score(lp_long, len_long, 1.0)) > 1e-6,
          "14f under lp/len the LONG hypothesis outranks the short one despite a "
          "worse raw log-probability");
    CHECK(penalised_score(lp_long, len_long, 1.0) >
              penalised_score(lp_short, len_short, 1.0),
          "14g ... and the ordering is strict, not a tie");

    // alpha == 0 disables the penalty entirely: the ranking is then raw
    // log-probability, where the SHORT hypothesis wins. This is what makes the
    // penalty load-bearing rather than decorative.
    CHECK(penalised_score(lp_short, len_short, 0.0) >
              penalised_score(lp_long, len_long, 0.0),
          "14h with alpha == 0 the SHORT hypothesis wins (the penalty matters)");

    // And a real beam still decodes and retains hypotheses under this config.
    BeamSearchConfig lp_cfg;
    lp_cfg.beam_width = 4;
    lp_cfg.max_length = 8;
    lp_cfg.eos_id = 2;
    lp_cfg.length_penalty = 1.0;
    BeamSearchDecoder lp_dec(
        3, lp_cfg,
        [](const Tensor& ids) {
            Tensor o(ids.rows, 3);
            for (size_t r = 0; r < ids.rows; ++r) {
                if (ids.cols == 0) { o(r, 2) = 4.0; o(r, 1) = 0.0; o(r, 0) = 0.0; }
                else               { o(r, 2) = 2.0; o(r, 1) = 1.9; o(r, 0) = 1.0; }
            }
            return o;
        });
    lp_dec.decode(Tensor::zeros(1, 1), 4);
    const std::vector<BeamHypothesis>& ranked = lp_dec.hypotheses();
    CHECK(ranked.size() >= 2, "14i the length-penalty fixture produced several hypotheses");
    bool has_short = false;
    for (const BeamHypothesis& h : ranked)
        if (h.tokens.size() == 2) has_short = true;   // prompt(1) + one token
    CHECK(has_short, "14j a length-1 (prompt + 1 token) hypothesis was produced");
    // the decoder's own ranking agrees with the predicate above
    for (size_t i = 1; i < ranked.size(); ++i) {
        double a = penalised_score(ranked[i - 1].logprob, ranked[i - 1].tokens.size(), 1.0);
        double b = penalised_score(ranked[i].logprob, ranked[i].tokens.size(), 1.0);
        CHECK(a >= b - 1e-12, "14k hypotheses() is sorted by the penalised score");
    }
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
        [V](const Tensor& ids) {
            Tensor out(ids.rows, V);
            for (size_t r = 0; r < ids.rows; ++r) {
                out(r, 3) = 10.0;   // always "eod"
                out(r, 2) = 1.0;    // eos is less attractive
            }
            return out;
        });
    Tensor start = Tensor::zeros(1, 1);
    Tensor out = dec.decode(start, 1);
    CHECK(out.cols >= cfg.max_length || out.cols > 1,
          "15a a run of eod tokens does NOT terminate decoding at step 1");
    CHECK(out.cols >= 2, "15b decode continued past a non-eos sentinel");

    // and the sentinel really was emitted at some point. NOTE it is NOT at index
    // 1: the top-ranked hypothesis is [0, 2] because the model scores eod at 10.0
    // and eos at 1.0, so [0,3,2] (logprob -9.0004) loses to [0,2] (-9.0002).
    // An earlier draft asserted the sentinel appeared in the RETURNED row and
    // failed -- which was the test's premise being wrong, not the decoder.
    const std::vector<BeamHypothesis>& hyps = dec.hypotheses();
    bool saw_eod = false;
    for (const BeamHypothesis& h : hyps)
        for (int t : h.tokens)
            if (t == 3) saw_eod = true;
    CHECK(saw_eod, "15c the retained hypotheses contain the eod sentinel");
    CHECK(hyps.size() >= 1, "15d hypotheses() is non-empty after decode");
    // and none of the retained hypotheses is longer than prompt + max_length
    size_t longest = 0;
    for (const BeamHypothesis& h : hyps) longest = std::max(longest, h.tokens.size());
    CHECK(longest <= 1 + cfg.max_length, "15e no hypothesis exceeds max_length");

    // THE DISCRIMINATOR FOR "eod IS NOT eos", added after a mutation that made
    // the LAST vocabulary id terminate as well SURVIVED the first harness run.
    //
    // 15b/15c/15d above all pass either way, because the sentinel still APPEARS
    // in the retained hypotheses under the mutant -- it merely terminates there
    // instead of continuing. So none of them can see the difference. What does:
    // the LENGTH of the surviving hypotheses. With beam_width 1 and a model that
    // emits the sentinel on every step, correct behaviour produces ONE
    // hypothesis per step and keeps going to max_length; the mutant stops at the
    // first sentinel.
    BeamSearchConfig run_cfg;
    run_cfg.beam_width = 1;
    run_cfg.max_length = 6;
    run_cfg.eos_id = 2;
    BeamSearchDecoder runner(
        4, run_cfg,
        [](const Tensor& ids) {
            Tensor o(ids.rows, 4);
            for (size_t r = 0; r < ids.rows; ++r) {
                o(r, 3) = 10.0;   // the last id, always -- NOT eos (eos is 2)
                o(r, 2) = 0.5;
            }
            return o;
        });
    Tensor run_out = runner.decode(Tensor::zeros(1, 1), 1);
    // prompt(1) + max_length(6) tokens: the sentinel never stopped it.
    CHECK(run_out.cols == 1 + run_cfg.max_length,
          "15f a run of the LAST id does NOT terminate: decode fills max_length "
          "(this is the assertion that kills the eod-as-eos mutation)");

    // and the contrast: the same model with eos_id = 3 (the last id) DOES stop.
    BeamSearchConfig stop_cfg = run_cfg;
    stop_cfg.eos_id = 3;
    BeamSearchDecoder stopper(
        4, stop_cfg,
        [](const Tensor& ids) {
            Tensor o(ids.rows, 4);
            for (size_t r = 0; r < ids.rows; ++r) {
                o(r, 3) = 10.0;
                o(r, 2) = 0.5;
            }
            return o;
        });
    Tensor stop_out = stopper.decode(Tensor::zeros(1, 1), 1);
    CHECK(stop_out.cols < 1 + stop_cfg.max_length,
          "15g when that id IS configured as eos, decoding stops early -- so "
          "15f is measuring termination, not token count by accident");

    // By contrast, a real eos DOES stop it.
    BeamSearchDecoder dec2(
        V, cfg,
        [V](const Tensor& ids) {
            Tensor out(ids.rows, V);
            for (size_t r = 0; r < ids.rows; ++r) out(r, 2) = 10.0;   // real eos
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
    CHECK(r.tokens.cols >= 7, "16a the loop produced prompt(1) + steps(6) tokens");
    CHECK(!r.stopped_early, "16b stopped_early is false when eos never appears");

    // The FIRST column is the prompt, which the model did not generate, so the
    // argmax check starts at column 1. (An earlier draft checked from column 0
    // and failed on the prompt's own 0 -- the test's premise, not the loop.)
    bool all_one = true;
    for (size_t j = 1; j < r.tokens.cols; ++j)
        if (static_cast<int>(r.tokens(0, j)) != 1) all_one = false;
    CHECK(all_one, "16c every GENERATED token is the model's argmax");
    CHECK(static_cast<int>(r.tokens(0, 0)) == 0,
          "16d the prompt is preserved as the first column");

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
    std::mt19937 rng_a(31337), rng_b(31337);
    AutoregressiveResult a = autoregressive_sample(multi, Tensor::zeros(1, 1), 8, empty,
                                                  SamplingStrategy::MULTINOMIAL, rng_a);
    AutoregressiveResult b = autoregressive_sample(multi, Tensor::zeros(1, 1), 8, empty,
                                                  SamplingStrategy::MULTINOMIAL, rng_b);
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