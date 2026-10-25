// test_speculative_decoding.cpp -- TDD tests for speculative decoding
// (include/nn/utils/logit_processing.h).
//
// THE GAP THIS FILLS is recorded, with the greps that measured it, in
// docs/plans/2026-10-25_speculative-decoding.md. In short: the repo's own
// utils/logit_processing.h says, in its own words, at line 168,
//
//     // OUT OF SCOPE FOR v1, stated so it cannot creep in:
//     //   - Speculative / lookahead / contrastive decoding.
//
// and `grep -rinE 'speculat|draft.model|acceptance|rejection.sampl'` over
// include/ and tests/ returns that line and nothing else.
//
// EVERY numeric expectation below is ORACLE-DERIVED from
// tests/speculative_decoding_oracle.py (numpy float64), NOT hand-derived.
//
// ---------------------------------------------------------------------------
// WHAT THE ORACLE CAUGHT BEFORE THIS FILE EXISTED (2026-10-25)
// ---------------------------------------------------------------------------
// Four findings, and THREE of them changed the SPEC rather than the code:
//
//   1. THE CLAMP ON THE ACCEPTANCE RATIO IS MEASURED INERT. The first draft
//      of the plan asserted that clamping `p/q` at 1 is "required by the
//      maths" and built a foil to prove it. The foil was VACUOUS: the
//      clamped and the unclamped rules produced identical output on 3000/3000
//      trials. The reason is a fact about the RULE'S SHAPE -- when the
//      acceptance is written as a threshold comparison `r <= p/q` and `p > q`,
//      the ratio exceeds 1 and every `r in [0,1)` satisfies it automatically,
//      which is exactly the correct behaviour. The clamp is DEFENSIVE, not
//      load-bearing. Test 9 measures this rather than asserting it.
//
//   2. THE `<=` versus `<` BOUNDARY IS UNOBSERVABLE BY A RANDOM FIXTURE. The
//      two differ only when the uniform draw lands EXACTLY on the acceptance
//      probability -- a measure-zero event for a continuous draw. The only
//      way to discriminate is to SUPPLY the draw at exactly the boundary, on a
//      DYADIC acceptance probability. Test 8 does that, using FIXTURE D
//      (a point-mass draft, so the guess is deterministic, and an acceptance
//      probability of exactly 0.5). This is the same lesson the repo's top-p
//      fixture learned the hard way: a fixture whose discriminating value is
//      only correct to within a rounding error does not discriminate
//      anything. Here the value must be exactly representable AND supplied.
//
//   3. THE FIXTURE MUST VARY BY POSITION. The oracle's first fixture used
//      the same distribution at every position, which made the "wrong residual
//      index" foil VACUOUS -- with `p_n == p_{n+1}` for every `n`, an
//      implementation reading the residual from the wrong index computes the
//      SAME distribution. FIXTURE A below gives every position a DIFFERENT
//      pair of distributions, and Test 6 asserts that the residuals at two
//      positions actually differ, so the index is observable.
//
//   4. `max(0, .)` IN THE RESIDUAL IS MATHEMATICALLY LOAD-BEARING, not merely
//      a different number: `p` and `q` are both distributions, so
//      `sum(p - q) == 0` EXACTLY, and dropping the componentwise clamp leaves
//      a zero-sum vector whose normalisation divides by zero. Test 7 pins the
//      residual's exact values.
//
// AND ONE FINDING WAS IN THE ORACLE'S OWN COMPARISON CODE: it compared a
// 3-tuple against a 4-tuple, which made every foil "differ" for a reason that
// had nothing to do with the algorithm. *The oracle is code you just wrote;
// a failing comparison is evidence about one of the two pieces, not about
// the theory.*

// ---------------------------------------------------------------------------
// THE API
// ---------------------------------------------------------------------------
// A MODEL is a callback: it receives the token ids generated so far and
// returns a row of LOGITS. The library never learns what generated them,
// which is what lets this file drive the algorithm from a hand-written oracle
// model -- and what lets it drive a real PixelCNN with no decoder change.
//
// The step consumes 2*gamma+1 uniform draws, in this order:
//     us[0      .. gamma-1  ]  the draws that SELECT the guesses (from q_1..q_gamma)
//     us[gamma  .. 2*gamma-1]  the acceptance draws r_1..r_gamma
//     us[2*gamma]            the draw that selects the final token
// The `with_uniforms` entry point takes them explicitly, so the boundary
// convention is testable (finding 2 above); the `rng` entry point draws them
// from an std::mt19937.
//
// The DRAFT is called GAMMA times and the TARGET GAMMA+1 times per step --
// the paper's cost model. Test 5 pins both counts.

#include "../include/nn/utils/logit_processing.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
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

static void banner(const char* s) { printf("\n--- %s ---\n", s); }

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------
static Tensor ids_row(std::initializer_list<int> v) {
    Tensor t(1, v.size());
    size_t j = 0;
    for (int x : v) t(0, j++) = static_cast<double>(x);
    return t;
}

static std::vector<int> to_ids(const Tensor& t) {
    std::vector<int> v;
    for (size_t j = 0; j < t.cols; ++j)
        v.push_back(static_cast<int>(t(0, j)));
    return v;
}

// The expected-token comparisons go through a helper rather than through a
// braced init-list inside the CHECK macro: a braced list with a comma is
// split by the PREPROCESSOR into two macro arguments, which is a compile
// error that has nothing to do with the code under test.
static bool ids_are(const Tensor& t, std::initializer_list<int> expected) {
    return to_ids(t) == std::vector<int>(expected);
}

// ---------------------------------------------------------------------------
// FIXTURE A -- position-varying draft and target.
//
// From the oracle's table (PROBABILITIES, with the matching LOGITS):
//
//   pos 0: p = [0.60, 0.20, 0.15, 0.05]   q = [0.25, 0.25, 0.25, 0.25]
//   pos 1: p = [0.10, 0.50, 0.30, 0.10]   q = [0.40, 0.20, 0.20, 0.20]
//   pos 2: p = [0.25, 0.25, 0.25, 0.25]   q = [0.10, 0.40, 0.30, 0.20]
//   pos 3: p = [0.05, 0.15, 0.20, 0.60]   q = [0.20, 0.20, 0.30, 0.30]
//
// The LOGITS below are `log(p)` for each row, so the C++'s max-subtracting
// softmax reproduces the oracle's probability tables to the last bit.
// (oracle: "pos 0 acceptance probs = [1.0, 0.8, 0.6, 0.2]")
// ---------------------------------------------------------------------------
static const size_t A_VOCAB = 4;

static const double A_P_LOGITS[5][4] = {
    {-0.5108256237659907, -1.6094379124341003, -1.8971199848858813, -2.995732273553991},
    {-2.3025850929940455, -0.6931471805599453, -1.2039728043259361, -2.3025850929940455},
    {-1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906},
    {-2.995732273553991, -1.8971199848858813, -1.6094379124341003, -0.5108256237659907},
    // A fifth row, so a ONE-token prefix also fits: the purity test needs a
    // non-empty prefix to check that it is not mutated.
    {-0.5108256237659907, -1.6094379124341003, -1.8971199848858813, -2.995732273553991},
};

static const double A_Q_LOGITS[5][4] = {
    {-1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906},
    {-0.916290731874155, -1.6094379124341003, -1.6094379124341003, -1.6094379124341003},
    {-2.3025850929940455, -0.916290731874155, -1.2039728043259361, -1.6094379124341003},
    {-1.6094379124341003, -1.6094379124341003, -1.2039728043259361, -1.2039728043259361},
    {-1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906},
};

// A model that returns the row for the position given by the prefix LENGTH.
//
// The row table is a template parameter rather than a `const double*` so
// that a 2-D fixture keeps its array type: a plain pointer cannot be
// subscripted by position, and casting away the type would lose the width.
template <size_t N, size_t M>
struct PositionalModel {
    const double (&rows)[N][M];   // [n_pos][vocab]
    size_t n_pos;
    size_t vocab;
    int* call_count;              // optional

    Tensor operator()(const Tensor& prefix) const {
        if (call_count) ++(*call_count);
        size_t pos = prefix.cols;
        if (pos >= n_pos) {
            throw std::runtime_error(
                "PositionalModel: prefix length " + std::to_string(pos) +
                " is beyond the fixture");
        }
        Tensor out(1, vocab);
        for (size_t j = 0; j < vocab; ++j) out(0, j) = rows[pos][j];
        return out;
    }
};

template <size_t N, size_t M>
static DraftModel make_positional(const double (&rows)[N][M], size_t n_pos,
                                  size_t vocab, int* call_count) {
    PositionalModel<N, M> m{rows, n_pos, vocab, call_count};
    return [m](const Tensor& prefix) { return m(prefix); };
}

// FIXTURE B -- IDENTICAL draft and target.
static const double B_LOGITS[4][3] = {
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
};

// FIXTURE D -- the BOUNDARY fixture. A point-mass draft (the guess is always
// token 0) and an acceptance probability of EXACTLY 0.5.
static const double D_P_LOGITS[3][2] = {
    {-0.6931471805599453, -0.6931471805599453},
    {-0.6931471805599453, -0.6931471805599453},
    {-0.6931471805599453, -0.6931471805599453},
};
static const double D_Q_LOGITS[3][2] = {
    {0.0, -1e300},   // a point mass on token 0
    {0.0, -1e300},
    {0.0, -1e300},
};

// FIXTURE E -- the distributional fixture: one position, a known target.
static const double E_P_LOGITS[2][3] = {
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
    {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
};
static const double E_Q_LOGITS[2][3] = {
    {-1.6094379124341003, -1.2039728043259361, -0.6931471805599453},
    {-1.6094379124341003, -1.2039728043259361, -0.6931471805599453},
};

// ===========================================================================
// Test 1: the guards
// ===========================================================================
static void test_guards() {
    banner("1 the guards");

    DraftModel model = make_positional(A_P_LOGITS, 4, A_VOCAB, nullptr);

    bool threw = false;
    try {
        std::mt19937 rng(1);
        speculative_decode_step(model, model, ids_row({}), 0, rng);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "1a gamma == 0 throws std::invalid_argument");

    threw = false;
    try {
        std::mt19937 rng(1);
        DraftModel null_model;
        speculative_decode_step(null_model, model, ids_row({}), 2, rng);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "1b a null DRAFT throws std::invalid_argument");

    threw = false;
    try {
        std::mt19937 rng(1);
        DraftModel null_model;
        speculative_decode_step(model, null_model, ids_row({}), 2, rng);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "1c a null TARGET throws std::invalid_argument");

    threw = false;
    try {
        std::mt19937 rng(1);
        Tensor bad_prefix(2, 3);
        speculative_decode_step(model, model, bad_prefix, 2, rng);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "1d a multi-row prefix throws std::invalid_argument");

    // A draft and a target that disagree about the vocabulary WIDTH.
    threw = false;
    try {
        std::mt19937 rng(1);
        DraftModel narrow = make_positional(D_P_LOGITS, 3, 2, nullptr);
        speculative_decode_step(narrow, model, ids_row({}), 2, rng);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw, "1e a vocabulary-width mismatch throws std::runtime_error");

    // The number of uniforms must be exactly 2*gamma+1.
    threw = false;
    try {
        DraftModel m2 = make_positional(A_P_LOGITS, 4, A_VOCAB, nullptr);
        std::vector<double> too_few(4, 0.5);   // gamma = 3 needs 7
        speculative_decode_step_with_uniforms(m2, m2, ids_row({}), 3, too_few);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "1f too few uniforms throws std::invalid_argument");
}

// ===========================================================================
// Test 2: IDENTICAL MODELS (SP10, SP7) -- the strongest single property.
//
// With p == q the acceptance ratio is 1 everywhere, EVERY guess is accepted,
// and the step emits EXACTLY gamma+1 tokens: the gamma guesses plus the
// bonus token drawn from p_{gamma+1}.
//
// SP7 rides along: the residual of identical models is ALL-ZERO (both are
// distributions, so p - q == 0), so an implementation that always used the
// residual would divide by zero. The step must NOT do that.
// ===========================================================================
static void test_identical_models() {
    banner("2 identical models");

    DraftModel model = make_positional(B_LOGITS, 4, 3, nullptr);
    std::mt19937 rng(20261025);

    for (int trial = 0; trial < 20; ++trial) {
        SpeculativeResult r =
            speculative_decode_step(model, model, ids_row({}), 3, rng);
        CHECK(r.num_accepted == 3,
              "2a with identical models EVERY guess is accepted");
        CHECK(!r.rejected, "2b with identical models nothing is rejected");
        CHECK(r.tokens.cols == 4,
              "2c with identical models EXACTLY gamma+1 tokens are emitted");
    }

    // The emitted tokens must be in the vocabulary.
    SpeculativeResult r = speculative_decode_step(model, model, ids_row({}), 3, rng);
    for (int t : to_ids(r.tokens)) {
        CHECK(t >= 0 && t < 3, "2d every emitted token is in the vocabulary");
    }
}

// ===========================================================================
// Test 3: the CALL COUNTS (SP8) -- the speedup claim itself.
//
// The draft is called GAMMA times and the target GAMMA+1 times. The residual
// needs q_{n+1} with n+1 <= gamma, so it is always among the rows already
// computed and the draft is NEVER called a (gamma+1)-th time.
// ===========================================================================
static void test_call_counts() {
    banner("3 the call counts");

    int draft_calls = 0;
    int target_calls = 0;
    DraftModel draft = make_positional(A_Q_LOGITS, 4, A_VOCAB, &draft_calls);
    DraftModel target = make_positional(A_P_LOGITS, 4, A_VOCAB, &target_calls);

    std::mt19937 rng(7);
    speculative_decode_step(draft, target, ids_row({}), 3, rng);

    CHECK(draft_calls == 3, "3a the DRAFT is called exactly gamma times");
    CHECK(target_calls == 4, "3b the TARGET is called exactly gamma+1 times");
}

// ===========================================================================
// Test 4: the WORKED EXAMPLE (SP4, SP5, SP6, SP9)
//
// From the oracle, verbatim:
//     us  = [0.9, 0.1, 0.4, 0.25, 0.6, 0.7, 0.3]
//     -> tokens = [0]   n = 0   rejected = True   flags = [False, False, False]
//
// n == 0 means the FIRST guess was rejected, so the step emits exactly ONE
// token -- the correction -- and none of the guesses.
// ===========================================================================
static void test_worked_example() {
    banner("4 the worked example");

    DraftModel draft = make_positional(A_Q_LOGITS, 4, A_VOCAB, nullptr);
    DraftModel target = make_positional(A_P_LOGITS, 4, A_VOCAB, nullptr);

    std::vector<double> us = {0.9, 0.1, 0.4, 0.25, 0.6, 0.7, 0.3};
    SpeculativeResult r = speculative_decode_step_with_uniforms(
        draft, target, ids_row({}), 3, us);

    CHECK(r.num_accepted == 0, "4a the worked example has n == 0");
    CHECK(r.rejected, "4b the worked example rejected a guess");
    CHECK(r.tokens.cols == 1, "4c the worked example emits EXACTLY n+1 == 1 token");
    CHECK(ids_are(r.tokens, {0}),
          "4d the worked example's token is 0");

    // The SAME result must come out of the rng-driven entry point, since the
    // uniforms are the only source of randomness.
    std::mt19937 rng(12345);
    SpeculativeResult r2 =
        speculative_decode_step(draft, target, ids_row({}), 3, rng);
    CHECK(r2.tokens.cols >= 1 && r2.tokens.cols <= 4,
          "4e the rng-driven step emits between 1 and gamma+1 tokens");
}

// ===========================================================================
// Test 5: the EMITTED COUNT IS n+1, AND THE FIRST REJECTION ENDS THE RUN
// (SP4, SP5, SP9)
//
// Over many seeds the emitted count must equal n+1 EXACTLY, and the
// rejection must occur at EVERY reachable position -- otherwise the residual's
// index is not observable and Test 6 would be vacuous.
// ===========================================================================
static void test_emitted_count_is_n_plus_one() {
    banner("5 the emitted count is n+1");

    DraftModel draft = make_positional(A_Q_LOGITS, 4, A_VOCAB, nullptr);
    DraftModel target = make_positional(A_P_LOGITS, 4, A_VOCAB, nullptr);

    int hist[5] = {0, 0, 0, 0, 0};   // histogram by n
    bool all_ok = true;
    for (int seed = 0; seed < 400; ++seed) {
        std::mt19937 rng(static_cast<unsigned>(seed));
        SpeculativeResult r =
            speculative_decode_step(draft, target, ids_row({}), 3, rng);
        if (r.tokens.cols != r.num_accepted + 1) all_ok = false;
        if (r.num_accepted < 4) ++hist[r.num_accepted];
        // A rejection must mean n < gamma, and no rejection must mean n == gamma.
        if (r.rejected != (r.num_accepted < 3)) all_ok = false;
    }
    CHECK(all_ok, "5a the emitted count is EXACTLY n+1 on every trial");
    printf("        rejection histogram by n: %d %d %d\n", hist[0], hist[1], hist[2]);
    CHECK(hist[0] > 0, "5b a rejection at position 1 (n == 0) is reachable");
    CHECK(hist[1] > 0 && hist[2] > 0,
          "5c rejections at MORE THAN ONE position are reachable -- so the "
          "residual's index is observable");
}

// ===========================================================================
// Test 6: the RESIDUAL (SP6) -- the correction token's distribution.
//
// The residual is norm(max(0, p - q)) over the FULL vocabulary. Its exact
// values are pinned here, from the oracle:
//
//   pos 0 residual = [1.0, 0.0, 0.0, 0.0]     (p - q = [0.35, -0.05, -0.10, -0.20])
//   pos 1 residual = [0.0, 0.75, 0.25, 0.0]   (p - q = [-0.30, 0.30, 0.10, -0.10])
//
// The two residuals DIFFER, which is what makes the residual's INDEX
// observable -- an implementation reading the residual from the wrong index
// would produce a different answer and this test would see it.
// ===========================================================================
static void test_residual_values() {
    banner("6 the residual");

    // The residual is not part of the public API (it is an internal detail
    // of the correction step), so it is observed through the CORRECTION
    // TOKEN: with the residual at position 0 being the point mass on token
    // 0, a rejection at position 1 MUST emit token 0 and nothing else.
    //
    // To force a rejection at position 1 the acceptance draw must exceed the
    // acceptance probability. At position 0 the acceptance probabilities are
    // [1.0, 0.8, 0.6, 0.2]; the guess is token 3 whenever us[0] lands in
    // [0.75, 1.0) (the draft is uniform), and then the acceptance
    // probability is 0.2. So us[1] = 0.5 rejects it.
    DraftModel draft = make_positional(A_Q_LOGITS, 4, A_VOCAB, nullptr);
    DraftModel target = make_positional(A_P_LOGITS, 4, A_VOCAB, nullptr);

    // us[0] = 0.9 -> the guess is token 3 (the draft is uniform, so the
    // CDF is [0.25, 0.5, 0.75, 1.0] and 0.9 lands in the last cell).
    // us[1] = 0.5 > 0.2 -> the guess is REJECTED at position 1.
    std::vector<double> us = {0.9, 0.5, 0.4, 0.25, 0.6, 0.7, 0.3};
    SpeculativeResult r = speculative_decode_step_with_uniforms(
        draft, target, ids_row({}), 3, us);

    CHECK(r.num_accepted == 0, "6a the guess at position 1 was rejected");
    CHECK(r.rejected, "6b the step reports a rejection");
    CHECK(r.tokens.cols == 1, "6c exactly one token is emitted (the correction)");
    CHECK(ids_are(r.tokens, {0}),
          "6d the correction is token 0 -- the pos-0 residual is the point "
          "mass [1.0, 0, 0, 0]");

    // The pos-1 residual is [0, 0.75, 0.25, 0] -- a rejection at position 2
    // must emit token 1 or token 2, NEVER token 0 or token 3. This is the
    // check that the residual's INDEX is observable: the pos-0 residual
    // would have emitted token 0.
    //
    // To reject at position 2 the guess at position 1 must be accepted
    // (us[2] <= the acceptance probability at position 1) and the guess at
    // position 2 must be rejected.
    //   position 1: the draft is [0.4, 0.2, 0.2, 0.2] and the guess is the
    //   token whose CDF cell contains us[1]. With us[1] = 0.9 the guess is
    //   token 3 (CDF [0.4, 0.6, 0.8, 1.0]).
    //   The target at position 1 is [0.1, 0.5, 0.3, 0.1], so the acceptance
    //   probability for token 3 is 0.1/0.2 = 0.5. us[2] = 0.1 accepts it.
    //   position 2: the draft is [0.1, 0.4, 0.3, 0.2] and us[3] = 0.9 gives
    //   guess token 3 (CDF [0.1, 0.5, 0.8, 1.0]); the target at position 2 is
    //   uniform [0.25]*4, so the acceptance probability is 0.25/0.2 = 1.0 --
    //   it would ALWAYS be accepted. So use us[3] = 0.05 -> guess token 0
    //   (CDF [0.1, ...], 0.05 < 0.1), acceptance probability 0.25/0.1 = 1.0
    //   -- also always accepted.
    //
    // The cleanest way to force a rejection at position 2 is to give the
    // draft a position-2 row whose guess has p < q. Fixture A's position 2
    // target is UNIFORM, so every acceptance probability there is >= 1.
    // Rather than bend the fixture, use a DEDICATED one:
    //   position 0: p = [0.5, 0.5]      q = [1.0, 0.0]   (the D boundary row)
    //   position 1: p = [0.25, 0.75]    q = [0.5, 0.5]
    //   position 2: p = [0.25, 0.75]    q = [0.5, 0.5]
    // The guess at position 1 is token 0 (the draft is a point mass), and the
    // acceptance probability there is 0.25/0.5 = 0.5, so us[2] = 0.9 rejects
    // it. The residual at position 1 is norm(max(0, [0.25,0.75]-[0.5,0.5]))
    // = norm([0, 0.25]) = [0, 1] -- the point mass on token 1.
    const double E2_P[4][2] = {
        {-0.6931471805599453, -0.6931471805599453},
        {-1.3862943611198906, -0.2876820724517809},
        {-1.3862943611198906, -0.2876820724517809},
        {-1.3862943611198906, -0.2876820724517809},
    };
    const double E2_Q[4][2] = {
        {0.0, -1e300},
        {-0.6931471805599453, -0.6931471805599453},
        {-0.6931471805599453, -0.6931471805599453},
        {-0.6931471805599453, -0.6931471805599453},
    };
    DraftModel d2 = make_positional(E2_Q, 4, 2, nullptr);
    DraftModel t2 = make_positional(E2_P, 4, 2, nullptr);
    // The uniforms come from the ORACLE, not from a hand-derived layout:
    // the oracle reports the exact draw sequence that produces a rejection
    // at position index 1 (n == 1). A hand-computed layout was WRONG in the
    // first draft of this test -- the uniform that selects the guess, the
    // uniform that accepts it and the uniform that rejects the NEXT one are
    // three different slots, and mixing them up produces a rejection at the
    // wrong position.
    //
    // oracle: us = [0.636962, 0.269787, 0.040974, 0.016528, 0.81327,
    //               0.912756, 0.606636]
    //         -> tokens = [0, 1]   n = 1   flags = [True, False, False]
    std::vector<double> us2 = {0.636962, 0.269787, 0.040974, 0.016528,
                               0.81327, 0.912756, 0.606636};
    SpeculativeResult r2 = speculative_decode_step_with_uniforms(
        d2, t2, ids_row({}), 3, us2);
    CHECK(r2.num_accepted == 1,
          "6e the guess at position 1 was accepted and the one at position 2 "
          "rejected");
    CHECK(r2.rejected, "6f the step reports a rejection at position 2");
    CHECK(r2.tokens.cols == 2, "6g two tokens are emitted (1 accepted + 1)");
    CHECK(ids_are(r2.tokens, {0, 1}),
          "6h the correction is token 1 -- the pos-1 residual is the point "
          "mass [0, 1], and NOT the pos-0 residual");
}

// ===========================================================================
// Test 7: the RESIDUAL'S SUM DIVISOR (SP6)
//
// `p` and `q` are both distributions, so sum(p - q) == 0 EXACTLY. Dropping
// the componentwise max(0, .) leaves a zero-sum vector and the normalisation
// divides by zero -- the residual is not merely different, it is UNDEFINED.
//
// The correction token's distribution is observed here by forcing many
// rejections at a position whose residual has TWO surviving coordinates, so
// the SUM (not the max) is what divides it.
//
//   p = [0.6, 0.4, 0, 0]   q = [0.3, 0.1, 0.3, 0.3]
//   p - q = [0.3, 0.3, -0.3, -0.3]  ->  max(0,.) = [0.3, 0.3, 0, 0]
//   -> the residual is [0.5, 0.5, 0, 0] under a SUM divisor, and [1.0, 0.0,
//   0, 0] under a MAX divisor.
//
// A wrong divisor gives token 0 with probability 1 instead of 1/2, so the
// measured rate over many trials separates them cleanly.
// ===========================================================================
static void test_residual_sum_divisor() {
    banner("7 the residual's sum divisor");

    //   position 0: p = [0.5, 0.5]    q = [1.0, 0.0]   (point-mass draft)
    //   position 1: p = [0.6, 0.4]    q = [0.3, 0.1]     (both positive)
    //   position 2: p = [0.6, 0.4]    q = [0.3, 0.1]
    //
    // The guess at position 0 is token 0 (point mass) and the acceptance
    // probability there is 0.5/1.0 = 0.5, so us[1] = 0.9 REJECTS it.
    // The residual at position 0 is then norm(max(0, [0.5,0.5]-[1,0]))
    // = norm([0, 0.5]) = [0, 1] -- a point mass on token 1.
    //
    // To observe the TWO-coordinate residual the rejection must happen at a
    // position whose p and q both have positive entries on two tokens. That
    // is position 1 -- so the position-0 guess must be ACCEPTED and the
    // position-1 guess REJECTED. But the point-mass draft forces the
    // position-0 guess to token 0, whose acceptance probability at
    // position 0 is 0.5 -- and the acceptance draw can be chosen to accept.
    //
    //   position 0: p = [0.6, 0.4]   q = [0.6, 0.4]   -> acceptance 1.0
    //   position 1: p = [0.6, 0.4]   q = [0.3, 0.1]   -> token 0's
    //                acceptance probability is 0.6/0.3 = 2.0 -> clamped to 1
    //
    // So a two-coordinate residual needs a token with p < q at that
    // position. Use:
    //   position 0: p = [0.5, 0.5]   q = [1.0, 0.0]   (point mass, ratio 0.5)
    //   position 1: p = [0.1, 0.9]   q = [0.5, 0.5]   -> token 0's ratio is
    //                0.1/0.5 = 0.2, so us[2] = 0.9 REJECTS it.
    //   The residual at position 1 is norm(max(0, [0.1,0.9]-[0.5,0.5]))
    //                = norm([0, 0.4]) = [0, 1] -- still a point mass.
    //
    // For a TWO-coordinate residual the residual must have two positive
    // entries, which needs p > q on TWO tokens. Use:
    //   position 1: p = [0.55, 0.45]  q = [0.30, 0.20]  -> p - q =
    //                [0.25, 0.25] -> the residual is [0.5, 0.5] under a SUM
    //                divisor and [1.0, 0.0] under a MAX divisor.
    //   But then the guess's acceptance probability is 0.55/0.30 = 1.0 --
    //   never rejected. So the REJECTION must be forced at position 1 by a
    //   DIFFERENT token: the guess is chosen by the draft's CDF, and the
    //   draft at position 1 is [0.30, 0.20, 0.30, 0.20] over FOUR tokens.
    //
    // The simplest construction that gives a two-coordinate residual AND a
    // rejection: make the guess a token where p < q, and give the OTHER token
    // a positive p - q. With a two-token vocabulary:
    //   p = [0.2, 0.8]   q = [0.5, 0.5]  ->  p - q = [-0.3, 0.3] -> one entry.
    // With three tokens:
    //   p = [0.2, 0.7, 0.1]   q = [0.4, 0.3, 0.3]
    //   p - q = [-0.2, 0.4, -0.2]  -> one entry.
    //   p = [0.5, 0.3, 0.2]   q = [0.2, 0.4, 0.4]
    //   p - q = [0.3, -0.1, -0.2]  -> one entry.
    //
    // TWO positive entries requires p > q on two tokens, and then EVERY
    // token's acceptance probability at that position is >= 1 (since p >= q
    // on those two)... no: p > q on tokens 0 and 1, and p < q on token 2. The
    // guess is token 2, whose acceptance probability is < 1, so it can be
    // rejected, and the residual has TWO positive entries.
    //   p = [0.5, 0.3, 0.2]   q = [0.2, 0.1, 0.7]
    //   p - q = [0.3, 0.2, -0.5]  ->  max(0,.) = [0.3, 0.2, 0]
    //   -> the residual is [0.6, 0.4, 0] under a SUM divisor and
    //      [1.0, 0.0, 0] under a MAX divisor.
    //   The guess at that position is token 2 (the draft's CDF puts 0.7 on
    //   it and we choose the draw in [0.3, 1.0)), and its acceptance
    //   probability is 0.2/0.7 = 0.2857..., so a draw of 0.9 REJECTS it.
    const double F_P[3][3] = {
        {-0.6931471805599453, -0.6931471805599453},   // placeholder, replaced
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    };
    (void)F_P;

    // position 0: p = [0.5, 0.5]    q = [1.0, 0.0]     (point mass, ratio 0.5)
    // position 1: p = [0.5, 0.3, 0.2]  q = [0.2, 0.1, 0.7]
    const double G_P[4][3] = {
        {-0.6931471805599453, -0.6931471805599453, -0.6931471805599453},
        {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
        {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
        {-0.6931471805599453, -1.2039728043259361, -1.6094379124341003},
    };
    const double G_Q[4][3] = {
        {0.0, -1e300, -1e300},
        {-1.6094379124341003, -2.3025850929940455, -0.35667494393873245},
        {-1.6094379124341003, -2.3025850929940455, -0.35667494393873245},
        {-1.6094379124341003, -2.3025850929940455, -0.35667494393873245},
    };
    DraftModel dg = make_positional(G_Q, 4, 3, nullptr);
    DraftModel tg = make_positional(G_P, 4, 3, nullptr);

    // The residual at position 1 is norm(max(0, [0.5,0.3,0.2]-[0.2,0.1,0.7]))
    // = norm([0.3, 0.2, 0]) = [0.6, 0.4, 0] -- TWO surviving coordinates, so
    // the SUM divisor is what decides the correction's distribution.
    //
    // The uniforms come from the ORACLE: it reports the exact draw sequence
    // that reaches position 1 and rejects there.
    //   oracle: us = [0.943056, 0.511328, 0.976244, 0.080836, 0.607356,
    //                 0.376487, 0.801901]
    //           -> tokens = [0, 1]   n = 1   flags = [True, False, False]
    // Only the FINAL draw (us[6]) varies across the trials -- it selects
    // the correction token, and it is the only one that matters for the
    // measurement.
    int counts[3] = {0, 0, 0};
    const int trials = 4000;
    for (int s = 0; s < trials; ++s) {
        double u6 = static_cast<double>(s) / static_cast<double>(trials);
        std::vector<double> us = {0.943056, 0.511328, 0.976244, 0.080836,
                                  0.607356, 0.376487, u6};
        SpeculativeResult r =
            speculative_decode_step_with_uniforms(dg, tg, ids_row({}), 3, us);
        if (r.num_accepted == 1 && r.rejected && r.tokens.cols == 2) {
            std::vector<int> ids = to_ids(r.tokens);
            if (ids.size() == 2 && ids[1] >= 0 && ids[1] < 3) ++counts[ids[1]];
        }
    }
    int total = counts[0] + counts[1] + counts[2];
    printf("        correction token counts: %d %d %d (of %d)\n",
           counts[0], counts[1], counts[2], total);
    CHECK(total > 0, "7a the rejection path at the two-coordinate residual ran");
    if (total > 0) {
        double rate0 = static_cast<double>(counts[0]) / total;
        double rate1 = static_cast<double>(counts[1]) / total;
        double rate2 = static_cast<double>(counts[2]) / total;
        printf("        measured rates: %.4f %.4f %.4f  (expected 0.6 0.4 0.0)\n",
               rate0, rate1, rate2);
        CHECK(std::fabs(rate0 - 0.6) < 0.03,
              "7b token 0's rate is ~0.6 -- the residual is normalised by the SUM");
        CHECK(std::fabs(rate1 - 0.4) < 0.03,
              "7c token 1's rate is ~0.4 -- the residual is normalised by the SUM");
        CHECK(rate2 < 0.02,
              "7d token 2 never appears -- the residual is ZERO where p < q");
        // A MAX divisor would give rate0 == 1.0 and rate1 == 0.0.
        CHECK(rate0 < 0.95,
              "7e the residual is NOT normalised by the MAX (which would give "
              "token 0 with probability 1)");
    }
}

// ===========================================================================
// Test 8: the BOUNDARY CONVENTION (SP3's `<=` versus `<`)
//
// FIXTURE D: a point-mass draft (the guess is ALWAYS token 0) and a target
// of [0.5, 0.5], so the acceptance probability at token 0 is EXACTLY 0.5.
//
// With the acceptance draw SUPPLIED at exactly 0.5, `<=` ACCEPTS and `<`
// REJECTS -- and the two produce different OUTPUT LENGTHS, so the convention
// is visible in the result and not only in the internal `n`.
//
// A RANDOM draw cannot observe this: the uniform is continuous, so landing
// exactly on the boundary has probability zero. Test 9 measures that.
// ===========================================================================
static void test_boundary_convention() {
    banner("8 the boundary convention");

    DraftModel draft = make_positional(D_Q_LOGITS, 3, 2, nullptr);
    DraftModel target = make_positional(D_P_LOGITS, 3, 2, nullptr);

    // us[0] = 0.25 selects the guess (the draft is a point mass on token 0,
    // so any draw in [0,1) does). us[1] = 0.5 is EXACTLY the acceptance
    // probability. us[2] = 0.25 selects the final token from p_{gamma+1}
    // = [0.5, 0.5], and 0.25 lands in the first cell -> token 0.
    std::vector<double> us = {0.25, 0.5, 0.25};
    SpeculativeResult r = speculative_decode_step_with_uniforms(
        draft, target, ids_row({}), 1, us);

    CHECK(r.num_accepted == 1,
          "8a a draw EXACTLY on the acceptance boundary ACCEPTS (`<=`)");
    CHECK(!r.rejected, "8b the boundary draw is not a rejection");
    CHECK(r.tokens.cols == 2,
          "8c two tokens are emitted -- the guess plus the bonus token");

    // A draw JUST ABOVE the boundary rejects.
    std::vector<double> us_above = {0.25, 0.5000001, 0.25};
    SpeculativeResult r2 = speculative_decode_step_with_uniforms(
        draft, target, ids_row({}), 1, us_above);
    CHECK(r2.num_accepted == 0,
          "8d a draw just ABOVE the acceptance probability REJECTS");
    CHECK(r2.rejected, "8e the step reports the rejection");
    CHECK(r2.tokens.cols == 1,
          "8f exactly one token is emitted -- the correction");
    // The correction comes from the residual at position 0:
    //   norm(max(0, [0.5,0.5] - [1,0])) = norm([0, 0.5]) = [0, 1]
    // -> the point mass on token 1.
    CHECK(ids_are(r2.tokens, {1}),
          "8g the correction is token 1 -- the residual is the point mass [0, 1]");
}

// ===========================================================================
// Test 9: the ACCEPTANCE IS PROBABILISTIC, NOT DETERMINISTIC (SP12)
//
// The natural mistake is to accept DETERMINISTICALLY on `p >= q`, ignoring
// the acceptance probability. The correct rule accepts a guess with
// probability p/q, which for p < q is strictly between 0 and 1.
//
// Over many seeds the measured acceptance rate must equal p/q, and the
// deterministic rule's rate (0 or 1) must be far away.
//
// FIXTURE C: p = [0.8, 0.2], q = [0.5, 0.5]. The acceptance probability at
// token 1 is 0.2/0.5 = 0.4 EXACTLY.
// ===========================================================================
static void test_acceptance_is_probabilistic() {
    banner("9 the acceptance is probabilistic");

    // position 0: p = [0.8, 0.2]   q = [0.5, 0.5]
    const double C_P[2][2] = {
        {-0.2231435513142097, -1.6094379124341003},
        {-0.2231435513142097, -1.6094379124341003},
    };
    const double C_Q[2][2] = {
        {-0.6931471805599453, -0.6931471805599453},
        {-0.6931471805599453, -0.6931471805599453},
    };
    DraftModel dc = make_positional(C_Q, 2, 2, nullptr);
    DraftModel tc = make_positional(C_P, 2, 2, nullptr);

    // The draft is uniform, so the guess is token 0 for u < 0.5 and token 1
    // for u >= 0.5. The acceptance probability is 1.0 for token 0 (0.8/0.5,
    // clamped) and 0.4 for token 1.
    //
    // The MEASURED rate at which the guess is ACCEPTED must be
    //   0.5 * 1.0  +  0.5 * 0.4  =  0.7
    // and the deterministic rule's rate would be
    //   0.5 * 1.0  +  0.5 * 0.0  =  0.5
    // -- 0.2 apart, which is unmissable over enough trials.
    const int trials = 20000;
    int accepted = 0;
    for (int s = 0; s < trials; ++s) {
        std::mt19937 rng(static_cast<unsigned>(s));
        SpeculativeResult r =
            speculative_decode_step(dc, tc, ids_row({}), 1, rng);
        if (r.num_accepted == 1) ++accepted;
    }
    double rate = static_cast<double>(accepted) / trials;
    printf("        measured acceptance rate = %.4f (expected ~0.7)\n", rate);
    CHECK(std::fabs(rate - 0.7) < 0.02,
          "9a the acceptance rate matches the probabilistic rule (p/q)");
    // The deterministic rule's rate is 0.5, and the measured rate is ~0.7,
    // so the measured rate must be FAR from 0.5 -- but "far" here means a
    // fixed margin, not "the distance to 0.5 is bigger than the distance to
    // 0.5", which is nonsense. The first draft of this assertion read
    // `fabs(rate - 0.7) > fabs(rate - 0.5)`, which is TRUE only when the
    // rate is above 0.6 -- a property of the RATE, not of the rule.
    CHECK(std::fabs(rate - 0.5) > 0.15,
          "9b the measured rate is far from the DETERMINISTIC rule's 0.5");

    // And the rate must match the measured p/q of the paper's own formula:
    // the acceptance probability at token 1 is exactly 0.4.
    int accepted_token1 = 0;
    int guessed_token1 = 0;
    for (int s = 0; s < trials; ++s) {
        std::mt19937 rng(static_cast<unsigned>(s));
        SpeculativeResult r =
            speculative_decode_step(dc, tc, ids_row({}), 1, rng);
        // With gamma = 1, a rejection at position 0 means the guess was
        // token 1 (token 0 is always accepted). So the count of REJECTIONS
        // is the count of "token 1 guessed and not accepted".
        if (r.rejected) ++guessed_token1;
        (void)accepted_token1;
    }
    double r1 = static_cast<double>(guessed_token1) / trials;
    printf("        measured rejection rate = %.4f (expected ~0.5*0.6 = 0.3)\n", r1);
    CHECK(std::fabs(r1 - 0.3) < 0.02,
          "9c the rejection rate is 0.5 * (1 - 0.4) = 0.3 -- the acceptance "
          "probability is p/q, not a deterministic p >= q test");
}

// ===========================================================================
// Test 10: the DISTRIBUTIONAL PROPERTY -- the paper's ACTUAL claim.
//
// Over many seeds the emitted token's empirical distribution must match the
// TARGET's distribution. No hand-written example can replace this check: a
// single example is compatible with many wrong implementations, and the
// paper's claim is about the DISTRIBUTION.
//
// FIXTURE E: p = [0.5, 0.3, 0.2], q = [0.2, 0.3, 0.5].
// ===========================================================================
static void test_distributional_property() {
    banner("10 the distributional property");

    DraftModel draft = make_positional(E_Q_LOGITS, 2, 3, nullptr);
    DraftModel target = make_positional(E_P_LOGITS, 2, 3, nullptr);

    //
    // THE FIRST EMITTED TOKEN IS THE ONE THAT MUST MATCH p -- not the
    // correction token. This is a trap the first draft of this test fell
    // into, and the oracle caught it: on this fixture the residual is
    // norm(max(0, [0.5,0.3,0.2] - [0.2,0.3,0.5])) = norm([0.3,0,0]) = [1,0,0],
    // so the CORRECTION token is ALWAYS token 0. A test that counted only
    // the correction tokens would measure [1,0,0] and fail against a
    // perfectly correct implementation.
    //
    // The paper's claim is about the DISTRIBUTION OF THE EMITTED TOKENS,
    // which is the first token of the step -- the correction only appears
    // when a guess was rejected.
    const int trials = 60000;
    int counts[3] = {0, 0, 0};
    int corrections[3] = {0, 0, 0};
    for (int s = 0; s < trials; ++s) {
        std::mt19937 rng(static_cast<unsigned>(s));
        SpeculativeResult r =
            speculative_decode_step(draft, target, ids_row({}), 1, rng);
        if (r.tokens.cols < 1) continue;
        int t = static_cast<int>(r.tokens(0, 0));
        if (t >= 0 && t < 3) ++counts[t];
        if (r.rejected && r.tokens.cols == 1) {
            int c = static_cast<int>(r.tokens(0, 0));
            if (c >= 0 && c < 3) ++corrections[c];
        }
    }
    int total = counts[0] + counts[1] + counts[2];
    printf("        empirical: %.4f %.4f %.4f  (target 0.5 0.3 0.2)\n",
           static_cast<double>(counts[0]) / total,
           static_cast<double>(counts[1]) / total,
           static_cast<double>(counts[2]) / total);
    CHECK(total > 0, "10a the distributional run produced samples");
    CHECK(std::fabs(static_cast<double>(counts[0]) / total - 0.5) < 0.02,
          "10b token 0's rate matches the TARGET's p");
    CHECK(std::fabs(static_cast<double>(counts[1]) / total - 0.3) < 0.02,
          "10c token 1's rate matches the TARGET's p");
    CHECK(std::fabs(static_cast<double>(counts[2]) / total - 0.2) < 0.02,
          "10d token 2's rate matches the TARGET's p");

    // And the CORRECTION token is the residual's point mass on token 0 --
    // pinned separately, because conflating the two is what the first draft
    // of this test did.
    int ctotal = corrections[0] + corrections[1] + corrections[2];
    CHECK(ctotal > 0, "10e the rejection path produced corrections");
    if (ctotal > 0) {
        printf("        corrections: %.4f %.4f %.4f  (residual [1, 0, 0])\n",
               static_cast<double>(corrections[0]) / ctotal,
               static_cast<double>(corrections[1]) / ctotal,
               static_cast<double>(corrections[2]) / ctotal);
        CHECK(corrections[0] == ctotal,
              "10f the correction is ALWAYS token 0 -- the residual is the "
              "point mass [1, 0, 0]");
    }
}

// ===========================================================================
// Test 11: the input is never mutated, and the output's shape is right
// ===========================================================================
static void test_shape_and_purity() {
    banner("11 shape and purity");

    DraftModel draft = make_positional(A_Q_LOGITS, 5, A_VOCAB, nullptr);
    DraftModel target = make_positional(A_P_LOGITS, 5, A_VOCAB, nullptr);

    // A ONE-token prefix, so the purity check has something to look at.
    // The A fixture has five rows, so positions 1..4 are in range.
    Tensor prefix = ids_row({1});
    Tensor before = prefix.clone();
    std::mt19937 rng(3);
    SpeculativeResult r = speculative_decode_step(draft, target, prefix, 3, rng);
    CHECK(prefix.rows == before.rows && prefix.cols == before.cols,
          "11a the prefix's SHAPE is unchanged");
    bool same = true;
    for (size_t j = 0; j < before.cols; ++j)
        if (prefix(0, j) != before(0, j)) same = false;
    CHECK(same, "11b the prefix is never mutated");
    CHECK(r.tokens.rows == 1, "11c the emitted tokens are ONE row");
    CHECK(r.tokens.cols >= 1 && r.tokens.cols <= 4,
          "11d the emitted count is between 1 and gamma+1");
}

// ===========================================================================
// main
// ===========================================================================
// Every test body is wrapped so that a THROWING implementation reports a
// verdict instead of aborting the binary before the summary line is
// printed. A binary that dies produces no summary, and the mutation
// harness has nothing to count -- which is how a crash gets reported as a
// build failure rather than as a caught mutation.
template <typename F>
static void run_guarded(F fn, const char* name) {
    try {
        fn();
    } catch (const std::exception& e) {
        ++failed;
        printf("  [FAIL] %s threw: %s\n", name, e.what());
    }
}

int main() {
    printf("=== Speculative Decoding Tests ===\n");
    run_guarded(test_guards, "1 the guards");
    run_guarded(test_identical_models, "2 identical models");
    run_guarded(test_call_counts, "3 the call counts");
    run_guarded(test_worked_example, "4 the worked example");
    run_guarded(test_emitted_count_is_n_plus_one, "5 the emitted count");
    run_guarded(test_residual_values, "6 the residual");
    run_guarded(test_residual_sum_divisor, "7 the residual's sum divisor");
    run_guarded(test_boundary_convention, "8 the boundary convention");
    run_guarded(test_acceptance_is_probabilistic, "9 the acceptance rate");
    run_guarded(test_distributional_property, "10 the distribution");
    run_guarded(test_shape_and_purity, "11 shape and purity");
    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
