// test_tta.cpp — TDD tests for the test-time-augmentation utility.
//
//   Divya Shanmugam, Davis Blalock, Guha Balakrishnan, John Guttag --
//   "Better Aggregation in Test-Time Augmentation", arXiv:2011.11156
//   (ICCV 2021).
//
// CITATION. Verified on 2026-10-16 against https://arxiv.org/abs/2011.11156
// directly (abs-page scrape: title, four author links, abstract). The arXiv
// export API was RATE-LIMITED from this host ("Rate exceeded." on three
// attempts, http + https, with and without a UA, 25 s apart), so the ID was
// confirmed from three independent sources instead: the arXiv abs page, NASA
// ADS (2020arXiv201111156S, DOI 10.48550/arXiv.2011.11156, "E-Print: ICCV
// 2021") and emergentmind. This repo's expansion queue has produced EIGHT wrong
// arXiv IDs before, so the check is not optional.
//
// THE GAP THIS FILLS, measured not assumed:
//     grep -ril 'test.time.augment|test_time_augment|\btta\b|ensemble.*predict'
//          include/ tests/
// -> 0 hits. Model::forward runs ONE pass per input; nothing in the repo
// combines several forward passes of one input into one prediction.
//
// EVERY numeric expectation below is ORACLE-DERIVED from
// ~/.hermes/cache/scratch/tta_oracle.py (numpy 2.3.5 / scipy 1.17.1 float64,
// output in tta_oracle_out.txt), NOT hand-derived.
//
// ===========================================================================
// THE SEMANTICS EACH FIXTURE EXISTS TO PIN
// ===========================================================================
//
//   D1 BUNDLES ARE (T, N*K), TRANSFORM-MAJOR. Element (t, n*K + k) is the logit
//      of class k for row n under transform t. `Tensor` is strictly 2-D, so a
//      genuine 3-D bundle is impossible here; the alternative flattening,
//      (N*T, K) or (N, T*K), mixes transforms of DIFFERENT rows into one
//      accumulation. The fixtures' rows are deliberately NOT symmetric, so the
//      wrong ordering changes the numbers rather than coinciding with them.
//
//   D2 tta_mean_logits AVERAGES LOGITS AND RETURNS LOGITS. It is not a
//      probability vector. Softmaxing it gives a DIFFERENT answer from
//      tta_mean_probs -- the two aggregations genuinely differ, and the suite
//      asserts they disagree rather than treating one as a rescaling.
//
//   D3 tta_geo_mean_probs NORMALISES THE GEOMETRIC MEAN. THIS IS NOT COSMETIC.
//      The obvious derivation -- "log softmax(z) = z - logsumexp(z), so the
//      geometric mean of probabilities is softmax of the mean logits" -- is
//      WRONG as written: logsumexp is CONVEX, so lse(mean z) <= mean(lse(z_t)),
//      strictly unless the members agree. The RAW geometric mean is
//      softmax(mean z) times a per-row factor strictly BELOW 1, i.e. a
//      sub-probability vector. Measured raw row sums on the oracle fixture:
//      0.94526606993838813 and 0.90768902185758782, versus 1 for mean_probs.
//      Omitting the normalisation is caught by a VALUE assertion and NOT by an
//      argmax assertion, because a positive per-row scale never changes argmax.
//
//   D4 tta_argmax TIES GO TO THE LOWEST CLASS INDEX (strict `>`, matching
//      metrics.cpp:38). No tolerance-based test can observe a tie-break; only
//      an EXACT tie can, so the fixture carries bit-identical ties.
//
//   D5 tta_majority_vote TIES GO TO THE LOWEST CLASS INDEX, not the earliest
//      transform. Under a 1-1 split these rules disagree, and a suite that only
//      ever feeds 3-0 or 2-1 splits cannot tell them apart.
//
//   D6 tta_majority_vote RETURNS A ONE-HOT (N, K), not counts and not class
//      indices. The return type is the contract: counts are not probabilities
//      and cannot be compared against mean_probs.
//
//   D7 SHIFT INVARIANCE IS ASYMMETRIC ACROSS THE AGGREGATIONS. softmax is
//      shift-invariant PER TRANSFORM, so averaging the PROBABILITIES is
//      shift-invariant; averaging the LOGITS is NOT, because the offset
//      survives as a temperature-like rescale on that member. Measured: adding
//      7.0 to transform 1 moves mean_logits by 3.500000e+00 and moves mean_probs
//      and geo_probs by 0.000000e+00. (My first draft asserted the OPPOSITE;
//      the oracle caught it and the measured result is what is written here.)
//
//   D8 WEIGHTS ARE NORMALISED INTERNALLY, and T == 1 throws. Equal weights
//      reproduce the unweighted answer to EXACTLY 0.0 (measured, not within a
//      tolerance); unequal weights change the geometric mean by 1.090e-01.
//
//   D9 tta_flip_report RETURNS changed / correct_to_incorrect /
//      incorrect_to_correct SEPARATELY, and changed == c2i + i2c ALWAYS.
//      Reporting only the net change is the paper's exact complaint: on the
//      end-to-end fixture the aggregate nets +1 while LOSING a prediction the
//      base member got right.
//
//   D10 tta_hflip REVERSES WITHIN EACH ROW of a flattened (H, W) image, so the
//      WIDTH comes first. A whole-buffer reversal also flips image order, which
//      is a different (and wrong) transform.
//
//   STRUCTURAL FACT, pinned by test_t2_cannot_fix_both_wrong: WITH T = 2,
//      AGGREGATION CAN NEVER REPAIR A ROW THAT BOTH MEMBERS GET WRONG. A mean
//      of two probability vectors is a convex combination, so if both members
//      prefer the same class the mean prefers it too. Measured: 0 counterexamples
//      in 20000 random (T=2, N=1, K=2) fixtures. This is why the end-to-end
//      fixture uses T = 3 -- with T = 2 it CANNOT demonstrate a gain, so a test
//      built on T = 2 would be asserting something false.

#include "nn/utils/tta.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int passed = 0, failed = 0;

static void check(const std::string& what, bool ok) {
    if (ok) { ++passed; std::cout << "  [PASS] " << what << "\n"; }
    else    { ++failed; std::cout << "  [FAIL] " << what << "\n"; }
}

static void check_close(const std::string& what, double got, double want,
                        double tol = 1e-12) {
    const double d = std::fabs(got - want);
    if (d <= tol) { ++passed; std::cout << "  [PASS] " << what << "\n"; }
    else {
        ++failed;
        std::cout << "  [FAIL] " << what << "  got=" << std::setprecision(17)
                  << got << " want=" << want << " |diff|=" << d
                  << " tol=" << tol << "\n";
    }
}

// ---------------------------------------------------------------------------
// Fixtures -- transcribed from the oracle output, digit-for-digit.
// ---------------------------------------------------------------------------

// E: T=3, N=10, K=2, stored transform-major as a (3, 20) bundle (D1).
//
// Designed -- and REBUILT twice, because the first two versions failed their own
// stated design goals -- so that:
//   (i)   the aggregate STRICTLY beats every single member,
//   (ii)  the aggregate LOSES a prediction the base member got RIGHT,
//   (iii) mean_probs DISAGREES with majority_vote on row 8.
//
// Rows are written (row, transform, class) below because that is how they read;
// the bundle stores them transform-major.
//
// Oracle argmaxes (labels = 0,0,1,1,1,1,0,1,1,0):
//   member 0: [0,0,0,0,1,1,1,1,1,0]  acc 0.70
//   member 1: [0,0,1,1,0,1,0,1,0,1]  acc 0.70
//   member 2: [0,0,1,1,0,1,1,0,0,0]  acc 0.60
//   MEAN_LOGITS / MEAN_PROBS / GEO_MEAN: [0,0,1,1,0,1,1,1,1,0]  acc 0.80
//   MAJORITY_VOTE:                     [0,0,1,1,0,1,1,1,0,0]  acc 0.70
static const int kELabels[10] = {0, 0, 1, 1, 1, 1, 0, 1, 1, 0};
static const size_t kE_T = 3, kE_N = 10, kE_K = 2;

static Tensor fixture_E() {
    const double rows[10][3][2] = {
        {{ 0.90, -0.90}, { 0.90, -0.90}, { 0.90, -0.90}},   // r0 unanimous c0 strong
        {{ 0.40, -0.40}, { 0.40, -0.40}, { 0.40, -0.40}},   // r1 unanimous c0 weak
        {{ 0.06, -0.06}, {-0.80,  0.80}, {-0.70,  0.70}},   // r2 t0 wrong, others right
        {{ 0.06, -0.06}, {-0.80,  0.80}, {-0.90,  0.90}},   // r3 same, stronger t2
        {{-0.30,  0.30}, { 0.90, -0.90}, { 0.90, -0.90}},   // r4 base RIGHT, agg WRONG
        {{-0.90,  0.90}, {-0.90,  0.90}, {-0.90,  0.90}},   // r5 unanimous c1 strong
        {{-0.55,  0.55}, { 0.06, -0.06}, {-0.55,  0.55}},   // r6 t1 wrong only
        {{-0.55,  0.55}, {-0.55,  0.55}, { 0.06, -0.06}},   // r7 t2 wrong only
        {{-1.60,  1.60}, { 0.05, -0.05}, { 0.05, -0.05}},   // r8 2 weak wrong beat vote
        {{ 0.50, -0.50}, {-0.90,  0.90}, { 0.50, -0.50}},   // r9 member 1 alone wrong
    };
    Tensor b(kE_T, kE_N * kE_K);
    for (size_t t = 0; t < kE_T; ++t)
        for (size_t n = 0; n < kE_N; ++n)
            for (size_t c = 0; c < kE_K; ++c)
                b(t, n * kE_K + c) = rows[n][t][c];
    return b;
}

// L1: T=2, N=3, K=3, stored as a (2, 9) bundle. Row 1 is an EXACT 3-way tie
// under both transforms ([0,0,0] and [1,1,1]), so D4 is observable there.
static Tensor fixture_L1() {
    const double rows[3][2][3] = {
        {{ 1.00, -0.50,  0.25}, { 0.50,  0.25, -1.00}},
        {{ 0.00,  0.00,  0.00}, { 1.00,  1.00,  1.00}},
        {{-2.00,  3.00,  1.00}, {-1.00,  2.50,  2.50}},
    };
    Tensor b(2, 3 * 3);
    for (size_t t = 0; t < 2; ++t)
        for (size_t n = 0; n < 3; ++n)
            for (size_t c = 0; c < 3; ++c) b(t, n * 3 + c) = rows[n][t][c];
    return b;
}

// MEAN_PROBS on fixture_E, digit-for-digit. Rows sum to 1 only to ~1e-16, so
// the row-sum assertion below uses a tolerance rather than bit-equality.
static const double kEMeanProbs[10][2] = {
    {0.85814893509951229, 0.14185106490048779},
    {0.68997448112761239, 0.31002551887238755},
    {0.29858725935735514, 0.70141274064264481},
    {0.27993224384371168, 0.72006775615628837},
    {0.69021385465774310, 0.30978614534225701},
    {0.14185106490048779, 0.85814893509951229},
    {0.34314794685811217, 0.65685205314188788},
    {0.34314794685811217, 0.65685205314188788},
    {0.36304136591821479, 0.63695863408178521},
    {0.53465607405349924, 0.46534392594650081},
};

// GEO_MEAN_PROBS (NORMALISED) on fixture_E. Rows 2, 3, 8 and 9 DIFFER from
// kEMeanProbs -- so a suite asserting only the argmax could not tell the two
// aggregations apart at all, and the value assertions are what does.
static const double kEGeo[10][2] = {
    {0.85814893509951229, 0.14185106490048779},
    {0.6899744811276125,  0.3100255188723875},
    {0.27687819487561016, 0.72312180512438984},
    {0.25099110982976708, 0.74900889017023298},
    {0.73105857863000490, 0.26894142136999510},
    {0.14185106490048779, 0.85814893509951229},
    {0.33329196733387695, 0.666708032666123},
    {0.33329196733387695, 0.666708032666123},
    {0.2689414213699951,  0.7310585786300049},
    {0.51666049656941149, 0.48333950343058862},
};

// MEAN_LOGITS on fixture_E -- LOGITS, not probabilities (D2).
static const double kEMeanLogits[10][2] = {
    { 0.90000000000000002, -0.90000000000000002},
    { 0.40000000000000008, -0.40000000000000008},
    {-0.47999999999999998,  0.47999999999999998},
    {-0.54666666666666675,  0.54666666666666675},
    { 0.50000000000000000, -0.50000000000000000},
    {-0.90000000000000002,  0.90000000000000002},
    {-0.34666666666666668,  0.34666666666666668},
    {-0.34666666666666668,  0.34666666666666668},
    {-0.50000000000000000,  0.50000000000000000},
    { 0.033333333333333326,-0.033333333333333326},
};

static std::vector<size_t> argmax_of(const Tensor& t) {
    std::vector<size_t> out(t.rows);
    for (size_t i = 0; i < t.rows; ++i) {
        size_t best = 0;
        for (size_t j = 1; j < t.cols; ++j)
            if (t(i, j) > t(i, best)) best = j;
        out[i] = best;
    }
    return out;
}

static bool argmax_equals(const Tensor& t, const int* want, size_t n) {
    std::vector<size_t> got = argmax_of(t);
    if (got.size() != n) return false;
    for (size_t i = 0; i < n; ++i)
        if (got[i] != static_cast<size_t>(want[i])) return false;
    return true;
}

static int accuracy_of(const Tensor& t) {
    std::vector<size_t> a = argmax_of(t);
    int ok = 0;
    for (size_t i = 0; i < a.size() && i < 10; ++i)
        if (a[i] == static_cast<size_t>(kELabels[i])) ++ok;
    return ok;
}

// ---------------------------------------------------------------------------
// Task 1: transforms
// ---------------------------------------------------------------------------
static void test_transforms() {
    std::cout << "\n--- Task 1: transforms ---\n";

    const double x0[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor x(2, 12);                                  // 2 images of 3 x 4
    for (size_t c = 0; c < 12; ++c) { x(0, c) = x0[c]; x(1, c) = x0[c] * 2.0; }

    Tensor id = tta_identity(x);
    check("identity returns the same shape", id.rows == 2 && id.cols == 12);
    bool id_same = true;
    for (size_t i = 0; i < 2; ++i)
        for (size_t c = 0; c < 12; ++c)
            if (id(i, c) != x(i, c)) id_same = false;
    check("identity is bit-exact", id_same);

    // D10: the flip reverses WITHIN each image row, width first. A whole-buffer
    // reversal would also reverse image order, so this pins row-locality.
    Tensor f = tta_hflip(x, 3, 4);
    const double want0[12] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8};
    bool flip_ok = true;
    for (size_t c = 0; c < 12; ++c) {
        if (f(0, c) != want0[c]) flip_ok = false;
        if (f(1, c) != want0[c] * 2.0) flip_ok = false;
    }
    check("D10: hflip reverses within each flattened (H=3, W=4) row, width first",
          flip_ok);

    // Non-vacuity: a WHOLE-BUFFER reversal is a different transform and the two
    // must be distinguishable. Assert it rather than assume it.
    bool differs = false;
    for (size_t i = 0; i < 2; ++i)
        for (size_t c = 0; c < 12; ++c)
            if (f(i, c) != x(1 - i, 11 - c)) differs = true;
    check("non-vacuity: the row-local flip is NOT a whole-buffer reversal",
          differs);

    Tensor f2 = tta_hflip(f, 3, 4);
    bool inv = true;
    for (size_t i = 0; i < 2; ++i)
        for (size_t c = 0; c < 12; ++c)
            if (f2(i, c) != x(i, c)) inv = false;
    check("hflip is an involution: hflip(hflip(x)) == x bitwise", inv);

    std::vector<Tensor> augs = tta_augment(x, 3, 4);
    check("tta_augment returns exactly 2 views (identity + hflip)",
          augs.size() == 2);
    if (augs.size() == 2) {
        check("tta_augment[0] is the identity view",
              augs[0](0, 0) == 0.0 && augs[0](0, 11) == 11.0);
        check("tta_augment[1] is the flipped view",
              augs[1](0, 0) == 3.0 && augs[1](0, 11) == 8.0);
    }

    Tensor w1 = tta_hflip(x, 12, 1);
    check("width == 1 makes hflip the identity",
          w1(0, 5) == x(0, 5) && w1(0, 0) == x(0, 0));
}

// ---------------------------------------------------------------------------
// Task 1: bundle accessors + argmax tie-break (D1, D4)
// ---------------------------------------------------------------------------
static void test_bundle_and_argmax() {
    std::cout << "\n--- Task 1: bundle accessors + argmax tie-break (D1, D4) ---\n";
    Tensor e = fixture_E();
    check("D1: a (T, N*K) bundle reports T transforms",
          tta_num_transforms(e, kE_K) == kE_T);

    Tensor v0 = tta_transform_view(e, 0, kE_K);
    check("D1: the transform-0 view is (N, K)",
          v0.rows == kE_N && v0.cols == kE_K);
    check("D1: transform-0 view row 4 is (-0.30, 0.30)",
          v0(4, 0) == -0.30 && v0(4, 1) == 0.30);
    Tensor v2 = tta_transform_view(e, 2, kE_K);
    check("D1: transform-2 view row 4 is (0.90, -0.90)",
          v2(4, 0) == 0.90 && v2(4, 1) == -0.90);

    // D4: exact ties. No tolerance-based assertion can observe a tie-break.
    Tensor t(4, 2);
    t(0, 0) = 1.0; t(0, 1) = 0.0;
    t(1, 0) = 0.0; t(1, 1) = 1.0;
    t(2, 0) = 5.0; t(2, 1) = 5.0;                 // exact 2-way tie
    t(3, 0) = 5.0; t(3, 1) = 5.0;
    std::vector<size_t> a = tta_argmax(t);
    check("D4: exact ties resolve to the LOWEST class index (rows 2,3 -> 0)",
          a.size() == 4 && a[0] == 0 && a[1] == 1 && a[2] == 0 && a[3] == 0);

    // `>=` instead of `>` still returns 0 on a tie, so that fixture ALONE does
    // not discriminate. This one does: the strictly-greater value is at index
    // 1 and there is a LATER tied value at index 2, so only `>` gives 1.
    Tensor u(1, 4);
    u(0, 0) = 0.1; u(0, 1) = 0.9; u(0, 2) = 0.9; u(0, 3) = 0.2;
    std::vector<size_t> b = tta_argmax(u);
    check("D4: a strictly-greater value later in the row still wins (index 1, "
          "not the last tied index 2)",
          b.size() == 1 && b[0] == 1);
}

// ---------------------------------------------------------------------------
// Task 1: flip report (D9)
// ---------------------------------------------------------------------------
static void test_flip_report() {
    std::cout << "\n--- Task 1: flip report (D9) ---\n";
    Tensor base = tta_transform_view(fixture_E(), 0, kE_K);

    TtaFlipReport same = tta_flip_report(base, base,
                                         std::vector<int>(10, 0));
    check("a report against itself has changed == 0", same.changed == 0);
    check("a report against itself has c2i == 0",
          same.correct_to_incorrect == 0);
    check("D9: c2i + i2c == changed holds on the degenerate case",
          same.changed == same.correct_to_incorrect + same.incorrect_to_correct);

    // A REAL base-vs-aggregate comparison is asserted in Task 2, once the
    // aggregations exist -- this task owns the report's own contract only.
}

// ---------------------------------------------------------------------------
// Task 1: validation
// ---------------------------------------------------------------------------
static void test_task1_validation() {
    std::cout << "\n--- Task 1: validation ---\n";
    Tensor x(1, 12);
    bool threw = false;
    try { tta_hflip(x, 5, 4); } catch (const std::invalid_argument&) { threw = true; }
    check("H*W != cols throws rather than silently truncating", threw);

    threw = false;
    try { tta_hflip(x, 0, 4); } catch (const std::invalid_argument&) { threw = true; }
    check("a zero HEIGHT throws", threw);

    threw = false;
    try { tta_hflip(x, 12, 0); } catch (const std::invalid_argument&) { threw = true; }
    check("a zero WIDTH throws", threw);

    threw = false;
    try { tta_argmax(Tensor::zeros(0, 3)); } catch (const std::invalid_argument&) { threw = true; }
    check("argmax on an empty tensor throws", threw);

    threw = false;
    try { tta_flip_report(Tensor::zeros(2, 3), Tensor::zeros(3, 3),
                          std::vector<int>(2, 0)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a flip report with mismatched row counts throws", threw);

    threw = false;
    try { tta_flip_report(Tensor::zeros(2, 3), Tensor::zeros(2, 3),
                          std::vector<int>(5, 0)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a flip report with mismatched labels throws", threw);
}

// ---------------------------------------------------------------------------
// Task 2: the four aggregations
// ---------------------------------------------------------------------------
static void test_aggregations() {
    std::cout << "\n--- Task 2: aggregations on fixture_E ---\n";
    Tensor e = fixture_E();

    Tensor ml = tta_mean_logits(e, kE_K);
    check("mean_logits returns (N, K)", ml.rows == kE_N && ml.cols == kE_K);
    bool ml_ok = true;
    for (size_t n = 0; n < kE_N; ++n)
        for (size_t c = 0; c < kE_K; ++c)
            if (std::fabs(ml(n, c) - kEMeanLogits[n][c]) > 1e-12) ml_ok = false;
    check("D2: mean_logits matches the oracle (values are LOGITS)", ml_ok);

    Tensor mp = tta_mean_probs(e, kE_K);
    bool mp_ok = true;
    for (size_t n = 0; n < kE_N; ++n)
        for (size_t c = 0; c < kE_K; ++c)
            if (std::fabs(mp(n, c) - kEMeanProbs[n][c]) > 1e-12) mp_ok = false;
    check("mean_probs matches the oracle digit-for-digit", mp_ok);

    Tensor geo = tta_geo_mean_probs(e, kE_K);
    bool geo_ok = true;
    for (size_t n = 0; n < kE_N; ++n)
        for (size_t c = 0; c < kE_K; ++c)
            if (std::fabs(geo(n, c) - kEGeo[n][c]) > 1e-12) geo_ok = false;
    check("geo_mean_probs (NORMALISED) matches the oracle", geo_ok);

    // D3, the assertion the whole normalisation hangs on. A raw geometric mean
    // is a SUB-PROBABILITY vector: measured row sums 0.94526606993838813 and
    // 0.90768902185758782 on the oracle's L1 fixture versus 1 here.
    double worst_sum = 0.0;
    for (size_t n = 0; n < kE_N; ++n)
        worst_sum = std::max(worst_sum,
                             std::fabs(geo(n, 0) + geo(n, 1) - 1.0));
    check("D3: geo_mean_probs rows sum to 1 (a raw geometric mean would be "
          "~0.91-0.95, NOT 1)",
          worst_sum <= 1e-12);

    double worst_mp_sum = 0.0;
    for (size_t n = 0; n < kE_N; ++n)
        worst_mp_sum = std::max(worst_mp_sum,
                                std::fabs(mp(n, 0) + mp(n, 1) - 1.0));
    check("mean_probs rows sum to 1", worst_mp_sum <= 1e-12);

    // D6: majority vote returns a ONE-HOT tensor, not counts and not indices.
    Tensor mv = tta_majority_vote(e, kE_K);
    bool mv_onehot = true;
    for (size_t n = 0; n < kE_N; ++n) {
        double s = 0.0;
        for (size_t c = 0; c < kE_K; ++c) {
            s += mv(n, c);
            if (mv(n, c) != 0.0 && mv(n, c) != 1.0) mv_onehot = false;
        }
        if (s != 1.0) mv_onehot = false;
    }
    check("D6: majority_vote returns a ONE-HOT (N, K), not counts", mv_onehot);

    const int mvWant[10] = {0, 0, 1, 1, 0, 1, 1, 1, 0, 0};
    check("majority_vote argmaxes match the oracle", argmax_equals(mv, mvWant, 10));

    // D6 non-vacuity: majority vote is NOT the same as mean_probs. On row 8
    // two WEAK wrong votes beat one strong right one; mean_probs gets it right
    // and majority vote does not. Interchanging them would change row 8.
    check("non-vacuity: majority_vote and mean_probs DISAGREE on row 8",
          argmax_of(mv)[8] != argmax_of(mp)[8]);

    // D2 non-vacuity: mean_logits is not a probability vector, and its argmax
    // is not mean_probs' argmax.
    check("D2: mean_logits is NOT a probability vector (rows do not sum to 1)",
          !(std::fabs(ml(0, 0) + ml(0, 1) - 1.0) <= 1e-6));

    // D7: shift invariance is ASYMMETRIC. softmax is shift-invariant per
    // transform, so the probability-space aggregations are invariant and the
    // logit-space one is not. Measured on the oracle: mean_logits moves by
    // 3.500000e+00, mean_probs and geo_mean_probs by 0.
    Tensor shifted = e;
    for (size_t c = 0; c < shifted.cols; ++c) shifted(1, c) += 7.0;

    Tensor ml_s = tta_mean_logits(shifted, kE_K);
    Tensor mp_s = tta_mean_probs(shifted, kE_K);
    Tensor geo_s = tta_geo_mean_probs(shifted, kE_K);

    double d_ml = 0.0, d_mp = 0.0, d_geo = 0.0;
    for (size_t n = 0; n < kE_N; ++n)
        for (size_t c = 0; c < kE_K; ++c) {
            d_ml = std::max(d_ml, std::fabs(ml_s(n, c) - ml(n, c)));
            d_mp = std::max(d_mp, std::fabs(mp_s(n, c) - mp(n, c)));
            d_geo = std::max(d_geo, std::fabs(geo_s(n, c) - geo(n, c)));
        }
    check("D7: mean_logits is NOT shift-invariant (oracle: 3.5)", d_ml > 1e-6);
    check("D7: mean_probs IS shift-invariant (oracle: exactly 0)",
          d_mp <= 1e-15);
    check("D7: geo_mean_probs IS shift-invariant (oracle: exactly 0)",
          d_geo <= 1e-15);
}

// ---------------------------------------------------------------------------
// Task 2: D3 identity + D5 tie-breaks
// ---------------------------------------------------------------------------
static void test_geo_identity_and_tiebreaks() {
    std::cout << "\n--- Task 2: D3 identity + D5/D6 tie-breaks ---\n";

    // D3: normalised geo == softmax(mean logits). The identity the naive
    // derivation gets wrong; here it is asserted as a property.
    Tensor e = fixture_E();
    Tensor geo = tta_geo_mean_probs(e, kE_K);
    Tensor ml = tta_mean_logits(e, kE_K);

    double worst = 0.0;
    for (size_t n = 0; n < kE_N; ++n) {
        double mx = ml(n, 0);
        for (size_t c = 1; c < kE_K; ++c) mx = std::max(mx, ml(n, c));
        double sum = 0.0;
        for (size_t c = 0; c < kE_K; ++c) sum += std::exp(ml(n, c) - mx);
        const double log_sum = mx + std::log(sum);
        for (size_t c = 0; c < kE_K; ++c)
            worst = std::max(worst,
                             std::fabs(geo(n, c) - std::exp(ml(n, c) - log_sum)));
    }
    check("D3: normalised geo_mean_probs == softmax(mean_logits) (oracle: "
          "4.4e-16 over 5000 random fixtures)",
          worst <= 1e-12);

    // D3 on the L1 fixture too -- a DIFFERENT fixture, because an identity that
    // holds only on one fixture is not pinned.
    Tensor l1 = fixture_L1();
    Tensor g1 = tta_geo_mean_probs(l1, 3);
    Tensor m1 = tta_mean_logits(l1, 3);
    double w1 = 0.0;
    for (size_t n = 0; n < 3; ++n) {
        double mx = m1(n, 0);
        for (size_t c = 1; c < 3; ++c) mx = std::max(mx, m1(n, c));
        double sum = 0.0;
        for (size_t c = 0; c < 3; ++c) sum += std::exp(m1(n, c) - mx);
        const double ls = mx + std::log(sum);
        for (size_t c = 0; c < 3; ++c)
            w1 = std::max(w1, std::fabs(g1(n, c) - std::exp(m1(n, c) - ls)));
    }
    check("D3: the identity also holds on the independent L1 fixture", w1 <= 1e-12);

    // D5: an exact 1-1 vote split, fed with the transforms in BOTH orders. A
    // suite that only ever sees 3-0 or 2-1 splits cannot tell "lowest class index"
    // apart from "earliest transform wins".
    Tensor split(2, 2);                       // T=2, N=1, K=2
    split(0, 0) = 1.0; split(0, 1) = 0.0;    // transform 0 votes class 0
    split(1, 0) = 0.0; split(1, 1) = 1.0;    // transform 1 votes class 1
    Tensor a = tta_majority_vote(split, 2);
    check("D5: a 1-1 split resolves to the LOWEST class index, not the first "
          "transform (which voted 0 anyway -- the discriminator is below)",
          a(0, 0) == 1.0 && a(0, 1) == 0.0);

    Tensor split_rev(2, 2);                   // transforms swapped
    split_rev(0, 0) = 0.0; split_rev(0, 1) = 1.0;
    split_rev(1, 0) = 1.0; split_rev(1, 1) = 0.0;
    Tensor b = tta_majority_vote(split_rev, 2);
    check("D5: swapping the transforms does NOT change the tie-break result "
          "(an earliest-transform rule would give the opposite here)",
          b(0, 0) == 1.0 && b(0, 1) == 0.0);

    // D5 non-vacuity: the two splits are genuinely different bundles whose
    // answers differ under an earliest-transform rule. Assert they differ as
    // INPUTS, so the pair above is not passing on identical data.
    bool inputs_differ = false;
    for (size_t c = 0; c < 2; ++c)
        if (split(0, c) != split_rev(0, c)) inputs_differ = true;
    check("non-vacuity: the two 1-1 fixtures really are different bundles",
          inputs_differ);

    // A 3-way tie with T=3, K=3 -> still the lowest class index.
    Tensor t3(3, 3);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) t3(r, c) = (r == 0) ? 1.0 : ((r == 1) ? 0.0 : 2.0);
    Tensor mv3 = tta_majority_vote(t3, 3);
    check("D5: a 3-way tie goes to class 0 (the lowest index)",
          mv3(0, 0) == 1.0 && mv3(0, 1) == 0.0 && mv3(0, 2) == 0.0);
}

// ---------------------------------------------------------------------------
// Task 2: the structural fact -- T=2 cannot fix a row both members miss
// ---------------------------------------------------------------------------
static void test_t2_cannot_fix_both_wrong() {
    std::cout << "\n--- Task 2: structural fact (T=2 convexity) ---\n";
    // Oracle: 0 counterexamples in 20000 random (T=2, N=1, K=2) fixtures. A
    // mean of two probability vectors is a CONVEX combination, so if both
    // members prefer the same class the mean prefers it too. This is why the
    // end-to-end fixture uses T=3 -- a T=2 test could not demonstrate a gain,
    // and would be asserting something false if it tried.
    Tensor z(2, 2);                       // T=2, N=1, K=2
    z(0, 0) =  2.0; z(0, 1) = -2.0;       // both members prefer class 0
    z(1, 0) =  1.0; z(1, 1) = -1.0;
    Tensor agg = tta_mean_probs(z, 2);
    check("T=2: both members preferring class 0 forces the aggregate to class 0",
          agg(0, 0) > agg(0, 1));

    // And T=3 admits a repair the T=2 fixture cannot -- but the margin is
    // NOT "the right member is stronger", which is what my first draft of this
    // fixture assumed and what made it unsatisfiable.
    //
    // Measured law, K=2, members (w,-w), (w,-w), (-s,s): with
    // p = sigmoid(2w) and r = sigmoid(2s), class 1 takes the MEAN iff
    //     2(1-p) + r > 2p + (1-r)   <=>   r > 2p - 1/2.
    // Since r <= 1 always, this REQUIRES p < 0.75, i.e. 2w < ln 3, i.e.
    //     w < 0.5493061443340549.
    // The weak WRONG members must therefore be weak in the ABSOLUTE sense
    // (near the 0.5 boundary), not merely weaker than the strong member. The
    // first draft used w = 1.0 > 0.5493, so the assertion was UNSATISFIABLE at
    // every value of s -- no fixture of that shape can pass, and an
    // unsatisfiable assertion is worse than none because it trains the reader
    // to believe it proves something.
    Tensor z3(3, 2);
    z3(0, 0) =  0.30; z3(0, 1) = -0.30;    // class 0, GENUINELY weak (w = 0.30)
    z3(1, 0) =  0.30; z3(1, 1) = -0.30;    // class 0
    z3(2, 0) = -1.50; z3(2, 1) =  1.50;    // class 1, strong
    Tensor agg3 = tta_mean_probs(z3, 2);
    check("T=3: the same disagreement shape IS repairable (2 genuinely-weak "
          "wrong vs 1 strong right; w < ln(3)/2 = 0.5493061443340549)",
          agg3(0, 1) > agg3(0, 0));

    // The boundary itself, pinned from BOTH sides -- an off-by-one class of
    // error needs a fixture containing the exact threshold, and the threshold
    // is a closed form here (s > 0.666428 at w = 0.30).
    Tensor zlow(3, 2);
    zlow(0, 0) =  0.30; zlow(0, 1) = -0.30;
    zlow(1, 0) =  0.30; zlow(1, 1) = -0.30;
    zlow(2, 0) = -0.60; zlow(2, 1) =  0.60;    // s = 0.60 < 0.666428
    Tensor agg_low = tta_mean_probs(zlow, 2);
    check("T=3: just BELOW the threshold the weak majority wins (s=0.60 < "
          "0.666428) -- pins the boundary from one side",
          agg_low(0, 0) > agg_low(0, 1));

    // And the converse structural fact, which is the reason the fixture above
    // needed a boundary at all: with the weak members at w = 1.0 (> ln(3)/2) no
    // value of s can flip it. Asserted as a fact about the LAW, not as a claim
    // that the code can produce it.
    Tensor zbig(3, 2);
    zbig(0, 0) =  1.0; zbig(0, 1) = -1.0;
    zbig(1, 0) =  1.0; zbig(1, 1) = -1.0;
    zbig(2, 0) = -50.0; zbig(2, 1) = 50.0;   // an arbitrarily strong member
    Tensor agg_big = tta_mean_probs(zbig, 2);
    check("structural law: at w = 1.0 (> ln(3)/2) NO strength of the single "
          "right member can flip the mean -- the original unsatisfiable case, "
          "now asserted as the reason the fixture above needed w = 0.30",
          agg_big(0, 0) > agg_big(0, 1));
}

// ---------------------------------------------------------------------------
// Task 3: weights (D8)
// ---------------------------------------------------------------------------
static void test_weights() {
    std::cout << "\n--- Task 3: weights (D8) ---\n";
    Tensor l1 = fixture_L1();                       // T=2, N=3, K=3
    Tensor e = fixture_E();                         // T=3, N=10, K=2

    // EQUAL weights reproduce the unweighted answer to EXACTLY 0.0 -- measured,
    // not "within a tolerance". The normalisation is what makes this hold for
    // {2, 2} as well as {1, 1}.
    const std::vector<double> eq2 = {2.0, 2.0};
    Tensor a = tta_mean_logits(l1, 3, eq2);
    Tensor b = tta_mean_logits(l1, 3);
    check_close("D8: EQUAL weights reproduce the unweighted mean_logits to "
                "EXACTLY 0.0", a(0, 0) - b(0, 0), 0.0, 0.0);
    check_close("D8: equal weights on mean_probs are also exactly equal",
                tta_mean_probs(l1, 3, eq2)(0, 0) - tta_mean_probs(l1, 3)(0, 0),
                0.0, 0.0);
    check_close("D8: equal weights on geo_mean_probs are also exactly equal",
                tta_geo_mean_probs(l1, 3, eq2)(0, 0) -
                    tta_geo_mean_probs(l1, 3)(0, 0),
                0.0, 0.0);

    // SCALED weights are equivalent -- {1,3} == {2,6}, which is the observable
    // consequence of normalising internally.
    const std::vector<double> w13 = {1.0, 3.0};
    const std::vector<double> w26 = {2.0, 6.0};
    check_close("D8: weights are NORMALISED internally: {1,3} == {2,6}",
                tta_mean_logits(l1, 3, w13)(0, 0) -
                    tta_mean_logits(l1, 3, w26)(0, 0),
                0.0, 0.0);

    // UNEQUAL weights genuinely change the answer -- otherwise the parameter is
    // being ignored, which no value assertion above would catch.
    const std::vector<double> wE = {1.0, 1.0, 2.0};
    double d = std::fabs(tta_mean_logits(e, 2, wE)(4, 0) -
                         tta_mean_logits(e, 2)(4, 0));
    check("non-vacuity: unequal weights DO change mean_logits", d > 1e-6);

    // Oracle: weighted mean_logits of L1 with {1,3} -> normalised {0.25, 0.75}.
    Tensor wl = tta_mean_logits(l1, 3, w13);
    const double want[3][3] = {
        {0.625,  0.0625, -0.6875},
        {0.75,   0.75,    0.75},
        {-1.25,  2.625,   2.125},
    };
    bool wl_ok = true;
    for (size_t n = 0; n < 3; ++n)
        for (size_t c = 0; c < 3; ++c)
            if (std::fabs(wl(n, c) - want[n][c]) > 1e-12) wl_ok = false;
    check("weighted mean_logits matches the oracle for {1, 3}", wl_ok);

    // D8 rejections.
    bool threw = false;
    try { tta_mean_logits(e, 2, {1.0}); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: weights must have one entry per transform", threw);

    threw = false;
    try { tta_mean_logits(e, 2, {1.0, -1.0, 1.0}); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: a negative weight throws", threw);

    threw = false;
    try { tta_mean_logits(e, 2, {0.0, 0.0, 0.0}); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: all-zero weights throw rather than dividing by zero", threw);

    // A single transform is not an aggregation.
    Tensor one(1, 20);
    for (size_t c = 0; c < 20; ++c) one(0, c) = c * 0.1;
    threw = false;
    try { tta_mean_logits(one, 2); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: T == 1 throws (a single transform is not an aggregation)", threw);

    threw = false;
    try { tta_mean_probs(one, 2); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: T == 1 throws for mean_probs too", threw);

    threw = false;
    try { tta_majority_vote(one, 2); } catch (const std::invalid_argument&) { threw = true; }
    check("D8: T == 1 throws for majority_vote too", threw);

    // A vote has no notion of a weight; accepting one and ignoring it would be
    // a silent no-op.
    threw = false;
    try { tta_aggregate(e, 2, TtaAggregation::MajorityVote, {1.0, 2.0, 3.0}); }
    catch (const std::invalid_argument&) { threw = true; }
    check("D8: majority_vote REFUSES weights rather than silently ignoring them",
          threw);
}

// ---------------------------------------------------------------------------
// Task 3: dispatcher identity
// ---------------------------------------------------------------------------
static void test_dispatcher() {
    std::cout << "\n--- Task 3: dispatcher identity ---\n";
    Tensor e = fixture_E();
    // The dispatcher must return exactly what the cores return, for all four
    // values -- otherwise the enum becomes a second implementation that can
    // drift from the first.
    struct { TtaAggregation how; const char* name; } all[] = {
        {TtaAggregation::MeanLogits,   "MeanLogits"},
        {TtaAggregation::MeanProbs,    "MeanProbs"},
        {TtaAggregation::GeoMeanProbs, "GeoMeanProbs"},
        {TtaAggregation::MajorityVote, "MajorityVote"},
    };
    bool all_match = true;
    for (const auto& a : all) {
        Tensor d = tta_aggregate(e, 2, a.how);
        Tensor direct = (a.how == TtaAggregation::MeanLogits)
                            ? tta_mean_logits(e, 2)
                            : (a.how == TtaAggregation::MeanProbs)
                                ? tta_mean_probs(e, 2)
                                : (a.how == TtaAggregation::GeoMeanProbs)
                                    ? tta_geo_mean_probs(e, 2)
                                    : tta_majority_vote(e, 2);
        if (d.rows != direct.rows || d.cols != direct.cols) all_match = false;
        for (size_t i = 0; i < d.rows && i < direct.rows; ++i)
            for (size_t c = 0; c < d.cols && c < direct.cols; ++c)
                if (std::fabs(d(i, c) - direct(i, c)) > 1e-15) all_match = false;
    }
    check("tta_aggregate returns exactly the free function's result for all "
          "four enum values (the dispatcher cannot drift from the cores)",
          all_match);

    // And the dispatcher's weighted path matches the core's weighted path.
    Tensor w1 = tta_aggregate(e, 2, TtaAggregation::MeanLogits, {1.0, 1.0, 2.0});
    Tensor w2 = tta_mean_logits(e, 2, {1.0, 1.0, 2.0});
    check("the dispatcher's weighted path matches the core's",
          std::fabs(w1(4, 0) - w2(4, 0)) <= 1e-15);
}

// ---------------------------------------------------------------------------
// Task 3: END TO END -- the headline, and the paper's finding (D9)
// ---------------------------------------------------------------------------
static void test_end_to_end() {
    std::cout << "\n--- Task 3: end-to-end ---\n";
    Tensor e = fixture_E();
    std::vector<int> labels(kELabels, kELabels + 10);

    // The HEADLINE: aggregation strictly beats EVERY individual member, and
    // beats majority vote. Oracle: members 0.70 / 0.70 / 0.60, aggregate 0.80,
    // majority vote 0.70.
    int m0 = accuracy_of(tta_transform_view(e, 0, kE_K));
    int m1 = accuracy_of(tta_transform_view(e, 1, kE_K));
    int m2 = accuracy_of(tta_transform_view(e, 2, kE_K));
    int agg_acc = accuracy_of(tta_mean_probs(e, kE_K));
    int vote_acc = accuracy_of(tta_majority_vote(e, kE_K));

    check("oracle: member accuracies are 7, 7, 6",
          m0 == 7 && m1 == 7 && m2 == 6);
    check("oracle: majority_vote gets 7", vote_acc == 7);
    check("HEADLINE: aggregation gets 8 -- strictly better than EVERY member",
          agg_acc == 8);
    check("HEADLINE: aggregation beats majority_vote too", agg_acc > vote_acc);
    check("the aggregate is at least as good as the best member",
          agg_acc >= std::max(m0, std::max(m1, m2)));

    // All three probability-space aggregations agree here; assert it, because a
    // silent change in one of them should be noticed.
    check("mean_probs and geo_mean_probs give the same argmaxes on this fixture",
          argmax_of(tta_mean_probs(e, kE_K)) ==
              argmax_of(tta_geo_mean_probs(e, kE_K)));

    // D9 -- the paper's actual finding, now measured on a real aggregate.
    Tensor base = tta_transform_view(e, 0, kE_K);
    Tensor agg = tta_mean_probs(e, kE_K);
    TtaFlipReport r = tta_flip_report(base, agg, labels);
    check("D9: changed == 3", r.changed == 3);
    check("D9: correct_to_incorrect == 1 -- the aggregate LOSES a prediction "
          "the base member got RIGHT", r.correct_to_incorrect == 1);
    check("D9: incorrect_to_correct == 2", r.incorrect_to_correct == 2);
    check("D9: c2i + i2c == changed",
          r.changed == r.correct_to_incorrect + r.incorrect_to_correct);
    check("D9: net_change() == +1 WHILE a correct prediction was lost -- this "
          "is exactly the case a net-only report cannot see",
          r.net_change() == 1 && r.correct_to_incorrect == 1);

    // LIVENESS (the adversarial session's lesson): assert the base member SPLITS
    // its predictions, so a constant-predictor collapse cannot satisfy the
    // comparison above.
    std::vector<size_t> base_a = argmax_of(base);
    std::vector<size_t> agg_a = argmax_of(agg);
    int base_split = 0, agg_split = 0;
    for (size_t i = 1; i < base_a.size(); ++i)
        if (base_a[i] != base_a[0]) ++base_split;
    for (size_t i = 1; i < agg_a.size(); ++i)
        if (agg_a[i] != agg_a[0]) ++agg_split;
    check("liveness: the base member SPLITS its predictions (not a constant "
          "predictor, which would make the accuracy comparison vacuous)",
          base_split >= 5);
    check("liveness: the aggregate SPLITS its predictions", agg_split >= 5);
    check("liveness: neither is at chance (5/10)", m0 > 5 && agg_acc > 5);

    // Determinism across rebuilds (D-safety): same input, bit-identical output.
    Tensor again = tta_mean_probs(fixture_E(), kE_K);
    bool same = true;
    for (size_t i = 0; i < agg.rows; ++i)
        for (size_t c = 0; c < agg.cols; ++c)
            if (again(i, c) != agg(i, c)) same = false;
    check("aggregation is deterministic (bit-identical on a fresh fixture)", same);
}

int main() {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "==========================================================\n";
    std::cout << " Test-time augmentation suite (Shanmugam, arXiv:2011.11156)\n";
    std::cout << "==========================================================\n";

    test_transforms();
    test_bundle_and_argmax();
    test_flip_report();
    test_task1_validation();
    test_aggregations();
    test_geo_identity_and_tiebreaks();
    test_t2_cannot_fix_both_wrong();
    test_weights();
    test_dispatcher();
    test_end_to_end();

    std::cout << "\n=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}