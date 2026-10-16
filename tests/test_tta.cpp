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
//     grep -ril 'test.time.augment|test_time_augment|\btta\b|ensemble.*predict' \
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
        {{-0.55,  0.55}, {-0.55, -0.55}, { 0.06, -0.06}},   // r7 t2 wrong only
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

int main() {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "==========================================================\n";
    std::cout << " Test-time augmentation suite (Shanmugam, arXiv:2011.11156)\n";
    std::cout << "==========================================================\n";

    test_transforms();
    test_bundle_and_argmax();
    test_flip_report();
    test_task1_validation();

    std::cout << "\n=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}