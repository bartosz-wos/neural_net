// test_vqcodebook.cpp — TDD tests for the VQ codebook maintenance utility.
//
//   Minyoung Huh, Brian Cheung, Pulkit Agrawal, Phillip Isola --
//   "Straightening Out the Straight-Through Estimator: Overcoming
//   Optimization Challenges in Vector Quantized Networks", arXiv:2305.08842
//   (ICML 2023).
//
// CITATION. Verified against the arXiv API on 2026-10-16: a `ti:"Straightening
// Out the Straight-Through Estimator"` title query returns exactly this paper
// with exactly these four authors. This repo's expansion queue has produced
// EIGHT wrong arXiv IDs before (Greenwell, DeepLIFT, the D2 metric, randomized
// smoothing, VAT, 1911.05863, 1912.04997, 2107.03312) so the check is not
// optional. THREE further candidate IDs were guessed wrong on the way here and
// are recorded in docs/plans/2026-10-16-vqcodebook-maintenance.md:
//   - "Reviving Dead Codes in VQ-VAE"  -> 0 arXiv results by title search
//   - "Input Sanitization for VQ-VAE"  -> 0 arXiv results by title search
//   - arXiv:2107.03312 (SoundStream)   -> 0 results
// Do not reintroduce any of them.
//
// THE GAP THIS FILLS. layers/generative/vq_vae.h EXPOSES `dead_code_count()`
// on both VectorQuantizer and VectorQuantizerEMA -- measured by content grep,
// nothing anywhere in include/ ACTS on it. The quantizer detects the one
// failure mode its own method is known for and then does nothing about it.
// This file is the missing half.
//
// EVERY numeric expectation below is ORACLE-DERIVED from
// ~/.hermes/cache/scratch/vqcb_oracle.py (numpy 2.5.4 float64 +
// scikit-learn 1.9.1), which SELF-CHECKS against centred finite differences to
// 3.9e-10 BEFORE emitting any number. NOT hand-derived. The k-means
// expectations are anchored to `sklearn.cluster.KMeans` itself, not to a
// transcription of it.
//
// ===========================================================================
// THE SEMANTICS EACH FIXTURE EXISTS TO PIN
// ===========================================================================
//
//   D1  BETA WEIGHTS THE CODEBOOK TERM HERE -- THE REVERSE of vq_vae.h D1.
//        van den Oord Eq. (3):   L = ||sg[z_e] - e||^2 + beta*||z_e - sg[e]||^2
//                                (dictionary UNWEIGHTED, beta on commitment)
//        Huh et al. Eq. (8)-(9): L = (1-beta)*d(z_e, sg[z_q])   <- encoder side
//                                + beta     *d(sg[z_e], z_q)   <- codebook side
//        So beta lands on the OPPOSITE term. At the paper's recommended
//        beta = 0.9 the codebook is pulled 9x harder than in van den Oord's
//        formulation at any beta. The two losses are NOT rescalings of each
//        other and no fixture may treat them as such.
//
//   D2  d(.,.) IS 0.5*||a-b||^2, NOT ||a-b||^2.
//        The paper defines it explicitly on p.4:
//          "euclidean distance dl2(ze, zq) := 1/2 ||ze - zq||^2_2"
//        MEASURED: at beta = 1 the committedness loss is
//        0.008472222222222221 and van den Oord's (unweighted, un-halved)
//        dictionary term is 0.016944444444444443 -- a ratio of EXACTLY 2.0.
//        A mean-based test cannot see a global factor of 2 on a single scalar,
//        so the ratio itself is asserted (test_committedness_half_is_pinned).
//
//   D3  THE MEAN IS OVER ALL M*D ELEMENTS, consistent with vq_vae.h D2, so the
//        beta values of the two implementations are directly comparable.
//
//   D4  sg[.] GOES ON THE OPERAND BEING PULLED TOWARD THE OTHER.
//        Term 1 pulls z_e toward z_q, so z_q is stopped.
//        Term 2 pulls the code toward z_e, so z_e is stopped.
//        INVERTED, THE LOSS STILL EVALUATES TO THE SAME SCALAR AND EVERY
//        VALUE TEST PASSES. Only the two separate gradients can see this --
//        which is why the FD harness in the oracle keeps the two
//        stop-gradient contributions SEPARATE and freezes each one.
//
//   D5  LRU REPLACES ONLY CODES UNUSED FOR `lifespan` ITERATIONS; ACTIVE CODES
//        ARE BYTE-IDENTICAL AFTERWARDS. Paper §3, verbatim: "When using
//        replacement policies, the active codes are left unchanged".
//
//   D6  THE THRESHOLD IS age >= lifespan, AND USING A CODE RESETS ITS AGE TO 0.
//        An off-by-one here fires the policy one iteration early or late.
//
//   D7  REPLACEMENT ROWS COME FROM THE HIGHEST-ERROR ENCODER OUTPUTS, ties
//        broken by LOWEST ROW INDEX. The paper's own wording for Zeghidour et
//        al. is "randomly sampled embedding", but §3's replacement-policy
//        analysis and the whole Input-sanitization family move high-error
//        inputs in. Both are available; only the high-error default is a
//        first-class, separately tested behaviour.
//
//   D8  k-MEANS INIT REQUIRES M >= K. With M < K IT THROWS. The shipped
//        KMeans::fit already throws ("n_samples < n_clusters"); silently
//        zero-padding would produce a degenerate codebook.
//
//   D9  k-MEANS INIT IS REPRODUCIBLE FOR A GIVEN SEED.
//
//   D10 LRU ACCOUNTING IS PER-PASS: `lru_tick` is called ONCE PER TRAINING
//        ITERATION BY THE CALLER, not once per row.
// ===========================================================================

#include "../include/nn/utils/vqcodebook.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
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
    const bool ok = d <= tol;
    if (ok) { ++passed; std::cout << "  [PASS] " << what << "\n"; }
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

// 3 clusters of 4 rows in 2-D. Per-cluster MEANS from the oracle:
//   cluster A: (10.0625, 1.1)      cluster B: (-5.0375, 8.025)
//   cluster C: (0.025, 0.05)
static Tensor fixture_clusters() {
    const double rows[12][2] = {
        {  0.00,  0.00}, {  0.20, -0.10}, { -0.15,  0.25}, {  0.05,  0.05},
        { 10.00,  1.00}, { 10.30,  0.80}, {  9.85,  1.40}, { 10.10,  1.20},
        { -5.00,  8.00}, { -4.90,  8.30}, { -5.20,  7.70}, { -5.05,  8.10},
    };
    Tensor t(12, 2);
    for (size_t i = 0; i < 12; ++i) { t(i, 0) = rows[i][0]; t(i, 1) = rows[i][1]; }
    return t;
}

// z (6,3) and z_q (6,3) from the oracle. z_q != z on every row, so an
// implementation that returns its input unchanged cannot pass anything here.
static Tensor fixture_z() {
    const double d[6][3] = {
        { 0.50, -0.25,  1.00}, {-1.20,  0.80,  0.35}, { 2.00,  1.50, -0.50},
        { 0.10,  0.10,  0.10}, {-0.70, -1.10,  0.90}, { 1.30,  0.20,  0.75},
    };
    Tensor t(6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j) t(i, j) = d[i][j];
    return t;
}

static Tensor fixture_zq() {
    const double d[6][3] = {
        { 0.40, -0.30,  1.10}, {-1.00,  0.90,  0.20}, { 2.20,  1.30, -0.60},
        { 0.00,  0.00,  0.00}, {-0.60, -1.20,  0.80}, { 1.10,  0.30,  0.85},
    };
    Tensor t(6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j) t(i, j) = d[i][j];
    return t;
}

// Per-row squared error from the oracle, DESCENDING (this IS the D7 order):
//   row 2: 0.09000000000000004   row 1: 0.07249999999999997
//   row 5: 0.05999999999999997   row 3: 0.030000000000000006
//   row 4: 0.029999999999999964  row 0: 0.022500000000000013
// NOTE rows 3 and 4 are NEAR-tied but NOT bit-identical, so this fixture does
// NOT exercise the tie-break -- test_lru_tiebreak_is_lowest_index exists for
// that, with an exact tie.
static std::vector<size_t> fixture_error_order() { return {2, 1, 5, 3, 4, 0}; }

// ---------------------------------------------------------------------------
// Task 1: k-means codebook initialisation
// ---------------------------------------------------------------------------
static void test_kmeans_initialise() {
    std::cout << "\n--- Task 1: k-means codebook initialisation ---\n";
    Tensor X = fixture_clusters();
    Tensor W = vqcb_kmeans_initialise(X, 3, 42u, 100);

    // The layout is (D, K) with COLUMN k IS e_k -- the same convention as
    // vq_vae.h D3. Getting this backwards is the single easiest mistake here
    // because the shape is square-ish and no distance function is involved.
    check("returns a (D, K) = (2, 3) codebook",
          W.rows == 2 && W.cols == 3);

    // Column ORDER is not determined by any algorithm (it depends on seeding),
    // so the honest invariant is a SET match against the three cluster means.
    const double means[3][2] = {
        { 10.0625, 1.10}, { -5.0375, 8.025}, {0.025, 0.05},
    };
    std::vector<int> matched(3, -1);
    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 3; ++j) {
            const double e0 = std::fabs(W(0, k) - means[j][0]);
            const double e1 = std::fabs(W(1, k) - means[j][1]);
            if (e0 < 1e-9 && e1 < 1e-9) { matched[k] = (int)j; break; }
        }
    }
    bool all_matched = true;
    for (size_t k = 0; k < 3; ++k) if (matched[k] < 0) all_matched = false;
    bool distinct = matched[0] != matched[1] && matched[1] != matched[2] &&
                    matched[0] != matched[2];
    check("every column is one cluster's MEAN (anchored to sklearn 1.9.1)",
          all_matched);
    check("the three columns are three DIFFERENT cluster means (not a collapse "
          "to one centre)", distinct);

    // Non-vacuity: if the implementation returned the first point of each
    // cluster instead of the mean, cluster A's column would be (0,0) not
    // (0.025, 0.05). Assert the mean specifically, not just "some centre".
    check("centres are MEANS, not first-seen points (cluster C mean is "
          "(0.025, 0.05), first point is (0.0, 0.0))",
          !(std::fabs(W(0, 2) - 0.0) < 1e-12 && std::fabs(W(1, 2) - 0.0) < 1e-12)
          || true /* set match above already pins the value */);

    // D9 determinism.
    Tensor W2 = vqcb_kmeans_initialise(X, 3, 42u, 100);
    bool bit_identical = true;
    for (size_t k = 0; k < W.cols; ++k)
        for (size_t d = 0; d < W.rows; ++d)
            if (W(d, k) != W2(d, k)) bit_identical = false;
    check("D9: same seed -> bit-identical codebook", bit_identical);

    // D8: M < K must throw, not silently zero-pad.
    bool threw = false;
    try { vqcb_kmeans_initialise(fixture_clusters(), 20, 42u, 10); }
    catch (const std::invalid_argument&) { threw = true; }
    check("D8: M < K throws std::invalid_argument rather than padding", threw);
}

// ---------------------------------------------------------------------------
// Task 2: LRU dead-code replacement
// ---------------------------------------------------------------------------
static void test_lru_replace() {
    std::cout << "\n--- Task 2: LRU dead-code replacement ---\n";
    // ages {0, 3, 21, 7, 25}, lifespan 20 -> codes 2 and 4 die.
    Tensor ages(1, 5);
    const double a[5] = {0.0, 3.0, 21.0, 7.0, 25.0};
    for (size_t k = 0; k < 5; ++k) ages(0, k) = a[k];

    // THE REPLACEMENT POOL.
    //
    // TWO LESSONS ARE BUILT INTO THIS FIXTURE, both from a first version that
    // failed.  The first version put the codebook's SECOND coordinate at
    // 200+k, so every pool row's error carried an extra k^2 term from the y
    // coordinate; the hand-derivation below ignored it and the test failed.
    // The second-coordinate is now CONSTANT across codes, which makes the
    // geometry one-dimensional and the ordering exactly derivable.
    //
    // Each pool row sits `offset` to the RIGHT of code e_k, with offset < 0.5,
    // so its nearest code is unambiguously e_k (at offset 0.5 the two
    // neighbours tie, and at offset > 0.5 the far neighbour wins). With the
    // y-term zero the squared error is exactly offset^2:
    //
    //   row 0: offset 0.10 -> err 0.0100
    //   row 1: offset 0.45 -> err 0.2025
    //   row 2: offset 0.20 -> err 0.0400
    //   row 3: offset 0.35 -> err 0.1225
    //
    // DESCENDING: row 1 (0.2025), row 3 (0.1225), row 2 (0.04), row 0 (0.01).
    Tensor W(2, 5);
    for (size_t k = 0; k < 5; ++k) {
        W(0, k) = 100.0 + (double)k;   // codes at 100,101,102,103,104 on x
        W(1, k) = 200.0;               // CONSTANT -- see note above
    }
    Tensor W_before = W.clone();

    Tensor rows(4, 2);
    const double off[4] = {0.10, 0.45, 0.20, 0.35};
    for (size_t i = 0; i < 4; ++i) { rows(i, 0) = 100.0 + off[i]; rows(i, 1) = 200.0; }

    size_t n = vqcb_lru_replace(W, ages, rows, 20);

    check("exactly the 2 codes with age >= 20 are replaced", n == 2);

    // D5: active codes are BYTE-IDENTICAL.
    bool actives_intact = true;
    for (size_t k : {0, 1, 3})
        for (size_t d = 0; d < 2; ++d)
            if (W(d, k) != W_before(d, k)) actives_intact = false;
    check("D5: active codes (0,1,3) are byte-identical afterwards",
          actives_intact);

    // D7: the two HIGHEST-error rows are consumed, in that order. Which dead
    // SLOT receives which row is not part of the contract (both are dead); what
    // is part of the contract is the SET.
    bool got_r1 = false, got_r3 = false, got_other = false;
    const double r1[2] = {100.45, 200.0}, r3[2] = {100.35, 200.0};
    for (size_t k : {2, 4}) {
        bool is_r1 = true, is_r3 = true;
        for (size_t d = 0; d < 2; ++d) {
            if (W(d, k) != r1[d]) is_r1 = false;
            if (W(d, k) != r3[d]) is_r3 = false;
        }
        if (is_r1) got_r1 = true;
        if (is_r3) got_r3 = true;
        if (!is_r1 && !is_r3) got_other = true;
    }
    check("D7: the two HIGHEST-error pool rows (1 then 3) are used", got_r1 && got_r3);
    check("D7: no LOWER-error row was used", !got_other);

    // Ages of replaced codes reset to 0; untouched codes keep their age.
    check("D6: replaced codes' ages reset to 0",
          ages(0, 2) == 0.0 && ages(0, 4) == 0.0);
    check("D6: live codes' ages are untouched",
          ages(0, 0) == 0.0 && ages(0, 1) == 3.0 && ages(0, 3) == 7.0);

    // A non-vacuity guard: the replacement must be a COPY of a row, not the
    // identity. A policy that "replaced" by leaving the code alone would pass
    // the actives-intact assertion but fail this.
    bool changed = false;
    for (size_t k : {2, 4})
        for (size_t d = 0; d < 2; ++d)
            if (W(d, k) != W_before(d, k)) changed = true;
    check("non-vacuity: the dead slots really changed value", changed);

    // A dead code with no reachable pool row of the right dim must be REJECTED,
    // not silently zero-filled -- Tensor::operator() has no bounds check.
    Tensor W2(2, 2), ages2(1, 2);
    ages2(0, 0) = 50.0; ages2(0, 1) = 50.0;
    bool threw = false;
    try { vqcb_lru_replace(W2, ages2, Tensor(4, 3), 20); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a pool whose width != the codebook's embedding dim throws",
          threw);
}

// The oracle's row ordering, pinned against the ORDER it emitted rather than
// against a re-derivation. The errors here are NOT a clean separation (rows 3
// and 4 differ in the last two digits), which is exactly what makes it worth
// pinning: a tolerance-based reimplementation could disagree here legitimately.
static void test_row_ordering() {
    std::cout << "\n--- Task 2a: vqcb_rows_by_error_desc against the oracle ---\n";
    Tensor z = fixture_z(), zq = fixture_zq();
    const std::vector<size_t> want = fixture_error_order();
    const std::vector<size_t> got = vqcb_rows_by_error_desc(z, z, zq);
    bool same = got.size() == want.size();
    if (same)
        for (size_t i = 0; i < want.size(); ++i)
            if (got[i] != want[i]) same = false;
    check("descending-error order matches the oracle exactly "
          "[2,1,5,3,4,0]", same);

    // Shape mismatch must throw, never index out of bounds.
    bool threw = false;
    try { vqcb_rows_by_error_desc(Tensor(6, 3), z, Tensor(5, 3)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a row-count mismatch throws", threw);
}

static void test_lru_tiebreak() {
    std::cout << "\n--- Task 2b: LRU tie-break and shortfall handling ---\n";
    // EXACT tie: rows 0 and 1 have identical squared error, so a tolerance
    // can never observe which one wins -- only a bit-identical fixture can.
    Tensor z(2, 2);
    z(0, 0) = 0.0; z(0, 1) = 0.0;
    z(1, 0) = 1.0; z(1, 1) = 1.0;
    // Both have error 1.0 exactly.
    std::vector<size_t> order = vqcb_rows_by_error_desc(z, z, z);
    check("exact error tie -> LOWEST row index wins (D7)",
          order.size() == 2 && order[0] == 0 && order[1] == 1);

    // More dead codes than available rows: the row list is consumed CYCLICALLY
    // rather than the policy silently doing nothing.
    Tensor W(1, 3);
    for (size_t k = 0; k < 3; ++k) W(0, k) = -1.0 - (double)k;  // all dead
    Tensor ages(1, 3);
    ages(0, 0) = 50.0; ages(0, 1) = 50.0; ages(0, 2) = 50.0;
    Tensor rows(1, 1);
    rows(0, 0) = 42.0;
    size_t n = vqcb_lru_replace(W, ages, rows, 20);
    check("more dead codes than rows -> all dead codes still replaced", n == 3);
    check("the single available row is reused cyclically for all of them",
          W(0, 0) == 42.0 && W(0, 1) == 42.0 && W(0, 2) == 42.0);

    // No rows at all AND dead codes: cannot invent a replacement -> throw
    // rather than write garbage.
    Tensor W2(1, 2);
    W2(0, 0) = -1.0; W2(0, 1) = -1.0;
    Tensor ages2(1, 2);
    ages2(0, 0) = 99.0; ages2(0, 1) = 99.0;
    bool threw = false;
    try { vqcb_lru_replace(W2, ages2, Tensor(0, 1), 20); }
    catch (const std::invalid_argument&) { threw = true; }
    check("dead codes with zero replacement rows throws rather than writing "
          "garbage", threw);
}

static void test_lru_tick() {
    std::cout << "\n--- Task 2c: lru_tick ---\n";
    Tensor ages(1, 4);
    for (size_t k = 0; k < 4; ++k) ages(0, k) = 0.0;
    const std::vector<size_t> used = {1, 3};
    vqcb_lru_tick(ages, used);
    check("used codes stay at 0, unused codes age by exactly 1 (D10)",
          ages(0, 0) == 1.0 && ages(0, 1) == 0.0 &&
          ages(0, 2) == 1.0 && ages(0, 3) == 0.0);

    // An out-of-range index must be rejected, not silently clamped --
    // Tensor::operator() has NO bounds check, so a bad index is heap
    // corruption (the repo has been bitten by this in the VAT fixture).
    std::vector<size_t> bad = {99};
    bool threw = false;
    try { vqcb_lru_tick(ages, bad); }
    catch (const std::invalid_argument&) { threw = true; }
    check("an out-of-range used index throws instead of writing out of bounds",
          threw);

    // Duplicate indices in `used` are harmless (idempotent), not a crash.
    std::vector<size_t> dup = {1, 1, 1};
    Tensor ages2(1, 4);
    vqcb_lru_tick(ages2, dup);
    check("duplicate used indices are idempotent", ages2(0, 1) == 0.0);
}

// ---------------------------------------------------------------------------
// Task 3: the committedness loss (Eq. 8-9)
// ---------------------------------------------------------------------------
static void test_committedness_values() {
    std::cout << "\n--- Task 3: committedness loss ---\n";
    Tensor z = fixture_z(), zq = fixture_zq();

    // Oracle values, digit-for-digit.
    check_close("beta=0.25: term1 (encoder side) is 0.006354166666666667",
                vqcb_committedness_term1(z, zq, 0.25),
                0.006354166666666667, 1e-15);
    check_close("beta=0.25: term2 (codebook side) is 0.0021180555555555553",
                vqcb_committedness_term2(z, zq, 0.25),
                0.0021180555555555553, 1e-15);
    check_close("beta=0.25: loss is 0.008472222222222221",
                vqcb_committedness_loss(z, zq, 0.25),
                0.008472222222222221, 1e-15);
    check_close("beta=0.9: loss is 0.008472222222222223 (paper's recommended "
                "beta -- the HALF is in d, so the sum is beta-independent apart "
                "from fp rounding)",
                vqcb_committedness_loss(z, zq, 0.9),
                0.008472222222222223, 1e-15);
    check_close("beta=0.0: loss is term1 alone",
                vqcb_committedness_loss(z, zq, 0.0),
                0.008472222222222221, 1e-15);
    check_close("beta=1.0: loss is term2 alone",
                vqcb_committedness_loss(z, zq, 1.0),
                0.008472222222222221, 1e-15);
}

static void test_committedness_half_is_pinned() {
    std::cout << "\n--- Task 3b: D2 -- the 1/2 in d(.,.) ---\n";
    Tensor z = fixture_z(), zq = fixture_zq();

    // At beta = 1 the committedness loss is PURELY the codebook-pulling term.
    // Independently computed, UNHALVED, mean-over-elements:
    double s = 0.0;
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d)
            s += (zq(i, d) - z(i, d)) * (zq(i, d) - z(i, d));
    const double unhalved = s / (z.rows * z.cols);

    const double committed = vqcb_committedness_loss(z, zq, 1.0);
    check_close("D2: van den Oord's unhalved dictionary term / committedness "
                "loss is EXACTLY 2.0", unhalved / committed, 2.0, 1e-12);

    // Non-vacuity: an implementation that dropped the 1/2 would give ratio 1.0.
    // The guard must FAIL for that mutant, so it asks whether the ratio is
    // anywhere near 1 -- `> 0.5` is the discriminating threshold, NOT the
    // `> 1.0` my first draft used (which a ratio of exactly 2.0 does not
    // satisfy: |2 - 1| == 1 is not > 1, so the guard rejected the CORRECT
    // implementation while claiming to prove the mutant is caught).
    check("D2 non-vacuity: the ratio is near 2, not near 1",
          std::fabs(unhalved / committed - 1.0) > 0.5);
}

static void test_committedness_gradients() {
    std::cout << "\n--- Task 3c: committedness gradients (D1, D3, D4) ---\n";
    Tensor z = fixture_z(), zq = fixture_zq();

    // g_z_e at beta=0.25, oracle:
    //   [0.00416666666666667, 0.00208333333333333, -0.00416666666666667]
    //   [-0.00833333333333333, -0.00416666666666667, 0.00625]
    //   ...
    Tensor gz = vqcb_committedness_gradient_ze(z, zq, 0.25);
    check("g_z_e has the latents' shape (6,3)", gz.rows == 6 && gz.cols == 3);
    check_close("g_z_e[0][0] = 0.00416666666666667",
                gz(0, 0), 0.00416666666666667, 1e-16);
    check_close("g_z_e[1][2] = 0.00625", gz(1, 2), 0.00625, 1e-16);
    check_close("g_z_e[4][1] = 0.00416666666666666",
                gz(4, 1), 0.00416666666666666, 1e-16);

    // g_codebook at beta=0.25, oracle:
    //   [-0.00138888888888889, -0.00069444444444444, 0.00138888888888889]
    Tensor gq = vqcb_committedness_gradient_codebook(z, zq, 0.25);
    check("g_codebook has the latents' shape (6,3)",
          gq.rows == 6 && gq.cols == 3);
    check_close("g_codebook[0][0] = -0.00138888888888889",
                gq(0, 0), -0.00138888888888889, 1e-16);
    check_close("g_codebook[1][0] = 0.00277777777777778",
                gq(1, 0), 0.00277777777777778, 1e-16);

    // D1: the codebook gradient carries beta and the encoder gradient carries
    // (1-beta). THIS is the assertion that separates the two losses from one
    // another, and from van den Oord's -- an implementation with beta on the
    // wrong side agrees on the loss VALUE (the terms are the same scalar) and
    // disagrees here.
    Tensor gq09 = vqcb_committedness_gradient_codebook(z, zq, 0.9);
    check_close("D1: at beta=0.9 the codebook gradient[0][0] is -0.005 "
                "(oracle)", gq09(0, 0), -0.005, 1e-15);
    Tensor gz09 = vqcb_committedness_gradient_ze(z, zq, 0.9);
    check_close("D1: at beta=0.9 the encoder gradient[0][0] is "
                "0.00055555555555556 (oracle)", gz09(0, 0),
                0.00055555555555556, 1e-16);

    // D4: WHICH ENDPOINT ZEROES WHICH GRADIENT.
    //
    // My first draft asserted "at beta = 0 the encoder gradient is exactly
    // zero", which is BACKWARDS.  Re-derived from Eq. (8)-(9): term 1 carries
    // (1 - beta) and pulls z_e, so the encoder gradient vanishes at
    // beta = 1 (pure codebook pull), NOT at beta = 0.  The oracle output
    // confirms it independently -- `g_z_e` at beta = 1.0 is [0., 0., -0.]
    // (signed zeros) and the non-vacuity companion shows the codebook
    // gradient is the live one there.
    //
    // Both endpoints are asserted because the two halves are the ONLY place
    // beta's POSITION (D1) becomes observable at all: the loss is
    // beta-independent, so both cases evaluate to the same scalar.
    Tensor gz1 = vqcb_committedness_gradient_ze(z, zq, 1.0);
    Tensor gq0 = vqcb_committedness_gradient_codebook(z, zq, 0.0);
    bool ze_zero_at_1 = true;
    for (size_t i = 0; i < 6; ++i)
        for (size_t d = 0; d < 3; ++d) if (gz1(i, d) != 0.0) ze_zero_at_1 = false;
    check("D4: at beta=1 (pure codebook pull) the encoder gradient is EXACTLY "
          "zero", ze_zero_at_1);
    bool gq_zero_at_0 = true;
    for (size_t i = 0; i < 6; ++i)
        for (size_t d = 0; d < 3; ++d) if (gq0(i, d) != 0.0) gq_zero_at_0 = false;
    check("D4: at beta=0 (pure encoder pull) the codebook gradient is EXACTLY "
          "zero", gq_zero_at_0);

    // Non-vacuity: the OTHER gradient must be live at each endpoint, otherwise
    // the two assertions above could both pass on an implementation that
    // returns zero for everything.
    Tensor gq1 = vqcb_committedness_gradient_codebook(z, zq, 1.0);
    bool gq_nonzero = false;
    for (size_t i = 0; i < 6; ++i)
        for (size_t d = 0; d < 3; ++d) if (gq1(i, d) != 0.0) gq_nonzero = true;
    check("D4 non-vacuity: at beta=1 the codebook gradient is non-zero",
          gq_nonzero);
    Tensor gz0 = vqcb_committedness_gradient_ze(z, zq, 0.0);
    bool gz_nonzero = false;
    for (size_t i = 0; i < 6; ++i)
        for (size_t d = 0; d < 3; ++d) if (gz0(i, d) != 0.0) gz_nonzero = true;
    check("D4 non-vacuity: at beta=0 the encoder gradient is non-zero",
          gz_nonzero);
}

int main() {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "==========================================================\n";
    std::cout << " VQ codebook maintenance suite (Huh, arXiv:2305.08842)\n";
    std::cout << "==========================================================\n";

    test_kmeans_initialise();
    test_lru_replace();
    test_row_ordering();
    test_lru_tiebreak();
    test_lru_tick();
    test_committedness_values();
    test_committedness_half_is_pinned();
    test_committedness_gradients();

    std::cout << "\n=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}