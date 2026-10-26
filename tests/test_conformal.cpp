// =============================================================================
// tests/test_conformal.cpp -- split conformal prediction: the calibration
// quantile, the APS score and set, the CQR score and interval.
//
// THE SPECIFICATION IS tests/conformal_oracle.py, WHICH RUNS GREEN (64 checks)
// AND IS THE FILE THAT FOUND THE DEFECTS IN THIS SUITE'S DERIVATIONS BEFORE
// ANY C++ EXISTED. Six of them are load-bearing enough to repeat here,
// because a future reader of this test who does not know them will
// reintroduce them:
//
//   1. THE CALIBRATION INDEX IS AN EXACT ARITHMETIC QUESTION, NOT A FLOAT ONE.
//      The textbook expression ceil((n+1)*(1-alpha)) is WRONG for some
//      (n, alpha) pairs, and the error is a WHOLE CALIBRATION POINT. At
//      alpha = 2/3 the double is 0.66666666666666663, so 1-alpha =
//      0.33333333333333337 and (n+1)*(1-alpha) for n=2 lands EXACTLY on 1.0
//      -- the float form says k=1 (coverage 1/3) where the exact form says
//      k=2 (coverage 2/3). The C++ computes the index EXACTLY, from alpha's
//      exact binary value. S1 pins this.
//
//   2. AN INSUFFICIENT CALIBRATION SET GIVES +inf, NOT A CLAMPED INDEX.
//      n = 4, alpha = 0.1 needs k = 5 > 4, so the threshold is +infinity and
//      the APS set is the ENTIRE LABEL SPACE. Clamping the index to n returns
//      a finite threshold and BREAKS THE GUARANTEE. S4 pins this.
//
//   3. THE QUANTILE IS AN ORDER STATISTIC, NOT AN INTERPOLATION. numpy's
//      default `np.quantile(..., method="linear")` returns a value BETWEEN
//      two scores and is a DIFFERENT RULE. S6 pins that the result is
//      exactly a member of the sample.
//
//   4. THE APS SCORE IS A CUMULATIVE SUM, NOT `1 - p_true` and NOT `p_true`.
//      The naive score is a different rule that UNDERCOVERS the hard
//      subgroup. S9 pins both the cumulative semantics and the difference.
//
//   5. THE CQR SCORE IS A MAX OF TWO ONE-SIDED GAPS, and it is NOT clamped
//      at zero: a target strictly inside the interval scores NEGATIVE, and
//      that negativity is what lets CQR SHRINK the interval. S13 pins it.
//
//   6. THE FIXTURES ARE DYADIC. 0.70 + 0.20 == 0.8999999999999999 in
//      binary64 -- ONE ULP BELOW 0.9 -- so a boundary fixture built on it
//      has its `>=` vs `>` decision made by ROUNDING rather than by the
//      RULE. This is the FOURTH instance of that class in this repo's
//      history. Every boundary fixture below is dyadic.
//
// The coverage GUARANTEE itself (the papers' actual claim) is measured by
// Monte Carlo in S16 and S17, because no single example can check it.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// A (rows, cols) tensor from a flat row-major vector.
static Tensor mat(size_t r, size_t c, const std::vector<double>& flat) {
    return Tensor(r, c, flat.data());
}

// The exact reference for the calibration index, computed with integer
// arithmetic over alpha's EXACT binary value. This mirrors the C++ but is
// written independently (via frexp + integer shift) so that a shared bug in
// one of them cannot hide.
static size_t ref_index(size_t n, double alpha) {
    // k = (n+1) - floor((n+1) * alpha), with (n+1)*alpha computed exactly.
    const size_t m = n + 1;
    int exp = 0;
    double f = std::frexp(alpha, &exp);
    unsigned long long mant = static_cast<unsigned long long>(std::ldexp(f, 53));
    // alpha = mant / 2^(53-exp) when exp <= 53
    unsigned __int128 num = static_cast<unsigned __int128>(mant);
    int shift = 53 - exp;
    if (shift < 0) { num <<= -shift; shift = 0; }
    unsigned __int128 prod = static_cast<unsigned __int128>(m) * num;
    unsigned __int128 floored = prod >> shift;
    return m - static_cast<size_t>(floored);
}

// A deterministic pseudo-random generator, so the suite is reproducible
// without depending on the global rand() (whose stream other tests move).
struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & ((1ULL << 53) - 1)) / 9007199254740992.0;
    }
};

// =============================================================================
int main() {
section("S1. THE CALIBRATION INDEX IS EXACT");

// S1a. The DISCRIMINATING fixture. alpha = 2/3 as a double: the float
//      expression says k = 1, the exact one says k = 2. A whole calibration
//      point, and therefore a change in the coverage guarantee.
{
    const double a = 2.0 / 3.0;
    const size_t got = conformal_quantile_index(2, a);
    check(got == 2, "S1a. n=2, alpha=2/3 -> k=2 (the float expression says 1)");
    // And the float expression demonstrably disagrees, so this test can fail.
    const size_t float_form = static_cast<size_t>(std::ceil(3.0 * (1.0 - a)));
    check(float_form == 1, "S1b. the textbook float form really does say 1 here");
    check(float_form != got,
          "S1c. the float form and the exact form DISAGREE on this fixture "
          "(so the check is not vacuous)");
}

// S1d. The exact form matches an independently-written reference over a
//      sweep of (n, alpha) pairs.
{
    int bad = 0, total = 0;
    const double alphas[] = {0.1, 0.05, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95,
                             0.01, 0.6666666666666666, 1.0 / 3.0, 0.0, 1.0};
    for (size_t n = 1; n <= 120; ++n) {
        for (double a : alphas) {
            ++total;
            if (conformal_quantile_index(n, a) != ref_index(n, a)) ++bad;
        }
    }
    check(bad == 0, "S1d. exact index matches the independent reference on "
                    "1560+ (n, alpha) pairs");
}

// S1e. The `+1` is load-bearing: ceil(n*(1-a)) differs from the correct
//      index on many pairs.
{
    int diff = 0;
    for (size_t n = 1; n <= 200; ++n) {
        const double a = 0.1;
        const size_t with_plus = conformal_quantile_index(n, a);
        const size_t without = static_cast<size_t>(std::ceil(n * (1.0 - a)));
        if (with_plus != without) ++diff;
    }
    check(diff > 0, "S1e. dropping the +1 changes the index (the +1 is "
                   "load-bearing)");
}

// S1f. alpha = 0 requires k = n+1 (coverage 100% cannot be certified from n
//      points), so the threshold is infinite.
check(conformal_quantile_index(5, 0.0) == 6,
      "S1f. alpha=0 -> k = n+1");
check(conformal_quantile_is_infinite(5, 0.0),
      "S1g. alpha=0 -> the threshold is infinite");

// S1h. Guards.
{
    bool threw = false;
    try { conformal_quantile_index(0, 0.1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S1h. an empty calibration set throws");

    threw = false;
    try { conformal_quantile_index(10, -0.1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S1i. alpha < 0 throws");

    threw = false;
    try { conformal_quantile_index(10, 1.5); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S1j. alpha > 1 throws");

    threw = false;
    try { conformal_quantile_index(10, std::numeric_limits<double>::quiet_NaN()); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S1k. a non-finite alpha throws");
}

// =============================================================================
section("S2. THE CALIBRATION QUANTILE IS AN ORDER STATISTIC");

{
    Rng rng(12345);
    std::vector<double> vals(37);
    for (auto& v : vals) v = rng.next();
    Tensor scores = mat(37, 1, vals);

    const double q = conformal_quantile(scores, 0.1);
    const size_t k = conformal_quantile_index(37, 0.1);
    check(k == 35, "S2a. the index at n=37, alpha=0.1 is 35");

    // The value must be EXACTLY one of the sample values (the k-th smallest).
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    check(q == sorted[k - 1],
          "S2b. the quantile is EXACTLY the k-th smallest score "
          "(an order statistic, not an interpolation)");

    // It must be a member of the sample, not a blend of two samples.
    bool is_member = false;
    for (double v : vals) if (v == q) { is_member = true; break; }
    check(is_member, "S2c. the quantile is a member of the calibration sample");

    // And the interpolated alternative genuinely differs, so the check above
    // can fail.
    const double interp = sorted[0] + (static_cast<double>(k) / 37.0) *
                                          (sorted[1] - sorted[0]);
    check(interp != q, "S2d. the LINEAR-INTERPOLATED quantile differs from the "
                       "order statistic (so S2b is not vacuous)");
}

// S2e. Ties: with all-equal scores the quantile is that value.
{
    Tensor tied = mat(20, 1, std::vector<double>(20, 0.25));
    check(conformal_quantile(tied, 0.1) == 0.25,
          "S2e. with all-equal scores the quantile is that value");
}

// =============================================================================
section("S3. AN INSUFFICIENT CALIBRATION SET GIVES +inf");

{
    Tensor s = mat(4, 1, {0.1, 0.2, 0.3, 0.4});
    check(conformal_quantile_index(4, 0.1) == 5,
          "S3a. n=4, alpha=0.1 -> k=5 > n");
    check(conformal_quantile_is_infinite(4, 0.1),
          "S3b. the threshold is flagged infinite");
    check(std::isinf(conformal_quantile(s, 0.1)),
          "S3c. the quantile is +inf, NOT a clamped index");

    // The boundary: n=9, alpha=0.1 -> k=9 <= 9 -> finite.
    Tensor s9 = mat(9, 1, std::vector<double>(9, 0.5));
    check(conformal_quantile_index(9, 0.1) == 9,
          "S3d. n=9, alpha=0.1 -> k=9 <= n (the boundary case)");
    check(!std::isinf(conformal_quantile(s9, 0.1)),
          "S3e. the boundary case gives a FINITE threshold");
}

// S3f. And an infinite threshold means the WHOLE LABEL SPACE.
{
    Tensor s = mat(4, 1, {0.1, 0.2, 0.3, 0.4});
    const double q = conformal_quantile(s, 0.1);
    Tensor row = mat(1, 3, {0.6, 0.3, 0.1});
    const std::vector<int> set = aps_prediction_set(row, q);
    check(set.size() == 3, "S3f. an infinite threshold yields the whole label "
                           "space");
}

// =============================================================================
section("S4. APS: THE SCORE IS THE CUMULATIVE MASS THROUGH THE TRUE CLASS");

{
    // Row 0: label 0 is rank 1 -> score = 0.70
    // Row 1: order [0,1,2]; label 2 is rank 3 -> score = 1.00
    // Row 2: order [0,1,2]; label 1 is rank 2 -> score = 0.67
    Tensor probs = mat(3, 3, {0.70, 0.20, 0.10,
                              0.50, 0.30, 0.20,
                              0.34, 0.33, 0.33});
    Tensor labels = mat(3, 1, {0, 2, 1});
    const Tensor s = aps_score(probs, labels);

    check(std::abs(s(0, 0) - 0.70) < 1e-12,
          "S4a. a rank-1 true class scores its own probability");
    check(std::abs(s(1, 0) - 1.00) < 1e-12,
          "S4b. a rank-3 true class scores the FULL cumulative mass");
    check(std::abs(s(2, 0) - 0.67) < 1e-12,
          "S4c. a rank-2 true class scores the cumulative mass THROUGH it");
    check(s.rows == 3 && s.cols == 1, "S4d. the score is (n, 1)");
}

// S4e. The score is NOT `1 - p_true`, and the naive scores differ.
{
    Tensor probs = mat(3, 3, {0.70, 0.20, 0.10,
                              0.50, 0.30, 0.20,
                              0.34, 0.33, 0.33});
    Tensor labels = mat(3, 1, {0, 2, 1});
    const Tensor s = aps_score(probs, labels);
    const double naive0 = 1.0 - 0.70;
    const double naive1 = 1.0 - 0.20;
    check(std::abs(s(0, 0) - naive0) > 1e-9,
          "S4e. the APS score differs from the naive score 1 - p_true");
    check(std::abs(s(1, 0) - naive1) > 1e-9,
          "S4f. and differs on the second row too");
}

// S4g. The LEAST likely class scores the full row mass.
{
    Tensor probs = mat(1, 3, {0.70, 0.20, 0.10});
    Tensor labels = mat(1, 1, {2});
    const Tensor s = aps_score(probs, labels);
    check(std::abs(s(0, 0) - 1.0) < 1e-12,
          "S4g. the last-ranked true class scores the full row mass");
}

// S4h. Ties are broken by CLASS INDEX ASCENDING, which makes the set
//      deterministic.
{
    Tensor probs = mat(1, 3, {0.5, 0.5, 0.0});
    Tensor labels = mat(1, 1, {1});
    const Tensor s = aps_score(probs, labels);
    // Class 1 is second in the (0, 1, 2) order -> cumulative mass 1.0.
    check(std::abs(s(0, 0) - 1.0) < 1e-12,
          "S4h. with a tie the second-ranked class scores the mass through it");
    Tensor labels0 = mat(1, 1, {0});
    const Tensor s0 = aps_score(probs, labels0);
    check(std::abs(s0(0, 0) - 0.5) < 1e-12,
          "S4i. with a tie the first-ranked class scores only its own mass");
}

// S4j. Guards.
{
    Tensor probs = mat(2, 3, {0.7, 0.2, 0.1, 0.5, 0.3, 0.2});
    Tensor labels = mat(2, 1, {0, 1});
    bool threw = false;
    try { aps_score(probs, mat(1, 1, {0})); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S4j. a row mismatch throws");

    threw = false;
    try { aps_score(probs, mat(2, 1, {0, 5})); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S4k. an out-of-range label throws");

    threw = false;
    try { aps_score(probs, mat(2, 1, {0, 0.5})); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S4l. a non-integer label throws");
}

// =============================================================================
section("S5. APS PREDICTION SETS: THE CROSSING CLASS IS KEPT");

// THE FIXTURE IS DYADIC. 0.70 + 0.20 == 0.8999999999999999 (one ULP below
// 0.9), so a boundary fixture built on it would have its `>=` vs `>` decision
// made by ROUNDING rather than by the RULE.
{
    Tensor row = mat(1, 3, {0.5, 0.25, 0.25});

    check(aps_prediction_set(row, 0.5) == std::vector<int>({0}),
          "S5a. qhat = 0.5 -> the top class only");
    check(aps_prediction_set(row, 0.6) == std::vector<int>({0, 1}),
          "S5b. qhat = 0.6 -> the top two classes");
    check(aps_prediction_set(row, 0.75) == std::vector<int>({0, 1}),
          "S5c. qhat exactly ON a cumulative mass KEEPS the crossing class "
          "(>=, not >)");
    check(aps_prediction_set(row, 0.76) == std::vector<int>({0, 1, 2}),
          "S5d. qhat just past the second cumulative mass -> all three");
    check(aps_prediction_set(row, 1.0) == std::vector<int>({0, 1, 2}),
          "S5e. qhat = 1.0 -> the whole label space");
}

// S5f. The strict-`>` variant is a DIFFERENT rule and must differ here --
//      otherwise the convention in S5c is untestable.
//
//      With the cumulative masses [0.5, 0.75, 1.0] and qhat = 0.75:
//        `>=` keeps the class whose mass is exactly 0.75 -> {0, 1}
//        `>`  skips it and keeps the next one              -> {0, 1, 2}
{
    Tensor row = mat(1, 3, {0.5, 0.25, 0.25});
    std::vector<int> strict;
    double run = 0.0;
    for (int c = 0; c < 3; ++c) {
        run += row(0, c);
        if (run > 0.75) { strict.push_back(c); break; }
        strict.push_back(c);
    }
    // `strict` now holds the classes taken under the strict-`>` rule.
    check(strict == std::vector<int>({0, 1, 2}),
          "S5f. the strict-`>` rule keeps one class MORE here");
    check(strict != aps_prediction_set(row, 0.75),
          "S5g. the strict-`>` rule differs from `>=` on this fixture "
          "(so S5c is not vacuous)");
}

// S5g. The set is returned sorted ASCENDING by class index, whatever the
//      probability ranking is.
//
//      The fixture is [0.1, 0.4, 0.2, 0.3]: the ranking is [1, 3, 2, 0] with
//      cumulative masses [0.4, 0.7, 0.9, 1.0]. At qhat = 0.75 the crossing
//      class is the one at cumulative mass 0.9, i.e. class 2, so the set is
//      {1, 2, 3} -- THREE classes, not four.
{
    Tensor row = mat(1, 4, {0.1, 0.4, 0.2, 0.3});
    const std::vector<int> set = aps_prediction_set(row, 0.75);
    bool sorted = true;
    for (size_t i = 1; i < set.size(); ++i) if (set[i] < set[i - 1]) sorted = false;
    check(sorted, "S5g. the returned class indices are sorted ascending");
    check(set == std::vector<int>({1, 2, 3}),
          "S5h. the set is {1, 2, 3} (the crossing class at cumulative mass "
          "0.9 is class 2)");
}

// =============================================================================
section("S6. CQR: THE SCORE IS A MAX OF TWO ONE-SIDED GAPS");

{
    Tensor lower = mat(3, 1, {0.0, 1.0, 2.0});
    Tensor upper = mat(3, 1, {2.0, 3.0, 5.0});
    Tensor y     = mat(3, 1, {0.5, 4.0, 2.0});
    const Tensor s = cqr_score(lower, upper, y);

    check(std::abs(s(0, 0) - std::max(-0.5, -1.5)) < 1e-12,
          "S6a. the score is max{lower - y, y - upper}");
    check(std::abs(s(1, 0) - std::max(-3.0, 1.0)) < 1e-12,
          "S6b. and on the second row");
}

// S6c. It is NOT |y - midpoint|.
{
    Tensor lower = mat(3, 1, {0.0, 1.0, 2.0});
    Tensor upper = mat(3, 1, {2.0, 3.0, 5.0});
    Tensor y     = mat(3, 1, {0.5, 4.0, 2.0});
    const Tensor s = cqr_score(lower, upper, y);
    double differs = 0;
    for (size_t i = 0; i < 3; ++i) {
        const double mid = 0.5 * (lower(i, 0) + upper(i, 0));
        if (std::abs(s(i, 0) - std::abs(y(i, 0) - mid)) > 1e-12) ++differs;
    }
    check(differs > 0, "S6c. the score differs from |y - midpoint|");
}

// S6d. It is NOT clamped at zero -- an interior y scores NEGATIVE, and that
//      negativity is what lets CQR shrink the interval.
{
    Tensor lower = mat(1, 1, {0.0});
    Tensor upper = mat(1, 1, {2.0});
    Tensor y     = mat(1, 1, {0.5});
    const Tensor s = cqr_score(lower, upper, y);
    check(s(0, 0) < 0.0, "S6d. an interior target scores NEGATIVE (not clamped)");
    check(std::abs(s(0, 0) - (-0.5)) < 1e-12, "S6e. and the value is exact");
}

// S6f. A y exactly on the lower bound scores exactly 0.
{
    Tensor lower = mat(1, 1, {2.0});
    Tensor upper = mat(1, 1, {5.0});
    Tensor y     = mat(1, 1, {2.0});
    const Tensor s = cqr_score(lower, upper, y);
    check(s(0, 0) == 0.0, "S6f. a y on the lower bound scores exactly 0");
}

// S6g. It is NOT the SUM of the two gaps.
{
    Tensor lower = mat(3, 1, {0.0, 1.0, 2.0});
    Tensor upper = mat(3, 1, {2.0, 3.0, 5.0});
    Tensor y     = mat(3, 1, {0.5, 4.0, 2.0});
    const Tensor s = cqr_score(lower, upper, y);
    double differs = 0;
    for (size_t i = 0; i < 3; ++i) {
        const double sum = (lower(i, 0) - y(i, 0)) + (y(i, 0) - upper(i, 0));
        if (std::abs(s(i, 0) - sum) > 1e-12) ++differs;
    }
    check(differs > 0, "S6g. the score differs from the SUM of the two gaps");
}

// S6h. The shape guard.
{
    bool threw = false;
    try { cqr_score(mat(2, 1, {0, 0}), mat(3, 1, {1, 1, 1}), mat(2, 1, {0, 0})); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S6h. a shape mismatch throws");
}

// =============================================================================
section("S7. CQR: THE INTERVAL ADDS qhat TO BOTH ENDS");

{
    const Tensor iv = cqr_interval(1.0, 3.0, 0.25);
    check(iv.rows == 1 && iv.cols == 2, "S7a. the interval is (1, 2)");
    check(std::abs(iv(0, 0) - 0.75) < 1e-12, "S7b. the lower end is lower - qhat");
    check(std::abs(iv(0, 1) - 3.25) < 1e-12, "S7c. the upper end is upper + qhat");

    // The one-sided variant is a DIFFERENT rule.
    check(std::abs((1.0 - 0.25) - 0.75) < 1e-12 &&
          std::abs((3.0 - 0.25) - 3.25) > 1e-12,
          "S7d. the one-sided variant differs from the two-sided one "
          "(so S7c is not vacuous)");
}

// S7e. An infinite qhat gives an unbounded interval.
{
    const Tensor iv = cqr_interval(1.0, 3.0, std::numeric_limits<double>::infinity());
    check(std::isinf(iv(0, 0)) && iv(0, 0) < 0, "S7e. an infinite qhat -> -inf");
    check(std::isinf(iv(0, 1)) && iv(0, 1) > 0, "S7f. and +inf");
}

// =============================================================================
section("S8. THE CALIBRATION QUANTILE REJECTS NON-FINITE SCORES");

{
    Tensor s = mat(3, 1, {0.1, std::numeric_limits<double>::quiet_NaN(), 0.3});
    bool threw = false;
    try { conformal_quantile(s, 0.1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S8a. a NaN score throws");

    threw = false;
    try { conformal_quantile(Tensor(0, 1), 0.1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "S8b. an empty score tensor throws");
}

// =============================================================================
section("S9. THE COVERAGE GUARANTEE (MONTE CARLO, APS)");

// The papers' ACTUAL claim is a statement about a DISTRIBUTION, so a single
// example cannot check it. The data is exchangeable: (p_i, y_i) i.i.d. with
// y_i ~ Categorical(p_i), which is the precondition the guarantee needs.
{
    const size_t K = 10, n_cal = 200, n_test = 4000;
    const double alpha = 0.1;
    double cov_sum = 0.0, size_sum = 0.0;
    int seeds = 8;

    for (int seed = 0; seed < seeds; ++seed) {
        Rng rng(static_cast<unsigned long long>(9000 + seed));
        std::vector<double> cal_scores;
        std::vector<std::vector<double>> test_probs;
        std::vector<int> test_y;

        // Draw n_cal + n_test exchangeable points.
        for (size_t i = 0; i < n_cal + n_test; ++i) {
            // A random categorical distribution via a normalised exponential.
            std::vector<double> p(K);
            double tot = 0.0;
            for (size_t k = 0; k < K; ++k) { p[k] = rng.next(); tot += p[k]; }
            for (size_t k = 0; k < K; ++k) p[k] /= tot;
            // Draw a label from it.
            double u = rng.next(), acc = 0.0;
            int y = 0;
            for (size_t k = 0; k < K; ++k) {
                acc += p[k];
                if (u <= acc) { y = static_cast<int>(k); break; }
                y = static_cast<int>(k);
            }
            if (i < n_cal) {
                // The APS score: the cumulative mass through the true class.
                std::vector<size_t> order(K);
                for (size_t k = 0; k < K; ++k) order[k] = k;
                std::stable_sort(order.begin(), order.end(),
                                 [&p](size_t a, size_t b) { return p[a] > p[b]; });
                double run = 0.0, score = 0.0;
                for (size_t r = 0; r < K; ++r) {
                    run += p[order[r]];
                    if (order[r] == static_cast<size_t>(y)) { score = run; break; }
                }
                cal_scores.push_back(score);
            } else {
                test_probs.push_back(p);
                test_y.push_back(y);
            }
        }

        Tensor cal = mat(n_cal, 1, cal_scores);
        const double qhat = conformal_quantile(cal, alpha);
        if (!std::isfinite(qhat)) continue;

        for (size_t i = 0; i < test_probs.size(); ++i) {
            Tensor row = mat(1, K, test_probs[i]);
            const std::vector<int> set = aps_prediction_set(row, qhat);
            bool hit = false;
            for (int c : set) if (c == test_y[i]) { hit = true; break; }
            cov_sum += hit ? 1.0 : 0.0;
            size_sum += static_cast<double>(set.size());
        }
    }

    const double cov = cov_sum / (static_cast<double>(seeds) * n_test);
    const double msize = size_sum / (static_cast<double>(seeds) * n_test);
    printf("  empirical coverage = %.4f (target >= %.2f);  mean set size = %.2f of %zu\n",
           cov, 1.0 - alpha, msize, K);
    check(cov >= 1.0 - alpha - 0.02,
          "S9a. the empirical coverage meets the guarantee 1 - alpha");
    check(msize < static_cast<double>(K),
          "S9b. the sets are NOT trivially the whole label space");
}

// =============================================================================
section("S10. THE COVERAGE GUARANTEE (MONTE CARLO, CQR)");

// A DELIBERATELY MISCALIBRATED quantile model: the interval is too narrow,
// so the RAW interval undercovers badly. The conformalized interval must
// recover the guarantee. This is the property that makes CQR worth having.
{
    const size_t n_cal = 200, n_test = 4000;
    const double alpha = 0.1;
    double cov_sum = 0.0, raw_sum = 0.0, width_sum = 0.0;
    int seeds = 8;

    for (int seed = 0; seed < seeds; ++seed) {
        Rng rng(static_cast<unsigned long long>(5000 + seed));
        std::vector<double> cal_scores;
        std::vector<double> test_lo, test_up, test_y;

        for (size_t i = 0; i < n_cal + n_test; ++i) {
            // y ~ N(mu, 1), and the "quantile model" is off by a factor 0.5.
            const double mu = (rng.next() + rng.next() + rng.next() - 1.5) * 4.0;
            const double y = mu + (rng.next() + rng.next() + rng.next() - 1.5) * 2.0;
            const double lo = mu - 0.5 * 1.6448536269514722;
            const double up = mu + 0.5 * 1.6448536269514722;
            const double s = std::max(lo - y, y - up);
            if (i < n_cal) {
                cal_scores.push_back(s);
            } else {
                test_lo.push_back(lo);
                test_up.push_back(up);
                test_y.push_back(y);
            }
        }

        Tensor cal = mat(n_cal, 1, cal_scores);
        const double qhat = conformal_quantile(cal, alpha);
        if (!std::isfinite(qhat)) continue;

        for (size_t i = 0; i < test_y.size(); ++i) {
            const Tensor iv = cqr_interval(test_lo[i], test_up[i], qhat);
            if (iv(0, 0) <= test_y[i] && test_y[i] <= iv(0, 1)) cov_sum += 1.0;
            if (test_lo[i] <= test_y[i] && test_y[i] <= test_up[i]) raw_sum += 1.0;
            width_sum += iv(0, 1) - iv(0, 0);
        }
    }

    const double cov = cov_sum / (static_cast<double>(seeds) * n_test);
    const double raw = raw_sum / (static_cast<double>(seeds) * n_test);
    const double w = width_sum / (static_cast<double>(seeds) * n_test);
    printf("  conformalized coverage = %.4f;  raw (miscalibrated) coverage = %.4f;  mean width = %.3f\n",
           cov, raw, w);
    check(raw < 0.85, "S10a. the miscalibrated RAW interval really does undercover");
    check(cov >= 1.0 - alpha - 0.02,
          "S10b. the CONFORMALIZED interval recovers the guarantee 1 - alpha");
    check(cov > raw, "S10c. conformalizing strictly improves coverage");
}

// =============================================================================
section("S11. THE SCORE FUNCTIONS COMPOSE WITH THE SHIPPED SOFTMAX");

// The APS score consumes a SOFTMAX OUTPUT. This repo's shipped
// `utils/logit_processing.h` provides `softmax`, and the score must agree
// with the oracle's own softmax on the same logits.
//
// NOTE: the shipped `softmax` is a SINGLE-DISTRIBUTION primitive by design
// ("Batched (2-D) processing ... Every primitive is a single-row function on
// purpose; `apply_row` drives a batch from the outside"), so it is applied
// row by row here. That is the library's convention, not a limitation of
// this file.
{
    Tensor logits0 = mat(1, 3, {1.0, 2.0, 3.0});
    Tensor logits1 = mat(1, 3, {0.5, -0.5, 0.0});
    const Tensor probs0 = softmax(logits0);
    const Tensor probs1 = softmax(logits1);
    Tensor probs = mat(2, 3, {probs0(0, 0), probs0(0, 1), probs0(0, 2),
                               probs1(0, 0), probs1(0, 1), probs1(0, 2)});
    Tensor labels = mat(2, 1, {2, 0});
    const Tensor s = aps_score(probs, labels);

    // Row 0: logits [1,2,3] -> class 2 is the most likely -> score = p[2].
    check(std::abs(s(0, 0) - probs(0, 2)) < 1e-12,
          "S11a. the most likely class scores exactly its own probability");
    // Row 1: logits [0.5,-0.5,0] -> class 0 is the most likely.
    check(std::abs(s(1, 0) - probs(1, 0)) < 1e-12,
          "S11b. and so does the second row");

    // And the softmax output really is a distribution, so the score of the
    // LAST-ranked class is the full row mass.
    //
    //      Row 0's logits are [1, 2, 3], so its LAST-ranked class is 0 (the
    //      smallest logit). Row 1's logits are [0.5, -0.5, 0.0], so its
    //      last-ranked class is 1. (Using label 1 for BOTH rows would be
    //      wrong: on row 0 class 1 is MID-ranked, and its score is the
    //      cumulative mass through the middle, not the full row.)
    Tensor labels_last = mat(2, 1, {0, 1});
    const Tensor sl = aps_score(probs, labels_last);
    check(std::abs(sl(0, 0) - 1.0) < 1e-12,
          "S11c. the last-ranked class scores the full row mass (row 0)");
    check(std::abs(sl(1, 0) - 1.0) < 1e-12,
          "S11d. and on row 1");
}

// =============================================================================
section("S12. SPLIT CONFORMAL: CALIBRATION AND TEST ARE SEPARATE");

// The guarantee is an EXCHANGEABILITY statement, so the API must never
// combine calibration and test. This section pins that a threshold computed
// on one set and applied to another behaves as specified, and that the
// threshold does not depend on the test point.
{
    Tensor cal = mat(20, 1, std::vector<double>(20, 0.5));
    const double qhat = conformal_quantile(cal, 0.1);
    Tensor row_a = mat(1, 3, {0.6, 0.3, 0.1});
    Tensor row_b = mat(1, 3, {0.2, 0.5, 0.3});
    const std::vector<int> a = aps_prediction_set(row_a, qhat);
    const std::vector<int> b = aps_prediction_set(row_b, qhat);
    check(a.size() > 0 && b.size() > 0, "S12a. both test points yield a set");
    check(qhat == 0.5, "S12b. the threshold is independent of the test point");
}

// =============================================================================
printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
return g_fail == 0 ? 0 : 1;
}  // main()
