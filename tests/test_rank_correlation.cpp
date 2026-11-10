// =============================================================================
// tests/test_rank_correlation.cpp -- the rank correlation and Gaussian-copula
// utilities.
//
// THE SPECIFICATION IS tests/rank_correlation_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed and was green on its
// first execution (96/96). A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING
// ABOUT ITS OWN SENSITIVITY, so its discriminating power was MEASURED
// instead of narrated: tests/rank_correlation_mutate_oracle.py breaks the
// oracle in seven ways and requires that it fail each time (7/7 caught).
//
// =============================================================================
// WHY THIS FEATURE EXISTS (measured 2026-11-10)
// =============================================================================
//
//     grep -rinE 'copula|rank.?correl|spearman|kendall' include/ tests/
//     -> 0 content hits
//
// The repo has ~250 layers and 60+ utility files and NO correlation
// machinery at all: not Pearson's r, not Spearman's rho, not Kendall's
// tau, and nothing copula-shaped.
//
// =============================================================================
// THE THREE TIE CONVENTIONS, AND WHY THE FIXTURE CARRIES JOINT TIES
// =============================================================================
//
//  1. RANKS ARE MID-RANKS. A tied group gets the AVERAGE of the ranks it
//     spans, not the first rank of the group.
//
//  2. KENDALL'S TAU-B'S TIE CORRECTION: the joint ties count in BOTH xtie
//     AND ytie. Counting them in NEITHER is the classic bug, and it is
//     INVISIBLE on a fixture with no ties.
//
//  3. THE PLOTTING POSITION IS (r - 0.5)/n (Hazen), not Blomqvist's
//     (r - 0.375)/(n + 0.25) and not the Weibull r/(n+1).
//
// THE PINNED FIXTURE carries JOINT TIES on purpose, so all three
// conventions are exercised at once and a fixture that cannot tell them
// apart cannot pass.
//
// The pinned numbers were TRANSFERRED from the oracle's emitted C++
// literals, not hand-transcribed. The repo has been burned by
// hand-transcribed numbers before (see tests/_dump_fixture.py): both C++
// numbers were CORRECT for the fixture they were given, and the fixture was
// simply not the oracle's.
// =============================================================================

#include "nn/utils/rank_correlation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// `Tensor` is at GLOBAL scope in this repo (include/nn/core/tensor.h has no
// namespace -- measured, `grep -c 'namespace'` -> 0), so it needs no
// qualification and no using-declaration. The utility functions live in
// `nn::utils`. This is the same convention tests/test_off_policy_eval.cpp
// uses.
using nn::utils::gaussian_copula_correlation;
using nn::utils::gaussian_copula_rho_from_spearman;
using nn::utils::gaussian_copula_rho_from_tau;
using nn::utils::gaussian_copula_spearman_from_rho;
using nn::utils::gaussian_copula_tau_from_rho;
using nn::utils::kendall_tau_b;
using nn::utils::mid_ranks;
using nn::utils::normal_scores;
using nn::utils::pearson_r;
using nn::utils::spearman_rho;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& msg) {
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << std::endl;
    }
}

static void close_to(double got, double want, double tol,
                     const std::string& msg) {
    const bool ok = std::fabs(got - want) <= tol;
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (got " << got << ", want " << want
                  << ", diff " << (got - want) << ")" << std::endl;
    }
}

// A 1-D sample as a (N,1) Tensor.
static Tensor col(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// The same values as a (1,N) Tensor -- the convention is that the VALUES
// are what matter, so this must give identical answers.
static Tensor row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t i = 0; i < v.size(); ++i) t(0, i) = v[i];
    return t;
}

// ---------------------------------------------------------------------------
// The guard helpers.
//
// A message is read AFTER the call it belongs to. C++ evaluates a function
// call's arguments in UNSPECIFIED order, so `check(threw_with(fn, needle),
// "..." + what)` can print the message from the PREVIOUS call: the
// verdicts stay right and the human-readable text goes stale, which is the
// worst failure mode, because a reader trusts the text.
// ---------------------------------------------------------------------------
static bool threw_with(const std::function<void()>& fn, const std::string& needle,
                       std::string* what_out) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (what_out) *what_out = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void expect_throw(const std::function<void()>& fn, const std::string& needle,
                         const std::string& msg) {
    std::string what;
    const bool threw = threw_with(fn, needle, &what);
    if (threw) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "   (expected a throw mentioning \""
                  << needle << "\", got \"" << what << "\")" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// THE PINNED FIXTURE. Emitted by tests/rank_correlation_oracle.py.
//
// It carries JOINT TIES (the pairs (1,2), (2,3), (4,5) and (9,10) of fx
// are tied in BOTH fx and fy), so all three tie conventions are exercised.
// ---------------------------------------------------------------------------
static const std::vector<double> kFx = {1, 1, 2, 2, 3, 4, 4, 5, 6, 6};
static const std::vector<double> kFy = {2, 2, 1, 3, 3, 4, 5, 5, 6, 4};

static const double kMidX[] = {1.5, 1.5, 3.5, 3.5, 5.0, 6.5, 6.5, 8.0, 9.5, 9.5};
static const double kMidY[] = {2.5, 2.5, 1.0, 4.5, 4.5, 6.5, 8.5, 8.5, 10.0, 6.5};

static const double kPearson = 0.85185185185185186;
static const double kSpearman = 0.8571428571428571;
static const double kTauB = 0.73170731707317072;
static const double kCopulaCorr = 0.79775318291921482;
static const double kNormalScoresX[] = {
    -1.2815515655446004,  -1.2815515655446004, -0.52440051270804089,
    -0.52440051270804089, -0.12566134685507402, 0.25334710313579972,
    0.25334710313579972,  0.67448975019608171,  1.2815515655446004,
    1.2815515655446004};

// =============================================================================
// Section 1: mid-ranks
// =============================================================================
static void test_mid_ranks() {
    std::cout << "\n-- Test 1: mid-ranks --" << std::endl;

    // 1.1 a hand-computed fixture: the tied pair spans ranks 2 and 3, so it
    //     gets 2.5.
    Tensor x = col({10, 20, 20, 30});
    Tensor r = mid_ranks(x);
    close_to(r(0, 0), 1.0, 1e-15, "1.1a mid-ranks: the smallest value gets rank 1");
    close_to(r(1, 0), 2.5, 1e-15, "1.1b mid-ranks: the tied pair spans 2 and 3, so it gets 2.5");
    close_to(r(2, 0), 2.5, 1e-15, "1.1c mid-ranks: both members of the tie get 2.5");
    close_to(r(3, 0), 4.0, 1e-15, "1.1d mid-ranks: the largest value gets rank 4");

    // 1.2 NO TIES -> 1..n, and the transform is ORDER-PRESERVING.
    Tensor y = col({5, 1, 3, 2, 4});
    Tensor ry = mid_ranks(y);
    double seen[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < 5; ++i) seen[static_cast<int>(ry(i, 0)) - 1] = y(i, 0);
    bool perm = true;
    for (int i = 0; i < 5; ++i) perm = perm && (seen[i] == i + 1);
    check(perm, "1.2 mid-ranks of a tie-free sample are the permutation 1..n, in order");

    // 1.3 THE "FIRST RANK" FOIL MUST DIFFER. A tied group gets the AVERAGE
    //     of the ranks it spans, not the first of them.
    Tensor tie4 = col({1, 1, 1, 1, 5});
    Tensor r4 = mid_ranks(tie4);
    close_to(r4(0, 0), 2.5, 1e-15,
             "1.3a a 4-way tie spans ranks 1..4 and gets 2.5 (not the first rank, 1)");
    double first_rank_foil = 1.0;
    check(std::fabs(r4(0, 0) - first_rank_foil) > 1e-9,
          "1.3b the 'first rank of the tie' foil DIFFERS from the mid-rank");

    // 1.4 the pinned fixture's mid-ranks, exactly.
    Tensor fx = col(kFx);
    Tensor fy = col(kFy);
    Tensor mfx = mid_ranks(fx);
    Tensor mfy = mid_ranks(fy);
    bool mid_ok = true;
    for (size_t i = 0; i < kFx.size(); ++i) {
        mid_ok = mid_ok && std::fabs(mfx(i, 0) - kMidX[i]) < 1e-12;
        mid_ok = mid_ok && std::fabs(mfy(i, 0) - kMidY[i]) < 1e-12;
    }
    check(mid_ok, "1.4 the pinned fixture's mid-ranks match the oracle exactly");

    // 1.5 the shape is preserved.
    Tensor r2 = mid_ranks(row(kFx));
    check(r2.rows == 1 && r2.cols == kFx.size(),
          "1.5 mid_ranks preserves the input shape");
}

// =============================================================================
// Section 2: Spearman's rho
// =============================================================================
static void test_spearman() {
    std::cout << "\n-- Test 2: Spearman's rho --" << std::endl;

    Tensor fx = col(kFx);
    Tensor fy = col(kFy);

    // 2.1 the pinned value.
    close_to(spearman_rho(fx, fy), kSpearman, 1e-12,
             "2.1 the pinned fixture's Spearman rho matches the oracle exactly");

    // 2.2 THE d^2 SHORTCUT DIFFERS WHEN THERE ARE TIES. The shortcut
    //     1 - 6*sum(d^2)/(n*(n^2-1)) is exact only without ties.
    {
        const size_t n = kFx.size();
        double sd2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = kMidX[i] - kMidY[i];
            sd2 += d * d;
        }
        const double shortcut = 1.0 - 6.0 * sd2 / (static_cast<double>(n) *
                                                   (static_cast<double>(n) * n - 1.0));
        printf("    the d^2 shortcut = %.12f, the definition = %.12f\n", shortcut,
               kSpearman);
        check(std::fabs(shortcut - kSpearman) > 1e-6,
              "2.2 the d^2 shortcut DIFFERS from the mid-rank definition on a tied fixture");

        // ...and it AGREES when there are no ties, so the shortcut is not
        // merely wrong, it is conditionally right.
        const std::vector<double> tx = {1, 2, 3, 4, 5, 6};
        const std::vector<double> ty = {2, 1, 4, 3, 6, 5};
        Tensor a = col(tx), b = col(ty);
        Tensor ra = mid_ranks(a), rb = mid_ranks(b);
        double sd2b = 0.0;
        for (size_t i = 0; i < tx.size(); ++i) {
            const double d = ra(i, 0) - rb(i, 0);
            sd2b += d * d;
        }
        const double short_b = 1.0 - 6.0 * sd2b / (static_cast<double>(tx.size()) *
                                                     (static_cast<double>(tx.size()) * tx.size() - 1.0));
        close_to(short_b, spearman_rho(a, b), 1e-12,
                 "2.3 the d^2 shortcut AGREES with the definition on a tie-free fixture");
    }

    // 2.4 MONOTONE INVARIANCE. Spearman depends on the ORDER, not on the
    //     values, so it is unchanged by any strictly increasing transform.
    {
        std::vector<double> vx = {1, 3, 2, 5, 4, 7, 6, 8, 10, 9};
        std::vector<double> vy = {2, 1, 4, 3, 6, 5, 8, 7, 9, 10};
        Tensor a = col(vx), b = col(vy);
        const double base = spearman_rho(a, b);
        std::vector<double> ex(vx.size()), ey(vy.size());
        for (size_t i = 0; i < vx.size(); ++i) {
            ex[i] = std::exp(vx[i]);
            ey[i] = std::pow(vy[i], 3.0);
        }
        close_to(spearman_rho(col(ex), col(ey)), base, 1e-12,
                 "2.4 Spearman is invariant under a strictly increasing transform of either sample");
    }

    // 2.5 symmetry, and the extremes.
    {
        Tensor a = col({1, 2, 3, 4, 5});
        Tensor b = col({5, 4, 3, 2, 1});
        close_to(spearman_rho(a, b), spearman_rho(b, a), 1e-15,
                 "2.5a Spearman is symmetric");
        close_to(spearman_rho(a, a), 1.0, 1e-15, "2.5b Spearman(x, x) == 1");
        close_to(spearman_rho(a, b), -1.0, 1e-15, "2.5c Spearman(x, -x) == -1");
    }

    // 2.6 the shape convention: a (N,1) and a (1,N) sample are the SAME
    //     sample, so the answers must agree.
    {
        Tensor a = col(kFx), b = col(kFy);
        Tensor ar = row(kFx), br = row(kFy);
        close_to(spearman_rho(ar, br), spearman_rho(a, b), 1e-15,
                 "2.6 Spearman gives the same answer for a (N,1) and a (1,N) sample");
    }
}

// =============================================================================
// Section 3: Kendall's tau-b
// =============================================================================
static void test_kendall() {
    std::cout << "\n-- Test 3: Kendall's tau-b --" << std::endl;

    Tensor fx = col(kFx);
    Tensor fy = col(kFy);

    // 3.1 the pinned value.
    close_to(kendall_tau_b(fx, fy), kTauB, 1e-12,
             "3.1 the pinned fixture's tau-b matches the oracle exactly");

    // 3.2 THE JOINT-TIE FOIL MUST DIFFER. Counting the joint ties in
    //     NEITHER xtie NOR ytie is the classic tau-b bug.
    {
        const size_t n = kFx.size();
        double tot = 0.5 * static_cast<double>(n) * static_cast<double>(n - 1);
        double xtie = 0, ytie = 0, ntie = 0, dis = 0, con = 0;
        double xtie_foil = 0, ytie_foil = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double dx = kFx[i] - kFx[j];
                const double dy = kFy[i] - kFy[j];
                if (dx == 0.0 && dy == 0.0) {
                    ntie += 1;
                    xtie += 1;
                    ytie += 1;
                } else if (dx == 0.0) {
                    xtie += 1;
                } else if (dy == 0.0) {
                    ytie += 1;
                } else if ((dx > 0.0) == (dy > 0.0)) {
                    con += 1;
                } else {
                    dis += 1;
                }
                if (dx == 0.0 && dy == 0.0) {
                    // the foil: joint ties counted in NEITHER
                } else if (dx == 0.0) {
                    xtie_foil += 1;
                } else if (dy == 0.0) {
                    ytie_foil += 1;
                }
            }
        }
        const double cmd = tot - xtie - ytie + ntie - 2.0 * dis;
        const double foil_cmd = tot - xtie_foil - ytie_foil + ntie - 2.0 * dis;
        const double right = cmd / std::sqrt((tot - xtie) * (tot - ytie));
        const double foil = foil_cmd / std::sqrt((tot - xtie_foil) * (tot - ytie_foil));
        printf("    the joint-tie foil = %.12f, the correct tau-b = %.12f\n", foil, right);
        check(std::fabs(right - foil) > 1e-9,
              "3.2 the joint-tie foil DIFFERS from the correct tau-b");
        close_to(right, kTauB, 1e-12,
                 "3.3 the hand-computed tau-b on the pinned fixture agrees with the oracle");
    }

    // 3.4 with NO TIES, tau-b reduces to (C-D)/tot.
    {
        const std::vector<double> tx = {1, 2, 3, 4, 5};
        const std::vector<double> ty = {1, 3, 2, 5, 4};
        double con = 0, dis = 0;
        for (size_t i = 0; i < tx.size(); ++i) {
            for (size_t j = i + 1; j < tx.size(); ++j) {
                const double dx = tx[i] - tx[j];
                const double dy = ty[i] - ty[j];
                if ((dx > 0) == (dy > 0)) ++con; else ++dis;
            }
        }
        const double want = (con - dis) / 10.0;
        close_to(kendall_tau_b(col(tx), col(ty)), want, 1e-12,
                 "3.4 tau-b reduces to (C-D)/tot when there are no ties");
    }

    // 3.5 monotone invariance, symmetry, extremes.
    {
        std::vector<double> vx = {1, 3, 2, 5, 4, 7, 6, 8, 10, 9};
        std::vector<double> vy = {2, 1, 4, 3, 6, 5, 8, 7, 9, 10};
        Tensor a = col(vx), b = col(vy);
        const double base = kendall_tau_b(a, b);
        std::vector<double> ex(vx.size()), ey(vy.size());
        for (size_t i = 0; i < vx.size(); ++i) {
            ex[i] = std::exp(vx[i]);
            ey[i] = std::pow(vy[i], 3.0);
        }
        close_to(kendall_tau_b(col(ex), col(ey)), base, 1e-12,
                 "3.5a tau-b is invariant under a strictly increasing transform");
        close_to(kendall_tau_b(a, b), kendall_tau_b(b, a), 1e-15, "3.5b tau-b is symmetric");
        Tensor same = col({1, 2, 3, 4, 5});
        close_to(kendall_tau_b(same, same), 1.0, 1e-15, "3.5c tau-b(x, x) == 1");
        close_to(kendall_tau_b(same, col({5, 4, 3, 2, 1})), -1.0, 1e-15,
                 "3.5d tau-b(x, -x) == -1");
    }

    // 3.6 tau-b of a monotone pair WITH ties is exactly 1.
    {
        Tensor a = col({1, 1, 2, 3, 3, 4});
        Tensor b = col({10, 10, 20, 30, 30, 40});
        close_to(kendall_tau_b(a, b), 1.0, 1e-15,
                 "3.6 tau-b of a monotone pair with ties is exactly 1");
    }
}

// =============================================================================
// Section 4: the normal scores
// =============================================================================
static void test_normal_scores() {
    std::cout << "\n-- Test 4: the normal scores --" << std::endl;

    // 4.1 the pinned values.
    Tensor z = normal_scores(col(kFx));
    bool ok = true;
    for (size_t i = 0; i < kFx.size(); ++i) {
        ok = ok && std::fabs(z(i, 0) - kNormalScoresX[i]) < 1e-12;
    }
    check(ok, "4.1 the pinned fixture's normal scores match the oracle exactly");

    // 4.2 THE PLOTTING-POSITION ROUND-TRIP. Phi(z_i) must return
    //     (r_i - 0.5)/n exactly. This is what pins the convention.
    {
        const std::vector<double> v = {3, 1, 4, 1, 5, 9, 2, 6};
        Tensor s = normal_scores(col(v));
        Tensor r = mid_ranks(col(v));
        const double n = static_cast<double>(v.size());
        double worst = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            const double p = (r(i, 0) - 0.5) / n;
            // Phi(z) via the same erfc the library uses, so the round-trip is
            // checked against the transformation's own inverse
            const double back = 0.5 * std::erfc(-s(i, 0) / std::sqrt(2.0));
            worst = std::max(worst, std::fabs(back - p));
        }
        printf("    worst |Phi(z) - (r-0.5)/n| = %.3e\n", worst);
        check(worst < 1e-12,
              "4.2 Phi(normal_scores) reproduces the (r-0.5)/n plotting position");

        // the two neighbouring conventions must NOT give the same answer
        const double blomqvist = (r(0, 0) - 0.375) / (n + 0.25);
        const double hazen = (r(0, 0) - 0.5) / n;
        check(std::fabs(blomqvist - hazen) > 1e-6,
              "4.3 the Blomqvist plotting position differs from Hazen's, so the convention is observable");
    }

    // 4.4 ORDER-PRESERVING, and the transform is MONOTONE-INVARIANT.
    {
        std::vector<double> v = {5, 2, 9, 1, 7, 3, 8, 4, 6, 0};
        Tensor z = normal_scores(col(v));
        bool ordered = true;
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            if (v[i] < v[i + 1] && !(z(i, 0) < z(i + 1, 0))) ordered = false;
            if (v[i] > v[i + 1] && !(z(i, 0) > z(i + 1, 0))) ordered = false;
        }
        check(ordered, "4.4 the normal scores preserve the order of the data");

        std::vector<double> ev(v.size());
        for (size_t i = 0; i < v.size(); ++i) ev[i] = std::exp(v[i]);
        Tensor ze = normal_scores(col(ev));
        double worst = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            worst = std::max(worst, std::fabs(ze(i, 0) - z(i, 0)));
        }
        check(worst < 1e-12,
              "4.5 the normal scores are invariant under a strictly increasing transform");
    }

    // 4.6 the shape is preserved.
    Tensor zr = normal_scores(row(kFx));
    check(zr.rows == 1 && zr.cols == kFx.size(),
          "4.6 normal_scores preserves the input shape");
}

// =============================================================================
// Section 5: the Gaussian copula correlation
// =============================================================================
static void test_copula_correlation() {
    std::cout << "\n-- Test 5: the Gaussian copula correlation --" << std::endl;

    Tensor fx = col(kFx);
    Tensor fy = col(kFy);

    // 5.1 the pinned value.
    close_to(gaussian_copula_correlation(fx, fy), kCopulaCorr, 1e-12,
             "5.1 the pinned fixture's copula correlation matches the oracle exactly");

    // 5.2 MARGINAL-FREENESS -- the defining property of a copula
    //     statistic. It is INVARIANT under any strictly increasing
    //     transformation of either marginal.
    {
        std::vector<double> vx = {1, 3, 2, 5, 4, 7, 6, 8, 10, 9};
        std::vector<double> vy = {2, 1, 4, 3, 6, 5, 8, 7, 9, 10};
        Tensor a = col(vx), b = col(vy);
        const double base = gaussian_copula_correlation(a, b);
        std::vector<double> ex(vx.size()), ey(vy.size());
        for (size_t i = 0; i < vx.size(); ++i) {
            ex[i] = std::exp(vx[i]);
            ey[i] = std::pow(vy[i], 3.0);
        }
        close_to(gaussian_copula_correlation(col(ex), col(ey)), base, 1e-12,
                 "5.2 the copula correlation is INVARIANT under a strictly increasing transform "
                 "of either marginal");

        // 5.3 and the RAW Pearson correlation is NOT, which is what makes
        //     5.2 a real property rather than a tautology.
        const double raw_base = pearson_r(a, b);
        const double raw_exp = pearson_r(col(ex), b);
        printf("    raw pearson = %.12f, raw pearson of exp = %.12f\n", raw_base, raw_exp);
        check(std::fabs(raw_base - raw_exp) > 1e-6,
              "5.3 the RAW Pearson correlation is NOT marginal-free, so 5.2 discriminates");
    }

    // 5.4 symmetry and the extremes.
    {
        Tensor a = col({1, 2, 3, 4, 5});
        Tensor b = col({5, 4, 3, 2, 1});
        close_to(gaussian_copula_correlation(a, b), gaussian_copula_correlation(b, a), 1e-15,
                 "5.4a the copula correlation is symmetric");
        close_to(gaussian_copula_correlation(a, a), 1.0, 1e-15,
                 "5.4b the copula correlation of a sample with itself is 1");
    }

    // 5.5 the copula correlation is DIFFERENT from the raw Pearson
    //     correlation on a tied fixture -- so the two are not the same
    //     function wearing different names.
    //
    //     THE FIXTURE MUST BE TWO DISTINCT SAMPLES. An earlier draft used
    //     `a == b`, where BOTH statistics are exactly 1 by definition (a
    //     sample correlates perfectly with itself), so the fixture could not
    //     discriminate between them at all. **A CHECK THAT CANNOT FAIL IS
    //     THE WORST KIND OF CHECK**, and this one was failing for the
    //     right reason -- the fixture, not the implementation.
    {
        Tensor a = col({1, 1, 2, 2, 3, 4, 4, 5, 6, 6});
        Tensor b = col({2, 2, 1, 3, 3, 4, 5, 5, 6, 4});
        const double raw = pearson_r(a, b);
        const double cop = gaussian_copula_correlation(a, b);
        printf("    raw pearson = %.12f, copula correlation = %.12f\n", raw, cop);
        check(std::fabs(raw - cop) > 1e-6,
              "5.5 the copula correlation and the raw Pearson correlation DIFFER on a tied fixture");
    }

    // 5.6 THE LARGE-OFFSET FIXTURE. `pearson_r` uses the TWO-PASS CENTRED
    //     form, and this check is what makes that choice observable.
    //
    //     **THE MUTATION HARNESS CAUGHT THIS GAP.** The M10 mutation
    //     (replace the centred form with the one-pass
    //     `sum(xy) - sum(x)sum(y)/n` form) SURVIVED the first version of
    //     this suite, because every fixture had small, well-conditioned
    //     values where the two forms agree to 1e-15. The two forms are
    //     algebraically IDENTICAL and differ only in FLOATING POINT: the
    //     one-pass form subtracts two nearly-equal large numbers, and the
    //     loss is catastrophic for large-offset data.
    //
    //     Measured on the fixture below (base 1e8): the centred form gives
    //     0.90286102487357334 and the one-pass form gives
    //     0.77032888651964315 -- a difference of 1.3e-1. On data with a
    //     larger offset the one-pass form does not merely lose precision,
    //     it returns -inf and then nan (the variance term underflows to
    //     zero or goes negative). **A SURVIVOR IS A HOLE IN THE SUITE**,
    //     and this is the fixture that closes it.
    {
        std::vector<double> ba, bb;
        for (int i = 0; i < 50; ++i) {
            ba.push_back(1e8 + 0.5 * i + 0.25 * ((i * 7919) % 13));
            bb.push_back(1e8 + 0.25 * i + 0.5 * ((i * 104729) % 11));
        }
        Tensor a = col(ba), b = col(bb);
        const double got = pearson_r(a, b);
        printf("    large-offset pearson_r = %.17g\n", got);

        // The centred form is the reference: recompute it here, INDEPENDENT
        // of the library, and require agreement. If the library used the
        // one-pass form, this would disagree by ~1.3e-1.
        double ma = 0.0, mb = 0.0;
        for (size_t i = 0; i < ba.size(); ++i) { ma += ba[i]; mb += bb[i]; }
        ma /= static_cast<double>(ba.size());
        mb /= static_cast<double>(ba.size());
        double num = 0.0, saa = 0.0, sbb = 0.0;
        for (size_t i = 0; i < ba.size(); ++i) {
            const double da = ba[i] - ma;
            const double db = bb[i] - mb;
            num += da * db;
            saa += da * da;
            sbb += db * db;
        }
        const double want = num / std::sqrt(saa * sbb);
        close_to(got, want, 1e-12,
                 "5.6 pearson_r uses the CENTRED form: it matches the two-pass reference "
                 "on large-offset data");

        // and the one-pass form genuinely DISAGREYS there, so the check
        // discriminates rather than passing vacuously.
        double op_num = 0.0, op_saa = 0.0, op_sbb = 0.0;
        for (size_t i = 0; i < ba.size(); ++i) {
            op_num += ba[i] * bb[i];
            op_saa += ba[i] * ba[i];
            op_sbb += bb[i] * bb[i];
        }
        const double nn_ = static_cast<double>(ba.size());
        op_num -= nn_ * ma * mb;
        op_saa -= nn_ * ma * ma;
        op_sbb -= nn_ * mb * mb;
        const double one_pass = op_num / std::sqrt(op_saa * op_sbb);
        printf("    the one-pass foil      = %.17g\n", one_pass);
        check(std::fabs(got - one_pass) > 1e-6,
              "5.6b the one-pass Pearson foil DIFFERS from pearson_r on large-offset data, "
              "so 5.6 discriminates");
    }
}

// =============================================================================
// Section 6: the four exact population correspondences
// =============================================================================
static void test_population_correspondences() {
    std::cout << "\n-- Test 6: the Gaussian-copula population correspondences --" << std::endl;

    const double kPi = 3.14159265358979323846;

    // 6.1 tau = (2/pi)asin(rho), against the oracle's INDEPENDENT
    //     quadrature-verified values (the closed forms, pinned from the
    //     oracle's section 6).
    {
        struct { double rho, tau; } cases[] = {
            {0.0, 0.0},
            {0.25, 2.0 / kPi * std::asin(0.25)},
            {-0.5, 2.0 / kPi * std::asin(-0.5)},
            {0.75, 2.0 / kPi * std::asin(0.75)},
        };
        bool ok = true;
        for (const auto& c : cases) {
            ok = ok && std::fabs(gaussian_copula_tau_from_rho(c.rho) - c.tau) < 1e-15;
        }
        check(ok, "6.1 tau = (2/pi)arcsin(rho) holds at the pinned points");

        // the factor-2 foil must be distinguishable
        check(std::fabs(gaussian_copula_tau_from_rho(0.6) - (1.0 / kPi) * std::asin(0.6)) > 1e-6,
              "6.2 the factor-2 foil for tau is distinguishable");
    }

    // 6.3 rho_S = (6/pi)asin(rho/2).
    {
        bool ok = true;
        for (double rho : {0.0, 0.25, -0.5, 0.75}) {
            ok = ok && std::fabs(gaussian_copula_spearman_from_rho(rho) -
                                 (6.0 / kPi) * std::asin(rho / 2.0)) < 1e-15;
        }
        check(ok, "6.3 rho_S = (6/pi)arcsin(rho/2) holds at the pinned points");

        // the dropped-/2 foil must be distinguishable
        check(std::fabs(gaussian_copula_spearman_from_rho(0.6) -
                        (6.0 / kPi) * std::asin(0.6)) > 1e-6,
              "6.4 the rho/2-dropped foil for rho_S is distinguishable");
    }

    // 6.5 the inverses are exact inverses.
    {
        bool ok = true;
        for (double rho : {0.0, 0.25, -0.5, 0.75, 0.9, -0.9}) {
            ok = ok && std::fabs(gaussian_copula_rho_from_tau(
                       gaussian_copula_tau_from_rho(rho)) - rho) < 1e-14;
            ok = ok && std::fabs(gaussian_copula_rho_from_spearman(
                       gaussian_copula_spearman_from_rho(rho)) - rho) < 1e-14;
        }
        check(ok, "6.5 the two inverses are exact inverses (round-trip to 1e-14)");
    }

    // 6.6 the endpoints.
    close_to(gaussian_copula_tau_from_rho(1.0), 1.0, 1e-15, "6.6a tau(rho=1) == 1");
    close_to(gaussian_copula_tau_from_rho(-1.0), -1.0, 1e-15, "6.6b tau(rho=-1) == -1");
    close_to(gaussian_copula_spearman_from_rho(1.0), 1.0, 1e-15, "6.6c rho_S(rho=1) == 1");
    close_to(gaussian_copula_spearman_from_rho(-1.0), -1.0, 1e-15, "6.6d rho_S(rho=-1) == -1");
    close_to(gaussian_copula_tau_from_rho(0.0), 0.0, 1e-15, "6.6e tau(rho=0) == 0");
    close_to(gaussian_copula_spearman_from_rho(0.0), 0.0, 1e-15, "6.6f rho_S(rho=0) == 0");

    // 6.7 the two population values are ORDERED for 0 < rho < 1.
    {
        bool ok = true;
        for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            ok = ok && gaussian_copula_tau_from_rho(rho) <
                        gaussian_copula_spearman_from_rho(rho);
        }
        check(ok, "6.7 for 0 < rho < 1 the population tau is strictly below the population rho_S");
    }
}

// =============================================================================
// Section 7: the guards and the degenerate inputs
// =============================================================================
static void test_guards() {
    std::cout << "\n-- Test 7: the guards and the degenerate inputs --" << std::endl;

    // 7.1 an empty sample is REJECTED.
    Tensor empty(0, 1);
    Tensor one = col({1.0});
    expect_throw([&]() { mid_ranks(empty); }, "the sample is empty",
                 "7.1a mid_ranks rejects an empty sample");
    expect_throw([&]() { spearman_rho(empty, one); }, "the sample is empty",
                 "7.1b spearman_rho rejects an empty sample");
    expect_throw([&]() { kendall_tau_b(empty, one); }, "the sample is empty",
                 "7.1c kendall_tau_b rejects an empty sample");
    expect_throw([&]() { pearson_r(empty, one); }, "the sample is empty",
                 "7.1d pearson_r rejects an empty sample");
    expect_throw([&]() { normal_scores(empty); }, "the sample is empty",
                 "7.1e normal_scores rejects an empty sample");
    expect_throw([&]() { gaussian_copula_correlation(empty, one); },
                 "the sample is empty",
                 "7.1f gaussian_copula_correlation rejects an empty sample");

    // 7.2 a length MISMATCH is rejected -- and with its OWN message, not
    //     the empty one.
    Tensor a = col({1.0, 2.0, 3.0});
    Tensor b = col({1.0, 2.0});
    expect_throw([&]() { spearman_rho(a, b); }, "same number of samples",
                 "7.2a spearman_rho rejects a length mismatch");
    expect_throw([&]() { kendall_tau_b(a, b); }, "same number of samples",
                 "7.2b kendall_tau_b rejects a length mismatch");
    expect_throw([&]() { pearson_r(a, b); }, "same number of samples",
                 "7.2c pearson_r rejects a length mismatch");
    expect_throw([&]() { gaussian_copula_correlation(a, b); }, "same number of samples",
                 "7.2d gaussian_copula_correlation rejects a length mismatch");

    // 7.3 fewer than two points cannot carry an association.
    expect_throw([&]() { spearman_rho(one, one); }, "at least two samples",
                 "7.3a spearman_rho rejects a single point");
    expect_throw([&]() { kendall_tau_b(one, one); }, "at least two samples",
                 "7.3b kendall_tau_b rejects a single point");

    // 7.4 A CONSTANT SAMPLE IS NOT AN ERROR. Its correlation is genuinely
    //     UNDEFINED, and the reference implementation returns NaN. The
    //     module returns NaN, and the constant-sample case is a DIFFERENT
    //     situation from the empty one, with a different answer.
    {
        Tensor konst = col({5.0, 5.0, 5.0, 5.0});
        Tensor vary = col({1.0, 2.0, 3.0, 4.0});
        const double sp = spearman_rho(konst, vary);
        const double ka = kendall_tau_b(konst, vary);
        const double pc = pearson_r(konst, vary);
        const double cc = gaussian_copula_correlation(konst, vary);
        check(std::isnan(sp), "7.4a spearman_rho of a constant sample is NaN, not an error");
        check(std::isnan(ka), "7.4b kendall_tau_b of a constant sample is NaN, not an error");
        check(std::isnan(pc), "7.4c pearson_r of a constant sample is NaN, not an error");
        check(std::isnan(cc), "7.4d gaussian_copula_correlation of a constant sample is NaN, not an error");

        // and it is NOT zero -- zero would be a confident wrong answer.
        check(sp != 0.0 && ka != 0.0,
              "7.4e the undefined correlations are NaN and NOT zero");
    }

    // 7.5 the population correspondences reject an argument outside
    //     [-1, 1]. asin of it would be NaN, which would silently poison a
    //     downstream computation.
    expect_throw([]() { gaussian_copula_tau_from_rho(1.5); }, "must be in [-1, 1]",
                 "7.5a gaussian_copula_tau_from_rho rejects rho > 1");
    expect_throw([]() { gaussian_copula_rho_from_tau(-2.0); }, "must be in [-1, 1]",
                 "7.5b gaussian_copula_rho_from_tau rejects tau < -1");
    expect_throw([]() { gaussian_copula_spearman_from_rho(1.0001); }, "must be in [-1, 1]",
                 "7.5c gaussian_copula_spearman_from_rho rejects rho > 1");
    expect_throw([]() { gaussian_copula_rho_from_spearman(-1.5); }, "must be in [-1, 1]",
                 "7.5d gaussian_copula_rho_from_spearman rejects rho_S < -1");

    // 7.6 the guard messages are DISTINCT IN KIND. An empty sample, a
    //     length mismatch and a one-point sample are three DIFFERENT
    //     situations and must not share a message.
    {
        std::string m_empty, m_mismatch, m_short;
        threw_with([&]() { spearman_rho(empty, one); }, "the sample is empty", &m_empty);
        threw_with([&]() { spearman_rho(a, b); }, "same number of samples", &m_mismatch);
        threw_with([&]() { spearman_rho(one, one); }, "at least two samples", &m_short);
        check(!m_empty.empty() && !m_mismatch.empty() && !m_short.empty() &&
                  m_empty != m_mismatch && m_mismatch != m_short && m_empty != m_short,
              "7.6 the three guard messages are distinct from each other");
    }
}

// =============================================================================
// Section 8: the whole thing together, on data with a KNOWN answer
// =============================================================================
static void test_end_to_end() {
    std::cout << "\n-- Test 8: end to end --" << std::endl;

    // 8.1 a MONOTONE pair: every coefficient must be exactly 1.
    {
        Tensor a = col({1, 2, 3, 4, 5, 6, 7, 8});
        Tensor b = col({2, 4, 6, 8, 10, 12, 14, 16});
        close_to(pearson_r(a, b), 1.0, 1e-15, "8.1a pearson_r of a linear pair is 1");
        close_to(spearman_rho(a, b), 1.0, 1e-15, "8.1b spearman_rho of a linear pair is 1");
        close_to(kendall_tau_b(a, b), 1.0, 1e-15, "8.1c kendall_tau_b of a linear pair is 1");
        close_to(gaussian_copula_correlation(a, b), 1.0, 1e-15,
                 "8.1d the copula correlation of a linear pair is 1");
    }

    // 8.2 a STRICTLY DECREASING pair: every coefficient must be exactly -1.
    {
        Tensor a = col({1, 2, 3, 4, 5, 6, 7, 8});
        Tensor b = col({16, 14, 12, 10, 8, 6, 4, 2});
        close_to(spearman_rho(a, b), -1.0, 1e-15, "8.2a spearman_rho of a reversed pair is -1");
        close_to(kendall_tau_b(a, b), -1.0, 1e-15, "8.2b kendall_tau_b of a reversed pair is -1");
    }

    // 8.3 the THREE statistics are three DIFFERENT functions of the data.
    //     A module that computed the same thing three times would pass
    //     every check above, so the difference is asserted explicitly.
    {
        Tensor a = col({1, 1, 2, 3, 3, 4, 5, 5, 6, 6});
        Tensor b = col({2, 2, 1, 3, 3, 4, 5, 5, 6, 4});
        const double sp = spearman_rho(a, b);
        const double ka = kendall_tau_b(a, b);
        const double cc = gaussian_copula_correlation(a, b);
        printf("    pearson=%.12f spearman=%.12f tau_b=%.12f copula=%.12f\n",
               pearson_r(a, b), sp, ka, cc);
        check(std::fabs(sp - ka) > 1e-6, "8.3a spearman_rho and kendall_tau_b DIFFER on a tied fixture");
        check(std::fabs(sp - cc) > 1e-6, "8.3b spearman_rho and the copula correlation DIFFER on a tied fixture");
    }

    // 8.4 the normal scores of the pinned fixture, checked against the
    //     oracle's pinned values one more time, through the public API.
    {
        Tensor z = normal_scores(col(kFx));
        bool ok = true;
        for (size_t i = 0; i < kFx.size(); ++i) {
            ok = ok && std::fabs(z(i, 0) - kNormalScoresX[i]) < 1e-12;
        }
        check(ok, "8.4 the public normal_scores matches the oracle's pinned values");
    }
}

int main() {
    std::cout << "=== test_rank_correlation: the rank correlation and Gaussian-copula utilities ==="
              << std::endl;

    test_mid_ranks();
    test_spearman();
    test_kendall();
    test_normal_scores();
    test_copula_correlation();
    test_population_correspondences();
    test_guards();
    test_end_to_end();

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
