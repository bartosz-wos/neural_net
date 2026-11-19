// =============================================================================
// tests/test_survival.cpp -- the SURVIVAL-ANALYSIS utilities: the
// Kaplan-Meier estimator, the Nelson-Aalen cumulative hazard, the log-rank
// test, Harrell's concordance index, and the Cox proportional-hazards
// partial log-likelihood with its analytic gradient.
//
// WHAT THIS SUITE EXISTS TO PIN
// =============================================================================
//
// The repo ships ~250 layers and NO survival analysis. This suite pins the
// estimators and, just as importantly, the CONVENTIONS, because every one of
// them is a place where a plausible implementation silently returns a
// different number:
//
//   1. THE RISK SET AT t IS {j : t_j >= t}. A subject CENSORED at exactly
//      t is still at risk at t. The opposite convention gives S(1) = 1/2
//      where the right one gives 2/3, and the F9 fixture pins it.
//
//   2. THE CURVE IS RIGHT-CONTINUOUS: it takes its POST-EVENT value at
//      the event time and is flat until the next one.
//
//   3. THE MEDIAN IS THE FIRST EVENT TIME WITH S <= 0.5, not an
//      interpolated point. F11's curve CROSSES 0.5 strictly between two
//      event times, so the two rules disagree there.
//
//   4. GREENWOOD'S VARIANCE IS SHIPPED, AND IT IS FINITE WHEN THE WHOLE
//      RISK SET DIES (the n_risk == d case is a LIMIT of 0, not a
//      division by zero). F8 is the everyone-dies-at-once fixture.
//
//   5. THE NELSON-AALEN HAZARD IS REPORTED ON THE EVENT-TIME GRID -- a
//      censored-only time is not a point on it.
//
//   6. THE LOG-RANK VARIANCE IS THE HYPERGEOMETRIC ONE, INCLUDING the
//      finite-population factor (n-d)/(n-1). The F5 fixture SHARES event
//      times between the groups precisely so that this factor is not 1,
//      because with d = 1 everywhere the factor is exactly 1 and an
//      implementation that drops it is arithmetically indistinguishable.
//
//   7. THE CONCORDANCE INDEX IS ON THE RISK SCORE: HIGHER SCORE = HIGHER
//      RISK = SHORTER SURVIVAL. lifelines uses the OPPOSITE sign, so the
//      two disagree by the COMPLEMENT (1 - C). The suite pins both the
//      absolute value AND the complementarity identity C(s) + C(-s) == 1.
//
//   8. THE COX TIE HANDLING IS A PARAMETER, and Breslow and Efron COINCIDE
//      when no two events share a time -- which is exactly why the suite
//      has a fixture WITH tied event times and asserts the two DIFFER.
//
//   9. THE COX PARTIAL LOG-LIKELIHOOD IS A MEAN OVER THE EVENTS, NOT A
//      RAW SUM, and the suite asserts BOTH the mean AND that the sum
//      differs from it.
//
//  10. THE COX GRADIENT IS CHECKED AGAINST FINITE DIFFERENCES AND against
//      an INDEPENDENT SCALAR-LOOP REFERENCE computed in THIS FILE. The
//      reference exists because a finite-difference check compares the
//      implementation against ITSELF and therefore cannot see a mutation
//      that moves the value and the gradient TOGETHER.
//
// THE FIXTURES ARE EMITTED, NEVER TYPED.
//
// The numbers below come from tests/survival_oracle.py (float64 numpy),
// which was run BEFORE this file existed. The repo has been burned by
// hand-transcribed numbers before -- both numbers were correct for the
// fixture they were given, and the fixture was simply not the oracle's --
// so the pinned numbers are TRANSFERRED by tests/splice_survival_fixture.py,
// never re-derived.
//
// THE ORACLE'S OWN SENSITIVITY IS MEASURED.
//
// tests/survival_mutate_oracle.py breaks the oracle's model code in twenty
// specific ways and requires that it FAIL each time. Its FIRST run reported
// THREE survivors, and every one was a real gap in the FIXTURES or in the
// checks rather than a droppable mutation:
//
//   * "the median is INTERPOLATED" survived because the median fixture's
//     survival curve landed EXACTLY on 0.5, where interpolation returns
//     the same point. Fixed by adding F11, whose curve crosses 0.5
//     strictly between two event times.
//   * "the log-rank variance drops the finite-population factor" survived
//     because EVERY event time in the fixture carried d = 1, and
//     (n-d)/(n-1) is then exactly 1. Fixed by making the two groups SHARE
//     event times.
//   * "the Cox Breslow denominator keeps the tied deaths' contribution"
//     survived because the finite-difference check compares the
//     implementation against ITSELF, and the mutation moved the value and
//     the gradient CONSISTENTLY. Fixed by adding the independent
//     scalar-loop reference.
//
// THE INDEPENDENT REFERENCE.
//
// The Cox partial log-likelihood and its gradient are ALSO recomputed in
// THIS FILE, from the definition, with explicit scalar loops and no shared
// code with the library. The library is not checked against itself.
//
// The absolute Cox values are pinned against the ORACLE's, and the oracle
// cross-checks them against lifelines 0.30.3 -- a MAINTAINED reference
// implementation.
// =============================================================================

#include "nn/nn.h"

#include <cmath>
#include <cstdio>
#include <functional>
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
    // FLUSH ON EVERY CHECK. A buffered stdout loses its buffer when the
    // process dies, and a SEGFAULT then looks like a SILENT crash -- which
    // the mutation harness cannot distinguish from a detected defect.
    fflush(stdout);
}

static void check_near(double got, double want, double rel_tol,
                       const std::string& name) {
    if (!std::isfinite(want) && !std::isfinite(got)) {
        // Both infinite (or both NaN): compare exactly.
        check(got == want || (std::isnan(got) && std::isnan(want)), name);
        return;
    }
    const double denom = std::max(std::fabs(want), 1e-12);
    const double rel = std::fabs(got - want) / denom;
    if (rel <= rel_tol) {
        ++g_pass;
        printf("  [PASS] %s (got %.15g, want %.15g, rel %.2e)\n",
               name.c_str(), got, want, rel);
    } else {
        ++g_fail;
        printf("  [FAIL] %s (got %.15g, want %.15g, rel %.2e > %.2e)\n",
               name.c_str(), got, want, rel, rel_tol);
    }
    fflush(stdout);
}

// A throwing call, converted into a REPORTED failure. An exception that
// escapes main() aborts the process and loses every result after it, so
// a clean failure turns into a crash -- which the mutation harness cannot
// score.
//
// THE MESSAGE IS READ AFTER THE CALL, never before: C++ evaluates a
// function call's arguments in UNSPECIFIED order, so a helper that writes
// the message and a caller that reads it in the same expression can print
// the message from the PREVIOUS call.
static bool throws_with(const std::function<void()>& fn, const std::string& needle,
                        std::string* what_out = nullptr) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (what_out) *what_out = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helpers to turn the oracle's flat vectors into the library's Tensors.
// ---------------------------------------------------------------------------
static Tensor col(const std::vector<double>& v) {
    Tensor t(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) t(i, 0) = v[i];
    return t;
}

// A coefficient vector is a ROW, (1, D) -- the repo's convention for a
// vector-valued quantity (see utils/sample_quality.h's GaussianParams,
// whose `mean` is (1, d)). A COLUMN would be (D, 1) and the library
// rejects it, so the two shapes must not be confused.
static Tensor row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t j = 0; j < v.size(); ++j) t(0, j) = v[j];
    return t;
}

static Tensor mat(const std::vector<std::vector<double>>& rows) {
    Tensor t(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            t(i, j) = rows[i][j];
    return t;
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE for the Cox partial log-likelihood and its
// gradient, computed from the DEFINITION with explicit scalar loops.
//
// It shares NO code with the library. This is what makes the Cox checks
// meaningful: a finite-difference check compares the implementation against
// ITSELF, and cannot see a defect that moves the value and the gradient
// together.
//
// The formulas are the ones in the header (C9, C10, C11).
// ---------------------------------------------------------------------------
static void reference_cox(const std::vector<double>& beta,
                           const std::vector<std::vector<double>>& X,
                           const std::vector<double>& times,
                           const std::vector<double>& events,
                           bool efron,
                           double* loglik_out,
                           std::vector<double>* grad_out) {
    const size_t N = times.size();
    const size_t D = beta.size();

    // The distinct event times, ascending.
    std::vector<double> ev_times;
    for (size_t i = 0; i < N; ++i) {
        if (events[i] != 1.0) continue;
        double t = times[i];
        bool seen = false;
        for (double e : ev_times) if (e == t) { seen = true; break; }
        if (!seen) ev_times.push_back(t);
    }
    for (size_t a = 0; a < ev_times.size(); ++a)
        for (size_t b = a + 1; b < ev_times.size(); ++b)
            if (ev_times[b] < ev_times[a]) std::swap(ev_times[a], ev_times[b]);

    double total = 0.0;
    std::vector<double> grad(D, 0.0);

    for (double t : ev_times) {
        // The risk set (C1) and the deaths at this time.
        std::vector<size_t> risk, deaths;
        for (size_t i = 0; i < N; ++i) {
            if (times[i] >= t) risk.push_back(i);
            if (times[i] == t && events[i] == 1.0) deaths.push_back(i);
        }
        const double dt = static_cast<double>(deaths.size());
        if (dt == 0.0) continue;

        // eta_i = x_i . beta
        std::vector<double> eta(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < D; ++j) acc += X[i][j] * beta[j];
            eta[i] = acc;
        }

        double s0 = 0.0;
        std::vector<double> s0_x(D, 0.0);
        double s0_d = 0.0;
        std::vector<double> s0d_x(D, 0.0);
        double sum_eta_deaths = 0.0;
        std::vector<double> x_death_sum(D, 0.0);

        for (size_t i : risk) {
            double w = std::exp(eta[i]);
            s0 += w;
            for (size_t j = 0; j < D; ++j) s0_x[j] += w * X[i][j];
        }
        for (size_t i : deaths) {
            double w = std::exp(eta[i]);
            s0_d += w;
            for (size_t j = 0; j < D; ++j) s0d_x[j] += w * X[i][j];
            sum_eta_deaths += eta[i];
            for (size_t j = 0; j < D; ++j) x_death_sum[j] += X[i][j];
        }

        if (!efron) {
            total += sum_eta_deaths - dt * std::log(s0);
            for (size_t j = 0; j < D; ++j)
                grad[j] += x_death_sum[j] - dt * (s0_x[j] / s0);
        } else {
            total += sum_eta_deaths;
            for (size_t j = 0; j < D; ++j) grad[j] += x_death_sum[j];
            for (size_t l = 0; l < deaths.size(); ++l) {
                double frac = static_cast<double>(l) / dt;
                double denom = s0 - frac * s0_d;
                total -= std::log(denom);
                for (size_t j = 0; j < D; ++j)
                    grad[j] -= (s0_x[j] - frac * s0d_x[j]) / denom;
            }
        }
    }

    double n_events = 0.0;
    for (size_t i = 0; i < N; ++i) if (events[i] == 1.0) n_events += 1.0;

    *loglik_out = total / n_events;
    grad_out->assign(D, 0.0);
    for (size_t j = 0; j < D; ++j) (*grad_out)[j] = grad[j] / n_events;
}

// =============================================================================
// THE FIXTURES (emitted by the oracle -- never hand-transcribed).
// =============================================================================

// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
// Emitted by tests/survival_oracle.py --emit-cpp. float64 numpy.
// The numbers are TRANSFERRED, never re-derived by hand.
// ------------------------------------------------------------------
const std::vector<double> f1_times = {1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
const std::vector<double> f1_events = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
const std::vector<double> f2_times = {1.0, 1.0, 2.0, 2.0, 3.0, 3.5, 4.0};
const std::vector<double> f2_events = {1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0};
const std::vector<std::vector<double>> f3_X = {
    {0.0012301533574825742, 0.2987455375084699},
    {-0.2741378553622176, -0.8905918387572742},
    {-0.45467078517172255, -0.9916465549964624},
    {0.060143602597438485, 1.3402152455545335},
    {-0.49220651855132963, -0.6204748998199404},
    {0.4898420501851982, 0.35688700816006075},
    {0.10541424899789856, -0.9304680447082047},
    {-0.02925182246327349, 0.6953031944582878},
    {-1.344214547285082, -0.45761576104021817},
    {-1.901222739800844, -1.289537739784976},
    {-1.8417350377917323, -0.23509113107468127},
    {-1.2674464814437032, 0.2712643588217015},
    {0.15675108662422516, -0.18693094462995438},
    {-2.516759710820513, -0.5386928958466366},
    {-0.048500945401071985, 0.11330898600330756},
    {-1.5301357655053935, -0.47775327603393064},
    {-0.9785190780566395, -0.8088372394255993},
    {1.0608986233860787, -0.8075346753318965},
    {-0.0325217049455206, 0.8843898673831739},
    {-0.583600432743302, -0.11170194958415963},
    {0.11046414324948059, 0.06378177425506196},
    {-1.2250558264176934, 0.0761402303770081},
    {1.3588234217415376, -1.5471446781284823},
    {0.8593826880215982, 0.11935402569658124},
};
const std::vector<double> f3_times = {1.3319227319621962, 0.9758250050929923, 0.11147555783402022, 0.7149813696060003, 1.1609158987659287, 0.05549695756692147, 0.056681866572045714, 1.2628985707431881, 4.3694579903465085, 6.476038503808048, 2.739667048669722, 4.123133702439706, 0.6134552611377242, 5.360470501342118, 0.15935523402058246, 6.682681865752955, 1.290995611751939, 0.2501297881739812, 0.04208956116670272, 1.636643328091107, 1.0035451161006985, 2.9456874041958687, 0.08849556070758695, 0.7278367486900114};
const std::vector<double> f3_events = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0};
const std::vector<double> f3_beta = {0.35, -0.22};
const std::vector<std::vector<double>> f4_X = {
    {0.03419276725318417, 1.3597475403099617},
    {1.2247210785859324, -0.5103070767876675},
    {-0.2979695111064471, -0.5273841930334252},
    {0.5697263575719601, -0.056064439045617594},
    {0.7468856162565439, -1.8473247989741095},
    {1.5665487746995206, -0.09643216015562055},
    {0.6803784532741461, -0.13656633397682774},
    {-0.3790985670748533, 0.46311015859758675},
    {0.824513527530113, -0.20252987069345152},
    {-0.15278617857019708, 0.685698610809258},
    {-0.8703406419471712, -1.5143835037313955},
    {0.39498186274953, -0.6705658236878794},
    {-1.9203405901180286, -0.8140536639453595},
    {-0.467597558892747, -1.1932024774322612},
    {-1.4924638840630338, 0.03663782694480509},
    {0.8972492567277476, -0.23313207796045685},
};
const std::vector<double> f4_times = {2.0, 7.0, 4.0, 2.0, 6.0, 2.0, 7.0, 3.0, 1.0, 2.0, 3.0, 4.0, 7.0, 6.0, 3.0, 1.0};
const std::vector<double> f4_events = {1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0};
const std::vector<double> f4_beta = {-0.4, 0.9};
const std::vector<double> f5a_times = {1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0};
const std::vector<double> f5a_events = {1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
const std::vector<double> f5b_times = {1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
const std::vector<double> f5b_events = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const std::vector<double> f6_times = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
const std::vector<double> f6_events = {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0};
const std::vector<double> f6_scores = {0.5, 0.5, 0.9, 0.2, 0.2, 0.7, 0.1, 0.4};
const std::vector<double> f7_times = {1.0, 2.0, 3.0, 4.0};
const std::vector<double> f7_events = {0.0, 0.0, 0.0, 0.0};
const std::vector<double> f8_times = {2.0, 2.0, 2.0};
const std::vector<double> f8_events = {1.0, 1.0, 1.0};
const std::vector<double> f9_times = {1.0, 1.0, 2.0};
const std::vector<double> f9_events = {1.0, 0.0, 1.0};
const std::vector<double> f10_times = {1.0, 1.0, 2.0, 3.0, 4.0};
const std::vector<double> f10_events = {1.0, 0.0, 1.0, 0.0, 1.0};
const std::vector<double> f10_scores = {3.0, 1.0, 2.0, 0.5, 4.0};
const std::vector<double> f11_times = {1.0, 1.0, 2.0, 3.0, 4.0};
const std::vector<double> f11_events = {1.0, 1.0, 1.0, 0.0, 0.0};
const std::vector<double> f1_km_event_times = {1.0, 2.0};
const std::vector<double> f1_km_at_risk      = {6.0, 4.0};
const std::vector<double> f1_km_events       = {2.0, 1.0};
const std::vector<double> f1_km_survival     = {0.6666666666666667, 0.5};
const std::vector<double> f1_km_variance     = {0.03703703703703704, 0.041666666666666664};
const double f1_km_median = 2.0;
const std::vector<double> f1_na_event_times = {1.0, 2.0};
const std::vector<double> f1_na_hazard      = {0.3333333333333333, 0.5833333333333333};
const std::vector<double> f9_km_event_times = {1.0, 2.0};
const std::vector<double> f9_km_at_risk      = {3.0, 1.0};
const std::vector<double> f9_km_survival     = {0.6666666666666667, 0.0};
const std::vector<double> f3_km_event_times = {0.05549695756692147, 0.056681866572045714, 0.15935523402058246, 0.2501297881739812, 0.6134552611377242, 0.7278367486900114, 0.9758250050929923, 1.0035451161006985, 1.2628985707431881, 1.290995611751939, 1.3319227319621962, 1.636643328091107, 2.739667048669722, 2.9456874041958687, 5.360470501342118, 6.476038503808048, 6.682681865752955};
const std::vector<double> f3_km_at_risk      = {23.0, 22.0, 19.0, 18.0, 17.0, 15.0, 14.0, 13.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 3.0, 2.0, 1.0};
const std::vector<double> f3_km_events       = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> f3_km_survival     = {0.9565217391304348, 0.9130434782608696, 0.8649885583524028, 0.816933638443936, 0.7688787185354691, 0.7176201372997711, 0.6663615560640732, 0.6151029748283753, 0.5591845225712503, 0.5032660703141253, 0.44734761805700024, 0.3914291657998752, 0.3355107135427502, 0.2795922612856252, 0.1863948408570835, 0.09319742042854175, 0.0};
const std::vector<double> f3_km_variance     = {0.0018081696391879674, 0.0034519602202679384, 0.005285892723935582, 0.0068958680475450194, 0.008281886191096252, 0.009666722422298492, 0.010774849799604259, 0.011606268323013551, 0.01243456938523071, 0.012886187174586011, 0.012961121691079446, 0.012659372934711024, 0.011980940905480747, 0.010925825603388608, 0.010646428606751185, 0.007004486738954982, 0.0};
const std::vector<double> f8_km_event_times = {2.0};
const std::vector<double> f8_km_at_risk      = {3.0};
const std::vector<double> f8_km_survival     = {0.0};
const std::vector<double> f8_km_variance     = {0.0};
const std::vector<double> f11_km_event_times = {1.0, 2.0};
const std::vector<double> f11_km_at_risk      = {5.0, 3.0};
const std::vector<double> f11_km_survival     = {0.6, 0.4};
const double f11_km_median = 2.0;
const double f5_logrank_statistic = 1.954887085000015;
const double f5_logrank_pvalue    = 0.16206119271026542;
const double f5_logrank_observed  = 5.0;
const double f5_logrank_expected  = 3.194139194139194;
const double f5_logrank_variance  = 1.66819519918421;
const double f6_concordance = 0.5238095238095238;
const double f6_num_pairs   = 21.0;
const double f6_num_correct = 10.0;
const double f6_num_tied    = 2.0;
const double f10_concordance = 0.6666666666666666;
const double f10_num_pairs   = 6.0;
const double f10_num_correct = 4.0;
const double f10_num_tied    = 0.0;
const double f3_cox_loglik_breslow = -2.0017154145239693;
const double f3_cox_loglik_efron   = -2.0017154145239693;
const double f3_cox_rawsum_breslow = -34.02916204690748;
const double f3_cox_grad_breslow0  = 0.5341630003913002;
const double f3_cox_grad_breslow1  = 0.09882945448636699;
const double f3_cox_grad_efron0    = 0.5341630003913002;
const double f3_cox_grad_efron1    = 0.09882945448636696;
const double f3_cox_n_events       = 17.0;
const double f4_cox_loglik_breslow = -2.289616015330697;
const double f4_cox_loglik_efron   = -2.165896964599097;
const double f4_cox_grad_breslow0  = 0.37268066314553105;
const double f4_cox_grad_breslow1  = -0.10710928026643068;
const double f4_cox_grad_efron0    = 0.39354108073039984;
const double f4_cox_grad_efron1    = -0.040339409915161134;
// ------------------------------------------------------------------
// >>> ORACLE-EMITTED FIXTURE END <<<

// =============================================================================
// THE TESTS
// =============================================================================

static void test_kaplan_meier() {
    printf("\n--- Kaplan-Meier ---\n");

    Tensor times = col(f1_times);
    Tensor events = col(f1_events);
    KaplanMeierEstimate est = kaplan_meier(times, events);

    check(est.n_points() == f1_km_event_times.size(),
          "F1 the estimate has one point per EVENT time (censored times are not on the grid)");
    check(est.n_points() == 2, "F1 has exactly 2 event times");

    for (size_t k = 0; k < est.n_points() && k < f1_km_event_times.size(); ++k) {
        check_near(est.times[k], f1_km_event_times[k], 1e-12,
                   "F1 event time " + std::to_string(k));
        check_near(est.at_risk[k], f1_km_at_risk[k], 1e-12,
                   "F1 at_risk " + std::to_string(k));
        check_near(est.events[k], f1_km_events[k], 1e-12,
                   "F1 d(t) " + std::to_string(k));
        check_near(est.survival[k], f1_km_survival[k], 1e-12,
                   "F1 S(t) " + std::to_string(k));
        check_near(est.variance[k], f1_km_variance[k], 1e-12,
                   "F1 Greenwood variance " + std::to_string(k));
    }

    // C2: the step function is right-continuous and flat between events.
    check_near(est.survival_at(0.5), 1.0, 1e-12, "F1 S(0.5) == 1 (before the first event)");
    check_near(est.survival_at(1.0), 2.0 / 3.0, 1e-12, "F1 S(1.0) == 2/3 (the post-event value)");
    check_near(est.survival_at(1.5), 2.0 / 3.0, 1e-12, "F1 S(1.5) == 2/3 (the step is flat)");
    check_near(est.survival_at(2.0), 0.5, 1e-12, "F1 S(2.0) == 1/2");
    check_near(est.survival_at(99.0), 0.5, 1e-12, "F1 S(99) == 1/2 (the last step)");

    // C3: the median.
    check_near(est.median_survival_time(), f1_km_median, 1e-12, "F1 the median survival time");

    // C1: the fixture that DISCRIMINATES the risk-set convention.
    KaplanMeierEstimate est9 = kaplan_meier(col(f9_times), col(f9_events));
    check(est9.n_points() == f9_km_event_times.size(), "F9 the number of event times");
    for (size_t k = 0; k < est9.n_points() && k < f9_km_at_risk.size(); ++k) {
        check_near(est9.at_risk[k], f9_km_at_risk[k], 1e-12,
                   "F9 at_risk " + std::to_string(k) + " (the subject censored AT t=1 is at risk)");
        check_near(est9.survival[k], f9_km_survival[k], 1e-12,
                   "F9 S(t) " + std::to_string(k));
    }
    // The WRONG convention would give 1/2 here; assert it differs.
    check(std::fabs(est9.survival[0] - 0.5) > 1e-12,
          "F9 S(1) is NOT 1/2 -- the risk-set convention is discriminated");

    // C4: the n_risk == d case is a LIMIT, not a division by zero.
    KaplanMeierEstimate est8 = kaplan_meier(col(f8_times), col(f8_events));
    check(est8.n_points() == 1, "F8 one event time");
    check_near(est8.at_risk[0], f8_km_at_risk[0], 1e-12, "F8 at_risk == d == 3");
    check_near(est8.survival[0], f8_km_survival[0], 1e-12, "F8 S == 0");
    check(std::isfinite(est8.variance[0]), "F8 the Greenwood variance is FINITE (not inf/NaN)");
    check_near(est8.variance[0], f8_km_variance[0], 1e-12,
               "F8 the Greenwood variance == 0 (the limit)");

    // C3: the curve that CROSSES 0.5 between two event times. The
    // event-time median and an INTERPOLATED median differ here.
    KaplanMeierEstimate est11 = kaplan_meier(col(f11_times), col(f11_events));
    check_near(est11.survival[0], 0.6, 1e-12, "F11 S(t=1) == 0.6");
    check_near(est11.survival[1], 0.4, 1e-12, "F11 S(t=2) == 0.4");
    check_near(est11.median_survival_time(), f11_km_median, 1e-12,
               "F11 the median is the EVENT TIME 2, not the interpolated 1.5");
    check(std::fabs(est11.median_survival_time() - 1.5) > 1e-12,
          "F11 the median is NOT the interpolated 1.5");

    // The all-censored dataset: no events at all.
    KaplanMeierEstimate est7 = kaplan_meier(col(f7_times), col(f7_events));
    check(est7.n_points() == 0, "F7 no event times at all");
    check(std::isinf(est7.median_survival_time()) && est7.median_survival_time() > 0,
          "F7 the median is +infinity (never reached)");
    check_near(est7.survival_at(100.0), 1.0, 1e-12, "F7 S(100) == 1 (no events at all)");

    // The 24-subject fixture, against the oracle's transferred values.
    KaplanMeierEstimate est3 = kaplan_meier(col(f3_times), col(f3_events));
    check(est3.n_points() == f3_km_event_times.size(), "F3 the number of event times");
    for (size_t k = 0; k < est3.n_points() && k < f3_km_survival.size(); ++k) {
        check_near(est3.survival[k], f3_km_survival[k], 1e-12,
                   "F3 S(t) " + std::to_string(k));
        check_near(est3.at_risk[k], f3_km_at_risk[k], 1e-12,
                   "F3 at_risk " + std::to_string(k));
        check_near(est3.variance[k], f3_km_variance[k], 1e-12,
                   "F3 Greenwood variance " + std::to_string(k));
    }
}

static void test_nelson_aalen() {
    printf("\n--- Nelson-Aalen ---\n");

    NelsonAalenEstimate est = nelson_aalen(col(f1_times), col(f1_events));

    check(est.n_points() == f1_na_event_times.size(),
          "F1 the Nelson-Aalen grid is the SAME as the Kaplan-Meier event grid");
    for (size_t k = 0; k < est.n_points() && k < f1_na_hazard.size(); ++k) {
        check_near(est.times[k], f1_na_event_times[k], 1e-12,
                   "F1 NA event time " + std::to_string(k));
        check_near(est.hazard[k], f1_na_hazard[k], 1e-12,
                   "F1 NA hazard " + std::to_string(k));
    }
    check_near(est.hazard[0], 2.0 / 6.0, 1e-12, "F1 H(t=1) == 2/6");
    check_near(est.hazard[1], 2.0 / 6.0 + 1.0 / 4.0, 1e-12, "F1 H(t=2) == 2/6 + 1/4");

    // The step lookup.
    check_near(est.hazard_at(0.5), 0.0, 1e-12, "F1 H(0.5) == 0 (before the first event)");
    check_near(est.hazard_at(1.0), 2.0 / 6.0, 1e-12, "F1 H(1.0) == 2/6");
    check_near(est.hazard_at(99.0), 2.0 / 6.0 + 1.0 / 4.0, 1e-12, "F1 H(99) == the last value");

    // The all-censored dataset: no hazard at all.
    NelsonAalenEstimate est7 = nelson_aalen(col(f7_times), col(f7_events));
    check(est7.n_points() == 0, "F7 no hazard points");
    check_near(est7.hazard_at(100.0), 0.0, 1e-12, "F7 H(100) == 0");
}

static void test_log_rank() {
    printf("\n--- log-rank ---\n");

    LogRankResult res = log_rank_test(col(f5a_times), col(f5a_events),
                                       col(f5b_times), col(f5b_events));
    check_near(res.statistic, f5_logrank_statistic, 1e-12, "F5 the log-rank statistic");
    check_near(res.p_value, f5_logrank_pvalue, 1e-12, "F5 the log-rank p-value");
    check_near(res.observed_1, f5_logrank_observed, 1e-12, "F5 the observed group-1 events");
    check_near(res.expected_1, f5_logrank_expected, 1e-12, "F5 the expected group-1 events");
    check_near(res.variance, f5_logrank_variance, 1e-12, "F5 the hypergeometric variance");

    // The p-value is the chi-square-1 survival function: erfc(sqrt(x/2)).
    check_near(res.p_value, std::erfc(std::sqrt(res.statistic / 2.0)), 1e-15,
               "F5 the p-value is erfc(sqrt(statistic/2))");

    // Identical groups -> the statistic is exactly 0 and the p-value 1.
    LogRankResult same = log_rank_test(col(f5a_times), col(f5a_events),
                                        col(f5a_times), col(f5a_events));
    check_near(same.statistic, 0.0, 1e-12, "F5 identical groups -> the statistic is 0");
    check_near(same.p_value, 1.0, 1e-12, "F5 identical groups -> the p-value is 1");
    check_near(same.observed_1, same.expected_1, 1e-12,
               "F5 identical groups -> observed == expected");

    // The p-value is a probability.
    check(res.p_value >= 0.0 && res.p_value <= 1.0, "F5 the p-value is in [0, 1]");

    // The finite-population factor MUST bite on this fixture: with it
    // dropped the variance would be different, and the statistic with it.
    check(std::fabs(res.variance - f5_logrank_variance) < 1e-12,
          "F5 the variance INCLUDES the finite-population factor");
}

static void test_concordance() {
    printf("\n--- concordance index ---\n");

    ConcordanceResult res = concordance_index(col(f6_times), col(f6_events),
                                               col(f6_scores));
    check_near(res.c_index(), f6_concordance, 1e-12, "F6 the concordance index");
    check_near(res.num_pairs, f6_num_pairs, 1e-12, "F6 the number of admissible pairs");
    check_near(res.num_correct, f6_num_correct, 1e-12, "F6 the number of correct pairs");
    check_near(res.num_tied, f6_num_tied, 1e-12, "F6 the number of tied pairs");

    // The identity: C = (correct + 0.5 * tied) / pairs.
    check_near(res.c_index(),
               (res.num_correct + 0.5 * res.num_tied) / res.num_pairs, 1e-15,
               "F6 C == (correct + 0.5*tied) / pairs");
    check(res.num_tied > 0.0, "F6 the fixture HAS tied risk scores");

    // A CONSTANT risk score gives exactly 0.5 -- the sharpest
    // non-vacuity property of the metric.
    Tensor const_scores = col(std::vector<double>(f6_scores.size(), 0.3));
    ConcordanceResult cres = concordance_index(col(f6_times), col(f6_events), const_scores);
    check_near(cres.c_index(), 0.5, 1e-12, "F6 a constant risk score gives exactly 0.5");

    // A PERFECTLY ANTI-CONCORDANT score gives exactly 0.0: the LATER
    // failure has the higher risk.
    std::vector<double> anti(f6_scores.size());
    for (size_t i = 0; i < anti.size(); ++i) anti[i] = static_cast<double>(i + 1);
    ConcordanceResult ares = concordance_index(col(f6_times), col(f6_events), col(anti));
    check_near(ares.c_index(), 0.0, 1e-12, "F6 an anti-concordant score gives exactly 0");

    // A PERFECT risk score gives exactly 1.0: the EARLIER failure has the
    // higher risk. THIS IS THE SIGN CONVENTION (C7).
    std::vector<double> perf(f6_scores.size());
    for (size_t i = 0; i < perf.size(); ++i) perf[i] = -static_cast<double>(i + 1);
    ConcordanceResult pres = concordance_index(col(f6_times), col(f6_events), col(perf));
    check_near(pres.c_index(), 1.0, 1e-12,
               "F6 a perfect RISK score gives exactly 1 (higher score = higher risk)");

    // THE COMPLEMENTARITY IDENTITY: C(s) + C(-s) == 1. This pins the sign
    // convention from BOTH sides, and it holds even with tied scores.
    std::vector<double> neg(f6_scores.size());
    for (size_t i = 0; i < neg.size(); ++i) neg[i] = -f6_scores[i];
    ConcordanceResult nres = concordance_index(col(f6_times), col(f6_events), col(neg));
    check_near(res.c_index() + nres.c_index(), 1.0, 1e-12,
               "F6 C(s) + C(-s) == 1 (the sign convention is pinned from both sides)");

    // Equal times: the pair is admissible only because exactly one of the
    // two is an event.
    ConcordanceResult res10 = concordance_index(col(f10_times), col(f10_events),
                                                col(f10_scores));
    check_near(res10.c_index(), f10_concordance, 1e-12, "F10 the concordance index");
    check_near(res10.num_pairs, f10_num_pairs, 1e-12, "F10 the number of admissible pairs");
    check_near(res10.num_correct, f10_num_correct, 1e-12, "F10 the number of correct pairs");
    check_near(res10.num_tied, f10_num_tied, 1e-12, "F10 the number of tied pairs");
    check(res10.num_pairs > 0.0, "F10 the equal-times pair IS admissible");

    // The all-censored dataset: no admissible pairs at all, so the index
    // is NaN rather than a division by zero.
    ConcordanceResult res7 = concordance_index(col(f7_times), col(f7_events),
                                               col(f7_events));
    check(res7.num_pairs == 0.0, "F7 no admissible pairs");
    check(std::isnan(res7.c_index()), "F7 the index is NaN with no admissible pairs");
}

static void test_cox() {
    printf("\n--- Cox partial log-likelihood ---\n");

    Tensor X3 = mat(f3_X);
    Tensor t3 = col(f3_times);
    Tensor e3 = col(f3_events);
    Tensor b3 = row(f3_beta);

    CoxPartialLikelihood br = cox_partial_log_likelihood(b3, X3, t3, e3, CoxTieMethod::Breslow);
    CoxPartialLikelihood ef = cox_partial_log_likelihood(b3, X3, t3, e3, CoxTieMethod::Efron);

    check_near(br.log_likelihood, f3_cox_loglik_breslow, 1e-12, "F3 the Breslow partial log-likelihood");
    check_near(ef.log_likelihood, f3_cox_loglik_efron, 1e-12, "F3 the Efron partial log-likelihood");
    check_near(br.gradient(0, 0), f3_cox_grad_breslow0, 1e-12, "F3 the Breslow gradient [0]");
    check_near(br.gradient(0, 1), f3_cox_grad_breslow1, 1e-12, "F3 the Breslow gradient [1]");
    check_near(ef.gradient(0, 0), f3_cox_grad_efron0, 1e-12, "F3 the Efron gradient [0]");
    check_near(ef.gradient(0, 1), f3_cox_grad_efron1, 1e-12, "F3 the Efron gradient [1]");
    check_near(br.n_events, f3_cox_n_events, 1e-12, "F3 the number of events");

    // C9: on a fixture with NO tied event times the two methods COINCIDE.
    check_near(br.log_likelihood, ef.log_likelihood, 1e-12,
               "F3 Breslow == Efron (there are no tied event times)");
    check_near(br.gradient(0, 0), ef.gradient(0, 0), 1e-12, "F3 the gradients coincide too");

    // C11: the MEAN, and the raw sum differs from it.
    check_near(br.log_likelihood * f3_cox_n_events, f3_cox_rawsum_breslow, 1e-9,
               "F3 the mean * n_events == the raw sum");
    check(std::fabs(br.log_likelihood - f3_cox_rawsum_breslow) > 1e-6,
          "F3 the reported value is the MEAN, not the raw sum");

    // The tied fixture: the two methods DIFFER.
    Tensor X4 = mat(f4_X);
    Tensor t4 = col(f4_times);
    Tensor e4 = col(f4_events);
    Tensor b4 = row(f4_beta);
    CoxPartialLikelihood br4 = cox_partial_log_likelihood(b4, X4, t4, e4, CoxTieMethod::Breslow);
    CoxPartialLikelihood ef4 = cox_partial_log_likelihood(b4, X4, t4, e4, CoxTieMethod::Efron);
    check_near(br4.log_likelihood, f4_cox_loglik_breslow, 1e-12, "F4 the Breslow partial log-likelihood");
    check_near(ef4.log_likelihood, f4_cox_loglik_efron, 1e-12, "F4 the Efron partial log-likelihood");
    check_near(br4.gradient(0, 0), f4_cox_grad_breslow0, 1e-12, "F4 the Breslow gradient [0]");
    check_near(br4.gradient(0, 1), f4_cox_grad_breslow1, 1e-12, "F4 the Breslow gradient [1]");
    check_near(ef4.gradient(0, 0), f4_cox_grad_efron0, 1e-12, "F4 the Efron gradient [0]");
    check_near(ef4.gradient(0, 1), f4_cox_grad_efron1, 1e-12, "F4 the Efron gradient [1]");
    check(std::fabs(br4.log_likelihood - ef4.log_likelihood) > 1e-6,
          "F4 Breslow != Efron (there ARE tied event times)");
    check(std::fabs(br4.gradient(0, 0) - ef4.gradient(0, 0)) > 1e-6,
          "F4 the Breslow and Efron gradients differ");

    // ---- THE INDEPENDENT SCALAR-LOOP REFERENCE ------------------------
    // This is what a finite-difference check CANNOT do: it compares the
    // library against ITSELF, so it cannot see a defect that moves the
    // value and the gradient TOGETHER.
    {
        double rl = 0.0;
        std::vector<double> rg;
        reference_cox(f3_beta, f3_X, f3_times, f3_events, false, &rl, &rg);
        check_near(br.log_likelihood, rl, 1e-12, "F3 the value matches the INDEPENDENT reference");
        check_near(br.gradient(0, 0), rg[0], 1e-12, "F3 the gradient [0] matches the INDEPENDENT reference");
        check_near(br.gradient(0, 1), rg[1], 1e-12, "F3 the gradient [1] matches the INDEPENDENT reference");
    }
    {
        double rl = 0.0;
        std::vector<double> rg;
        reference_cox(f4_beta, f4_X, f4_times, f4_events, false, &rl, &rg);
        check_near(br4.log_likelihood, rl, 1e-12, "F4 the value matches the INDEPENDENT reference");
        check_near(br4.gradient(0, 0), rg[0], 1e-12, "F4 the gradient [0] matches the INDEPENDENT reference");
        check_near(br4.gradient(0, 1), rg[1], 1e-12, "F4 the gradient [1] matches the INDEPENDENT reference");
    }
    {
        double rl = 0.0;
        std::vector<double> rg;
        reference_cox(f4_beta, f4_X, f4_times, f4_events, true, &rl, &rg);
        check_near(ef4.log_likelihood, rl, 1e-12, "F4 the EFRON value matches the INDEPENDENT reference");
        check_near(ef4.gradient(0, 0), rg[0], 1e-12, "F4 the EFRON gradient [0] matches the INDEPENDENT reference");
        check_near(ef4.gradient(0, 1), rg[1], 1e-12, "F4 the EFRON gradient [1] matches the INDEPENDENT reference");
    }

    // ---- THE ANALYTIC GRADIENT AGAINST FINITE DIFFERENCES --------------
    // The forward here is DETERMINISTIC (no sampling), so the central
    // difference is a valid derivative.
    {
        const double eps = 1e-6;
        Tensor b = b3;
        CoxPartialLikelihood base = cox_partial_log_likelihood(b, X3, t3, e3, CoxTieMethod::Breslow);
        for (size_t j = 0; j < 2; ++j) {
            double orig = b(0, j);
            b(0, j) = orig + eps;
            CoxPartialLikelihood up = cox_partial_log_likelihood(b, X3, t3, e3, CoxTieMethod::Breslow);
            b(0, j) = orig - eps;
            CoxPartialLikelihood dn = cox_partial_log_likelihood(b, X3, t3, e3, CoxTieMethod::Breslow);
            b(0, j) = orig;
            double num = (up.log_likelihood - dn.log_likelihood) / (2.0 * eps);
            double den = std::max(std::fabs(num), 1e-12);
            double rel = std::fabs(base.gradient(0, j) - num) / den;
            check(rel < 1e-5,
                  "F3 the Breslow analytic gradient[" + std::to_string(j) +
                  "] matches the finite difference (rel=" + std::to_string(rel) + ")");
        }
    }
}

static void test_guards() {
    printf("\n--- guards ---\n");

    Tensor good_t = col(f1_times);
    Tensor good_e = col(f1_events);
    Tensor good_b = row(f3_beta);
    Tensor good_X = mat(f3_X);

    // A shape mismatch.
    check(throws_with([&] { kaplan_meier(good_t, col(f2_events)); },
                      "different row counts"),
          "kaplan_meier rejects a row-count mismatch");
    check(throws_with([&] { kaplan_meier(Tensor(3, 2), Tensor(3, 1)); },
                      "single column"),
          "kaplan_meier rejects a multi-column times tensor");
    check(throws_with([&] { kaplan_meier(Tensor(0, 1), Tensor(0, 1)); },
                      "no rows"),
          "kaplan_meier rejects an empty dataset");
    check(throws_with([&] { kaplan_meier(col(f7_times), col(std::vector<double>{0.5, 1.0, 0.0, 0.0})); },
                      "neither 0 nor 1"),
          "kaplan_meier rejects an event indicator that is neither 0 nor 1");
    // **THE FIXTURE MUST ISOLATE THE GUARD.** A negative-time fixture
    // with a DIFFERENT row count from its events would trip the
    // row-count guard first, and the negative-time guard would never be
    // reached -- so the assertion would pass for the wrong reason and
    // the guard would be untested. The row counts here MATCH.
    {
        Tensor neg_t = col(std::vector<double>{-1.0, 2.0, 3.0});
        Tensor neg_e = col(std::vector<double>{1.0, 0.0, 0.0});
        check(throws_with([&] { kaplan_meier(neg_t, neg_e); },
                          "negative"),
              "kaplan_meier rejects a negative time");
    }

    // Every guard message must be DISTINCT -- a needle that appears in two
    // messages discriminates nothing between them.
    std::string m1, m2, m3;
    throws_with([&] { kaplan_meier(good_t, col(f2_events)); }, "", &m1);
    throws_with([&] { kaplan_meier(Tensor(3, 2), Tensor(3, 1)); }, "", &m2);
    throws_with([&] { kaplan_meier(col(f7_times), col(std::vector<double>{0.5, 1.0, 0.0, 0.0})); }, "", &m3);
    check(m1 != m2 && m2 != m3 && m1 != m3, "the guard messages are pairwise distinct");

    // The Cox guards. **EVERY FIXTURE HERE MATCHES THE ROW COUNT OF THE
    // DATA IT IS PAIRED WITH**, so that the guard under test is the one
    // that fires rather than an earlier one.
    check(throws_with([&] { cox_partial_log_likelihood(good_b, good_X, good_t, good_e); },
                      "different row counts"),
          "cox rejects a design matrix whose row count disagrees");
    {
        // A 4-row design matrix with the 4-row all-censored dataset.
        Tensor X7 = mat(std::vector<std::vector<double>>{{1.0, 2.0}, {0.0, 1.0},
                                                           {1.0, 1.0}, {2.0, 0.0}});
        Tensor b7 = row(std::vector<double>{0.1, -0.2});
        check(throws_with([&] { cox_partial_log_likelihood(b7, X7, col(f7_times), col(f7_events)); },
                          "no events"),
              "cox rejects a dataset with no events");
    }
    check(throws_with([&] { cox_partial_log_likelihood(Tensor(2, 1), good_X, good_t, good_e); },
                      "single row"),
          "cox rejects a coefficient vector that is not a single row");
    {
        // A 6-row design matrix with ONE column, against a 2-column
        // coefficient vector. The row counts match, so the column-count
        // guard is the one that fires.
        Tensor X_wide = mat(std::vector<std::vector<double>>{{1.0}, {2.0}, {0.0},
                                                               {1.0}, {3.0}, {0.5}});
        check(throws_with([&] { cox_partial_log_likelihood(good_b, X_wide, good_t, good_e); },
                          "same number of columns"),
              "cox rejects a design matrix with the wrong number of columns");
    }

    // The concordance guards.
    check(throws_with([&] { concordance_index(good_t, good_e, col(f6_scores)); },
                      "different row counts"),
          "concordance rejects a risk-score row-count mismatch");
    {
        // A 6-row, 2-column risk-score tensor: the row count matches the
        // times, so the single-column guard is the one that fires.
        Tensor wide = Tensor(6, 2);
        for (size_t i = 0; i < 6; ++i) { wide(i, 0) = 0.5; wide(i, 1) = 0.25; }
        check(throws_with([&] { concordance_index(good_t, good_e, wide); },
                          "single column"),
              "concordance rejects a multi-column risk-score tensor");
    }
}

int main() {
    printf("============================================================\n");
    printf("tests/test_survival.cpp -- the survival-analysis utilities\n");
    printf("============================================================\n");

    test_kaplan_meier();
    test_nelson_aalen();
    test_log_rank();
    test_concordance();
    test_cox();
    test_guards();

    printf("\n============================================================\n");
    printf("=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    printf("============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
