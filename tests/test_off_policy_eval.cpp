// =============================================================================
// tests/test_off_policy_eval.cpp -- the off-policy evaluation estimators.
//
// THE SPECIFICATION IS tests/off_policy_eval_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed and was green on its
// first execution (38/38). **A GREEN ORACLE ON ITS FIRST RUN PROVES
// NOTHING ABOUT ITS OWN SENSITIVITY**, so its discriminating power was
// MEASURED instead of narrated: tests/off_policy_eval_mutate_oracle.py
// breaks the oracle in seven ways and requires that it fail each time
// (7/7 caught).
//
// =============================================================================
// WHY THIS FEATURE EXISTS (measured 2026-10-30)
// =============================================================================
//
//     grep -rinE '\bsnr\b|off.?policy|behavior.?polic|importance.?sampl' include/ tests/
//     -> 0 hits
//
// The repo has ~250 layers and 60+ utility files and NO off-policy
// machinery. The nearest shipped things are all DIFFERENT objects:
//
//   * utils/mmd_loss.h    -- a KERNEL two-sample statistic. No importance
//                            weights, no policy, no value.
//   * utils/conformal.h   -- calibrates a prediction that was ALREADY MADE.
//   * utils/cross_validation.h -- resamples the SAME data. Off-policy
//                            evaluation is the opposite: it combines data
//                            collected under ONE policy with a DIFFERENT
//                            policy's action probabilities.
//   * utils/sinkhorn.h    -- an entropic transport PLAN, not a value.
//
// There is no `reward` concept anywhere in include/nn either, so the module
// is MODEL-FREE OVER LOGGED DATA: not a layer, not an environment.
//
// =============================================================================
// THE MATHEMATICS, AND THE THREE PLACES IT IS EASY TO GET WRONG
// =============================================================================
//
// Logged data: N samples (s_i, a_i, r_i) with a_i ~ pi_0(.|s_i) (the
// BEHAVIOUR policy). The TARGET policy is pi_1. The importance weight is
//
//     rho_i = pi_1(a_i|s_i) / pi_0(a_i|s_i)
//
// and the estimators are
//
//     IPS  V = (1/N) sum_i rho_i r_i
//     SNIS V = (sum_i rho_i r_i) / (sum_i rho_i)
//     DM   V = (1/N) sum_i Vhat(s_i)
//     DR   V = (1/N) sum_i [ Vhat(s_i) + rho_i (r_i - Rhat(s_i,a_i)) ]
//
// where the reward model gives Vhat(s) = sum_a pi_1(a|s) Rhat(s,a) and the
// per-sample logged-action value Rhat(s_i,a_i).
//
//  1. **THE SNIS DENOMINATOR IS sum(rho), NOT N.** Using N silently returns
//     IPS. The suite pins the exact identity V_SNIS == V_IPS / mean(rho).
//
//  2. **THE `1/N` FACTOR.** Every estimator is a MEAN over the N samples.
//     A SUM is off by exactly the factor N, which is a fingerprint of a
//     normalisation error rather than a structural one.
//
//  3. **THE DR CANCELLATION.** Writing it out, with rho(a) = pi_1/pi_0:
//
//         E_{a~pi_0}[ Vhat(s) + rho(a)(r(a) - Rhat(s,a)) ]
//             = Vhat(s) + sum_a pi_1(a)(E[r|s,a] - Rhat(s,a))
//             = Vhat(s) + V_pi1 - sum_a pi_1(a) Rhat(s,a)
//             = V_pi1                                <- the Vhat TERM CANCELS
//
//     So DR is unbiased when EITHER the propensities are exact (the first
//     Vhat cancels) OR the reward model is exact (the correction term
//     vanishes identically). **THE SUITE CHECKS BOTH ARMS BY EXACT
//     ENUMERATION over a finite action support** -- no Monte Carlo, no
//     tolerance to argue about. An implementation that drops the Vhat term,
//     flips the correction's sign, or pairs the wrong sample's values
//     breaks one of the two arms.
//
// =============================================================================
// WHY THE FIXTURES ARE FIXED AND NOT RANDOM
// =============================================================================
//
// A random fixture would not be checkable: there would be no independent
// value to compare against. So:
//
//   * the ENUMERATION checks use a finite 3-action bandit with rational
//     propensities, so the expectation over the logging distribution is
//     computed EXACTLY by summing over all logged action sequences;
//   * the VALUE checks use a fixed 4-sample table whose expected values are
//     pinned by the oracle AND re-derived by hand in the test file;
//   * the SENSITIVITY checks (scale invariance) deliberately scale the
//     weights, because a fixture that does not vary the scale of the
//     weights cannot tell SNIS from IPS at all.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// ---------------------------------------------------------------------------
// A guarded call.
//
// Every call that CAN throw is wrapped here, so a defect becomes a FAIL with
// a count instead of an abort with nothing. This is the repo's recorded
// defect class: a throwing call outside a try/catch escapes main, the
// binary dies with exit 134, and the suite's verdict count is LOST.
// ---------------------------------------------------------------------------
template <typename F>
static bool call_guarded(F fn, double* out) {
    try {
        *out = fn();
        return true;
    } catch (const std::exception& e) {
        printf("    [THREW] %s\n", e.what());
        return false;
    }
}

// Does the message contain the given substring? Used to assert that a
// SPECIFIC guard fired. **The discriminator is the MESSAGE, not the throw**:
// a test that only asserts "it threw" passes while the guard it means to
// cover is gone.
//
// The message is copied OUT into `what_out`, and the caller must READ IT
// AFTER this function returns -- see `expect_throw` below, which exists
// because of a bug this very file had: `check(threw_with(...), "..." + what)`
// evaluates its arguments in UNSPECIFIED ORDER, so the printed message was
// sometimes the one from the PREVIOUS call. The needles still matched (the
// guards were right); only the text displayed alongside the verdict lagged.
static bool threw_with(const std::function<void()>& fn, const std::string& needle,
                       std::string* what_out) {
    try {
        fn();
        return false;
    } catch (const std::exception& e) {
        *what_out = e.what();
        return what_out->find(needle) != std::string::npos;
    }
}

// Assert that `fn` throws AND that the message contains `needle`, then
// report it. The message is read AFTER the call, so it always belongs to
// THIS call.
static void expect_throw(const std::function<void()>& fn, const std::string& needle,
                         const std::string& name) {
    std::string what;
    const bool threw = threw_with(fn, needle, &what);
    if (threw) {
        check(true, name + ": " + what);
    } else {
        check(false, name + " (expected a throw containing \"" + needle +
                  "\", got: " + (what.empty() ? std::string("<no throw>") : what) +
                  ")");
    }
}

static double rel_err(double got, double want) {
    double denom = std::max(std::abs(got), std::abs(want));
    if (denom < 1e-12) return std::abs(got - want);
    return std::abs(got - want) / denom;
}

// Used to keep the relative-error helper honest: it is exercised on the
// pinned values, so it cannot rot into an unused function (which is what
// produced an -Wunused-function warning in the first draft of this file).
static void check_close(double got, double want, double tol, const std::string& name) {
    check(rel_err(got, want) <= tol, name);
}

static Tensor vec(const std::vector<double>& v) {
    return Tensor(v.size() > 0 ? 1 : 0, v.size() > 0 ? v.size() : 0,
                  v.empty() ? nullptr : v.data());
}

// ===========================================================================
// THE INDEPENDENT REFERENCE IMPLEMENTATIONS.
//
// These share NO code with the library under test: they are the formulas
// written out in plain C++ arithmetic. Every value the suite asserts is
// checked against BOTH the library and one of these, so a bug that lives in
// both places is the only thing that could pass -- and the pinned literals
// from the oracle are a third route.
// ===========================================================================
static double ref_ips(const std::vector<double>& r, const std::vector<double>& w) {
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i) s += w[i] * r[i];
    return s / static_cast<double>(r.size());
}

static double ref_snis(const std::vector<double>& r, const std::vector<double>& w) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < r.size(); ++i) { num += w[i] * r[i]; den += w[i]; }
    return num / den;
}

static double ref_dm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

static double ref_dr(const std::vector<double>& r, const std::vector<double>& w,
                     const std::vector<double>& v, const std::vector<double>& q) {
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i) s += v[i] + w[i] * (r[i] - q[i]);
    return s / static_cast<double>(r.size());
}

static double ref_ess(const std::vector<double>& w) {
    double s = 0.0, s2 = 0.0;
    for (double x : w) { s += x; s2 += x * x; }
    return s * s / s2;
}

// ===========================================================================
// THE EXACT ENUMERATOR.
//
// For a finite action support and a finite number of samples, the
// expectation of an estimator over the LOGGING distribution is computed by
// enumerating every logged action sequence with its EXACT probability.
//
// **THE LOGGING DISTRIBUTION IS pi_0, THE BEHAVIOUR POLICY.** Enumerating
// under pi_1 makes the weighted expectation collapse to something that is
// not V_pi1, so the unbiasedness check fails against a CORRECT
// implementation -- the "the fixture is wrong, not the code" trap.
//
// The enumerator's own total probability mass is asserted to be 1, and a
// control statistic is checked against a hand-computed value FIRST, so a
// broken enumerator cannot make every downstream check meaningless.
// ===========================================================================
struct Bandit {
    std::vector<double> pi0;      // the BEHAVIOUR / logging policy
    std::vector<double> pi1;      // the TARGET policy
    std::vector<double> mu;       // the true (deterministic) reward per action
};

// The number of actions, and a check that every per-action table the
// enumerator indexes has EXACTLY that many entries.
//
// **THIS GUARD EXISTS BECAUSE ADDRESSSANITIZER FOUND A REAL
// HEAP-BUFFER-OVERFLOW IN THIS FILE.** The first draft of section 5 passed a
// two-element `rho` to a three-action bandit, and the enumerator indexed it
// as `rho[seq[t]]` with `seq[t]` in {0,1,2}. **AT -O2 THAT READ WENT PAST
// THE END OF THE VECTOR AND INTO THE ALLOCATOR'S SLACK SPACE, PRODUCING A
// PLAUSIBLE-LOOKING RESULT AND A GREEN SUITE** -- the read is only visible
// under a sanitizer, because the bytes it returned were never used by the
// control estimator (which ignores the weights).
//
// The general rule this instance of: **DERIVE LOOP BOUNDS FROM THE BUFFER,
// NEVER FROM A SIBLING QUANTITY, AND ASSERT THE BOUND.** A per-action table
// indexed by an action index must be CHECKED to have one entry per action.
//
// Returns false and prints a diagnostic if the sizes disagree.
static bool validate_action_tables(const Bandit& b,
                                    const std::vector<double>& rho,
                                    const std::vector<double>* rhat) {
    const size_t A = b.pi0.size();
    bool ok = true;
    if (b.mu.size() != A) {
        printf("    [ENUMERATOR DEFECT] the reward table has %zu entries, "
               "expected %zu\n", b.mu.size(), A);
        ok = false;
    }
    if (rho.size() != A) {
        printf("    [ENUMERATOR DEFECT] the weight table has %zu entries, "
               "expected %zu (one per action)\n", rho.size(), A);
        ok = false;
    }
    if (rhat != nullptr && rhat->size() != A) {
        printf("    [ENUMERATOR DEFECT] the model table has %zu entries, "
               "expected %zu (one per action)\n", rhat->size(), A);
        ok = false;
    }
    return ok;
}

// The exact expectation of `estimator` over the logging distribution.
// `estimator` takes the per-sample rewards, weights, model target values
// and logged action values, and returns the estimator's value.
static double enumerate_expectation(const Bandit& b, int n_samples,
                                    const std::function<double(
                                        const std::vector<double>&,
                                        const std::vector<double>&,
                                        const std::vector<double>&,
                                        const std::vector<double>&)>& estimator,
                                    const std::vector<double>& rho,
                                    double vhat_const) {
    const int A = static_cast<int>(b.pi0.size());
    if (!validate_action_tables(b, rho, nullptr)) return std::nan("");
    double total = 0.0, mass = 0.0;

    // Enumerate every action sequence of length n_samples. **THERE ARE A^n
    // OF THEM, NOT 2^n** -- see the note in `enumerate_expectation_with_model`.
    long long n_sequences = 1;
    for (int t = 0; t < n_samples; ++t) n_sequences *= A;
    std::vector<int> seq(n_samples, 0);
    for (long long idx = 0; idx < n_sequences; ++idx) {
        // Decode idx into the action sequence (base A).
        long long code = idx;
        for (int t = n_samples - 1; t >= 0; --t) {
            seq[t] = static_cast<int>(code % A);
            code /= A;
        }

        double p = 1.0;
        for (int t = 0; t < n_samples; ++t) p *= b.pi0[seq[t]];
        mass += p;
        if (p == 0.0) continue;

        std::vector<double> r(n_samples), w(n_samples), v(n_samples), q(n_samples);
        for (int t = 0; t < n_samples; ++t) {
            r[t] = b.mu[seq[t]];
            w[t] = rho[seq[t]];
            v[t] = vhat_const;
            q[t] = b.mu[seq[t]];   // filled by the caller below if needed
        }
        total += p * estimator(r, w, v, q);
    }

    if (std::abs(mass - 1.0) > 1e-12) {
        printf("    [ENUMERATOR DEFECT] total probability mass is %.17g, not 1\n",
               mass);
        return std::nan("");
    }
    return total;
}

// The same enumeration, but with the logged-action model value taken from a
// caller-supplied table (so the "wrong reward model" arm can be exercised).
static double enumerate_expectation_with_model(
        const Bandit& b, int n_samples,
        const std::function<double(const std::vector<double>&,
                                   const std::vector<double>&,
                                   const std::vector<double>&,
                                   const std::vector<double>&)>& estimator,
        const std::vector<double>& rho,
        double vhat_const, const std::vector<double>& rhat) {
    const int A = static_cast<int>(b.pi0.size());
    if (!validate_action_tables(b, rho, &rhat)) return std::nan("");
    double total = 0.0, mass = 0.0;
    std::vector<int> seq(n_samples, 0);
    // **THE NUMBER OF SEQUENCES IS A^n, NOT 2^n.** The first draft of this
    // file looped `1LL << n_samples`, which enumerates only 8 of the 27
    // sequences of length 3 over 3 actions -- and the enumerator's own
    // probability-mass assertion caught it, reporting a mass of 0.48
    // instead of 1. **THE SELF-CHECK IS WHAT CAUGHT IT**, which is the
    // whole reason it is there.
    long long n_sequences = 1;
    for (int t = 0; t < n_samples; ++t) n_sequences *= A;
    for (long long idx = 0; idx < n_sequences; ++idx) {
        long long code = idx;
        for (int t = n_samples - 1; t >= 0; --t) {
            seq[t] = static_cast<int>(code % A);
            code /= A;
        }
        double p = 1.0;
        for (int t = 0; t < n_samples; ++t) p *= b.pi0[seq[t]];
        mass += p;
        if (p == 0.0) continue;

        std::vector<double> r(n_samples), w(n_samples), v(n_samples), q(n_samples);
        for (int t = 0; t < n_samples; ++t) {
            r[t] = b.mu[seq[t]];
            w[t] = rho[seq[t]];
            v[t] = vhat_const;
            q[t] = rhat[seq[t]];
        }
        total += p * estimator(r, w, v, q);
    }
    if (std::abs(mass - 1.0) > 1e-12) {
        printf("    [ENUMERATOR DEFECT] total probability mass is %.17g, not 1\n",
               mass);
        return std::nan("");
    }
    return total;
}

int main() {
    printf("=== off-policy evaluation estimators ===\n");

    // -----------------------------------------------------------------------
    section("1. THE IMPORTANCE WEIGHT");
    // -----------------------------------------------------------------------
    {
        double w = 0.0;
        bool ok = call_guarded([] { return utils::ope_importance_weight(0.4, 0.2); }, &w);
        check(ok && std::abs(w - 2.0) < 1e-15,
              "1.1 the weight is pi_1 / pi_0");
        check(ok && std::abs(w - 2.0) / 2.0 < 1e-15,
              "1.2 the weight is exact, not approximate");

        // A NON-POSITIVE behaviour probability is a division by zero.
        expect_throw([] { utils::ope_importance_weight(0.4, 0.0); },
                     "behaviour probability",
                     "1.3 a zero behaviour probability is REJECTED (not "
                     "silently inf)");
        expect_throw([] { utils::ope_importance_weight(0.4, -0.1); },
                     "behaviour probability",
                     "1.4 a negative behaviour probability is REJECTED");
        expect_throw([] { utils::ope_importance_weight(-0.2, 0.5); },
                     "target probability",
                     "1.5 a negative target probability is REJECTED");

        // The batch form.
        Tensor tp = vec({0.2, 0.5, 0.3});
        Tensor bp = vec({0.5, 0.3, 0.2});
        Tensor wv;
        double dummy = 0.0;
        ok = call_guarded([&] { wv = utils::ope_importance_weights(tp, bp); return 0.0; }, &dummy);
        check(ok, "1.6 the batch weight form runs");
        if (ok) {
            check(std::abs(wv(0, 0) - 0.4) < 1e-15 &&
                  std::abs(wv(0, 1) - 0.5 / 0.3) < 1e-14 &&
                  std::abs(wv(0, 2) - 1.5) < 1e-15,
                  "1.7 the batch weights are elementwise pi_1 / pi_0");
            // The shape is irrelevant: (1,N) and (N,1) are the same data.
            Tensor tp_col(3, 1);
            tp_col(0, 0) = 0.2; tp_col(1, 0) = 0.5; tp_col(2, 0) = 0.3;
            Tensor bp_col(3, 1);
            bp_col(0, 0) = 0.5; bp_col(1, 0) = 0.3; bp_col(2, 0) = 0.2;
            Tensor wv_col;
            call_guarded([&] { wv_col = utils::ope_importance_weights(tp_col, bp_col); return 0.0; }, &dummy);
            check(std::abs(wv_col(0, 0) - wv(0, 0)) < 1e-15 &&
                  std::abs(wv_col(1, 0) - wv(0, 1)) < 1e-15 &&
                  std::abs(wv_col(2, 0) - wv(0, 2)) < 1e-15,
                  "1.8 a column vector gives the same weights as a row vector "
                  "(the shape is irrelevant, the values are what matter)");
        }
    }

    // -----------------------------------------------------------------------
    section("2. THE FIXED FIXTURE -- THE PINNED VALUES");
    //
    // rewards     = (1.0, -0.5, 2.0, 0.25)
    // target probs= (0.4, 0.1, 0.6, 0.0)
    // behav probs = (0.5, 0.25, 0.2, 0.8)
    // weights     = (0.8, 0.4, 3.0, 0.0)
    // Vhat        = (0.1, -0.2, 0.5, 0.05)
    // Rhat        = (0.3, 0.15, -0.1, 0.4)
    //
    // IPS  = 6.6 / 4   = 1.65
    // SNIS = 6.6 / 4.2 = 1.5714285714285714...
    // DM   = 0.45 / 4  = 0.1125
    // DR   = 7.05 / 4  = 1.7625
    // ESS  = 17.64 / 9.8 = 1.8
    // -----------------------------------------------------------------------
    const std::vector<double> fx_r = {1.0, -0.5, 2.0, 0.25};
    const std::vector<double> fx_w = {0.8, 0.4, 3.0, 0.0};
    const std::vector<double> fx_v = {0.1, -0.2, 0.5, 0.05};
    const std::vector<double> fx_q = {0.3, 0.15, -0.1, 0.4};

    Tensor R = vec(fx_r), W = vec(fx_w), V = vec(fx_v), Q = vec(fx_q);

    {
        double got = 0.0;
        call_guarded([&] { return utils::ope_ips_estimate(R, W); }, &got);
        check_close(got, 1.65, 1e-12, "2.1 IPS equals the pinned 1.65");
        check_close(got, ref_ips(fx_r, fx_w), 1e-12,
                    "2.2 IPS matches the independent reference");

        call_guarded([&] { return utils::ope_self_normalized_ips(R, W); }, &got);
        check_close(got, 1.5714285714285714, 1e-12,
                    "2.3 SNIS equals the pinned 6.6/4.2");
        check_close(got, ref_snis(fx_r, fx_w), 1e-12,
                    "2.4 SNIS matches the independent reference");

        call_guarded([&] { return utils::ope_direct_method_estimate(V); }, &got);
        check_close(got, 0.1125, 1e-12, "2.5 DM equals the pinned 0.1125");
        check_close(got, ref_dm(fx_v), 1e-12,
                    "2.6 DM matches the independent reference");

        call_guarded([&] { return utils::ope_doubly_robust_estimate(R, W, V, Q); }, &got);
        check_close(got, 1.7625, 1e-12, "2.7 DR equals the pinned 1.7625");
        check_close(got, ref_dr(fx_r, fx_w, fx_v, fx_q), 1e-12,
                    "2.8 DR matches the independent reference");

        call_guarded([&] { return utils::ope_effective_sample_size(W); }, &got);
        check_close(got, 1.8, 1e-12, "2.9 ESS equals the pinned 1.8");
        check_close(got, ref_ess(fx_w), 1e-12,
                    "2.10 ESS matches the independent reference");
    }

    // -----------------------------------------------------------------------
    section("3. SNIS AND IPS ARE DIFFERENT FUNCTIONS");
    //
    // SNIS divides by sum(rho); IPS divides by N. A fixture with mean(rho)
    // == 1 cannot tell them apart, so THIS fixture has mean(rho) = 1.05.
    // -----------------------------------------------------------------------
    {
        double ips = 0.0, snis = 0.0;
        call_guarded([&] { return utils::ope_ips_estimate(R, W); }, &ips);
        call_guarded([&] { return utils::ope_self_normalized_ips(R, W); }, &snis);

        check(std::abs(snis - ips) > 1e-6,
              "3.1 SNIS and IPS DIFFER on this fixture, so it discriminates");
        const double mean_w = (0.8 + 0.4 + 3.0 + 0.0) / 4.0;
        check(std::abs(snis - ips / mean_w) < 1e-12,
              "3.2 SNIS == IPS / mean(rho) EXACTLY");
        check(std::abs(mean_w - 1.05) < 1e-12,
              "3.3 the fixture's mean(rho) is 1.05, not 1 (so the two "
              "estimators genuinely differ)");

        // And IPS is a MEAN, not a SUM.
        check(std::abs(ips - 6.6 / 4.0) < 1e-12,
              "3.4 IPS divides by N (the sample count)");
    }

    // -----------------------------------------------------------------------
    section("4. SCALE INVARIANCE -- THE SHARPEST DISCRIMINATOR");
    //
    // SNIS(c*rho) == SNIS(rho) for any c > 0: the scale cancels between
    // numerator and denominator. IPS(c*rho) == c * IPS(rho): it is NOT
    // scale invariant. **A FIXTURE THAT DOES NOT VARY THE SCALE OF THE
    // WEIGHTS CANNOT TELL THESE APART.**
    // -----------------------------------------------------------------------
    {
        double snis0 = 0.0, ips0 = 0.0;
        call_guarded([&] { return utils::ope_self_normalized_ips(R, W); }, &snis0);
        call_guarded([&] { return utils::ope_ips_estimate(R, W); }, &ips0);

        for (double c : {0.25, 3.0, 100.0}) {
            std::vector<double> ws(fx_w.size());
            for (size_t i = 0; i < fx_w.size(); ++i) ws[i] = c * fx_w[i];
            Tensor Ws = vec(ws);
            double snis_c = 0.0, ips_c = 0.0;
            call_guarded([&] { return utils::ope_self_normalized_ips(R, Ws); }, &snis_c);
            call_guarded([&] { return utils::ope_ips_estimate(R, Ws); }, &ips_c);
            check(std::abs(snis_c - snis0) < 1e-12,
                  "4.1 SNIS is SCALE INVARIANT (c=" + std::to_string(c) + ")");
            check(std::abs(ips_c - c * ips0) < 1e-12,
                  "4.2 IPS scales LINEARLY with c=" + std::to_string(c) +
                  " (it is NOT scale invariant)");
        }
    }

    // -----------------------------------------------------------------------
    section("5. THE ENUMERATOR IS ITSELF CORRECT");
    //
    // A broken enumerator would make every downstream check meaningless, so
    // it is validated FIRST against a hand-computed value.
    //
    // pi_0 = (0.5, 0.3, 0.2), pi_1 = (0.2, 0.5, 0.3), mu = (1, 2, -1)
    //     sum_a pi_0(a) mu(a) = 0.5*1 + 0.3*2 + 0.2*(-1) = 0.9
    // -----------------------------------------------------------------------
    Bandit b;
    b.pi0 = {0.5, 0.3, 0.2};
    b.pi1 = {0.2, 0.5, 0.3};
    b.mu  = {1.0, 2.0, -1.0};

    const double v_pi1 = 0.2 * 1.0 + 0.5 * 2.0 + 0.3 * (-1.0);   // == 0.9

    std::vector<double> rho;
    for (size_t a = 0; a < b.pi0.size(); ++a) rho.push_back(b.pi1[a] / b.pi0[a]);

    {
        // The plain mean of the logged rewards under pi_0, with unit weights.
        auto plain_mean = [](const std::vector<double>& r, const std::vector<double>&,
                             const std::vector<double>&, const std::vector<double>&) {
            double s = 0.0;
            for (double x : r) s += x;
            return s / static_cast<double>(r.size());
        };
        // **THE `rho` TABLE MUST HAVE ONE ENTRY PER ACTION.** The first
        // draft passed a two-element vector to a three-action bandit, and
        // the enumerator indexed it by action -- a heap-buffer-overflow
        // that AddressSanitizer found and -O2 hid. The control estimators
        // below IGNORE the weights, so the bad read never changed a result;
        // it was invisible without a sanitizer.
        const std::vector<double> rho_per_action =
            std::vector<double>(b.pi0.size(), 1.0);
        double got = enumerate_expectation(b, 2, plain_mean, rho_per_action, 0.0);
        check(std::abs(got - 0.9) < 1e-12,
              "5.1 the enumerator reproduces a hand-computed expectation");

        auto plain_sum = [](const std::vector<double>& r, const std::vector<double>&,
                             const std::vector<double>&, const std::vector<double>&) {
            double s = 0.0;
            for (double x : r) s += x;
            return s;
        };
        double got_sum = enumerate_expectation(b, 2, plain_sum, rho_per_action, 0.0);
        check(std::abs(got_sum - 1.8) < 1e-12,
              "5.2 the enumerator's SUM control is exactly N times the mean "
              "control (so a mean/sum confusion is visible)");

        // The per-action-table guard must actually FIRE on a mis-sized table
        // -- otherwise it is a comment, not a check.
        {
            const std::vector<double> too_short(b.pi0.size() - 1, 1.0);
            double bad = enumerate_expectation(b, 2, plain_mean, too_short, 0.0);
            check(std::isnan(bad),
                  "5.3 the enumerator REJECTS a weight table that is not one "
                  "entry per action (the heap-buffer-overflow guard)");
        }
    }

    // -----------------------------------------------------------------------
    section("6. IPS IS UNBIASED WITH EXACT PROPENSITIES");
    //
    // Checked by EXACT ENUMERATION, so the identity is exact.
    // -----------------------------------------------------------------------
    {
        for (int n : {2, 3}) {
            // The weights DEPEND on which action was logged, so the
            // estimator is applied per sequence. The library's own
            // estimator is used on the per-sequence data, so this is a
            // check of the LIBRARY, not of a reference.
            const int A = 3;
            double total = 0.0, mass = 0.0;
            std::vector<int> seq(n, 0);
            // A^n sequences, not 2^n -- see the note above.
            long long n_sequences = 1;
            for (int t = 0; t < n; ++t) n_sequences *= A;
            for (long long idx = 0; idx < n_sequences; ++idx) {
                long long code = idx;
                for (int t = n - 1; t >= 0; --t) { seq[t] = (int)(code % A); code /= A; }
                double p = 1.0;
                for (int t = 0; t < n; ++t) p *= b.pi0[seq[t]];
                mass += p;
                if (p == 0.0) continue;
                std::vector<double> rr(n), ww(n);
                for (int t = 0; t < n; ++t) { rr[t] = b.mu[seq[t]]; ww[t] = rho[seq[t]]; }
                Tensor Rr = vec(rr), Ww = vec(ww);
                double val = 0.0;
                call_guarded([&] { return utils::ope_ips_estimate(Rr, Ww); }, &val);
                total += p * val;
            }
            check(std::abs(mass - 1.0) < 1e-12,
                  "6.1 the enumeration's total probability mass is exactly 1");
            check(std::abs(total - v_pi1) < 1e-13,
                  "6.2 E[IPS] over " + std::to_string(n) +
                  " samples == V_pi1 EXACTLY");
        }
    }

    // -----------------------------------------------------------------------
    section("7. DOUBLE ROBUSTNESS -- BOTH ARMS, EXACTLY");
    //
    //   ARM A: propensities EXACT, reward model WRONG  -> still unbiased.
    //   ARM B: reward model EXACT, propensities WRONG  -> still unbiased.
    //   ARM C: BOTH wrong                               -> biased.
    //
    // ARM C is what makes the fixture discriminate: if the estimator were
    // accidentally unbiased in every case, the check would be vacuous.
    // -----------------------------------------------------------------------
    {
        const std::vector<double> rhat_wrong = {0.5, -0.5, 0.25};
        const std::vector<double> rhat_exact = b.mu;
        std::vector<double> pi0_wrong = {0.4, 0.4, 0.2};
        std::vector<double> rho_wrong;
        for (size_t a = 0; a < b.pi0.size(); ++a) rho_wrong.push_back(b.pi1[a] / pi0_wrong[a]);

        double vhat_wrong = 0.0, vhat_exact = 0.0;
        for (size_t a = 0; a < b.pi1.size(); ++a) {
            vhat_wrong += b.pi1[a] * rhat_wrong[a];
            vhat_exact += b.pi1[a] * rhat_exact[a];
        }

        // The library's estimator, applied per enumerated sequence.
        auto lib_dr = [](const std::vector<double>& r, const std::vector<double>& w,
                         const std::vector<double>& v, const std::vector<double>& q) {
            Tensor Rr = vec(r), Ww = vec(w), Vv = vec(v), Qq = vec(q);
            double val = 0.0;
            call_guarded([&] { return utils::ope_doubly_robust_estimate(Rr, Ww, Vv, Qq); },
                         &val);
            return val;
        };

        double arm_a = enumerate_expectation_with_model(b, 3, lib_dr, rho,
                                                          vhat_wrong, rhat_wrong);
        check(std::abs(arm_a - v_pi1) < 1e-13,
              "7.1 ARM A: DR with EXACT propensities and a WRONG reward model "
              "is exactly unbiased");

        double arm_b = enumerate_expectation_with_model(b, 3, lib_dr, rho_wrong,
                                                          vhat_exact, rhat_exact);
        check(std::abs(arm_b - v_pi1) < 1e-13,
              "7.2 ARM B: DR with an EXACT reward model and WRONG "
              "propensities is exactly unbiased");

        double arm_c = enumerate_expectation_with_model(b, 3, lib_dr, rho_wrong,
                                                          vhat_wrong, rhat_wrong);
        check(std::abs(arm_c - v_pi1) > 1e-6,
              "7.3 ARM C: DR with BOTH wrong is measurably biased, so the "
              "fixture discriminates");

        // And the direct method is unbiased ONLY with the exact model --
        // so DR's extra term is doing real work.
        double dm_wrong = 0.0, dm_exact = 0.0;
        Tensor Vw = vec(std::vector<double>(3, vhat_wrong));
        Tensor Ve = vec(std::vector<double>(3, vhat_exact));
        call_guarded([&] { return utils::ope_direct_method_estimate(Vw); }, &dm_wrong);
        call_guarded([&] { return utils::ope_direct_method_estimate(Ve); }, &dm_exact);
        check(std::abs(dm_exact - v_pi1) < 1e-13,
              "7.4 DM with an EXACT reward model is exactly unbiased");
        check(std::abs(dm_wrong - v_pi1) > 1e-6,
              "7.5 DM with a WRONG reward model is measurably biased");

        // -------------------------------------------------------------------
        // WHY THE TWO ARMS ARE NOT INTERCHANGEABLE -- PINNED, NOT ASSUMED.
        //
        // **THE MUTATION HARNESS MEASURED THIS**: flipping the correction's
        // sign, or dropping the correction's weight, breaks ONLY ARM A.
        // The reason is that in ARM B the reward model is EXACT, so
        // r_i - q_i is IDENTICALLY ZERO for every sample -- and a term that
        // is identically zero cannot be changed by anything done to it.
        // **ARM B IS THEREFORE STRUCTURALLY INSENSITIVE TO THE CORRECTION
        // TERM**, and ARM A is the arm that pins it. The arms are
        // complementary: ARM A pins the correction, ARM B pins the Vhat term.
        //
        // This check pins the STRUCTURE (the correction vanishes in ARM B),
        // so the asymmetry is a property of the fixture rather than a lucky
        // accident of which mutations happen to be caught.
        // -------------------------------------------------------------------
        {
            double max_residual = 0.0;
            for (size_t a = 0; a < b.mu.size(); ++a)
                max_residual = std::max(max_residual,
                                        std::abs(b.mu[a] - rhat_exact[a]));
            check(max_residual == 0.0,
                  "7.6 in ARM B the reward model is EXACT, so r - q is "
                  "IDENTICALLY ZERO for every sample (which is why ARM B "
                  "cannot see a defect in the correction term)");
            check(std::abs(std::abs(dm_wrong - v_pi1) - std::abs(dm_exact - v_pi1)) > 1e-6,
                  "7.7 the two arms really are different regimes: the WRONG "
                  "model's direct method is biased and the EXACT model's is "
                  "not");
        }
    }

    // -----------------------------------------------------------------------
    section("8. THE DECOMPOSITION DR == DM + THE CORRECTION");
    // -----------------------------------------------------------------------
    {
        double dr = 0.0, dm = 0.0;
        call_guarded([&] { return utils::ope_doubly_robust_estimate(R, W, V, Q); }, &dr);
        call_guarded([&] { return utils::ope_direct_method_estimate(V); }, &dm);
        double correction = 0.0;
        for (size_t i = 0; i < fx_r.size(); ++i)
            correction += fx_w[i] * (fx_r[i] - fx_q[i]);
        correction /= static_cast<double>(fx_r.size());
        check(std::abs(dr - (dm + correction)) < 1e-12,
              "8.1 DR == DM + (1/N) sum_i rho_i (r_i - Rhat_i)");
        check(std::abs(correction) > 1e-6,
              "8.2 the correction term is non-zero on this fixture, so the "
              "decomposition is a real test");
    }

    // -----------------------------------------------------------------------
    section("9. THE EFFECTIVE SAMPLE SIZE");
    // -----------------------------------------------------------------------
    {
        double ess = 0.0;
        call_guarded([&] { return utils::ope_effective_sample_size(W); }, &ess);
        check(std::abs(ess - 1.8) < 1e-12, "9.1 ESS matches the pinned 1.8");
        check(ess <= 4.0 + 1e-12, "9.2 ESS <= N");

        Tensor eq = vec(std::vector<double>(5, 2.5));
        double ess_eq = 0.0;
        call_guarded([&] { return utils::ope_effective_sample_size(eq); }, &ess_eq);
        check(std::abs(ess_eq - 5.0) < 1e-12,
              "9.3 ESS == N when all the weights are equal (whatever their "
              "scale)");

        Tensor dom = vec({100.0, 0.0, 0.0, 0.0});
        double ess_dom = 0.0;
        call_guarded([&] { return utils::ope_effective_sample_size(dom); }, &ess_dom);
        check(std::abs(ess_dom - 1.0) < 1e-12,
              "9.4 ESS == 1 when a single weight carries everything");

        for (double c : {0.25, 3.0, 100.0}) {
            std::vector<double> ws(fx_w.size());
            for (size_t i = 0; i < fx_w.size(); ++i) ws[i] = c * fx_w[i];
            Tensor Ws = vec(ws);
            double ess_c = 0.0;
            call_guarded([&] { return utils::ope_effective_sample_size(Ws); }, &ess_c);
            check(std::abs(ess_c - ess) < 1e-12,
                  "9.5 ESS is SCALE INVARIANT (c=" + std::to_string(c) + ")");
        }
    }

    // -----------------------------------------------------------------------
    section("10. THE GUARDS -- EACH MESSAGE IS DISTINCT");
    //
    // **THE DISCRIMINATOR IS THE MESSAGE, NOT THE THROW.** A test that only
    // asserts "it threw" passes while the guard it means to cover is gone.
    // -----------------------------------------------------------------------
    {
        Tensor empty = vec({});
        Tensor one = vec({1.0});
        Tensor two = vec({1.0, 2.0});

        expect_throw([&] { utils::ope_ips_estimate(empty, one); }, "empty",
                     "10.1 an empty reward vector is rejected");
        expect_throw([&] { utils::ope_ips_estimate(one, two); },
                     "same number of samples",
                     "10.2 mismatched sample counts are rejected");
        expect_throw([&] { utils::ope_self_normalized_ips(one, two); },
                     "same number of samples",
                     "10.3 SNIS rejects mismatched sample counts too");
        expect_throw([&] { utils::ope_doubly_robust_estimate(one, one, one, two); },
                     "same number of samples",
                     "10.4 DR rejects mismatched sample counts");

        // A zero total weight makes SNIS's denominator zero.
        Tensor zeros = vec({0.0, 0.0});
        Tensor rw = vec({1.0, 2.0});
        expect_throw([&] { utils::ope_self_normalized_ips(rw, zeros); },
                     "total weight",
                     "10.5 a zero total weight is rejected by SNIS");

        // A NEGATIVE weight is meaningless for an importance weight.
        Tensor neg = vec({1.0, -2.0});
        expect_throw([&] { utils::ope_ips_estimate(rw, neg); }, "weight",
                     "10.6 a negative weight is rejected");
        expect_throw([&] { utils::ope_effective_sample_size(neg); }, "weight",
                     "10.7 ESS rejects a negative weight");

        // The weight guard's message must be DISTINCT from the total-weight
        // guard's, and from the sample-count guard's -- otherwise a test
        // that searches for a common word cannot tell which guard fired.
        // This is the repo's recorded "the discriminator is the MESSAGE"
        // trap, so the distinctness is asserted rather than assumed.
        std::string m_count, m_neg, m_total;
        threw_with([&] { utils::ope_ips_estimate(one, two); },
                   "same number of samples", &m_count);
        threw_with([&] { utils::ope_ips_estimate(rw, neg); }, "weight", &m_neg);
        threw_with([&] { utils::ope_self_normalized_ips(rw, zeros); },
                   "total weight", &m_total);
        check(m_count != m_neg && m_neg != m_total && m_count != m_total,
              "10.8 the three guard messages are DISTINCT from one another");
    }

    // -----------------------------------------------------------------------
    section("11. THE BUNDLED REPORT");
    // -----------------------------------------------------------------------
    {
        utils::OPEReport rep;
        double dummy = 0.0;
        bool ok = call_guarded([&] {
            rep = utils::ope_evaluate(R, W, V, Q);
            return 0.0;
        }, &dummy);
        check(ok, "11.1 the bundled report runs");
        check(std::abs(rep.ips - 1.65) < 1e-12, "11.2 the report's IPS is right");
        check(std::abs(rep.snis - 1.5714285714285714) < 1e-12,
              "11.3 the report's SNIS is right");
        check(std::abs(rep.dm - 0.1125) < 1e-12, "11.4 the report's DM is right");
        check(std::abs(rep.dr - 1.7625) < 1e-12, "11.5 the report's DR is right");
        check(std::abs(rep.ess - 1.8) < 1e-12, "11.6 the report's ESS is right");
    }

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
