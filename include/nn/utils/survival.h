#ifndef NN_UTILS_SURVIVAL_H
#define NN_UTILS_SURVIVAL_H

#include "../core/tensor.h"

#include <cstddef>
#include <vector>

// ===========================================================================
// SURVIVAL ANALYSIS: the Kaplan-Meier estimator, the Nelson-Aalen cumulative
// hazard, the log-rank test, Harrell's concordance index, and the Cox
// proportional-hazards partial log-likelihood with its analytic gradient.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-19)
// ---------------------------------------------------------------------------
//
//   grep -rinE 'survival|kaplan|cox.proportional|hazard|time.to.event|censoring'
//           include/ tests/ --include=*.h --include=*.cpp
//   -> include/nn/layers/utility/stochastic_depth.cpp:11
//        ("Sample a single Bernoulli for block SURVIVAL")   <- FALSE POSITIVE
//   -> include/nn/utils/sinkhorn.cpp:359
//        ("an argument-evaluation-order HAZARD")              <- FALSE POSITIVE
//
//   ZERO IMPLEMENTATION.
//
// The repo ships ~250 layers and 60+ utility files and NO survival analysis.
// The neighbouring statistics it DOES ship -- conformal prediction
// (utils/conformal.h), rank correlation (utils/rank_correlation.h), isotonic
// regression (utils/isotonic_regression.h), the Gaussian process
// (utils/gaussian_process.h) -- are all DISTRIBUTION-FREE or
// NON-PARAMETRIC in exactly the same sense, so this is the same family of
// tool and not a new kind of thing.
//
// It is NOT redundant with any of them:
//   * utils/conformal.h measures a SET's coverage. It has no notion of a
//     survival time, of censoring, or of a hazard.
//   * utils/rank_correlation.h measures the association between two
//     quantities observed WITHOUT censoring. A censored observation breaks
//     exactly the assumption that a rank statistic rests on.
//   * utils/gaussian_process.h regresses a value on covariates. The subject
//     here is a TIME TO EVENT, which is only observed up to a censoring
//     time, and the estimator that handles that is a different algorithm
//     (a risk set, not a design matrix).
//
// ---------------------------------------------------------------------------
// WHAT IT IS MODEL-FREE AND WHY THAT MATTERS
// ---------------------------------------------------------------------------
//
// Every core below is a free function over `Tensor`. The whole file is
// therefore testable with no Model, no optimizer and no network -- the same
// shape `utils/sliced_wasserstein.h`, `utils/crf.h`, `utils/conformal.h`,
// `utils/rank_correlation.h` and `utils/off_policy_eval.h` were built with.
// Nothing here is random.
//
// ---------------------------------------------------------------------------
// THE DATA
// ---------------------------------------------------------------------------
//
// A SURVIVAL DATASET is a list of OBSERVED TIMES and a list of EVENT
// INDICATORS:
//
//   times   (N, 1)  the observed time for each subject. For a subject who
//                  failed, this is the failure time; for a subject who was
//                  censored, this is the time they were last seen alive.
//   events  (N, 1)  1.0 if the subject FAILED (an "event" was observed),
//                  0.0 if the subject was CENSORED (right-censored: they
//                  were alive at `times` and then nothing more is known).
//
// **CENSORING IS NOT MISSING DATA.** A censored subject is known to have
// survived to `times`; that is real information, and every estimator below
// uses it. The subject is in the risk set up to and including their
// censoring time.
//
// ---------------------------------------------------------------------------
// THE CONVENTIONS, AND EACH ONE IS A MEASURED DECISION
// ---------------------------------------------------------------------------
//
// The full argument for each one, with the numbers, is in the header's
// companion oracle (tests/survival_oracle.py) and is summarised here.
//
// **C1. THE RISK SET AT TIME t IS {j : t_j >= t}.** A subject CENSORED at
// exactly t is still AT RISK at t -- it is known to have survived to t --
// and is removed only for times strictly after t. The opposite convention
// ({j : t_j > t}) drops a censored-at-t subject from the risk set at the
// very time it is known to be alive, which biases the estimator downward.
//
// MEASURED, and the fixture DISCRIMINATES the two conventions: on the
// three-subject fixture (times [1,1,2], events [1,0,1]) the at-risk count
// at t=1 is 3 and S(1) = 2/3; the opposite convention gives 2 and 1/2.
// A fixture that cannot tell the two apart pins nothing.
//
// Cross-checked against lifelines' KaplanMeierFitter: the at-risk counts
// and the survival curve agree to < 1e-12.
//
// **C2. THE KAPLAN-MEIER CURVE IS RIGHT-CONTINUOUS.** It is CONSTANT on
// [t_k, t_{k+1}) and takes its POST-EVENT value at t_k:
//     S(t) = prod over event times u <= t of (1 - d(u) / n_risk(u))
//
// **C3. THE MEDIAN SURVIVAL TIME IS THE FIRST EVENT TIME AT WHICH
// S(t) <= 0.5**, and it is +INFINITY when S never reaches 0.5. It is NOT
// interpolated. A curve that lands exactly on 0.5 at an event time cannot
// distinguish this rule from an interpolated median, so the suite pins a
// curve that CROSSES 0.5 strictly between two event times.
//
// **C4. GREENWOOD'S VARIANCE IS SHIPPED ALONGSIDE THE CURVE:**
//     Var(S(t)) = S(t)^2 * sum over event times u <= t of
//                 d(u) / (n_risk(u) * (n_risk(u) - d(u)))
// The term is 0 when n_risk(u) == d(u) (the whole risk set died) -- that
// is a LIMIT, not a division by zero. A variance estimator that divides by
// zero there produces inf or NaN, and a metric that reports NaN reports
// nothing.
//
// **C5. THE NELSON-AALEN CUMULATIVE HAZARD IS THE NAIVE ONE:**
//     H(t) = sum over event times u <= t of d(u) / n_risk(u)
// It is reported on the SAME grid as the Kaplan-Meier curve -- the EVENT
// times -- so the two are directly comparable row by row. A censored-only
// time is NOT on the grid: the hazard does not change there.
//
// **C6. THE LOG-RANK TEST IS THE TWO-SAMPLE, ONE-DEGREE-OF-FREEDOM
// VERSION.** Its statistic is
//     (O1 - E1)^2 / V
// with O1 the observed group-1 events,
//     E1 = sum over event times of d(t) * n1(t) / n(t)
// and V the summed HYPERGEOMETRIC variance
//     d(t) * (n1(t)/n(t)) * (1 - n1(t)/n(t)) * ((n(t) - d(t))/(n(t) - 1))
// The p-value is the chi-square-1 survival function, which has the exact
// closed form
//     p = erfc(sqrt(statistic / 2))
// -- no incomplete gamma, no special-function library.
//
// **k > 2 GROUPS IS OUT OF SCOPE**, and says so: the statistic generalises
// to k-1 degrees of freedom, but its p-value does not have a closed form
// in erfc, and shipping a general incomplete gamma for one extra function
// is a scope decision this file does not make silently.
//
// **C7. THE CONCORDANCE INDEX IS HARRELL'S C ON THE RISK SCORE: HIGHER
// SCORE = HIGHER RISK = SHORTER SURVIVAL.** The pair is concordant when the
// subject who failed FIRST has the HIGHER risk score.
//
// **THIS IS A SIGN CONVENTION, AND THE TWO READINGS EXIST IN THE WILD.**
// lifelines' `concordance_index(event_times, predicted_scores, ...)` treats
// the score as a SURVIVAL-TIME-like quantity -- HIGHER SCORE = LONGER
// SURVIVAL -- and its own Cox example therefore passes
// `-cph.predict_partial_hazard(...)`, NEGATED.
//
// MEASURED on a four-subject fixture with times [1,2,3,4] and all events:
//     risk score [4,3,2,1] (high = early failure)    -> lifelines  0.0
//     survival score [1,2,3,4] (high = late failure) -> lifelines  1.0
// So lifelines' convention is the OPPOSITE of the risk convention, and an
// implementation that silently picks one reports the COMPLEMENT of the
// other: 1 - C.
//
// **THIS FILE SHIPS THE RISK CONVENTION**, because a Cox model's own
// output is a risk score (the linear predictor, higher = more risk) and
// every published C-index for a Cox model uses that orientation. The
// oracle cross-checks against lifelines on the NEGATED score and
// separately asserts the COMPLEMENTARITY IDENTITY
//     C_risk(s) + C_risk(-s) == 1
// which holds exactly for every score vector (including one with tied
// scores), so the relationship is pinned from both sides.
//
// The admissibility rule (the pairs that count at all):
//   * two observations with the SAME time are admissible only if exactly
//     ONE of them is an event -- then the event is known to have come
//     first;
//   * two observations with DIFFERENT times are admissible if the one with
//     the EARLIER time is an event -- it is then known to have failed
//     before the other;
//   * a pair with EQUAL risk scores counts as HALF correct;
//   * for a pair with equal times where one is an event, "correct" means
//     the EVENT's risk score is HIGHER.
//
// **C8. THE COX PARTIAL LOG-LIKELIHOOD IS THE LOG-PARTIAL-LIKELIHOOD, NOT
// THE FULL LIKELIHOOD:** it has no baseline-hazard term, so it is a
// function of the COEFFICIENTS ONLY and is comparable across different
// censoring patterns.
//
// **C9. THE COX TIE HANDLING IS AN EXPLICIT PARAMETER, not a silent
// default:**
//   * BRESLOW:  l = sum over event times t of
//               [ sum over i in D(t) of x_i.beta
//                 - d(t) * log( S0(t) ) ]
//               with S0(t) = sum over the risk set of exp(x_j.beta)
//   * EFRON:    the same, except the d(t) identical terms are replaced by
//               d(t) terms with the tied deaths' OWN contribution removed
//               in d(t) equal steps:
//               l = sum over t of
//                   [ sum over i in D(t) of x_i.beta
//                     - sum over l = 0..d(t)-1 of
//                       log( S0(t) - (l/d(t)) * S0_D(t) ) ]
// Breslow and Efron COINCIDE when no two events share a time, which is
// exactly why a test that never produces a tie cannot tell them apart --
// so the suite has a fixture WITH tied event times and asserts that the
// two methods DIFFER there.
//
// **C10. THE COX GRADIENT IS THE ANALYTIC ONE, IN CLOSED FORM:**
//   Breslow at event time t:
//       sum over i in D(t) of x_i
//       - d(t) * ( sum over j in R(t) of x_j exp(x_j.beta) ) / S0(t)
//   and Efron replaces the single ratio by the d(t) ratios with the tied
//   deaths' weighted mean subtracted.
//
// **C11. THE PARTIAL LOG-LIKELIHOOD IS REPORTED AS A MEAN OVER THE NUMBER
// OF EVENTS, NOT A RAW SUM.**
//
// MEASURED: lifelines reports the SUM (-28.552063357741332 on the
// 24-subject fixture) and the MEAN is exactly that divided by the event
// count (17), giving -1.6795331386906664.
//
// The reason is the repo's own, twice-established convention:
// `utils/losses.h` documents that the mean reduction divides by `numel`,
// NOT by the row count (D7), and NOT_FIXED.md's VAE entry records the bug
// that happens when two halves of one loss are scaled differently -- "a
// loss that scales with the batch size is not a per-sample quantity". The
// partial likelihood is a sum over EVENT TIMES, so the analogous per-event
// quantity divides by the number of events.
//
// The GRADIENT is the mean too, so the step size does not change with the
// number of events -- which is what makes the objective usable by the
// repo's optimizers.
//
// ---------------------------------------------------------------------------
// THE CITATION
// ---------------------------------------------------------------------------
//
//   - Kaplan, E. L. and Meier, P. (1958). "Nonparametric Estimation from
//     Incomplete Observations." Journal of the American Statistical
//     Association, 53(282), 457-481.
//   - Nelson, W. (1972). "Theory of a Partial Sum Process." Annals of
//     Mathematical Statistics, 43(3), 1011-1016.
//   - Cox, D. R. (1972). "Regression Models and Life-Tables." Journal of
//     the Royal Statistical Society. Series B, 34(2), 187-202.
//   - Efron, B. (1977). "The Efficiency of Cox's Partial Likelihood
//     Function." Journal of the American Statistical Association, 72(359),
//     557-565.
//   - Harrell, F. E., Lee, K. L. and Mark, D. B. (1996). "Multivariable
//     Prognostic Models: Issues in Developing Models, Evaluating
//     Assumptions and Adequacy, and Measuring and Reducing Errors."
//     Statistics in Medicine, 15(4), 361-387.
//   - Mantel, N. (1966). "Evaluation of a Single-Statistic Test."
//     Cancer, 19(4), 699-706.
//
// All six are pre-arXiv or have no arXiv record. **NO arXiv ID IS
// ASSERTED** -- this repo's expansion queue has produced fifteen wrong arXiv
// identifiers, every one a plausible-looking neighbour of the right paper.
//
// The reference implementation used for the cross-check is lifelines 0.30.3,
// a MAINTAINED implementation (the one scikit-survival also portions from).
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// How the Cox partial log-likelihood handles TIED event times.
// ---------------------------------------------------------------------------
enum class CoxTieMethod {
    Breslow,  // the classic partial likelihood (C9)
    Efron     // the tie-corrected variant (C9)
};

// ---------------------------------------------------------------------------
// The Kaplan-Meier estimate.
//
// `times` is the grid of EVENT times, ascending. It does NOT contain the
// censoring times: the curve does not change at a censoring, so a censored
// time is not a point on the curve (C5).
//
// `at_risk` is n_risk(t) -- the size of the risk set at that time.
// `events`  is d(t) -- the number of failures at that time.
// `survival` is S(t) -- the survival probability AFTER the failures at t.
// `variance` is Greenwood's variance of S(t) (C4).
// ---------------------------------------------------------------------------
struct KaplanMeierEstimate {
    std::vector<double> times;
    std::vector<double> at_risk;
    std::vector<double> events;
    std::vector<double> survival;
    std::vector<double> variance;

    size_t n_points() const { return times.size(); }

    // S(t) for an ARBITRARY t: the post-event value at the last event time
    // <= t, or 1.0 if t precedes the first event (C2).
    double survival_at(double t) const;

    // The median survival time: the FIRST event time at which S(t) <= 0.5,
    // or +INFINITY if S never reaches 0.5 (C3).
    double median_survival_time() const;
};

// ---------------------------------------------------------------------------
// kaplan_meier -- the Kaplan-Meier estimator with Greenwood's variance.
//
//   times   (N, 1) observed times. Must be finite and NON-NEGATIVE.
//   events  (N, 1) event indicators, each EXACTLY 0.0 or 1.0.
//
// The estimate is computed on the EVENT-time grid (C2, C5).
//
// THROWS std::invalid_argument when:
//   - `times` and `events` have different row counts;
//   - either has no rows;
//   - either is not a single column;
//   - any time is not finite, or is negative;
//   - any event indicator is not exactly 0.0 or 1.0.
//
// Every message is DISTINCT: a substring that appears in two different
// messages discriminates nothing between them.
// ---------------------------------------------------------------------------
KaplanMeierEstimate kaplan_meier(const Tensor& times, const Tensor& events);

// ---------------------------------------------------------------------------
// The Nelson-Aalen cumulative hazard.
//
// `times` is the SAME grid as the Kaplan-Meier estimate -- the EVENT times
// (C5) -- and `hazard` is H(t) after each of them.
// ---------------------------------------------------------------------------
struct NelsonAalenEstimate {
    std::vector<double> times;
    std::vector<double> at_risk;
    std::vector<double> events;
    std::vector<double> hazard;

    size_t n_points() const { return times.size(); }

    // H(t) for an ARBITRARY t: the value at the last event time <= t, or
    // 0.0 if t precedes the first event.
    double hazard_at(double t) const;
};

// ---------------------------------------------------------------------------
// nelson_aalen -- the naive Nelson-Aalen cumulative hazard.
//
// Takes the SAME inputs as `kaplan_meier` and enforces the SAME guards.
// ---------------------------------------------------------------------------
NelsonAalenEstimate nelson_aalen(const Tensor& times, const Tensor& events);

// ---------------------------------------------------------------------------
// The two-sample log-rank test.
//
// `observed_1` is the number of group-1 failures, `expected_1` the number
// expected under the null of equal hazards, `variance` the summed
// hypergeometric variance, `statistic` the chi-square-1 statistic and
// `p_value` its p-value (C6).
// ---------------------------------------------------------------------------
struct LogRankResult {
    double observed_1 = 0.0;
    double expected_1 = 0.0;
    double variance = 0.0;
    double statistic = 0.0;
    double p_value = 1.0;
};

// ---------------------------------------------------------------------------
// log_rank_test -- the two-sample log-rank test (C6).
//
//   times_a   (N1, 1) group-1 observed times.
//   events_a  (N1, 1) group-1 event indicators.
//   times_b   (N2, 1) group-2 observed times.
//   events_b  (N2, 1) group-2 event indicators.
//
// The statistic is 0 and the p-value is 1 when the two groups are
// identical, or when the variance is 0 (which happens when there are no
// events at all).
//
// THROWS std::invalid_argument when either group violates the same
// preconditions as `kaplan_meier`, or when the two groups have different
// row counts (which would be a caller error, not a data property).
// ---------------------------------------------------------------------------
LogRankResult log_rank_test(const Tensor& times_a, const Tensor& events_a,
                             const Tensor& times_b, const Tensor& events_b);

// ---------------------------------------------------------------------------
// Harrell's concordance index on the RISK score (C7).
//
// `num_correct` counts the admissible pairs the model ordered correctly,
// `num_tied` the admissible pairs whose risk scores were EQUAL (each
// counting as half), and `num_pairs` the admissible pairs in total.
//
// The C-index is (num_correct + 0.5 * num_tied) / num_pairs.
//
// **THE SIGN CONVENTION IS C7: HIGHER SCORE = HIGHER RISK = SHORTER
// SURVIVAL.** A Cox model's linear predictor is exactly such a score. If
// you pass lifelines' `concordance_index` the SAME vector, you will get
// its COMPLEMENT (1 - C), because lifelines treats the score as a
// survival-time-like quantity. The oracle pins the complementarity
// identity C(s) + C(-s) == 1.
// ---------------------------------------------------------------------------
struct ConcordanceResult {
    double num_correct = 0.0;
    double num_tied = 0.0;
    double num_pairs = 0.0;

    // The C-index itself. NaN when there are no admissible pairs at all
    // (which happens for an all-censored dataset), because a ratio with a
    // zero denominator is not a number.
    double c_index() const;
};

// ---------------------------------------------------------------------------
// concordance_index -- Harrell's C on the risk score (C7).
//
//   times         (N, 1) observed times.
//   events        (N, 1) event indicators.
//   risk_scores   (N, 1) the RISK score. Higher = more risk.
//
// THROWS std::invalid_argument when the three tensors have different row
// counts, or when any of them violates the preconditions of
// `kaplan_meier` (the risk score must additionally be finite).
// ---------------------------------------------------------------------------
ConcordanceResult concordance_index(const Tensor& times, const Tensor& events,
                                    const Tensor& risk_scores);

// ---------------------------------------------------------------------------
// The Cox partial log-likelihood and its gradient.
//
// `log_likelihood` is the MEAN over the number of events (C11) and
// `gradient` is d(partial log-likelihood) / d(beta), shape (1, D),
// likewise a mean.
//
// `n_events` is the number of events -- the divisor, reported so a caller
// can recover the raw sum that lifelines and R report.
// ---------------------------------------------------------------------------
struct CoxPartialLikelihood {
    double log_likelihood = 0.0;
    Tensor gradient;   // (1, D)
    double n_events = 0.0;
};

// ---------------------------------------------------------------------------
// cox_partial_log_likelihood -- the Cox partial log-likelihood and its
// analytic gradient (C8, C9, C10, C11).
//
//   beta    (1, D) the coefficient vector. Must be finite.
//   X       (N, D) the design matrix. Must be finite.
//   times   (N, 1) observed times. Must be finite and non-negative.
//   events  (N, 1) event indicators, each exactly 0.0 or 1.0.
//   method  Breslow (the default) or Efron.
//
// **AT LEAST ONE EVENT IS REQUIRED.** With no events the partial
// likelihood is empty and its gradient is identically zero, which is a
// silently useless answer rather than an error.
//
// THROWS std::invalid_argument when:
//   - `beta` is not a single row, or `X` does not have D columns;
//   - the row counts of `beta`/`X`/`times`/`events` disagree;
//   - there are fewer than 2 rows;
//   - `X` has no columns;
//   - any entry is not finite;
//   - any time is negative;
//   - any event indicator is not exactly 0.0 or 1.0;
//   - there are NO events.
//
// Every message is DISTINCT.
// ---------------------------------------------------------------------------
CoxPartialLikelihood cox_partial_log_likelihood(const Tensor& beta, const Tensor& X,
                                                 const Tensor& times, const Tensor& events,
                                                 CoxTieMethod method = CoxTieMethod::Breslow);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_SURVIVAL_H
