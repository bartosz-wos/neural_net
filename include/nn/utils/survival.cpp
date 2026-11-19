#include "survival.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// ===========================================================================
// utils/survival.cpp -- the survival-analysis utilities: the Kaplan-Meier
// estimator, the Nelson-Aalen cumulative hazard, the log-rank test, Harrell's
// concordance index, and the Cox proportional-hazards partial log-likelihood
// with its analytic gradient.
//
// THE SPECIFICATION is tests/survival_oracle.py (float64 numpy), which was
// run BEFORE this file existed. Its discriminating power is MEASURED, not
// narrated: tests/survival_mutate_oracle.py breaks the oracle's model code in
// twenty specific ways and requires that it FAIL each time. The C++ fixtures
// are EMITTED by that oracle (tests/splice_survival_fixture.py), never
// hand-transcribed.
//
// The conventions -- the risk set, the right-continuity, the median rule,
// Greenwood's variance, the log-rank's hypergeometric variance, the
// concordance SIGN, the Cox tie handling and the mean-over-events -- are
// stated and justified in the header.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// Guards. EVERY message is DISTINCT, for the reason this repo has hit
// several separate times: a needle that appears in two different messages
// discriminates nothing between them, and the mutation harness then
// reports a survivor that is really a missing test.
// ---------------------------------------------------------------------------
void validate_survival_columns(const Tensor& times, const Tensor& events,
                               const char* who) {
    if (times.rows != events.rows)
        throw std::invalid_argument(std::string(who) +
                                    ": the times and the events have different "
                                    "row counts");
    if (times.rows == 0)
        throw std::invalid_argument(std::string(who) +
                                    ": the dataset has no rows");
    if (times.cols != 1)
        throw std::invalid_argument(std::string(who) +
                                    ": the times must be a single column");
    if (events.cols != 1)
        throw std::invalid_argument(std::string(who) +
                                    ": the events must be a single column");
    for (size_t i = 0; i < times.rows; ++i) {
        double t = times(i, 0);
        if (!std::isfinite(t))
            throw std::invalid_argument(std::string(who) +
                                        ": the times have a non-finite entry");
        if (t < 0.0)
            throw std::invalid_argument(std::string(who) +
                                        ": the times have a negative entry");
        double e = events(i, 0);
        if (!std::isfinite(e))
            throw std::invalid_argument(std::string(who) +
                                        ": the events have a non-finite entry");
        // The event indicator must be EXACTLY 0 or 1. A tolerance here
        // would silently accept 0.5 as "half an event", which is not a
        // thing a survival dataset can express.
        if (e != 0.0 && e != 1.0)
            throw std::invalid_argument(std::string(who) +
                                        ": an event indicator is neither 0 nor 1");
    }
}

// The distinct event times of a dataset, ascending.
//
// A subject CENSORED at time t is AT RISK at t (C1), so the risk set at t
// is {j : t_j >= t}.
struct EventGrid {
    std::vector<double> times;   // the distinct EVENT times, ascending
    std::vector<double> at_risk; // n_risk(t)
    std::vector<double> events;  // d(t)
};

EventGrid build_event_grid(const Tensor& times, const Tensor& events) {
    EventGrid g;
    // Collect the distinct event times.
    std::vector<double> ev_times;
    for (size_t i = 0; i < times.rows; ++i) {
        if (events(i, 0) == 1.0) {
            double t = times(i, 0);
            bool seen = false;
            for (double e : ev_times) {
                if (e == t) { seen = true; break; }
            }
            if (!seen) ev_times.push_back(t);
        }
    }
    std::sort(ev_times.begin(), ev_times.end());

    g.times = ev_times;
    g.at_risk.resize(ev_times.size());
    g.events.resize(ev_times.size());
    for (size_t k = 0; k < ev_times.size(); ++k) {
        double t = ev_times[k];
        double n_risk = 0.0;
        double d = 0.0;
        for (size_t i = 0; i < times.rows; ++i) {
            // C1: the risk set is {j : t_j >= t}. A subject censored at
            // exactly t is still at risk at t.
            if (times(i, 0) >= t) {
                n_risk += 1.0;
                if (times(i, 0) == t && events(i, 0) == 1.0) d += 1.0;
            }
        }
        g.at_risk[k] = n_risk;
        g.events[k] = d;
    }
    return g;
}

// The right-continuous step lookup: the value at the last grid point <= t,
// or `before` if t precedes the first grid point.
double step_lookup(const std::vector<double>& grid, const std::vector<double>& values,
                   double t, double before) {
    size_t idx = 0;
    bool found = false;
    for (size_t k = 0; k < grid.size(); ++k) {
        if (grid[k] <= t) { idx = k; found = true; }
        else break;
    }
    if (!found) return before;
    return values[idx];
}

}  // namespace

// ---------------------------------------------------------------------------
// KaplanMeierEstimate
// ---------------------------------------------------------------------------
double KaplanMeierEstimate::survival_at(double t) const {
    return step_lookup(times, survival, t, 1.0);
}

double KaplanMeierEstimate::median_survival_time() const {
    // C3: the FIRST event time at which S(t) <= 0.5, else +infinity.
    for (size_t k = 0; k < times.size(); ++k) {
        if (survival[k] <= 0.5) return times[k];
    }
    return std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// NelsonAalenEstimate
// ---------------------------------------------------------------------------
double NelsonAalenEstimate::hazard_at(double t) const {
    return step_lookup(times, hazard, t, 0.0);
}

// ---------------------------------------------------------------------------
// ConcordanceResult
// ---------------------------------------------------------------------------
double ConcordanceResult::c_index() const {
    if (num_pairs <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return (num_correct + 0.5 * num_tied) / num_pairs;
}

// ---------------------------------------------------------------------------
// kaplan_meier
// ---------------------------------------------------------------------------
KaplanMeierEstimate kaplan_meier(const Tensor& times, const Tensor& events) {
    validate_survival_columns(times, events, "kaplan_meier");

    EventGrid g = build_event_grid(times, events);
    KaplanMeierEstimate est;
    est.times = g.times;
    est.at_risk = g.at_risk;
    est.events = g.events;
    est.survival.resize(g.times.size());
    est.variance.resize(g.times.size());

    double s = 1.0;
    double greenwood = 0.0;
    for (size_t k = 0; k < g.times.size(); ++k) {
        // C2: S(t) = prod over event times u <= t of (1 - d(u)/n_risk(u)).
        s *= (1.0 - g.events[k] / g.at_risk[k]);
        est.survival[k] = s;

        // C4: Greenwood's variance. The term d/(n*(n-d)) is a LIMIT of 0
        // when n == d (the whole risk set died) -- it is NOT a division
        // by zero. Guarding it is what keeps the variance finite.
        double nr = g.at_risk[k];
        if (nr - g.events[k] > 0.0)
            greenwood += g.events[k] / (nr * (nr - g.events[k]));
        est.variance[k] = s * s * greenwood;
    }
    return est;
}

// ---------------------------------------------------------------------------
// nelson_aalen
// ---------------------------------------------------------------------------
NelsonAalenEstimate nelson_aalen(const Tensor& times, const Tensor& events) {
    validate_survival_columns(times, events, "nelson_aalen");

    EventGrid g = build_event_grid(times, events);
    NelsonAalenEstimate est;
    est.times = g.times;
    est.at_risk = g.at_risk;
    est.events = g.events;
    est.hazard.resize(g.times.size());

    // C5: the naive cumulative hazard on the EVENT-time grid.
    double cum = 0.0;
    for (size_t k = 0; k < g.times.size(); ++k) {
        cum += g.events[k] / g.at_risk[k];
        est.hazard[k] = cum;
    }
    return est;
}

// ---------------------------------------------------------------------------
// log_rank_test
// ---------------------------------------------------------------------------
LogRankResult log_rank_test(const Tensor& times_a, const Tensor& events_a,
                             const Tensor& times_b, const Tensor& events_b) {
    validate_survival_columns(times_a, events_a, "log_rank_test");
    validate_survival_columns(times_b, events_b, "log_rank_test");

    // The pooled event times: the distinct times at which EITHER group had
    // a failure.
    std::vector<double> ev_times;
    auto collect = [&](const Tensor& t, const Tensor& e) {
        for (size_t i = 0; i < t.rows; ++i) {
            if (e(i, 0) != 1.0) continue;
            double tv = t(i, 0);
            bool seen = false;
            for (double x : ev_times) {
                if (x == tv) { seen = true; break; }
            }
            if (!seen) ev_times.push_back(tv);
        }
    };
    collect(times_a, events_a);
    collect(times_b, events_b);
    std::sort(ev_times.begin(), ev_times.end());

    LogRankResult res;
    for (double t : ev_times) {
        // The risk sets. C1 applies to both groups.
        double n1 = 0.0, n2 = 0.0;
        for (size_t i = 0; i < times_a.rows; ++i)
            if (times_a(i, 0) >= t) n1 += 1.0;
        for (size_t i = 0; i < times_b.rows; ++i)
            if (times_b(i, 0) >= t) n2 += 1.0;
        double n = n1 + n2;

        double d1 = 0.0, d2 = 0.0;
        for (size_t i = 0; i < times_a.rows; ++i)
            if (times_a(i, 0) == t && events_a(i, 0) == 1.0) d1 += 1.0;
        for (size_t i = 0; i < times_b.rows; ++i)
            if (times_b(i, 0) == t && events_b(i, 0) == 1.0) d2 += 1.0;
        double d = d1 + d2;

        if (d == 0.0 || n == 0.0) continue;

        res.observed_1 += d1;
        res.expected_1 += d * n1 / n;
        // The HYPERGEOMETRIC variance, including the finite-population
        // factor (n-d)/(n-1). Without that factor the estimator is the
        // with-replacement variance and is a different statistic.
        if (n > 1.0)
            res.variance += d * (n1 / n) * (1.0 - n1 / n) * ((n - d) / (n - 1.0));
    }

    if (res.variance > 0.0)
        res.statistic = (res.observed_1 - res.expected_1) * (res.observed_1 - res.expected_1) /
                        res.variance;
    else
        res.statistic = 0.0;

    // C6: the chi-square-1 survival function has the exact closed form
    // erfc(sqrt(statistic/2)). No incomplete gamma is needed.
    res.p_value = std::erfc(std::sqrt(res.statistic / 2.0));
    return res;
}

// ---------------------------------------------------------------------------
// concordance_index
// ---------------------------------------------------------------------------
ConcordanceResult concordance_index(const Tensor& times, const Tensor& events,
                                    const Tensor& risk_scores) {
    validate_survival_columns(times, events, "concordance_index");
    if (risk_scores.rows != times.rows)
        throw std::invalid_argument(std::string("concordance_index") +
                                    ": the risk scores and the times have "
                                    "different row counts");
    if (risk_scores.cols != 1)
        throw std::invalid_argument(std::string("concordance_index") +
                                    ": the risk scores must be a single column");
    for (size_t i = 0; i < risk_scores.rows; ++i) {
        if (!std::isfinite(risk_scores(i, 0)))
            throw std::invalid_argument(std::string("concordance_index") +
                                        ": the risk scores have a non-finite entry");
    }

    ConcordanceResult res;
    size_t n = times.rows;
    for (size_t a = 0; a < n; ++a) {
        double ta = times(a, 0);
        bool ea = events(a, 0) == 1.0;
        double sa = risk_scores(a, 0);
        for (size_t b = a + 1; b < n; ++b) {
            double tb = times(b, 0);
            bool eb = events(b, 0) == 1.0;
            double sb = risk_scores(b, 0);

            // Which of the pair failed FIRST? That subject is the one
            // whose risk score SHOULD be higher (C7).
            bool a_failed_first;
            if (ta == tb) {
                // A pair with the same time is admissible only if exactly
                // ONE of them is an event -- then the event is known to
                // have come first. Two events or two censorings at the
                // same time tell us nothing about the order.
                if (ea == eb) continue;
                a_failed_first = ea;
            } else {
                size_t earlier = (ta < tb) ? a : b;
                bool e_earlier = (earlier == a) ? ea : eb;
                // The earlier subject must have FAILED for the pair to be
                // informative: if they were censored, we do not know
                // whether the other subject failed before or after them.
                if (!e_earlier) continue;
                a_failed_first = (earlier == a);
            }

            res.num_pairs += 1.0;
            if (sa == sb) {
                // A tied risk score counts as HALF correct.
                res.num_tied += 1.0;
            } else {
                double risk_first = a_failed_first ? sa : sb;
                double risk_second = a_failed_first ? sb : sa;
                if (risk_first > risk_second) res.num_correct += 1.0;
            }
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// cox_partial_log_likelihood
// ---------------------------------------------------------------------------
CoxPartialLikelihood cox_partial_log_likelihood(const Tensor& beta, const Tensor& X,
                                                 const Tensor& times, const Tensor& events,
                                                 CoxTieMethod method) {
    if (beta.rows != 1)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the coefficient vector must be a single row");
    if (X.cols != beta.cols)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the design matrix does not have the same "
                                    "number of columns as the coefficient vector");
    if (X.rows != times.rows)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the design matrix and the times have "
                                    "different row counts");
    if (beta.cols == 0)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the coefficient vector has no columns");
    if (X.rows < 2)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the dataset needs at least 2 rows");
    validate_survival_columns(times, events, "cox_partial_log_likelihood");

    for (size_t j = 0; j < beta.cols; ++j) {
        if (!std::isfinite(beta(0, j)))
            throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                        ": the coefficient vector has a non-finite entry");
    }
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            if (!std::isfinite(X(i, j)))
                throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                            ": the design matrix has a non-finite entry");
        }
    }

    size_t D = beta.cols;
    size_t N = X.rows;

    // eta_i = x_i . beta
    std::vector<double> eta(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < D; ++j) acc += X(i, j) * beta(0, j);
        eta[i] = acc;
    }

    // The event grid, built over the SAME risk-set convention (C1).
    EventGrid g = build_event_grid(times, events);

    double n_events = 0.0;
    for (size_t i = 0; i < N; ++i)
        if (events(i, 0) == 1.0) n_events += 1.0;
    if (n_events == 0.0)
        throw std::invalid_argument(std::string("cox_partial_log_likelihood") +
                                    ": the dataset contains no events, so the "
                                    "partial likelihood is empty");

    double loglik = 0.0;
    Tensor grad(1, D);
    for (size_t j = 0; j < D; ++j) grad(0, j) = 0.0;

    for (size_t k = 0; k < g.times.size(); ++k) {
        double t = g.times[k];
        double dt = g.events[k];
        if (dt == 0.0) continue;

        // S0 = sum over the risk set of exp(eta), and the matching
        // weighted sum of the covariates.
        double s0 = 0.0;
        std::vector<double> s0_x(D, 0.0);
        double s0_d = 0.0;
        std::vector<double> s0d_x(D, 0.0);
        double sum_eta_deaths = 0.0;
        std::vector<double> x_death_sum(D, 0.0);

        for (size_t i = 0; i < N; ++i) {
            bool in_risk = times(i, 0) >= t;
            bool is_death = (times(i, 0) == t && events(i, 0) == 1.0);
            if (is_death) {
                sum_eta_deaths += eta[i];
                for (size_t j = 0; j < D; ++j) x_death_sum[j] += X(i, j);
            }
            if (in_risk) {
                double w = std::exp(eta[i]);
                s0 += w;
                for (size_t j = 0; j < D; ++j) s0_x[j] += w * X(i, j);
                if (is_death) {
                    s0_d += w;
                    for (size_t j = 0; j < D; ++j) s0d_x[j] += w * X(i, j);
                }
            }
        }

        if (method == CoxTieMethod::Breslow) {
            // C9: l += sum_i x_i.beta - d(t) * log(S0(t)).
            loglik += sum_eta_deaths - dt * std::log(s0);
            // C10: the gradient's ratio term.
            for (size_t j = 0; j < D; ++j)
                grad(0, j) += x_death_sum[j] - dt * (s0_x[j] / s0);
        } else {
            // C9 (Efron): the tied deaths' own contribution is removed
            // from the denominator in dt equal steps.
            loglik += sum_eta_deaths;
            for (size_t j = 0; j < D; ++j) grad(0, j) += x_death_sum[j];
            for (size_t l = 0; l < static_cast<size_t>(dt); ++l) {
                double frac = static_cast<double>(l) / dt;
                double denom = s0 - frac * s0_d;
                loglik -= std::log(denom);
                for (size_t j = 0; j < D; ++j)
                    grad(0, j) -= (s0_x[j] - frac * s0d_x[j]) / denom;
            }
        }
    }

    CoxPartialLikelihood res;
    // C11: the MEAN over the number of events, not the raw sum. lifelines
    // and R report the sum; the factor is reported so a caller can
    // recover it.
    res.log_likelihood = loglik / n_events;
    for (size_t j = 0; j < D; ++j) grad(0, j) /= n_events;
    res.gradient = grad;
    res.n_events = n_events;
    return res;
}

}  // namespace utils
}  // namespace nn
