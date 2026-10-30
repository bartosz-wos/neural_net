#include "off_policy_eval.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// utils/off_policy_eval.cpp -- the off-policy evaluation estimators.
//
// THE SPECIFICATION is tests/off_policy_eval_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed. It was green on its
// first execution, so its discriminating power was MEASURED rather than
// narrated: tests/off_policy_eval_mutate_oracle.py breaks the oracle in
// seven ways and requires that it fail each time (7/7 caught).
//
// The three decisions the test suite pins:
//
//   1. THE SNIS DENOMINATOR IS sum(rho), NOT N. Using N silently returns
//      IPS.
//   2. EVERY ESTIMATOR IS A MEAN OVER THE N SAMPLES.
//   3. THE DR ESTIMATOR KEEPS THE Vhat(s) TERM, AND THE CORRECTION HAS THE
//      SIGN rho * (r - Rhat). Dropping or flipping either breaks one of
//      the two arms of double robustness.
//
// See the header for the derivations and for the citations.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// The values of a Tensor, in row-major order, as a flat vector.
//
// The estimators care about the VALUES, not the shape: a (N,1) and a (1,N)
// tensor holding the same values are the same sample. This helper is the
// ONLY place a sample is turned into a list of values.
// ---------------------------------------------------------------------------
std::vector<double> values_of(const Tensor& t) {
    std::vector<double> v;
    v.reserve(t.rows * t.cols);
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            v.push_back(t(i, j));
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// The guards. EVERY message here is DISTINCT.
//
// A substring that appears in two different messages discriminates nothing
// between them, and this repo has hit that trap several separate times
// (each time the assertion searched for a word the OTHER guard's message
// also contained, so the mutant survived). The suite therefore asserts each
// guard's own message, not merely that something was thrown.
//
// The guard messages are also distinct IN KIND:
//   * "empty"                     -- the input has no samples at all.
//   * "same number of samples"    -- the vectors disagree in length.
//   * "behaviour probability"     -- the denominator of the ratio.
//   * "target probability"        -- the numerator of the ratio.
//   * "weight"                    -- a negative importance weight.
//   * "total weight"              -- a zero denominator.
// ---------------------------------------------------------------------------
void require_non_empty(const Tensor& t, const char* who) {
    if (t.rows == 0 || t.cols == 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": the sample vector is empty");
    }
}

void require_same_length(const std::vector<double>& a,
                         const std::vector<double>& b, const char* who) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::string(who) +
            ": every input must have the same number of samples");
    }
}

void require_non_negative_weights(const std::vector<double>& w, const char* who) {
    for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] < 0.0) {
            throw std::invalid_argument(
                std::string(who) + ": every importance weight must be "
                "non-negative");
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// ope_importance_weight -- the importance weight rho = pi_1 / pi_0.
// ---------------------------------------------------------------------------
double ope_importance_weight(double target_prob, double behavior_prob) {
    if (!(behavior_prob > 0.0)) {
        throw std::invalid_argument(
            "ope_importance_weight: the behaviour probability must be "
            "positive");
    }
    if (target_prob < 0.0) {
        throw std::invalid_argument(
            "ope_importance_weight: the target probability must be "
            "non-negative");
    }
    return target_prob / behavior_prob;
}

// ---------------------------------------------------------------------------
// ope_importance_weights -- the batched form.
// ---------------------------------------------------------------------------
Tensor ope_importance_weights(const Tensor& target_probs,
                               const Tensor& behavior_probs) {
    require_non_empty(target_probs, "ope_importance_weights");
    require_non_empty(behavior_probs, "ope_importance_weights");
    if (target_probs.rows * target_probs.cols !=
        behavior_probs.rows * behavior_probs.cols) {
        throw std::invalid_argument(
            "ope_importance_weights: every input must have the same number "
            "of samples");
    }

    Tensor out(target_probs.rows, target_probs.cols);
    for (size_t i = 0; i < target_probs.rows; ++i) {
        for (size_t j = 0; j < target_probs.cols; ++j) {
            const double tp = target_probs(i, j);
            const double bp = behavior_probs(i, j);
            if (!(bp > 0.0)) {
                throw std::invalid_argument(
                    "ope_importance_weights: the behaviour probability must "
                    "be positive");
            }
            if (tp < 0.0) {
                throw std::invalid_argument(
                    "ope_importance_weights: the target probability must be "
                    "non-negative");
            }
            out(i, j) = tp / bp;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// ope_ips_estimate -- (1/N) sum_i rho_i r_i.
//
// The 1/N is the MEAN over the samples; a sum is off by exactly the factor
// N. The estimate is NOT scale invariant, which is what the suite checks
// against SNIS.
// ---------------------------------------------------------------------------
double ope_ips_estimate(const Tensor& rewards, const Tensor& weights) {
    require_non_empty(rewards, "ope_ips_estimate");
    require_non_empty(weights, "ope_ips_estimate");

    const std::vector<double> r = values_of(rewards);
    const std::vector<double> w = values_of(weights);
    require_same_length(r, w, "ope_ips_estimate");
    require_non_negative_weights(w, "ope_ips_estimate");

    double total = 0.0;
    for (size_t i = 0; i < r.size(); ++i) total += w[i] * r[i];
    return total / static_cast<double>(r.size());
}

// ---------------------------------------------------------------------------
// ope_self_normalized_ips -- sum_i rho_i r_i / sum_i rho_i.
//
// **THE DENOMINATOR IS sum(rho), NOT N.** Using N silently returns IPS.
//
// Because the scale of the weights cancels between the numerator and the
// denominator, this estimator is SCALE INVARIANT -- the property the suite
// uses to tell it apart from IPS.
// ---------------------------------------------------------------------------
double ope_self_normalized_ips(const Tensor& rewards, const Tensor& weights) {
    require_non_empty(rewards, "ope_self_normalized_ips");
    require_non_empty(weights, "ope_self_normalized_ips");

    const std::vector<double> r = values_of(rewards);
    const std::vector<double> w = values_of(weights);
    require_same_length(r, w, "ope_self_normalized_ips");
    require_non_negative_weights(w, "ope_self_normalized_ips");

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < r.size(); ++i) {
        num += w[i] * r[i];
        den += w[i];
    }
    if (den == 0.0) {
        throw std::invalid_argument(
            "ope_self_normalized_ips: the total weight is zero");
    }
    return num / den;
}

// ---------------------------------------------------------------------------
// ope_direct_method_estimate -- (1/N) sum_i Vhat(s_i).
//
// It needs no weights at all.
// ---------------------------------------------------------------------------
double ope_direct_method_estimate(const Tensor& model_target_values) {
    require_non_empty(model_target_values, "ope_direct_method_estimate");

    const std::vector<double> v = values_of(model_target_values);
    double total = 0.0;
    for (size_t i = 0; i < v.size(); ++i) total += v[i];
    return total / static_cast<double>(v.size());
}

// ---------------------------------------------------------------------------
// ope_doubly_robust_estimate -- (1/N) sum_i [Vhat(s_i) + rho_i (r_i - Rhat_i)].
//
// THE Vhat TERM IS LOAD-BEARING. With rho(a) = pi_1(a)/pi_0(a),
//
//     E_{a~pi_0}[ Vhat(s) + rho(a)(r(a) - Rhat(s,a)) ]
//         = Vhat(s) + sum_a pi_1(a)(E[r|s,a] - Rhat(s,a))
//         = Vhat(s) + V_pi1 - sum_a pi_1(a) Rhat(s,a)
//         = V_pi1
//
// so the Vhat term cancels exactly against the model's own target-policy
// average. Dropping it (or flipping the correction's sign) breaks one of
// the two arms of double robustness.
// ---------------------------------------------------------------------------
double ope_doubly_robust_estimate(const Tensor& rewards, const Tensor& weights,
                                  const Tensor& model_target_values,
                                  const Tensor& logged_action_values) {
    require_non_empty(rewards, "ope_doubly_robust_estimate");
    require_non_empty(weights, "ope_doubly_robust_estimate");
    require_non_empty(model_target_values, "ope_doubly_robust_estimate");
    require_non_empty(logged_action_values, "ope_doubly_robust_estimate");

    const std::vector<double> r = values_of(rewards);
    const std::vector<double> w = values_of(weights);
    const std::vector<double> v = values_of(model_target_values);
    const std::vector<double> q = values_of(logged_action_values);

    require_same_length(r, w, "ope_doubly_robust_estimate");
    require_same_length(r, v, "ope_doubly_robust_estimate");
    require_same_length(r, q, "ope_doubly_robust_estimate");
    require_non_negative_weights(w, "ope_doubly_robust_estimate");

    double total = 0.0;
    for (size_t i = 0; i < r.size(); ++i) {
        total += v[i] + w[i] * (r[i] - q[i]);
    }
    return total / static_cast<double>(r.size());
}

// ---------------------------------------------------------------------------
// ope_effective_sample_size -- (sum rho)^2 / sum rho^2.
//
// The number of "effective" equally-weighted samples the weights are worth.
// ESS <= N, with equality iff all the weights are equal, and a single
// dominant weight drives it towards 1. It is SCALE INVARIANT.
// ---------------------------------------------------------------------------
double ope_effective_sample_size(const Tensor& weights) {
    require_non_empty(weights, "ope_effective_sample_size");

    const std::vector<double> w = values_of(weights);
    require_non_negative_weights(w, "ope_effective_sample_size");

    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        sum += w[i];
        sum_sq += w[i] * w[i];
    }
    if (sum == 0.0) {
        throw std::invalid_argument(
            "ope_effective_sample_size: the total weight is zero");
    }
    return sum * sum / sum_sq;
}

// ---------------------------------------------------------------------------
// ope_evaluate -- all five at once.
// ---------------------------------------------------------------------------
OPEReport ope_evaluate(const Tensor& rewards, const Tensor& weights,
                       const Tensor& model_target_values,
                       const Tensor& logged_action_values) {
    OPEReport rep;
    rep.ips = ope_ips_estimate(rewards, weights);
    rep.snis = ope_self_normalized_ips(rewards, weights);
    rep.dm = ope_direct_method_estimate(model_target_values);
    rep.dr = ope_doubly_robust_estimate(rewards, weights, model_target_values,
                                         logged_action_values);
    rep.ess = ope_effective_sample_size(weights);
    return rep;
}

}  // namespace utils
}  // namespace nn
