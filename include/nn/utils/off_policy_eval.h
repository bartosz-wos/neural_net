#ifndef NN_UTILS_OFF_POLICY_EVAL_H
#define NN_UTILS_OFF_POLICY_EVAL_H

#include "../core/tensor.h"
#include <cstddef>

// ===========================================================================
// OFF-POLICY EVALUATION: ESTIMATING A TARGET POLICY'S VALUE FROM LOGGED
// DATA COLLECTED UNDER A DIFFERENT POLICY.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-30)
// ---------------------------------------------------------------------------
//
//     grep -rinE '\bsnr\b|off.?policy|behavior.?polic|importance.?sampl' include/ tests/
//     -> 0 hits
//
// So the repo has NO off-policy machinery of any kind. What it HAD was:
//
//   * `utils/mmd_loss.h` -- a KERNEL two-sample statistic. No importance
//     weights, no policy, no value.
//   * `utils/conformal.h` -- calibrates a prediction that was ALREADY MADE.
//     It is not an estimator of a policy's value.
//   * `utils/cross_validation.h` -- resamples the SAME data. Off-policy
//     evaluation is the opposite: it combines data collected under ONE
//     policy with a DIFFERENT policy's action probabilities.
//   * `utils/sinkhorn.h` -- an entropic transport PLAN. A plan, not a value.
//
// **THERE IS NO `reward` CONCEPT ANYWHERE IN `include/nn` EITHER** (the same
// grep for `reward` over the headers returns 0 hits), which is why this
// module is MODEL-FREE OVER LOGGED DATA rather than a layer, an environment
// or a simulator. It is arithmetic over logged samples, which is exactly
// what makes it checkable against a closed-form oracle.
//
// ---------------------------------------------------------------------------
// THE MATHEMATICS
// ---------------------------------------------------------------------------
//
// Logged data: N samples (s_i, a_i, r_i) with a_i ~ pi_0(.|s_i) -- pi_0 is
// the BEHAVIOUR policy that collected the data -- and r_i ~ P(r|s_i,a_i).
// The TARGET policy is pi_1. The importance weight is
//
//     rho_i = pi_1(a_i|s_i) / pi_0(a_i|s_i)
//
// and the four estimators are
//
//     IPS  V = (1/N) sum_i rho_i r_i
//     SNIS V = (sum_i rho_i r_i) / (sum_i rho_i)
//     DM   V = (1/N) sum_i Vhat(s_i)
//     DR   V = (1/N) sum_i [ Vhat(s_i) + rho_i (r_i - Rhat(s_i,a_i)) ]
//
// where the reward model supplies the model-based target value
// Vhat(s) = sum_a pi_1(a|s) Rhat(s,a) and the per-sample logged-action
// value Rhat(s_i,a_i).
//
// ---------------------------------------------------------------------------
// THE THREE PLACES THIS IS EASY TO GET WRONG, AND WHY THE HEADER IS THIS
// LONG ABOUT THEM
// ---------------------------------------------------------------------------
//
//  1. **THE SNIS DENOMINATOR IS sum(rho), NOT N.** Using N silently returns
//     IPS -- the two estimators then agree on every fixture whose weights
//     happen to average to 1, and disagree everywhere else. The identity
//     V_SNIS == V_IPS / mean(rho) is exact and is what separates them.
//
//  2. **EVERY ESTIMATOR IS A MEAN OVER THE N SAMPLES.** A SUM is off by
//     exactly the factor N.
//
//  3. **THE DR CANCELLATION.** With rho(a) = pi_1(a)/pi_0(a),
//
//         E_{a~pi_0}[ Vhat(s) + rho(a) (r(a) - Rhat(s,a)) ]
//             = Vhat(s) + sum_a pi_1(a) (E[r|s,a] - Rhat(s,a))
//             = Vhat(s) + V_pi1 - sum_a pi_1(a) Rhat(s,a)
//             = V_pi1                            <- the Vhat TERM CANCELS
//
//     So DR is unbiased for the target value when EITHER the propensities
//     are exact (the first Vhat cancels) OR the reward model is exact (the
//     correction term vanishes identically). **THE TEST SUITE CHECKS BOTH
//     ARMS BY EXACT ENUMERATION over a finite action support** -- not by
//     Monte Carlo, so the identity either holds exactly or does not hold.
//     An implementation that drops the Vhat term, flips the correction's
//     sign, or pairs the wrong sample's values breaks one of the two arms.
//
// **A SECOND, SHARPER DISCRIMINATOR: SCALE INVARIANCE.** SNIS(c*rho) is
// SNIS(rho) for any c > 0, because the scale cancels between numerator and
// denominator. IPS(c*rho) is c * IPS(rho). A fixture that does not vary the
// SCALE of the weights cannot tell these two functions apart at all, so the
// suite scales the weights deliberately.
//
// ---------------------------------------------------------------------------
// THE CITATIONS. Verified 2026-10-30 by arXiv TITLE search, and then by
// scraping each abs page's `citation_title` / `citation_author` meta tags.
// Every title below matches VERBATIM and every author list matches IN
// ORDER. This repo's expansion queue has produced fourteen wrong arXiv
// identifiers and every one was a plausible-looking neighbour of the right
// paper, so every id here was scraped rather than recalled.
//
//   1604.00923  "Data-Efficient Off-Policy Policy Evaluation for
//               Reinforcement Learning". Thomas, Philip S.; Brunskill, Emma.
//               One-step OPE; the direct-method and doubly-robust baselines.
//   1511.03722  "Doubly Robust Off-policy Value Evaluation for
//               Reinforcement Learning". Jiang, Nan; Li, Lihong.
//               **The bandit DR estimator was transcribed from this paper's
//               own LaTeX source** (eq:dr):
//                   DR_I = V(s_I) + rho_I (r_I - R(s_I,a_I)),
//                   rho = pi_1(a|s)/pi_0(a|s),  V(s) = sum_a pi_1(a|s) R(s,a).
//   2006.10460  "Confident Off-Policy Evaluation and Selection through
//               Self-Normalized Importance Weighting". Kuzborskij, Ilja;
//               Vernade, Claire; György, András; Szepesvári, Csaba.
//               The self-normalised estimator.
//   1910.12809  "Minimax Weight and Q-Function Learning for Off-Policy
//               Evaluation". Uehara, Masatoshi; Huang, Jiawei; Jiang, Nan.
//               Why the effective sample size is the diagnostic that decides
//               whether the importance weights are usable at all.
//   2104.12820  "Universal Off-Policy Evaluation". Chandak, Yash; Niekum,
//               Scott; da Silva, Bruno Castro; Learned-Miller, Erik;
//               Brunskill, Emma; Thomas, Philip S.
//
// **NO arXiv ID IS ASSERTED FOR THE SNR METRIC.** The expansion queue's
// grep included `\bsnr\b`, but an arXiv title search for "sample noise
// ratio" returns 0 entries and a scan of all 1233 PMLR v162 titles finds no
// such paper, so no identifier and no DEFINITION is claimed for it.
//
// ---------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, STATED SO IT CANNOT CREEP IN
// ---------------------------------------------------------------------------
//
//   * SEQUENTIAL / MULTI-STEP estimators (Jiang & Li's per-decision DR with
//     gamma-discounting). They need trajectory storage, discounting and
//     per-step cumulative weights -- a different data layout, not a small
//     extension of this one.
//   * Any environment, simulator, replay buffer or `Layer` integration.
//   * Behaviour-policy ESTIMATION (propensity-model fitting). The caller
//     supplies the propensities.
//   * The variance of the estimators. The effective sample size IS shipped,
//     because it is the standard closed-form diagnostic for whether the
//     weights are usable; the standard errors are not asserted here.
//
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// All five estimators at once.
//
// `model_target_values` is Vhat(s_i) -- the model-based value of the
// TARGET policy at the logged state -- and `logged_action_values` is
// Rhat(s_i,a_i). A caller who has no reward model can still use the
// model-free estimators (`ope_ips_estimate`, `ope_self_normalized_ips`,
// `ope_effective_sample_size`) directly.
// ---------------------------------------------------------------------------
struct OPEReport {
    double ips = 0.0;   // the inverse-propensity-scoring estimate
    double snis = 0.0;  // the self-normalised estimate
    double dm = 0.0;    // the direct-method estimate
    double dr = 0.0;    // the doubly-robust estimate
    double ess = 0.0;   // the effective sample size of the weights
};

// ---------------------------------------------------------------------------
// The importance weight rho = pi_1(a|s) / pi_0(a|s).
//
//   target_prob    the probability of the action under the TARGET policy.
//   behavior_prob  the probability of the same action under the BEHAVIOUR
//                  policy that collected the data.
//
// Throws std::invalid_argument if `behavior_prob` is not positive (the
// ratio is undefined) or `target_prob` is negative (it is a probability).
// A target probability of exactly ZERO is legal and gives a weight of zero:
// the target policy never takes that action, so the sample carries no
// information about it.
// ---------------------------------------------------------------------------
double ope_importance_weight(double target_prob, double behavior_prob);

// ---------------------------------------------------------------------------
// The batched form of the importance weight.
//
// `target_probs` and `behavior_probs` must have the same number of
// elements. The result has the same shape as the inputs.
//
// **THE SHAPE IS IRRELEVANT AND THE VALUES ARE WHAT MATTER**: a (N,1) and
// a (1,N) tensor holding the same values are the same data, and the result
// keeps the input's shape.
//
// Throws std::invalid_argument on: an empty input; a mismatch in the
// number of elements; a non-positive behaviour probability; a negative
// target probability.
// ---------------------------------------------------------------------------
Tensor ope_importance_weights(const Tensor& target_probs,
                               const Tensor& behavior_probs);

// ---------------------------------------------------------------------------
// The inverse-propensity-scoring estimate.
//
//     V_IPS = (1/N) sum_i rho_i r_i
//
// Unbiased for the target policy's value when the propensities are exact.
// It is NOT scale invariant: scaling every weight by c scales the estimate
// by c.
//
// Throws std::invalid_argument on: an empty input; a mismatch in the
// number of samples between the rewards and the weights; a negative weight.
// ---------------------------------------------------------------------------
double ope_ips_estimate(const Tensor& rewards, const Tensor& weights);

// ---------------------------------------------------------------------------
// The self-normalised importance-sampling estimate.
//
//     V_SNIS = (sum_i rho_i r_i) / (sum_i rho_i)
//
// **THE DENOMINATOR IS sum(rho), NOT N.** It is the biased-but-lower-
// variance member of the importance-sampling family, and it is SCALE
// INVARIANT: scaling every weight by c leaves the estimate unchanged.
//
// Throws std::invalid_argument on: an empty input; a mismatch in the
// number of samples; a negative weight; a total weight of zero (the
// denominator would be zero).
// ---------------------------------------------------------------------------
double ope_self_normalized_ips(const Tensor& rewards, const Tensor& weights);

// ---------------------------------------------------------------------------
// The direct-method estimate.
//
//     V_DM = (1/N) sum_i Vhat(s_i)
//
// It needs NO weights at all -- only the model-based value of the target
// policy at each logged state.
//
// Throws std::invalid_argument on an empty input.
// ---------------------------------------------------------------------------
double ope_direct_method_estimate(const Tensor& model_target_values);

// ---------------------------------------------------------------------------
// The doubly-robust estimate.
//
//     V_DR = (1/N) sum_i [ Vhat(s_i) + rho_i (r_i - Rhat(s_i,a_i)) ]
//
//   model_target_values   Vhat(s_i), the model-based value of the TARGET
//                         policy at the logged state.
//   logged_action_values  Rhat(s_i,a_i), the model's value at the action
//                         that was actually taken.
//
// It is unbiased for the target policy's value when EITHER the propensities
// are exact OR the reward model is exact -- see the header's cancellation
// derivation.
//
// **THE TWO ARMS ARE NOT INTERCHANGEABLE, AND THE MEASUREMENT SHOWS WHY.**
// Dropping the Vhat term breaks BOTH arms. But flipping the correction's
// sign, or dropping the correction's weight, breaks ONLY ARM A -- because
// in ARM B the reward model is EXACT, which makes r_i - q_i identically
// ZERO for every sample, and a term that is identically zero cannot be
// changed by anything done to it. **ARM B IS THEREFORE STRUCTURALLY
// INSENSITIVE TO THE CORRECTION TERM**, and ARM A is the arm that pins it.
// The two arms are complementary: ARM A pins the correction, ARM B pins the
// Vhat term.
//
// Throws std::invalid_argument on: an empty input; a mismatch in the number
// of samples between any two of the four vectors; a negative weight.
// ---------------------------------------------------------------------------
double ope_doubly_robust_estimate(const Tensor& rewards, const Tensor& weights,
                                  const Tensor& model_target_values,
                                  const Tensor& logged_action_values);

// ---------------------------------------------------------------------------
// The effective sample size of a set of importance weights.
//
//     ESS = (sum_i rho_i)^2 / sum_i rho_i^2
//
// It is the number of "effective" equally-weighted samples the weights are
// worth, and it is the diagnostic that says whether the importance weights
// are usable at all: ESS <= N, with equality iff all the weights are equal,
// and a single dominant weight drives it towards 1. It is SCALE INVARIANT.
//
// Throws std::invalid_argument on: an empty input; a negative weight; a
// total weight of zero (the numerator and denominator are both zero).
// ---------------------------------------------------------------------------
double ope_effective_sample_size(const Tensor& weights);

// ---------------------------------------------------------------------------
// All five estimates at once, from one call.
//
// The four vectors must all have the same number of samples.
//
// Throws std::invalid_argument on the same conditions as the individual
// estimators.
// ---------------------------------------------------------------------------
OPEReport ope_evaluate(const Tensor& rewards, const Tensor& weights,
                       const Tensor& model_target_values,
                       const Tensor& logged_action_values);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_OFF_POLICY_EVAL_H
