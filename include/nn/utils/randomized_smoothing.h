#ifndef NN_UTILS_RANDOMIZED_SMOOTHING_H
#define NN_UTILS_RANDOMIZED_SMOOTHING_H

#include "../core/tensor.h"
#include "../core/model.h"
#include <cstddef>
#include <functional>
#include <vector>

// ===========================================================================
// Randomized smoothing — certified adversarial robustness
//
//   Jeremy M. Cohen, Elan Rosenfeld, J. Zico Kolter — "Certified Adversarial
//   Robustness via Randomized Smoothing", ICML 2019, arXiv:1902.02918.
//
// CITATION NOTE. The expansion queue proposed arXiv:1612.00483 for this paper.
// That ID is WRONG: the arXiv API returns "A Large-Scale Spectroscopic Survey
// of Methanol and OH Line Emission from the Galactic Center" (Cotton &
// Yusef-Zadeh, astro-ph.GA) for 1612.00483, verified 2026-10-14. The correct ID
// is 1902.02918, also verified against the arXiv API that date.
//
// THE IDEA, AND WHY IT IS THE COMPLEMENT OF adversarial.h. A gradient attack
// asks "how far can a δ move this model?" and answers by SEARCH — it finds one
// perturbation and cannot say whether a better one exists. Randomized smoothing
// inverts the question: wrap the classifier in a distribution, and the answer
// becomes a CLOSED FORM. With ε ~ N(0, σ²I),
//
//     g(x) = argmax_c P(f(x + ε) = c)                              (paper Eq. 1)
//
// and Theorem 1 gives, for ANY f whatsoever, that g is provably constant on
// the whole ℓ2 ball of radius
//
//     R = σ/2 · (Φ⁻¹(p_A) − Φ⁻¹(p_B))                                (Theorem 1)
//
// This is a guarantee, not a measurement: adversarial.{h,cpp} can tell you it
// found a δ at radius 3.0 and be wrong that nothing works at 3.1.
//
// WHY IT LIVES HERE AND NOT IN layers/: like adversarial.h and
// knowledge_distillation.h, this is model-free. The only interface to a network
// is a callback `int -> int` (the base classifier), so every statistical core
// is testable with a synthetic oracle and no network at all.
//
// ===========================================================================
// Semantics, and the rules a reimplementation gets wrong
// ===========================================================================
//
// (D1) NOISE IS ADDITIVE AND `sigma` IS THE STANDARD DEVIATION. Eq. (1) is
//      `f(x + ε)` with `ε ~ N(0, σ²I)`. A multiplicative or per-dimension-varying
//      noise law is a different method; so is treating σ as the variance.
//
// (D2) THE RADIUS REQUIRES p_lower > 1/2, STRICTLY, OR THE CALLER ABSTAINS.
//      The pseudocode reads `if p_A > 1/2 return ... else return ABSTAIN`. At
//      p_lower == 1/2 the radius is exactly 0, which certifies nothing. Note
//      this bites in practice and is not an edge case: at k = n (a unanimous
//      sample) the Clopper-Pearson bound at α = 0.001 is only 0.072930496291540997
//      for n = 1000 — still below 1/2, so a unanimous 1000-sample run ABSTAINS.
//      This is why the paper needs n = 100,000 to certify large radii, and the
//      test suite asserts the abstention rather than papering over it.
//
// (D3) PREDICT'S HYPOTHESIS TEST USES ONLY THE TOP TWO COUNTS, with
//      `n = nA + nB` — NOT the full sample count. Pseudocode:
//      `BINOMPVALUE(nA, nA + nB, 0.5)`. Passing the whole-sample `n` (as a
//      ClassDetector might) is a different statistic with a different p-value
//      at the same α, and it silently abstains on fixtures that should not.
//
// (D4) THE TEST IS TWO-SIDED AT p = 0.5, AND ABSTAIN IFF p_value > alpha.
//      The paper's Appendix C invokes Hung & Fithian (2019). scipy's
//      `binomtest(nA, nA+nB, 0.5).pvalue` is the oracle, and it matches
//      `min(1, 2·min(P(X<=k), P(X>=k)))` exactly. A one-sided implementation
//      returns the smaller tail and answers twice as eagerly; the suite pins
//      `binom_pvalue_half(nA,nB) == binom_pvalue_half(nB,nA)`, which a one-sided
//      form cannot satisfy in general.
//
// (D5) CERTIFY'S GUESSING SAMPLES ARE DISCARDED. Appendix C.2: "first, use n0
//      samples ... to take a guess ĉA. Second, use n samples ... to obtain some
//      p_A". The Clopper-Pearson bound is formed from the SECOND round only.
//      Pooling the two rounds gives n = n0 + n samples with a different — and
//      unsound, since the class was chosen using them — statistic. `certify`
//      therefore returns `count == num_samples`, never `n0 + num_samples`.
//
// (D6) THE CLOPPER-PEARSON LOWER BOUND IS `Beta^-1(alpha; k, n-k+1)` — AND THE
//      PAPER'S OWN CITED SNIPPET LOOKS LIKE IT SHOULD NOT BE. Appendix C.2 says
//
//          proportion_confint(k, n, alpha=2*alpha, method="beta")[0]
//
//      which reads like a 2α (i.e. 98%-confidence) bound. It is not, and the
//      reason is inside statsmodels: with the default two-sided alternative it
//      halves alpha internally (`proportion.py:208-210`, `alpha = alpha/2.0`)
//      and the "beta" branch is literally `stats.beta.ppf(alpha, k, n-k+1)`
//      (`:271`). The `2*` cancels the halving, so the real formula is the
//      one-sided (1−α) bound above.
//
//      This was verified numerically rather than by reading, because the two
//      forms differ by a clean factor of two in the bound's DEFINING property:
//      at `p = Beta^-1(alpha; k, n-k+1)` the tail `P(X >= k)` is exactly α
//      (measured 0.0500000000 on every fixture at α=0.05); at
//      `Beta^-1(2*alpha; ...)` it is 0.1000000000. A bound that is twice too
//      optimistic is a certification that is wrong half the time — the exact
//      failure this method is supposed to eliminate.
//
// (D7) AT k = 0 THE BOUND IS EXACTLY 0.0, NOT NaN. `scipy.stats.beta.ppf(a, 0, n+1)`
//      returns NaN because Beta(0, ·) is degenerate. The Clopper–Pearson lower
//      bound at zero observed successes is exactly zero, and the correct
//      behaviour is a radius of 0 (hence an abstention under D2), not a NaN
//      that poisons a whole batch.
//
// (D8) THEOREM 1'S GENERAL FORM IS `σ/2 · (Φ⁻¹(p_A) − Φ⁻¹(p_B))`. It collapses
//      to `σ·Φ⁻¹(p_A)` ONLY because `CERTIFY` sets `p_B = 1 − p_A`
//      (Appendix C.2: "the mass of f(x+ε) not allocated to c_A is ... allocated
//      entirely to one runner-up class. Therefore, the quantity 1 − p_A is a
//      reasonably tight upper bound on p_B"). Both forms are public; the suite
//      asserts they agree when p_B = 1−p_A and MUST DIFFER when it is not, so
//      the symmetric one can never be a copy of the general one.
//
// (D9) A GIVEN `seed` MUST PRODUCE BIT-IDENTICAL COUNTS. Randomised smoothing
//      results are compared against each other in every paper table, and the
//      noise is drawn with an EXPLICIT Box–Muller transform on a local
//      `std::mt19937` rather than `std::normal_distribution` — the distribution
//      objects are not portable across standard library implementations, so
//      reproducibility would not survive a toolchain change. (The repo's
//      `Tensor::random` uses bare `rand()` and is deliberately not used here.)
// ===========================================================================

// ---------------------------------------------------------------------------
// The base-classifier oracle: given an input, return the predicted CLASS
// INDEX. This is the entire interface between this file and any network —
// deliberately narrower than `adversarial.h`'s `GradientFn`, because a class
// index is all PREDICT/CERTIFY ever consume.
// ---------------------------------------------------------------------------
using BaseClassifierFn = std::function<int(const Tensor&)>;

// ---------------------------------------------------------------------------
// The standard normal CDF, Phi(x) = 0.5·erfc(-x/sqrt(2)).
// Exact to the last ulp of `erfc` (matches scipy.stats.norm.cdf to < 1e-15
// over [-8, 8]).
// ---------------------------------------------------------------------------
double normal_cdf(double x);

// ---------------------------------------------------------------------------
// Phi^-1(p), by bisection on `normal_cdf`.
//
// WHY BISECTION AND NOT WICHURA'S AS241. AS241 is the standard rational
// approximation and is what most libraries ship, but measured against
// scipy.stats.norm.ppf it loses ~1e-6 RELATIVE accuracy in the tails, which
// is exactly where a certification lives (p_A is near 1, so Φ^-1(p_A) is
// large). Bisection on `erfc` matches scipy to 4.5e-16 relative across
// p ∈ [1e-12, 1−1e-12] for the cost of ~60 `erfc` calls.
//
// Throws std::invalid_argument unless 0 < p < 1. (p = 1 is excluded on
// purpose: Φ^-1(1) = +∞, and a caller reaching it has an over-confident
// Clopper–Pearson bound rather than a valid input.)
// ---------------------------------------------------------------------------
double normal_quantile(double p);

// ---------------------------------------------------------------------------
// The one-sided (1−alpha) Clopper–Pearson LOWER confidence bound on a binomial
// proportion, given k successes in n trials: `Beta^-1(alpha; k, n-k+1)` (D6).
//
// Implemented as a continued-fraction regularised incomplete beta I_x(a,b)
// followed by a bisection inverse; measured to agree with scipy.stats.beta.ppf
// to 1.8e-13 worst-case relative over the full fixture set.
//
// Returns exactly 0.0 when k == 0 (D7). Throws std::invalid_argument for
// n == 0, k < 0, k > n, or alpha outside (0, 1).
// ---------------------------------------------------------------------------
double clopper_pearson_lower(int k, int n, double alpha);

// ---------------------------------------------------------------------------
// Two-sided binomial-test p-value under H0: p = 0.5, given `n_a` observations
// of one outcome and `n_b` of the other (D4). Equals scipy.stats.binomtest
// exactly.
//
// Equivalently `2·min(P(X<=n_a), P(X>=n_a))` capped at 1. Returns 1.0 when
// n_a + n_b == 0 (a vacuous test is not evidence). Throws
// std::invalid_argument if either count is negative.
// ---------------------------------------------------------------------------
double binom_pvalue_half(int n_a, int n_b);

// ---------------------------------------------------------------------------
// Theorem 1's certified radius in its GENERAL two-class form (D8):
//     R = sigma/2 · (Phi^-1(p_a) − Phi^-1(p_b))
//
// Throws std::invalid_argument unless sigma > 0, 0 < p_a <= 1, 0 <= p_b <= 1,
// and p_a >= p_b (a "top" class less likely than the runner-up is a bug in
// the caller, and silently returning a negative radius would hide it).
// ---------------------------------------------------------------------------
double certified_radius_from_probs(double sigma, double p_a, double p_b);

// ---------------------------------------------------------------------------
// Theorem 1's radius under CERTIFY's assumption p_b = 1 − p_a (D8):
//     R = sigma · Phi^-1(p_lower)
//
// This is the form `certify` returns. Note it is NOT a bound on p_b; it
// ASSUMES the runner-up holds all the remaining mass. It is valid only for
// p_lower > 1/2 (D2); the value is defined at p_lower == 0.5 (radius 0) and
// rejected outside (0, 1). Throws std::invalid_argument unless sigma > 0.
// ---------------------------------------------------------------------------
double certified_radius(double sigma, double p_lower);

// ---------------------------------------------------------------------------
// Theorem 6's lower bound on the true certified test-set accuracy, given that
// CERTIFY returned the correct label AND a radius > r on `y` of `m` examples,
// with per-example failure probability `alpha` and overall confidence `rho`.
//
//     (1/(1−alpha)) · (y − alpha·m − sqrt(max(0, 2·m·alpha·(1−alpha)·ln(1/rho)
//                                         − ln(1/rho)/3))) / m
//
// (Appendix D. The algebra is taken from the PROOF on p. 24, because the
// displayed Eq. 16 is mangled by the PDF's fraction layout.)
//
// This is a STATISTIC, not a report: it can return a NEGATIVE number (no
// certified examples) and can slightly EXCEED 1 (a perfect certification at
// small m). Clamping is the caller's decision, and clamping inside would
// silently inflate every small-sample result. Throws std::invalid_argument if
// m < 0, y < 0, y > m, or alpha/rho outside (0, 1). Returns 0.0 when m == 0.
// ---------------------------------------------------------------------------
double certified_accuracy_lower_bound(int y, int m, double alpha, double rho);

// ---------------------------------------------------------------------------
// Appendix C's SAMPLEUNDERNOISE: draw `num_samples` noise vectors
// ε_i ~ N(0, sigma²I), run x + ε_i through the base classifier, and return the
// class counts as `counts[c] = #{i : f(x + ε_i) = c}` (D1, D9).
//
// `seed` makes the draw reproducible (D9); the default 0 is deterministic.
//
// Throws std::invalid_argument unless sigma >= 0 and num_samples >= 0.
// (num_samples == 0 legitimately returns an empty vector.) Returns the vector
// sized to max class index seen + 1, so `counts[c]` is valid for every class the
// classifier actually emitted.
// ---------------------------------------------------------------------------
std::vector<int> sample_under_noise(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples,
                                    unsigned seed = 0);

// ---------------------------------------------------------------------------
// Appendix C's PREDICT (Proposition 1): draw `num_samples` noisy samples, take
// the top two counts nA >= nB, and return ĉA iff the two-sided binomial test
// of "nA ~ Binomial(nA+nB, 1/2)" has p-value <= alpha; otherwise abstain (D3,
// D4).
//
// Proposition 1's guarantee: with probability at least 1−alpha over PREDICT's
// randomness, PREDICT either abstains or returns g(x).
// ---------------------------------------------------------------------------
struct SmoothedPrediction {
    bool abstained = true;
    int class_index = -1;
    int n_a = 0;         // the TOP count
    int n_b = 0;         // the runner-up count (D3: n_a + n_b may be < num_samples)
    double p_value = 1.0;
};

SmoothedPrediction smoothed_predict(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples, double alpha,
                                    unsigned seed = 0);

// ---------------------------------------------------------------------------
// Configuration for CERTIFY. The defaults are the paper's own CIFAR-10 setting:
// sigma = 0.25 and alpha = 0.001 (§4.2, p. 30), n0 = 100 (Appendix J.2), and
// n = 1000 (Fig. 8, right, where n ranges up to 10,000).
//
// NOTE n = 1000 DOES NOT CERTIFY A LARGE RADIUS. Appendix D makes this
// quantitative: for a perfectly-separable example, n = 1000 at alpha = 0.001
// certifies at most sigma·Φ^-1(α^(1/1000)). Raise `num_samples` for real work;
// the paper used 100,000.
// ---------------------------------------------------------------------------
struct SmoothingConfig {
    double sigma = 0.25;
    int num_guess_samples = 100;
    int num_samples = 1000;
    double alpha = 0.001;
    unsigned seed = 0;
};

struct CertifiedPrediction {
    bool abstained = true;
    int class_index = -1;
    int count = 0;        // successes in the SECOND round only (D5)
    double p_lower = 0.0; // the Clopper–Pearson bound
    double radius = 0.0;  // 0 when abstained
};

// ---------------------------------------------------------------------------
// Appendix C's CERTIFY (Proposition 2): sample `num_guess_samples` times to
// guess ĉA (those samples are DISCARDED — D5), then sample `num_samples` more
// times, form the Clopper–Pearson lower bound p_A on the count of ĉA, and
// return (ĉA, sigma·Φ^-1(p_A)) iff p_A > 1/2; otherwise abstain (D2).
//
// Proposition 2's guarantee: with probability at least 1−alpha, if CERTIFY
// returns a class and a radius then g(x+δ) = ĉA for all ||δ||₂ < R.
//
// Throws std::invalid_argument unless sigma >= 0, num_guess_samples >= 1,
// num_samples >= 1, and 0 < alpha < 1.
// ---------------------------------------------------------------------------
CertifiedPrediction certify(const BaseClassifierFn& f, const Tensor& x,
                            const SmoothingConfig& cfg);

// ---------------------------------------------------------------------------
// Train `model` under randomized smoothing (the paper's §4 / Appendix J.2
// outer minimisation): each example is presented with Gaussian noise of
// standard deviation `cfg.sigma`, and the loss is the repo's canonical
// `CrossEntropyLoss` (utils/losses.h) on INTEGER class indices — the same
// convention `adversarial_train` uses.
//
// Appendix J.2 records that the authors' PyTorch implementation added Gaussian
// noise and then STANDARDISED the data (per-channel mean/std), so the network
// saw standardised inputs with noise applied in the original coordinates. This
// function applies noise directly and does not standardise; supply a
// normalisation layer in the model if you need that regime.
//
// Throws std::invalid_argument on a model with no layers, X.rows != y.rows, or
// an out-of-range `cfg`. `X` holds one example per row; `y` holds integer class
// indices in a column.
// ---------------------------------------------------------------------------
void smoothing_train(Model& model, const Tensor& X, const Tensor& y,
                     const SmoothingConfig& cfg, double learning_rate, int epochs);

#endif  // NN_UTILS_RANDOMIZED_SMOOTHING_H