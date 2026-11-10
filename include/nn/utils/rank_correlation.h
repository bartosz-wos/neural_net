#ifndef NN_UTILS_RANK_CORRELATION_H
#define NN_UTILS_RANK_CORRELATION_H

#include "../core/tensor.h"
#include <cstddef>

// ===========================================================================
// RANK CORRELATION AND THE GAUSSIAN COPULA
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-10)
// ---------------------------------------------------------------------------
//
//     grep -rinE 'copula|rank.?correl|spearman|kendall' include/ tests/
//     -> 0 content hits
//
//     grep -rinE '\bpearson\b|\bcorrelation\b' include/nn/utils/*.h
//     -> only "Clopper-Pearson" (a BINOMIAL BOUND, a different object)
//
// So the repo has ~250 layers and 60+ utility files and NO correlation
// machinery of any kind: not Pearson's r, not Spearman's rho, not Kendall's
// tau, and nothing copula-shaped.
//
// What it HAS is adjacent but DIFFERENT:
//
//   * `utils/metrics.h` -- ROC-AUC, AP, NDCG, MAP, MRR. The RANKING
//     QUALITY of a model against labels. Not the AGREEMENT BETWEEN TWO
//     RANKINGS, which is what a rank correlation measures.
//   * `utils/self_supervised_losses.h` -- a cross-correlation MATRIX for the
//     Barlow-Twins objective. A Gram matrix, not a rank statistic.
//   * `utils/linear_algebra.h` -- the Frechet distance. A distance between
//     two GAUSSIANS, again not a rank statistic.
//
// ---------------------------------------------------------------------------
// WHAT THIS MODULE IS FOR
// ---------------------------------------------------------------------------
//
// "Do these two rankings agree?" is a question the repo currently cannot
// answer at all, and it is asked constantly: two models' output orderings,
// a feature-importance ranking against a ground truth, an ensemble's
// members against each other, two hyperparameter sweeps' trial orderings.
//
// And "is this dependence driven by the tails or by the bulk?" is the
// question a COPULA statistic answers, and a linear correlation cannot.
//
// ---------------------------------------------------------------------------
// THE THREE CONVENTIONS, WHICH ARE THE WHOLE GAME HERE
// ---------------------------------------------------------------------------
//
// THREE SEPARATE TIE CONVENTIONS ARE IN PLAY AND THEY ARE EASY TO CONFLATE.
// Each is stated here because each is individually invisible on a fixture
// that does not exercise it.
//
//   1. RANKS ARE MID-RANKS. A tied group receives the AVERAGE of the ranks
//      it spans. This is R's `rank(..., ties.method="average")` and
//      scipy's `rankdata(method="average")`. NOT the first rank of the
//      group, NOT the last, NOT the dense rank.
//
//   2. KENDALL'S TAU-B'S TIE CORRECTION. With
//          tot  = n(n-1)/2        (all pairs)
//          xtie = pairs tied in x
//          ytie = pairs tied in y
//          ntie = pairs tied in BOTH x and y
//          dis  = discordant pairs
//      the numerator is
//          con_minus_dis = tot - xtie - ytie + ntie - 2*dis
//      and the denominator is
//          sqrt((tot - xtie) * (tot - ytie)).
//
//      **THE JOINT TIES COUNT IN *BOTH* xtie AND ytie.** This is the single
//      most common way to get tau-b wrong, and it is INVISIBLE on a
//      fixture with no ties. Measured: the counting-them-in-neither variant
//      gives -0.045454545454545456 where scipy gives
//      -0.04761904761904762.
//
//   3. THE NORMAL-SCORES PLOTTING POSITION. z_i = Phi^-1((r_i - 0.5)/n) with
//      r_i the mid-rank. The (r - 0.5)/n position is the Hazen position
//      used by the van der Waerden / copula transform. It is NOT
//      Blomqvist's (r - 0.375)/(n + 0.25) and NOT the Weibull r/(n+1).
//
// ---------------------------------------------------------------------------
// THE CLASSIC SPEARMAN SHORTCUT IS WRONG WITH TIES
// ---------------------------------------------------------------------------
//
// The textbook shortcut
//
//     rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
//
// is EXACT only when there are NO TIES AT ALL. With ties the DEFINITION is
// Pearson's r of the mid-ranks, and the two DIFFER. The shortcut is not
// used here; `spearman_rho` computes the definition.
//
// ---------------------------------------------------------------------------
// THE FOUR EXACT POPULATION CORRESPONDENCES
// ---------------------------------------------------------------------------
//
// For a bivariate GAUSSIAN with correlation rho -- equivalently, the
// Gaussian copula with parameter rho -- the POPULATION rank statistics are
//
//     Kendall's tau   tau   = (2/pi) * arcsin(rho)
//     Spearman's rho   rho_S = (6/pi) * arcsin(rho/2)
//
// and their inverses are
//
//     rho = sin(pi * tau / 2)
//     rho = 2 * sin(pi * rho_S / 6)
//
// Each closed form is verified in the oracle against an INDEPENDENT
// NUMERICAL COMPUTATION that does not contain it: `tau` against
// E[sign(U)sign(V)] by conditional quadrature of the bivariate normal
// concordance probability, and `rho_S` against 12*E[Phi(X)Phi(Y)] - 3 by
// Gauss-Hermite quadrature. A closed form checked against a quadrature that
// does not contain it is evidence; a closed form checked against itself is
// not.
//
// **THESE ARE POPULATION RELATIONSHIPS.** The SAMPLE statistics converge to
// them as the sample grows; they are not equal to them at finite n, and
// the suite asserts the CONVERGENCE rather than an equality.
//
// ---------------------------------------------------------------------------
// WHY THE COPULA HALF IS WORTH BUILDING
// ---------------------------------------------------------------------------
//
// **A COPULA STATISTIC IS INVARIANT UNDER ANY STRICTLY INCREASING
// TRANSFORMATION OF EITHER MARGINAL.** That is checkable to machine
// precision, and it is exactly what separates a copula statistic from a
// linear one:
//
//   * Pearson's r of the RAW values is NOT invariant -- a monotone
//     squashing of one marginal changes it.
//   * Pearson's r of the NORMAL SCORES IS invariant, because the
//     normal-scores transform is a monotone function of the marginal.
//
// The suite asserts the invariance AND asserts that the raw-Pearson foil
// FAILS it, so the property is not vacuous.
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT SHIPPED, AND WHY
// ---------------------------------------------------------------------------
//
// **BLOMQVIST'S BETA IS AMBIGUOUS IN THE LITERATURE and is not shipped under
// that name.** The `wdm` R package (Thomas Nagler, MIT) -- which the
// maintained `easystats/correlation` package DELEGATES to for its
// "blomqvist" method (`wdm::wdm(x, y, method = "blomqvist")`) -- implements
// it as a MEDIAN-QUADRANT statistic,
//
//     beta = 2 * (mass in the lower-left + upper-right quadrants about the
//                 medians) / (total mass) - 1,
//
// (read from `wdm/inst/include/wdm/bbeta.hpp` in CRAN wdm_0.2.6). Other
// sources describe a normal-scores ("medial") correlation. THESE ARE
// DIFFERENT FUNCTIONS OF THE DATA. Shipping one of them under the name
// "Blomqvist" would ship a definition this repo cannot justify.
//
// **The normal-scores correlation IS shipped -- under the name
// `gaussian_copula_correlation`, which says exactly what it computes -- and
// the median-quadrant variant is not shipped at all**, because nothing in
// the repo needs it.
//
// OTHER THINGS NOT SHIPPED, stated so they cannot creep in:
//   * Hypothesis tests and p-values. The asymptotic normal approximations
//     carry their own tie corrections and the exact permutation test is a
//     different order of complexity. The STATISTICS ship; the TESTS do not.
//   * Multivariate rank correlation (Kendall's W, the multivariate
//     Spearman footrule). A different data layout.
//   * Any `Layer`, any randomness, any model. This module is model-free.
//
// ---------------------------------------------------------------------------
// THE CITATIONS
// ---------------------------------------------------------------------------
//
//   * Spearman, C. (1904). "The Proof and Measurement of Association
//     between Two Things". American Journal of Psychology, 15(1), 72-101.
//     (Verified 2026-11-10 by a web search on the exact title: the JSTOR
//     record gives "Vol. 15, No. 1 (Jan., 1904), pp. 72-101".)
//   * Kendall, M. G. (1938). "A New Measure of Rank Correlation".
//     Biometrika, 30(1-2), 81-93. (Verified 2026-11-10 by a web search on
//     the exact title: the DOI 10.1093/biomet/30.1-2.81 and the Google
//     Scholar record agree on volume 30, pages 81-93.)
//   * Nelsen, R. B. (2006). "An Introduction to Copulas", 2nd edition.
//     Springer. The source of the Gaussian-copula population
//     correspondences. (Verified 2026-11-10 against the Springer record,
//     DOI 10.1007/0-387-28678-0.)
//
// **NO arXiv ID IS ASSERTED FOR ANY OF THESE.** All three predate arXiv,
// and this repo's expansion queue has produced fourteen wrong arXiv
// identifiers -- every one a plausible-looking neighbour of the right
// paper -- so an unverified bare number is exactly the thing to avoid.
//
// ---------------------------------------------------------------------------
// THE SHAPE CONVENTION
// ---------------------------------------------------------------------------
//
// Every statistic here takes TWO 1-D SAMPLES of the SAME NUMBER OF
// ELEMENTS, given as Tensors. THE VALUES ARE WHAT MATTER: a (N,1) and an
// (1,N) tensor holding the same values are the SAME sample. This is the
// same convention `utils/off_policy_eval.h` uses.
//
// ---------------------------------------------------------------------------
// ERRORS AND DEGENERATE INPUTS
// ---------------------------------------------------------------------------
//
// Every guard's message is DISTINCT. A substring that appears in two
// different messages discriminates nothing between them, and this repo has
// hit that trap repeatedly -- each time the assertion searched for a word
// the OTHER guard's message also contained, so the mutant survived.
//
// Throws std::invalid_argument on:
//   * an empty sample;
//   * the two samples having different lengths;
//   * fewer than two elements (a correlation needs two points).
//
// **A CONSTANT SAMPLE IS NOT AN ERROR.** Pearson's r of a constant sample
// is genuinely UNDEFINED (0/0), and `scipy.stats.spearmanr` returns NaN
// for it. This module returns NaN, matching the reference, and says so.
// **A NaN is the honest answer here; throwing would be a lie about the
// mathematics.** The distinction matters: the constant-sample case and the
// empty-sample case are DIFFERENT situations and get DIFFERENT answers.
//
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// mid_ranks -- the mid-rank transform.
//
//   x         (N, ...) the sample. The VALUES are what matter.
//
// Returns a Tensor of the same SHAPE whose elements are the mid-ranks: a
// tied group receives the AVERAGE of the ranks it spans, 1-based.
//
//   x = [10, 20, 20, 30]  ->  ranks = [1, 2.5, 2.5, 4]
//
// Throws std::invalid_argument if the sample is empty.
// ---------------------------------------------------------------------------
Tensor mid_ranks(const Tensor& x);

// ---------------------------------------------------------------------------
// pearson_r -- Pearson's product-moment correlation.
//
// Returns the correlation of the RAW values. This is the LINEAR statistic,
// included as the baseline the rank statistics are compared against, and
// because `spearman_rho` and `gaussian_copula_correlation` are DEFINED as
// Pearson's r of a transformed sample.
//
// Returns NaN if either sample is constant (the correlation is undefined).
//
// Throws std::invalid_argument on: an empty sample; the two samples having
// different lengths.
// ---------------------------------------------------------------------------
double pearson_r(const Tensor& x, const Tensor& y);

// ---------------------------------------------------------------------------
// spearman_rho -- Spearman's rank correlation coefficient.
//
// Pearson's r of the MID-RANKS of the two samples.
//
// **THE CLASSIC SHORTCUT `1 - 6*sum(d^2)/(n*(n^2-1))` IS NOT USED**: it
// is exact only when there are no ties at all, and with ties it differs
// from this definition.
//
// Returns NaN if either sample's mid-ranks are constant.
//
// Throws std::invalid_argument on: an empty sample; the two samples having
// different lengths.
// ---------------------------------------------------------------------------
double spearman_rho(const Tensor& x, const Tensor& y);

// ---------------------------------------------------------------------------
// kendall_tau_b -- Kendall's tau-b, the tie-corrected concordance
// coefficient.
//
// With tot/xtie/ytie/ntie/dis as defined at the top of this header,
//
//     tau_b = (tot - xtie - ytie + ntie - 2*dis)
//             / sqrt((tot - xtie) * (tot - ytie))
//
// **THE JOINT TIES COUNT IN BOTH xtie AND ytie.** See the header.
//
// Returns NaN if the denominator vanishes -- which happens exactly when
// one sample is entirely tied, i.e. constant.
//
// Throws std::invalid_argument on: an empty sample; the two samples having
// different lengths.
// ---------------------------------------------------------------------------
double kendall_tau_b(const Tensor& x, const Tensor& y);

// ---------------------------------------------------------------------------
// normal_scores -- the normal-scores (van der Waerden) transform.
//
//   z_i = Phi^-1((r_i - 0.5)/n)
//
// where r_i is the mid-rank of x_i and n is the sample size. The
// (r - 0.5)/n position is the HAZEN position.
//
// The result is a sample of the standard normal whose ORDER matches the
// input's, and it is INVARIANT under any strictly increasing
// transformation of the input -- which is what makes the correlation of two
// such transforms a COPULA statistic.
//
// Returns a Tensor of the same SHAPE as the input.
//
// Throws std::invalid_argument if the sample is empty.
// ---------------------------------------------------------------------------
Tensor normal_scores(const Tensor& x);

// ---------------------------------------------------------------------------
// gaussian_copula_correlation -- the MARGINAL-FREE correlation.
//
// Pearson's r of the normal scores of the two samples.
//
// **THIS IS INVARIANT UNDER ANY STRICTLY INCREASING TRANSFORMATION OF
// EITHER MARGINAL**, which is the defining property of a copula statistic
// and is what separates it from `pearson_r`.
//
// Returns NaN if either sample's normal scores are constant.
//
// Throws std::invalid_argument on: an empty sample; the two samples having
// different lengths.
// ---------------------------------------------------------------------------
double gaussian_copula_correlation(const Tensor& x, const Tensor& y);

// ---------------------------------------------------------------------------
// The four EXACT population correspondences for the Gaussian copula.
//
// For a bivariate Gaussian with correlation rho (the Gaussian copula with
// parameter rho):
//
//     tau   = (2/pi) * arcsin(rho)      rho = sin(pi * tau / 2)
//     rho_S = (6/pi) * arcsin(rho/2)    rho = 2 * sin(pi * rho_S / 6)
//
// These are POPULATION relationships. The SAMPLE statistics converge to
// them as the sample grows; they are not equal to them at finite n.
//
// Throws std::invalid_argument if the argument is outside [-1, 1]: a
// correlation coefficient is in [-1, 1], and an argument outside it is a
// caller error rather than a quantity to extrapolate.
// ---------------------------------------------------------------------------
double gaussian_copula_tau_from_rho(double rho);
double gaussian_copula_rho_from_tau(double tau);
double gaussian_copula_spearman_from_rho(double rho);
double gaussian_copula_rho_from_spearman(double rho_s);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_RANK_CORRELATION_H
