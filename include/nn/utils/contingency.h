#ifndef NN_UTILS_CONTINGENCY_H
#define NN_UTILS_CONTINGENCY_H

#include "../core/tensor.h"

#include <cstddef>
#include <limits>

// ===========================================================================
// CONTINGENCY STATISTICS: the categorical-agreement and
// contingency-table-inference utilities.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-19)
//
//   grep -rinE 'cohen|kappa'            include/  -> 6 hits, ALL FALSE POSITIVES
//                                               (megalodon's kappa_q / kappa_k
//                                                are DiT modulation parameters;
//                                                randomized_smoothing cites
//                                                "Jeremy M. Cohen" the author)
//   grep -rinE 'matthews|\bmcc\b'        include/  -> 0
//   grep -rinE 'mcnemar'                 include/  -> 0
//   grep -rinE 'fisher.?exact'           include/  -> 0
//   grep -rinE 'contingency'             include/  -> 0
//   grep -rinE 'cramer|phi.?coefficient' include/  -> 0
//   grep -rinE 'likelihood.?ratio'       include/  -> 0
//   grep -rinE 'chi.?square|chisq|chi2'  include/  -> 3 hits, ALL the df=1
//                                               closed form in utils/survival.{h,cpp}
//   grep -rinE 'incomplete.?gamma'       include/  -> 3 hits, ALL comments saying
//                                               the repo does NOT have one
//
//   ZERO IMPLEMENTATION.
//
// The repo ships ~250 layers and 60+ utility files and NO categorical
// agreement or categorical association machinery at all.
//
// It is NOT redundant with the shipped neighbours:
//   * utils/metrics.h -- accuracy, F1, ROC-AUC, ECE, NDCG, MRR. Every one
//     is a SCORE-vs-LABEL statistic. None of them compares two LABELINGS
//     OF THE SAME ITEMS, which is what kappa and McNemar do, and none of
//     them corrects for the agreement two raters would reach BY CHANCE.
//   * utils/rank_correlation.h -- Pearson / Spearman / Kendall on
//     CONTINUOUS quantities. There is no categorical analogue of
//     association anywhere in the repo.
//   * utils/survival.h -- the log-rank test's p-value is the CHI-SQUARE-1
//     survival function erfc(sqrt(x/2)), a closed form that exists ONLY for
//     one degree of freedom. A 2x3 table has df = 2 and does NOT fit, so
//     the repo cannot currently produce the p-value for any table bigger
//     than 2x2.
//
// ---------------------------------------------------------------------------
// WHAT THIS MODULE IS MODEL-FREE AND WHY THAT MATTERS
//
// Every core below is a free function over `Tensor`. The whole file is
// therefore testable with no Model, no optimizer and no network -- the same
// shape utils/survival.{h,cpp}, utils/rank_correlation.{h,cpp},
// utils/conformal.h and utils/sample_quality.h were built with. Nothing here
// is random.
//
// ---------------------------------------------------------------------------
// THE CONVENTIONS, AND EACH ONE IS A MEASURED DECISION
//
// The full argument for each one, with the numbers, is in the header's
// companion oracle (tests/contingency_oracle.py) and in the plan
// (docs/plans/2026-11-19-contingency-statistics.md). The references the
// measurements were taken against are scipy 1.17.1, scikit-learn 1.9.0 and
// R 4.5.3, all on this host.
//
// **C1. THE df = 1 PATH IS THE REPO'S EXISTING CLOSED FORM, BIT FOR BIT.**
// The shipped utils/survival.cpp computes `std::erfc(std::sqrt(statistic /
// 2.0))` for the log-rank test's p-value. `chi_square_survival` returns
// exactly that for df == 1, so the two code paths cannot drift. The GENERAL
// path is a SEPARATE implementation (the regularized upper incomplete
// gamma) and is checked against the closed form at df = 1 to 1e-14 -- two
// independent routes to the same number, which is what makes the
// incomplete gamma trustworthy.
//
// MEASURED against scipy.stats.chi2.sf over df in {1..100} and x in
// [1e-8, 1e5]: worst relative error 2.9e-14.
//
// **C2. THE WEIGHT MATRIX IS BUILT OVER THE EXPLICIT CLASS-INDEX SPACE, NOT
// OVER THE PRESENT LABELS.** This is a real fork and it changes the answer.
// MEASURED on a fixture whose present classes are {0, 1, 3} (class 2
// absent):
//
//   weights   | over present labels (compressed) | over the explicit index space
//   ----------+------------------------------------+----------------------------
//   unweighted| 0.304347826087                    | 0.304347826087
//   linear    | 0.304347826087                    | 0.360000000000   <- DIFFERS
//   quadratic | 0.360000000000                    | 0.421686746988   <- DIFFERS
//
// `scikit-learn`'s `labels=None` path compresses; its `labels=[...]` path
// does not. The repo's own convention is explicit -- utils/metrics.h's
// `confusion_matrix` takes an explicit `num_classes` -- so THIS MODULE USES
// THE EXPLICIT CLASS-INDEX SPACE. The UNWEIGHTED kappa is
// space-independent (the off-diagonal indicator matrix is the same whatever
// the class indices are), which is why only the weighted rows differ.
//
// **C3. THE UNDEFINED CASE IS NaN.** When the two raters share only one
// label the expected agreement is 1 and the denominator vanishes.
// `scikit-learn` returns NaN and raises an UndefinedMetricWarning;
// utils/metrics.h uses NaN for its undefined metrics too. The kappa result
// therefore carries NaN for `kappa` PLUS the `observed_agreement` /
// `expected_agreement` components, so a caller can see WHY.
//
// **C4. scikit-learn's `expected` MATRIX IS TRANSPOSED RELATIVE TO THE
// CONVENTIONAL ONE -- AND IT DOES NOT MATTER, FOR A PROVABLE REASON.**
// `sklearn` builds `expected = outer(col_sums, row_sums) / N`, i.e. the
// transpose of the textbook `E_ij = r_i c_j / N`. Every weight matrix this
// module supports (the off-diagonal indicator, `|i-j|`, `(i-j)^2`) is
// SYMMETRIC, and for symmetric `w` the sums `sum_ij w_ij E_ij` and
// `sum_ij w_ij E_ji` are equal by relabelling. So the transposition cancels
// exactly. This module uses the CONVENTIONAL orientation.
//
// **C5. THE MATTHEWS DEGENERATE CASE IS 0.0, NOT NaN -- AND THAT IS THE
// OPPOSITE OF scikit-learn's KAPPA DEGENERATE CASE.** Two different
// conventions in one library, MEASURED:
//
//                       sklearn cohen_kappa_score   sklearn matthews_corrcoef
//   degenerate case    NaN (+ UndefinedMetricWarning)  0.0
//
// The MCC denominator vanishes when either marginal is concentrated (all
// predictions in one class, or all truths in one class); sklearn returns
// 0.0 there, and so does this module. The kappa undefined case returns
// NaN, and so does this module. THESE ARE NOT INTERCHANGEABLE and the
// fixtures pin both.
//
// **C6. YATES' CONTINUITY CORRECTION IS APPLIED ONLY TO A 2x2 TABLE, AND THE
// FLAG IS HONOURED ONLY THERE.** scipy.stats.chi2_contingency and
// R::chisq.test agree exactly on this: the correction is applied when
// `correct && nrow == 2 && ncol == 2`, and ignored for every larger table.
// MEASURED on the 2x2 [[10,20],[30,15]]: corrected 6.752232142857143,
// uncorrected 8.035714285714286; scipy and R both report the corrected
// value by default and the uncorrected value on request.
//
// **C6-bis. THE CORRECTION IS CLAMPED TO THE CELL'S OWN DEVIATION, AND THE
// UNCORRECTED TEXTBOOK FORM IS WRONG.** Found by reading
// scipy.stats.chi2_contingency's source and stats:::chisq.test's source:
//
//   scipy: diff = expected - observed
//          direction = sign(diff)
//          magnitude = min(0.5, abs(diff))        <-- the CLAMP
//          observed += magnitude * direction
//          X2 = sum((observed - expected)^2 / expected)
//
//   R:     YATES <- min(0.5, abs(x - E))          <-- elementwise clamp
//          X2 <- sum((abs(x - E) - YATES)^2 / E)
//
// Both references therefore compute
//
//   X2_yates = sum_ij ( |O_ij - E_ij| - min(0.5, |O_ij - E_ij|) )^2 / E_ij
//
// which is NOT `sum_ij (|O_ij - E_ij| - 0.5)^2 / E_ij` whenever some cell
// has `|O - E| < 0.5`. The discriminating fixture is the 2x2 [[1,1],[1,2]],
// where every cell has `|O - E| = 0.2`:
//
//   formula                                        value    scipy    R
//   sum((|d| - 0.5)^2/E)   -- the UNCLAMPED form   0.3125   --        --
//   sum((|d| - min(0.5,|d|))^2/E) -- the CLAMPED    0.0      0.0      0.0
//
// MEASURED over EVERY 2x2 table with entries in 0..8 and no empty row or
// column (6272 tables), scipy and R agree to 1.8e-15 on the corrected
// statistic.
//
// **C6-ter. THE p-VALUE USES df = (R-1)(C-1), NOT scipy's power_divergence
// DEGREES OF FREEDOM.** scipy.stats.power_divergence computes its own
// p-value with df = number_of_cells - 1, which is WRONG for a contingency
// table (for a 4x3 table it uses 11 where the contingency df is 6).
// scipy.stats.chi2_contingency uses the correct (R-1)(C-1); measured over
// 300 random R x C tables, its p-value agrees with
// `chi2.sf(statistic, (R-1)*(C-1))` to 0.0. This module uses (R-1)(C-1) for
// the chi-square test, for the G-test, and for the power_divergence-style
// G p-value.
//
// **C6-quater. scipy.stats.power_divergence's `lambda_` IS INVERTED FROM THE
// OBVIOUS READING, AND `lambda_=1` IS PEARSON WHILE `lambda_=0` IS THE
// G-TEST.** MEASURED: `power_divergence(obs, exp, lambda_=1)` equals
// `sum((O-E)^2/E)` to 0.0, and `lambda_=0` equals `2 * sum(O log(O/E))` to
// 5.6e-16 over 600 random R x C tables.
//
// **C7. AN EMPTY ROW OR COLUMN MAKES THE EXPECTED COUNT ZERO AND THE
// STATISTIC UNDEFINED.** scipy raises ValueError ("The internally computed
// table of expected frequencies has a zero element") and R returns NaN.
// This module THROWS std::invalid_argument with its own message, which is
// the repo's established convention for invalid input (and utils/metrics.h's).
//
// **C8. A ZERO OBSERVED CELL CONTRIBUTES 0, NOT `0 * log(0)`.** The term
// `O log(O/E)` has limit 0 as O -> 0, so a zero cell contributes exactly 0.
// This is the classic trap: the naive expression is `0 * (-inf) = NaN` and
// the whole statistic is destroyed. The fixture MUST contain a zero cell,
// or the mutation that deletes the guard is invisible.
//
// **C9. McNEMAR'S CONTINUITY CORRECTION IS NOT APPLIED WHEN THE TABLE IS
// SYMMETRIC -- AND THAT IS THE NON-OBVIOUS PART.** R::mcnemar.test's
// condition is `correct && (r == 2) && any(x - t(x) != 0)`. So when
// `n01 == n10` the correction is SKIPPED and the statistic is the
// uncorrected 0, NOT `(0 - 1)^2 / (2*n01)`. MEASURED on [[10,10],[10,10]]:
// R reports statistic 0, p-value 1. A naive implementation that always
// subtracts 1 reports 1/20 there. THAT IS THE DISCRIMINATING FIXTURE.
//
// **C9-bis. "SYMMETRIC" IS ABOUT THE OFF-DIAGONAL CELLS.** R's source is
//
//   if (correct && r == 2 && any(x - t(x) != 0)) y <- abs(x - t(x)) - 1
//   else                                         y <- x - t(x)
//   x <- x + t(x)
//   STATISTIC <- sum(y[upper.tri(x)]^2 / x[upper.tri(x)])
//
// For a 2x2 table [[n00,n01],[n10,n11]], `x - t(x)` has entries
// [[0, n01-n10],[n10-n01, 0]], so `any(x - t(x) != 0)` is exactly
// `n01 != n10` -- the diagonal cells never enter the statistic.
//
// **C9-ter. R'S CORRECTION IS APPLIED TO THE OFF-DIAGONAL DIFFERENCE ONLY,
// SO THE RESULT IS `( |n01 - n10| - 1 )^2 / (n01 + n10)` -- WITH THE
// ABSOLUTE VALUE.** The signed form `(n01 - n10 - 1)^2` is wrong whenever
// `n01 < n10`: MEASURED on [[0,3],[5,0]] (n01 = 3, n10 = 5) R reports
// `(|3-5|-1)^2/8 = 0.125`, whereas the signed form gives 0.5625.
//
// **C9-quater. THE RULE IS PRECISELY: apply the continuity correction IFF
// `n01 != n10`; otherwise the statistic is 0.** MEASURED on four tables
// that discriminate this: [[10,10],[10,10]] -> 0 (symmetric, skipped);
// [[30,5],[5,25]] -> 0 (symmetric off-diagonal, skipped); [[30,10],[5,25]]
// -> 1.06666667 (asymmetric, corrected); [[5,20],[10,15]] -> 2.70000000
// (asymmetric, corrected). All four match R exactly.
//
// **C10. n01 + n10 == 0 IS NaN, NOT 0.** With no discordant pairs the
// statistic is 0/0. R reports NaN for both the statistic and the p-value.
// MEASURED on [[30,0],[0,30]].
//
// **C11. THE EXACT p-VALUE IS THE REPO'S EXISTING `binom_pvalue_half`.**
// The exact McNemar test is a two-sided binomial test at p = 1/2 on the
// `n01 + n10` discordant pairs. utils/randomized_smoothing.h already ships
// that function (at GLOBAL scope, no namespace), and its own oracle pins it
// to scipy.stats.binomtest. This module COMPOSES it rather than
// hand-rolling a second binomial tail. (The composition is checked against
// scipy in the oracle: F10 exact p-value 0.7265625, scipy 0.7265625.)
//
// **C12. FISHER'S TWO-SIDED p-VALUE USES THE PROBABILITY ORDERING RULE,
// NOT A DOUBLED ONE-SIDED TAIL.** MEASURED against
// scipy.stats.fisher_exact on eight tables including three with ties: the
// probability-ordering rule agrees to 0.0 on every one, while the
// doubled-tail rule does not. It is also R::fisher.test's rule.
//
// **C12-bis. THE PROBABILITY ORDERING COMPARISON CARRIES A 1 + 1e-14
// RELATIVE TOLERANCE, AND THE TOLERANCE MATTERS.** Two tables can share the
// observed hypergeometric pmf EXACTLY IN EXACT ARITHMETIC while their
// float64 evaluations differ in the last bits. The comparison
//
//   pmf(x) <= pmf(observed) * (1 + 1e-14)
//
// is what includes the second one; a bare `pmf(x) <= pmf(observed)`
// EXCLUDES it and the p-value HALVES. MEASURED on [[0,2],[5,3]]: scipy
// reports 0.44444444444444453, the sweep with the tolerance reports
// 0.44444444444444459, the sweep WITHOUT it reports 0.2222222222222221.
// The fixture was found by SEARCH against the model's own float64
// arithmetic; a first attempt ([[0,3],[7,4]]) has a tie that is bit-exact
// in the model's lgamma evaluation, so dropping the tolerance changed
// nothing there and the mutation SURVIVED.
//
// **C13. FISHER'S ODDS RATIO IS THE UNCONDITIONAL (CROSS-PRODUCT) ONE, AND
// R'S IS DIFFERENT.** `scipy` returns `0.25` for `[[10,20],[30,15]]` -- the
// cross-product `(a*d)/(b*c)`. `R` reports `0.2550704` -- the **conditional
// MLE**, which is a different estimator. The module ships the cross-product
// ratio, and the header says so, because the cross-product is the quantity
// the contingency table actually determines.
//
// **C14. THE SUMS ARE ACCUMULATED IN NUMPY'S PAIRWISE ORDER, NOT IN A
// NAIVE LEFT-TO-RIGHT LOOP.** This is the one convention here that is not
// about the STATISTICS but about the ARITHMETIC, and it is load-bearing
// rather than cosmetic.
//
// The oracle's model sums with numpy, and numpy's `ndarray.sum()` is NOT a
// naive loop: it accumulates in blocks of EIGHT with EIGHT partial sums,
// combines them pairwise, and splits the array in half above 128 elements.
// A C++ loop that accumulates sequentially disagrees with the model in the
// last bits -- and in the kappa's near-zero regime, where
// `1 - numerator/denominator` cancels catastrophically, that last bit is
// worth 5.6e-04 of RELATIVE difference, which is 9 orders of magnitude
// outside the 1e-12 relative tolerance the repository's own `check_near`
// uses.
//
// MEASURED, over 1200 random (labels, labels, num_classes, weights) cases
// covering all three weight schemes: with numpy's pairwise reduction the
// kappa agrees with the oracle's model BIT FOR BIT on 1200/1200; with a
// sequential accumulation it agrees on 732/1200 only. The two reductions
// differ on the same near-zero cases the oracle's own two witnesses
// (`cohen_kappa_model` and `cohen_kappa_reference`) disagree on -- the
// model sums with numpy and the reference sums sequentially, so they
// cannot agree there either.
//
// The same routine reduces the DIAGONAL, because `np.trace` is numpy's
// sum over the diagonal and not a Python loop. MEASURED: `np.trace` matches
// the pairwise diagonal sum on 56/56 random 2-D arrays and the sequential
// diagonal sum on 40/56 only.
//
// The pairwise routine is transcribed from numpy's own
// `pairwise_sum_DOUBLE` (numpy/core/src/umath/loops_arithm_fp.dispatch.c.src)
// and lives in the implementation's anonymous namespace.
//
// **C14-bis. FISHER'S TIE DECISION IS THE ONE PLACE WHERE THE MODULE AND
// THE ORACLE CAN DISAGREE, AND THE MODULE IS THE MORE ACCURATE OF THE
// TWO.** The tie rule (C12-bis) compares pmf values that are computed in
// LOG SPACE, so the comparison sits at the ~1e-13 level -- the same order
// as the 1e-14 relative tolerance and the same order as the difference
// between two libms' `lgamma`. A table whose pmf ties the observed one
// EXACTLY IN EXACT ARITHMETIC can therefore land on either side of the
// window, and WHICH side it lands on depends on the libm.
//
// The oracle's model uses CPython's `math.lgamma`, which since CPython
// 3.11 is an INTERNAL Lanczos implementation; this module uses the system
// `std::lgamma` (glibc). MEASURED on this host: the two disagree on 2465
// of 4399 arguments, by up to 1135 ulp. Against EXACT rational arithmetic
// (Fractions, no floating point at all) over 3000 random 2x2 tables:
// the ORACLE'S MODEL's included set is the exact set on 2990/3000, and
// THIS MODULE'S included set is the exact set on 2996/3000. On every
// table where the two disagree, the module sides with the exact
// arithmetic and the model does not.
//
// The consequence is measured and bounded: over 800 random 2x2 tables the
// module and the model disagree on the p-value on 3 of them, and the
// module is the correct one on all three. The disagreement is confined to
// tables whose pmf ties the observed one EXACTLY, where the tolerance is
// the only thing that decides the answer. The FIXTURE tables are
// unaffected: the F16 tie table `[[0,2],[5,3]]` has its tie BIT-EXACT in
// glibc, so both the model and the module include it and both report
// 0.44444444444444459 (the model) / 0.44444444444444342 (the module), a
// relative difference of 2.6e-15, and the table `[[0,2],[15,13]]` has its
// tie inexact in BOTH libms and is the one that observes the tolerance
// (0.48275862068964726 the model, 0.48275862068965414 the module, a
// relative difference of 1.4e-14).
//
// This is a property of the COMPARISON, not of the module's arithmetic:
// no choice of `lgamma` can make a float64 tie comparison agree with
// another libm's, because the tie is decided at the 1e-14 level and the
// libms differ at the 1e-13 level. The module therefore implements the
// rule (C12-bis) as written and lets the libm decide the boundary case,
// which is the same choice the oracle makes.
//
// ---------------------------------------------------------------------------
// THE DEPENDENCIES, AND WHY THEY ARE WHAT THEY ARE
//
// * `utils/randomized_smoothing.h` is included for EXACTLY ONE symbol,
//   `binom_pvalue_half` (C11). That header lives at GLOBAL scope (no
//   namespace) and pulls in core/model.h, so it is included by the
//   IMPLEMENTATION ONLY -- the public header of this module depends on
//   core/tensor.h and nothing else.
//
// * `utils/metrics.h` is NOT used. Its `confusion_matrix` would do, but
//   that header is at GLOBAL scope too and it drags core/model.h into every
//   consumer. This module therefore computes its own contingency table
//   locally, which also lets it build the table over the EXPLICIT class
//   index space (C2) rather than over whatever classes happen to appear.
//
// ---------------------------------------------------------------------------
// THE SHAPE CONVENTION
//
// Every label vector here is an (N, 1) tensor of INTEGER CLASS INDICES, and
// every contingency table is an (R, C) tensor of COUNTS. A (N,1) and an
// (1,N) tensor holding the same values are NOT the same sample here: this
// module's inputs are COLUMN VECTORS, and a non-single-column input is
// rejected rather than silently reinterpreted.
//
// ---------------------------------------------------------------------------
// ERRORS AND DEGENERATE INPUTS
//
// Every guard's message is DISTINCT. A substring that appears in two
// different messages discriminates nothing between them, and this repo has
// hit that trap repeatedly.
//
// Throws std::invalid_argument on:
//   * a label tensor that is not a single column, or has no rows;
//   * a label that is not a whole number, is not finite, or is outside
//     [0, num_classes);
//   * two label vectors with different row counts;
//   * `num_classes == 0`;
//   * a contingency table with fewer than two rows or two columns;
//   * a contingency table with an EMPTY ROW OR COLUMN, i.e. a zero
//     expected count (C7);
//   * a 2x2 table whose entries are not non-negative whole numbers
//     (Fisher's exact test).
//
// **A DEGENERATE STATISTIC IS NOT AN ERROR.** Cohen's kappa is NaN when it
// is undefined (C3); the Matthews correlation coefficient is 0.0 when its
// denominator vanishes (C5); McNemar's statistic is NaN when there are no
// discordant pairs (C10). These are the mathematical answers, they are the
// answers the references give, and the results carry the COMPONENTS (the
// agreement terms, the discordant counts) so a caller can see why.
//
// ---------------------------------------------------------------------------
// THE CITATIONS
//
//   - Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales."
//     Educational and Psychological Measurement, 20(1), 37-46.
//   - Cohen, J. (1968). "Weighted Kappa: The Relation of Unweighted, Rated,
//     and Percentile Agreement Nominal to Ordinal." Educational and
//     Psychological Measurement, 28(1), 109-127.
//   - Matthews, B. W. (1975). "Comparison of the Predicted and Observed
//     Secondary Structure of a Protein." Biochimica et Biophysica Acta,
//     405(2), 442-451.
//   - Pearson, K. (1900). "X. On the Criterion That a Given System of
//     Deviations from the Probable in the Case of Correlated Responses
//     Should Be Judged." The London, Edinburgh, and Dublin Philosophical
//     Magazine and Journal of Science, 50(302), 157-175.
//   - Fisher, R. A. (1922). "On the Probable Error of the Chi-Square
//     Statistic." Journal of the Royal Statistical Society, 85(1), 87-94.
//   - Yates, F. (1934). "Contingency Tables and the Analysis of Data."
//     Biometrika, 26(4), 301-304.
//   - McNemar, Q. (1947). "Note on the Sampling Error of the Difference
//     Between Correlated Proportions or Percentages." Psychometrika,
//     12(2), 153-157.
//   - Goodman, L. A. and Kruskal, W. H. (1954). "Measures of Association for
//     Cross Classifications." Journal of the American Statistical
//     Association, 49(267), 1097-1121.
//
// All of them predate arXiv, and **NO arXiv ID IS ASSERTED** -- this repo's
// expansion queue has produced fifteen wrong arXiv identifiers, every one a
// plausible-looking neighbour of the right paper.
//
// The G-test's usual name is the "likelihood-ratio chi-square test" or
// "G-test"; the G statistic's normalisation (2 * sum O log(O/E)) is the one
// used here.
//
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// chi_square_survival -- P(X >= statistic) for X ~ chi2(degrees_of_freedom).
//
// **C1. THE df == 1 PATH IS THE REPO'S EXISTING CLOSED FORM, BIT FOR BIT:**
// it returns exactly `std::erfc(std::sqrt(statistic / 2.0))`, the same
// expression utils/survival.cpp uses for the log-rank test's p-value, so
// the two code paths cannot drift. For df > 1 the GENERAL path is used: the
// regularized upper incomplete gamma, the SERIES for P(a,x) when x < a+1
// and the CONTINUED FRACTION for Q(a,x) otherwise.
//
// MEASURED against scipy.stats.chi2.sf over df in {1..100} and x in
// [1e-8, 1e5]: worst relative error 2.9e-14. The general path and the
// df == 1 closed form agree with each other to 1e-14 -- two independent
// routes to the same number.
//
// A statistic of exactly 0 returns 1.0, because P(X >= 0) = 1 for every df
// (and scipy's chi2.sf(0, df) is 1). The guard is not cosmetic: the
// general path evaluates log(x) with x = statistic/2, so a statistic of 0
// is log(0) = -inf and the whole expression is NaN.
//
// THROWS std::invalid_argument when:
//   - the statistic is NEGATIVE (a negative test statistic is not a
//     possible input to a chi-square survival function, and the
//     alternative -- returning 1.0 -- would silently answer a question
//     nobody asked);
//   - the statistic is not finite;
//   - the degrees of freedom are NOT POSITIVE, or are not finite.
//
// The degrees of freedom are a `double` on purpose: the general path is
// defined for fractional df as well (the incomplete gamma is), and nothing
// here needs an integer.
// ---------------------------------------------------------------------------
double chi_square_survival(double statistic, double degrees_of_freedom);

// ---------------------------------------------------------------------------
// The disagreement weights of Cohen's kappa.
//
//   Unweighted -- w_ij = 1 for i != j, 0 on the diagonal. The plain
//                 (chance-corrected) kappa.
//   Linear     -- w_ij = |i - j|. For an ORDINAL scale, where the classes
//                 have a natural order and disagreeing by one step is not
//                 the same disagreement as disagreeing by five.
//   Quadratic  -- w_ij = (i - j)^2. The same idea with the distance
//                 squared.
//
// **ALL THREE ARE SYMMETRIC**, which is what makes C4 (the orientation of
// the expected-count matrix) provably irrelevant.
// ---------------------------------------------------------------------------
enum class KappaWeights { Unweighted, Linear, Quadratic };

// ---------------------------------------------------------------------------
// The result of Cohen's kappa.
//
// `kappa` is NaN when it is UNDEFINED (C3) -- that happens exactly when
// the two raters share only one label, so the chance agreement is 1 and
// the denominator vanishes.
//
// The agreement components are reported alongside it so a caller can see
// WHY the answer is what it is: `observed_agreement` is p_o and
// `expected_agreement` is p_e, both in [0, 1].
// ---------------------------------------------------------------------------
struct KappaResult {
    double kappa = std::numeric_limits<double>::quiet_NaN();
    double observed_agreement = 0.0;
    double expected_agreement = 0.0;
    size_t num_agreements = 0;
    size_t num_total = 0;
};

// ---------------------------------------------------------------------------
// cohen_kappa -- Cohen's kappa over the EXPLICIT class-index space (C2).
//
//   labels_a    (N,1) integer class indices in [0, num_classes)
//   labels_b    (N,1) integer class indices in [0, num_classes)
//   num_classes the SIZE OF THE CLASS SPACE, not the number of classes
//               that happen to appear in the data
//
// The weights are built over the EXPLICIT index space 0..num_classes-1, so
// an ABSENT intermediate class does NOT compress the distances (C2). On a
// fixture whose present classes are {0,1,3}, the linear kappa is 0.36 in
// this explicit space and 0.3043 in the compressed one -- MEASURED, and the
// fixture that pins it.
//
// The statistic is
//
//   kappa_w = 1 - (sum_ij w_ij C_ij) / (sum_ij w_ij E_ij)
//
// with C the observed contingency table and E_ij = r_i c_j / N the
// conventional expected counts (C4: the orientation does not matter,
// because every supported weight matrix is symmetric).
//
// THROWS std::invalid_argument when:
//   - `num_classes` is 0;
//   - the two label vectors have different row counts;
//   - either label vector is not a single column, or has no rows;
//   - a label is not finite, is not a whole number, or is outside
//     [0, num_classes).
// ---------------------------------------------------------------------------
KappaResult cohen_kappa(const Tensor& labels_a, const Tensor& labels_b,
                        size_t num_classes,
                        KappaWeights weights = KappaWeights::Unweighted);

// ---------------------------------------------------------------------------
// The result of the Matthews correlation coefficient.
//
// `mcc` is 0.0 in the DEGENERATE case (C5) -- that is, when either
// marginal is concentrated (every truth in one class, or every prediction
// in one class), so the denominator vanishes. **This is the OPPOSITE of
// Cohen's kappa, whose undefined case is NaN (C3)**, and the two are not
// interchangeable.
//
// The MCC is TRANSPOSE-INVARIANT: MEASURED |MCC(t,p) - MCC(p,t)| is
// exactly 0 over 3000 random pairs, because `cov_ytyp` uses
// dot(p_sum, t_sum), which is symmetric, and the denominator is a PRODUCT
// of the two marginal variances. The order of `truth` and `prediction`
// therefore does not change the number.
// ---------------------------------------------------------------------------
struct MatthewsResult {
    double mcc = 0.0;
    size_t num_correct = 0;
    size_t num_total = 0;
};

// ---------------------------------------------------------------------------
// matthews_correlation -- the general MULTICLASS Matthews correlation
// coefficient (C5).
//
//   truth       (N,1) integer class indices in [0, num_classes)
//   prediction  (N,1) integer class indices in [0, num_classes)
//
// With t_k the number of truths in class k, p_k the number of predictions
// in class k, c the number of correct predictions and N the total:
//
//   cov_ytyp = c*N - sum_k p_k t_k
//   cov_ypyp = N^2 - sum_k p_k^2
//   cov_ytyt = N^2 - sum_k t_k^2
//   MCC      = cov_ytyp / sqrt(cov_ytyt * cov_ypyp)
//
// The MCC is INVARIANT TO THE LABEL SPACE (MEASURED: the same data gives
// the same MCC for num_classes 3 and 5), because the formula only uses the
// marginal counts.
//
// THROWS std::invalid_argument on the same grounds as `cohen_kappa`.
// ---------------------------------------------------------------------------
MatthewsResult matthews_correlation(const Tensor& truth, const Tensor& prediction,
                                    size_t num_classes);

// ---------------------------------------------------------------------------
// The result of Pearson's chi-square test of independence.
//
// `min_expected_count` is the SMALLEST expected count, so a caller can
// apply the familiar "no expected count below 5" rule of thumb without
// recomputing the expected table.
// ---------------------------------------------------------------------------
struct ChiSquareTestResult {
    double statistic = 0.0;
    double p_value = 1.0;
    size_t degrees_of_freedom = 0;
    double min_expected_count = 0.0;
};

// ---------------------------------------------------------------------------
// chi_square_test_of_independence -- Pearson's chi-square test of
// independence on an R x C table of counts.
//
//   table   (R,C) non-negative counts
//   correct apply Yates' continuity correction? (honoured ONLY for a 2x2
//           table -- C6)
//
//   E_ij = r_i * c_j / N
//   X2   = sum_ij ( |O_ij - E_ij| - min(0.5, |O_ij - E_ij|) )^2 / E_ij
//          when the correction applies (C6-bis: the CLAMP is the whole
//          point -- the unclamped textbook form gives 0.3125 where scipy
//          and R both give 0.0 on the 2x2 [[1,1],[1,2]])
//        = sum_ij (O_ij - E_ij)^2 / E_ij
//          otherwise
//   df    = (R-1) * (C-1)      (C6-ter -- NOT scipy's power_divergence df)
//   p     = chi_square_survival(X2, df)
//
// THROWS std::invalid_argument when:
//   - the table has fewer than two rows or two columns;
//   - the table has an EMPTY ROW OR COLUMN, i.e. a zero expected count
//     (C7).
// ---------------------------------------------------------------------------
ChiSquareTestResult chi_square_test_of_independence(const Tensor& table,
                                                     bool correct = true);

// ---------------------------------------------------------------------------
// The result of the G-test (the likelihood-ratio chi-square test).
// ---------------------------------------------------------------------------
struct GTestResult {
    double statistic = 0.0;
    double p_value = 1.0;
    size_t degrees_of_freedom = 0;
};

// ---------------------------------------------------------------------------
// g_test_of_independence -- the G-test (log-likelihood ratio) on the same
// R x C table.
//
//   G = 2 * sum_ij O_ij * log(O_ij / E_ij), over the cells with O_ij > 0
//   df = (R-1) * (C-1)
//   p  = chi_square_survival(G, df)
//
// **A ZERO OBSERVED CELL CONTRIBUTES EXACTLY 0 AND IS SKIPPED** (C8). The
// naive expression `0 * log(0/E)` is `0 * (-inf) = NaN` and destroys the
// whole statistic.
//
// THROWS std::invalid_argument on the same grounds as
// `chi_square_test_of_independence`.
// ---------------------------------------------------------------------------
GTestResult g_test_of_independence(const Tensor& table);

// ---------------------------------------------------------------------------
// The result of McNemar's test.
//
// `statistic` and `p_value` are NaN when there are NO DISCORDANT PAIRS
// (C10): the statistic is 0/0 there, and R reports NaN for both.
//
// `exact_p_value` is the EXACT binomial p-value, which is 1.0 when there
// are no discordant pairs (a vacuous test is not evidence of anything) and
// is the repo's existing `binom_pvalue_half` otherwise (C11).
// ---------------------------------------------------------------------------
struct McNemarResult {
    double statistic = std::numeric_limits<double>::quiet_NaN();
    double p_value = std::numeric_limits<double>::quiet_NaN();
    double exact_p_value = 1.0;
    size_t n01 = 0;           // #(a = 0, b = 1)
    size_t n10 = 0;           // #(a = 1, b = 0)
    size_t n_discordant = 0;  // n01 + n10
};

// ---------------------------------------------------------------------------
// mcnemar_test -- McNemar's test for PAIRED BINARY outcomes.
//
//   a, b   (N,1) tensors whose values are EXACTLY 0 or 1
//   correct apply the continuity correction? (honoured only when the
//           table is asymmetric -- C9)
//   df      1 for the chi-square p-value
//
// The statistic follows R's stats::mcnemar.test EXACTLY (C9, C9-bis,
// C9-ter, C9-quater):
//
//   * the continuity correction is applied IFF n01 != n10;
//   * the corrected numerator is (|n01 - n10| - 1)^2 -- WITH the absolute
//     value;
//   * when n01 == n10 the statistic is 0, NOT 1/(2*n01) -- the correction
//     is SKIPPED;
//   * when n01 + n10 == 0 the statistic is NaN (0/0), not 0;
//   * the uncorrected statistic is (n01 - n10)^2 / (n01 + n10);
//   * p_value = chi_square_survival(statistic, 1);
//   * exact_p_value = binom_pvalue_half(min(n01,n10), max(n01,n10)).
//
// MEASURED against R on four tables that discriminate the rule:
// [[10,10],[10,10]] -> 0 (skipped); [[30,5],[5,25]] -> 0 (skipped);
// [[30,10],[5,25]] -> 1.06666667; [[5,20],[10,15]] -> 2.70000000.
//
// THROWS std::invalid_argument when:
//   - the two outcome vectors have different row counts;
//   - either is not a single column;
//   - an outcome is not exactly 0 or 1.
// ---------------------------------------------------------------------------
McNemarResult mcnemar_test(const Tensor& a, const Tensor& b, bool correct = true);

// ---------------------------------------------------------------------------
// The result of Fisher's exact test.
//
// `odds_ratio` is the UNCONDITIONAL cross-product ratio (a*d)/(b*c), NOT
// R's conditional MLE (C13) -- they are different estimators, and the
// cross-product is the quantity the contingency table determines.
//
// `p_value` is the TWO-SIDED probability-ordering p-value (C12).
// ---------------------------------------------------------------------------
struct FisherExactResult {
    double p_value = 1.0;
    double odds_ratio = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// fisher_exact_test -- Fisher's exact test for a 2x2 table of counts.
//
//   table   (2,2) non-negative whole numbers
//
// The two-sided p-value uses the PROBABILITY ORDERING rule: the sum of the
// hypergeometric pmf over every x in the support whose pmf is at most
// `pmf(observed) * (1 + 1e-14)`.
//
// **THE 1e-14 TOLERANCE MATTERS** (C12-bis). A table whose pmf ties the
// observed one exactly in exact arithmetic is INCLUDED by the tolerance
// and EXCLUDED without it, and the p-value then HALVES. MEASURED on
// [[0,2],[5,3]]: 0.44444444444444459 with the tolerance, 0.2222222222222221
// without.
//
// A table with an EMPTY ROW OR COLUMN gives p_value 1.0 and an odds ratio
// of NaN -- which is what scipy returns.
//
// THROWS std::invalid_argument when:
//   - the table is not 2x2;
//   - an entry is not finite, is not a whole number, or is negative.
// ---------------------------------------------------------------------------
FisherExactResult fisher_exact_test(const Tensor& table);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_CONTINGENCY_H
