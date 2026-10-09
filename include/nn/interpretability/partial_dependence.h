#ifndef PARTIAL_DEPENDENCE_H
#define PARTIAL_DEPENDENCE_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <vector>

// =============================================================================
// Partial Dependence / Individual Conditional Expectation (ICE)
//
// Greenwell, Brandon M., "Partial Dependence Plots", The R Journal,
// 9(1):421-436, 2017. https://journal.r-project.org/articles/RJ-2017-016/RJ-2017-016.pdf
//
// CITATION CORRECTION. The expansion queue attributed this method to
// "Greenwell arXiv:1409.7002". That identifier is WRONG: the arXiv API returns
// "Entropy and Optimization of Portfolios" (a Markowitz asset-pricing paper) for
// 1409.7002, verified 2026-10-09. The R Journal article above is the actual
// source of record for the method.
//
// PDP and ICE answer a GLOBAL question, where the already-shipped attribution
// methods answer a LOCAL one:
//
//   PD(z)     = (1/N) · Σ_i  f(x_i with x_j := z)     the AVERAGE response as
//   ICE[i](z) =                f(x_i with x_j := z)     feature j varies
//
// The pairing with LIME (already shipped, arXiv:1606.03878) is the point: LIME
// answers "what did feature j do for THIS prediction", PD answers "what does
// feature j do ON AVERAGE, marginalizing over the other features". LIME's
// Gaussian neighborhood also has to CHOOSE its perturbation scale and width, and
// PD curves are the natural oracle for checking that choice — the effect of
// feature j over the observed range of j is exactly the curve LIME is trying to
// approximate with one straight line.
//
// Like LIME, this is a MODEL-AGNOSTIC method: the black box is only ever called
// as instance -> scalar. Like LIME again, it uses NO gradients, which is why the
// whole method is verifiable in CLOSED FORM rather than to a tolerance (see
// below).
//
// AUTHORITATIVE SOURCE. The grid and averaging rules were read from the
// reference implementation on 2026-10-09, NOT from memory:
//
//   sklearn/inspection/_partial_dependence.py:224-231  one-way PD definition:
//       "The brute method explicitly averages the predictions of an estimator
//        over a grid of feature values."
//   ..._partial_dependence.py:377     percentiles=(0.05, 0.95)
//   ..._partial_dependence.py:378     grid_resolution=100
//   ..._partial_dependence.py:149-153 np.linspace(lo, hi, num=g, endpoint=True)
//   ..._partial_dependence.py:138-142 if n_uniques < grid_resolution, use the
//                                 sorted uniques INSTEAD of a linspace
//   ..._partial_dependence.py:144-148 if the two empirical percentiles are
//                                 allclose, the reference RAISES
//   sklearn/utils/stats.py:13-38     _nanquantile -> np.nanquantile, i.e.
//                                 LINEAR interpolation between order statistics
//   sklearn/inspection/_plot/partial_dependence.py:1249,1254
//                                 centered=True -> preds -= preds[..., 0, None]
//   ..._partial_dependence.py:498-499 subsample limits how many ICE CURVES are
//                                 returned; the PD average still uses ALL of X
//
// THREE SEMANTICS THAT A REASONABLE REIMPLEMENTATION GETS WRONG:
//
//  1. The grid endpoints are the 5th and 95th PERCENTILES of the column, not its
//     min and max. That is the reference's default, and it is why PD curves are
//     not dominated by isolated outliers. `(0,1)` is legal but must be CHOSEN.
//  2. The quantile is LINEARLY INTERPOLATED. `lower`/`higher`/`nearest` are all
//     defensible-looking alternatives that produce a different grid.
//  3. Centering subtracts the value at GRID POINT 0, per instance for ICE and on
//     the average for PD — not the per-curve mean, and not each curve's response
//     at that instance's own observed feature value.
//
// CLOSED-FORM ORACLE (why this is unusually testable). For a LINEAR black box
// f(x) = Σ_k c_k·x_k with no bias, PD has an exact answer that does not depend on
// the spread of X at all:
//
//   PD(z)     = c_j·z + Σ_{k≠j} c_k·mean(X[:,k])
//   ICE[i](z) = c_j·z + Σ_{k≠j} c_k·X[i][k]
//
// So the PD curve is an exact LINE of slope c_j whose intercept is the mean
// response of the dataset with column j removed. That single identity pins the
// grid, the column replacement, the averaging and the absence of any hidden
// normalization in a single assertion — the same class of oracle as LIME's
// β_j = c_j·σ_j.
//
// CONVENTION — identical to attribution.h, shapley.h and lime.h: one feature is
// one input COLUMN of X, the model output is (rows, n_targets), and `target`
// selects the scalar column being explained. Attribution is on the LOGIT, never a
// softmax probability, because probabilities saturate and flatten the curves that
// this method exists to show.
//
// Deterministic for a given `seed`: a local xorshift64*, never std::rand(), so an
// explanation pass cannot perturb the RNG stream of an unrelated suite (see the
// test_adaln_zero intermittent-NaN entry in NOT_FIXED.md).
//
// FEATURE GRANULARITY — continuous columns only. The reference's categorical
// handling (use the unique categories as the axis) is out of scope for v1;
// a low-cardinality numeric column already gets the unique-value axis through the
// grid_resolution fallback below, which is most of that behaviour.
// =============================================================================

// Grid of feature values to evaluate partial dependence at.
//
// `X` supplies the empirical distribution: the axis endpoints are the
// `percentile_lo` / `percentile_hi` linear-interpolating quantiles of
// X[:, feature]. If the column has FEWER than `grid_resolution` distinct values,
// the axis is those distinct values (sorted) instead of an equally-spaced grid —
// a 3-valued integer column should be evaluated at its 3 values, not at 100
// interpolated points between them.
//
// Throws std::invalid_argument if X has no rows, if `feature` is out of range, if
// a percentile is outside [0,1], if percentile_lo >= percentile_hi, if
// grid_resolution <= 1, or if the two empirical percentiles are so close that the
// axis would have no width (the reference raises in that case rather than
// returning a flat curve).
//
// Does not modify `X` — the column is sorted on a copy.
std::vector<double> pd_grid(const Tensor& X,
                            size_t feature,
                            double percentile_lo = 0.05,
                            double percentile_hi = 0.95,
                            size_t grid_resolution = 100);

// Average an ICE matrix over instances to get the PD curve.
//
// `ice` is (n_instances, n_points); the result has length n_points and is the
// UNWEIGHTED column mean. This is a model-free function: it is the one
// mathematical operation `partial_dependence` performs on the responses, so a
// caller with their own response table can use it directly.
//
// Throws std::invalid_argument if `ice` is empty, has zero columns, or is ragged
// (rows of differing length).
std::vector<double> average_ice(const std::vector<std::vector<double>>& ice);

// Centered PD curve: pd[k] - pd[0].
//
// The reference's `centered=True` (partial_dependence.py:1254), which puts the
// curve's origin at zero so shapes can be compared without the level term.
// NOTE this subtracts element 0, NOT the mean of the curve — the two differ on
// any curve that is not flat, and mean-centering is a different (also useful)
// operation.
std::vector<double> center_pd(const std::vector<double>& pd);

// Centered ICE curves: ice[i][k] - ice[i][0], per instance.
//
// The per-instance form (partial_dependence.py:1249). Each row is shifted by its
// OWN first element, so centered ICE curves show the SHAPE of each instance's
// response rather than its level — which is the whole reason ICE plots exist.
void center_ice(std::vector<std::vector<double>>& ice);

// Configuration for one partial-dependence pass. Defaults reproduce the
// reference's one-way defaults.
struct PartialDependenceConfig {
    // Axis endpoints as empirical quantiles of X[:, feature]. NOT (0,1): the
    // reference default is (0.05, 0.95).
    double percentile_lo = 0.05;
    double percentile_hi = 0.95;
    // Number of equally-spaced axis points (subject to the low-cardinality
    // fallback in pd_grid).
    size_t grid_resolution = 100;
    // Maximum number of ICE CURVES stored. 0 means "store every instance".
    // NOTE this does NOT limit the PD average: `average` always averages over
    // every row of X (the reference's `subsample` limits curves DRAWN, not the
    // average — partial_dependence.py:498-499). Making these two agree was an
    // explicit decision, and it is the difference between a subsampled estimate
    // and the method's actual definition.
    size_t subsample = 1000;
    unsigned seed = 42;
    // Center `average` and `individual` in place on return.
    bool centered = false;
};

// Result of one partial-dependence pass.
struct PartialDependenceResult {
    // The axis, length n_points.
    std::vector<double> grid;
    // The PD curve: average_i ICE[i][k], length n_points.
    std::vector<double> average;
    // The ICE curves, n_curves x n_points. n_curves may be less than n_instances
    // when cfg.subsample caps it.
    std::vector<std::vector<double>> individual;
    size_t feature = 0;
    size_t target = 0;
    size_t n_points = 0;
    // Rows of X actually supplied.
    size_t n_instances = 0;
    // Curves retained (<= n_instances).
    size_t n_curves = 0;
    bool centered = false;
};

// Partial dependence and ICE curves for one feature of `X` against the model's
// `target` output column.
//
// Builds the axis with pd_grid, replaces X[:, feature] with each axis value,
// evaluates the model, and averages over instances. The whole grid is evaluated
// in ONE batched forward pass (n_points x n_instances rows) for the same reason
// lime_explain batches its neighborhood.
//
// Throws std::invalid_argument on the same conditions as pd_grid, and
// std::out_of_range if `target` exceeds the model output width. Target
// validation happens BEFORE any forward pass, so a bad target cannot leave the
// model's parameter gradients half-touched.
//
// Leaves no residue in the model's parameter gradients (zeroed before and after),
// matching lime_explain, shapley and the IG pass.
PartialDependenceResult partial_dependence(Model& model,
                                           const Tensor& X,
                                           size_t feature,
                                           const PartialDependenceConfig& cfg,
                                           size_t target = 0);

#endif // PARTIAL_DEPENDENCE_H