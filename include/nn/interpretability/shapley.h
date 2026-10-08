#ifndef SHAPLEY_H
#define SHAPLEY_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <string>
#include <vector>

// =============================================================================
// Shapley values and KernelSHAP.
//
// Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
// NeurIPS 2017, arXiv:1705.07874.
//
// The Shapley value of a coalition game v(S) attributes to player i the
// average marginal contribution of i across all orderings of the coalition:
//
//   phi_i = sum_{S subset N\{i}} |S|!(M-|S|-1)!/M! * (v(S u {i}) - v(S))
//
// and it is the unique solution of the Shapley axioms that is symmetric,
// efficient and additive. EFFICIENCY is the property that matters most for
// attribution:
//
//   sum_i phi_i = v(N) - v(empty)
//
// i.e. the attributions ALWAYS sum to the model's actual output change, for
// ANY v. Integrated Gradients (attribution.h) only satisfies this
// approximately, with error equal to the Riemann quadrature error. Here the
// sum is exact to machine precision by construction, not by convergence.
//
// KernelSHAP (§4 of the paper) turns the exponential 2^M enumeration into a
// tractable weighted least-squares problem over sampled coalitions. The
// weights come from the SHAP kernel; the reference implementation writes them
// as 1/(M * C(M-1, d)) for a coalition of size d, which is the form used
// here. Both forms are the same kernel up to a constant absorbed into the
// least-squares scale.
//
// AUTHORITATIVE SOURCE. The formulas were read from the reference
// implementation (shap/explainers/_coalition.py::_compute_weight and
// shap/explainers/_kernel.py::KernelExplainer.solve), not from memory:
//
//   _compute_weight(total, selected) = 1 / (total * math.comb(total-1, selected))
//   phi[nonzero_inds[-1]] = (fx - fnull) - sum(w)      # efficiency residual
//   normal equations:     (X' W X) phi = X' W y      # WX = kernelWeights[:,None]*X
//
// Why no new infrastructure was needed: this repo's `Model::forward` is
// callable any number of times and `Dense::backward` accumulates, so an
// attribution pass zeroes parameter gradients before and after and leaves no
// residue in training state.
//
// CONVENTION — same as attribution.h: the model output must be
// (rows, n_classes) and `target` selects which output column is the scalar
// being attributed. Values are attributed on the LOGIT, never a softmax
// probability, because probabilities saturate and their gradients vanish.
//
// FEATURE GRANULARITY — one Shapley feature per input COLUMN of `x`. The
// repo's Tensor is real-only 2-D, so a column mask is the only mask that is
// simultaneously well-defined and invertible. Masked (absent) features are
// set to the corresponding entry of `baseline`; this is the
// SimpleImputer-style masker, i.e. the "interventional"/baseline convention.
// Dependence-aware (observational) marginals over a background dataset are out
// of scope for v1.
// =============================================================================

// Result of a Shapley attribution pass.
struct ShapleyResult {
    // One attribution per input feature (column of `x`), shape (1, n_features).
    // The horizontal layout is the repo convention: a per-feature vector is a
    // (1, n) tensor, matching Dense's (1, out) bias layout.
    Tensor attributions;
    // f(baseline) for the attributed target. sum(attributions) always equals
    // model_output(x) - base_value.
    double base_value = 0.0;
    // sum(attributions) - (f(x) - f(baseline)). Zero to machine precision for
    // every method here: the SAMPLED estimators (kernel / permutation) get it
    // by construction from finalize_shapley's efficiency residual, and
    // exact_shapley satisfies it because exhaustive enumeration already sums
    // to the output change. Kept as an explicit, asserted field rather than an
    // implied invariant.
    double completeness_delta = 0.0;
    size_t n_features = 0;
    // Model evaluations actually performed by this estimator.
    size_t n_coalitions = 0;
    std::string method;  // "exact" | "kernel" | "permutation"
};

// Model output for one target column, averaged over rows of `input`.
//
// The scalar being attributed. Each Shapley method treats this as the
// game v(S); it is the only thing the estimator needs from the model, which
// is what makes these methods model-agnostic.
//
// Throws std::out_of_range if `target` exceeds the model output width, and
// std::invalid_argument if `input` has no columns.
double model_output(Model& model, const Tensor& input, size_t target = 0);

// SHAP least-squares kernel weights for M features, indexed by coalition size.
//
// weights[d] = 1 / (M * C(M-1, d)) for d = 0..M, matching the reference
// implementation. Out is resized to (1, M+1).
//
// Note d = M: C(M-1, M) is zero, so the raw formula would divide by zero.
// The convention used here gives that entry 1/M, the same value as d = 0.
// (This is harmless — the solver never uses the d = M row, because the
// efficiency constraint determines the full coalition's contribution.)
//
// Throws std::invalid_argument if M == 0.
void kernel_weights(size_t m, Tensor& weights);

// Exact Shapley values by exhaustive enumeration of the coalition lattice.
//
// Ground truth for any other estimator, and exact to machine precision for
// any M it accepts. Cost is O(M * 2^(M-1)) model evaluations.
//
// Throws std::invalid_argument if baseline's shape differs from x's, and
// std::out_of_range if M > kExactShapleyMaxFeatures (2^12 coalitions per
// feature is already ~49k model calls; beyond that use kernel_shap).
//
// Returns the same ShapleyResult shape as the other two estimators so a
// caller can switch between exact and approximate without changing code.
// Efficiency is exact (see finalize_shapley), so comparing against
// kernel_shap's efficiency is always a meaningful check.
ShapleyResult exact_shapley(Model& model,
                            const Tensor& x,
                            const Tensor& baseline,
                            size_t target = 0);

// Largest feature count exact_shapley accepts.
extern const size_t kExactShapleyMaxFeatures;

// KernelSHAP: sampled weighted least squares over coalitions.
//
// Approximates the Shapley values using `n_samples` coalition evaluations
// weighted by the SHAP kernel, solving (X'WX) phi = X'Wy. Efficiency is exact
// by construction; accuracy improves with n_samples.
//
// Coalitions are drawn by walking the lattice by coalition SIZE (d = 1..M-1)
// rather than by drawing independent uniform bitmask draws. Uniform bitmask
// draws over 2^M under-sample the mid-size coalitions that carry the most
// information and badly degrade the conditioning of the normal equations.
//
// BUG FOUND 2026-10-07 — THE EFFICIENCY-CENTERED DESIGN MATRIX. The first
// version regressed the plain target `v(z) - f(baseline)` on the plain design
// `Z`, i.e. a model with an implicit intercept pinned at 0. That is the wrong
// problem. The reference centers BOTH on the residual feature
// (shap/explainers/_kernel.py:738-741):
//     etmp = Z - Z[:, last];   eyAdj2 = eyAdj - Z[:, last] * (fx - fnull)
// and then recovers phi[last] = (fx - fnull) - sum(w) (line 786). Proof the
// plain version was wrong, independent of sampling: on the linear game
// f(x) = sum c_i x_i it returned (3.25, 5.0, 7.75) against an analytic oracle
// of (0.75, 2.5, 5.25) even with EXHAUSTIVE coalitions and exact SHAP weights;
// the centered version returns the oracle exactly.
//
// Deterministic for a given `seed`: uses a local LCG, never the global rand(),
// so attribution cannot perturb the RNG stream of an unrelated suite.
// Throws std::invalid_argument if n_samples == 0 or shapes mismatch.
ShapleyResult kernel_shap(Model& model,
                          const Tensor& x,
                          const Tensor& baseline,
                          size_t target = 0,
                          size_t n_samples = 200,
                          unsigned seed = 42);

// Path-dependent Shapley values via the permutation estimator.
//
// Each iteration draws a random ordering of the features and walks it,
// crediting every feature with the marginal contribution it makes at its
// position. The marginals are then averaged UNWEIGHTED — there is no
// per-position weight.
//
// BUG FOUND 2026-10-07 (this cost a debugging round and is worth recording):
// this originally applied an "Owen weight" to each marginal. That is wrong,
// and provably so. On a LINEAR game every marginal of feature i equals
// phi_i regardless of the prefix it joins, so averaging over orderings can
// only recover phi_i if the weights are unweighted (or sum to exactly the
// normalization). Three candidate weight families were tested against the
// analytic linear-game oracle (phi_i = c_i * (x_i - b_i)) and all failed:
//   - 1/(M*C(M-1,k))       sums to 2/3 at M=4  -> biased by 1.5x
//   - C(M-1,k)/2^(M-1)     sums to 3 at M=4      -> biased by 3x
//   - k!(M-1-k)!/(M-1)!    sums to 8/3 at M=4    -> biased by 1.5x
// The diagnostic that settles it is the ORDER of error: a wrong weight is a
// SYSTEMATIC bias, so the error does not shrink as n_permutations grows.
// That is the signature to look for when a sampled estimator fails to
// converge to exact_shapley.
//
// The reference implementation (shap/explainers/_permutation.py:209-221) is
// the authority: it accumulates `row_values[ind] += outputs[i+1] - outputs[i]`
// with no weight, walks each ordering forward AND backward (ANTITHETIC
// sampling, which pairs each prefix with a complementary suffix and roughly
// halves the variance), and divides by 2*n_permutations.
//
// Distinct from exact_shapley in cost: O(M) model evaluations per ordering
// instead of O(2^M). Convergence improves as 1/sqrt(n_permutations).
//
// Deterministic for a given `seed` (local LCG, no global rand()).
// Throws std::invalid_argument if n_permutations == 0 or shapes mismatch.
ShapleyResult path_dependent_shap(Model& model,
                                  const Tensor& x,
                                  const Tensor& baseline,
                                  size_t target = 0,
                                  size_t n_permutations = 200,
                                  unsigned seed = 42);

// Fill a ShapleyResult from a raw per-feature attribution vector.
//
// `fx` and `fbaseline` are the model outputs at `x` and `baseline`; they must
// already have been computed by the caller (so the caller controls the number
// of extra model evaluations). `n_coalitions` and `method` are copied
// through. `raw` must have M entries.
//
// `apply_efficiency_residual` controls the reference implementation's
//   phi[nonzero_inds[-1]] = (fx - fnull) - sum(w)
// step (shap/explainers/_kernel.py:786), in which the last feature absorbs the
// error of the other M-1 so that sum(phi) == fx - fnull exactly.
//
// Pass TRUE for a SAMPLED estimator (kernel_shap, path_dependent_shap), whose
// coalition set is a strict subset of the lattice and therefore not efficient
// on its own.
//
// Pass FALSE for exact_shapley. Its exhaustive enumeration is already
// efficient to machine precision, and the residual is not merely redundant
// there — in floating point it overwrites an exactly-computed phi[M-1] with
// the rounding residue of the other M-1, which breaks the SYMMETRY axiom and
// makes the result depend on feature ORDER. See the comment at the call site
// in shapley.cpp.
ShapleyResult finalize_shapley(Model& model,
                               const Tensor& raw,
                               double fx,
                               double fbaseline,
                               size_t n_coalitions,
                               const std::string& method,
                               bool apply_efficiency_residual = true);

#endif // SHAPLEY_H