#ifndef LIME_H
#define LIME_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <string>
#include <vector>

// =============================================================================
// LIME — Local Interpretable Model-agnostic Explanations.
//
// Ribeiro, Singh, Guestrin, "Why Should I Trust You?: Explaining the
// Predictions of Any Classifier", KDD 2016, arXiv:1606.03878.
//
// LIME fits a LINEAR model to the model's behaviour in a small neighborhood
// around the instance being explained:
//
//   g(z) = b0 + sum_j b_j * z_j
//
// fitted by weighted least squares, where z is the interpretable
// representation and each sample's weight is a decreasing function of its
// distance to the instance:
//
//   w_i = sqrt(exp(-(d_i / omega)^2))      omega = sqrt(M) * 0.75
//
// Three properties follow, and they are what the method is FOR:
//
//  1. LOCAL. The surrogate is fit around ONE instance, so it explains that
//     prediction rather than the model's global behaviour.
//  2. MODEL-AGNOSTIC. The black box is only ever used as a function
//     instance -> scalar. Nothing here inspects gradients or weights.
//  3. SPARSE. The paper's headline claim is that humans use few features;
//     `num_features` truncates the fit to the largest-|b| features.
//
// This is the only attribution method in the repo that uses NO GRADIENTS at
// all — IG (attribution.h) and SHAP (shapley.h) both integrate or average
// dF/dx. That is not just a diversity point: it means the reference is a
// closed-form linear-algebra solve rather than a converged approximation, so
// the whole method is verifiable to machine precision instead of to a
// tolerance (see "CLOSED-FORM ORACLE" below).
//
// AUTHORITATIVE SOURCE. The formulas were read from the reference
// implementations on 2026-10-08, not from memory:
//
//   lime/lime_tabular.py:245-253   kernel_width = sqrt(n_features) * 0.75
//                                  kernel = sqrt(exp(-(d**2)/kernel_width**2))
//   lime/lime_tabular.py:475-540   gaussian neighborhood around the TRAINING MEAN
//                                  data[0] = the instance
//   lime/lime_tabular.py (explain)  scaled_data = (data - mean)/scale
//                                  distances = euclidean(scaled_data, scaled_data[0])
//   lime/lime_base.py:158-200      weights = kernel(distances); Ridge(alpha=1,
//                                  fit_intercept=True).fit(Z, y, sample_weight=w)
//   sklearn .../_ridge.py          objective ||y - Xw||^2 + alpha*||w||^2
//                                  (NO 1/n_samples normalisation)
//
// TWO DETAILS THAT ARE EASY TO GET WRONG AND THAT NO EYE-BALLING CATCHES:
//
//  1. The kernel is sqrt(exp(-d^2/w^2)) — i.e. exp(-d^2/2w^2) — NOT
//     exp(-d^2/w^2). The sqrt() is applied to the RESULT of exp(). The two
//     differ by a factor of two in the exponent, so the wrong one is a
//     perfectly smooth, perfectly plausible, wrong kernel.
//  2. Ridge with fit_intercept=True centers on the SAMPLE-WEIGHTED mean
//     (sklearn's `_preprocess_data`), not the unweighted mean. LIME's kernel
//     weights are strongly non-uniform by construction — the instance itself
//     has weight 1.0 while distant samples have weight ~0 — so the two means
//     differ materially.
//
// CLOSED-FORM ORACLE (why this is unusually testable). For a linear black box
// f(x) = sum_j c_j x_j with no bias, and the default neighborhood
// `raw[i][j] = z[i][j] * sd_j + mean_j`, the response is EXACTLY affine in the
// interpretable coordinates:
//
//   f(raw[i]) = sum_j c_j sd_j * z[i][j] + sum_j c_j mean_j
//
// so the exact ridge solution (at alpha = 0) is
//
//   b_j = c_j * sd_j          b0 = sum_j c_j * mean_j
//   g(z_0) = sum_j b_j z_0[j] + b0 = f(x)          <- the local prediction
//
// That identity pins the kernel, the weighting, the centering, the intercept
// reconstruction, the perturbation scale AND the distance metric in a single
// assertion. A bug anywhere in that chain moves the answer.
//
// CONVENTION — identical to attribution.h and shapley.h: `x` is ONE row (the
// instance to explain), the model output is (rows, n_classes), and `target`
// selects the scalar column being explained. Attribution is on the LOGIT,
// never a softmax probability, because probabilities saturate and the
// gradients/curvature that distinguish nearby inputs vanish.
//
// FEATURE GRANULARITY — one feature per input COLUMN of `x`, matching
// shapley.h. Continuous (Gaussian-perturbation) mode only; the reference's
// categorical / discretizer path is out of scope for v1.
//
// Deterministic for a given `seed`: a local xorshift64* generator, never
// std::rand()/std::normal_distribution, so an explanation cannot perturb the
// RNG stream of an unrelated suite (see the test_adaln_zero intermittent-NaN
// entry in NOT_FIXED.md).
// =============================================================================

// One column of the interpretable design matrix, and the response/weight for
// each neighborhood sample. Public so a caller with their OWN neighborhood
// (e.g. image superpixels, a tabular masker of their own) can fit the LIME
// surrogate directly — the method is the solve, so the solve should be
// reachable without going through lime_explain.
struct LocalSurrogate {
    // Fitted slopes, INDEX-ALIGNED WITH THE CORRESPONDING ENTRY OF
    // LimeResult::ranking (i.e. in descending |b| order). LimeResult also
    // carries `used_features` so the mapping is explicit; use
    // LimeResult::coefficient_for() to look one up by feature index.
    std::vector<double> coefficients;
    // Intercept, reconstructed after weighted centering.
    double intercept = 0.0;
    // sklearn's Ridge.score: 1 - SS_res/SS_tot, with the weights applied to
    // both sums and SS_tot taken about the WEIGHTED mean of y. 1.0 is a
    // perfect local fit; negative means the surrogate is worse than the
    // weighted mean.
    double score = 0.0;
};

// Configuration for lime_explain. Defaults reproduce the reference
// implementation's defaults for a continuous tabular explainer.
struct LimeConfig {
    // Neighborhood size (the reference's `num_samples`, default 5000).
    size_t n_samples = 5000;
    // Number of features kept in the surrogate. 0 means "all features",
    // matching the reference's convention that `num_features` is a maximum
    // rather than a requirement.
    size_t num_features = 0;
    // Ridge alpha. The reference uses 1. NOTE: any alpha > 0 shrinks the
    // coefficients, so tests asserting EXACT coefficient recovery must use
    // alpha = 0 (see the header's closed-form oracle).
    double alpha = 1.0;
    unsigned seed = 42;
    // Per-feature standardization used to perturb the neighborhood
    // (the reference's scaler mean/scale). Shape (1, M). Left empty they
    // default to 0 and 1 respectively, i.e. no standardization.
    Tensor mean;
    Tensor std;
    // Distance-kernel width. 0 (default) means exp_kernel_width(M).
    double kernel_width = 0.0;
};

// Result of one LIME explanation.
struct LimeResult {
    // Slopes, in descending |b| order. See LocalSurrogate::coefficients and
    // `ranking` for the feature mapping.
    std::vector<double> coefficients;
    // Feature indices, in descending |b| order. ranking[i] is the feature
    // that coefficients[i] belongs to.
    std::vector<size_t> ranking;
    // |coefficients|, same order. Always non-negative.
    std::vector<double> importances;
    double intercept = 0.0;
    // The surrogate's prediction AT THE INSTANCE, g(z_0). For an exactly
    // linear black box at alpha=0 this equals f(x) to machine precision;
    // for a nonlinear one it is the surrogate's (approximate) answer, which
    // is the number LIME actually reports to a user.
    double local_prediction = 0.0;
    // f(x) itself — the quantity `local_prediction` is approximating.
    double base_prediction = 0.0;
    // Local fidelity: the weighted R^2 of the surrogate on the neighborhood.
    double score = 0.0;
    size_t n_features = 0;
    size_t n_samples = 0;
    size_t target = 0;
    // The configuration actually used, with mean/std/kernel_width filled in.
    LimeConfig config;
};

// LIME's exponential proximity kernel: sqrt(exp(-(d/omega)^2)).
//
// The sqrt() is applied to the RESULT of exp() — see the header. Returns 1.0
// at d == 0.
double exp_kernel(double d, double kernel_width);

// The reference's default kernel width for M features: sqrt(M) * 0.75.
// Throws std::invalid_argument if n_features == 0.
double exp_kernel_width(size_t n_features);

// Weighted ridge regression with an intercept: fit y ~ b0 + sum_j b_j z_j
// minimizing sum_i w_i (y_i - g(z_i))^2 + alpha * ||b||^2.
//
// `z` holds one column per entry of `z[j]` (each of length n_samples), `y`
// and `weights` are length n_samples. Centering is by the SAMPLE-WEIGHTED
// mean, matching sklearn's Ridge(fit_intercept=True).
//
// The `+ alpha * I` term is part of the objective — without it `alpha` would
// be inert and this would silently be ordinary least squares.
//
// Throws std::invalid_argument if alpha < 0, if any vector has a length other
// than n_samples, if `z` is empty, or if n_samples == 0.
LocalSurrogate fit_local_surrogate(const std::vector<std::vector<double>>& z,
                                   const std::vector<double>& y,
                                   const std::vector<double>& weights,
                                   double alpha);

// Explain the model's `target` output at `x` with a local linear surrogate.
//
// Builds a Gaussian neighborhood of size `cfg.n_samples` around the training
// mean implied by `cfg.mean` / `cfg.std`, replaces row 0 with `x` itself
// (so the instance always gets kernel weight 1.0 and appears in the fit),
// evaluates the model on every row in ONE batched forward pass, and fits the
// weighted ridge surrogate.
//
// Throws std::invalid_argument on n_samples == 0, a non-positive std entry,
// or a mean/std whose width differs from x's; std::out_of_range if `target`
// exceeds the model output width.
//
// Leaves no residue in the model's parameter gradients (zeroed before and
// after), matching input_gradient and the SHAP estimators.
LimeResult lime_explain(Model& model,
                        const Tensor& x,
                        const LimeConfig& cfg,
                        size_t target = 0);

#endif // LIME_H