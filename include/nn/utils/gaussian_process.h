#ifndef NN_UTILS_GAUSSIAN_PROCESS_H
#define NN_UTILS_GAUSSIAN_PROCESS_H

#include "../core/tensor.h"
#include <cstddef>

// ===========================================================================
// Gaussian process regression.
//
//   C. E. Rasmussen, C. K. I. Williams -- "Gaussian Processes for Machine
//   Learning", the MIT Press, Cambridge, 2006. ISBN 026218253X.
//
// CITATION NOTE -- **THERE IS NO arXiv ID AND NONE MAY BE INVENTED.** The
// book is NOT on arXiv. Verified 2026-10-27 by TITLE search against the arXiv
// API (the API itself sanity-checked in the same run -- `id_list=1706.04599`
// returned "On Calibration of Modern Neural Networks", so the endpoint was
// live): a title query for "Gaussian Processes for Machine Learning" returns
// totalResults = 2 and BOTH are unrelated papers (2411.09123 "Quantum
// multi-output Gaussian Processes..." and 2305.12871 "MMGP: a Mesh Morphing
// Gaussian Process..."); a title query with `AND au:Williams_C` returns 0; and
// the official book page gaussianprocess.org/gpml carries no arXiv pointer.
//
// The ID 0605074 that circulates for this book is **WRONG**:
// `https://arxiv.org/abs/0605074` returns HTTP 404, and `id_list=cs/0605074`
// resolves to "SAT Solving for Argument Filterings". This is the exact failure
// class this repo's queue has hit thirteen times: a wrong arXiv ID that is a
// plausible-looking neighbour of the right paper.
//
// The formulas here are read from the primary source PDF
// (https://gaussianprocess.org/gpml/chapters/RW.pdf), and the equation numbers
// quoted in the comments are the book's own.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
//
//     grep -rinE 'gaussian.?process|kernel.?regression|rbf.?kernel|matern|
//                 squaredexp|log.?marginal|gp_regress|gp_fit|gp_predict'
//            include/ tests/ --include=*.h --include=*.cpp
//     -> 0 hits.
//
// The repo has NO Bayes. Every shipped uncertainty method is POST-HOC --
// `utils/conformal.h`, `utils/temperature_scaling.h`, `utils/randomized_
// smoothing.h` and `utils/metrics.h` all calibrate or bound a prediction that
// was already made. A GP is the only one of these that produces the
// predictive distribution FROM THE DATA, with a fitted noise level and a
// fitted length-scale.
//
// The nearest shipped neighbours, and why none of them is this:
//   - `utils/mmd_loss.h` computes RBF/polynomial/IMQ kernel matrices, but for a
//     DISCREPANCY BETWEEN TWO POINT SETS. No posterior, no predictive
//     distribution, no hyperparameters. Its RBF has no `sf^2` amplitude and no
//     length-scale in the GP parameterisation; the two must not be conflated.
//   - `utils/gmm.h` is a FINITE mixture over latent components. A GP is an
//     infinite mixture with a non-diagonal covariance.
//   - `layers/architectures/knn_classifier.h` is non-parametric but
//     DETERMINISTIC: no predictive variance, no marginal likelihood.
//   - `utils/grid_search.h` and `utils/lr_finder.h` search hyperparameters of
//     an EXISTING model's training loop. There is no model to search over --
//     a GP is fitted by the marginal likelihood, not by SGD.
//
// ---------------------------------------------------------------------------
// THE SEMANTICS, AND THE RULES A REIMPLEMENTATION GETS WRONG
//
// (1) THE LENGTH-SCALE IS l, NOT l^2, AND THE AMPLITUDE IS sf^2, NOT sf.
//     The squared-exponential kernel (eq. 2.16, parameterised as eq. 2.31) is
//
//         k(xp, xq) = sf^2 * exp( -0.5 * |xp - xq|^2 / l^2 )
//
//     so the exponent divides by `l^2`. At r = 2, l = 2 the kernel is
//     exp(-0.5) = 0.606531 -- NOT exp(-1) = 0.367879 (the `/l` reading) and
//     NOT exp(-0.25) = 0.778801.
//
// (2) THE NOISE IS ADDED TO THE COVARIANCE OF THE TARGETS, NOT TO THE
//     KERNEL ITSELF. `Ky = Kf + sn^2 I` (eq. 5.8). The noise is `sn^2`, not
//     `sn`.
//
// (3) THE CHOLESKY PATH, NEVER AN EXPLICIT INVERSE. GPML Algorithm 2.1
//     factorises `Ky = L L^T` and solves against `L`. This is not a
//     micro-optimisation: `Ky` is singular whenever `sn = 0` and two training
//     points coincide, and the Cholesky path DETECTS that (a non-positive
//     pivot) where an explicit inverse would silently produce inf/nan.
//
// (4) THE COMPLEXITY PENALTY IS -sum(log L_ii), NOT -1/2 sum(log L_ii).
//     The Cholesky form of `-1/2 log|Ky|` is `-sum(log L_ii)` because
//     `log|Ky| = 2 sum(log L_ii)`. Halving it is a factor-of-two error.
//
// (5) THE PREDICTIVE VARIANCE IS OF THE LATENT FUNCTION f, NOT OF THE
//     TARGET y. It is `k(x*,x*) - v^T v` with `v = L^-1 k_*` (eq. 2.26). The
//     noise variance `sn^2` is ADDED to it for a NOISY test target, per
//     Algorithm 2.1's own note.
//
// (6) THE HYPERPARAMETER GRADIENT IS eq. (5.9):
//
//         d/dtheta log p(y|X, theta) = 0.5 * tr( (alpha alpha^T - Ky^-1) dKy/dtheta )
//
//     with `alpha = Ky^-1 y`. THE `0.5` AND THE `tr` ARE THE WHOLE FORMULA.
//     The book's own footnote 5 prescribes the computation: the
//     vector-by-matrix product first, and ONLY the diagonal terms of the
//     matrix product in the trace.
//
//     **THE TRACE IS NOT THE ELEMENTWISE SUM.** For symmetric M and dK,
//     `tr(M dK)` and `sum_ij M_ij dK_ij` are EXACTLY equal (measured:
//     -8.028646365323734 against -8.028646365323732), so an implementation
//     that computes the elementwise sum is silently relying on symmetry. This
//     file computes `sum_ij M_ij dK_ji`, which is correct for any dK.
//
// (7) THE LEAVE-ONE-OUT FORMULA USES Ky, THE NOISY MATRIX. GPML eq. (5.12)
//     is written in terms of "K", which is AMBIGUOUS -- eq. (5.8) had just
//     introduced `Ky = Kf + sn^2 I` as a distinct object. The book's own
//     tie-breaker sentence ("the mean mu_i is in fact independent of y_i")
//     does NOT decide it, because that identity is pure algebra and holds for
//     EITHER matrix. It was resolved by EXACT REFIT (drop point i, fit on the
//     remaining n-1, predict i): the `Ky` reading agrees to 3.331e-16 (mean)
//     and 2.220e-16 (variance); the `Kf` reading differs by 1.007476 and
//     0.427904. **This file uses Ky.**
//
// (8) THE LOO INDEPENDENCE IS PER-i, NOT GLOBAL. From eq. (5.12),
//
//         [Ky^-1 y]_i = [Ky^-1]_ii y_i + sum_{j != i} [Ky^-1]_ij y_j
//         mu_i = y_i - [Ky^-1 y]_i / [Ky^-1]_ii
//              = y_i - y_i - sum_{j != i} [Ky^-1]_ij y_j / [Ky^-1]_ii
//
//     the two `y_i` terms CANCEL EXACTLY, so `mu_i` alone is independent of
//     `y_i` -- while the OTHER `mu_j` legitimately MOVE, because for them
//     `y_i` is a training target.
//
// ---------------------------------------------------------------------------
// WHY THERE IS NO FINITE-DIFFERENCE TEST FOR THE POSTERIOR MEAN
//
// The predictive mean is linear in `y` and smooth in the hyperparameters, so
// a central difference is valid there. It is used in the test suite for the
// HYPERPARAMETER GRADIENT with eps = 1e-5, which was MEASURED to be the
// sweet spot (rel err 7.084e-10; below 1e-6 the round-off term takes over,
// above 1e-3 the truncation term does).
//
// ---------------------------------------------------------------------------
// OUT OF SCOPE FOR v1, stated so it cannot creep in:
//   - Sparse / variational GPs (Titsias; Wilson et al., "GPyTorch",
//     arXiv:1810.00303). A different algorithm with a different cost.
//   - Multi-output / multi-task GPs. A different model.
//   - Non-Gaussian likelihoods (Bernoulli / Bernoulli-probit GPs).
//   - GP classification (Margossian et al., arXiv:1710.09430). A different
//     deliverable.
//   - The rational-exponential and other kernels; only the SE and the Matern
//     3/2 and 5/2 families are provided.
//   - GPU / blocked computation.
//   - Any change to `utils/mmd_loss.h`.
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// The covariance functions.
// ---------------------------------------------------------------------------
enum GPKernelType {
    // Squared exponential (eq. 2.16 / 2.31):
    //     k(x, x') = sf^2 * exp( -0.5 * ||x - x'||^2 / l^2 )
    GP_KERNEL_SE = 0,
    // Matern 3/2 (eq. 4.17):
    //     k(x, x') = sf^2 * (1 + sqrt(3) r / l) exp( -sqrt(3) r / l )
    GP_KERNEL_MATERN32 = 1,
    // Matern 5/2 (eq. 4.17):
    //     k(x, x') = sf^2 * (1 + sqrt(5) r / l + 5 r^2 / (3 l^2))
    //                    * exp( -sqrt(5) r / l )
    GP_KERNEL_MATERN52 = 2
};

// ---------------------------------------------------------------------------
// The posterior over the latent function f, at a set of query points.
//
// `mean` and `variance` are (m, 1) where m is the number of query points.
// `variance` is the posterior variance of the LATENT f -- the noise variance
// is NOT included. Use `predict` for the noisy-target variance.
// ---------------------------------------------------------------------------
struct GPPosterior {
    Tensor mean;      // (m, 1)
    Tensor variance;  // (m, 1)
};

// ---------------------------------------------------------------------------
// The leave-one-out predictive quantities (eq. 5.12), for every training
// point. `mean` and `variance` are (n, 1).
//
// `variance` is the predictive variance of a NOISY target y_i, i.e. it
// INCLUDES sn^2. It does NOT depend on the targets at all.
// ---------------------------------------------------------------------------
struct GPLoo {
    Tensor mean;      // (n, 1)
    Tensor variance;  // (n, 1)
};

// ---------------------------------------------------------------------------
// The gradient of the log marginal likelihood with respect to the three
// hyperparameters (eq. 5.9). The derivatives are with respect to l, sf and
// sn THEMSELVES -- not to log l, log sf or log sn.
// ---------------------------------------------------------------------------
struct GPHypergradient {
    double d_lengthscale = 0.0;
    double d_sigma_f     = 0.0;
    double d_sigma_n     = 0.0;
};

// ---------------------------------------------------------------------------
// gp_kernel_matrix -- the covariance matrix k(X, X2).
//
//   X    (n, d)   the first point set.
//   X2   (m, d)   the second point set; must have the same number of
//                 COLUMNS as X (the same dimensionality), but may have a
//                 different number of rows.
//   lengthscale   must be finite and > 0.
//   sigma_f       must be finite and > 0.
//   kernel        one of GPKernelType.
//
// Returns (n, m). The squared Euclidean distance is over ALL d dimensions.
//
// THROWS std::invalid_argument on: an empty X or X2; a column-count mismatch
// between X and X2; a non-positive or non-finite length-scale or signal
// variance; an unknown kernel type. Every message is DISTINCT -- a substring
// shared by two messages discriminates nothing.
// ---------------------------------------------------------------------------
Tensor gp_kernel_matrix(const Tensor& X, const Tensor& X2,
                         double lengthscale, double sigma_f,
                         GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_log_marginal_likelihood -- log p(y | X, theta) (eq. 2.30 / 5.8).
//
//   log p = -0.5 * y^T alpha - sum(log L_ii) - (n/2) * log(2 pi)
//   with   Ky = Kf + sn^2 I,  L = chol(Ky),  alpha = L^-T L^-1 y.
//
// `y` is (n, 1) and must have as many rows as X has.
//
// THROWS std::invalid_argument on: an empty X; a y whose row count differs
// from X's; a non-positive or non-finite length-scale or signal variance; a
// negative or non-finite noise variance; an unknown kernel type.
//
// THROWS std::invalid_argument on a SINGULAR Ky -- specifically when `sn` is
// 0 and two training points coincide, or more generally when the Cholesky
// factorisation encounters a non-positive pivot. The message says "not
// positive definite". **This is load-bearing**: the alternative (forming the
// inverse explicitly) would silently return inf/nan instead.
// ---------------------------------------------------------------------------
double gp_log_marginal_likelihood(const Tensor& X, const Tensor& y,
                                  double lengthscale, double sigma_f,
                                  double sigma_n, GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_posterior -- the posterior over the latent function f at the TRAINING
// points, i.e. the query set is X itself.
//
// Returns a GPPosterior with `mean` and `variance` both (n, 1).
//
// `variance` is the posterior variance of the LATENT f, so it is ~0 at a
// training point when `sn = 0` (the noise-free GP interpolates its data) and
// is always bounded by `sigma_f^2`.
//
// Throws the same way as gp_log_marginal_likelihood.
// ---------------------------------------------------------------------------
GPPosterior gp_posterior(const Tensor& X, const Tensor& y,
                          double lengthscale, double sigma_f,
                          double sigma_n, GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_predict -- the posterior at ARBITRARY query points.
//
//   Xs    (m, d)   the query points; d must equal X's column count.
//   Xs    may be empty, in which case the result is empty.
//
// The returned `variance` is the variance of the LATENT f. For the variance
// of a NOISY target add `sigma_n^2` to it -- see `gp_predict_noisy`.
//
// Throws the same way as gp_log_marginal_likelihood, plus on an empty Xs with
// a distinct message.
// ---------------------------------------------------------------------------
GPPosterior gp_predict(const Tensor& X, const Tensor& y, const Tensor& Xs,
                        double lengthscale, double sigma_f,
                        double sigma_n, GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_predict_noisy -- gp_predict, with `sigma_n^2` ADDED to the variance.
//
// This is the predictive distribution of the TARGET y* rather than of the
// latent f*, per Algorithm 2.1's note: "to compute the predictive
// distribution for noisy test data y*, simply add the noise variance sn^2 to
// the predictive variance of f*".
// ---------------------------------------------------------------------------
GPPosterior gp_predict_noisy(const Tensor& X, const Tensor& y, const Tensor& Xs,
                              double lengthscale, double sigma_f,
                              double sigma_n, GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_hyperparameter_gradient -- d/dtheta log p(y | X, theta) (eq. 5.9).
//
// The derivatives are with respect to the hyperparameters THEMSELVES.
//
// This is the quantity a GP is FITTED with: the hyperparameters are chosen to
// maximise the log marginal likelihood (eq. 5.8). It is not a gradient of any
// loss over a trained network, and this repo's optimizers are not the
// intended consumer -- see the header's note that the GP is fitted by the
// marginal likelihood, not by SGD.
//
// The computation follows the book's footnote 5: the vector-by-matrix
// product first, and only the diagonal terms of the matrix product in the
// trace.
//
// Throws the same way as gp_log_marginal_likelihood.
// ---------------------------------------------------------------------------
GPHypergradient gp_hyperparameter_gradient(const Tensor& X, const Tensor& y,
                                           double lengthscale, double sigma_f,
                                           double sigma_n, GPKernelType kernel);

// ---------------------------------------------------------------------------
// gp_leave_one_out -- the leave-one-out predictive quantities (eq. 5.12),
//
//     mu_i = y_i - [Ky^-1 y]_i / [Ky^-1]_ii
//     s2_i = 1 / [Ky^-1]_ii
//
// where Ky = Kf + sn^2 I is the NOISY covariance matrix (see the header's
// rule (7): the book writes "K", which is ambiguous, and the oracle resolved
// it by exact refit).
//
// `mean` is the LOO prediction of y_i and `variance` is the LOO predictive
// variance of y_i, which INCLUDES sn^2.
//
// `mu_i` is INDEPENDENT of `y_i` (the two y_i terms cancel exactly in the
// algebra of eq. 5.12), while the other `mu_j` legitimately move when `y_i`
// changes. `variance` is independent of the targets entirely.
//
// Throws the same way as gp_log_marginal_likelihood.
// ---------------------------------------------------------------------------
GPLoo gp_leave_one_out(const Tensor& X, const Tensor& y,
                        double lengthscale, double sigma_f,
                        double sigma_n, GPKernelType kernel);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_GAUSSIAN_PROCESS_H
