#ifndef NN_UTILS_HUTCHINSON_H
#define NN_UTILS_HUTCHINSON_H

#include "../core/tensor.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <cstddef>
#include <functional>
#include <vector>

// ===========================================================================
// Hutchinson stochastic Hessian-trace estimation and forward-over-reverse
// Hessian-vector products.
//
//   Hutchinson, M. F. (1989/90). "A stochastic estimator of the trace of the
//   influence matrix for Laplacian smoothing splines." Comm. Statist.
//   Simulation and Computation 18(3):1059-1076.
//
//   Pearlmutter, B. A. (1995). "Gradient calculations for dynamic recurrent
//   neural networks: a survey." IEEE Trans. Neural Networks 6(5):1212-1228.
//
// BOTH SOURCES ARE PRE-arXiv, so there is NO arXiv ID to verify here and NONE
// MAY BE INVENTED -- this file has a sibling note about a queue that produced
// nine wrong arXiv IDs. The Pearlmutter paper is very widely cited as "1994",
// INCLUDING by PyTorch (whose torch.autograd.functional.hvp docstring says
// "d. Pearlmutter (1994)"), but dblp's authoritative bibliographic record is
// journals/tnn/Pearlmutter95 -- IEEE Trans. Neural Networks 6(5):1212-1228,
// 1995. Recorded here rather than silently reconciled; do not "fix" it to 1994.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
//
// This repo's only second-order optimizer, Sophia, documents in its OWN header
// what it wants and cannot get:
//
//     optimizers/sophia.h:25
//         "before step() -- typically with a Hutchinson trace estimate."
//
// and exposes `set_hessian_estimates(layer_ptr, h_diag)` to receive one. But
//
//     grep -rn 'set_hessian_estimates' --include=*.cpp --include=*.h .
//
// returns ONLY sophia.{h,cpp} and tests/test_sophia.cpp -- and the test feeds
// it a HAND-MADE tensor. Nothing in the repository computes an estimate. Every
// real training run therefore falls back to Sophia-G's empirical Fisher
// h = g (.) g, and the higher-order path is unreachable in practice. A file
// that documents a limitation IS a specification; this one names the
// estimator, in the repo's own words.
//
// It is deliberately NOT a Newton step, a trust region or an L-BFGS. The
// measured gap is *an estimator nobody computes*, not a missing optimizer --
// see the queue's recurring lesson, which is that an absence proves a feature
// is missing, not that it belongs.
//
// ---------------------------------------------------------------------------
// WHY THERE IS NO FINITE-DIFFERENCE TEST FOR THE ESTIMATOR
//
// The forward here is STOCHASTIC. A centred difference (f(x+e)-f(x-e))/2e is
// the difference of two UNCORRELATED noise draws, not a consistent difference
// quotient of any function (this repo's systematic-debugging 5c-bis). The
// estimator is verified by VALUE against a self-checking float64 numpy oracle
// plus mutation testing. Do not add an FD "gradient check" here; it will pass
// vacuously or fail for reasons that have nothing to do with the code.
//
// ---------------------------------------------------------------------------
// THE ORACLE CAUGHT ITS OWN BUG BEFORE ANY C++ EXISTED
//
// The numpy oracle's first variance check drew TWO independent probes, i.e. it
// measured E[z1' H z2] with z1 != z2. For symmetric H that expectation is
// ZERO, not tr(H) -- the symptom was means hovering around 0.0 while the true
// trace was 0.442. The fingerprint (large, shape-independent error) said "my
// estimator is wrong", not "the theory is wrong" (systematic-debugging 4c:
// in a differential test, suspect the piece of code you just wrote). The
// estimator is z' H z with ONE z.
//
// ---------------------------------------------------------------------------
// NUMBERS THE ORACLE FIXED, WHICH ARE THE SEMANTICS BELOW
//
//   E[z'Hz] == tr(H), Rademacher probes        z = 0.06 s.e.
//   E[z'Hz] == tr(H), Gaussian probes           z = 0.60 s.e.
//   spread at k=1 is 3004% of |tr(H)|           -> k=1 is a poor estimator
//   spread falls as 1/sqrt(k), ratio 0.98-1.01  over k = 1..1024
//   FOM error falls ~10x per 10x eps            -> FIRST order, O(eps)
//   eps = 1e-16 gives rel_err 0.985              -> cancellation floor below ~1e-12
// ===========================================================================

// The gradient oracle the estimator consumes: given the current point, return
// dL/d(that point). Identical in shape to utils/adversarial.h's `GradientFn` --
// same convention, same rationale: it is the whole interface between this
// utility and any model, and it means the estimator is testable with no Model,
// no optimizer and no network at all.
using HessianGradFn = std::function<Tensor(const Tensor&)>;

// ---------------------------------------------------------------------------
// Trace of the Hessian of the loss at `x`, estimated with `probes` random
// vectors:
//
//     tr(H) = E_z[ z' H z ]        for E[z z'] = I
//
//   grad(x + z) - grad(x) == H z   EXACTLY, for any H (the linear term of the
//   gradient cancels), which is why the estimator needs no loss value and no
//   second backward pass.
//
//   dim    -- dimension of the probe; must equal x.cols. Must be > 0.
//   grad   -- the gradient oracle. Must be callable.
//   x      -- the point at which to estimate tr(H).
//   probes -- number of probes to average; must be >= 1. The estimate is
//             UNBIASED for any probes >= 1, but its spread falls as
//             1/sqrt(probes): at probes == 1 the spread is thousands of percent
//             of the trace on a typical fixture, so treat k=1 as a smoke test
//             and k >= 20 as a usable estimate.
//   seed   -- seed for the probe RNG. seed == 0 selects a fixed default seed,
//             so the default is REPRODUCIBLE. Two calls with the same seed
//             return bit-identical results; two calls with different seeds
//             return different results.
//
// Throws std::invalid_argument if dim == 0, probes < 1, x.cols != dim, or
// grad is empty.
//
// D3  a single probe is a poor estimator -- this is a property of the METHOD,
//     not a bug, and the test asserts it rather than hiding it.
// D7  determinism comes from a LOCAL generator, never the global rand(): a
//     global-RNG implementation would be unreproducible AND would perturb
//     every other stochastic layer in the repo.
// D9  dim == 0 or probes < 1 throws rather than returning 0.0 or NaN, which
//     would silently poison a Sophia run downstream.
double hutch_trace_rademacher(size_t dim, const HessianGradFn& grad,
                              const Tensor& x, int probes = 1, unsigned seed = 0);

double hutch_trace_gaussian(size_t dim, const HessianGradFn& grad,
                            const Tensor& x, int probes = 1, unsigned seed = 0);

// ---------------------------------------------------------------------------
// Forward-over-reverse Hessian-vector product: returns H v, an approximation
// to the Hessian of the loss at x applied to v.
//
//     H v ~= ( grad(x + eps*v) - grad(x) ) / eps
//
// The error is O(eps) -- FIRST order. This is a forward DIFFERENCE, not a
// symmetric one, and the order is pinned by a test because the symmetric
// variant also converges and an implementation might silently swap them.
// Despite the name it needs only TWO gradient evaluations, not two backward
// passes through a model.
//
//   eps -- must be > 0 and >= 1e-12. The lower bound is NOT arbitrary: the
//          numerator is a difference of two nearly equal gradient vectors, so
//          below ~1e-12 catastrophic cancellation returns a plausible-looking
//          WRONG answer rather than an obvious failure (measured rel_err 0.985
//          at eps = 1e-16). The header throws rather than returning garbage.
//          In practice eps ~ 1e-4 to 1e-2 is accurate; there is no accuracy to
//          be gained below the floor, only error to be added.
//
// The returned tensor is always FRESH -- it never aliases x or v.
Tensor hutch_hvp(const HessianGradFn& grad, const Tensor& x, const Tensor& v,
                 double eps = 1e-4);

// ---------------------------------------------------------------------------
// PER-TENSOR HESSIAN DIAGONAL ESTIMATES for a Model's parameters -- the shape
// Sophia::set_hessian_estimates(layer_ptr, h_diag) expects, which is the
// actual reason this file exists.
//
// The outer product identity decomposes per tensor:
//
//     tr(H) = sum_i z_i' H_ii z_i
//
// where H_ii is the diagonal BLOCK of the Hessian with respect to parameter
// tensor i. Each parameter tensor gets one Tensor of the same shape as itself,
// in the same order as Layer::parameters() returns them, so the result can be
// handed to set_hessian_estimates WITHOUT reindexing.
//
// Every parameter is restored BIT-IDENTICALLY before this function returns: the
// estimator perturbs theta to theta + eps*v and must not leave the caller's
// model modified. This is the same snapshot discipline utils/lr_multiplier.cpp
// implements, and for the same reason -- a helper that mutates the caller's
// model is a landmine even when it restores the value.
//
// `probes < 1`, or a model with no parameters, throws std::invalid_argument.
std::vector<std::vector<Tensor>> hutch_trace_diagonal(
    Model& model, const Tensor& x, const Tensor& dL_dy,
    int probes = 1, unsigned seed = 0, double eps = 1e-4);

#endif  // NN_UTILS_HUTCHINSON_H