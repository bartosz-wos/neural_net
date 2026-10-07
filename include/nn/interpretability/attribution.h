#ifndef ATTRIBUTION_H
#define ATTRIBUTION_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <string>
#include <vector>

// =============================================================================
// Input attribution — Integrated Gradients and the gradient-only baseline.
//
// Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks", ICML
// 2017, arXiv:1703.01365.
//
//   IG_i(x) = (x_i - x'_i) * integral_0^1 d(alpha) [ dF(x' + alpha(x - x'))/dx_i ]
//
// The defining property, the COMPLETENESS axiom:
//
//   F(x) - F(x') = sum_i IG_i(x)
//
// i.e. the attributions of a single example always sum to the model's actual
// output change. That is what distinguishes IG from gradient x input saliency,
// whose attributions need not sum to anything meaningful.
//
// Why this needed no new infrastructure: this repo's `Layer::backward()`
// RETURNS dL/d(input) rather than storing it, and `Model::backward()` threads
// that returned tensor down to the model input. So dF/dx is already available
// to a caller. (Contrast Grad-CAM, which needs dL/d(input) of an INTERMEDIATE
// layer — nothing caches that here.)
//
// CONVENTION — the model output must be (rows, n_classes) and `target`
// selects which output column is the scalar being attributed.
//
//   IG attributes the LOGIT, never a softmax probability. The paper defines F
//   on the pre-softmax score; probabilities saturate and their gradients
//   vanish, which is the very pathology IG exists to fix.
// =============================================================================

// Result of an attribution pass.
struct IGAttribution {
    // Per-element attributions, same (rows, cols) shape as the input.
    Tensor attributions;
    // sum(attributions) - (F(x) - F(x')). Zero is the completeness axiom;
    // a nonzero value is the Riemann quadrature error, which shrinks as
    // n_steps grows. Always finite.
    double completeness_delta = 0.0;
    size_t n_steps = 0;         // number of steps m actually used
    std::string method;         // e.g. "riemann_middle"
};

// Gradient of a single scalar model output w.r.t. the model input.
//
// Equivalent to d(logit[target])/dx. Performs one forward + one backward.
// Note: `Dense::backward` accumulates into parameter gradients, so this
// function zeroes the model's parameter gradients before and after the pass —
// it is side-effect-free with respect to training state.
Tensor input_gradient(Model& model, const Tensor& input, size_t target = 0);

// Riemann sum parameters, mirroring PyTorch captum's
// `captum/attr/_utils/approximation_methods.py` exactly.
//
// Supported methods and their (step_sizes, alphas):
//   riemann_left       step 1/n;             alphas = k/n           for k=0..n-1
//   riemann_right      step 1/n;             alphas = (k+1)/n
//   riemann_middle     step 1/n;             alphas = (k+0.5)/n     [default]
//   riemann_trapezoid  endpoints halved;     alphas = k/(n-1)
//
// For every method the step sizes sum to exactly 1.
void riemann_params(const std::string& method, size_t n,
                    std::vector<double>& step_sizes,
                    std::vector<double>& alphas);

// Integrated Gradients attribution of logit `target` to the input.
//
// `baseline` is the reference point x' — conventionally zeros, an empirical
// mean image, or a blurred version of the input. Its shape must match
// `input`. Throws std::invalid_argument on a shape mismatch or an unknown
// method, and std::out_of_range if `target` exceeds the model's output width.
//
// The aggregation order matches captum: gradients are weighted by step size
// and summed FIRST, then multiplied by (input - baseline). Doing it the other
// way round gives a different answer whenever the step sizes are unequal
// (i.e. for the trapezoidal rule).
IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   const Tensor& baseline,
                                   size_t target = 0,
                                   size_t n_steps = 50,
                                   const std::string& method = "riemann_middle");

// Zero-baseline convenience overload.
IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   size_t target = 0,
                                   size_t n_steps = 50,
                                   const std::string& method = "riemann_middle");

#endif // ATTRIBUTION_H