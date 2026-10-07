#include "attribution.h"
#include <stdexcept>

Tensor input_gradient(Model& model, const Tensor& input, size_t target) {
    if (target >= input.cols) {
        throw std::out_of_range("input_gradient: target index out of range");
    }

    // Dense::backward ACCUMULATES into grad_weights/grad_bias (layer.cpp:79).
    // Zero first so the pass does not add to whatever training left behind.
    for (auto& l : model.layers) l->zero_grad();

    Tensor out = model.forward(input);
    if (target >= out.cols) {
        throw std::out_of_range(
            "input_gradient: target index out of range for model output");
    }

    // Seeding the output with a one-hot on `target` makes Model::backward
    // compute d(logit_target)/d(input) — the integral in the IG definition.
    Tensor seed(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i) {
        seed[i][target] = 1.0;
    }

    Tensor grad = model.backward(seed, 0.0);

    // Leave no residue: this is an attribution pass, not a training step.
    for (auto& l : model.layers) l->zero_grad();
    return grad;
}

void riemann_params(const std::string& method, size_t n,
                    std::vector<double>& step_sizes,
                    std::vector<double>& alphas) {
    if (n < 1) {
        throw std::invalid_argument("riemann_params: n must be >= 1");
    }
    const double dn = static_cast<double>(n);

    step_sizes.assign(n, 1.0 / dn);
    alphas.assign(n, 0.0);

    if (n == 1) {
        // Degenerate but must not divide by zero: treat the single node as the
        // right endpoint with full weight.
        step_sizes[0] = 1.0;
        alphas[0] = 1.0;
        return;
    }

    if (method == "riemann_trapezoid") {
        // NOTE: a deliberate, documented deviation from captum.
        // captum's riemann_builders uses a base step of 1/n for ALL variants,
        // but its trapezoid alphas are linspace(0, 1, n) — i.e. spaced 1/(n-1)
        // apart. Pairing 1/(n-1)-spaced nodes with a 1/n step makes captum's
        // trapezoid weights sum to (n-1)/n, not 1, so its trapezoidal rule
        // under-integrates by that factor. We use the matching 1/(n-1) base
        // step so the quadrature weights sum to exactly 1, as the other three
        // rules do. Verified numerically: for n=7 captum's weights sum to
        // 0.857143; these sum to 1.0.
        const double base = 1.0 / (dn - 1.0);
        step_sizes.assign(n, base);
        step_sizes[0] /= 2.0;
        step_sizes[n - 1] /= 2.0;
        for (size_t k = 0; k < n; ++k) {
            alphas[k] = static_cast<double>(k) / (dn - 1.0);
        }
    } else if (method == "riemann_left") {
        for (size_t k = 0; k < n; ++k) {
            alphas[k] = static_cast<double>(k) / dn;
        }
    } else if (method == "riemann_right") {
        for (size_t k = 0; k < n; ++k) {
            alphas[k] = (static_cast<double>(k) + 1.0) / dn;
        }
    } else if (method == "riemann_middle") {
        for (size_t k = 0; k < n; ++k) {
            alphas[k] = (static_cast<double>(k) + 0.5) / dn;
        }
    } else {
        throw std::invalid_argument(
            "riemann_params: unknown method '" + method + "'");
    }
}

IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   const Tensor& baseline,
                                   size_t target,
                                   size_t n_steps,
                                   const std::string& method) {
    if (input.rows != baseline.rows || input.cols != baseline.cols) {
        throw std::invalid_argument(
            "integrated_gradients: baseline shape must match input");
    }

    std::vector<double> step_sizes, alphas;
    IGAttribution result;
    result.n_steps = n_steps;
    result.method = method;
    result.attributions = Tensor::zeros(input.rows, input.cols);

    // n_steps == 0 is a degenerate but legal request: there is no path to
    // integrate, so the attribution is exactly zero. Handle it here rather
    // than letting riemann_params reject n=0, so the returned value is
    // well-defined (and free of 0/0) instead of throwing.
    if (n_steps == 0) {
        Tensor fx = model.forward(input);
        Tensor fb = model.forward(baseline);
        if (target < fx.cols && target < fb.cols) {
            result.completeness_delta = -(fx[0][target] - fb[0][target]);
        }
        return result;
    }

    riemann_params(method, n_steps, step_sizes, alphas);

    Tensor total_grad = Tensor::zeros(input.rows, input.cols);
    double total_weight = 0.0;

    for (size_t k = 0; k < n_steps; ++k) {
        const double a = alphas[k];

        // Point on the straight-line path from baseline to input.
        Tensor point(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                point[i][j] = baseline[i][j] + a * (input[i][j] - baseline[i][j]);
            }
        }

        Tensor g = input_gradient(model, point, target);

        for (size_t i = 0; i < total_grad.rows; ++i) {
            for (size_t j = 0; j < total_grad.cols; ++j) {
                total_grad[i][j] += step_sizes[k] * g[i][j];
            }
        }
        total_weight += step_sizes[k];
    }

    // Every supported rule has total weight exactly 1, so this is a no-op for
    // them — it guards a future unequal-weight quadrature and keeps the
    // completeness delta honest.
    if (total_weight > 0.0) {
        total_grad = total_grad * (1.0 / total_weight);
    }

    // captum order: multiply by (input - baseline) AFTER the weighted sum.
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            result.attributions[i][j] =
                (input[i][j] - baseline[i][j]) * total_grad[i][j];
        }
    }

    // Completeness axiom: sum(IG) should equal F(x) - F(x').
    Tensor fx = model.forward(input);
    Tensor fb = model.forward(baseline);
    if (target < fx.cols && target < fb.cols) {
        result.completeness_delta =
            result.attributions.sum() - (fx[0][target] - fb[0][target]);
    }
    return result;
}

IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   size_t target,
                                   size_t n_steps,
                                   const std::string& method) {
    return integrated_gradients(model, input, Tensor::zeros(input.rows, input.cols),
                                target, n_steps, method);
}