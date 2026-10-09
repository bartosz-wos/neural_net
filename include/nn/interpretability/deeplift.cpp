#include "deeplift.h"

#include "../core/layer.h"
#include "../core/model.h"
#include "../core/tensor.h"

#include <stdexcept>

DeepLiftResult deep_lift(Model& model,
                         const Tensor& input,
                         const Tensor& baseline,
                         size_t target,
                         double eps) {
    if (input.rows != baseline.rows || input.cols != baseline.cols) {
        throw std::invalid_argument(
            "deep_lift: baseline shape must match input");
    }
    if (!(eps > 0.0)) {
        throw std::invalid_argument("deep_lift: eps must be > 0");
    }

    const size_t L = model.layers.size();
    if (L == 0) {
        throw std::invalid_argument("deep_lift: model has no layers");
    }

    // Dense::backward ACCUMULATES into grad_weights/grad_bias (layer.cpp:78).
    // Zero first so the backward walk does not add to whatever training left
    // behind, and again at the end so the pass leaves no residue.
    for (auto& l : model.layers) l->zero_grad();

    // Per-layer (input, output) activations for both passes.
    //
    // The REFERENCE pass runs FIRST (plan D6). Dense::backward reads its cached
    // last_input, so if the reference ran last, every cached last_input would
    // hold the reference activations during the backward walk. The returned
    // grad_input happens not to use last_input, but relying on that internal
    // detail is exactly the fragility this ordering avoids.
    std::vector<Tensor> ref_in, ref_out, act_in, act_out;
    ref_in.reserve(L);
    ref_out.reserve(L);
    act_in.reserve(L);
    act_out.reserve(L);

    Tensor cur = baseline;
    for (auto& l : model.layers) {
        ref_in.push_back(cur);
        cur = l->forward(cur);
        ref_out.push_back(cur);
    }
    cur = input;
    for (auto& l : model.layers) {
        act_in.push_back(cur);
        cur = l->forward(cur);
        act_out.push_back(cur);
    }

    const Tensor& out = act_out.back();
    if (target >= out.cols) {
        throw std::out_of_range("deep_lift: target index out of range");
    }

    // The adjoint starts as one-hot on `target`: we are propagating the
    // derivative of logit[target] itself, exactly as input_gradient's seed.
    Tensor G = Tensor::zeros(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i) G[i][target] = 1.0;

    DeepLiftResult res;
    res.n_layers = L;
    res.attributions = Tensor::zeros(input.rows, input.cols);

    for (size_t k = L; k-- > 0;) {
        Layer* layer = model.layers[k].get();
        // Linear layers: dL/d(in) = J^T dL/d(out), which is precisely what the
        // layer's own backward computes, so the identity multiplier is free.
        // Elementwise nonlinear layers: the Rescale multiplier rescales it.
        const Tensor g_in = layer->supports_rescale()
            ? layer->rescale_backward(G, act_in[k], ref_in[k], eps)
            : layer->backward(G, 0.0);
        const Tensor delta = act_in[k] - ref_in[k];  // (a - a')
        if (k == 0) {
            // The INPUT layer's contribution IS the attribution.
            res.attributions = delta.hadamard(g_in);
        }
        G = g_in;
    }

    const double fx = act_out.back()[0][target];
    const double fp = ref_out.back()[0][target];
    res.completeness_delta = res.attributions.sum() - (fx - fp);

    for (auto& l : model.layers) l->zero_grad();
    return res;
}

DeepLiftResult deep_lift(Model& model,
                         const Tensor& input,
                         size_t target,
                         double eps) {
    return deep_lift(model, input, Tensor::zeros(input.rows, input.cols),
                     target, eps);
}

DeepLiftResult deep_lift_shap(Model& model,
                              const Tensor& input,
                              const std::vector<Tensor>& baselines,
                              size_t target,
                              double eps) {
    if (baselines.empty()) {
        throw std::invalid_argument(
            "deep_lift_shap: at least one baseline is required");
    }
    for (const Tensor& b : baselines) {
        if (b.rows != input.rows || b.cols != input.cols) {
            throw std::invalid_argument(
                "deep_lift_shap: baseline shape must match input");
        }
    }

    // Mean over reference points, so sum(attributions) stays a valid
    // completeness statement for the AVERAGED baseline output: each pass
    // satisfies sum(attr_k) = F(x) - F(x'_k), and the mean of sums is the sum
    // of the mean by linearity.
    const double inv_k = 1.0 / static_cast<double>(baselines.size());
    Tensor total = Tensor::zeros(input.rows, input.cols);
    double delta_total = 0.0;
    size_t n_layers = 0;
    for (const Tensor& b : baselines) {
        DeepLiftResult r = deep_lift(model, input, b, target, eps);
        for (size_t i = 0; i < total.rows; ++i) {
            for (size_t j = 0; j < total.cols; ++j) {
                total[i][j] += r.attributions[i][j] * inv_k;
            }
        }
        delta_total += r.completeness_delta * inv_k;
        n_layers = r.n_layers;
    }

    DeepLiftResult res;
    res.attributions = total;
    res.completeness_delta = delta_total;
    res.n_layers = n_layers;
    return res;
}