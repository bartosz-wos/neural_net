#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "../../core/layer.h"

// LayerScale — Cai, Gan, Han, Liu, Chen, Wang (ICCV 2022, arXiv:2103.17239)
// "Going Deeper with Image Transformers" §3.1.
//
// A learnable per-channel multiplicative gate used inside residual blocks:
//   out[i, j] = lambda[j] * x[i, j]
//
// One learnable parameter tensor `lambda ∈ R^{features}`, initialised to
// a small constant `init_value` (paper default 1e-4) so the residual branch
// starts close to identity and the network can learn to scale it up during
// training. Critical for stable training of very deep transformers (Cai et al.
// show this enables > 200-layer vision transformers to converge).
class LayerScale : public Layer {
public:
    Tensor lambda_;           // shape (1, features) — per-channel scale, init init_value
    Tensor last_input;        // cached for backward
    Tensor grad_lambda_;      // shape (1, features)
    Tensor grad_x;            // shape (batch, features)

    size_t features_;
    double init_value_;

    LayerScale(size_t features, double init_value = 1e-4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return lambda_; }
    Tensor get_gradients() const override { return grad_lambda_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    size_t features() const { return features_; }
    double init_value() const { return init_value_; }
    std::string name() const override { return "LayerScale"; }
};

#endif