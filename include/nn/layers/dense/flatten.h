#ifndef FLATTEN_H
#define FLATTEN_H

#include "../../core/layer.h"

// Flattens spatial dimensions into a single channel
// Input: (batch, spatial_features) or (batch, channels, H, W) stored as (batch, channels*H*W)
// Output: (batch, flattened)
class Flatten : public Layer {
public:
    Flatten() {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /* learning_rate */) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
};

#endif