#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include "../../core/layer.h"

// Layer Normalization: normalize over features (like Transformer pre-norm)
// y = gamma * (x - mu) / sqrt(var + eps) + beta
class LayerNorm : public Layer {
public:
    Tensor gamma;  // scale (features)
    Tensor beta;   // shift (features)
    double eps;
    Tensor last_mean;
    Tensor last_var;
    Tensor last_x;
    bool training;

    LayerNorm(size_t features, double eps = 1e-5);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return gamma; } // placeholder
    Tensor grad_gamma_;
    Tensor grad_beta_;
    Tensor grad_x;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training = t; }
};

// Dropout layer
class Dropout : public Layer {
public:
    double p;
    bool training;
    Tensor mask;
    explicit Dropout(double p = 0.5) : p(p), training(true) {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /* learning_rate */) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    void set_training(bool t) { training = t; }
};

#endif