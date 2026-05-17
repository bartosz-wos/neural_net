#ifndef SPATIAL_DROPOUT_H
#define SPATIAL_DROPOUT_H

#include "../../core/layer.h"

// Dropout1D for sequential data — drops entire time steps (feature channels).
// Training: randomly zeros entire channels with probability p.
// Inference: identity (no dropout).
class Dropout1D : public Layer {
public:
    Dropout1D(double p = 0.5) : p_(p), training_(false) {}
    void set_training(bool t) { training_ = t; }
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override {}
    void zero_grad() override {}
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    double p_;
    bool training_;
    std::vector<std::vector<char>> mask_; // char mask per channel
    Tensor last_output_;
    void apply_mask(Tensor& input);
};

// Dropout2D (SpatialDropout2D in train mode) — drops entire feature maps (channels).
// Instead of dropping individual elements, drops the entire channel with probability p.
// For 2D feature maps of shape (batch, channels, height, width) stored as (batch, channels*h*w).
class Dropout2D : public Layer {
public:
    Dropout2D(double p = 0.5) : p_(p), training_(false) {}
    void set_training(bool t) { training_ = t; }
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override {}
    void zero_grad() override {}
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
private:
    double p_;
    bool training_;
    std::vector<char> channel_mask_; // per-channel mask
    size_t n_channels_;
    size_t spatial_size_; // h * w per channel
    Tensor last_output_;
    void build_mask(size_t channels);
};

#endif