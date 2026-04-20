#ifndef CAPSNET_H
#define CAPSNET_H

#include "../../core/layer.h"

// Capsule layer: each capsule outputs a vector with dynamic routing by agreement.
class CapsuleLayer : public Layer {
public:
    CapsuleLayer(size_t input_dim, size_t num_capsules,
                 size_t dim_capsule, size_t num_routing = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    size_t num_capsules_, dim_capsule_, num_routing_, input_dim_;
    std::vector<Tensor> W_; // transformation matrices per output capsule
    Tensor last_v_;
    Tensor last_b_;
};

// Full CapsNet: primary caps → digit caps (with routing) + reconstruction decoder
class CapsNet : public Layer {
public:
    CapsNet(size_t input_channels, size_t H, size_t W,
            size_t num_classes, size_t dim_capsule = 16,
            size_t primary_dim = 8, size_t primary_channels = 32 * 6 * 6,
            size_t num_routing = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    double reconstruction_loss(const Tensor& input,
                               const Tensor& digit_capsules,
                               size_t correct_label);

private:
    Dense primary_caps_fc_; // primary capsule: flatten→FC→squash
    CapsuleLayer digit_caps_;
    Dense fc1_, fc2_, fc3_; // reconstruction decoder
    Tensor last_input_;
    Tensor last_capsule_output_;
};

#endif