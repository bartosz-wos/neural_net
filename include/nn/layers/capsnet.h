#ifndef CAPSNET_H
#define CAPSNET_H

#include "../core/layer.h"

// Capsule layer output: a tensor where each column is a capsule's output vector
// Each capsule outputs a vector (batch, num_capsules, dim_capsule)

// Dynamic routing by agreement between capsules.
// Iteratively refines coupling coefficients c_ij based on agreement a_ij = u_j · v_j.
class CapsuleLayer : public Layer {
public:
    // input_dim: total input capsule dimension (prod of spatial dims)
    // num_capsules: number of output capsules
    // dim_capsule: dimensionality of each output capsule vector
    // num_routing: number of routing iterations (typically 3)
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
    std::vector<Tensor> W_; // transformation matrices per input capsule
    Tensor last_v_; // output capsule vectors (for backward)
    Tensor last_b_;  // biases
    std::vector<double> routing_activations_;
};

// Full CapsNet: encoder (Conv + PrimaryCaps + Capsule) + reconstruction decoder
class CapsNet : public Layer {
public:
    // num_classes: number of digit capsules, dim_capsule: capsule vector dim
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

    // Reconstruction loss from capsule outputs and correct digit labels
    double reconstruction_loss(const Tensor& input,
                               const Tensor& digit_capsules,
                               size_t correct_label);

private:
    Conv2D conv1_; // initial conv
    Dense primary_caps_fc_; // primary capsule fully connected (flattened conv output → capsules)
    CapsuleLayer digit_caps_;
    // Decoder: three FC layers to reconstruct image
    Dense fc1_, fc2_, fc3_;
    Tensor last_input_;
    Tensor last_capsule_output_;
};

#endif