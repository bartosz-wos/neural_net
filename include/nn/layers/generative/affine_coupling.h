#ifndef AFFINE_COUPLING_H
#define AFFINE_COUPLING_H

#include "coupling_layer.h"
#include <vector>

// AffineCoupling — RealNVP stack with alternating coupling layers
// Implements a sequence of alternating coupling layers that together
// form a full invertible normalizing flow.
//
// For depth coupling layers alternating split_dim between rows and cols:
//   - Even layers: split on rows (mask x1 = top half)
//   - Odd layers: split on cols (mask x1 = left half)
// This ensures every dimension participates in the transformation.
class AffineCoupling : public Layer {
public:
    // input_dim: dimensionality of input (rows * cols for flattened)
    // depth: number of coupling layers
    // hidden_size: hidden dimension for s/t MLPs
    // For simplicity, we use rows=1 split_dim=1 (col-wise split for batch=1 tensors)
    // depth: number of coupling layers in the stack
    AffineCoupling(size_t input_dim, size_t depth, size_t hidden_size);

    // Forward pass: transform base distribution to target
    // Returns transformed tensor and accumulated log_det_jacobian
    Tensor forward(const Tensor& input) override;

    // Inverse pass: sample from target distribution (generation)
    Tensor inverse(const Tensor& y);

    // Log determinant of the full flow
    double log_det_jacobian() const { return log_det_jacobian_; }

    // Backward pass
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    // Sample from the flow (inverse pass starting from Gaussian noise)
    Tensor sample(size_t num_samples);

private:
    size_t input_dim_;
    size_t depth_;
    std::vector<CouplingLayer*> layers_;  // owned coupling layers
    double log_det_jacobian_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};

#endif