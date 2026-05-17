#ifndef GIN_H
#define GIN_H

#include "../../core/layer.h"
#include "../normalization/batch_norm.h"
#include <vector>
#include <cmath>

// Graph Isomorphism Network (GIN) layer
// Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019
//
// GIN update: h_{k+1} = MLP( (1 + eps_k) * h_k + sum_{j in N(i)} h_j )
// - eps_k is a learnable per-layer scalar (initialized to 0)
// - MLP is a 2-layer MLP: Linear -> BatchNorm -> ReLU -> Linear
class GINLayer : public Layer {
public:
    GINLayer(size_t in_features, size_t out_features, size_t hidden_dim = 32,
             size_t num_mlp_layers = 2);
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    // MLP layers
    std::vector<Dense> fc_layers_;
    std::vector<BatchNorm1D> bn_layers_;
    // Learnable epsilon: we store (1+eps) directly as a scalar
    double one_plus_eps_;
    // Cached for backward
    Tensor last_input_;
    Tensor last_agg_;       // aggregated: (1+eps)*h_i + sum_neighbor (before MLP)
    std::vector<Tensor> pre_bn_outputs_;    // fc output before BN for each hidden layer
    Tensor adj_;            // stored adjacency for backward
    size_t num_nodes_;
    size_t in_features_;
};

// GIN0Layer: simplified GIN with eps=0, single linear layer (no MLP)
// h'_i = W @ (h_i + sum_{j in N(i)} h_j)
class GIN0Layer : public Layer {
public:
    GIN0Layer(size_t in_features, size_t out_features);
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Dense W_;
    Tensor last_input_;     // input features (h_i)
    Tensor last_agg_;        // output after W transform
    Tensor input_plus_agg_; // (h_i + sum_neighbor) before W, for backward
    Tensor adj_;            // stored adjacency for backward
    size_t num_nodes_;
    size_t out_features_;
};

#endif