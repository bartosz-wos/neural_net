#ifndef EDGECONV_H
#define EDGECONV_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// EdgeConv / Dynamic Graph CNN (DGCNN)
// Wang, Su, Bronstein, Jegelka. "Dynamic Graph CNN for Learning on Point Clouds".
// ACM Transactions on Graphics (TOG) 2019.
//
// Per layer, for each node i:
//   1. Build a k-nearest-neighbour graph in feature space (Euclidean).
//      The graph is constructed from the current node features; in the
//      standard simplified implementation (used here, matching common
//      reference code) the graph is treated as fixed during the backward
//      pass, so the gradient does not back-propagate through the k-NN.
//   2. For each neighbour j in N(i), compute an edge feature
//         e_{ij} = h_j - h_i            (asymmetric edge function, default)
//   3. Concatenate h_i with e_{ij} and pass through an MLP:
//         msg_{ij} = MLP([h_i || e_{ij}])
//   4. Aggregate:  h_i' = MAX_{j in N(i)} msg_{ij}  (max over neighbours,
//      element-wise across the output feature axis).
//   5. Apply the post-aggregation transform (Dense + optional activation
//      handled outside the layer by callers).
//
// A simpler variant (sometimes called "EdgeConv-vanilla") instead uses
//     e_{ij} = h_j
// and a corresponding MLP. The asymmetric form is the more common one
// in the DGCNN paper, so we adopt it.
//
// The MLP is a 2-layer Dense stack: in -> 2*out -> out. Output is (N, out).
//
// Gradient computation:
//   - The k-NN graph is fixed during the backward pass (no gradient
//     flows through "which neighbour won the k-NN comparison"). This
//     is the standard DGCNN reference behaviour.
//   - For the max aggregator, the gradient is routed to the neighbour
//     that produced the maximum value at each (node, output feature)
//     position. We cache argmax indices per output feature.
//   - For the edge asymmetry h_j - h_i, gradient is split between the
//     source node (h_j) and the target node (h_i). When j == i (a
//     self-loop, which we always include) the contribution is zero.
//   - The MLP backward is a standard Dense chain.
//
// References:
//   Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019.
//   https://arxiv.org/abs/1801.07829

class EdgeConvLayer : public Layer {
public:
    // in_features: per-node input feature dimension
    // out_features: per-node output feature dimension
    // k: number of nearest neighbours (excluding self, unless self_loops)
    // self_loops: include the node itself in its own neighbourhood
    //             (paper convention: usually include self)
    EdgeConvLayer(size_t in_features, size_t out_features, size_t k = 20,
                  bool self_loops = true);
    ~EdgeConvLayer() override = default;

    // EdgeConv requires the adjacency structure but the layer itself
    // builds a k-NN graph from features. We still accept an `adj`
    // parameter for API compatibility; it is currently unused (the
    // k-NN graph overrides it). If you want to constrain the k-NN
    // search to a pre-existing graph, use a k equal to or smaller
    // than the in-degree of each node in `adj`.
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "EdgeConvLayer"; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t k_;
    bool self_loops_;

    // MLP: 2-layer Dense. Input is concat(h_i, h_j - h_i) of size 2*in_features.
    // Hidden dim: 2 * out_features (paper convention).
    Dense mlp_fc1_;  // (2*in_features) -> (2*out_features)
    Dense mlp_fc2_;  // (2*out_features) -> out_features

    // Cached state for backward
    Tensor last_input_;       // (N, in_features) — clone of input
    size_t num_nodes_;

    // Per-node list of k neighbour indices (including self if self_loops_).
    // shape: last_neighbors_[i] is a vector of length up to k (some
    // nodes may have fewer available neighbours if the graph is small).
    std::vector<std::vector<size_t>> last_neighbors_;

    // For the max aggregator: last_argmax_[i] is a vector of length
    // out_features; the value at position f is the index of the
    // neighbour (into last_neighbors_[i]) that produced the maximum
    // value at output feature f.
    std::vector<std::vector<size_t>> last_argmax_;

    // Cached messages and concat input to MLP, useful for debugging
    // (not strictly required for backward since Dense caches its own
    // last_input, but kept for clarity).
    Tensor last_concat_msgs_;  // (N * num_msgs_per_node, 2*in_features)
};

// Full EdgeConv / DGCNN-style model: optional input projection + a stack
// of EdgeConv layers (each followed by ReLU) + a global pool + classifier.
class EdgeConvModel : public Layer {
public:
    // in_features: per-node input feature dimension
    // hidden_dim: hidden dimension for EdgeConv layers
    // out_features: per-node output dimension (used as classifier input dim)
    // num_classes: classifier output dimension (if 0, no classifier)
    // num_layers: number of EdgeConv layers to stack
    // k: number of nearest neighbours per node
    EdgeConvModel(size_t in_features, size_t hidden_dim, size_t out_features,
                  size_t num_classes = 0, size_t num_layers = 2, size_t k = 20);
    ~EdgeConvModel() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "EdgeConvModel"; }

private:
    size_t in_features_;
    size_t hidden_dim_;
    size_t out_features_;
    size_t num_classes_;
    size_t num_layers_;

    // Optional input projection (identity if in_features == hidden_dim)
    Dense input_proj_;

    // Stack of EdgeConv layers
    std::vector<EdgeConvLayer> edge_layers_;

    // Pre-classifier projection: hidden_dim -> out_features
    Dense head_proj_;

    // Output classifier (Dense: out_features -> num_classes)
    Dense classifier_;

    // Cached state for backward
    Tensor last_input_;
    Tensor adj_;
    // pre-ReLU outputs of each layer, used to recover the ReLU mask
    std::vector<Tensor> layer_pre_relu_;
    // Pre-head-projection cached value (post-final-ReLU)
    Tensor last_head_input_;
};

#endif
