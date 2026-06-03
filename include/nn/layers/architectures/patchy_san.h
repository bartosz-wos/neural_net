#ifndef PATCHY_SAN_H
#define PATCHY_SAN_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <algorithm>

// PATCHY-SAN: Graph CNN via learned local "patches"
// Niepert, Ahmed, Kutzkov. "Learning Convolutional Neural Networks for Graphs".
// ICML 2016.
//
// Procedure (per node, then aggregated):
//   1. Node Selection: use all nodes as anchors (canonical ordering: by node id).
//   2. Neighborhood Assembly: BFS expansion of width `k` (number of hops),
//      collecting up to `w` neighbors (or `w` + the anchor itself).
//   3. Graph Normalization: sort neighborhood by (label, id) where label is
//      a WL-1-style coloring (degree + neighbors' degrees). This produces a
//      fixed-length sequence.
//   4. Convolutional Step: apply a learned per-position Dense layer over the
//      sequence; sum over positions to produce a single patch embedding.
//
// Final representation: a Dense layer maps the patch embedding to `out_features`,
// giving one (out_features) vector per anchor node.

class PatchySANLayer : public Layer {
public:
    // in_features: per-node input feature dimension
    // out_features: per-node output feature dimension
    // w: width of receptive field (max sequence length, including anchor)
    // k: number of BFS hops (neighborhood assembly depth)
    PatchySANLayer(size_t in_features, size_t out_features,
                   size_t w = 4, size_t k = 2);

    Tensor forward(const Tensor& input) override;

    // input: (num_nodes, in_features)
    // adj:   (num_nodes, num_nodes) binary adjacency
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "PatchySANLayer"; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t w_;            // receptive-field width
    size_t k_;            // BFS depth

    // w Dense layers, one per position in the sequence.
    std::vector<Dense> conv_layers_;   // each: in_features -> out_features

    // Final dense producing the per-node output (with bias).
    Dense output_dense_;               // out_features -> out_features

    // Cached state for backward
    Tensor last_input_;                // (N, in_features)
    Tensor adj_;                       // (N, N)
    std::vector<std::vector<size_t>> last_patches_;   // patches_[i] = ordered neighbor ids
    Tensor last_patch_sum_;            // (N, out_features) pre-output_dense features

    // Helpers
    // Build the canonical sequence (of length <= w_) starting at anchor.
    // Sequence is the anchor followed by BFS-k neighbors, sorted by (label, id).
    std::vector<size_t> build_patch(size_t anchor, const Tensor& adj) const;
    // Labeling: a simple 1-WL-style hash based on (degree, sorted neighbor ids).
    std::vector<uint64_t> compute_labels(const Tensor& adj) const;
};

// Full PATCHY-SAN model: optional input projection + PatchySANLayer + classifier
class PatchySANModel : public Layer {
public:
    PatchySANModel(size_t in_features, size_t hidden_dim, size_t out_features,
                   size_t w = 4, size_t k = 2);

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "PatchySANModel"; }

private:
    size_t in_features_;
    size_t hidden_dim_;
    size_t out_features_;
    size_t w_;
    size_t k_;

    Dense input_proj_;         // in_features -> hidden_dim (acts as in-FC for patches)
    PatchySANLayer patchy_layer_;  // hidden_dim -> hidden_dim
    Dense classifier_;         // hidden_dim -> out_features

    // Cache
    Tensor last_input_;
    Tensor adj_;
    Tensor last_proj_;         // projected node features (N, hidden_dim)
    Tensor last_output_;
};

#endif
