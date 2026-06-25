#ifndef GRAPHSAGE_H
#define GRAPHSAGE_H

#include "../../core/layer.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>

// =============================================================================
// GraphSAGE — Hamilton, Ying, Leskovec 2017
//   "Inductive Representation Learning on Large Graphs"
//   https://arxiv.org/abs/1706.02216
//
// Core update rule per node i:
//   h_{N(i)} = AGGREGATE({ h_u : u in N(i) })           // neighbour embedding
//   h_i'     = sigma( W · CONCAT(h_i, h_{N(i)}) + b )    // dense on [self || agg]
//
// Three aggregators are implemented (matching the original paper):
//   - "mean"  :  h_{N(i)} = mean( { h_u : u in N(i) } )
//   - "pool"  :  h_{N(i)} = max( { sigma(W_pool · h_u + b_pool) : u in N(i) } )
//                 (element-wise max over neighbours, applied to a Dense-transformed copy)
//   - "max"   :  h_{N(i)} = max( { h_u : u in N(i) } )    (element-wise max, no transform)
//
// Two design choices that match the reference DGL/PyG implementations:
//   1. CONCAT(h_i, h_{N(i)}) is the canonical GraphSAGE formulation.
//      (GCN-style "add self after" is sometimes used as an alternative —
//      we keep the original SAGE form here because the paper specifically
//      shows that concatenation outperforms addition.)
//   2. Optional L2-normalization of h_i' before returning (the paper
//      recommends it for stable training when stacking many layers; default ON).
//
// Neighbour sampling:
//   When `num_samples_per_layer[k]` is set on the k-th layer, exactly that
//   many neighbours are uniformly sampled per node at forward time. The
//   backward pass uses the *same* sample indices that were used in the
//   matching forward (cached), so the gradient is consistent with what
//   the forward actually computed. With num_samples=0 (default) all
//   neighbours are used.
//
// Backward pass (full BPTT through all components):
//   (1) L2-normalization backward (if used): the classic
//           d/dx (x / ||x||_2) = (I - x x^T / ||x||^2) / ||x||
//       chain, applied per row.
//   (2) Dense W/b backward through CONCAT(h_i, h_{N(i)}).
//   (3) Aggregator backward:
//        - MEAN:    d h_u receives d h_{N(i)} / count for each neighbour
//                   (plus cached count_neighbours per node).
//        - POOL:    element-wise max — gradient flows to the neighbour that
//                   produced the maximum value at each output feature.
//        - MAX:     same as POOL but on the un-transformed neighbour features.
//        - POOL additionally carries the Dense W_pool/b_pool gradients through
//          the pre-pool-transform.
//   (4) Concat split: h_i receives the slice of d_concat that comes from
//       the self branch, neighbours receive the slice from the aggregation
//       branch.
//
// All gradients accumulated via Dense::grad_weights / Dense::grad_bias
// following the existing repo convention (Dense: y = x @ W^T + b,
// W has shape (out, in)).
// =============================================================================

class GraphSAGELayer : public Layer {
public:
    // aggregator: one of {"mean", "pool", "max"}
    // normalize:  apply L2 normalization to h_i' before returning (default true)
    // self_loop:  add self to the neighbour set (default true — paper convention)
    GraphSAGELayer(size_t in_features, size_t out_features,
                   const std::string& aggregator = "mean",
                   bool normalize = true,
                   bool self_loop = true);
    ~GraphSAGELayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    // Sample a fixed number of neighbours per node for scalability.
    // Call BEFORE forward_with_adj to apply to the next forward.
    // num_samples = 0 means use all neighbours.
    void set_num_samples(size_t num_samples) { num_samples_ = num_samples; }
    size_t get_num_samples() const { return num_samples_; }

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "GraphSAGELayer"; }
    const std::string& aggregator() const { return aggregator_; }

private:
    // Hyperparameters
    size_t in_features_;
    size_t out_features_;
    std::string aggregator_;
    bool normalize_;
    bool self_loop_;
    size_t num_samples_ = 0;  // 0 = use all neighbours

    // Dense: CONCAT(h_i, h_{N(i)}) (size 2 * in_features) -> out_features
    Dense W_;

    // For "pool" aggregator: an internal Dense (in_features -> in_features)
    // applied to each neighbour before the element-wise max. Lazily
    // constructed only when needed (Dense has no default constructor).
    std::unique_ptr<Dense> W_pool_;
    bool has_pool_ = false;

    // Cached for backward
    Tensor last_input_;          // (N, in_features)
    Tensor adj_;                 // (N, N) adjacency
    Tensor last_concat_;         // (N, 2 * in_features)
    Tensor last_pre_act_;        // (N, out_features)  — pre-activation
    Tensor last_output_;         // (N, out_features)  — post-activation
    Tensor last_norm_out_;       // (N, out_features)  — post-L2-normalization (== last_output_ when normalize_=false)
    std::vector<std::vector<size_t>> last_neighbors_;  // per-node sampled neighbour indices
    std::vector<std::vector<size_t>> last_argmax_;     // for pool/max: which neighbour won per (node, feature)
    bool has_cache_ = false;
};

// Full GraphSAGE model: stack of GraphSAGELayers + per-node classifier.
class GraphSAGEModel : public Layer {
public:
    // in_features: per-node input feature dim
    // hidden_dim:  hidden dim for intermediate GraphSAGE layers
    // out_features: per-node classifier output dim
    // num_layers:  number of GraphSAGE layers (default 2)
    // aggregator:  passed to each GraphSAGELayer
    // normalize:   passed to each GraphSAGELayer
    GraphSAGEModel(size_t in_features, size_t hidden_dim, size_t out_features,
                   size_t num_layers = 2,
                   const std::string& aggregator = "mean",
                   bool normalize = true);
    ~GraphSAGEModel() override;

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "GraphSAGEModel"; }

private:
    std::vector<GraphSAGELayer*> layers_;
    Dense input_proj_;     // (in_features, hidden_dim)  — only used if in_features != hidden_dim on layer 0
    Dense classifier_;     // (hidden_dim, out_features)
    bool use_proj_;
};

#endif