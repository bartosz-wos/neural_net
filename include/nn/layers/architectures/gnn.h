#ifndef GNN_H
#define GNN_H

#include "../../core/layer.h"
#include <vector>

// Graph Neural Network layers.
// GCNLayer: Graph Convolutional Network (Kipf & Welling 2017).
// GATLayer: Graph Attention Network (Veličković et al. 2018) with multi-head attention.
class GCNLayer : public Layer {
public:
    GCNLayer(size_t in_features, size_t out_features);
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
    Tensor last_output_;
    Tensor last_input_;
    Tensor last_AW_;
    Tensor adj_norm_;
    std::vector<std::vector<double>> relu_mask_;
};

// Helper: per-head attention parameters.
// W: (head_dim, in_features) — projects input.
// a: (head_dim*2, 1) — attention vector; e_ij = a^T [Wh_i || Wh_j].
// grad_W, grad_a: same shapes as W, a.
struct GATHeadParams {
    Tensor W;       // (head_dim, in_features)
    Tensor a;       // (head_dim*2, 1) — flat-indexed as a[k][0] for k in [0, 2*head_dim)
    Tensor grad_W;
    Tensor grad_a;
};

class GATLayer : public Layer {
public:
    GATLayer(size_t in_features, size_t out_features, size_t num_heads = 4,
             bool concat_heads = true);
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
    size_t num_heads_;
    bool concat_heads_;
    size_t in_features_;
    size_t out_features_;
    size_t head_dim_;
    std::vector<GATHeadParams> heads_;

    // Cached state for backward
    Tensor last_output_;
    Tensor last_input_;
    std::vector<Tensor> last_Wh_heads_;     // (N, head_dim) per head — pre-softmax Wh
    Tensor last_alpha_;                     // (N, N * num_heads_) post-softmax attention
    Tensor last_e_;                         // (N, N * num_heads_) pre-softmax LeakyReLU scores
    Tensor last_head_pre_;                  // (N, head_dim * num_heads_) pre-LeakyReLU weighted sums
    Tensor adj_;                            // (N, N) stored adjacency
};

class GraphNetwork : public Layer {
public:
    GraphNetwork(const std::vector<size_t>& hidden_dims, bool use_gat = false);
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
    std::vector<GCNLayer> gcn_layers_;
    std::vector<GATLayer> gat_layers_;
    bool use_gat_;
    Tensor last_output_;
    Tensor last_input_;
    Tensor last_adj_;
};

#endif
