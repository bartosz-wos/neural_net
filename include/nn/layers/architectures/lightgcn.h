#ifndef LIGHTGCN_H
#define LIGHTGCN_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// LightGCN: Linear Graph Convolutional Network
// He et al., SIGIR 2020
//
// Key differences from GCN:
// - No activation functions (pure linear)
// - No weight transformation (no W matrix between layers)
// - Layer combination (readout): H_final = sum_{k=0}^{K} alpha_k * H^{(k)}
//
// Propagation: H^{(k+1)} = D^{-1/2} (A + I) D^{-1/2} @ H^{(k)}
// (self-loops added internally)

class LightGCNLayer : public Layer {
public:
    // num_layers: number of propagation steps K (outputs H^{(0)}...H^{(K)})
    // combination: if true, learn combination weights; if false, uniform (1/(K+1))
    LightGCNLayer(size_t num_layers, bool learnable_combination = false);
    
    // forward: standalone single-layer (rarely used directly)
    Tensor forward(const Tensor& input) override;
    
    // forward_with_adj: full propagation with adjacency
    // input: (num_nodes, embedding_dim) initial embeddings or features
    // adj: (num_nodes, num_nodes) adjacency matrix (self-loops added internally)
    // Returns: (num_nodes, embedding_dim) combined embedding
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "LightGCNLayer"; }

private:
    size_t num_layers_;
    bool learnable_combination_;
    Tensor alpha_;  // combination weights (K+1 values), softmax normalized
    
    // Cached for backward
    std::vector<Tensor> layer_outputs_;  // H^{(0)}, H^{(1)}, ..., H^{(K)}
    Tensor adj_norm_;  // D^{-1/2} (A+I) D^{-1/2}
    
    void normalize_adjacency(const Tensor& adj);
    Tensor compute_output() const;
};

// Full LightGCN model: embedding table + LightGCNLayer
class LightGCNModel : public Layer {
public:
    LightGCNModel(size_t num_nodes, size_t in_features, size_t embedding_dim,
                  size_t num_layers, bool learnable_combination = false);
    
    Tensor forward(const Tensor& input) override;
    
    // input: (batch_size, ...) node indices or features
    // adj: (num_nodes, num_nodes) adjacency matrix
    // If input is (batch_size,) integer indices: lookup embeddings
    // If input is (num_nodes, in_features): use as feature input
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embedding_; }
    Tensor get_gradients() const override { return grad_embedding_; }
    std::string name() const override { return "LightGCNModel"; }

private:
    size_t num_nodes_;
    size_t in_features_;
    size_t embedding_dim_;
    size_t num_layers_;
    bool learnable_combination_;
    bool has_learned_embeddings_;  // true if using learned embeddings, false if using raw features
    
    Tensor embedding_;      // (num_nodes, embedding_dim) learnable embeddings
    Tensor grad_embedding_; // gradient for embeddings
    LightGCNLayer gcn_layer_;
    
    // Cached for backward
    Tensor last_input_;      // original input (indices or features)
    Tensor last_emb_input_;  // embedded input to gcn_layer_
    Tensor last_output_;     // final output
    Tensor adj_;             // stored adjacency
};

#endif