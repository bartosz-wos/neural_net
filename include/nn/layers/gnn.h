#ifndef GNN_H
#define GNN_H

#include "../core/layer.h"
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
    std::vector<Dense> W_heads_;
    std::vector<Dense> a_heads_;
    Tensor last_output_;
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
};

#endif