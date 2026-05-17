#ifndef DEEP_GCN_H
#define DEEP_GCN_H

#include "../../core/layer.h"
#include "../normalization/batch_norm.h"
#include "../utility/spatial_dropout.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// DeepGCN — Li et al., "DeepGCN: Can GCNs Go As Deep As CNNs?" KDD 2020
// ============================================================================
//
// DeepGCNBlock: One residual GCN block with pre-activation design.
//   pre-activation order: BN → ReLU → GCN → (+ residual) → Dropout
//   Residual: h^{(k)} connected to h^{(k+2)} (hop-wise residual from 2 steps back)
//
// DeepGCNStack: Stack of DeepGCNBlocks with learned alpha per block.

class DeepGCNBlock : public Layer {
public:
    DeepGCNBlock(size_t in_features, size_t out_features,
                 bool use_bn = true, bool use_dropout = true, double dropout_p = 0.1,
                 bool use_attention = false);
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    void set_training(bool t);

    // Public for DeepGCNStack to set residual connection
    Tensor last_x_prev_;

private:
    Dense W_gcn_;
    bool use_bn_;
    BatchNorm1D bn_;
    BatchNorm1D bn_res_;
    bool use_dropout_;
    double dropout_p_;
    std::unique_ptr<Dropout1D> dropout_;
    bool use_attention_;
    // Attention weights (optional, unused for now)
    Dense attn_W_;
    Dense attn_a_;
    // Residual scaling (learnable per block)
    Tensor alpha_;
    // Cached for backward
    Tensor last_x_;
    Tensor last_x_bn_;
    Tensor last_x_act_;
    Tensor last_x_gcn_;
    Tensor last_x_out_;
    Tensor adj_norm_;
    bool training_;
};

class DeepGCNStack : public Layer {
public:
    DeepGCNStack(const std::vector<size_t>& hidden_dims,
                 bool use_bn = true, bool use_dropout = true, double dropout_p = 0.1,
                 bool use_attention = false);
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    void set_training(bool t);

private:
    std::vector<DeepGCNBlock> blocks_;
    Tensor last_output_;
    Tensor last_input_;
    Tensor last_adj_;
};

// ============================================================================
// GCNII — Chen et al., "Simple and Deep GCNII" ICML 2020
// ============================================================================
//
// GCNIILayer: Single GCNII layer.
//   Formula:
//     h^{(k+1)} = Dropout( ReLU( P @ h^{(k)} @ W^{(k)} + I @ h^{(k)} @ W^{(k)}_0 ) ) + h^{(k-1)}
//
//   Where:
//     P = A @ D^{-1}  (renormalized adjacency without self-loops)
//     W^{(k)}  = regular weight matrix (dim → dim)
//     W^{(k)}_0 = initial residual weight (shared across layers, same dim → dim)
//     α^{(k)}  = 2 / (1 + k)  (decreasing coefficient)
//
// GCNIIModel: Stack of GCNIILayers with shared W0.

class GCNIILayer : public Layer {
public:
    GCNIILayer(size_t in_features, size_t out_features);
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
    Dense W_;       // weight matrix for graph convolution path
    Dense W0_;      // initial residual weight (shared)
    Tensor adj_;    // stored adjacency
    size_t num_nodes_;
    size_t out_features_;
    Tensor last_input_;
    Tensor last_Pxh_;
    Tensor last_Ixh0_;
    Tensor last_preact_;
    double alpha_;
};

class GCNIIModel : public Layer {
public:
    GCNIIModel(size_t in_features, size_t hidden_features,
               size_t num_layers, double dropout_p = 0.1,
               bool use_bn = true, double beta = 1.0);
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    void set_training(bool t);

private:
    std::vector<GCNIILayer> layers_;
    std::vector<BatchNorm1D> bns_;
    std::vector<std::unique_ptr<Dropout1D>> drops_;
    std::vector<double> alphas_;
    Dense W0_shared_;
    size_t num_layers_;
    double dropout_p_;
    bool use_bn_;
    double beta_;
    Tensor last_output_;
    Tensor last_input_;
    Tensor last_adj_;
};

#endif