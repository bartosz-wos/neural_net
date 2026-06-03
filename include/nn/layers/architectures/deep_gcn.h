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
    Dense attn_W_;
    Dense attn_a_;
    Tensor alpha_;
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
// GCNIILayer: Single layer implementing:
//   h^{(l+1)} = ReLU( ((1-alpha)*A_norm@h^{(l)} + alpha*h^{(0)})@W + beta*h^{(l)}@V )
//
// - A_norm = D^{-1/2} A D^{-1/2} (symmetric normalized adjacency)
// - h^{(0)} = initial residual (set by model)
// - alpha = transition factor (0 < alpha <= 1)
// - beta  = identity scaling factor (0 <= beta <= 1)
// - W     = weight for the graph convolution path
// - V     = weight for the identity mapping path

class GCNIILayer : public Layer {
public:
    GCNIILayer(size_t in_features, size_t out_features,
               double alpha = 0.1, double beta = 0.5, double dropout_p = 0.0);
    ~GCNIILayer() override = default;

    Tensor forward(const Tensor& input) override;
    // Uses pre-computed D^{-1/2} A D^{-1/2} (adj already symmetric normalized)
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj_norm);
    // Takes raw adjacency, computes normalization internally
    Tensor forward_with_sparse(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_.weights; }
    Tensor get_gradients() const override { return W_.grad_weights; }
    std::string name() const override { return "GCNIILayer"; }

    void set_h0(const Tensor& h0) { h0_ = h0; h0_set_ = true; }
    void set_training(bool t);

private:
    Dense W_;               // graph convolution path: (in_features x out_features)
    Dense V_;               // identity mapping path: (in_features x out_features)
    double alpha_;          // transition factor
    double beta_;           // identity scaling factor
    std::unique_ptr<Dropout1D> dropout_;

    // Cached for backward
    Tensor h0_;             // initial residual (set by model)
    Tensor last_input_;     // input to this layer (h^{(l)})
    Tensor last_adj_norm_;  // stored normalized adjacency
    Tensor last_Axh_;       // (1-alpha)*A_norm@h + alpha*h0 before W
    Tensor last_preact_;    // after W transform, before ReLU
    Tensor last_output_;    // final output h^{(l+1)}
    bool h0_set_;
    bool training_;
    size_t num_nodes_;
    size_t in_features_;
    size_t out_features_;

    // Compute D^{-1/2} A D^{-1/2} from raw adjacency
    Tensor normalize_adjacency(const Tensor& adj) const;
};

class DeepGCNModel : public Layer {
public:
    DeepGCNModel(size_t in_features, size_t hidden_features,
                size_t out_features, size_t num_layers,
                double alpha = 0.1, double beta = 0.5, double dropout_p = 0.0,
                bool use_bn = true);

    Tensor forward(const Tensor& input) override;
    // adj: raw (num_nodes, num_nodes) adjacency matrix
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "DeepGCNModel"; }
    void set_training(bool t);

private:
    // First layer: projects input to hidden dimension, creates h0
    Dense W0_;  // (in_features x hidden_features)

    // Hidden GCNII layers (num_layers)
    std::vector<std::unique_ptr<GCNIILayer>> layers_;
    std::vector<BatchNorm1D> bns_;       // one per hidden layer
    std::vector<std::unique_ptr<Dropout1D>> drops_;  // one per hidden layer

    // Final projection to out_features
    Dense W_out_;  // (hidden_features x out_features)

    std::vector<double> alphas_;  // alpha_k = 2/(1+k) per layer
    size_t num_layers_;
    double dropout_p_;
    bool use_bn_;
    size_t in_features_;
    size_t out_features_;

    // Cached
    Tensor h0_;            // initial residual shared across all GCNII layers
    Tensor last_input_;
    Tensor last_output_;
    Tensor last_adj_;
    bool h0_set_;
    bool training_;
};

// ============================================================================
// GCNIIModel — multi-layer GCNII with optional BN/dropout, NO final projection
// ============================================================================
//
// A "pure" GCNII model: input -> W0 projection -> num_layers x GCNII -> output
// Output dimension equals hidden_features (no W_out projection).
//
// Use case: graph-level representation learning where hidden dim is the final
// node embedding dim. Common for semi-supervised node classification (GCNII
// paper setup) and small-graph tasks.
//
// Constructor: (in_features, hidden_features, num_layers, dropout_p, use_bn, beta)
//   - dropout_p: dropout probability after each GCNII layer
//   - use_bn:    whether to apply BatchNorm1D after each GCNII layer
//   - beta:      identity-mapping scaling (0.0 = no identity path; 1.0 = full)
//   - alpha_k is set per-layer to 2/(1+k) following the GCNII paper.

class GCNIIModel : public Layer {
public:
    GCNIIModel(size_t in_features, size_t hidden_features,
               size_t num_layers, double dropout_p = 0.0,
               bool use_bn = true, double beta = 0.5);

    Tensor forward(const Tensor& input) override;
    // adj: raw (num_nodes, num_nodes) adjacency matrix
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "GCNIIModel"; }
    void set_training(bool t);

private:
    // First layer: projects input to hidden dimension, creates h0
    Dense W0_;  // (in_features x hidden_features)

    // Hidden GCNII layers (num_layers)
    std::vector<std::unique_ptr<GCNIILayer>> layers_;
    std::vector<BatchNorm1D> bns_;             // one per hidden layer (if use_bn)
    std::vector<std::unique_ptr<Dropout1D>> drops_;  // one per hidden layer

    std::vector<double> alphas_;  // alpha_k = 2/(1+k) per layer
    size_t num_layers_;
    double dropout_p_;
    bool use_bn_;
    size_t in_features_;
    size_t hidden_features_;

    // Cached
    Tensor h0_;            // initial residual shared across all GCNII layers
    Tensor last_input_;
    Tensor last_output_;
    Tensor last_adj_;
    bool h0_set_;
    bool training_;
};


#endif