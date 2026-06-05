#ifndef DMON_H
#define DMON_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// DMon (Diffusion Module Network)
// ----------------------------------------------------------------------------
// Di Giovanni, Rowbottom, Glastonbury, et al. "DMon: Diffusion-inspired
// graph neural networks" (2022), and earlier diffusion-based GNNs from
// Atwood & Towsley "Diffusion-Convolutional Neural Networks" (NeurIPS 2016)
// and Klicpera et al. "Predict then Propagate" (ICLR 2019).
//
// Key idea: instead of single-step neighborhood aggregation (GCN), apply
// the GRAPH DIFFUSION OPERATOR based on the heat kernel at MULTIPLE
// diffusion time scales. Each layer corresponds to a different time scale
// tau_k. The heat kernel is computed as a power series of the (normalized)
// random-walk transition matrix:
//
//   H^{(k)} = exp(tau_k * (A_norm - I))  *  X
//
// where A_norm is the symmetrically normalized adjacency with self-loops
// D^{-1/2} (A + I) D^{-1/2}, and the exponential is the matrix exponential.
//
// In practice, the heat kernel is approximated via a truncated Taylor
// expansion (K steps):
//
//   exp(tau * (A_norm - I)) = sum_{r=0..R} (tau^r / r!) * (A_norm - I)^r
//
// For each diffusion scale tau_k, the layer output is H^{(k)} @ W_k.
// The per-layer outputs are concatenated along the feature axis and
// fed to a final Dense to produce the layer output. This is the DMon
// design (Di Giovanni et al. 2022): multi-scale diffusion features
// preserve both local (small tau) and global (large tau) information
// and have been shown to be at least as expressive as the 1-WL test on
// regular graphs.
//
// Forward:
//
//   For each scale tau in [tau_0, ..., tau_{K-1}]:
//     T_k = heat_kernel(A_norm, tau_k, R)        // (N, N)
//     Z_k = T_k @ X                              // (N, in_features)
//     Z   = concat([Z_0, Z_1, ..., Z_{K-1}])     // (N, K * in_features)
//   Output = W_out @ Z + b_out                   // (N, out_features)
//
// Backward:
//   grad_X gets contributions from each scale (T_k @ grad_Z_k).
//   grad_T_k = grad_Z_k @ X^T (matrix form).
//   grad_tau_k comes from the Taylor series derivative.
//   grad_W_out, grad_b_out from the final dense.
//
// We freeze the Taylor expansion R during gradient computation (standard
// practice for DMon). The adjacency is also treated as a constant (no
// edge feature learning, like PATCHY-SAN, EdgeConv, etc.).
// ============================================================================

class DMonLayer : public Layer {
public:
    // in_features: per-node input feature dim
    // out_features: per-node output feature dim
    // num_scales: number of diffusion time scales K
    // taylor_terms: truncation order R for the heat-kernel approximation
    // initial_scales: if non-empty, use these as tau values; otherwise
    //                 use a geometric schedule tau_k = 1.5^k (default).
    //                 The geometric schedule is the recommendation from
    //                 the DMon paper for undirected graphs.
    DMonLayer(size_t in_features, size_t out_features,
              size_t num_scales = 3, size_t taylor_terms = 4,
              const std::vector<double>& initial_scales = {});

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "DMonLayer"; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t num_scales_;
    size_t taylor_terms_;
    std::vector<double> scales_;     // length K
    std::vector<double> grad_scales_;// length K

    // Output Dense: (K * in_features) -> out_features
    Dense output_dense_;

    // Cached state for backward
    Tensor last_input_;            // (N, in_features)
    Tensor adj_;                   // (N, N) — original adjacency (no self-loops)
    Tensor adj_norm_;              // (N, N) — normalized with self-loops
    std::vector<Tensor> T_k_;      // (N, N) per scale — heat kernel
    std::vector<Tensor> Z_k_;      // (N, in_features) per scale — T_k @ X
    Tensor last_concat_;           // (N, K * in_features)
    Tensor last_output_;           // (N, out_features)
    size_t num_nodes_;

    // Helper: build the symmetric normalized adjacency with self-loops.
    void normalize_adjacency(const Tensor& adj);

    // Helper: compute the heat-kernel heat(tau) = exp(tau * (A_norm - I))
    // via truncated Taylor series with `taylor_terms_` terms.
    Tensor compute_heat_kernel(double tau) const;
};

// ============================================================================
// DMonModel: input projection + DMon stack + classifier
// ============================================================================
class DMonModel : public Layer {
public:
    DMonModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
              size_t out_features, size_t num_layers = 2,
              size_t num_scales = 3, size_t taylor_terms = 4,
              const std::vector<double>& initial_scales = {});

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "DMonModel"; }

private:
    size_t num_nodes_;
    size_t in_features_;
    size_t hidden_dim_;
    size_t out_features_;
    size_t num_layers_;
    size_t num_scales_;
    size_t taylor_terms_;
    std::vector<double> initial_scales_;

    Dense input_proj_;
    std::vector<DMonLayer> dmon_layers_;
    Dense classifier_;

    // Cached for backward
    Tensor last_input_;
    Tensor last_input_proj_;
    // Pre-ReLU DMon layer outputs (after DMon, before ReLU). Used as
    // the ReLU mask in backward.
    std::vector<Tensor> layer_outputs_pre_relu_;
    Tensor adj_;
};

#endif
