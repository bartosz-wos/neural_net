#ifndef EGNN_H
#define EGNN_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Equivariant Graph Neural Network (EGNN)
// Satorras, Hoogeboom, Welling. "E(n) Equivariant Graph Neural Networks".
// ICML 2021, arXiv:2102.09843.
//
// The key idea: standard message-passing GNNs ignore the spatial positions
// of the nodes, even when they are available (molecules, point clouds, ...).
// EGNN adds an explicit coordinate update that is *equivariant* to
// rotations and translations (E(3) group), and *invariant* to reflections
// of the input coordinates, while remaining a fully-learned local message
// passing network.
//
// Per node i (h_i in R^d, x_i in R^coord_dim), per edge (i, j) in E (adjacency):
//
//   Step 1 — Edge MLP (invariant input):
//     m_ij = phi_e( [ h_i || h_j || ||x_i - x_j||^2 + a_ij ] )          (eq. 2)
//
//   Step 2 — Aggregate messages:
//     m_i = sum_{j in N(i)} m_ij                                       (eq. 3, sum)
//
//   Step 3 — Coordinate update (equivariant):
//     x_i' = x_i + sum_{j in N(i)} (x_i - x_j) * phi_x(m_ij)           (eq. 4)
//
//   Step 4 — Node feature update:
//     h_i' = phi_h( [ h_i || m_i ] )                                    (eq. 5)
//
// Why is it equivariant? The coordinate update only ever uses the
// difference (x_i - x_j). Differences are translation-invariant; rotating
// all x's rotates the differences by the same rotation, and the sum of
// rotated differences rotated back equals the rotated x_i'. phi_e and
// phi_h are applied to invariant features only, so they commute with
// rigid motions.
//
// This file provides:
//   - EGNNLayer:     one EGNN message-passing block (h, x, adj [, attr, weights]) -> (h', x').
//   - EGNNModel:     optional input projection + stack of EGNNLayer + classifier.

class EGNNLayer : public Layer {
public:
    EGNNLayer(size_t in_features, size_t hidden_dim, size_t coord_dim = 3,
              size_t n_edge_attrs = 0, const std::string& activation = "relu",
              bool coordinate_output = true);
    ~EGNNLayer() override = default;

    // Stub: empty adjacency (no edges). Not really useful; prefer forward_with_adj.
    Tensor forward(const Tensor& input) override;

    // Full forward with adjacency, optional edge attributes, optional weights.
    //   input_h:  (N, in_features)  node features
    //   input_x:  (N, coord_dim)    node coordinates
    //   adj:      (N, N)            binary/weighted adjacency
    //   edge_attr (optional): (N, N, n_edge_attrs) — if n_edge_attrs_==0,
    //                this is ignored.
    //   weights   (optional): (N, N)  per-edge weights. If rows=0 (default
    //                Tensor), we use adj as the weight.
    //
    // Returns the updated node features (N, hidden_dim). Updated coords
    // are available via get_last_x().
    Tensor forward_with_adj(const Tensor& input_h, const Tensor& input_x,
                            const Tensor& adj,
                            const Tensor& edge_attr = Tensor(),
                            const Tensor& weights = Tensor());

    // h-only backward (used by model training where the loss is on h).
    // Returns d_input_h. The x gradient from d_h_out is ignored here
    // (see backward_coord for the x path).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    // x-only backward. Returns d_input_x. The h gradient from d_x_out is
    // also computed and added to d_input_h (so d_x affects d_h through
    // dist2 -> phi_e -> m -> h). The full d_input_h is returned via
    // the d_input_h_out pointer (if non-null).
    Tensor backward_coord(const Tensor& grad_x, double learning_rate,
                          Tensor* d_input_h_out = nullptr);

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "EGNNLayer"; }

    Tensor get_last_x() const { return last_x_out_; }
    size_t n_edge_attrs() const { return n_edge_attrs_; }
    size_t coord_dim() const { return coord_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }
    size_t in_features() const { return in_features_; }

private:
    size_t in_features_;
    size_t hidden_dim_;
    size_t coord_dim_;
    size_t n_edge_attrs_;
    std::string activation_;
    bool coordinate_output_;

    // Edge MLP phi_e:  in = (h_i, h_j, dist^2, optional edge_attr)
    //                 => hidden_dim
    //                 = (2 * in_features + 1 + n_edge_attrs) -> hidden_dim
    Dense phi_e_;
    // Node MLP phi_h: (h_i, m_i) = 2 * hidden_dim -> hidden_dim
    Dense phi_h_;
    // phi_x: scalar projection of m_ij -> 1 scalar
    Dense phi_x_;

    // Cached state for backward
    Tensor last_input_h_;        // (N, in_features)
    Tensor last_input_x_;        // (N, coord_dim)
    Tensor last_adj_;            // (N, N)
    Tensor last_edge_attr_;      // (N, N, n_edge_attrs) — may be zeros
    bool last_use_edge_weights_ = false;
    Tensor last_edge_weights_;   // (N, N) — only set if last_use_edge_weights_
    size_t last_n_ = 0;
    Tensor last_h_out_;          // (N, hidden_dim) post-phi_h output
    Tensor last_x_out_;          // (N, coord_dim) post coord update
    Tensor last_m_;              // (N, hidden_dim) aggregated messages
    Tensor last_m_ij_;           // (N, N, hidden_dim) per-edge messages (post-ReLU)
    Tensor last_px_;             // (N, N) per-edge phi_x (post-ReLU)
    Tensor last_h_pre_relu_;     // (N, hidden_dim) pre-ReLU of phi_h output
    Tensor last_dist2_;          // (N, N) per-edge squared distance
    Tensor last_h_concat_;       // (N, 2 * hidden_dim) input to phi_h
    Tensor last_px_pre_relu_;    // (N, N) pre-ReLU of phi_x output (saved for backward)

    void relu_inplace(Tensor& t) const;
    bool is_relu() const { return activation_ == "relu"; }
};

// Full EGNN model: input projection (Dense) -> stack of EGNNLayer -> classifier.
class EGNNModel : public Layer {
public:
    EGNNModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
              size_t out_features, size_t coord_dim = 3,
              size_t n_layers = 3, size_t n_edge_attrs = 0,
              const std::string& activation = "relu");
    ~EGNNModel() override = default;

    // base-class forward: takes a flat (N, in_features) tensor; uses a
    // stored adjacency (set via set_adjacency) and a stored x (set via
    // set_coords).
    Tensor forward(const Tensor& input) override;

    Tensor forward_with_adj(const Tensor& input_h, const Tensor& input_x,
                            const Tensor& adj,
                            const Tensor& edge_attr = Tensor(),
                            const Tensor& weights = Tensor());

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "EGNNModel"; }

    Tensor get_last_x() const;
    void set_adjacency(const Tensor& adj) { stored_adj_ = adj; }
    void set_coords(const Tensor& x) { stored_x_ = x; }

private:
    size_t num_nodes_;
    size_t in_features_;
    size_t hidden_dim_;
    size_t out_features_;
    size_t coord_dim_;
    size_t n_layers_;
    size_t n_edge_attrs_;
    std::string activation_;

    Dense input_proj_;
    std::vector<EGNNLayer> layers_;
    Dense classifier_;

    // Cached state for backward
    Tensor last_input_h_;
    Tensor last_input_x_;
    Tensor last_adj_;
    Tensor last_edge_attr_;
    bool last_use_edge_weights_ = false;
    Tensor last_edge_weights_;

    // Generic-forward state
    Tensor stored_adj_;
    Tensor stored_x_;
    Tensor last_h_;
    Tensor last_x_;
};

#endif
