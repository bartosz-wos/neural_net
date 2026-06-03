#ifndef PNA_H
#define PNA_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>

// Principal Neighbourhood Aggregation (PNA)
// Corso, Cavalleri, Bezenac, Glavas, Bronstein, Vandergheynst.
// "Principal Neighbourhood Aggregation for Graph Nets".
// NeurIPS 2020.
//
// PNA generalises graph convolution by combining multiple aggregators
// with multiple degree-based scalers. The design is motivated by
// the observation that no single aggregator works well across all
// graph distributions (degree profile, homophily, etc.).
//
// Per node i, for each feature dimension f:
//   1. AGGREGATE:  for a in {mean, max, min, std}:
//                     m_i^{(a)}[f] = AGG_a({ h_j[f] : j in N(i) }).
//   2. SCALE:      for s in {identity, amplification, attenuation}:
//                     scale_s(deg_i) * m_i^{(a)}[f].
//                  identity:            m_i^{(a)}[f]
//                  amplification:       log(deg_i + 1) / delta * m_i^{(a)}[f]
//                  attenuation:         delta / log(deg_i + 1) * m_i^{(a)}[f]
//                  where delta is a per-layer learnable average of the
//                  amplifier log-deg statistic (kept simple here as a fixed
//                  hyperparameter; the PNA paper learns it via a Dense layer
//                  from degree log; we use a single scalar constant for
//                  numerical robustness).
//   3. CONCATENATE all (aggregator, scaler) pairs:
//         x_i = [ agg_1_scal_1 || agg_1_scal_2 || ... || agg_A_scal_S ].
//      Dimension:  in_features * num_aggregators * num_scalers.
//   4. POST-AGG dense: x_i = W * x_i + b  (no nonlinearity at output;
//      the next layer's activation provides it).
//
// Gradients:
//   - Each aggregator is differentiable (we hand-code backward for
//     mean, max, min, std).
//   - The scalers are scalar multiplications of the aggregated tensor,
//     so the backward pass simply scales gradients accordingly.
//   - The post-aggregation Dense handles its own backward pass.
//
// Forward/backward store a per-node list of (aggregator_value,
// index_of_argmin/argmax, neighbour_features) so that gradient
// computation can route the contribution to the right neighbour.

class PNALayer : public Layer {
public:
    // in_features: per-node input feature dimension
    // out_features: per-node output feature dimension
    // deg_log_delta: scaling constant used in amplification/attenuation
    //                scalers (per-layer learnable scalar would be the
    //                paper version; we keep a constant for simplicity)
    PNALayer(size_t in_features, size_t out_features, double deg_log_delta = 1.0);
    ~PNALayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "PNALayer"; }

private:
    size_t in_features_;
    size_t out_features_;
    double deg_log_delta_;

    // num_aggregators = 4 (mean, max, min, std)
    // num_scalers = 3 (identity, amplification, attenuation)
    static constexpr size_t num_aggregators_ = 4;
    static constexpr size_t num_scalers_ = 3;

    // Post-aggregation dense: (in_features * num_aggregators * num_scalers) -> out_features
    Dense post_agg_;

    // Cached state for backward
    Tensor last_input_;          // (N, in_features)
    Tensor adj_;                 // (N, N) adjacency (no self-loops)
    Tensor last_concat_;         // (N, in_features * num_aggregators * num_scalers)
    Tensor last_deg_log_;        // (N,) log(deg_i + 1)
    size_t num_nodes_;

    // Per-aggregator, per-feature, per-node bookkeeping for backward
    // last_agg_values_[a] has shape (N, in_features) — the aggregator output
    std::vector<Tensor> last_agg_values_;
    // For max aggregator: which neighbour won (per node, per feature).
    // shape: (N, in_features) holding neighbour index for max.
    std::vector<std::vector<size_t>> last_argmax_;
    // For min aggregator: same layout as last_argmax_.
    std::vector<std::vector<size_t>> last_argmin_;

    // Compute degree vector (no self-loops): deg[i] = sum_j adj[i][j]
    std::vector<size_t> compute_degrees(const Tensor& adj) const;
    // Aggregator implementations
    Tensor agg_mean(const Tensor& input, const Tensor& adj) const;
    Tensor agg_max(const Tensor& input, const Tensor& adj, std::vector<std::vector<size_t>>& argmax) const;
    Tensor agg_min(const Tensor& input, const Tensor& adj, std::vector<std::vector<size_t>>& argmin) const;
    Tensor agg_std(const Tensor& input, const Tensor& adj, const Tensor& mean_agg) const;
    // Scaler applications
    Tensor apply_scaler_identity(const Tensor& agg) const;
    Tensor apply_scaler_amplification(const Tensor& agg, const std::vector<double>& deg_log) const;
    Tensor apply_scaler_attenuation(const Tensor& agg, const std::vector<double>& deg_log) const;
};

// Full PNA model: optional input projection + PNALayer stack + classifier
class PNAModel : public Layer {
public:
    // num_nodes: number of nodes in the graph
    // in_features: per-node input feature dimension
    // hidden_dim: hidden dimension for PNA layers
    // out_features: per-node output feature dimension
    // num_layers: number of PNA layers to stack
    // deg_log_delta: scaler delta (per-layer)
    PNAModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
             size_t out_features, size_t num_layers = 2, double deg_log_delta = 1.0);
    ~PNAModel() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_adj(const Tensor& input, const Tensor& adj);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "PNAModel"; }

private:
    size_t num_nodes_;
    size_t in_features_;
    size_t hidden_dim_;
    size_t out_features_;
    size_t num_layers_;

    // Input projection (identity if in_features == hidden_dim)
    Dense input_proj_;
    // Stack of PNA layers
    std::vector<PNALayer> pna_layers_;
    // Output classifier
    Dense classifier_;

    // Cached state for backward
    Tensor last_input_;
    Tensor last_input_proj_;
    // Pre-ReLU PNA output of each layer; used as the ReLU mask in backward.
    std::vector<Tensor> layer_inputs_;
    Tensor adj_;
};

#endif
