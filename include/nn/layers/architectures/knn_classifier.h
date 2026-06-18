#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================================
// k-Nearest Neighbors Classifier (non-parametric layer)
//
// A non-parametric classifier that compares query feature vectors against a
// stored "support set" of (feature, label) pairs and produces class
// probabilities via distance-weighted soft-voting over the k nearest
// neighbors.
//
// Reference: Cover & Hart 1967 "Nearest Neighbor Pattern Classification"
//   and the distance-weighted variants in Atiya 1991 / Dudani 1976.
//
// This layer is "non-parametric" — it has no learned weights — but it is
// still differentiable with respect to its INPUT, which makes it usable
// as a black-box layer inside a larger model (e.g. a deep feature
// extractor trained to produce embeddings that classify well via kNN,
// the classic approach used in metric learning papers and in
// Prototypical Networks / Matching Networks for few-shot learning).
//
// Conventions:
//   - Support features: Tensor (N_support, feature_dim).
//   - Support labels:   std::vector<int> of size N_support, values in
//                       [0, num_classes).
//   - Query features:   Tensor (batch, feature_dim).
//   - Output:           Tensor (batch, num_classes) — class probabilities
//                       (rows sum to 1).
//
// Algorithm (per query q):
//   1. Compute squared-Euclidean distance d^2(q, s_i) for all support i.
//   2. Find the k indices i with smallest d^2.
//   3. Compute softmax weights over the k: w_i = softmax(-d^2_i / tau)
//      where the softmax is restricted to the k nearest (the other N-k
//      entries contribute zero weight).
//   4. Output: p[c] = sum_{i in top-k} w_i * 1[label_i == c].
//
// Backward:
//   - Only the query input has a gradient (the support set is fixed data).
//   - The gradient flows: d p -> d w_i -> d (-d^2_i / tau) -> d d^2 -> d q.
//   - d w_i / d a_j = w_i (delta_ij - w_j) where a_i = -d^2_i / tau.
//   - d d^2(q, s_i) / d q[d] = 2 (q[d] - s_i[d]).
//
// Edge cases:
//   - k is clamped to N_support.
//   - If N_support == 0 the layer returns uniform probabilities (and
//     backward returns a zero gradient).
// ============================================================================

class KNNClassifier : public Layer {
public:
    // feature_dim:   dimension of input/query/support feature vectors
    // num_classes:   number of output classes
    // k:             number of nearest neighbors to use (clamped to support size)
    // temperature:   softmax temperature on the squared-distance logits
    //                (smaller = sharper voting; larger = softer)
    KNNClassifier(size_t feature_dim, size_t num_classes,
                  size_t k = 3, double temperature = 1.0);

    // Fit / replace the support set.
    // support_features: (N_support, feature_dim)
    // support_labels:   length-N_support vector of integer class labels
    //                  (must be in [0, num_classes)).
    void fit(const Tensor& support_features, const std::vector<int>& support_labels);

    // Convenience: does the support set have any data?
    bool is_fitted() const { return support_features_.rows > 0; }

    size_t support_size() const { return support_features_.rows; }

    // Layer interface.
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override;     // returns the support set as a flat tensor
    Tensor get_gradients() const override;   // returns last input-grad (for inspection)

    std::string name() const override { return "KNNClassifier"; }

private:
    size_t feature_dim_;
    size_t num_classes_;
    size_t k_;
    double temperature_;

    // Stored support set
    Tensor support_features_;                       // (N_support, feature_dim)
    std::vector<int> support_labels_;               // length N_support

    // Forward cache (per batch element, since the top-k set is per-query)
    Tensor last_input_;                             // (batch, feature_dim)
    // For each query q: which support indices were selected and with what weight.
    // Both have shape (batch, k); stored as raw vectors since top-k is per-row.
    std::vector<std::vector<size_t>> last_topk_idx_;   // (batch, k)
    std::vector<std::vector<double>> last_topk_w_;     // (batch, k) softmax weights
    // Per-query, per-support gradient accumulator (used in backward).
    // We accumulate dL/d(-d^2_i / tau) into a (batch, N_support) buffer and
    // use it to drive dL/d_input. Only the top-k entries are nonzero.
    Tensor last_negd2_tau_;                        // (batch, N_support) — the
                                                   //   -d^2 / tau values for
                                                   //   the top-k (others 0)
};

#endif