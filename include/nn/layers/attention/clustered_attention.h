#ifndef CLUSTERED_ATTENTION_H
#define CLUSTERED_ATTENTION_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Clustered Attention — Vyas, Katharopoulos, Fleuret (NeurIPS 2020)
//   "Fast Transformers with Clustered Attention"
//   (https://arxiv.org/abs/2007.04825)
//
// A fast O(N·C·D) approximation to softmax attention via query clustering.
//
// Vanilla attention has O(N²D) cost because each of N queries computes a full
// softmax against N keys. Clustered attention replaces that with an O(N·C·D)
// approximation: queries are grouped into C non-overlapping clusters
// (computed via K-Means on the query vectors), softmax attention is computed
// ONCE per centroid rather than once per query, and the resulting per-cluster
// output is BROADCAST to all queries in the same cluster. With C=N this
// collapses to vanilla attention (each query is its own cluster, no
// approximation loss).
//
// Per-head forward (Q, K, V ∈ R^{n × d_k}, C clusters, optional top-k = 0):
//
//   1. Cluster queries → S ∈ {0,1}^{n × C} one-hot, Q_c ∈ R^{C × d_k} centroids
//      (K-Means, `iters` Lloyd iters, random init seeded by the caller).
//   2. A_c = softmax(Q_c · K^T / √d_k) ∈ R^{C × n}
//   3. V_c = A_c · V                       ∈ R^{C × d_v}
//   4. V̂_i = V_c[cluster_assign[i], :]     ∈ R^{d_v}   (broadcast)
//
// Improved-clustered variant (top-k redistribution, paper §3.3):
//   The paper's fix for approximation error: per cluster, identify the top-k
//   keys by A_c weight, then redistribute the mass M̂_j assigned to those
//   keys to exact per-query softmax(Q_i · K_l/√d_k) over those positions.
//   With top_k = n (full mask) this also reduces to vanilla attention bit
//   exactly — a useful FD-check anchor.
//
// Multi-head: d_model split into num_heads × head_dim, run per head, concat,
// project through W_o. Pre-LN block pattern matches the rest of this repo.
//
// Backward (vanilla clustered, no top-k):
//   The cluster assignment is fixed during backward (paper's "fixed centroid"
//   assumption — matches the paper's analysis):
//     - grad_V_c is accumulated from per-query grads (broadcast-back)
//     - grad_A_c = V^T · grad_V_c           (matmul backward)
//     - grad_A_pre_c = A_c ⊙ (grad_A_c − sum_l A_c⊗grad_A_c)  (softmax backward)
//     - grad_Q_c = (K · grad_A_pre_c^T / √d_k)
//     - grad_Q[i] += grad_Q_c[cluster_assign[i]] / |cluster|  (centroid update)
//     - grad_K = (Q_c · grad_A_pre_c / √d_k)
//
// Why this layer matters:
//   - The canonical "query-clustering" sparse-attention primitive (paper is
//     cited as the standard reference for any O(N·C) attention approximation
//     in speech / ASR).
//   - Goes well with GlobalTokens / Windowed / SlidingWindow variants (all
//     focus on keys; this one focuses on queries) — the classic missing
//     piece in this repo's attention library.
//   - Backward is the only meaningful complexity; FD-check anchor via C=N
//     (clustering recovery) and top_k=N (improved-clustered recovery)
//     makes it straightforward to verify.
// ============================================================================

class ClusteredAttention : public Layer {
public:
    ClusteredAttention(size_t d_model, size_t num_heads, size_t num_clusters,
                      size_t top_k = 0, size_t iters = 5, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q_; }
    Tensor get_gradients() const override { return grad_W_q_; }
    std::string name() const override { return "ClusteredAttention"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t num_clusters() const { return num_clusters_; }
    size_t top_k() const { return top_k_; }
    size_t iters() const { return iters_; }
    bool causal() const { return causal_; }
    size_t last_seq_len() const { return last_n_; }
    size_t last_num_clusters() const { return last_c_actual_; }
    const std::vector<int>& last_cluster_assign() const { return last_assign_; }

    // Test/inspection setters
    void set_seed(unsigned s) { seed_ = s; }
    void set_weights_for_test(const Tensor& W_q, const Tensor& W_k,
                              const Tensor& W_v, const Tensor& W_o) {
        W_q_ = W_q.clone(); W_k_ = W_k.clone(); W_v_ = W_v.clone(); W_o_ = W_o.clone();
        grad_W_q_ = Tensor::zeros(W_q.rows, W_q.cols);
        grad_W_k_ = Tensor::zeros(W_k.rows, W_k.cols);
        grad_W_v_ = Tensor::zeros(W_v.rows, W_v.cols);
        grad_W_o_ = Tensor::zeros(W_o.rows, W_o.cols);
    }

private:
    // Hyperparams
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;        // d_model / num_heads
    size_t num_clusters_;    // user's C
    size_t top_k_;
    size_t iters_;
    bool causal_;
    double scale_;           // 1/sqrt(head_dim)
    unsigned seed_;

    // Parameters (4 projections, each (d_model, d_model))
    Tensor W_q_, W_k_, W_v_, W_o_;
    Tensor grad_W_q_, grad_W_k_, grad_W_v_, grad_W_o_;
    Tensor last_input_;

    // Per-head caches
    Tensor last_Q_, last_K_, last_V_;
    Tensor last_output_;
    std::vector<int> last_assign_;        // per-query cluster index
    size_t last_n_;                      // sequence length at forward
    size_t last_c_actual_;               // actual clusters used (≤ min(num_clusters_, n))
};

#endif
