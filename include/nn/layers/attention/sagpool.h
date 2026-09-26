#ifndef SAGPOOL_H
#define SAGPOOL_H

#include "../../core/layer.h"
#include <vector>
#include <utility>

// ============================================================================
// SAGPool — Self-Attention Graph Pooling
//   Lee, Lee, Kang — ICML 2019, https://arxiv.org/abs/1904.08082
//
// Graph U-Net-style pooling layer. Computes a per-node attention score via a
// single GraphConv (GCN), takes a hard top-`⌈kN⌉` selection by score, then
// pools both the node features (multiplied element-wise by their score) and
// the adjacency (sub-selected rows/cols).
//
// Forward (input X ∈ R^{N × in_features}, adj A ∈ R^{N × N}):
//   1. Â = D̂^{-1/2}(A + I)D̂^{-1/2}     — symmetric norm with self-loops
//   2. Z_raw = ReLU(Â X W_pool)         ∈ R^{N × 1}   (W_pool ∈ R^{in_features × 1})
//   3. Z     = softmax(Z_raw)           ∈ R^{N}        (over all N nodes)
//   4. idx   = topk_idx(Z, ⌈kN⌉)        ∈ R^{N_out}   (N_out = ⌈kN⌉)
//   5. X'    = (X ⊙ Z)[idx, :]          ∈ R^{N_out × in_features}
//   6. A'    = A[idx, idx]              ∈ R^{N_out × N_out}
//
// Two variants exposed:
//   - use_gcn=true (default): score via the GCN step (1–2) — topology-aware
//   - use_gcn=false: score = X W_pool directly — simpler, no adj needed
//
// The topk is treated as a fixed mask during backward (standard practice; the
// PyG reference also treats it this way). Gradient flows only to the selected
// nodes, not to the discarded ones.
//
// Backward chain (in order, starting from grad_X' / grad_A'):
//   1. dL/d(X[some_row, j]) += grad_X'[i, j] * Z[some_row]   for some_row = idx[i]
//   2. dL/d(Z[idx[i]])     = Σ_j grad_X'[i, j] * X[idx[i], j]
//   3. dL/d(Z[k])           = 0 if k ∉ idx (dropped nodes get no signal)
//   4. softmax backward:   dL/d(Z_raw[k]) = Z[k] * (dZ[k] − Σ_m Z[m]·dZ[m])
//   5. ReLU backward:      dL/d(G_pre[k]) = (Z_raw[k] > 0) ? dL/d(Z_raw[k]) : 0
//   6. GCN backward:
//        dL/d(W_pool)  = X^T · (Â^T · dL/d(G_pre))        ∈ R^{in_features × 1}
//        dL/d(X) += (Â^T · dL/d(G_pre)) · W_pool^T         ∈ R^{N × in_features}
//        dL/d(Â)    = outer(dL/d(G_pre), G_pre^T)            ∈ R^{N × N}
//   7. grad_A: dL/d(A[i,j]) += D̂^{-1/2}[i]·D̂^{-1/2}[j]·dL/d(Â)[i,j]   (i≠j, fixed D̂)
//      (the diagonal of A is typically 0 anyway)
//
// Why this layer matters:
//   - Graph U-Net building block (the "pool" half of the encode-decode stack).
//   - Unlike random pooling (uniform drop), uses LEARNED per-node importance
//     computed via GCN so the topology is respected.
//   - Unlike DiffPool (soft assignment), uses HARD top-k — simpler, faster, often
//     more accurate for graph classification.
//   - Compositional with GAT/GATv2 (per-edge attention): a typical block is
//     `GCN → GAT → SAGPool → GCN → SAGPool → readout → classifier`.
// ============================================================================

class SAGPool : public Layer {
public:
    SAGPool(size_t in_features, double ratio = 0.5, bool use_gcn = true);

    // The Layer::forward interface without adj is not meaningful for a graph
    // pooling layer. Callers MUST use forward_with_adj. We throw to be loud.
    Tensor forward(const Tensor& input) override;

    // Real forward. Returns {X', A'}.
    std::pair<Tensor, Tensor> forward_with_adj(const Tensor& input, const Tensor& adj);

    // Backward. `grad_output` is the upstream gradient w.r.t. X'. `grad_A` (the
    // upstream gradient w.r.t. A') is treated as zero in this interface — the
    // standard Layer::backward(grad_output, lr) signature doesn't allow a
    // second gradient input. Callers that want to pass grad_A' explicitly
    // should use backward_with_adj_grads.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    Tensor backward_with_adj_grads(const Tensor& grad_X, const Tensor& grad_A,
                                    double learning_rate);

    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "SAGPool"; }

    size_t in_features() const { return in_features_; }
    double ratio() const { return ratio_; }
    bool use_gcn() const { return use_gcn_; }
    size_t num_nodes_out(size_t N) const;

    // For testing/inspection
    Tensor W_pool() const { return W_pool_; }
    Tensor grad_W_pool() const { return grad_W_pool_; }
    std::vector<size_t> last_idx() const { return last_idx_; }
    Tensor last_Z() const { return last_Z_; }
    Tensor last_grad_A() const { return last_grad_A_; }

private:
    size_t in_features_;
    double ratio_;
    bool use_gcn_;

    Tensor W_pool_;        // (in_features, 1)
    Tensor grad_W_pool_;   // (in_features, 1)

    // Cached state from last forward
    Tensor last_input_;         // (N, in_features)
    Tensor last_A_;             // (N, N) raw adjacency
    Tensor last_A_hat_;         // (N, N) symmetric-normalized (A + I)
    Tensor last_Z_raw_;         // (N, 1) pre-softmax scores
    Tensor last_Z_;             // (N, 1) post-softmax scores
    std::vector<size_t> last_idx_;  // (N_out,) selected indices
    size_t last_N_;             // original N
    size_t last_N_out_;         // ⌈ratio·N⌉

    // Last backward
    Tensor last_grad_A_;        // (N, N) — only meaningful if caller called backward_with_adj_grads
};

// ============================================================================
// SAGPoolHierarchical — alternating GCN/SAGPool + sum-readout + classifier.
// Implements the paper's "POOL_h" architecture.
// ============================================================================

class SAGPoolHierarchical : public Layer {
public:
    SAGPoolHierarchical(size_t in_features, size_t hidden_features, size_t num_classes,
                        double ratio = 0.5, size_t num_layers = 2);

    Tensor forward(const Tensor& input) override;
    std::pair<Tensor, Tensor> forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "SAGPoolHierarchical"; }

private:
    // Layout: [GCN_0, Pool_0, GCN_1, Pool_1, ..., GCN_{L-1}, Pool_{L-1}]
    // num_layers counts the number of (GCN, Pool) pairs.
    struct GCNBlock {
        Dense W;
        std::vector<std::vector<double>> relu_mask;  // (N, hidden) per cached forward
        Tensor last_input;
        Tensor last_A_hat;
        Tensor last_AX;
        Tensor last_output;

        GCNBlock() : W(1, 1) {}  // placeholder; real init via assignment below
        explicit GCNBlock(size_t in_features, size_t out_features) : W(in_features, out_features) {}
    };
    struct PoolBlock {
        SAGPool pool;
        std::vector<size_t> last_idx;
        Tensor last_input;
        Tensor last_Z;
        Tensor last_X_out;  // (X ⊙ Z)[idx, :]
        size_t last_N;
        size_t last_N_out;

        PoolBlock() : pool(1, 0.5, true), last_N(0), last_N_out(0) {}  // placeholder; real init via assignment below
        explicit PoolBlock(size_t in_features, double ratio)
            : pool(in_features, ratio, true), last_N(0), last_N_out(0) {}
    };

    size_t in_features_;
    size_t hidden_features_;
    size_t num_classes_;
    double ratio_;
    size_t num_layers_;

    std::vector<GCNBlock> gcn_blocks_;
    std::vector<PoolBlock> pool_blocks_;
    Dense readout_;          // hidden_features_ → num_classes_

    // Cache for the last forward
    std::vector<Tensor> layer_inputs_;  // [input, gcn0_out, pool0_out, gcn1_out, ...]
    Tensor last_A_;
    Tensor last_readout_input_;  // sum-pooled features from final pool block
    Tensor last_classifier_out_;
};

// ============================================================================
// SAGPoolGlobal — single GCN + single SAGPool + sum-readout + classifier.
// Implements the paper's "POOL_g" architecture.
// ============================================================================

class SAGPoolGlobal : public Layer {
public:
    SAGPoolGlobal(size_t in_features, size_t hidden_features, size_t num_classes,
                  double ratio = 0.5);

    Tensor forward(const Tensor& input) override;
    std::pair<Tensor, Tensor> forward_with_adj(const Tensor& input, const Tensor& adj);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "SAGPoolGlobal"; }

private:
    size_t in_features_;
    size_t hidden_features_;
    size_t num_classes_;
    double ratio_;

    Dense W_gcn_;
    SAGPool pool_;
    Dense readout_;
    Tensor last_input_;
    Tensor last_A_;
    Tensor last_A_hat_;
    Tensor last_AX_;
    std::vector<std::vector<double>> last_relu_mask_;
    Tensor last_gcn_out_;
    std::vector<size_t> last_idx_;
    Tensor last_X_pooled_;  // (X ⊙ Z)[idx, :]
    Tensor last_readout_input_;  // sum-pooled
    Tensor last_classifier_out_;
};

#endif
