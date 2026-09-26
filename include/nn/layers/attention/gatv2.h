#ifndef GATV2_ATTENTION_H
#define GATV2_ATTENTION_H

#include "../../core/layer.h"
#include <vector>

// GATv2 — Improved Graph Attention Network (Brody, Alon, Yahav 2022).
//
// "How Attentive are Graph Attention Networks?", ICLR 2023.
// https://arxiv.org/abs/2105.14491
//
// Per head h, with head_dim F':
//   Wh_i = W_h @ h_i                                    ∈ R^{F'}
//   s_ij = W_h @ [h_i || h_j]                           ∈ R^{F'}    (concat inputs FIRST)
//   e_ij = LeakyReLU( a_h^T · s_ij )                    ∈ R        (a_h ∈ R^{F'} — NOT 2F')
//   α_ij = softmax_j( e_ij )     over j ∈ N(i) ∪ {i}   (row-softmax)
//   h'_i = LeakyReLU( sum_j α_ij · Wh_j )               ∈ R^{F'}
//
// Multi-head: concat (default) or average across heads. Output dim = out_features.
//
// LeakyReLU slope α_LR = 0.2 (per paper, matches GAT v1 convention).
//
// Distinguishing property from GAT v1 (paper Theorem 1): v2's score `a^T W [h_i || h_j]`
// has a SHARED `W` projected against both halves, so the linear combination of `h_i`'s
// features that determines attention weight is COUPLED to the linear combination of `h_j`'s
// features through learning. GAT v1's `a^T [W h_i || W h_j]` decomposes into `a_1^T W h_i + a_2^T W h_j`
// — two INDEPENDENT projections — which yields the "static" attention problem (ranking of
// neighbors is the same for every query).
//
// Per-head parameter count: |W| + |a| = F'·in + F' = F'·(in+1). GAT v1's per-head is
// 2F'·in + 2F' = 2F'·(in+1) — exactly TWICE. With F' chosen so out_features matches, v2
// can use num_heads' = 2·num_heads to keep total param count equal.

struct GATv2HeadParams {
    Tensor W;          // (head_dim, in_features)
    Tensor a;          // (head_dim, 1) — note: F' NOT 2F'
    Tensor grad_W;
    Tensor grad_a;
};

class GATv2Layer : public Layer {
public:
    GATv2Layer(size_t in_features, size_t out_features, size_t num_heads = 4,
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

    size_t num_heads()    const { return num_heads_; }
    size_t in_features()  const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t head_dim()     const { return head_dim_; }
    bool   concat_heads() const { return concat_heads_; }

private:
    size_t num_heads_;
    bool concat_heads_;
    size_t in_features_;
    size_t out_features_;
    size_t head_dim_;
    std::vector<GATv2HeadParams> heads_;

    // Cached state for backward
    Tensor last_output_;
    Tensor last_input_;
    std::vector<Tensor> last_Wh_heads_;     // (N, head_dim) per head — pre-softmax Wh
    Tensor last_alpha_;                     // (N, N * num_heads_) post-softmax attention
    Tensor last_e_;                         // (N, N * num_heads_) pre-softmax LeakyReLU scores
    Tensor last_head_pre_;                  // (N, head_dim * num_heads_) pre-LeakyReLU weighted sums
    Tensor adj_;                            // (N, N) stored adjacency
};

// Stackable GATv2 model: input projection (Dense-equivalent via in_dim→d_model) +
// N GATv2Layers + output projection (mean-pool or per-node). Kept minimal here — v1 of
// the repo's GATLayer doesn't ship a GATModel counterpart either, but the paper expects
// multi-layer stacks. We provide a minimal version: a stack of GATv2Layers with no
// between-layer Dense, returning per-node features.
class GATv2Model : public Layer {
public:
    GATv2Model(size_t in_features, size_t hidden_features, size_t out_features,
               size_t num_layers = 2, size_t num_heads = 4);
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
    size_t in_features_;
    size_t hidden_features_;
    size_t out_features_;
    size_t num_layers_;
    size_t num_heads_;
    std::vector<GATv2Layer> layers_;

    Tensor last_input_;
    Tensor adj_;
    std::vector<Tensor> layer_inputs_;      // cached per-layer inputs for backward
};

#endif
