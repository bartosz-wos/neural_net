#ifndef GAT_ATTENTION_H
#define GAT_ATTENTION_H

#include "../../core/layer.h"
#include <vector>

// Graph Attention Network (Veličković et al. 2018, ICLR).
//
// Per head h, with head_dim F':
//   Wh_i = W_h @ h_i                                    ∈ R^{F'}
//   e_ij = LeakyReLU( a_h^T [ Wh_i || Wh_j ] )          ∈ R       (a_h ∈ R^{2F'})
//   α_ij = softmax_j( e_ij )     over j ∈ N(i) ∪ {i}   (row-softmax)
//   h'_i = LeakyReLU( sum_j α_ij Wh_j )                 ∈ R^{F'}
//
// Multi-head: concat (default) or average across heads. Output dim = out_features.
//
// LeakyReLU slope α_LR = 0.2 (per paper).
//
// Lives in layers/attention/ because GAT is fundamentally a graph-domain analogue
// of multi-head attention: the same softmax-over-scores pattern as
// transformer attention, with the adjacency matrix replacing the dense similarity
// kernel and a learned attention vector a_h replacing the dot-product.

struct GATHeadParams {
    Tensor W;       // (head_dim, in_features)
    Tensor a;       // (2*head_dim, 1) — flat-indexed as a[k][0] for k in [0, 2*head_dim)
    Tensor grad_W;
    Tensor grad_a;
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
    size_t in_features_;
    size_t out_features_;
    size_t head_dim_;
    std::vector<GATHeadParams> heads_;

    // Cached state for backward
    Tensor last_output_;
    Tensor last_input_;
    std::vector<Tensor> last_Wh_heads_;     // (N, head_dim) per head — pre-softmax Wh
    Tensor last_alpha_;                     // (N, N * num_heads_) post-softmax attention
    Tensor last_e_;                         // (N, N * num_heads_) pre-softmax LeakyReLU scores
    Tensor last_head_pre_;                  // (N, head_dim * num_heads_) pre-LeakyReLU weighted sums
    Tensor adj_;                            // (N, N) stored adjacency
};

#endif
