#ifndef LAMBDA_LAYER_H
#define LAMBDA_LAYER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Lambda Layer — Bello et al. ICLR 2021
//   "LambdaNetworks: Modeling Long-Range Interactions Without Attention"
//   https://arxiv.org/abs/2102.08602
//
// ============================================================================
//
// Lambda Layer replaces `softmax(Q·K^T)·V` with the application of a
// per-query "lambda matrix" that linearly maps queries to outputs:
//
//   Q = X · W_Q                                    ∈ R^{n × k}
//   K = X · W_K                                    ∈ R^{n × k}
//   V = X · W_V                                    ∈ R^{n × d}
//   K̄ = col_softmax(K)                            ∈ R^{n × k}
//                                                     (softmax across positions
//                                                      for each k-column)
//   λ_c   = K̄^T · V                                ∈ R^{k × d}
//                                                     (content lambda, shared
//                                                      across query positions)
//   λ_p_n = Σ_m E[n, m, :] · V[m, :]              ∈ R^{k × d}
//                                                     (position lambda, per
//                                                      query position)
//   λ_n   = λ_c + λ_p_n                            ∈ R^{k × d}
//   Y[n]  = Q[n] · λ_n                             ∈ R^d
//
// Relative position embeddings E ∈ R^{n×n×k} are learnable.
//
// Memory cost vs softmax attention:
//   * softmax attention:  O(n²·d) per layer for the attention map
//   * lambda:             O(n²·k) for E
//   With k ≤ d (multi-query), this is the dominant memory win.
//
// Key insight (Bello et al.): λ_c depends on content only and is shared; λ_p
// depends on positions only and varies per query. The two paths sum into a
// small (k, d) "linear function" λ_n that contextualizes the query q_n.
//
// ----------------------------------------------------------------------------
// Conventions (match the rest of the attention/ directory):
//   * Input/Output: (n, d_model) — n tokens, d_model features
//   * W_Q, W_K: raw (d_model, k) weights — Q = X · W_Q, K = X · W_K
//   * W_V: raw (d_model, d_model) weights — V = X · W_V
//   * position_emb_: stored as (max_seq_len, max_seq_len * k); index via
//                    E[n, m, kk] = position_emb_(n, m * k + kk)
//                    (2D storage keeps the tensor API consistent)
//   * Single-head v1; multi-query/multi-head is OUT OF SCOPE for v1.
//   * `k_depth = 0` in the constructor defaults to `d_model` (full-rank).
//   * `causal = true` zeros position embeddings for m > n (forward-only mask).
// ----------------------------------------------------------------------------

class LambdaAttention : public Layer {
public:
    // d_model: input/output feature dim
    // max_seq_len: maximum sequence length (size of the position-embedding table)
    // k_depth: query/key depth (defaults to d_model when 0)
    // causal: if true, restrict position embeddings to m ≤ n (lower-triangular mask)
    LambdaAttention(size_t d_model, size_t max_seq_len, size_t k_depth = 0, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_Q; }
    Tensor get_gradients() const override { return grad_W_Q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LambdaAttention"; }

    size_t d_model() const { return d_model_; }
    size_t max_seq_len() const { return max_seq_len_; }
    size_t k_depth() const { return k_depth_; }
    bool causal() const { return causal_; }

    // Public parameters (raw Tensors for direct FD-style gradient checks)
    Tensor W_Q;            // (d_model, k_depth)
    Tensor W_K;            // (d_model, k_depth)
    Tensor W_V;            // (d_model, d_model)
    Tensor position_emb_;  // (max_seq_len, max_seq_len * k_depth)
    Tensor grad_W_Q, grad_W_K, grad_W_V;
    Tensor grad_position_emb_;

    // Forward caches (mutable across forward calls; tests can read these)
    Tensor last_input_;          // (n, d_model)
    Tensor last_Q_, last_K_, last_V_;
    Tensor last_K_softmax_;      // (n, k_depth)
    Tensor last_lambda_content_; // (k_depth, d_model)
    Tensor last_lambda_pos_;     // (n, k_depth, d_model) — concatenated as flat (n, k_depth * d_model); see strides below
    Tensor last_lambda_total_;   // (n, k_depth, d_model) — flat (n, k_depth * d_model)
    Tensor last_output_;         // (n, d_model)

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t k_depth_;
    bool causal_;
};

// ----------------------------------------------------------------------------
// LambdaBlock — pre-LN -> LambdaAttention -> residual -> pre-LN -> GELU FFN -> residual
// (matches the pattern of GAUBlock; ensures compositional parity with the
//  other attention-block wrappers in the repo)
// ----------------------------------------------------------------------------
class LambdaBlock : public Layer {
public:
    LambdaBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim = 0,
                size_t k_depth = 0, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return attn.W_Q; }
    Tensor get_gradients() const override { return attn.grad_W_Q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LambdaBlock"; }

    LambdaAttention attn;
    LayerNorm ln1;            // (d_model,)
    LayerNorm ln2;            // (d_model,)
    Tensor W_ffn1_;           // (d_model, ffn_dim)
    Tensor W_ffn2_;           // (ffn_dim, d_model)
    Tensor b_ffn1_;           // (1, ffn_dim)
    Tensor b_ffn2_;           // (1, d_model)
    Tensor grad_W_ffn1_, grad_W_ffn2_, grad_b_ffn1_, grad_b_ffn2_;

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_h_pre_;
    Tensor last_h_act_;
    Tensor last_ffn_out_;
};

// ----------------------------------------------------------------------------
// LambdaModel — stack of LambdaBlocks + per-token classifier
// (matches GAUModel: no final LayerNorm — the last block ends with FFN+residual)
// ----------------------------------------------------------------------------
class LambdaModel : public Layer {
public:
    LambdaModel(size_t d_model, size_t max_seq_len, size_t out_features,
                size_t num_blocks, size_t ffn_dim = 0, size_t k_depth = 0, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return classifier_W_; }
    Tensor get_gradients() const override { return grad_classifier_W_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LambdaModel"; }

    std::vector<std::unique_ptr<LambdaBlock>> blocks;
    Tensor classifier_W_; // (d_model, out_features)
    Tensor classifier_b_; // (1, out_features)
    Tensor grad_classifier_W_, grad_classifier_b_;

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t out_features_;
    size_t num_blocks_;
    size_t ffn_dim_;
    Tensor last_input_;
    Tensor last_block_output_;
};

#endif // LAMBDA_LAYER_H
