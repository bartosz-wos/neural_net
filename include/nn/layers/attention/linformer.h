#ifndef LINFORMER_ATTENTION_H
#define LINFORMER_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Linformer — Wang et al. 2020, "Linformer: Self-Attention with Linear Complexity"
// ============================================================================
//
// Standard multi-head self-attention has O(n^2) compute and memory in the
// sequence length n because of the softmax(QK^T)V step. Linformer's central
// observation: for typical n, the attention matrix A = softmax(QK^T) is
// approximately low-rank. We can therefore project K and V down to a much
// smaller dimension k << n along the sequence axis using two linear maps E, F:
//
//   K_reduced = E @ K     # (k, d_k)
//   V_reduced = F @ V     # (k, d_k)
//   A         = softmax(Q @ K_reduced^T / sqrt(d_k))   # (n, k)  ← n*k, not n*n
//   out       = A @ V_reduced                            # (n, d_k)
//
// This brings the attention step to O(n*k + n*d_k*k) instead of O(n^2*d_k).
//
// In the original paper E, F are FIXED random projections (or Nyström-style
// landmark selection). We expose this through a `learned_projection` flag:
//
//   learned_projection = true  →  E, F are Dense layers trained end-to-end
//                                  (this is the most common modern variant;
//                                  gives the model a bit of extra capacity).
//   learned_projection = false →  E, F are FIXED at construction time and
//                                  never updated (matches the original paper's
//                                  default and gives a stronger inductive
//                                  prior of "approximate low-rank attention").
//
// When k == n, Linformer is mathematically equivalent to vanilla attention
// (E and F become identity-like). We test that equivalence.
//
// Implementation notes:
//   * We use single-head Linformer (callers can wrap into multi-head with
//     multiple instances). This keeps the math transparent and the
//     gradient check tractable.
//   * Bias on the K, V projections? No — matches paper convention.
//   * Bias on E, F? Yes for learned, N/A for fixed (no params).
//   * Q is NOT projected along the seq axis — only K and V are. This is
//     what makes the speedup asymmetric and what the paper recommends.
//   * W_q, W_k, W_v, W_o, and (optionally) E, F are all learnable.
//
// LinformerBlock: pre-LN → LinformerAttn → residual → pre-LN → FFN → residual
//   This mirrors the standard transformer block layout (Vaswani et al. 2017
//   "post-LN" or "pre-LN" — we use pre-LN which is the modern default and
//   is what most Linformer implementations use). The FFN is a 2-layer Dense
//   with GELU.
//
// LinformerModel: stack of `num_blocks` LinformerBlocks + classifier head.
//   For a (n, d) input, output is (n, out_features) — per-token logits.
// ============================================================================

class LinformerAttention : public Layer {
public:
    // d_model:    input/output feature dim
    // seq_len:    sequence length n (fixed at construction; E, F are (k, n))
    // proj_dim:   low-rank projection size k. k = seq_len is vanilla attention.
    // learned_projection: if true, E, F are Dense layers; else fixed random.
    LinformerAttention(size_t d_model, size_t seq_len, size_t proj_dim = 0,
                       bool learned_projection = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "LinformerAttention"; }

    // Accessors for tests
    size_t d_model()  const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t proj_dim() const { return proj_dim_; }
    bool is_learned_projection() const { return learned_projection_; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t proj_dim_;
    bool   learned_projection_;

    // Learned projections: W_q, W_k, W_v map (n, d_model) -> (n, d_model)
    Tensor W_q, W_k, W_v, W_o;     // (d_model, d_model)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // Sequence-axis projections for K and V. We store as raw Tensors
    // (not Dense) for clarity in the gradient check, and so we can share
    // the same backward code regardless of learned_projection_.
    //
    // Layout: (proj_dim, seq_len) — applies as K_reduced = E @ K.
    Tensor E_, F_;                 // (k, n)
    Tensor grad_E_, grad_F_;       // (k, n)  — only used if learned_projection_

    double scale_;                 // 1 / sqrt(d_model)  (we don't split into heads)

    // Cached for backward
    Tensor last_input_;            // (n, d_model)
    Tensor last_q_;                // (n, d_model)
    Tensor last_k_;                // (n, d_model)
    Tensor last_v_;                // (n, d_model)
    Tensor last_k_reduced_;        // (k, d_model)   = E @ K
    Tensor last_v_reduced_;        // (k, d_model)   = F @ V
    Tensor last_attn_;             // (n, k)         softmax(Q K_reduced^T / sqrt(d))
    Tensor last_attn_output_;      // (n, d_model)   = A @ V_reduced
    Tensor last_output_;           // (n, d_model)   = last_attn_output_ @ W_o^T
};

// ============================================================================
// LinformerBlock — pre-LN → LinformerAttn → residual → pre-LN → FFN → residual
// ============================================================================

class LinformerBlock : public Layer {
public:
    LinformerBlock(size_t d_model, size_t seq_len, size_t proj_dim = 0,
                   size_t ffn_dim = 0, bool learned_projection = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "LinformerBlock"; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t ffn_dim_;

    LayerNorm ln1_;                   // (d_model)  pre-attn
    LinformerAttention attn_;         // core linformer attention
    LayerNorm ln2_;                   // (d_model)  pre-FFN
    Dense ffn_fc1_;                   // (ffn_dim, d_model)
    Dense fcn_fc2_;                   // (d_model, ffn_dim)

    // Cached
    Tensor last_input_;
    Tensor last_z1_;                  // ln1(x)
    Tensor last_attn_out_;
    Tensor last_res1_;                // x + attn
    Tensor last_z2_;                  // ln2(res1)
    Tensor last_ffn_hidden_;          // GELU(ffn_fc1(z2))
    Tensor last_ffn_out_;             // fc2(hidden)
};

// ============================================================================
// LinformerModel — stack of LinformerBlocks + classifier head
// ============================================================================

class LinformerModel : public Layer {
public:
    LinformerModel(size_t d_model, size_t seq_len, size_t out_features,
                   size_t num_blocks = 1, size_t proj_dim = 0,
                   size_t ffn_dim = 0, bool learned_projection = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "LinformerModel"; }

private:
    size_t d_model_;
    size_t seq_len_;
    size_t out_features_;
    size_t num_blocks_;
    std::vector<LinformerBlock> blocks_;
    Dense classifier_;                // (out_features, d_model)

    Tensor last_input_;
};

#endif
