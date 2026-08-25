#ifndef SHLA_H
#define SHLA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// SHLA (Shared-Head Latent Attention) — variant of DeepSeek-MLA with SHARED
// K and V up-projection matrices across heads.
//
// Reference parent: "DeepSeek-V2: A Strong, Economical, and Efficient
//   Mixture-of-Experts Language Model" (https://arxiv.org/abs/2405.04434).
// See `mla.h` for the per-head-up-projection variant.
//
// The variant implemented here:
//
//   c_Q  = X @ W_dq                (n, d_c)       # shared Q down
//   Q_h  = c_Q @ W_uq[h]           (n, head_dim)  # PER-HEAD Q up (stacked)
//   c_KV = X @ W_dkv               (n, d_c)       # shared KV down (one latent)
//   K_h  = c_KV @ W_uk_shared      (n, head_dim)  # SHARED K up across all heads
//   V_h  = c_KV @ W_uv_shared      (n, head_dim)  # SHARED V up across all heads
//   attn_h   = softmax(Q_h @ K_h^T / sqrt(head_dim))
//   head_out = attn_h @ V_h
//   out      = concat_h head_out @ W_o
//
// Difference from MLA: K and V are computed ONCE (not per-head). The SHLA
// forward computes K_h = c_KV @ W_uk_shared and V_h = c_KV @ W_uv_shared ONCE
// and reuses the same tensors across all H heads. In MLA, every head had its
// own (d_c → head_dim) slice of W_uk / W_uv, making K_h and V_h head-specific.
// In SHLA, all heads see the same K, V — only Q varies per head (which is
// what makes multi-head learning possible). The backward accumulates
// d_c_KV from BOTH the K and V chains AND from BOTH heads (since both K and V
// are shared across heads, all heads' gradients flow into W_uk_shared, W_uv_shared,
// and d_c_KV).
//
// Conventions (matches MLA):
//   * Input:  (n, d_model)    — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be evenly divisible by num_heads
//   * head_dim = d_model / num_heads
//   * No bias on any projection (matches MLA/GQA)
//   * No causal mask in v1
//
// Param count breakdown:
//   W_dq:  d * d_c             (shared Q down)
//   W_uq:  d * d_c             (stacked per-head Q up; one (head_dim, d_c) block per head)
//   W_dkv: d * d_c             (shared KV down)
//   W_uk_shared: head_dim * d_c   (single shared K up)
//   W_uv_shared: head_dim * d_c   (single shared V up)
//   W_o:   d * d               (output projection)
//   Total: d*d_c (W_dq) + d*d_c (W_uq, stacked) + d*d_c (W_dkv)
//        + 2 * head_dim * d_c (shared K/V up)
//        + d * d  (W_o)
//        = (3*d + 2*head_dim) * d_c + d * d
//        with head_dim = d / num_heads, equivalent to (4 + H) * d * d_c / H... wait, let
//        me recompute: d * d_c (W_dq) + d * d_c (W_uq stacked) + d * d_c (W_dkv)
//        = 3 * d * d_c for the down-projections
//        + 2 * (d / H) * d_c for the shared K/V up-projections
//        + d * d for W_o
//        Total = (3 + 2/H) * d * d_c + d * d
//        For H=2: (3 + 1) * d * d_c + d * d = 4 * d * d_c + d * d.
//        For H=4: (3 + 0.5) * d * d_c + d * d.
//
// For comparison at iso-rank d_c:
//   MLA: 5 * d * d_c + d * d  (since W_uq + W_uk + W_uv all stacked per-head, that's
//        H * d_c * (d/H) = d*d_c per up-proj, summed to 3*d*d_c; plus 2*d*d_c for
//        W_dq + W_dkv = 5*d*d_c; plus d*d for W_o)
//   MHA: 4 * d * d
//
// SHLA beats MLA when (3 + 2/H)*d*d_c < 5*d*d_c → 2/H < 2 → H > 1 (always for H ≥ 2).
// At H=2, SHLA = 4*d*d_c + d*d vs MLA = 5*d*d_c + d*d, so SHLA saves 1*d*d_c parameters.
//
// ============================================================================

class SHLAAttention : public Layer {
public:
    SHLAAttention(size_t d_model, size_t num_heads, size_t d_c);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_dq; }
    Tensor get_gradients() const override { return grad_W_dq; }
    std::string name() const override { return "SHLAAttention"; }

    // Accessors for tests
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t d_c()       const { return d_c_; }

    // Public for test access / FD checks
    Tensor grad_input() const { return last_d_input_; }
    // Cached K and V tensors (test access). Both are shape (n, head_dim_)
    // and SHARED across all heads.
    const Tensor& get_last_k() const { return last_k_; }
    const Tensor& get_last_v() const { return last_v_; }

    // Public parameters (test access). Names match the math notation.
    // DOWN: shared Q and shared KV, (d_model, d_c) each.
    Tensor W_dq, W_dkv;
    Tensor grad_W_dq, grad_W_dkv;
    // UP: per-head Q (stacked along output dim), shared K, shared V.
    //   W_uq[h, :, :] at [h*head_dim_ : (h+1)*head_dim_, 0 : d_c_]
    //   W_uk_shared: (head_dim_, d_c) — one matrix, applied identically to all heads
    //   W_uv_shared: (head_dim_, d_c) — one matrix, applied identically to all heads
    Tensor W_uq, W_uk_shared, W_uv_shared;
    Tensor grad_W_uq, grad_W_uk_shared, grad_W_uv_shared;
    // OUTPUT: (d_model, d_model)
    Tensor W_o;
    Tensor grad_W_o;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t d_c_;
    double scale_;

    // BPTT cache (filled in forward, consumed in backward)
    Tensor last_input_;        // (n, d_model)  cloned
    Tensor last_c_q_;          // (n, d_c)      shared Q down
    Tensor last_c_kv_;         // (n, d_c)      shared KV down
    Tensor last_q_;            // (n, d_model)  per-head Q stacked
    Tensor last_k_;            // (n, head_dim) SHARED K (one for all heads)
    Tensor last_v_;            // (n, head_dim) SHARED V (one for all heads)
    Tensor last_attn_;         // (num_heads * n, n) softmax probs per Q head
    Tensor last_head_out_;     // (n, d_model)   concat of per-head outputs (pre-W_o)
    Tensor last_d_input_;      // (n, d_model)   cached from last backward
};

// ============================================================================
// SHLABlock — pre-LN → SHLAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

class SHLABlock : public Layer {
public:
    SHLABlock(size_t d_model, size_t num_heads, size_t d_c, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "SHLABlock"; }

    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;             // pre-attn
    SHLAAttention attn_;
    LayerNorm ln2_;             // pre-FFN
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_hidden_;    // PRE-GELU value
    Tensor last_ffn_out_;
    Tensor last_d_input_;
};

// ============================================================================
// SHLAModel — stack of SHLABlocks + classifier head
// ============================================================================

class SHLAModel : public Layer {
public:
    SHLAModel(size_t input_dim, size_t d_model, size_t output_dim,
              size_t num_blocks, size_t num_heads, size_t d_c, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "SHLAModel"; }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    Tensor W_in_;
    Tensor b_in_;
    Tensor W_out_;
    Tensor b_out_;
    Tensor grad_W_in_;
    Tensor grad_b_in_;
    Tensor grad_W_out_;
    Tensor grad_b_out_;
    std::vector<SHLABlock> blocks_;
    Tensor last_input_;      // (n, input_dim) from last forward
    Tensor last_proj_;       // (n, d_model) post input projection (pre-block input)
    Tensor last_block_out_;  // (n, d_model) post last block (pre-output projection)
};

#endif
