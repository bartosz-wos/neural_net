#ifndef GAU_H
#define GAU_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// GAU — Gated Attention Unit
//   Hua et al. 2022, "Transformer Quality in Linear Time"
//   https://arxiv.org/abs/2206.03637
//
// ============================================================================
//
// GAU replaces softmax attention with a **gated element-wise** attention
// mechanism. The single-head formulation we implement:
//
//   Q, K, V, U = X @ W_q, X @ W_k, X @ W_v, X @ W_u          (n, d_model) each
//
//   A_t[d]  = sum_{s≤t} γ_{t,s} * V_s[d]       (per-channel "value" sum)
//   B_t     = sum_{s≤t} γ_{t,s}                  (per-token normalizer)
//   O_t[d]  = U_t[d] * A_t[d] / (B_t + eps)       (gated elementwise output)
//
//   Y = O @ W_o^T + b_o                          (n, d_out)
//
// γ ∈ R^{n×n} is a learnable ADDITIVE position bias (shared across channels).
// eps is a small constant (1e-6) for numerical stability.
//
// This is the simplest tractable variant of GAU (single-head, identity φ):
// the paper's "GAUα" with learnable per-channel gating from U is exactly
// the "output gate" pattern that gives GAU its quality.
//
// ----------------------------------------------------------------------------
// Why GAU over softmax attention:
//   * No softmax → numerically simpler, no exp/normalize stability issues
//   * Output gate U_t⊙. provides per-channel "soft routing" without softmax
//   * ~50% fewer parameters than a standard MHA block (no QK^T head split)
//   * Linear in n per layer (after materializing γ)
// ----------------------------------------------------------------------------
//
// Backward derivation (per-t, with grad_O[t] = upstream gradient at the
// pre-W_o output):
//
//   O_t[d]  = U_t[d] * A_t[d] / (B_t + eps)
//   A_t[d]  = sum_{s≤t} γ_{t,s} * V_s[d]
//   B_t     = sum_{s≤t} γ_{t,s}
//
//   grad_U[t,d] += grad_O[t,d] * A_t[d] / (B_t + eps)
//   let dA_d     = grad_O[t,d] * U_t[d] / (B_t + eps)       (per-channel scalar)
//   let dB       = -sum_d grad_O[t,d] * U_t[d] * A_t[d] / (B_t + eps)^2
//
//   For each s ≤ t:
//     grad_γ[t,s] += sum_d dA_d * V_s[d] + dB           // (sum over channels + the dB lumped term)
//     grad_V[s,d] += dA_d * γ_{t,s} / (B_t + eps)        // V_s contributes to A_t
//   (and grad_K, grad_Q are zero since the formula above doesn't use them
//    — we still allocate W_q, W_k for API symmetry / future φ extensions;
//    in the simplest GAU they contribute zero gradient.)
//
// ----------------------------------------------------------------------------
// Conventions (match the rest of the attention/ directory):
//   * Input/Output: (n, d_model) — n tokens, d_model features
//   * W_q, W_k, W_v, W_u are raw Tensor weights (out_features, in_features)
//     shaped (d_model, d_model). The forward computes Q = X @ W_q^T manually
//     (no bias), matching how Based stores W_q.
//   * W_o is (d_out, d_model). Default d_out = d_model.
//   * γ is (max_seq_len, max_seq_len) initialized small random.
//   * Block: pre-LN -> GAU -> residual -> pre-LN -> GELU FFN -> residual
//   * Model: stack of `num_blocks` GAUBlocks + per-token classifier
// ----------------------------------------------------------------------------

class GAUAttention : public Layer {
public:
    // d_model: input/output feature dim (single-head v1)
    // max_seq_len: maximum number of tokens; position bias is (max_seq_len, max_seq_len)
    // d_out: output feature dim (default 0 -> d_model)
    GAUAttention(size_t d_model, size_t max_seq_len, size_t d_out = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "GAUAttention"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t max_seq_len() const { return max_seq_len_; }
    size_t d_out() const { return d_out_; }
    const Tensor& position_bias() const { return position_bias_; }
    const Tensor& last_A() const { return cache_A_; }
    const Tensor& last_B() const { return cache_B_; }
    const Tensor& last_O() const { return cache_O_; }
    const Tensor& last_input() const { return last_input_; }

    // Public parameters (raw Tensors for direct FD-style gradient checks)
    Tensor W_q;            // (d_model, d_model)
    Tensor W_k;            // (d_model, d_model)
    Tensor W_v;            // (d_model, d_model)
    Tensor W_u;            // (d_model, d_model)
    Tensor W_o;            // (d_out, d_model)
    Tensor b_o;            // (1, d_out)
    Tensor position_bias_;// (max_seq_len, max_seq_len) — learnable scalar position bias
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_u, grad_W_o, grad_b_o;
    Tensor grad_position_bias_;

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t d_out_;

    // Forward caches
    Tensor last_input_;
    Tensor last_Q_, last_K_, last_V_, last_U_;
    Tensor cache_A_;           // (n, d_model)
    Tensor cache_B_;           // (n, 1)
    Tensor cache_O_;           // (n, d_model) — U ⊙ A / (B + eps)
    Tensor last_output_pre_wo_; // (n, d_model) — input to W_o (== cache_O_)
};

// ----------------------------------------------------------------------------
// GAUBlock — pre-LN -> GAUAttention -> residual -> pre-LN -> GELU FFN -> residual
// ----------------------------------------------------------------------------
class GAUBlock : public Layer {
public:
    GAUBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim = 0, size_t d_out = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return attn.W_q; }
    Tensor get_gradients() const override { return attn.grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "GAUBlock"; }

    GAUAttention attn;
    LayerNorm ln1;        // (d_model,)
    LayerNorm ln2;        // (d_model,)
    Tensor W_ffn1_;       // (d_model, ffn_dim)
    Tensor W_ffn2_;       // (ffn_dim, d_model)
    Tensor b_ffn1_;       // (1, ffn_dim)
    Tensor b_ffn2_;       // (1, d_model)
    Tensor grad_W_ffn1_, grad_W_ffn2_, grad_b_ffn1_, grad_b_ffn2_;

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;       // block input
    Tensor last_z1_;      // ln1(x)
    Tensor last_attn_out_;// GAU output (pre-residual)
    Tensor last_res1_;    // z1 + attn_out
    Tensor last_z2_;      // ln2(res1)
    Tensor last_h_pre_;   // W_ffn1 · z2 + b_ffn1
    Tensor last_h_act_;   // GELU(h_pre)
    Tensor last_ffn_out_; // W_ffn2 · h_act + b_ffn2
};

// ----------------------------------------------------------------------------
// GAUModel — stack of GAUBlocks + per-token classifier
// ----------------------------------------------------------------------------
class GAUModel : public Layer {
public:
    GAUModel(size_t d_model, size_t max_seq_len, size_t out_features,
             size_t num_blocks, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return classifier_W_; }
    Tensor get_gradients() const override { return grad_classifier_W_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "GAUModel"; }

    std::vector<std::unique_ptr<GAUBlock>> blocks;
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
    Tensor last_block_output_; // output of the last block (input to classifier)
};

#endif // GAU_H