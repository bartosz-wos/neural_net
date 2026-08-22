#ifndef HYMB_H
#define HYMB_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/mamba.h"
#include "../attention/gqa.h"
#include "../../activations/activations.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// Hymba — NVIDIA, 2024
//   "Hymba: A Hybrid Head Architecture for Efficient Language Modeling"
//   https://arxiv.org/abs/2409.18290
//
// A Hymba block is a parallel-composition hybrid of Mamba (selective state-
// space model) and GQA (grouped query attention), with a learnable per-token
// per-channel λ mix that lets the model route each feature to whichever
// branch is more useful.
//
// Per-block forward (input x ∈ R^(T × d_model), T = sequence length):
//
//     ┌── ln_x ──► Mamba(ln_x)  ──────────► mamba_out ──┐
// x ──                                                       ├──⊕─► FFN(LN(·)) ──► + ──► out
//     └── ln_x ──► GQA(Q=ln_x, K=ln_x, V=ln_x) ─► attn ─┘
//                                                          ↑
//                                            mix: gate = σ(W_mix · [mamba_out ‖ attn_out])
//                                            mixed = gate[:,0,:] ⊙ mamba_out + gate[:,1,:] ⊙ attn_out
//
// Where:
//   * ln_x = LayerNorm(x)                                // shared pre-norm
//   * Mamba = MambaBlock(ln_x)                          // SSM path (global, recurrent)
//   * GQA = GQAAttention(ln_x)                          // attention path (exact recall)
//   * mix_proj = Dense(2*d_model, 2*d_model)            // softmax over 2 channels per (token, j)
//   * gate[t, 0, j] = softmax over 2 of [logit_mamba, logit_attn] for (t, j)
//                     initialized to bias toward Mamba at init
//
// Pre-norm residual FFN follows the mix:
//   mixed  = ln_x mix (as above)
//   y      = mixed + ffn2(GELU(ffn1(LayerNorm(mixed))))        // pre-norm residual FFN
//
// All Dense / LayerNorm / Mamba / GQA components already have working
// analytical backward passes.
//
// Conventions:
//   * Input:  (T, d_model)         — T tokens, d_model features
//   * Output: (T, d_model)
//   * d_model must be evenly divisible by num_heads
//   * num_kv_heads <= num_heads   (1 KV head = MQA mode)
//   * ffn_mult > 0 (default 4)
// ============================================================================

class HymbaBlock : public Layer {
public:
    size_t d_model_;
    size_t d_state_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t ffn_dim_;

    // Sub-layers (owned by value)
    LayerNorm ln_;                 // pre-norm (shared between Mamba and GQA paths)
    MambaBlock mamba_;             // SSM path
    GQAAttention attn_;            // attention path (single-block GQA — Q=K=V=ln_x)
    Dense mix_proj_;               // (2*d_model, 2*d_model) WITH bias; bias initialized to [+1, -1]
    LayerNorm ln_ffn_;             // pre-norm for FFN
    Dense ffn1_;                   // (d_model -> ffn_dim)
    Dense ffn2_;                   // (ffn_dim -> d_model)

    // Caches for backward
    Tensor last_input_;            // (T, d_model)    the original input
    Tensor last_ln_x_;             // (T, d_model)    shared pre-norm
    Tensor last_mamba_out_;        // (T, d_model)    Mamba path output
    Tensor last_attn_out_;         // (T, d_model)    GQA path output
    Tensor last_concat_;           // (T, 2*d_model)  [mamba_out ‖ attn_out]
    Tensor last_gate_;             // (T, 2*d_model)  softmax over 2 channels per (token, j)
    Tensor last_mixed_;            // (T, d_model)    weighted sum
    Tensor last_ln_mixed_;         // (T, d_model)    LayerNorm before FFN
    Tensor last_ffn_hidden_;       // (T, ffn_dim)    GELU input (ffn1 output)
    Tensor last_ffn_act_;          // (T, ffn_dim)    GELU(ffn1)
    Tensor last_d_gate_;           // (T, 2*d_model)  cached d_gate for mix backward
    Tensor last_d_mamba_out_;      // (T, d_model)
    Tensor last_d_attn_out_;       // (T, d_model)
    Tensor last_d_ln_x_;           // (T, d_model)

    HymbaBlock(size_t d_model, size_t d_state, size_t num_heads,
               size_t num_kv_heads, size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return ffn2_.get_weights(); }
    Tensor get_gradients() const override { return ffn2_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HymbaBlock"; }

    size_t get_d_model() const { return d_model_; }

    // Accessors for tests
    const Tensor& last_gate() const { return last_gate_; }
    const Tensor& last_mamba_out() const { return last_mamba_out_; }
    const Tensor& last_attn_out() const { return last_attn_out_; }
};

// HymbaModel — stack of HymbaBlocks + input projection + classifier
class HymbaModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t d_state_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t ffn_mult_;

    Dense input_proj_;
    std::vector<std::unique_ptr<HymbaBlock>> blocks_;
    Dense classifier_;

    // Caches
    Tensor last_input_;
    std::vector<Tensor> block_outputs_;  // (num_layers + 2) cached tensors

    HymbaModel(size_t input_dim, size_t d_model, size_t output_dim,
               size_t num_layers, size_t d_state,
               size_t num_heads, size_t num_kv_heads, size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return classifier_.get_weights(); }
    Tensor get_gradients() const override { return classifier_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HymbaModel"; }
};

#endif // HYMB_H
