#ifndef JAMBA_H
#define JAMBA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/mamba2.h"
#include "../attention/transformer.h"
#include "../architectures/mixture_of_experts.h"
#include <vector>
#include <memory>

// ============================================================================
// Jamba hybrid block — Lieber et al. 2024, "Jamba: A Hybrid Transformer-Mamba
// Language Model" (https://arxiv.org/abs/2403.19887)
//
// A single Jamba block fuses three sublayers via pre-norm residuals:
//
//   x = x + Mamba2Block(LayerNorm_1(x))                 // SSM path
//   x = x + MultiHeadAttention(LayerNorm_2(x))          // attention path
//   x = x + FFN(LayerNorm_3(x))                         // FFN path
//
// where the FFN is either a top-k Mixture-of-Experts (Jamba paper §2.2) or a
// dense 2-layer Dense MLP (used when num_experts=0). The MoE path is gated
// by `moe_every_n_` — the Jamba paper recommends MoE in every other block
// (moe_every_n=2), but we also support moe_every_n=1 for fully-MoE stacks.
//
// Forward shape: (T, d_model) -> (T, d_model). All three sublayers are
// pre-norm, so the residual stream is what we publish via get_weights()/
// get_gradients() (we proxy to the Mamba-2 layer).
//
// This is the canonical "Jamba block" per the paper. The JambaStack class
// composes N such blocks back-to-back to form a full architecture.
// ===========================================================================

class JambaBlock : public Layer {
public:
    // d_model:        input/output feature dim
    // num_heads:      number of attention heads (must divide d_model)
    // num_experts:    number of MoE FFN experts (0 = use dense FFN)
    // top_k:          top-k experts per token (default 2)
    // d_state:        Mamba-2 d_inner (default = 2 * d_model)
    // moe_every_n:    1 = MoE every block, 2 = MoE every other block
    // use_moe:        if true and num_experts > 0, route through MoE;
    //                  if false, use dense FFN
    JambaBlock(size_t d_model, size_t num_heads,
               size_t num_experts = 8, size_t top_k = 2,
               size_t d_state = 0, size_t moe_every_n = 1,
               bool use_moe = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "JambaBlock"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t num_experts() const { return num_experts_; }
    size_t top_k() const { return top_k_; }
    size_t d_state() const { return d_state_; }
    bool   uses_moe() const { return use_moe_ && num_experts_ > 0; }
    size_t moe_every_n() const { return moe_every_n_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t num_experts_;
    size_t top_k_;
    size_t d_state_;
    size_t moe_every_n_;
    bool   use_moe_;

    LayerNorm ln1_, ln2_, ln3_;
    std::unique_ptr<Mamba2Block> mamba_;
    std::unique_ptr<MultiHeadAttention> attn_;
    std::unique_ptr<MoELayer> moe_ffn_;

    // Dense FFN sublayer (used when MoE is disabled in this block)
    Dense w1_, b1_, w2_, b2_;

    // Forward cache (for backward)
    Tensor last_x_;
    Tensor last_h1_;
    Tensor last_h2_;
    Tensor last_ln1_, last_ln2_, last_ln3_;
    Tensor last_mamba_out_;
    Tensor last_attn_out_;
    Tensor last_ffn_inner_;
    Tensor last_ffn_out_;
};

// JambaStack: sequence of JambaBlocks. The simplest Jamba model.
class JambaStack : public Layer {
public:
    // d_model, num_heads: per-block dims
    // num_layers: number of stacked blocks
    // num_experts, top_k, d_state, moe_every_n: passed to each block
    // every_n_moe: which block indices get MoE (default: every block).
    //   With moe_every_n=2, indices 0,2,4,... get MoE; 1,3,5,... get dense FFN.
    JambaStack(size_t d_model, size_t num_heads, size_t num_layers,
               size_t num_experts = 8, size_t top_k = 2,
               size_t d_state = 0, size_t moe_every_n = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "JambaStack"; }

    size_t num_blocks() const { return blocks_.size(); }
    JambaBlock& block(size_t i) { return *blocks_[i]; }

private:
    std::vector<std::unique_ptr<JambaBlock>> blocks_;
};

#endif // JAMBA_H
