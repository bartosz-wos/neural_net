#ifndef STRIPED_HYENA_H
#define STRIPED_HYENA_H

#include "../../core/layer.h"
#include "../attention/transformer.h"
#include "../normalization/rms_norm.h"
#include "../utility/swiglu.h"
#include "hyena.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// StripedHyena — striped (interleaved) Hyena / attention hybrid
//
//   Poli, Massaro, Muntoni, Singhal, Beigi, et al. 2024.
//   "Mechanistic Design and Scaling of Hybrid Architectures",
//   arXiv:2403.17844v2, §3.3 (Topology) and Appendix C.2.
//
// §3.3: "In this work, we explore sequential striped topologies i.e. where
// different computational primitives are applied sequentially".
//
// Appendix C.2 (StripedHyena), verbatim:
//   "We use 3 striping schedule ratios: 1A:1H, 1A:3H, 1A:11H, where
//    A = Attention and H = Hyena along model depth. In instances where the
//    number of layers is not a multiple of the schedule, the ratio is
//    repeated until the target depth is reached."
//
// So striping is a MODEL-DEPTH concept, not an operator concept. A model of
// depth n lays out a repeating run of `1 attention block : hyena_ratio Hyena
// blocks`, where hyena_ratio ∈ {1, 3, 11} are the paper's three swept
// schedules. Finding 7 (§4.1) is that the compute-optimal attention
// allocation is 25%.
//
// Per block (matching §B.2: "blocks combine a sequence mixing layer with a
// subsequent channel mixing layer"):
//
//   u = x + Mixer(RMSNorm(x))          Mixer = Hyena_N or multi-head attn
//   y = u + SwiGLU(RMSNorm(u))         channel mixer (Appendix C.2: SwiGLU
//                                      for every architecture except Mamba)
//
// The Hyena mixer is the already-shipped `HyenaOperator` (arXiv:2302.10866v3
// Definition 3.1), used verbatim — StripedHyena's contribution is the
// interleaving topology, not a new operator.
//
// v1 scope: the three attention-free-of-MoE block kinds only. The paper's
// MoE variants (StripedHyena-MoE, StripedHyena Experts + MoE) and the
// multi-head Hyena head configuration (MultiHeadHyena) are separate sections
// and separate layers, deliberately not covered here.
//
// All tensors in repo convention: row-major (rows, cols); sequences are
// (seq_len, d_model).
// ============================================================================

// ----------------------------------------------------------------------------
// Free helper: the striping schedule.
//
//   pattern  = [ATTENTION, H, H, ... H]      (1 + hyena_ratio entries)
//   schedule = pattern repeated, truncated to n_layers
//
// Because the pattern BEGINS with an attention block, a depth that is not a
// multiple of (1 + hyena_ratio) still starts with attention, and truncation
// keeps the leading run rather than dropping the tail. The paper's "the ratio
// is repeated until the target depth is reached" is exactly this.
//
// hyena_ratio == 0 is a pure-attention model and is rejected: StripedHyena
// with no Hyena is just a transformer, so the class would be a lie.
// ----------------------------------------------------------------------------
std::vector<bool> build_stripe_schedule(size_t n_layers, size_t hyena_ratio);

// ----------------------------------------------------------------------------
// StripedHyenaBlock: a pre-norm residual block whose sequence mixer is either
// the Hyena operator or multi-head attention, selected at construction.
//
// One class with a discriminator (rather than two block classes) because the
// striping model needs both kinds to live in one flat std::vector, and
// duplicating the residual/FFN wiring for each kind would be the DRY
// violation. `is_attention()` is the discriminator; `is_attention(i)` on the
// model lets tests assert the layout directly instead of inferring it from
// output values.
//
// The mixers are held by unique_ptr — the Model-owns-it form. Declaring one as
// a stack local AND handing its address to an owning container is the
// double-free trap (see the systematic-debugging 5d note).
// ----------------------------------------------------------------------------
class StripedHyenaBlock : public Layer {
public:
    // Public config, named with a trailing underscore so the accessors below
    // can keep the repo's `name()` convention without colliding.
    size_t d_model_, seq_len_, ffn_mult_;

    // --- constructor
    // is_attention: true  -> multi-head attention mixer
    //               false -> Hyena_N mixer
    // ffn_mult:   channel-mixer width multiplier; 0 skips the channel mixer
    //             entirely (used by the FD checks to isolate the mixer chain).
    StripedHyenaBlock(size_t d_model, size_t seq_len, bool is_attention,
                      size_t ffn_mult, size_t num_heads,
                      size_t hyena_order = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "StripedHyenaBlock"; }

    bool is_attention() const { return is_attention_; }
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t ffn_mult() const { return ffn_mult_; }
    // Stable address into the mixer, for tests / introspection.
    HyenaOperator* hyena_mixer() { return hyena_.get(); }
    MultiHeadAttention* attention_mixer() { return attn_.get(); }

private:
    bool is_attention_;
    size_t num_heads_;
    size_t hyena_order_;

    std::unique_ptr<HyenaOperator> hyena_;   // null when is_attention_
    std::unique_ptr<MultiHeadAttention> attn_;  // null when !is_attention_
    std::unique_ptr<RMSNorm> ln1_;
    std::unique_ptr<RMSNorm> ln2_;
    // The channel mixer is a gated FFN: SwiGLU expands d_model -> ffn_hidden
    // and a Dense projects back ffn_hidden -> d_model. The down-projection is
    // REQUIRED, not optional: SwiGLU has no output projection of its own, so
    // without it the channel-mixer output is ffn_mult times too wide and the
    // residual add has nothing to add to. Same shape as MegalodonBlock
    // (ffn_ + ffn_down_) and TransformerBlock.
    std::unique_ptr<SwiGLU<>> ffn_;          // null when ffn_mult_ == 0
    std::unique_ptr<Dense> ffn_down_;        // null when ffn_mult_ == 0

    Tensor last_input_;
    Tensor last_n1_;   // ln1 output, (L, d_model)
    Tensor last_mix_;  // mixer output, (L, d_model)
    Tensor last_u_;    // x + mixer, (L, d_model)
    Tensor last_n2_;   // ln2 output, (L, d_model)
};

// ----------------------------------------------------------------------------
// StripedHyenaModel: input projection -> striping schedule of blocks ->
// final RMSNorm -> mean-pool over the sequence -> classifier.
// ----------------------------------------------------------------------------
class StripedHyenaModel : public Layer {
public:
    // hyena_ratio: the paper's H count per A block (1, 3, or 11 in the paper;
    //              any positive value is accepted).
    StripedHyenaModel(size_t input_dim, size_t d_model, size_t num_layers,
                      size_t output_dim, size_t seq_len, size_t num_heads,
                      size_t hyena_ratio = 3, size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return in_proj_.weights; }
    Tensor get_gradients() const override { return in_proj_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "StripedHyenaModel"; }

    size_t d_model() const { return d_model_; }
    size_t num_layers() const { return num_layers_; }
    size_t hyena_ratio() const { return hyena_ratio_; }
    size_t output_dim() const { return output_dim_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_heads() const { return num_heads_; }
    // The schedule: true = attention block at this depth, false = Hyena block.
    const std::vector<bool>& stripe_schedule() const { return schedule_; }
    bool is_attention(size_t layer) const { return schedule_[layer]; }
    // Number of attention vs Hyena blocks actually realised at this depth.
    size_t num_attention_blocks() const;
    size_t num_hyena_blocks() const;

    std::vector<StripedHyenaBlock> blocks;

private:
    Dense in_proj_;        // (d_model, input_dim)
    std::unique_ptr<RMSNorm> final_norm_;
    Dense classifier_;     // (output_dim, d_model)

    size_t d_model_, num_layers_, output_dim_, seq_len_, num_heads_, hyena_ratio_;
    std::vector<bool> schedule_;

    Tensor last_input_;
    Tensor last_proj_;   // in_proj output, (L, d_model)
    Tensor last_pooled_; // mean-pooled, (1, d_model)
};

#endif
