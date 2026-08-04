#ifndef MLP_MIXER_H
#define MLP_MIXER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../convolutions/conv_layer.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// MLP-Mixer — Tolstikhin et al., 2021, "An all-MLP Architecture for Vision"
// ============================================================================
//
// The paper's central observation is that both ViT and a pure-MLP architecture
// can match state-of-the-art on vision benchmarks without convolutions or
// self-attention. MLP-Mixer is the pure-MLP sibling of ViT.
//
// High-level idea:
//   1. Patch embed the image: (B, C_in, H, W) → (B, S, D) where
//      S = (H/patch)*(W/patch) patches and D is the model dim.
//      We implement patch-embed as a Conv2D with kernel=stride=patch_size.
//   2. For each of `depth` MixerBlocks:
//        a. LayerNorm per-token (over D) → (B, S, D)
//        b. Token-mixing MLP: same Dense applied independently per channel
//           across tokens. Mathematically: reshape to (B, D, S), apply
//           a 2-layer MLP with hidden width token_dim per row (each row is
//           one channel snapshot of length S), reshape back.
//        c. Residual.
//        d. LayerNorm per-channel (over S) → (B, S, D).
//        e. Channel-mixing MLP: same Dense applied independently per token
//           across channels. Mathematically: per row (length D), apply a
//           2-layer MLP with hidden width channel_dim.
//        f. Residual.
//   3. LayerNorm per-token → mean-pool over S → classifier.
//
// Why a single shared Dense for both mixers (broadcast over the orthogonal
// axis) is enough: the token-mix Dense has weights in R^{token_dim × S} and
// is applied to each of the D channel rows independently. The channel-mix
// Dense has weights in R^{channel_dim × D} and is applied to each of the S
// token rows independently. The shared-weight design forces the model to
// learn *axis-general* operations; the paper shows that with enough depth,
// this matches carefully tuned ViTs/Transformers on ImageNet.
//
// Backward: each Mixer's MLP backward is the standard Dense backward applied
// per row with weight-grad accumulation across rows. Two LayerNorms (per-token
// vs per-channel) share the same `LayerNorm` class but operate on different
// axes (D vs S), which we handle by reshaping the input.
//
// References:
//   - Paper: https://arxiv.org/abs/2105.01601
//   - Reference impl: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py
// ============================================================================

class MlpMixerBlock : public Layer {
public:
    // dim:         channel dim (D) — the patch-embedding output dim.
    // seq_len:     number of patches S — patch-embedding output sequence length.
    // token_dim:   hidden width of the token-mixing MLP.
    // channel_dim: hidden width of the channel-mixing MLP.
    //
    // Constraints:
    //   - dim > 0, seq_len > 0
    //   - token_dim > 0, channel_dim > 0
    MlpMixerBlock(size_t dim, size_t seq_len,
                  size_t token_dim, size_t channel_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MlpMixerBlock"; }

    // Public accessors for testing.
    size_t dim() const { return dim_; }
    size_t seq_len() const { return seq_len_; }
    size_t token_dim() const { return token_dim_; }
    size_t channel_dim() const { return channel_dim_; }

    // Public access to LayerNorms for test inspection.
    LayerNorm& ln_token() { return ln_token_; }
    LayerNorm& ln_channel() { return ln_channel_; }

private:
    size_t dim_;
    size_t seq_len_;
    size_t token_dim_;
    size_t channel_dim_;

    // Per-token LayerNorm (over dim)
    LayerNorm ln_token_;
    // Per-channel LayerNorm (over seq_len)
    LayerNorm ln_channel_;

    // Token-mixing MLP — same Dense broadcast over channels.
    // Each row of the per-channel-flattened input has length seq_len; we want
    // to map it through token_dim → seq_len. Dense(in, out) convention:
    //   tok_mlp_w1: in=seq_len, out=token_dim  → weights (token_dim, seq_len)
    //   tok_mlp_w2: in=token_dim, out=seq_len  → weights (seq_len, token_dim)
    Dense tok_mlp_w1_;  // Dense(in=seq_len, out=token_dim)
    Dense tok_mlp_w2_;  // Dense(in=token_dim, out=seq_len)

    // Channel-mixing MLP — same Dense broadcast over tokens.
    // Each row of the per-token-flattened input has length dim; we want to map
    // it through channel_dim → dim. Dense(in, out) convention:
    //   chan_mlp_w1: in=dim, out=channel_dim  → weights (channel_dim, dim)
    //   chan_mlp_w2: in=channel_dim, out=dim  → weights (dim, channel_dim)
    Dense chan_mlp_w1_;  // Dense(in=dim, out=channel_dim)
    Dense chan_mlp_w2_;  // Dense(in=channel_dim, out=dim)

    // Cached state for backward
    Tensor last_input_;         // (B, S*D) raw input to the block
    Tensor last_z_;             // (B, S*D) post per-token LN
    Tensor last_h1_;            // (B*dim, token_dim) — pre-activation of tok_mlp_w1
    Tensor last_h1_gelu_;       // (B*dim, token_dim) — post-GELU pre-activation of tok_mlp_w1
    Tensor last_tok_pre_;       // (B, S*D) post token-mix MLP (before residual)
    Tensor last_y1_;            // (B, S*D) post residual1 (input to LN per-channel)
    Tensor last_v_;             // (B, S*D) post per-channel LN
    Tensor last_h2_;            // (B*S, channel_dim) — pre-activation of chan_mlp_w1
    Tensor last_h2_gelu_;       // (B*S, channel_dim) — post-GELU pre-activation of chan_mlp_w1
    Tensor last_chan_pre_;      // (B, S*D) post channel-mix MLP (before residual)
    Tensor last_output_;        // (B, S*D) final block output (after residual2)
};

// ============================================================================
// MlpMixerModel: patch-embed → stack of MixerBlocks → LN → mean-pool → classifier
// ============================================================================

class MlpMixerModel : public Layer {
public:
    // image_size:   H = W (square images only)
    // patch_size:   patch side (must divide image_size evenly)
    // in_channels:  number of input image channels (3 for RGB)
    // num_classes:  classifier output dim
    // dim:          mixer channel dim (D)
    // depth:        number of MixerBlocks to stack
    // token_dim:    token-mixing MLP hidden width
    // channel_dim:  channel-mixing MLP hidden width
    MlpMixerModel(size_t image_size, size_t patch_size, size_t in_channels,
                  size_t num_classes, size_t dim, size_t depth,
                  size_t token_dim, size_t channel_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MlpMixerModel"; }

    // Public accessors for testing
    size_t num_patches() const { return num_patches_; }
    size_t dim() const { return dim_; }

private:
    size_t image_size_;
    size_t patch_size_;
    size_t in_channels_;
    size_t num_classes_;
    size_t dim_;
    size_t depth_;
    size_t token_dim_;
    size_t channel_dim_;
    size_t num_patches_;  // S = (image_size/patch_size)^2

    // Patch-embedding Conv2D: kernel = stride = patch_size, in_ch = C_in,
    // out_ch = dim. Output spatial: (image_size/patch_size)^2 = num_patches.
    Conv2D patch_embed_;

    // Pre-head LayerNorm (per-token over dim)
    LayerNorm head_ln_;

    // Classifier
    Dense classifier_;

    // Stack of MixerBlocks (owning by value — they have no shared state).
    std::vector<MlpMixerBlock> blocks_;

    // Cached state
    Tensor last_input_;       // (B, C_in*H*W)
    Tensor last_patches_;     // (B, S*dim) post patch-embed
    Tensor last_head_pre_ln_; // (B, S*dim) pre-HeadLN, post-block-stack
    Tensor last_head_ln_;     // (B, S*dim) post head LN
    Tensor last_pooled_;      // (B, dim) post mean-pool
    Tensor last_logits_;      // (B, num_classes) post classifier
};

#endif
