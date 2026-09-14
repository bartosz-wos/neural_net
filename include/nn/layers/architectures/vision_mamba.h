#ifndef VISION_MAMBA_H
#define VISION_MAMBA_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../convolutions/conv_layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/mamba.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// Vision Mamba (Vim) — Zhu et al., ICLR 2024
//   "Vision Mamba: Efficient Visual Representation Learning with Bidirectional
//    State Space Models" (https://arxiv.org/abs/2401.09417)
//
// The Mamba-based replacement for ViT. Replaces the quadratic-cost
// self-attention mixer in ViT with a BIDIRECTIONAL state-space mixer:
// two parallel MambaBlocks scan the patch sequence (one forward, one
// backward), and their outputs are summed. Everything else (patch embedding
// via non-overlapping Conv2D, learnable class token + position embedding,
// FFN, pre-norm residual) follows the ViT recipe.
//
// Pipeline per image (single-image forward, N=1):
//   patches = PatchEmbed(image)                            (N_p, D)
//   cls_tok = class_token_                                 (1, D)   (learnable)
//   seq     = concat(cls_tok, patches) + pos_embed_        (1+N_p, D)
//   for each VimBlock: x = block(x)                        (1+N_p, D)
//   x       = final_ln_(x)                                (1+N_p, D)
//   cls     = x[0]                                          (1, D)
//   logits  = classifier_(cls)                             (1, num_classes)
//
// VimBlock structure (pre-norm residual, like ViT-with-pre-LN):
//   ln1_out   = ln1_(x)                                    (T, D)
//   fwd_out   = mamba_fwd_(ln1_out)                        (T, D)
//   bwd_out   = reverse(mamba_bwd_(reverse(ln1_out)))      (T, D)
//   mixer     = fwd_out + bwd_out                          (T, D)
//   res1      = x + mixer                                  (T, D)
//   ln2_out   = ln2_(res1)                                 (T, D)
//   ffn_h     = ffn_proj1_(ln2_out)                        (T, D*ffn_mult)
//   ffn_act   = gelu(ffn_h)                                (T, D*ffn_mult)
//   ffn_out   = ffn_proj2_(ffn_act)                        (T, D)
//   out       = res1 + ffn_out                             (T, D)
//
// Note: the two MambaBlocks are independent (their weights are NOT tied).
// Paper §3.1: "we use two independent SSM blocks for forward and backward
// scanning, and sum their outputs".
// ============================================================================

class PatchEmbed : public Layer {
public:
    // Static validation. Throws std::invalid_argument for any invalid arg.
    // Use this BEFORE constructing PatchEmbed if you need to validate args
    // without triggering Conv2D's stride=0 FPE in the initializer list.
    static void validate(size_t in_channels, size_t embed_dim,
                         size_t H, size_t W, size_t patch_size) {
        if (in_channels == 0 || embed_dim == 0) {
            throw std::invalid_argument("PatchEmbed: in_channels and embed_dim must be > 0");
        }
        if (H == 0 || W == 0) {
            throw std::invalid_argument("PatchEmbed: H and W must be > 0");
        }
        if (patch_size == 0) {
            throw std::invalid_argument("PatchEmbed: patch_size must be > 0");
        }
        if (H % patch_size != 0 || W % patch_size != 0) {
            throw std::invalid_argument("PatchEmbed: H and W must be divisible by patch_size");
        }
    }

    PatchEmbed(size_t in_channels, size_t embed_dim, size_t H, size_t W, size_t patch_size)
        : conv_(static_cast<int>(in_channels), static_cast<int>(embed_dim),
                static_cast<int>(patch_size), static_cast<int>(patch_size),
                static_cast<int>(H), static_cast<int>(W),
                /*stride_h=*/static_cast<int>(patch_size),
                /*stride_w=*/static_cast<int>(patch_size),
                /*pad_h=*/0, /*pad_w=*/0),
          in_channels_(in_channels), embed_dim_(embed_dim),
          H_(H), W_(W), patch_size_(patch_size),
          num_patches_(0)
    {
        // Validate in body. For invalid args that would cause Conv2D to FPE
        // (e.g. patch_size=0), the FPE happens in conv_'s initializer BEFORE
        // we reach this body. Callers must call validate(...) first if they
        // need to check args without triggering the FPE. This is acceptable
        // because patch_size=0 / H=0 are clear programmer errors.
        validate(in_channels, embed_dim, H, W, patch_size);
        num_patches_ = (H / patch_size) * (W / patch_size);
    }

    // Forward: (1, in_channels * H * W)  ->  (N_p, embed_dim)
    //   where N_p = (H/patch_size) * (W/patch_size)
    Tensor forward(const Tensor& input) override;
    // Backward: grad_output is (N_p, embed_dim); returns (1, in_channels * H * W)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return conv_.get_weights(); }
    Tensor get_gradients() const override { return conv_.get_gradients(); }
    std::string name() const override { return "PatchEmbed"; }

    // Accessors
    size_t in_channels() const { return in_channels_; }
    size_t embed_dim() const { return embed_dim_; }
    size_t H() const { return H_; }
    size_t W() const { return W_; }
    size_t patch_size() const { return patch_size_; }
    size_t num_patches() const { return num_patches_; }

    // Internal — public for test introspection
    Conv2D conv_;

private:
    size_t in_channels_;
    size_t embed_dim_;
    size_t H_;
    size_t W_;
    size_t patch_size_;
    size_t num_patches_;

    // Forward cache
    Tensor last_image_;             // (1, in_channels * H * W)
};

class VimBlock : public Layer {
public:
    // d_model:  token feature dim
    // d_state:  Mamba SSM state dim
    // d_inner:  Mamba inner dim (default = 2 * d_model, matching Mamba-1 convention)
    // ffn_mult: FFN expansion factor
    VimBlock(size_t d_model, size_t d_state, size_t d_inner = 0, size_t ffn_mult = 4);

    // Forward: (T, d_model)  ->  (T, d_model)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ln1_.gamma; }
    Tensor get_gradients() const override { return ln1_.grad_gamma_; }
    std::string name() const override { return "VimBlock"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }
    size_t d_inner() const { return d_inner_; }
    size_t ffn_mult() const { return ffn_mult_; }
    size_t ffn_hidden() const { return ffn_hidden_; }

    // Public sublayers (for gradient inspection / mutation tests)
    LayerNorm ln1_;
    LayerNorm ln2_;
    MambaBlock mamba_fwd_;
    MambaBlock mamba_bwd_;
    Dense ffn_proj1_;
    Dense ffn_proj2_;

    // Forward caches (public for tests)
    Tensor last_input;          // (T, d_model)
    Tensor last_ln1_out;        // (T, d_model)
    Tensor last_fwd_out;        // (T, d_model)
    Tensor last_bwd_out;        // (T, d_model)
    Tensor last_res1;           // (T, d_model)
    Tensor last_ln2_out;        // (T, d_model)
    Tensor last_ffn_hidden;     // (T, ffn_hidden)
    Tensor last_ffn_act;        // (T, ffn_hidden) — gelu
    Tensor last_ffn_out;        // (T, d_model)

private:
    size_t d_model_;
    size_t d_state_;
    size_t d_inner_;
    size_t ffn_mult_;
    size_t ffn_hidden_;
};

class VimModel : public Layer {
public:
    // C_in:      input image channels
    // H, W:      input image spatial dims
    // patch_size: non-overlapping patch size
    // D:         token / model dim
    // num_classes
    // d_state:   Mamba SSM state dim
    // d_inner:   Mamba inner dim (default 2*D)
    // num_layers: number of VimBlocks
    // ffn_mult:  FFN expansion factor
    VimModel(size_t C_in, size_t H, size_t W, size_t patch_size,
             size_t D, size_t num_classes, size_t d_state,
             size_t d_inner = 0, size_t num_layers = 2, size_t ffn_mult = 4);

    // Forward: (1, C_in * H * W)  ->  (1, num_classes)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return pos_embed_; }
    Tensor get_gradients() const override { return grad_pos_embed_; }
    std::string name() const override { return "VimModel"; }

    // Accessors
    size_t C_in() const { return C_in_; }
    size_t H() const { return H_; }
    size_t W() const { return W_; }
    size_t patch_size() const { return patch_size_; }
    size_t D() const { return D_; }
    size_t num_classes() const { return num_classes_; }
    size_t d_state() const { return d_state_; }
    size_t num_layers() const { return num_layers_; }
    size_t num_patches() const { return num_patches_; }

    // Public for tests / mutation
    PatchEmbed patch_embed_;
    Tensor class_token_;               // (1, D) — learnable
    Tensor pos_embed_;                 // (1 + N_p, D) — learnable
    Tensor grad_class_token_;          // (1, D)
    Tensor grad_pos_embed_;            // (1 + N_p, D)
    std::vector<std::unique_ptr<VimBlock>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;                 // (D, num_classes)

    // Forward caches
    Tensor last_image_;                // (1, C_in * H * W)
    Tensor last_patches_;              // (N_p, D)
    Tensor last_seq_;                  // (1 + N_p, D)   class token + patches + pos embed
    std::vector<Tensor> last_block_outs_;  // cached block outputs (kept for backward)

private:
    size_t C_in_;
    size_t H_;
    size_t W_;
    size_t patch_size_;
    size_t D_;
    size_t num_classes_;
    size_t d_state_;
    size_t num_layers_;
    size_t num_patches_;
};

#endif // VISION_MAMBA_H