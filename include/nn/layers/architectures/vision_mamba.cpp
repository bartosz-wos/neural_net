#include "vision_mamba.h"
#include "../../activations/activations.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// PatchEmbed
//
// A wrapper around Conv2D that produces a (N_p, embed_dim) patch sequence
// from a flat (1, in_channels * H * W) image.
//
// The underlying Conv2D produces output of shape (1, embed_dim * N_p) where
// the column indexing is (channel_out, spatial_position) interleaved:
//   column c = out_ch * N_p + spatial_pos   (out_ch ∈ [0, D), spatial_pos ∈ [0, N_p))
//
// To reshape to (N_p, D), we permute the two dims:
//   out[p][d] = conv_out[0][d * N_p + p]
//
// And the inverse permutation for backward.
// ============================================================================

// Constructor is defined inline in the header (see vision_mamba.h).

Tensor PatchEmbed::forward(const Tensor& input) {
    if (input.rows != 1) {
        throw std::invalid_argument("PatchEmbed v1 only supports single-image forward (input.rows must be 1)");
    }
    if (input.cols != static_cast<size_t>(in_channels_) * H_ * W_) {
        throw std::invalid_argument("PatchEmbed: input cols must equal in_channels * H * W");
    }
    last_image_ = input.clone();

    // Run Conv2D: (1, C*H*W) -> (1, D * N_p)
    Tensor conv_out = conv_.forward(input);
    // conv_out shape: (1, D * N_p)
    // Layout: conv_out[0][d * N_p + p]   =   patch p, channel d

    // Permute to (N_p, D): out[p][d] = conv_out[0][d * N_p + p]
    Tensor patches(num_patches_, embed_dim_);
    for (size_t p = 0; p < num_patches_; ++p) {
        for (size_t d = 0; d < embed_dim_; ++d) {
            patches(p, d) = conv_out(0, d * num_patches_ + p);
        }
    }
    return patches;
}

Tensor PatchEmbed::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.rows != num_patches_ || grad_output.cols != embed_dim_) {
        throw std::invalid_argument("PatchEmbed::backward: grad_output shape mismatch");
    }

    // Reshape grad_output (N_p, D) -> (1, D * N_p) (inverse permutation)
    Tensor grad_conv_in(1, embed_dim_ * num_patches_);
    for (size_t p = 0; p < num_patches_; ++p) {
        for (size_t d = 0; d < embed_dim_; ++d) {
            grad_conv_in(0, d * num_patches_ + p) = grad_output(p, d);
        }
    }

    // Backward through Conv2D: returns grad_image (1, C*H*W)
    Tensor grad_image = conv_.backward(grad_conv_in, learning_rate);

    // (Don't apply update_weights here — let the wrapper decide.)
    return grad_image;
}

void PatchEmbed::update_weights(double learning_rate) {
    conv_.update_weights(learning_rate);
}

void PatchEmbed::zero_grad() {
    conv_.zero_grad();
}

std::vector<Tensor*> PatchEmbed::parameters() {
    return conv_.parameters();
}

std::vector<Tensor*> PatchEmbed::gradients() {
    return conv_.gradients();
}

// ============================================================================
// VimBlock
//
// Pre-norm residual block with bidirectional Mamba mixer + FFN.
//
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
// ============================================================================

VimBlock::VimBlock(size_t d_model, size_t d_state, size_t d_inner, size_t ffn_mult)
    : ln1_(d_model),
      ln2_(d_model),
      mamba_fwd_(d_model, d_state, d_inner == 0 ? 2 * d_model : d_inner),
      mamba_bwd_(d_model, d_state, d_inner == 0 ? 2 * d_model : d_inner),
      ffn_proj1_(d_model, d_model * ffn_mult),
      ffn_proj2_(d_model * ffn_mult, d_model),
      d_model_(d_model),
      d_state_(d_state),
      d_inner_(d_inner == 0 ? 2 * d_model : d_inner),
      ffn_mult_(ffn_mult),
      ffn_hidden_(d_model * ffn_mult)
{
    if (d_model == 0 || d_state == 0 || ffn_mult == 0) {
        throw std::invalid_argument("VimBlock: d_model, d_state, ffn_mult must be > 0");
    }
}

Tensor VimBlock::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("VimBlock: input cols must equal d_model");
    }

    size_t T = input.rows;
    last_input = input.clone();

    // 1. Pre-norm mixer input
    last_ln1_out = ln1_.forward(input);

    // 2. Forward Mamba on ln1_out
    last_fwd_out = mamba_fwd_.forward(last_ln1_out);

    // 3. Backward Mamba on reversed ln1_out
    Tensor ln1_reversed(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            ln1_reversed(t, d) = last_ln1_out(T - 1 - t, d);
        }
    }
    Tensor bwd_reversed = mamba_bwd_.forward(ln1_reversed);
    // Reverse back to original order
    last_bwd_out = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            last_bwd_out(t, d) = bwd_reversed(T - 1 - t, d);
        }
    }

    // 4. Mixer = fwd + bwd
    Tensor mixer(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            mixer(t, d) = last_fwd_out(t, d) + last_bwd_out(t, d);
        }
    }

    // 5. Residual 1: x + mixer
    last_res1 = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            last_res1(t, d) = input(t, d) + mixer(t, d);
        }
    }

    // 6. Pre-norm FFN input
    last_ln2_out = ln2_.forward(last_res1);

    // 7. FFN up + GELU + down
    last_ffn_hidden = ffn_proj1_.forward(last_ln2_out);
    last_ffn_act = last_ffn_hidden.apply(GELU{});
    last_ffn_out = ffn_proj2_.forward(last_ffn_act);

    // 8. Residual 2: res1 + ffn_out
    Tensor output(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            output(t, d) = last_res1(t, d) + last_ffn_out(t, d);
        }
    }
    return output;
}

Tensor VimBlock::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.rows != last_input.rows || grad_output.cols != d_model_) {
        throw std::invalid_argument("VimBlock::backward: grad_output shape mismatch");
    }
    size_t T = last_input.rows;

    // ===== Residual 2 split: out = res1 + ffn_out =====
    // grad_res1_from_ffn = grad_output (initially), grad_res1 += grad_output
    Tensor grad_res1(grad_output);

    // FFN backward chain
    Tensor grad_ffn_act = ffn_proj2_.backward(grad_output, learning_rate);
    // d/dx gelu(x) = sigmoid-style: gelu_deriv. We use activations.h GELU.
    Tensor grad_ffn_hidden(T, ffn_hidden_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < ffn_hidden_; ++k) {
            grad_ffn_hidden(t, k) = grad_ffn_act(t, k) * GELU{}.derivative(last_ffn_hidden(t, k));
        }
    }
    Tensor grad_ln2_out = ffn_proj1_.backward(grad_ffn_hidden, learning_rate);
    Tensor grad_res1_from_ffn = ln2_.backward(grad_ln2_out, learning_rate);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_res1(t, d) += grad_res1_from_ffn(t, d);
        }
    }

    // ===== Residual 1 split: res1 = x + mixer =====
    // grad_mixer = grad_res1 (the path through mixer)
    // grad_x_from_mixer added later after we compute grad through ln1
    Tensor grad_mixer(grad_res1);

    // Forward Mamba backward: takes grad_mixer (T, D), returns grad on ln1_out
    Tensor grad_ln1_from_fwd = mamba_fwd_.backward(grad_mixer, learning_rate);

    // Backward Mamba backward: needs grad on bwd-Mamba output in REVERSED order
    Tensor grad_mixer_reversed(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_mixer_reversed(t, d) = grad_mixer(T - 1 - t, d);
        }
    }
    Tensor grad_ln1_reversed = mamba_bwd_.backward(grad_mixer_reversed, learning_rate);
    // Reverse back to original order
    Tensor grad_ln1_from_bwd(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_ln1_from_bwd(t, d) = grad_ln1_reversed(T - 1 - t, d);
        }
    }

    // grad on ln1_out (input to both mamba branches)
    Tensor grad_ln1(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_ln1(t, d) = grad_ln1_from_fwd(t, d) + grad_ln1_from_bwd(t, d);
        }
    }
    // Backward through ln1: returns grad on x (input to block)
    Tensor grad_input_from_mixer = ln1_.backward(grad_ln1, learning_rate);

    // Final grad_input = grad_res1 (the residual path bypassing the mixer) + grad_input_from_mixer
    Tensor grad_input(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_input(t, d) = grad_res1(t, d) + grad_input_from_mixer(t, d);
        }
    }
    return grad_input;
}

void VimBlock::update_weights(double learning_rate) {
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    mamba_fwd_.update_weights(learning_rate);
    mamba_bwd_.update_weights(learning_rate);
    ffn_proj1_.update_weights(learning_rate);
    ffn_proj2_.update_weights(learning_rate);
}

void VimBlock::zero_grad() {
    ln1_.zero_grad();
    ln2_.zero_grad();
    mamba_fwd_.zero_grad();
    mamba_bwd_.zero_grad();
    ffn_proj1_.zero_grad();
    ffn_proj2_.zero_grad();
}

std::vector<Tensor*> VimBlock::parameters() {
    auto p = ln1_.parameters();
    auto p2 = ln2_.parameters();
    auto p3 = mamba_fwd_.parameters();
    auto p4 = mamba_bwd_.parameters();
    auto p5 = ffn_proj1_.parameters();
    auto p6 = ffn_proj2_.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    p.insert(p.end(), p5.begin(), p5.end());
    p.insert(p.end(), p6.begin(), p6.end());
    return p;
}

std::vector<Tensor*> VimBlock::gradients() {
    auto g = ln1_.gradients();
    auto g2 = ln2_.gradients();
    auto g3 = mamba_fwd_.gradients();
    auto g4 = mamba_bwd_.gradients();
    auto g5 = ffn_proj1_.gradients();
    auto g6 = ffn_proj2_.gradients();
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), g3.begin(), g3.end());
    g.insert(g.end(), g4.begin(), g4.end());
    g.insert(g.end(), g5.begin(), g5.end());
    g.insert(g.end(), g6.begin(), g6.end());
    return g;
}

// ============================================================================
// VimModel
//
// Full vision-mamba image classifier. Single-image forward (N=1):
//   patches = patch_embed_(image)                          (N_p, D)
//   seq[0]  = class_token_                                  (1, D)
//   seq[1:] = patches                                       (N_p, D)
//   seq    += pos_embed_                                    (1+N_p, D)
//   for each block: x = blocks_[i]->forward(x)
//   x      = final_ln_(x)
//   cls    = x[0]
//   logits = classifier_(cls)
// ============================================================================

VimModel::VimModel(size_t C_in, size_t H, size_t W, size_t patch_size,
                   size_t D, size_t num_classes, size_t d_state,
                   size_t d_inner, size_t num_layers, size_t ffn_mult)
    : patch_embed_(C_in, D, H, W, patch_size),
      class_token_(1, D),
      pos_embed_(1 + (H / patch_size) * (W / patch_size), D),
      grad_class_token_(1, D),
      grad_pos_embed_(1 + (H / patch_size) * (W / patch_size), D),
      final_ln_(D),
      classifier_(D, num_classes),
      C_in_(C_in), H_(H), W_(W), patch_size_(patch_size),
      D_(D), num_classes_(num_classes), d_state_(d_state),
      num_layers_(num_layers),
      num_patches_((H / patch_size) * (W / patch_size))
{
    if (C_in == 0 || D == 0 || num_classes == 0 || d_state == 0 || num_layers == 0) {
        throw std::invalid_argument("VimModel: C_in, D, num_classes, d_state, num_layers must be > 0");
    }
    if (d_inner != 0 && d_inner < 2 * D) {
        // Not a hard requirement, but warn-style; allow smaller to keep tests tractable.
    }

    // Random init for class_token and pos_embed (small scale)
    std::mt19937 gen(123);
    std::normal_distribution<double> nd(0.0, 0.02);
    for (auto& v : class_token_.data) v = nd(gen);
    for (auto& v : pos_embed_.data)   v = nd(gen);

    // Build blocks
    for (size_t i = 0; i < num_layers_; ++i) {
        size_t actual_d_inner = (d_inner == 0) ? 2 * D : d_inner;
        blocks_.push_back(std::make_unique<VimBlock>(D, d_state, actual_d_inner, ffn_mult));
    }
}

Tensor VimModel::forward(const Tensor& input) {
    if (input.rows != 1) {
        throw std::invalid_argument("VimModel v1 only supports single-image forward (input.rows must be 1)");
    }
    if (input.cols != C_in_ * H_ * W_) {
        throw std::invalid_argument("VimModel: input cols must equal C_in * H * W");
    }
    last_image_ = input.clone();

    // 1. Patch embed -> (N_p, D)
    last_patches_ = patch_embed_.forward(input);

    // 2. Concat class_token + patches -> (1 + N_p, D)
    size_t seq_len = 1 + num_patches_;
    Tensor seq(seq_len, D_);
    for (size_t d = 0; d < D_; ++d) {
        seq(0, d) = class_token_(0, d);
    }
    for (size_t p = 0; p < num_patches_; ++p) {
        for (size_t d = 0; d < D_; ++d) {
            seq(1 + p, d) = last_patches_(p, d);
        }
    }
    // Add position embedding
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < D_; ++d) {
            seq(i, d) += pos_embed_(i, d);
        }
    }
    last_seq_ = seq;

    // 3. Run blocks
    Tensor x = seq;
    last_block_outs_.clear();
    for (size_t i = 0; i < num_layers_; ++i) {
        x = blocks_[i]->forward(x);
        last_block_outs_.push_back(x);
    }

    // 4. Final LayerNorm
    x = final_ln_.forward(x);

    // 5. Take class token (row 0)
    Tensor cls(1, D_);
    for (size_t d = 0; d < D_; ++d) {
        cls(0, d) = x(0, d);
    }

    // 6. Classifier
    Tensor logits = classifier_.forward(cls);
    return logits;
}

Tensor VimModel::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.rows != 1 || grad_output.cols != num_classes_) {
        throw std::invalid_argument("VimModel::backward: grad_output shape mismatch");
    }

    // 1. Backward through classifier
    Tensor grad_cls = classifier_.backward(grad_output, learning_rate);

    // 2. Build full-sequence gradient (1+N_p, D) with row 0 = grad_cls, rest = 0
    size_t seq_len = 1 + num_patches_;
    Tensor grad_seq(seq_len, D_);
    grad_seq.fill(0.0);
    for (size_t d = 0; d < D_; ++d) {
        grad_seq(0, d) = grad_cls(0, d);
    }

    // 3. Backward through final_ln_
    Tensor grad_after_ln = final_ln_.backward(grad_seq, learning_rate);

    // 4. Backward through blocks (reverse order)
    Tensor grad = grad_after_ln;
    for (size_t i = num_layers_; i > 0; --i) {
        grad = blocks_[i - 1]->backward(grad, learning_rate);
    }

    // 5. Add position embedding gradient: pos_embed contributes via add, so
    //    grad_pos_embed += grad. And class_token gradient: same row 0.
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < D_; ++d) {
            grad_pos_embed_(i, d) += grad(i, d);
        }
    }
    for (size_t d = 0; d < D_; ++d) {
        grad_class_token_(0, d) += grad(0, d);
    }

    // 6. Extract patches-only gradient (skip class token row)
    Tensor grad_patches(num_patches_, D_);
    for (size_t p = 0; p < num_patches_; ++p) {
        for (size_t d = 0; d < D_; ++d) {
            grad_patches(p, d) = grad(1 + p, d);
        }
    }

    // 7. Backward through patch_embed (returns grad_image)
    Tensor grad_image = patch_embed_.backward(grad_patches, learning_rate);
    return grad_image;
}

void VimModel::update_weights(double learning_rate) {
    // pos_embed & class_token: simple SGD (we maintain grad tensors, not in a wrapper)
    for (size_t i = 0; i < class_token_.data.size(); ++i) {
        class_token_.data[i] -= learning_rate * grad_class_token_.data[i];
    }
    grad_class_token_.fill(0.0);
    for (size_t i = 0; i < pos_embed_.data.size(); ++i) {
        pos_embed_.data[i] -= learning_rate * grad_pos_embed_.data[i];
    }
    grad_pos_embed_.fill(0.0);

    patch_embed_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void VimModel::zero_grad() {
    grad_class_token_.fill(0.0);
    grad_pos_embed_.fill(0.0);
    patch_embed_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> VimModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&class_token_);
    p.push_back(&pos_embed_);
    auto pe_params = patch_embed_.parameters();
    p.insert(p.end(), pe_params.begin(), pe_params.end());
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fln = final_ln_.parameters();
    p.insert(p.end(), fln.begin(), fln.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> VimModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_class_token_);
    g.push_back(&grad_pos_embed_);
    auto pe_grads = patch_embed_.gradients();
    g.insert(g.end(), pe_grads.begin(), pe_grads.end());
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fln = final_ln_.gradients();
    g.insert(g.end(), fln.begin(), fln.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}