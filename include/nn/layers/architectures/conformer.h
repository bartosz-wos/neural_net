#ifndef CONFORMER_H
#define CONFORMER_H

#include "../../core/layer.h"
#include "../../layers/normalization/layer_norm.h"
#include "../../layers/normalization/batch_norm.h"
#include "../../layers/attention/transformer.h"
#include "../../layers/utility/conv1d.h"
#include <vector>
#include <random>

// ============================================================================
// Conformer — Gulati et al. 2020, "Conformer: Convolution-augmented Transformer
// for Speech Recognition" (https://arxiv.org/abs/2010.05656).
//
// The distinctive feature of Conformer is the block sandwich:
//   x -> x + 0.5 * FFN1(LN(x))   # macaron FFN (½)
//   x -> x + MHSA(LN(x))         # multi-head self-attention
//   x -> x + ConvModule(LN(x))   # 1D depthwise conv module
//   x -> x + 0.5 * FFN2(LN(x))   # macaron FFN (½)
//   out = LN(x)
//
// The "macaron" structure (half-step FFNs on both sides of the block) is the
// canonical Conformer choice — it improves over the standard Transformer's
// single full-step FFN. The 1D ConvModule in the middle mixes local context
// (via a depthwise convolution over the time axis), which is what makes
// Conformer good at speech recognition while the attention captures global
// context. The result is used in Whisper's encoder, wav2vec 2.0, USM, and
// nearly every modern production ASR system.
//
// Layout convention used in this implementation: (rows = d_model, cols = seq_len).
// This matches the MultiHeadAttention and (d_model, seq_len) conventions used
// elsewhere in this codebase.
// ============================================================================


// ----------------------------------------------------------------------------
// FeedForward — used twice in each Conformer block.
// Implements FFN(x) = Swish(W1·x + b1) @ W2 + b2
// where W1: dim -> dim*ffn_expansion, W2: dim*ffn_expansion -> dim.
// All weights are stored as raw Tensors (NOT Dense) so we can keep the
// (d_model, seq_len) layout throughout the block — Dense expects (batch, in).
// ----------------------------------------------------------------------------
class FeedForward : public Layer {
public:
    size_t dim_;
    size_t expansion_;
    Tensor W1_;          // (dim*expansion, dim)
    Tensor b1_;          // (1, dim*expansion)
    Tensor W2_;          // (dim, dim*expansion)
    Tensor b2_;          // (1, dim)
    Tensor grad_W1_;
    Tensor grad_b1_;
    Tensor grad_W2_;
    Tensor grad_b2_;

    Tensor last_input_;
    Tensor last_h_pre_;
    Tensor last_h_act_;

    explicit FeedForward(size_t dim, size_t expansion = 4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W1_; }
    Tensor get_gradients() const override { return grad_W1_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FeedForward"; }
};


// ----------------------------------------------------------------------------
// ConvModule — the 1D conv sandwich inside each Conformer block.
// Layout: (d_model, seq_len)
//   1. LayerNorm
//   2. Pointwise expand (linear: dim -> 2*dim)
//   3. GLU: split(2*dim, dim) halves -> a * sigmoid(b)
//   4. Depthwise Conv1D (kernel_size, symmetric padding for same-length)
//   5. BatchNorm (in eval mode -> identity when gamma=1, beta=0)
//   6. Swish (= SiLU)
//   7. Pointwise project (linear: dim -> dim)
//
// Depthwise is implemented as per-channel Conv1D (in_ch == out_ch == dim).
// The depthwise weight matrix is initialized with the identity-like pattern
// (center tap = 1) so the conv stage initially behaves as a passthrough.
// ----------------------------------------------------------------------------
class ConvModule : public Layer {
public:
    LayerNorm ln_;
    Tensor pw_expand_W_;     // (2*dim, dim)
    Tensor pw_expand_b_;     // (1, 2*dim)
    Tensor grad_pw_expand_W_;
    Tensor grad_pw_expand_b_;

    Conv1D dw_conv_;         // (dim, dim, kernel_size, seq_len)

    BatchNorm1D bn_;
    Tensor pw_project_W_;    // (dim, dim)
    Tensor pw_project_b_;    // (1, dim)
    Tensor grad_pw_project_W_;
    Tensor grad_pw_project_b_;

    size_t dim_;
    size_t seq_len_;
    size_t kernel_size_;

    // Cache for backward
    Tensor last_input_;
    Tensor last_ln_out_;
    Tensor last_expand_out_;
    Tensor last_a_;
    Tensor last_b_;
    Tensor last_glu_out_;
    Tensor last_conv_out_;
    Tensor last_bn_out_;
    Tensor last_silu_out_;
    Tensor last_pw_project_in_;
    bool bn_training_;

    ConvModule(size_t dim, size_t seq_len, size_t kernel_size = 15);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return pw_expand_W_; }
    Tensor get_gradients() const override { return grad_pw_expand_W_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ConvModule"; }
    void set_bn_training(bool t) { bn_training_ = t; bn_.set_training(t); }
};


// ----------------------------------------------------------------------------
// ConformerBlock — the canonical sandwich.
// Layout: (d_model, seq_len) -> (d_model, seq_len).
//   x' = x + 0.5 * ffn1(ln_1(x))
//   x'' = x' + mhsa(ln_2(x'))
//   x''' = x'' + conv(ln_3(x''))
//   out = ln_5(x''' + 0.5 * ffn2(ln_4(x''')))
// ----------------------------------------------------------------------------
class ConformerBlock : public Layer {
public:
    size_t dim_;
    size_t seq_len_;
    LayerNorm ln_1_;
    FeedForward ffn1_;
    LayerNorm ln_2_;
    MultiHeadAttention mhsa_;
    LayerNorm ln_3_;
    ConvModule conv_;
    LayerNorm ln_4_;
    FeedForward ffn2_;
    LayerNorm ln_5_;

    // Cache for backward
    Tensor last_input_;
    Tensor last_ln1_out_;
    Tensor last_ffn1_out_;
    Tensor last_r1_;
    Tensor last_ln2_out_;
    Tensor last_mhsa_out_;
    Tensor last_r2_;
    Tensor last_ln3_out_;
    Tensor last_conv_out_;
    Tensor last_r3_;
    Tensor last_ln4_out_;
    Tensor last_ffn2_out_;
    Tensor last_r4_;
    Tensor last_output_;

    ConformerBlock(size_t dim, size_t num_heads, size_t seq_len,
                   size_t ffn_expansion = 4, size_t conv_kernel_size = 15);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ConformerBlock"; }
};


// ----------------------------------------------------------------------------
// ConformerModel — full conformer for sequence classification.
// Input layout: (input_dim, seq_len) — features along rows.
// Forward: input_proj -> [ConformerBlock]*depth -> mean pool over time -> LN -> classifier.
// ----------------------------------------------------------------------------
class ConformerModel : public Layer {
public:
    size_t input_dim_;
    size_t num_classes_;
    size_t dim_;
    size_t depth_;
    size_t num_heads_;
    size_t seq_len_;
    size_t ffn_expansion_;
    size_t conv_kernel_size_;

    // Use raw Tensors for input_proj and classifier to keep (d_model, seq_len) layout
    // for the input projection and (num_classes, 1) layout for the classifier.
    Tensor input_proj_W_;       // (dim, input_dim)
    Tensor input_proj_b_;       // (1, dim)
    Tensor grad_input_proj_W_;
    Tensor grad_input_proj_b_;

    std::vector<ConformerBlock> blocks_;
    LayerNorm ln_out_;

    Tensor classifier_W_;       // (num_classes, dim)
    Tensor classifier_b_;       // (1, num_classes)
    Tensor grad_classifier_W_;
    Tensor grad_classifier_b_;

    // Cache
    Tensor last_input_;
    Tensor last_proj_out_;
    std::vector<Tensor> last_block_outs_;
    Tensor last_pool_;
    Tensor last_ln_out_;
    Tensor last_output_;

    ConformerModel(size_t input_dim, size_t num_classes,
                   size_t dim = 128, size_t depth = 4, size_t num_heads = 4,
                   size_t seq_len = 64, size_t ffn_expansion = 4,
                   size_t conv_kernel_size = 15);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return input_proj_W_; }
    Tensor get_gradients() const override { return grad_input_proj_W_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ConformerModel"; }
};

#endif
