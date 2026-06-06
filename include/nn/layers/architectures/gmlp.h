#ifndef GMLP_H
#define GMLP_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// gMLP — Liu, Dai, Lu, Le, Welling, "Pay Attention to MLPs" (NeurIPS 2021)
// ============================================================================
//
// A pure-MLP architecture that, for the first time, demonstrates that an
// all-MLP model can match (or beat) carefully tuned Transformers on several
// key language and vision benchmarks. The central idea is a SPATIAL GATING
// UNIT (SGU) that lets adjacent tokens communicate with each other — replacing
// self-attention with a fixed-size learnable linear map over the token axis.
//
// Why a learnable (S, S) spatial matrix works at all is non-obvious: the
// gradient signal from the gating operation forces W to specialize per-token
// mixing patterns. With enough depth and a large enough hidden dim, this
// competes with attention despite not having a softmax(QK^T)V dynamic kernel.
//
// Block (one gMLPBlock, applied to (S, d) inputs):
//   z       = LayerNorm(x)                          # (S, d)
//   u       = fc_in(z)                              # (S, 2d)  Dense(2d, d)
//   u1, u2  = chunk_last_axis(u, 2)                # (S, d) each
//   v       = u1 * (W_spatial @ GELU(u2))           # W_spatial: (S, S), v: (S, d)
//   v       = v + b_sgu                              # optional bias
//   y       = fc_out(v)                             # (S, d)   Dense(d, d)
//   out     = x + alpha * y                          # alpha: learnable scalar, init 0.01
//
// Notes:
//   * The default W_spatial is initialized to a near-zero (small uniform) tensor
//     so the block starts as approximately an identity on the first forward
//     pass. This is similar to the inits used in the paper.
//   * alpha is a learnable scalar that lets the entire block be softly
//     "turned off" at init — the gMLP paper found this critical for stable
//     training of deep stacks.
//   * We use GELU for the activation (matches the original paper; the authors
//     also experimented with ReLU and GeLU-Tanh). GELU is already defined
//     in activations.h.
//   * No attention, no convolution over tokens, no positional encoding — just
//     two Dense layers per block plus a fixed-size token-mixing matrix.
//
// gMLPModel stacks `num_blocks` gMLPBlocks with a final classifier head:
//
//   x_in   = (S, d_model) input
//   x      = gMLPBlock[0](x_in)
//   x      = gMLPBlock[1](x)
//   ...
//   x      = gMLPBlock[num_blocks-1](x)
//   logits = classifier(x)                         # Dense(out_features, d_model)
//
// Forward and backward both expect (S, d) shaped tensors where S is the
// sequence length and d is the model dimension. The (S, S) spatial matrix
// means gMLP is fundamentally a sequence-to-sequence architecture (S must be
// fixed across calls in a given model instance).
// ============================================================================

class gMLPBlock : public Layer {
public:
    // dim:          model dimension d
    // seq_len:      token dimension S — sets the size of W_spatial (S, S)
    // ff_mult:      expansion factor for the channel-wise Dense (default 4 —
    //                the paper uses 2x in the block so 2d comes from chunking).
    //                We keep 2x for parity with the paper's gMLP block; the
    //                hidden dim of fc_in is therefore 2*dim.
    // alpha_init:   initial value for the learnable scalar alpha (default 0.01
    //                matches the paper)
    // sgu_bias:     whether to add a bias after the SGU elementwise gate
    gMLPBlock(size_t dim, size_t seq_len,
              double alpha_init = 0.01, bool sgu_bias = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "gMLPBlock"; }

private:
    size_t dim_;
    size_t seq_len_;

    LayerNorm norm_;     // (dim) — pre-norm
    Dense fc_in_;        // (2*dim, dim) — channel expansion via Dense (W stored as (2d, d))
    Dense fc_out_;       // (dim, dim)   — channel projection
    Tensor W_spatial_;   // (S, S) — token-mixing linear map
    Tensor b_sgu_;       // (1, dim) — bias after SGU (optional)
    Tensor alpha_;       // (1, 1) — learnable scalar residual weight
    bool sgu_bias_;

    // Gradients
    Tensor grad_W_spatial_;
    Tensor grad_b_sgu_;
    Tensor grad_alpha_;

    // Cached state for backward
    Tensor last_input_;        // (S, d)
    Tensor last_z_;            // (S, d)  LN output
    Tensor last_u_;            // (S, 2d) fc_in output
    Tensor last_u1_;           // (S, d)
    Tensor last_u2_;           // (S, d)
    Tensor last_gelu_u2_;      // (S, d)
    Tensor last_Wu2_;          // (S, d)  W_spatial @ GELU(u2)
    Tensor last_v_;            // (S, d)  elementwise gated output
    Tensor last_y_;            // (S, d)  fc_out output
    Tensor last_residual_in_;  // (S, d)  the original input to the block
};

// ============================================================================
// gMLPModel: stack of gMLPBlocks + classifier
// ============================================================================

class gMLPModel : public Layer {
public:
    // dim:           model dimension d
    // seq_len:       token dimension S
    // out_features:  classifier output dim
    // num_blocks:    number of gMLPBlocks to stack
    gMLPModel(size_t dim, size_t seq_len, size_t out_features,
              size_t num_blocks = 1,
              double alpha_init = 0.01, bool sgu_bias = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "gMLPModel"; }

private:
    size_t dim_;
    size_t seq_len_;
    size_t out_features_;
    size_t num_blocks_;
    std::vector<gMLPBlock> blocks_;
    Dense classifier_;

    Tensor last_input_;
};

#endif
