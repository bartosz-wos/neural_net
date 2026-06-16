#ifndef PIXELCNN_H
#define PIXELCNN_H

#include "../../core/layer.h"
#include <cmath>
#include <vector>
#include <string>

// ============================================================================
// PixelCNN — van den Oord et al. 2016
//   "Pixel Recurrent Neural Networks"      (https://arxiv.org/abs/1601.06759)
//   "Conditional Image Generation with PixelCNN Decoders"
//                                  (https://arxiv.org/abs/1606.05328)
//
// Autoregressive image generation model that factorizes the joint distribution
// of pixels as a product of conditionals:
//   p(x) = prod_{i,j} p(x_{i,j} | x_{<i, *}, x_{i, <j})
//
// Each pixel is predicted as a categorical distribution over `n_vals` discrete
// values (e.g. 256 for 8-bit grayscale).
//
// Architecture:
//   1. MaskedConv2d (type A) — first layer masks the center pixel so each
//      output depends only on "earlier" pixels (the strict autoregressive
//      property for the input layer).
//   2. Stack of GatedPixelCNNBlock layers. Each block has:
//        (a) MaskedConv2d (type B) with GATED activation
//            y = tanh(W_f * x) * sigmoid(W_g * x)
//            [with optional conditioning: y = tanh(W_f*x + V_f*h) * sigmoid(W_g*x + V_g*h)]
//        (b) 1x1 conv (MaskedConv2d with kH=1, kW=1)
//        (c) Residual connection back to the input channel dim
//      The gated activation (tanh * sigmoid) is the "Gated PixelCNN" innovation
//      that improves over the original ReLU between layers.
//   3. Final classifier: 1x1 conv that maps hidden dim to (C_in * n_vals) and
//      a per-pixel softmax over the n_vals categories per channel.
//
// The mask in MaskedConv2d is enforced by zeroing out blocked kernel positions
// in the weights (and re-applying the mask after every weight update so the
// zero-constraint is permanent). Backward applies the same mask to grad_weights
// so gradients can never flow into blocked positions.
//
// Conventions (matching the rest of this repo):
//   * Single-image (N=1) batch with channels-first storage:
//     input tensor shape: (1, in_channels * H * W)
//     output tensor shape: (1, out_channels * H * W)
//   * Dense / Linear convention: y = X @ W^T + b for the 1x1 conv path
//     (each MaskedConv2d stores weights in (out_channels, in_channels * kH * kW)
//      layout, like the existing Conv2D).
//   * Conditioning vector h: shape (1, cond_dim) — broadcast across pixels.
// ============================================================================

// ---------------------------------------------------------------------------
// MaskedConv2d — Conv2D with a fixed binary mask on the kernel.
//
//   mask_type 'A': center pixel (kH/2, kW/2) is BLOCKED. Used for the first
//                  layer so output pixel never depends on the input pixel
//                  at the same location.
//   mask_type 'B': center pixel is ALLOWED. Used for subsequent layers
//                  (and for the residual 1x1 paths within a block).
//
// The mask is independent of channel (shared across in_ch / out_ch). This is
// the standard PixelCNN convention.
//
// Backward: gradients for blocked kernel positions are zeroed out so the
// masked constraint is preserved. Mask is re-applied to weights after every
// update_weights call.
// ---------------------------------------------------------------------------
class MaskedConv2d : public Layer {
public:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int H, W;       // input spatial dims
    int H_out, W_out; // output spatial dims
    char mask_type; // 'A' or 'B'

    // weights: (out_channels, in_channels * kernel_h * kernel_w)
    Tensor weights;
    Tensor bias;     // (out_channels, 1)
    Tensor grad_weights;
    Tensor grad_bias;

    // Caches for backward
    Tensor last_input; // (N, in_channels * H * W)
    Tensor col;        // im2col: (in_channels * kernel_h * kernel_w, N * H_out * W_out)
    Tensor mask;       // (1, kernel_h * kernel_w) — 1 for allowed, 0 for blocked
    // The full per-(out_ch, in_ch, kH*kW) mask (broadcast of mask_ to weight shape)
    Tensor weight_mask; // (out_channels, in_channels * kernel_h * kernel_w)

    MaskedConv2d();
    MaskedConv2d(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
                 char mask_in = 'B',
                 int stride_h = 1, int stride_w = 1,
                 int pad_h = -1, int pad_w = -1,
                 int dilation_h = 1, int dilation_w = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }
    std::string name() const override { return "MaskedConv2d"; }

    // Helper: build the (1, kH*kW) mask given mask_type
    static Tensor build_kernel_mask(int kH, int kW, char mask_type);
    // Helper: apply the mask to weights (zero out blocked positions)
    void enforce_mask();

protected:
    // Reused from Conv2D: im2col / col2im (static helpers; same algorithm)
    static Tensor im2col(const Tensor& input, int N, int C, int H, int W,
                         int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int dilation_h, int dilation_w,
                         int& H_out, int& W_out);
    static void col2im(Tensor& grad_input, const Tensor& grad_col,
                       int N, int C, int H, int W,
                       int kH, int kW, int stride_h, int stride_w,
                       int pad_h, int pad_w, int dilation_h, int dilation_w);
};

// ---------------------------------------------------------------------------
// GatedPixelCNNBlock — gated masked conv + 1x1 residual.
//
//   v = conv_masked_B(x)  with C_hidden = 2 * C_hidden_out  (split f and g)
//   h = tanh(v[:, :C_out]) * sigmoid(v[:, C_out:])
//   u = conv_1x1(h)         // C_in (skip dim)
//   out = x + u             // residual (same C_in for next block)
//
// With conditioning h_cond:
//   v = conv_masked_B(x)  with C_hidden = 2 * C_hidden_out
//   v += projection(h_cond)   // V matrix multiplies cond, broadcast across pixels
//   h = tanh(v[:, :C_out]) * sigmoid(v[:, C_out:])
//   ... rest is identical
// ---------------------------------------------------------------------------
class GatedPixelCNNBlock : public Layer {
public:
    int in_channels, hidden_channels;
    int H, W;
    int kH, kW;
    int cond_dim;  // 0 = no conditioning

    // Main gated masked conv: in_channels -> 2*hidden_channels
    // (f and g halves, then we apply tanh * sigmoid elementwise)
    MaskedConv2d conv_v_;
    // 1x1 conv: hidden_channels -> in_channels (residual path)
    MaskedConv2d conv_u_;
    // Conditioning projection: cond_dim -> 2*hidden_channels
    // Stored as Dense for simplicity (V_f and V_g stacked, both (cond_dim, 2*hidden))
    Dense cond_proj_;
    // Per-pixel bias for the 2*hidden_channels pre-activation
    Tensor bias_v_;
    Tensor grad_bias_v_;
    // Per-channel (across 2*hidden) bias for the residual 1x1 conv
    // (already in conv_u_.bias)

    GatedPixelCNNBlock(int in_ch, int hidden_ch, int kH, int kW, int H_in, int W_in,
                       char mask = 'B', int cond_dim_in = 0);

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_cond(const Tensor& input, const Tensor& cond);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    Tensor backward_with_cond(const Tensor& grad_output, double learning_rate,
                              const Tensor& cond);
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return conv_v_.weights; }
    Tensor get_gradients() const override { return conv_v_.grad_weights; }
    std::string name() const override { return "GatedPixelCNNBlock"; }

    // Caches for backward
    Tensor last_input_;          // (1, in_ch * H * W)
    Tensor last_cond_;           // (1, cond_dim) — if conditioning is used
    Tensor last_v_;              // (1, 2*hidden * H * W) — pre-activation
    Tensor last_h_;              // (1, hidden * H * W) — post tanh*sigmoid
    Tensor last_u_;              // (1, in_ch * H * W) — residual contribution
    Tensor last_output_;         // (1, in_ch * H * W) — final block output
    bool used_cond_ = false;     // true if last forward used conditioning
};

// ---------------------------------------------------------------------------
// PixelCNN — full autoregressive image model.
//
//   forward:
//     h0 = first_mask_conv_A(x)        // (1, hidden * H * W)
//     h1 = gated_block_B(h0)           // (1, in_ch * H * W) [residual back to in_ch]
//     h2 = gated_block_B(h1) * n_blocks
//     logits = classifier_1x1(hN)      // (1, C_in * n_vals * H * W)
//
// Note: the first_mask_conv outputs hidden_channels (so subsequent blocks
// can have a wider hidden dim), and each GatedPixelCNNBlock has hidden -> in_ch
// residual. To preserve hidden-channel width, we add a "pre-block projection"
// (1x1 conv) only if the input to a block isn't in_ch wide. For simplicity
// in v1, we keep all internal channels = `hidden_channels` and use 1x1
// residual projections back to in_ch.
//
// Per-pixel softmax: applied across the n_vals dimension inside each (i,j)
// position. For training with cross-entropy we expose logits and let the
// caller apply softmax. The output is the raw logits.
// ---------------------------------------------------------------------------
class PixelCNN : public Layer {
public:
    int in_channels;
    int hidden_channels;
    int n_blocks;
    int n_vals;       // number of categories per pixel per channel
    int H, W;
    int kH, kW;
    char first_mask;  // mask type for the first conv (typically 'A')
    int cond_dim;     // 0 = no conditioning

    // First masked conv (type A by default): in_channels -> hidden_channels
    MaskedConv2d first_conv_;
    // Stack of gated blocks
    std::vector<std::unique_ptr<GatedPixelCNNBlock>> blocks_;
    // Final classifier: hidden_channels -> in_channels * n_vals (1x1 conv)
    MaskedConv2d classifier_;

    PixelCNN(int in_ch, int hidden_ch, int kH, int kW, int H_in, int W_in,
             int n_blocks_in = 1, int n_vals_in = 256,
             char first_mask_in = 'A', int cond_dim_in = 0);

    Tensor forward(const Tensor& input) override;
    Tensor forward_with_cond(const Tensor& input, const Tensor& cond);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    Tensor backward_with_cond(const Tensor& grad_output, double learning_rate,
                              const Tensor& cond);
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return first_conv_.weights; }
    Tensor get_gradients() const override { return first_conv_.grad_weights; }
    std::string name() const override { return "PixelCNN"; }

    // Caches
    Tensor last_input_;
    Tensor last_cond_;
    Tensor last_h0_;
    std::vector<Tensor> last_block_outputs_;  // block output after each block
    bool used_cond_ = false;
};

#endif // PIXELCNN_H
