#ifndef CONVNEXT_H
#define CONVNEXT_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include "../normalization/layer_scale.h"
#include <vector>

// ConvNeXt — Liu, Zhuang, Lin, Luo (CVPR 2022, arXiv:2201.03545)
// "A ConvNet for the 2020s".
//
// Built entirely from already-shipped repo primitives: Conv2D, Dense, LayerScale,
// GELU. The one genuinely NEW piece is ChannelLayerNorm2D — the repo's shipped
// `LayerNorm` normalizes over an entire ROW (cols = C*H*W), which for a
// flattened feature map mixes channels with spatial positions. ConvNeXt's
// LayerNorm normalizes over the CHANNEL axis at each spatial position, so it
// needs its own class.

// ---------------------------------------------------------------------------
// ChannelLayerNorm2D — LayerNorm over the channel axis of an (N, C*H*W) tensor.
//
//   index:  x[n][c * (H*W) + s]     c in [0,C),  s = h*W + w in [0, H*W)
//
//   for each (n, s):
//     mu   = (1/C) * sum_c x[n][c*S + s]
//     var  = (1/C) * sum_c (x[n][c*S + s] - mu)^2
//     xhat = (x[n][c*S + s] - mu) * rsqrt(var + eps)
//     out  = gamma[c] * xhat + beta[c]
//
// This is a per-spatial-position normalization with SHARED affine parameters
// across spatial positions — matching torchvision's `LayerNorm2d`/ConvNeXt
// reference, and distinct from the repo's row-wise `LayerNorm`.
// ---------------------------------------------------------------------------
class ChannelLayerNorm2D : public Layer {
public:
    Tensor gamma, beta;        // (1, C)
    Tensor grad_gamma, grad_beta;
    double eps;

    ChannelLayerNorm2D(size_t channels, size_t H, size_t W, double eps = 1e-5);
    ~ChannelLayerNorm2D() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ChannelLayerNorm2D"; }

    size_t channels() const { return channels_; }
    size_t height()  const { return H_; }
    size_t width()   const { return W_; }
    size_t spatial() const { return H_ * W_; }

    // Caches from the last forward, needed by backward.
    Tensor last_xhat;      // (N, C*S)
    Tensor last_inv_std;   // (N, S)

private:
    size_t channels_, H_, W_;
};

// ---------------------------------------------------------------------------
// ConvNeXtBlock — §2.1 Eq. (1), isotropic variant.
//
//   x_out = x_in + gamma * f( LN( DWConv7x7(x_in) ) )
//   f(u)  = PWConv2( GELU( PWConv1(u) ) )
//   PWConv1: C -> 4C  (bias)   PWConv2: 4C -> C  (bias)
//   DWConv : C -> C   (7x7, NO bias, stride 1, pad 3)
//   gamma  : per-channel LayerScale, init 1e-6
//
// Note both orderings, which are easy to get backwards and are each covered by
// a dedicated test: the dwconv runs BEFORE the LayerNorm, and the LayerNorm
// sits INSIDE the residual branch (post-norm).
// ---------------------------------------------------------------------------
class ConvNeXtBlock : public Layer {
public:
    ConvNeXtBlock(size_t dim, size_t H, size_t W,
                  double layer_scale_init = 1e-6, size_t kernel = 7);
    ~ConvNeXtBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ConvNeXtBlock"; }

    size_t dim() const { return dim_; }
    size_t height() const { return H_; }
    size_t width()  const { return W_; }
    size_t spatial() const { return H_ * W_; }
    size_t mlp_hidden() const { return 4 * dim_; }
    size_t kernel() const { return kernel_; }
    double layer_scale_init() const { return layer_scale_init_; }

    // Sub-layers, exposed for the FD gradient checks.
    Conv2D&              dwconv(size_t c)   { return dw_[c]; }
    ChannelLayerNorm2D&  ln()               { return ln_; }
    Dense&               pwconv1()          { return pwconv1_; }
    Dense&               pwconv2()          { return pwconv2_; }
    LayerScale&          gamma()            { return gamma_; }

private:
    // Depthwise: one Conv2D(1,1,k,k) per channel, no bias — same structure the
    // shipped DepthwiseSeparableConv uses for its depthwise stage.
    Tensor depthwise_forward(const Tensor& input);
    Tensor depthwise_backward(const Tensor& grad, double lr);

    size_t dim_, H_, W_, kernel_;
    double layer_scale_init_;
    std::vector<Conv2D> dw_;      // size dim_, each 1->1
    ChannelLayerNorm2D ln_;
    Dense pwconv1_;                // (4C, C)
    Dense pwconv2_;                // (C, 4C)
    LayerScale gamma_;             // (1, C)
    Tensor last_input_;
    Tensor last_pre_gelu_;
};

// ---------------------------------------------------------------------------
// ConvNeXt — §3, isotropic ConvNeXt-B layout (Table 11).
//
//   stem : Conv2d(in_ch, dims[0], 4, stride 4)
//   for each stage s:  blocks_s (depth depths[s] ConvNeXtBlock)
//   between stages:   ChannelLayerNorm2D -> Conv2d(2, stride 2)   (§3 "downsample")
//   head : ChannelLayerNorm2D -> global average pool -> Linear
//
// Defaults are the paper's isotropic 100M config: depths {3,3,9,3}, dims
// {96,192,384,768}. v1 simplification: isotropic only (no FLOPs-shaped ratios
// of §3.2) and no stochastic depth / layer-scale schedule (§4.1).
// ---------------------------------------------------------------------------
class ConvNeXt : public Layer {
public:
    ConvNeXt(size_t in_channels, size_t H, size_t W, size_t num_classes,
             const std::vector<size_t>& depths = {3, 3, 9, 3},
             const std::vector<size_t>& dims   = {96, 192, 384, 768},
             double layer_scale_init = 1e-6);
    ~ConvNeXt() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ConvNeXt"; }

    size_t num_stages() const { return depths_.size(); }
    size_t stem_dim() const { return dims_[0]; }
    size_t head_dim() const { return dims_.back(); }
    // Spatial extent entering stage `s` (stage 0 is after the 4x4 stride-4 stem).
    size_t stage_spatial(size_t s) const;

private:
    void build_downsamplers();

    size_t in_channels_, H_in_, W_in_, num_classes_;
    std::vector<size_t> depths_, dims_;
    Conv2D stem_;
    // One (LayerNorm, Conv2D 2x2 stride 2) pair per stage boundary.
    std::vector<ChannelLayerNorm2D> down_ln_;
    std::vector<Conv2D> down_conv_;
    std::vector<std::vector<ConvNeXtBlock>> blocks_;
    ChannelLayerNorm2D final_ln_;
    Dense head_;
    Tensor last_input_;
};

#endif
