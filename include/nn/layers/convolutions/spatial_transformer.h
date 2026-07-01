#ifndef SPATIAL_TRANSFORMER_H
#define SPATIAL_TRANSFORMER_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <string>

// Spatial Transformer Network (STN) — Jaderberg et al. 2015
// "Spatial Transformer Networks" (https://arxiv.org/abs/1506.02025).
//
// A learnable module that applies a 2D affine transformation to a feature
// map (or input image) end-to-end. Three sub-modules:
//
//   1. Localization Network — takes the input feature map (N, C*H*W) and
//      outputs the 6 parameters of a 2D affine transformation matrix
//      theta ∈ R^{2×3}. Implemented as a small Conv2D(8 filters, 3x3) →
//      MaxPool2D(2x2) → ReLU → Dense(32) → ReLU → Dense(6) stack.
//
//   2. Grid Generator — given theta, produces a sampling grid G of shape
//      (H_out, W_out, 2) where each entry G(i, j) gives the (x, y)
//      source-coordinate in the normalized input frame [-1, 1]^2 to sample
//      from. Affine:
//         (x_t, y_t)^T = A · (x_s_norm, y_s_norm, 1)^T
//      where (x_s_norm, y_s_norm) = ((2j+1)/W - 1, (2i+1)/H - 1) and A
//      is the first two rows of theta.
//
//   3. Bilinear Sampler — uses the grid to sample the input feature map at
//      the (possibly non-integer) source coordinates via bilinear
//      interpolation. Out-of-bounds samples return 0. The bilinear weights
//      are cached for backward.
//
// Forward input shape:  (N, C * H * W)
// Forward output shape: (N, C * H * W)
//
// The output spatial dimensions equal the input spatial dimensions by
// default (same-shape affine transform). This is the canonical STN
// configuration used in Jaderberg et al.
//
// Full BPTT including:
//   - bilinear sample backward (∂L/∂input via the cached bilinear weights)
//   - grid backward (∂L/∂θ via the affine chain — for fixed theta this is
//     a contraction with the per-pixel source coordinates)
//   - localization backward (∂L/∂W, ∂L/∂b through conv → pool → fc chain).
//
// Use set_theta(theta) to inject a fixed transformation (e.g. identity
// for sanity tests, or a known affine for a baseline).
class SpatialTransformer : public Layer {
public:
    int N, C, H, W;          // input dims (set in constructor)
    int H_out, W_out;        // output spatial dims (default == H, W)

    // Localization-network sub-layers
    Tensor loc_conv_W;       // (8, C * 3 * 3)
    Tensor loc_conv_b;       // (1, 8)
    Tensor loc_dense1_W;     // (32, 8 * H_pool * W_pool)
    Tensor loc_dense1_b;     // (1, 32)
    Tensor loc_dense2_W;     // (6, 32)
    Tensor loc_dense2_b;     // (1, 6)

    // Gradients (mirror the param tensors)
    Tensor grad_loc_conv_W;
    Tensor grad_loc_conv_b;
    Tensor grad_loc_dense1_W;
    Tensor grad_loc_dense1_b;
    Tensor grad_loc_dense2_W;
    Tensor grad_loc_dense2_b;

    // Cached for backward
    Tensor last_input;       // (N, C*H*W) — for d_input
    Tensor last_theta;       // (2, 3) — fixed during this forward (or from loc net)
    Tensor last_grid;        // (H_out * W_out, 2) — source coords per output pixel
    std::vector<std::vector<double>> last_bilinear_w; // per output pixel: 4 bilinear weights

    // Fixed-theta mode: when set to true via set_theta, skip localization
    // network and use the supplied theta directly.
    bool use_fixed_theta_;
    Tensor fixed_theta_;     // (2, 3)

    SpatialTransformer(int batch, int channels, int H_in, int W_in);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return loc_conv_W; }
    Tensor get_gradients() const override { return grad_loc_conv_W; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "SpatialTransformer"; }

    // Get the most recently computed theta (after forward)
    Tensor get_theta() const { return last_theta; }

    // Force a specific theta for the next forward pass (skips the
    // localization network). Pass a (2, 3) tensor.
    void set_theta(const Tensor& theta);
    void clear_fixed_theta() { use_fixed_theta_ = false; }

private:
    // Sub-layer implementations (re-implemented locally for self-containment —
    // the existing Dense uses a different layout convention).

    // Conv2D forward on a single-channel (or C-channel) input padded with
    // (kH-1)/2 same-padding. Returns (N, out_ch * H * W).
    Tensor conv2d_forward(const Tensor& input, const Tensor& W, const Tensor& b,
                          int in_ch, int out_ch, int kH, int kW);
    // Backward returns d_input; updates dW and db in place.
    Tensor conv2d_backward(const Tensor& grad_output, const Tensor& W,
                            int in_ch, int out_ch, int kH, int kW,
                            Tensor& dW, Tensor& db);

    // Max pool 2x2 stride 2 — forward returns (N, C * H_pool * W_pool).
    Tensor maxpool2d_forward(const Tensor& input) const;
    // Backward takes grad_output (N, C * H_pool * W_pool), returns d_input.
    Tensor maxpool2d_backward(const Tensor& grad_output, int H_pool, int W_pool);

    // Dense forward (y = xW^T + b)
    Tensor dense_forward(const Tensor& input, const Tensor& W, const Tensor& b) const;
    // Backward returns d_input; updates dW and db in place.
    Tensor dense_backward(const Tensor& grad_output, const Tensor& W,
                           Tensor& dW, Tensor& db);

    // ReLU forward / backward
    Tensor relu_forward(const Tensor& input) const;
    Tensor relu_backward(const Tensor& grad_output, const Tensor& input) const;

    // Localization-network forward: returns theta (2, 3).
    Tensor localization_forward(const Tensor& input);
    // Localization-network backward: takes d_theta (2, 3), updates internal
    // grad_loc_* tensors. Returns grad_input to the loc net (which equals
    // grad_input to the bilinear sampler when use_fixed_theta_ is false).
    Tensor localization_backward(const Tensor& d_theta);

    // Grid generator: produces (H_out, W_out, 2) tensor of source coords in
    // normalized [-1, 1]^2 frame from theta (2, 3).
    Tensor generate_grid(const Tensor& theta) const;
    // Grid backward: takes d_grid (H_out * W_out, 2), returns d_theta (2, 3).
    Tensor grid_backward(const Tensor& d_grid) const;

    // Bilinear sample forward: (N, C*H*W) input + grid -> (N, C*H_out*W_out) output
    Tensor bilinear_sample_forward(const Tensor& input, const Tensor& grid);
    // Bilinear sample backward: takes d_output (N, C*H_out*W_out), returns
    // d_input (N, C*H*W) and d_grid (H_out*W_out, 2).
    Tensor bilinear_sample_backward_input(const Tensor& grad_output);
    Tensor bilinear_sample_backward_grid(const Tensor& grad_output);

    // im2col-style conv (small, supports stride=1, dilation=1 only).
    Tensor conv_im2col(const Tensor& input, int in_ch, int kH, int kW);
    void conv_col2im(Tensor& grad_input, const Tensor& grad_col,
                     int in_ch, int kH, int kW);
};

#endif