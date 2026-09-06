#ifndef MAMBA_CONV_H
#define MAMBA_CONV_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// MambaConv — Canonical Mamba-1 block with depthwise causal 1D convolution
//   "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//   Gu & Dao, 2023  (https://arxiv.org/abs/2312.00752)
//
// This is the CANONICAL Mamba-1 design from the paper (Figure 3 in §3.1):
//
//   in_proj -> split into [x_pre, gate]
//                       |        |
//                       v        v
//              depthwise causal 1D conv (kernel k, per-channel)
//                       |
//                       v
//                      SiLU
//                       |
//                       v
//              selective SSM  (input-dependent Δ, B, C)
//                       |
//                       v
//                       y
//                       |
//   gate -> SiLU -> (.) * (y + D_skip * x_ssm)
//                            |
//                            v
//                       out_proj
//
// WHEREAS the existing `MambaBlock` deliberately omits the convolution
// (see header comment in `mamba.h`: "We do NOT include the depthwise 1D
// conv that the canonical Mamba has..."), this layer adds it back.
//
// Causal left-padding convention: at position t, the conv reads
// x_pre[t - j] for j = 0..k-1 with x_pre[< 0] = 0. This means the first
// (k-1) positions see fewer than k inputs. For T < k the conv still
// works — contributions from out-of-range positions are zero.
//
// Depthwise: ONE filter per channel, weight shape (d_inner, k), bias (d_inner).
//
// All 14 parameter tensors are learnable:
//   in_proj, out_proj, dt_proj, B_proj, C_proj (5 Dense layers, 10 tensors)
//   A_log (d_inner, d_state)            — unconstrained; A = -exp(A_log)
//   D_skip (1, d_inner)                 — learnable skip connection
//   conv_weight (d_inner, conv_kernel)  — depthwise causal conv filters
//   conv_bias   (1, d_inner)            — depthwise causal conv biases
//
// Shape conventions:
//   x:                (T, d_model)
//   in_proj output:   (T, 2*d_inner)         weights: (2*d_inner, d_model)
//   conv output:      (T, d_inner)           weight: (d_inner, conv_kernel)
//   dt_proj output:   (T, d_inner)
//   B_proj output:    (T, d_state)
//   C_proj output:    (T, d_state)
//   A_log:            (d_inner, d_state)
//   y, gated:         (T, d_inner)
//   out:              (T, d_model)
// ============================================================================

class MambaConvBlock : public Layer {
public:
    MambaConvBlock(size_t d_model, size_t d_state, size_t d_inner = 0, size_t conv_kernel = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "MambaConvBlock"; }

    // Accessors for tests
    size_t d_model()     const { return d_model_; }
    size_t d_state()     const { return d_state_; }
    size_t d_inner()     const { return d_inner_; }
    size_t conv_kernel() const { return conv_kernel_; }

    // Public parameter tensors (test access)
    Dense in_proj;          // d_model -> 2*d_inner
    Dense out_proj;         // d_inner -> d_model
    Dense dt_proj;          // d_model -> d_inner
    Dense B_proj;           // d_model -> d_state
    Dense C_proj;           // d_model -> d_state
    Tensor A_log;           // (d_inner, d_state)
    Tensor D_skip;          // (1, d_inner)
    Tensor conv_weight;     // (d_inner, conv_kernel)  — depthwise
    Tensor conv_bias;       // (1, d_inner)

    // Hidden gradient buffers
    Tensor grad_A_log_;
    Tensor grad_D_skip_;
    Tensor grad_conv_weight_;
    Tensor grad_conv_bias_;

private:
    size_t d_model_;
    size_t d_state_;
    size_t d_inner_;
    size_t conv_kernel_;

    // Forward caches
    Tensor last_input_;         // (T, d_model)
    Tensor last_p_;             // (T, 2*d_inner)
    Tensor last_x_pre_;         // (T, d_inner)   raw post-in_proj
    Tensor last_x_conv_;        // (T, d_inner)   post-conv + bias
    Tensor last_ssm_in_;        // (T, d_inner)   post-SiLU
    Tensor last_gate_;          // (T, d_inner)
    Tensor last_Delta_;         // (T, d_inner)
    Tensor last_Delta_pre_;     // (T, d_inner)
    Tensor last_B_t_;           // (T, d_state)
    Tensor last_C_t_;           // (T, d_state)
    Tensor last_A_bar_;         // (T, d_inner, d_state)
    Tensor last_B_bar_;         // (T, d_inner, d_state)
    Tensor last_h_;             // (T+1, d_inner, d_state)
    Tensor last_y_;             // (T, d_inner)
    Tensor last_gated_;         // (T, d_inner)

    // Numerically stable helpers
    static double softplus(double x) {
        if (x > 30.0) return x;
        if (x < -30.0) return std::exp(x);
        return std::log(1.0 + std::exp(x));
    }
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double silu(double x) {
        return x * sigmoid(x);
    }
    static double silu_deriv(double x) {
        // d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        double s = sigmoid(x);
        return s * (1.0 + x * (1.0 - s));
    }
};

#endif // MAMBA_CONV_H
