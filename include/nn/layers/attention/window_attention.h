#ifndef WINDOW_ATTENTION_H
#define WINDOW_ATTENTION_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <string>

// ============================================================================
// Window Attention — Liu et al. 2021, "Swin Transformer: Hierarchical Vision
// Transformer using Shifted Windows" (https://arxiv.org/abs/2103.14030,
// ICCV 2021 Best Paper, §3.2)
//
// The Swin block's local self-attention primitive. For a (H, W) grid of
// tokens, partition into ⌈H/M⌉·⌈W/M⌉ non-overlapping M×M windows, run
// standard multi-head attention WITHIN each window, and add a learnable
// relative position bias B ∈ R^{num_heads, (2M-1)²} indexed by the
// pair-wise relative offset between tokens in the same window
// (paper §3.2: "We propose to introduce a relative position bias ...").
//
// Complexity per window is O(M²·d_model); the per-image cost is
// O(N·M·d_model) instead of O(N²·d_model) for vanilla attention, where
// N = H·W. With M=7 (paper default) and H=W=224, that's a ~32× reduction.
//
// Conventions:
//   * Input:  (N, d_model) — flat token grid for one example, with
//             N = H_pad * W_pad where H_pad % M == 0 and W_pad % M == 0.
//             The caller is responsible for ensuring divisibility; the
//             constructor validates this. H_pad_ and W_pad_ accessors
//             are exposed for tests.
//   * Output: (N, d_model) — same shape as input (window attention is
//             a per-position mix).
//   * d_model must be evenly divisible by num_heads
//   * bias_type controls the relative-position-bias table:
//       "relative"  (default) — relative position bias B, the paper's choice
//       "none"               — no bias (vanilla window attention)
//   * QKV projections are bias-free; the output projection has a per-head
//     output bias b_o ∈ R^{1, d_model}. Matches the other attention
//     variants in this repo.
//
// Deliberate deviations from the paper (recorded for v1):
//   1. NO shifted-window partitioning — the §3.2 "Shifted Window" trick
//      uses a cyclic ⌊M/2⌋ shift before the second block and a masked
//      attention to keep cross-window information local. v1 ships only
//      the non-shifted path; the shifted variant is a future TODO.
//   2. NO padding helper — callers must pre-pad H, W to multiples of M
//      before calling forward. The constructor validates divisibility.
//   3. NO qkv_bias vector — the canonical Swin concatenates QKV into a
//      single (3, d_model) bias; we use bias-free QKV (matches every
//      other attention variant in this repo).
//   4. NO attn_drop / proj_drop — the repo has no attention-dropout
//      convention.
//
// Classes:
//   WindowAttention       — the local-window multi-head self-attention layer
// ============================================================================

class WindowAttention : public Layer {
public:
    // d_model:      input/output feature dim
    // num_heads:    number of attention heads (d_model must be divisible by num_heads)
    // window_size:  M — side length of each non-overlapping window
    // H_pad, W_pad: pre-padded grid dimensions; H_pad % M == 0, W_pad % M == 0
    // bias_type:    "relative" (default) or "none"
    WindowAttention(size_t d_model, size_t num_heads, size_t window_size,
                    size_t H_pad, size_t W_pad,
                    const std::string& bias_type = "relative");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "WindowAttention"; }

    // Accessors
    size_t d_model()         const { return d_model_; }
    size_t num_heads()       const { return num_heads_; }
    size_t head_dim()        const { return head_dim_; }
    size_t window_size()     const { return window_size_; }
    size_t H_pad()           const { return H_pad_; }
    size_t W_pad()           const { return W_pad_; }
    size_t num_windows()     const { return num_windows_; }
    const std::string& bias_type() const { return bias_type_; }

    // Public parameter tensors (test access + mutation)
    Tensor W_q, W_k, W_v, W_o;                  // (d_model, d_model) each
    Tensor b_o;                                  // (1, d_model)
    Tensor relative_position_bias_;              // (num_heads, (2M-1)²)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_o;
    Tensor grad_relative_position_bias_;

private:
    size_t d_model_, num_heads_, head_dim_;
    size_t window_size_, H_pad_, W_pad_, num_windows_;
    size_t M_, M2_;                              // window_size_, M_*M_
    size_t rel_bias_size_;                       // (2M-1)²
    std::string bias_type_;
    double scale_;

    // BPTT cache
    Tensor last_input_;                          // (N, d_model)
    Tensor last_q_, last_k_, last_v_;            // (N, d_model) — projections
    // Attention cached as flat (num_windows_ * num_heads_ * M², M²)
    Tensor last_attn_;
    Tensor last_head_out_;                       // (N, d_model)
    Tensor last_out_pre_;                        // (N, d_model)

    // window_indices_[w * M² + i] = flat input-row index for window w, in-window pos i.
    std::vector<size_t> window_indices_;

    // 4-D → flat helper
    double a4(const Tensor& t, size_t w, size_t h, size_t qi, size_t ki) const {
        return t[(w * num_heads_ + h) * M2_ + qi][ki];
    }
    double& a4(Tensor& t, size_t w, size_t h, size_t qi, size_t ki) {
        return t[(w * num_heads_ + h) * M2_ + qi][ki];
    }
};

#endif
