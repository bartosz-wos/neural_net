#ifndef DEFORMABLE_ATTENTION_H
#define DEFORMABLE_ATTENTION_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// Deformable 1D Attention — Xia et al., ICLR 2023
//   "Vision Transformers with Deformable Attention"
//   https://arxiv.org/abs/2201.00520
//
// Adapted to sequence modeling (1D): each query learns K sampling OFFSETS
// relative to a fixed reference grid, and attends only to the K bilinearly-
// sampled key/value pairs — O(N·K) instead of O(N²). Sparsity is LEARNED
// (unlike NSA's block-level sparsity or sliding window's fixed bandwidth).
//
// Math (per head h, head_dim = d_model / num_heads):
//
//   Q_h = (X · W_q)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}
//   K_h = (X · W_k)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}
//   V_h = (X · W_v)[:, h·head_dim : (h+1)·head_dim]    in R^{n × head_dim}
//
//   Offsets per (query, point, head) — paper Eq. 2, signed and bounded:
//     ΔQ_raw_h = (X · W_offsets)[:, h·num_points : (h+1)·num_points]  in R^{n × num_points}
//     ΔQ_h     = s · tanh(ΔQ_raw_h)                                   in R^{n × num_points}, values in (-s, s)
//   where s = offset_scale = max(1, (n-1)/2) is the maximum displacement.
//   tanh (not sigmoid) so the offset is SIGNED: a query can look BACKWARD as
//   well as forward from its reference point. An unsigned σ(·)·(n-1) offset
//   would only ever push right and saturate against the n-1 clamp, making the
//   early part of the sequence unreachable.
//
//   Sampling positions per (query t, point k):
//     pos_{t, k} = clamp(reference_points[k] · (n-1) + ΔQ_h[t, k], 0, n - 1)
//
//   Bilinear sampling of K_h and V_h at positions pos (with j clamped):
//     i = floor(pos), j = min(i+1, n-1), α = pos - i, β = 1 - α
//     K_sampled[t, k, d] = β · K_h[i, d] + α · K_h[j, d]
//     V_sampled[t, k, d] = β · V_h[i, d] + α · V_h[j, d]
//
//   Attention (paper Eq. 5):
//     score_{t, k} = (Q_h[t] · K_sampled[t, k]) / sqrt(head_dim)
//     attn_{t, k}  = softmax_k(score_{t, k})                          in R^{n × num_points}
//     head_out[t]   = Σ_k attn_{t, k} · V_sampled[t, k]                in R^{head_dim}
//
//   Output:
//     concat heads → (n × d_model)
//     out = concat · W_o                                              in R^{n × d_model}
//
// Conventions:
//   * Input / output: (n, d_model)
//   * d_model must be evenly divisible by num_heads
//   * num_points >= 1, num_ref_points >= 1
//   * num_ref_points must equal num_points (each ref point corresponds to one
//     learned offset per (query, head)) — a 1:1 mapping per paper §3.2.
//   * Non-causal. The paper's deformable attention is bidirectional; users wrap
//     for causality if needed.
//   * No bias on any of the projections.
//   * reference_points is a learnable (1, num_ref_points) parameter, init to
//     uniform grid (r/N for r = 0..N-1, but with values in (0, 1) since the
//     forward scales by (n-1)).
//
// Param count breakdown:
//   W_q:            d_model × d_model      (per-head block-stacked)
//   W_k:            d_model × d_model
//   W_v:            d_model × d_model
//   W_offsets:      d_model × (num_heads · num_points)
//   reference_pts:  1 × num_points
//   W_o:            d_model × d_model
// ============================================================================

class Deformable1DAttention : public Layer {
public:
    Deformable1DAttention(size_t d_model, size_t num_heads,
                          size_t num_points, size_t num_ref_points = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "Deformable1DAttention"; }

    // Accessors
    size_t d_model()         const { return d_model_; }
    size_t num_heads()       const { return num_heads_; }
    size_t num_points()      const { return num_points_; }
    size_t num_ref_points()  const { return num_ref_points_; }
    size_t head_dim()        const { return head_dim_; }

    // === Members declared in initializer-list order (private accessors come later) ===
    size_t d_model_;
    size_t num_heads_;
    size_t num_points_;
    size_t num_ref_points_;
    size_t head_dim_;
    Dense W_q, W_k, W_v;
    Dense W_offsets;        // (d_model, num_heads × num_points)
    Dense W_o;              // (d_model, d_model)
    Tensor reference_points; // (1, num_ref_points)
    Tensor grad_reference_points;

    // Caches for backward
    Tensor last_input_;           // (n, d_model)
    Tensor last_q_;               // (n, d_model)
    Tensor last_k_;               // (n, d_model)
    Tensor last_v_;               // (n, d_model)
    Tensor last_offsets_;         // (n, num_heads × num_points) — RAW pre-sigmoid
    Tensor last_positions_;       // (n, num_heads, num_points) — sampling positions in [0, n-1]
    Tensor last_K_sampled_;       // (n, num_heads, num_points, head_dim)
    Tensor last_V_sampled_;       // (n, num_heads, num_points, head_dim)
    Tensor last_attn_;            // (n, num_heads, num_points) softmax probs
    Tensor last_concat_;          // (n, d_model) pre-W_o
};

// ----------------------------------------------------------------------------
// Deformable1DBlock — pre-LN deformable attn -> residual -> pre-LN GELU FFN ->
// residual. Mirrors the convention of TokenformerBlock / StickBreakingBlock.
// ----------------------------------------------------------------------------
class Deformable1DBlock : public Layer {
public:
    Deformable1DBlock(size_t d_model, size_t num_heads, size_t num_points,
                      size_t num_ref_points = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.W_q.grad_weights; }
    std::string name() const override { return "Deformable1DBlock"; }

    Deformable1DAttention attn;
    Dense ffn1, ffn2;
    std::unique_ptr<LayerNorm> ln1_, ln2_;

    Tensor last_ln1_;       // cache for backward
    Tensor last_ln2_;       // cache
    Tensor last_f1_;        // cache: pre-GELU FFN1 output
    Tensor last_f1g_;       // cache: post-GELU FFN1 output
};

// ----------------------------------------------------------------------------
// Deformable1DModel — input projection -> N blocks -> final LN -> classifier.
// ----------------------------------------------------------------------------
class Deformable1DModel : public Layer {
public:
    Deformable1DModel(size_t d_input, size_t d_model, size_t d_output,
                      size_t num_blocks, size_t num_heads, size_t num_points,
                      size_t num_ref_points = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj.weights; }
    Tensor get_gradients() const override { return input_proj.grad_weights; }
    std::string name() const override { return "Deformable1DModel"; }

    Dense input_proj;
    Dense classifier;
    std::vector<std::unique_ptr<Deformable1DBlock>> blocks_;
    std::unique_ptr<LayerNorm> final_ln_;
};

#endif