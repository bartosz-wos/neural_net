#ifndef MULTI_SCALE_DEFORMABLE_ATTENTION_H
#define MULTI_SCALE_DEFORMABLE_ATTENTION_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// Multi-Scale Deformable 1D Attention — Zhu et al., ICLR 2021
//   "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
//   https://arxiv.org/abs/2010.04159
//
// 1D-adapted Eq. 3. Each query attends to only L*K (level, point) pairs
// sampled from L strided-pool feature levels of the input. The attention
// weights A[t, l, k] are predicted DIRECTLY from the query (softmax over
// L*K jointly) — no Q·K dot-product. This is the paper's key efficiency
// trick: removes the K projection entirely from the attention path.
//
// Math (per head h, head_dim = d_model / num_heads, L = num_levels,
// K = num_points per level):
//
//   Q       = X · W_q                            in R^{n × d_model}
//   Δ_raw   = X · W_offsets                      in R^{n × (H·L·K)}      # raw offsets
//   A_raw   = X · W_attn                         in R^{n × (H·L·K)}      # raw attn logits
//   V^l     = level_v_projs[l].forward(X)        in R^{n_l × d_model}    # level l features
//   n_l     = ⌈n / 2^l⌉                                              # strided pooling
//
//   Attention (paper's key efficiency — softmax over L*K jointly):
//     A[t, h, l, k] = softmax_{l, k}(A_raw[t, h·L·K + l·K + k])
//
//   Sampling positions (paper Eq. 3, signed bounded, per-level scaled):
//     ref_l[t] = (t / max(1, n − 1)) · (n_l − 1)               in [0, n_l − 1]
//     pos_{t, h, l, k} = clamp( ref_l[t] + s_l · tanh(Δ_raw[t, h·L·K + l·K + k]),
//                                0, n_l − 1 )
//     where s_l = max(1, (n_l − 1) / 2)
//
//   Bilinear sampling of V^l at pos (per head h slice):
//     i = floor(pos), j = min(i+1, n_l−1), α = pos − i, β = 1 − α
//     V_sampled[t, h, l, k, d] = β · V^l[i, h·head_dim + d]
//                              + α · V^l[j, h·head_dim + d]
//
//   Output per (t, d):
//     out[t, h, d] = Σ_{l, k} A[t, h, l, k] · V_sampled[t, h, l, k, d]
//     concat heads → (n × d_model) → W_o → (n × d_model)
//
// Conventions:
//   * Input/output: (n, d_model)
//   * d_model must be evenly divisible by num_heads
//   * num_levels >= 1, num_points >= 1
//   * Non-causal. Pooling is strided (every 2^l-th row).
//   * Biases on Q/attn/offsets projections; biases on level V projections too
//     (Dense default).
//   * reference_points is NOT a learnable parameter — it's the query-row-index
//     scaled to each level's length (the canonical DETR §3.2 convention).
//
// Param count breakdown (per level L, per head H):
//   W_q:         d_model × d_model
//   W_offsets:   d_model × (H·L·K)
//   W_attn:      d_model × (H·L·K)
//   W_o:         d_model × d_model
//   level_v_projs[l]:  d_model × d_model     (L of them)
// ============================================================================

class MultiScaleDeformable1DAttention : public Layer {
public:
    MultiScaleDeformable1DAttention(size_t d_model, size_t num_heads,
                                    size_t num_levels, size_t num_points);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "MultiScaleDeformable1DAttention"; }

    // Accessors
    size_t d_model()     const { return d_model_; }
    size_t num_heads()   const { return num_heads_; }
    size_t num_levels()  const { return num_levels_; }
    size_t num_points()  const { return num_points_; }
    size_t head_dim()    const { return head_dim_; }

    // === Members declared in initializer-list order ===
    size_t d_model_;
    size_t num_heads_;
    size_t num_levels_;
    size_t num_points_;
    size_t head_dim_;
    size_t LK_;            // L*K
    size_t HLK_;           // H*L*K
    Dense W_q;             // (d_model, d_model)
    Dense W_offsets;       // (d_model, H·L·K)
    Dense W_attn;          // (d_model, H·L·K)
    Dense W_o;             // (d_model, d_model)
    std::vector<Dense> level_v_projs_;   // L of (d_model, d_model)

    // Caches for backward (all in row-major flat layout)
    Tensor last_input_;           // (n, d_model)
    Tensor last_q_;               // (n, d_model)
    Tensor last_offsets_;         // (n, H·L·K) RAW pre-tanh
    Tensor last_attn_;            // (n, H·L·K) post-softmax A
    Tensor last_positions_;       // (n, H·L·K) sampling positions in [0, n_l-1]
    Tensor last_V_sampled_;       // (n, H·L·K·head_dim)
    Tensor last_levels_;          // packed per-level V^l (used for bilinear backward)
                                  //   layout: level l starts at row offset sum_{m<l} n_m
                                  //   so last_levels_.row_offset(l) = sum_{m<l} n_m
    Tensor last_concat_;          // (n, d_model) pre-W_o
    // Per-level row offsets (sum of n_m for m<l) — cached for the level-scatter backward
    std::vector<size_t> level_row_offsets_;
    std::vector<size_t> level_lengths_;   // n_l for each level
    // Source row for each pooled row (row r of level l came from row r·2^l of X)
    // We don't need to cache the per-level V^l tensors as members because
    // last_levels_ holds them contiguously.
};

// ----------------------------------------------------------------------------
// MultiScaleDeformable1DBlock — pre-LN multi-scale deformable attn -> residual
// -> pre-LN GELU FFN -> residual.
// ----------------------------------------------------------------------------
class MultiScaleDeformable1DBlock : public Layer {
public:
    MultiScaleDeformable1DBlock(size_t d_model, size_t num_heads,
                               size_t num_levels, size_t num_points);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.W_q.grad_weights; }
    std::string name() const override { return "MultiScaleDeformable1DBlock"; }

    MultiScaleDeformable1DAttention attn;
    Dense ffn1, ffn2;
    std::unique_ptr<LayerNorm> ln1_, ln2_;

    Tensor last_ln1_;
    Tensor last_ln2_;
    Tensor last_f1_;
    Tensor last_f1g_;
};

#endif
