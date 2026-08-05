#ifndef HYENA_H
#define HYENA_H

#include "../../core/layer.h"
#include "../../activations/activations.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// Hyena Hierarchy (Poli et al. 2023) — "Towards Larger Convolutional Language
// Models". https://arxiv.org/abs/2302.10866
//
// Innovation: replace the O(n^2) self-attention matrix multiply with a
// recurrence of (a) a long implicit convolution (parameterized by a small
// MLP that consumes positional embeddings) and (b) element-wise multiplicative
// gating. The result is subquadratic in sequence length while preserving
// quality comparable to attention on long-context language modeling.
//
// Canonical HyenaOperator (order = 2):
//   u_in = in_proj(x)                              // (B, L, (order+1)*D)
//   u_in = transpose to (B, (order+1)*D, L)        // channels-first
//   short = depthwise_conv1d(u_in, k=3, pad=1, ...)// (B, (order+1)*D, L)
//   [gate_0, gate_1, v] = split(short, D)          // 3 chunks of D channels
//   v = v * gate_1
//   v = long_conv(v, k_0, D_0)                     // y = conv(k_0, v) + D_0*v
//   v = v * gate_0
//   v = long_conv(v, k_1, D_1)
//   out = v.transpose to (B, L, D)                 // back to channels-last
//   out = out_proj(out)
//
// We use a naive O(L^2) convolution for the long-conv primitive so the
// gradient check is tractable at small L (L=8) — the FFT-based version is
// a clean drop-in upgrade but not necessary for correctness verification.
//
// Filter generation (HyenaFilter::filter(L)):
//   z = positional_embedding(L, emb_dim)           // (L, emb_dim)
//   h = MLP(z)                                     // (L, D) via:
//        Linear(emb_dim, P) → sin(freq * x) → Linear(P, P) → sin → ... →
//        Linear(P, P) → sin → Linear(P, D, bias=false)
//   h = exp(-delta * t) * h                        // exponential decay (causality)
//   D_skip = D_param                              // per-channel skip
//   return (h, D_skip)
//
// Block (HyenaBlock):
//   x = x + HyenaOperator(LayerNorm(x))
//   x = x + FFN(LayerNorm(x))                       // GELU FFN, ffn_mult expansion
//
// Model (HyenaModel):
//   stack of HyenaBlocks → per-token mean pool → classifier
//
// All tensors in repo convention: row-major (rows, cols).
// ============================================================================

// ----------------------------------------------------------------------------
// Free helper: 1D causal convolution.
//   y[t] = sum_{s=0..L-1, s<=t} k[s] * v[t-s]
// where v, k, y are all length L.
// Forward: y = conv(v, k)  (k is the kernel; causal in time)
// Backward:
//   dL/dv[t'] = sum_t grad_y[t] * k[t - t']
//   dL/dk[s]  = sum_t grad_y[t] * v[t - s]
// Both are again causal convolutions (v*grad_y for dL/dk uses 'full' correlation).
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// HyenaFilter: implicit long-conv kernel generator.
//
// Parameters (all learnable):
//   mlp_in_W (P, emb_dim), mlp_in_b (1, P)
//   sin_freq (1, P)                  // frequency of the sin activation
//   mlp_W[0..num_inner-1] (P, P), mlp_b[0..num_inner-1] (1, P)
//   mlp_out_W (D, P)                  // final projection to per-channel filter
//   deltas (1, D)                     // per-channel exponential decay rates
//   D_skip (1, D)                     // per-channel skip
//
// filter(L) returns (h, D_skip) where h is (L, D) and D_skip is (1, D).
// ----------------------------------------------------------------------------
class HyenaFilter {
public:
    size_t d_model;          // D — channel count
    size_t l_max;            // L — max sequence length
    size_t filter_order;     // P — hidden width of implicit MLP
    size_t emb_dim;          // emb_dim — positional embedding dim (>= 3, odd)
    size_t num_inner;        // num_inner — number of inner Linear+sin layers
    uint64_t instance_gen;   // unique generation id for cache disambiguation

    // Implicit MLP parameters
    Tensor mlp_in_W;         // (P, emb_dim)
    Tensor mlp_in_b;         // (1, P)
    Tensor sin_freq;         // (1, P) — learnable frequency of sin activation
    std::vector<Tensor> mlp_W;  // num_inner tensors, each (P, P)
    std::vector<Tensor> mlp_b;  // num_inner tensors, each (1, P)
    Tensor mlp_out_W;        // (D, P) — final projection
    Tensor deltas;           // (1, D) — exponential decay
    Tensor D_skip;           // (1, D) — skip parameter

    // Gradient buffers (matched shapes)
    Tensor grad_mlp_in_W, grad_mlp_in_b, grad_sin_freq;
    std::vector<Tensor> grad_mlp_W, grad_mlp_b;
    Tensor grad_mlp_out_W, grad_deltas, grad_D_skip;

    HyenaFilter(size_t d_model, size_t l_max, size_t filter_order = 16,
                size_t emb_dim = 3, size_t num_inner = 1);

    // Generate filter for sequence length L (L <= l_max).
    // Returns (h, D_skip) where h is (L, d_model), D_skip is (1, d_model).
    // Caches intermediate activations for backward.
    std::pair<Tensor, Tensor> filter(size_t L);

    // Backward: receives grad_h (L, D) and grad_D_skip (1, D), updates
    // grad_* on the parameters. Returns the gradient of the loss w.r.t.
    // the positional embedding (not needed since z is not learnable).
    void backward(const Tensor& grad_h, const Tensor& grad_D_skip_in);

    void zero_grad();

    // Parameter accessors for the optimizer / gradient-check framework.
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

// ----------------------------------------------------------------------------
// HyenaOperator: a single Hyena operator layer.
// Input: (B, L, d_model). Output: (B, L, d_model).
// Contains in_proj, depthwise conv1d, split, long_conv stages, out_proj.
// ----------------------------------------------------------------------------
class HyenaOperator : public Layer {
public:
    size_t d_model;          // D
    size_t l_max;            // L
    size_t order;            // number of long-conv stages (2 in canonical Hyena)
    size_t filter_order;     // P

    // in_proj: applied per-token (D → 3D). Implemented as Dense(D, 3D) and
    // applied to each token column independently.
    Dense in_proj;
    // out_proj: per-token (D → D)
    Dense out_proj;
    // short_conv: depthwise Conv1d with kernel=3, groups=(order+1)*D, causal
    //   via left padding only (pad_left=1, pad_right=0)
    Tensor short_W;          // ((order+1)*D, 3)
    Tensor short_b;          // ((order+1)*D, 1)
    // Long-conv filter parameters (one HyenaFilter shared across stages)
    HyenaFilter hyena_filter;

    // Cached forward state (for backward)
    Tensor last_input;       // (B, L, D)
    Tensor last_in_proj;     // (B, L, 3D)
    Tensor last_short;       // (B, 3D, L) post-conv
    Tensor last_g0;          // (B, D, L) gate 0
    Tensor last_g1;          // (B, D, L) gate 1
    Tensor last_v0;          // (B, D, L) input to first long-conv
    Tensor last_y0;          // (B, D, L) output of first long-conv
    Tensor last_v1;          // (B, D, L) input to second long-conv
    Tensor last_y1;          // (B, D, L) output of second long-conv (= final v)
    Tensor last_filter_h_0;  // (L, D) cached filter for stage 0
    Tensor last_filter_h_1;  // (L, D) cached filter for stage 1
    Tensor last_filter_D_0;  // (1, D) cached D_skip for stage 0
    Tensor last_filter_D_1;  // (1, D) cached D_skip for stage 1

    // Gradient buffers
    Tensor grad_short_W, grad_short_b;

    HyenaOperator(size_t d_model, size_t l_max, size_t order = 2,
                  size_t filter_order = 16);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaOperator"; }
};

// ----------------------------------------------------------------------------
// HyenaBlock: pre-LN → HyenaOperator → residual → pre-LN → FFN(GELU) → residual
// ----------------------------------------------------------------------------
class HyenaBlock : public Layer {
public:
    size_t d_model;
    size_t l_max;
    size_t order;
    size_t filter_order;
    size_t ffn_mult;        // FFN hidden width = ffn_mult * d_model

    LayerNorm ln1;
    LayerNorm ln2;
    HyenaOperator hyena;
    Dense ffn1;             // d_model → ffn_mult * d_model
    Dense ffn2;             // ffn_mult * d_model → d_model

    Tensor last_input;      // (B, L, D)

    HyenaBlock(size_t d_model, size_t l_max, size_t order = 2,
               size_t filter_order = 16, size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return hyena.in_proj.weights; }
    Tensor get_gradients() const override { return hyena.in_proj.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaBlock"; }
};

// ----------------------------------------------------------------------------
// HyenaModel: stack of HyenaBlocks + per-token mean pool + classifier.
// Input: (B, L, d_model). Output: (B, num_classes).
// ----------------------------------------------------------------------------
class HyenaModel : public Layer {
public:
    size_t d_model;
    size_t l_max;
    size_t depth;
    size_t num_classes;
    size_t order;
    size_t filter_order;

    std::vector<HyenaBlock> blocks;
    Dense classifier;       // d_model → num_classes

    Tensor last_input;

    HyenaModel(size_t d_model, size_t l_max, size_t depth, size_t num_classes,
               size_t order = 2, size_t filter_order = 16);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaModel"; }
};

#endif
