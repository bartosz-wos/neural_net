#ifndef DIFF_TRANSFORMER_H
#define DIFF_TRANSFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// Differential Transformer — Ye et al. 2025 (ICLR 2025)
//   "Differential Transformer"
//   https://arxiv.org/abs/2410.05258
//
// Innovation: noise-cancelling attention via the DIFFERENCE of two softmax
// attention maps. The key observation is that standard softmax attention
// distributes probability mass over many tokens, including a noisy "background"
// component. By computing two softmax attention maps from split-Q / split-K
// projections and taking their difference, the noise cancels (because the noise
// is approximately uncorrelated between the two maps) while the signal
// amplifies. The result is a sharper, more focused attention pattern.
//
// Math (per head h, head_dim = d_model / num_heads, half_dim = head_dim / 2):
//
//   Q_h = X @ W_q[h]      shape (n, head_dim)
//   K_h = X @ W_k[h]      shape (n, head_dim)
//   V_h = X @ W_v[h]      shape (n, head_dim)
//
//   K1_h = K_h[:, :half_dim]    (n, half_dim)     # split along head_dim
//   K2_h = K_h[:, half_dim:]    (n, half_dim)
//   Q1_h = Q_h[:, :half_dim]    (n, half_dim)     # Q split (no learnable params)
//   Q2_h = Q_h[:, half_dim:]    (n, half_dim)
//
//   A1_h = softmax(Q1_h @ K1_h^T / sqrt(half_dim))   (n, n)
//   A2_h = softmax(Q2_h @ K2_h^T / sqrt(half_dim))   (n, n)
//
//   λ_h       = exp(lambda_log[h]) * lambda_init     # scalar per head, ≥ 0
//   Diff_h    = A1_h - λ_h * A2_h                    (n, n)  (rows sum to 1 - λ_h, NOT 1)
//
//   O_h      = Diff_h @ V_h                          (n, head_dim)
//   out      = concat_h O_h @ W_o^T                  (n, d_model)
//
// The λ reparam is `exp(lambda_log) * lambda_init` (NOT sigmoid). At init,
// `lambda_log = 0` so `λ_h = lambda_init` (default 0.8). During training,
// lambda_log is updated freely. The Diff attention map is NOT row-normalized
// — its rows intentionally sum to `1 - λ_h`, not 1. This is the key departure
// from vanilla softmax attention: negative entries in Diff_h push attention
// AWAY from tokens that the layer has learned to consider noise.
//
// ----------------------------------------------------------------------------
// Why the row sums are NOT 1
// ----------------------------------------------------------------------------
//   row_sum(Diff_h)[t] = sum_s (A1_h[t, s] - λ_h * A2_h[t, s])
//                      = sum_s A1_h[t, s] - λ_h * sum_s A2_h[t, s]
//                      = 1.0 - λ_h * 1.0
//                      = 1 - λ_h
// This is intentional and desirable: it lets the layer output a normalized
// weight only when λ_h ≈ 0, and lets the layer produce a zero-weighted sum
// (effectively a uniform-inhibition pattern) when λ_h ≈ 1.
//
// ----------------------------------------------------------------------------
// Backward differences from vanilla softmax
// ----------------------------------------------------------------------------
//   * grad_A2 carries an extra factor of `-λ_h` (vs vanilla softmax backward
//     which has +1.0).
//   * lambda_log gradient: d_lambda_log[h] += -λ_h * sum_{t,s} A2_h[t,s] * d_Diff_h[t,s]
//     (Note: reparam is `exp`, NOT `sigmoid`, so the chain is just
//     `d_λ/d_lambda_log = lambda_init * exp(lambda_log) = λ_h`.)
//   * The K1/K2 split means `d_K = concat(d_K1, d_K2)` — no learnable params
//     for the split, just gradient routing.
//
// ----------------------------------------------------------------------------
// Conventions (matching GQA, MLA, etc.)
// ----------------------------------------------------------------------------
//   * Input/output: (n, d_model) row-major.
//   * Dense: y = X @ W^T + b, W stored as (out, in).
//   * d_model must be evenly divisible by num_heads.
//   * head_dim must be even (we split into halves).
//   * No causal mask in v1 — cleanest gradient checks; callers can wrap.
//   * num_heads=1 gives single-head diff attention (simplest test path).
//
// Classes:
//   DiffAttention         — single-layer diff attention
//   DiffTransformerBlock  — pre-LN → DiffAttention → residual → pre-LN → FFN → residual
//   DiffTransformerModel  — stack of DiffTransformerBlocks + classifier
// ============================================================================

class DiffAttention : public Layer {
public:
    // d_model:     input/output feature dim
    // num_heads:   number of attention heads (default 1)
    // lambda_init: initial scale for the per-head λ (paper default 0.8)
    DiffAttention(size_t d_model, size_t num_heads = 1, double lambda_init = 0.8);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "DiffAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t half_dim() const { return half_dim_; }
    double lambda_init() const { return lambda_init_; }
    double lambda_value(size_t h) const {
        return std::exp(lambda_log(0, h)) * lambda_init_;
    }

    // Parameter tensors (public for tests to mutate / copy)
    Dense W_q, W_k, W_v, W_o;             // (d_model, d_model) per-head stacks
    Tensor lambda_log;                     // (1, num_heads)  per-head λ log-reparam
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_lambda_log;

    // BPTT cache (public for tests to inspect Diff attention maps)
    Tensor last_input_;    // (N, d_model)
    Tensor last_Q_;        // (N, d_model)   flat per-head
    Tensor last_K_;        // (N, d_model)   flat per-head
    Tensor last_V_;        // (N, d_model)   flat per-head
    Tensor last_O_;        // (N, d_model)   pre-output-projection
    Tensor last_A1_;       // (num_heads, N, N)  softmax 1 (flattened 3D)
    Tensor last_A2_;       // (num_heads, N, N)  softmax 2 (flattened 3D)
    Tensor last_Diff_;     // (num_heads, N, N)  A1 - λ·A2
    Tensor last_lambda_;   // (1, num_heads)     λ values used in last forward

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;      // d_model / num_heads
    size_t half_dim_;      // head_dim / 2
    double lambda_init_;
    size_t N_last_;        // cached N from forward
};
// ----------------------------------------------------------------------------
// DiffTransformerBlock — pre-LN → DiffAttention → residual → pre-LN → FFN → residual
// ----------------------------------------------------------------------------
class DiffTransformerBlock : public Layer {
public:
    DiffTransformerBlock(size_t d_model, size_t num_heads = 1,
                         double lambda_init = 0.8, size_t ffn_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ffn_fc1_.get_weights(); }
    Tensor get_gradients() const override { return ffn_fc1_.get_gradients(); }
    std::string name() const override { return "DiffTransformerBlock"; }

private:
    DiffAttention attn;
    LayerNorm ln1, ln2;
    size_t d_model_;
    size_t ffn_dim_;
    Dense ffn_fc1_;          // (ffn_dim, d_model)
    Dense ffn_fc2_;          // (d_model, ffn_dim)
    Tensor last_x;
    Tensor last_ln1_out, last_attn_out, last_residual1_out;
    Tensor last_ln2_out, last_ffn_hidden, last_ffn_out;
};

// ----------------------------------------------------------------------------
// DiffTransformerModel — stack of DiffTransformerBlocks + classifier
// ----------------------------------------------------------------------------
class DiffTransformerModel : public Layer {
public:
    DiffTransformerModel(size_t input_dim, size_t d_model, size_t output_dim,
                         size_t num_blocks, size_t num_heads = 1,
                         double lambda_init = 0.8);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj.weights; }
    Tensor get_gradients() const override { return input_proj.grad_weights; }
    std::string name() const override { return "DiffTransformerModel"; }

private:
    Dense input_proj;
    std::vector<std::unique_ptr<DiffTransformerBlock>> blocks;
    Dense classifier;
};

#endif
