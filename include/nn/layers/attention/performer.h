#ifndef PERFORMER_H
#define PERFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Performer (FAVOR+) — Choromanski et al. 2021
//   "Rethinking Attention with Performers" (ICLR 2021, arXiv:2009.14794)
//
// Performer is a *linear-time* attention mechanism that approximates softmax
// attention via random Fourier features. Standard self-attention computes
//
//   out = softmax(Q K^T / sqrt(d_k)) V       (O(n^2 d))
//
// which is the dominant cost for long sequences. Performer's central insight
// is that the softmax kernel can be *unbiasedly estimated* via Bochner-
// style random features:
//
//   exp(x^T y) ≈ φ(x)^T φ(y)     for a feature map φ : R^d -> R^m
//
// and once we have this, the attention computation becomes
//
//   KV  = φ(K)^T V              (m, d)    — ONE matrix multiply
//   Z   = φ(K)^T 1              (m,)      — row-sums of φ(K)
//   out = φ(Q) KV  /  φ(Q) Z    (n, d)    — O(n m d), independent of n^2
//
// This brings the attention step to O(n m d) instead of O(n^2 d) — a huge
// speedup for long sequences (n >> m).
//
// ----------------------------------------------------------------------------
// FAVOR+ random feature map (the "positive" version from the paper)
//
// For m even, we draw m/2 random directions ω_i ~ N(0, I_d_k) and define
//
//   φ(x) = (1/√(m/2)) * exp(-||x||²/2) *  (cos(ω_i^T x),  sin(ω_i^T x))_{i=1..m/2}
//
// which is 2*(m/2) = m real features. With ω_i ~ N(0, I) (i.e. σ²=1) we
// approximate the Gaussian kernel
//
//   E[φ(x)^T φ(y)] = exp(-||x-y||²/2)
//
// and the softmax kernel is recovered as
//
//   exp(x^T y) = exp(||x||²/2) * exp(-||x-y||²/2) * exp(||y||²/2)
//             ≈ exp(||x||²/2) * φ(x)^T φ(y) * exp(||y||²/2)
//
// This is the "positive random features" form of the paper: φ(x) is
// non-negative (after the exp(−||x||²/2) factor), which is essential for
// the unbiased estimator to work with attention.
//
// The projection matrix W_prj ∈ R^{m/2 × d_k} is FIXED at construction
// (drawn once from N(0, 1)), so φ(·) is a fixed linear-then-nonlinear map.
// There are no learnable parameters in φ itself — only in W_q, W_k, W_v, W_o.
// (The paper also describes a "learnable" variant where ω_i is trained;
// we use the fixed variant for simplicity and reproducibility.)
//
// ----------------------------------------------------------------------------
// Numerical stability
//
//   * We use eps=1e-6 in the denominator floor to prevent divide-by-zero
//     when all keys are far from a query (φ(Q) @ Z would be 0).
//   * The exp(−||x||²/2) factor can underflow for very negative ||x||²
//     (rare in practice for normalized Q/K), but we don't clamp it — if
//     the user normalizes their inputs this is fine.
//   * φ(x) is computed for all n rows of Q and K at once, so the
//     `exp(-||x||²/2)` scalar per row is broadcast across the m features.
//
// ----------------------------------------------------------------------------
// API conventions (matching Linformer / Transformer in this repo)
//
//   * (n, d_model) input/output — row-major
//   * Single-head: for multi-head, call multiple PerformerAttentions and
//     concatenate. (Same convention as LinformerAttention.)
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual).
//
// PerformerAttention — the attention layer itself
// PerformerBlock     — pre-LN transformer block wrapper
// PerformerModel     — stack of blocks + classifier
// ============================================================================

class PerformerAttention : public Layer {
public:
    // d_model:    input/output feature dim
    // seq_len:    sequence length n (FIXED at construction — feature map
    //             uses a (m/2, d_k) projection, independent of n)
    // num_features: number of random features m (must be even). m=64 or 128
    //               is typical. Larger m = closer to softmax, more compute.
    PerformerAttention(size_t d_model, size_t seq_len, size_t num_features = 64);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "PerformerAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_features() const { return num_features_; }
    size_t m_half() const { return m_half_; }
    const Tensor& projection() const { return W_prj_; }  // (m/2, d_k)
    const Tensor& bias() const { return b_prj_; }        // (m/2,)

private:
    size_t d_model_;
    size_t seq_len_;
    size_t num_features_;     // m — total number of features (must be even)
    size_t m_half_;           // m/2 — number of (cos, sin) pairs
    size_t d_k_;              // = d_model (single-head)

    // Learned Q/K/V/O projections — Dense: y = xW^T + b
    // W shape: (d_model, d_model) — y = x @ W^T (matching layer.h Dense)
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    // Fixed (non-learnable) FAVOR+ projection matrix
    //   W_prj_: (m/2, d_k)  — projects x -> ω^T x  (then cos/sin)
    //   b_prj_: (m/2,)      — frequency offset (small, fixes bias for the
    //                          half-period issue described in the paper)
    Tensor W_prj_;
    Tensor b_prj_;

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;       // (n, d_model)
    Tensor last_q_;           // (n, d_k)  after W_q x + b_q
    Tensor last_k_;           // (n, d_k)  after W_k x + b_k
    Tensor last_v_;           // (n, d_k)  after W_v x + b_v
    Tensor last_phi_q_;       // (n, m)    φ(Q)
    Tensor last_phi_k_;       // (n, m)    φ(K)
    Tensor last_norm_q_;      // (n,)      ||q_t||²/2  (the per-row exp factor)
    Tensor last_norm_k_;      // (n,)      ||k_t||²/2
    Tensor last_KV_;          // (m, d)    φ(K)^T V
    Tensor last_Ksum_;        // (m,)      sum_t φ(K_t)
    Tensor last_Z_;           // (n,)      φ(Q) @ Ksum  — the per-query normalizer
    Tensor last_out_;         // (n, d)    Q_proj @ KV / Z  — pre-W_o
};

// ----------------------------------------------------------------------------
// PerformerBlock: pre-LN → PerformerAttn → residual → pre-LN → FFN → residual
//   Mirrors the standard transformer block layout. The FFN is a 2-layer
//   Dense with GELU.
// ----------------------------------------------------------------------------
class PerformerBlock : public Layer {
public:
    size_t d_model_, num_features_;
    PerformerAttention attn;
    LayerNorm ln1, ln2;
    // FFN: 2 Dense layers, GELU in between
    Tensor W1, b1, W2, b2;
    Tensor grad_W1, grad_b1, grad_W2, grad_b2;

    // BPTT cache
    Tensor last_x_;
    Tensor last_ln1_out_;
    Tensor last_attn_out_;
    Tensor last_resid1_;
    Tensor last_ln2_out_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_out_;

    PerformerBlock(size_t d_model, size_t seq_len, size_t num_features = 64);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::string name() const override { return "PerformerBlock"; }
};

// ----------------------------------------------------------------------------
// PerformerModel: stack of `num_blocks` PerformerBlocks + classifier head
//   (n, d_model) input → (n, out_features) per-token logits.
// ----------------------------------------------------------------------------
class PerformerModel : public Layer {
public:
    size_t d_model_, num_blocks_, out_features_;
    std::vector<PerformerBlock> blocks_;
    LayerNorm final_ln_;
    Tensor W_out_, b_out_;
    Tensor grad_W_out_, grad_b_out_;

    // BPTT cache for input projection + classifier
    Tensor W_in_, b_in_;
    Tensor grad_W_in_, grad_b_in_;
    Tensor last_input_;
    Tensor last_in_proj_;
    Tensor last_final_ln_;
    Tensor last_logits_;

    PerformerModel(size_t d_model, size_t seq_len, size_t out_features,
                   size_t num_blocks = 2, size_t num_features = 64);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "PerformerModel"; }
};

#endif // PERFORMER_H
