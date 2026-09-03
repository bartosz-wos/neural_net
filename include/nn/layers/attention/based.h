#ifndef BASED_H
#define BASED_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Based Linear Attention — Arora et al. 2024
//   "Simple linear attention language models balance the recall-throughput
//    tradeoff" (https://arxiv.org/abs/2402.18668)
//
// The Based paper's main sequence mixer is a 2nd-order Taylor-approximation
// of softmax attention, plus a sliding-window attention (we implement only
// the linear-attention part here — the SWA half is a plain wrapper).
//
// The 2nd-order Taylor feature map is:
//
//   phi(x) = [ 1 ,  x / sqrt(sqrt(d)) ,  vec(x ⊗ x) / sqrt(2*d) ]
//   ∈ R^{1 + d + d²}
//
// where d is the input feature dim, and `vec(x ⊗ x)` is the full row-major
// flatten of the d×d outer product (i.e. d² cross-terms, INCLUDING diagonal
// — matching the reference PyTorch implementation).
//
// The polynomial identity this implements is:
//
//   phi(x)^T phi(y) = 1 + (x·y)/sqrt(d) + (x·y)^2 / (2*d)
//
// which is the 3-term Taylor expansion of exp(x·y / sqrt(d)) — the softmax
// kernel up to a constant scale. The Based paper's design uses the standard
// linear-attention trick to make this O(n·d²) instead of O(n²·d):
//
//   Q' = phi(Q)   K' = phi(K)        (n, 1+d+d²)
//   A = tril(Q' K'^T)                 (n, n)  — causal mask
//   y = A V                          (n, head_dim)
//   z = Q' · cumsum(K', dim=seq) + eps (n,)  — per-query normalizer
//   out = y / z                      (n, head_dim)
//
// The denominator trick `out = A V / (Q' · cumsum(K') + eps)` is the
// standard "linearized softmax" — `z[i]` approximates the row-sum of
// `tril(Q'_i · K'^T)`, which is the softmax normalizer.
//
// Reference implementation:
//   https://github.com/HazyResearch/based/blob/main/based/models/mixers/
//     linear_attention.py
//   (class `TaylorExp` lines 53-72, class `LinearAttention` lines 80+,
//    "quadratic" branch of parallel_forward)
//
// ----------------------------------------------------------------------------
// Conventions (matching Performer / Linformer / cosFormer in this repo)
//
//   * (n, d_model) input / output — row-major
//   * Single-head: for multi-head, call multiple BasedAttentions and concat.
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual).
//   * No bias on Q/K/V/O projections (matches paper convention).
//   * Causal mask (lower-triangular) — the Based paper's LinearAttention is
//     causal; the SWA half is local-window causal. We implement the causal
//     version here; non-causal is one mask-toggle away.
// ----------------------------------------------------------------------------

class BasedAttention : public Layer {
public:
    // d_model:     input/output feature dim
    // seq_len:     sequence length n (FIXED at construction)
    // feature_dim: Q/K projection dim (default 16 — matches paper).
    //              The feature map expands this to 1 + d + d².
    // eps:         denominator stabilizer for the softmax normalizer
    BasedAttention(size_t d_model, size_t seq_len, size_t feature_dim = 16,
                   double eps = 1e-12);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "BasedAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t feature_dim() const { return feature_dim_; }
    size_t phi_dim() const { return phi_dim_; }   // = 1 + d + d²
    double eps() const { return eps_; }

    const Tensor& last_phi_q() const { return last_phi_q_; }
    const Tensor& last_phi_k() const { return last_phi_k_; }
    const Tensor& last_A() const { return last_A_; }
    const Tensor& last_z() const { return last_z_; }

    // Learned Q/K/V/O projections: y = xW^T (no bias — matches Based paper).
    // Shapes: W_q, W_k : (d_model, feature_dim)   — projected to feature space
    //         W_v      : (d_model, d_model)      — V stays in model dim
    //         W_o      : (d_model, d_model)      — output projection
    // Public for gradient checks (matches Performer convention).
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // BPTT cache (filled in forward, consumed in backward) — public for tests
    Tensor last_input_;        // (n, d_model)
    Tensor last_q_pre_;        // (n, feature_dim)  — W_q · x
    Tensor last_k_pre_;        // (n, feature_dim)  — W_k · x
    Tensor last_v_;            // (n, d_model)      — W_v · x
    Tensor last_phi_q_;        // (n, phi_dim)
    Tensor last_phi_k_;        // (n, phi_dim)
    Tensor last_A_;            // (n, n)  — tril(phi_q · phi_k^T)
    Tensor last_Ksum_;         // (n, phi_dim) — cumsum(phi_k, dim=seq)
    Tensor last_z_;            // (n,)    — phi_q · last_Ksum + eps
    Tensor last_out_pre_;      // (n, d_model) — A · V / z   (pre-W_o)
    Tensor last_output_;       // (n, d_model) — pre-O + W_o^T

private:
    size_t d_model_;
    size_t seq_len_;
    size_t feature_dim_;
    size_t phi_dim_;          // = 1 + feature_dim_ + feature_dim_²
    double eps_;
};

// ----------------------------------------------------------------------------
// BasedBlock: pre-LN → BasedAttn → residual → pre-LN → FFN(GELU) → residual
//   Standard transformer-block layout, with Based linear attention as the
//   mixer. The FFN is a 2-layer Dense with GELU.
// ----------------------------------------------------------------------------

class BasedBlock : public Layer {
public:
    size_t d_model_;
    BasedAttention attn;
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

    BasedBlock(size_t d_model, size_t seq_len, size_t feature_dim = 16,
               size_t ffn_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::string name() const override { return "BasedBlock"; }
};

// ----------------------------------------------------------------------------
// BasedModel: stack of `num_blocks` BasedBlocks + input projection +
//   final LayerNorm + classifier. (n, d_model) → (n, out_features).
// ----------------------------------------------------------------------------

class BasedModel : public Layer {
public:
    size_t d_model_, num_blocks_, out_features_;
    std::vector<BasedBlock> blocks_;
    LayerNorm final_ln_;
    Tensor W_out_, b_out_;
    Tensor grad_W_out_, grad_b_out_;

    // Input projection (so the model can accept arbitrary in_dim, but the
    // blocks run in d_model). If in_dim == d_model the projection is identity.
    Tensor W_in_, b_in_;
    Tensor grad_W_in_, grad_b_in_;

    // BPTT cache
    Tensor last_input_;
    Tensor last_in_proj_;
    Tensor last_final_ln_;
    Tensor last_logits_;

    BasedModel(size_t in_dim, size_t d_model, size_t seq_len,
               size_t out_features, size_t num_blocks = 2,
               size_t feature_dim = 16, size_t ffn_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "BasedModel"; }
};

#endif // BASED_H
