#ifndef BIGBIRD_H
#define BIGBIRD_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <random>

// ============================================================================
// BigBird Sparse Attention — Zaheer et al. 2020
//   "Big Bird: Transformers for Longer Sequences" (NeurIPS 2020, arXiv:2007.14062)
//
// Standard multi-head self-attention computes, for every pair (t, s):
//   attn[t, s] = softmax_t(Q_t . K_s / sqrt(d_k)) V_s
// which is O(n^2) in both memory and compute. For long sequences this is
// prohibitive.
//
// BigBird's key insight: by restricting each query t to attend to only a
// CONSTANT-size subset of positions (independent of n), we get linear-time
// attention that is still expressive enough — empirically as strong as
// full attention on long-context tasks (the paper proves Turing-completeness
// of the encoder variant with the three-component scheme below).
//
// The three components of the BigBird sparse attention mask (combined per
// row t):
//
//   1. W — WINDOW (local):
//        For each t, attend to the sliding window of `window_size` tokens
//        centered at t (clipped to [0, n)).
//
//   2. R — RANDOM:
//        For each t, attend to `num_random` randomly-chosen tokens.
//        The random pattern is fixed at construction (deterministic per-seed)
//        so gradient checks are exact.
//
//   3. G — GLOBAL:
//        `num_global` special tokens (by default, the FIRST `num_global`
//        tokens in the sequence) attend to ALL tokens (and are attended to
//        by ALL tokens). These tokens act as "hubs" that route information
//        across the whole sequence.
//
// Each query t therefore attends to at most
//   |A(t)| <= window_size + num_random + num_global
// positions, independent of n. The full attention pattern is the union of
// these three components.
//
// ----------------------------------------------------------------------------
// Why this is sub-quadratic:
//
//   Standard self-attention:  O(n^2 d)  memory + compute
//   BigBird (this impl):      O(n (w + r + g) d)   where w = window_size,
//                                                       r = num_random,
//                                                       g = num_global
//                            These are hyperparameters (default w=3, r=2,
//                            g=2), so the attention is O(n) — linear in n.
//
// ----------------------------------------------------------------------------
// Forward math (single-head):
//
//   Q = x W_q^T + b_q,   K = x W_k^T + b_k,   V = x W_v^T + b_v      (n, d)
//   scores[t, s] = Q_t . K_s / sqrt(d)                                (n, n)
//   mask[t, s] = 1 if s ∈ A(t) else 0
//   masked_scores[t, s] = scores[t, s] - big_neg * (1 - mask[t, s])
//   attn[t, s] = softmax_t(masked_scores)                              (n, n) sparse
//   head_out = attn @ V                                                 (n, d)
//   output = head_out @ W_o^T + b_o                                     (n, d)
//
// In backward, only the entries (t, s) ∈ A flow gradients — the mask is
// fixed across forward and backward (same seed, same permutation).
//
// ----------------------------------------------------------------------------
// Conventions (match Performer / GQA / LSH / AFT in this repo):
//
//   * (n, d_model) input/output — row-major, single-head
//   * Dense convention: W stored as (out, in), y = x W^T + b in forward
//   * Multi-head: build multiple BigBirdAttention layers and concat (we
//     don't split internally — simpler gradients, faster test cycle)
//   * pre-LN block pattern (pre-LN → BigBird attn → residual →
//     pre-LN → FFN → residual)
//
// Classes:
//   BigBirdAttention  — single-head BigBird sparse attention
//   BigBirdBlock      — pre-LN → BigBirdAttention → residual → pre-LN → FFN → residual
//   BigBirdModel      — stack of BigBirdBlocks + per-token classifier
// ============================================================================

class BigBirdAttention : public Layer {
public:
    size_t d_model_;        // input/output feature dim
    size_t seq_len_;        // n — sequence length (fixed at construction)
    size_t window_size_;    // w — sliding window half-width on each side
                            //   (total window = 2*window_size_ + 1)
    size_t num_random_;     // r — random tokens per query
    size_t num_global_;     // g — number of global "hub" tokens (first g positions)
    size_t d_k_;            // = d_model (single-head)

    // Learned Q/K/V/O projections (Dense convention: y = x W^T + b)
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    // Random projection (fixed at init): per query t, samples `num_random_`
    // distinct indices from [0, n) excluding t and (optionally) the global
    // tokens. Stored as `random_idx_(t, r)` for r = 0..num_random_-1.
    Tensor random_idx_;     // (n, num_random_), int values cast to double

    // BPTT cache
    Tensor last_input_;     // (n, d_model)
    Tensor last_q_;         // (n, d_k)
    Tensor last_k_;         // (n, d_k)
    Tensor last_v_;         // (n, d_k)
    Tensor last_attn_;      // (n, n) row-softmax of masked scores (full, sparse)
    Tensor last_head_out_;  // (n, d_k)
    Tensor last_output_;    // (n, d_model)

    BigBirdAttention(size_t d_model,
                     size_t seq_len,
                     size_t window_size = 3,
                     size_t num_random  = 2,
                     size_t num_global  = 2,
                     unsigned seed      = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "BigBirdAttention"; }

    // Accessors for tests
    size_t d_model()     const { return d_model_; }
    size_t seq_len()     const { return seq_len_; }
    size_t window_size() const { return window_size_; }
    size_t num_random()  const { return num_random_; }
    size_t num_global()  const { return num_global_; }
    const Tensor& random_idx() const { return random_idx_; }

    // Build the boolean sparse attention mask of shape (n, n).
    // Returns a (n, n) Tensor of doubles: 1.0 if (t, s) is in A(t) else 0.0
    Tensor attention_mask() const;

private:
    // Compute the (n, n) sparse softmax scores. Each row is a softmax over
    // only the positions in A(t). Equivalent to filling the masked-out
    // positions with a large negative number and then row-softmaxing the
    // whole (n, n) matrix — simpler and gradient-friendly.
    Tensor compute_attention(const Tensor& Q, const Tensor& K, const Tensor& V);
};

// ----------------------------------------------------------------------------
// BigBirdBlock: pre-LN → BigBirdAttention → residual → pre-LN → FFN → residual
// ----------------------------------------------------------------------------
class BigBirdBlock : public Layer {
public:
    size_t d_model_;
    BigBirdAttention attn;
    LayerNorm ln1, ln2;
    // FFN: 2 Dense layers with GELU
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

    BigBirdBlock(size_t d_model, size_t seq_len,
                 size_t window_size = 3, size_t num_random = 2,
                 size_t num_global = 2, unsigned seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::string name() const override { return "BigBirdBlock"; }
};

// ----------------------------------------------------------------------------
// BigBirdModel: stack of BigBirdBlocks + per-token classifier
// ----------------------------------------------------------------------------
class BigBirdModel : public Layer {
public:
    size_t d_model_;
    size_t n_blocks_;
    size_t out_dim_;
    std::vector<std::unique_ptr<BigBirdBlock>> blocks_;
    Tensor W_out_, b_out_;
    Tensor grad_W_out_, grad_b_out_;

    // BPTT cache
    std::vector<Tensor> block_outputs_;
    Tensor last_logits_;

    BigBirdModel(size_t d_model, size_t seq_len, size_t n_blocks, size_t out_dim,
                 size_t window_size = 3, size_t num_random = 2,
                 size_t num_global = 2, unsigned seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_out_; }
    Tensor get_gradients() const override { return grad_W_out_; }
    std::string name() const override { return "BigBirdModel"; }
};

#endif
