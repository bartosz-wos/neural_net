#ifndef LSH_ATTENTION_H
#define LSH_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <random>

// ============================================================================
// LSH Attention (Reformer-style) — Kitaev, Kaiser, Levskaya 2020
//   "Reformer: The Efficient Transformer" (ICLR 2020, arXiv:2001.04451)
//
// LSH attention replaces the O(n^2) softmax(QK^T/sqrt(d))V with a
// LOCALITY-SENSITIVE HASHING (LSH) bucketing step that groups tokens with
// similar K vectors into the same bucket. Within each bucket we run a
// standard (chunked) attention, so the cost becomes O(n * log n) per layer
// (and O(n) amortized in the LSH-Chunk variant of the paper).
//
// ----------------------------------------------------------------------------
// Why this is sub-quadratic:
//
//   Standard self-attention computes, for every pair (t, s):
//     attn[t, s] = softmax_t(Q_t . K_s / sqrt(d_k)) V_s
//   which is O(n^2) in both memory and compute.
//
//   The key observation: if Q_t and K_s are very different (||Q_t - K_s||
//   is large), then exp(Q_t . K_s / sqrt(d_k)) is tiny — and we can SKIP
//   that (t, s) pair entirely. LSH attention finds "similar" K vectors
//   (and by the standard Q == K trick, also similar Q vectors) and only
//   computes attention within groups of similar tokens.
//
// ----------------------------------------------------------------------------
// Math summary (this implementation):
//
//   1. Q = x W_q^T + b_q,  K = x W_k^T + b_k,  V = x W_v^T + b_k   (n, d_k)
//   2. LSH bucket assignment for each token t:
//        bucket_t = argmax_{r in [0, R)}  hash(K_t)        R = num_buckets
//      where hash(x) = ((R . bpr_j) + (R . bpr_j) + ...) is a
//      single random-projection trick: we project x onto R - 1 random
//      directions and pick the argmax. (This is the standard
//      "angular LSH" approximation from the Reformer paper.)
//   3. Sort tokens by (bucket_t, t). Within each bucket, compute standard
//      softmax(QK^T / sqrt(d_k)) V chunk-by-chunk (per Reformer §3.2).
//   4. Apply a "sort-stability" tiebreak: secondary key is the original
//      position. This makes the sort deterministic for the gradient check.
//   5. Output: out @ W_o^T + b_o  ∈ R^{n x d_model}
//
//   For GRADIENT VERIFIABILITY (which is what we care about for tests),
//   we use the SIMPLE form: same Q/K for keys and queries (the paper's
//   trick to make attention "self-similar"), and we run attention only
//   within the same bucket, ignoring cross-bucket (which is the whole
//   point of LSH attention — the cross-bucket contributions are
//   exponentially small in d_k for the softmax kernel).
//
// ----------------------------------------------------------------------------
// Complexity:
//
//   Standard softmax attention:   O(n^2 d)  memory + compute
//   LSH attention (this impl):    O(n b d)  for bucket size b
//                                  (typically b ~ log n with R = n/b buckets)
//                                  (and O(n) * d) for the sort + the
//                                   O(n) hash table construction.
//
// ----------------------------------------------------------------------------
// Conventions (match Performer / GQA / AFT in this repo):
//
//   * (n, d_model) input/output — row-major, single-head
//   * Dense convention: W stored as (out, in), y = x W^T + b in forward.
//   * For multi-head: call multiple LSHAttentions and concatenate.
//   * Pre-LN block pattern (pre-LN → LSH attn → residual → pre-LN → FFN → residual).
//
// Classes:
//   LSHAttention       — single-head LSH attention
//   LSHAttentionBlock  — pre-LN → LSHAttention → residual → pre-LN → FFN → residual
//   LSHAttentionModel  — stack of LSHBlocks + per-token classifier
// ============================================================================

// ----------------------------------------------------------------------------
// Hash bucket
// ----------------------------------------------------------------------------
// The Reformer paper uses a "random rotation" LSH where the bucket assignment
// for each token depends on the K vector (content-based hash). This is great
// for sub-quadratic attention approximation, but makes the analytical
// gradient ill-defined: when W_k is perturbed, the K vector changes, the
// bucket assignment changes, and the "analytical" gradient (computed under
// the *original* bucket assignment) is no longer the right thing to compare
// against the numerical gradient (which uses the *perturbed* bucket
// assignment).
//
// For this implementation we provide TWO hash modes:
//
//   1. POSITIONAL HASH (default, used by the tests for grad-checkability):
//        bucket_t = t % num_buckets_
//      The hash depends only on the token's position in the sequence, not
//      on the K vector. This makes the gradient check exact (modulo the
//      sort, which is also position-based and thus also exact). This is
//      essentially "blocked local attention": each token attends to all
//      tokens in the same block of size num_buckets_.
//
//   2. CONTENT-BASED HASH (Reformer paper scheme, opt-in via
//      use_content_hash_=true in the constructor):
//        bucket_t = argmax_j (hash_proj_[j] . K_t)
//      Similar tokens end up in the same bucket, approximating softmax
//      attention. The gradient w.r.t. K is NOT well-defined in the
//      conventional sense (the Reformer paper addresses this with
//      multi-round hashing and other tricks) but the forward pass is
//      still correct and the OTHER gradients (W_q, W_v, W_o, b_q, b_v,
//      b_o) can still be verified.
// ----------------------------------------------------------------------------
class LSHAttention : public Layer {
public:
    size_t d_model_;        // input/output feature dim
    size_t seq_len_;        // n — sequence length (fixed at construction)
    size_t num_buckets_;    // R — number of LSH buckets
    size_t bucket_size_;    // max tokens per bucket
    size_t d_k_;            // = d_model (single-head)
    bool use_content_hash_; // if true, hash depends on K (Reformer-style)

    // Learned Q/K/V/O projections (Dense convention: y = x W^T + b)
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    // Fixed (non-learnable) random hash projection: R ∈ R^{d_k}
    // bucket_t = argmax_{j=0..num_buckets_-1} (R_j . K_t)  (we treat the
    // num_buckets rows of R as the R different "anchor" directions).
    Tensor hash_proj_;      // (num_buckets, d_k)

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;     // (n, d_model)
    Tensor last_q_;         // (n, d_k)  after W_q x + b_q
    Tensor last_k_;         // (n, d_k)  after W_k x + b_k
    Tensor last_v_;         // (n, d_k)  after W_v x + b_v
    Tensor last_buckets_;   // (n,)      bucket assignment for each token
    Tensor last_attn_out_;  // (n, d_k)  attention output BEFORE W_o projection
    Tensor last_scores_;    // (n, bucket_size) cached pre-softmax scores per (q_t, k_{s in bucket_t})

    // Auxiliary structures for the sort/unsort
    // sorted_idx_[t] = position in sorted order of original token t
    // We need these for the backward pass (to map gradients back to the
    // original token order).
    std::vector<size_t> sorted_idx_;   // sorted_idx_[i] = original token idx at sort position i
    std::vector<size_t> rank_;          // rank_[t] = sort position of original token t
    std::vector<size_t> bucket_starts_; // index in sorted order where each bucket starts
    std::vector<size_t> bucket_ends_;   // ...and ends (exclusive)

    LSHAttention(size_t d_model,
                 size_t seq_len,
                 size_t num_buckets = 0,    // 0 → auto = seq_len
                 size_t bucket_size = 0,    // 0 → auto = max(2, ceil(sqrt(seq_len)))
                 bool use_content_hash = false); // default: positional hash (grad-checkable)

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "LSHAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_buckets() const { return num_buckets_; }
    size_t bucket_size() const { return bucket_size_; }
    const Tensor& hash_projection() const { return hash_proj_; }
    const std::vector<size_t>& sorted_indices() const { return sorted_idx_; }
    const std::vector<size_t>& bucket_starts() const { return bucket_starts_; }
    const std::vector<size_t>& bucket_ends() const { return bucket_ends_; }

private:
    // Compute hash buckets for all n tokens given the (n, d_k) K matrix
    // Returns a (n,) tensor of bucket indices in [0, num_buckets_)
    Tensor compute_buckets(const Tensor& K) const;

    // Sort the n tokens by (bucket, original_position).
    // Fills sorted_idx_ and the bucket_starts_/bucket_ends_ arrays.
    // Reorders the (n, d_k) Q, K, V row-wise into sorted_q_, sorted_k_, sorted_v_.
    void sort_by_bucket(const Tensor& Q, const Tensor& K, const Tensor& V,
                        Tensor& sorted_q, Tensor& sorted_k, Tensor& sorted_v);

    // Run chunked attention within each bucket (assumes Q, K, V are sorted
    // by bucket). Returns (n, d_k) pre-W_o output, then we unsort.
    Tensor attend_sorted(const Tensor& sorted_q,
                         const Tensor& sorted_k,
                         const Tensor& sorted_v);

    // Unsort: take sorted-order output back to original token order.
    Tensor unsort(const Tensor& sorted_out) const;
};

// ----------------------------------------------------------------------------
// LSHBlock: pre-LN → LSHAttention → residual → pre-LN → FFN → residual
// ----------------------------------------------------------------------------
class LSHBlock : public Layer {
public:
    size_t d_model_;
    LSHAttention attn;
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

    LSHBlock(size_t d_model, size_t seq_len,
             size_t num_buckets = 0, size_t bucket_size = 0,
             bool use_content_hash = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::string name() const override { return "LSHBlock"; }
};

// ----------------------------------------------------------------------------
// LSHModel: stack of LSHBlocks + per-token classifier (Dense from d_model → out_dim)
// ----------------------------------------------------------------------------
class LSHModel : public Layer {
public:
    size_t d_model_;
    size_t n_blocks_;
    size_t out_dim_;
    std::vector<std::unique_ptr<LSHBlock>> blocks_;
    Tensor W_out_, b_out_;
    Tensor grad_W_out_, grad_b_out_;

    // BPTT cache
    std::vector<Tensor> block_outputs_;  // (n_blocks+1) entries: input + each block's output
    Tensor last_logits_;                  // (n, out_dim)

    LSHModel(size_t d_model, size_t seq_len, size_t n_blocks, size_t out_dim,
             size_t num_buckets = 0, size_t bucket_size = 0,
             bool use_content_hash = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_out_; }
    Tensor get_gradients() const override { return grad_W_out_; }
    std::string name() const override { return "LSHModel"; }
};

#endif
