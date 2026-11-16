#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../../core/layer.h"
#include "../../layers/normalization/layer_norm.h"
#include "../../utils/kv_cache.h"
#include <memory>
#include <vector>

class MultiHeadAttention : public Layer {
public:
    size_t d_model, num_heads, d_k;
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor last_q, last_k, last_v;   // cached Q, K, V after projection
    Tensor last_scores;              // cached attention scores for backward
    Tensor last_attn_out;
    Tensor last_attn_acc;   // pre-W_o attention output (input to the W_o matmul)
    Tensor last_x;
    size_t batch_size;

    MultiHeadAttention(size_t d_model, size_t num_heads);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    // -----------------------------------------------------------------------
    // THE INFERENCE-TIME KV CACHE (added 2026-11-16)
    //
    // `forward()` is the ORDINARY path: it attends over the whole sequence
    // it is given, in O(T^2). It is UNCHANGED by this feature.
    //
    // `forward_cached()` is the INCREMENTAL path: the input holds only the
    // NEW tokens, and the cache holds the previous tokens' keys and values.
    // It is a SEPARATE entry point on purpose -- there is no ambiguity about
    // whether an input is a full sequence or an increment, and attaching a
    // cache cannot change the ordinary forward.
    //
    // THE INVARIANT: for a causal attention, the incremental decode equals
    // the full-sequence forward, TOKEN FOR TOKEN. The tests assert it to the
    // LAST BIT, because the cached path performs the same arithmetic in the
    // same order.
    //
    // IT IS AN INFERENCE PATH. There is no adjoint for a decode, so
    // `backward()` after a cached forward throws rather than silently
    // producing a gradient that is wrong.
    // -----------------------------------------------------------------------

    // Attach a cache. `layer_index` selects this layer's slot in the cache,
    // so that ONE cache can serve a whole stack of attention layers.
    void set_kv_cache(std::shared_ptr<nn::utils::KVCache> cache,
                      int layer_index = 0);
    void clear_kv_cache();
    bool has_kv_cache() const { return kv_cache_ != nullptr; }
    int kv_cache_layer_index() const { return kv_layer_index_; }

    // The incremental forward. `input` is (d_model, n_new) and the output is
    // (d_model, n_new).
    Tensor forward_cached(const Tensor& input);

private:
    // The Q/K/V projection, shared by the ordinary and the cached path so
    // that the two cannot drift apart. `x_tokens` is (tokens, d_model).
    void project_qkv(const Tensor& x_tokens, Tensor& Q, Tensor& K, Tensor& V) const;

    std::shared_ptr<nn::utils::KVCache> kv_cache_;
    int kv_layer_index_ = 0;
    bool last_forward_was_cached_ = false;
};

class TransformerBlock : public Layer {
public:
    size_t d_model, num_heads;
    MultiHeadAttention attn;
    LayerNorm ln1, ln2;
    Tensor W1, b1, W2, b2;
    Tensor grad_W1, grad_b1, grad_W2, grad_b2;
    Tensor last_x, last_attn_out, last_ffn_out, last_ffn_pregelu;

    TransformerBlock(size_t d_model, size_t num_heads);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    // -----------------------------------------------------------------------
    // THE BLOCK-LEVEL CACHED DECODE (added 2026-11-16)
    //
    // `forward()` is the ORDINARY path: it attends over the whole sequence
    // it is given, in O(T^2). It is UNCHANGED by this feature.
    //
    // `forward_cached()` is the INCREMENTAL path: the input holds only the
    // NEW tokens, and the attention's key/value cache holds the previous
    // tokens'. It is a SEPARATE entry point on purpose -- there is no
    // ambiguity about whether an input is a full sequence or an increment.
    //
    // THE INVARIANT: for a causal block, the incremental decode equals the
    // full-sequence forward, TOKEN FOR TOKEN, and the tests assert it to
    // the LAST BIT -- because the cached path performs the same arithmetic
    // in the same order, and the block's post-attention tail is ROW-WISE.
    //
    // **NO RESIDUAL-STREAM CACHE IS NEEDED, AND THAT IS A MEASURED FACT,
    // NOT AN ASSUMPTION.** This repo's LayerNorm normalises over the
    // FEATURE axis of each row (see layer_norm.cpp), so it is PER TOKEN,
    // and the block's tail -- residual -> ln1 -> FFN -> + -> ln2 -- is
    // POSITION-WISE. The ONLY cross-token coupling in the whole block is
    // the ATTENTION, which the key/value cache already covers exactly.
    // Measured: the tail computed on the last two tokens alone equals the
    // full sequence's rows to 0.0, while the COUNTERFACTUAL (a LayerNorm
    // over the token axis) differs by 1.54 -- so the property is a
    // property of this repo's convention, not a tautology. A second cache
    // would be a second source of truth to keep in sync for zero
    // behavioural difference.
    //
    // IT IS AN INFERENCE PATH. There is no adjoint for a decode, so
    // `backward()` after a cached forward throws rather than silently
    // producing a gradient that is wrong.
    // -----------------------------------------------------------------------

    // Attach a cache. `layer_index` selects this block's slot in the cache,
    // so that ONE cache can serve a whole stack of blocks.
    void set_kv_cache(std::shared_ptr<nn::utils::KVCache> cache,
                      int layer_index = 0);
    void clear_kv_cache();
    bool has_kv_cache() const { return attn.has_kv_cache(); }
    int kv_cache_layer_index() const { return attn.kv_cache_layer_index(); }

    // The incremental forward. `input` is (d_model, n_new) and the output is
    // (d_model, n_new).
    Tensor forward_cached(const Tensor& input);

private:
    // The block's POST-ATTENTION tail, shared by the ordinary and the
    // cached path so that the two cannot drift apart.
    //
    //   residual = x + attn
    //   norm1    = LayerNorm(residual)
    //   ffn      = W2 @ GELU(W1 @ norm1 + b1) + b2
    //   out      = LayerNorm(norm1 + ffn)
    //
    // `x_tokens` and `attn_tokens` are (tokens, d_model) -- the block's
    // internal layout. The loop bodies are the ones that used to live
    // inside forward(); they were moved here VERBATIM so that the ordinary
    // path's arithmetic is bit-for-bit what it was.
    Tensor compute_tail(const Tensor& x_tokens, const Tensor& attn_tokens);

    bool last_forward_was_cached_ = false;
};

class PositionalEncoding : public Layer {
public:
    Tensor pe;
    PositionalEncoding(size_t max_len, size_t d_model);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return pe; }
    Tensor get_gradients() const override { return pe; }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}

    // -----------------------------------------------------------------------
    // THE ABSOLUTE-POSITION FORM (added 2026-11-16)
    //
    // `forward()` uses the tensor's LOCAL COLUMN INDEX as the position:
    //
    //     output[f][s] = input[f][s] + pe[s][f]
    //
    // That is correct for a WHOLE sequence, where column s IS token s. It
    // is WRONG for an incremental decode, where the caller hands over only
    // the NEW tokens: column 0 is then the new token, whose absolute
    // position is `past`, not 0. A decode built on the local form would be
    // SILENTLY MIS-POSITIONED -- a wrong answer with no error, which is
    // the worst kind of defect.
    //
    // `forward_at(input, offset)` uses the ABSOLUTE position `offset + s`:
    //
    //     output[f][s] = input[f][s] + pe[offset + s][f]
    //
    // `forward(input)` is exactly `forward_at(input, 0)`.
    //
    // TWO LATENT DEFECTS IN THE SAME SIX LINES ARE FIXED BY THIS CHANGE,
    // and both were silent:
    //
    //  1. `input.rows` was NEVER checked against `d_model`. `pe[s][f]`
    //     reads `pe`'s columns with `f` running to `input.rows-1`, and
    //     `pe` has `d_model` columns -- so a caller with more rows than
    //     `d_model` read OUT OF BOUNDS.
    //  2. Positions past `max_len` were SILENTLY ZEROED. The output
    //     tensor is value-initialised to 0.0 and the loop stops at
    //     `max_len`, so the tail of an over-long sequence had its token
    //     content DESTROYED with no error.
    //
    // Both are now explicit, DISTINCT errors.
    // -----------------------------------------------------------------------
    Tensor forward_at(const Tensor& input, size_t offset);
};

#endif