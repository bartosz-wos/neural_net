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
};

#endif