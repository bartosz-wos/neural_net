#ifndef NN_UTILS_KV_CACHE_H
#define NN_UTILS_KV_CACHE_H

#include "../core/tensor.h"

#include <cstddef>
#include <vector>

// ===========================================================================
// THE INFERENCE-TIME KEY/VALUE CACHE
// ===========================================================================
//
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-16):
//
//   grep -rinE '\bkv.?cache|paged.attn|prefix.cache|past_key' include/
//   -> 3 hits, ALL COMMENTS (mla.h:33, sliding_window.h:17, megalodon.h:67)
//   -> ZERO implementation across ~250 layers.
//
// **THE TRAP**: many attention layers hold `last_k_` / `last_v_`, but those
// are caches for the BACKWARD pass. A `grep -ril 'kv_cache'` hits them and
// reports the feature as PRESENT. The absence above was measured with the
// MATCHING LINES printed, not the filenames.
//
// The repo had no incremental decoding path at all: every transformer
// forward is O(T^2) over the WHOLE sequence, so autoregressive generation
// re-computes the entire prefix at every step.
//
// ---------------------------------------------------------------------------
// WHAT THIS CLASS IS, AND WHAT IT IS NOT
// ---------------------------------------------------------------------------
//
// It is a BOOKKEEPING class. It owns no attention maths: it owns the
// per-layer storage, the length, the capacity and the reset. The attention
// that CONSUMES it lives with the attention layer (see
// MultiHeadAttention::forward_cached in
// include/nn/layers/attention/transformer.h), because the maths belongs
// where the maths is.
//
// It is deliberately NOT a "smart" cache: there is no eviction policy, no
// paging, no compression. A cache that silently drops the oldest tokens
// changes the model's output, and a cache that does that BY DEFAULT is a
// trap. The capacity is a hard bound and exceeding it is an ERROR, not a
// silent truncation.
//
// ---------------------------------------------------------------------------
// THE CONTRACT
// ---------------------------------------------------------------------------
//
//   reserve(layer, d_kv)      allocate (capacity, d_kv) buffers for `layer`.
//                             Re-reserving with the SAME width is allowed and
//                             keeps the content; with a DIFFERENT width it
//                             is an error.
//   append(layer, k, v)       append n rows of keys and n rows of values.
//                             k and v are (n, d_kv).
//   size(layer)               the number of cached tokens.
//   key_buffer(layer)         the (capacity, d_kv) key buffer; the VALID rows
//                             are [0, size(layer)). The buffer is returned
//                             by reference and is NOT a copy: the consumer
//                             reads rows directly, which is the point of a
//                             preallocated cache.
//   value_buffer(layer)       the same for the values.
//   reset(layer) / reset_all() empty the layer(s). A reset layer is
//                             REUSABLE.
//   total_tokens()            the sum of the layers' lengths.
//   memory_bytes()            two buffers of capacity x d_kv doubles per layer.
//
// EVERY ERROR MESSAGE IS DISTINCT -- a substring that appears in two
// messages discriminates nothing. The tests assert the MESSAGE, not merely
// "it threw": a check that only asserts the throw is blind to a guard that
// was removed while a different guard still fires.
//
// NO CITATION IS ASSERTED. The key/value cache is an implementation
// technique rather than a paper, and this repo's expansion queue has
// produced fifteen wrong arXiv identifiers -- every one a plausible-looking
// neighbour of the right paper. The one reference that IS asserted is the
// standard "Fast Transformer Decoding: One Write-Head is All You Need"
// (Shazeer, 2019), which is a TITLE and a venue, not an identifier.
// ===========================================================================

namespace nn {
namespace utils {

class KVCache {
public:
    // `capacity` is the maximum number of tokens a layer may hold. It is a
    // HARD bound: exceeding it throws.
    explicit KVCache(int capacity = 128);

    int capacity() const { return capacity_; }
    int layers() const { return static_cast<int>(layers_.size()); }

    // Allocate the buffers for `layer`. Every layer must be reserved before
    // it is appended to.
    void reserve(int layer, int d_kv);

    // Append n rows of keys and n rows of values for `layer`. k and v are
    // (n, d_kv) tensors. Throws on: an unreserved layer; k and v with
    // different row counts; k and v with different widths; a width that
    // differs from the layer's; zero rows; a capacity overflow.
    void append(int layer, const Tensor& k, const Tensor& v);

    // Whether `layer` has been reserved. This is a QUERY, not a semantic
    // change: it exists so that a consumer can reserve a layer lazily
    // without having to catch an exception to find out.
    bool has_layer(int layer) const;

    // The number of cached tokens in `layer`.
    int size(int layer) const;

    // The width (d_kv) the layer was reserved with.
    int d_kv(int layer) const;

    // The key / value buffers. Rows [0, size(layer)) are the valid ones.
    const Tensor& key_buffer(int layer) const;
    const Tensor& value_buffer(int layer) const;

    // Empty one layer, or every layer. A reset layer is reusable.
    void reset(int layer);
    void reset_all();

    // The sum of the layers' lengths, and the bytes the cache holds.
    long long total_tokens() const;
    long long memory_bytes() const;

private:
    struct Layer {
        Tensor keys;       // (capacity, d_kv)
        Tensor values;     // (capacity, d_kv)
        int len = 0;       // the number of VALID rows
        int width = 0;     // d_kv
    };

    std::vector<Layer> layers_;
    int capacity_;
};

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_KV_CACHE_H
