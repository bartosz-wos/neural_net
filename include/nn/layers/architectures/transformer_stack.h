#ifndef NN_LAYERS_ARCHITECTURES_TRANSFORMER_STACK_H
#define NN_LAYERS_ARCHITECTURES_TRANSFORMER_STACK_H

#include "../../core/layer.h"
#include "../attention/transformer.h"
#include "../../utils/kv_cache.h"

#include <memory>
#include <string>
#include <vector>

// ============================================================================
// TransformerStack -- a stack of TransformerBlocks driven through ONE shared
// key/value cache.
//
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-17):
//
//     grep -rinE 'stack.?level|stack_decode|decode_stack|forward_cached' include/
//     -> MultiHeadAttention::forward_cached and TransformerBlock::forward_cached
//        exist; NO stack drives a sequence of blocks through ONE cache.
//
// The repo shipped a KV cache whose own documentation says it exists "so
// that ONE cache can serve a whole stack of attention layers"
// (include/nn/utils/kv_cache.h), and a block-level cached decode that was
// proven for exactly ONE block. What was never shipped -- and never
// proven -- is the thing a decoder actually needs: N blocks sharing ONE
// cache.
//
// ---------------------------------------------------------------------------
// WHY THE PER-BLOCK SLOT IS THE WHOLE POINT
// ---------------------------------------------------------------------------
//
// Each block has its OWN attention, so each block has its OWN keys and
// values. A stack that gave every block the SAME cache slot would not
// throw, would not crash, and would not look wrong: it would silently
// attend to the FIRST block's keys and values from EVERY block. Nothing in
// the type system catches that, and no amount of testing of the block-level
// path catches it either -- the block-level path only ever used one block.
//
// So this class assigns slot i to block i, and the tests assert it three
// ways: the per-slot cache lengths, the per-slot CONTENT (each block's
// cached keys must differ from the first block's), and an explicit ALIASED
// FOIL -- the answer a stack that shared one slot would produce -- which
// must DIFFER from the answer this class produces.
//
// ---------------------------------------------------------------------------
// THE INVARIANT
// ---------------------------------------------------------------------------
//
// For a causal stack, decoding token by token through the shared cache
// equals the full-sequence stack forward, TOKEN FOR TOKEN. The cached path
// performs the same arithmetic in the same order, and this repo's
// LayerNorm normalises over the FEATURE axis of each row -- so it is PER
// TOKEN -- which makes the block's tail POSITION-WISE. The ONLY
// cross-token coupling in the whole stack is the ATTENTION, which the
// cache covers exactly. (That is a property of this repo's convention, and
// it is MEASURED by the oracle, not assumed.)
//
// ---------------------------------------------------------------------------
// IT IS AN INFERENCE PATH
// ---------------------------------------------------------------------------
//
// `forward_cached()` is a decode. There is no adjoint for it, so
// `backward()` after a cached forward throws rather than silently
// producing a gradient that is wrong. The ORDINARY `forward()` is
// trainable and `backward()` chains the blocks in reverse.
// ---------------------------------------------------------------------------

class TransformerStack : public Layer {
public:
    // `max_seq_len` is the number of tokens the shared cache may hold. It is
    // a HARD bound: exceeding it throws rather than silently truncating.
    TransformerStack(size_t d_model, size_t num_heads, size_t num_layers,
                     size_t max_seq_len = 128);

    // The ORDINARY path: the whole sequence through every block, in order.
    // Trainable -- backward() chains the blocks in reverse.
    Tensor forward(const Tensor& input) override;

    // The INCREMENTAL path: only the NEW tokens, through every block, with
    // ONE shared cache. `input` is (d_model, n_new) and the output is
    // (d_model, n_new).
    Tensor forward_cached(const Tensor& input);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "TransformerStack"; }

    // -----------------------------------------------------------------------
    // THE SHARED CACHE
    // -----------------------------------------------------------------------

    // Attach ONE cache. SLOT i BELONGS TO BLOCK i -- that is the whole
    // point of this class. The slots are reserved up front, so a cache that
    // is too narrow for this stack fails immediately rather than at the
    // first decode.
    void set_kv_cache(std::shared_ptr<nn::utils::KVCache> cache);

    // Detach the cache. The blocks lose it too.
    void clear_kv_cache();

    // Empty every slot. The stack is then ready for the next sequence.
    void reset_cache();

    bool has_kv_cache() const { return kv_cache_ != nullptr; }
    std::shared_ptr<nn::utils::KVCache> kv_cache() const { return kv_cache_; }

    // The number of slots the stack reserved: one per block.
    int kv_cache_layers() const;

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t num_layers() const { return blocks_.size(); }

    // Direct access to a block, for inspection and for setting weights.
    TransformerBlock& block(size_t i);
    const TransformerBlock& block(size_t i) const;

private:
    // Give every block its own slot of the shared cache.
    void attach_cache_to_blocks();

    size_t d_model_;
    size_t num_heads_;
    std::vector<TransformerBlock> blocks_;
    std::shared_ptr<nn::utils::KVCache> kv_cache_;
};

#endif  // NN_LAYERS_ARCHITECTURES_TRANSFORMER_STACK_H
