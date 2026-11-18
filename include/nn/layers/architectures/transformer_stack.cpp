#include "transformer_stack.h"

#include <stdexcept>

// ============================================================================
// TransformerStack -- implementation.
//
// The stack is deliberately THIN. Every block of maths already exists in
// TransformerBlock; the stack's job is to (a) chain the blocks in order and
// (b) hand each of them its OWN slot of ONE shared key/value cache.
//
// It does NOT re-implement the block, and it does NOT touch the attention
// maths. A second implementation of the block inside the stack would be a
// second source of truth to keep in sync -- and this repo's own history
// says what happens when two copies of the same maths drift apart.
// ============================================================================

TransformerStack::TransformerStack(size_t d_model, size_t num_heads,
                                   size_t num_layers, size_t max_seq_len)
    : d_model_(d_model), num_heads_(num_heads) {
    if (d_model == 0) {
        throw std::invalid_argument(
            "TransformerStack: d_model must be positive, got 0");
    }
    if (num_heads == 0) {
        throw std::invalid_argument(
            "TransformerStack: num_heads must be positive, got 0");
    }
    if (num_layers == 0) {
        throw std::invalid_argument(
            "TransformerStack: num_layers must be positive, got 0");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument(
            "TransformerStack: max_seq_len must be positive, got 0");
    }

    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(d_model, num_heads);
    }

    // The stack OWNS a cache by default, so that forward_cached() works out
    // of the box and a caller who wants a shared cache can replace it.
    kv_cache_ = std::make_shared<nn::utils::KVCache>(static_cast<int>(max_seq_len));
    attach_cache_to_blocks();
}

void TransformerStack::attach_cache_to_blocks() {
    if (!kv_cache_) return;
    // SLOT i BELONGS TO BLOCK i. Reserving the slots up front means a cache
    // that is too narrow fails HERE, with a message about the slot, rather
    // than at the first decode with a message about a row count.
    for (size_t i = 0; i < blocks_.size(); ++i) {
        kv_cache_->reserve(static_cast<int>(i), static_cast<int>(d_model_));
        blocks_[i].set_kv_cache(kv_cache_, static_cast<int>(i));
    }
}

void TransformerStack::set_kv_cache(std::shared_ptr<nn::utils::KVCache> cache) {
    kv_cache_ = cache;
    attach_cache_to_blocks();
}

void TransformerStack::clear_kv_cache() {
    kv_cache_.reset();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        blocks_[i].clear_kv_cache();
    }
}

void TransformerStack::reset_cache() {
    if (kv_cache_) kv_cache_->reset_all();
}

int TransformerStack::kv_cache_layers() const {
    return kv_cache_ ? kv_cache_->layers() : 0;
}

Tensor TransformerStack::forward(const Tensor& input) {
    if (input.rows != d_model_) {
        throw std::invalid_argument(
            "TransformerStack::forward: the input must have d_model = " +
            std::to_string(d_model_) + " rows, got " +
            std::to_string(input.rows));
    }
    Tensor h = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        h = blocks_[i].forward(h);
    }
    return h;
}

Tensor TransformerStack::forward_cached(const Tensor& input) {
    if (!kv_cache_) {
        throw std::runtime_error(
            "TransformerStack::forward_cached: no KV cache is attached; "
            "call set_kv_cache() first");
    }
    if (input.rows != d_model_) {
        throw std::invalid_argument(
            "TransformerStack::forward_cached: the input must have d_model = " +
            std::to_string(d_model_) + " rows, got " +
            std::to_string(input.rows));
    }
    Tensor h = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        h = blocks_[i].forward_cached(h);
    }
    return h;
}

Tensor TransformerStack::backward(const Tensor& grad_output,
                                  double learning_rate) {
    // The blocks chain: block N-1's input gradient is block N-2's output
    // gradient. Each block's backward returns dL/d(input) in the same
    // (d_model, seq_len) layout it was given.
    Tensor g = grad_output;
    for (size_t i = blocks_.size(); i-- > 0;) {
        g = blocks_[i].backward(g, learning_rate);
    }
    return g;
}

void TransformerStack::update_weights(double learning_rate) {
    for (size_t i = 0; i < blocks_.size(); ++i) {
        blocks_[i].update_weights(learning_rate);
    }
}

Tensor TransformerStack::get_weights() const {
    return blocks_.empty() ? Tensor(0, 0) : blocks_[0].get_weights();
}

Tensor TransformerStack::get_gradients() const {
    return blocks_.empty() ? Tensor(0, 0) : blocks_[0].get_gradients();
}

std::vector<Tensor*> TransformerStack::parameters() {
    std::vector<Tensor*> out;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        std::vector<Tensor*> p = blocks_[i].parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

std::vector<Tensor*> TransformerStack::gradients() {
    std::vector<Tensor*> out;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        std::vector<Tensor*> g = blocks_[i].gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    return out;
}

void TransformerStack::zero_grad() {
    for (size_t i = 0; i < blocks_.size(); ++i) {
        blocks_[i].zero_grad();
    }
}

TransformerBlock& TransformerStack::block(size_t i) {
    if (i >= blocks_.size()) {
        throw std::out_of_range(
            "TransformerStack::block: the block index " + std::to_string(i) +
            " is out of range for a stack of " + std::to_string(blocks_.size()) +
            " blocks");
    }
    return blocks_[i];
}

const TransformerBlock& TransformerStack::block(size_t i) const {
    if (i >= blocks_.size()) {
        throw std::out_of_range(
            "TransformerStack::block: the block index " + std::to_string(i) +
            " is out of range for a stack of " + std::to_string(blocks_.size()) +
            " blocks");
    }
    return blocks_[i];
}
