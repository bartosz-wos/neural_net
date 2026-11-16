#include "kv_cache.h"

#include <stdexcept>
#include <string>

namespace nn {
namespace utils {

namespace {

// Every error message is DISTINCT: a substring that appears in two messages
// discriminates nothing, and the tests assert the message rather than
// merely "it threw".
std::string layer_text(int layer) {
    return "layer " + std::to_string(layer);
}

}  // namespace

KVCache::KVCache(int capacity) : capacity_(capacity) {
    if (capacity <= 0) {
        throw std::invalid_argument(
            "KVCache: the capacity must be positive, got " +
            std::to_string(capacity));
    }
}

void KVCache::reserve(int layer, int d_kv) {
    if (layer < 0) {
        throw std::invalid_argument("KVCache: a layer index must be non-negative");
    }
    if (d_kv <= 0) {
        throw std::invalid_argument(
            "KVCache: the key/value width must be positive, got " +
            std::to_string(d_kv));
    }
    if (layer < static_cast<int>(layers_.size()) && layers_[layer].width != 0) {
        if (layers_[layer].width != d_kv) {
            throw std::invalid_argument(
                "KVCache: " + layer_text(layer) +
                " is already reserved with a width of " +
                std::to_string(layers_[layer].width) +
                ", so it cannot be reserved again with a width of " +
                std::to_string(d_kv));
        }
        // Re-reserving with the SAME width keeps the content.
        return;
    }
    if (layer >= static_cast<int>(layers_.size())) {
        layers_.resize(layer + 1);
    }
    Layer& l = layers_[layer];
    l.keys = Tensor::zeros(capacity_, d_kv);
    l.values = Tensor::zeros(capacity_, d_kv);
    l.len = 0;
    l.width = d_kv;
}

void KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer < 0) {
        throw std::invalid_argument("KVCache: a layer index must be non-negative");
    }
    if (layer >= static_cast<int>(layers_.size()) || layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    Layer& l = layers_[layer];
    if (k.rows != v.rows) {
        throw std::invalid_argument(
            "KVCache: the keys and the values must have the same number of "
            "rows, got " + std::to_string(k.rows) + " and " +
            std::to_string(v.rows));
    }
    if (k.cols != v.cols) {
        throw std::invalid_argument(
            "KVCache: the keys and the values must have the same number of "
            "columns, got " + std::to_string(k.cols) + " and " +
            std::to_string(v.cols));
    }
    if (static_cast<int>(k.cols) != l.width) {
        throw std::invalid_argument(
            "KVCache: the keys have " + std::to_string(k.cols) +
            " columns, but " + layer_text(layer) +
            " is reserved with a width of " + std::to_string(l.width));
    }
    if (k.rows == 0) {
        throw std::invalid_argument(
            "KVCache: cannot append zero rows to " + layer_text(layer));
    }
    if (l.len + static_cast<int>(k.rows) > capacity_) {
        throw std::invalid_argument(
            "KVCache: the capacity of " + std::to_string(capacity_) +
            " tokens is exceeded on " + layer_text(layer) + " (" +
            std::to_string(l.len) + " held, " + std::to_string(k.rows) +
            " appended)");
    }
    const int at = l.len;
    for (size_t i = 0; i < k.rows; ++i) {
        for (size_t j = 0; j < k.cols; ++j) {
            l.keys(at + static_cast<int>(i), j) = k(i, j);
            l.values(at + static_cast<int>(i), j) = v(i, j);
        }
    }
    l.len += static_cast<int>(k.rows);
}

bool KVCache::has_layer(int layer) const {
    return layer >= 0 && layer < static_cast<int>(layers_.size()) &&
           layers_[layer].width != 0;
}

int KVCache::size(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) ||
        layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    return layers_[layer].len;
}

int KVCache::d_kv(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) ||
        layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    return layers_[layer].width;
}

const Tensor& KVCache::key_buffer(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) ||
        layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    return layers_[layer].keys;
}

const Tensor& KVCache::value_buffer(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) ||
        layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    return layers_[layer].values;
}

void KVCache::reset(int layer) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) ||
        layers_[layer].width == 0) {
        throw std::invalid_argument("KVCache: " + layer_text(layer) +
                                    " is not reserved");
    }
    layers_[layer].len = 0;
}

void KVCache::reset_all() {
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].len = 0;
    }
}

long long KVCache::total_tokens() const {
    long long total = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        total += layers_[i].len;
    }
    return total;
}

long long KVCache::memory_bytes() const {
    // Two buffers of capacity x d_kv doubles per layer.
    long long bytes = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        bytes += 2LL * capacity_ * layers_[i].width * static_cast<long long>(sizeof(double));
    }
    return bytes;
}

}  // namespace utils
}  // namespace nn
