#ifndef DATALOADER_H
#define DATALOADER_H

#include "../../core/tensor.h"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// Abstract Dataset: returns (data, target) pair by index
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    // Returns a single sample as (1, features) tensor and its target
    virtual Tensor get_sample(size_t idx) = 0;
    virtual Tensor get_target(size_t idx) = 0;
};

// Simple in-memory dataset wrapping X (N x features) and y (N x targets)
class TensorDataset : public Dataset {
    Tensor X_;
    Tensor y_;
public:
    TensorDataset(const Tensor& X, const Tensor& y) : X_(X), y_(y) {}
    size_t size() const override { return X_.rows; }
    Tensor get_sample(size_t idx) override {
        Tensor row(1, X_.cols);
        for (size_t j = 0; j < X_.cols; ++j) row[0][j] = X_[idx][j];
        return row;
    }
    Tensor get_target(size_t idx) override {
        Tensor row(1, y_.cols);
        for (size_t j = 0; j < y_.cols; ++j) row[0][j] = y_[idx][j];
        return row;
    }
    const Tensor& X() const { return X_; }
    const Tensor& y() const { return y_; }
};

// DataLoader: iterates over a Dataset in mini-batches
class DataLoader {
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
    size_t pos_ = 0;

public:
    DataLoader(Dataset& dataset, size_t batch_size, bool shuffle = false, unsigned seed = 42)
        : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), rng_(seed) {
        indices_.resize(dataset.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
    }

    // Returns {X_batch, y_batch} stacked as (batch_size, features/targets)
    // Stops early if not enough samples remain for a full batch
    bool has_next() const { return pos_ < dataset_.size(); }

    std::pair<Tensor, Tensor> next_batch() {
        // FIX (Bug 10): dataset_.get_sample(0) / get_target(0) -> use idx from indices_.
        // Also: pre-compute column sizes before the loop, not from sample 0 each time.
        size_t N = std::min(batch_size_, dataset_.size() - pos_);
        Tensor X_batch(N, dataset_.get_sample(indices_[pos_]).cols);
        Tensor y_batch(N, dataset_.get_target(indices_[pos_]).cols);
        for (size_t i = 0; i < N; ++i) {
            size_t idx = indices_[pos_ + i];
            for (size_t j = 0; j < X_batch.cols; ++j)
                X_batch[i][j] = dataset_.get_sample(idx)[0][j];
            for (size_t j = 0; j < y_batch.cols; ++j)
                y_batch[i][j] = dataset_.get_target(idx)[0][j];
        }
        pos_ += N;
        if (shuffle_ && pos_ >= dataset_.size()) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
            pos_ = 0;
        }
        return {X_batch, y_batch};
    }

    void reset() {
        pos_ = 0;
        if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
    }

    size_t batch_size() const { return batch_size_; }
    size_t remaining() const { return dataset_.size() - pos_; }
};

#endif