#ifndef DATALOADER_H
#define DATALOADER_H

#include "../../core/tensor.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// Abstract Dataset: returns (data, target) pair by index
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual Tensor get_sample(size_t idx) = 0;
    virtual Tensor get_target(size_t idx) = 0;
};

// TensorDataset: wraps in-memory X and y tensors as a dataset
class TensorDataset : public Dataset {
    Tensor X_;
    Tensor y_;
public:
    TensorDataset(const Tensor& X, const Tensor& y) : X_(X), y_(y) {
        if (X_.rows != y_.rows) {
            throw std::invalid_argument("TensorDataset requires X and y to have the same number of rows");
        }
    }
    size_t size() const override { return X_.rows; }
    Tensor get_sample(size_t idx) override {
        if (idx >= X_.rows) throw std::out_of_range("TensorDataset sample index out of range");
        Tensor row(1, X_.cols);
        for (size_t c = 0; c < X_.cols; ++c) row(0, c) = X_(idx, c);
        return row;
    }
    Tensor get_target(size_t idx) override {
        if (idx >= y_.rows) throw std::out_of_range("TensorDataset target index out of range");
        if (y_.cols == 1) {
            Tensor t(1, 1);
            t(0, 0) = y_(idx, 0);
            return t;
        } else {
            Tensor t(1, y_.cols);
            for (size_t c = 0; c < y_.cols; ++c) t(0, c) = y_(idx, c);
            return t;
        }
    }
};

// DataLoader: mini-batch iterator over a Dataset
class DataLoader {
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
    size_t pos_ = 0;

public:
    struct Batch {
        std::vector<Tensor> data;
        std::vector<Tensor> targets;
        size_t batch_size = 0;

        Tensor data_tensor() const;
        Tensor targets_tensor() const;
    };

    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
               bool shuffle = false, bool drop_last = false, unsigned seed = 42);

    // Returns the next batch, or empty Batch if exhausted
    Batch next();

    // Resets the iterator to the beginning of epoch
    void reset();

    // Total number of batches per epoch
    size_t batches_per_epoch() const;

    // Total dataset size
    size_t size() const { return dataset_->size(); }

    // Whether there is another batch available
    bool has_next() const;

private:
    void shuffle_indices();
};

#endif