#include "dataloader.h"

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
                       bool shuffle, bool drop_last, unsigned seed)
    : dataset_(std::move(dataset)), batch_size_(batch_size),
      shuffle_(shuffle), drop_last_(drop_last), rng_(seed) {
    if (batch_size_ == 0) throw std::invalid_argument("batch_size must be > 0");
    size_t n = dataset_->size();
    indices_.resize(n);
    std::iota(indices_.begin(), indices_.end(), 0);
    shuffle_indices();
}

void DataLoader::shuffle_indices() {
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

void DataLoader::reset() {
    pos_ = 0;
}

bool DataLoader::has_next() const {
    if (pos_ >= indices_.size()) return false;
    if (drop_last_ && pos_ + batch_size_ > indices_.size()) return false;
    return true;
}

size_t DataLoader::batches_per_epoch() const {
    size_t n = indices_.size();
    if (drop_last_) return n / batch_size_;
    return (n + batch_size_ - 1) / batch_size_;
}

DataLoader::Batch DataLoader::next() {
    Batch batch;
    batch.batch_size = 0;

    if (!has_next()) return batch;

    size_t start = pos_;
    size_t end = std::min(pos_ + batch_size_, indices_.size());
    batch.batch_size = end - start;

    batch.data.reserve(batch.batch_size);
    batch.targets.reserve(batch.batch_size);

    for (size_t i = start; i < end; ++i) {
        size_t idx = indices_[i];
        batch.data.push_back(dataset_->get_sample(idx));
        batch.targets.push_back(dataset_->get_target(idx));
    }

    pos_ = end;
    return batch;
}