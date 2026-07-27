#include "cross_validation.h"

#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

// =================================================================
// KFolder
// =================================================================
KFolder::KFolder(size_t n_splits, bool shuffle, uint32_t random_seed)
    : n_splits_(n_splits), shuffle_(shuffle), random_seed_(random_seed) {
    if (n_splits_ < 2) {
        throw std::invalid_argument("KFolder: n_splits must be >= 2 (got " +
                                    std::to_string(n_splits_) + ")");
    }
}

namespace {
std::vector<Fold> contiguous_kfold_indices(size_t n_samples, size_t n_splits) {
    // Without shuffle: contiguous blocks of (mostly) equal size.
    // First (n_samples % n_splits) blocks have size ceil(n/n_splits); rest have floor(n/n_splits).
    std::vector<Fold> folds(n_splits);
    size_t start = 0;
    size_t base = n_samples / n_splits;
    size_t remainder = n_samples % n_splits;
    for (size_t i = 0; i < n_splits; ++i) {
        size_t sz = base + (i < remainder ? 1 : 0);
        size_t end = start + sz;
        folds[i].test_indices.reserve(sz);
        for (size_t j = start; j < end; ++j) folds[i].test_indices.push_back(j);
        for (size_t j = 0; j < n_samples; ++j) {
            if (j < start || j >= end) folds[i].train_indices.push_back(j);
        }
        start = end;
    }
    return folds;
}
}  // namespace

std::vector<Fold> KFolder::split(size_t n_samples) const {
    if (n_samples == 0) {
        throw std::invalid_argument("KFolder::split: n_samples == 0");
    }
    if (n_splits_ > n_samples) {
        throw std::invalid_argument("KFolder::split: n_splits (" +
                                    std::to_string(n_splits_) + ") > n_samples (" +
                                    std::to_string(n_samples) + ")");
    }
    if (!shuffle_) {
        return contiguous_kfold_indices(n_samples, n_splits_);
    }
    // Shuffled: build a permutation, then chunk contiguously.
    std::vector<size_t> perm(n_samples);
    std::iota(perm.begin(), perm.end(), size_t{0});
    std::mt19937 rng(random_seed_);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<Fold> folds(n_splits_);
    size_t base = n_samples / n_splits_;
    size_t remainder = n_samples % n_splits_;
    size_t start = 0;
    for (size_t i = 0; i < n_splits_; ++i) {
        size_t sz = base + (i < remainder ? 1 : 0);
        size_t end = start + sz;
        folds[i].test_indices.reserve(sz);
        for (size_t j = start; j < end; ++j) folds[i].test_indices.push_back(perm[j]);
        for (size_t j = 0; j < n_samples; ++j) {
            if (j < start || j >= end) folds[i].train_indices.push_back(perm[j]);
        }
        start = end;
    }
    return folds;
}

// =================================================================
// StratifiedKFolder
// =================================================================
StratifiedKFolder::StratifiedKFolder(size_t n_splits, bool shuffle, uint32_t random_seed)
    : n_splits_(n_splits), shuffle_(shuffle), random_seed_(random_seed) {
    if (n_splits_ < 2) {
        throw std::invalid_argument("StratifiedKFolder: n_splits must be >= 2 (got " +
                                    std::to_string(n_splits_) + ")");
    }
}

std::vector<Fold> StratifiedKFolder::split(const Tensor& labels) const {
    if (labels.rows == 0) {
        throw std::invalid_argument("StratifiedKFolder::split: labels has 0 rows");
    }
    if (labels.cols != 1) {
        throw std::invalid_argument("StratifiedKFolder::split: labels must be (N, 1) integer tensor");
    }
    // Bucket per-class indices.
    std::unordered_map<int, std::vector<size_t>> per_class;
    for (size_t i = 0; i < labels.rows; ++i) {
        const int cls = static_cast<int>(labels(i, 0));
        per_class[cls].push_back(i);
    }
    for (const auto& kv : per_class) {
        if (kv.second.size() < n_splits_) {
            throw std::invalid_argument(
                "StratifiedKFolder::split: class " + std::to_string(kv.first) +
                " has only " + std::to_string(kv.second.size()) +
                " samples, fewer than n_splits=" + std::to_string(n_splits_));
        }
    }
    // Optionally shuffle per-class index lists (deterministic via random_seed_).
    if (shuffle_) {
        std::mt19937 rng(random_seed_);
        for (auto& kv : per_class) {
            std::shuffle(kv.second.begin(), kv.second.end(), rng);
        }
    }
    // Per-class: contiguous KFolder partition.
    std::vector<std::vector<std::vector<size_t>>> per_class_folds(n_splits_);
    for (size_t f = 0; f < n_splits_; ++f) per_class_folds[f].reserve(per_class.size());
    for (auto& kv : per_class) {
        std::vector<Fold> class_folds = contiguous_kfold_indices(kv.second.size(), n_splits_);
        for (size_t f = 0; f < n_splits_; ++f) {
            std::vector<size_t> actual_indices;
            actual_indices.reserve(class_folds[f].test_indices.size());
            for (size_t j : class_folds[f].test_indices) actual_indices.push_back(kv.second[j]);
            per_class_folds[f].push_back(std::move(actual_indices));
        }
    }
    // Merge across classes: per fold, concat per-class test sets and the complement is the train.
    std::vector<Fold> folds(n_splits_);
    for (size_t f = 0; f < n_splits_; ++f) {
        for (const auto& bucket : per_class_folds[f]) {
            folds[f].test_indices.insert(folds[f].test_indices.end(),
                                         bucket.begin(), bucket.end());
        }
        std::sort(folds[f].test_indices.begin(), folds[f].test_indices.end());
        // Train = complement of test.
        std::vector<bool> in_test(labels.rows, false);
        for (size_t idx : folds[f].test_indices) in_test[idx] = true;
        for (size_t i = 0; i < labels.rows; ++i) {
            if (!in_test[i]) folds[f].train_indices.push_back(i);
        }
    }
    return folds;
}

// =================================================================
// LeaveOneOut
// =================================================================
std::vector<Fold> LeaveOneOut::split(size_t n_samples) const {
    std::vector<Fold> folds;
    folds.reserve(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        Fold f;
        f.test_indices.push_back(i);
        for (size_t j = 0; j < n_samples; ++j) {
            if (j != i) f.train_indices.push_back(j);
        }
        folds.push_back(std::move(f));
    }
    return folds;
}

// =================================================================
// slice_by_index (header-declared, defined here so cross_validate_impl can use it)
// =================================================================
Tensor slice_by_index(const Tensor& src, const std::vector<size_t>& indices) {
    Tensor out(indices.size(), src.cols);
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t row = indices[i];
        for (size_t c = 0; c < src.cols; ++c) out(i, c) = src(row, c);
    }
    return out;
}
