// Cross-validation utilities: k-fold, stratified k-fold, leave-one-out, and a
// high-level cross_validate driver. See cross_validation.cpp for the
// implementation and the docs/plans/2026-07-27-cross-validation.md plan for
// the API contract.

#ifndef NN_UTILS_CROSS_VALIDATION_H
#define NN_UTILS_CROSS_VALIDATION_H

#include "../core/tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

struct Fold {
    std::vector<size_t> train_indices;
    std::vector<size_t> test_indices;
};

class KFolder {
public:
    explicit KFolder(size_t n_splits, bool shuffle = false, uint32_t random_seed = 0);
    std::vector<Fold> split(size_t n_samples) const;
    size_t n_splits() const { return n_splits_; }
    bool shuffle() const { return shuffle_; }
    uint32_t random_seed() const { return random_seed_; }
private:
    size_t n_splits_;
    bool shuffle_;
    uint32_t random_seed_;
};

class StratifiedKFolder {
public:
    explicit StratifiedKFolder(size_t n_splits, bool shuffle = false, uint32_t random_seed = 0);
    std::vector<Fold> split(const Tensor& labels) const;
    size_t n_splits() const { return n_splits_; }
    bool shuffle() const { return shuffle_; }
    uint32_t random_seed() const { return random_seed_; }
private:
    size_t n_splits_;
    bool shuffle_;
    uint32_t random_seed_;
};

class LeaveOneOut {
public:
    std::vector<Fold> split(size_t n_samples) const;
    size_t n_splits(size_t n_samples) const { return n_samples; }
};

struct CrossValidateResult {
    Tensor scores_per_fold;
    double mean_score = 0.0;
    double std_score = 0.0;
    size_t fold_count = 0;
    size_t n_samples = 0;
    double fit_time_ms = 0.0;
    double eval_time_ms = 0.0;
};

// Internal: slice a Tensor by integer indices into a new (n, cols) Tensor.
Tensor slice_by_index(const Tensor& src, const std::vector<size_t>& indices);

// Internal: run the cross_validate driver given a pre-computed vector of folds.
// Used by both the KFolder- and StratifiedKFolder-overloaded public templates.
template <typename FitFn, typename EvalFn>
CrossValidateResult cross_validate_impl(
    const Tensor& X, const Tensor& y,
    const std::vector<Fold>& folds,
    size_t n_samples,
    FitFn&& fit_fn,
    EvalFn&& eval_fn) {
    if (X.rows != y.rows)
        throw std::invalid_argument("cross_validate: X.rows != y.rows");
    if (X.rows == 0)
        throw std::invalid_argument("cross_validate: X has 0 rows");
    if (folds.empty())
        throw std::invalid_argument("cross_validate: no folds");

    CrossValidateResult result;
    result.scores_per_fold = Tensor(folds.size(), 1);
    result.fold_count = folds.size();
    result.n_samples = n_samples;

    double sum = 0.0;
    for (size_t i = 0; i < folds.size(); ++i) {
        Tensor X_train = slice_by_index(X, folds[i].train_indices);
        Tensor y_train = slice_by_index(y, folds[i].train_indices);
        Tensor X_test  = slice_by_index(X, folds[i].test_indices);
        Tensor y_test  = slice_by_index(y, folds[i].test_indices);

        auto fit_t0 = std::chrono::steady_clock::now();
        fit_fn(X_train, y_train, i);
        auto fit_t1 = std::chrono::steady_clock::now();
        result.fit_time_ms += std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count();

        auto eval_t0 = std::chrono::steady_clock::now();
        double s = static_cast<double>(eval_fn(X_test, y_test, i));
        auto eval_t1 = std::chrono::steady_clock::now();
        result.eval_time_ms += std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count();

        result.scores_per_fold(i, 0) = s;
        sum += s;
    }

    const double k = static_cast<double>(folds.size());
    const double mean = sum / k;
    // population std (matches scikit-learn's `cross_val_score` convention).
    // Numerically stable two-pass formula: variance = (1/k) * sum_i (s_i - mean)^2
    // (avoids catastrophic cancellation of the naive sum_sq/k - mean^2 form).
    double var = 0.0;
    for (size_t i = 0; i < folds.size(); ++i) {
        const double d = result.scores_per_fold(i, 0) - mean;
        var += d * d;
    }
    var /= k;
    result.mean_score = mean;
    result.std_score = std::sqrt(std::max(0.0, var));
    return result;
}

// Public template overload for KFolder.
template <typename FitFn, typename EvalFn>
CrossValidateResult cross_validate(
    const Tensor& X, const Tensor& y,
    const KFolder& kfolder,
    FitFn&& fit_fn,
    EvalFn&& eval_fn) {
    if (kfolder.n_splits() == 0)
        throw std::invalid_argument("cross_validate: KFolder n_splits == 0");
    auto folds = kfolder.split(X.rows);
    return cross_validate_impl(X, y, folds, X.rows,
                                std::forward<FitFn>(fit_fn),
                                std::forward<EvalFn>(eval_fn));
}

// Public template overload for StratifiedKFolder.
template <typename FitFn, typename EvalFn>
CrossValidateResult cross_validate(
    const Tensor& X, const Tensor& y,
    const StratifiedKFolder& kfolder,
    FitFn&& fit_fn,
    EvalFn&& eval_fn) {
    if (kfolder.n_splits() == 0)
        throw std::invalid_argument("cross_validate: StratifiedKFolder n_splits == 0");
    auto folds = kfolder.split(y);
    return cross_validate_impl(X, y, folds, X.rows,
                                std::forward<FitFn>(fit_fn),
                                std::forward<EvalFn>(eval_fn));
}

#endif
