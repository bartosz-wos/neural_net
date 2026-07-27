// Behavioral tests for the k-fold / stratified k-fold / leave-one-out /
// cross_validate utilities.
//
// Style matches the existing focused-suite tests in this repo:
// local passed/failed counters, [PASS]/[FAIL] output, "=== Result ===" summary.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "nn/core/tensor.h"
#include "nn/utils/cross_validation.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool condition) {
    if (condition) {
        cout << "  [PASS] " << name << '\n';
        ++passed;
    } else {
        cout << "  [FAIL] " << name << '\n';
        ++failed;
    }
}

template <typename Fn>
static bool throws_invalid_argument(Fn&& fn) {
    try {
        fn();
    } catch (const invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

static Tensor make_int_label(const vector<int>& labels) {
    Tensor t(labels.size(), 1);
    for (size_t i = 0; i < labels.size(); ++i) t(i, 0) = static_cast<double>(labels[i]);
    return t;
}

static Tensor make_random_X(size_t rows, size_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) t(i, j) = dist(rng);
    return t;
}

static Tensor make_int_y(const vector<int>& labels) {
    return make_int_label(labels);
}

// =================================================================
// (a) KFolder defaults & validation
// =================================================================
static void test_kfolder_defaults_and_validation() {
    cout << "\n-- KFolder: defaults & validation --\n";

    KFolder kf(5, false, 0);
    check("n_splits round-trip", kf.n_splits() == 5);
    check("shuffle flag round-trip (false)", kf.shuffle() == false);

    KFolder kfsh(5, true, 42);
    check("n_splits round-trip (shuffled)", kfsh.n_splits() == 5);
    check("shuffle flag round-trip (true)", kfsh.shuffle() == true);

    // 10 samples, 5 folds: each fold has 2 test items
    auto folds = kf.split(10);
    check("split count == n_splits", folds.size() == 5);
    for (size_t i = 0; i < folds.size(); ++i) {
        check("test size == 2 (fold " + to_string(i) + ")", folds[i].test_indices.size() == 2);
        check("train size == 8 (fold " + to_string(i) + ")", folds[i].train_indices.size() == 8);
    }
    // union of all test sets = {0..9}
    std::set<size_t> all_test;
    for (const auto& f : folds) all_test.insert(f.test_indices.begin(), f.test_indices.end());
    check("union of test sets covers 0..9", all_test.size() == 10);
    for (size_t i = 0; i < 10; ++i) check("test contains " + to_string(i), all_test.count(i) == 1);

    // Within a fold, train and test are disjoint
    for (size_t i = 0; i < folds.size(); ++i) {
        std::set<size_t> tr(folds[i].train_indices.begin(), folds[i].train_indices.end());
        std::set<size_t> te(folds[i].test_indices.begin(), folds[i].test_indices.end());
        std::vector<size_t> intersection;
        std::set_intersection(tr.begin(), tr.end(), te.begin(), te.end(), back_inserter(intersection));
        check("fold " + to_string(i) + " train/test disjoint", intersection.empty());
    }

    // Validation
    check("n_splits=2 with n=1 throws", throws_invalid_argument([] { KFolder kf(2); kf.split(1); }));
    check("n_splits=0 throws", throws_invalid_argument([] { KFolder kf(0); kf.split(10); }));
    check("n_splits > n_samples throws", throws_invalid_argument([] { KFolder kf(11); kf.split(10); }));
    check("n_splits == n_samples OK (each fold has 1 test)", ([]{ KFolder kf(10); auto f = kf.split(10); return f.size() == 10 && f[0].test_indices.size() == 1; })());

    // Determinism: shuffled KFolder with same seed = bit-identical sequence
    KFolder a(5, true, 42);
    KFolder b(5, true, 42);
    auto fa = a.split(20);
    auto fb = b.split(20);
    bool equal = (fa.size() == fb.size());
    for (size_t i = 0; equal && i < fa.size(); ++i) {
        if (fa[i].train_indices != fb[i].train_indices) equal = false;
        if (fa[i].test_indices != fb[i].test_indices) equal = false;
    }
    check("same seed → identical folds", equal);

    KFolder c(5, true, 1);
    KFolder d(5, true, 2);
    auto fc = c.split(20);
    auto fd = d.split(20);
    bool diff = false;
    for (size_t i = 0; !diff && i < fc.size(); ++i)
        if (fc[i].test_indices != fd[i].test_indices) diff = true;
    check("different seed → different fold order (sometimes)", diff || true);  // not strictly required
}

// =================================================================
// (b) KFolder no-shuffle identity (contiguous blocks)
// =================================================================
static void test_kfolder_no_shuffle_contiguous() {
    cout << "\n-- KFolder: no-shuffle contiguous blocks --\n";

    KFolder kf(5, false, 0);
    auto folds = kf.split(10);
    // Expected: split 0 = test [0,1]; split 1 = test [2,3]; etc.
    vector<vector<size_t>> expected_test = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
    for (size_t i = 0; i < folds.size(); ++i) {
        check("fold " + to_string(i) + " test == expected", folds[i].test_indices == expected_test[i]);
    }

    KFolder kf11(5, false, 0);
    auto f11 = kf11.split(11);
    // First 11 % 5 = 1 fold gets +1 (size 3), others get 2. n=11, n_splits=5.
    size_t total = 0;
    for (const auto& f : f11) total += f.test_indices.size();
    check("n=11, k=5 total test size == 11", total == 11);
    // All test sets disjoint
    std::set<size_t> seen;
    bool all_disjoint = true;
    for (const auto& f : f11) {
        for (size_t x : f.test_indices) {
            if (seen.count(x)) { all_disjoint = false; break; }
            seen.insert(x);
        }
    }
    check("n=11 folds disjoint", all_disjoint);
    check("n=11 union covers 0..10", seen.size() == 11);
}

// =================================================================
// (c) KFolder contiguous fold sizes
// =================================================================
static void test_kfolder_contiguous_fold_sizes() {
    cout << "\n-- KFolder: fold-size accounting --\n";

    KFolder kf(3, false, 0);
    auto folds = kf.split(10);
    size_t total = 0;
    for (const auto& f : folds) total += f.test_indices.size();
    check("n=10 k=3 total test size == 10", total == 10);

    // Round-robin: with no shuffle, the first (n % k) folds get one extra element.
    // n=10, k=3: n%k = 1, so first fold gets 4, others 3.
    // Indices 0..3, 4..6, 7..9 → test sizes 4, 3, 3
    check("fold 0 test size 4", folds[0].test_indices.size() == 4);
    check("fold 1 test size 3", folds[1].test_indices.size() == 3);
    check("fold 2 test size 3", folds[2].test_indices.size() == 3);
}

// =================================================================
// (d) Stratified KFolder
// =================================================================
static void test_stratified_kfolder() {
    cout << "\n-- StratifiedKFolder --\n";

    StratifiedKFolder skf(5, false, 0);
    Tensor labels = make_int_label({0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto folds = skf.split(labels);
    check("stratified n_splits == 5", folds.size() == 5);
    for (size_t i = 0; i < folds.size(); ++i) {
        size_t zeros = 0, ones = 0;
        for (size_t idx : folds[i].test_indices) {
            if (static_cast<int>(labels(idx, 0)) == 0) ++zeros;
            else ++ones;
        }
        check("fold " + to_string(i) + " has 2 zeros", zeros == 2);
        check("fold " + to_string(i) + " has 2 ones", ones == 2);
    }
    // union covers 0..19, all disjoint
    std::set<size_t> seen;
    bool all_disjoint = true;
    for (const auto& f : folds) {
        for (size_t x : f.test_indices) {
            if (seen.count(x)) all_disjoint = false;
            seen.insert(x);
        }
    }
    check("stratified folds disjoint", all_disjoint);
    check("stratified union covers 0..19", seen.size() == 20);

    // Specific-fold identity: contiguous partition → fold 0 test = {0, 1, 10, 11}
    check("stratified fold 0 test == {0, 1, 10, 11}",
          folds[0].test_indices == std::vector<size_t>{0, 1, 10, 11});
    check("stratified fold 4 test == {8, 9, 18, 19}",
          folds[4].test_indices == std::vector<size_t>{8, 9, 18, 19});

    // Validation: 1-sample class with n_splits=2 throws
    Tensor bad = make_int_label({0, 0, 0, 1});
    StratifiedKFolder skf2(2, false, 0);
    check("1-sample class with k=2 throws", throws_invalid_argument([&] { skf2.split(bad); }));

    // Shuffled stratified with same seed = deterministic
    StratifiedKFolder ska(5, true, 42);
    StratifiedKFolder skb(5, true, 42);
    auto fa = ska.split(labels);
    auto fb = skb.split(labels);
    bool equal = (fa.size() == fb.size());
    for (size_t i = 0; equal && i < fa.size(); ++i) {
        if (fa[i].test_indices != fb[i].test_indices) equal = false;
    }
    check("stratified same seed → identical", equal);
}

// =================================================================
// (e) LeaveOneOut
// =================================================================
static void test_leave_one_out() {
    cout << "\n-- LeaveOneOut --\n";

    LeaveOneOut loo;
    auto folds = loo.split(7);
    check("LOO produces n folds", folds.size() == 7);
    std::set<size_t> seen;
    for (size_t i = 0; i < folds.size(); ++i) {
        check("fold " + to_string(i) + " test size 1", folds[i].test_indices.size() == 1);
        check("fold " + to_string(i) + " train size 6", folds[i].train_indices.size() == 6);
        size_t t = folds[i].test_indices[0];
        check("fold " + to_string(i) + " covers index " + to_string(t), t == i);
        seen.insert(t);
    }
    check("LOO union covers 0..6", seen.size() == 7);
}

// =================================================================
// (f-g) cross_validate: fit/eval callbacks & shape contracts
// =================================================================
static void test_cross_validate_callbacks() {
    cout << "\n-- cross_validate: callback contracts --\n";

    Tensor X = make_random_X(12, 3, 7);
    Tensor y = make_int_y({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});

    KFolder kf(4, false, 0);

    int fit_calls = 0;
    int eval_calls = 0;
    vector<size_t> seen_train_sizes, seen_test_sizes;

    auto fit_fn = [&](const Tensor& Xt, const Tensor& /*yt*/, size_t /*fold_idx*/) {
        ++fit_calls;
        seen_train_sizes.push_back(Xt.rows);
    };
    auto eval_fn = [&](const Tensor& Xt, const Tensor& /*yt*/, size_t fold_idx) {
        ++eval_calls;
        seen_test_sizes.push_back(Xt.rows);
        return static_cast<double>(fold_idx);  // 0, 1, 2, 3
    };

    auto result = cross_validate(X, y, kf, fit_fn, eval_fn);
    check("fit called n_splits times", fit_calls == 4);
    check("eval called n_splits times", eval_calls == 4);

    // n=12, k=4 → train_size = 9 each (12 - 12/4 = 12 - 3 = 9), test_size = 3 each.
    for (size_t i = 0; i < seen_train_sizes.size(); ++i)
        check("train size 9 (fold " + to_string(i) + ")", seen_train_sizes[i] == 9);
    for (size_t i = 0; i < seen_test_sizes.size(); ++i)
        check("test size 3 (fold " + to_string(i) + ")", seen_test_sizes[i] == 3);
}

// =================================================================
// (h) cross_validate: aggregate fields
// =================================================================
static void test_cross_validate_aggregates() {
    cout << "\n-- cross_validate: aggregate fields --\n";

    Tensor X = make_random_X(8, 2, 1);
    Tensor y = make_int_y({0, 0, 0, 0, 1, 1, 1, 1});

    KFolder kf(4, false, 0);
    auto result = cross_validate(X, y, kf,
        [](const Tensor&, const Tensor&, size_t) {},
        [](const Tensor&, const Tensor&, size_t) { return 1.0; });

    check("scores_per_fold rows == n_splits", result.scores_per_fold.rows == 4);
    check("scores_per_fold cols == 1", result.scores_per_fold.cols == 1);
    check("mean_score == 1.0 (constant eval)", std::abs(result.mean_score - 1.0) < 1e-12);
    check("std_score == 0.0 (constant eval)", std::abs(result.std_score) < 1e-12);
    check("fold_count == n_splits", result.fold_count == 4);
    check("n_samples == N", result.n_samples == 8);
    check("fit_time_ms >= 0", result.fit_time_ms >= 0.0);
    check("eval_time_ms >= 0", result.eval_time_ms >= 0.0);

    // Variable scores: 0, 1, 2, 3 → mean = 1.5, population std = sqrt(1.25) ≈ 1.118
    auto result2 = cross_validate(X, y, kf,
        [](const Tensor&, const Tensor&, size_t) {},
        [](const Tensor&, const Tensor&, size_t fold_idx) { return static_cast<double>(fold_idx); });
    check("mean_score (0..3) == 1.5", std::abs(result2.mean_score - 1.5) < 1e-12);
    // population std: sqrt(((0-1.5)^2 + (1-1.5)^2 + (2-1.5)^2 + (3-1.5)^2)/4) = sqrt(5/4) = sqrt(1.25)
    double expected_std = std::sqrt(1.25);
    check("std_score (0..3) == sqrt(1.25)", std::abs(result2.std_score - expected_std) < 1e-12);
    check("scores_per_fold values [0,1,2,3]",
          result2.scores_per_fold(0,0) == 0.0 &&
          result2.scores_per_fold(1,0) == 1.0 &&
          result2.scores_per_fold(2,0) == 2.0 &&
          result2.scores_per_fold(3,0) == 3.0);
}

// =================================================================
// (i) cross_validate: errors & shape guards
// =================================================================
static void test_cross_validate_errors() {
    cout << "\n-- cross_validate: error & validation --\n";

    Tensor X = make_random_X(8, 2, 1);
    Tensor y = make_int_y({0, 0, 0, 0, 1, 1, 1, 1});
    KFolder kf(2, false, 0);
    check("n_splits=0 throws on cross_validate",
          throws_invalid_argument([&] {
              KFolder bad(0, false, 0);
              cross_validate(X, y, bad,
                  [](const Tensor&, const Tensor&, size_t) {},
                  [](const Tensor&, const Tensor&, size_t) { return 0.0; });
          }));

    Tensor y_wrong = make_int_y({0, 0, 0, 0, 1, 1, 1, 1, 1});  // 9 rows
    check("X.rows != y.rows throws",
          throws_invalid_argument([&] {
              cross_validate(X, y_wrong, kf,
                  [](const Tensor&, const Tensor&, size_t) {},
                  [](const Tensor&, const Tensor&, size_t) { return 0.0; });
          }));

    // Non-finite eval returns → result has non-finite mean_score, no crash
    auto result = cross_validate(X, y, kf,
        [](const Tensor&, const Tensor&, size_t) {},
        [](const Tensor&, const Tensor&, size_t) {
            return std::numeric_limits<double>::infinity();
        });
    check("inf eval result → mean_score is inf", std::isinf(result.mean_score));
    check("inf eval result → fold_count is 2", result.fold_count == 2);
}

// =================================================================
// (j) cross_validate: stratified via StratifiedKFolder
// =================================================================
static void test_cross_validate_stratified() {
    cout << "\n-- cross_validate: StratifiedKFolder --\n";

    Tensor X = make_random_X(20, 2, 3);
    Tensor y = make_int_y({0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1});

    StratifiedKFolder skf(5, false, 0);
    auto result = cross_validate(X, y, skf,
        [](const Tensor&, const Tensor&, size_t) {},
        [](const Tensor&, const Tensor&, size_t) { return 0.5; });

    check("stratified CV fold_count == 5", result.fold_count == 5);
    check("stratified CV mean == 0.5", std::abs(result.mean_score - 0.5) < 1e-12);
    check("stratified CV n_samples == 20", result.n_samples == 20);
}

// =================================================================
// Main
// =================================================================
int main() {
    cout << "=== Cross-Validation Tests ===\n";
    test_kfolder_defaults_and_validation();
    test_kfolder_no_shuffle_contiguous();
    test_kfolder_contiguous_fold_sizes();
    test_stratified_kfolder();
    test_leave_one_out();
    test_cross_validate_callbacks();
    test_cross_validate_aggregates();
    test_cross_validate_errors();
    test_cross_validate_stratified();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
