# k-Fold Cross-Validation Utility Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a professional cross-validation toolkit — standard k-fold, stratified k-fold, leave-one-out, and a high-level `cross_validate` orchestrator — filling a real gap in the repo (the library has metrics, training utilities, and 30+ optimizers, but no CV helper).

**Architecture:** Stand-alone `KFolder`, `StratifiedKFolder`, `LeaveOneOut`, and free `cross_validate` function living in `include/nn/utils/cross_validation.{h,cpp}`. Operates on integer index vectors so it stays decoupled from the layer/model system. The `cross_validate` driver takes user-supplied `FitFn` and `EvalFn` callables, returning per-fold and aggregate metrics (mean ± std, plus per-fold records).

**Tech Stack:** C++17. Reuses `Tensor` only as a value-type for accumulating per-fold metric scores (so users can dump a `Tensor` of fold scores into the existing `metrics` / `early_stopping` / `training_history` machinery). No new dependencies.

**Source-of-truth for the algorithm**: scikit-learn's `model_selection.KFold`, `StratifiedKFold`, `LeaveOneOut` (BSD-3). Behavior contracts matched bit-for-bit: deterministic with `random_seed` (default-seeded `std::mt19937`), consistent `train` / `test` index sets, exhaustive coverage, and standard edge-case handling (`n_splits ≥ 2`, `n_splits ≤ n_samples`, `shuffle=false` ≡ identity splits).

---

### Task 1: Write the failing test file

**Files:**
- Create: `tests/test_cross_validation.cpp`
- Create: `include/nn/utils/cross_validation.h` (skeleton — empty namespace)
- Create: `include/nn/utils/cross_validation.cpp` (skeleton — empty `.cpp`)

**Step 1:** Write `tests/test_cross_validation.cpp` with 60+ checks across the following invariant buckets — *no production code yet*:

- **(a) KFolder defaults & validation**: n=10, n_splits=5 → 5 splits, each (train_size=8, test_size=2), indices disjoint within split, all test indices across folds cover {0..9} exactly once, n_splits=2 throws on n=1, n_splits=0 throws, n_splits=11 throws, shuffle=true with random_seed=42 produces bit-identical sequence across two instances.
- **(b) KFolder no-shuffle identity**: n=10 n_splits=5 shuffle=false → splits are (train=[2,3,4,5,6,7,8,9], test=[0,1]), then ([0,1,4..9], [2,3]), etc. — i.e. contiguous blocks.
- **(c) KFolder contiguous fold sizes**: n=11 n_splits=5 → fold sizes are [3,2,2,2,2] or [2,3,2,2,2] (round-robin), union of test sets is exactly {0..10}, no overlap.
- **(d) Stratified KFolder**: n=20, labels = 10 zeros + 10 ones, n_splits=5 → each fold's test set has exactly 2 zeros and 2 ones; all folds' test sets are disjoint, union = {0..19}. n_splits=2 with 1-sample class throws.
- **(e) LeaveOneOut**: n=7 → produces exactly 7 splits, each with train_size=6 test_size=1, every test index 0..6 appears exactly once.
- **(f) cross_validate: identity `fit_fn` / `eval_fn`**: n=12, n_splits=4, `eval_fn(test_X, test_y, fold_idx)` returns `(double)fold_idx` → mean of scores = 1.5 (mean of {0,1,2,3}), per-fold Tensor has shape (4,), std is `sqrt(2.5)`, no NaN.
- **(g) cross_validate: callback contract** — `fit_fn` and `eval_fn` invoked exactly once per fold, X_train and y_train have the right row count (n - n/n_splits), X_test and y_test have `ceil(n/n_splits)`, the union of train sets across folds covers {0..n-1} (with overlap — by design).
- **(h) cross_validate: aggregate fields**: `scores_per_fold` matches user's per-fold return, `mean_score` is the mean of those, `std_score` is the population std, `fold_count == n_splits`, `n_samples == N`, `fit_time_ms` and `eval_time_ms` are non-negative.
- **(i) cross_validate: errors / validation** — `n_splits=0` throws, empty `X` throws, `X.rows != y.rows` throws, fit_fn returning wrong train size throws (caught), `eval_fn` returning `inf` produces a non-finite aggregate and is reported (no crash).
- **(j) Determinism & RNG**: shuffled KFolder with `random_seed=123` reproduces exactly across two runs; different `random_seed` produces different fold order; KFolder with no `shuffle` flag does NOT touch the RNG.

**Step 2:** Run `make build/test_cross_validation` — expected: compile error because the header doesn't define the symbols yet.

### Task 2: Implement `KFolder` and `StratifiedKFolder`

**Files:**
- Modify: `include/nn/utils/cross_validation.h`
- Modify: `include/nn/utils/cross_validation.cpp`

Implement:

```cpp
struct Fold {
    std::vector<size_t> train_indices;  // sorted
    std::vector<size_t> test_indices;   // sorted
};

class KFolder {
public:
    explicit KFolder(size_t n_splits, bool shuffle = false, uint32_t random_seed = 0);
    std::vector<Fold> split(size_t n_samples) const;
    size_t n_splits() const { return n_splits_; }
    bool shuffle() const { return shuffle_; }
private:
    size_t n_splits_;
    bool shuffle_;
    uint32_t random_seed_;
};

class StratifiedKFolder {
public:
    explicit StratifiedKFolder(size_t n_splits, bool shuffle = false, uint32_t random_seed = 0);
    std::vector<Fold> split(const Tensor& labels) const;  // labels is (N, 1) integer
    size_t n_splits() const { return n_splits_; }
private:
    size_t n_splits_;
    bool shuffle_;
    uint32_t random_seed_;
};
```

`KFolder::split` (no-shuffle, n=11, n_splits=5): place first `n % n_splits` folds with size `n/n_splits+1`, rest with `n/n_splits`. Indices are contiguous blocks. With `shuffle=true`, Fisher-Yates shuffle the index vector (using `std::mt19937{random_seed_}`), then chunk into folds. All train/test index sets are sorted ascending for stable behavior.

`StratifiedKFolder::split` (no-shuffle): per-class, partition class indices into n_splits contiguous blocks (same contiguous-block rule as KFolder, per-class), then merge per-fold across classes by sorted-union. With `shuffle=true`, Fisher-Yates shuffle the per-class index vectors, then partition. Each fold's test set then has the same class distribution as the original. Validation: requires at least one class has ≥ n_splits members, throws `std::invalid_argument` if not.

### Task 3: Implement `LeaveOneOut` and `cross_validate`

**Files:**
- Modify: `include/nn/utils/cross_validation.h`
- Modify: `include/nn/utils/cross_validation.cpp`

```cpp
class LeaveOneOut {
public:
    std::vector<Fold> split(size_t n_samples) const;  // n_splits = n_samples
    size_t n_splits() const { return 0; }              // dynamic, depends on input
};

// High-level orchestrator
struct CrossValidateResult {
    Tensor scores_per_fold;        // (n_splits, 1) doubles
    double mean_score = 0.0;
    double std_score = 0.0;        // population std (divide by k, not k-1) — matches sklearn
    size_t fold_count = 0;
    size_t n_samples = 0;
    double fit_time_ms = 0.0;
    double eval_time_ms = 0.0;
};

template <typename FitFn, typename EvalFn>
CrossValidateResult cross_validate(
    const Tensor& X, const Tensor& y,
    const KFolder& kfolder,
    FitFn&& fit_fn,
    EvalFn&& eval_fn);
```

`cross_validate` iterates the folds, slices `X` and `y` by the train/test index sets, calls `fit_fn(X_train, y_train, fold_idx)` and `eval_fn(X_test, y_test, fold_idx)`, accumulates per-fold scores into a `Tensor`. Times each `fit_fn` and `eval_fn` call with `std::chrono::steady_clock`. Returns the aggregate struct.

**Compile-time checks**: `cross_validate` validates `X.rows == y.rows` and `X.rows > 0`; empty model returns zeros; per-fold `eval_fn` may return any value convertible to `double`. Score is the same shape across folds (we don't infer per-fold shape — the user is responsible for returning a scalar per fold, which is the common case).

### Task 4: Wire into `include/nn/nn.h`, Makefile, run tests

- Add `#include "utils/cross_validation.h"` to `include/nn/nn.h`
- Add compile rule `$(BUILD_DIR)/test_cross_validation: $(LIB_OBJS) $(BUILD_DIR)/test_cross_validation.o` to Makefile
- Add `$(BUILD_DIR)/test_cross_validation` to the `tests:` deps list
- Add `@echo "=== Running Cross-Validation Tests ===" && ./$(BUILD_DIR)/test_cross_validation` to `run_tests`
- Run `make tests` to verify it builds; run `make run_tests` to verify all tests pass

### Task 5: Mutation-testing the impl

1. Replace `mean_score = sum/k` with `mean_score = 0.0` → mean test fails.
2. Replace `std_score = sqrt(sum_sq_diff / k)` with `0.0` → std test fails.
3. Skip `fit_fn` invocation → fit_time_ms test fails (catches "was it called?").
4. Replace KFolder contiguous-block fold with `Fold{train={0..k-1}, test={k..n-1}}` always → no-shuffle identity test fails.
5. Replace StratifiedKFolder's per-class partition with random sampling without class check → class-distribution-per-fold test fails.
6. Drop the `y.rows == X.rows` check → shape-mismatch test fails.

All six mutations must be caught.

### Task 6: Commit, push, report

- `git add include/nn/utils/cross_validation.{h,cpp} tests/test_cross_validation.cpp include/nn/nn.h Makefile`
- `git commit -m "feat(utils): add k-fold, stratified k-fold, leave-one-out, and cross_validate"`
- `git push origin master`
- Append an entry to `EXPANSION_QUEUE.md` under `## Done` (or delete the pending item if the queue still listed it).
