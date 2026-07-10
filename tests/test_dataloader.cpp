// DataLoader / TensorDataset tests
//
// Covers professional dataset-loader behavior:
//   1. TensorDataset returns row samples/targets and owns a deep copy.
//   2. TensorDataset rejects X/y row mismatches.
//   3. DataLoader emits deterministic in-order batches with a final partial batch.
//   4. drop_last skips incomplete final batches.
//   5. Shuffle is reproducible for equal seeds.
//   6. reset() starts a new shuffled epoch instead of replaying the same order.
//   7. Batch convenience helpers stack vectors of row tensors into dense tensors.
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "nn/layers/utility/dataloader.h"

using std::cout;
using std::endl;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, msg)                                  \
    do {                                                                       \
        double a_ = (actual);                                                   \
        double e_ = (expected);                                                 \
        if (std::fabs(a_ - e_) <= (tol)) {                                      \
            cout << "[PASS] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << endl;                                \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << " got=" << a_                          \
                 << " expected=" << e_ << " diff=" << std::fabs(a_ - e_)       \
                 << endl;                                                      \
        }                                                                      \
        ++total;                                                               \
    } while (0)

static Tensor make_features(size_t rows, size_t cols) {
    Tensor x(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            x(i, j) = 10.0 * static_cast<double>(i) + static_cast<double>(j);
        }
    }
    return x;
}

static Tensor make_targets(size_t rows, size_t cols) {
    Tensor y(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            y(i, j) = 100.0 + 10.0 * static_cast<double>(i) + static_cast<double>(j);
        }
    }
    return y;
}

static std::vector<size_t> collect_order(DataLoader& loader) {
    std::vector<size_t> order;
    while (loader.has_next()) {
        DataLoader::Batch batch = loader.next();
        for (const Tensor& sample : batch.data) {
            order.push_back(static_cast<size_t>(std::round(sample(0, 0) / 10.0)));
        }
    }
    return order;
}

static void test_tensor_dataset_rows_and_deep_copy() {
    cout << "-- Test 1: TensorDataset row extraction + deep copy --" << endl;
    Tensor X = make_features(4, 3);
    Tensor y = make_targets(4, 2);
    TensorDataset ds(X, y);

    X(2, 1) = -999.0;
    y(2, 0) = -777.0;

    CHECK(ds.size() == 4, "TensorDataset size equals X.rows");
    Tensor sample = ds.get_sample(2);
    Tensor target = ds.get_target(2);
    CHECK(sample.rows == 1 && sample.cols == 3, "get_sample returns a 1 x feature_dim row tensor");
    CHECK(target.rows == 1 && target.cols == 2, "get_target returns a 1 x target_dim row tensor");
    CHECK_NEAR(sample(0, 0), 20.0, 1e-12, "sample row preserves feature[2,0]");
    CHECK_NEAR(sample(0, 1), 21.0, 1e-12, "sample row is deep-copied from constructor input");
    CHECK_NEAR(target(0, 0), 120.0, 1e-12, "target row is deep-copied from constructor input");
}

static void test_tensor_dataset_rejects_row_mismatch() {
    cout << "-- Test 2: TensorDataset rejects row mismatch --" << endl;
    Tensor X = make_features(3, 2);
    Tensor y = make_targets(2, 1);
    bool threw = false;
    try {
        TensorDataset bad(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "X.rows != y.rows throws invalid_argument");
}

static void test_in_order_batches_with_partial_final_batch() {
    cout << "-- Test 3: in-order batching with partial final batch --" << endl;
    auto ds = std::make_shared<TensorDataset>(make_features(5, 2), make_targets(5, 1));
    DataLoader loader(ds, 2, false, false, 42);

    CHECK(loader.size() == 5, "loader.size() returns dataset size");
    CHECK(loader.batches_per_epoch() == 3, "ceil(5/2) = 3 batches when drop_last=false");

    DataLoader::Batch b1 = loader.next();
    CHECK(b1.batch_size == 2, "first batch has two samples");
    CHECK_NEAR(b1.data[0](0, 0), 0.0, 1e-12, "first batch sample 0 is dataset row 0");
    CHECK_NEAR(b1.data[1](0, 0), 10.0, 1e-12, "first batch sample 1 is dataset row 1");

    DataLoader::Batch b2 = loader.next();
    CHECK(b2.batch_size == 2, "second batch has two samples");
    CHECK_NEAR(b2.data[0](0, 0), 20.0, 1e-12, "second batch starts at dataset row 2");

    DataLoader::Batch b3 = loader.next();
    CHECK(b3.batch_size == 1, "final partial batch has one sample");
    CHECK_NEAR(b3.data[0](0, 0), 40.0, 1e-12, "final partial batch contains dataset row 4");
    CHECK(!loader.has_next(), "loader is exhausted after final partial batch");
}

static void test_drop_last_batches() {
    cout << "-- Test 4: drop_last skips incomplete final batch --" << endl;
    auto ds = std::make_shared<TensorDataset>(make_features(5, 2), make_targets(5, 1));
    DataLoader loader(ds, 2, false, true, 42);

    CHECK(loader.batches_per_epoch() == 2, "floor(5/2) = 2 batches when drop_last=true");
    DataLoader::Batch b1 = loader.next();
    DataLoader::Batch b2 = loader.next();
    CHECK(b1.batch_size == 2 && b2.batch_size == 2, "two full batches are emitted");
    CHECK(!loader.has_next(), "incomplete final sample is skipped");
}

static void test_shuffle_reproducible_for_same_seed() {
    cout << "-- Test 5: shuffle is reproducible for equal seeds --" << endl;
    auto ds1 = std::make_shared<TensorDataset>(make_features(8, 2), make_targets(8, 1));
    auto ds2 = std::make_shared<TensorDataset>(make_features(8, 2), make_targets(8, 1));
    DataLoader a(ds1, 3, true, false, 123);
    DataLoader b(ds2, 3, true, false, 123);

    std::vector<size_t> order_a = collect_order(a);
    std::vector<size_t> order_b = collect_order(b);
    CHECK(order_a == order_b, "two loaders with same seed produce identical first epoch order");

    std::set<size_t> seen(order_a.begin(), order_a.end());
    CHECK(seen.size() == 8 && order_a.size() == 8, "shuffled epoch visits every sample exactly once");
}

static void test_reset_reshuffles_when_enabled() {
    cout << "-- Test 6: reset reshuffles when shuffle=true --" << endl;
    auto ds = std::make_shared<TensorDataset>(make_features(9, 2), make_targets(9, 1));
    DataLoader loader(ds, 3, true, false, 7);

    std::vector<size_t> epoch1 = collect_order(loader);
    loader.reset();
    std::vector<size_t> epoch2 = collect_order(loader);

    std::set<size_t> seen1(epoch1.begin(), epoch1.end());
    std::set<size_t> seen2(epoch2.begin(), epoch2.end());
    CHECK(seen1.size() == 9 && seen2.size() == 9, "both shuffled epochs cover every sample");
    CHECK(epoch1 != epoch2, "reset() starts a new shuffled epoch instead of replaying epoch 1");
}

static void test_batch_stack_helpers() {
    cout << "-- Test 7: Batch tensor stacking helpers --" << endl;
    auto ds = std::make_shared<TensorDataset>(make_features(4, 3), make_targets(4, 2));
    DataLoader loader(ds, 3, false, false, 42);
    DataLoader::Batch batch = loader.next();

    Tensor X_batch = batch.data_tensor();
    Tensor y_batch = batch.targets_tensor();

    CHECK(X_batch.rows == 3 && X_batch.cols == 3, "data_tensor stacks 3 row samples into a 3 x feature_dim tensor");
    CHECK(y_batch.rows == 3 && y_batch.cols == 2, "targets_tensor stacks 3 row targets into a 3 x target_dim tensor");
    CHECK_NEAR(X_batch(2, 1), 21.0, 1e-12, "data_tensor preserves row/sample ordering");
    CHECK_NEAR(y_batch(2, 1), 121.0, 1e-12, "targets_tensor preserves target values");

    DataLoader::Batch empty;
    Tensor empty_x = empty.data_tensor();
    Tensor empty_y = empty.targets_tensor();
    CHECK(empty_x.rows == 0 && empty_x.cols == 0, "empty data batch stacks to 0 x 0 tensor");
    CHECK(empty_y.rows == 0 && empty_y.cols == 0, "empty target batch stacks to 0 x 0 tensor");
}

int main() {
    cout << "=== DataLoader / TensorDataset Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    test_tensor_dataset_rows_and_deep_copy();
    test_tensor_dataset_rejects_row_mismatch();
    test_in_order_batches_with_partial_final_batch();
    test_drop_last_batches();
    test_shuffle_reproducible_for_same_seed();
    test_reset_reshuffles_when_enabled();
    test_batch_stack_helpers();

    cout << "=== Result: " << passed << "/" << total << " checks passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
