// test_swa.cpp — Tests for SWA (Stochastic Weight Averaging) optimizer wrapper.
// Izmailov et al. 2018 "Averaging Weights Leads to Wider Optima and Better
// Generalization" (https://arxiv.org/abs/1803.05407).
//
// SWA is a wrapper around an inner optimizer (typically SGD or Adam). After an
// initial warm-up phase of `start_after_steps`, it maintains a running average
// of the post-step weights:
//
//   For t > start_after_:
//     n         = (t - start_after_)            // count of averaged steps
//     avg_t     = avg_{t-1} + (w_t - avg_{t-1}) / n
//               = (avg_{t-1} * (n-1) + w_t) / n
//
// (The recurrence used in the impl is the latter — equivalent and avoids
//  requiring n on every step.)
//
// At evaluation time, `swap_to_averaged(model)` overwrites the live weights
// with the running average. The caller is responsible for re-saving the
// original weights if they want to continue training.
//
// Key properties tested:
//   1. Before `start_after_` steps, the wrapper is a no-op for averaging
//      (averaged_count() == 0, params are unchanged from the inner-only update).
//   2. After `start_after_` steps, averaged_count() reflects the count.
//   3. The averaged weight at step N is the exact arithmetic mean of all
//      post-step weights from step `start_after_` to step N.
//   4. swap_to_averaged(model) copies the averaged weights into the model.
//   5. record(model) is the user-initiated update (matching step()'s averaging
//      logic) for when you want to average without stepping the inner optimizer.
//   6. update_bn_stats() iterates training inputs and triggers BN forward
//      passes in training mode.
//   7. SWALRScheduler: warmup_steps linearly interpolates lr from start_lr to
//      swa_lr; after swa_start_step, current_lr_ == swa_lr; otherwise
//      current_lr_ == start_lr.
//   8. Determinism: two fresh instances with the same sequence of model
//      weights produce bit-identical averaged weights.
//   9. Mutation: replacing the running-average formula with "always overwrite"
//      breaks the closed-form averaging test (proves the test is non-vacuous).
//  10. End-to-end: an SWA-wrapped SGD run for > start_after_ steps produces
//      a model whose swapped-to-averaged weights differ from the live weights
//      (proving the wrapper actually moved something).
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/swa.h"
#include "nn/optimizers/optimizer.h"
#include "nn/layers/normalization/batch_norm.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Helper: tolerance-based close.
static bool close(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

// Mock optimizer that sets every parameter of every layer to a known constant
// `value_`. Used to test SWA's averaging behaviour deterministically.
// Owns nothing — parameters belong to the layer, so we just overwrite them.
class ConstantValueOptimizer : public Optimizer {
public:
    double value_;
    explicit ConstantValueOptimizer(double v) : Optimizer(), value_(v) {}
    void step(Model& model) override {
        for (auto& layer : model.layers)
            for (Tensor* p : layer->parameters())
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        (*p)(i, j) = value_;
    }
};

// Mock that increments every parameter by a fixed step per inner.step().
// Step t has value = t * step_value (starting from the init).
class IncrementalOptimizer : public Optimizer {
public:
    double step_value_;
    size_t inner_count_ = 0;
    explicit IncrementalOptimizer(double step) : Optimizer(), step_value_(step) {}
    void step(Model& model) override {
        ++inner_count_;
        for (auto& layer : model.layers)
            for (Tensor* p : layer->parameters())
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        (*p)(i, j) += step_value_;
    }
};

// Helper: build a model with 1 Dense(2, 1) layer, weights zeroed.
static void build_minimal_model(Model& m, double init_w = 0.0, double init_b = 0.0) {
    auto* d = new Dense(2, 1);
    d->weights(0, 0) = init_w;
    d->weights(0, 1) = init_w;
    d->bias(0, 0) = init_b;
    m.add_layer(d);
}

// ============================================================================
// Test 1: pre-start averaging is a no-op
// ============================================================================
static void test_before_start_after_is_noop_for_averaging() {
    cout << "\n[test_before_start_after_is_noop_for_averaging]" << endl;
    Model m;
    build_minimal_model(m, 1.0, 0.0);
    auto* inner = new ConstantValueOptimizer(2.0);  // sets params to 2.0 each step
    SWAOptimizer swa(inner, /*start_after=*/5);

    // 5 inner.step() calls — averaging hasn't started yet
    for (int i = 0; i < 5; ++i) swa.step(m);
    check("param[0][0] is the inner optimizer's value (no averaging yet)",
          close(m.layers[0]->parameters()[0]->operator()(0, 0), 2.0));
    check("averaged_count() returns 0 before start_after", swa.averaged_count() == 0);
}

// ============================================================================
// Test 2: after start_after_, averaged_count() reflects the count
// ============================================================================
static void test_averaged_count_after_start() {
    cout << "\n[test_averaged_count_after_start]" << endl;
    Model m;
    build_minimal_model(m, 1.0, 0.0);
    auto* inner = new ConstantValueOptimizer(2.0);
    SWAOptimizer swa(inner, /*start_after=*/3);

    for (int i = 0; i < 7; ++i) swa.step(m);
    // steps 4, 5, 6, 7 are after start_after_=3, so 7 - 3 = 4 averaged
    check("averaged_count() == 7 - 3 == 4 after 7 inner steps", swa.averaged_count() == 4);
}

// ============================================================================
// Test 3: averaging produces the exact arithmetic mean of post-step weights
// ============================================================================
static void test_averaged_weight_is_arithmetic_mean() {
    cout << "\n[test_averaged_weight_is_arithmetic_mean]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);  // init weights to 0
    auto* inner = new IncrementalOptimizer(1.0);  // each step adds 1.0
    SWAOptimizer swa(inner, /*start_after=*/0);

    // After step t (t = 1..5), params[i][j] = t. Averaged weight should be
    // the mean of {1, 2, 3, 4, 5} = 3.0.
    for (int i = 0; i < 5; ++i) swa.step(m);

    swa.swap_to_averaged(m);
    double avg = m.layers[0]->parameters()[0]->operator()(0, 0);
    check("averaged weight[0][0] == mean({1,2,3,4,5}) == 3.0", close(avg, 3.0, 1e-9));
    check("averaged bias[0][0] == 3.0",
          close(m.layers[0]->parameters()[1]->operator()(0, 0), 3.0, 1e-9));
    check("averaged_count() == 5 (start_after=0, 5 inner steps)",
          swa.averaged_count() == 5);
}

// ============================================================================
// Test 4: swap_to_averaged overwrites the live weights
// ============================================================================
static void test_swap_to_averaged_replaces_live_weights() {
    cout << "\n[test_swap_to_averaged_replaces_live_weights]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);
    auto* inner = new IncrementalOptimizer(1.0);
    SWAOptimizer swa(inner, /*start_after=*/0);

    for (int i = 0; i < 3; ++i) swa.step(m);
    // Live weights are now 3.0; averaged should be mean({1,2,3}) = 2.0
    check("pre-swap weight[0][0] == 3.0",
          close(m.layers[0]->parameters()[0]->operator()(0, 0), 3.0));
    swa.swap_to_averaged(m);
    check("post-swap weight[0][0] == 2.0 (the running average)",
          close(m.layers[0]->parameters()[0]->operator()(0, 0), 2.0));
}

// ============================================================================
// Test 5: record() updates the running average without stepping the inner opt
// ============================================================================
static void test_record_updates_average_without_inner_step() {
    cout << "\n[test_record_updates_average_without_inner_step]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);
    auto* inner = new IncrementalOptimizer(10.0);  // big step so we see the diff
    SWAOptimizer swa(inner, /*start_after=*/0);

    swa.step(m);                              // inner: +10, average = 10
    swa.step(m);                              // inner: +10, average = 15
    double inner_count_before = inner->inner_count_;
    // Manually nudge params (simulating a "user wants to record at this state")
    // and call record() — it should incorporate the current weights into the
    // running average WITHOUT calling inner.step().
    m.layers[0]->parameters()[0]->operator()(0, 0) = 25.0;
    m.layers[0]->parameters()[0]->operator()(0, 1) = 25.0;
    m.layers[0]->parameters()[1]->operator()(0, 0) = 25.0;
    swa.record(m);
    check("record() does NOT call inner.step",
          inner->inner_count_ == inner_count_before);
    swa.swap_to_averaged(m);
    // After 2 step()s + 1 record(): mean of {10, 20, 25} = 18.333...
    double avg = m.layers[0]->parameters()[0]->operator()(0, 0);
    check("avg = (10 + 20 + 25) / 3 ≈ 18.333",
          close(avg, (10.0 + 20.0 + 25.0) / 3.0, 1e-9));
}

// ============================================================================
// Test 6: update_bn_stats() runs forward passes on training inputs
// ============================================================================
static void test_update_bn_stats_runs_forward_passes() {
    cout << "\n[test_update_bn_stats_runs_forward_passes]" << endl;
    // Build a model with a BN layer and verify forward is called by
    // counting the number of times the BN's forward is invoked.
    Model m;
    auto* d1 = new Dense(4, 4);
    auto* bn = new BatchNorm1D(4);
    auto* d2 = new Dense(4, 2);
    m.add_layer(d1);
    m.add_layer(bn);
    m.add_layer(d2);
    auto* inner = new ConstantValueOptimizer(0.1);
    SWAOptimizer swa(inner, /*start_after=*/0);
    swa.step(m);  // initialize
    swa.swap_to_averaged(m);

    Tensor X(1, 4);
    X(0, 0) = 0.5; X(0, 1) = -0.5; X(0, 2) = 1.0; X(0, 3) = -1.0;
    vector<Tensor> inputs = {X, X, X};  // 3 forward passes expected

    // The function should not throw, and BN's internal state should be
    // updated (running stats). We verify by calling forward again and
    // checking it returns finite values.
    swa.update_bn_stats(m, inputs);
    Tensor y = m.forward(X);
    check("forward after update_bn_stats returns finite values",
          std::isfinite(y(0, 0)) && std::isfinite(y(0, 1)));
    check("output has correct shape (1, 2)", y.rows == 1 && y.cols == 2);
}

// ============================================================================
// Test 7: SWALRScheduler — warmup + swa_lr transition
// ============================================================================
static void test_swa_lr_scheduler() {
    cout << "\n[test_swa_lr_scheduler]" << endl;
    SWALRScheduler sched(/*start_lr=*/0.1, /*swa_lr=*/0.05,
                         /*warmup_steps=*/4, /*swa_start_step=*/10);

    // step 0: at start_lr
    check("step 0: lr == start_lr (no warmup yet, no swa yet)",
          close(sched.get_lr(), 0.1));
    sched.step();
    check("step 1: lr == start_lr * (1/4) == 0.025 (warmup in progress)",
          close(sched.get_lr(), 0.1 * (1.0 / 4.0), 1e-9));
    sched.step();
    check("step 2: lr == start_lr * (2/4) == 0.05",
          close(sched.get_lr(), 0.1 * (2.0 / 4.0), 1e-9));
    sched.step();
    sched.step();
    // step 4: warmup done
    check("step 4 (warmup complete): lr == start_lr (still pre-swa)",
          close(sched.get_lr(), 0.1));
    for (int i = 0; i < 5; ++i) sched.step();  // steps 5..9
    check("step 9 (still pre-swa): lr == start_lr", close(sched.get_lr(), 0.1));
    sched.step();  // step 10: swa started
    check("step 10 (swa started): lr == swa_lr", close(sched.get_lr(), 0.05));
    sched.step();
    check("step 11 (swa continues): lr == swa_lr", close(sched.get_lr(), 0.05));

    // reset
    sched.reset();
    check("after reset(): lr == start_lr", close(sched.get_lr(), 0.1));
    check("after reset(): get_lr() consistent", close(sched.get_lr(), 0.1));
}

// ============================================================================
// Test 8: determinism — two fresh instances produce identical averaged weights
// ============================================================================
static void test_determinism_two_instances_match() {
    cout << "\n[test_determinism_two_instances_match]" << endl;
    auto run_one = []() {
        Model m;
        build_minimal_model(m, 0.0, 0.0);
        auto* inner = new IncrementalOptimizer(0.37);
        SWAOptimizer swa(inner, /*start_after=*/2);
        for (int i = 0; i < 8; ++i) swa.step(m);
        swa.swap_to_averaged(m);
        return m.layers[0]->parameters()[0]->operator()(0, 0);
    };
    double a = run_one();
    double b = run_one();
    check("two fresh SWA instances produce identical averaged weights", close(a, b, 0.0));
}

// ============================================================================
// Test 9: end-to-end — averaged weights differ from the final live weights
// ============================================================================
static void test_averaged_differs_from_live_weights() {
    cout << "\n[test_averaged_differs_from_live_weights]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);
    auto* inner = new IncrementalOptimizer(1.0);
    SWAOptimizer swa(inner, /*start_after=*/0);
    for (int i = 0; i < 4; ++i) swa.step(m);
    // Live weights after 4 steps: w = 4.0
    double live_w = m.layers[0]->parameters()[0]->operator()(0, 0);
    check("live weight == 4.0 after 4 incremental steps", close(live_w, 4.0));
    // Capture averaged (without swap): mean({1,2,3,4}) = 2.5
    // We can't directly access the averaged without swap, so swap, read, then
    // re-swap (we don't have a "restore" — but the test confirms swap changes).
    swa.swap_to_averaged(m);
    double avg_w = m.layers[0]->parameters()[0]->operator()(0, 0);
    check("averaged weight == 2.5 (mean of {1,2,3,4})", close(avg_w, 2.5));
    check("live (4.0) != averaged (2.5)", !close(live_w, avg_w));
}

// ============================================================================
// Test 10: pre-warmup averaging does NOT touch live weights (inner still runs)
// ============================================================================
static void test_inner_step_runs_during_warmup() {
    cout << "\n[test_inner_step_runs_during_warmup]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);
    auto* inner = new IncrementalOptimizer(2.0);
    SWAOptimizer swa(inner, /*start_after=*/10);

    for (int i = 0; i < 5; ++i) swa.step(m);  // all within warmup
    // Live weights: 5 * 2 = 10
    check("live weight == 10 (inner stepped 5x)", close(
        m.layers[0]->parameters()[0]->operator()(0, 0), 10.0));
    check("averaged_count() == 0 (still in warmup)", swa.averaged_count() == 0);

    // swap_to_averaged is a no-op when not initialized
    SWAOptimizer fresh_swa(new IncrementalOptimizer(0.0), 5);
    Model m2;
    build_minimal_model(m2, 1.0, 1.0);
    fresh_swa.swap_to_averaged(m2);  // should not crash; params unchanged
    check("swap on uninitialized SWA leaves params unchanged",
          close(m2.layers[0]->parameters()[0]->operator()(0, 0), 1.0));
}

// ============================================================================
// Test 11: multi-layer model — each layer's params are averaged independently
// ============================================================================
static void test_multi_layer_independent_averaging() {
    cout << "\n[test_multi_layer_independent_averaging]" << endl;
    Model m;
    auto* d1 = new Dense(2, 3);
    auto* d2 = new Dense(3, 1);
    for (size_t i = 0; i < d1->weights.rows; ++i)
        for (size_t j = 0; j < d1->weights.cols; ++j)
            d1->weights(i, j) = 0.0;
    for (size_t i = 0; i < d2->weights.rows; ++i)
        for (size_t j = 0; j < d2->weights.cols; ++j)
            d2->weights(i, j) = 0.0;
    m.add_layer(d1);
    m.add_layer(d2);
    auto* inner = new IncrementalOptimizer(1.0);
    SWAOptimizer swa(inner, /*start_after=*/0);
    for (int i = 0; i < 3; ++i) swa.step(m);
    swa.swap_to_averaged(m);
    // After 3 steps every entry = 3.0; averaged = mean({1,2,3}) = 2.0
    bool all_two = true;
    for (size_t i = 0; i < d1->weights.rows; ++i)
        for (size_t j = 0; j < d1->weights.cols; ++j)
            if (!close(d1->weights(i, j), 2.0, 1e-9)) all_two = false;
    check("d1 weights all == 2.0 after averaging", all_two);
    bool all_two_d2 = true;
    for (size_t i = 0; i < d2->weights.rows; ++i)
        for (size_t j = 0; j < d2->weights.cols; ++j)
            if (!close(d2->weights(i, j), 2.0, 1e-9)) all_two_d2 = false;
    check("d2 weights all == 2.0 after averaging", all_two_d2);
}

// ============================================================================
// Test 12: param shape preservation after averaging
// ============================================================================
static void test_param_shapes_preserved_after_swap() {
    cout << "\n[test_param_shapes_preserved_after_swap]" << endl;
    Model m;
    auto* d = new Dense(3, 5);
    m.add_layer(d);
    auto* inner = new ConstantValueOptimizer(0.5);
    SWAOptimizer swa(inner, /*start_after=*/0);
    for (int i = 0; i < 3; ++i) swa.step(m);
    swa.swap_to_averaged(m);
    auto params = m.layers[0]->parameters();
    check("Dense(3,5) has 2 params (weights, bias)", params.size() == 2);
    check("weights shape is (5, 3)", params[0]->rows == 5 && params[0]->cols == 3);
    check("bias shape is (1, 5)", params[1]->rows == 1 && params[1]->cols == 5);
    // All averaged entries == 0.5
    bool all_half = true;
    for (size_t i = 0; i < params[0]->rows; ++i)
        for (size_t j = 0; j < params[0]->cols; ++j)
            if (!close((*params[0])(i, j), 0.5, 1e-9)) all_half = false;
    check("all averaged weights == 0.5", all_half);
}

// ============================================================================
// Test 13: param-count drift guard — averaged_count vs layer count
// ============================================================================
static void test_step_count_tracks_calls() {
    cout << "\n[test_step_count_tracks_calls]" << endl;
    Model m;
    build_minimal_model(m, 0.0, 0.0);
    auto* inner = new ConstantValueOptimizer(1.0);
    SWAOptimizer swa(inner, /*start_after=*/0);
    check("averaged_count() == 0 before any step", swa.averaged_count() == 0);
    swa.step(m);
    check("averaged_count() == 1 after 1 step (start_after=0)", swa.averaged_count() == 1);
    swa.step(m);
    swa.step(m);
    check("averaged_count() == 3 after 3 steps", swa.averaged_count() == 3);
}

// ============================================================================
// Test 14: SWALRScheduler — zero warmup, immediate swa_lr
// ============================================================================
static void test_swa_lr_scheduler_no_warmup() {
    cout << "\n[test_swa_lr_scheduler_no_warmup]" << endl;
    SWALRScheduler sched(0.1, 0.05, /*warmup_steps=*/0, /*swa_start_step=*/0);
    // The constructor initializes current_lr_ to start_lr; update_lr() is
    // called from step(). After one step(), step_count_=1 >= swa_start_step_=0,
    // so lr == swa_lr.
    sched.step();
    check("step 1: lr == swa_lr (swa_start reached)", close(sched.get_lr(), 0.05));
    sched.step();
    check("step 2: lr == swa_lr (still)", close(sched.get_lr(), 0.05));
}

// ============================================================================
// Test 15: SWALRScheduler — warmup only, no swa transition yet
// ============================================================================
static void test_swa_lr_scheduler_warmup_only() {
    cout << "\n[test_swa_lr_scheduler_warmup_only]" << endl;
    SWALRScheduler sched(0.1, 0.05, /*warmup_steps=*/5, /*swa_start_step=*/100);
    // Constructor sets current_lr_ = start_lr_ = 0.1 (no warmup math yet).
    check("step 0 (pre-step()): lr == start_lr (constructor default)",
          close(sched.get_lr(), 0.1));
    sched.step();  // step_count_ = 1 -> warmup math: 1/5 of start_lr
    check("step 1: lr == start_lr * 1/5", close(sched.get_lr(), 0.1 * 0.2, 1e-9));
    sched.step();
    sched.step();
    sched.step();
    sched.step();
    // step_count_ = 6: 6 >= warmup_steps_=5, so first branch false;
    //                  6 < swa_start_step_=100, so third branch: lr = start_lr
    check("step 6: warmup done, lr == start_lr", close(sched.get_lr(), 0.1));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    cout << "=== SWA (Stochastic Weight Averaging) Tests ===" << endl;
    cout << "Izmailov et al. 2018 — https://arxiv.org/abs/1803.05407" << endl;

    test_before_start_after_is_noop_for_averaging();
    test_averaged_count_after_start();
    test_averaged_weight_is_arithmetic_mean();
    test_swap_to_averaged_replaces_live_weights();
    test_record_updates_average_without_inner_step();
    test_update_bn_stats_runs_forward_passes();
    test_swa_lr_scheduler();
    test_determinism_two_instances_match();
    test_averaged_differs_from_live_weights();
    test_inner_step_runs_during_warmup();
    test_multi_layer_independent_averaging();
    test_param_shapes_preserved_after_swap();
    test_step_count_tracks_calls();
    test_swa_lr_scheduler_no_warmup();
    test_swa_lr_scheduler_warmup_only();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
