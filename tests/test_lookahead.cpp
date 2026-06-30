// test_lookahead.cpp — Tests for Lookahead optimizer (Zhang et al. 2019)
// Paper: "Lookahead Optimizer: k steps forward, 1 step back"
// (https://arxiv.org/abs/1907.08610)
//
// Lookahead is a meta-optimizer that wraps any inner optimizer. It maintains
// "slow" weights and every k inner updates, interpolates the model weights
// toward the slow weights by alpha, then resets slow = current.
//
//   for t = 1, 2, ...:
//       inner.step(model)                       // fast weights move
//       if t % k == 0:
//           weights  = weights + alpha * (slow - weights)   // slow pull
//           slow     = weights                              // snapshot
//
// Key properties tested:
//   - First call initializes slow_weights to current weights
//   - Inner optimizer drives the fast updates
//   - On k-th step, weights are pulled toward slow snapshot by alpha
//   - alpha=0 leaves weights unchanged (no slow pull)
//   - alpha=1 resets weights exactly to slow snapshot
//   - Slow snapshots are updated AFTER the interpolation
//   - Inner step() is called EVERY Lookahead step (not gated by k)
//   - get_k() / get_alpha() / inner() accessors return the configuration
//
// IMPORTANT: Lookahead takes ownership of the inner optimizer via unique_ptr.
// All test mocks below heap-allocate the inner to avoid double-free.
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/lookahead.h"
#include "nn/optimizers/optimizer.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Mock optimizer that sets W[0,0] of the first Dense to a known value.
// Used to test the Lookahead pull behaviour deterministically.
class ConstantWOptimizer : public Optimizer {
public:
    double value_;
    explicit ConstantWOptimizer(double v) : Optimizer(), value_(v) {}
    void step(Model& model) override {
        for (auto& layer : model.layers) {
            auto* d = dynamic_cast<Dense*>(layer.get());
            if (!d) continue;
            d->weights(0, 0) = value_;
        }
    }
};

// Mock optimizer that counts how many times step() is called.
class CountingOptimizer : public Optimizer {
public:
    int count = 0;
    void step(Model&) override { ++count; }
};

int main() {
    cout << setprecision(10);
    cout << "=== Lookahead Optimizer Tests ===" << endl << endl;

    // ---------------------------------------------------------------
    // Test 1: Configuration accessors return the configured values
    // ---------------------------------------------------------------
    cout << "Test 1: accessors (get_k, get_alpha, inner)" << endl;
    {
        auto* sgd = new CountingOptimizer();
        sgd->lr = 0.1;
        Lookahead la(sgd, 5, 0.5);
        check("Test 1a: get_k() returns k", la.get_k() == 5);
        check("Test 1b: get_alpha() returns alpha", la.get_alpha() == 0.5);
        check("Test 1c: inner() returns the wrapped optimizer",
              la.inner() == sgd);
    }

    // ---------------------------------------------------------------
    // Test 2: slow_weights_ initialization and pull
    // ---------------------------------------------------------------
    cout << endl << "Test 2: slow_weights_ initialization and pull" << endl;
    {
        Model model;
        Dense* dense = new Dense(2, 2);
        // Pre-set known weights so we can predict behaviour
        dense->weights(0, 0) = 0.5;
        dense->weights(0, 1) = 0.6;
        dense->weights(1, 0) = 0.7;
        dense->weights(1, 1) = 0.8;
        model.add_layer(dense);

        Tensor W_initial = dense->weights;  // snapshot
        double W_target = 1.0;
        auto* inner = new ConstantWOptimizer(W_target);
        Lookahead la(inner, 2, 0.5);

        // Step 1: initialize slow = W_initial, inner sets W = W_target
        la.step(model);
        check("Test 2a: step 1 — fast weights set by inner",
              std::abs(dense->weights(0, 0) - W_target) < 1e-12);

        // Step 2: k-boundary (2 % 2 == 0)
        //   inner sets W = W_target, then pull:
        //   weights = weights + 0.5 * (slow - weights)
        //           = W_target + 0.5 * (W_initial - W_target)
        la.step(model);
        double expected = W_target + 0.5 * (W_initial(0, 0) - W_target);
        check("Test 2b: step 2 (k-boundary) — weights pulled halfway to slow",
              std::abs(dense->weights(0, 0) - expected) < 1e-12);

        // Step 3: inner overwrites W = W_target
        la.step(model);
        check("Test 2c: step 3 — inner overwrites weights",
              std::abs(dense->weights(0, 0) - W_target) < 1e-12);

        // Step 4: k-boundary. slow was updated after step 2 to the
        // post-pull weights, then step 3 inner wrote W_target over it.
        // After step 4 inner: W = W_target. Pull toward (post-step-2 weights).
        double post_step2 = W_target + 0.5 * (W_initial(0, 0) - W_target);
        double expected2 = W_target + 0.5 * (post_step2 - W_target);
        la.step(model);
        check("Test 2d: step 4 (k-boundary) — pull toward updated slow",
              std::abs(dense->weights(0, 0) - expected2) < 1e-12);
    }

    // ---------------------------------------------------------------
    // Test 3: alpha=0 → no pull on k-boundary, weights stay at inner output
    // ---------------------------------------------------------------
    cout << endl << "Test 3: alpha=0 means no pull" << endl;
    {
        Model model;
        Dense* dense = new Dense(2, 2);
        dense->weights(0, 0) = 0.0;
        model.add_layer(dense);

        auto* inner = new ConstantWOptimizer(7.0);
        Lookahead la(inner, 1, 0.0);  // k=1, alpha=0 → pull never moves weights
        la.step(model);
        la.step(model);

        check("Test 3: alpha=0 — weights unchanged by pull",
              std::abs(dense->weights(0, 0) - 7.0) < 1e-12);
    }

    // ---------------------------------------------------------------
    // Test 4: alpha=1 → on k-boundary, weights reset to slow snapshot
    // ---------------------------------------------------------------
    cout << endl << "Test 4: alpha=1 resets weights to slow" << endl;
    {
        Model model;
        Dense* dense = new Dense(2, 2);
        double W_init = dense->weights(0, 0);  // initial value (Dense inits randomly)
        model.add_layer(dense);

        auto* inner = new ConstantWOptimizer(99.0);
        Lookahead la(inner, 1, 1.0);  // k=1, alpha=1

        la.step(model);  // step 1: init slow = W_init, inner sets W = 99, then pull
                          //   k=1 means EVERY step is a k-boundary, so the pull happens
                          //   even on step 1: W = 99 + 1.0 * (W_init - 99) = W_init
        check("Test 4a: step 1 — alpha=1 pull snaps W back to init",
              std::abs(dense->weights(0, 0) - W_init) < 1e-12);

        la.step(model);  // step 2: k=1 again — same cycle
        check("Test 4b: alpha=1, k=1 — weights stay at slow (= init)",
              std::abs(dense->weights(0, 0) - W_init) < 1e-12);
    }

    // ---------------------------------------------------------------
    // Test 5: Lookahead calls inner.step every step (not gated by k)
    // ---------------------------------------------------------------
    cout << endl << "Test 5: inner.step called every Lookahead step" << endl;
    {
        Model model;
        Dense* dense = new Dense(2, 2);
        model.add_layer(dense);

        auto* inner = new CountingOptimizer();
        Lookahead la(inner, 5, 0.5);  // k=5, but inner should still be called every step

        for (int i = 0; i < 10; ++i) la.step(model);
        check("Test 5: inner.step called 10 times over 10 Lookahead steps",
              inner->count == 10);
    }

    // ---------------------------------------------------------------
    // Test 6: Lookahead + SGD doesn't crash on a real Model with Dense
    // ---------------------------------------------------------------
    cout << endl << "Test 6: Lookahead + SGD on a 1-D linear regression" << endl;
    {
        Model model;
        Dense* dense = new Dense(1, 1);
        dense->weights(0, 0) = 0.0;
        dense->bias(0, 0) = 0.0;
        model.add_layer(dense);

        // Inner: an SGD-like optimizer that nudges W by a small fixed amount
        class TinySGD : public Optimizer {
        public:
            void step(Model& model) override {
                for (auto& layer : model.layers) {
                    auto* d = dynamic_cast<Dense*>(layer.get());
                    if (!d) continue;
                    for (size_t r = 0; r < d->weights.rows; ++r)
                        for (size_t c = 0; c < d->weights.cols; ++c)
                            d->weights(r, c) -= 0.01;
                    for (size_t c = 0; c < d->bias.cols; ++c)
                        d->bias(0, c) -= 0.01;
                }
            }
        };

        auto* sgd = new TinySGD();
        Lookahead la(sgd, 3, 0.5);
        for (int i = 0; i < 30; ++i) la.step(model);

        bool finite = true;
        for (size_t r = 0; r < dense->weights.rows; ++r)
            for (size_t c = 0; c < dense->weights.cols; ++c)
                if (std::isnan(dense->weights(r, c)) || std::isinf(dense->weights(r, c))) finite = false;

        check("Test 6: 30 Lookahead steps — weights remain finite", finite);
        check("Test 6b: 30 Lookahead steps — weights moved (not equal to init)",
              std::abs(dense->weights(0, 0) - 0.0) > 1e-6);
    }

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}