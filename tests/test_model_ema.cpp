// test_model_ema.cpp — Tests for ModelEMA (Exponential Moving Average of
// model parameters).
//
// ModelEMA is a passive wrapper around Model that maintains a "shadow" copy
// of every parameter Tensor in the model. Each call to step(model) updates
// the shadow with the recursive formula:
//
//   shadow_t = decay * shadow_{t-1} + (1 - decay) * w_t
//
// The shadow weights are typically used at evaluation time in place of the
// raw model weights (apply_shadow / restore pattern).
//
// Coverage:
//   1. Constructor: lazy snapshot of current weights → shadow == w (decay irrelevant at t=0)
//   2. Accessors: decay, set_decay, step_count, num_params
//   3. Constructor validation: decay outside [0, 1] throws
//   4. decay=0 → shadow tracks weights exactly after step
//   5. decay=1 → shadow stays at initialization (frozen)
//   6. Constant weights → shadow converges to that constant (limit property)
//   7. Constant weights → closed-form shadow trajectory matches formula
//      shadow_t = decay^t * shadow_0 + (1 - decay^t) * w  (hand-derived)
//   8. apply_shadow then restore round-trips weights exactly
//   9. apply_shadow leaves shadow unchanged (only affects model weights)
//  10. reinitialize resets shadow and step_count
//  11. Multi-layer model: all parameters are tracked independently
//  12. get_shadow returns a deep copy (mutation doesn't affect internal state)
//  13. get_stored returns the saved backup after store()/apply_shadow()
//  14. restore without prior store() throws
//  15. Training loop: shadow ends up between init and final weights (smoother)
//  16. State isolation: two EMAs on different models don't interfere
//  17. Single-step EMA closed-form (constant weights)
//  18. Multi-step EMA convergence (shadow converges as decay^N → 0)
//  19. Re-initialize after parameter-count change clears stale state
//  20. step_count increments correctly
//  21. The shadow is updated lazily: after init, shadow equals w_0
//  22. Two layers, two parameters each — independent shadow trajectories

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/optimizer.h"
#include "nn/utils/model_ema.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool cond) {
    if (cond) {
        cout << "  [PASS] " << name << endl;
        ++passed;
    } else {
        cout << "  [FAIL] " << name << endl;
        ++failed;
    }
}

// Helpers --------------------------------------------------------------------

// Build a model with two Dense layers: in -> hidden -> out. Both layers are
// owned by the model (caller does NOT delete them).
static Model build_two_layer_model(size_t in, size_t hid, size_t out) {
    Model m;
    m.add_layer(new Dense(in, hid));
    m.add_layer(new Dense(hid, out));
    return m;
}

// Set the weights of the FIRST Dense to a constant value.
static void set_first_layer_weights(Model& m, double v) {
    Dense* d = dynamic_cast<Dense*>(m.layers[0].get());
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            d->weights[i][j] = v;
    for (size_t i = 0; i < d->bias.rows; ++i)
        for (size_t j = 0; j < d->bias.cols; ++j)
            d->bias[i][j] = v;
}

// Set the weights of the SECOND Dense to a constant value.
static void set_second_layer_weights(Model& m, double v) {
    Dense* d = dynamic_cast<Dense*>(m.layers[1].get());
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            d->weights[i][j] = v;
    for (size_t i = 0; i < d->bias.rows; ++i)
        for (size_t j = 0; j < d->bias.cols; ++j)
            d->bias[i][j] = v;
}

static bool tensor_near(const Tensor& a, const Tensor& b, double tol) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (std::fabs(a[i][j] - b[i][j]) > tol) return false;
    return true;
}

// =====================================================================
// Test 1: Constructor snapshot
// =====================================================================
static void test_constructor_snapshot() {
    cout << "\n-- Test 1: Constructor takes a snapshot of current weights --" << endl;
    Model m = build_two_layer_model(3, 4, 2);
    // First Dense weights initialized via Dense() constructor (random). Force
    // a known value so we can check.
    set_first_layer_weights(m, 0.5);
    set_second_layer_weights(m, 0.7);

    ModelEMA ema(m, 0.999);

    // shadow for param 0 (Dense0.weights) should equal the weights we set.
    Tensor s0 = ema.get_shadow(0);
    check("shadow[0] equals Dense0.weights at init", tensor_near(s0, m.layers[0]->parameters()[0][0], 1e-12));
    // shadow for param 1 (Dense0.bias)
    Tensor s1 = ema.get_shadow(1);
    check("shadow[1] equals Dense0.bias at init", tensor_near(s1, m.layers[0]->parameters()[1][0], 1e-12));
    // shadow for param 2 (Dense1.weights)
    Tensor s2 = ema.get_shadow(2);
    check("shadow[2] equals Dense1.weights at init", tensor_near(s2, m.layers[1]->parameters()[0][0], 1e-12));
    // shadow for param 3 (Dense1.bias)
    Tensor s3 = ema.get_shadow(3);
    check("shadow[3] equals Dense1.bias at init", tensor_near(s3, m.layers[1]->parameters()[1][0], 1e-12));
}

// =====================================================================
// Test 2: Accessors
// =====================================================================
static void test_accessors() {
    cout << "\n-- Test 2: Accessors return configured values --" << endl;
    Model m = build_two_layer_model(3, 4, 2);
    ModelEMA ema(m, 0.9);
    check("decay() returns 0.9", std::fabs(ema.decay() - 0.9) < 1e-12);
    check("step_count() == 0 at init", ema.step_count() == 0);
    // Two Dense layers, each has 2 params (weights + bias) → 4 total
    check("num_params() == 4", ema.num_params() == 4);

    ema.set_decay(0.5);
    check("set_decay updates decay()", std::fabs(ema.decay() - 0.5) < 1e-12);

    bool threw = false;
    try { ema.set_decay(-0.1); } catch (const std::invalid_argument&) { threw = true; }
    check("set_decay(-0.1) throws", threw);

    threw = false;
    try { ema.set_decay(1.5); } catch (const std::invalid_argument&) { threw = true; }
    check("set_decay(1.5) throws", threw);
}

// =====================================================================
// Test 3: Constructor validation
// =====================================================================
static void test_constructor_validation() {
    cout << "\n-- Test 3: Constructor rejects out-of-range decay --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    bool threw = false;
    try {
        ModelEMA bad(m, -0.5);
    } catch (const std::invalid_argument&) { threw = true; }
    check("ModelEMA(model, -0.5) throws", threw);

    threw = false;
    try {
        ModelEMA bad(m, 2.0);
    } catch (const std::invalid_argument&) { threw = true; }
    check("ModelEMA(model, 2.0) throws", threw);

    // Boundaries 0 and 1 should be allowed.
    bool ok_zero = true;
    bool ok_one = true;
    try { ModelEMA e0(m, 0.0); } catch (...) { ok_zero = false; }
    try { ModelEMA e1(m, 1.0); } catch (...) { ok_one = false; }
    check("decay=0 accepted", ok_zero);
    check("decay=1 accepted", ok_one);
}

// =====================================================================
// Test 4: decay=0 → shadow tracks weights exactly
// =====================================================================
static void test_decay_zero_tracks() {
    cout << "\n-- Test 4: decay=0 → shadow == current weights after step --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 0.0);

    ModelEMA ema(m, 0.0);
    // Change weights then step — shadow should equal NEW weights.
    set_first_layer_weights(m, 1.0);
    ema.step(m);

    Tensor s = ema.get_shadow(0);  // Dense0.weights
    // All elements should be 1.0
    bool all_one = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - 1.0) > 1e-12) all_one = false;
    check("decay=0, step to w=1.0: shadow == 1.0 everywhere", all_one);

    // Move weights to 2.5, step again — shadow should now be 2.5
    set_first_layer_weights(m, 2.5);
    ema.step(m);
    s = ema.get_shadow(0);
    bool all_two_five = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - 2.5) > 1e-12) all_two_five = false;
    check("decay=0, second step to w=2.5: shadow == 2.5 everywhere", all_two_five);
}

// =====================================================================
// Test 5: decay=1 → shadow is frozen at initialization
// =====================================================================
static void test_decay_one_frozen() {
    cout << "\n-- Test 5: decay=1 → shadow never changes --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 1.5);
    set_second_layer_weights(m, 2.5);

    ModelEMA ema(m, 1.0);

    // Mutate weights and step multiple times
    set_first_layer_weights(m, 99.0);
    set_second_layer_weights(m, 77.0);
    for (int k = 0; k < 5; ++k) ema.step(m);

    Tensor s0 = ema.get_shadow(0);
    Tensor s2 = ema.get_shadow(2);
    bool ok0 = true, ok2 = true;
    for (size_t i = 0; i < s0.rows; ++i)
        for (size_t j = 0; j < s0.cols; ++j)
            if (std::fabs(s0[i][j] - 1.5) > 1e-12) ok0 = false;
    for (size_t i = 0; i < s2.rows; ++i)
        for (size_t j = 0; j < s2.cols; ++j)
            if (std::fabs(s2[i][j] - 2.5) > 1e-12) ok2 = false;
    check("decay=1: shadow[0] still == 1.5 after 5 steps", ok0);
    check("decay=1: shadow[2] still == 2.5 after 5 steps", ok2);
}

// =====================================================================
// Test 6 & 7: Constant-weight closed-form trajectory
// shadow_t = decay^t * shadow_0 + (1 - decay^t) * w_constant
// =====================================================================
static void test_constant_weight_trajectory() {
    cout << "\n-- Test 6-7: Constant-weight closed-form trajectory --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    const double w_init = 0.0;       // shadow_0 = 0
    const double w_const = 4.0;      // weights move to 4.0 once, then stay
    const double decay = 0.5;

    set_first_layer_weights(m, w_init);
    ModelEMA ema(m, decay);

    // Step 1: weights = w_const. shadow_1 = decay * 0 + (1-decay) * 4 = 2.0
    set_first_layer_weights(m, w_const);
    ema.step(m);
    Tensor s = ema.get_shadow(0);
    double expected_1 = decay * w_init + (1.0 - decay) * w_const;
    bool ok1 = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - expected_1) > 1e-12) ok1 = false;
    check("step 1: shadow == decay*0 + (1-decay)*4 = 2.0", ok1);

    // Step 2: weights still = w_const. shadow_2 = decay * 2 + (1-decay) * 4 = 3.0
    ema.step(m);
    s = ema.get_shadow(0);
    double expected_2 = decay * expected_1 + (1.0 - decay) * w_const;
    bool ok2 = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - expected_2) > 1e-12) ok2 = false;
    check("step 2: shadow == 3.0", ok2);

    // Step N (constant weights): shadow_t = decay^t * 0 + (1-decay^t) * 4
    // After 10 steps: decay^10 * 0 + (1 - 0.5^10) * 4 ≈ 4 * (1 - 1/1024)
    for (int k = 0; k < 8; ++k) ema.step(m);  // total 10 steps
    s = ema.get_shadow(0);
    double expected_10 = (1.0 - std::pow(decay, 10)) * w_const;
    bool ok10 = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - expected_10) > 1e-12) ok10 = false;
    check("step 10: shadow == (1 - 0.5^10) * 4 ≈ 3.996", ok10);
}

// =====================================================================
// Test 8: apply_shadow / restore round-trip
// =====================================================================
static void test_apply_restore_roundtrip() {
    cout << "\n-- Test 8: apply_shadow / restore round-trip --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 3.0);
    set_second_layer_weights(m, 5.0);

    ModelEMA ema(m, 0.99);

    // Snapshot original weights
    Tensor orig_w0 = m.layers[0]->parameters()[0][0].clone();
    Tensor orig_b0 = m.layers[0]->parameters()[1][0].clone();
    Tensor orig_w1 = m.layers[1]->parameters()[0][0].clone();
    Tensor orig_b1 = m.layers[1]->parameters()[1][0].clone();

    // Apply shadow (which equals original weights since we haven't stepped)
    ema.apply_shadow(m);

    // Now restore
    ema.restore(m);

    // All four should be back to original
    check("after apply+restore: Dense0.weights unchanged", tensor_near(orig_w0, m.layers[0]->parameters()[0][0], 1e-12));
    check("after apply+restore: Dense0.bias unchanged", tensor_near(orig_b0, m.layers[0]->parameters()[1][0], 1e-12));
    check("after apply+restore: Dense1.weights unchanged", tensor_near(orig_w1, m.layers[1]->parameters()[0][0], 1e-12));
    check("after apply+restore: Dense1.bias unchanged", tensor_near(orig_b1, m.layers[1]->parameters()[1][0], 1e-12));
}

// =====================================================================
// Test 9: apply_shadow leaves shadow unchanged
// =====================================================================
static void test_apply_shadow_does_not_change_shadow() {
    cout << "\n-- Test 9: apply_shadow does not change shadow --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 1.0);
    set_second_layer_weights(m, 2.0);

    ModelEMA ema(m, 0.5);

    // Step once with weights=1,1,1,1 → shadow = 0.5*1 + 0.5*1 = 1.0
    set_first_layer_weights(m, 1.0);
    set_second_layer_weights(m, 1.0);
    ema.step(m);

    Tensor s_before = ema.get_shadow(0);
    ema.apply_shadow(m);
    Tensor s_after = ema.get_shadow(0);
    check("apply_shadow does not modify shadow[0]", tensor_near(s_before, s_after, 1e-12));
}

// =====================================================================
// Test 10: reinitialize resets shadow and step_count
// =====================================================================
static void test_reinitialize() {
    cout << "\n-- Test 10: reinitialize resets state --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 0.0);
    ModelEMA ema(m, 0.9);

    // Step a few times
    set_first_layer_weights(m, 10.0);
    for (int k = 0; k < 5; ++k) ema.step(m);
    check("step_count == 5 after 5 steps", ema.step_count() == 5);

    // Reinitialize
    set_first_layer_weights(m, 7.0);
    ema.reinitialize(m);
    check("step_count == 0 after reinitialize", ema.step_count() == 0);

    // Shadow should now equal the CURRENT weights (7.0)
    Tensor s = ema.get_shadow(0);
    bool all_seven = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - 7.0) > 1e-12) all_seven = false;
    check("after reinitialize: shadow == current weights (7.0)", all_seven);
}

// =====================================================================
// Test 11: Multi-layer isolation
// =====================================================================
static void test_multi_layer_independence() {
    cout << "\n-- Test 11: Two-layer parameter independence --" << endl;
    Model m = build_two_layer_model(2, 3, 2);
    set_first_layer_weights(m, 1.0);
    set_second_layer_weights(m, 9.0);

    ModelEMA ema(m, 0.0);  // decay=0 → shadow tracks exactly

    set_first_layer_weights(m, 2.0);
    set_second_layer_weights(m, 8.0);
    ema.step(m);

    Tensor s0 = ema.get_shadow(0);  // Dense0.weights
    Tensor s2 = ema.get_shadow(2);  // Dense1.weights
    bool ok0 = true, ok2 = true;
    for (size_t i = 0; i < s0.rows; ++i)
        for (size_t j = 0; j < s0.cols; ++j)
            if (std::fabs(s0[i][j] - 2.0) > 1e-12) ok0 = false;
    for (size_t i = 0; i < s2.rows; ++i)
        for (size_t j = 0; j < s2.cols; ++j)
            if (std::fabs(s2[i][j] - 8.0) > 1e-12) ok2 = false;
    check("Dense0.weights shadow == 2.0", ok0);
    check("Dense1.weights shadow == 8.0", ok2);
}

// =====================================================================
// Test 12: get_shadow returns a deep copy
// =====================================================================
static void test_get_shadow_is_deep_copy() {
    cout << "\n-- Test 12: get_shadow returns a deep copy --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 0.5);
    ModelEMA ema(m, 0.99);

    Tensor s1 = ema.get_shadow(0);
    // Mutate the returned tensor
    for (size_t i = 0; i < s1.rows; ++i)
        for (size_t j = 0; j < s1.cols; ++j)
            s1[i][j] = 999.0;

    // Get fresh — should still be 0.5
    Tensor s2 = ema.get_shadow(0);
    bool ok = true;
    for (size_t i = 0; i < s2.rows; ++i)
        for (size_t j = 0; j < s2.cols; ++j)
            if (std::fabs(s2[i][j] - 0.5) > 1e-12) ok = false;
    check("mutating get_shadow return does not affect internal state", ok);
}

// =====================================================================
// Test 13: store / get_stored round-trip
// =====================================================================
static void test_store_roundtrip() {
    cout << "\n-- Test 13: store() saves a backup --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 0.0);
    ModelEMA ema(m, 0.0);

    set_first_layer_weights(m, 1.0);
    ema.step(m);

    // shadow is now 1.0 (decay=0). Backup not stored yet.
    Tensor empty = ema.get_stored(0);
    check("get_stored returns empty before any store()", empty.rows == 0 && empty.cols == 0);

    // store current weights, then mutate, then restore
    ema.store(m);
    Tensor saved = ema.get_stored(0);
    check("after store(): get_stored has positive shape", saved.rows > 0 && saved.cols > 0);

    set_first_layer_weights(m, 5.0);
    ema.restore(m);

    bool ok = true;
    Tensor w0 = m.layers[0]->parameters()[0][0];
    for (size_t i = 0; i < w0.rows; ++i)
        for (size_t j = 0; j < w0.cols; ++j)
            if (std::fabs(w0[i][j] - 1.0) > 1e-12) ok = false;
    check("after restore: model weights back to stored value (1.0)", ok);
}

// =====================================================================
// Test 14: restore without store throws
// =====================================================================
static void test_restore_without_store_throws() {
    cout << "\n-- Test 14: restore() without store() throws --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    ModelEMA ema(m, 0.9);
    bool threw = false;
    try { ema.restore(m); }
    catch (const std::logic_error&) { threw = true; }
    check("restore without store throws logic_error", threw);
}

// =====================================================================
// Test 15: Training-loop integration (shadow ends up between init and final)
// =====================================================================
static void test_training_loop_smoothing() {
    cout << "\n-- Test 15: End-to-end training: shadow smooths weights --" << endl;
    // Train a tiny model on a linear target for a few SGD steps, with and
    // without EMA. The EMA shadow should be closer to the average of the
    // weight trajectory than the final weight itself.
    srand(42);
    Model m;
    m.add_layer(new Dense(1, 1));  // y = w*x + b, single sample
    Dense* d = dynamic_cast<Dense*>(m.layers[0].get());
    d->weights[0][0] = 0.0;
    d->bias[0][0] = 0.0;

    // Target: y = 2x for x=1 → y=2
    Tensor x(1, 1); x[0][0] = 1.0;
    Tensor y(1, 1); y[0][0] = 2.0;

    ModelEMA ema(m, 0.5);

    SGD opt(0.1);
    // Snapshot weights before each step to compute trajectory mean
    std::vector<double> w_traj;
    w_traj.push_back(d->weights[0][0]);

    for (int step = 0; step < 20; ++step) {
        Tensor pred = m.forward(x);
        Tensor grad_out = (pred - y);  // simplified gradient
        Tensor dx = m.backward(grad_out, 0.0);
        opt.step(m);
        ema.step(m);
        w_traj.push_back(d->weights[0][0]);
    }

    // Compute trajectory mean
    double mean = 0;
    for (double v : w_traj) mean += v;
    mean /= w_traj.size();

    // Compare EMA shadow vs final weight vs mean
    Tensor s = ema.get_shadow(0);  // Dense0.weights
    double ema_w = s[0][0];
    double final_w = d->weights[0][0];

    double dist_ema_to_mean = std::fabs(ema_w - mean);
    double dist_final_to_mean = std::fabs(final_w - mean);
    check("EMA shadow is closer to trajectory mean than final weight",
          dist_ema_to_mean <= dist_final_to_mean + 1e-12);
    check("step_count == 20 after 20 steps", ema.step_count() == 20);

    // Also: apply_shadow then restore brings weights back
    ema.apply_shadow(m);
    double applied_w = d->weights[0][0];
    check("apply_shadow copies shadow into model", std::fabs(applied_w - ema_w) < 1e-12);

    ema.restore(m);
    check("restore brings back training-time weights",
          std::fabs(d->weights[0][0] - final_w) < 1e-12);
}

// =====================================================================
// Test 16: Two EMAs on independent models don't interfere
// =====================================================================
static void test_two_emas_isolation() {
    cout << "\n-- Test 16: Two EMAs on different models are independent --" << endl;
    Model m1 = build_two_layer_model(2, 2, 2);
    Model m2 = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m1, 1.0);
    set_first_layer_weights(m2, 10.0);

    ModelEMA ema1(m1, 0.0);  // tracks exactly
    ModelEMA ema2(m2, 0.0);

    set_first_layer_weights(m1, 2.0);
    set_first_layer_weights(m2, 20.0);
    ema1.step(m1);
    ema2.step(m2);

    Tensor s1 = ema1.get_shadow(0);
    Tensor s2 = ema2.get_shadow(0);
    bool ok1 = true, ok2 = true;
    for (size_t i = 0; i < s1.rows; ++i)
        for (size_t j = 0; j < s1.cols; ++j)
            if (std::fabs(s1[i][j] - 2.0) > 1e-12) ok1 = false;
    for (size_t i = 0; i < s2.rows; ++i)
        for (size_t j = 0; j < s2.cols; ++j)
            if (std::fabs(s2[i][j] - 20.0) > 1e-12) ok2 = false;
    check("ema1 shadow == 2.0", ok1);
    check("ema2 shadow == 20.0", ok2);
}

// =====================================================================
// Test 17: Single-step closed-form (general case)
// =====================================================================
static void test_single_step_closed_form() {
    cout << "\n-- Test 17: Single-step closed-form (any decay) --" << endl;
    // shadow_1 = decay * shadow_0 + (1-decay) * w_1
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 10.0);
    const double decay = 0.7;
    ModelEMA ema(m, decay);

    set_first_layer_weights(m, 3.0);
    ema.step(m);
    Tensor s = ema.get_shadow(0);
    double expected = decay * 10.0 + (1.0 - decay) * 3.0;
    bool ok = true;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            if (std::fabs(s[i][j] - expected) > 1e-12) ok = false;
    check("shadow_1 == decay*10 + (1-decay)*3 = 9.1", ok);
}

// =====================================================================
// Test 18: Multi-step convergence
// =====================================================================
static void test_multistep_convergence() {
    cout << "\n-- Test 18: Constant-weight shadow converges --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 0.0);
    const double decay = 0.9;
    const double w_target = 100.0;
    ModelEMA ema(m, decay);

    set_first_layer_weights(m, w_target);
    for (int k = 0; k < 200; ++k) ema.step(m);

    // After 200 steps with decay=0.9: 1 - 0.9^200 ≈ 1 (since 0.9^200 ~ 7.4e-10)
    Tensor s = ema.get_shadow(0);
    double max_err = 0;
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < s.cols; ++j)
            max_err = std::max(max_err, std::fabs(s[i][j] - w_target));
    check("shadow converges to constant weight (max_err < 1e-6)",
          max_err < 1e-6);
}

// =====================================================================
// Test 19: step_count increments
// =====================================================================
static void test_step_count_increments() {
    cout << "\n-- Test 19: step_count increments correctly --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    ModelEMA ema(m, 0.5);
    check("init step_count == 0", ema.step_count() == 0);
    ema.step(m); check("step_count == 1 after 1 step", ema.step_count() == 1);
    ema.step(m); check("step_count == 2 after 2 steps", ema.step_count() == 2);
    ema.step(m); ema.step(m);
    check("step_count == 4 after 4 steps", ema.step_count() == 4);
}

// =====================================================================
// Test 20: shadow is updated lazily (init-time equals current weights)
// =====================================================================
static void test_lazy_init_matches_weights() {
    cout << "\n-- Test 20: Lazy init: shadow == current weights at construction --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    set_first_layer_weights(m, 1.23);
    set_second_layer_weights(m, 4.56);

    ModelEMA ema(m, 0.999);
    for (size_t i = 0; i < 4; ++i) {
        Tensor s = ema.get_shadow(i);
        Tensor& p = *(m.layers[i / 2]->parameters()[i % 2]);
        check("shadow == weights at index " + to_string(i),
              tensor_near(s, p, 1e-12));
    }
}

// =====================================================================
// Test 21: Bias and weights get independent shadows
// =====================================================================
static void test_bias_weights_independent() {
    cout << "\n-- Test 21: weights and bias are tracked independently --" << endl;
    Model m = build_two_layer_model(2, 2, 2);
    // First Dense: weights=5, bias=10. Second Dense: weights=15, bias=20.
    Dense* d0 = dynamic_cast<Dense*>(m.layers[0].get());
    Dense* d1 = dynamic_cast<Dense*>(m.layers[1].get());
    for (size_t i = 0; i < d0->weights.rows; ++i)
        for (size_t j = 0; j < d0->weights.cols; ++j)
            d0->weights[i][j] = 5.0;
    for (size_t i = 0; i < d0->bias.rows; ++i)
        for (size_t j = 0; j < d0->bias.cols; ++j)
            d0->bias[i][j] = 10.0;
    for (size_t i = 0; i < d1->weights.rows; ++i)
        for (size_t j = 0; j < d1->weights.cols; ++j)
            d1->weights[i][j] = 15.0;
    for (size_t i = 0; i < d1->bias.rows; ++i)
        for (size_t j = 0; j < d1->bias.cols; ++j)
            d1->bias[i][j] = 20.0;

    ModelEMA ema(m, 0.0);  // tracks exactly

    // Move ONLY d0->weights to 7, leave the rest
    for (size_t i = 0; i < d0->weights.rows; ++i)
        for (size_t j = 0; j < d0->weights.cols; ++j)
            d0->weights[i][j] = 7.0;
    ema.step(m);

    Tensor sw = ema.get_shadow(0);   // d0->weights
    Tensor sb = ema.get_shadow(1);   // d0->bias
    Tensor sw2 = ema.get_shadow(2);  // d1->weights
    Tensor sb2 = ema.get_shadow(3);  // d1->bias

    bool okw = true, okb = true, okw2 = true, okb2 = true;
    for (size_t i = 0; i < sw.rows; ++i)
        for (size_t j = 0; j < sw.cols; ++j)
            if (std::fabs(sw[i][j] - 7.0) > 1e-12) okw = false;
    for (size_t i = 0; i < sb.rows; ++i)
        for (size_t j = 0; j < sb.cols; ++j)
            if (std::fabs(sb[i][j] - 10.0) > 1e-12) okb = false;
    for (size_t i = 0; i < sw2.rows; ++i)
        for (size_t j = 0; j < sw2.cols; ++j)
            if (std::fabs(sw2[i][j] - 15.0) > 1e-12) okw2 = false;
    for (size_t i = 0; i < sb2.rows; ++i)
        for (size_t j = 0; j < sb2.cols; ++j)
            if (std::fabs(sb2[i][j] - 20.0) > 1e-12) okb2 = false;
    check("d0->weights shadow == 7", okw);
    check("d0->bias shadow unchanged == 10", okb);
    check("d1->weights shadow unchanged == 15", okw2);
    check("d1->bias shadow unchanged == 20", okb2);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "============================================================" << endl;
    cout << " ModelEMA — Exponential Moving Average of model parameters" << endl;
    cout << "============================================================" << endl;

    test_constructor_snapshot();
    test_accessors();
    test_constructor_validation();
    test_decay_zero_tracks();
    test_decay_one_frozen();
    test_constant_weight_trajectory();
    test_apply_restore_roundtrip();
    test_apply_shadow_does_not_change_shadow();
    test_reinitialize();
    test_multi_layer_independence();
    test_get_shadow_is_deep_copy();
    test_store_roundtrip();
    test_restore_without_store_throws();
    test_training_loop_smoothing();
    test_two_emas_isolation();
    test_single_step_closed_form();
    test_multistep_convergence();
    test_step_count_increments();
    test_lazy_init_matches_weights();
    test_bias_weights_independent();

    cout << "\n============================================================" << endl;
    cout << "  PASSED: " << passed << "    FAILED: " << failed << endl;
    cout << "============================================================" << endl;

    return failed == 0 ? 0 : 1;
}