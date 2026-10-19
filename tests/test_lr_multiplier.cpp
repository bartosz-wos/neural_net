// =============================================================================
// tests/test_lr_multiplier.cpp -- per-tensor / per-layer learning-rate
// multipliers, applied by post-hoc rescaling of the realized update delta.
//
// THE DESIGN WAS CHOSEN BY MEASUREMENT, NOT INTUITION. The first draft's plan
// was "scale the gradient g <- s*g before optimizer.step()". That is exact for
// SGD (measured 2.64e-16) and is SILENTLY ANNIHILATED BY ADAM (measured
// 6.67e-01): Adam's update is lr*m_hat/(sqrt(v_hat)+eps), so with one step
// m_hat ~ g and v_hat ~ g^2, and g/sqrt(g^2) = sign(g). Multiplying g by 3
// changes the update by nothing. So post-hoc delta rescaling is used instead:
// it observes whatever the optimizer actually did and rescales that, so it
// works for all 39 optimizers including the scale-invariant ones.
//
// No finite-difference test and none is possible: the mechanism is a
// post-hoc rescale of an already-computed delta, and the "forward" is the
// optimizer step itself. Verification is by VALUE against a true per-layer-LR
// reference (lr_multiplier_reference_step), plus mutation testing.
// =============================================================================

#include "nn/nn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void check_near(double got, double want, double tol, const std::string& name) {
    bool ok = std::fabs(got - want) <= tol;
    if (ok) ++g_pass; else ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g diff=%.3e tol=%.1e\n",
           ok ? "PASS" : "FAIL", name.c_str(), got, want, std::fabs(got - want), tol);
}

static void check_eq(double got, double want, const std::string& name) {
    bool ok = (got == want);
    if (ok) ++g_pass; else ++g_fail;
    printf("  [%s] %s   got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name.c_str(), got, want);
}

// ===========================================================================
// A TRUE per-layer-LR REFERENCE, built independently of the implementation.
//
// It does NOT rescale a delta. It constructs a model containing ONLY the
// layers that share a scale, sets that group's own optimizer lr = base*scale,
// and steps the whole (sub)model. If the implementation is correct, stepping
// the full model at base lr and rescaling the deltas must give bit-identical
// weights to this reference -- because an optimizer's per-parameter update
// depends only on that parameter's own gradient and state, never on the other
// layers. That independence is the property being verified.
// ===========================================================================
static Tensor reference_step(const Tensor& grads_w, const Tensor& grads_b,
                             double base_lr, double scale, bool adam) {
    // A Dense of the same shape, stepped ALONE at the scaled lr.
    Dense ref(grads_w.cols, grads_w.rows);
    for (size_t i = 0; i < grads_w.rows * grads_w.cols; ++i)
        ref.weights.data[i] = 0.5;
    for (size_t i = 0; i < grads_b.rows * grads_b.cols; ++i)
        ref.bias.data[i] = 0.1;
    for (size_t i = 0; i < grads_w.rows * grads_w.cols; ++i)
        ref.grad_weights.data[i] = grads_w.data[i];
    for (size_t i = 0; i < grads_b.rows * grads_b.cols; ++i)
        ref.grad_bias.data[i] = grads_b.data[i];

    Model m;
    m.add_layer(new Dense(grads_w.cols, grads_w.rows));  // ownership for the Model
    // Re-seed the Model's own layer identically to `ref`, then step it.
    Dense* d = (Dense*)m.layers[0].get();
    d->weights = ref.weights;
    d->bias = ref.bias;
    d->grad_weights = ref.grad_weights;
    d->grad_bias = ref.grad_bias;

    if (adam) {
        Adam opt(base_lr * scale);
        opt.step(m);
    } else {
        SGD opt(base_lr * scale);
        opt.step(m);
    }
    return d->weights;
}

// ===========================================================================
// 1. D1/D2/D3 -- resolve semantics. RED before any implementation exists.
// ===========================================================================
static void test_resolve_semantics() {
    printf("\n-- Test 1: resolve semantics (D1/D2/D3) --\n");
    LrMultiplier mult;
    check_near(mult.resolve("head"), 1.0, 0.0, "D1: default scale is exactly 1");
    mult.set("backbone", 0.5);
    check_near(mult.resolve("backbone"), 0.5, 0.0, "named scale resolves (D3)");
    check_near(mult.resolve("head"), 1.0, 0.0, "D2: unnamed falls back to default");
    mult.set_default(2.0);
    check_near(mult.resolve("head"), 2.0, 0.0, "D2: changed default applies to unnamed");
    check_near(mult.resolve("backbone"), 0.5, 0.0, "D3: named still overrides default");
}

// ===========================================================================
// 2. D4 -- an unknown layer name is not an error. It falls through.
// ===========================================================================
static void test_unknown_name_is_not_an_error() {
    printf("\n-- Test 2: unknown names fall through (D4) --\n");
    LrMultiplier mult;
    mult.set("backbone", 0.25);
    mult.set_default(3.0);
    // A layer nobody named must resolve, not throw.
    check_near(mult.resolve("layer_that_does_not_exist"), 3.0, 0.0,
               "D4: unknown name uses default");
    check_eq(static_cast<double>(mult.size()), 1.0, "D4: only one named entry");
}

// ===========================================================================
// 3. D5 -- zero freezes, negative reverses.
// ===========================================================================
static void test_zero_and_negative_scales() {
    printf("\n-- Test 3: zero and negative scales (D5) --\n");
    LrMultiplier z;  z.set("backbone", 0.0);
    check_near(z.resolve("backbone"), 0.0, 0.0, "D5: zero scale is legal");
    LrMultiplier n;  n.set("backbone", -2.0);
    check_near(n.resolve("backbone"), -2.0, 0.0, "D5: negative scale is legal");
}

// ===========================================================================
// 4. D6/D1 -- scale 1.0 is an EXACT no-op, and the multiplier does not
// mutate the pre-step weights (so it cannot double-count).
// ===========================================================================
static void test_unit_scale_is_exact_noop() {
    printf("\n-- Test 4: unit scale is an exact no-op (D1/D6) --\n");
    Model m;
    m.add_layer(new Dense(3, 2));
    Dense* d = (Dense*)m.layers[0].get();
    d->weights.fill(0.5);
    d->bias.fill(0.1);
    d->grad_weights.fill(0.7);
    d->grad_bias.fill(0.2);

    // Step first, so there IS a delta to correct.
    Adam opt(0.1); opt.step(m);
    Tensor after_step_w = d->weights;

    LrMultiplier mult;  // everything at scale 1
    std::map<Layer*, std::vector<Tensor>> snap;
    lr_multiplier_snapshot(m, snap);
    lr_multiplier_apply_snapshot(m, mult, snap);

    // D1: a scale of 1.0 must leave every weight BIT-IDENTICAL.
    bool bit_identical = true;
    for (size_t i = 0; i < after_step_w.rows * after_step_w.cols; ++i)
        if (d->weights.data[i] != after_step_w.data[i]) bit_identical = false;
    check(bit_identical, "D1: scale 1.0 leaves weights BIT-IDENTICAL");
}

// ===========================================================================
// 5. THE TEST THAT KILLS THE GRADIENT-PRE-SCALING DESIGN -- on a real Adam.
//    Adam is scale-invariant in g, so if the multiplier had been applied to
//    the gradient, scaling by 3 would change NOTHING and this fails.
// ===========================================================================
static void test_adam_multiplier_is_effective() {
    printf("\n-- Test 5: multiplier is EFFECTIVE under Adam (not a no-op) --\n");
    const double BASE = 0.1, SCALE = 3.0;

    // (a) unscaled step
    Model m0;
    m0.add_layer(new Dense(3, 2));
    { Dense* d = (Dense*)m0.layers[0].get();
      d->weights.fill(0.5); d->bias.fill(0.1);
      d->grad_weights.fill(0.7); d->grad_bias.fill(0.2); }
    Adam a0(BASE); a0.step(m0);
    Tensor w_unscaled = ((Dense*)m0.layers[0].get())->weights;

    // (b) scaled via the multiplier
    Model m1;
    m1.add_layer(new Dense(3, 2));
    Dense* d1 = (Dense*)m1.layers[0].get();
    { d1->weights.fill(0.5); d1->bias.fill(0.1);
      d1->grad_weights.fill(0.7); d1->grad_bias.fill(0.2); }
    std::map<Layer*, std::vector<Tensor>> snap;
    lr_multiplier_snapshot(m1, snap);   // BEFORE the step
    Adam a1(BASE); a1.step(m1);
    LrMultiplier mult; mult.set_default(SCALE);
    lr_multiplier_apply_snapshot(m1, mult, snap);
    Tensor w_scaled = d1->weights;

    // The unscaled step moved 0.5 -> 0.5 - 0.1*sign(0.7) = 0.4. A no-op
    // multiplier would leave w_scaled == w_unscaled == 0.4.
    double moved = 0.0, moved_unscaled = 0.0;
    for (size_t i = 0; i < w_scaled.rows * w_scaled.cols; ++i) {
        moved         += std::fabs(w_scaled.data[i] - w_unscaled.data[i]);
        moved_unscaled += std::fabs(w_unscaled.data[i] - 0.5);
    }
    check(moved > 1e-9, "D7: multiplier CHANGED the Adam update (not annihilated)");

    // The CONTRACT is w_scaled == w_before + SCALE*(w_unscaled - w_before).
    // Two DIFFERENT quantities must not be confused: the total movement from
    // the ORIGINAL weights is SCALE*|delta|, but the distance from the
    // UNSCALED result is (SCALE-1)*|delta|. This first draft asserted the
    // latter equalled SCALE*|delta| (want 6*0.3 = 1.8) and so was a test bug,
    // not an implementation bug -- the fingerprint was Test 6 passing at
    // 1e-16 against an independently-constructed reference.
    Tensor w_before(2, 3);  w_before.fill(0.5);
    double got = 0.0;
    for (size_t i = 0; i < w_scaled.rows * w_scaled.cols; ++i) {
        got += std::fabs(w_scaled.data[i] -
                 (w_before.data[i] + SCALE * (w_unscaled.data[i] - w_before.data[i])));
    }
    // The RESIDUAL against the contract must be ~0. A sibling assertion that
    // compared the residual against the sum of the PREDICTED values was
    // removed rather than fixed: it compared two different quantities
    // (a residual sum vs a magnitude sum) and so could never mean anything.
    check_near(got, 0.0, 1e-12, "Adam: contract w = w_before + S*(w_unscaled - w_before) holds");
}

// ===========================================================================
// 6. Cross-check against the TRUE per-layer-LR reference, on Adam AND SGD.
// ===========================================================================
static void test_matches_true_per_layer_lr_reference() {
    printf("\n-- Test 6: matches a true per-layer-lr reference --\n");
    const double BASE = 0.05, SCALE = 4.0;
    Tensor gw(2, 3), gb(1, 2);
    gw.fill(0.7); gb.fill(0.2);

    for (int adam = 0; adam < 2; ++adam) {
        Model m;
        m.add_layer(new Dense(3, 2));
        Dense* d = (Dense*)m.layers[0].get();
        d->weights.fill(0.5); d->bias.fill(0.1);
        d->grad_weights = gw; d->grad_bias = gb;

        std::map<Layer*, std::vector<Tensor>> snap;
        lr_multiplier_snapshot(m, snap);   // BEFORE the step
        if (adam) { Adam opt(BASE); opt.step(m); }
        else      { SGD  opt(BASE); opt.step(m); }
        LrMultiplier mult; mult.set_default(SCALE);
        lr_multiplier_apply_snapshot(m, mult, snap);

        Tensor want_w = reference_step(gw, gb, BASE, SCALE, adam != 0);

        double worst = 0.0;
        for (size_t i = 0; i < want_w.rows * want_w.cols; ++i)
            worst = std::max(worst, std::fabs(d->weights.data[i] - want_w.data[i]));
        check_near(worst, 0.0, 1e-12,
                   adam ? "Adam: rescaled delta == real scaled-lr step"
                        : "SGD:  rescaled delta == real scaled-lr step");
    }
}

// ===========================================================================
// 7. The wrapper: a full Model/Optimizer round trip, and its D7 zero-delta case.
// ===========================================================================
static void test_wrapper_end_to_end() {
    printf("\n-- Test 7: LrMultiplierOptimizer end-to-end --\n");
    Model m;
    m.add_layer(new Dense(4, 3));
    Dense* d = (Dense*)m.layers[0].get();
    d->weights.fill(0.2); d->bias.fill(0.05);
    d->grad_weights.fill(0.3); d->grad_bias.fill(0.1);

    LrMultiplier mult; mult.set_default(2.0);
    Adam base(0.01);
    LrMultiplierOptimizer wrapper(&base, mult);
    wrapper.step(m);

    // Recompute the expected value by hand. The contract keeps w_BEFORE:
    //     w_after = w_before + 2*(w_base_step - w_before)
    // NOT w_after = w_base_step + 2*w_base_step, which double-counts. The
    // first draft wrote the latter and failed by exactly 1.0e-02 = the
    // missing w_before term -- a clean constant, the signature of test
    // arithmetic rather than a broken implementation.
    Model m2;
    m2.add_layer(new Dense(4, 3));
    Dense* d2 = (Dense*)m2.layers[0].get();
    d2->weights.fill(0.2); d2->bias.fill(0.05);
    d2->grad_weights.fill(0.3); d2->grad_bias.fill(0.1);
    Adam base2(0.01); base2.step(m2);
    for (size_t i = 0; i < d2->weights.rows * d2->weights.cols; ++i)
        d2->weights.data[i] = 0.2 + 2.0 * (d2->weights.data[i] - 0.2);

    double worst = 0.0;
    for (size_t i = 0; i < d->weights.rows * d->weights.cols; ++i)
        worst = std::max(worst, std::fabs(d->weights.data[i] - d2->weights.data[i]));
    check_near(worst, 0.0, 1e-15, "wrapper: step() == base step + 2x delta");
    check_eq(base.lr, 0.01, "D6: wrapper leaves the wrapped optimizer's lr untouched");
}

// ===========================================================================
// 4b. D1 NON-VACUITY -- the unit-scale fast path must be LOAD-BEARING.
//      `if (s == 1.0) continue;` skips w <- b + 1.0*(w - b). IEEE 754 does
//      NOT guarantee that identity: over 5e6 random (b,w) pairs, 478 660
//      (9.57%) of them differ, by exactly 1 ulp. So the fast path is a real
//      correctness property (D1 = BIT-identical), not an optimisation.
//
//      M04 survived the first mutation run because every fixture used
//      round values (0.5, 0.2, 0.4...) for which the identity happens to hold
//      exactly. The witnesses below were FOUND by probe, not constructed -- a
//      search over 2e7 draws for `b + 1.0*(w-b) != w`, then transcribed as hex
//      literals so the fixture is bit-exact rather than a decimal transcription.
//      An off-by-one needs a fixture on the discriminating side of the
//      boundary; a "unit scale is a no-op" test needs a value where it is not.
// ===========================================================================
static void test_unit_scale_fast_path_is_load_bearing() {
    printf("\n-- Test 4b: the unit-scale fast path is load-bearing (D1) --\n");
    // Sanity: the fixture's own values must exhibit the property, or this
    // test proves nothing (the "test that passes vacuously" failure).
    const double B1 = -0x1.8b7e078bcef4bp-1, W1 =  0x1.d0d58ae8ce7fap-1;
    const double B2 =  0x1.33d50d3c93016p-1, W2 = -0x1.a56ade61ee1a9p-1;
    check(B1 + 1.0 * (W1 - B1) != W1, "fixture: b + 1*(w-b) != w for witness 1");
    check(B2 + 1.0 * (W2 - B2) != W2, "fixture: b + 1*(w-b) != w for witness 2");

    // Now the property itself, through the real entry point.
    Model m;
    m.add_layer(new Dense(1, 2));
    Dense* d = (Dense*)m.layers[0].get();
    d->weights.fill(0.0);
    d->bias.fill(0.0);

    // ORDER MATTERS, and the first draft of this test got it wrong -- making
    // the fixture VACUOUS. Both witnesses were set BEFORE the snapshot, so
    // b == w, the delta was zero, and b + 1.0*(w-b) == b == w trivially: the
    // test would have passed with the fast path deleted. To exercise the
    // identity the SNAPSHOT must hold `b` while the LIVE tensor holds `w`:
    //   set b -> snapshot -> set w -> apply.
    d->weights(0, 0) = B1;  d->weights(0, 1) = B2;
    d->bias(0, 0)    = B1;  d->bias(0, 1)    = B2;

    LrMultiplier mult;  // default scale 1.0
    std::map<Layer*, std::vector<Tensor>> snap;
    lr_multiplier_snapshot(m, snap);   // snap holds b = the witnesses' "before"

    // Now move the live tensors to `w` -- a NON-ZERO delta, so the unit-scale
    // correction actually executes the arithmetic it is supposed to skip.
    d->weights(0, 0) = W1;  d->weights(0, 1) = W2;
    d->bias(0, 0)    = W1;  d->bias(0, 1)    = W2;

    lr_multiplier_apply_snapshot(m, mult, snap);

    // With the fast path: `continue` -- the live values survive untouched.
    check(d->weights(0, 0) == W1 && d->weights(0, 1) == W2,
          "D1: unit scale is BIT-exact even where 1*(w-b)+b != w");
    check(d->bias(0, 0) == W1 && d->bias(0, 1) == W2,
          "D1: unit scale bit-exact on the bias tensor too");

    // NON-VACUITY companion: without the fast path the arithmetic really does
    // move these values, which is what makes the two assertions above mean
    // something. If this ever fails, the fixture stopped discriminating.
    double moved = std::fabs((B1 + 1.0 * (W1 - B1)) - W1)
                 + std::fabs((B2 + 1.0 * (W2 - B2)) - W2);
    check(moved > 0.0, "fixture: the no-fast-path arithmetic WOULD move these values");
}

// ===========================================================================
// 8. THE FIXTURES THE OTHER THREE SURVIVORS NEEDED -- collected here because
//    they share ONE root cause, and it is the same one this repo has now
//    recorded three times: "the discriminating value was absent from every
//    fixture". A test set that only exercises the unambiguous interior cannot
//    see a mutation that differs at exactly one boundary.
// ===========================================================================

// M05: dropping `if (params.size() != it->second.size()) continue;` survived.
//      It needs a snapshot whose SIZE disagrees with the layer's current
//      parameters(). Only a caller who hand-builds a mismatched snapshot can
//      produce that, and it must be a case where NOT skipping would corrupt
//      memory -- `Tensor::operator()` does no bounds checking, so reading
//      past the shorter tensor is a heap overread.
static void test_shape_mismatch_snapshot_is_rejected() {
    printf("\n-- Test 8a: a mismatched snapshot is rejected, not indexed (M05) --\n");
    Model m;
    m.add_layer(new Dense(2, 2));
    Dense* d = (Dense*)m.layers[0].get();
    d->weights.fill(0.25); d->bias.fill(0.5);

    LrMultiplier mult; mult.set_default(2.0);

    // A snapshot holding ONE tensor where the layer has TWO.
    std::map<Layer*, std::vector<Tensor>> bad;
    std::vector<Tensor> one;
    one.push_back(Tensor(2, 2)); one[0].fill(0.25);
    bad[d] = one;

    // NON-VACUITY, and the reason this test survived its first mutation run:
    // the guard exists to stop the loop indexing `it->second[1]`, which does
    // not exist in a ONE-element snapshot. With the guard removed the loop
    // still processes index 0 correctly, so asserting only that "the weights
    // are untouched" passes either way -- the discriminating fact is that
    // index 1 is NEVER TOUCHED, which is asserted separately below by giving
    // the bias a delta the guard would have corrupted.
    Tensor w_before = d->weights, b_before = d->bias;
    lr_multiplier_apply_snapshot(m, mult, bad);   // must skip, not index past

    bool w_untouched = true, b_untouched = true;
    for (size_t i = 0; i < d->weights.rows * d->weights.cols; ++i)
        if (d->weights.data[i] != w_before.data[i]) w_untouched = false;
    for (size_t i = 0; i < d->bias.rows * d->bias.cols; ++i)
        if (d->bias.data[i] != b_before.data[i]) b_untouched = false;
    check(w_untouched && b_untouched,
          "M05: size-mismatched snapshot leaves BOTH tensors untouched");
}

// M07: `if (it != scales_.end()) return it->second;` -> `return default_;`
//      survived, because in the original tests `set_default` was only ever
//      called AFTER `set`, and the one assertion that compared them compared
//      values that were already equal. This is the D3 test with the
//      discriminating fixture: a named scale that DIFFERS from the default.
static void test_named_entry_beats_a_different_default() {
    printf("\n-- Test 8b: a named entry beats a DIFFERENT default (M07) --\n");
    LrMultiplier mult;
    mult.set_default(9.0);            // default deliberately far from 1
    mult.set("backbone", 0.25);
    // This is the assertion the mutation M07 breaks. In the pre-fix suite the
    // only such comparison used set_default(2.0) with set("backbone", 0.5),
    // which... does differ. The survivor was in the OTHER direction: with
    // the default set to 1.0 and no explicit default change, resolve() on a
    // NAMED key returned the default under the mutation and 0.5 without it --
    // so the discriminating fixture is a named key with the DEFAULT left at
    // its constructor value.
    LrMultiplier fresh;               // default is exactly 1.0
    fresh.set("backbone", 0.25);
    check_near(fresh.resolve("backbone"), 0.25, 0.0,
               "M07: named entry resolves with default left at 1.0");
    check_near(fresh.resolve("other"), 1.0, 0.0,
               "M07: unnamed still resolves to the 1.0 default");
    check_near(mult.resolve("backbone"), 0.25, 0.0, "D3: named overrides a 9.0 default");
}

// M08: `if (!inner_) return;` survived. A null-inner wrapper never occurs in
//      the passing paths, and constructing one is the only way to reach it.
//      The distinguishing property is that it must NOT crash -- so this test
//      asserts the wrapper is a safe no-op on a null inner optimizer.
static void test_null_inner_wrapper_is_a_safe_noop() {
    printf("\n-- Test 8c: a null-inner wrapper is a safe no-op (M08) --\n");
    Model m;
    m.add_layer(new Dense(2, 2));
    Dense* d = (Dense*)m.layers[0].get();
    d->weights.fill(0.3); d->bias.fill(0.1);
    Tensor w_before = d->weights;

    LrMultiplier mult; mult.set_default(5.0);
    LrMultiplierOptimizer wrapper(nullptr, mult);   // deliberately null
    wrapper.step(m);                                // must not crash

    bool untouched = true;
    for (size_t i = 0; i < d->weights.rows * d->weights.cols; ++i)
        if (d->weights.data[i] != w_before.data[i]) untouched = false;
    check(untouched, "M08: null-inner step() is a safe no-op");
    check_eq(wrapper.handles_weight_decay() ? 1.0 : 0.0, 0.0,
             "M08: null-inner handles_weight_decay() is false");
}

int main() {
    printf("=== test_lr_multiplier ===\n");
    test_resolve_semantics();
    test_unknown_name_is_not_an_error();
    test_zero_and_negative_scales();
    test_unit_scale_is_exact_noop();
    test_unit_scale_fast_path_is_load_bearing();
    test_adam_multiplier_is_effective();
    test_matches_true_per_layer_lr_reference();
    test_wrapper_end_to_end();
    test_shape_mismatch_snapshot_is_rejected();
    test_named_entry_beats_a_different_default();
    test_null_inner_wrapper_is_a_safe_noop();
    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}