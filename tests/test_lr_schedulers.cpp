// LR Scheduler tests — CosineAnnealingWarmRestarts / MultiStepLR /
//                     PolynomialLR / CyclicLR
//
// Each scheduler has:
//   1. Accessors (get_lr, configuration knobs).
//   2. Closed-form step formula (compare against hand-computed value at
//      several steps).
//   3. The "warm restart" / "milestone reached" / "cycle reset" behavior
//      that distinguishes it from a plain CosineAnnealingLR / StepLR.
//   4. LR is actually written into the wrapped Optimizer::lr when step()
//      is called (the whole point of the scheduler).
//   5. Property: monotonicity / periodicity / boundary behavior.
//
// All schedulers target the Optimizer::lr field on a wrapped optimizer
// via set_optimizer(), matching the existing scheduler convention.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/optimizers/scheduler.h"
#include "nn/optimizers/optimizer.h"

using namespace std;

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) { cout << "  [PASS] " << msg << endl; ++passed; }            \
        else       { cout << "  [FAIL] " << msg << endl; ++failed; }           \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, msg)                                  \
    do {                                                                       \
        double a_ = (actual);                                                   \
        double e_ = (expected);                                                 \
        if (std::fabs(a_ - e_) <= (tol)) {                                      \
            cout << "  [PASS] " << msg << " got=" << a_                         \
                 << " expected=" << e_ << endl;                                 \
            ++passed;                                                          \
        } else {                                                               \
            cout << "  [FAIL] " << msg << " got=" << a_                         \
                 << " expected=" << e_ << " diff=" << std::fabs(a_ - e_)       \
                 << endl;                                                      \
            ++failed;                                                          \
        }                                                                      \
    } while (0)

static Model make_empty_model() {
    Model m;
    return m;
}

// ---------------------------------------------------------------------------
// CosineAnnealingWarmRestarts (SGDR — Loshchilov & Hutter 2017)
// Convention (PyTorch-aligned): cycle 1 spans last_epoch=0..T_0-1. On the
// step that pushes last_epoch to T_0, last_epoch resets to 0 and cycle
// index bumps. So last_epoch never sits at T_0 within a cycle.
// ---------------------------------------------------------------------------

static void test_cawr_formula() {
    cout << "Test 1: CosineAnnealingWarmRestarts closed-form" << endl;
    CosineAnnealingWarmRestarts cawr(1.0, 10, 1, 0.0);
    // last_epoch=0: cos(0)=1, lr = 0.5*1*2 = 1.0
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12, "last_epoch=0 -> lr = eta_max");
    // 5 step_dry()s: last_epoch=5, cos(pi/2)=0, lr = 0.5*1*1 = 0.5
    for (int i = 0; i < 5; ++i) cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 0.5, 1e-12,
               "last_epoch=5 (half-cycle) -> lr = (eta_max+eta_min)/2 = 0.5");
    // 4 more: last_epoch=9 (last epoch of cycle 1), cos(9π/10) = -cos(π/10) ≈ -0.951
    //   lr = 0 + 0.5*1*(1 - 0.951) ≈ 0.0245
    for (int i = 0; i < 4; ++i) cawr.step_dry();
    double expected_9 = 0.5 * (1.0 + std::cos(9.0 * M_PI / 10.0));
    CHECK_NEAR(cawr.get_lr(), expected_9, 1e-12,
               "last_epoch=9 (last epoch of cycle 1) -> lr near eta_min (cos(9π/10) ≈ -0.951)");
    // 1 more: last_epoch goes 9→10, then reset: last_epoch=0 cycle 2, lr = 1.0
    cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12,
               "10th step: cycle reset triggered, lr back at eta_max (cycle 2 start)");
    // 1 more: last_epoch=1 cycle 2, T_cur=1, lr = 0.5*(1+cos(π/10))
    cawr.step_dry();
    double expected_cyc2 = 0.5 * (1.0 + std::cos(M_PI / 10.0));
    CHECK_NEAR(cawr.get_lr(), expected_cyc2, 1e-12,
               "11th step (T_cur=1 in cycle 2) -> lr = 0.5*(1+cos(π/10))");
    cout << endl;
}

static void test_cawr_T_mult_growth() {
    cout << "Test 2: CosineAnnealingWarmRestarts T_mult grows cycles" << endl;
    // T_0=2, T_mult=2 → cycle lengths are 2, 4, 8, 16, ...
    CosineAnnealingWarmRestarts cawr(1.0, 2, 2, 0.0);
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12, "cycle 1, T_cur=0 -> lr=1");
    cawr.step_dry(); // last_epoch=1, T_i=2, cos(π/2)=0, lr = 0.5
    CHECK_NEAR(cawr.get_lr(), 0.5, 1e-12, "cycle 1, T_cur=1 -> lr=0.5");
    // 1 more: last_epoch goes 1→2, reset to 0, cycle_index=1, T_i=4
    cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12,
               "cycle 2 starts (T_0 * T_mult^1 = 4), T_cur=0 -> lr=1");
    CHECK(cawr.get_cycle_index() == 1, "cycle_index incremented to 1");
    // 2 more: last_epoch=1 then 2 in cycle 2 (length 4). T_cur=2, cos(π/2)=0, lr=0.5
    cawr.step_dry();
    cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 0.5, 1e-12,
               "cycle 2 (length 4), T_cur=2 -> lr=0.5");
    // 2 more: last_epoch goes 2→3 then 3→4, second call resets to 0 cycle 3 (length 8)
    cawr.step_dry();
    cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12,
               "cycle 3 starts (T_0 * T_mult^2 = 8), T_cur=0 -> lr=1");
    CHECK(cawr.get_cycle_index() == 2, "cycle_index incremented to 2");
    cout << endl;
}

static void test_cawr_writes_optimizer_lr() {
    cout << "Test 3: CosineAnnealingWarmRestarts writes to Optimizer::lr" << endl;
    CosineAnnealingWarmRestarts cawr(0.5, 5, 1, 0.0);
    SGD opt(0.5);
    cawr.set_optimizer(&opt);
    // Before step(): the inherited Optimizer::lr is initialized to 0.001 (base default).
    // SGD has its own `lr` member that shadows the base — `opt.lr` reads SGD::lr,
    // while the scheduler writes to `Optimizer::lr` (the base member). This is a
    // known design wart in the existing scheduler convention; verify via base ptr.
    Optimizer* base = &opt;
    CHECK_NEAR(base->lr, 0.001, 1e-12, "before step(): Optimizer::lr is base default");
    // step 1 -> last_epoch=1, lr = 0 + 0.5*(0.5-0)*(1+cos(π/5))
    Model m = make_empty_model();
    cawr.step(m);
    double expected = 0.5 * (0.5 - 0.0) * (1.0 + std::cos(M_PI / 5.0));
    CHECK_NEAR(base->lr, expected, 1e-12,
               "step 1: Optimizer::lr updated to scheduler's computed lr");
    cout << endl;
}

static void test_cawr_eta_min_offset() {
    cout << "Test 4: CosineAnnealingWarmRestarts respects eta_min floor" << endl;
    // eta_max=1, eta_min=0.1, T_0=10
    CosineAnnealingWarmRestarts cawr(1.0, 10, 1, 0.1);
    // last_epoch=0: lr = 0.1 + 0.5*0.9*2 = 1.0
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12, "last_epoch=0 with eta_min=0.1 -> lr = eta_max=1");
    // 9 step_dry()s: last_epoch=9, cos(9π/10) ≈ -0.951, lr = 0.1 + 0.5*0.9*(1 - 0.951)
    //   ≈ 0.1 + 0.022 ≈ 0.122
    for (int i = 0; i < 9; ++i) cawr.step_dry();
    double expected_9 = 0.1 + 0.5 * 0.9 * (1.0 + std::cos(9.0 * M_PI / 10.0));
    CHECK_NEAR(cawr.get_lr(), expected_9, 1e-12,
               "last_epoch=9 -> lr near eta_min but not equal (cos(9π/10) ≈ -0.951)");
    // 1 more: cycle reset → lr = 1.0 again
    cawr.step_dry();
    CHECK_NEAR(cawr.get_lr(), 1.0, 1e-12,
               "10th step: cycle restart, lr back at eta_max=1");
    cout << endl;
}

static void test_cawr_accessors() {
    cout << "Test 5: CosineAnnealingWarmRestarts accessors" << endl;
    CosineAnnealingWarmRestarts cawr(2.0, 5, 2, 0.05);
    CHECK_NEAR(cawr.get_eta_max(), 2.0, 1e-12, "get_eta_max()");
    CHECK_NEAR(cawr.get_eta_min(), 0.05, 1e-12, "get_eta_min()");
    CHECK(cawr.get_T_0() == 5, "get_T_0()");
    CHECK(cawr.get_T_mult() == 2, "get_T_mult()");
    CHECK(cawr.get_last_epoch() == 0, "get_last_epoch() initial");
    CHECK(cawr.get_cycle_index() == 0, "get_cycle_index() initial");
    cawr.step_dry();
    CHECK(cawr.get_last_epoch() == 1, "get_last_epoch() advances");
    cout << endl;
}

// ---------------------------------------------------------------------------
// MultiStepLR
// ---------------------------------------------------------------------------

static void test_multistep_basic() {
    cout << "Test 6: MultiStepLR closed-form" << endl;
    // milestones [3, 7], gamma=0.5
    // last_epoch=0: lr = 1.0
    // last_epoch=1,2: lr = 1.0
    // last_epoch=3: lr = 0.5
    // last_epoch=4,5,6: lr = 0.5
    // last_epoch=7: lr = 0.25
    MultiStepLR mslr(1.0, {3, 7}, 0.5);
    CHECK_NEAR(mslr.get_lr(), 1.0, 1e-12, "epoch 0: no milestone yet -> lr = initial");
    mslr.step_dry();
    mslr.step_dry();
    CHECK_NEAR(mslr.get_lr(), 1.0, 1e-12, "epoch 2: still no milestone -> lr = initial");
    mslr.step_dry();
    CHECK_NEAR(mslr.get_lr(), 0.5, 1e-12, "epoch 3: hit first milestone -> lr *= gamma = 0.5");
    mslr.step_dry();
    mslr.step_dry();
    mslr.step_dry();
    CHECK_NEAR(mslr.get_lr(), 0.5, 1e-12, "epoch 6: between milestones -> lr unchanged");
    mslr.step_dry();
    CHECK_NEAR(mslr.get_lr(), 0.25, 1e-12, "epoch 7: hit second milestone -> lr *= gamma = 0.25");
    cout << endl;
}

static void test_multistep_past_last_milestone() {
    cout << "Test 7: MultiStepLR after last milestone" << endl;
    MultiStepLR mslr(1.0, {2, 4}, 0.1);
    for (int i = 0; i < 10; ++i) mslr.step_dry();
    // After epoch 4: 0.01, no more changes
    CHECK_NEAR(mslr.get_lr(), 0.01, 1e-12, "epoch 10: past all milestones -> lr stays at last value");
    cout << endl;
}

static void test_multistep_writes_optimizer_lr() {
    cout << "Test 8: MultiStepLR writes to Optimizer::lr" << endl;
    MultiStepLR mslr(0.5, {2}, 0.1);
    Adam opt(0.5);
    mslr.set_optimizer(&opt);
    Optimizer* base = &opt;
    Model m = make_empty_model();
    mslr.step(m); // epoch 1, no milestone yet, scheduler writes initial_lr=0.5
    CHECK_NEAR(base->lr, 0.5, 1e-12,
               "step 1 (epoch 1, no milestone): Optimizer::lr = initial_lr = 0.5");
    mslr.step(m); // epoch 2, milestone hit, lr = 0.5 * 0.1 = 0.05
    CHECK_NEAR(base->lr, 0.05, 1e-12,
               "step 2 (epoch 2, milestone): Optimizer::lr = 0.5 * 0.1 = 0.05");
    mslr.step(m); // epoch 3, past milestone, lr stays at 0.05
    CHECK_NEAR(base->lr, 0.05, 1e-12,
               "step 3 (epoch 3, past milestone): Optimizer::lr stays at 0.05");
    cout << endl;
}

static void test_multistep_accessors() {
    cout << "Test 9: MultiStepLR accessors" << endl;
    MultiStepLR mslr(1.0, {3, 6, 9}, 0.5);
    CHECK(mslr.get_milestones().size() == 3, "get_milestones() size");
    CHECK(mslr.get_milestones()[0] == 3, "milestone[0]");
    CHECK(mslr.get_milestones()[2] == 9, "milestone[2]");
    CHECK_NEAR(mslr.get_gamma(), 0.5, 1e-12, "get_gamma()");
    CHECK_NEAR(mslr.get_initial_lr(), 1.0, 1e-12, "get_initial_lr()");
    cout << endl;
}

static void test_multistep_unsorted_milestones() {
    cout << "Test 10: MultiStepLR auto-sorts milestones" << endl;
    MultiStepLR mslr(1.0, {9, 3, 6}, 0.5);  // unsorted
    CHECK(mslr.get_milestones()[0] == 3, "sorted[0] = 3");
    CHECK(mslr.get_milestones()[1] == 6, "sorted[1] = 6");
    CHECK(mslr.get_milestones()[2] == 9, "sorted[2] = 9");
    cout << endl;
}

// ---------------------------------------------------------------------------
// PolynomialLR
// ---------------------------------------------------------------------------

static void test_polynomial_linear() {
    cout << "Test 11: PolynomialLR linear decay (power=1)" << endl;
    // initial=1, end=0, max_epoch=10, power=1.0
    // lr = (1 - 0) * (1 - t/10)^1 + 0 = 1 - t/10
    PolynomialLR plr(1.0, 10, 1.0, 0.0);
    CHECK_NEAR(plr.get_lr(), 1.0, 1e-12, "epoch 0: lr = initial");
    for (int i = 1; i <= 10; ++i) {
        plr.step_dry();
        double expected = 1.0 - static_cast<double>(i) / 10.0;
        CHECK_NEAR(plr.get_lr(), expected, 1e-12,
                   "epoch " + std::to_string(i) + " (linear): lr = 1 - t/10");
    }
    cout << endl;
}

static void test_polynomial_quadratic() {
    cout << "Test 12: PolynomialLR quadratic decay (power=2)" << endl;
    // initial=1, end=0, max_epoch=10, power=2.0
    // lr = (1 - t/10)^2
    PolynomialLR plr(1.0, 10, 2.0, 0.0);
    CHECK_NEAR(plr.get_lr(), 1.0, 1e-12, "epoch 0: lr = initial (1-0)^2 = 1");
    plr.step_dry();
    CHECK_NEAR(plr.get_lr(), 0.81, 1e-12, "epoch 1: lr = 0.9^2 = 0.81");
    plr.step_dry();
    CHECK_NEAR(plr.get_lr(), 0.64, 1e-12, "epoch 2: lr = 0.8^2 = 0.64");
    plr.step_dry();
    plr.step_dry();
    plr.step_dry();
    CHECK_NEAR(plr.get_lr(), 0.25, 1e-12, "epoch 5: lr = 0.5^2 = 0.25");
    for (int i = 0; i < 5; ++i) plr.step_dry();
    CHECK_NEAR(plr.get_lr(), 0.0, 1e-12, "epoch 10: lr = 0^2 = 0");
    cout << endl;
}

static void test_polynomial_with_end_lr() {
    cout << "Test 13: PolynomialLR with non-zero end_lr" << endl;
    // initial=1, end=0.2, max_epoch=10, power=1.0
    // lr = (1 - 0.2) * (1 - t/10) + 0.2 = 0.8*(1 - t/10) + 0.2
    PolynomialLR plr(1.0, 10, 1.0, 0.2);
    CHECK_NEAR(plr.get_lr(), 1.0, 1e-12, "epoch 0: lr = initial");
    for (int i = 1; i <= 5; ++i) {
        plr.step_dry();
        double expected = 0.8 * (1.0 - static_cast<double>(i) / 10.0) + 0.2;
        CHECK_NEAR(plr.get_lr(), expected, 1e-12, "epoch " + std::to_string(i));
    }
    for (int i = 0; i < 5; ++i) plr.step_dry();
    CHECK_NEAR(plr.get_lr(), 0.2, 1e-12, "epoch 10: lr = end_lr");
    cout << endl;
}

static void test_polynomial_writes_optimizer_lr() {
    cout << "Test 14: PolynomialLR writes to Optimizer::lr" << endl;
    PolynomialLR plr(1.0, 4, 1.0, 0.0);
    Adam opt(1.0);
    plr.set_optimizer(&opt);
    Optimizer* base = &opt;
    Model m = make_empty_model();
    plr.step(m); // epoch 1 -> lr=0.75
    CHECK_NEAR(base->lr, 0.75, 1e-12, "step 1: Optimizer::lr = 0.75");
    plr.step(m); // epoch 2 -> lr=0.5
    CHECK_NEAR(base->lr, 0.5, 1e-12, "step 2: Optimizer::lr = 0.5");
    cout << endl;
}

static void test_polynomial_accessors() {
    cout << "Test 15: PolynomialLR accessors" << endl;
    PolynomialLR plr(2.0, 100, 0.5, 0.01);
    CHECK_NEAR(plr.get_initial_lr(), 2.0, 1e-12, "get_initial_lr()");
    CHECK(plr.get_max_epoch() == 100, "get_max_epoch()");
    CHECK_NEAR(plr.get_power(), 0.5, 1e-12, "get_power()");
    CHECK_NEAR(plr.get_end_lr(), 0.01, 1e-12, "get_end_lr()");
    cout << endl;
}

static void test_polynomial_clamping_past_max_epoch() {
    cout << "Test 16: PolynomialLR clamps past max_epoch" << endl;
    PolynomialLR plr(1.0, 4, 1.0, 0.0);
    for (int i = 0; i < 10; ++i) plr.step_dry();
    // After 10 step_dry()s, last_epoch > max_epoch, lr should clamp to end_lr=0
    CHECK_NEAR(plr.get_lr(), 0.0, 1e-12,
               "epoch 10 (past max_epoch=4): lr stays clamped at end_lr");
    cout << endl;
}

// ---------------------------------------------------------------------------
// CyclicLR — triangular policy (simplest, Smith 2017)
// Convention (PyTorch-aligned): at iteration 0, lr = base_lr.
// lr rises to max_lr by step_size/2, falls to base_lr at step_size, then
// continues down/up in the next half of the cycle.
// ---------------------------------------------------------------------------

static void test_cyclic_triangular_basic() {
    cout << "Test 17: CyclicLR triangular policy closed-form" << endl;
    // base=0.1, max=1.0, step_size=4 (4 steps up, 4 steps down, total cycle 8)
    CyclicLR cyc(0.1, 1.0, 4, CyclicLR::Policy::TRIANGULAR);
    // iteration=0: cycle=floor(0/8)=0, x=|0/4 - 0 - 1|=|0-1|=1, scale=max(0,1-1)=0
    //   lr = base + amp*0 = base
    CHECK_NEAR(cyc.get_lr(), 0.1, 1e-12, "t=0: x=1, scale=0 -> lr = base_lr");
    cyc.step_dry();
    // t=1: cycle=0, x=|0.25-1|=0.75, scale=0.25
    //   lr = 0.1 + 0.9*0.25 = 0.325
    CHECK_NEAR(cyc.get_lr(), 0.325, 1e-12, "t=1: x=0.75, scale=0.25, lr=base+amp*0.25");
    cyc.step_dry();
    cyc.step_dry();
    // t=3: cycle=0, x=|0.75-1|=0.25, scale=0.75
    //   lr = 0.1 + 0.9*0.75 = 0.775
    CHECK_NEAR(cyc.get_lr(), 0.775, 1e-12, "t=3: x=0.25, scale=0.75, lr=base+amp*0.75");
    cyc.step_dry();
    // t=4: cycle=0, x=|1-1|=0, scale=1 -> lr = max
    CHECK_NEAR(cyc.get_lr(), 1.0, 1e-12, "t=4: x=0, scale=1, lr = max_lr (peak)");
    cyc.step_dry();
    // t=5: cycle=0, x=|1.25-1|=0.25, scale=0.75 -> lr = base + amp*0.75 = 0.775
    CHECK_NEAR(cyc.get_lr(), 0.775, 1e-12, "t=5: descending, x=0.25, scale=0.75, lr=0.775");
    cyc.step_dry();
    cyc.step_dry();
    cyc.step_dry();
    // t=8: cycle=floor(8/8)=1, x=|2 - 2 - 1|=1, scale=0 -> lr = base
    CHECK_NEAR(cyc.get_lr(), 0.1, 1e-12, "t=8: end of cycle 1, x=1, lr = base_lr (valley)");
    cout << endl;
}

static void test_cyclic_writes_optimizer_lr() {
    cout << "Test 18: CyclicLR writes to Optimizer::lr" << endl;
    CyclicLR cyc(0.01, 0.1, 4);
    SGD opt(0.01);
    cyc.set_optimizer(&opt);
    Optimizer* base = &opt;
    Model m = make_empty_model();
    cyc.step(m);
    // step 1: t=1, x=|0.25-1|=0.75, scale=0.25, lr = 0.01 + 0.09*0.25 = 0.0325
    CHECK_NEAR(base->lr, 0.0325, 1e-12, "step 1: Optimizer::lr updated to scheduler value");
    cout << endl;
}

static void test_cyclic_triangular2() {
    cout << "Test 19: CyclicLR triangular2 policy (amplitude halves each cycle)" << endl;
    // base=0, max=1, step_size=4
    // triangular2: amp' = amp / 2^cycle, where cycle = floor(t / (2*step_size))
    CyclicLR cyc(0.0, 1.0, 4, CyclicLR::Policy::TRIANGULAR2);
    // t=0: cycle=0, amp=1, x=1, scale=0, lr = 0 + 1*0 = 0
    CHECK_NEAR(cyc.get_lr(), 0.0, 1e-12, "cycle 0, t=0: x=1, lr=base=0");
    cyc.step_dry();
    // t=1: cycle=0, amp=1, x=0.75, scale=0.25, lr = 0 + 1*0.25 = 0.25
    CHECK_NEAR(cyc.get_lr(), 0.25, 1e-12, "cycle 0, t=1: scale=0.25, lr=0.25");
    // t=8: cycle=1, amp=0.5, x=|2-2-1|=1, scale=0, lr=0
    for (int i = 0; i < 7; ++i) cyc.step_dry();
    CHECK_NEAR(cyc.get_lr(), 0.0, 1e-12, "cycle 1, t=8: amp halved, valley lr=0");
    // t=9: cycle=1, amp=0.5, x=|2.25-3|=0.75, scale=0.25, lr=0+0.5*0.25=0.125
    cyc.step_dry();
    CHECK_NEAR(cyc.get_lr(), 0.125, 1e-12, "cycle 1, t=9: amp halved, scale=0.25, lr=0.125");
    // t=16: cycle=2, amp=0.25, x=1, scale=0
    for (int i = 0; i < 7; ++i) cyc.step_dry();
    CHECK_NEAR(cyc.get_lr(), 0.0, 1e-12, "cycle 2, t=16: amp halved again, valley lr=0");
    cout << endl;
}

static void test_cyclic_exp_range() {
    cout << "Test 20: CyclicLR exp_range policy (geometric decay of amplitude)" << endl;
    // base=0, max=1, step_size=4, gamma=0.5
    // exp_range: lr = base + amp * max(0, 1 - x) * gamma^t
    CyclicLR cyc(0.0, 1.0, 4, CyclicLR::Policy::EXP_RANGE, 0.5);
    // t=0: x=1, scale=0, gamma^0=1, lr = 0 + 1*0*1 = 0
    CHECK_NEAR(cyc.get_lr(), 0.0, 1e-12, "t=0: at valley, lr=0");
    cyc.step_dry();
    // t=1: x=|0.25-1|=0.75, scale=0.25, gamma^1=0.5, lr = 0 + 1*0.25*0.5 = 0.125
    CHECK_NEAR(cyc.get_lr(), 0.125, 1e-12, "t=1: amp decays by gamma=0.5");
    cyc.step_dry();
    // t=2: x=|0.5-1|=0.5, scale=0.5, gamma^2=0.25, lr = 0 + 1*0.5*0.25 = 0.125
    CHECK_NEAR(cyc.get_lr(), 0.125, 1e-12, "t=2: gamma^2=0.25, lr=0.125");
    cout << endl;
}

static void test_cyclic_accessors() {
    cout << "Test 21: CyclicLR accessors" << endl;
    CyclicLR cyc(0.001, 0.01, 100, CyclicLR::Policy::TRIANGULAR2, 0.999);
    CHECK_NEAR(cyc.get_base_lr(), 0.001, 1e-12, "get_base_lr()");
    CHECK_NEAR(cyc.get_max_lr(), 0.01, 1e-12, "get_max_lr()");
    CHECK(cyc.get_step_size() == 100, "get_step_size()");
    CHECK(cyc.get_policy() == CyclicLR::Policy::TRIANGULAR2, "get_policy()");
    CHECK_NEAR(cyc.get_gamma(), 0.999, 1e-12, "get_gamma()");
    CHECK(cyc.get_last_step() == 0, "get_last_step() initial");
    cyc.step_dry();
    CHECK(cyc.get_last_step() == 1, "get_last_step() advances");
    cout << endl;
}

// ---------------------------------------------------------------------------
// Integration: all four schedulers work end-to-end with a Model and Optimizer
// ---------------------------------------------------------------------------

static void test_end_to_end_lr_trajectory() {
    cout << "Test 22: end-to-end LR trajectory via Model+Optimizer" << endl;
    Model m = make_empty_model();
    m.add_layer(new Dense(2, 1));
    SGD opt(1.0);
    CosineAnnealingWarmRestarts cawr(1.0, 4, 2, 0.0);
    cawr.set_optimizer(&opt);
    Optimizer* base = &opt;
    std::vector<double> lrs;
    for (int i = 0; i < 5; ++i) {
        cawr.step(m);
        lrs.push_back(base->lr);
    }
    // Each cawr.step() advances last_epoch by 1 (PyTorch convention).
    // After 5 steps: last_epoch went 1→2→3→4→5.
    // last_epoch=4 triggers reset (cycle 1 length is 4), so last_epoch resets to 0
    //   and cycle_index bumps to 1 (T_i = 8 for cycle 2).
    // step 1 (last_epoch=1, T_i=4): cos(π/4)≈0.707, lr = 0.5*(1+0.707) ≈ 0.854
    double expected_1 = 0.5 * (1.0 + std::cos(M_PI / 4.0));
    CHECK_NEAR(lrs[0], expected_1, 1e-12,
               "step 1 (last_epoch=1 in cycle 1): cos(π/4)≈0.707, lr≈0.854");
    // step 2 (last_epoch=2): cos(π/2)=0, lr=0.5
    CHECK_NEAR(lrs[1], 0.5, 1e-12, "step 2 (last_epoch=2): cos(π/2)=0, lr=0.5");
    // step 3 (last_epoch=3): cos(3π/4)≈-0.707, lr≈0.146
    double expected_3 = 0.5 * (1.0 + std::cos(3.0 * M_PI / 4.0));
    CHECK_NEAR(lrs[2], expected_3, 1e-12,
               "step 3 (last_epoch=3): cos(3π/4)≈-0.707, lr≈0.146");
    // step 4: last_epoch 3→4, reset to 0 cycle 2 (T_i=8). lr = 0.5*2 = 1.0
    CHECK_NEAR(lrs[3], 1.0, 1e-12,
               "step 4: cycle 2 starts (T_i=8), lr=1.0");
    // step 5: last_epoch=1 cycle 2, T_i=8. cos(π/8)≈0.924, lr = 0.5*(1+0.924) ≈ 0.962
    double expected_5 = 0.5 * (1.0 + std::cos(M_PI / 8.0));
    CHECK_NEAR(lrs[4], expected_5, 1e-12,
               "step 5 (last_epoch=1 in cycle 2, T_i=8): cos(π/8)≈0.924, lr≈0.962");
    cout << endl;
}

static void test_mutation_vacuity_polynomial() {
    cout << "Test 23: mutation test — polynomial formula must actually decay" << endl;
    PolynomialLR plr(1.0, 4, 1.0, 0.0);
    plr.step_dry(); // epoch 1
    // If the impl returned initial_lr (constant), plr.get_lr() would be 1.0 here,
    // not 0.75. This test would FAIL in that case — proving we're actually testing
    // the formula, not a stub.
    CHECK_NEAR(plr.get_lr(), 0.75, 1e-12,
               "polynomial actually decays; if it returned 1.0 (broken), this would fail");
    cout << endl;
}

static void test_mutation_vacuity_multistep() {
    cout << "Test 24: mutation test — MultiStepLR must actually decay at milestones" << endl;
    MultiStepLR mslr(1.0, {1}, 0.5);
    mslr.step_dry(); // epoch 1 -> milestone hit, lr *= 0.5 = 0.5
    // If the impl didn't decay, lr would still be 1.0
    CHECK_NEAR(mslr.get_lr(), 0.5, 1e-12,
               "multistep decays at milestones; if it didn't (broken), lr would still be 1.0");
    cout << endl;
}

static void test_mutation_vacuity_cyclic() {
    cout << "Test 25: mutation test — CyclicLR must actually oscillate" << endl;
    CyclicLR cyc(0.0, 1.0, 4);
    // At t=4, x=0, scale=1, lr = base + amp*1 = max = 1.0
    for (int i = 0; i < 4; ++i) cyc.step_dry();
    // If the impl returned constant 0 (broken), lr would be 0 here
    CHECK_NEAR(cyc.get_lr(), 1.0, 1e-12,
               "cyclic reaches max at peak; if it returned 0 (broken), lr would still be 0");
    cout << endl;
}

int main() {
    cout << setprecision(10);
    cout << "=== LR Scheduler Tests ===" << endl << endl;

    test_cawr_formula();
    test_cawr_T_mult_growth();
    test_cawr_writes_optimizer_lr();
    test_cawr_eta_min_offset();
    test_cawr_accessors();
    test_multistep_basic();
    test_multistep_past_last_milestone();
    test_multistep_writes_optimizer_lr();
    test_multistep_accessors();
    test_multistep_unsorted_milestones();
    test_polynomial_linear();
    test_polynomial_quadratic();
    test_polynomial_with_end_lr();
    test_polynomial_writes_optimizer_lr();
    test_polynomial_accessors();
    test_polynomial_clamping_past_max_epoch();
    test_cyclic_triangular_basic();
    test_cyclic_writes_optimizer_lr();
    test_cyclic_triangular2();
    test_cyclic_exp_range();
    test_cyclic_accessors();
    test_end_to_end_lr_trajectory();
    test_mutation_vacuity_polynomial();
    test_mutation_vacuity_multistep();
    test_mutation_vacuity_cyclic();

    cout << "=== Result: " << passed << "/" << (passed + failed) << " passed ===" << endl;
    return failed == 0 ? 0 : 1;
}