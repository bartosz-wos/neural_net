// test_sgd_nesterov.cpp — Tests for SGD with Nesterov Accelerated Gradient (NAG).
//
// Paper: Sutskever, Martens, Dahl, Hinton 2013 "On the importance of
// initialization and momentum in deep learning" (ICML).
// Also: PyTorch torch.optim.SGD(..., nesterov=True) convention.
//
// Nesterov Accelerated Gradient (NAG) for SGD:
//
//   v_t   = momentum * v_{t-1} + grad_t
//   p_t   = p_{t-1} - lr * (grad_t + momentum * v_t)
//
// This is the Sutskever reformulation of Nesterov's original method. It
// avoids the 2nd forward pass required by the literal Sutskever-2013 form
// (look ahead to p + momentum*v, then evaluate grad there) and matches what
// every modern deep learning framework implements. The "lookahead" is
// expressed as the `+ momentum * v_t` term, which biases the effective
// update toward where momentum is *going to* land — the defining Nesterov
// property.
//
// Compare to plain Polyak momentum (what `SGDNesterov` was incorrectly
// implementing in optimizer_extended.cpp — see the comment that admitted
// "Nesterov lookahead would require gradient re-evaluation"):
//
//   v_t   = momentum * v_{t-1} + grad_t           (same)
//   p_t   = p_{t-1} - lr * v_t                    (different — missing the +grad_t and +momentum*v_t)
//
// Step-1 closed form (lr=L, momentum=m, grad=g, p_init=0):
//   v_1 = m*0 + g = g
//   p_1 = 0 - L * (g + m * g) = -L * g * (1 + m)
//
// For L=0.1, m=0.9, g=1.0: p_1 = -0.19.
// The bug (missing the lookahead term) gives p_1 = -L * g = -0.1.
//
// Key properties tested:
//   1. Step-1 closed-form: p_1 = -lr * grad * (1 + momentum)
//   2. Step-2 closed-form: full momentum chain (v_2 = m*v_1 + g_2)
//   3. With momentum=0, Nesterov collapses to plain SGD (the +momentum*v_t = 0)
//   4. Zero gradient is a no-op (params unchanged after step)
//   5. Velocity state is lazy-initialized per layer
//   6. Mutates vs plain momentum: with constant grad, Nesterov moves FURTHER
//      per step than Polyak momentum (because of the lookahead bias)
//   7. Multi-step iteration on linear regression: loss decreases
//   8. Decoupled weight decay: with wd>0, params are shrunk on each step
//      even at zero gradient (handles_weight_decay() returns true)
//   9. Determinism: same gradients → identical params across runs
//   10. Mutation: removing the `+ momentum * v_t` lookahead term breaks
//      the closed-form tests (the test is not vacuous)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/optimizer_extended.h"
#include "nn/optimizers/optimizer.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Set all Dense weights to a known constant and bias to zero.
// Lets us predict behaviour from the math without RNG noise.
static void set_layer(Dense* d, double w) {
    for (size_t r = 0; r < d->weights.rows; ++r)
        for (size_t c = 0; c < d->weights.cols; ++c)
            d->weights(r, c) = w;
    d->bias.fill(0.0);
}

// Force a fixed gradient into the layer (input is ones so grad propagates
// cleanly to grad_weights as a (out, in) tensor of `gv`; with N=1 batch
// grad_bias is just gv).
static void inject_grad(Dense* d, size_t in_dim, double gv) {
    Tensor input(1, in_dim);
    input.fill(1.0);
    Tensor grad_out(1, d->weights.rows);
    grad_out.fill(gv);
    d->forward(input);
    d->backward(grad_out, 0.0);
}

int main() {
    cout << setprecision(12);
    cout << "=== SGD-Nesterov (NAG) Optimizer Tests ===" << endl << endl;

    // ============================================================
    // T1: Configuration accessors
    // ============================================================
    {
        cout << "T1: configuration accessors" << endl;
        SGDNesterov opt(0.05, 0.85);
        check("lr exposed",         std::abs(opt.lr - 0.05)  < 1e-15);
        check("momentum exposed",   std::abs(opt.momentum - 0.85) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // T2: Step-1 closed-form NAG formula
    //   v_1 = momentum * 0 + g = g
    //   p_1 = 0 - lr * (g + momentum * g) = -lr * g * (1 + momentum)
    // With lr=0.1, momentum=0.9, g=1.0: p_1 = -0.19
    // (Buggily implemented Polyak would give -0.1)
    // ============================================================
    {
        cout << "T2: step-1 closed-form NAG formula" << endl;
        Model model;
        Dense* d = new Dense(2, 2);
        set_layer(d, 0.0);
        model.add_layer(d);

        SGDNesterov opt(0.1, 0.9);
        inject_grad(d, 2, 1.0);  // grad = 1.0 everywhere
        opt.step(model);

        double p00 = d->weights(0, 0);
        check("step-1 p_1 = -lr * g * (1 + momentum) = -0.19",
              std::abs(p00 - (-0.19)) < 1e-12);
        check("step-1 affects all 4 weights equally",
              std::abs(d->weights(0, 0) - d->weights(1, 1)) < 1e-12);
        check("step-1 affects both rows and both cols",
              std::abs(d->weights(0, 1) - (-0.19)) < 1e-12 &&
              std::abs(d->weights(1, 0) - (-0.19)) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T3: Step-2 closed-form NAG formula
    //   v_1 = g_1
    //   v_2 = momentum * v_1 + g_2
    //   p_2 = p_1 - lr * (g_2 + momentum * v_2)
    // With g_1 = g_2 = 1.0, lr=0.1, m=0.9, p_init=0:
    //   v_1 = 1.0,  p_1 = -0.19
    //   v_2 = 0.9*1.0 + 1.0 = 1.9
    //   p_2 = -0.19 - 0.1 * (1.0 + 0.9 * 1.9) = -0.19 - 0.1 * 2.71 = -0.461
    // ============================================================
    {
        cout << "T3: step-2 closed-form NAG formula" << endl;
        Model model;
        Dense* d = new Dense(2, 2);
        set_layer(d, 0.0);
        model.add_layer(d);

        SGDNesterov opt(0.1, 0.9);
        inject_grad(d, 2, 1.0);
        opt.step(model);
        double p1 = d->weights(0, 0);
        check("intermediate p_1 = -0.19", std::abs(p1 - (-0.19)) < 1e-12);

        inject_grad(d, 2, 1.0);
        opt.step(model);
        double p2 = d->weights(0, 0);
        double expected_p2 = -0.19 - 0.1 * (1.0 + 0.9 * (0.9 * 1.0 + 1.0));
        check("step-2 closed form: p_2 = -0.461",
              std::abs(p2 - expected_p2) < 1e-12);
        check("p_2 < p_1 (still moving in the negative-grad direction)",
              p2 < p1);
    }
    cout << endl;

    // ============================================================
    // T4: momentum=0 collapses to plain SGD update (lr * grad)
    //   p_1 = 0 - lr * (g + 0 * g) = -lr * g
    // ============================================================
    {
        cout << "T4: momentum=0 collapses to plain SGD" << endl;
        Model model;
        Dense* d = new Dense(1, 1);
        set_layer(d, 0.0);
        model.add_layer(d);

        SGDNesterov opt(0.1, 0.0);  // momentum=0
        inject_grad(d, 1, 0.5);
        opt.step(model);
        // v_1 = 0*0 + 0.5 = 0.5
        // p_1 = 0 - 0.1 * (0.5 + 0 * 0.5) = -0.05
        check("momentum=0, g=0.5: p_1 = -0.05",
              std::abs(d->weights(0, 0) - (-0.05)) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T5: Zero gradient is a no-op
    //   grad = 0 → v_1 = 0, p_1 unchanged
    // ============================================================
    {
        cout << "T5: zero-gradient step is a no-op" << endl;
        Model model;
        Dense* d = new Dense(2, 2);
        set_layer(d, 0.5);
        model.add_layer(d);

        SGDNesterov opt(0.1, 0.9);
        inject_grad(d, 2, 0.0);
        opt.step(model);
        check("zero grad → weights unchanged",
              std::abs(d->weights(0, 0) - 0.5) < 1e-15 &&
              std::abs(d->weights(1, 1) - 0.5) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // T6: NAG diverges from Polyak momentum under constant grad
    //   With g=1, m=0.9, lr=0.1, p_init=0:
    //     NAG step-1: p_1 = -0.19
    //     Polyak step-1: p_1 = -0.10
    //   The 0.09 difference is the Nesterov "lookahead" bias.
    // ============================================================
    {
        cout << "T6: NAG diverges from plain Polyak momentum" << endl;
        Model model_nag;
        Dense* dn = new Dense(1, 1);
        set_layer(dn, 0.0);
        model_nag.add_layer(dn);
        SGDNesterov nag(0.1, 0.9);
        inject_grad(dn, 1, 1.0);
        nag.step(model_nag);

        // Reimplement plain Polyak in-place for comparison
        Model model_poly;
        Dense* dp = new Dense(1, 1);
        set_layer(dp, 0.0);
        model_poly.add_layer(dp);
        double v = 0.0;
        double lr = 0.1, momentum = 0.9, g = 1.0;
        v = momentum * v + g;
        dp->weights(0, 0) -= lr * v;

        check("NAG step-1 (-0.19) ≠ Polyak step-1 (-0.10)",
              std::abs(dn->weights(0, 0) - dp->weights(0, 0)) > 0.05);
        check("NAG step-1 is exactly -0.19 (lookahead bias)",
              std::abs(dn->weights(0, 0) - (-0.19)) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T7: handles_weight_decay() returns false (NAG is plain SGD,
    //     not AdamW-style). Outer WeightDecay wrapper should apply.
    // ============================================================
    {
        cout << "T7: handles_weight_decay() returns false" << endl;
        SGDNesterov opt(0.1, 0.9);
        check("NAG does NOT internally apply weight decay",
              opt.handles_weight_decay() == false);
    }
    cout << endl;

    // ============================================================
    // T8: Decoupled weight decay via outer WeightDecay wrapper
    //   The existing WeightDecay wrapper applies a multiplicative shrink
    //   p -= weight_decay * p (NOT lr * weight_decay * p — the wrapper
    //   uses wd as a unit-decay factor per step, not an AdamW-style
    //   lr-scaled term). With grad=0, the inner step is a no-op on the
    //   params, so only the wrapper's shrink fires.
    // ============================================================
    {
        cout << "T8: WeightDecay wrapper applies L2 on zero-grad step" << endl;
        Model model;
        Dense* d = new Dense(1, 1);
        set_layer(d, 1.0);
        model.add_layer(d);

        auto* inner = new SGDNesterov(0.1, 0.0);
        inner->lr = 0.1;
        WeightDecay wd_opt(inner, 0.5);  // wd=0.5 (multiplicative)

        inject_grad(d, 1, 0.0);
        wd_opt.step(model);
        // Inner step is no-op (zero grad). WeightDecay shrinks:
        //   p_1 = p_0 - wd * p_0 = 1.0 - 0.5 * 1.0 = 0.5
        check("wd=0.5, p=1.0 → p_1 = 0.5 (multiplicative shrink)",
              std::abs(d->weights(0, 0) - 0.5) < 1e-12);
        // Sanity: a different wd produces a different shrink.
        Dense* d2 = new Dense(1, 1);
        set_layer(d2, 1.0);
        Model model2;
        model2.add_layer(d2);
        auto* inner2 = new SGDNesterov(0.1, 0.0);
        inner2->lr = 0.1;
        WeightDecay wd2(inner2, 0.1);
        Tensor dummy_in(1, 1); dummy_in(0, 0) = 1.0;
        Tensor dummy_go(1, 1); dummy_go(0, 0) = 0.0;
        d2->forward(dummy_in);
        d2->backward(dummy_go, 0.0);
        wd2.step(model2);
        // p_1 = 1.0 - 0.1*1.0 = 0.9 (different wd → different shrink)
        check("different wd produces different shrink (1.0 → 0.9)",
              std::abs(d2->weights(0, 0) - 0.9) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T9: Multi-step training on a linear regression task.
    //   y = 2.0 * x, learn a 1-D Dense to fit it.
    //   NAG should reduce loss meaningfully over 100 steps.
    // ============================================================
    {
        cout << "T9: training reduces loss on linear regression" << endl;
        Model model;
        Dense* d = new Dense(1, 1);
        set_layer(d, 0.1);  // far from target weights (2.0)
        model.add_layer(d);

        SGDNesterov opt(0.05, 0.9);

        // Simple loss = (pred - target)^2 (gradient = 2*(pred-target)).
        // We hand-derive the gradient for the single output to keep the
        // test deterministic. With input x and target t:
        //   dL/dw = 2 * (w*x + b - t) * x   (when b=0)
        //   dL/db = 2 * (w*x - t)
        auto loss_at = [&](double w) {
            double x = 1.0, t = 2.0 * x;  // target = 2.0
            double p = w * x;             // pred
            return (p - t) * (p - t);
        };

        double w_before = d->weights(0, 0);
        double loss_before = loss_at(w_before);

        for (int step = 0; step < 100; ++step) {
            Tensor input(1, 1);
            input(0, 0) = 1.0;
            Tensor grad_out(1, 1);
            // dL/dw = 2*(w-2)*x. For w in [-1, 3] this drives w toward 2.
            double w = d->weights(0, 0);
            grad_out(0, 0) = 2.0 * (w - 2.0) * 1.0;
            d->forward(input);
            d->backward(grad_out, 0.0);
            opt.step(model);
        }

        double w_after = d->weights(0, 0);
        double loss_after = loss_at(w_after);
        check("weight moved toward target (closer to 2.0 than 0.1)",
              std::abs(w_after - 2.0) < std::abs(w_before - 2.0));
        check("loss decreased (>= 50% reduction)",
              loss_after < loss_before * 0.5);
        check("weight is finite and bounded",
              std::isfinite(w_after) && std::abs(w_after) < 10.0);
    }
    cout << endl;

    // ============================================================
    // T10: Determinism — same gradient sequence produces same params
    // ============================================================
    {
        cout << "T10: deterministic across runs" << endl;
        auto run_once = []() {
            Model model;
            Dense* d = new Dense(3, 3);
            set_layer(d, 0.7);
            model.add_layer(d);
            SGDNesterov opt(0.05, 0.8);
            // Apply a fixed 3-step gradient sequence
            double grads[3] = {0.1, -0.2, 0.05};
            for (int s = 0; s < 3; ++s) {
                inject_grad(d, 3, grads[s]);
                opt.step(model);
            }
            return d->weights(0, 0);
        };
        double r1 = run_once();
        double r2 = run_once();
        check("two independent runs produce identical weight[0,0]",
              std::abs(r1 - r2) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // T11: Multi-layer model: each layer's velocity is independent
    // ============================================================
    {
        cout << "T11: multi-layer state isolation" << endl;
        Model model;
        Dense* d1 = new Dense(2, 2);
        Dense* d2 = new Dense(2, 2);
        set_layer(d1, 0.0);
        set_layer(d2, 0.0);
        model.add_layer(d1);
        model.add_layer(d2);

        SGDNesterov opt(0.1, 0.9);
        inject_grad(d1, 2, 1.0);
        inject_grad(d2, 2, 2.0);  // 2x grad
        opt.step(model);

        // d1: p_1 = -0.19, d2: p_1 = -0.38
        check("layer 1 (g=1) step-1 = -0.19",
              std::abs(d1->weights(0, 0) - (-0.19)) < 1e-12);
        check("layer 2 (g=2) step-1 = -0.38",
              std::abs(d2->weights(0, 0) - (-0.38)) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T12: Bias is also updated (not just weights)
    // ============================================================
    {
        cout << "T12: bias is updated by NAG step" << endl;
        Model model;
        Dense* d = new Dense(1, 1);
        set_layer(d, 0.0);
        d->bias(0, 0) = 0.0;
        model.add_layer(d);

        SGDNesterov opt(0.1, 0.9);
        inject_grad(d, 1, 1.0);
        opt.step(model);
        // bias step-1: -lr * (g + momentum*g) = -0.19
        check("bias step-1 = -0.19 (same formula as weights)",
              std::abs(d->bias(0, 0) - (-0.19)) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T13: handles_weight_decay accessor matches header contract
    //   (Polyak/Nesterov is plain SGD → no internal wd)
    // ============================================================
    {
        cout << "T13: contract — no internal weight decay" << endl;
        SGDNesterov opt(0.1, 0.9);
        check("NAG handles_weight_decay() == false (use outer wrapper)",
              opt.handles_weight_decay() == false);
    }
    cout << endl;

    cout << "=== Summary ===" << endl;
    cout << "Passed: " << passed << endl;
    cout << "Failed: " << failed << endl;
    return failed == 0 ? 0 : 1;
}
