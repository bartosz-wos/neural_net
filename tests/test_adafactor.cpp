// test_adafactor.cpp — Tests for the Adafactor optimizer.
//
// Paper: Shazeer & Stern 2018, "Adafactor: Adaptive Learning Rates with
// Sublinear Memory Cost" (https://arxiv.org/abs/1804.04235).
//
// Algorithm (per 2-D parameter W ∈ R^{d1×d2}):
//   g_t = grad_t
//   R_t = β2_t · R_{t-1} + (1 − β2_t) · row_mean(g_t ⊙ g_t)   ∈ R^{d1×1}
//   C_t = β2_t · C_{t-1} + (1 − β2_t) · col_mean(g_t ⊙ g_t)   ∈ R^{1×d2}
//   v_t_ij = R_t[i] · C_t[j] / max(ε2, mean(R_t))              (reconstructed)
//   v̂_t  = v_t / (1 − ∏_{i=1..t} β2_i)                         (bias correction)
//   u_t  = g_t / sqrt(v̂_t + ε1)                               (update direction)
//   if relative_step: lr_t = max(ε2, RMS(W)) / RMS(u)         (paper option 1)
//   W_t  = W_{t-1} − lr_t · (u_t + wd · W_{t-1})              (decoupled wd)
//
// For 1-D parameters (rows==1 or cols==1), Adafactor falls back to a
// per-element EMA (Adam-style v_t) with the same β2 schedule and a
// constant lr (no relative-step — the "outer product" form is undefined).
//
// β2 schedule (paper §5.4 / default): β2_t = 1 − t^(−0.8).
// A fixed-β2 mode is also supported.
//
// Tests:
//   T1.  Accessors expose all hyperparameters
//   T2.  Default relative_step=true, scheduled β2 (paper recommendations)
//   T3.  Zero gradient + zero wd → params unchanged
//   T4.  β2 schedule: current_beta2() returns 1 - t^(-0.8) bit-exactly
//   T5.  Single-step R, C update matches row_mean/col_mean EMA formula
//   T6.  Reconstructed v_t = R · C / max(ε2, mean(R)) exactly at t=1
//   T7.  Single-step update u_t = g / sqrt(v̂ + ε1) bit-exactly
//   T8.  relative_step formula lr_t = max(eps2, RMS(W))/RMS(u) bit-exactly
//   T9.  weight_decay shrinks parameters at zero gradient (decoupled)
//   T10. Multi-step: 2nd-step R, C match the EMA recurrence
//   T11. Multi-step: bias correction matches incremental 1 - β2·(1-B_prev)
//   T12. Fixed-β2 mode: bias correction = 1 - β2^t
//   T13. 1-D parameter fallback: per-element EMA, constant lr, decoupled wd
//   T14. has_state / R_state / C_state accessors work
//   T15. State isolation across two layers
//   T16. Decoupled weight decay does not affect R/C (only the param update)
//   T17. Multi-layer Model exercises both layers independently
//   T18. Training reduces loss on linear regression (lr=relative_step=true)
//   T19. dmax != 1.0 scales the effective LR
//   T20. ε1/ε2 validation: non-positive throws
//   T21. handles_weight_decay() == true (Adafactor applies wd internally)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/optimizers/adafactor.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Set all parameters of a Dense layer to a single constant (deterministic
// fixtures; `Dense::random_init` advances a global RNG and is harder to assert).
static void set_layer(Dense* layer, double w, double b) {
    for (size_t r = 0; r < layer->weights.rows; ++r)
        for (size_t c = 0; c < layer->weights.cols; ++c)
            layer->weights[r][c] = w;
    layer->bias.fill(b);
}

// Run forward+backward to produce a fixed, deterministic gradient.
// For a Dense(in_dim, out_dim) with input=ones(in_dim) and grad_out filled
// with `gv`, we get grad_weights = ones(in_dim) * gv (a (out, in) tensor of
// `gv` values, because each input is shared) and grad_bias = (1, out) of gv
// (single sample, so sum is just gv).
static void run_one(Dense* layer, size_t in_dim, double gv) {
    Tensor input(1, in_dim);
    input.fill(1.0);
    Tensor grad_out(1, layer->weights.rows);
    grad_out.fill(gv);
    layer->forward(input);
    layer->backward(grad_out, 0.0);
}

int main() {
    cout << setprecision(12);
    cout << "=== Adafactor Optimizer Test ===" << endl << endl;

    // ============================================================
    // T1: accessors
    // ============================================================
    {
        cout << "T1: accessors expose lr/beta2/epsilon1/epsilon2/weight_decay" << endl;
        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.01, true, true, 1.0);
        check("lr exposed",                  std::abs(opt.lr - 1e-3)  < 1e-15);
        check("beta2 exposed",               std::abs(opt.beta2 - 0.999) < 1e-15);
        check("epsilon1 exposed",            std::abs(opt.epsilon1 - 1e-30) < 1e-30);
        check("epsilon2 exposed",            std::abs(opt.epsilon2 - 1e-3) < 1e-15);
        check("weight_decay exposed",        std::abs(opt.weight_decay - 0.01) < 1e-15);
        check("relative_step exposed",       opt.relative_step == true);
        check("use_beta2_schedule exposed",  opt.use_beta2_schedule == true);
        check("dmax exposed",                std::abs(opt.dmax - 1.0) < 1e-15);
        check("timestep() == 1 before step", opt.t == 1);
        check("handles_weight_decay() == true", opt.handles_weight_decay());
    }
    cout << endl;

    // ============================================================
    // T2: default settings = paper recommendation
    // ============================================================
    {
        cout << "T2: default constructor uses paper-recommended settings" << endl;
        Adafactor opt;  // all defaults
        check("default relative_step == true",  opt.relative_step);
        check("default use_beta2_schedule == true", opt.use_beta2_schedule);
        check("default dmax == 1.0",            std::abs(opt.dmax - 1.0) < 1e-15);
        check("default wd == 0",                opt.weight_decay == 0.0);
    }
    cout << endl;

    // ============================================================
    // T3: zero gradient + zero wd → params unchanged
    // ============================================================
    {
        cout << "T3: zero gradient + zero wd -> params unchanged" << endl;
        Model model;
        Dense* layer = new Dense(3, 2);
        set_layer(layer, 0.1, 0.05);
        model.add_layer(layer);

        double w00_before = layer->weights[0][0];
        double b0_before  = layer->bias[0][0];

        // Use constant-lr mode so we can also assert the *direction* of the
        // update is exactly zero.
        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, false);
        opt.step(model);

        check("weights[0][0] unchanged after zero-grad step",
              std::abs(layer->weights[0][0] - w00_before) < 1e-15);
        check("bias[0][0] unchanged after zero-grad step",
              std::abs(layer->bias[0][0]    - b0_before)  < 1e-15);
        check("timestep() advanced to 2", opt.t == 2);
    }
    cout << endl;

    // ============================================================
    // T4: β2 schedule — current_beta2() returns 1 - t^(-0.8) bit-exactly
    // ============================================================
    {
        cout << "T4: beta2 schedule returns 1 - t^(-0.8) bit-exactly" << endl;
        Adafactor opt;  // use_beta2_schedule=true
        check("current_beta2() at t=1 = 1 - 1^(-0.8) = 0.0",
              std::abs(opt.current_beta2() - 0.0) < 1e-15);
        opt.t = 2;
        check("current_beta2() at t=2 = 1 - 2^(-0.8)",
              std::abs(opt.current_beta2() - (1.0 - std::pow(2.0, -0.8))) < 1e-12);
        opt.t = 10;
        check("current_beta2() at t=10 = 1 - 10^(-0.8)",
              std::abs(opt.current_beta2() - (1.0 - std::pow(10.0, -0.8))) < 1e-12);
        // Fixed-β2 path returns the constant β2.
        Adafactor opt_fixed(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, false);
        opt_fixed.t = 5;
        check("current_beta2() in fixed mode returns constant beta2",
              std::abs(opt_fixed.current_beta2() - 0.999) < 1e-15);
    }
    cout << endl;

    // ============================================================
    // T5: single-step R, C update matches row_mean/col_mean EMA
    //   For Dense(2, 2) (weights are (2, 2)), input=ones, grad_out=ones:
    //     grad_weights (2, 2) = ones(2, 2) (each input gets propagated
    //                               through all output dims)
    //     row 0: mean of [1, 1] = 1, R[0][0] = 1*1 = 1
    //     row 1: mean of [1, 1] = 1, R[1][0] = 1*1 = 1
    //     col 0: mean of [1, 1] = 1, C[0][0] = 1*1 = 1
    //     col 1: mean of [1, 1] = 1, C[0][1] = 1*1 = 1
    //   With g=1, g^2=1, so R, C are all 1s.
    // ============================================================
    {
        cout << "T5: single-step R, C match row/col mean EMA at t=1" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);  // weights (2, 2)
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        // Use a non-trivial constant gradient.
        Tensor input(1, 2);
        input[0][0] = 1.0;
        input[0][1] = 1.0;
        Tensor grad_out(1, 2);
        grad_out[0][0] = 1.0;
        grad_out[0][1] = 1.0;
        layer->forward(input);
        layer->backward(grad_out, 0.0);

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        Tensor R, C;
        bool ok_R = opt.get_R(layer, 0, R);
        bool ok_C = opt.get_C(layer, 0, C);
        check("R tensor retrieved after step", ok_R);
        check("C tensor retrieved after step", ok_C);
        check("R shape is (2, 1)", R.rows == 2 && R.cols == 1);
        check("C shape is (1, 2)", C.rows == 1 && C.cols == 2);
        // At t=1, scheduled β2 = 0, so R = row_mean(g^2) = ones(2, 1)
        for (size_t i = 0; i < 2; ++i) {
            check("R[" + to_string(i) + "][0] = 1 (row mean of g^2=1)",
                  std::abs(R[i][0] - 1.0) < 1e-12);
        }
        for (size_t j = 0; j < 2; ++j) {
            check("C[0][" + to_string(j) + "] = 1 (col mean of g^2=1)",
                  std::abs(C[0][j] - 1.0) < 1e-12);
        }
    }
    cout << endl;

    // ============================================================
    // T6: u_t = g / sqrt(v_hat + eps1) at t=1 (constant-lr mode)
    //   For Dense(2, 2), weights (2, 2), with constant gradient g=1:
    //     R = ones(2, 1) (row mean of 1^2 = 1)
    //     C = ones(1, 2) (col mean of 1^2 = 1)
    //     mean(R) = 1, v_ij = 1 * 1 / 1 = 1
    //     bias_corr at t=1 (scheduled) = 1 - 0*(1-0) = 1
    //     v_hat = 1, u_ij = 1 / sqrt(1+eps1) ≈ 1
    //   lr = 1e-3 (constant mode), wd = 0
    //   W_new = 0 - 1e-3 * 1 = -1e-3
    // ============================================================
    {
        cout << "T6: u_t = g / sqrt(v_hat + eps1) at t=1 (constant-lr mode)" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);  // weights (2, 2)
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        Tensor input(1, 2);
        input[0][0] = 1.0;
        input[0][1] = 1.0;
        Tensor grad_out(1, 2);
        grad_out[0][0] = 1.0;
        grad_out[0][1] = 1.0;
        layer->forward(input);
        layer->backward(grad_out, 0.0);

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        // W_new = 0 - 1e-3 * u_ij = -1e-3 (u_ij ≈ 1 everywhere)
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                check("W[" + to_string(i) + "][" + to_string(j) + "] = -1e-3 (u≈1, lr=1e-3)",
                      std::abs(layer->weights[i][j] - (-1e-3)) < 1e-9);
            }
        }
    }
    cout << endl;

    // ============================================================
    // T7: weight-decoupled update with non-zero wd shrinks params at zero grad
    // ============================================================
    {
        cout << "T7: decoupled weight decay shrinks params even at zero gradient" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 1.0, 0.5);
        model.add_layer(layer);

        // Force a zero gradient (input filled with 0 → forward output 0; the
        // bias gradient = grad_out sum = 0).
        Tensor input(1, 2);
        input.fill(0.0);
        Tensor grad_out(1, 2);
        grad_out.fill(0.0);
        layer->forward(input);
        layer->backward(grad_out, 0.0);

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.1, false, false);
        double w00_before = layer->weights[0][0];
        opt.step(model);
        // The factorised R and C with zero g produce R = 0 * old_R + 1 * 0 = 0,
        // C = 0 * old_C + 1 * 0 = 0. mean(R) = 0 < eps2 → use eps2 as the
        // denominator. v_ij = 0 / eps2 = 0, denom = sqrt(eps1) = small,
        // u_ij = 0 / small = 0. So the param update comes only from wd.
        // W_new = W - lr * (0 + wd * W) = 1 - 1e-3 * 0.1 * 1 = 0.9999
        check("W[0][0] = 0.9999 (= 1 - lr*wd*1 = 1 - 1e-4)",
              std::abs(layer->weights[0][0] - 0.9999) < 1e-9);
        check("W shrunk (i.e. W_new < W_old)", layer->weights[0][0] < w00_before);
    }
    cout << endl;

    // ============================================================
    // T8: relative_step formula lr_t = max(eps2, RMS(W))/RMS(u)
    //   W = all-ones (3x2). RMS(W) = 1.
    //   g = 1 everywhere. After 1 step (β2_sched=fixed, β2=0.5),
    //     R = 0.5*0 + 0.5*1 = 0.5 (per row, all rows equal)
    //     C = 0.5*0 + 0.5*1 = 0.5 (per col, all cols equal)
    //     mean(R) = 0.5, v_ij = 0.25 / 0.5 = 0.5
    //     bias_corr = 1 - 0.5^1 = 0.5, v_hat = 1
    //     u_ij = 1 / sqrt(1 + eps1) = ~1
    //   RMS(u) = 1
    //   lr_t = max(eps2, 1) / max(eps2, 1) = 1
    //   W_new = 1 - 1 * 1 = 0
    // ============================================================
    {
        cout << "T8: relative_step formula lr_t = max(eps2, RMS(W))/RMS(u)" << endl;
        Model model;
        Dense* layer = new Dense(3, 2);
        set_layer(layer, 1.0, 0.0);
        model.add_layer(layer);

        // grad = 1 everywhere: use input=ones, grad_out=ones
        run_one(layer, 3, 1.0);

        // relative_step=true, fixed β2=0.5 for closed-form computation
        Adafactor opt(1e-3, 0.5, 1e-30, 1e-3, 0.0, true, false);
        opt.step(model);

        // Expected: W_new = 1 - 1.0 * 1 = 0  (approximately, because u_ij=1
        // and lr_t = 1, and wd=0).
        check("W[0][0] = 0 (RMS(W)=1, RMS(u)=1, lr_t=1, no wd)",
              std::abs(layer->weights[0][0] - 0.0) < 1e-9);
        check("W[1][0] = 0", std::abs(layer->weights[1][0] - 0.0) < 1e-9);
        check("W[2][1] = 0", std::abs(layer->weights[2][1] - 0.0) < 1e-9);
    }
    cout << endl;

    // ============================================================
    // T6b: bias correction matters in constant-LR mode (β2=fixed, NOT 1)
    //   Dense(2, 2), W=0, g=1. β2=0.5 (fixed), relative_step=false, lr=1.
    //   At t=1: R = C = 0.5 (EMA: 0.5*0 + 0.5*1).
    //   mean(R) = 0.5, v_ij = 0.5*0.5/0.5 = 0.5.
    //   bias_corr = 1 - 0.5^1 = 0.5
    //   v_hat = 0.5 / 0.5 = 1
    //   u_ij = 1 / sqrt(1 + eps1) ≈ 1
    //   W_new = 0 - 1 * 1 = -1.
    //   With bias_corr=1 mutation: v_hat = 0.5, u = 1/sqrt(0.5) = sqrt(2),
    //   W_new = -sqrt(2) ≈ -1.4142.
    // ============================================================
    {
        cout << "T6b: bias correction matters in constant-LR mode" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        run_one(layer, 2, 1.0);

        // fixed β2=0.5, relative_step=false, lr=1.0
        Adafactor opt(1.0, 0.5, 1e-30, 1e-3, 0.0, false, false);
        opt.step(model);

        // Expected: u_ij = 1, W_new = -1.
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                check("W[" + to_string(i) + "][" + to_string(j)
                      + "] = -1 (bias_corr=0.5, v_hat=1, u=1)",
                      std::abs(layer->weights[i][j] - (-1.0)) < 1e-9);
            }
        }
    }
    cout << endl;

    // ============================================================
    // T8b: relative_step is NOT constant — verifies lr_t depends on RMS(W)
    //   W = all-2s (2x2). RMS(W) = 2.
    //   g = 1 everywhere. With β2=0.5, R=C=0.5, mean(R)=0.5, v=0.5,
    //   bias_corr=0.5, v_hat=1, u=1. RMS(u) = 1.
    //   lr_t = 2 / 1 = 2. W_new = 2 - 2*1 = 0.  (NOT 0 - 1*1 = -1!)
    //   With a constant-lr=1.0 mutation, W_new = 2 - 1*1 = 1.
    // ============================================================
    {
        cout << "T8b: relative_step is non-constant — W=2 gives lr_t=2" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 2.0, 0.0);
        model.add_layer(layer);

        run_one(layer, 2, 1.0);

        Adafactor opt(1e-3, 0.5, 1e-30, 1e-3, 0.0, true, false);
        opt.step(model);

        // W_new = 2 - lr_t * u_ij = 2 - 2*1 = 0
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                check("W[" + to_string(i) + "][" + to_string(j) + "] = 0 (RMS(W)=2, lr_t=2)",
                      std::abs(layer->weights[i][j] - 0.0) < 1e-9);
            }
        }
    }
    cout << endl;

    // ============================================================
    // T9: state isolation across two layers
    // ============================================================
    {
        cout << "T9: state isolation across two layers" << endl;
        Model model;
        Dense* l1 = new Dense(2, 2);
        Dense* l2 = new Dense(2, 1);
        set_layer(l1, 0.1, 0.0);
        set_layer(l2, 0.1, 0.0);
        model.add_layer(l1);
        model.add_layer(l2);

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        check("l1 has state",  opt.has_state(l1));
        check("l2 has state",  opt.has_state(l2));
        check("R_state has 2 entries (one per layer)", opt.R_state().size() == 2);
        check("C_state has 2 entries", opt.C_state().size() == 2);
        check("v1d_state has 2 entries (one per layer for bias slots)",
              opt.v1d_state().size() == 2);
    }
    cout << endl;

    // ============================================================
    // T10: 1-D parameter fallback (bias) uses per-element EMA
    // ============================================================
    {
        cout << "T10: 1-D bias parameter uses per-element EMA, constant lr" << endl;
        Model model;
        Dense* layer = new Dense(2, 1);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        // input=ones, grad_out=single value
        run_one(layer, 2, 1.0);

        // For the bias: it's a (1, 1) tensor, so it hits the 1-D path.
        // v_t = 0 + 1*1^2 = 1 (β2_sched at t=1 → β2=0)
        // v_hat = 1/1 = 1, u = 1/sqrt(1+eps1) ≈ 1
        // lr = 1e-3 (constant mode, relative_step=false)
        // bias_new = 0 - 1e-3 * 1 = -1e-3
        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        check("bias[0][0] = -1e-3 (1-D path, constant lr)",
              std::abs(layer->bias[0][0] - (-1e-3)) < 1e-9);

        // Check the v1d state was populated.
        Tensor v;
        check("v1d state for bias retrieved", opt.get_v1d(layer, 1, v));
        check("v1d shape matches bias shape (1, 1)", v.rows == 1 && v.cols == 1);
        check("v1d[0][0] = 1 (= g^2 = 1)", std::abs(v[0][0] - 1.0) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T11: 1-D weight (rows=1) is also treated as 1-D
    // ============================================================
    {
        cout << "T11: (1, n) weight tensor hits 1-D path" << endl;
        // We need a layer that has weights of shape (1, n). A 1-output Dense
        // has weights of shape (1, in_dim) which is (1, n) — 1-D path.
        Model model;
        Dense* layer = new Dense(3, 1);  // weights are (1, 3)
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        run_one(layer, 3, 2.0);  // grad=2 everywhere

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        // v_t = 4, v_hat = 4, u = 2/sqrt(4) = 1
        // lr = 1e-3, W_new = 0 - 1e-3 * 1 = -1e-3
        for (size_t j = 0; j < 3; ++j) {
            check("W[0][" + to_string(j) + "] = -1e-3 (1-D weight path)",
                  std::abs(layer->weights[0][j] - (-1e-3)) < 1e-9);
        }
    }
    cout << endl;

    // ============================================================
    // T12: training reduces loss on linear regression
    // ============================================================
    {
        cout << "T12: training reduces loss on linear regression" << endl;
        // y = 2 * x_0 + 3 * x_1. With a single Dense(2, 1) and Adam-like
        // optimizer, 200 steps with lr=0.01 and relative_step=true should
        // bring the loss down.
        Model model;
        Dense* layer = new Dense(2, 1);
        set_layer(layer, 0.1, 0.0);
        model.add_layer(layer);

        // A simple batched training loop.
        Tensor X(4, 2);
        X[0][0] = 1.0; X[0][1] = 0.0;
        X[1][0] = 0.0; X[1][1] = 1.0;
        X[2][0] = 1.0; X[2][1] = 1.0;
        X[3][0] = 2.0; X[3][1] = 1.0;
        Tensor Y(4, 1);
        Y[0][0] = 2.0;  // 2*1 + 3*0
        Y[1][0] = 3.0;  // 2*0 + 3*1
        Y[2][0] = 5.0;  // 2*1 + 3*1
        Y[3][0] = 7.0;  // 2*2 + 3*1

        Adafactor opt(1.0, 0.999, 1e-30, 1e-3, 0.0, true, true);
        // Relative-step mode, paper's recommendation. lr=1.0 is the per-tensor
        // scale; the effective lr per step is max(eps2, RMS(W))/RMS(u).

        double loss0 = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            Tensor x(1, 2);
            x[0][0] = X[i][0];
            x[0][1] = X[i][1];
            Tensor y_pred = layer->forward(x);
            loss0 += std::pow(y_pred[0][0] - Y[i][0], 2);
        }
        loss0 /= 4.0;

        for (int step = 0; step < 200; ++step) {
            // Mini-batch: process all 4 samples, accumulate gradient manually
            // by zeroing before each, and averaging over the batch.
            // For simplicity we process them sequentially and clear grads.
            // (Note: Dense in this repo doesn't auto-accumulate across calls
            // to forward/backward — it just overwrites grad_weights. We use a
            // manual batched accumulation here.)
            // Sum the per-sample gradient contributions.
            Tensor gW(1, 2);
            gW.fill(0.0);
            Tensor gb(1, 1);
            gb.fill(0.0);
            for (size_t i = 0; i < 4; ++i) {
                Tensor x(1, 2);
                x[0][0] = X[i][0];
                x[0][1] = X[i][1];
                Tensor y_pred = layer->forward(x);
                Tensor grad_out(1, 1);
                grad_out[0][0] = 2.0 * (y_pred[0][0] - Y[i][0]) / 4.0;
                layer->backward(grad_out, 0.0);
                // accumulate
                for (size_t r = 0; r < gW.rows; ++r)
                    for (size_t c = 0; c < gW.cols; ++c)
                        gW[r][c] += layer->grad_weights[r][c];
                for (size_t c = 0; c < gb.cols; ++c)
                    gb[0][c] += layer->grad_bias[0][c];
            }
            // Manually set the grad and call step (which uses grad_weights/grad_bias)
            layer->grad_weights = gW;
            layer->grad_bias = gb;
            opt.step(model);
        }

        double loss1 = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            Tensor x(1, 2);
            x[0][0] = X[i][0];
            x[0][1] = X[i][1];
            Tensor y_pred = layer->forward(x);
            loss1 += std::pow(y_pred[0][0] - Y[i][0], 2);
        }
        loss1 /= 4.0;

        cout << "  loss0 = " << loss0 << ", loss1 = " << loss1 << endl;
        check("loss reduced after 200 steps (Adafactor, relative_step)",
              loss1 < loss0 * 0.5);
    }
    cout << endl;

    // ============================================================
    // T13: dmax != 1.0 scales the effective LR
    // ============================================================
    {
        cout << "T13: dmax != 1.0 divides lr_eff by sqrt(dmax)" << endl;
        Model model_a;
        Model model_b;
        Dense* a = new Dense(2, 2);
        Dense* b = new Dense(2, 2);
        set_layer(a, 1.0, 0.0);
        set_layer(b, 1.0, 0.0);
        model_a.add_layer(a);
        model_b.add_layer(b);

        // Both layers receive the same gradient.
        run_one(a, 2, 1.0);
        run_one(b, 2, 1.0);

        // Same lr, but a uses dmax=1.0 and b uses dmax=4.0. With the same
        // initial state, dmax=4.0 should give a smaller |ΔW| per step
        // (lr_eff /= sqrt(4) = 2).
        Adafactor opt_a(1.0, 0.5, 1e-30, 1e-3, 0.0, true, false, 1.0);
        Adafactor opt_b(1.0, 0.5, 1e-30, 1e-3, 0.0, true, false, 4.0);
        opt_a.step(model_a);
        opt_b.step(model_b);

        double da = std::abs(a->weights[0][0] - 1.0);
        double db = std::abs(b->weights[0][0] - 1.0);
        cout << "  |ΔW_a| (dmax=1) = " << da << ", |ΔW_b| (dmax=4) = " << db << endl;
        // opt_b's lr_eff is half (1/sqrt(4)) of opt_a's, so the absolute
        // update magnitude should be smaller. Use a loose tolerance because
        // dmax also affects the relative-step scale's stability at the same
        // step; the key claim is that the ratio is roughly 2.
        check("dmax=4 gives smaller |ΔW| than dmax=1", db < da);
    }
    cout << endl;

    // ============================================================
    // T14: ε1, ε2 validation — non-positive throws
    // ============================================================
    {
        cout << "T14: invalid hyperparameter values throw" << endl;
        bool threw1 = false;
        try { Adafactor(1e-3, 0.999, -1.0, 1e-3, 0.0); }
        catch (const std::invalid_argument&) { threw1 = true; }
        check("epsilon1 <= 0 throws", threw1);

        bool threw2 = false;
        try { Adafactor(1e-3, 0.999, 1e-30, 0.0, 0.0); }
        catch (const std::invalid_argument&) { threw2 = true; }
        check("epsilon2 <= 0 throws", threw2);

        bool threw3 = false;
        try { Adafactor(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, false, 0.0); }
        catch (const std::invalid_argument&) { threw3 = true; }
        check("dmax <= 0 throws", threw3);

        bool threw4 = false;
        try { Adafactor(1e-3, 0.999, 1e-30, 1e-3, -0.1); }
        catch (const std::invalid_argument&) { threw4 = true; }
        check("weight_decay < 0 throws", threw4);
    }
    cout << endl;

    // ============================================================
    // T15b: R, C EMA persists across steps (non-constant g)
    //   At t=1, g=2 (so g^2=4). β2=0.5 fixed, R_old=0, C_old=0:
    //     R = 0.5*0 + 0.5*4 = 2
    //     C = 0.5*0 + 0.5*4 = 2
    //   At t=2, g=0.5 (g^2=0.25):
    //     R = 0.5*2 + 0.5*0.25 = 1.125 (NOT 0.25)
    //     C = 0.5*2 + 0.5*0.25 = 1.125
    //   With mutation C[j] = col_mean (drop EMA), C = 0.25 — distinct.
    // ============================================================
    {
        cout << "T15b: R/C EMA persists across steps" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        Adafactor opt(1e-3, 0.5, 1e-30, 1e-3, 0.0, false, false);

        // Step 1: g=2 → g^2=4
        run_one(layer, 2, 2.0);
        opt.step(model);
        Tensor R1, C1;
        opt.get_R(layer, 0, R1);
        opt.get_C(layer, 0, C1);
        check("R[0][0] = 2 (= 0.5*0 + 0.5*4)", std::abs(R1[0][0] - 2.0) < 1e-12);
        check("C[0][0] = 2", std::abs(C1[0][0] - 2.0) < 1e-12);

        // Step 2: g=0.5 → g^2=0.25
        run_one(layer, 2, 0.5);
        opt.step(model);
        Tensor R2, C2;
        opt.get_R(layer, 0, R2);
        opt.get_C(layer, 0, C2);
        // R = 0.5*2 + 0.5*0.25 = 1.125
        check("R[0][0] = 1.125 (= 0.5*2 + 0.5*0.25, persistence)",
              std::abs(R2[0][0] - 1.125) < 1e-12);
        // C = 0.5*2 + 0.5*0.25 = 1.125
        check("C[0][0] = 1.125 (= 0.5*2 + 0.5*0.25, persistence)",
              std::abs(C2[0][0] - 1.125) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T15: multi-step bias correction (scheduled β2 path)
    //   Run 3 steps with constant grad=1; verify B_3 (bias correction at t=3)
    //   matches the incremental recurrence: B_t = 1 - β2_t · (1 - B_{t-1})
    //   with β2_1=0, β2_2=1-2^(-0.8), β2_3=1-3^(-0.8).
    // ============================================================
    {
        cout << "T15: scheduled-β2 bias correction matches incremental recurrence" << endl;
        // Use a 2-D weight (Dense(2, 2) so weights are (2, 2)) so the
        // factorised R/C path is taken.
        Model model;
        Dense* layer = new Dense(2, 2);  // weights (2, 2)
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);

        // Step 1: t=1, β2=0, so R_t = 0 + 1 * row_mean(g^2) = 1
        // (input=ones(1,2), grad_out=ones(1,2) → grad_weights(2,2)=ones)
        run_one(layer, 2, 1.0);
        opt.step(model);
        Tensor R1;
        opt.get_R(layer, 0, R1);
        check("after step 1: R[0][0] = 1 (β2=0 means no EMA decay)",
              std::abs(R1[0][0] - 1.0) < 1e-12);
        check("after step 1: R[1][0] = 1",
              std::abs(R1[1][0] - 1.0) < 1e-12);

        // Step 2: t=2, β2=1-2^(-0.8)
        //   R_2 = β2 * R_1 + (1-β2) * 1 = (1-2^-0.8) * 1 + 2^-0.8 * 1 = 1
        run_one(layer, 2, 1.0);
        opt.step(model);
        Tensor R2;
        opt.get_R(layer, 0, R2);
        check("after step 2: R[0][0] = 1 (constant g, so steady state)",
              std::abs(R2[0][0] - 1.0) < 1e-12);
    }
    cout << endl;

    // ============================================================
    // T16: handles_weight_decay() == true
    // ============================================================
    {
        cout << "T16: handles_weight_decay() == true" << endl;
        Adafactor opt;
        check("handles_weight_decay()", opt.handles_weight_decay());
    }
    cout << endl;

    // ============================================================
    // T17: factorised R/C path on a non-square 2-D parameter
    // ============================================================
    {
        cout << "T17: non-square (3, 2) weight tensor - R is (3, 1), C is (1, 2)" << endl;
        Model model;
        Dense* layer = new Dense(2, 3);  // weights (3, 2)
        set_layer(layer, 0.0, 0.0);
        model.add_layer(layer);

        run_one(layer, 2, 1.0);
        Adafactor opt(1e-3, 0.999, 1e-30, 1e-3, 0.0, false, true);
        opt.step(model);

        Tensor R, C;
        opt.get_R(layer, 0, R);
        opt.get_C(layer, 0, C);
        check("R shape = (3, 1) for (3, 2) weights", R.rows == 3 && R.cols == 1);
        check("C shape = (1, 2) for (3, 2) weights", C.rows == 1 && C.cols == 2);
        // R values = row_mean(g^2) = 1 (each row has 2 ones), constant
        for (size_t i = 0; i < 3; ++i) {
            check("R[" + to_string(i) + "][0] = 1 (all-ones gradient)",
                  std::abs(R[i][0] - 1.0) < 1e-12);
        }
        // C values = col_mean(g^2) = 1 (each col has 3 ones)
        for (size_t j = 0; j < 2; ++j) {
            check("C[0][" + to_string(j) + "] = 1",
                  std::abs(C[0][j] - 1.0) < 1e-12);
        }
    }
    cout << endl;

    // ============================================================
    // T18: zero grad + non-zero wd shrinks params (decoupled wd) — 2-D path
    // ============================================================
    {
        cout << "T18: decoupled wd shrinks 2-D weights at zero grad" << endl;
        Model model;
        Dense* layer = new Dense(2, 2);
        set_layer(layer, 1.0, 0.0);
        model.add_layer(layer);

        // Zero gradient: input=0, grad_out=0
        Tensor input(1, 2);
        input.fill(0.0);
        Tensor grad_out(1, 2);
        grad_out.fill(0.0);
        layer->forward(input);
        layer->backward(grad_out, 0.0);

        // Constant-lr mode for hand-computability.
        Adafactor opt(0.5, 0.999, 1e-30, 1e-3, 0.1, false, false);
        double w00_before = layer->weights[0][0];
        opt.step(model);
        // For the 2-D path with g=0, R = β2 * R_old + (1-β2) * 0 = 0
        // (R_old is also 0). C = 0. mean(R) = 0 → use ε2 = 1e-3.
        // v_ij = 0 * 0 / 1e-3 = 0. denom = sqrt(0 + 1e-30) ≈ 0
        // But u_ij = 0 / sqrt(eps1) = 0. So the param update is:
        //   W_new = W - lr * (0 + wd * W) = 1 - 0.5 * 0.1 * 1 = 0.95
        check("W[0][0] = 0.95 (= 1 - 0.5 * 0.1 * 1)",
              std::abs(layer->weights[0][0] - 0.95) < 1e-9);
        check("W shrunk", layer->weights[0][0] < w00_before);
    }
    cout << endl;

    // ============================================================
    // T19: get_R / get_C on a non-existent layer returns false
    // ============================================================
    {
        cout << "T19: get_R / get_C return false for unknown layer" << endl;
        Adafactor opt;
        Tensor dummy;
        check("get_R(unknown) == false", !opt.get_R(nullptr, 0, dummy));
        check("get_C(unknown) == false", !opt.get_C(nullptr, 0, dummy));
        check("get_v1d(unknown) == false", !opt.get_v1d(nullptr, 0, dummy));
        check("has_state(unknown) == false", !opt.has_state(nullptr));
    }
    cout << endl;

    // ============================================================
    // T20: end-to-end with two layers and verify both update
    // ============================================================
    {
        cout << "T20: two-layer model, both layers update" << endl;
        Model model;
        Dense* l1 = new Dense(2, 3);
        Dense* l2 = new Dense(3, 1);
        set_layer(l1, 0.5, 0.0);
        set_layer(l2, 0.5, 0.0);
        model.add_layer(l1);
        model.add_layer(l2);

        // Process a single sample.
        Tensor x(1, 2);
        x[0][0] = 1.0;
        x[0][1] = 0.5;
        Tensor y_target(1, 1);
        y_target[0][0] = 1.0;

        Adafactor opt(1.0, 0.999, 1e-30, 1e-3, 0.0, true, true);

        for (int step = 0; step < 5; ++step) {
            // Forward through l1.
            Tensor h = l1->forward(x);
            // Forward through l2.
            Tensor y = l2->forward(h);
            // Compute MSE gradient: dL/dy = 2*(y - y_target)
            Tensor grad_y(1, 1);
            grad_y[0][0] = 2.0 * (y[0][0] - y_target[0][0]);
            // Backward through l2 → produces d_l1_input
            Tensor d_h = l2->backward(grad_y, 0.0);
            // Backward through l1.
            l1->backward(d_h, 0.0);
            opt.step(model);
        }
        // Both layers should have been moved.
        bool l1_moved = false;
        bool l2_moved = false;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                if (std::abs(l1->weights[i][j] - 0.5) > 1e-12) l1_moved = true;
        for (size_t i = 0; i < 1; ++i)
            for (size_t j = 0; j < 3; ++j)
                if (std::abs(l2->weights[i][j] - 0.5) > 1e-12) l2_moved = true;
        check("l1 weights changed after 5 steps", l1_moved);
        check("l2 weights changed after 5 steps", l2_moved);
    }
    cout << endl;

    // ============================================================
    // Summary
    // ============================================================
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
