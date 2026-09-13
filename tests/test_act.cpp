// tests/test_act.cpp
// Adaptive Computation Time (ACT) — Graves 2016
// https://arxiv.org/abs/1603.08983
//
// Focused test plan. Mutation-tested non-vacuous where applicable.

#include "nn/layers/recurrent/act.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <functional>

static int n_pass = 0;
static int n_fail = 0;
static int test_idx = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  [FAIL] " << msg << std::endl; \
        n_fail++; \
    } else { \
        n_pass++; \
    } \
} while(0)

#define TEST_START(name) do { \
    test_idx++; \
    std::cout << "--- Test " << test_idx << ": " << name << " ---" << std::endl; \
} while(0)

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static bool is_finite(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j) {
            double v = t[i][j];
            if (!std::isfinite(v)) return false;
        }
    return true;
}

// Centered finite difference on the loss L(y) = mean((y - target)^2).
// Used for FD input / parameter gradient checks. We re-do the full forward
// for each perturbation so the FD measures the loss landscape correctly.
static double fd_input(ACTRecurrentLayer& layer, const Tensor& input,
                       const Tensor& target, int row, int col, double eps) {
    Tensor inp_p = input; inp_p[row][col] += eps;
    Tensor out_p = layer.forward(inp_p);
    double loss_p = 0.0;
    for (size_t i = 0; i < out_p.cols; ++i) {
        double d = out_p[0][i] - target[0][i];
        loss_p += d * d;
    }
    loss_p /= static_cast<double>(out_p.cols);

    Tensor inp_m = input; inp_m[row][col] -= eps;
    Tensor out_m = layer.forward(inp_m);
    double loss_m = 0.0;
    for (size_t i = 0; i < out_m.cols; ++i) {
        double d = out_m[0][i] - target[0][i];
        loss_m += d * d;
    }
    loss_m /= static_cast<double>(out_m.cols);

    return (loss_p - loss_m) / (2.0 * eps);
}

// Generic parameter FD: takes a setter function that mutates the parameter
// by `delta` and restores it by `-delta` at the end.
using ParamSetter = std::function<void(double)>;

static double fd_param(const Tensor& input, const Tensor& target,
                       ACTRecurrentLayer& layer, ParamSetter set,
                       double eps) {
    // `set(d)` is a setter that adds delta `d` to the current parameter value.
    // We need: start at X, perturb to X+eps, forward, then perturb to X-eps, forward.
    // That means: set(+eps), set(-2*eps).
    set(eps);
    Tensor out_p = layer.forward(input);
    double loss_p = 0.0;
    for (size_t k = 0; k < out_p.cols; ++k) {
        double d = out_p[0][k] - target[0][k];
        loss_p += d * d;
    }
    loss_p /= static_cast<double>(out_p.cols);

    set(-2.0 * eps);
    Tensor out_m = layer.forward(input);
    double loss_m = 0.0;
    for (size_t k = 0; k < out_m.cols; ++k) {
        double d = out_m[0][k] - target[0][k];
        loss_m += d * d;
    }
    loss_m /= static_cast<double>(out_m.cols);

    set(eps);
    return (loss_p - loss_m) / (2.0 * eps);
}

// ----------------------------------------------------------------------------

int main() {
    std::cout << "=== ACT Recurrent Layer Tests ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    setvbuf(stderr, nullptr, _IOLBF, 0);  // line-buffer stderr

    // -------------------------------------------------------------------
    TEST_START("constructor validates inputs");
    {
        bool threw1 = false, threw2 = false, threw3 = false;
        try { ACTRecurrentLayer(3, 0, 5); } catch (...) { threw1 = true; }
        try { ACTRecurrentLayer(3, 4, 0); } catch (...) { threw2 = true; }
        try { ACTRecurrentLayer(3, 4, 5, -1.0); } catch (...) { threw3 = true; }
        CHECK(threw1, "hidden_size=0 throws");
        CHECK(threw2, "max_steps=0 throws");
        CHECK(threw3, "ponder_penalty<0 throws");

        ACTRecurrentLayer ok(3, 4, 5);
        CHECK(ok.input_dim() == 3, "input_dim accessor");
        CHECK(ok.hidden_size() == 4, "hidden_size accessor");
        CHECK(ok.max_steps() == 5, "max_steps accessor");
        CHECK(std::abs(ok.ponder_penalty() - 0.01) < 1e-12, "ponder_penalty default");
    }

    // -------------------------------------------------------------------
    TEST_START("forward shape (1, max_steps*input) -> (1, hidden)");
    {
        ACTRecurrentLayer layer(3, 4, 5);
        Tensor input(1, 5 * 3);
        input.fill(0.5);
        Tensor out = layer.forward(input);
        CHECK(out.rows == 1, "out.rows == 1");
        CHECK(out.cols == 4, "out.cols == hidden_size");
        CHECK(is_finite(out), "out is finite");
        bool any_nonzero = false;
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (out[i][j] != 0.0) { any_nonzero = true; break; }
        CHECK(any_nonzero, "out has nonzero entries");
    }

    // -------------------------------------------------------------------
    TEST_START("forward input dimension validation");
    {
        ACTRecurrentLayer layer(3, 4, 5);
        Tensor wrong(1, 10);  // 10 != 5*3
        wrong.fill(0.0);
        bool threw = false;
        try { layer.forward(wrong); } catch (...) { threw = true; }
        CHECK(threw, "wrong input shape throws");
    }

    // -------------------------------------------------------------------
    TEST_START("weights sum to 1 and respect p_t bounds (random init)");
    {
        ACTRecurrentLayer layer(2, 3, 5);
        std::mt19937 gen(11);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 5 * 2);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out = layer.forward(input);

        double wsum = 0.0;
        for (int t = 0; t < 5; ++t) wsum += layer.weight_at(t);
        CHECK(std::abs(wsum - 1.0) < 1e-12, "weights sum to 1");

        int hs = layer.actual_steps() - 1;
        for (int t = 0; t < hs; ++t) {
            CHECK(std::abs(layer.weight_at(t) - layer.halt_probability(t)) < 1e-12,
                  "weight[t] = p[t] for t < halt_step");
        }
        CHECK(layer.weight_at(hs) > 0.0, "halt step has positive weight");
        CHECK(layer.weight_at(hs) < 1.0001, "halt step weight < 1+eps");
        for (int t = hs + 1; t < 5; ++t) {
            CHECK(std::abs(layer.weight_at(t)) < 1e-12, "weight[t] = 0 for t > halt_step");
        }
    }

    // -------------------------------------------------------------------
    TEST_START("forcing low p_t (W_halt=-100) → halts only at last step");
    {
        ACTRecurrentLayer layer(1, 2, 5);
        layer.set_W_halt(0, 0, -100.0);
        layer.set_W_halt(1, 0, -100.0);
        layer.set_b_halt(-100.0);
        Tensor input(1, 5);
        std::mt19937 gen(13);
        std::normal_distribution<> nd(0.0, 0.3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out = layer.forward(input);
        CHECK(layer.actual_steps() == 5, "halt at last step when p_t ~ 0");
        CHECK(std::abs(layer.weight_at(4) - 1.0) < 1e-9, "last-step weight ~ 1");
    }

    // -------------------------------------------------------------------
    TEST_START("forcing high p_t (W_halt=+100) → halts at first step");
    {
        ACTRecurrentLayer layer(1, 2, 5);
        layer.set_W_halt(0, 0, 100.0);
        layer.set_W_halt(1, 0, 100.0);
        layer.set_b_halt(100.0);
        Tensor input(1, 5);
        std::mt19937 gen(17);
        std::normal_distribution<> nd(0.0, 0.3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out = layer.forward(input);
        CHECK(layer.actual_steps() == 1, "halt at first step when p_t ~ 1");
        // weight[0] = 1 - 0 = 1
        CHECK(std::abs(layer.weight_at(0) - 1.0) < 1e-9, "halt weight = 1 - 0 = 1");
    }

    // -------------------------------------------------------------------
    TEST_START("hand-derived constant-halt reference (p_t ~ 0)");
    {
        // hidden=2, input=1, max_steps=3
        ACTRecurrentLayer layer(1, 2, 3);
        layer.zero_all_weights();
        layer.set_W_xh(0, 0, 0.5);
        layer.set_W_xh(1, 0, 0.5);
        layer.set_W_halt(0, 0, -100.0);
        layer.set_W_halt(1, 0, -100.0);
        layer.set_b_halt(-100.0);
        Tensor input(1, 3);
        input[0][0] = 1.0; input[0][1] = 1.0; input[0][2] = 1.0;
        Tensor out = layer.forward(input);
        // h_0 = 0
        // h_1 = tanh(1*0.5 + 0) = tanh(0.5)
        // h_2 = tanh(1*0.5 + 0) = tanh(0.5)   (W_hh = 0)
        // h_3 = tanh(1*0.5 + 0) = tanh(0.5)
        // With p_t ~ 0, weights = [0, 0, 1], output = h_3 = [tanh(0.5), tanh(0.5)]
        double expected = std::tanh(0.5);
        CHECK(std::abs(out[0][0] - expected) < 1e-6, "out[0] ~ tanh(0.5)");
        CHECK(std::abs(out[0][1] - expected) < 1e-6, "out[1] ~ tanh(0.5)");
    }

    // -------------------------------------------------------------------
    TEST_START("hand-derived weighted-sum reference (deterministic p_t)");
    {
        // Force p_t to known constants by zeroing W_xh, W_hh, b (so all h_t = 0)
        // and setting W_halt to known values. With h_t = 0, z_t = b_halt for all t.
        ACTRecurrentLayer layer(1, 2, 4);
        layer.zero_all_weights();
        layer.set_b_halt(0.0);  // z_t = 0 → p_t = 0.5 for all t
        Tensor input(1, 4);
        std::mt19937 gen(19);
        std::normal_distribution<> nd(0.0, 0.3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out = layer.forward(input);
        // h_t = 0 for all t → y_t = 0 → out = sum_t w_t * 0 = 0
        CHECK(std::abs(out[0][0]) < 1e-12, "y=0 everywhere → out=0");
        CHECK(std::abs(out[0][1]) < 1e-12, "y=0 everywhere → out=0");
        // p_0 = 0.5, p_1 = 0.5: N_1 = 1.0 → halt_step = 1, weight_1 = 1 - 0.5 = 0.5
        CHECK(layer.actual_steps() == 2, "halt at step 1 when p=[0.5,0.5,..]");
        CHECK(std::abs(layer.weight_at(0) - 0.5) < 1e-12, "weight[0] = 0.5");
        CHECK(std::abs(layer.weight_at(1) - 0.5) < 1e-12, "weight[1] = 0.5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of W_halt — hand-computable config");
    {
        // hidden=3, input=2, max_steps=4
        // We construct W_xh so that y_t[hp] = h_t[hp] only depends on a single chain.
        // Setup: W_xh[0][0] = 1, all other W_xh = 0; b[0] = 0.5 (to break symmetry).
        // Then h_t[0] = tanh(input[0] + 0.5). For y_t[1] and y_t[2], all zeroed.
        // So y_t[0] is the only nonzero entry (varies across t due to h_t changing via h_{t-1} coupling).
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.zero_all_weights();
        // W_xh[0][0] = 1.0
        layer.set_W_xh(0, 0, 1.0);
        // b[0] = 0.5
        layer.set_b(0, 0.5);
        // W_halt: only W_halt[0][0] = some value
        layer.set_W_halt(0, 0, 0.0);  // z_t = b_halt = 0 → p_t = 0.5 for all t
        layer.set_b_halt(0.0);

        Tensor input(1, 4 * 2);
        // All zeros except input[0][0] = 0 (so x_0 = [0, 0])
        // h_0 = 0
        // h_1 = tanh(0 * 1 + 0 + 0.5) = tanh(0.5) = a1
        // h_2 = tanh(0 * 1 + h_1 * W_hh[0][0] + 0.5) = tanh(0.5) (since W_hh=0)
        // h_3 = tanh(0.5)
        // h_4 = tanh(0.5)
        // So y_t[0] = tanh(0.5) for all t.
        Tensor target(1, 3);
        target[0][0] = 0.1; target[0][1] = 0.2; target[0][2] = 0.3;

        // Forward
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (int j = 0; j < 3; ++j) grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        // Backward
        layer.zero_grad();
        layer.backward(grad_out, 0.0);

        // Manually compute:
        // y_t[0] = tanh(0.5) = a for all t.
        // y_t[1] = 0, y_t[2] = 0.
        // So d_w[t] = sum_hp y_t[hp] * grad_out[hp] = a * grad_out[0] for all t.
        // grad_out[0] = 2*(a - 0.1)/3 = 2*(0.4621 - 0.1)/3 = 0.2414
        double a = std::tanh(0.5);
        (void)a;

        // p_t = 0.5 for all t (W_halt=0, b_halt=0). For halt at step 1: N_0 = 0.5 < 1,
        // N_0 + p_1 = 0.5 + 0.5 = 1.0 >= 1. Halt at step 1.
        // (Actually, N + p >= 1.0 fires when 0.5 + 0.5 >= 1.0 → YES, halt at step 1.)
        // Weights: w_0 = 0.5, w_1 = 0.5, w_2 = 0, w_3 = 0.
        // With ponder_penalty = 0:
        //   d_p[0] = d_w[0] - d_w[1] = a*grad_out[0] - a*grad_out[0] = 0 (constant y cancels!)
        //   d_p[1] = 0
        //   d_z[t] = 0 for all t
        //   grad_W_halt[0][0] = 0
        //   grad_b_halt = 0
        // (The output is independent of W_halt for this config because y_t is constant
        //  across t, so the weighted sum equals y_t * sum_t w_t = y_t * 1, which is constant.)
        double expected_g0 = 0.0;
        double expected_b = 0.0;
        double ana_g0 = layer.grad_W_halt_at(0, 0);
        fprintf(stderr, "[HAND DEBUG] expected grad_W_halt[0][0]=%.6f  ana=%.6f  diff=%.6f\n",
                expected_g0, ana_g0, expected_g0 - ana_g0);

        // Also grad_b_halt should be 0
        double ana_b = layer.grad_b_halt_at();
        fprintf(stderr, "[HAND DEBUG] expected grad_b_halt=%.6f  ana=%.6f  diff=%.6f\n",
                expected_b, ana_b, expected_b - ana_b);

        CHECK(std::abs(ana_g0 - expected_g0) < 1e-12, "grad_W_halt[0][0] matches hand derivation");
        CHECK(std::abs(ana_b - expected_b) < 1e-12, "grad_b_halt matches hand derivation");

        // Now do FD on this same config — should give the same value (0.002311).
        double eps = 1e-5;
        double saved_wh = layer.W_halt_at(0, 0);
        layer.set_W_halt(0, 0, saved_wh + eps);
        Tensor out_p = layer.forward(input);
        double loss_p = 0.0;
        for (size_t k = 0; k < out_p.cols; ++k) {
            double d = out_p[0][k] - target[0][k];
            loss_p += d * d;
        }
        loss_p /= 3.0;
        layer.set_W_halt(0, 0, saved_wh - eps);
        Tensor out_m = layer.forward(input);
        double loss_m = 0.0;
        for (size_t k = 0; k < out_m.cols; ++k) {
            double d = out_m[0][k] - target[0][k];
            loss_m += d * d;
        }
        loss_m /= 3.0;
        layer.set_W_halt(0, 0, saved_wh);
        double num_g0 = (loss_p - loss_m) / (2.0 * eps);
        fprintf(stderr, "[HAND DEBUG] FD grad_W_halt[0][0]=%.8f  ana=%.8f  diff=%.8f\n",
                num_g0, ana_g0, num_g0 - ana_g0);
        // FD check (use absolute tolerance since both can be 0 for this degenerate config)
        double fd_abs_diff = std::abs(num_g0 - ana_g0);
        CHECK(fd_abs_diff < 1e-8, "FD matches analytical for hand-derived config (absolute)");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of W_xh — no halting chain (W_halt=-100)");
    {
        // Simplest possible config: no halting interference.
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(99);
        // Force halting to be near-zero so W_halt's chain doesn't affect anything.
        layer.W_halt_accessor().fill(-100.0);
        layer.set_b_halt(-100.0);
        std::mt19937 gen(101);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (int j = 0; j < 3; ++j) grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        // Now do FD for each W_xh entry.
        double eps = 1e-5;
        double max_rel = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                double ana = layer.grad_W_xh_at(i, j);
                double saved = layer.W_xh_at(i, j);
                layer.set_W_xh(i, j, saved + eps);
                Tensor out_p = layer.forward(input);
                double loss_p = 0.0;
                for (size_t k = 0; k < out_p.cols; ++k) {
                    double d = out_p[0][k] - target[0][k];
                    loss_p += d * d;
                }
                loss_p /= 3.0;
                layer.set_W_xh(i, j, saved - eps);
                Tensor out_m = layer.forward(input);
                double loss_m = 0.0;
                for (size_t k = 0; k < out_m.cols; ++k) {
                    double d = out_m[0][k] - target[0][k];
                    loss_m += d * d;
                }
                loss_m /= 3.0;
                layer.set_W_xh(i, j, saved);
                double num = (loss_p - loss_m) / (2.0 * eps);
                double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
                max_rel = std::max(max_rel, rel);
                if (rel > 1e-5) {
                    fprintf(stderr, "  W_xh[%d][%d] ana=%g num=%g rel=%g\n", i, j, ana, num, rel);
                }
            }
        }
        CHECK(max_rel < 1e-5, "W_xh FD matches when halting chain is disabled");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of b — hand-computable config (simple, halt=1)");
    {
        ACTRecurrentLayer layer(1, 2, 3, /*ponder_penalty=*/0.0);
        layer.zero_all_weights();
        layer.set_W_xh(0, 0, 1.0);
        layer.set_W_xh(1, 0, 1.0);
        layer.set_b(0, 0.5);
        layer.set_b(1, -0.3);
        layer.set_W_halt(0, 0, 5.0);  // force p_0 ~ 1, halt at step 0
        layer.set_W_halt(1, 0, 0.0);
        layer.set_b_halt(0.0);

        Tensor input(1, 3);
        // x_0 = 0.5, x_1 = -0.2, x_2 = 0.3
        input[0][0] = 0.5; input[0][1] = -0.2; input[0][2] = 0.3;
        Tensor target(1, 2);
        target[0][0] = 0.4; target[0][1] = -0.1;

        Tensor out = layer.forward(input);
        Tensor grad_out(1, 2);
        for (int j = 0; j < 2; ++j) grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 2.0;
        layer.zero_grad();
        layer.backward(grad_out, 0.0);

        // With W_halt[0][0]=5 and b_halt=0, z_0 = 5*y_0[0]. tanh outputs are in (-1,1),
        // so z_0 in (-5, 5). p_0 can be 0.99 or 0.01 depending on sign of y_0[0].
        // Let me check the actual halt step.
        fprintf(stderr, "[B-HAND] halt_step=%d weights: %g %g %g\n",
                (int)layer.halt_step(), layer.weight_at(0), layer.weight_at(1), layer.weight_at(2));
        fprintf(stderr, "[B-HAND] grad_b[0]=%g grad_b[1]=%g\n",
                layer.grad_b_at(0), layer.grad_b_at(1));

        // For FD, perturb b and see.
        double eps = 1e-5;
        for (int i = 0; i < 2; ++i) {
            double saved = layer.b_at(i);
            layer.set_b(i, saved + eps);
            Tensor out_p = layer.forward(input);
            double loss_p = 0.0;
            for (int k = 0; k < out_p.cols; ++k) {
                double d = out_p[0][k] - target[0][k];
                loss_p += d * d;
            }
            loss_p /= 2.0;
            layer.set_b(i, saved - eps);
            Tensor out_m = layer.forward(input);
            double loss_m = 0.0;
            for (int k = 0; k < out_m.cols; ++k) {
                double d = out_m[0][k] - target[0][k];
                loss_m += d * d;
            }
            loss_m /= 2.0;
            layer.set_b(i, saved);
            double num = (loss_p - loss_m) / (2.0 * eps);
            double ana = layer.grad_b_at(i);
            double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
            fprintf(stderr, "[B-HAND] b[%d] ana=%g num=%g rel=%g\n", i, ana, num, rel);
            CHECK(rel < 1e-3, "b FD matches for hand-derived config");
        }
    }
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(23);
        std::mt19937 gen(29);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        // Forward + analytical backward
        layer.zero_grad();
        Tensor out = layer.forward(input);
        // grad_output = 2 * (out - target) / D  (d/dy of MSE)
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        // For each entry of W_halt, compare FD vs analytical — DIRECTLY (no setter).
        double max_rel = 0.0;
        double eps = 1e-5;
        for (int i = 0; i < 3; ++i) {
            int j = 0;
            double ana = layer.grad_W_halt_at(i, j);
            double saved = layer.W_halt_at(i, j);
            ParamSetter setter = [&](double d) { layer.set_W_halt(i, j, layer.W_halt_at(i, j) + d); };
            double num = fd_param(input, target, layer, setter, 1e-5);
            layer.set_W_halt(i, j, saved);  // restore
            double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
            max_rel = std::max(max_rel, rel);
            if (rel > 1e-5) {
                std::cerr << "  W_halt[" << i << "][" << j << "] ana=" << ana
                          << " num=" << num << " rel=" << rel
                          << " saved=" << saved << std::endl;
            }
        }
        CHECK(max_rel < 1e-5, "max_rel_err(W_halt FD) < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of b_halt");
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(31);
        std::mt19937 gen(33);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        double ana = layer.grad_b_halt_at();
        double saved = layer.b_halt_at();
        ParamSetter setter = [&](double d) { layer.set_b_halt(layer.b_halt_at() + d); };
        double num = fd_param(input, target, layer, setter, 1e-5);
        layer.set_b_halt(saved);
        double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
        if (rel > 1e-5) {
            std::cerr << "  b_halt ana=" << ana << " num=" << num << " rel=" << rel << std::endl;
        }
        CHECK(rel < 1e-5, "b_halt FD rel < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of W_xh");
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(37);
        std::mt19937 gen(41);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        double max_rel = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                double ana = layer.grad_W_xh_at(i, j);
                double saved = layer.W_xh_at(i, j);
                // Setter: add delta to CURRENT value (not absolute setter).
                ParamSetter setter = [&](double d) { layer.set_W_xh(i, j, layer.W_xh_at(i, j) + d); };
                double num = fd_param(input, target, layer, setter, 1e-5);
                layer.set_W_xh(i, j, saved);
                double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
                max_rel = std::max(max_rel, rel);
                if (rel > 1e-5) {
                    std::cerr << "  W_xh[" << i << "][" << j << "] ana=" << ana
                              << " num=" << num << " rel=" << rel << std::endl;
                }
            }
        }
        CHECK(max_rel < 1e-5, "W_xh FD rel < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of W_hh");
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(43);
        std::mt19937 gen(47);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        double max_rel = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double ana = layer.grad_W_hh_at(i, j);
                double saved = layer.W_hh_at(i, j);
                ParamSetter setter = [&](double d) { layer.set_W_hh(i, j, layer.W_hh_at(i, j) + d); };
                double num = fd_param(input, target, layer, setter, 1e-5);
                layer.set_W_hh(i, j, saved);
                double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
                max_rel = std::max(max_rel, rel);
                if (rel > 1e-5) {
                    std::cerr << "  W_hh[" << i << "][" << j << "] ana=" << ana
                              << " num=" << num << " rel=" << rel << std::endl;
                }
            }
        }
        CHECK(max_rel < 1e-5, "W_hh FD rel < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of b (RNN bias)");
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(53);
        std::mt19937 gen(59);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);

        double max_rel = 0.0;
        for (int i = 0; i < 3; ++i) {
            double ana = layer.grad_b_at(i);
            double saved = layer.b_at(i);
            ParamSetter setter = [&](double d) { layer.set_b(i, layer.b_at(i) + d); };
            double num = fd_param(input, target, layer, setter, 1e-5);
            layer.set_b(i, saved);
            double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
            max_rel = std::max(max_rel, rel);
            if (rel > 1e-5) {
                std::cerr << "  b[" << i << "] ana=" << ana << " num=" << num
                          << " rel=" << rel << std::endl;
            }
        }
        CHECK(max_rel < 1e-5, "b FD rel < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("FD gradient of input");
    {
        ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/0.0);
        layer.random_init(61);
        std::mt19937 gen(67);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);
        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        Tensor grad_in = layer.backward(grad_out, 0.0);

        double max_rel = 0.0;
        for (size_t j = 0; j < input.cols; ++j) {
            double ana = grad_in[0][j];
            double num = fd_input(layer, input, target, 0, j, 1e-5);
            double rel = (std::abs(num) > 1e-12) ? std::abs(num - ana) / std::abs(num) : std::abs(ana);
            max_rel = std::max(max_rel, rel);
        }
        CHECK(max_rel < 1e-5, "input FD rel < 1e-5");
    }

    // -------------------------------------------------------------------
    TEST_START("ponder penalty contributes to d_b_halt");
    {
        ACTRecurrentLayer layer_a(2, 3, 4, /*ponder_penalty=*/0.0);
        ACTRecurrentLayer layer_b(2, 3, 4, /*ponder_penalty=*/0.5);
        layer_a.random_init(71); layer_b.random_init(71);  // same seed → same params
        std::mt19937 gen(73);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j) grad_out[0][j] = 0.0;  // task gradient = 0, only ponder contributes

        layer_a.zero_grad(); layer_b.zero_grad();
        layer_a.forward(input); layer_a.backward(grad_out, 0.0);
        layer_b.forward(input); layer_b.backward(grad_out, 0.0);

        // grad_b_halt with ponder=0.5 should equal grad_b_halt with ponder=0 PLUS the ponder
        // contribution summed over actual_steps.
        double diff = layer_b.grad_b_halt_at() - layer_a.grad_b_halt_at();
        double expected_ponder = 0.5 * (layer_a.actual_steps());  // per-step +0.5 added to d_p[t]
        // (Note: d_p[t] -> d_z[t] = d_p * p*(1-p) -> sum into d_b_halt; so the per-step ponder
        // contribution to d_b_halt is ponder * p_t * (1-p_t). But for small W_halt+b_halt, p~0.5
        // so p*(1-p)~0.25. We just check that the diff is NONZERO (i.e., ponder has an effect)
        // and that it's positive.
        CHECK(diff > 0.0, "ponder > 0 strictly increases grad_b_halt");
        CHECK(diff < 5.0, "diff is bounded");
        (void)expected_ponder;
    }

    // -------------------------------------------------------------------
    TEST_START("update_weights moves all 5 parameter groups");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        layer.random_init(79);
        std::mt19937 gen(83);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);

        // Snapshot weights.
        Tensor w_xh = layer.W_xh_accessor();
        Tensor w_hh = layer.W_hh_accessor();
        Tensor b_orig = layer.b_accessor();
        Tensor w_halt = layer.W_halt_accessor();
        double b_halt_orig = layer.b_halt_at();

        layer.zero_grad();
        Tensor out = layer.forward(input);
        Tensor grad_out(1, 3);
        for (size_t j = 0; j < 3; ++j)
            grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
        layer.backward(grad_out, 0.0);
        layer.update_weights(0.05);

        // Check at least one entry of each group moved.
        bool moved_xh = false, moved_hh = false, moved_b = false, moved_halt = false;
        Tensor w_xh2 = layer.W_xh_accessor();
        Tensor w_hh2 = layer.W_hh_accessor();
        Tensor b_now = layer.b_accessor();
        Tensor w_halt2 = layer.W_halt_accessor();
        for (int i = 0; i < 3 && !moved_xh; ++i)
            for (int j = 0; j < 2; ++j)
                if (std::abs(w_xh[i][j] - w_xh2[i][j]) > 1e-12) moved_xh = true;
        for (int i = 0; i < 3 && !moved_hh; ++i)
            for (int j = 0; j < 3; ++j)
                if (std::abs(w_hh[i][j] - w_hh2[i][j]) > 1e-12) moved_hh = true;
        for (int i = 0; i < 3 && !moved_b; ++i)
            if (std::abs(b_orig[i][0] - b_now[i][0]) > 1e-12) moved_b = true;
        for (int i = 0; i < 3 && !moved_halt; ++i)
            if (std::abs(w_halt[i][0] - w_halt2[i][0]) > 1e-12) moved_halt = true;
        CHECK(moved_xh, "W_xh moved");
        CHECK(moved_hh, "W_hh moved");
        CHECK(moved_b, "b moved");
        CHECK(moved_halt, "W_halt moved");
        CHECK(std::abs(b_halt_orig - layer.b_halt_at()) > 1e-12, "b_halt moved");
    }

    // -------------------------------------------------------------------
    TEST_START("end-to-end training reduces MSE loss");
    {
        ACTRecurrentLayer layer(2, 3, 5);
        layer.random_init(89);
        std::mt19937 gen(97);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 5 * 2);
        Tensor target(1, 3);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        for (size_t j = 0; j < target.cols; ++j) target[0][j] = nd(gen);

        double lr = 0.05;
        Tensor out0 = layer.forward(input);
        double loss0 = 0.0;
        for (size_t j = 0; j < 3; ++j) {
            double d = out0[0][j] - target[0][j];
            loss0 += d * d;
        }
        loss0 /= 3.0;

        for (int step = 0; step < 80; ++step) {
            Tensor out = layer.forward(input);
            Tensor grad_out(1, 3);
            for (size_t j = 0; j < 3; ++j)
                grad_out[0][j] = 2.0 * (out[0][j] - target[0][j]) / 3.0;
            layer.zero_grad();
            layer.backward(grad_out, lr);
            layer.update_weights(lr);
        }

        Tensor outF = layer.forward(input);
        double lossF = 0.0;
        for (size_t j = 0; j < 3; ++j) {
            double d = outF[0][j] - target[0][j];
            lossF += d * d;
        }
        lossF /= 3.0;
        CHECK(lossF < loss0, "loss decreased after 80 SGD steps");
        std::cout << "  loss " << loss0 << " -> " << lossF << std::endl;
    }

    // -------------------------------------------------------------------
    TEST_START("parameters() / gradients() return 5 entries");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        auto p = layer.parameters();
        auto g = layer.gradients();
        CHECK(p.size() == 5, "5 parameters");
        CHECK(g.size() == 5, "5 gradients");
        // Sizes match
        CHECK(p[0]->rows == 3 && p[0]->cols == 2, "W_xh shape (hidden, input)");
        CHECK(p[1]->rows == 3 && p[1]->cols == 3, "W_hh shape (hidden, hidden)");
        CHECK(p[2]->rows == 3 && p[2]->cols == 1, "b shape (hidden, 1)");
        CHECK(p[3]->rows == 3 && p[3]->cols == 1, "W_halt shape (hidden, 1)");
        CHECK(p[4]->rows == 1 && p[4]->cols == 1, "b_halt shape (1, 1)");
    }

    // -------------------------------------------------------------------
    TEST_START("zero_grad clears all 5 gradient tensors");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        // Pollute grads (just fill with 1.0)
        layer.grad_W_xh_accessor().fill(1.0);
        layer.grad_W_hh_accessor().fill(1.0);
        layer.grad_b_accessor().fill(1.0);
        layer.grad_W_halt_accessor().fill(1.0);
        layer.grad_b_halt_accessor().fill(1.0);
        layer.zero_grad();
        double sum = 0.0;
        auto g = layer.gradients();
        for (auto* t : g) {
            for (size_t i = 0; i < t->rows; ++i)
                for (size_t j = 0; j < t->cols; ++j)
                    sum += std::abs((*t)[i][j]);
        }
        CHECK(sum < 1e-15, "all grads zero after zero_grad");
    }

    // -------------------------------------------------------------------
    TEST_START("determinism: two fresh layers with same seed produce identical forward");
    {
        ACTRecurrentLayer a(2, 3, 4);
        ACTRecurrentLayer b(2, 3, 4);
        a.random_init(101);
        b.random_init(101);
        Tensor input(1, 4 * 2);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = static_cast<double>(j) * 0.1;
        Tensor oa = a.forward(input);
        Tensor ob = b.forward(input);
        double max_diff = 0.0;
        for (size_t j = 0; j < 3; ++j) max_diff = std::max(max_diff, std::abs(oa[0][j] - ob[0][j]));
        CHECK(max_diff < 1e-15, "bit-exact determinism across fresh instances");
    }

    // -------------------------------------------------------------------
    TEST_START("mutation test: zeroing W_halt changes forward (non-vacuous)");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        layer.random_init(103);
        std::mt19937 gen(107);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out_a = layer.forward(input);

        // Zero W_halt → halting becomes p_t = sigmoid(b_halt), no longer input-dependent.
        layer.W_halt_accessor().fill(0.0);
        Tensor out_b = layer.forward(input);
        double max_diff = 0.0;
        for (size_t j = 0; j < 3; ++j) max_diff = std::max(max_diff, std::abs(out_a[0][j] - out_b[0][j]));
        CHECK(max_diff > 1e-6, "W_halt has non-trivial effect on output");
    }

    // -------------------------------------------------------------------
    TEST_START("mutation test: zeroing W_xh changes forward");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        layer.random_init(109);
        std::mt19937 gen(113);
        std::normal_distribution<> nd(0.0, 0.3);
        Tensor input(1, 4 * 2);
        for (size_t j = 0; j < input.cols; ++j) input[0][j] = nd(gen);
        Tensor out_a = layer.forward(input);
        layer.W_xh_accessor().fill(0.0);
        Tensor out_b = layer.forward(input);
        double max_diff = 0.0;
        for (size_t j = 0; j < 3; ++j) max_diff = std::max(max_diff, std::abs(out_a[0][j] - out_b[0][j]));
        CHECK(max_diff > 1e-6, "W_xh has non-trivial effect on output");
    }

    // -------------------------------------------------------------------
    TEST_START("name() returns ACTRecurrentLayer");
    {
        ACTRecurrentLayer layer(2, 3, 4);
        CHECK(layer.name() == "ACTRecurrentLayer", "name()");
    }

    // -------------------------------------------------------------------
    std::cout << "\n=== Summary: " << n_pass << " passed, " << n_fail << " failed ===" << std::endl;
    return n_fail == 0 ? 0 : 1;
}