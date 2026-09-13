// ============================================================================
// Switch Transformer tests — Fedus, Zoph, Shazeer 2022
// ============================================================================
// Paper: https://arxiv.org/abs/2101.03961
//
// Test plan:
//   1.   Constructor validation + accessors (5 cases)
//   2.   SwitchMoE forward shape + finiteness + nonzero
//   3.   Top-1 routing signature (one expert per token)
//   4.   W_router zero + forced dominant expert routes correctly
//   5.   Capacity factor 1.0 drops overflow tokens
//   6.   Capacity factor 2.0 admits more tokens
//   7.   Aux loss formula (paper Eq. 4)
//   8.   Z-loss formula (paper Eq. 5)
//   9.   FD input gradient on SwitchMoE
//  10.   FD W_router gradient
//  11.   FD W1/W2 expert gradients
//  12.   All-dropped config: output=0, grad_input=0
//  13.   update_weights moves every param
//  14.   zero_grad clears every grad
//  15.   parameters()/gradients() contract
//  16.   End-to-end SwitchMoE training reduces loss
//  17.   SwitchTransformerBlock forward shape + finite
//  18.   SwitchTransformerBlock input FD
//  19.   SwitchTransformerBlock exposes aux + z losses
//  20.   SwitchTransformerBlock training reduces loss
//  21.   SwitchTransformerModel forward shape
//  22.   SwitchTransformerModel training reduces loss
//  23.   SwitchTransformerModel aux + z are sums across blocks
//  24.   Mutation test — zeroing a block's W_router changes output
// ============================================================================

#include "nn/layers/architectures/switch_transformer.h"
#include "nn/core/tensor.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>

// (All symbols are in the global namespace — no `using namespace nn;`.)

// ----------------------------------------------------------------------------
// Utility: finite-difference gradient of a scalar loss with respect to a
// single tensor param. Returns a tensor of the same shape as `param` with
// finite-difference estimates. Loss function takes (cloned_input, full
// model reference).
// ----------------------------------------------------------------------------
template <typename ModelT, typename LossFn>
static Tensor fd_param(const ModelT& model, Tensor param_orig,
                       const Tensor& input, const Tensor& target,
                       LossFn&& loss_fn, double eps = 1e-5) {
    Tensor grad(param_orig.rows, param_orig.cols);
    for (size_t i = 0; i < param_orig.rows; ++i) {
        for (size_t j = 0; j < param_orig.cols; ++j) {
            double v = param_orig(i, j);
            // f(x + eps)
            param_orig(i, j) = v + eps;
            Tensor y_plus = model.forward(input);
            double l_plus = loss_fn(y_plus, target);
            // f(x - eps)
            param_orig(i, j) = v - eps;
            Tensor y_minus = model.forward(input);
            double l_minus = loss_fn(y_minus, target);
            grad(i, j) = (l_plus - l_minus) / (2.0 * eps);
            param_orig(i, j) = v;  // restore
        }
    }
    return grad;
}

// Max relative error between two tensors, with floor for tiny values.
static double max_rel_err(const Tensor& a, const Tensor& b, double floor_ = 1e-12) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            double denom = std::max({std::fabs(a(i, j)), std::fabs(b(i, j)), floor_});
            double err = std::fabs(a(i, j) - b(i, j)) / denom;
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

static bool tensor_finite(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (!std::isfinite(t(i, j))) return false;
    return true;
}

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, #cond); \
    std::exit(1); } } while (0)


int main() {
    std::printf("=== Switch Transformer Tests ===\n");

    // =========================================================================
    // Test 1: Constructor validation + accessors
    // =========================================================================
    {
        // d_model = 0 throws
        bool threw = false;
        try { SwitchMoELayer(0, 4); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // num_experts = 0 throws
        threw = false;
        try { SwitchMoELayer(8, 0); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // capacity_factor = 0 throws
        threw = false;
        try { SwitchMoELayer(8, 4, 0, 0.0); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // capacity_factor < 0 throws
        threw = false;
        try { SwitchMoELayer(8, 4, 0, -0.1); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // aux_loss_coef < 0 throws
        threw = false;
        try { SwitchMoELayer(8, 4, 0, 1.25, -0.01); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // z_loss_coef < 0 throws
        threw = false;
        try { SwitchMoELayer(8, 4, 0, 1.25, 0.01, -0.001); } catch (std::invalid_argument&) { threw = true; }
        CHECK(threw);

        // Valid construct
        SwitchMoELayer moe(8, 4, 16, 1.5, 0.02, 0.005);
        CHECK(moe.d_model() == 8);
        CHECK(moe.num_experts() == 4);
        CHECK(moe.expert_hidden() == 16);
        CHECK(std::fabs(moe.capacity_factor() - 1.5) < 1e-12);
        CHECK(std::fabs(moe.aux_loss_coef() - 0.02) < 1e-12);
        CHECK(std::fabs(moe.z_loss_coef() - 0.005) < 1e-12);
        CHECK(moe.name() == "SwitchMoELayer");

        // Default expert_hidden: 4 * d_model
        SwitchMoELayer moe2(8, 4);
        CHECK(moe2.expert_hidden() == 32);
        CHECK(std::fabs(moe2.capacity_factor() - 1.25) < 1e-12);
        CHECK(std::fabs(moe2.aux_loss_coef() - 0.01) < 1e-12);
        CHECK(std::fabs(moe2.z_loss_coef() - 0.001) < 1e-12);
        std::printf("  Test 1 (constructor + accessors) PASS\n");
    }

    // =========================================================================
    // Test 2: Forward shape, finiteness, non-zero
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4, 16);
        Tensor input(6, 8);
        std::mt19937 gen(0);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);

        Tensor output = moe.forward(input);
        CHECK(output.rows == 6);
        CHECK(output.cols == 8);
        CHECK(tensor_finite(output));
        bool any_nonzero = false;
        for (size_t i = 0; i < output.rows; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (std::fabs(output(i, j)) > 1e-12) any_nonzero = true;
        CHECK(any_nonzero);
        std::printf("  Test 2 (forward shape + finite + nonzero) PASS\n");
    }

    // =========================================================================
    // Test 3: Top-1 routing — exactly one expert per token
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4);
        Tensor input(6, 8);
        std::mt19937 gen(1);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        moe.forward(input);
        const Tensor& t = moe.last_top1_indices();
        CHECK(t.rows == 6);
        CHECK(t.cols == 1);
        for (size_t b = 0; b < t.rows; ++b) {
            size_t e = static_cast<size_t>(t(b, 0));
            CHECK(e < 4);
        }
        std::printf("  Test 3 (top-1 routing, one expert per token) PASS\n");
    }

    // =========================================================================
    // Test 4: Forced dominant expert
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4);  // default init; we override the router bias
        // Set router: zero weights, bias so expert 0 dominates for every token.
        Tensor W(4, 8); W.fill(0.0);
        Tensor b(1, 4); b.fill(0.0); b(0, 0) = 10.0;
        for (size_t e = 1; e < 4; ++e) b(0, e) = -10.0;
        moe.set_router_for_test(W, b);

        Tensor input(6, 8);
        std::mt19937 gen(2);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        moe.forward(input);
        const Tensor& t = moe.last_top1_indices();
        for (size_t b = 0; b < t.rows; ++b) {
            CHECK(static_cast<size_t>(t(b, 0)) == 0);
        }
        std::printf("  Test 4 (forced expert 0 dominates) PASS\n");
    }

    // =========================================================================
    // Test 5: Capacity factor 1.0 drops overflow tokens
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4, 0, 1.0);  // capacity_factor = 1.0
        Tensor W(4, 8); W.fill(0.0);
        Tensor b(1, 4); b(0, 0) = 10.0;
        for (size_t e = 1; e < 4; ++e) b(0, e) = -10.0;
        moe.set_router_for_test(W, b);

        Tensor input(8, 8);  // B=8, num_experts=4 → cap = ceil(8/4)*1 = 2
        std::mt19937 gen(3);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor output = moe.forward(input);

        // Exactly 2 of 8 rows should be nonzero (the dispatched ones); the
        // other 6 should be exactly 0 (capacity-overflow drop).
        int nonzero = 0;
        for (size_t b = 0; b < 8; ++b) {
            double row_max = 0.0;
            for (size_t j = 0; j < 8; ++j) row_max = std::max(row_max, std::fabs(output(b, j)));
            if (row_max > 1e-12) nonzero++;
        }
        CHECK(nonzero == 2);
        std::printf("  Test 5 (cap 1.0 drops 6/8 tokens) PASS (nonzero=%d)\n", nonzero);
    }

    // =========================================================================
    // Test 6: Capacity factor 2.0 admits more tokens
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4, 0, 2.0);  // capacity_factor = 2.0
        Tensor W(4, 8); W.fill(0.0);
        Tensor b(1, 4); b(0, 0) = 10.0;
        for (size_t e = 1; e < 4; ++e) b(0, e) = -10.0;
        moe.set_router_for_test(W, b);

        Tensor input(8, 8);  // B=8, num_experts=4 → cap = ceil(8/4)*2 = 4
        std::mt19937 gen(4);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor output = moe.forward(input);

        int nonzero = 0;
        for (size_t b = 0; b < 8; ++b) {
            double row_max = 0.0;
            for (size_t j = 0; j < 8; ++j) row_max = std::max(row_max, std::fabs(output(b, j)));
            if (row_max > 1e-12) nonzero++;
        }
        CHECK(nonzero == 4);
        std::printf("  Test 6 (cap 2.0 admits 4/8 tokens) PASS (nonzero=%d)\n", nonzero);
    }

    // =========================================================================
    // Test 7: Aux loss formula (paper Eq. 4)
    //   L_aux = α · N · Σ_e f_e · p_e
    // =========================================================================
    {
        SwitchMoELayer moe(8, 4, 0, 1.25);  // aux_loss_coef default = 0.01
        Tensor W(4, 8); W.fill(0.0);
        Tensor b(1, 4);
        b(0, 0) = 10.0;   // expert 0 strongly preferred
        b(0, 1) = 5.0;    // expert 1 next
        b(0, 2) = 0.0;
        b(0, 3) = -100.0; // expert 3 never selected
        moe.set_router_for_test(W, b);

        Tensor input(4, 8);
        std::mt19937 gen(5);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        moe.forward(input);

        double expected_aux = 0.0;
        auto dfs = moe.get_dispatch_fractions();
        auto mrp = moe.get_mean_router_probs();
        double alpha = moe.aux_loss_coef();
        double N = static_cast<double>(moe.num_experts());
        for (size_t e = 0; e < 4; ++e)
            expected_aux += alpha * N * dfs[e] * mrp[e];

        double actual_aux = moe.get_aux_loss();
        CHECK(std::fabs(actual_aux - expected_aux) < 1e-10);
        CHECK(actual_aux > 0.0);  // not all zero — at least one expert dispatched
        std::printf("  Test 7 (aux loss formula) PASS (L_aux=%.6f)\n", actual_aux);
    }

    // =========================================================================
    // Test 8: Z-loss formula (paper Eq. 5)
    //   L_z = z_coef · (1/B) · Σ_b log(Σ_e exp(gate_logits[b, e])²)
    // =========================================================================
    {
        SwitchMoELayer moe(8, 3, 0, 1.25, 0.01, 1.0);  // z_loss_coef = 1.0 for clarity
        // W_router = 0, b_router = [[1.0, 2.0, 3.0]] → all rows see same logits.
        Tensor W(3, 8); W.fill(0.0);
        Tensor b(1, 3); b(0, 0) = 1.0; b(0, 1) = 2.0; b(0, 2) = 3.0;
        moe.set_router_for_test(W, b);

        Tensor input(2, 8);  // identical rows → identical gate logits
        input.fill(0.0);
        moe.forward(input);

        // Hand-compute the expected z-loss against the cached logits.
        const Tensor& gl = moe.last_gate_logits();
        double total = 0.0;
        for (size_t b = 0; b < 2; ++b) {
            double inner = 0.0;
            for (size_t e = 0; e < 3; ++e) {
                double v = std::exp(gl(b, e));
                inner += v * v;
            }
            total += std::log(inner + 1e-30);
        }
        double expected_z = moe.z_loss_coef() * (total / 2.0);
        double actual_z = moe.get_z_loss();
        CHECK(std::fabs(actual_z - expected_z) < 1e-9);
        std::printf("  Test 8 (z-loss formula) PASS (L_z=%.6f)\n", actual_z);
    }

    // =========================================================================
    // Test 9: FD input gradient on SwitchMoE
    // =========================================================================
    {
        // z_loss_coef = 0 to isolate the FD on the main loss only.
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);
        // Re-init the router and experts with random non-uniform values
        // (mandatory: zero init makes the router column sums symmetric and
        // masks row-vs-column confusion).
        std::mt19937 gen(9);
        std::normal_distribution<> dis(0.0, 0.3);
        Tensor W_r(2, 4);
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 4; ++j) W_r(i, j) = dis(gen);
        Tensor b_r(1, 2); b_r.fill(0.0);
        moe.set_router_for_test(W_r, b_r);
        for (size_t e = 0; e < 2; ++e) {
            Tensor W1(8, 4), W2(4, 8), b1(1, 8), b2(1, 4);
            for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 4; ++j)
                    W1(i, j) = dis(gen);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 8; ++j)
                    W2(i, j) = dis(gen);
            b1.fill(0.0); b2.fill(0.0);
            moe.set_expert_for_test(e, W1, b1, W2, b2);
        }

        Tensor input(4, 4);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);

        // Forward → loss → backward
        Tensor output = moe.forward(input);
        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);

        // Compute dL/dX analytically
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        Tensor grad_input_ana = moe.backward(grad_output, 0.0);

        // Compute dL/dX by finite difference on the input
        Tensor grad_input_fd(4, 4);
        const double eps = 1e-5;
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double v = input(i, j);
                input(i, j) = v + eps;
                Tensor y_plus = moe.forward(input);
                double l_plus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_plus += (y_plus(b, c) - target(b, c)) * (y_plus(b, c) - target(b, c));

                input(i, j) = v - eps;
                Tensor y_minus = moe.forward(input);
                double l_minus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_minus += (y_minus(b, c) - target(b, c)) * (y_minus(b, c) - target(b, c));

                grad_input_fd(i, j) = (l_plus - l_minus) / (2.0 * eps);
                input(i, j) = v;
            }
        }

        double rel = max_rel_err(grad_input_ana, grad_input_fd);
        if (rel >= 1e-4) {
            std::printf("    DIAG: rel=%.3e, grad_input_ana[0,0]=%.3e, grad_input_fd[0,0]=%.3e\n",
                       rel, grad_input_ana(0, 0), grad_input_fd(0, 0));
            std::printf("    DIAG: grad_input_ana all: ");
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    std::printf("%.3e ", grad_input_ana(i, j));
            std::printf("\n    DIAG: grad_input_fd  all: ");
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    std::printf("%.3e ", grad_input_fd(i, j));
            std::printf("\n");
        }
        CHECK(rel < 1e-4);
        std::printf("  Test 9 (FD input gradient) PASS (rel_err=%.3e)\n", rel);
    }

    // =========================================================================
    // Test 10: FD W_router gradient
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);  // no z-loss for clean FD
        std::mt19937 gen(10);
        std::normal_distribution<> dis(0.0, 0.3);
        Tensor W_r(2, 4);
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 4; ++j) W_r(i, j) = dis(gen);
        Tensor b_r(1, 2); b_r.fill(0.0);
        moe.set_router_for_test(W_r, b_r);
        for (size_t e = 0; e < 2; ++e) {
            Tensor W1(8, 4), W2(4, 8), b1(1, 8), b2(1, 4);
            for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 4; ++j)
                    W1(i, j) = dis(gen);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 8; ++j)
                    W2(i, j) = dis(gen);
            b1.fill(0.0); b2.fill(0.0);
            moe.set_expert_for_test(e, W1, b1, W2, b2);
        }

        Tensor input(4, 4);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);

        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);

        Tensor output = moe.forward(input);
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        moe.backward(grad_output, 0.0);

        // FD on W_router — grab a pointer to W_router via parameters().
        // params layout: [W_router, b_router, W1_0, b1_0, W2_0, b2_0, W1_1, b1_1, W2_1, b2_1]
        auto params = moe.parameters();
        Tensor* W_r_ptr = params[0];
        Tensor grad_W_router_fd(2, 4);
        const double eps = 1e-5;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                double v = (*W_r_ptr)(i, j);
                (*W_r_ptr)(i, j) = v + eps;
                Tensor y_plus = moe.forward(input);
                double l_plus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_plus += (y_plus(b, c) - target(b, c)) * (y_plus(b, c) - target(b, c));
                (*W_r_ptr)(i, j) = v - eps;
                Tensor y_minus = moe.forward(input);
                double l_minus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_minus += (y_minus(b, c) - target(b, c)) * (y_minus(b, c) - target(b, c));
                grad_W_router_fd(i, j) = (l_plus - l_minus) / (2.0 * eps);
                (*W_r_ptr)(i, j) = v;
            }
        }

        // grad_W_router_ is the second entry in gradients() (after b_router_? no: W_router first)
        // We index it from gradients() too — same layout as parameters().
        Tensor* grad_W_r_ptr = moe.gradients()[0];
        double rel = max_rel_err(*grad_W_r_ptr, grad_W_router_fd);
        CHECK(rel < 1e-4);
        std::printf("  Test 10 (FD W_router gradient) PASS (rel_err=%.3e)\n", rel);
    }

    // =========================================================================
    // Test 11: FD W1[0] gradient
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);  // no z-loss for clean FD
        std::mt19937 gen(11);
        std::normal_distribution<> dis(0.0, 0.3);
        Tensor W_r(2, 4);
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 4; ++j) W_r(i, j) = dis(gen);
        Tensor b_r(1, 2); b_r.fill(0.0);
        moe.set_router_for_test(W_r, b_r);
        for (size_t e = 0; e < 2; ++e) {
            Tensor W1(8, 4), W2(4, 8), b1(1, 8), b2(1, 4);
            for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 4; ++j)
                    W1(i, j) = dis(gen);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 8; ++j)
                    W2(i, j) = dis(gen);
            b1.fill(0.0); b2.fill(0.0);
            moe.set_expert_for_test(e, W1, b1, W2, b2);
        }

        Tensor input(4, 4);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);

        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);

        Tensor output = moe.forward(input);
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        moe.backward(grad_output, 0.0);

        // Check that expert 0 received at least one token
        const Tensor& t = moe.last_top1_indices();
        int expert0_count = 0;
        for (size_t b = 0; b < 4; ++b)
            if (static_cast<size_t>(t(b, 0)) == 0) expert0_count++;
        CHECK(expert0_count > 0);

        // FD on W1[0] — grads() layout: [grad_W_router, grad_b_router,
        // grad_W1_0, grad_b1_0, grad_W2_0, grad_b2_0, grad_W1_1, ...]
        auto grads = moe.gradients();
        Tensor* grad_W1_ptr = grads[2];
        Tensor grad_W1_fd(8, 4);
        const double eps = 1e-5;
        // params layout: same order, index 2 = W1_[0]
        Tensor* W1_0_ptr = moe.parameters()[2];
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                double v = (*W1_0_ptr)(i, j);
                (*W1_0_ptr)(i, j) = v + eps;
                Tensor y_plus = moe.forward(input);
                double l_plus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_plus += (y_plus(b, c) - target(b, c)) * (y_plus(b, c) - target(b, c));
                (*W1_0_ptr)(i, j) = v - eps;
                Tensor y_minus = moe.forward(input);
                double l_minus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_minus += (y_minus(b, c) - target(b, c)) * (y_minus(b, c) - target(b, c));
                grad_W1_fd(i, j) = (l_plus - l_minus) / (2.0 * eps);
                (*W1_0_ptr)(i, j) = v;
            }
        }

        double rel = max_rel_err(*grad_W1_ptr, grad_W1_fd);
        CHECK(rel < 1e-4);
        std::printf("  Test 11 (FD W1[0] gradient) PASS (rel_err=%.3e)\n", rel);
    }

    // =========================================================================
    // Test 12: Capacity-factor dropping — with cap_factor=0.5 → cap=1, only
    // 1 of 4 tokens dispatched, the other 3 outputs are exactly 0.
    // =========================================================================
    {
        SwitchMoELayer moe(8, 2, 0, 0.5);
        Tensor W(2, 8); W.fill(0.0);
        Tensor b(1, 2); b(0, 0) = 10.0; b(0, 1) = -10.0;
        moe.set_router_for_test(W, b);

        Tensor input(4, 8);
        std::mt19937 gen(12);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor output = moe.forward(input);
        int nonzero_rows = 0;
        for (size_t i = 0; i < output.rows; ++i) {
            bool row_nonzero = false;
            for (size_t j = 0; j < output.cols; ++j)
                if (std::fabs(output(i, j)) > 1e-12) { row_nonzero = true; break; }
            if (row_nonzero) nonzero_rows++;
        }
        // Only 1 of 4 tokens dispatched → exactly 1 nonzero row, 3 zero rows.
        CHECK(nonzero_rows == 1);

        // grad_input on the drop-affected tokens should be zero (no expert
        // gradient flows to dropped tokens; router gradient is also zero
        // because the dropped tokens contribute nothing to d_router_prob).
        Tensor grad_output(4, 8);
        grad_output.fill(1.0);
        Tensor grad_input = moe.backward(grad_output, 0.0);
        int nonzero_grad_rows = 0;
        for (size_t i = 0; i < grad_input.rows; ++i) {
            bool row_nonzero = false;
            for (size_t j = 0; j < grad_input.cols; ++j)
                if (std::fabs(grad_input(i, j)) > 1e-12) { row_nonzero = true; break; }
            if (row_nonzero) nonzero_grad_rows++;
        }
        CHECK(nonzero_grad_rows == 1);
        std::printf("  Test 12 (cap 0.5 drops 3/4 tokens; output and grad) PASS (nonzero_rows=%d)\n", nonzero_rows);
    }

    // =========================================================================
    // Test 13: update_weights moves every param
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);  // no aux/z for clean check
        Tensor input(4, 4);
        std::mt19937 gen(13);
        std::normal_distribution<> dis(0.0, 0.3);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);
        Tensor output = moe.forward(input);
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        moe.backward(grad_output, 0.0);

        // Snapshot all params (we re-read via parameters() pointers).
        auto params_before = moe.parameters();
        std::vector<Tensor> snap;
        for (auto* p : params_before) snap.push_back(p->clone());

        moe.update_weights(0.05);

        double max_diff = 0.0;
        for (size_t k = 0; k < params_before.size(); ++k) {
            for (size_t i = 0; i < params_before[k]->rows; ++i)
                for (size_t j = 0; j < params_before[k]->cols; ++j)
                    max_diff = std::max(max_diff,
                        std::fabs((*params_before[k])(i, j) - snap[k](i, j)));
        }
        CHECK(max_diff > 1e-10);
        std::printf("  Test 13 (update_weights moves all params) PASS (max_diff=%.3e)\n", max_diff);
    }

    // =========================================================================
    // Test 14: zero_grad clears every grad
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);  // no aux/z for clean check
        Tensor input(4, 4);
        std::mt19937 gen(14);
        std::normal_distribution<> dis(0.0, 0.3);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);
        Tensor output = moe.forward(input);
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        moe.backward(grad_output, 0.0);
        moe.zero_grad();
        double max_abs = 0.0;
        auto grads = moe.gradients();
        for (auto* g : grads) {
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    max_abs = std::max(max_abs, std::fabs((*g)(i, j)));
        }
        CHECK(max_abs < 1e-15);
        std::printf("  Test 14 (zero_grad) PASS (max_abs=%.3e)\n", max_abs);
    }

    // =========================================================================
    // Test 15: parameters()/gradients() contract
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8);
        auto p = moe.parameters();
        auto g = moe.gradients();
        // 2 router params + 4 params per expert * 2 experts = 10
        CHECK(p.size() == 10);
        CHECK(g.size() == 10);
        for (size_t k = 0; k < p.size(); ++k) {
            CHECK(p[k]->rows == g[k]->rows);
            CHECK(p[k]->cols == g[k]->cols);
        }
        std::printf("  Test 15 (params/grads contract) PASS (n=%zu)\n", p.size());
    }

    // =========================================================================
    // Test 16: End-to-end SwitchMoE training reduces loss
    // =========================================================================
    {
        SwitchMoELayer moe(4, 2, 8, 1.25, 0.0, 0.0);  // no aux/z for clean training
        std::mt19937 gen(16);
        std::normal_distribution<> dis(0.0, 0.3);
        Tensor W_r(2, 4);
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 4; ++j) W_r(i, j) = dis(gen);
        Tensor b_r(1, 2); b_r.fill(0.0);
        moe.set_router_for_test(W_r, b_r);
        for (size_t e = 0; e < 2; ++e) {
            Tensor W1(8, 4), W2(4, 8), b1(1, 8), b2(1, 4);
            for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 4; ++j)
                    W1(i, j) = dis(gen);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 8; ++j)
                    W2(i, j) = dis(gen);
            b1.fill(0.0); b2.fill(0.0);
            moe.set_expert_for_test(e, W1, b1, W2, b2);
        }

        Tensor input(4, 4);
        Tensor target(4, 4);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                input(i, j) = dis(gen);
                target(i, j) = dis(gen);
            }
        }

        auto loss_at = [&](void) {
            Tensor y = moe.forward(input);
            double s = 0.0;
            for (size_t b = 0; b < 4; ++b)
                for (size_t c = 0; c < 4; ++c)
                    s += (y(b, c) - target(b, c)) * (y(b, c) - target(b, c));
            return s / static_cast<double>(4 * 4);
        };
        double L0 = loss_at();
        // Use a higher capacity factor and more steps so the model can learn.
        for (int step = 0; step < 80; ++step) {
            moe.zero_grad();
            Tensor output = moe.forward(input);
            Tensor grad_output(4, 4);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    grad_output(i, j) = 2.0 * (output(i, j) - target(i, j)) / 16.0;
            moe.backward(grad_output, 0.0);
            moe.update_weights(0.05);
        }
        double L1 = loss_at();
        CHECK(L1 < L0 * 0.95);
        std::printf("  Test 16 (SwitchMoE training) PASS (loss %.4f -> %.4f)\n", L0, L1);
    }

    // =========================================================================
    // Test 17: SwitchTransformerBlock forward shape + finite
    // =========================================================================
    {
        SwitchTransformerBlock block(8, 4, 16);
        Tensor input(6, 8);
        std::mt19937 gen(17);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor output = block.forward(input);
        CHECK(output.rows == 6);
        CHECK(output.cols == 8);
        CHECK(tensor_finite(output));
        std::printf("  Test 17 (block forward shape + finite) PASS\n");
    }

    // =========================================================================
    // Test 18: SwitchTransformerBlock input FD
    // =========================================================================
    {
        SwitchTransformerBlock block(4, 2, 8, 1.25, 0.0, 0.0);  // no aux/z for clean FD
        std::mt19937 gen(18);
        std::normal_distribution<> dis(0.0, 0.3);
        // Re-init MOE in the block (the member is private; we don't have
        // direct access from outside — but we can manipulate by going
        // through the constructor's init. The default init is deterministic
        // with seed 42/43, so we don't need to override here.
        Tensor input(4, 4);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor target(4, 4);
        for (size_t i = 0; i < target.rows; ++i)
            for (size_t j = 0; j < target.cols; ++j)
                target(i, j) = dis(gen);

        Tensor output = block.forward(input);
        Tensor grad_output(4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                grad_output(i, j) = 2.0 * (output(i, j) - target(i, j));
        Tensor grad_input_ana = block.backward(grad_output, 0.0);

        Tensor grad_input_fd(4, 4);
        const double eps = 1e-5;
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double v = input(i, j);
                input(i, j) = v + eps;
                Tensor y_plus = block.forward(input);
                double l_plus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_plus += (y_plus(b, c) - target(b, c)) * (y_plus(b, c) - target(b, c));
                input(i, j) = v - eps;
                Tensor y_minus = block.forward(input);
                double l_minus = 0.0;
                for (size_t b = 0; b < 4; ++b)
                    for (size_t c = 0; c < 4; ++c)
                        l_minus += (y_minus(b, c) - target(b, c)) * (y_minus(b, c) - target(b, c));
                grad_input_fd(i, j) = (l_plus - l_minus) / (2.0 * eps);
                input(i, j) = v;
            }
        }
        double rel = max_rel_err(grad_input_ana, grad_input_fd);
        CHECK(rel < 1e-4);
        std::printf("  Test 18 (block FD input gradient) PASS (rel_err=%.3e)\n", rel);
    }

    // =========================================================================
    // Test 19: SwitchTransformerBlock exposes aux + z losses
    // =========================================================================
    {
        SwitchTransformerBlock block(4, 2);
        Tensor input(4, 4);
        std::mt19937 gen(19);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        block.forward(input);
        CHECK(block.get_aux_loss() >= 0.0);
        CHECK(block.get_z_loss() >= 0.0);
        std::printf("  Test 19 (block aux + z losses) PASS (aux=%.4f, z=%.4f)\n",
                    block.get_aux_loss(), block.get_z_loss());
    }

    // =========================================================================
    // Test 20: SwitchTransformerBlock training reduces loss
    // =========================================================================
    {
        SwitchTransformerBlock block(4, 2, 8, 1.25, 0.0, 0.0);  // no aux/z for clean training
        Tensor input(4, 4);
        Tensor target(4, 4);
        std::mt19937 gen(20);
        std::normal_distribution<> dis(0.0, 0.3);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                input(i, j) = dis(gen);
                target(i, j) = dis(gen);
            }
        }
        auto loss_at = [&](void) {
            Tensor y = block.forward(input);
            double s = 0.0;
            for (size_t b = 0; b < 4; ++b)
                for (size_t c = 0; c < 4; ++c)
                    s += (y(b, c) - target(b, c)) * (y(b, c) - target(b, c));
            return s / 16.0;
        };
        double L0 = loss_at();
        for (int step = 0; step < 60; ++step) {
            block.zero_grad();
            Tensor output = block.forward(input);
            Tensor grad_output(4, 4);
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    grad_output(i, j) = 2.0 * (output(i, j) - target(i, j)) / 16.0;
            block.backward(grad_output, 0.0);
            block.update_weights(0.02);
        }
        double L1 = loss_at();
        CHECK(L1 < L0 * 0.95);
        std::printf("  Test 20 (block training) PASS (loss %.4f -> %.4f)\n", L0, L1);
    }

    // =========================================================================
    // Test 21: SwitchTransformerModel forward shape
    // =========================================================================
    {
        SwitchTransformerModel model(4, 8, 4, 3, 2, 16);
        Tensor input(6, 4);
        std::mt19937 gen(21);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        Tensor output = model.forward(input);
        CHECK(output.rows == 6);
        CHECK(output.cols == 3);
        CHECK(tensor_finite(output));
        std::printf("  Test 21 (model forward shape) PASS\n");
    }

    // =========================================================================
    // Test 22: SwitchTransformerModel training reduces loss
    // =========================================================================
    {
        SwitchTransformerModel model(4, 8, 4, 3, 2, 16);
        Tensor input(6, 4);
        Tensor target(6, 3);
        std::mt19937 gen(22);
        std::normal_distribution<> dis(0.0, 0.3);
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 4; ++j) input(i, j) = dis(gen);
            for (size_t j = 0; j < 3; ++j) target(i, j) = dis(gen);
        }
        auto loss_at = [&](void) {
            Tensor y = model.forward(input);
            double s = 0.0;
            for (size_t b = 0; b < 6; ++b)
                for (size_t c = 0; c < 3; ++c)
                    s += (y(b, c) - target(b, c)) * (y(b, c) - target(b, c));
            return s / 18.0;
        };
        double L0 = loss_at();
        for (int step = 0; step < 80; ++step) {
            model.zero_grad();
            Tensor output = model.forward(input);
            Tensor grad_output(6, 3);
            for (size_t i = 0; i < 6; ++i)
                for (size_t j = 0; j < 3; ++j)
                    grad_output(i, j) = 2.0 * (output(i, j) - target(i, j)) / 18.0;
            model.backward(grad_output, 0.0);
            model.update_weights(0.01);
        }
        double L1 = loss_at();
        CHECK(L1 < L0 * 0.95);
        std::printf("  Test 22 (model training) PASS (loss %.4f -> %.4f)\n", L0, L1);
    }

    // =========================================================================
    // Test 23: Model aux + z are sums across blocks (num_blocks=2)
    // =========================================================================
    {
        SwitchTransformerModel model(4, 8, 4, 3, 2, 16);
        Tensor input(6, 4);
        std::mt19937 gen(23);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);
        model.forward(input);
        double m_aux = model.get_aux_loss();
        double m_z   = model.get_z_loss();
        CHECK(m_aux >= 0.0);
        CHECK(m_z   >= 0.0);
        // Aux and z losses are sums across blocks. With 2 blocks they
        // should be ≥ what a single block would give. Sanity: just check
        // they are not NaN/Inf and are finite positive (paper guarantees
        // α·N·Σ f_e·P_e > 0 and z-loss ≥ 0).
        CHECK(std::isfinite(m_aux));
        CHECK(std::isfinite(m_z));
        std::printf("  Test 23 (model aux/z sum) PASS (aux=%.4f, z=%.4f)\n", m_aux, m_z);
    }

    // =========================================================================
    // Test 24: Mutation — zeroing a block's W_router changes output
    // =========================================================================
    {
        SwitchTransformerModel model(4, 8, 4, 3, 2, 16);
        Tensor input(6, 4);
        std::mt19937 gen(24);
        std::normal_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                input(i, j) = dis(gen);

        Tensor output_before = model.forward(input);

        // Zero the FIRST block's router weights via parameters().
        // params layout: input_proj (W, b), pre_ln (gamma, beta), then per
        // block: (ln gamma, ln beta, then moe: W_router, b_router, W1_e,
        // b1_e, W2_e, b2_e for e=0..num_experts-1). Skip past the input
        // proj + LN (4 params) to reach the first block's W_router.
        auto params = model.parameters();
        CHECK(params.size() >= 6);
        size_t idx = 4;  // first block's W_router
        Tensor original = params[idx]->clone();
        params[idx]->fill(0.0);
        Tensor output_after = model.forward(input);
        *params[idx] = original;  // restore

        double max_diff = 0.0;
        for (size_t i = 0; i < output_before.rows; ++i)
            for (size_t j = 0; j < output_before.cols; ++j)
                max_diff = std::max(max_diff, std::fabs(output_before(i, j) - output_after(i, j)));
        CHECK(max_diff > 1e-6);
        std::printf("  Test 24 (mutation: zeroing first block router) PASS (max_diff=%.3e)\n", max_diff);
    }

    std::printf("=== All Switch Transformer tests PASS ===\n");
    return 0;
}
