// Hyena Hierarchy tests — Poli et al. 2023
// "Towards Larger Convolutional Language Models"
//   https://arxiv.org/abs/2302.10866
//
// Tests:
//   1. HyenaOperator forward shape: (B, L*D) -> (B, L*D)
//   2. HyenaOperator output is finite
//   3. HyenaOperator numerical gradient check on input
//   4. HyenaOperator numerical gradient check on in_proj weights
//   5. HyenaOperator numerical gradient check on out_proj weights
//   6. HyenaOperator numerical gradient check on short conv weights
//   7. HyenaOperator zero_grad clears all gradients
//   8. HyenaFilter (D_skip + per-channel filter) shape correctness
//   9. HyenaFilter D_skip gradient (analytic vs FD)
//  10. HyenaFilter delta gradient (analytic vs FD on a single delta)
//  11. HyenaBlock forward shape
//  12. HyenaBlock input gradient check
//  13. HyenaModel forward shape (B, L*D) -> (B, num_classes)
//  14. HyenaModel training reduces loss
//  15. HyenaBlock deeper stack (depth=2) forward shape
//  16. HyenaModel zero_grad clears all gradients
//  17. HyenaOperator non-zero output (random init produces non-degenerate output)
//  18. HyenaBlock nonzero parameter gradient on input
//  19. HyenaModel end-to-end: classifier W gradient check
//  20. HyenaOperator deterministic across two fresh instances

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/layers/architectures/hyena.h"

using namespace std;

static double max_rel_err(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double av = a.data[i], bv = b.data[i];
        double denom = max(fabs(av), fabs(bv));
        if (denom < 1e-8) denom = 1e-8;
        m = max(m, fabs(av - bv) / denom);
    }
    return m;
}

static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}

static void fill_random(Tensor& t, std::mt19937& gen, double scale = 0.3) {
    std::normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dis(gen);
}

static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// FD grad check on input via central differences
static Tensor finite_diff_grad_input(HyenaOperator& op, Tensor& input, const Tensor& target,
                                     double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = op.forward(input);
            double lp = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = op.forward(input);
            double lm = l2_loss_value(out_m, target);
            input(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// FD grad check on a parameter tensor (e.g. weights, biases). Caller mutates the
// param tensor's data and calls op.forward again.
template <typename OpT>
static Tensor finite_diff_grad_param(OpT& op, const Tensor& input, const Tensor& target,
                                     Tensor& param, double eps = 1e-5) {
    Tensor grad(param.rows, param.cols);
    for (size_t i = 0; i < param.rows; ++i) {
        for (size_t j = 0; j < param.cols; ++j) {
            double orig = param(i, j);
            param(i, j) = orig + eps;
            Tensor out_p = op.forward(input);
            double lp = l2_loss_value(out_p, target);
            param(i, j) = orig - eps;
            Tensor out_m = op.forward(input);
            double lm = l2_loss_value(out_m, target);
            param(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

int main() {
    cout << "=== Hyena Hierarchy Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    auto check = [&](bool ok, const string& msg) {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << msg << "\n"; }
        else    { cout << "[FAIL] " << msg << "\n"; }
    };

    // -------------------------------------------------------------------
    // Tests 1-2: HyenaOperator forward shape + finite output.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 6, B = 2;
        HyenaOperator op(D, L, /*order=*/2, /*filter_order=*/4);
        std::mt19937 gen(123);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor output = op.forward(input);
        check(output.rows == B && output.cols == L * D, "HyenaOperator forward shape (B,L*D)->(B,L*D)");

        bool finite = true;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!std::isfinite(output.data[i])) { finite = false; break; }
        }
        check(finite, "HyenaOperator output is finite");
    }

    // -------------------------------------------------------------------
    // Test 3: HyenaOperator input gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, B = 1;
        HyenaOperator op(D, L, 2, 4);
        std::mt19937 gen(7);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        Tensor out = op.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        op.zero_grad();
        Tensor grad_in_ana = op.backward(grad_out, 0.0);

        Tensor grad_in_fd = finite_diff_grad_input(op, input, target);
        double rel = max_rel_err(grad_in_ana, grad_in_fd);
        check(rel < 1e-4, "HyenaOperator input gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 4: HyenaOperator in_proj.weights gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, B = 1;
        HyenaOperator op(D, L, 2, 4);
        std::mt19937 gen(11);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        // Run once to populate caches and backward to populate gradients
        Tensor out = op.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        op.zero_grad();
        op.backward(grad_out, 0.0);

        // Analytical
        Tensor ana = op.in_proj.grad_weights.clone();
        // FD
        Tensor fd = finite_diff_grad_param(op, input, target, op.in_proj.weights);
        double rel = max_rel_err(ana, fd);
        check(rel < 1e-4, "HyenaOperator in_proj.W gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 5: HyenaOperator out_proj.weights gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, B = 1;
        HyenaOperator op(D, L, 2, 4);
        std::mt19937 gen(13);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        Tensor out = op.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        op.zero_grad();
        op.backward(grad_out, 0.0);

        Tensor ana = op.out_proj.grad_weights.clone();
        Tensor fd = finite_diff_grad_param(op, input, target, op.out_proj.weights);
        double rel = max_rel_err(ana, fd);
        check(rel < 1e-4, "HyenaOperator out_proj.W gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 6: HyenaOperator short conv weights gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, B = 1;
        HyenaOperator op(D, L, 2, 4);
        std::mt19937 gen(17);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        Tensor out = op.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        op.zero_grad();
        op.backward(grad_out, 0.0);

        Tensor ana = op.grad_short_W.clone();
        // FD: do it element by element
        Tensor fd(op.short_W.rows, op.short_W.cols);
        for (size_t i = 0; i < op.short_W.rows; ++i) {
            for (size_t j = 0; j < op.short_W.cols; ++j) {
                double orig = op.short_W(i, j);
                op.short_W(i, j) = orig + 1e-5;
                Tensor out_p = op.forward(input);
                double lp = l2_loss_value(out_p, target);
                op.short_W(i, j) = orig - 1e-5;
                Tensor out_m = op.forward(input);
                double lm = l2_loss_value(out_m, target);
                op.short_W(i, j) = orig;
                fd(i, j) = (lp - lm) / 2e-5;
            }
        }
        double rel = max_rel_err(ana, fd);
        check(rel < 1e-4, "HyenaOperator short conv W gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 7: HyenaOperator zero_grad clears all gradients.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, B = 1;
        HyenaOperator op(D, L, 2, 4);
        std::mt19937 gen(19);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        Tensor out = op.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        op.backward(grad_out, 0.0);
        op.zero_grad();

        bool all_zero = true;
        for (auto* g : op.gradients()) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (std::abs(g->data[i]) > 1e-20) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
        check(all_zero, "HyenaOperator zero_grad clears all gradients");
    }

    // -------------------------------------------------------------------
    // Test 8: HyenaFilter shape correctness.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 5, P = 4;
        HyenaFilter f(D, L, P);
        auto out = f.filter(L);
        check(out.first.rows == L && out.first.cols == D,
              "HyenaFilter filter(L) returns (L, D) tensor");
        check(out.second.rows == 1 && out.second.cols == D,
              "HyenaFilter filter(L) returns D_skip (1, D)");
    }

    // -------------------------------------------------------------------
    // Test 9: HyenaFilter D_skip gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 4, P = 3;
        HyenaFilter f(D, L, P);
        std::mt19937 gen(23);
        Tensor grad_h(L, D);
        fill_random(grad_h, gen, 0.2);
        Tensor grad_D(1, D);
        fill_random(grad_D, gen, 0.1);
        f.zero_grad();
        f.backward(grad_h, grad_D);
        // Analytical: grad_D_skip should equal grad_D (since D_skip is a direct add)
        Tensor ana = f.grad_D_skip.clone();
        double rel = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = std::abs(ana[0][d] - grad_D[0][d]);
            double denom = std::max(std::abs(ana[0][d]), std::abs(grad_D[0][d]));
            if (denom < 1e-8) denom = 1e-8;
            rel = std::max(rel, diff / denom);
        }
        check(rel < 1e-8, "HyenaFilter D_skip gradient matches input grad_D exactly");
    }

    // -------------------------------------------------------------------
    // Test 10: HyenaFilter delta gradient check (FD vs analytical on a single delta).
    // -------------------------------------------------------------------
    {
        const size_t D = 2, L = 4, P = 3;
        HyenaFilter f(D, L, P);
        std::mt19937 gen(29);
        Tensor grad_h(L, D);
        fill_random(grad_h, gen, 0.2);
        Tensor grad_D(1, D);
        fill_random(grad_D, gen, 0.1);

        // First cache the filter output
        f.filter(L);
        f.zero_grad();
        f.backward(grad_h, grad_D);
        double ana_delta_0 = f.grad_deltas[0][0];

        // FD: perturb delta[0][0] by eps and recompute grad_h via filter.
        // For FD on delta we need a downstream loss; we'll use sum(grad_h * filter(L)).
        double eps = 1e-5;
        double orig = f.deltas[0][0];
        f.deltas[0][0] = orig + eps;
        auto plus = f.filter(L);
        double lp = 0.0;
        for (size_t i = 0; i < grad_h.data.size(); ++i) lp += grad_h.data[i] * plus.first.data[i];
        f.deltas[0][0] = orig - eps;
        auto minus = f.filter(L);
        double lm = 0.0;
        for (size_t i = 0; i < grad_h.data.size(); ++i) lm += grad_h.data[i] * minus.first.data[i];
        f.deltas[0][0] = orig;
        double fd_delta_0 = (lp - lm) / (2.0 * eps);

        double denom = std::max(std::abs(ana_delta_0), std::abs(fd_delta_0));
        if (denom < 1e-8) denom = 1e-8;
        double rel = std::abs(ana_delta_0 - fd_delta_0) / denom;
        check(rel < 1e-3, "HyenaFilter delta gradient rel_err < 1e-3 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 11: HyenaBlock forward shape.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 5, B = 2;
        HyenaBlock block(D, L, 2, 4, 2);
        std::mt19937 gen(31);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor out = block.forward(input);
        check(out.rows == B && out.cols == L * D, "HyenaBlock forward shape (B, L*D) -> (B, L*D)");
    }

    // -------------------------------------------------------------------
    // Test 12: HyenaBlock input gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 4, B = 1;
        HyenaBlock block(D, L, 2, 4, 2);
        std::mt19937 gen(37);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);

        Tensor out = block.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor ana = block.backward(grad_out, 0.0);

        // FD
        Tensor fd(B, L * D);
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + 1e-5;
                Tensor op = block.forward(input);
                double lp = l2_loss_value(op, target);
                input(i, j) = orig - 1e-5;
                Tensor om = block.forward(input);
                double lm = l2_loss_value(om, target);
                input(i, j) = orig;
                fd(i, j) = (lp - lm) / 2e-5;
            }
        }
        double rel = max_rel_err(ana, fd);
        check(rel < 1e-4, "HyenaBlock input gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 13: HyenaModel forward shape.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 5, B = 2;
        HyenaModel model(D, L, /*depth=*/1, /*num_classes=*/3, 2, 4);
        std::mt19937 gen(41);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor out = model.forward(input);
        check(out.rows == B && out.cols == 3, "HyenaModel forward shape (B, L*D) -> (B, num_classes)");
    }

    // -------------------------------------------------------------------
    // Test 14: HyenaModel training reduces loss.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 4, B = 2;
        HyenaModel model(D, L, 1, 2, 2, 4);
        std::mt19937 gen(43);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, 2);
        for (size_t b = 0; b < B; ++b) {
            target[b][0] = 0.7; target[b][1] = 0.3;
        }
        double lr = 0.01;
        double initial = l2_loss_value(model.forward(input), target);
        for (int step = 0; step < 50; ++step) {
            Tensor out = model.forward(input);
            Tensor grad_out = l2_loss_grad(out, target);
            model.zero_grad();
            model.backward(grad_out, 0.0);
            model.update_weights(lr);
        }
        double final = l2_loss_value(model.forward(input), target);
        check(final < initial * 0.5, "HyenaModel training reduces loss > 50% (initial " +
              to_string(initial) + " -> final " + to_string(final) + ")");
    }

    // -------------------------------------------------------------------
    // Test 15: HyenaModel deeper stack (depth=2) forward shape.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 5, B = 2;
        HyenaModel model(D, L, /*depth=*/2, /*num_classes=*/3, 2, 4);
        std::mt19937 gen(47);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor out = model.forward(input);
        check(out.rows == B && out.cols == 3, "HyenaModel depth=2 forward shape");
    }

    // -------------------------------------------------------------------
    // Test 16: HyenaModel zero_grad clears all gradients.
    // -------------------------------------------------------------------
    {
        const size_t D = 4, L = 5, B = 2;
        HyenaModel model(D, L, 1, 2, 2, 4);
        std::mt19937 gen(53);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, 2);
        fill_random(target, gen, 0.1);
        Tensor out = model.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        model.backward(grad_out, 0.0);
        model.zero_grad();
        bool all_zero = true;
        for (auto* g : model.gradients()) {
            for (size_t i = 0; i < g->data.size(); ++i) {
                if (std::abs(g->data[i]) > 1e-20) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
        check(all_zero, "HyenaModel zero_grad clears all gradients");
    }

    // -------------------------------------------------------------------
    // Test 17: HyenaOperator non-zero output (random init produces non-degenerate output).
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 4, B = 1;
        HyenaOperator op(D, L, 2, 4);
        Tensor input(B, L * D);
        // Avoid constant zeros (which would give all-zero output trivially)
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.3 * std::sin(i * 0.7);
        Tensor out = op.forward(input);
        double abs_sum = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) abs_sum += std::abs(out.data[i]);
        check(abs_sum > 1e-6, "HyenaOperator non-zero output for non-zero input (abs sum = " +
              to_string(abs_sum) + ")");
    }

    // -------------------------------------------------------------------
    // Test 18: HyenaBlock gradient chain through hyena operator is wired.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 4, B = 1;
        HyenaBlock block(D, L, 2, 4, 2);
        std::mt19937 gen(59);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, L * D);
        fill_random(target, gen, 0.1);
        Tensor out = block.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        block.zero_grad();
        Tensor ana_input = block.backward(grad_out, 0.0);
        double in_grad_norm = 0.0;
        for (size_t i = 0; i < ana_input.data.size(); ++i)
            in_grad_norm += ana_input.data[i] * ana_input.data[i];
        check(in_grad_norm > 1e-6, "HyenaBlock input gradient norm is non-zero (norm2 = " +
              to_string(in_grad_norm) + ")");
    }

    // -------------------------------------------------------------------
    // Test 19: HyenaModel classifier W gradient check.
    // -------------------------------------------------------------------
    {
        const size_t D = 3, L = 4, B = 1;
        HyenaModel model(D, L, 1, 2, 2, 4);
        std::mt19937 gen(61);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, 2);
        fill_random(target, gen, 0.1);

        Tensor out = model.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        model.zero_grad();
        model.backward(grad_out, 0.0);
        Tensor ana = model.classifier.grad_weights.clone();

        // FD on classifier.weights
        Tensor fd = finite_diff_grad_param(model, input, target, model.classifier.weights);
        double rel = max_rel_err(ana, fd);
        check(rel < 1e-4, "HyenaModel classifier.W gradient rel_err < 1e-4 (got " + to_string(rel) + ")");
    }

    // -------------------------------------------------------------------
    // Test 20: HyenaOperator determinism (two fresh instances same output).
    // -------------------------------------------------------------------
    // Hyena initialises weights from a static std::mt19937 (hyena_gen), so two
    // fresh instances draw DIFFERENT parameters. The standard determinism test
    // pattern is to construct one instance, copy its parameters into the
    // second, then assert both forward outputs match bit-exactly.
    {
        const size_t D = 3, L = 4, B = 1;
        HyenaOperator op1(D, L, 2, 4);
        HyenaOperator op2(D, L, 2, 4);
        // Copy op1's weights into op2 (same-shape tensors).
        for (size_t i = 0; i < op1.in_proj.weights.data.size(); ++i)
            op2.in_proj.weights.data[i] = op1.in_proj.weights.data[i];
        for (size_t i = 0; i < op1.in_proj.bias.data.size(); ++i)
            op2.in_proj.bias.data[i] = op1.in_proj.bias.data[i];
        for (size_t i = 0; i < op1.out_proj.weights.data.size(); ++i)
            op2.out_proj.weights.data[i] = op1.out_proj.weights.data[i];
        for (size_t i = 0; i < op1.out_proj.bias.data.size(); ++i)
            op2.out_proj.bias.data[i] = op1.out_proj.bias.data[i];
        for (size_t i = 0; i < op1.short_W.data.size(); ++i)
            op2.short_W.data[i] = op1.short_W.data[i];
        for (size_t i = 0; i < op1.short_b.data.size(); ++i)
            op2.short_b.data[i] = op1.short_b.data[i];
        // Copy HyenaFilter parameters.
        auto& f1 = op1.hyena_filter;
        auto& f2 = op2.hyena_filter;
        for (size_t i = 0; i < f1.mlp_in_W.data.size(); ++i) f2.mlp_in_W.data[i] = f1.mlp_in_W.data[i];
        for (size_t i = 0; i < f1.mlp_in_b.data.size(); ++i) f2.mlp_in_b.data[i] = f1.mlp_in_b.data[i];
        for (size_t i = 0; i < f1.sin_freq.data.size(); ++i) f2.sin_freq.data[i] = f1.sin_freq.data[i];
        for (size_t i = 0; i < f1.mlp_out_W.data.size(); ++i) f2.mlp_out_W.data[i] = f1.mlp_out_W.data[i];
        for (size_t i = 0; i < f1.deltas.data.size(); ++i) f2.deltas.data[i] = f1.deltas.data[i];
        for (size_t i = 0; i < f1.D_skip.data.size(); ++i) f2.D_skip.data[i] = f1.D_skip.data[i];
        for (size_t k = 0; k < f1.mlp_W.size(); ++k) {
            for (size_t i = 0; i < f1.mlp_W[k].data.size(); ++i) f2.mlp_W[k].data[i] = f1.mlp_W[k].data[i];
            for (size_t i = 0; i < f1.mlp_b[k].data.size(); ++i) f2.mlp_b[k].data[i] = f1.mlp_b[k].data[i];
        }
        std::mt19937 gen(67);
        Tensor input(B, L * D);
        fill_random(input, gen, 0.3);
        Tensor out1 = op1.forward(input);
        Tensor out2 = op2.forward(input);
        double diff = 0.0;
        for (size_t i = 0; i < out1.data.size(); ++i) {
            diff += std::abs(out1.data[i] - out2.data[i]);
        }
        check(diff < 1e-10, "HyenaOperator determinism (bit-exact across two fresh instances with copied params)");
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===" << endl;
    return (passed == total) ? 0 : 1;
}
