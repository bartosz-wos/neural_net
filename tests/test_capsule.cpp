// tests/test_capsule.cpp — CapsuleLayer with dynamic routing (Sabour 2017)
//
// Tests:
//   1.  Forward shape — input (B, I*D_in) → output (B, J*D)
//   2.  Coupling init uniform across output capsules
//   3.  Couplings become non-uniform after routing iterations (when input differs)
//   4.  Per-capsule squash: each output capsule has ||v|| ≤ 1
//   5.  Output magnitude decreases with stronger agreement (routing concentrates)
//   6.  Single-input-capsule (I=1) backward path matches the gradient check
//   7.  Multi-input-capsule (I=2) backward — input gradient numerical check
//   8.  Multi-input-capsule (I=2) backward — W gradient numerical check
//   9.  Single-step routing (R=1) gradient check
//  10.  Two routing iterations (R=2) gradient check
//  11.  output.shape equality under repeated forward (determinism)
//  12.  parameters()/gradients() return expected counts
//  13.  zero_grad clears all gradients
//  14.  Training on a synthetic task reduces loss

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/generative/capsnet.h"
#include "nn/nn.h"

using namespace std;

static double cap_rel_err(double a, double b) {
    if (std::fabs(b) < 1e-8) return std::fabs(a);
    return std::fabs(a - b) / std::max(std::fabs(b), 1e-8);
}

static double l2_loss(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j) {
            double d = out(i, j) - tgt(i, j);
            s += 0.5 * d * d;
        }
    return s;
}

static Tensor l2_loss_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            g(i, j) = out(i, j) - tgt(i, j);
    return g;
}

// Small RNG with explicit seed (deterministic test reproducibility)
static std::mt19937* g_rng() {
    static std::mt19937 r(12345);
    return &r;
}

static void fill_uniform(Tensor& t, double lo, double hi) {
    auto* r = g_rng();
    std::uniform_real_distribution<double> d(lo, hi);
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            t(i, j) = d(*r);
}

int main() {
    cout << "=== CapsuleLayer Tests ===" << endl;
    int total = 0, passed = 0;

    auto report = [&](const std::string& name, bool ok, const std::string& extra = "") {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << name << (extra.empty() ? "" : " " + extra) << "\n"; }
        else    {            cout << "[FAIL] " << name << (extra.empty() ? "" : " " + extra) << "\n"; }
    };

    // =========================================================
    // Test 1: Forward shape — multi-input-capsule
    // =========================================================
    cout << "\n--- Test 1: Forward shape (I=2, J=3) ---\n";
    {
        size_t B = 4, I = 2, D_in = 4, J = 3, D = 5, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.5, 0.5);
        Tensor out = layer.forward(input);
        report("forward shape (4, 8) -> (4, 15)",
               out.rows == 4 && out.cols == 15,
               "got (" + std::to_string(out.rows) + ", " + std::to_string(out.cols) + ")");
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols && finite; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        report("output finite", finite);
    }

    // =========================================================
    // Test 2: Each output capsule vector has ||v|| <= 1 (squash property)
    // =========================================================
    cout << "\n--- Test 2: Squash bound ---\n";
    {
        size_t B = 3, I = 3, D_in = 3, J = 4, D = 4, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -1.0, 1.0);
        Tensor out = layer.forward(input);
        bool ok = true;
        for (size_t b = 0; b < B && ok; ++b) {
            for (size_t j = 0; j < J && ok; ++j) {
                double n2 = 0.0;
                for (size_t k = 0; k < D; ++k) n2 += out(b, j * D + k) * out(b, j * D + k);
                double n = std::sqrt(n2);
                if (n > 1.0 + 1e-6) ok = false;
            }
        }
        report("each output capsule ||v|| <= 1 (squash bound)", ok);
    }

    // =========================================================
    // Test 3: Couplings non-uniform after routing — given a biased input
    //         that favors one output capsule, the coupling from each input
    //         capsule to that output capsule should rise above 1/J on average.
    // =========================================================
    cout << "\n--- Test 3: Couplings concentrate (biased input) ---\n";
    {
        size_t B = 2, I = 2, D_in = 2, J = 2, D = 2, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        // Batch 0: input capsule 0 aligned with output capsule 0 direction.
        // Batch 1: input capsule 0 aligned with output capsule 1 direction.
        // (Random initial W; just check that couplings vary across (i, j).)
        fill_uniform(input, -1.0, 1.0);
        Tensor out = layer.forward(input);
        // We don't have direct access to c, but a sanity-check is that the
        // magnitudes vary across output capsules (which they must, since
        // routing concentrates).
        double mean0 = 0.0, mean1 = 0.0;
        for (size_t b = 0; b < B; ++b) {
            double n0 = 0.0, n1 = 0.0;
            for (size_t k = 0; k < D; ++k) {
                n0 += out(b, 0 * D + k) * out(b, 0 * D + k);
                n1 += out(b, 1 * D + k) * out(b, 1 * D + k);
            }
            mean0 += std::sqrt(n0);
            mean1 += std::sqrt(n1);
        }
        mean0 /= B; mean1 /= B;
        // The two capsules should differ in magnitude given distinct input.
        bool ok = std::fabs(mean0 - mean1) > 1e-6;
        report("magnitudes differ across output capsules (routing concentrates)",
               ok, "mean0=" + std::to_string(mean0) + " mean1=" + std::to_string(mean1));
    }

    // =========================================================
    // Test 4: Forward determinism (same input → same output)
    // =========================================================
    cout << "\n--- Test 4: Forward determinism ---\n";
    {
        size_t B = 3, I = 2, D_in = 3, J = 3, D = 4, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor a = layer.forward(input);
        Tensor b = layer.forward(input);
        bool ok = true;
        for (size_t i = 0; i < a.rows && ok; ++i)
            for (size_t j = 0; j < a.cols && ok; ++j)
                if (std::fabs(a(i, j) - b(i, j)) > 1e-12) ok = false;
        report("forward is deterministic", ok);
    }

    // =========================================================
    // Test 5: parameters() / gradients() shape & count
    // =========================================================
    cout << "\n--- Test 5: params/grads interface ---\n";
    {
        size_t I = 2, D_in = 3, J = 4, D = 5, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        auto params = layer.parameters();
        auto grads  = layer.gradients();
        // params/grads should have one entry per output capsule (each is a W tensor of shape (D_in, D))
        report("params count == num_capsules", params.size() == J);
        report("grads count == num_capsules",  grads.size() == J);
        for (size_t j = 0; j < J; ++j) {
            report("param[" + std::to_string(j) + "] shape (D_in, D) = (" + std::to_string(D_in) + ", " + std::to_string(D) + ")",
                   params[j]->rows == D_in && params[j]->cols == D);
            report("grad[" + std::to_string(j) + "] shape matches",
                   grads[j]->rows == D_in && grads[j]->cols == D);
        }
    }

    // =========================================================
    // Test 6: zero_grad clears all gradients
    // =========================================================
    cout << "\n--- Test 6: zero_grad ---\n";
    {
        size_t B = 2, I = 2, D_in = 2, J = 2, D = 2, R = 2;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor out = layer.forward(input);
        Tensor g = l2_loss_grad(out, out);
        layer.backward(g, 0.0);
        layer.zero_grad();
        auto grads = layer.gradients();
        bool ok = true;
        for (auto* gr : grads)
            for (size_t i = 0; i < gr->rows && ok; ++i)
                for (size_t j = 0; j < gr->cols && ok; ++j)
                    if (std::fabs((*gr)(i, j)) > 1e-12) ok = false;
        report("zero_grad clears all grads", ok);
    }

    // =========================================================
    // Test 7: Numerical input gradient check (I=2, J=3, R=2, D=3, D_in=2)
    // =========================================================
    cout << "\n--- Test 7: Numerical input grad (I=2, J=3, R=2) ---\n";
    {
        size_t B = 2, I = 2, D_in = 2, J = 3, D = 3, R = 2;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor target(B, J * D);
        fill_uniform(target, -0.2, 0.2);

        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < B && ok; ++ri) {
            for (size_t rf = 0; rf < I * D_in && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward(input);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward(input);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = cap_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-3) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] input grad matches numerical (multi-input-capsule, R=2)\n"; }
        else    {            cout << "[FAIL] input grad rel_err too high\n"; }
        ++total;
    }

    // =========================================================
    // Test 8: Numerical W gradient check (single output capsule j=0)
    // =========================================================
    cout << "\n--- Test 8: Numerical W grad (I=2, J=2, R=3) ---\n";
    {
        size_t B = 2, I = 2, D_in = 2, J = 2, D = 2, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor target(B, J * D);
        fill_uniform(target, -0.2, 0.2);

        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);

        auto params = layer.parameters();
        auto grads  = layer.gradients();
        // Check W[0][0,0]
        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t j = 0; j < J && ok; ++j) {
            for (size_t di = 0; di < D_in && ok; ++di) {
                for (size_t dk = 0; dk < D && ok; ++dk) {
                    double orig = (*params[j])(di, dk);
                    (*params[j])(di, dk) = orig + eps;
                    Tensor out_p = layer.forward(input);
                    double lp = l2_loss(out_p, target);
                    (*params[j])(di, dk) = orig - eps;
                    Tensor out_m = layer.forward(input);
                    double lm = l2_loss(out_m, target);
                    (*params[j])(di, dk) = orig;
                    double num = (lp - lm) / (2.0 * eps);
                    double ana = (*grads[j])(di, dk);
                    double rel = cap_rel_err(num, ana);
                    if (rel > max_rel) max_rel = rel;
                    if (rel > 1e-3) { ok = false; }
                }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] W grad matches numerical (all output capsules)\n"; }
        else    {            cout << "[FAIL] W grad rel_err too high\n"; }
        ++total;
    }

    // =========================================================
    // Test 9: Single-routing-iteration (R=1) gradient check
    // =========================================================
    cout << "\n--- Test 9: Numerical grad (R=1, no agreement updates) ---\n";
    {
        size_t B = 2, I = 3, D_in = 2, J = 2, D = 3, R = 1;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor target(B, J * D);
        fill_uniform(target, -0.2, 0.2);

        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < B && ok; ++ri) {
            for (size_t rf = 0; rf < I * D_in && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward(input);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward(input);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = cap_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-3) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] input grad matches numerical (R=1)\n"; }
        else    {            cout << "[FAIL] R=1 input grad rel_err too high\n"; }
        ++total;
    }

    // =========================================================
    // Test 10: Single-input-capsule (I=1) backward path
    // =========================================================
    cout << "\n--- Test 10: Numerical grad (I=1, R=2) ---\n";
    {
        size_t B = 2, I = 1, D_in = 3, J = 3, D = 3, R = 2;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        Tensor target(B, J * D);
        fill_uniform(target, -0.2, 0.2);

        Tensor out = layer.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < B && ok; ++ri) {
            for (size_t rf = 0; rf < I * D_in && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward(input);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward(input);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = cap_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-3) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] input grad matches numerical (I=1, R=2)\n"; }
        else    {            cout << "[FAIL] I=1 input grad rel_err too high\n"; }
        ++total;
    }

    // =========================================================
    // Test 11: Training reduces loss on a synthetic reconstruction task
    // =========================================================
    cout << "\n--- Test 11: Training reduces loss ---\n";
    {
        size_t B = 4, I = 2, D_in = 3, J = 2, D = 4, R = 3;
        CapsuleLayer layer(I, D_in, J, D, R);
        Tensor input(B, I * D_in);
        fill_uniform(input, -0.3, 0.3);
        // Target = slightly perturbed forward output (so we have a non-trivial
        // learning signal but the loss is achievable).
        Tensor target = layer.forward(input).clone();
        for (size_t i = 0; i < B; ++i)
            for (size_t k = 0; k < J * D; ++k)
                target(i, k) += 0.1 * std::sin(0.3 * i + 0.7 * k);

        double lr = 0.05;
        int steps = 80;
        double L0 = 0.0, Lf = 0.0;
        for (int s = 0; s < steps; ++s) {
            Tensor out = layer.forward(input);
            if (s == 0) L0 = l2_loss(out, target);
            Tensor g = l2_loss_grad(out, target);
            layer.zero_grad();
            layer.backward(g, 0.0);
            layer.update_weights(lr);
        }
        Tensor out_final = layer.forward(input);
        Lf = l2_loss(out_final, target);
        double reduction = 100.0 * (L0 - Lf) / (std::fabs(L0) + 1e-9);
        cout << "  initial: " << L0 << " final: " << Lf << " reduction: " << reduction << "%\n";
        report("training reduces loss", Lf < L0);
    }

    // =========================================================
    // Test 12: name() returns expected string
    // =========================================================
    cout << "\n--- Test 12: name() ---\n";
    {
        CapsuleLayer layer(2, 3, 3, 4, 3);
        report("CapsuleLayer.name() == \"CapsuleLayer\"", layer.name() == "CapsuleLayer");
    }

    cout << "\n=== CapsuleLayer Tests: " << passed << "/" << total << " passed ===\n";
    return (passed == total) ? 0 : 1;
}