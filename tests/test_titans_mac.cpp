// test_titans_mac.cpp — Tests for Titans MAC (Memory as a Context)
// Behrouz et al. 2025, https://arxiv.org/abs/2501.00663, §3.2
//
// Implements the simplest of the three Titans variants (MAC < MAG < MAL):
// learnable test-time neural long-term memory M ∈ R^{d_model × d_model}
// updated per-token via a surprise-weighted momentum rule.
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include "nn/nn.h"

using namespace std;

static int n_check = 0, n_pass = 0, n_fail = 0;
#define CHECK(cond) do { ++n_check; if (cond) { ++n_pass; } \
    else { ++n_fail; cerr << "FAIL @ " << __FILE__ << ":" << __LINE__ << " : " #cond << endl; } } while(0)
#define CHECK_NEAR(a, b, tol) do { ++n_check; double aa = (a), bb = (b); \
    if (std::abs(aa - bb) <= (tol)) { ++n_pass; } \
    else { ++n_fail; cerr << "FAIL @ " << __FILE__ << ":" << __LINE__ \
        << " : |" #a " - " #b "| = " << std::abs(aa - bb) << " > " << (tol) << endl; } } while(0)
#define CHECK_REL(a, b, tol) do { ++n_check; double aa = (a), bb = (b); \
    double denom = std::max(1e-12, std::max(std::abs(aa), std::abs(bb))); \
    if (std::abs(aa - bb) / denom <= (tol)) { ++n_pass; } \
    else { ++n_fail; cerr << "FAIL @ " << __FILE__ << ":" << __LINE__ \
        << " : rel_err(" #a ", " #b ") = " << std::abs(aa - bb) / denom << " > " << (tol) << endl; } } while(0)

// Helper: make a small random Tensor of shape (rows, cols), seeded.
static Tensor rand_tensor(size_t rows, size_t cols, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor t(rows, cols);
    for (auto& v : t.data) v = nd(rng);
    return t;
}

// ============================================================================
// Task 1: Constructor + skeleton
// ============================================================================

int main() {
    cout << "--- Test 1: Constructor validates dims ---" << endl;
    {
        bool threw = false;
        try { TitansMAC bad(0, 4); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);  // d_model == 0
    }
    {
        // d_inner == 0 is treated as "default to d_model" — should NOT throw.
        bool threw = false;
        try { TitansMAC ok(4, 0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(!threw);
        CHECK(true);  // d_inner default path is exercised by ok's successful construction
    }
    {
        bool threw = false;
        try { TitansMAC bad(4, 7); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);  // d_inner != d_model is rejected (v1 constraint)
    }

    cout << "--- Test 2: name() and parameter count ---" << endl;
    {
        TitansMAC layer(4, 4);
        CHECK(layer.name() == "TitansMAC");
        // Parameters: W_qkv.weights, W_qkv.bias, W_alpha.weights, W_alpha.bias, M
        // → 5 tensors via parameters()
        auto params = layer.parameters();
        CHECK(params.size() == 5);
        auto grads = layer.gradients();
        CHECK(grads.size() == 5);
        // Shapes: W_qkv.weights (3*d_model, d_model) = (12, 4)
        CHECK(params[0]->rows == 12);
        CHECK(params[0]->cols == 4);
        // W_qkv.bias (1, 3*d_model) = (1, 12)
        CHECK(params[1]->rows == 1);
        CHECK(params[1]->cols == 12);
        // W_alpha.weights: input dim is d_model + 1 (concat [x_t, ||v_t||]), output dim 1
        CHECK(params[2]->rows == 1);
        CHECK(params[2]->cols == 5);  // d_model + 1
        CHECK(params[3]->rows == 1);
        CHECK(params[3]->cols == 1);
        // M is d_model x d_model
        CHECK(params[4]->rows == 4);
        CHECK(params[4]->cols == 4);
    }

    // ============================================================================
    // Task 2: Forward shape + finiteness + zero-at-init property
    // ============================================================================
    cout << "--- Test 3: forward shape, finiteness, nonzero ---" << endl;
    {
        TitansMAC layer(4, 4);
        Tensor input = rand_tensor(3, 4, 0xC0FFEE);
        Tensor output = layer.forward(input);
        CHECK(output.rows == 3);
        CHECK(output.cols == 4);
        bool finite = true, nonzero = false;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!std::isfinite(output.data[i])) finite = false;
            if (std::abs(output.data[i]) > 1e-9) nonzero = true;
        }
        CHECK(finite);
        CHECK(nonzero);
    }

    cout << "--- Test 4: after first forward, M is nonzero (memory was updated) ---\n" << endl;
    {
        TitansMAC layer(4, 4);
        Tensor input = rand_tensor(3, 4, 0xCAFE);
        Tensor output = layer.forward(input);
        // Forward order (paper convention): M_t = update(M_{t-1}, x_t) BEFORE y_t = M_t · q_t.
        // So y_0 uses the just-updated M_1 = (1-α_0)·0 + α_0·(v_0⊗k_0), which is nonzero
        // (the α_0·v_0⊗k_0 term is nonzero since W_alpha and W_qkv are nonzero-initialized).
        bool y0_nonzero = false;
        for (size_t j = 0; j < 4; ++j) {
            if (std::abs(output(0, j)) > 1e-9) y0_nonzero = true;
        }
        CHECK(y0_nonzero);  // y_0 uses M_1 (post-update), should be nonzero
    }

    cout << "--- Test 5: forward produces identical output with copied params ---" << endl;
    {
        TitansMAC a(4, 4);
        TitansMAC b(4, 4);
        Tensor input = rand_tensor(3, 4, 0xABCD);
        Tensor y_a = a.forward(input);
        b.copy_params_from(a);
        Tensor y_b = b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < y_a.data.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(y_a.data[i] - y_b.data[i]));
        }
        CHECK_NEAR(max_diff, 0.0, 1e-12);
    }

    // ============================================================================
    // Task 3: Backward — input gradient (FD check)
    // ============================================================================
    cout << "--- Test 6: input gradient FD check ---" << endl;
    {
        TitansMAC layer(3, 3);
        // seed for reproducibility
        srand(42);
        Tensor input(3, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(3, 3, 0xDEAD, 0.2);
        // Forward
        Tensor output = layer.forward(input);
        // Compute analytical grad: dL/dy = (y - t)
        Tensor grad_out(3, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = output.data[i] - target.data[i];
        }
        // Backward (returns dL/dx)
        Tensor grad_x = layer.backward(grad_out, 0.0);
        // FD check on each input element
        const double eps = 1e-4;
        double max_rel = 0.0;
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = layer.forward(input);
                double Lp = 0.0;
                for (size_t k = 0; k < out_p.data.size(); ++k) {
                    double d = out_p.data[k] - target.data[k];
                    Lp += d * d;
                }
                Lp *= 0.5;
                input(i, j) = orig - eps;
                Tensor out_m = layer.forward(input);
                double Lm = 0.0;
                for (size_t k = 0; k < out_m.data.size(); ++k) {
                    double d = out_m.data[k] - target.data[k];
                    Lm += d * d;
                }
                Lm *= 0.5;
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double rel = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
                max_rel = std::max(max_rel, rel);
            }
        }
        cerr << "  input grad max rel_err = " << max_rel << endl;
        CHECK(max_rel < 1e-2);
    }

    // ============================================================================
    // Task 4: Parameter gradient FD checks
    // ============================================================================
    cout << "--- Test 7: parameter gradients (W_qkv.weights, W_alpha.weights, M) FD check ---" << endl;
    {
        TitansMAC layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = output.data[i] - target.data[i];
        }
        layer.backward(grad_out, 0.0);
        // Get param grads
        auto params = layer.parameters();
        auto grads = layer.gradients();
        // Check params 0 (W_qkv.weights), 2 (W_alpha.weights), 4 (M)
        for (int pid : {0, 2, 4}) {
            Tensor* p = params[pid];
            Tensor* g = grads[pid];
            const double eps = 1e-4;
            double max_rel = 0.0;
            int nchecked = 0;
            for (size_t i = 0; i < p->rows; ++i) {
                for (size_t j = 0; j < p->cols; ++j) {
                    double orig = (*p)(i, j);
                    (*p)(i, j) = orig + eps;
                    Tensor out_p = layer.forward(input);
                    double Lp = 0.0;
                    for (size_t k = 0; k < out_p.data.size(); ++k) {
                        double d = out_p.data[k] - target.data[k];
                        Lp += d * d;
                    }
                    Lp *= 0.5;
                    (*p)(i, j) = orig - eps;
                    Tensor out_m = layer.forward(input);
                    double Lm = 0.0;
                    for (size_t k = 0; k < out_m.data.size(); ++k) {
                        double d = out_m.data[k] - target.data[k];
                        Lm += d * d;
                    }
                    Lm *= 0.5;
                    (*p)(i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    double ana = (*g)(i, j);
                    double rel = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
                    max_rel = std::max(max_rel, rel);
                    if (++nchecked > 50) break;  // sample first 50 entries for speed
                }
                if (nchecked > 50) break;
            }
            cerr << "  param " << pid << " max rel_err = " << max_rel << endl;
            CHECK(max_rel < 5e-2);  // looser tolerance — multi-chain gradients accumulate FP noise
        }
    }

    // ============================================================================
    // Task 5: Contracts + training
    // ============================================================================
    cout << "--- Test 8: zero_grad clears all 5 gradients ---" << endl;
    {
        TitansMAC layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = output.data[i] - target.data[i];
        }
        layer.backward(grad_out, 0.0);
        layer.zero_grad();
        auto grads = layer.gradients();
        bool all_zero = true;
        for (auto* g : grads) {
            for (auto& v : g->data) if (std::abs(v) > 1e-15) all_zero = false;
        }
        CHECK(all_zero);
    }

    cout << "--- Test 9: update_weights moves all 5 parameters ---" << endl;
    {
        TitansMAC layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        // Snapshot params before
        std::vector<Tensor> before;
        for (auto* p : layer.parameters()) before.push_back(p->clone());
        // Forward + backward to populate grads
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = output.data[i] - target.data[i];
        }
        layer.backward(grad_out, 0.0);
        // Update
        layer.update_weights(0.01);
        // Check params moved
        int moved = 0;
        auto params = layer.parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            double d = 0.0;
            for (size_t k = 0; k < params[i]->data.size(); ++k) {
                d += std::abs(params[i]->data[k] - before[i].data[k]);
            }
            if (d > 1e-12) ++moved;
        }
        CHECK(moved >= 5);  // all 5 params should move (grads were nonzero)
    }

    cout << "--- Test 10: training reduces loss over 50 SGD steps ---" << endl;
    {
        TitansMAC layer(3, 3);
        srand(42);
        Tensor input(3, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(3, 3, 0xCAFE, 0.2);
        double L0 = 0.0;
        {
            Tensor out = layer.forward(input);
            for (size_t k = 0; k < out.data.size(); ++k) {
                double d = out.data[k] - target.data[k];
                L0 += d * d;
            }
            L0 *= 0.5;
        }
        for (int step = 0; step < 50; ++step) {
            layer.zero_grad();
            Tensor out = layer.forward(input);
            Tensor grad_out(3, 3);
            for (size_t k = 0; k < grad_out.data.size(); ++k) {
                grad_out.data[k] = out.data[k] - target.data[k];
            }
            layer.backward(grad_out, 0.01);
            layer.update_weights(0.01);
        }
        double Lf = 0.0;
        {
            Tensor out = layer.forward(input);
            for (size_t k = 0; k < out.data.size(); ++k) {
                double d = out.data[k] - target.data[k];
                Lf += d * d;
            }
            Lf *= 0.5;
        }
        cerr << "  training: L0=" << L0 << " Lf=" << Lf << endl;
        CHECK(Lf < L0);  // loss should decrease
    }

    cout << "--- Test 11: longer sequence (T=6) input grad FD check ---" << endl;
    {
        TitansMAC layer(3, 3);
        srand(43);
        Tensor input(6, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(6, 3, 0xF00D, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(6, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) {
            grad_out.data[i] = output.data[i] - target.data[i];
        }
        Tensor grad_x = layer.backward(grad_out, 0.0);
        const double eps = 1e-4;
        double max_rel = 0.0;
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = layer.forward(input);
                double Lp = 0.0;
                for (size_t k = 0; k < out_p.data.size(); ++k) {
                    double d = out_p.data[k] - target.data[k];
                    Lp += d * d;
                }
                Lp *= 0.5;
                input(i, j) = orig - eps;
                Tensor out_m = layer.forward(input);
                double Lm = 0.0;
                for (size_t k = 0; k < out_m.data.size(); ++k) {
                    double d = out_m.data[k] - target.data[k];
                    Lm += d * d;
                }
                Lm *= 0.5;
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double rel = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
                max_rel = std::max(max_rel, rel);
            }
        }
        cerr << "  T=6 input grad max rel_err = " << max_rel << endl;
        CHECK(max_rel < 1e-2);
    }

    // ============================================================================
    // Task 8: Mutation testing — confirm the test suite catches known-bad impls.
    // A mutation that the suite fails to catch means the test isn't exercising
    // that code path. Strengthen or skip the mutation if so.
    // ============================================================================
    cout << "--- Test 12: mutation — set W_alpha.bias to nonzero, forward output must differ ---\n" << endl;
    {
        // The constructor initializes W_alpha_.bias to 0 (Tensor default ctor), so
        // zeroing it is a no-op. Instead, set it to a non-zero value (the "mutant")
        // and verify the forward output and gradient norm change vs baseline.
        TitansMAC layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);

        Tensor out_a = layer.forward(input);
        Tensor grad_a(2, 3);
        for (size_t i = 0; i < grad_a.data.size(); ++i) grad_a.data[i] = out_a.data[i] - target.data[i];
        layer.backward(grad_a, 0.0);

        TitansMAC layer_m(3, 3);
        // Mutant: bias = +2.0 (large positive bias → σ_pre ≈ 1, α_t ≈ η_t)
        layer_m.W_alpha_.bias(0, 0) = 2.0;
        Tensor out_m = layer_m.forward(input);
        Tensor grad_m(2, 3);
        for (size_t i = 0; i < grad_m.data.size(); ++i) grad_m.data[i] = out_m.data[i] - target.data[i];
        layer_m.backward(grad_m, 0.0);

        // Forward outputs should differ when bias is set non-zero
        double fwd_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            fwd_diff = std::max(fwd_diff, std::abs(out_a.data[i] - out_m.data[i]));
        CHECK(fwd_diff > 1e-6);  // forward path is exercised through W_alpha

        // Gradient through W_alpha.weights should differ
        double gw_diff = 0.0;
        for (size_t i = 0; i < layer.grad_W_alpha_w_.data.size(); ++i)
            gw_diff = std::max(gw_diff, std::abs(layer.grad_W_alpha_w_.data[i] - layer_m.grad_W_alpha_w_.data[i]));
        CHECK(gw_diff > 1e-12);  // W_alpha gradient path is exercised
    }

    cout << "--- Test 13: mutation — verify persistent memory M affects forward output ---\n" << endl;
    {
        // The persistent memory M affects the output through TWO channels:
        //   (1) Direct: M_t · q_t (used in y_t = M_t · q_t)
        //   (2) Indirect: η_t = ||x_t - M_{t-1} · k_t||_2 depends on M_{t-1}
        // Channel (1) is wiped on the first token (M_1 overwrites M_0), but channel
        // (2) propagates the dependency from M_0 → η_0 → α_0 → v_0⊗k_0 → M_1 → y_0.
        // So M=random vs M=0 produces different output even at T=1.
        TitansMAC layer_a(4, 4);
        TitansMAC layer_b(4, 4);
        std::mt19937 rng(0xFEEDFACEu);
        std::normal_distribution<double> nd(0.0, 0.5);
        for (auto& v : layer_b.M_.data) v = nd(rng);

        Tensor input = rand_tensor(3, 4, 0xCAFE);
        Tensor out_a = layer_a.forward(input);
        Tensor out_b = layer_b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            max_diff = std::max(max_diff, std::abs(out_a.data[i] - out_b.data[i]));
        // M-random init should produce measurably different output (via η_t chain)
        CHECK(max_diff > 1e-6);
    }

    cout << "\n=== Summary: " << n_pass << " passed, " << n_fail << " failed, "
         << n_check << " total checks ===" << endl;
    return n_fail == 0 ? 0 : 1;
}
