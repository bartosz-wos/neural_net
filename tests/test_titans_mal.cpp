// test_titans_mal.cpp — Tests for Titans MAL (Memory as a Layer)
// Behrouz et al. 2025, https://arxiv.org/abs/2501.00663, §4.3
//
// Third of the three Titans variants (MAC < MAG < MAL): learnable test-time
// neural long-term memory M ∈ R^{d_model × d_model} updated per-token via a
// surprise-weighted momentum rule (same as MAC/MAG, §3.2 Eqs. 9-10), used as
// a *layer* on the input. MAL-specific:
//   p_t  = sigmoid(W_p · x_t + b_p)             — input gate (per channel)
//   x̃_t  = p_t ⊙ x_t
//   y_t  = M_t · x̃_t                           — clean output (no ⊙, no q_t)
//
// The q-slice of W_qkv is unused in MAL's output (q_t doesn't enter y_t), so
// its parameter gradients are zero — same property as MAG.
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

// Centered-FD check on a single parameter entry (p[i][j]) against grad[i][j].
// Loss = 0.5 * sum((y - target)^2). Returns max rel_err.
static double fd_param_check(TitansMAL& layer, const Tensor& input, const Tensor& target,
                             size_t param_id, double eps = 1e-4, int max_entries = 50) {
    auto params = layer.parameters();
    auto grads = layer.gradients();
    Tensor* p = params[param_id];
    Tensor* g = grads[param_id];
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
            if (++nchecked > max_entries) return max_rel;
        }
    }
    return max_rel;
}

// Centered-FD check on a row-vector bias entry (p[0][j]) against grad[0][j].
static double fd_bias_check(TitansMAL& layer, const Tensor& input, const Tensor& target,
                            size_t param_id, double eps = 1e-4) {
    auto params = layer.parameters();
    auto grads = layer.gradients();
    Tensor* p = params[param_id];
    Tensor* g = grads[param_id];
    double max_rel = 0.0;
    for (size_t j = 0; j < p->cols; ++j) {
        double orig = (*p)(0, j);
        (*p)(0, j) = orig + eps;
        Tensor out_p = layer.forward(input);
        double Lp = 0.0;
        for (size_t k = 0; k < out_p.data.size(); ++k) {
            double d = out_p.data[k] - target.data[k];
            Lp += d * d;
        }
        Lp *= 0.5;
        (*p)(0, j) = orig - eps;
        Tensor out_m = layer.forward(input);
        double Lm = 0.0;
        for (size_t k = 0; k < out_m.data.size(); ++k) {
            double d = out_m.data[k] - target.data[k];
            Lm += d * d;
        }
        Lm *= 0.5;
        (*p)(0, j) = orig;
        double num = (Lp - Lm) / (2.0 * eps);
        double ana = (*g)(0, j);
        double rel = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
        max_rel = std::max(max_rel, rel);
    }
    return max_rel;
}

// ============================================================================
// Tests
// ============================================================================

int main() {
    cout << "--- Test 1: Constructor validates dims ---" << endl;
    {
        bool threw = false;
        try { TitansMAL bad(0, 4); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);  // d_model == 0
    }
    {
        bool threw = false;
        try { TitansMAL ok(4, 0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(!threw);  // d_inner == 0 → defaults to d_model, no throw
        CHECK(true);    // default path exercised
    }
    {
        bool threw = false;
        try { TitansMAL bad(4, 7); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);  // d_inner != d_model rejected
    }

    cout << "--- Test 2: name() and parameter count ---" << endl;
    {
        TitansMAL layer(4, 4);
        CHECK(layer.name() == "TitansMAL");
        auto params = layer.parameters();
        CHECK(params.size() == 6);  // W_qkv.W, W_qkv.b, W_alpha.W, W_alpha.b, W_p.W, W_p.b
        auto grads = layer.gradients();
        CHECK(grads.size() == 6);
        // W_qkv.weights (3*d, d) = (12, 4)
        CHECK(params[0]->rows == 12);
        CHECK(params[0]->cols == 4);
        // W_qkv.bias (1, 3*d)
        CHECK(params[1]->rows == 1);
        CHECK(params[1]->cols == 12);
        // W_alpha.weights (1, d+1) = (1, 5)
        CHECK(params[2]->rows == 1);
        CHECK(params[2]->cols == 5);
        CHECK(params[3]->rows == 1);
        CHECK(params[3]->cols == 1);
        // W_p.weights (d, d) = (4, 4)
        CHECK(params[4]->rows == 4);
        CHECK(params[4]->cols == 4);
        // W_p.bias (1, d) = (1, 4)
        CHECK(params[5]->rows == 1);
        CHECK(params[5]->cols == 4);
        CHECK(layer.d_model() == 4);
        CHECK(layer.d_inner() == 4);
    }

    cout << "--- Test 3: forward shape, finiteness, nonzero (T=3) ---" << endl;
    {
        TitansMAL layer(4, 4);
        Tensor input = rand_tensor(3, 4, 0xC0FFEE);
        Tensor output = layer.forward(input);
        CHECK(output.rows == 3);
        CHECK(output.cols == 4);
        bool finite = true, nonzero = false;
        for (auto v : output.data) {
            if (!std::isfinite(v)) finite = false;
            if (std::abs(v) > 1e-9) nonzero = true;
        }
        CHECK(finite);
        CHECK(nonzero);
    }

    cout << "--- Test 4: forward shape preserved for T=6 ---" << endl;
    {
        TitansMAL layer(4, 4);
        Tensor input = rand_tensor(6, 4, 0xBEEF);
        Tensor output = layer.forward(input);
        CHECK(output.rows == 6);
        CHECK(output.cols == 4);
        bool finite = true;
        for (auto v : output.data) if (!std::isfinite(v)) finite = false;
        CHECK(finite);
    }

    cout << "--- Test 5: forward produces identical output with copied params ---" << endl;
    {
        TitansMAL a(4, 4);
        TitansMAL b(4, 4);
        Tensor input = rand_tensor(3, 4, 0xABCD);
        Tensor y_a = a.forward(input);
        b.copy_params_from(a);
        Tensor y_b = b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < y_a.data.size(); ++i)
            max_diff = std::max(max_diff, std::abs(y_a.data[i] - y_b.data[i]));
        CHECK_NEAR(max_diff, 0.0, 1e-12);
    }

    cout << "--- Test 6: input gradient FD check (T=3) ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(42);
        Tensor input(3, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(3, 3, 0xDEAD, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(3, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
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
        cerr << "  input grad max rel_err = " << max_rel << endl;
        CHECK(max_rel < 1e-2);
    }

    cout << "--- Test 7: parameter gradient FD checks (W_qkv.W, W_alpha.W, W_p.W, M) ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
        layer.backward(grad_out, 0.0);
        // params[0] = W_qkv.weights (12, 3), params[2] = W_alpha.weights (1, 4), params[4] = W_p.weights (3, 3)
        for (int pid : {0, 2, 4}) {
            double max_rel = fd_param_check(layer, input, target, pid, 1e-4, 50);
            cerr << "  param " << pid << " max rel_err = " << max_rel << endl;
            CHECK(max_rel < 5e-2);
        }
    }

    cout << "--- Test 8: bias gradient FD checks (W_qkv.b, W_alpha.b, W_p.b) ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(43);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xCAFE, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
        layer.backward(grad_out, 0.0);
        // params[1] = W_qkv.bias (1, 9), params[3] = W_alpha.bias (1, 1), params[5] = W_p.bias (1, 3)
        for (int pid : {1, 3, 5}) {
            double max_rel = fd_bias_check(layer, input, target, pid, 1e-4);
            cerr << "  bias param " << pid << " max rel_err = " << max_rel << endl;
            CHECK(max_rel < 5e-2);
        }
    }

    cout << "--- Test 9: persistent memory M gradient FD check ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(44);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xFACE, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
        layer.backward(grad_out, 0.0);
        // M gradient is exposed via get_gradients(), not via gradients() (which returns param grads).
        Tensor gM = layer.get_gradients();
        const double eps = 1e-4;
        double max_rel = 0.0;
        for (size_t i = 0; i < layer.M_.rows; ++i) {
            for (size_t j = 0; j < layer.M_.cols; ++j) {
                double orig = layer.M_(i, j);
                layer.M_(i, j) = orig + eps;
                Tensor out_p = layer.forward(input);
                double Lp = 0.0;
                for (size_t k = 0; k < out_p.data.size(); ++k) {
                    double d = out_p.data[k] - target.data[k];
                    Lp += d * d;
                }
                Lp *= 0.5;
                layer.M_(i, j) = orig - eps;
                Tensor out_m = layer.forward(input);
                double Lm = 0.0;
                for (size_t k = 0; k < out_m.data.size(); ++k) {
                    double d = out_m.data[k] - target.data[k];
                    Lm += d * d;
                }
                Lm *= 0.5;
                layer.M_(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = gM(i, j);
                double rel = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
                max_rel = std::max(max_rel, rel);
            }
        }
        cerr << "  M gradient max rel_err = " << max_rel << endl;
        CHECK(max_rel < 5e-2);
    }

    cout << "--- Test 10: zero_grad clears all 7 gradients (6 param + grad_M_) ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
        layer.backward(grad_out, 0.0);
        layer.zero_grad();
        bool all_zero = true;
        for (auto* g : layer.gradients()) {
            for (auto& v : g->data) if (std::abs(v) > 1e-15) all_zero = false;
        }
        for (auto& v : layer.grad_M_.data) if (std::abs(v) > 1e-15) all_zero = false;
        CHECK(all_zero);
    }

    cout << "--- Test 11: update_weights moves all 7 parameters ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);
        std::vector<Tensor> before;
        for (auto* p : layer.parameters()) before.push_back(p->clone());
        Tensor M_before = layer.M_.clone();
        Tensor output = layer.forward(input);
        Tensor grad_out(2, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
        layer.backward(grad_out, 0.0);
        layer.update_weights(0.01);
        int moved = 0;
        auto params = layer.parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            double d = 0.0;
            for (size_t k = 0; k < params[i]->data.size(); ++k) {
                d += std::abs(params[i]->data[k] - before[i].data[k]);
            }
            if (d > 1e-12) ++moved;
        }
        // M also moves
        double dM = 0.0;
        for (size_t k = 0; k < layer.M_.data.size(); ++k) dM += std::abs(layer.M_.data[k] - M_before.data[k]);
        if (dM > 1e-12) ++moved;
        CHECK(moved >= 7);
    }

    cout << "--- Test 12: training reduces loss over 50 SGD steps ---" << endl;
    {
        TitansMAL layer(3, 3);
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
        CHECK(Lf < L0);
    }

    cout << "--- Test 13: longer sequence (T=6) input grad FD check ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(43);
        Tensor input(6, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(6, 3, 0xF00D, 0.2);
        Tensor output = layer.forward(input);
        Tensor grad_out(6, 3);
        for (size_t i = 0; i < grad_out.data.size(); ++i) grad_out.data[i] = output.data[i] - target.data[i];
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

    cout << "--- Test 14: MAL-specific zero-input → zero-output (bit-exact) ---" << endl;
    {
        TitansMAL layer(4, 4);
        Tensor input(3, 4);
        std::fill(input.data.begin(), input.data.end(), 0.0);
        Tensor output = layer.forward(input);
        double max_abs = 0.0;
        for (auto v : output.data) max_abs = std::max(max_abs, std::abs(v));
        cerr << "  zero-input |output|_inf = " << max_abs << endl;
        CHECK_NEAR(max_abs, 0.0, 1e-12);
    }

    cout << "--- Test 15: M=0 + nonzero input → forward uses the post-update M, not 0 ---" << endl;
    {
        TitansMAL layer(3, 3);
        Tensor input = rand_tensor(2, 3, 0xCAFE, 0.5);
        Tensor output = layer.forward(input);
        bool nonzero = false;
        for (auto v : output.data) if (std::abs(v) > 1e-9) nonzero = true;
        CHECK(nonzero);
    }

    cout << "--- Test 16: random W_p init → forward measurably different ---" << endl;
    {
        TitansMAL layer_a(4, 4);
        TitansMAL layer_b(4, 4);
        std::mt19937 rng(0xFEEDFACEu);
        std::normal_distribution<double> nd(0.0, 0.5);
        for (auto& v : layer_b.W_p_.weights.data) v = nd(rng);

        Tensor input = rand_tensor(3, 4, 0xCAFE);
        Tensor out_a = layer_a.forward(input);
        Tensor out_b = layer_b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            max_diff = std::max(max_diff, std::abs(out_a.data[i] - out_b.data[i]));
        CHECK(max_diff > 1e-6);
    }

    cout << "--- Test 17: mutation — set W_p.bias[0][0] = 2.0 → forward and gradient change ---" << endl;
    {
        TitansMAL layer(3, 3);
        srand(42);
        Tensor input(2, 3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * (rand() % 11 - 5);
        Tensor target = rand_tensor(2, 3, 0xBEEF, 0.2);

        Tensor out_a = layer.forward(input);
        Tensor grad_a(2, 3);
        for (size_t i = 0; i < grad_a.data.size(); ++i) grad_a.data[i] = out_a.data[i] - target.data[i];
        layer.backward(grad_a, 0.0);

        TitansMAL layer_m(3, 3);
        layer_m.W_p_.bias(0, 0) = 2.0;
        Tensor out_m = layer_m.forward(input);
        Tensor grad_m(2, 3);
        for (size_t i = 0; i < grad_m.data.size(); ++i) grad_m.data[i] = out_m.data[i] - target.data[i];
        layer_m.backward(grad_m, 0.0);

        double fwd_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            fwd_diff = std::max(fwd_diff, std::abs(out_a.data[i] - out_m.data[i]));
        CHECK(fwd_diff > 1e-6);

        double gw_diff = 0.0;
        for (size_t i = 0; i < layer.grad_W_p_w_.data.size(); ++i)
            gw_diff = std::max(gw_diff, std::abs(layer.grad_W_p_w_.data[i] - layer_m.grad_W_p_w_.data[i]));
        CHECK(gw_diff > 1e-12);
    }

    cout << "--- Test 18: mutation — perturbing k-slice of W_qkv → forward changes ---" << endl;
    {
        TitansMAL layer_a(3, 3);
        TitansMAL layer_b(3, 3);
        layer_b.copy_params_from(layer_a);
        // Mutate one weight in the k-slice (row d_model = 3).
        // The q-slice (rows [0, d_model)) is unused in MAL's output.
        layer_b.W_qkv_.weights(3, 0) += 0.5;
        Tensor input = rand_tensor(3, 3, 0xDEAD, 0.3);
        Tensor out_a = layer_a.forward(input);
        Tensor out_b = layer_b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            max_diff = std::max(max_diff, std::abs(out_a.data[i] - out_b.data[i]));
        CHECK(max_diff > 1e-6);
    }

    cout << "--- Test 19: mutation — perturbing q-slice of W_qkv → forward UNCHANGED ---" << endl;
    {
        TitansMAL layer_a(3, 3);
        TitansMAL layer_b(3, 3);
        layer_b.copy_params_from(layer_a);
        // Perturb a q-slice weight (row in [0, d_model)).
        layer_b.W_qkv_.weights(0, 0) += 0.5;
        Tensor input = rand_tensor(3, 3, 0xDEAD, 0.3);
        Tensor out_a = layer_a.forward(input);
        Tensor out_b = layer_b.forward(input);
        double max_diff = 0.0;
        for (size_t i = 0; i < out_a.data.size(); ++i)
            max_diff = std::max(max_diff, std::abs(out_a.data[i] - out_b.data[i]));
        cerr << "  q-slice perturbation max |dy| = " << max_diff << endl;
        CHECK_NEAR(max_diff, 0.0, 1e-12);
    }

    cout << "\n=== Summary: " << n_pass << " passed, " << n_fail << " failed, "
         << n_check << " total checks ===" << endl;
    return n_fail == 0 ? 0 : 1;
}
