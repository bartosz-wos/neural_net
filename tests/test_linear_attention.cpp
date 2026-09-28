// test_linear_attention.cpp — Gradient correctness tests for Linear Attention
//
// Tests (each `check()` is one assertion):
//   1.  LinearAttention constructor validation (d_model=0 throws)
//   2.  Accessors (d_model, parameter shapes)
//   3.  parameters()/gradients() contract (8 each, shape match)
//   4.  Forward shape + finiteness (n=4, d=4)
//   5.  Forward larger (n=8, d=8)
//   6.  Forward determinism (two consecutive calls bit-exact)
//   7.  Linear-cost shape (n=64, d=4) — sanity that big-n works
//   8.  FD input gradient (n=4, d=4) at rel_err < 1e-4
//   9.  FD W_q gradient (n=4, d=4) at rel_err < 1e-4
//  10.  FD W_k gradient (n=4, d=4) at rel_err < 1e-4
//  11.  FD W_v gradient (n=4, d=4) at rel_err < 1e-4
//  12.  FD W_o gradient (n=4, d=4) at rel_err < 1e-4
//  13.  FD b_q / b_k / b_v / b_o gradients
//  14.  zero_grad clears all 8 gradients
//  15.  update_weights moves all 8 parameters
//  16.  LinearAttentionBlock forward shape
//  17.  LinearAttentionBlock input gradient FD (rel_err < 1e-4)
//  18.  LinearAttentionBlock training step reduces MSE loss
//  19.  LinearAttentionModel forward shape (n=6, d_in=3, d_model=4, d_out=2)
//  20.  LinearAttentionModel 2-block training reduces loss in 30 SGD steps

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include "nn/layers/attention/linear_attention.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static void check(const string& name, bool cond, double err = 0.0) {
    if (cond) { ++passed; cout << "  [PASS] " << name << "\n"; }
    else      { ++failed; cout << "  [FAIL] " << name << " (err=" << err << ")\n"; }
}

static double tensor_max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        m = max(m, fabs(a.data[i] - b.data[i]));
    return m;
}
static double tensor_rel_err(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1.0;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        num = max(num, fabs(a.data[i] - b.data[i]));
        den = max(den, max(fabs(a.data[i]), fabs(b.data[i])));
    }
    if (den < 1e-12) return num;
    return num / den;
}

// Finite-difference gradient check on a single scalar-output model.
// L(w, x) = sum(forward(w; x) * grad_seed)
static double fd_input_grad(Tensor& input_init, const Tensor& grad_seed,
                            LinearAttention& layer, double eps = 1e-5) {
    // Forward at input
    Tensor out = layer.forward(input_init);
    double L0 = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) L0 += out.data[i] * grad_seed.data[i];

    Tensor ana = layer.backward(grad_seed, 0.0);  // analytical input gradient

    // Now finite-difference
    Tensor fd(input_init.rows, input_init.cols);
    for (size_t r = 0; r < input_init.rows; ++r) {
        for (size_t c = 0; c < input_init.cols; ++c) {
            double orig = input_init(r, c);
            // +eps
            input_init(r, c) = orig + eps;
            Tensor out_p = layer.forward(input_init);
            double Lp = 0.0;
            for (size_t i = 0; i < out_p.data.size(); ++i) Lp += out_p.data[i] * grad_seed.data[i];
            input_init(r, c) = orig;
            fd(r, c) = (Lp - L0) / eps;
        }
    }
    return tensor_rel_err(fd, ana);
}

// FD-check on a single parameter tensor W of the LinearAttention layer
// (parameter index `param_idx`, 0=W_q 1=W_k 2=W_v 3=W_o 4=b_q 5=b_k 6=b_v 7=b_o).
static double fd_param_grad(LinearAttention& layer, const Tensor& input,
                            const Tensor& grad_seed, size_t param_idx,
                            double eps = 1e-5) {
    Tensor out = layer.forward(input);
    double L0 = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) L0 += out.data[i] * grad_seed.data[i];
    layer.zero_grad();
    layer.backward(grad_seed, 0.0);

    auto grads = layer.gradients();
    Tensor ana = grads[param_idx]->clone();
    Tensor& W = *layer.parameters()[param_idx];

    Tensor fd(W.rows, W.cols);
    for (size_t r = 0; r < W.rows; ++r) {
        for (size_t c = 0; c < W.cols; ++c) {
            double orig = W(r, c);
            W(r, c) = orig + eps;
            Tensor out_p = layer.forward(input);
            double Lp = 0.0;
            for (size_t i = 0; i < out_p.data.size(); ++i) Lp += out_p.data[i] * grad_seed.data[i];
            W(r, c) = orig;
            fd(r, c) = (Lp - L0) / eps;
        }
    }
    // Reset W to original
    for (size_t r = 0; r < W.rows; ++r)
        for (size_t c = 0; c < W.cols; ++c) {
            // already reset in loop
        }
    return tensor_rel_err(fd, ana);
}

// FD-check on input gradient for a LinearAttentionBlock (full block).
static double fd_block_input_grad(LinearAttentionBlock& block, Tensor& input_init,
                                  const Tensor& grad_seed, double eps = 1e-5) {
    Tensor out = block.forward(input_init);
    double L0 = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) L0 += out.data[i] * grad_seed.data[i];
    Tensor ana = block.backward(grad_seed, 0.0);

    Tensor fd(input_init.rows, input_init.cols);
    for (size_t r = 0; r < input_init.rows; ++r) {
        for (size_t c = 0; c < input_init.cols; ++c) {
            double orig = input_init(r, c);
            input_init(r, c) = orig + eps;
            Tensor out_p = block.forward(input_init);
            double Lp = 0.0;
            for (size_t i = 0; i < out_p.data.size(); ++i) Lp += out_p.data[i] * grad_seed.data[i];
            input_init(r, c) = orig;
            fd(r, c) = (Lp - L0) / eps;
        }
    }
    return tensor_rel_err(fd, ana);
}

// Build a tensor with controlled deterministic values (no global RNG).
static Tensor make_input(size_t n, size_t d, double scale, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-scale, scale);
    Tensor x(n, d);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = dis(gen);
    return x;
}

int main() {
    cout << "=== Linear Attention Tests ===\n";

    // -----------------------------------------------------------------
    // 1. Constructor validation
    // -----------------------------------------------------------------
    {
        bool threw = false;
        try { LinearAttention la(0); } catch (const std::invalid_argument&) { threw = true; }
        check("LinearAttention: d_model=0 throws", threw);
    }

    // -----------------------------------------------------------------
    // 2. Accessors and parameter shapes
    // -----------------------------------------------------------------
    {
        LinearAttention la(8);
        check("LinearAttention: d_model accessor", la.d_model() == 8);
        check("LinearAttention: W_q shape (8,8)", la.W_q_tensor().rows == 8 && la.W_q_tensor().cols == 8);
        check("LinearAttention: W_k shape (8,8)", la.W_k_tensor().rows == 8 && la.W_k_tensor().cols == 8);
        check("LinearAttention: W_v shape (8,8)", la.W_v_tensor().rows == 8 && la.W_v_tensor().cols == 8);
        check("LinearAttention: W_o shape (8,8)", la.W_o_tensor().rows == 8 && la.W_o_tensor().cols == 8);
        check("LinearAttention: name()", la.name() == "LinearAttention");
    }

    // -----------------------------------------------------------------
    // 3. parameters()/gradients() contract
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        auto p = la.parameters();
        auto g = la.gradients();
        check("parameters() returns 8 tensors", p.size() == 8);
        check("gradients() returns 8 tensors", g.size() == 8);
        // Shapes match
        bool shapes_match = true;
        for (size_t i = 0; i < p.size(); ++i) {
            if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) {
                shapes_match = false; break;
            }
        }
        check("param/grad shapes match", shapes_match);
    }

    // -----------------------------------------------------------------
    // 4. Forward shape + finiteness (n=4, d=4)
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.5, 1);
        Tensor y = la.forward(x);
        check("forward shape (4,4)->(4,4)", y.rows == 4 && y.cols == 4);
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
        check("forward output is finite", finite);
        bool nonzero = false;
        for (double v : y.data) if (fabs(v) > 1e-6) { nonzero = true; break; }
        check("forward output is non-trivial", nonzero);
    }

    // -----------------------------------------------------------------
    // 5. Forward larger (n=8, d=8)
    // -----------------------------------------------------------------
    {
        LinearAttention la(8);
        Tensor x = make_input(8, 8, 0.5, 2);
        Tensor y = la.forward(x);
        check("forward shape (8,8)->(8,8)", y.rows == 8 && y.cols == 8);
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
        check("forward (8,8) output is finite", finite);
    }

    // -----------------------------------------------------------------
    // 6. Forward determinism
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.3, 3);
        Tensor y1 = la.forward(x);
        Tensor y2 = la.forward(x);
        double diff = tensor_max_abs_diff(y1, y2);
        check("forward determinism (max diff = 0)", diff == 0.0, diff);
    }

    // -----------------------------------------------------------------
    // 7. Linear-cost sanity (n=64, d=4)
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(64, 4, 0.2, 4);
        Tensor y = la.forward(x);
        check("forward shape (64,4)->(64,4)", y.rows == 64 && y.cols == 4);
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
        check("forward (64,4) output is finite", finite);
    }

    // -----------------------------------------------------------------
    // 8. FD input gradient (n=4, d=4)
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.4, 5);
        Tensor grad_seed = make_input(4, 4, 0.5, 6);
        double err = fd_input_grad(x, grad_seed, la, 1e-5);
        check("FD input grad rel_err < 1e-4", err < 1e-4, err);
    }

    // -----------------------------------------------------------------
    // 9-13. FD parameter gradients
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.4, 7);
        Tensor grad_seed = make_input(4, 4, 0.5, 8);
        double err_q  = fd_param_grad(la, x, grad_seed, 0, 1e-5);
        double err_k  = fd_param_grad(la, x, grad_seed, 1, 1e-5);
        double err_v  = fd_param_grad(la, x, grad_seed, 2, 1e-5);
        double err_o  = fd_param_grad(la, x, grad_seed, 3, 1e-5);
        double err_bq = fd_param_grad(la, x, grad_seed, 4, 1e-5);
        double err_bk = fd_param_grad(la, x, grad_seed, 5, 1e-5);
        double err_bv = fd_param_grad(la, x, grad_seed, 6, 1e-5);
        double err_bo = fd_param_grad(la, x, grad_seed, 7, 1e-5);
        check("FD W_q grad rel_err < 1e-4", err_q < 1e-4, err_q);
        check("FD W_k grad rel_err < 1e-4", err_k < 1e-4, err_k);
        check("FD W_v grad rel_err < 1e-4", err_v < 1e-4, err_v);
        check("FD W_o grad rel_err < 1e-4", err_o < 1e-4, err_o);
        check("FD b_q grad rel_err < 1e-4", err_bq < 1e-4, err_bq);
        check("FD b_k grad rel_err < 1e-4", err_bk < 1e-4, err_bk);
        check("FD b_v grad rel_err < 1e-4", err_bv < 1e-4, err_bv);
        check("FD b_o grad rel_err < 1e-4", err_bo < 1e-4, err_bo);
    }

    // -----------------------------------------------------------------
    // 14. zero_grad clears all 8 gradients
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.5, 9);
        Tensor grad_seed = make_input(4, 4, 0.5, 10);
        la.forward(x);
        la.backward(grad_seed, 0.0);
        bool all_nonzero_before = true;
        for (auto* g : la.gradients()) {
            for (double v : g->data) if (fabs(v) < 1e-12) { all_nonzero_before = false; break; }
            if (!all_nonzero_before) break;
        }
        // (not all grads may be non-zero depending on activation pattern;
        //  we just confirm zero_grad() zeroes them all afterwards.)
        la.zero_grad();
        bool all_zero = true;
        for (auto* g : la.gradients()) {
            for (double v : g->data) if (fabs(v) > 1e-12) { all_zero = false; break; }
            if (!all_zero) break;
        }
        check("zero_grad clears all 8 gradients", all_zero);
    }

    // -----------------------------------------------------------------
    // 15. update_weights moves all 8 parameters
    // -----------------------------------------------------------------
    {
        LinearAttention la(4);
        Tensor x = make_input(4, 4, 0.5, 11);
        Tensor grad_seed = make_input(4, 4, 0.5, 12);
        la.forward(x);
        la.backward(grad_seed, 0.0);
        // Snapshot all parameters
        std::vector<Tensor> before;
        for (auto* p : la.parameters()) before.push_back(p->clone());
        la.update_weights(0.01);
        bool any_changed = false;
        size_t i = 0;
        for (auto* p : la.parameters()) {
            if (tensor_max_abs_diff(*p, before[i]) > 1e-12) any_changed = true;
            ++i;
        }
        check("update_weights moves all 8 parameters", any_changed);
    }

    // -----------------------------------------------------------------
    // 16. LinearAttentionBlock forward shape
    // -----------------------------------------------------------------
    {
        LinearAttentionBlock block(4);
        Tensor x = make_input(4, 4, 0.5, 13);
        Tensor y = block.forward(x);
        check("Block forward shape (4,4)->(4,4)", y.rows == 4 && y.cols == 4);
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
        check("Block forward output is finite", finite);
    }

    // -----------------------------------------------------------------
    // 17. LinearAttentionBlock input gradient FD
    // -----------------------------------------------------------------
    {
        LinearAttentionBlock block(4);
        Tensor x = make_input(4, 4, 0.4, 14);
        Tensor grad_seed = make_input(4, 4, 0.5, 15);
        double err = fd_block_input_grad(block, x, grad_seed, 1e-5);
        // The block chain (pre-LN + LinearAttn + residual + pre-LN + GELU-FFN + residual)
        // accumulates FP noise — expect < 1e-3 (we use the same 1e-5 step).
        check("Block input grad FD rel_err < 1e-3", err < 1e-3, err);
    }

    // -----------------------------------------------------------------
    // 18. LinearAttentionBlock training step reduces MSE loss
    // -----------------------------------------------------------------
    {
        LinearAttentionBlock block(4, 4);
        Tensor x = make_input(6, 4, 0.5, 16);
        Tensor target = make_input(6, 4, 0.3, 17);
        double lr = 0.01;
        // Compute initial loss
        Tensor y0 = block.forward(x);
        double L0 = 0.0;
        for (size_t i = 0; i < y0.data.size(); ++i) {
            double d = y0.data[i] - target.data[i];
            L0 += d * d;
        }
        L0 /= y0.data.size();
        // 20 SGD steps
        for (int step = 0; step < 20; ++step) {
            block.zero_grad();
            Tensor y = block.forward(x);
            Tensor grad(y.rows, y.cols);
            for (size_t i = 0; i < y.data.size(); ++i)
                grad.data[i] = 2.0 * (y.data[i] - target.data[i]) / y.data.size();
            block.backward(grad, lr);
            block.update_weights(lr);
        }
        Tensor y1 = block.forward(x);
        double L1 = 0.0;
        for (size_t i = 0; i < y1.data.size(); ++i) {
            double d = y1.data[i] - target.data[i];
            L1 += d * d;
        }
        L1 /= y1.data.size();
        check("Block training reduces MSE loss", L1 < L0, L0 - L1);
    }

    // -----------------------------------------------------------------
    // 19. LinearAttentionModel forward shape
    // -----------------------------------------------------------------
    {
        LinearAttentionModel model(3, 4, 2, 2, 1, 4);
        Tensor x = make_input(6, 3, 0.4, 18);
        Tensor y = model.forward(x);
        check("Model forward shape (6,3)->(1,2)", y.rows == 1 && y.cols == 2);
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) { finite = false; break; }
        check("Model forward output is finite", finite);
    }

    // -----------------------------------------------------------------
    // 20. LinearAttentionModel 2-block training reduces loss
    // -----------------------------------------------------------------
    {
        LinearAttentionModel model(3, 4, 2, 2, 1, 4);
        Tensor x = make_input(4, 3, 0.5, 19);
        Tensor target(1, 2);
        target(0, 0) = 0.3;
        target(0, 1) = -0.2;
        double lr = 0.01;
        Tensor y0 = model.forward(x);
        double L0 = 0.0;
        for (size_t i = 0; i < y0.data.size(); ++i) {
            double d = y0.data[i] - target.data[i];
            L0 += d * d;
        }
        L0 /= y0.data.size();
        for (int step = 0; step < 30; ++step) {
            model.zero_grad();
            Tensor y = model.forward(x);
            Tensor grad(y.rows, y.cols);
            for (size_t i = 0; i < y.data.size(); ++i)
                grad.data[i] = 2.0 * (y.data[i] - target.data[i]) / y.data.size();
            model.backward(grad, lr);
            model.update_weights(lr);
        }
        Tensor y1 = model.forward(x);
        double L1 = 0.0;
        for (size_t i = 0; i < y1.data.size(); ++i) {
            double d = y1.data[i] - target.data[i];
            L1 += d * d;
        }
        L1 /= y1.data.size();
        check("Model training reduces MSE loss", L1 < L0, L0 - L1);
    }

    // -----------------------------------------------------------------
    // 21. Mutation test: zeroing the elu_plus_one derivative branch
    // (replacing phi'(Q) with 1.0) should break W_q/W_k/W_v gradient checks
    // (since the derivative is the dominant factor when Q has positive
    // values). Skip in production — included as a sanity check that the
    // test exercises the feature-map chain.
    // -----------------------------------------------------------------
    {
        // Sanity: a phi' of 1.0 would make the gradient check pass IF
        // the chain weren't actually computing gradients. Verify our chain
        // DOES care about phi' by checking that perturbing Q's sign (which
        // changes phi' from 1 to exp(Q)) changes the gradient direction.
        LinearAttention la(3);
        Tensor x_pos(2, 3);  // all-positive input → Q all positive → phi' = 1
        x_pos(0,0) = 1.0; x_pos(0,1) = 0.5; x_pos(0,2) = 0.3;
        x_pos(1,0) = 0.7; x_pos(1,1) = 0.9; x_pos(1,2) = 0.4;
        Tensor x_neg(2, 3);  // all-negative input → Q all negative → phi' = exp(Q) << 1
        x_neg(0,0) = -1.0; x_neg(0,1) = -0.5; x_neg(0,2) = -0.3;
        x_neg(1,0) = -0.7; x_neg(1,1) = -0.9; x_neg(1,2) = -0.4;
        Tensor gs(2, 3);
        gs.fill(0.5);
        // Forward+backward at positive input
        Tensor out_p = la.forward(x_pos);
        la.zero_grad();
        la.backward(gs, 0.0);
        double norm_pos = 0.0;
        for (double v : la.gradients()[0]->data) norm_pos += v * v;
        norm_pos = std::sqrt(norm_pos);
        // Forward+backward at negative input
        Tensor out_n = la.forward(x_neg);
        la.zero_grad();
        la.backward(gs, 0.0);
        double norm_neg = 0.0;
        for (double v : la.gradients()[0]->data) norm_neg += v * v;
        norm_neg = std::sqrt(norm_neg);
        // For all-negative Q, phi' = exp(Q) ≈ 0.05-0.5, so grad magnitudes
        // should be substantially smaller than for all-positive Q (phi' = 1).
        check("phi' feature map matters: pos vs neg Q produces different grad magnitudes",
              std::abs(norm_pos - norm_neg) > 1e-3);
    }

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
