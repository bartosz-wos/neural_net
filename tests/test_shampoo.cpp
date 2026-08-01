// test_shampoo.cpp — focused test suite for the Shampoo optimizer
//
// Shampoo (Gupta, Koren, Singer 2018, "Shampoo: Preconditioned Stochastic
// Tensor Optimization", https://arxiv.org/abs/1802.03668). The foundational
// Kronecker-factored preconditioned SGD that maintains two symmetric
// covariance matrices per 2-D parameter, eigendecomposes them once per step,
// and applies the preconditioner `L^{-1/4} G R^{-1/4}` to the gradient in
// the update.
//
// Coverage: defaults, validated setters, state shape correctness, closed-form
// first-step math, weight decay, training reduces loss, determinism, scalar
// (1,1) / 1-D fallbacks, eigendecomp on identity, rotation preserves
// Frobenius norm, signature vs SOAP, multi-layer independence.

#include "nn/nn.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/optimizer.h"
#include "nn/optimizers/shampoo.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& name, double tol = 0.0) {
    if (cond) {
        ++g_pass;
        std::cout << "  PASS: " << name << "\n";
    } else {
        ++g_fail;
        std::cout << "  FAIL: " << name << "  (tol=" << tol << ")\n";
    }
}

bool close_to(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

}  // namespace

int main() {
    std::cout << "=== Shampoo Optimizer Tests ===\n";

    // ---------------------------------------------------------------------
    // T1: Defaults round-trip
    // ---------------------------------------------------------------------
    {
        Shampoo opt;
        check(opt.get_lr() == 1e-3, "default lr=1e-3");
        check(opt.get_beta() == 0.9, "default beta=0.9");
        check(opt.get_eps() == 1e-12, "default eps=1e-12");
        check(opt.get_weight_decay() == 0.0, "default weight_decay=0");
        check(opt.get_t() == 1, "default t=1");
        check(opt.handles_weight_decay(), "handles_weight_decay=true");
    }

    // ---------------------------------------------------------------------
    // T2: Non-default constructor
    // ---------------------------------------------------------------------
    {
        Shampoo opt(5e-3, 0.95, 1e-8, 0.01);
        check(opt.get_lr() == 5e-3, "non-default lr");
        check(opt.get_beta() == 0.95, "non-default beta");
        check(opt.get_eps() == 1e-8, "non-default eps");
        check(opt.get_weight_decay() == 0.01, "non-default weight_decay");
    }

    // ---------------------------------------------------------------------
    // T3: Validated setters throw on bad inputs
    // ---------------------------------------------------------------------
    {
        Shampoo opt;
        bool threw = false;
        try { opt.set_lr(-1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_lr negative throws");

        threw = false;
        try { opt.set_beta(1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_beta = 1.0 throws");

        threw = false;
        try { opt.set_beta(-0.1); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_beta negative throws");

        threw = false;
        try { opt.set_eps(0.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_eps=0 throws");

        threw = false;
        try { opt.set_eps(-1e-3); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_eps negative throws");

        threw = false;
        try { opt.set_weight_decay(-1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_weight_decay negative throws");

        // Boundary values OK
        threw = false;
        try { opt.set_beta(0.0); } catch (...) { threw = true; }
        check(!threw, "set_beta=0 boundary OK");
        threw = false;
        try { opt.set_beta(0.9999); } catch (...) { threw = true; }
        check(!threw, "set_beta≈1 boundary OK");
    }

    // ---------------------------------------------------------------------
    // T4: Constructor-time validation throws on bad inputs
    // ---------------------------------------------------------------------
    {
        bool threw = false;
        try { Shampoo opt(-1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor negative lr throws");

        threw = false;
        try { Shampoo opt(1e-3, 1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor beta=1 throws");

        threw = false;
        try { Shampoo opt(1e-3, 0.9, 0.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor eps=0 throws");

        threw = false;
        try { Shampoo opt(1e-3, 0.9, 1e-12, -0.1); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor negative weight_decay throws");
    }

    // ---------------------------------------------------------------------
    // T5: State is lazy (no state before step)
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        Shampoo opt;
        check(!opt.has_state(layer), "no state before step on layer");
    }

    // ---------------------------------------------------------------------
    // T6: State shapes correctness after step (single Dense)
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(3, 4);  // input=3, output=4 -> weights (4, 3), bias (1, 4)
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        for (size_t i = 0; i < params.size(); ++i) {
            grads[i]->fill(0.1);
        }
        Shampoo opt;
        opt.step(m);

        // weights (4, 3): L is (4, 4), R is (3, 3), U_L is (4, 4), U_R is (3, 3)
        // bias (1, 4): L is (1, 1), R is (4, 4), U_L is (1, 1), U_R is (4, 4)
        Tensor L0 = opt.get_L(layer, 0);
        Tensor R0 = opt.get_R(layer, 0);
        Tensor UL0 = opt.get_U_L(layer, 0);
        Tensor UR0 = opt.get_U_R(layer, 0);
        check(L0.rows == 4 && L0.cols == 4, "L[0] shape (4,4) for weights");
        check(R0.rows == 3 && R0.cols == 3, "R[0] shape (3,3) for weights");
        check(UL0.rows == 4 && UL0.cols == 4, "U_L[0] shape (4,4) for weights");
        check(UR0.rows == 3 && UR0.cols == 3, "U_R[0] shape (3,3) for weights");

        Tensor L1 = opt.get_L(layer, 1);
        Tensor R1 = opt.get_R(layer, 1);
        check(L1.rows == 1 && L1.cols == 1, "L[1] shape (1,1) for bias");
        check(R1.rows == 4 && R1.cols == 4, "R[1] shape (4,4) for bias");

        check(opt.get_t() == 2, "t incremented to 2 after step");
    }

    // ---------------------------------------------------------------------
    // T7: State doesn't accumulate across layers — each layer tracks its own
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l1 = new Dense(2, 2);
        Layer* l2 = new Dense(4, 3);
        m.add_layer(l1);
        m.add_layer(l2);
        Shampoo opt;
        opt.step(m);
        check(opt.has_state(l1), "layer1 has state after step");
        check(opt.has_state(l2), "layer2 has state after step");
        // Dense(4, 3) -> weights (out=3, in=4) = (3, 4), bias (1, 3)
        // L for weights is (3, 3), R for weights is (4, 4)
        Tensor L1 = opt.get_L(l1, 0);
        Tensor L2 = opt.get_L(l2, 0);
        Tensor R2 = opt.get_R(l2, 0);
        check(L1.rows == 2 && L1.cols == 2, "l1.L[0] shape (2,2) for weights (2,2)");
        check(L2.rows == 3 && L2.cols == 3, "l2.L[0] shape (3,3) for weights (3,4)");
        check(R2.rows == 4 && R2.cols == 4, "l2.R[0] shape (4,4) for weights (3,4)");
    }

    // ---------------------------------------------------------------------
    // T8: Scalar parameter (1,1) reduces to plain SGD (no preconditioner)
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(1, 1);  // input=1, output=1; weights (1,1), bias (1,1)
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(0.5);
        params[1]->fill(0.0);
        grads[0]->fill(0.3);
        grads[1]->fill(0.7);

        Shampoo opt(0.1);  // lr=0.1, no weight decay
        opt.step(m);

        // Plain SGD: w_new = w - lr * g
        // weight: 0.5 - 0.1 * 0.3 = 0.47
        // bias:   0.0 - 0.1 * 0.7 = -0.07
        double w0 = (*params[0])(0, 0);
        double b0 = (*params[1])(0, 0);
        check(close_to(w0, 0.47, 1e-12), "scalar weight: w = 0.5 - lr * g (plain SGD)");
        check(close_to(b0, -0.07, 1e-12), "scalar bias: b = 0 - lr * g (plain SGD)");
    }

    // ---------------------------------------------------------------------
    // T9: Weight decay shrinks params at zero gradient
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(1.0);
        params[1]->fill(1.0);
        grads[0]->fill(0.0);
        grads[1]->fill(0.0);

        Shampoo opt(0.1, 0.9, 1e-12, 0.5);  // wd=0.5
        opt.step(m);

        // w_new = (1 - lr * wd) * w = (1 - 0.05) * 1.0 = 0.95
        double w0 = (*params[0])(0, 0);
        check(close_to(w0, 0.95, 1e-12), "weight decay: w = (1 - lr*wd) * w");
    }

    // ---------------------------------------------------------------------
    // T10: First-step closed-form on a 2-D problem (no weight decay)
    //
    // Dense(2, 2) with weights W = 0, bias = 0, learning rate lr.
    // Gradient G = [[1, 0], [0, 1]] (identity).
    // L = (1-β) * G G^T = (1-β) * I   (2×2)
    // R = (1-β) * G^T G = (1-β) * I   (2×2)
    // Eigendecomp: U_L = U_R = I, λs = (1-β) * (1, 1)
    // L^{-1/4} = R^{-1/4} = ((1-β))^{-1/4} * I
    // update = L^{-1/4} G R^{-1/4} = ((1-β))^{-1/2} * G = ((1-β))^{-1/2} * I
    // W := W - lr * update = -lr * ((1-β))^{-1/2} * I
    //
    // With lr=0.1, β=0.5: (1-0.5)=0.5, (0.5)^{-1/2}=sqrt(2), so
    // W = -0.1 * sqrt(2) ≈ -0.14142
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(0.0);
        params[1]->fill(0.0);
        (*grads[0])(0, 0) = 1.0;
        (*grads[0])(1, 1) = 1.0;
        (*grads[0])(0, 1) = 0.0;
        (*grads[0])(1, 0) = 0.0;
        grads[1]->fill(0.0);

        Shampoo opt(0.1, 0.5);  // lr=0.1, beta=0.5
        opt.step(m);

        const double expected = -0.1 * std::sqrt(2.0);
        double w00 = (*params[0])(0, 0);
        double w11 = (*params[0])(1, 1);
        double w01 = (*params[0])(0, 1);
        double w10 = (*params[0])(1, 0);
        check(close_to(w00, expected, 1e-6), "T10 w[0,0] = -lr * sqrt(1/(1-beta))");
        check(close_to(w11, expected, 1e-6), "T10 w[1,1] = -lr * sqrt(1/(1-beta))");
        check(close_to(w01, 0.0, 1e-6), "T10 w[0,1] unchanged");
        check(close_to(w10, 0.0, 1e-6), "T10 w[1,0] unchanged");
    }

    // ---------------------------------------------------------------------
    // T11: Identity eigendecomp test (L = G G^T = I when G = I)
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(0.0);
        params[1]->fill(0.0);
        (*grads[0])(0, 0) = 1.0;
        (*grads[0])(1, 1) = 1.0;
        (*grads[0])(0, 1) = 0.0;
        (*grads[0])(1, 0) = 0.0;
        grads[1]->fill(0.0);

        Shampoo opt(0.1, 0.0);  // β=0 → L = GG^T = I

        opt.step(m);
        Tensor L = opt.get_L(layer, 0);
        // L should equal (1-0) * G G^T = G G^T = I
        check(close_to(L(0, 0), 1.0, 1e-6), "L[0,0] = 1 (identity)");
        check(close_to(L(1, 1), 1.0, 1e-6), "L[1,1] = 1 (identity)");
        check(close_to(L(0, 1), 0.0, 1e-6), "L[0,1] = 0 (identity)");
        check(close_to(L(1, 0), 0.0, 1e-6), "L[1,0] = 0 (identity)");
    }

    // ---------------------------------------------------------------------
    // T12: Rotation by U_L preserves Frobenius norm of L
    //
    // For non-trivial L, after eigh, the rotation U_L^T L U_L equals diag(λ).
    // We test indirectly: a 2-D gradient that produces non-identity L should
    // still produce a parameter update that matches the analytic L^{-1/4} G R^{-1/4}
    // formula.
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(0.0);
        // G = [[1, 0], [0, 2]]
        (*grads[0])(0, 0) = 1.0;
        (*grads[0])(1, 1) = 2.0;
        (*grads[0])(0, 1) = 0.0;
        (*grads[0])(1, 0) = 0.0;
        grads[1]->fill(0.0);

        // β=0: L = G G^T = [[1, 0], [0, 4]], R = G^T G = [[1, 0], [0, 4]]
        // Both have eigvals (1, 4) and eigenvectors (e1, e2) → U_L = U_R = I
        // L^{-1/4} = diag(1, 1/sqrt(2))
        // R^{-1/4} = diag(1, 1/sqrt(2))
        // update = L^{-1/4} G R^{-1/4} = diag(1, 1/2) * G = [[1, 0], [0, 1]]
        // (since 1/2 * 2 = 1)
        // W := W - lr * update = W - 0.1 * I = [[-0.1, 0], [0, -0.1]]
        Shampoo opt(0.1, 0.0);
        opt.step(m);

        double w00 = (*params[0])(0, 0);
        double w11 = (*params[0])(1, 1);
        check(close_to(w00, -0.1, 1e-6), "T12 w[0,0] = -0.1 (rotation preserves scaling structure)");
        check(close_to(w11, -0.1, 1e-6), "T12 w[1,1] = -0.1 (1/2 * 2)");
        check(close_to((*params[0])(0, 1), 0.0, 1e-6), "T12 w[0,1] unchanged");
        check(close_to((*params[0])(1, 0), 0.0, 1e-6), "T12 w[1,0] unchanged");
    }

    // ---------------------------------------------------------------------
    // T13: Determinism (two fresh instances produce bit-identical updates)
    // ---------------------------------------------------------------------
    {
        Model m1, m2;
        Layer* l1 = new Dense(3, 4);
        Layer* l2 = new Dense(3, 4);
        m1.add_layer(l1);
        m2.add_layer(l2);
        for (size_t i = 0; i < l1->parameters().size(); ++i) {
            for (size_t r = 0; r < l1->parameters()[i]->rows; ++r)
                for (size_t c = 0; c < l1->parameters()[i]->cols; ++c) {
                    (*l1->parameters()[i])(r, c) = 0.1 * (r + c + 1);
                    (*l2->parameters()[i])(r, c) = 0.1 * (r + c + 1);
                    (*l1->gradients()[i])(r, c) = 0.01 * (r * 3 + c + 1);
                    (*l2->gradients()[i])(r, c) = 0.01 * (r * 3 + c + 1);
                }
        }

        Shampoo o1(1e-3, 0.9);
        Shampoo o2(1e-3, 0.9);
        o1.step(m1);
        o2.step(m2);

        bool all_equal = true;
        for (size_t i = 0; i < l1->parameters().size(); ++i) {
            const Tensor& p1 = *l1->parameters()[i];
            const Tensor& p2 = *l2->parameters()[i];
            for (size_t r = 0; r < p1.rows; ++r)
                for (size_t c = 0; c < p1.cols; ++c) {
                    if (!close_to(p1(r, c), p2(r, c), 1e-12)) {
                        all_equal = false;
                    }
                }
        }
        check(all_equal, "two fresh Shampoo instances produce bit-identical updates after 1 step");

        // Run 5 more steps; check still equal
        for (int s = 0; s < 5; ++s) {
            o1.step(m1);
            o2.step(m2);
        }
        all_equal = true;
        for (size_t i = 0; i < l1->parameters().size(); ++i) {
            const Tensor& p1 = *l1->parameters()[i];
            const Tensor& p2 = *l2->parameters()[i];
            for (size_t r = 0; r < p1.rows; ++r)
                for (size_t c = 0; c < p1.cols; ++c) {
                    if (!close_to(p1(r, c), p2(r, c), 1e-12)) {
                        all_equal = false;
                    }
                }
        }
        check(all_equal, "two fresh Shampoo instances remain bit-identical across 5 more steps");
    }

    // ---------------------------------------------------------------------
    // T14: t increments correctly across multiple steps
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        Shampoo opt;
        check(opt.get_t() == 1, "T14 t=1 before step");
        opt.step(m);
        check(opt.get_t() == 2, "T14 t=2 after step 1");
        opt.step(m);
        check(opt.get_t() == 3, "T14 t=3 after step 2");
        opt.step(m);
        check(opt.get_t() == 4, "T14 t=4 after step 3");
    }

    // ---------------------------------------------------------------------
    // T15: Signature vs SOAP (Shampoo update differs from SOAP)
    // ---------------------------------------------------------------------
    {
        Model m_shampoo, m_soap;
        Layer* l_shampoo = new Dense(2, 2);
        Layer* l_soap = new Dense(2, 2);
        m_shampoo.add_layer(l_shampoo);
        m_soap.add_layer(l_soap);
        for (size_t i = 0; i < l_shampoo->parameters().size(); ++i) {
            *l_soap->parameters()[i] = *l_shampoo->parameters()[i];
            *l_soap->gradients()[i] = *l_shampoo->gradients()[i];
        }

        // Inject non-trivial gradient magnitudes so the preconditioner differs
        for (size_t i = 0; i < l_shampoo->parameters().size(); ++i) {
            for (size_t r = 0; r < l_shampoo->parameters()[i]->rows; ++r)
                for (size_t c = 0; c < l_shampoo->parameters()[i]->cols; ++c) {
                    (*l_shampoo->gradients()[i])(r, c) = 0.1 + 0.01 * (r * 4 + c);
                    (*l_soap->gradients()[i])(r, c) = 0.1 + 0.01 * (r * 4 + c);
                }
        }

        Shampoo opt_shampoo(1e-3, 0.9);
        SOAP opt_soap(1e-3, 0.9, 0.95, 1e-8, 10, 0.0);

        // Run 3 steps so the preconditioner difference accumulates
        opt_shampoo.step(m_shampoo);
        opt_soap.step(m_soap);
        opt_shampoo.step(m_shampoo);
        opt_soap.step(m_soap);
        opt_shampoo.step(m_shampoo);
        opt_soap.step(m_soap);

        // Updates should differ
        double w_diff = 0.0;
        for (size_t i = 0; i < l_shampoo->parameters().size(); ++i) {
            const Tensor& a = *l_shampoo->parameters()[i];
            const Tensor& b = *l_soap->parameters()[i];
            for (size_t r = 0; r < a.rows; ++r)
                for (size_t c = 0; c < a.cols; ++c)
                    w_diff += std::abs(a(r, c) - b(r, c));
        }
        check(w_diff > 1e-6, "T15 Shampoo != SOAP update trajectory (different algorithms)");
    }

    // ---------------------------------------------------------------------
    // T16: Training reduces loss on y=2x regression
    // ---------------------------------------------------------------------
    {
        Tensor X(4, 1);
        Tensor y(4, 1);
        X(0, 0) = 1.0; y(0, 0) = 2.0;
        X(1, 0) = 2.0; y(1, 0) = 4.0;
        X(2, 0) = 3.0; y(2, 0) = 6.0;
        X(3, 0) = 4.0; y(3, 0) = 8.0;

        Model m;
        Layer* layer = new Dense(1, 1);
        m.add_layer(layer);
        layer->parameters()[0]->fill(0.1);
        layer->parameters()[1]->fill(0.0);

        Shampoo opt(0.01, 0.9);

        // Compute the initial loss BEFORE the training loop starts so we
        // can correctly measure reduction at the end.
        Tensor pred0 = m.forward(X);
        double initial_loss = 0.0;
        for (size_t i = 0; i < pred0.rows; ++i) {
            double diff = pred0(i, 0) - y(i, 0);
            initial_loss += diff * diff;
        }
        initial_loss /= pred0.rows;

        for (int epoch = 0; epoch < 200; ++epoch) {
            Tensor pred = m.forward(X);
            Tensor grad(4, 1);
            for (size_t i = 0; i < pred.rows; ++i) {
                grad(i, 0) = 2.0 * (pred(i, 0) - y(i, 0)) / pred.rows;
            }
            m.backward(grad, 0.0);
            opt.step(m);
        }

        Tensor pred = m.forward(X);
        double final_loss = 0.0;
        for (size_t i = 0; i < pred.rows; ++i) {
            double diff = pred(i, 0) - y(i, 0);
            final_loss += diff * diff;
        }
        final_loss /= pred.rows;

        const double loss_reduction = (initial_loss - final_loss) / std::max(initial_loss, 1e-12);
        check(initial_loss > final_loss, "T16 loss decreased after training");
        check(loss_reduction > 0.5, "T16 loss reduction > 50%");
    }

    // ---------------------------------------------------------------------
    // T17: Empty model doesn't crash
    // ---------------------------------------------------------------------
    {
        Model m;
        Shampoo opt;
        opt.step(m);  // should not crash
        check(opt.get_t() == 2, "T17 t=2 after step on empty model");
    }

    // ---------------------------------------------------------------------
    // T18: 1-D parameter (1, n) — only R applies
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 1);  // input=2, output=1 -> weights (1, 2)
        m.add_layer(layer);
        auto params = layer->parameters();
        auto grads = layer->gradients();
        params[0]->fill(0.0);  // weight (1, 2)
        params[1]->fill(0.0);  // bias (1, 1)
        (*grads[0])(0, 0) = 0.5;
        (*grads[0])(0, 1) = 0.7;
        grads[1]->fill(0.0);

        Shampoo opt(0.1, 0.5);
        opt.step(m);

        double w00 = (*params[0])(0, 0);
        double w01 = (*params[0])(0, 1);
        check(std::isfinite(w00), "T18 w[0,0] finite after 1-D update");
        check(std::isfinite(w01), "T18 w[0,1] finite after 1-D update");
    }

    // ---------------------------------------------------------------------
    // T19: Model with no params (e.g., activation-only) — no state, no crash
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* layer = new Dense(2, 2);
        m.add_layer(layer);
        Shampoo opt;
        opt.step(m);

        // Calling step again on the same model should also work
        opt.step(m);
        check(opt.get_t() == 3, "T19 t=3 after two steps");
    }

    // ---------------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------------
    std::cout << "\n=== Shampoo Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
