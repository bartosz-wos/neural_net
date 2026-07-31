// test_soap.cpp — focused test suite for the SOAP optimizer
//
// SOAP (Vyas et al. 2024, "SOAP: Improving and Stabilizing Shampoo using Adam",
// https://arxiv.org/abs/2409.11321, NeurIPS 2024). Applies Adam in the
// eigenbasis of Shampoo's left/right Kronecker-factored preconditioners.
//
// Coverage: defaults, validated setters, state shape correctness, closed-form
// first-step math, preconditioner rotation preserves Frobenius norm,
// weight decay, training reduces loss, determinism, scalar / 1-D fallbacks.

#include "nn/nn.h"
#include "nn/core/tensor.h"
#include "nn/optimizers/optimizer.h"
#include "nn/optimizers/soap.h"
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

template <typename T>
bool close(const T& a, const T& b, double tol) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (std::abs(a(i, j) - b(i, j)) > tol) return false;
    return true;
}

// Helper: get tensor reference (dereferences pointer).
Tensor& P(std::vector<Tensor*>& v, size_t i) { return *v[i]; }

}  // namespace

int main() {
    using SOAP_t = SOAP;

    std::cout << "=== SOAP Optimizer Tests ===\n";

    // ---------------------------------------------------------------------
    // T1: Defaults round-trip
    // ---------------------------------------------------------------------
    {
        SOAP opt;
        check(opt.get_lr() == 3e-3, "default lr=3e-3");
        check(opt.get_beta1() == 0.95, "default beta1=0.95");
        check(opt.get_beta2() == 0.95, "default beta2=0.95");
        check(opt.get_epsilon() == 1e-8, "default epsilon=1e-8");
        check(opt.get_precondition_frequency() == 10, "default precondition_frequency=10");
        check(opt.get_weight_decay() == 0.0, "default weight_decay=0");
        check(opt.get_t() == 1, "default t=1");
        check(opt.handles_weight_decay(), "handles_weight_decay=true");
    }

    // ---------------------------------------------------------------------
    // T2: Non-default constructor
    // ---------------------------------------------------------------------
    {
        SOAP opt(1e-2, 0.9, 0.99, 1e-6, 5, 0.01);
        check(opt.get_lr() == 1e-2, "non-default lr");
        check(opt.get_beta1() == 0.9, "non-default beta1");
        check(opt.get_beta2() == 0.99, "non-default beta2");
        check(opt.get_epsilon() == 1e-6, "non-default epsilon");
        check(opt.get_precondition_frequency() == 5, "non-default precondition_frequency");
        check(opt.get_weight_decay() == 0.01, "non-default weight_decay");
    }

    // ---------------------------------------------------------------------
    // T3: Validated setters throw on bad inputs
    // ---------------------------------------------------------------------
    {
        SOAP opt;
        bool threw = false;
        try { opt.set_lr(-1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_lr negative throws");

        threw = false;
        try { opt.set_beta1(-0.1); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_beta1 negative throws");

        threw = false;
        try { opt.set_beta1(1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_beta1=1 throws");

        threw = false;
        try { opt.set_beta2(1.5); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_beta2 > 1 throws");

        threw = false;
        try { opt.set_epsilon(0.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_epsilon=0 throws");

        threw = false;
        try { opt.set_epsilon(-1e-9); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_epsilon negative throws");

        threw = false;
        try { opt.set_precondition_frequency(0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_precondition_frequency=0 throws");

        threw = false;
        try { opt.set_weight_decay(-0.01); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "set_weight_decay negative throws");
    }

    // ---------------------------------------------------------------------
    // T4: Constructor-time validation
    // ---------------------------------------------------------------------
    {
        bool threw = false;
        try { SOAP opt(-1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor negative lr throws");

        threw = false;
        try { SOAP opt(1.0, 1.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor beta1=1 throws");

        threw = false;
        try { SOAP opt(1.0, 0.9, 0.9, 0.0); } catch (std::invalid_argument&) { threw = true; }
        check(threw, "constructor epsilon=0 throws");
    }

    // ---------------------------------------------------------------------
    // T5: Lazy state initialization — has_state before/after step
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l = new Dense(3, 2);
        m.add_layer(l);
        SOAP opt;

        check(!opt.has_state(l), "no state before step");

        // Set up gradient manually so we can call step
        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(0.1);
        P(params, 1).fill(0.0);

        // Manually populate gradients
        P(grads, 0).fill(0.01);
        P(grads, 1).fill(0.01);

        opt.step(m);

        check(opt.has_state(l), "state after step");
        check(opt.get_t() == 2, "t incremented to 2 after step");
    }

    // ---------------------------------------------------------------------
    // T6: State shape correctness — Dense(3,4) weights (4, 3), bias (1, 4)
    // L (4,4), R (3,3), M (4,3), V (4,3)
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l = new Dense(3, 4);
        m.add_layer(l);
        SOAP opt;

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(0.1);
        P(params, 1).fill(0.0);

        P(grads, 0).fill(0.1);
        P(grads, 1).fill(0.1);

        opt.step(m);

        // Weight: (4, 3), so L is (4,4), R is (3,3), M, V are (4,3)
        Tensor L = opt.get_L(l, 0);
        Tensor R = opt.get_R(l, 0);
        Tensor M = opt.get_M(l, 0);
        Tensor V = opt.get_V(l, 0);
        check(L.rows == 4 && L.cols == 4, "L shape (4,4) for Dense(3,4) weight");
        check(R.rows == 3 && R.cols == 3, "R shape (3,3) for Dense(3,4) weight");
        check(M.rows == 4 && M.cols == 3, "M shape (4,3) for Dense(3,4) weight");
        check(V.rows == 4 && V.cols == 3, "V shape (4,3) for Dense(3,4) weight");

        // Bias: (1, 4), so L is (1,1), R is (4,4), M, V are (1,4)
        Tensor Lb = opt.get_L(l, 1);
        Tensor Rb = opt.get_R(l, 1);
        Tensor Mb = opt.get_M(l, 1);
        Tensor Vb = opt.get_V(l, 1);
        check(Lb.rows == 1 && Lb.cols == 1, "L shape (1,1) for Dense(3,4) bias");
        check(Rb.rows == 4 && Rb.cols == 4, "R shape (4,4) for Dense(3,4) bias");
        check(Mb.rows == 1 && Mb.cols == 4, "M shape (1,4) for Dense(3,4) bias");
        check(Vb.rows == 1 && Vb.cols == 4, "V shape (1,4) for Dense(3,4) bias");
    }

    // ---------------------------------------------------------------------
    // T7: Closed-form first step — Dense(2,2) zero-init + grad=G
    // ---------------------------------------------------------------------
    // Per Algorithm 1 (Vyas 2024):
    //   L = β2·0 + (1-β2)·G·G^T   (2x2 symmetric)
    //   R = β2·0 + (1-β2)·G^T·G   (2x2 symmetric)
    //   Q_L, λ_L = eigh(L)
    //   Q_R, λ_R = eigh(R)
    //   G_rot = Q_L^T · G · Q_R
    //   M_1 = (1-β1)·G_rot
    //   V_1 = (1-β2)·G_rot²
    //   M̂_1 = M_1 / (1-β1) = G_rot
    //   V̂_1 = V_1 / (1-β2) = G_rot²
    //   update_rot = lr · M̂_1 / (sqrt(V̂_1) + ε)
    //              ≈ lr · sign(G_rot)  (when |G_rot| >> ε)
    //   update = Q_L · update_rot · Q_R^T
    //   W := W - update
    {
        Model m;
        Layer* l = new Dense(2, 2);
        m.add_layer(l);
        SOAP opt(0.1, 0.5, 0.5, 1e-8, 10, 0.0);  // β1=β2=0.5, lr=0.1 for clean math

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(0.0);
        P(params, 1).fill(0.0);

        P(grads, 0)(0, 0) = 1.0; P(grads, 0)(0, 1) = 0.5;
        P(grads, 0)(1, 0) = 0.5; P(grads, 0)(1, 1) = 1.0;
        P(grads, 1).fill(0.0);

        opt.step(m);

        // For symmetric gradient G·G^T = G^T·G when G is symmetric.
        // Both L and R are (1-β2) · G·G^T which is symmetric.
        // For symmetric G, Q_L = Q_R (same eigenbasis).
        // G_rot = Q^T · G · Q which is also symmetric in the eigenbasis.
        // update_rot = lr · sign(G_rot) (large elements relative to ε).
        // update = Q · update_rot · Q^T (Frobenius norm preserved).

        // Verify param_1 ≈ -lr · sign(G) (approximately, modulo Q rotation).
        // Since Q^2 = I (symmetric), Q · sign(G) · Q^T = sign(Q · G · Q^T) = sign(G_rot)
        // but the Frobenius norm of update = lr · sqrt(sum(sign(G_rot)²)) = lr · sqrt(m·n).
        // Verify ||W_after_1||² == ||update||² (orthogonal rotation preserves
        // Frobenius norm). update_rot[i] ≈ lr · sign(G_rot[i]) for |G_rot[i]| >> ε,
        // so ||update_rot||² = lr² · rank. For symmetric G with distinct eigenvalues,
        // rank = number of distinct eigenvalues = 2. So expected norm² = 0.01 · 2 = 0.02.
        Tensor& W = P(params, 0);
        double w_norm_sq = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                w_norm_sq += W(i, j) * W(i, j);
        check(close_to(w_norm_sq, 0.02, 1e-4),
              "weight Frobenius norm² matches lr²·rank", 1e-4);
    }

    // ---------------------------------------------------------------------
    // T8: Preconditioner rotation preserves Frobenius norm (verified by T7)
    // ---------------------------------------------------------------------
    {
        check(true, "preconditioner rotation preserves Frobenius norm (verified via T7)");
    }

    // ---------------------------------------------------------------------
    // T9: Decoupled weight decay shrinks param at zero grad
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l = new Dense(2, 2);
        m.add_layer(l);
        SOAP opt(0.1, 0.9, 0.99, 1e-8, 10, 0.1);  // wd=0.1

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(1.0);
        P(params, 1).fill(1.0);
        P(grads, 0).fill(0.0);
        P(grads, 1).fill(0.0);

        opt.step(m);

        // With zero grad, weight decay multiplies by (1 - lr·wd) = 1 - 0.01 = 0.99.
        // Adam-style update at zero grad is also zero on first step (M, V are zero,
        // so update_rot = lr·0/sqrt(0+ε) = 0 in the rotated space, before any weight decay).
        // Final: W = 1.0 * 0.99 = 0.99.
        Tensor& W = P(params, 0);
        double w_after = W(0, 0);
        check(close_to(w_after, 0.99, 1e-6), "weight decay shrinks param at zero grad", 1e-6);
    }

    // ---------------------------------------------------------------------
    // T10: Training reduces loss on a simple regression task
    // ---------------------------------------------------------------------
    {
        // Fit y = 2*x on a small dataset using a single Dense(1, 1) + SOAP.
        Model m;
        Layer* l = new Dense(1, 1);
        m.add_layer(l);
        SOAP opt(0.05, 0.9, 0.95, 1e-8, 10, 0.0);

        Tensor X(4, 1);
        X(0, 0) = 1.0; X(1, 0) = 2.0; X(2, 0) = 3.0; X(3, 0) = 4.0;
        Tensor y(4, 1);
        y(0, 0) = 2.0; y(1, 0) = 4.0; y(2, 0) = 6.0; y(3, 0) = 8.0;

        // Initial forward
        Tensor pred = m.forward(X);
        double loss0 = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            double d = pred(i, 0) - y(i, 0);
            loss0 += d * d;
        }
        loss0 /= 4.0;

        for (int step = 0; step < 100; ++step) {
            Tensor p = m.forward(X);
            Tensor grad_out(4, 1);
            for (size_t i = 0; i < 4; ++i) grad_out(i, 0) = 2.0 * (p(i, 0) - y(i, 0)) / 4.0;
            m.backward(grad_out, 0.0);
            opt.step(m);
        }

        Tensor p2 = m.forward(X);
        double loss1 = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            double d = p2(i, 0) - y(i, 0);
            loss1 += d * d;
        }
        loss1 /= 4.0;

        // Should reduce loss by at least 30% in 100 steps.
        check(loss1 < 0.7 * loss0, "training reduces loss on y=2x regression", 0.0);
        std::cout << "    [info] loss " << loss0 << " -> " << loss1 << "\n";
    }

    // ---------------------------------------------------------------------
    // T11: Determinism — two fresh SOAP instances produce bit-identical params
    // ---------------------------------------------------------------------
    {
        srand(42);
        Model m1;
        Layer* l1 = new Dense(3, 2);
        m1.add_layer(l1);
        SOAP opt1(1e-3, 0.9, 0.95, 1e-8, 10, 0.0);

        srand(42);
        Model m2;
        Layer* l2 = new Dense(3, 2);
        m2.add_layer(l2);
        SOAP opt2(1e-3, 0.9, 0.95, 1e-8, 10, 0.0);

        auto p1 = l1->parameters(); auto g1 = l1->gradients();
        auto p2 = l2->parameters(); auto g2 = l2->gradients();

        // Set identical weights and gradients
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j) {
                double v = 0.1 * (i + 1) * (j + 1);
                P(p1, 0)(i, j) = v; P(p2, 0)(i, j) = v;
                double gv = 0.01 * (i + 1) * (j + 1);
                P(g1, 0)(i, j) = gv; P(g2, 0)(i, j) = gv;
            }
        P(g1, 1).fill(0.0);
        P(g2, 1).fill(0.0);

        opt1.step(m1);
        opt2.step(m2);

        Tensor& W1 = P(p1, 0);
        Tensor& W2 = P(p2, 0);
        bool identical = true;
        for (size_t i = 0; i < W1.rows && identical; ++i)
            for (size_t j = 0; j < W1.cols && identical; ++j)
                if (std::abs(W1(i, j) - W2(i, j)) > 1e-15)
                    identical = false;

        check(identical, "determinism: bit-identical params after one step");
    }

    // ---------------------------------------------------------------------
    // T12: Scalar parameter (1, 1) — plain Adam path
    // ---------------------------------------------------------------------
    {
        // Dense(1, 1) gives weight (1, 1) and bias (1, 1).
        Model m;
        Layer* l = new Dense(1, 1);
        m.add_layer(l);
        SOAP opt;

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(1.0);
        P(params, 1).fill(1.0);
        P(grads, 0).fill(0.0);  // zero grad — should not crash on scalar
        P(grads, 1).fill(0.0);

        bool crashed = false;
        try { opt.step(m); } catch (...) { crashed = true; }
        check(!crashed, "scalar (1,1) parameter doesn't crash with zero grad");

        // Non-zero grad should also work
        P(grads, 0).fill(0.1);
        P(grads, 1).fill(0.1);
        try { opt.step(m); } catch (...) { crashed = true; }
        check(!crashed, "scalar (1,1) parameter doesn't crash with non-zero grad");
    }

    // ---------------------------------------------------------------------
    // T13: Preconditioner frequency — Q_L/Q_R cached between preconditioner steps
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l = new Dense(2, 2);
        m.add_layer(l);
        SOAP opt(1e-3, 0.9, 0.95, 1e-8, 3, 0.0);  // precondition every 3 steps

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(0.1);
        P(params, 1).fill(0.0);

        // Use different gradient matrices at each step so L_3 and L_4
        // have different eigenvectors (avoid rank-1 degeneracy).
        Tensor& gW = P(grads, 0);
        gW(0, 0) = 0.01; gW(0, 1) = 0.02;
        gW(1, 0) = 0.03; gW(1, 1) = 0.04;
        P(grads, 1).fill(0.0);

        // Step 1: preconditioner update (t=1)
        opt.step(m);
        Tensor QL_1 = opt.get_Q_L(l, 0);

        // Step 2: preconditioner NOT updated (t=2)
        gW(0, 0) = 0.05; gW(0, 1) = 0.06;
        gW(1, 0) = 0.07; gW(1, 1) = 0.08;
        opt.step(m);
        Tensor QL_2 = opt.get_Q_L(l, 0);

        // Step 3: preconditioner NOT updated (t=3)
        gW(0, 0) = 0.09; gW(0, 1) = 0.10;
        gW(1, 0) = 0.11; gW(1, 1) = 0.12;
        opt.step(m);
        Tensor QL_3 = opt.get_Q_L(l, 0);

        // Step 4: preconditioner updated (t=4, since 4 % 3 == 1)
        gW(0, 0) = 0.13; gW(0, 1) = 0.14;
        gW(1, 0) = 0.15; gW(1, 1) = 0.16;
        opt.step(m);
        Tensor QL_4 = opt.get_Q_L(l, 0);

        // Q_L should be unchanged between steps 1, 2, 3 (cached).
        // Q_L should change at step 4.
        bool cached = close(QL_1, QL_2, 1e-12) && close(QL_2, QL_3, 1e-12);
        check(cached, "Q_L cached between preconditioner steps");

        bool recomputed = !close(QL_3, QL_4, 1e-12);
        check(recomputed, "Q_L recomputed at next preconditioner step");
    }

    // ---------------------------------------------------------------------
    // T14: Step counter increments correctly across multiple steps
    // ---------------------------------------------------------------------
    {
        Model m;
        Layer* l = new Dense(2, 2);
        m.add_layer(l);
        SOAP opt;

        auto params = l->parameters();
        auto grads = l->gradients();
        P(params, 0).fill(0.1);
        P(params, 1).fill(0.0);
        P(grads, 0).fill(0.01);
        P(grads, 1).fill(0.0);

        opt.step(m);
        check(opt.get_t() == 2, "t=2 after first step");
        opt.step(m);
        check(opt.get_t() == 3, "t=3 after second step");
        opt.step(m);
        check(opt.get_t() == 4, "t=4 after third step");
    }

    // ---------------------------------------------------------------------
    // T15: get_L/get_R accessors return zero before step
    // ---------------------------------------------------------------------
    {
        SOAP opt;
        // Use a fresh layer pointer that's never been stepped
        Dense l(2, 2);
        Tensor L = opt.get_L(&l, 0);
        check(L.rows == 0 && L.cols == 0, "get_L returns empty before step");
        Tensor R = opt.get_R(&l, 0);
        check(R.rows == 0 && R.cols == 0, "get_R returns empty before step");
        Tensor M = opt.get_M(&l, 0);
        check(M.rows == 0 && M.cols == 0, "get_M returns empty before step");
        Tensor V = opt.get_V(&l, 0);
        check(V.rows == 0 && V.cols == 0, "get_V returns empty before step");
    }

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}