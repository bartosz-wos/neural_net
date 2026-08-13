// ============================================================================
// Neural ODE / ODE-RNN tests — Chen et al. 2018, De Brouwer et al. 2019
//   "Neural Ordinary Differential Equations"   https://arxiv.org/abs/1806.07366
//   "GRU-ODE-Bayes"                           https://arxiv.org/abs/1905.04374
// ============================================================================
//
// Test coverage:
//   - ODE solvers (Euler / Midpoint / RK4 / DOPRI5): convergence on a trivial
//     ODE f(h)=h (analytic: h(t)=exp(t)) and agreement between RK4 and the
//     analytic answer at machine precision.
//   - ODEFunc: forward/backward/shape contract, parameter/gradient contract.
//   - NeuralODE: forward shape, direct-backward numerical gradient check,
//     adjoint-backward matches direct at ~1e-5, training reduces loss,
//     parameter/gradient shape contract.
//   - ODERNN: forward_seq shape, backward numerical gradient check, training
//     reduces loss on a synthetic irregular-time sequence.
//
// Layer-under-test: include/nn/layers/architectures/neural_ode.h
// ============================================================================

#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/layers/architectures/neural_ode.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace {

int g_tests_passed = 0;
int g_tests_failed = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  [FAIL] %s (line %d)\n", msg, __LINE__); \
        ++g_tests_failed; \
    } else { \
        ++g_tests_passed; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, tol, msg) do { \
    double _a = (double)(a); \
    double _b = (double)(b); \
    if (std::fabs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "  [FAIL] %s (line %d): |%g - %g| = %g > %g\n", \
                     msg, __LINE__, _a, _b, std::fabs(_a - _b), (double)(tol)); \
        ++g_tests_failed; \
    } else { \
        ++g_tests_passed; \
    } \
} while(0)

// Relative error helper
double rel_err(double a, double b) {
    double denom = std::max(std::fabs(a), std::fabs(b));
    denom = std::max(denom, 1e-12);
    return std::fabs(a - b) / denom;
}

// ----------------------------------------------------------------------------
// Tests for the solver free functions
// ----------------------------------------------------------------------------

void test_solver_euler_step_exponential() {
    std::fprintf(stderr, "test_solver_euler_step_exponential ... ");
    // f(h, t, x) = h  ==>  h(t) = exp(t) * h(0)
    auto f = [](const Tensor& h, double /*t*/, const Tensor& /*x*/) -> Tensor {
        return h.clone();
    };
    Tensor h(1, 1);
    h[0][0] = 1.0;
    Tensor x(1, 1); x[0][0] = 0.0;

    // Single Euler step with dt=0.1: h(0.1) ≈ 1.1
    Tensor h1 = odesolver::euler_step(f, h, 0.0, 0.1, x);
    EXPECT_NEAR(h1[0][0], 1.1, 1e-12, "euler step 1 matches h(0.1) ≈ 1.1");
}

void test_solver_rk4_converges_to_exp() {
    std::fprintf(stderr, "test_solver_rk4_converges_to_exp ... ");
    // RK4 should match exp(1) ≈ 2.71828 to ~1e-5 with dt=0.1 over 10 steps
    auto f = [](const Tensor& h, double /*t*/, const Tensor& /*x*/) -> Tensor {
        return h.clone();
    };
    Tensor h(1, 1); h[0][0] = 1.0;
    Tensor x(1, 1); x[0][0] = 0.0;
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
        h = odesolver::rk4_step(f, h, t, 0.1, x);
        t += 0.1;
    }
    EXPECT_NEAR(h[0][0], std::exp(1.0), 1e-5, "RK4 10 steps matches exp(1)");
}

void test_solver_euler_converges_to_exp() {
    std::fprintf(stderr, "test_solver_euler_converges_to_exp ... ");
    auto f = [](const Tensor& h, double /*t*/, const Tensor& /*x*/) -> Tensor {
        return h.clone();
    };
    Tensor h(1, 1); h[0][0] = 1.0;
    Tensor x(1, 1); x[0][0] = 0.0;
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
        h = odesolver::euler_step(f, h, t, 0.1, x);
        t += 0.1;
    }
    // Euler converges as O(dt), so at dt=0.1 we expect ~5% error
    EXPECT(rel_err(h[0][0], std::exp(1.0)) < 0.1, "Euler 10 steps roughly matches exp(1) (within 10%)");
}

void test_solver_midpoint_converges_to_exp() {
    std::fprintf(stderr, "test_solver_midpoint_converges_to_exp ... ");
    auto f = [](const Tensor& h, double /*t*/, const Tensor& /*x*/) -> Tensor {
        return h.clone();
    };
    Tensor h(1, 1); h[0][0] = 1.0;
    Tensor x(1, 1); x[0][0] = 0.0;
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
        h = odesolver::midpoint_step(f, h, t, 0.1, x);
        t += 0.1;
    }
    // Midpoint is 2nd-order, error ~O(dt^2) → expect ~1% error at dt=0.1
    EXPECT(rel_err(h[0][0], std::exp(1.0)) < 0.02, "Midpoint 10 steps matches exp(1) (within 2%)");
}

void test_solver_dopri5_converges_to_exp() {
    std::fprintf(stderr, "test_solver_dopri5_converges_to_exp ... ");
    auto f = [](const Tensor& h, double /*t*/, const Tensor& /*x*/) -> Tensor {
        return h.clone();
    };
    Tensor h(1, 1); h[0][0] = 1.0;
    Tensor x(1, 1); x[0][0] = 0.0;
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
        h = odesolver::dopri5_step(f, h, t, 0.1, x);
        t += 0.1;
    }
    // DOPRI5 is 5th-order, error should be ~1e-7 at dt=0.1
    EXPECT(rel_err(h[0][0], std::exp(1.0)) < 1e-6, "DOPRI5 10 steps matches exp(1) (within 1e-6)");
}

// ----------------------------------------------------------------------------
// Tests for ODEFunc
// ----------------------------------------------------------------------------

void test_odefunc_constructor() {
    std::fprintf(stderr, "test_odefunc_constructor ... ");
    ODEFunc f(2, 4);
    EXPECT(f.input_dim_ == 2, "input_dim stored");
    EXPECT(f.hidden_dim_ == 4, "hidden_dim stored");

    bool threw = false;
    try { ODEFunc g(0, 4); } catch (std::invalid_argument&) { threw = true; }
    EXPECT(threw, "input_dim=0 throws");

    threw = false;
    try { ODEFunc g(2, 0); } catch (std::invalid_argument&) { threw = true; }
    EXPECT(threw, "hidden_dim=0 throws");

    EXPECT(f.name() == "ODEFunc", "name() returns ODEFunc");
}

void test_odefunc_forward_shape() {
    std::fprintf(stderr, "test_odefunc_forward_shape ... ");
    ODEFunc f(2, 4);
    Tensor h(1, 4);
    h[0][0] = 0.1; h[0][1] = 0.2; h[0][2] = 0.0; h[0][3] = 0.0;
    Tensor x(1, 2);
    x[0][0] = 0.3; x[0][1] = 0.4;
    Tensor out = f.forward_with_cache(h, 0.0, x);
    EXPECT(out.rows == 1 && out.cols == 4, "forward output shape (1, hidden_dim)");
    bool finite = true;
    for (size_t i = 0; i < out.rows * out.cols; ++i) {
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "forward output finite");
}

void test_odefunc_zero_input_zero_output() {
    std::fprintf(stderr, "test_odefunc_zero_input_zero_output ... ");
    ODEFunc f(2, 4);
    Tensor h(1, 4);  // zeros (hidden_dim)
    Tensor x(1, 2);  // zeros (input_dim)
    Tensor out = f.forward_with_cache(h, 0.0, x);
    bool all_zero = true;
    for (size_t i = 0; i < out.rows * out.cols; ++i) {
        if (std::fabs(out.data[i]) > 1e-12) { all_zero = false; break; }
    }
    EXPECT(all_zero, "ODEFunc(0, 0, 0) ≈ 0");
}

void test_odefunc_param_count() {
    std::fprintf(stderr, "test_odefunc_param_count ... ");
    ODEFunc f(2, 4);
    auto params = f.parameters();
    // dense1: (hidden + input) -> hidden: W1 (hidden x (hidden+input)), b1 (1 x hidden)
    // dense2: hidden -> hidden: W2 (hidden x hidden), b2 (1 x hidden)
    // Total = 4 parameter tensors
    EXPECT(params.size() == 4, "ODEFunc has 4 parameter tensors (W1, b1, W2, b2)");
}

void test_odefunc_grad_count() {
    std::fprintf(stderr, "test_odefunc_grad_count ... ");
    ODEFunc f(2, 4);
    auto grads = f.gradients();
    EXPECT(grads.size() == 4, "ODEFunc has 4 gradient tensors (W1, b1, W2, b2)");
}

void test_odefunc_zero_grad() {
    std::fprintf(stderr, "test_odefunc_zero_grad ... ");
    ODEFunc f(2, 4);
    // Run a forward + backward to populate grads
    Tensor h(1, 4); h[0][0] = 0.1; h[0][1] = 0.2; h[0][2] = 0.0; h[0][3] = 0.0;
    Tensor x(1, 2); x[0][0] = 0.3; x[0][1] = 0.4;
    Tensor out = f.forward_with_cache(h, 0.0, x);
    Tensor grad_out(1, 4);
    grad_out.fill(1.0);
    f.backward(grad_out, 0.0);

    f.zero_grad();
    auto grads = f.gradients();
    bool all_zero = true;
    for (auto* g : grads) {
        for (size_t i = 0; i < g->rows * g->cols; ++i) {
            if (std::fabs((*g).data[i]) > 0) { all_zero = false; break; }
        }
    }
    EXPECT(all_zero, "zero_grad clears all 4 gradients");
}

// ----------------------------------------------------------------------------
// Tests for NeuralODE
// ----------------------------------------------------------------------------

void test_neuralode_constructor() {
    std::fprintf(stderr, "test_neuralode_constructor ... ");
    NeuralODE node(2, 4, 3, "euler", 10, 0.1, false);
    EXPECT(node.input_dim_ == 2, "input_dim stored");
    EXPECT(node.hidden_dim_ == 4, "hidden_dim stored");
    EXPECT(node.output_dim_ == 3, "output_dim stored");
    EXPECT(node.solver_type_ == "euler", "solver_type stored");
    EXPECT(node.n_steps_ == 10, "n_steps stored");
    EXPECT(node.dt_ == 0.1, "dt stored");
    EXPECT(!node.is_using_adjoint(), "use_adjoint_ defaults to false");
    EXPECT(node.name() == "NeuralODE", "name() returns NeuralODE");

    bool threw = false;
    try { NeuralODE bad(2, 4, 3, "rk999"); } catch (std::invalid_argument&) { threw = true; }
    EXPECT(threw, "invalid solver_type throws");

    threw = false;
    try { NeuralODE bad(2, 4, 3, "euler", 0); } catch (std::invalid_argument&) { threw = true; }
    EXPECT(threw, "n_steps=0 throws");

    threw = false;
    try { NeuralODE bad(2, 4, 3, "euler", 10, 0.0); } catch (std::invalid_argument&) { threw = true; }
    EXPECT(threw, "dt=0 throws");
}

void test_neuralode_forward_shape() {
    std::fprintf(stderr, "test_neuralode_forward_shape ... ");
    NeuralODE node(2, 4, 3, "euler", 5, 0.1, false);
    Tensor x(1, 2);
    x[0][0] = 0.1; x[0][1] = 0.2;
    Tensor y = node.forward(x);
    EXPECT(y.rows == 1 && y.cols == 3, "forward output shape (1, output_dim)");
    bool finite = true;
    for (size_t i = 0; i < y.rows * y.cols; ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "forward output finite");
}

void test_neuralode_zero_input_zero_output() {
    std::fprintf(stderr, "test_neuralode_zero_input_zero_output ... ");
    NeuralODE node(2, 4, 3, "euler", 5, 0.1, false);
    Tensor x(1, 2);  // zeros
    Tensor y = node.forward(x);
    bool all_zero = true;
    for (size_t i = 0; i < y.rows * y.cols; ++i) {
        if (std::fabs(y.data[i]) > 1e-12) { all_zero = false; break; }
    }
    EXPECT(all_zero, "NeuralODE(0) ≈ 0 (linear dynamics, zero init)");
}

void test_neuralode_param_count() {
    std::fprintf(stderr, "test_neuralode_param_count ... ");
    NeuralODE node(2, 4, 3, "euler", 5, 0.1, false);
    auto params = node.parameters();
    // ODEFunc (4) + output_proj (W, b) = 6
    EXPECT(params.size() == 6, "NeuralODE has 6 parameter tensors (4 ODEFunc + 2 output_proj)");
}

void test_neuralode_determinism() {
    std::fprintf(stderr, "test_neuralode_determinism ... ");
    NeuralODE node1(2, 4, 3, "euler", 5, 0.1, false);
    NeuralODE node2(2, 4, 3, "euler", 5, 0.1, false);

    // Copy params from node1 to node2
    auto p1 = node1.parameters();
    auto p2 = node2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) {
        *p2[i] = p1[i]->clone();
    }

    Tensor x(1, 2);
    x[0][0] = 0.1; x[0][1] = 0.2;
    Tensor y1 = node1.forward(x);
    Tensor y2 = node2.forward(x);
    double max_diff = 0.0;
    for (size_t i = 0; i < y1.rows * y1.cols; ++i) {
        max_diff = std::max(max_diff, std::fabs(y1.data[i] - y2.data[i]));
    }
    EXPECT(max_diff < 1e-12, "two NeuralODEs with copied params produce identical forward");
}

// Numerical gradient check for NeuralODE
// We compare the analytical backward against a centered finite-difference.
void test_neuralode_direct_grad_check() {
    std::fprintf(stderr, "test_neuralode_direct_grad_check ... ");
    // Use a tiny config for tractability
    NeuralODE node(2, 3, 2, "euler", 3, 0.1, false);

    Tensor x(1, 2);
    x[0][0] = 0.3; x[0][1] = -0.2;

    // Forward + backward
    Tensor y = node.forward(x);
    Tensor grad_y(1, 2);
    grad_y[0][0] = 0.7; grad_y[0][1] = -0.5;
    Tensor grad_x_ana = node.backward(grad_y, 0.0);

    // Numerical grad w.r.t. x: perturb each element
    double eps = 1e-5;
    for (size_t j = 0; j < x.cols; ++j) {
        Tensor xp = x.clone(); xp[0][j] += eps;
        Tensor xm = x.clone(); xm[0][j] -= eps;
        // Need to copy params (backward modifies state) — re-init
        NeuralODE node_p(2, 3, 2, "euler", 3, 0.1, false);
        NeuralODE node_m(2, 3, 2, "euler", 3, 0.1, false);
        auto src_p = node.parameters();
        auto dst_p = node_p.parameters();
        for (size_t i = 0; i < src_p.size(); ++i) *dst_p[i] = src_p[i]->clone();
        auto src_m = node.parameters();
        auto dst_m = node_m.parameters();
        for (size_t i = 0; i < src_m.size(); ++i) *dst_m[i] = src_m[i]->clone();

        Tensor yp = node_p.forward(xp);
        Tensor ym = node_m.forward(xm);
        double num = 0.0;
        for (size_t k = 0; k < grad_y.cols; ++k) {
            num += grad_y[0][k] * (yp[0][k] - ym[0][k]) / (2.0 * eps);
        }
        double ana = grad_x_ana[0][j];
        double err = rel_err(num, ana);
        EXPECT(err < 1e-4, "input grad direct vs FD (per-element rel_err < 1e-4)");
    }
}

void test_neuralode_adjoint_grad_check() {
    std::fprintf(stderr, "test_neuralode_adjoint_grad_check ... ");
    NeuralODE node(2, 3, 2, "euler", 3, 0.1, /*adjoint=*/true);

    Tensor x(1, 2);
    x[0][0] = 0.3; x[0][1] = -0.2;

    Tensor y = node.forward(x);
    Tensor grad_y(1, 2);
    grad_y[0][0] = 0.7; grad_y[0][1] = -0.5;
    Tensor grad_x_ana = node.backward(grad_y, 0.0);

    // Compare against direct-mode analytical grad
    NeuralODE node_direct(2, 3, 2, "euler", 3, 0.1, /*adjoint=*/false);
    auto src = node.parameters();
    auto dst = node_direct.parameters();
    for (size_t i = 0; i < src.size(); ++i) *dst[i] = src[i]->clone();
    node_direct.forward(x);
    Tensor grad_x_direct = node_direct.backward(grad_y, 0.0);

    double err = 0.0;
    for (size_t i = 0; i < grad_x_ana.rows * grad_x_ana.cols; ++i) {
        err = std::max(err, rel_err(grad_x_ana.data[i], grad_x_direct.data[i]));
    }
    EXPECT(err < 1e-4, "adjoint input grad matches direct input grad (rel_err < 1e-4)");
}

// Param gradient check: compare direct-mode grad_W1 against FD
void test_neuralode_param_grad_check() {
    std::fprintf(stderr, "test_neuralode_param_grad_check ... ");
    NeuralODE node(2, 3, 2, "euler", 3, 0.1, false);
    Tensor x(1, 2);
    x[0][0] = 0.3; x[0][1] = -0.2;

    Tensor y = node.forward(x);
    Tensor grad_y(1, 2);
    grad_y[0][0] = 0.7; grad_y[0][1] = -0.5;
    node.backward(grad_y, 0.0);

    // Get analytical grad for dense1 W
    auto grads = node.gradients();
    Tensor grad_W1_ana = grads[0]->clone();  // dense1.weights

    // FD: perturb each element of W1, re-forward, observe delta_y · grad_y
    double eps = 1e-5;
    double max_err = 0.0;
    for (size_t i = 0; i < grad_W1_ana.rows; ++i) {
        for (size_t j = 0; j < grad_W1_ana.cols; ++j) {
            // Perturb +eps
            NeuralODE node_p(2, 3, 2, "euler", 3, 0.1, false);
            NeuralODE node_m(2, 3, 2, "euler", 3, 0.1, false);
            auto src_p = node.parameters();
            auto dst_p = node_p.parameters();
            for (size_t k = 0; k < src_p.size(); ++k) *dst_p[k] = src_p[k]->clone();
            auto src_m = node.parameters();
            auto dst_m = node_m.parameters();
            for (size_t k = 0; k < src_m.size(); ++k) *dst_m[k] = src_m[k]->clone();
            (*dst_p[0])[i][j] += eps;
            (*dst_m[0])[i][j] -= eps;
            Tensor yp = node_p.forward(x);
            Tensor ym = node_m.forward(x);
            double num = 0.0;
            for (size_t k = 0; k < grad_y.cols; ++k) {
                num += grad_y[0][k] * (yp[0][k] - ym[0][k]) / (2.0 * eps);
            }
            double ana = (*grads[0])[i][j];
            max_err = std::max(max_err, rel_err(num, ana));
        }
    }
    EXPECT(max_err < 1e-3, "dense1.W grad direct vs FD (max rel_err < 1e-3)");
}

void test_neuralode_training_reduces_loss() {
    std::fprintf(stderr, "test_neuralode_training_reduces_loss ... ");
    // Train on a simple regression target: y_target = sin(input)
    NeuralODE node(2, 4, 2, "euler", 5, 0.1, false);

    // Synthetic dataset: x = [t, t], target = [sin(t), cos(t)]
    auto params = node.parameters();
    Tensor x(1, 2); x[0][0] = 0.5; x[0][1] = 0.5;

    double lr = 0.05;
    double initial_loss = 0.0, final_loss = 0.0;
    for (int step = 0; step < 50; ++step) {
        // Build target
        double t = 0.5;
        Tensor y_target(1, 2);
        y_target[0][0] = std::sin(t);
        y_target[0][1] = std::cos(t);

        Tensor y = node.forward(x);
        // MSE loss
        double loss = 0.0;
        for (size_t k = 0; k < y.cols; ++k) {
            double d = y[0][k] - y_target[0][k];
            loss += d * d;
        }
        loss /= (double)y.cols;

        if (step == 0) initial_loss = loss;
        if (step == 49) final_loss = loss;

        Tensor grad_y(1, 2);
        for (size_t k = 0; k < y.cols; ++k) {
            grad_y[0][k] = 2.0 * (y[0][k] - y_target[0][k]) / (double)y.cols;
        }
        node.backward(grad_y, lr);
        node.update_weights(lr);
        node.zero_grad();
    }
    EXPECT(final_loss < initial_loss, "training reduces loss over 50 SGD steps");
    EXPECT(final_loss < initial_loss * 0.5, "training reduces loss by at least 50%");
}

// ----------------------------------------------------------------------------
// Tests for ODERNN
// ----------------------------------------------------------------------------

void test_odernn_constructor() {
    std::fprintf(stderr, "test_odernn_constructor ... ");
    ODERNN rnn(2, 4, 3, "euler", 0.1);
    EXPECT(rnn.input_dim_ == 2, "input_dim stored");
    EXPECT(rnn.hidden_dim_ == 4, "hidden_dim stored");
    EXPECT(rnn.output_dim_ == 3, "output_dim stored");
    EXPECT(rnn.solver_type_ == "euler", "solver_type stored");
    EXPECT(rnn.name() == "ODERNN", "name() returns ODERNN");
}

void test_odernn_forward_shape() {
    std::fprintf(stderr, "test_odernn_forward_shape ... ");
    ODERNN rnn(2, 4, 3, "euler", 0.1);
    Tensor X(3, 2);
    X[0][0] = 0.1; X[0][1] = 0.2;
    X[1][0] = 0.3; X[1][1] = 0.4;
    X[2][0] = 0.5; X[2][1] = 0.6;
    std::vector<double> T_seq = {0.0, 0.1, 0.2};
    Tensor Y = rnn.forward_seq(X, T_seq);
    EXPECT(Y.rows == 3 && Y.cols == 3, "forward_seq returns (T, output_dim)");
    bool finite = true;
    for (size_t i = 0; i < Y.rows * Y.cols; ++i) {
        if (!std::isfinite(Y.data[i])) { finite = false; break; }
    }
    EXPECT(finite, "forward_seq output finite");
}

void test_odernn_param_count() {
    std::fprintf(stderr, "test_odernn_param_count ... ");
    ODERNN rnn(2, 4, 3, "euler", 0.1);
    auto params = rnn.parameters();
    // ODEFunc (4) + input_proj (W, b) + hidden_proj (W, b) + output_proj (W, b) = 10
    EXPECT(params.size() == 10, "ODERNN has 10 parameter tensors");
}

void test_odernn_grad_check() {
    std::fprintf(stderr, "test_odernn_grad_check ... ");
    // Simple 2-step sequence, tiny hidden dim, euler ODE.
    ODERNN rnn(2, 3, 2, "euler", 0.1);
    Tensor X(2, 2);
    X[0][0] = 0.3; X[0][1] = -0.2;
    X[1][0] = 0.1; X[1][1] = 0.4;
    std::vector<double> T_seq = {0.0, 0.2};

    Tensor Y = rnn.forward_seq(X, T_seq);
    Tensor grad_Y(2, 2);
    grad_Y[0][0] = 0.7; grad_Y[0][1] = -0.5;
    grad_Y[1][0] = 0.2; grad_Y[1][1] = 0.6;
    rnn.backward(grad_Y, 0.0);
    auto grads = rnn.gradients();
    // Compare grad of dense1.W in ODEFunc (index 0) vs FD
    Tensor grad_W1_ana = grads[0]->clone();
    double eps = 1e-5;
    double max_err = 0.0;
    for (size_t i = 0; i < grad_W1_ana.rows; ++i) {
        for (size_t j = 0; j < grad_W1_ana.cols; ++j) {
            ODERNN rnn_p(2, 3, 2, "euler", 0.1);
            ODERNN rnn_m(2, 3, 2, "euler", 0.1);
            auto src_p = rnn.parameters(); auto dst_p = rnn_p.parameters();
            for (size_t k = 0; k < src_p.size(); ++k) *dst_p[k] = src_p[k]->clone();
            auto src_m = rnn.parameters(); auto dst_m = rnn_m.parameters();
            for (size_t k = 0; k < src_m.size(); ++k) *dst_m[k] = src_m[k]->clone();
            (*dst_p[0])[i][j] += eps;
            (*dst_m[0])[i][j] -= eps;
            Tensor Yp = rnn_p.forward_seq(X, T_seq);
            Tensor Ym = rnn_m.forward_seq(X, T_seq);
            double num = 0.0;
            for (size_t r = 0; r < grad_Y.rows; ++r) {
                for (size_t k = 0; k < grad_Y.cols; ++k) {
                    num += grad_Y[r][k] * (Yp[r][k] - Ym[r][k]) / (2.0 * eps);
                }
            }
            double ana = (*grads[0])[i][j];
            max_err = std::max(max_err, rel_err(num, ana));
        }
    }
    EXPECT(max_err < 1e-3, "ODERNN dense1.W grad vs FD (max rel_err < 1e-3)");
}

void test_odernn_training_reduces_loss() {
    std::fprintf(stderr, "test_odernn_training_reduces_loss ... ");
    ODERNN rnn(2, 4, 2, "euler", 0.1);
    Tensor X(3, 2);
    X[0][0] = 0.5; X[0][1] = 0.5;
    X[1][0] = 0.6; X[1][1] = 0.6;
    X[2][0] = 0.7; X[2][1] = 0.7;
    std::vector<double> T_seq = {0.0, 0.1, 0.2};

    double lr = 0.02;
    double initial_loss = 0.0, final_loss = 0.0;
    for (int step = 0; step < 80; ++step) {
        Tensor Y = rnn.forward_seq(X, T_seq);
        // Target: y_target = 0.5 * (1 + sin(t))
        double loss = 0.0;
        for (size_t r = 0; r < Y.rows; ++r) {
            double t = T_seq[r];
            for (size_t k = 0; k < Y.cols; ++k) {
                double tgt = 0.5 * (1.0 + std::sin(t + 0.1 * k));
                double d = Y[r][k] - tgt;
                loss += d * d;
            }
        }
        loss /= (double)(Y.rows * Y.cols);
        if (step == 0) initial_loss = loss;
        if (step == 79) final_loss = loss;
        Tensor grad_Y(3, 2);
        for (size_t r = 0; r < Y.rows; ++r) {
            double t = T_seq[r];
            for (size_t k = 0; k < Y.cols; ++k) {
                double tgt = 0.5 * (1.0 + std::sin(t + 0.1 * k));
                grad_Y[r][k] = 2.0 * (Y[r][k] - tgt) / (double)(Y.rows * Y.cols);
            }
        }
        rnn.backward(grad_Y, lr);
        rnn.update_weights(lr);
        rnn.zero_grad();
    }
    EXPECT(final_loss < initial_loss, "ODERNN training reduces loss over 80 SGD steps");
}

}  // namespace

int main() {
    std::fprintf(stderr, "=== Running Neural ODE Tests ===\n");

    // Solvers
    test_solver_euler_step_exponential();
    test_solver_euler_converges_to_exp();
    test_solver_midpoint_converges_to_exp();
    test_solver_rk4_converges_to_exp();
    test_solver_dopri5_converges_to_exp();

    // ODEFunc
    test_odefunc_constructor();
    test_odefunc_forward_shape();
    test_odefunc_zero_input_zero_output();
    test_odefunc_param_count();
    test_odefunc_grad_count();
    test_odefunc_zero_grad();

    // NeuralODE
    test_neuralode_constructor();
    test_neuralode_forward_shape();
    test_neuralode_zero_input_zero_output();
    test_neuralode_param_count();
    test_neuralode_determinism();
    test_neuralode_direct_grad_check();
    test_neuralode_adjoint_grad_check();
    test_neuralode_param_grad_check();
    test_neuralode_training_reduces_loss();

    // ODERNN
    test_odernn_constructor();
    test_odernn_forward_shape();
    test_odernn_param_count();
    test_odernn_grad_check();
    test_odernn_training_reduces_loss();

    std::fprintf(stderr, "\n=== Summary: %d passed, %d failed ===\n", g_tests_passed, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
