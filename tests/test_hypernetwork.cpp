// test_hypernetwork.cpp — Tests for HyperNetwork
// Ha, Dai, Le 2016, "HyperNetworks" (https://arxiv.org/abs/1609.09106).
//
// A HyperNetwork is a small network whose output IS the weights of another
// (main) network. The generated weights depend on a context vector z, so the
// main network's behaviour is conditioned on z.
//
// In this implementation we generate a single Dense-like main layer
// (W ∈ R^{out×in}, b ∈ R^{1×out}) from a context z ∈ R^{1×context_dim}.
// The hyper-MLP is a 2-layer Dense → tanh → Dense.
//
// Tests:
//   1.  Constructor: throws on zero/empty dims; params / grads sizes match
//   2.  Forward shape: output is (B, out_features) and finite
//   3.  Same context → same output (deterministic for fixed weights)
//   4.  Different context → different output (context is actually used)
//   5.  Parameter count: 2 hyper Dense layers (= 4 tensors)
//   6.  Backward: input gradient is non-zero
//   7.  Backward: numerical vs analytical input gradient (rel_err < 1e-7)
//   8.  Backward: numerical vs analytical W_h1 gradient (rel_err < 1e-7)
//   9.  Backward: numerical vs analytical W_h2 gradient (rel_err < 1e-7)
//  10.  Backward: numerical vs analytical context gradient (rel_err < 1e-7)
//  11.  End-to-end training: loss decreases on a context-conditional task
//  12.  Generated W, b shape correctness (sanity)
//  13.  use_bias=false path works
//  14.  Mutation test: zeroing hyper1_.weights changes the output
//                       (the test isn't vacuous)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/layers/utility/hypernetwork.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"
#include "nn/core/model.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.data.size(); ++i) s += t.data[i] * t.data[i];
    return std::sqrt(s);
}

// Compute MSE loss between y and target, summed over all entries.
static double mse(const Tensor& y, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < y.data.size(); ++i) {
        double d = y.data[i] - target.data[i];
        s += d * d;
    }
    return s;
}

// Numerical gradient w.r.t. input via a function that takes a mutable input,
// calls layer.forward_with_context, returns the loss.
static Tensor numerical_grad_input(HyperNetwork& layer,
                                   Tensor input, const Tensor& context,
                                   const Tensor& target, double eps = 1e-4) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor yp = layer.forward_with_context(input, context);
            double lp = mse(yp, target);
            input(i, j) = orig - eps;
            Tensor ym = layer.forward_with_context(input, context);
            double lm = mse(ym, target);
            input(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor numerical_grad_W_h1(HyperNetwork& layer,
                                   const Tensor& input, const Tensor& context,
                                   const Tensor& target, double eps = 1e-4) {
    Tensor& W = layer.hyper1().weights;
    Tensor grad(W.rows, W.cols);
    for (size_t i = 0; i < W.rows; ++i) {
        for (size_t j = 0; j < W.cols; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor yp = layer.forward_with_context(input, context);
            double lp = mse(yp, target);
            W(i, j) = orig - eps;
            Tensor ym = layer.forward_with_context(input, context);
            double lm = mse(ym, target);
            W(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor numerical_grad_W_h2(HyperNetwork& layer,
                                   const Tensor& input, const Tensor& context,
                                   const Tensor& target, double eps = 1e-4) {
    Tensor& W = layer.hyper2().weights;
    Tensor grad(W.rows, W.cols);
    for (size_t i = 0; i < W.rows; ++i) {
        for (size_t j = 0; j < W.cols; ++j) {
            double orig = W(i, j);
            W(i, j) = orig + eps;
            Tensor yp = layer.forward_with_context(input, context);
            double lp = mse(yp, target);
            W(i, j) = orig - eps;
            Tensor ym = layer.forward_with_context(input, context);
            double lm = mse(ym, target);
            W(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor numerical_grad_context(HyperNetwork& layer,
                                      const Tensor& input, Tensor context,
                                      const Tensor& target, double eps = 1e-4) {
    Tensor grad(1, context.cols);
    for (size_t j = 0; j < context.cols; ++j) {
        double orig = context(0, j);
        context(0, j) = orig + eps;
        Tensor yp = layer.forward_with_context(input, context);
        double lp = mse(yp, target);
        context(0, j) = orig - eps;
        Tensor ym = layer.forward_with_context(input, context);
        double lm = mse(ym, target);
        context(0, j) = orig;
        grad(0, j) = (lp - lm) / (2.0 * eps);
    }
    return grad;
}

// Relative error
static double rel_err(const Tensor& a, const Tensor& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        num += (a.data[i] - b.data[i]) * (a.data[i] - b.data[i]);
        den += a.data[i] * a.data[i];
    }
    if (den < 1e-20) {
        // Analytical is ~0; switch to absolute
        return std::sqrt(num);
    }
    return std::sqrt(num / den);
}

// =====================================================================
// Test 1: Constructor sanity
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: HyperNetwork constructor --" << endl;

    bool threw_zero = false;
    try { HyperNetwork hn(0, 4, 2, 8, true); }
    catch (const std::invalid_argument&) { threw_zero = true; }
    check("Throws on context_dim=0", threw_zero);

    HyperNetwork hn(4, 3, 2, 8, true);
    check("Parameter count: 4 (W_h1, b_h1, W_h2, b_h2)",
          hn.parameters().size() == 4);
    check("Gradient count matches parameter count",
          hn.gradients().size() == hn.parameters().size());
    check("context_dim() correct",  hn.context_dim()  == 4);
    check("in_features() correct",  hn.in_features()  == 3);
    check("out_features() correct", hn.out_features() == 2);
    check("hyper_hidden() correct", hn.hyper_hidden() == 8);
    check("use_bias() correct",     hn.use_bias()     == true);
}

// =====================================================================
// Test 2: Forward shape
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 2: HyperNetwork forward shape --" << endl;

    HyperNetwork hn(4, 3, 5, 8, true);
    Tensor context(1, 4);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t j = 0; j < 4; ++j) context(0, j) = dist(rng);

    Tensor input(7, 3);
    for (size_t i = 0; i < 7; ++i)
        for (size_t j = 0; j < 3; ++j)
            input(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    check("Output shape (B, out_features) = (7, 5)",
          y.rows == 7 && y.cols == 5);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { finite = false; break; }
    }
    check("Output is finite", finite);

    check("last_W_ shape (out_features, in_features) = (5, 3)",
          hn.last_W().rows == 5 && hn.last_W().cols == 3);
    check("last_b_ shape (1, out_features) = (1, 5)",
          hn.last_b().rows == 1 && hn.last_b().cols == 5);
}

// =====================================================================
// Test 3: Determinism (same context -> same output)
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 3: Determinism --" << endl;

    HyperNetwork hn(4, 3, 2, 8, true);
    Tensor context(1, 4);
    context(0, 0) = 0.1; context(0, 1) = 0.2;
    context(0, 2) = 0.3; context(0, 3) = 0.4;

    Tensor input(2, 3);
    input(0, 0) = 1.0; input(0, 1) = 0.5; input(0, 2) = -0.5;
    input(1, 0) = 0.0; input(1, 1) = 1.0; input(1, 2) = 0.0;

    Tensor y1 = hn.forward_with_context(input, context);
    Tensor y2 = hn.forward_with_context(input, context);
    bool same = true;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (std::abs(y1.data[i] - y2.data[i]) > 1e-12) { same = false; break; }
    }
    check("Same context + same input → same output (deterministic)", same);
}

// =====================================================================
// Test 4: Different context -> different output
// =====================================================================
static void test_context_changes_output() {
    cout << endl << "-- Test 4: Context conditioning --" << endl;

    HyperNetwork hn(4, 3, 2, 16, true);

    Tensor input(2, 3);
    input(0, 0) = 1.0; input(0, 1) = 0.5; input(0, 2) = -0.5;
    input(1, 0) = 0.0; input(1, 1) = 1.0; input(1, 2) = 0.0;

    Tensor z1(1, 4), z2(1, 4);
    z1(0, 0) = 0.1; z1(0, 1) = 0.2; z1(0, 2) = 0.3; z1(0, 3) = 0.4;
    z2(0, 0) = 0.4; z2(0, 1) = 0.3; z2(0, 2) = 0.2; z2(0, 3) = 0.1;

    Tensor y1 = hn.forward_with_context(input, z1);
    Tensor y2 = hn.forward_with_context(input, z2);
    bool diff = false;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (std::abs(y1.data[i] - y2.data[i]) > 1e-6) { diff = true; break; }
    }
    check("Different context → different output (context is used)", diff);
    check("Output norms both nonzero", l2norm(y1) > 1e-6 && l2norm(y2) > 1e-6);
}

// =====================================================================
// Test 5: Backward input gradient (smoke)
// =====================================================================
static void test_backward_input_smoke() {
    cout << endl << "-- Test 5: Backward input gradient (smoke) --" << endl;

    HyperNetwork hn(3, 4, 2, 6, true);
    Tensor context(1, 3);
    context(0, 0) = 0.1; context(0, 1) = 0.5; context(0, 2) = -0.3;

    Tensor input(2, 4);
    input(0, 0) = 0.7; input(0, 1) = -0.2; input(0, 2) = 0.1; input(0, 3) = 0.9;
    input(1, 0) = -0.5; input(1, 1) = 0.3; input(1, 2) = 0.8; input(1, 3) = 0.0;

    Tensor y = hn.forward_with_context(input, context);
    Tensor grad_out(y.rows, y.cols);
    grad_out.fill(1.0);
    hn.zero_grad();
    Tensor grad_x = hn.backward(grad_out, 0.0);
    check("grad_x shape (B, in_features) = (2, 4)",
          grad_x.rows == 2 && grad_x.cols == 4);
    check("grad_x is non-zero", l2norm(grad_x) > 1e-10);
    check("hyper1 grad_weights is non-zero", l2norm(hn.hyper1().grad_weights) > 1e-10);
    check("hyper2 grad_weights is non-zero", l2norm(hn.hyper2().grad_weights) > 1e-10);
    check("last_context_grad_ is non-zero", l2norm(hn.last_context_grad()) > 1e-10);
}

// =====================================================================
// Test 6: Numerical vs analytical input gradient
// =====================================================================
static void test_numerical_input_grad() {
    cout << endl << "-- Test 6: numerical input gradient --" << endl;

    HyperNetwork hn(3, 4, 2, 6, true);
    Tensor context(1, 3);
    context(0, 0) = 0.1; context(0, 1) = 0.5; context(0, 2) = -0.3;

    Tensor input(3, 4);
    std::mt19937 rng(7);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor target(3, 2);
    for (size_t i = 0; i < target.rows; ++i)
        for (size_t j = 0; j < target.cols; ++j)
            target(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    Tensor loss_grad = (y - target) * 2.0;
    hn.zero_grad();
    Tensor grad_x_ana = hn.backward(loss_grad, 0.0);

    Tensor grad_x_num = numerical_grad_input(hn, input, context, target, 1e-5);
    double re = rel_err(grad_x_ana, grad_x_num);
    cout << "  [DEBUG] input grad rel_err = " << re << endl;
    check("Input grad rel_err < 1e-7", re < 1e-7);
}

// =====================================================================
// Test 7: Numerical vs analytical W_h1 gradient
// =====================================================================
static void test_numerical_W_h1() {
    cout << endl << "-- Test 7: numerical W_h1 gradient --" << endl;

    HyperNetwork hn(3, 4, 2, 5, true);
    Tensor context(1, 3);
    context(0, 0) = 0.2; context(0, 1) = -0.1; context(0, 2) = 0.4;

    Tensor input(3, 4);
    std::mt19937 rng(11);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor target(3, 2);
    for (size_t i = 0; i < target.rows; ++i)
        for (size_t j = 0; j < target.cols; ++j)
            target(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    Tensor loss_grad = (y - target) * 2.0;
    hn.zero_grad();
    hn.backward(loss_grad, 0.0);
    Tensor grad_W_h1_ana = hn.hyper1().grad_weights;

    Tensor grad_W_h1_num = numerical_grad_W_h1(hn, input, context, target, 1e-5);
    double re = rel_err(grad_W_h1_ana, grad_W_h1_num);
    cout << "  [DEBUG] W_h1 grad rel_err = " << re << endl;
    check("W_h1 grad rel_err < 1e-7", re < 1e-7);
}

// =====================================================================
// Test 8: Numerical vs analytical W_h2 gradient
// =====================================================================
static void test_numerical_W_h2() {
    cout << endl << "-- Test 8: numerical W_h2 gradient --" << endl;

    HyperNetwork hn(3, 4, 2, 5, true);
    Tensor context(1, 3);
    context(0, 0) = 0.2; context(0, 1) = -0.1; context(0, 2) = 0.4;

    Tensor input(3, 4);
    std::mt19937 rng(13);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor target(3, 2);
    for (size_t i = 0; i < target.rows; ++i)
        for (size_t j = 0; j < target.cols; ++j)
            target(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    Tensor loss_grad = (y - target) * 2.0;
    hn.zero_grad();
    hn.backward(loss_grad, 0.0);
    Tensor grad_W_h2_ana = hn.hyper2().grad_weights;

    Tensor grad_W_h2_num = numerical_grad_W_h2(hn, input, context, target, 1e-5);
    double re = rel_err(grad_W_h2_ana, grad_W_h2_num);
    cout << "  [DEBUG] W_h2 grad rel_err = " << re << endl;
    check("W_h2 grad rel_err < 1e-7", re < 1e-7);
}

// =====================================================================
// Test 9: Numerical vs analytical context gradient
// =====================================================================
static void test_numerical_context() {
    cout << endl << "-- Test 9: numerical context gradient --" << endl;

    HyperNetwork hn(3, 4, 2, 5, true);
    Tensor context(1, 3);
    context(0, 0) = 0.2; context(0, 1) = -0.1; context(0, 2) = 0.4;

    Tensor input(3, 4);
    std::mt19937 rng(17);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor target(3, 2);
    for (size_t i = 0; i < target.rows; ++i)
        for (size_t j = 0; j < target.cols; ++j)
            target(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    Tensor loss_grad = (y - target) * 2.0;
    hn.zero_grad();
    hn.backward(loss_grad, 0.0);
    Tensor grad_z_ana = hn.last_context_grad();

    Tensor grad_z_num = numerical_grad_context(hn, input, context, target, 1e-5);
    double re = rel_err(grad_z_ana, grad_z_num);
    cout << "  [DEBUG] context grad rel_err = " << re << endl;
    check("Context grad rel_err < 1e-7", re < 1e-7);
}

// =====================================================================
// Test 10: End-to-end training reduces loss on a context-conditional task
// =====================================================================
static void test_training() {
    cout << endl << "-- Test 10: Training reduces loss (context-conditional) --" << endl;

    // Task: a single hypernetwork must produce 2 DIFFERENT linear maps
    // depending on the context. When z = [1, 0, 0] the target is W1 * x.
    // When z = [0, 1, 0] the target is W2 * x. We train the hypernetwork
    // to learn both mappings simultaneously — impossible for a fixed
    // Dense layer but trivial for a HyperNetwork.
    const size_t context_dim = 3;
    const size_t in_dim      = 3;
    const size_t out_dim     = 2;

    // The two "true" target linear maps
    Tensor W1(out_dim, in_dim), W2(out_dim, in_dim);
    W1(0, 0) =  1.0; W1(0, 1) = -0.5; W1(0, 2) =  0.3;
    W1(1, 0) =  0.2; W1(1, 1) =  0.8; W1(1, 2) = -0.1;
    W2(0, 0) = -0.7; W2(0, 1) =  0.4; W2(0, 2) =  0.6;
    W2(1, 0) =  0.5; W2(1, 1) = -0.3; W2(1, 2) =  0.9;

    HyperNetwork hn(context_dim, in_dim, out_dim, 16, true);

    // Build batch with B = 4: 2 samples with z=[1,0,0] (target W1*x),
    //                          2 samples with z=[0,1,0] (target W2*x).
    Tensor input(4, in_dim);
    std::mt19937 rng(99);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    auto build_targets = [&](const Tensor& z_for_row) {
        Tensor target(4, out_dim);
        for (size_t i = 0; i < 4; ++i) {
            const Tensor& W = (z_for_row(0, 0) > 0.5) ? W1 : W2;
            for (size_t o = 0; o < out_dim; ++o) {
                double s = 0.0;
                for (size_t k = 0; k < in_dim; ++k) s += W(o, k) * input(i, k);
                target(i, o) = s;
            }
        }
        return target;
    };

    // Two "contexts" we cycle through
    Tensor z_a(1, context_dim), z_b(1, context_dim);
    z_a(0, 0) = 1.0; z_a(0, 1) = 0.0; z_a(0, 2) = 0.0;
    z_b(0, 0) = 0.0; z_b(0, 1) = 1.0; z_b(0, 2) = 0.0;

    Tensor target_a = build_targets(z_a);
    Tensor target_b = build_targets(z_b);

    auto run_loss = [&](const Tensor& z, const Tensor& target) {
        Tensor y = hn.forward_with_context(input, z);
        return mse(y, target) / (double)input.rows;
    };

    double initial_a = run_loss(z_a, target_a);
    double initial_b = run_loss(z_b, target_b);
    double initial_total = initial_a + initial_b;

    double lr = 0.05;
    for (int step = 0; step < 200; ++step) {
        // Alternate contexts
        const Tensor& z    = (step % 2 == 0) ? z_a    : z_b;
        const Tensor& targ = (step % 2 == 0) ? target_a : target_b;

        Tensor y = hn.forward_with_context(input, z);
        Tensor loss_grad = (y - targ) * (2.0 / (double)input.rows);
        hn.zero_grad();
        hn.backward(loss_grad, 0.0);
        hn.update_weights(lr);
    }

    double final_a = run_loss(z_a, target_a);
    double final_b = run_loss(z_b, target_b);
    double final_total = final_a + final_b;
    cout << "  [DEBUG] loss: " << initial_total << " -> " << final_total
         << "  (a: " << initial_a << "->" << final_a
         << ", b: " << initial_b << "->" << final_b << ")" << endl;
    check("Total loss decreased", final_total < initial_total * 0.5);
    check("Loss_a < 0.05 (well-learned)", final_a < 0.05);
    check("Loss_b < 0.05 (well-learned)", final_b < 0.05);
}

// =====================================================================
// Test 11: use_bias=false path
// =====================================================================
static void test_no_bias() {
    cout << endl << "-- Test 11: use_bias=false --" << endl;

    HyperNetwork hn(3, 4, 2, 5, false);
    Tensor context(1, 3);
    context(0, 0) = 0.1; context(0, 1) = 0.2; context(0, 2) = 0.3;

    Tensor input(2, 4);
    std::mt19937 rng(31);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor y = hn.forward_with_context(input, context);
    check("use_bias=false: output shape (B, out)", y.rows == 2 && y.cols == 2);

    // Verify that the bias slice of the flat output is NOT used
    // (i.e. last_b_ should be all zeros, by construction).
    bool b_zero = true;
    for (size_t j = 0; j < hn.last_b().cols; ++j) {
        if (std::abs(hn.last_b()(0, j)) > 1e-12) { b_zero = false; break; }
    }
    check("use_bias=false: last_b_ stays zero (not used)", b_zero);

    // Backward should not throw
    Tensor grad_out(2, 2);
    grad_out.fill(1.0);
    hn.zero_grad();
    Tensor grad_x = hn.backward(grad_out, 0.0);
    check("use_bias=false: backward returns grad_x of correct shape",
          grad_x.rows == 2 && grad_x.cols == 4);
    check("use_bias=false: hyper2 grad non-zero",
          l2norm(hn.hyper2().grad_weights) > 1e-10);
}

// =====================================================================
// Test 12: zero_grad() clears
// =====================================================================
static void test_zero_grad() {
    cout << endl << "-- Test 12: zero_grad clears --" << endl;

    HyperNetwork hn(3, 4, 2, 5, true);
    Tensor context(1, 3);
    context(0, 0) = 0.1; context(0, 1) = 0.2; context(0, 2) = 0.3;
    Tensor input(2, 4);
    std::mt19937 rng(43);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    // Run forward+backward once
    Tensor y = hn.forward_with_context(input, context);
    Tensor grad_out(2, 2);
    grad_out.fill(1.0);
    hn.backward(grad_out, 0.0);
    bool nonzero = l2norm(hn.hyper1().grad_weights) > 1e-10 &&
                   l2norm(hn.hyper2().grad_weights) > 1e-10;
    check("Before zero_grad: hyper grads are non-zero", nonzero);

    hn.zero_grad();
    bool cleared = l2norm(hn.hyper1().grad_weights) < 1e-20 &&
                   l2norm(hn.hyper2().grad_weights) < 1e-20;
    check("After zero_grad: hyper grads are exactly zero", cleared);
}

// =====================================================================
// Test 13: Mutation test — zeroing hyper1 weights changes the output
// =====================================================================
static void test_mutation_hyper1() {
    cout << endl << "-- Test 13: Mutation test (zeroing hyper1 changes output) --" << endl;

    HyperNetwork hn(3, 4, 2, 5, true);
    Tensor context(1, 3);
    context(0, 0) = 0.7; context(0, 1) = -0.3; context(0, 2) = 0.5;
    Tensor input(2, 4);
    std::mt19937 rng(53);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input(i, j) = dist(rng);

    Tensor y_before = hn.forward_with_context(input, context);
    hn.hyper1().weights.fill(0.0);
    Tensor y_after  = hn.forward_with_context(input, context);

    bool changed = false;
    for (size_t i = 0; i < y_before.data.size(); ++i) {
        if (std::abs(y_before.data[i] - y_after.data[i]) > 1e-6) {
            changed = true; break;
        }
    }
    check("Zeroing hyper1.weights changes the output (not vacuous)", changed);
}

// =====================================================================
// Test 14: Determinism with different input but same context
// =====================================================================
static void test_input_changes_output() {
    cout << endl << "-- Test 14: Different input -> different output --" << endl;

    HyperNetwork hn(3, 4, 2, 8, true);
    Tensor context(1, 3);
    context(0, 0) = 0.1; context(0, 1) = 0.2; context(0, 2) = 0.3;

    Tensor x1(2, 4), x2(2, 4);
    std::mt19937 rng(61);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            x1(i, j) = dist(rng);
            x2(i, j) = dist(rng);
        }

    Tensor y1 = hn.forward_with_context(x1, context);
    Tensor y2 = hn.forward_with_context(x2, context);
    bool diff = false;
    for (size_t i = 0; i < y1.data.size(); ++i) {
        if (std::abs(y1.data[i] - y2.data[i]) > 1e-6) { diff = true; break; }
    }
    check("Different input → different output", diff);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== HyperNetwork Tests ===" << endl;
    cout << setprecision(8);

    test_constructor();
    test_forward_shape();
    test_determinism();
    test_context_changes_output();
    test_backward_input_smoke();
    test_numerical_input_grad();
    test_numerical_W_h1();
    test_numerical_W_h2();
    test_numerical_context();
    test_training();
    test_no_bias();
    test_zero_grad();
    test_mutation_hyper1();
    test_input_changes_output();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}