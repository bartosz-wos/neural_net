// test_tree_lstm.cpp — Tests for the Child-Sum Tree-LSTM layer.
//
// Reference: Tai, Socher, Manning 2015 "Improved Semantic Representations
//   From Tree-Structured Long Short-Term Memory Networks", §2.2.
//
// Tests:
//   1. TreeLSTM forward shape on a chain (3 nodes, single chain)
//   2. TreeLSTM forward shape on a binary tree (7 nodes, root + 2 + 4 leaves)
//   3. TreeLSTM output is finite
//   4. Hand-computed reference for a tiny tree (4 nodes, hidden=1, all
//      parameters fixed by hand to match a known-good Python/numpy impl)
//   5. Input gradient check (numerical vs analytical) on a chain
//   6. Input gradient check on a binary tree
//   7. W gradient check
//   8. U gradient check
//   9. b gradient check
//  10. Parameters / gradients shape consistency
//  11. Forget-bias init = 1 convention
//  12. Training step reduces loss
//  13. TreeLSTMModel forward + backward end-to-end on a binary tree
//  14. Determinism: two fresh instances with same seed → identical output

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "nn/layers/architectures/tree_lstm.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << "\n"; ++passed; }
    else       { cout << "  [FAIL] " << name << "\n"; ++failed; }
    return pass;
}
static double rel_err(double a, double b) {
    double m = max(fabs(a), fabs(b));
    if (m < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / m;
}

// L2 loss and gradient (0.5 * sum (y - target)^2)
static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// Numerical input gradient (centered finite differences).
static Tensor numerical_grad_input(
    TreeLSTM& cell,
    Tensor& input,
    const std::vector<std::vector<int>>& children,
    const Tensor& target,
    double eps = 1e-5)
{
    Tensor ng(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = cell.forward(input, children);
            double lp = l2_loss_value(out_p, target);

            input(i, j) = orig - eps;
            Tensor out_m = cell.forward(input, children);
            double lm = l2_loss_value(out_m, target);

            input(i, j) = orig;
            ng(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return ng;
}

// Same for TreeLSTMModel (binary classifier-style loss).
static Tensor numerical_grad_input_model(
    TreeLSTMModel& model,
    Tensor& input,
    const std::vector<std::vector<int>>& children,
    const Tensor& target,
    double eps = 1e-5)
{
    Tensor ng(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = model.forward_with_tree(input, children);
            double lp = l2_loss_value(out_p, target);

            input(i, j) = orig - eps;
            Tensor out_m = model.forward_with_tree(input, children);
            double lm = l2_loss_value(out_m, target);

            input(i, j) = orig;
            ng(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return ng;
}

// =====================================================================
// Test 1: forward shape on a chain
// =====================================================================
static void test_chain_shape() {
    cout << "\n--- Test 1: forward shape on a chain (3 nodes) ---\n";
    // Tree: 0 -> 1 -> 2 (each node has 1 child except leaf)
    std::vector<std::vector<int>> children = {
        {1},   // children of 0
        {2},   // children of 1
        {}     // children of 2 (leaf)
    };

    size_t in_dim = 3, hidden = 4;
    TreeLSTM cell(in_dim, hidden);

    Tensor input(3, in_dim);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);

    Tensor out = cell.forward(input, children);
    check("output shape (3, 4)", out.rows == 3 && out.cols == 4);
}

// =====================================================================
// Test 2: forward shape on a binary tree
// =====================================================================
static void test_binary_tree_shape() {
    cout << "\n--- Test 2: forward shape on a binary tree (7 nodes) ---\n";
    // Tree:        0
    //              / \
    //             1   2
    //            / \ / \
    //           3  4 5  6
    // (root=0; children[0]={1,2}, children[1]={3,4}, children[2]={5,6}, leaves empty)
    std::vector<std::vector<int>> children = {
        {1, 2},
        {3, 4},
        {5, 6},
        {}, {}, {}, {}
    };

    size_t in_dim = 5, hidden = 6;
    TreeLSTM cell(in_dim, hidden);

    Tensor input(7, in_dim);
    std::mt19937 gen(7);
    std::normal_distribution<> nd(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = nd(gen);

    Tensor out = cell.forward(input, children);
    check("output shape (7, 6)", out.rows == 7 && out.cols == 6);
}

// =====================================================================
// Test 3: output is finite
// =====================================================================
static void test_finite() {
    cout << "\n--- Test 3: output is finite ---\n";
    std::vector<std::vector<int>> children = {
        {1, 2}, {3}, {}, {}
    };
    TreeLSTM cell(3, 5);
    Tensor input(4, 3);
    std::mt19937 gen(1);
    std::normal_distribution<> nd(0.0, 1.0);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = nd(gen);

    Tensor out = cell.forward(input, children);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (!std::isfinite(out.data[i])) finite = false;
    }
    check("all output entries finite", finite);
}

// =====================================================================
// Test 4: hand-computed reference for a tiny tree
//
// We construct a minimal tree with hidden=1, in_dim=1, and a fixed W, U, b
// and verify the forward output matches what we compute by hand.
// =====================================================================
static void test_hand_computed_reference() {
    cout << "\n--- Test 4: hand-computed reference (4 nodes, hidden=1) ---\n";
    // Tree: 0 -> {1, 2}, 2 -> {3}, 1 leaf, 3 leaf.  Root = 0.
    //  (root=0; children[0]={1,2}, children[2]={3}, leaves empty)
    std::vector<std::vector<int>> children = {
        {1, 2}, {}, {3}, {}
    };
    // in_dim=1, hidden=1.
    TreeLSTM cell(1, 1);
    // Hand-set all weights and biases to known values.
    // Combined W (4*1, 1) layout [i, f, o, u]. Biases too.
    // Forget bias b_f = 1 (init default), others 0.
    cell.W(0, 0) = 0.5;  // i_x
    cell.W(1, 0) = 0.0;  // f_x
    cell.W(2, 0) = 0.0;  // o_x
    cell.W(3, 0) = 0.0;  // u_x
    cell.U(0, 0) = 0.0;  // i_hsum
    cell.U(1, 0) = 0.0;  // f_hsum (forget uses per-child h_k, but U_f is the
                          //   per-child h_k weight — we set to 0 so f_jk
                          //   depends only on the forget bias)
    cell.U(2, 0) = 0.0;  // o_hsum
    cell.U(3, 0) = 0.0;  // u_hsum
    // b: layout [i, f, o, u] — forget bias = 1
    cell.b(0, 0) = 0.0;
    cell.b(0, 1) = 1.0;
    cell.b(0, 2) = 0.0;
    cell.b(0, 3) = 0.0;

    // Input features: x_0 = 0.4, x_1 = 0.0 (leaf), x_2 = 0.2, x_3 = 0.0
    // (x_3 kept at 0 so leaves are trivially zero for clean hand-calc.)
    Tensor input(4, 1);
    input(0, 0) = 0.4;
    input(1, 0) = 0.0;
    input(2, 0) = 0.2;
    input(3, 0) = 0.0;

    // Hand calculation (post-order: 1, 3, 2, 0):
    //   Leaf 1: h_sum = 0, h_1 = 0 (no children, c_1 = 0)
    //   Leaf 3: h_sum = 0, h_3 = 0, c_3 = 0
    //   Node 2 (one child 3):
    //     h_sum_2 = h_3 = 0
    //     i_2_pre = W_i*x_2 + U_i*h_sum_2 + b_i = 0.5*0.2 + 0 + 0 = 0.1
    //     i_2 = σ(0.1) = 0.52498
    //     f_2,3_pre = W_f*x_2 + U_f*h_3 + b_f = 0 + 0 + 1 = 1
    //     f_2,3 = σ(1) = 0.73106
    //     o_2_pre = 0, o_2 = 0.5
    //     u_2_pre = 0, u_2 = 0
    //     c_2 = i_2 * u_2 + f_2,3 * c_3 = 0 + 0 = 0
    //     h_2 = o_2 * tanh(c_2) = 0.5 * 0 = 0
    //   Node 0 (children 1, 2):
    //     h_sum_0 = h_1 + h_2 = 0 + 0 = 0
    //     i_0_pre = 0.5 * 0.4 = 0.2
    //     i_0 = σ(0.2) = 0.54983
    //     f_0,1 = σ(1) = 0.73106
    //     f_0,2 = σ(1) = 0.73106
    //     o_0 = 0.5
    //     u_0 = 0
    //     c_0 = i_0 * u_0 + f_0,1*c_1 + f_0,2*c_2 = 0 + 0 + 0 = 0
    //     h_0 = 0
    //   h_1 = 0, h_3 = 0, h_2 = 0, h_0 = 0  (all zeros)
    Tensor out = cell.forward(input, children);
    bool all_zero = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (fabs(out.data[i]) > 1e-9) all_zero = false;
    }
    check("trivial hand calc: all zero", all_zero);

    // Now make it non-trivial: set W_o, U_o, U_u to small nonzero values
    // so the output is non-degenerate but still hand-computable.
    cell.b(0, 3) = 0.0;  // u bias
    cell.W(3, 0) = 0.5;  // u_x = 0.5
    cell.U(3, 0) = 0.1;  // u_hsum = 0.1
    cell.W(2, 0) = 0.2;  // o_x = 0.2
    cell.U(2, 0) = 0.1;  // o_hsum = 0.1
    cell.W(0, 0) = 0.0;  // i_x = 0
    cell.U(0, 0) = 0.0;  // i_hsum = 0

    Tensor out2 = cell.forward(input, children);
    // Recompute by hand with new values (x_3 = 0 keeps leaves at 0):
    //   Leaf 1, 3: h = c = 0
    //   Node 2: h_sum = 0
    //     i_2_pre = 0*0.2 + 0*0 + 0 = 0
    //     i_2 = σ(0) = 0.5
    //     f_2,3 = σ(0*0.2 + 0*0 + 1) = σ(1) = 0.73106
    //     o_2_pre = 0.2*0.2 + 0.1*0 + 0 = 0.04
    //     o_2 = σ(0.04) = 0.51000
    //     u_2_pre = 0.5*0.2 + 0.1*0 + 0 = 0.1
    //     u_2 = tanh(0.1) = 0.09967
    //     c_2 = 0.5*0.09967 + 0.73106*0 = 0.04984
    //     h_2 = 0.51000 * tanh(0.04984) = 0.51000 * 0.04979 = 0.02539
    //   Node 0: h_sum = h_1 + h_2 = 0 + 0.02539 = 0.02539
    //     i_0_pre = 0*0.4 + 0*0.02539 + 0 = 0
    //     i_0 = σ(0) = 0.5
    //     f_0,1 = f_0,2 = σ(0*0.4 + 0*0.02539 + 1) = σ(1) = 0.73106
    //     o_0_pre = 0.2*0.4 + 0.1*0.02539 + 0 = 0.08254
    //     o_0 = σ(0.08254) = 0.52061
    //     u_0_pre = 0.5*0.4 + 0.1*0.02539 + 0 = 0.20254
    //     u_0 = tanh(0.20254) = 0.20000
    //     c_0 = 0.5*0.20000 + 0.73106*0 + 0.73106*0.04984 = 0.10000 + 0.03643 = 0.13643
    //     h_0 = 0.52061 * tanh(0.13643) = 0.52061 * 0.13565 = 0.07063
    double h2_expected = 0.02539;
    double h0_expected = 0.07063;
    double h2_got = out2(2, 0);
    double h0_got = out2(0, 0);
    cout << "  hand calc: h_2 expected=" << h2_expected << " got=" << h2_got
         << " (rel_err=" << rel_err(h2_got, h2_expected) << ")\n";
    cout << "  hand calc: h_0 expected=" << h0_expected << " got=" << h0_got
         << " (rel_err=" << rel_err(h0_got, h0_expected) << ")\n";
    check("hand-calc h_2 matches", rel_err(h2_got, h2_expected) < 2e-3);
    check("hand-calc h_0 matches", rel_err(h0_got, h0_expected) < 2e-3);
}

// =====================================================================
// Test 5: input gradient check on a chain
// =====================================================================
static void test_input_grad_chain() {
    cout << "\n--- Test 5: input gradient check on a chain ---\n";
    std::vector<std::vector<int>> children = { {1}, {2}, {} };
    size_t in_dim = 2, hidden = 2;
    TreeLSTM cell(in_dim, hidden);
    Tensor input(3, in_dim);
    for (size_t i = 0; i < 3; ++i)
        for (size_t k = 0; k < in_dim; ++k)
            input(i, k) = 0.2 * sin(0.5 * (i + 1)) + 0.1 * (k + 1);

    Tensor target(3, hidden);
    for (size_t i = 0; i < 3; ++i)
        for (size_t k = 0; k < hidden; ++k)
            target(i, k) = 0.3 - 0.1 * (k + 1);

    Tensor out = cell.forward(input, children);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    Tensor grad_x = cell.backward(grad_loss, 0.0);
    Tensor grad_x_num = numerical_grad_input(cell, input, children, target);

    double max_err = 0.0;
    for (size_t i = 0; i < grad_x.data.size(); ++i) {
        double e = rel_err(grad_x.data[i], grad_x_num.data[i]);
        if (e > max_err) max_err = e;
    }
    cout << "  max rel_err = " << max_err << "\n";
    check("input grad matches numerical (max rel_err < 1e-3)", max_err < 1e-3);
}

// =====================================================================
// Test 6: input gradient check on a binary tree
// =====================================================================
static void test_input_grad_binary() {
    cout << "\n--- Test 6: input gradient check on binary tree (7 nodes) ---\n";
    std::vector<std::vector<int>> children = {
        {1, 2}, {3, 4}, {5, 6}, {}, {}, {}, {}
    };
    size_t in_dim = 3, hidden = 3;
    TreeLSTM cell(in_dim, hidden);
    Tensor input(7, in_dim);
    std::mt19937 gen(11);
    std::normal_distribution<> nd(0.0, 0.4);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = nd(gen);

    Tensor target(7, hidden);
    for (size_t i = 0; i < 7; ++i)
        for (size_t k = 0; k < hidden; ++k)
            target(i, k) = 0.2 * cos(0.3 * (i + 1)) + 0.05 * (k + 1);

    Tensor out = cell.forward(input, children);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    Tensor grad_x = cell.backward(grad_loss, 0.0);
    Tensor grad_x_num = numerical_grad_input(cell, input, children, target);

    double max_err = 0.0;
    for (size_t i = 0; i < grad_x.data.size(); ++i) {
        double e = rel_err(grad_x.data[i], grad_x_num.data[i]);
        if (e > max_err) max_err = e;
    }
    cout << "  max rel_err = " << max_err << "\n";
    check("input grad matches numerical on binary tree", max_err < 1e-3);
}

// =====================================================================
// Test 7: W gradient check
// =====================================================================
static void test_W_grad() {
    cout << "\n--- Test 7: W gradient check (chain) ---\n";
    std::vector<std::vector<int>> children = { {1}, {2}, {} };
    size_t in_dim = 2, hidden = 2;
    TreeLSTM cell(in_dim, hidden);
    Tensor input(3, in_dim);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);
    Tensor target(3, hidden);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05 * (i + 1);

    Tensor out = cell.forward(input, children);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    // Numerical W gradient: perturb each W element, recompute loss, finite-diff.
    double eps = 1e-5;
    double max_err = 0.0;
    for (size_t r = 0; r < cell.W.rows; ++r) {
        for (size_t c = 0; c < cell.W.cols; ++c) {
            double orig = cell.W(r, c);
            cell.W(r, c) = orig + eps;
            Tensor op = cell.forward(input, children);
            double lp = l2_loss_value(op, target);
            cell.W(r, c) = orig - eps;
            Tensor om = cell.forward(input, children);
            double lm = l2_loss_value(om, target);
            cell.W(r, c) = orig;
            double num = (lp - lm) / (2.0 * eps);
            double ana = cell.grad_W(r, c);
            double e = rel_err(ana, num);
            if (e > max_err) max_err = e;
        }
    }
    cout << "  W max rel_err = " << max_err << "\n";
    check("W grad matches numerical", max_err < 1e-3);
}

// =====================================================================
// Test 8: U gradient check
// =====================================================================
static void test_U_grad() {
    cout << "\n--- Test 8: U gradient check (chain) ---\n";
    std::vector<std::vector<int>> children = { {1}, {2}, {} };
    size_t in_dim = 2, hidden = 2;
    TreeLSTM cell(in_dim, hidden);
    Tensor input(3, in_dim);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);
    Tensor target(3, hidden);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05 * (i + 1);

    Tensor out = cell.forward(input, children);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    double eps = 1e-5;
    double max_err = 0.0;
    for (size_t r = 0; r < cell.U.rows; ++r) {
        for (size_t c = 0; c < cell.U.cols; ++c) {
            double orig = cell.U(r, c);
            cell.U(r, c) = orig + eps;
            Tensor op = cell.forward(input, children);
            double lp = l2_loss_value(op, target);
            cell.U(r, c) = orig - eps;
            Tensor om = cell.forward(input, children);
            double lm = l2_loss_value(om, target);
            cell.U(r, c) = orig;
            double num = (lp - lm) / (2.0 * eps);
            double ana = cell.grad_U(r, c);
            double e = rel_err(ana, num);
            if (e > max_err) max_err = e;
        }
    }
    cout << "  U max rel_err = " << max_err << "\n";
    check("U grad matches numerical", max_err < 1e-3);
}

// =====================================================================
// Test 9: b gradient check
// =====================================================================
static void test_b_grad() {
    cout << "\n--- Test 9: b gradient check (chain) ---\n";
    std::vector<std::vector<int>> children = { {1}, {2}, {} };
    size_t in_dim = 2, hidden = 2;
    TreeLSTM cell(in_dim, hidden);
    Tensor input(3, in_dim);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);
    Tensor target(3, hidden);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.05 * (i + 1);

    Tensor out = cell.forward(input, children);
    Tensor grad_loss = l2_loss_grad(out, target);
    cell.zero_grad();
    cell.backward(grad_loss, 0.0);

    double eps = 1e-5;
    double max_err = 0.0;
    for (size_t r = 0; r < cell.b.rows; ++r) {
        for (size_t c = 0; c < cell.b.cols; ++c) {
            double orig = cell.b(r, c);
            cell.b(r, c) = orig + eps;
            Tensor op = cell.forward(input, children);
            double lp = l2_loss_value(op, target);
            cell.b(r, c) = orig - eps;
            Tensor om = cell.forward(input, children);
            double lm = l2_loss_value(om, target);
            cell.b(r, c) = orig;
            double num = (lp - lm) / (2.0 * eps);
            double ana = cell.grad_b(r, c);
            double e = rel_err(ana, num);
            if (e > max_err) max_err = e;
        }
    }
    cout << "  b max rel_err = " << max_err << "\n";
    check("b grad matches numerical", max_err < 1e-3);
}

// =====================================================================
// Test 10: parameters / gradients shape consistency
// =====================================================================
static void test_param_shapes() {
    cout << "\n--- Test 10: parameters/gradients shape consistency ---\n";
    size_t in_dim = 3, hidden = 4;
    TreeLSTM cell(in_dim, hidden);
    auto p = cell.parameters();
    auto g = cell.gradients();
    check("3 parameters (W, U, b)", p.size() == 3);
    check("3 gradients", g.size() == 3);
    check("W shape (4H, in_dim)", p[0]->rows == 4 * hidden && p[0]->cols == in_dim);
    check("U shape (4H, H)",     p[1]->rows == 4 * hidden && p[1]->cols == hidden);
    check("b shape (1, 4H)",     p[2]->rows == 1           && p[2]->cols == 4 * hidden);
    check("grad_W shape == W shape", g[0]->rows == p[0]->rows && g[0]->cols == p[0]->cols);
    check("grad_U shape == U shape", g[1]->rows == p[1]->rows && g[1]->cols == p[1]->cols);
    check("grad_b shape == b shape", g[2]->rows == p[2]->rows && g[2]->cols == p[2]->cols);
}

// =====================================================================
// Test 11: forget-bias init = 1 convention
// =====================================================================
static void test_forget_bias_init() {
    cout << "\n--- Test 11: forget-bias init = 1 ---\n";
    size_t in_dim = 2, hidden = 3;
    TreeLSTM cell(in_dim, hidden);
    bool ok = true;
    for (size_t h = 0; h < hidden; ++h) {
        if (fabs(cell.b(0, hidden + h) - 1.0) > 1e-12) ok = false;
    }
    check("forget-gate bias init = 1.0", ok);
    // Other biases should be 0
    bool zero_others = true;
    for (size_t i = 0; i < 4 * hidden; ++i) {
        if (i >= hidden && i < 2 * hidden) continue;  // skip forget rows
        if (fabs(cell.b(0, i)) > 1e-12) zero_others = false;
    }
    check("other biases init = 0", zero_others);
}

// =====================================================================
// Test 12: training step reduces loss
// =====================================================================
static void test_training_step() {
    cout << "\n--- Test 12: training step reduces loss ---\n";
    // Use the TreeLSTMModel (cell + classifier). Training the bare cell
    // on all 9 hidden outputs from a 3-node chain is a high-dim regression
    // that converges slowly; the model (which only targets the root via
    // the classifier) is the natural training setup and trains fast.
    std::vector<std::vector<int>> children = { {1, 2}, {3}, {}, {} };
    size_t in_dim = 2, hidden = 3, out_dim = 2;
    TreeLSTMModel model(in_dim, hidden, out_dim);
    Tensor input(4, in_dim);
    std::mt19937 gen(13);
    std::normal_distribution<> nd(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = nd(gen);
    Tensor target(1, out_dim);
    target(0, 0) = 1.0;
    target(0, 1) = 0.0;

    double lr = 0.05;
    double loss_prev = 0.0;
    double loss_curr = 0.0;
    int steps = 50;
    for (int step = 0; step < steps; ++step) {
        Tensor out = model.forward_with_tree(input, children);
        loss_prev = l2_loss_value(out, target);
        Tensor grad_loss = l2_loss_grad(out, target);
        model.zero_grad();
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
    }
    Tensor out_final = model.forward_with_tree(input, children);
    loss_curr = l2_loss_value(out_final, target);
    cout << "  loss: " << loss_prev << " -> " << loss_curr
         << " (reduction " << 100.0 * (loss_prev - loss_curr) / max(loss_prev, 1e-8) << "%)\n";
    // With a small input (4 nodes, 2 dims) the cell output is already in a
    // narrow range and the loss starts small. We just check that ANY
    // training progress is made (the gradient is correctly flowing).
    check("training reduces loss by >5%", loss_curr < loss_prev * 0.95);
}

// =====================================================================
// Test 13: TreeLSTMModel end-to-end forward+backward on binary tree
// =====================================================================
static void test_model_end_to_end() {
    cout << "\n--- Test 13: TreeLSTMModel end-to-end ---\n";
    std::vector<std::vector<int>> children = {
        {1, 2}, {3, 4}, {5, 6}, {}, {}, {}, {}
    };
    size_t in_dim = 2, hidden = 3, out_dim = 2;
    TreeLSTMModel model(in_dim, hidden, out_dim);
    Tensor input(7, in_dim);
    std::mt19937 gen(17);
    std::normal_distribution<> nd(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = nd(gen);
    Tensor target(1, out_dim);
    target(0, 0) = 1.0;
    target(0, 1) = 0.0;

    Tensor out = model.forward_with_tree(input, children);
    check("model output shape (1, out_dim)", out.rows == 1 && out.cols == out_dim);

    Tensor grad_loss = l2_loss_grad(out, target);
    model.zero_grad();
    Tensor grad_x = model.backward(grad_loss, 0.0);
    check("grad_input shape (7, in_dim)", grad_x.rows == 7 && grad_x.cols == in_dim);

    // Numerical input gradient
    Tensor grad_x_num = numerical_grad_input_model(model, input, children, target);
    double max_err = 0.0;
    for (size_t i = 0; i < grad_x.data.size(); ++i) {
        double e = rel_err(grad_x.data[i], grad_x_num.data[i]);
        if (e > max_err) max_err = e;
    }
    cout << "  model input grad max rel_err = " << max_err << "\n";
    check("model input grad matches numerical", max_err < 1e-3);

    // End-to-end training
    double lr = 0.05;
    double l0 = 0.0, l1 = 0.0;
    for (int step = 0; step < 30; ++step) {
        Tensor o = model.forward_with_tree(input, children);
        if (step == 0) l0 = l2_loss_value(o, target);
        if (step == 29) l1 = l2_loss_value(o, target);
        Tensor gl = l2_loss_grad(o, target);
        model.zero_grad();
        model.backward(gl, 0.0);
        model.update_weights(lr);
    }
    cout << "  model loss: " << l0 << " -> " << l1
         << " (reduction " << 100.0 * (l0 - l1) / max(l0, 1e-8) << "%)\n";
    check("model training reduces loss by >30%", l1 < l0 * 0.7);
}

// =====================================================================
// Test 14: determinism — two fresh instances → identical output
// =====================================================================
static void test_determinism() {
    cout << "\n--- Test 14: determinism ---\n";
    std::vector<std::vector<int>> children = { {1, 2}, {3}, {}, {} };
    Tensor input(4, 2);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * (i + 1);
    TreeLSTM a(2, 3);
    TreeLSTM b(2, 3);
    Tensor oa = a.forward(input, children);
    Tensor ob = b.forward(input, children);
    bool same = true;
    for (size_t i = 0; i < oa.data.size(); ++i) {
        if (oa.data[i] != ob.data[i]) same = false;
    }
    check("two fresh instances produce identical output", same);
}

int main() {
    cout << "=== Tree-LSTM (Child-Sum) Tests ===\n";
    cout.setf(std::ios::unitbuf);
    test_chain_shape();
    test_binary_tree_shape();
    test_finite();
    test_hand_computed_reference();
    test_input_grad_chain();
    test_input_grad_binary();
    test_W_grad();
    test_U_grad();
    test_b_grad();
    test_param_shapes();
    test_forget_bias_init();
    test_training_step();
    test_model_end_to_end();
    test_determinism();

    cout << "\n=== Summary ===\n";
    cout << "passed: " << passed << " / " << (passed + failed) << "\n";
    cout << "failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
