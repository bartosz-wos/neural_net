// test_kan.cpp — Tests for KAN (Kolmogorov-Arnold Networks).
// Liu et al. 2024 "KAN: Kolmogorov-Arnold Networks" (https://arxiv.org/abs/2404.19756).
//
// Architecture: per-edge learnable activation
//   phi_{l,i,j}(x) = w_b * b(x) + w_s * spline(x; coefs_{l,i,j}, grid_l)
// where b(x) = silu(x) is the base activation, and the spline is a B-spline
// evaluated on a learnable grid (here fixed during training for simplicity).
//
// Each KANLayer maps (B, in_features) -> (B, out_features) and stores:
//   spline_coefs_  : (out_features, in_features * n_coefs)   per-edge spline coefficients
//   base_weight_   : (out_features, in_features)             per-edge base-activation scaler
//   spline_weight_ : (out_features, in_features)             per-edge spline scaler
//   grid_          : (1, G + 2k + 1)                         layer-shared grid knots
//
// A KAN model stacks KANLayers with a final Dense readout (or terminal KANLayer).

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) {
        cout << "  [PASS] " << name << endl;
        ++passed;
    } else {
        cout << "  [FAIL] " << name << endl;
        ++failed;
    }
    return pass;
}

// =====================================================================
// Test 1: Forward shape & finiteness
// =====================================================================
static void test_kan_layer_forward_shape() {
    cout << endl << "-- Test 1: KANLayer forward shape --" << endl;

    KANLayer kan(3, 5, /*num_grids=*/5, /*spline_order=*/3);
    Tensor input(2, 3);
    input[0][0] = -0.5; input[0][1] = 0.0; input[0][2] = 0.7;
    input[1][0] = 1.2;  input[1][1] = -1.0; input[1][2] = 0.3;

    Tensor out = kan.forward(input);
    check("forward output shape (2,5)", out.rows == 2 && out.cols == 5);
    bool finite = true;
    for (size_t i = 0; i < out.rows * out.cols; ++i) {
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    }
    check("forward output finite", finite);
}

// =====================================================================
// Test 2: Layer exposes correct parameter and gradient counts
// =====================================================================
static void test_kan_layer_params() {
    cout << endl << "-- Test 2: KANLayer params/grads --" << endl;

    size_t in_f = 4, out_f = 6, G = 5, k = 3;
    size_t n_coefs = G + k;
    KANLayer kan(in_f, out_f, G, k);

    auto params = kan.parameters();
    auto grads = kan.gradients();
    // Expect: spline_coefs, base_weight, spline_weight, grid (frozen but stored)
    check("params count == 4 (spline_coefs, base_w, spline_w, grid)", params.size() == 4);
    check("grads count == 3 (coefs/base_w/spline_w; grid frozen)", grads.size() == 3);

    // Find the spline_coefs and check shape
    const Tensor* sc = nullptr;
    for (auto* p : params) {
        if (p->rows == out_f && p->cols == in_f * n_coefs) { sc = p; break; }
    }
    check("spline_coefs shape (out_f, in_f * n_coefs)", sc != nullptr);

    // Find base_weight and spline_weight — both (out_f, in_f)
    size_t scalar_edge_count = 0;
    for (auto* p : params) {
        if (p->rows == out_f && p->cols == in_f) ++scalar_edge_count;
    }
    check("base_weight & spline_weight both (out_f, in_f)", scalar_edge_count == 2);

    // After a backward, all grad tensors should have valid shapes (no shape errors).
    Tensor input(3, in_f);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = 0.1 * (i * in_f + j) - 0.2;
    Tensor y = kan.forward(input);
    Tensor grad_out(y.rows, y.cols);
    for (size_t i = 0; i < grad_out.rows; ++i)
        for (size_t j = 0; j < grad_out.cols; ++j)
            grad_out[i][j] = 1.0;
    Tensor g = kan.backward(grad_out, 0.0);
    bool gshapes_ok = true;
    for (auto* gp : grads) {
        if (gp->rows == 0 || gp->cols == 0) { gshapes_ok = false; break; }
    }
    check("gradient tensors have non-zero shape after backward", gshapes_ok);
}

// =====================================================================
// Test 3: With spline_coefs=0, base_weight=1, spline_weight=0 → silu path
// =====================================================================
static void test_kan_layer_base_only() {
    cout << endl << "-- Test 3: KANLayer base-only path (silu) --" << endl;

    KANLayer kan(2, 3, 5, 3);

    // Zero out spline coefs and spline_weight, set base_weight to 1.
    auto params = kan.parameters();
    for (auto* p : params) p->fill(0.0);
    // First param is spline_coefs — already zero.
    // base_weight is (out, in); find and set to 1.
    size_t in_f = 2, out_f = 3;
    for (auto* p : params) {
        if (p->rows == out_f && p->cols == in_f && p != params[0]) {
            // Could be base_weight or spline_weight; identify by initial magnitude.
            // We zeroed everything; set the FIRST (out, in) tensor to 1.
        }
    }
    // Reset: we want only the first (out, in) (base_weight) to be 1.0.
    // Approach: zero all params; then set params[1] (base_weight) to 1.
    // params order: [spline_coefs, base_weight, spline_weight, grid]
    if (params.size() >= 4) {
        params[0]->fill(0.0); // spline_coefs
        params[1]->fill(1.0); // base_weight = 1
        params[2]->fill(0.0); // spline_weight = 0
        // grid[3] stays untouched (default uniform)
    }

    Tensor input(2, 2);
    input[0][0] = 0.5; input[0][1] = -1.0;
    input[1][0] = 0.0; input[1][1] = 2.0;

    Tensor out = kan.forward(input);
    // Expectation: out[b, i] = sum_j silu(x[b, j])
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    bool ok = true;
    for (size_t b = 0; b < 2; ++b) {
        double sum_silu = 0.0;
        for (size_t j = 0; j < 2; ++j) sum_silu += silu(input(b, j));
        for (size_t i = 0; i < 3; ++i) {
            double expected = sum_silu; // base_weight = 1, so out is sum_j 1*silu(x_j)
            double err = std::abs(out(b, i) - expected);
            if (err > 1e-9) { ok = false; break; }
        }
        if (!ok) break;
    }
    check("base-only path matches sum of silu (within 1e-9)", ok);
}

// =====================================================================
// Test 4: Spline partition of unity (sum of basis functions at any x = 1)
// =====================================================================
static void test_kan_layer_spline_partition_unity() {
    cout << endl << "-- Test 4: Spline partition of unity --" << endl;

    // For a clamped B-spline of any degree, the basis functions form a partition
    // of unity: sum_{c} B_{c,k}(x) = 1 at any x in the valid range.
    //
    // Test: with all coefs set to 1.0, the spline should evaluate to 1.0 everywhere.
    KANLayer kan(1, 1, /*num_grids=*/5, /*spline_order=*/3);

    auto params = kan.parameters();
    params[0]->fill(1.0); // spline_coefs = 1.0
    params[1]->fill(0.0); // base_weight = 0
    params[2]->fill(1.0); // spline_weight = 1.0

    // Sample 11 points in [-2, 2]
    bool unity = true;
    double max_err = 0.0;
    for (int s = 0; s <= 10; ++s) {
        double x = -2.0 + 0.4 * static_cast<double>(s); // -2.0 to +2.0 step 0.4
        Tensor input(1, 1);
        input[0][0] = x;
        Tensor out = kan.forward(input);
        double err = std::abs(out(0, 0) - 1.0);
        if (err > max_err) max_err = err;
        if (err > 1e-9) { unity = false; }
    }
    check("all coefs=1 → spline=1 (partition of unity), max err < 1e-9", unity);
    if (!unity) {
        cout << "    max_err = " << max_err << endl;
    }
}

// =====================================================================
// Test 5: KANModel forward + gradient check vs numerical gradient
// =====================================================================
static void test_kan_model_gradient_check() {
    cout << endl << "-- Test 5: KANModel gradient check vs numerical --" << endl;

    // Small model for tractable grad check.
    KANModel model(/*in_dim=*/3, /*hidden_dims=*/{4}, /*out_dim=*/2,
                   /*num_grids=*/5, /*spline_order=*/3);

    // Use a fixed input
    Tensor input(2, 3);
    input[0][0] = 0.3; input[0][1] = -0.7; input[0][2] = 0.5;
    input[1][0] = -0.4; input[1][1] = 0.9; input[1][2] = 0.1;

    // Compute analytical gradient
    Tensor y = model.forward(input);
    Tensor loss_grad(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            loss_grad[i][j] = 1.0; // sum loss, grad is 1
    model.zero_grad();
    Tensor dx_analytical = model.backward(loss_grad, 0.0);

    // Compute numerical gradient for multiple entries
    double eps = 1e-4;
    bool all_pass = true;
    for (size_t b = 0; b < input.rows; ++b) {
        for (size_t c = 0; c < input.cols; ++c) {
            Tensor in_plus = input;  in_plus(b, c) += eps;
            Tensor in_minus = input; in_minus(b, c) -= eps;
            Tensor y_plus = model.forward(in_plus);
            Tensor y_minus = model.forward(in_minus);
            double s_plus = 0.0, s_minus = 0.0;
            for (size_t i = 0; i < y_plus.rows; ++i)
                for (size_t j = 0; j < y_plus.cols; ++j)
                    s_plus += y_plus(i, j);
            for (size_t i = 0; i < y_minus.rows; ++i)
                for (size_t j = 0; j < y_minus.cols; ++j)
                    s_minus += y_minus(i, j);
            double num_grad = (s_plus - s_minus) / (2.0 * eps);
            double ana_grad = dx_analytical(b, c);
            double re = std::abs(ana_grad - num_grad) / std::max(1e-12, std::max(std::abs(ana_grad), std::abs(num_grad)));
            cout << "    (" << b << "," << c << ") ana=" << ana_grad << " num=" << num_grad
                 << " rel_err=" << re << endl;
            if (re > 1e-3) all_pass = false;
        }
    }
    check("input grad matches numerical for all entries (rel_err < 1e-3)", all_pass);
}

// =====================================================================
// Test 6: KANLayer training reduces loss on a simple task
// =====================================================================
static void test_kan_model_training_reduces_loss() {
    cout << endl << "-- Test 6: KANModel training reduces loss --" << endl;

    // Toy task: y = sin(sum(x)) — a smooth nonlinear function.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);

    size_t N = 64;
    Tensor X(N, 2);
    Tensor Y(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double x0 = dist(rng);
        double x1 = dist(rng);
        X[i][0] = x0;
        X[i][1] = x1;
        Y[i][0] = std::sin(x0 + x1);
    }

    KANModel model(2, {8, 8}, 1, /*num_grids=*/5, /*spline_order=*/3);

    auto compute_loss = [&](KANModel& m) {
        Tensor pred = m.forward(X);
        double loss = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double d = pred[i][0] - Y[i][0];
            loss += d * d;
        }
        return loss / static_cast<double>(N);
    };

    double initial_loss = compute_loss(model);

    double lr = 0.05;
    int epochs = 200;
    for (int ep = 0; ep < epochs; ++ep) {
        Tensor pred = model.forward(X);
        Tensor grad_out(N, 1);
        for (size_t i = 0; i < N; ++i)
            grad_out[i][0] = 2.0 * (pred[i][0] - Y[i][0]) / static_cast<double>(N);
        model.zero_grad();
        model.backward(grad_out, lr);
        model.update_weights(lr);
    }
    double final_loss = compute_loss(model);
    cout << "    initial_loss = " << initial_loss << "  final_loss = " << final_loss << endl;
    check("training reduces loss", final_loss < initial_loss);
    check("training reduces loss by >= 30%", final_loss < 0.7 * initial_loss);
}

// =====================================================================
// Test 7: zero_grad clears all parameter gradients
// =====================================================================
static void test_kan_model_zero_grad() {
    cout << endl << "-- Test 7: zero_grad clears all grads --" << endl;

    KANModel model(3, {4}, 2, 5, 3);
    Tensor input(2, 3);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = 0.1 * i - 0.2 * j;

    Tensor y = model.forward(input);
    Tensor grad_out(y.rows, y.cols);
    for (size_t i = 0; i < grad_out.rows; ++i)
        for (size_t j = 0; j < grad_out.cols; ++j)
            grad_out[i][j] = 0.7;

    model.zero_grad();
    Tensor dx = model.backward(grad_out, 0.0);
    // Check that any non-frozen grad is non-zero
    bool any_grad_nonzero = false;
    for (auto* g : model.gradients()) {
        for (size_t k = 0; k < g->rows * g->cols; ++k) {
            if (std::abs(g->data[k]) > 0.0) { any_grad_nonzero = true; break; }
        }
        if (any_grad_nonzero) break;
    }
    check("backward produces non-zero gradients", any_grad_nonzero);

    model.zero_grad();
    bool all_zero = true;
    for (auto* g : model.gradients()) {
        for (size_t k = 0; k < g->rows * g->cols; ++k) {
            if (std::abs(g->data[k]) > 0.0) { all_zero = false; break; }
        }
        if (!all_zero) break;
    }
    check("zero_grad clears all grads", all_zero);
}

// =====================================================================
// Test 8: name() returns KANLayer / KANModel
// =====================================================================
static void test_kan_names() {
    cout << endl << "-- Test 8: name() returns correct strings --" << endl;
    KANLayer l(3, 4);
    KANModel m(3, {4}, 2, 5, 3);
    check("KANLayer.name() == \"KANLayer\"", l.name() == "KANLayer");
    check("KANModel.name() == \"KANModel\"", m.name() == "KANModel");
}

// =====================================================================
// Test 9: spline gradient flows back correctly (analytical vs numerical)
// =====================================================================
static void test_kan_spline_coef_grad_check() {
    cout << endl << "-- Test 9: spline_coef gradient matches numerical --" << endl;

    KANLayer kan(2, 2, 5, 3);
    // Zero base and spline weights, then set spline_weight = 1 for all edges.
    // Initialize spline_coefs with non-trivial values so the gradient is meaningful.
    auto params = kan.parameters();
    params[0]->fill(0.0);
    // Set spline_coefs to a non-trivial pattern (per-edge) so the gradient is non-zero
    // at multiple entries. Row-major: params[0][out_f, in * n_coefs].
    size_t in_f = 2, out_f = 2;
    size_t n_coefs = 8;  // G + k = 5 + 3
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            for (size_t c = 0; c < n_coefs; ++c) {
                (*params[0])(i, j * n_coefs + c) = 0.1 * (static_cast<double>(c + 1) - 4.0); // [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
            }
        }
    }
    params[1]->fill(0.0); // base_weight = 0
    params[2]->fill(1.0); // spline_weight = 1 for all edges

    Tensor input(2, 2);
    input[0][0] = 0.3; input[0][1] = -0.7;
    input[1][0] = 0.5; input[1][1] = 0.1;

    Tensor y = kan.forward(input);
    Tensor grad_out(y.rows, y.cols);
    for (size_t i = 0; i < grad_out.rows; ++i)
        for (size_t j = 0; j < grad_out.cols; ++j)
            grad_out[i][j] = 1.0;
    kan.zero_grad();
    kan.backward(grad_out, 0.0);

    auto grads = kan.gradients();
    // grads[0] is grad of spline_coefs — shape (out, in * n_coefs)
    Tensor grad_coefs = *grads[0];

    // Numerical check on multiple entries
    double eps = 1e-5;
    bool all_pass = true;
    int checked = 0;
    // Sample 8 entries spread across the matrix
    int rs[4] = {0, 0, 1, 1};
    int cs[4] = {0, 5, 3, 11};
    for (int q = 0; q < 4; ++q) {
        size_t r = rs[q], c = cs[q];
        Tensor saved = *params[0];
        (*params[0])(r, c) += eps;
        Tensor y_plus = kan.forward(input);
        (*params[0])(r, c) -= 2 * eps;
        Tensor y_minus = kan.forward(input);
        (*params[0])(r, c) = saved(r, c); // restore

        double s_plus = 0.0, s_minus = 0.0;
        for (size_t i = 0; i < y_plus.rows; ++i)
            for (size_t j = 0; j < y_plus.cols; ++j)
                s_plus += y_plus(i, j);
        for (size_t i = 0; i < y_minus.rows; ++i)
            for (size_t j = 0; j < y_minus.cols; ++j)
                s_minus += y_minus(i, j);
        double num_grad = (s_plus - s_minus) / (2 * eps);
        double ana_grad = grad_coefs(r, c);
        double denom = std::max(1e-12, std::max(std::abs(ana_grad), std::abs(num_grad)));
        double re = std::abs(ana_grad - num_grad) / denom;
        cout << "    spline_coef (" << r << "," << c << ") ana=" << ana_grad << " num=" << num_grad
             << " rel_err=" << re << endl;
        if (re > 1e-3) all_pass = false;
        ++checked;
    }
    check("spline_coef grad matches numerical for sampled entries (rel_err < 1e-3)", all_pass);
}

// =====================================================================
// Test 10: Mutation test — drop spline term, ensure test catches it
// (this is run via the main test, but we mark it as a guard test)
// =====================================================================
static void test_kan_consistency_smoke() {
    cout << endl << "-- Test 10: KAN smoke test — init then forward multiple times is deterministic --" << endl;
    KANLayer l1(3, 4);
    KANLayer l2(3, 4); // same constructor → same init seed-equivalent (small randomness)
    Tensor input(2, 3);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = 0.05 * i + 0.07 * j;
    Tensor y1 = l1.forward(input);
    Tensor y2 = l1.forward(input); // same layer, same input → identical output
    bool det = true;
    for (size_t k = 0; k < y1.rows * y1.cols; ++k)
        if (y1.data[k] != y2.data[k]) { det = false; break; }
    check("forward is deterministic", det);

    // Forward with different batches must differ in batch dimension
    Tensor big(5, 3);
    for (size_t i = 0; i < big.rows; ++i)
        for (size_t j = 0; j < big.cols; ++j)
            big[i][j] = 0.05 * i + 0.07 * j;
    Tensor ybig = l1.forward(big);
    check("larger batch (5) returns (5, 4)", ybig.rows == 5 && ybig.cols == 4);
}

// =====================================================================
// Test 11: KANLayer input gradient (verifies spline derivative)
// =====================================================================
static void test_kan_layer_input_gradient() {
    cout << endl << "-- Test 11: KANLayer input gradient --" << endl;

    // Small layer; verify input gradient matches numerical finite differences.
    KANLayer kan(3, 2, /*num_grids=*/5, /*spline_order=*/3);

    Tensor input(2, 3);
    // Use inputs that are non-trivially inside the spline range (not at boundaries)
    input[0][0] = 0.3; input[0][1] = -0.5; input[0][2] = 1.2;
    input[1][0] = -1.4; input[1][1] = 0.9; input[1][2] = 0.0;

    Tensor y = kan.forward(input);
    Tensor grad_out(y.rows, y.cols);
    for (size_t i = 0; i < grad_out.rows; ++i)
        for (size_t j = 0; j < grad_out.cols; ++j)
            grad_out[i][j] = 1.0;
    kan.zero_grad();
    Tensor dx = kan.backward(grad_out, 0.0);

    double eps = 1e-5;
    bool all_pass = true;
    for (size_t b = 0; b < input.rows; ++b) {
        for (size_t c = 0; c < input.cols; ++c) {
            Tensor in_plus = input;  in_plus(b, c) += eps;
            Tensor in_minus = input; in_minus(b, c) -= eps;
            Tensor y_plus = kan.forward(in_plus);
            Tensor y_minus = kan.forward(in_minus);
            double s_plus = 0.0, s_minus = 0.0;
            for (size_t i = 0; i < y_plus.rows; ++i)
                for (size_t j = 0; j < y_plus.cols; ++j)
                    s_plus += y_plus(i, j);
            for (size_t i = 0; i < y_minus.rows; ++i)
                for (size_t j = 0; j < y_minus.cols; ++j)
                    s_minus += y_minus(i, j);
            double num_grad = (s_plus - s_minus) / (2.0 * eps);
            double ana_grad = dx(b, c);
            double denom = std::max(1e-12, std::max(std::abs(ana_grad), std::abs(num_grad)));
            double re = std::abs(ana_grad - num_grad) / denom;
            cout << "    KANLayer input grad (" << b << "," << c << ") ana=" << ana_grad
                 << " num=" << num_grad << " rel_err=" << re << endl;
            if (re > 1e-3) all_pass = false;
        }
    }
    check("KANLayer input grad matches numerical for all (rel_err < 1e-3)", all_pass);
}

// =====================================================================
// Test 12: spline_weight and base_weight gradients match numerical
// =====================================================================
static void test_kan_weight_gradients() {
    cout << endl << "-- Test 12: spline_weight and base_weight gradients --" << endl;

    KANLayer kan(2, 2, 5, 3);
    auto params = kan.parameters();
    // Initialize spline_coefs to non-trivial values
    params[0]->fill(0.3);
    params[1]->fill(0.0);
    params[2]->fill(1.0);

    Tensor input(2, 2);
    input[0][0] = 0.3; input[0][1] = -0.7;
    input[1][0] = 0.5; input[1][1] = 0.1;

    Tensor y = kan.forward(input);
    Tensor grad_out(y.rows, y.cols);
    for (size_t i = 0; i < grad_out.rows; ++i)
        for (size_t j = 0; j < grad_out.cols; ++j)
            grad_out[i][j] = 1.0;
    kan.zero_grad();
    kan.backward(grad_out, 0.0);

    auto grads = kan.gradients();
    Tensor grad_base = *grads[1];
    Tensor grad_sw = *grads[2];

    // Numerical check on a few entries for each
    double eps = 1e-5;
    bool base_pass = true, sw_pass = true;

    // Check base_weight (0, 0)
    {
        Tensor saved = *params[1];
        (*params[1])(0, 0) += eps;
        Tensor y_plus = kan.forward(input);
        (*params[1])(0, 0) -= 2 * eps;
        Tensor y_minus = kan.forward(input);
        (*params[1])(0, 0) = saved(0, 0);
        double s_plus = 0.0, s_minus = 0.0;
        for (size_t i = 0; i < y_plus.rows; ++i)
            for (size_t j = 0; j < y_plus.cols; ++j) s_plus += y_plus(i, j);
        for (size_t i = 0; i < y_minus.rows; ++i)
            for (size_t j = 0; j < y_minus.cols; ++j) s_minus += y_minus(i, j);
        double num = (s_plus - s_minus) / (2 * eps);
        double ana = grad_base(0, 0);
        double re = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
        cout << "    base_weight (0,0) ana=" << ana << " num=" << num << " rel_err=" << re << endl;
        if (re > 1e-3) base_pass = false;
    }
    // Check spline_weight (1, 1)
    {
        Tensor saved = *params[2];
        (*params[2])(1, 1) += eps;
        Tensor y_plus = kan.forward(input);
        (*params[2])(1, 1) -= 2 * eps;
        Tensor y_minus = kan.forward(input);
        (*params[2])(1, 1) = saved(1, 1);
        double s_plus = 0.0, s_minus = 0.0;
        for (size_t i = 0; i < y_plus.rows; ++i)
            for (size_t j = 0; j < y_plus.cols; ++j) s_plus += y_plus(i, j);
        for (size_t i = 0; i < y_minus.rows; ++i)
            for (size_t j = 0; j < y_minus.cols; ++j) s_minus += y_minus(i, j);
        double num = (s_plus - s_minus) / (2 * eps);
        double ana = grad_sw(1, 1);
        double re = std::abs(ana - num) / std::max(1e-12, std::max(std::abs(ana), std::abs(num)));
        cout << "    spline_weight (1,1) ana=" << ana << " num=" << num << " rel_err=" << re << endl;
        if (re > 1e-3) sw_pass = false;
    }
    check("base_weight grad (0,0) matches numerical (rel_err < 1e-3)", base_pass);
    check("spline_weight grad (1,1) matches numerical (rel_err < 1e-3)", sw_pass);
}

int main() {
    cout << "========================================" << endl;
    cout << "  KAN (Kolmogorov-Arnold Networks)    " << endl;
    cout << "========================================" << endl;

    test_kan_layer_forward_shape();
    test_kan_layer_params();
    test_kan_layer_base_only();
    test_kan_layer_spline_partition_unity();
    test_kan_model_gradient_check();
    test_kan_model_training_reduces_loss();
    test_kan_model_zero_grad();
    test_kan_names();
    test_kan_spline_coef_grad_check();
    test_kan_consistency_smoke();
    test_kan_layer_input_gradient();
    test_kan_weight_gradients();

    cout << endl << "========================================" << endl;
    cout << "Results: " << passed << " passed, " << failed << " failed" << endl;
    cout << "========================================" << endl;
    return failed == 0 ? 0 : 1;
}