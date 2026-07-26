// test_spectral_norm.cpp — Test coverage for SpectralNorm
//
// SpectralNorm (Miyato et al. 2018, SNGAN) constrains the spectral norm
// (largest singular value) of a weight matrix to ~1 by maintaining power-
// iteration vectors u and v and rescaling W by u^T W v / ||u|| / ||v||.
//
// These tests verify the public surface area:
//   - Constructor shape checks, gamma initial state, normalizes u/v on init
//   - Power iteration approximates largest singular value on identity, diagonal
//   - Power iteration on a known singular matrix (singular = 5)
//   - Forward shape, finite outputs, gamma scales output
//   - Backward (grad_output @ last_W) returns correct shape
//   - Parameter access (only gamma), gradient access (empty), name()
//   - Determinism with same u/v init
//   - update_weights is a no-op (won't crash on gamma)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>

#include "nn/layers/normalization/spectral_norm.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static bool feq(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

// =====================================================================
// Test 1: Constructor — shape checks, initial values, normalize u/v
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: SpectralNorm constructor --" << endl;

    srand(123);
    Tensor W(3, 4);
    W[0][0] = 1.0; W[0][1] = 0.5; W[0][2] = 0.0; W[0][3] = -0.3;
    W[1][0] = 0.2; W[1][1] = 1.0; W[1][2] = 0.4; W[1][3] = 0.0;
    W[2][0] = 0.0; W[2][1] = 0.3; W[2][2] = 1.0; W[2][3] = 0.7;

    SpectralNorm sn(W, 5, 1e-6);

    // gamma initialized to 1.0 (scalar, 1x1 tensor)
    check("gamma shape (1, 1)", sn.gamma.rows == 1 && sn.gamma.cols == 1);
    check("gamma[0][0] = 1.0", feq(sn.gamma[0][0], 1.0));

    // u is (rows, 1), v is (cols, 1)
    check("u shape (rows=W.rows, 1)", sn.u.rows == W.rows && sn.u.cols == 1);
    check("v shape (cols=W.cols, 1)", sn.v.rows == W.cols && sn.v.cols == 1);

    // u and v are normalized after init (Euclidean norm == 1 to eps tolerance)
    double u_norm_sq = 0.0;
    for (size_t i = 0; i < sn.u.rows; ++i) u_norm_sq += sn.u[i][0] * sn.u[i][0];
    check("u is unit vector after init", feq(u_norm_sq, 1.0, 1e-5));

    double v_norm_sq = 0.0;
    for (size_t i = 0; i < sn.v.rows; ++i) v_norm_sq += sn.v[i][0] * sn.v[i][0];
    check("v is unit vector after init", feq(v_norm_sq, 1.0, 1e-5));

    // power_iterations and eps stored
    check("power_iterations = 5", sn.power_iterations == 5);
    check("eps = 1e-6", feq(sn.eps, 1e-6));

    // Empty weight matrix throws
    Tensor empty_W(0, 4);
    bool threw = false;
    try { SpectralNorm bad(empty_W); }
    catch (std::invalid_argument&) { threw = true; }
    check("Constructor throws on empty W (rows=0)", threw);

    threw = false;
    Tensor empty_W2(3, 0);
    try { SpectralNorm bad2(empty_W2); }
    catch (std::invalid_argument&) { threw = true; }
    check("Constructor throws on empty W (cols=0)", threw);
}

// =====================================================================
// Test 2: Power iteration — sigma approximates the largest singular value
// =====================================================================
static void test_spectral_norm_identity() {
    cout << endl << "-- Test 2: Spectral norm = 1 for identity W --" << endl;

    srand(42);
    Tensor W = Tensor::zeros(3, 3);
    for (size_t i = 0; i < 3; ++i) W[i][i] = 1.0;

    SpectralNorm sn(W, 50, 1e-8);  // many iterations for tight approximation
    double sigma = sn.compute_spectral_norm();
    check("σ(I_3) ≈ 1.0 (within 1e-2)", feq(sigma, 1.0, 1e-2));

    // For diagonal W with [2, 3, 0.5], largest singular value is 3.
    Tensor W2 = Tensor::zeros(3, 3);
    W2[0][0] = 2.0;
    W2[1][1] = 3.0;
    W2[2][2] = 0.5;
    SpectralNorm sn2(W2, 50, 1e-8);
    double sigma2 = sn2.compute_spectral_norm();
    check("σ(diag([2,3,0.5])) ≈ 3.0 (within 1e-2)", feq(sigma2, 3.0, 1e-2));
}

// =====================================================================
// Test 3: Spectral norm accuracy on a matrix with known singular value
// =====================================================================
static void test_spectral_norm_known() {
    cout << endl << "-- Test 3: Spectral norm on matrix with σ=5 --" << endl;

    // Construct W = U Σ V^T with U = V = identity (so W is exactly diagonal)
    // and Σ = diag([5, 1, 0.5]) → largest singular value = 5.
    srand(7);
    Tensor W = Tensor::zeros(3, 3);
    W[0][0] = 5.0;
    W[1][1] = 1.0;
    W[2][2] = 0.5;

    SpectralNorm sn(W, 100, 1e-10);  // tight tolerance
    double sigma = sn.compute_spectral_norm();
    check("σ(diag([5, 1, 0.5])) ≈ 5.0 (within 1e-3)",
          feq(sigma, 5.0, 1e-3));
}

// =====================================================================
// Test 4: Forward pass — shape, finiteness, gamma scaling
// =====================================================================
static void test_forward() {
    cout << endl << "-- Test 4: Forward pass shape and gamma scaling --" << endl;

    srand(99);
    Tensor W = Tensor::zeros(4, 4);
    for (size_t i = 0; i < 4; ++i) W[i][i] = 2.0;

    SpectralNorm sn(W, 5);

    Tensor input(3, 4);  // 3 batch, 4 features (in_features = W.cols = 4)
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            input[i][j] = (i + 1) * (j + 1) * 0.1;

    Tensor out = sn.forward(input);
    check("Forward output shape (batch, in_features) = (3, 4)",
          out.rows == 3 && out.cols == 4);

    bool finite = true;
    for (size_t i = 0; i < out.rows && finite; ++i)
        for (size_t j = 0; j < out.cols && finite; ++j)
            if (!std::isfinite(out[i][j])) finite = false;
    check("Forward output all finite", finite);

    // For W = 2*I_4, σ(W) = 2 (max singular value), so
    //   W_normalized = γ * W / σ = 1 * (2I) / 2 = I
    // output = input @ I^T = input
    bool identity_output = true;
    for (size_t i = 0; i < 3 && identity_output; ++i)
        for (size_t j = 0; j < 4 && identity_output; ++j)
            if (!feq(out[i][j], input[i][j], 1e-3))
                identity_output = false;
    check("Forward output = input (σ=2 normalization of 2*I → I)",
          identity_output);

    // Manually setting gamma = 4 scales by 4x: out = 4 * input @ I^T = 4 * input
    sn.gamma[0][0] = 4.0;
    Tensor out2 = sn.forward(input);
    bool fourx_input = true;
    for (size_t i = 0; i < 3 && fourx_input; ++i)
        for (size_t j = 0; j < 4 && fourx_input; ++j)
            if (!feq(out2[i][j], 4.0 * input[i][j], 1e-3))
                fourx_input = false;
    check("Forward output = 4 * input (gamma=4 → 4× identity scaling)",
          fourx_input);
}

// =====================================================================
// Test 5: Backward — shape correctness (grad_output @ last_W)
// =====================================================================
static void test_backward() {
    cout << endl << "-- Test 5: Backward pass shape --" << endl;

    srand(11);
    Tensor W(2, 3);  // out_features=2, in_features=3
    W[0][0] = 1.0; W[0][1] = 0.5; W[0][2] = 0.0;
    W[1][0] = 0.0; W[1][1] = 1.0; W[1][2] = -0.5;
    SpectralNorm sn(W, 5);

    Tensor input(2, 3);  // batch=2, in=3
    input[0][0] = 1.0; input[0][1] = 0.0; input[0][2] = 0.0;
    input[1][0] = 0.0; input[1][1] = 1.0; input[1][2] = 0.0;
    Tensor out = sn.forward(input);
    check("Pre-backward: forward shape (2, 2)",
          out.rows == 2 && out.cols == 2);

    Tensor grad_output(2, 2);
    grad_output[0][0] = 1.0; grad_output[0][1] = 0.0;
    grad_output[1][0] = 0.0; grad_output[1][1] = 1.0;

    Tensor grad_input = sn.backward(grad_output, 0.001);

    // Backward returns grad_output @ last_W
    //   grad_output: (2, 2); last_W: (2, 3); → grad_input: (2, 3)
    check("Backward grad_input shape (batch=2, in_features=3)",
          grad_input.rows == 2 && grad_input.cols == 3);

    bool finite = true;
    for (size_t i = 0; i < grad_input.rows && finite; ++i)
        for (size_t j = 0; j < grad_input.cols && finite; ++j)
            if (!std::isfinite(grad_input[i][j])) finite = false;
    check("Backward grad_input all finite", finite);
}

// =====================================================================
// Test 6: update_weights is a no-op
// =====================================================================
static void test_update_weights_noop() {
    cout << endl << "-- Test 6: update_weights is no-op --" << endl;

    srand(31);
    Tensor W(3, 3);
    W[0][0] = 2.0; W[1][1] = 3.0; W[2][2] = 4.0;
    SpectralNorm sn(W, 1);

    // Capture state
    double orig_gamma = sn.gamma[0][0];
    size_t orig_W_size = sn.W.rows * sn.W.cols;
    Tensor orig_W = sn.W.clone();

    bool threw = false;
    try { sn.update_weights(0.01); }
    catch (...) { threw = true; }
    check("update_weights does not throw", !threw);

    // gamma and W are untouched (update is a no-op per the implementation)
    check("gamma unchanged after update_weights",
          feq(sn.gamma[0][0], orig_gamma));
    check("W size unchanged after update_weights",
          sn.W.rows * sn.W.cols == orig_W_size);
    bool W_unchanged = true;
    for (size_t i = 0; i < sn.W.rows && W_unchanged; ++i)
        for (size_t j = 0; j < sn.W.cols && W_unchanged; ++j)
            if (!feq(sn.W[i][j], orig_W[i][j])) W_unchanged = false;
    check("W values unchanged after update_weights (noop contract)", W_unchanged);
}

// =====================================================================
// Test 7: Parameter / gradient / name accessors
// =====================================================================
static void test_accessors() {
    cout << endl << "-- Test 7: parameters(), gradients(), name() --" << endl;

    srand(55);
    Tensor W(2, 2);
    W[0][0] = 1.0; W[0][1] = 0.5;
    W[1][0] = 0.5; W[1][1] = 1.0;
    SpectralNorm sn(W);

    auto params = sn.parameters();
    check("parameters() returns exactly one Tensor", params.size() == 1);
    check("parameters()[0] == &gamma", params[0] == &sn.gamma);

    auto grads = sn.gradients();
    check("gradients() returns empty vector (no local grad buffer)",
          grads.empty());

    check("name() == \"SpectralNorm\"", sn.name() == "SpectralNorm");

    // get_weights returns last_W (normalized weight)
    sn.forward(Tensor(1, 2));  // populate last_W
    Tensor w = sn.get_weights();
    check("get_weights shape == W (out, in)",
          w.rows == W.rows && w.cols == W.cols);

    // get_gradients returns empty Tensor (per header signature)
    Tensor g = sn.get_gradients();
    check("get_gradients returns empty Tensor (rows=cols=0)",
          g.rows == 0 && g.cols == 0);

    // zero_grad doesn't throw (it's a no-op)
    bool zg_threw = false;
    try { sn.zero_grad(); }
    catch (...) { zg_threw = true; }
    check("zero_grad does not throw", !zg_threw);
}

// =====================================================================
// Test 8: Determinism with same init
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 8: Determinism — same seed produces same result --" << endl;

    Tensor W(3, 4);
    W[0][0] = 1.0; W[0][1] = 2.0; W[0][2] = 3.0; W[0][3] = 4.0;
    W[1][0] = 0.5; W[1][1] = 1.5; W[1][2] = 2.5; W[1][3] = 3.5;
    W[2][0] = 0.0; W[2][1] = 1.0; W[2][2] = 2.0; W[2][3] = 3.0;

    srand(100);
    SpectralNorm sn1(W, 5);
    Tensor input(2, 4);
    input[0][0] = 1.0; input[0][1] = 1.0; input[0][2] = 1.0; input[0][3] = 1.0;
    input[1][0] = 0.5; input[1][1] = 0.5; input[1][2] = 0.5; input[1][3] = 0.5;
    Tensor out1 = sn1.forward(input);

    srand(100);  // same seed
    SpectralNorm sn2(W, 5);
    Tensor out2 = sn2.forward(input);

    bool same = true;
    for (size_t i = 0; i < out1.rows && same; ++i)
        for (size_t j = 0; j < out1.cols && same; ++j)
            if (!feq(out1[i][j], out2[i][j])) same = false;
    check("Same seed → bit-identical forward output", same);
}

// =====================================================================
// Test 9: normalize() static helper
// =====================================================================
static void test_normalize() {
    cout << endl << "-- Test 9: normalize() static helper --" << endl;

    // Column vector with norm sqrt(4+9+16) = sqrt(29) ≈ 5.385
    Tensor v(3, 1);
    v[0][0] = 2.0;
    v[1][0] = 3.0;
    v[2][0] = 4.0;

    Tensor n = SpectralNorm::normalize(v);
    check("normalize output shape (3, 1)",
          n.rows == 3 && n.cols == 1);

    double norm_sq = 0.0;
    for (size_t i = 0; i < n.rows; ++i)
        norm_sq += n[i][0] * n[i][0];
    check("normalize output has unit L2 norm", feq(std::sqrt(norm_sq), 1.0, 1e-9));
    check("normalize[0][0] = 2/√29", feq(n[0][0], 2.0 / std::sqrt(29.0), 1e-9));
    check("normalize[1][0] = 3/√29", feq(n[1][0], 3.0 / std::sqrt(29.0), 1e-9));
    check("normalize[2][0] = 4/√29", feq(n[2][0], 4.0 / std::sqrt(29.0), 1e-9));

    // Zero vector stays zero (norm < eps branch)
    Tensor z = Tensor::zeros(3, 1);
    Tensor nz = SpectralNorm::normalize(z);
    check("normalize(0) returns 0",
          feq(nz[0][0], 0.0) && feq(nz[1][0], 0.0) && feq(nz[2][0], 0.0));
}

// =====================================================================
// Test 10: End-to-end — Lipschitz behavior (bound on operator norm)
// =====================================================================
static void test_end_to_end_lipschitz() {
    cout << endl << "-- Test 10: End-to-end Lipschitz bound --" << endl;

    // A 2x3 weight matrix.
    srand(2024);
    Tensor W(2, 3);
    W[0][0] = 1.7;  W[0][1] = -0.5; W[0][2] = 0.9;
    W[1][0] = 0.3;  W[1][1] = 1.1;  W[1][2] = -0.7;

    SpectralNorm sn(W, 50, 1e-8);
    double sigma = sn.compute_spectral_norm();

    // Apply to a unit vector → output L2 norm should equal σ (operator norm).
    Tensor unit(1, 3);
    unit[0][0] = 1.0 / std::sqrt(3.0);
    unit[0][1] = 1.0 / std::sqrt(3.0);
    unit[0][2] = 1.0 / std::sqrt(3.0);

    Tensor out = sn.forward(unit);
    double out_norm = std::sqrt(out[0][0] * out[0][0] + out[0][1] * out[0][1]);

    // output = unit @ (gamma*W/σ)^T → norm = |gamma * unit · (W/σ)^T row|
    // For any unit v, ||v W|| ≤ σ(W) by definition. Rescaling by gamma=1/σ gives ≤ 1.
    // We just sanity-check the output is on the order of unity (Lipschitz-1-ish).
    check("Lipschitz-bounded output norm (≤ ~1.5×)",
          out_norm < 1.5);
    check("Lipschitz-bounded output norm (≥ 0 — i.e. non-trivially finite)",
          std::isfinite(out_norm));
    (void)sigma;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== SpectralNorm Tests ===" << endl;

    test_constructor();
    test_spectral_norm_identity();
    test_spectral_norm_known();
    test_forward();
    test_backward();
    test_update_weights_noop();
    test_accessors();
    test_determinism();
    test_normalize();
    test_end_to_end_lipschitz();

    cout << endl;
    cout << "Summary: " << passed << " passed, " << failed << " failed" << endl;
    return failed == 0 ? 0 : 1;
}
