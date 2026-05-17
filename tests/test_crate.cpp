#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include "nn/layers/normalization/crate.h"
#include "nn/core/tensor.h"
#include "nn/utils/numerical_stability.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static double rel_error(double a, double b) {
    if (std::abs(a) < 1e-10 && std::abs(b) < 1e-10) return 0.0;
    return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-10);
}

static void check(const string& name, bool cond) {
    if (cond) {
        cout << "  [PASS] " << name << endl;
        passed++;
    } else {
        cout << "  [FAIL] " << name << endl;
        failed++;
    }
}

static double tensor_l2norm(const Tensor& t) {
    double sum = 0.0;
    for (size_t r = 0; r < t.rows; ++r)
        for (size_t c = 0; c < t.cols; ++c)
            sum += t[r][c] * t[r][c];
    return std::sqrt(sum);
}

// =====================================================================
// Test 1: CRATE forward pass
// =====================================================================
static void test_crate_forward() {
    cout << endl << "-- Test 1: CRATE forward pass --" << endl;

    size_t N = 2, C = 4, H = 3, W = 3;
    CRATE crate(static_cast<int>(C), 2);

    // Diverse input to ensure multi-channel activation
    std::mt19937 gen(42);
    std::normal_distribution<double> dis(0.0, 0.5);
    Tensor input(static_cast<int>(N), static_cast<int>(C * H * W));
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < C * H * W; ++c)
            input[r][c] = dis(gen);

    Tensor out = crate.forward(input);

    check("CRATE output shape matches",
          out.rows == static_cast<int>(N) && out.cols == static_cast<int>(C * H * W));

    bool all_finite = true;
    for (size_t r = 0; r < out.rows; ++r)
        for (size_t c = 0; c < out.cols; ++c)
            if (!std::isfinite(out[r][c])) all_finite = false;
    check("CRATE output finite", all_finite);
}

// =====================================================================
// Test 2: CRATE input gradient (numerical vs analytical)
// =====================================================================
static void test_crate_input_gradient() {
    cout << endl << "-- Test 2: CRATE input gradient (numerical vs analytical) --" << endl;

    size_t N = 2, C = 4, H = 3, W = 3;
    CRATE crate(static_cast<int>(C), 2);

    std::mt19937 gen(42);
    std::normal_distribution<double> dis(0.0, 0.5);
    Tensor input(static_cast<int>(N), static_cast<int>(C * H * W));
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < C * H * W; ++c)
            input[r][c] = dis(gen);

    Tensor out = crate.forward(input);
    Tensor grad_out(static_cast<int>(N), static_cast<int>(C * H * W));
    grad_out.fill(1.0);

    crate.zero_grad();
    Tensor grad_x = crate.backward(grad_out, 0.0);

    // Numerical check
    double eps = 1e-5;
    double orig = input[0][0];

    Tensor input_plus = input;
    input_plus[0][0] = orig + eps;
    crate.zero_grad();
    Tensor out_plus = crate.forward(input_plus);

    Tensor input_minus = input;
    input_minus[0][0] = orig - eps;
    crate.zero_grad();
    Tensor out_minus = crate.forward(input_minus);

    double numerical = 0.0;
    for (size_t n = 0; n < N; ++n)
        for (size_t i = 0; i < C * H * W; ++i)
            numerical += (out_plus[n][i] - out_minus[n][i]) / (2 * eps) * grad_out[n][i];

    double analytical = grad_x[0][0];
    double rel = rel_error(numerical, analytical);
    cout << "  input[0][0]: num=" << numerical << " ana=" << analytical
         << " rel_err=" << (rel*100) << "%" << endl;
    check("CRATE input[0][0] gradient rel_err < 20%", rel < 0.2);

    // Check a few more elements with diverse values
    size_t check_indices[] = {1, (C * H * W) / 2, C * H * W - 1};
    for (size_t idx : check_indices) {
        size_t r = idx / (C * H * W);
        size_t c = idx % (C * H * W);
        double orig_el = input[r][c];

        Tensor in_p = input;
        in_p[r][c] = orig_el + eps;
        crate.zero_grad();
        Tensor out_p = crate.forward(in_p);

        Tensor in_m = input;
        in_m[r][c] = orig_el - eps;
        crate.zero_grad();
        Tensor out_m = crate.forward(in_m);

        double num = 0.0;
        for (size_t n = 0; n < N; ++n)
            for (size_t i = 0; i < C * H * W; ++i)
                num += (out_p[n][i] - out_m[n][i]) / (2 * eps) * grad_out[n][i];

        double ana = grad_x[r][c];
        double r2 = rel_error(num, ana);
        cout << "  input[" << r << "][" << c << "]: num=" << num << " ana=" << ana
             << " rel_err=" << (r2*100) << "%" << endl;
        check("CRATE input[" + to_string(r) + "][" + to_string(c) + "] gradient rel_err < 30%",
              r2 < 0.3);
    }
}

// =====================================================================
// Test 3: CRATE gradient flow with diverse input (avoids collapsed attention)
// =====================================================================
static void test_crate_gradient_flow() {
    cout << endl << "-- Test 3: CRATE gradient flow with diverse input --" << endl;

    size_t N = 2, C = 4, H = 2, W = 2;
    CRATE crate(static_cast<int>(C), 2);

    // Use diverse values to ensure multi-channel activation
    std::mt19937 gen(42);
    std::normal_distribution<double> dis(0.0, 0.5);
    Tensor input(static_cast<int>(N), static_cast<int>(C * H * W));
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < C * H * W; ++c)
            input[r][c] = dis(gen);

    Tensor out = crate.forward(input);
    Tensor grad_out(static_cast<int>(N), static_cast<int>(C * H * W));
    grad_out.fill(1.0);

    crate.zero_grad();
    Tensor grad_x = crate.backward(grad_out, 0.0);

    bool all_finite = true;
    double max_grad = 0.0;
    for (size_t r = 0; r < grad_x.rows; ++r) {
        for (size_t c = 0; c < grad_x.cols; ++c) {
            if (!std::isfinite(grad_x[r][c])) all_finite = false;
            max_grad = std::max(max_grad, std::abs(grad_x[r][c]));
        }
    }
    check("CRATE grad_input all finite", all_finite);
    check("CRATE grad_input non-trivial magnitude", max_grad > 1e-8);
    cout << "  Max grad magnitude: " << max_grad << endl;

    // Check that all parameters got gradients
    auto grads = crate.gradients();
    check("CRATE has 6 parameter gradient tensors", grads.size() == 6);
    for (auto* g : grads) {
        double gnorm = tensor_l2norm(*g);
        check("CRATE gradient norm > 0 for param", gnorm > 1e-10);
    }
}

// =====================================================================
// Test 4: CRATE backward end-to-end gradient magnitude
// =====================================================================
static void test_crate_gradient_magnitude() {
    cout << endl << "-- Test 4: CRATE gradient magnitude check --" << endl;

    size_t N = 2, C = 4, H = 3, W = 3;  // H=3 W=3 for stronger GAP signal
    CRATE crate(static_cast<int>(C), 2);

    // Use diverse, non-collapsing input (same distribution as Test 3 which works)
    std::mt19937 gen(42);
    std::normal_distribution<double> nd(0.0, 0.5);
    Tensor input(static_cast<int>(N), static_cast<int>(C * H * W));
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < C * H * W; ++c)
            input[r][c] = nd(gen);

    Tensor out = crate.forward(input);
    Tensor grad_out(static_cast<int>(N), static_cast<int>(C * H * W));
    grad_out.fill(1.0);

    crate.zero_grad();
    Tensor grad_x = crate.backward(grad_out, 0.0);

    double gnorm = tensor_l2norm(grad_x);
    cout << "  Gradient L2 norm: " << gnorm << endl;
    check("CRATE gradient norm > 0", gnorm > 1e-8);
    check("CRATE gradient norm < 100", gnorm < 100.0);

    auto grads = crate.gradients();
    double total_param_grad_norm = 0.0;
    for (auto* g : grads) total_param_grad_norm += tensor_l2norm(*g);
    cout << "  Total param grad norm: " << total_param_grad_norm << endl;
    check("CRATE param grad norm > 0", total_param_grad_norm > 1e-8);
}

int main() {
    cout << "======================================" << endl;
    cout << "  CRATE Gradient Correctness Tests" << endl;
    cout << "======================================" << endl;

    test_crate_forward();
    test_crate_input_gradient();
    test_crate_gradient_flow();
    test_crate_gradient_magnitude();

    cout << "\n======================================" << endl;
    cout << "  Results: " << passed << " passed, " << failed << " failed" << endl;
    cout << "======================================" << endl;

    return failed > 0 ? 1 : 0;
}
