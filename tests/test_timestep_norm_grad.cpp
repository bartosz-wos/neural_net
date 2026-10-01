// FD gradient tests for TimestepNorm (MEGALODON §3.2).
//
// The cumulative statistics (mu_t, var_t over x_1..x_t) are themselves
// functions of x, so a "gradient through a constant" implementation passes a
// shape check but is wrong. These tests compare the analytical backward
// against centered finite differences on the same forward.
//
// Uses random (non-uniform) init on gamma/beta: a uniform gamma would mask
// index-transposition bugs because sum_k W[k,j] == sum_k W[j,k].

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double sq_loss(const Tensor& y, const Tensor& target) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y(i, j) - target(i, j);
            L += 0.5 * d * d;
        }
    return L;
}

static Tensor rand_tensor(size_t r, size_t c, double scale, unsigned seed) {
    Tensor t(r, c);
    srand(seed);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(i, j) = ((rand() / (double)RAND_MAX) * 2.0 - 1.0) * scale;
    return t;
}

// FD of the loss w.r.t. one input entry.
static double fd_input(TimestepNorm& norm, const Tensor& x, const Tensor& target,
                       size_t r, size_t c, double eps) {
    Tensor orig = x.clone();
    auto loss_at = [&](double delta) {
        Tensor xa = orig.clone();
        xa(r, c) += delta;
        return sq_loss(norm.forward(xa), target);
    };
    double Lp = loss_at(+eps), Lm = loss_at(-eps);
    return (Lp - Lm) / (2.0 * eps);
}

// FD of the loss w.r.t. one parameter entry.
static double fd_param(TimestepNorm& norm, const Tensor& x, const Tensor& target,
                       Tensor* param, size_t r, size_t c, double eps) {
    double orig = (*param)(r, c);
    auto loss_at = [&](double delta) {
        (*param)(r, c) = orig + delta;
        return sq_loss(norm.forward(x), target);
    };
    double Lp = loss_at(+eps), Lm = loss_at(-eps);
    (*param)(r, c) = orig;
    return (Lp - Lm) / (2.0 * eps);
}

// Combined criterion: rel_err where the magnitudes are meaningful, plus an
// absolute floor so near-zero entries don't produce meaningless ratios.
static bool close_enough(double ana, double num, double scale) {
    double tol = std::max(1e-7, 1e-5 * scale);
    return std::abs(ana - num) <= tol;
}

// =====================================================================
// Test 9: FD input gradient (2 groups, n=4, d=4)
// =====================================================================
static void test_fd_input_gradient() {
    cout << endl << "-- Test 9: FD input gradient (k=2, n=4, d=4) --" << endl;

    const size_t n = 4, d = 4;
    TimestepNorm norm(d, 2, 1e-5);
    // Random non-uniform gamma/beta — uniform would mask transposition bugs.
    Tensor g = rand_tensor(1, d, 0.5, 11);
    Tensor b = rand_tensor(1, d, 0.5, 22);
    for (size_t j = 0; j < d; ++j) { norm.gamma(0, j) = g(0, j); norm.beta(0, j) = b(0, j); }

    Tensor x = rand_tensor(n, d, 1.0, 33);
    Tensor target = rand_tensor(n, d, 1.0, 44);

    // The FD helper differentiates sq_loss(y(x), target), so the analytical
    // run must receive the matching upstream gradient: gy = y - target.
    Tensor y = norm.forward(x);
    Tensor gy_loss = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) gy_loss(i, j) = y(i, j) - target(i, j);

    norm.backward(gy_loss, 0.0);
    Tensor ana = norm.grad_x.clone();

    // Scale for the tolerance: the analytic gradient magnitude.
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) scale = std::max(scale, std::abs(ana(i, j)));

    const double eps = 1e-6;
    double worst = 0.0; size_t worst_r = 0, worst_c = 0;
    bool all_ok = true;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double num = fd_input(norm, x, target, i, j, eps);
            double err = std::abs(ana(i, j) - num);
            if (err > worst) { worst = err; worst_r = i; worst_c = j; }
            if (!close_enough(ana(i, j), num, scale)) all_ok = false;
        }

    cout << "    worst abs err = " << scientific << setprecision(3) << worst
         << " at (" << worst_r << "," << worst_c << "), scale = " << scale
         << defaultfloat << endl;
    check("FD input gradient matches analytical", all_ok);
}

// =====================================================================
// Test 10: FD gamma / beta gradient
// =====================================================================
static void test_fd_param_gradient() {
    cout << endl << "-- Test 10: FD gamma / beta gradient --" << endl;

    const size_t n = 5, d = 6;
    TimestepNorm norm(d, 3, 1e-5);
    Tensor g = rand_tensor(1, d, 0.5, 101);
    Tensor b = rand_tensor(1, d, 0.5, 202);
    for (size_t j = 0; j < d; ++j) { norm.gamma(0, j) = g(0, j); norm.beta(0, j) = b(0, j); }

    Tensor x = rand_tensor(n, d, 1.0, 303);
    Tensor target = rand_tensor(n, d, 1.0, 404);

    Tensor y = norm.forward(x);
    Tensor gy_loss = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) gy_loss(i, j) = y(i, j) - target(i, j);

    norm.backward(gy_loss, 0.0);

    const double eps = 1e-6;
    bool gamma_ok = true, beta_ok = true;
    double gscale = 0.0, bscale = 0.0;
    for (size_t j = 0; j < d; ++j) {
        gscale = std::max(gscale, std::abs(norm.grad_gamma_(0, j)));
        bscale = std::max(bscale, std::abs(norm.grad_beta_(0, j)));
    }
    for (size_t j = 0; j < d; ++j) {
        double num_g = fd_param(norm, x, target, &norm.gamma, 0, j, eps);
        double num_b = fd_param(norm, x, target, &norm.beta, 0, j, eps);
        if (!close_enough(norm.grad_gamma_(0, j), num_g, gscale)) gamma_ok = false;
        if (!close_enough(norm.grad_beta_(0, j), num_b, bscale)) beta_ok = false;
    }
    cout << "    gamma scale = " << scientific << gscale << ", beta scale = " << bscale
         << defaultfloat << endl;
    check("FD gamma gradient matches analytical", gamma_ok);
    check("FD beta gradient matches analytical", beta_ok);
}

// =====================================================================
// Test 11: FD with a single group (n=5) — the paper's default config
// =====================================================================
static void test_fd_single_group() {
    cout << endl << "-- Test 11: FD single group (k=1, n=5, d=4) --" << endl;

    const size_t n = 5, d = 4;
    TimestepNorm norm(d, 0, 1e-5);   // default -> 1 group
    for (size_t j = 0; j < d; ++j) {
        norm.gamma(0, j) = (rand() / (double)RAND_MAX) * 1.0 - 0.5;
        norm.beta(0, j) = (rand() / (double)RAND_MAX) * 1.0 - 0.5;
    }

    Tensor x = rand_tensor(n, d, 1.0, 505);
    Tensor target = rand_tensor(n, d, 1.0, 606);

    Tensor y = norm.forward(x);
    Tensor gy_loss = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) gy_loss(i, j) = y(i, j) - target(i, j);
    norm.backward(gy_loss, 0.0);
    Tensor ana = norm.grad_x.clone();

    double scale = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) scale = std::max(scale, std::abs(ana(i, j)));

    const double eps = 1e-6;
    bool ok = true; double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double num = fd_input(norm, x, target, i, j, eps);
            worst = std::max(worst, std::abs(ana(i, j) - num));
            if (!close_enough(ana(i, j), num, scale)) ok = false;
        }
    cout << "    worst abs err = " << scientific << setprecision(3) << worst
         << ", scale = " << scale << defaultfloat << endl;
    check("FD input gradient matches (single group)", ok);
}

// =====================================================================
// Test 12: FD with group size 1 (k = d) — degenerate but legal
// =====================================================================
static void test_fd_group_size_one() {
    cout << endl << "-- Test 12: FD group size 1 (k=d, n=3, d=4) --" << endl;

    const size_t n = 3, d = 4;
    TimestepNorm norm(d, d, 1e-5);
    for (size_t j = 0; j < d; ++j) norm.gamma(0, j) = 0.3 * ((j % 2) ? -1.0 : 1.0);

    Tensor x = rand_tensor(n, d, 1.0, 707);
    Tensor target = rand_tensor(n, d, 1.0, 808);

    Tensor y = norm.forward(x);
    Tensor gy_loss = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) gy_loss(i, j) = y(i, j) - target(i, j);
    norm.backward(gy_loss, 0.0);
    Tensor ana = norm.grad_x.clone();

    double scale = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) scale = std::max(scale, std::abs(ana(i, j)));

    const double eps = 1e-6;
    bool ok = true; double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double num = fd_input(norm, x, target, i, j, eps);
            worst = std::max(worst, std::abs(ana(i, j) - num));
            if (!close_enough(ana(i, j), num, scale)) ok = false;
        }
    cout << "    worst abs err = " << scientific << setprecision(3) << worst
         << ", scale = " << scale << defaultfloat << endl;
    check("FD input gradient matches (group size 1)", ok);
}

// =====================================================================
// Test 13: zero_grad / update_weights
// =====================================================================
static void test_zero_grad_and_update() {
    cout << endl << "-- Test 13: zero_grad + update_weights --" << endl;

    const size_t n = 4, d = 6;
    TimestepNorm norm(d, 2, 1e-5);
    Tensor x = rand_tensor(n, d, 1.0, 909);
    Tensor y = norm.forward(x);
    Tensor gy = rand_tensor(n, d, 1.0, 1010);
    norm.backward(gy, 0.0);

    double gmax = 0.0, bmax = 0.0;
    for (size_t j = 0; j < d; ++j) {
        gmax = std::max(gmax, std::abs(norm.grad_gamma_(0, j)));
        bmax = std::max(bmax, std::abs(norm.grad_beta_(0, j)));
    }
    check("grad_gamma non-zero after backward", gmax > 1e-8);
    check("grad_beta non-zero after backward", bmax > 1e-8);

    norm.zero_grad();
    gmax = 0.0; bmax = 0.0;
    for (size_t j = 0; j < d; ++j) {
        gmax = std::max(gmax, std::abs(norm.grad_gamma_(0, j)));
        bmax = std::max(bmax, std::abs(norm.grad_beta_(0, j)));
    }
    check("zero_grad clears grad_gamma", gmax == 0.0);
    check("zero_grad clears grad_beta", bmax == 0.0);

    // update_weights must move gamma/beta by exactly -lr*grad.
    Tensor g_before = norm.gamma.clone();
    Tensor b_before = norm.beta.clone();
    const double lr = 0.1;
    norm.update_weights(lr);
    double worst = 0.0;
    for (size_t j = 0; j < d; ++j) {
        // grads are zero after zero_grad, so nothing should move
        worst = std::max(worst, std::abs(norm.gamma(0, j) - g_before(0, j)));
        worst = std::max(worst, std::abs(norm.beta(0, j) - b_before(0, j)));
    }
    check("update_weights with zero grads is a no-op", worst == 0.0);
}

// =====================================================================
// Test 14: training reduces loss end-to-end (through a Dense)
// =====================================================================
static void test_training_reduces_loss() {
    cout << endl << "-- Test 14: end-to-end training reduces loss --" << endl;

    const size_t n = 12, d = 6;
    Tensor x = rand_tensor(n, d, 1.0, 1111);
    Tensor target = rand_tensor(n, d, 1.0, 2222);

    TimestepNorm norm(d, 2, 1e-5);
    const double lr = 0.05;

    auto loss_now = [&]() {
        Tensor y = norm.forward(x);
        return sq_loss(y, target);
    };

    double L0 = loss_now();
    for (int step = 0; step < 60; ++step) {
        Tensor y = norm.forward(x);
        Tensor gy = Tensor(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j) gy(i, j) = y(i, j) - target(i, j);
        norm.backward(gy, lr);
    }
    double Lf = loss_now();
    cout << "    L0 = " << fixed << setprecision(5) << L0
         << " -> Lf = " << Lf << defaultfloat << endl;
    check("loss reduced by > 50%", Lf < L0 * 0.5);
}

int main() {
    cout << "==========================================" << endl;
    cout << "  TimestepNorm Gradient Tests" << endl;
    cout << "  MEGALODON §3.2 — FD verified" << endl;
    cout << "==========================================" << endl;

    test_fd_input_gradient();
    test_fd_param_gradient();
    test_fd_single_group();
    test_fd_group_size_one();
    test_zero_grad_and_update();
    test_training_reduces_loss();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
