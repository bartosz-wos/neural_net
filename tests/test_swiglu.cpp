// test_swiglu.cpp — FD-checked gradient tests for SwiGLU.
//
//   FFN(x) = SiLU(W1 @ x) * (W2 @ x)
//
// Regression origin: SwiGLU::backward computed the SiLU derivative from
// `w1_.last_input`, believing it held the pre-activation W1 @ x. Dense caches
// its INPUT in `last_input` (`last_input = input.clone()` in
// Dense::forward), so the derivative was evaluated at the wrong point:
// sigmoid(x) instead of sigmoid(W1 @ x). The existing test suite never
// exercised this layer, so the bug shipped silently.
//
// These tests check BOTH shapes:
//   - square       (dim_input == dim_hidden) — the wrong-value bug is fatal
//   - rectangular  (dim_input != dim_hidden) — different magnitudes, and it
//     guards the case where the shapes happen to coincide
//
// Random non-uniform init and random input throughout: a uniform fixture can
// mask row-vs-column confusion in the W1/W2 gradients.

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

static Tensor rand_tensor(size_t r, size_t c, double scale, unsigned seed) {
    Tensor t(r, c);
    srand(seed);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(i, j) = ((rand() / (double)RAND_MAX) * 2.0 - 1.0) * scale;
    return t;
}

static void set_all_params(SwiGLU<Swish>& f, const vector<Tensor>& vals) {
    auto p = f.parameters();
    for (size_t i = 0; i < p.size(); ++i) *p[i] = vals[i].clone();
}

static vector<Tensor> get_all_params(SwiGLU<Swish>& f) {
    vector<Tensor> out;
    for (auto* p : f.parameters()) out.push_back(p->clone());
    return out;
}

static vector<Tensor> get_all_grads(SwiGLU<Swish>& f) {
    vector<Tensor> out;
    for (auto* g : f.gradients()) out.push_back(g->clone());
    return out;
}

// L = 0.5 * ||forward(x) - target||^2
static double sq_loss(const Tensor& y, const Tensor& target) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y(i, j) - target(i, j);
            L += 0.5 * d * d;
        }
    return L;
}

// Combined criterion: relative error on meaningful magnitudes, with an
// absolute floor so near-zero entries don't produce meaningless ratios.
static bool close_enough(double ana, double num, double scale) {
    return fabs(ana - num) <= std::max(1e-7, 1e-5 * scale);
}

// =====================================================================
// Test 1: constructor + parameter contract
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: constructor + parameter contract --" << endl;

    SwiGLU<Swish> f(4, 8);
    check("name() == \"SwiGLU\"", f.name() == "SwiGLU");

    auto p = f.parameters();
    auto g = f.gradients();
    check("4 parameters (W1.w, W1.b, W2.w, W2.b)", p.size() == 4);
    check("4 gradients", g.size() == 4);
    check("W1.weights is (dim_hidden, dim_input)", p[0]->rows == 8 && p[0]->cols == 4);
    check("W1.bias is (1, dim_hidden)", p[1]->rows == 1 && p[1]->cols == 8);
    check("W2.weights is (dim_hidden, dim_input)", p[2]->rows == 8 && p[2]->cols == 4);
    check("W2.bias is (1, dim_hidden)", p[3]->rows == 1 && p[3]->cols == 8);

    bool shapes_match = true;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes_match = false;
    check("gradient shapes match parameter shapes", shapes_match);

    // Forward shape for a rectangular config.
    Tensor y = f.forward(rand_tensor(3, 4, 1.0, 1));
    check("forward (3,4) -> (3,8)", y.rows == 3 && y.cols == 8);

    // Hand-derived reference: with W2 = 0 the output is exactly 0.
    SwiGLU<Swish> z(3, 5);
    set_all_params(z, {rand_tensor(5, 3, 0.5, 2), rand_tensor(1, 5, 0.5, 3),
                       Tensor::zeros(5, 3), Tensor::zeros(1, 5)});
    Tensor yz = z.forward(rand_tensor(2, 3, 1.0, 4));
    double maxy = 0.0;
    for (size_t i = 0; i < yz.rows; ++i)
        for (size_t j = 0; j < yz.cols; ++j) maxy = max(maxy, fabs(yz(i, j)));
    check("W2 = 0 gives exactly zero output", maxy == 0.0);

    // Hand-derived reference: with W1 = I, W2 = I and both biases zero,
    // h1 = x, h2 = x, so output = SiLU(x) * x.
    SwiGLU<Swish> h(3, 3);
    Tensor I3 = Tensor::zeros(3, 3);
    for (size_t i = 0; i < 3; ++i) I3(i, i) = 1.0;
    set_all_params(h, {I3, Tensor::zeros(1, 3), I3, Tensor::zeros(1, 3)});
    Tensor x = rand_tensor(2, 3, 1.0, 6);
    Tensor yh = h.forward(x);
    double worst = 0.0;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double xv = x(i, j);
            double silu = xv / (1.0 + exp(-xv));
            worst = max(worst, fabs(yh(i, j) - silu * xv));
        }
    cout << "    hand-derived SiLU(x)*x max diff = " << scientific << worst << defaultfloat << endl;
    check("W1=I, W2=I, b=0 gives SiLU(x)*x exactly", worst < 1e-12);
}

// =====================================================================
// Test 2: FD input gradient — SQUARE config
// =====================================================================
static void test_fd_input_square() {
    cout << endl << "-- Test 2: FD input gradient (square, dim=4) --" << endl;

    const size_t d = 4, n = 3;
    SwiGLU<Swish> f(d, d);
    set_all_params(f, {rand_tensor(d, d, 0.5, 11), rand_tensor(1, d, 0.5, 12),
                       rand_tensor(d, d, 0.5, 13), rand_tensor(1, d, 0.5, 14)});
    Tensor x = rand_tensor(n, d, 1.0, 15);
    Tensor target = rand_tensor(n, d, 1.0, 16);

    f.zero_grad();
    Tensor y = f.forward(x);
    Tensor gy = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) gy(i, j) = y(i, j) - target(i, j);
    Tensor gx = f.backward(gy, 0.0);

    const double eps = 1e-6;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            Tensor xp = x.clone(); xp(i, j) += eps;
            Tensor xm = x.clone(); xm(i, j) -= eps;
            double num = (sq_loss(f.forward(xp), target) - sq_loss(f.forward(xm), target)) / (2 * eps);
            worst = max(worst, fabs(gx(i, j) - num) / std::max(1e-12, fabs(num)));
        }
    cout << "    worst input-grad rel_err = " << scientific << worst << defaultfloat << endl;
    check("square: input gradient matches FD", worst < 1e-4);
}

// =====================================================================
// Test 3: FD input gradient — RECTANGULAR config
// =====================================================================
static void test_fd_input_rect() {
    cout << endl << "-- Test 3: FD input gradient (rectangular, 4 -> 8) --" << endl;

    const size_t di = 4, dh = 8, n = 3;
    SwiGLU<Swish> f(di, dh);
    set_all_params(f, {rand_tensor(dh, di, 0.5, 21), rand_tensor(1, dh, 0.5, 22),
                       rand_tensor(dh, di, 0.5, 23), rand_tensor(1, dh, 0.5, 24)});
    Tensor x = rand_tensor(n, di, 1.0, 25);
    Tensor target = rand_tensor(n, dh, 1.0, 26);

    f.zero_grad();
    Tensor y = f.forward(x);
    Tensor gy = Tensor(n, dh);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < dh; ++j) gy(i, j) = y(i, j) - target(i, j);
    Tensor gx = f.backward(gy, 0.0);

    const double eps = 1e-6;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < di; ++j) {
            Tensor xp = x.clone(); xp(i, j) += eps;
            Tensor xm = x.clone(); xm(i, j) -= eps;
            double num = (sq_loss(f.forward(xp), target) - sq_loss(f.forward(xm), target)) / (2 * eps);
            worst = max(worst, fabs(gx(i, j) - num) / std::max(1e-12, fabs(num)));
        }
    cout << "    worst input-grad rel_err = " << scientific << worst << defaultfloat << endl;
    check("rectangular: input gradient matches FD", worst < 1e-4);
}

// =====================================================================
// Test 4: FD parameter gradients — W1, W2, b1, b2 (both shapes)
// =====================================================================
static void test_fd_params() {
    cout << endl << "-- Test 4: FD parameter gradients (W1/W2/b1/b2) --" << endl;

    for (int shape = 0; shape < 2; ++shape) {
        size_t di = (shape == 0) ? 4 : 4;
        size_t dh = (shape == 0) ? 4 : 8;
        const size_t n = 3;
        string tag = (shape == 0) ? "square" : "rect";

        SwiGLU<Swish> f(di, dh);
        set_all_params(f, {rand_tensor(dh, di, 0.5, 31), rand_tensor(1, dh, 0.5, 32),
                           rand_tensor(dh, di, 0.5, 33), rand_tensor(1, dh, 0.5, 34)});
        Tensor x = rand_tensor(n, di, 1.0, 35);
        Tensor target = rand_tensor(n, dh, 1.0, 36);

        f.zero_grad();
        Tensor y = f.forward(x);
        Tensor gy = Tensor(n, dh);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < dh; ++j) gy(i, j) = y(i, j) - target(i, j);
        f.backward(gy, 0.0);

        auto p = f.parameters();
        auto g = f.gradients();
        const double eps = 1e-6;
        for (size_t pi = 0; pi < 4; ++pi) {
            double worst = 0.0, scale = 0.0;
            // W1/W2 are (dh, di) — probe every entry (small fixture).
            // Biases are (1, dh) — probe every entry too.
            for (size_t i = 0; i < p[pi]->rows; ++i)
                for (size_t j = 0; j < p[pi]->cols; ++j) {
                    double orig = (*p[pi])(i, j);
                    (*p[pi])(i, j) = orig + eps;
                    double Lp = sq_loss(f.forward(x), target);
                    (*p[pi])(i, j) = orig - eps;
                    double Lm = sq_loss(f.forward(x), target);
                    (*p[pi])(i, j) = orig;
                    double num = (Lp - Lm) / (2 * eps);
                    double ana = (*g[pi])(i, j);
                    scale = max(scale, fabs(num));
                    double rel = fabs(ana - num) / std::max(1e-9, fabs(num));
                    if (rel > worst) worst = rel;
                }
            cout << "    " << tag << " param[" << pi << "] (" << p[pi]->rows
                 << "," << p[pi]->cols << ")  worst rel_err = " << scientific
                 << worst << "  (grad scale " << scale << ")" << defaultfloat << endl;
            check(tag + ": param " + to_string(pi) + " gradient matches FD", worst < 1e-3);
        }
    }
}

// =====================================================================
// Test 5: gradient accumulation + zero_grad + update_weights
// =====================================================================
static void test_accumulation() {
    cout << endl << "-- Test 5: accumulation / zero_grad / update_weights --" << endl;

    const size_t di = 4, dh = 8, n = 3;
    SwiGLU<Swish> f(di, dh);
    Tensor x = rand_tensor(n, di, 1.0, 41);
    Tensor gy = rand_tensor(n, dh, 1.0, 42);

    f.zero_grad();
    f.forward(x); f.backward(gy, 0.0);
    auto g1 = get_all_grads(f);

    f.forward(x); f.backward(gy, 0.0);
    auto g2 = get_all_grads(f);

    double worst = 0.0;
    for (size_t k = 0; k < 4; ++k)
        for (size_t i = 0; i < g1[k].rows; ++i)
            for (size_t j = 0; j < g1[k].cols; ++j)
                worst = max(worst, fabs(g2[k](i, j) - 2.0 * g1[k](i, j)));
    cout << "    max|grad after 2 calls - 2 * grad after 1 call| = "
         << scientific << worst << defaultfloat << endl;
    check("gradients accumulate across backward calls", worst < 1e-12);

    f.zero_grad();
    auto g3 = get_all_grads(f);
    double worst0 = 0.0;
    for (size_t k = 0; k < 4; ++k)
        for (size_t i = 0; i < g3[k].rows; ++i)
            for (size_t j = 0; j < g3[k].cols; ++j) worst0 = max(worst0, fabs(g3[k](i, j)));
    check("zero_grad clears all gradients", worst0 == 0.0);

    // update_weights must move every parameter by exactly -lr * grad.
    auto p = f.parameters();
    auto before = get_all_params(f);
    f.forward(x); f.backward(gy, 0.0);
    auto gr = get_all_grads(f);
    const double lr = 0.01;
    f.update_weights(lr);
    double worst_u = 0.0;
    for (size_t k = 0; k < 4; ++k)
        for (size_t i = 0; i < p[k]->rows; ++i)
            for (size_t j = 0; j < p[k]->cols; ++j)
                worst_u = max(worst_u, fabs((*p[k])(i, j) - (before[k](i, j) - lr * gr[k](i, j))));
    check("update_weights moves every parameter by -lr * grad", worst_u < 1e-15);
}

// =====================================================================
// Test 6: determinism
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 6: determinism --" << endl;
    SwiGLU<Swish> f(4, 8);
    Tensor x = rand_tensor(3, 4, 1.0, 51);
    Tensor y1 = f.forward(x);
    Tensor y2 = f.forward(x);
    double md = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j) md = max(md, fabs(y1(i, j) - y2(i, j)));
    check("two consecutive forwards bit-exact", md == 0.0);
}

int main() {
    cout << "==========================================" << endl;
    cout << "  SwiGLU Tests" << endl;
    cout << "  FD-checked: regression for the" << endl;
    cout << "  swish-derivative-evaluated-at-x bug" << endl;
    cout << "==========================================" << endl;

    test_constructor();
    test_fd_input_square();
    test_fd_input_rect();
    test_fd_params();
    test_accumulation();
    test_determinism();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
