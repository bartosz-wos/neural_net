// test_complex_ema.cpp — Tests for ComplexEMA (MEGALODON §3.1, Eq. 2-3).
//
// Ma, Yang, Xiong, Chen, Yu, Zhang, Ma, Zhou, "MEGALODON: Efficient LLM
// Pretraining and Inference with Unlimited Context Length"
// (https://arxiv.org/abs/2404.08801).
//
// Multi-dimensional EMA extended to the COMPLEX domain. Per channel j and
// complex state k = 0..h-1:
//
//   h_{t,j,k} = p_{j,k}·x_{t,j} + q_{j,k}·h_{t-1,j,k}          (complex)
//   y_{t,j}   = Re( SUM_k eta_{j,k}·h_{t,j,k} ) + omega_j·x_{t,j}
//
// Parameterization (official implementation, _calc_coeffs):
//
//   theta_{j,k} = (k+1) * sigmoid(theta_j) * (2*pi/h)     (Eq. 3)
//   p_{j,k}     = sigmoid(alpha_{j,k})                             in (0,1)
//   q_{j,k}     = (1 - alpha·delta) * (cos(theta) + i sin(theta))
//   eta_{j,k}   in C  (two real parameters), scaled by 1/sqrt(h)
//   omega_j     in R
//
// |q| = 1 - alpha*delta < 1 is the DECAY property the paper calls out as
// key to the kernel structure — that is asserted directly below.
//
// Tensor is real-only in this repo, so the complex hidden state is stored as
// a (d, h, 2) real tensor: [...,0] = real part, [...,1] = imaginary part.

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

// =====================================================================
// Test 1: Constructor validation + accessors
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: constructor validation --" << endl;

    bool threw_d = false;
    try { ComplexEMA e(0, 4); } catch (...) { threw_d = true; }
    check("d_model=0 throws", threw_d);

    bool threw_h = false;
    try { ComplexEMA e(4, 0); } catch (...) { threw_h = true; }
    check("ndim=0 throws", threw_h);

    bool ok = false;
    try { ComplexEMA e(8, 16); ok = true; } catch (...) {}
    check("valid construct (d=8, ndim=16)", ok);

    ComplexEMA e(8, 4);
    check("d_model() == 8", e.d_model() == 8);
    check("ndim() == 4", e.ndim() == 4);
    check("name() == \"ComplexEMA\"", e.name() == "ComplexEMA");
    check("eta_scale() == 1/sqrt(ndim)",
          std::abs(e.eta_scale() - 1.0 / std::sqrt(4.0)) < 1e-15);
}

// =====================================================================
// Test 2: Parameter / gradient contract (6 tensors)
// =====================================================================
static void test_parameter_contract() {
    cout << endl << "-- Test 2: parameter/gradient contract --" << endl;

    ComplexEMA e(6, 4);
    auto p = e.parameters();
    auto g = e.gradients();
    check("6 parameters", p.size() == 6);
    check("6 gradients", g.size() == 6);

    check("alpha shape (6,4)", p[0]->rows == 6 && p[0]->cols == 4);
    check("delta shape (6,4)", p[1]->rows == 6 && p[1]->cols == 4);
    check("theta shape (6,1)", p[2]->rows == 6 && p[2]->cols == 1);
    check("eta_re shape (6,4)", p[3]->rows == 6 && p[3]->cols == 4);
    check("eta_im shape (6,4)", p[4]->rows == 6 && p[4]->cols == 4);
    check("omega shape (6,1)", p[5]->rows == 6 && p[5]->cols == 1);

    check("grad shapes match params", g[0]->rows == 6 && g[0]->cols == 4 &&
                                       g[1]->rows == 6 && g[1]->cols == 4 &&
                                       g[2]->rows == 6 && g[2]->cols == 1 &&
                                       g[3]->rows == 6 && g[3]->cols == 4 &&
                                       g[4]->rows == 6 && g[4]->cols == 4 &&
                                       g[5]->rows == 6 && g[5]->cols == 1);
}

// =====================================================================
// Test 3: Coefficient computation — decay property
// =====================================================================
static void test_decay_property() {
    cout << endl << "-- Test 3: |q| < 1 decay property --" << endl;

    const size_t d = 6, h = 4;
    ComplexEMA e(d, h);
    // Random non-trivial params.
    for (size_t j = 0; j < d; ++j) {
        e.alpha(j, 0) = 0.3 * (j + 1);
        e.delta(j, 0) = -0.2 * j;
        e.theta(j, 0) = 0.15 * (j + 1);
        for (size_t k = 0; k < h; ++k) {
            e.alpha(j, k) = 0.25 * ((j + k) % 5) - 0.5;
            e.delta(j, k) = 0.2 * ((j * 3 + k) % 4) - 0.3;
            e.eta_re(j, k) = 0.5 * std::cos(0.7 * k);
            e.eta_im(j, k) = 0.5 * std::sin(0.7 * k);
        }
    }

    e.compute_coeffs();
    const Tensor& qr = e.last_q_re();
    const Tensor& qi = e.last_q_im();

    bool all_decay = true;
    double worst = 0.0;
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k) {
            double mag = std::sqrt(qr(j, k) * qr(j, k) + qi(j, k) * qi(j, k));
            worst = std::max(worst, mag);
            if (mag >= 1.0) all_decay = false;
        }
    cout << "    max |q| = " << scientific << setprecision(4) << worst << defaultfloat << endl;
    check("all |q| < 1 (kernel decays)", all_decay);

    // p = sigmoid(alpha) must be in (0,1)
    const Tensor& pv = e.last_p();
    bool p_ok = true;
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k)
            if (!(pv(j, k) > 0.0 && pv(j, k) < 1.0)) p_ok = false;
    check("p = sigmoid(alpha) in (0,1)", p_ok);

    // |q| should equal 1 - p*delta exactly.
    double worst_mag = 0.0;
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k) {
            double mag = std::sqrt(qr(j, k) * qr(j, k) + qi(j, k) * qi(j, k));
            double expect = 1.0 - pv(j, k) * (1.0 / (1.0 + std::exp(-e.delta(j, k))));
            worst_mag = std::max(worst_mag, std::abs(mag - expect));
        }
    cout << "    max |‖q‖ - (1 - p*delta)| = " << scientific << setprecision(3)
         << worst_mag << defaultfloat << endl;
    check("|q| == 1 - p*delta to 1e-12", worst_mag < 1e-12);
}

// =====================================================================
// Test 4: theta spacing (Eq. 3) — uniformly spaced arguments
// =====================================================================
static void test_theta_spacing() {
    cout << endl << "-- Test 4: theta spacing (Eq. 3) --" << endl;

    const size_t d = 2, h = 8;
    ComplexEMA e(d, h);
    e.theta(0, 0) = 0.7;
    e.compute_coeffs();

    // theta_{j,k} = (k+1) * sigmoid(theta_j) * (2*pi/h)
    double base = 1.0 / (1.0 + std::exp(-0.7)) * (2.0 * M_PI / 8.0);
    // The phase of q is theta_{j,k}; recover it via atan2(im, re).
    const Tensor& qr = e.last_q_re();
    const Tensor& qi = e.last_q_im();
    double worst = 0.0;
    for (size_t k = 0; k < h; ++k) {
        double expect = (k + 1) * base;
        // q = mag * exp(i*expect);  atan2(qi, qr) == expect mod 2pi
        double got = std::atan2(qi(0, k), qr(0, k));
        double diff = std::fabs(got - expect);
        while (diff > M_PI) diff = std::fabs(diff - 2.0 * M_PI);
        worst = std::max(worst, diff);
    }
    cout << "    max phase err = " << scientific << setprecision(3) << worst << defaultfloat << endl;
    check("theta_{j,k} = (k+1)*sigmoid(theta_j)*2pi/h", worst < 1e-12);
}

// =====================================================================
// Test 5: Forward shape + finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 5: forward shape + finiteness --" << endl;

    ComplexEMA e(8, 4);
    Tensor x = rand_tensor(6, 8, 1.0, 42);
    Tensor y = e.forward(x);
    check("forward shape (6,8) -> (6,8)", y.rows == 6 && y.cols == 8);

    bool finite = true, nonzero = false;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            if (!std::isfinite(y(i, j))) finite = false;
            if (std::abs(y(i, j)) > 1e-9) nonzero = true;
        }
    check("forward finite", finite);
    check("forward nonzero", nonzero);

    // n=1 degenerate
    Tensor x1 = rand_tensor(1, 8, 1.0, 43);
    Tensor y1 = e.forward(x1);
    check("n=1 forward finite", std::isfinite(y1(0, 0)));
}

// =====================================================================
// Test 6: Hand-derived reference (n=1, d=1, h=1)
// =====================================================================
//
//   h_0 = p*x_0
//   y_0 = Re(eta * h_0) + omega*x_0 = Re(eta)*p*x_0 + omega*x_0
static void test_hand_derived_single() {
    cout << endl << "-- Test 6: hand-derived n=1,d=1,h=1 --" << endl;

    ComplexEMA e(1, 1);
    // Choose values giving clean sigmoid/theta values.
    e.alpha(0, 0) = 0.0;    // p = 0.5
    e.delta(0, 0) = 0.0;    // sigmoid(0) = 0.5
    e.theta(0, 0) = 0.0;    // sigmoid(0) = 0.5 -> base = 0.5*2pi = pi
    e.eta_re(0, 0) = 2.0;
    e.eta_im(0, 0) = 3.0;
    e.omega(0, 0) = -0.5;

    Tensor x(1, 1);
    x(0, 0) = 4.0;
    Tensor y = e.forward(x);

    // p = 0.5; q phase = pi -> q = (1 - 0.25)*exp(i*pi) = -0.75 (imag ~ 0)
    // h_0 = p*x_0 = 0.5*4 = 2
    // y_0 = Re(eta)*h + omega*x = 2*2 + (-0.5)*4 = 4 - 2 = 2
    double expect = 2.0 * 2.0 + (-0.5) * 4.0;
    cout << "    y_0 = " << y(0, 0) << ", expect = " << expect << endl;
    check("hand-derived single-cell forward matches (< 1e-12)",
          std::abs(y(0, 0) - expect) < 1e-12);
}

// =====================================================================
// Test 7: Zero initial state — the recurrence starts from h_{-1} = 0
// =====================================================================
//
// With eta_im = 0 and all omega = 0, the h_{t} at t=0 depends only on x_0.
// Setting the hidden state explicitly and checking the (d,h,2) cache is zero
// before the recurrence is the direct test.
static void test_zero_initial_state() {
    cout << endl << "-- Test 7: zero initial hidden state --" << endl;

    ComplexEMA e(3, 2);
    Tensor x = rand_tensor(4, 3, 1.0, 77);
    e.forward(x);
    const Tensor& h = e.last_h();
    // last_h is the interleaved complex state: (d, 2*ndim) with columns
    // [re0, im0, re1, im1, ...]. ndim=2 -> 2*2 = 4 columns.
    check("hidden state cache shape (3, 2*ndim=4)", h.rows == 3 && h.cols == 4);

    // Direct check: with n=1 the state must be exactly p*x_0 for each k.
    ComplexEMA e2(2, 3);
    Tensor x1 = rand_tensor(1, 2, 1.0, 88);
    e2.forward(x1);
    const Tensor& h1 = e2.last_h();
    e2.compute_coeffs();
    const Tensor& p2 = e2.last_p();
    double worst = 0.0;
    for (size_t j = 0; j < 2; ++j)
        for (size_t k = 0; k < 3; ++k) {
            worst = std::max(worst, std::abs(h1(j, 2 * k) - p2(j, k) * x1(0, j)));
            worst = std::max(worst, std::abs(h1(j, 2 * k + 1) - 0.0));  // imag part zero
        }
    cout << "    worst err (h_0 == p*x_0) = " << scientific << setprecision(3)
         << worst << defaultfloat << endl;
    check("n=1 hidden state is exactly p*x_0 (starts from h=0)", worst < 1e-12);
}

// =====================================================================
// Test 8: Recurrence matches an explicit hand-rolled loop (n=4, d=2, h=3)
// =====================================================================
static void test_recurrence_reference() {
    cout << endl << "-- Test 8: recurrence vs hand-rolled loop --" << endl;

    const size_t n = 4, d = 2, h = 3;
    ComplexEMA e(d, h);
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k) {
            e.alpha(j, k) = 0.2 * std::sin(1.3 * (double)(j * h + k));
            e.delta(j, k) = 0.15 * std::cos(0.9 * (double)(j * h + k));
            e.eta_re(j, k) = 0.4 * std::cos(0.5 * k);
            e.eta_im(j, k) = 0.4 * std::sin(0.5 * k);
        }
    for (size_t j = 0; j < d; ++j) {
        e.theta(j, 0) = 0.1 * j;
        e.omega(j, 0) = 0.05 * (j + 1);
    }

    Tensor x = rand_tensor(n, d, 1.0, 99);
    Tensor y = e.forward(x);
    e.compute_coeffs();

    const Tensor& p = e.last_p();
    const Tensor& qr = e.last_q_re();
    const Tensor& qi = e.last_q_im();
    // last_eta is the interleaved (d, 2*ndim) complex eta with eta_scale()
    // ALREADY applied by compute_coeffs — do not scale it again here.
    const Tensor& eta = e.last_eta();

    double worst = 0.0;
    for (size_t j = 0; j < d; ++j) {
        std::vector<double> hr(h, 0.0), hi(h, 0.0);
        for (size_t t = 0; t < n; ++t) {
            for (size_t k = 0; k < h; ++k) {
                double nhr = p(j, k) * x(t, j) + (qr(j, k) * hr[k] - qi(j, k) * hi[k]);
                double nhi = (qr(j, k) * hi[k] + qi(j, k) * hr[k]);
                hr[k] = nhr; hi[k] = nhi;
            }
            double acc = 0.0;
            for (size_t k = 0; k < h; ++k) {
                acc += eta(j, 2 * k) * hr[k] - eta(j, 2 * k + 1) * hi[k];  // Re(eta*h)
            }
            acc += e.omega(j, 0) * x(t, j);
            worst = std::max(worst, std::abs(y(t, j) - acc));
        }
    }
    cout << "    worst abs err = " << scientific << setprecision(3) << worst << defaultfloat << endl;
    check("recurrence matches hand-rolled reference (< 1e-12)", worst < 1e-12);
}

// =====================================================================
// Test 9: Determinism
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 9: determinism --" << endl;

    ComplexEMA e(6, 4);
    Tensor x = rand_tensor(5, 6, 1.0, 1234);
    Tensor y1 = e.forward(x);
    Tensor y2 = e.forward(x);
    double md = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j) md = std::max(md, std::abs(y1(i, j) - y2(i, j)));
    check("two consecutive forwards bit-exact", md == 0.0);
}

// =====================================================================
// Test 10: real EMA specialization
// =====================================================================
//
// Zeroing eta_im makes the OUTPUT real, but it does NOT make the state real:
// q is a complex rotation, so h_im can still be non-zero and h_re still
// depends on h_im. A plain (real) EMA results only when q is real too, i.e.
// when every phase phi_{j,k} = (k+1)*sigma(theta_j)*2*pi/h is a multiple of pi.
// With h=1 and theta=0 we get sigma(0)*2*pi = pi exactly, so q = -|q| and
// h stays real for all t. (With h=2, theta=0 the k=0 mode sits at phi=pi/2
// and q is purely imaginary — the recurrence is NOT a real EMA there.)
static void test_real_path() {
    cout << endl << "-- Test 10: real EMA specialization --" << endl;

    const size_t n = 3, d = 2, h = 1;   // h=1 => the only phase is pi => q real
    ComplexEMA e(d, h);
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k) {
            e.alpha(j, k) = 0.1 * (j + 1);
            e.delta(j, k) = 0.2 * k;
            e.eta_im(j, k) = 0.0;             // pure real output
            e.eta_re(j, k) = 1.0;
        }
    for (size_t j = 0; j < d; ++j) { e.theta(j, 0) = 0.0; e.omega(j, 0) = 0.0; }
    e.compute_coeffs();

    // Precondition: q must actually be real, else the reference below is not
    // a real EMA. Guards against the test silently testing the wrong thing.
    double worst_im = 0.0;
    for (size_t j = 0; j < d; ++j)
        for (size_t k = 0; k < h; ++k)
            worst_im = std::max(worst_im, std::abs(e.last_q_im()(j, k)));
    cout << "    max |Im(q)| = " << scientific << setprecision(3) << worst_im
         << defaultfloat << endl;
    check("precondition: q is real (max |Im q| < 1e-12)", worst_im < 1e-12);

    Tensor x = rand_tensor(n, d, 1.0, 555);
    Tensor y = e.forward(x);
    e.compute_coeffs();
    const Tensor& p = e.last_p();
    const Tensor& qr = e.last_q_re();
    const double scale = e.eta_scale();

    double worst = 0.0;
    for (size_t j = 0; j < d; ++j) {
        std::vector<double> hr(h, 0.0);
        for (size_t t = 0; t < n; ++t) {
            double acc = 0.0;
            for (size_t k = 0; k < h; ++k) {
                // real-only recurrence (q is real, so the state stays real)
                hr[k] = p(j, k) * x(t, j) + qr(j, k) * hr[k];
                acc += 1.0 * scale * hr[k];    // eta_re = 1, eta_im = 0
            }
            worst = std::max(worst, std::abs(y(t, j) - acc));
        }
    }
    cout << "    worst abs err = " << scientific << setprecision(3) << worst << defaultfloat << endl;
    check("real specialization matches hand-rolled EMA", worst < 1e-12);
}

int main() {
    cout << "==========================================" << endl;
    cout << "  ComplexEMA Tests" << endl;
    cout << "  MEGALODON §3.1 (Eq. 2-3)" << endl;
    cout << "==========================================" << endl;

    test_constructor();
    test_parameter_contract();
    test_decay_property();
    test_theta_spacing();
    test_forward_shape();
    test_hand_derived_single();
    test_zero_initial_state();
    test_recurrence_reference();
    test_determinism();
    test_real_path();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
