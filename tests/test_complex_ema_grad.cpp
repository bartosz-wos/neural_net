// Central-difference gradient checks for ComplexEMA.
//
// FD reference is a fully independent Python-derived formula implemented here
// from the MEGALODON recurrence (arXiv 2404.08801 Eq. 2-3) — it does not call
// ComplexEMA::forward, so agreement is a real cross-check.
#include "../include/nn/layers/recurrent/complex_ema.h"
#include "../include/nn/core/tensor.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int passed = 0, failed = 0;

static void check(const std::string& name, bool ok, const std::string& info = "") {
    if (ok) { ++passed; }
    else { ++failed; printf("  FAIL %s %s\n", name.c_str(), info.c_str()); }
}

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Independent forward: computes loss = 0.5 * sum_t sum_j gy[t][j] * y[t][j]^2
// directly from the parameter tensors, without touching ComplexEMA.
static double fd_loss(const Tensor& x, const Tensor& gy,
                      const Tensor& alpha, const Tensor& delta, const Tensor& theta,
                      const Tensor& eta_re, const Tensor& eta_im, const Tensor& omega) {
    const size_t n = x.rows, d = x.cols;
    const size_t h = alpha.cols;
    const double scale = 1.0 / std::sqrt(static_cast<double>(h));
    const double step = 2.0 * std::acos(-1.0) / static_cast<double>(h);

    std::vector<double> hr(d * h, 0.0), hi(d * h, 0.0);
    double loss = 0.0;
    for (size_t t = 0; t < n; ++t) {
        std::vector<double> p(d * h), qre(d * h), qim(d * h);
        for (size_t j = 0; j < d; ++j) {
            const double base = sigmoid(theta[j][0]) * step;
            for (size_t k = 0; k < h; ++k) {
                const size_t i = j * h + k;
                p[i]  = sigmoid(alpha[j][k]);
                const double dl = sigmoid(delta[j][k]);
                const double phi = static_cast<double>(k + 1) * base;
                const double mag = 1.0 - p[i] * dl;
                qre[i] = mag * std::cos(phi);
                qim[i] = mag * std::sin(phi);
                const double nr = p[i] * x[t][j] + qre[i] * hr[i] - qim[i] * hi[i];
                const double ni = qre[i] * hi[i] + qim[i] * hr[i];
                hr[i] = nr; hi[i] = ni;
            }
            double acc = 0.0;
            for (size_t k = 0; k < h; ++k) {
                const size_t i = j * h + k;
                acc += eta_re[j][k] * scale * hr[i] - eta_im[j][k] * scale * hi[i];
            }
            acc += omega[j][0] * x[t][j];
            loss += 0.5 * gy[t][j] * acc * acc;
        }
    }
    return loss;
}

static Tensor make_input(size_t n, size_t d, unsigned seed) {
    // xorshift so the test is deterministic without touching global RNG state
    unsigned s = seed * 2654435761u + 1u;
    Tensor t(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            t[i][j] = ((s % 2000) / 1000.0) - 1.0;   // in (-1, 1)
        }
    return t;
}

static void set_params(ComplexEMA& m, unsigned seed) {
    unsigned s = seed * 40503u + 7u;
    auto rnd = [&s](double lo, double hi) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return lo + ((s % 2001) / 2000.0) * (hi - lo);
    };
    for (auto& t : {&m.alpha_, &m.delta_, &m.theta_, &m.eta_re_, &m.eta_im_, &m.omega_})
        for (size_t i = 0; i < t->rows; ++i)
            for (size_t j = 0; j < t->cols; ++j) (*t)[i][j] = rnd(-1.0, 1.0);
}

// Return false and print if any entry of `a` differs from `b` by more than tol.
static bool close(const Tensor& a, const Tensor& b, double tol, const std::string& label) {
    double worst = 0.0;
    size_t wi = 0, wj = 0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            const double e = std::fabs(a[i][j] - b[i][j]);
            if (e > worst) { worst = e; wi = i; wj = j; }
        }
    if (worst > tol) {
        char buf[256];
        snprintf(buf, sizeof(buf), "worst=%.3e at [%zu][%zu] ana=%.8f fd=%.8f (tol %.0e)",
                 worst, wi, wj, a[wi][wj], b[wi][wj], tol);
        check(label, false, buf);
        return false;
    }
    check(label, true);
    return true;
}

int main() {
    // n=4 (>=3 so the multi-timestep adjoint sum is exercised), d=3, h=2.
    // Random (non-uniform) params so row/col asymmetries are not masked.
    const size_t N = 4, D = 3, H = 2;

    for (unsigned seed = 1; seed <= 2; ++seed) {
        ComplexEMA m(D, H);
        set_params(m, seed);
        Tensor x = make_input(N, D, seed + 100);
        Tensor gy = make_input(N, D, seed + 200);

        const Tensor y = m.forward(x);
        { // sanity: backward's internal y must equal forward's y
            double worst = 0.0;
            Tensor gz(N, D);  // zeros -> G=0, not useful; instead check d/domega
            // dL/domega_j = sum_t (gy*y)[t][j] * x[t][j]  -- closed form
            Tensor fd_om(D, 1);
            for (size_t j = 0; j < D; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t) acc += gy[t][j] * y[t][j] * x[t][j];
                fd_om[j][0] = acc;
            }
            Tensor gyy_chk(N, D);
            for (size_t i = 0; i < N; ++i) for (size_t j = 0; j < D; ++j) gyy_chk[i][j] = gy[i][j]*y[i][j];
            m.backward(gyy_chk, 0.0);
            double w2 = 0.0;
            for (size_t j = 0; j < D; ++j) w2 = std::max(w2, std::fabs(m.grad_omega_[j][0]-fd_om[j][0]));
            if (w2 > 1e-10) { printf("  [sanity] closed-form d/domega mismatch=%.3e\n", w2); }
        }
        // Upstream gradient: dL/dy with L = 0.5 * sum gy * y^2  =>  gy * y
        Tensor gyy(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j) gyy[i][j] = gy[i][j] * y[i][j];

        m.backward(gyy, 0.0);

        // Reference loss value at the base point
        const double L0 = fd_loss(x, gy, m.alpha_, m.delta_, m.theta_,
                                  m.eta_re_, m.eta_im_, m.omega_);

        const double eps = 1e-6;
        const struct { const char* name; Tensor ComplexEMA::*param; Tensor ComplexEMA::*grad; } items[] = {
            {"alpha",  &ComplexEMA::alpha_,  &ComplexEMA::grad_alpha_},
            {"delta",  &ComplexEMA::delta_,  &ComplexEMA::grad_delta_},
            {"theta",  &ComplexEMA::theta_,  &ComplexEMA::grad_theta_},
            {"eta_re", &ComplexEMA::eta_re_, &ComplexEMA::grad_eta_re_},
            {"eta_im", &ComplexEMA::eta_im_, &ComplexEMA::grad_eta_im_},
            {"omega",  &ComplexEMA::omega_,  &ComplexEMA::grad_omega_},
        };

        for (const auto& it : items) {
            Tensor& P = m.*(it.param);
            Tensor G = m.*(it.grad);
            Tensor fd(P.rows, P.cols);
            for (size_t i = 0; i < P.rows; ++i) {
                for (size_t j = 0; j < P.cols; ++j) {
                    const double save = P[i][j];
                    P[i][j] = save + eps;
                    const double Lp = fd_loss(x, gy, m.alpha_, m.delta_, m.theta_,
                                             m.eta_re_, m.eta_im_, m.omega_);
                    P[i][j] = save - eps;
                    const double Lm = fd_loss(x, gy, m.alpha_, m.delta_, m.theta_,
                                             m.eta_re_, m.eta_im_, m.omega_);
                    P[i][j] = save;
                    fd[i][j] = (Lp - Lm) / (2 * eps);
                }
            }
            close(G, fd, 1e-7, std::string("seed") + std::to_string(seed) + " d/d" + it.name);
        }

        // Input gradient: backward() returns dL/dx
        Tensor gx = m.backward(gyy, 0.0);
        Tensor fdx(N, D);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                const double save = x[i][j];
                x[i][j] = save + eps;
                const double Lp = fd_loss(x, gy, m.alpha_, m.delta_, m.theta_,
                                         m.eta_re_, m.eta_im_, m.omega_);
                x[i][j] = save - eps;
                const double Lm = fd_loss(x, gy, m.alpha_, m.delta_, m.theta_,
                                         m.eta_re_, m.eta_im_, m.omega_);
                x[i][j] = save;
                fdx[i][j] = (Lp - Lm) / (2 * eps);
            }
        }
        close(gx, fdx, 1e-7, "seed" + std::to_string(seed) + " d/dx");
        (void)L0;
    }

    printf("=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
