#include "hutchinson.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace {

// The eps floor below which the forward-over-reverse difference is pure
// cancellation noise. Documented in the header; enforced here.
constexpr double kMinEps = 1e-12;

void validate_trace_args(size_t dim, const HessianGradFn& grad, const Tensor& x,
                         int probes) {
    if (!grad) throw std::invalid_argument("hutch: gradient oracle is empty");
    if (dim == 0) throw std::invalid_argument("hutch: dim must be > 0");
    if (probes < 1) throw std::invalid_argument("hutch: probes must be >= 1");
    if (x.cols != dim)
        throw std::invalid_argument("hutch: x.cols does not match dim");
}

void validate_eps(double eps) {
    if (!(eps > 0.0) || !std::isfinite(eps))
        throw std::invalid_argument("hutch: eps must be finite and > 0");
    if (eps < kMinEps)
        throw std::invalid_argument("hutch: eps below the cancellation floor");
}

// Both trace routines share this shape and differ ONLY in the probe law:
//   tr(H) = E_z[ z'(grad(x+z) - grad(x)) ] = E_z[ z' H z ],
// and the identity holds for ANY H because the linear term of the gradient
// cancels in the difference. Sharing the body is deliberate -- a duplicated
// contraction is a duplicated place for the two laws to drift apart.
template <typename Draw>
double trace_impl(size_t dim, const HessianGradFn& grad, const Tensor& x,
                  int probes, unsigned seed, Draw draw) {
    validate_trace_args(dim, grad, x, probes);

    std::mt19937 rng(seed ? seed : 20240614u);
    Tensor g0 = grad(x);
    if (g0.rows == 0 || g0.cols == 0)
        throw std::invalid_argument("hutch: gradient oracle returned an empty tensor");

    double acc = 0.0;
    for (int k = 0; k < probes; ++k) {
        Tensor zp = x.clone();
        for (size_t j = 0; j < dim; ++j) zp(0, j) += draw(rng);

        Tensor gp = grad(zp);
        // z is the PERTURBATION, i.e. (zp - x), NOT zp itself. Contracting
        // against zp instead silently adds an x'Hz term and biases the
        // estimate -- a bug no unbiasedness fixture at x == 0 would catch.
        double q = 0.0;
        for (size_t j = 0; j < dim; ++j)
            q += (zp(0, j) - x(0, j)) * (gp(0, j) - g0(0, j));
        acc += q;
    }
    return acc / static_cast<double>(probes);
}

}  // namespace

double hutch_trace_rademacher(size_t dim, const HessianGradFn& grad,
                              const Tensor& x, int probes, unsigned seed) {
    // A Rademacher probe is +-1 with equal probability. It must NOT be 0: a
    // zero probe makes z'Hz identically 0 and the estimator returns 0 for
    // every probe, which looks like a converged answer.
    return trace_impl(dim, grad, x, probes, seed,
                      [](std::mt19937& rng) {
                          std::uniform_int_distribution<int> coin(0, 1);
                          return coin(rng) ? 1.0 : -1.0;
                      });
}

double hutch_trace_gaussian(size_t dim, const HessianGradFn& grad,
                            const Tensor& x, int probes, unsigned seed) {
    return trace_impl(dim, grad, x, probes, seed,
                      [](std::mt19937& rng) {
                          std::normal_distribution<double> nd(0.0, 1.0);
                          return nd(rng);
                      });
}

Tensor hutch_hvp(const HessianGradFn& grad, const Tensor& x, const Tensor& v,
                 double eps) {
    if (!grad) throw std::invalid_argument("hutch: gradient oracle is empty");
    validate_eps(eps);
    if (x.cols != v.cols)
        throw std::invalid_argument("hutch: x.cols does not match v.cols");
    if (x.rows != 1 || v.rows != 1)
        throw std::invalid_argument("hutch: x and v must be single-row tensors");

    Tensor g0 = grad(x);
    Tensor gp = grad(x + v * eps);

    // Built fresh: never aliases g0 or gp.
    Tensor hv(g0.rows, g0.cols);
    for (size_t j = 0; j < g0.cols; ++j)
        hv(0, j) = (gp(0, j) - g0(0, j)) / eps;
    return hv;
}