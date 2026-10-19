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

std::vector<std::vector<Tensor>> hutch_trace_diagonal(
    Model& model, const Tensor& x, const Tensor& dL_dy,
    int probes, unsigned seed, double eps) {
    if (probes < 1) throw std::invalid_argument("hutch: probes must be >= 1");
    validate_eps(eps);

    // Flatten the (layer, tensor) list ONCE, recording each tensor's LOCAL
    // index within its own layer. Walking the flat list and recomputing the
    // local index by counting matches is what makes the result index-aligned
    // with parameters() without a second map.
    struct Entry {
        Layer* layer;
        Tensor* param;
        size_t local;      // index within this layer's own parameters()
        Tensor snapshot;   // for an exact restore
    };
    std::vector<Entry> entries;
    for (auto& holder : model.layers) {
        Layer* lp = holder.get();
        if (!lp) continue;
        auto params = lp->parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            if (!params[i]) continue;
            entries.push_back(Entry{lp, params[i], i, *params[i]});
        }
    }
    if (entries.empty())
        throw std::invalid_argument("hutch: model has no parameters");

    // ONE probe vector per parameter tensor, drawn ONCE and reused across all
    // probes. This matters: the outer-product decomposition
    //     tr(H) = sum_i z_i' H_ii z_i
    // estimates the TOTAL trace only when every tensor is perturbed along a
    // probe from the same draw. Drawing fresh z per probe is still unbiased
    // per-tensor, but then the per-tensor values are not jointly comparable.
    std::mt19937 rng(seed ? seed : 20240614u);
    std::uniform_int_distribution<int> coin(0, 1);
    std::vector<std::vector<double>> z;
    z.reserve(entries.size());
    for (const Entry& e : entries) {
        std::vector<double> zi(e.param->rows * e.param->cols);
        for (size_t k = 0; k < zi.size(); ++k) zi[k] = coin(rng) ? 1.0 : -1.0;
        z.push_back(std::move(zi));
    }

    std::vector<std::vector<Tensor>> out;
    out.reserve(probes);

    for (int p = 0; p < probes; ++p) {
        // theta <- theta + eps * z
        for (size_t i = 0; i < entries.size(); ++i) {
            Tensor& t = *entries[i].param;
            for (size_t k = 0; k < t.rows * t.cols; ++k)
                t.data[k] += eps * z[i][k];
        }

        Tensor y = model.forward(x);
        (void)y;

        // Dense::backward ACCUMULATES into grad_weights (`grad_weights +=
        // grad_w`, core/layer.cpp:78). Without this zeroing, probe k reads
        // probe k-1's gradient PLUS its own and every estimate after the first
        // is contaminated -- a failure invisible to a single-probe test.
        for (auto& holder : model.layers)
            if (holder.get()) holder.get()->zero_grad();

        model.backward(dL_dy, 0.0);

        // Restore BEFORE reading the gradients, so the caller's model is never
        // left perturbed even on an exception path.
        for (Entry& e : entries) {
            Tensor& t = *e.param;
            for (size_t k = 0; k < t.rows * t.cols; ++k)
                t.data[k] = e.snapshot.data[k];
        }

        // Read the gradients at theta + eps*z. They ARE the diagonal-block
        // estimate z_i' H_ii z_i, already in the shape set_hessian_estimates
        // expects -- no transposition, no flattening, no reindexing.
        std::vector<Tensor> diag;
        diag.reserve(entries.size());
        for (const Entry& e : entries) {
            auto grads = e.layer->gradients();
            Tensor d(e.param->rows, e.param->cols);
            if (e.local < grads.size() && grads[e.local]) d = *grads[e.local];
            diag.push_back(std::move(d));
        }
        out.push_back(std::move(diag));
    }

    return out;
}