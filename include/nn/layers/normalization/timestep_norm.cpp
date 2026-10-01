#include "timestep_norm.h"
#include <stdexcept>
#include <cmath>

// ============================================================================
// TimestepNorm — MEGALODON §3.2 (Eq. 4) + Appendix B.2 (plus-1 reparam).
//
// Forward, for each timestep t = 0..n-1 (cnt = t+1) and group g:
//
//   mu[t][g]    = (1/(cnt*dg)) * sum_{i<=t} sum_{j in g} x[i][j]
//   var[t][g]   = (1/(cnt*dg)) * sum_{i<=t} sum_{j in g} (x[i][j]-mu[t][g])^2
//   y[t][j]     = (gamma[j]+1) * (x[t][j]-mu[t][g(j)]) * inv_std[t][g(j)]
//                 + beta[j]
//
// ---------------------------------------------------------------------------
// Backward derivation (all partials are w.r.t. the *statistics*, which are
// themselves functions of x — this is the part that distinguishes a correct
// backward from a "gradient through a constant" one).
//
// Let G[t][j] = dL/dy[t][j] and xhat[t][j] = (x[t][j]-mu[t][g])*inv_std[t][g].
//
//   d beta_j           = sum_t G[t][j]
//   d gamma_j          = sum_t G[t][j] * xhat[t][j]
//   d xhat[t][j]       = G[t][j] * (gamma[j] + 1)                          (A)
//
// Unpacking xhat = (x - mu) * inv_std with inv_std = (var+eps)^{-1/2}:
//   for j in group g of row t:
//     dx[t][j]      += d_xhat[t][j] * inv_std[t][g]                       (direct)
//     d mu[t][g]    += -d_xhat[t][j] * inv_std[t][g]
//     d inv_std[t][g] += d_xhat[t][j] * (x[t][j] - mu[t][g])
//   d var[t][g] += -0.5 * d inv_std[t][g] * inv_std[t][g]^3
//
// Now the cumulative-statistics chain. With S_t = sum_{i<=t} sum_{j in g} x[i][j]
// and Q_t = sum_{i<=t} sum_{j in g} (x[i][j]-mu_t)^2:
//
//   mu_t  = S_t / (cnt*dg)      =>  d mu_t / d x[i][j] = 1_{i<=t, j in g} / (cnt*dg)
//   var_t = Q_t / (cnt*dg)
//
//   dQ_t/dx[i][j] = 2*1_{i<=t,j in g}*(x[i][j]-mu_t) - 2*mu_t*cnt*dg*dmu_t/dx[i][j]
//                 = 2*1_{i<=t,j in g}*(x[i][j] - 2*mu_t)
//
//   d var_t / d x[i][j] = 2*1_{i<=t,j in g}*(x[i][j] - 2*mu_t) / (cnt*dg)
//
// Assembling (using 1_{i<=t} == "t runs from i upward"):
//
//   dx[i][j] += sum_{t>=i} [ d mu[t][g(j)] + d var[t][g(j)]*2*(x[i][j]-2*mu[t][g(j)]) ]
//                    / (cnt_t * dg)
//
// This is O(n^2 * d). The forward statistics are themselves O(n^2 * d) (every
// cumulative mean is a fresh reduction), so the layer is O(n^2 * d) overall
// and consistent. A Welford/Kahan streaming form (the paper's Appendix B.1
// GPU implementation) would make both O(n * d); that is a future optimization
// and deliberately not taken here because the direct form is what finite
// differences can check most transparently.
// ============================================================================

TimestepNorm::TimestepNorm(size_t features, size_t num_groups, double eps)
    : features_(features), num_groups_(num_groups), eps_(eps) {
    if (features == 0) throw std::invalid_argument("TimestepNorm: features must be > 0");
    if (eps <= 0.0) throw std::invalid_argument("TimestepNorm: eps must be > 0");

    // num_groups == 0 means one group spanning all features (the paper's
    // default). An explicit k in [1, features] is honoured as-is, including
    // k == features (group size 1) and k == 1 (group size features).
    if (num_groups > features) {
        throw std::invalid_argument("TimestepNorm: num_groups must be <= features");
    }
    if (num_groups != 0 && features % num_groups != 0) {
        throw std::invalid_argument("TimestepNorm: features must be divisible by num_groups");
    }
    size_t groups = (num_groups == 0) ? 1 : num_groups;
    num_groups_ = groups;
    group_size_ = features / groups;

    gamma = Tensor::zeros(1, features);
    beta = Tensor::zeros(1, features);
    grad_gamma_ = Tensor::zeros(1, features);
    grad_beta_ = Tensor::zeros(1, features);
}

Tensor TimestepNorm::forward(const Tensor& input) {
    const size_t n = input.rows;
    const size_t d = input.cols;
    const size_t g = num_groups_;
    const size_t dg = group_size_;

    last_x = input;
    last_mu = Tensor(n, g);
    last_var = Tensor(n, g);
    last_inv_std = Tensor(n, g);

    Tensor out(n, d);

    for (size_t t = 0; t < n; ++t) {
        const double cnt = static_cast<double>(t + 1);

        for (size_t gi = 0; gi < g; ++gi) {
            const size_t j0 = gi * dg;
            const size_t j1 = j0 + dg;

            double mu = 0.0;
            for (size_t i = 0; i <= t; ++i)
                for (size_t j = j0; j < j1; ++j) mu += input(i, j);
            mu /= cnt * static_cast<double>(dg);

            double var = 0.0;
            for (size_t i = 0; i <= t; ++i)
                for (size_t j = j0; j < j1; ++j) {
                    const double d_ij = input(i, j) - mu;
                    var += d_ij * d_ij;
                }
            var /= cnt * static_cast<double>(dg);

            last_mu(t, gi) = mu;
            last_var(t, gi) = var;
            last_inv_std(t, gi) = 1.0 / std::sqrt(var + eps_);
        }

        for (size_t gi = 0; gi < g; ++gi) {
            const size_t j0 = gi * dg;
            const size_t j1 = j0 + dg;
            const double inv = last_inv_std(t, gi);
            const double mu = last_mu(t, gi);
            for (size_t j = j0; j < j1; ++j) {
                const double xhat = (input(t, j) - mu) * inv;
                // Plus-1 reparameterization (Appendix B.2).
                out(t, j) = (gamma(0, j) + 1.0) * xhat + beta(0, j);
            }
        }
    }

    return out;
}

Tensor TimestepNorm::backward(const Tensor& grad_output, double learning_rate) {
    const size_t n = last_x.rows;
    const size_t d = last_x.cols;
    const size_t g = num_groups_;
    const size_t dg = group_size_;

    // Per-group accumulators for the two statistics, indexed by timestep.
    Tensor d_mu(n, g);
    Tensor d_inv(n, g);
    Tensor xhat_cache(n, d);

    grad_beta_ = Tensor::zeros(1, d);
    grad_gamma_ = Tensor::zeros(1, d);
    grad_x = Tensor::zeros(n, d);

    // Pass 1: elementwise chain + the direct path into x.
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            const size_t gi = j / dg;
            const double G = grad_output(t, j);
            const double inv = last_inv_std(t, gi);
            const double mu = last_mu(t, gi);

            const double xhat = (last_x(t, j) - mu) * inv;
            xhat_cache(t, j) = xhat;

            grad_beta_(0, j) += G;
            grad_gamma_(0, j) += G * xhat;

            // (A) d xhat[t][j] / d y[t][j], then unpack xhat.
            const double dxhat = G * (gamma(0, j) + 1.0);

            grad_x(t, j) += dxhat * inv;             // direct
            d_mu(t, gi) += -dxhat * inv;             // through the mean
            d_inv(t, gi) += dxhat * (last_x(t, j) - mu);
        }
    }

    // Pass 2: inv_std -> var, then scatter the cumulative-statistics chain
    // back onto every x[i][j] with i <= t.
    for (size_t t = 0; t < n; ++t) {
        const double cnt = static_cast<double>(t + 1);
        const double scale = 1.0 / (cnt * static_cast<double>(dg));

        for (size_t gi = 0; gi < g; ++gi) {
            // d inv_std / d var = -0.5 * inv_std^3
            const double dvar = -0.5 * d_inv(t, gi) * std::pow(last_inv_std(t, gi), 3.0);
            if (dvar == 0.0 && d_mu(t, gi) == 0.0) continue;

            const size_t j0 = gi * dg;
            const size_t j1 = j0 + dg;

            // d mu_t/dx[i][j]  = 1_{i<=t, j in g} / (cnt*dg)
            // d var_t/dx[i][j] = 2*1_{i<=t, j in g}*(x[i][j] - mu_t) / (cnt*dg)
            //
            // The (x - mu_t) form — NOT (x - 2*mu_t). Writing
            //   var_t = S2_t/(cnt*dg) - mu_t^2   with S2_t = sum_{i<=t} x_ij^2
            // gives d var_t/dx_ij = 2*x_ij/(cnt*dg) - 2*mu_t/(cnt*dg). Since
            // mu_t and cnt depend on t only (never on i), the causal indicator
            // 1_{i<=t} multiplies the WHOLE term. Double-counting the mean here
            // (writing x - 2*mu_t) silently produces a wrong input gradient
            // while leaving the gamma/beta gradients — which do not touch this
            // term — perfectly correct.
            for (size_t i = 0; i <= t; ++i) {
                for (size_t j = j0; j < j1; ++j) {
                    grad_x(i, j) += (d_mu(t, gi) + 2.0 * dvar * (last_x(i, j) - last_mu(t, gi))) * scale;
                }
            }
        }
    }

    update_weights(learning_rate);
    return grad_x;
}

void TimestepNorm::update_weights(double learning_rate) {
    if (learning_rate == 0.0) return;
    for (size_t j = 0; j < features_; ++j) {
        gamma(0, j) -= learning_rate * grad_gamma_(0, j);
        beta(0, j) -= learning_rate * grad_beta_(0, j);
    }
}

void TimestepNorm::zero_grad() {
    grad_gamma_ = Tensor::zeros(1, features_);
    grad_beta_ = Tensor::zeros(1, features_);
    grad_x = Tensor::zeros(0, 0);
}

std::vector<Tensor*> TimestepNorm::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> TimestepNorm::gradients() {
    return {&grad_gamma_, &grad_beta_};
}
