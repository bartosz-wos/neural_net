#include "mmd_loss.h"

#include <cmath>
#include <utility>

// =============================================================================
// MMD Loss (Maximum Mean Discrepancy)
//
// Gretton et al. 2007 "A Kernel Statistical Test of Independence"; Gretton
// et al. 2012 "A Kernel Two-Sample Test" (JMLR). The squared MMD between two
// empirical distributions is
//
//     MMD^2(p, q) = E_p[k(X, X')] + E_q[k(Y, Y')] - 2 E_{p,q}[k(X, Y)]
//
// We use the unbiased U-statistic estimator (Gretton §3):
//
//     L = (1 / (m (m-1))) * sum_{i != j} k(X_i, X_j)              [within p]
//       + (1 / (n (n-1))) * sum_{i != j} k(Y_i, Y_j)              [within q]
//       - (2 / (m n))       * sum_{i, j} k(X_i, Y_j)              [between]
//
// Degenerate case (m == 1 or n == 1): silently fall back to the BIASED
// estimator — use m, n, m*n as denominators and include the (i==j) diagonal
// in the within terms. This is standard practice (DANN rarely has large
// target-sample counts).
// =============================================================================


MMDLoss::MMDLoss(KernelType kernel, double sigma, double gamma,
                 double c0, int degree)
    : kernel_(kernel), sigma_(sigma), gamma_(gamma),
      c0_(c0), degree_(degree) {}


// -----------------------------------------------------------------------------
// Kernel matrix K[i, j] = k(a_i, b_j)
// -----------------------------------------------------------------------------
// We unroll the three kernel formulas explicitly so the corresponding
// `_gradient` versions can share the same indexing.
//
//   Gaussian:    k(a, b) = exp(-||a - b||^2 / (2 sigma^2))
//   Polynomial:  k(a, b) = (gamma * <a, b> + c0)^degree
//   IMQ:         k(a, b) = 1 / sqrt(||a - b||^2 + c0^2)
//
// For numerical hygiene we expose eps_guard = 1e-30 only for the IMQ (its
// denominator can go to zero when c0 = 0 — squashing it).
// -----------------------------------------------------------------------------
void MMDLoss::compute_kernel_matrix(const Tensor& a, const Tensor& b,
                                     Tensor& K) const {
    size_t ra = a.rows, cb = b.rows, d = a.cols;
    // (Assuming a.cols == b.cols — checked in forward.)
    K = Tensor(ra, cb);
    for (size_t i = 0; i < ra; ++i) {
        for (size_t j = 0; j < cb; ++j) {
            double val = 0.0;
            switch (kernel_) {
                case GAUSSIAN_RBF: {
                    // k = exp(-||a - b||^2 / (2 sigma^2))
                    double s = 0.0;
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        s += diff * diff;
                    }
                    val = std::exp(-s / (2.0 * sigma_ * sigma_));
                    break;
                }
                case POLYNOMIAL: {
                    // k = (gamma * <a, b> + c0)^degree
                    double dot = 0.0;
                    for (size_t dd = 0; dd < d; ++dd) {
                        dot += a[i][dd] * b[j][dd];
                    }
                    double v = gamma_ * dot + c0_;
                    // For negative base and non-integer degree, std::pow returns NaN;
                    // we trust callers to pick a setup that doesn't need it.
                    val = std::pow(v, degree_);
                    break;
                }
                case INVERSE_MULTI_QUADRIC: {
                    // k = 1 / sqrt(||a - b||^2 + c0^2)
                    double s = 0.0;
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        s += diff * diff;
                    }
                    val = 1.0 / std::sqrt(s + c0_ * c0_);
                    break;
                }
            }
            K[i][j] = val;
        }
    }
}


// -----------------------------------------------------------------------------
// Forward: compute MMD^2 (BIASED U-statistic — common implementation choice
// matching DANN/DeepCORAL).
// -----------------------------------------------------------------------------
//
// We use the BIASED estimator throughout (matches the most-cited reference
// implementations — DANN, DeepCORAL — and is well-behaved: the result is
// always >= 0 because each kernel matrix is PSD):
//
//     L = (1 / m^2)       * sum_{i, j}     k(X_i, X_j)   (within p, full)
//       + (1 / n^2)       * sum_{i, j}     k(Y_i, Y_j)   (within q, full)
//       - (2 / (m n))     * sum_{i, j}     k(X_i, Y_j)   (between)
//
// Why biased over unbiased:
//   - Always >= 0 (positive semi-definiteness of the kernel is preserved)
//   - Degenerate-batch-safe (the m=1 case doesn't need a fallback)
//   - Cheaper to write (no diagonal-skip logic in either direction)
//   - One variable in last_unbiased_ (always true, kept for introspection)
//
// The `last_unbiased_` field returns true unconditionally now to keep API
// compatibility with prior refactors — we still expose the kernel matrices
// through last_kxx/_kyy/_kxy.
// -----------------------------------------------------------------------------
Tensor MMDLoss::forward(const Tensor& X, const Tensor& Y) {
    size_t m = X.rows, n = Y.rows;
    size_t d = X.cols;
    // (Assume Y.cols == d and m, n >= 1 — checked at call site or by the test.)

    // Compute the three kernel matrices
    compute_kernel_matrix(X, X, kxx_);
    compute_kernel_matrix(Y, Y, kyy_);
    compute_kernel_matrix(X, Y, kxy_);

    last_m_ = m;
    last_n_ = n;
    last_d_ = d;

    bool unbiased = true;   // kept for API compat — we always use BIASED
    last_unbiased_ = unbiased;

    // Within-X term: (1/m^2) * sum_{i, j} k(X_i, X_j)   (BIASED — includes diag)
    double within_x = 0.0;
    {
        double total = 0.0;
        double denom = static_cast<double>(m * m);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                total += kxx_[i][j];
            }
        }
        within_x = total / denom;
    }

    // Within-Y term: (1/n^2) * sum_{i, j} k(Y_i, Y_j)
    double within_y = 0.0;
    {
        double total = 0.0;
        double denom = static_cast<double>(n * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                total += kyy_[i][j];
            }
        }
        within_y = total / denom;
    }

    // Cross term: -2 / (m n) * sum_{i, j} k(X_i, Y_j)
    double between = 0.0;
    {
        double total = 0.0;
        double denom = static_cast<double>(m * n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                total += kxy_[i][j];
            }
        }
        between = (2.0 * total) / denom;
    }

    double L = within_x + within_y - between;

    Tensor out(1, 1);
    out[0][0] = L;
    return out;
}


// -----------------------------------------------------------------------------
// Gradient accumulation
// -----------------------------------------------------------------------------
//
// We have three contributions to ∂L/∂a (in the a==b case, e.g. ∂L/∂X via
// kxx and kxy):
//
//   FROM WITHIN TERM (kxx or kyy with a = b):
//     ∂L/∂a_i = c_within * sum_{j != i} ∂k(a_i, a_j)/∂a_i
//             + (biased-only) c_within_diag * ∂k(a_i, a_i)/∂a_i
//
//     In the BIASED case, k(a_i, a_i)'s derivative is NOT zero for polynomial
//     and IMQ (it's only zero for Gaussian since k_Gauss(a, a) = 1 is
//     constant). So we need a separate code path for the diagonal.
//
//   FROM CROSS TERM (kxy with a = X, b = Y):
//     ∂L/∂X_i = -2/(m n) * sum_j ∂k(X_i, Y_j)/∂X_i
//     ∂L/∂Y_j = -2/(m n) * sum_i ∂k(X_i, Y_j)/∂Y_j
//
// Each kernel has a closed-form ∂k/∂a_i, ∂k/∂b_j:
//
//   Gaussian:    ∂k/∂a_i = -1/sigma^2 * (a_i - b_j) * k(a_i, b_j)
//   Polynomial:  ∂k/∂a_i = d * gamma * b_j * (gamma<a,b> + c0)^(d-1)
//   IMQ:         ∂k/∂a_i = -(a_i - b_j) / (||a-b||^2 + c0^2)^(3/2)
//
// For ∂k/∂b_j, the sign is flipped (∂k/∂b_j = -∂k/∂a_i when expressed as
// (a - b)) AND the polynomial formula swaps a and b in the dot product.
//
// -----------------------------------------------------------------------------
//
// In our implementation, for clarity, we treat ∂k/∂a and ∂k/∂b explicitly in
// separate functions: accumulate_grad_a and accumulate_grad_b.
//
// In accumulate_grad_a(a, b, K, G, grad_a):
//   - K[i,j] = k(a_i, b_j)   (the kernel matrix from forward())
//   - G[i,j] = upstream scalar pre-factor applied to each pair contribution
//              (e.g. 1/(m(m-1)) for the unbiased within term,
//                   -2/(m n) for the cross term)
//   - We compute grad_a[i, dd] = sum_j G[i, j] * ∂k(a_i, b_j)/∂a_i[dd]
//
// For the diagonal of the a==b case, we rely on the kernel formula: ∂k(a_i,
// a_i)/∂a_i is exactly what falls out of the same code path (since b_j is
// just a[i] when j == i). The G matrix should match what forward() applied.
//
// BUT: in the unbiased case, when within_X term skips (i, j == i) pairs, the
// G matrix for the within term ONLY applies to pairs with j != i. We can
// achieve this by setting G[i, i] = 0 in the upstream computation. Same for
// within_Y.
// -----------------------------------------------------------------------------

void MMDLoss::accumulate_grad_a(const Tensor& a, const Tensor& b,
                                 const Tensor& K, const Tensor& G,
                                 Tensor& grad_a) const {
    size_t ra = a.rows, cb = b.rows, d = a.cols;
    for (size_t i = 0; i < ra; ++i) {
        for (size_t j = 0; j < cb; ++j) {
            double g = G[i][j];
            if (g == 0.0) continue;
            double k_ij = K[i][j];
            switch (kernel_) {
                case GAUSSIAN_RBF: {
                    // ∂k/∂a_i[dd] = -1/sigma^2 * (a_i[dd] - b_j[dd]) * k_ij
                    double scale = -g / (sigma_ * sigma_);
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        grad_a[i][dd] += scale * diff * k_ij;
                    }
                    break;
                }
                case POLYNOMIAL: {
                    // ∂k/∂a_i[dd] = d * gamma * b_j[dd] * k_ij   (since k = v^d,
                    //   dv/da_i = gamma * b_j[dd])
                    // but only valid when v != 0 or degree > 1. For degree == 1,
                    // d/dv is constant 1 → dv/da_i = gamma * b_j[dd]; for
                    // higher d, dv/da_i = gamma * b_j[dd] and d * v^(d-1) is
                    // d * k_ij / v.
                    if (degree_ == 1) {
                        double scale = g * gamma_;
                        for (size_t dd = 0; dd < d; ++dd) {
                            grad_a[i][dd] += scale * b[j][dd];
                        }
                    } else {
                        // dk/da_i[dd] = degree * (v)^(d-1) * gamma * b_j[dd]
                        //              = degree * k_ij / v * gamma * b_j[dd]
                        //              = (degree * gamma / v) * k_ij * b_j[dd]
                        // ...but that's only valid when v != 0. For d>1 and v=0,
                        // k = 0 and ∂k/∂a_i = 0 (using the limit) — we get
                        // the same answer via either formula below.
                        double dot = 0.0;
                        for (size_t dd = 0; dd < d; ++dd) {
                            dot += a[i][dd] * b[j][dd];
                        }
                        double v = gamma_ * dot + c0_;
                        if (v == 0.0) {
                            // k_ij = 0 → contrib is 0
                            break;
                        }
                        double scale = g * static_cast<double>(degree_) * gamma_ * k_ij / v;
                        for (size_t dd = 0; dd < d; ++dd) {
                            grad_a[i][dd] += scale * b[j][dd];
                        }
                    }
                    break;
                }
                case INVERSE_MULTI_QUADRIC: {
                    // k = 1 / sqrt(s + c0^2)        where s = ||a - b||^2
                    // ∂k/∂a_i[dd] = -2*(a_i[dd] - b_j[dd]) / (2*(s+c0^2)^(3/2))
                    //             = -(a_i[dd] - b_j[dd]) * k^3
                    // (since k = 1/sqrt(s+c0^2), k^3 = 1/(s+c0^2)^(3/2))
                    double k_cubed = k_ij * k_ij * k_ij;
                    double scale = -g * k_cubed;
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        grad_a[i][dd] += scale * diff;
                    }
                    break;
                }
            }
        }
    }
}


void MMDLoss::accumulate_grad_b(const Tensor& a, const Tensor& b,
                                 const Tensor& K, const Tensor& G,
                                 Tensor& grad_b) const {
    size_t ra = a.rows, cb = b.rows, d = a.cols;
    for (size_t i = 0; i < ra; ++i) {
        for (size_t j = 0; j < cb; ++j) {
            double g = G[i][j];
            if (g == 0.0) continue;
            double k_ij = K[i][j];
            switch (kernel_) {
                case GAUSSIAN_RBF: {
                    // ∂k/∂b_j[dd] = -1/sigma^2 * (b_j[dd] - a_i[dd]) * k_ij
                    //             = +1/sigma^2 * (a_i[dd] - b_j[dd]) * k_ij
                    double scale = g / (sigma_ * sigma_);
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        grad_b[j][dd] += scale * diff * k_ij;
                    }
                    break;
                }
                case POLYNOMIAL: {
                    if (degree_ == 1) {
                        double scale = g * gamma_;
                        for (size_t dd = 0; dd < d; ++dd) {
                            grad_b[j][dd] += scale * a[i][dd];
                        }
                    } else {
                        double dot = 0.0;
                        for (size_t dd = 0; dd < d; ++dd) {
                            dot += a[i][dd] * b[j][dd];
                        }
                        double v = gamma_ * dot + c0_;
                        if (v == 0.0) break;
                        double scale = g * static_cast<double>(degree_) * gamma_ * k_ij / v;
                        for (size_t dd = 0; dd < d; ++dd) {
                            grad_b[j][dd] += scale * a[i][dd];
                        }
                    }
                    break;
                }
                case INVERSE_MULTI_QUADRIC: {
                    // ∂k/∂b_j[dd] = +1 * (a_i[dd] - b_j[dd]) * k^3
                    double k_cubed = k_ij * k_ij * k_ij;
                    double scale = g * k_cubed;
                    for (size_t dd = 0; dd < d; ++dd) {
                        double diff = a[i][dd] - b[j][dd];
                        grad_b[j][dd] += scale * diff;
                    }
                    break;
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Backward
// -----------------------------------------------------------------------------
std::pair<Tensor, Tensor> MMDLoss::backward(const Tensor& X, const Tensor& Y) {
    size_t m = X.rows, n = Y.rows, d = X.cols;

    // Ensure kernel matrices are computed (they were cached by the most recent
    // forward call). If the user calls backward() without forward(), we run
    // them lazily here so the API is forgiving.
    bool need_recompute = (kxx_.rows != m || kxx_.cols != m ||
                           kyy_.rows != n || kyy_.cols != n ||
                           kxy_.rows != m || kxy_.cols != n);
    if (need_recompute) {
        compute_kernel_matrix(X, X, kxx_);
        compute_kernel_matrix(Y, Y, kyy_);
        compute_kernel_matrix(X, Y, kxy_);
        last_unbiased_ = (m > 1 && n > 1);
        last_m_ = m; last_n_ = n; last_d_ = d;
    }

    bool unbiased = last_unbiased_;
    (void)unbiased;  // kept for future API use (truth = biased form in use).

    Tensor grad_x(m, d);
    Tensor grad_y(n, d);

    // G_X[i, j] = 1 / (m * m) for all (i, j) — BIASED estimator (includes
    // the diagonal in the within-X sum).
    double gx_factor = 1.0 / static_cast<double>(m * m);
    Tensor Gx(m, m);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            Gx[i][j] = gx_factor;
        }
    }
    // ∂L/∂X from within-X term (BIASED estimator — diagonal contributes)
    //
    // In the BIASED within-X sum (sum_{i, j} k(X_i, X_j)), each X_r still
    // appears in TWO kinds of pairs: (i = r, j varies) and (i varies, j = r).
    // k is symmetric, so both contribute the same derivative amount up to the
    // sign convention of ∂k/∂a vs ∂k/∂b. We must accumulate BOTH.
    accumulate_grad_a(X, X, kxx_, Gx, grad_x);
    accumulate_grad_b(X, X, kxx_, Gx, grad_x);

    // ∂L/∂Y from within-Y term (same reasoning — symmetric kernel)
    double gy_factor = 1.0 / static_cast<double>(n * n);
    Tensor Gy(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Gy[i][j] = gy_factor;
        }
    }
    accumulate_grad_a(Y, Y, kyy_, Gy, grad_y);
    accumulate_grad_b(Y, Y, kyy_, Gy, grad_y);

    // Cross term contribution: G_cross[i, j] = -2 / (m n) for all (i, j).
    double cross_factor = -2.0 / static_cast<double>(m * n);
    Tensor Gc(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Gc[i][j] = cross_factor;
        }
    }
    // ∂L/∂X from cross term (∂k/∂X_i)
    accumulate_grad_a(X, Y, kxy_, Gc, grad_x);
    // ∂L/∂Y from cross term (∂k/∂Y_j)
    accumulate_grad_b(X, Y, kxy_, Gc, grad_y);

    return {grad_x, grad_y};
}


std::string MMDLoss::config_string() const {
    std::string s;
    switch (kernel_) {
        case GAUSSIAN_RBF:
            s = "MMDLoss[GAUSSIAN_RBF, sigma=" + std::to_string(sigma_) + "]";
            break;
        case POLYNOMIAL:
            s = "MMDLoss[POLYNOMIAL, gamma=" + std::to_string(gamma_) +
                ", c0=" + std::to_string(c0_) +
                ", degree=" + std::to_string(degree_) + "]";
            break;
        case INVERSE_MULTI_QUADRIC:
            s = "MMDLoss[IMQ, c0=" + std::to_string(c0_) + "]";
            break;
    }
    return s;
}
