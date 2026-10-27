#include "gaussian_process.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {
namespace utils {
namespace {

// ---------------------------------------------------------------------------
// The shared validation. Every public entry point runs these, so a caller
// gets the SAME message for the same mistake regardless of which function
// they called. The messages are DISTINCT from one another -- a substring that
// appears in two different messages discriminates nothing.
// ---------------------------------------------------------------------------
void validate_points(const Tensor& X, const Tensor* X2, const char* what) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(std::string("gp: ") + what +
                                    " points must not be empty");
    }
    if (X2 != nullptr) {
        if (X2->rows == 0 || X2->cols == 0) {
            throw std::invalid_argument(std::string("gp: ") + what +
                                        " query points must not be empty");
        }
        if (X2->cols != X.cols) {
            throw std::invalid_argument(
                "gp: query points must have the same number of columns as the "
                "training points");
        }
    }
}

void validate_hyperparameters(double lengthscale, double sigma_f,
                               double sigma_n, bool has_noise) {
    if (!std::isfinite(lengthscale) || lengthscale <= 0.0) {
        throw std::invalid_argument(
            "gp: the length-scale must be finite and strictly positive");
    }
    if (!std::isfinite(sigma_f) || sigma_f <= 0.0) {
        throw std::invalid_argument(
            "gp: the signal variance must be finite and strictly positive");
    }
    if (has_noise && (!std::isfinite(sigma_n) || sigma_n < 0.0)) {
        throw std::invalid_argument(
            "gp: the noise variance must be finite and non-negative");
    }
}

void validate_kernel(GPKernelType kernel) {
    if (kernel != GP_KERNEL_SE && kernel != GP_KERNEL_MATERN32 &&
        kernel != GP_KERNEL_MATERN52) {
        throw std::invalid_argument("gp: unknown kernel type");
    }
}

void validate_targets(const Tensor& X, const Tensor& y) {
    if (y.rows != X.rows || y.cols != 1) {
        throw std::invalid_argument(
            "gp: the target must be a column vector with one row per training "
            "point");
    }
    for (size_t i = 0; i < y.rows; ++i) {
        if (!std::isfinite(y(i, 0))) {
            throw std::invalid_argument("gp: the target must be finite");
        }
    }
}

// ---------------------------------------------------------------------------
// The squared Euclidean distance matrix, (n, m). The `max(., 0)` guards
// against a tiny negative value from the cancellation in
// |a|^2 + |b|^2 - 2 a.b, which would make sqrt() return NaN.
// ---------------------------------------------------------------------------
Tensor squared_distances(const Tensor& A, const Tensor& B) {
    Tensor D(A.rows, B.rows);
    for (size_t i = 0; i < A.rows; ++i) {
        double ai = 0.0;
        for (size_t k = 0; k < A.cols; ++k) ai += A(i, k) * A(i, k);
        for (size_t j = 0; j < B.rows; ++j) {
            double bj = 0.0, dot = 0.0;
            for (size_t k = 0; k < A.cols; ++k) {
                bj += B(j, k) * B(j, k);
                dot += A(i, k) * B(j, k);
            }
            double s = ai + bj - 2.0 * dot;
            if (s < 0.0) s = 0.0;
            D(i, j) = s;
        }
    }
    return D;
}

// The raw kernel (no amplitude applied) evaluated at a squared distance.
//
//   SE        exp(-0.5 r^2 / l^2)
//   Matern 3/2   (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)
//   Matern 5/2   (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)
//
// The squared distance is passed in (not the distance) so the SE branch does
// not need a sqrt; the Matern branches do.
double kernel_from_r2(double r2, double lengthscale, GPKernelType kernel) {
    switch (kernel) {
        case GP_KERNEL_SE: {
            // NOTE: the exponent divides by l^2, NOT l.
            return std::exp(-0.5 * r2 / (lengthscale * lengthscale));
        }
        case GP_KERNEL_MATERN32: {
            const double r = std::sqrt(r2);
            const double a = std::sqrt(3.0) * r / lengthscale;
            return (1.0 + a) * std::exp(-a);
        }
        case GP_KERNEL_MATERN52: {
            const double r = std::sqrt(r2);
            const double a = std::sqrt(5.0) * r / lengthscale;
            return (1.0 + a + 5.0 * r2 / (3.0 * lengthscale * lengthscale)) *
                   std::exp(-a);
        }
    }
    return 0.0;
}

// The noise-free kernel matrix Kf = k(X, X2) (no amplitude applied here).
Tensor kernel_matrix_impl(const Tensor& X, const Tensor& X2, double lengthscale,
                           double sigma_f, GPKernelType kernel) {
    Tensor D = squared_distances(X, X2);
    Tensor K(X.rows, X2.rows);
    const double sf2 = sigma_f * sigma_f;
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X2.rows; ++j)
            K(i, j) = sf2 * kernel_from_r2(D(i, j), lengthscale, kernel);
    return K;
}

// ---------------------------------------------------------------------------
// The in-place Cholesky factorisation of a symmetric positive definite matrix.
// `A` is (n, n) and is OVERWRITTEN with its lower-triangular Cholesky
// factor; the strict upper triangle is zeroed.
//
// Returns false when the matrix is not positive definite (a non-positive
// pivot), which is the singular case the header promises to detect.
// ---------------------------------------------------------------------------
bool cholesky_in_place(Tensor& A) {
    const size_t n = A.rows;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = A(i, j);
            for (size_t k = 0; k < j; ++k) s -= A(i, k) * A(j, k);
            if (i == j) {
                if (!(s > 0.0)) return false;
                A(i, j) = std::sqrt(s);
            } else {
                A(i, j) = s / A(j, j);
            }
        }
        // Zero the strict upper triangle so the factor is clean.
        for (size_t j = i + 1; j < n; ++j) A(i, j) = 0.0;
    }
    return true;
}

// Solves L z = b for a lower-triangular L, in place on b.
// `L` is (n, n) lower triangular; `b` is (n, m).
void forward_solve(const Tensor& L, Tensor& b) {
    const size_t n = L.rows;
    const size_t m = b.cols;
    for (size_t j = 0; j < m; ++j) {
        for (size_t i = 0; i < n; ++i) {
            double s = b(i, j);
            for (size_t k = 0; k < i; ++k) s -= L(i, k) * b(k, j);
            b(i, j) = s / L(i, i);
        }
    }
}

// Solves L^T z = b for a lower-triangular L (i.e. an upper-triangular solve
// with L^T), in place on b.
void backward_solve(const Tensor& L, Tensor& b) {
    const size_t n = L.rows;
    const size_t m = b.cols;
    for (size_t j = 0; j < m; ++j) {
        for (size_t ii = n; ii-- > 0;) {
            double s = b(ii, j);
            for (size_t k = ii + 1; k < n; ++k) s -= L(k, ii) * b(k, j);
            b(ii, j) = s / L(ii, ii);
        }
    }
}

// ---------------------------------------------------------------------------
// The shared factorisation: builds Ky = Kf + sn^2 I and factorises it.
//
// Returns the Cholesky factor L. Throws when Ky is not positive definite.
// ---------------------------------------------------------------------------
Tensor factorize_ky(const Tensor& X, double lengthscale, double sigma_f,
                     double sigma_n, GPKernelType kernel) {
    Tensor Ky = kernel_matrix_impl(X, X, lengthscale, sigma_f, kernel);
    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < Ky.rows; ++i) Ky(i, i) += sn2;
    if (!cholesky_in_place(Ky)) {
        throw std::invalid_argument(
            "gp: the covariance matrix is not positive definite (the "
            "Cholesky factorisation failed)");
    }
    return Ky;
}

// The hyperparameter gradient of the log marginal likelihood (eq. 5.9).
//
// The derivative of the kernel matrix with respect to one hyperparameter.
// Only the SE kernel has an analytic derivative here; for the Matern kernels
// the derivative is computed by a central difference of the KERNEL itself,
// which is exact to O(eps^2) and is the same quantity the marginal
// likelihood differentiates.
Tensor dK_dlengthscale(const Tensor& X, const Tensor& X2, double lengthscale,
                        double sigma_f, GPKernelType kernel) {
    Tensor D = squared_distances(X, X2);
    Tensor dK(X.rows, X2.rows);
    const double sf2 = sigma_f * sigma_f;
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X2.rows; ++j) {
            const double r2 = D(i, j);
            switch (kernel) {
                case GP_KERNEL_SE: {
                    // k = sf^2 exp(-0.5 r^2 / l^2)
                    // dk/dl = k * r^2 / l^3
                    dK(i, j) = sf2 * std::exp(-0.5 * r2 / (lengthscale * lengthscale)) *
                               r2 / (lengthscale * lengthscale * lengthscale);
                    break;
                }
                case GP_KERNEL_MATERN32:
                case GP_KERNEL_MATERN52: {
                    // A central difference of the kernel itself. The
                    // perturbation is relative to the length-scale so it
                    // scales with the parameter.
                    const double eps = lengthscale * 1e-6;
                    const double lp = kernel_from_r2(r2, lengthscale + eps, kernel);
                    const double lm = kernel_from_r2(r2, lengthscale - eps, kernel);
                    dK(i, j) = sf2 * (lp - lm) / (2.0 * eps);
                    break;
                }
            }
        }
    }
    return dK;
}

Tensor dK_dsigma_f(const Tensor& X, const Tensor& X2, double lengthscale,
                   double sigma_f, GPKernelType kernel) {
    // dk/dsf = 2 sf * kernel_without_amplitude, because k = sf^2 * g.
    Tensor D = squared_distances(X, X2);
    Tensor dK(X.rows, X2.rows);
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X2.rows; ++j)
            dK(i, j) = 2.0 * sigma_f *
                       kernel_from_r2(D(i, j), lengthscale, kernel);
    return dK;
}

// ---------------------------------------------------------------------------
// Ky^-1 = L^-T L^-1, assembled by the TWO triangular solves on the identity.
//
// **THE FIRST DRAFT OF THIS FUNCTION INVERTED `L` BY HAND AND GOT IT WRONG
// TWICE**: it set the diagonal of the forward substitution to 1.0 (but the
// equation being solved is `L x = e_j`, whose diagonal entry is
// `(1 - sum_{k<j} L(j,k) x_k) / L(j,j)`, not `1`), and it then applied a
// BACK substitution on top, which solves `L^T x = e_j` and therefore
// produces `L^-T`, not `L^-1`. Both bugs cancelled in the LOO variance and
// produced a plausible-looking but wrong number.
//
// THE FIX IS TO NOT INVERT L BY HAND AT ALL: apply the two solves to the
// identity matrix, which is exactly the composition L^-T L^-1.
//
// The two solves are the SAME `forward_solve` / `backward_solve` used
// everywhere else in this file, so there is one implementation of each.
// ---------------------------------------------------------------------------
Tensor inverse_from_cholesky(const Tensor& L) {
    const size_t n = L.rows;
    Tensor Kinv(n, n);
    for (size_t i = 0; i < n; ++i) Kinv(i, i) = 1.0;
    forward_solve(L, Kinv);   // Kinv = L^-1
    backward_solve(L, Kinv);  // Kinv = L^-T L^-1 = Ky^-1
    return Kinv;
}

}  // namespace

// ===========================================================================
// The public API.
// ===========================================================================

Tensor gp_kernel_matrix(const Tensor& X, const Tensor& X2, double lengthscale,
                         double sigma_f, GPKernelType kernel) {
    validate_points(X, &X2, "training");
    validate_hyperparameters(lengthscale, sigma_f, 0.0, false);
    validate_kernel(kernel);
    return kernel_matrix_impl(X, X2, lengthscale, sigma_f, kernel);
}

double gp_log_marginal_likelihood(const Tensor& X, const Tensor& y,
                                  double lengthscale, double sigma_f,
                                  double sigma_n, GPKernelType kernel) {
    validate_points(X, nullptr, "training");
    validate_hyperparameters(lengthscale, sigma_f, sigma_n, true);
    validate_kernel(kernel);
    validate_targets(X, y);

    const size_t n = X.rows;
    Tensor L = factorize_ky(X, lengthscale, sigma_f, sigma_n, kernel);

    // alpha = L^-T L^-1 y
    Tensor alpha = y.clone();
    forward_solve(L, alpha);
    backward_solve(L, alpha);

    double quad = 0.0;
    for (size_t i = 0; i < n; ++i) quad += y(i, 0) * alpha(i, 0);

    // log|Ky| = 2 * sum(log L_ii), so -1/2 log|Ky| = -sum(log L_ii).
    double logdet_half = 0.0;
    for (size_t i = 0; i < n; ++i) logdet_half += std::log(L(i, i));

    return -0.5 * quad - logdet_half -
           0.5 * static_cast<double>(n) * std::log(2.0 * M_PI);
}

GPPosterior gp_posterior(const Tensor& X, const Tensor& y, double lengthscale,
                          double sigma_f, double sigma_n, GPKernelType kernel) {
    return gp_predict(X, y, X, lengthscale, sigma_f, sigma_n, kernel);
}

GPPosterior gp_predict(const Tensor& X, const Tensor& y, const Tensor& Xs,
                        double lengthscale, double sigma_f, double sigma_n,
                        GPKernelType kernel) {
    validate_points(X, nullptr, "training");
    validate_points(X, &Xs, "training");
    validate_hyperparameters(lengthscale, sigma_f, sigma_n, true);
    validate_kernel(kernel);
    validate_targets(X, y);

    const size_t n = X.rows;
    const size_t m = Xs.rows;

    Tensor L = factorize_ky(X, lengthscale, sigma_f, sigma_n, kernel);

    // alpha = L^-T L^-1 y
    Tensor alpha = y.clone();
    forward_solve(L, alpha);
    backward_solve(L, alpha);

    // Ks = k(X, Xs) -- the cross-covariance, (n, m).
    Tensor Ks = kernel_matrix_impl(X, Xs, lengthscale, sigma_f, kernel);
    // Kss = the diagonal of k(Xs, Xs), (m,).
    Tensor Dss = squared_distances(Xs, Xs);
    Tensor Kss(m, 1);
    for (size_t j = 0; j < m; ++j)
        Kss(j, 0) = sigma_f * sigma_f *
                    kernel_from_r2(Dss(j, j), lengthscale, kernel);

    // v = L^-1 Ks, (n, m).
    Tensor V = Ks.clone();
    forward_solve(L, V);

    GPPosterior out;
    out.mean = Tensor(m, 1);
    out.variance = Tensor(m, 1);
    for (size_t j = 0; j < m; ++j) {
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) mean += Ks(i, j) * alpha(i, 0);
        out.mean(j, 0) = mean;

        double vv = 0.0;
        for (size_t i = 0; i < n; ++i) vv += V(i, j) * V(i, j);
        out.variance(j, 0) = Kss(j, 0) - vv;
    }
    return out;
}

GPPosterior gp_predict_noisy(const Tensor& X, const Tensor& y, const Tensor& Xs,
                              double lengthscale, double sigma_f,
                              double sigma_n, GPKernelType kernel) {
    GPPosterior p = gp_predict(X, y, Xs, lengthscale, sigma_f, sigma_n, kernel);
    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < p.variance.rows; ++i) p.variance(i, 0) += sn2;
    return p;
}

GPHypergradient gp_hyperparameter_gradient(const Tensor& X, const Tensor& y,
                                           double lengthscale, double sigma_f,
                                           double sigma_n, GPKernelType kernel) {
    validate_points(X, nullptr, "training");
    validate_hyperparameters(lengthscale, sigma_f, sigma_n, true);
    validate_kernel(kernel);
    validate_targets(X, y);

    const size_t n = X.rows;

    // Ky = L L^T and alpha = Ky^-1 y.
    Tensor Ky = kernel_matrix_impl(X, X, lengthscale, sigma_f, kernel);
    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < n; ++i) Ky(i, i) += sn2;
    Tensor L = Ky.clone();
    if (!cholesky_in_place(L)) {
        throw std::invalid_argument(
            "gp: the covariance matrix is not positive definite (the "
            "Cholesky factorisation failed)");
    }
    Tensor alpha = y.clone();
    forward_solve(L, alpha);
    backward_solve(L, alpha);

    // Ky^-1, assembled from the triangular solves: Ky^-1 = L^-T L^-1.
    Tensor Kinv = inverse_from_cholesky(L);

    // M = alpha alpha^T - Ky^-1.
    Tensor M(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = alpha(i, 0) * alpha(j, 0) - Kinv(i, j);

    // For each hyperparameter: 0.5 * tr(M dK).
    //
    // THE TRACE IS COMPUTED AS sum_ij M_ij dK_ji -- with the TRANSPOSE. For
    // symmetric M and dK that equals the elementwise sum, so an implementation
    // using `sum(M .* dK)` would silently rely on symmetry; the transposed
    // form is correct for ANY dK.
    auto trace_product = [&](const Tensor& dK) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) s += M(i, j) * dK(j, i);
        return s;
    };

    GPHypergradient g;

    // d/d(lengthscale)
    Tensor dK_l = dK_dlengthscale(X, X, lengthscale, sigma_f, kernel);
    g.d_lengthscale = 0.5 * trace_product(dK_l);

    // d/d(sigma_f): dk/dsf = 2 sf * kernel_without_amplitude.
    Tensor dK_sf = dK_dsigma_f(X, X, lengthscale, sigma_f, kernel);
    g.d_sigma_f = 0.5 * trace_product(dK_sf);

    // d/d(sigma_n): dKy/dsn = 2 sn I.
    double t = 0.0;
    for (size_t i = 0; i < n; ++i) t += M(i, i) * (2.0 * sigma_n);
    g.d_sigma_n = 0.5 * t;

    return g;
}

GPLoo gp_leave_one_out(const Tensor& X, const Tensor& y, double lengthscale,
                        double sigma_f, double sigma_n, GPKernelType kernel) {
    validate_points(X, nullptr, "training");
    validate_hyperparameters(lengthscale, sigma_f, sigma_n, true);
    validate_kernel(kernel);
    validate_targets(X, y);

    const size_t n = X.rows;

    // Ky = L L^T.
    Tensor Ky = kernel_matrix_impl(X, X, lengthscale, sigma_f, kernel);
    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < n; ++i) Ky(i, i) += sn2;
    Tensor L = Ky.clone();
    if (!cholesky_in_place(L)) {
        throw std::invalid_argument(
            "gp: the covariance matrix is not positive definite (the "
            "Cholesky factorisation failed)");
    }

    // Ky^-1, from the triangular solves.
    Tensor Kinv = inverse_from_cholesky(L);

    // alpha = Ky^-1 y.
    Tensor alpha(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) s += Kinv(i, j) * y(j, 0);
        alpha(i, 0) = s;
    }

    // eq. (5.12):
    //     mu_i = y_i - [Ky^-1 y]_i / [Ky^-1]_ii
    //     s2_i = 1 / [Ky^-1]_ii
    //
    // NOTE the matrix is Ky, the NOISY one -- see the header's rule (7).
    GPLoo out;
    out.mean = Tensor(n, 1);
    out.variance = Tensor(n, 1);
    for (size_t i = 0; i < n; ++i) {
        out.mean(i, 0) = y(i, 0) - alpha(i, 0) / Kinv(i, i);
        out.variance(i, 0) = 1.0 / Kinv(i, i);
    }
    return out;
}

}  // namespace utils
}  // namespace nn
