// Implementation of MultivariateGaussian, KMeans, and GMM classes.

#include "nn/utils/gmm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// =============================================================================
// Small helpers (private to this TU)
// =============================================================================

namespace {

inline double log_sum_exp(const std::vector<double>& xs) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : xs) if (x > m) m = x;
    if (!std::isfinite(m)) return m;  // all -inf or +inf
    double s = 0.0;
    for (double x : xs) s += std::exp(x - m);
    return m + std::log(s);
}

// Check that a tensor is non-empty and finite.
void check_finite(const Tensor& t, const char* name) {
    if (t.rows == 0 || t.cols == 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            if (!std::isfinite(t(i, j))) {
                throw std::invalid_argument(
                    std::string(name) + " contains non-finite values");
            }
        }
    }
}

}  // namespace

// =============================================================================
// MultivariateGaussian
// =============================================================================

MultivariateGaussian::MultivariateGaussian(const Tensor& mean, const Tensor& cov)
    : mean_(mean), cov_(cov) {
    if (mean.rows != 1) {
        throw std::invalid_argument("MultivariateGaussian: mean must have rows == 1");
    }
    if (cov.rows != cov.cols) {
        throw std::invalid_argument("MultivariateGaussian: cov must be square");
    }
    if (cov.rows != static_cast<size_t>(mean.cols)) {
        throw std::invalid_argument(
            "MultivariateGaussian: cov dimension must match mean.cols");
    }
    check_finite(mean, "MultivariateGaussian::mean");
    check_finite(cov,  "MultivariateGaussian::cov");
    // Symmetry sanity (not strictly required, but catch obvious mistakes).
    for (size_t i = 0; i < cov.rows; ++i) {
        for (size_t j = i + 1; j < cov.cols; ++j) {
            if (std::fabs(cov(i, j) - cov(j, i)) > 1e-8) {
                throw std::invalid_argument(
                    "MultivariateGaussian: cov must be symmetric (off-diag tol 1e-8)");
            }
        }
    }
    recompute_decompositions();
}

void MultivariateGaussian::solve_lower(const Tensor& L,
                                       const std::vector<double>& b,
                                       std::vector<double>& y) {
    size_t d = L.rows;
    y.assign(d, 0.0);
    for (size_t i = 0; i < d; ++i) {
        double s = b[i];
        for (size_t k = 0; k < i; ++k) s -= L(i, k) * y[k];
        if (L(i, i) <= 0.0) {
            throw std::runtime_error("MultivariateGaussian: non-positive L diagonal");
        }
        y[i] = s / L(i, i);
    }
}

void MultivariateGaussian::solve_upper(const Tensor& L,
                                       const std::vector<double>& b,
                                       std::vector<double>& y) {
    size_t d = L.rows;
    y.assign(d, 0.0);
    for (size_t ii = 0; ii < d; ++ii) {
        size_t i = d - 1 - ii;  // walk backwards
        double s = b[i];
        for (size_t k = i + 1; k < d; ++k) s -= L(k, i) * y[k];  // L^T[i,k] = L[k,i]
        if (L(i, i) <= 0.0) {
            throw std::runtime_error("MultivariateGaussian: non-positive L diagonal");
        }
        y[i] = s / L(i, i);
    }
}

void MultivariateGaussian::recompute_decompositions() {
    const size_t d = cov_.rows;
    // Cholesky: cov_ = L L^T, L lower triangular.
    L_ = Tensor(d, d);  // zero-init
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = cov_(i, j);
            for (size_t k = 0; k < j; ++k) s -= L_(i, k) * L_(j, k);
            if (i == j) {
                if (s <= 0.0) {
                    throw std::runtime_error(
                        "MultivariateGaussian: cov is not positive-definite");
                }
                L_(i, j) = std::sqrt(s);
            } else {
                L_(i, j) = s / L_(j, j);
            }
        }
    }
    log_det_ = 0.0;
    for (size_t i = 0; i < d; ++i) log_det_ += 2.0 * std::log(L_(i, i));

    // cov_inv_ = (L L^T)^{-1} = L^{-T} L^{-1}.
    // Compute L^{-1} by solving L x = e_k for each k, store columns of inv_L.
    inv_L_ = Tensor(d, d);  // this will be L^{-1}
    std::vector<double> e_k(d, 0.0), col(d);
    for (size_t k = 0; k < d; ++k) {
        e_k[k] = 1.0;
        solve_lower(L_, e_k, col);
        for (size_t i = 0; i < d; ++i) inv_L_(i, k) = col[i];
        e_k[k] = 0.0;
    }
    // cov_inv_ = inv_L_^T * inv_L_.
    cov_inv_ = Tensor(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += inv_L_(k, i) * inv_L_(k, j);
            cov_inv_(i, j) = s;
        }
    }
}

double MultivariateGaussian::mahalanobis_sq_impl(const Tensor& x) const {
    const size_t d = mean_.cols;
    std::vector<double> diff(d);
    for (size_t j = 0; j < d; ++j) diff[j] = x(0, j) - mean_(0, j);
    // Solve L y = diff, then y^T y = (x-μ)^T L^{-T} L^{-1} (x-μ) = (x-μ)^T Σ^{-1} (x-μ).
    std::vector<double> y(d);
    solve_lower(L_, diff, y);
    double q = 0.0;
    for (double v : y) q += v * v;
    return q;
}

double MultivariateGaussian::mahalanobis_sq(const Tensor& x) const {
    if (x.rows != 1 || x.cols != mean_.cols) {
        throw std::invalid_argument(
            "MultivariateGaussian::mahalanobis_sq: bad shape");
    }
    return mahalanobis_sq_impl(x);
}

Tensor MultivariateGaussian::log_pdf(const Tensor& x) const {
    if (x.cols != mean_.cols) {
        throw std::invalid_argument("MultivariateGaussian::log_pdf: bad shape");
    }
    const size_t d = mean_.cols;
    const double norm_const = -0.5 * static_cast<double>(d) * std::log(2.0 * M_PI);
    Tensor out(x.rows, 1);
    for (size_t i = 0; i < x.rows; ++i) {
        Tensor row = x.get_row(i);
        out(i, 0) = norm_const - 0.5 * log_det_ - 0.5 * mahalanobis_sq_impl(row);
    }
    return out;
}

Tensor MultivariateGaussian::log_pdf_grad(const Tensor& x) const {
    if (x.rows != 1 || x.cols != mean_.cols) {
        throw std::invalid_argument("MultivariateGaussian::log_pdf_grad: bad shape");
    }
    const size_t d = mean_.cols;
    // grad = -Σ^{-1} (x - μ) = -L^{-T} L^{-1} (x - μ)
    std::vector<double> diff(d);
    for (size_t j = 0; j < d; ++j) diff[j] = x(0, j) - mean_(0, j);
    // y1 = L^{-1} (x - μ)  via forward solve
    std::vector<double> y1(d);
    solve_lower(L_, diff, y1);
    // y2 = L^{-T} y1      via back solve on L
    std::vector<double> y2(d);
    solve_upper(L_, y1, y2);
    Tensor grad(1, d);
    for (size_t j = 0; j < d; ++j) grad(0, j) = -y2[j];
    return grad;
}

double MultivariateGaussian::kl_to(const MultivariateGaussian& q) const {
    // KL(N_p || N_q) where this == p, q == q
    //   = 0.5 * [ tr(Σ_q^{-1} Σ_p) + (μ_p - μ_q)^T Σ_q^{-1} (μ_p - μ_q)
    //            - d + log|Σ_q| - log|Σ_p| ]
    const size_t d = mean_.cols;
    if (q.dim() != d) {
        throw std::invalid_argument("MultivariateGaussian::kl_to: dim mismatch");
    }
    // Term 1: tr(Σ_q^{-1} Σ_p). Both Σ are symmetric, so
    //   tr(AB) = sum_{i,j} A[i,j] B[j,i] = sum_{i,j} A[i,j] B[i,j].
    // We compute Σ_q^{-1} = cov_inv_ of q, and sum the elementwise product.
    double trace_term = 0.0;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            trace_term += q.cov_inv_(i, j) * cov_(i, j);
        }
    }

    // Term 2: (μ_p - μ_q)^T Σ_q^{-1} (μ_p - μ_q). Use q's public mahalanobis_sq
    // evaluated at the point μ_p (which uses q's own μ_q internally).
    double mahalanobis_sq_val = q.mahalanobis_sq(mean_);

    double log_det_term = q.log_det_ - log_det_;

    return 0.5 * (trace_term + mahalanobis_sq_val
                  - static_cast<double>(d) + log_det_term);
}

// =============================================================================
// KMeans
// =============================================================================

KMeans::KMeans(size_t n_clusters, size_t max_iter, double tol,
               unsigned int seed, Init init)
    : n_clusters_(n_clusters), max_iter_(max_iter),
      tol_(tol), seed_(seed), init_(init) {
    if (n_clusters_ == 0) {
        throw std::invalid_argument("KMeans: n_clusters must be > 0");
    }
    if (max_iter_ == 0) {
        throw std::invalid_argument("KMeans: max_iter must be > 0");
    }
}

void KMeans::init_centers(const Tensor& X, std::mt19937& rng) {
    const size_t N = X.rows;
    const size_t d = X.cols;
    if (N < n_clusters_) {
        throw std::invalid_argument(
            "KMeans: n_samples < n_clusters");
    }
    centers_ = Tensor(n_clusters_, d);
    std::vector<size_t> chosen;

    if (init_ == Init::RANDOM) {
        // Reservoir-style: pick K distinct indices uniformly.
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (size_t k = 0; k < n_clusters_; ++k) {
            size_t row = idx[k];
            for (size_t j = 0; j < d; ++j) centers_(k, j) = X(row, j);
            chosen.push_back(row);
        }
    } else {
        // K-means++: first center uniform random, rest ∝ D(x)^2.
        std::uniform_int_distribution<size_t> unif(0, N - 1);
        size_t first = unif(rng);
        for (size_t j = 0; j < d; ++j) centers_(0, j) = X(first, j);
        chosen.push_back(first);

        std::vector<double> min_d2(N, std::numeric_limits<double>::infinity());
        for (size_t k = 1; k < n_clusters_; ++k) {
            // Update D(x)^2 = min_c ||x - c||^2 with the latest center.
            for (size_t i = 0; i < N; ++i) {
                double dist = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    double diff = X(i, j) - centers_(k - 1, j);
                    dist += diff * diff;
                }
                if (dist < min_d2[i]) min_d2[i] = dist;
            }
            // Pick next center with probability ∝ min_d2.
            std::discrete_distribution<size_t> dd(min_d2.begin(), min_d2.end());
            size_t next = dd(rng);
            for (size_t j = 0; j < d; ++j) centers_(k, j) = X(next, j);
            chosen.push_back(next);
        }
    }
}

double KMeans::lloyd_step(const Tensor& X,
                          [[maybe_unused]] std::vector<size_t>& labels,
                          std::vector<size_t>& new_labels) {
    const size_t N = X.rows;
    const size_t d = X.cols;
    new_labels.assign(N, 0);

    // Assign step
    double inertia = 0.0;
    for (size_t i = 0; i < N; ++i) {
        size_t best = 0;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_clusters_; ++k) {
            double dist = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = X(i, j) - centers_(k, j);
                dist += diff * diff;
            }
            if (dist < best_d2) {
                best_d2 = dist;
                best = k;
            }
        }
        new_labels[i] = best;
        inertia += best_d2;
    }

    // Update step: recompute each center as the mean of its assigned points.
    std::vector<Tensor> sums(n_clusters_, Tensor(1, d));
    std::vector<size_t> counts(n_clusters_, 0);
    for (size_t i = 0; i < N; ++i) {
        size_t k = new_labels[i];
        counts[k]++;
        for (size_t j = 0; j < d; ++j) sums[k](0, j) += X(i, j);
    }
    // For empty clusters, keep the previous center (standard sklearn convention).
    for (size_t k = 0; k < n_clusters_; ++k) {
        if (counts[k] == 0) continue;
        for (size_t j = 0; j < d; ++j) centers_(k, j) = sums[k](0, j) / counts[k];
    }
    return inertia;
}

double KMeans::fit(const Tensor& X) {
    return fit_with_history(X).back();
}

std::vector<double> KMeans::fit_with_history(const Tensor& X) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument("KMeans::fit: empty input");
    }
    if (X.rows < n_clusters_) {
        throw std::invalid_argument(
            "KMeans::fit: n_samples < n_clusters");
    }
    std::mt19937 rng(seed_);
    init_centers(X, rng);

    std::vector<size_t> labels(X.rows, 0), new_labels(X.rows, 0);
    std::vector<double> history;
    n_iter_ = 0;
    for (size_t it = 0; it < max_iter_; ++it) {
        n_iter_ = it + 1;
        double inertia = lloyd_step(X, labels, new_labels);
        history.push_back(inertia);
        // Convergence: assignments did not change.
        bool stable = (new_labels == labels);
        labels = new_labels;
        last_inertia_ = inertia;
        if (stable && it > 0) break;
    }
    return history;
}

Tensor KMeans::predict(const Tensor& X) const {
    if (centers_.rows == 0) {
        throw std::runtime_error("KMeans::predict: not fitted");
    }
    Tensor out(X.rows, 1);
    for (size_t i = 0; i < X.rows; ++i) {
        size_t best = 0;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_clusters_; ++k) {
            double dist = 0.0;
            for (size_t j = 0; j < X.cols; ++j) {
                double diff = X(i, j) - centers_(k, j);
                dist += diff * diff;
            }
            if (dist < best_d2) { best_d2 = dist; best = k; }
        }
        out(i, 0) = static_cast<double>(best);
    }
    return out;
}

// =============================================================================
// GMM
// =============================================================================

GMM::GMM(size_t n_components, size_t max_iter, double tol,
         double reg_covar, unsigned int seed)
    : n_components_(n_components), max_iter_(max_iter),
      tol_(tol), reg_covar_(reg_covar), seed_(seed) {
    if (n_components_ == 0) {
        throw std::invalid_argument("GMM: n_components must be > 0");
    }
    if (reg_covar_ < 0.0) {
        throw std::invalid_argument("GMM: reg_covar must be >= 0");
    }
    weights_ = Tensor(n_components_, 1);
    for (size_t k = 0; k < n_components_; ++k) weights_(k, 0) = 1.0 / n_components_;
}

void GMM::init_from_kmeans(const Tensor& X, std::mt19937& rng) {
    const size_t N = X.rows;
    const size_t d = X.cols;
    // K-means++ via the same recipe as KMeans::init_centers.
    std::vector<Tensor> ctrs;
    std::uniform_int_distribution<size_t> unif(0, N - 1);
    size_t first = unif(rng);
    Tensor first_center = X.get_row(first);
    ctrs.push_back(first_center);
    std::vector<double> min_d2(N, std::numeric_limits<double>::infinity());
    for (size_t k = 1; k < n_components_; ++k) {
        for (size_t i = 0; i < N; ++i) {
            double dist = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = X(i, j) - ctrs.back()(0, j);
                dist += diff * diff;
            }
            if (dist < min_d2[i]) min_d2[i] = dist;
        }
        std::discrete_distribution<size_t> dd(min_d2.begin(), min_d2.end());
        size_t next = dd(rng);
        ctrs.push_back(X.get_row(next));
    }
    // Assign each point to its nearest center, then compute cluster mean
    // and covariance.  This gives us better initial covariances than
    // using the raw seeds (which would give degenerate single-point
    // covariances).
    means_.clear();
    covs_.clear();
    means_.reserve(n_components_);
    covs_.reserve(n_components_);
    std::vector<size_t> counts(n_components_, 0);
    std::vector<Tensor> sums(n_components_, Tensor(1, d));
    std::vector<std::vector<size_t>> membership(n_components_);
    for (size_t i = 0; i < N; ++i) {
        size_t best = 0;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_components_; ++k) {
            double dist = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = X(i, j) - ctrs[k](0, j);
                dist += diff * diff;
            }
            if (dist < best_d2) { best_d2 = dist; best = k; }
        }
        membership[best].push_back(i);
        counts[best]++;
        for (size_t j = 0; j < d; ++j) sums[best](0, j) += X(i, j);
    }
    for (size_t k = 0; k < n_components_; ++k) {
        if (counts[k] == 0) {
            // Empty initial cluster — fall back to the raw seed.
            means_.push_back(ctrs[k]);
        } else {
            Tensor m(1, d);
            for (size_t j = 0; j < d; ++j) m(0, j) = sums[k](0, j) / counts[k];
            means_.push_back(m);
        }
        // Covariance
        Tensor cov(d, d);
        if (counts[k] <= 1) {
            // Single-point cluster: init to identity scaled by global var.
            for (size_t i = 0; i < d; ++i) {
                double v = 0.0;
                for (size_t r = 0; r < N; ++r) {
                    double dv = X(r, i) - means_[k](0, i);
                    v += dv * dv;
                }
                v = (v > 1e-12) ? (v / N) : 1.0;
                cov(i, i) = v;
            }
        } else {
            for (size_t r : membership[k]) {
                for (size_t a = 0; a < d; ++a) {
                    for (size_t b = 0; b < d; ++b) {
                        double da = X(r, a) - means_[k](0, a);
                        double db = X(r, b) - means_[k](0, b);
                        cov(a, b) += da * db / counts[k];
                    }
                }
            }
        }
        // Regularize so the first E-step doesn't blow up.
        for (size_t i = 0; i < d; ++i) cov(i, i) += reg_covar_ + 1e-6;
        covs_.push_back(cov);
    }
    weights_ = Tensor(n_components_, 1);
    for (size_t k = 0; k < n_components_; ++k) {
        weights_(k, 0) = static_cast<double>(counts[k]) / N;
        if (weights_(k, 0) < 1e-3) weights_(k, 0) = 1e-3;  // avoid dead comps
    }
    double wsum = 0.0;
    for (size_t k = 0; k < n_components_; ++k) wsum += weights_(k, 0);
    for (size_t k = 0; k < n_components_; ++k) weights_(k, 0) /= wsum;
}

void GMM::rebuild_components() {
    components_.clear();
    components_.reserve(n_components_);
    for (size_t k = 0; k < n_components_; ++k) {
        components_.push_back(MultivariateGaussian(means_[k], covs_[k]));
    }
}

Tensor GMM::e_step_log_resp(const Tensor& X) const {
    // log_resp[i, k] = log(pi_k) + log N(x_i | μ_k, Σ_k)
    const size_t N = X.rows;
    Tensor log_resp(N, n_components_);
    for (size_t i = 0; i < N; ++i) {
        Tensor row = X.get_row(i);
        std::vector<double> log_pi_k(n_components_);
        std::vector<double> log_pk(n_components_);
        for (size_t k = 0; k < n_components_; ++k) {
            log_pi_k[k] = std::log(std::max(weights_(k, 0), 1e-300));
            Tensor row_pdf = components_[k].log_pdf(row);
            log_pk[k]   = row_pdf(0, 0);
        }
        // log_resp_unnorm[i,k] = log_pi_k + log_pk
        // softmax across k = subtract row-max for stability.
        std::vector<double> unnorm(n_components_);
        for (size_t k = 0; k < n_components_; ++k) unnorm[k] = log_pi_k[k] + log_pk[k];
        double row_max = *std::max_element(unnorm.begin(), unnorm.end());
        double denom = 0.0;
        for (size_t k = 0; k < n_components_; ++k) denom += std::exp(unnorm[k] - row_max);
        double log_norm = row_max + std::log(denom);
        for (size_t k = 0; k < n_components_; ++k) {
            log_resp(i, k) = unnorm[k] - log_norm;
        }
    }
    return log_resp;
}

void GMM::m_step(const Tensor& X, const Tensor& log_resp) {
    const size_t N = X.rows;
    const size_t d = X.cols;

    // Soft responsibilities γ_{ik} = exp(log_resp[i,k]).
    // For numerical robustness in the M-step, also compute log_N_k = log(sum_i γ_{ik})
    // and use that for normalization.
    std::vector<double> log_N_k(n_components_, -std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < n_components_; ++k) {
            double v = log_resp(i, k);
            if (v > log_N_k[k]) log_N_k[k] = v;
        }
    }
    Tensor resp(N, n_components_);
    std::vector<double> N_k(n_components_, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < n_components_; ++k) {
            resp(i, k) = std::exp(log_resp(i, k) - log_N_k[k]);
            N_k[k] += resp(i, k);
        }
    }

    // Update weights.
    double total = 0.0;
    for (size_t k = 0; k < n_components_; ++k) total += N_k[k];
    for (size_t k = 0; k < n_components_; ++k) {
        weights_(k, 0) = std::max(N_k[k] / total, 1e-15);
    }

    // Update means and covariances.
    means_.clear();
    covs_.clear();
    means_.reserve(n_components_);
    covs_.reserve(n_components_);
    for (size_t k = 0; k < n_components_; ++k) {
        if (N_k[k] < 1e-12) {
            // Degenerate: keep the old parameters (rare). Could also re-init.
            means_.push_back(components_[k].mean());
            Tensor cov(d, d);
            for (size_t i = 0; i < d; ++i) cov(i, i) = 1.0;
            covs_.push_back(cov);
            continue;
        }
        Tensor new_mean(1, d);
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < N; ++i) s += resp(i, k) * X(i, j);
            new_mean(0, j) = s / N_k[k];
        }
        means_.push_back(new_mean);

        Tensor new_cov(d, d);
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> diff(d);
            for (size_t j = 0; j < d; ++j) diff[j] = X(i, j) - new_mean(0, j);
            for (size_t a = 0; a < d; ++a) {
                for (size_t b = 0; b < d; ++b) {
                    new_cov(a, b) += resp(i, k) * diff[a] * diff[b] / N_k[k];
                }
            }
        }
        // Regularize for numerical stability.
        for (size_t i = 0; i < d; ++i) new_cov(i, i) += reg_covar_;
        covs_.push_back(new_cov);
    }
    rebuild_components();
}

double GMM::log_likelihood(const Tensor& X) const {
    if (!fitted_) {
        // Allow internal use during fit() — only enforce for external callers.
        // We use a heuristic: if components_ is empty, definitely not fitted.
        if (components_.empty()) {
            throw std::runtime_error("GMM::log_likelihood: not fitted");
        }
    }
    const size_t N = X.rows;
    double total = 0.0;
    for (size_t i = 0; i < N; ++i) {
        Tensor row = X.get_row(i);
        std::vector<double> log_pk(n_components_);
        for (size_t k = 0; k < n_components_; ++k) {
            Tensor row_pdf = components_[k].log_pdf(row);
            log_pk[k] = std::log(std::max(weights_(k, 0), 1e-300))
                      + row_pdf(0, 0);
        }
        total += log_sum_exp(log_pk);
    }
    return total;
}

double GMM::fit(const Tensor& X) {
    std::vector<double> history;
    return fit_with_history(X, history);
}

double GMM::fit_with_history(const Tensor& X, std::vector<double>& history) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument("GMM::fit: empty input");
    }
    if (X.rows < n_components_) {
        throw std::invalid_argument("GMM::fit: n_samples < n_components");
    }
    input_dim_ = X.cols;
    std::mt19937 rng(seed_);
    init_from_kmeans(X, rng);
    rebuild_components();

    history.clear();
    double prev_ll = -std::numeric_limits<double>::infinity();
    n_iter_ = 0;
    for (size_t it = 0; it < max_iter_; ++it) {
        n_iter_ = it + 1;
        Tensor log_resp = e_step_log_resp(X);
        m_step(X, log_resp);
        double ll = log_likelihood(X);
        history.push_back(ll);
        if (std::isfinite(prev_ll) && std::fabs(ll - prev_ll) < tol_) break;
        prev_ll = ll;
    }
    fitted_ = true;
    return history.empty() ? 0.0 : history.back();
}

Tensor GMM::predict_proba(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("GMM::predict_proba: not fitted");
    }
    if (X.cols != input_dim_) {
        throw std::invalid_argument("GMM::predict_proba: dim mismatch");
    }
    Tensor log_resp = e_step_log_resp(X);
    Tensor proba(X.rows, n_components_);
    for (size_t i = 0; i < X.rows; ++i) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_components_; ++k) {
            if (log_resp(i, k) > row_max) row_max = log_resp(i, k);
        }
        double denom = 0.0;
        for (size_t k = 0; k < n_components_; ++k) {
            proba(i, k) = std::exp(log_resp(i, k) - row_max);
            denom += proba(i, k);
        }
        for (size_t k = 0; k < n_components_; ++k) proba(i, k) /= denom;
    }
    return proba;
}

Tensor GMM::predict(const Tensor& X) const {
    Tensor proba = predict_proba(X);
    Tensor out(X.rows, 1);
    for (size_t i = 0; i < X.rows; ++i) {
        size_t best = 0;
        double best_v = proba(i, 0);
        for (size_t k = 1; k < n_components_; ++k) {
            if (proba(i, k) > best_v) { best_v = proba(i, k); best = k; }
        }
        out(i, 0) = static_cast<double>(best);
    }
    return out;
}

Tensor GMM::sample(size_t n, unsigned int sample_seed) const {
    if (!fitted_) throw std::runtime_error("GMM::sample: not fitted");
    std::mt19937 rng(sample_seed);
    std::vector<double> weight_cdf(n_components_, 0.0);
    double wsum = 0.0;
    for (size_t k = 0; k < n_components_; ++k) wsum += weights_(k, 0);
    double acc = 0.0;
    for (size_t k = 0; k < n_components_; ++k) {
        acc += weights_(k, 0) / wsum;
        weight_cdf[k] = acc;
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    Tensor out(n, input_dim_);
    for (size_t i = 0; i < n; ++i) {
        double u = uniform(rng);
        size_t k = 0;
        while (k + 1 < n_components_ && u > weight_cdf[k]) ++k;
        // Sample N(0, I) and transform via Cholesky of cov_k and shift by mean_k.
        // x = μ + L^T z  where cov = L L^T and z ~ N(0, I).
        const Tensor& Lc = components_[k].cholesky_L();
        const size_t d = input_dim_;
        std::vector<double> z(d);
        for (size_t jj = 0; jj < d; ++jj) z[jj] = normal(rng);
        for (size_t jj = 0; jj < d; ++jj) {
            double s = 0.0;
            for (size_t kk = jj; kk < d; ++kk) s += Lc(kk, jj) * z[kk];
            out(i, jj) = means_[k](0, jj) + s;
        }
    }
    return out;
}

double GMM::score(const Tensor& X) const {
    return log_likelihood(X) / static_cast<double>(X.rows);
}

double GMM::bic(const Tensor& X) const {
    // p = K - 1 (weights, with constraint sum=1) + K * d (means) + K * d*(d+1)/2 (cov, symmetric)
    const size_t d = input_dim_;
    size_t p = (n_components_ - 1)
             + n_components_ * d
             + n_components_ * d * (d + 1) / 2;
    double ll = log_likelihood(X);
    double n = static_cast<double>(X.rows);
    return -2.0 * ll + static_cast<double>(p) * std::log(n);
}

double GMM::aic(const Tensor& X) const {
    const size_t d = input_dim_;
    size_t p = (n_components_ - 1)
             + n_components_ * d
             + n_components_ * d * (d + 1) / 2;
    double ll = log_likelihood(X);
    return -2.0 * ll + 2.0 * static_cast<double>(p);
}