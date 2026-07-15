#ifndef GMM_H
#define GMM_H

#include "../core/tensor.h"
#include <random>
#include <vector>

// =============================================================================
// Multivariate Gaussian — single full-covariance multivariate normal.
// =============================================================================
//
// Stores mean (1, d) and covariance (d, d). Used both as a standalone
// distribution (for log-pdf and KL) and as a building block inside GMM.
//
// log_pdf(X) returns a column vector (N, 1) of log-densities under the Gaussian.
// Internally uses a Cholesky factorization of the covariance so log|Σ| and
// (x-μ)^T Σ^{-1} (x-μ) are computed in O(d^2) with guaranteed-positive log|Σ|
// (no det sign issues). log_pdf_grad returns the gradient of the log-density
// with respect to the input x: ∂log N(x|μ,Σ)/∂x = -Σ^{-1}(x-μ).
//
// KL(p || q) uses the standard multivariate formula
//   KL(N_p || N_q) = 0.5 * (tr(Σ_q^{-1} Σ_p)
//                          + (μ_p - μ_q)^T Σ_q^{-1} (μ_p - μ_q)
//                          - d + log|Σ_q| - log|Σ_p|)
// For p == q this is exactly 0; the implementation handles degenerate cases
// (singular cov) by clamping to a small epsilon diagonal.
//
// All inputs are validated: mu.cols must match cov.rows == cov.cols.

class MultivariateGaussian {
public:
    // Construct from mean tensor (1, d) and covariance (d, d). Both must be
    // finite; covariance must be square and symmetric. Throws
    // std::invalid_argument on shape mismatch.
    MultivariateGaussian(const Tensor& mean, const Tensor& cov);

    // log N(X | μ, Σ) for each row of X (N, d). Returns (N, 1) tensor.
    // Also accepts a single-row (1, d) tensor and returns a scalar in (0, 0).
    Tensor log_pdf(const Tensor& X) const;

    // log N(x | μ, Σ) for a single point x (1, d). Returns scalar.
    // Convenience wrapper over log_pdf(X)(0, 0).
    double log_pdf_single(const Tensor& x) const { return log_pdf(x)(0, 0); }

    // ∂log N(x | μ, Σ) / ∂x for a single point x (1, d). Returns (1, d) tensor.
    Tensor log_pdf_grad(const Tensor& x) const;

    // KL(this || q) — KL divergence from this Gaussian to q.
    double kl_to(const MultivariateGaussian& q) const;

    // Accessors
    const Tensor& mean() const { return mean_; }
    const Tensor& cov()  const { return cov_; }
    const Tensor& cholesky_L() const { return L_; }
    double log_det() const { return log_det_; }
    // (x - μ)^T Σ^{-1} (x - μ) for a single point x (1, d).
    double mahalanobis_sq(const Tensor& x) const;
    size_t dim()   const { return mean_.cols; }

private:
    Tensor mean_;           // (1, d)
    Tensor cov_;            // (d, d)
    Tensor cov_inv_;        // (d, d), pre-inverted (cached)
    Tensor L_;              // (d, d) lower-triangular Cholesky factor
    double log_det_;        // log|Σ| cached from L
    Tensor inv_L_;          // (d, d), L^{-1} cached for KL

    // Recompute cov_inv_, L_, log_det_, inv_L_ from cov_.
    void recompute_decompositions();

    // Helper: solve L y = b in place (forward sub), L lower triangular.
    static void solve_lower(const Tensor& L, const std::vector<double>& b,
                            std::vector<double>& y);
    // Helper: solve L^T y = b in place (back sub).
    static void solve_upper(const Tensor& L, const std::vector<double>& b,
                            std::vector<double>& y);
    // Helper: compute Mahalanobis distance squared (x - μ)^T Σ^{-1} (x - μ) for
    // a single point x (1, d) using the cached Cholesky factor.
    double mahalanobis_sq_impl(const Tensor& x) const;
};

// =============================================================================
// KMeans — classical Lloyd's algorithm with k-means++ initialization.
// =============================================================================
//
// Two initialization strategies: RANDOM (pick K distinct points uniformly) and
// KMEANS_PLUS_PLUS (Arthur & Vassilvitskii 2007 — probability ∝ D(x)^2 where
// D(x) is the distance to the nearest already-chosen seed). Default is
// KMEANS_PLUS_PLUS because it dramatically improves convergence on real data.
//
// Lloyd's loop: assign each point to its nearest center, then update each
// center to the mean of its assigned points. Converges when the assignment
// does not change or when max_iter is reached.
//
// `fit_with_history(X)` returns the per-iteration inertia
// (sum of squared distances to nearest center). Useful for testing the
// monotonically-decreasing invariant.

class KMeans {
public:
    enum class Init { RANDOM, KMEANS_PLUS_PLUS };

    KMeans(size_t n_clusters, size_t max_iter = 300, double tol = 1e-4,
           unsigned int seed = 42, Init init = Init::KMEANS_PLUS_PLUS);

    // Run Lloyd's algorithm on X (N, d). Returns the final inertia.
    double fit(const Tensor& X);

    // Same as fit() but also returns the per-iteration inertia history.
    std::vector<double> fit_with_history(const Tensor& X);

    // Predict hard cluster labels for X. Returns (N, 1) tensor of int-as-double.
    Tensor predict(const Tensor& X) const;

    // Accessors
    const Tensor& cluster_centers() const { return centers_; }
    double inertia() const { return last_inertia_; }
    size_t n_iter() const { return n_iter_; }

private:
    size_t n_clusters_;
    size_t max_iter_;
    double tol_;
    unsigned int seed_;
    Init init_;

    Tensor centers_;      // (K, d)
    double last_inertia_ = 0.0;
    size_t n_iter_ = 0;

    // Pick K initial centers. Uses internal RNG seeded from seed_.
    void init_centers(const Tensor& X, std::mt19937& rng);

    // One Lloyd step: assign + update. Returns the inertia of the new assignment.
    // Updates centers_ in place.
    double lloyd_step(const Tensor& X, std::vector<size_t>& labels,
                      std::vector<size_t>& new_labels);
};

// =============================================================================
// GMM — Gaussian Mixture Model fitted via Expectation-Maximization.
// =============================================================================
//
// Stores K weights π_k, K mean vectors μ_k (each (1, d)), and K covariance
// matrices Σ_k (each (d, d)). The mixture density is
//     p(x) = Σ_k π_k N(x | μ_k, Σ_k)
//
// E-step: per-point log-responsibility
//     log γ_{ik} = log π_k − 0.5 d log(2π) − 0.5 log|Σ_k|
//                  − 0.5 (x_i − μ_k)^T Σ_k^{-1} (x_i − μ_k)
// then log-sum-exp normalise per row to make responsibilities sum to 1.
//
// M-step:
//     N_k       = Σ_i γ_{ik}
//     π_k       = N_k / N
//     μ_k       = (Σ_i γ_{ik} x_i) / N_k
//     Σ_k       = (Σ_i γ_{ik} (x_i − μ_k)(x_i − μ_k)^T) / N_k + reg_covar · I
//
// Init: k-means++ by default (uses KMeans under the hood to produce K well-
// separated seeds; initial weights are uniform, initial covariances are the
// per-cluster covariance from KMeans with a small regularisation).
//
// API mirrors scikit-learn's GaussianMixture where reasonable:
//   fit(X) / fit_with_history(X, history)
//   predict_proba(X)  — (N, K) matrix of soft responsibilities
//   predict(X)        — (N, 1) hard cluster labels (argmax of responsibilities)
//   sample(n, seed)   — (n, d) samples from the fitted mixture
//   score(X)          — mean log-likelihood per sample
//   bic(X) / aic(X)   — model-selection criteria
//
// log_likelihood(X) returns the TOTAL log-likelihood (sum over rows), not the
// mean — that's `score()`.

class GMM {
public:
    GMM(size_t n_components, size_t max_iter = 100, double tol = 1e-3,
        double reg_covar = 1e-6, unsigned int seed = 42);

    // Fit the mixture to X (N, d) via EM. Returns the final log-likelihood.
    double fit(const Tensor& X);

    // Same as fit() but also fills history with the per-iteration log-likelihood.
    double fit_with_history(const Tensor& X, std::vector<double>& history);

    // Soft responsibilities: γ_{ik} = p(z_i = k | x_i).
    Tensor predict_proba(const Tensor& X) const;

    // Hard cluster labels: argmax_k γ_{ik}.
    Tensor predict(const Tensor& X) const;

    // Draw n samples from the fitted mixture.
    Tensor sample(size_t n, unsigned int sample_seed = 42) const;

    // Mean log-likelihood per sample (score convention).
    double score(const Tensor& X) const;

    // Total log-likelihood Σ_i log p(x_i).
    double log_likelihood(const Tensor& X) const;

    // Bayesian Information Criterion: −2 log L + p log N, p = free parameters.
    double bic(const Tensor& X) const;
    // Akaike Information Criterion: −2 log L + 2 p.
    double aic(const Tensor& X) const;

    // Accessors
    const Tensor& weights() const { return weights_; }
    const std::vector<Tensor>& means() const { return means_; }
    const std::vector<Tensor>& covariances() const { return covs_; }
    const std::vector<MultivariateGaussian>& components() const { return components_; }
    size_t n_components() const { return n_components_; }
    double reg_covar() const { return reg_covar_; }
    void set_reg_covar(double r) { reg_covar_ = r; }
    size_t n_iter() const { return n_iter_; }

private:
    size_t n_components_;
    size_t max_iter_;
    double tol_;
    double reg_covar_;
    unsigned int seed_;

    Tensor weights_;                  // (K, 1) mixture weights
    std::vector<Tensor> means_;       // K tensors each (1, d)
    std::vector<Tensor> covs_;        // K tensors each (d, d)
    std::vector<MultivariateGaussian> components_;  // K pre-built components
    size_t n_iter_ = 0;
    bool fitted_ = false;
    size_t input_dim_ = 0;

    // Initialise means / weights / covariances from KMeans++ on X.
    void init_from_kmeans(const Tensor& X, std::mt19937& rng);

    // Build components_ from current means_/covs_/weights_.
    void rebuild_components();

    // E-step: returns (N, K) log-responsibilities.
    Tensor e_step_log_resp(const Tensor& X) const;

    // M-step: update weights_, means_, covs_ from (N, K) log-responsibilities
    // and X (N, d). Uses the LOG-responsibilities (numerically stable) and
    // softmaxes them per row to get responsibilities.
    void m_step(const Tensor& X, const Tensor& log_resp);
};

#endif // GMM_H