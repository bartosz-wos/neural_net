#ifndef MMD_LOSS_H
#define MMD_LOSS_H

#include "../core/tensor.h"
#include <cmath>
#include <string>
#include <utility>

// =============================================================================
// Maximum Mean Discrepancy (MMD) Loss
// =============================================================================
// Gretton et al. 2007 "A Kernel Statistical Test of Independence"; Gretton et
// al. 2012 "A Kernel Two-Sample Test" (JMLR 13, 723-773). Used heavily in
// domain adaptation (DeepCORAL, DANN) and implicit generative models.
//
// Given two empirical distributions p (sample X ∈ R^{m x d}) and q (sample
// Y ∈ R^{n x d}), the squared MMD between their kernel mean embeddings in
// an RKHS H is:
//
//     MMD^2(p, q) = ||E_p[φ(X)] - E_q[φ(Y)]||^2_H
//                 =  E_{X,X'~p}[k(X, X')] + E_{Y,Y'~q}[k(Y, Y')]
//                  - 2 * E_{X~p, Y~q}[k(X, Y)]
//
// We use the UNBIASED U-statistic estimator (standard in DANN / DeepCORAL):
//
//     L = (1/(m(m-1))) * sum_{i != j} k(X_i, X_j)             [within p]
//       + (1/(n(n-1))) * sum_{i != j} k(Y_i, Y_j)             [within q]
//       - (2 / (m n)) * sum_{i, j} k(X_i, Y_j)               [between]
//
// When m == 1 or n == 1 we silently fall back to the BIASED estimator
// (use m, n, m*n as denominators and include the diagonal). This is the
// standard practice: a one-per-batch target sample is common in early
// stages of training, and the biased form is still a valid discrepancy.
//
// Three kernels are supported (all characteristic — MMD^2 = 0 iff p = q):
//   - GAUSSIAN_RBF:           k(x, y) = exp(-||x - y||² / (2 σ²))
//   - POLYNOMIAL:             k(x, y) = (γ · <x, y> + c0)^d
//   - INVERSE_MULTI_QUADRIC:  k(x, y) = 1 / sqrt(||x - y||² + c0²)
//
// Conventions:
//   - forward(X, Y) → Tensor(1, 1)  (scalar L).
//   - backward(X, Y) → (grad_X, grad_Y)  where grad_X has shape (m, d) and
//     grad_Y has shape (n, d). Each element is ∂L/∂X[i] and ∂L/∂Y[j].
//
// Time complexity per forward and per backward: O((m+n)² · d). Same as
// computing the full kernel matrix. Subsample if your batches are huge.

class MMDLoss {
public:
    enum KernelType {
        GAUSSIAN_RBF = 0,
        POLYNOMIAL = 1,
        INVERSE_MULTI_QUADRIC = 2
    };

    // Construct with explicit kernel + hyperparameters.
    //   - GAUSSIAN_RBF:            sigma       (default 1.0)
    //   - POLYNOMIAL:              gamma, c0, degree   (defaults 1.0, 0.0, 3)
    //   - INVERSE_MULTI_QUADRIC:   c0          (default 1.0)
    MMDLoss(KernelType kernel,
            double sigma = 1.0,
            double gamma = 1.0,
            double c0 = 1.0,
            int degree = 3);

    // Convenience constructors
    static MMDLoss gaussian(double sigma = 1.0) {
        return MMDLoss(GAUSSIAN_RBF, sigma);
    }
    static MMDLoss polynomial(double gamma = 1.0, double c0 = 0.0, int degree = 3) {
        // Signature is MMDLoss(kernel, sigma, gamma, c0, degree).
        return MMDLoss(POLYNOMIAL, 1.0 /*sigma (unused)*/, gamma, c0, degree);
    }
    static MMDLoss imq(double c0 = 1.0) {
        return MMDLoss(INVERSE_MULTI_QUADRIC, 1.0, 1.0, c0, 0);
    }

    Tensor forward(const Tensor& X, const Tensor& Y);
    std::pair<Tensor, Tensor> backward(const Tensor& X, const Tensor& Y);

    // Accessors
    KernelType kernel_type() const { return kernel_; }
    double sigma() const { return sigma_; }
    double gamma() const { return gamma_; }
    double c0()   const { return c0_; }
    int    degree() const { return degree_; }

    // Last computed kernel matrices (for testing / introspection):
    //   last_kxx_ ∈ R^{m, m}, last_kyy_ ∈ R^{n, n}, last_kxy_ ∈ R^{m, n}
    const Tensor& last_kxx() const { return kxx_; }
    const Tensor& last_kyy() const { return kyy_; }
    const Tensor& last_kxy() const { return kxy_; }

    // Was the last forward in unbiased form (m > 1 && n > 1)?
    bool last_unbiased() const { return last_unbiased_; }

    std::string config_string() const;

private:
    // Fill K (rows_a × rows_b) with k(a_row_i, b_row_j). Used by forward.
    void compute_kernel_matrix(const Tensor& a, const Tensor& b, Tensor& K) const;

    // Compute (∂L/∂a) given that K[i, j] = k(a_i, b_j) and the upstream
    // gradient G[i, j] = ∂L/∂K[i, j]. The diagonal terms (i == j in the
    // a == b case) need to be carefully handled to support the unbiased
    // estimator (where the diagonal is excluded from the sum).
    void accumulate_grad_a(const Tensor& a, const Tensor& b,
                           const Tensor& K, const Tensor& G,
                           Tensor& grad_a) const;
    void accumulate_grad_b(const Tensor& a, const Tensor& b,
                           const Tensor& K, const Tensor& G,
                           Tensor& grad_b) const;

    // Configuration
    KernelType kernel_;
    double sigma_;
    double gamma_;
    double c0_;
    int    degree_;

    // Last computed kernel matrices (m, m), (n, n), (m, n)
    Tensor kxx_;
    Tensor kyy_;
    Tensor kxy_;

    // Last input sizes (debug aid for non-vacuous testing)
    size_t last_m_ = 0;
    size_t last_n_ = 0;
    size_t last_d_ = 0;

    // Whether last forward was unbiased
    bool last_unbiased_ = false;
};

#endif
