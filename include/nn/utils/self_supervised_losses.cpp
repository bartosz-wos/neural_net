#include "self_supervised_losses.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// BYOL — Grill et al. 2020
// =============================================================================
// Forward:
//   L = (1/(2N·D)) * Σ_a ||z_a − z_target_{a⊕1}||²
//
// Backward (w.r.t. z_a only; z_target is treated as constant):
//   dL/dz_a = (2/(2N·D)) * (z_a − z_target_{a⊕1})
//
// For paired anchor a, the positive is a⊕1 = a XOR 1.

Tensor BYOL::forward(const Tensor& z, const Tensor& z_target) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2ND = 1.0 / static_cast<double>(twoN * D);

    double total = 0.0;
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        double row = 0.0;
        for (size_t j = 0; j < D; ++j) {
            double d = z[a][j] - z_target[pos][j];
            row += d * d;
        }
        total += row;
    }
    Tensor out(1, 1);
    out[0][0] = total * inv_2ND;
    return out;
}

Tensor BYOL::backward(const Tensor& z, const Tensor& z_target) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double factor = 2.0 / static_cast<double>(twoN * D);

    Tensor g(twoN, D);
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        for (size_t j = 0; j < D; ++j) {
            g[a][j] = factor * (z[a][j] - z_target[pos][j]);
        }
    }
    return g;
}


// =============================================================================
// VICReg — Bardes & Weston 2022
// =============================================================================
// Forward:
//   μ_d  = (1/(2N)) Σ_a z_a_d                                  (per-dim mean)
//   z_c_a = z_a − μ                                            (centred)
//   Var_d = (1/(2N)) Σ_a z_c_a_d²                              (per-dim variance)
//   Var-loss_d = max(0, γ − sqrt(Var_d))                       (hinge)
//   L_var = (1/D) Σ_d Var-loss_d
//
//   Inv-loss = (1/(2N)) Σ_a ||z_a − z_{a⊕1}||²
//
//   C_{ij} = (1/(2N)) Σ_a z_c_a_i · z_c_a_j                    (cross-corr matrix)
//   L_cov = (1/D) Σ_{i≠j} C_{ij}²
//
//   L = λ · L_inv + μ · L_var + ν · L_cov
//
// Backward (w.r.t. z_a):
//   dL_inv/dz_a_d = (2/(2N)) * (z_a_d − z_{a⊕1}_d) * λ
//   dL_cov/dz_a_d chain:
//     dL_cov/dC_{ij} = (2/D) * C_{ij}  for i≠j,  else 0
//     dC_{ij}/dz_c_a_d = δ_{id} z_c_a_j + δ_{jd} z_c_a_i
//     dC_{ij}/dz_a_d   = dC_{ij}/dz_c_a_d  (mean-centring adds 1/(2N) to each row of dμ)
//   dL_var/dz_a_d chain (subgradient at Var_d = γ²):
//     std_d = sqrt(Var_d); hinge active iff std_d < γ
//     dL_var/d std_d = (-μ/D) for active, 0 otherwise
//     d std_d / d Var_d = 1/(2 std_d)
//     d Var_d / dz_c_a_d = (2/(2N)) z_c_a_d
//     d Var_d / dz_a_d = d Var_d / dz_c_a_d − (2/(2N)²) Σ_b z_c_b_d
//                        (mean-centring chain — but in our cached-centred
//                         form we apply the centring at the end)
//
// Implementation: we cache mean_ (1×D) and centred_ (2N×D) and C_ (D×D).
// Backward computes dL/dz_c_a first, then folds the centring chain in a
// single pass:
//   dL/dz_a_d = dL/dz_c_a_d − (1/(2N)) Σ_b dL/dz_c_b_d

Tensor VICReg::forward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    // 1) mean_ (1, D)
    mean_ = Tensor(1, D);
    for (size_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (size_t a = 0; a < twoN; ++a) s += z[a][d];
        mean_[0][d] = s * inv_2N;
    }
    // 2) centred_ (2N, D)
    centered_ = Tensor(twoN, D);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            centered_[a][d] = z[a][d] - mean_[0][d];

    // 3) Cross-correlation matrix C_ (D, D) and per-dim variance.
    C_ = Tensor(D, D);
    std::vector<double> var_d(D, 0.0);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double s = 0.0;
            for (size_t a = 0; a < twoN; ++a) {
                s += centered_[a][i] * centered_[a][j];
            }
            C_[i][j] = s * inv_2N;
        }
    }
    for (size_t d = 0; d < D; ++d) var_d[d] = C_[d][d];

    // 4) L_inv (mean over 2N).
    double l_inv = 0.0;
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        double r = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = z[a][d] - z[pos][d];
            r += diff * diff;
        }
        l_inv += r;
    }
    l_inv /= static_cast<double>(twoN);

    // 5) L_var (mean over D) using hinge max(0, γ − std).
    double l_var = 0.0;
    for (size_t d = 0; d < D; ++d) {
        double std_d = std::sqrt(std::max(var_d[d], 0.0));
        double hinge = gamma_ - std_d;
        if (hinge > 0.0) l_var += hinge;
    }
    l_var /= static_cast<double>(D);

    // 6) L_cov (off-diagonal sum of squares, mean over D).
    double l_cov = 0.0;
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            if (i == j) continue;
            l_cov += C_[i][j] * C_[i][j];
        }
    }
    l_cov /= static_cast<double>(D);

    double total = sim_w_ * l_inv + std_w_ * l_var + cov_w_ * l_cov;
    Tensor out(1, 1);
    out[0][0] = total;
    return out;
}

Tensor VICReg::backward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    // If the cache (mean_/centered_/C_) doesn't match z, recompute it.
    // We compare shape as a cheap check.
    bool cache_valid = (mean_.rows == 1 && mean_.cols == D &&
                        centered_.rows == twoN && centered_.cols == D &&
                        C_.rows == D && C_.cols == D);
    if (!cache_valid) {
        forward(z);
    }

    // ----- Invariance term -----
    // L_inv = (1/(2N)) Σ_a ||z_a − z_{a⊕1}||²
    // For anchor a, the term z_a_d appears in TWO pairwise MSEs:
    //   (z_a − z_{a⊕1})  (as anchor a)
    //   (z_{a⊕1} − z_a)  (as the OTHER view of a⊕1)
    // Both contribute 2(z_a_d − z_{a⊕1}_d). So:
    //   dL_inv/dz_a_d = (1/(2N)) · 4 · (z_a_d − z_{a⊕1}_d)
    //                  = (2/N) · (z_a_d − z_{a⊕1}_d)
    Tensor d_inv(twoN, D);
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        for (size_t d = 0; d < D; ++d) {
            d_inv[a][d] = (4.0 * inv_2N) * (z[a][d] - z[pos][d]) * sim_w_;
        }
    }

    // ----- Covariance term -----
    // d L_cov / d C_{ij} = (2/D) C_{ij}  (off-diag only)
    // d C_{ij} / d z_c_a_d = (1/(2N)) (δ_{id} z_c_a_j + δ_{jd} z_c_a_i)
    // Therefore, d L_cov / d z_c_a_d = (1/(2N)) · 2 · (2·cov_w_/D) Σ_{j≠d} C_{d,j} z_c_a_j
    //                              = (2·cov_w_/(N·D)) Σ_{j≠d} C_{d,j} z_c_a_j
    //                              = (4·cov_w_/(2N·D)) Σ_{j≠d} C_{d,j} z_c_a_j
    Tensor d_cov(twoN, D);
    {
        double factor = (4.0 * cov_w_) / (static_cast<double>(twoN) * static_cast<double>(D));
        for (size_t a = 0; a < twoN; ++a) {
            for (size_t d = 0; d < D; ++d) {
                double s = 0.0;
                for (size_t j = 0; j < D; ++j) {
                    if (j == d) continue;
                    s += C_[d][j] * centered_[a][j];
                }
                d_cov[a][d] = factor * s;
            }
        }
    }

    // ----- Variance term (hinge) -----
    Tensor d_var(twoN, D);
    {
        double factor = std_w_ / static_cast<double>(D);
        for (size_t a = 0; a < twoN; ++a) {
            for (size_t d = 0; d < D; ++d) {
                double std_d = std::sqrt(std::max(C_[d][d], 0.0));
                double hinge_active = (std_d > 0.0 && std_d < gamma_) ? 1.0 : 0.0;
                d_var[a][d] = -factor * hinge_active * (centered_[a][d] / (static_cast<double>(twoN) * std_d));
            }
        }
    }

    // Sum
    Tensor d_zc(twoN, D);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            d_zc[a][d] = d_inv[a][d] + d_cov[a][d] + d_var[a][d];

    // Fold centring chain
    Tensor grad(twoN, D);
    std::vector<double> sum_over_a(D, 0.0);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            sum_over_a[d] += d_zc[a][d];
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            grad[a][d] = d_zc[a][d] - inv_2N * sum_over_a[d];

    return grad;
}


// =============================================================================
// BarlowTwins — Zbontar et al. 2021
// =============================================================================
// 1) μ_d = (1/2N) Σ_a z_a_d
// 2) z_c_a = z_a − μ
// 3) C_{ij} = (1/(2N)) Σ_a z_c_a_i z_c_a_j
// 4) L = Σ_i (1 − C_{ii})² + λ Σ_{i≠j} C_{ij}²
//
// Backward:
//   dL/dC_{ii} = −2(1 − C_{ii})       (diagonal)
//   dL/dC_{ij} = 2 λ C_{ij}           (off-diagonal)
//   dC_{ij}/dz_c_a_d = (1/(2N)) (δ_{id} z_c_a_j + δ_{jd} z_c_a_i)
//   dC_{ij}/dz_a_d   = dC_{ij}/dz_c_a_d − (1/2N) Σ_b (dC_{ij}/dz_c_b_d)
//                      (centring chain)
//
// Implementation caches mean_, centered_, C_.

Tensor BarlowTwins::forward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    mean_ = Tensor(1, D);
    for (size_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (size_t a = 0; a < twoN; ++a) s += z[a][d];
        mean_[0][d] = s * inv_2N;
    }
    centered_ = Tensor(twoN, D);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            centered_[a][d] = z[a][d] - mean_[0][d];

    C_ = Tensor(D, D);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double s = 0.0;
            for (size_t a = 0; a < twoN; ++a) s += centered_[a][i] * centered_[a][j];
            C_[i][j] = s * inv_2N;
        }
    }

    double l_diag = 0.0;
    double l_off = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double d = 1.0 - C_[i][i];
        l_diag += d * d;
        for (size_t j = 0; j < D; ++j) {
            if (i == j) continue;
            l_off += C_[i][j] * C_[i][j];
        }
    }
    Tensor out(1, 1);
    out[0][0] = l_diag + lambda_ * l_off;
    return out;
}

Tensor BarlowTwins::backward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    bool cache_valid = (mean_.rows == 1 && mean_.cols == D &&
                        centered_.rows == twoN && centered_.cols == D &&
                        C_.rows == D && C_.cols == D);
    if (!cache_valid) {
        forward(z);
    }

    // dL/dC_{ii} = −2(1 − C_{ii})       (diagonal)
    // dL/dC_{ij} = 2 λ C_{ij}           (off-diagonal)
    // dC_{ij}/dz_c_a_d = (1/(2N)) (δ_{id} z_c_a_j + δ_{jd} z_c_a_i)
    //
    // dL/dz_c_a_d = (1/(2N)) · 2 · Σ_j dL/dC_{dj} · z_c_a_j  (δ_{id} and δ_{jd}
    //                                              contributions are equal by C and dL/dC symmetry)
    //             = (1/N) · (dL/dC)_{row d} · z_c_a
    Tensor d_C(D, D);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            if (i == j) d_C[i][j] = -2.0 * (1.0 - C_[i][i]);
            else        d_C[i][j] = 2.0 * lambda_ * C_[i][j];
        }
    }

    Tensor d_zc(twoN, D);
    double scale = 2.0 * inv_2N;  // = 1/N
    for (size_t a = 0; a < twoN; ++a) {
        for (size_t d = 0; d < D; ++d) {
            double s = 0.0;
            for (size_t j = 0; j < D; ++j) s += d_C[d][j] * centered_[a][j];
            d_zc[a][d] = scale * s;
        }
    }

    // Fold centring chain: dL/dz_a = dL/dz_c_a − (1/2N) Σ_b dL/dz_c_b
    Tensor grad(twoN, D);
    std::vector<double> sum_over_a(D, 0.0);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            sum_over_a[d] += d_zc[a][d];
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            grad[a][d] = d_zc[a][d] - inv_2N * sum_over_a[d];

    return grad;
}


// =============================================================================
// DINO — Caron et al. 2021
// =============================================================================
namespace {
Tensor softmax_with_temperature(const Tensor& z, double tau) {
    size_t rows = z.rows;
    size_t K = z.cols;
    Tensor p(rows, K);
    double inv_tau = 1.0 / tau;
    for (size_t a = 0; a < rows; ++a) {
        double mx = -1e300;
        for (size_t k = 0; k < K; ++k) mx = std::max(mx, z[a][k] * inv_tau);
        double sum_exp = 0.0;
        for (size_t k = 0; k < K; ++k) {
            double e = std::exp(z[a][k] * inv_tau - mx);
            p[a][k] = e;
            sum_exp += e;
        }
        if (sum_exp < 1e-30) sum_exp = 1e-30;
        for (size_t k = 0; k < K; ++k) p[a][k] /= sum_exp;
    }
    return p;
}
}  // namespace

Tensor DINO::forward(const Tensor& z_student, const Tensor& z_teacher) {
    size_t twoN = z_student.rows;
    size_t K = z_student.cols;

    student_probs_ = softmax_with_temperature(z_student, tau_s_);

    Tensor centered_teacher(twoN, K);
    {
        double inv_2N = 1.0 / static_cast<double>(twoN);
        for (size_t k = 0; k < K; ++k) {
            double s = 0.0;
            for (size_t a = 0; a < twoN; ++a) s += z_teacher[a][k];
            double m = s * inv_2N;
            for (size_t a = 0; a < twoN; ++a) centered_teacher[a][k] = z_teacher[a][k] - m;
        }
    }
    teacher_probs_ = softmax_with_temperature(centered_teacher, tau_t_);

    double total = 0.0;
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        double row = 0.0;
        for (size_t k = 0; k < K; ++k) {
            double t = teacher_probs_[pos][k];
            if (t <= 0.0) continue;
            double s = student_probs_[a][k];
            if (s <= 0.0) continue;
            row += t * std::log(s);
        }
        total += -row;
    }
    Tensor out(1, 1);
    out[0][0] = total / static_cast<double>(twoN);
    return out;
}

Tensor DINO::backward(const Tensor& z_student, const Tensor& z_teacher) {
    size_t twoN = z_student.rows;
    size_t K = z_student.cols;

    bool cache_valid = (student_probs_.rows == twoN && student_probs_.cols == K &&
                        teacher_probs_.rows == twoN && teacher_probs_.cols == K);
    if (!cache_valid) {
        forward(z_student, z_teacher);
    }

    double factor = 1.0 / (static_cast<double>(twoN) * tau_s_);

    Tensor grad(twoN, K);
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        for (size_t k = 0; k < K; ++k) {
            grad[a][k] = factor * (student_probs_[a][k] - teacher_probs_[pos][k]);
        }
    }
    return grad;
}


// =============================================================================
// W-MSE — Ermolov et al. 2021
// =============================================================================
Tensor WMSE::forward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    // Centre
    mean_ = Tensor(1, D);
    for (size_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (size_t a = 0; a < twoN; ++a) s += z[a][d];
        mean_[0][d] = s * inv_2N;
    }
    centered_ = Tensor(twoN, D);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            centered_[a][d] = z[a][d] - mean_[0][d];

    // A = (1/2N) z_cᵀ z_c + ε I  (D × D)
    A_ = Tensor(D, D);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double s = 0.0;
            for (size_t a = 0; a < twoN; ++a) s += centered_[a][i] * centered_[a][j];
            A_[i][j] = s * inv_2N + (i == j ? eps_ : 0.0);
        }
    }

    // Cholesky: A = L Lᵀ with L lower-triangular
    std::vector<std::vector<double>> L(D, std::vector<double>(D, 0.0));
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = A_[i][j];
            for (size_t k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s < 1e-30) s = 1e-30;
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }

    // W = L^{-T} (inverse of Lᵀ)
    std::vector<std::vector<double>> LT(D, std::vector<double>(D, 0.0));
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < D; ++j)
            LT[i][j] = L[j][i];

    std::vector<std::vector<double>> invLT(D, std::vector<double>(D, 0.0));
    for (size_t col = 0; col < D; ++col) {
        for (size_t i = D; i-- > 0;) {
            double s = (i == static_cast<size_t>(col)) ? 1.0 : 0.0;
            for (size_t k = i + 1; k < D; ++k) s -= LT[i][k] * invLT[k][col];
            invLT[i][col] = s / LT[i][i];
        }
    }
    W_ = Tensor(D, D);
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < D; ++j)
            W_[i][j] = invLT[i][j];

    // Compute whitened z_c (W z_c_a, shape 2N × D)
    std::vector<std::vector<double>> ZW(twoN, std::vector<double>(D, 0.0));
    for (size_t a = 0; a < twoN; ++a) {
        for (size_t i = 0; i < D; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < D; ++j) s += W_[i][j] * centered_[a][j];
            ZW[a][i] = s;
        }
    }

    double total = 0.0;
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        double r = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = ZW[a][d] - ZW[pos][d];
            r += diff * diff;
        }
        total += r;
    }
    total /= static_cast<double>(twoN * D);
    Tensor out(1, 1);
    out[0][0] = total;
    return out;
}

Tensor WMSE::backward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    bool cache_valid = (mean_.rows == 1 && mean_.cols == D &&
                        centered_.rows == twoN && centered_.cols == D &&
                        W_.rows == D && W_.cols == D);
    if (!cache_valid) {
        forward(z);
    }
    double factor = 2.0 / static_cast<double>(twoN * D);

    // Re-compute whitened z_c
    std::vector<std::vector<double>> ZW(twoN, std::vector<double>(D, 0.0));
    for (size_t a = 0; a < twoN; ++a) {
        for (size_t i = 0; i < D; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < D; ++j) s += W_[i][j] * centered_[a][j];
            ZW[a][i] = s;
        }
    }

    // d L / d (W z_c_a)_i: each pair (a, a⊕1) appears TWICE in Σ_a (once as
    // anchor a, once as anchor a⊕1's other-view) so dL/dzw_a gets 2x the
    // direct chain:
    //   d L / d (W z_c_a)_i = (4/(2N·D)) * ((W z_c_a)_i − (W z_c_{a⊕1})_i)
    Tensor d_ZW(twoN, D);
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        for (size_t i = 0; i < D; ++i) {
            d_ZW[a][i] = 2.0 * factor * (ZW[a][i] - ZW[pos][i]);
        }
    }

    // d L / d z_c_a_d = Σ_i (d L / d (W z_c_a)_i) * W_{i, d}
    Tensor d_zc(twoN, D);
    for (size_t a = 0; a < twoN; ++a) {
        for (size_t d = 0; d < D; ++d) {
            double s = 0.0;
            for (size_t i = 0; i < D; ++i) s += d_ZW[a][i] * W_[i][d];
            d_zc[a][d] = s;
        }
    }

    // Fold centring chain
    Tensor grad(twoN, D);
    std::vector<double> sum_over_a(D, 0.0);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            sum_over_a[d] += d_zc[a][d];
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            grad[a][d] = d_zc[a][d] - inv_2N * sum_over_a[d];

    return grad;
}


// Forward with an explicit whitening matrix W. Used by tests to verify the
// gradient when W is held fixed (the canonical W-MSE convention).
Tensor WMSE::forward_with_W(const Tensor& z, const Tensor& W) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    double inv_2N = 1.0 / static_cast<double>(twoN);

    Tensor mean_local(1, D);
    for (size_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (size_t a = 0; a < twoN; ++a) s += z[a][d];
        mean_local[0][d] = s * inv_2N;
    }
    Tensor z_c(twoN, D);
    for (size_t a = 0; a < twoN; ++a)
        for (size_t d = 0; d < D; ++d)
            z_c[a][d] = z[a][d] - mean_local[0][d];

    std::vector<std::vector<double>> ZW(twoN, std::vector<double>(D, 0.0));
    for (size_t a = 0; a < twoN; ++a) {
        for (size_t i = 0; i < D; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < D; ++j) s += W[i][j] * z_c[a][j];
            ZW[a][i] = s;
        }
    }
    double total = 0.0;
    for (size_t a = 0; a < twoN; ++a) {
        size_t pos = a ^ 1;
        double r = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = ZW[a][d] - ZW[pos][d];
            r += diff * diff;
        }
        total += r;
    }
    total /= static_cast<double>(twoN * D);
    Tensor out(1, 1);
    out[0][0] = total;
    return out;
}
