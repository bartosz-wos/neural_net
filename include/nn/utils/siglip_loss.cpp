// siglip_loss.cpp — SigLIP-style elementwise sigmoid loss for image-text pairs.
//
// For input z_img (N, D) and z_txt (N, D):
//   S[i][j]   = cos(z_img[i], z_txt[j])   if normalize=true (default)
//             = z_img[i] . z_txt[j]        if normalize=false
//   z*[i][j]  = scale * S[i][j] + bias
//   t[i][j]   = +1   if i == j
//             = -1   otherwise
//   L[i]      = (1/N) * sum_j softplus(-z*[i][j] * t[i][j])
//   L         = (1/N) * sum_i L[i]         (mean over batch rows)
//
// Backward chain (derived in header):
//   dL/dz*[i][j] = (1/N) * t[i][j] * (1 - sigmoid(z*[i][j] * t[i][j]))
//   dL/dS        = (1/N) * scale * t * (1 - sigmoid(z* * t))
// Then dL/dS propagates to dL/dz_img, dL/dz_txt through the similarity chain
// (cosine: L2-normalize + inner product; dot: direct inner product).

#include "siglip_loss.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

// Numerically-stable softplus:
//     softplus(x) = max(x, 0) + log(1 + exp(-|x|))
// Always finite for any real x.
inline double softplus_stable(double x) {
    if (x > 0.0) {
        return x + std::log1p(std::exp(-x));
    } else {
        return std::log1p(std::exp(x));
    }
}

// Numerically-stable sigmoid: avoids overflow for large |x|.
inline double sigmoid_stable(double x) {
    if (x >= 0.0) {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// L2-normalize a (rows, cols) tensor row-wise in place into `out`.
// If a row has norm < eps, leave it zero (matches InfoNCE convention).
void l2_normalize_rows(const Tensor& z, Tensor& out, double eps) {
    size_t rows = z.rows;
    size_t cols = z.cols;
    for (size_t i = 0; i < rows; ++i) {
        double sumsq = 0.0;
        for (size_t j = 0; j < cols; ++j) sumsq += z[i][j] * z[i][j];
        double norm = std::sqrt(sumsq);
        if (norm < eps) {
            for (size_t j = 0; j < cols; ++j) out[i][j] = 0.0;
        } else {
            double inv = 1.0 / norm;
            for (size_t j = 0; j < cols; ++j) out[i][j] = z[i][j] * inv;
        }
    }
}

}  // anonymous namespace


Tensor SigLIPLoss::forward(const Tensor& z_img, const Tensor& z_txt) {
    size_t N = z_img.rows;
    size_t D = z_img.cols;
    // Defensive shape checks: must be (N, D) for both, with same N and D.
    if (z_txt.rows != N || z_txt.cols != D) {
        // If shapes don't match, the user has a bug; return a zero loss
        // and an empty cache rather than crashing. (Could also throw —
        // match the existing contrastive_losses style which doesn't throw.)
        Tensor empty(1, 1);
        empty[0][0] = 0.0;
        return empty;
    }
    last_N_ = static_cast<int>(N);
    last_D_ = static_cast<int>(D);

    // Step 1: Optionally L2-normalize rows.
    Tensor z_img_eff(N, D);
    Tensor z_txt_eff(N, D);
    if (normalize_) {
        l2_normalize_rows(z_img, z_img_eff, eps_);
        l2_normalize_rows(z_txt, z_txt_eff, eps_);
    } else {
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < D; ++k) {
                z_img_eff[i][k] = z_img[i][k];
                z_txt_eff[i][k] = z_txt[i][k];
            }
        }
    }
    last_z_img_norm_ = z_img_eff;
    last_z_txt_norm_ = z_txt_eff;

    // Step 2: Similarity matrix S[i][j] = z_img_eff[i] . z_txt_eff[j]
    Tensor S(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < D; ++k) s += z_img_eff[i][k] * z_txt_eff[j][k];
            S[i][j] = s;
        }
    }
    last_S_ = S;

    // Step 3: Targets t[i][j] = +1 if i==j, else -1
    Tensor t(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            t[i][j] = (i == j) ? 1.0 : -1.0;
        }
    }
    last_targets_ = t;

    // Step 4: z* = scale * S + bias
    Tensor zstar(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            zstar[i][j] = scale_ * S[i][j] + bias_;
        }
    }
    last_zstar_ = zstar;

    // Step 5: Per-row loss L[i] = (1/N) * sum_j softplus(-z*[i][j] * t[i][j])
    // Total loss L = (1/N) * sum_i L[i] = (1/N^2) * sum_{i,j} softplus(-z*[i][j] * t[i][j])
    double total = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            double x = -zstar[i][j] * t[i][j];
            row_sum += softplus_stable(x);
        }
        total += row_sum;
    }
    double inv_N2 = 1.0 / static_cast<double>(N * N);
    Tensor out(1, 1);
    out[0][0] = total * inv_N2;
    return out;
}


std::pair<Tensor, Tensor> SigLIPLoss::backward(const Tensor& z_img, const Tensor& z_txt) {
    size_t N = static_cast<size_t>(last_N_);
    size_t D = static_cast<size_t>(last_D_);
    if (N == 0 || D == 0) {
        // backward() called before forward(): no cache. Return empty tensors.
        // Callers should call forward() first to populate the cache.
        return std::make_pair(Tensor(0, 0), Tensor(0, 0));
    }

    // Step 1: dL/dz*[i][j] = (1/N^2) * t[i][j] * (sigmoid(z*[i][j] * t[i][j]) - 1)
    // (The (1 - sigmoid) form is wrong-signed; sigmoid(a) - 1 is correct.)
    // d/dz* softplus(-z* t) = -t * sigmoid(-z* t) = -t * (1 - sigmoid(z* t))
    //                        =  t * (sigmoid(z* t) - 1)
    // Combined with the 1/N^2 mean normalization:
    //   dL/dz*[i][j] = (1/N^2) * t * (sigmoid(z* * t) - 1)
    // Then dL/dS = scale * dL/dz*.
    Tensor grad_zstar(N, N);
    double inv_N2 = 1.0 / static_cast<double>(N * N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double t_ij = last_targets_[i][j];
            double z_ij = last_zstar_[i][j];
            double sig = sigmoid_stable(z_ij * t_ij);
            grad_zstar[i][j] = inv_N2 * t_ij * (sig - 1.0);
        }
    }

    // Step 2: dL/dS = scale * grad_zstar
    Tensor grad_S(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            grad_S[i][j] = scale_ * grad_zstar[i][j];
        }
    }

    // Step 3: Chain to grad_z_img and grad_z_txt through similarity.
    Tensor grad_img(N, D);
    Tensor grad_txt(N, D);
    grad_img.fill(0.0);
    grad_txt.fill(0.0);

    // Use the post-L2-normalized inputs for the cosine chain (cached).
    // For dot-product mode, last_*_norm_ == raw input.
    const Tensor& u_img = last_z_img_norm_;
    const Tensor& u_txt = last_z_txt_norm_;

    if (!normalize_) {
        // Dot product: ds_ij/dz_img[i][k] = z_txt[j][k]
        //             ds_ij/dz_txt[j][k] = z_img[i][k]
        // grad_z_img[i][k] = sum_j grad_S[i][j] * z_txt[j][k]
        // grad_z_txt[j][k] = sum_i grad_S[i][j] * z_img[i][k]
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double g = grad_S[i][j];
                for (size_t k = 0; k < D; ++k) {
                    grad_img[i][k] += g * z_txt[j][k];
                    grad_txt[j][k] += g * z_img[i][k];
                }
            }
        }
    } else {
        // Cosine similarity on L2-normalized inputs.
        // Let u_img = z_img / ||z_img||, u_txt = z_txt / ||z_txt|| (cached).
        // s_ij = u_img[i] . u_txt[j].
        //
        // Chain through inner product:
        //   ds_ij/d u_img[i][k] = u_txt[j][k]
        //   ds_ij/d u_txt[j][k] = u_img[i][k]
        //   grad_u_img[i] = sum_j grad_S[i][j] * u_txt[j]
        //   grad_u_txt[j] = sum_i grad_S[i][j] * u_img[i]
        //
        // Then chain through L2-norm: u_i = z_i / ||z_i||.
        //   du_i[k]/dz_i[k] = (1/||z_i||) - z_i[k] * z_i[k] / ||z_i||^3
        //                    = (1/||z_i||) * (1 - u_i[k]^2)
        //   du_i[k]/dz_i[m] (k != m) = - z_i[k] * z_i[m] / ||z_i||^3
        //                            = - u_i[k] * u_i[m] / ||z_i||
        // In matrix form:
        //   grad_z[i] = (grad_u[i] - u_i * <grad_u[i], u_i>) / ||z_i||
        //
        // We need ||z_i|| — recover from the input z_img or z_txt.

        // First pass: grad_u_img, grad_u_txt
        Tensor grad_u_img(N, D);
        Tensor grad_u_txt(N, D);
        grad_u_img.fill(0.0);
        grad_u_txt.fill(0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double g = grad_S[i][j];
                for (size_t k = 0; k < D; ++k) {
                    grad_u_img[i][k] += g * u_txt[j][k];
                    grad_u_txt[j][k] += g * u_img[i][k];
                }
            }
        }

        // Second pass: chain through L2-normalization for z_img
        for (size_t i = 0; i < N; ++i) {
            double znorm_sq = 0.0;
            for (size_t k = 0; k < D; ++k) znorm_sq += z_img[i][k] * z_img[i][k];
            double znorm = std::sqrt(znorm_sq);
            if (znorm < eps_) {
                // Degenerate: leave grad zero (matches forward's zero sim row).
                continue;
            }
            double dot_gu_u = 0.0;
            for (size_t k = 0; k < D; ++k) {
                dot_gu_u += grad_u_img[i][k] * u_img[i][k];
            }
            double inv_znorm = 1.0 / znorm;
            for (size_t k = 0; k < D; ++k) {
                grad_img[i][k] = (grad_u_img[i][k] - u_img[i][k] * dot_gu_u) * inv_znorm;
            }
        }
        // Same for z_txt
        for (size_t j = 0; j < N; ++j) {
            double znorm_sq = 0.0;
            for (size_t k = 0; k < D; ++k) znorm_sq += z_txt[j][k] * z_txt[j][k];
            double znorm = std::sqrt(znorm_sq);
            if (znorm < eps_) {
                continue;
            }
            double dot_gu_u = 0.0;
            for (size_t k = 0; k < D; ++k) {
                dot_gu_u += grad_u_txt[j][k] * u_txt[j][k];
            }
            double inv_znorm = 1.0 / znorm;
            for (size_t k = 0; k < D; ++k) {
                grad_txt[j][k] = (grad_u_txt[j][k] - u_txt[j][k] * dot_gu_u) * inv_znorm;
            }
        }
    }

    return std::make_pair(grad_img, grad_txt);
}
