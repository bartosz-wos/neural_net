// contrastive_losses.cpp — InfoNCE / NT-Xent contrastive loss.
//
// For input z ∈ R^{2N x D}, with 2N = (batch_size * 2 views), the loss is:
//
//     L = (1 / (2N)) * sum_i L_i
//     L_i = -logits[i][pos_i] + logsumexp_{k != i}(logits[i][k])
//     logits[i][j] = sim(z_i, z_j) / T
//
// where:
//   - sim(z_i, z_j) = cos(z_i, z_j)            (default, normalize=true)
//                  = z_i . z_j / (||z_i|| ||z_j||)
//   - sim(z_i, z_j) = z_i . z_j                (dot-product mode, normalize=false)
//   - pos_i = i XOR 1                            (SimCLR convention)
//            or positive_indices_[i]              (override)
//
// Backward chain:
//   dL/dlogits[i][j] = (1 / (2N)) * (p_ij - delta_{j, pos_i})
//   where p_ij = softmax_{k != i}(logits[i][k]).
//   Then chain to dL/dz via the sim derivative:
//     - Dot: ds_ij/dz_i = z_j,  ds_ij/dz_j = z_i
//     - Cosine: needs both L2-normalization chain and inner-product chain.

#include "contrastive_losses.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

// Compute the positive index of row i under the XOR convention.
inline int default_pos(int i) { return i ^ 1; }

// L2-normalize a (rows, cols) tensor row-wise in place into `out`.
// If a row has norm < eps, leave it zero (caller's problem).
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

// Stable logsumexp over a vector of doubles, ignoring entries with `mask` = false.
// Returns -inf if all are masked out (shouldn't happen here since we always
// have at least 2N-1 valid entries).
double masked_logsumexp(const Tensor& logits, int row, const std::vector<bool>& mask) {
    size_t cols = logits.cols;
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < cols; ++k) {
        if (mask[k]) {
            double v = logits[row][k];
            if (v > max_val) max_val = v;
        }
    }
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return -std::numeric_limits<double>::infinity();
    }
    double sum_exp = 0.0;
    for (size_t k = 0; k < cols; ++k) {
        if (mask[k]) {
            sum_exp += std::exp(logits[row][k] - max_val);
        }
    }
    return max_val + std::log(sum_exp);
}

}  // anonymous namespace


Tensor InfoNCELoss::forward(const Tensor& z) {
    size_t twoN = z.rows;
    size_t D = z.cols;
    last_2N_ = static_cast<int>(twoN);
    last_D_ = static_cast<int>(D);
    last_logits_ = Tensor(twoN, twoN);
    last_probs_ = Tensor(twoN, twoN);
    last_probs_.fill(0.0);

    // Step 1: Optionally L2-normalize rows.
    Tensor z_norm(twoN, D);
    if (normalize_) {
        l2_normalize_rows(z, z_norm, eps_);
    } else {
        for (size_t i = 0; i < twoN; ++i)
            for (size_t j = 0; j < D; ++j)
                z_norm[i][j] = z[i][j];
    }
    last_normalized_ = z_norm;

    // Step 2: Similarity matrix.
    Tensor S(twoN, twoN);
    for (size_t i = 0; i < twoN; ++i) {
        for (size_t j = 0; j < twoN; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < D; ++k) s += z_norm[i][k] * z_norm[j][k];
            S[i][j] = s;
        }
    }

    // Step 3: logits = S / T.
    double inv_T = 1.0 / temperature_;
    for (size_t i = 0; i < twoN; ++i)
        for (size_t j = 0; j < twoN; ++j)
            last_logits_[i][j] = S[i][j] * inv_T;

    // Step 4: Per-anchor loss L_i = -logits[i][pos_i] + logsumexp_{k != i}(logits[i][k])
    std::vector<bool> mask(twoN, true);
    double total_loss = 0.0;
    for (size_t i = 0; i < twoN; ++i) {
        int pos_i = positive_indices_.empty()
                    ? default_pos(static_cast<int>(i))
                    : positive_indices_[i];

        // Mask out the diagonal for the softmax (and also any entry explicitly
        // equal to i if positive_indices_ points to a different row -- but i is
        // always excluded from its own denominator).
        for (size_t k = 0; k < twoN; ++k) mask[k] = (k != i);

        double lse = masked_logsumexp(last_logits_, static_cast<int>(i), mask);
        double Li = -last_logits_[i][pos_i] + lse;
        total_loss += Li;

        // Cache softmax probs: p[i][k] = exp(logits[i][k] - lse) for k != i
        // We don't store i itself (it's masked out, so p[i][i] = 0).
        for (size_t k = 0; k < twoN; ++k) {
            if (k == i) {
                last_probs_[i][k] = 0.0;
            } else {
                last_probs_[i][k] = std::exp(last_logits_[i][k] - lse);
            }
        }
    }

    // Step 5: Mean over 2N anchors.
    Tensor out(1, 1);
    out[0][0] = total_loss / static_cast<double>(twoN);
    return out;
}


Tensor InfoNCELoss::backward(const Tensor& z) {
    size_t twoN = static_cast<size_t>(last_2N_);
    size_t D = static_cast<size_t>(last_D_);

    // Step 1: grad_logits[i][j] = (1 / (2N)) * (p[i][j] - delta_{j, pos_i})
    // No extra 1/T factor yet — we'll fold it into the chain through sim.
    // Actually, dL/dlogits = (1/(2N)) * (p - e_pos). And dL/dS = dL/dlogits / T.
    Tensor grad_logits(twoN, twoN);
    double inv_2N = 1.0 / static_cast<double>(twoN);
    for (size_t i = 0; i < twoN; ++i) {
        int pos_i = positive_indices_.empty()
                    ? default_pos(static_cast<int>(i))
                    : positive_indices_[i];
        for (size_t j = 0; j < twoN; ++j) {
            double delta = (j == static_cast<size_t>(pos_i)) ? 1.0 : 0.0;
            grad_logits[i][j] = inv_2N * (last_probs_[i][j] - delta);
        }
    }

    // Step 2: Chain to dL/dS = grad_logits / T  (since logits = S / T).
    Tensor grad_S(twoN, twoN);
    double inv_T = 1.0 / temperature_;
    for (size_t i = 0; i < twoN; ++i)
        for (size_t j = 0; j < twoN; ++j)
            grad_S[i][j] = grad_logits[i][j] * inv_T;

    // Step 3: Chain through similarity to grad_z.
    Tensor grad_z(twoN, D);
    grad_z.fill(0.0);

    // Use the cached normalized input last_normalized_ for the cosine path.
    // For dot-product path, last_normalized_ == z (no normalization applied).
    const Tensor& z_eff = last_normalized_;

    if (!normalize_) {
        // Dot product: ds_ij/dz_i = z_j, ds_ij/dz_j = z_i
        // grad_z[i] = sum_j grad_S[i][j] * z_j + sum_j grad_S[j][i] * z_j
        //                     ^-- ds_ij/dz_i for varying j   ^-- ds_ji/dz_i for varying j
        // Both terms share z_j, so:
        // grad_z[i] = sum_j (grad_S[i][j] + grad_S[j][i]) * z_j
        for (size_t i = 0; i < twoN; ++i) {
            for (size_t j = 0; j < twoN; ++j) {
                double g = grad_S[i][j] + grad_S[j][i];
                for (size_t k = 0; k < D; ++k) {
                    grad_z[i][k] += g * z_eff[j][k];
                }
            }
        }
    } else {
        // Cosine similarity on L2-normalized inputs.
        // Let u_i = z_norm[i] (cached), s_ij = u_i . u_j.
        // Chain through inner product: ds_ij/d u_i[k] = u_j[k], ds_ij/d u_j[k] = u_i[k].
        // So:
        //   grad_u[i] = sum_j grad_S[i][j] * u_j + sum_j grad_S[j][i] * u_j
        //            = sum_j (grad_S[i][j] + grad_S[j][i]) * u_j
        //
        // Then chain through L2 norm: u_i = z_i / ||z_i||.
        //   du_i[k]/dz_i[k] = (1/||z_i||) - z_i[k] * z_i[k] / ||z_i||^3
        //                    = (1/||z_i||) * (1 - u_i[k]^2)
        //   du_i[k]/dz_i[m] (k != m) = - z_i[k] * z_i[m] / ||z_i||^3
        //                            = - u_i[k] * u_i[m] / ||z_i||
        //
        // In matrix form:
        //   grad_z[i] = (grad_u[i] - u_i * <grad_u[i], u_i>) / ||z_i||
        //
        // We need ||z_i|| — recover from ||u_i|| = 1 (after normalization):
        //   ||z_i|| = ||u_i|| * ||z_i|| / ||u_i||... actually simpler to
        //   re-derive ||z_i|| from the input z.

        Tensor grad_u(twoN, D);
        grad_u.fill(0.0);
        for (size_t i = 0; i < twoN; ++i) {
            for (size_t j = 0; j < twoN; ++j) {
                double g = grad_S[i][j] + grad_S[j][i];
                for (size_t k = 0; k < D; ++k) {
                    grad_u[i][k] += g * z_eff[j][k];
                }
            }
        }

        for (size_t i = 0; i < twoN; ++i) {
            // Compute ||z_i|| from the input z (z is the user's input).
            double znorm_sq = 0.0;
            for (size_t k = 0; k < D; ++k) znorm_sq += z[i][k] * z[i][k];
            double znorm = std::sqrt(znorm_sq);
            if (znorm < eps_) {
                // Degenerate: leave grad zero (matches forward's zero output).
                continue;
            }
            // Compute <grad_u[i], u_i>
            double dot_gu_u = 0.0;
            for (size_t k = 0; k < D; ++k) {
                dot_gu_u += grad_u[i][k] * z_eff[i][k];
            }
            double inv_znorm = 1.0 / znorm;
            for (size_t k = 0; k < D; ++k) {
                grad_z[i][k] = (grad_u[i][k] - z_eff[i][k] * dot_gu_u) * inv_znorm;
            }
        }
    }

    return grad_z;
}