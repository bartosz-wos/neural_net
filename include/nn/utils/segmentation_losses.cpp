#include "segmentation_losses.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

// Helper: per-row numerator / denominator for the (symmetric-eps) Dice / Tversky
// form. `target` is a (rows, cols) tensor. We treat (rows, cols) as a flat
// (N, C) layout in the loss-side formulas.
inline void dice_tversky_accumulate(
    const Tensor& pred, const Tensor& target, size_t b,
    double alpha, double beta,
    double& num, double& den) {
    num = 0.0;
    den = 0.0;
    const size_t C = pred.cols;
    for (size_t c = 0; c < C; ++c) {
        num += pred[b][c] * target[b][c];
        den += pred[b][c] * target[b][c]
             + alpha * pred[b][c] * (1.0 - target[b][c])
             + beta  * (1.0 - pred[b][c]) * target[b][c];
    }
}

// Sort indices of `errors` in DESCENDING order by value.
// `errors` is replaced in-place with the sorted values; `order` gets the
// permutation (such that sorted[i] = errors[order[i]] in the original layout).
void sort_desc(std::vector<double>& errors, std::vector<size_t>& order) {
    const size_t C = errors.size();
    struct Pair { double v; size_t i; };
    std::vector<Pair> pairs(C);
    for (size_t i = 0; i < C; ++i) pairs[i] = {errors[i], i};
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.v > b.v; });
    order.resize(C);
    std::vector<double> sorted(C);
    for (size_t i = 0; i < C; ++i) {
        sorted[i] = pairs[i].v;
        order[i] = pairs[i].i;
    }
    errors.swap(sorted);
}

} // namespace


// =============================================================================
// DiceLoss
// =============================================================================

Tensor DiceLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("DiceLoss::forward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        // Canonical Dice: D = (2*inter + eps) / (|P| + |T| + eps).
        double inter = 0.0, sum_p = 0.0, sum_t = 0.0;
        for (size_t c = 0; c < C; ++c) {
            inter += pred[b][c] * target[b][c];
            sum_p += pred[b][c];
            sum_t += target[b][c];
        }
        total += 1.0 - (2.0 * inter + eps_) / (sum_p + sum_t + eps_);
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor DiceLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("DiceLoss::backward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        double inter = 0.0, sum_p = 0.0, sum_t = 0.0;
        for (size_t c = 0; c < C; ++c) {
            inter += pred[b][c] * target[b][c];
            sum_p += pred[b][c];
            sum_t += target[b][c];
        }
        // U = 2*inter + eps, V = |P| + |T| + eps.
        // dU/dp[bc] = 2 t[bc], dV/dp[bc] = 1
        // d(1 - U/V)/dp = (U*1 - V*2t) / V^2
        const double U = 2.0 * inter + eps_;
        const double V = sum_p + sum_t + eps_;
        for (size_t c = 0; c < C; ++c) {
            double t = target[b][c];
            grad[b][c] = (U - 2.0 * V * t) / (V * V * static_cast<double>(N));
        }
    }
    return grad;
}


// =============================================================================
// TverskyLoss
// =============================================================================

Tensor TverskyLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("TverskyLoss::forward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        dice_tversky_accumulate(pred, target, b, alpha_, beta_, num, den);
        total += 1.0 - (num + eps_) / (den + eps_);
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor TverskyLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("TverskyLoss::backward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        dice_tversky_accumulate(pred, target, b, alpha_, beta_, num, den);
        const double U = num + eps_;
        const double V = den + eps_;
        // dU/dp[bc] = t[bc]
        // dV/dp[bc] = t[bc] + α (1 - t[bc]) - β t[bc] = α + t[bc] (1 - α - β)
        // d(1 - U/V)/dp = (U * dV/dp - V * dU/dp) / V^2
        for (size_t c = 0; c < C; ++c) {
            double t = target[b][c];
            double dU = t;
            double dV = alpha_ + t * (1.0 - alpha_ - beta_);
            grad[b][c] = (U * dV - V * dU) / (V * V * static_cast<double>(N));
        }
    }
    return grad;
}


// =============================================================================
// FocalDiceLoss
// =============================================================================

Tensor FocalDiceLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("FocalDiceLoss::forward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        // Per-row Tversky Index (depends only on b) — sums over n in row b.
        double num = 0.0, den = 0.0;
        dice_tversky_accumulate(pred, target, b, alpha_, beta_, num, den);
        double ti = (num + eps_) / (den + eps_);
        double d_loss_per_row = 1.0 - ti;
        // Modulator S_b = Σ_c (1 - p_t(b,c))^γ — per-cell focusing.
        double S = 0.0;
        for (size_t c = 0; c < C; ++c) {
            double t = target[b][c];
            double p_t = (t >= 0.5) ? pred[b][c] : (1.0 - pred[b][c]);
            S += std::pow(std::max(0.0, 1.0 - p_t), gamma_);
        }
        total += d_loss_per_row * S;
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor FocalDiceLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("FocalDiceLoss::backward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        // Per-row quantities: U, V for the Tversky denominator, plus S = sum_c mod(b,c).
        // Forward: L = (1/N) * Σ_b D_b * S_b  where D_b = 1 - U/V (depends on b only,
        //                   sums over n in row b) and m_bc = (1 - p_t(b,c))^γ.
        double num = 0.0, den = 0.0;
        dice_tversky_accumulate(pred, target, b, alpha_, beta_, num, den);
        const double U = num + eps_;
        const double V = den + eps_;
        const double D = 1.0 - U / V;

        // Σ_c m(b,c)
        double S = 0.0;
        for (size_t c = 0; c < C; ++c) {
            double t = target[b][c];
            double p_t = (t >= 0.5) ? pred[b][c] : (1.0 - pred[b][c]);
            S += std::pow(std::max(0.0, 1.0 - p_t), gamma_);
        }

        for (size_t c = 0; c < C; ++c) {
            // d(D_b)/dp[bc]:
            //   dU/dp[bc] = t[bc]
            //   dV/dp[bc] = α + t[bc] (1 - α - β)
            //   d(D_b)/dp = (U*dV - V*dU)/V^2
            double t = target[b][c];
            double dU = t;
            double dV = alpha_ + t * (1.0 - alpha_ - beta_);
            double dD_dp = (U * dV - V * dU) / (V * V);

            // d(m_bc)/dp[bc]:
            //   When t=1: m = (1 - p)^γ, dm/dp = -γ (1 - p)^(γ-1)
            //   When t=0: m = p^γ,        dm/dp = +γ p^(γ-1)
            double dm_dp;
            if (t >= 0.5) {
                double one_m_p = 1.0 - pred[b][c];
                double base = std::max(1e-30, one_m_p);
                dm_dp = -gamma_ * std::pow(base, gamma_ - 1.0);
            } else {
                double p_v = pred[b][c];
                double base = std::max(1e-30, p_v);
                dm_dp = gamma_ * std::pow(base, gamma_ - 1.0);
            }

            // Chain rule: dL/dp[bc] = (1/N) * [dD_dp * S + D * dm_dp]
            double d_loss_dp = dD_dp * S + D * dm_dp;
            grad[b][c] = d_loss_dp / static_cast<double>(N);
        }
    }
    return grad;
}


// =============================================================================
// Lovász-Hinge Loss
// =============================================================================

Tensor LovaszHingeLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("LovaszHingeLoss::forward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        std::vector<double> errors(C);
        std::vector<double> gt(C);
        for (size_t c = 0; c < C; ++c) {
            errors[c] = (target[b][c] >= 0.5) ? (1.0 - pred[b][c]) : pred[b][c];
            gt[c] = target[b][c];
        }
        std::vector<size_t> order;
        sort_desc(errors, order);
        std::vector<double> gt_sorted(C);
        for (size_t c = 0; c < C; ++c) gt_sorted[c] = gt[order[c]];

        std::vector<double> cum_pos(C, 0.0), cum_neg(C, 0.0);
        double sp = 0.0, sn = 0.0;
        for (size_t i = 0; i < C; ++i) {
            sp += gt_sorted[i];
            sn += 1.0 - gt_sorted[i];
            cum_pos[i] = sp;
            cum_neg[i] = sn;
        }
        std::vector<double> jd(C, 0.0);
        for (size_t i = 0; i < C; ++i) {
            double inter = cum_pos[i] - gt_sorted[i];
            double union_ = cum_pos[i] + cum_neg[i];
            jd[i] = (union_ > 0.0) ? (1.0 - inter / union_) : 0.0;
        }
        std::vector<double> lg(C, 0.0);
        lg[0] = jd[0];
        for (size_t i = 1; i < C; ++i) lg[i] = jd[i] - jd[i - 1];

        double row_loss = 0.0;
        for (size_t i = 0; i < C; ++i) row_loss += errors[i] * lg[i];
        total += row_loss / static_cast<double>(C);
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor LovaszHingeLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("LovaszHingeLoss::backward: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        std::vector<double> errors(C);
        for (size_t c = 0; c < C; ++c) {
            errors[c] = (target[b][c] >= 0.5) ? (1.0 - pred[b][c]) : pred[b][c];
        }
        std::vector<size_t> order;
        sort_desc(errors, order);

        std::vector<double> gt_sorted(C);
        for (size_t c = 0; c < C; ++c) gt_sorted[c] = target[b][order[c]];

        std::vector<double> cum_pos(C, 0.0), cum_neg(C, 0.0);
        double sp = 0.0, sn = 0.0;
        for (size_t i = 0; i < C; ++i) {
            sp += gt_sorted[i];
            sn += 1.0 - gt_sorted[i];
            cum_pos[i] = sp;
            cum_neg[i] = sn;
        }
        std::vector<double> jd(C, 0.0);
        for (size_t i = 0; i < C; ++i) {
            double inter = cum_pos[i] - gt_sorted[i];
            double union_ = cum_pos[i] + cum_neg[i];
            jd[i] = (union_ > 0.0) ? (1.0 - inter / union_) : 0.0;
        }
        std::vector<double> lg(C, 0.0);
        lg[0] = jd[0];
        for (size_t i = 1; i < C; ++i) lg[i] = jd[i] - jd[i - 1];

        // d(row_loss)/dp[bc]:
        //   errors[c] = (1 - p[c]) if t[c]=1 else p[c]
        //   d(errors)/dp[c] = -1 if t[c]=1, +1 if t[c]=0 ⇒ (1 - 2 t[c])
        //   But errors runs in SORTED order; the slot for column `c` in the
        //   sorted vector is `order[i]==c`. So:
        //   d(row_loss)/dp[bc] = (1 - 2 t[bc]) * lg[rank(b,c)]
        //   Divide by (C * N) for the (mean over C) * (mean over N) formula.
        for (size_t c = 0; c < C; ++c) {
            size_t rank = 0;
            for (size_t i = 0; i < C; ++i) if (order[i] == c) { rank = i; break; }
            double t = target[b][c];
            grad[b][c] = (1.0 - 2.0 * t) * lg[rank]
                       / static_cast<double>(C * N);
        }
    }
    return grad;
}
