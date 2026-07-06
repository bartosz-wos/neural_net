#include "distribution_losses.h"

#include <cmath>
#include <algorithm>


// =============================================================================
// KLDivergence
// =============================================================================
//
// Forward: returns mean over batch of KL(p_row || q_row).
//   KL(p || q) = sum_k p_k * (log(p_k) - log(q_k))
//   Per-element rules:
//     - p_k == 0            -> contribution is 0  (0 * log(...) = 0)
//     - p_k  > 0, q_k >= eps -> use log(p_k / q_k)
//     - p_k  > 0, q_k <  eps -> contribution is log(p_k / eps_q_)  (very large positive)
//
// Backward (w.r.t. q):
//   d/dq_k  [(1/N) * p_k * log(p_k / q_k)] = -(1/N) * p_k / q_k
// So grad[b][k] = -(1/N) * p[b][k] / max(q[b][k], eps_q_)   for q > eps
//                = -(1/N) * p[b][k] / eps_q_                 for q < eps (clamped, large negative)

Tensor KLDivergence::forward(const Tensor& p, const Tensor& q) {
    size_t N = p.rows;
    size_t K = p.cols;

    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        double row_sum = 0.0;
        for (size_t k = 0; k < K; ++k) {
            double pk = p[b][k];
            if (pk == 0.0) {
                // 0 * log(0 / anything) = 0  — by limit
                continue;
            }
            double qk = q[b][k];
            if (qk < eps_q_) {
                qk = eps_q_;
            }
            row_sum += pk * std::log(pk / qk);
        }
        total += row_sum;
    }
    double mean_loss = total / static_cast<double>(N);

    Tensor out(1, 1);
    out[0][0] = mean_loss;
    return out;
}

Tensor KLDivergence::backward(const Tensor& p, const Tensor& q) {
    size_t N = p.rows;
    size_t K = p.cols;
    double inv_N = 1.0 / static_cast<double>(N);

    Tensor grad(N, K);
    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pk = p[b][k];
            double qk = q[b][k];
            if (qk < eps_q_) qk = eps_q_;
            // dKL/dq_k = -p_k / q_k, then divide by N
            grad[b][k] = -pk / qk * inv_N;
        }
    }
    return grad;
}


// =============================================================================
// JSDivergence
// =============================================================================
//
// Forward:
//   JSD(p, q) = 0.5 * (KL(p || m) + KL(q || m))    where m = (p + q) / 2
//   No 0/0 traps because even if one of p_i or q_i is 0, m_i is the other / 2 > 0
//   for any well-formed input. We still clamp m at eps_q_ to guard against both
//   being exactly 0.
//
// Backward (w.r.t. q):
//   Let m_k = (p_k + q_k) / 2.
//   JSD = 0.5 * sum_i  [ p_i * (log(p_i) - log(m_i))  +  q_i * (log(q_i) - log(m_i)) ]   / N
//
//   dJSD/dq_k:
//     = 0.5 * /N * [ -p_k / m_k * 0.5       // from KL(p||m), m_k derivative w.r.t. q_k
//                  + (1 + log(q_k) - log(m_k) - q_k/(2*m_k)) ]   // from KL(q||m)
//     = 0.5 / N * [ -0.5*p_k/m_k + log(q_k/m_k) + 1 - 0.5*q_k/m_k ]
//     = 0.5 / N * [ log(q_k/m_k) + 1 - 0.5*(p_k+q_k)/m_k ]
//     = 0.5 / N * [ log(q_k/m_k) + 1 - 1 ]                  (because m_k = (p_k+q_k)/2)
//     = 0.5 / N * log(q_k / m_k)
//
//   Edge case at q_k = 0:  we use the limit, log(0 / m_k) -> -inf, large negative
//   grad (which is correct: decreasing q drives JSD up). We clamp m at eps_q_ to
//   avoid log(0).

Tensor JSDivergence::forward(const Tensor& p, const Tensor& q) {
    size_t N = p.rows;
    size_t K = p.cols;

    auto kl_pq = [&](double pk, double mk) -> double {
        if (pk == 0.0) return 0.0;
        if (mk < eps_q_) mk = eps_q_;
        return pk * std::log(pk / mk);
    };
    auto kl_qq = [&](double qk, double mk) -> double {
        if (qk == 0.0) return 0.0;
        if (mk < eps_q_) mk = eps_q_;
        return qk * std::log(qk / mk);
    };

    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        double row_sum = 0.0;
        for (size_t k = 0; k < K; ++k) {
            double pk = p[b][k];
            double qk = q[b][k];
            double mk = 0.5 * (pk + qk);
            row_sum += 0.5 * (kl_pq(pk, mk) + kl_qq(qk, mk));
        }
        total += row_sum;
    }
    double mean_loss = total / static_cast<double>(N);

    Tensor out(1, 1);
    out[0][0] = mean_loss;
    return out;
}

Tensor JSDivergence::backward(const Tensor& p, const Tensor& q) {
    size_t N = p.rows;
    size_t K = p.cols;
    double inv_N = 1.0 / static_cast<double>(N);
    double half_inv_N = 0.5 * inv_N;

    Tensor grad(N, K);
    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pk = p[b][k];
            double qk = q[b][k];
            double mk = 0.5 * (pk + qk);
            if (mk < eps_q_) mk = eps_q_;
            // dJSD/dq_k = (0.5 / N) * log(q_k / m_k)
            // When qk == 0: in the limit log(0 / m_k) = -inf. Use -inf finite surrogate:
            //   log(eps_q_ / m_k) — large negative — to keep the grad finite.
            double qk_eff = (qk < eps_q_) ? eps_q_ : qk;
            grad[b][k] = half_inv_N * std::log(qk_eff / mk);
        }
    }
    return grad;
}


// =============================================================================
// HuberLoss
// =============================================================================
//
// Forward (mean over batch):
//   L_b = 0.5 * (y_b - pred_b)^2                  if |err_b| <= delta
//      =  delta * (|err_b| - 0.5 * delta)         if |err_b|  > delta
//   L = (1/N) * sum_b L_b
// (Note: continuous at |err|=delta — both formulas give 0.5 * delta^2.)
//
// Backward (w.r.t. pred):
//   dL/dpred_b = -(1/N) * dL/de_b  where e = y - pred
//   If |err| <= delta:   dL/de_b = e_b     -> grad = -e_b/N = (pred_b - y_b)/N
//   If |err|  > delta:   dL/de_b = delta*sign(e_b)  -> grad = -delta*sign(e_b)/N

Tensor HuberLoss::forward(const Tensor& pred, const Tensor& y) {
    size_t N = pred.rows;
    size_t K = pred.cols;
    double total = 0.0;

    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pb = pred[b][k];
            double yb = y[b][k];
            double err = yb - pb;
            double ae = std::abs(err);
            double row_term;
            if (ae <= delta_) {
                row_term = 0.5 * err * err;
            } else {
                row_term = delta_ * (ae - 0.5 * delta_);
            }
            total += row_term;
        }
    }
    Tensor out(1, 1);
    out[0][0] = total / static_cast<double>(N);
    return out;
}

Tensor HuberLoss::backward(const Tensor& pred, const Tensor& y) {
    size_t N = pred.rows;
    size_t K = pred.cols;
    double inv_N = 1.0 / static_cast<double>(N);

    Tensor grad(N, K);
    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pb = pred[b][k];
            double yb = y[b][k];
            double err = yb - pb;
            double ae = std::abs(err);
            if (ae <= delta_) {
                // quadratic regime: dL/de = err, dL/dpred = -err/N = (pred - y)/N
                grad[b][k] = (pb - yb) * inv_N;
            } else {
                // linear regime: dL/de = delta*sign(err), dL/dpred = -delta*sign(err)/N
                // Equivalent: (delta/N) when pred < y, (-delta/N) when pred > y
                // Use sign of (pb - yb) for clean branching:
                double sign_pred_minus_y = (pb > yb) ? 1.0 : (pb < yb ? -1.0 : 0.0);
                grad[b][k] = delta_ * inv_N * sign_pred_minus_y;
            }
        }
    }
    return grad;
}


// =============================================================================
// QuantileLoss
// =============================================================================
//
// Forward (mean over batch):
//   e_b = y_b - pred_b
//   L_b = e_b * q           if e_b >= 0  (under-pred)
//      = e_b * (q - 1)      if e_b <  0  (over-pred)
//   L = (1/N) * sum_b L_b
//
// Backward (w.r.t. pred):
//   dL/dpred_b = -(1/N) * dL/de_b where:
//     if e_b >= 0:  dL/de_b = q          -> grad = -q/N
//     if e_b <  0:  dL/de_b = (q - 1)    -> grad = (1 - q)/N
//   We use the under-prediction branch when e == 0 exactly (this is arbitrary but
//   the test asserts -q/N).

Tensor QuantileLoss::forward(const Tensor& pred, const Tensor& y) {
    size_t N = pred.rows;
    size_t K = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pb = pred[b][k];
            double yb = y[b][k];
            double e = yb - pb;
            if (e >= 0.0) {
                total += e * q_;
            } else {
                total += e * (q_ - 1.0);
            }
        }
    }
    Tensor out(1, 1);
    out[0][0] = total / static_cast<double>(N);
    return out;
}

Tensor QuantileLoss::backward(const Tensor& pred, const Tensor& y) {
    size_t N = pred.rows;
    size_t K = pred.cols;
    double inv_N = 1.0 / static_cast<double>(N);
    Tensor grad(N, K);
    for (size_t b = 0; b < N; ++b) {
        for (size_t k = 0; k < K; ++k) {
            double pb = pred[b][k];
            double yb = y[b][k];
            double e = yb - pb;
            if (e >= 0.0) {
                // under-prediction (including exactly equal): grad = -q/N
                grad[b][k] = -q_ * inv_N;
            } else {
                // over-prediction: grad = (1 - q)/N
                grad[b][k] = (1.0 - q_) * inv_N;
            }
        }
    }
    return grad;
}
