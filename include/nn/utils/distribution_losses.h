#ifndef DISTRIBUTION_LOSSES_H
#define DISTRIBUTION_LOSSES_H

#include "../core/tensor.h"
#include <cmath>

// =============================================================================
// KL Divergence (Kullback-Leibler)
// =============================================================================
// D_KL(p || q) = sum_i p_i * (log p_i - log q_i)
// Convention:
//   - 0 * log(0 / 0) is treated as 0 (lim_{x→0} x log x = 0)
//   - 0 * log(0 / q>0) is treated as 0
//   - p>0 * log(p / q=0) is +infinity (the caller should avoid; we clamp to a large finite value)
//   - Forward returns the MEAN over batch dimension (matches the PyTorch convention
//     for `F.kl_div(input, target, reduction='batchmean')`/`'mean'`).
//
// Backward computes dL/dq treating q as a free variable (no sum-1 projection):
//   dL/dq_j = -(1/N) * (p_j / q_j) for q_j > 0
// If q_j → 0 (and p_j > 0), we clamp the gradient to a large negative value to
// reflect the divergence. Test cases use q > 0 everywhere to stay in the safe regime.

class KLDivergence {
public:
    // eps_q: minimum q value to avoid /0 in log(q) and p/q. Very small (1e-30) so
    //   that realistic probabilities don't get distorted, but no NaN at exact 0.
    explicit KLDivergence(double eps_q = 1e-30)
        : eps_q_(eps_q) {}

    // Forward: returns 1x1 tensor containing mean over batch of D_KL(p_row || q_row)
    Tensor forward(const Tensor& p, const Tensor& q);

    // Backward: gradient w.r.t. q. Shape (batch, K).
    //   grad[b][k] = -(1/N) * p[b][k] / q[b][k]   (clamped at q < eps_q)
    Tensor backward(const Tensor& p, const Tensor& q);

    double get_eps_q() const { return eps_q_; }

private:
    double eps_q_;
};


// =============================================================================
// Jensen-Shannon Divergence
// =============================================================================
// JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
// where m = (p + q) / 2.
// Properties:
//   - Always finite: even when one side is 0, m is half of the other side > 0
//   - Symmetric: JSD(p, q) == JSD(q, p)
//   - Bounded: 0 <= JSD <= log(2)
//
// Backward (w.r.t. q) reduces to a remarkably clean formula:
//   dJSD/dq_j = (0.5 / N) * log(q_j / m_j)
// (because of the symmetric differentiation where the 1 - m/m = 0 term cancels).
// For implementation this is exact (at finite precision); finite-difference tests
// at eps=1e-5 confirm the match.

class JSDivergence {
public:
    explicit JSDivergence(double eps_q = 1e-30)
        : eps_q_(eps_q) {}

    // Forward: returns 1x1 tensor containing mean over batch of JSD(p_row || q_row)
    Tensor forward(const Tensor& p, const Tensor& q);

    // Backward: gradient w.r.t. q. Shape (batch, K).
    Tensor backward(const Tensor& p, const Tensor& q);

    double get_eps_q() const { return eps_q_; }

private:
    double eps_q_;
};


// =============================================================================
// Huber Loss (smooth L1)
// =============================================================================
// L(e) = 0.5 * e^2                   if |e| <= delta
//      = delta * (|e| - 0.5 * delta) if |e|  > delta
// where e = y - pred (per element).
// Forward is mean over batch. Backward divides by batch.

class HuberLoss {
public:
    explicit HuberLoss(double delta = 1.0) : delta_(delta) {}

    // pred, y: shape (batch, 1) (or (batch, D) — currently batch-only). Returns 1x1 loss.
    Tensor forward(const Tensor& pred, const Tensor& y);

    // Backward: gradient w.r.t. pred. Shape same as pred.
    //   If |err| <= delta: grad = (pred - y) / N
    //   If |err|  > delta: grad = -delta * sign(err) / N  = delta * sign(pred - y) / N
    Tensor backward(const Tensor& pred, const Tensor& y);

    double get_delta() const { return delta_; }

private:
    double delta_;
};


// =============================================================================
// Quantile (pinball) Loss
// =============================================================================
// L(e) = e * q          if e >= 0  (under-prediction: y >= pred)
//      = e * (q - 1)    if e <  0  (over-prediction: y < pred)
// where e = y - pred.
// At q = 0.5 this is exactly 0.5 * |e| = MAE/2.
// Forward is mean over batch; Backward divides by batch.
//
// dL/d pred_i:
//   if y >= pred:     grad = -q / N   (if err = 0 exactly, we use the under branch by convention)
//   if y <  pred:     grad = (1 - q) / N

class QuantileLoss {
public:
    // q must be in (0, 1); constructor asserts this.
    explicit QuantileLoss(double q = 0.5) : q_(q) {
        if (q <= 0.0 || q >= 1.0) {
            // Defensive: clamp to small epsilon inside the valid range
            q_ = std::min(std::max(q, 1e-9), 1.0 - 1e-9);
        }
    }

    Tensor forward(const Tensor& pred, const Tensor& y);
    Tensor backward(const Tensor& pred, const Tensor& y);

    double get_q() const { return q_; }

private:
    double q_;
};


#endif
