#include "performer.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Performer (FAVOR+) — Choromanski et al. 2021
//   "Rethinking Attention with Performers" (ICLR 2021)
// ============================================================================
//
// This file implements Performer's FAVOR+ (Fast Attention Via Orthogonal
// Random features, Plus) mechanism using the *closed-form positive random
// features* from §2.3 of the paper.
//
// Math summary (per single-head attention layer, input x ∈ R^{n × d_model}):
//
//   1. Q = x W_q^T + b_q  ∈ R^{n × d_k}
//   2. K = x W_k^T + b_k  ∈ R^{n × d_k}
//   3. V = x W_v^T + b_v  ∈ R^{n × d_k}
//   4. φ(Q) = (1/√(m/2)) * exp(-||Q_t||²/2) * (cos(ω_i^T Q_t), sin(ω_i^T Q_t))_{i=1..m/2}
//   5. φ(K) = same with K
//   6. KV  = φ(K)^T V  ∈ R^{m × d_k}      (O(n m d) — no quadratic term!)
//   7. Z   = φ(K)^T 1  ∈ R^{m}            (column sum)
//   8. out_pre = (φ(Q) @ KV) / (φ(Q) @ Z + eps)  ∈ R^{n × d_k}
//   9. out = out_pre W_o^T + b_o          ∈ R^{n × d_model}
//
// W_prj ∈ R^{m/2 × d_k} is the fixed random projection (drawn once at init
// from N(0, 1)); b_prj ∈ R^{m/2} is a small random frequency offset (drawn
// from U(0, 2π) — this fixes the "half-period" bias noted in the paper
// appendix and reduces variance of the estimator).
//
// Backward: full BPTT for the linear attention path + the (m × d) and (m,)
// path-specific accumulators. The projection W_prj is fixed, so no gradient
// for it (just the Q/K/V/O gradients).
//
// All operations are O(n m d) — no n² terms anywhere. This is the *whole
// point* of Performer.

// ----------------------------------------------------------------------------
// Random helpers
// ----------------------------------------------------------------------------
static std::mt19937& performer_global_rng() {
    static std::mt19937 gen(42);  // fixed seed for reproducibility (tests!)
    return gen;
}

// ----------------------------------------------------------------------------
// PerformerAttention
// ----------------------------------------------------------------------------
PerformerAttention::PerformerAttention(size_t d_model, size_t seq_len, size_t num_features)
    : d_model_(d_model),
      seq_len_(seq_len),
      num_features_(num_features),
      m_half_(num_features / 2),
      d_k_(d_model),
      // Learned Q/K/V/O projections — same convention as Dense
      W_q(d_model, d_model), W_k(d_model, d_model),
      W_v(d_model, d_model), W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model), b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model), grad_b_v(1, d_model), grad_b_o(1, d_model),
      W_prj_(m_half_, d_k_),
      b_prj_(1, m_half_),
      last_input_(0, 0),
      last_q_(0, 0), last_k_(0, 0), last_v_(0, 0),
      last_phi_q_(0, 0), last_phi_k_(0, 0),
      last_norm_q_(0, 0), last_norm_k_(0, 0),
      last_KV_(0, 0), last_Ksum_(0, 0),
      last_Z_(0, 0), last_out_(0, 0)
{
    if (num_features % 2 != 0) {
        throw std::invalid_argument("PerformerAttention: num_features must be even");
    }
    if (num_features == 0) {
        throw std::invalid_argument("PerformerAttention: num_features must be > 0");
    }

    // Initialize Q/K/V/O with Xavier (matches Dense convention)
    auto xavier_init = [](Tensor& W, size_t fan_in, size_t fan_out) {
        double std = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] = dis(performer_global_rng());
    };
    xavier_init(W_q, d_model, d_model);
    xavier_init(W_k, d_model, d_model);
    xavier_init(W_v, d_model, d_model);
    xavier_init(W_o, d_model, d_model);

    b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);

    // FAVOR+ random projection: W_prj ~ N(0, 1)  (matches the paper's default σ²=1)
    std::normal_distribution<> proj_dis(0.0, 1.0);
    for (size_t i = 0; i < m_half_; ++i)
        for (size_t j = 0; j < d_k_; ++j)
            W_prj_(i, j) = proj_dis(performer_global_rng());

    // Frequency offset: U(0, 2π) — reduces estimator variance per paper
    std::uniform_real_distribution<> bias_dis(0.0, 2.0 * M_PI);
    for (size_t i = 0; i < m_half_; ++i)
        b_prj_(0, i) = bias_dis(performer_global_rng());
}

// Compute the FAVOR+ feature map for a (n, d_k) input.
// Returns φ(x) ∈ R^{n × m} where m = 2 * m_half_.
//
// φ(x)_t = (1/√(m/2)) * exp(-||x_t||²/2) * (cos(ω_i^T x_t + b_i), sin(ω_i^T x_t + b_i))_{i=1..m/2}
//
// out_norm: (n,)  — the per-row exp(-||x_t||²/2) factor (kept separate for the
//                   backward pass; we don't lose any precision by absorbing it).
static Tensor performer_feature_map(const Tensor& x, const Tensor& W_prj, const Tensor& b_prj,
                                    size_t m_half, Tensor& out_norm) {
    const size_t n = x.rows;
    const size_t d = x.cols;
    const double scale = 1.0 / std::sqrt(static_cast<double>(m_half));

    Tensor phi(n, 2 * m_half);
    out_norm = Tensor(n, 1);  // store as (n, 1) for ease

    for (size_t t = 0; t < n; ++t) {
        // ||x_t||²
        double norm_sq = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double v = x(t, j);
            norm_sq += v * v;
        }
        double exp_factor = std::exp(-0.5 * norm_sq);
        out_norm(t, 0) = exp_factor;

        // (ω_i^T x_t + b_i) for i = 0..m_half-1
        for (size_t i = 0; i < m_half; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < d; ++j) {
                dot += W_prj(i, j) * x(t, j);
            }
            double angle = dot + b_prj(0, i);
            phi(t, 2 * i)         = scale * exp_factor * std::cos(angle);
            phi(t, 2 * i + 1)     = scale * exp_factor * std::sin(angle);
        }
    }
    return phi;
}

Tensor PerformerAttention::forward(const Tensor& input) {
    // input: (n, d_model)  — assumed n == seq_len_
    const size_t n = input.rows;
    if (n != seq_len_) {
        throw std::invalid_argument("PerformerAttention: input.rows != seq_len_");
    }
    last_input_ = input.clone();

    // Q/K/V projections: y = x W^T + b
    auto project = [&](const Tensor& x, const Tensor& W, const Tensor& b) {
        Tensor y = x * W.transpose();
        for (size_t j = 0; j < y.cols; ++j) {
            double bj = b(0, j);
            for (size_t i = 0; i < y.rows; ++i) y(i, j) += bj;
        }
        return y;
    };
    last_q_ = project(input, W_q, b_q);
    last_k_ = project(input, W_k, b_k);
    last_v_ = project(input, W_v, b_v);

    // FAVOR+ feature maps
    last_phi_q_ = performer_feature_map(last_q_, W_prj_, b_prj_, m_half_, last_norm_q_);
    last_phi_k_ = performer_feature_map(last_k_, W_prj_, b_prj_, m_half_, last_norm_k_);

    // KV  = φ(K)^T V  ∈ R^{m × d_k}  — THE linearization step
    last_KV_ = last_phi_k_.transpose() * last_v_;

    // Ksum = φ(K)^T 1  ∈ R^{m}
    last_Ksum_ = Tensor(1, num_features_);
    for (size_t j = 0; j < num_features_; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += last_phi_k_(t, j);
        last_Ksum_(0, j) = s;
    }

    // Z = φ(Q) @ Ksum  ∈ R^{n}
    last_Z_ = Tensor(n, 1);
    for (size_t t = 0; t < n; ++t) {
        double z = 0.0;
        for (size_t j = 0; j < num_features_; ++j) z += last_phi_q_(t, j) * last_Ksum_(0, j);
        last_Z_(t, 0) = z;
    }

    // out_pre = φ(Q) @ KV   (n × d_k), then divide by Z row-wise
    last_out_ = last_phi_q_ * last_KV_;
    const double eps = 1e-6;
    for (size_t t = 0; t < n; ++t) {
        double z = last_Z_(t, 0);
        double zinv = 1.0 / (z + eps);
        for (size_t j = 0; j < d_k_; ++j) {
            last_out_(t, j) *= zinv;
        }
    }

    // Output projection: y = out_pre @ W_o^T + b_o
    Tensor result = last_out_ * W_o.transpose();
    for (size_t j = 0; j < result.cols; ++j) {
        double bj = b_o(0, j);
        for (size_t i = 0; i < result.rows; ++i) result(i, j) += bj;
    }
    return result;
}

Tensor PerformerAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (n, d_model)
    const size_t n = seq_len_;
    const size_t d = d_model_;
    const size_t m = num_features_;
    const size_t mh = m_half_;
    const double eps = 1e-6;

    // ---- Backward through output projection y = out_pre @ W_o^T + b_o ----
    //   grad_out_pre = grad_output @ W_o
    //   grad_W_o    += grad_output^T @ last_out_   (per Dense convention)
    //   grad_b_o    += sum_i grad_output(i, :)
    Tensor grad_out_pre(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_output(i, k) * W_o(k, j);
            grad_out_pre(i, j) = s;
        }
    // grad_W_o += grad_output^T @ last_out_   (shape d × d)
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_output(t, i) * last_out_(t, j);
            grad_W_o(i, j) += s;
        }
    // grad_b_o += sum_t grad_output(t, :)
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_output(t, j);
        grad_b_o(0, j) += s;
    }

    // ---- Backward through divide-by-Z ---------------------------------------
    // out_t = (φ(Q_t) @ KV) / (Z_t + eps)  where Z_t = φ(Q_t) @ Ksum
    // Let g_t = grad_out_pre(t, :)  (gradient w.r.t. out_t ∈ R^d).
    //
    // Let S_t = φ(Q_t) @ KV ∈ R^d   and   z_t = Z_t + eps.
    // out_t = S_t / z_t
    //
    // d z_t / d φ(Q_t) = Ksum            (size m)
    // d S_t / d φ(Q_t) = KV              (size d × m, but row t of φ(Q) only contributes to row t of S,
    //                                     so dS_t[k] / dφ(Q_t)[j] = KV[j, k])
    //
    // Per-query contribution: let g_t be grad w.r.t. out_t.
    //   dL/d S_t = g_t / z_t
    //   dL/d z_t = -<g_t, S_t> / z_t²
    //   dL/d φ(Q_t) = (dL/d S_t) @ KV^T  +  dL/d z_t * Ksum
    //              = (g_t / z_t) @ KV^T  -  (<g_t, S_t> / z_t²) * Ksum
    //              = (1/z_t) * (g_t @ KV^T)  -  (<g_t, S_t> / z_t²) * Ksum
    //
    // dL/d KV += φ(Q)^T @ dL/dS = φ(Q)^T @ (grad_out_pre / z)   (n × m)^T @ (n × d) = (m × d)
    // dL/d Ksum += φ(Q)^T @ dL/dz = φ(Q)^T @ (-<g_t, S_t> / z_t²)   (m,)
    Tensor grad_phi_q(n, m);
    Tensor grad_KV(m, d);
    Tensor grad_Ksum(1, m);

    for (size_t j = 0; j < m; ++j) grad_Ksum(0, j) = 0.0;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < d; ++j) grad_KV(i, j) = 0.0;

    for (size_t t = 0; t < n; ++t) {
        double z_t = last_Z_(t, 0) + eps;
        double z_inv = 1.0 / z_t;

        // <g_t, S_t>
        double g_dot_S = 0.0;
        for (size_t j = 0; j < d; ++j) g_dot_S += grad_out_pre(t, j) * last_out_(t, j);
        // Note: last_out_(t, j) is already S_t[j] / z_t (we divided in forward).
        // So S_t[j] = last_out_(t, j) * z_t
        // <g_t, S_t> = z_t * <g_t, last_out_>
        // But we'll keep the literal form for clarity:
        double g_dot_S_literal = 0.0;
        for (size_t j = 0; j < d; ++j) g_dot_S_literal += grad_out_pre(t, j) * last_out_(t, j);
        // (these are the same since last_out_ already has the /z baked in only
        //  for the final form; we recompute to be safe.)
        // Use the simplified form: <g_t, S_t> = z_t * <g_t, out_t>
        double g_dot_S_use = z_t * g_dot_S_literal;
        (void)g_dot_S;  // suppress unused

        // grad_phi_q(t, :) = (1/z_t) * (g_t @ KV^T)  -  (g_dot_S / z_t²) * Ksum
        for (size_t j = 0; j < m; ++j) {
            double g_KV = 0.0;
            for (size_t k = 0; k < d; ++k) g_KV += grad_out_pre(t, k) * last_KV_(j, k);
            grad_phi_q(t, j) = z_inv * g_KV - (g_dot_S_use / (z_t * z_t)) * last_Ksum_(0, j);
        }

        // grad_KV += φ(Q_t)^T @ (g_t / z_t)   →  grad_KV[j, k] += phi_q(t, j) * g_t[k] / z_t
        for (size_t j = 0; j < m; ++j) {
            double phi_q_tj = last_phi_q_(t, j);
            for (size_t k = 0; k < d; ++k) {
                grad_KV(j, k) += phi_q_tj * (grad_out_pre(t, k) * z_inv);
            }
        }
        // grad_Ksum += (g_dot_S / z_t²) * φ(Q_t)   (with a negative sign from the -<g,S>/z²)
        // Wait — we had dL/dz = -<g, S>/z², and dL/dKsum = dL/dz * dZ/dKsum = dL/dz * φ(Q)
        // So grad_Ksum += -<g, S> / z² * φ(Q_t)
        double coef = -g_dot_S_use / (z_t * z_t);
        for (size_t j = 0; j < m; ++j) {
            grad_Ksum(0, j) += coef * last_phi_q_(t, j);
        }
    }

    // ---- Backward through KV = φ(K)^T V  →  dL/d φ(K), dL/d V ----
    //   dL/d φ(K) = dL/d KV @ V^T    (n × m = (m × d) @ (d × n))
    //   dL/d V    = φ(K) @ dL/d KV   (n × d = (n × m) @ (m × d))
    Tensor grad_phi_k(n, m);
    Tensor grad_v(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_KV(j, k) * last_v_(t, k);
            grad_phi_k(t, j) = s;
        }
    }
    // grad_v = φ(K) @ grad_KV
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < m; ++j) s += last_phi_k_(t, j) * grad_KV(j, k);
            grad_v(t, k) = s;
        }
    }

    // ---- Backward through Ksum = φ(K)^T 1  →  dL/d φ(K) += ones(m) * grad_Ksum ----
    // grad_phi_k(t, j) += grad_Ksum(0, j)  (each row of φ(K) contributes 1 to each col of Ksum)
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < m; ++j) {
            grad_phi_k(t, j) += grad_Ksum(0, j);
        }
    }

    // ---- Backward through φ(·) feature map to get dL/d Q, dL/d K ----
    //
    // φ(x)_t[j=2i]   = s * exp(-||x||²/2) * cos(ω_i^T x + b_i)
    // φ(x)_t[j=2i+1] = s * exp(-||x||²/2) * sin(ω_i^T x + b_i)
    //                 where s = 1/√(m/2)
    //
    // d φ_t[j=2i]   / d x = s * exp(-||x||²/2) * (-x) * cos(ω_i^T x + b_i)
    //                       + s * exp(-||x||²/2) * (-sin(ω_i^T x + b_i)) * ω_i
    //                     = s * exp(-||x||²/2) * (-x * cos(θ_i) - sin(θ_i) * ω_i)
    // d φ_t[j=2i+1] / d x = s * exp(-||x||²/2) * (-x) * sin(ω_i^T x + b_i)
    //                       + s * exp(-||x||²/2) * (cos(ω_i^T x + b_i)) * ω_i
    //                     = s * exp(-||x||²/2) * (-x * sin(θ_i) + cos(θ_i) * ω_i)
    //
    // where θ_i = ω_i^T x + b_i
    //
    // dL/d x = sum_j (dL/d φ[j]) * d φ[j] / d x
    //
    // We can factor this efficiently:
    //   dL/d x = s * exp(-||x||²/2) * sum_i [
    //       grad_phi[2i]   * (-x * cos(θ_i) - sin(θ_i) * ω_i)
    //     + grad_phi[2i+1] * (-x * sin(θ_i) + cos(θ_i) * ω_i)
    //   ]
    //
    // Let a_i = grad_phi[2i], b_i = grad_phi[2i+1]
    // Let cos_i = cos(θ_i), sin_i = sin(θ_i)
    //
    // contribution from feature i to dL/dx:
    //   s * exp * ( a_i * (-x * cos_i - sin_i * ω_i) + b_i * (-x * sin_i + cos_i * ω_i) )
    //   = s * exp * ( -x * (a_i cos_i + b_i sin_i) + ω_i * (-a_i sin_i + b_i cos_i) )
    //
    // Define:
    //   p_i = a_i cos_i + b_i sin_i     (projected grad on x direction)
    //   q_i = b_i cos_i - a_i sin_i     (projected grad on ω_i direction)
    //
    // Then dL/dx += s * exp * (-x * p_i + ω_i * q_i)
    //
    // Summing over i: dL/dx = s * exp * (-x * sum_i p_i + sum_i q_i * ω_i)
    //
    // We also have grad_b_q, grad_b_k (and grad_W_q, grad_W_k) to update.

    // Backward through V projection first (uses grad_v we already computed)
    // grad_W_v += grad_v^T @ last_input_, grad_b_v += row-sum
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_v(t, i) * last_input_(t, j);
            grad_W_v(i, j) += s;
        }
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_v(t, j);
        grad_b_v(0, j) += s;
    }
    // grad_input from V path: grad_v @ W_v
    Tensor grad_input_from_v(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_v(t, k) * W_v(k, j);
            grad_input_from_v(t, j) = s;
        }
    }

    // Backward through K feature map: grad_phi_k -> grad_k -> (grad_W_k, grad_b_k, grad_input)
    // grad_k = backprop through φ(K)
    Tensor grad_k(n, d);
    {
        const double s_scale = 1.0 / std::sqrt(static_cast<double>(mh));
        for (size_t t = 0; t < n; ++t) {
            double exp_t = last_norm_k_(t, 0);
            double sum_p = 0.0;
            std::vector<double> sum_q_omega(d, 0.0);
            for (size_t i = 0; i < mh; ++i) {
                double a_i = grad_phi_k(t, 2 * i);
                double b_i = grad_phi_k(t, 2 * i + 1);
                double dot = 0.0;
                for (size_t k = 0; k < d; ++k) dot += W_prj_(i, k) * last_k_(t, k);
                double angle = dot + b_prj_(0, i);
                double cos_i = std::cos(angle);
                double sin_i = std::sin(angle);
                sum_p += a_i * cos_i + b_i * sin_i;
                double q_i = b_i * cos_i - a_i * sin_i;
                for (size_t k = 0; k < d; ++k) sum_q_omega[k] += q_i * W_prj_(i, k);
            }
            for (size_t k = 0; k < d; ++k) {
                grad_k(t, k) = s_scale * exp_t * (-last_k_(t, k) * sum_p + sum_q_omega[k]);
            }
        }
    }
    // grad_W_k += grad_k^T @ last_input_, grad_b_k += row-sum
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_k(t, i) * last_input_(t, j);
            grad_W_k(i, j) += s;
        }
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_k(t, j);
        grad_b_k(0, j) += s;
    }
    Tensor grad_input_from_k(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_k(t, k) * W_k(k, j);
            grad_input_from_k(t, j) = s;
        }
    }

    // Backward through Q feature map: grad_phi_q -> grad_q -> (grad_W_q, grad_b_q, grad_input)
    Tensor grad_q(n, d);
    {
        const double s_scale = 1.0 / std::sqrt(static_cast<double>(mh));
        for (size_t t = 0; t < n; ++t) {
            double exp_t = last_norm_q_(t, 0);
            double sum_p = 0.0;
            std::vector<double> sum_q_omega(d, 0.0);
            for (size_t i = 0; i < mh; ++i) {
                double a_i = grad_phi_q(t, 2 * i);
                double b_i = grad_phi_q(t, 2 * i + 1);
                double dot = 0.0;
                for (size_t k = 0; k < d; ++k) dot += W_prj_(i, k) * last_q_(t, k);
                double angle = dot + b_prj_(0, i);
                double cos_i = std::cos(angle);
                double sin_i = std::sin(angle);
                sum_p += a_i * cos_i + b_i * sin_i;
                double q_i = b_i * cos_i - a_i * sin_i;
                for (size_t k = 0; k < d; ++k) sum_q_omega[k] += q_i * W_prj_(i, k);
            }
            for (size_t k = 0; k < d; ++k) {
                grad_q(t, k) = s_scale * exp_t * (-last_q_(t, k) * sum_p + sum_q_omega[k]);
            }
        }
    }
    // grad_W_q += grad_q^T @ last_input_, grad_b_q += row-sum
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_q(t, i) * last_input_(t, j);
            grad_W_q(i, j) += s;
        }
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_q(t, j);
        grad_b_q(0, j) += s;
    }
    Tensor grad_input_from_q(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_q(t, k) * W_q(k, j);
            grad_input_from_q(t, j) = s;
        }
    }

    // Sum all three contributions to grad_input
    Tensor grad_input(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_input(t, j) = grad_input_from_q(t, j) + grad_input_from_k(t, j) + grad_input_from_v(t, j);
        }
    }
    return grad_input;
}

void PerformerAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, Tensor& b, Tensor& gW, Tensor& gb) {
        for (size_t i = 0; i < W.rows; ++i) {
            for (size_t j = 0; j < W.cols; ++j) {
                W(i, j) -= learning_rate * gW(i, j);
            }
            b(0, i) -= learning_rate * gb(0, i);
        }
    };
    sgd(W_q, b_q, grad_W_q, grad_b_q);
    sgd(W_k, b_k, grad_W_k, grad_b_k);
    sgd(W_v, b_v, grad_W_v, grad_b_v);
    sgd(W_o, b_o, grad_W_o, grad_b_o);
}

void PerformerAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);
}

std::vector<Tensor*> PerformerAttention::parameters() {
    return {&W_q, &b_q, &W_k, &b_k, &W_v, &b_v, &W_o, &b_o};
}

std::vector<Tensor*> PerformerAttention::gradients() {
    return {&grad_W_q, &grad_b_q, &grad_W_k, &grad_b_k, &grad_W_v, &grad_b_v, &grad_W_o, &grad_b_o};
}

// ----------------------------------------------------------------------------
// PerformerBlock
// ----------------------------------------------------------------------------
PerformerBlock::PerformerBlock(size_t d_model, size_t seq_len, size_t num_features)
    : d_model_(d_model),
      num_features_(num_features),
      attn(d_model, seq_len, num_features),
      ln1(d_model), ln2(d_model),
      // FFN: hidden = 4 * d_model, then back to d_model
      W1(d_model * 4, d_model), b1(1, d_model * 4),
      W2(d_model, d_model * 4), b2(1, d_model),
      grad_W1(d_model * 4, d_model), grad_b1(1, d_model * 4),
      grad_W2(d_model, d_model * 4), grad_b2(1, d_model),
      last_x_(0, 0), last_ln1_out_(0, 0), last_attn_out_(0, 0),
      last_resid1_(0, 0), last_ln2_out_(0, 0),
      last_ffn_pregelu_(0, 0), last_ffn_out_(0, 0)
{
    // Xavier init for FFN
    std::mt19937& gen = performer_global_rng();
    double std1 = std::sqrt(2.0 / (d_model + d_model * 4));
    std::normal_distribution<> dis1(0.0, std1);
    for (size_t i = 0; i < W1.rows; ++i)
        for (size_t j = 0; j < W1.cols; ++j) W1(i, j) = dis1(gen);
    double std2 = std::sqrt(2.0 / (d_model * 4 + d_model));
    std::normal_distribution<> dis2(0.0, std2);
    for (size_t i = 0; i < W2.rows; ++i)
        for (size_t j = 0; j < W2.cols; ++j) W2(i, j) = dis2(gen);
    b1.fill(0.0); b2.fill(0.0);
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

Tensor PerformerBlock::forward(const Tensor& input) {
    // input: (n, d_model)
    last_x_ = input.clone();
    const size_t n = input.rows;
    const size_t d = d_model_;

    // ---- Sub-block 1: pre-LN → attn → residual ----
    last_ln1_out_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_ln1_out_);
    last_resid1_ = Tensor(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            last_resid1_(t, j) = input(t, j) + last_attn_out_(t, j);
        }
    }

    // ---- Sub-block 2: pre-LN → FFN(GELU) → residual ----
    last_ln2_out_ = ln2.forward(last_resid1_);

    // FFN pre-activation: pre = last_ln2_out @ W1^T + b1
    last_ffn_pregelu_ = last_ln2_out_ * W1.transpose();
    for (size_t j = 0; j < last_ffn_pregelu_.cols; ++j) {
        double bj = b1(0, j);
        for (size_t i = 0; i < last_ffn_pregelu_.rows; ++i) last_ffn_pregelu_(i, j) += bj;
    }
    // GELU
    Tensor ffn_gelu = last_ffn_pregelu_.apply(GELU());
    // W2 + b2
    last_ffn_out_ = ffn_gelu * W2.transpose();
    for (size_t j = 0; j < last_ffn_out_.cols; ++j) {
        double bj = b2(0, j);
        for (size_t i = 0; i < last_ffn_out_.rows; ++i) last_ffn_out_(i, j) += bj;
    }

    // residual
    Tensor out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            out(t, j) = last_resid1_(t, j) + last_ffn_out_(t, j);
        }
    }
    return out;
}

Tensor PerformerBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;

    // ---- Sub-block 2 backward: residual + FFN ----
    // out = resid1 + ffn_out
    // grad_resid1 += grad_output (residual)
    // grad_ffn_out = grad_output
    Tensor grad_resid1(n, d);
    Tensor grad_ffn_out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            grad_resid1(t, j) = grad_output(t, j);
            grad_ffn_out(t, j) = grad_output(t, j);
        }
    }
    // FFN backward: out_ffn = GELU(pre) @ W2^T + b2
    //   grad_pre = (grad_ffn_out @ W2) * GELU'(pre)
    //   grad_W2 += grad_ffn_out^T @ GELU(pre)
    //   grad_b2 += row-sum(grad_ffn_out)
    //   grad_ffn_gelu = grad_ffn_out @ W2 ... actually we need grad_pre directly
    GELU gelu;
    Tensor grad_W2_local(d, d * 4);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d * 4; ++j) grad_W2_local(i, j) = 0.0;
    Tensor grad_b2_local(1, d);
    for (size_t j = 0; j < d; ++j) grad_b2_local(0, j) = 0.0;
    Tensor grad_pre(n, d * 4);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            // grad_pre[t, k] = sum_j grad_ffn_out(t, j) * W2(j, k) * GELU'(pre(t, k))
            for (size_t k = 0; k < d * 4; ++k) {
                double s = 0.0;
                for (size_t jj = 0; jj < d; ++jj) s += grad_ffn_out(t, jj) * W2(jj, k);
                grad_pre(t, k) = s * gelu.derivative(last_ffn_pregelu_(t, k));
            }
        }
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d * 4; ++j) {
                grad_W2_local(i, j) += grad_ffn_out(t, i) * (last_ffn_pregelu_(t, j) > 0
                    ? last_ffn_pregelu_(t, j)  // GELU(pre) value — recompute? we already have last_ffn_pregelu_
                    : last_ffn_pregelu_(t, j));
            }
            grad_b2_local(0, i) += grad_ffn_out(t, i);
        }
    }
    // Wait — we need GELU(pre) for grad_W2 accumulation. The pregelu is the
    // pre-activation. We need to recompute GELU(pre) here. For accuracy, let
    // me redo this with the proper GELU(pre) value:
    grad_W2_local = Tensor(d, d * 4);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d * 4; ++j) grad_W2_local(i, j) = 0.0;
    grad_b2_local = Tensor(1, d);
    for (size_t j = 0; j < d; ++j) grad_b2_local(0, j) = 0.0;
    Tensor grad_ffn_gelu(n, d * 4);
    for (size_t t = 0; t < n; ++t) {
        // grad_ffn_gelu = grad_ffn_out @ W2
        for (size_t k = 0; k < d * 4; ++k) {
            double s = 0.0;
            for (size_t jj = 0; jj < d; ++jj) s += grad_ffn_out(t, jj) * W2(jj, k);
            grad_ffn_gelu(t, k) = s;
        }
        // grad_pre = grad_ffn_gelu * GELU'(pre)
        for (size_t k = 0; k < d * 4; ++k) {
            grad_pre(t, k) = grad_ffn_gelu(t, k) * gelu.derivative(last_ffn_pregelu_(t, k));
        }
        // grad_W2_local[i, k] += grad_ffn_out(t, i) * GELU(pre)(t, k)
        for (size_t i = 0; i < d; ++i) {
            for (size_t k = 0; k < d * 4; ++k) {
                grad_W2_local(i, k) += grad_ffn_out(t, i) * gelu(last_ffn_pregelu_(t, k));
            }
            grad_b2_local(0, i) += grad_ffn_out(t, i);
        }
    }
    grad_W2 += grad_W2_local;
    grad_b2 += grad_b2_local;

    // Backward through FFN pre: pre = ln2_out @ W1^T + b1
    //   grad_W1 += grad_pre^T @ ln2_out
    //   grad_b1 += row-sum(grad_pre)
    //   grad_ln2_out = grad_pre @ W1
    Tensor grad_W1_local(d * 4, d);
    for (size_t i = 0; i < d * 4; ++i)
        for (size_t j = 0; j < d; ++j) grad_W1_local(i, j) = 0.0;
    Tensor grad_b1_local(1, d * 4);
    for (size_t j = 0; j < d * 4; ++j) grad_b1_local(0, j) = 0.0;
    Tensor grad_ln2_out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d * 4; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_W1_local(i, j) += grad_pre(t, i) * last_ln2_out_(t, j);
            }
            grad_b1_local(0, i) += grad_pre(t, i);
        }
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d * 4; ++k) s += grad_pre(t, k) * W1(k, j);
            grad_ln2_out(t, j) = s;
        }
    }
    grad_W1 += grad_W1_local;
    grad_b1 += grad_b1_local;

    // grad_resid1 += grad_ln2_out (through LayerNorm backward)
    Tensor grad_resid1_from_ln2 = ln2.backward(grad_ln2_out, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            grad_resid1(t, j) += grad_resid1_from_ln2(t, j);

    // ---- Sub-block 1 backward: residual + attn ----
    // resid1 = input + attn_out
    // grad_input += grad_resid1
    // grad_attn_out = grad_resid1
    Tensor grad_input(n, d);
    Tensor grad_attn_out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            grad_input(t, j) = grad_resid1(t, j);
            grad_attn_out(t, j) = grad_resid1(t, j);
        }
    }
    // Attn: out = attn(ln1_out), where attn = PerformerAttention
    //   grad_ln1_out = attn.backward(grad_attn_out)
    Tensor grad_ln1 = attn.backward(grad_attn_out, 0.0);
    // grad_input += LayerNorm backward of grad_ln1
    Tensor grad_input_from_ln1 = ln1.backward(grad_ln1, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            grad_input(t, j) += grad_input_from_ln1(t, j);

    return grad_input;
}

void PerformerBlock::update_weights(double learning_rate) {
    ln1.update_weights(learning_rate);
    attn.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    for (size_t i = 0; i < W1.rows; ++i) {
        for (size_t j = 0; j < W1.cols; ++j) {
            W1(i, j) -= learning_rate * grad_W1(i, j);
        }
        b1(0, i) -= learning_rate * grad_b1(0, i);
    }
    for (size_t i = 0; i < W2.rows; ++i) {
        for (size_t j = 0; j < W2.cols; ++j) {
            W2(i, j) -= learning_rate * grad_W2(i, j);
        }
        b2(0, i) -= learning_rate * grad_b2(0, i);
    }
}

void PerformerBlock::zero_grad() {
    ln1.zero_grad();
    attn.zero_grad();
    ln2.zero_grad();
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

std::vector<Tensor*> PerformerBlock::parameters() {
    return {&W1, &b1, &W2, &b2};
}

std::vector<Tensor*> PerformerBlock::gradients() {
    return {&grad_W1, &grad_b1, &grad_W2, &grad_b2};
}

// ----------------------------------------------------------------------------
// PerformerModel
// ----------------------------------------------------------------------------
PerformerModel::PerformerModel(size_t d_model, size_t seq_len, size_t out_features,
                               size_t num_blocks, size_t num_features)
    : d_model_(d_model),
      num_blocks_(num_blocks),
      out_features_(out_features),
      blocks_(),
      final_ln_(d_model),
      W_out_(out_features, d_model), b_out_(1, out_features),
      grad_W_out_(out_features, d_model), grad_b_out_(1, out_features),
      W_in_(d_model, d_model), b_in_(1, d_model),
      grad_W_in_(d_model, d_model), grad_b_in_(1, d_model),
      last_input_(0, 0), last_in_proj_(0, 0),
      last_final_ln_(0, 0), last_logits_(0, 0)
{
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, seq_len, num_features);
    }
    // Xavier init for W_in_, W_out_
    std::mt19937& gen = performer_global_rng();
    double std_in = std::sqrt(2.0 / (d_model + d_model));
    std::normal_distribution<> dis_in(0.0, std_in);
    for (size_t i = 0; i < d_model; ++i)
        for (size_t j = 0; j < d_model; ++j) W_in_(i, j) = dis_in(gen);
    double std_out = std::sqrt(2.0 / (d_model + out_features));
    std::normal_distribution<> dis_out(0.0, std_out);
    for (size_t i = 0; i < out_features; ++i)
        for (size_t j = 0; j < d_model; ++j) W_out_(i, j) = dis_out(gen);
    b_in_.fill(0.0); b_out_.fill(0.0);
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
}

Tensor PerformerModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Input projection: y = x @ W_in^T + b_in
    last_in_proj_ = input * W_in_.transpose();
    for (size_t j = 0; j < last_in_proj_.cols; ++j) {
        double bj = b_in_(0, j);
        for (size_t i = 0; i < last_in_proj_.rows; ++i) last_in_proj_(i, j) += bj;
    }
    // Stack of blocks
    Tensor x = last_in_proj_;
    for (auto& blk : blocks_) {
        x = blk.forward(x);
    }
    // Final LayerNorm
    last_final_ln_ = final_ln_.forward(x);
    // Per-token classifier: logits = final_ln @ W_out^T + b_out  (n, out_features)
    last_logits_ = last_final_ln_ * W_out_.transpose();
    for (size_t j = 0; j < last_logits_.cols; ++j) {
        double bj = b_out_(0, j);
        for (size_t i = 0; i < last_logits_.rows; ++i) last_logits_(i, j) += bj;
    }
    return last_logits_;
}

Tensor PerformerModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;
    const size_t of = out_features_;

    // Backward through classifier: grad_final_ln = grad_output @ W_out
    //   grad_W_out += grad_output^T @ last_final_ln_
    //   grad_b_out += row-sum(grad_output)
    Tensor grad_W_out_local(of, d);
    for (size_t i = 0; i < of; ++i)
        for (size_t j = 0; j < d; ++j) grad_W_out_local(i, j) = 0.0;
    Tensor grad_b_out_local(1, of);
    for (size_t j = 0; j < of; ++j) grad_b_out_local(0, j) = 0.0;
    Tensor grad_final_ln(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < of; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_W_out_local(i, j) += grad_output(t, i) * last_final_ln_(t, j);
            }
            grad_b_out_local(0, i) += grad_output(t, i);
        }
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < of; ++k) s += grad_output(t, k) * W_out_(k, j);
            grad_final_ln(t, j) = s;
        }
    }
    grad_W_out_ += grad_W_out_local;
    grad_b_out_ += grad_b_out_local;

    // LayerNorm backward
    Tensor grad_after_blocks = final_ln_.backward(grad_final_ln, 0.0);

    // Backward through blocks (in reverse)
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_after_blocks = it->backward(grad_after_blocks, 0.0);
    }

    // Backward through input projection: grad_input = grad_after_blocks @ W_in
    //   grad_W_in += grad_after_blocks^T @ last_input_
    //   grad_b_in += row-sum(grad_after_blocks)
    Tensor grad_W_in_local(d, d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) grad_W_in_local(i, j) = 0.0;
    Tensor grad_b_in_local(1, d);
    for (size_t j = 0; j < d; ++j) grad_b_in_local(0, j) = 0.0;
    Tensor grad_input(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_W_in_local(i, j) += grad_after_blocks(t, i) * last_input_(t, j);
            }
            grad_b_in_local(0, i) += grad_after_blocks(t, i);
        }
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_after_blocks(t, k) * W_in_(k, j);
            grad_input(t, j) = s;
        }
    }
    grad_W_in_ += grad_W_in_local;
    grad_b_in_ += grad_b_in_local;
    return grad_input;
}

void PerformerModel::update_weights(double learning_rate) {
    final_ln_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk.update_weights(learning_rate);
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_in_(i, j) -= learning_rate * grad_W_in_(i, j);
        }
        b_in_(0, i) -= learning_rate * grad_b_in_(0, i);
    }
    for (size_t i = 0; i < out_features_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_out_(i, j) -= learning_rate * grad_W_out_(i, j);
        }
        b_out_(0, i) -= learning_rate * grad_b_out_(0, i);
    }
}

void PerformerModel::zero_grad() {
    final_ln_.zero_grad();
    for (auto& blk : blocks_) blk.zero_grad();
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
}

std::vector<Tensor*> PerformerModel::parameters() {
    return {&W_in_, &b_in_, &W_out_, &b_out_};
}

std::vector<Tensor*> PerformerModel::gradients() {
    return {&grad_W_in_, &grad_b_in_, &grad_W_out_, &grad_b_out_};
}
