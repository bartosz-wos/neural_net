#include "yarn_rope.h"
#include <cmath>
#include <stdexcept>
#include <tuple>

// ============================================================================
// YaRN constructor
//
// dim:        embedding dimension (must be even)
// max_seq_len:maximum sequence length the cache can hold
// base:       RoPE theta base (default 10000.0)
// scale:      context extension factor (1.0 = vanilla RoPE; e.g. 8.0 for
//             8K -> 64K context extension)
// alpha:      NTK ramp coefficient (paper default 0.1). At alpha = 0 the
//             per-dim factor is 1 everywhere and YaRN degenerates to vanilla
//             RoPE (the scale parameter is then irrelevant).
// ramp_factor:attention temperature ramp factor in [0, 1] (default 0). At
//             ramp_factor = 0 the temperature is 1 (no scaling); at
//             ramp_factor > 0 it grows as sqrt(1 / t) where t = 1 - ramp.
// ============================================================================

YaRNRoPE::YaRNRoPE(int dim, int max_seq_len, float base,
                   float scale, float alpha, float ramp_factor)
    : dim_(dim)
    , max_seq_len_(max_seq_len)
    , base_(base)
    , scale_(scale)
    , alpha_(alpha)
    , ramp_factor_(ramp_factor)
    , current_seq_len_(0)
    , cos_cache(max_seq_len, dim)
    , sin_cache(max_seq_len, dim)
    , grad_cos_cache(max_seq_len, dim)
    , grad_sin_cache(max_seq_len, dim)
    , freq_scale_by_dim_(dim / 2, 1)   // (half_dim,) — values default to 1.0
    , last_q_()
    , last_k_()
    , last_v_()
    , last_grad_q_()
    , last_grad_k_()
    , last_grad_v_()
{
    if (dim % 2 != 0) {
        throw std::invalid_argument(
            "YaRNRoPE: dim must be even, got " + std::to_string(dim));
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument(
            "YaRNRoPE: max_seq_len must be > 0, got " +
            std::to_string(max_seq_len));
    }
    if (scale < 1.0f) {
        throw std::invalid_argument(
            "YaRNRoPE: scale must be >= 1.0, got " + std::to_string(scale));
    }
    if (alpha < 0.0f) {
        throw std::invalid_argument(
            "YaRNRoPE: alpha must be >= 0, got " + std::to_string(alpha));
    }
    if (ramp_factor < 0.0f || ramp_factor > 1.0f) {
        throw std::invalid_argument(
            "YaRNRoPE: ramp_factor must be in [0, 1], got " +
            std::to_string(ramp_factor));
    }
}

// ============================================================================
// Precompute theta_i = base^(-2i/d), then divide by the per-pair NTK factor:
//   freq_scale_by_dim[i] = 1 / (alpha · i / ((d/2) - 1) + 1)
//   theta_i_yarn         = theta_i / freq_scale_by_dim[i]
//   angle(pos, i)        = pos · theta_i_yarn
//
// At alpha = 0, freq_scale_by_dim[i] = 1 everywhere — vanilla RoPE.
// At alpha = 0.1, freq_scale_by_dim[0] = 1 / 1.1 (~0.909) and
// freq_scale_by_dim[d/2-1] = 1 (no scaling at the lowest-frequency pair).
//
// We fill cache[*, i] and cache[*, i+d/2] with the same value, mirroring
// RoPE's paired-dim convention.
// ============================================================================

void YaRNRoPE::precompute_theta_freqs(int seq_len) {
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument(
            "YaRNRoPE::precompute_theta_freqs: seq_len " +
            std::to_string(seq_len) + " exceeds max_seq_len_ " +
            std::to_string(max_seq_len_));
    }
    current_seq_len_ = seq_len;

    int half_dim = dim_ / 2;
    // Match vanilla RoPE's float-precision log computation so that
    // YaRN(scale=1, alpha=0) produces bit-exact identical caches.
    double log_base = std::log(base_);

    // 1. Precompute per-pair NTK freq scale. half_dim >= 1 (because dim is even
    //    and > 0 — the constructor doesn't enforce > 0 but `dim / 2 > 0` for
    //    any dim >= 2, which is the practical minimum).
    //
    //    Formula: freq_scale_by_dim[i] = 1 / (alpha · i / (half_dim - 1) + 1)
    //    For half_dim = 1, i is always 0 and the formula degenerates to
    //    1 / (alpha · 0 + 1) = 1 — i.e. no scaling. We handle this edge case
    //    explicitly to avoid the divide-by-zero.
    if (half_dim == 1) {
        freq_scale_by_dim_(0, 0) = 1.0;
    } else {
        for (int i = 0; i < half_dim; ++i) {
            double ramp = static_cast<double>(i) / static_cast<double>(half_dim - 1);
            double scale_factor = 1.0 / (static_cast<double>(alpha_) * ramp + 1.0);
            freq_scale_by_dim_(i, 0) = scale_factor;
        }
    }

    // 2. Build cos/sin cache using YaRN-scaled angles.
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            double theta_i = std::exp(
                -2.0 * static_cast<double>(i) / static_cast<double>(dim_) * log_base);
            // YaRN per-dim scaling: divide theta_i by freq_scale_by_dim[i].
            double scale_factor = freq_scale_by_dim_(i, 0);
            double theta_yarn = theta_i / scale_factor;
            double angle = static_cast<double>(pos) * theta_yarn;
            double cos_val = std::cos(angle);
            double sin_val = std::sin(angle);

            cos_cache(pos, i)             = cos_val;
            sin_cache(pos, i)             = sin_val;
            cos_cache(pos, i + half_dim)  = cos_val;
            sin_cache(pos, i + half_dim)  = sin_val;
        }
    }
}

std::tuple<Tensor, Tensor, Tensor>
YaRNRoPE::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    size_t batch = q.rows;
    int tensor_seq_len = q.cols / dim_;
    if (tensor_seq_len == 0) {
        throw std::runtime_error(
            "YaRNRoPE::forward: tensor has invalid shape (cols not divisible by dim)");
    }
    if (tensor_seq_len > current_seq_len_) {
        throw std::runtime_error(
            "YaRNRoPE::forward: tensor seq_len " + std::to_string(tensor_seq_len) +
            " exceeds precomputed current_seq_len_ " +
            std::to_string(current_seq_len_) +
            " (call precompute_theta_freqs with larger seq_len first)");
    }
    if (k.rows != batch || k.cols != q.cols || v.rows != batch || v.cols != q.cols) {
        throw std::runtime_error(
            "YaRNRoPE::forward: q, k, v must share the same (batch, seq*dim) shape");
    }

    int half_dim = dim_ / 2;

    last_q_ = Tensor(q);
    last_k_ = Tensor(k);
    last_v_ = Tensor(v);

    Tensor q_out = q.clone();
    Tensor k_out = k.clone();
    Tensor v_out = v.clone();

    for (size_t b = 0; b < batch; ++b) {
        for (int pos = 0; pos < tensor_seq_len; ++pos) {
            int col_base = pos * dim_;

            const double* cos_row = &cos_cache(pos, 0);
            const double* sin_row = &sin_cache(pos, 0);

            // Q
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = q_out[b][col_base + i];
                double x_iphd = q_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                q_out[b][col_base + i]            = x_i * cos_t - x_iphd * sin_t;
                q_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
            // K
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = k_out[b][col_base + i];
                double x_iphd = k_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                k_out[b][col_base + i]            = x_i * cos_t - x_iphd * sin_t;
                k_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
            // V
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = v_out[b][col_base + i];
                double x_iphd = v_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                v_out[b][col_base + i]            = x_i * cos_t - x_iphd * sin_t;
                v_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
        }
    }

    return std::make_tuple(q_out, k_out, v_out);
}

Tensor YaRNRoPE::backward_qkv(const Tensor& grad_q,
                              const Tensor& grad_k,
                              const Tensor& grad_v) {
    size_t batch = grad_q.rows;
    int tensor_seq_len = grad_q.cols / dim_;
    int half_dim = dim_ / 2;

    last_grad_q_ = Tensor(grad_q);
    last_grad_k_ = Tensor(grad_k);
    last_grad_v_ = Tensor(grad_v);

    grad_cos_cache.fill(0.0);
    grad_sin_cache.fill(0.0);

    for (size_t b = 0; b < batch; ++b) {
        for (int pos = 0; pos < tensor_seq_len; ++pos) {
            int col_base = pos * dim_;

            const double* cos_row = &cos_cache(pos, 0);
            const double* sin_row = &sin_cache(pos, 0);

            // dL/dQ backward (transpose of rotation matrix)
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = last_grad_q_[b][col_base + i];
                double dr_iphd = last_grad_q_[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];
                last_grad_q_[b][col_base + i]            = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_q_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }
            // dL/dK
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = last_grad_k_[b][col_base + i];
                double dr_iphd = last_grad_k_[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];
                last_grad_k_[b][col_base + i]            = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_k_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }
            // dL/dV
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = last_grad_v_[b][col_base + i];
                double dr_iphd = last_grad_v_[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];
                last_grad_v_[b][col_base + i]            = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_v_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }

            // Cache gradient accumulation across Q, K, V (sum over batch).
            // For each pair (i, i+hd):
            //   dL/dcos = sum over (q,k,v) of [-dr_i * x_iphd + dr_iphd * x_i]
            //   dL/dsin = sum over (q,k,v) of [-dr_i * x_i     - dr_iphd * x_iphd]
            // Same as RoPEWithV.
            for (int i = 0; i < half_dim; ++i) {
                double dr_q_i    = grad_q[b][col_base + i];
                double dr_q_iphd = grad_q[b][col_base + i + half_dim];
                double dr_k_i    = grad_k[b][col_base + i];
                double dr_k_iphd = grad_k[b][col_base + i + half_dim];
                double dr_v_i    = grad_v[b][col_base + i];
                double dr_v_iphd = grad_v[b][col_base + i + half_dim];

                double x_q_i    = last_q_[b][col_base + i];
                double x_q_iphd = last_q_[b][col_base + i + half_dim];
                double x_k_i    = last_k_[b][col_base + i];
                double x_k_iphd = last_k_[b][col_base + i + half_dim];
                double x_v_i    = last_v_[b][col_base + i];
                double x_v_iphd = last_v_[b][col_base + i + half_dim];

                double grad_c = -dr_q_i * x_q_iphd + dr_q_iphd * x_q_i
                              + -dr_k_i * x_k_iphd + dr_k_iphd * x_k_i
                              + -dr_v_i * x_v_iphd + dr_v_iphd * x_v_i;

                double grad_s = -dr_q_i * x_q_i - dr_q_iphd * x_q_iphd
                              + -dr_k_i * x_k_i - dr_k_iphd * x_k_iphd
                              + -dr_v_i * x_v_i - dr_v_iphd * x_v_iphd;

                grad_cos_cache(pos, i)            += grad_c;
                grad_cos_cache(pos, i + half_dim) += grad_c;
                grad_sin_cache(pos, i)            += grad_s;
                grad_sin_cache(pos, i + half_dim) += grad_s;
            }
        }
    }

    return last_grad_q_;
}

Tensor YaRNRoPE::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Single-input backward path: only dL/dV is meaningful here (the Layer
    // interface contract rotates a single input). Pass zero K and V grads.
    Tensor zero_k(grad_output.rows, grad_output.cols);
    zero_k.fill(0.0);
    Tensor zero_v(grad_output.rows, grad_output.cols);
    zero_v.fill(0.0);
    return backward_qkv(zero_k, zero_k, grad_output);
}

void YaRNRoPE::update_weights(double /* learning_rate */) {
    // YaRN has no learnable parameters — cos/sin cache is derived from
    // position + the per-dim NTK scale factor, both fixed.
}

std::vector<Tensor*> YaRNRoPE::parameters() {
    return {};
}

std::vector<Tensor*> YaRNRoPE::gradients() {
    return {&grad_cos_cache, &grad_sin_cache};
}

void YaRNRoPE::zero_grad() {
    grad_cos_cache.fill(0.0);
    grad_sin_cache.fill(0.0);
    last_grad_q_ = Tensor();
    last_grad_k_ = Tensor();
    last_grad_v_ = Tensor();
}

// ============================================================================
// Attention temperature (paper §3.3)
//
// t = 1 - ramp_factor · (1 - step / total)
// multiplier = sqrt(1 / t)
//
//   ramp_factor = 0, any step     -> t = 1, multiplier = 1 (no scaling)
//   ramp_factor in (0, 1), step=0 -> t = 1 - ramp, multiplier > 1
//   ramp_factor = 1,   step=total -> t = 1, multiplier = 1
//
// For simplicity, `attention_temperature()` returns sqrt(1/t) using the
// current `ramp_factor_` as if step=0 (the most extreme temperature).
// Callers wanting a different step should use `temperature_for_step`.
// ============================================================================

double YaRNRoPE::attention_temperature() const {
    double t = 1.0 - static_cast<double>(ramp_factor_);
    if (t < 1e-12) t = 1e-12;  // clamp at very small to avoid /0
    return std::sqrt(1.0 / t);
}

double YaRNRoPE::temperature_for_step(int step, int total) const {
    if (total <= 0) {
        return attention_temperature();  // degenerate: caller passed total=0
    }
    double progress = static_cast<double>(step) / static_cast<double>(total);
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;
    double t = 1.0 - static_cast<double>(ramp_factor_) * (1.0 - progress);
    if (t < 1e-12) t = 1e-12;
    return std::sqrt(1.0 / t);
}
