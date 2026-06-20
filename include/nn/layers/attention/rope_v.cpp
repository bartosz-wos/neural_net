#include "rope_v.h"
#include <cmath>
#include <stdexcept>
#include <tuple>

RoPEWithV::RoPEWithV(int dim, int max_seq_len, float base)
    : dim_(dim)
    , max_seq_len_(max_seq_len)
    , base_(base)
    , current_seq_len_(0)
    , cos_cache(max_seq_len, dim)
    , sin_cache(max_seq_len, dim)
    , grad_cos_cache(max_seq_len, dim)
    , grad_sin_cache(max_seq_len, dim)
{
    if (dim % 2 != 0) {
        throw std::invalid_argument("RoPEWithV: dim must be even, got " + std::to_string(dim));
    }
}

void RoPEWithV::precompute_theta_freqs(int seq_len) {
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument(
            "RoPEWithV::precompute_theta_freqs: seq_len " + std::to_string(seq_len) +
            " exceeds max_seq_len_ " + std::to_string(max_seq_len_));
    }
    current_seq_len_ = seq_len;

    int half_dim = dim_ / 2;
    double log_base = std::log(base_);

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            double theta_i = std::exp(-2.0 * static_cast<double>(i) / dim_ * log_base);
            double angle = static_cast<double>(pos) * theta_i;
            double cos_val = std::cos(angle);
            double sin_val = std::sin(angle);

            cos_cache(pos, i)           = cos_val;
            sin_cache(pos, i)           = sin_val;
            cos_cache(pos, i + half_dim) = cos_val;
            sin_cache(pos, i + half_dim) = sin_val;
        }
    }
}

std::tuple<Tensor, Tensor, Tensor>
RoPEWithV::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    size_t batch = q.rows;
    int tensor_seq_len = q.cols / dim_;
    if (tensor_seq_len == 0) {
        throw std::runtime_error("RoPEWithV::forward: tensor has invalid shape (cols not divisible by dim)");
    }
    if (tensor_seq_len > current_seq_len_) {
        throw std::runtime_error("RoPEWithV::forward: tensor seq_len " + std::to_string(tensor_seq_len) +
            " exceeds precomputed current_seq_len_ " + std::to_string(current_seq_len_) +
            " (call precompute_theta_freqs with larger seq_len first)");
    }
    if (k.rows != batch || k.cols != q.cols || v.rows != batch || v.cols != q.cols) {
        throw std::runtime_error("RoPEWithV::forward: q, k, v must share the same (batch, seq*dim) shape");
    }

    int half_dim = dim_ / 2;

    // Cache inputs for backward
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

            // Rotate Q
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = q_out[b][col_base + i];
                double x_iphd = q_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                q_out[b][col_base + i]           = x_i * cos_t - x_iphd * sin_t;
                q_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
            // Rotate K
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = k_out[b][col_base + i];
                double x_iphd = k_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                k_out[b][col_base + i]           = x_i * cos_t - x_iphd * sin_t;
                k_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
            // Rotate V (same rotation per position — the RoPE-on-K-and-V extension)
            for (int i = 0; i < half_dim; ++i) {
                double x_i    = v_out[b][col_base + i];
                double x_iphd = v_out[b][col_base + i + half_dim];
                double cos_t  = cos_row[i];
                double sin_t  = sin_row[i];
                v_out[b][col_base + i]           = x_i * cos_t - x_iphd * sin_t;
                v_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
        }
    }

    return std::make_tuple(q_out, k_out, v_out);
}

void RoPEWithV::rotate_row(double* row, const double* cos_row, const double* sin_row, int dim) {
    int half_dim = dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        double x_i    = row[i];
        double x_iphd = row[i + half_dim];
        double cos_t  = cos_row[i];
        double sin_t  = sin_row[i];
        row[i]           = x_i * cos_t - x_iphd * sin_t;
        row[i + half_dim] = x_iphd * cos_t + x_i * sin_t;
    }
}

Tensor RoPEWithV::backward_qkv(const Tensor& grad_q, const Tensor& grad_k, const Tensor& grad_v) {
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
                last_grad_q_[b][col_base + i]           = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_q_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }
            // dL/dK backward
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = last_grad_k_[b][col_base + i];
                double dr_iphd = last_grad_k_[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];
                last_grad_k_[b][col_base + i]           = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_k_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }
            // dL/dV backward
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = last_grad_v_[b][col_base + i];
                double dr_iphd = last_grad_v_[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];
                last_grad_v_[b][col_base + i]           = dr_i * cos_t + dr_iphd * sin_t;
                last_grad_v_[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }

            // Cache gradient accumulation across Q, K, V (each tensor
            // contributes its own dL/dcos, dL/dsin, summed over batch).
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

                grad_cos_cache(pos, i)           += grad_c;
                grad_cos_cache(pos, i + half_dim) += grad_c;
                grad_sin_cache(pos, i)           += grad_s;
                grad_sin_cache(pos, i + half_dim) += grad_s;
            }
        }
    }

    // Return the dL/dQ tensor (matches RoPE convention of returning one tensor)
    return last_grad_q_;
}

Tensor RoPEWithV::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Single-input backward path: we have no dL/dK and no dL/dV from this
    // signature, so only the V path is meaningful here (the Layer interface
    // contract returns one gradient tensor). Build zero K and V gradients
    // so the dV path still produces sensible results.
    Tensor zero_k(grad_output.rows, grad_output.cols);
    zero_k.fill(0.0);
    Tensor zero_v(grad_output.rows, grad_output.cols);
    zero_v.fill(0.0);
    // Treat grad_output as dL/dV (since the Layer interface rotates a single
    // input and returns its gradient).
    return backward_qkv(zero_k, zero_k, grad_output);
}

void RoPEWithV::update_weights(double /* learning_rate */) {
    // No learnable parameters.
}

std::vector<Tensor*> RoPEWithV::parameters() {
    return {};
}

std::vector<Tensor*> RoPEWithV::gradients() {
    return {&grad_cos_cache, &grad_sin_cache};
}

void RoPEWithV::zero_grad() {
    grad_cos_cache.fill(0.0);
    grad_sin_cache.fill(0.0);
    last_grad_q_ = Tensor();
    last_grad_k_ = Tensor();
    last_grad_v_ = Tensor();
}