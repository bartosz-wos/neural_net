#include "rope.h"
#include <cmath>
#include <stdexcept>

RoPE::RoPE(int dim, int max_seq_len, float base)
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
        throw std::invalid_argument("RoPE: dim must be even, got " + std::to_string(dim));
    }
}

void RoPE::precompute_theta_freqs(int seq_len) {
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument(
            "RoPE::precompute_theta_freqs: seq_len " + std::to_string(seq_len) +
            " exceeds max_seq_len_ " + std::to_string(max_seq_len_));
    }
    current_seq_len_ = seq_len;

    int half_dim = dim_ / 2;
    double log_base = std::log(base_);

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            // theta_i = base^(-2i/d)
            double theta_i = std::exp(-2.0 * static_cast<double>(i) / dim_ * log_base);
            double angle = static_cast<double>(pos) * theta_i;
            double cos_val = std::cos(angle);
            double sin_val = std::sin(angle);

            // cos_cache and sin_cache for the pair (i, i+half_dim) have same values
            cos_cache(pos, i)          = cos_val;
            sin_cache(pos, i)          = sin_val;
            cos_cache(pos, i + half_dim) = cos_val;
            sin_cache(pos, i + half_dim) = sin_val;
        }
        // Remaining dimensions (if any beyond the paired region) keep 0 or stay as initialized
    }
}

std::pair<Tensor, Tensor> RoPE::forward(const Tensor& q, const Tensor& k) {
    // Input shapes: (batch, seq * dim) — each row is batch, cols are seq concatenated dim values
    // Internal layout: (batch, seq, dim) collapsed to (batch, seq * dim)
    // For rotation we treat each token position independently.

    size_t batch = q.rows;
    int tensor_seq_len = q.cols / dim_;  // actual seq_len from tensor shape
    if (tensor_seq_len == 0) {
        throw std::runtime_error("RoPE::forward: tensor has invalid shape (cols not divisible by dim)");
    }
    if (tensor_seq_len > current_seq_len_) {
        throw std::runtime_error("RoPE::forward: tensor seq_len " + std::to_string(tensor_seq_len) +
            " exceeds precomputed current_seq_len_ " + std::to_string(current_seq_len_) +
            " (call precompute_theta_freqs with larger seq_len first)");
    }

    int half_dim = dim_ / 2;

    // Cache inputs for backward (deep copy since q_out/k_out are modified in-place)
    last_q_ = Tensor(q);
    last_k_ = Tensor(k);
    // Resize cache to match actual tensor seq_len (not the precomputed seq_len)
    last_cos_.data.resize(batch * tensor_seq_len * dim_);
    last_cos_.rows = batch;
    last_cos_.cols = tensor_seq_len * dim_;
    last_sin_.data.resize(batch * tensor_seq_len * dim_);
    last_sin_.rows = batch;
    last_sin_.cols = tensor_seq_len * dim_;

    Tensor q_out = q.clone();
    Tensor k_out = k.clone();

    for (size_t b = 0; b < batch; ++b) {
        for (int pos = 0; pos < tensor_seq_len; ++pos) {
            // col indexes into the 2D tensor (batch, seq_len * dim_)
            // col = token position * dim within this row
            int col_base = pos * dim_;

            // cos_row and sin_row point into the precomputed cache for position `pos`
            const double* cos_row = &cos_cache(pos, 0);
            const double* sin_row = &sin_cache(pos, 0);

            // Rotate first half of dimensions using (i, i+half_dim) pairs
            for (int i = 0; i < half_dim; ++i) {
                double x_i     = q_out[b][col_base + i];
                double x_iphd  = q_out[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];

                q_out[b][col_base + i]         = x_i * cos_t - x_iphd * sin_t;
                q_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
            // Cache the cos/sin values actually used (for backward)
            for (int d = 0; d < dim_; ++d) {
                last_cos_(b, col_base + d) = cos_row[d % dim_];
                last_sin_(b, col_base + d) = sin_row[d % dim_];
            }

            // Same rotation for k
            for (int i = 0; i < half_dim; ++i) {
                double x_i     = k_out[b][col_base + i];
                double x_iphd  = k_out[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];

                k_out[b][col_base + i]         = x_i * cos_t - x_iphd * sin_t;
                k_out[b][col_base + i + half_dim] = x_iphd * cos_t + x_i * sin_t;
            }
        }
    }

    return {q_out, k_out};
}

void RoPE::rotate_row(double* row, const double* cos_row, const double* sin_row, int dim) {
    int half_dim = dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        double x_i    = row[i];
        double x_iphd = row[i + half_dim];
        double cos_t  = cos_row[i];
        double sin_t  = sin_row[i];

        row[i]         = x_i * cos_t - x_iphd * sin_t;
        row[i + half_dim] = x_iphd * cos_t + x_i * sin_t;
    }
}

Tensor RoPE::backward(const Tensor& grad_output, double) {
    // grad_output: (batch, seq * dim) — gradients of the loss w.r.t. rotated q
    // We need grad_wrt_input (unrotated q), and grad_wrt cos/sin cache.

    size_t batch = grad_output.rows;
    int tensor_seq_len = grad_output.cols / dim_;  // actual seq_len from tensor shape
    int half_dim = dim_ / 2;

    Tensor grad_q(grad_output);  // deep copy — we'll mutate grad_q in-place and return it

    // Clear cache gradients (accumulate across batch)
    grad_cos_cache.fill(0.0);
    grad_sin_cache.fill(0.0);

    // For each position in each batch row, compute the gradient wrt the original (unrotated) q
    // The backward rotation is: dL/dx_i = dL/dr_i * cos - dL/dr_{i+hd} * sin
    //                         dL/dx_{i+hd} = dL/dr_{i+hd} * cos + dL/dr_i * sin
    // This is the transpose of the forward rotation matrix.

    for (size_t b = 0; b < batch; ++b) {
        for (int pos = 0; pos < tensor_seq_len; ++pos) {
            int col_base = pos * dim_;

            const double* cos_row = &cos_cache(pos, 0);
            const double* sin_row = &sin_cache(pos, 0);

            // Backward through rotation: apply transpose of rotation matrix
            // grad_x = R^T @ grad_r, where R is the rotation matrix for this position
            // For each pair (i, i+hd):
            //   grad_x_i     = grad_r_i * cos + grad_r_{i+hd} * sin
            //   grad_x_{i+hd} = -grad_r_i * sin + grad_r_{i+hd} * cos
            // Wait - let me re-derive.
            // Forward: r_i     = x_i * cos - x_{i+hd} * sin
            //          r_{i+hd} = x_i * sin + x_{i+hd} * cos
            //
            // So r = R @ x where R = [[cos, -sin], [sin, cos]]
            // Then dx = R^T @ dr
            //   dx_i     = dr_i * cos + dr_{i+hd} * sin
            //   dx_{i+hd} = -dr_i * sin + dr_{i+hd} * cos

            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = grad_q[b][col_base + i];
                double dr_iphd = grad_q[b][col_base + i + half_dim];
                double cos_t   = cos_row[i];
                double sin_t   = sin_row[i];

                grad_q[b][col_base + i]         = dr_i * cos_t + dr_iphd * sin_t;
                grad_q[b][col_base + i + half_dim] = -dr_i * sin_t + dr_iphd * cos_t;
            }

            // Cache gradient for cos and sin (accumulate over batch)
            // dL/dcos_{pos,i} = dL/dr_i * (-x_{i+hd}) + dL/dr_{i+hd} * x_i
            // dL/dsin_{pos,i} = dL/dr_i * (-x_i) + dL/dr_{i+hd} * x_{i+hd}
            // (same for paired dimension i+hd)
            for (int i = 0; i < half_dim; ++i) {
                double dr_i    = grad_output[b][col_base + i];
                double dr_iphd = grad_output[b][col_base + i + half_dim];
                double x_i     = last_q_[b][col_base + i];
                double x_iphd  = last_q_[b][col_base + i + half_dim];

                double grad_c = -dr_i * x_iphd + dr_iphd * x_i;
                double grad_s = -dr_i * x_i - dr_iphd * x_iphd;

                // Accumulate into cache gradients (same for both paired dims)
                grad_cos_cache(pos, i) += grad_c;
                grad_cos_cache(pos, i + half_dim) += grad_c;
                grad_sin_cache(pos, i) += grad_s;
                grad_sin_cache(pos, i + half_dim) += grad_s;
            }
        }
    }

    // Also compute gradient for k (same transformation)
    // Note: we return only grad_q here since that's what the Layer interface expects.
    // The caller (e.g., MultiHeadAttention) should separately handle k gradients.
    return grad_q;
}

void RoPE::update_weights(double /* learning_rate */) {
    // RoPE has no learnable weights — all parameters (cos/sin cache) are fixed
    // and derived from position, not trainable.
    // Subclasses or wrapping layers may override this.
}

std::vector<Tensor*> RoPE::parameters() {
    // No learnable parameters; cache is derived from position
    return {};
}

std::vector<Tensor*> RoPE::gradients() {
    return {&grad_cos_cache, &grad_sin_cache};
}

void RoPE::zero_grad() {
    grad_cos_cache.fill(0.0);
    grad_sin_cache.fill(0.0);
}

void RoPE::serialize(std::ostream&) const {
    // No-op: serialization not yet integrated with Tensor I/O
}

RoPE* RoPE::deserialize(std::istream&, Layer*) {
    // Placeholder: serialization not yet integrated with Tensor I/O
    return nullptr;
}

void RoPE::compute_cache_gradients(int seq_len, int dim_per_head) {
    // Helper called during backward if needed — gradient is already accumulated in grad_cos/sin_cache
    (void)seq_len;
    (void)dim_per_head;
}