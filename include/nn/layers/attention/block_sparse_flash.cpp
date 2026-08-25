#include "block_sparse_flash.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <limits>

namespace {

// Project (rows, in) @ (in, out) → (rows, out)
Tensor project_right(const Tensor& input, const Tensor& weights) {
    Tensor output(input.rows, weights.cols);
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t out_feature = 0; out_feature < weights.cols; ++out_feature) {
            double value = 0.0;
            for (size_t in_feature = 0; in_feature < input.cols; ++in_feature) {
                value += input(row, in_feature) * weights(in_feature, out_feature);
            }
            output(row, out_feature) = value;
        }
    }
    return output;
}

void accumulate_weight_gradient(const Tensor& input,
                                const Tensor& grad_output,
                                Tensor& grad_weights) {
    for (size_t in_feature = 0; in_feature < input.cols; ++in_feature) {
        for (size_t out_feature = 0; out_feature < grad_output.cols; ++out_feature) {
            double value = 0.0;
            for (size_t row = 0; row < input.rows; ++row) {
                value += input(row, in_feature) * grad_output(row, out_feature);
            }
            grad_weights(in_feature, out_feature) += value;
        }
    }
}

void accumulate_input_gradient(const Tensor& grad_output,
                               const Tensor& weights,
                               Tensor& grad_input) {
    for (size_t row = 0; row < grad_output.rows; ++row) {
        for (size_t in_feature = 0; in_feature < weights.rows; ++in_feature) {
            double value = 0.0;
            for (size_t out_feature = 0; out_feature < weights.cols; ++out_feature) {
                value += grad_output(row, out_feature)
                       * weights(in_feature, out_feature);
            }
            grad_input(row, in_feature) += value;
        }
    }
}

// ============================================================================
// The core recurrence: per-head online-softmax with block-mask gating.
// Operates on per-head tiles. If mask[i_block, j_block] == 0, the entire
// K_j × V_j tile is skipped.
// ============================================================================
void flash_attn_block_sparse_tile(
    const Tensor& Q_h,      // (n, d_k)
    const Tensor& K_h,      // (n, d_k)
    const Tensor& V_h,      // (n, d_k)
    const Tensor& mask,     // (n_q_blocks, n_k_blocks) in {0, 1}
    size_t n,
    size_t d_k,
    size_t query_block_size,
    size_t key_block_size,
    double scale,
    Tensor& O_h,            // (n, d_k) — output accumulator
    std::vector<double>& L_h,    // (n,) — row sums
    std::vector<double>& m_h) {  // (n,) — row maxes

    const size_t n_q_blocks = (n + query_block_size - 1) / query_block_size;
    const size_t n_k_blocks = (n + key_block_size - 1) / key_block_size;

    // Initialize accumulators
    for (size_t t = 0; t < n; ++t) {
        m_h[t] = -std::numeric_limits<double>::infinity();
        L_h[t] = 0.0;
        for (size_t dk = 0; dk < d_k; ++dk)
            O_h(t, dk) = 0.0;
    }

    // Outer loop: Q-blocks
    for (size_t i = 0; i < n_q_blocks; ++i) {
        size_t row_start = i * query_block_size;
        size_t tile_rows = std::min(query_block_size, n - row_start);

        // Inner loop: K-blocks gated by mask[i, j]
        for (size_t j = 0; j < n_k_blocks; ++j) {
            // BLOCK-SPARSE GATE: skip masked-out tiles
            if (mask(i, j) == 0.0) continue;

            size_t col_start = j * key_block_size;
            size_t tile_cols = std::min(key_block_size, n - col_start);

            // Online softmax update for each row in the Q-tile
            for (size_t p = 0; p < tile_rows; ++p) {
                size_t t = row_start + p;

                // S[t, c] = sum_dk Q[t, dk] * K[c, dk] * scale, c in [col_start, col_start+tile_cols)
                // Find row max
                double m_ij = -std::numeric_limits<double>::infinity();
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t gc = col_start + c;
                    double s = 0.0;
                    for (size_t dk = 0; dk < d_k; ++dk)
                        s += Q_h(t, dk) * K_h(gc, dk);
                    s *= scale;
                    if (s > m_ij) m_ij = s;
                }

                if (!std::isfinite(m_ij)) continue;  // empty tile — skip

                // Compute P, l_ij, and add P @ V_j to O[t]
                double l_ij = 0.0;
                std::vector<double> P(tile_cols, 0.0);
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t gc = col_start + c;
                    double s = 0.0;
                    for (size_t dk = 0; dk < d_k; ++dk)
                        s += Q_h(t, dk) * K_h(gc, dk);
                    s *= scale;
                    P[c] = std::exp(s - m_ij);
                    l_ij += P[c];
                }

                // Rescale old accumulator
                double m_old = m_h[t];
                double m_new = std::max(m_old, m_ij);
                if (std::isfinite(m_old)) {
                    double factor = std::exp(m_old - m_new);
                    for (size_t dk = 0; dk < d_k; ++dk)
                        O_h(t, dk) *= factor;
                    L_h[t] *= factor;
                }

                double factor_new = std::exp(m_ij - m_new);
                for (size_t dk = 0; dk < d_k; ++dk) {
                    double v = 0.0;
                    for (size_t c = 0; c < tile_cols; ++c) {
                        size_t gc = col_start + c;
                        v += P[c] * V_h(gc, dk);
                    }
                    O_h(t, dk) += factor_new * v;
                }
                L_h[t] += factor_new * l_ij;
                m_h[t] = m_new;
            }
        }
    }

    // Normalize
    for (size_t t = 0; t < n; ++t) {
        if (L_h[t] > 1e-30) {
            double inv = 1.0 / L_h[t];
            for (size_t dk = 0; dk < d_k; ++dk)
                O_h(t, dk) *= inv;
        }
        // else: degenerate row — leave O[t] = 0 (no attended blocks)
    }
}

}  // namespace

// ============================================================================
// BlockSparseFlashAttention
// ============================================================================

BlockSparseFlashAttention::BlockSparseFlashAttention(size_t d_model,
                                                      size_t num_heads,
                                                      size_t num_kv_heads,
                                                      size_t query_block_size,
                                                      size_t key_block_size)
    : d_model_(d_model),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      group_size_(num_heads == 0 ? 0
                                  : (num_kv_heads == 0 ? num_heads : num_kv_heads)),
      query_block_size_(query_block_size),
      key_block_size_(key_block_size),
      scale_(head_dim_ == 0 ? 0.0
                            : 1.0 / std::sqrt(static_cast<double>(head_dim_))),
      n_q_blocks_(0),
      n_k_blocks_(0),
      last_n_(0),
      has_forward_cache_(false) {
    if (d_model_ == 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: d_model must be > 0");
    }
    if (num_heads_ == 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: num_heads must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: d_model must be divisible by num_heads");
    }
    if (query_block_size_ == 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: query_block_size must be > 0");
    }
    if (key_block_size_ == 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: key_block_size must be > 0");
    }
    // GQA validation
    if (num_kv_heads_ > num_heads_) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: num_kv_heads must be <= num_heads");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention: num_heads must be divisible by num_kv_heads");
    }
    // Recompute group_size now that everything is validated
    group_size_ = num_heads_ / num_kv_heads_;

    W_q = Tensor::random(d_model_, d_model_, 0.3);
    W_k = Tensor::random(d_model_, d_model_, 0.3);
    W_v = Tensor::random(d_model_, d_model_, 0.3);
    W_o = Tensor::random(d_model_, d_model_, 0.3);

    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
}

std::vector<Tensor*> BlockSparseFlashAttention::parameters() {
    return { &W_q, &W_k, &W_v, &W_o };
}

std::vector<Tensor*> BlockSparseFlashAttention::gradients() {
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o };
}

void BlockSparseFlashAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

void BlockSparseFlashAttention::update_weights(double learning_rate) {
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_q(i, j) -= learning_rate * grad_W_q(i, j);
            W_k(i, j) -= learning_rate * grad_W_k(i, j);
            W_v(i, j) -= learning_rate * grad_W_v(i, j);
            W_o(i, j) -= learning_rate * grad_W_o(i, j);
        }
    }
}

// Mask builders — straightforward pattern construction
Tensor BlockSparseFlashAttention::build_dense_mask(size_t n_q_blocks,
                                                    size_t n_k_blocks) {
    Tensor m(n_q_blocks, n_k_blocks);
    m.fill(1.0);
    return m;
}

Tensor BlockSparseFlashAttention::build_causal_mask(size_t n_q_blocks,
                                                     size_t n_k_blocks) {
    Tensor m(n_q_blocks, n_k_blocks);
    for (size_t i = 0; i < n_q_blocks; ++i) {
        for (size_t j = 0; j < n_k_blocks; ++j) {
            m(i, j) = (j <= i) ? 1.0 : 0.0;
        }
    }
    return m;
}

Tensor BlockSparseFlashAttention::build_sliding_window_mask(size_t n_q_blocks,
                                                             size_t n_k_blocks,
                                                             size_t window_n_blocks) {
    Tensor m(n_q_blocks, n_k_blocks);
    for (size_t i = 0; i < n_q_blocks; ++i) {
        for (size_t j = 0; j < n_k_blocks; ++j) {
            size_t lo = (i + 1 >= window_n_blocks) ? (i + 1 - window_n_blocks) : 0;
            m(i, j) = (j >= lo && j <= i) ? 1.0 : 0.0;
        }
    }
    return m;
}

Tensor BlockSparseFlashAttention::build_strided_mask(size_t n_q_blocks,
                                                      size_t n_k_blocks,
                                                      size_t stride) {
    Tensor m(n_q_blocks, n_k_blocks);
    if (stride == 0) stride = 1;
    for (size_t i = 0; i < n_q_blocks; ++i) {
        for (size_t j = 0; j < n_k_blocks; ++j) {
            m(i, j) = ((i % stride) == (j % stride)) ? 1.0 : 0.0;
        }
    }
    return m;
}

Tensor BlockSparseFlashAttention::build_bigbird_mask(size_t n_q_blocks,
                                                      size_t n_k_blocks,
                                                      size_t window_n_blocks,
                                                      size_t n_global_blocks,
                                                      uint32_t seed) {
    Tensor m = build_sliding_window_mask(n_q_blocks, n_k_blocks, window_n_blocks);
    // Global blocks: rows/cols in [0, n_global_blocks) are always attended
    for (size_t i = 0; i < n_q_blocks; ++i) {
        for (size_t j = 0; j < n_k_blocks && j < n_global_blocks; ++j) {
            m(i, j) = 1.0;
        }
    }
    for (size_t i = 0; i < n_q_blocks && i < n_global_blocks; ++i) {
        for (size_t j = 0; j < n_k_blocks; ++j) {
            m(i, j) = 1.0;
        }
    }
    // Random blocks: per-row, pick 2 random blocks (deterministic from seed)
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, n_k_blocks - 1);
    for (size_t i = 0; i < n_q_blocks; ++i) {
        size_t r1 = dist(rng);
        size_t r2 = dist(rng);
        m(i, r1) = 1.0;
        m(i, r2) = 1.0;
    }
    return m;
}

// Forward with mask — the workhorse
Tensor BlockSparseFlashAttention::forward_with_mask(const Tensor& input,
                                                     const Tensor& mask) {
    if (input.cols != d_model_) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention.forward_with_mask: input cols must equal d_model");
    }
    const size_t n = input.rows;
    last_n_ = n;

    // Validate mask shape
    const size_t exp_q_blocks = (n + query_block_size_ - 1) / query_block_size_;
    const size_t exp_k_blocks = (n + key_block_size_ - 1) / key_block_size_;
    if (mask.rows != exp_q_blocks || mask.cols != exp_k_blocks) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention.forward_with_mask: mask shape mismatch");
    }
    // Validate mask values are in {0, 1}
    for (size_t i = 0; i < mask.data.size(); ++i) {
        double v = mask.data[i];
        if (v != 0.0 && v != 1.0) {
            throw std::invalid_argument(
                "BlockSparseFlashAttention.forward_with_mask: mask values must be in {0,1}");
        }
    }

    n_q_blocks_ = exp_q_blocks;
    n_k_blocks_ = exp_k_blocks;

    // Project to Q, K, V: (n, d_model) each
    Tensor Q = project_right(input, W_q);
    Tensor K = project_right(input, W_k);
    Tensor V = project_right(input, W_v);
    last_query_ = Q;
    last_key_ = K;
    last_value_ = V;
    last_input_ = input;
    last_mask_ = mask;

    // Per-head context accumulator
    Tensor context(n, d_model_);
    context.fill(0.0);

    // Per-head attention
    for (size_t h = 0; h < num_heads_; ++h) {
        size_t kv_head = h / group_size_;

        // Extract per-head Q, K, V slices: (n, head_dim_)
        Tensor Q_h(n, head_dim_);
        Tensor K_h(n, head_dim_);
        Tensor V_h(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h(t, dk) = Q(t, h * head_dim_ + dk);
                K_h(t, dk) = K(t, kv_head * head_dim_ + dk);
                V_h(t, dk) = V(t, kv_head * head_dim_ + dk);
            }
        }

        Tensor O_h(n, head_dim_);
        std::vector<double> L_h(n, 0.0);
        std::vector<double> m_h(n, -std::numeric_limits<double>::infinity());

        flash_attn_block_sparse_tile(
            Q_h, K_h, V_h, mask,
            n, head_dim_,
            query_block_size_, key_block_size_, scale_,
            O_h, L_h, m_h);

        // Scatter into context[h*head_dim_ : (h+1)*head_dim_]
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                context(t, h * head_dim_ + dk) = O_h(t, dk);
            }
        }
    }

    last_context_ = context;
    has_forward_cache_ = true;

    // Output projection: (n, d_model) @ (d_model, d_model) → (n, d_model)
    Tensor output = project_right(context, W_o);
    return output;
}

// Layer-interface forward: uses last_mask_ if cached, else all-ones mask
Tensor BlockSparseFlashAttention::forward(const Tensor& input) {
    if (!has_forward_cache_ || last_mask_.data.empty()) {
        // Build dense mask for the given n
        size_t exp_q = (input.rows + query_block_size_ - 1) / query_block_size_;
        size_t exp_k = (input.rows + key_block_size_ - 1) / key_block_size_;
        Tensor dense = build_dense_mask(exp_q, exp_k);
        return forward_with_mask(input, dense);
    }
    return forward_with_mask(input, last_mask_);
}

// Backward — full BPTT through the masked recurrence
Tensor BlockSparseFlashAttention::backward(const Tensor& grad_output, double) {
    if (!has_forward_cache_) {
        throw std::logic_error(
            "BlockSparseFlashAttention.backward: must call forward first");
    }
    const size_t n = last_n_;
    const Tensor& Q = last_query_;
    const Tensor& K = last_key_;
    const Tensor& V = last_value_;
    const Tensor& mask = last_mask_;
    const Tensor& context = last_context_;

    if (grad_output.rows != n || grad_output.cols != d_model_) {
        throw std::invalid_argument(
            "BlockSparseFlashAttention.backward: grad_output shape mismatch");
    }

    // 1) W_o: grad_W_o += context^T @ grad_output
    accumulate_weight_gradient(context, grad_output, grad_W_o);

    // 2) d_context = grad_output @ W_o^T
    Tensor d_context(n, d_model_);
    accumulate_input_gradient(grad_output, W_o, d_context);

    // 3) Per-head backward through the masked flash recurrence.
    Tensor dQ(n, d_model_);
    Tensor dK(n, d_model_);
    Tensor dV(n, d_model_);
    dQ.fill(0.0);
    dK.fill(0.0);
    dV.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        size_t kv_head = h / group_size_;

        // Per-head Q, K, V slices
        Tensor Q_h(n, head_dim_);
        Tensor K_h(n, head_dim_);
        Tensor V_h(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h(t, dk) = Q(t, h * head_dim_ + dk);
                K_h(t, dk) = K(t, kv_head * head_dim_ + dk);
                V_h(t, dk) = V(t, kv_head * head_dim_ + dk);
            }
        }

        // Per-head d_context slice
        Tensor dO_h(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                dO_h(t, dk) = d_context(t, h * head_dim_ + dk);
            }
        }

        // Per-head gradients
        Tensor dQ_h(n, head_dim_);
        Tensor dK_h(n, head_dim_);
        Tensor dV_h(n, head_dim_);
        dQ_h.fill(0.0);
        dK_h.fill(0.0);
        dV_h.fill(0.0);

        // Per-block backward — skip masked-out blocks entirely
        for (size_t i = 0; i < n_q_blocks_; ++i) {
            size_t row_start = i * query_block_size_;
            size_t tile_rows = std::min(query_block_size_, n - row_start);

            for (size_t j = 0; j < n_k_blocks_; ++j) {
                if (mask(i, j) == 0.0) continue;  // MASK GATE

                size_t col_start = j * key_block_size_;
                size_t tile_cols = std::min(key_block_size_, n - col_start);

                // Recompute S_ij and P_ij for this (i, j) block
                std::vector<std::vector<double>> S(tile_rows,
                                                    std::vector<double>(tile_cols, 0.0));
                std::vector<std::vector<double>> P(tile_rows,
                                                    std::vector<double>(tile_cols, 0.0));
                std::vector<double> m_ij_row(tile_rows, 0.0);

                for (size_t p = 0; p < tile_rows; ++p) {
                    size_t t = row_start + p;
                    double m_ij = -std::numeric_limits<double>::infinity();
                    for (size_t c = 0; c < tile_cols; ++c) {
                        size_t gc = col_start + c;
                        double s = 0.0;
                        for (size_t dk = 0; dk < head_dim_; ++dk)
                            s += Q_h(t, dk) * K_h(gc, dk);
                        s *= scale_;
                        S[p][c] = s;
                        if (s > m_ij) m_ij = s;
                    }
                    if (!std::isfinite(m_ij)) continue;

                    m_ij_row[p] = m_ij;
                    for (size_t c = 0; c < tile_cols; ++c) {
                        P[p][c] = std::exp(S[p][c] - m_ij);
                    }
                }

                // dV_h[c, dk] += sum_p P[p, c] * dO_h[t, dk]
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t gc = col_start + c;
                    for (size_t dk = 0; dk < head_dim_; ++dk) {
                        double v = 0.0;
                        for (size_t p = 0; p < tile_rows; ++p) {
                            size_t t = row_start + p;
                            v += P[p][c] * dO_h(t, dk);
                        }
                        dV_h(gc, dk) += v;
                    }
                }

                // dP[p, c] = sum_dk dO_h[t, dk] * V_h[gc, dk]
                std::vector<std::vector<double>> dP(tile_rows,
                                                     std::vector<double>(tile_cols, 0.0));
                for (size_t p = 0; p < tile_rows; ++p) {
                    size_t t = row_start + p;
                    for (size_t c = 0; c < tile_cols; ++c) {
                        size_t gc = col_start + c;
                        double v = 0.0;
                        for (size_t dk = 0; dk < head_dim_; ++dk)
                            v += dO_h(t, dk) * V_h(gc, dk);
                        dP[p][c] = v;
                    }
                }

                // Softmax Jacobian: row_sum_corr[p] = sum_c P[p, c] * dP[p, c]
                std::vector<double> row_sum_corr(tile_rows, 0.0);
                for (size_t p = 0; p < tile_rows; ++p) {
                    double s = 0.0;
                    for (size_t c = 0; c < tile_cols; ++c)
                        s += P[p][c] * dP[p][c];
                    row_sum_corr[p] = s;
                }

                // dQ_h[t, dk] += sum_c P[p, c] * (dP[p, c] - row_sum_corr[p]) * scale_ * K_h[gc, dk]
                // dK_h[gc, dk] += sum_p P[p, c] * (dP[p, c] - row_sum_corr[p]) * scale_ * Q_h[t, dk]
                for (size_t p = 0; p < tile_rows; ++p) {
                    size_t t = row_start + p;
                    for (size_t c = 0; c < tile_cols; ++c) {
                        size_t gc = col_start + c;
                        double corr = P[p][c] * (dP[p][c] - row_sum_corr[p]) * scale_;
                        for (size_t dk = 0; dk < head_dim_; ++dk) {
                            dQ_h(t, dk) += corr * K_h(gc, dk);
                            dK_h(gc, dk) += corr * Q_h(t, dk);
                        }
                    }
                }
            }
        }

        // Scatter per-head dQ_h back into full Q gradient (Q is per-head)
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                dQ(t, h * head_dim_ + dk) = dQ_h(t, dk);
            }
        }
        // K and V are shared across group_size_ heads; accumulate dK/dV.
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                dK(t, kv_head * head_dim_ + dk) += dK_h(t, dk);
                dV(t, kv_head * head_dim_ + dk) += dV_h(t, dk);
            }
        }
    }

    // 4) W_q, W_k, W_v: grad_W_q += last_input_^T @ dQ, etc.
    accumulate_weight_gradient(last_input_, dQ, grad_W_q);
    accumulate_weight_gradient(last_input_, dK, grad_W_k);
    accumulate_weight_gradient(last_input_, dV, grad_W_v);

    // 5) d_input = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T
    Tensor d_input(n, d_model_);
    accumulate_input_gradient(dQ, W_q, d_input);
    accumulate_input_gradient(dK, W_k, d_input);
    accumulate_input_gradient(dV, W_v, d_input);

    return d_input;
}