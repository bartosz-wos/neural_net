#include "block_sparse_flash.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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

    // Per-head attention. We cache the per-head m_h[t] and L_h[t] for backward.
    // Use plain std::vector storage (avoids Tensor copy-assignment issues).
    size_t cache_n = num_heads_ * n;
    // FIRST: keep Tensor wrappers in sync for test access
    last_m_h_ = Tensor(num_heads_, n);
    last_L_h_ = Tensor(num_heads_, n);
    // THEN: fill the std::vector storage
    last_m_h_storage_.assign(cache_n, -std::numeric_limits<double>::infinity());
    last_L_h_storage_.assign(cache_n, 0.0);
    // Sync Tensor data with storage
    for (size_t i = 0; i < cache_n; ++i) {
        last_m_h_.data[i] = last_m_h_storage_[i];
        last_L_h_.data[i] = last_L_h_storage_[i];
    }

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

        // Cache global m_h, L_h for backward
        for (size_t t = 0; t < n; ++t) {
            last_m_h_storage_[h * n + t] = m_h[t];
            last_L_h_storage_[h * n + t] = L_h[t];
            last_m_h_(h, t) = m_h[t];
            last_L_h_(h, t) = L_h[t];
        }

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

                // Compute GLOBAL row_sum_corr[t] = sum_{c: all unmasked} P[t, c] * D[t, c]
                // where D[t, c] = sum_dk dO_h[t, dk] * V_h[c, dk].
                // This is needed for the softmax Jacobian — it depends on ALL unmasked
                // positions, not just the per-block tile.
                Tensor global_D(n, n);
                global_D.fill(0.0);
                for (size_t t = 0; t < n; ++t) {
                    for (size_t c = 0; c < n; ++c) {
                        double s = 0.0;
                        for (size_t dk = 0; dk < head_dim_; ++dk)
                            s += dO_h(t, dk) * V_h(c, dk);
                        global_D(t, c) = s;
                    }
                }
                // Compute global P[t, c] for all (t, c), and global row_sum_corr[t].
                // global P uses the global m_h, L_h cached from forward.
                std::vector<double> global_row_sum_corr(n, 0.0);
                for (size_t t = 0; t < n; ++t) {
                    double m_g = last_m_h_storage_[h * n + t];
                    double L_g = last_L_h_storage_[h * n + t];
                    if (L_g <= 1e-30) continue;  // degenerate row
                    for (size_t c = 0; c < n; ++c) {
                        double s = 0.0;
                        for (size_t dk = 0; dk < head_dim_; ++dk)
                            s += Q_h(t, dk) * K_h(c, dk);
                        s *= scale_;
                        double P = std::exp(s - m_g) / L_g;
                        global_row_sum_corr[t] += P * global_D(t, c);
                    }
                }

                // Per-block backward — skip masked-out blocks entirely
                for (size_t i = 0; i < n_q_blocks_; ++i) {
                    size_t row_start = i * query_block_size_;
                    size_t tile_rows = std::min(query_block_size_, n - row_start);

                    for (size_t j = 0; j < n_k_blocks_; ++j) {
                        if (mask(i, j) == 0.0) continue;  // MASK GATE

                        size_t col_start = j * key_block_size_;
                        size_t tile_cols = std::min(key_block_size_, n - col_start);

                        // Recompute S_ij for this (i, j) block, and use the GLOBAL
                        // P[t, gc] (full softmax) + GLOBAL row_sum_corr[t] for the
                        // softmax Jacobian.
                        std::vector<std::vector<double>> S(tile_rows,
                                                            std::vector<double>(tile_cols, 0.0));
                        std::vector<std::vector<double>> P(tile_rows,
                                                            std::vector<double>(tile_cols, 0.0));

                        for (size_t p = 0; p < tile_rows; ++p) {
                            size_t t = row_start + p;
                            double m_g = last_m_h_storage_[h * n + t];
                            double L_g = last_L_h_storage_[h * n + t];
                            for (size_t c = 0; c < tile_cols; ++c) {
                                size_t gc = col_start + c;
                                double s = 0.0;
                                for (size_t dk = 0; dk < head_dim_; ++dk)
                                    s += Q_h(t, dk) * K_h(gc, dk);
                                s *= scale_;
                                S[p][c] = s;
                                if (L_g > 1e-30) {
                                    P[p][c] = std::exp(s - m_g) / L_g;
                                } else {
                                    P[p][c] = 0.0;
                                }
                            }
                        }

                        // dV_h[c, dk] += sum_p P[p, c] * dO_h(t, dk)
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

                        // dP[p, c] = sum_dk dO_h(t, dk) * V_h(gc, dk) — use GLOBAL D here
                        std::vector<std::vector<double>> dP(tile_rows,
                                                             std::vector<double>(tile_cols, 0.0));
                        for (size_t p = 0; p < tile_rows; ++p) {
                            size_t t = row_start + p;
                            for (size_t c = 0; c < tile_cols; ++c) {
                                size_t gc = col_start + c;
                                dP[p][c] = global_D(t, gc);
                            }
                        }

                        // Softmax Jacobian: use GLOBAL row_sum_corr
                        for (size_t p = 0; p < tile_rows; ++p) {
                            size_t t = row_start + p;
                            for (size_t c = 0; c < tile_cols; ++c) {
                                size_t gc = col_start + c;
                                double corr = P[p][c] * (dP[p][c] - global_row_sum_corr[t]) * scale_;
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

// ============================================================================
// BlockSparseFlashBlock
// ============================================================================

BlockSparseFlashBlock::BlockSparseFlashBlock(size_t d_model, size_t num_heads,
                                              size_t num_kv_heads,
                                              size_t query_block_size,
                                              size_t key_block_size,
                                              size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      query_block_size_(query_block_size),
      key_block_size_(key_block_size),
      n_q_blocks_(0),
      n_k_blocks_(0),
      ln1_(d_model),
      attn_(d_model, num_heads, num_kv_heads,
            query_block_size, key_block_size),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model) {}

std::vector<Tensor*> BlockSparseFlashBlock::parameters() {
    auto p = ln1_.parameters();
    auto a = attn_.parameters();
    auto q = ln2_.parameters();
    auto f1 = ffn_fc1_.parameters();
    auto f2 = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

std::vector<Tensor*> BlockSparseFlashBlock::gradients() {
    auto p = ln1_.gradients();
    auto a = attn_.gradients();
    auto q = ln2_.gradients();
    auto f1 = ffn_fc1_.gradients();
    auto f2 = ffn_fc2_.gradients();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

void BlockSparseFlashBlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void BlockSparseFlashBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

namespace {
// GELU element-wise in place (matches SHLABlock convention)
inline void gelu_inplace(Tensor& t) {
    for (size_t i = 0; i < t.data.size(); ++i) {
        double x = t.data[i];
        t.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
}
// GELU backward
inline Tensor gelu_backward(const Tensor& grad, const Tensor& pre_gelu) {
    Tensor res(grad.rows, grad.cols);
    for (size_t i = 0; i < grad.data.size(); ++i) {
        double x = pre_gelu.data[i];
        double dy = grad.data[i];
        double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        double gelu_prime = cdf + x * pdf;
        res.data[i] = dy * gelu_prime;
    }
    return res;
}
}  // namespace

Tensor BlockSparseFlashBlock::forward_with_mask(const Tensor& input,
                                                 const Tensor& mask) {
    last_input_ = input;
    last_mask_ = mask;
    Tensor z1 = ln1_.forward(input);
    last_z1_ = z1;
    Tensor attn_out = attn_.forward_with_mask(z1, mask);
    last_attn_out_ = attn_out;
    Tensor res1 = z1 + attn_out;
    last_res1_ = res1;
    if (ffn_dim_ == 0) {
        // Attention-only block: no FFN sublayer
        last_z2_ = Tensor();
        last_ffn_hidden_ = Tensor();
        last_ffn_out_ = Tensor();
        return res1;
    }
    Tensor z2 = ln2_.forward(res1);
    last_z2_ = z2;
    Tensor ffn_hidden_pre = ffn_fc1_.forward(z2);
    last_ffn_hidden_ = ffn_hidden_pre;
    Tensor ffn_hidden = ffn_hidden_pre;
    gelu_inplace(ffn_hidden);
    Tensor ffn_out = ffn_fc2_.forward(ffn_hidden);
    last_ffn_out_ = ffn_out;
    Tensor output = res1 + ffn_out;
    return output;
}

Tensor BlockSparseFlashBlock::forward(const Tensor& input) {
    // Build a dense mask for this input's sequence length. (Equivalent to
    // running plain FlashAttention for the Block's internal self-attention.)
    const size_t n = input.rows;
    const size_t exp_q = (n + query_block_size_ - 1) / query_block_size_;
    const size_t exp_k = (n + key_block_size_ - 1) / key_block_size_;
    n_q_blocks_ = exp_q;
    n_k_blocks_ = exp_k;
    Tensor dense = BlockSparseFlashAttention::build_dense_mask(exp_q, exp_k);
    return forward_with_mask(input, dense);
}

Tensor BlockSparseFlashBlock::backward(const Tensor& grad_output, double lr) {
    if (ffn_dim_ == 0) {
        // Attention-only path
        Tensor d_res1 = grad_output;
        Tensor d_attn_out = grad_output;
        Tensor d_z1_from_attn = attn_.backward(d_attn_out, lr);
        Tensor d_z1(d_res1.rows, d_res1.cols);
        for (size_t i = 0; i < d_z1.data.size(); ++i)
            d_z1.data[i] = d_res1.data[i] + d_z1_from_attn.data[i];
        Tensor d_input = ln1_.backward(d_z1, lr);
        return d_input;
    }
    // Full block: ffn_out branch + residual bypass
    Tensor d_ffn_out = grad_output;
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_out, lr);
    Tensor d_ffn_h_pre = gelu_backward(d_ffn_h, last_ffn_hidden_);
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_h_pre, lr);
    Tensor d_res1_from_ln2 = ln2_.backward(d_z2, lr);
    Tensor d_res1(d_res1_from_ln2.rows, d_res1_from_ln2.cols);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] = grad_output.data[i] + d_res1_from_ln2.data[i];
    Tensor d_z1_residual = d_res1;
    Tensor d_attn_out = d_res1;
    Tensor d_z1_from_attn = attn_.backward(d_attn_out, lr);
    Tensor d_z1(d_z1_residual.rows, d_z1_residual.cols);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] = d_z1_residual.data[i] + d_z1_from_attn.data[i];
    Tensor d_input = ln1_.backward(d_z1, lr);
    return d_input;
}

// ============================================================================
// BlockSparseFlashModel
// ============================================================================

BlockSparseFlashModel::BlockSparseFlashModel(size_t input_dim, size_t d_model,
                                             size_t output_dim,
                                             size_t num_blocks, size_t num_heads,
                                             size_t num_kv_heads,
                                             size_t query_block_size,
                                             size_t key_block_size,
                                             size_t ffn_dim)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_blocks_(num_blocks) {
    W_in_       = Tensor::random(input_dim, d_model, 0.05);
    b_in_       = Tensor::random(1, d_model, 0.05);
    grad_W_in_  = Tensor::zeros(input_dim, d_model);
    grad_b_in_  = Tensor::zeros(1, d_model);
    W_out_      = Tensor::random(d_model, output_dim, 0.05);
    b_out_      = Tensor::random(1, output_dim, 0.05);
    grad_W_out_ = Tensor::zeros(d_model, output_dim);
    grad_b_out_ = Tensor::zeros(1, output_dim);
    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, num_heads, num_kv_heads,
                             query_block_size, key_block_size, ffn_dim);
    }
}

std::vector<Tensor*> BlockSparseFlashModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_in_);
    p.push_back(&b_in_);
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&W_out_);
    p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> BlockSparseFlashModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_W_in_);
    g.push_back(&grad_b_in_);
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_W_out_);
    g.push_back(&grad_b_out_);
    return g;
}

void BlockSparseFlashModel::zero_grad() {
    grad_W_in_.fill(0.0);
    grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void BlockSparseFlashModel::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_in_,  grad_W_in_);
    sgd(b_in_,  grad_b_in_);
    sgd(W_out_, grad_W_out_);
    sgd(b_out_, grad_b_out_);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor BlockSparseFlashModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor proj(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = b_in_(0, j);
            for (size_t k = 0; k < input_dim_; ++k) v += input(i, k) * W_in_(k, j);
            proj(i, j) = v;
        }
    }
    last_proj_ = proj;
    Tensor h = proj;
    for (auto& b : blocks_) h = b.forward(h);
    last_block_out_ = h;
    Tensor output(h.rows, output_dim_);
    for (size_t i = 0; i < h.rows; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double v = b_out_(0, j);
            for (size_t k = 0; k < d_model_; ++k) v += h(i, k) * W_out_(k, j);
            output(i, j) = v;
        }
    }
    return output;
}

Tensor BlockSparseFlashModel::backward(const Tensor& grad_output, double lr) {
    Tensor d_block_out(grad_output.rows, d_model_);
    d_block_out.fill(0.0);
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < output_dim_; ++j) v += grad_output(i, j) * W_out_(k, j);
            d_block_out(i, k) = v;
        }
    }
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < grad_output.rows; ++i) wg += last_block_out_(i, k) * grad_output(i, j);
            grad_W_out_(k, j) += wg;
        }
    }
    for (size_t j = 0; j < output_dim_; ++j) {
        double bg = 0.0;
        for (size_t i = 0; i < grad_output.rows; ++i) bg += grad_output(i, j);
        grad_b_out_(0, j) += bg;
    }

    Tensor d_proj = d_block_out;
    for (auto b_it = blocks_.rbegin(); b_it != blocks_.rend(); ++b_it) {
        d_proj = b_it->backward(d_proj, lr);
    }

    // Backward through input projection
    Tensor d_input(grad_output.rows, input_dim_);
    d_input.fill(0.0);
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t k = 0; k < input_dim_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) v += d_proj(i, j) * W_in_(k, j);
            d_input(i, k) = v;
        }
    }
    for (size_t k = 0; k < input_dim_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < grad_output.rows; ++i) wg += last_input_(i, k) * d_proj(i, j);
            grad_W_in_(k, j) += wg;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        double bg = 0.0;
        for (size_t i = 0; i < grad_output.rows; ++i) bg += d_proj(i, j);
        grad_b_in_(0, j) += bg;
    }
    return d_input;
}
