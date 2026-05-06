#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include "../../core/layer.h"
#include <cmath>
#include <vector>
#include <algorithm>

class FlashAttentionLayer : public Layer {
public:
    size_t d_model_, num_heads_, d_k_;
    size_t seq_len_;

    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // Cached tensors for backward pass
    Tensor last_q;       // (tokens, d_model) Q after projection
    Tensor last_k;       // (tokens, d_model) K after projection
    Tensor last_v;       // (tokens, d_model) V after projection
    Tensor last_scores;  // (num_heads * tokens, tokens) cached pre-softmax scores
    Tensor last_attn_out; // (tokens, d_model) attention output BEFORE W_o projection (output_acc)
    Tensor last_x;       // (tokens, d_model) original input

    static constexpr size_t TILE = 64;

    FlashAttentionLayer(size_t d_model, size_t num_heads);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    // Tile-based online softmax attention for one head
    // QTile, KTile, VTile: (tile_size, d_k) row-major
    // O_out: (tokens, d_k) output accumulator
    // L_out: (tokens,) row sums for normalization
    // m_out: (tokens,) row max for normalization
    // Returns O_out
    void flash_attn_tile(
        const Tensor& Q,   // (tokens, d_k)
        const Tensor& K,   // (tokens, d_k)
        const Tensor& V,   // (tokens, d_k)
        size_t tokens,
        size_t d_k,
        size_t seq_len,
        Tensor& O_out,
        std::vector<double>& L_out,
        std::vector<double>& m_out,
        bool causal
    ) const;
};

FlashAttentionLayer::FlashAttentionLayer(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads), d_k_(d_model / num_heads), seq_len_(0)
{
    W_q = Tensor::random(d_model_, d_model_, 0.01);
    W_k = Tensor::random(d_model_, d_model_, 0.01);
    W_v = Tensor::random(d_model_, d_model_, 0.01);
    W_o = Tensor::random(d_model_, d_model_, 0.01);

    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
}

std::vector<Tensor*> FlashAttentionLayer::parameters() {
    return { &W_q, &W_k, &W_v, &W_o };
}

std::vector<Tensor*> FlashAttentionLayer::gradients() {
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o };
}

void FlashAttentionLayer::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

void FlashAttentionLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_q[i][j] -= learning_rate * grad_W_q[i][j];
            W_k[i][j] -= learning_rate * grad_W_k[i][j];
            W_v[i][j] -= learning_rate * grad_W_v[i][j];
            W_o[i][j] -= learning_rate * grad_W_o[i][j];
        }
    }
}

void FlashAttentionLayer::flash_attn_tile(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    size_t tokens,
    size_t d_k,
    size_t seq_len,
    Tensor& O_out,
    std::vector<double>& L_out,
    std::vector<double>& m_out,
    bool causal
) const {
    // Rescale factor: 1 / sqrt(d_k)
    double scale = 1.0 / std::sqrt(static_cast<double>(d_k) + 1e-9);

    size_t tile_B = TILE;
    size_t tiles_K = (seq_len + tile_B - 1) / tile_B;

    // Initialize accumulators
    for (size_t t = 0; t < tokens; ++t) {
        m_out[t] = -1e9;
        L_out[t] = 0.0;
        for (size_t dk = 0; dk < d_k; ++dk)
            O_out[t][dk] = 0.0;
    }

    // Process each column block (K/V tiles)
    for (size_t j = 0; j < tiles_K; ++j) {
        size_t col_start = j * tile_B;
        size_t tile_cols = std::min(tile_B, seq_len - col_start);

        // Extract K tile: K_j (tile_cols, d_k) row-major
        Tensor K_j(tile_cols, d_k);
        for (size_t r = 0; r < tile_cols; ++r) {
            size_t src_row = col_start + r;
            for (size_t dk = 0; dk < d_k; ++dk)
                K_j[r][dk] = K[src_row][dk];
        }

        // Extract V tile: V_j (tile_cols, d_k) row-major
        Tensor V_j(tile_cols, d_k);
        for (size_t r = 0; r < tile_cols; ++r) {
            size_t src_row = col_start + r;
            for (size_t dk = 0; dk < d_k; ++dk)
                V_j[r][dk] = V[src_row][dk];
        }

        // Compute S_ij = Q_i @ K_j^T / sqrt(d_k): (tile_B, tile_cols) per row-block
        // We process Q in row blocks too (same tile size)
        size_t tiles_B = (tokens + tile_B - 1) / tile_B;

        for (size_t i = 0; i < tiles_B; ++i) {
            size_t row_start = i * tile_B;
            size_t tile_rows = std::min(tile_B, tokens - row_start);

            // Extract Q tile: Q_i (tile_rows, d_k)
            Tensor Q_i(tile_rows, d_k);
            for (size_t r = 0; r < tile_rows; ++r) {
                size_t src_row = row_start + r;
                for (size_t dk = 0; dk < d_k; ++dk)
                    Q_i[r][dk] = Q[src_row][dk];
            }

            // S_ij = Q_i @ K_j^T: (tile_rows, tile_cols)
            // S_ij[p][c] = sum_dk Q_i[p][dk] * K_j[c][dk]
            std::vector<std::vector<double>> S(tile_rows, std::vector<double>(tile_cols, 0.0));
            for (size_t p = 0; p < tile_rows; ++p) {
                for (size_t c = 0; c < tile_cols; ++c) {
                    double s = 0.0;
                    for (size_t dk = 0; dk < d_k; ++dk)
                        s += Q_i[p][dk] * K_j[c][dk];
                    double val = s * scale;

                    // Causal mask: can only attend to positions <= current
                    size_t global_row = row_start + p;
                    size_t global_col = col_start + c;
                    if (causal && global_col > global_row)
                        val = -1e9;

                    S[p][c] = val;
                }
            }

            // Online softmax update per row in this block
            for (size_t p = 0; p < tile_rows; ++p) {
                size_t t = row_start + p;

                // Row max of S[p][:]
                double m_ij = S[p][0];
                for (size_t c = 1; c < tile_cols; ++c)
                    if (S[p][c] > m_ij) m_ij = S[p][c];

                // Correct for causal: rows where all attended positions are masked
                // If all S[p][c] were -inf (due to causal), m_ij will be -inf
                // In that case skip this block (contribution is zero)
                if (m_ij < -1e8) continue;

                // P_ij = exp(S[p][c] - m_ij)
                double l_ij = 0.0;
                std::vector<double> P(tile_cols);
                for (size_t c = 0; c < tile_cols; ++c) {
                    P[c] = std::exp(S[p][c] - m_ij);
                    l_ij += P[c];
                }

                // Rescale old accumulator
                double m_old = m_out[t];
                double m_new = std::max(m_old, m_ij);

                if (m_old > -1e8) {
                    double factor = std::exp(m_old - m_new);
                    for (size_t dk = 0; dk < d_k; ++dk)
                        O_out[t][dk] *= factor;
                    L_out[t] *= factor;
                }

                // New contribution: P_ij @ V_j
                // O_out[t] += exp(m_ij - m_new) * sum_c P[c] * V_j[c]
                double factor_new = std::exp(m_ij - m_new);
                for (size_t dk = 0; dk < d_k; ++dk) {
                    double v = 0.0;
                    for (size_t c = 0; c < tile_cols; ++c)
                        v += P[c] * V_j[c][dk];
                    O_out[t][dk] += factor_new * v;
                }
                L_out[t] += factor_new * l_ij;
                m_out[t] = m_new;
            }
        }
    }

    // Normalize: O_out[t] /= L_out[t]
    for (size_t t = 0; t < tokens; ++t) {
        if (L_out[t] > 1e-9) {
            double inv = 1.0 / L_out[t];
            for (size_t dk = 0; dk < d_k; ++dk)
                O_out[t][dk] *= inv;
        }
    }
}

Tensor FlashAttentionLayer::forward(const Tensor& input) {
    // input: (d_model, seq_len)
    size_t seq_len = input.cols;
    size_t tokens = seq_len;
    seq_len_ = seq_len;

    // Reshape to (tokens, d_model): each row is one token embedding
    Tensor x(tokens, d_model_);
    for (size_t f = 0; f < d_model_; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x[s][f] = input[f][s];

    last_x = x;

    // Project to Q, K, V: each (tokens, d_model_)
    Tensor Q(tokens, d_model_), K(tokens, d_model_), V(tokens, d_model_);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                qv += x[i][k] * W_q[k][j];
                kv += x[i][k] * W_k[k][j];
                vv += x[i][k] * W_v[k][j];
            }
            Q[i][j] = qv; K[i][j] = kv; V[i][j] = vv;
        }
    }
    last_q = Q; last_k = K; last_v = V;

    // Split into heads: Q_h, K_h, V_h each (tokens, d_k)
    std::vector<Tensor> Q_h(num_heads_, Tensor(tokens, d_k_));
    std::vector<Tensor> K_h(num_heads_, Tensor(tokens, d_k_));
    std::vector<Tensor> V_h(num_heads_, Tensor(tokens, d_k_));

    for (size_t h = 0; h < num_heads_; ++h) {
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t dk = 0; dk < d_k_; ++dk) {
                Q_h[h][t][dk] = Q[t][h * d_k_ + dk];
                K_h[h][t][dk] = K[t][h * d_k_ + dk];
                V_h[h][t][dk] = V[t][h * d_k_ + dk];
            }
        }
    }

    // Attention per head
    Tensor output_acc(tokens, d_model_);
    output_acc.fill(0.0);

    // Cache last_scores: flattened layout for backward
    // Store (num_heads * tile_range, tile_range) - for simplicity store full matrix per head
    last_scores = Tensor(num_heads_ * tokens, tokens);
    last_scores.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Flash attention for this head
        Tensor O(tokens, d_k_);
        std::vector<double> L(tokens, 0.0);
        std::vector<double> m(tokens, -1e9);

        flash_attn_tile(Q_h[h], K_h[h], V_h[h], tokens, d_k_, seq_len, O, L, m, true);

        // Store pre-softmax scores for backward (recompute and cache)
        // We need to store the scores matrix for each head
        // Save into last_scores at row offset h*tokens
        // For backward we need the full scores, so cache them now
        double scale = 1.0 / std::sqrt(static_cast<double>(d_k_) + 1e-9);
        size_t tile_B = TILE;
        (void)tile_B;  // unused in fallback path
        (void)seq_len;  // unused in fallback path

        // Compute and store full scores (pre-softmax, post-mask)
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t j = 0; j < tokens; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < d_k_; ++dk)
                    s += Q_h[h][i][dk] * K_h[h][j][dk];
                s *= scale;
                if (j > i) s = -1e9;
                last_scores[h * tokens + i][j] = s;
            }
        }

        // Add to output_acc at positions [h*d_k_ ... (h+1)*d_k_]
        for (size_t t = 0; t < tokens; ++t)
            for (size_t dk = 0; dk < d_k_; ++dk)
                output_acc[t][h * d_k_ + dk] += O[t][dk];
    }

    // Final projection: output_acc @ W_o^T: (tokens, d_model_) @ (d_model_, d_model_)
    // last_attn_out stores pre-projection attention output (output_acc)
    last_attn_out = output_acc;

    Tensor output(tokens, d_model_);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double val = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                val += output_acc[i][k] * W_o[k][j];
            output[i][j] = val;
        }
    }

    // Reshape output back to (d_model_, seq_len)
    Tensor out_back(d_model_, seq_len);
    for (size_t f = 0; f < d_model_; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            out_back[f][s] = output[s][f];

    return out_back;
}

Tensor FlashAttentionLayer::backward(const Tensor& grad_output, double) {
    // grad_output: (d_model, seq_len)
    size_t seq_len = grad_output.cols;
    size_t tokens = seq_len;

    // Reshape grad_output to (tokens, d_model_)
    Tensor grad_out(tokens, d_model_);
    for (size_t f = 0; f < d_model_; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_out[s][f] = grad_output[f][s];

    // Accumulate grad_W_o += grad_out^T @ last_attn_out (pre-projection)
    // last_attn_out is the attention output before W_o projection (output_acc)
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t t = 0; t < tokens; ++t)
                v += grad_out[t][i] * last_attn_out[t][j];
            grad_W_o[i][j] += v;
        }
    }

    // Propagate through W_o: grad_proj = grad_out @ W_o^T
    Tensor grad_proj(tokens, d_model_);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                v += grad_out[i][k] * W_o[k][j];
            grad_proj[i][j] = v;
        }
    }

    // Split grad_proj per head: grad_proj[:, h*d_k_:(h+1)*d_k_] -> grad_attn_h (tokens, d_k_)
    // Reconstruct per-head Q_h, K_h, V_h from last_q, last_k, last_v
    Tensor grad_q(tokens, d_model_), grad_k(tokens, d_model_), grad_v(tokens, d_model_);
    grad_q.fill(0.0); grad_k.fill(0.0); grad_v.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Reconstruct per-head Q_h (tokens, d_k_), K_h (tokens, d_k_), V_h (tokens, d_k_)
        Tensor Q_h(tokens, d_k_), K_h(tokens, d_k_), V_h(tokens, d_k_);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t dk = 0; dk < d_k_; ++dk) {
                Q_h[t][dk] = last_q[t][h * d_k_ + dk];
                K_h[t][dk] = last_k[t][h * d_k_ + dk];
                V_h[t][dk] = last_v[t][h * d_k_ + dk];
            }
        }

        // Extract grad_proj slice for this head: grad_attn_h (tokens, d_k_)
        Tensor grad_attn_h(tokens, d_k_);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t dk = 0; dk < d_k_; ++dk)
                grad_attn_h[t][dk] = grad_proj[t][h * d_k_ + dk];

        // Retrieve cached pre-softmax scores for this head
        Tensor attn_scores(tokens, tokens);
        for (size_t i = 0; i < tokens; ++i)
            for (size_t j = 0; j < tokens; ++j)
                attn_scores[i][j] = last_scores[h * tokens + i][j];

        // Softmax of cached scores to get attn_probs
        Tensor attn_probs(tokens, tokens);
        for (size_t i = 0; i < tokens; ++i) {
            double max_s = attn_scores[i][0];
            for (size_t j = 1; j < tokens; ++j)
                if (attn_scores[i][j] > max_s) max_s = attn_scores[i][j];
            double sum_exp = 0.0;
            for (size_t j = 0; j < tokens; ++j) {
                attn_scores[i][j] = std::exp(attn_scores[i][j] - max_s);
                sum_exp += attn_scores[i][j];
            }
            for (size_t j = 0; j < tokens; ++j)
                attn_probs[i][j] = attn_scores[i][j] / sum_exp;
        }

        // dL/dV_h = attn_probs^T @ grad_attn_h
        // Shape: (tokens, tokens) @ (tokens, d_k_) = (tokens, d_k_)
        Tensor grad_V_h_t(tokens, d_k_);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k_; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < tokens; ++t)
                    v += attn_probs[t][i] * grad_attn_h[t][dk];
                grad_V_h_t[i][dk] = v;
            }
        }

        // Compute dP = grad_attn_h @ V_h: (tokens, d_k_) @ (tokens, d_k_)^T = (tokens, tokens)
        // dP[j][i] = sum_dk grad_attn_h[j][dk] * V_h[i][dk]
        // This is the first step for both dK and dQ gradients
        Tensor dP(tokens, tokens);
        for (size_t j = 0; j < tokens; ++j) {
            for (size_t i = 0; i < tokens; ++i) {
                double v = 0.0;
                for (size_t dk = 0; dk < d_k_; ++dk)
                    v += grad_attn_h[j][dk] * V_h[i][dk];
                dP[j][i] = v;
            }
        }

        // Compute row sums of attn_probs for softmax Jacobian correction
        // row_sum[j] = sum_c attn_probs[j][c]
        std::vector<double> row_sum(tokens, 0.0);
        for (size_t j = 0; j < tokens; ++j) {
            double s = 0.0;
            for (size_t c = 0; c < tokens; ++c)
                s += attn_probs[j][c];
            row_sum[j] = s;
        }

        // Softmax Jacobian correction for dP (chain through attention scores)
        // For softmax: dL/dS_ji = P_ji * (dP_ji - sum_c P_jc * dP_jc)
        // where dP_ji = dL/d(score_ji) = sum_k grad_attn_h[j][k] * V_i[k]
        std::vector<double> row_sum_corr(tokens, 0.0);
        for (size_t j = 0; j < tokens; ++j) {
            double s = 0.0;
            for (size_t c = 0; c < tokens; ++c)
                s += attn_probs[j][c] * dP[j][c];
            row_sum_corr[j] = s;
        }

        // dL/dK_h[i][dk] = sum_j P[j][i] * Q_h[j][dk] * (dP[j][i] - row_sum_corr[j]) / sqrt(d_k)
        // dL/dQ_h[i][dk] = sum_j P[j][i] * K_h[j][dk] * (dP[j][i] - row_sum_corr[j]) / sqrt(d_k)
        Tensor grad_K_h_t(tokens, d_k_), grad_Q_h_t(tokens, d_k_);
        double scale = 1.0 / std::sqrt((double)d_k_);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k_; ++dk) {
                double gk = 0.0, gq = 0.0;
                for (size_t j = 0; j < tokens; ++j) {
                    double correction = attn_probs[j][i] * (dP[j][i] - row_sum_corr[j]) * scale;
                    gk += correction * Q_h[j][dk];
                    gq += correction * K_h[j][dk];
                }
                grad_K_h_t[i][dk] = gk;
                grad_Q_h_t[i][dk] = gq;
            }
        }

        // dL/dK[i][dk] = sum_j P[j][i] * Q_h[j][dk] * (dP[j][i] - row_sum_corr[j]) / sqrt(d_k)
        // dL/dQ[i][dk] = sum_j P[j][i] * K_h[j][dk] * (dP[j][i] - row_sum_corr[j]) / sqrt(d_k)
        // dL/dV[i][dk] = sum_j attn_probs[j][i] * grad_attn_h[j][dk]

        // Merge head gradients back into full Q, K, V gradient space
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t dk = 0; dk < d_k_; ++dk) {
                grad_q[t][h * d_k_ + dk] = grad_Q_h_t[t][dk];
                grad_k[t][h * d_k_ + dk] = grad_K_h_t[t][dk];
                grad_v[t][h * d_k_ + dk] = grad_V_h_t[t][dk];
            }
        }

        // grad_W_v += last_x^T @ grad_v
        // grad_W_k += last_x^T @ grad_k
        // grad_W_q += last_x^T @ grad_q
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double gq = 0.0, gk = 0.0, gv = 0.0;
                for (size_t t = 0; t < tokens; ++t) {
                    gq += last_x[t][i] * grad_q[t][j];
                    gk += last_x[t][i] * grad_k[t][j];
                    gv += last_x[t][i] * grad_v[t][j];
                }
                grad_W_q[i][j] += gq;
                grad_W_k[i][j] += gk;
                grad_W_v[i][j] += gv;
            }
        }
    }

    // Backprop through Q=x@W_q^T: grad_x = grad_q @ W_q
    Tensor grad_x(tokens, d_model_);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                v += grad_q[i][k] * W_q[k][j];
            grad_x[i][j] = v;
        }
    }

    // Reshape grad_x (tokens, d_model_) back to (d_model_, seq_len)
    Tensor grad_input(d_model_, seq_len);
    for (size_t f = 0; f < d_model_; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_input[f][s] = grad_x[s][f];

    return grad_input;
}

#endif