// ============================================================================
// Axial Attention — Ho, Kalchbrenner, Weissenborn, Salimans 2019 implementation
// ============================================================================

#include "axial_attention.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Row-softmax over an (n_rows, n_cols) tensor.
inline Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j)
            if (x[i][j] > row_max) row_max = x[i][j];
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j)
            result[i][j] *= inv;
    }
    return result;
}

// GELU activation + derivative (matches ExpireSpanAttention convention)
inline double gelu_val(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    return 0.5 * xc * (1.0 + std::tanh(u));
}
inline double gelu_deriv(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    double th = std::tanh(u);
    double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + th) + 0.5 * xc * (1.0 - th * th) * du;
}

}  // namespace

// ============================================================================
// AxialAttention
// ============================================================================

AxialAttention::AxialAttention(size_t d_model, size_t H, size_t W,
                               size_t num_heads, bool causal)
    : W_q_row(d_model, d_model), W_k_row(d_model, d_model),
      W_v_row(d_model, d_model), W_o_row(d_model, d_model),
      W_q_col(d_model, d_model), W_k_col(d_model, d_model),
      W_v_col(d_model, d_model), W_o_col(d_model, d_model),
      b_row(1, d_model), b_col(1, d_model),
      grad_W_q_row(d_model, d_model), grad_W_k_row(d_model, d_model),
      grad_W_v_row(d_model, d_model), grad_W_o_row(d_model, d_model),
      grad_W_q_col(d_model, d_model), grad_W_k_col(d_model, d_model),
      grad_W_v_col(d_model, d_model), grad_W_o_col(d_model, d_model),
      grad_b_row(1, d_model), grad_b_col(1, d_model),
      d_model_(d_model), H_(H), W_(W), num_heads_(num_heads),
      causal_(causal)
{
    // Validate BEFORE computing head_dim_/scale_ to avoid div-by-zero.
    if (d_model_ == 0) {
        throw std::invalid_argument("AxialAttention: d_model must be > 0");
    }
    if (H_ == 0) {
        throw std::invalid_argument("AxialAttention: H must be > 0");
    }
    if (W_ == 0) {
        throw std::invalid_argument("AxialAttention: W must be > 0");
    }
    if (num_heads_ == 0) {
        throw std::invalid_argument("AxialAttention: num_heads must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "AxialAttention: d_model must be evenly divisible by num_heads");
    }
    head_dim_ = d_model_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Initialize all 10 weight tensors with small random values.
    {
        Tensor W_q_init = Tensor::random(d_model_, d_model_, 0.02);
        W_q_row = W_q_init;
        W_q_col = W_q_init.clone();
        Tensor W_k_init = Tensor::random(d_model_, d_model_, 0.02);
        W_k_row = W_k_init;
        W_k_col = W_k_init.clone();
        Tensor W_v_init = Tensor::random(d_model_, d_model_, 0.02);
        W_v_row = W_v_init;
        W_v_col = W_v_init.clone();
        Tensor W_o_init = Tensor::random(d_model_, d_model_, 0.02);
        W_o_row = W_o_init;
        W_o_col = W_o_init.clone();
        Tensor b_init = Tensor::random(1, d_model_, 0.02);
        b_row = b_init;
        b_col = b_init.clone();
    }
}

std::vector<Tensor*> AxialAttention::parameters() {
    return {&W_q_row, &W_k_row, &W_v_row, &W_o_row,
            &W_q_col, &W_k_col, &W_v_col, &W_o_col,
            &b_row, &b_col};
}

std::vector<Tensor*> AxialAttention::gradients() {
    return {&grad_W_q_row, &grad_W_k_row, &grad_W_v_row, &grad_W_o_row,
            &grad_W_q_col, &grad_W_k_col, &grad_W_v_col, &grad_W_o_col,
            &grad_b_row, &grad_b_col};
}

void AxialAttention::zero_grad() {
    grad_W_q_row.fill(0.0);
    grad_W_k_row.fill(0.0);
    grad_W_v_row.fill(0.0);
    grad_W_o_row.fill(0.0);
    grad_W_q_col.fill(0.0);
    grad_W_k_col.fill(0.0);
    grad_W_v_col.fill(0.0);
    grad_W_o_col.fill(0.0);
    grad_b_row.fill(0.0);
    grad_b_col.fill(0.0);
}

void AxialAttention::update_weights(double learning_rate) {
    W_q_row -= grad_W_q_row * learning_rate;
    W_k_row -= grad_W_k_row * learning_rate;
    W_v_row -= grad_W_v_row * learning_rate;
    W_o_row -= grad_W_o_row * learning_rate;
    W_q_col -= grad_W_q_col * learning_rate;
    W_k_col -= grad_W_k_col * learning_rate;
    W_v_col -= grad_W_v_col * learning_rate;
    W_o_col -= grad_W_o_col * learning_rate;
    b_row   -= grad_b_row   * learning_rate;
    b_col   -= grad_b_col   * learning_rate;
}

Tensor AxialAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (n != H_ * W_) {
        throw std::invalid_argument(
            "AxialAttention.forward: input.rows must equal H*W");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument(
            "AxialAttention.forward: input.cols must equal d_model");
    }

    last_input_ = input.clone();

    // Q/K/V projections for both axes (no bias).  Shape (H*W, d_model).
    Tensor Q_row(n, d_model_), K_row(n, d_model_), V_row(n, d_model_);
    Tensor Q_col(n, d_model_), K_col(n, d_model_), V_col(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qr = 0.0, kr = 0.0, vr = 0.0;
            double qc = 0.0, kc = 0.0, vc = 0.0;
            for (size_t f = 0; f < d_model_; ++f) {
                double x = input[t][f];
                qr += x * W_q_row[j][f];
                kr += x * W_k_row[j][f];
                vr += x * W_v_row[j][f];
                qc += x * W_q_col[j][f];
                kc += x * W_k_col[j][f];
                vc += x * W_v_col[j][f];
            }
            Q_row[t][j] = qr;
            K_row[t][j] = kr;
            V_row[t][j] = vr;
            Q_col[t][j] = qc;
            K_col[t][j] = kc;
            V_col[t][j] = vc;
        }
    }
    last_q_row_ = Q_row; last_k_row_ = K_row; last_v_row_ = V_row;
    last_q_col_ = Q_col; last_k_col_ = K_col; last_v_col_ = V_col;

    // ---- Row-axis attention ----
    // For each query (i, j), attend over keys (i, j') within the same row.
    // Mask: -1e9 if j' != j, OR (causal && j' > j).
    Tensor scores_row(n, n);
    for (size_t qi = 0; qi < H_; ++qi) {
        for (size_t qj = 0; qj < W_; ++qj) {
            const size_t q_idx = qi * W_ + qj;
            for (size_t ki = 0; ki < H_; ++ki) {
                for (size_t kj = 0; kj < W_; ++kj) {
                    const size_t k_idx = ki * W_ + kj;
                    if (ki != qi) {
                        scores_row[q_idx][k_idx] = -1e9;
                        continue;
                    }
                    if (causal_ && kj > qj) {
                        scores_row[q_idx][k_idx] = -1e9;
                        continue;
                    }
                    double s = 0.0;
                    for (size_t d = 0; d < d_model_; ++d) {
                        s += Q_row[q_idx][d] * K_row[k_idx][d];
                    }
                    scores_row[q_idx][k_idx] = s * scale_;
                }
            }
        }
    }
    Tensor A_row = row_softmax(scores_row);
    last_attn_row_ = A_row;

    // head_out_row = A_row @ V_row  : (n, d_model)
    Tensor head_out_row(n, d_model_);
    head_out_row.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += A_row[t][k] * V_row[k][j];
            head_out_row[t][j] = s;
        }
    }
    last_head_out_row_ = head_out_row;

    // out_row_pre = head_out_row @ W_o_row + b_row  : (n, d_model)
    Tensor out_row_pre(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_row[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += head_out_row[t][k] * W_o_row[k][j];
            out_row_pre[t][j] = s;
        }
    }
    last_out_row_ = out_row_pre;

    // ---- Column-axis attention ----
    // For each query (i, j), attend over keys (i', j) within the same column.
    // Mask: -1e9 if j' != j, OR (causal && i' > i).
    Tensor scores_col(n, n);
    for (size_t qi = 0; qi < H_; ++qi) {
        for (size_t qj = 0; qj < W_; ++qj) {
            const size_t q_idx = qi * W_ + qj;
            for (size_t ki = 0; ki < H_; ++ki) {
                for (size_t kj = 0; kj < W_; ++kj) {
                    const size_t k_idx = ki * W_ + kj;
                    if (kj != qj) {
                        scores_col[q_idx][k_idx] = -1e9;
                        continue;
                    }
                    if (causal_ && ki > qi) {
                        scores_col[q_idx][k_idx] = -1e9;
                        continue;
                    }
                    double s = 0.0;
                    for (size_t d = 0; d < d_model_; ++d) {
                        s += Q_col[q_idx][d] * K_col[k_idx][d];
                    }
                    scores_col[q_idx][k_idx] = s * scale_;
                }
            }
        }
    }
    Tensor A_col = row_softmax(scores_col);
    last_attn_col_ = A_col;

    Tensor head_out_col(n, d_model_);
    head_out_col.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += A_col[t][k] * V_col[k][j];
            head_out_col[t][j] = s;
        }
    }
    last_head_out_col_ = head_out_col;

    Tensor out_col_pre(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_col[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += head_out_col[t][k] * W_o_col[k][j];
            out_col_pre[t][j] = s;
        }
    }
    last_out_col_ = out_col_pre;

    // ---- Residual fuse ----
    // output = out_row_pre + out_col_pre + input
    Tensor output(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            output[t][j] = out_row_pre[t][j] + out_col_pre[t][j] + input[t][j];
        }
    }
    return output;
}

Tensor AxialAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    if (n != H_ * W_) {
        throw std::invalid_argument(
            "AxialAttention.backward: grad_output.rows must equal H*W");
    }
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument(
            "AxialAttention.backward: grad_output.cols must equal d_model");
    }

    // Each axis receives the full gradient (the residual sum splits it
    // equally; each pre-fuse term gets grad_output).
    Tensor d_out_row = grad_output.clone();
    Tensor d_out_col = grad_output.clone();

    Tensor d_input(n, d_model_);
    d_input.fill(0.0);

    // ---- Backward through row axis ----
    {
        // d_head_out_row = d_out_row @ W_o_row^T  : (n, d_model)
        Tensor d_head_out_row(n, d_model_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t k = 0; k < d_model_; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s += d_out_row[t][j] * W_o_row[k][j];
                d_head_out_row[t][k] = s;
            }
        }
        // grad_W_o_row[k, j] += Σ_t d_out_row[t, j] * head_out_row[t, k]
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t t = 0; t < n; ++t)
                    s += d_out_row[t][j] * last_head_out_row_[t][k];
                grad_W_o_row[k][j] += s;
            }
        }
        // grad_b_row[j] += Σ_t d_out_row[t, j]
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += d_out_row[t][j];
            grad_b_row[0][j] += s;
        }

        // Per-head backward (MHA over num_heads_, head_dim_)
        Tensor d_Q_row(n, d_model_); d_Q_row.fill(0.0);
        Tensor d_K_row(n, d_model_); d_K_row.fill(0.0);
        Tensor d_V_row(n, d_model_); d_V_row.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;

            // d_V_h = A^T @ d_head_out_h  : (n, head_dim_)
            // d_A_h = d_head_out_h @ V_h^T : (n, n)
            Tensor d_V_h(n, head_dim_);
            Tensor d_A_h(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        s += last_attn_row_[j][i] * d_head_out_row[j][h_off + d];
                    d_V_h[i][d] = s;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d)
                        s += d_head_out_row[i][h_off + d] * last_v_row_[j][h_off + d];
                    d_A_h[i][j] = s;
                }
            }
            // Softmax backward: d_pre_soft = A * (d_A - row_sum(d_A * A))
            Tensor d_pre_soft(n, n);
            for (size_t i = 0; i < n; ++i) {
                double rsum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    rsum += last_attn_row_[i][j] * d_A_h[i][j];
                for (size_t j = 0; j < n; ++j)
                    d_pre_soft[i][j] = last_attn_row_[i][j] * (d_A_h[i][j] - rsum);
            }
            // d_Q_h = scale * d_pre_soft @ K_h   : (n, head_dim_)
            // d_K_h = scale * d_pre_soft^T @ Q_h : (n, head_dim_)
            Tensor d_Q_h(n, head_dim_);
            Tensor d_K_h(n, head_dim_);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double qv = 0.0, kv = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        qv += d_pre_soft[i][j] * last_k_row_[j][h_off + d];
                        kv += d_pre_soft[j][i] * last_q_row_[j][h_off + d];
                    }
                    d_Q_h[i][d] = qv * scale_;
                    d_K_h[i][d] = kv * scale_;
                }
            }
            // Accumulate into per-head slots.
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q_row[i][h_off + d] += d_Q_h[i][d];
                    d_K_row[i][h_off + d] += d_K_h[i][d];
                    d_V_row[i][h_off + d] += d_V_h[i][d];
                }
            }
        }
        // grad_W_q_row[f, j] += Σ_t input[t, f] * d_Q_row[t, j]
        //  W has shape (d_model, d_model), row index = out-feature j, col index = in-feature f.
        //  d_Q_row[t, j] = Σ_f input[t, f] * W_q_row[j, f]
        //  → d_W_q_row[j, f] += Σ_t input[t, f] * d_Q_row[t, j]
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t f = 0; f < d_model_; ++f) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (size_t t = 0; t < n; ++t) {
                    double x = last_input_[t][f];
                    sq += x * d_Q_row[t][j];
                    sk += x * d_K_row[t][j];
                    sv += x * d_V_row[t][j];
                }
                grad_W_q_row[j][f] += sq;
                grad_W_k_row[j][f] += sk;
                grad_W_v_row[j][f] += sv;
            }
        }
        // d_input contribution from Q/K/V projections:
        //   d_input[t, f] += Σ_j (d_Q_row[t, j] * W_q_row[j, f]
        //                       + d_K_row[t, j] * W_k_row[j, f]
        //                       + d_V_row[t, j] * W_v_row[j, f])
        for (size_t t = 0; t < n; ++t) {
            for (size_t f = 0; f < d_model_; ++f) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    s += d_Q_row[t][j] * W_q_row[j][f]
                       + d_K_row[t][j] * W_k_row[j][f]
                       + d_V_row[t][j] * W_v_row[j][f];
                }
                d_input[t][f] += s;
            }
        }
    }

    // ---- Backward through column axis ----
    {
        Tensor d_head_out_col(n, d_model_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t k = 0; k < d_model_; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s += d_out_col[t][j] * W_o_col[k][j];
                d_head_out_col[t][k] = s;
            }
        }
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t t = 0; t < n; ++t)
                    s += d_out_col[t][j] * last_head_out_col_[t][k];
                grad_W_o_col[k][j] += s;
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += d_out_col[t][j];
            grad_b_col[0][j] += s;
        }

        Tensor d_Q_col(n, d_model_); d_Q_col.fill(0.0);
        Tensor d_K_col(n, d_model_); d_K_col.fill(0.0);
        Tensor d_V_col(n, d_model_); d_V_col.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor d_V_h(n, head_dim_);
            Tensor d_A_h(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        s += last_attn_col_[j][i] * d_head_out_col[j][h_off + d];
                    d_V_h[i][d] = s;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d)
                        s += d_head_out_col[i][h_off + d] * last_v_col_[j][h_off + d];
                    d_A_h[i][j] = s;
                }
            }
            Tensor d_pre_soft(n, n);
            for (size_t i = 0; i < n; ++i) {
                double rsum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    rsum += last_attn_col_[i][j] * d_A_h[i][j];
                for (size_t j = 0; j < n; ++j)
                    d_pre_soft[i][j] = last_attn_col_[i][j] * (d_A_h[i][j] - rsum);
            }
            Tensor d_Q_h(n, head_dim_);
            Tensor d_K_h(n, head_dim_);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double qv = 0.0, kv = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        qv += d_pre_soft[i][j] * last_k_col_[j][h_off + d];
                        kv += d_pre_soft[j][i] * last_q_col_[j][h_off + d];
                    }
                    d_Q_h[i][d] = qv * scale_;
                    d_K_h[i][d] = kv * scale_;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q_col[i][h_off + d] += d_Q_h[i][d];
                    d_K_col[i][h_off + d] += d_K_h[i][d];
                    d_V_col[i][h_off + d] += d_V_h[i][d];
                }
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t f = 0; f < d_model_; ++f) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (size_t t = 0; t < n; ++t) {
                    double x = last_input_[t][f];
                    sq += x * d_Q_col[t][j];
                    sk += x * d_K_col[t][j];
                    sv += x * d_V_col[t][j];
                }
                grad_W_q_col[j][f] += sq;
                grad_W_k_col[j][f] += sk;
                grad_W_v_col[j][f] += sv;
            }
        }
        for (size_t t = 0; t < n; ++t) {
            for (size_t f = 0; f < d_model_; ++f) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    s += d_Q_col[t][j] * W_q_col[j][f]
                       + d_K_col[t][j] * W_k_col[j][f]
                       + d_V_col[t][j] * W_v_col[j][f];
                }
                d_input[t][f] += s;
            }
        }
    }

    // Add residual contribution: d_output is gradient w.r.t. (out_row + out_col + input)
    // so d_input += d_output.
    for (size_t t = 0; t < n; ++t) {
        for (size_t f = 0; f < d_model_; ++f) {
            d_input[t][f] += grad_output[t][f];
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// AxialAttentionBlock
// ============================================================================

AxialAttentionBlock::AxialAttentionBlock(size_t d_model, size_t H, size_t W,
                                         size_t num_heads, size_t ffn_dim,
                                         bool causal)
    : d_model_(d_model), ffn_dim_(ffn_dim),
      attn_(d_model, H, W, num_heads, causal),
      ln1_(d_model), ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim > 0 ? ffn_dim : 1),      // (in=d_model, out=ffn_dim)
      ffn_fc2_(ffn_dim > 0 ? ffn_dim : 1, d_model)        // (in=ffn_dim, out=d_model)
{
    if (d_model_ == 0) {
        throw std::invalid_argument("AxialAttentionBlock: d_model must be > 0");
    }
    if (H == 0 || W == 0) {
        throw std::invalid_argument("AxialAttentionBlock: H and W must be > 0");
    }
}

std::vector<Tensor*> AxialAttentionBlock::parameters() {
    std::vector<Tensor*> p = attn_.parameters();
    auto ln1p = ln1_.parameters();
    auto ln2p = ln2_.parameters();
    p.insert(p.end(), ln1p.begin(), ln1p.end());
    p.insert(p.end(), ln2p.begin(), ln2p.end());
    auto fc1p = ffn_fc1_.parameters();
    auto fc2p = ffn_fc2_.parameters();
    p.insert(p.end(), fc1p.begin(), fc1p.end());
    p.insert(p.end(), fc2p.begin(), fc2p.end());
    return p;
}

std::vector<Tensor*> AxialAttentionBlock::gradients() {
    std::vector<Tensor*> g = attn_.gradients();
    auto ln1g = ln1_.gradients();
    auto ln2g = ln2_.gradients();
    g.insert(g.end(), ln1g.begin(), ln1g.end());
    g.insert(g.end(), ln2g.begin(), ln2g.end());
    auto fc1g = ffn_fc1_.gradients();
    auto fc2g = ffn_fc2_.gradients();
    g.insert(g.end(), fc1g.begin(), fc1g.end());
    g.insert(g.end(), fc2g.begin(), fc2g.end());
    return g;
}

void AxialAttentionBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void AxialAttentionBlock::update_weights(double lr) {
    attn_.update_weights(lr);
    ln1_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor AxialAttentionBlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    Tensor z1 = ln1_.forward(input);
    last_z1_ = z1;

    Tensor attn_out = attn_.forward(z1);
    last_attn_out_ = attn_out;

    Tensor res1 = input + attn_out;
    last_res1_ = res1;

    if (ffn_dim_ > 0) {
        Tensor z2 = ln2_.forward(res1);
        last_z2_ = z2;
        Tensor ffn_pre = ffn_fc1_.forward(z2);
        last_ffn_pre_ = ffn_pre;
        // GELU element-wise
        Tensor ffn_hidden(ffn_pre.rows, ffn_pre.cols);
        for (size_t i = 0; i < ffn_pre.rows; ++i)
            for (size_t j = 0; j < ffn_pre.cols; ++j)
                ffn_hidden[i][j] = gelu_val(ffn_pre[i][j]);
        last_ffn_hidden_ = ffn_hidden;
        Tensor ffn_out = ffn_fc2_.forward(ffn_hidden);
        last_ffn_out_ = ffn_out;
        return res1 + ffn_out;
    } else {
        return res1;
    }
}

Tensor AxialAttentionBlock::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d_res1;
    if (ffn_dim_ > 0) {
        // d_res1 = grad_output  +  d_res1_from_ffn
        // ffn_out = ffn_fc2(ffn_hidden); ffn_hidden = gelu(ffn_fc1(z2)); z2 = ln2(res1)
        Tensor d_ffn_out = grad_output.clone();

        Tensor d_ffn_hidden = ffn_fc2_.backward(d_ffn_out, learning_rate);
        // d_ffn_pre = d_ffn_hidden * gelu'(ffn_pre)
        Tensor d_ffn_pre(d_ffn_hidden.rows, d_ffn_hidden.cols);
        for (size_t i = 0; i < d_ffn_hidden.rows; ++i)
            for (size_t j = 0; j < d_ffn_hidden.cols; ++j)
                d_ffn_pre[i][j] = d_ffn_hidden[i][j] * gelu_deriv(last_ffn_pre_[i][j]);
        Tensor d_z2 = ffn_fc1_.backward(d_ffn_pre, learning_rate);
        Tensor d_res1_from_ffn = ln2_.backward(d_z2, learning_rate);
        d_res1 = grad_output + d_res1_from_ffn;
    } else {
        d_res1 = grad_output;
    }

    // res1 = input + attn_out  →  d_input = d_res1, d_attn_out = d_res1
    Tensor d_attn_out = d_res1.clone();
    Tensor d_input_from_res = d_res1.clone();

    // attn_out = attn(z1)  →  d_z1 = attn.backward(d_attn_out)
    Tensor d_z1 = attn_.backward(d_attn_out, learning_rate);

    // z1 = ln1(input)  →  d_input += ln1.backward(d_z1)
    Tensor d_input_from_ln1 = ln1_.backward(d_z1, learning_rate);

    Tensor d_input = d_input_from_res + d_input_from_ln1;
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// AxialAttentionModel
// ============================================================================

AxialAttentionModel::AxialAttentionModel(size_t d_input, size_t d_model,
                                         size_t d_output, size_t H, size_t W,
                                         size_t num_blocks, size_t num_heads,
                                         size_t ffn_dim, bool causal)
    : d_input_(d_input), d_model_(d_model), d_output_(d_output),
      W_in_(d_input, d_model), W_out_(d_model, d_output),
      ln_final_(d_model),
      blocks_(num_blocks, AxialAttentionBlock(d_model, H, W, num_heads, ffn_dim, causal))
{
    if (d_input_ == 0) {
        throw std::invalid_argument("AxialAttentionModel: d_input must be > 0");
    }
    if (d_model_ == 0) {
        throw std::invalid_argument("AxialAttentionModel: d_model must be > 0");
    }
    if (d_output_ == 0) {
        throw std::invalid_argument("AxialAttentionModel: d_output must be > 0");
    }
    if (num_blocks == 0) {
        throw std::invalid_argument("AxialAttentionModel: num_blocks must be > 0");
    }
    if (H == 0 || W == 0) {
        throw std::invalid_argument("AxialAttentionModel: H and W must be > 0");
    }
}

std::vector<Tensor*> AxialAttentionModel::parameters() {
    std::vector<Tensor*> p = W_in_.parameters();
    auto wo = W_out_.parameters();
    p.insert(p.end(), wo.begin(), wo.end());
    auto lf = ln_final_.parameters();
    p.insert(p.end(), lf.begin(), lf.end());
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}

std::vector<Tensor*> AxialAttentionModel::gradients() {
    std::vector<Tensor*> g = W_in_.gradients();
    auto wo = W_out_.gradients();
    g.insert(g.end(), wo.begin(), wo.end());
    auto lf = ln_final_.gradients();
    g.insert(g.end(), lf.begin(), lf.end());
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    return g;
}

void AxialAttentionModel::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    ln_final_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
}

void AxialAttentionModel::update_weights(double lr) {
    W_in_.update_weights(lr);
    W_out_.update_weights(lr);
    ln_final_.update_weights(lr);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor AxialAttentionModel::forward(const Tensor& input) {
    if (input.cols != d_input_) {
        throw std::invalid_argument(
            "AxialAttentionModel.forward: input cols must equal d_input");
    }
    last_input_ = input.clone();
    Tensor proj = W_in_.forward(input);
    last_proj_ = proj;
    for (auto& b : blocks_) proj = b.forward(proj);
    Tensor final = ln_final_.forward(proj);
    return W_out_.forward(final);
}

Tensor AxialAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d_final = W_out_.backward(grad_output, learning_rate);
    Tensor d_proj  = ln_final_.backward(d_final, learning_rate);
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        d_proj = blocks_[i].backward(d_proj, learning_rate);
    }
    Tensor d_input = W_in_.backward(d_proj, learning_rate);
    return d_input;
}