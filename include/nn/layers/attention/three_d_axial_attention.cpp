// ============================================================================
// 3D Axial Attention — Ho, Kalchbrenner, Weissenborn, Salimans 2019 §3 (video)
// ============================================================================

#include "three_d_axial_attention.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Row-softmax over an (n_rows, n_cols) tensor (same as 2D axial).
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

// GELU activation + derivative (matches AxialAttention convention).
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
// ThreeDAxialAttention
// ============================================================================

ThreeDAxialAttention::ThreeDAxialAttention(size_t d_model, size_t H, size_t W, size_t T,
                                           size_t num_heads,
                                           bool causal_h, bool causal_w, bool causal_t)
    : W_q_h(d_model, d_model), W_k_h(d_model, d_model),
      W_v_h(d_model, d_model), W_o_h(d_model, d_model),
      W_q_w(d_model, d_model), W_k_w(d_model, d_model),
      W_v_w(d_model, d_model), W_o_w(d_model, d_model),
      W_q_t(d_model, d_model), W_k_t(d_model, d_model),
      W_v_t(d_model, d_model), W_o_t(d_model, d_model),
      b_h(1, d_model), b_w(1, d_model), b_t(1, d_model),
      grad_W_q_h(d_model, d_model), grad_W_k_h(d_model, d_model),
      grad_W_v_h(d_model, d_model), grad_W_o_h(d_model, d_model),
      grad_W_q_w(d_model, d_model), grad_W_k_w(d_model, d_model),
      grad_W_v_w(d_model, d_model), grad_W_o_w(d_model, d_model),
      grad_W_q_t(d_model, d_model), grad_W_k_t(d_model, d_model),
      grad_W_v_t(d_model, d_model), grad_W_o_t(d_model, d_model),
      grad_b_h(1, d_model), grad_b_w(1, d_model), grad_b_t(1, d_model),
      d_model_(d_model), H_(H), W_(W), T_(T), num_heads_(num_heads),
      causal_h_(causal_h), causal_w_(causal_w), causal_t_(causal_t)
{
    // Validate BEFORE computing head_dim_/scale_ to avoid div-by-zero.
    if (d_model_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttention: d_model must be > 0");
    }
    if (H_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttention: H must be > 0");
    }
    if (W_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttention: W must be > 0");
    }
    if (T_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttention: T must be > 0");
    }
    if (num_heads_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttention: num_heads must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "ThreeDAxialAttention: d_model must be evenly divisible by num_heads");
    }
    head_dim_ = d_model_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Initialize all 15 weight tensors with small random values.
    {
        Tensor W_q_init = Tensor::random(d_model_, d_model_, 0.02);
        W_q_h = W_q_init; W_q_w = W_q_init.clone(); W_q_t = W_q_init.clone();
        Tensor W_k_init = Tensor::random(d_model_, d_model_, 0.02);
        W_k_h = W_k_init; W_k_w = W_k_init.clone(); W_k_t = W_k_init.clone();
        Tensor W_v_init = Tensor::random(d_model_, d_model_, 0.02);
        W_v_h = W_v_init; W_v_w = W_v_init.clone(); W_v_t = W_v_init.clone();
        Tensor W_o_init = Tensor::random(d_model_, d_model_, 0.02);
        W_o_h = W_o_init; W_o_w = W_o_init.clone(); W_o_t = W_o_init.clone();
        Tensor b_init = Tensor::random(1, d_model_, 0.02);
        b_h = b_init; b_w = b_init.clone(); b_t = b_init.clone();
    }
}

std::vector<Tensor*> ThreeDAxialAttention::parameters() {
    return {&W_q_h, &W_k_h, &W_v_h, &W_o_h,
            &W_q_w, &W_k_w, &W_v_w, &W_o_w,
            &W_q_t, &W_k_t, &W_v_t, &W_o_t,
            &b_h, &b_w, &b_t};
}

std::vector<Tensor*> ThreeDAxialAttention::gradients() {
    return {&grad_W_q_h, &grad_W_k_h, &grad_W_v_h, &grad_W_o_h,
            &grad_W_q_w, &grad_W_k_w, &grad_W_v_w, &grad_W_o_w,
            &grad_W_q_t, &grad_W_k_t, &grad_W_v_t, &grad_W_o_t,
            &grad_b_h, &grad_b_w, &grad_b_t};
}

void ThreeDAxialAttention::zero_grad() {
    grad_W_q_h.fill(0.0); grad_W_k_h.fill(0.0);
    grad_W_v_h.fill(0.0); grad_W_o_h.fill(0.0);
    grad_W_q_w.fill(0.0); grad_W_k_w.fill(0.0);
    grad_W_v_w.fill(0.0); grad_W_o_w.fill(0.0);
    grad_W_q_t.fill(0.0); grad_W_k_t.fill(0.0);
    grad_W_v_t.fill(0.0); grad_W_o_t.fill(0.0);
    grad_b_h.fill(0.0); grad_b_w.fill(0.0); grad_b_t.fill(0.0);
}

void ThreeDAxialAttention::update_weights(double learning_rate) {
    W_q_h -= grad_W_q_h * learning_rate; W_k_h -= grad_W_k_h * learning_rate;
    W_v_h -= grad_W_v_h * learning_rate; W_o_h -= grad_W_o_h * learning_rate;
    W_q_w -= grad_W_q_w * learning_rate; W_k_w -= grad_W_k_w * learning_rate;
    W_v_w -= grad_W_v_w * learning_rate; W_o_w -= grad_W_o_w * learning_rate;
    W_q_t -= grad_W_q_t * learning_rate; W_k_t -= grad_W_k_t * learning_rate;
    W_v_t -= grad_W_v_t * learning_rate; W_o_t -= grad_W_o_t * learning_rate;
    b_h -= grad_b_h * learning_rate;
    b_w -= grad_b_w * learning_rate;
    b_t -= grad_b_t * learning_rate;
}

// Helper: row-major flatten of (i, j, t) into idx.  Public so tests can mirror.
static inline size_t flat_idx_3d(size_t i, size_t j, size_t t,
                                  size_t W, size_t T) {
    return i * (W * T) + j * T + t;
}

Tensor ThreeDAxialAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    const size_t WT = W_ * T_;
    if (n != H_ * WT) {
        throw std::invalid_argument(
            "ThreeDAxialAttention.forward: input.rows must equal H*W*T");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument(
            "ThreeDAxialAttention.forward: input.cols must equal d_model");
    }

    last_input_ = input.clone();

    // Q/K/V projections for all three axes (no bias). Shape (n, d_model).
    Tensor Q_h(n, d_model_), K_h(n, d_model_), V_h(n, d_model_);
    Tensor Q_w(n, d_model_), K_w(n, d_model_), V_w(n, d_model_);
    Tensor Q_t(n, d_model_), K_t(n, d_model_), V_t(n, d_model_);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qh = 0.0, kh = 0.0, vh = 0.0;
            double qwv = 0.0, kw = 0.0, vw = 0.0;
            double qt = 0.0, kt = 0.0, vt = 0.0;
            for (size_t f = 0; f < d_model_; ++f) {
                double x = input[q][f];
                qh += x * W_q_h[j][f];
                kh += x * W_k_h[j][f];
                vh += x * W_v_h[j][f];
                qwv += x * W_q_w[j][f];
                kw += x * W_k_w[j][f];
                vw += x * W_v_w[j][f];
                qt += x * W_q_t[j][f];
                kt += x * W_k_t[j][f];
                vt += x * W_v_t[j][f];
            }
            Q_h[q][j] = qh; K_h[q][j] = kh; V_h[q][j] = vh;
            Q_w[q][j] = qwv; K_w[q][j] = kw; V_w[q][j] = vw;
            Q_t[q][j] = qt; K_t[q][j] = kt; V_t[q][j] = vt;
        }
    }
    last_q_h_ = Q_h; last_k_h_ = K_h; last_v_h_ = V_h;
    last_q_w_ = Q_w; last_k_w_ = K_w; last_v_w_ = V_w;
    last_q_t_ = Q_t; last_k_t_ = K_t; last_v_t_ = V_t;

    // ---- H-axis attention ----
    // For each query (i, j, t), attend over keys (i', j, t) within the same (j, t).
    // Mask: -1e9 if (j' != j) or (t' != t), OR (causal_h && i' > i).
    Tensor scores_h(n, n);
    scores_h.fill(0.0);
    for (size_t qi = 0; qi < H_; ++qi) {
        for (size_t qj = 0; qj < W_; ++qj) {
            for (size_t qt = 0; qt < T_; ++qt) {
                const size_t q_idx = flat_idx_3d(qi, qj, qt, W_, T_);
                for (size_t ki = 0; ki < H_; ++ki) {
                    for (size_t kj = 0; kj < W_; ++kj) {
                        for (size_t kt = 0; kt < T_; ++kt) {
                            const size_t k_idx = flat_idx_3d(ki, kj, kt, W_, T_);
                            if (kj != qj || kt != qt) {
                                scores_h[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            if (causal_h_ && ki > qi) {
                                scores_h[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            double s = 0.0;
                            for (size_t d = 0; d < d_model_; ++d) {
                                s += Q_h[q_idx][d] * K_h[k_idx][d];
                            }
                            scores_h[q_idx][k_idx] = s * scale_;
                        }
                    }
                }
            }
        }
    }
    Tensor A_h = row_softmax(scores_h);
    last_attn_h_ = A_h;

    // head_out_h = A_h @ V_h : (n, d_model)
    Tensor head_out_h(n, d_model_);
    head_out_h.fill(0.0);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += A_h[q][k] * V_h[k][j];
            head_out_h[q][j] = s;
        }
    }
    last_head_out_h_ = head_out_h;

    // out_h_pre = head_out_h @ W_o_h + b_h : (n, d_model)
    Tensor out_h_pre(n, d_model_);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_h[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += head_out_h[q][k] * W_o_h[k][j];
            out_h_pre[q][j] = s;
        }
    }
    last_out_h_ = out_h_pre;

    // ---- W-axis attention ----
    // For each query (i, j, t), attend over keys (i, j', t) within the same (i, t).
    // Mask: -1e9 if (i' != i) or (t' != t), OR (causal_w && j' > j).
    Tensor scores_w(n, n);
    scores_w.fill(0.0);
    for (size_t qi = 0; qi < H_; ++qi) {
        for (size_t qj = 0; qj < W_; ++qj) {
            for (size_t qt = 0; qt < T_; ++qt) {
                const size_t q_idx = flat_idx_3d(qi, qj, qt, W_, T_);
                for (size_t ki = 0; ki < H_; ++ki) {
                    for (size_t kj = 0; kj < W_; ++kj) {
                        for (size_t kt = 0; kt < T_; ++kt) {
                            const size_t k_idx = flat_idx_3d(ki, kj, kt, W_, T_);
                            if (ki != qi || kt != qt) {
                                scores_w[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            if (causal_w_ && kj > qj) {
                                scores_w[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            double s = 0.0;
                            for (size_t d = 0; d < d_model_; ++d) {
                                s += Q_w[q_idx][d] * K_w[k_idx][d];
                            }
                            scores_w[q_idx][k_idx] = s * scale_;
                        }
                    }
                }
            }
        }
    }
    Tensor A_w = row_softmax(scores_w);
    last_attn_w_ = A_w;

    Tensor head_out_w(n, d_model_);
    head_out_w.fill(0.0);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += A_w[q][k] * V_w[k][j];
            head_out_w[q][j] = s;
        }
    }
    last_head_out_w_ = head_out_w;

    Tensor out_w_pre(n, d_model_);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_w[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += head_out_w[q][k] * W_o_w[k][j];
            out_w_pre[q][j] = s;
        }
    }
    last_out_w_ = out_w_pre;

    // ---- T-axis (depth) attention ----
    // For each query (i, j, t), attend over keys (i, j, t') within the same (i, j).
    // Mask: -1e9 if (i' != i) or (j' != j), OR (causal_t && t' > t).
    Tensor scores_t(n, n);
    scores_t.fill(0.0);
    for (size_t qi = 0; qi < H_; ++qi) {
        for (size_t qj = 0; qj < W_; ++qj) {
            for (size_t qt = 0; qt < T_; ++qt) {
                const size_t q_idx = flat_idx_3d(qi, qj, qt, W_, T_);
                for (size_t ki = 0; ki < H_; ++ki) {
                    for (size_t kj = 0; kj < W_; ++kj) {
                        for (size_t kt = 0; kt < T_; ++kt) {
                            const size_t k_idx = flat_idx_3d(ki, kj, kt, W_, T_);
                            if (ki != qi || kj != qj) {
                                scores_t[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            if (causal_t_ && kt > qt) {
                                scores_t[q_idx][k_idx] = -1e9;
                                continue;
                            }
                            double s = 0.0;
                            for (size_t d = 0; d < d_model_; ++d) {
                                s += Q_t[q_idx][d] * K_t[k_idx][d];
                            }
                            scores_t[q_idx][k_idx] = s * scale_;
                        }
                    }
                }
            }
        }
    }
    Tensor A_t = row_softmax(scores_t);
    last_attn_t_ = A_t;

    Tensor head_out_t(n, d_model_);
    head_out_t.fill(0.0);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += A_t[q][k] * V_t[k][j];
            head_out_t[q][j] = s;
        }
    }
    last_head_out_t_ = head_out_t;

    Tensor out_t_pre(n, d_model_);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_t[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += head_out_t[q][k] * W_o_t[k][j];
            out_t_pre[q][j] = s;
        }
    }
    last_out_t_ = out_t_pre;

    // ---- Residual fuse ----
    // output = out_h_pre + out_w_pre + out_t_pre + input
    Tensor output(n, d_model_);
    for (size_t q = 0; q < n; ++q) {
        for (size_t j = 0; j < d_model_; ++j) {
            output[q][j] = out_h_pre[q][j] + out_w_pre[q][j]
                         + out_t_pre[q][j] + input[q][j];
        }
    }
    return output;
}

Tensor ThreeDAxialAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    const size_t WT = W_ * T_;
    if (n != H_ * WT) {
        throw std::invalid_argument(
            "ThreeDAxialAttention.backward: grad_output.rows must equal H*W*T");
    }
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument(
            "ThreeDAxialAttention.backward: grad_output.cols must equal d_model");
    }

    // Each axis receives the full gradient (the residual sum splits it
    // equally; each pre-fuse term gets grad_output). Same convention as 2D.
    Tensor d_out_h = grad_output.clone();
    Tensor d_out_w = grad_output.clone();
    Tensor d_out_t = grad_output.clone();

    Tensor d_input(n, d_model_);
    d_input.fill(0.0);

    // ---- Backward through H axis ----
    {
        // d_head_out_h = d_out_h @ W_o_h^T  : (n, d_model)
        Tensor d_head_out_h(n, d_model_);
        for (size_t q = 0; q < n; ++q) {
            for (size_t k = 0; k < d_model_; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s += d_out_h[q][j] * W_o_h[k][j];
                d_head_out_h[q][k] = s;
            }
        }
        // grad_W_o_h[k, j] += Σ_q d_out_h[q, j] * head_out_h[q, k]
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t q = 0; q < n; ++q)
                    s += d_out_h[q][j] * last_head_out_h_[q][k];
                grad_W_o_h[k][j] += s;
            }
        }
        // grad_b_h[j] += Σ_q d_out_h[q, j]
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t q = 0; q < n; ++q) s += d_out_h[q][j];
            grad_b_h[0][j] += s;
        }

        // Per-head backward
        Tensor d_Q_h(n, d_model_); d_Q_h.fill(0.0);
        Tensor d_K_h(n, d_model_); d_K_h.fill(0.0);
        Tensor d_V_h(n, d_model_); d_V_h.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor d_V_ax(n, head_dim_);
            Tensor d_A_ax(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        s += last_attn_h_[j][i] * d_head_out_h[j][h_off + d];
                    d_V_ax[i][d] = s;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d)
                        s += d_head_out_h[i][h_off + d] * last_v_h_[j][h_off + d];
                    d_A_ax[i][j] = s;
                }
            }
            // Softmax backward: d_pre_soft = A * (d_A - row_sum(d_A * A))
            Tensor d_pre_soft(n, n);
            for (size_t i = 0; i < n; ++i) {
                double rsum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    rsum += last_attn_h_[i][j] * d_A_ax[i][j];
                for (size_t j = 0; j < n; ++j)
                    d_pre_soft[i][j] = last_attn_h_[i][j] * (d_A_ax[i][j] - rsum);
            }
            Tensor d_Q_ax(n, head_dim_);
            Tensor d_K_ax(n, head_dim_);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double qv = 0.0, kv = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        qv += d_pre_soft[i][j] * last_k_h_[j][h_off + d];
                        kv += d_pre_soft[j][i] * last_q_h_[j][h_off + d];
                    }
                    d_Q_ax[i][d] = qv * scale_;
                    d_K_ax[i][d] = kv * scale_;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q_h[i][h_off + d] += d_Q_ax[i][d];
                    d_K_h[i][h_off + d] += d_K_ax[i][d];
                    d_V_h[i][h_off + d] += d_V_ax[i][d];
                }
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t f = 0; f < d_model_; ++f) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    double x = last_input_[q][f];
                    sq += x * d_Q_h[q][j];
                    sk += x * d_K_h[q][j];
                    sv += x * d_V_h[q][j];
                }
                grad_W_q_h[j][f] += sq;
                grad_W_k_h[j][f] += sk;
                grad_W_v_h[j][f] += sv;
            }
        }
        for (size_t q = 0; q < n; ++q) {
            for (size_t f = 0; f < d_model_; ++f) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    s += d_Q_h[q][j] * W_q_h[j][f]
                       + d_K_h[q][j] * W_k_h[j][f]
                       + d_V_h[q][j] * W_v_h[j][f];
                }
                d_input[q][f] += s;
            }
        }
    }

    // ---- Backward through W axis ----
    {
        Tensor d_head_out_w(n, d_model_);
        for (size_t q = 0; q < n; ++q) {
            for (size_t k = 0; k < d_model_; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s += d_out_w[q][j] * W_o_w[k][j];
                d_head_out_w[q][k] = s;
            }
        }
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t q = 0; q < n; ++q)
                    s += d_out_w[q][j] * last_head_out_w_[q][k];
                grad_W_o_w[k][j] += s;
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t q = 0; q < n; ++q) s += d_out_w[q][j];
            grad_b_w[0][j] += s;
        }

        Tensor d_Q_w(n, d_model_); d_Q_w.fill(0.0);
        Tensor d_K_w(n, d_model_); d_K_w.fill(0.0);
        Tensor d_V_w(n, d_model_); d_V_w.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor d_V_ax(n, head_dim_);
            Tensor d_A_ax(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        s += last_attn_w_[j][i] * d_head_out_w[j][h_off + d];
                    d_V_ax[i][d] = s;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d)
                        s += d_head_out_w[i][h_off + d] * last_v_w_[j][h_off + d];
                    d_A_ax[i][j] = s;
                }
            }
            Tensor d_pre_soft(n, n);
            for (size_t i = 0; i < n; ++i) {
                double rsum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    rsum += last_attn_w_[i][j] * d_A_ax[i][j];
                for (size_t j = 0; j < n; ++j)
                    d_pre_soft[i][j] = last_attn_w_[i][j] * (d_A_ax[i][j] - rsum);
            }
            Tensor d_Q_ax(n, head_dim_);
            Tensor d_K_ax(n, head_dim_);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double qv = 0.0, kv = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        qv += d_pre_soft[i][j] * last_k_w_[j][h_off + d];
                        kv += d_pre_soft[j][i] * last_q_w_[j][h_off + d];
                    }
                    d_Q_ax[i][d] = qv * scale_;
                    d_K_ax[i][d] = kv * scale_;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q_w[i][h_off + d] += d_Q_ax[i][d];
                    d_K_w[i][h_off + d] += d_K_ax[i][d];
                    d_V_w[i][h_off + d] += d_V_ax[i][d];
                }
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t f = 0; f < d_model_; ++f) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    double x = last_input_[q][f];
                    sq += x * d_Q_w[q][j];
                    sk += x * d_K_w[q][j];
                    sv += x * d_V_w[q][j];
                }
                grad_W_q_w[j][f] += sq;
                grad_W_k_w[j][f] += sk;
                grad_W_v_w[j][f] += sv;
            }
        }
        for (size_t q = 0; q < n; ++q) {
            for (size_t f = 0; f < d_model_; ++f) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    s += d_Q_w[q][j] * W_q_w[j][f]
                       + d_K_w[q][j] * W_k_w[j][f]
                       + d_V_w[q][j] * W_v_w[j][f];
                }
                d_input[q][f] += s;
            }
        }
    }

    // ---- Backward through T (depth) axis ----
    {
        Tensor d_head_out_t(n, d_model_);
        for (size_t q = 0; q < n; ++q) {
            for (size_t k = 0; k < d_model_; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s += d_out_t[q][j] * W_o_t[k][j];
                d_head_out_t[q][k] = s;
            }
        }
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t q = 0; q < n; ++q)
                    s += d_out_t[q][j] * last_head_out_t_[q][k];
                grad_W_o_t[k][j] += s;
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t q = 0; q < n; ++q) s += d_out_t[q][j];
            grad_b_t[0][j] += s;
        }

        Tensor d_Q_t(n, d_model_); d_Q_t.fill(0.0);
        Tensor d_K_t(n, d_model_); d_K_t.fill(0.0);
        Tensor d_V_t(n, d_model_); d_V_t.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor d_V_ax(n, head_dim_);
            Tensor d_A_ax(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        s += last_attn_t_[j][i] * d_head_out_t[j][h_off + d];
                    d_V_ax[i][d] = s;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d)
                        s += d_head_out_t[i][h_off + d] * last_v_t_[j][h_off + d];
                    d_A_ax[i][j] = s;
                }
            }
            Tensor d_pre_soft(n, n);
            for (size_t i = 0; i < n; ++i) {
                double rsum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    rsum += last_attn_t_[i][j] * d_A_ax[i][j];
                for (size_t j = 0; j < n; ++j)
                    d_pre_soft[i][j] = last_attn_t_[i][j] * (d_A_ax[i][j] - rsum);
            }
            Tensor d_Q_ax(n, head_dim_);
            Tensor d_K_ax(n, head_dim_);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    double qv = 0.0, kv = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        qv += d_pre_soft[i][j] * last_k_t_[j][h_off + d];
                        kv += d_pre_soft[j][i] * last_q_t_[j][h_off + d];
                    }
                    d_Q_ax[i][d] = qv * scale_;
                    d_K_ax[i][d] = kv * scale_;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q_t[i][h_off + d] += d_Q_ax[i][d];
                    d_K_t[i][h_off + d] += d_K_ax[i][d];
                    d_V_t[i][h_off + d] += d_V_ax[i][d];
                }
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t f = 0; f < d_model_; ++f) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    double x = last_input_[q][f];
                    sq += x * d_Q_t[q][j];
                    sk += x * d_K_t[q][j];
                    sv += x * d_V_t[q][j];
                }
                grad_W_q_t[j][f] += sq;
                grad_W_k_t[j][f] += sk;
                grad_W_v_t[j][f] += sv;
            }
        }
        for (size_t q = 0; q < n; ++q) {
            for (size_t f = 0; f < d_model_; ++f) {
                double s = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    s += d_Q_t[q][j] * W_q_t[j][f]
                       + d_K_t[q][j] * W_k_t[j][f]
                       + d_V_t[q][j] * W_v_t[j][f];
                }
                d_input[q][f] += s;
            }
        }
    }

    // Add residual contribution.
    for (size_t q = 0; q < n; ++q) {
        for (size_t f = 0; f < d_model_; ++f) {
            d_input[q][f] += grad_output[q][f];
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// ThreeDAxialAttentionBlock
// ============================================================================

ThreeDAxialAttentionBlock::ThreeDAxialAttentionBlock(size_t d_model, size_t H,
                                                     size_t W, size_t T,
                                                     size_t num_heads,
                                                     size_t ffn_dim,
                                                     bool causal_h, bool causal_w, bool causal_t)
    : d_model_(d_model), ffn_dim_(ffn_dim),
      attn_(d_model, H, W, T, num_heads, causal_h, causal_w, causal_t),
      ln1_(d_model), ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim > 0 ? ffn_dim : 1),
      ffn_fc2_(ffn_dim > 0 ? ffn_dim : 1, d_model)
{
    if (d_model_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionBlock: d_model must be > 0");
    }
    if (H == 0 || W == 0 || T == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionBlock: H, W, T must be > 0");
    }
}

std::vector<Tensor*> ThreeDAxialAttentionBlock::parameters() {
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

std::vector<Tensor*> ThreeDAxialAttentionBlock::gradients() {
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

void ThreeDAxialAttentionBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void ThreeDAxialAttentionBlock::update_weights(double lr) {
    attn_.update_weights(lr);
    ln1_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor ThreeDAxialAttentionBlock::forward(const Tensor& input) {
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

Tensor ThreeDAxialAttentionBlock::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d_res1;
    if (ffn_dim_ > 0) {
        Tensor d_ffn_out = grad_output.clone();
        Tensor d_ffn_hidden = ffn_fc2_.backward(d_ffn_out, learning_rate);
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

    Tensor d_attn_out = d_res1.clone();
    Tensor d_input_from_res = d_res1.clone();

    Tensor d_z1 = attn_.backward(d_attn_out, learning_rate);

    Tensor d_input_from_ln1 = ln1_.backward(d_z1, learning_rate);

    Tensor d_input = d_input_from_res + d_input_from_ln1;
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// ThreeDAxialAttentionModel
// ============================================================================

ThreeDAxialAttentionModel::ThreeDAxialAttentionModel(size_t d_input, size_t d_model,
                                                     size_t d_output,
                                                     size_t H, size_t W, size_t T,
                                                     size_t num_blocks, size_t num_heads,
                                                     size_t ffn_dim,
                                                     bool causal_h, bool causal_w, bool causal_t)
    : d_input_(d_input), d_model_(d_model), d_output_(d_output),
      W_in_(d_input, d_model), W_out_(d_model, d_output),
      ln_final_(d_model),
      blocks_(num_blocks, ThreeDAxialAttentionBlock(d_model, H, W, T,
                                                    num_heads, ffn_dim,
                                                    causal_h, causal_w, causal_t))
{
    if (d_input_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionModel: d_input must be > 0");
    }
    if (d_model_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionModel: d_model must be > 0");
    }
    if (d_output_ == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionModel: d_output must be > 0");
    }
    if (num_blocks == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionModel: num_blocks must be > 0");
    }
    if (H == 0 || W == 0 || T == 0) {
        throw std::invalid_argument("ThreeDAxialAttentionModel: H, W, T must be > 0");
    }
}

std::vector<Tensor*> ThreeDAxialAttentionModel::parameters() {
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

std::vector<Tensor*> ThreeDAxialAttentionModel::gradients() {
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

void ThreeDAxialAttentionModel::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    ln_final_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
}

void ThreeDAxialAttentionModel::update_weights(double lr) {
    W_in_.update_weights(lr);
    W_out_.update_weights(lr);
    ln_final_.update_weights(lr);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor ThreeDAxialAttentionModel::forward(const Tensor& input) {
    if (input.cols != d_input_) {
        throw std::invalid_argument(
            "ThreeDAxialAttentionModel.forward: input cols must equal d_input");
    }
    last_input_ = input.clone();
    Tensor proj = W_in_.forward(input);
    last_proj_ = proj;
    for (auto& b : blocks_) proj = b.forward(proj);
    Tensor final = ln_final_.forward(proj);
    return W_out_.forward(final);
}

Tensor ThreeDAxialAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d_final = W_out_.backward(grad_output, learning_rate);
    Tensor d_proj  = ln_final_.backward(d_final, learning_rate);
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        d_proj = blocks_[i].backward(d_proj, learning_rate);
    }
    Tensor d_input = W_in_.backward(d_proj, learning_rate);
    return d_input;
}
