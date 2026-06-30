#include "alibi.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// ALiBi (Attention with Linear Biases) — Press et al. 2022
//   "Train Short, Test Long: Attention with Linear Biases Enables Input
//    Length Extrapolation"
// ============================================================================
//
// This file implements single-head and multi-head ALiBi self-attention.
// The ALiBi bias replaces standard positional encodings entirely.
//
// Math summary per head h (input x ∈ R^{n × d_model}, head_dim = d_model / n_h):
//
//   Q = x W_q^T + b_q  ∈ R^{n × d_model}
//   K = x W_k^T + b_k  ∈ R^{n × d_model}
//   V = x W_v^T + b_v  ∈ R^{n × d_model}
//   For each head h, slice the d_model dim into n_h blocks of head_dim:
//     Q_h = Q[:, h*head_dim : (h+1)*head_dim]    ∈ R^{n × head_dim}
//     K_h = K[:, h*head_dim : (h+1)*head_dim]
//     V_h = V[:, h*head_dim : (h+1)*head_dim]
//   scores_h = Q_h K_h^T / sqrt(head_dim)        ∈ R^{n × n}
//   scores_h[q, k] -= m_h * (q - k)               (ALiBi linear bias)
//   A_h      = row_softmax(scores_h)             ∈ R^{n × n}
//   out_h    = A_h @ V_h                          ∈ R^{n × head_dim}
//   out      = concat_h out_h @ W_o^T + b_o      ∈ R^{n × d_model}
//
// Slopes: m_h = 2^(-8/n_h * (h+1)), h = 0..n_h - 1.
//
// Backward: the bias is a CONSTANT w.r.t. softmax output (m_h is fixed).
// So softmax backward applies unchanged, and Q/K/V/O gradients are identical
// to standard scaled dot-product attention. The bias does NOT contribute
// any gradient to itself — no gradient flows to slopes_.
//
// Storage: since the Tensor class is 2D-only, we store per-head scores and
// attention weights as a flat (num_heads * n, n) 2D tensor, where row
// index `h * n + q` is the q-th query of head h.

namespace {

// Deterministic per-thread RNG for reproducibility.
static std::mt19937& alibi_global_rng() {
    static std::mt19937 gen(42);
    return gen;
}

// Row-wise softmax over the second axis of a 2D tensor. Returns a new tensor.
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

// Row-softmax backward: given dA (output gradient) and the softmax output A,
// compute dS where S = row-softmax(logits). The Jacobian for row i is:
//   dS_i[k] = A_i[k] * (dA_i[k] - sum_l A_i[l] * dA_i[l])
inline Tensor row_softmax_backward(const Tensor& dA, const Tensor& A) {
    Tensor dS(dA.rows, dA.cols);
    for (size_t i = 0; i < dA.rows; ++i) {
        double row_dot = 0.0;
        for (size_t j = 0; j < dA.cols; ++j) {
            row_dot += A(i, j) * dA(i, j);
        }
        for (size_t j = 0; j < dA.cols; ++j) {
            dS(i, j) = A(i, j) * (dA(i, j) - row_dot);
        }
    }
    return dS;
}

// Extract a column slice [c0, c0+width) from a (rows, cols) tensor.
inline Tensor col_slice(const Tensor& t, size_t c0, size_t width) {
    Tensor out(t.rows, width);
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < width; ++j)
            out(i, j) = t(i, c0 + j);
    return out;
}

// Write a column slice [c0, c0+width) of `src` into column slice [c0, c0+width)
// of `dst` (in-place, accumulates).
inline void col_slice_add(Tensor& dst, const Tensor& src, size_t c0) {
    for (size_t i = 0; i < src.rows; ++i)
        for (size_t j = 0; j < src.cols; ++j)
            dst(i, c0 + j) += src(i, j);
}

} // namespace

// ----------------------------------------------------------------------------
// AlibiAttention
// ----------------------------------------------------------------------------

AlibiAttention::AlibiAttention(size_t d_model, size_t seq_len, size_t num_heads)
    : d_model_(d_model),
      seq_len_(seq_len),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? d_model : d_model / num_heads),
      W_q(d_model, d_model), W_k(d_model, d_model),
      W_v(d_model, d_model), W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model), b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model),
      grad_b_v(1, d_model), grad_b_o(1, d_model),
      slopes_(1, num_heads == 0 ? 1 : num_heads),
      alibi_bias_(num_heads == 0 ? 1 : num_heads, seq_len * seq_len),
      last_input_(0, 0), last_q_(0, 0), last_k_(0, 0), last_v_(0, 0),
      last_scores_(0, 0), last_attn_(0, 0), last_out_(0, 0)
{
    if (num_heads == 0) {
        throw std::invalid_argument("AlibiAttention: num_heads must be > 0");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("AlibiAttention: d_model must be divisible by num_heads");
    }
    if (seq_len == 0) {
        throw std::invalid_argument("AlibiAttention: seq_len must be > 0");
    }

    // Initialize Q/K/V/O with Xavier
    auto xavier_init = [&](Tensor& W, size_t fan_in, size_t fan_out) {
        double std = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) = dis(alibi_global_rng());
    };
    xavier_init(W_q, d_model, d_model);
    xavier_init(W_k, d_model, d_model);
    xavier_init(W_v, d_model, d_model);
    xavier_init(W_o, d_model, d_model);

    b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);

    // ----------------------------------------------------------------------
    // ALiBi slopes: m_h = 2^(-8 / n_h * (h + 1))
    // ----------------------------------------------------------------------
    for (size_t h = 0; h < num_heads_; ++h) {
        double exponent = -8.0 / static_cast<double>(num_heads_) * static_cast<double>(h + 1);
        slopes_(0, h) = std::pow(2.0, exponent);
    }

    // ----------------------------------------------------------------------
    // Precompute the ALiBi bias as (num_heads, seq_len*seq_len) flat storage.
    //   Row h*seq_len*seq_len + q*seq_len + k gives bias_h[q, k] = -m_h * (q - k)
    // ----------------------------------------------------------------------
    for (size_t h = 0; h < num_heads_; ++h) {
        double m = slopes_(0, h);
        for (size_t q = 0; q < seq_len_; ++q) {
            for (size_t k = 0; k < seq_len_; ++k) {
                alibi_bias_(h, q * seq_len_ + k) = -m * (static_cast<double>(q) - static_cast<double>(k));
            }
        }
    }
}

Tensor AlibiAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (n != seq_len_) {
        throw std::invalid_argument("AlibiAttention: input.rows != seq_len_");
    }
    last_input_ = input.clone();

    // Q/K/V/O projections — Dense convention: y = x @ W^T + b
    auto project = [&](const Tensor& x, const Tensor& W, const Tensor& b) {
        Tensor y = x * W.transpose();
        for (size_t j = 0; j < y.cols; ++j) {
            double bj = b(0, j);
            for (size_t i = 0; i < y.rows; ++i) y(i, j) += bj;
        }
        return y;
    };

    last_q_ = project(input, W_q, b_q);  // (n, d_model)
    last_k_ = project(input, W_k, b_k);
    last_v_ = project(input, W_v, b_v);

    // ----------------------------------------------------------------------
    // Per-head scores: scores_h = Q_h K_h^T / sqrt(head_dim) - m_h * (q - k)
    // Stored as (num_heads * n, n) 2D tensor. Row index h * n + q.
    // ----------------------------------------------------------------------
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    last_scores_ = Tensor(num_heads_ * n, n);
    last_attn_   = Tensor(num_heads_ * n, n);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t c0 = h * head_dim_;
        const size_t row_off = h * n;
        const size_t bias_off = h * seq_len_ * seq_len_;

        // scores_h[q, k] = sum_j Q_h[q, j] * K_h[k, j] * scale - m_h * (q - k)
        for (size_t q = 0; q < n; ++q) {
            for (size_t k = 0; k < n; ++k) {
                double dot = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    dot += last_q_(q, c0 + j) * last_k_(k, c0 + j);
                }
                last_scores_(row_off + q, k) = dot * scale + alibi_bias_(0, bias_off + q * seq_len_ + k);
            }
        }

        // row-softmax over k (per query)
        for (size_t q = 0; q < n; ++q) {
            double row_max = last_scores_(row_off + q, 0);
            for (size_t k = 1; k < n; ++k)
                if (last_scores_(row_off + q, k) > row_max) row_max = last_scores_(row_off + q, k);
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                double e = std::exp(last_scores_(row_off + q, k) - row_max);
                last_attn_(row_off + q, k) = e;
                sum += e;
            }
            double inv = 1.0 / (sum + 1e-12);
            for (size_t k = 0; k < n; ++k) last_attn_(row_off + q, k) *= inv;
        }
    }

    // ----------------------------------------------------------------------
    // Per-head output assembly: out_h = A_h @ V_h, place into pre_o[:, h*head_dim : (h+1)*head_dim]
    // ----------------------------------------------------------------------
    Tensor pre_o(n, d_model_);
    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t c0 = h * head_dim_;
        const size_t row_off = h * n;
        for (size_t q = 0; q < n; ++q) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    s += last_attn_(row_off + q, k) * last_v_(k, c0 + j);
                }
                pre_o(q, c0 + j) = s;
            }
        }
    }
    last_out_ = pre_o.clone();

    // Final projection: out = pre_o @ W_o^T + b_o
    Tensor out = pre_o * W_o.transpose();
    for (size_t j = 0; j < out.cols; ++j) {
        double bj = b_o(0, j);
        for (size_t i = 0; i < out.rows; ++i) out(i, j) += bj;
    }
    return out;
}

Tensor AlibiAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (n, d_model)
    const size_t n = seq_len_;

    // ---- 1. dL/d(pre_o) = grad_output @ W_o ----
    Tensor d_pre_o(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                s += grad_output(i, k) * W_o(k, j);
            }
            d_pre_o(i, j) = s;
        }
    }

    // Accumulate per-head dQ_h/dK_h/dV_h into dQ/dK/dV (shape (n, d_model_))
    Tensor dQ(n, d_model_);
    Tensor dK(n, d_model_);
    Tensor dV(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            dQ(i, j) = 0.0;
            dK(i, j) = 0.0;
            dV(i, j) = 0.0;
        }

    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t c0 = h * head_dim_;
        const size_t row_off = h * n;

        // dV_h[k, j] = sum_q d_pre_o_h[q, j] * A_h[q, k]
        Tensor dV_h(n, head_dim_);
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double s = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    s += last_attn_(row_off + q, k) * d_pre_o(q, c0 + j);
                }
                dV_h(k, j) = s;
            }
        }
        col_slice_add(dV, dV_h, c0);

        // dA_h[q, k] = sum_j d_pre_o_h[q, j] * V_h[k, j]
        // Build dA_h as a 2D (n, n) view via temporary
        Tensor dA_h(n, n);
        for (size_t q = 0; q < n; ++q) {
            for (size_t k = 0; k < n; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    s += d_pre_o(q, c0 + j) * last_v_(k, c0 + j);
                }
                dA_h(q, k) = s;
            }
        }

        // Extract the corresponding slice of last_attn_ for this head (n, n)
        Tensor A_h(n, n);
        for (size_t q = 0; q < n; ++q)
            for (size_t k = 0; k < n; ++k)
                A_h(q, k) = last_attn_(row_off + q, k);

        // softmax backward: dscores_h = row_softmax_backward(dA_h, A_h)
        Tensor dscores_h = row_softmax_backward(dA_h, A_h);

        // dQ_h[q, j] = scale * sum_k dscores_h[q, k] * K_h[k, j]
        Tensor dQ_h(n, head_dim_);
        for (size_t q = 0; q < n; ++q) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    s += dscores_h(q, k) * last_k_(k, c0 + j);
                }
                dQ_h(q, j) = s * scale;
            }
        }
        col_slice_add(dQ, dQ_h, c0);

        // dK_h[k, j] = scale * sum_q dscores_h[q, k] * Q_h[q, j]
        Tensor dK_h(n, head_dim_);
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double s = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    s += dscores_h(q, k) * last_q_(q, c0 + j);
                }
                dK_h(k, j) = s * scale;
            }
        }
        col_slice_add(dK, dK_h, c0);
    }

    // ---- 3. dW_q/dW_k/dW_v/dW_o via Dense convention ----
    //   dW[k, j] += sum_i dY[i, k] * X[i, j]
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_q(k, j) += last_input_(i, j) * dQ(i, k);
                grad_W_k(k, j) += last_input_(i, j) * dK(i, k);
                grad_W_v(k, j) += last_input_(i, j) * dV(i, k);
                grad_W_o(k, j) += last_out_(i, j)  * d_pre_o(i, k);
            }
        }
        for (size_t k = 0; k < d_model_; ++k) {
            grad_b_q(0, k) += dQ(i, k);
            grad_b_k(0, k) += dK(i, k);
            grad_b_v(0, k) += dV(i, k);
            grad_b_o(0, k) += d_pre_o(i, k);
        }
    }

    // ---- 4. dX = dQ @ W_q + dK @ W_k + dV @ W_v (Dense convention) ----
    Tensor dX(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s_q = 0.0, s_k = 0.0, s_v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                s_q += dQ(i, k) * W_q(k, j);
                s_k += dK(i, k) * W_k(k, j);
                s_v += dV(i, k) * W_v(k, j);
            }
            dX(i, j) = s_q + s_k + s_v;
        }
    }
    return dX;
}

void AlibiAttention::update_weights(double learning_rate) {
    auto sgd_update = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) -= learning_rate * gW(i, j);
    };
    auto sgd_update_b = [&](Tensor& b, const Tensor& gb) {
        for (size_t j = 0; j < b.cols; ++j)
            b(0, j) -= learning_rate * gb(0, j);
    };
    sgd_update(W_q, grad_W_q); sgd_update(W_k, grad_W_k);
    sgd_update(W_v, grad_W_v); sgd_update(W_o, grad_W_o);
    sgd_update_b(b_q, grad_b_q); sgd_update_b(b_k, grad_b_k);
    sgd_update_b(b_v, grad_b_v); sgd_update_b(b_o, grad_b_o);
}

void AlibiAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);
}

std::vector<Tensor*> AlibiAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &b_q, &b_k, &b_v, &b_o};
}

std::vector<Tensor*> AlibiAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o,
            &grad_b_q, &grad_b_k, &grad_b_v, &grad_b_o};
}

// ----------------------------------------------------------------------------
// AlibiBlock
// ----------------------------------------------------------------------------

AlibiBlock::AlibiBlock(size_t d_model, size_t seq_len, size_t num_heads)
    : d_model_(d_model),
      num_heads_(num_heads),
      attn(d_model, seq_len, num_heads),
      ln1(d_model),
      ln2(d_model),
      W1(d_model * 4, d_model), b1(1, d_model * 4),
      W2(d_model, d_model * 4),     b2(1, d_model),
      grad_W1(d_model * 4, d_model), grad_b1(1, d_model * 4),
      grad_W2(d_model, d_model * 4), grad_b2(1, d_model)
{
    std::mt19937 gen(42);
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

Tensor AlibiBlock::forward(const Tensor& input) {
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
    Tensor ffn_gelu = last_ffn_pregelu_.apply(GELU());
    last_ffn_out_ = ffn_gelu * W2.transpose();
    for (size_t j = 0; j < last_ffn_out_.cols; ++j) {
        double bj = b2(0, j);
        for (size_t i = 0; i < last_ffn_out_.rows; ++i) last_ffn_out_(i, j) += bj;
    }

    Tensor out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            out(t, j) = last_resid1_(t, j) + last_ffn_out_(t, j);
        }
    }
    return out;
}

Tensor AlibiBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;

    // ---- Sub-block 2 backward ----
    Tensor grad_resid1(n, d);
    Tensor grad_ffn_out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            grad_resid1(t, j) = grad_output(t, j);
            grad_ffn_out(t, j) = grad_output(t, j);
        }
    }

    GELU gelu;
    Tensor grad_pre(n, d * 4);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d * 4; ++k) {
            double s = 0.0;
            for (size_t jj = 0; jj < d; ++jj) s += grad_ffn_out(t, jj) * W2(jj, k);
            grad_pre(t, k) = s * gelu.derivative(last_ffn_pregelu_(t, k));
        }
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d * 4; ++j) {
                double gp_val = gelu(last_ffn_pregelu_(t, j));
                grad_W2(i, j) += grad_ffn_out(t, i) * gp_val;
            }
            grad_b2(0, i) += grad_ffn_out(t, i);
        }
    }
    Tensor grad_ln2_out(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d * 4; ++k) s += grad_pre(t, k) * W1(k, j);
            grad_ln2_out(t, j) = s;
        }
        for (size_t k = 0; k < d * 4; ++k) {
            for (size_t j = 0; j < d; ++j) {
                grad_W1(k, j) += grad_pre(t, k) * last_ln2_out_(t, j);
            }
            grad_b1(0, k) += grad_pre(t, k);
        }
    }

    Tensor grad_resid1_ln = ln2.backward(grad_ln2_out, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) grad_resid1(t, j) += grad_resid1_ln(t, j);

    // ---- Sub-block 1 backward ----
    Tensor grad_attn_out = grad_resid1;
    Tensor grad_ln1_out = attn.backward(grad_attn_out, 0.0);
    Tensor grad_x_from_resid(n, d);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            grad_x_from_resid(t, j) = grad_resid1(t, j);

    Tensor grad_x_from_ln1 = ln1.backward(grad_ln1_out, 0.0);
    Tensor grad_input(n, d);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            grad_input(t, j) = grad_x_from_resid(t, j) + grad_x_from_ln1(t, j);
    return grad_input;
}

void AlibiBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    for (size_t i = 0; i < W1.rows; ++i)
        for (size_t j = 0; j < W1.cols; ++j) W1(i, j) -= learning_rate * grad_W1(i, j);
    for (size_t j = 0; j < b1.cols; ++j) b1(0, j) -= learning_rate * grad_b1(0, j);
    for (size_t i = 0; i < W2.rows; ++i)
        for (size_t j = 0; j < W2.cols; ++j) W2(i, j) -= learning_rate * grad_W2(i, j);
    for (size_t j = 0; j < b2.cols; ++j) b2(0, j) -= learning_rate * grad_b2(0, j);
}

void AlibiBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W1.fill(0.0); grad_W2.fill(0.0);
    grad_b1.fill(0.0); grad_b2.fill(0.0);
}

std::vector<Tensor*> AlibiBlock::parameters() {
    auto p = attn.parameters();
    auto l1 = ln1.parameters();
    auto l2 = ln2.parameters();
    p.insert(p.end(), l1.begin(), l1.end());
    p.insert(p.end(), l2.begin(), l2.end());
    p.push_back(&W1); p.push_back(&b1);
    p.push_back(&W2); p.push_back(&b2);
    return p;
}

std::vector<Tensor*> AlibiBlock::gradients() {
    auto g = attn.gradients();
    auto l1 = ln1.gradients();
    auto l2 = ln2.gradients();
    g.insert(g.end(), l1.begin(), l1.end());
    g.insert(g.end(), l2.begin(), l2.end());
    g.push_back(&grad_W1); g.push_back(&grad_b1);
    g.push_back(&grad_W2); g.push_back(&grad_b2);
    return g;
}

// ----------------------------------------------------------------------------
// AlibiModel
// ----------------------------------------------------------------------------

AlibiModel::AlibiModel(size_t d_model, size_t seq_len, size_t out_features,
                       size_t num_blocks, size_t num_heads)
    : d_model_(d_model),
      num_blocks_(num_blocks),
      out_features_(out_features),
      blocks_(),
      final_ln_(d_model),
      W_out_(out_features, d_model), b_out_(1, out_features),
      grad_W_out_(out_features, d_model), grad_b_out_(1, out_features),
      W_in_(d_model, d_model), b_in_(1, d_model),
      grad_W_in_(d_model, d_model), grad_b_in_(1, d_model)
{
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, seq_len, num_heads);
    }
    std::mt19937 gen(42);
    double std_in = std::sqrt(2.0 / (d_model + d_model));
    std::normal_distribution<> dis_in(0.0, std_in);
    for (size_t i = 0; i < W_in_.rows; ++i)
        for (size_t j = 0; j < W_in_.cols; ++j) W_in_(i, j) = dis_in(gen);
    double std_out = std::sqrt(2.0 / (d_model + out_features));
    std::normal_distribution<> dis_out(0.0, std_out);
    for (size_t i = 0; i < W_out_.rows; ++i)
        for (size_t j = 0; j < W_out_.cols; ++j) W_out_(i, j) = dis_out(gen);
    b_in_.fill(0.0); b_out_.fill(0.0);
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
}

Tensor AlibiModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    const size_t n = input.rows;

    last_in_proj_ = input * W_in_.transpose();
    for (size_t j = 0; j < last_in_proj_.cols; ++j) {
        double bj = b_in_(0, j);
        for (size_t i = 0; i < last_in_proj_.rows; ++i) last_in_proj_(i, j) += bj;
    }

    Tensor h = last_in_proj_;
    for (auto& block : blocks_) {
        h = block.forward(h);
    }

    last_final_ln_ = final_ln_.forward(h);

    last_logits_ = last_final_ln_ * W_out_.transpose();
    for (size_t j = 0; j < last_logits_.cols; ++j) {
        double bj = b_out_(0, j);
        for (size_t i = 0; i < last_logits_.rows; ++i) last_logits_(i, j) += bj;
    }
    return last_logits_;
}

Tensor AlibiModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;

    Tensor d_ln_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < out_features_; ++k) s += grad_output(i, k) * W_out_(k, j);
            d_ln_out(i, j) = s;
        }
        for (size_t k = 0; k < out_features_; ++k) {
            grad_b_out_(0, k) += grad_output(i, k);
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_out_(k, j) += grad_output(i, k) * last_final_ln_(i, j);
            }
        }
    }

    Tensor d_blocks_out = final_ln_.backward(d_ln_out, 0.0);
    Tensor d_h = d_blocks_out;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_h = it->backward(d_h, 0.0);
    }

    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) s += d_h(i, k) * W_in_(k, j);
            d_input(i, j) = s;
        }
        for (size_t k = 0; k < d_model_; ++k) {
            grad_b_in_(0, k) += d_h(i, k);
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_in_(k, j) += d_h(i, k) * last_input_(i, j);
            }
        }
    }
    return d_input;
}

void AlibiModel::update_weights(double learning_rate) {
    for (auto& block : blocks_) block.update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    for (size_t i = 0; i < W_in_.rows; ++i)
        for (size_t j = 0; j < W_in_.cols; ++j) W_in_(i, j) -= learning_rate * grad_W_in_(i, j);
    for (size_t j = 0; j < b_in_.cols; ++j) b_in_(0, j) -= learning_rate * grad_b_in_(0, j);
    for (size_t i = 0; i < W_out_.rows; ++i)
        for (size_t j = 0; j < W_out_.cols; ++j) W_out_(i, j) -= learning_rate * grad_W_out_(i, j);
    for (size_t j = 0; j < b_out_.cols; ++j) b_out_(0, j) -= learning_rate * grad_b_out_(0, j);
}

void AlibiModel::zero_grad() {
    for (auto& block : blocks_) block.zero_grad();
    final_ln_.zero_grad();
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
}

std::vector<Tensor*> AlibiModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& block : blocks_) {
        auto bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fl = final_ln_.parameters();
    p.insert(p.end(), fl.begin(), fl.end());
    p.push_back(&W_in_); p.push_back(&b_in_);
    p.push_back(&W_out_); p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> AlibiModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& block : blocks_) {
        auto bg = block.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fl = final_ln_.gradients();
    g.insert(g.end(), fl.begin(), fl.end());
    g.push_back(&grad_W_in_); g.push_back(&grad_b_in_);
    g.push_back(&grad_W_out_); g.push_back(&grad_b_out_);
    return g;
}