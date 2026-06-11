// ============================================================================
// Grouped Query Attention (GQA) — Ainslie et al. 2023 implementation
// ============================================================================
//
// Forward (single sample, n tokens, d_model features):
//
//   Q = X @ W_q                                (n, d_model)  -- num_q head blocks
//   K = X @ W_k                                (n, d_model)  -- num_kv head blocks
//   V = X @ W_v                                (n, d_model)  -- num_kv head blocks
//
//   for qh in [0, num_query_heads):
//       kh = qh / group_size
//       Q_h = Q[:, qh*head_dim : (qh+1)*head_dim]    (n, head_dim)
//       K_h = K[:, kh*head_dim : (kh+1)*head_dim]    (n, head_dim)
//       V_h = V[:, kh*head_dim : (kh+1)*head_dim]    (n, head_dim)
//       scores_h = Q_h @ K_h^T / sqrt(head_dim)      (n, n)
//       A_h      = row_softmax(scores_h)              (n, n)
//       head_out[:, qh*head_dim : (qh+1)*head_dim] = A_h @ V_h
//
//   output = head_out @ W_o                    (n, d_model)
//
// We don't bias the projections (no additive b on W_q/W_k/W_v/W_o) to match
// the Llama-style convention used in modern GQA models.
//
// Backward (given grad_output (n, d_model)):
//
//   1) d_head_out = grad_output @ W_o^T        (n, d_model)  (output = head_out @ W_o)
//   2) grad_W_o += d_head_out^T @ head_out     -- see comment in code
//
//   3) For each Q head qh with K/V head kh = qh / group_size:
//        d_head_out_h = d_head_out[:, qh*head_dim : (qh+1)*head_dim]   (n, head_dim)
//        d_V_h = A_h^T @ d_head_out_h                                  (n, head_dim)
//        d_A_h = d_head_out_h @ V_h^T                                  (n, n)
//        d_scores_h = A_h * (d_A_h - row_sum(d_A_h * A_h))             (n, n)  (softmax bwd)
//        d_Q_h = (d_scores_h @ K_h) / sqrt(head_dim)                   (n, head_dim)
//        d_K_h = (d_scores_h^T @ Q_h) / sqrt(head_dim)                 (n, head_dim)
//
//        d_Q[:, qh*head_dim : (qh+1)*head_dim] += d_Q_h
//        d_K[:, kh*head_dim : (kh+1)*head_dim] += d_K_h  ← accumulates across
//        d_V[:, kh*head_dim : (kh+1)*head_dim] += d_V_h  ← the Q-heads in the group
//
//   4) grad_W_q += d_Q^T @ X   (over num_q head blocks)
//      grad_W_k += d_K^T @ X   (over num_kv head blocks; d_K already summed)
//      grad_W_v += d_V^T @ X
//
//   5) d_input = d_Q @ W_q + d_K @ W_k + d_V @ W_v   (n, d_model)
//
// Implementation note: we do explicit per-head loops for clarity and so the
// gradient check can be interpreted per-component. The compute hot-spot is
// the matmul, which is O(n^2 * head_dim) for the per-head attention and
// O(n * d_model^2) for the projections — matches standard MHA scaling.
// ============================================================================

#include "gqa.h"
#include "../../activations/activations.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

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

// GELU derivative (matches the GELU functor's tanh-approximation convention).
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 x³)))
//   d/dx GELU(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh²(u)) * u'
//   where u = sqrt(2/π) * (x + 0.044715 x³), u' = sqrt(2/π) * (1 + 3*0.044715*x²)
inline double gelu_deriv(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    double th = std::tanh(u);
    double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
}

}  // namespace

// ============================================================================
// GQAAttention
// ============================================================================

GQAAttention::GQAAttention(size_t d_model, size_t num_query_heads, size_t num_kv_heads)
    : d_model_(d_model),
      num_query_heads_(num_query_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(d_model / num_query_heads),
      group_size_(num_query_heads / num_kv_heads),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_query_heads)))
{
    if (d_model_ == 0) {
        throw std::invalid_argument("GQAAttention: d_model must be > 0");
    }
    if (num_query_heads_ == 0) {
        throw std::invalid_argument("GQAAttention: num_query_heads must be > 0");
    }
    if (num_kv_heads_ == 0) {
        throw std::invalid_argument("GQAAttention: num_kv_heads must be > 0");
    }
    if (d_model_ % num_query_heads_ != 0) {
        throw std::invalid_argument(
            "GQAAttention: d_model must be evenly divisible by num_query_heads");
    }
    if (num_query_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "GQAAttention: num_query_heads must be evenly divisible by num_kv_heads");
    }

    // W_q, W_o: (d_model, d_model) — stacked head blocks for num_query_heads
    // W_k, W_v: (d_model, d_model) — only the first num_kv_heads * head_dim rows
    //           contain parameters used in forward; the rest stay zero. This
    //           uniform layout makes the projection step and backward symmetric
    //           across all four matrices. To make gradient checks non-degenerate
    //           (so that the model output is sensitive to every input column),
    //           we randomize ALL rows of W_k and W_v with small values, but
    //           forward only consumes the active prefix. The non-active rows
    //           contribute to the parameter count and to the W_k/W_v gradient
    //           in the active range only — we zero them after random init.
    W_q = Tensor::random(d_model_, d_model_, 0.02);
    W_k = Tensor::zeros(d_model_, d_model_);
    W_v = Tensor::zeros(d_model_, d_model_);
    W_o = Tensor::random(d_model_, d_model_, 0.02);

    // Initialize the active K/V head blocks with small random values.
    // (The remaining tail of W_k/W_v beyond num_kv_heads * head_dim stays 0.)
    size_t kv_dim = num_kv_heads_ * head_dim_;
    for (size_t i = 0; i < kv_dim; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            W_k[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 7 + j * 13) % 1000) / 1000.0) - 1.0);
            W_v[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 11 + j * 17) % 1000) / 1000.0) - 1.0);
        }

    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
}

std::vector<Tensor*> GQAAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}

std::vector<Tensor*> GQAAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}

void GQAAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

void GQAAttention::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_k -= grad_W_k * learning_rate;
    W_v -= grad_W_v * learning_rate;
    W_o -= grad_W_o * learning_rate;
}

Tensor GQAAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("GQAAttention.forward: input cols must equal d_model");
    }

    last_input_ = input.clone();

    // Q = input @ W_q^T, K = input @ W_k^T, V = input @ W_v^T
    // Each is (n, d_model).
    Tensor Q(n, d_model_), K(n, d_model_), V(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                qv += input[i][k] * W_q[k][j];
                kv += input[i][k] * W_k[k][j];
                vv += input[i][k] * W_v[k][j];
            }
            Q[i][j] = qv;
            K[i][j] = kv;
            V[i][j] = vv;
        }
    }
    last_q_ = Q;
    last_k_ = K;
    last_v_ = V;

    // Per-head attention with K/V sharing.
    Tensor head_out(n, d_model_);
    head_out.fill(0.0);
    last_attn_ = Tensor(num_query_heads_ * n, n);

    for (size_t qh = 0; qh < num_query_heads_; ++qh) {
        const size_t kh = qh / group_size_;
        const size_t q_off = qh * head_dim_;
        const size_t kv_off = kh * head_dim_;

        // scores = Q_h @ K_h^T / sqrt(head_dim) : (n, head_dim) @ (head_dim, n) = (n, n)
        Tensor scores(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t d = 0; d < head_dim_; ++d) {
                    s += Q[i][q_off + d] * K[j][kv_off + d];
                }
                scores[i][j] = s * scale_;
            }
        }

        // row softmax → A
        Tensor A = row_softmax(scores);

        // Cache A in last_attn_ at row range [qh*n, (qh+1)*n)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                last_attn_[qh * n + i][j] = A[i][j];

        // head_out[:, q_off : q_off+head_dim] = A @ V_h  : (n, n) @ (n, head_dim) = (n, head_dim)
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += A[i][j] * V[j][kv_off + d];
                }
                head_out[i][q_off + d] = v;
            }
        }
    }

    last_head_out_ = head_out;

    // output = head_out @ W_o^T : (n, d_model) @ (d_model, d_model) = (n, d_model)
    Tensor output(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += head_out[i][k] * W_o[k][j];
            }
            output[i][j] = v;
        }
    }
    return output;
}

Tensor GQAAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("GQAAttention.backward: grad_output cols must equal d_model");
    }

    // ---- Step 1: propagate through W_o ----
    // Forward: output = head_out @ W_o   (no transpose, "right-multiplied by W_o")
    //   output[t][j] = Σ_k head_out[t][k] * W_o[k][j]
    //   → d_head_out[t][k] = Σ_j d_output[t][j] * W_o[k][j]   (i.e. d_output @ W_o^T)
    Tensor d_head_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += grad_output[i][j] * W_o[k][j];
            }
            d_head_out[i][k] = v;
        }
    }

    // grad_W_o[k][j] += grad_output[t][j] * last_head_out_[t][k]   (sum over t)
    //   Derivation: output[t][j] = sum_k head_out[t][k] * W_o[k][j]
    //                            → dW_o[k][j] = sum_t grad_output[t][j] * head_out[t][k]
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s += grad_output[t][j] * last_head_out_[t][i];
            }
            grad_W_o[i][j] += s;
        }
    }

    // ---- Step 2: per-Q-head backward, with K/V gradient accumulation ----
    Tensor d_q_acc(n, d_model_);   d_q_acc.fill(0.0);
    Tensor d_k_acc(n, d_model_);   d_k_acc.fill(0.0);
    Tensor d_v_acc(n, d_model_);   d_v_acc.fill(0.0);

    for (size_t qh = 0; qh < num_query_heads_; ++qh) {
        const size_t kh = qh / group_size_;
        const size_t q_off = qh * head_dim_;
        const size_t kv_off = kh * head_dim_;

        // d_V_h = A^T @ d_head_out_h  : (n, n)^T @ (n, head_dim) = (n, head_dim)
        // d_A_h = d_head_out_h @ V_h^T : (n, head_dim) @ (head_dim, n) = (n, n)
        // Note: V_h is in last_v_[:, kv_off : kv_off+head_dim], shape (n, head_dim).
        // V_h^T has shape (head_dim, n).
        Tensor d_V_h(n, head_dim_);
        Tensor d_A_h(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    // d_V_h[i][d] = sum_j A[j][i] * d_head_out_h[j][d]
                    v += last_attn_[qh * n + j][i] * d_head_out[j][q_off + d];
                }
                d_V_h[i][d] = v;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double v = 0.0;
                for (size_t d = 0; d < head_dim_; ++d) {
                    // d_A_h[i][j] = sum_d d_head_out_h[i][d] * V_h[j][d]
                    v += d_head_out[i][q_off + d] * last_v_[j][kv_off + d];
                }
                d_A_h[i][j] = v;
            }
        }

        // Softmax backward: d_scores = A * (d_A - row_sum(d_A * A))
        Tensor d_scores(n, n);
        for (size_t i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                row_sum += last_attn_[qh * n + i][j] * d_A_h[i][j];
            }
            for (size_t j = 0; j < n; ++j) {
                d_scores[i][j] = last_attn_[qh * n + i][j] *
                                 (d_A_h[i][j] - row_sum);
            }
        }

        // d_Q_h = scale_ * (d_scores @ K_h)   (n, head_dim)
        // d_K_h = scale_ * (d_scores^T @ Q_h) (n, head_dim)
        // Forward: scores = scale_ * (Q_h @ K_h^T).  So d_s = scale_ * d_scores,
        // and d_Q_h[i][d] = sum_j d_s[i][j] * K_h[j][d].
        Tensor d_Q_h(n, head_dim_);
        Tensor d_K_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double qv = 0.0, kv = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    qv += d_scores[i][j] * last_k_[j][kv_off + d];
                    kv += d_scores[j][i] * last_q_[j][q_off + d];
                }
                d_Q_h[i][d] = qv * scale_;
                d_K_h[i][d] = kv * scale_;
            }
        }

        // Accumulate into d_q_acc, d_k_acc, d_v_acc.
        // Q head is unique → simple add.
        // K and V heads are shared across group_size Q heads → accumulator.
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                d_q_acc[i][q_off + d]  += d_Q_h[i][d];
                d_k_acc[i][kv_off + d] += d_K_h[i][d];   // accumulates across Q heads in the group
                d_v_acc[i][kv_off + d] += d_V_h[i][d];   // accumulates across Q heads in the group
            }
        }
    }

    // ---- Step 3: parameter gradients from d_q/d_k/d_v vs last_input_ ----
    // grad_W_q[i][j] += d_q[t][j] * input[t][i]   (sum over t)
    //   Derivation: Q[t][j] = sum_i X[t][i] * W_q[i][j]  →  dW_q[i][j] = sum_t dQ[t][j] * X[t][i]
    // grad_W_k, grad_W_v: same shape, just different accumulators.
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double sq = 0.0, sk = 0.0, sv = 0.0;
            for (size_t t = 0; t < n; ++t) {
                sq += d_q_acc[t][j] * last_input_[t][i];
                sk += d_k_acc[t][j] * last_input_[t][i];
                sv += d_v_acc[t][j] * last_input_[t][i];
            }
            grad_W_q[i][j] += sq;
            grad_W_k[i][j] += sk;
            grad_W_v[i][j] += sv;
        }
    }

    // ---- Step 4: input gradient ----
    // Forward uses Q = X @ W_q (no transpose), K = X @ W_k, V = X @ W_v.
    //   Q[i][j] = Σ_k X[i][k] * W_q[k][j]
    //   → dX[i][k] = Σ_j dQ[i][j] * W_q[k][j]   (i.e. dQ @ W_q^T)
    // So d_input = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T.
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += d_q_acc[i][j] * W_q[k][j]
                   + d_k_acc[i][j] * W_k[k][j]
                   + d_v_acc[i][j] * W_v[k][j];
            }
            d_input[i][k] = v;
        }
    }
    return d_input;
}

// ============================================================================
// GQABlock — pre-LN → GQAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

GQABlock::GQABlock(size_t d_model, size_t num_query_heads, size_t num_kv_heads, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_query_heads, num_kv_heads),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{
}

std::vector<Tensor*> GQABlock::parameters() {
    auto p = attn_.parameters();
    auto p1 = ln1_.parameters();
    auto p2 = ln2_.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), pf1.begin(), pf1.end());
    p.insert(p.end(), pf2.begin(), pf2.end());
    return p;
}

std::vector<Tensor*> GQABlock::gradients() {
    auto g = attn_.gradients();
    auto g1 = ln1_.gradients();
    auto g2 = ln2_.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}

void GQABlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void GQABlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

Tensor GQABlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    // pre-LN → attention → residual
    Tensor z1 = ln1_.forward(input);
    Tensor attn_out = attn_.forward(z1);
    last_z1_ = z1;
    last_attn_out_ = attn_out;

    Tensor res1(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            res1[i][j] = input[i][j] + attn_out[i][j];
        }
    }
    last_res1_ = res1;

    // pre-LN → GELU FFN → residual
    Tensor z2 = ln2_.forward(res1);
    last_z2_ = z2;

    Tensor ffn_pre = ffn_fc1_.forward(z2);          // (n, ffn_dim)  pre-GELU
    GELU gelu;
    Tensor ffn_hidden = ffn_pre.apply(gelu);        // (n, ffn_dim)  post-GELU
    last_ffn_hidden_ = ffn_hidden;

    Tensor ffn_out = ffn_fc2_.forward(ffn_hidden);  // (n, d_model)
    last_ffn_out_ = ffn_out;

    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            output[i][j] = res1[i][j] + ffn_out[i][j];
        }
    }
    return output;
}

Tensor GQABlock::backward(const Tensor& grad_output, double learning_rate) {
    // Block forward layout:
    //   z1      = ln1(input)
    //   attn_o  = attn(z1)
    //   res1    = input + attn_o
    //   z2      = ln2(res1)
    //   ffn_pre = fc1(z2)
    //   ffn_h   = gelu(ffn_pre)
    //   ffn_o   = fc2(ffn_h)
    //   output  = res1 + ffn_o
    //
    // Given d_output = grad_output, we flow back through the residual adds
    // and the sublayers in reverse.

    // Step 1: residual split at the top.
    //   d_res1   = d_output  (residual add → gradient on both branches)
    //   d_ffn_o  = d_output
    Tensor d_res1 = grad_output.clone();
    Tensor d_ffn_o = grad_output.clone();

    // Step 2: fcn_fc2 backward → d_ffn_h  (gradient w.r.t. the FC2 input = ffn_h,
    //   shape (n, ffn_dim). This is NOT a contribution to d_res1; it flows
    //   further back through the FFN chain.)
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_o, learning_rate);
    // Note: we do NOT add d_ffn_h to d_res1 here. The residual at the top is
    // output = res1 + ffn_o, so the gradient w.r.t. res1 is d_output plus
    // (later) the chain rule back through ln2 from the FFN path. d_ffn_h
    // belongs to a different variable (ffn_h, in ffn_dim space) and must
    // not be mixed in.

    // Step 3: GELU backward. We need the pre-GELU activation; recompute it
    // from last_z2_ via ffn_fc1_ (matches the LinformerBlock convention).
    Tensor ffn_pre_recomp = ffn_fc1_.forward(last_z2_);
    Tensor d_ffn_pre(ffn_pre_recomp.rows, ffn_pre_recomp.cols);
    for (size_t i = 0; i < ffn_pre_recomp.rows; ++i) {
        for (size_t j = 0; j < ffn_pre_recomp.cols; ++j) {
            d_ffn_pre[i][j] = d_ffn_h[i][j] * gelu_deriv(ffn_pre_recomp[i][j]);
        }
    }

    // Step 4: ffn_fc1 backward → d_z2  (gradient w.r.t. the FC1 input = z2,
    //   shape (n, d_model). This IS in d_model space.)
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_pre, learning_rate);

    // Step 5: Chain rule back through ln2 → gradient contribution to res1.
    //   The residual is output = res1 + ffn_o. The gradient w.r.t. the
    //   res1 *variable* is d_output (from the residual split) PLUS
    //   dL/dz2 * dz2/d(res1) = ln2.backward(d_z2) (since z2 = ln2(res1)).
    //   This must be added in d_model space (n, d_model), not in ffn_dim space.
    Tensor d_res1_from_ln2 = ln2_.backward(d_z2, learning_rate);
    for (size_t i = 0; i < d_res1.rows; ++i)
        for (size_t j = 0; j < d_res1.cols; ++j)
            d_res1[i][j] += d_res1_from_ln2[i][j];

    // d_res1_pre (the gradient w.r.t. res1) is just d_res1 here
    // (we already chained through ln2 above). d_res1 is now dL/d(res1) and
    // res1 is the input node to ln2 in the forward pass.
    Tensor d_res1_pre = d_res1;

    // Step 6: residual split at res1 = input + attn_o.
    // d_residual_input = d_res1_pre
    // d_attn_o         = d_res1_pre
    Tensor d_attn_o = d_res1_pre.clone();

    // Step 7: attn backward → d_z1.
    Tensor d_z1 = attn_.backward(d_attn_o, learning_rate);

    // Step 8: ln1 backward + residual add.
    // d_input = d_x_via_ln1 + d_x_via_res
    //        = ln1.backward(d_z1) + d_res1_pre
    Tensor d_x_via_res = d_res1_pre;
    Tensor d_input = ln1_.backward(d_z1, learning_rate);
    for (size_t i = 0; i < d_input.rows; ++i)
        for (size_t j = 0; j < d_input.cols; ++j)
            d_input[i][j] += d_x_via_res[i][j];

    return d_input;
}

// ============================================================================
// GQAModel — stack of GQABlocks + classifier head
// ============================================================================

GQAModel::GQAModel(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
                   size_t out_features, size_t num_blocks, size_t ffn_dim)
    : d_model_(d_model),
      out_features_(out_features),
      blocks_(),
      classifier_(d_model, out_features)
{
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, num_query_heads, num_kv_heads, ffn_dim);
    }
}

std::vector<Tensor*> GQAModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> GQAModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

void GQAModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
}

void GQAModel::update_weights(double learning_rate) {
    for (auto& b : blocks_) b.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

Tensor GQAModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& b : blocks_) {
        x = b.forward(x);
    }
    return classifier_.forward(x);
}

Tensor GQAModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d = classifier_.backward(grad_output, learning_rate);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d = it->backward(d, learning_rate);
    }
    return d;
}
