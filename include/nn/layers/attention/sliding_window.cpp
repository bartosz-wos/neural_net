// ============================================================================
// Sliding Window Attention — implementation
// ============================================================================
//
// Forward (single sample, n tokens, d_model features, num_query_heads heads,
// num_kv_heads K/V heads, group_size = num_query_heads / num_kv_heads):
//
//   Q = X @ W_q^T                (n, d_model)
//   K = X @ W_k^T                (n, d_model)
//   V = X @ W_v^T                (n, d_model)
//
//   For each query head qh in [0, num_query_heads):
//       kh     = qh / group_size       (which K/V head this Q head uses)
//       q_off  = qh * head_dim
//       kv_off = kh * head_dim
//       scores_h = scale_ * (Q[:, q_off:q_off+head_dim] @ K[:, kv_off:kv_off+head_dim]^T)
//                (n, head_dim) @ (head_dim, n) = (n, n)
//       apply window mask: scores_h[i, j] = -1e9 if j out of [i-W+1, i] (causal)
//                          or |i-j| > W/2 (non-causal)
//                  ... unless i or j is in [0, num_global) (global tokens)
//       A_h = row_softmax(scores_h)  (n, n)
//       head_out[:, q_off:q_off+head_dim] = A_h @ V[:, kv_off:kv_off+head_dim]
//                                          (n, n) @ (n, head_dim) = (n, head_dim)
//
//   output = head_out @ W_o^T         (n, d_model)
//
// Backward: standard attention backward, with mask-zeroing baked in via
// softmax output (masked positions have attn[i, j] = 0, so d_scores[i, j]
// at masked positions is 0). K/V gradient accumulation follows the GQA
// convention (sum across all Q heads sharing the same K/V head).

#include "sliding_window.h"
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Row-wise softmax over the second axis of a 2D tensor.
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
inline double gelu_deriv(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    double th = std::tanh(u);
    double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
}

}  // namespace

// ============================================================================
// SlidingWindowAttention
// ============================================================================

SlidingWindowAttention::SlidingWindowAttention(size_t d_model,
                                               size_t num_query_heads,
                                               size_t num_kv_heads,
                                               size_t window_size,
                                               size_t num_global,
                                               bool causal)
    : d_model_(d_model),
      num_query_heads_(num_query_heads),
      num_kv_heads_(num_kv_heads),
      // Use dummy values for head_dim_, group_size_, scale_ in the initializer
      // list — they will be re-derived AFTER validation. Avoids 1/0 SIGFPE
      // when an invalid arg (e.g. num_query_heads=0) is passed.
      head_dim_(d_model > 0 ? d_model : 1),
      group_size_(num_query_heads > 0 ? num_query_heads : 1),
      window_size_(window_size),
      num_global_(num_global),
      causal_(causal),
      scale_(1.0),
      use_window_mask_(true)
{
    if (d_model_ == 0)
        throw std::invalid_argument("SlidingWindowAttention: d_model must be > 0");
    if (num_query_heads_ == 0)
        throw std::invalid_argument("SlidingWindowAttention: num_query_heads must be > 0");
    if (num_kv_heads_ == 0)
        throw std::invalid_argument("SlidingWindowAttention: num_kv_heads must be > 0");
    if (window_size_ == 0)
        throw std::invalid_argument("SlidingWindowAttention: window_size must be > 0");
    if (d_model_ % num_query_heads_ != 0)
        throw std::invalid_argument(
            "SlidingWindowAttention: d_model must be evenly divisible by num_query_heads");
    if (num_query_heads_ % num_kv_heads_ != 0)
        throw std::invalid_argument(
            "SlidingWindowAttention: num_query_heads must be evenly divisible by num_kv_heads");

    // NOW safe to derive the actual values.
    head_dim_   = d_model_ / num_query_heads_;
    group_size_ = num_query_heads_ / num_kv_heads_;
    scale_      = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // W_q, W_o: (d_model, d_model) — random init
    // W_k, W_v: only the first num_kv_heads_ * head_dim_ rows are used in
    //           forward; randomize those (and zero the rest for clarity).
    W_q = Tensor::random(d_model_, d_model_, 0.02);
    W_o = Tensor::random(d_model_, d_model_, 0.02);
    W_k = Tensor::zeros(d_model_, d_model_);
    W_v = Tensor::zeros(d_model_, d_model_);
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

std::vector<Tensor*> SlidingWindowAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}

std::vector<Tensor*> SlidingWindowAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}

void SlidingWindowAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

void SlidingWindowAttention::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_k -= grad_W_k * learning_rate;
    W_v -= grad_W_v * learning_rate;
    W_o -= grad_W_o * learning_rate;
}

Tensor SlidingWindowAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("SlidingWindowAttention.forward: input cols must equal d_model");
    if (num_global_ > n)
        throw std::invalid_argument("SlidingWindowAttention: num_global must be <= n");

    last_input_ = input.clone();

    // Q = input @ W_q^T, K = input @ W_k^T, V = input @ W_v^T  — (n, d_model)
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

        // scores = scale_ * (Q_h @ K_h^T) : (n, head_dim) @ (head_dim, n) = (n, n)
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

        // Apply window mask (unless disabled).
        if (use_window_mask_) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    bool i_global = (i < num_global_);
                    bool j_global = (j < num_global_);
                    if (i_global || j_global) continue;  // global tokens ignore window
                    bool in_window;
                    if (causal_) {
                        // j in [i - W + 1, i]
                        long lo = (long)i - (long)window_size_ + 1;
                        if (lo < 0) lo = 0;
                        in_window = ((long)j >= lo) && ((long)j <= (long)i);
                    } else {
                        // |i - j| <= W/2
                        long diff = (long)i - (long)j;
                        if (diff < 0) diff = -diff;
                        in_window = diff <= (long)(window_size_ / 2);
                    }
                    if (!in_window) {
                        scores[i][j] = -1e9;
                    }
                }
            }
        }

        // row softmax → A
        Tensor A = row_softmax(scores);

        // Cache A in last_attn_ at row range [qh*n, (qh+1)*n)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                last_attn_[qh * n + i][j] = A[i][j];
        // Cache per-head slice for test access (resize on first call)
        if (last_attn_by_head_.size() != num_query_heads_) {
            last_attn_by_head_.resize(num_query_heads_);
        }
        last_attn_by_head_[qh] = A;

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

Tensor SlidingWindowAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    if (grad_output.cols != d_model_)
        throw std::invalid_argument("SlidingWindowAttention.backward: grad_output cols must equal d_model");

    // ---- Step 1: propagate through W_o ----
    // Forward: output = head_out @ W_o  (right-multiplied)
    //   output[t][j] = Σ_k head_out[t][k] * W_o[k][j]
    //   → d_head_out[t][k] = Σ_j d_output[t][j] * W_o[k][j]
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
    // grad_W_o[k][j] += Σ_t grad_output[t][j] * last_head_out_[t][k]
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
        Tensor d_V_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += last_attn_[qh * n + j][i] * d_head_out[j][q_off + d];
                }
                d_V_h[i][d] = v;
            }
        }
        // d_A_h = d_head_out_h @ V_h^T : (n, head_dim) @ (head_dim, n) = (n, n)
        Tensor d_A_h(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double v = 0.0;
                for (size_t d = 0; d < head_dim_; ++d) {
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
        // Note: softmax backward already correctly gives dS=0 at masked
        // positions because A=0 there. So the mask is "implicit" through the
        // softmax cache — no explicit d_scores zeroing needed.

        // d_Q_h = scale_ * (d_scores @ K_h)   (n, head_dim)
        Tensor d_Q_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += d_scores[i][j] * last_k_[j][kv_off + d];
                }
                d_Q_h[i][d] = v * scale_;
            }
        }
        // d_K_h = scale_ * (d_scores^T @ Q_h) (n, head_dim)
        Tensor d_K_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += d_scores[j][i] * last_q_[j][q_off + d];
                }
                d_K_h[i][d] = v * scale_;
            }
        }

        // Accumulate into d_q_acc, d_k_acc, d_v_acc
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                d_q_acc[i][q_off + d] += d_Q_h[i][d];
                d_k_acc[i][kv_off + d] += d_K_h[i][d];
                d_v_acc[i][kv_off + d] += d_V_h[i][d];
            }
        }
    }

    // ---- Step 3: grad_W_q, grad_W_k, grad_W_v from d_q_acc, d_k_acc, d_v_acc ----
    // Forward: Q[i][j] = Σ_k input[i][k] * W_q[k][j]
    //   → grad_W_q[k][j] += Σ_i input[i][k] * d_q_acc[i][j]
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s += last_input_[t][k] * d_q_acc[t][j];
            }
            grad_W_q[k][j] += s;
            s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s += last_input_[t][k] * d_k_acc[t][j];
            }
            grad_W_k[k][j] += s;
            s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s += last_input_[t][k] * d_v_acc[t][j];
            }
            grad_W_v[k][j] += s;
        }
    }

    // ---- Step 4: propagate to input: d_input = d_q_acc @ W_q + d_k_acc @ W_k + d_v_acc @ W_v ----
    // d_input[i][k] = Σ_j d_q[i][j] * W_q[k][j] + Σ_j d_k[i][j] * W_k[k][j] + Σ_j d_v[i][j] * W_v[k][j]
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += d_q_acc[i][j] * W_q[k][j];
                v += d_k_acc[i][j] * W_k[k][j];
                v += d_v_acc[i][j] * W_v[k][j];
            }
            d_input[i][k] = v;
        }
    }
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// SlidingWindowBlock
// ============================================================================

SlidingWindowBlock::SlidingWindowBlock(size_t d_model,
                                       size_t num_query_heads,
                                       size_t num_kv_heads,
                                       size_t window_size,
                                       size_t num_global,
                                       bool causal,
                                       size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim),
      attn_(d_model, num_query_heads, num_kv_heads, window_size, num_global, causal),
      ln1_(d_model),
      ln2_(d_model)
{
    if (ffn_dim_ > 0) {
        W1_ = Tensor::random(d_model, ffn_dim_, 0.02);
        b1_ = Tensor::zeros(1, ffn_dim_);
        W2_ = Tensor::random(ffn_dim_, d_model, 0.02);
        b2_ = Tensor::zeros(1, d_model_);
        grad_W1_ = Tensor::zeros(d_model, ffn_dim_);
        grad_b1_ = Tensor::zeros(1, ffn_dim_);
        grad_W2_ = Tensor::zeros(ffn_dim_, d_model);
        grad_b2_ = Tensor::zeros(1, d_model_);
    }
}

std::vector<Tensor*> SlidingWindowBlock::parameters() {
    if (ffn_dim_ == 0) {
        // Just the attention params.
        return attn_.parameters();
    }
    auto p = attn_.parameters();
    p.push_back(&W1_);
    p.push_back(&b1_);
    p.push_back(&W2_);
    p.push_back(&b2_);
    return p;
}

std::vector<Tensor*> SlidingWindowBlock::gradients() {
    if (ffn_dim_ == 0) {
        return attn_.gradients();
    }
    auto g = attn_.gradients();
    g.push_back(&grad_W1_);
    g.push_back(&grad_b1_);
    g.push_back(&grad_W2_);
    g.push_back(&grad_b2_);
    return g;
}

void SlidingWindowBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    if (ffn_dim_ > 0) {
        grad_W1_.fill(0.0);
        grad_b1_.fill(0.0);
        grad_W2_.fill(0.0);
        grad_b2_.fill(0.0);
    }
}

void SlidingWindowBlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    if (ffn_dim_ > 0) {
        W1_ -= grad_W1_ * learning_rate;
        b1_ -= grad_b1_ * learning_rate;
        W2_ -= grad_W2_ * learning_rate;
        b2_ -= grad_b2_ * learning_rate;
    }
}

Tensor SlidingWindowBlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    // pre-LN → SWA → residual
    last_ln1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_ln1_);
    last_res1_ = (input + last_attn_out_);   // store element-wise sum

    if (ffn_dim_ == 0) {
        return last_res1_;
    }

    // pre-LN → FFN(GELU) → residual
    last_ln2_ = ln2_.forward(last_res1_);
    // FFN: pregelu = ln2 @ W1 + b1
    const size_t n = last_ln2_.rows;
    Tensor pregelu(n, ffn_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double s = b1_[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += last_ln2_[i][k] * W1_[k][j];
            pregelu[i][j] = s;
        }
    last_ffn_pregelu_ = pregelu;
    // GELU + W2+b2
    Tensor ffn_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        // gelu(pregelu) @ W2 + b2
        // First compute gelu(pregelu)
        Tensor gelu_p(n, ffn_dim_);
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double x = pregelu[i][j];
            // GELU tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            double u = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
            double th = std::tanh(u);
            gelu_p[i][j] = 0.5 * x * (1.0 + th);
        }
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b2_[0][j];
            for (size_t k = 0; k < ffn_dim_; ++k)
                s += gelu_p[i][k] * W2_[k][j];
            ffn_out[i][j] = s;
        }
    }
    last_ffn_out_ = ffn_out;

    // residual
    Tensor out(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            out[i][j] = last_res1_[i][j] + ffn_out[i][j];
    return out;
}

Tensor SlidingWindowBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: dL/d(block_out), shape (n, d_model_)
    // We walk backward:
    //   if ffn: out = res1 + ffn(ln2(res1))
    //     d_res1 = grad_output + dL/d(res1) from ffn path
    //     d_ffn_out = grad_output
    //     d_ln2, grad_W2, grad_b2 from ffn
    //     d_ln2 += d_res1 from ln2(res1) residual path (the ln2 input is res1)
    //   d_ln1 (and from res1 path) = attn path's grad_output
    //   d_input from pre-LN of ln1 + residual of input

    Tensor d_res1(grad_output.rows, grad_output.cols);

    if (ffn_dim_ == 0) {
        // out = res1 directly, so d_res1 = grad_output.
        d_res1 = grad_output.clone();
    } else {
        // d_ffn_out = grad_output (the FFN sub-block adds, so its gradient flows through unchanged)
        Tensor d_ffn_out = grad_output.clone();
        // Standard GELU-FFN backward:
        // ffn_out[i][j] = Σ_k gelu(pregelu[i][k]) * W2[k][j] + b2[0][j]
        // → d_gelu_p[i][k] = Σ_j d_ffn_out[i][j] * W2[k][j]
        // → grad_W2[k][j] += Σ_i d_ffn_out[i][j] * gelu_p[i][k]
        // → grad_b2[0][j] += Σ_i d_ffn_out[i][j]
        const size_t n = d_ffn_out.rows;
        Tensor d_gelu_p(n, ffn_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < ffn_dim_; ++k) {
                double v = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    v += d_ffn_out[i][j] * W2_[k][j];
                d_gelu_p[i][k] = v;
            }
        // grad_W2, grad_b2
        // gelu_p[i][k] needs to be reconstructed. We have last_ffn_pregelu_.
        // gelu_p[i][k] = GELU(last_ffn_pregelu_[i][k])
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < ffn_dim_; ++k) {
                double x = last_ffn_pregelu_[i][k];
                double u = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
                double th = std::tanh(u);
                double gelu_p = 0.5 * x * (1.0 + th);
                for (size_t j = 0; j < d_model_; ++j)
                    grad_W2_[k][j] += d_ffn_out[i][j] * gelu_p;
            }
        }
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d_ffn_out[i][j];
            grad_b2_[0][j] += s;
        }
        // d_pregelu = d_gelu_p * GELU'(pregelu)
        Tensor d_pregelu(n, ffn_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < ffn_dim_; ++k) {
                double x = last_ffn_pregelu_[i][k];
                d_pregelu[i][k] = d_gelu_p[i][k] * gelu_deriv(x);
            }
        // d_ln2[i][k] = Σ_j d_pregelu[i][j] * W1[k][j]
        Tensor d_ln2(n, d_model_);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < d_model_; ++k) {
                double v = 0.0;
                for (size_t j = 0; j < ffn_dim_; ++j)
                    v += d_pregelu[i][j] * W1_[k][j];
                d_ln2[i][k] = v;
            }
        // grad_W1[k][j] += Σ_i last_ln2_[i][k] * d_pregelu[i][j]
        for (size_t k = 0; k < d_model_; ++k) {
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i)
                    s += last_ln2_[i][k] * d_pregelu[i][j];
                grad_W1_[k][j] += s;
            }
        }
        // grad_b1[0][j] += Σ_i d_pregelu[i][j]
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d_pregelu[i][j];
            grad_b1_[0][j] += s;
        }
        // Now d_res1 has TWO contributions:
        //   (a) the residual bypass: out = res1 + ffn, so d_res1 += grad_output
        //   (b) the ln2 input path: ln2 takes res1 as input
        // → d_res1 = grad_output + ln2.backward(d_ln2)
        Tensor d_res1_via_ln2 = ln2_.backward(d_ln2, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                d_res1[i][j] = grad_output[i][j] + d_res1_via_ln2[i][j];
    }

    // d_res1 is dL/d(res1).
    // res1 = input + attn_out, where attn_out = attn(ln1(input))
    // → d_attn_out = d_res1
    // → d_input from residual bypass: d_input += d_res1
    // → d_ln1 = attn.backward(d_res1)
    // → d_input += ln1.backward(d_ln1)
    Tensor d_input = d_res1.clone();   // residual bypass contribution
    Tensor d_attn_out = d_res1;
    Tensor d_ln1 = attn_.backward(d_attn_out, 0.0);
    Tensor d_input_via_ln1 = ln1_.backward(d_ln1, 0.0);
    for (size_t i = 0; i < d_input.rows; ++i)
        for (size_t j = 0; j < d_input.cols; ++j)
            d_input[i][j] += d_input_via_ln1[i][j];
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// SlidingWindowModel
// ============================================================================

SlidingWindowModel::SlidingWindowModel(size_t input_dim,
                                       size_t d_model,
                                       size_t output_dim,
                                       size_t num_blocks,
                                       size_t num_query_heads,
                                       size_t num_kv_heads,
                                       size_t window_size,
                                       size_t num_global,
                                       bool causal,
                                       size_t ffn_dim)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim)
{
    W_in_  = Tensor::random(input_dim, d_model, 0.02);
    b_in_  = Tensor::zeros(1, d_model);
    W_out_ = Tensor::random(d_model, output_dim, 0.02);
    b_out_ = Tensor::zeros(1, output_dim);
    grad_W_in_  = Tensor::zeros(input_dim, d_model);
    grad_b_in_  = Tensor::zeros(1, d_model);
    grad_W_out_ = Tensor::zeros(d_model, output_dim);
    grad_b_out_ = Tensor::zeros(1, output_dim);

    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, num_query_heads, num_kv_heads,
                              window_size, num_global, causal, ffn_dim);
    }
}

std::vector<Tensor*> SlidingWindowModel::parameters() {
    std::vector<Tensor*> p = {&W_in_, &b_in_};
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&W_out_);
    p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> SlidingWindowModel::gradients() {
    std::vector<Tensor*> g = {&grad_W_in_, &grad_b_in_};
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_W_out_);
    g.push_back(&grad_b_out_);
    return g;
}

void SlidingWindowModel::zero_grad() {
    grad_W_in_.fill(0.0);
    grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void SlidingWindowModel::update_weights(double lr) {
    W_in_  -= grad_W_in_ * lr;
    b_in_  -= grad_b_in_ * lr;
    W_out_ -= grad_W_out_ * lr;
    b_out_ -= grad_b_out_ * lr;
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor SlidingWindowModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    const size_t n = input.rows;
    // Project to d_model
    Tensor proj(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_in_[0][j];
            for (size_t k = 0; k < input_dim_; ++k)
                s += input[i][k] * W_in_[k][j];
            proj[i][j] = s;
        }
    last_proj_ = proj;

    Tensor cur = proj;
    for (auto& b : blocks_) {
        cur = b.forward(cur);
    }
    last_block_out_ = cur;

    // Classifier: out = last_block_out @ W_out + b_out  (n, out_dim)
    Tensor out(n, output_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < output_dim_; ++j) {
            double s = b_out_[0][j];
            for (size_t k = 0; k < d_model_; ++k)
                s += cur[i][k] * W_out_[k][j];
            out[i][j] = s;
        }
    return out;
}

Tensor SlidingWindowModel::backward(const Tensor& grad_output, double lr) {
    const size_t n = grad_output.rows;
    // d_block_out = grad_output @ W_out^T  (n, d_model)
    Tensor d_block_out(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < output_dim_; ++j)
                v += grad_output[i][j] * W_out_[k][j];
            d_block_out[i][k] = v;
        }
    // grad_W_out[k][j] += Σ_t last_block_out_[t][k] * grad_output[t][j]
    for (size_t k = 0; k < d_model_; ++k)
        for (size_t j = 0; j < output_dim_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t)
                s += last_block_out_[t][k] * grad_output[t][j];
            grad_W_out_[k][j] += s;
        }
    // grad_b_out[0][j] += Σ_t grad_output[t][j]
    for (size_t j = 0; j < output_dim_; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_output[t][j];
        grad_b_out_[0][j] += s;
    }

    // Walk blocks backward
    Tensor d_cur = d_block_out;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_cur = it->backward(d_cur, lr);
    }

    // d_proj = d_cur; backprop input projection
    // W_in: input_dim × d_model, used as proj = input @ W_in^T
    // d_proj[i][j] = Σ_k input[i][k] * W_in[k][j]
    // → d_input[i][k] = Σ_j d_proj[i][j] * W_in[k][j]
    // → grad_W_in[k][j] += Σ_i input[i][k] * d_proj[i][j]
    Tensor d_input(n, input_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < input_dim_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                v += d_cur[i][j] * W_in_[k][j];
            d_input[i][k] = v;
        }
    for (size_t k = 0; k < input_dim_; ++k)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i)
                s += last_input_[i][k] * d_cur[i][j];
            grad_W_in_[k][j] += s;
        }
    // grad_b_in[0][j] += Σ_i d_cur[i][j]
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += d_cur[i][j];
        grad_b_in_[0][j] += s;
    }
    return d_input;
}
