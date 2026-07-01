// transformer_decoder.cpp — implementation of the Encoder-Decoder Transformer
// (Vaswani et al. 2017 §3.2).
//
// Conventions:
//   * (tokens, d_model) layout throughout. matches the repo's modern layers
//     (GQA, ALiBi, MLA, etc.).
//   * d_model must divide evenly into num_heads * head_dim.
//   * W_q, W_k, W_v, W_o all flat (d_model, d_model) with stacked head blocks.
//   * No biases on Q/K/V/O projections (Llama-style).
//
// The cross-attention class needs Q from one source and K/V from another,
// which doesn't fit the Layer single-input interface — we provide
// `forward_two` / `backward_two` plus a default Layer::forward that uses the
// cached `last_encoder_cached_` for callers who don't need to re-supply it.

#include "transformer_decoder.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {
// Softmax of a single row.
static void softmax_row(Tensor& t, size_t row, size_t cols) {
    double max_v = t(row, 0);
    for (size_t j = 1; j < cols; ++j) if (t(row, j) > max_v) max_v = t(row, j);
    double sum_exp = 0.0;
    for (size_t j = 0; j < cols; ++j) {
        double e = std::exp(t(row, j) - max_v);
        t(row, j) = e;
        sum_exp += e;
    }
    double inv = 1.0 / sum_exp;
    for (size_t j = 0; j < cols; ++j) t(row, j) *= inv;
}

// Standard matmul: out(i, j) = sum_k a(i, k) * b(k, j).
// Used so the cross-attention / self-attention backprop can be expressed
// in a single, uniform way.
static Tensor matmul(const Tensor& a, const Tensor& b) {
    size_t M = a.rows, K = a.cols, N = b.cols;
    Tensor out(M, N);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < K; ++k) v += a(i, k) * b(k, j);
            out(i, j) = v;
        }
    return out;
}

// Cached matmul-A^T-B: out(i, j) = sum_k a(k, i) * b(k, j).
static Tensor matmul_at_b(const Tensor& a, const Tensor& b) {
    size_t K = a.rows, M = a.cols, N = b.cols;
    Tensor out(M, N);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < K; ++k) v += a(k, i) * b(k, j);
            out(i, j) = v;
        }
    return out;
}
} // namespace

// ============================================================================
// MaskedMultiHeadSelfAttention
// ============================================================================
MaskedMultiHeadSelfAttention::MaskedMultiHeadSelfAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads), head_dim_(d_model / num_heads),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_heads) + 1e-9)),
      W_q(Tensor::random(d_model, d_model, 0.3)),
      W_k(Tensor::random(d_model, d_model, 0.3)),
      W_v(Tensor::random(d_model, d_model, 0.3)),
      W_o(Tensor::random(d_model, d_model, 0.3)),
      grad_W_q(Tensor::zeros(d_model, d_model)),
      grad_W_k(Tensor::zeros(d_model, d_model)),
      grad_W_v(Tensor::zeros(d_model, d_model)),
      grad_W_o(Tensor::zeros(d_model, d_model)),
      last_n_q_(0), last_n_kv_(0)
{
    if (num_heads == 0 || d_model % num_heads != 0)
        throw std::invalid_argument("MaskedMultiHeadSelfAttention: d_model must be divisible by num_heads");
}

std::vector<Tensor*> MaskedMultiHeadSelfAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}
std::vector<Tensor*> MaskedMultiHeadSelfAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}
void MaskedMultiHeadSelfAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
}
void MaskedMultiHeadSelfAttention::update_weights(double lr) {
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            W_q(i, j) -= lr * grad_W_q(i, j);
            W_k(i, j) -= lr * grad_W_k(i, j);
            W_v(i, j) -= lr * grad_W_v(i, j);
            W_o(i, j) -= lr * grad_W_o(i, j);
        }
}

Tensor MaskedMultiHeadSelfAttention::forward(const Tensor& input) {
    // Default: n_q = n_kv = input.rows, no separate K/V source.
    last_n_q_  = input.rows;
    last_n_kv_ = input.rows;
    last_input_ = input;

    size_t n = input.rows;

    // Q = X @ W_q, K = X @ W_k, V = X @ W_v   (n, d_model) each
    last_q_ = matmul(input, W_q);
    last_k_ = matmul(input, W_k);
    last_v_ = matmul(input, W_v);

    // Multi-head attention
    last_attn_ = Tensor(num_heads_ * n, n);
    Tensor output_acc(n, d_model_);
    output_acc.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Slice Q_h, K_h, V_h per head: rows (n, head_dim)
        Tensor Q_h(n, head_dim_), K_h(n, head_dim_), V_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h(i, dk) = last_q_(i, h * head_dim_ + dk);
                K_h(i, dk) = last_k_(i, h * head_dim_ + dk);
                V_h(i, dk) = last_v_(i, h * head_dim_ + dk);
            }

        // scores = Q_h @ K_h^T / scale_   (n, n)
        Tensor scores(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    s += Q_h(i, dk) * K_h(j, dk);
                scores(i, j) = s * scale_;
            }

        // Causal mask: j > i ⇒ −1e9 (key in future relative to query i)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (j > i) scores(i, j) += -1e9;

        // Row-softmax
        for (size_t i = 0; i < n; ++i) softmax_row(scores, i, n);

        // Cache softmax probs per head: (num_heads * n, n)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                last_attn_(h * n + i, j) = scores(i, j);

        // head_out_h = scores @ V_h   (n, head_dim)
        Tensor head_out(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j)
                    v += scores(i, j) * V_h(j, dk);
                head_out(i, dk) = v;
            }

        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                output_acc(i, h * head_dim_ + dk) = head_out(i, dk);
    }

    // Final projection: out = output_acc @ W_o   (n, d_model)
    last_head_out_ = output_acc;
    Tensor out = matmul(output_acc, W_o);
    return out;
}

Tensor MaskedMultiHeadSelfAttention::backward(const Tensor& grad_output, double /*lr*/) {
    size_t n = last_n_q_;

    // grad_proj = grad_output @ W_o^T   (n, d_model)
    // Forward: out[t, k] = sum_j output_acc[t, j] * W_o[j, k]
    //   ⇒ dL/doutput_acc[t, j] = sum_k dL/dout[t, k] * W_o[j, k]
    Tensor grad_proj(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) v += grad_output(i, k) * W_o(j, k);
            grad_proj(i, j) = v;
        }

    // grad_W_o += grad_output^T @ last_head_out_  (where forward: out = head_out @ W_o)
    // Forward: out[t, j] = sum_i head_out[t, i] * W_o[i, j]
    //   ⇒ dL/dW_o[i, j] = sum_t dL/dout[t, j] * head_out[t, i]
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t t = 0; t < n; ++t) v += grad_output(t, j) * last_head_out_(t, i);
            grad_W_o(i, j) += v;
        }

    // Per-head dQ, dK, dV
    Tensor grad_q(n, d_model_), grad_k(n, d_model_), grad_v(n, d_model_);
    grad_q.fill(0.0); grad_k.fill(0.0); grad_v.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Per-head grad
        Tensor grad_attn_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                grad_attn_h(i, dk) = grad_proj(i, h * head_dim_ + dk);

        // Reconstruct Q_h, K_h, V_h from cache
        Tensor Q_h(n, head_dim_), K_h(n, head_dim_), V_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h(i, dk) = last_q_(i, h * head_dim_ + dk);
                K_h(i, dk) = last_k_(i, h * head_dim_ + dk);
                V_h(i, dk) = last_v_(i, h * head_dim_ + dk);
            }

        Tensor attn = Tensor(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                attn(i, j) = last_attn_(h * n + i, j);

        // dV_h = attn^T @ grad_attn_h   (n, head_dim)
        Tensor grad_V_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < n; ++t) v += attn(t, i) * grad_attn_h(t, dk);
                grad_V_h(i, dk) = v;
            }

        // d_scores = grad_attn_h @ V_h^T
        Tensor d_scores(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    v += grad_attn_h(i, dk) * V_h(j, dk);
                d_scores(i, j) = v;
            }

        // Softmax backward: d_logits = attn * (d_scores - row_sum(attn * d_scores))
        for (size_t i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < n; ++j) row_sum += attn(i, j) * d_scores(i, j);
            for (size_t j = 0; j < n; ++j) d_scores(i, j) = attn(i, j) * (d_scores(i, j) - row_sum);
        }

        // Apply scale_ factor (it was applied to pre-softmax scores)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) d_scores(i, j) *= scale_;

        // Zero out gradient at masked positions (causal mask) — ensures no
        // gradient leaks to "future" keys even though the softmax probs are
        // already zero there.
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (j > i) d_scores(i, j) = 0.0;

        // dQ_h = d_scores @ K_h   (n, head_dim)
        Tensor grad_Q_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) v += d_scores(i, j) * K_h(j, dk);
                grad_Q_h(i, dk) = v;
            }

        // dK_h = d_scores^T @ Q_h
        Tensor grad_K_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) v += d_scores(j, i) * Q_h(j, dk);
                grad_K_h(i, dk) = v;
            }

        // Scatter back into (n, d_model) tensors
        for (size_t i = 0; i < n; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                grad_q(i, h * head_dim_ + dk) = grad_Q_h(i, dk);
                grad_k(i, h * head_dim_ + dk) = grad_K_h(i, dk);
                grad_v(i, h * head_dim_ + dk) = grad_V_h(i, dk);
            }
    }

    // grad_W_q += last_input^T @ grad_q  (same for k, v)
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double gq = 0.0, gk = 0.0, gv = 0.0;
            for (size_t t = 0; t < n; ++t) {
                gq += last_input_(t, i) * grad_q(t, j);
                gk += last_input_(t, i) * grad_k(t, j);
                gv += last_input_(t, i) * grad_v(t, j);
            }
            grad_W_q(i, j) += gq;
            grad_W_k(i, j) += gk;
            grad_W_v(i, j) += gv;
        }

    // d_input = grad_q @ W_q^T + grad_k @ W_k^T + grad_v @ W_v^T
    // Forward: q[t, k] = sum_j X[t, j] * W_q[j, k]
    //   ⇒ dL/dX[t, j] = sum_k dL/dq[t, k] * W_q[j, k]
    Tensor grad_input(n, d_model_);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += grad_q(t, k) * W_q(j, k);
                v += grad_k(t, k) * W_k(j, k);
                v += grad_v(t, k) * W_v(j, k);
            }
            grad_input(t, j) = v;
        }

    return grad_input;
}

// ============================================================================
// MultiHeadCrossAttention
// ============================================================================
MultiHeadCrossAttention::MultiHeadCrossAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads), head_dim_(d_model / num_heads),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_heads) + 1e-9)),
      W_q(Tensor::random(d_model, d_model, 0.3)),
      W_k(Tensor::random(d_model, d_model, 0.3)),
      W_v(Tensor::random(d_model, d_model, 0.3)),
      W_o(Tensor::random(d_model, d_model, 0.3)),
      grad_W_q(Tensor::zeros(d_model, d_model)),
      grad_W_k(Tensor::zeros(d_model, d_model)),
      grad_W_v(Tensor::zeros(d_model, d_model)),
      grad_W_o(Tensor::zeros(d_model, d_model)),
      last_n_q_(0), last_n_kv_(0)
{
    if (num_heads == 0 || d_model % num_heads != 0)
        throw std::invalid_argument("MultiHeadCrossAttention: d_model must be divisible by num_heads");
}

std::vector<Tensor*> MultiHeadCrossAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}
std::vector<Tensor*> MultiHeadCrossAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}
void MultiHeadCrossAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
}
void MultiHeadCrossAttention::update_weights(double lr) {
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            W_q(i, j) -= lr * grad_W_q(i, j);
            W_k(i, j) -= lr * grad_W_k(i, j);
            W_v(i, j) -= lr * grad_W_v(i, j);
            W_o(i, j) -= lr * grad_W_o(i, j);
        }
}

void MultiHeadCrossAttention::forward_two(const Tensor& decoder_hidden,
                                          const Tensor& encoder_output,
                                          Tensor& out) {
    last_decoder_ = decoder_hidden;
    last_encoder_ = encoder_output;
    last_n_q_  = decoder_hidden.rows;
    last_n_kv_ = encoder_output.rows;
    size_t n_q = last_n_q_, n_kv = last_n_kv_;

    // Q from decoder, K/V from encoder.
    last_q_ = matmul(decoder_hidden, W_q);
    last_k_ = matmul(encoder_output, W_k);
    last_v_ = matmul(encoder_output, W_v);

    last_attn_ = Tensor(num_heads_ * n_q, n_kv);
    Tensor output_acc(n_q, d_model_);
    output_acc.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        Tensor Q_h(n_q, head_dim_), K_h(n_kv, head_dim_), V_h(n_kv, head_dim_);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                Q_h(i, dk) = last_q_(i, h * head_dim_ + dk);
        for (size_t i = 0; i < n_kv; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                K_h(i, dk) = last_k_(i, h * head_dim_ + dk);
                V_h(i, dk) = last_v_(i, h * head_dim_ + dk);
            }

        // scores = Q_h @ K_h^T / scale_   (n_q, n_kv)
        Tensor scores(n_q, n_kv);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk) s += Q_h(i, dk) * K_h(j, dk);
                scores(i, j) = s * scale_;
            }

        // No causal mask in cross-attention.
        for (size_t i = 0; i < n_q; ++i) softmax_row(scores, i, n_kv);

        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j)
                last_attn_(h * n_q + i, j) = scores(i, j);

        // head_out_h = scores @ V_h   (n_q, head_dim)
        Tensor head_out(n_q, head_dim_);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n_kv; ++j) v += scores(i, j) * V_h(j, dk);
                head_out(i, dk) = v;
            }

        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                output_acc(i, h * head_dim_ + dk) = head_out(i, dk);
    }

    last_head_out_ = output_acc;
    out = matmul(output_acc, W_o);
}

void MultiHeadCrossAttention::backward_two(const Tensor& grad_output,
                                            Tensor& grad_decoder_hidden,
                                            Tensor& grad_encoder_output) {
    size_t n_q = last_n_q_, n_kv = last_n_kv_;

    // grad_proj = grad_output @ W_o^T   (n_q, d_model)
    // Forward: out[t, k] = sum_j output_acc[t, j] * W_o[j, k]
    //   ⇒ dL/doutput_acc[t, j] = sum_k dL/dout[t, k] * W_o[j, k]
    Tensor grad_proj(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) v += grad_output(i, k) * W_o(j, k);
            grad_proj(i, j) = v;
        }

    // grad_W_o += grad_output^T @ last_head_out_  (forward: out = head_out @ W_o)
    // Forward: out[t, k] = sum_j head_out[t, j] * W_o[j, k]
    //   ⇒ dL/dW_o[j, k] = sum_t dL/dout[t, k] * head_out[t, j]
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t t = 0; t < n_q; ++t) v += grad_output(t, j) * last_head_out_(t, i);
            grad_W_o(i, j) += v;
        }

    Tensor grad_q(n_q, d_model_);
    Tensor grad_k(n_kv, d_model_);
    Tensor grad_v(n_kv, d_model_);
    grad_q.fill(0.0); grad_k.fill(0.0); grad_v.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        Tensor grad_attn_h(n_q, head_dim_);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                grad_attn_h(i, dk) = grad_proj(i, h * head_dim_ + dk);

        Tensor Q_h(n_q, head_dim_), K_h(n_kv, head_dim_), V_h(n_kv, head_dim_);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                Q_h(i, dk) = last_q_(i, h * head_dim_ + dk);
        for (size_t i = 0; i < n_kv; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                K_h(i, dk) = last_k_(i, h * head_dim_ + dk);
                V_h(i, dk) = last_v_(i, h * head_dim_ + dk);
            }

        Tensor attn = Tensor(n_q, n_kv);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j)
                attn(i, j) = last_attn_(h * n_q + i, j);

        // dV_h = attn^T @ grad_attn_h   (n_kv, head_dim)
        Tensor grad_V_h(n_kv, head_dim_);
        for (size_t i = 0; i < n_kv; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < n_q; ++t) v += attn(t, i) * grad_attn_h(t, dk);
                grad_V_h(i, dk) = v;
            }

        // d_scores = grad_attn_h @ V_h^T   (n_q, n_kv)
        Tensor d_scores(n_q, n_kv);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk) v += grad_attn_h(i, dk) * V_h(j, dk);
                d_scores(i, j) = v;
            }

        // Softmax backward (per-row)
        for (size_t i = 0; i < n_q; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < n_kv; ++j) row_sum += attn(i, j) * d_scores(i, j);
            for (size_t j = 0; j < n_kv; ++j) d_scores(i, j) = attn(i, j) * (d_scores(i, j) - row_sum);
        }
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) d_scores(i, j) *= scale_;

        // dQ_h = d_scores @ K_h   (n_q, head_dim)
        Tensor grad_Q_h(n_q, head_dim_);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n_kv; ++j) v += d_scores(i, j) * K_h(j, dk);
                grad_Q_h(i, dk) = v;
            }

        // dK_h = d_scores^T @ Q_h   (n_kv, head_dim)
        Tensor grad_K_h(n_kv, head_dim_);
        for (size_t i = 0; i < n_kv; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < n_q; ++j) v += d_scores(j, i) * Q_h(j, dk);
                grad_K_h(i, dk) = v;
            }

        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                grad_q(i, h * head_dim_ + dk) = grad_Q_h(i, dk);
        for (size_t i = 0; i < n_kv; ++i)
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                grad_k(i, h * head_dim_ + dk) = grad_K_h(i, dk);
                grad_v(i, h * head_dim_ + dk) = grad_V_h(i, dk);
            }
    }

    // grad_W_q += last_decoder_^T @ grad_q
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t t = 0; t < n_q; ++t) v += last_decoder_(t, i) * grad_q(t, j);
            grad_W_q(i, j) += v;
        }
    // grad_W_k += last_encoder_^T @ grad_k   (same for v)
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double gk = 0.0, gv = 0.0;
            for (size_t t = 0; t < n_kv; ++t) {
                gk += last_encoder_(t, i) * grad_k(t, j);
                gv += last_encoder_(t, i) * grad_v(t, j);
            }
            grad_W_k(i, j) += gk;
            grad_W_v(i, j) += gv;
        }

    // grad_decoder_hidden = grad_q @ W_q^T   (n_q, d_model)
    // Forward: q[t, k] = sum_j X[t, j] * W_q[j, k]
    //   ⇒ dL/dX[t, j] = sum_k dL/dq[t, k] * W_q[j, k]
    grad_decoder_hidden = Tensor(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) v += grad_q(i, k) * W_q(j, k);
            grad_decoder_hidden(i, j) = v;
        }
    // grad_encoder_output = grad_k @ W_k^T + grad_v @ W_v^T
    // Forward: k[t, c] = sum_j enc[t, j] * W_k[j, c]
    //   ⇒ dL/d_enc[t, j] = sum_c dL/dk[t, c] * W_k[j, c]
    grad_encoder_output = Tensor(n_kv, d_model_);
    for (size_t i = 0; i < n_kv; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += grad_k(i, k) * W_k(j, k);
                v += grad_v(i, k) * W_v(j, k);
            }
            grad_encoder_output(i, j) = v;
        }
}

// Layer::forward fallback — uses the cached encoder output (so the layer
// can be used as a regular sub-layer in a network that wires encoder
// outputs via set_encoder()).
Tensor MultiHeadCrossAttention::forward(const Tensor& input) {
    Tensor out;
    forward_two(input, last_encoder_cached_, out);
    return out;
}
Tensor MultiHeadCrossAttention::backward(const Tensor& grad_output, double lr) {
    Tensor gd, ge;
    backward_two(grad_output, gd, ge);
    // Ignore encoder grad (no path back to set_encoder).
    return gd;
}

// ============================================================================
// TransformerDecoderBlock
// ============================================================================
TransformerDecoderBlock::TransformerDecoderBlock(size_t d_model, size_t num_heads, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      self_attn_(d_model, num_heads),
      ln2_(d_model),
      cross_attn_(d_model, num_heads),
      ln3_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{}

void TransformerDecoderBlock::forward_two(const Tensor& decoder_in,
                                          const Tensor& encoder_out,
                                          Tensor& decoder_out) {
    last_decoder_in_ = decoder_in;
    last_encoder_in_ = encoder_out;

    // Sub-layer 1: masked self-attention with pre-LN and residual.
    last_z1_ = ln1_.forward(decoder_in);
    Tensor sa_out = self_attn_.forward(last_z1_);
    last_sa_out_ = sa_out;
    // residual = decoder_in + sa_out
    Tensor res1(decoder_in.rows, decoder_in.cols);
    for (size_t i = 0; i < decoder_in.rows; ++i)
        for (size_t j = 0; j < decoder_in.cols; ++j)
            res1(i, j) = decoder_in(i, j) + sa_out(i, j);
    last_res1_ = res1;

    // Sub-layer 2: cross-attention with pre-LN and residual.
    last_z2_ = ln2_.forward(res1);
    Tensor ca_out;
    cross_attn_.forward_two(last_z2_, encoder_out, ca_out);
    last_ca_out_ = ca_out;
    // residual = res1 + ca_out
    Tensor res2(res1.rows, res1.cols);
    for (size_t i = 0; i < res1.rows; ++i)
        for (size_t j = 0; j < res1.cols; ++j)
            res2(i, j) = res1(i, j) + ca_out(i, j);
    last_res2_ = res2;

    // Sub-layer 3: FFN with pre-LN and residual.
    last_z3_ = ln3_.forward(res2);
    last_ffn_hidden_ = ffn_fc1_.forward(last_z3_);
    // Cache pre-GELU activations for backward (GELU derivative needs the pre-activations).
    last_ffn_pregelu_ = last_ffn_hidden_;
    // Apply GELU exactly matching the existing TransformerBlock convention.
    for (size_t i = 0; i < last_ffn_hidden_.rows; ++i)
        for (size_t j = 0; j < last_ffn_hidden_.cols; ++j) {
            double x = last_ffn_hidden_(i, j);
            last_ffn_hidden_(i, j) = 0.5 * x * (1.0 + std::tanh(
                std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        }
    Tensor ffn_out = ffn_fc2_.forward(last_ffn_hidden_);
    last_ffn_out_ = ffn_out;
    // residual = res2 + ffn_out
    Tensor out(res2.rows, res2.cols);
    for (size_t i = 0; i < res2.rows; ++i)
        for (size_t j = 0; j < res2.cols; ++j)
            out(i, j) = res2(i, j) + ffn_out(i, j);
    decoder_out = out;
}

void TransformerDecoderBlock::backward_two(const Tensor& grad_decoder_out,
                                            Tensor& grad_decoder_in,
                                            Tensor& grad_encoder_in) {
    size_t n = grad_decoder_out.rows;
    size_t d = d_model_;

    // Backprop through the third residual: out = res2 + ffn_out
    // ⇒ dres2 += grad, dffn_out = grad.
    Tensor grad_res2(n, d);
    Tensor grad_ffn_out(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            grad_res2(i, j) = grad_decoder_out(i, j);
            grad_ffn_out(i, j) = grad_decoder_out(i, j);
        }

    // FFN backward: ffn_fc2 → GELU → ffn_fc1 → ln3.
    Tensor grad_ffn_hidden = ffn_fc2_.backward(grad_ffn_out, 0.0);
    // GELU backward. Forward: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // dy/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    // where u = sqrt(2/pi) * (x + 0.044715 * x^3). The pre-GELU x must be used.
    for (size_t i = 0; i < grad_ffn_hidden.rows; ++i)
        for (size_t j = 0; j < grad_ffn_hidden.cols; ++j) {
            double x = last_ffn_pregelu_(i, j);
            double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
            double t = std::tanh(inner);
            double sech2 = 1.0 - t * t;
            double dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * std::sqrt(2.0 / M_PI) *
                              (1.0 + 3.0 * 0.044715 * x * x);
            grad_ffn_hidden(i, j) *= dgelu_dx;
        }
    Tensor grad_z3 = ffn_fc1_.backward(grad_ffn_hidden, 0.0);
    Tensor grad_res2_pre_ln3 = ln3_.backward(grad_z3, 0.0);
    // Add the ln3-bypass residual: res2 = res2_pre_ln3 + ffn_out, so
    // dres2_pre_ln3 = grad_res2_pre_ln3; we also add the original residual
    // contribution grad_res2 to res2_pre_ln3 path.
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            grad_res2(i, j) += grad_res2_pre_ln3(i, j);

    // Backprop through the second residual: res2 = res1 + ca_out
    // ⇒ dres1 += grad_res2 (residual contribution), dca_out = grad_res2.
    Tensor grad_res1(n, d);
    Tensor grad_ca_out(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            grad_res1(i, j) = grad_res2(i, j);
            grad_ca_out(i, j) = grad_res2(i, j);
        }
    // Cross-attn backward: pass grad_ca_out (= dL/dz2) directly. The cross-attn
    // returns grad_into_z2_from_ca (= dL/dz2 from its side) and grad_encoder.
    // We then chain grad_into_z2_from_ca back through ln2 to get the gradient
    // into res1 from the cross-attn path: dres1 += ln2.backward(grad_into_z2_from_ca).
    Tensor grad_into_z2_from_ca, grad_encoder;
    cross_attn_.backward_two(grad_ca_out, grad_into_z2_from_ca, grad_encoder);
    Tensor grad_res1_from_ca = ln2_.backward(grad_into_z2_from_ca, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            grad_res1(i, j) += grad_res1_from_ca(i, j);

    // Backprop through the first residual: res1 = decoder_in + sa_out
    // ⇒ ddecoder_in += grad_res1 (residual contribution), dsa_out = grad_res1.
    grad_decoder_in = Tensor(n, d);
    Tensor grad_sa_out(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            grad_decoder_in(i, j) = grad_res1(i, j);
            grad_sa_out(i, j) = grad_res1(i, j);
        }
    // Self-attn backward: pass grad_sa_out (= dL/dz1) directly to the
    // self-attention (its input is z1 = ln1(decoder_in)). The backward
    // returns grad_into_z1_from_sa (= dL/dz1 from the self-attn side).
    // We then chain grad_into_z1_from_sa back through ln1 to get the
    // gradient into decoder_in from the self-attn path.
    Tensor grad_z1 = grad_sa_out;  // alias for clarity
    Tensor grad_decoder_in_via_sa_input = self_attn_.backward(grad_z1, 0.0);
    // self_attn_.backward returned dL/dz1 from the self-attn side. Chain
    // through ln1 to get dL/ddecoder_in from the self-attn path.
    Tensor grad_decoder_in_via_sa = ln1_.backward(grad_decoder_in_via_sa_input, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            grad_decoder_in(i, j) += grad_decoder_in_via_sa(i, j);

    grad_encoder_in = grad_encoder;
}

std::vector<Tensor*> TransformerDecoderBlock::parameters() {
    auto p1 = ln1_.parameters();
    auto p2 = self_attn_.parameters();
    auto p3 = ln2_.parameters();
    auto p4 = cross_attn_.parameters();
    auto p5 = ln3_.parameters();
    auto p6 = ffn_fc1_.parameters();
    auto p7 = ffn_fc2_.parameters();
    std::vector<Tensor*> all;
    all.insert(all.end(), p1.begin(), p1.end());
    all.insert(all.end(), p2.begin(), p2.end());
    all.insert(all.end(), p3.begin(), p3.end());
    all.insert(all.end(), p4.begin(), p4.end());
    all.insert(all.end(), p5.begin(), p5.end());
    all.insert(all.end(), p6.begin(), p6.end());
    all.insert(all.end(), p7.begin(), p7.end());
    return all;
}

std::vector<Tensor*> TransformerDecoderBlock::gradients() {
    auto p1 = ln1_.gradients();
    auto p2 = self_attn_.gradients();
    auto p3 = ln2_.gradients();
    auto p4 = cross_attn_.gradients();
    auto p5 = ln3_.gradients();
    auto p6 = ffn_fc1_.gradients();
    auto p7 = ffn_fc2_.gradients();
    std::vector<Tensor*> all;
    all.insert(all.end(), p1.begin(), p1.end());
    all.insert(all.end(), p2.begin(), p2.end());
    all.insert(all.end(), p3.begin(), p3.end());
    all.insert(all.end(), p4.begin(), p4.end());
    all.insert(all.end(), p5.begin(), p5.end());
    all.insert(all.end(), p6.begin(), p6.end());
    all.insert(all.end(), p7.begin(), p7.end());
    return all;
}

void TransformerDecoderBlock::zero_grad() {
    ln1_.zero_grad();
    self_attn_.zero_grad();
    ln2_.zero_grad();
    cross_attn_.zero_grad();
    ln3_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void TransformerDecoderBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    self_attn_.update_weights(lr);
    ln2_.update_weights(lr);
    cross_attn_.update_weights(lr);
    ln3_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

// Layer::forward fallback (uses cached encoder)
Tensor TransformerDecoderBlock::forward(const Tensor& input) {
    Tensor out;
    forward_two(input, last_encoder_cached_, out);
    return out;
}
Tensor TransformerDecoderBlock::backward(const Tensor& grad_output, double lr) {
    Tensor gd, ge;
    backward_two(grad_output, gd, ge);
    return gd;
}

// ============================================================================
// TransformerDecoder
// ============================================================================
TransformerDecoder::TransformerDecoder(size_t d_model, size_t num_heads, size_t out_features,
                                        size_t num_blocks, size_t ffn_dim)
    : d_model_(d_model), out_features_(out_features),
      blocks_(),
      classifier_(d_model, out_features)
{
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i)
        blocks_.emplace_back(d_model, num_heads, ffn_dim);
}

void TransformerDecoder::set_encoder(Tensor enc) {
    for (auto& b : blocks_) b.set_encoder(enc);
}

void TransformerDecoder::forward_two(const Tensor& decoder_in,
                                      const Tensor& encoder_out,
                                      Tensor& decoder_out) {
    last_decoder_in_ = decoder_in;
    last_encoder_in_ = encoder_out;
    Tensor h = decoder_in;
    for (auto& b : blocks_) {
        Tensor h_new;
        b.forward_two(h, encoder_out, h_new);
        h = h_new;
    }
    decoder_out = classifier_.forward(h);
}

void TransformerDecoder::backward_two(const Tensor& grad_decoder_out,
                                       Tensor& grad_decoder_in,
                                       Tensor& grad_encoder_in) {
    size_t n = grad_decoder_out.rows;
    Tensor grad_h = classifier_.backward(grad_decoder_out, 0.0);
    // No encoder gradient path through the classifier.
    Tensor grad_enc_acc = Tensor(last_encoder_in_.rows, last_encoder_in_.cols);
    grad_enc_acc.fill(0.0);

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        Tensor gd_next, ge_next;
        it->backward_two(grad_h, gd_next, ge_next);
        grad_h = gd_next;
        // Accumulate encoder gradient (chain rule across blocks).
        for (size_t i = 0; i < grad_enc_acc.rows; ++i)
            for (size_t j = 0; j < grad_enc_acc.cols; ++j)
                grad_enc_acc(i, j) += ge_next(i, j);
    }
    grad_decoder_in = grad_h;
    grad_encoder_in = grad_enc_acc;
}

std::vector<Tensor*> TransformerDecoder::parameters() {
    std::vector<Tensor*> all;
    for (auto& b : blocks_) {
        auto p = b.parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    auto p = classifier_.parameters();
    all.insert(all.end(), p.begin(), p.end());
    return all;
}

std::vector<Tensor*> TransformerDecoder::gradients() {
    std::vector<Tensor*> all;
    for (auto& b : blocks_) {
        auto p = b.gradients();
        all.insert(all.end(), p.begin(), p.end());
    }
    auto p = classifier_.gradients();
    all.insert(all.end(), p.begin(), p.end());
    return all;
}

void TransformerDecoder::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
}

void TransformerDecoder::update_weights(double lr) {
    for (auto& b : blocks_) b.update_weights(lr);
    classifier_.update_weights(lr);
}

Tensor TransformerDecoder::forward(const Tensor& input) {
    Tensor out;
    forward_two(input, last_encoder_in_, out);
    return out;
}
Tensor TransformerDecoder::backward(const Tensor& grad_output, double lr) {
    Tensor gd, ge;
    backward_two(grad_output, gd, ge);
    return gd;
}