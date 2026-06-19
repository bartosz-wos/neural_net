// ============================================================================
// MultiHead-Conv Attention — Yang et al. 2023
//   "Convolutional Self-Attention Networks" (https://arxiv.org/abs/2308.01462)
//
// See conv_attention.h for the full mathematical formulation.
//
// This file implements:
//   * ConvAttention        — multi-head self-attention with 1D-conv Q/K/V.
//   * ConvAttentionBlock   — pre-LN → ConvAttention → residual →
//                            pre-LN → GELU FFN → residual.
//   * ConvAttentionModel   — stack of blocks + per-token classifier.
//
// DATA LAYOUT:
//   The input to ConvAttention is X ∈ R^{n × d_model} (n sequence positions,
//   d_model features each).  The 1D convolutions for Q/K/V should slide
//   across the sequence axis.  We treat d_model as the conv "in_channels"
//   and n as the "seq_len", with a single batch element.  So the conv1d
//   input is reshaped to (1, d_model * seq_len) with channels-first
//   layout: input_conv[c * seq_len + t] = X[t, c].  The conv1d output is
//   reshaped back to (n, d_model) by Q[t, c] = Q_flat[c, t].
//
//   This way, the conv1d sees each channel c's full sequence of length
//   seq_len and produces a per-position value for that channel — exactly
//   the semantics we want for Q/K/V.
//
// We follow the "Dense convention" used throughout this repo:
//   Dense::forward computes y = X @ W^T + b, with W stored as (out, in).
// So for ConvAttention, the W_o output projection is exactly this. The 1D
// convs for Q/K/V are implemented directly using the same math as the
// existing Conv1D layer (im2col → matmul → bias → reshape) but inlined for
// clarity and so we can stash the im2col columns for the backward pass.
// ============================================================================

#include "conv_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Local helpers — im2col / col2im for a 1D conv with the (1, in_ch * seq) layout
// ---------------------------------------------------------------------------

static void conv_im2col(const Tensor& input, size_t in_channels, size_t kernel_size,
                        size_t seq_in, size_t seq_out, size_t stride, size_t pad,
                        Tensor& col) {
    // input: (1, in_channels * seq_in) — channels-first layout
    // col:    (in_channels * kernel_size, 1 * seq_out)  — 1 batch, seq_out columns
    // Each output column is a "window" of the input: col[c*k + j, t_out] =
    // input[c * seq_in + (t_out * stride + j - pad)] when in-bounds, else 0.
    col = Tensor(in_channels * kernel_size, seq_out);
    col.fill(0.0);
    for (size_t c = 0; c < in_channels; ++c) {
        for (size_t j = 0; j < kernel_size; ++j) {
            size_t row_idx = c * kernel_size + j;
            for (size_t t_out = 0; t_out < seq_out; ++t_out) {
                int t = static_cast<int>(t_out * stride + j) - static_cast<int>(pad);
                if (t >= 0 && t < static_cast<int>(seq_in)) {
                    col[row_idx][t_out] = input[0][c * seq_in + t];
                }
            }
        }
    }
}

static void conv_col2im(const Tensor& dX_col, size_t in_channels, size_t kernel_size,
                        size_t seq_in, size_t seq_out, size_t stride, size_t pad,
                        Tensor& grad_input) {
    // grad_input: (1, in_channels * seq_in)  (channels-first layout)
    // dX_col:     (in_channels * kernel_size, seq_out)
    // Accumulate dX_col back to grad_input.  Multiple windows may write
    // to the same input position (when stride < kernel_size), so we
    // accumulate (i.e. +=).
    grad_input = Tensor(1, in_channels * seq_in);
    grad_input.fill(0.0);
    for (size_t c = 0; c < in_channels; ++c) {
        for (size_t j = 0; j < kernel_size; ++j) {
            size_t row_idx = c * kernel_size + j;
            for (size_t t_out = 0; t_out < seq_out; ++t_out) {
                int t = static_cast<int>(t_out * stride + j) - static_cast<int>(pad);
                if (t >= 0 && t < static_cast<int>(seq_in)) {
                    grad_input[0][c * seq_in + t] += dX_col[row_idx][t_out];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConvAttention
// ---------------------------------------------------------------------------

ConvAttention::ConvAttention(size_t d_model, size_t seq_len, size_t num_heads,
                             size_t kernel_size)
    : d_model_(d_model),
      seq_len_(seq_len),
      num_heads_(num_heads),
      head_dim_(d_model / num_heads),
      kernel_size_(kernel_size),
      pad_((kernel_size - 1) / 2),
      W_o(d_model, d_model),
      Wq_w_(d_model, d_model * kernel_size),
      Wk_w_(d_model, d_model * kernel_size),
      Wv_w_(d_model, d_model * kernel_size),
      Wq_b_(d_model, 1),
      Wk_b_(d_model, 1),
      Wv_b_(d_model, 1),
      grad_Wq_w_(d_model, d_model * kernel_size),
      grad_Wk_w_(d_model, d_model * kernel_size),
      grad_Wv_w_(d_model, d_model * kernel_size),
      grad_Wq_b_(d_model, 1),
      grad_Wk_b_(d_model, 1),
      grad_Wv_b_(d_model, 1),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_heads)))
{
    if (d_model == 0 || seq_len == 0 || num_heads == 0 || kernel_size == 0) {
        throw std::invalid_argument("ConvAttention: d_model, seq_len, num_heads, kernel_size must all be > 0");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("ConvAttention: d_model must be divisible by num_heads");
    }
    size_t seq_out = (seq_len + 2 * pad_ - kernel_size) / 1 + 1;
    if (seq_out != seq_len) {
        // For odd kernel with our default pad (k-1)/2, seq_out == seq_len.
        // If user picks even kernel or different padding such that the seq
        // length changes, we'd need to handle seq_out != seq_len, but we
        // throw early to keep the math simple.
        throw std::invalid_argument("ConvAttention: kernel_size/pad combo must preserve seq_len (use odd kernel)");
    }

    // Xavier-ish init for Q/K/V convs; W_o uses Dense's own init scheme.
    double s_qkv = std::sqrt(2.0 / static_cast<double>(d_model * kernel_size + d_model));
    std::mt19937 gen(42);
    std::normal_distribution<> dis_qkv(0.0, s_qkv);
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_ * kernel_size; ++j) {
            Wq_w_(i, j) = dis_qkv(gen);
            Wk_w_(i, j) = dis_qkv(gen);
            Wv_w_(i, j) = dis_qkv(gen);
        }
        Wq_b_(i, 0) = 0.0;
        Wk_b_(i, 0) = 0.0;
        Wv_b_(i, 0) = 0.0;
    }
}

Tensor ConvAttention::forward(const Tensor& input) {
    // input: (n, d_model)
    // Reshape to (1, d_model * seq_len) for the conv1d math (channels-first).
    size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("ConvAttention: input.cols must equal d_model");
    }
    if (n != seq_len_) {
        throw std::invalid_argument("ConvAttention: input.rows must equal seq_len (fixed at construction)");
    }
    last_input_ = input.clone();

    Tensor input_conv(1, d_model_ * seq_len_);
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            input_conv[0][c * seq_len_ + t] = input[t][c];
        }
    }

    // (1) im2col for input — shared across Q/K/V
    Tensor col_x;
    conv_im2col(input_conv, d_model_, kernel_size_, seq_len_, seq_len_, 1, pad_, col_x);
    col_q_ = col_x.clone();
    col_k_ = col_x.clone();
    col_v_ = col_x.clone();

    // (2) Project: y = W @ col + b
    // W_q: (d_model, d_model*k) @ col (d_model*k, seq_out)  =  (d_model, seq_out)
    Tensor Q_flat = Wq_w_ * col_q_;
    Tensor K_flat = Wk_w_ * col_k_;
    Tensor V_flat = Wv_w_ * col_v_;
    for (size_t o = 0; o < d_model_; ++o) {
        for (size_t i = 0; i < seq_len_; ++i) {
            Q_flat[o][i] += Wq_b_(o, 0);
            K_flat[o][i] += Wk_b_(o, 0);
            V_flat[o][i] += Wv_b_(o, 0);
        }
    }
    // Reshape to (n, d_model):  Q[t, c] = Q_flat[c, t]
    Tensor Q(n, d_model_), K(n, d_model_), V(n, d_model_);
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            Q[t][c] = Q_flat[c][t];
            K[t][c] = K_flat[c][t];
            V[t][c] = V_flat[c][t];
        }
    }
    last_q_ = Q.clone();
    last_k_ = K.clone();
    last_v_ = V.clone();

    // (3) Multi-head attention.
    Tensor attn_full(num_heads_ * n * n, 1);
    attn_full.fill(0.0);
    Tensor output(n, d_model_);
    output.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        Tensor Qh(n, head_dim_), Kh(n, head_dim_), Vh(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                size_t c = h * head_dim_ + i;
                Qh[t][i] = Q[t][c];
                Kh[t][i] = K[t][c];
                Vh[t][i] = V[t][c];
            }
        }

        // scores = Q_h @ K_h^T, scaled
        Tensor scores(n, n);
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double acc = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) acc += Qh[t][i] * Kh[s][i];
                scores[t][s] = acc * scale_;
            }
        }

        // row-softmax
        for (size_t t = 0; t < n; ++t) {
            double m = scores[t][0];
            for (size_t s = 1; s < n; ++s) m = std::max(m, scores[t][s]);
            double sum = 0.0;
            for (size_t s = 0; s < n; ++s) {
                scores[t][s] = std::exp(scores[t][s] - m);
                sum += scores[t][s];
            }
            for (size_t s = 0; s < n; ++s) {
                scores[t][s] /= sum;
                attn_full[h * n * n + t * n + s][0] = scores[t][s];
            }
        }

        // out_h = attn @ V_h
        Tensor out_h(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double acc = 0.0;
                for (size_t s = 0; s < n; ++s) acc += scores[t][s] * Vh[s][i];
                out_h[t][i] = acc;
            }
        }

        // Scatter back into output[t, c] for c = h*head_dim + i
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                size_t c = h * head_dim_ + i;
                output[t][c] = out_h[t][i];
            }
        }
    }
    last_attn_ = attn_full;
    last_output_pre_o_ = output.clone();

    // (4) Output projection: y = output @ W_o^T + b_o
    return W_o.forward(output);
}

Tensor ConvAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = last_input_.rows;
    W_o.zero_grad();
    grad_Wq_w_.fill(0.0); grad_Wk_w_.fill(0.0); grad_Wv_w_.fill(0.0);
    grad_Wq_b_.fill(0.0); grad_Wk_b_.fill(0.0); grad_Wv_b_.fill(0.0);

    // (1) Backward through W_o: y = x @ W_o^T + b_o
    Tensor grad_pre_o(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_output(t, j) * W_o.weights(j, k);
            grad_pre_o(t, k) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_output_pre_o_(t, k);
            W_o.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // (2) Backward through the multi-head attention.
    // dQ, dK, dV are (n, d_model) tensors matching the Q/K/V storage.
    Tensor dQ(n, d_model_), dK(n, d_model_), dV(n, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Re-extract per-head Q, K, V from storage.
        Tensor Qh(n, head_dim_), Kh(n, head_dim_), Vh(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                size_t c = h * head_dim_ + i;
                Qh[t][i] = last_q_[t][c];
                Kh[t][i] = last_k_[t][c];
                Vh[t][i] = last_v_[t][c];
            }
        }
        // Re-extract attn_h from last_attn_.
        Tensor attn_h(n, n);
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                attn_h[t][s] = last_attn_[h * n * n + t * n + s][0];
            }
        }
        // d_out_h
        Tensor d_out_h(n, head_dim_);
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                size_t c = h * head_dim_ + i;
                d_out_h[t][i] = grad_pre_o(t, c);
            }
        }
        // d_attn_h, d_V_h
        Tensor d_attn_h(n, n);
        Tensor dVh(n, head_dim_);
        dVh.fill(0.0);
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double acc = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) acc += d_out_h[t][i] * Vh[s][i];
                d_attn_h[t][s] = acc;
            }
        }
        for (size_t s = 0; s < n; ++s) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double acc = 0.0;
                for (size_t t = 0; t < n; ++t) acc += attn_h[t][s] * d_out_h[t][i];
                dVh[s][i] = acc;
            }
        }
        // softmax backward: d_scores = attn * (d_attn - attn * sum_{s'} d_attn * attn) row-wise
        //   dL/ds[s] = p[s] * (dL/dp[s] - sum_{s'} p[s'] * dL/dp[s'])
        // Then multiply by scale_ to get dL/d(Q@K^T) (since scores = scale * QK^T).
        Tensor d_scores(n, n);
        for (size_t t = 0; t < n; ++t) {
            double sum_da_a = 0.0;
            for (size_t s = 0; s < n; ++s) sum_da_a += d_attn_h[t][s] * attn_h[t][s];
            for (size_t s = 0; s < n; ++s) {
                d_scores[t][s] = attn_h[t][s] * (d_attn_h[t][s] - sum_da_a) * scale_;
            }
        }
        // d_Q_h, d_K_h
        Tensor dQh(n, head_dim_), dKh(n, head_dim_);
        dQh.fill(0.0); dKh.fill(0.0);
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double acc_q = 0.0;
                for (size_t s = 0; s < n; ++s) acc_q += d_scores[t][s] * Kh[s][i];
                dQh[t][i] = acc_q;
            }
        }
        for (size_t s = 0; s < n; ++s) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double acc_k = 0.0;
                for (size_t t = 0; t < n; ++t) acc_k += d_scores[t][s] * Qh[t][i];
                dKh[s][i] = acc_k;
            }
        }
        // Scatter into dQ, dK, dV at (t, h*head_dim + i)
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < head_dim_; ++i) {
                size_t c = h * head_dim_ + i;
                dQ[t][c] = dQh[t][i];
                dK[t][c] = dKh[t][i];
                dV[t][c] = dVh[t][i];
            }
        }
    }

    // (3) Backward through the conv projection.
    // The forward was:
    //   Q_flat = Wq_w @ col_x + b_q  (where col_x = im2col(input_conv))
    //   Q[t, c] = Q_flat[c, t]
    // So gradient w.r.t. Q_flat is:
    //   dQ_flat[c, t] = dQ[t, c]
    // Then:
    //   d_Wq_w = dQ_flat @ col_x^T
    //   d_b_q  = sum_t dQ_flat[:, t]
    //   d_col_x = Wq_w^T @ dQ_flat
    // (Same for K, V.)  We accumulate d_col_x into a single
    // (d_model*k, seq_len) gradient, which is the col2im of the input.
    Tensor dQ_flat(d_model_, seq_len_);
    Tensor dK_flat(d_model_, seq_len_);
    Tensor dV_flat(d_model_, seq_len_);
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            dQ_flat[c][t] = dQ[t][c];
            dK_flat[c][t] = dK[t][c];
            dV_flat[c][t] = dV[t][c];
        }
    }
    // d_Wq = dQ_flat @ col_x^T  →  (d_model, d_model*k)
    Tensor col_x_T = col_q_.transpose();  // (seq_len, d_model*k)
    Tensor dWq = dQ_flat * col_x_T;
    Tensor dWk = dK_flat * col_x_T;
    Tensor dWv = dV_flat * col_x_T;
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_ * kernel_size_; ++j) {
            grad_Wq_w_(i, j) += dWq(i, j);
            grad_Wk_w_(i, j) += dWk(i, j);
            grad_Wv_w_(i, j) += dWv(i, j);
        }
    }
    // d_b: sum over (seq_len) axis for each output channel
    for (size_t c = 0; c < d_model_; ++c) {
        double bq = 0.0, bk = 0.0, bv = 0.0;
        for (size_t i = 0; i < seq_len_; ++i) {
            bq += dQ_flat[c][i];
            bk += dK_flat[c][i];
            bv += dV_flat[c][i];
        }
        grad_Wq_b_(c, 0) += bq;
        grad_Wk_b_(c, 0) += bk;
        grad_Wv_b_(c, 0) += bv;
    }
    // d_col_x = Wq_w^T @ dQ_flat + Wk_w^T @ dK_flat + Wv_w^T @ dV_flat
    // Each is (d_model*k, seq_len).
    Tensor Wq_T = Wq_w_.transpose();  // (d_model*k, d_model)
    Tensor Wk_T = Wk_w_.transpose();
    Tensor Wv_T = Wv_w_.transpose();
    Tensor dX_col_q = Wq_T * dQ_flat;
    Tensor dX_col_k = Wk_T * dK_flat;
    Tensor dX_col_v = Wv_T * dV_flat;
    Tensor dX_col(d_model_ * kernel_size_, seq_len_);
    dX_col.fill(0.0);
    for (size_t i = 0; i < d_model_ * kernel_size_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            dX_col[i][j] = dX_col_q[i][j] + dX_col_k[i][j] + dX_col_v[i][j];
        }
    }
    // col2im: dX_col (d_model*k, seq_len) → grad_input_conv (1, d_model*seq_len)
    Tensor grad_input_conv;
    conv_col2im(dX_col, d_model_, kernel_size_, seq_len_, seq_len_, 1, pad_, grad_input_conv);
    // Reshape to (n, d_model):  grad_input[t, c] = grad_input_conv[0, c * seq_len + t]
    Tensor grad_input(n, d_model_);
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            grad_input[t][c] = grad_input_conv[0][c * seq_len_ + t];
        }
    }
    return grad_input;
}

void ConvAttention::update_weights(double learning_rate) {
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_ * kernel_size_; ++j) {
            Wq_w_(i, j) -= learning_rate * grad_Wq_w_(i, j);
            Wk_w_(i, j) -= learning_rate * grad_Wk_w_(i, j);
            Wv_w_(i, j) -= learning_rate * grad_Wv_w_(i, j);
        }
        Wq_b_(i, 0) -= learning_rate * grad_Wq_b_(i, 0);
        Wk_b_(i, 0) -= learning_rate * grad_Wk_b_(i, 0);
        Wv_b_(i, 0) -= learning_rate * grad_Wv_b_(i, 0);
    }
}

void ConvAttention::zero_grad() {
    W_o.zero_grad();
    grad_Wq_w_.fill(0.0);
    grad_Wk_w_.fill(0.0);
    grad_Wv_w_.fill(0.0);
    grad_Wq_b_.fill(0.0);
    grad_Wk_b_.fill(0.0);
    grad_Wv_b_.fill(0.0);
}

std::vector<Tensor*> ConvAttention::parameters() {
    return {&Wq_w_, &Wq_b_, &Wk_w_, &Wk_b_, &Wv_w_, &Wv_b_,
            &W_o.weights, &W_o.bias};
}

std::vector<Tensor*> ConvAttention::gradients() {
    return {&grad_Wq_w_, &grad_Wq_b_, &grad_Wk_w_, &grad_Wk_b_, &grad_Wv_w_, &grad_Wv_b_,
            &W_o.grad_weights, &W_o.grad_bias};
}

// ---------------------------------------------------------------------------
// ConvAttentionBlock
// ---------------------------------------------------------------------------

static inline double block_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
static inline double block_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

ConvAttentionBlock::ConvAttentionBlock(size_t d_model, size_t seq_len, size_t num_heads,
                                       size_t kernel_size, size_t ffn_dim)
    : d_model_(d_model),
      seq_len_(seq_len),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, seq_len, num_heads, kernel_size),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model)
{
    if (d_model == 0 || seq_len == 0) {
        throw std::invalid_argument("ConvAttentionBlock: d_model and seq_len must be > 0");
    }
}

Tensor ConvAttentionBlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    // Pre-LN → ConvAttention → residual
    last_z1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_z1_);
    last_res1_ = input + last_attn_out_;

    // Pre-LN → FFN → residual
    last_z2_ = ln2_.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    last_h_act_ = Tensor(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.rows; ++i) {
        for (size_t j = 0; j < last_h_pre_.cols; ++j) {
            last_h_act_(i, j) = block_gelu(last_h_pre_(i, j));
        }
    }
    last_ffn_out_ = ffn_fc2_.forward(last_h_act_);
    return last_res1_ + last_ffn_out_;
}

Tensor ConvAttentionBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();

    size_t n = grad_output.rows;

    // (1) ffn_fc2 backward
    Tensor d_h_act(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_output(t, j) * ffn_fc2_.weights(j, k);
            d_h_act(t, k) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_h_act_(t, k);
            ffn_fc2_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        ffn_fc2_.grad_bias(0, j) += b_acc;
    }

    // (2) GELU backward on h_pre
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h_act(t, k) * block_gelu_deriv(last_h_pre_(t, k));
        }
    }

    // (3) ffn_fc1 backward
    Tensor d_z2(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < ffn_dim_; ++j) acc += d_h_pre(t, j) * ffn_fc1_.weights(j, k);
            d_z2(t, k) = acc;
        }
    }
    for (size_t j = 0; j < ffn_dim_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += d_h_pre(t, j) * last_z2_(t, k);
            ffn_fc1_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_h_pre(t, j);
        ffn_fc1_.grad_bias(0, j) += b_acc;
    }

    // (4) ln2 backward
    Tensor d_res1_from_ffn = ln2_.backward(d_z2, 0.0);
    Tensor d_res1 = d_res1_from_ffn + grad_output;

    // (5) Direct residual contribution: d_x += d_res1
    Tensor d_x = d_res1.clone();

    // (6) ConvAttention backward, then ln1.backward
    Tensor d_z1 = attn_.backward(d_res1, 0.0);
    Tensor d_x_from_ln1 = ln1_.backward(d_z1, 0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            d_x(t, k) += d_x_from_ln1(t, k);
        }
    }
    return d_x;
}

void ConvAttentionBlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void ConvAttentionBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> ConvAttentionBlock::parameters() {
    auto p = attn_.parameters();
    auto a = ln1_.parameters();
    auto b = ln2_.parameters();
    auto c = ffn_fc1_.parameters();
    auto d = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> ConvAttentionBlock::gradients() {
    auto g = attn_.gradients();
    auto a = ln1_.gradients();
    auto b = ln2_.gradients();
    auto c = ffn_fc1_.gradients();
    auto d = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}

// ---------------------------------------------------------------------------
// ConvAttentionModel
// ---------------------------------------------------------------------------

ConvAttentionModel::ConvAttentionModel(size_t d_model, size_t seq_len, size_t out_features,
                                       size_t num_blocks, size_t num_heads,
                                       size_t kernel_size, size_t ffn_dim)
    : d_model_(d_model),
      seq_len_(seq_len),
      out_features_(out_features),
      num_blocks_(num_blocks),
      num_heads_(num_heads),
      classifier_(d_model, out_features)
{
    if (d_model == 0 || seq_len == 0 || out_features == 0 || num_blocks == 0) {
        throw std::invalid_argument("ConvAttentionModel: d_model, seq_len, out_features, num_blocks must be > 0");
    }
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.push_back(std::make_unique<ConvAttentionBlock>(
            d_model, seq_len, num_heads, kernel_size, ffn_dim));
    }
}

Tensor ConvAttentionModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    last_block_output_ = x.clone();
    return classifier_.forward(x);
}

Tensor ConvAttentionModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = grad_output.rows;
    classifier_.zero_grad();

    Tensor grad_x(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < out_features_; ++j) acc += grad_output(t, j) * classifier_.weights(j, k);
            grad_x(t, k) = acc;
        }
    }
    for (size_t j = 0; j < out_features_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_block_output_(t, k);
            classifier_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        classifier_.grad_bias(0, j) += b_acc;
    }

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_x = (*it)->backward(grad_x, 0.0);
    }
    return grad_x;
}

void ConvAttentionModel::update_weights(double learning_rate) {
    for (auto& block : blocks_) block->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void ConvAttentionModel::zero_grad() {
    for (auto& block : blocks_) block->zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> ConvAttentionModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& block : blocks_) {
        auto bp = block->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> ConvAttentionModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& block : blocks_) {
        auto bg = block->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
