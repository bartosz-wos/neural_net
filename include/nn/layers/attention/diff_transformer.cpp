// ============================================================================
// Differential Transformer — Ye et al. 2025 (ICLR 2025) implementation
//   "Differential Transformer"
//   https://arxiv.org/abs/2410.05258
//
// See diff_transformer.h for the full mathematical formulation. This file
// implements:
//   * DiffAttention         — single-layer diff attention
//   * DiffTransformerBlock  — pre-LN → DiffAttention → residual →
//                             pre-LN → FFN (GELU) → residual
//   * DiffTransformerModel  — stack of DiffTransformerBlocks + classifier
//
// Conventions match the rest of the repo (GQA / MLA / Linformer):
//   * Dense: y = X @ W^T + b, W stored as (out, in).
//   * (n, d_model) input/output, row-major.
//   * Per-head layouts are stored as flat (d_model, d_model) tensors with
//     head blocks stacked along the OUT axis. head h occupies columns
//     [h*head_dim : (h+1)*head_dim] of the W tensor.
// ============================================================================

#include "diff_transformer.h"
#include "../../activations/activations.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

// Softmax forward on a 2D row-major tensor (N, M).
// Writes softmax into the same buffer (in-place).
static void row_softmax_inplace(Tensor& A) {
    const size_t n = A.rows;
    const size_t m = A.cols;
    for (size_t i = 0; i < n; ++i) {
        double row_max = -1e300;
        for (size_t j = 0; j < m; ++j) {
            if (A[i][j] > row_max) row_max = A[i][j];
        }
        double sum = 0.0;
        for (size_t j = 0; j < m; ++j) {
            A[i][j] = std::exp(A[i][j] - row_max);
            sum += A[i][j];
        }
        const double inv = 1.0 / sum;
        for (size_t j = 0; j < m; ++j) A[i][j] *= inv;
    }
}

// ============================================================================
// DiffAttention
// ============================================================================

DiffAttention::DiffAttention(size_t d_model, size_t num_heads, double lambda_init)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      lambda_log(1, num_heads),
      grad_W_q(d_model, d_model),
      grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model),
      grad_W_o(d_model, d_model),
      grad_lambda_log(1, num_heads),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(0),
      half_dim_(0),
      lambda_init_(lambda_init),
      N_last_(0)
{
    if (d_model == 0) {
        throw std::invalid_argument("DiffAttention: d_model must be > 0");
    }
    if (num_heads == 0) {
        throw std::invalid_argument("DiffAttention: num_heads must be > 0");
    }
    head_dim_  = d_model / num_heads;
    half_dim_  = head_dim_ / 2;
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("DiffAttention: d_model must be divisible by num_heads");
    }
    if ((d_model / num_heads) % 2 != 0) {
        throw std::invalid_argument("DiffAttention: head_dim (d_model / num_heads) must be even");
    }
    if (lambda_init <= 0.0) {
        throw std::invalid_argument("DiffAttention: lambda_init must be > 0");
    }

    // Initialize weights with small random values.
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.05);
    auto init_dense = [&](Dense& W) {
        for (size_t i = 0; i < W.weights.rows; ++i)
            for (size_t j = 0; j < W.weights.cols; ++j)
                W.weights(i, j) = dis(gen);
        for (size_t j = 0; j < W.bias.cols; ++j)
            W.bias(0, j) = dis(gen) * 0.1;
    };
    init_dense(W_q);
    init_dense(W_k);
    init_dense(W_v);
    init_dense(W_o);

    // lambda_log = 0  →  λ = lambda_init (default 0.8)
    lambda_log.fill(0.0);
    grad_lambda_log.fill(0.0);
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

std::vector<Tensor*> DiffAttention::parameters() {
    return {&W_q.weights, &W_q.bias,
            &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias,
            &W_o.weights, &W_o.bias,
            &lambda_log};
}

std::vector<Tensor*> DiffAttention::gradients() {
    return {&grad_W_q, &W_q.grad_bias,
            &grad_W_k, &W_k.grad_bias,
            &grad_W_v, &W_v.grad_bias,
            &grad_W_o, &W_o.grad_bias,
            &grad_lambda_log};
}

void DiffAttention::zero_grad() {
    grad_W_q.fill(0.0); W_q.grad_bias.fill(0.0);
    grad_W_k.fill(0.0); W_k.grad_bias.fill(0.0);
    grad_W_v.fill(0.0); W_v.grad_bias.fill(0.0);
    grad_W_o.fill(0.0); W_o.grad_bias.fill(0.0);
    grad_lambda_log.fill(0.0);
}

void DiffAttention::update_weights(double learning_rate) {
    W_q.weights -= grad_W_q * learning_rate;
    W_q.bias    -= W_q.grad_bias * learning_rate;
    W_k.weights -= grad_W_k * learning_rate;
    W_k.bias    -= W_k.grad_bias * learning_rate;
    W_v.weights -= grad_W_v * learning_rate;
    W_v.bias    -= W_v.grad_bias * learning_rate;
    W_o.weights -= grad_W_o * learning_rate;
    W_o.bias    -= W_o.grad_bias * learning_rate;
    lambda_log  -= grad_lambda_log * learning_rate;
}

Tensor DiffAttention::forward(const Tensor& input) {
    const size_t N = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("DiffAttention: input.cols must equal d_model");
    }
    N_last_ = N;

    last_input_ = input.clone();

    // Project to Q, K, V via Dense (y = X @ W^T + b). No bias for K/V
    // in attention convention; we keep the bias but it cancels via the
    // softmax subtraction (paper style). Actually no — bias is harmless
    // but adds a parameter count. We keep Dense convention with bias;
    // it's what the rest of the repo does and the bias gradient is
    // handled in backward.

    Tensor Q = W_q.forward(input);   // (N, d_model)
    Tensor K = W_k.forward(input);   // (N, d_model)
    Tensor V = W_v.forward(input);   // (N, d_model)
    last_Q_ = Q;
    last_K_ = K;
    last_V_ = V;

    // Cache λ values used this forward
    last_lambda_ = Tensor(1, num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        last_lambda_(0, h) = std::exp(lambda_log(0, h)) * lambda_init_;
    }

    // Per-head: split K_h into K1, K2 along head_dim; split Q_h into Q1, Q2
    // Compute A1 = softmax(Q1 K1^T / sqrt(half_dim)), A2 similarly.
    // Diff_h = A1 - λ_h * A2.
    // O_h = Diff_h @ V_h.
    // Concat heads along head_dim.
    Tensor O(N, d_model_);  // pre-output-projection per-head concat
    last_A1_ = Tensor(num_heads_, N * N);  // flatten per-head rows
    last_A2_ = Tensor(num_heads_, N * N);
    last_Diff_ = Tensor(num_heads_, N * N);

    const double inv_scale = 1.0 / std::sqrt(static_cast<double>(half_dim_));

    for (size_t h = 0; h < num_heads_; ++h) {
        // Slices [h*head_dim, h*head_dim + half_dim) and [h*head_dim + half_dim, (h+1)*head_dim)
        const size_t h_off = h * head_dim_;
        const size_t h_half1_off = h_off;
        const size_t h_half2_off = h_off + half_dim_;

        // Q1 (N, half_dim), Q2 (N, half_dim)
        Tensor Q1(N, half_dim_), Q2(N, half_dim_);
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < half_dim_; ++j) {
                Q1(t, j) = Q(t, h_half1_off + j);
                Q2(t, j) = Q(t, h_half2_off + j);
            }
        }
        // K1, K2
        Tensor K1(N, half_dim_), K2(N, half_dim_);
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < half_dim_; ++j) {
                K1(t, j) = K(t, h_half1_off + j);
                K2(t, j) = K(t, h_half2_off + j);
            }
        }
        // V_h (N, head_dim) full
        Tensor V_h(N, head_dim_);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < head_dim_; ++j)
                V_h(t, j) = V(t, h_off + j);

        // A1 = Q1 K1^T * inv_scale  (N, N)
        Tensor S1(N, N), S2(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                double a1 = 0.0, a2 = 0.0;
                for (size_t j = 0; j < half_dim_; ++j) {
                    a1 += Q1(t, j) * K1(s, j);
                    a2 += Q2(t, j) * K2(s, j);
                }
                S1(t, s) = a1 * inv_scale;
                S2(t, s) = a2 * inv_scale;
            }
        row_softmax_inplace(S1);   // S1 is now A1
        row_softmax_inplace(S2);   // S2 is now A2

        // Cache post-softmax
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                last_A1_(h, t * N + s) = S1(t, s);
                last_A2_(h, t * N + s) = S2(t, s);
            }

        const double lambda_h = last_lambda_(0, h);
        // Diff_h[t, s] = A1[t, s] - λ_h * A2[t, s]
        Tensor Diff(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s)
                Diff(t, s) = S1(t, s) - lambda_h * S2(t, s);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s)
                last_Diff_(h, t * N + s) = Diff(t, s);

        // O_h = Diff_h @ V_h  (N, head_dim)
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t s = 0; s < N; ++s)
                    acc += Diff(t, s) * V_h(s, j);
                O(t, h_off + j) = acc;
            }
    }
    last_O_ = O;

    // Output projection: y = O @ W_o^T + b_o
    Tensor output = W_o.forward(O);   // (N, d_model)
    return output;
}

Tensor DiffAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != N_last_ || grad_output.cols != d_model_) {
        throw std::invalid_argument("DiffAttention::backward: grad_output shape mismatch");
    }
    const size_t N = N_last_;
    const double inv_scale = 1.0 / std::sqrt(static_cast<double>(half_dim_));

    // Zero gradients
    grad_W_q.fill(0.0); W_q.grad_bias.fill(0.0);
    grad_W_k.fill(0.0); W_k.grad_bias.fill(0.0);
    grad_W_v.fill(0.0); W_v.grad_bias.fill(0.0);
    grad_W_o.fill(0.0); W_o.grad_bias.fill(0.0);
    grad_lambda_log.fill(0.0);

    // Step 1: backward through W_o. y = O @ W_o^T + b_o.
    // d_O[t, j] = sum_jp grad_y[t, jp] * W_o.weights[jp, j]
    Tensor d_O(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t jp = 0; jp < d_model_; ++jp)
                acc += grad_output(t, jp) * W_o.weights(jp, j);
            d_O(t, j) = acc;
        }
    // W_o gradients: grad_W_o[jp, j] += sum_t grad_y[t, jp] * O[t, j]
    // grad_b_o[jp] += sum_t grad_y[t, jp]
    for (size_t jp = 0; jp < d_model_; ++jp) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += grad_output(t, jp);
        W_o.grad_bias(0, jp) += bacc;
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += grad_output(t, jp) * last_O_(t, j);
            grad_W_o(jp, j) += acc;
        }
    }

    // Step 2: backward through per-head Diff attention.
    // d_O is split per-head. For each head:
    //   d_V_h[s, j] = sum_t Diff_h[t, s] * d_O_h[t, j]
    //   d_Diff_h[t, s] = sum_j d_O_h[t, j] * V_h(s, j)
    //   d_A1[t, s] += d_Diff[t, s]                   (no scale)
    //   d_A2[t, s] += -λ_h * d_Diff[t, s]
    //   d_lambda_log[h] += -λ_h * sum_{t,s} A2[t, s] * d_Diff[t, s]
    //
    //   softmax backward on A1 and A2:
    //     d_S1[t, s] = A1[t, s] * (d_A1[t, s] - sum_{s'} A1[t, s'] * d_A1[t, s'])
    //     d_S2[t, s] = A2[t, s] * (d_A2[t, s] - sum_{s'} A2[t, s'] * d_A2[t, s'])
    //
    //   Q1, K1 gradients:
    //     d_Q1[t, j] = inv_scale * sum_s d_S1[t, s] * K1[s, j]
    //     d_K1[s, j] = inv_scale * sum_t d_S1[t, s] * Q1[t, j]
    //   similarly for Q2/K2.
    //
    //   d_Q[:, h_half1_off+j] += d_Q1[:, j]; d_Q[:, h_half2_off+j] += d_Q2[:, j]
    //   d_K[:, h_half1_off+j] += d_K1[:, j]; d_K[:, h_half2_off+j] += d_K2[:, j]
    //   d_V[:, h_off+j] += d_V_h[:, j]
    //
    // W gradients:
    //   grad_W_q[i, j] += sum_t d_Q[t, i] * input[t, j]
    //   grad_W_k[i, j] += sum_t d_K[t, i] * input[t, j]
    //   grad_W_v[i, j] += sum_t d_V[t, i] * input[t, j]
    //   grad_b_q[i] += sum_t d_Q[t, i]   (same for K, V)

    Tensor d_Q(N, d_model_);
    Tensor d_K(N, d_model_);
    Tensor d_V(N, d_model_);
    d_Q.fill(0.0);
    d_K.fill(0.0);
    d_V.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t h_off = h * head_dim_;
        const size_t h_half1_off = h_off;
        const size_t h_half2_off = h_off + half_dim_;

        // Per-head d_O (N, head_dim)
        Tensor d_O_h(N, head_dim_);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < head_dim_; ++j)
                d_O_h(t, j) = d_O(t, h_off + j);

        // V_h from cache
        Tensor V_h(N, head_dim_);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < head_dim_; ++j)
                V_h(t, j) = last_V_(t, h_off + j);

        // d_V_h[s, j] = sum_t Diff[t, s] * d_O_h[t, j]
        Tensor d_V_h(N, head_dim_);
        for (size_t s = 0; s < N; ++s)
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t)
                    acc += last_Diff_(h, t * N + s) * d_O_h(t, j);
                d_V_h(s, j) = acc;
            }
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < head_dim_; ++j)
                d_V(t, h_off + j) += d_V_h(t, j);

        // d_Diff[t, s] = sum_j d_O_h[t, j] * V_h[s, j]
        Tensor d_Diff(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                double acc = 0.0;
                for (size_t j = 0; j < head_dim_; ++j)
                    acc += d_O_h(t, j) * V_h(s, j);
                d_Diff(t, s) = acc;
            }

        // d_A1 = d_Diff, d_A2 = -λ_h * d_Diff
        const double lambda_h = last_lambda_(0, h);
        Tensor d_A1(N, N), d_A2(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                d_A1(t, s) = d_Diff(t, s);
                d_A2(t, s) = -lambda_h * d_Diff(t, s);
            }

        // λ gradient: d_lambda_log[h] += -λ_h * sum_{t,s} A2[t, s] * d_Diff[t, s]
        double lambda_grad_acc = 0.0;
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s)
                lambda_grad_acc += last_A2_(h, t * N + s) * d_Diff(t, s);
        // d_λ/d_lambda_log = lambda_init * exp(lambda_log) = λ_h
        grad_lambda_log(0, h) += -lambda_h * lambda_grad_acc;

        // Softmax backward for A1 (A1 is post-softmax)
        Tensor A1(N, N), A2(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                A1(t, s) = last_A1_(h, t * N + s);
                A2(t, s) = last_A2_(h, t * N + s);
            }
        // For each row t, compute sum_s' A[t, s'] * d_A[t, s']
        Tensor d_S1(N, N), d_S2(N, N);
        for (size_t t = 0; t < N; ++t) {
            double sum1 = 0.0, sum2 = 0.0;
            for (size_t s = 0; s < N; ++s) {
                sum1 += A1(t, s) * d_A1(t, s);
                sum2 += A2(t, s) * d_A2(t, s);
            }
            for (size_t s = 0; s < N; ++s) {
                d_S1(t, s) = A1(t, s) * (d_A1(t, s) - sum1);
                d_S2(t, s) = A2(t, s) * (d_A2(t, s) - sum2);
            }
        }

        // d_Q1[t, j] = inv_scale * sum_s d_S1[t, s] * K1[s, j]
        // We need K1, K2 from last_K_ (slice along head_dim)
        Tensor K1(N, half_dim_), K2(N, half_dim_);
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < half_dim_; ++j) {
                K1(t, j) = last_K_(t, h_half1_off + j);
                K2(t, j) = last_K_(t, h_half2_off + j);
            }
        }
        Tensor Q1(N, half_dim_), Q2(N, half_dim_);
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < half_dim_; ++j) {
                Q1(t, j) = last_Q_(t, h_half1_off + j);
                Q2(t, j) = last_Q_(t, h_half2_off + j);
            }
        }

        Tensor d_Q1(N, half_dim_), d_Q2(N, half_dim_);
        Tensor d_K1(N, half_dim_), d_K2(N, half_dim_);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < half_dim_; ++j) {
                double q1 = 0.0, k1 = 0.0;
                double q2 = 0.0, k2 = 0.0;
                for (size_t s = 0; s < N; ++s) {
                    q1 += d_S1(t, s) * K1(s, j);
                    k1 += d_S1(t, s) * Q1(t, j);  // careful: K1 grad uses Q1[t, j] (constant for this t)
                    q2 += d_S2(t, s) * K2(s, j);
                    k2 += d_S2(t, s) * Q2(t, j);
                }
                d_Q1(t, j) = inv_scale * q1;
                d_Q2(t, j) = inv_scale * q2;
            }
        // K gradients: d_K1[s, j] = inv_scale * sum_t d_S1[t, s] * Q1[t, j]
        for (size_t s = 0; s < N; ++s)
            for (size_t j = 0; j < half_dim_; ++j) {
                double k1 = 0.0, k2 = 0.0;
                for (size_t t = 0; t < N; ++t) {
                    k1 += d_S1(t, s) * Q1(t, j);
                    k2 += d_S2(t, s) * Q2(t, j);
                }
                d_K1(s, j) = inv_scale * k1;
                d_K2(s, j) = inv_scale * k2;
            }

        // Now scatter d_Q1, d_Q2 into d_Q[:, h_half1_off:h_half2_off]
        //      scatter d_K1, d_K2 into d_K[:, h_half1_off:h_half2_off]
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < half_dim_; ++j) {
                d_Q(t, h_half1_off + j) += d_Q1(t, j);
                d_Q(t, h_half2_off + j) += d_Q2(t, j);
                d_K(t, h_half1_off + j) += d_K1(t, j);
                d_K(t, h_half2_off + j) += d_K2(t, j);
            }
        }
    }

    // Step 3: backward through W_q, W_k, W_v (Dense backward).
    // For y = X @ W^T + b: d_W[i, j] = sum_t d_out[t, i] * x[t, j]
    //                      d_b[i]   = sum_t d_out[t, i]
    //                      d_x[t, j] = sum_i d_out[t, i] * W[i, j]
    for (size_t i = 0; i < d_model_; ++i) {
        double bq = 0.0, bk = 0.0, bv = 0.0;
        for (size_t t = 0; t < N; ++t) {
            bq += d_Q(t, i);
            bk += d_K(t, i);
            bv += d_V(t, i);
        }
        W_q.grad_bias(0, i) += bq;
        W_k.grad_bias(0, i) += bk;
        W_v.grad_bias(0, i) += bv;
        for (size_t j = 0; j < d_model_; ++j) {
            double gq = 0.0, gk = 0.0, gv = 0.0;
            for (size_t t = 0; t < N; ++t) {
                gq += d_Q(t, i) * last_input_(t, j);
                gk += d_K(t, i) * last_input_(t, j);
                gv += d_V(t, i) * last_input_(t, j);
            }
            grad_W_q(i, j) += gq;
            grad_W_k(i, j) += gk;
            grad_W_v(i, j) += gv;
        }
    }

    // d_input = d_Q @ W_q + d_K @ W_k + d_V @ W_v
    // For W (out, in): y = X @ W^T → d_X = d_Y @ W. So d_input[t, j] = sum_i d_Q[t, i] * W_q.weights[i, j]
    Tensor d_input(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i) {
                acc += d_Q(t, i) * W_q.weights(i, j)
                     + d_K(t, i) * W_k.weights(i, j)
                     + d_V(t, i) * W_v.weights(i, j);
            }
            d_input(t, j) = acc;
        }
    return d_input;
}

// ============================================================================
// DiffTransformerBlock
// ============================================================================

DiffTransformerBlock::DiffTransformerBlock(size_t d_model, size_t num_heads,
                                           double lambda_init, size_t ffn_dim)
    : attn(d_model, num_heads, lambda_init),
      ln1(d_model),
      ln2(d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{
}

std::vector<Tensor*> DiffTransformerBlock::parameters() {
    auto p = attn.parameters();
    auto p1 = ln1.parameters();
    auto p2 = ln2.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), pf1.begin(), pf1.end());
    p.insert(p.end(), pf2.begin(), pf2.end());
    return p;
}

std::vector<Tensor*> DiffTransformerBlock::gradients() {
    auto g = attn.gradients();
    auto g1 = ln1.gradients();
    auto g2 = ln2.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}

void DiffTransformerBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void DiffTransformerBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

Tensor DiffTransformerBlock::forward(const Tensor& input) {
    last_x = input.clone();
    // pre-LN → DiffAttention → residual
    Tensor z1 = ln1.forward(input);
    last_ln1_out = z1;
    Tensor attn_out = attn.forward(z1);
    last_attn_out = attn_out;
    Tensor res1(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            res1[i][j] = input[i][j] + attn_out[i][j];
    last_residual1_out = res1;

    // pre-LN → GELU FFN → residual
    Tensor z2 = ln2.forward(res1);
    last_ln2_out = z2;
    Tensor ffn_pre = ffn_fc1_.forward(z2);
    GELU gelu;
    Tensor ffn_hidden = ffn_pre.apply(gelu);
    last_ffn_hidden = ffn_hidden;
    Tensor ffn_out = ffn_fc2_.forward(ffn_hidden);
    last_ffn_out = ffn_out;
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            output[i][j] = res1[i][j] + ffn_out[i][j];
    return output;
}

Tensor DiffTransformerBlock::backward(const Tensor& grad_output, double learning_rate) {
    // Forward layout:
    //   z1      = ln1(input)
    //   attn_o  = attn(z1)
    //   res1    = input + attn_o
    //   z2      = ln2(res1)
    //   ffn_h   = gelu(ffn_fc1(z2))
    //   ffn_o   = ffn_fc2(ffn_h)
    //   output  = res1 + ffn_o

    // Step 1: split grad_output
    Tensor d_res1 = grad_output.clone();
    Tensor d_ffn_o = grad_output.clone();

    // Step 2: backward through ffn_fc2: d_ffn_h
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_o, learning_rate);

    // Step 3: GELU backward. We need gelu'(ffn_pre). ffn_pre = ffn_fc1_(last_ln2_out).
    // Recompute ffn_pre by running the forward (no state mutation, just recompute).
    GELU gelu;
    Tensor ffn_pre_recomp = ffn_fc1_.forward(last_ln2_out);
    Tensor d_ffn_pre(ffn_pre_recomp.rows, ffn_pre_recomp.cols);
    for (size_t i = 0; i < ffn_pre_recomp.rows; ++i)
        for (size_t j = 0; j < ffn_pre_recomp.cols; ++j)
            d_ffn_pre(i, j) = d_ffn_h(i, j) * gelu.derivative(ffn_pre_recomp(i, j));

    // Step 4: backward through ffn_fc1: d_z2
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_pre, learning_rate);

    // Step 5: residual add for ln2: d_res1 += d_z2
    for (size_t i = 0; i < d_res1.rows; ++i)
        for (size_t j = 0; j < d_res1.cols; ++j)
            d_res1[i][j] += d_z2[i][j];

    // Step 6: backward through ln2: d_res1 (pre-LN)
    Tensor d_res1_pre = ln2.backward(d_res1, learning_rate);

    // Step 7: split: d_input from res1 (residual) + d_attn_o (attention)
    Tensor d_input = d_res1_pre.clone();
    Tensor d_attn_o = d_res1_pre.clone();

    // Step 8: backward through attn: d_z1
    Tensor d_z1 = attn.backward(d_attn_o, learning_rate);

    // Step 9: backward through ln1: d_input
    Tensor d_input_final = ln1.backward(d_z1, learning_rate);
    // d_input += d_res1_pre (residual from res1 = input + attn_o)
    for (size_t i = 0; i < d_input.rows; ++i)
        for (size_t j = 0; j < d_input.cols; ++j)
            d_input[i][j] += d_input_final[i][j];

    return d_input;
}

// ============================================================================
// DiffTransformerModel
// ============================================================================

DiffTransformerModel::DiffTransformerModel(size_t input_dim, size_t d_model,
                                           size_t output_dim, size_t num_blocks,
                                           size_t num_heads, double lambda_init)
    : input_proj(input_dim, d_model), classifier(d_model, output_dim)
{
    if (input_dim == 0) {
        throw std::invalid_argument("DiffTransformerModel: input_dim must be > 0");
    }
    if (d_model == 0) {
        throw std::invalid_argument("DiffTransformerModel: d_model must be > 0");
    }
    if (output_dim == 0) {
        throw std::invalid_argument("DiffTransformerModel: output_dim must be > 0");
    }
    if (num_blocks == 0) {
        throw std::invalid_argument("DiffTransformerModel: num_blocks must be > 0");
    }
    if (num_heads == 0) {
        throw std::invalid_argument("DiffTransformerModel: num_heads must be > 0");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("DiffTransformerModel: d_model must be divisible by num_heads");
    }
    if ((d_model / num_heads) % 2 != 0) {
        throw std::invalid_argument("DiffTransformerModel: head_dim must be even");
    }
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks.push_back(std::make_unique<DiffTransformerBlock>(d_model, num_heads, lambda_init));
    }
}

Tensor DiffTransformerModel::forward(const Tensor& input) {
    Tensor x = input_proj.forward(input);
    for (auto& blk : blocks) x = blk->forward(x);
    return classifier.forward(x);
}

Tensor DiffTransformerModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = classifier.backward(grad_output, learning_rate);
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
    grad = input_proj.backward(grad, learning_rate);
    return grad;
}

void DiffTransformerModel::update_weights(double learning_rate) {
    input_proj.update_weights(learning_rate);
    for (auto& blk : blocks) blk->update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}

void DiffTransformerModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& blk : blocks) blk->zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> DiffTransformerModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&input_proj.weights); p.push_back(&input_proj.bias);
    for (auto& blk : blocks) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&classifier.weights); p.push_back(&classifier.bias);
    return p;
}

std::vector<Tensor*> DiffTransformerModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&input_proj.grad_weights); g.push_back(&input_proj.grad_bias);
    for (auto& blk : blocks) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&classifier.grad_weights); g.push_back(&classifier.grad_bias);
    return g;
}
