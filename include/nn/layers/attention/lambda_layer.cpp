// Lambda Layer — Bello et al. ICLR 2021
//   https://arxiv.org/abs/2102.08602
// See lambda_layer.h for the full mathematical formulation.

#include "lambda_layer.h"
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Static helpers
// ============================================================================

namespace {

// Per-column softmax of K over the position axis (n).
// Input: K ∈ R^{n × k}. Output: Ksm ∈ R^{n × k} where for each column kk:
//   Ksm[m, kk] = exp(K[m, kk] - max_m K[m, kk]) / Σ_m exp(K[m, kk] - max)
Tensor col_softmax_over_positions(const Tensor& K) {
    const size_t n = K.rows;
    const size_t k = K.cols;
    Tensor out(n, k);
    for (size_t kk = 0; kk < k; ++kk) {
        double col_max = K(0, kk);
        for (size_t m = 1; m < n; ++m) col_max = std::max(col_max, K(m, kk));
        double sum = 0.0;
        for (size_t m = 0; m < n; ++m) {
            double e = std::exp(K(m, kk) - col_max);
            out(m, kk) = e;
            sum += e;
        }
        double inv = 1.0 / sum;
        for (size_t m = 0; m < n; ++m) out(m, kk) *= inv;
    }
    return out;
}

// Per-column softmax backward. Given Ksm (already softmaxed) and dL/dKsm,
// returns dL/dK_pre (the pre-softmax gradient).
// For column kk:
//   dL/dK_pre[m, kk] = Ksm[m, kk] * ( dL/dKsm[m, kk] - Σ_m' Ksm[m', kk] * dL/dKsm[m', kk] )
Tensor col_softmax_backward(const Tensor& Ksm, const Tensor& dL_dKsm) {
    const size_t n = Ksm.rows;
    const size_t k = Ksm.cols;
    Tensor out(n, k);
    for (size_t kk = 0; kk < k; ++kk) {
        double weighted_sum = 0.0;
        for (size_t m = 0; m < n; ++m)
            weighted_sum += Ksm(m, kk) * dL_dKsm(m, kk);
        for (size_t m = 0; m < n; ++m)
            out(m, kk) = Ksm(m, kk) * (dL_dKsm(m, kk) - weighted_sum);
    }
    return out;
}

// (n, d) @ (d, k) -> (n, k)  (i.e. Y = X · W where X is (n, d) and W is (d, k))
Tensor matmul_n_d_d_k(const Tensor& X, const Tensor& W) {
    const size_t n = X.rows;
    const size_t d = X.cols;
    const size_t k = W.cols;
    Tensor out(n, k);
    for (size_t i = 0; i < n; ++i)
        for (size_t kk = 0; kk < k; ++kk) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) acc += X(i, j) * W(j, kk);
            out(i, kk) = acc;
        }
    return out;
}

// (n, k) @ (k, d) -> (n, d)
Tensor matmul_n_k_k_d(const Tensor& A, const Tensor& B) {
    const size_t n = A.rows;
    const size_t k = A.cols;
    const size_t d = B.cols;
    Tensor out(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k; ++kk) acc += A(i, kk) * B(kk, j);
            out(i, j) = acc;
        }
    return out;
}

} // anonymous namespace

// ============================================================================
// LambdaAttention
// ============================================================================

LambdaAttention::LambdaAttention(size_t d_model, size_t max_seq_len, size_t k_depth, bool causal)
    : d_model_(d_model),
      max_seq_len_(max_seq_len),
      k_depth_(k_depth == 0 ? d_model : k_depth),
      causal_(causal)
{
    if (d_model_ == 0) throw std::invalid_argument("LambdaAttention: d_model must be > 0");
    if (max_seq_len_ == 0) throw std::invalid_argument("LambdaAttention: max_seq_len must be > 0");

    W_Q = Tensor::random(d_model_, k_depth_, 0.02);
    W_K = Tensor::random(d_model_, k_depth_, 0.02);
    W_V = Tensor::random(d_model_, d_model_, 0.02);
    position_emb_ = Tensor(max_seq_len_, max_seq_len_ * k_depth_);

    grad_W_Q = Tensor::zeros(d_model_, k_depth_);
    grad_W_K = Tensor::zeros(d_model_, k_depth_);
    grad_W_V = Tensor::zeros(d_model_, d_model_);
    grad_position_emb_ = Tensor::zeros(max_seq_len_, max_seq_len_ * k_depth_);

    // Initialize position embeddings small-random (per (n, m, kk) entry).
    std::mt19937 gen(0x1da);
    double bound = 0.05;
    std::uniform_real_distribution<> dis(-bound, bound);
    for (size_t i = 0; i < position_emb_.rows; ++i)
        for (size_t j = 0; j < position_emb_.cols; ++j)
            position_emb_(i, j) = dis(gen);

    last_input_         = Tensor();
    last_Q_             = Tensor();
    last_K_             = Tensor();
    last_V_             = Tensor();
    last_K_softmax_     = Tensor();
    last_lambda_content_= Tensor();
    last_lambda_pos_    = Tensor();
    last_lambda_total_  = Tensor();
    last_output_        = Tensor();
}

Tensor LambdaAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("LambdaAttention::forward: input.cols != d_model");
    if (n > max_seq_len_)
        throw std::invalid_argument("LambdaAttention::forward: n > max_seq_len");

    last_input_ = input.clone();

    // Projections (manual matmul — no matmul primitive in this codebase)
    last_Q_ = matmul_n_d_d_k(input, W_Q);  // (n, k)
    last_K_ = matmul_n_d_d_k(input, W_K);  // (n, k)
    last_V_ = matmul_n_d_d_k(input, W_V);  // (n, d_model)

    // Per-column softmax over positions
    last_K_softmax_ = col_softmax_over_positions(last_K_);  // (n, k)

    // Content lambda: λ_c = Ksm^T · V -> (k, d_model)
    last_lambda_content_ = matmul_n_k_k_d( // wait — need Ksm^T @ V
        // Ksm^T has shape (k, n); V has shape (n, d); result is (k, d)
        Tensor(last_K_softmax_).transpose(),
        last_V_
    );  // (k, d)

    // Position lambda: λ_p[n, kk, v] = Σ_m E[n, m, kk] * V[m, v]
    // We store this as a flat (n, k * d_model) tensor; index [n, kk * d_model + v].
    last_lambda_pos_ = Tensor(n, k_depth_ * d_model_);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        for (size_t kk = 0; kk < k_depth_; ++kk) {
            for (size_t v = 0; v < d_model_; ++v) {
                double acc = 0.0;
                for (size_t m = 0; m < n; ++m) {
                    if (causal_ && m > n_idx) continue;
                    double e_nmk = position_emb_(n_idx, m * k_depth_ + kk);
                    acc += e_nmk * last_V_(m, v);
                }
                last_lambda_pos_(n_idx, kk * d_model_ + v) = acc;
            }
        }
    }

    // Total lambda: λ[n, kk, v] = λ_c[kk, v] + λ_p[n, kk, v]
    last_lambda_total_ = Tensor(n, k_depth_ * d_model_);
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t kk = 0; kk < k_depth_; ++kk)
            for (size_t v = 0; v < d_model_; ++v)
                last_lambda_total_(n_idx, kk * d_model_ + v) =
                    last_lambda_content_(kk, v) + last_lambda_pos_(n_idx, kk * d_model_ + v);

    // Apply lambda to query: Y[n, v] = Σ_kk Q[n, kk] * λ[n, kk, v]
    Tensor Y(n, d_model_);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        for (size_t v = 0; v < d_model_; ++v) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k_depth_; ++kk)
                acc += last_Q_(n_idx, kk) * last_lambda_total_(n_idx, kk * d_model_ + v);
            Y(n_idx, v) = acc;
        }
    }
    last_output_ = Y.clone();
    return Y;
}

Tensor LambdaAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = last_input_.rows;
    const size_t d = d_model_;
    const size_t k = k_depth_;

    // dL/dQ[n, kk] = Σ_v g[n, v] * λ_total[n, kk, v]
    Tensor grad_Q(n, k);
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t kk = 0; kk < k; ++kk) {
            double acc = 0.0;
            for (size_t v = 0; v < d; ++v) acc += grad_output(n_idx, v) * last_lambda_total_(n_idx, kk * d + v);
            grad_Q(n_idx, kk) = acc;
        }

    // grad_W_Q[i, kk] = Σ_n last_input_[n, i] * grad_Q[n, kk]
    for (size_t i = 0; i < d; ++i)
        for (size_t kk = 0; kk < k; ++kk) {
            double acc = 0.0;
            for (size_t n_idx = 0; n_idx < n; ++n_idx) acc += last_input_(n_idx, i) * grad_Q(n_idx, kk);
            grad_W_Q(i, kk) = acc;
        }

    // dL/dλ_total[n, kk, v] = g[n, v] * Q[n, kk]
    Tensor grad_lambda_total(n, k * d);
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t kk = 0; kk < k; ++kk)
            for (size_t v = 0; v < d; ++v)
                grad_lambda_total(n_idx, kk * d + v) = grad_output(n_idx, v) * last_Q_(n_idx, kk);

    // Split: dL/dλ_c[kk, v] = Σ_n grad_lambda_total[n, kk, v]
    Tensor grad_lambda_c(k, d);
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t kk = 0; kk < k; ++kk)
            for (size_t v = 0; v < d; ++v)
                grad_lambda_c(kk, v) += grad_lambda_total(n_idx, kk * d + v);

    // dL/dλ_p = grad_lambda_total (same shape; we keep it as the same flat tensor)

    // dL/dV: from both λ_c and λ_p
    // From λ_c: dL/dV[m, v] += Σ_kk Ksm[m, kk] * grad_lambda_c[kk, v]
    // From λ_p: dL/dV[m, v] += Σ_{n, kk} E[n, m, kk] * grad_lambda_total[n, kk, v]
    Tensor grad_V(n, d);
    for (size_t m = 0; m < n; ++m)
        for (size_t v = 0; v < d; ++v) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k; ++kk) acc += last_K_softmax_(m, kk) * grad_lambda_c(kk, v);
            grad_V(m, v) = acc;
        }
    for (size_t m = 0; m < n; ++m)
        for (size_t v = 0; v < d; ++v)
            for (size_t n_idx = 0; n_idx < n; ++n_idx)
                for (size_t kk = 0; kk < k; ++kk) {
                    if (causal_ && m > n_idx) continue;
                    double e_nmk = position_emb_(n_idx, m * k + kk);
                    grad_V(m, v) += e_nmk * grad_lambda_total(n_idx, kk * d + v);
                }

    // grad_W_V[i, v] = Σ_n last_input_[n, i] * grad_V[n, v]
    for (size_t i = 0; i < d; ++i)
        for (size_t v = 0; v < d; ++v) {
            double acc = 0.0;
            for (size_t n_idx = 0; n_idx < n; ++n_idx) acc += last_input_(n_idx, i) * grad_V(n_idx, v);
            grad_W_V(i, v) = acc;
        }

    // dL/dKsm[m, kk] = Σ_v V[m, v] * grad_lambda_c[kk, v]
    Tensor grad_Ksm(n, k);
    for (size_t m = 0; m < n; ++m)
        for (size_t kk = 0; kk < k; ++kk) {
            double acc = 0.0;
            for (size_t v = 0; v < d; ++v) acc += last_V_(m, v) * grad_lambda_c(kk, v);
            grad_Ksm(m, kk) = acc;
        }

    // dL/dK_pre = softmax_backward(dL/dKsm, Ksm)
    Tensor grad_K_pre = col_softmax_backward(last_K_softmax_, grad_Ksm);

    // grad_W_K[i, kk] = Σ_n last_input_[n, i] * grad_K_pre[n, kk]
    for (size_t i = 0; i < d; ++i)
        for (size_t kk = 0; kk < k; ++kk) {
            double acc = 0.0;
            for (size_t n_idx = 0; n_idx < n; ++n_idx) acc += last_input_(n_idx, i) * grad_K_pre(n_idx, kk);
            grad_W_K(i, kk) = acc;
        }

    // grad_position_emb_[n_idx, m * k + kk] = Σ_v grad_lambda_total[n_idx, kk, v] * V[m, v]
    // (subject to causal mask: zero if m > n_idx)
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t m = 0; m < n; ++m) {
            if (causal_ && m > n_idx) {
                for (size_t kk = 0; kk < k; ++kk)
                    grad_position_emb_(n_idx, m * k + kk) = 0.0;
                continue;
            }
            for (size_t kk = 0; kk < k; ++kk) {
                double acc = 0.0;
                for (size_t v = 0; v < d; ++v) acc += grad_lambda_total(n_idx, kk * d + v) * last_V_(m, v);
                grad_position_emb_(n_idx, m * k + kk) = acc;
            }
        }

    // grad_input: three contributions (Q, K, V)
    // grad_input_from_Q[n, i] = Σ_kk grad_Q[n, kk] * W_Q[i, kk]
    // grad_input_from_K[n, i] = Σ_kk grad_K_pre[n, kk] * W_K[i, kk]
    // grad_input_from_V[n, i] = Σ_v grad_V[n, v] * W_V[i, v]
    Tensor grad_input(n, d);
    for (size_t n_idx = 0; n_idx < n; ++n_idx)
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += grad_Q(n_idx, kk) * W_Q(i, kk);
                acc += grad_K_pre(n_idx, kk) * W_K(i, kk);
            }
            for (size_t v = 0; v < d; ++v) acc += grad_V(n_idx, v) * W_V(i, v);
            grad_input(n_idx, i) = acc;
        }

    return grad_input;
}

void LambdaAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& G) {
        for (size_t i = 0; i < W.data.size(); ++i)
            W.data[i] -= learning_rate * G.data[i];
    };
    sgd(W_Q, grad_W_Q);
    sgd(W_K, grad_W_K);
    sgd(W_V, grad_W_V);
    sgd(position_emb_, grad_position_emb_);
}

void LambdaAttention::zero_grad() {
    grad_W_Q.fill(0.0);
    grad_W_K.fill(0.0);
    grad_W_V.fill(0.0);
    grad_position_emb_.fill(0.0);
}

std::vector<Tensor*> LambdaAttention::parameters() {
    return {&W_Q, &W_K, &W_V, &position_emb_};
}

std::vector<Tensor*> LambdaAttention::gradients() {
    return {&grad_W_Q, &grad_W_K, &grad_W_V, &grad_position_emb_};
}

// ============================================================================
// LambdaBlock — pre-LN -> LambdaAttention -> residual -> pre-LN -> GELU FFN -> residual
// ============================================================================

namespace { double gau_gelu(double x); } // forward; definition below

LambdaBlock::LambdaBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim, size_t k_depth, bool causal)
    : attn(d_model, max_seq_len, k_depth, causal),
      ln1(d_model),
      ln2(d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    W_ffn1_ = Tensor::random(d_model_, ffn_dim_, 0.02);
    b_ffn1_ = Tensor(1, ffn_dim_);
    W_ffn2_ = Tensor::random(ffn_dim_, d_model_, 0.02);
    b_ffn2_ = Tensor(1, d_model_);

    grad_W_ffn1_ = Tensor::zeros(d_model_, ffn_dim_);
    grad_b_ffn1_ = Tensor::zeros(1, ffn_dim_);
    grad_W_ffn2_ = Tensor::zeros(ffn_dim_, d_model_);
    grad_b_ffn2_ = Tensor::zeros(1, d_model_);
}

Tensor LambdaBlock::forward(const Tensor& input) {
    const size_t n = input.rows;
    const size_t d = d_model_;

    last_x_ = input.clone();
    last_z1_ = ln1.forward(input);

    // attn out
    Tensor attn_out = attn.forward(last_z1_);
    last_attn_out_ = attn_out.clone();

    last_res1_ = Tensor(n, d);
    for (size_t i = 0; i < n * d; ++i) last_res1_.data[i] = last_z1_.data[i] + attn_out.data[i];

    last_z2_ = ln2.forward(last_res1_);

    // FFN: y = GELU(z2 · W1 + b1) · W2 + b2
    Tensor h_pre = matmul_n_d_d_k(last_z2_, W_ffn1_);
    // broadcast bias b_ffn1 across all n rows
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < ffn_dim_; ++j) h_pre(i, j) += b_ffn1_(0, j);
    last_h_pre_ = h_pre.clone();

    last_h_act_ = Tensor(n, ffn_dim_);
    for (size_t i = 0; i < n * ffn_dim_; ++i) last_h_act_.data[i] = gau_gelu(last_h_pre_.data[i]);

    last_ffn_out_ = matmul_n_k_k_d(last_h_act_, W_ffn2_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) last_ffn_out_(i, j) += b_ffn2_(0, j);

    Tensor out(n, d);
    for (size_t i = 0; i < n * d; ++i) out.data[i] = last_res1_.data[i] + last_ffn_out_.data[i];
    return out;
}

Tensor LambdaBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = last_x_.rows;
    const size_t d = d_model_;

    // d_res1 from FFN residual + upstream
    Tensor d_res1(n, d);
    for (size_t i = 0; i < n * d; ++i) d_res1.data[i] = grad_output.data[i];

    // d_ffn_out from residual (1:1) — already in grad_output
    // Chain through b_ffn2, W_ffn2, h_act, h_pre, W_ffn1, b_ffn1, z2

    // grad_b_ffn2[j] = Σ_n d_ffn_out[n, j] = Σ_n grad_output[n, j]
    for (size_t j = 0; j < d; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) acc += grad_output(i, j);
        grad_b_ffn2_(0, j) = acc;
    }

    // We need grad_input to W_ffn2 (i.e. d_h_act) and grad_W_ffn2.
    Tensor d_h_act(n, ffn_dim_);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < ffn_dim_; ++j) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                double g = grad_output(n_idx, i);
                grad_W_ffn2_(j, i) += g * last_h_act_(n_idx, j);
                d_h_act(n_idx, j) += g * W_ffn2_(j, i);
            }
        }

    // GELU derivative
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t i = 0; i < n * ffn_dim_; ++i) {
        double x = last_h_pre_.data[i];
        double h = last_h_act_.data[i];
        // GELU'(x) — using the same closed-form as the rest of the repo (tanh approximation).
        // For stability, recompute from h rather than x to avoid double-precision loss.
        // But we need dGELU/dx — use the tanh form for consistency with gau_gelu.
        double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
        double t = std::tanh(inner);
        double dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
        d_h_pre.data[i] = d_h_act.data[i] * dgelu;
        (void)h; // unused in the tanh approximation
    }

    // grad_b_ffn1[j] = Σ_n d_h_pre[n, j]
    for (size_t j = 0; j < ffn_dim_; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) acc += d_h_pre(i, j);
        grad_b_ffn1_(0, j) = acc;
    }

    // d_z2[n, i] = Σ_j d_h_pre[n, j] * W_ffn1[i, j]
    // grad_W_ffn1[i, j] = Σ_n last_z2[n, i] * d_h_pre[n, j]
    Tensor d_z2(n, d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < ffn_dim_; ++j) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                double dh = d_h_pre(n_idx, j);
                grad_W_ffn1_(i, j) += last_z2_(n_idx, i) * dh;
                d_z2(n_idx, i) += dh * W_ffn1_(i, j);
            }
        }

    // d_res1_from_z2 = ln2.backward(d_z2) — gradient w.r.t. res1 from the z2 path
    Tensor d_res1_from_ln2 = ln2.backward(d_z2, 0.0);
    // Add to d_res1 (which already has the FFN-residual contribution)
    for (size_t i = 0; i < n * d; ++i) d_res1.data[i] += d_res1_from_ln2.data[i];

    // Now d_res1 is the TOTAL gradient w.r.t. res1. Chain through the residual
    // res1 = z1 + attn_out — both branches get d_res1.
    Tensor d_attn_out(n, d);
    for (size_t i = 0; i < n * d; ++i) {
        d_attn_out.data[i] = d_res1.data[i];
    }
    // d_z1 from residual = d_res1 (will be combined with d_z1_from_attn)
    Tensor d_z1_residual(n, d);
    for (size_t i = 0; i < n * d; ++i) d_z1_residual.data[i] = d_res1.data[i];

    // d_z1_from_attn = attn.backward(d_attn_out)
    Tensor d_z1_from_attn = attn.backward(d_attn_out, 0.0);

    // d_z1_total = d_z1_residual + d_z1_from_attn
    Tensor d_z1_total(n, d);
    for (size_t i = 0; i < n * d; ++i) d_z1_total.data[i] = d_z1_residual.data[i] + d_z1_from_attn.data[i];

    // Chain through ln1 backward
    Tensor grad_input = ln1.backward(d_z1_total, 0.0);

    return grad_input;
}

void LambdaBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    auto sgd = [&](Tensor& W, const Tensor& G) {
        for (size_t i = 0; i < W.data.size(); ++i)
            W.data[i] -= learning_rate * G.data[i];
    };
    sgd(W_ffn1_, grad_W_ffn1_);
    sgd(b_ffn1_, grad_b_ffn1_);
    sgd(W_ffn2_, grad_W_ffn2_);
    sgd(b_ffn2_, grad_b_ffn2_);
}

void LambdaBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W_ffn1_.fill(0.0);
    grad_b_ffn1_.fill(0.0);
    grad_W_ffn2_.fill(0.0);
    grad_b_ffn2_.fill(0.0);
}

std::vector<Tensor*> LambdaBlock::parameters() {
    std::vector<Tensor*> p = attn.parameters();
    p.push_back(&W_ffn1_); p.push_back(&b_ffn1_);
    p.push_back(&W_ffn2_); p.push_back(&b_ffn2_);
    return p;
}

std::vector<Tensor*> LambdaBlock::gradients() {
    std::vector<Tensor*> g = attn.gradients();
    g.push_back(&grad_W_ffn1_); g.push_back(&grad_b_ffn1_);
    g.push_back(&grad_W_ffn2_); g.push_back(&grad_b_ffn2_);
    return g;
}

// ============================================================================
// LambdaModel — stack of LambdaBlocks + per-token classifier
// ============================================================================

LambdaModel::LambdaModel(size_t d_model, size_t max_seq_len, size_t out_features,
                         size_t num_blocks, size_t ffn_dim, size_t k_depth, bool causal)
    : d_model_(d_model),
      max_seq_len_(max_seq_len),
      out_features_(out_features),
      num_blocks_(num_blocks),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    for (size_t i = 0; i < num_blocks_; ++i)
        blocks.push_back(std::make_unique<LambdaBlock>(d_model_, max_seq_len_, ffn_dim_, k_depth, causal));

    classifier_W_ = Tensor::random(d_model_, out_features_, 0.02);
    classifier_b_ = Tensor(1, out_features_);
    grad_classifier_W_ = Tensor::zeros(d_model_, out_features_);
    grad_classifier_b_ = Tensor::zeros(1, out_features_);
}

Tensor LambdaModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& blk : blocks) x = blk->forward(x);
    last_block_output_ = x.clone();

    Tensor logits(x.rows, out_features_);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double acc = classifier_b_(0, j);
            for (size_t kk = 0; kk < d_model_; ++kk) acc += x(i, kk) * classifier_W_(kk, j);
            logits(i, j) = acc;
        }
    return logits;
}

Tensor LambdaModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = last_block_output_.rows;

    // grad_b: Σ_n
    for (size_t j = 0; j < out_features_; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) acc += grad_output(i, j);
        grad_classifier_b_(0, j) = acc;
    }

    // grad_W[k, j] = Σ_n last_block_output_[n, k] * grad_output[n, j]
    // d_block_output[n, k] = Σ_j grad_output[n, j] * classifier_W[k, j]
    Tensor d_block_output(n, d_model_);
    for (size_t kk = 0; kk < d_model_; ++kk)
        for (size_t j = 0; j < out_features_; ++j) {
            double acc = 0.0;
            for (size_t n_idx = 0; n_idx < n; ++n_idx) acc += last_block_output_(n_idx, kk) * grad_output(n_idx, j);
            grad_classifier_W_(kk, j) = acc;
            for (size_t n_idx = 0; n_idx < n; ++n_idx) d_block_output(n_idx, kk) += grad_output(n_idx, j) * classifier_W_(kk, j);
        }

    // Chain backward through blocks (reverse)
    Tensor grad_x = d_block_output;
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
        grad_x = (*it)->backward(grad_x, 0.0);
    return grad_x;
}

void LambdaModel::update_weights(double learning_rate) {
    for (auto& blk : blocks) blk->update_weights(learning_rate);
    auto sgd = [&](Tensor& W, const Tensor& G) {
        for (size_t i = 0; i < W.data.size(); ++i)
            W.data[i] -= learning_rate * G.data[i];
    };
    sgd(classifier_W_, grad_classifier_W_);
    sgd(classifier_b_, grad_classifier_b_);
}

void LambdaModel::zero_grad() {
    for (auto& blk : blocks) blk->zero_grad();
    grad_classifier_W_.fill(0.0);
    grad_classifier_b_.fill(0.0);
}

std::vector<Tensor*> LambdaModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& blk : blocks) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&classifier_W_);
    p.push_back(&classifier_b_);
    return p;
}

std::vector<Tensor*> LambdaModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& blk : blocks) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_classifier_W_);
    g.push_back(&grad_classifier_b_);
    return g;
}

// ============================================================================
// Shared GELU (matches GAUBlock)
// ============================================================================
namespace {
double gau_gelu(double x) {
    // tanh approximation (matches the rest of the repo's activations).
    double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}
} // anonymous namespace
