// ============================================================================
// cosFormer — Qin et al. 2022 implementation
// ============================================================================
//
// Forward (single sample, n tokens, d_model features):
//
//   Q = X @ W_q^T                (n, d_model)
//   K = X @ W_k^T                (n, d_model)
//   V = X @ W_v^T                (n, d_model)
//
//   Q_relu = ReLU(Q)             (n, d_model)   — non-negative feature map
//   K_relu = ReLU(K)             (n, d_model)
//   Q_cos[t, j] = Q_relu[t, j] * cos(π t / (2M))
//   Q_sin[t, j] = Q_relu[t, j] * sin(π t / (2M))
//   K_cos[t, j] = K_relu[t, j] * cos(π t / (2M))
//   K_sin[t, j] = K_relu[t, j] * sin(π t / (2M))
//
//   KV_cos = K_cos^T @ V          (d_model, d_model)
//   KV_sin = K_sin^T @ V          (d_model, d_model)
//   Ksum_cos = Σ_t K_cos[t, :]    (d_model,)
//   Ksum_sin = Σ_t K_sin[t, :]    (d_model,)
//
//   num[t, k] = Q_cos[t, :] @ KV_cos[:, k]  +  Q_sin[t, :] @ KV_sin[:, k]   (n, d_model)
//   den[t]    = Q_cos[t, :] @ Ksum_cos      +  Q_sin[t, :] @ Ksum_sin       (n,)
//   out_pre[t, k] = num[t, k] / (den[t] + eps)                              (n, d_model)
//
//   output = out_pre @ W_o^T                 (n, d_model)
//
// We don't bias the projections (no additive b on W_q/W_k/W_v/W_o) to match
// the paper convention and the rest of this repo's attention layers.
//
// ----------------------------------------------------------------------------
// Backward (given grad_output (n, d_model)):
//
//   1) d_out_pre = grad_output @ W_o         (n, d_model)   (output = out_pre @ W_o^T)
//      grad_W_o += grad_output^T @ out_pre   (d, d)
//
//   2) Backward through division by den (per query):
//      Let S[t, k] = num[t, k], z_t = den[t] + eps.  out[t, k] = S[t, k] / z_t.
//        d S[t, k]  = grad_out_pre[t, k] / z_t
//        d z_t      = -Σ_k grad_out_pre[t, k] · S[t, k] / z_t²
//
//   3) From d S[t, k] and d z_t:
//        d KV_cos   += Q_cos^T @ dS          (d, d)
//        d KV_sin   += Q_sin^T @ dS          (d, d)
//        d Ksum_cos += Q_cos^T · d z_t       (d,)   (per-row accumulate)
//        d Ksum_sin += Q_sin^T · d z_t       (d,)
//        d Q_cos    += dS @ KV_cos^T         (n, d)
//        d Q_sin    += dS @ KV_sin^T         (n, d)
//
//   4) Backward through K_cos^T V etc:
//        d V        = K_cos @ dKV_cos  +  K_sin @ dKV_sin        (n, d)
//        d K_cos    = dKV_cos @ V^T                              (n, d)
//        d K_sin    = dKV_sin @ V^T                              (n, d)
//      And d Ksum_cos/sin route back to d K_cos/K_sin via broadcasting
//      (since Ksum is a row sum, d K_cos[t, j] += d Ksum_cos[j]).
//
//   5) Backward through per-position cos/sin scaling:
//      d Q_relu[t, j] += cos(π t / (2M)) · d Q_cos[t, j]
//                      + sin(π t / (2M)) · d Q_sin[t, j]
//      d K_relu[t, j] similarly.
//      Then ReLU backward: d Q[t, j] = (Q[t, j] > 0) ? d Q_relu[t, j] : 0
//                          d K[t, j] = (K[t, j] > 0) ? d K_relu[t, j] : 0
//
//   6) Projections:
//      d X  += d Q @ W_q   +   d K @ W_k   +   d V @ W_v        (n, d_model)
//      grad_W_q += d Q^T @ X   (d, d)
//      grad_W_k += d K^T @ X   (d, d)
//      grad_W_v += d V^T @ X   (d, d)
//
// ----------------------------------------------------------------------------

#include "cosformer.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Construction
// ============================================================================

CosFormerAttention::CosFormerAttention(size_t d_model, size_t seq_len, size_t M)
    : d_model_(d_model), seq_len_(seq_len), M_(M == 0 ? seq_len : M) {

    if (d_model == 0)        throw std::invalid_argument("CosFormerAttention: d_model must be > 0");
    if (seq_len == 0)        throw std::invalid_argument("CosFormerAttention: seq_len must be > 0");
    if (M_ < seq_len_)       throw std::invalid_argument("CosFormerAttention: M must be >= seq_len (paper requires M >= N)");

    // Q/K/V/O projections, no bias (paper convention)
    auto init = [](Tensor& W) {
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 0.3 / std::sqrt((double)W.cols));
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) = dist(gen);
    };
    W_q = Tensor(d_model_, d_model_); init(W_q);  grad_W_q = Tensor(d_model_, d_model_);
    W_k = Tensor(d_model_, d_model_); init(W_k);  grad_W_k = Tensor(d_model_, d_model_);
    W_v = Tensor(d_model_, d_model_); init(W_v);  grad_W_v = Tensor(d_model_, d_model_);
    W_o = Tensor(d_model_, d_model_); init(W_o);  grad_W_o = Tensor(d_model_, d_model_);

    // cos_pos_ / sin_pos_ are filled lazily on first forward (allows T to
    // change in tests if we wanted, though the constructor already fixes
    // seq_len_). Initialize to zeros so update_weights is no-op safe.
    cos_pos_ = Tensor(1, 1);
    sin_pos_ = Tensor(1, 1);
}

void CosFormerAttention::ensure_position_vectors() {
    if (cos_pos_.rows == (size_t)seq_len_ && cos_pos_.cols == 1) return;
    cos_pos_ = Tensor(seq_len_, 1);
    sin_pos_ = Tensor(seq_len_, 1);
    const double coef = M_PI / (2.0 * (double)M_);
    for (size_t t = 0; t < seq_len_; ++t) {
        double ang = coef * (double)t;
        cos_pos_(t, 0) = std::cos(ang);
        sin_pos_(t, 0) = std::sin(ang);
    }
}

// ============================================================================
// forward
// ============================================================================

Tensor CosFormerAttention::forward(const Tensor& input) {
    ensure_position_vectors();

    const size_t n = input.rows;
    if (input.cols != d_model_) throw std::invalid_argument("CosFormerAttention: input cols must match d_model");
    if (n != seq_len_)          throw std::invalid_argument("CosFormerAttention: input rows must match seq_len");

    last_input_ = input;  // clone semantics via Tensor copy

    // Q = X @ W_q^T (n, d)
    Tensor Q = input * W_q.transpose();
    Tensor K = input * W_k.transpose();
    Tensor V = input * W_v.transpose();

    // ReLU(Q), ReLU(K)
    Tensor Q_relu(Q.rows, Q.cols);
    Tensor K_relu(K.rows, K.cols);
    for (size_t i = 0; i < Q.rows; ++i)
        for (size_t j = 0; j < Q.cols; ++j) {
            Q_relu(i, j) = Q(i, j) > 0.0 ? Q(i, j) : 0.0;
            K_relu(i, j) = K(i, j) > 0.0 ? K(i, j) : 0.0;
        }

    // Q_cos/Q_sin and K_cos/K_sin via per-row scaling
    Tensor Q_cos(Q_relu), Q_sin(Q_relu);
    Tensor K_cos(K_relu), K_sin(K_relu);
    for (size_t t = 0; t < n; ++t) {
        double c = cos_pos_(t, 0);
        double s = sin_pos_(t, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            Q_cos(t, j) = Q_relu(t, j) * c;
            Q_sin(t, j) = Q_relu(t, j) * s;
            K_cos(t, j) = K_relu(t, j) * c;
            K_sin(t, j) = K_relu(t, j) * s;
        }
    }

    // KV_cos = K_cos^T @ V,  KV_sin = K_sin^T @ V
    Tensor KV_cos = K_cos.transpose() * V;
    Tensor KV_sin = K_sin.transpose() * V;

    // Ksum_cos = Σ_t K_cos[t, :], Ksum_sin = Σ_t K_sin[t, :]
    Tensor Ksum_cos(1, d_model_);
    Tensor Ksum_sin(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double a = 0.0, b = 0.0;
        for (size_t t = 0; t < n; ++t) {
            a += K_cos(t, j);
            b += K_sin(t, j);
        }
        Ksum_cos(0, j) = a;
        Ksum_sin(0, j) = b;
    }

    // num = Q_cos @ KV_cos + Q_sin @ KV_sin  (n, d)
    Tensor num = Q_cos * KV_cos + Q_sin * KV_sin;

    // den = Q_cos @ Ksum_cos^T + Q_sin @ Ksum_sin^T  (n,)
    // (Ksum_cos is (1, d), so Ksum_cos^T is (d, 1); matmul (n, d)·(d, 1) = (n, 1))
    Tensor den = Q_cos * Ksum_cos.transpose() + Q_sin * Ksum_sin.transpose();
    Tensor den_row(n, 1);
    for (size_t t = 0; t < n; ++t) den_row(t, 0) = den(t, 0);

    // out_pre = num / (den + eps) row-wise
    const double eps = 1e-6;
    Tensor out_pre(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        double inv = 1.0 / (den(t, 0) + eps);
        for (size_t k = 0; k < d_model_; ++k) {
            out_pre(t, k) = num(t, k) * inv;
        }
    }

    // Cache for backward
    last_v_ = V;
    last_q_relu_ = Q_relu;
    last_k_relu_ = K_relu;
    last_q_cos_ = Q_cos;
    last_q_sin_ = Q_sin;
    last_k_cos_ = K_cos;
    last_k_sin_ = K_sin;
    last_KV_cos_ = KV_cos;
    last_KV_sin_ = KV_sin;
    last_Ksum_cos_ = Ksum_cos;
    last_Ksum_sin_ = Ksum_sin;
    last_den_ = den_row;
    last_out_ = out_pre;

    // output = out_pre @ W_o^T  (n, d)
    return out_pre * W_o.transpose();
}

// ============================================================================
// backward
// ============================================================================

Tensor CosFormerAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = seq_len_;
    const size_t d = d_model_;
    const double eps = 1e-6;

    // 1) d_out_pre = grad_output @ W_o    (n, d)
    //    grad_W_o += grad_output^T @ out_pre
    Tensor grad_out_pre(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_output(i, k) * W_o(k, j);
            grad_out_pre(i, j) = s;
        }
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_output(t, i) * last_out_(t, j);
            grad_W_o(i, j) += s;
        }

    // 2) Backward through division by den (per query):
    //    out_pre[t, k] = num[t, k] / (den[t] + eps)
    //    ⇒ d num[t, k] = grad_out_pre[t, k] / z_t
    //    ⇒ d den[t]    = -Σ_k grad_out_pre[t, k] · num[t, k] / z_t²
    //       (NOT multiplied by z_t — we want the literal form)
    //    Note: we use last_out_(t, k) = num(t, k) * inv(t) so num(t, k) = last_out_(t, k) * z_t
    Tensor grad_num(n, d);
    Tensor grad_den(n, 1);
    for (size_t t = 0; t < n; ++t) {
        double z_t = last_den_(t, 0) + eps;
        double z_inv = 1.0 / z_t;
        double z_inv2 = z_inv * z_inv;

        double g_dot_num = 0.0;
        for (size_t k = 0; k < d; ++k) {
            double num_tk = last_out_(t, k) * z_t;  // un-divide to get num[t, k]
            g_dot_num += grad_out_pre(t, k) * num_tk;
        }
        grad_den(t, 0) = -g_dot_num * z_inv2;

        for (size_t k = 0; k < d; ++k) {
            grad_num(t, k) = grad_out_pre(t, k) * z_inv;
        }
    }

    // 3) Backward through num = Q_cos · KV_cos + Q_sin · KV_sin
    //    and      den = Q_cos · Ksum_cos^T + Q_sin · Ksum_sin^T
    //
    //    d KV_cos += Q_cos^T @ grad_num                (d, d)
    //    d KV_sin += Q_sin^T @ grad_num                (d, d)
    //    d Q_cos  += grad_num @ KV_cos^T                (n, d)
    //    d Q_sin  += grad_num @ KV_sin^T                (n, d)
    //    d Ksum_cos += Q_cos^T · grad_den   (broadcast, sum rows)  (d,)
    //    d Ksum_sin += Q_sin^T · grad_den                          (d,)
    //    d Q_cos  += grad_den · Ksum_cos    (broadcast)           (n, d)
    //    d Q_sin  += grad_den · Ksum_sin                           (n, d)
    Tensor d_KV_cos(d, d), d_KV_sin(d, d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s1 += last_q_cos_(t, i) * grad_num(t, j);
                s2 += last_q_sin_(t, i) * grad_num(t, j);
            }
            d_KV_cos(i, j) = s1;
            d_KV_sin(i, j) = s2;
        }

    Tensor d_Ksum_cos(1, d), d_Ksum_sin(1, d);
    for (size_t j = 0; j < d; ++j) {
        double a = 0.0, b = 0.0;
        for (size_t t = 0; t < n; ++t) {
            a += last_q_cos_(t, j) * grad_den(t, 0);
            b += last_q_sin_(t, j) * grad_den(t, 0);
        }
        d_Ksum_cos(0, j) = a;
        d_Ksum_sin(0, j) = b;
    }

    // d_Q_cos = grad_num @ KV_cos^T  +  grad_den · Ksum_cos
    Tensor d_Q_cos(n, d), d_Q_sin(n, d);
    for (size_t t = 0; t < n; ++t) {
        double gdt = grad_den(t, 0);
        for (size_t j = 0; j < d; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (size_t k = 0; k < d; ++k) {
                s1 += grad_num(t, k) * last_KV_cos_(j, k);  // KV^T[k, j] = KV(j, k)
                s2 += grad_num(t, k) * last_KV_sin_(j, k);
            }
            d_Q_cos(t, j) = s1 + gdt * last_Ksum_cos_(0, j);
            d_Q_sin(t, j) = s2 + gdt * last_Ksum_sin_(0, j);
        }
    }

    // 4) Backward through KV = K^T @ V  (for both cos and sin)
    //    KV_cos ∈ R^{d × d}, V ∈ R^{n × d}, K_cos ∈ R^{n × d}.
    //    d V[t, j]        = Σ_k K_cos[t, k] · d_KV_cos[k, j]    (and same for sin)
    //    d K_cos[t, k]    = Σ_j d_KV_cos[k, j] · V[t, j]
    Tensor d_V(n, d);
    Tensor d_K_cos(n, d), d_K_sin(n, d);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double v_cos = 0.0, v_sin = 0.0;
            for (size_t k = 0; k < d; ++k) {
                v_cos += last_k_cos_(t, k) * d_KV_cos(k, j);
                v_sin += last_k_sin_(t, k) * d_KV_sin(k, j);
            }
            d_V(t, j) = v_cos + v_sin;
        }
        for (size_t k = 0; k < d; ++k) {
            double a = 0.0, b = 0.0;
            for (size_t j = 0; j < d; ++j) {
                a += d_KV_cos(k, j) * last_v_(t, j);
                b += d_KV_sin(k, j) * last_v_(t, j);
            }
            d_K_cos(t, k) = a;
            d_K_sin(t, k) = b;
        }
    }

    // Add Ksum gradient: Ksum is sum over t of K[t, :], so d K[t, :] += d Ksum[:]
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d; ++j) {
            d_K_cos(t, j) += d_Ksum_cos(0, j);
            d_K_sin(t, j) += d_Ksum_sin(0, j);
        }
    }

    // 5) Backward through per-position cos/sin scaling and ReLU:
    //    Q_cos[t, j] = ReLU(Q[t, j]) * cos_pos[t]
    //    Q_sin[t, j] = ReLU(Q[t, j]) * sin_pos[t]
    //    ⇒ d ReLU(Q)[t, j] = d Q_cos[t, j] * cos_pos[t] + d Q_sin[t, j] * sin_pos[t]
    //    K analogous.
    //    Then ReLU backward: d Q[t, j] = (Q[t, j] > 0) ? d ReLU(Q)[t, j] : 0
    Tensor d_Q_relu(n, d), d_K_relu(n, d);
    Tensor d_Q(n, d), d_K(n, d);
    for (size_t t = 0; t < n; ++t) {
        double c = cos_pos_(t, 0);
        double s = sin_pos_(t, 0);
        for (size_t j = 0; j < d; ++j) {
            d_Q_relu(t, j) = d_Q_cos(t, j) * c + d_Q_sin(t, j) * s;
            d_K_relu(t, j) = d_K_cos(t, j) * c + d_K_sin(t, j) * s;

            // ReLU backward. Need Q, K from forward — we cached last_q_relu_, last_k_relu_.
            d_Q(t, j) = last_q_relu_(t, j) > 0.0 ? d_Q_relu(t, j) : 0.0;
            d_K(t, j) = last_k_relu_(t, j) > 0.0 ? d_K_relu(t, j) : 0.0;
        }
    }

    // 6) Backward through projections: Q = X @ W_q^T, K = X @ W_k^T, V = X @ W_v^T
    //    d X  += d Q @ W_q  +  d K @ W_k  +  d V @ W_v   (n, d)
    //    grad_W_q += d Q^T @ X
    //    grad_W_k += d K^T @ X
    //    grad_W_v += d V^T @ X
    Tensor d_input(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) {
                s += d_Q(i, k) * W_q(k, j);
                s += d_K(i, k) * W_k(k, j);
                s += d_V(i, k) * W_v(k, j);
            }
            d_input(i, j) = s;
        }
    }
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sq = 0.0, sk = 0.0, sv = 0.0;
            for (size_t t = 0; t < n; ++t) {
                sq += d_Q(t, i) * last_input_(t, j);
                sk += d_K(t, i) * last_input_(t, j);
                sv += d_V(t, i) * last_input_(t, j);
            }
            grad_W_q(i, j) += sq;
            grad_W_k(i, j) += sk;
            grad_W_v(i, j) += sv;
        }

    return d_input;
}

// ============================================================================
// update_weights / zero_grad
// ============================================================================

void CosFormerAttention::update_weights(double learning_rate) {
    if (learning_rate == 0.0) return;
    auto sgd = [&](Tensor& W, Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) -= learning_rate * gW(i, j);
    };
    sgd(W_q, grad_W_q);
    sgd(W_k, grad_W_k);
    sgd(W_v, grad_W_v);
    sgd(W_o, grad_W_o);
}

void CosFormerAttention::zero_grad() {
    for (size_t i = 0; i < grad_W_q.rows; ++i)
        for (size_t j = 0; j < grad_W_q.cols; ++j) {
            grad_W_q(i, j) = 0.0;
            grad_W_k(i, j) = 0.0;
            grad_W_v(i, j) = 0.0;
            grad_W_o(i, j) = 0.0;
        }
}

// ============================================================================
// parameters / gradients
// ============================================================================

std::vector<Tensor*> CosFormerAttention::parameters() {
    return { &W_q, &W_k, &W_v, &W_o };
}

std::vector<Tensor*> CosFormerAttention::gradients() {
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o };
}

// ============================================================================
// Block
// ============================================================================

CosFormerBlock::CosFormerBlock(size_t d_model, size_t seq_len, size_t ffn_dim, size_t M)
    : d_model_(d_model), ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model), attn_(d_model, seq_len, M), ln2_(d_model),
      ffn_fc1_(d_model_, ffn_dim_), ffn_fc2_(ffn_dim_, d_model_) {}

Tensor CosFormerBlock::forward(const Tensor& input) {
    last_input_ = input;
    last_z1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_z1_);

    // residual 1
    last_res1_ = Tensor(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_res1_(i, j) = input(i, j) + last_attn_out_(i, j);

    last_z2_ = ln2_.forward(last_res1_);
    last_ffn_pre_gelu_ = ffn_fc1_.forward(last_z2_);
    last_ffn_hidden_ = Tensor(last_ffn_pre_gelu_.rows, last_ffn_pre_gelu_.cols);
    // GELU activation (inline)
    static const double GELU_COEF = std::sqrt(2.0 / M_PI);
    for (size_t i = 0; i < last_ffn_hidden_.rows; ++i)
        for (size_t j = 0; j < last_ffn_hidden_.cols; ++j) {
            double x = last_ffn_pre_gelu_(i, j);
            last_ffn_hidden_(i, j) = 0.5 * x * (1.0 + std::tanh(GELU_COEF * (x + 0.044715 * x * x * x)));
        }
    last_ffn_out_ = ffn_fc2_.forward(last_ffn_hidden_);

    Tensor out(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            out(i, j) = last_res1_(i, j) + last_ffn_out_(i, j);
    return out;
}

Tensor CosFormerBlock::backward(const Tensor& grad_output, double lr) {
    const size_t n = grad_output.rows;

    // d_res1 = grad_output + d_ffn_out
    Tensor d_res1(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_res1(i, j) = grad_output(i, j);

    // d_ffn_out
    Tensor d_ffn_hidden = ffn_fc2_.backward(d_res1, lr);
    // GELU backward (use cached pre-GELU activations)
    static const double GELU_COEF = std::sqrt(2.0 / M_PI);
    Tensor d_ffn_pre(n, ffn_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double pre = last_ffn_pre_gelu_(i, j);
            // GELU derivative (tanh approximation, matches the activations.h GELU)
            double xc = std::max(-4.0, std::min(4.0, pre));
            double u  = GELU_COEF * (xc + 0.044715 * xc * xc * xc);
            double th = std::tanh(u);
            double du = GELU_COEF * (1.0 + 3.0 * 0.044715 * xc * xc);
            double dgelu = 0.5 * (1.0 + th) + 0.5 * xc * (1.0 - th * th) * du;
            d_ffn_pre(i, j) = d_ffn_hidden(i, j) * dgelu;
        }
    // dL/d(ln2_out) = ffn_fc1_.backward(dL/d(ffn_pre))
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_pre, lr);
    Tensor d_ln2 = ln2_.backward(d_z2, lr);
    // d_res1 += d_ln2 (residual path through ln2)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_res1(i, j) += d_ln2(i, j);

    Tensor d_attn = attn_.backward(d_res1, lr);
    Tensor d_ln1 = ln1_.backward(d_attn, lr);
    // d_input = d_ln1 + d_res1: the residual bypass from res1 = input + attn_out
    // sends the FULL dL/d(res1) to input (not just grad_output).
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_input(i, j) = d_ln1(i, j) + d_res1(i, j);
    return d_input;
}

void CosFormerBlock::update_weights(double lr) {
    attn_.update_weights(lr);
    ln1_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

void CosFormerBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> CosFormerBlock::parameters() {
    auto p = attn_.parameters();
    auto p1 = ln1_.parameters();
    auto p2 = ln2_.parameters();
    auto p3 = ffn_fc1_.parameters();
    auto p4 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    return p;
}

std::vector<Tensor*> CosFormerBlock::gradients() {
    auto g = attn_.gradients();
    auto g1 = ln1_.gradients();
    auto g2 = ln2_.gradients();
    auto g3 = ffn_fc1_.gradients();
    auto g4 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), g3.begin(), g3.end());
    g.insert(g.end(), g4.begin(), g4.end());
    return g;
}

// ============================================================================
// Model
// ============================================================================

CosFormerModel::CosFormerModel(size_t d_model, size_t seq_len, size_t out_features,
                              size_t num_blocks, size_t ffn_dim, size_t M)
    : d_model_(d_model), out_features_(out_features),
      classifier_(d_model, out_features) {
    if (num_blocks == 0) throw std::invalid_argument("CosFormerModel: num_blocks must be > 0");
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, seq_len, ffn_dim, M);
    }
}

Tensor CosFormerModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor x = input;
    for (auto& b : blocks_) x = b.forward(x);
    // mean-pool over sequence to a single (1, d) row, then classifier
    Tensor pooled(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < x.rows; ++t) s += x(t, j);
        pooled(0, j) = s / (double)x.rows;
    }
    return classifier_.forward(pooled);
}

Tensor CosFormerModel::backward(const Tensor& grad_output, double lr) {
    Tensor d_pooled = classifier_.backward(grad_output, lr);
    // broadcast gradient back across the sequence
    const size_t n = last_input_.rows;
    Tensor d_block_out(n, d_model_);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            d_block_out(t, j) = d_pooled(0, j) / (double)n;
    for (size_t i = blocks_.size(); i > 0; --i) {
        d_block_out = blocks_[i - 1].backward(d_block_out, lr);
    }
    return d_block_out;
}

void CosFormerModel::update_weights(double lr) {
    for (auto& b : blocks_) b.update_weights(lr);
    classifier_.update_weights(lr);
}

void CosFormerModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> CosFormerModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> CosFormerModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}