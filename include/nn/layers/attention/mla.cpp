// Multi-Head Latent Attention (MLA) — implementation
//   DeepSeek-AI 2024 (https://arxiv.org/abs/2405.04434)
//
// See mla.h for the full design write-up.
#include "mla.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// MLAAttention
// ============================================================================

MLAAttention::MLAAttention(size_t d_model, size_t num_heads, size_t d_c)
    : d_model_(d_model), num_heads_(num_heads),
      head_dim_(d_model / num_heads), d_c_(d_c),
      scale_(1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9))
{
    // Shared down-projections
    W_dq  = Tensor::random(d_model, d_c, 0.05);
    W_dkv = Tensor::random(d_model, d_c, 0.05);
    grad_W_dq  = Tensor::zeros(d_model, d_c);
    grad_W_dkv = Tensor::zeros(d_model, d_c);

    // Per-head up-projections, stacked as (d_model, d_c) with head blocks
    // along the first dim.
    W_uq = Tensor::random(d_model, d_c, 0.05);
    W_uk = Tensor::random(d_model, d_c, 0.05);
    W_uv = Tensor::random(d_model, d_c, 0.05);
    grad_W_uq = Tensor::zeros(d_model, d_c);
    grad_W_uk = Tensor::zeros(d_model, d_c);
    grad_W_uv = Tensor::zeros(d_model, d_c);

    // Output projection
    W_o     = Tensor::random(d_model, d_model, 0.05);
    grad_W_o = Tensor::zeros(d_model, d_model);
}

std::vector<Tensor*> MLAAttention::parameters() {
    return { &W_dq, &W_uq, &W_dkv, &W_uk, &W_uv, &W_o };
}

std::vector<Tensor*> MLAAttention::gradients() {
    return { &grad_W_dq, &grad_W_uq, &grad_W_dkv, &grad_W_uk, &grad_W_uv, &grad_W_o };
}

void MLAAttention::zero_grad() {
    grad_W_dq.fill(0.0);
    grad_W_uq.fill(0.0);
    grad_W_dkv.fill(0.0);
    grad_W_uk.fill(0.0);
    grad_W_uv.fill(0.0);
    grad_W_o.fill(0.0);
}

void MLAAttention::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_dq,  grad_W_dq);
    sgd(W_uq,  grad_W_uq);
    sgd(W_dkv, grad_W_dkv);
    sgd(W_uk,  grad_W_uk);
    sgd(W_uv,  grad_W_uv);
    sgd(W_o,   grad_W_o);
}

Tensor MLAAttention::forward(const Tensor& input) {
    // input: (n, d_model) — n tokens, d_model features
    size_t n = input.rows;
    size_t d = input.cols;
    (void)d;  // implicit: must equal d_model_

    // Clone input for the backward (caller may mutate)
    last_input_ = input;

    // ---- Step 1: shared down-projections ----
    // c_Q  = X @ W_dq : (n, d_c)
    // c_KV = X @ W_dkv: (n, d_c)
    Tensor c_Q(n, d_c_);
    Tensor c_KV(n, d_c_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < d_c_; ++c) {
            double sq = 0.0, sk = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                sq += input(i, k) * W_dq(k, c);
                sk += input(i, k) * W_dkv(k, c);
            }
            c_Q(i, c)  = sq;
            c_KV(i, c) = sk;
        }
    }
    last_c_q_  = c_Q;
    last_c_kv_ = c_KV;

    // ---- Step 2: per-head up-projections, attention, concat ----
    // Stacked Q/K/V of shape (n, d_model) (head blocks along dim 1).
    Tensor Q(n, d_model_), K(n, d_model_), V(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (size_t c = 0; c < d_c_; ++c) {
                qv += c_Q(i, c)  * W_uq(j, c);
                kv += c_KV(i, c) * W_uk(j, c);
                vv += c_KV(i, c) * W_uv(j, c);
            }
            Q(i, j) = qv;
            K(i, j) = kv;
            V(i, j) = vv;
        }
    }
    last_q_ = Q;
    last_k_ = K;
    last_v_ = V;

    // Per-head attention. attn[head] is (n, n), head_out[head] is (n, head_dim).
    std::vector<Tensor> attn(num_heads_, Tensor(n, n));
    Tensor head_out_cat(n, d_model_);
    head_out_cat.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Extract per-head slices: Q_h, K_h, V_h are (n, head_dim)
        // Q_h[t][dk] = Q[t, h*head_dim_ + dk]
        std::vector<std::vector<double>> Q_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> K_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> V_h(n, std::vector<double>(head_dim_));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h[t][dk] = Q(t, h * head_dim_ + dk);
                K_h[t][dk] = K(t, h * head_dim_ + dk);
                V_h[t][dk] = V(t, h * head_dim_ + dk);
            }
        }

        // scores[t][s] = sum_{dk} Q_h[t][dk] * K_h[s][dk] * scale
        std::vector<std::vector<double>> scores(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    v += Q_h[t][dk] * K_h[s][dk];
                scores[t][s] = v * scale_;
            }
        }

        // row-softmax: attn[t][s] = exp(scores[t][s] - m_t) / l_t
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            double m = scores[t][0];
            for (size_t s = 1; s < n; ++s) if (scores[t][s] > m) m = scores[t][s];
            double l = 0.0;
            for (size_t s = 0; s < n; ++s) {
                A[t][s] = std::exp(scores[t][s] - m);
                l += A[t][s];
            }
            double inv_l = 1.0 / l;
            for (size_t s = 0; s < n; ++s) A[t][s] *= inv_l;
        }

        // head_out_h[t][dk] = sum_s A[t][s] * V_h[s][dk]
        std::vector<std::vector<double>> head_out(n, std::vector<double>(head_dim_, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t s = 0; s < n; ++s)
                    v += A[t][s] * V_h[s][dk];
                head_out[t][dk] = v;
            }
        }

        // Scatter head_out into the concatenated output at columns [h*head_dim_ : (h+1)*head_dim_]
        for (size_t t = 0; t < n; ++t)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                head_out_cat(t, h * head_dim_ + dk) = head_out[t][dk];

        // Cache attn for backward
        for (size_t t = 0; t < n; ++t)
            for (size_t s = 0; s < n; ++s)
                attn[h](t, s) = A[t][s];
    }
    last_attn_ = Tensor(num_heads_ * n, n);
    for (size_t h = 0; h < num_heads_; ++h)
        for (size_t t = 0; t < n; ++t)
            for (size_t s = 0; s < n; ++s)
                last_attn_(h * n + t, s) = attn[h](t, s);

    last_head_out_ = head_out_cat;

    // ---- Step 3: output projection ----
    // output[i, j] = sum_k head_out_cat[i, k] * W_o[k, j]
    Tensor output(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                v += head_out_cat(i, k) * W_o(k, j);
            output(i, j) = v;
        }
    }

    return output;
}

Tensor MLAAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (n, d_model)
    size_t n = grad_output.rows;
    size_t d = grad_output.cols;
    (void)d;

    // ---- dW_o + d_head_out_cat ----
    // dW_o[k, j] += sum_i head_out_cat[i, k] * grad_output[i, j]
    // d_head_out_cat[i, k] = sum_j grad_output[i, j] * W_o[k, j]
    Tensor d_head_out_cat(n, d_model_);
    d_head_out_cat.fill(0.0);
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < n; ++i) {
                wg += last_head_out_(i, k) * grad_output(i, j);
            }
            grad_W_o(k, j) += wg;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                v += grad_output(i, j) * W_o(k, j);
            d_head_out_cat(i, k) = v;
        }
    }

    // Per-head gradients and accumulation into c_Q, c_KV shared latents
    Tensor d_c_Q(n, d_c_);
    Tensor d_c_KV(n, d_c_);
    d_c_Q.fill(0.0);
    d_c_KV.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Extract per-head slices from last_q_, last_k_, last_v_ and from d_head_out_cat
        std::vector<std::vector<double>> Q_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> K_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> V_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> d_head_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h[t][dk] = last_q_(t, h * head_dim_ + dk);
                K_h[t][dk] = last_k_(t, h * head_dim_ + dk);
                V_h[t][dk] = last_v_(t, h * head_dim_ + dk);
                d_head_h[t][dk] = d_head_out_cat(t, h * head_dim_ + dk);
            }
        }
        for (size_t t = 0; t < n; ++t)
            for (size_t s = 0; s < n; ++s)
                A[t][s] = last_attn_(h * n + t, s);

        // dV_h[i, j] = sum_t A[t, i] * d_head_h[t, j]
        // dA[t, s]   = sum_j d_head_h[t, j] * V_h[s, j]
        std::vector<std::vector<double>> dV(n, std::vector<double>(head_dim_, 0.0));
        std::vector<std::vector<double>> dA(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double v = 0.0;
                for (size_t t = 0; t < n; ++t) v += A[t][i] * d_head_h[t][j];
                dV[i][j] = v;
            }
        }
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double v = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) v += d_head_h[t][j] * V_h[s][j];
                dA[t][s] = v;
            }
        }

        // Softmax backward: d_scores[t, s] = A[t, s] * (dA[t, s] - sum_c A[t, c] * dA[t, c])
        // We need the row sum sum_c A[t, c] * dA[t, c] per row.
        std::vector<double> row_dot(n, 0.0);
        for (size_t t = 0; t < n; ++t) {
            double s = 0.0;
            for (size_t c = 0; c < n; ++c) s += A[t][c] * dA[t][c];
            row_dot[t] = s;
        }
        std::vector<std::vector<double>> dS(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                dS[t][s] = A[t][s] * (dA[t][s] - row_dot[t]);
            }
        }

        // dQ_h[t, j] = scale * sum_s dS[t, s] * K_h[s, j]
        // dK_h[s, j] = scale * sum_t dS[t, s] * Q_h[t, j]
        std::vector<std::vector<double>> dQ(n, std::vector<double>(head_dim_, 0.0));
        std::vector<std::vector<double>> dK(n, std::vector<double>(head_dim_, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double vq = 0.0;
                for (size_t s = 0; s < n; ++s) vq += dS[t][s] * K_h[s][j];
                dQ[t][j] = vq * scale_;
            }
        }
        for (size_t s = 0; s < n; ++s) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double vk = 0.0;
                for (size_t t = 0; t < n; ++t) vk += dS[t][s] * Q_h[t][j];
                dK[s][j] = vk * scale_;
            }
        }

        // dW_uq[h*head_dim + j, c] += sum_i c_Q[i, c] * dQ[i, j]
        // dW_uk[h*head_dim + j, c] += sum_i c_KV[i, c] * dK[i, j]
        // dW_uv[h*head_dim + j, c] += sum_i c_KV[i, c] * dV[i, j]
        size_t h_off = h * head_dim_;
        for (size_t j = 0; j < head_dim_; ++j) {
            for (size_t c = 0; c < d_c_; ++c) {
                double guq = 0.0, guk = 0.0, guv = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    guq += last_c_q_(i, c)  * dQ[i][j];
                    guk += last_c_kv_(i, c) * dK[i][j];
                    guv += last_c_kv_(i, c) * dV[i][j];
                }
                grad_W_uq(h_off + j, c) += guq;
                grad_W_uk(h_off + j, c) += guk;
                grad_W_uv(h_off + j, c) += guv;
            }
        }

        // d_c_Q[i, c] += sum_j dQ[i, j] * W_uq[h_off + j, c]
        // d_c_KV[i, c] += sum_j dK[i, j] * W_uk[h_off + j, c]
        //                + sum_j dV[i, j] * W_uv[h_off + j, c]   (the shared-latent coupling)
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < d_c_; ++c) {
                double dcq = 0.0, dck = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    dcq += dQ[i][j] * W_uq(h_off + j, c);
                    dck += dK[i][j] * W_uk(h_off + j, c);
                    dck += dV[i][j] * W_uv(h_off + j, c);
                }
                d_c_Q(i, c)  += dcq;
                d_c_KV(i, c) += dck;
            }
        }
    }

    // dW_dq[k, c] += sum_i X[i, k] * d_c_Q[i, c]
    // dW_dkv[k, c] += sum_i X[i, k] * d_c_KV[i, c]
    // dX[i, k] = sum_c d_c_Q[i, c] * W_dq[k, c] + sum_c d_c_KV[i, c] * W_dkv[k, c]
    Tensor d_input(n, d_model_);
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t c = 0; c < d_c_; ++c) {
            double gdq = 0.0, gdkv = 0.0;
            for (size_t i = 0; i < n; ++i) {
                gdq  += last_input_(i, k) * d_c_Q(i, c);
                gdkv += last_input_(i, k) * d_c_KV(i, c);
            }
            grad_W_dq(k, c)  += gdq;
            grad_W_dkv(k, c) += gdkv;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t c = 0; c < d_c_; ++c) {
                v += d_c_Q(i, c)  * W_dq(k, c);
                v += d_c_KV(i, c) * W_dkv(k, c);
            }
            d_input(i, k) = v;
        }
    }

    return d_input;
}

// ============================================================================
// MLABlock — pre-LN → MLAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

MLABlock::MLABlock(size_t d_model, size_t num_heads, size_t d_c, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_heads, d_c),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),    // in=d_model, out=ffn_dim
      ffn_fc2_(ffn_dim_, d_model)     // in=ffn_dim, out=d_model
{}

std::vector<Tensor*> MLABlock::parameters() {
    auto p = ln1_.parameters();
    auto a = attn_.parameters();
    auto q = ln2_.parameters();
    auto f1 = ffn_fc1_.parameters();
    auto f2 = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

std::vector<Tensor*> MLABlock::gradients() {
    auto p = ln1_.gradients();
    auto a = attn_.gradients();
    auto q = ln2_.gradients();
    auto f1 = ffn_fc1_.gradients();
    auto f2 = ffn_fc2_.gradients();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

void MLABlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void MLABlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor MLABlock::forward(const Tensor& input) {
    // input: (n, d_model)
    last_input_ = input;
    Tensor z1 = ln1_.forward(input);                 // pre-LN
    Tensor attn_out = attn_.forward(z1);             // (n, d_model)
    last_attn_out_ = attn_out;
    Tensor res1 = z1 + attn_out;                     // residual
    last_res1_ = res1;
    Tensor z2 = ln2_.forward(res1);
    last_z2_ = z2;
    Tensor ffn_hidden_pre = ffn_fc1_.forward(z2);
    last_ffn_hidden_ = ffn_hidden_pre;   // PRE-GELU value (needed for GELU backward)
    // GELU element-wise (in place)
    for (size_t i = 0; i < ffn_hidden_pre.data.size(); ++i) {
        double x = ffn_hidden_pre.data[i];
        // GELU (exact erf-based) = x * Phi(x) where Phi is the standard normal CDF
        ffn_hidden_pre.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    Tensor ffn_out = ffn_fc2_.forward(ffn_hidden_pre);
    last_ffn_out_ = ffn_out;
    Tensor output = res1 + ffn_out;                  // second residual
    return output;
}

Tensor MLABlock::backward(const Tensor& grad_output, double lr) {
    // grad_output: dL/d output (n, d_model)
    // output = res1 + ffn_out
    // d_res1 (direct from residual) = grad_output
    // d_ffn_out (direct from residual) = grad_output
    Tensor d_ffn_out = grad_output;

    // d_ffn_h = ffn_fc2.backward(d_ffn_out, lr) — gives dL/d(ffn_h) which is post-GELU
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_out, lr);
    // GELU backward: d_ffn_h_pre = d_ffn_h * gelu'(ffn_h_pre)
    // last_ffn_hidden_ is the POST-GELU value; we need the PRE-GELU value
    // (the ffn_fc1 output). We re-derive it: ffn_h_pre = fc1(z2), but we
    // don't cache ffn_h_pre directly. Since gelu(x) ≈ x for the small
    // region where the loss is non-trivial (and we cache post-gelu as
    // last_ffn_hidden_), we cache ffn_h_pre explicitly via a copy.
    // Fix: use last_ffn_hidden_ as PRE-GELU (the ffn_fc1 output).
    Tensor d_ffn_h_pre(d_ffn_h.rows, d_ffn_h.cols);
    for (size_t i = 0; i < d_ffn_h_pre.data.size(); ++i) {
        double x = last_ffn_hidden_.data[i];   // PRE-GELU value
        double dy = d_ffn_h.data[i];
        double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        double gelu_prime = cdf + x * pdf;
        d_ffn_h_pre.data[i] = dy * gelu_prime;
    }
    // d_z2 = ffn_fc1.backward(d_ffn_h_pre, lr) — gives dL/d(z2)
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_h_pre, lr);
    // d_res1_from_ln2 = ln2.backward(d_z2, lr) — gives dL/d(res1) via the ln2 chain
    Tensor d_res1_from_ln2 = ln2_.backward(d_z2, lr);
    // d_res1_total = d_res1_direct + d_res1_from_ln2 = grad_output + d_res1_from_ln2
    Tensor d_res1(d_res1_from_ln2.rows, d_res1_from_ln2.cols);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] = grad_output.data[i] + d_res1_from_ln2.data[i];
    // res1 = z1 + attn_out, so:
    //   d_z1 from residual branch = d_res1 (the direct + residual contribution)
    //   d_attn_out = d_res1
    Tensor d_z1_residual = d_res1;
    Tensor d_attn_out = d_res1;
    // d_z1 from attn chain = attn_.backward(d_attn_out, lr)
    Tensor d_z1_from_attn = attn_.backward(d_attn_out, lr);
    // d_z1_total = d_z1_residual + d_z1_from_attn
    Tensor d_z1(d_z1_residual.rows, d_z1_residual.cols);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] = d_z1_residual.data[i] + d_z1_from_attn.data[i];
    // d_input = ln1.backward(d_z1, lr) — gives dL/d(input)
    Tensor d_input = ln1_.backward(d_z1, lr);
    return d_input;
}

// ============================================================================
// MLAModel — stack of MLABlocks + classifier head
// ============================================================================

MLAModel::MLAModel(size_t d_model, size_t num_heads, size_t d_c,
                   size_t out_features, size_t num_blocks, size_t ffn_dim)
    : d_model_(d_model), out_features_(out_features),
      blocks_(num_blocks, MLABlock(d_model, num_heads, d_c, ffn_dim)),
      classifier_(d_model, out_features)   // in=d_model, out=out_features
{}

std::vector<Tensor*> MLAModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> MLAModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

void MLAModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
}

void MLAModel::update_weights(double lr) {
    for (auto& b : blocks_) b.update_weights(lr);
    classifier_.update_weights(lr);
}

Tensor MLAModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor h = input;
    for (auto& b : blocks_) h = b.forward(h);
    Tensor logits = classifier_.forward(h);
    return logits;
}

Tensor MLAModel::backward(const Tensor& grad_output, double lr) {
    Tensor d = classifier_.backward(grad_output, lr);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d = it->backward(d, lr);
    }
    return d;
}
