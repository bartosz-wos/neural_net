// SHLA (Shared-Head Latent Attention) — implementation
//   Variant of MLA with SHARED K and V up-projection matrices across heads.
//
// See shla.h for the full design write-up.
#include "shla.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// SHLAAttention
// ============================================================================

SHLAAttention::SHLAAttention(size_t d_model, size_t num_heads, size_t d_c)
    : d_model_(d_model), num_heads_(num_heads),
      head_dim_(0),  // placeholder; set after validation succeeds
      d_c_(d_c),
      scale_(1.0)    // placeholder; set after validation succeeds
{
    // Validation FIRST (before any division — prevents SIGFPE on bad inputs,
    // a defensive pattern other layers in this repo use)
    if (d_model_ == 0 || num_heads_ == 0 || d_c_ == 0) {
        throw std::invalid_argument("SHLAAttention: d_model, num_heads, d_c must all be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("SHLAAttention: d_model must be divisible by num_heads");
    }

    // Now safe to compute divisions
    head_dim_ = d_model_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9);

    // Shared down-projections (d_model, d_c)
    W_dq       = Tensor::random(d_model, d_c, 0.05);
    W_dkv      = Tensor::random(d_model, d_c, 0.05);
    grad_W_dq  = Tensor::zeros(d_model, d_c);
    grad_W_dkv = Tensor::zeros(d_model, d_c);

    // Per-head up-projection for Q, stacked as (d_model, d_c)
    W_uq       = Tensor::random(d_model, d_c, 0.05);
    grad_W_uq  = Tensor::zeros(d_model, d_c);

    // SHARED up-projection for K (head_dim, d_c)
    W_uk_shared        = Tensor::random(head_dim_, d_c, 0.05);
    grad_W_uk_shared   = Tensor::zeros(head_dim_, d_c);

    // SHARED up-projection for V (head_dim, d_c)
    W_uv_shared        = Tensor::random(head_dim_, d_c, 0.05);
    grad_W_uv_shared   = Tensor::zeros(head_dim_, d_c);

    // Output projection (d_model, d_model)
    W_o      = Tensor::random(d_model, d_model, 0.05);
    grad_W_o = Tensor::zeros(d_model, d_model);
}

std::vector<Tensor*> SHLAAttention::parameters() {
    return { &W_dq, &W_uq, &W_dkv, &W_uk_shared, &W_uv_shared, &W_o };
}

std::vector<Tensor*> SHLAAttention::gradients() {
    return { &grad_W_dq, &grad_W_uq, &grad_W_dkv, &grad_W_uk_shared, &grad_W_uv_shared, &grad_W_o };
}

void SHLAAttention::zero_grad() {
    grad_W_dq.fill(0.0);
    grad_W_uq.fill(0.0);
    grad_W_dkv.fill(0.0);
    grad_W_uk_shared.fill(0.0);
    grad_W_uv_shared.fill(0.0);
    grad_W_o.fill(0.0);
}

void SHLAAttention::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_dq,        grad_W_dq);
    sgd(W_uq,        grad_W_uq);
    sgd(W_dkv,       grad_W_dkv);
    sgd(W_uk_shared, grad_W_uk_shared);
    sgd(W_uv_shared, grad_W_uv_shared);
    sgd(W_o,         grad_W_o);
}

Tensor SHLAAttention::forward(const Tensor& input) {
    // input: (n, d_model)
    size_t n = input.rows;
    (void)input.cols;  // implicit: must equal d_model_

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

    // ---- Step 2: per-head Q up-projection ----
    // Q = c_Q @ W_uq : (n, d_model)   (per-head stack along output dim)
    Tensor Q(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t c = 0; c < d_c_; ++c) v += c_Q(i, c) * W_uq(j, c);
            Q(i, j) = v;
        }
    }
    last_q_ = Q;

    // ---- Step 2b: SHARED K and V up-projections (computed once) ----
    // K = c_KV @ W_uk_shared^T : (n, head_dim)
    //    We store W_uk_shared as (head_dim, d_c), so K[i, j] = sum_c c_KV[i, c] * W_uk_shared[j, c]
    // V = c_KV @ W_uv_shared^T : (n, head_dim)
    Tensor K(n, head_dim_);
    Tensor V(n, head_dim_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < head_dim_; ++j) {
            double vk = 0.0, vv = 0.0;
            for (size_t c = 0; c < d_c_; ++c) {
                vk += c_KV(i, c) * W_uk_shared(j, c);
                vv += c_KV(i, c) * W_uv_shared(j, c);
            }
            K(i, j) = vk;
            V(i, j) = vv;
        }
    }
    last_k_ = K;
    last_v_ = V;

    // ---- Step 3: per-head attention ----
    // For each head h:
    //   Q_h[t, dk] = Q[t, h*head_dim + dk]            (n, head_dim)
    //   K_h[t, dk] = K[t, dk]                         (n, head_dim) — SAME for all heads
    //   V_h[t, dk] = V[t, dk]                         (n, head_dim) — SAME for all heads
    //   scores[t, s] = sum_dk Q_h[t, dk] * K_h[s, dk] * scale
    //   A[t, s] = softmax_row(scores[t])
    //   head_out_h[t, dk] = sum_s A[t, s] * V_h[s, dk]
    //   concat: out_concat[t, h*head_dim + dk] = head_out_h[t, dk]
    std::vector<Tensor> attn(num_heads_, Tensor(n, n));
    Tensor head_out_cat(n, d_model_);
    head_out_cat.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Extract per-head Q slice (K_h and V_h are the shared K and V — same for all heads)
        std::vector<std::vector<double>> Q_h(n, std::vector<double>(head_dim_));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h[t][dk] = Q(t, h * head_dim_ + dk);
            }
        }

        // scores[t][s] = sum_dk Q_h[t, dk] * K[s, dk] * scale
        std::vector<std::vector<double>> scores(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    v += Q_h[t][dk] * K(s, dk);
                scores[t][s] = v * scale_;
            }
        }

        // row-softmax
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

        // head_out_h[t, dk] = sum_s A[t, s] * V[s, dk]
        std::vector<std::vector<double>> head_out(n, std::vector<double>(head_dim_, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t s = 0; s < n; ++s) v += A[t][s] * V(s, dk);
                head_out[t][dk] = v;
            }
        }

        // Scatter into head_out_cat at columns [h*head_dim_ : (h+1)*head_dim_]
        for (size_t t = 0; t < n; ++t)
            for (size_t dk = 0; dk < head_dim_; ++dk)
                head_out_cat(t, h * head_dim_ + dk) = head_out[t][dk];

        // Cache A
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

    // ---- Step 4: output projection ----
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

Tensor SHLAAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (n, d_model)
    size_t n = grad_output.rows;
    (void)grad_output.cols;

    // ---- dW_o + d_head_out_cat ----
    // dW_o[k, j] += sum_i head_out_cat[i, k] * grad_output[i, j]
    // d_head_out_cat[i, k] = sum_j grad_output[i, j] * W_o[k, j]
    Tensor d_head_out_cat(n, d_model_);
    d_head_out_cat.fill(0.0);
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < n; ++i) wg += last_head_out_(i, k) * grad_output(i, j);
            grad_W_o(k, j) += wg;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) v += grad_output(i, j) * W_o(k, j);
            d_head_out_cat(i, k) = v;
        }
    }

    // ---- Per-head gradients ----
    // Note: K and V are SHARED across heads, so dK_h, dV_h for each head all
    // funnel into the SAME dK, dV (summed across heads in the gradient
    // accumulators for W_uk_shared, W_uv_shared, and d_c_KV).
    //
    // We use per-head temporaries dQ_h, dK_h, dV_h; then accumulate dK_h, dV_h
    // across heads into the SHARED dK, dV. Per-head dQ goes only to its own
    // block of W_uq (per-head) and to d_c_Q via the per-head up-projection.
    // dK, dV are accumulated across heads since K, V are SHARED across all heads
    std::vector<double> dK(n * head_dim_, 0.0);
    std::vector<double> dV(n * head_dim_, 0.0);

    // Tensor-backed d_c_Q and d_c_KV accumulators (over the shared latents)
    Tensor d_c_Q(n, d_c_);
    Tensor d_c_KV(n, d_c_);
    d_c_Q.fill(0.0);
    d_c_KV.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Q slice for this head, K_h and V_h are the shared K, V (same for all heads)
        std::vector<std::vector<double>> Q_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> d_head_h(n, std::vector<double>(head_dim_));
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                Q_h[t][dk] = last_q_(t, h * head_dim_ + dk);
                d_head_h[t][dk] = d_head_out_cat(t, h * head_dim_ + dk);
            }
        }
        for (size_t t = 0; t < n; ++t)
            for (size_t s = 0; s < n; ++s)
                A[t][s] = last_attn_(h * n + t, s);

        // dV_h[i, dk] = sum_t A[t, i] * d_head_h[t, dk]     (per-head dV)
        // dA[t, s]    = sum_dk d_head_h[t, dk] * V[s, dk]   (uses shared V)
        std::vector<std::vector<double>> dV_h(n, std::vector<double>(head_dim_, 0.0));
        std::vector<std::vector<double>> dA(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double v = 0.0;
                for (size_t t = 0; t < n; ++t) v += A[t][i] * d_head_h[t][j];
                dV_h[i][j] = v;
            }
        }
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s < n; ++s) {
                double v = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) v += d_head_h[t][j] * last_v_(s, j);
                dA[t][s] = v;
            }
        }

        // Softmax backward: d_scores[t, s] = A[t, s] * (dA[t, s] - row_dot[t])
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

        // dQ_h[t, j] = scale * sum_s dS[t, s] * K[s, j]   (uses shared K)
        // dK_h[s, j] = scale * sum_t dS[t, s] * Q_h[t, j]  (per-head dK into shared K)
        std::vector<std::vector<double>> dQ_h(n, std::vector<double>(head_dim_, 0.0));
        std::vector<std::vector<double>> dK_h(n, std::vector<double>(head_dim_, 0.0));
        for (size_t t = 0; t < n; ++t) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double vq = 0.0;
                for (size_t s = 0; s < n; ++s) vq += dS[t][s] * last_k_(s, j);
                dQ_h[t][j] = vq * scale_;
            }
        }
        for (size_t s = 0; s < n; ++s) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double vk = 0.0;
                for (size_t t = 0; t < n; ++t) vk += dS[t][s] * Q_h[t][j];
                dK_h[s][j] = vk * scale_;
            }
        }

        // Accumulate dK_h, dV_h across heads into the SHARED K, V gradients
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < head_dim_; ++j) {
                dK[i * head_dim_ + j] += dK_h[i][j];
                dV[i * head_dim_ + j] += dV_h[i][j];
            }

        // dW_uq[h_off + j, c] += sum_i c_Q[i, c] * dQ_h[i, j]   (per-head)
        // d_c_Q[i, c] += sum_j dQ_h[i, j] * W_uq[h_off + j, c]  (per-head fan-in)
        size_t h_off = h * head_dim_;
        for (size_t j = 0; j < head_dim_; ++j) {
            for (size_t c = 0; c < d_c_; ++c) {
                double guq = 0.0;
                for (size_t i = 0; i < n; ++i) guq += last_c_q_(i, c) * dQ_h[i][j];
                grad_W_uq(h_off + j, c) += guq;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < d_c_; ++c) {
                double v = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) v += dQ_h[i][j] * W_uq(h_off + j, c);
                d_c_Q(i, c) += v;
            }
        }
    }

    // ---- Compute SHARED dW_uk_shared and dW_uv_shared from the accumulated
    //      dK, dV (summed across all heads) ----
    // dW_uk_shared[j, c] += sum_i c_KV[i, c] * dK[i, j]   (where dK is summed across heads)
    // dW_uv_shared[j, c] += sum_i c_KV[i, c] * dV[i, j]
    for (size_t j = 0; j < head_dim_; ++j) {
        for (size_t c = 0; c < d_c_; ++c) {
            double guk = 0.0, guv = 0.0;
            for (size_t i = 0; i < n; ++i) {
                guk += last_c_kv_(i, c) * dK[i * head_dim_ + j];
                guv += last_c_kv_(i, c) * dV[i * head_dim_ + j];
            }
            grad_W_uk_shared(j, c) += guk;
            grad_W_uv_shared(j, c) += guv;
        }
    }

    // ---- Compute SHARED d_c_KV from dK and dV (summed across all heads,
    //      and accumulating contributions from BOTH the K and V chains) ----
    // d_c_KV[i, c] += sum_j dK[i, j] * W_uk_shared[j, c]
    //               +  sum_j dV[i, j] * W_uv_shared[j, c]
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < d_c_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < head_dim_; ++j) {
                v += dK[i * head_dim_ + j] * W_uk_shared(j, c);
                v += dV[i * head_dim_ + j] * W_uv_shared(j, c);
            }
            d_c_KV(i, c) += v;
        }
    }

    // ---- dW_dq, dW_dkv, d_input ----
    // dW_dq[k, c]  += sum_i X[i, k] * d_c_Q[i, c]
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

    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// SHLABlock — pre-LN → SHLAAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

SHLABlock::SHLABlock(size_t d_model, size_t num_heads, size_t d_c, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_heads, d_c),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),    // in=d_model, out=ffn_dim
      ffn_fc2_(ffn_dim_, d_model)     // in=ffn_dim, out=d_model
{}

std::vector<Tensor*> SHLABlock::parameters() {
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

std::vector<Tensor*> SHLABlock::gradients() {
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

void SHLABlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void SHLABlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor SHLABlock::forward(const Tensor& input) {
    last_input_ = input;
    Tensor z1 = ln1_.forward(input);
    Tensor attn_out = attn_.forward(z1);
    last_attn_out_ = attn_out;
    Tensor res1 = z1 + attn_out;
    last_res1_ = res1;
    Tensor z2 = ln2_.forward(res1);
    last_z2_ = z2;
    Tensor ffn_hidden_pre = ffn_fc1_.forward(z2);
    last_ffn_hidden_ = ffn_hidden_pre;   // PRE-GELU
    // GELU element-wise (in place)
    for (size_t i = 0; i < ffn_hidden_pre.data.size(); ++i) {
        double x = ffn_hidden_pre.data[i];
        ffn_hidden_pre.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    Tensor ffn_out = ffn_fc2_.forward(ffn_hidden_pre);
    last_ffn_out_ = ffn_out;
    Tensor output = res1 + ffn_out;
    return output;
}

Tensor SHLABlock::backward(const Tensor& grad_output, double lr) {
    Tensor d_ffn_out = grad_output;
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_out, lr);
    Tensor d_ffn_h_pre(d_ffn_h.rows, d_ffn_h.cols);
    for (size_t i = 0; i < d_ffn_h_pre.data.size(); ++i) {
        double x = last_ffn_hidden_.data[i];
        double dy = d_ffn_h.data[i];
        double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        double gelu_prime = cdf + x * pdf;
        d_ffn_h_pre.data[i] = dy * gelu_prime;
    }
    Tensor d_z2 = ffn_fc1_.backward(d_ffn_h_pre, lr);
    Tensor d_res1_from_ln2 = ln2_.backward(d_z2, lr);
    Tensor d_res1(d_res1_from_ln2.rows, d_res1_from_ln2.cols);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] = grad_output.data[i] + d_res1_from_ln2.data[i];
    Tensor d_z1_residual = d_res1;
    Tensor d_attn_out = d_res1;
    Tensor d_z1_from_attn = attn_.backward(d_attn_out, lr);
    Tensor d_z1(d_z1_residual.rows, d_z1_residual.cols);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] = d_z1_residual.data[i] + d_z1_from_attn.data[i];
    Tensor d_input = ln1_.backward(d_z1, lr);
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// SHLAModel — stack of SHLABlocks + classifier head
// ============================================================================

SHLAModel::SHLAModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_blocks, size_t num_heads, size_t d_c, size_t ffn_dim)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim)
{
    W_in_       = Tensor::random(input_dim, d_model, 0.05);
    b_in_       = Tensor::random(1, d_model, 0.05);
    grad_W_in_  = Tensor::zeros(input_dim, d_model);
    grad_b_in_  = Tensor::zeros(1, d_model);
    W_out_      = Tensor::random(d_model, output_dim, 0.05);
    b_out_      = Tensor::random(1, output_dim, 0.05);
    grad_W_out_ = Tensor::zeros(d_model, output_dim);
    grad_b_out_ = Tensor::zeros(1, output_dim);
    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, num_heads, d_c, ffn_dim);
    }
}

std::vector<Tensor*> SHLAModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_in_);
    p.push_back(&b_in_);
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&W_out_);
    p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> SHLAModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_W_in_);
    g.push_back(&grad_b_in_);
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_W_out_);
    g.push_back(&grad_b_out_);
    return g;
}

void SHLAModel::zero_grad() {
    grad_W_in_.fill(0.0);
    grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void SHLAModel::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_in_,  grad_W_in_);
    sgd(b_in_,  grad_b_in_);
    sgd(W_out_, grad_W_out_);
    sgd(b_out_, grad_b_out_);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor SHLAModel::forward(const Tensor& input) {
    last_input_ = input;
    // Project input: (n, input_dim) -> (n, d_model) by W_in_ + b_in_
    Tensor proj(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = b_in_(0, j);
            for (size_t k = 0; k < input_dim_; ++k) v += input(i, k) * W_in_(k, j);
            proj(i, j) = v;
        }
    }
    last_proj_ = proj;
    Tensor h = proj;
    for (auto& b : blocks_) h = b.forward(h);
    last_block_out_ = h;
    // Output projection: (n, d_model) -> (n, output_dim)
    Tensor output(h.rows, output_dim_);
    for (size_t i = 0; i < h.rows; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double v = b_out_(0, j);
            for (size_t k = 0; k < d_model_; ++k) v += h(i, k) * W_out_(k, j);
            output(i, j) = v;
        }
    }
    return output;
}

Tensor SHLAModel::backward(const Tensor& grad_output, double lr) {
    // grad_output: (n, output_dim)
    //   out[i, j] = b_out_[0, j] + sum_k last_block_out_[i, k] * W_out_[k, j]
    //   d_last_block_out_[i, k] = sum_j grad_output[i, j] * W_out_[k, j]
    //   dW_out_[k, j] += sum_i last_block_out_[i, k] * grad_output[i, j]
    //   db_out_[0, j] += sum_i grad_output[i, j]
    Tensor d_block_out(grad_output.rows, d_model_);
    d_block_out.fill(0.0);
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < output_dim_; ++j) v += grad_output(i, j) * W_out_(k, j);
            d_block_out(i, k) = v;
        }
    }
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < grad_output.rows; ++i) wg += last_block_out_(i, k) * grad_output(i, j);
            grad_W_out_(k, j) += wg;
        }
    }
    for (size_t j = 0; j < output_dim_; ++j) {
        double bg = 0.0;
        for (size_t i = 0; i < grad_output.rows; ++i) bg += grad_output(i, j);
        grad_b_out_(0, j) += bg;
    }

    // Backward through blocks (reverse order) using the cached block outputs
    Tensor d_proj = d_block_out;
    for (auto b_it = blocks_.rbegin(); b_it != blocks_.rend(); ++b_it) {
        d_proj = b_it->backward(d_proj, lr);
    }

    // Backward through input projection
    //   proj[i, j] = b_in_[0, j] + sum_k input[i, k] * W_in_[k, j]
    //   d_input[i, k] = sum_j d_proj[i, j] * W_in_[k, j]
    //   dW_in_[k, j] += sum_i input[i, k] * d_proj[i, j]
    //   db_in_[0, j] += sum_i d_proj[i, j]
    Tensor d_input(last_input_.rows, input_dim_);
    d_input.fill(0.0);
    for (size_t i = 0; i < last_input_.rows; ++i) {
        for (size_t k = 0; k < input_dim_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) v += d_proj(i, j) * W_in_(k, j);
            d_input(i, k) = v;
        }
    }
    for (size_t k = 0; k < input_dim_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double wg = 0.0;
            for (size_t i = 0; i < last_input_.rows; ++i) wg += last_input_(i, k) * d_proj(i, j);
            grad_W_in_(k, j) += wg;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        double bg = 0.0;
        for (size_t i = 0; i < last_input_.rows; ++i) bg += d_proj(i, j);
        grad_b_in_(0, j) += bg;
    }

    return d_input;
}
