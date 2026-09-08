// Tokenformer / Pattention — implementation
//   Wang et al., 2024, https://arxiv.org/abs/2410.23168
//
// See tokenformer.h for the design write-up and the backward derivation.
#include "tokenformer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Pattention
// ============================================================================

Pattention::Pattention(size_t d_in, size_t d_out, size_t num_param_tokens)
    : d_in_(d_in), d_out_(d_out), P_(num_param_tokens)
{
    if (d_in_ == 0 || d_out_ == 0 || P_ == 0) {
        throw std::invalid_argument("Pattention: d_in, d_out, num_param_tokens must all be > 0");
    }
    // Random init with small scale so softmax doesn't saturate at init.
    K_p = Tensor::random(P_, d_in_, 0.1);
    V_p = Tensor::random(P_, d_out_, 0.1);
    b   = Tensor::zeros(1, d_out_);
    logit_g = Tensor::random(P_, 1, 0.5);    // sigmoid(0.5*N(0,1)) ≈ 0.5 ± 0.1
    grad_K_p = Tensor::zeros(P_, d_in_);
    grad_V_p = Tensor::zeros(P_, d_out_);
    grad_b   = Tensor::zeros(1, d_out_);
    grad_logit_g = Tensor::zeros(P_, 1);
}

std::vector<Tensor*> Pattention::parameters() {
    return { &K_p, &V_p, &b, &logit_g };
}

std::vector<Tensor*> Pattention::gradients() {
    return { &grad_K_p, &grad_V_p, &grad_b, &grad_logit_g };
}

void Pattention::zero_grad() {
    grad_K_p.fill(0.0);
    grad_V_p.fill(0.0);
    grad_b.fill(0.0);
    grad_logit_g.fill(0.0);
}

void Pattention::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(K_p, grad_K_p);
    sgd(V_p, grad_V_p);
    sgd(b,   grad_b);
    sgd(logit_g, grad_logit_g);
}

Tensor Pattention::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_input_ = input.clone();

    // Gate values g_p = sigmoid(logit_g[p])
    std::vector<double> g(P_);
    for (size_t p = 0; p < P_; ++p) g[p] = 1.0 / (1.0 + std::exp(-logit_g(p, 0)));
    last_g_ = Tensor(P_, 1);
    for (size_t p = 0; p < P_; ++p) last_g_(p, 0) = g[p];

    // S = X · K_p^T  shape (n, P)
    Tensor S(n, P_);
    for (size_t i = 0; i < n; ++i)
        for (size_t p = 0; p < P_; ++p) {
            double s = 0.0;
            for (size_t c = 0; c < d_in_; ++c) s += input(i, c) * K_p(p, c);
            S(i, p) = s;
        }
    last_S_ = S.clone();

    // α = row_softmax(S), numerically stable (max-subtract per row).
    Tensor alpha(n, P_);
    for (size_t i = 0; i < n; ++i) {
        double m = S(i, 0);
        for (size_t p = 1; p < P_; ++p) m = std::max(m, S(i, p));
        double sum = 0.0;
        for (size_t p = 0; p < P_; ++p) {
            double e = std::exp(S(i, p) - m);
            alpha(i, p) = e;
            sum += e;
        }
        double inv = (sum > 0.0) ? 1.0 / sum : 0.0;
        for (size_t p = 0; p < P_; ++p) alpha(i, p) *= inv;
    }
    last_alpha_ = alpha.clone();

    // α_g[p] = g[p] · α[p] / Z_g,  Z_g = Σ g·α
    Tensor alpha_g(n, P_);
    std::vector<double> Z_g(n);
    for (size_t i = 0; i < n; ++i) {
        double z = 0.0;
        for (size_t p = 0; p < P_; ++p) z += g[p] * alpha(i, p);
        Z_g[i] = (z > 0.0) ? z : 1.0;       // safety: avoid /0
        double inv = 1.0 / Z_g[i];
        for (size_t p = 0; p < P_; ++p) alpha_g(i, p) = g[p] * alpha(i, p) * inv;
    }
    last_alpha_g_ = alpha_g.clone();
    last_Z_g_ = Tensor(n, 1);
    for (size_t i = 0; i < n; ++i) last_Z_g_(i, 0) = Z_g[i];

    // out = α_g · V_p + b   shape (n, d_out)
    Tensor out(n, d_out_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t o = 0; o < d_out_; ++o) {
            double s = b(0, o);
            for (size_t p = 0; p < P_; ++p) s += alpha_g(i, p) * V_p(p, o);
            out(i, o) = s;
        }
    }
    return out;
}

Tensor Pattention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = last_input_.rows;
    const Tensor& alpha = last_alpha_;
    const Tensor& alpha_g = last_alpha_g_;
    const Tensor& X = last_input_;
    const Tensor& Z_g_t = last_Z_g_;
    std::vector<double> Z_g(n);
    for (size_t i = 0; i < n; ++i) Z_g[i] = Z_g_t(i, 0);
    std::vector<double> g(P_);
    for (size_t p = 0; p < P_; ++p) g[p] = last_g_(p, 0);

    // dV_p[p, :] += α_g[t, p] · dO[t, :]
    for (size_t p = 0; p < P_; ++p)
        for (size_t o = 0; o < d_out_; ++o) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += alpha_g(i, p) * grad_output(i, o);
            grad_V_p(p, o) = s;
        }

    // db[:] += dO[t, :]
    for (size_t o = 0; o < d_out_; ++o) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += grad_output(i, o);
        grad_b(0, o) = s;
    }

    // dα_g[t, p] = dO[t, :] · V_p[p, :]^T   shape (n, P)
    Tensor d_alpha_g(n, P_);
    for (size_t i = 0; i < n; ++i)
        for (size_t p = 0; p < P_; ++p) {
            double s = 0.0;
            for (size_t o = 0; o < d_out_; ++o) s += grad_output(i, o) * V_p(p, o);
            d_alpha_g(i, p) = s;
        }

    // Backprop through α_g = (g · α) / Z_g.
    //   dα_g = (1/Z_g) · dg · α − (g · α / Z_g²) · dZ_g
    // where dZ_g = Σ dg · α + Σ g · dα
    //
    // Concretely, for a fixed t and varying p:
    //   Let u_p = g[p] · α[t, p]   (un-normalized gated weight)
    //   Z_g[t] = Σ u_p'
    //   α_g[t, p] = u_p / Z_g[t]
    //
    // Backward (treating u and Z_g as independent at first):
    //   d(u_p) = dα_g[t, p] / Z_g[t]                           (partial du_p, Z_g fixed)
    //   d(Z_g) = -Σ_{p'} dα_g[t, p'] · u_{p'} / Z_g[t]²       (partial Z_g, u fixed)
    //         = -Σ_{p'} dα_g[t, p'] · α_g[t, p'] / Z_g[t]
    //   Then dα[t, p] = g[p] · (d(u_p) + d(Z_g))
    //   And  dg[p] = Σ_t α[t, p] · (d(u_p) + d(Z_g))
    //
    // In code below we compute dα and dg directly.

    Tensor dX(n, d_in_);
    dX.fill(0.0);
    std::vector<double> dg(P_, 0.0);     // per-parameter-token gate gradient (pre-sigmoid: applied via grad_logit_g below)

    for (size_t i = 0; i < n; ++i) {
        double dZ_g = 0.0;
        for (size_t p = 0; p < P_; ++p) dZ_g -= d_alpha_g(i, p) * alpha_g(i, p) / Z_g[i];
        // dα[t, p] = g[p] · (d(u_p) + d(Z_g))
        //          = g[p] · (dα_g[i,p] / Z_g[i] + dZ_g)
        Tensor dalpha_row(n > 0 ? P_ : 0, 1);  // placeholder; we only need per-(i,p)
        std::vector<double> dalpha(P_);
        for (size_t p = 0; p < P_; ++p) {
            dalpha[p] = g[p] * (d_alpha_g(i, p) / Z_g[i] + dZ_g);
            dg[p] += alpha(i, p) * (d_alpha_g(i, p) / Z_g[i] + dZ_g);
        }
        // softmax backward: dS[i, p] = α[i, p] · (dα[i, p] − Σ_{p'} α[i, p'] · dα[i, p'])
        double row_dot = 0.0;
        for (size_t p = 0; p < P_; ++p) row_dot += alpha(i, p) * dalpha[p];
        for (size_t p = 0; p < P_; ++p) {
            double ds = alpha(i, p) * (dalpha[p] - row_dot);
            // dK_p[p, :] += ds · X[i, :]
            for (size_t c = 0; c < d_in_; ++c) grad_K_p(p, c) += ds * X(i, c);
            // dX[i, :] += Σ_p ds · K_p[p, :]
            for (size_t c = 0; c < d_in_; ++c) dX(i, c) += ds * K_p(p, c);
        }
    }

    // grad_logit_g[p] = dg[p] · sigmoid'(logit_g[p]) = dg[p] · g[p] · (1 − g[p])
    for (size_t p = 0; p < P_; ++p) {
        grad_logit_g(p, 0) = dg[p] * g[p] * (1.0 - g[p]);
    }

    return dX;
}

// ============================================================================
// TokenformerBlock
// ============================================================================

TokenformerBlock::TokenformerBlock(size_t d_model, size_t num_heads, size_t num_param_tokens)
    : d_model_(d_model), num_heads_(num_heads),
      attn_W_q(d_model, d_model), attn_W_k(d_model, d_model),
      attn_W_v(d_model, d_model), attn_W_o(d_model, d_model)
{
    if (d_model_ == 0) {
        throw std::invalid_argument("TokenformerBlock: d_model must be > 0");
    }
    if (num_heads_ == 0) {
        throw std::invalid_argument("TokenformerBlock: num_heads must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("TokenformerBlock: d_model must be divisible by num_heads");
    }
    if (num_param_tokens == 0) {
        throw std::invalid_argument("TokenformerBlock: num_param_tokens must be > 0");
    }

    head_dim_ = d_model_ / num_heads_;

    ln1_ = std::make_unique<LayerNorm>(d_model_);
    ln2_ = std::make_unique<LayerNorm>(d_model_);
    ffn_ = std::make_unique<Pattention>(d_model_, d_model_, num_param_tokens);
}

std::vector<Tensor*> TokenformerBlock::parameters() {
    // LayerNorm params first, then Dense weights/biases, then Pattention params.
    std::vector<Tensor*> ps;
    ps.push_back(&ln1_->gamma);
    ps.push_back(&ln1_->beta);
    ps.push_back(&attn_W_q.weights);
    ps.push_back(&attn_W_q.bias);
    ps.push_back(&attn_W_k.weights);
    ps.push_back(&attn_W_k.bias);
    ps.push_back(&attn_W_v.weights);
    ps.push_back(&attn_W_v.bias);
    ps.push_back(&attn_W_o.weights);
    ps.push_back(&attn_W_o.bias);
    for (auto* p : ffn_->parameters()) ps.push_back(p);
    return ps;
}

std::vector<Tensor*> TokenformerBlock::gradients() {
    std::vector<Tensor*> gs;
    gs.push_back(&ln1_->grad_gamma_);
    gs.push_back(&ln1_->grad_beta_);
    gs.push_back(&attn_W_q.grad_weights);
    gs.push_back(&attn_W_q.grad_bias);
    gs.push_back(&attn_W_k.grad_weights);
    gs.push_back(&attn_W_k.grad_bias);
    gs.push_back(&attn_W_v.grad_weights);
    gs.push_back(&attn_W_v.grad_bias);
    gs.push_back(&attn_W_o.grad_weights);
    gs.push_back(&attn_W_o.grad_bias);
    for (auto* g : ffn_->gradients()) gs.push_back(g);
    return gs;
}

void TokenformerBlock::zero_grad() {
    ln1_->zero_grad();
    ln2_->zero_grad();
    attn_W_q.zero_grad();
    attn_W_k.zero_grad();
    attn_W_v.zero_grad();
    attn_W_o.zero_grad();
    ffn_->zero_grad();
}

void TokenformerBlock::update_weights(double lr) {
    ln1_->update_weights(lr);
    ln2_->update_weights(lr);
    attn_W_q.update_weights(lr);
    attn_W_k.update_weights(lr);
    attn_W_v.update_weights(lr);
    attn_W_o.update_weights(lr);
    ffn_->update_weights(lr);
}

Tensor TokenformerBlock::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_input_ = input.clone();

    // ----- Self-attention sublayer (pre-LN -> attention -> residual) -----
    Tensor ln1_out = ln1_->forward(input);   // (n, d_model)
    // Q, K, V projections
    Tensor Q = attn_W_q.forward(ln1_out);
    Tensor K = attn_W_k.forward(ln1_out);
    Tensor V = attn_W_v.forward(ln1_out);
    last_q_ = Q; last_k_ = K; last_v_ = V;

    // Causal multi-head attention: O_h = softmax(Q_h K_h^T / sqrt(d_h)) V_h, j <= i.
    const double inv_temp = 1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9);
    Tensor attn_concat(n, d_model_);
    last_scores_.clear();
    last_scores_.reserve(num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        // scores[i, j] = Q_h[i] . K_h[j] * inv_temp, -inf for j > i
        Tensor scores(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (j > i) { scores(i, j) = -1e30; continue; }
                double s = 0.0;
                for (size_t c = 0; c < head_dim_; ++c) {
                    s += Q(i, h * head_dim_ + c) * K(j, h * head_dim_ + c);
                }
                scores(i, j) = s * inv_temp;
            }
        }
        // row softmax
        for (size_t i = 0; i < n; ++i) {
            double m = scores(i, 0);
            for (size_t j = 1; j < n; ++j) m = std::max(m, scores(i, j));
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) { scores(i, j) = std::exp(scores(i, j) - m); sum += scores(i, j); }
            double inv = (sum > 0.0) ? 1.0 / sum : 0.0;
            for (size_t j = 0; j < n; ++j) scores(i, j) *= inv;
        }
        // Cache the per-head softmax probs for backward.
        last_scores_.push_back(scores.clone());
        // attn_out[i, h*d_h + c] = sum_j scores[i, j] * V[j, h*d_h + c]
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < head_dim_; ++c) {
                double s = 0.0;
                for (size_t j = 0; j < n; ++j) s += scores(i, j) * V(j, h * head_dim_ + c);
                attn_concat(i, h * head_dim_ + c) = s;
            }
        }
    }

    // attn_out = attn_concat @ W_o
    Tensor attn_proj = attn_W_o.forward(attn_concat);

    // Residual: h1 = input + attn_proj
    Tensor h1(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            h1(i, j) = input(i, j) + attn_proj(i, j);

    // ----- FFN sublayer (pre-LN -> Pattention -> residual) -----
    Tensor ln2_out = ln2_->forward(h1);
    Tensor ffn_out = ffn_->forward(ln2_out);

    Tensor h2(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            h2(i, j) = h1(i, j) + ffn_out(i, j);
    return h2;
}

Tensor TokenformerBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = last_input_.rows;
    const Tensor& X = last_input_;

    // ----- FFN sublayer backward (residual -> ffn -> ln2 -> ...) -----
    Tensor d_h1(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_h1(i, j) = grad_output(i, j);     // identity of residual
    // d_input_to_ffn = d_h1 (residual)
    Tensor d_ffn = ffn_->backward(d_h1, 0.0);
    // d_ln2_out = d_ffn (identity through ffn_ output)
    Tensor d_ln2 = d_ffn;
    // d_h1_from_ln2 = ln2_->backward(d_ln2)
    Tensor d_h1_from_ln2 = ln2_->backward(d_ln2, 0.0);
    // d_h1 accumulates: d_h1 (from outer residual) + d_h1_from_ln2 (from ffn path through ln2)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_h1(i, j) += d_h1_from_ln2(i, j);

    // ----- Self-attention sublayer backward (residual -> attn_proj -> W_o -> attn_concat -> per-head -> Q/K/V -> ln1) -----
    Tensor d_attn_proj(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_attn_proj(i, j) = d_h1(i, j);
    // W_o backward (Dense::backward internally accumulates into grad_weights/grad_bias)
    Tensor d_attn_concat = attn_W_o.backward(d_attn_proj, 0.0);

    // Per-head backward through multi-head attention.
    Tensor d_ln1_out(d_model_ > 0 ? n : 0, d_model_);
    d_ln1_out.fill(0.0);
    const double inv_temp = 1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9);

    // We need per-head scores for the backward. Recompute them (cheap, and the
    // test fixture's pattern is "recompute forward state" for backward too).
    const Tensor& Q = last_q_;
    const Tensor& K = last_k_;
    const Tensor& V = last_v_;
    Tensor dQ(n, d_model_); dQ.fill(0.0);
    Tensor dK_(n, d_model_); dK_.fill(0.0);
    Tensor dV_(n, d_model_); dV_.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Use the cached per-head softmax probs from the forward.
        const Tensor& scores = last_scores_[h];
        // dV_h[c, j] = sum_i scores[i, j] * dO[i, c] ; dO = d_attn_concat[:, h*d_h:(h+1)*d_h]
        for (size_t j = 0; j < n; ++j) {
            for (size_t c = 0; c < head_dim_; ++c) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i) s += scores(i, j) * d_attn_concat(i, h * head_dim_ + c);
                dV_(j, h * head_dim_ + c) += s;
            }
        }
        // d_attn[i, j] = sum_c dO[i, c] * V[j, c]
        Tensor d_attn_h(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t c = 0; c < head_dim_; ++c) {
                    s += d_attn_concat(i, h * head_dim_ + c) * V(j, h * head_dim_ + c);
                }
                d_attn_h(i, j) = s;
            }
        }
        // softmax backward: dS[i, j] = scores[i, j] * (d_attn[i, j] - sum_j' scores[i, j'] * d_attn[i, j'])
        Tensor dS_h(n, n);
        for (size_t i = 0; i < n; ++i) {
            double row_dot = 0.0;
            for (size_t j = 0; j < n; ++j) row_dot += scores(i, j) * d_attn_h(i, j);
            for (size_t j = 0; j < n; ++j) {
                if (j > i) { dS_h(i, j) = 0.0; continue; }
                dS_h(i, j) = scores(i, j) * (d_attn_h(i, j) - row_dot);
            }
        }
        // dQ_h[i, c] = sum_j dS[i, j] * inv_temp * K[j, c]
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < head_dim_; ++c) {
                double s = 0.0;
                for (size_t j = 0; j < n; ++j) s += dS_h(i, j) * inv_temp * K(j, h * head_dim_ + c);
                dQ(i, h * head_dim_ + c) += s;
            }
        }
        // dK_h[j, c] = sum_i dS[i, j] * inv_temp * Q[i, c]
        for (size_t j = 0; j < n; ++j) {
            for (size_t c = 0; c < head_dim_; ++c) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i) s += dS_h(i, j) * inv_temp * Q(i, h * head_dim_ + c);
                dK_(j, h * head_dim_ + c) += s;
            }
        }
    }

    // Now backprop dQ, dK, dV through the Q/K/V projections (Dense::backward
    // internally accumulates into grad_weights/grad_bias — no manual add needed).
    Tensor dQ_via_Wq = attn_W_q.backward(dQ, 0.0);
    Tensor dK_via_Wk = attn_W_k.backward(dK_, 0.0);
    Tensor dV_via_Wv = attn_W_v.backward(dV_, 0.0);

    // d_ln1_out = dQ_via_Wq + dK_via_Wk + dV_via_Wv  (Dense.backward returns d_input)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_ln1_out(i, j) = dQ_via_Wq(i, j) + dK_via_Wk(i, j) + dV_via_Wv(i, j);

    Tensor d_ln1 = ln1_->backward(d_ln1_out, 0.0);

    // d_input: residual + ln1 path
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_input(i, j) = d_h1(i, j) + d_ln1(i, j);
    (void)X;
    return d_input;
}

// ============================================================================
// TokenformerModel
// ============================================================================

TokenformerModel::TokenformerModel(size_t d_input, size_t d_model, size_t d_output,
                                   size_t num_blocks, size_t num_heads, size_t num_param_tokens)
    : d_input_(d_input), d_model_(d_model), d_output_(d_output),
      input_proj(d_input, d_model), classifier(d_model, d_output)
{
    if (d_input_ == 0 || d_model_ == 0 || d_output_ == 0 || num_blocks == 0) {
        throw std::invalid_argument("TokenformerModel: d_input/d_model/d_output/num_blocks must all be > 0");
    }
    if (d_model_ % num_heads != 0) {
        throw std::invalid_argument("TokenformerModel: d_model must be divisible by num_heads");
    }
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(std::make_unique<TokenformerBlock>(d_model_, num_heads, num_param_tokens));
    }
    final_ln_ = std::make_unique<LayerNorm>(d_model_);
}

std::vector<Tensor*> TokenformerModel::parameters() {
    std::vector<Tensor*> ps;
    ps.push_back(&input_proj.weights);
    ps.push_back(&input_proj.bias);
    ps.push_back(&final_ln_->gamma);
    ps.push_back(&final_ln_->beta);
    ps.push_back(&classifier.weights);
    ps.push_back(&classifier.bias);
    for (auto& b : blocks_) {
        for (auto* p : b->parameters()) ps.push_back(p);
    }
    return ps;
}

std::vector<Tensor*> TokenformerModel::gradients() {
    std::vector<Tensor*> gs;
    gs.push_back(&input_proj.grad_weights);
    gs.push_back(&input_proj.grad_bias);
    gs.push_back(&final_ln_->grad_gamma_);
    gs.push_back(&final_ln_->grad_beta_);
    gs.push_back(&classifier.grad_weights);
    gs.push_back(&classifier.grad_bias);
    for (auto& b : blocks_) {
        for (auto* g : b->gradients()) gs.push_back(g);
    }
    return gs;
}

void TokenformerModel::zero_grad() {
    input_proj.zero_grad();
    classifier.zero_grad();
    final_ln_->zero_grad();
    for (auto& b : blocks_) b->zero_grad();
}

void TokenformerModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    classifier.update_weights(lr);
    final_ln_->update_weights(lr);
    for (auto& b : blocks_) b->update_weights(lr);
}

Tensor TokenformerModel::forward(const Tensor& input) {
    Tensor h = input_proj.forward(input);
    for (auto& b : blocks_) {
        h = b->forward(h);
    }
    h = final_ln_->forward(h);
    h = classifier.forward(h);
    return h;
}

Tensor TokenformerModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // classifier backward (Dense::backward internally accumulates grad_weights/grad_bias)
    Tensor d_pre_classifier = classifier.backward(grad_output, 0.0);
    // final ln
    Tensor d_blocks = final_ln_->backward(d_pre_classifier, 0.0);
    // blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_blocks = (*it)->backward(d_blocks, 0.0);
    }
    // input proj
    Tensor d_input = input_proj.backward(d_blocks, 0.0);
    return d_input;
}
