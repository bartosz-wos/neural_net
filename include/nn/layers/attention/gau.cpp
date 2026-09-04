// GAU — Gated Attention Unit — Hua et al. 2022
//   "Transformer Quality in Linear Time"
//   https://arxiv.org/abs/2206.03637
//
// See gau.h for the full mathematical formulation. This file implements:
//   * GAUAttention : single-head GAU with learnable position bias γ
//   * GAUBlock     : pre-LN -> GAU -> residual -> pre-LN -> FFN -> residual
//   * GAUModel     : stack of GAUBlocks + per-token classifier
//
// Weights are stored as raw Tensors (W shape: (out_features, in_features)).
// Forward computes Y = X @ W^T (no bias on Q/K/V/U; output projection has bias).
//
// Backward summary (the bits easy to get wrong):
//   grad_O[t,d] = upstream @ W_o^T                                     (Dense chain)
//   let dA_d     = grad_O[t,d] * U[t,d] / (B_t + eps)                   (per-channel scalar)
//   let dB       = -sum_d grad_O[t,d] * U[t,d] * A[t,d] / (B_t + eps)^2 (lumped denom grad)
//   for s ≤ t:
//     grad_γ[t,s] += sum_d dA_d * V[s,d] + dB
//     grad_V[s,d] += dA_d * γ[t,s] / (B_t + eps)
//   grad_U[t,d] += grad_O[t,d] * A[t,d] / (B_t + eps)
//   (Q and K get zero gradients in the simplest GAU — we still allocate them
//    for API symmetry and to allow future φ extensions.)

#include "gau.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace { static inline double gau_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}}
namespace { static inline double gau_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}}

// ============================================================================
// GAUAttention
// ============================================================================

GAUAttention::GAUAttention(size_t d_model, size_t max_seq_len, size_t d_out)
    : W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      W_u(d_model ? d_model : 1, d_model ? d_model : 1),
      W_o(d_out ? d_out : (d_model ? d_model : 1), d_model ? d_model : 1),
      b_o(1, d_out ? d_out : (d_model ? d_model : 1)),
      position_bias_(max_seq_len ? max_seq_len : 1, max_seq_len ? max_seq_len : 1),
      grad_W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_u(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_o(d_out ? d_out : (d_model ? d_model : 1), d_model ? d_model : 1),
      grad_b_o(1, d_out ? d_out : (d_model ? d_model : 1)),
      grad_position_bias_(max_seq_len ? max_seq_len : 1, max_seq_len ? max_seq_len : 1),
      d_model_(d_model), max_seq_len_(max_seq_len),
      d_out_(d_out ? d_out : d_model)
{
    if (d_model == 0)        throw std::invalid_argument("GAUAttention: d_model must be > 0");
    if (max_seq_len == 0)    throw std::invalid_argument("GAUAttention: max_seq_len must be > 0");
    // Initialize position bias small random — early training doesn't favor any position.
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.01);
    for (size_t i = 0; i < max_seq_len_; ++i)
        for (size_t j = 0; j < max_seq_len_; ++j)
            position_bias_(i, j) = dis(gen);
    grad_position_bias_.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_u.fill(0.0);
    grad_W_o.fill(0.0); grad_b_o.fill(0.0);
    b_o.fill(0.0);
    // Initialize W_q, W_k, W_v, W_u, W_o with Xavier uniform (small).
    auto xavier_init = [&](Tensor& W) {
        double s = std::sqrt(6.0 / (double)(W.rows + W.cols));
        std::uniform_real_distribution<> u(-s, s);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) = u(gen);
    };
    xavier_init(W_q); xavier_init(W_k); xavier_init(W_v); xavier_init(W_u);
    xavier_init(W_o);
}

Tensor GAUAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("GAUAttention: input.cols must equal d_model");
    }
    if (n > max_seq_len_) {
        throw std::invalid_argument("GAUAttention: input.rows exceeds max_seq_len");
    }
    last_input_ = input.clone();

    // Q = X @ W_q^T  (n, d_model) <- (n, d_model) @ (d_model, d_model)
    last_Q_ = Tensor(n, d_model_);
    last_K_ = Tensor(n, d_model_);
    last_V_ = Tensor(n, d_model_);
    last_U_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double q_acc = 0.0, k_acc = 0.0, v_acc = 0.0, u_acc = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                double xk = input(t, k);
                q_acc += xk * W_q(i, k);
                k_acc += xk * W_k(i, k);
                v_acc += xk * W_v(i, k);
                u_acc += xk * W_u(i, k);
            }
            last_Q_(t, i) = q_acc;
            last_K_(t, i) = k_acc;
            last_V_(t, i) = v_acc;
            last_U_(t, i) = u_acc;
        }
    }

    // Per-position: A_t[d] = sum_{s≤t} γ_{t,s} * V_s[d], B_t = sum_{s≤t} γ_{t,s}
    cache_A_ = Tensor(n, d_model_);
    cache_B_ = Tensor(n, 1);
    cache_O_ = Tensor(n, d_model_);
    const double eps = 1e-6;

    for (size_t t = 0; t < n; ++t) {
        double B = 0.0;
        for (size_t s = 0; s <= t; ++s) B += position_bias_(t, s);
        cache_B_(t, 0) = B;
        const double inv_den = 1.0 / (B + eps);
        for (size_t d = 0; d < d_model_; ++d) {
            double A = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                A += position_bias_(t, s) * last_V_(s, d);
            }
            cache_A_(t, d) = A;
            cache_O_(t, d) = last_U_(t, d) * A * inv_den;
        }
    }
    last_output_pre_wo_ = cache_O_.clone();

    // Y = O @ W_o^T + b_o  (n, d_out) <- (n, d_model) @ (d_out, d_model)^T
    Tensor Y(n, d_out_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_out_; ++j) {
            double acc = b_o(0, j);
            for (size_t i = 0; i < d_model_; ++i) {
                acc += cache_O_(t, i) * W_o(j, i);
            }
            Y(t, j) = acc;
        }
    }
    return Y;
}

Tensor GAUAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = last_input_.rows;

    // (1) grad_O via W_o^T: grad_O[t, i] = sum_j grad_output[t, j] * W_o[j, i]
    Tensor grad_O(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_out_; ++j) acc += grad_output(t, j) * W_o(j, i);
            grad_O(t, i) = acc;
        }
    }
    // grad_W_o[j, i] = sum_t grad_output[t, j] * O[t, i]
    // grad_b_o[j]    = sum_t grad_output[t, j]
    for (size_t j = 0; j < d_out_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * cache_O_(t, i);
            grad_W_o(j, i) = acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        grad_b_o(0, j) = b_acc;
    }

    // (2) Backward through the GAU core.
    //
    //   O_t[d]  = U_t[d] * A_t[d] / (B_t + eps)
    //   A_t[d]  = sum_{s≤t} γ_{t,s} * V_s[d]
    //   B_t     = sum_{s≤t} γ_{t,s}
    //
    // Chain rule per (t, d):
    //   grad_U[t,d] += grad_O[t,d] * A_t[d] / (B_t + eps)
    //   let dA_d = grad_O[t,d] * U_t[d] / (B_t + eps)
    //   let dB   = -sum_d grad_O[t,d] * U_t[d] * A_t[d] / (B_t + eps)^2
    //
    //   For each s ≤ t:
    //     grad_γ[t,s] += sum_d dA_d * V_s[d] + dB
    //     grad_V[s,d] += dA_d * γ[t,s] / (B_t + eps)
    Tensor grad_U(n, d_model_);
    Tensor grad_V(n, d_model_);
    grad_U.fill(0.0);
    grad_V.fill(0.0);

    const double eps = 1e-6;
    grad_W_v.fill(0.0);
    grad_W_u.fill(0.0);
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_position_bias_.fill(0.0);

    for (size_t t = 0; t < n; ++t) {
        const double B = cache_B_(t, 0);
        const double inv_den = 1.0 / (B + eps);
        const double inv_den2 = inv_den * inv_den;
        double dB = 0.0;
        for (size_t d = 0; d < d_model_; ++d) {
            const double Atd = cache_A_(t, d);
            const double Ut_d = last_U_(t, d);
            grad_U(t, d) += grad_O(t, d) * Atd * inv_den;
            dB -= grad_O(t, d) * Ut_d * Atd * inv_den2;
        }
        for (size_t s = 0; s <= t; ++s) {
            const double gamma_ts = position_bias_(t, s);
            double sum_dA_V = 0.0;
            for (size_t d = 0; d < d_model_; ++d) {
                const double dA_d = grad_O(t, d) * last_U_(t, d) * inv_den;
                sum_dA_V += dA_d * last_V_(s, d);
                grad_V(s, d) += dA_d * gamma_ts;
            }
            grad_position_bias_(t, s) += sum_dA_V + dB;
        }
    }

    // (3) Chain through W_v and W_u (Q and K receive zero gradients in simplest GAU).
    // For Dense y = X @ W^T + b (no bias on Q/K/V/U projections):
    //   grad_W[i, k] += sum_t grad_y[t, i] * X[t, k]
    //   grad_X[t, k] += sum_i grad_y[t, i] * W[i, k]
    Tensor grad_input(n, d_model_);
    grad_input.fill(0.0);

    // W_v
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_V(t, i) * last_input_(t, k);
            grad_W_v(i, k) = acc;
        }
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_V(t, i) * W_v(i, k);
    }
    // W_u
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_U(t, i) * last_input_(t, k);
            grad_W_u(i, k) = acc;
        }
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_U(t, i) * W_u(i, k);
    }
    // Q/K have zero gradients in simplest GAU (no path from A_t/B_t to Q/K).
    // For API symmetry, grad_W_q, grad_W_k are kept zero.

    return grad_input;
}

void GAUAttention::update_weights(double learning_rate) {
    auto apply = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w(i, j) -= learning_rate * gw(i, j);
    };
    apply(W_q, grad_W_q); apply(W_k, grad_W_k);
    apply(W_v, grad_W_v); apply(W_u, grad_W_u);
    apply(W_o, grad_W_o);
    for (size_t j = 0; j < d_out_; ++j) b_o(0, j) -= learning_rate * grad_b_o(0, j);
    for (size_t i = 0; i < max_seq_len_; ++i)
        for (size_t j = 0; j < max_seq_len_; ++j)
            position_bias_(i, j) -= learning_rate * grad_position_bias_(i, j);
}

void GAUAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_u.fill(0.0);
    grad_W_o.fill(0.0); grad_b_o.fill(0.0);
    grad_position_bias_.fill(0.0);
}

std::vector<Tensor*> GAUAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_u, &W_o, &b_o, &position_bias_};
}

std::vector<Tensor*> GAUAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_u, &grad_W_o, &grad_b_o, &grad_position_bias_};
}

// ============================================================================
// GAUBlock
// ============================================================================

GAUBlock::GAUBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim, size_t d_out)
    : attn(d_model, max_seq_len, d_out),
    ln1(d_model), ln2(d_model),
    W_ffn1_(d_model, ffn_dim ? ffn_dim : 4 * d_model),
    W_ffn2_(ffn_dim ? ffn_dim : 4 * d_model, d_model),
    b_ffn1_(1, ffn_dim ? ffn_dim : 4 * d_model),
    b_ffn2_(1, d_model),
    grad_W_ffn1_(d_model, ffn_dim ? ffn_dim : 4 * d_model),
    grad_W_ffn2_(ffn_dim ? ffn_dim : 4 * d_model, d_model),
    grad_b_ffn1_(1, ffn_dim ? ffn_dim : 4 * d_model),
    grad_b_ffn2_(1, d_model),
    d_model_(d_model), ffn_dim_(ffn_dim ? ffn_dim : 4 * d_model)
{
    if (d_model == 0) throw std::invalid_argument("GAUBlock: d_model must be > 0");
    if (max_seq_len == 0) throw std::invalid_argument("GAUBlock: max_seq_len must be > 0");
    grad_W_ffn1_.fill(0.0); grad_W_ffn2_.fill(0.0);
    grad_b_ffn1_.fill(0.0); grad_b_ffn2_.fill(0.0);
    b_ffn1_.fill(0.0);    b_ffn2_.fill(0.0);
    // Xavier init for the FFN weights (use a fresh RNG).
    std::mt19937 gen(7);
    auto xavier = [&](Tensor& W) {
        double s = std::sqrt(6.0 / (double)(W.rows + W.cols));
        std::uniform_real_distribution<> u(-s, s);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) = u(gen);
    };
    xavier(W_ffn1_); xavier(W_ffn2_);
}

Tensor GAUBlock::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_x_ = input.clone();
    // ln1(x)
    last_z1_ = ln1.forward(input);
    // GAU attn
    Tensor attn_out = attn.forward(last_z1_);
    last_attn_out_ = attn_out.clone();
    // residual: z1 + attn_out  (no dropout, no scaling)
    last_res1_ = Tensor(n, d_model_);
    for (size_t i = 0; i < n * d_model_; ++i)
        last_res1_.data[i] = last_z1_.data[i] + attn_out.data[i];
    // ln2(res1)
    last_z2_ = ln2.forward(last_res1_);
    // FFN: h_pre = W_ffn1 · z2 + b_ffn1, h_act = GELU(h_pre), ffn_out = W_ffn2 · h_act + b_ffn2
    last_h_pre_ = Tensor(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double acc = b_ffn1_(0, j);
            for (size_t i = 0; i < d_model_; ++i) acc += last_z2_(t, i) * W_ffn1_(i, j);
            last_h_pre_(t, j) = acc;
        }
    }
    last_h_act_ = Tensor(n, ffn_dim_);
    for (size_t i = 0; i < n * ffn_dim_; ++i) last_h_act_.data[i] = gau_gelu(last_h_pre_.data[i]);
    last_ffn_out_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = b_ffn2_(0, j);
            for (size_t i = 0; i < ffn_dim_; ++i) acc += last_h_act_(t, i) * W_ffn2_(i, j);
            last_ffn_out_(t, j) = acc;
        }
    }
    // residual: res1 + ffn_out
    Tensor out(n, d_model_);
    for (size_t i = 0; i < n * d_model_; ++i) out.data[i] = last_res1_.data[i] + last_ffn_out_.data[i];
    return out;
}

Tensor GAUBlock::backward(const Tensor& grad_output, double /* lr */) {
    const size_t n = grad_output.rows;
    // d_res1 += grad_output  (residual path into res1)
    // d_ffn_out += grad_output (residual path into ffn_out)
    Tensor d_ffn_out = grad_output.clone();
    Tensor d_res1 = grad_output.clone();

    // (1) Backward through W_ffn2: d_h_act, d_W_ffn2, d_b_ffn2
    // ffn_out[t, j] = sum_i h_act[t, i] * W_ffn2[i, j] + b_ffn2[j]
    Tensor d_h_act(n, ffn_dim_);
    d_h_act.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < ffn_dim_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += d_ffn_out(t, j) * W_ffn2_(i, j);
            d_h_act(t, i) = acc;
        }
    }
    for (size_t i = 0; i < ffn_dim_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += d_ffn_out(t, j) * last_h_act_(t, i);
            grad_W_ffn2_(i, j) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_ffn_out(t, j);
        grad_b_ffn2_(0, j) = b_acc;
    }

    // (2) GELU derivative: d_h_pre = d_h_act ⊙ GELU'(h_pre)
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < ffn_dim_; ++j) {
            d_h_pre(t, j) = d_h_act(t, j) * gau_gelu_deriv(last_h_pre_(t, j));
        }
    }

    // (3) Backward through W_ffn1: d_z2, d_W_ffn1, d_b_ffn1
    Tensor d_z2(n, d_model_);
    d_z2.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < ffn_dim_; ++j) acc += d_h_pre(t, j) * W_ffn1_(i, j);
            d_z2(t, i) = acc;
        }
    }
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double acc = 0.0;
            for (size_t t2 = 0; t2 < n; ++t2) acc += d_h_pre(t2, j) * last_z2_(t2, i);
            grad_W_ffn1_(i, j) = acc;
        }
    }
    for (size_t j = 0; j < ffn_dim_; ++j) {
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_h_pre(t, j);
        grad_b_ffn1_(0, j) = b_acc;
    }

    // (4) ln2 backward: d_res1 (raw) += d_z2 chain through ln2
    Tensor d_res1_from_ln = ln2.backward(d_z2, 0.0);
    for (size_t i = 0; i < n * d_model_; ++i) d_res1.data[i] += d_res1_from_ln.data[i];

    // (5) d_attn_out = d_res1 (from residual: res1 = z1 + attn_out).
        Tensor d_attn_out = d_res1.clone();

        // (6) d_z1 = d_res1 (residual) + attn.backward(d_attn_out) (attn path).
        // d_z1 is the gradient w.r.t. ln1's output = z1, from TWO independent paths:
        //   - res1 = z1 + attn_out: d_z1 += d_res1 (residual direct)
        //   - z1 → attn → res1 → out: d_z1 += attn.backward(d_res1) (chain through GAU)
        // Then d_x = ln1.backward(d_z1_total) — called ONCE.
        Tensor d_z1_from_attn = attn.backward(d_attn_out, 0.0);
        Tensor d_z1_total(n, d_model_);
        for (size_t i = 0; i < n * d_model_; ++i) {
            d_z1_total.data[i] = d_res1.data[i] + d_z1_from_attn.data[i];
        }
        Tensor d_x = ln1.backward(d_z1_total, 0.0);
        return d_x;
}

void GAUBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    auto apply = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w(i, j) -= learning_rate * gw(i, j);
    };
    apply(W_ffn1_, grad_W_ffn1_); apply(W_ffn2_, grad_W_ffn2_);
    for (size_t j = 0; j < ffn_dim_; ++j) b_ffn1_(0, j) -= learning_rate * grad_b_ffn1_(0, j);
    for (size_t j = 0; j < d_model_; ++j) b_ffn2_(0, j) -= learning_rate * grad_b_ffn2_(0, j);
}

void GAUBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W_ffn1_.fill(0.0); grad_W_ffn2_.fill(0.0);
    grad_b_ffn1_.fill(0.0); grad_b_ffn2_.fill(0.0);
}

std::vector<Tensor*> GAUBlock::parameters() {
    return {&attn.W_q, &attn.W_k, &attn.W_v, &attn.W_u, &attn.W_o, &attn.b_o, &attn.position_bias_,
            &W_ffn1_, &W_ffn2_, &b_ffn1_, &b_ffn2_,
            &ln1.gamma, &ln1.beta, &ln2.gamma, &ln2.beta};
}

std::vector<Tensor*> GAUBlock::gradients() {
    return {&attn.grad_W_q, &attn.grad_W_k, &attn.grad_W_v, &attn.grad_W_u,
            &attn.grad_W_o, &attn.grad_b_o, &attn.grad_position_bias_,
            &grad_W_ffn1_, &grad_W_ffn2_, &grad_b_ffn1_, &grad_b_ffn2_,
            &ln1.grad_gamma_, &ln1.grad_beta_, &ln2.grad_gamma_, &ln2.grad_beta_};
}

// ============================================================================
// GAUModel
// ============================================================================

GAUModel::GAUModel(size_t d_model, size_t max_seq_len, size_t out_features,
                   size_t num_blocks, size_t ffn_dim)
    : classifier_W_(d_model, out_features),
      classifier_b_(1, out_features),
      grad_classifier_W_(d_model, out_features),
      grad_classifier_b_(1, out_features),
      d_model_(d_model), max_seq_len_(max_seq_len), out_features_(out_features),
      num_blocks_(num_blocks), ffn_dim_(ffn_dim ? ffn_dim : 4 * d_model)
{
    if (d_model == 0 || max_seq_len == 0 || out_features == 0 || num_blocks == 0)
        throw std::invalid_argument("GAUModel: all dims must be > 0");
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks.push_back(std::make_unique<GAUBlock>(d_model, max_seq_len, ffn_dim_));
    }
    std::mt19937 gen(99);
    double s = std::sqrt(6.0 / (double)(d_model + out_features));
    std::uniform_real_distribution<> u(-s, s);
    for (size_t i = 0; i < d_model; ++i)
        for (size_t j = 0; j < out_features; ++j) classifier_W_(i, j) = u(gen);
    classifier_b_.fill(0.0);
    grad_classifier_W_.fill(0.0); grad_classifier_b_.fill(0.0);
}

Tensor GAUModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& blk : blocks) x = blk->forward(x);
    last_block_output_ = x.clone();
    // Per-token classifier: Y[t, j] = sum_i block[t, i] * W[i, j] + b[j]
    const size_t n = x.rows;
    Tensor Y(n, out_features_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < out_features_; ++j) {
            double acc = classifier_b_(0, j);
            for (size_t i = 0; i < d_model_; ++i) acc += x(t, i) * classifier_W_(i, j);
            Y(t, j) = acc;
        }
    }
    return Y;
}

Tensor GAUModel::backward(const Tensor& grad_output, double /* lr */) {
    const size_t n = grad_output.rows;
    // grad into block output
    Tensor d_block_out(n, d_model_);
    d_block_out.fill(0.0);
    for (size_t j = 0; j < out_features_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_block_output_(t, i);
            grad_classifier_W_(i, j) = acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        grad_classifier_b_(0, j) = b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t i = 0; i < d_model_; ++i)
                d_block_out(t, i) += grad_output(t, j) * classifier_W_(i, j);
    }
    // Now propagate backward through the stack of blocks.
    Tensor d_out = d_block_out;
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        d_out = (*it)->backward(d_out, 0.0);
    }
    return d_out;
}

void GAUModel::update_weights(double learning_rate) {
    for (auto& blk : blocks) blk->update_weights(learning_rate);
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < out_features_; ++j)
            classifier_W_(i, j) -= learning_rate * grad_classifier_W_(i, j);
    for (size_t j = 0; j < out_features_; ++j)
        classifier_b_(0, j) -= learning_rate * grad_classifier_b_(0, j);
}

void GAUModel::zero_grad() {
    for (auto& blk : blocks) blk->zero_grad();
    grad_classifier_W_.fill(0.0);
    grad_classifier_b_.fill(0.0);
}

std::vector<Tensor*> GAUModel::parameters() {
    std::vector<Tensor*> out;
    for (auto& blk : blocks) {
        auto p = blk->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    out.push_back(&classifier_W_);
    out.push_back(&classifier_b_);
    return out;
}

std::vector<Tensor*> GAUModel::gradients() {
    std::vector<Tensor*> out;
    for (auto& blk : blocks) {
        auto g = blk->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    out.push_back(&grad_classifier_W_);
    out.push_back(&grad_classifier_b_);
    return out;
}