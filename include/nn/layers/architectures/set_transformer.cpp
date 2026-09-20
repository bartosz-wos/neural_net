// Set Transformer — Lee et al. 2019 (ICML)
//   https://arxiv.org/abs/1810.00825
//
// Conventions (matches repo-wide):
//   - Tensor layout: (n_tokens, d_model). Rows = tokens, cols = features.
//   - Single-batch (B=1) for v1.
//   - Pre-LN (LayerNorm before attention and FFN, with residuals).
//   - All W projections stored as (out_dim, in_dim) matrices — same convention
//     as Dense().
//
// Implementation: MAB is the fundamental cross-attention block. SAB, ISAB,
// PMA are thin wrappers composing MAB. SetTransformer wires the full model.

#include "set_transformer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace {

// ============================================================================
// Per-row LayerNorm (per-token normalization). Rows of x are tokens; each row
// is normalized independently. gamma/beta are (1, d_model) — one scale and
// shift per feature, shared across all tokens.
//
// Returns `out` of shape (n_rows, d_model). Fills `cache_x`, `cache_mean`,
// `cache_var` (shape (1, n_rows) each).
// ============================================================================
inline void layernorm_per_row_forward(const Tensor& x,
                                       const Tensor& gamma, const Tensor& beta,
                                       double eps,
                                       Tensor& out,
                                       Tensor& cache_x, Tensor& cache_mean, Tensor& cache_var) {
    const size_t n_rows = x.rows;
    const size_t d_model = x.cols;
    out = Tensor(n_rows, d_model);
    cache_x = x;
    cache_mean = Tensor(1, n_rows);
    cache_var  = Tensor(1, n_rows);
    for (size_t t = 0; t < n_rows; ++t) {
        double mean = 0.0;
        for (size_t f = 0; f < d_model; ++f) mean += x(t, f);
        mean /= d_model;
        double var = 0.0;
        for (size_t f = 0; f < d_model; ++f) {
            double d = x(t, f) - mean;
            var += d * d;
        }
        var /= d_model;
        cache_mean(0, t) = mean;
        cache_var(0, t)  = var;
        const double sv = std::sqrt(var + eps);
        for (size_t f = 0; f < d_model; ++f) {
            const double n = (x(t, f) - mean) / sv;
            out(t, f) = gamma(0, f) * n + beta(0, f);
        }
    }
}

// ============================================================================
// Backward of layernorm_per_row_forward. d_y is (n_rows, d_model). Returns
// d_x (n_rows, d_model). Accumulates gamma and beta gradients.
// ============================================================================
inline void layernorm_per_row_backward(const Tensor& d_y,
                                        const Tensor& gamma,
                                        const Tensor& cache_x,
                                        const Tensor& cache_mean,
                                        const Tensor& cache_var,
                                        double eps,
                                        Tensor& grad_gamma, Tensor& grad_beta,
                                        Tensor& d_x) {
    const size_t n_rows = d_y.rows;
    const size_t d_model = d_y.cols;
    d_x = Tensor(n_rows, d_model);
    // Per-feature gradient accumulation for gamma/beta
    for (size_t f = 0; f < d_model; ++f) {
        double gg = 0.0, gb = 0.0;
        for (size_t t = 0; t < n_rows; ++t) {
            const double norm = (cache_x(t, f) - cache_mean(0, t)) /
                                std::sqrt(cache_var(0, t) + eps);
            gg += d_y(t, f) * norm;
            gb += d_y(t, f);
        }
        grad_gamma(0, f) += gg;
        grad_beta(0, f)  += gb;
    }
    // Per-token d_x (the standard LN backward)
    for (size_t t = 0; t < n_rows; ++t) {
        const double sv = std::sqrt(cache_var(0, t) + eps);
        const double m_dy = [&]() {
            double s = 0.0;
            for (size_t f = 0; f < d_model; ++f) s += d_y(t, f);
            return s / d_model;
        }();
        const double m_dyn = [&]() {
            double s = 0.0;
            for (size_t f = 0; f < d_model; ++f) {
                const double norm = (cache_x(t, f) - cache_mean(0, t)) / sv;
                s += d_y(t, f) * norm;
            }
            return s / d_model;
        }();
        for (size_t f = 0; f < d_model; ++f) {
            const double norm = (cache_x(t, f) - cache_mean(0, t)) / sv;
            d_x(t, f) = (gamma(0, f) / sv) * (d_y(t, f) - m_dy - norm * m_dyn);
        }
    }
}

}  // namespace

// ============================================================================
// MAB
// ============================================================================

MAB::MAB(size_t d_model, size_t num_heads, size_t ffn_hidden)
    : d_model_(d_model == 0 ? 1 : d_model),  // placeholder; validated below
      num_heads_(num_heads == 0 ? 1 : num_heads),
      d_k_(d_model_ / num_heads_),
      ffn_hidden_(ffn_hidden == 0 ? (4 * (d_model == 0 ? 1 : d_model)) : ffn_hidden),
      last_was_self_(false),
      W_q_(d_model_, d_model_), W_k_(d_model_, d_model_),
      W_v_(d_model_, d_model_), W_O_(d_model_, d_model_),
      grad_W_q_(d_model_, d_model_), grad_W_k_(d_model_, d_model_),
      grad_W_v_(d_model_, d_model_), grad_W_O_(d_model_, d_model_),
      W_ffn_(ffn_hidden_, d_model_), b_ffn_(1, ffn_hidden_),
      W_ffn_out_(d_model_, ffn_hidden_), b_ffn_out_(1, d_model_),
      grad_W_ffn_(ffn_hidden_, d_model_), grad_b_ffn_(1, ffn_hidden_),
      grad_W_ffn_out_(d_model_, ffn_hidden_), grad_b_ffn_out_(1, d_model_),
      ln1_gamma_(1, d_model_), ln1_beta_(1, d_model_),
      ln2_gamma_(1, d_model_), ln2_beta_(1, d_model_),
      grad_ln1_gamma_(1, d_model_), grad_ln1_beta_(1, d_model_),
      grad_ln2_gamma_(1, d_model_), grad_ln2_beta_(1, d_model_)
{
    if (d_model == 0)
        throw std::invalid_argument("MAB: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MAB: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("MAB: d_model must be divisible by num_heads");
    if (ffn_hidden == 0)
        throw std::invalid_argument("MAB: ffn_hidden must be > 0");

    // Init all weights with random values. Use a moderate scale (0.1). Larger
    // scales (0.3) make softmax sharp enough to attend, but blow up after
    // multiple MABs in a stack. 0.1 strikes a balance — softmax is non-
    // uniform AND the chain stays numerically stable over 4+ MABs.
    W_q_    = Tensor::random(d_model, d_model, 0.1);
    W_k_    = Tensor::random(d_model, d_model, 0.1);
    W_v_    = Tensor::random(d_model, d_model, 0.1);
    W_O_    = Tensor::random(d_model, d_model, 0.1);
    W_ffn_  = Tensor::random(ffn_hidden_, d_model_, 0.1);
    b_ffn_  = Tensor::zeros(1, ffn_hidden_);
    W_ffn_out_ = Tensor::random(d_model_, ffn_hidden_, 0.1);
    b_ffn_out_ = Tensor::zeros(1, d_model_);

    // LayerNorm: gamma = 1, beta = 0 (standard pre-LN init)
    ln1_gamma_.fill(1.0);
    ln1_beta_.fill(0.0);
    ln2_gamma_.fill(1.0);
    ln2_beta_.fill(0.0);

    zero_grad();
}

std::vector<Tensor*> MAB::parameters() {
    return {&W_q_, &W_k_, &W_v_, &W_O_,
            &W_ffn_, &b_ffn_, &W_ffn_out_, &b_ffn_out_,
            &ln1_gamma_, &ln1_beta_, &ln2_gamma_, &ln2_beta_};
}

std::vector<Tensor*> MAB::gradients() {
    return {&grad_W_q_, &grad_W_k_, &grad_W_v_, &grad_W_O_,
            &grad_W_ffn_, &grad_b_ffn_, &grad_W_ffn_out_, &grad_b_ffn_out_,
            &grad_ln1_gamma_, &grad_ln1_beta_,
            &grad_ln2_gamma_, &grad_ln2_beta_};
}

void MAB::zero_grad() {
    grad_W_q_.fill(0.0); grad_W_k_.fill(0.0);
    grad_W_v_.fill(0.0); grad_W_O_.fill(0.0);
    grad_W_ffn_.fill(0.0); grad_b_ffn_.fill(0.0);
    grad_W_ffn_out_.fill(0.0); grad_b_ffn_out_.fill(0.0);
    grad_ln1_gamma_.fill(0.0); grad_ln1_beta_.fill(0.0);
    grad_ln2_gamma_.fill(0.0); grad_ln2_beta_.fill(0.0);
    last_was_self_ = false;
}

Tensor MAB::forward(const Tensor& q_input, const Tensor& kv_input) {
    if (q_input.cols != d_model_ || kv_input.cols != d_model_)
        throw std::invalid_argument("MAB: input cols must be d_model");
    if (q_input.rows == 0 || kv_input.rows == 0)
        throw std::invalid_argument("MAB: n_tokens must be > 0");

    const size_t n_q  = q_input.rows;
    const size_t n_kv = kv_input.rows;

    last_q_in_raw_  = q_input;
    last_kv_in_raw_ = kv_input;

    // ---------- Pre-LN1: normalize Q and K/V inputs (shared gamma/beta) ----
    layernorm_per_row_forward(q_input, ln1_gamma_, ln1_beta_, LN_EPS_,
                              last_q_normed_,
                              last_q_pre_norm_, last_q_mean_, last_q_var_);
    layernorm_per_row_forward(kv_input, ln1_gamma_, ln1_beta_, LN_EPS_,
                              last_kv_normed_,
                              last_kv_pre_norm_, last_kv_mean_, last_kv_var_);

    // ---------- Q, K, V projections ----------
    // last_q_normed_  (n_q, d_model) @ W_q_  (d_model, d_model)  → (n_q, d_model)
    // last_kv_normed_ (n_kv, d_model) @ W_k_ (d_model, d_model) → (n_kv, d_model)
    // last_kv_normed_ (n_kv, d_model) @ W_v_ (d_model, d_model) → (n_kv, d_model)
    Tensor Q(n_q, d_model_), K(n_kv, d_model_), V(n_kv, d_model_);
    for (size_t i = 0; i < n_q; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) s += last_q_normed_(i, k) * W_q_(k, j);
            Q(i, j) = s;
        }
    }
    for (size_t i = 0; i < n_kv; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double sk = 0.0, sv = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                sk += last_kv_normed_(i, k) * W_k_(k, j);
                sv += last_kv_normed_(i, k) * W_v_(k, j);
            }
            K(i, j) = sk;
            V(i, j) = sv;
        }
    }
    last_Q_ = Q; last_K_ = K; last_V_ = V;

    // ---------- Multi-head attention ----------
    const double scale = 1.0 / std::sqrt((double)d_k_ + 1e-9);
    last_A_ = Tensor(num_heads_ * n_q, n_kv);
    Tensor attn_concat(n_q, d_model_);
    attn_concat.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // scores[i, j] = sum_dk Q[i, h*dk + dk] * K[j, h*dk + dk] * scale
        Tensor scores(n_q, n_kv);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < d_k_; ++dk) {
                    s += Q(i, h * d_k_ + dk) * K(j, h * d_k_ + dk);
                }
                scores(i, j) = s * scale;
            }
        // Per-row softmax
        for (size_t i = 0; i < n_q; ++i) {
            double mx = scores(i, 0);
            for (size_t j = 1; j < n_kv; ++j) mx = std::max(mx, scores(i, j));
            double denom = 0.0;
            for (size_t j = 0; j < n_kv; ++j) {
                double e = std::exp(scores(i, j) - mx);
                scores(i, j) = e;
                denom += e;
            }
            const double inv = 1.0 / denom;
            for (size_t j = 0; j < n_kv; ++j) {
                scores(i, j) *= inv;
                last_A_(h * n_q + i, j) = scores(i, j);
            }
        }
        // head_out[i, dk] = sum_j scores[i, j] * V[j, h*dk + dk]
        for (size_t i = 0; i < n_q; ++i)
            for (size_t dk = 0; dk < d_k_; ++dk) {
                double s = 0.0;
                for (size_t j = 0; j < n_kv; ++j) {
                    s += scores(i, j) * V(j, h * d_k_ + dk);
                }
                attn_concat(i, h * d_k_ + dk) = s;
            }
    }
    last_attn_concat_ = attn_concat;

    // ---------- attn_out = attn_concat @ W_O ----------
    Tensor attn_out(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) s += attn_concat(i, k) * W_O_(k, j);
            attn_out(i, j) = s;
        }
    last_attn_out_ = attn_out;

    // ---------- Residual 1: res1 = attn_out + q_in ----------
    Tensor res1(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) res1(i, j) = attn_out(i, j) + q_input(i, j);
    last_res1_ = res1;

    // ---------- Pre-LN2 ----------
    layernorm_per_row_forward(res1, ln2_gamma_, ln2_beta_, LN_EPS_,
                              last_ln2_out_,
                              last_res1_pre_norm_, last_res1_mean_, last_res1_var_);

    // ---------- FFN: pre = ln2_out @ W_ffn^T + b_ffn ----------
    // last_ln2_out_ (n_q, d_model) @ W_ffn_ (ffn_hidden, d_model)
    //   → (n_q, ffn_hidden) using W_ffn_[j, k] is the k-th input feature of
    //   the j-th output feature: out[i, j] = sum_k ln2_out[i, k] * W_ffn[j, k].
    Tensor pre(n_q, ffn_hidden_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double s = b_ffn_(0, j);
            for (size_t k = 0; k < d_model_; ++k) s += last_ln2_out_(i, k) * W_ffn_(j, k);
            pre(i, j) = s;
        }
    last_ffn_pre_ = pre;

    // GELU activation: gelu(x) = 0.5 x (1 + erf(x / sqrt(2)))
    Tensor act(n_q, ffn_hidden_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            const double x = pre(i, j);
            act(i, j) = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        }
    last_ffn_act_ = act;

    // ---------- ffn_out = act @ W_ffn_out^T + b_ffn_out ----------
    // act (n_q, ffn_hidden) @ W_ffn_out_ (d_model, ffn_hidden)
    //   → (n_q, d_model): out[i, d] = sum_j act[i, j] * W_ffn_out[d, j].
    Tensor ffn_out(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t d = 0; d < d_model_; ++d) {
            double s = b_ffn_out_(0, d);
            for (size_t j = 0; j < ffn_hidden_; ++j) s += act(i, j) * W_ffn_out_(d, j);
            ffn_out(i, d) = s;
        }
    last_ffn_out_ = ffn_out;

    // ---------- Residual 2: res2 = ffn_out + res1 ----------
    Tensor res2(n_q, d_model_);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) res2(i, j) = ffn_out(i, j) + res1(i, j);
    last_res2_ = res2;
    return res2;
}

Tensor MAB::forward(const Tensor& input) {
    last_was_self_ = true;
    return forward(input, input);
}

MAB::GradPair MAB::backward_full(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.rows != last_res2_.rows || grad_output.cols != d_model_)
        throw std::invalid_argument("MAB::backward_full: grad_output shape mismatch");

    const size_t n_q  = grad_output.rows;
    const size_t n_kv = last_K_.rows;
    const double scale = 1.0 / std::sqrt((double)d_k_ + 1e-9);

    // ---------- Backward through residual 2: res2 = ffn_out + res1 ----------
    // d_ffn_out = grad_output
    // d_res1 += grad_output (the residual contribution)
    Tensor d_ffn_out = grad_output;
    Tensor d_res1 = grad_output.clone();

    // ---------- FFN out: ffn_out[i, d] = sum_j act[i, j] * W_ffn_out[d, j] + b_ffn_out[d] ----------
    // d_act[i, j] = sum_d d_ffn_out[i, d] * W_ffn_out[d, j]
    // d_W_ffn_out[d, j] += sum_i d_ffn_out[i, d] * act[i, j]
    // d_b_ffn_out[d] += sum_i d_ffn_out[i, d]
    Tensor d_act(n_q, ffn_hidden_);
    d_act.fill(0.0);
    for (size_t d = 0; d < d_model_; ++d) {
        double gb = 0.0;
        for (size_t i = 0; i < n_q; ++i) gb += d_ffn_out(i, d);
        grad_b_ffn_out_(0, d) += gb;
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double gw = 0.0;
            for (size_t i = 0; i < n_q; ++i) {
                gw += d_ffn_out(i, d) * last_ffn_act_(i, j);
                d_act(i, j) += d_ffn_out(i, d) * W_ffn_out_(d, j);
            }
            grad_W_ffn_out_(d, j) += gw;
        }
    }

    // ---------- GELU backward ----------
    // GELU'(x) = phi(x) + x * phi'(x)
    //          = 0.5(1 + erf(x/√2)) + x · 1/√(2π) · exp(-x²/2)
    Tensor d_pre(n_q, ffn_hidden_);
    {
        const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < ffn_hidden_; ++j) {
                const double x = last_ffn_pre_(i, j);
                const double phi = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
                const double gp = phi + x * inv_sqrt2pi * std::exp(-0.5 * x * x);
                d_pre(i, j) = d_act(i, j) * gp;
            }
    }

    // ---------- pre = ln2_out @ W_ffn^T + b_ffn ----------
    // pre[i, j] = sum_k ln2_out[i, k] * W_ffn[j, k] + b_ffn[j]
    // d_ln2_out[i, k] += sum_j d_pre[i, j] * W_ffn[j, k]
    // d_W_ffn[j, k] += sum_i d_pre[i, j] * ln2_out[i, k]
    // d_b_ffn[j] += sum_i d_pre[i, j]
    Tensor d_ln2_out(n_q, d_model_);
    d_ln2_out.fill(0.0);
    for (size_t j = 0; j < ffn_hidden_; ++j) {
        double gb = 0.0;
        for (size_t i = 0; i < n_q; ++i) gb += d_pre(i, j);
        grad_b_ffn_(0, j) += gb;
        for (size_t k = 0; k < d_model_; ++k) {
            double gw = 0.0;
            for (size_t i = 0; i < n_q; ++i) {
                d_ln2_out(i, k) += d_pre(i, j) * W_ffn_(j, k);
                gw += d_pre(i, j) * last_ln2_out_(i, k);
            }
            grad_W_ffn_(j, k) += gw;
        }
    }

    // ---------- LN2 backward ----------
    Tensor d_res1_from_ln2(n_q, d_model_);
    layernorm_per_row_backward(d_ln2_out, ln2_gamma_,
                                last_res1_pre_norm_, last_res1_mean_, last_res1_var_,
                                LN_EPS_,
                                grad_ln2_gamma_, grad_ln2_beta_,
                                d_res1_from_ln2);
    // d_res1 += d_res1_from_ln2
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_res1(i, j) += d_res1_from_ln2(i, j);

    // ---------- Backward through residual 1: res1 = attn_out + q_input ----------
    // d_attn_out = d_res1; d_q_input_from_res1 = d_res1.
    Tensor d_attn_out = d_res1;
    Tensor d_q_input_from_res1 = d_res1;

    // ---------- attn_out = attn_concat @ W_O ----------
    // attn_out[i, j] = sum_k attn_concat[i, k] * W_O[k, j]
    // d_attn_concat[i, k] = sum_j d_attn_out[i, j] * W_O[k, j]
    // d_W_O[k, j] += sum_i d_attn_out[i, j] * attn_concat[i, k]
    Tensor d_attn_concat(n_q, d_model_);
    d_attn_concat.fill(0.0);
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double gw = 0.0;
            for (size_t i = 0; i < n_q; ++i) {
                gw += d_attn_out(i, j) * last_attn_concat_(i, k);
            }
            grad_W_O_(k, j) += gw;
        }
        for (size_t i = 0; i < n_q; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += d_attn_out(i, j) * W_O_(k, j);
            d_attn_concat(i, k) = s;
        }
    }

    // ---------- Per-head Q/K/V backward through softmax ----------
    Tensor dQ(n_q, d_model_);
    Tensor dK(n_kv, d_model_);
    Tensor dV(n_kv, d_model_);
    for (size_t h = 0; h < num_heads_; ++h) {
        // d_scores_raw[i, j] = sum_dk d_attn_concat[i, h*dk + dk] * V[j, h*dk + dk]
        Tensor d_scores(n_q, n_kv);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < d_k_; ++dk) {
                    s += d_attn_concat(i, h * d_k_ + dk) * last_V_(j, h * d_k_ + dk);
                }
                d_scores(i, j) = s;
            }
        // Softmax backward: dZ[i, j] = A[i, j] * (dS[i, j] - sum_k A[i, k] * dS[i, k])
        for (size_t i = 0; i < n_q; ++i) {
            double sum_ad = 0.0;
            for (size_t k = 0; k < n_kv; ++k) {
                sum_ad += last_A_(h * n_q + i, k) * d_scores(i, k);
            }
            for (size_t j = 0; j < n_kv; ++j) {
                double a = last_A_(h * n_q + i, j);
                d_scores(i, j) = a * (d_scores(i, j) - sum_ad);
            }
        }
        // dZ *= scale (the scale on Q·K^T)
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < n_kv; ++j) d_scores(i, j) *= scale;

        // dQ[i, h*dk + dk] += sum_j d_scores[i, j] * K[j, h*dk + dk]
        // dK[j, h*dk + dk] += sum_i d_scores[i, j] * Q[i, h*dk + dk]
        for (size_t dk = 0; dk < d_k_; ++dk) {
            for (size_t i = 0; i < n_q; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < n_kv; ++j) {
                    s += d_scores(i, j) * last_K_(j, h * d_k_ + dk);
                }
                dQ(i, h * d_k_ + dk) = s;
            }
            for (size_t j = 0; j < n_kv; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < n_q; ++i) {
                    s += d_scores(i, j) * last_Q_(i, h * d_k_ + dk);
                }
                dK(j, h * d_k_ + dk) = s;
            }
        }
        // dV[j, h*dk + dk] = sum_i A[i, j] * d_attn_concat[i, h*dk + dk]
        //                   (transpose: V is (n_kv, d_model), attn_concat is (n_q, d_model))
        for (size_t dk = 0; dk < d_k_; ++dk) {
            for (size_t j = 0; j < n_kv; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < n_q; ++i) {
                    s += last_A_(h * n_q + i, j) * d_attn_concat(i, h * d_k_ + dk);
                }
                dV(j, h * d_k_ + dk) = s;
            }
        }
    }

    // ---------- Q = last_q_normed_ @ W_q_ ----------
    // Q[i, j] = sum_k last_q_normed_[i, k] * W_q_[k, j]
    // d_q_normed[i, k] = sum_j dQ[i, j] * W_q_[k, j]
    // d_W_q_[k, j] += sum_i dQ[i, j] * last_q_normed_[i, k]
    Tensor d_q_normed(n_q, d_model_);
    d_q_normed.fill(0.0);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t k = 0; k < d_model_; ++k) {
                d_q_normed(i, k) += dQ(i, j) * W_q_(k, j);
                grad_W_q_(k, j) += dQ(i, j) * last_q_normed_(i, k);
            }
        }

    // ---------- K = last_kv_normed_ @ W_k_, V = last_kv_normed_ @ W_v_ ----------
    // K[i, j] = sum_k last_kv_normed_[i, k] * W_k_[k, j]
    // V[i, j] = sum_k last_kv_normed_[i, k] * W_v_[k, j]
    // d_kv_normed[i, k] = sum_j (dK[i, j] * W_k_[k, j] + dV[i, j] * W_v_[k, j])
    // d_W_k_[k, j] += sum_i dK[i, j] * last_kv_normed_[i, k]
    // d_W_v_[k, j] += sum_i dV[i, j] * last_kv_normed_[i, k]
    Tensor d_kv_normed(n_kv, d_model_);
    d_kv_normed.fill(0.0);
    for (size_t i = 0; i < n_kv; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            for (size_t k = 0; k < d_model_; ++k) {
                d_kv_normed(i, k) += dK(i, j) * W_k_(k, j) + dV(i, j) * W_v_(k, j);
                grad_W_k_(k, j) += dK(i, j) * last_kv_normed_(i, k);
                grad_W_v_(k, j) += dV(i, j) * last_kv_normed_(i, k);
            }
        }

    // ---------- LN1 q-side backward ----------
    Tensor d_q_input(n_q, d_model_);
    layernorm_per_row_backward(d_q_normed, ln1_gamma_,
                                last_q_pre_norm_, last_q_mean_, last_q_var_, LN_EPS_,
                                grad_ln1_gamma_, grad_ln1_beta_,
                                d_q_input);
    // Add residual-1 contribution to d_q_input
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_q_input(i, j) += d_q_input_from_res1(i, j);

    // ---------- LN1 kv-side backward ----------
    Tensor d_kv_input(n_kv, d_model_);
    layernorm_per_row_backward(d_kv_normed, ln1_gamma_,
                                last_kv_pre_norm_, last_kv_mean_, last_kv_var_, LN_EPS_,
                                grad_ln1_gamma_, grad_ln1_beta_,
                                d_kv_input);

    GradPair result;
    if (last_was_self_) {
        // Self-attention: d_q == d_kv (same input). Add them.
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < d_model_; ++j) d_q_input(i, j) += d_kv_input(i, j);
        result.d_q = d_q_input;
        result.d_kv = Tensor(0, 0);
    } else {
        result.d_q = d_q_input;
        result.d_kv = d_kv_input;
    }
    return result;
}

Tensor MAB::backward(const Tensor& grad_output, double learning_rate) {
    GradPair g = backward_full(grad_output, learning_rate);
    return g.d_q;
}

void MAB::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j) w(i, j) -= learning_rate * g(i, j);
    };
    sgd(W_q_, grad_W_q_); sgd(W_k_, grad_W_k_);
    sgd(W_v_, grad_W_v_); sgd(W_O_, grad_W_O_);
    sgd(W_ffn_, grad_W_ffn_); sgd(b_ffn_, grad_b_ffn_);
    sgd(W_ffn_out_, grad_W_ffn_out_); sgd(b_ffn_out_, grad_b_ffn_out_);
    sgd(ln1_gamma_, grad_ln1_gamma_); sgd(ln1_beta_, grad_ln1_beta_);
    sgd(ln2_gamma_, grad_ln2_gamma_); sgd(ln2_beta_, grad_ln2_beta_);
}

// ============================================================================
// SAB
// ============================================================================

SAB::SAB(size_t d_model, size_t num_heads, size_t ffn_hidden)
    : mab_(d_model, num_heads, ffn_hidden) {}

Tensor SAB::forward(const Tensor& input) {
    return mab_.forward(input);  // last_was_self_ set inside mab_.forward
}
Tensor SAB::backward(const Tensor& grad_output, double learning_rate) {
    return mab_.backward(grad_output, learning_rate);
}
void SAB::update_weights(double learning_rate) { mab_.update_weights(learning_rate); }
void SAB::zero_grad() { mab_.zero_grad(); }
std::vector<Tensor*> SAB::parameters() { return mab_.parameters(); }
std::vector<Tensor*> SAB::gradients() { return mab_.gradients(); }

// ============================================================================
// ISAB
// ============================================================================

ISAB::ISAB(size_t d_model, size_t num_heads, size_t num_inds, size_t ffn_hidden)
    : mab1_(d_model == 0 ? 1 : d_model,
            num_heads == 0 ? 1 : num_heads,
            ffn_hidden == 0 ? (4 * (d_model == 0 ? 1 : d_model)) : ffn_hidden),
      mab2_(d_model == 0 ? 1 : d_model,
            num_heads == 0 ? 1 : num_heads,
            ffn_hidden == 0 ? (4 * (d_model == 0 ? 1 : d_model)) : ffn_hidden),
      I_(num_inds == 0 ? 1 : num_inds, d_model == 0 ? 1 : d_model),
      grad_I_(num_inds == 0 ? 1 : num_inds, d_model == 0 ? 1 : d_model)
{
    if (d_model == 0)
        throw std::invalid_argument("ISAB: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("ISAB: num_heads must be > 0");
    if (num_inds == 0)
        throw std::invalid_argument("ISAB: num_inds must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("ISAB: d_model must be divisible by num_heads");

    I_ = Tensor::random(num_inds, d_model, 0.1);
    grad_I_.fill(0.0);
}

Tensor ISAB::forward(const Tensor& input) {
    if (input.cols != I_.cols)
        throw std::invalid_argument("ISAB: input cols must match d_model");
    // ISAB(X, I) = MAB(MAB(X, I), I)
    Tensor H = mab1_.forward(input, I_);
    Tensor out = mab2_.forward(H, I_);
    return out;
}

Tensor ISAB::backward(const Tensor& grad_output, double learning_rate) {
    // Backward through mab2 then mab1. Both forward calls used I_, so the
    // backward_full() returns d_kv for the I_ input. Sum those into grad_I_.
    MAB::GradPair g2 = mab2_.backward_full(grad_output, learning_rate);
    MAB::GradPair g1 = mab1_.backward_full(g2.d_q, learning_rate);

    // grad_I_ accumulates the kv-side gradient from both MAB calls.
    if (g2.d_kv.rows > 0) {
        if (grad_I_.rows != g2.d_kv.rows || grad_I_.cols != g2.d_kv.cols)
            grad_I_ = Tensor(g2.d_kv.rows, g2.d_kv.cols);
        for (size_t i = 0; i < g2.d_kv.rows; ++i)
            for (size_t j = 0; j < g2.d_kv.cols; ++j)
                grad_I_(i, j) += g2.d_kv(i, j);
    }
    if (g1.d_kv.rows > 0) {
        for (size_t i = 0; i < g1.d_kv.rows; ++i)
            for (size_t j = 0; j < g1.d_kv.cols; ++j)
                grad_I_(i, j) += g1.d_kv(i, j);
    }

    return g1.d_q;  // dL/dX
}

void ISAB::update_weights(double learning_rate) {
    mab1_.update_weights(learning_rate);
    mab2_.update_weights(learning_rate);
    for (size_t i = 0; i < I_.rows; ++i)
        for (size_t j = 0; j < I_.cols; ++j)
            I_(i, j) -= learning_rate * grad_I_(i, j);
}

void ISAB::zero_grad() {
    mab1_.zero_grad();
    mab2_.zero_grad();
    grad_I_.fill(0.0);
}

std::vector<Tensor*> ISAB::parameters() {
    auto p1 = mab1_.parameters();
    auto p2 = mab2_.parameters();
    std::vector<Tensor*> all;
    all.reserve(p1.size() + p2.size() + 1);
    for (auto* p : p1) all.push_back(p);
    for (auto* p : p2) all.push_back(p);
    all.push_back(&I_);
    return all;
}

std::vector<Tensor*> ISAB::gradients() {
    auto g1 = mab1_.gradients();
    auto g2 = mab2_.gradients();
    std::vector<Tensor*> all;
    all.reserve(g1.size() + g2.size() + 1);
    for (auto* g : g1) all.push_back(g);
    for (auto* g : g2) all.push_back(g);
    all.push_back(&grad_I_);
    return all;
}

// ============================================================================
// PMA
// ============================================================================

PMA::PMA(size_t d_model, size_t num_heads, size_t num_seeds, size_t ffn_hidden)
    : num_seeds_(num_seeds == 0 ? 1 : num_seeds),
      mab_(d_model == 0 ? 1 : d_model,
           num_heads == 0 ? 1 : num_heads,
           ffn_hidden == 0 ? (4 * (d_model == 0 ? 1 : d_model)) : ffn_hidden),
      S_(num_seeds == 0 ? 1 : num_seeds, d_model == 0 ? 1 : d_model),
      grad_S_(num_seeds == 0 ? 1 : num_seeds, d_model == 0 ? 1 : d_model)
{
    if (d_model == 0)
        throw std::invalid_argument("PMA: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("PMA: num_heads must be > 0");
    if (num_seeds == 0)
        throw std::invalid_argument("PMA: num_seeds must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("PMA: d_model must be divisible by num_heads");

    S_ = Tensor::random(num_seeds, d_model, 0.1);
    grad_S_.fill(0.0);
}

Tensor PMA::forward(const Tensor& input) {
    return mab_.forward(S_, input);  // MAB(S, X): Q from S, K/V from input
}

Tensor PMA::backward(const Tensor& grad_output, double learning_rate) {
    MAB::GradPair g = mab_.backward_full(grad_output, learning_rate);
    grad_S_ = g.d_q;       // gradient w.r.t. the seed tensor S
    return g.d_kv;          // gradient w.r.t. the input X
}

void PMA::update_weights(double learning_rate) {
    mab_.update_weights(learning_rate);
    for (size_t i = 0; i < S_.rows; ++i)
        for (size_t j = 0; j < S_.cols; ++j)
            S_(i, j) -= learning_rate * grad_S_(i, j);
}

void PMA::zero_grad() {
    mab_.zero_grad();
    grad_S_.fill(0.0);
}

std::vector<Tensor*> PMA::parameters() {
    auto p = mab_.parameters();
    std::vector<Tensor*> all;
    all.reserve(p.size() + 1);
    for (auto* pp : p) all.push_back(pp);
    all.push_back(&S_);
    return all;
}

std::vector<Tensor*> PMA::gradients() {
    auto g = mab_.gradients();
    std::vector<Tensor*> all;
    all.reserve(g.size() + 1);
    for (auto* gg : g) all.push_back(gg);
    all.push_back(&grad_S_);
    return all;
}

// ============================================================================
// SetTransformer
// ============================================================================

SetTransformer::SetTransformer(size_t input_dim, size_t hidden_dim, size_t num_heads,
                                size_t num_inds, size_t num_seeds,
                                size_t num_enc_blocks, size_t num_dec_blocks,
                                size_t output_dim)
    : input_dim_(input_dim == 0 ? 1 : input_dim),
      hidden_dim_(hidden_dim == 0 ? 1 : hidden_dim),
      num_heads_(num_heads == 0 ? 1 : num_heads),
      num_inds_(num_inds == 0 ? 1 : num_inds),
      num_seeds_(num_seeds == 0 ? 1 : num_seeds),
      num_enc_blocks_(num_enc_blocks == 0 ? 1 : num_enc_blocks),
      num_dec_blocks_(num_dec_blocks == 0 ? 1 : num_dec_blocks),
      output_dim_(output_dim == 0 ? 1 : output_dim),
      input_proj_(input_dim == 0 ? 1 : input_dim,
                  hidden_dim == 0 ? 1 : hidden_dim),
      pma_(hidden_dim == 0 ? 1 : hidden_dim,
           num_heads == 0 ? 1 : num_heads,
           num_seeds == 0 ? 1 : num_seeds),
      head_(num_seeds * hidden_dim,  // in_features = num_seeds * hidden_dim
            output_dim == 0 ? 1 : output_dim)
{
    if (input_dim == 0)
        throw std::invalid_argument("SetTransformer: input_dim must be > 0");
    if (hidden_dim == 0)
        throw std::invalid_argument("SetTransformer: hidden_dim must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("SetTransformer: num_heads must be > 0");
    if (num_inds == 0)
        throw std::invalid_argument("SetTransformer: num_inds must be > 0");
    if (num_seeds == 0)
        throw std::invalid_argument("SetTransformer: num_seeds must be > 0");
    if (num_enc_blocks == 0)
        throw std::invalid_argument("SetTransformer: num_enc_blocks must be > 0");
    if (num_dec_blocks == 0)
        throw std::invalid_argument("SetTransformer: num_dec_blocks must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("SetTransformer: output_dim must be > 0");
    if (hidden_dim % num_heads != 0)
        throw std::invalid_argument("SetTransformer: hidden_dim must be divisible by num_heads");

    enc_.reserve(num_enc_blocks);
    for (size_t i = 0; i < num_enc_blocks; ++i) {
        enc_.emplace_back(hidden_dim, num_heads, num_inds,
                          4 * hidden_dim);
    }
    dec_.reserve(num_dec_blocks);
    for (size_t i = 0; i < num_dec_blocks; ++i) {
        dec_.emplace_back(hidden_dim, num_heads, 4 * hidden_dim);
    }
}

Tensor SetTransformer::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("SetTransformer: input cols must be input_dim");
    if (input.rows == 0)
        throw std::invalid_argument("SetTransformer: input rows must be > 0");

    last_input_ = input;
    const size_t n_tokens = input.rows;

    // ---- Manual input projection: proj[i, j] = sum_k input[i, k] * W_in[j, k] + b_in[j] ----
    // input_proj_.weights shape (hidden_dim, input_dim); proj shape (n_tokens, hidden_dim).
    Tensor proj(n_tokens, hidden_dim_);
    for (size_t i = 0; i < n_tokens; ++i)
        for (size_t j = 0; j < hidden_dim_; ++j) {
            double s = input_proj_.bias(0, j);
            for (size_t k = 0; k < input_dim_; ++k) s += input(i, k) * input_proj_.weights(j, k);
            proj(i, j) = s;
        }
    last_proj_ = proj;

    // ---- Encoder stack ----
    Tensor x = proj;
    last_enc_.clear();
    last_enc_.reserve(enc_.size());
    for (auto& block : enc_) {
        x = block.forward(x);
        last_enc_.push_back(x);
    }

    // ---- PMA ----
    Tensor pooled = pma_.forward(x);   // (num_seeds, hidden_dim)
    last_pma_ = pooled;

    // ---- Decoder stack ----
    Tensor y_dec = pooled;
    last_dec_.clear();
    last_dec_.reserve(dec_.size());
    for (auto& block : dec_) {
        y_dec = block.forward(y_dec);
        last_dec_.push_back(y_dec);
    }

    // ---- Flatten the decoder output (num_seeds, hidden_dim) → (1, num_seeds * hidden_dim) ----
    Tensor flat(1, num_seeds_ * hidden_dim_);
    for (size_t s = 0; s < num_seeds_; ++s)
        for (size_t d = 0; d < hidden_dim_; ++d)
            flat(0, s * hidden_dim_ + d) = y_dec(s, d);
    last_flat_ = flat;

    // ---- Manual head: logits[0, j] = sum_k head.weights[j, k] * flat[0, k] + head.bias[0, j] ----
    // head_.weights shape (output_dim, num_seeds * hidden_dim); logits shape (1, output_dim).
    Tensor logits(1, output_dim_);
    for (size_t j = 0; j < output_dim_; ++j) {
        double s = head_.bias(0, j);
        for (size_t k = 0; k < num_seeds_ * hidden_dim_; ++k) {
            s += head_.weights(j, k) * flat(0, k);
        }
        logits(0, j) = s;
    }
    return logits;
}

Tensor SetTransformer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != 1 || grad_output.cols != output_dim_)
        throw std::invalid_argument("SetTransformer::backward: grad_output shape mismatch");

    // ---- Backward through head ----
    // logits[0, j] = sum_k head.weights[j, k] * flat[0, k] + head.bias[0, j]
    // d_flat[0, k] = sum_j head.weights[j, k] * grad_output[0, j]
    // d_head.weights[j, k] += grad_output[0, j] * flat[0, k]
    // d_head.bias[0, j] += grad_output[0, j]
    Tensor d_flat(1, num_seeds_ * hidden_dim_);
    for (size_t k = 0; k < num_seeds_ * hidden_dim_; ++k) {
        double s = 0.0;
        for (size_t j = 0; j < output_dim_; ++j) {
            s += head_.weights(j, k) * grad_output(0, j);
            head_.grad_weights(j, k) += grad_output(0, j) * last_flat_(0, k);
        }
        d_flat(0, k) = s;
    }
    for (size_t j = 0; j < output_dim_; ++j) {
        head_.grad_bias(0, j) += grad_output(0, j);
    }

    // ---- Reshape d_flat to (num_seeds, hidden_dim) ----
    Tensor d_pooled(num_seeds_, hidden_dim_);
    for (size_t s = 0; s < num_seeds_; ++s)
        for (size_t d = 0; d < hidden_dim_; ++d)
            d_pooled(s, d) = d_flat(0, s * hidden_dim_ + d);

    // ---- Backward through decoder stack (reverse order) ----
    Tensor d = d_pooled;
    for (size_t i = dec_.size(); i > 0; --i) {
        d = dec_[i - 1].backward(d, 0.0);
    }

    // ---- Backward through PMA ----
    Tensor d_x = pma_.backward(d, 0.0);

    // ---- Backward through encoder stack (reverse order) ----
    for (size_t i = enc_.size(); i > 0; --i) {
        d_x = enc_[i - 1].backward(d_x, 0.0);
    }

    // ---- Backward through input projection ----
    // proj[i, j] = sum_k input[i, k] * input_proj_.weights[j, k] + input_proj_.bias[0, j]
    // d_input[i, k] = sum_j d_proj[i, j] * input_proj_.weights[j, k]
    // d_input_proj_.weights[j, k] += sum_i d_proj[i, j] * input[i, k]
    // d_input_proj_.bias[0, j] += sum_i d_proj[i, j]
    Tensor d_input(last_input_.rows, input_dim_);
    for (size_t i = 0; i < last_input_.rows; ++i)
        for (size_t k = 0; k < input_dim_; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < hidden_dim_; ++j) {
                s += d_x(i, j) * input_proj_.weights(j, k);
                input_proj_.grad_weights(j, k) += d_x(i, j) * last_input_(i, k);
            }
            d_input(i, k) = s;
        }
    for (size_t j = 0; j < hidden_dim_; ++j) {
        double gb = 0.0;
        for (size_t i = 0; i < last_input_.rows; ++i) gb += d_x(i, j);
        input_proj_.grad_bias(0, j) += gb;
    }

    return d_input;
}

void SetTransformer::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& b : enc_) b.update_weights(learning_rate);
    pma_.update_weights(learning_rate);
    for (auto& b : dec_) b.update_weights(learning_rate);
    head_.update_weights(learning_rate);
}

void SetTransformer::zero_grad() {
    input_proj_.zero_grad();
    for (auto& b : enc_) b.zero_grad();
    pma_.zero_grad();
    for (auto& b : dec_) b.zero_grad();
    head_.zero_grad();
}

std::vector<Tensor*> SetTransformer::parameters() {
    std::vector<Tensor*> all;
    auto p = input_proj_.parameters();
    for (auto* pp : p) all.push_back(pp);
    for (auto& b : enc_) {
        auto bp = b.parameters();
        for (auto* pp : bp) all.push_back(pp);
    }
    auto pp = pma_.parameters();
    for (auto* ppp : pp) all.push_back(ppp);
    for (auto& b : dec_) {
        auto bp = b.parameters();
        for (auto* pp : bp) all.push_back(pp);
    }
    auto hp = head_.parameters();
    for (auto* pp : hp) all.push_back(pp);
    return all;
}

std::vector<Tensor*> SetTransformer::gradients() {
    std::vector<Tensor*> all;
    auto g = input_proj_.gradients();
    for (auto* gg : g) all.push_back(gg);
    for (auto& b : enc_) {
        auto bg = b.gradients();
        for (auto* gg : bg) all.push_back(gg);
    }
    auto pg = pma_.gradients();
    for (auto* gg : pg) all.push_back(gg);
    for (auto& b : dec_) {
        auto bg = b.gradients();
        for (auto* gg : bg) all.push_back(gg);
    }
    auto hg = head_.gradients();
    for (auto* gg : hg) all.push_back(gg);
    return all;
}

size_t SetTransformer::num_parameters() const {
    size_t total = 0;
    for (auto* p : const_cast<SetTransformer*>(this)->parameters()) {
        total += p->rows * p->cols;
    }
    return total;
}