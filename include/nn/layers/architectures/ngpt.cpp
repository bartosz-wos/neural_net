// nGPT — Normalized Transformer on the Hypersphere — implementation
//   Loshchilov, Hsieh, Sun, Ginsburg (NVIDIA 2024), https://arxiv.org/abs/2410.01131
//
// See ngpt.h for the full derivation of the row-normalization Jacobian and the
// normalized-LERP residual backward.
#include "ngpt.h"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace {

constexpr double kNormEps = 1e-12;

inline double gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
inline double gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    return cdf + x * pdf;
}

}  // namespace

// ============================================================================
// Shared Norm helpers
// ============================================================================

Tensor ngpt_row_normalize(const Tensor& v, std::vector<double>& norms) {
    Tensor out(v.rows, v.cols);
    norms.assign(v.rows, 0.0);
    for (size_t r = 0; r < v.rows; ++r) {
        double n2 = 0.0;
        for (size_t c = 0; c < v.cols; ++c) n2 += v(r, c) * v(r, c);
        double n = std::sqrt(n2);
        norms[r] = n;
        if (n > kNormEps) {
            double inv = 1.0 / n;
            for (size_t c = 0; c < v.cols; ++c) out(r, c) = v(r, c) * inv;
        } else {
            for (size_t c = 0; c < v.cols; ++c) out(r, c) = 0.0;
        }
    }
    return out;
}

// dL/dv = (dv_hat - v_hat * (v_hat . dv_hat)) / ||v||
Tensor ngpt_norm_backward(const Tensor& v_hat, const std::vector<double>& norms,
                          const Tensor& d_v_hat) {
    Tensor out(v_hat.rows, v_hat.cols);
    out.fill(0.0);
    for (size_t r = 0; r < v_hat.rows; ++r) {
        double n = norms[r];
        if (n <= kNormEps) continue;
        double radial = 0.0;
        for (size_t c = 0; c < v_hat.cols; ++c) radial += v_hat(r, c) * d_v_hat(r, c);
        double inv = 1.0 / n;
        for (size_t c = 0; c < v_hat.cols; ++c)
            out(r, c) = (d_v_hat(r, c) - v_hat(r, c) * radial) * inv;
    }
    return out;
}

// ============================================================================
// HypersphereLinear
// ============================================================================

HypersphereLinear::HypersphereLinear(size_t in_features, size_t out_features)
    : in_(in_features), out_(out_features)
{
    if (in_ == 0 || out_ == 0) {
        throw std::invalid_argument(
            "HypersphereLinear: in_features and out_features must both be > 0");
    }
    // Random init then immediate projection onto the sphere. Independent RNG draws
    // (not a uniform fill) so FD checks exercise row-vs-column asymmetry.
    W = Tensor(out_, in_);
    static std::mt19937 gen(1337);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = dist(gen);
    renormalize();

    s = Tensor(1, out_);
    s.fill(1.0);

    grad_W = Tensor::zeros(out_, in_);
    grad_s = Tensor::zeros(1, out_);
}

void HypersphereLinear::renormalize() {
    for (size_t o = 0; o < W.rows; ++o) {
        double n2 = 0.0;
        for (size_t c = 0; c < W.cols; ++c) n2 += W(o, c) * W(o, c);
        double n = std::sqrt(n2);
        if (n > kNormEps) {
            double inv = 1.0 / n;
            for (size_t c = 0; c < W.cols; ++c) W(o, c) *= inv;
        } else {
            // Degenerate row: reset to a canonical basis vector so the row stays
            // on the sphere rather than collapsing to zero.
            for (size_t c = 0; c < W.cols; ++c) W(o, c) = (c == 0) ? 1.0 : 0.0;
        }
    }
}

Tensor HypersphereLinear::normalized_weights() const {
    std::vector<double> tmp;
    return ngpt_row_normalize(W, tmp);
}

std::vector<Tensor*> HypersphereLinear::parameters() { return { &W, &s }; }
std::vector<Tensor*> HypersphereLinear::gradients()  { return { &grad_W, &grad_s }; }

void HypersphereLinear::zero_grad() {
    grad_W.fill(0.0);
    grad_s.fill(0.0);
}

void HypersphereLinear::update_weights(double lr) {
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] -= lr * grad_W.data[i];
    for (size_t i = 0; i < s.data.size(); ++i) s.data[i] -= lr * grad_s.data[i];
    // Re-projection onto the sphere IS part of the step (paper §2.6).
    renormalize();
}

Tensor HypersphereLinear::forward(const Tensor& input) {
    if (input.cols != in_) {
        throw std::invalid_argument("HypersphereLinear::forward: input.cols != in_features");
    }
    const size_t N = input.rows;
    last_input_ = input.clone();
    last_W_hat_ = ngpt_row_normalize(W, last_norms_);

    last_pre_s_ = Tensor(N, out_);
    Tensor y(N, out_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t o = 0; o < out_; ++o) {
            double acc = 0.0;
            for (size_t c = 0; c < in_; ++c) acc += input(t, c) * last_W_hat_(o, c);
            last_pre_s_(t, o) = acc;
            y(t, o) = s(0, o) * acc;
        }
    }
    return y;
}

Tensor HypersphereLinear::backward(const Tensor& grad_output, double /*lr*/) {
    const size_t N = grad_output.rows;

    // ds[o] = sum_t dy[t, o] * pre_s[t, o]
    for (size_t o = 0; o < out_; ++o) {
        double g = 0.0;
        for (size_t t = 0; t < N; ++t) g += grad_output(t, o) * last_pre_s_(t, o);
        grad_s(0, o) += g;
    }

    // dW_hat[o, c] = sum_t dy[t, o] * s[o] * x[t, c]
    Tensor dW_hat(out_, in_);
    dW_hat.fill(0.0);
    for (size_t o = 0; o < out_; ++o) {
        double so = s(0, o);
        for (size_t c = 0; c < in_; ++c) {
            double g = 0.0;
            for (size_t t = 0; t < N; ++t) g += grad_output(t, o) * last_input_(t, c);
            dW_hat(o, c) = g * so;
        }
    }
    // Project through the row-normalization Jacobian.
    Tensor dW = ngpt_norm_backward(last_W_hat_, last_norms_, dW_hat);
    for (size_t i = 0; i < grad_W.data.size(); ++i) grad_W.data[i] += dW.data[i];

    // dx[t, c] = sum_o dy[t, o] * s[o] * W_hat[o, c]
    Tensor d_input(N, in_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t c = 0; c < in_; ++c) {
            double acc = 0.0;
            for (size_t o = 0; o < out_; ++o)
                acc += grad_output(t, o) * s(0, o) * last_W_hat_(o, c);
            d_input(t, c) = acc;
        }
    }
    return d_input;
}

// ============================================================================
// NGPTBlock
// ============================================================================

NGPTBlock::NGPTBlock(size_t d_model, size_t num_heads, size_t ffn_dim)
    : W_q(d_model == 0 ? 1 : d_model, d_model == 0 ? 1 : d_model),
      W_k(d_model == 0 ? 1 : d_model, d_model == 0 ? 1 : d_model),
      W_v(d_model == 0 ? 1 : d_model, d_model == 0 ? 1 : d_model),
      W_o(d_model == 0 ? 1 : d_model, d_model == 0 ? 1 : d_model),
      fc1(d_model == 0 ? 1 : d_model,
          ffn_dim == 0 ? 4 * (d_model == 0 ? 1 : d_model) : ffn_dim),
      fc2(ffn_dim == 0 ? 4 * (d_model == 0 ? 1 : d_model) : ffn_dim,
          d_model == 0 ? 1 : d_model),
      d_model_(d_model), num_heads_(num_heads),
      head_dim_(0), ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      sqrt_dh_(1.0)
{
    // Validate AFTER the initializer list (which uses dummy sizes when d_model==0)
    // so a zero d_model cannot trigger a divide-by-zero in head_dim_.
    if (d_model_ == 0 || num_heads_ == 0) {
        throw std::invalid_argument("NGPTBlock: d_model and num_heads must both be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("NGPTBlock: d_model must be divisible by num_heads");
    }
    head_dim_ = d_model_ / num_heads_;
    sqrt_dh_ = std::sqrt(static_cast<double>(head_dim_));

    s_qk = Tensor(1, 1);
    s_qk.fill(1.0);
    // Eigen learning rates. Paper uses ~0.05/sqrt(d_model); we use a plain 0.05
    // because d_model here is small and the 1/sqrt(d) factor would put alpha at
    // the FD noise floor (see the deviation table in ngpt.h).
    alpha_attn = Tensor(1, d_model_);
    alpha_attn.fill(0.05);
    alpha_mlp = Tensor(1, d_model_);
    alpha_mlp.fill(0.05);

    grad_s_qk = Tensor::zeros(1, 1);
    grad_alpha_attn = Tensor::zeros(1, d_model_);
    grad_alpha_mlp = Tensor::zeros(1, d_model_);
}

std::vector<Tensor*> NGPTBlock::parameters() {
    std::vector<Tensor*> p;
    for (HypersphereLinear* l : { &W_q, &W_k, &W_v, &W_o, &fc1, &fc2 }) {
        auto lp = l->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    p.push_back(&s_qk);
    p.push_back(&alpha_attn);
    p.push_back(&alpha_mlp);
    return p;
}

std::vector<Tensor*> NGPTBlock::gradients() {
    std::vector<Tensor*> g;
    for (HypersphereLinear* l : { &W_q, &W_k, &W_v, &W_o, &fc1, &fc2 }) {
        auto lg = l->gradients();
        g.insert(g.end(), lg.begin(), lg.end());
    }
    g.push_back(&grad_s_qk);
    g.push_back(&grad_alpha_attn);
    g.push_back(&grad_alpha_mlp);
    return g;
}

void NGPTBlock::zero_grad() {
    W_q.zero_grad(); W_k.zero_grad(); W_v.zero_grad(); W_o.zero_grad();
    fc1.zero_grad(); fc2.zero_grad();
    grad_s_qk.fill(0.0);
    grad_alpha_attn.fill(0.0);
    grad_alpha_mlp.fill(0.0);
}

void NGPTBlock::update_weights(double lr) {
    W_q.update_weights(lr); W_k.update_weights(lr);
    W_v.update_weights(lr); W_o.update_weights(lr);
    fc1.update_weights(lr); fc2.update_weights(lr);
    for (size_t i = 0; i < s_qk.data.size(); ++i) s_qk.data[i] -= lr * grad_s_qk.data[i];
    for (size_t i = 0; i < alpha_attn.data.size(); ++i)
        alpha_attn.data[i] -= lr * grad_alpha_attn.data[i];
    for (size_t i = 0; i < alpha_mlp.data.size(); ++i)
        alpha_mlp.data[i] -= lr * grad_alpha_mlp.data[i];
}

Tensor NGPTBlock::forward(const Tensor& input) {
    const size_t N = input.rows;
    last_input_ = input.clone();

    // ---------------- Attention sublayer ----------------
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    const double logit_scale = s_qk(0, 0) * sqrt_dh_;
    last_attn_ = Tensor(num_heads_ * N, N);
    last_attn_.fill(0.0);
    last_head_out_ = Tensor(N, d_model_);
    last_head_out_.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            std::vector<double> sc(t + 1, 0.0);
            double m = -std::numeric_limits<double>::infinity();
            for (size_t sIdx = 0; sIdx <= t; ++sIdx) {
                double dot = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    dot += last_Q_(t, off + dk) * last_K_(sIdx, off + dk);
                sc[sIdx] = dot * logit_scale;
                if (sc[sIdx] > m) m = sc[sIdx];
            }
            double l = 0.0;
            for (size_t sIdx = 0; sIdx <= t; ++sIdx) { sc[sIdx] = std::exp(sc[sIdx] - m); l += sc[sIdx]; }
            const double inv_l = 1.0 / l;
            for (size_t sIdx = 0; sIdx <= t; ++sIdx) {
                sc[sIdx] *= inv_l;
                last_attn_(h * N + t, sIdx) = sc[sIdx];
            }
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t sIdx = 0; sIdx <= t; ++sIdx) v += sc[sIdx] * last_V_(sIdx, off + dk);
                last_head_out_(t, off + dk) = v;
            }
        }
    }

    last_f_attn_ = W_o.forward(last_head_out_);
    last_u_attn_ = ngpt_row_normalize(last_f_attn_, last_n_f_attn_);

    // Normalized LERP: m = h + alpha * (u - h);  h_mid = Norm(m)
    last_m_attn_ = Tensor(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t c = 0; c < d_model_; ++c)
            last_m_attn_(t, c) = input(t, c)
                               + alpha_attn(0, c) * (last_u_attn_(t, c) - input(t, c));
    last_h_mid_ = ngpt_row_normalize(last_m_attn_, last_n_m_attn_);

    // ---------------- MLP sublayer ----------------
    last_fc1_pre_ = fc1.forward(last_h_mid_);
    last_fc1_act_ = Tensor(N, ffn_dim_);
    for (size_t i = 0; i < last_fc1_pre_.data.size(); ++i)
        last_fc1_act_.data[i] = gelu(last_fc1_pre_.data[i]);
    last_f_mlp_ = fc2.forward(last_fc1_act_);
    last_u_mlp_ = ngpt_row_normalize(last_f_mlp_, last_n_f_mlp_);

    last_m_mlp_ = Tensor(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t c = 0; c < d_model_; ++c)
            last_m_mlp_(t, c) = last_h_mid_(t, c)
                              + alpha_mlp(0, c) * (last_u_mlp_(t, c) - last_h_mid_(t, c));
    last_h_out_ = ngpt_row_normalize(last_m_mlp_, last_n_m_mlp_);

    return last_h_out_;
}

Tensor NGPTBlock::backward(const Tensor& grad_output, double lr) {
    const size_t N = grad_output.rows;

    // ---------------- MLP sublayer backward ----------------
    // h_out = Norm(m_mlp)
    Tensor d_m_mlp = ngpt_norm_backward(last_h_out_, last_n_m_mlp_, grad_output);

    // m_mlp = h_mid + alpha_mlp * (u_mlp - h_mid)
    Tensor d_u_mlp(N, d_model_);
    Tensor d_h_mid(N, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double dm = d_m_mlp(t, c);
            double a = alpha_mlp(0, c);
            grad_alpha_mlp(0, c) += dm * (last_u_mlp_(t, c) - last_h_mid_(t, c));
            d_u_mlp(t, c) = dm * a;
            d_h_mid(t, c) = dm * (1.0 - a);
        }
    }

    // u_mlp = Norm(f_mlp)
    Tensor d_f_mlp = ngpt_norm_backward(last_u_mlp_, last_n_f_mlp_, d_u_mlp);
    Tensor d_fc1_act = fc2.backward(d_f_mlp, lr);
    Tensor d_fc1_pre(N, ffn_dim_);
    for (size_t i = 0; i < d_fc1_pre.data.size(); ++i)
        d_fc1_pre.data[i] = d_fc1_act.data[i] * gelu_deriv(last_fc1_pre_.data[i]);
    Tensor d_h_mid_from_mlp = fc1.backward(d_fc1_pre, lr);
    for (size_t i = 0; i < d_h_mid.data.size(); ++i)
        d_h_mid.data[i] += d_h_mid_from_mlp.data[i];

    // ---------------- Attention sublayer backward ----------------
    // h_mid = Norm(m_attn)
    Tensor d_m_attn = ngpt_norm_backward(last_h_mid_, last_n_m_attn_, d_h_mid);

    Tensor d_u_attn(N, d_model_);
    Tensor d_input(N, d_model_);
    d_input.fill(0.0);
    for (size_t t = 0; t < N; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double dm = d_m_attn(t, c);
            double a = alpha_attn(0, c);
            grad_alpha_attn(0, c) += dm * (last_u_attn_(t, c) - last_input_(t, c));
            d_u_attn(t, c) = dm * a;
            d_input(t, c) += dm * (1.0 - a);
        }
    }

    // u_attn = Norm(f_attn);  f_attn = W_o(head_out)
    Tensor d_f_attn = ngpt_norm_backward(last_u_attn_, last_n_f_attn_, d_u_attn);
    Tensor d_head = W_o.backward(d_f_attn, lr);

    // Per-head softmax attention backward.
    const double logit_scale = s_qk(0, 0) * sqrt_dh_;
    Tensor dQ(N, d_model_), dK(N, d_model_), dV(N, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);
    double d_logit_scale = 0.0;

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            std::vector<double> dA(t + 1, 0.0);
            double row_dot = 0.0;
            for (size_t sIdx = 0; sIdx <= t; ++sIdx) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    v += d_head(t, off + dk) * last_V_(sIdx, off + dk);
                dA[sIdx] = v;
                row_dot += last_attn_(h * N + t, sIdx) * v;
            }
            for (size_t sIdx = 0; sIdx <= t; ++sIdx) {
                double a = last_attn_(h * N + t, sIdx);
                double dS = a * (dA[sIdx] - row_dot);   // softmax Jacobian
                // dV
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    dV(sIdx, off + dk) += a * d_head(t, off + dk);
                // scores = dot * logit_scale
                double dot = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    dot += last_Q_(t, off + dk) * last_K_(sIdx, off + dk);
                d_logit_scale += dS * dot;
                for (size_t dk = 0; dk < head_dim_; ++dk) {
                    dQ(t, off + dk)    += logit_scale * dS * last_K_(sIdx, off + dk);
                    dK(sIdx, off + dk) += logit_scale * dS * last_Q_(t, off + dk);
                }
            }
        }
    }
    // logit_scale = s_qk * sqrt(head_dim)
    grad_s_qk(0, 0) += d_logit_scale * sqrt_dh_;

    Tensor dx_q = W_q.backward(dQ, lr);
    Tensor dx_k = W_k.backward(dK, lr);
    Tensor dx_v = W_v.backward(dV, lr);
    for (size_t i = 0; i < d_input.data.size(); ++i)
        d_input.data[i] += dx_q.data[i] + dx_k.data[i] + dx_v.data[i];

    return d_input;
}

// ============================================================================
// NGPTModel
// ============================================================================

NGPTModel::NGPTModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_blocks, size_t num_heads, size_t ffn_dim)
    : input_proj(input_dim == 0 ? 1 : input_dim, d_model == 0 ? 1 : d_model),
      classifier(d_model == 0 ? 1 : d_model, output_dim == 0 ? 1 : output_dim),
      input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim)
{
    if (input_dim == 0 || d_model == 0 || output_dim == 0 || num_blocks == 0) {
        throw std::invalid_argument(
            "NGPTModel: input_dim, d_model, output_dim, num_blocks must all be > 0");
    }
    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
        blocks_.emplace_back(new NGPTBlock(d_model, num_heads, ffn_dim));
}

std::vector<Tensor*> NGPTModel::parameters() {
    auto p = input_proj.parameters();
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> NGPTModel::gradients() {
    auto g = input_proj.gradients();
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

void NGPTModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    classifier.zero_grad();
}

void NGPTModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    for (auto& b : blocks_) b->update_weights(lr);
    classifier.update_weights(lr);
}

Tensor NGPTModel::forward(const Tensor& input) {
    Tensor h = input_proj.forward(input);
    // Project the embedded tokens onto the sphere before the first block, so the
    // residual stream starts on the manifold the blocks assume (the projection-based
    // equivalent of the paper's normalized embedding table).
    last_embed_norm_ = ngpt_row_normalize(h, last_embed_norms_);
    h = last_embed_norm_;
    for (auto& b : blocks_) h = b->forward(h);
    return classifier.forward(h);
}

Tensor NGPTModel::backward(const Tensor& grad_output, double lr) {
    Tensor d = classifier.backward(grad_output, lr);
    for (size_t i = blocks_.size(); i-- > 0; ) d = blocks_[i]->backward(d, lr);
    // Backward through the pre-block row normalization, then the input projection.
    d = ngpt_norm_backward(last_embed_norm_, last_embed_norms_, d);
    return input_proj.backward(d, lr);
}
