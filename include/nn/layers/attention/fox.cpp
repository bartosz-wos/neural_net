// Forgetting Transformer (FoX) — implementation
//   Lin, Yang, Sun et al., ICLR 2025, https://arxiv.org/abs/2503.02130
//
// See fox.h for the full design write-up and the backward derivation.
#include "fox.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// FoXAttention
// ============================================================================

FoXAttention::FoXAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads),
      head_dim_(0),   // placeholder; set after validation (avoids SIGFPE on 1/0)
      scale_(1.0)
{
    if (d_model_ == 0 || num_heads_ == 0) {
        throw std::invalid_argument("FoXAttention: d_model and num_heads must both be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("FoXAttention: d_model must be divisible by num_heads");
    }

    head_dim_ = d_model_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9);

    W_q = Tensor::random(d_model_, d_model_, 0.05);
    W_k = Tensor::random(d_model_, d_model_, 0.05);
    W_v = Tensor::random(d_model_, d_model_, 0.05);
    W_o = Tensor::random(d_model_, d_model_, 0.05);
    // Forget-gate projection: (d_model, num_heads)
    W_f = Tensor::random(d_model_, num_heads_, 0.05);

    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
    grad_W_f = Tensor::zeros(d_model_, num_heads_);
}

std::vector<Tensor*> FoXAttention::parameters() {
    return { &W_q, &W_k, &W_v, &W_o, &W_f };
}

std::vector<Tensor*> FoXAttention::gradients() {
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_W_f };
}

void FoXAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_W_f.fill(0.0);
}

void FoXAttention::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_q, grad_W_q);
    sgd(W_k, grad_W_k);
    sgd(W_v, grad_W_v);
    sgd(W_o, grad_W_o);
    sgd(W_f, grad_W_f);
}

Tensor FoXAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_input_ = input.clone();

    // ---- Step 1: Q / K / V projections ----
    Tensor Q(n, d_model_), K(n, d_model_), V(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double q = 0.0, k = 0.0, v = 0.0;
            for (size_t c = 0; c < d_model_; ++c) {
                double x = input(i, c);
                q += x * W_q(c, j);
                k += x * W_k(c, j);
                v += x * W_v(c, j);
            }
            Q(i, j) = q; K(i, j) = k; V(i, j) = v;
        }
    }
    last_q_ = Q; last_k_ = K; last_v_ = V;

    // ---- Step 2: forget gate + cumulative log-decay ----
    // z[t, h] = (X @ W_f)[t, h];  f = sigmoid(z);  D[t, h] = sum_{i<=t} log f[i, h]
    Tensor f(n, num_heads_), D(n, num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        double acc = 0.0;
        for (size_t t = 0; t < n; ++t) {
            double z = 0.0;
            for (size_t c = 0; c < d_model_; ++c) z += input(t, c) * W_f(c, h);
            double fv = 1.0 / (1.0 + std::exp(-z));
            f(t, h) = fv;
            // log sigmoid(z) computed stably: -log(1 + exp(-z))
            double logf = (z >= 0.0) ? -std::log1p(std::exp(-z))
                                     : (z - std::log1p(std::exp(z)));
            acc += logf;
            D(t, h) = acc;
        }
    }
    last_f_ = f;
    last_D_ = D;

    // ---- Step 3: per-head causal attention with additive forget bias ----
    Tensor attn(num_heads_ * n, n);
    attn.fill(0.0);
    Tensor head_out(n, d_model_);
    head_out.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t off = h * head_dim_;
        for (size_t t = 0; t < n; ++t) {
            // scores over s <= t
            std::vector<double> sc(t + 1, 0.0);
            double m = -std::numeric_limits<double>::infinity();
            for (size_t s = 0; s <= t; ++s) {
                double dot = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    dot += Q(t, off + dk) * K(s, off + dk);
                sc[s] = dot * scale_ + (D(t, h) - D(s, h));
                if (sc[s] > m) m = sc[s];
            }
            double l = 0.0;
            for (size_t s = 0; s <= t; ++s) { sc[s] = std::exp(sc[s] - m); l += sc[s]; }
            const double inv_l = 1.0 / l;
            for (size_t s = 0; s <= t; ++s) {
                sc[s] *= inv_l;
                attn(h * n + t, s) = sc[s];
            }
            for (size_t dk = 0; dk < head_dim_; ++dk) {
                double v = 0.0;
                for (size_t s = 0; s <= t; ++s) v += sc[s] * V(s, off + dk);
                head_out(t, off + dk) = v;
            }
        }
    }
    last_attn_ = attn;
    last_head_out_ = head_out;

    // ---- Step 4: output projection ----
    Tensor output(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t c = 0; c < d_model_; ++c) v += head_out(i, c) * W_o(c, j);
            output(i, j) = v;
        }
    }
    return output;
}

Tensor FoXAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;

    // ---- dW_o + d_head_out ----
    Tensor d_head(n, d_model_);
    d_head.fill(0.0);
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t j = 0; j < d_model_; ++j) {
            double g = 0.0;
            for (size_t i = 0; i < n; ++i) g += last_head_out_(i, c) * grad_output(i, j);
            grad_W_o(c, j) += g;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < d_model_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) v += grad_output(i, j) * W_o(c, j);
            d_head(i, c) = v;
        }
    }

    // ---- Per-head attention backward ----
    Tensor dQ(n, d_model_), dK(n, d_model_), dV(n, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);
    Tensor dD(n, num_heads_);
    dD.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t off = h * head_dim_;
        // dS[t][s] (causal, s <= t)
        std::vector<std::vector<double>> dS(n);
        for (size_t t = 0; t < n; ++t) {
            dS[t].assign(t + 1, 0.0);
            // dA[t, s] = sum_dk d_head[t, off+dk] * V[s, off+dk]
            // row_dot  = sum_s A[t, s] * dA[t, s]
            double row_dot = 0.0;
            std::vector<double> dA(t + 1, 0.0);
            for (size_t s = 0; s <= t; ++s) {
                double v = 0.0;
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    v += d_head(t, off + dk) * last_v_(s, off + dk);
                dA[s] = v;
                row_dot += last_attn_(h * n + t, s) * v;
            }
            for (size_t s = 0; s <= t; ++s) {
                double a = last_attn_(h * n + t, s);
                dS[t][s] = a * (dA[s] - row_dot);
                // dV[s] += A[t, s] * d_head[t]
                for (size_t dk = 0; dk < head_dim_; ++dk)
                    dV(s, off + dk) += a * d_head(t, off + dk);
            }
        }

        // dQ, dK from dS; dD from the additive bias (D[t, h] - D[s, h])
        for (size_t t = 0; t < n; ++t) {
            for (size_t s = 0; s <= t; ++s) {
                double g = dS[t][s];
                for (size_t dk = 0; dk < head_dim_; ++dk) {
                    dQ(t, off + dk) += scale_ * g * last_k_(s, off + dk);
                    dK(s, off + dk) += scale_ * g * last_q_(t, off + dk);
                }
                dD(t, h) += g;
                dD(s, h) -= g;
            }
        }
    }

    // ---- Forget-gate chain: dD -> d_logf -> dz ----
    // D[t, h] = sum_{i <= t} logf[i, h]  =>  d_logf[i, h] = sum_{t >= i} dD[t, h]
    // d/dz log sigmoid(z) = 1 - sigmoid(z)
    Tensor dz(n, num_heads_);
    dz.fill(0.0);
    for (size_t h = 0; h < num_heads_; ++h) {
        double suffix = 0.0;
        for (size_t i = n; i-- > 0; ) {
            suffix += dD(i, h);
            dz(i, h) = suffix * (1.0 - last_f_(i, h));
        }
    }

    // ---- Projection backwards + d_input ----
    Tensor d_input(n, d_model_);
    d_input.fill(0.0);

    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t j = 0; j < d_model_; ++j) {
            double gq = 0.0, gk = 0.0, gv = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double x = last_input_(i, c);
                gq += x * dQ(i, j);
                gk += x * dK(i, j);
                gv += x * dV(i, j);
            }
            grad_W_q(c, j) += gq;
            grad_W_k(c, j) += gk;
            grad_W_v(c, j) += gv;
        }
    }
    for (size_t c = 0; c < d_model_; ++c) {
        for (size_t h = 0; h < num_heads_; ++h) {
            double gf = 0.0;
            for (size_t i = 0; i < n; ++i) gf += last_input_(i, c) * dz(i, h);
            grad_W_f(c, h) += gf;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < d_model_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += dQ(i, j) * W_q(c, j);
                v += dK(i, j) * W_k(c, j);
                v += dV(i, j) * W_v(c, j);
            }
            for (size_t h = 0; h < num_heads_; ++h) v += dz(i, h) * W_f(c, h);
            d_input(i, c) = v;
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// FoXBlock
// ============================================================================

FoXBlock::FoXBlock(size_t d_model, size_t num_heads, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_heads),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model)
{}

std::vector<Tensor*> FoXBlock::parameters() {
    auto p  = ln1_.parameters();
    auto a  = attn_.parameters();
    auto q  = ln2_.parameters();
    auto f1 = ffn_fc1_.parameters();
    auto f2 = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

std::vector<Tensor*> FoXBlock::gradients() {
    auto p  = ln1_.gradients();
    auto a  = attn_.gradients();
    auto q  = ln2_.gradients();
    auto f1 = ffn_fc1_.gradients();
    auto f2 = ffn_fc2_.gradients();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

void FoXBlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void FoXBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor FoXBlock::forward(const Tensor& input) {
    Tensor z1 = ln1_.forward(input);
    Tensor attn_out = attn_.forward(z1);
    Tensor res1 = z1 + attn_out;
    last_res1_ = res1;
    Tensor z2 = ln2_.forward(res1);
    Tensor hidden = ffn_fc1_.forward(z2);
    last_ffn_hidden_ = hidden.clone();   // PRE-GELU
    for (size_t i = 0; i < hidden.data.size(); ++i) {
        double x = hidden.data[i];
        hidden.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    Tensor ffn_out = ffn_fc2_.forward(hidden);
    return res1 + ffn_out;
}

Tensor FoXBlock::backward(const Tensor& grad_output, double lr) {
    Tensor d_h = ffn_fc2_.backward(grad_output, lr);
    Tensor d_pre(d_h.rows, d_h.cols);
    for (size_t i = 0; i < d_pre.data.size(); ++i) {
        double x = last_ffn_hidden_.data[i];
        double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        d_pre.data[i] = d_h.data[i] * (cdf + x * pdf);
    }
    Tensor d_z2 = ffn_fc1_.backward(d_pre, lr);
    Tensor d_res1_from_ln2 = ln2_.backward(d_z2, lr);
    Tensor d_res1(d_res1_from_ln2.rows, d_res1_from_ln2.cols);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] = grad_output.data[i] + d_res1_from_ln2.data[i];

    Tensor d_z1_from_attn = attn_.backward(d_res1, lr);
    Tensor d_z1(d_res1.rows, d_res1.cols);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] = d_res1.data[i] + d_z1_from_attn.data[i];

    Tensor d_input = ln1_.backward(d_z1, lr);
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// FoXModel
// ============================================================================

FoXModel::FoXModel(size_t input_dim, size_t d_model, size_t output_dim,
                   size_t num_blocks, size_t num_heads, size_t ffn_dim)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim)
{
    if (input_dim == 0 || d_model == 0 || output_dim == 0 || num_blocks == 0) {
        throw std::invalid_argument(
            "FoXModel: input_dim, d_model, output_dim, num_blocks must all be > 0");
    }
    W_in_  = Tensor::random(input_dim, d_model, 0.05);
    b_in_  = Tensor::random(1, d_model, 0.05);
    W_out_ = Tensor::random(d_model, output_dim, 0.05);
    b_out_ = Tensor::random(1, output_dim, 0.05);
    grad_W_in_  = Tensor::zeros(input_dim, d_model);
    grad_b_in_  = Tensor::zeros(1, d_model);
    grad_W_out_ = Tensor::zeros(d_model, output_dim);
    grad_b_out_ = Tensor::zeros(1, output_dim);
    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) blocks_.emplace_back(d_model, num_heads, ffn_dim);
}

std::vector<Tensor*> FoXModel::parameters() {
    std::vector<Tensor*> p{ &W_in_, &b_in_ };
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&W_out_);
    p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> FoXModel::gradients() {
    std::vector<Tensor*> g{ &grad_W_in_, &grad_b_in_ };
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_W_out_);
    g.push_back(&grad_b_out_);
    return g;
}

void FoXModel::zero_grad() {
    grad_W_in_.fill(0.0);
    grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void FoXModel::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_in_,  grad_W_in_);
    sgd(b_in_,  grad_b_in_);
    sgd(W_out_, grad_W_out_);
    sgd(b_out_, grad_b_out_);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor FoXModel::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_input_ = input.clone();

    Tensor h(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double v = b_in_(0, j);
            for (size_t c = 0; c < input_dim_; ++c) v += input(i, c) * W_in_(c, j);
            h(i, j) = v;
        }
    last_proj_ = h.clone();

    for (auto& b : blocks_) h = b.forward(h);
    last_block_out_ = h.clone();

    Tensor out(n, output_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < output_dim_; ++j) {
            double v = b_out_(0, j);
            for (size_t c = 0; c < d_model_; ++c) v += h(i, c) * W_out_(c, j);
            out(i, j) = v;
        }
    return out;
}

Tensor FoXModel::backward(const Tensor& grad_output, double lr) {
    const size_t n = grad_output.rows;

    // Output projection
    Tensor d_block_out(n, d_model_);
    d_block_out.fill(0.0);
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t j = 0; j < output_dim_; ++j) {
            double g = 0.0;
            for (size_t i = 0; i < n; ++i) g += last_block_out_(i, c) * grad_output(i, j);
            grad_W_out_(c, j) += g;
        }
    for (size_t j = 0; j < output_dim_; ++j) {
        double g = 0.0;
        for (size_t i = 0; i < n; ++i) g += grad_output(i, j);
        grad_b_out_(0, j) += g;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < d_model_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < output_dim_; ++j) v += grad_output(i, j) * W_out_(c, j);
            d_block_out(i, c) = v;
        }

    // Blocks, reverse order
    Tensor d = d_block_out;
    for (size_t b = blocks_.size(); b-- > 0; ) d = blocks_[b].backward(d, lr);

    // Input projection
    for (size_t c = 0; c < input_dim_; ++c)
        for (size_t j = 0; j < d_model_; ++j) {
            double g = 0.0;
            for (size_t i = 0; i < n; ++i) g += last_input_(i, c) * d(i, j);
            grad_W_in_(c, j) += g;
        }
    for (size_t j = 0; j < d_model_; ++j) {
        double g = 0.0;
        for (size_t i = 0; i < n; ++i) g += d(i, j);
        grad_b_in_(0, j) += g;
    }
    Tensor d_input(n, input_dim_);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < input_dim_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) v += d(i, j) * W_in_(c, j);
            d_input(i, c) = v;
        }
    return d_input;
}
