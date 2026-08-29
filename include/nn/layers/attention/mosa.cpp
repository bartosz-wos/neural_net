// MoSA (Mixture of Sparse Attention) — implementation
//   Piękos, Csordás, Schmidhuber, May 2025
//   https://arxiv.org/abs/2505.00315
//
// See mosa.h for the full design write-up and the backward derivation.
#include "mosa.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

// ============================================================================
// MoSAAttention
// ============================================================================

MoSAAttention::MoSAAttention(size_t d_model, size_t num_heads, size_t top_k)
    : d_model_(d_model), num_heads_(num_heads),
      head_dim_(0),     // placeholder; set after validation (avoids SIGFPE on 1/0)
      top_k_(top_k),
      scale_(1.0)
{
    if (d_model_ == 0 || num_heads_ == 0) {
        throw std::invalid_argument("MoSAAttention: d_model and num_heads must both be > 0");
    }
    if (top_k_ == 0) {
        throw std::invalid_argument("MoSAAttention: top_k must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("MoSAAttention: d_model must be divisible by num_heads");
    }

    head_dim_ = d_model_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_) + 1e-9);

    // Per-head parameters: W_q[h], W_k[h], W_v[h] : (d, head_dim)
    //                     W_o[h] : (head_dim, d)
    //                     W_r[h] : (d, 1)
    W_q.resize(num_heads_);
    W_k.resize(num_heads_);
    W_v.resize(num_heads_);
    W_o.resize(num_heads_);
    W_r.resize(num_heads_);
    grad_W_q.resize(num_heads_);
    grad_W_k.resize(num_heads_);
    grad_W_v.resize(num_heads_);
    grad_W_o.resize(num_heads_);
    grad_W_r.resize(num_heads_);

    for (size_t h = 0; h < num_heads_; ++h) {
        W_q[h] = Tensor::random(d_model_, head_dim_, 0.05);
        W_k[h] = Tensor::random(d_model_, head_dim_, 0.05);
        W_v[h] = Tensor::random(d_model_, head_dim_, 0.05);
        W_o[h] = Tensor::random(head_dim_, d_model_, 0.05);
        W_r[h] = Tensor::random(d_model_, 1, 0.05);

        grad_W_q[h] = Tensor::zeros(d_model_, head_dim_);
        grad_W_k[h] = Tensor::zeros(d_model_, head_dim_);
        grad_W_v[h] = Tensor::zeros(d_model_, head_dim_);
        grad_W_o[h] = Tensor::zeros(head_dim_, d_model_);
        grad_W_r[h] = Tensor::zeros(d_model_, 1);
    }
}

std::vector<Tensor*> MoSAAttention::parameters() {
    std::vector<Tensor*> p;
    p.reserve(num_heads_ * 5);
    for (size_t h = 0; h < num_heads_; ++h) {
        p.push_back(&W_q[h]);
        p.push_back(&W_k[h]);
        p.push_back(&W_v[h]);
        p.push_back(&W_o[h]);
        p.push_back(&W_r[h]);
    }
    return p;
}

std::vector<Tensor*> MoSAAttention::gradients() {
    std::vector<Tensor*> g;
    g.reserve(num_heads_ * 5);
    for (size_t h = 0; h < num_heads_; ++h) {
        g.push_back(&grad_W_q[h]);
        g.push_back(&grad_W_k[h]);
        g.push_back(&grad_W_v[h]);
        g.push_back(&grad_W_o[h]);
        g.push_back(&grad_W_r[h]);
    }
    return g;
}

void MoSAAttention::zero_grad() {
    for (size_t h = 0; h < num_heads_; ++h) {
        grad_W_q[h].fill(0.0);
        grad_W_k[h].fill(0.0);
        grad_W_v[h].fill(0.0);
        grad_W_o[h].fill(0.0);
        grad_W_r[h].fill(0.0);
    }
}

void MoSAAttention::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    for (size_t h = 0; h < num_heads_; ++h) {
        sgd(W_q[h], grad_W_q[h]);
        sgd(W_k[h], grad_W_k[h]);
        sgd(W_v[h], grad_W_v[h]);
        sgd(W_o[h], grad_W_o[h]);
        sgd(W_r[h], grad_W_r[h]);
    }
}

// Top-k indices + values (descending by value), for a 1-D vector.
static void topk_descending(const std::vector<double>& v, size_t k,
                            std::vector<size_t>& idx, std::vector<double>& vals) {
    const size_t n = v.size();
    k = std::min(k, n);
    // Create (value, index) pairs
    std::vector<std::pair<double, size_t>> pairs(n);
    for (size_t i = 0; i < n; ++i) pairs[i] = {v[i], i};
    // Partial sort: top-k by descending value
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                          return a.first > b.first;
                      });
    idx.resize(k);
    vals.resize(k);
    for (size_t i = 0; i < k; ++i) {
        idx[i] = pairs[i].second;
        vals[i] = pairs[i].first;
    }
}

Tensor MoSAAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    last_input_ = input.clone();
    const size_t k = std::min(top_k_, n);

    // ---- Step 1: per-head router scores r_h = sigmoid(X @ W_r^h) ----
    Tensor r(n, num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        for (size_t t = 0; t < n; ++t) {
            double z = 0.0;
            for (size_t c = 0; c < d_model_; ++c) z += input(t, c) * W_r[h](c, 0);
            r(t, h) = 1.0 / (1.0 + std::exp(-z));
        }
    }
    last_r_ = r;

    // ---- Step 2: per-head top-k selection and gather ----
    last_I_.assign(num_heads_, std::vector<size_t>(k));
    last_r_topk_.assign(num_heads_, std::vector<double>(k, 0.0));
    last_X_s_.assign(num_heads_, Tensor(k, d_model_));
    for (size_t h = 0; h < num_heads_; ++h) {
        std::vector<double> r_col(n);
        for (size_t t = 0; t < n; ++t) r_col[t] = r(t, h);
        topk_descending(r_col, k, last_I_[h], last_r_topk_[h]);
        for (size_t i = 0; i < k; ++i) {
            const size_t src = last_I_[h][i];
            for (size_t c = 0; c < d_model_; ++c) {
                last_X_s_[h](i, c) = input(src, c);
            }
        }
    }

    // ---- Step 3: per-head Q/K/V projections, causal attention, output ----
    last_Q_.assign(num_heads_, Tensor(k, head_dim_));
    last_K_.assign(num_heads_, Tensor(k, head_dim_));
    last_V_.assign(num_heads_, Tensor(k, head_dim_));
    last_A_.assign(num_heads_, Tensor(k, k));
    last_A_scaled_.assign(num_heads_, Tensor(k, k));
    last_head_pre_o_.assign(num_heads_, Tensor(k, head_dim_));

    Tensor head_out(n, d_model_);
    head_out.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Project
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double q = 0.0, kk = 0.0, v = 0.0;
                for (size_t c = 0; c < d_model_; ++c) {
                    double x = last_X_s_[h](i, c);
                    q += x * W_q[h](c, j);
                    kk += x * W_k[h](c, j);
                    v += x * W_v[h](c, j);
                }
                last_Q_[h](i, j) = q;
                last_K_[h](i, j) = kk;
                last_V_[h](i, j) = v;
            }
        }

        // Causal attention with mask based on original positions
        for (size_t i = 0; i < k; ++i) {
            std::vector<double> sc(k, -std::numeric_limits<double>::infinity());
            for (size_t j = 0; j < k; ++j) {
                if (last_I_[h][i] >= last_I_[h][j]) {
                    double dot = 0.0;
                    for (size_t dd = 0; dd < head_dim_; ++dd)
                        dot += last_Q_[h](i, dd) * last_K_[h](j, dd);
                    sc[j] = dot * scale_;
                }
            }
            double m = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < k; ++j) m = std::max(m, sc[j]);
            // If m is -inf, all entries were -inf: leave row as uniform (1/k) to avoid NaN
            if (std::isinf(m) && m < 0) {
                double u = 1.0 / static_cast<double>(k);
                for (size_t j = 0; j < k; ++j) last_A_[h](i, j) = u;
            } else {
                double l = 0.0;
                for (size_t j = 0; j < k; ++j) { sc[j] = std::exp(sc[j] - m); l += sc[j]; }
                const double inv_l = 1.0 / l;
                for (size_t j = 0; j < k; ++j) {
                    sc[j] *= inv_l;
                    last_A_[h](i, j) = sc[j];
                }
            }
        }

        // Row-scale by router scores
        for (size_t i = 0; i < k; ++i) {
            const double r_topk_i = last_r_topk_[h][i];
            for (size_t j = 0; j < k; ++j) {
                last_A_scaled_[h](i, j) = r_topk_i * last_A_[h](i, j);
            }
        }

        // head_pre_o = A_scaled @ V
        for (size_t i = 0; i < k; ++i) {
            for (size_t dd = 0; dd < head_dim_; ++dd) {
                double v = 0.0;
                for (size_t j = 0; j < k; ++j) v += last_A_scaled_[h](i, j) * last_V_[h](j, dd);
                last_head_pre_o_[h](i, dd) = v;
            }
        }

        // Per-head W_o projection AND scatter-add to head_out
        for (size_t i = 0; i < k; ++i) {
            const size_t dst = last_I_[h][i];
            for (size_t j = 0; j < d_model_; ++j) {
                double v = 0.0;
                for (size_t c = 0; c < head_dim_; ++c) v += last_head_pre_o_[h](i, c) * W_o[h](c, j);
                head_out(dst, j) += v;
            }
        }
    }
    last_head_out_ = head_out;

    return head_out;
}

Tensor MoSAAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    const size_t k = std::min(top_k_, n);

    Tensor d_input(n, d_model_);
    d_input.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        // Gather d_head_out for this head's selected positions
        Tensor d_head_pre_o(k, head_dim_);  // dL/d(A_scaled @ V)
        d_head_pre_o.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            const size_t dst = last_I_[h][i];
            for (size_t dd = 0; dd < head_dim_; ++dd) {
                double v = 0.0;
                for (size_t j = 0; j < d_model_; ++j) v += grad_output(dst, j) * W_o[h](dd, j);
                d_head_pre_o(i, dd) = v;
            }
        }

        // dL/dW_o[h] += last_head_pre_o[h]^T @ d_head_out
        // where d_head_out[i, j] = grad_output[last_I_[h][i], j]  (shape (k, d_model))
        for (size_t c = 0; c < head_dim_; ++c) {
            for (size_t j = 0; j < d_model_; ++j) {
                double g = 0.0;
                for (size_t i = 0; i < k; ++i) g += last_head_pre_o_[h](i, c) * grad_output(last_I_[h][i], j);
                grad_W_o[h](c, j) += g;
            }
        }

        // dL/dA_scaled (k, k): from d_head_pre_o = A_scaled @ V => dA_scaled = d_head_pre_o @ V^T
        Tensor dA_scaled(k, k);
        dA_scaled.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double v = 0.0;
                for (size_t dd = 0; dd < head_dim_; ++dd) v += d_head_pre_o(i, dd) * last_V_[h](j, dd);
                dA_scaled(i, j) = v;
            }
        }
        // dL/dV: A_scaled^T @ d_head_pre_o
        Tensor dV(k, head_dim_);
        dV.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            for (size_t dd = 0; dd < head_dim_; ++dd) {
                double v = 0.0;
                for (size_t j = 0; j < k; ++j) v += last_A_scaled_[h](j, i) * d_head_pre_o(j, dd);
                dV(i, dd) = v;
            }
        }

        // dL/dA = diag(r_topk) @ dA_scaled  (each row scaled by r_topk)
        Tensor dA(k, k);
        dA.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            const double r_topk_i = last_r_topk_[h][i];
            for (size_t j = 0; j < k; ++j) {
                dA(i, j) = r_topk_i * dA_scaled(i, j);
            }
        }

        // dL/dr_topk[i] = sum_j dA_scaled(i, j) * A(i, j)
        std::vector<double> d_r_topk(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < k; ++j) s += dA_scaled(i, j) * last_A_[h](i, j);
            d_r_topk[i] = s;
        }

        // Softmax backward: dL/dS_ij = A_ij * (dA_ij - sum_l A_il * dA_il)
        Tensor dS(k, k);
        dS.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            double row_dot = 0.0;
            for (size_t j = 0; j < k; ++j) {
                if (last_I_[h][i] >= last_I_[h][j]) {
                    row_dot += last_A_[h](i, j) * dA(i, j);
                }
            }
            for (size_t j = 0; j < k; ++j) {
                if (last_I_[h][i] >= last_I_[h][j]) {
                    dS(i, j) = last_A_[h](i, j) * (dA(i, j) - row_dot);
                }
                // else: dS = 0 (mask)
            }
        }

        // dL/dQ = dS @ K * scale; dL/dK = dS^T @ Q * scale
        Tensor dQ(k, head_dim_), dK(k, head_dim_);
        dQ.fill(0.0); dK.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            for (size_t dd = 0; dd < head_dim_; ++dd) {
                double vq = 0.0, vk = 0.0;
                for (size_t j = 0; j < k; ++j) {
                    vq += dS(i, j) * last_K_[h](j, dd);
                    vk += dS(j, i) * last_Q_[h](j, dd);
                }
                dQ(i, dd) = vq * scale_;
                dK(i, dd) = vk * scale_;
            }
        }

        // dL/dW_q[h] += X_s^T @ dQ; dL/dW_k[h] += X_s^T @ dK; dL/dW_v[h] += X_s^T @ dV
        for (size_t c = 0; c < d_model_; ++c) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double gq = 0.0, gk = 0.0, gv = 0.0;
                for (size_t i = 0; i < k; ++i) {
                    gq += last_X_s_[h](i, c) * dQ(i, j);
                    gk += last_X_s_[h](i, c) * dK(i, j);
                    gv += last_X_s_[h](i, c) * dV(i, j);
                }
                grad_W_q[h](c, j) += gq;
                grad_W_k[h](c, j) += gk;
                grad_W_v[h](c, j) += gv;
            }
        }

        // dL/dX_s = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T
        Tensor d_X_s(k, d_model_);
        d_X_s.fill(0.0);
        for (size_t i = 0; i < k; ++i) {
            for (size_t c = 0; c < d_model_; ++c) {
                double v = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    v += dQ(i, j) * W_q[h](c, j);
                    v += dK(i, j) * W_k[h](c, j);
                    v += dV(i, j) * W_v[h](c, j);
                }
                d_X_s(i, c) = v;
            }
        }

        // Scatter d_X_s into d_input at positions I_h[i]
        for (size_t i = 0; i < k; ++i) {
            const size_t dst = last_I_[h][i];
            for (size_t c = 0; c < d_model_; ++c) {
                d_input(dst, c) += d_X_s(i, c);
            }
        }

        // Router gradient: dL/dW_r[h][c] += sum_t d_r[t] * r[t, h] * (1 - r[t, h]) * X[t, c]
        // where d_r[t] = d_r_topk[i] if t = I_h[i], else 0.
        std::vector<double> d_r(n, 0.0);
        for (size_t i = 0; i < k; ++i) d_r[last_I_[h][i]] = d_r_topk[i];
        for (size_t t = 0; t < n; ++t) {
            const double drt = d_r[t];
            if (drt == 0.0) continue;
            const double rth = last_r_(t, h);
            const double sig_deriv = rth * (1.0 - rth);
            for (size_t c = 0; c < d_model_; ++c) {
                grad_W_r[h](c, 0) += drt * sig_deriv * last_input_(t, c);
                // dL/dx[t, c] via the r_topk -> x path: r_topk[i] = sigmoid(X[I[i], :] @ W_r)
                // so dr_topk[i]/dx[t, c] = 1[t == I[i]] * W_r[c, 0] * r[t, h] * (1 - r[t, h])
                d_input(t, c) += drt * sig_deriv * W_r[h](c, 0);
            }
        }
    }
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// MoSABlock
// ============================================================================

MoSABlock::MoSABlock(size_t d_model, size_t num_heads, size_t top_k, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? d_model * 4 : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_heads, top_k),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{}

std::vector<Tensor*> MoSABlock::parameters() {
    auto p = ln1_.parameters();
    auto a = attn_.parameters();
    auto p2 = ln2_.parameters();
    auto fc1 = ffn_fc1_.parameters();
    auto fc2 = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), fc1.begin(), fc1.end());
    p.insert(p.end(), fc2.begin(), fc2.end());
    return p;
}

std::vector<Tensor*> MoSABlock::gradients() {
    auto g = ln1_.gradients();
    auto a = attn_.gradients();
    auto g2 = ln2_.gradients();
    auto fc1 = ffn_fc1_.gradients();
    auto fc2 = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), fc1.begin(), fc1.end());
    g.insert(g.end(), fc2.begin(), fc2.end());
    return g;
}

void MoSABlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void MoSABlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

Tensor MoSABlock::forward(const Tensor& input) {
    Tensor normed = ln1_.forward(input);
    Tensor attn_out = attn_.forward(normed);
    Tensor res1 = input + attn_out;
    last_res1_ = res1.clone();
    Tensor normed2 = ln2_.forward(res1);
    Tensor ffn_h = ffn_fc1_.forward(normed2);
    last_ffn_hidden_ = ffn_h.clone();
    // GELU
    for (size_t i = 0; i < ffn_h.data.size(); ++i) {
        double x = ffn_h.data[i];
        // Tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double gelu = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        ffn_h.data[i] = gelu;
    }
    Tensor ffn_o = ffn_fc2_.forward(ffn_h);
    Tensor out = res1 + ffn_o;
    return out;
}

Tensor MoSABlock::backward(const Tensor& grad_output, double learning_rate) {
    // Residual backward: d_res1 = grad_output, d_ffn2_out = grad_output
    Tensor d_res1 = grad_output.clone();
    Tensor d_ffn2_out = grad_output.clone();
    // ffn2 backward
    Tensor d_ffn_h = ffn_fc2_.backward(d_ffn2_out, learning_rate);
    // GELU backward: d_x = d_y * (1 + erf(x/sqrt(2))) approximation
    // For the tanh-approx: derivative w.r.t. x is complicated; use the standard
    // GELU derivative: 0.5*(1 + erf(x/sqrt(2))) + x * exp(-x^2/2)/sqrt(2*pi)
    for (size_t i = 0; i < d_ffn_h.data.size(); ++i) {
        double x = last_ffn_hidden_.data[i];
        double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        double gelu_d = cdf + x * pdf;
        d_ffn_h.data[i] *= gelu_d;
    }
    // ffn1 backward
    Tensor d_normed2 = ffn_fc1_.backward(d_ffn_h, learning_rate);
    // ln2 backward
    Tensor d_res1_from_ln2 = ln2_.backward(d_normed2, learning_rate);
    Tensor d_res1_combined = d_res1 + d_res1_from_ln2;
    // attn backward
    Tensor d_normed = attn_.backward(d_res1_combined, learning_rate);
    // Two contributions to d_input from res1 = input + attn_out:
    //   1) d_res1/d_input = I (residual bypass)
    //   2) d_normed -> input via ln1
    Tensor d_input_via_ln1 = ln1_.backward(d_normed, learning_rate);
    Tensor d_input = d_res1_combined + d_input_via_ln1;
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// MoSAModel
// ============================================================================

MoSAModel::MoSAModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_blocks, size_t num_heads, size_t top_k, size_t ffn_dim)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim)
{
    W_in_ = Tensor::random(d_model_, input_dim_, 0.1);
    b_in_ = Tensor::zeros(1, d_model_);
    W_out_ = Tensor::random(output_dim_, d_model_, 0.1);
    b_out_ = Tensor::zeros(1, output_dim_);
    grad_W_in_ = Tensor::zeros(d_model_, input_dim_);
    grad_b_in_ = Tensor::zeros(1, d_model_);
    grad_W_out_ = Tensor::zeros(output_dim_, d_model_);
    grad_b_out_ = Tensor::zeros(1, output_dim_);

    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, num_heads, top_k, ffn_dim);
    }
}

std::vector<Tensor*> MoSAModel::parameters() {
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

std::vector<Tensor*> MoSAModel::gradients() {
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

void MoSAModel::zero_grad() {
    grad_W_in_.fill(0.0);
    grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void MoSAModel::update_weights(double lr) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) w.data[i] -= lr * g.data[i];
    };
    sgd(W_in_, grad_W_in_);
    sgd(b_in_, grad_b_in_);
    sgd(W_out_, grad_W_out_);
    sgd(b_out_, grad_b_out_);
    for (auto& b : blocks_) b.update_weights(lr);
}

Tensor MoSAModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Input projection: (n, input_dim) -> (n, d_model)
    const size_t n = input.rows;
    Tensor proj(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = b_in_(0, j);
            for (size_t c = 0; c < input_dim_; ++c) v += input(i, c) * W_in_(j, c);
            proj(i, j) = v;
        }
    }
    last_proj_ = proj.clone();

    Tensor x = proj;
    for (auto& b : blocks_) {
        x = b.forward(x);
    }
    last_block_out_ = x.clone();

    // Output projection: (n, d_model) -> (n, output_dim)
    Tensor out(n, output_dim_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double v = b_out_(0, j);
            for (size_t c = 0; c < d_model_; ++c) v += x(i, c) * W_out_(j, c);
            out(i, j) = v;
        }
    }
    return out;
}

Tensor MoSAModel::backward(const Tensor& grad_output, double learning_rate) {
    const size_t n = grad_output.rows;
    // d_block_out
    Tensor d_block_out(n, d_model_);
    d_block_out.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < d_model_; ++c) {
            double v = 0.0;
            for (size_t j = 0; j < output_dim_; ++j) v += grad_output(i, j) * W_out_(j, c);
            d_block_out(i, c) = v;
        }
    }
    // dW_out, db_out
    for (size_t j = 0; j < output_dim_; ++j) {
        for (size_t c = 0; c < d_model_; ++c) {
            double g = 0.0;
            for (size_t i = 0; i < n; ++i) g += last_block_out_(i, c) * grad_output(i, j);
            grad_W_out_(j, c) += g;
        }
    }
    for (size_t j = 0; j < output_dim_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += grad_output(i, j);
        grad_b_out_(0, j) += s;
    }

    // Backprop through blocks
    Tensor d_x = d_block_out;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_x = it->backward(d_x, learning_rate);
    }

    // dW_in, db_in
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t c = 0; c < input_dim_; ++c) {
            double g = 0.0;
            for (size_t i = 0; i < n; ++i) g += last_input_(i, c) * d_x(i, j);
            grad_W_in_(j, c) += g;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += d_x(i, j);
        grad_b_in_(0, j) += s;
    }
    return d_x;
}
