#include "timesnet.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <complex>

// ============================================================================
// Free helpers: FFT and top-K period selection
// ============================================================================

// Naive O(N^2) DFT. Input is a real 1-D signal; output is the full complex
// spectrum of length N.
std::vector<std::complex<double>> naive_dft(const std::vector<double>& x) {
    const int N = (int)x.size();
    std::vector<std::complex<double>> X(N);
    const double PI2 = 2.0 * M_PI;
    for (int k = 0; k < N; ++k) {
        std::complex<double> s(0.0, 0.0);
        for (int n = 0; n < N; ++n) {
            double theta = -PI2 * k * n / N;
            s += std::complex<double>(std::cos(theta), std::sin(theta)) * x[n];
        }
        X[k] = s;
    }
    return X;
}

// Top-K dominant periods from a full amplitude spectrum (length N).
// Scan k = 1..N/2 (skip DC at k=0). period = N/k.
std::vector<int> topk_periods(const std::vector<double>& amplitude, int K) {
    const int N = (int)amplitude.size();
    if (N < 4 || K <= 0) return {};
    std::vector<std::pair<int, double>> period_amp;
    for (int k = 1; k <= N / 2; ++k) {
        int period = N / k;
        if (period < 2) continue;
        period_amp.push_back({period, amplitude[k]});
    }
    std::sort(period_amp.begin(), period_amp.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<int> periods;
    for (int i = 0; i < std::min(K, (int)period_amp.size()); ++i) {
        periods.push_back(period_amp[i].first);
    }
    return periods;
}

// GELU helpers — same formulation as ConformerBlock.
static inline double gelu_f(double x) {
    const double c = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(c));
}
static inline double gelu_prime(double x) {
    const double c0 = std::sqrt(2.0 / M_PI);
    double cc = c0 * (x + 0.044715 * x * x * x);
    double th = std::tanh(cc);
    double sech2 = 1.0 - th * th;
    double dc_dx = c0 * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + th) + 0.5 * x * sech2 * dc_dx;
}

// ============================================================================
// Conv2DBlock — 2-layer Conv2D + GELU + 1×1 projection
// ============================================================================

Conv2DBlock::Conv2DBlock(int in_channels, int hidden_channels, int out_channels,
                         int H, int W)
    : in_channels_(in_channels), hidden_channels_(hidden_channels),
      out_channels_(out_channels), H_(H), W_(W),
      conv1_(in_channels, hidden_channels, 3, 3, H, W, 1, 1, 1, 1),
      conv2_(hidden_channels, hidden_channels, 3, 3, H, W, 1, 1, 1, 1),
      proj_(hidden_channels, out_channels, 1, 1, H, W, 1, 1, 0, 0) {}

Tensor Conv2DBlock::forward(const Tensor& x) {
    last_input_ = x.clone();
    Tensor h1 = conv1_.forward(x);
    last_h1_pre_gelu_ = h1.clone();
    for (size_t i = 0; i < h1.data.size(); ++i) h1.data[i] = gelu_f(h1.data[i]);
    Tensor h2 = conv2_.forward(h1);
    last_h2_pre_gelu_ = h2.clone();
    for (size_t i = 0; i < h2.data.size(); ++i) h2.data[i] = gelu_f(h2.data[i]);
    return proj_.forward(h2);
}

Tensor Conv2DBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor g_h2_pre = proj_.backward(grad_output, 0.0);
    // Backward through post-GELU → pre-GELU
    for (size_t i = 0; i < g_h2_pre.data.size(); ++i) {
        g_h2_pre.data[i] *= gelu_prime(last_h2_pre_gelu_.data[i]);
    }
    Tensor g_h1_post = conv2_.backward(g_h2_pre, 0.0);
    for (size_t i = 0; i < g_h1_post.data.size(); ++i) {
        g_h1_post.data[i] *= gelu_prime(last_h1_pre_gelu_.data[i]);
    }
    return conv1_.backward(g_h1_post, 0.0);
}

void Conv2DBlock::update_weights(double lr) {
    conv1_.update_weights(lr);
    conv2_.update_weights(lr);
    proj_.update_weights(lr);
}
void Conv2DBlock::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
    proj_.zero_grad();
}
std::vector<Tensor*> Conv2DBlock::parameters() {
    std::vector<Tensor*> all;
    auto p1 = conv1_.parameters();
    auto p2 = conv2_.parameters();
    auto pp = proj_.parameters();
    all.insert(all.end(), p1.begin(), p1.end());
    all.insert(all.end(), p2.begin(), p2.end());
    all.insert(all.end(), pp.begin(), pp.end());
    return all;
}
std::vector<Tensor*> Conv2DBlock::gradients() {
    std::vector<Tensor*> all;
    auto g1 = conv1_.gradients();
    auto g2 = conv2_.gradients();
    auto gp = proj_.gradients();
    all.insert(all.end(), g1.begin(), g1.end());
    all.insert(all.end(), g2.begin(), g2.end());
    all.insert(all.end(), gp.begin(), gp.end());
    return all;
}

// ============================================================================
// TimesBlock
// ============================================================================

TimesBlock::TimesBlock(int d_model, int seq_len, int top_k, int d_ff)
    : d_model_(d_model), seq_len_(seq_len), top_k_(top_k),
      d_ff_(d_ff > 0 ? d_ff : 4 * d_model),
      ln1_(d_model), ln2_(d_model),
      conv_block_(d_model, d_model, d_model, seq_len, seq_len),
      ffn1_(d_model, d_ff_), ffn2_(d_ff_, d_model) {}

Tensor TimesBlock::forward(const Tensor& input) {
    const int B = (int)input.rows / seq_len_;
    const int T = seq_len_;
    const int D = d_model_;
    const int K = top_k_;

    last_input_ = input.clone();

    // ---- 1. Period selection ----
    std::vector<int> periods;
    bool used_fixed = use_fixed_periods_ && !fixed_periods_.empty();
    if (used_fixed) {
        periods = fixed_periods_;
        while ((int)periods.size() < K) {
            periods.push_back(periods.back());
        }
    } else {
        // Sum of amplitudes across batch and feature dims.
        std::vector<double> amp_sum(T, 0.0);
        for (int b = 0; b < B; ++b) {
            std::vector<double> sig(T, 0.0);
            for (int d = 0; d < D; ++d) {
                for (int t = 0; t < T; ++t) {
                    sig[t] += input.data[(size_t)(b * T + t) * D + d];
                }
            }
            auto X = naive_dft(sig);
            for (int k = 0; k < T; ++k) amp_sum[k] += std::abs(X[k]);
        }
        periods = topk_periods(amp_sum, K);
        while ((int)periods.size() < K) {
            periods.push_back(T);
        }
    }
    last_periods_ = periods;
    last_used_fixed_ = used_fixed;

    // ---- 2. max_period, max_len ----
    int max_period = 0, max_len = 0;
    for (int p : periods) {
        int len = (T + p - 1) / p;
        if (p > max_period) max_period = p;
        if (len > max_len) max_len = len;
    }
    last_max_period_ = max_period;
    last_max_len_ = max_len;

    // ---- 3. Pre-norm (LN over per-token D features) ----
    // input is (B*T, D); ln1_.forward applies per-row LN — exactly right.
    Tensor ln1_out = ln1_.forward(input);
    last_ln1_out_ = ln1_out.clone();

    // ---- 4. Reshape (B*T, D) → stacked (B*K, D, max_period, max_len) ----
    //    Stacked buffer layout (N=B*K rows, each row has D*max_period*max_len
    //    elements in channel-major order):
    //      row = bk
    //      offset = bk * (D*max_period*max_len) + d * (max_period*max_len) + p * max_len + l
    //    Source: ln1_out[b, t_src, d]  with t_src = p * len + l (capped to T)
    Tensor stacked((size_t)(B * K), (size_t)(D * max_period * max_len));
    stacked.fill(0.0);
    for (int b = 0; b < B; ++b) {
        for (int ki = 0; ki < K; ++ki) {
            int p = periods[ki];
            int len = (T + p - 1) / p;
            int bk = b * K + ki;
            for (int pi = 0; pi < p; ++pi) {
                for (int li = 0; li < len; ++li) {
                    int t_src = pi * len + li;
                    if (t_src >= T) continue;
                    for (int d = 0; d < D; ++d) {
                        size_t src = (size_t)(b * T + t_src) * D + d;
                        size_t dst = (size_t)bk * D * max_period * max_len
                                   + (size_t)d * max_period * max_len
                                   + (size_t)pi * max_len + li;
                        stacked.data[dst] = ln1_out.data[src];
                    }
                }
            }
        }
    }

    // ---- 5. Pad to seq_len×seq_len (the conv block's spatial dims) ----
    Tensor padded_stacked((size_t)(B * K), (size_t)(D * T * T));
    padded_stacked.fill(0.0);
    for (int bk = 0; bk < B * K; ++bk) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < max_period; ++h) {
                for (int w = 0; w < max_len; ++w) {
                    size_t src = (size_t)bk * D * max_period * max_len
                               + (size_t)d * max_period * max_len
                               + (size_t)h * max_len + w;
                    size_t dst = (size_t)bk * D * T * T
                               + (size_t)d * T * T
                               + (size_t)h * T + w;
                    padded_stacked.data[dst] = stacked.data[src];
                }
            }
        }
    }

    // ---- 6. Conv2DBlock ----
    Tensor conv_padded = conv_block_.forward(padded_stacked);

    // ---- 7. Slice back to (B*K, D, max_period, max_len) ----
    Tensor conv_out((size_t)(B * K), (size_t)(D * max_period * max_len));
    conv_out.fill(0.0);
    for (int bk = 0; bk < B * K; ++bk) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < max_period; ++h) {
                for (int w = 0; w < max_len; ++w) {
                    size_t src = (size_t)bk * D * T * T
                               + (size_t)d * T * T
                               + (size_t)h * T + w;
                    size_t dst = (size_t)bk * D * max_period * max_len
                               + (size_t)d * max_period * max_len
                               + (size_t)h * max_len + w;
                    conv_out.data[dst] = conv_padded.data[src];
                }
            }
        }
    }

    // ---- 8. Average over K → (B*T, D) ----
    Tensor reduced((size_t)(B * T), (size_t)D);
    reduced.fill(0.0);
    for (int b = 0; b < B; ++b) {
        for (int ki = 0; ki < K; ++ki) {
            int p = periods[ki];
            int len = (T + p - 1) / p;
            int bk = b * K + ki;
            for (int pi = 0; pi < p; ++pi) {
                for (int li = 0; li < len; ++li) {
                    int t_src = pi * len + li;
                    if (t_src >= T) continue;
                    for (int d = 0; d < D; ++d) {
                        size_t src = (size_t)bk * D * max_period * max_len
                                   + (size_t)d * max_period * max_len
                                   + (size_t)pi * max_len + li;
                        size_t dst = (size_t)(b * T + t_src) * D + d;
                        reduced.data[dst] += conv_out.data[src] / (double)K;
                    }
                }
            }
        }
    }
    last_conv_out_ = reduced.clone();

    // ---- 9. Residual: after_conv_res = input + reduced ----
    Tensor after_conv_res((size_t)(B * T), (size_t)D);
    for (size_t i = 0; i < after_conv_res.data.size(); ++i) {
        after_conv_res.data[i] = input.data[i] + reduced.data[i];
    }

    // ---- 10. Pre-norm FFN ----
    Tensor ln2_out = ln2_.forward(after_conv_res);
    last_ln2_out_ = ln2_out.clone();
    Tensor ffn1_out = ffn1_.forward(ln2_out);
    last_ffn1_pre_gelu_ = ffn1_out.clone();
    for (size_t i = 0; i < ffn1_out.data.size(); ++i) {
        ffn1_out.data[i] = gelu_f(ffn1_out.data[i]);
    }
    Tensor ffn2_out = ffn2_.forward(ffn1_out);

    // ---- 11. Residual: output = after_conv_res + ffn2_out ----
    Tensor output((size_t)(B * T), (size_t)D);
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] = after_conv_res.data[i] + ffn2_out.data[i];
    }
    return output;
}

Tensor TimesBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const int B = (int)last_input_.rows / seq_len_;
    const int T = seq_len_;
    const int D = d_model_;
    const int K = top_k_;
    const int max_period = last_max_period_;
    const int max_len = last_max_len_;

    // ---- Backward step 1: residual2 (output = after_conv_res + ffn2_out) ----
    // Both paths sum. grad_after_conv_res = grad_output + grad_ffn2_path.
    Tensor grad_after_conv_res((size_t)(B * T), (size_t)D);
    for (size_t i = 0; i < grad_after_conv_res.data.size(); ++i) {
        grad_after_conv_res.data[i] = grad_output.data[i];
    }

    // ffn2 path: input was ffn1_post (post-GELU), output was ffn2_out.
    Tensor g_ffn2_in = ffn2_.backward(grad_output, 0.0);  // (B*T, d_ff)
    for (size_t i = 0; i < g_ffn2_in.data.size(); ++i) {
        g_ffn2_in.data[i] *= gelu_prime(last_ffn1_pre_gelu_.data[i]);
    }
    // g_ffn2_in is now the grad at ffn1 pre-GELU input. Backward through ffn1:
    Tensor g_ln2_out = ffn1_.backward(g_ffn2_in, 0.0);  // (B*T, d_model)
    // Backward through ln2:
    Tensor g_after_conv_res_from_ffn = ln2_.backward(g_ln2_out, 0.0);
    for (size_t i = 0; i < grad_after_conv_res.data.size(); ++i) {
        grad_after_conv_res.data[i] += g_after_conv_res_from_ffn.data[i];
    }

    // ---- Backward step 2: residual1 (after_conv_res = input + reduced) ----
    // grad at reduced = grad_after_conv_res; grad at input (skip) = grad_after_conv_res.
    Tensor grad_reduced = grad_after_conv_res.clone();
    Tensor grad_input_from_skip = grad_after_conv_res.clone();

    // ---- Backward step 3: average backward → grad at conv_out slices ----
    // grad_conv_out[bk][d, p, l] = grad_reduced[b, t_src, d] / K  (same indexing as forward).
    Tensor grad_conv_out((size_t)(B * K), (size_t)(D * max_period * max_len));
    grad_conv_out.fill(0.0);
    for (int b = 0; b < B; ++b) {
        for (int ki = 0; ki < K; ++ki) {
            int p = last_periods_[ki];
            int len = (T + p - 1) / p;
            int bk = b * K + ki;
            for (int pi = 0; pi < p; ++pi) {
                for (int li = 0; li < len; ++li) {
                    int t_src = pi * len + li;
                    if (t_src >= T) continue;
                    for (int d = 0; d < D; ++d) {
                        size_t src = (size_t)(b * T + t_src) * D + d;
                        size_t dst = (size_t)bk * D * max_period * max_len
                                   + (size_t)d * max_period * max_len
                                   + (size_t)pi * max_len + li;
                        grad_conv_out.data[dst] = grad_reduced.data[src] / (double)K;
                    }
                }
            }
        }
    }

    // ---- Backward step 4: pad grad to seq_len×seq_len (zero pad) ----
    Tensor grad_padded((size_t)(B * K), (size_t)(D * T * T));
    grad_padded.fill(0.0);
    for (int bk = 0; bk < B * K; ++bk) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < max_period; ++h) {
                for (int w = 0; w < max_len; ++w) {
                    size_t src = (size_t)bk * D * max_period * max_len
                               + (size_t)d * max_period * max_len
                               + (size_t)h * max_len + w;
                    size_t dst = (size_t)bk * D * T * T
                               + (size_t)d * T * T
                               + (size_t)h * T + w;
                    grad_padded.data[dst] = grad_conv_out.data[src];
                }
            }
        }
    }
    Tensor g_padded_stacked = conv_block_.backward(grad_padded, 0.0);

    // ---- Backward step 5: slice back to (B*K, D, max_period, max_len) ----
    Tensor g_stacked((size_t)(B * K), (size_t)(D * max_period * max_len));
    g_stacked.fill(0.0);
    for (int bk = 0; bk < B * K; ++bk) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < max_period; ++h) {
                for (int w = 0; w < max_len; ++w) {
                    size_t src = (size_t)bk * D * T * T
                               + (size_t)d * T * T
                               + (size_t)h * T + w;
                    size_t dst = (size_t)bk * D * max_period * max_len
                               + (size_t)d * max_period * max_len
                               + (size_t)h * max_len + w;
                    g_stacked.data[dst] = g_padded_stacked.data[src];
                }
            }
        }
    }

    // ---- Backward step 6: backward through reshape (scatter-add into (B*T, D)) ----
    Tensor g_ln1_out((size_t)(B * T), (size_t)D);
    g_ln1_out.fill(0.0);
    for (int b = 0; b < B; ++b) {
        for (int ki = 0; ki < K; ++ki) {
            int p = last_periods_[ki];
            int len = (T + p - 1) / p;
            int bk = b * K + ki;
            for (int pi = 0; pi < p; ++pi) {
                for (int li = 0; li < len; ++li) {
                    int t_src = pi * len + li;
                    if (t_src >= T) continue;
                    for (int d = 0; d < D; ++d) {
                        size_t src = (size_t)bk * D * max_period * max_len
                                   + (size_t)d * max_period * max_len
                                   + (size_t)pi * max_len + li;
                        size_t dst = (size_t)(b * T + t_src) * D + d;
                        g_ln1_out.data[dst] += g_stacked.data[src];
                    }
                }
            }
        }
    }

    // ---- Backward step 7: backward through ln1 ----
    Tensor g_input_from_conv = ln1_.backward(g_ln1_out, 0.0);

    // ---- Combine with skip connection ----
    Tensor grad_input((size_t)(B * T), (size_t)D);
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        grad_input.data[i] = grad_input_from_skip.data[i] + g_input_from_conv.data[i];
    }
    return grad_input;
}

void TimesBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    conv_block_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn1_.update_weights(lr);
    ffn2_.update_weights(lr);
}
void TimesBlock::zero_grad() {
    ln1_.zero_grad();
    conv_block_.zero_grad();
    ln2_.zero_grad();
    ffn1_.zero_grad();
    ffn2_.zero_grad();
}
std::vector<Tensor*> TimesBlock::parameters() {
    std::vector<Tensor*> p;
    auto a = ln1_.parameters();
    auto b = conv_block_.parameters();
    auto c = ln2_.parameters();
    auto d = ffn1_.parameters();
    auto e = ffn2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    p.insert(p.end(), e.begin(), e.end());
    return p;
}
std::vector<Tensor*> TimesBlock::gradients() {
    std::vector<Tensor*> g;
    auto a = ln1_.gradients();
    auto b = conv_block_.gradients();
    auto c = ln2_.gradients();
    auto d = ffn1_.gradients();
    auto e = ffn2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    g.insert(g.end(), e.begin(), e.end());
    return g;
}

// ============================================================================
// TimesNet
// ============================================================================

TimesNet::TimesNet(int in_dim, int out_dim, int seq_len, int pred_len,
                   int d_model, int e_layers, int top_k, int d_ff)
    : in_dim_(in_dim), out_dim_(out_dim), d_model_(d_model),
      seq_len_(seq_len), pred_len_(pred_len), e_layers_(e_layers),
      top_k_(top_k),
      embed_(in_dim, d_model),
      final_ln_(d_model),
      projector_(d_model, out_dim) {
    for (int i = 0; i < e_layers_; ++i) {
        blocks_.emplace_back(std::make_unique<TimesBlock>(
            d_model, seq_len, top_k, d_ff > 0 ? d_ff : 4 * d_model));
    }
}

Tensor TimesNet::forward(const Tensor& input) {
    const int B = (int)input.rows / seq_len_;
    last_input_ = input.clone();
    Tensor h = embed_.forward(input);  // (B*T, d_model)
    last_embed_out_ = h.clone();
    for (auto& blk : blocks_) h = blk->forward(h);
    h = final_ln_.forward(h);  // (B*T, d_model)

    // Slice last pred_len tokens per batch.
    int start_tok = seq_len_ - pred_len_;
    Tensor last_tokens((size_t)(B * pred_len_), (size_t)d_model_);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < pred_len_; ++t) {
            int src = b * seq_len_ + start_tok + t;
            for (int j = 0; j < d_model_; ++j) {
                last_tokens.data[(size_t)(b * pred_len_ + t) * d_model_ + j] =
                    h.data[(size_t)src * d_model_ + j];
            }
        }
    }
    Tensor proj = projector_.forward(last_tokens);  // (B*pred_len, out_dim)
    // Reshape to (B, pred_len*out_dim)
    Tensor output((size_t)B, (size_t)(pred_len_ * out_dim_));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < pred_len_; ++t) {
            for (int j = 0; j < out_dim_; ++j) {
                output.data[(size_t)b * pred_len_ * out_dim_ + (size_t)t * out_dim_ + j] =
                    proj.data[(size_t)(b * pred_len_ + t) * out_dim_ + j];
            }
        }
    }
    return output;
}

Tensor TimesNet::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const int B = (int)last_input_.rows / seq_len_;

    // 1. Unflatten grad_output (B, pred_len*out_dim) → (B*pred_len, out_dim)
    Tensor grad_proj((size_t)(B * pred_len_), (size_t)out_dim_);
    grad_proj.fill(0.0);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < pred_len_; ++t) {
            for (int j = 0; j < out_dim_; ++j) {
                grad_proj.data[(size_t)(b * pred_len_ + t) * out_dim_ + j] =
                    grad_output.data[(size_t)b * pred_len_ * out_dim_ + (size_t)t * out_dim_ + j];
            }
        }
    }
    Tensor grad_last_tokens = projector_.backward(grad_proj, 0.0);

    // 2. Inject grad_last_tokens into grad at h positions (last pred_len tokens per batch)
    Tensor grad_h((size_t)(B * seq_len_), (size_t)d_model_);
    grad_h.fill(0.0);
    int start_tok = seq_len_ - pred_len_;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < pred_len_; ++t) {
            int dst = b * seq_len_ + start_tok + t;
            for (int j = 0; j < d_model_; ++j) {
                grad_h.data[(size_t)dst * d_model_ + j] +=
                    grad_last_tokens.data[(size_t)(b * pred_len_ + t) * d_model_ + j];
            }
        }
    }

    // 3. Backward through final_ln (input was (B*T, D))
    Tensor grad_post_blocks = final_ln_.backward(grad_h, 0.0);

    // 4. Backward through each block in reverse
    Tensor grad_blocks_out = grad_post_blocks;
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        grad_blocks_out = blocks_[i]->backward(grad_blocks_out, 0.0);
    }

    // 5. Backward through embed — Dense backward expects (N, d_model)
    return embed_.backward(grad_blocks_out, 0.0);
}

void TimesNet::update_weights(double lr) {
    for (auto& blk : blocks_) blk->update_weights(lr);
    final_ln_.update_weights(lr);
    projector_.update_weights(lr);
    embed_.update_weights(lr);
}
void TimesNet::zero_grad() {
    for (auto& blk : blocks_) blk->zero_grad();
    final_ln_.zero_grad();
    projector_.zero_grad();
    embed_.zero_grad();
}
std::vector<Tensor*> TimesNet::parameters() {
    std::vector<Tensor*> p;
    auto ep = embed_.parameters();
    p.insert(p.end(), ep.begin(), ep.end());
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto flp = final_ln_.parameters();
    p.insert(p.end(), flp.begin(), flp.end());
    auto pp = projector_.parameters();
    p.insert(p.end(), pp.begin(), pp.end());
    return p;
}
std::vector<Tensor*> TimesNet::gradients() {
    std::vector<Tensor*> g;
    auto eg = embed_.gradients();
    g.insert(g.end(), eg.begin(), eg.end());
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto flg = final_ln_.gradients();
    g.insert(g.end(), flg.begin(), flg.end());
    auto pg = projector_.gradients();
    g.insert(g.end(), pg.begin(), pg.end());
    return g;
}
