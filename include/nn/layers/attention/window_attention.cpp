#include "window_attention.h"
#include <random>
#include <stdexcept>

WindowAttention::WindowAttention(size_t d_model, size_t num_heads,
                                  size_t window_size, size_t H_pad, size_t W_pad,
                                  const std::string& bias_type)
    : d_model_(d_model), num_heads_(num_heads),
      window_size_(window_size), H_pad_(H_pad), W_pad_(W_pad),
      bias_type_(bias_type)
{
    if (d_model_ == 0)
        throw std::invalid_argument("WindowAttention: d_model must be > 0");
    if (num_heads_ == 0)
        throw std::invalid_argument("WindowAttention: num_heads must be > 0");
    if (d_model_ % num_heads_ != 0)
        throw std::invalid_argument("WindowAttention: d_model must be divisible by num_heads");
    if (window_size_ == 0)
        throw std::invalid_argument("WindowAttention: window_size must be > 0");
    if (H_pad_ % window_size_ != 0)
        throw std::invalid_argument("WindowAttention: H_pad must be divisible by window_size");
    if (W_pad_ % window_size_ != 0)
        throw std::invalid_argument("WindowAttention: W_pad must be divisible by window_size");
    if (bias_type_ != "relative" && bias_type_ != "none")
        throw std::invalid_argument("WindowAttention: bias_type must be 'relative' or 'none'");

    head_dim_ = d_model_ / num_heads_;
    M_ = window_size_;
    M2_ = M_ * M_;
    num_windows_ = (H_pad_ / M_) * (W_pad_ / M_);
    rel_bias_size_ = (2 * M_ - 1) * (2 * M_ - 1);
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Xavier-uniform init for Q, K, V, O (scale = sqrt(1/d_model))
    const double proj_scale = std::sqrt(1.0 / static_cast<double>(d_model_));
    W_q = Tensor::random(d_model_, d_model_, proj_scale);
    W_k = Tensor::random(d_model_, d_model_, proj_scale);
    W_v = Tensor::random(d_model_, d_model_, proj_scale);
    W_o = Tensor::random(d_model_, d_model_, proj_scale);
    b_o = Tensor::zeros(1, d_model_);

    // Relative position bias: small truncated-normal init (paper §4.2)
    relative_position_bias_ = Tensor::zeros(num_heads_, rel_bias_size_);
    if (bias_type_ == "relative") {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.02);
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t i = 0; i < rel_bias_size_; ++i) {
                double v;
                do { v = dist(rng); } while (std::fabs(v) > 0.04);  // truncated at 2σ
                relative_position_bias_[h][i] = v;
            }
        }
    }

    // Allocate gradient tensors (zeros)
    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
    grad_b_o = Tensor::zeros(1, d_model_);
    grad_relative_position_bias_ = Tensor::zeros(num_heads_, rel_bias_size_);
}

std::vector<Tensor*> WindowAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &b_o, &relative_position_bias_};
}

std::vector<Tensor*> WindowAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_b_o, &grad_relative_position_bias_};
}

void WindowAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
    grad_relative_position_bias_.fill(0.0);
}

void WindowAttention::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_k -= grad_W_k * learning_rate;
    W_v -= grad_W_v * learning_rate;
    W_o -= grad_W_o * learning_rate;
    b_o -= grad_b_o * learning_rate;
    if (bias_type_ == "relative") {
        relative_position_bias_ -= grad_relative_position_bias_ * learning_rate;
    }
}

// =============================================================================
// Forward: partition (H_pad * W_pad, d_model) into num_windows_ non-overlapping
// MxM windows; for each window compute MHA over its M² tokens; add learnable
// relative position bias B[h, (dr, dc) index] to the QK^T/sqrt(d) logits;
// softmax row-wise; weighted sum of V; output projection W_o + b_o.
// =============================================================================
Tensor WindowAttention::forward(const Tensor& input) {
    const size_t N = input.rows;
    const size_t expected_N = H_pad_ * W_pad_;
    if (N != expected_N) {
        throw std::invalid_argument(
            "WindowAttention.forward: input.rows must equal H_pad*W_pad");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument(
            "WindowAttention.forward: input.cols must equal d_model");
    }

    last_input_ = input.clone();

    // ---- 1. Partition input into windows ----
    const size_t nH = H_pad_ / M_;
    const size_t nW = W_pad_ / M_;
    window_indices_.assign(num_windows_ * M2_, 0);
    for (size_t wh = 0; wh < nH; ++wh) {
        for (size_t ww = 0; ww < nW; ++ww) {
            const size_t w = wh * nW + ww;
            for (size_t r = 0; r < M_; ++r) {
                for (size_t c = 0; c < M_; ++c) {
                    const size_t in_pos = r * M_ + c;
                    const size_t gi = (wh * M_ + r) * W_pad_ + (ww * M_ + c);
                    window_indices_[w * M2_ + in_pos] = gi;
                }
            }
        }
    }

    // ---- 2. QKV projections (bias-free) ----
    Tensor Q(N, d_model_), K(N, d_model_), V(N, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double q = 0.0, k = 0.0, vv = 0.0;
            for (size_t f = 0; f < d_model_; ++f) {
                double x = input[t][f];
                q += x * W_q[j][f];
                k += x * W_k[j][f];
                vv += x * W_v[j][f];
            }
            Q[t][j] = q;
            K[t][j] = k;
            V[t][j] = vv;
        }
    }
    last_q_ = Q;
    last_k_ = K;
    last_v_ = V;

    // ---- 3. Per-window attention ----
    // last_attn_ layout: (num_windows_ * num_heads_ * M², M²)
    //   row (w * num_heads_ + h) * M² + qi  =  attention row qi for window w, head h
    last_attn_ = Tensor(num_windows_ * num_heads_ * M2_, M2_);
    Tensor head_out_flat(N, d_model_);
    head_out_flat.fill(0.0);

    for (size_t w = 0; w < num_windows_; ++w) {
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor scores(M2_, M2_);
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                const size_t qr = qi / M_;
                const size_t qc = qi % M_;
                for (size_t ki = 0; ki < M2_; ++ki) {
                    const size_t g_k = window_indices_[w * M2_ + ki];
                    const size_t kr = ki / M_;
                    const size_t kc = ki % M_;
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        s += Q[g_q][h_off + d] * K[g_k][h_off + d];
                    }
                    s *= scale_;
                    if (bias_type_ == "relative") {
                        const long dr = (long)kr - (long)qr;
                        const long dc = (long)kc - (long)qc;
                        const size_t bias_idx =
                            ((size_t)(dr + (long)(M_ - 1))) * (2 * M_ - 1)
                            + ((size_t)(dc + (long)(M_ - 1)));
                        s += relative_position_bias_[h][bias_idx];
                    }
                    scores[qi][ki] = s;
                }
            }
            // Row-wise softmax
            for (size_t qi = 0; qi < M2_; ++qi) {
                double maxv = -1e30;
                for (size_t ki = 0; ki < M2_; ++ki)
                    maxv = std::max(maxv, scores[qi][ki]);
                double sum = 0.0;
                for (size_t ki = 0; ki < M2_; ++ki) {
                    scores[qi][ki] = std::exp(scores[qi][ki] - maxv);
                    sum += scores[qi][ki];
                }
                if (sum < 1e-30) sum = 1e-30;
                for (size_t ki = 0; ki < M2_; ++ki)
                    scores[qi][ki] /= sum;
            }
            // Cache attention (per-(w,h, qi), a row of M2_ values)
            const size_t attn_row = (w * num_heads_ + h) * M2_;
            for (size_t qi = 0; qi < M2_; ++qi)
                for (size_t ki = 0; ki < M2_; ++ki)
                    last_attn_[attn_row + qi][ki] = scores[qi][ki];
            // head output: A @ V_slice  → (M², head_dim_)
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t ki = 0; ki < M2_; ++ki) {
                        const size_t g_k = window_indices_[w * M2_ + ki];
                        s += scores[qi][ki] * V[g_k][h_off + d];
                    }
                    head_out_flat[g_q][h_off + d] = s;
                }
            }
        }
    }
    last_head_out_ = head_out_flat;

    // ---- 4. Output projection ----
    Tensor out_flat(N, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_o[0][j];
            for (size_t k = 0; k < d_model_; ++k) {
                s += head_out_flat[t][k] * W_o[k][j];
            }
            out_flat[t][j] = s;
        }
    }
    last_out_pre_ = out_flat;
    return out_flat;
}

// Stub — full impl in Task 4. Returns zeros so the calling test fails at
// runtime (rel_err > 1 against FD), not at link time.
Tensor WindowAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0);
    grad_W_o.fill(0.0); grad_b_o.fill(0.0);
    grad_relative_position_bias_.fill(0.0);
    return Tensor(grad_output.rows, grad_output.cols);
}
