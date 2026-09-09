// Deformable 1D Attention — Xia et al., ICLR 2023
//   "Vision Transformers with Deformable Attention"
//   https://arxiv.org/abs/2201.00520
//
// See deformable_attention.h for the full formulation and backward derivation.

#include "deformable_attention.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace {

// Signed, bounded offset activation (paper Eq. 2: Δp = s · tanh(θ_offset(q))).
// tanh keeps the offset in (-1, 1) and, unlike a sigmoid, lets a query look
// BACKWARD from its reference point.
inline double da_tanh_deriv(double th) {
    return 1.0 - th * th;
}

inline double da_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double da_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

} // namespace

// ============================================================================
// Deformable1DAttention
// ============================================================================

Deformable1DAttention::Deformable1DAttention(size_t d_model, size_t num_heads,
                                             size_t num_points, size_t num_ref_points)
    : d_model_(d_model), num_heads_(num_heads), num_points_(num_points),
      num_ref_points_(num_ref_points == 0 ? num_points : num_ref_points),
      head_dim_(d_model && num_heads ? d_model / num_heads : 0),
      W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model),
      W_offsets(d_model, num_heads * num_points),
      W_o(d_model, d_model) {
    if (d_model == 0)
        throw std::invalid_argument("Deformable1DAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("Deformable1DAttention: num_heads must be > 0");
    if (num_points == 0)
        throw std::invalid_argument("Deformable1DAttention: num_points must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "Deformable1DAttention: d_model must be divisible by num_heads");
    if (num_ref_points_ != num_points)
        throw std::invalid_argument(
            "Deformable1DAttention: num_ref_points must equal num_points (or 0)");
    head_dim_ = d_model / num_heads;

    reference_points = Tensor(1, num_points);
    grad_reference_points = Tensor(1, num_points);
    grad_reference_points.fill(0.0);
    // Init reference_points to the CELL-CENTRE uniform grid u_i = (i + 0.5) / P
    // (the standard grid-sample / deformable convention). Cell centres — rather
    // than the endpoint grid i/(P-1) — keep every reference point strictly
    // INTERIOR, so no sampling point starts pinned against the [0, n-1] clamp.
    // A point on the clamp boundary receives zero offset gradient (the forward is
    // locally constant there), so an endpoint grid would waste part of the
    // sampling budget from step 0.
    for (size_t i = 0; i < num_points; ++i) {
        reference_points(0, i) = ((double)i + 0.5) / (double)num_points;
    }
}

Tensor Deformable1DAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("Deformable1DAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("Deformable1DAttention: input.rows must be > 0");
    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t P = num_points_;
    const size_t dh = head_dim_;

    last_input_ = input.clone();

    last_q_ = W_q.forward(input);    // (N, d_model)
    last_k_ = W_k.forward(input);    // (N, d_model)
    last_v_ = W_v.forward(input);    // (N, d_model)
    last_offsets_ = W_offsets.forward(input);  // (N, H*P)  RAW pre-sigmoid

    // Compute sigmoid + scale, sample positions
    last_positions_ = Tensor(N, H * P);
    last_K_sampled_ = Tensor(N, H * P * dh);
    last_V_sampled_ = Tensor(N, H * P * dh);

    const double n_minus_1 = (N > 1) ? (double)(N - 1) : 1.0;
    // Maximum signed displacement around a reference point (paper Eq. 2's `s`).
    const double offset_scale = std::max(1.0, n_minus_1 * 0.5);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double raw = last_offsets_(t, oh);
                double th = std::tanh(raw);
                double pos = reference_points(0, k) * n_minus_1 + th * offset_scale;
                // Clamp to [0, n-1] so pos at the boundary floors exactly to N-1
                if (pos < 0.0) pos = 0.0;
                if (pos > (double)(N - 1)) pos = (double)(N - 1);
                last_positions_(t, oh) = pos;

                size_t i = (size_t)std::floor(pos);
                size_t j = (i >= N - 1) ? i : (i + 1);
                double alpha_w = pos - (double)i;
                double beta_w = 1.0 - alpha_w;

                // Bilinear sample K_h[t] (we don't have per-head slicing yet — use the head slice)
                for (size_t d = 0; d < dh; ++d) {
                    size_t feat = h * dh + d;
                    double kv_i = last_k_(i, feat);
                    double kv_j = last_k_(j, feat);
                    last_K_sampled_(t, oh * dh + d) = beta_w * kv_i + alpha_w * kv_j;
                    double vv_i = last_v_(i, feat);
                    double vv_j = last_v_(j, feat);
                    last_V_sampled_(t, oh * dh + d) = beta_w * vv_i + alpha_w * vv_j;
                }
            }
        }
    }

    // Compute attention scores and softmax
    last_attn_ = Tensor(N, H * P);
    last_concat_ = Tensor(N, d_model_);

    const double inv_sqrt_dh = 1.0 / std::sqrt((double)dh);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            // Compute scores [t, h*P + k]
            double mx = -1e30;
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d) {
                    dot += last_q_(t, h * dh + d) * last_K_sampled_(t, oh * dh + d);
                }
                double s = dot * inv_sqrt_dh;
                last_attn_(t, oh) = s;
                if (s > mx) mx = s;
            }
            // Softmax
            double sum = 0.0;
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double e = std::exp(last_attn_(t, oh) - mx);
                last_attn_(t, oh) = e;
                sum += e;
            }
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                last_attn_(t, oh) /= sum;
            }
            // Compute head output
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t k = 0; k < P; ++k) {
                    size_t oh = h * P + k;
                    acc += last_attn_(t, oh) * last_V_sampled_(t, oh * dh + d);
                }
                last_concat_(t, h * dh + d) = acc;
            }
        }
    }

    // Apply output projection
    return W_o.forward(last_concat_);
}

Tensor Deformable1DAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t N = last_input_.rows;
    const size_t H = num_heads_;
    const size_t P = num_points_;
    const size_t dh = head_dim_;

    if (grad_output.rows != N || grad_output.cols != d_model_)
        throw std::invalid_argument(
            "Deformable1DAttention::backward: grad_output shape mismatch");

    // ---- Output projection: Y = last_concat_ · W_o^T + b_o ----
    // d_concat = dY · W_o ; dW_o += ... ; db_o += colsum(dY)
    // Note: W_o.weights is (d_model, d_model), so Y[t, j] = Σ_i last_concat_[t, i] * W_o.weights[j, i]
    Tensor d_concat(N, d_model_);
    d_concat.fill(0.0);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i) {
                acc += grad_output(t, i) * W_o.weights(i, j);
            }
            d_concat(t, j) = acc;
        }
    }
    // dW_o[j, i] += Σ_t last_concat_[t, i] * grad_output[t, j]
    for (size_t j = 0; j < d_model_; ++j)
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += last_concat_(t, i) * grad_output(t, j);
            W_o.grad_weights(j, i) += acc;
        }
    // db_o[j] += Σ_t grad_output[t, j]
    for (size_t j = 0; j < d_model_; ++j) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += grad_output(t, j);
        W_o.grad_bias(0, j) += bacc;
    }

    // ---- Per-head: split d_concat by head, backprop through softmax + bilinear sampling ----
    Tensor d_V_sampled(N, H * P * dh);
    Tensor d_K_sampled(N, H * P * dh);
    Tensor d_attn(N, H * P);
    Tensor d_score(N, H * P);
    Tensor d_q(N, d_model_);
    d_q.fill(0.0);
    Tensor d_v(N, d_model_);
    d_v.fill(0.0);
    Tensor d_k(N, d_model_);
    d_k.fill(0.0);

    const double inv_sqrt_dh = 1.0 / std::sqrt((double)dh);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            // d_attn[t, h*P+k] = Σ_d d_concat[t, h*dh+d] * V_sampled[t, (h*P+k)*dh+d]
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double a = 0.0;
                for (size_t d = 0; d < dh; ++d) {
                    a += d_concat(t, h * dh + d) * last_V_sampled_(t, oh * dh + d);
                }
                d_attn(t, oh) = a;
            }
            // d_V_sampled[t, h*P+k, d] = Attn[t, h*P+k] * d_concat[t, h*dh+d]
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                for (size_t d = 0; d < dh; ++d) {
                    d_V_sampled(t, oh * dh + d) =
                        last_attn_(t, oh) * d_concat(t, h * dh + d);
                }
            }
            // Softmax backward: d_score[t, k] = attn[t, k] * (d_attn[t, k] - Σ_k' attn[t, k'] * d_attn[t, k'])
            double inner = 0.0;
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                inner += last_attn_(t, oh) * d_attn(t, oh);
            }
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                d_score(t, oh) = last_attn_(t, oh) * (d_attn(t, oh) - inner);
            }
            // d_K_sampled[t, h*P+k, d] = d_score[t, h*P+k] * Q[t, h*dh+d] * inv_sqrt_dh
            // d_Q[t, h*dh+d] += Σ_k d_score[t, h*P+k] * K_sampled[t, (h*P+k)*dh+d] * inv_sqrt_dh
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double ds = d_score(t, oh) * inv_sqrt_dh;
                for (size_t d = 0; d < dh; ++d) {
                    d_K_sampled(t, oh * dh + d) = ds * last_q_(t, h * dh + d);
                    d_q(t, h * dh + d) += ds * last_K_sampled_(t, oh * dh + d);
                }
            }
        }
    }

    // ---- Bilinear backward: K_sampled, V_sampled → K, V, positions ----
    // We also accumulate d_offsets_raw[t, h*P+k] (via d_pos * n-1 * sigmoid_deriv(raw))
    // and d_reference_points[k] (via d_pos * n-1).
    Tensor d_pos(N, H * P);
    d_pos.fill(0.0);
    Tensor d_offset_raw(N, H * P);
    d_offset_raw.fill(0.0);

    const double n_minus_1 = (N > 1) ? (double)(N - 1) : 1.0;
    const double offset_scale = std::max(1.0, n_minus_1 * 0.5);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t k = 0; k < P; ++k) {
                size_t oh = h * P + k;
                double pos = last_positions_(t, oh);
                size_t i = (size_t)std::floor(pos);
                size_t j = (i >= N - 1) ? i : (i + 1);
                double alpha_w = pos - (double)i;
                double beta_w = 1.0 - alpha_w;

                // K_sampled = beta_w * K[i] + alpha_w * K[j], where alpha_w depends on pos
                // (independent of K/V). So the bilinear backward is:
                //   d_alpha = Σ_d (K[j,d] - K[i,d]) * dK_sampled[t,oh,d]   (and similarly for V)
                //   d_pos   = d_alpha  (chain through alpha_w -> pos)
                //   d_K[i]  = beta_w * dK_sampled
                //   d_K[j]  = alpha_w * dK_sampled
                //   d_V[i]  = beta_w * dV_sampled
                //   d_V[j]  = alpha_w * dV_sampled
                double d_alpha_w = 0.0;
                for (size_t d = 0; d < dh; ++d) {
                    size_t feat = h * dh + d;
                    double diff_k = last_k_(j, feat) - last_k_(i, feat);
                    double diff_v = last_v_(j, feat) - last_v_(i, feat);
                    d_alpha_w += diff_k * d_K_sampled(t, oh * dh + d);
                    d_alpha_w += diff_v * d_V_sampled(t, oh * dh + d);
                }
                double dpos_local = d_alpha_w;
                for (size_t d = 0; d < dh; ++d) {
                    size_t feat = h * dh + d;
                    double dks = d_K_sampled(t, oh * dh + d);
                    double dvs = d_V_sampled(t, oh * dh + d);
                    d_k(i, feat) += beta_w * dks;
                    d_k(j, feat) += alpha_w * dks;
                    d_v(i, feat) += beta_w * dvs;
                    d_v(j, feat) += alpha_w * dvs;
                }
                // The position is clamped to [0, N-1]. On a saturated boundary the
                // forward is locally constant in `pos`, so NO gradient may flow to
                // the offset or the reference point (dclamp/dpos = 0 there).
                bool saturated = (pos <= 0.0) || (pos >= (double)(N - 1));
                if (saturated) dpos_local = 0.0;
                d_pos(t, oh) = dpos_local;

                // pos = reference_points[k] * (n-1) + tanh(raw) * offset_scale
                //   d_reference_points[k] += d_pos * (n-1)
                //   d_raw                  = d_pos * offset_scale * (1 - tanh(raw)^2)
                grad_reference_points(0, k) += d_pos(t, oh) * n_minus_1;
                double th = std::tanh(last_offsets_(t, oh));
                d_offset_raw(t, oh) = d_pos(t, oh) * offset_scale * da_tanh_deriv(th);
            }
        }
    }

    // ---- Backward through X · W_q, X · W_k, X · W_v, X · W_offsets ----
    // For W_q, W_k, W_v: dW += X^T @ d_q, d_k, d_v ; dX += d_q, d_k, d_v @ W^T
    // For W_offsets: dW_offsets += X^T @ d_offset_raw ; dX += d_offset_raw @ W_offsets^T
    // For biases: db += colsum(d_*)
    Tensor d_X(N, d_model_);
    d_X.fill(0.0);

    // X = last_input_
        // Forward: Q = X · W_q^T + b_q, where W_q is (d_model, d_model) (Dense convention).
        // Y[t, j] = Σ_i X[t, i] * W_q.weights[j, i];  dW_q[j, i] += Σ_t X[t, i] * dY[t, j]
        // dX[t, i] += Σ_j dY[t, j] * W_q.weights[j, i]
        // db_q[j] += Σ_t dY[t, j]
        for (size_t j = 0; j < d_model_; ++j)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t) acc += last_input_(t, i) * d_q(t, j);
                W_q.grad_weights(j, i) += acc;
            }
        for (size_t t = 0; t < N; ++t)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) acc += d_q(t, j) * W_q.weights(j, i);
                d_X(t, i) += acc;
            }
        for (size_t j = 0; j < d_model_; ++j) {
            double bacc = 0.0;
            for (size_t t = 0; t < N; ++t) bacc += d_q(t, j);
            W_q.grad_bias(0, j) += bacc;
        }

        for (size_t j = 0; j < d_model_; ++j)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t) acc += last_input_(t, i) * d_k(t, j);
                W_k.grad_weights(j, i) += acc;
            }
        for (size_t t = 0; t < N; ++t)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) acc += d_k(t, j) * W_k.weights(j, i);
                d_X(t, i) += acc;
            }
        for (size_t j = 0; j < d_model_; ++j) {
            double bacc = 0.0;
            for (size_t t = 0; t < N; ++t) bacc += d_k(t, j);
            W_k.grad_bias(0, j) += bacc;
        }

        for (size_t j = 0; j < d_model_; ++j)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t) acc += last_input_(t, i) * d_v(t, j);
                W_v.grad_weights(j, i) += acc;
            }
        for (size_t t = 0; t < N; ++t)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) acc += d_v(t, j) * W_v.weights(j, i);
                d_X(t, i) += acc;
            }
        for (size_t j = 0; j < d_model_; ++j) {
            double bacc = 0.0;
            for (size_t t = 0; t < N; ++t) bacc += d_v(t, j);
            W_v.grad_bias(0, j) += bacc;
        }

    // W_offsets has shape (out_features=H*P, in_features=d_model). Forward: Y = X · W_offsets^T
    // Y[t, j] = Σ_k X[t, k] * W_offsets.weights[j, k];  dW_offsets[j, k] += X[t, k] * dY[t, j]
    for (size_t j = 0; j < (size_t)(H * P); ++j)
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += last_input_(t, k) * d_offset_raw(t, j);
            W_offsets.grad_weights(j, k) += acc;
        }
    // dX[t, k] += Σ_j dY[t, j] * W_offsets.weights[j, k]
    for (size_t t = 0; t < N; ++t)
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < (size_t)(H * P); ++j) acc += d_offset_raw(t, j) * W_offsets.weights(j, k);
            d_X(t, k) += acc;
        }
    // db_offsets[j] = Σ_t d_offset_raw[t, j]
    for (size_t j = 0; j < (size_t)(H * P); ++j) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += d_offset_raw(t, j);
        W_offsets.grad_bias(0, j) += bacc;
    }

    return d_X;
}

void Deformable1DAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_offsets.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    // SGD-style for reference_points
    for (size_t k = 0; k < num_points_; ++k) {
        reference_points(0, k) -= learning_rate * grad_reference_points(0, k);
    }
}

void Deformable1DAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_offsets.zero_grad();
    W_o.zero_grad();
    for (double& v : grad_reference_points.data) v = 0.0;
}

std::vector<Tensor*> Deformable1DAttention::parameters() {
    return {&W_q.weights, &W_q.bias, &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias, &W_offsets.weights, &W_offsets.bias,
            &reference_points, &W_o.weights, &W_o.bias};
}

std::vector<Tensor*> Deformable1DAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias, &W_k.grad_weights, &W_k.grad_bias,
            &W_v.grad_weights, &W_v.grad_bias, &W_offsets.grad_weights, &W_offsets.grad_bias,
            &grad_reference_points, &W_o.grad_weights, &W_o.grad_bias};
}

// ============================================================================
// Deformable1DBlock
// ============================================================================

Deformable1DBlock::Deformable1DBlock(size_t d_model, size_t num_heads, size_t num_points,
                                     size_t num_ref_points)
    : attn(d_model, num_heads, num_points, num_ref_points),
      ffn1(d_model, d_model * 2), ffn2(d_model * 2, d_model),
      ln1_(std::make_unique<LayerNorm>(d_model)), ln2_(std::make_unique<LayerNorm>(d_model)) {}

Tensor Deformable1DBlock::forward(const Tensor& x) {
    // Pre-LN -> attn -> residual
    last_ln1_ = ln1_->forward(x);
    Tensor a = attn.forward(last_ln1_);
    Tensor h1 = x + a;

    // Pre-LN -> FFN -> residual
    last_ln2_ = ln2_->forward(h1);
    last_f1_ = ffn1.forward(last_ln2_);
    last_f1g_ = last_f1_.apply(da_gelu);
    Tensor f2 = ffn2.forward(last_f1g_);
    Tensor h2 = h1 + f2;
    return h2;
}

Tensor Deformable1DBlock::backward(const Tensor& grad_output, double learning_rate) {
    // h2 = h1 + f2 -> df2 = grad_output
    Tensor df_f2 = grad_output;

    // f2 = ffn2(f1g)
    Tensor df1g = ffn2.backward(df_f2, learning_rate);
    // f1g = GELU(f1)  ->  df1 = df1g * GELU'(f1)
    Tensor df1 = df1g.hadamard(last_f1_.apply(da_gelu_deriv));
    Tensor dln2 = ffn1.backward(df1, learning_rate);
    // ln2(h1) -> dh1
    Tensor dh1_part = ln2_->backward(dln2, learning_rate);
    Tensor dh1 = grad_output + dh1_part;

    // h1 = x + a -> dx = dh1 + grad_a; grad_a = dh1
    Tensor dln1_a = attn.backward(dh1, learning_rate);
    Tensor dx_part = ln1_->backward(dln1_a, learning_rate);
    Tensor dx = dh1 + dx_part;
    return dx;
}

void Deformable1DBlock::update_weights(double lr) {
    attn.update_weights(lr);
    ffn1.update_weights(lr); ffn2.update_weights(lr);
    ln1_->update_weights(lr); ln2_->update_weights(lr);
}
void Deformable1DBlock::zero_grad() {
    attn.zero_grad();
    ffn1.zero_grad(); ffn2.zero_grad();
    ln1_->zero_grad(); ln2_->zero_grad();
}
std::vector<Tensor*> Deformable1DBlock::parameters() {
    std::vector<Tensor*> p = attn.parameters();
    auto f1 = ffn1.parameters();
    p.insert(p.end(), f1.begin(), f1.end());
    auto f2 = ffn2.parameters();
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}
std::vector<Tensor*> Deformable1DBlock::gradients() {
    std::vector<Tensor*> g = attn.gradients();
    auto f1 = ffn1.gradients();
    g.insert(g.end(), f1.begin(), f1.end());
    auto f2 = ffn2.gradients();
    g.insert(g.end(), f2.begin(), f2.end());
    return g;
}

// ============================================================================
// Deformable1DModel
// ============================================================================

Deformable1DModel::Deformable1DModel(size_t d_input, size_t d_model, size_t d_output,
                                     size_t num_blocks, size_t num_heads, size_t num_points,
                                     size_t num_ref_points)
    : input_proj(d_input, d_model), classifier(d_model, d_output),
      final_ln_(std::make_unique<LayerNorm>(d_model)) {
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.push_back(std::make_unique<Deformable1DBlock>(d_model, num_heads, num_points, num_ref_points));
    }
}

Tensor Deformable1DModel::forward(const Tensor& x) {
    Tensor h = input_proj.forward(x);
    for (auto& b : blocks_) h = b->forward(h);
    h = final_ln_->forward(h);
    return classifier.forward(h);
}

Tensor Deformable1DModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d = classifier.backward(grad_output, learning_rate);
    d = final_ln_->backward(d, learning_rate);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d = (*it)->backward(d, learning_rate);
    }
    return input_proj.backward(d, learning_rate);
}

void Deformable1DModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    for (auto& b : blocks_) b->update_weights(lr);
    classifier.update_weights(lr);
    final_ln_->update_weights(lr);
}
void Deformable1DModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    classifier.zero_grad();
    final_ln_->zero_grad();
}
std::vector<Tensor*> Deformable1DModel::parameters() {
    std::vector<Tensor*> p = input_proj.parameters();
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}
std::vector<Tensor*> Deformable1DModel::gradients() {
    std::vector<Tensor*> g = input_proj.gradients();
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}