#include "focal_modulation.h"

#include <algorithm>
#include <cmath>
#include <random>

// ============================================================================
// Forward chain helpers (file-local)
// ============================================================================
//
// We compute three quantities during forward so they're available in
// backward:
//
//   - z_l[b, s, d] for l=0..L-1 — level-l's hierarchical sequential
//     feature at position s. Built from the input directly (z_0 = input)
//     and shifted-averaged as the level goes up:
//         z_0[b, s, d] = input_3d[b, s, d]
//         z_l[b, s, d] = (z_{l-1}[b, s, d] + z_{l-1}[b, max(0,s-1), d]) / 2
//     so each level carries a wider receptive field.
//   - g_l = w_l ⊙ z_l — the gated level output.
//   - m_l — the modulator after the per-level MLP; shape (B, D) and
//     broadcast across S.
//
// During the output projection we keep `last_y_` (post-aggregate, pre-proj)
// so the W_o backward is the standard Dense chain.

static inline double sigmoid_fn(double x) {
    if (x >= 0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = std::exp(x);
    return z / (1.0 + z);
}

// Numerically stable GELU (matches the activations.h::GELU formula:
/// 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715 x^3))). Applied row-wise below.)
static inline double gelu_scalar(double x) {
    constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
    double xc = std::max(-4.0, std::min(4.0, x));
    return 0.5 * xc * (1.0 + std::tanh(SQRT_2_OVER_PI * (xc + 0.044715 * xc * xc * xc)));
}

// ============================================================================
// FocalModulation
// ============================================================================

FocalModulation::FocalModulation(size_t d_model, size_t seq_len,
                                 size_t num_levels,
                                 size_t mlp_dim)
    : d_model_(d_model),
      seq_len_(seq_len),
      num_levels_(num_levels),
      mlp_dim_(mlp_dim == 0 ? (d_model * 2) : mlp_dim) {
    if (d_model == 0)
        throw std::invalid_argument("FocalModulation: d_model must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("FocalModulation: seq_len must be > 0");
    if (num_levels == 0)
        throw std::invalid_argument("FocalModulation: num_levels must be > 0");
    if (num_levels > seq_len + 1)
        // Receptive field at the deepest level can reach seq_len+1; allow up to that.
        throw std::invalid_argument("FocalModulation: num_levels cannot exceed (seq_len + 1)");

    // Per-level parameters
    level_scalars_.resize(num_levels_);
    grad_level_scalars_.resize(num_levels_);
    W1_.resize(num_levels_);  b1_.resize(num_levels_);
    W2_.resize(num_levels_);  b2_.resize(num_levels_);
    grad_W1_.resize(num_levels_);  grad_b1_.resize(num_levels_);
    grad_W2_.resize(num_levels_);  grad_b2_.resize(num_levels_);

    std::mt19937 gen(42);
    double std_w = std::sqrt(1.0 / (double)d_model_);
    std::normal_distribution<> nd(0.0, std_w);

    size_t cnt = 0;
    for (size_t l = 0; l < num_levels_; ++l) {
        // level_scalars (1, D) — initialized to 1.0 (level-on) so a fresh
        // network isn't dead. Treated as a parameter, gets its own small LR
        // via the standard optimizer chain.
        level_scalars_[l] = Tensor(1, d_model_);
        grad_level_scalars_[l] = Tensor(1, d_model_);
        for (size_t d = 0; d < d_model_; ++d)
            level_scalars_[l][0][d] = 1.0;
        grad_level_scalars_[l].fill(0.0);
        cnt += d_model_;

        // W1_l (mlp_dim, D), b1_l (1, mlp_dim)
        W1_[l] = Tensor(mlp_dim_, d_model_);
        b1_[l] = Tensor(1, mlp_dim_);
        grad_W1_[l] = Tensor(mlp_dim_, d_model_);
        grad_b1_[l] = Tensor(1, mlp_dim_);
        for (size_t i = 0; i < W1_[l].rows; ++i)
            for (size_t j = 0; j < W1_[l].cols; ++j)
                W1_[l][i][j] = nd(gen);
        b1_[l].fill(0.0);
        grad_W1_[l].fill(0.0);
        grad_b1_[l].fill(0.0);
        cnt += W1_[l].rows * W1_[l].cols + b1_[l].cols;

        // W2_l (D, mlp_dim), b2_l (1, D)
        W2_[l] = Tensor(d_model_, mlp_dim_);
        b2_[l] = Tensor(1, d_model_);
        grad_W2_[l] = Tensor(d_model_, mlp_dim_);
        grad_b2_[l] = Tensor(1, d_model_);
        for (size_t i = 0; i < W2_[l].rows; ++i)
            for (size_t j = 0; j < W2_[l].cols; ++j)
                W2_[l][i][j] = nd(gen) * 0.5;
        b2_[l].fill(0.0);
        grad_W2_[l].fill(0.0);
        grad_b2_[l].fill(0.0);
        cnt += W2_[l].rows * W2_[l].cols + b2_[l].cols;
    }

    // Output projection: W_o (D, D), b_o (1, D)
    W_o = Tensor(d_model_, d_model_);
    b_o = Tensor(1, d_model_);
    grad_W_o = Tensor(d_model_, d_model_);
    grad_b_o = Tensor(1, d_model_);
    double std_o = std::sqrt(1.0 / (double)d_model_);
    std::normal_distribution<> ndo(0.0, std_o);
    for (size_t i = 0; i < W_o.rows; ++i)
        for (size_t j = 0; j < W_o.cols; ++j)
            W_o[i][j] = ndo(gen);
    b_o.fill(0.0);
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
    cnt += W_o.rows * W_o.cols + b_o.cols;

    cached_param_count_ = cnt;
}

// We treat a (B, S*D) flat tensor as a (B, S, D) view by the layout
// (b, s, d) ↔ flat(b, s*D + d). This is the same as MlpMixerBlock.

static inline double input3d(const Tensor& flat, size_t b, size_t S, size_t D,
                              size_t s, size_t d) {
    (void)S;
    return flat[b][s * D + d];
}

Tensor FocalModulation::forward(const Tensor& input) {
    last_input_ = input.clone();
    size_t B = input.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // ------------------------------------------------------------------
    // 1. Build z_l chain (size L); also compute g_l = w_l ⊙ z_l for
    //    the per-level gated output.
    // ------------------------------------------------------------------
    last_z_l_.resize(num_levels_);
    last_g_l_.resize(num_levels_);
    // z_0 = input
    last_z_l_[0] = Tensor(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                last_z_l_[0][b][s * D + d] = input3d(input, b, S, D, s, d);

    for (size_t l = 1; l < num_levels_; ++l) {
        last_z_l_[l] = Tensor(B, S * D);
        const Tensor& prev = last_z_l_[l - 1];
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                size_t sp = (s == 0) ? 0 : (s - 1);
                for (size_t d = 0; d < D; ++d) {
                    double v = prev[b][s * D + d] + prev[b][sp * D + d];
                    v *= 0.5;
                    last_z_l_[l][b][s * D + d] = v;
                }
            }
        }
    }

    for (size_t l = 0; l < num_levels_; ++l) {
        last_g_l_[l] = Tensor(B, S * D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t d = 0; d < D; ++d) {
                    last_g_l_[l][b][s * D + d] =
                        level_scalars_[l][0][d] * last_z_l_[l][b][s * D + d];
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Query q = GAP over S of input → (B, D).
    //    Operate on the *raw* input (the paper uses the pre-shift input).
    // ------------------------------------------------------------------
    Tensor q(B, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            double sum = 0.0;
            for (size_t s = 0; s < S; ++s)
                sum += input3d(input, b, S, D, s, d);
            q[b][d] = sum / static_cast<double>(S);
        }
    }
    last_q_ = q.clone();

    // ------------------------------------------------------------------
    // 3. Per-level modulator MLP. For each level l:
    //       h_pre = q @ W1_l^T + b1_l   shape (B, mlp_dim)
    //       h_act = GELU(h_pre)         shape (B, mlp_dim)
    //       m_pre = h_act @ W2_l^T + b2_l  shape (B, D)
    //       m_l = sigmoid(m_pre)        shape (B, D)
    //    m_l is then broadcast across S.
    // ------------------------------------------------------------------
    last_m_l_.resize(num_levels_);
    last_mlp_h_.resize(num_levels_);
    last_mlp_h_act_.resize(num_levels_);

    for (size_t l = 0; l < num_levels_; ++l) {
        // h_pre = q @ W1_l^T + b1_l
        Tensor h_pre(B, mlp_dim_);
        for (size_t b = 0; b < B; ++b) {
            for (size_t j = 0; j < mlp_dim_; ++j) {
                double acc = b1_[l][0][j];
                for (size_t k = 0; k < D; ++k)
                    acc += q[b][k] * W1_[l][j][k];  // W1_l[j, k], x[b, k]
                h_pre[b][j] = acc;
            }
        }
        // h_act = GELU(h_pre)
        Tensor h_act = h_pre.apply([](double v) { return gelu_scalar(v); });
        last_mlp_h_[l] = h_pre;
        last_mlp_h_act_[l] = h_act;

        // m_pre = h_act @ W2_l^T + b2_l
        Tensor m_pre(B, D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t j = 0; j < D; ++j) {
                double acc = b2_[l][0][j];
                for (size_t k = 0; k < mlp_dim_; ++k)
                    acc += h_act[b][k] * W2_[l][j][k];
                m_pre[b][j] = acc;
            }
        }
        // m_l = sigmoid(m_pre)
        last_m_l_[l] = m_pre.apply([](double v) { return sigmoid_fn(v); });
    }

    // ------------------------------------------------------------------
    // 4. Aggregate: y[b, s, d] = sum_l m_l[b, d] * g_l[b, s, d].
    //    Also flatten to (B, S*D).
    // ------------------------------------------------------------------
    last_y_ = Tensor(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t l = 0; l < num_levels_; ++l)
                    acc += last_m_l_[l][b][d] * last_g_l_[l][b][s * D + d];
                last_y_[b][s * D + d] = acc;
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Output projection: out = y @ W_o^T + b_o.
    //    W_o has shape (D, D). y is reshaped view as (B*S, D) for the
    //    matmul and rebroadcast bias.
    // ------------------------------------------------------------------
    Tensor out_flat(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d_out = 0; d_out < D; ++d_out) {
                double acc = b_o[0][d_out];
                for (size_t d = 0; d < D; ++d)
                    acc += last_y_[b][s * D + d] * W_o[d_out][d];
                out_flat[b][s * D + d_out] = acc;
            }
        }
    }
    return out_flat;
}

Tensor FocalModulation::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = last_input_.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // ------------------------------------------------------------------
    // 1. d_y = grad_output @ W_o    (the output-projection chain)
    //    grad_W_o = grad_output^T · last_y_
    //    grad_b_o = sum over batch,sequence of grad_output
    // ------------------------------------------------------------------
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);

    Tensor d_y(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d_out = 0; d_out < D; ++d_out) {
                double go = grad_output[b][s * D + d_out];
                grad_b_o[0][d_out] += go;
                for (size_t d = 0; d < D; ++d) {
                    grad_W_o[d_out][d] += go * last_y_[b][s * D + d];
                    d_y[b][s * D + d] += go * W_o[d_out][d];
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Backprop through the aggregate: d_m_l[b, d] and d_g_l[b, s, d].
    //    y[b, s, d] = sum_l m_l[b, d] * g_l[b, s, d].
    //    d_g_l[b, s, d] = d_y[b, s, d] * m_l[b, d]
    //    d_m_l[b, d]    = sum_{s} d_y[b, s, d] * g_l[b, s, d]
    // ------------------------------------------------------------------
    std::vector<Tensor> d_g_l(num_levels_, Tensor(B, S * D));
    std::vector<Tensor> d_m_l_pre(num_levels_, Tensor(B, D));
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t l = 0; l < num_levels_; ++l) {
                double acc = 0.0;
                for (size_t s = 0; s < S; ++s)
                    acc += d_y[b][s * D + d] * last_g_l_[l][b][s * D + d];
                d_m_l_pre[l][b][d] = acc;
            }
        }
    }
    // Apply d_m_l = d_m_l_pre * sigmoid'(m_l_pre) — local-chain through
    // sigmoid: m_l = σ(m_pre); dm_pre = dm_l * m_l * (1 - m_l).
    // Then propagate dm_pre into d_g_l[m_l] via d_y -> g_l.
    // (Note: d_y -> g_l already was used above; it doesn't see m's
    //  derivative until we are inside the modulator MLP. Here we ONLY
    //  compute dm/dm_pre for the modulator. The contribution of m_l's
    //  derivative onto d_y was already done above because we treated m_l
    //  as a constant in the d_y → d_g_l path. That's wrong: d_y also
    //  depends on m_l through g_l. Below we fix that.)
    //
    // Wait — we have a subtle bookkeeping error. y[b, s, d] depends on
    // BOTH m_l[b, d] and g_l[b, s, d]. The total derivative is:
    //   dy/dm_l = g_l
    //   dy/dg_l = m_l
    // We split it as:
    //   d_g_l += d_y * m_l  (from y's m_l·g_l structure)
    //   d_m_l += d_y * g_l  (already computed as d_m_l_pre in m_pre space)
    //
    // In the d_y → d_g_l loop above we used last_g_l in
    // d_m_l_pre but never wrote to d_g_l. Need to fix:
    //   d_g_l[b, s, d] = sum_{l} d_y[b, s, d] * m_l[b, d] (contribution
    //   from one term of y; we accumulate over levels because the sum
    //   over l is y).
    // We do that in the next block.

    // Initialize d_g_l via d_y and m_l, accounting for the m_l·g_l sum:
    for (size_t l = 0; l < num_levels_; ++l)
        d_g_l[l].fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double dy = d_y[b][s * D + d];
                if (dy == 0.0) continue;
                for (size_t l = 0; l < num_levels_; ++l) {
                    d_g_l[l][b][s * D + d] += dy * last_m_l_[l][b][d];
                }
            }
        }
    }

    // Apply sigmoid' to d_m_l_pre: m_l = σ(m_pre)
    // dm_pre[b, d] = dm_l[b, d] * m_l[b, d] * (1 - m_l[b, d])
    for (size_t l = 0; l < num_levels_; ++l) {
        for (size_t b = 0; b < B; ++b) {
            for (size_t d = 0; d < D; ++d) {
                double ml = last_m_l_[l][b][d];
                d_m_l_pre[l][b][d] = d_m_l_pre[l][b][d] * ml * (1.0 - ml);
            }
        }
    }

    // d_m_l already absorbed the per-y term; no further change.

    // ------------------------------------------------------------------
    // 3. Backprop through the modulator MLPs.
    //    For each level l:
    //      h_pre  = q @ W1_l^T + b1_l
    //      h_act  = GELU(h_pre)
    //      m_pre  = h_act @ W2_l^T + b2_l
    // Given d_m_pre (shape (B, D)):
    //      grad_b2_l = sum_b d_m_pre[b, :]
    //      grad_W2_l = d_m_pre^T · h_act  (D × mlp_dim)
    //      d_h_act   = d_m_pre @ W2_l    (B × mlp_dim)
    //      d_h_pre   = d_h_act * GELU'(h_pre)
    //      grad_b1_l = sum_b d_h_pre[b, :]
    //      grad_W1_l = d_h_pre^T · q     (mlp_dim × D)
    //      d_q       = d_h_pre @ W1_l    (B × D)
    // ------------------------------------------------------------------
    Tensor d_q(B, D);
    d_q.fill(0.0);

    for (size_t l = 0; l < num_levels_; ++l) {
        grad_W2_[l].fill(0.0);
        grad_b2_[l].fill(0.0);
        grad_W1_[l].fill(0.0);
        grad_b1_[l].fill(0.0);

        // grad_W2_l = d_m_pre^T · h_act, grad_b2_l = sum_b d_m_pre[b, :]
        for (size_t b = 0; b < B; ++b) {
            for (size_t j = 0; j < D; ++j) {
                double dm = d_m_l_pre[l][b][j];
                if (dm == 0.0) continue;
                grad_b2_[l][0][j] += dm;
                for (size_t k = 0; k < mlp_dim_; ++k)
                    grad_W2_[l][j][k] += dm * last_mlp_h_act_[l][b][k];
            }
        }
        // d_h_act = d_m_pre @ W2_l
        Tensor d_h_act(B, mlp_dim_);
        d_h_act.fill(0.0);
        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < mlp_dim_; ++k) {
                double acc = 0.0;
                for (size_t j = 0; j < D; ++j)
                    acc += d_m_l_pre[l][b][j] * W2_[l][j][k];
                d_h_act[b][k] = acc;
            }
        }
        // d_h_pre = d_h_act * GELU'(h_pre). GELU' analytically:
        // inner cdf = 0.5 * (1 + tanh(a)), a = √(2/π)(x + 0.044715 x³).
        // GELU(x) = x * cdf. dGELU/dx = cdf + x * 0.5 * sech²(a) * da/dx.
        // For implementation we follow the activations.h::GELU::derivative
        // pattern by computing it inline.
        constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
        auto gelu_grad = [&](double x) -> double {
            double xc = std::max(-4.0, std::min(4.0, x));
            double inner = SQRT_2_OVER_PI * (xc + 0.044715 * xc * xc * xc);
            double t = std::tanh(inner);
            double cdf = 0.5 * (1.0 + t);
            double dinner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * xc * xc);
            // sech^2(inner) = 1 - tanh^2(inner)
            double sech2 = 1.0 - t * t;
            return cdf + xc * 0.5 * sech2 * dinner;
        };
        Tensor d_h_pre(B, mlp_dim_);
        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < mlp_dim_; ++k)
                d_h_pre[b][k] = d_h_act[b][k] * gelu_grad(last_mlp_h_[l][b][k]);
        }
        // grad_b1_l = sum_b d_h_pre[b, :]
        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < mlp_dim_; ++k)
                grad_b1_[l][0][k] += d_h_pre[b][k];
        }
        // grad_W1_l = d_h_pre^T · q  (re-derived cleanly; we don't
        // accumulate — write directly)
        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < mlp_dim_; ++k) {
                double dhp = d_h_pre[b][k];
                if (dhp == 0.0) continue;
                for (size_t j = 0; j < D; ++j)
                    grad_W1_[l][k][j] += dhp * last_q_[b][j];
            }
        }
        // d_q += d_h_pre @ W1_l    (accumulate across levels)
        for (size_t b = 0; b < B; ++b) {
            for (size_t j = 0; j < D; ++j) {
                double acc = 0.0;
                for (size_t k = 0; k < mlp_dim_; ++k)
                    acc += d_h_pre[b][k] * W1_[l][k][j];
                d_q[b][j] += acc;
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Backprop the query. q[b, d] = (1/S) * sum_s input[b, s*D + d].
    //    d_input[b, s*D + d] += d_q[b, d] * (1/S) for all s.
    // ------------------------------------------------------------------
    Tensor d_input(B, S * D);
    d_input.fill(0.0);
    double inv_S = 1.0 / static_cast<double>(S);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            double v = d_q[b][d] * inv_S;
            if (v == 0.0) continue;
            for (size_t s = 0; s < S; ++s)
                d_input[b][s * D + d] += v;
        }
    }

    // ------------------------------------------------------------------
    // 5. Backprop through the gated levels. g_l[b, s, d] = w_l[d] * z_l[b, s, d].
    //    grad_level_scalars[l][d] = sum_{b, s} d_g_l[b, s, d] * z_l[b, s, d]
    //    d_z_l[b, s, d]            = d_g_l[b, s, d] * w_l[d]
    // ------------------------------------------------------------------
    grad_level_scalars_.clear();
    grad_level_scalars_.resize(num_levels_, Tensor(1, D));
    std::vector<Tensor> d_z_l(num_levels_, Tensor(B, S * D));
    for (size_t l = 0; l < num_levels_; ++l) {
        grad_level_scalars_[l].fill(0.0);
        d_z_l[l].fill(0.0);
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t d = 0; d < D; ++d) {
                    double dg = d_g_l[l][b][s * D + d];
                    grad_level_scalars_[l][0][d] += dg * last_z_l_[l][b][s * D + d];
                    d_z_l[l][b][s * D + d] = dg * level_scalars_[l][0][d];
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Backprop through the z_l chain.
    //    z_0[b, s, d] = input[b, s*D + d]
    //    z_l[b, s, d] = 0.5 * (z_{l-1}[b, s, d] + z_{l-1}[b, max(0, s-1), d])
    //    Note: for s = 0, both terms are z_{l-1}[b, 0, d] (clamped), so
    //          ∂z_l[b, 0, d]/∂z_{l-1}[b, 0, d] = 1.0.
    //    For s ≥ 1, ∂z_l[b, s, d]/∂z_{l-1}[b, s, d] = 0.5 and
    //                   ∂z_l[b, s+1, d]/∂z_{l-1}[b, s, d] = 0.5
    //                   for s+1 < S.
    // ------------------------------------------------------------------
    for (size_t l = 1; l < num_levels_; ++l) {
        const Tensor& cur = d_z_l[l];
        Tensor& prev = d_z_l[l - 1];
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t d = 0; d < D; ++d) {
                    double coef = (s == 0) ? 1.0 : 0.5;
                    prev[b][s * D + d] += coef * cur[b][s * D + d];
                    if (s + 1 < S)
                        prev[b][s * D + d] += 0.5 * cur[b][(s + 1) * D + d];
                }
            }
        }
    }
    // z_0 = input → d_input[b, s*D + d] += d_z_0[b, s, d]
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d)
                d_input[b][s * D + d] += d_z_l[0][b][s * D + d];
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

void FocalModulation::update_weights(double learning_rate) {
    for (size_t l = 0; l < num_levels_; ++l) {
        for (size_t d = 0; d < d_model_; ++d)
            level_scalars_[l][0][d] -= learning_rate * grad_level_scalars_[l][0][d];
        for (size_t i = 0; i < W1_[l].rows; ++i)
            for (size_t j = 0; j < W1_[l].cols; ++j)
                W1_[l][i][j] -= learning_rate * grad_W1_[l][i][j];
        for (size_t k = 0; k < b1_[l].cols; ++k)
            b1_[l][0][k] -= learning_rate * grad_b1_[l][0][k];
        for (size_t i = 0; i < W2_[l].rows; ++i)
            for (size_t j = 0; j < W2_[l].cols; ++j)
                W2_[l][i][j] -= learning_rate * grad_W2_[l][i][j];
        for (size_t k = 0; k < b2_[l].cols; ++k)
            b2_[l][0][k] -= learning_rate * grad_b2_[l][0][k];
    }
    for (size_t i = 0; i < W_o.rows; ++i)
        for (size_t j = 0; j < W_o.cols; ++j)
            W_o[i][j] -= learning_rate * grad_W_o[i][j];
    for (size_t k = 0; k < b_o.cols; ++k)
        b_o[0][k] -= learning_rate * grad_b_o[0][k];
}

void FocalModulation::zero_grad() {
    for (size_t l = 0; l < num_levels_; ++l) {
        grad_level_scalars_[l].fill(0.0);
        grad_W1_[l].fill(0.0);
        grad_b1_[l].fill(0.0);
        grad_W2_[l].fill(0.0);
        grad_b2_[l].fill(0.0);
    }
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
}

std::vector<Tensor*> FocalModulation::parameters() {
    std::vector<Tensor*> out;
    for (size_t l = 0; l < num_levels_; ++l) {
        out.push_back(&level_scalars_[l]);
        out.push_back(&W1_[l]);
        out.push_back(&b1_[l]);
        out.push_back(&W2_[l]);
        out.push_back(&b2_[l]);
    }
    out.push_back(&W_o);
    out.push_back(&b_o);
    return out;
}

std::vector<Tensor*> FocalModulation::gradients() {
    std::vector<Tensor*> out;
    for (size_t l = 0; l < num_levels_; ++l) {
        out.push_back(&grad_level_scalars_[l]);
        out.push_back(&grad_W1_[l]);
        out.push_back(&grad_b1_[l]);
        out.push_back(&grad_W2_[l]);
        out.push_back(&grad_b2_[l]);
    }
    out.push_back(&grad_W_o);
    out.push_back(&grad_b_o);
    return out;
}

// ============================================================================
// FocalModulationBlock
// ============================================================================

FocalModulationBlock::FocalModulationBlock(size_t d_model, size_t seq_len,
                                           size_t num_levels,
                                           size_t mlp_dim,
                                           size_t ffn_mult)
    : d_model_(d_model),
      seq_len_(seq_len),
      ffn_mult_(ffn_mult),
      has_ffn_(ffn_mult > 0),
      ln1_(d_model),
      ln2_(d_model),
      focal_(d_model, seq_len, num_levels, mlp_dim) {
    if (d_model == 0)
        throw std::invalid_argument("FocalModulationBlock: d_model must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("FocalModulationBlock: seq_len must be > 0");
    if (num_levels == 0)
        throw std::invalid_argument("FocalModulationBlock: num_levels must be > 0");

    if (has_ffn_) {
        size_t H = ffn_mult * d_model_;
        ffn_W1_ = Tensor(H, d_model_);
        ffn_b1_ = Tensor(1, H);
        ffn_W2_ = Tensor(d_model_, H);
        ffn_b2_ = Tensor(1, d_model_);
        grad_ffn_W1_ = Tensor(H, d_model_);
        grad_ffn_b1_ = Tensor(1, H);
        grad_ffn_W2_ = Tensor(d_model_, H);
        grad_ffn_b2_ = Tensor(1, d_model_);
        grad_ffn_W1_.fill(0.0);
        grad_ffn_b1_.fill(0.0);
        grad_ffn_W2_.fill(0.0);
        grad_ffn_b2_.fill(0.0);
        std::mt19937 gen(43);
        double std1 = std::sqrt(1.0 / (double)d_model_);
        std::normal_distribution<> n1(0.0, std1);
        for (size_t i = 0; i < ffn_W1_.rows; ++i)
            for (size_t j = 0; j < ffn_W1_.cols; ++j)
                ffn_W1_[i][j] = n1(gen);
        ffn_b1_.fill(0.0);
        for (size_t i = 0; i < ffn_W2_.rows; ++i)
            for (size_t j = 0; j < ffn_W2_.cols; ++j)
                ffn_W2_[i][j] = n1(gen);
        ffn_b2_.fill(0.0);
    }
}

Tensor FocalModulationBlock::forward(const Tensor& input) {
    size_t B = input.rows;
    size_t D = d_model_;
    size_t S = seq_len_;

    // z1 = LN1(x). Reshape per-token: (B, S*D) → (B*S, D) → (B, S*D).
    Tensor z1_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z1_flat[b * S + s][d] = input[b][s * D + d];
    Tensor z1_norm = ln1_.forward(z1_flat);
    Tensor z1_pre(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z1_pre[b][s * D + d] = z1_norm[b * S + s][d];
    last_z1_ = z1_pre.clone();

    // focal_out = focal(z1)
    Tensor focal_out = focal_.forward(z1_pre);
    last_focal_out_ = focal_out.clone();

    // x' = x + focal_out
    Tensor x_prime(B, S * D);
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < S * D; ++j)
            x_prime[i][j] = input[i][j] + focal_out[i][j];

    if (!has_ffn_) {
        last_d_input_ = Tensor(0, 0);
        return x_prime;
    }

    // z2 = LN2(x'). Same reshape-per-token as z1.
    Tensor z2_flat_input(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z2_flat_input[b * S + s][d] = x_prime[b][s * D + d];
    Tensor z2_norm = ln2_.forward(z2_flat_input);
    Tensor z2_pre(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z2_pre[b][s * D + d] = z2_norm[b * S + s][d];
    last_z2_ = z2_pre.clone();

    // FFN: h = GELU(z2 @ W1^T + b1), out = h @ W2^T + b2
    size_t H = ffn_W1_.rows;
    Tensor ffn_h_pre(B * S, H);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t h = 0; h < H; ++h) {
                double acc = ffn_b1_[0][h];
                for (size_t d = 0; d < D; ++d)
                    acc += z2_pre[b][s * D + d] * ffn_W1_[h][d];
                ffn_h_pre[b * S + s][h] = acc;
            }
        }
    }
    Tensor ffn_h_act = ffn_h_pre.apply([](double v) { return gelu_scalar(v); });
    last_ffn_h_act_ = ffn_h_act.clone();

    Tensor ffn_out(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = ffn_b2_[0][d];
                for (size_t h = 0; h < H; ++h)
                    acc += ffn_h_act[b * S + s][h] * ffn_W2_[d][h];
                ffn_out[b][s * D + d] = acc;
            }
        }
    }
    last_ffn_out_ = ffn_out.clone();

    // out = x_prime + ffn_out
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < S * D; ++j)
            ffn_out[b][j] += x_prime[b][j];
    }
    return ffn_out;
}

Tensor FocalModulationBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = grad_output.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // d_x_prime = d_focal_out_path (residual) + from FFN
    // If no FFN: d_x_prime = grad_output (residual); d_focal_out = grad_output.
    if (!has_ffn_) {
        Tensor d_focal_out = grad_output;
        // focal backward
        Tensor d_z1 = focal_.backward(d_focal_out, 0.0);
        // Reshape d_z1 from (B, S*D) to (B*S, D) for LN1.backward.
        Tensor d_z1_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_z1_flat[b * S + s][d] = d_z1[b][s * D + d];
        Tensor d_x_flat = ln1_.backward(d_z1_flat, 0.0);
        Tensor d_x(B, S * D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_x[b][s * D + d] = d_x_flat[b * S + s][d];
        // d_input = grad_output (residual) + d_x
        Tensor d_input(B, S * D);
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < S * D; ++j)
                d_input[i][j] = grad_output[i][j] + d_x[i][j];
        last_d_input_ = d_input;
        return d_input;
    }

    // With FFN
    // d_ffn_out = grad_output (residual path through FFN's output)
    Tensor d_ffn_out = grad_output;
    // d_x_prime contribution from this block: grad_output (residual) plus
    // FFN-derived contribution to x_prime.
    size_t H = ffn_W1_.rows;

    // FFN backward: d_h_act, d_h_pre, then chain to z2_pre, ln2, x_prime.
    grad_ffn_W1_.fill(0.0);
    grad_ffn_b1_.fill(0.0);
    grad_ffn_W2_.fill(0.0);
    grad_ffn_b2_.fill(0.0);

    // Reshape to (B*S, H) for h_act.
    Tensor d_h_act(B * S, H);
    d_h_act.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double go = d_ffn_out[b][s * D + d];
                if (go == 0.0) continue;
                for (size_t h = 0; h < H; ++h) {
                    grad_ffn_W2_[d][h] += go * last_ffn_h_act_[b * S + s][h];
                    d_h_act[b * S + s][h] += go * ffn_W2_[d][h];
                }
            }
        }
    }
    for (size_t b = 0; b < B; ++b)
        (void)b;
    // d_h_pre = d_h_act * GELU'
    constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
    auto gelu_grad = [&](double x) -> double {
        double xc = std::max(-4.0, std::min(4.0, x));
        double inner = SQRT_2_OVER_PI * (xc + 0.044715 * xc * xc * xc);
        double t = std::tanh(inner);
        double cdf = 0.5 * (1.0 + t);
        double dinner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * xc * xc);
        double sech2 = 1.0 - t * t;
        return cdf + xc * 0.5 * sech2 * dinner;
    };
    Tensor d_h_pre(B * S, H);
    // (no pre-loop needed — we recompute h_pre below and use it directly.)
    // Recompute h_pre on the fly: same forward math as in `forward()`,
    // but here we cache it before applying GELU'. Use last_z2_.
    Tensor h_pre(B * S, H);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t h = 0; h < H; ++h) {
                double acc = ffn_b1_[0][h];
                for (size_t d = 0; d < D; ++d)
                    acc += last_z2_[b][s * D + d] * ffn_W1_[h][d];
                h_pre[b * S + s][h] = acc;
            }
        }
    }
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t h = 0; h < H; ++h) {
                double dh = d_h_act[b * S + s][h] * gelu_grad(h_pre[b * S + s][h]);
                d_h_pre[b * S + s][h] = dh;
                grad_ffn_b1_[0][h] += dh;
                for (size_t d = 0; d < D; ++d)
                    grad_ffn_W1_[h][d] += dh * last_z2_[b][s * D + d];
            }
        }
    }

    // d_z2 = d_h_pre @ W1  (B*S, H) @ (H, D) = (B*S, D)
    Tensor d_z2_flat(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t h = 0; h < H; ++h)
                    acc += d_h_pre[b * S + s][h] * ffn_W1_[h][d];
                d_z2_flat[b][s * D + d] = acc;
            }
        }
    }
    // grad_ffn_b2 from d_ffn_out
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t s = 0; s < S; ++s)
                grad_ffn_b2_[0][d] += d_ffn_out[b][s * D + d];
        }
    }
    // Reshape d_z2_flat from (B, S*D) to (B*S, D) for LN2.backward.
    Tensor d_z2_for_ln(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_z2_for_ln[b * S + s][d] = d_z2_flat[b][s * D + d];
    Tensor d_x_prime_from_ln2_flat = ln2_.backward(d_z2_for_ln, 0.0);
    Tensor d_x_prime_from_ln2(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_x_prime_from_ln2[b][s * D + d] = d_x_prime_from_ln2_flat[b * S + s][d];
    // d_x_prime = d_ffn_out (residual) + d_x_prime_from_ln2
    Tensor d_x_prime(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < S * D; ++j)
            d_x_prime[b][j] = d_ffn_out[b][j] + d_x_prime_from_ln2[b][j];
    }
    // x' = x + focal_out  → d_x += d_x_prime; d_focal_out += d_x_prime.
    Tensor d_focal_out = d_x_prime.clone();
    // focal backward
    Tensor d_z1 = focal_.backward(d_focal_out, 0.0);
    // Reshape d_z1 from (B, S*D) to (B*S, D) for LN1.backward, then reshape result.
    Tensor d_z1_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_z1_flat[b * S + s][d] = d_z1[b][s * D + d];
    Tensor d_x_from_ln1_flat = ln1_.backward(d_z1_flat, 0.0);
    Tensor d_x_from_ln1(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_x_from_ln1[b][s * D + d] = d_x_from_ln1_flat[b * S + s][d];
    // d_input = d_x_prime (residual) + d_x_from_ln1
    Tensor d_input(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < S * D; ++j)
            d_input[b][j] = d_x_prime[b][j] + d_x_from_ln1[b][j];
    }
    last_d_input_ = d_input;
    return d_input;
}

void FocalModulationBlock::update_weights(double learning_rate) {
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    focal_.update_weights(learning_rate);
    if (has_ffn_) {
        for (size_t i = 0; i < ffn_W1_.rows; ++i)
            for (size_t j = 0; j < ffn_W1_.cols; ++j)
                ffn_W1_[i][j] -= learning_rate * grad_ffn_W1_[i][j];
        for (size_t k = 0; k < ffn_b1_.cols; ++k)
            ffn_b1_[0][k] -= learning_rate * grad_ffn_b1_[0][k];
        for (size_t i = 0; i < ffn_W2_.rows; ++i)
            for (size_t j = 0; j < ffn_W2_.cols; ++j)
                ffn_W2_[i][j] -= learning_rate * grad_ffn_W2_[i][j];
        for (size_t k = 0; k < ffn_b2_.cols; ++k)
            ffn_b2_[0][k] -= learning_rate * grad_ffn_b2_[0][k];
    }
}

void FocalModulationBlock::zero_grad() {
    ln1_.zero_grad();
    ln2_.zero_grad();
    focal_.zero_grad();
    if (has_ffn_) {
        grad_ffn_W1_.fill(0.0);
        grad_ffn_b1_.fill(0.0);
        grad_ffn_W2_.fill(0.0);
        grad_ffn_b2_.fill(0.0);
    }
}

std::vector<Tensor*> FocalModulationBlock::parameters() {
    std::vector<Tensor*> out;
    auto p = focal_.parameters();
    out.insert(out.end(), p.begin(), p.end());
    // LayerNorm params
    out.push_back(&ln1_.gamma);
    out.push_back(&ln1_.beta);
    if (has_ffn_) {
        out.push_back(&ffn_W1_);
        out.push_back(&ffn_b1_);
        out.push_back(&ffn_W2_);
        out.push_back(&ffn_b2_);
    }
    out.push_back(&ln2_.gamma);
    out.push_back(&ln2_.beta);
    return out;
}

std::vector<Tensor*> FocalModulationBlock::gradients() {
    std::vector<Tensor*> out;
    auto g = focal_.gradients();
    out.insert(out.end(), g.begin(), g.end());
    out.push_back(&ln1_.grad_gamma_);
    out.push_back(&ln1_.grad_beta_);
    if (has_ffn_) {
        out.push_back(&grad_ffn_W1_);
        out.push_back(&grad_ffn_b1_);
        out.push_back(&grad_ffn_W2_);
        out.push_back(&grad_ffn_b2_);
    }
    out.push_back(&ln2_.grad_gamma_);
    out.push_back(&ln2_.grad_beta_);
    return out;
}

Tensor FocalModulationBlock::get_weights() const { return focal_.get_weights(); }
Tensor FocalModulationBlock::get_gradients() const { return focal_.get_gradients(); }

// ============================================================================
// FocalModulationModel
// ============================================================================

FocalModulationModel::FocalModulationModel(size_t input_dim, size_t d_model, size_t output_dim,
                                           size_t seq_len,
                                           size_t num_blocks,
                                           size_t num_levels,
                                           size_t mlp_dim,
                                           size_t ffn_mult)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      seq_len_(seq_len),
      num_blocks_(num_blocks),
      input_proj_(input_dim, d_model),
      head_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0)
        throw std::invalid_argument("FocalModulationModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("FocalModulationModel: d_model must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("FocalModulationModel: output_dim must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("FocalModulationModel: seq_len must be > 0");
    if (num_blocks == 0)
        throw std::invalid_argument("FocalModulationModel: num_blocks must be > 0");

    blocks_.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i)
        blocks_.emplace_back(d_model_, seq_len, num_levels, mlp_dim, ffn_mult);
}

Tensor FocalModulationModel::forward(const Tensor& input) {
    size_t B = input.rows;
    last_input_ = input.clone();
    // Apply input_proj (Dense input_dim → d_model) per-token. Input layout
    // is (B, S*input_dim) flat; per-token indices are (b, s, i) at index
    // b*(S*input_dim) + s*input_dim + i.
    Tensor x_proj(B, seq_len_ * d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t d = 0; d < d_model_; ++d) {
                double acc = input_proj_.bias[0][d];
                for (size_t i = 0; i < input_dim_; ++i)
                    acc += input[b][s * input_dim_ + i] * input_proj_.weights[d][i];
                x_proj[b][s * d_model_ + d] = acc;
            }
        }
    }
    last_z_proj_ = x_proj.clone();

    // Stack of blocks
    Tensor cur = x_proj;
    for (size_t i = 0; i < num_blocks_; ++i) {
        cur = blocks_[i].forward(cur);
    }
    last_blocks_out_ = cur.clone();

    // Final LN (per-token). Reshape cur (B, S*D) → (B*S, D), apply head LN,
    // reshape back.
    Tensor head_flat(B * seq_len_, d_model_);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t d = 0; d < d_model_; ++d)
                head_flat[b * seq_len_ + s][d] = cur[b][s * d_model_ + d];
    Tensor head_norm = head_ln_.forward(head_flat);
    Tensor head(B, seq_len_ * d_model_);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t d = 0; d < d_model_; ++d)
                head[b][s * d_model_ + d] = head_norm[b * seq_len_ + s][d];
    last_head_ln_ = head.clone();

    // Mean-pool over S → (B, d_model)
    Tensor pooled(B, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < d_model_; ++d) {
            double sum = 0.0;
            for (size_t s = 0; s < seq_len_; ++s)
                sum += head[b][s * d_model_ + d];
            pooled[b][d] = sum / static_cast<double>(seq_len_);
        }
    }
    last_pooled_ = pooled.clone();

    // Classifier (Dense d_model → output_dim)
    Tensor logits = classifier_.forward(pooled);
    last_logits_ = logits.clone();
    return logits;
}

Tensor FocalModulationModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = grad_output.rows;
    size_t S = seq_len_;

    // classifier backward gives d_pooled
    Tensor d_pooled = classifier_.backward(grad_output, 0.0);

    // d_head: broadcast mean-pool gradient. For each (b, d) the gradient
    // contribution is d_pooled[b, d] / S; for each (b, s, d) it is the
    // same. So d_head[b, s*d_model + d] = d_pooled[b, d] / S.
    Tensor d_head(B, S * d_model_);
    double inv_S = 1.0 / static_cast<double>(S);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < d_model_; ++d)
                d_head[b][s * d_model_ + d] = d_pooled[b][d] * inv_S;
        }
    }

    // ln forward via head_ln, then backward. Need same per-token reshape.
    Tensor d_head_flat_input(B * S, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < d_model_; ++d)
                d_head_flat_input[b * S + s][d] = d_head[b][s * d_model_ + d];
        }
    }
    Tensor d_blocks_out_flat = head_ln_.backward(d_head_flat_input, 0.0);
    Tensor d_blocks_out(B, S * d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < d_model_; ++d)
                d_blocks_out[b][s * d_model_ + d] = d_blocks_out_flat[b * S + s][d];
        }
    }

    // Stack backward: blocks in reverse
    Tensor d_cur = d_blocks_out;
    for (size_t i = num_blocks_; i > 0; --i) {
        d_cur = blocks_[i - 1].backward(d_cur, 0.0);
    }

    // input_proj backward — manual reshape version (we did manual forward).
    // Forward: for each (b, s) and each d_out:
    //   x_proj[b, s, d_out] = bias[d_out] + sum_i input[b, s, i] * W[d_out, i]
    // Backward:
    //   d_W[d_out, i] += sum_{b, s} d_cur[b, s, d_out] * input[b, s, i]
    //   d_b[d_out]    += sum_{b, s} d_cur[b, s, d_out]
    //   d_input[b, s, i] = sum_{d_out} d_cur[b, s, d_out] * W[d_out, i]
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < d_model_; ++d) {
                double g = d_cur[b][s * d_model_ + d];
                if (g == 0.0) continue;
                for (size_t i = 0; i < input_dim_; ++i)
                    input_proj_.grad_weights[d][i] += g * last_input_[b][s * input_dim_ + i];
                input_proj_.grad_bias[0][d] += g;
            }
        }
    }
    Tensor d_input(B, S * input_dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t i = 0; i < input_dim_; ++i) {
                double acc = 0.0;
                for (size_t d = 0; d < d_model_; ++d)
                    acc += d_cur[b][s * d_model_ + d] * input_proj_.weights[d][i];
                d_input[b][s * input_dim_ + i] = acc;
            }
        }
    }
    last_d_input_ = d_input;
    return d_input;
}

void FocalModulationModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    head_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
    for (size_t i = 0; i < num_blocks_; ++i)
        blocks_[i].update_weights(learning_rate);
}

void FocalModulationModel::zero_grad() {
    input_proj_.zero_grad();
    head_ln_.zero_grad();
    classifier_.zero_grad();
    for (size_t i = 0; i < num_blocks_; ++i)
        blocks_[i].zero_grad();
}

std::vector<Tensor*> FocalModulationModel::parameters() {
    std::vector<Tensor*> out;
    auto p = input_proj_.parameters();
    out.insert(out.end(), p.begin(), p.end());
    for (size_t i = 0; i < num_blocks_; ++i) {
        auto bp = blocks_[i].parameters();
        out.insert(out.end(), bp.begin(), bp.end());
    }
    auto hp = head_ln_.parameters();
    out.insert(out.end(), hp.begin(), hp.end());
    auto cp = classifier_.parameters();
    out.insert(out.end(), cp.begin(), cp.end());
    return out;
}

std::vector<Tensor*> FocalModulationModel::gradients() {
    std::vector<Tensor*> out;
    auto g = input_proj_.gradients();
    out.insert(out.end(), g.begin(), g.end());
    for (size_t i = 0; i < num_blocks_; ++i) {
        auto bg = blocks_[i].gradients();
        out.insert(out.end(), bg.begin(), bg.end());
    }
    auto hg = head_ln_.gradients();
    out.insert(out.end(), hg.begin(), hg.end());
    auto cg = classifier_.gradients();
    out.insert(out.end(), cg.begin(), cg.end());
    return out;
}
