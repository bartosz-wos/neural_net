#include "mlp_mixer.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Shared-Dense-applied-per-row helpers
// ============================================================================
//
// The MLP-Mixer block uses the same Dense layer independently for each row of
// an input tensor. For a tensor X of shape (N, D_in), the forward produces
// Y of shape (N, D_out) where Y[n] = Dense.forward(X[n]). For backward, the
// gradient of Y must produce:
//   - a gradient of shape (N, D_in) — the input gradient
//   - an accumulated gradient on the Dense weights/bias (summed over rows)
// ============================================================================

namespace {

// Apply the same Dense to each row of `x` (shape N x D_in) → return shape N x D_out.
// This calls Dense.forward once per row. The Dense's internal cached state
// (last_input) is overwritten on every row — which is fine because we never
// call Dense.backward; we do the per-row backward manually in
// `mlp2_per_row_backward`.
Tensor dense_per_row_forward(const Tensor& x, Dense& d, size_t D_in, size_t D_out) {
    if (x.cols != D_in) {
        throw std::runtime_error("dense_per_row_forward: input cols mismatch");
    }
    Tensor y(x.rows, D_out);
    for (size_t n = 0; n < x.rows; ++n) {
        Tensor row_in(1, D_in);
        for (size_t j = 0; j < D_in; ++j) row_in(0, j) = x(n, j);
        Tensor row_out = d.forward(row_in);
        for (size_t j = 0; j < D_out; ++j) y(n, j) = row_out(0, j);
    }
    return y;
}

// Two-layer MLP per-row: y = W2^T (GELU(W1^T x + b1)) + b2
// Returns grad_x of shape (N, D_in) and accumulates grad_W1, grad_b1, grad_W2, grad_b2.
struct MLP2Weights { Dense* w1; Dense* w2; };

Tensor mlp2_per_row_backward(const Tensor& grad_y,
                             const MLP2Weights& mlp,
                             const Tensor& cached_x,    // (N, D_in)
                             const Tensor& cached_h,    // (N, H) — W1^T x + b1
                             const Tensor& cached_u,    // (N, H) — GELU(h)
                             size_t D_in, size_t H) {
    // Forward:
    //   h = W1^T x + b1     (N, H)
    //   u = GELU(h)         (N, H)
    //   y = W2^T u + b2     (N, D_in)
    //
    // Backward:
    //   grad_u = grad_y @ W2  (N, H)
    //   grad_h = grad_u * GELU'(h)
    //   grad_x = grad_h @ W1  (N, D_in)
    //
    // Accumulate on w2.grad_weights, w2.grad_bias, w1.grad_weights, w1.grad_bias.
    const Tensor& W2 = mlp.w2->weights;
    Tensor grad_u(grad_y.rows, H);
    for (size_t n = 0; n < grad_y.rows; ++n) {
        for (size_t j = 0; j < H; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < D_in; ++k) {
                acc += grad_y(n, k) * W2(k, j);
            }
            grad_u(n, j) = acc;
        }
    }
    // grad_h = grad_u * GELU'(h)
    Tensor grad_h(grad_y.rows, H);
    for (size_t n = 0; n < grad_y.rows; ++n) {
        for (size_t j = 0; j < H; ++j) {
            grad_h(n, j) = grad_u(n, j) * GELU{}.derivative(cached_h(n, j));
        }
    }
    // grad_x = grad_h @ W1   (W1 has shape (H, D_in))
    const Tensor& W1 = mlp.w1->weights;
    Tensor grad_x(grad_y.rows, D_in);
    for (size_t n = 0; n < grad_y.rows; ++n) {
        for (size_t j = 0; j < D_in; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < H; ++k) {
                acc += grad_h(n, k) * W1(k, j);
            }
            grad_x(n, j) = acc;
        }
    }
    // Accumulate w2.grad_weights (D_in, H): += grad_y^T @ cached_u
    // Accumulate w2.grad_bias (1, D_in): bias is (1, out_features=D_in)
    for (size_t d_out = 0; d_out < D_in; ++d_out) {
        double b_acc = 0.0;
        for (size_t n = 0; n < grad_y.rows; ++n) {
            b_acc += grad_y(n, d_out);
        }
        mlp.w2->grad_bias(0, d_out) += b_acc;
        for (size_t j = 0; j < H; ++j) {
            double acc = 0.0;
            for (size_t n = 0; n < grad_y.rows; ++n) {
                acc += grad_y(n, d_out) * cached_u(n, j);
            }
            mlp.w2->grad_weights(d_out, j) += acc;
        }
    }
    // Accumulate w1.grad_weights (H, D_in): += grad_h^T @ cached_x
    // Accumulate w1.grad_bias (1, H): bias is (1, out_features=H)
    for (size_t h = 0; h < H; ++h) {
        double b_acc = 0.0;
        for (size_t n = 0; n < grad_y.rows; ++n) {
            b_acc += grad_h(n, h);
        }
        mlp.w1->grad_bias(0, h) += b_acc;
        for (size_t j = 0; j < D_in; ++j) {
            double acc = 0.0;
            for (size_t n = 0; n < grad_y.rows; ++n) {
                acc += grad_h(n, h) * cached_x(n, j);
            }
            mlp.w1->grad_weights(h, j) += acc;
        }
    }
    return grad_x;
}

// Per-token LayerNorm: input shape (B, S*D), applies LayerNorm(D) over each
// of the B*S rows of length D. Output shape (B, S*D).
//
// IMPORTANT: We do NOT reuse the repo's `LayerNorm.forward`/`backward` here,
// because that class is designed for a single batch call (one row per batch
// element, mean/var per row over features). Calling it per-row from inside
// these helpers would corrupt last_x/last_mean/last_var (each call overwrites
// them) and accumulate grad_gamma/grad_beta using only the LAST row's state.
// Instead we compute mean/var per row from the cached pre-LN input (`x`) and
// do the LN math directly. grad_gamma and grad_beta are accumulated into the
// LayerNorm's existing grad_gamma_/grad_beta_ buffers so the optimizer can
// still find them via parameters()/gradients().
Tensor ln_per_token_forward(const Tensor& x, LayerNorm& ln, size_t S, size_t D) {
    size_t B = x.rows;
    if (x.cols != S * D) {
        throw std::runtime_error("ln_per_token_forward: x.cols must be S*D");
    }
    Tensor out(B, S * D);
    const double eps = ln.eps;
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            // Compute mean and var for this row (length D).
            double mean = 0.0;
            for (size_t j = 0; j < D; ++j) mean += x(b, s * D + j);
            mean /= static_cast<double>(D);
            double var = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double d = x(b, s * D + j) - mean;
                var += d * d;
            }
            var = std::max(var / static_cast<double>(D), 1e-7);
            double sqrt_var = std::sqrt(var + eps);
            // Apply gamma * (x - mean) / sqrt_var + beta.
            for (size_t j = 0; j < D; ++j) {
                double norm = (x(b, s * D + j) - mean) / sqrt_var;
                out(b, s * D + j) = ln.gamma(0, j) * norm + ln.beta(0, j);
            }
        }
    }
    return out;
}

Tensor ln_per_token_backward(const Tensor& grad_out, const Tensor& cached_x,
                             LayerNorm& ln, size_t S, size_t D) {
    // grad_out shape (B, S*D), cached_x shape (B, S*D) — the pre-LN input.
    // Returns grad_in shape (B, S*D). Accumulates grad_gamma/grad_beta on ln.
    size_t B = grad_out.rows;
    if (grad_out.cols != S * D || cached_x.cols != S * D) {
        throw std::runtime_error("ln_per_token_backward: shape mismatch");
    }
    const double eps = ln.eps;
    // Ensure grad buffers exist (para-layered with the LayerNorm constructor).
    if (ln.grad_gamma_.rows == 0) ln.grad_gamma_ = Tensor(1, D);
    if (ln.grad_beta_.rows == 0)  ln.grad_beta_  = Tensor(1, D);

    Tensor grad_in(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            // Recompute mean/var for this row from cached_x.
            double mean = 0.0;
            for (size_t j = 0; j < D; ++j) mean += cached_x(b, s * D + j);
            mean /= static_cast<double>(D);
            double var = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double d = cached_x(b, s * D + j) - mean;
                var += d * d;
            }
            var = std::max(var / static_cast<double>(D), 1e-7);
            double sqrt_var = std::sqrt(var + eps);
            double inv_var = 1.0 / sqrt_var;

            // Accumulate grad_gamma[f] += grad_out[f] * norm[f]
            // Accumulate grad_beta[f] += grad_out[f]
            for (size_t j = 0; j < D; ++j) {
                double norm = (cached_x(b, s * D + j) - mean) * inv_var;
                ln.grad_gamma_(0, j) += grad_out(b, s * D + j) * norm;
                ln.grad_beta_(0, j)  += grad_out(b, s * D + j);
            }

            // dNorm[f] = grad_out[f] * gamma[f]
            // dVar = sum_f dNorm[f] * (x[f] - mean) * (-0.5) * inv_var^3
            // dMu  = -sum_f dNorm[f] * inv_var
            // grad_x[f] = dNorm[f] * inv_var + dVar * 2 * (x[f] - mean) / D + dMu / D
            double dVar = 0.0, dMu = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double dNorm = grad_out(b, s * D + j) * ln.gamma(0, j);
                dVar += dNorm * (cached_x(b, s * D + j) - mean);
                dMu  -= dNorm * inv_var;
            }
            dVar *= -0.5 * inv_var * inv_var * inv_var;
            for (size_t j = 0; j < D; ++j) {
                double dNorm = grad_out(b, s * D + j) * ln.gamma(0, j);
                double dx_norm = dNorm * inv_var;
                double dx_var = dVar * 2.0 * (cached_x(b, s * D + j) - mean) / static_cast<double>(D);
                double dx_mu = dMu / static_cast<double>(D);
                grad_in(b, s * D + j) = dx_norm + dx_var + dx_mu;
            }
        }
    }
    return grad_in;
}

// Per-channel LayerNorm: input shape (B, S*D). For each batch, and for each
// channel j ∈ {0..D-1}, normalize the S values {x[b, s*D + j] : s ∈ {0..S-1}}
// using mean/var over S. As with the per-token LN, we do the math directly
// (see ln_per_token_forward for the reasoning).
Tensor ln_per_channel_forward(const Tensor& x, LayerNorm& ln, size_t S, size_t D) {
    size_t B = x.rows;
    if (x.cols != S * D) {
        throw std::runtime_error("ln_per_channel_forward: x.cols must be S*D");
    }
    Tensor out(B, S * D);
    const double eps = ln.eps;
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < D; ++j) {
            // Compute mean/var over the S values belonging to channel j.
            double mean = 0.0;
            for (size_t s = 0; s < S; ++s) mean += x(b, s * D + j);
            mean /= static_cast<double>(S);
            double var = 0.0;
            for (size_t s = 0; s < S; ++s) {
                double d = x(b, s * D + j) - mean;
                var += d * d;
            }
            var = std::max(var / static_cast<double>(S), 1e-7);
            double sqrt_var = std::sqrt(var + eps);
            for (size_t s = 0; s < S; ++s) {
                double norm = (x(b, s * D + j) - mean) / sqrt_var;
                out(b, s * D + j) = ln.gamma(0, s) * norm + ln.beta(0, s);
            }
        }
    }
    return out;
}

Tensor ln_per_channel_backward(const Tensor& grad_out, const Tensor& cached_x,
                               LayerNorm& ln, size_t S, size_t D) {
    // grad_out shape (B, S*D), cached_x shape (B, S*D) — pre-LN input.
    // Returns grad_in shape (B, S*D). Each LN row is length S (one channel),
    // so the gamma buffer has length S and grad_gamma_/grad_beta_ accumulate
    // over channels (D) and over batch (B).
    size_t B = grad_out.rows;
    if (grad_out.cols != S * D || cached_x.cols != S * D) {
        throw std::runtime_error("ln_per_channel_backward: shape mismatch");
    }
    const double eps = ln.eps;
    if (ln.grad_gamma_.rows == 0) ln.grad_gamma_ = Tensor(1, S);
    if (ln.grad_beta_.rows == 0)  ln.grad_beta_  = Tensor(1, S);

    Tensor grad_in(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < D; ++j) {
            double mean = 0.0;
            for (size_t s = 0; s < S; ++s) mean += cached_x(b, s * D + j);
            mean /= static_cast<double>(S);
            double var = 0.0;
            for (size_t s = 0; s < S; ++s) {
                double d = cached_x(b, s * D + j) - mean;
                var += d * d;
            }
            var = std::max(var / static_cast<double>(S), 1e-7);
            double sqrt_var = std::sqrt(var + eps);
            double inv_var = 1.0 / sqrt_var;

            // gamma has length S — index `s` over the S values of the channel.
            for (size_t s = 0; s < S; ++s) {
                double norm = (cached_x(b, s * D + j) - mean) * inv_var;
                ln.grad_gamma_(0, s) += grad_out(b, s * D + j) * norm;
                ln.grad_beta_(0, s)  += grad_out(b, s * D + j);
            }

            double dVar = 0.0, dMu = 0.0;
            for (size_t s = 0; s < S; ++s) {
                double dNorm = grad_out(b, s * D + j) * ln.gamma(0, s);
                dVar += dNorm * (cached_x(b, s * D + j) - mean);
                dMu  -= dNorm * inv_var;
            }
            dVar *= -0.5 * inv_var * inv_var * inv_var;
            for (size_t s = 0; s < S; ++s) {
                double dNorm = grad_out(b, s * D + j) * ln.gamma(0, s);
                double dx_norm = dNorm * inv_var;
                double dx_var = dVar * 2.0 * (cached_x(b, s * D + j) - mean) / static_cast<double>(S);
                double dx_mu = dMu / static_cast<double>(S);
                grad_in(b, s * D + j) = dx_norm + dx_var + dx_mu;
            }
        }
    }
    return grad_in;
}

} // anonymous namespace

// ============================================================================
// MlpMixerBlock
// ============================================================================

MlpMixerBlock::MlpMixerBlock(size_t dim, size_t seq_len,
                             size_t token_dim, size_t channel_dim)
    : dim_(dim), seq_len_(seq_len),
      token_dim_(token_dim), channel_dim_(channel_dim),
      ln_token_(dim),
      ln_channel_(seq_len),
      tok_mlp_w1_(seq_len, token_dim),     // Dense(in=seq_len, out=token_dim)
      tok_mlp_w2_(token_dim, seq_len),     // Dense(in=token_dim, out=seq_len)
      chan_mlp_w1_(dim, channel_dim),      // Dense(in=dim, out=channel_dim)
      chan_mlp_w2_(channel_dim, dim)       // Dense(in=channel_dim, out=dim)
{
    if (dim == 0 || seq_len == 0 || token_dim == 0 || channel_dim == 0) {
        throw std::invalid_argument("MlpMixerBlock: all dims must be positive");
    }
}

Tensor MlpMixerBlock::forward(const Tensor& input) {
    if (input.cols != seq_len_ * dim_) {
        throw std::runtime_error("MlpMixerBlock::forward: input.cols must be S*D");
    }
    last_input_ = input.clone();

    // ----- Step 1: LayerNorm per-token (over D) -----
    Tensor z = ln_per_token_forward(input, ln_token_, seq_len_, dim_);
    last_z_ = z.clone();

    // ----- Step 2: Token-mixing MLP -----
    // Reshape z to (B*dim, seq_len): for each batch b, each dim j, take the
    // token sequence (S values across patches). Apply 2-layer MLP with shared
    // weights, hidden width token_dim.
    size_t B = input.rows;
    Tensor z_tok_view(B * dim_, seq_len_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            for (size_t s = 0; s < seq_len_; ++s) {
                z_tok_view(b * dim_ + j, s) = z(b, s * dim_ + j);
            }
        }
    }
    // h1 = tok_mlp_w1.forward(z_tok_view): (B*dim, token_dim)
    Tensor h1 = dense_per_row_forward(z_tok_view, tok_mlp_w1_, seq_len_, token_dim_);
    // Apply GELU. Cache pre-activation (h1) and post-GELU (h1_gelu).
    Tensor h1_gelu = h1.clone();
    for (size_t i = 0; i < h1.data.size(); ++i) {
        h1_gelu.data[i] = GELU{}(h1.data[i]);
    }
    last_h1_ = h1.clone();
    last_h1_gelu_ = h1_gelu.clone();
    // tok_pre = tok_mlp_w2.forward(h1_gelu): (B*dim, seq_len)
    Tensor tok_pre_view = dense_per_row_forward(h1_gelu, tok_mlp_w2_, token_dim_, seq_len_);
    // Reshape back to (B, S*D)
    Tensor tok_pre(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            for (size_t s = 0; s < seq_len_; ++s) {
                tok_pre(b, s * dim_ + j) = tok_pre_view(b * dim_ + j, s);
            }
        }
    }
    last_tok_pre_ = tok_pre.clone();

    // ----- Step 3: Residual 1 -----
    Tensor y1(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < seq_len_ * dim_; ++i) {
            y1.data[b * (seq_len_ * dim_) + i] =
                z.data[b * (seq_len_ * dim_) + i] + tok_pre.data[b * (seq_len_ * dim_) + i];
        }
    }
    last_y1_ = y1.clone();

    // ----- Step 4: LayerNorm per-channel (over S) -----
    Tensor v = ln_per_channel_forward(y1, ln_channel_, seq_len_, dim_);
    last_v_ = v.clone();

    // ----- Step 5: Channel-mixing MLP -----
    // Apply 2-layer MLP with shared weights, hidden width channel_dim, per
    // row of length dim. Row count = B * S.
    Tensor chan_view(B * seq_len_, dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                chan_view(b * seq_len_ + s, j) = v(b, s * dim_ + j);
            }
        }
    }
    Tensor h2 = dense_per_row_forward(chan_view, chan_mlp_w1_, dim_, channel_dim_);
    Tensor h2_gelu = h2.clone();
    for (size_t i = 0; i < h2.data.size(); ++i) {
        h2_gelu.data[i] = GELU{}(h2.data[i]);
    }
    Tensor chan_pre_view = dense_per_row_forward(h2_gelu, chan_mlp_w2_, channel_dim_, dim_);
    last_h2_ = h2.clone();
    last_h2_gelu_ = h2_gelu.clone();
    // Reshape back to (B, S*D)
    Tensor chan_pre(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                chan_pre(b, s * dim_ + j) = chan_pre_view(b * seq_len_ + s, j);
            }
        }
    }
    last_chan_pre_ = chan_pre.clone();

    // ----- Step 6: Residual 2 -----
    Tensor out(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < seq_len_ * dim_; ++i) {
            out.data[b * (seq_len_ * dim_) + i] =
                y1.data[b * (seq_len_ * dim_) + i] + chan_pre.data[b * (seq_len_ * dim_) + i];
        }
    }
    last_output_ = out.clone();
    return out;
}

Tensor MlpMixerBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.cols != seq_len_ * dim_) {
        throw std::runtime_error("MlpMixerBlock::backward: grad_output shape mismatch");
    }
    size_t B = grad_output.rows;

    // ----- Step 6 (reverse): residual 2: out = y1 + chan_pre -----
    // grad_y1 (from residual) will be filled in after the channel-mix block:
    // grad_y1 = LN_channel.backward(grad_v_total) where
    //   grad_v_total = grad_chan_pre (residual) + grad_v_from_mlp (MLP backward).
    Tensor grad_chan_pre = grad_output.clone();  // d_chan_pre from residual

    // ----- Step 5 (reverse): channel-mix MLP backward -----
    // grad_chan_pre (B, S*D) → reshape to (B*S, D) → backward through chan MLP
    Tensor grad_chan_view(B * seq_len_, dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                grad_chan_view(b * seq_len_ + s, j) = grad_chan_pre(b, s * dim_ + j);
            }
        }
    }
    // Reshape cached last_v_ to (B*S, D) as the cached input.
    Tensor chan_view(B * seq_len_, dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                chan_view(b * seq_len_ + s, j) = last_v_(b, s * dim_ + j);
            }
        }
    }
    // ----- Step 5 (reverse): channel-mix MLP backward -----
    MLP2Weights chan_mlp{ &chan_mlp_w1_, &chan_mlp_w2_ };
    Tensor grad_v_from_mlp_view = mlp2_per_row_backward(
        grad_chan_view, chan_mlp, chan_view, last_h2_, last_h2_gelu_,
        /*D_in=*/dim_, /*H=*/channel_dim_);
    // grad_v_from_mlp_view shape (B*S, D). Reshape to (B, S*D).
    Tensor grad_v_from_mlp(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                grad_v_from_mlp(b, s * dim_ + j) =
                    grad_v_from_mlp_view(b * seq_len_ + s, j);
            }
        }
    }
    // Gradient at v: only the chan_mix MLP path contributes. The residual
    // `out = y1 + chan_pre` means dL/d_chan_pre = loss_grad (the upstream
    // gradient at chan_pre), and then dL/d_v is computed by the chan_mix MLP
    // backward from THAT — i.e. grad_v_from_mlp. The residual gradient does
    // NOT add directly to grad_v (it would be double-counting). The previous
    // implementation added grad_chan_pre here, which is the root-cause bug
    // that caused the LN_channel gradients (and ultimately the input
    // gradients) to disagree with the FD by ~50x.
    Tensor grad_v = grad_v_from_mlp;

    // ----- Step 4 (reverse): per-channel LN backward (on combined grad_v) -----
    // Pre-LN input to the per-channel LN was `last_y1_`.
    // The residual2 `out = y1 + chan_pre` gives dL/d_y1_residual = grad_output.
    // The LN_channel path adds dL/d_y1_from_v = LN_chan_backward(grad_v, last_y1_, ...).
    // The total gradient at y1 is the sum of both contributions.
    Tensor grad_y1_from_ln = ln_per_channel_backward(grad_v, last_y1_, ln_channel_, seq_len_, dim_);
    Tensor grad_y1(B, seq_len_ * dim_);
    for (size_t i = 0; i < grad_y1.data.size(); ++i) {
        grad_y1.data[i] = grad_output.data[i] + grad_y1_from_ln.data[i];
    }

    // ----- Step 3 (reverse): residual 1: y1 = z + tok_pre -----
    Tensor grad_z_from_res = grad_y1.clone();    // d_z from residual
    Tensor grad_tok_pre = grad_y1.clone();        // d_tok_pre from residual

    // ----- Step 1 (reverse): token-mix MLP backward -----
    // grad_tok_pre (B, S*D) → reshape to (B*dim, S) → backward through tok MLP
    Tensor grad_tok_view(B * dim_, seq_len_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            for (size_t s = 0; s < seq_len_; ++s) {
                grad_tok_view(b * dim_ + j, s) = grad_tok_pre(b, s * dim_ + j);
            }
        }
    }
    // Reshape cached last_z_ to (B*dim, S) as the cached input.
    Tensor z_view(B * dim_, seq_len_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            for (size_t s = 0; s < seq_len_; ++s) {
                z_view(b * dim_ + j, s) = last_z_(b, s * dim_ + j);
            }
        }
    }
    MLP2Weights tok_mlp{ &tok_mlp_w1_, &tok_mlp_w2_ };
    Tensor grad_z_from_mlp_view = mlp2_per_row_backward(
        grad_tok_view, tok_mlp, z_view, last_h1_, last_h1_gelu_,
        /*D_in=*/seq_len_, /*H=*/token_dim_);
    // grad_z_from_mlp_view has shape (B*dim, S). Reshape back to (B, S*D) and
    // ADD to grad_z_from_res. The total grad_z feeds into per-token LN backward.
    Tensor grad_z_from_mlp(B, seq_len_ * dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            for (size_t s = 0; s < seq_len_; ++s) {
                grad_z_from_mlp(b, s * dim_ + j) =
                    grad_z_from_mlp_view(b * dim_ + j, s);
            }
        }
    }

    // ----- Step 2 (reverse): per-token LN backward (on combined grad_z) -----
    // Pre-LN input to the per-token LN was the block's `input`.
    // The total gradient at z is the sum of two contributions:
    //   1. residual1: y1 = z + tok_pre, so dL/d_z_residual = grad_y1.
    //   2. token-mix MLP path: dL/d_z_from_tokmix = grad_z_from_mlp.
    // Both add up (the token-mix MLP backward does NOT include the residual
    // contribution; that lives in `grad_y1`).
    Tensor grad_z(B, seq_len_ * dim_);
    for (size_t i = 0; i < grad_z.data.size(); ++i) {
        grad_z.data[i] = grad_z_from_res.data[i] + grad_z_from_mlp.data[i];
    }
    Tensor grad_input_ln = ln_per_token_backward(grad_z, last_input_, ln_token_, seq_len_, dim_);

    return grad_input_ln;
}

void MlpMixerBlock::update_weights(double learning_rate) {
    ln_token_.update_weights(learning_rate);
    ln_channel_.update_weights(learning_rate);
    tok_mlp_w1_.update_weights(learning_rate);
    tok_mlp_w2_.update_weights(learning_rate);
    chan_mlp_w1_.update_weights(learning_rate);
    chan_mlp_w2_.update_weights(learning_rate);
}

void MlpMixerBlock::zero_grad() {
    ln_token_.zero_grad();
    ln_channel_.zero_grad();
    tok_mlp_w1_.zero_grad();
    tok_mlp_w2_.zero_grad();
    chan_mlp_w1_.zero_grad();
    chan_mlp_w2_.zero_grad();
}

std::vector<Tensor*> MlpMixerBlock::parameters() {
    std::vector<Tensor*> p;
    auto ln_t_params = ln_token_.parameters();
    auto ln_c_params = ln_channel_.parameters();
    p.insert(p.end(), ln_t_params.begin(), ln_t_params.end());
    p.insert(p.end(), ln_c_params.begin(), ln_c_params.end());
    auto t1 = tok_mlp_w1_.parameters();
    auto t2 = tok_mlp_w2_.parameters();
    auto c1 = chan_mlp_w1_.parameters();
    auto c2 = chan_mlp_w2_.parameters();
    p.insert(p.end(), t1.begin(), t1.end());
    p.insert(p.end(), t2.begin(), t2.end());
    p.insert(p.end(), c1.begin(), c1.end());
    p.insert(p.end(), c2.begin(), c2.end());
    return p;
}

std::vector<Tensor*> MlpMixerBlock::gradients() {
    std::vector<Tensor*> g;
    auto ln_t_grads = ln_token_.gradients();
    auto ln_c_grads = ln_channel_.gradients();
    g.insert(g.end(), ln_t_grads.begin(), ln_t_grads.end());
    g.insert(g.end(), ln_c_grads.begin(), ln_c_grads.end());
    auto t1 = tok_mlp_w1_.gradients();
    auto t2 = tok_mlp_w2_.gradients();
    auto c1 = chan_mlp_w1_.gradients();
    auto c2 = chan_mlp_w2_.gradients();
    g.insert(g.end(), t1.begin(), t1.end());
    g.insert(g.end(), t2.begin(), t2.end());
    g.insert(g.end(), c1.begin(), c1.end());
    g.insert(g.end(), c2.begin(), c2.end());
    return g;
}

// ============================================================================
// MlpMixerModel
// ============================================================================

MlpMixerModel::MlpMixerModel(size_t image_size, size_t patch_size, size_t in_channels,
                             size_t num_classes, size_t dim, size_t depth,
                             size_t token_dim, size_t channel_dim)
    : image_size_(image_size), patch_size_(patch_size),
      in_channels_(in_channels), num_classes_(num_classes),
      dim_(dim), depth_(depth),
      token_dim_(token_dim), channel_dim_(channel_dim),
      num_patches_((image_size / patch_size) * (image_size / patch_size)),
      patch_embed_(in_channels, dim, (int)patch_size, (int)patch_size,
                   (int)image_size, (int)image_size,
                   (int)patch_size, (int)patch_size),
      head_ln_(dim),
      classifier_(dim, num_classes)
{
    if (image_size % patch_size != 0) {
        throw std::invalid_argument("MlpMixerModel: patch_size must divide image_size");
    }
    if (depth == 0) {
        throw std::invalid_argument("MlpMixerModel: depth must be >= 1");
    }
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.emplace_back(dim, num_patches_, token_dim, channel_dim);
    }
}

Tensor MlpMixerModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Patch embed: (B, C_in*H*W) → (B, dim*S)
    Tensor patches = patch_embed_.forward(input);
    last_patches_ = patches.clone();

    // Stack of MixerBlocks
    Tensor h = patches;
    for (auto& blk : blocks_) {
        h = blk.forward(h);
    }
    last_head_pre_ln_ = h.clone();

    // Per-token LayerNorm
    Tensor h_ln = ln_per_token_forward(h, head_ln_, num_patches_, dim_);
    last_head_ln_ = h_ln.clone();

    // Mean-pool over S (num_patches_)
    size_t B = h_ln.rows;
    Tensor pooled(B, dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < dim_; ++j) {
            double sum = 0.0;
            for (size_t s = 0; s < num_patches_; ++s) {
                sum += h_ln(b, s * dim_ + j);
            }
            pooled(b, j) = sum / static_cast<double>(num_patches_);
        }
    }
    last_pooled_ = pooled.clone();

    // Classifier
    Tensor logits = classifier_.forward(pooled);
    last_logits_ = logits.clone();
    return logits;
}

Tensor MlpMixerModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.cols != num_classes_) {
        throw std::runtime_error("MlpMixerModel::backward: grad_output shape mismatch");
    }
    size_t B = grad_output.rows;

    // Classifier backward
    Tensor grad_pooled = classifier_.backward(grad_output, 0.0);

    // Mean-pool backward: grad_in shape (B, S*dim), grad_pooled shape (B, dim)
    // Each pooled element is (1/S) * sum_s h_ln[b, s*D + j], so
    // grad_h_ln[b, s*D + j] = (1/S) * grad_pooled[b, j].
    Tensor grad_head_ln(B, num_patches_ * dim_);
    double inv_S = 1.0 / static_cast<double>(num_patches_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < num_patches_; ++s) {
            for (size_t j = 0; j < dim_; ++j) {
                grad_head_ln(b, s * dim_ + j) = inv_S * grad_pooled(b, j);
            }
        }
    }

    // Per-token LN backward — pre-LN input was `last_head_pre_ln_`.
    Tensor grad_h = ln_per_token_backward(grad_head_ln, last_head_pre_ln_, head_ln_, num_patches_, dim_);

    // Walk blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_h = it->backward(grad_h, 0.0);
    }

    // Patch embed backward
    Tensor grad_input = patch_embed_.backward(grad_h, 0.0);
    return grad_input;
}

void MlpMixerModel::update_weights(double learning_rate) {
    patch_embed_.update_weights(learning_rate);
    head_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
    for (auto& blk : blocks_) {
        blk.update_weights(learning_rate);
    }
}

void MlpMixerModel::zero_grad() {
    patch_embed_.zero_grad();
    head_ln_.zero_grad();
    classifier_.zero_grad();
    for (auto& blk : blocks_) {
        blk.zero_grad();
    }
}

std::vector<Tensor*> MlpMixerModel::parameters() {
    std::vector<Tensor*> p;
    auto pe_params = patch_embed_.parameters();
    p.insert(p.end(), pe_params.begin(), pe_params.end());
    auto hl_params = head_ln_.parameters();
    p.insert(p.end(), hl_params.begin(), hl_params.end());
    auto cl_params = classifier_.parameters();
    p.insert(p.end(), cl_params.begin(), cl_params.end());
    for (auto& blk : blocks_) {
        auto bp = blk.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}

std::vector<Tensor*> MlpMixerModel::gradients() {
    std::vector<Tensor*> g;
    auto pe_grads = patch_embed_.gradients();
    g.insert(g.end(), pe_grads.begin(), pe_grads.end());
    auto hl_grads = head_ln_.gradients();
    g.insert(g.end(), hl_grads.begin(), hl_grads.end());
    auto cl_grads = classifier_.gradients();
    g.insert(g.end(), cl_grads.begin(), cl_grads.end());
    for (auto& blk : blocks_) {
        auto bg = blk.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    return g;
}
