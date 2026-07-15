#include "adaln_zero.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// Helpers
// =============================================================================
namespace {
inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
inline double sigmo(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Row-wise layer norm with no affine. Returns (mean per row, rstd per row,
// normalized tensor). Mean/rstd stored as row vectors of length B.
void layer_norm_row(const Tensor& x, double eps,
                    Tensor& mean_out, Tensor& rstd_out, Tensor& normed_out) {
    size_t B = x.rows, D = x.cols;
    mean_out = Tensor(1, B);
    rstd_out = Tensor(1, B);
    normed_out = Tensor(B, D);
    for (size_t i = 0; i < B; ++i) {
        double m = 0.0;
        for (size_t j = 0; j < D; ++j) m += x[i][j];
        m /= (double)D;
        mean_out[0][i] = m;
        double v = 0.0;
        for (size_t j = 0; j < D; ++j) {
            double d = x[i][j] - m;
            v += d * d;
        }
        v /= (double)D;
        double r = 1.0 / std::sqrt(v + eps);
        rstd_out[0][i] = r;
        for (size_t j = 0; j < D; ++j) {
            normed_out[i][j] = (x[i][j] - m) * r;
        }
    }
}
}  // namespace

// =============================================================================
// AdaLNModulation
// =============================================================================
AdaLNModulation::AdaLNModulation(size_t cond_dim, size_t d_model, size_t hidden_mult)
    : proj1_(cond_dim, hidden_mult * d_model),
      proj2_(hidden_mult * d_model, 3 * d_model),
      cond_dim_(cond_dim),
      d_model_(d_model),
      hidden_(hidden_mult * d_model) {
    if (cond_dim == 0) {
        throw std::invalid_argument("AdaLNModulation: cond_dim must be > 0");
    }
    if (d_model == 0) {
        throw std::invalid_argument("AdaLNModulation: d_model must be > 0");
    }
    if (hidden_mult == 0) {
        throw std::invalid_argument("AdaLNModulation: hidden_mult must be > 0");
    }
    proj1_.init_weights("xavier");
    proj2_.weights = Tensor::zeros(proj2_.weights.rows, proj2_.weights.cols);
    proj2_.bias    = Tensor::zeros(proj2_.bias.rows,    proj2_.bias.cols);
}

std::vector<Tensor> AdaLNModulation::forward(const Tensor& cond) {
    if (cond.cols != cond_dim_) {
        throw std::invalid_argument("AdaLNModulation::forward: cond feature dim mismatch");
    }
    // Layer 1
    Tensor h_pre = proj1_.forward(cond);  // (B, hidden), cached via Dense::last_input
    last_pre_silu_ = proj1_.last_input.clone();   // ACTUAL pre-SiLU (Dense::last_input)
    Tensor h_act(h_pre.rows, h_pre.cols);
    for (size_t i = 0; i < h_pre.rows; ++i)
        for (size_t j = 0; j < h_pre.cols; ++j)
            h_act[i][j] = silu(h_pre[i][j]);
    last_hidden_ = h_act.clone();

    // Layer 2 (zero-init)
    Tensor mod = proj2_.forward(h_act);  // (B, 3*d_model)
    // split into (shift, scale, gate)
    Tensor shift(mod.rows, d_model_);
    Tensor scale(mod.rows, d_model_);
    Tensor gate(mod.rows, d_model_);
    for (size_t i = 0; i < mod.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            shift[i][j] = mod[i][j];
            scale[i][j] = mod[i][d_model_ + j];
            gate[i][j]  = mod[i][2 * d_model_ + j];
        }
    }
    return {std::move(shift), std::move(scale), std::move(gate)};
}

std::vector<Tensor*> AdaLNModulation::parameters() {
    return {&proj1_.weights, &proj1_.bias, &proj2_.weights, &proj2_.bias};
}

std::vector<Tensor*> AdaLNModulation::gradients() {
    return {&proj1_.grad_weights, &proj1_.grad_bias, &proj2_.grad_weights, &proj2_.grad_bias};
}

void AdaLNModulation::update_weights(double lr) {
    // Delegate to the underlying Dense::update_weights, which already knows
    // the right shape conventions.
    proj1_.update_weights(lr);
    proj2_.update_weights(lr);
}

void AdaLNModulation::zero_grad() {
    proj1_.zero_grad();
    proj2_.zero_grad();
}

// =============================================================================
// AdaLNZeroBlock
// =============================================================================
AdaLNZeroBlock::AdaLNZeroBlock(size_t cond_dim, size_t d_model,
                               size_t hidden_mult, double eps)
    : cond_dim_(cond_dim), d_model_(d_model), eps_(eps),
      adaln_(cond_dim, d_model, hidden_mult) {
    if (d_model == 0) {
        throw std::invalid_argument("AdaLNZeroBlock: d_model must be > 0");
    }
    gamma_ = Tensor(1, d_model);
    beta_  = Tensor(1, d_model);
    for (size_t j = 0; j < d_model; ++j) gamma_[0][j] = 1.0;
    grad_gamma_ = Tensor(1, d_model);
    grad_beta_  = Tensor(1, d_model);
}

Tensor AdaLNZeroBlock::forward(const Tensor& input, const Tensor& cond) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("AdaLNZeroBlock::forward: input feature dim mismatch");
    }
    if (cond.cols != cond_dim_) {
        throw std::invalid_argument("AdaLNZeroBlock::forward: cond feature dim mismatch");
    }
    last_x_ = input.clone();
    last_cond_ = cond.clone();

    // 1. modulation
    auto mods = adaln_.forward(cond);
    last_shift_ = mods[0].clone();
    last_scale_ = mods[1].clone();
    last_gate_  = mods[2].clone();

    // 2. layer normalize (no affine for this version — pure zero-init anyway)
    Tensor mean, rstd, normed;
    layer_norm_row(input, eps_, mean, rstd, normed);
    last_mu_   = mean.clone();
    last_rstd_ = rstd.clone();
    last_normed_ = normed.clone();

    // 3. affine (γ=1, β=0 at init) → h
    Tensor h(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            h[i][j] = gamma_[0][j] * normed[i][j] + beta_[0][j];
    last_h_ = h.clone();

    // 4. modulate: y_mod = (1 + scale) ⊙ h + shift
    // shape: x is (B, D), scale/shift/gate may be (1, D) for global conditioning
    // or (B, D) for per-sample conditioning. Broadcast row-axis when cond.rows == 1.
    Tensor y_mod(input.rows, input.cols);
    {
        size_t mod_rows = last_scale_.rows;
        for (size_t i = 0; i < input.rows; ++i) {
            size_t mi = (mod_rows == 1) ? 0 : i;
            for (size_t j = 0; j < input.cols; ++j) {
                y_mod[i][j] = (1.0 + last_scale_[mi][j]) * h[i][j] + last_shift_[mi][j];
            }
        }
    }
    last_y_mod_ = y_mod.clone();

    // 5. gated residual: y = x + gate ⊙ y_mod
    Tensor y(input.rows, input.cols);
    {
        size_t mod_rows = last_gate_.rows;
        for (size_t i = 0; i < input.rows; ++i) {
            size_t mi = (mod_rows == 1) ? 0 : i;
            for (size_t j = 0; j < input.cols; ++j) {
                y[i][j] = input[i][j] + last_gate_[mi][j] * y_mod[i][j];
            }
        }
    }
    last_y_ = y.clone();
    return y;
}

Tensor AdaLNZeroBlock::forward(const Tensor& input) {
    Tensor zero_cond(1, cond_dim_);
    return forward(input, zero_cond);
}

Tensor AdaLNZeroBlock::backward(const Tensor& grad_output, double /*lr*/) {
    // Step A: dL/d_y_mod and dL/d_gate from grad_output
    // y = x + gate ⊙ y_mod  →  dL/d_y_mod = grad_output ⊙ last_gate_
    //                           dL/d_gate  = grad_output ⊙ last_y_mod_
    // Also: dL/d_x has a residual term (y = x + ...) so dL/d_x += grad_output directly.
    size_t B = last_x_.rows;
    size_t D = last_x_.cols;

    Tensor d_y_mod(B, D);
    Tensor d_gate(B, D);
    {
        size_t mod_rows = last_gate_.rows;
        for (size_t i = 0; i < B; ++i) {
            size_t mi = (mod_rows == 1) ? 0 : i;
            for (size_t j = 0; j < D; ++j) {
                d_y_mod[i][j] = grad_output[i][j] * last_gate_[mi][j];
                d_gate[i][j]  = grad_output[i][j] * last_y_mod_[i][j];
            }
        }
    }

    // Step B: build the (B, 3D) gradient tensor that flows back to
    // AdaLNModulation. The modulation parameters are proj1 (cond→hidden)
    // and proj2 (hidden → 3D). We need d/d_proj2 → d/d_hidden → SiLU
    // backward → d/d_proj1 → d/d_cond.
    //
    // Slice positions:
    //   d_mod[:, 0:D]    = d_y_mod                                  (for shift)
    //   d_mod[:, D:2D]   = d_y_mod ⊙ last_h                          (for scale)
    //   d_mod[:, 2D:3D]  = d_gate                                    (for gate)
    //
    // Forward: mod = SiLU(cond @ W1^T + b1) @ W2^T + b2, with W1 (hidden, cond_dim),
    // W2 (3D, hidden).
    // So d_hidden_act = d_mod @ W2  (shape (B, hidden)).
    // d_pre_silu     = d_hidden_act ⊙ silu'(last_pre_silu_)
    // d_cond          = d_pre_silu @ W1  (shape (B, cond_dim)).
    Tensor d_mod(B, 3 * D);
    {
        size_t mod_rows = last_scale_.rows;
        for (size_t i = 0; i < B; ++i) {
            size_t mi = (mod_rows == 1) ? 0 : i;
            for (size_t j = 0; j < D; ++j) {
                d_mod[i][j]            = d_y_mod[i][j];                    // shift
                d_mod[i][D + j]       = d_y_mod[i][j] * last_h_[i][j];    // scale
                d_mod[i][2 * D + j]   = d_gate[i][j];                     // gate
            }
            (void)mi;
        }
    }
    // Gradient w.r.t. proj2's weights and bias.
    // proj2: mod[i, o] = sum_h last_hidden_[i, h] * W2[o, h] + b2[o]
    // dW2[o, h] = sum_i d_mod[i, o] * last_hidden_[i, h]
    // db2[o]    = sum_i d_mod[i, o]
    {
        const Tensor& W2 = adaln_.proj2().weights;
        const Tensor& h_act = adaln_.last_hidden();
        Tensor gW(W2.rows, W2.cols);
        // bias is (1, out_features) per Dense convention — match its shape.
        Tensor gb(1, W2.rows);
        for (size_t o = 0; o < W2.rows; ++o) {
            double bs = 0.0;
            for (size_t h = 0; h < W2.cols; ++h) {
                double s = 0.0;
                for (size_t i = 0; i < B; ++i) s += d_mod[i][o] * h_act[i][h];
                gW[o][h] = s;
            }
            for (size_t i = 0; i < B; ++i) bs += d_mod[i][o];
            gb[0][o] = bs;  // shape (1, out_features)
        }
        adaln_.grad_proj2_w() += gW;
        adaln_.grad_proj2_b() += gb;
    }

    // d_hidden_act = d_mod @ W2.
    Tensor d_hidden_act(B, adaln_.last_hidden().cols);
    {
        const Tensor& W2 = adaln_.proj2().weights;
        for (size_t i = 0; i < B; ++i) {
            for (size_t k = 0; k < (size_t)W2.cols; ++k) {
                double s = 0.0;
                for (size_t r = 0; r < (size_t)W2.rows; ++r) {
                    s += d_mod[i][r] * W2[r][k];
                }
                d_hidden_act[i][k] = s;
            }
        }
    }

    // SiLU backward: σ(z)(1 + z(1 - σ(z))), where z = last_pre_silu_.
    // d_pre_silu = d_hidden_act ⊙ silu'(z)
    Tensor d_pre_silu(B, adaln_.last_pre_silu().cols);
    for (size_t i = 0; i < B; ++i) {
        for (size_t k = 0; k < (size_t)adaln_.last_pre_silu().cols; ++k) {
            double z = adaln_.last_pre_silu()[i][k];
            double s = sigmo(z);
            d_pre_silu[i][k] = d_hidden_act[i][k] * s * (1.0 + z * (1.0 - s));
        }
    }

    // Gradient w.r.t. proj1's weights and bias.
    // proj1: pre_silu[i, h] = sum_c cond[i, c] * W1[h, c] + b1[h]
    // dW1[h, c] = sum_i d_pre_silu[i, h] * cond[i, c]
    // db1[h]    = sum_i d_pre_silu[i, h]
    {
        const Tensor& W1 = adaln_.proj1().weights;
        Tensor gW(W1.rows, W1.cols);
        // bias is (1, out_features) per Dense convention.
        Tensor gb(1, W1.rows);
        for (size_t h = 0; h < W1.rows; ++h) {
            double bs = 0.0;
            for (size_t c = 0; c < W1.cols; ++c) {
                double s = 0.0;
                for (size_t i = 0; i < B; ++i) s += d_pre_silu[i][h] * last_cond_[i][c];
                gW[h][c] = s;
            }
            for (size_t i = 0; i < B; ++i) bs += d_pre_silu[i][h];
            gb[0][h] = bs;
        }
        adaln_.grad_proj1_w() += gW;
        adaln_.grad_proj1_b() += gb;
    }

    // Step C: chain back to h (the post-affine value).
    // y_mod = (1 + scale) * h + shift ⇒ d_h = d_y_mod ⊙ (1 + scale) + d_y_mod ⊙ last_scale ⊙ (something)
    // Actually: y_mod[i, j] = (1 + scale[i, j]) * h[i, j] + shift[i, j]
    //          ∂y_mod[i, j]/∂h[i, j] = (1 + scale[i, j])
    // So d_h[i, j] = d_y_mod[i, j] * (1 + scale[i, j])
    //
    // Wait — that is also wrong. d_h is d_y_mod scaled by (1 + scale), not by
    // a derived quantity. Let me re-derive:
    //   L = sum over output. f = y_mod[i, j] = (1+scale[i,j])*h[i,j] + shift[i,j]
    //   ∂L/∂h[i, j] = ∂L/∂f[i, j] * ∂f[i, j]/∂h[i, j] = d_y_mod[i, j] * (1 + scale[i, j])
    //
    // d_h has TWO contributions: from the direct loss path, plus from the scale
    // path which is intermediated by AdaLNModulation. But the scale path goes
    // through AdaLNModulation → proj2 → proj1 → SiLU → ... → cond (NOT back to h).
    // So d_h from THIS branch is just d_y_mod * (1+scale). The scale gradient
    // accumulates into AdaLNModulation's parameters, but it doesn't flow back to h.
    // That's correct.
    Tensor d_h(B, D);
    {
        size_t mod_rows = last_scale_.rows;
        for (size_t i = 0; i < B; ++i) {
            size_t mi = (mod_rows == 1) ? 0 : i;
            for (size_t j = 0; j < D; ++j)
                d_h[i][j] = d_y_mod[i][j] * (1.0 + last_scale_[mi][j]);
        }
    }

    // Step D: back through affine. h = γ ⊙ normed + β.
    // d_normed[i, j] = sum over k of (d_h[i, k] * ∂h[i, k]/∂normed[i, j]).
    // Since h[i, j] depends only on normed[i, j] (same indices), we have:
    //   d_normed[i, j] = d_h[i, j] * γ[j]
    // d_γ[j] = sum_i d_h[i, j] * normed[i, j]
    // d_β[j] = sum_i d_h[i, j]
    Tensor d_normed(B, D);
    for (size_t j = 0; j < D; ++j) {
        double sg = 0.0, sb = 0.0;
        for (size_t i = 0; i < B; ++i) {
            d_normed[i][j] = d_h[i][j] * gamma_[0][j];
            sg += d_h[i][j] * last_normed_[i][j];
            sb += d_h[i][j];
        }
        grad_gamma_[0][j] += sg;
        grad_beta_[0][j]  += sb;
    }

    // Step E: back through row-wise LN to get d_x.
    // LayerNorm_backward formula (per row, with affine=γ, β):
    //   dxhat[i, j] = d_normed[i, j] * γ[j]
    //   dvar_i  = sum_j dxhat[i, j] * (x[i, j] - μ_i) * (-0.5) * rstd_i^3
    //   dmean_i = sum_j dxhat[i, j] * (-rstd_i) + dvar_i * (-2/N) * sum_j (x[i, j] - μ_i)
    //   dx[i, j] = dxhat[i, j] * rstd_i + dvar_i * 2 (x[i, j] - μ_i) / N + dmean_i / N
    // But since γ = 1, β = 0 at init we can simplify; keep the general form.
    // Implementation:
    Tensor d_x(B, D);
    {
        for (size_t i = 0; i < B; ++i) {
            double mu = last_mu_[0][i];
            double rs = last_rstd_[0][i];
            // dxhat (already computed via d_normed; recompute here for clarity)
            Tensor dxhat(B, D);
            for (size_t j = 0; j < D; ++j) dxhat[i][j] = d_normed[i][j] * gamma_[0][j];
            // dvar
            double dvar = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double c = last_x_[i][j] - mu;
                dvar += dxhat[i][j] * c * (-0.5) * rs * rs * rs;
            }
            // dmean
            double dmean = 0.0;
            for (size_t j = 0; j < D; ++j) dmean += dxhat[i][j] * (-rs);
            dmean += dvar * 0.0;  // sum_j (x[i, j] - μ_i) = 0
            for (size_t j = 0; j < D; ++j) {
                double c = last_x_[i][j] - mu;
                d_x[i][j] = dxhat[i][j] * rs + dvar * 2.0 * c / (double)D + dmean / (double)D;
            }
        }
    }
    // Add the residual contribution: y = x + gate * y_mod ⇒ d_x += grad_output.
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < D; ++j)
            d_x[i][j] += grad_output[i][j];
    return d_x;
}

void AdaLNZeroBlock::update_weights(double lr) {
    adaln_.update_weights(lr);
    for (size_t j = 0; j < d_model_; ++j) {
        gamma_[0][j] -= lr * grad_gamma_[0][j];
        beta_[0][j]  -= lr * grad_beta_[0][j];
    }
}

std::vector<Tensor*> AdaLNZeroBlock::parameters() {
    auto p = adaln_.parameters();
    p.push_back(&gamma_);
    p.push_back(&beta_);
    return p;
}

std::vector<Tensor*> AdaLNZeroBlock::gradients() {
    auto g = adaln_.gradients();
    g.push_back(&grad_gamma_);
    g.push_back(&grad_beta_);
    return g;
}

void AdaLNZeroBlock::zero_grad() {
    adaln_.zero_grad();
    for (size_t j = 0; j < d_model_; ++j) {
        grad_gamma_[0][j] = 0.0;
        grad_beta_[0][j] = 0.0;
    }
}
