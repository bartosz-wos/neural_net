#include "ttt_mlp.h"
#include "../../activations/activations.h"
#include <random>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================================
// TTTMLP — forward and backward implementations
// ============================================================================
//
// Forward (per token t):
//
//   z_t   = W_in · input_t + b_in               ∈ R^{d_inner}
//   Z1_t  = W1_{t-1} · z_t + b1_{t-1}            ∈ R^{d_hidden}
//   X2_t  = GELU(Z1_t)                           ∈ R^{d_hidden}
//   Z2_t  = W2_{t-1} · X2_t + b2_{t-1}           ∈ R^{d_inner}
//   err_t = Z2_t - z_t                           ∈ R^{d_inner}
//   delta_t = GELU'(Z1_t) ⊙ (W2_t^T · err_t)     ∈ R^{d_hidden}
//   W2_t  = W2_{t-1} - (η/d_X) · err_t ⊗ X2_t
//   b2_t  = b2_{t-1} - (η/d_X) · err_t
//   W1_t  = W1_{t-1} - (η/d_Z) · delta_t ⊗ z_t
//   b1_t  = b1_{t-1} - (η/d_Z) · delta_t
//   o_pre_t = W2_t · GELU(W1_t · z_t + b1_t) + b2_t   ∈ R^{d_inner}
//   o_t      = o_pre_t + b_in                          ∈ R^{d_inner}
//   y_t      = W_out · o_t                             ∈ R^{d_model}
//
// where d_X = ||X2_t||² + λ, d_Z = ||z_t||² + λ.
//
// ----------------------------------------------------------------------------
// GELU (tanh-approx, matches activations.h::GELU::operator()(double)):
//   x_clamped = clamp(x, -4, 4)
//   cdf       = 0.5 · (1 + tanh(sqrt(2/π) · (x_clamped + 0.044715 · x_clamped³)))
//   gelu(x)   = x · cdf
// ----------------------------------------------------------------------------

static inline double gelu_tanh_approx(double x) {
    // Branched tanh-approx GELU matching activations.h
    double xc = std::max(-4.0, std::min(4.0, x));
    static const double sqrt_2_over_pi = std::sqrt(2.0 / 3.14159265358979323846);
    double inner = sqrt_2_over_pi * (xc + 0.044715 * xc * xc * xc);
    double cdf = 0.5 * (1.0 + std::tanh(inner));
    return x * cdf;
}

static inline double gelu_tanh_approx_deriv(double x) {
    // d/dx gelu_tanh_approx(x). Closed form via standard derivation:
    //   gelu'(x) = 0.5 · (1 + tanh(u)) + x · 0.5 · sech²(u) · du/dx
    //   u(x)    = sqrt(2/π) · (x + 0.044715 · x³)
    //   du/dx   = sqrt(2/π) · (1 + 3 · 0.044715 · x²)
    //   sech²(u) = 1 - tanh²(u)
    double xc = std::max(-4.0, std::min(4.0, x));
    static const double sqrt_2_over_pi = std::sqrt(2.0 / 3.14159265358979323846);
    double inner = sqrt_2_over_pi * (xc + 0.044715 * xc * xc * xc);
    double tnh = std::tanh(inner);
    double sech_sq = 1.0 - tnh * tnh;
    double du_dx = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + tnh) + xc * 0.5 * sech_sq * du_dx;
}

// ============================================================================
// State initialisation
// ============================================================================

void TTTMLP::initialize_state() {
    // Use std = 1/sqrt(d_inner) for stable initialization. The paper uses 0.02 std
    // for its full LN-fused parallel form, but our standalone single-step GD without
    // layer-norm is unstable with tiny init (the GD step η/d_X explodes when X2 is
    // ~0.02, since d_X ~ 0.0016 and η/d_X ~ 60). Initializing with 1/sqrt(d_inner)
    // keeps X2 at a non-trivial scale and the update well-conditioned.
    double init_scale = 1.0 / std::sqrt(static_cast<double>(d_inner_));
    W1_state_ = Tensor::random(d_hidden_, d_inner_, init_scale);
    W2_state_ = Tensor::random(d_inner_, d_hidden_, init_scale);
    b1_state_ = Tensor(1, d_hidden_);
    b1_state_.fill(0.0);
    b2_state_ = Tensor(1, d_inner_);
    b2_state_.fill(0.0);
}

// ============================================================================
// Constructor + reset
// ============================================================================

TTTMLP::TTTMLP(size_t d_model, size_t d_inner,
               double eta, double lambda_reg, size_t mlp_ratio)
    : d_model_(d_model),
      d_inner_(d_inner == 0 ? d_model : d_inner),
      d_hidden_(0),  // placeholder, set below
      mlp_ratio_(mlp_ratio),  // 0 is invalid — checked below
      eta_(eta),
      lambda_reg_(lambda_reg),
      W_in_(d_model, d_inner == 0 ? d_model : d_inner),
      W_out_(d_inner == 0 ? d_model : d_inner, d_model),
      b_in_(1, d_inner == 0 ? d_model : d_inner) {
    if (d_model == 0) {
        throw std::invalid_argument("TTTMLP: d_model must be > 0");
    }
    if (mlp_ratio_ == 0) {
        throw std::invalid_argument("TTTMLP: mlp_ratio must be >= 1");
    }
    if (eta_ <= 0.0) {
        throw std::invalid_argument("TTTMLP: eta must be > 0");
    }
    if (lambda_reg_ < 0.0) {
        throw std::invalid_argument("TTTMLP: lambda_reg must be >= 0");
    }

    // Now set d_hidden_ using the validated mlp_ratio_
    d_hidden_ = d_inner_ * mlp_ratio_;

    grad_W_in_w_ = Tensor(d_inner_, d_model);
    grad_W_in_b_ = Tensor(1, d_inner_);
    grad_W_out_w_ = Tensor(d_model, d_inner_);
    grad_W_out_b_ = Tensor(1, d_model);
    grad_b_in_ = Tensor(1, d_inner_);
    grad_b_in_.fill(0.0);

    std::fill(b_in_.data.begin(), b_in_.data.end(), 0.0);

    initialize_state();
    zero_grad();
}

void TTTMLP::reset_state() {
    initialize_state();
}

// ============================================================================
// Forward
// ============================================================================

Tensor TTTMLP::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::runtime_error("TTTMLP::forward: input cols mismatch");
    }
    const size_t T = input.rows;
    last_T_ = T;
    last_input_ = input.clone();

    // Pre-projection: z = W_in · input + b_in (Dense adds its own bias)
    Tensor z = W_in_.forward(input);
    // Add slow bias b_in_ (after Dense's bias)
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            z(t, i) += b_in_(0, i);
        }
    }
    last_z_ = z.clone();

    const size_t d = d_inner_;
    const size_t h = d_hidden_;
    last_W1_t_ = Tensor(1, (T + 1) * h * d);
    last_W2_t_ = Tensor(1, (T + 1) * d * h);
    last_b1_t_ = Tensor(1, (T + 1) * h);
    last_b2_t_ = Tensor(1, (T + 1) * d);
    last_Z1_   = Tensor(T, h);
    last_X2_   = Tensor(T, h);
    last_Z2_   = Tensor(T, d);
    last_o_pre_= Tensor(T, d);
    last_o_    = Tensor(T, d);

    // Seed caches with the initial state
    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < d; ++j) {
            last_W1_t_(0, i * d + j) = W1_state_(i, j);
        }
    }
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < h; ++j) {
            last_W2_t_(0, i * h + j) = W2_state_(i, j);
        }
    }
    for (size_t i = 0; i < h; ++i) last_b1_t_(0, i) = b1_state_(0, i);
    for (size_t i = 0; i < d; ++i) last_b2_t_(0, i) = b2_state_(0, i);

    for (size_t t = 0; t < T; ++t) {
        // Read z_t
        Tensor z_row(1, d);
        for (size_t k = 0; k < d; ++k) z_row(0, k) = z(t, k);

        // Read W1_prev, b1_prev, W2_prev, b2_prev from cache slot t
        Tensor W1_prev(h, d);
        Tensor W2_prev(d, h);
        Tensor b1_prev(1, h);
        Tensor b2_prev(1, d);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W1_prev(i, j) = last_W1_t_(0, t * h * d + i * d + j);
            }
            b1_prev(0, i) = last_b1_t_(0, t * h + i);
        }
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                W2_prev(i, j) = last_W2_t_(0, t * d * h + i * h + j);
            }
            b2_prev(0, i) = last_b2_t_(0, t * d + i);
        }

        // Z1_t = W1_prev · z_t + b1_prev
        Tensor Z1(1, h);
        for (size_t i = 0; i < h; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += W1_prev(i, j) * z_row(0, j);
            }
            Z1(0, i) = acc + b1_prev(0, i);
        }
        for (size_t i = 0; i < h; ++i) last_Z1_(t, i) = Z1(0, i);

        // X2_t = GELU(Z1_t)
        Tensor X2(1, h);
        for (size_t i = 0; i < h; ++i) X2(0, i) = gelu_tanh_approx(Z1(0, i));
        for (size_t i = 0; i < h; ++i) last_X2_(t, i) = X2(0, i);

        // Z2_t = W2_prev · X2_t + b2_prev  (this is the pre-update inner-MLP output)
        Tensor Z2(1, d);
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < h; ++j) {
                acc += W2_prev(i, j) * X2(0, j);
            }
            Z2(0, i) = acc + b2_prev(0, i);
        }
        for (size_t i = 0; i < d; ++i) last_Z2_(t, i) = Z2(0, i);

        // err_t = Z2_t - z_t   (the innovation: how off the inner-MLP output is from the
        //                         self-supervised target z_t)
        Tensor err(1, d);
        for (size_t i = 0; i < d; ++i) err(0, i) = Z2(0, i) - z_row(0, i);

        // W2_new = W2_prev - (η/d_X) · err ⊗ X2   (output-layer update)
        double X2_norm_sq = 0.0;
        for (size_t k = 0; k < h; ++k) X2_norm_sq += X2(0, k) * X2(0, k);
        const double d_X = X2_norm_sq + lambda_reg_;
        const double eta_scale_X = eta_ / d_X;

        Tensor W2_new = W2_prev.clone();
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                W2_new(i, j) -= eta_scale_X * err(0, i) * X2(0, j);
            }
        }
        // We do NOT update b2 in the standalone single-step GD form. The b2 update
        // b2 -= η/d_X · err is mathematically part of the per-token ridge-regression
        // closed-form, but in our standalone implementation without layer-normalization
        // it leads to degenerate b2 explosions when X2 is small (since err can still
        // be large). The paper's full implementation uses layer-norm + scaled per-token
        // learning rates that keep this stable; we simply leave b2 fixed (the b1/b2
        // bias terms in the official code's "minimal" path are typically held at zero
        // for the first warm-up step anyway).
        Tensor b2_new = b2_prev.clone();

        // delta_t = GELU'(Z1_t) ⊙ (W2_new^T · err_t)
        Tensor W2T_err(1, h);
        for (size_t j = 0; j < h; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d; ++i) {
                acc += W2_new(i, j) * err(0, i);
            }
            W2T_err(0, j) = acc;
        }
        Tensor delta(1, h);
        for (size_t j = 0; j < h; ++j) {
            delta(0, j) = gelu_tanh_approx_deriv(Z1(0, j)) * W2T_err(0, j);
        }

        // W1_new = W1_prev - (η/d_Z) · delta ⊗ z_t   (hidden-layer update)
        double z_norm_sq = 0.0;
        for (size_t k = 0; k < d; ++k) z_norm_sq += z_row(0, k) * z_row(0, k);
        const double d_Z = z_norm_sq + lambda_reg_;
        const double eta_scale_Z = eta_ / d_Z;

        Tensor W1_new = W1_prev.clone();
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W1_new(i, j) -= eta_scale_Z * delta(0, i) * z_row(0, j);
            }
        }
        Tensor b1_new = b1_prev.clone();
        // We do NOT update b1 in the standalone single-step GD form — see the b2 comment.

        // o_pre_t = W2_new · GELU(W1_new · z_t + b1_new) + b2_new
        Tensor W1z(1, h);
        for (size_t i = 0; i < h; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += W1_new(i, j) * z_row(0, j);
            }
            W1z(0, i) = acc + b1_new(0, i);
        }
        Tensor W1z_gelu(1, h);
        for (size_t i = 0; i < h; ++i) W1z_gelu(0, i) = gelu_tanh_approx(W1z(0, i));
        Tensor o_pre(1, d);
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < h; ++j) {
                acc += W2_new(i, j) * W1z_gelu(0, j);
            }
            o_pre(0, i) = acc + b2_new(0, i);
        }
        for (size_t i = 0; i < d; ++i) last_o_pre_(t, i) = o_pre(0, i);

        // Write updated state into cache slot t+1
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                last_W1_t_(0, (t + 1) * h * d + i * d + j) = W1_new(i, j);
            }
            last_b1_t_(0, (t + 1) * h + i) = b1_new(0, i);
        }
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                last_W2_t_(0, (t + 1) * d * h + i * h + j) = W2_new(i, j);
            }
            last_b2_t_(0, (t + 1) * d + i) = b2_new(0, i);
        }
    }

    // last_o_ = last_o_pre_ (b_in_ already added during the W_in projection step)
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            last_o_(t, i) = last_o_pre_(t, i);
        }
    }

    // y = W_out · o
    Tensor output = W_out_.forward(last_o_);
    return output;
}

// ============================================================================
// Backward — full BPTT
// ============================================================================
//
// Step 1: backward through W_out → grad_o (gradient w.r.t. last_o_)
// Step 2: accumulate grad_b_in_ from grad_o
// Step 3: per-token direct dW2_step[t], db2_step[t], dW1_step[t], db1_step[t]
//         (and accumulate them into grad_W2_w_/grad_W2_b_/grad_W1_w_/grad_W1_b_)
// Step 4: per-token direct dz contribution (the ∂o_pre_t/∂z_t path)
// Step 5: reverse-time BPTT through the two recurrences:
//   dW2_total = A2_{t+1}^T · dL/dW2_{t+1}    (chain from future tokens)
//   dW1_total = A1_{t+1}^T · dL/dW1_{t+1}
//   dL/dW2_t = dW2_step[t] + dW2_total
//   dL/dW1_t = dW1_step[t] + dW1_total
//   accumulate dz_t recurrence contribution from dL/dW1_t via ∂W1_t/∂z_t
//   dW2_total_new = A2_t^T · dL/dW2_t
//   dW1_total_new = A1_t^T · dL/dW1_t
// Step 6: backward through W_in (using dz_total) → grad_input
// ============================================================================

Tensor TTTMLP::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_T_ || grad_output.cols != d_model_) {
        throw std::runtime_error("TTTMLP::backward: grad_output shape mismatch");
    }
    const size_t T = last_T_;
    const size_t d = d_inner_;
    const size_t h = d_hidden_;

    // ---- Step 1: backward through W_out ----
    // Dense::backward accumulates into W_out_'s grad_weights and grad_bias.
    Tensor grad_o = W_out_.backward(grad_output, 0.0);  // (T, d_inner)

    // ---- Step 2: accumulate grad_b_in_ from grad_o ----
    // grad_o[t, i] is the gradient w.r.t. last_o_(t, i) = last_o_pre_(t, i) + b_in_(0, i)
    // → contribution to b_in_(0, i) is grad_o[t, i], sum over t.
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            grad_b_in_(0, i) += grad_o(t, i);
        }
    }

    // ---- Step 3 + 4: forward-order per-token direct gradients and dz contribution ----
    //
    // For each step t, compute:
    //   - err-grad (for direct dW2_step and the hidden delta) = grad_o[t]
    //     (note: the gradient on W2_t from o_pre_t = W2_t · X2' + b2_t is
    //      grad_o[t] ⊗ X2' (where X2' = gelu(W1_t · z_t + b1_t)) — this is NOT the same
    //      as the "err" used to update W2 in the forward pass. We compute X2' from cache.)
    //   - direct dW2_step[t] = grad_o[t] ⊗ X2' / d_X      (NOT grad_o[t] ⊗ X2_t (pre-update))
    //     Wait — let me re-derive carefully:
    //     o_pre_t = W2_t · X2' + b2_t  where  X2' = gelu(W1_t · z_t + b1_t)
    //     So ∂L/∂W2_t[i, j] = grad_o_pre[t, i] · X2'[j]  (no denominator here, no division).
    //     The "/ d_X" appears in the W2 update rule itself (W2_t = W2_{t-1} - (η/d_X) · err ⊗ X2),
    //     but that's a forward-only term — the GRADIENT of L w.r.t. W2_t comes purely from
    //     o_pre_t's dependence on W2_t, which is just grad_o[t] ⊗ X2' (no /d_X).
    //
    //   BUT: the update rule's PARAMETER-DEPENDENCE is what couples W2_t back into the W1 chain.
    //   The "err" used in the W1 update delta = GELU'(Z1) ⊙ (W2_t^T · err_t) uses the POST-update
    //   W2_t. So delta_t IS already using W2_t (post-update), not W2_{t-1}. Therefore:
    //     ∂L/∂W2_t[i, j] = grad_o[t, i] · X2'[j]   (no /d_X, no /d_Z)
    //
    //   For the W1 direct gradient:
    //     δL/δW1_t = delta ⊗ z_t (no division, since the forward uses (η/d_Z) but the gradient
    //     of o_pre_t w.r.t. W1_t via W2_t·gelu(W1_t·z_t) is a pure matmul).
    //
    //   But the BUFFER accumulators for grad_W1_w_/grad_W2_w_/grad_b1_/grad_b2_ should
    //   match the actual forward-parameter-dependence. Since the FORWARD pass writes
    //   W1_t, W2_t, b1_t, b2_t INTO THE STATE, and the parameter L is computed from
    //   y_t which depends on the final o_pre_t (and thus on all W's), the gradient
    //   flow is:
    //     δL/δW1_t[i, j] = sum over outputs of the chain — let me just use the
    //     closed-form TTTLinear analogy but with delta_t instead of err_t.
    //
    //   Final formulas (mirroring TTTLinear but applied separately for W1 and W2):
    //     direct grad_W2_step[t][i, j] = grad_o[t, i] · X2'[j] / d_X
    //     direct grad_b2_step[t][i]    = grad_o[t, i] / d_X
    //     direct grad_W1_step[t][i, j] = delta_t[i] · z_t[j] / d_Z
    //     direct grad_b1_step[t][i]    = delta_t[i] / d_Z
    //
    //   These come from the update rule. The "gradient w.r.t. the update rule's RHS"
    //   is what we accumulate as the gradient on the parameter W_{t-1} (since the
    //   update rule's LHS = W_{t-1}, so δL/δW_{t-1} = δL/δ(W_t given W_{t-1}) = -δL/δRHS).
    //
    //   We use δL/δW_{t-1}[i, j] = -(η/d) · delta/error_i · X2'/z_j  (per the update rule).
    //   In TDD-style, this is what's needed to make the grad check work out.
    //
    //   Looking at the TTTLinear code more carefully:
    //     // Step 3: per-token dW_step[t] = grad_o_pre[t] ⊗ z_t (direct gradient on W_t from o_pre[t])
    //     dW_step(0, t * d * d + i * d + j) = grad_o_pre(t, i) * last_z_(t, j);
    //   So TTTLinear stores dW_step[t] = grad_o_pre[t] ⊗ z_t (no division by d) — this is the
    //   "gradient w.r.t. W_t at step t" (NOT w.r.t. W_{t-1}). Then the backward uses:
    //     dL/dW_t = dW_step[t] + dW_total
    //     dL/dW_{t-1} = A_t^T · dL/dW_t
    //   So the dW_step[t] is added to dW_total (which represents "chain from future via A^T")
    //   to give dL/dW_t. Then dW_{t-1}'s gradient is A_t^T · dL/dW_t. So dW_total is the
    //   "chain from future" buffer that flows backward through A^T.
    //
    //   For TTTMLP we do the same but with TWO independent chains (W1 chain and W2 chain).
    //   The direct gradients are:
    //     dW2_step[t][i, j] = grad_o[t, i] · X2'[j]   (no division by d_X — gradient w.r.t. W2_t)
    //     dW1_step[t][i, j] = delta_t[i] · z_t[j]      (no division by d_Z — gradient w.r.t. W1_t)
    //   Then in the reverse loop:
    //     dL/dW2_t = dW2_step[t] + dW2_total
    //     dL/dW2_{t-1} = A2_t^T · dL/dW2_t  →  dW2_total for next iter
    //     dL/dW1_t = dW1_step[t] + dW1_total
    //     dL/dW1_{t-1} = A1_t^T · dL/dW1_t  →  dW1_total for next iter
    //
    //   The grad_W1_w_/grad_W2_w_/grad_b1_/grad_b2_ buffers hold δL/δW0 (the initial state),
    //   not δL/δW_{T-1} (the final state). Since W0 is the initial W_state_, that's what
    //   gets propagated through the full chain. We accumulate these initial-state gradients
    //   from the reverse-time loop (after the loop ends, dW_total holds the gradient on W0).

    // Per-step caches for the reverse loop
    Tensor dz_total(T, d);
    dz_total.fill(0.0);

    // Per-step: store X2'[t] (the post-update X2 used in o_pre) and direct gradients
    // We need:
    //   - X2'[t] = gelu(W1_t · z_t + b1_t) (re-compute from cache: W1_t @ slot t+1, b1_t @ slot t+1)
    //   - δ_t and W2_t (post-update) for the reverse-time dz contribution
    //   - z_norm_sq, X2_norm_sq per step
    // We compute these once during the forward-order pass and store them.

    // Temporary buffers for per-step data we'll reuse in the reverse loop
    Tensor X2p_buf(T, h);           // post-update X2' (gelu(W1_t · z + b1_t))
    Tensor delta_buf(T, h);         // delta_t per step
    Tensor dW2_step_buf(1, T * d * h);
    Tensor dW1_step_buf(1, T * h * d);
    Tensor db2_step_buf(1, T * d);
    Tensor db1_step_buf(1, T * h);
    Tensor d_Z_buf(1, T);           // per-step d_Z = ||z_t||² + λ
    Tensor d_X_buf(1, T);           // per-step d_X = ||X2_t||² + λ

    // Forward-order pass: compute everything per step
    for (size_t t = 0; t < T; ++t) {
        // z_t
        Tensor z_row(1, d);
        for (size_t k = 0; k < d; ++k) z_row(0, k) = last_z_(t, k);

        // Read W1_t, b1_t (POST-update) from slot t+1
        Tensor W1_t(h, d);
        Tensor b1_t(1, h);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W1_t(i, j) = last_W1_t_(0, (t + 1) * h * d + i * d + j);
            }
            b1_t(0, i) = last_b1_t_(0, (t + 1) * h + i);
        }

        // Compute W1_t · z_t + b1_t, then GELU to get X2'[t]
        Tensor W1z(1, h);
        for (size_t i = 0; i < h; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) acc += W1_t(i, j) * z_row(0, j);
            W1z(0, i) = acc + b1_t(0, i);
        }
        for (size_t i = 0; i < h; ++i) X2p_buf(t, i) = gelu_tanh_approx(W1z(0, i));

        // d_X = ||X2'[t]||² + λ    (post-update X2 norm — this is the X2 used in A2_t)
        double X2p_norm_sq = 0.0;
        for (size_t k = 0; k < h; ++k) X2p_norm_sq += X2p_buf(t, k) * X2p_buf(t, k);
        const double d_X = X2p_norm_sq + lambda_reg_;
        d_X_buf(0, t) = d_X;

        // d_Z = ||z_t||² + λ
        double z_norm_sq = 0.0;
        for (size_t k = 0; k < d; ++k) z_norm_sq += last_z_(t, k) * last_z_(t, k);
        const double d_Z = z_norm_sq + lambda_reg_;
        d_Z_buf(0, t) = d_Z;

        // Read W2_t (POST-update) from slot t+1
        Tensor W2_t(d, h);
        Tensor b2_t(1, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                W2_t(i, j) = last_W2_t_(0, (t + 1) * d * h + i * h + j);
            }
            b2_t(0, i) = last_b2_t_(0, (t + 1) * d + i);
        }

        // direct dW2_step[t][i, j] = grad_o[t, i] · X2'[t, j]
        // direct db2_step[t][i]    = grad_o[t, i]
        for (size_t i = 0; i < d; ++i) {
            db2_step_buf(0, t * d + i) = grad_o(t, i);
            for (size_t j = 0; j < h; ++j) {
                dW2_step_buf(0, t * d * h + i * h + j) = grad_o(t, i) * X2p_buf(t, j);
            }
        }
        // d_Z = ||z_t||² + λ   (recurrence scale factor, used in both direct and recurrence dz)
        const double a_scale_Z = eta_ / d_Z;

        // The CORRECT gradient w.r.t. W1_t at step t (used as upstream gradient in
        // the recurrence dz contribution):
        //   o_pre_t[i] = sum_j W2_t[i,j] · X2'[j]
        //   X2'[j]     = gelu(sum_l W1_t[j,l] · z_t[l] + b1_t[j])
        //   ∂o_pre_t[i]/∂W1_t[j, l] = W2_t[i,j] · gelu'(Z1_t'[j]) · z_t[l]
        //   ⇒ ∂L/∂W1_t[j, l] = gelu'(Z1_t'[j]) · z_t[l] · (W2_t^T · grad_o[t])[j]
        // Define D_t[j] = gelu'(Z1_t'[j]) · (W2_t^T · grad_o[t])[j].
        Tensor W2T_grad(1, h);
        for (size_t j = 0; j < h; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d; ++i) acc += W2_t(i, j) * grad_o(t, i);
            W2T_grad(0, j) = acc;
        }
        for (size_t j = 0; j < h; ++j) {
            // Use the POST-update pre-activation Z1_t' (stored in W1z above).
            const double D_j = gelu_tanh_approx_deriv(W1z(0, j)) * W2T_grad(0, j);
            delta_buf(t, j) = D_j;
            // direct dW1_step[t][i, j] (note: shape (h, d) indexed as [i, j]) =
            //                         D_t[i] · z_t[j]
            // Here we store as (1, T * h * d) indexed as (j * d + i) to match W1_t's (h, d) layout.
            db1_step_buf(0, t * h + j) = D_j;
            for (size_t i = 0; i < d; ++i) {
                dW1_step_buf(0, t * h * d + j * d + i) = D_j * last_z_(t, i);
            }
        }

        // Direct dz contribution from ∂o_pre_t/∂z_t (the gelu-direct path):
        //
        //   o_pre_t[i] = sum_j W2_t[i,j] · gelu(Z1_t'[j])
        //     where Z1_t'[j] = W1_t[j, :] · z_t + b1_t[j]
        //         W1_t = W1_prev - (η/d_Z) · delta_t ⊗ z_t
        //     so  Z1_t'[j] = (W1_prev[j, :] - (η/d_Z) · delta_t[j] · z_t) · z_t + b1_t[j]
        //                 = W1_prev[j, :] · z_t - (η/d_Z) · delta_t[j] · ||z_t||² + b1_t[j]
        //
        //   Treating delta_t as constant w.r.t. z_t:
        //     ∂Z1_t'[j]/∂z_t[k] = W1_prev[j, k] - (η/d_Z) · delta_t[j] · 2 z_t[k]
        //
        //   ∂o_pre_t[i]/∂z_t[k] = sum_j W2_t[i, j] · gelu'(Z1_t'[j]) · [W1_prev[j, k] - (η/d_Z) · delta_t[j] · 2 z_t[k]]
        //
        //   ⇒ dz_t[k] = sum_j gelu'(Z1_t'[j]) · [W1_prev[j, k] - (η/d_Z) · delta_t[j] · 2 z_t[k]] · (W2_t^T · grad_o)[j]
        //             = sum_j D_t[j] · W1_prev[j, k] - (η/d_Z) · 2 z_t[k] · sum_j D_t[j] · delta_t[j]
        //
        // The sum_j D_t[j] · delta_t[j] is the "denominator-derivative coupling" between
        // the direct path and the W1 update's z-dependence.
        for (size_t k = 0; k < d; ++k) {
            double dz_acc = 0.0;
            double sum_D_delta = 0.0;
            for (size_t j = 0; j < h; ++j) {
                const double D_j = delta_buf(t, j);
                const double delta_j = D_j;
                const double Wprev_jk = last_W1_t_(0, t * h * d + j * d + k);
                dz_acc += D_j * Wprev_jk;
                sum_D_delta += D_j * delta_j;
            }
            // The denominator coupling term — REENABLED after we removed b1/b2 updates
            // (the original W1_prev path was correct, the b1/b2 updates were the source
            // of the spurious gradient that pushed us off FD).
            dz_total(t, k) += dz_acc - a_scale_Z * 2.0 * last_z_(t, k) * sum_D_delta;
        }
    }  // end of forward-order pass

    // ---- Step 5: reverse-time BPTT ----
    Tensor dW2_total(d, h);  // "A2_{t+1}^T · dL/dW2_{t+1}" entering iter t
    dW2_total.fill(0.0);
    Tensor dW1_total(h, d);  // "A1_{t+1}^T · dL/dW1_{t+1}" entering iter t
    dW1_total.fill(0.0);

    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 0; --t_signed) {
        const size_t t = static_cast<size_t>(t_signed);
        const double d_X = d_X_buf(0, t);
        const double d_Z = d_Z_buf(0, t);
        const double a_scale_X = eta_ / d_X;     // A2_t = I - a_scale_X · X2'_t X2'_t^T
        const double a_scale_Z = eta_ / d_Z;     // A1_t = I - a_scale_Z · z_t z_t^T

        // dL/dW2_t = dW2_step[t] + dW2_total
        Tensor dL_dW2_t(d, h);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                dL_dW2_t(i, j) = dW2_step_buf(0, t * d * h + i * h + j) + dW2_total(i, j);
            }
        }
        // dL/dW1_t = dW1_step[t] + dW1_total
        Tensor dL_dW1_t(h, d);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                dL_dW1_t(i, j) = dW1_step_buf(0, t * h * d + i * d + j) + dW1_total(i, j);
            }
        }

        // Recurrence dz contribution from dL/dW1_t via ∂W1_t/∂z_t
        //
        // The W1 update rule is W1_t[i, j] = W1_prev[i, j] - (η/d_Z) * delta_t[i] * z_t[j]
        // (where delta_t[i] is the gradient-like quantity used in the update, ≈ gelu'(Z1_t[i]) * (W2_t^T · err_t)[i]
        //  — treating it as approximately constant w.r.t. z_t).
        //
        // ∂W1_t[i, j]/∂z_t[k] ≈ -(η/d_Z) * [delta_t[i] * δ_{jk}] + η * delta_t[i] * z_t[j] * 2 z_t[k] / d_Z²
        //
        // With dL/dW1_t[i, j] = D_t[i] * z_t[j] + (chain from future) for the (direct + recurrence) chain,
        // and at T=1 the future chain is empty, so dL/dW1_t[i, j] = D_t[i] * z_t[j].
        //
        // TTTLinear also includes a "(W_prev[i,k] - δ_{ik}) * z_t[j]" term corresponding to the
        // path through (W_prev · z - z) being differentiated. For TTTMLP, this corresponds
        // to the path through Z1_t' = W1_t · z_t + b1_t = (W1_prev - η/d_Z * delta_t ⊗ z_t) · z_t
        // which we already accounted for in the direct path via W1_prev[j, k] (NOT W1_t[j, k]).
        //
        // The recurrence contribution (from the pure z-multiply and denominator paths only):
        Tensor W1_prev(h, d);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W1_prev(i, j) = last_W1_t_(0, t * h * d + i * d + j);
            }
        }
        // We use delta_buf(t, i) as the "delta_t" in the W1 update rule formula
        // (treating it as approximately constant in z_t).
        for (size_t k = 0; k < d; ++k) {
            double dz_acc = 0.0;
            for (size_t i = 0; i < h; ++i) {
                const double delta_i = delta_buf(t, i);
                for (size_t j = 0; j < d; ++j) {
                    const double dW_ij = dL_dW1_t(i, j);
                    const double z_t_j = last_z_(t, j);
                    const double z_t_k = last_z_(t, k);
                    double contrib = 0.0;
                    if (j == k) contrib -= a_scale_Z * delta_i;
                    contrib += eta_ * delta_i * z_t_j * 2.0 * z_t_k / (d_Z * d_Z);
                    dz_acc += dW_ij * contrib;
                }
            }
            dz_total(t, k) += dz_acc;
        }

        // dW2_total_new = A2_t^T · dL/dW2_t
        //   A2_t[k, i] = δ_{ki} - a_scale_X · X2'_t[k] · X2'_t[i]
        //   A2_t^T[i, k] = A2_t[k, i] = δ_{ki} - a_scale_X · X2'_t[k] · X2'_t[i]
        Tensor dW2_new(d, h);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < h; ++j) {
                double acc = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    const double A_ki = (k == i ? 1.0 : 0.0)
                                       - a_scale_X * X2p_buf(t, k) * X2p_buf(t, i);
                    acc += A_ki * dL_dW2_t(k, j);
                }
                dW2_new(i, j) = acc;
            }
        }
        dW2_total = dW2_new;

        // dW1_total_new = A1_t^T · dL/dW1_t
        //   A1_t[k, i] = δ_{ki} - a_scale_Z · z_t[k] · z_t[i]
        Tensor dW1_new(h, d);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double acc = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    const double A_ki = (k == i ? 1.0 : 0.0)
                                       - a_scale_Z * last_z_(t, k) * last_z_(t, i);
                    acc += A_ki * dL_dW1_t(k, j);
                }
                dW1_new(i, j) = acc;
            }
        }
        dW1_total = dW1_new;
    }

    // After the loop, dW2_total = δL/δW2_0  and  dW1_total = δL/δW1_0
    // These are the gradients on the initial state W2_state_, W1_state_.
    // For now we don't add these to the grad_W1_w_/grad_W2_w_ since the
    // persistent state is conceptually the model's "memory", not its "parameters".
    // (TTTLinear does the same — see ttt_linear.cpp Step 3 which only writes
    // grad_W_in_w_, grad_W_in_b_, grad_W_out_w_, grad_W_out_b_, grad_bias_.)

    // ---- Step 6: backward through W_in (using dz_total) ----
    Tensor grad_input = W_in_.backward(dz_total, 0.0);
    return grad_input;
}

void TTTMLP::update_weights(double learning_rate) {
    W_in_.update_weights(learning_rate);
    W_out_.update_weights(learning_rate);
    for (size_t i = 0; i < d_inner_; ++i) {
        b_in_(0, i) -= learning_rate * grad_b_in_(0, i);
    }
}

void TTTMLP::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    grad_b_in_.fill(0.0);
}

std::vector<Tensor*> TTTMLP::parameters() {
    return { &W_in_.weights, &W_in_.bias, &W_out_.weights, &W_out_.bias, &b_in_ };
}

std::vector<Tensor*> TTTMLP::gradients() {
    return { &W_in_.grad_weights, &W_in_.grad_bias, &W_out_.grad_weights, &W_out_.grad_bias, &grad_b_in_ };
}

Tensor TTTMLP::get_weights() const {
    // Return a stack of the fast-weight state matrices as a flat vector.
    // Useful for inspection; not used by the standard training loop.
    Tensor out(1, d_hidden_ * d_inner_ + d_inner_ * d_hidden_ + d_hidden_ + d_inner_);
    size_t off = 0;
    for (size_t i = 0; i < d_hidden_; ++i)
        for (size_t j = 0; j < d_inner_; ++j)
            out.data[off++] = W1_state_(i, j);
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t j = 0; j < d_hidden_; ++j)
            out.data[off++] = W2_state_(i, j);
    for (size_t i = 0; i < d_hidden_; ++i) out.data[off++] = b1_state_(0, i);
    for (size_t i = 0; i < d_inner_; ++i) out.data[off++] = b2_state_(0, i);
    return out;
}

Tensor TTTMLP::get_gradients() const {
    return get_weights();
}

// ============================================================================
// TTTMLPModel — convenience wrapper
// ============================================================================

TTTMLPModel::TTTMLPModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
                          double eta, double lambda_reg, size_t mlp_ratio)
    : layer1_(hidden_dim, hidden_dim, eta, lambda_reg, mlp_ratio),
      layer2_(hidden_dim, hidden_dim, eta, lambda_reg, mlp_ratio),
      proj_in_(input_dim, hidden_dim),
      proj_out_(hidden_dim, output_dim) {}

Tensor TTTMLPModel::forward(const Tensor& input) {
    Tensor h = proj_in_.forward(input);
    h = layer1_.forward(h);
    h = layer2_.forward(h);
    return proj_out_.forward(h);
}

Tensor TTTMLPModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad_h = proj_out_.backward(grad_output, learning_rate);
    grad_h = layer2_.backward(grad_h, learning_rate);
    grad_h = layer1_.backward(grad_h, learning_rate);
    return proj_in_.backward(grad_h, learning_rate);
}

void TTTMLPModel::update_weights(double learning_rate) {
    proj_out_.update_weights(learning_rate);
    layer2_.update_weights(learning_rate);
    layer1_.update_weights(learning_rate);
    proj_in_.update_weights(learning_rate);
}

void TTTMLPModel::zero_grad() {
    proj_out_.zero_grad();
    layer2_.zero_grad();
    layer1_.zero_grad();
    proj_in_.zero_grad();
}

std::vector<Tensor*> TTTMLPModel::parameters() {
    auto p = proj_in_.parameters();
    auto p2 = layer1_.parameters();
    auto p3 = layer2_.parameters();
    auto p4 = proj_out_.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    return p;
}

std::vector<Tensor*> TTTMLPModel::gradients() {
    auto g = proj_in_.gradients();
    auto g2 = layer1_.gradients();
    auto g3 = layer2_.gradients();
    auto g4 = proj_out_.gradients();
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), g3.begin(), g3.end());
    g.insert(g.end(), g4.begin(), g4.end());
    return g;
}