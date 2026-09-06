#include "mamba_conv.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ============================================================================
// MambaConv — Canonical Mamba-1 with depthwise causal 1D convolution
//   Gu & Dao 2023, https://arxiv.org/abs/2312.00752
//
// Storage convention (matches `MambaBlock`):
//   All 3D intermediates with shape (T, d_inner, d_state) are stored as
//   2D tensors of shape (T * d_inner, d_state). Indexing uses
//   `tensor(t * d_inner + i, d)` for the (t, i, d) entry.
// ============================================================================

MambaConvBlock::MambaConvBlock(size_t d_model, size_t d_state, size_t d_inner, size_t conv_kernel)
    : in_proj(d_model, 0),
      out_proj(0, d_model),
      dt_proj(d_model, 0),
      B_proj(d_model, d_state),
      C_proj(d_model, d_state),
      A_log(0, 0),
      D_skip(1, 0),
      conv_weight(0, 0),
      conv_bias(1, 0),
      d_model_(d_model),
      d_state_(d_state),
      d_inner_(d_inner),
      conv_kernel_(conv_kernel)
{
    if (d_model == 0) throw std::invalid_argument("MambaConvBlock: d_model must be > 0");
    if (conv_kernel == 0) throw std::invalid_argument("MambaConvBlock: conv_kernel must be > 0");
    if (d_inner == 0) d_inner = 2 * d_model;
    d_inner_ = d_inner;

    in_proj  = Dense(d_model, 2 * d_inner);
    dt_proj  = Dense(d_model, d_inner);
    out_proj = Dense(d_inner, d_model);

    A_log = Tensor(d_inner, d_state);
    std::mt19937 gen(42);
    std::normal_distribution<> A_dis(0.0, 1.0);
    for (size_t i = 0; i < d_inner; ++i)
        for (size_t d = 0; d < d_state; ++d)
            A_log(i, d) = A_dis(gen) * 0.5;

    D_skip = Tensor(1, d_inner);
    for (size_t i = 0; i < d_inner; ++i) D_skip(0, i) = 1.0;

    conv_weight = Tensor(d_inner, conv_kernel);
    conv_bias   = Tensor(1, d_inner);
    double init_scale = 1.0 / std::sqrt((double)conv_kernel);
    std::normal_distribution<> cw_dis(0.0, init_scale);
    for (size_t i = 0; i < d_inner; ++i)
        for (size_t k = 0; k < conv_kernel; ++k)
            conv_weight(i, k) = cw_dis(gen);
    for (size_t i = 0; i < d_inner; ++i) conv_bias(0, i) = 0.0;

    grad_A_log_       = Tensor(d_inner, d_state);
    grad_D_skip_      = Tensor(1, d_inner);
    grad_conv_weight_ = Tensor(d_inner, conv_kernel);
    grad_conv_bias_   = Tensor(1, d_inner);

    in_proj.bias.fill(0.0);
    dt_proj.bias.fill(0.0);
    B_proj.bias.fill(0.0);
    C_proj.bias.fill(0.0);
    out_proj.bias.fill(0.0);
}

void MambaConvBlock::zero_grad() {
    in_proj.zero_grad();
    dt_proj.zero_grad();
    out_proj.zero_grad();
    B_proj.zero_grad();
    C_proj.zero_grad();
    grad_A_log_.fill(0.0);
    grad_D_skip_.fill(0.0);
    grad_conv_weight_.fill(0.0);
    grad_conv_bias_.fill(0.0);
}

std::vector<Tensor*> MambaConvBlock::parameters() {
    return {
        &in_proj.weights, &in_proj.bias,
        &dt_proj.weights, &dt_proj.bias,
        &out_proj.weights, &out_proj.bias,
        &B_proj.weights, &B_proj.bias,
        &C_proj.weights, &C_proj.bias,
        &A_log, &D_skip,
        &conv_weight, &conv_bias
    };
}

std::vector<Tensor*> MambaConvBlock::gradients() {
    return {
        &in_proj.grad_weights, &in_proj.grad_bias,
        &dt_proj.grad_weights, &dt_proj.grad_bias,
        &out_proj.grad_weights, &out_proj.grad_bias,
        &B_proj.grad_weights, &B_proj.grad_bias,
        &C_proj.grad_weights, &C_proj.grad_bias,
        &grad_A_log_, &grad_D_skip_,
        &grad_conv_weight_, &grad_conv_bias_
    };
}

void MambaConvBlock::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    dt_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    B_proj.update_weights(learning_rate);
    C_proj.update_weights(learning_rate);

    for (size_t i = 0; i < A_log.rows; ++i)
        for (size_t j = 0; j < A_log.cols; ++j)
            A_log(i, j) -= learning_rate * grad_A_log_(i, j);

    for (size_t i = 0; i < D_skip.cols; ++i)
        D_skip(0, i) -= learning_rate * grad_D_skip_(0, i);

    for (size_t i = 0; i < conv_weight.rows; ++i)
        for (size_t j = 0; j < conv_weight.cols; ++j)
            conv_weight(i, j) -= learning_rate * grad_conv_weight_(i, j);

    for (size_t i = 0; i < conv_bias.cols; ++i)
        conv_bias(0, i) -= learning_rate * grad_conv_bias_(0, i);
}

// ----------------------------------------------------------------------------
// forward
// ----------------------------------------------------------------------------
Tensor MambaConvBlock::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("MambaConvBlock: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("MambaConvBlock: input must have at least one token");
    }
    last_input_ = input.clone();

    // Step 1: in_proj → split
    last_p_ = in_proj.forward(input);
    last_x_pre_ = Tensor(T, d_inner_);
    last_gate_  = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i) {
            last_x_pre_(t, i) = last_p_(t, i);
            last_gate_(t, i)  = last_p_(t, d_inner_ + i);
        }

    // Step 2: depthwise causal 1D conv on x_pre
    last_x_conv_ = Tensor(T, d_inner_);
    for (size_t i = 0; i < d_inner_; ++i) {
        for (size_t t = 0; t < T; ++t) {
            double acc = conv_bias(0, i);
            for (size_t j = 0; j < conv_kernel_; ++j) {
                if (t >= j) acc += conv_weight(i, j) * last_x_pre_(t - j, i);
            }
            last_x_conv_(t, i) = acc;
        }
    }

    // Step 3: SiLU → x_ssm
    last_ssm_in_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            last_ssm_in_(t, i) = silu(last_x_conv_(t, i));

    // Step 4: input-dependent SSM parameters
    last_Delta_pre_ = dt_proj.forward(input);
    last_Delta_     = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            last_Delta_(t, i) = softplus(last_Delta_pre_(t, i));
    last_B_t_ = B_proj.forward(input);
    last_C_t_ = C_proj.forward(input);

    // Step 5: discretisation — flattened 2D storage (T*d_inner, d_state)
    last_A_bar_ = Tensor(T * d_inner_, d_state_);
    last_B_bar_ = Tensor(T * d_inner_, d_state_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            size_t row = t * d_inner_ + i;
            for (size_t d = 0; d < d_state_; ++d) {
                double A = -std::exp(A_log(i, d));
                last_A_bar_(row, d) = std::exp(last_Delta_(t, i) * A);
                last_B_bar_(row, d) = last_Delta_(t, i) * last_B_t_(t, d);
            }
        }
    }

    // Step 6: selective scan. last_h_ is (T+1)*d_inner rows × d_state cols.
    // row t*d_inner + i = h_t[i, d]
    last_h_ = Tensor((T + 1) * d_inner_, d_state_);
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t d = 0; d < d_state_; ++d)
            last_h_(0 * d_inner_ + i, d) = 0.0;
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t d = 0; d < d_state_; ++d) {
                size_t row_prev = t * d_inner_ + i;
                size_t row_curr = (t + 1) * d_inner_ + i;
                double prev = last_h_(row_prev, d);
                double curr = last_A_bar_(row_prev, d) * prev
                            + last_B_bar_(row_prev, d) * last_ssm_in_(t, i);
                last_h_(row_curr, d) = curr;
            }
        }
    }

    // Step 7: y_t[i] = Σ_d C_t[d] * h_t[i, d]
    last_y_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double s = 0.0;
            for (size_t d = 0; d < d_state_; ++d)
                s += last_C_t_(t, d) * last_h_((t + 1) * d_inner_ + i, d);
            last_y_(t, i) = s;
        }
    }

    // Step 8: gating + skip
    last_gated_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i) {
            double sg = silu(last_gate_(t, i));
            last_gated_(t, i) = sg * (last_y_(t, i) + D_skip(0, i) * last_ssm_in_(t, i));
        }

    return out_proj.forward(last_gated_);
}

// ----------------------------------------------------------------------------
// backward
// ----------------------------------------------------------------------------
Tensor MambaConvBlock::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("MambaConvBlock.backward: grad_output.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("MambaConvBlock.backward: grad_output must have at least one token");
    }

    // ----- Step A: out_proj backward → d_gated -------------------------------
    Tensor d_gated = out_proj.backward(grad_output, learning_rate);

    // ----- Step B: split gated = silu(gate) * (y + D_skip * x_ssm) ----------
    Tensor d_y        = Tensor(T, d_inner_);
    Tensor d_xssm_skip = Tensor(T, d_inner_);
    Tensor d_silu_gate = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i) {
            double sg = silu(last_gate_(t, i));
            d_y(t, i)         = d_gated(t, i) * sg;
            d_xssm_skip(t, i) = d_gated(t, i) * sg * D_skip(0, i);
            d_silu_gate(t, i) = d_gated(t, i) * (last_y_(t, i) + D_skip(0, i) * last_ssm_in_(t, i));
        }
    // grad_D_skip
    for (size_t i = 0; i < d_inner_; ++i) {
        double s = 0.0;
        for (size_t t = 0; t < T; ++t)
            s += d_gated(t, i) * silu(last_gate_(t, i)) * last_ssm_in_(t, i);
        grad_D_skip_(0, i) += s;
    }
    // d_gate = d_silu_gate * silu'(gate)
    Tensor d_gate = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            d_gate(t, i) = d_silu_gate(t, i) * silu_deriv(last_gate_(t, i));

    // ----- Step C: SSM BPTT ---------------------------------------------------
    // grad_C_proj, grad_B_proj contributions: accumulate per token.
    // d_C_t[d] = Σ_i d_y[t, i] * h_t[i, d]   (the direct path)
    // d_B_t[d] = Σ_i d_B̄_t[i, d] * Δ_t[i]   (from B̄_t = Δ_t * B_t)
    Tensor grad_C_proj_local = Tensor(d_model_, d_state_);
    Tensor grad_B_proj_local = Tensor(d_model_, d_state_);
    Tensor d_C_proj_bias = Tensor(1, d_state_);
    Tensor d_B_proj_bias = Tensor(1, d_state_);
    grad_C_proj_local.fill(0.0);
    grad_B_proj_local.fill(0.0);
    d_C_proj_bias.fill(0.0);
    d_B_proj_bias.fill(0.0);

    // Per-token d_C_t and d_B_t (used for both input-grad and bias grad).
    Tensor d_C_t_per_t = Tensor(T, d_state_);
    Tensor d_B_t_per_t = Tensor(T, d_state_);

    // Temporary storage for d_A_bar, d_B_bar (T, d_inner, d_state) flat.
    Tensor d_A_bar_local = Tensor(T * d_inner_, d_state_);
    Tensor d_B_bar_local = Tensor(T * d_inner_, d_state_);
    d_A_bar_local.fill(0.0);
    d_B_bar_local.fill(0.0);

    // d_Delta[t, i] accumulates contributions from d_A_bar + d_B_bar.
    Tensor d_Delta = Tensor(T, d_inner_);
    d_Delta.fill(0.0);

    // Walk t from T down to 1, maintaining d_h_t (current state gradient).
    Tensor d_h = Tensor(d_inner_, d_state_);
    d_h.fill(0.0);

    for (size_t t_idx = T; t_idx >= 1; --t_idx) {
        size_t t = t_idx - 1;
        // Add direct d_y contribution: d_h[i, d] += d_y[t, i] * C_t[d]
        for (size_t i = 0; i < d_inner_; ++i)
            for (size_t d = 0; d < d_state_; ++d)
                d_h(i, d) += d_y(t, i) * last_C_t_(t, d);

        // At this point, d_h = d_h_t (current step's state gradient).
        // Accumulate gradients and propagate.
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t d = 0; d < d_state_; ++d) {
                double dh = d_h(i, d);
                if (dh == 0.0) continue;
                size_t row = t * d_inner_ + i;
                // d_xssm[t, i] += Σ_d B̄_t[i, d] * dh  (accumulated separately below)
                d_A_bar_local(row, d) += last_h_(t * d_inner_ + i, d) * dh;
                d_B_bar_local(row, d) += last_ssm_in_(t, i) * dh;
                // Propagate to d_h_{t-1}
                d_h(i, d) = last_A_bar_(row, d) * dh;
            }
        }
    }

    // Now translate d_A_bar_local, d_B_bar_local → grad_A_log, grad_B_proj, d_Delta, d_xssm.
    // And d_y → grad_C_proj and d_C_t_per_t.
    Tensor d_xssm = Tensor(T, d_inner_);
    d_xssm.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        // d_C_t[d] = Σ_i d_y[t, i] * h_t[i, d]
        // C_proj.weights is (d_state, d_model): grad_C_proj.weights(d, j) += input(t, j) * d_C_t(d)
        for (size_t d = 0; d < d_state_; ++d) {
            double s = 0.0;
            for (size_t i = 0; i < d_inner_; ++i)
                s += d_y(t, i) * last_h_((t + 1) * d_inner_ + i, d);
            d_C_t_per_t(t, d) = s;
            d_C_proj_bias(0, d) += s;
            for (size_t j = 0; j < d_model_; ++j)
                grad_C_proj_local(j, d) += last_input_(t, j) * s;
        }
        // d_B_t[d] = Σ_i d_B̄_t[i, d] * Δ_t[i]
        // B_proj.weights is (d_state, d_model): grad_B_proj.weights(d, j) += input(t, j) * d_B_t(d)
        for (size_t d = 0; d < d_state_; ++d) {
            double s = 0.0;
            for (size_t i = 0; i < d_inner_; ++i)
                s += d_B_bar_local(t * d_inner_ + i, d) * last_Delta_(t, i);
            d_B_t_per_t(t, d) = s;
            d_B_proj_bias(0, d) += s;
            for (size_t j = 0; j < d_model_; ++j)
                grad_B_proj_local(j, d) += last_input_(t, j) * s;
        }
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t d = 0; d < d_state_; ++d) {
                double d_A = d_A_bar_local(t * d_inner_ + i, d);
                double d_B = d_B_bar_local(t * d_inner_ + i, d);
                double A_val = -std::exp(A_log(i, d));
                double A_bar = last_A_bar_(t * d_inner_ + i, d);
                if (d_A != 0.0) {
                    // d/dΔ Ā_t = A * Ā_t, so d_Delta += d_A_bar * A * Ā_t
                    d_Delta(t, i) += d_A * A_val * A_bar;
                    // d/dA_log Ā_t = d/dA_log exp(Δ * A) = Δ * A * exp(Δ*A) = Δ * A * Ā_t,
                    // and A = -exp(A_log) so dA/dA_log = -exp(A_log) = A.
                    grad_A_log_(i, d) += d_A * last_Delta_(t, i) * A_val * A_bar;
                }
                if (d_B != 0.0) {
                    d_Delta(t, i) += d_B * last_B_t_(t, d);
                }
            }
        }
    }

    // Re-walk to populate d_xssm cleanly: at each step, after d_y+C contributes
    // d_h, d_h holds d_h_t. The xssm contribution from this step is Σ_d B̄ * d_h_t.
    {
        Tensor dh = Tensor(d_inner_, d_state_);
        dh.fill(0.0);
        for (size_t t_idx = T; t_idx >= 1; --t_idx) {
            size_t t = t_idx - 1;
            for (size_t i = 0; i < d_inner_; ++i)
                for (size_t d = 0; d < d_state_; ++d)
                    dh(i, d) += d_y(t, i) * last_C_t_(t, d);
            for (size_t i = 0; i < d_inner_; ++i) {
                double sx = 0.0;
                for (size_t d = 0; d < d_state_; ++d) {
                    sx += last_B_bar_(t * d_inner_ + i, d) * dh(i, d);
                    dh(i, d) = last_A_bar_(t * d_inner_ + i, d) * dh(i, d);
                }
                d_xssm(t, i) += sx;
            }
        }
    }

    // ----- Step E: Δ → dt_proj (via softplus) ---------------------------------
    Tensor grad_dt_proj_local = Tensor(d_model_, d_inner_);
    Tensor d_dt_proj_bias = Tensor(1, d_inner_);
    grad_dt_proj_local.fill(0.0);
    d_dt_proj_bias.fill(0.0);

    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double sig = sigmoid(last_Delta_pre_(t, i));
            double z_grad = d_Delta(t, i) * sig;
            for (size_t j = 0; j < d_model_; ++j)
                grad_dt_proj_local(j, i) += z_grad * last_input_(t, j);
            d_dt_proj_bias(0, i) += z_grad;
        }
    }

    // ----- Step F: accumulate into Dense gradients ----------------------------
    // Note: B_proj.weights, C_proj.weights have shape (d_state, d_model) per the Dense
    // convention `weights(out_features, in_features)`. dt_proj.weights is (d_inner, d_model).
    for (size_t j = 0; j < d_model_; ++j)
        for (size_t i = 0; i < d_inner_; ++i)
            dt_proj.grad_weights(i, j) += grad_dt_proj_local(j, i);
    for (size_t i = 0; i < d_inner_; ++i)
        dt_proj.grad_bias(0, i) += d_dt_proj_bias(0, i);
    for (size_t d = 0; d < d_state_; ++d)
        for (size_t j = 0; j < d_model_; ++j) {
            C_proj.grad_weights(d, j) += grad_C_proj_local(j, d);
            B_proj.grad_weights(d, j) += grad_B_proj_local(j, d);
        }
    for (size_t d = 0; d < d_state_; ++d) {
        C_proj.grad_bias(0, d) += d_C_proj_bias(0, d);
        B_proj.grad_bias(0, d) += d_B_proj_bias(0, d);
    }

    // ----- Step G: d_xssm → d_xconv (via SiLU) --------------------------------
    Tensor d_xssm_total = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            d_xssm_total(t, i) = d_xssm(t, i) + d_xssm_skip(t, i);
    Tensor d_xconv = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            d_xconv(t, i) = d_xssm_total(t, i) * silu_deriv(last_x_conv_(t, i));

    // ----- Step H: depthwise causal conv backward -----------------------------
    for (size_t i = 0; i < d_inner_; ++i) {
        double sb = 0.0;
        for (size_t t = 0; t < T; ++t) sb += d_xconv(t, i);
        grad_conv_bias_(0, i) += sb;
        for (size_t j = 0; j < conv_kernel_; ++j) {
            double sw = 0.0;
            for (size_t t = j; t < T; ++t)
                sw += d_xconv(t, i) * last_x_pre_(t - j, i);
            grad_conv_weight_(i, j) += sw;
        }
    }
    Tensor d_xpre = Tensor(T, d_inner_);
    d_xpre.fill(0.0);
    for (size_t i = 0; i < d_inner_; ++i) {
        for (size_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (size_t j = 0; j < conv_kernel_; ++j) {
                if (t + j < T) s += conv_weight(i, j) * d_xconv(t + j, i);
            }
            d_xpre(t, i) = s;
        }
    }

    // ----- Step I: assemble d_p and run in_proj.backward ---------------------
    Tensor d_p(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            d_p(t, i)            = d_xpre(t, i);
            d_p(t, d_inner_ + i) = d_gate(t, i);
        }
    }
    Tensor d_input_from_inproj = in_proj.backward(d_p, learning_rate);

    // ----- Step J: input grad from dt_proj, B_proj, C_proj ---------------------
    Tensor d_x_dt = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                double sig = sigmoid(last_Delta_pre_(t, i));
                s += d_Delta(t, i) * sig * dt_proj.weights(i, j);
            }
            d_x_dt(t, j) = s;
        }
    Tensor d_x_B = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t d = 0; d < d_state_; ++d)
                s += d_B_t_per_t(t, d) * B_proj.weights(d, j);
            d_x_B(t, j) = s;
        }
    Tensor d_x_C = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t d = 0; d < d_state_; ++d)
                s += d_C_t_per_t(t, d) * C_proj.weights(d, j);
            d_x_C(t, j) = s;
        }

    Tensor d_input(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            d_input(t, j) = d_input_from_inproj(t, j) + d_x_dt(t, j) + d_x_B(t, j) + d_x_C(t, j);

    return d_input;
}
