#include "mamba3.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ============================================================================
// Mamba-3 implementation
// ============================================================================
//
// Per-channel complex arithmetic helper conventions:
//   We store complex numbers as (real, imag) 2-tuples in the last dimension.
//   Addition: (a_r, a_i) + (b_r, b_i) = (a_r + b_r, a_i + b_i)
//   Scalar × complex: (s · a_r, s · a_i)
//   Complex × complex (for the discretization): see m3_complex_div / mul
//     (a_r + i·a_i)(b_r + i·b_i) = (a_r·b_r - a_i·b_i) + i·(a_r·b_i + a_i·b_r)
//   Division by complex z = (b_r + i·b_i): multiply by conj/|z|².
//   |z|² = b_r² + b_i², conj = (b_r, -b_i).
//
// Numerical safety: when |denom|² < eps, we clamp to (eps, 0) — prevents
// division blowup in the trapezoidal form when Δ_t·A approaches 2.

// ---------- complex helpers ----------

static inline void m3_complex_add_inplace(double& ar, double& ai, double br, double bi) {
    ar += br;
    ai += bi;
}
static inline void m3_complex_scale_inplace(double& ar, double& ai, double s) {
    ar *= s;
    ai *= s;
}

// (a_r + i·a_i) / (b_r + i·b_i)  →  (a_r + i·a_i) · (b_r - i·b_i) / (b_r² + b_i²)
static inline void m3_complex_div(double ar, double ai, double br, double bi,
                                  double& out_r, double& out_i) {
    double denom = br * br + bi * bi;
    if (denom < 1e-30) {
        // Clamp: replace denom with epsilon (preserves direction at large b)
        denom = 1e-30;
        if (br < 0.0) br = -std::sqrt(denom);
        else          br =  std::sqrt(denom);
        bi = 0.0;
    }
    out_r = (ar * br + ai * bi) / denom;
    out_i = (ai * br - ar * bi) / denom;
}

// (a_r + i·a_i)(b_r + i·b_i) = (a_r·b_r - a_i·b_i) + i·(a_r·b_i + a_i·b_r)
static inline void m3_complex_mul(double ar, double ai, double br, double bi,
                                  double& out_r, double& out_i) {
    out_r = ar * br - ai * bi;
    out_i = ar * bi + ai * br;
}

// ---------- constructor ----------

Mamba3Block::Mamba3Block(size_t d_model, size_t n_heads, size_t d_inner)
    : d_model_(d_model),
      n_heads_(n_heads),
      d_inner_(d_inner == 0 ? 2 * d_model : d_inner),
      head_dim_(0),  // set after validation
      in_proj(d_model, 0),  // placeholder, will fix up below
      out_proj(0, d_model),
      dt_proj(d_model, n_heads),
      b_proj(d_model, 0)
{
    if (d_model == 0) {
        throw std::invalid_argument("Mamba3Block: d_model must be > 0");
    }
    if (n_heads == 0) {
        throw std::invalid_argument("Mamba3Block: n_heads must be > 0");
    }
    if (d_inner_ == 0) {
        throw std::invalid_argument("Mamba3Block: d_inner must be > 0");
    }
    if (d_inner_ % n_heads != 0) {
        throw std::invalid_argument("Mamba3Block: d_inner must be divisible by n_heads");
    }

    head_dim_ = d_inner_ / n_heads_;

    // Re-initialize Denses with correct shapes (in-place rebuild).
    // Use member-init-list + placement-new isn't easy without rtti; instead,
    // assign fresh Dense objects into the existing slots via copy-assignment.
    // The Dense class has copy assignment by default since the only fields
    // are Tensors (which copy by value).
    //
    // Simpler: use a delegate-constructor pattern — but C++ doesn't allow
    // delegating to a non-constructor, and Dense doesn't have a default
    // constructor (always (in, out)). So we use the in_proj(b_model, 0)
    // placeholder above, then rebuild in the body.
    //
    // Actually — the simplest reliable pattern is: create a fresh Dense
    // locally, then move-assign. But Dense may not have a move ctor
    // explicitly. We'll just construct-then-rebuild the Dense in place
    // by calling its constructor semantics via the standard Tensor copy.

    // Rebuild via direct assignment of new Tensors into the Dense slots.
    // Each Dense holds `weights` (out, in), `bias` (1, out), `grad_*`,
    // and `last_input`. We rebuild all of these.
    Dense new_in_proj(d_model, 2 * d_inner_);
    Dense new_out_proj(d_inner_, d_model);
    Dense new_dt_proj(d_model, n_heads_);
    Dense new_b_proj(d_model, d_inner_);
    in_proj = new_in_proj;
    out_proj = new_out_proj;
    dt_proj = new_dt_proj;
    b_proj = new_b_proj;

    // Per-channel complex-eigenvalue SSM parameters
    A_log       = Tensor(1, d_inner_);
    theta       = Tensor(1, d_inner_);
    theta_base  = Tensor(1, d_inner_);
    D_skip      = Tensor(1, d_inner_);
    dt_bias     = Tensor(1, n_heads_);
    grad_A_log_       = Tensor(1, d_inner_);
    grad_theta_       = Tensor(1, d_inner_);
    grad_theta_base_  = Tensor(1, d_inner_);
    grad_D_skip_      = Tensor(1, d_inner_);
    grad_dt_bias_     = Tensor(1, n_heads_);

    // Init A_log = 0.0 → exp(A_log) = 1.0 (decay magnitude = 1)
    for (size_t c = 0; c < d_inner_; ++c) A_log(0, c) = 0.0;
    // Init theta = 0 → no rotation at init
    for (size_t c = 0; c < d_inner_; ++c) theta(0, c) = 0.0;
    // Init theta_base = 0.01 (paper convention, small)
    for (size_t c = 0; c < d_inner_; ++c) theta_base(0, c) = 0.01;
    // Init D_skip = 1.0 (matching Mamba-1 convention)
    for (size_t c = 0; c < d_inner_; ++c) D_skip(0, c) = 1.0;
    // Init dt_bias = 1.0 (sigmoid(1) ≈ 0.731; softplus(sigmoid^-1(0.5)) ≈ 0.5)
    for (size_t h = 0; h < n_heads_; ++h) dt_bias(0, h) = 1.0;

    // Zero out gradient buffers
    for (size_t c = 0; c < d_inner_; ++c) {
        grad_A_log_(0, c) = 0.0;
        grad_theta_(0, c) = 0.0;
        grad_theta_base_(0, c) = 0.0;
        grad_D_skip_(0, c) = 0.0;
    }
    for (size_t h = 0; h < n_heads_; ++h) {
        grad_dt_bias_(0, h) = 0.0;
    }
}

// ---------- forward ----------

Tensor Mamba3Block::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("Mamba3Block: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("Mamba3Block: input must have at least one token");
    }

    last_input_ = input.clone();

    // Step 1: in_proj — (T, d_model) -> (T, 2*d_inner)
    last_p_ = in_proj.forward(input);

    // Split into ssm path and gate path
    last_x_ssm_ = Tensor(T, d_inner_);
    last_gate_  = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            last_x_ssm_(t, c) = last_p_(t, c);
            last_gate_(t, c)  = last_p_(t, d_inner_ + c);
        }
    }

    // Step 2: per-head scalar Δ_t and per-channel value b_t
    Tensor dt_pre = dt_proj.forward(input);  // (T, n_heads)
    last_dt_pre_ = Tensor(T, n_heads_);
    last_dt_     = Tensor(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double pre = dt_pre(t, h) + dt_bias(0, h);
            last_dt_pre_(t, h) = pre;
            // softplus(pre) = log(1 + exp(pre)) — numerically stable
            double v = (pre > 0.0) ? pre + std::log1p(std::exp(-pre)) : std::log1p(std::exp(pre));
            last_dt_(t, h) = v;
        }
    }

    last_b_ = b_proj.forward(input);  // (T, d_inner)

    // Step 3: Rotational input encoding (RoPE-style per channel)
    // Stored as (T * d_inner, 2): row (t * d_inner + c), col 0 = real, col 1 = imag.
    // angle[c, t] = t * theta_base[c]
    // x_ssm_t_rot[c, real] = x_ssm_t[c] * cos(angle)
    // x_ssm_t_rot[c, imag] = x_ssm_t[c] * sin(angle)
    last_x_ssm_rotated_ = Tensor(T * d_inner_, 2);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            double angle = static_cast<double>(t) * theta_base(0, c);
            double co = std::cos(angle);
            double si = std::sin(angle);
            double x_real = last_x_ssm_(t, c);
            size_t idx = t * d_inner_ + c;
            last_x_ssm_rotated_(idx, 0) = x_real * co;
            last_x_ssm_rotated_(idx, 1) = x_real * si;
        }
    }

    // Step 4: Trapezoidal discretization per channel
    // A[c] = exp(A_log[c]) * exp(i * theta[c]) = (mag*cos(theta), mag*sin(theta))
    // denom[c, t] = 1 - 0.5 * Δ_t * A[c]      (Δ_t comes from head h = c / head_dim_)
    // A_bar[c, t] = (1 + 0.5 * Δ_t * A[c]) / denom[c, t]
    // B_bar[c, t] = Δ_t * b[c, t] / denom[c, t]
    last_A_bar_ = Tensor(T * d_inner_, 2);
    last_B_bar_ = Tensor(T * d_inner_, 2);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            size_t h = c / head_dim_;
            size_t idx = t * d_inner_ + c;
            double dt_t = last_dt_(t, h);

            double mag = std::exp(A_log(0, c));  // decay magnitude ∈ (0, 1] at init
            double th  = theta(0, c);
            double A_r = mag * std::cos(th);
            double A_i = mag * std::sin(th);

            // denom = 1 - 0.5 * Δ_t * A
            double denom_r = 1.0 - 0.5 * dt_t * A_r;
            double denom_i = -0.5 * dt_t * A_i;

            // A_bar numerator = 1 + 0.5 * Δ_t * A
            double num_r = 1.0 + 0.5 * dt_t * A_r;
            double num_i = 0.5 * dt_t * A_i;

            double abr, abi;
            m3_complex_div(num_r, num_i, denom_r, denom_i, abr, abi);
            last_A_bar_(idx, 0) = abr;
            last_A_bar_(idx, 1) = abi;

            // B_bar = Δ_t * b[c, t] / denom = (Δ_t * b) * (1/denom) — same denominator
            double bval = dt_t * last_b_(t, c);
            double bbr, bbi;
            m3_complex_div(bval, 0.0, denom_r, denom_i, bbr, bbi);
            last_B_bar_(idx, 0) = bbr;
            last_B_bar_(idx, 1) = bbi;
        }
    }

    // Step 5: SSD recurrence
    // h_0 = 0 (complex)
    // h_t = A_bar_t · h_{t-1} + B_bar_t · x_ssm_rotated_t
    last_h_ = Tensor((T + 1) * d_inner_, 2);
    for (size_t c = 0; c < d_inner_; ++c) {
        last_h_(0 * d_inner_ + c, 0) = 0.0;
        last_h_(0 * d_inner_ + c, 1) = 0.0;
    }

    last_o_ = Tensor(T, d_inner_);

    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            size_t row_prev = t * d_inner_ + c;
            size_t row_curr = (t + 1) * d_inner_ + c;
            size_t idx_t = t * d_inner_ + c;

            // h_t = A_bar_t · h_{t-1} + B_bar_t · x_ssm_rotated_t
            double h_prev_r = last_h_(row_prev, 0);
            double h_prev_i = last_h_(row_prev, 1);

            double A_r = last_A_bar_(idx_t, 0);
            double A_i = last_A_bar_(idx_t, 1);
            double B_r = last_B_bar_(idx_t, 0);
            double B_i = last_B_bar_(idx_t, 1);
            double X_r = last_x_ssm_rotated_(idx_t, 0);
            double X_i = last_x_ssm_rotated_(idx_t, 1);

            // A · h_prev (complex mul)
            double ah_r = A_r * h_prev_r - A_i * h_prev_i;
            double ah_i = A_r * h_prev_i + A_i * h_prev_r;
            // B · x_rotated
            double bx_r = B_r * X_r - B_i * X_i;
            double bx_i = B_r * X_i + B_i * X_r;

            last_h_(row_curr, 0) = ah_r + bx_r;
            last_h_(row_curr, 1) = ah_i + bx_i;

            // o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t[c]
            last_o_(t, c) = last_h_(row_curr, 0) + D_skip(0, c) * last_x_ssm_(t, c);
        }
    }

    // Step 6: gate + output projection
    last_gated_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            double g = silu(last_gate_(t, c));
            last_gated_(t, c) = g * last_o_(t, c);
        }
    }

    return out_proj.forward(last_gated_);
}

// ---------- backward (full analytical BPTT) ----------

Tensor Mamba3Block::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("Mamba3Block: grad_output.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("Mamba3Block: grad_output must have at least one token");
    }

    // Step 6 (reverse of forward): chain through out_proj.
    // out_proj.forward took `last_gated_` (T × d_inner) → grad_output (T × d_model).
    // Its backward expects the same input shape and produces grad_input (T × d_inner).
    // We need to feed it the cached `last_gated_` and the upstream grad_output.
    // (Dense::backward mutates grad_weights/grad_bias internally and returns grad_input.)
    Tensor grad_gated = out_proj.backward(grad_output, learning_rate);

    // Step 5: chain through gated_t = silu(gate_t) ⊙ o_t.
    // grad_o_t[c] = silu(gate_t) ⊙ grad_gated_t[c]   (since ∂gated/∂o = silu(g))
    // grad_gate_t = silu'(gate_t) ⊙ o_t ⊙ grad_gated_t
    Tensor grad_o(T, d_inner_);
    Tensor grad_gate(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            double g  = last_gate_(t, c);
            double o  = last_o_(t, c);
            double sg = silu(g);
            double sd = silu_deriv(g);
            grad_o(t, c)   = sg * grad_gated(t, c);
            grad_gate(t, c) = sd * o * grad_gated(t, c);
        }
    }

    // Step 4 (chain through D_skip): o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t_raw[c]
    // grad_x_ssm_t_raw[c] += D_skip[c] · grad_o_t[c]
    // grad_D_skip[c]      += grad_o_t[c] · x_ssm_t_raw[c]
    Tensor grad_x_ssm_raw(T, d_inner_);
    for (size_t c = 0; c < d_inner_; ++c) grad_D_skip_(0, c) = 0.0;  // accumulate fresh
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            grad_x_ssm_raw(t, c) = D_skip(0, c) * grad_o(t, c);
            grad_D_skip_(0, c)  += grad_o(t, c) * last_x_ssm_(t, c);
        }
    }

    // Step 3: grad into h_t (complex).
    // o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t_raw[c]
    // ⇒ grad_h_t_real[c] = grad_o_t[c]; grad_h_t_imag[c] = 0.
    // We accumulate per-timestep into grad_h_ stored as ((T+1) * d_inner, 2).
    Tensor grad_h((T + 1) * d_inner_, 2);
    for (size_t i = 0; i < (T + 1) * d_inner_; ++i) {
        grad_h(i, 0) = 0.0;
        grad_h(i, 1) = 0.0;
    }
    // grad_h at t=T (the LAST used step) starts as grad_Re_h_T; we'll
    // accumulate into grad_h_(row_curr, :) as we go backwards.
    Tensor grad_h_t_total(T * d_inner_, 2);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            grad_h_t_total(t * d_inner_ + c, 0) = grad_o(t, c);  // Re(h_t) contributes
            grad_h_t_total(t * d_inner_ + c, 1) = 0.0;
        }
    }

    // Step 4: BPTT through the complex recurrence.
    // h_t = A_bar_t · h_{t-1} + B_bar_t · x_ssm_rotated_t  (complex)
    //
    // Chain rules (treating complex as 2D real with the convention that grad
    // is taken w.r.t. (real, imag) and the inner product ⟨grad, x⟩ means the
    // real-valued bilinear pairing):
    //   grad_A_bar_t[c] += grad_h_t_total[c] · conj(h_{t-1}[c])
    //                     (i.e. grad_A_bar_real = gR·hR + gI·hI,
    //                           grad_A_bar_imag = gI·hR - gR·hI)
    //   grad_B_bar_t[c] += grad_h_t_total[c] · conj(x_ssm_rotated_t[c])
    //                     (same form)
    //   grad_h_{t-1}_carried[c] += conj(A_bar_t[c]) · grad_h_t_total[c]
    //                     (conj·grad = (aR,-aI)·(gR,gI) = (aR·gR + aI·gI, aR·gI - aI·gR))
    //   grad_x_ssm_rotated_t[c] += conj(B_bar_t[c]) · grad_h_t_total[c]
    //
    // We process t from T-1 down to 0, accumulating grad_h at t-1.
    Tensor grad_x_ssm_rotated(T * d_inner_, 2);  // gradient w.r.t. rotated input
    for (size_t i = 0; i < T * d_inner_; ++i) {
        grad_x_ssm_rotated(i, 0) = 0.0;
        grad_x_ssm_rotated(i, 1) = 0.0;
    }

    // Per-timestep per-channel gradients from the trapezoidal step
    Tensor grad_A_bar_t(T * d_inner_, 2);
    Tensor grad_B_bar_t(T * d_inner_, 2);
    for (size_t i = 0; i < T * d_inner_; ++i) {
        grad_A_bar_t(i, 0) = 0.0; grad_A_bar_t(i, 1) = 0.0;
        grad_B_bar_t(i, 0) = 0.0; grad_B_bar_t(i, 1) = 0.0;
    }

    for (size_t t = T; t-- > 0; ) {  // t = T-1, T-2, ..., 0
        for (size_t c = 0; c < d_inner_; ++c) {
            size_t idx_t = t * d_inner_ + c;
            size_t row_prev = t * d_inner_ + c;          // h_{t-1}[c] lives here (in last_h_)
            size_t row_curr_h = (t + 1) * d_inner_ + c;  // not used; we just need grad_h_t_total[idx_t]

            double gR = grad_h_t_total(idx_t, 0);
            double gI = grad_h_t_total(idx_t, 1);
            double hR = last_h_(row_prev, 0);
            double hI = last_h_(row_prev, 1);
            double Xr = last_x_ssm_rotated_(idx_t, 0);
            double Xi = last_x_ssm_rotated_(idx_t, 1);
            double Ar = last_A_bar_(idx_t, 0);
            double Ai = last_A_bar_(idx_t, 1);
            double Br = last_B_bar_(idx_t, 0);
            double Bi = last_B_bar_(idx_t, 1);

            // grad_A_bar += grad_h · conj(h_prev)
            // grad_A_bar_real += gR·hR + gI·hI
            // grad_A_bar_imag += gI·hR - gR·hI
            grad_A_bar_t(idx_t, 0) += gR * hR + gI * hI;
            grad_A_bar_t(idx_t, 1) += gI * hR - gR * hI;
            // grad_B_bar += grad_h · conj(X)
            grad_B_bar_t(idx_t, 0) += gR * Xr + gI * Xi;
            grad_B_bar_t(idx_t, 1) += gI * Xr - gR * Xi;

            // grad_h_{t-1}_carried: chain through h_t = A_bar_t · h_{t-1}
            //   The Jacobian is the rotation-scaling matrix:
            //     ∂h_t_r/∂h_{t-1}_r = A_bar_r, ∂h_t_r/∂h_{t-1}_i = -A_bar_i
            //     ∂h_t_i/∂h_{t-1}_r = A_bar_i, ∂h_t_i/∂h_{t-1}_i = A_bar_r
            //   The transpose (chain rule adjoint) gives:
            //     ∂L/∂h_{t-1}_r += A_bar_r · gR + A_bar_i · gI
            //     ∂L/∂h_{t-1}_i += -A_bar_i · gR + A_bar_r · gI
            //   This is equivalent to "complex mul" of (A_bar_r + i·A_bar_i) by
            //   (gR + i·gI), but stored as (real, imag). The complex product is:
            //     (Ar + i·Ai)(gR + i·gI) = (Ar·gR - Ai·gI) + i·(Ar·gI + Ai·gR)
            //   Note: this is NOT what we want. We want (Ar·gR + Ai·gI) + i·(Ar·gI - Ai·gR),
            //   which is the conjugate-product: conj(A_bar) · g (since conj(A_bar)·g =
            //   (Ar - i·Ai)(gR + i·gI) = (Ar·gR + Ai·gI) + i·(Ar·gI - Ai·gR)).
            if (t > 0) {
                size_t idx_prev = (t - 1) * d_inner_ + c;
                grad_h_t_total(idx_prev, 0) += Ar * gR + Ai * gI;
                grad_h_t_total(idx_prev, 1) += Ar * gI - Ai * gR;
            }
            // grad_x_ssm_rotated += conj(B_bar) · grad_h
            // conj(B_bar) = (Br, -Bi); product (Br·gR + Bi·gI) + i·(Br·gI - Bi·gR)
            grad_x_ssm_rotated(idx_t, 0) += Br * gR + Bi * gI;
            grad_x_ssm_rotated(idx_t, 1) += Br * gI - Bi * gR;
        }
    }

    // Step 5: chain through trapezoidal discretization to get gradients for
    //   Δ_t (per head), A[c] (= A_log[c], theta[c]), b_t[c].
    //
    // Forward formulas:
    //   α = 0.5 · Δ_t · A[c]            (complex, A[c] = exp(A_log)·exp(i·θ))
    //   A_bar = (1 + α) / (1 - α)
    //   B_bar = Δ_t · b[c, t] / (1 - α)
    //
    // Derivative formulas (complex):
    //   dA_bar/dα = 2 / (1 - α)²
    //   dB_bar/dα = Δ_t · b / (1 - α)²
    //   dα/dΔ = 0.5 · A
    //   dα/dA = 0.5 · Δ
    //   dA/dA_log = A (since A = exp(A_log) · exp(i·θ), A_log is real log)
    //   dA/dθ = i · A   ⇒   dA_r/dθ = -A_i, dA_i/dθ = A_r
    //
    // Composing, for each channel c at timestep t:
    //   grad_α += grad_A_bar · 2/(1-α)² + grad_B_bar · Δ_t·b/(1-α)²
    //   grad_Δ += grad_α · 0.5·A_real_part_chain
    //     (need to handle real/imag properly: α is complex)
    //
    // Actually it's cleaner to compute the contribution to each parameter
    // directly. For real-valued parameters Δ_t (per head) and A_log[c], θ[c]
    // (per channel), we compute their partial derivatives by the chain rule
    // on the complex quantities.

    // Reset per-head / per-channel accumulators for the parameters with
    // multi-timestep contributions.
    for (size_t h = 0; h < n_heads_; ++h) grad_dt_bias_(0, h) = 0.0;
    // Note: grad_dt_pre_ doesn't exist (it folds into the Dense dt_proj grad
    // via backward); only grad_dt_bias is exposed. The per-timestep grad_dt
    // flows into dt_proj via grad_x_ssm chain through the projection.

    for (size_t c = 0; c < d_inner_; ++c) {
        grad_A_log_(0, c) = 0.0;
        grad_theta_(0, c) = 0.0;
    }
    Tensor grad_dt(T, n_heads_);
    Tensor grad_b(T, d_inner_);
    // Debug: track per-timestep grad_dt for diagnostic
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) grad_dt(t, h) = 0.0;
        for (size_t c = 0; c < d_inner_; ++c) grad_b(t, c) = 0.0;
    }

    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            size_t idx_t = t * d_inner_ + c;
            size_t h = c / head_dim_;

            double gA_r = grad_A_bar_t(idx_t, 0);
            double gA_i = grad_A_bar_t(idx_t, 1);
            double gB_r = grad_B_bar_t(idx_t, 0);
            double gB_i = grad_B_bar_t(idx_t, 1);

            double dt_t = last_dt_(t, h);

            double mag = std::exp(A_log(0, c));
            double th  = theta(0, c);
            double A_r = mag * std::cos(th);
            double A_i = mag * std::sin(th);

            // α = 0.5 · Δ_t · A   (complex)
            double alpha_r = 0.5 * dt_t * A_r;
            double alpha_i = 0.5 * dt_t * A_i;
            // 1 - α = (1 - α_r, -α_i)
            double denom_r = 1.0 - alpha_r;
            double denom_i = -alpha_i;
            double denom_norm_sq = denom_r * denom_r + denom_i * denom_i;
            if (denom_norm_sq < 1e-30) denom_norm_sq = 1e-30;

            // grad_α_r = gA_r · dA_bar_real/dα_r + gA_i · dA_bar_imag/dα_r
            //          + gB_r · dB_bar_real/dα_r + gB_i · dB_bar_imag/dα_r
            //
            // dA_bar/dα (Wirtinger) = 2/(1-α)² = 2·z²   (holomorphic f)
            //   dA_bar_real/dα_r = Re(2·z²), dA_bar_imag/dα_r = Im(2·z²)
            // dB_bar/dα (Wirtinger) = Δ·b/(1-α)² = Δ·b·z²
            //   dB_bar_real/dα_r = Re(Δ·b·z²), dB_bar_imag/dα_r = Im(Δ·b·z²)
            //
            // (Note: α is complex, but when α_r changes by δ (real), α changes
            //  by δ (real), so dA_bar/dα_r = Re(f') and dA_bar_imag/dα_r = Im(f').)

            // z = 1/(1-α) = conj(1-α) / |1-α|²
            double z_r = denom_r / denom_norm_sq;
            double z_i = -denom_i / denom_norm_sq;
            // z² = z·z (complex square)
            double z2_r = z_r * z_r - z_i * z_i;
            double z2_i = 2.0 * z_r * z_i;
            // dA_bar/dα = 2·z²
            double dAbar_da_r = 2.0 * z2_r;
            double dAbar_da_i = 2.0 * z2_i;
            // dB_bar/dα = Δ·b·z²
            double Db = dt_t * last_b_(t, c);
            double dBbar_da_r = Db * z2_r;
            double dBbar_da_i = Db * z2_i;

            // grad_α_r = gA_r · dAbar_da_r + gA_i · dAbar_da_i
            //          + gB_r · dBbar_da_r + gB_i · dBbar_da_i
            double grad_a_r = gA_r * dAbar_da_r + gA_i * dAbar_da_i
                            + gB_r * dBbar_da_r + gB_i * dBbar_da_i;
            double grad_a_i = 0.0;  // not used; we treat α_r, α_i as independent

            // Now α = 0.5 · Δ · A. For each real-valued parameter we want:
            //   ∂A_bar/∂Δ (treating A as fixed): dA_bar/dα · ∂α/∂Δ = dA_bar/dα · 0.5·A
            //   ∂A_bar/∂A_log: chain through A = exp(A_log)·exp(iθ)
            //     dA_bar/dA_r = dA_bar/dα · ∂α_r/∂A_r = dAbar_da_r · 0.5·Δ
            //     dA_bar/dA_i = dA_bar/dα · ∂α_i/∂A_i = dAbar_da_i · 0.5·Δ
            //     then dA_bar/dA_log = dA_bar/dA_r · A_r + dA_bar/dA_i · A_i
            //                         = (dAbar_da_r · A_r + dAbar_da_i · A_i) · 0.5·Δ
            //   dA_bar/dθ: chain through dA_r/dθ = -A_i, dA_i/dθ = A_r
            //     dA_bar/dθ = dA_bar/dA_r · (-A_i) + dA_bar/dA_i · A_r
            //               = (dAbar_da_r · (-A_i) + dAbar_da_i · A_r) · 0.5·Δ
            //
            // For B_bar = Δ·b/(1-α):
            //   ∂B_bar/∂Δ = b · z + Δ·b · dz/dΔ
            //             = b · z + Δ·b · 0.5·A · z²
            //   ∂B_bar/∂b = Δ · z
            //   ∂B_bar/∂A_log = (∂B_bar/∂α) · 0.5·Δ · A (same α chain)
            //   ∂B_bar/∂θ = (∂B_bar/∂α) · 0.5·Δ · (-A_i + i·A_r)? Actually
            //                dA/dθ = i·A ⇒ dA_r/dθ = -A_i, dA_i/dθ = A_r
            //                so ∂B_bar/∂θ = (∂B_bar/∂α_r) · (-A_i) + (∂B_bar/∂α_i) · A_r) · 0.5·Δ

            double half_dt = 0.5 * dt_t;
            // grad for Δ_t (per head; accumulate into grad_dt[t, h])
            //
            // Holomorphic chain:
            //   dA_bar/dΔ = dA_bar/dα · dα/dΔ
            //             = (2·z²) · (0.5·A) = z²·A
            //   ⇒ dA_bar_real/dΔ = Re(z²·A) = z²_r·A_r - z²_i·A_i
            //      dA_bar_imag/dΔ = Im(z²·A) = z²_r·A_i + z²_i·A_r
            //
            //   dB_bar/dΔ = b·z + Δ·b·dz/dΔ = b·z + Δ·b·z²·(0.5·A) = b·(z + 0.5·Δ·A·z²)
            //   ⇒ dB_bar_real/dΔ = b·z_r + b·0.5·Δ·(A_r·z²_r - A_i·z²_i)
            //      dB_bar_imag/dΔ = b·z_i + b·0.5·Δ·(A_r·z²_i + A_i·z²_r)
            //
            // contribution to grad_Δ_t = gA_r·dA_real/dΔ + gA_i·dA_imag/dΔ
            //                          + gB_r·dB_real/dΔ + gB_i·dB_imag/dΔ
            double bval = last_b_(t, c);
            double dA_real_dD = z2_r * A_r - z2_i * A_i;
            double dA_imag_dD = z2_r * A_i + z2_i * A_r;
            double dB_real_dD = bval * z_r + bval * half_dt * (A_r * z2_r - A_i * z2_i);
            double dB_imag_dD = bval * z_i + bval * half_dt * (A_r * z2_i + A_i * z2_r);

            double contrib_D = gA_r * dA_real_dD + gA_i * dA_imag_dD
                             + gB_r * dB_real_dD + gB_i * dB_imag_dD;
            grad_dt(t, h) += contrib_D;

            // grad for A_log[c]: chain through A = exp(A_log)·exp(iθ)
            //   dA_bar/dA_log = dA_bar/dA_r · A_r + dA_bar/dA_i · A_i
            //   dA_bar/dA_r = dAbar_da_r · 0.5·Δ
            //   dA_bar/dA_i = dAbar_da_i · 0.5·Δ
            //   dB_bar/dA_log similarly
            double half_DA = half_dt;  // ∂α_r/∂A_r = 0.5·Δ, same for α_i
            double dA_real_dAlog = half_DA * (dAbar_da_r * A_r - dAbar_da_i * A_i);
            double dA_imag_dAlog = half_DA * (dAbar_da_r * A_i + dAbar_da_i * A_r);
            double dB_real_dAlog = half_DA * (dBbar_da_r * A_r - dBbar_da_i * A_i);
            double dB_imag_dAlog = half_DA * (dBbar_da_r * A_i + dBbar_da_i * A_r);
            double contrib_Alog = gA_r * dA_real_dAlog + gA_i * dA_imag_dAlog
                                + gB_r * dB_real_dAlog + gB_i * dB_imag_dAlog;
            grad_A_log_(0, c) += contrib_Alog;

            // grad for theta[c]: dA/dθ = i·A ⇒ dA_r/dθ = -A_i, dA_i/dθ = A_r
            double dA_real_dth = half_DA * (dAbar_da_r * (-A_i) - dAbar_da_i * A_r);
            double dA_imag_dth = half_DA * (dAbar_da_r * A_r   + dAbar_da_i * (-A_i));
            double dB_real_dth = half_DA * (dBbar_da_r * (-A_i) - dBbar_da_i * A_r);
            double dB_imag_dth = half_DA * (dBbar_da_r * A_r   + dBbar_da_i * (-A_i));
            double contrib_th = gA_r * dA_real_dth + gA_i * dA_imag_dth
                              + gB_r * dB_real_dth + gB_i * dB_imag_dth;
            grad_theta_(0, c) += contrib_th;

            // grad for b_t[c]: dB_bar/db = Δ · z
            //   dB_bar_real/db = Δ·z_r
            //   dB_bar_imag/db = Δ·z_i
            double dB_real_db = dt_t * z_r;
            double dB_imag_db = dt_t * z_i;
            double contrib_b = gB_r * dB_real_db + gB_i * dB_imag_db;
            grad_b(t, c) = contrib_b;
            // (b_proj.grad_weights accumulated separately below)
        }
    }

    // Chain through b_proj.backward to get the input gradient contribution from b_t.
    // b_proj.backward also accumulates b_proj.grad_weights/grad_bias internally,
    // so we don't manually accumulate here.
    Tensor grad_input_from_b = b_proj.backward(grad_b, learning_rate);

    // Step 6: chain grad_dt into grad_dt_bias (per head, sum over t) and
    // grad_dt_proj.bias / grad_dt_proj.weights (via Dense backward).
    // The chain is:
    //   Δ_t = softplus(dt_pre_t + dt_bias[h])
    //   grad_dt_pre_t[h] = grad_dt(t, h) · softplus_deriv(dt_pre_t[h] + dt_bias[h])
    //   grad_dt_bias[h]   += grad_dt(t, h) · softplus_deriv(...)
    Tensor grad_dt_pre(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            // Note: last_dt_pre_(t, h) is cached as dt_proj.forward + dt_bias
            // (see forward), so z = last_dt_pre_(t, h) is already the full
            // softplus input. softplus'(z) = sigmoid(z).
            double z = last_dt_pre_(t, h);
            double s = sigmoid(z);
            grad_dt_pre(t, h) = grad_dt(t, h) * s;
            // grad_dt_bias is OUR own accumulator (dt_bias is a non-Dense Tensor
            // parameter; dt_proj.backward handles the Dense ones).
            grad_dt_bias_(0, h) += grad_dt_pre(t, h);
        }
    }
    // Chain through dt_proj.backward to get the input gradient contribution from Δ_t.
    // dt_proj.backward also accumulates dt_proj.grad_weights/grad_bias internally.
    Tensor grad_input_from_dt = dt_proj.backward(grad_dt_pre, learning_rate);

    // Step 7: undo RoPE rotation to recover grad_x_ssm_raw from grad_x_ssm_rotated.
    // Forward: x_rot[c, real] = x_raw[c] · cos(angle), x_rot[c, imag] = x_raw[c] · sin(angle)
    //   where angle = t · theta_base[c]
    // Reverse: grad_x_raw[c] = grad_x_rot_real · cos(angle) + grad_x_rot_imag · sin(angle)
    //          grad_theta_base[c] += sum_t (grad_x_rot_real · (-x_raw·sin·t) + grad_x_rot_imag · (x_raw·cos·t))
    for (size_t c = 0; c < d_inner_; ++c) grad_theta_base_(0, c) = 0.0;
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            double angle = static_cast<double>(t) * theta_base(0, c);
            double co = std::cos(angle), si = std::sin(angle);
            size_t idx_t = t * d_inner_ + c;
            double grR = grad_x_ssm_rotated(idx_t, 0);
            double grI = grad_x_ssm_rotated(idx_t, 1);
            // grad_x_ssm_raw contribution from the rotation
            double contrib = grR * co + grI * si;
            grad_x_ssm_raw(t, c) += contrib;
            // grad_theta_base[c] += (grR · d(x_rot_real)/dθ_base + grI · d(x_rot_imag)/dθ_base)
            // d(x_rot_real)/dθ_base = -x_raw · sin · t
            // d(x_rot_imag)/dθ_base = +x_raw · cos · t
            double x_raw = last_x_ssm_(t, c);
            grad_theta_base_(0, c) += grR * (-x_raw * si * static_cast<double>(t))
                                    + grI * ( x_raw * co * static_cast<double>(t));
        }
    }

    // Step 8: concatenate grad_x_ssm_raw and grad_gate into grad_p (T × 2*d_inner)
    Tensor grad_p(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_inner_; ++c) {
            grad_p(t, c)             = grad_x_ssm_raw(t, c);
            grad_p(t, d_inner_ + c)  = grad_gate(t, c);
        }
    }

    // Step 9: chain through in_proj.backward to get grad_input and accumulate W_inproj / b_inproj grads.
    Tensor grad_input_from_inproj = in_proj.backward(grad_p, learning_rate);
    Tensor grad_input = grad_input_from_inproj + grad_input_from_b + grad_input_from_dt;

    return grad_input;
}

// ---------- zero_grad / parameters / gradients / update_weights ----------

void Mamba3Block::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    dt_proj.zero_grad();
    b_proj.zero_grad();
    for (size_t c = 0; c < d_inner_; ++c) {
        grad_A_log_(0, c)       = 0.0;
        grad_theta_(0, c)       = 0.0;
        grad_theta_base_(0, c)  = 0.0;
        grad_D_skip_(0, c)      = 0.0;
    }
    for (size_t h = 0; h < n_heads_; ++h) {
        grad_dt_bias_(0, h) = 0.0;
    }
}

std::vector<Tensor*> Mamba3Block::parameters() {
    std::vector<Tensor*> params;
    for (Tensor* p : in_proj.parameters()) params.push_back(p);
    for (Tensor* p : out_proj.parameters()) params.push_back(p);
    for (Tensor* p : dt_proj.parameters()) params.push_back(p);
    for (Tensor* p : b_proj.parameters()) params.push_back(p);
    params.push_back(&A_log);
    params.push_back(&theta);
    params.push_back(&theta_base);
    params.push_back(&D_skip);
    params.push_back(&dt_bias);
    return params;
}

std::vector<Tensor*> Mamba3Block::gradients() {
    std::vector<Tensor*> grads;
    for (Tensor* g : in_proj.gradients()) grads.push_back(g);
    for (Tensor* g : out_proj.gradients()) grads.push_back(g);
    for (Tensor* g : dt_proj.gradients()) grads.push_back(g);
    for (Tensor* g : b_proj.gradients()) grads.push_back(g);
    grads.push_back(&grad_A_log_);
    grads.push_back(&grad_theta_);
    grads.push_back(&grad_theta_base_);
    grads.push_back(&grad_D_skip_);
    grads.push_back(&grad_dt_bias_);
    return grads;
}

void Mamba3Block::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    dt_proj.update_weights(learning_rate);
    b_proj.update_weights(learning_rate);
    // Per-channel parameters
    for (size_t c = 0; c < d_inner_; ++c) {
        A_log(0, c)       -= learning_rate * grad_A_log_(0, c);
        theta(0, c)       -= learning_rate * grad_theta_(0, c);
        theta_base(0, c)  -= learning_rate * grad_theta_base_(0, c);
        D_skip(0, c)      -= learning_rate * grad_D_skip_(0, c);
    }
    for (size_t h = 0; h < n_heads_; ++h) {
        dt_bias(0, h) -= learning_rate * grad_dt_bias_(0, h);
    }
}
