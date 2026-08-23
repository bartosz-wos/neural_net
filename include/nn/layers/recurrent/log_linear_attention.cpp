#include "log_linear_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// ----------------------------------------------------------------------------
// LogLinearAttention implementation — Guo, Yang, Goel, Xing, Dao, Kim, ICLR 2026
//
// The recurrent form (Eq. 3) of the paper: at each step t, per head h, per
// level ℓ, we maintain a state S^(ℓ)_t[h] ∈ R^{head_dim × head_dim}
// (since we set head_dim = d_state for log-linear attention, the per-head
// state is a square matrix). Output is a weighted sum of q_t^T · S^(ℓ)_t
// across all L levels, where the weights λ^(ℓ)_t are learnable.
//
// Fenwick-tree state update (page 5 recurrence):
//   lssb(t+1) = index of least significant set bit of (t+1)
//   For ℓ in 0..L-1:
//     S^(ℓ)_t = {
//       b_t ⊗ k_t                                 if ℓ == 0
//       0                                         if 0 < ℓ ≤ lssb(t+1)
//       Σ_{ℓ'<ℓ} a_t · S^(ℓ')_{t-1}               if ℓ == lssb(t+1)+1 and ℓ < L
//       a_t · S^(ℓ)_{t-1}                         if ℓ > lssb(t+1)+1
//     }
//
// Output:
//   o_t[h, dh] = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · Σ_{ds} q_t[h, dh, ds] · S^(ℓ)_t[h, dh, ds]
// ----------------------------------------------------------------------------

static inline double lla_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}
static inline double lla_silu(double x) {
    return x * lla_sigmoid(x);
}

// Least significant set bit index of (t+1). For t+1=1, lssb=0; for 2, lssb=1;
// for 3, lssb=0; for 4, lssb=2; for 5..7, lssb=0; for 8, lssb=3. This is the
// classical bit-twiddling trick: lssb(x) = __builtin_ctz(x) for non-zero x.
static inline size_t lssb(size_t x) {
    size_t k = 0;
    while ((x & 1ULL) == 0) { x >>= 1; ++k; }
    return k;
}

// ---------- constructor ----------

LogLinearAttention::LogLinearAttention(size_t d_model, size_t n_heads, size_t d_state,
                                       size_t d_inner, size_t max_levels)
    : in_proj(d_model, 0),
      out_proj(0, d_model),
      a_proj(d_model, 0),
      b_proj(d_model, 0),
      k_proj(d_model, 0),
      q_proj(d_model, 0),
      lambda_proj(d_model, 0),
      D_skip(1, 0),
      dt_bias(1, 0),
      d_model_(d_model), n_heads_(n_heads), d_state_(d_state)
{
    if (d_model == 0 || n_heads == 0 || d_state == 0) {
        throw std::invalid_argument("LogLinearAttention: d_model, n_heads, d_state must be > 0");
    }
    if (d_inner == 0) d_inner = 2 * d_model;
    if (d_inner % n_heads != 0) {
        throw std::invalid_argument("LogLinearAttention: d_inner must be divisible by n_heads");
    }
    size_t head_dim = d_inner / n_heads;
    if (head_dim != d_state) {
        throw std::invalid_argument(
            "LogLinearAttention: head_dim (= d_inner / n_heads) must equal d_state "
            "(so that per-head state S is head_dim × head_dim = head_dim^2 / d_inner, "
            "and d_inner = n_heads * head_dim = n_heads * d_state gives a square state)");
    }

    d_inner_  = d_inner;
    head_dim_ = head_dim;
    L_        = max_levels;  // set later in forward if 0

    // Reinitialize projections.
    in_proj      = Dense(d_model, 2 * d_inner);
    out_proj     = Dense(d_inner, d_model);
    a_proj       = Dense(d_model, n_heads);
    b_proj       = Dense(d_model, d_inner);
    k_proj       = Dense(d_model, d_inner);
    q_proj       = Dense(d_model, d_inner);
    lambda_proj  = Dense(d_model, n_heads * max_levels);  // placeholder — overridden in forward

    // dt_bias: logit(0.9) ≈ 2.197 — "soft open" decay
    dt_bias = Tensor(1, n_heads);
    for (size_t h = 0; h < n_heads; ++h) {
        dt_bias(0, h) = std::log(0.9 / (1.0 - 0.9));
    }

    // D_skip: init to 1.0
    D_skip = Tensor(1, d_inner);
    for (size_t i = 0; i < d_inner; ++i) D_skip(0, i) = 1.0;

    grad_D_skip_  = Tensor(1, d_inner);
    grad_dt_bias_ = Tensor(1, n_heads);

    // Initialize cache Tensors to small non-empty sizes so that any pointer
    // arithmetic doesn't read uninitialized memory. They will be resized in
    // forward() with the correct shape.
    last_input_  = Tensor(1, 1);
    last_p_      = Tensor(1, 1);
    last_x_ssm_  = Tensor(1, 1);
    last_gate_   = Tensor(1, 1);
    last_a_pre_  = Tensor(1, 1);
    last_a_      = Tensor(1, 1);
    last_b_      = Tensor(1, 1);
    last_k_      = Tensor(1, 1);
    last_q_      = Tensor(1, 1);
    last_lambda_ = Tensor(1, 1);
    last_S_      = Tensor(1, 1);
    last_o_      = Tensor(1, 1);
    last_gated_  = Tensor(1, 1);

    // Zero biases so projections start pure linear.
    in_proj.bias.fill(0.0);
    out_proj.bias.fill(0.0);
    a_proj.bias.fill(0.0);
    b_proj.bias.fill(0.0);
    k_proj.bias.fill(0.0);
    q_proj.bias.fill(0.0);
    lambda_proj.bias.fill(0.0);  // placeholder; will resize lambda_proj below
}

// ---------- forward ----------

Tensor LogLinearAttention::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("LogLinearAttention: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("LogLinearAttention: input must have at least one token");
    }
    last_input_ = input.clone();
    last_T_     = T;

    // Determine L (number of Fenwick levels). max_levels=0 means auto: ceil(log2(T)) + 1.
    size_t L;
    if (L_ > 0) {
        L = L_;
    } else {
        // ceil(log2(T)) + 1 — accommodates the sentinel + Fenwick levels
        size_t e = 0, v = 1;
        while (v < T) { v <<= 1; ++e; }
        L = e + 1;
    }
    if (L_ > 0 && L_ != L) {
        // User specified a max_levels. Use that.
        L = L_;
    } else if (L_ == 0) {
        L_ = L;  // cache for backward
    }

    // Resize lambda_proj if first call: output dim is n_heads * L.
    // Use a one-shot resize: build a brand new Dense in place via the
    // assignment operator. (This matches Mamba2Block's pattern.)
    if (lambda_proj.weights.rows != n_heads_ * L) {
        lambda_proj = Dense(d_model_, n_heads_ * L);
        lambda_proj.bias.fill(0.0);
    }

    // Step 1: in_proj
    last_p_ = in_proj.forward(input);

    last_x_ssm_ = Tensor(T, d_inner_);
    last_gate_  = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            last_x_ssm_(t, i) = last_p_(t, i);
            last_gate_(t, i)  = last_p_(t, d_inner_ + i);
        }
    }

    // Step 2: per-head decay and V/K/Q projections
    Tensor a_pre = a_proj.forward(input);
    last_a_pre_ = Tensor(T, n_heads_);
    last_a_     = Tensor(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double pre = a_pre(t, h) + dt_bias(0, h);
            last_a_pre_(t, h) = pre;
            last_a_(t, h)     = lla_sigmoid(pre);
        }
    }

    last_b_ = b_proj.forward(input);
    last_k_ = k_proj.forward(input);
    last_q_ = q_proj.forward(input);

    // λ_proj output: (T, n_heads * L). Reshape to per-(token, head, level).
    last_lambda_ = lambda_proj.forward(input);

    // Step 3: Fenwick-tree SSD recurrence.
    // Cache S at every (t, ℓ) as a flat tensor (T * L, d_inner).
    // For head h: state slice at row (t * L + ℓ), cols [h*head_dim, (h+1)*head_dim).
    // Within head h: dh-th row, ds-th col of the (head_dim × head_dim) state is
    // stored at flat col (h * head_dim + dh * head_dim + ds) = (h * head_dim + dh * head_dim + ds).
    last_S_    = Tensor(T * L, d_inner_ * head_dim_);
    last_o_    = Tensor(T, d_inner_);
    last_case_ = std::vector<uint8_t>(T * L, 0);

    for (size_t t = 0; t < T; ++t) {
        // Determine the Fenwick-tree case at this step.
        // lssb_t = lssb(t+1) — the level at which the bucket gets promoted/cleared.
        size_t lssb_t = lssb(t + 1);

        for (size_t h = 0; h < n_heads_; ++h) {
            double a_t_h = last_a_(t, h);

            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                double b_t_i = last_b_(t, i);

                for (size_t ds = 0; ds < head_dim_; ++ds) {
                    size_t col = h * head_dim_ * head_dim_ + li * head_dim_ + ds;
                    double k_t_ds = last_k_(t, h * head_dim_ + ds);

                    // Default: every level writes its S^(ℓ)_t at row (t * L + ℓ), col `col`.
                    // Process level-by-level. Need to be careful about merge-promoted since
                    // it reads from previous-step states (t-1, ℓ' < ℓ).

                    // We iterate ℓ from 0..L-1, computing each level.
                    for (size_t ell = 0; ell < L; ++ell) {
                        size_t row_curr = t * L + ell;
                        double s_val = 0.0;
                        uint8_t case_id = 0;

                        if (ell == 0) {
                            // Case: immediate. S^(0)_t = b_t ⊗ k_t
                            s_val = b_t_i * k_t_ds;
                            case_id = 0;
                        } else if (t == 0 || ell <= lssb_t) {
                            // Cleared (t=0: no previous step; 0 < ell <= lssb_t: bucket was promoted up)
                            s_val = 0.0;
                            case_id = 1;
                        } else if (ell == lssb_t + 1 && ell < L) {
                            // Merge-promoted: S^(ℓ)_t = a_t · Σ_{ℓ'<ℓ} S^(ℓ')_{t-1}
                            double acc = 0.0;
                            for (size_t ell_p = 0; ell_p < ell; ++ell_p) {
                                size_t row_prev = (t - 1) * L + ell_p;
                                acc += last_S_(row_prev, col);
                            }
                            s_val = a_t_h * acc;
                            case_id = 2;
                        } else {
                            // Carry-forward: S^(ℓ)_t = a_t · S^(ℓ)_{t-1}
                            size_t row_prev = (t - 1) * L + ell;
                            s_val = a_t_h * last_S_(row_prev, col);
                            case_id = 3;
                        }

                        last_S_(row_curr, col) = s_val;
                        last_case_[row_curr]   = case_id;
                    }
                }
            }
        }

        // Compute o_t[h, dh, ds] = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · q_t[h, dh, ds] · S^(ℓ)_t[h, dh, ds]
        // For our flattened layout: per head h, for each (li, ds), the contribution is
        //   sum_ell λ^(ell)_t[h] * q_t_segment[h, ds] * S^(ell)_t[h, li, ds]
        // where λ^(ell)_t[h] is at last_lambda_(t, h * L + ell).
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                double acc = 0.0;
                for (size_t ds = 0; ds < head_dim_; ++ds) {
                    size_t col = h * head_dim_ * head_dim_ + li * head_dim_ + ds;
                    double q_t_ds = last_q_(t, h * head_dim_ + ds);
                    for (size_t ell = 0; ell < L; ++ell) {
                        size_t row = t * L + ell;
                        double lam = last_lambda_(t, h * L + ell);
                        acc += lam * q_t_ds * last_S_(row, col);
                    }
                }
                last_o_(t, i) = acc;
            }
        }
    }

    // Step 4: skip + gating + output projection
    Tensor gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double y = last_o_(t, i) + D_skip(0, i) * last_x_ssm_(t, i);
            double g_act = lla_silu(last_gate_(t, i));
            gated(t, i) = g_act * y;
        }
    }
    last_gated_ = gated;
    Tensor output = out_proj.forward(gated);
    return output;
}

// ---------- backward ----------
//
// Per-timestep gradient flow:
//   dL/d(o_t[i])    = grad_gated[t][i] * silu(g_t[i])
//   dL/d(D_skip[i]) = sum_t grad_gated[t][i] * silu(g_t[i]) * x_ssm_t[i]
//   dL/d(x_ssm_t[i])= grad_gated[t][i] * silu(g_t[i]) * D_skip[i]
//   dL/d(g_t[i])    = grad_gated[t][i] * silu'(g_t[i]) * (o_t[i] + D_skip[i]*x_ssm_t[i])
//
// Then the SSD output gradient:
//   dL/d(q_t[h, dh, ds])   += Σ_{ℓ} λ^(ℓ)_t[h] · S^(ℓ)_t[h, dh, ds] · dL/d(o_t[h*head_dim + dh])
//   dL/d(λ^(ℓ)_t[h])      += Σ_{dh, ds} q_t[h, dh, ds] · S^(ℓ)_t[h, dh, ds] · dL/d(o_t[h*head_dim + dh])
//   dL/d(S^(ℓ)_t[h, dh, ds]) = λ^(ℓ)_t[h] · q_t[h, dh, ds] · dL/d(o_t[h*head_dim + dh])
//
// The state gradient dS propagates backward through the Fenwick-tree case at each (t, ℓ).

Tensor LogLinearAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = last_input_.rows;
    size_t L = L_;
    if (grad_output.rows != T || grad_output.cols != d_model_) {
        throw std::invalid_argument("LogLinearAttention: grad_output shape mismatch");
    }

    in_proj.zero_grad();
    out_proj.zero_grad();
    a_proj.zero_grad();
    b_proj.zero_grad();
    k_proj.zero_grad();
    q_proj.zero_grad();
    lambda_proj.zero_grad();

    // ------ (1) out_proj backward ------
    Tensor grad_gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_output(t, j) * out_proj.weights(j, i);
            grad_gated(t, i) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < T; ++t) acc += grad_output(t, j) * last_gated_(t, i);
            out_proj.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < T; ++t) b_acc += grad_output(t, j);
        out_proj.grad_bias(0, j) += b_acc;
    }

    // ------ (2) split into grad_gate, grad_o, grad_x_ssm, grad_D_skip ------
    Tensor grad_gate(T, d_inner_);
    Tensor grad_o(T, d_inner_);
    Tensor grad_x_ssm(T, d_inner_);
    Tensor grad_D_skip_acc(1, d_inner_);
    grad_D_skip_acc.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double g  = last_gate_(t, i);
            double y  = last_o_(t, i) + D_skip(0, i) * last_x_ssm_(t, i);
            double sg = lla_sigmoid(g);
            double silu_prime = sg * (1.0 + g * (1.0 - sg));
            grad_gate(t, i) = grad_gated(t, i) * silu_prime * y;
            double d_y = grad_gated(t, i) * lla_silu(g);
            grad_o(t, i)    = d_y;
            grad_x_ssm(t, i)= d_y * D_skip(0, i);
            grad_D_skip_acc(0, i) += d_y * last_x_ssm_(t, i);
        }
    }

    // ------ (3) SSD output gradient: grad_o → grad_q_t, grad_lambda, grad_S^(ℓ)_t ------
    Tensor grad_q_t(T, d_inner_);     grad_q_t.fill(0.0);
    Tensor grad_lambda(T, n_heads_ * L);  grad_lambda.fill(0.0);
    // g_S is the gradient w.r.t. the cached last_S_, shape (T * L, d_inner_).
    Tensor g_S(T * L, d_inner_ * head_dim_);
    g_S.fill(0.0);

    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                double grad_o_i = grad_o(t, i);
                for (size_t ds = 0; ds < head_dim_; ++ds) {
                    size_t col = h * head_dim_ * head_dim_ + li * head_dim_ + ds;
                    double q_t_ds = last_q_(t, h * head_dim_ + ds);
                    // Contribution from each level
                    for (size_t ell = 0; ell < L; ++ell) {
                        size_t row = t * L + ell;
                        double lam = last_lambda_(t, h * L + ell);
                        // dL/d(q_t_segment[h, ds]) += lam * S^(ℓ)_t[h, li, ds] * grad_o_i
                        grad_q_t(t, h * head_dim_ + ds) += lam * last_S_(row, col) * grad_o_i;
                        // dL/d(λ^(ℓ)_t[h]) += q_t_ds * S^(ℓ)_t[h, li, ds] * grad_o_i
                        grad_lambda(t, h * L + ell) += q_t_ds * last_S_(row, col) * grad_o_i;
                        // dL/d(S^(ℓ)_t[h, li, ds]) += lam * q_t_ds * grad_o_i
                        g_S(row, col) += lam * q_t_ds * grad_o_i;
                    }
                }
            }
        }
    }

    // ------ (4) BPTT through Fenwick-tree state recurrence ------
    //
    // For each (t, ℓ), we have last_case_(t*L + ell) telling us which case fired.
    // We walk t from T-1 down to 0 and route g_S[row_curr] back to g_S[row_prev]
    // for each case.
    //
    // Cases:
    //   0 (immediate): S^(0)_t = b_t ⊗ k_t. NO backward to row_prev (S is built
    //                  fresh from inputs at this step). Backward goes to b_t, k_t.
    //   1 (cleared):   S^(ℓ)_t = 0. NO backward; g_S[row_curr] is overwritten
    //                  to 0 effectively (the state was zero, so its gradient
    //                  contribution should propagate as 0).
    //   2 (merge-promoted at ℓ == lssb+1): S^(ℓ)_t = a_t · Σ_{ℓ'<ℓ} S^(ℓ')_{t-1}.
    //                  Backward: dL/d(S^(ℓ')_{t-1}) += g_S * a_t for all ℓ' < ℓ.
    //                  Also: dL/d(a_t) += g_S · Σ_{ℓ'<ℓ} S^(ℓ')_{t-1} (per head).
    //   3 (carry):    S^(ℓ)_t = a_t · S^(ℓ)_{t-1}. Backward: dL/d(S^(ℓ)_{t-1}) += g_S · a_t.
    //                  Also: dL/d(a_t) += g_S · S^(ℓ)_{t-1} (per head).
    //
    // After computing grad_a (per-timestep per-head), we chain through sigmoid.
    //
    // For cases 0 (immediate) and 1 (cleared), the b_t, k_t gradients come from
    // the direct contribution g_S(0, ·) at row (t*L+0). Case 0:
    //   dL/d(b_t[h, li])        += Σ_{ds} g_S(t*L+0, col) · k_t[h, ds]
    //   dL/d(k_t[h, ds])        += Σ_{li} g_S(t*L+0, col) · b_t[h, li]
    // Case 1: S is zero, g_S contribution from case 1 is zero (already 0).
    //
    Tensor grad_a(T, n_heads_);  grad_a.fill(0.0);
    Tensor grad_b(T, d_inner_);  grad_b.fill(0.0);
    Tensor grad_k(T, d_inner_);  grad_k.fill(0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        double a_t_h_arr[16];  // small stack buffer; n_heads typically small
        for (size_t h = 0; h < n_heads_; ++h) a_t_h_arr[h] = last_a_(t, h);

        // For each (h, li, ds), walk all L levels and route per-case.
        for (size_t h = 0; h < n_heads_; ++h) {
            double a_t_h = a_t_h_arr[h];
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                for (size_t ds = 0; ds < head_dim_; ++ds) {
                    size_t col = h * head_dim_ * head_dim_ + li * head_dim_ + ds;

                    // Case 0 (immediate, ℓ=0): contributes to b_t, k_t.
                    {
                        size_t row = t * L + 0;
                        double gh = g_S(row, col);
                        double b_t_i = last_b_(t, i);
                        double k_t_ds = last_k_(t, h * head_dim_ + ds);
                        // dL/d(b_t[i]) += gh * k_t_ds
                        grad_b(t, i) += gh * k_t_ds;
                        // dL/d(k_t_segment[h, ds]) += gh * b_t_i
                        grad_k(t, h * head_dim_ + ds) += gh * b_t_i;
                    }

                    // Cases 1/2/3 (ℓ >= 1): route backward through the recurrence.
                    for (size_t ell = 1; ell < L; ++ell) {
                        size_t row_curr = t * L + ell;
                        uint8_t case_id = last_case_[row_curr];
                        double gh = g_S(row_curr, col);

                        if (case_id == 1) {
                            // Cleared: S was zero at this step. Drop gradient here.
                            continue;
                        } else if (case_id == 2) {
                            // Merge-promoted: S^(ℓ)_t = a_t · Σ_{ℓ'<ℓ} S^(ℓ')_{t-1}.
                            // (Only valid for t > 0; at t=0 case_id is always 1 for ℓ > 0.)
                            // Backward:
                            //   dL/d(S^(ℓ')_{t-1}) += g_S(row_curr) · a_t for ℓ' < ℓ
                            //   dL/d(a_t)          += g_S(row_curr) · Σ_{ℓ'<ℓ} S^(ℓ')_{t-1}
                            double sum_prev = 0.0;
                            for (size_t ell_p = 0; ell_p < ell; ++ell_p) {
                                size_t row_prev = (t - 1) * L + ell_p;
                                double s_prev = last_S_(row_prev, col);
                                g_S(row_prev, col) += gh * a_t_h;
                                sum_prev += s_prev;
                            }
                            grad_a(t, h) += gh * sum_prev;
                        } else if (case_id == 3) {
                            // Carry: S^(ℓ)_t = a_t · S^(ℓ)_{t-1}.
                            // Backward:
                            //   dL/d(S^(ℓ)_{t-1}) += g_S(row_curr) · a_t
                            //   dL/d(a_t)          += g_S(row_curr) · S^(ℓ)_{t-1}
                            size_t row_prev = (t - 1) * L + ell;
                            double s_prev = last_S_(row_prev, col);
                            g_S(row_prev, col) += gh * a_t_h;
                            grad_a(t, h) += gh * s_prev;
                        }
                    }
                }
            }
        }
    }

    // ------ (5) chain rule: a_t = sigmoid(a_pre_t + dt_bias[h]) ------
    Tensor grad_a_pre(T, n_heads_);
    Tensor grad_dt_bias_acc(1, n_heads_);
    grad_dt_bias_acc.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double a_t_h = last_a_(t, h);
            double sig_prime = a_t_h * (1.0 - a_t_h);
            double d = grad_a(t, h) * sig_prime;
            grad_a_pre(t, h) = d;
            grad_dt_bias_acc(0, h) += d;
        }
    }

    // ------ (6) backprop through projections ------
    // q_proj: (T, d_inner) from (T, d_model). grad_y = grad_q_t.
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_q_t;
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                q_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            q_proj.grad_bias(0, i) += b_acc;
        }
    }

    // k_proj
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_k;
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                k_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            k_proj.grad_bias(0, i) += b_acc;
        }
    }

    // b_proj
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_b;
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                b_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            b_proj.grad_bias(0, i) += b_acc;
        }
    }

    // a_proj
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_a_pre;
        for (size_t i = 0; i < n_heads_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                a_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            a_proj.grad_bias(0, i) += b_acc;
        }
    }

    // lambda_proj: (T, n_heads * L) from (T, d_model). grad_y = grad_lambda.
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_lambda;
        size_t out_dim = n_heads_ * L;
        for (size_t i = 0; i < out_dim; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                lambda_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            lambda_proj.grad_bias(0, i) += b_acc;
        }
    }

    // in_proj
    Tensor grad_p(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            grad_p(t, i)               = grad_x_ssm(t, i);
            grad_p(t, d_inner_ + i)    = grad_gate(t, i);
        }
    }
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_p;
        size_t out_dim = 2 * d_inner_;
        for (size_t i = 0; i < out_dim; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                in_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            in_proj.grad_bias(0, i) += b_acc;
        }
    }

    // ------ (7) input gradient: sum contributions from all 6 projections ------
    Tensor grad_input(T, d_model_);
    grad_input.fill(0.0);

    // From in_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < 2 * d_inner_; ++i) acc += grad_p(t, i) * in_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }
    // From a_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < n_heads_; ++i) acc += grad_a_pre(t, i) * a_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }
    // From b_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) acc += grad_b(t, i) * b_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }
    // From k_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) acc += grad_k(t, i) * k_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }
    // From q_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) acc += grad_q_t(t, i) * q_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }
    // From lambda_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < n_heads_ * L; ++i) acc += grad_lambda(t, i) * lambda_proj.weights(i, k);
            grad_input(t, k) += acc;
        }
    }

    // ------ (8) stash non-Dense grads ------
    if (grad_D_skip_.rows != 1 || grad_D_skip_.cols != d_inner_) {
        grad_D_skip_  = Tensor(1, d_inner_);
        grad_dt_bias_ = Tensor(1, n_heads_);
        grad_D_skip_.fill(0.0);
        grad_dt_bias_.fill(0.0);
    }
    for (size_t i = 0; i < d_inner_; ++i) grad_D_skip_(0, i) = grad_D_skip_acc(0, i);
    for (size_t h = 0; h < n_heads_; ++h) grad_dt_bias_(0, h) = grad_dt_bias_acc(0, h);

    return grad_input;
}

// ---------- update_weights ----------

void LogLinearAttention::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    a_proj.update_weights(learning_rate);
    b_proj.update_weights(learning_rate);
    k_proj.update_weights(learning_rate);
    q_proj.update_weights(learning_rate);
    lambda_proj.update_weights(learning_rate);
    if (grad_D_skip_.cols == d_inner_) {
        for (size_t i = 0; i < d_inner_; ++i) D_skip(0, i) -= learning_rate * grad_D_skip_(0, i);
    }
    if (grad_dt_bias_.cols == n_heads_) {
        for (size_t h = 0; h < n_heads_; ++h) dt_bias(0, h) -= learning_rate * grad_dt_bias_(0, h);
    }
}

// ---------- zero_grad ----------

void LogLinearAttention::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    a_proj.zero_grad();
    b_proj.zero_grad();
    k_proj.zero_grad();
    q_proj.zero_grad();
    lambda_proj.zero_grad();
    grad_D_skip_  = Tensor(1, d_inner_);
    grad_dt_bias_ = Tensor(1, n_heads_);
}

// ---------- parameters / gradients ----------

std::vector<Tensor*> LogLinearAttention::parameters() {
    std::vector<Tensor*> p;
    auto append_dense = [&](Dense& d) {
        p.push_back(&d.weights);
        p.push_back(&d.bias);
    };
    append_dense(in_proj);
    append_dense(out_proj);
    append_dense(a_proj);
    append_dense(b_proj);
    append_dense(k_proj);
    append_dense(q_proj);
    append_dense(lambda_proj);
    p.push_back(&D_skip);
    p.push_back(&dt_bias);
    return p;
}

std::vector<Tensor*> LogLinearAttention::gradients() {
    std::vector<Tensor*> g;
    auto append_dense = [&](Dense& d) {
        g.push_back(&d.grad_weights);
        g.push_back(&d.grad_bias);
    };
    append_dense(in_proj);
    append_dense(out_proj);
    append_dense(a_proj);
    append_dense(b_proj);
    append_dense(k_proj);
    append_dense(q_proj);
    append_dense(lambda_proj);
    g.push_back(&grad_D_skip_);
    g.push_back(&grad_dt_bias_);
    return g;
}

// ============================================================================
// LogLinearAttentionModel
// ============================================================================

LogLinearAttentionModel::LogLinearAttentionModel(size_t input_dim, size_t d_model,
                                                 size_t output_dim, size_t num_layers,
                                                 size_t n_heads, size_t d_state,
                                                 size_t d_inner, size_t max_levels)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_layers_(num_layers), n_heads_(n_heads), d_state_(d_state),
      d_inner_(d_inner), L_(max_levels),
      input_proj_(input_dim, d_model),
      classifier_(d_model, output_dim)
{
    if (input_dim == 0 || d_model == 0 || output_dim == 0 || num_layers == 0) {
        throw std::invalid_argument("LogLinearAttentionModel: dims and num_layers must be > 0");
    }
    if (n_heads == 0 || d_state == 0) {
        throw std::invalid_argument("LogLinearAttentionModel: n_heads and d_state must be > 0");
    }
    if (d_inner > 0 && d_inner % n_heads != 0) {
        throw std::invalid_argument("LogLinearAttentionModel: d_inner must be divisible by n_heads");
    }
    if (d_inner > 0 && (d_inner / n_heads) != d_state) {
        throw std::invalid_argument("LogLinearAttentionModel: head_dim (= d_inner / n_heads) must equal d_state");
    }
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.push_back(std::make_unique<LogLinearAttention>(
            d_model, n_heads, d_state, d_inner, max_levels));
    }
    classifier_ = Dense(d_model, output_dim);
    classifier_.bias.fill(0.0);
}

Tensor LogLinearAttentionModel::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != input_dim_) {
        throw std::invalid_argument("LogLinearAttentionModel: input cols mismatch");
    }
    last_input_ = input.clone();
    Tensor h = input_proj_.forward(input);  // (T, d_model)
    block_outputs_.clear();
    block_outputs_.push_back(h);
    for (size_t i = 0; i < num_layers_; ++i) {
        h = blocks_[i]->forward(h);
        block_outputs_.push_back(h);
    }
    // Use last-token pooling for classification
    Tensor h_last(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) h_last(0, j) = h(T - 1, j);
    return classifier_.forward(h_last);
}

Tensor LogLinearAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = last_input_.rows;
    // classifier backward
    Tensor grad_h_last(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double acc = 0.0;
        for (size_t k = 0; k < output_dim_; ++k) acc += grad_output(0, k) * classifier_.weights(k, j);
        grad_h_last(0, j) = acc;
    }
    for (size_t k = 0; k < output_dim_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = grad_output(0, k) * block_outputs_.back()(T - 1, j);
            classifier_.grad_weights(k, j) += acc;
        }
        double b_acc = grad_output(0, k);
        classifier_.grad_bias(0, k) += b_acc;
    }

    // Broadcast grad_h_last into the last time step of grad_h
    Tensor grad_h(T, d_model_);
    grad_h.fill(0.0);
    for (size_t j = 0; j < d_model_; ++j) grad_h(T - 1, j) = grad_h_last(0, j);

    // Backprop through blocks
    for (int i = (int)num_layers_ - 1; i >= 0; --i) {
        grad_h = blocks_[i]->backward(grad_h, learning_rate);
        if (i > 0) {
            // grad_h is for block i output; need to backprop through skip identity
            // (no extra op — block_outputs_[i] was the input to block i)
        }
    }

    // backprop through input_proj
    Tensor grad_p(T, d_model_);
    grad_p.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < input_dim_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_h(t, j) * input_proj_.weights(j, k);
            grad_p(t, k) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < input_dim_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < T; ++t) acc += grad_h(t, j) * last_input_(t, k);
            input_proj_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < T; ++t) b_acc += grad_h(t, j);
        input_proj_.grad_bias(0, j) += b_acc;
    }
    return grad_p;
}

void LogLinearAttentionModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void LogLinearAttentionModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> LogLinearAttentionModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&input_proj_.weights);
    p.push_back(&input_proj_.bias);
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&classifier_.weights);
    p.push_back(&classifier_.bias);
    return p;
}

std::vector<Tensor*> LogLinearAttentionModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&input_proj_.grad_weights);
    g.push_back(&input_proj_.grad_bias);
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&classifier_.grad_weights);
    g.push_back(&classifier_.grad_bias);
    return g;
}