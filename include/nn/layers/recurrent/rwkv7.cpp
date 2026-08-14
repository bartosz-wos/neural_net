#include "rwkv7.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// RWKV7TimeMix implementation
// ------------------------------------------------------------------------
//
// Forward per-step (per head h, head_dim m = d/num_heads):
//   Token shift: x_t^□ = lerp(x_t, x_{t-1}, μ_□)
//   r_t = W_r · x_t^r + b_r
//   k_t = W_k · x_t^k + b_k
//   v_t = W_v · x_t^v + b_v
//   d_pre_t = W_d · x_t^d + b_d
//   a_pre_t = W_a · x_t^a + b_a
//   d_t = tanh(d_pre_t)
//   w_t = exp(-exp(-0.5) · sigmoid(d_t))                  ∈ (0.687, 1)
//   a_t = sigmoid(a_pre_t)                                ∈ (0, 1)
//   κ_t = k_t ⊙ ξ                                          (removal key)
//   κ̂_t[h] = κ_t[h] / ||κ_t[h]||_2                       (per-head L2 norm)
//   k̃_t = k_t ⊙ lerp(1, a_t, α) = k_t ⊙ (1 + α·(a_t - 1))
//
//   G_t = diag(w_t[h]) - κ̂_t[h]^T · (a_t[h] ⊙ κ̂_t[h])    (m × m)
//   wkv_t[h] = wkv_{t-1}[h] · G_t + v_t[h]^T · k̃_t[h]
//   wkv_0 = 0
//
//   o_t[h, j] = sum_i wkv_t[h, i, j] · r_t[h, i]
//
// Backward (single-step BPTT, t = T-1 down to 0):
//   g_o_t = grad_output[t]  (direct upstream gradient)
//   grad_wkv_next carries dL/dwkv_{t+1} across the BPTT boundary.
//
//   For each head h:
//     g_wkv_direct[h, i, j] = r_t[h, i] · g_o_t[h, j]
//     grad_wkv_curr[h, i, j] = g_wkv_direct[h, i, j]
//                             + sum_{j'} G_{t+1}[j, j'] · grad_wkv_next[h, i, j']
//                             (if t < T-1; otherwise just the direct part)
//
//     dL/dG_t[h, i, j] = sum_{i'} wkv_{t-1}[h, i', i] · grad_wkv_curr[h, i', j]
//
//     dL/dw_t[h, i]    = dL/dG_t[h, i, i]
//       → dL/dd_pre_t[h, i] = dL/dw_t[h, i] · (-exp(-0.5)) · w_t · sig · (1-sig)
//                              where sig = sigmoid(d_pre_t[h, i])
//
//     dL/dκ̂_t[h, i, j] = -a_t[h, i] · dL/dG_t[h, i, j]
//     dL/da_t[h, i] += -sum_j κ̂_t[h, i, j] · dL/dG_t[h, i, j]
//
//     dL/dκ_t[h, i] = (1/||κ_t[h]||) · [dL/dκ̂_t[h, i] - κ̂_t[h, i] · inner]
//                       where inner = sum_k dL/dκ̂_t[h, k] · κ̂_t[h, k]
//                                 (projection-back-to-tangent-space term)
//     dL/dξ[h, i] += k_t[h, i] · dL/dκ_t[h, i]
//     dL/dk_t[h, i] += ξ[h, i] · dL/dκ_t[h, i]
//
//     dL/dk̃_t[h, j] = sum_i v_t[h, i] · grad_wkv_curr[h, i, j]
//     dL/dv_t[h, i] = sum_j k̃_t[h, j] · grad_wkv_curr[h, i, j]
//
//     dL/dk_t[h, i] += (1 + α·(a_t[h, i] - 1)) · dL/dk̃_t[h, i]
//     dL/da_t[h, i] += α · k_t[h, i] · dL/dk̃_t[h, i]
//     dL/dα += sum_i k_t[h, i] · (a_t[h, i] - 1) · dL/dk̃_t[h, i]
//
//     dL/dr_t[h, i] = sum_j wkv_t[h, i, j] · g_o_t[h, j]
//                     (only the DIRECT gradient, not the carrier)
//
//   Token shift backward at the very end:
//     dL/dx_t = sum_□ (dL/dx_t^□ ⊙ μ_□) + carry-forward from x_{t+1}
//     dL/dμ_□ += dL/dx_t^□ ⊙ (x_t - x_{t-1})
// ============================================================================

static inline double rwkv7_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
// Private validating helper: throws if d/num_heads is invalid.
// This lets us validate BEFORE entering the initializer list (which would
// otherwise compute head_dim_ = d/num_heads, potentially SIGFPE on num_heads=0).
RWKV7TimeMix::RWKV7TimeMix(size_t d, size_t num_heads, bool /*validate_tag*/)
    : d_(d), num_heads_(num_heads), head_dim_(d / num_heads),
      W_r(d, d), W_k(d, d), W_v(d, d), W_d(d, d), W_a(d, d),
      xi(1, d), alpha(1, 1),
      mu_r(1, d), mu_k(1, d), mu_v(1, d), mu_d(1, d), mu_a(1, d),
      grad_xi_(1, d), grad_alpha_(1, 1),
      grad_mu_r_(1, d), grad_mu_k_(1, d), grad_mu_v_(1, d),
      grad_mu_d_(1, d), grad_mu_a_(1, d)
{
    // xi: 1.0 init → κ_t = k_t (neutral starting point per RWKV-7 paper).
    xi.fill(1.0);
    // alpha: 0.0 init → k̃_t = k_t (matches DeltaNet-like initial behavior).
    alpha.fill(0.0);
    // mu_□: 0.5 init → 50/50 mix of current and previous token.
    for (size_t j = 0; j < d_; ++j) {
        mu_r(0, j) = 0.5;
        mu_k(0, j) = 0.5;
        mu_v(0, j) = 0.5;
        mu_d(0, j) = 0.5;
        mu_a(0, j) = 0.5;
    }

    // Zero all gradient buffers.
    grad_xi_.fill(0.0);
    grad_alpha_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
    grad_mu_d_.fill(0.0);
    grad_mu_a_.fill(0.0);
}

RWKV7TimeMix::RWKV7TimeMix(size_t d, size_t num_heads)
    : RWKV7TimeMix(d, num_heads,
                   [&]() {
                       if (d == 0 || num_heads == 0 || d % num_heads != 0) {
                           throw std::invalid_argument(
                               "RWKV7TimeMix: d must be > 0 and divisible by num_heads");
                       }
                       return true;
                   }() ? false : true)  // dummy, never reached because of throw above
{
    // Delegate to validating constructor. The lambda above is evaluated BEFORE
    // the member init list runs, allowing us to throw on bad args first.
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RWKV7TimeMix::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("RWKV7TimeMix: input.cols must equal d");
    }
    size_t T = input.rows;
    size_t m = head_dim_;
    last_input_ = input.clone();

    // Allocate cache
    last_x_shift_ = Tensor(T, d_); last_x_shift_.fill(0.0);
    last_r_in_ = Tensor(T, d_); last_k_in_ = Tensor(T, d_); last_v_in_ = Tensor(T, d_);
    last_d_in_ = Tensor(T, d_); last_a_in_ = Tensor(T, d_);
    last_r_ = Tensor(T, d_); last_k_ = Tensor(T, d_); last_v_ = Tensor(T, d_);
    last_d_pre_ = Tensor(T, d_); last_a_pre_ = Tensor(T, d_);
    last_d_ = Tensor(T, d_); last_w_ = Tensor(T, d_); last_a_ = Tensor(T, d_);
    last_kappa_ = Tensor(T, d_); last_kappa_hat_ = Tensor(T, d_);
    last_kappa_norm_ = Tensor(T, num_heads_);
    last_k_tilde_ = Tensor(T, d_);
    last_wkv_ = Tensor(T + 1, num_heads_ * m * m);
    last_wkv_.fill(0.0);

    // Step 1: token shift
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double xt = last_input_(t, j);
            double xsh = (t == 0) ? 0.0 : last_input_(t - 1, j);
            last_x_shift_(t, j) = xsh;
            last_r_in_(t, j) = mu_r(0, j) * xt + (1.0 - mu_r(0, j)) * xsh;
            last_k_in_(t, j) = mu_k(0, j) * xt + (1.0 - mu_k(0, j)) * xsh;
            last_v_in_(t, j) = mu_v(0, j) * xt + (1.0 - mu_v(0, j)) * xsh;
            last_d_in_(t, j) = mu_d(0, j) * xt + (1.0 - mu_d(0, j)) * xsh;
            last_a_in_(t, j) = mu_a(0, j) * xt + (1.0 - mu_a(0, j)) * xsh;
        }
    }

    // Step 2: projections
    last_r_ = W_r.forward(last_r_in_);
    last_k_ = W_k.forward(last_k_in_);
    last_v_ = W_v.forward(last_v_in_);
    last_d_pre_ = W_d.forward(last_d_in_);
    last_a_pre_ = W_a.forward(last_a_in_);

    // Step 3: derive w_t, a_t, κ_t, κ̂_t, k̃_t (per token, per head)
    double a_alpha = alpha(0, 0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            // w_t, a_t
            for (size_t i = 0; i < m; ++i) {
                double d_pre = last_d_pre_(t, base + i);
                double d_th = std::tanh(d_pre);
                last_d_(t, base + i) = d_th;
                double s = rwkv7_sigmoid(d_th);
                last_w_(t, base + i) = std::exp(-std::exp(-0.5) * s);
                last_a_(t, base + i) = rwkv7_sigmoid(last_a_pre_(t, base + i));
            }
            // κ_t = k_t ⊙ ξ
            for (size_t i = 0; i < m; ++i) {
                last_kappa_(t, base + i) = last_k_(t, base + i) * xi(0, base + i);
            }
            // ||κ_t[h]||_2 (per head)
            double norm = 0.0;
            for (size_t i = 0; i < m; ++i) {
                double k = last_kappa_(t, base + i);
                norm += k * k;
            }
            norm = std::sqrt(norm);
            if (norm < 1e-12) norm = 1e-12;  // safety
            last_kappa_norm_(t, h) = norm;
            // κ̂_t = κ_t / norm
            for (size_t i = 0; i < m; ++i) {
                last_kappa_hat_(t, base + i) = last_kappa_(t, base + i) / norm;
            }
            // k̃_t = k_t ⊙ (1 + α·(a_t - 1))
            for (size_t i = 0; i < m; ++i) {
                double lerp = 1.0 + a_alpha * (last_a_(t, base + i) - 1.0);
                last_k_tilde_(t, base + i) = last_k_(t, base + i) * lerp;
            }
        }
    }

    // Step 4: wkv update per head + Step 5: output
    Tensor output(T, d_);
    for (size_t t = 0; t < T; ++t) {
        // wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t   (per head)
        // G_t[i, j] = δ_{ij} · w_t[h, j] - κ̂_t[h, i] · a_t[h, j] · κ̂_t[h, j]
        // Note: a_t index is j (column), NOT i (row).
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* wkv_prev = &last_wkv_(t, h * m * m);
            double* wkv_curr = &last_wkv_(t + 1, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < m; ++k) {
                        double G_kj;
                        if (k == j) {
                            G_kj = last_w_(t, base + j);
                        } else {
                            G_kj = 0.0;
                        }
                        // Rank-1: -κ̂_t[k] · a_t[j] · κ̂_t[j]
                        // k is the row of G we're multiplying by wkv_prev's col.
                        G_kj -= last_kappa_hat_(t, base + k)
                              * last_a_(t, base + j)
                              * last_kappa_hat_(t, base + j);
                        s += wkv_prev[i * m + k] * G_kj;
                    }
                    s += last_v_(t, base + i) * last_k_tilde_(t, base + j);
                    wkv_curr[i * m + j] = s;
                }
            }
        }
        // Output: o_t[h, j] = sum_i wkv_t[h, i, j] * r_t[h, i]
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* wkv_curr = &last_wkv_(t + 1, h * m * m);
            for (size_t j = 0; j < m; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    s += wkv_curr[i * m + j] * last_r_(t, base + i);
                }
                output(t, base + j) = s;
            }
        }
    }

    return output;
}

// ----------------------------------------------------------------------------
// Backward (single-step BPTT through the wkv state)
// ----------------------------------------------------------------------------
Tensor RWKV7TimeMix::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("RWKV7TimeMix: grad_output.cols must equal d");
    }
    size_t T = grad_output.rows;
    size_t m = head_dim_;

    Tensor grad_x(T, d_);
    grad_x.fill(0.0);

    // Zero Dense grads before backward so we get exact gradients from this pass.
    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_d.zero_grad();
    W_a.zero_grad();

    Tensor grad_r_pre(T, d_); grad_r_pre.fill(0.0);
    Tensor grad_k_pre(T, d_); grad_k_pre.fill(0.0);
    Tensor grad_v_pre(T, d_); grad_v_pre.fill(0.0);
    Tensor grad_d_pre_acc(T, d_); grad_d_pre_acc.fill(0.0);
    Tensor grad_a_pre_acc(T, d_); grad_a_pre_acc.fill(0.0);

    // BPTT through wkv state.
    // grad_wkv_next holds dL/dwkv_{t+1} (for use as the recurrence carrier into step t).
    // Scratch buffers (avoid re-allocating per step).
    // m ≤ 16 in our tests; allocate generously.
    const size_t MSCRATCH = 16;
    double grad_G[MSCRATCH * MSCRATCH];
    // grad_wkv_curr is per-head, stored in a tensor that survives across heads
    // (it's the carrier that flows back to step t-1).
    Tensor grad_wkv_curr(1, num_heads_ * m * m);
    grad_wkv_curr.fill(0.0);
    // grad_wkv_next is the carrier from step t+1 (saved across heads).
    Tensor grad_wkv_next(1, num_heads_ * m * m);
    grad_wkv_next.fill(0.0);

    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;

        // -------- For each head: compute g_wkv_direct[h] = r_t[h] · g_o_t[h]^T ---------
        //                  plus add carrier G_{t+1}^T · grad_wkv_next[h, i, :] ---------
        // grad_wkv_curr[h, i, j] is stored per-head in grad_wkv_curr tensor.
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* gwc_h = &grad_wkv_curr(0, h * m * m);  // grad_wkv_curr for this head
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    gwc_h[i * m + j] = last_r_(t, base + i) * grad_output(t, base + j);
                }
            }
            // Add carrier: G_{t+1}^T · grad_wkv_next[h, i, :]
            // grad_wkv_curr[h, i, j] += sum_{j'} G_{t+1}[j, j'] · grad_wkv_next[h, i, j']
            if (ti > 0) {  // ti == 0 means t == T-1, no t+1 exists
                double* gwv_next_h = &grad_wkv_next(0, h * m * m);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < m; ++j) {
                        double s = 0.0;
                        for (size_t jp = 0; jp < m; ++jp) {
                            double G_j_jp;
                            if (j == jp) {
                                G_j_jp = last_w_(t + 1, base + jp);
                            } else {
                                G_j_jp = 0.0;
                            }
                            G_j_jp -= last_kappa_hat_(t + 1, base + j)
                                    * last_a_(t + 1, base + jp)
                                    * last_kappa_hat_(t + 1, base + jp);
                            s += G_j_jp * gwv_next_h[i * m + jp];
                        }
                        gwc_h[i * m + j] += s;
                    }
                }
            }
        }

        // -------- Per-head gradient computations --------
        // The transition matrix G_t[h, i, j] = δ_{ij} · w_t[h, j] - κ̂_t[h, i] · a_t[h, j] · κ̂_t[h, j]
        // (rank-1 outer product with a on the column index).
        //
        // Three sources of dependency on κ̂_t[i]:
        //   - Row-i of G_t: G_t[i, j] depends on κ̂_t[i] via -κ̂_t[i] · a_t[j] · κ̂_t[j]
        //     dG_t[i, j]/dκ̂_t[i] = -a_t[j] · κ̂_t[j]
        //   - Col-i of G_t: G_t[r, i] depends on κ̂_t[i] via -κ̂_t[r] · a_t[i] · κ̂_t[i]
        //     dG_t[r, i]/dκ̂_t[i] = -κ̂_t[r] · a_t[i]
        // Three sources of dependency on a_t[i]:
        //   - Col-i of G_t: G_t[r, i] depends on a_t[i] via -κ̂_t[r] · a_t[i] · κ̂_t[i]
        //     dG_t[r, i]/da_t[i] = -κ̂_t[r] · κ̂_t[i]
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* wkv_prev = &last_wkv_(t, h * m * m);   // wkv_{t-1}[h]
            double* gwc_h = &grad_wkv_curr(0, h * m * m);  // grad_wkv_curr for this head
            // grad_G[i, j] = sum_{i'} wkv_prev[i', i] · grad_wkv_curr[h, i', j]
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double s = 0.0;
                    for (size_t ip = 0; ip < m; ++ip) {
                        s += wkv_prev[ip * m + i] * gwc_h[ip * m + j];
                    }
                    grad_G[i * m + j] = s;
                }
            }

            // ---- dL/dw_t[h, i] = grad_G[i, i] ----
            //   chain: w_t = exp(-exp(-0.5) · sigmoid(tanh(d_pre_t[h, i])))
            //   d(w_t)/d(d_pre) = -exp(-0.5) · w_t · sig · (1-sig) · (1 - tanh²)
            for (size_t i = 0; i < m; ++i) {
                double d_w = grad_G[i * m + i];
                double d_th = last_d_(t, base + i);          // tanh(d_pre_t)
                double s_sig = rwkv7_sigmoid(d_th);          // sigmoid(tanh(d_pre_t))
                double w_t = last_w_(t, base + i);
                double factor = -std::exp(-0.5) * w_t * s_sig * (1.0 - s_sig) * (1.0 - d_th * d_th);
                grad_d_pre_acc(t, base + i) += d_w * factor;
            }

            // ---- dL/dκ̂_t[h, i] (row + col contributions), dL/da_t[h, i] (col contributions) ----
            //   dL/dκ̂_t[i] = -sum_j grad_G[i, j] · a_t[j] · κ̂_t[j]              [row contribution]
            //               - sum_r grad_G[r, i] · κ̂_t[r] · a_t[i]               [col contribution]
            //   dL/da_t[i] = -sum_r grad_G[r, i] · κ̂_t[r] · κ̂_t[i]               [col contribution]
            // Then dL/da_t[h, i] gets another contribution from the k̃_t chain below.
            double d_kappa_hat[MSCRATCH];  // per-i in this head
            for (size_t i = 0; i < m; ++i) {
                // Row contribution: -sum_j a_t[j] · κ̂_t[j] · grad_G[i, j]
                double row_sum = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    row_sum += last_a_(t, base + j) * last_kappa_hat_(t, base + j) * grad_G[i * m + j];
                }
                // Col contribution: -a_t[i] · sum_r grad_G[r, i] · κ̂_t[r]
                // (NOT -κ̂_t[i] · a_t[i] · sum_r grad_G[r, i] — the κ̂_t is inside the sum)
                double col_kappa_dot_grad = 0.0;
                for (size_t r = 0; r < m; ++r) {
                    col_kappa_dot_grad += last_kappa_hat_(t, base + r) * grad_G[r * m + i];
                }
                double col_sum = -last_a_(t, base + i) * col_kappa_dot_grad;
                d_kappa_hat[i] = -row_sum + col_sum;
                // dL/da_t[i] += -κ̂_t[i] · sum_r κ̂_t[r] · grad_G[r, i]   [col contribution]
                double col_a_sum = 0.0;
                for (size_t r = 0; r < m; ++r) {
                    col_a_sum += last_kappa_hat_(t, base + r) * grad_G[r * m + i];
                }
                grad_a_pre_acc(t, base + i) += -last_kappa_hat_(t, base + i) * col_a_sum;
            }

            // ---- dL/dκ_t[h, i] via L2-normalize projection-back ----
            //   inner = sum_k d_kappa_hat[k] · κ̂_t[h, k]
            //   dL/dκ_t[h, i] = (1/norm) · [d_kappa_hat[i] - κ̂_t[h, i] · inner]
            double inner = 0.0;
            for (size_t i = 0; i < m; ++i) {
                inner += d_kappa_hat[i] * last_kappa_hat_(t, base + i);
            }
            double inv_norm = 1.0 / last_kappa_norm_(t, h);
            double d_kappa[MSCRATCH];
            for (size_t i = 0; i < m; ++i) {
                d_kappa[i] = inv_norm * (d_kappa_hat[i]
                                        - last_kappa_hat_(t, base + i) * inner);
                // dL/dξ[h, i] += k_t[h, i] · dL/dκ_t[h, i]
                grad_xi_(0, base + i) += last_k_(t, base + i) * d_kappa[i];
            }

            // ---- dL/dk̃_t[h, j] and dL/dv_t[h, i] (from outer-product term) ----
            //   v_t^T · k̃_t has [i, j] = v_t[i] · k̃_t[j]
            //   dL/dk̃_t[h, j] = sum_i v_t[h, i] · grad_wkv_curr[h, i, j]
            //   dL/dv_t[h, i] = sum_j k̃_t[h, j] · grad_wkv_curr[h, i, j]
            double d_k_tilde[MSCRATCH];
            for (size_t j = 0; j < m; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    s += last_v_(t, base + i) * gwc_h[i * m + j];
                }
                d_k_tilde[j] = s;
            }
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    s += last_k_tilde_(t, base + j) * gwc_h[i * m + j];
                }
                grad_v_pre(t, base + i) += s;
            }

            // ---- Distribute dL/dk̃_t to k_t, a_t, alpha ----
            //   dL/dk_t[h, i] += (1 + α·(a_t[h, i] - 1)) · dL/dk̃_t[h, i]
            //   dL/da_t[h, i] += α · k_t[h, i] · dL/dk̃_t[h, i]
            //   dL/dα += sum_i k_t[h, i] · (a_t[h, i] - 1) · dL/dk̃_t[h, i]
            //   dL/dk_t[h, i] += ξ[h, i] · dL/dκ_t[h, i]
            for (size_t i = 0; i < m; ++i) {
                double a_t_i = last_a_(t, base + i);
                double lerp = 1.0 + alpha(0, 0) * (a_t_i - 1.0);
                grad_k_pre(t, base + i) += lerp * d_k_tilde[i];
                grad_a_pre_acc(t, base + i) += alpha(0, 0) * last_k_(t, base + i) * d_k_tilde[i];
                grad_alpha_(0, 0) += last_k_(t, base + i) * (a_t_i - 1.0) * d_k_tilde[i];
                grad_k_pre(t, base + i) += xi(0, base + i) * d_kappa[i];
            }

            // ---- dL/dr_t[h, i] = sum_j wkv_t[h, i, j] · g_o_t[h, j] ----
            // (only the DIRECT upstream gradient, not the carrier)
            double* wkv_curr = &last_wkv_(t + 1, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    s += wkv_curr[i * m + j] * grad_output(t, base + j);
                }
                grad_r_pre(t, base + i) = s;
            }
        }  // end per-head loop

        // -------- Update grad_wkv_next for the next iteration (t-1) --------
        // grad_wkv_next[h, i, j] ← grad_wkv_curr[h, i, j]   for all heads
        if (ti < T - 1) {  // only if there's a t-1 to propagate to
            for (size_t h = 0; h < num_heads_; ++h) {
                double* gwv_next_h = &grad_wkv_next(0, h * m * m);
                double* gwc_h = &grad_wkv_curr(0, h * m * m);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < m; ++j) {
                        gwv_next_h[i * m + j] = gwc_h[i * m + j];
                    }
                }
            }
        }
    }  // end t loop

    // Backward through Dense projections.
    Tensor grad_r_in = W_r.backward(grad_r_pre, 0.0);
    Tensor grad_k_in = W_k.backward(grad_k_pre, 0.0);
    Tensor grad_v_in = W_v.backward(grad_v_pre, 0.0);
    Tensor grad_d_in = W_d.backward(grad_d_pre_acc, 0.0);
    // Chain through sigmoid: grad_a_pre_acc currently holds dL/da_t;
    // dL/da_pre_t = dL/da_t · a_t · (1 - a_t).
    Tensor grad_a_pre = Tensor(T, d_);
    grad_a_pre.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double a_t = last_a_(t, j);
            grad_a_pre(t, j) = grad_a_pre_acc(t, j) * a_t * (1.0 - a_t);
        }
    }
    Tensor grad_a_in = W_a.backward(grad_a_pre, 0.0);

    // Token-shift backward.
    Tensor grad_x_shifted(T, d_); grad_x_shifted.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double gr = grad_r_in(t, j);
            double gk = grad_k_in(t, j);
            double gv = grad_v_in(t, j);
            double gd = grad_d_in(t, j);
            double ga = grad_a_in(t, j);
            double x_diff = last_input_(t, j) - last_x_shift_(t, j);
            grad_x(t, j) += mu_r(0, j) * gr + mu_k(0, j) * gk + mu_v(0, j) * gv
                          + mu_d(0, j) * gd + mu_a(0, j) * ga;
            grad_x_shifted(t, j) = (1.0 - mu_r(0, j)) * gr
                                 + (1.0 - mu_k(0, j)) * gk
                                 + (1.0 - mu_v(0, j)) * gv
                                 + (1.0 - mu_d(0, j)) * gd
                                 + (1.0 - mu_a(0, j)) * ga;
            grad_mu_r_(0, j) += gr * x_diff;
            grad_mu_k_(0, j) += gk * x_diff;
            grad_mu_v_(0, j) += gv * x_diff;
            grad_mu_d_(0, j) += gd * x_diff;
            grad_mu_a_(0, j) += ga * x_diff;
        }
    }
    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            grad_x(t - 1, j) += grad_x_shifted(t, j);
        }
    }

    (void)grad_G;

    return grad_x;
}

// ----------------------------------------------------------------------------
// update_weights / zero_grad / parameters / gradients
// ----------------------------------------------------------------------------
void RWKV7TimeMix::update_weights(double learning_rate) {
    W_r.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_d.update_weights(learning_rate);
    W_a.update_weights(learning_rate);

    for (size_t j = 0; j < d_; ++j) {
        xi(0, j)     -= learning_rate * grad_xi_(0, j);
        mu_r(0, j)   -= learning_rate * grad_mu_r_(0, j);
        mu_k(0, j)   -= learning_rate * grad_mu_k_(0, j);
        mu_v(0, j)   -= learning_rate * grad_mu_v_(0, j);
        mu_d(0, j)   -= learning_rate * grad_mu_d_(0, j);
        mu_a(0, j)   -= learning_rate * grad_mu_a_(0, j);
    }
    alpha(0, 0) -= learning_rate * grad_alpha_(0, 0);
}

void RWKV7TimeMix::zero_grad() {
    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_d.zero_grad();
    W_a.zero_grad();
    grad_xi_.fill(0.0);
    grad_alpha_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
    grad_mu_d_.fill(0.0);
    grad_mu_a_.fill(0.0);
}

std::vector<Tensor*> RWKV7TimeMix::parameters() {
    return {
        &W_r.weights, &W_r.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &W_d.weights, &W_d.bias,
        &W_a.weights, &W_a.bias,
        &xi, &alpha,
        &mu_r, &mu_k, &mu_v, &mu_d, &mu_a
    };
}

std::vector<Tensor*> RWKV7TimeMix::gradients() {
    return {
        &W_r.grad_weights, &W_r.grad_bias,
        &W_k.grad_weights, &W_k.grad_bias,
        &W_v.grad_weights, &W_v.grad_bias,
        &W_d.grad_weights, &W_d.grad_bias,
        &W_a.grad_weights, &W_a.grad_bias,
        &grad_xi_, &grad_alpha_,
        &grad_mu_r_, &grad_mu_k_, &grad_mu_v_, &grad_mu_d_, &grad_mu_a_
    };
}