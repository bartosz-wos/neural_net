#include "rwkv5.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// RWKV5TimeMix implementation
//
// Forward (per head h, i, j ∈ [0, m) where m = d / num_heads):
//   1) Token-shift: r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}; same for k, v.
//   2) Projections: r_t = W_r · r_in_t + b_r; k_t = W_k · k_in_t + b_k; v_t = W_v · v_in_t + b_v.
//   3) Per-head WKV recurrence:
//        kv_t[h, i, j] = k_t[h*m+i] · v_t[h*m+j]
//        a[h*m+i] = exp(-exp(log_w[h*m+i]))
//        s_t[h, i, j] = a[h*m+i] · s_{t-1}[h, i, j] + kv_t[h, i, j]
//        wkv_t[h, i, j] = s_t[h, i, j] + u[h*m+i] · kv_t[h, i, j]
//   4) out_pre_t[h*m+i] = Σ_j r_t[h*m+i] · wkv_t[h, i, j]
//   5) out_t = out_pre_t · W_o^T + b_o
//
// Backward (per token, in reverse order t = T-1, ..., 0):
//
//   Define the LOCAL gradient into wkv:
//     d_wkv[h, i, j] = r[h*m+i] · d_out_pre[h*m+i]      (from out_pre = Σ_j r · wkv)
//
//   wkv depends on s (coeff 1) AND on kv (coeff u_i):
//     d_s[h, i, j] = d_wkv[h, i, j]                      (∂wkv/∂s = 1)
//     d_kv[h, i, j] gets u_i · d_wkv[h, i, j]           (∂wkv/∂kv = u_i)
//
//   s depends on s_{t-1} (coeff a_i) AND on kv (coeff 1):
//     d_s[h, i, j] += grad_s_prev[h, i, j]               (carrier from t+1: d_s[t+1] · a_i)
//     d_kv[h, i, j] += d_s[h, i, j] = d_wkv[h, i, j] + grad_s_prev[h, i, j]
//
//   Combine:
//     d_kv[h, i, j] = u_i · d_wkv[h, i, j] + d_wkv[h, i, j] + grad_s_prev[h, i, j]
//                   = (1 + u_i) · d_wkv[h, i, j] + grad_s_prev[h, i, j]
//
//   Update carrier for next iteration (t → t-1):
//     new_grad_s_prev[h, i, j] = d_s[h, i, j] · a_i = (d_wkv[h, i, j] + grad_s_prev[h, i, j]) · a_i
//
//   Per-channel: grad_a[h*m+i] += Σ_j d_s[h, i, j] · s_{t-1}[h, i, j]
//     grad_log_w[h*m+i] = grad_a[h*m+i] · a[h*m+i] · (-exp(log_w[h*m+i]))
//
//   Per-channel: grad_u[h*m+i] += Σ_j kv[h, i, j] · d_wkv[h, i, j]
//
//   Outer product backward (kv = k ⊗ v):
//     d_k[h*m+i] += Σ_j d_kv[h, i, j] · v[h*m+j]
//     d_v[h*m+j] += Σ_i d_kv[h, i, j] · k[h*m+i]
//
//   r-path backward (out_pre = Σ_j r · wkv):
//     d_r[h*m+i] += Σ_j wkv[h, i, j] · d_out_pre[h*m+i]
// ============================================================================

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
RWKV5TimeMix::RWKV5TimeMix(size_t d, size_t num_heads)
    : d_(d),
      num_heads_(num_heads),
      head_dim_((num_heads == 0) ? 0 : d / num_heads),
      W_r(d, d), W_k(d, d), W_v(d, d), W_o(d, d),
      log_w(1, d), u(1, d),
      mu_r(1, d), mu_k(1, d), mu_v(1, d),
      grad_log_w_(1, d), grad_u_(1, d),
      grad_mu_r_(1, d), grad_mu_k_(1, d), grad_mu_v_(1, d)
{
    if (d == 0) {
        throw std::invalid_argument("RWKV5TimeMix: d must be > 0");
    }
    if (num_heads == 0) {
        throw std::invalid_argument("RWKV5TimeMix: num_heads must be > 0");
    }
    if (d % num_heads != 0) {
        throw std::invalid_argument("RWKV5TimeMix: d must be divisible by num_heads");
    }

    // Eagle Appendix I:
    //   * W_r, W_k, W_v, W_o: xavier-uniform (Dense default — already done by Dense ctor).
    //   * b_*: zero (Dense default).
    //   * Override: W_o weights and bias are zero-initialized (Appendix I, "W_o: zero
    //     init → output is exactly 0 at the start — training warmup"). This matches
    //     the rwkv6 convention. Dense ctor uses xavier, so we override here.
    W_o.weights.fill(0.0);
    W_o.bias.fill(0.0);
    log_w.fill(-0.5);            // Eagle convention (moderate decay)
    u.fill(0.0);                  // no current-token bonus at start
    mu_r.fill(0.5);
    mu_k.fill(0.5);
    mu_v.fill(0.5);

    // Zero all gradient buffers.
    grad_log_w_.fill(0.0);
    grad_u_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
}

RWKV5TimeMix::~RWKV5TimeMix() = default;

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RWKV5TimeMix::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("RWKV5TimeMix: input.cols must equal d");
    }
    size_t T = input.rows;
    size_t H = num_heads_;
    size_t m = head_dim_;
    size_t HMm = H * m * m;

    last_input_ = input.clone();  // preserve the input

    // Allocate cache
    last_x_shift_ = Tensor(T, d_); last_x_shift_.fill(0.0);
    last_r_in_    = Tensor(T, d_); last_k_in_    = Tensor(T, d_); last_v_in_    = Tensor(T, d_);
    last_r_       = Tensor(T, d_); last_k_       = Tensor(T, d_); last_v_       = Tensor(T, d_);
    last_kv_      = Tensor(T, HMm);
    last_s_       = Tensor(T + 1, HMm); last_s_.fill(0.0);
    last_wkv_     = Tensor(T, HMm);
    last_out_pre_ = Tensor(T, d_);
    last_a_       = Tensor(1, d_);

    // Per-channel decay (constants per forward): a[i] = exp(-exp(log_w[i]))
    for (size_t j = 0; j < d_; ++j) {
        last_a_(0, j) = std::exp(-std::exp(log_w(0, j)));
    }

    // Step 1: token-shift mixing
    //   r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}    (x_{-1} := 0)
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double xt   = last_input_(t, j);
            double xsh  = (t == 0) ? 0.0 : last_input_(t - 1, j);
            last_x_shift_(t, j) = xsh;
            last_r_in_(t, j) = mu_r(0, j) * xt + (1.0 - mu_r(0, j)) * xsh;
            last_k_in_(t, j) = mu_k(0, j) * xt + (1.0 - mu_k(0, j)) * xsh;
            last_v_in_(t, j) = mu_v(0, j) * xt + (1.0 - mu_v(0, j)) * xsh;
        }
    }

    // Step 2: projections (Dense)
    last_r_ = W_r.forward(last_r_in_);
    last_k_ = W_k.forward(last_k_in_);
    last_v_ = W_v.forward(last_v_in_);

    // Step 3: per-head WKV recurrence (matrix-valued state)
    // Step 4: out_pre[t, h*m+j] = Σ_i r[t, h*m+i] · wkv[t, h, i, j]
    //         (rwkv6 §3.1 convention: r and the matrix index i are paired; j is the output channel)
    for (size_t t = 0; t < T; ++t) {
        const double* r_t = &last_r_(t, 0);
        const double* k_t = &last_k_(t, 0);
        const double* v_t = &last_v_(t, 0);
        const double* s_prev = &last_s_(t, 0);     // s_{t-1} flat
        double* s_curr = &last_s_(t + 1, 0);       // s_t flat
        double* kv_t = &last_kv_(t, 0);
        double* wkv_t = &last_wkv_(t, 0);
        // zero out_pre row
        for (size_t j = 0; j < d_; ++j) last_out_pre_(t, j) = 0.0;

        for (size_t h = 0; h < H; ++h) {
            size_t head_off = h * m;
            for (size_t i = 0; i < m; ++i) {
                size_t ii = head_off + i;
                double a_i = last_a_(0, ii);
                double u_i = u(0, ii);
                double r_ii = r_t[ii];
                for (size_t j = 0; j < m; ++j) {
                    size_t jj = head_off + j;
                    size_t flat = h * m * m + i * m + j;
                    double kv_ij = k_t[ii] * v_t[jj];
                    kv_t[flat] = kv_ij;
                    double s_ij = a_i * s_prev[flat] + kv_ij;
                    s_curr[flat] = s_ij;
                    double wkv_ij = s_ij + u_i * kv_ij;
                    wkv_t[flat] = wkv_ij;
                    // accumulate out_pre: out_pre[h*m+j] += r[h*m+i] · wkv[h, i, j]
                    //                    (j is the OUTPUT channel index, i is summed over)
                    last_out_pre_(t, jj) += r_ii * wkv_ij;
                }
            }
        }
    }

    // Step 5: output projection
    Tensor output = W_o.forward(last_out_pre_);
    return output;
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
Tensor RWKV5TimeMix::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("RWKV5TimeMix: grad_output.cols must equal d");
    }
    size_t T = grad_output.rows;
    size_t H = num_heads_;
    size_t m = head_dim_;
    size_t HMm = H * m * m;

    // Backprop through W_o to get grad_out_pre (Dense.backward returns grad_input
    // of the W_o projection — that IS grad_out_pre).
    W_o.zero_grad();
    Tensor grad_out_pre = W_o.backward(grad_output, 0.0);   // (T, d)

    // Per-token gradient buffers
    Tensor grad_kv(T, HMm);    grad_kv.fill(0.0);
    Tensor grad_r_pre(T, d_);  grad_r_pre.fill(0.0);
    Tensor grad_k_pre(T, d_);  grad_k_pre.fill(0.0);
    Tensor grad_v_pre(T, d_);  grad_v_pre.fill(0.0);

    // Recurrence carrier: grad_s_prev (1, HMm) — dL/d(s_{t-1}) carried from t+1.
    Tensor grad_s_prev(1, HMm); grad_s_prev.fill(0.0);

    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;  // backward

        const double* s_prev = &last_s_(t, 0);       // s_{t-1} flat (row t of cache)
        const double* kv_t   = &last_kv_(t, 0);
        const double* wkv_t  = &last_wkv_(t, 0);
        const double* r_t    = &last_r_(t, 0);
        const double* k_t    = &last_k_(t, 0);
        const double* v_t    = &last_v_(t, 0);

        // We will compute for each head (h, i, j):
        //   out_pre[h*m+j] = Σ_i r[h*m+i] · wkv[h, i, j]   (forward)
        //   d_wkv[h, i, j] = r[h*m+i] · d_out_pre[h*m+j]   (∂out_pre/∂wkv = r[i])
        //   d_s[h, i, j]   = d_wkv[h, i, j] + grad_s_prev[h, i, j]   (wkv = s + u·kv + carrier)
        //   d_kv[h, i, j]  = (1 + u_i) · d_wkv[h, i, j] + grad_s_prev[h, i, j]   (combined)
        //   grad_u[h*m+i] += Σ_j kv[h, i, j] · d_wkv[h, i, j]
        //   grad_a[h*m+i] += Σ_j d_s[h, i, j] · s_prev[h, i, j]
        //   new_grad_s_prev[h, i, j] = d_s[h, i, j] · a_i
        //   grad_r_pre[h*m+i] += Σ_j wkv[h, i, j] · d_out_pre[h*m+j]   (∂out_pre/∂r[i] = wkv[i, j])
        //   d_k[h*m+i] += Σ_j d_kv[h, i, j] · v[h*m+j]
        //   d_v[h*m+j] += Σ_i d_kv[h, i, j] · k[h*m+i]

        for (size_t h = 0; h < H; ++h) {
            size_t head_off = h * m;
            for (size_t i = 0; i < m; ++i) {
                size_t ii = head_off + i;
                double a_i = last_a_(0, ii);
                double u_i = u(0, ii);
                double r_ii = r_t[ii];

                double grad_a_acc = 0.0;
                double grad_u_acc = 0.0;
                double grad_r_acc = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    size_t jj = head_off + j;
                    size_t flat = h * m * m + i * m + j;
                    double kv_ij = kv_t[flat];
                    double wkv_ij = wkv_t[flat];
                    double grad_s_prev_ij = grad_s_prev(0, flat);
                    double d_out_pre_j = grad_out_pre(t, jj);

                    // d_wkv[h, i, j] = r[h*m+i] · d_out_pre[h*m+j]
                    double d_wkv_ij = r_ii * d_out_pre_j;

                    // grad_u[h*m+i] += kv[h, i, j] · d_wkv[h, i, j]
                    grad_u_acc += kv_ij * d_wkv_ij;

                    // d_s[h, i, j] = d_wkv[h, i, j] + grad_s_prev[h, i, j]
                    double d_s_ij = d_wkv_ij + grad_s_prev_ij;

                    // Combined d_kv[h, i, j] = (1 + u_i) · d_wkv_ij + grad_s_prev_ij
                    grad_kv(t, flat) = (1.0 + u_i) * d_wkv_ij + grad_s_prev_ij;

                    // grad_a[h*m+i] += d_s[h, i, j] · s_prev[h, i, j]
                    grad_a_acc += d_s_ij * s_prev[flat];

                    // grad_r_pre[h*m+i] += wkv[h, i, j] · d_out_pre[h*m+j]
                    grad_r_acc += wkv_ij * d_out_pre_j;
                }
                // Single updates after the per-j loop:
                grad_u_(0, ii) += grad_u_acc;
                grad_log_w_(0, ii) += grad_a_acc * a_i * (-std::exp(log_w(0, ii)));
                grad_r_pre(t, ii) += grad_r_acc;
            }
        }

        // Update grad_s_prev carrier for the next iteration (t → t-1):
        //   new_grad_s_prev[h, i, j] = (d_wkv[h, i, j] + grad_s_prev[h, i, j]) · a[h*m+i]
        //   d_wkv[h, i, j] = r[h*m+i] · d_out_pre[h*m+j]
        for (size_t h = 0; h < H; ++h) {
            size_t head_off = h * m;
            for (size_t i = 0; i < m; ++i) {
                size_t ii = head_off + i;
                double a_i = last_a_(0, ii);
                double r_ii = r_t[ii];
                for (size_t j = 0; j < m; ++j) {
                    size_t jj = head_off + j;
                    size_t flat = h * m * m + i * m + j;
                    double grad_s_prev_ij = grad_s_prev(0, flat);
                    double d_out_pre_j = grad_out_pre(t, jj);
                    double d_wkv_ij = r_ii * d_out_pre_j;
                    double d_s_ij = d_wkv_ij + grad_s_prev_ij;
                    grad_s_prev(0, flat) = d_s_ij * a_i;
                }
            }
        }

        // Outer product backward (kv = k ⊗ v):
        //   d_k[h*m+i] += Σ_j d_kv[h, i, j] · v[h*m+j]
        //   d_v[h*m+j] += Σ_i d_kv[h, i, j] · k[h*m+i]
        for (size_t h = 0; h < H; ++h) {
            size_t head_off = h * m;
            for (size_t i = 0; i < m; ++i) {
                size_t ii = head_off + i;
                double dk_acc = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    size_t jj = head_off + j;
                    size_t flat = h * m * m + i * m + j;
                    dk_acc += grad_kv(t, flat) * v_t[jj];
                }
                grad_k_pre(t, ii) += dk_acc;
            }
            for (size_t j = 0; j < m; ++j) {
                size_t jj = head_off + j;
                double dv_acc = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    size_t ii = head_off + i;
                    size_t flat = h * m * m + i * m + j;
                    dv_acc += grad_kv(t, flat) * k_t[ii];
                }
                grad_v_pre(t, jj) += dv_acc;
            }
        }
    }

    // Backprop through Dense projections to get grad_r_in, grad_k_in, grad_v_in
    // and accumulate W_*.weights / b_* gradients.
    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    Tensor grad_r_in = W_r.backward(grad_r_pre, 0.0);
    Tensor grad_k_in = W_k.backward(grad_k_pre, 0.0);
    Tensor grad_v_in = W_v.backward(grad_v_pre, 0.0);

    // μ-mix backward. r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}.
    Tensor grad_x(T, d_); grad_x.fill(0.0);
    Tensor grad_x_shifted(T, d_); grad_x_shifted.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double gr = grad_r_in(t, j);
            double gk = grad_k_in(t, j);
            double gv = grad_v_in(t, j);
            double x_diff = last_input_(t, j) - last_x_shift_(t, j);
            // Direct contribution to grad_x[t]
            grad_x(t, j) += mu_r(0, j) * gr + mu_k(0, j) * gk + mu_v(0, j) * gv;
            // Contribution to grad_x[t-1] (carry forward)
            grad_x_shifted(t, j) = (1.0 - mu_r(0, j)) * gr
                                  + (1.0 - mu_k(0, j)) * gk
                                  + (1.0 - mu_v(0, j)) * gv;
            // μ gradient
            grad_mu_r_(0, j) += gr * x_diff;
            grad_mu_k_(0, j) += gk * x_diff;
            grad_mu_v_(0, j) += gv * x_diff;
        }
    }
    // Apply the carry-forward: grad_x[t-1] += grad_x_shifted[t]
    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            grad_x(t - 1, j) += grad_x_shifted(t, j);
        }
    }

    return grad_x;
}

// ----------------------------------------------------------------------------
// update_weights / zero_grad / parameters / gradients
// ----------------------------------------------------------------------------
void RWKV5TimeMix::update_weights(double learning_rate) {
    W_r.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);

    // log_w, u, mu_r, mu_k, mu_v
    for (size_t j = 0; j < d_; ++j) {
        log_w(0, j) -= learning_rate * grad_log_w_(0, j);
        u(0, j)     -= learning_rate * grad_u_(0, j);
        mu_r(0, j)  -= learning_rate * grad_mu_r_(0, j);
        mu_k(0, j)  -= learning_rate * grad_mu_k_(0, j);
        mu_v(0, j)  -= learning_rate * grad_mu_v_(0, j);
    }
}

void RWKV5TimeMix::zero_grad() {
    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    grad_log_w_.fill(0.0);
    grad_u_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
}

std::vector<Tensor*> RWKV5TimeMix::parameters() {
    return {
        &W_r.weights, &W_r.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &W_o.weights, &W_o.bias,
        &log_w, &u, &mu_r, &mu_k, &mu_v
    };
}

std::vector<Tensor*> RWKV5TimeMix::gradients() {
    return {
        &W_r.grad_weights, &W_r.grad_bias,
        &W_k.grad_weights, &W_k.grad_bias,
        &W_v.grad_weights, &W_v.grad_bias,
        &W_o.grad_weights, &W_o.grad_bias,
        &grad_log_w_, &grad_u_, &grad_mu_r_, &grad_mu_k_, &grad_mu_v_
    };
}