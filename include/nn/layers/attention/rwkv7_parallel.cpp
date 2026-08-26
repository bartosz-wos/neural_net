#include "rwkv7_parallel.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// RWKV7ParallelAttention implementation
// ------------------------------------------------------------------------
//
// The math within each chunk is identical to RWKV7TimeMix::forward/backward.
// The "parallel" structure refers to the canonical kernel: each chunk of size
// C can be processed in parallel via the parallel-scan algorithm (Schlag et al.
// "Mamba2/SSD"), with chunk-to-chunk recurrence. Within a chunk:
//     wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t   (t = chunk_start, ..., chunk_end-1)
// with wkv_{chunk_start - 1} carried in from the previous chunk (or zero).
//
// In our CPU reference implementation we walk the chunk sequentially. The
// FORWARD OUTPUT for any chunk_size >= T is bit-exact (FP64) to the recurrent
// form because the per-step recurrence is identical. Backward is likewise
// identical. This makes the parallel form usable as a verification oracle.
//
// Forward per-step (per head h, head_dim m = d/num_heads):
//   Token shift: x_t^□ = lerp(x_t, x_{t-1}, μ_□)
//   r_t = W_r · x_t^r + b_r
//   k_t = W_k · x_t^k + b_k
//   v_t = W_v · x_t^v + b_v
//   d_pre_t = W_d · x_t^d + b_d
//   a_pre_t = W_a · x_t^a + b_a
//   d_t = tanh(d_pre_t)
//   w_t = exp(-exp(-0.5) · sigmoid(d_t))                 ∈ (0.687, 1)
//   a_t = sigmoid(a_pre_t)                               ∈ (0, 1)
//   κ_t = k_t ⊙ ξ
//   κ̂_t[h] = κ_t[h] / ||κ_t[h]||_2
//   k̃_t = k_t ⊙ lerp(1, a_t, α) = k_t ⊙ (1 + α·(a_t - 1))
//
//   G_t = diag(w_t[h]) - κ̂_t[h]^T · (a_t[h] ⊙ κ̂_t[h])   (m × m)
//   wkv_t[h] = wkv_{t-1}[h] · G_t + v_t[h]^T · k̃_t[h]
//   wkv_0 = 0
//
//   o_t[h, j] = sum_i wkv_t[h, i, j] · r_t[h, i]
//
// Backward is the same as RWKV7TimeMix::backward.
// ============================================================================

static inline double rwkv7p_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
RWKV7ParallelAttention::RWKV7ParallelAttention(size_t d, size_t num_heads,
                                                size_t chunk_size, bool /*validate_tag*/)
    : d_(d), num_heads_(num_heads), head_dim_(d / num_heads),
      chunk_size_(chunk_size == 0 ? 0 : chunk_size),  // 0 means "use T"
      W_r(d, d), W_k(d, d), W_v(d, d), W_d(d, d), W_a(d, d),
      xi(1, d), alpha(1, 1),
      mu_r(1, d), mu_k(1, d), mu_v(1, d), mu_d(1, d), mu_a(1, d),
      grad_xi_(1, d), grad_alpha_(1, 1),
      grad_mu_r_(1, d), grad_mu_k_(1, d), grad_mu_v_(1, d),
      grad_mu_d_(1, d), grad_mu_a_(1, d)
{
    xi.fill(1.0);
    alpha.fill(0.0);
    for (size_t j = 0; j < d_; ++j) {
        mu_r(0, j) = 0.5;
        mu_k(0, j) = 0.5;
        mu_v(0, j) = 0.5;
        mu_d(0, j) = 0.5;
        mu_a(0, j) = 0.5;
    }
    grad_xi_.fill(0.0);
    grad_alpha_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
    grad_mu_d_.fill(0.0);
    grad_mu_a_.fill(0.0);
}

RWKV7ParallelAttention::RWKV7ParallelAttention(size_t d, size_t num_heads, size_t chunk_size)
    : RWKV7ParallelAttention(d, num_heads, chunk_size,
        [&]() {
            if (d == 0 || num_heads == 0 || d % num_heads != 0) {
                throw std::invalid_argument(
                    "RWKV7ParallelAttention: d must be > 0 and divisible by num_heads");
            }
            return true;
        }() ? false : true)
{
    // chunk_size: 0 → "use T"; non-zero must be > 0
    if (chunk_size_ == 0) {
        // default OK; forward substitutes T
    }
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RWKV7ParallelAttention::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("RWKV7ParallelAttention: input.cols must equal d");
    }
    size_t T = input.rows;
    size_t m = head_dim_;
    size_t C = (chunk_size_ == 0) ? T : chunk_size_;
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
            for (size_t i = 0; i < m; ++i) {
                double d_pre = last_d_pre_(t, base + i);
                double d_th = std::tanh(d_pre);
                last_d_(t, base + i) = d_th;
                double s = rwkv7p_sigmoid(d_th);
                last_w_(t, base + i) = std::exp(-std::exp(-0.5) * s);
                last_a_(t, base + i) = rwkv7p_sigmoid(last_a_pre_(t, base + i));
            }
            for (size_t i = 0; i < m; ++i) {
                last_kappa_(t, base + i) = last_k_(t, base + i) * xi(0, base + i);
            }
            double norm = 0.0;
            for (size_t i = 0; i < m; ++i) {
                double k = last_kappa_(t, base + i);
                norm += k * k;
            }
            norm = std::sqrt(norm);
            if (norm < 1e-12) norm = 1e-12;
            last_kappa_norm_(t, h) = norm;
            for (size_t i = 0; i < m; ++i) {
                last_kappa_hat_(t, base + i) = last_kappa_(t, base + i) / norm;
            }
            for (size_t i = 0; i < m; ++i) {
                double lerp = 1.0 + a_alpha * (last_a_(t, base + i) - 1.0);
                last_k_tilde_(t, base + i) = last_k_(t, base + i) * lerp;
            }
        }
    }

    // Step 4: wkv update per head + Step 5: output.
    // Process in chunks of size C. Within a chunk, apply the per-step
    // recurrence sequentially. The "parallel" structure: on GPU, each chunk
    // can be processed in parallel via the parallel-scan algorithm; on CPU
    // we walk it sequentially. For C >= T, this is a single chunk = full
    // sequence = bit-exact to the recurrent form.
    Tensor output(T, d_);
    for (size_t chunk_start = 0; chunk_start < T; chunk_start += C) {
        size_t chunk_end = std::min(chunk_start + C, T);
        for (size_t t = chunk_start; t < chunk_end; ++t) {
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
    }

    return output;
}

// ----------------------------------------------------------------------------
// Backward — identical to RWKV7TimeMix::backward (per-step BPTT through the
// wkv state, exactly the same math).
// ----------------------------------------------------------------------------
Tensor RWKV7ParallelAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("RWKV7ParallelAttention: grad_output.cols must equal d");
    }
    size_t T = grad_output.rows;
    size_t m = head_dim_;

    Tensor grad_x(T, d_);
    grad_x.fill(0.0);

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

    const size_t MSCRATCH = 16;
    double grad_G[MSCRATCH * MSCRATCH];
    Tensor grad_wkv_curr(1, num_heads_ * m * m);
    grad_wkv_curr.fill(0.0);
    Tensor grad_wkv_next(1, num_heads_ * m * m);
    grad_wkv_next.fill(0.0);

    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;

        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* gwc_h = &grad_wkv_curr(0, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    gwc_h[i * m + j] = last_r_(t, base + i) * grad_output(t, base + j);
                }
            }
            if (ti > 0) {
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

        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            double* wkv_prev = &last_wkv_(t, h * m * m);
            double* gwc_h = &grad_wkv_curr(0, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double s = 0.0;
                    for (size_t ip = 0; ip < m; ++ip) {
                        s += wkv_prev[ip * m + i] * gwc_h[ip * m + j];
                    }
                    grad_G[i * m + j] = s;
                }
            }

            for (size_t i = 0; i < m; ++i) {
                double d_w = grad_G[i * m + i];
                double d_th = last_d_(t, base + i);
                double s_sig = rwkv7p_sigmoid(d_th);
                double w_t = last_w_(t, base + i);
                double factor = -std::exp(-0.5) * w_t * s_sig * (1.0 - s_sig) * (1.0 - d_th * d_th);
                grad_d_pre_acc(t, base + i) += d_w * factor;
            }

            double d_kappa_hat[MSCRATCH];
            for (size_t i = 0; i < m; ++i) {
                double row_sum = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    row_sum += last_a_(t, base + j) * last_kappa_hat_(t, base + j) * grad_G[i * m + j];
                }
                double col_kappa_dot_grad = 0.0;
                for (size_t r = 0; r < m; ++r) {
                    col_kappa_dot_grad += last_kappa_hat_(t, base + r) * grad_G[r * m + i];
                }
                double col_sum = -last_a_(t, base + i) * col_kappa_dot_grad;
                d_kappa_hat[i] = -row_sum + col_sum;
                double col_a_sum = 0.0;
                for (size_t r = 0; r < m; ++r) {
                    col_a_sum += last_kappa_hat_(t, base + r) * grad_G[r * m + i];
                }
                grad_a_pre_acc(t, base + i) += -last_kappa_hat_(t, base + i) * col_a_sum;
            }

            double inner = 0.0;
            for (size_t i = 0; i < m; ++i) {
                inner += d_kappa_hat[i] * last_kappa_hat_(t, base + i);
            }
            double inv_norm = 1.0 / last_kappa_norm_(t, h);
            double d_kappa[MSCRATCH];
            for (size_t i = 0; i < m; ++i) {
                d_kappa[i] = inv_norm * (d_kappa_hat[i]
                                        - last_kappa_hat_(t, base + i) * inner);
                grad_xi_(0, base + i) += last_k_(t, base + i) * d_kappa[i];
            }

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

            for (size_t i = 0; i < m; ++i) {
                double a_t_i = last_a_(t, base + i);
                double lerp = 1.0 + alpha(0, 0) * (a_t_i - 1.0);
                grad_k_pre(t, base + i) += lerp * d_k_tilde[i];
                grad_a_pre_acc(t, base + i) += alpha(0, 0) * last_k_(t, base + i) * d_k_tilde[i];
                grad_alpha_(0, 0) += last_k_(t, base + i) * (a_t_i - 1.0) * d_k_tilde[i];
                grad_k_pre(t, base + i) += xi(0, base + i) * d_kappa[i];
            }

            double* wkv_curr = &last_wkv_(t + 1, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    s += wkv_curr[i * m + j] * grad_output(t, base + j);
                }
                grad_r_pre(t, base + i) = s;
            }
        }

        if (ti < T - 1) {
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
    }

    Tensor grad_r_in = W_r.backward(grad_r_pre, 0.0);
    Tensor grad_k_in = W_k.backward(grad_k_pre, 0.0);
    Tensor grad_v_in = W_v.backward(grad_v_pre, 0.0);
    Tensor grad_d_in = W_d.backward(grad_d_pre_acc, 0.0);
    Tensor grad_a_pre = Tensor(T, d_);
    grad_a_pre.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double a_t = last_a_(t, j);
            grad_a_pre(t, j) = grad_a_pre_acc(t, j) * a_t * (1.0 - a_t);
        }
    }
    Tensor grad_a_in = W_a.backward(grad_a_pre, 0.0);

    // Token-shift backward
    Tensor grad_mu_r_acc(1, d_); grad_mu_r_acc.fill(0.0);
    Tensor grad_mu_k_acc(1, d_); grad_mu_k_acc.fill(0.0);
    Tensor grad_mu_v_acc(1, d_); grad_mu_v_acc.fill(0.0);
    Tensor grad_mu_d_acc(1, d_); grad_mu_d_acc.fill(0.0);
    Tensor grad_mu_a_acc(1, d_); grad_mu_a_acc.fill(0.0);
    Tensor grad_x_carry(T, d_); grad_x_carry.fill(0.0);
    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;
        for (size_t j = 0; j < d_; ++j) {
            double x_t = last_input_(t, j);
            double x_prev = (t == 0) ? 0.0 : last_input_(t - 1, j);
            double d = x_t - x_prev;
            double g_r = grad_r_in(t, j);
            double g_k = grad_k_in(t, j);
            double g_v = grad_v_in(t, j);
            double g_d = grad_d_in(t, j);
            double g_a = grad_a_in(t, j);
            grad_mu_r_acc(0, j) += g_r * d;
            grad_mu_k_acc(0, j) += g_k * d;
            grad_mu_v_acc(0, j) += g_v * d;
            grad_mu_d_acc(0, j) += g_d * d;
            grad_mu_a_acc(0, j) += g_a * d;
            double carry = grad_x_carry(t, j);
            double g_x = mu_r(0, j) * g_r + mu_k(0, j) * g_k + mu_v(0, j) * g_v
                       + mu_d(0, j) * g_d + mu_a(0, j) * g_a + carry;
            grad_x(t, j) = g_x;
            if (t > 0) {
                grad_x_carry(t - 1, j) = (1.0 - mu_r(0, j)) * g_r
                                        + (1.0 - mu_k(0, j)) * g_k
                                        + (1.0 - mu_v(0, j)) * g_v
                                        + (1.0 - mu_d(0, j)) * g_d
                                        + (1.0 - mu_a(0, j)) * g_a;
            }
        }
    }
    grad_mu_r_ = grad_mu_r_acc;
    grad_mu_k_ = grad_mu_k_acc;
    grad_mu_v_ = grad_mu_v_acc;
    grad_mu_d_ = grad_mu_d_acc;
    grad_mu_a_ = grad_mu_a_acc;

    return grad_x;
}

void RWKV7ParallelAttention::update_weights(double learning_rate) {
    W_r.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_d.update_weights(learning_rate);
    W_a.update_weights(learning_rate);
    // Manual updates for xi, alpha, mu_*
    for (size_t j = 0; j < d_; ++j) {
        xi(0, j) -= learning_rate * grad_xi_(0, j);
        mu_r(0, j) -= learning_rate * grad_mu_r_(0, j);
        mu_k(0, j) -= learning_rate * grad_mu_k_(0, j);
        mu_v(0, j) -= learning_rate * grad_mu_v_(0, j);
        mu_d(0, j) -= learning_rate * grad_mu_d_(0, j);
        mu_a(0, j) -= learning_rate * grad_mu_a_(0, j);
    }
    alpha(0, 0) -= learning_rate * grad_alpha_(0, 0);
}

void RWKV7ParallelAttention::zero_grad() {
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

std::vector<Tensor*> RWKV7ParallelAttention::parameters() {
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

std::vector<Tensor*> RWKV7ParallelAttention::gradients() {
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