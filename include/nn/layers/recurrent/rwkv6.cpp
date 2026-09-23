#include "rwkv6.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// RWKV-6 "Finch" implementation
// =============================================================================

static inline double rwkv6_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ----------------------------------------------------------------------------
// Constructor (validating helper)
// ----------------------------------------------------------------------------
RWKV6TimeMix::RWKV6TimeMix(size_t d, size_t num_heads, size_t num_lora_ranks, bool /*validate_tag*/)
    : d_(d), num_heads_(num_heads),
      head_dim_((num_heads == 0) ? 0 : d / num_heads),
      num_lora_ranks_(num_lora_ranks),
      W_r(d, d), W_k(d, d), W_v(d, d), W_g(d, d), W_o(d, d),
      u(1, d), mu_x(1, d), mu_w(1, d),
      A_r(d, num_lora_ranks),      B_r(num_lora_ranks, d),      lambda_r(1, d),
      A_k(d, num_lora_ranks),      B_k(num_lora_ranks, d),      lambda_k(1, d),
      A_v(d, num_lora_ranks),      B_v(num_lora_ranks, d),      lambda_v(1, d),
      A_g(d, num_lora_ranks),      B_g(num_lora_ranks, d),      lambda_g(1, d),
      A_d(d, 2 * num_lora_ranks),  B_d(2 * num_lora_ranks, d),  lambda_d(1, d),
      A_x(d, num_lora_ranks),      B_x(num_lora_ranks, d),      lambda_x(1, d),
      grad_u_(1, d),
      grad_mu_x_(1, d), grad_mu_w_(1, d),
      grad_lambda_r_(1, d), grad_lambda_k_(1, d), grad_lambda_v_(1, d),
      grad_lambda_g_(1, d), grad_lambda_d_(1, d), grad_lambda_x_(1, d),
      grad_A_r_(d, num_lora_ranks), grad_B_r_(num_lora_ranks, d),
      grad_A_k_(d, num_lora_ranks), grad_B_k_(num_lora_ranks, d),
      grad_A_v_(d, num_lora_ranks), grad_B_v_(num_lora_ranks, d),
      grad_A_g_(d, num_lora_ranks), grad_B_g_(num_lora_ranks, d),
      grad_A_d_(d, 2 * num_lora_ranks), grad_B_d_(2 * num_lora_ranks, d),
      grad_A_x_(d, num_lora_ranks), grad_B_x_(num_lora_ranks, d)
{
    if (d == 0 || num_heads == 0 || d % num_heads != 0) {
        throw std::invalid_argument(
            "RWKV6TimeMix: d must be > 0 and divisible by num_heads");
    }
    if (num_lora_ranks == 0) {
        throw std::invalid_argument(
            "RWKV6TimeMix: num_lora_ranks must be > 0");
    }

    W_o.weights.fill(0.0);
    W_o.bias.fill(0.0);

    for (size_t i = 0; i < d_; ++i) {
        double r0 = 0.5;
        double v = r0 * (1.0 - (double)i / (double)(d_ - 1))
                 + 0.1 * ((double)((i + 1) % 3));
        u(0, i) = v;
    }

    for (size_t i = 0; i < d_; ++i) {
        double v = 1.0 - (double)i / (double)d_;
        mu_x(0, i) = v;
        mu_w(0, i) = v;
    }

    lambda_r.fill(0.0); lambda_k.fill(0.0); lambda_v.fill(0.0);
    lambda_g.fill(0.0); lambda_d.fill(0.0); lambda_x.fill(0.0);

    auto init_lora_AB = [&](Tensor& A, Tensor& B) {
        for (size_t i = 0; i < A.rows; ++i)
            for (size_t j = 0; j < A.cols; ++j)
                A(i, j) = 1e-4 * std::sin((double)(i * 7 + j * 13 + 1));
        for (size_t i = 0; i < B.rows; ++i)
            for (size_t j = 0; j < B.cols; ++j)
                B(i, j) = 1e-4 * std::sin((double)(i * 11 + j * 17 + 3));
    };
    init_lora_AB(A_g, B_g); init_lora_AB(A_d, B_d); init_lora_AB(A_x, B_x);

    grad_u_.fill(0.0);
    grad_mu_x_.fill(0.0); grad_mu_w_.fill(0.0);
    grad_lambda_r_.fill(0.0); grad_lambda_k_.fill(0.0); grad_lambda_v_.fill(0.0);
    grad_lambda_g_.fill(0.0); grad_lambda_d_.fill(0.0); grad_lambda_x_.fill(0.0);
    grad_A_r_.fill(0.0); grad_B_r_.fill(0.0);
    grad_A_k_.fill(0.0); grad_B_k_.fill(0.0);
    grad_A_v_.fill(0.0); grad_B_v_.fill(0.0);
    grad_A_g_.fill(0.0); grad_B_g_.fill(0.0);
    grad_A_d_.fill(0.0); grad_B_d_.fill(0.0);
    grad_A_x_.fill(0.0); grad_B_x_.fill(0.0);
}

RWKV6TimeMix::RWKV6TimeMix(size_t d, size_t num_heads, size_t num_lora_ranks)
    : RWKV6TimeMix(d, num_heads, num_lora_ranks, false) {}

RWKV6TimeMix::~RWKV6TimeMix() = default;

void rwv6_test_probe(const char* /*tag*/) {}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RWKV6TimeMix::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("RWKV6TimeMix: input.cols must equal d");
    }
    size_t T = input.rows;
    size_t m = head_dim_;

    last_input_ = input.clone();

    // Cache x_{t-1} (zero row at t=0). This MUST be a separate buffer from
    // last_x_shift_ (the shift_x_t cache), because ddlerp writes back to its
    // shift_out argument and would otherwise overwrite the input cache.
    last_x_prev_ = Tensor(T, d_);
    last_x_prev_.fill(0.0);
    for (size_t t = 1; t < T; ++t)
        for (size_t j = 0; j < d_; ++j)
            last_x_prev_(t, j) = input(t - 1, j);

    last_x_shift_ = Tensor(T, d_);
    last_x_shift_.fill(0.0);

    last_r_shift_ = Tensor(T, d_); last_k_shift_ = Tensor(T, d_); last_v_shift_ = Tensor(T, d_);
    last_g_shift_ = Tensor(T, d_); last_d_shift_ = Tensor(T, d_);
    last_lora_r_in_ = Tensor(T, d_); last_lora_k_in_ = Tensor(T, d_); last_lora_v_in_ = Tensor(T, d_);
    last_lora_g_in_ = Tensor(T, d_); last_lora_d_in_ = Tensor(T, d_); last_lora_x_in_ = Tensor(T, d_);
    last_lora_r_ = Tensor(T, d_); last_lora_k_ = Tensor(T, d_); last_lora_v_ = Tensor(T, d_);
    last_lora_g_ = Tensor(T, d_); last_lora_d_ = Tensor(T, d_); last_lora_x_ = Tensor(T, d_);
    last_r_ = Tensor(T, d_); last_k_ = Tensor(T, d_); last_v_ = Tensor(T, d_);
    last_g_ = Tensor(T, d_); last_d_pre_ = Tensor(T, d_);
    last_w_ = Tensor(T, d_);
    last_kv_ = Tensor(T, num_heads_ * m * m);
    last_s_ = Tensor(T + 1, num_heads_ * m * m);
    last_rwkv_ = Tensor(T, d_);
    last_s_.fill(0.0);

    for (size_t t = 0; t < T; ++t) {
        auto ddlerp = [&](const Tensor& A, const Tensor& B, const Tensor& lam,
                          const Tensor& mu, Tensor& shift_out,
                          Tensor& lora_in_out, Tensor& lora_out_) {
            size_t R = A.cols;
            for (size_t j = 0; j < d_; ++j) {
                double a = input(t, j);
                double b = last_x_prev_(t, j);
                lora_in_out(t, j) = a + (b - a) * mu(0, j);
            }
            for (size_t j = 0; j < d_; ++j) {
                double a = input(t, j);
                double b = last_x_prev_(t, j);
                double delta = b - a;
                double s = lam(0, j);
                for (size_t r = 0; r < R; ++r) {
                    double dot = 0.0;
                    for (size_t c = 0; c < d_; ++c) {
                        dot += lora_in_out(t, c) * A(c, r);
                    }
                    double th = std::tanh(dot);
                    s += th * B(r, j);
                }
                lora_out_(t, j) = s;
                shift_out(t, j) = a + delta * s;
            }
        };

        ddlerp(A_r, B_r, lambda_r, mu_x, last_r_shift_, last_lora_r_in_, last_lora_r_);
        ddlerp(A_k, B_k, lambda_k, mu_x, last_k_shift_, last_lora_k_in_, last_lora_k_);
        ddlerp(A_v, B_v, lambda_v, mu_x, last_v_shift_, last_lora_v_in_, last_lora_v_);
        ddlerp(A_g, B_g, lambda_g, mu_x, last_g_shift_, last_lora_g_in_, last_lora_g_);
        ddlerp(A_d, B_d, lambda_d, mu_w, last_d_shift_, last_lora_d_in_, last_lora_d_);
        ddlerp(A_x, B_x, lambda_x, mu_x, last_x_shift_, last_lora_x_in_, last_lora_x_);

        for (size_t j = 0; j < d_; ++j) {
            double sr = W_r.bias(0, j);
            double sk = W_k.bias(0, j);
            double sv = W_v.bias(0, j);
            double sg = W_g.bias(0, j);
            for (size_t i = 0; i < d_; ++i) {
                sr += last_r_shift_(t, i) * W_r.weights(j, i);
                sk += last_k_shift_(t, i) * W_k.weights(j, i);
                sv += last_v_shift_(t, i) * W_v.weights(j, i);
                sg += last_g_shift_(t, i) * W_g.weights(j, i);
            }
            last_r_(t, j) = sr;
            last_k_(t, j) = sk;
            last_v_(t, j) = sv;
            last_g_(t, j) = sg;
        }

        for (size_t j = 0; j < d_; ++j) {
            double d_t = last_lora_d_(t, j);
            last_d_pre_(t, j) = d_t;
            last_w_(t, j) = std::exp(-std::exp(d_t));
        }

        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            const double* s_prev = &last_s_(t, h * m * m);
            double* s_curr = &last_s_(t + 1, h * m * m);
            double* kv_curr = &last_kv_(t, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double kv_ij = last_k_(t, base + i) * last_v_(t, base + j);
                    kv_curr[i * m + j] = kv_ij;
                    s_curr[i * m + j] = last_w_(t, base + i) * s_prev[i * m + j] + kv_ij;
                }
            }
            for (size_t j = 0; j < m; ++j) {
                double o = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    double wkv_ij = s_curr[i * m + j] + u(0, base + i) * kv_curr[i * m + j];
                    o += last_r_(t, base + i) * wkv_ij;
                }
                last_rwkv_(t, base + j) = o;
            }
        }
    }

    Tensor output(T, d_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double s = W_o.bias(0, j);
            for (size_t i = 0; i < d_; ++i) {
                s += last_rwkv_(t, i) * W_o.weights(j, i);
            }
            output(t, j) = s;
        }
    }
    return output;
}

// ----------------------------------------------------------------------------
// Backward — hand-derived BPTT
// ----------------------------------------------------------------------------
Tensor RWKV6TimeMix::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("RWKV6TimeMix: grad_output.cols must equal d");
    }
    size_t T = grad_output.rows;
    size_t m = head_dim_;

    Tensor grad_x(T, d_); grad_x.fill(0.0);

    W_r.zero_grad(); W_k.zero_grad(); W_v.zero_grad(); W_g.zero_grad(); W_o.zero_grad();

    grad_u_.fill(0.0);
    grad_mu_x_.fill(0.0); grad_mu_w_.fill(0.0);
    grad_lambda_r_.fill(0.0); grad_lambda_k_.fill(0.0); grad_lambda_v_.fill(0.0);
    grad_lambda_g_.fill(0.0); grad_lambda_d_.fill(0.0); grad_lambda_x_.fill(0.0);
    grad_A_r_.fill(0.0); grad_B_r_.fill(0.0);
    grad_A_k_.fill(0.0); grad_B_k_.fill(0.0);
    grad_A_v_.fill(0.0); grad_B_v_.fill(0.0);
    grad_A_g_.fill(0.0); grad_B_g_.fill(0.0);
    grad_A_d_.fill(0.0); grad_B_d_.fill(0.0);
    grad_A_x_.fill(0.0); grad_B_x_.fill(0.0);

    Tensor grad_r_pre(T, d_), grad_k_pre(T, d_), grad_v_pre(T, d_), grad_g_pre(T, d_);
    grad_r_pre.fill(0.0); grad_k_pre.fill(0.0); grad_v_pre.fill(0.0); grad_g_pre.fill(0.0);
    Tensor grad_w_acc(T, d_); grad_w_acc.fill(0.0);

    Tensor grad_s_prev(num_heads_, m * m);
    grad_s_prev.fill(0.0);

    // BPTT over time, t = T-1 down to 0
    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;

        // ---- Top: out_t = rwkv_t · W_o^T + b_o ----
        Tensor grad_rwkv(1, d_);
        for (size_t i = 0; i < d_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_; ++j) {
                W_o.grad_weights(j, i) += grad_output(t, j) * last_rwkv_(t, i);
                s += grad_output(t, j) * W_o.weights(j, i);
            }
            grad_rwkv(0, i) = s;
        }
        for (size_t j = 0; j < d_; ++j) {
            W_o.grad_bias(0, j) += grad_output(t, j);
        }

        // ---- rwkv_t[h, j] = sum_i r_t[h, i] * wkv_t[h, i, j] ----
        Tensor grad_wkv(num_heads_, m * m); grad_wkv.fill(0.0);
        Tensor grad_r_local(1, d_); grad_r_local.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    grad_wkv(h, i * m + j) = grad_rwkv(0, base + j) * last_r_(t, base + i);
                }
            }
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    double wkv_ij = last_s_(t + 1, h * m * m + i * m + j)
                                  + u(0, base + i) * last_kv_(t, h * m * m + i * m + j);
                    s += grad_rwkv(0, base + j) * wkv_ij;
                }
                grad_r_local(0, base + i) = s;
            }
        }

        // ---- Combined: dL/d_kv[h, i, j] = (grad_wkv + grad_s_prev)[h, i, j] * (1 + u[h, i]) ----
        Tensor grad_kv(num_heads_, m * m); grad_kv.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            const double* kv_t_h = &last_kv_(t, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    // Correct chain:
                    //   dL/d_kv[h,i,j] = dL/d_wkv[h,i,j] * u[h,i]  (wkv = s + u·kv)
                    //                  + dL/d_s_t[h,i,j]            (s_t = w_t·s_{t-1} + kv_t)
                    //   dL/d_s_t[h,i,j] = dL/d_wkv[h,i,j] + grad_s_prev[h,i,j]
                    //   ⇒ dL/d_kv[h,i,j] = grad_wkv[h,i,j] * (1 + u[h,i]) + grad_s_prev[h,i,j]
                    grad_kv(h, i * m + j) = grad_wkv(h, i * m + j) * (1.0 + u(0, base + i))
                                          + grad_s_prev(h, i * m + j);
                }
            }
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    s += grad_wkv(h, i * m + j) * kv_t_h[i * m + j];
                }
                grad_u_(0, base + i) += s;
            }
        }
        // Accumulate dL/d_w_t and update carrier
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            const double* s_prev_h = &last_s_(t, h * m * m);
            for (size_t i = 0; i < m; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    double grad_s_ij = grad_wkv(h, i * m + j) + grad_s_prev(h, i * m + j);
                    s += grad_s_ij * s_prev_h[i * m + j];
                }
                grad_w_acc(t, base + i) += s;
            }
        }
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            for (size_t i = 0; i < m; ++i) {
                double w_i = last_w_(t, base + i);
                for (size_t j = 0; j < m; ++j) {
                    double grad_s_ij = grad_wkv(h, i * m + j) + grad_s_prev(h, i * m + j);
                    grad_s_prev(h, i * m + j) = grad_s_ij * w_i;
                }
            }
        }

        // ---- kv_t[h, i, j] = k_t[h, i] · v_t[h, j] ----
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t base = h * m;
            for (size_t i = 0; i < m; ++i) {
                double sk = 0.0, sv = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    sk += grad_kv(h, i * m + j) * last_v_(t, base + j);
                    sv += grad_kv(h, j * m + i) * last_k_(t, base + j);
                }
                grad_k_pre(t, base + i) += sk;
                grad_v_pre(t, base + i) += sv;
            }
        }

        for (size_t j = 0; j < d_; ++j) {
            grad_r_pre(t, j) += grad_r_local(0, j);
        }
    }

    // ---- d_t = lora_d_value; w_t = exp(-exp(d_t)); chain ----
    Tensor grad_lora_d_pre(T, d_);
    grad_lora_d_pre.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            double d_t = last_d_pre_(t, j);
            double w_t = last_w_(t, j);
            grad_lora_d_pre(t, j) = -std::exp(d_t) * w_t * grad_w_acc(t, j);
        }
    }

    // ---- Per-direction grad_shift_input ----
    Tensor grad_r_shift_in(T, d_), grad_k_shift_in(T, d_), grad_v_shift_in(T, d_), grad_g_shift_in(T, d_);
    grad_r_shift_in.fill(0.0); grad_k_shift_in.fill(0.0);
    grad_v_shift_in.fill(0.0); grad_g_shift_in.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_; ++i) {
            for (size_t j = 0; j < d_; ++j) {
                grad_r_shift_in(t, i) += grad_r_pre(t, j) * W_r.weights(j, i);
                grad_k_shift_in(t, i) += grad_k_pre(t, j) * W_k.weights(j, i);
                grad_v_shift_in(t, i) += grad_v_pre(t, j) * W_v.weights(j, i);
                grad_g_shift_in(t, i) += grad_g_pre(t, j) * W_g.weights(j, i);
            }
        }
    }
    Tensor grad_d_shift_in(T, d_); grad_d_shift_in.fill(0.0);
    // NOTE: for the d-direction the loss depends on `last_lora_d_` (the LoRA
    // output), NOT on `last_d_shift_` (the ddlerp output). The downstream
    // ddlerp_backward for d uses the special `grad_lora_d_pre` interpretation,
    // so we leave grad_d_shift_in == 0 here and pass `grad_lora_d_pre` as the
    // direct gradient w.r.t. the lora output.
    (void)grad_d_shift_in;
    Tensor grad_x_shift_in(T, d_); grad_x_shift_in.fill(0.0);

    // grad_x accumulators
    Tensor grad_x_a(T, d_), grad_x_b(T, d_);
    grad_x_a.fill(0.0); grad_x_b.fill(0.0);

    auto ddlerp_backward = [&](const Tensor& A, const Tensor& B,
                                const Tensor& lora_in_cache, const Tensor& lora_out_cache,
                                const Tensor& grad_shift_input,
                                const Tensor& mu,
                                Tensor& grad_lambda_buf,
                                Tensor& grad_A_buf, Tensor& grad_B_buf,
                                Tensor& grad_mu_buf,
                                Tensor& grad_x_a, Tensor& grad_x_b) {
        // grad_shift_input is the gradient w.r.t. `shift_out` (the ddlerp
        // output). For r/k/v/g/x this is correct (the chain to x_t goes
        // through shift_out). For the d-direction the loss depends on the
        // LoRA output `last_lora_d_` (NOT on shift_out) — see d_lora_backward
        // below for that special case.
        size_t R = A.cols;
        for (size_t t = 0; t < T; ++t) {
            for (size_t j = 0; j < d_; ++j) {
                double delta = last_x_prev_(t, j) - last_input_(t, j);
                double G_shift = grad_shift_input(t, j);
                double lora_out = lora_out_cache(t, j);
                double G_lora_out = delta * G_shift;
                grad_lambda_buf(0, j) += G_lora_out;
                grad_x_a(t, j) += G_shift * (1.0 - lora_out);
                grad_x_b(t, j) += G_shift * lora_out;
            }
            Tensor lora_tanh_one(1, R);
            for (size_t r = 0; r < R; ++r) {
                double dot = 0.0;
                for (size_t c = 0; c < d_; ++c) {
                    dot += lora_in_cache(t, c) * A(c, r);
                }
                lora_tanh_one(0, r) = std::tanh(dot);
            }
            Tensor dlora_tanh(1, R);
            for (size_t r = 0; r < R; ++r) {
                double s = 0.0;
                for (size_t j = 0; j < d_; ++j) {
                    double delta = last_x_prev_(t, j) - last_input_(t, j);
                    s += delta * grad_shift_input(t, j) * B(r, j);
                }
                dlora_tanh(0, r) = s;
            }
            for (size_t r = 0; r < R; ++r) {
                double one_m_tanh_sq = 1.0 - lora_tanh_one(0, r) * lora_tanh_one(0, r);
                for (size_t c = 0; c < d_; ++c) {
                    grad_A_buf(c, r) += dlora_tanh(0, r) * one_m_tanh_sq * lora_in_cache(t, c);
                }
            }
            for (size_t j = 0; j < d_; ++j) {
                double delta = last_x_prev_(t, j) - last_input_(t, j);
                double G_lora_out_j = delta * grad_shift_input(t, j);
                for (size_t r = 0; r < R; ++r) {
                    grad_B_buf(r, j) += G_lora_out_j * lora_tanh_one(0, r);
                }
            }
            for (size_t c = 0; c < d_; ++c) {
                double s = 0.0;
                for (size_t r = 0; r < R; ++r) {
                    double one_m_tanh_sq = 1.0 - lora_tanh_one(0, r) * lora_tanh_one(0, r);
                    s += dlora_tanh(0, r) * A(c, r) * one_m_tanh_sq;
                }
                double delta_c = last_x_prev_(t, c) - last_input_(t, c);
                grad_x_a(t, c) += s * (1.0 - mu(0, c));
                grad_x_b(t, c) += s * mu(0, c);
                grad_mu_buf(0, c) += s * delta_c;
            }
        }
    };

    // d-direction: loss depends on the LoRA output last_lora_d_ directly,
    // NOT on the ddlerp shift output last_d_shift_. The backward chain is:
    //   grad_lora_out_j   = grad_lora_d_pre[t, j]    (the d_loss / d d_t signal)
    //   grad_lambda_d_j  += grad_lora_out_j          (∂lora_out_j/∂lambda_j = 1)
    //   grad_B_d(r, j)   += grad_lora_out_j * tanh_r (∂lora_out_j/∂B[r,j] = tanh_r)
    //   grad_A_d(c, r)   += (sum_j grad_lora_out_j * B[r,j]) * (1-tanh²) * lora_in[c]
    //   grad_lora_in_c    = sum_j grad_lora_out_j * sum_r A[c,r]*(1-tanh²)*B[r,j]
    //   grad_x_a(t, c)   += grad_lora_in_c * (1 - mu_w[c])
    //   grad_x_b(t, c)   += grad_lora_in_c * mu_w[c]
    //   grad_mu_w[c]     += grad_lora_in_c * (x_prev - x_t)
    // NO `delta` multiplier on the per-j terms (unlike the ddlerp case,
    // because last_lora_d_ is the lora output, not the shift output).
    auto d_lora_backward = [&](const Tensor& A, const Tensor& B,
                               const Tensor& lora_in_cache,
                               const Tensor& grad_lora_out_in,  // (T, d)
                               const Tensor& mu,
                               Tensor& grad_lambda_buf,
                               Tensor& grad_A_buf, Tensor& grad_B_buf,
                               Tensor& grad_mu_buf,
                               Tensor& grad_x_a, Tensor& grad_x_b) {
        size_t R = A.cols;
        for (size_t t = 0; t < T; ++t) {
            for (size_t j = 0; j < d_; ++j) {
                double G = grad_lora_out_in(t, j);
                grad_lambda_buf(0, j) += G;
            }
            Tensor lora_tanh_one(1, R);
            for (size_t r = 0; r < R; ++r) {
                double dot = 0.0;
                for (size_t c = 0; c < d_; ++c) {
                    dot += lora_in_cache(t, c) * A(c, r);
                }
                lora_tanh_one(0, r) = std::tanh(dot);
            }
            // dlora_tanh[r] = sum_j grad_lora_out_j * B[r, j]   (no delta factor)
            Tensor dlora_tanh(1, R);
            for (size_t r = 0; r < R; ++r) {
                double s = 0.0;
                for (size_t j = 0; j < d_; ++j) {
                    s += grad_lora_out_in(t, j) * B(r, j);
                }
                dlora_tanh(0, r) = s;
            }
            for (size_t r = 0; r < R; ++r) {
                double one_m_tanh_sq = 1.0 - lora_tanh_one(0, r) * lora_tanh_one(0, r);
                for (size_t c = 0; c < d_; ++c) {
                    grad_A_buf(c, r) += dlora_tanh(0, r) * one_m_tanh_sq * lora_in_cache(t, c);
                }
            }
            for (size_t j = 0; j < d_; ++j) {
                for (size_t r = 0; r < R; ++r) {
                    grad_B_buf(r, j) += grad_lora_out_in(t, j) * lora_tanh_one(0, r);
                }
            }
            // grad_lora_in_c = sum_r dlora_tanh[r] * A[c,r] * (1-tanh²)
            for (size_t c = 0; c < d_; ++c) {
                double s = 0.0;
                for (size_t r = 0; r < R; ++r) {
                    double one_m_tanh_sq = 1.0 - lora_tanh_one(0, r) * lora_tanh_one(0, r);
                    s += dlora_tanh(0, r) * A(c, r) * one_m_tanh_sq;
                }
                double delta_c = last_x_prev_(t, c) - last_input_(t, c);
                grad_x_a(t, c) += s * (1.0 - mu(0, c));
                grad_x_b(t, c) += s * mu(0, c);
                grad_mu_buf(0, c) += s * delta_c;
            }
        }
    };

    ddlerp_backward(A_r, B_r, last_lora_r_in_, last_lora_r_, grad_r_shift_in, mu_x,
                    grad_lambda_r_, grad_A_r_, grad_B_r_, grad_mu_x_, grad_x_a, grad_x_b);
    ddlerp_backward(A_k, B_k, last_lora_k_in_, last_lora_k_, grad_k_shift_in, mu_x,
                    grad_lambda_k_, grad_A_k_, grad_B_k_, grad_mu_x_, grad_x_a, grad_x_b);
    ddlerp_backward(A_v, B_v, last_lora_v_in_, last_lora_v_, grad_v_shift_in, mu_x,
                    grad_lambda_v_, grad_A_v_, grad_B_v_, grad_mu_x_, grad_x_a, grad_x_b);
    ddlerp_backward(A_g, B_g, last_lora_g_in_, last_lora_g_, grad_g_shift_in, mu_x,
                    grad_lambda_g_, grad_A_g_, grad_B_g_, grad_mu_x_, grad_x_a, grad_x_b);
    // (d-direction is handled separately below — see d_lora_backward)
    ddlerp_backward(A_x, B_x, last_lora_x_in_, last_lora_x_, grad_x_shift_in, mu_x,
                    grad_lambda_x_, grad_A_x_, grad_B_x_, grad_mu_x_, grad_x_a, grad_x_b);
    // d-direction: loss depends on last_lora_d_ directly, not on last_d_shift_
    d_lora_backward(A_d, B_d, last_lora_d_in_, grad_lora_d_pre, mu_w,
                    grad_lambda_d_, grad_A_d_, grad_B_d_, grad_mu_w_, grad_x_a, grad_x_b);

    // Combine grad_x_a (x_t) and grad_x_b (x_{t-1} → x at step t-1)
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            grad_x(t, j) += grad_x_a(t, j);
            if (t > 0) {
                grad_x(t - 1, j) += grad_x_b(t, j);
            }
        }
    }

    // ---- W_r, W_k, W_v, W_g parameter gradients ----
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_; ++j) {
            for (size_t i = 0; i < d_; ++i) {
                W_r.grad_weights(j, i) += grad_r_pre(t, j) * last_r_shift_(t, i);
                W_k.grad_weights(j, i) += grad_k_pre(t, j) * last_k_shift_(t, i);
                W_v.grad_weights(j, i) += grad_v_pre(t, j) * last_v_shift_(t, i);
                W_g.grad_weights(j, i) += grad_g_pre(t, j) * last_g_shift_(t, i);
            }
            W_r.grad_bias(0, j) += grad_r_pre(t, j);
            W_k.grad_bias(0, j) += grad_k_pre(t, j);
            W_v.grad_bias(0, j) += grad_v_pre(t, j);
            W_g.grad_bias(0, j) += grad_g_pre(t, j);
        }
    }

    return grad_x;
}

// ----------------------------------------------------------------------------
// update_weights / zero_grad / parameters / gradients
// ----------------------------------------------------------------------------
void RWKV6TimeMix::update_weights(double learning_rate) {
    W_r.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_g.update_weights(learning_rate);
    W_o.update_weights(learning_rate);

    auto sgd = [&](Tensor& p, const Tensor& g) {
        for (size_t i = 0; i < p.rows; ++i)
            for (size_t j = 0; j < p.cols; ++j)
                p(i, j) -= learning_rate * g(i, j);
    };
    sgd(u, grad_u_);
    sgd(mu_x, grad_mu_x_);
    sgd(mu_w, grad_mu_w_);
    sgd(lambda_r, grad_lambda_r_);
    sgd(lambda_k, grad_lambda_k_);
    sgd(lambda_v, grad_lambda_v_);
    sgd(lambda_g, grad_lambda_g_);
    sgd(lambda_d, grad_lambda_d_);
    sgd(lambda_x, grad_lambda_x_);
    sgd(A_r, grad_A_r_); sgd(B_r, grad_B_r_);
    sgd(A_k, grad_A_k_); sgd(B_k, grad_B_k_);
    sgd(A_v, grad_A_v_); sgd(B_v, grad_B_v_);
    sgd(A_g, grad_A_g_); sgd(B_g, grad_B_g_);
    sgd(A_d, grad_A_d_); sgd(B_d, grad_B_d_);
    sgd(A_x, grad_A_x_); sgd(B_x, grad_B_x_);
}

void RWKV6TimeMix::zero_grad() {
    W_r.zero_grad(); W_k.zero_grad(); W_v.zero_grad(); W_g.zero_grad(); W_o.zero_grad();
    grad_u_.fill(0.0);
    grad_mu_x_.fill(0.0); grad_mu_w_.fill(0.0);
    grad_lambda_r_.fill(0.0); grad_lambda_k_.fill(0.0); grad_lambda_v_.fill(0.0);
    grad_lambda_g_.fill(0.0); grad_lambda_d_.fill(0.0); grad_lambda_x_.fill(0.0);
    grad_A_r_.fill(0.0); grad_B_r_.fill(0.0);
    grad_A_k_.fill(0.0); grad_B_k_.fill(0.0);
    grad_A_v_.fill(0.0); grad_B_v_.fill(0.0);
    grad_A_g_.fill(0.0); grad_B_g_.fill(0.0);
    grad_A_d_.fill(0.0); grad_B_d_.fill(0.0);
    grad_A_x_.fill(0.0); grad_B_x_.fill(0.0);
}

std::vector<Tensor*> RWKV6TimeMix::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_r.weights); p.push_back(&W_r.bias);
    p.push_back(&W_k.weights); p.push_back(&W_k.bias);
    p.push_back(&W_v.weights); p.push_back(&W_v.bias);
    p.push_back(&W_g.weights); p.push_back(&W_g.bias);
    p.push_back(&W_o.weights); p.push_back(&W_o.bias);
    auto append = [&](Tensor& t) { p.push_back(&t); };
    append(u); append(mu_x); append(mu_w);
    append(lambda_r); append(lambda_k); append(lambda_v);
    append(lambda_g); append(lambda_d); append(lambda_x);
    append(A_r); append(B_r); append(A_k); append(B_k);
    append(A_v); append(B_v); append(A_g); append(B_g);
    append(A_d); append(B_d); append(A_x); append(B_x);
    return p;
}

std::vector<Tensor*> RWKV6TimeMix::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&W_r.grad_weights); g.push_back(&W_r.grad_bias);
    g.push_back(&W_k.grad_weights); g.push_back(&W_k.grad_bias);
    g.push_back(&W_v.grad_weights); g.push_back(&W_v.grad_bias);
    g.push_back(&W_g.grad_weights); g.push_back(&W_g.grad_bias);
    g.push_back(&W_o.grad_weights); g.push_back(&W_o.grad_bias);
    g.push_back(&grad_u_);
    g.push_back(&grad_mu_x_); g.push_back(&grad_mu_w_);
    g.push_back(&grad_lambda_r_); g.push_back(&grad_lambda_k_); g.push_back(&grad_lambda_v_);
    g.push_back(&grad_lambda_g_); g.push_back(&grad_lambda_d_); g.push_back(&grad_lambda_x_);
    g.push_back(&grad_A_r_); g.push_back(&grad_B_r_);
    g.push_back(&grad_A_k_); g.push_back(&grad_B_k_);
    g.push_back(&grad_A_v_); g.push_back(&grad_B_v_);
    g.push_back(&grad_A_g_); g.push_back(&grad_B_g_);
    g.push_back(&grad_A_d_); g.push_back(&grad_B_d_);
    g.push_back(&grad_A_x_); g.push_back(&grad_B_x_);
    return g;
}