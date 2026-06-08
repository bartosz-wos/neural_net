#include "mamba2.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Mamba2Block implementation
// ----------------------------------------------------------------------------
//
// This is the "Mamba-2 single-step SSD recurrence" — the simplest form of
// structured state-space duality. It is a multi-head linear-attention-like
// recurrence with a *scalar* per-head exponential decay:
//
//   H_t = diag(a_t) @ H_{t-1} + outer(b_t, k_t)     per head
//   o_t = q_t @ H_t                                  per head
//
// The scalar-decay structure is what makes the recurrence
// *equivalent* to linear attention with a structured causal mask
// M_{t,s} = a_t * a_{t-1} * ... * a_{s+1} for s<t, 0 otherwise.
//
// This implementation follows the same style as MambaBlock — a Python-style
// recurrence loop that caches all intermediate states for analytical
// backward. We use d_inner = 2 * d_model by default (matching Mamba-1).
// ----------------------------------------------------------------------------

static inline double m2_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}
static inline double m2_silu(double x) {
    return x * m2_sigmoid(x);
}

// ---------- constructor ----------

Mamba2Block::Mamba2Block(size_t d_model, size_t n_heads, size_t d_inner)
    : in_proj(d_model, 0),         // placeholder, fixed below
      out_proj(0, d_model),        // (d_inner, d_model)
      a_proj(d_model, 0),          // placeholder
      b_proj(d_model, 0),          // placeholder
      k_proj(d_model, 0),          // placeholder
      q_proj(d_model, 0),          // placeholder
      D_skip(1, 0),                // placeholder
      dt_bias(1, 0),               // placeholder
      d_model_(d_model), n_heads_(n_heads)
{
    if (d_inner == 0) d_inner = 2 * d_model;
    if (d_model == 0 || n_heads == 0 || d_inner == 0) {
        throw std::invalid_argument("Mamba2Block: d_model, n_heads, d_inner must be > 0");
    }
    if (d_inner % n_heads != 0) {
        throw std::invalid_argument("Mamba2Block: d_inner must be divisible by n_heads");
    }
    d_inner_ = d_inner;
    head_dim_ = d_inner / n_heads;

    // Reinitialize projections with correct output dims.
    // Dense convention: y = x W^T + b, weights shape (out, in).
    in_proj  = Dense(d_model, 2 * d_inner);
    a_proj   = Dense(d_model, n_heads);
    b_proj   = Dense(d_model, d_inner);
    k_proj   = Dense(d_model, d_inner);
    q_proj   = Dense(d_model, d_inner);
    out_proj = Dense(d_inner, d_model);

    // dt_bias: small init, "soft" open decay. log(decay) = 0 -> decay = 0.5.
    // We initialize to log(0.9) ≈ -0.105 so sigmoid(dt_bias) starts near 0.474,
    // which is the Mamba-2 paper's recommended init for stable early training.
    dt_bias = Tensor(1, n_heads);
    for (size_t h = 0; h < n_heads; ++h) {
        dt_bias(0, h) = std::log(0.9 / (1.0 - 0.9));  // = logit(0.9) ≈ 2.197
    }

    // D_skip: init to 1.0 (Mamba convention).
    D_skip = Tensor(1, d_inner);
    for (size_t i = 0; i < d_inner; ++i) D_skip(0, i) = 1.0;

    // Initialize hidden grad buffers to correct shape.
    grad_D_skip_  = Tensor(1, d_inner);
    grad_dt_bias_ = Tensor(1, n_heads);

    // Zero out Dense biases (default init is small random — we want exactly 0
    // so the projections start pure linear, letting the network learn any
    // offset the data requires).
    in_proj.bias.fill(0.0);
    a_proj.bias.fill(0.0);
    b_proj.bias.fill(0.0);
    k_proj.bias.fill(0.0);
    q_proj.bias.fill(0.0);
    out_proj.bias.fill(0.0);

    // Note: a_proj's bias is the same as dt_bias conceptually, but we keep
    // them separate to make the math clear: dt_bias is the *learnable* per-head
    // scalar offset added before sigmoid. a_proj.bias is the Dense's standard
    // bias term. We zero a_proj.bias (handled above).
}

// ---------- forward ----------

Tensor Mamba2Block::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("Mamba2Block: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("Mamba2Block: input must have at least one token");
    }
    last_input_ = input.clone();

    // Step 1: in_proj — (T, d_model) -> (T, 2*d_inner)
    last_p_ = in_proj.forward(input);

    // Split into ssm path and gate path.
    last_x_ssm_ = Tensor(T, d_inner_);
    last_gate_  = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            last_x_ssm_(t, i) = last_p_(t, i);
            last_gate_(t, i)  = last_p_(t, d_inner_ + i);
        }
    }

    // Step 2: per-head decay and the V/K/Q projections.
    // a_proj output: (T, n_heads). Add dt_bias to each head's pre-sigmoid.
    Tensor a_pre = a_proj.forward(input);  // (T, n_heads)
    last_a_pre_ = Tensor(T, n_heads_);
    last_a_     = Tensor(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double pre = a_pre(t, h) + dt_bias(0, h);
            last_a_pre_(t, h) = pre;
            last_a_(t, h)     = m2_sigmoid(pre);  // decay ∈ (0, 1)
        }
    }

    last_b_ = b_proj.forward(input);  // (T, d_inner)
    last_k_ = k_proj.forward(input);  // (T, d_inner)
    last_q_ = q_proj.forward(input);  // (T, d_inner)

    // Step 3: Selective SSD recurrence.
    // last_H_ shape: ((T+1) * d_inner, head_dim). Time step t stored at rows
    //   [t*d_inner, (t+1)*d_inner). Head h occupies rows [h*head_dim, (h+1)*head_dim).
    last_H_ = Tensor((T + 1) * d_inner_, head_dim_);
    // H_0 = 0
    for (size_t i = 0; i < d_inner_; ++i) {
        for (size_t j = 0; j < head_dim_; ++j) {
            last_H_(0 * d_inner_ + i, j) = 0.0;
        }
    }

    last_o_ = Tensor(T, d_inner_);

    for (size_t t = 0; t < T; ++t) {
        // Update H_t per head:
        //   H_t[h, li, j] = a_t[h] * H_{t-1}[h, li, j] + b_t[h, li] * k_t[h, j]
        // where the head h of b_t is b_t[h*head_dim : (h+1)*head_dim]
        //   (so b_t[h, li] = last_b_(t, h*head_dim + li)),
        // and similarly k_t[h, j] = last_k_(t, h*head_dim + j).
        // The matrix state is stored flattened as last_H_[(t+1)*d_inner + i, j]
        // where i = h*head_dim + li.
        for (size_t h = 0; h < n_heads_; ++h) {
            double a_t_h = last_a_(t, h);
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                size_t row_prev = t * d_inner_ + i;
                size_t row_curr = (t + 1) * d_inner_ + i;
                double b_t_i = last_b_(t, i);
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_t_j = last_k_(t, h * head_dim_ + j);
                    double h_prev = last_H_(row_prev, j);
                    last_H_(row_curr, j) = a_t_h * h_prev + b_t_i * k_t_j;
                }
            }
        }

        // Compute o_t: o_t[i] = sum_j H_t[i, j] * q_t_segment[h, j]  (head h of i)
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                size_t row = (t + 1) * d_inner_ + i;
                double acc = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    acc += last_H_(row, j) * last_q_(t, h * head_dim_ + j);
                }
                last_o_(t, i) = acc;
            }
        }
    }

    // Step 4: skip + gating + output projection.
    // y_t = o_t + D_skip ⊙ x_ssm_t
    // gated_t = silu(g_t) ⊙ y_t
    // out_t = out_proj(gated_t)
    Tensor gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double y = last_o_(t, i) + D_skip(0, i) * last_x_ssm_(t, i);
            double g_act = m2_silu(last_gate_(t, i));
            gated(t, i) = g_act * y;
        }
    }
    last_gated_ = gated;
    Tensor output = out_proj.forward(gated);
    return output;
}

// ---------- backward ----------
//
// We follow the MambaBlock structure: zero out Dense grads, then walk the
// computation graph in reverse order, accumulating local gradients.

Tensor Mamba2Block::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = last_input_.rows;
    if (grad_output.rows != T || grad_output.cols != d_model_) {
        throw std::invalid_argument("Mamba2Block: grad_output shape mismatch");
    }

    in_proj.zero_grad();
    out_proj.zero_grad();
    a_proj.zero_grad();
    b_proj.zero_grad();
    k_proj.zero_grad();
    q_proj.zero_grad();

    // ------ (1) out_proj backward ------
    // out_t = gated_t @ W_out^T + b_out
    // grad_gated[t][i] = sum_j grad_out[t][j] * W_out[j][i]
    Tensor grad_gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                acc += grad_output(t, j) * out_proj.weights(j, i);
            }
            grad_gated(t, i) = acc;
        }
    }
    // grad_W_out[j][i] = sum_t grad_out[t][j] * gated[t][i]
    for (size_t j = 0; j < d_model_; ++j)
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < T; ++t) acc += grad_output(t, j) * last_gated_(t, i);
            out_proj.grad_weights(j, i) += acc;
        }
    for (size_t j = 0; j < d_model_; ++j) {
        double acc = 0.0;
        for (size_t t = 0; t < T; ++t) acc += grad_output(t, j);
        out_proj.grad_bias(0, j) += acc;
    }

    // ------ (2) split grad_gated into grad_gate and grad_y ------
    // gated = silu(g) * y,  y = o + D_skip ⊙ x_ssm
    //   dL/dg = grad_gated * silu'(g) * y
    //   dL/dy = grad_gated * silu(g)
    //   dL/d(D_skip[i]) = sum_t dL/dy[t][i] * x_ssm[t][i]
    //   dL/d(o_t[i])   = dL/dy[t][i]
    //   dL/d(x_ssm_t[i]) = dL/dy[t][i] * D_skip[i]
    Tensor grad_gate(T, d_inner_);
    Tensor grad_o(T, d_inner_);
    Tensor grad_x_ssm(T, d_inner_);
    Tensor grad_D_skip_acc(1, d_inner_);
    grad_D_skip_acc.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double g = last_gate_(t, i);
            double y = last_o_(t, i) + D_skip(0, i) * last_x_ssm_(t, i);
            double silu_g = m2_silu(g);
            double sig_g = m2_sigmoid(g);
            double silu_prime = sig_g * (1.0 + g * (1.0 - sig_g));
            grad_gate(t, i) = grad_gated(t, i) * silu_prime * y;
            double d_y = grad_gated(t, i) * silu_g;
            grad_o(t, i) = d_y;
            grad_x_ssm(t, i) = d_y * D_skip(0, i);
            grad_D_skip_acc(0, i) += d_y * last_x_ssm_(t, i);
        }
    }

    // ------ (3) SSD recurrence backward (BPTT through H_t) ------
    //
    // o_t[i] = sum_j H_t[i, j] * q_t_segment[h][j]  (head h of channel i)
    //
    // dL/dH_t[i, j] (from o_t) = grad_o[t][i] * q_t_segment[h][j]
    // dL/dq_t_segment[h][j]   = sum_i grad_o[t][i] * H_t[i, j]
    //
    // Then propagate g_H backward through the recurrence:
    //   H_t[i, j] = a_t[h] * H_{t-1}[i, j] + b_t[i] * k_t_segment[h][j]
    // So:
    //   dL/d(H_{t-1}[i, j]) += g_H[t][i, j] * a_t[h]
    //   dL/d(a_t[h])        += sum_{i in head h} sum_j g_H[t][i, j] * H_{t-1}[i, j]
    //   dL/d(b_t[i])        += sum_j g_H[t][i, j] * k_t_segment[h][j]
    //   dL/d(k_t_segment[h][j]) = sum_{i in head h} g_H[t][i, j] * b_t[i]

    Tensor g_H((T + 1) * d_inner_, head_dim_);
    g_H.fill(0.0);

    // Direct contribution from o_t at t = T-1, T-2, ..., 0.
    Tensor grad_q_t(T, d_inner_);
    grad_q_t.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                size_t h_row = (t + 1) * d_inner_ + i;  // H_t row in last_H_
                for (size_t j = 0; j < head_dim_; ++j) {
                    g_H(h_row, j) += grad_o(t, i) * last_q_(t, h * head_dim_ + j);
                    grad_q_t(t, h * head_dim_ + j) += grad_o(t, i) * last_H_(h_row, j);
                }
            }
        }
    }

    // BPTT: walk t from T-1 down to 0, propagating g_H to the previous step
    // and accumulating gradients w.r.t. a_t, b_t, k_t.
    Tensor grad_a(T, n_heads_);   grad_a.fill(0.0);
    Tensor grad_b(T, d_inner_);   grad_b.fill(0.0);
    Tensor grad_k(T, d_inner_);   grad_k.fill(0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        for (size_t h = 0; h < n_heads_; ++h) {
            double a_t_h = last_a_(t, h);
            for (size_t li = 0; li < head_dim_; ++li) {
                size_t i = h * head_dim_ + li;
                size_t row_curr = (t + 1) * d_inner_ + i;
                size_t row_prev = t * d_inner_ + i;
                double b_t_i = last_b_(t, i);
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_t_j = last_k_(t, h * head_dim_ + j);
                    double gh = g_H(row_curr, j);
                    // dL/d(H_{t-1}[i, j]) += g_H[t][i, j] * a_t
                    g_H(row_prev, j) += gh * a_t_h;
                    // dL/d(a_t[h]) += gh * H_{t-1}[i, j]
                    grad_a(t, h) += gh * last_H_(row_prev, j);
                    // dL/d(b_t[i]) += gh * k_t_segment[h][j]
                    grad_b(t, i) += gh * k_t_j;
                    // dL/d(k_t_segment[h][j]) += gh * b_t[i]
                    grad_k(t, h * head_dim_ + j) += gh * b_t_i;
                }
            }
        }
    }

    // ------ (4) chain rule: a_t = sigmoid(a_pre_t + dt_bias[h]) ------
    //   dL/d(a_pre_t[h]) = grad_a[t][h] * sigmoid'(a_pre_t[h]) = grad_a[t][h] * a_t[h] * (1 - a_t[h])
    //   dL/d(dt_bias[h]) = sum_t grad_a[t][h] * a_t[h] * (1 - a_t[h])
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

    // ------ (5) backprop through 5 projections (Dense layers) ------
    // Each: y = x @ W^T + b.  W is (out, in), x is (T, in), y is (T, out).
    //   dL/dW[i, k] = sum_t dL/dy[t][i] * x[t][k]
    //   dL/db[i]   = sum_t dL/dy[t][i]
    //   dL/dx[t][k] = sum_i dL/dy[t][i] * W[i, k]

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

    // k_proj: (T, d_inner) from (T, d_model). grad_y = grad_k.
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

    // b_proj: (T, d_inner) from (T, d_model). grad_y = grad_b.
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

    // a_proj: (T, n_heads) from (T, d_model). grad_y = grad_a_pre.
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

    // in_proj: (T, 2*d_inner) from (T, d_model).
    // grad_in_proj[t][i] = grad_x_ssm[t][i]      for i in [0, d_inner)
    //                   = grad_gate[t][i-d_inner] for i in [d_inner, 2*d_inner)
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

    // ------ (6) input gradient: sum contributions from all 5 projections ------
    Tensor grad_input(T, d_model_);
    grad_input.fill(0.0);

    // From in_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < 2 * d_inner_; ++i) {
                acc += grad_p(t, i) * in_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From a_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < n_heads_; ++i) {
                acc += grad_a_pre(t, i) * a_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From b_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                acc += grad_b(t, i) * b_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From k_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                acc += grad_k(t, i) * k_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From q_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                acc += grad_q_t(t, i) * q_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }

    // ------ (7) stash non-Dense grads (D_skip, dt_bias) ------
    if (grad_D_skip_.rows != 1 || grad_D_skip_.cols != d_inner_) {
        grad_D_skip_  = Tensor(1, d_inner_);
        grad_dt_bias_ = Tensor(1, n_heads_);
        grad_D_skip_.fill(0.0);
        grad_dt_bias_.fill(0.0);
    }
    for (size_t i = 0; i < d_inner_; ++i) {
        grad_D_skip_(0, i) = grad_D_skip_acc(0, i);
    }
    for (size_t h = 0; h < n_heads_; ++h) {
        grad_dt_bias_(0, h) = grad_dt_bias_acc(0, h);
    }

    return grad_input;
}

// ---------- update_weights ----------

void Mamba2Block::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    a_proj.update_weights(learning_rate);
    b_proj.update_weights(learning_rate);
    k_proj.update_weights(learning_rate);
    q_proj.update_weights(learning_rate);
    if (grad_D_skip_.cols == d_inner_) {
        for (size_t i = 0; i < d_inner_; ++i) {
            D_skip(0, i) -= learning_rate * grad_D_skip_(0, i);
        }
    }
    if (grad_dt_bias_.cols == n_heads_) {
        for (size_t h = 0; h < n_heads_; ++h) {
            dt_bias(0, h) -= learning_rate * grad_dt_bias_(0, h);
        }
    }
}

// ---------- zero_grad ----------

void Mamba2Block::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    a_proj.zero_grad();
    b_proj.zero_grad();
    k_proj.zero_grad();
    q_proj.zero_grad();
    grad_D_skip_  = Tensor(1, d_inner_);
    grad_dt_bias_ = Tensor(1, n_heads_);
}

// ---------- parameters / gradients ----------

std::vector<Tensor*> Mamba2Block::parameters() {
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
    p.push_back(&D_skip);
    p.push_back(&dt_bias);
    return p;
}

std::vector<Tensor*> Mamba2Block::gradients() {
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
    g.push_back(&grad_D_skip_);
    g.push_back(&grad_dt_bias_);
    return g;
}
