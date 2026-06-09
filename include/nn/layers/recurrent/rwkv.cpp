#include "rwkv.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// RWKVTimeMix implementation
// ----------------------------------------------------------------------------
//
// Forward per-step (all vector ops are element-wise; i indexes channels):
//   r_in_t = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_{t-1}   (x_{-1} := 0)
//   k_in_t = μ_k ⊙ x_t + (1 - μ_k) ⊙ x_{t-1}
//   v_in_t = μ_v ⊙ x_t + (1 - μ_v) ⊙ x_{t-1}
//
//   r_pre_t = W_r · r_in_t + b_r
//   k_pre_t = W_k · k_in_t + b_k
//   v_pre_t = W_v · v_in_t + b_v
//
//   a[i]   = exp(-exp(log_w[i]))           in (0, 1)
//   b[i]   = exp(u[i]) - 1                  (signed, can be negative or positive)
//
//   p_t[i] = a[i] · p_{t-1}[i] + k_pre_t · v_pre_t[i]
//   wkv_t[i] = p_t[i] + b[i] · (k_pre_t · v_pre_t[i])
//   o_t = sigmoid(r_pre_t) ⊙ wkv_t
//
// Backward per-step (recurrence + output):
//   Let g_t = dL/do_t,  r_sig_t = sigmoid(r_pre_t).
//
//   dL/d(wkv_t)    = g_t ⊙ r_sig_t
//   dL/d(sig_t)    = g_t ⊙ wkv_t
//   dL/d(r_pre_t)  = dL/d(sig_t) ⊙ sig_t ⊙ (1 - sig_t)
//
//   Let kv_t[i] = k_pre_t · v_pre_t[i].
//   wkv_t[i] = p_t[i] + b[i] · kv_t[i].
//     dL/d(p_t[i])    = dL/d(wkv_t[i])                            (per channel)
//     dL/d(b[i])      = dL/d(wkv_t[i]) · kv_t[i]                 (bonus path)
//     dL/d(kv_t[i])   = dL/d(wkv_t[i]) · b[i]                    (bonus path)
//
//   p_t[i] = a[i] · p_{t-1}[i] + kv_t[i]:
//     dL/d(p_{t-1}[i]) += dL/d(p_t[i]) · a[i]                    (recurrence carrier)
//     dL/d(a[i])       += dL/d(p_t[i]) · p_{t-1}[i]              (per channel)
//     dL/d(kv_t[i])   += dL/d(p_t[i])                            (additive contribution)
//
//   So dL/d(kv_t[i]) = dL/d(wkv_t[i]) · b[i] + dL/d(p_t[i]).
//   kv_t[i] = k_pre_t · v_pre_t[i]:
//     dL/d(k_pre_t)  += sum_i  dL/d(kv_t[i]) · v_pre_t[i]
//     dL/d(v_pre_t[i]) += dL/d(kv_t[i]) · k_pre_t
//   Note: dL/d(k_pre_t) sums over i; dL/d(v_pre_t[i]) is per channel.
//
//   a[i] = exp(-exp(log_w[i])):
//     dL/d(log_w[i]) = dL/d(a[i]) · a[i] · (-exp(log_w[i]))     (chain: a = exp(-z), z = exp(log_w), dz/dlog_w = z = exp(log_w))
//   b[i] = exp(u[i]) - 1:
//     dL/d(u[i]) = dL/d(b[i]) · exp(u[i])
//
//   dL/d(r_pre_t), dL/d(k_pre_t), dL/d(v_pre_t) flow through Dense.backward
//   to give dL/d(r_in_t), dL/d(k_in_t), dL/d(v_in_t).
//
//   r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}:
//     dL/d(x_t)      += dL/d(r_in_t) ⊙ μ_r + dL/d(k_in_t) ⊙ μ_k + dL/d(v_in_t) ⊙ μ_v
//     dL/d(x_{t-1})  += dL/d(r_in_t) ⊙ (1-μ_r) + ... (1-μ_k) + ... (1-μ_v)  (passed to next step's grad)
//     dL/d(μ_r)      += dL/d(r_in_t) ⊙ (x_t - x_{t-1})   (and same for μ_k, μ_v)
// ============================================================================

static inline double rwkv_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
RWKVTimeMix::RWKVTimeMix(size_t d)
    : d_(d),
      W_r(d, d), W_k(d, d), W_v(d, d),
      log_w(1, d), u(1, d),
      mu_r(1, d), mu_k(1, d), mu_v(1, d),
      grad_log_w_(1, d), grad_u_(1, d),
      grad_mu_r_(1, d), grad_mu_k_(1, d), grad_mu_v_(1, d)
{
    if (d == 0) {
        throw std::invalid_argument("RWKVTimeMix: d must be > 0");
    }

    // Dense initializers (xavier) — fine for W_r, W_k, W_v.
    // biases default to 0 in Dense::init_weights.

    // log_w: -5.0 init → w = -exp(-5) ≈ -0.0067, a = exp(w) ≈ 0.993 (slow decay).
    log_w.fill(-5.0);

    // u: 0.0 init → bonus = exp(0) - 1 = 0 (no j=t boost initially).
    u.fill(0.0);

    // mu_r, mu_k, mu_v: 0.5 init (50/50 mix of current and previous).
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

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RWKVTimeMix::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("RWKVTimeMix: input.cols must equal d");
    }
    size_t T = input.rows;
    last_input_ = input.clone();  // preserve the input

    // Allocate cache
    last_x_shift_ = Tensor(T, d_); last_x_shift_.fill(0.0);
    last_r_in_    = Tensor(T, d_); last_k_in_    = Tensor(T, d_); last_v_in_    = Tensor(T, d_);
    last_r_pre_   = Tensor(T, d_); last_k_pre_   = Tensor(T, d_); last_v_pre_   = Tensor(T, d_);
    last_r_sig_   = Tensor(T, d_);
    last_p_       = Tensor(T + 1, d_); last_p_.fill(0.0);
    last_a_       = Tensor(1, d_);
    last_bonus_   = Tensor(1, d_);
    last_wkv_     = Tensor(T, d_);
    last_kv_      = Tensor(T, d_);

    // Compute a[i] = exp(-exp(log_w[i])) and b[i] = exp(u[i]) - 1 once (constants per forward).
    for (size_t j = 0; j < d_; ++j) {
        last_a_(0, j)     = std::exp(-std::exp(log_w(0, j)));
        last_bonus_(0, j) = std::exp(u(0, j)) - 1.0;
    }

    // Step 1: token-shift mixing
    //   r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}   (x_{-1} := 0)
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

    // Step 2: projections
    last_r_pre_ = W_r.forward(last_r_in_);
    last_k_pre_ = W_k.forward(last_k_in_);
    last_v_pre_ = W_v.forward(last_v_in_);

    // Steps 3+4: WKV recurrence and gating.
    //
    // RWKV-4 single-block time-mix math (per channel i):
    //   p_t[i]   = a[i] · p_{t-1}[i] + k_pre_t[i] · v_pre_t[i]
    //   wkv_t[i] = p_t[i] + b[i] · (k_pre_t[i] · v_pre_t[i])    with b[i] = exp(u[i]) - 1
    //   o_t[i]   = sigmoid(r_pre_t[i]) · wkv_t[i]
    //
    // This is the element-wise per-channel form (k and v are both d-dim; the "key·value"
    // in the recurrence is the element-wise product at each channel). This is the
    // canonical RWKV-4 time-mix math, NOT a 2D outer-product state.
    Tensor output(T, d_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_; ++i) {
            double kv_i = last_k_pre_(t, i) * last_v_pre_(t, i);
            last_kv_(t, i) = kv_i;
            double p_t_i = last_a_(0, i) * last_p_(t, i) + kv_i;
            last_p_(t + 1, i) = p_t_i;
            double wkv_t_i = p_t_i + last_bonus_(0, i) * kv_i;
            last_wkv_(t, i) = wkv_t_i;
            double sig = rwkv_sigmoid(last_r_pre_(t, i));
            last_r_sig_(t, i) = sig;
            output(t, i) = sig * wkv_t_i;
        }
    }
    return output;
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
Tensor RWKVTimeMix::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("RWKVTimeMix: grad_output.cols must equal d");
    }
    size_t T = grad_output.rows;

    // Allocate per-step gradient buffers
    Tensor grad_x(T, d_);          // returned to caller (sum of x_t contributions)
    grad_x.fill(0.0);

    // Backprop through projections: we use Dense::backward for W_r, W_k, W_v and biases.
    // We need dL/d(r_in), dL/d(k_in), dL/d(v_in) to pass to Dense::backward.
    // We'll accumulate them, then call backward to get the param grads and the input grads
    // (which are dL/d(r_in) etc.) — we then add those to grad_x with the μ-mix.
    //
    // But Dense::backward writes to grad_weights/grad_bias via +=, and returns dL/d(last_input).
    // We zero the Dense grads before the call so we get exact gradients from this backward pass.
    // (The Dense accumulator is +=, so we want to call zero_grad on each Dense and let it accumulate.)

    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();

    Tensor grad_r_pre(T, d_); grad_r_pre.fill(0.0);
    Tensor grad_k_pre(T, d_); grad_k_pre.fill(0.0);
    Tensor grad_v_pre(T, d_); grad_v_pre.fill(0.0);

    // BPTT through the WKV recurrence: need a per-step p gradient that flows backward.
    Tensor grad_p_next(1, d_); grad_p_next.fill(0.0);  // dL/d(p_t[i]) accumulated from t+1's recurrence

    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;  // backward: t = T-1, T-2, ..., 0

        // dL/d(wkv_t) = g_t ⊙ r_sig_t
        // dL/d(sig)   = g_t ⊙ wkv_t
        // dL/d(r_pre) = dL/d(sig) ⊙ sig ⊙ (1 - sig)
        for (size_t i = 0; i < d_; ++i) {
            double g_o = grad_output(t, i);
            double sig = last_r_sig_(t, i);
            double wkv = last_wkv_(t, i);
            double d_sig = g_o * wkv;
            double d_r_pre = d_sig * sig * (1.0 - sig);
            grad_r_pre(t, i) = d_r_pre;
        }

        // dL/d(p_t[i]) comes from TWO sources:
        //   (1) dL/d(wkv_t[i])  (direct: wkv_t = p_t + bonus * kv)
        //   (2) grad_p_next[i]  (recurrence carrier: p_{t+1}[i] = a[i] * p_t[i] + kv_{t+1}[i])
        //   grad_p_next already accumulated dL/d(p_t) from step t+1's BPTT loop.
        // dL/d(bonus[i]) = dL/d(wkv_t[i]) * kv_t[i]
        // dL/d(kv_t[i])  = dL/d(wkv_t[i]) * bonus[i] + dL/d(p_t[i])   (both paths feed kv)
        // dL/d(a[i])    += dL/d(p_{t+1}[i]) * p_t[i]    (from t+1's recurrence; but we want dL/d(a)
        //                  for this step which feeds p_{t+1}, NOT p_t. Hmm. Let me re-examine.)
        //
        // Correction on grad_a: a[i] contributes to p_t[i] = a[i] * p_{t-1}[i] + kv_t[i].
        //   So dL/d(a[i]) += dL/d(p_t[i]) * p_{t-1}[i].
        // We use last_p_(t, i) as p_{t-1}[i] (because last_p_ has row t = state at end of step t-1).
        // Wait: last_p_(t+1, i) = p_t[i] (state after step t). So p_{t-1}[i] = last_p_(t, i).

        Tensor grad_p_curr(1, d_); grad_p_curr.fill(0.0);  // dL/d(p_t[i]) for the current step
        for (size_t i = 0; i < d_; ++i) {
            double d_wkv = grad_output(t, i) * last_r_sig_(t, i);
            double d_p = d_wkv + grad_p_next(0, i);  // dL/d(p_t[i]) — direct + recurrence
            grad_p_curr(0, i) = d_p;

            double kv_i = last_kv_(t, i);
            double d_bonus = d_wkv * kv_i;
            grad_u_(0, i) += d_bonus * std::exp(u(0, i));  // dL/d(u[i]) = dL/d(bonus) * exp(u[i])

            double d_a = d_p * last_p_(t, i);  // dL/d(a[i]) += dL/d(p_t[i]) * p_{t-1}[i]
            // dL/d(log_w[i]) = dL/d(a[i]) * a[i] * (-exp(log_w[i]))
            grad_log_w_(0, i) += d_a * last_a_(0, i) * (-std::exp(log_w(0, i)));

            double d_kv_i = d_wkv * last_bonus_(0, i) + d_p;
            // kv_i = k_pre_t[i] * v_pre_t[i]
            grad_k_pre(t, i) += d_kv_i * last_v_pre_(t, i);
            grad_v_pre(t, i) += d_kv_i * last_k_pre_(t, i);
        }

        // grad_p_next for the next iteration (t-1): dL/d(p_{t-1}[i]) from this step's recurrence.
        // p_t[i] = a[i] * p_{t-1}[i] + kv_t[i]
        //   dL/d(p_{t-1}[i]) += dL/d(p_t[i]) * a[i]
        for (size_t i = 0; i < d_; ++i) {
            grad_p_next(0, i) = grad_p_curr(0, i) * last_a_(0, i);
        }
    }

    // Backprop through Dense projections to get dL/d(r_in), dL/d(k_in), dL/d(v_in) AND
    // accumulate grad_W_r, grad_b_r, etc.
    // Dense::backward returns dL/d(input). We feed that back to μ-mix.
    Tensor grad_r_in = W_r.backward(grad_r_pre, 0.0);
    Tensor grad_k_in = W_k.backward(grad_k_pre, 0.0);
    Tensor grad_v_in = W_v.backward(grad_v_pre, 0.0);

    // μ-mix backward. r_in_t = μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1}.
    // For each t, contribution to grad_x[t] from r_in_t = μ_r ⊙ grad_r_in_t
    // and contribution to grad_x[t-1] from r_in_t = (1-μ_r) ⊙ grad_r_in_t.
    // We also accumulate grad_mu_r += grad_r_in_t ⊙ (x_t - x_{t-1}).
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
void RWKVTimeMix::update_weights(double learning_rate) {
    W_r.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);

    // log_w, u, mu_r, mu_k, mu_v
    for (size_t j = 0; j < d_; ++j) {
        log_w(0, j) -= learning_rate * grad_log_w_(0, j);
        u(0, j)     -= learning_rate * grad_u_(0, j);
        mu_r(0, j)  -= learning_rate * grad_mu_r_(0, j);
        mu_k(0, j)  -= learning_rate * grad_mu_k_(0, j);
        mu_v(0, j)  -= learning_rate * grad_mu_v_(0, j);
    }
}

void RWKVTimeMix::zero_grad() {
    W_r.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    grad_log_w_.fill(0.0);
    grad_u_.fill(0.0);
    grad_mu_r_.fill(0.0);
    grad_mu_k_.fill(0.0);
    grad_mu_v_.fill(0.0);
}

std::vector<Tensor*> RWKVTimeMix::parameters() {
    return {
        &W_r.weights, &W_r.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &log_w, &u, &mu_r, &mu_k, &mu_v
    };
}

std::vector<Tensor*> RWKVTimeMix::gradients() {
    return {
        &W_r.grad_weights, &W_r.grad_bias,
        &W_k.grad_weights, &W_k.grad_bias,
        &W_v.grad_weights, &W_v.grad_bias,
        &grad_log_w_, &grad_u_, &grad_mu_r_, &grad_mu_k_, &grad_mu_v_
    };
}
