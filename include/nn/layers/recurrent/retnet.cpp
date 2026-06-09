#include "retnet.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// RetNetRetention implementation (recurrent, single-step form)
//
// See retnet.h for the full math derivation. Quick recap of the per-step
// forward (one head, one step):
//
//   q_t, k_t, v_t in R^{d} (pre-rotation, unrotated, from Dense projections)
//   gamma[i] = sigmoid(gamma_raw[i])  in (0, 1)  per-channel
//
//   xPos rotation: pair (i, i+D/2) of head_dim D gets rotated by angle
//     theta_i(m) = m * base^(-2i/(D/2))    (i in [0, D/2))
//   Q is rotated by +theta, K is rotated by -theta (forward/inverse).
//   Result stored as last_q_rot_(t, h*D + ...) and last_k_rot_(t, ...).
//
//   Per-head S_t^(h) in R^{D × D}:
//     S_t = diag(gamma_h) · S_{t-1} + outer(k_t^h, v_t^h)
//     o_t^h = S_t · q_t^h
//
//   Concat head outputs to d-dim o_t, then y_t = W_O · o_t + b_O.
//
// Backward walks t = T-1, T-2, ..., 0 and propagates gradients through:
//   (1) Output projection W_O
//   (2) Head-split (d -> H heads of D)
//   (3) Recurrence carrier dL/dS_{t-1} += diag(gamma) · dL/dS_t
//   (4) Outer product dL/dk = dS · v, dL/dv = dS^T · k
//   (5) γ gradient dL/dgamma[i] += sum_j dS[i,j] · S_{t-1}[i,j]
//   (6) xPos rotation inverse (rotation matrices are orthogonal, inverse = transpose)
//   (7) Dense.backward on Q, K, V projections
// ============================================================================

// Numerical-stable sigmoid
static inline double retnet_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
RetNetRetention::RetNetRetention(size_t d_model, size_t num_heads)
    : d_model_(d_model),
      num_heads_(num_heads == 0 ? 1 : num_heads),
      rope_base_(100.0),  // small base keeps rotation angles tractable for grad check
      W_Q(d_model, d_model), W_K(d_model, d_model),
      W_V(d_model, d_model), W_O(d_model, d_model),
      gamma_raw(1, d_model),
      grad_gamma_raw_(1, d_model)
{
    if (d_model == 0) {
        throw std::invalid_argument("RetNetRetention: d_model must be > 0");
    }
    if (d_model % 2 != 0) {
        throw std::invalid_argument("RetNetRetention: d_model must be even (xPos pair rotation)");
    }
    if (num_heads_ == 0 || d_model % num_heads_ != 0) {
        throw std::invalid_argument("RetNetRetention: num_heads must divide d_model evenly");
    }
    head_dim_ = d_model_ / num_heads_;
    if (head_dim_ % 2 != 0) {
        throw std::invalid_argument("RetNetRetention: head_dim must be even (xPos pair rotation)");
    }

    // gamma_raw = 0 → γ = sigmoid(0) = 0.5. Half-life ~ln(0.5)/ln(0.5)=1 step, slow decay.
    gamma_raw.fill(0.0);
    grad_gamma_raw_.fill(0.0);
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor RetNetRetention::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("RetNetRetention: input.cols must equal d_model");
    }
    size_t T = input.rows;
    last_input_ = input.clone();

    // Allocate cache
    last_q_pre_ = Tensor(T, d_model_); last_q_pre_.fill(0.0);
    last_k_pre_ = Tensor(T, d_model_); last_k_pre_.fill(0.0);
    last_v_pre_ = Tensor(T, d_model_); last_v_pre_.fill(0.0);
    last_q_rot_ = Tensor(T, d_model_); last_q_rot_.fill(0.0);
    last_k_rot_ = Tensor(T, d_model_); last_k_rot_.fill(0.0);
    last_v_     = Tensor(T, d_model_);
    last_gamma_ = Tensor(1, d_model_);
    last_states_ = Tensor(T + 1, num_heads_ * head_dim_ * head_dim_);
    last_states_.fill(0.0);
    last_o_     = Tensor(T, d_model_);

    // Step 1: projections
    last_q_pre_ = W_Q.forward(input);
    last_k_pre_ = W_K.forward(input);
    last_v_pre_ = W_V.forward(input);

    // Step 2: compute gamma = sigmoid(gamma_raw)
    for (size_t j = 0; j < d_model_; ++j) {
        last_gamma_(0, j) = retnet_sigmoid(gamma_raw(0, j));
    }
    // Store v directly (no rotation, no decay).
    last_v_ = last_v_pre_.clone();

    // Step 3: xPos rotation. For pair (i, i+D/2) of head_dim D, rotation at
    // position t (0-indexed): angle = t * theta_i, where
    //   theta_i = base^(-2i/(D/2)),   i in [0, D/2)
    // Q is rotated by +angle, K by -angle.
    //
    // For pair (a, b) = (q_pre[h*D + i], q_pre[h*D + i + D/2]):
    //   q_rot[h*D + i]       = a*cos(angle) - b*sin(angle)
    //   q_rot[h*D + i + D/2] = b*cos(angle) + a*sin(angle)
    // K is the inverse rotation (Q gets +angle, K gets -angle; equivalently
    // the "swap sign" rotation), which for a 2D pair is just (a*cos + b*sin, b*cos - a*sin)
    // — i.e., same cos, sin with opposite sign. We use the formula directly.

    // Precompute theta_i for i in [0, D/2) — shared across heads and positions.
    std::vector<double> theta(head_dim_ / 2);
    {
        double half = (double)(head_dim_ / 2);
        for (size_t i = 0; i < head_dim_ / 2; ++i) {
            theta[i] = std::pow(rope_base_, -2.0 * (double)i / half);
        }
    }

    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t head_off = h * head_dim_;
            for (size_t i = 0; i < head_dim_ / 2; ++i) {
                double angle = (double)t * theta[i];
                double c = std::cos(angle);
                double s = std::sin(angle);
                // Q
                double qa = last_q_pre_(t, head_off + i);
                double qb = last_q_pre_(t, head_off + i + head_dim_ / 2);
                last_q_rot_(t, head_off + i)             = qa * c - qb * s;
                last_q_rot_(t, head_off + i + head_dim_/2) = qb * c + qa * s;
                // K (inverse rotation: angle -> -angle, i.e. sin -> -sin)
                double ka = last_k_pre_(t, head_off + i);
                double kb = last_k_pre_(t, head_off + i + head_dim_ / 2);
                last_k_rot_(t, head_off + i)             = ka * c + kb * s;
                last_k_rot_(t, head_off + i + head_dim_/2) = kb * c - ka * s;
            }
        }
    }

    // Step 4: per-head state recurrence S_t = diag(gamma_h) · S_{t-1} + outer(k_t, v_t).
    // For each head h, S_t in R^{D × D}.
    //   S_t[i, j] = gamma[h*D + i] · S_{t-1}[i, j] + k_t[h*D + i] · v_t[h*D + j]
    //   o_t[h*D + j] = sum_i S_t[i, j] * q_t[h*D + i]
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t head_off = h * head_dim_;
            size_t state_off = h * head_dim_ * head_dim_;
            for (size_t i = 0; i < head_dim_; ++i) {
                double gi = last_gamma_(0, head_off + i);
                double ki = last_k_rot_(t, head_off + i);
                for (size_t j = 0; j < head_dim_; ++j) {
                    double vj = last_v_(t, head_off + j);
                    double s_prev = last_states_(t, state_off + i * head_dim_ + j);
                    double s_new = gi * s_prev + ki * vj;
                    last_states_(t + 1, state_off + i * head_dim_ + j) = s_new;
                }
            }
            // o_t^h = S_t · q_t^h  (where S_t is the *new* state at row t+1)
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    acc += last_states_(t + 1, state_off + i * head_dim_ + j)
                         * last_q_rot_(t, head_off + i);
                }
                last_o_(t, head_off + j) = acc;
            }
        }
    }

    // Step 5: output projection
    return W_O.forward(last_o_);
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
Tensor RetNetRetention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("RetNetRetention: grad_output.cols must equal d_model");
    }
    size_t T = grad_output.rows;

    // W_O.backward gives dL/d(o_t). We zero W_O grads first.
    W_O.zero_grad();
    W_Q.zero_grad();
    W_K.zero_grad();
    W_V.zero_grad();

    // Per-step gradient buffers
    Tensor grad_x(T, d_model_);           // dL/dx_t
    grad_x.fill(0.0);

    // Backprop through W_O. Dense::backward takes grad of W_O's output
    // (= grad_output) and returns dL/d(W_O's input) = dL/d(o_t) and accumulates
    // grad_W_O, grad_b_O.
    Tensor grad_o = W_O.backward(grad_output, 0.0);
    // grad_o == dL/d(o_t). We split into per-head slots but since
    // W_O just sees a flat (T, d) input/output, we keep the (T, d) layout
    // and only access per-head slots in the recurrence backward.

    // Backward through the recurrence. We need per-head per-step dS
    // gradients that flow backward. We use a single "future carrier" tensor
    // `grad_S_future` indexed [num_heads_, head_dim_, head_dim_] that holds
    // dL/dS^≥(t+1) at the start of iteration t. With the convention that
    //   S^≥t  := state at end of step t  (last_states_(t+1, ...))
    //   S^<t  := state at start of step t (last_states_(t, ...), with t=0 ⇒ 0)
    //   S^<t+1 == S^≥t  (the state from end of step t = start of step t+1)
    // the per-step math is:
    //   o_t^h = S^≥t^h · q_t^h       →  dL/dS^≥t^h += grad_o_t^h ⊗ q_t^h
    //   S^≥t^h = diag(γ_h) · S^<(t+1)^h + outer(k_t^h, v_t^h)
    //         = diag(γ_h) · S^≥(t-1)^h + outer(k_t^h, v_t^h)
    //   So dL/dS^≥(t-1)^h += diag(γ_h) · dL/dS^≥t^h
    //
    // We track dL/dS^≥t as a flat (1, H*D*D) buffer, updated per iteration.
    // At the start of iteration t, grad_S_future holds dL/dS^≥(t+1). The
    // "γ * grad_S_future" term in the dS formula below absorbs the decay:
    //   dL/dS^≥t = (grad_o_t ⊗ q_t) + γ · dL/dS^≥(t+1)  (per-channel, per-head)
    // At the end of each iteration, we set grad_S_future = dS (= dL/dS^≥t),
    // which will be dL/dS^≥(future) for the next (earlier) iteration.
    Tensor grad_S_future(1, num_heads_ * head_dim_ * head_dim_);
    grad_S_future.fill(0.0);  // dL/dS^≥T = 0 (no future beyond last step)

    // We also need dL/d(q_pre), dL/d(k_pre), dL/d(v_pre) — these will be
    // filled by unrotating the (rotated) q/k gradients and then handed to
    // Dense::backward.
    Tensor grad_q_rot(T, d_model_); grad_q_rot.fill(0.0);
    Tensor grad_k_rot(T, d_model_); grad_k_rot.fill(0.0);
    Tensor grad_v_pre(T, d_model_); grad_v_pre.fill(0.0);

    for (size_t ti = 0; ti < T; ++ti) {
        size_t t = T - 1 - ti;

        // Per step we need two gradient quantities:
        //   dS_ij = dL/dS^≥t[i,j]  (for backprop into k, v, γ)
        //   dL/dq_t[i] = sum_j grad_o_t[j] * S^≥t[i,j]  (the direct matrix-product path)
        // These are NOT the same — dS is the gradient w.r.t. S^≥t's contribution
        // to itself, while dL/dq is via the matrix product with q. We compute
        // both: dS first (which is the direct + γ*carrier), then dL/dq, dL/dk,
        // dL/dv, dL/dγ from dS.
        Tensor dS(1, num_heads_ * head_dim_ * head_dim_);
        dS.fill(0.0);
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t head_off = h * head_dim_;
            size_t state_off = h * head_dim_ * head_dim_;
            for (size_t i = 0; i < head_dim_; ++i) {
                double qi = last_q_rot_(t, head_off + i);
                double gi = last_gamma_(0, head_off + i);
                for (size_t j = 0; j < head_dim_; ++j) {
                    // dS = dL/dS^≥t[i,j] = (direct) + (γ * carrier)
                    double dS_ij = grad_o(t, head_off + j) * qi
                                 + gi * grad_S_future(0, state_off + i * head_dim_ + j);
                    dS(0, state_off + i * head_dim_ + j) = dS_ij;

                    // dL/dq[i]  ←  sum_j grad_o[j] * S^≥t[i,j]  (the matrix-product path)
                    //            = sum_j grad_o[j] * last_states_(t+1, state_off + i*D + j)
                    grad_q_rot(t, head_off + i) +=
                        grad_o(t, head_off + j) * last_states_(t + 1, state_off + i * head_dim_ + j);

                    // dL/d(k_rot_t[h*D + i]) += dS_ij * v_t[h*D + j]
                    grad_k_rot(t, head_off + i) += dS_ij * last_v_(t, head_off + j);

                    // dL/d(v_pre_t[h*D + j]) += dS_ij * k_rot_t[h*D + i]
                    grad_v_pre(t, head_off + j) += dS_ij * last_k_rot_(t, head_off + i);

                    // dL/d(gamma_raw[h*D + i]) += dS_ij * S^<t[i,j] * gamma * (1 - gamma)
                    //   S^<t = state at start of step t = last_states_(t, ...)
                    //   (chain through sigmoid: dgamma/dgamma_raw = gamma*(1-gamma))
                    double S_prev = last_states_(t, state_off + i * head_dim_ + j);
                    grad_gamma_raw_(0, head_off + i) += dS_ij * S_prev * gi * (1.0 - gi);
                }
            }
        }
        // Update grad_S_future for the next (earlier) iteration.
        // grad_S_future is "dL/dS^≥(future)" where "future" = one step ahead
        // of the *current* iteration. After processing step t, the next
        // iteration is t-1, and we want grad_S_future to hold dL/dS^≥t for
        // that iteration. So grad_S_future ← dS (which is dL/dS^≥t for step t).
        // The γ multiplier on the decay is folded into the dS formula (the
        // `gi * grad_S_future` term when computing dS).
        size_t D = head_dim_;
        size_t total_slots = num_heads_ * D * D;
        for (size_t s = 0; s < total_slots; ++s) {
            grad_S_future(0, s) = dS(0, s);
        }
    }

    // 2) Unrotate q and k gradients (xPos inverse rotations).
    //    Q forward:  q_rot = R(θ)  q    →  Q inverse:  q = R(-θ) q_rot
    //    K forward:  k_rot = R(-θ) k    →  K inverse:  k = R(θ)  k_rot
    //    (R(θ) = [[c,-s],[s,c]], so R(θ)ᵀ = R(-θ) = [[c,s],[-s,c]].)
    //    For pair (a, b) = (x[i], x[i + D/2]) in a head, the unrotated
    //    gradient components are:
    //      Q inverse: a' = c·a + s·b,  b' = c·b - s·a
    //      K inverse: a' = c·a - s·b,  b' = c·b + s·a
    //    (The two inverses have opposite sin signs — the unrotation mirrors
    //     the forward rotation angle.)

    // Precompute theta for inverse (same as forward, position-dependent)
    std::vector<double> theta_inv(head_dim_ / 2);
    {
        double half = (double)(head_dim_ / 2);
        for (size_t i = 0; i < head_dim_ / 2; ++i) {
            theta_inv[i] = std::pow(rope_base_, -2.0 * (double)i / half);
        }
    }
    Tensor grad_q_pre(T, d_model_); grad_q_pre.fill(0.0);
    Tensor grad_k_pre(T, d_model_); grad_k_pre.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t head_off = h * head_dim_;
            for (size_t i = 0; i < head_dim_ / 2; ++i) {
                double angle = (double)t * theta_inv[i];
                double c = std::cos(angle);
                double s = std::sin(angle);
                // Q inverse
                double qa = grad_q_rot(t, head_off + i);
                double qb = grad_q_rot(t, head_off + i + head_dim_ / 2);
                grad_q_pre(t, head_off + i)               = qa * c + qb * s;
                grad_q_pre(t, head_off + i + head_dim_/2)  = qb * c - qa * s;
                // K inverse
                double ka = grad_k_rot(t, head_off + i);
                double kb = grad_k_rot(t, head_off + i + head_dim_ / 2);
                grad_k_pre(t, head_off + i)               = ka * c - kb * s;
                grad_k_pre(t, head_off + i + head_dim_/2)  = kb * c + ka * s;
            }
        }
    }

    // 3) Dense backward for Q, K, V projections
    //    W_Q.backward(grad_q_pre, 0.0) gives dL/d(x_t) from the Q path and
    //    accumulates grad_W_Q, grad_b_Q. Same for K, V. Sum these for grad_x.
    Tensor grad_x_Q = W_Q.backward(grad_q_pre, 0.0);
    Tensor grad_x_K = W_K.backward(grad_k_pre, 0.0);
    Tensor grad_x_V = W_V.backward(grad_v_pre, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_x(t, j) = grad_x_Q(t, j) + grad_x_K(t, j) + grad_x_V(t, j);
        }
    }
    return grad_x;
}

// ----------------------------------------------------------------------------
// update_weights / zero_grad / parameters / gradients
// ----------------------------------------------------------------------------
void RetNetRetention::update_weights(double learning_rate) {
    W_Q.update_weights(learning_rate);
    W_K.update_weights(learning_rate);
    W_V.update_weights(learning_rate);
    W_O.update_weights(learning_rate);
    for (size_t j = 0; j < d_model_; ++j) {
        gamma_raw(0, j) -= learning_rate * grad_gamma_raw_(0, j);
    }
}

void RetNetRetention::zero_grad() {
    W_Q.zero_grad();
    W_K.zero_grad();
    W_V.zero_grad();
    W_O.zero_grad();
    grad_gamma_raw_.fill(0.0);
}

std::vector<Tensor*> RetNetRetention::parameters() {
    return {
        &W_Q.weights, &W_Q.bias,
        &W_K.weights, &W_K.bias,
        &W_V.weights, &W_V.bias,
        &W_O.weights, &W_O.bias,
        &gamma_raw
    };
}

std::vector<Tensor*> RetNetRetention::gradients() {
    return {
        &W_Q.grad_weights, &W_Q.grad_bias,
        &W_K.grad_weights, &W_K.grad_bias,
        &W_V.grad_weights, &W_V.grad_bias,
        &W_O.grad_weights, &W_O.grad_bias,
        &grad_gamma_raw_
    };
}
