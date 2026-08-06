#include "h3.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ----------------------------------------------------------------------------
// H3Block implementation
// ----------------------------------------------------------------------------
//
// See h3.h for the full mathematical specification. This file implements:
//
//   Forward:
//     1. Q/K/V projections (Dense)
//     2. Shift SSM on K (sliding window)
//     3. Diagonal SSM on K̄ ⊗ V (per-channel decay-and-add)
//     4. Recall: O_t = Q_t · Z_t
//     5. Output projection (Dense)
//
//   Backward:
//     1. Reverse order through all 5 steps
//     2. The shift-SSM backward is the transpose of the sliding-window
//        forward.
//     3. The diagonal-SSM backward is a backward sweep over the gradient
//        sequence with the same scalar λ per channel.
//     4. Outer-product backward is the standard matrix contraction.
// ----------------------------------------------------------------------------

// ---------- helper: numerically stable sigmoid ----------
static inline double h3_sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = std::exp(x);
    return z / (1.0 + z);
}

double H3Block::sigmoid(double x) {
    return h3_sigmoid(x);
}

// ---------- constructor ----------
H3Block::H3Block(size_t d_model)
    : W_Q(d_model, d_model),
      W_K(d_model, d_model),
      W_V(d_model, d_model),
      W_O(d_model, d_model),
      lambda_log(d_model, d_model),
      d_model_(d_model)
{
    if (d_model == 0) {
        throw std::invalid_argument("H3Block: d_model must be > 0");
    }

    // Initialize λ_log so that λ = sigmoid(λ_log) ≈ 0.5 — i.e., λ_log ≈ 0.
    // λ ∈ (0, 1) for stability; λ = 0.5 (half-life of 1 step) is a reasonable
    // default for the diagonal SSM.
    for (size_t i = 0; i < d_model; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            lambda_log(i, j) = 0.0;
        }
    }

    // Initialize hidden grad buffer
    grad_lambda_log_ = Tensor(d_model, d_model);

    // Standard init for dense projections — init_weights() is called by Dense
    // constructor with xavier default. No need to override.
}

// ---------- forward ----------
Tensor H3Block::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("H3Block: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("H3Block: input must have at least one token");
    }

    // Cache input
    last_input_ = input.clone();

    // Step 1: Q/K/V projections
    last_Q_ = W_Q.forward(input);  // (T, d)
    last_K_ = W_K.forward(input);  // (T, d)
    last_V_ = W_V.forward(input);  // (T, d)

    // Step 2: Shift SSM on K
    // K̄_t[i] = K_{t-i} if i ≤ t, else 0
    // (Equivalent to: x_t[0] = K_t, x_t[i] = x_{t-1}[i-1] for i ≥ 1, with x_0 = 0.)
    last_K_bar_ = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            if (i <= t) {
                last_K_bar_(t, i) = last_K_(t - i, i);
            } else {
                last_K_bar_(t, i) = 0.0;
            }
        }
    }

    // Step 3: Diagonal SSM (per-channel λ)
    last_lambda_ = Tensor(d_model_, d_model_);
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_lambda_(i, j) = h3_sigmoid(lambda_log(i, j));
        }
    }

    last_Z_ = Tensor(T * d_model_, d_model_);
    last_T_ = T;
    Tensor O(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        // Compute z_t = K̄_t ⊗ V_t (outer product)
        // Z_t = λ·Z_{t-1} + z_t,  Z_{-1} = 0
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double z = last_K_bar_(t, i) * last_V_(t, j);
                double Z_t_ij;
                if (t == 0) {
                    Z_t_ij = z;
                } else {
                    Z_t_ij = last_lambda_(i, j) * last_Z_((t - 1) * d_model_ + i, j) + z;
                }
                last_Z_(t * d_model_ + i, j) = Z_t_ij;
            }
        }
        // O_t[i] = sum_j Q_t[j] * Z_t[j, i]  (matrix product Q_t · Z_t)
        for (size_t i = 0; i < d_model_; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                sum += last_Q_(t, j) * last_Z_(t * d_model_ + j, i);
            }
            O(t, i) = sum;
        }
    }

    // Step 5: output projection
    Tensor output = W_O.forward(O);
    return output;
}

// ---------- backward ----------
Tensor H3Block::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t T = last_T_;
    size_t d = d_model_;

    // Step 5 backward: output projection (Dense backward)
    // W_O.forward did O · W_O^T (with bias). Dense::backward returns
    // grad_input = grad_output * W_O.weights (Dense convention).
    Tensor grad_O = W_O.backward(grad_output, 0.0);
    // grad_O shape: (T, d)

    // Step 4 backward: Recall O_t = Q_t · Z_t
    // dQ_t[j] = sum_i dO_t[i] * Z_t[j, i]    (i.e., dQ_t = dO_t · Z_t^T)
    // dZ_t[j, i] = dO_t[i] * Q_t[j]          (i.e., outer product)
    Tensor grad_Q(T, d);
    // 3D-into-2D: (T, d, d) flattened to (T*d, d), indexed as grad_Z_(t*d + j, i)
    Tensor grad_Z(T * d, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double dq = 0.0;
            for (size_t i = 0; i < d; ++i) {
                dq += grad_O(t, i) * last_Z_(t * d + j, i);
            }
            grad_Q(t, j) = dq;
        }
        for (size_t j = 0; j < d; ++j) {
            for (size_t i = 0; i < d; ++i) {
                grad_Z(t * d + j, i) = grad_O(t, i) * last_Q_(t, j);
            }
        }
    }

    // Step 3 backward: Diagonal SSM with per-channel λ
    //
    // Forward recurrence: Z_t[i,j] = λ[i,j] * Z_{t-1}[i,j] + z_t[i,j]
    // The gradient at z_t is the sum of all future grad-Z contributions:
    //     dz_t = sum_{s ≥ t} λ^{s-t} grad_Z_s
    // computed via a backward sweep. Then dλ[i,j] comes from the chain
    //     dZ_t[i,j] / dλ[i,j]  for t ≥ 1 is  Z_{t-1}[i,j]
    // so dλ[i,j] = sum_{t ≥ 1} grad_Z[t] * Z_{t-1}   (per-channel scalar product)

    Tensor grad_z(T * d, d);  // (T, d, d) flattened to (T*d, d)
    // Initialize grad_z = grad_Z (no λ multiplier at the last step)
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_z(t * d + i, j) = grad_Z(t * d + i, j);
            }
        }
    }
    // Backward sweep: grad_z[t-1] += λ * grad_z[t] for t = T-1 down to 1
    for (size_t t = T - 1; t > 0; --t) {
        size_t t_prev = t - 1;
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_z(t_prev * d + i, j) += last_lambda_(i, j) * grad_z(t * d + i, j);
            }
        }
    }

    // d_lambda[i,j] = sum_{s=0}^{T-2} Z_s[i,j] * grad_z[s+1, i, j]
    //
    // Derivation: the forward recurrence Z_t = lambda*Z_{t-1} + z_t has the
    // chain rule dZ_t/dlambda = Z_{t-1} + lambda * dZ_{t-1}/dlambda, which
    // unrolls to dZ_t/dlambda = sum_{s=0}^{t-1} lambda^{t-1-s} * Z_s. Then
    //     dL/dlambda = sum_t G_t * dZ_t/dlambda
    //               = sum_t G_t * sum_{s<=t-1} lambda^{t-1-s} * Z_s
    //               = sum_s Z_s * sum_{t >= s+1} lambda^{t-1-s} * G_t
    //               = sum_s Z_s * grad_z[s+1]   (since grad_z[s+1] = sum_{t >= s+1} lambda^{t-1-s} * G_t).
    Tensor grad_lambda = Tensor(d, d);
    grad_lambda.fill(0.0);
    for (size_t s = 0; s + 1 < T; ++s) {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                grad_lambda(i, j) += last_Z_(s * d + i, j) * grad_z((s + 1) * d + i, j);
            }
        }
    }

    // Outer product backward: z_t = K̄_t ⊗ V_t
    // dK̄_t[i] = sum_j grad_z[t, i, j] · V_t[j]
    // dV_t[j]  = sum_i grad_z[t, i, j] · K̄_t[i]
    Tensor grad_K_bar(T, d);
    Tensor grad_V(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            double s1 = 0.0;
            for (size_t j = 0; j < d; ++j) {
                s1 += grad_z(t * d + i, j) * last_V_(t, j);
            }
            grad_K_bar(t, i) = s1;
        }
        for (size_t j = 0; j < d; ++j) {
            double s2 = 0.0;
            for (size_t i = 0; i < d; ++i) {
                s2 += grad_z(t * d + i, j) * last_K_bar_(t, i);
            }
            grad_V(t, j) = s2;
        }
    }

    // Step 2 backward: Shift SSM
    // Forward: K̄_t[i] = K_{t-i} if i ≤ t, else 0
    // So K_t gets contributions from K̄_{t+i}[i] for i in 0..d-1 (when t+i < T).
    // dK_t = sum_{i: t+i < T} dK̄_{t+i}[i]
    Tensor grad_K(T, d);
    grad_K.fill(0.0);
    for (size_t i = 0; i < d; ++i) {
        for (size_t t = 0; t < T; ++t) {
            if (t + i < T) {
                grad_K(t, i) = grad_K_bar(t + i, i);
            } else {
                grad_K(t, i) = 0.0;
            }
        }
    }

    // Step 1 backward: Q/K/V projections
    // Dense backward returns grad_input = grad_output * W.weights.
    Tensor grad_Q_in = W_Q.backward(grad_Q, 0.0);
    Tensor grad_K_in = W_K.backward(grad_K, 0.0);
    Tensor grad_V_in = W_V.backward(grad_V, 0.0);

    // Aggregate gradients at the input
    Tensor grad_input(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            grad_input(t, i) = grad_Q_in(t, i) + grad_K_in(t, i) + grad_V_in(t, i);
        }
    }

    // Accumulate λ_log gradient via dλ_log = dλ * λ * (1 - λ).
    // (Sigmoid derivative: dλ/dlogit = σ(x) · (1 - σ(x)) = λ · (1 - λ).)
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double lam = last_lambda_(i, j);
            grad_lambda_log_(i, j) += grad_lambda(i, j) * lam * (1.0 - lam);
        }
    }

    return grad_input;
}

// ---------- update_weights ----------
void H3Block::update_weights(double learning_rate) {
    W_Q.update_weights(learning_rate);
    W_K.update_weights(learning_rate);
    W_V.update_weights(learning_rate);
    W_O.update_weights(learning_rate);

    // Update λ_log
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            lambda_log(i, j) -= learning_rate * grad_lambda_log_(i, j);
        }
    }
}

// ---------- zero_grad ----------
void H3Block::zero_grad() {
    W_Q.zero_grad();
    W_K.zero_grad();
    W_V.zero_grad();
    W_O.zero_grad();
    grad_lambda_log_.fill(0.0);
}

// ---------- parameters ----------
std::vector<Tensor*> H3Block::parameters() {
    std::vector<Tensor*> p;
    auto qp = W_Q.parameters();
    auto kp = W_K.parameters();
    auto vp = W_V.parameters();
    auto op = W_O.parameters();
    p.insert(p.end(), qp.begin(), qp.end());
    p.insert(p.end(), kp.begin(), kp.end());
    p.insert(p.end(), vp.begin(), vp.end());
    p.insert(p.end(), op.begin(), op.end());
    p.push_back(&lambda_log);
    return p;
}

// ---------- gradients ----------
std::vector<Tensor*> H3Block::gradients() {
    std::vector<Tensor*> g;
    auto qg = W_Q.gradients();
    auto kg = W_K.gradients();
    auto vg = W_V.gradients();
    auto og = W_O.gradients();
    g.insert(g.end(), qg.begin(), qg.end());
    g.insert(g.end(), kg.begin(), kg.end());
    g.insert(g.end(), vg.begin(), vg.end());
    g.insert(g.end(), og.begin(), og.end());
    g.push_back(&grad_lambda_log_);
    return g;
}
