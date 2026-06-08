#include "xlstm.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// sLSTM implementation
// ----------------------------------------------------------------------------
//
// Forward per-step (all vector ops are element-wise):
//   m_t       = max(log σ(f_pre_t) + m_{t-1}, i_pre_t)         (stabilizer)
//   f'_t      = σ(f_pre_t) * exp(m_{t-1} - m_t)               (forget prime, in [0,1])
//   i'_t      = exp(i_pre_t - m_t)                            (input prime, in [0,1])
//   c_t       = f'_t ⊙ c_{t-1} + i'_t ⊙ tanh(z_t)             (cell state)
//   n_t       = f'_t ⊙ n_{t-1} + i'_t                         (normalizer)
//   h_t       = σ(o_pre_t) ⊙ tanh(c_t / n_t)                  (hidden output)
//
// Backward per-step (recurrence + output):
//   Let s_t = c_t / n_t,  h_t = σ(o_pre) * tanh(s_t).
//
//   dL/d(σ(o_pre)) = dL/dh_t * tanh(s_t)
//   dL/d(tanh_s)   = dL/dh_t * σ(o_pre)
//   dL/d(s_t)      = dL/d(tanh_s) * (1 - tanh_s^2)
//   dL/d(c_t)      = dL/d(s_t) * (1/n_t)        [local]
//   dL/d(n_t)      = dL/d(s_t) * (-c_t/n_t^2)   [local]
//   dL/d(o_pre_t)  = dL/d(σ(o_pre)) * σ(o_pre) * (1 - σ(o_pre))
//
//   c_t and n_t recurrence:
//     dL/d(f'_t) = dL/d(c_t) * c_{t-1} + dL/d(n_t) * n_{t-1}
//     dL/d(i'_t) = dL/d(c_t) * tanh(z_t) + dL/d(n_t)
//     dL/d(z_t)   = dL/d(c_t) * i'_t * (1 - tanh(z_t)^2)
//
//   f'_t = σ(f_pre_t) * exp(m_{t-1} - m_t):
//     dL/d(σ(f_pre_t))  = dL/d(f'_t) * exp(m_{t-1} - m_t)
//     dL/d(m_{t-1})     += dL/d(f'_t) * f'_t    (recurrence carrier)
//     dL/d(m_t)         += dL/d(f'_t) * (-f'_t) (direct, from m_t in exponent)
//     dL/d(log σ(f_pre))= dL/d(σ(f_pre_t)) * σ(f_pre_t)
//     dL/d(f_pre_t)     = dL/d(log σ(f_pre)) * (1 - σ(f_pre_t))
//
//   i'_t = exp(i_pre_t - m_t):
//     dL/d(i_pre_t)     = dL/d(i'_t) * i'_t
//     dL/d(m_t)        += dL/d(i'_t) * (-i'_t)  (direct)
//
//   m_t = max(log σ(f_pre) + m_{t-1}, i_pre):
//     If i_pre wins:   dL/d(i_pre) += dL/d(m_t);  dL/d(log σ(f_pre) + m_{t-1}) += 0
//     If other wins:   dL/d(i_pre) += 0;          dL/d(log σ(f_pre)) += dL/d(m_t);
//                                               dL/d(m_{t-1}) += dL/d(m_t)
// ============================================================================

static inline double xlstm_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

static inline double xlstm_log_sigmoid(double x) {
    // log(σ(x)) stable.  σ(x) = 1/(1+exp(-x))
    if (x >= 0.0) {
        // log σ(x) = -log(1+exp(-x))  (avoid overflow when -x is large negative)
        return -std::log(1.0 + std::exp(-x));
    } else {
        // log σ(x) = x - log(1+exp(x))  (avoid overflow when x is large positive)
        return x - std::log(1.0 + std::exp(x));
    }
}

// Free helper: did the i_pre branch win the max for m_t?
// m_t = max(log_sigma_f + m_prev, i_pre).
// Returns true if i_pre >= log_sigma_f + m_prev.
static inline bool i_pre_won_max(double log_sigma_f_plus_m_prev, double i_pre) {
    return i_pre >= log_sigma_f_plus_m_prev;
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
SLSTMCell::SLSTMCell(size_t input_size, size_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size)
{
    if (input_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("SLSTMCell: input_size and hidden_size must be > 0");
    }
    size_t in_dim = input_size + hidden_size;
    size_t out_dim = 4 * hidden_size;  // [z, i, f, o]

    // Xavier-uniform init for W.
    // bound = sqrt(6 / (fan_in + fan_out))
    std::mt19937 gen(42);
    double bound = std::sqrt(6.0 / (double)(in_dim + out_dim));
    std::uniform_real_distribution<> dis(-bound, bound);

    W = Tensor(out_dim, in_dim);
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = dis(gen);
    b = Tensor(1, out_dim);
    b.fill(0.0);

    // Forget-gate bias initialized to 1.0 (PyTorch convention).
    for (size_t j = 0; j < hidden_size; ++j) {
        b(0, 2 * hidden_size + j) = 1.0;
    }

    grad_W = Tensor(out_dim, in_dim);
    grad_b = Tensor(1, out_dim);
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor SLSTMCell::forward(const Tensor& input) {
    if (input.cols != input_size_) {
        throw std::invalid_argument("SLSTMCell::forward: input.cols != input_size");
    }
    size_t T = input.rows;
    if (T < 1) {
        throw std::invalid_argument("SLSTMCell::forward: input must have at least one row");
    }
    last_input_ = input.clone();

    // Allocate caches
    last_xh_ = Tensor(T, input_size_ + hidden_size_);
    last_z_        = Tensor(T, hidden_size_);
    last_i_pre_    = Tensor(T, hidden_size_);
    last_f_pre_    = Tensor(T, hidden_size_);
    last_o_pre_    = Tensor(T, hidden_size_);
    last_i_prime_  = Tensor(T, hidden_size_);
    last_f_prime_  = Tensor(T, hidden_size_);
    last_log_sigmoid_f_ = Tensor(T, hidden_size_);
    last_h_ = Tensor(T + 1, hidden_size_);
    last_c_ = Tensor(T + 1, hidden_size_);
    last_n_ = Tensor(T + 1, hidden_size_);
    last_m_ = Tensor(T + 1, hidden_size_);

    // Initial states (row 0 = state at t=-1)
    for (size_t j = 0; j < hidden_size_; ++j) {
        last_h_(0, j) = 0.0;
        last_c_(0, j) = 0.0;
        last_n_(0, j) = 1.0;  // convention: n_0 = 1 to avoid 0/0 at t=0
        last_m_(0, j) = 0.0;
    }

    Tensor output(T, hidden_size_);

    for (size_t t = 0; t < T; ++t) {
        // 1) Concatenate [x_t, h_{t-1}] -> (input_size + hidden_size)
        for (size_t k = 0; k < input_size_; ++k) {
            last_xh_(t, k) = input(t, k);
        }
        for (size_t k = 0; k < hidden_size_; ++k) {
            last_xh_(t, input_size_ + k) = last_h_(t, k);
        }

        // 2) Gate pre-activations: [z, i, f, o] = W @ [x_t; h_{t-1}] + b
        for (size_t j = 0; j < 4 * hidden_size_; ++j) {
            double acc = b(0, j);
            for (size_t k = 0; k < input_size_ + hidden_size_; ++k) {
                acc += W(j, k) * last_xh_(t, k);
            }
            if (j < hidden_size_) {
                last_z_(t, j) = acc;
            } else if (j < 2 * hidden_size_) {
                last_i_pre_(t, j - hidden_size_) = acc;
            } else if (j < 3 * hidden_size_) {
                last_f_pre_(t, j - 2 * hidden_size_) = acc;
            } else {
                last_o_pre_(t, j - 3 * hidden_size_) = acc;
            }
        }

        // 3) Stabilizer m_t and gate primes
        for (size_t j = 0; j < hidden_size_; ++j) {
            double i_pre = last_i_pre_(t, j);
            double f_pre = last_f_pre_(t, j);
            double log_sigma_f = xlstm_log_sigmoid(f_pre);
            last_log_sigmoid_f_(t, j) = log_sigma_f;
            double sigma_f = std::exp(log_sigma_f);
            double m_prev = last_m_(t, j);
            double log_sigma_f_plus_m_prev = log_sigma_f + m_prev;
            double m_t = std::max(log_sigma_f_plus_m_prev, i_pre);
            last_m_(t + 1, j) = m_t;
            // forget prime
            double f_prime = sigma_f * std::exp(m_prev - m_t);
            last_f_prime_(t, j) = f_prime;
            // input prime
            double i_prime = std::exp(i_pre - m_t);
            last_i_prime_(t, j) = i_prime;
        }

        // 4) Cell state and normalizer
        for (size_t j = 0; j < hidden_size_; ++j) {
            double f_p = last_f_prime_(t, j);
            double i_p = last_i_prime_(t, j);
            double z = last_z_(t, j);
            double tanh_z = std::tanh(z);
            double c_prev = last_c_(t, j);
            double n_prev = last_n_(t, j);
            last_c_(t + 1, j) = f_p * c_prev + i_p * tanh_z;
            last_n_(t + 1, j) = f_p * n_prev + i_p;
        }

        // 5) Hidden state
        for (size_t j = 0; j < hidden_size_; ++j) {
            double c = last_c_(t + 1, j);
            double n = last_n_(t + 1, j);
            double s = (n > 1e-12) ? (c / n) : 0.0;
            double tanh_s = std::tanh(s);
            double o_pre = last_o_pre_(t, j);
            double sig_o = xlstm_sigmoid(o_pre);
            last_h_(t + 1, j) = sig_o * tanh_s;
            output(t, j) = last_h_(t + 1, j);
        }
    }
    return output;
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
Tensor SLSTMCell::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_input_.rows || grad_output.cols != hidden_size_) {
        throw std::invalid_argument("SLSTMCell::backward: grad_output shape mismatch");
    }
    size_t T = last_input_.rows;

    // Zero parameter grads.
    grad_W.fill(0.0);
    grad_b.fill(0.0);

    // Local gradient buffers for recurrent state derivatives.
    // g_c, g_n, g_m indexed 0..T (g_x[0] is initial-state gradient, which we
    // don't return to the caller; g_x[T-1] receives output-side gradient).
    Tensor g_c(T + 1, hidden_size_);  g_c.fill(0.0);
    Tensor g_n(T + 1, hidden_size_);  g_n.fill(0.0);
    Tensor g_m(T + 1, hidden_size_);  g_m.fill(0.0);
    Tensor g_xh(T, input_size_ + hidden_size_);
    g_xh.fill(0.0);

    // g_h[t] = grad_output[t] + (recurrence contribution from time t+1).
    // The recurrence contribution from time t+1 is computed in the BPTT
    // iteration at time t+1 as the h_t portion of g_xh. We initialize g_h
    // with grad_output and then add the recurrence contribution in the loop.
    Tensor g_h(T, hidden_size_);
    for (size_t i = 0; i < T * hidden_size_; ++i) g_h.data[i] = grad_output.data[i];

    // BPTT — walk backward in time.
    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        // Propagate the h_t recurrence gradient from time t+1 to g_h[t].
        // (At t = T-1, g_h[t+1] doesn't exist — we used grad_output only.)
        if (t + 1 < T) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                g_h(t, j) += g_xh(t + 1, input_size_ + j);
            }
        }
        for (size_t j = 0; j < hidden_size_; ++j) {
            double gh = g_h(t, j);

            double c = last_c_(t + 1, j);
            double n = last_n_(t + 1, j);
            double n_safe = std::max(n, 1e-12);
            double s = c / n_safe;
            double tanh_s = std::tanh(s);
            double sig_o = xlstm_sigmoid(last_o_pre_(t, j));

            // h_t = sig_o * tanh(s)
            double d_tanh_s = gh * sig_o;
            double d_s = d_tanh_s * (1.0 - tanh_s * tanh_s);
            double dc_local = d_s / n_safe;
            double dn_local = -d_s * c / (n_safe * n_safe);
            double d_o_pre = gh * tanh_s * sig_o * (1.0 - sig_o);

            // Accumulate local c_t, n_t gradients
            g_c(t + 1, j) += dc_local;
            g_n(t + 1, j) += dn_local;
            // bias gradient for output gate
            grad_b(0, 3 * hidden_size_ + j) += d_o_pre;

            // Recurrence state values
            double f_p = last_f_prime_(t, j);
            double i_p = last_i_prime_(t, j);
            double z = last_z_(t, j);
            double tanh_z = std::tanh(z);
            double c_prev = last_c_(t, j);
            double n_prev = last_n_(t, j);
            double m_prev = last_m_(t, j);
            double m_t = last_m_(t + 1, j);
            double log_sigma_f = last_log_sigmoid_f_(t, j);
            double sigma_f = std::exp(log_sigma_f);
            double log_sigma_f_plus_m_prev = log_sigma_f + m_prev;
            double i_pre = last_i_pre_(t, j);

            double gc_t = g_c(t + 1, j);
            double gn_t = g_n(t + 1, j);

            // dL/d(f'_t), dL/d(i'_t), dL/d(z_t)
            double d_f_prime = gc_t * c_prev + gn_t * n_prev;
            double d_i_prime = gc_t * tanh_z + gn_t;
            double d_z = gc_t * i_p * (1.0 - tanh_z * tanh_z);

            // g_m(t+1, j) from gate prime expansions — accumulate BEFORE
            // using gm_t for the max-derivative (the gate prime contribution
            // is part of the total dL/d(m_t) that flows to the i_pre / f_pre
            // branches via the max).
            g_m(t + 1, j) += d_f_prime * (-f_p) + d_i_prime * (-i_p);

            // Now read the total dL/d(m_t) = (recurrence from t+1) + (gate prime from t)
            double gm_t = g_m(t + 1, j);

            // max-derivative: which branch won?
            bool i_won = i_pre_won_max(log_sigma_f_plus_m_prev, i_pre);

            // dL/d(i_pre_t)
            //   From i'_t: d_i_prime * i'_t
            //   Plus: if i_pre won the max for m_t, gm_t flows through
            double d_i_pre = d_i_prime * i_p;
            if (i_won) {
                d_i_pre += gm_t;
            }

            // dL/d(f_pre_t) chain through log σ(f_pre_t)
            //   dL/d(log σ(f_pre_t)) = d_f_prime * exp(m_prev - m_t)
            //                         (this is d_f_prime * (f'_t / sigma_f))
            //   dL/d(f_pre_t) = d_log_sigma_f * (1 - σ(f_pre_t))
            //   Plus: if log_sigma_f + m_prev won the max for m_t, gm_t flows
            //         through d(log σ(f_pre_t)) which is (1 - σ(f_pre_t)) per element.
            double d_log_sigma_f = d_f_prime * std::exp(m_prev - m_t);
            double d_f_pre = d_log_sigma_f * (1.0 - sigma_f);
            if (!i_won) {
                d_f_pre += gm_t * (1.0 - sigma_f);
            }

            // g_m(t, j) (recurrence carrier, m_{t-1} <- m_t via two paths):
            //   1. f' path: dL/d(m_{t-1}) += d_f_prime * f'_t
            //   2. max path: if (log_sigma_f + m_{t-1}) won the max for m_t,
            //                dL/d(m_{t-1}) += gm_t
            g_m(t, j) += d_f_prime * f_p;
            if (!i_won) {
                g_m(t, j) += gm_t;
            }

            // g_c(t, j), g_n(t, j) from c_t, n_t recurrence
            g_c(t, j) += gc_t * f_p;
            g_n(t, j) += gn_t * f_p;

            // bias gradients for z, i, f gates
            grad_b(0, 0 * hidden_size_ + j) += d_z;
            grad_b(0, 1 * hidden_size_ + j) += d_i_pre;
            grad_b(0, 2 * hidden_size_ + j) += d_f_pre;

            // g_xh(t, k) <- W^T @ [d_z; d_i_pre; d_f_pre; d_o_pre] (per column j)
            for (size_t k = 0; k < input_size_ + hidden_size_; ++k) {
                g_xh(t, k) += W(0 * hidden_size_ + j, k) * d_z
                            + W(1 * hidden_size_ + j, k) * d_i_pre
                            + W(2 * hidden_size_ + j, k) * d_f_pre
                            + W(3 * hidden_size_ + j, k) * d_o_pre;
            }

            // grad_W: dL/d(W[j', k]) = sum_t dL/d(gate_pre[t][j']) * xh[t][k]
            for (size_t k = 0; k < input_size_ + hidden_size_; ++k) {
                grad_W(0 * hidden_size_ + j, k) += d_z       * last_xh_(t, k);
                grad_W(1 * hidden_size_ + j, k) += d_i_pre   * last_xh_(t, k);
                grad_W(2 * hidden_size_ + j, k) += d_f_pre   * last_xh_(t, k);
                grad_W(3 * hidden_size_ + j, k) += d_o_pre   * last_xh_(t, k);
            }
        }
    }

    // Return only the input gradient (first input_size_ columns of g_xh).
    Tensor grad_input(T, input_size_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < input_size_; ++k) {
            grad_input(t, k) = g_xh(t, k);
        }
    }
    return grad_input;
}

// ----------------------------------------------------------------------------
// Update / zero_grad / params / grads
// ----------------------------------------------------------------------------
void SLSTMCell::update_weights(double learning_rate) {
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= learning_rate * grad_W.data[i];
    }
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] -= learning_rate * grad_b.data[i];
    }
    // Re-apply forget-bias=1 convention (PyTorch-style) so the bias doesn't drift.
    for (size_t j = 0; j < hidden_size_; ++j) {
        b(0, 2 * hidden_size_ + j) = 1.0;
    }
}

void SLSTMCell::zero_grad() {
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

std::vector<Tensor*> SLSTMCell::parameters() {
    return {&W, &b};
}

std::vector<Tensor*> SLSTMCell::gradients() {
    return {&grad_W, &grad_b};
}
