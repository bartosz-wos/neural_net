#include "mlstm.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// mLSTM implementation (Beck et al. 2024, matrix-memory xLSTM variant)
//
// Per-step forward (all matrix operations are shape d×d for C, N, d×1 for
// the rest):
//
//   1. Concat [x_t, h_{t-1}] -> (in + d)
//   2. [q, k, v, i, f, o] = W @ [x; h_prev] + b    (each in R^d)
//   3. m_t       = max(log σ(f) + m_{t-1}, i)
//   4. f'_t      = exp(log σ(f) + m_{t-1} - m_t)    (forget prime)
//   5. i'_t      = exp(i - m_t)                      (input prime)
//   6. C_t       = f' · C_{t-1} + i' · outer(v, k)   (covariance update)
//   7. N_t       = f' · N_{t-1} + i' · outer(k, k)   (Gram / normalizer)
//   8. h_norm_t  = max(1, q^T N_t q)                  (scalar denominator)
//   9. h_pre_t   = C_t · q / h_norm_t                 (normalized retrieval)
//  10. h_t       = σ(o) · h_pre_t
//
// Per-step backward:
//   h_t = σ(o) * h_pre
//     dh_pre = gh * σ(o)
//     d_o_pre = gh * h_pre * σ(o)(1-σ(o))
//   h_pre = Cq / h_norm:
//     dC[i,j]    += (dh_pre[i] / h_norm) * q[j]               [always]
//     d_h_norm    = -sum_i dh_pre[i] * (Cq)[i] / h_norm²       [when h_norm > 1]
//     dN[i,j]    += d_h_norm * q[i] * q[j]                    [when h_norm > 1]
//     d_q[i]     += 2 * d_h_norm * (N q)[i]                   [when h_norm > 1]
//   d_q_from_C  = C^T * dh_pre                                 [always]
//   d_k_from_C  = i' * (dC^T v)        — outer product chain
//   d_k_from_N  = i' * (dN^T k)        — gram outer product
//   d_v          = sum_j dC[i,j] * i'[j] * k[j]  (per i)
//   d_i'[j]     = k[j] * (dC^T v)[j]
//   d_f'[j]     = <dC[:,j], C_{t-1}[:,j]> + <dN[:,j], N_{t-1}[:,j]>
//   d_i_pre, d_f_pre, d_o_pre: stabilizer expansion (same as sLSTM)
//   g_C(t, i, j) += f'[j] * g_C(t+1, i, j)        (state carry)
//   g_N(t, i, j) += f'[j] * g_N(t+1, i, j)        (state carry)
//   g_xh(t, k) = W^T @ [d_q, d_k, d_v, d_i_pre, d_f_pre, d_o_pre]
// ============================================================================

static inline double mlstm_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

static inline double mlstm_log_sigmoid(double x) {
    // log σ(x) stable.
    if (x >= 0.0) {
        return -std::log(1.0 + std::exp(-x));
    } else {
        return x - std::log(1.0 + std::exp(x));
    }
}

static inline bool mlstm_i_pre_won_max(double log_sigma_f_plus_m_prev, double i_pre) {
    return i_pre >= log_sigma_f_plus_m_prev;
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
MLSTMCell::MLSTMCell(size_t input_size, size_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size)
{
    if (input_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("MLSTMCell: input_size and hidden_size must be > 0");
    }
    size_t in_dim = input_size + hidden_size;
    size_t out_dim = 6 * hidden_size;  // [q, k, v, i, f, o]

    // Xavier-uniform init for W.
    std::mt19937 gen(42);
    double bound = std::sqrt(6.0 / (double)(in_dim + out_dim));
    std::uniform_real_distribution<> dis(-bound, bound);

    W = Tensor(out_dim, in_dim);
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = dis(gen);
    b = Tensor(1, out_dim);
    b.fill(0.0);

    // Forget-gate bias initialized to 1.0 (PyTorch convention, matches sLSTM).
    // f is the 5th block (index 4) of 6 blocks of size hidden_size each.
    for (size_t j = 0; j < hidden_size; ++j) {
        b(0, 4 * hidden_size + j) = 1.0;
    }

    grad_W = Tensor(out_dim, in_dim);
    grad_b = Tensor(1, out_dim);
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
Tensor MLSTMCell::forward(const Tensor& input) {
    if (input.cols != input_size_) {
        throw std::invalid_argument("MLSTMCell::forward: input.cols != input_size");
    }
    size_t T = input.rows;
    if (T < 1) {
        throw std::invalid_argument("MLSTMCell::forward: input must have at least one row");
    }
    last_input_ = input.clone();

    // Allocate caches
    last_xh_ = Tensor(T, input_size_ + hidden_size_);
    last_q_             = Tensor(T, hidden_size_);
    last_k_             = Tensor(T, hidden_size_);
    last_v_             = Tensor(T, hidden_size_);
    last_i_pre_         = Tensor(T, hidden_size_);
    last_f_pre_         = Tensor(T, hidden_size_);
    last_o_pre_         = Tensor(T, hidden_size_);
    last_i_prime_       = Tensor(T, hidden_size_);
    last_f_prime_       = Tensor(T, hidden_size_);
    last_log_sigmoid_f_ = Tensor(T, hidden_size_);
    last_h_             = Tensor(T + 1, hidden_size_);
    last_m_             = Tensor(T + 1, hidden_size_);
    // C and N flattened: (T+1) * d matrices, each d x d.  Index with
    // (t*d + i, j)  for matrix t, row i, col j.
    last_C_ = Tensor((T + 1) * hidden_size_, hidden_size_);
    last_N_ = Tensor((T + 1) * hidden_size_, hidden_size_);
    last_h_norm_ = Tensor(T, 1);
    last_Cq_     = Tensor(T, hidden_size_);

    // Initial states (row 0 / matrix 0 = state at t = -1)
    for (size_t j = 0; j < hidden_size_; ++j) {
        last_h_(0, j) = 0.0;
        last_m_(0, j) = 0.0;
    }
    for (size_t idx = 0; idx < hidden_size_ * hidden_size_; ++idx) {
        last_C_.data[idx] = 0.0;
        last_N_.data[idx] = 0.0;
    }

    Tensor output(T, hidden_size_);

    for (size_t t = 0; t < T; ++t) {
        size_t d = hidden_size_;

        // 1) Concat [x_t, h_{t-1}]
        for (size_t k = 0; k < input_size_; ++k) {
            last_xh_(t, k) = input(t, k);
        }
        for (size_t k = 0; k < d; ++k) {
            last_xh_(t, input_size_ + k) = last_h_(t, k);
        }

        // 2) Gate pre-activations: [q, k, v, i, f, o] = W @ [x; h_prev] + b
        for (size_t j = 0; j < 6 * d; ++j) {
            double acc = b(0, j);
            for (size_t k = 0; k < input_size_ + d; ++k) {
                acc += W(j, k) * last_xh_(t, k);
            }
            if (j < d) {
                last_q_(t, j) = acc;
            } else if (j < 2 * d) {
                last_k_(t, j - d) = acc;
            } else if (j < 3 * d) {
                last_v_(t, j - 2 * d) = acc;
            } else if (j < 4 * d) {
                last_i_pre_(t, j - 3 * d) = acc;
            } else if (j < 5 * d) {
                last_f_pre_(t, j - 4 * d) = acc;
            } else {
                last_o_pre_(t, j - 5 * d) = acc;
            }
        }

        // 3,4,5) Stabilizer m_t and gate primes
        for (size_t j = 0; j < d; ++j) {
            double i_pre = last_i_pre_(t, j);
            double f_pre = last_f_pre_(t, j);
            double log_sigma_f = mlstm_log_sigmoid(f_pre);
            last_log_sigmoid_f_(t, j) = log_sigma_f;
            double sigma_f = std::exp(log_sigma_f);
            double m_prev = last_m_(t, j);
            double log_sigma_f_plus_m_prev = log_sigma_f + m_prev;
            double m_t = std::max(log_sigma_f_plus_m_prev, i_pre);
            last_m_(t + 1, j) = m_t;
            double f_prime = sigma_f * std::exp(m_prev - m_t);
            last_f_prime_(t, j) = f_prime;
            double i_prime = std::exp(i_pre - m_t);
            last_i_prime_(t, j) = i_prime;
        }

        // 6,7) C_t, N_t updates — covariance / outer-product rules
        for (size_t i = 0; i < d; ++i) {
            double v_i = last_v_(t, i);
            double k_i = last_k_(t, i);
            for (size_t j = 0; j < d; ++j) {
                double f_p = last_f_prime_(t, j);
                double i_p = last_i_prime_(t, j);
                double k_j = last_k_(t, j);
                last_C_((t + 1) * d + i, j) = f_p * last_C_(t * d + i, j) + i_p * v_i * k_j;
                last_N_((t + 1) * d + i, j) = f_p * last_N_(t * d + i, j) + i_p * k_i * k_j;
            }
        }

        // 8) h_norm_t = max(1, q^T N_t q)
        // 9) h_pre_t = C_t · q / h_norm_t  (also store un-normalized Cq for backward)
        // 10) h_t = σ(o) · h_pre_t

        // First compute N q to evaluate q^T N q.
        // We also need C q for h_pre; compute both.
        double qNq = 0.0;
        for (size_t k = 0; k < d; ++k) {
            double Nq_k = 0.0;
            for (size_t j = 0; j < d; ++j) {
                Nq_k += last_N_((t + 1) * d + k, j) * last_q_(t, j);
            }
            qNq += last_q_(t, k) * Nq_k;
        }
        double h_norm = std::max(1.0, qNq);
        last_h_norm_(t, 0) = h_norm;

        for (size_t i = 0; i < d; ++i) {
            double Cq_i = 0.0;
            for (size_t j = 0; j < d; ++j) {
                Cq_i += last_C_((t + 1) * d + i, j) * last_q_(t, j);
            }
            last_Cq_(t, i) = Cq_i;
        }

        for (size_t j = 0; j < d; ++j) {
            double h_pre_j = last_Cq_(t, j) / h_norm;
            double sig_o = mlstm_sigmoid(last_o_pre_(t, j));
            double h_j = sig_o * h_pre_j;
            last_h_(t + 1, j) = h_j;
            output(t, j) = h_j;
        }
    }
    return output;
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
Tensor MLSTMCell::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_input_.rows || grad_output.cols != hidden_size_) {
        throw std::invalid_argument("MLSTMCell::backward: grad_output shape mismatch");
    }
    size_t T = last_input_.rows;
    size_t d = hidden_size_;

    // Zero parameter grads.
    grad_W.fill(0.0);
    grad_b.fill(0.0);

    // Recurrent-state gradient buffers.
    Tensor g_C((T + 1) * d, d);   g_C.fill(0.0);
    Tensor g_N((T + 1) * d, d);   g_N.fill(0.0);
    Tensor g_m(T + 1, d);         g_m.fill(0.0);
    Tensor g_xh(T, input_size_ + d);
    g_xh.fill(0.0);

    // g_h[t] = grad_output[t] + recurrence contribution from t+1 (the
    // h_t portion of g_xh at t+1).  We start with grad_output and add
    // the recurrence in the loop.
    Tensor g_h(T, d);
    for (size_t i = 0; i < T * d; ++i) g_h.data[i] = grad_output.data[i];

    // Per-step scratch (T, d) — only row t is written, but the full
    // allocation is for simplicity.
    Tensor d_h_pre_buf(T, d);
    Tensor d_q_buf(T, d);
    Tensor d_k_buf(T, d);
    Tensor d_v_buf(T, d);
    Tensor d_i_prime_buf(T, d);
    Tensor d_f_prime_buf(T, d);
    Tensor d_i_pre_buf(T, d);
    Tensor d_f_pre_buf(T, d);
    Tensor d_o_pre_buf(T, d);
    d_h_pre_buf.fill(0.0);
    d_q_buf.fill(0.0);
    d_k_buf.fill(0.0);
    d_v_buf.fill(0.0);
    d_i_prime_buf.fill(0.0);
    d_f_prime_buf.fill(0.0);
    d_i_pre_buf.fill(0.0);
    d_f_pre_buf.fill(0.0);
    d_o_pre_buf.fill(0.0);

    // BPTT — walk backward in time.
    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        // Recurrence h contribution from t+1
        if (t + 1 < T) {
            for (size_t j = 0; j < d; ++j) {
                g_h(t, j) += g_xh(t + 1, input_size_ + j);
            }
        }

        double h_norm = last_h_norm_(t, 0);
        double inv_h_norm = 1.0 / h_norm;

        // ----- Step A: d_h_pre[i] = g_h(i) * σ(o_i)  and  d_h_norm -----
        for (size_t i = 0; i < d; ++i) {
            double sig_o = mlstm_sigmoid(last_o_pre_(t, i));
            d_h_pre_buf(t, i) = g_h(t, i) * sig_o;
        }
        double d_h_norm = 0.0;
        bool norm_active = (h_norm > 1.0 + 1e-12);
        if (norm_active) {
            for (size_t i = 0; i < d; ++i) {
                d_h_norm += d_h_pre_buf(t, i) * last_Cq_(t, i);
            }
            d_h_norm = -d_h_norm / (h_norm * h_norm);
        }

        // ----- Step B: add local dC[i,j] from h_pre retrieval -----
        // dC[i,j] = (d_h_pre[i] / h_norm) * q[j]
        for (size_t i = 0; i < d; ++i) {
            double d_h_pre_i_over_h = d_h_pre_buf(t, i) * inv_h_norm;
            for (size_t j = 0; j < d; ++j) {
                g_C((t + 1) * d + i, j) += d_h_pre_i_over_h * last_q_(t, j);
            }
        }

        // ----- Step C: add local dN[i,j] from the norm path -----
        if (norm_active) {
            for (size_t i = 0; i < d; ++i) {
                double q_i = last_q_(t, i);
                for (size_t j = 0; j < d; ++j) {
                    g_N((t + 1) * d + i, j) += d_h_norm * q_i * last_q_(t, j);
                }
            }
        }

        // ----- Step D: d_i', d_f' from the (now total) g_C(t+1) and g_N(t+1) -----
        // d_i'[j] = k[j] * sum_i g_C[i,j] * v[i]        = k[j] * (g_C^T v)[j]
        // d_f'[j] = sum_i g_C[i,j] * C_{t-1}[i,j] + sum_i g_N[i,j] * N_{t-1}[i,j]
        // d_k[j]  = i'[j] * (g_C^T v)[j]  +  i'[j] * (g_N^T k)[j]
        // d_v[i]  = sum_j g_C[i,j] * i'[j] * k[j]
        // d_q_from_C[k] = sum_i C[i,k] * d_h_pre[i]    = (C^T d_h_pre)[k]
        // d_q_from_norm[k] = 2 * d_h_norm * (N q)[k]    (when norm_active)
        for (size_t j = 0; j < d; ++j) {
            double i_p = last_i_prime_(t, j);
            double k_j = last_k_(t, j);

            // d_i', d_k, d_f'
            double gC_T_v_j = 0.0;
            double gN_T_k_j = 0.0;
            double d_f_prime_j = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double gC_ij = g_C((t + 1) * d + i, j);
                gC_T_v_j += gC_ij * last_v_(t, i);
                gN_T_k_j += g_N((t + 1) * d + i, j) * last_k_(t, i);
                d_f_prime_j += gC_ij * last_C_(t * d + i, j)
                             + g_N((t + 1) * d + i, j) * last_N_(t * d + i, j);
            }
            d_i_prime_buf(t, j) = k_j * gC_T_v_j;
            d_f_prime_buf(t, j) = d_f_prime_j;
            d_k_buf(t, j)       = i_p * gC_T_v_j + i_p * gN_T_k_j;
        }
        // d_v[i] = sum_j g_C[i,j] * i'[j] * k[j]
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += g_C((t + 1) * d + i, j) * last_i_prime_(t, j) * last_k_(t, j);
            }
            d_v_buf(t, i) = acc;
        }

        // ----- Step E: d_q (combined C and norm paths) -----
        for (size_t k = 0; k < d; ++k) {
            double C_T_dh_k = 0.0;
            for (size_t i = 0; i < d; ++i) {
                C_T_dh_k += last_C_((t + 1) * d + i, k) * d_h_pre_buf(t, i);
            }
            double d_q_k = C_T_dh_k;
            if (norm_active) {
                double Nq_k = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    Nq_k += last_N_((t + 1) * d + k, j) * last_q_(t, j);
                }
                d_q_k += 2.0 * d_h_norm * Nq_k;
            }
            d_q_buf(t, k) = d_q_k;
        }

        // ----- Step F: d_i_pre, d_f_pre, d_o_pre via stabilizer -----
        for (size_t j = 0; j < d; ++j) {
            double i_p = last_i_prime_(t, j);
            double f_p = last_f_prime_(t, j);
            double m_prev = last_m_(t, j);
            double m_t = last_m_(t + 1, j);
            double log_sigma_f = last_log_sigmoid_f_(t, j);
            double sigma_f = std::exp(log_sigma_f);
            double i_pre = last_i_pre_(t, j);
            double log_sigma_f_plus_m_prev = log_sigma_f + m_prev;
            double d_i_prime = d_i_prime_buf(t, j);
            double d_f_prime = d_f_prime_buf(t, j);

            // g_m(t+1, j) gets the direct -m_t contributions from i' and f'
            g_m(t + 1, j) += d_i_prime * (-i_p) + d_f_prime * (-f_p);

            double gm_t = g_m(t + 1, j);
            bool i_won = mlstm_i_pre_won_max(log_sigma_f_plus_m_prev, i_pre);

            // d_i_pre
            double d_i_pre = d_i_prime * i_p;
            if (i_won) d_i_pre += gm_t;

            // d_f_pre  (chain: d_f' -> d(log σ(f)) via exp(m_prev - m_t) -> d_f_pre)
            double d_f_pre = d_f_prime * std::exp(m_prev - m_t) * (1.0 - sigma_f);
            if (!i_won) d_f_pre += gm_t * (1.0 - sigma_f);

            // g_m(t, j) — recurrence carrier for m_{t-1}
            g_m(t, j) += d_f_prime * f_p;
            if (!i_won) g_m(t, j) += gm_t;

            d_i_pre_buf(t, j) = d_i_pre;
            d_f_pre_buf(t, j) = d_f_pre;

            // d_o_pre
            double sig_o = mlstm_sigmoid(last_o_pre_(t, j));
            double h_pre_j = last_Cq_(t, j) / h_norm;
            d_o_pre_buf(t, j) = g_h(t, j) * h_pre_j * sig_o * (1.0 - sig_o);

            // grad_b — bias gradients for all 6 blocks
            grad_b(0, 0 * d + j) += d_q_buf(t, j);
            grad_b(0, 1 * d + j) += d_k_buf(t, j);
            grad_b(0, 2 * d + j) += d_v_buf(t, j);
            grad_b(0, 3 * d + j) += d_i_pre;
            grad_b(0, 4 * d + j) += d_f_pre;
            grad_b(0, 5 * d + j) += d_o_pre_buf(t, j);
        }

        // ----- Step G: grad_W += outer([d_q, d_k, d_v, d_i_pre, d_f_pre, d_o_pre], xh) -----
        for (size_t k = 0; k < input_size_ + d; ++k) {
            double xh_tk = last_xh_(t, k);
            for (size_t j = 0; j < d; ++j) {
                grad_W(0 * d + j, k) += d_q_buf(t, j)       * xh_tk;
                grad_W(1 * d + j, k) += d_k_buf(t, j)       * xh_tk;
                grad_W(2 * d + j, k) += d_v_buf(t, j)       * xh_tk;
                grad_W(3 * d + j, k) += d_i_pre_buf(t, j)   * xh_tk;
                grad_W(4 * d + j, k) += d_f_pre_buf(t, j)   * xh_tk;
                grad_W(5 * d + j, k) += d_o_pre_buf(t, j)   * xh_tk;
            }
        }

        // ----- Step H: g_xh(t, k) = W^T @ [d_q, d_k, d_v, d_i_pre, d_f_pre, d_o_pre] -----
        for (size_t k = 0; k < input_size_ + d; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += W(0 * d + j, k) * d_q_buf(t, j);
                acc += W(1 * d + j, k) * d_k_buf(t, j);
                acc += W(2 * d + j, k) * d_v_buf(t, j);
                acc += W(3 * d + j, k) * d_i_pre_buf(t, j);
                acc += W(4 * d + j, k) * d_f_pre_buf(t, j);
                acc += W(5 * d + j, k) * d_o_pre_buf(t, j);
            }
            g_xh(t, k) += acc;
        }

        // ----- Step I: state-recurrence carrier for C and N -----
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double f_p = last_f_prime_(t, j);
                g_C(t * d + i, j) += f_p * g_C((t + 1) * d + i, j);
                g_N(t * d + i, j) += f_p * g_N((t + 1) * d + i, j);
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
void MLSTMCell::update_weights(double learning_rate) {
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= learning_rate * grad_W.data[i];
    }
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] -= learning_rate * grad_b.data[i];
    }
    // Re-apply forget-bias=1 convention (PyTorch-style).
    for (size_t j = 0; j < hidden_size_; ++j) {
        b(0, 4 * hidden_size_ + j) = 1.0;
    }
}

void MLSTMCell::zero_grad() {
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

std::vector<Tensor*> MLSTMCell::parameters() {
    return {&W, &b};
}

std::vector<Tensor*> MLSTMCell::gradients() {
    return {&grad_W, &grad_b};
}
