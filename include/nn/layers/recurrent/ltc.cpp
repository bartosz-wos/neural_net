#include "nn/layers/recurrent/ltc.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// LTC (Liquid Time-Constant Networks) — Hasani, Lechner, Amini, Rusch, Grosu 2021
//   "Liquid Time-Constant Networks" (AAAI 2021)
//   https://arxiv.org/abs/2006.04439
//
// Per-neuron, per-time step (single-step BPTT) math:
//
//   A_t[i]     = W_ih[i]·x_t + W_hh[i]·h_{t-1} + b[i]
//   z_τ_t[i]   = W_tx[i]·x_t + W_th[i]·h_{t-1} + b_τ[i]
//   τ_t[i]     = exp(log_τ_base[i]) · softplus(z_τ_t[i])    in R+
//   g_t[i]     = exp(-1/τ_t[i])                              in (0,1)
//   α_t[i]     = 1 - g_t[i]                                 in (0,1)
//   h_t[i]     = tanh(g_t[i]·h_{t-1}[i] + α_t[i]·A_t[i])
//
// Equivalent reformulation (used for a clean single backward chain):
//   h_t_input[i] = A_t[i] + g_t[i]·(h_{t-1}[i] - A_t[i])
//                 = (1-g_t[i])·A_t[i] + g_t[i]·h_{t-1}[i]
//   => dh_t_input/dg_t[i] = h_{t-1}[i] - A_t[i]    (single clean gradient)
//   => dh_t_input/dA_t[i] = 1 - g_t[i] = α_t[i]
//
// Backward (full BPTT, including the 3 paths to h_{t-1} via g, A, z_τ):
//
//   Let g_out = dL/dh_t (the gradient coming from outside).
//   dh_in   = g_out · (1 - h_t^2)                          (tanh derivative)
//
//   grad_g     = dh_in · (h_{t-1}[i] - A_t[i])
//   grad_A     = dh_in · α_t[i]                            = dh_in · (1 - g_t[i])
//
//   dg/dτ      = exp(-1/τ) · (1/τ^2) = g_t / τ_t^2
//   grad_τ     = grad_g · (g_t / τ_t^2)
//
//   dτ/dz_τ    = τ_t · (1 - sigmoid(z_τ_t))                (softplus derivative)
//   dτ/dlog_τ_base = τ_t
//   grad_z_τ   = grad_τ · τ_t · (1 - sigmoid(z_τ_t))
//
//   Param gradients:
//     grad_W_ih[i,k]  += grad_A · x_t[k]
//     grad_W_hh[i,j]  += grad_A · h_{t-1}[j]
//     grad_b[i]       += grad_A
//     grad_W_tx[i,k]  += grad_z_τ · x_t[k]
//     grad_W_th[i,j]  += grad_z_τ · h_{t-1}[j]
//     grad_b_τ[i]     += grad_z_τ
//     grad_log_τ_base[i] += grad_τ · τ_t
//
//   Input gradient at time t (for grad_input[t][k]):
//     dL/dx_t[k] += grad_A · W_ih[i,k] + grad_z_τ · W_tx[i,k]    (summed over i)
//
//   BPTT carrier (gradient wrt h_{t-1}[j]):
//     Three contributions from each (t, i):
//       (a) DIRECT: dh_t_input/d_h_{t-1}[j] via the g_t · h_{t-1}[i] term.
//           Only nonzero when j == i: dh_in · g_t[i].
//       (b) VIA A: dh_t_input/d_h_{t-1}[j] via A_t = W_hh @ h_{t-1}.
//           grad_A · W_hh[i, j].
//       (c) VIA z_τ: dh_t_input/d_h_{t-1}[j] via z_τ_t = W_th @ h_{t-1}.
//           grad_z_τ · W_th[i, j].
// ============================================================================

namespace {

inline double ltc_sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    double ez = std::exp(z);
    return ez / (1.0 + ez);
}

// Numerically stable softplus: max(z, 0) + log(1 + exp(-|z|))
inline double ltc_softplus(double z) {
    if (z > 0.0) return z + std::log1p(std::exp(-z));
    return std::log1p(std::exp(z));
}

}  // namespace

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
LTC::LTC(size_t input_dim, size_t hidden_size, double tau_base_init)
    : input_dim_(input_dim), hidden_size_(hidden_size),
      W_ih_(Tensor::random(hidden_size, input_dim, 0.1)),
      W_hh_(Tensor::random(hidden_size, hidden_size, 0.1) * 0.1),
      b_(Tensor::zeros(hidden_size, 1)),
      W_tx_(Tensor::random(hidden_size, input_dim, 0.1) * 0.1),
      W_th_(Tensor::random(hidden_size, hidden_size, 0.1) * 0.1),
      b_t_(Tensor::zeros(hidden_size, 1)) {
    if (input_dim == 0 || hidden_size == 0)
        throw std::invalid_argument("LTC requires input_dim > 0 and hidden_size > 0");
    if (!(tau_base_init > 0.0))
        throw std::invalid_argument("LTC requires tau_base_init > 0");

    // log_tau_base: init so tau_base = tau_base_init (positive unconstrained)
    log_tau_base_ = Tensor(hidden_size, 1);
    for (size_t i = 0; i < hidden_size; ++i)
        log_tau_base_[i][0] = std::log(tau_base_init);

    // Zero-init gradient buffers
    grad_W_ih_    = Tensor::zeros(hidden_size, input_dim);
    grad_W_hh_    = Tensor::zeros(hidden_size, hidden_size);
    grad_b_       = Tensor::zeros(hidden_size, 1);
    grad_W_tx_    = Tensor::zeros(hidden_size, input_dim);
    grad_W_th_    = Tensor::zeros(hidden_size, hidden_size);
    grad_b_t_     = Tensor::zeros(hidden_size, 1);
    grad_log_tau_base_ = Tensor::zeros(hidden_size, 1);
}

// ----------------------------------------------------------------------------
// Forward pass
// ----------------------------------------------------------------------------
Tensor LTC::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("LTC forward: input cols mismatch");

    const size_t T = input.rows;

    // Allocate output and caches
    Tensor h(T, hidden_size_);
    cache_x_         = Tensor(T, input_dim_);
    cache_A_         = Tensor(T, hidden_size_);
    cache_z_t_       = Tensor(T, hidden_size_);
    cache_tau_       = Tensor(T, hidden_size_);
    cache_g_         = Tensor(T, hidden_size_);
    cache_alpha_     = Tensor(T, hidden_size_);
    cache_pre_tanh_  = Tensor(T, hidden_size_);
    cache_h_         = Tensor(T, hidden_size_);

    // h_prev starts at zero (h_{-1} = 0)
    std::vector<double> h_prev(hidden_size_, 0.0);

    for (size_t t = 0; t < T; ++t) {
        // Snapshot x_t for backward
        for (size_t k = 0; k < input_dim_; ++k)
            cache_x_[t][k] = input[t][k];

        // Step 1: A_t = W_ih @ x_t + W_hh @ h_prev + b
        // Step 2: z_τ_t = W_tx @ x_t + W_th @ h_prev + b_τ
        for (size_t i = 0; i < hidden_size_; ++i) {
            double a = b_[i][0];
            double z = b_t_[i][0];
            for (size_t k = 0; k < input_dim_; ++k) {
                a += W_ih_[i][k] * input[t][k];
                z += W_tx_[i][k] * input[t][k];
            }
            for (size_t j = 0; j < hidden_size_; ++j) {
                a += W_hh_[i][j] * h_prev[j];
                z += W_th_[i][j] * h_prev[j];
            }
            cache_A_[t][i]   = a;
            cache_z_t_[t][i] = z;
        }

        // Step 3: τ_t = exp(log_τ_base) · softplus(z_τ_t)
        // Step 4: g_t = exp(-1/τ_t), α_t = 1 - g_t
        for (size_t i = 0; i < hidden_size_; ++i) {
            double tau_base_i = std::exp(log_tau_base_[i][0]);
            double tau_i      = tau_base_i * ltc_softplus(cache_z_t_[t][i]);
            double g_i        = std::exp(-1.0 / tau_i);
            cache_tau_[t][i]   = tau_i;
            cache_g_[t][i]     = g_i;
            cache_alpha_[t][i] = 1.0 - g_i;
        }

        // Step 5: h_t = tanh(g_t · h_prev + α_t · A_t)
        for (size_t i = 0; i < hidden_size_; ++i) {
            double z_in = cache_g_[t][i] * h_prev[i]
                        + cache_alpha_[t][i] * cache_A_[t][i];
            cache_pre_tanh_[t][i] = z_in;
            double h_i = std::tanh(z_in);
            h[t][i] = h_i;
            cache_h_[t][i] = h_i;
            h_prev[i] = h_i;
        }
    }

    return h;
}

// ----------------------------------------------------------------------------
// Backward pass (full BPTT)
// ----------------------------------------------------------------------------
Tensor LTC::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != hidden_size_)
        throw std::invalid_argument("LTC backward: grad_output cols mismatch");

    const size_t T = grad_output.rows;
    if (cache_x_.rows == 0 || cache_x_.rows != T)
        throw std::logic_error("LTC backward: forward must be called first");

    // Zero gradients at the start of each backward pass
    zero_grad();

    // grad_h[t] = grad_output[t] initially, then accumulates contributions from
    // t+1, t+2, ... via the BPTT carrier (the "future" gradient flow).
    Tensor grad_h(T, hidden_size_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < hidden_size_; ++i)
            grad_h[t][i] = grad_output[t][i];

    Tensor grad_input(T, input_dim_);

    // Iterate backward in time. h_prev at iteration t = h_{t-1} (or 0 if t==0).
    std::vector<double> h_prev(hidden_size_, 0.0);
    // Initialize h_prev for the FIRST iteration (t = T-1): h_prev = h_{T-2} (or 0 if T==1)
    // Actually h_prev at start of t = h_{t-1}. So for t = T-1, h_prev = h_{T-2}.
    // For t = 0, h_prev = h_{-1} = 0.
    // Set BEFORE the loop:
    for (size_t i = 0; i < hidden_size_; ++i) {
        h_prev[i] = (T >= 2) ? cache_h_[T-2][i] : 0.0;
    }

    for (int t = (int)T - 1; t >= 0; --t) {
        for (size_t i = 0; i < hidden_size_; ++i) {
            // 1) tanh derivative
            double h_t = cache_h_[t][i];
            double dh_in = grad_h[t][i] * (1.0 - h_t * h_t);

            // 2) Unified grad wrt g_t using h_t_input = A_t + g_t·(h_{t-1} - A_t)
            //    so dh_t_input/dg = h_{t-1}[i] - A_t[i]
            double A_t = cache_A_[t][i];
            double g_t = cache_g_[t][i];
            double alpha_t = cache_alpha_[t][i];
            double tau_t = cache_tau_[t][i];
            double z_tau_t = cache_z_t_[t][i];

            double grad_g = dh_in * (h_prev[i] - A_t);

            // 3) grad wrt A_t (independent of g coupling)
            double grad_A = dh_in * alpha_t;

            // 4) grad wrt τ_t via g
            //    g = exp(-1/τ), dg/dτ = g / τ²
            double grad_tau = grad_g * (g_t / (tau_t * tau_t));

            // 5) grad wrt z_τ_t and log_τ_base via softplus
            //    dτ/dz_τ = τ_base · sigmoid(z_τ) = τ · sigmoid(z_τ) / softplus(z_τ)
            //    We use the closed form: τ = τ_base · sp(z), so dτ/dz = τ_base · σ(z).
            //    Equivalently: τ · σ(z) / sp(z), but the τ_base form is cleaner.
            //    dτ/dlog_τ_base = τ_base · sp(z) = τ
            double tau_base_i = std::exp(log_tau_base_[i][0]);
            double sig_z = ltc_sigmoid(z_tau_t);
            double grad_z_tau = grad_tau * tau_base_i * sig_z;
            double grad_log_tb = grad_tau * tau_t;

            // 6) Accumulate parameter gradients
            for (size_t k = 0; k < input_dim_; ++k) {
                grad_W_ih_[i][k] += grad_A * cache_x_[t][k];
                grad_W_tx_[i][k] += grad_z_tau * cache_x_[t][k];
            }
            for (size_t j = 0; j < hidden_size_; ++j) {
                grad_W_hh_[i][j] += grad_A * h_prev[j];
                grad_W_th_[i][j] += grad_z_tau * h_prev[j];
            }
            grad_b_[i][0]    += grad_A;
            grad_b_t_[i][0]  += grad_z_tau;
            grad_log_tau_base_[i][0] += grad_log_tb;

            // 7) Accumulate grad_input[t][k] (summed over i)
            for (size_t k = 0; k < input_dim_; ++k) {
                grad_input[t][k] += grad_A * W_ih_[i][k] + grad_z_tau * W_tx_[i][k];
            }

            // 8) BPTT carrier to h_{t-1}[j] (3 contributions)
            if (t > 0) {
                for (size_t j = 0; j < hidden_size_; ++j) {
                    // (a) direct via g_t · h_{t-1}[i]  (only when j == i)
                    double contrib_g = (j == i) ? dh_in * g_t : 0.0;
                    // (b) via A_t = W_hh @ h_{t-1}
                    double contrib_A = grad_A * W_hh_[i][j];
                    // (c) via z_τ_t = W_th @ h_{t-1}
                    double contrib_z = grad_z_tau * W_th_[i][j];
                    grad_h[t-1][j] += contrib_g + contrib_A + contrib_z;
                }
            }
        }

        // Update h_prev for the NEXT iteration (next t goes one step earlier).
        // h_prev at start of next iter (t-1) = h_{t-1} = cache_h_[t-1] (or 0 if t==0).
        // After this update, the next iteration t-1 needs h_prev = h_{t-2}.
        // So set h_prev[i] = (t-1 > 0) ? cache_h_[t-2] : 0.
        // Equivalently: if t == 1 (next t is 0), h_prev becomes 0.
        for (size_t i = 0; i < hidden_size_; ++i) {
            h_prev[i] = (t >= 2) ? cache_h_[t-2][i] : 0.0;
        }
    }

    return grad_input;
}

// ----------------------------------------------------------------------------
// Weight update (plain SGD step; the optimizer that wraps us is responsible
// for adaptive moments / momentum if any).
// ----------------------------------------------------------------------------
void LTC::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] -= learning_rate * gW[i][j];
    };
    sgd(W_ih_,    grad_W_ih_);
    sgd(W_hh_,    grad_W_hh_);
    sgd(b_,       grad_b_);
    sgd(W_tx_,    grad_W_tx_);
    sgd(W_th_,    grad_W_th_);
    sgd(b_t_,     grad_b_t_);
    sgd(log_tau_base_, grad_log_tau_base_);
}

// ----------------------------------------------------------------------------
// Zero gradients
// ----------------------------------------------------------------------------
void LTC::zero_grad() {
    auto zero = [](Tensor& t) {
        for (size_t i = 0; i < t.rows; ++i)
            for (size_t j = 0; j < t.cols; ++j)
                t[i][j] = 0.0;
    };
    zero(grad_W_ih_);
    zero(grad_W_hh_);
    zero(grad_b_);
    zero(grad_W_tx_);
    zero(grad_W_th_);
    zero(grad_b_t_);
    zero(grad_log_tau_base_);
}

// ----------------------------------------------------------------------------
// Parameter / gradient discovery
// ----------------------------------------------------------------------------
std::vector<Tensor*> LTC::parameters() {
    return {&W_ih_, &W_hh_, &b_, &W_tx_, &W_th_, &b_t_, &log_tau_base_};
}

std::vector<Tensor*> LTC::gradients() {
    return {&grad_W_ih_, &grad_W_hh_, &grad_b_, &grad_W_tx_, &grad_W_th_, &grad_b_t_, &grad_log_tau_base_};
}