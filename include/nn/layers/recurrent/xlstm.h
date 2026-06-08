#ifndef XLSTM_H
#define XLSTM_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// xLSTM — Beck et al. 2024
//   "xLSTM: Extended Long Short-Term Memory"
//
// Two extensions of the original LSTM are introduced:
//
//   1. Exponential gating: replace sigmoid input/forget gates with
//      exp-normalized gates, using a per-head log-domain stabilizer
//      `m_t` to keep the exponentials from overflowing:
//
//         m_t       = max(log(σ(f_t)) + m_{t-1}, i_t)
//         f'_t      = exp(log(σ(f_t)) + m_{t-1} - m_t)   (forget prime, in [0,1])
//         i'_t      = exp(i_t - m_t)                     (input prime, in [0,1])
//
//   2. New memory structures:
//        * sLSTM — scalar memory: c_t is a vector in R^d, n_t is a vector in R^d
//        * mLSTM — matrix memory: C_t is a matrix in R^{d x d}, N_t in R^{d x d}
//
// In this implementation we focus on the sLSTM (scalar memory) variant — it is
// the conceptually simpler one and faithfully captures the "exponential gating"
// innovation. The mLSTM variant adds a covariance update rule that we leave
// as a follow-up.
//
// ----------------------------------------------------------------------------
// sLSTM per-timestep math (single head, head dim = hidden_size):
//
//   Inputs:
//     x_t           in R^{input_size}      (sequence input at step t)
//     h_{t-1}       in R^{hidden_size}     (hidden state from prev step)
//     c_{t-1}       in R^{hidden_size}     (cell state from prev step)
//     n_{t-1}       in R^{hidden_size}     (normalizer from prev step)
//     m_{t-1}       in R^{hidden_size}     (log-domain stabilizer from prev step)
//
//   Gate pre-activations (Dense convention: y = x W^T + b, but we
//   concatenate [x_t, h_{t-1}] for the input-projection W of shape
//   (4*hidden_size, input_size + hidden_size)):
//
//     [z_t, i_t, f_t, o_t] = W @ [x_t; h_{t-1}] + b   (each in R^{hidden_size})
//
//     where z is the candidate (no activation), i and f are input/forget
//     gate pre-activations, o is the output gate pre-activation.
//
//   Note: following the official sLSTM implementation we apply sigmoid to
//   the forget gate pre-activation *before* taking its log. This keeps
//   f in (0, 1) and log(σ(f)) < 0, so exp(log(σ(f)) + m_prev - m_t)
//   stays numerically small.
//
//   Stabilizer update (log-domain max trick):
//     m_t = max(log(σ(f_t)) + m_{t-1}, i_t)        in R^{hidden_size}
//
//   Gate "primes" (stabilized exponentials in [0, 1]):
//     f'_t = exp(log(σ(f_t)) + m_{t-1} - m_t)      in R^{hidden_size}
//     i'_t = exp(i_t - m_t)                        in R^{hidden_size}
//
//   Cell state update (additive, with exponential gates):
//     c_t = f'_t ⊙ c_{t-1} + i'_t ⊙ tanh(z_t)     in R^{hidden_size}
//
//   Normalizer update (running sum of input primes):
//     n_t = f'_t ⊙ n_{t-1} + i'_t                 in R^{hidden_size}
//
//   Hidden state (pre-output is normalized cell; output is gate-modulated):
//     h_t = sigmoid(o_t) ⊙ tanh(c_t / n_t)        in R^{hidden_size}
//
// ----------------------------------------------------------------------------
// Initialization convention (paper, §2.3 and official codebase):
//   * Forget-gate bias initialized to a small positive value (we use 1.0 — the
//     canonical "open the gate" default, matching what most LSTM code uses
//     and PyTorch's default forget_bias=1.0).
//   * Input-gate bias initialized to 0.
//   * Output-gate and candidate biases initialized to 0.
//   * Weights initialized with Xavier uniform scaling.
//
// ----------------------------------------------------------------------------
// Shape convention: forward takes a 2D input (T, input_size) and returns
// (T, hidden_size). The h-state, c-state, n-state, m-state are all carried
// internally as (T+1, hidden_size) caches (with row 0 = initial state 0).
// BPTT traverses these caches backward in `backward`.
//
// This class is the "block"-level layer (single head). Stack multiple for
// multi-head / multi-layer models, or use XLSTMBlock (see xLSTMBlock below)
// for the residual-block wrapper used in the xLSTM paper.
// ============================================================================

class SLSTMCell : public Layer {
public:
    // input_size: feature dim of x_t
    // hidden_size: feature dim of h_t, c_t, n_t, m_t
    SLSTMCell(size_t input_size, size_t hidden_size);

    // Forward pass on a full sequence.
    // input: (T, input_size)  ->  output: (T, hidden_size)
    Tensor forward(const Tensor& input) override;
    // Backward pass — grad_output: (T, hidden_size), returns grad_input: (T, input_size)
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W; }
    Tensor get_gradients() const override { return grad_W; }
    std::string name() const override { return "SLSTMCell"; }

    // Accessors for tests
    size_t input_size() const { return input_size_; }
    size_t hidden_size() const { return hidden_size_; }

    // Public parameters / gradients
    Tensor W;          // (4*hidden_size, input_size + hidden_size)
    Tensor b;          // (1, 4*hidden_size) — bias for [z, i, f, o]
    Tensor grad_W;     // (4*hidden_size, input_size + hidden_size)
    Tensor grad_b;     // (1, 4*hidden_size)

private:
    size_t input_size_;
    size_t hidden_size_;

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;         // (T, input_size)
    Tensor last_h_;             // (T+1, hidden_size)   h_0..h_T
    Tensor last_c_;             // (T+1, hidden_size)   c_0..c_T
    Tensor last_n_;             // (T+1, hidden_size)   n_0..n_T
    Tensor last_m_;             // (T+1, hidden_size)   m_0..m_T
    Tensor last_xh_;            // (T, input_size+hidden_size)  concatenated [x;h_{t-1}] input to W
    Tensor last_z_;             // (T, hidden_size)   candidate (raw, no activation)
    Tensor last_i_pre_;         // (T, hidden_size)   input gate pre-activation
    Tensor last_f_pre_;         // (T, hidden_size)   forget gate pre-activation (raw)
    Tensor last_o_pre_;         // (T, hidden_size)   output gate pre-activation
    Tensor last_i_prime_;       // (T, hidden_size)   exp(i - m)
    Tensor last_f_prime_;       // (T, hidden_size)   exp(log(σ(f)) + m_prev - m)
    Tensor last_log_sigmoid_f_; // (T, hidden_size)   log(σ(f_pre))

    // Internal accumulators for input gradient and parameter gradients,
    // not exposed to users. Set in backward, used in update_weights.

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double log_sigmoid_stable(double x) {
        // log(σ(x)) for any x — log(sigmoid(x)).
        // Numerically stable: log(σ(x)) = -softplus(-x) = log(1 + exp(x)) - x  if x < 0
        //                                     = -log(1 + exp(-x))        if x >= 0
        if (x >= 0.0) {
            return -std::log(1.0 + std::exp(-x));
        } else {
            double z = std::log(1.0 + std::exp(x));
            return z - x;
        }
    }
};

#endif // XLSTM_H
