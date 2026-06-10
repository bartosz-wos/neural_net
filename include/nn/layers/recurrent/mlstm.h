#ifndef MLSTM_H
#define MLSTM_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// mLSTM — Beck et al. 2024
//   "xLSTM: Extended Long Short-Term Memory" (matrix-memory variant)
//
// mLSTM is the *fully-parallelizable* sibling of sLSTM. Where sLSTM keeps
// a scalar cell state c_t ∈ R^d, mLSTM keeps a matrix cell C_t ∈ R^{d×d}
// with a *covariance update rule*:
//
//   C_t = f'_t · C_{t-1} + i'_t · outer(v_t, k_t)        (shape: d×d)
//
// and a matching matrix normalizer
//
//   N_t = f'_t · N_{t-1} + i'_t · outer(k_t, k_t)        (shape: d×d)
//
// where f'_t and i'_t are the same stabilized exponential gates as sLSTM:
//
//   m_t       = max(log σ(f_pre_t) + m_{t-1}, i_pre_t)
//   f'_t      = exp(log σ(f_pre_t) + m_{t-1} - m_t)    ∈ (0, 1]
//   i'_t      = exp(i_pre_t - m_t)                      ∈ (0, 1]
//
// The hidden pre-output is computed as a *normalized retrieval* from C_t:
//
//   h_norm_t   = max(1, q_t^T · N_t · q_t)              (scalar, with floor)
//   h_pre_t    = (C_t · q_t) / h_norm_t                  ∈ R^d
//   h_t        = σ(o_pre_t) ⊙ h_pre_t                    ∈ R^d
//
// The normalization makes the output a *linear-attention-style* query:
//   C_t = sum_{s<=t} (∏_{s'<s, s'>=s+1} f'_{s'}) · i'_s · outer(v_s, k_s)
//   N_t = sum_{s<=t} (∏_{s'<s, s'>=s+1} f'_{s'}) · i'_s · outer(k_s, k_s)
// so the un-normalized h_pre is sum_s α_s · v_s · <k_s, q>  (linear attention!)
// and the denominator makes the output scale-invariant.
//
// This is the same trick as linear attention with the "denominator" trick
// (Katharopoulos et al. 2020), with the addition of the exponential decay
// through f'_t and the input gate i'_t. It is also equivalent to a fast
// weight programmer (Schmidhuber 1992) with input-dependent learning rate
// and decay — see the xLSTM paper §3.2 for the formal equivalence.
//
// ----------------------------------------------------------------------------
// Projections (combined W·[x;h_{t-1}] + b of shape (6d, input_size+hidden_size)):
//
//   [q, k, v, i, f, o]_t  =  W · [x_t; h_{t-1}] + b
//
// with bias layout [q, k, v, i, f, o] (each of size hidden_size = d).
// Forget-bias init = 1 (PyTorch convention), other biases = 0.
//
// Initial state convention: C_0 = 0, N_0 = 0. h_0 = 0, m_0 = 0.
// (h_norm_t is always >= 1 by the max, so we never divide by 0.)
//
// ----------------------------------------------------------------------------
// State shape: C_t and N_t are d×d. For a typical "transformer-style" model
// dim of e.g. 256, that's 256² = 65536 floats per timestep per state matrix
// → 2 of them = 131072 floats = 524 KB per layer per timestep in fp32. This
// is the "quadratic memory" of mLSTM and the main practical drawback vs
// sLSTM. We don't optimize for that here — the goal is a *correct* reference
// implementation whose gradient check passes at machine precision.
//
// ----------------------------------------------------------------------------
// BPTT: the full per-step backward is implemented. The key non-trivial
// pieces are:
//
//   - Outer-product C/N path: dL/dk, dL/dv from dC, dN.
//   - Normalizer path: dL/d(q) and dL/d(N) through h_norm = max(1, q^T N q).
//     dL/d(q) accumulates 2 · (dL/d h_norm) · (N · q) when h_norm > 1.
//     dL/d(N) accumulates (dL/d h_norm) · outer(q, q) when h_norm > 1.
//   - Stabilizer m_t: same as sLSTM (indicator max + carrier from f' / i').
//
// Backward returns the gradient w.r.t. the input x_t only (the h_{t-1}
// gradient flows through the recurrence and gets exposed via grad_xh —
// we slice off the first input_size columns at the end).
// ============================================================================

class MLSTMCell : public Layer {
public:
    // input_size: feature dim of x_t
    // hidden_size: feature dim of h_t, also dim of the C_t, N_t matrices
    //              (so C_t, N_t are hidden_size x hidden_size)
    MLSTMCell(size_t input_size, size_t hidden_size);

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
    std::string name() const override { return "MLSTMCell"; }

    // Accessors for tests
    size_t input_size() const { return input_size_; }
    size_t hidden_size() const { return hidden_size_; }

    // Matrix-state accessors (for tests / debugging).
    //   last_C(t, i, j) = C_t[i, j]   (t = 0..T)
    //   last_N(t, i, j) = N_t[i, j]   (t = 0..T)
    // where t=0 is the initial state (zero) and t=T is the final state.
    // Returns the row-major (hidden_size, hidden_size) block at step t.
    Tensor last_C(size_t t) const {
        size_t d = hidden_size_;
        Tensor M(d, d);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                M(i, j) = last_C_.data[t * d * d + i * d + j];
        return M;
    }
    Tensor last_N(size_t t) const {
        size_t d = hidden_size_;
        Tensor M(d, d);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                M(i, j) = last_N_.data[t * d * d + i * d + j];
        return M;
    }

    // Public parameters / gradients
    Tensor W;          // (6*hidden_size, input_size + hidden_size)
                       // Layout: rows [0..d) = q, [d..2d) = k, [2d..3d) = v,
                       //        [3d..4d) = i, [4d..5d) = f, [5d..6d) = o
    Tensor b;          // (1, 6*hidden_size)  — bias for [q, k, v, i, f, o]
    Tensor grad_W;     // (6*hidden_size, input_size + hidden_size)
    Tensor grad_b;     // (1, 6*hidden_size)

private:
    size_t input_size_;
    size_t hidden_size_;

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;         // (T, input_size)
    Tensor last_h_;             // (T+1, hidden_size)   h_0..h_T
    Tensor last_C_;             // ((T+1)*hidden_size, hidden_size)  C_0..C_T
                                //   index: t*hidden_size + i, col j   =>   C_t[i,j]
    Tensor last_N_;             // ((T+1)*hidden_size, hidden_size)  N_0..N_T (same indexing)
    Tensor last_m_;             // (T+1, hidden_size)   m_0..m_T  (per-channel stabilizer)
    Tensor last_xh_;            // (T, input_size+hidden_size)  [x_t; h_{t-1}]
    Tensor last_q_;             // (T, hidden_size)   q_t
    Tensor last_k_;             // (T, hidden_size)   k_t
    Tensor last_v_;             // (T, hidden_size)   v_t
    Tensor last_i_pre_;         // (T, hidden_size)
    Tensor last_f_pre_;         // (T, hidden_size)
    Tensor last_o_pre_;         // (T, hidden_size)
    Tensor last_i_prime_;       // (T, hidden_size)   i'_t = exp(i - m_t)
    Tensor last_f_prime_;       // (T, hidden_size)   f'_t = exp(log σ(f) + m_prev - m_t)
    Tensor last_log_sigmoid_f_; // (T, hidden_size)   log σ(f_pre_t)
    Tensor last_h_norm_;        // (T, 1)             h_norm_t = max(1, q^T N q)
    Tensor last_Cq_;            // (T, hidden_size)   C_t · q_t  (pre-output un-normalized)

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double log_sigmoid_stable(double x) {
        // log σ(x).  Stable:
        //   x >= 0:  log σ(x) = -log(1+exp(-x))
        //   x <  0:  log σ(x) =  x - log(1+exp(x))
        if (x >= 0.0) {
            return -std::log(1.0 + std::exp(-x));
        } else {
            return x - std::log(1.0 + std::exp(x));
        }
    }
};

#endif // MLSTM_H
