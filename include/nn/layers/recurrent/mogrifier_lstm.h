#ifndef MOGRIFIER_LSTM_H
#define MOGRIFIER_LSTM_H

#include "../../core/layer.h"
#include <vector>

// ============================================================================
// Mogrifier LSTM — Melis et al. ICLR 2020
//   "Mogrifier LSTM" (https://arxiv.org/abs/1909.09592)
//
// A vanilla LSTM whose input x_t and previous hidden state h_{t-1} are passed
// through r rounds of MULTIPLICATIVE GATING before the standard 4-gate LSTM
// computation. The paper's central finding is that this small architectural
// change closes most of the quality gap between LSTM and Transformer language
// models at small-to-moderate scale.
//
// ----------------------------------------------------------------------------
// Per-timestep forward (with `r` rounds, default r = 2)
//
//   Round 1 (odd):  x_t        = 2 σ(h_{t-1} @ Q_0) ⊙ x_t
//   Round 2 (even): h_{t-1}    = 2 σ(x_t @ R_0)      ⊙ h_{t-1}
//   Round 3 (odd):  x_t        = 2 σ(h_{t-1} @ Q_1) ⊙ x_t
//   Round 4 (even): h_{t-1}    = 2 σ(x_t @ R_1)      ⊙ h_{t-1}
//   ...
//
//   Then standard LSTM gates on the MOGRIFIED [x_t; h_{t-1}]:
//
//     [z_i, z_f, z_g, z_o] = [x_t; h_{t-1}] @ W^T + b     ∈ R^{4·hidden}
//     i = σ(z_i), f = σ(z_f), g = tanh(z_g), o = σ(z_o)
//     c_t = f ⊙ c_{t-1} + i ⊙ g
//     h_t = o ⊙ tanh(c_t)
//
// The convention `Q ∈ R^{hidden × input}`, `R ∈ R^{input × hidden}` follows
// the standard Dense layout (out, in). With this layout, `x_t @ Q^T` would be
// wrong direction — we use the convention `out = in @ W^T` so for x-modifying
// rounds we compute `gate = h_{t-1} @ Q^T` (shape (input,)). Same shape as x_t
// so element-wise multiply works. (Equivalently: cache `Q^T` and use
// `gate = Q^T @ h_{t-1}`. We choose the @ Q^T form to keep W in Dense layout.)
//
// When `num_rounds = 0` the layer is a vanilla LSTM (no mogrification).
//
// ----------------------------------------------------------------------------
// Parameters
//
//   Q_list_  has ceil(num_rounds/2) tensors of shape (hidden, input)
//   R_list_  has floor(num_rounds/2) tensors of shape (input, hidden)
//   W_       (4·hidden, input + hidden)  — combined LSTM input projection
//   b_       (4·hidden,)                 — combined LSTM bias
//
// For num_rounds = 2 (paper default):  2 Q + 1 R + 1 W + 1 b = 5 tensors
// For num_rounds = 0 (vanilla):       0 + 0 + 1 W + 1 b = 2 tensors
// ----------------------------------------------------------------------------

class MogrifierLSTM : public Layer {
public:
    MogrifierLSTM(size_t input_dim, size_t hidden_size, size_t num_rounds = 2);

    Tensor forward(const Tensor& input) override;          // single-step (stateful)
    Tensor forward_sequence(const Tensor& seq);            // full-sequence (unrolled)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;

    Tensor get_weights() const override { return W_; }
    Tensor get_gradients() const override { return grad_W_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "MogrifierLSTM"; }

    void zero_grad() override;
    void init_weights();

    // Accessors (used by FD gradient checks).
    size_t input_dim() const { return input_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    size_t num_rounds() const { return num_rounds_; }

    std::vector<Tensor>& Q_list() { return Q_list_; }
    std::vector<Tensor>& R_list() { return R_list_; }
    Tensor& W() { return W_; }
    Tensor& b() { return b_; }

    std::vector<Tensor>& grad_Q_list() { return grad_Q_list_; }
    std::vector<Tensor>& grad_R_list() { return grad_R_list_; }
    Tensor& grad_W() { return grad_W_; }
    Tensor& grad_b() { return grad_b_; }

    // Cached mogrifier inputs from last forward_sequence: shape (T, input).
    // After round 0 (identity), round 1 (x-mogrified), round 2 (h-mogrified), etc.
    const std::vector<Tensor>& last_mog_x() const { return cached_mog_x_; }
    const std::vector<Tensor>& last_mog_h() const { return cached_mog_h_; }

    // Reset only the recurrent hidden/cell state (for clean FD re-evaluations).
    void reset_state();

private:
    size_t input_dim_, hidden_size_, num_rounds_;

    // Parameters
    std::vector<Tensor> Q_list_;          // ceil(r/2) entries of shape (hidden, input)
    std::vector<Tensor> R_list_;          // floor(r/2) entries of shape (input, hidden)
    Tensor W_;                            // (4·hidden, input + hidden)
    Tensor b_;                            // (4·hidden,)

    // Gradients (shape-matched)
    std::vector<Tensor> grad_Q_list_;
    std::vector<Tensor> grad_R_list_;
    Tensor grad_W_;
    Tensor grad_b_;

    // Stateful recurrent state (last h, c)
    Tensor h_;                            // (1, hidden)
    Tensor c_;                            // (1, hidden)

    // Caches (populated by forward_sequence)
    Tensor inputs_;                       // (T, input_dim)
    Tensor h_states_;                     // (T+1, hidden) — h_0..h_T
    Tensor c_states_;                     // (T+1, hidden) — c_0..c_T
    std::vector<Tensor> cached_mog_x_;    // per round (T, input_dim): index 0 = raw input, k = after round k
    std::vector<Tensor> cached_mog_h_;    // per round (T, hidden): index 0 = raw h_prev, k = after round k
    Tensor cached_gates_;    // (T, 4·hidden) — per timestep [z_i, z_f, z_g, z_o], concatenated along cols
    Tensor gh_carrier_;       // (T+1, hidden) — per timestep carrier for grad_h_{t-1} from mogrifier

    Tensor last_output_;                  // (1, hidden)
    bool cached_;                         // true after forward_sequence populates caches
};

#endif