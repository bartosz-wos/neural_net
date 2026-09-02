#ifndef MIN_LSTM_H
#define MIN_LSTM_H

#include "../../core/layer.h"

// MinLSTM — Feng et al. 2024 ("Were RNNs All We Needed?", arXiv:2410.01201, Alg. 3)
//
// The minimalist LSTM companion to MinGRU. Both the forget and the input gate
// are computed from the input alone (no hidden-to-hidden weights), and are then
// NORMALIZED to sum to 1 so the recurrence is a convex combination — the paper's
// key stability trick, which makes the hidden-state scale time-independent.
//
// Forward (per timestep, element-wise per hidden channel):
//   z_f_t = W_f · x_t + b_f      f_t = sigmoid(z_f_t)
//   z_i_t = W_i · x_t + b_i      i_t = sigmoid(z_i_t)
//   c_t   = W_h · x_t + b_h                          (candidate, linear)
//   s_t   = f_t + i_t
//   f'_t  = f_t / s_t            i'_t = i_t / s_t     (f'_t + i'_t == 1)
//   h_t   = f'_t ⊙ h_{t-1} + i'_t ⊙ c_t
//
// Contrast with MinGRU, which uses a single gate:
//   h_t = (1 - g_t) ⊙ h_{t-1} + g_t ⊙ c_t
// MinLSTM learns retention and write strengths independently, then normalizes.
//
// Since neither gate depends on h_{t-1}, the recurrence is associative:
//   ((a_i, b_i), (a_j, b_j)) -> (a_i·a_j, a_i·b_j + b_i)
// so a parallel scan computes the same h_t in O(log T) sequential steps.
//
// Parameters (all input-only projections):
//   W_f, W_i, W_h : (input_dim, hidden_size)
//   b_f, b_i, b_h : (1, hidden_size)
class MinLSTM : public Layer {
public:
    MinLSTM(size_t input_dim, size_t hidden_size);

    Tensor forward(const Tensor& input) override;      // single-step (stateful)
    Tensor forward_sequence(const Tensor& seq);        // full-sequence (unrolled)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return W_f_; }
    Tensor get_gradients() const override { return grad_W_f_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "MinLSTM"; }

    void init_weights();

    // Parameter accessors (gradient checks / direct inspection).
    Tensor& W_f() { return W_f_; }
    Tensor& b_f() { return b_f_; }
    Tensor& W_i() { return W_i_; }
    Tensor& b_i() { return b_i_; }
    Tensor& W_h() { return W_h_; }
    Tensor& b_h() { return b_h_; }

    const Tensor& grad_W_f() const { return grad_W_f_; }
    const Tensor& grad_b_f() const { return grad_b_f_; }
    const Tensor& grad_W_i() const { return grad_W_i_; }
    const Tensor& grad_b_i() const { return grad_b_i_; }
    const Tensor& grad_W_h() const { return grad_W_h_; }
    const Tensor& grad_b_h() const { return grad_b_h_; }

    // Cached normalized gates from the last forward_sequence, shape (T, hidden).
    // f'_t + i'_t == 1 by construction — the MinLSTM-specific invariant.
    const Tensor& last_f_norm() const { return cached_f_norm_; }
    const Tensor& last_i_norm() const { return cached_i_norm_; }
    const Tensor& last_cand() const { return cached_cand_; }

    size_t input_dim() const { return input_dim_; }
    size_t hidden_size() const { return hidden_size_; }

    // Reset only the recurrent hidden state (for clean FD re-evaluations).
    void reset_state() { h_.fill(0.0); }

private:
    size_t input_dim_, hidden_size_, seq_len_;

    Tensor W_f_, b_f_;     // forget-gate projection
    Tensor W_i_, b_i_;     // input-gate projection
    Tensor W_h_, b_h_;     // candidate projection

    Tensor grad_W_f_, grad_b_f_;
    Tensor grad_W_i_, grad_b_i_;
    Tensor grad_W_h_, grad_b_h_;

    Tensor h_;             // current hidden state (1, hidden_size)
    Tensor last_output_;

    // BPTT caches
    Tensor inputs_;            // (T, input_dim)
    Tensor hidden_states_;     // (T+1, hidden) — row 0 is h_{-1}
    Tensor cached_f_;          // (T, hidden) — raw sigmoid forget gate
    Tensor cached_i_;          // (T, hidden) — raw sigmoid input gate
    Tensor cached_f_norm_;     // (T, hidden) — f / (f + i)
    Tensor cached_i_norm_;     // (T, hidden) — i / (f + i)
    Tensor cached_cand_;       // (T, hidden) — candidate c_t
};

#endif
