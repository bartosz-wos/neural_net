#ifndef MIN_GRU_H
#define MIN_GRU_H

#include "../../core/layer.h"

// MinGRU — Feng et al. 2024 ("Were RNNs All We Needed?")
//
// A minimalist GRU that removes the hidden-to-hidden (recurrent) weights
// from both the gate and candidate pre-activations. The recurrence becomes
// a per-channel weighted EMA of past candidates, which admits a parallel
// scan (associative scan) over the time dimension.
//
// Forward (per timestep):
//   g_log_t = W_g · x_t + b_g                  (no h_prev dependence)
//   gates_t = sigmoid(g_log_t)                  ∈ (0, 1) per channel
//   cand_t  = W_h · x_t + b_h                   (no h_prev dependence)
//   h_t = (1 - gates_t) ⊙ h_{t-1} + gates_t ⊙ cand_t
//
// Equivalently: h_t = α_t ⊙ h_{t-1} + (1 - α_t) ⊙ cand_t with α_t = 1 - gates_t.
//
// The recurrence is associative:
//   ((α_i, β_i), (α_j, β_j)) -> (α_i·α_j, α_i·β_j + β_i)
// so a parallel scan computes the same h_t in O(log T) sequential steps.
//
// Parameters (none of which are recurrent — all input-only projections):
//   W_g  : (input_dim, hidden_size)
//   b_g  : (1, hidden_size)
//   W_h  : (input_dim, hidden_size)
//   b_h  : (1, hidden_size)
//
// Forward modes: single-step (stateful) and full-sequence (unrolled).
class MinGRU : public Layer {
public:
    MinGRU(size_t input_dim, size_t hidden_size);
    Tensor forward(const Tensor& input) override;
    Tensor forward_sequence(const Tensor& seq);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_g_; }
    Tensor get_gradients() const override { return grad_W_g_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void init_weights();

    // Parameter accessors (for gradient checks and direct inspection).
    Tensor& W_g() { return W_g_; }
    Tensor& b_g() { return b_g_; }
    Tensor& W_h() { return W_h_; }
    Tensor& b_h() { return b_h_; }
    const Tensor& grad_W_g() const { return grad_W_g_; }
    const Tensor& grad_b_g() const { return grad_b_g_; }
    const Tensor& grad_W_h() const { return grad_W_h_; }
    const Tensor& grad_b_h() const { return grad_b_h_; }

    // Reset only the recurrent hidden state (for clean FD re-evaluations).
    void reset_state() { h_.fill(0.0); }

private:
    size_t input_dim_, hidden_size_, seq_len_;

    Tensor W_g_, b_g_;     // gate projection: input -> gate logit
    Tensor W_h_, b_h_;     // candidate projection: input -> candidate

    Tensor grad_W_g_, grad_b_g_;
    Tensor grad_W_h_, grad_b_h_;

    Tensor h_;             // current hidden state (1, hidden_size)
    Tensor last_output_;
    Tensor gates_;         // cached gates (1, hidden_size) — sigmoid outputs
    Tensor cand_;          // cached candidate (1, hidden_size)
    Tensor h_prev_;        // cached h_{t-1} for backward

    // Caches for BPTT
    Tensor inputs_;            // (seq_len, input_dim)
    Tensor hidden_states_;     // ((seq_len+1), hidden_size)
    Tensor cached_gates_;      // (seq_len, hidden_size) — sigmoid outputs at each t
    Tensor cached_cand_;       // (seq_len, hidden_size) — candidate at each t
};

#endif
