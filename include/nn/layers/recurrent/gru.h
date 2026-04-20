#ifndef GRU_H
#define GRU_H

#include "../../core/layer.h"

// Gated Recurrent Unit — update gate, reset gate, candidate hidden state.
// Forward modes: single-step (stateful) and full-sequence (unrolled).
class GRU : public Layer {
public:
    GRU(size_t input_dim, size_t hidden_size);
    Tensor forward(const Tensor& input) override;
    Tensor forward_sequence(const Tensor& seq);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void init_weights();

private:
    size_t input_dim_, hidden_size_, seq_len_;

    // Weights: stacked gates for efficiency
    // combined W_zr: (input_dim x 2h), combined U_zr: (h x 2h)
    // W_h: (input_dim x h), U_h: (h x h)
    Tensor W_zr_;  // (input_dim x 2h)
    Tensor U_zr_;  // (h x 2h)
    Tensor b_zr_;  // (1 x 2h)
    Tensor W_h_;   // (input_dim x h)
    Tensor U_h_;   // (h x h)
    Tensor b_h_;   // (1 x h)

    Tensor grad_W_zr_, grad_U_zr_, grad_b_zr_;
    Tensor grad_W_h_, grad_U_h_, grad_b_h_;

    Tensor h_;        // current hidden state (1 x h)
    Tensor last_output_;
    Tensor z_;        // update gate (1 x h)
    Tensor r_;        // reset gate (1 x h)
    Tensor hc_;       // candidate (1 x h)
    Tensor h_prev_;   // previous hidden (for backward)

    // Caches for BPTT
    Tensor inputs_;       // (seq_len, input_dim) — cached input sequence
    Tensor hidden_states_; // ((seq_len+1), h) — cached hidden states

    void compute_gates(const Tensor& x, const Tensor& h_prev);
};

#endif
