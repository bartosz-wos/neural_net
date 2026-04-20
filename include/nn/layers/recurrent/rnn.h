#ifndef RNN_H
#define RNN_H

#include "../../core/layer.h"
#include <vector>

class SimpleRNN : public Layer {
public:
    int input_dim;
    int hidden_size;
    int seq_len; // expected sequence length (L)

    // Parameters
    Tensor W_xh; // (hidden, input)
    Tensor W_hh; // (hidden, hidden)
    Tensor b;    // (hidden)

    // Gradients
    Tensor grad_W_xh;
    Tensor grad_W_hh;
    Tensor grad_b;

    // Cache for BPTT
    Tensor inputs;       // (N, L, input) reshaped from flat input
    Tensor hidden_states; // (N, L+1, hidden), h0=0, h1..hL computed

    SimpleRNN(int input_dim, int hidden_size, int seq_len);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

#endif
