#ifndef LSTM_H
#define LSTM_H

#include "../../core/layer.h"
#include <vector>

class LSTM : public Layer {
public:
    int input_dim;
    int hidden_size;
    int seq_len;

    // Parameters: all gates: i, f, o, g (candidate)
    // Combined weight matrix: (4*hidden, hidden + input) for efficiency
    Tensor W; // (4*hidden, hidden + input)
    Tensor b; // (4*hidden, 1)

    // Gradients
    Tensor grad_W;
    Tensor grad_b;

    // BPTT cache
    Tensor inputs;          // (N, seq_len * input_dim) raw input
    Tensor h_states;        // ( (seq_len+1)*N, hidden ) h_0..h_L
    Tensor c_states;        // ( (seq_len+1)*N, hidden ) c_0..c_L, c_0=0
    Tensor last_output_;    // last forward output (batch, hidden)

    LSTM(int input_dim, int hidden_size, int seq_len);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W; }
    Tensor get_gradients() const override { return grad_W; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

#endif