#ifndef LSTM_BIDIRECTIONAL_H
#define LSTM_BIDIRECTIONAL_H

#include "lstm.h"

// Bidirectional LSTM — runs forward and backward LSTMs on the input sequence,
// then concatenates their outputs along the feature dimension.
// Input: (batch, seq_len * input_dim) — same format as LSTM
// Output: (batch, 2 * hidden_size) — last forward hidden state + last backward hidden state
class BiLSTM : public Layer {
public:
    BiLSTM(size_t input_dim, size_t hidden_size, size_t seq_len);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

private:
    Tensor last_output_;
    LSTM forward_lstm_;
    LSTM backward_lstm_;
    size_t input_dim_;
    size_t hidden_size_;
    size_t seq_len_;
};

#endif